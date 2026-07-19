//! Graph-form SPARQL result production.
//!
//! `SELECT` / `ASK` queries yield solution *bindings*; `CONSTRUCT` and
//! `DESCRIBE` yield an RDF *graph*. This module turns the executor's binding
//! solutions into concrete triples:
//!
//! * [`instantiate_construct`] substitutes a solution into a `CONSTRUCT`
//!   template, minting fresh blank nodes per row and skipping ill-formed rows.
//! * [`describe`] resolves the described node set and builds a Concise Bounded
//!   Description (subject-rooted, blank-node closure, cycle-safe).
//!
//! Both operate on the crate's own algebra [`Term`]/[`Triple`] types so callers
//! never leave the arq type universe.

use crate::algebra::{PropertyPath, Solution, Term, Triple, TriplePattern, Variable};
use crate::executor::dataset::Dataset;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Instantiate a `CONSTRUCT` template against a solution sequence, producing a
/// deduplicated set of RDF triples.
///
/// Semantics (SPARQL 1.1 §16.2.1):
///
/// * For each solution row, every template pattern is instantiated by
///   substituting bound variables.
/// * A template pattern that leaves **any** variable unbound in a given row is
///   skipped for that row (not an error).
/// * A pattern that would instantiate to an ill-formed RDF triple (literal or
///   blank node in predicate position, literal in subject position, an
///   unresolved variable, ...) is likewise skipped.
/// * Blank nodes in the template are **scoped to the row**: each solution row
///   mints its own fresh blank nodes, so structurally-identical rows do not
///   collapse into one another.
/// * Ground triples that recur across rows are deduplicated.
pub fn instantiate_construct(
    template: &[TriplePattern],
    solution: &Solution,
) -> Result<Vec<Triple>> {
    let mut out: Vec<Triple> = Vec::new();
    let mut seen: HashSet<Triple> = HashSet::new();
    // Monotonic counter so every minted blank node label is unique across the
    // whole call; per-row maps guarantee freshness between rows.
    let mut bnode_counter: u64 = 0;

    for binding in solution {
        let mut row_bnodes: HashMap<String, String> = HashMap::new();
        for pattern in template {
            let subject = match substitute(
                &pattern.subject,
                binding,
                &mut row_bnodes,
                &mut bnode_counter,
            ) {
                Some(term) => term,
                None => continue,
            };
            let predicate = match substitute(
                &pattern.predicate,
                binding,
                &mut row_bnodes,
                &mut bnode_counter,
            ) {
                Some(term) => term,
                None => continue,
            };
            let object = match substitute(
                &pattern.object,
                binding,
                &mut row_bnodes,
                &mut bnode_counter,
            ) {
                Some(term) => term,
                None => continue,
            };

            if !is_valid_subject(&subject)
                || !is_valid_predicate(&predicate)
                || !is_valid_object(&object)
            {
                continue;
            }

            let triple = TriplePattern {
                subject,
                predicate,
                object,
            };
            if seen.insert(triple.clone()) {
                out.push(triple);
            }
        }
    }

    Ok(out)
}

/// Substitute a single template term against `binding`.
///
/// Returns `None` when the term (or a nested component) is an unbound variable
/// or a complex (non-length-one) property-path term, signalling that the
/// enclosing triple must be skipped for this row. Blank nodes are mapped to a
/// freshly-minted label, stable within the row via `row_bnodes`.
///
/// The arq parser encodes every BGP/CONSTRUCT-template predicate as a property
/// path, so a plain-IRI / plain-variable predicate arrives as a length-one
/// `PropertyPath::Iri` / `PropertyPath::Variable`. Those are normalized here to
/// the equivalent plain `Term::Iri` / `Term::Variable` before substitution so a
/// path-encoded template predicate instantiates natively (no caller-side
/// pre-normalization required). Genuinely complex paths (sequence, inverse,
/// `*`, `+`, `?`, alternative, negated set) cannot appear in a well-formed
/// CONSTRUCT template and are dropped as ill-formed, matching the existing
/// property-path policy.
fn substitute(
    term: &Term,
    binding: &HashMap<Variable, Term>,
    row_bnodes: &mut HashMap<String, String>,
    counter: &mut u64,
) -> Option<Term> {
    match term {
        Term::Variable(var) => binding.get(var).cloned(),
        Term::Iri(_) | Term::Literal(_) => Some(term.clone()),
        Term::BlankNode(label) => {
            let minted = row_bnodes.entry(label.clone()).or_insert_with(|| {
                let fresh = format!("bnc{counter}");
                *counter += 1;
                fresh
            });
            Some(Term::BlankNode(minted.clone()))
        }
        Term::QuotedTriple(inner) => {
            let s = substitute(&inner.subject, binding, row_bnodes, counter)?;
            let p = substitute(&inner.predicate, binding, row_bnodes, counter)?;
            let o = substitute(&inner.object, binding, row_bnodes, counter)?;
            Some(Term::QuotedTriple(Box::new(TriplePattern {
                subject: s,
                predicate: p,
                object: o,
            })))
        }
        // A length-one property path is the parser's encoding of a plain
        // predicate term; normalize it to the equivalent plain term.
        Term::PropertyPath(PropertyPath::Iri(iri)) => Some(Term::Iri(iri.clone())),
        Term::PropertyPath(PropertyPath::Variable(var)) => binding.get(var).cloned(),
        // Any genuinely complex path cannot appear in a CONSTRUCT template; drop
        // the triple (ill-formed policy).
        Term::PropertyPath(_) => None,
    }
}

/// Valid RDF subject: IRI, blank node, or quoted triple (RDF-star).
fn is_valid_subject(term: &Term) -> bool {
    matches!(
        term,
        Term::Iri(_) | Term::BlankNode(_) | Term::QuotedTriple(_)
    )
}

/// Valid RDF predicate: IRI only.
fn is_valid_predicate(term: &Term) -> bool {
    matches!(term, Term::Iri(_))
}

/// Valid RDF object: IRI, literal, blank node, or quoted triple (RDF-star).
fn is_valid_object(term: &Term) -> bool {
    matches!(
        term,
        Term::Iri(_) | Term::Literal(_) | Term::BlankNode(_) | Term::QuotedTriple(_)
    )
}

/// Produce a `DESCRIBE` graph for a set of target nodes.
///
/// The described node set is the union of:
/// * the explicit IRI/blank-node `targets` (e.g. `DESCRIBE <iri>`), and
/// * every value bound to one of `target_vars` across the `solution` rows
///   (e.g. `DESCRIBE ?x WHERE { ... }`).
///
/// For each described node this builds a **symmetric Concise Bounded
/// Description (CBD)**: the triples with the node as subject *and* the triples
/// with the node as object (its incoming arcs), then recursively the CBD of any
/// blank node reached — as an object of an outgoing arc or as the subject of an
/// incoming arc — guarded by a visited-set so cycles terminate. The symmetric
/// form guarantees a resource that only ever appears as an object still
/// describes to a non-empty, blank-node-closed graph rather than an empty one.
/// Literals and non-node targets are ignored (they cannot be described).
///
/// The triples are read through [`Dataset::find_triples`], i.e. from whatever
/// dataset the caller passes in — its active default graph. Fuseki passes a
/// `FROM` / `FROM NAMED`-scoped dataset view end to end, so the CBD is drawn
/// from exactly the graphs the request selected. Automatically unioning every
/// named graph is a deliberate non-goal: it is the same semantic bug the
/// [`GraphSelector`](crate::executor::dataset::GraphSelector) abstraction exists
/// to prevent, so scoping stays with the caller-supplied dataset.
pub fn describe(
    targets: &[Term],
    target_vars: &[Variable],
    solution: &Solution,
    dataset: &dyn Dataset,
) -> Result<Vec<Triple>> {
    // Resolve the described node set, de-duplicated but order-preserving.
    let mut nodes: Vec<Term> = Vec::new();
    let mut node_seen: HashSet<Term> = HashSet::new();

    for target in targets {
        if is_describable(target) && node_seen.insert(target.clone()) {
            nodes.push(target.clone());
        }
    }
    for binding in solution {
        for var in target_vars {
            if let Some(value) = binding.get(var) {
                if is_describable(value) && node_seen.insert(value.clone()) {
                    nodes.push(value.clone());
                }
            }
        }
    }

    let mut out: Vec<Triple> = Vec::new();
    let mut emitted: HashSet<Triple> = HashSet::new();
    let mut visited: HashSet<Term> = HashSet::new();

    for node in nodes {
        concise_bounded_description(&node, dataset, &mut out, &mut emitted, &mut visited)?;
    }

    Ok(out)
}

/// Only IRIs and blank nodes can be described.
fn is_describable(term: &Term) -> bool {
    matches!(term, Term::Iri(_) | Term::BlankNode(_))
}

/// Recursively collect the symmetric CBD rooted at `node`.
///
/// Both the outgoing arcs (`node ?p ?o`) and the incoming arcs (`?s ?p node`)
/// are emitted, and the blank nodes on the boundary — blank objects of outgoing
/// arcs and blank subjects of incoming arcs — are recursed into. The shared
/// `visited` set makes the whole traversal cycle-safe, and `emitted` keeps the
/// output de-duplicated across both directions.
fn concise_bounded_description(
    node: &Term,
    dataset: &dyn Dataset,
    out: &mut Vec<Triple>,
    emitted: &mut HashSet<Triple>,
    visited: &mut HashSet<Term>,
) -> Result<()> {
    if !visited.insert(node.clone()) {
        return Ok(());
    }

    // Blank nodes on the boundary of this node's description, recursed into after
    // the borrows of `out`/`emitted` are released. Deduped via `queued` so the
    // same neighbour is not scheduled twice within this call.
    let mut next: Vec<Term> = Vec::new();
    let mut queued: HashSet<Term> = HashSet::new();

    // Outgoing arcs: `node ?p ?o`. Recurse into blank-node OBJECTS.
    let forward = TriplePattern {
        subject: node.clone(),
        predicate: Term::Variable(Variable::new_unchecked("p")),
        object: Term::Variable(Variable::new_unchecked("o")),
    };
    for (s, p, o) in dataset.find_triples(&forward)? {
        if matches!(o, Term::BlankNode(_)) && !visited.contains(&o) && queued.insert(o.clone()) {
            next.push(o.clone());
        }
        let triple = TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        };
        if emitted.insert(triple.clone()) {
            out.push(triple);
        }
    }

    // Incoming arcs (symmetric CBD): `?s ?p node`. Recurse into blank-node
    // SUBJECTS so a resource reachable only as an object still describes to a
    // non-empty, closed graph.
    let reverse = TriplePattern {
        subject: Term::Variable(Variable::new_unchecked("s")),
        predicate: Term::Variable(Variable::new_unchecked("p")),
        object: node.clone(),
    };
    for (s, p, o) in dataset.find_triples(&reverse)? {
        if matches!(s, Term::BlankNode(_)) && !visited.contains(&s) && queued.insert(s.clone()) {
            next.push(s.clone());
        }
        let triple = TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        };
        if emitted.insert(triple.clone()) {
            out.push(triple);
        }
    }

    for neighbour in next {
        concise_bounded_description(&neighbour, dataset, out, emitted, visited)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Literal, Term};
    use crate::executor::dataset::InMemoryDataset;
    use oxirs_core::model::NamedNode;

    fn iri(s: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(s))
    }

    fn var(name: &str) -> Variable {
        Variable::new_unchecked(name)
    }

    fn vt(name: &str) -> Term {
        Term::Variable(var(name))
    }

    fn lit(value: &str) -> Term {
        Term::Literal(Literal {
            value: value.to_string(),
            language: None,
            datatype: None,
        })
    }

    fn row(pairs: &[(&str, Term)]) -> HashMap<Variable, Term> {
        pairs
            .iter()
            .map(|(name, term)| (var(name), term.clone()))
            .collect()
    }

    #[test]
    fn construct_full_substitution() {
        // Template: ?s <p> ?o
        let template = vec![TriplePattern {
            subject: vt("s"),
            predicate: iri("http://ex/p"),
            object: vt("o"),
        }];
        let solution = vec![
            row(&[("s", iri("http://ex/a")), ("o", lit("1"))]),
            row(&[("s", iri("http://ex/b")), ("o", lit("2"))]),
        ];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert_eq!(triples.len(), 2);
        assert!(triples
            .iter()
            .any(|t| t.subject == iri("http://ex/a") && t.object == lit("1")));
        assert!(triples
            .iter()
            .any(|t| t.subject == iri("http://ex/b") && t.object == lit("2")));
    }

    #[test]
    fn construct_skips_unbound_var_rows() {
        let template = vec![TriplePattern {
            subject: vt("s"),
            predicate: iri("http://ex/p"),
            object: vt("o"),
        }];
        // Second row leaves ?o unbound -> its triple is skipped.
        let solution = vec![
            row(&[("s", iri("http://ex/a")), ("o", lit("1"))]),
            row(&[("s", iri("http://ex/b"))]),
        ];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert_eq!(triples.len(), 1, "row with unbound ?o must be skipped");
        assert_eq!(triples[0].subject, iri("http://ex/a"));
    }

    #[test]
    fn construct_skips_literal_subject() {
        // ?s in subject position bound to a literal -> ill-formed, skipped.
        let template = vec![TriplePattern {
            subject: vt("s"),
            predicate: iri("http://ex/p"),
            object: iri("http://ex/o"),
        }];
        let solution = vec![row(&[("s", lit("not-a-subject"))])];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert!(triples.is_empty(), "literal subject must be dropped");
    }

    #[test]
    fn construct_mints_distinct_blank_nodes_per_row() {
        // Template: _:b <p> ?o  -> each row must get a distinct blank node.
        let template = vec![TriplePattern {
            subject: Term::BlankNode("b".to_string()),
            predicate: iri("http://ex/p"),
            object: vt("o"),
        }];
        let solution = vec![row(&[("o", lit("1"))]), row(&[("o", lit("2"))])];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert_eq!(triples.len(), 2);
        let b0 = &triples[0].subject;
        let b1 = &triples[1].subject;
        assert!(matches!(b0, Term::BlankNode(_)));
        assert!(matches!(b1, Term::BlankNode(_)));
        assert_ne!(b0, b1, "blank nodes must be fresh per row");
    }

    #[test]
    fn construct_deduplicates_ground_triples() {
        // Constant template over two rows -> one output triple after dedup.
        let template = vec![TriplePattern {
            subject: iri("http://ex/a"),
            predicate: iri("http://ex/p"),
            object: iri("http://ex/o"),
        }];
        let solution = vec![row(&[("x", lit("1"))]), row(&[("x", lit("2"))])];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert_eq!(triples.len(), 1, "identical ground triples must dedup");
    }

    #[test]
    fn construct_path_encoded_iri_predicate_instantiates() {
        // The arq parser encodes a plain predicate as a length-one property path:
        // `?s PropertyPath::Iri(<p>) ?o`. instantiate_construct must accept it
        // natively (no caller-side normalization) and emit the plain-IRI triple.
        let template = vec![TriplePattern {
            subject: vt("s"),
            predicate: Term::PropertyPath(PropertyPath::Iri(NamedNode::new_unchecked(
                "http://ex/worksIn",
            ))),
            object: vt("o"),
        }];
        let solution = vec![row(&[
            ("s", iri("http://ex/a")),
            ("o", iri("http://ex/eng")),
        ])];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert_eq!(triples.len(), 1, "path-encoded predicate must instantiate");
        assert_eq!(triples[0].predicate, iri("http://ex/worksIn"));
        assert_eq!(triples[0].subject, iri("http://ex/a"));
        assert_eq!(triples[0].object, iri("http://ex/eng"));
    }

    #[test]
    fn construct_path_encoded_variable_predicate_instantiates() {
        // `?s PropertyPath::Variable(?p) ?o` must resolve ?p from the binding.
        let template = vec![TriplePattern {
            subject: vt("s"),
            predicate: Term::PropertyPath(PropertyPath::Variable(var("p"))),
            object: vt("o"),
        }];
        let solution = vec![row(&[
            ("s", iri("http://ex/a")),
            ("p", iri("http://ex/worksIn")),
            ("o", iri("http://ex/eng")),
        ])];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert_eq!(
            triples.len(),
            1,
            "path-var predicate must resolve and instantiate"
        );
        assert_eq!(triples[0].predicate, iri("http://ex/worksIn"));
    }

    #[test]
    fn construct_complex_path_predicate_is_dropped() {
        // A genuinely complex path (`p+`) cannot appear in a CONSTRUCT template;
        // it is ill-formed and the triple is dropped (existing policy retained).
        let inner = PropertyPath::Iri(NamedNode::new_unchecked("http://ex/p"));
        let template = vec![TriplePattern {
            subject: vt("s"),
            predicate: Term::PropertyPath(PropertyPath::OneOrMore(Box::new(inner))),
            object: vt("o"),
        }];
        let solution = vec![row(&[("s", iri("http://ex/a")), ("o", iri("http://ex/b"))])];
        let triples = instantiate_construct(&template, &solution).expect("construct");
        assert!(triples.is_empty(), "complex-path predicate must be dropped");
    }

    #[test]
    fn describe_subject_cbd() {
        let mut ds = InMemoryDataset::new();
        ds.add_triple(iri("http://ex/a"), iri("http://ex/p1"), lit("v1"));
        ds.add_triple(iri("http://ex/a"), iri("http://ex/p2"), iri("http://ex/b"));
        // Unrelated triple that must NOT appear.
        ds.add_triple(iri("http://ex/z"), iri("http://ex/p"), lit("other"));

        let triples = describe(&[iri("http://ex/a")], &[], &vec![], &ds).expect("describe");
        assert_eq!(triples.len(), 2, "CBD must include both subject-a triples");
        assert!(triples.iter().all(|t| t.subject == iri("http://ex/a")));
    }

    #[test]
    fn describe_from_bound_vars() {
        let mut ds = InMemoryDataset::new();
        ds.add_triple(iri("http://ex/a"), iri("http://ex/p"), lit("v"));
        let solution = vec![row(&[("x", iri("http://ex/a"))])];
        let triples = describe(&[], &[var("x")], &solution, &ds).expect("describe");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, iri("http://ex/a"));
    }

    #[test]
    fn describe_blank_node_closure_terminates_on_cycle() {
        // _:x -> _:y -> _:x  (a cycle through blank nodes).
        let mut ds = InMemoryDataset::new();
        let bx = Term::BlankNode("x".to_string());
        let by = Term::BlankNode("y".to_string());
        ds.add_triple(iri("http://ex/root"), iri("http://ex/has"), bx.clone());
        ds.add_triple(bx.clone(), iri("http://ex/next"), by.clone());
        ds.add_triple(by.clone(), iri("http://ex/next"), bx.clone());

        // Must terminate and include all three triples without infinite loop.
        let triples = describe(&[iri("http://ex/root")], &[], &vec![], &ds).expect("describe");
        assert_eq!(
            triples.len(),
            3,
            "CBD closure must follow blank nodes exactly once"
        );
    }

    #[test]
    fn describe_object_only_resource_includes_incoming_arc() {
        // `target` never appears as a subject, only as an object. A forward-only
        // CBD would describe it to an empty graph; symmetric CBD must surface its
        // incoming arc instead.
        let mut ds = InMemoryDataset::new();
        ds.add_triple(
            iri("http://ex/a"),
            iri("http://ex/p"),
            iri("http://ex/target"),
        );
        // Unrelated triple that must NOT appear.
        ds.add_triple(iri("http://ex/z"), iri("http://ex/q"), lit("other"));

        let triples = describe(&[iri("http://ex/target")], &[], &vec![], &ds).expect("describe");
        assert_eq!(
            triples.len(),
            1,
            "object-only resource must describe via its single incoming arc"
        );
        let t = &triples[0];
        assert_eq!(t.subject, iri("http://ex/a"));
        assert_eq!(t.predicate, iri("http://ex/p"));
        assert_eq!(t.object, iri("http://ex/target"));
    }

    #[test]
    fn describe_incoming_blank_subject_closed_over_cycle_safe() {
        // `target` is only ever an object. Its incoming arc comes from a blank
        // node _:x that participates in a blank-blank cycle (_:x <-> _:y).
        // Symmetric CBD must close over the incoming blank subject and terminate.
        let mut ds = InMemoryDataset::new();
        let bx = Term::BlankNode("x".to_string());
        let by = Term::BlankNode("y".to_string());
        ds.add_triple(bx.clone(), iri("http://ex/knows"), iri("http://ex/target"));
        ds.add_triple(bx.clone(), iri("http://ex/next"), by.clone());
        ds.add_triple(by.clone(), iri("http://ex/next"), bx.clone());

        let triples = describe(&[iri("http://ex/target")], &[], &vec![], &ds).expect("describe");
        assert_eq!(
            triples.len(),
            3,
            "symmetric CBD must close over the incoming blank subject exactly once"
        );
        assert!(
            triples.iter().any(|t| t.object == iri("http://ex/target")
                && t.subject == bx
                && t.predicate == iri("http://ex/knows")),
            "the incoming arc into target must be present"
        );
    }
}
