//! Blank-node property lists `[ … ]` and RDF collections `( … )`
//! (SPARQL 1.1 §4.2.2 / §4.2.3 Turtle-style sugar).
//!
//! Both were a hard 400 (known limitation). They now expand at parse time into
//! ordinary triple patterns anchored on a fresh anonymous node. These tests are
//! mostly STRUCTURAL — they assert the exact expanded triple vector for the
//! subject/object/empty/nested cases — plus one CONSTRUCT execution proving
//! template blank nodes are minted fresh per solution row.
//!
//! Context matters: in a WHERE pattern the anchor is a non-distinguished
//! `Term::Variable` (the store matches a query blank node as an existential),
//! while in a CONSTRUCT template it is a real `Term::BlankNode`.

use std::collections::HashMap;

use oxirs_arq::algebra::{Algebra, Binding, PropertyPath, Term, TriplePattern};
use oxirs_arq::instantiate_construct;
use oxirs_arq::query::parse_query;
use oxirs_arq::{Literal, Variable};
use oxirs_core::model::NamedNode;

const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
const EX: &str = "http://ex/";

fn ex(local: &str) -> String {
    format!("{EX}{local}")
}

/// The WHERE-clause BGP triples of a (single-BGP) query.
fn bgp(query: &str) -> Vec<TriplePattern> {
    let full = format!("PREFIX : <{EX}> {query}");
    let q = parse_query(&full).unwrap_or_else(|e| panic!("must parse `{full}`: {e}"));
    match q.where_clause {
        Algebra::Bgp(triples) => triples,
        other => panic!("expected a BGP WHERE, got {other:?}"),
    }
}

/// IRI string of an IRI term or a length-one `PropertyPath::Iri` predicate.
fn iri_of(t: &Term) -> Option<String> {
    match t {
        Term::Iri(n) => Some(n.as_str().to_string()),
        Term::PropertyPath(PropertyPath::Iri(n)) => Some(n.as_str().to_string()),
        _ => None,
    }
}

fn pred_is(t: &TriplePattern, iri: &str) -> bool {
    iri_of(&t.predicate).as_deref() == Some(iri)
}

/// The single triple whose predicate is `iri` (panics if not exactly one).
fn only_with_pred<'a>(triples: &'a [TriplePattern], iri: &str) -> &'a TriplePattern {
    let mut it = triples.iter().filter(|t| pred_is(t, iri));
    let found = it
        .next()
        .unwrap_or_else(|| panic!("no triple with predicate {iri}"));
    assert!(
        it.next().is_none(),
        "more than one triple with predicate {iri}"
    );
    found
}

// ── blank-node property lists `[ … ]` ────────────────────────────────────────

#[test]
fn object_blank_node_property_list_expands() {
    // `?s :q [ :p :o ]` → (?s :q _:b) + (_:b :p :o), same anchor.
    let t = bgp("SELECT ?s WHERE { ?s :q [ :p :o ] }");
    assert_eq!(t.len(), 2, "one outer + one anchored triple: {t:?}");

    let outer = only_with_pred(&t, &ex("q"));
    let inner = only_with_pred(&t, &ex("p"));
    let anchor = &outer.object;

    assert!(
        matches!(anchor, Term::Variable(_)),
        "a WHERE anonymous node must be a non-distinguished variable, got {anchor:?}"
    );
    assert_eq!(
        &inner.subject, anchor,
        "the `[ ]` triple hangs off the anchor"
    );
    assert_eq!(iri_of(&inner.object).as_deref(), Some(ex("o").as_str()));
    assert!(matches!(&outer.subject, Term::Variable(v) if v.name() == "s"));
}

#[test]
fn blank_node_multiple_predicates_share_one_anchor() {
    // `[ :p :o ; :q :r ]` → both triples share the SAME fresh subject.
    let t = bgp("SELECT ?x WHERE { ?x :link [ :p :o ; :q :r ] }");
    assert_eq!(t.len(), 3, "outer + two anchored predicates: {t:?}");
    let p = only_with_pred(&t, &ex("p"));
    let q = only_with_pred(&t, &ex("q"));
    assert_eq!(
        p.subject, q.subject,
        "`;` predicates must share one blank node"
    );
    let outer = only_with_pred(&t, &ex("link"));
    assert_eq!(outer.object, p.subject, "the anchor is the outer object");
}

#[test]
fn empty_anon_object_is_a_bare_fresh_node() {
    // `?s :q []` → exactly one triple with a fresh variable object.
    let t = bgp("SELECT ?s WHERE { ?s :q [] }");
    assert_eq!(t.len(), 1, "the empty `[]` anchors no triples: {t:?}");
    assert!(matches!(t[0].object, Term::Variable(_)), "object is fresh");
}

#[test]
fn subject_blank_node_property_list_expands() {
    // `[ :p :o ] :q ?x` → (_:b :p :o) + (_:b :q ?x), same subject.
    let t = bgp("SELECT ?x WHERE { [ :p :o ] :q ?x }");
    assert_eq!(t.len(), 2, "{t:?}");
    let inner = only_with_pred(&t, &ex("p"));
    let outer = only_with_pred(&t, &ex("q"));
    assert_eq!(
        inner.subject, outer.subject,
        "both hang off the same anchor"
    );
    assert!(matches!(&outer.object, Term::Variable(v) if v.name() == "x"));
}

#[test]
fn standalone_blank_node_property_list() {
    // `[ :p :o ]` as a whole statement → just its one anchored triple.
    let t = bgp("SELECT ?s WHERE { [ :p :o ] }");
    assert_eq!(t.len(), 1, "{t:?}");
    assert!(pred_is(&t[0], &ex("p")));
    assert!(matches!(t[0].subject, Term::Variable(_)));
}

// ── RDF collections `( … )` ──────────────────────────────────────────────────

/// Walk an `rdf:first`/`rdf:rest` chain from `head`, returning the item objects
/// in order and asserting it terminates at `rdf:nil`.
fn collect_list(triples: &[TriplePattern], head: &Term) -> Vec<Term> {
    let first: HashMap<&Term, &Term> = triples
        .iter()
        .filter(|t| pred_is(t, RDF_FIRST))
        .map(|t| (&t.subject, &t.object))
        .collect();
    let rest: HashMap<&Term, &Term> = triples
        .iter()
        .filter(|t| pred_is(t, RDF_REST))
        .map(|t| (&t.subject, &t.object))
        .collect();

    let mut items = Vec::new();
    let mut node = head.clone();
    loop {
        if iri_of(&node).as_deref() == Some(RDF_NIL) {
            break;
        }
        let f = first
            .get(&node)
            .unwrap_or_else(|| panic!("list node {node:?} has no rdf:first"));
        items.push((*f).clone());
        let r = rest
            .get(&node)
            .unwrap_or_else(|| panic!("list node {node:?} has no rdf:rest"));
        node = (*r).clone();
    }
    items
}

#[test]
fn object_collection_expands_to_first_rest_chain() {
    // `?s :q ( :a :b :c )` → 3-element chain (6 triples) + the outer triple.
    let t = bgp("SELECT ?s WHERE { ?s :q ( :a :b :c ) }");
    assert_eq!(t.len(), 7, "3 items × (first+rest) + outer: {t:?}");

    let outer = only_with_pred(&t, &ex("q"));
    let head = outer.object.clone();
    let items = collect_list(&t, &head);
    let item_iris: Vec<String> = items.iter().filter_map(iri_of).collect();
    assert_eq!(
        item_iris,
        vec![ex("a"), ex("b"), ex("c")],
        "the collection must expand in written order"
    );
}

#[test]
fn empty_collection_is_rdf_nil() {
    // `?s :q ()` → the object is rdf:nil directly; no blank nodes.
    let t = bgp("SELECT ?s WHERE { ?s :q () }");
    assert_eq!(t.len(), 1, "{t:?}");
    assert_eq!(
        iri_of(&t[0].object).as_deref(),
        Some(RDF_NIL),
        "the empty collection is rdf:nil"
    );
}

#[test]
fn nested_collection_inside_blank_node_property_list() {
    // `[ :p ( :a :b ) ]` — a collection nested inside a `[ ]`.
    let t = bgp("SELECT ?s WHERE { ?s :q [ :p ( :a :b ) ] }");
    // outer (?s :q _:anchor), (_:anchor :p _:listhead), then the 2-item chain
    // (4 triples): 6 total.
    assert_eq!(t.len(), 6, "{t:?}");
    let p = only_with_pred(&t, &ex("p"));
    let items: Vec<String> = collect_list(&t, &p.object)
        .iter()
        .filter_map(iri_of)
        .collect();
    assert_eq!(items, vec![ex("a"), ex("b")], "nested collection expands");
}

// ── CONSTRUCT template: real blank nodes, minted fresh per row ────────────────

fn var(name: &str) -> Variable {
    Variable::new(name).expect("valid variable")
}

fn lit(value: &str) -> Term {
    Term::Literal(Literal {
        value: value.to_string(),
        language: None,
        datatype: None,
    })
}

#[test]
fn construct_template_blank_nodes_are_fresh_per_row() {
    let q =
        parse_query("PREFIX : <http://ex/> CONSTRUCT { ?s :has [ :p ?o ] } WHERE { ?s :val ?o }")
            .expect("CONSTRUCT with `[ ]` in the template must parse");

    // The template anchor must be a real blank node (not a WHERE-style variable).
    assert!(
        q.construct_template
            .iter()
            .any(|t| matches!(t.subject, Term::BlankNode(_))
                || matches!(t.object, Term::BlankNode(_))),
        "CONSTRUCT template `[ ]` must lower to a Term::BlankNode: {:?}",
        q.construct_template
    );

    // Two solution rows.
    let mut r1 = Binding::new();
    r1.insert(var("s"), Term::Iri(NamedNode::new_unchecked(ex("x1"))));
    r1.insert(var("o"), lit("1"));
    let mut r2 = Binding::new();
    r2.insert(var("s"), Term::Iri(NamedNode::new_unchecked(ex("x2"))));
    r2.insert(var("o"), lit("2"));
    let solution = vec![r1, r2];

    let triples = instantiate_construct(&q.construct_template, &solution)
        .expect("instantiation must succeed");
    assert_eq!(triples.len(), 4, "2 rows × 2 template triples: {triples:?}");

    // For each row, the blank node linking `:x{n} :has _:b` must equal the
    // subject of `_:b :p "{n}"`; and the two rows must use DIFFERENT blank nodes.
    let has = ex("has");
    let p = ex("p");
    let bnode_for = |subj_iri: &str| -> String {
        // find (subj_iri :has BNODE)
        let link = triples
            .iter()
            .find(|t| iri_of(&t.subject).as_deref() == Some(subj_iri) && pred_is(t, &has))
            .unwrap_or_else(|| panic!("no :has triple for {subj_iri}"));
        match &link.object {
            Term::BlankNode(b) => b.clone(),
            other => panic!("expected a blank node object, got {other:?}"),
        }
    };
    let b1 = bnode_for(&ex("x1"));
    let b2 = bnode_for(&ex("x2"));
    assert_ne!(b1, b2, "each row must mint its OWN blank node");

    // Within row 1, the same blank node carries `:p "1"`.
    let carries = |bnode: &str, value: &str| {
        triples.iter().any(|t| {
            matches!(&t.subject, Term::BlankNode(b) if b == bnode)
                && pred_is(t, &p)
                && matches!(&t.object, Term::Literal(l) if l.value == value)
        })
    };
    assert!(carries(&b1, "1"), "row-1 blank node must carry :p \"1\"");
    assert!(carries(&b2, "2"), "row-2 blank node must carry :p \"2\"");
}
