//! # Store - WHERE-evaluating SPARQL Update handlers
//!
//! Real implementations of the pattern-based SPARQL 1.1 Update operations
//! (`DELETE { … } INSERT { … } WHERE { … }`, `INSERT { … } WHERE { … }`,
//! and `DELETE WHERE { … }`) bound to the **live** store.
//!
//! The previous handlers for these operations discarded the `WHERE` clause
//! entirely: they lifted the DELETE/INSERT *template* braces and applied them
//! as literal N-Triples data. That produced two failure modes:
//!
//! * templates containing variables (`?s ?p ?o`) failed to parse and returned
//!   HTTP 500 — the feature was effectively unusable; and
//! * a ground template such as `DELETE { <a> <b> <c> } WHERE { <nomatch> … }`
//!   deleted `<a> <b> <c>` *unconditionally* even though the `WHERE` matched
//!   nothing — silent wrong results and data loss.
//!
//! These handlers instead evaluate the `WHERE` basic graph pattern against the
//! live store to produce solution bindings, then instantiate the DELETE and
//! INSERT templates once per solution (delete-before-insert per SPARQL 1.1
//! §3.1.3). A `WHERE` that matches nothing yields zero modifications.
//!
//! ## Scope
//!
//! These handlers are reached only from the **AST-parsed** dispatch path
//! ([`dispatch_parsed_update`](crate::store::Store)), i.e. only when
//! `oxirs_arq::SparqlUpdateParser` successfully parses the statement. That
//! parser accepts only simple basic-graph-pattern `WHERE` clauses (no `FILTER`,
//! `OPTIONAL`, `GRAPH`, sub-`SELECT`, …); anything richer fails parsing and is
//! routed elsewhere. The basic-graph-pattern join implemented here is therefore
//! complete for every statement that can reach it.
//!
//! ## Known fidelity limitation
//!
//! `oxirs_arq`'s update tokeniser drops the datatype (`^^<iri>`) and language
//! (`@lang`) of a **constant** literal written directly in a template or
//! `WHERE` clause, exposing it as a plain (`xsd:string`) literal. Constant
//! literals in these positions are therefore matched/inserted as plain strings.
//! Variables — the overwhelmingly common case in real templates — are bound
//! from matched store quads and preserve full term fidelity (datatype, language,
//! blank nodes, IRIs) unchanged.
//!
//! 🤖 Authored for the production-hardening pass (fuseki-store package).

use super::*;
use oxirs_arq::{UpdatePatternTerm, UpdateTriplePattern};
use std::collections::HashMap;

/// A resolved binding environment: variable name → concrete term.
type Solution = HashMap<String, Term>;

impl Store {
    /// Execute `DELETE { del } INSERT { ins } WHERE { pattern }`
    /// (`SparqlUpdate::Modify`) against the live store.
    ///
    /// The `WHERE` pattern is evaluated first to produce the full solution
    /// sequence; deletions and insertions are computed from those solutions
    /// and then applied delete-before-insert (SPARQL 1.1 §3.1.3), so an
    /// inserted triple is never re-deleted by the same operation.
    pub(super) fn execute_modify_where(
        &self,
        store: &mut dyn CoreStore,
        delete: &[UpdateTriplePattern],
        insert: &[UpdateTriplePattern],
        where_clause: &[UpdateTriplePattern],
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let solutions = self.evaluate_bgp(store, where_clause)?;

        // Materialise every delete/insert instantiation from the pre-modification
        // solution sequence *before* touching the store.
        let mut delete_quads = Vec::new();
        for solution in &solutions {
            for pattern in delete {
                if let Some(quad) = instantiate_pattern(pattern, solution)? {
                    delete_quads.push(quad);
                }
            }
        }
        let mut insert_quads = Vec::new();
        for solution in &solutions {
            for pattern in insert {
                if let Some(quad) = instantiate_pattern(pattern, solution)? {
                    insert_quads.push(quad);
                }
            }
        }

        let mut deleted_count = 0;
        for quad in delete_quads {
            if store
                .remove_quad(&quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to remove quad: {e}")))?
            {
                deleted_count += 1;
            }
        }
        let mut inserted_count = 0;
        for quad in insert_quads {
            if store
                .insert_quad(quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                inserted_count += 1;
            }
        }

        Ok((
            "DELETE/INSERT",
            inserted_count,
            deleted_count,
            vec!["default".to_string()],
        ))
    }

    /// Execute `INSERT { template } WHERE { pattern }`
    /// (`SparqlUpdate::InsertWhere`) against the live store.
    pub(super) fn execute_insert_where(
        &self,
        store: &mut dyn CoreStore,
        template: &[UpdateTriplePattern],
        where_clause: &[UpdateTriplePattern],
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let solutions = self.evaluate_bgp(store, where_clause)?;
        let mut insert_quads = Vec::new();
        for solution in &solutions {
            for pattern in template {
                if let Some(quad) = instantiate_pattern(pattern, solution)? {
                    insert_quads.push(quad);
                }
            }
        }
        let mut inserted_count = 0;
        for quad in insert_quads {
            if store
                .insert_quad(quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to insert quad: {e}")))?
            {
                inserted_count += 1;
            }
        }
        Ok(("INSERT", inserted_count, 0, vec!["default".to_string()]))
    }

    /// Execute `DELETE WHERE { pattern }` (`SparqlUpdate::DeleteWhere`) against
    /// the live store.
    ///
    /// For `DELETE WHERE { P }` the delete template is `P` itself (SPARQL 1.1
    /// §3.1.3.3); the parser leaves `template` empty in that case, so the
    /// `where_clause` is used as the delete template when `template` is empty.
    pub(super) fn execute_delete_where_pattern(
        &self,
        store: &mut dyn CoreStore,
        template: &[UpdateTriplePattern],
        where_clause: &[UpdateTriplePattern],
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let solutions = self.evaluate_bgp(store, where_clause)?;
        let effective_template = if template.is_empty() {
            where_clause
        } else {
            template
        };
        let mut delete_quads = Vec::new();
        for solution in &solutions {
            for pattern in effective_template {
                if let Some(quad) = instantiate_pattern(pattern, solution)? {
                    delete_quads.push(quad);
                }
            }
        }
        let mut deleted_count = 0;
        for quad in delete_quads {
            if store
                .remove_quad(&quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to remove quad: {e}")))?
            {
                deleted_count += 1;
            }
        }
        Ok((
            "DELETE WHERE",
            0,
            deleted_count,
            vec!["default".to_string()],
        ))
    }

    /// Evaluate a basic graph pattern against the default graph of the live
    /// store, returning every solution binding.
    ///
    /// Patterns are joined with a left-to-right nested-loop join: each pattern
    /// extends the partial solutions produced so far. Variables that are
    /// already bound act as concrete filters; unbound variables act as
    /// wildcards and are bound from each matching quad. A variable that occurs
    /// more than once (within a pattern or across patterns) is constrained to a
    /// single consistent value.
    fn evaluate_bgp(
        &self,
        store: &mut dyn CoreStore,
        patterns: &[UpdateTriplePattern],
    ) -> FusekiResult<Vec<Solution>> {
        // An empty BGP has exactly one solution: the empty binding.
        let mut solutions: Vec<Solution> = vec![Solution::new()];

        let default_graph = GraphName::DefaultGraph;
        for pattern in patterns {
            let mut next: Vec<Solution> = Vec::new();
            for solution in &solutions {
                // Build concrete filters for positions that are bound or
                // constant; leave unbound-variable positions as wildcards.
                let subject_filter = resolve_subject_filter(&pattern.s, solution)?;
                let predicate_filter = resolve_predicate_filter(&pattern.p, solution)?;
                let object_filter = resolve_object_filter(&pattern.o, solution)?;

                // A constant/bound position that is not a valid RDF term for
                // its slot (e.g. a literal in subject position) can never match.
                let (subject_ref, predicate_ref, object_ref) =
                    match (&subject_filter, &predicate_filter, &object_filter) {
                        (SlotFilter::Unmatchable, _, _)
                        | (_, SlotFilter::Unmatchable, _)
                        | (_, _, SlotFilter::Unmatchable) => continue,
                        (s, p, o) => (s.as_ref_opt(), p.as_ref_opt(), o.as_ref_opt()),
                    };

                let matches = store
                    .find_quads(subject_ref, predicate_ref, object_ref, Some(&default_graph))
                    .map_err(|e| {
                        FusekiError::update_execution(format!(
                            "Failed to evaluate WHERE pattern: {e}"
                        ))
                    })?;

                for quad in matches {
                    let mut candidate = solution.clone();
                    let mut consistent = true;
                    bind_if_variable(
                        &pattern.s,
                        subject_to_term(quad.subject()),
                        &mut candidate,
                        &mut consistent,
                    );
                    if consistent {
                        bind_if_variable(
                            &pattern.p,
                            predicate_to_term(quad.predicate()),
                            &mut candidate,
                            &mut consistent,
                        );
                    }
                    if consistent {
                        bind_if_variable(
                            &pattern.o,
                            object_to_term(quad.object()),
                            &mut candidate,
                            &mut consistent,
                        );
                    }
                    if consistent {
                        next.push(candidate);
                    }
                }
            }
            solutions = next;
            if solutions.is_empty() {
                break;
            }
        }
        Ok(solutions)
    }
}

/// A resolved filter for one triple position.
enum SlotFilter<T> {
    /// Match anything (unbound variable).
    Any,
    /// Match exactly this term.
    Exact(T),
    /// The constant/bound term is invalid for this position and can never
    /// match (e.g. a literal used as a subject or predicate).
    Unmatchable,
}

impl<T> SlotFilter<T> {
    fn as_ref_opt(&self) -> Option<&T> {
        match self {
            SlotFilter::Exact(t) => Some(t),
            _ => None,
        }
    }
}

/// Resolve a subject-position pattern term into a `find_quads` filter.
fn resolve_subject_filter(
    term: &UpdatePatternTerm,
    solution: &Solution,
) -> FusekiResult<SlotFilter<Subject>> {
    match resolve_bound_term(term, solution)? {
        None => Ok(SlotFilter::Any),
        Some(t) => Ok(match term_as_subject(&t) {
            Some(s) => SlotFilter::Exact(s),
            None => SlotFilter::Unmatchable,
        }),
    }
}

/// Resolve a predicate-position pattern term into a `find_quads` filter.
fn resolve_predicate_filter(
    term: &UpdatePatternTerm,
    solution: &Solution,
) -> FusekiResult<SlotFilter<Predicate>> {
    match resolve_bound_term(term, solution)? {
        None => Ok(SlotFilter::Any),
        Some(t) => Ok(match term_as_predicate(&t) {
            Some(p) => SlotFilter::Exact(p),
            None => SlotFilter::Unmatchable,
        }),
    }
}

/// Resolve an object-position pattern term into a `find_quads` filter.
fn resolve_object_filter(
    term: &UpdatePatternTerm,
    solution: &Solution,
) -> FusekiResult<SlotFilter<Object>> {
    match resolve_bound_term(term, solution)? {
        None => Ok(SlotFilter::Any),
        Some(t) => Ok(SlotFilter::Exact(term_as_object(&t))),
    }
}

/// Return the concrete term a pattern position resolves to *right now*:
/// `None` for an unbound variable (wildcard), `Some(term)` for a constant or a
/// variable already bound in `solution`.
fn resolve_bound_term(term: &UpdatePatternTerm, solution: &Solution) -> FusekiResult<Option<Term>> {
    match term {
        UpdatePatternTerm::Variable(name) => Ok(solution.get(name).cloned()),
        other => Ok(Some(pattern_term_to_concrete(other)?)),
    }
}

/// If `pattern` is an unbound variable, bind it to `value` in `solution`;
/// if it is already bound to a different value, mark the join inconsistent.
fn bind_if_variable(
    pattern: &UpdatePatternTerm,
    value: Term,
    solution: &mut Solution,
    consistent: &mut bool,
) {
    if let UpdatePatternTerm::Variable(name) = pattern {
        match solution.get(name) {
            Some(existing) if existing != &value => *consistent = false,
            Some(_) => {}
            None => {
                solution.insert(name.clone(), value);
            }
        }
    }
}

/// Instantiate a template triple pattern against a solution binding, producing
/// a concrete quad in the default graph.
///
/// Returns `Ok(None)` — the triple is simply not instantiated — when a variable
/// in the template is unbound for this solution, or when a bound value is not a
/// valid RDF term for its position (SPARQL 1.1 §3.1.3: such template triples are
/// silently skipped, not errors). A genuinely malformed *constant* (e.g. an
/// invalid IRI literal) surfaces as an error.
fn instantiate_pattern(
    pattern: &UpdateTriplePattern,
    solution: &Solution,
) -> FusekiResult<Option<Quad>> {
    let subject_term = match resolve_template_term(&pattern.s, solution)? {
        Some(t) => t,
        None => return Ok(None),
    };
    let predicate_term = match resolve_template_term(&pattern.p, solution)? {
        Some(t) => t,
        None => return Ok(None),
    };
    let object_term = match resolve_template_term(&pattern.o, solution)? {
        Some(t) => t,
        None => return Ok(None),
    };

    let subject = match term_as_subject(&subject_term) {
        Some(s) => s,
        None => return Ok(None),
    };
    let predicate = match term_as_predicate(&predicate_term) {
        Some(p) => p,
        None => return Ok(None),
    };
    let object = term_as_object(&object_term);

    Ok(Some(Quad::new(
        subject,
        predicate,
        object,
        GraphName::DefaultGraph,
    )))
}

/// Resolve a template position: constants become their concrete term, a bound
/// variable becomes its binding, an unbound variable yields `None` (skip).
fn resolve_template_term(
    term: &UpdatePatternTerm,
    solution: &Solution,
) -> FusekiResult<Option<Term>> {
    match term {
        UpdatePatternTerm::Variable(name) => Ok(solution.get(name).cloned()),
        other => Ok(Some(pattern_term_to_concrete(other)?)),
    }
}

/// Convert a non-variable pattern term into a concrete [`Term`].
fn pattern_term_to_concrete(term: &UpdatePatternTerm) -> FusekiResult<Term> {
    match term {
        UpdatePatternTerm::Iri(iri) => {
            let node = NamedNode::new(iri).map_err(|e| {
                FusekiError::update_execution(format!("Invalid IRI '<{iri}>' in update: {e}"))
            })?;
            Ok(Term::NamedNode(node))
        }
        UpdatePatternTerm::BlankNode(label) => {
            let node = BlankNode::new(label).map_err(|e| {
                FusekiError::update_execution(format!(
                    "Invalid blank node '_:{label}' in update: {e}"
                ))
            })?;
            Ok(Term::BlankNode(node))
        }
        UpdatePatternTerm::Literal(token) => Ok(Term::Literal(parse_literal_token(token))),
        UpdatePatternTerm::Variable(name) => Err(FusekiError::update_execution(format!(
            "internal error: variable ?{name} passed to constant-term conversion"
        ))),
    }
}

/// Parse a raw literal token as produced by the update tokeniser (the value is
/// wrapped in matching `"…"` or `'…'` quotes, with backslash escapes) into a
/// plain literal. Datatype/language are not present on the token and therefore
/// cannot be recovered here (see module-level fidelity note).
fn parse_literal_token(token: &str) -> Literal {
    let bytes = token.as_bytes();
    let value = if token.len() >= 2
        && ((bytes[0] == b'"' && bytes[token.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[token.len() - 1] == b'\''))
    {
        unescape_literal(&token[1..token.len() - 1])
    } else {
        token.to_string()
    };
    Literal::new_simple_literal(value)
}

/// Unescape the standard N-Triples/Turtle escape sequences inside a literal
/// body (quotes already stripped).
fn unescape_literal(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('b') => out.push('\u{0008}'),
            Some('f') => out.push('\u{000C}'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some('\\') => out.push('\\'),
            Some('u') => push_unicode_escape(&mut chars, &mut out, 4),
            Some('U') => push_unicode_escape(&mut chars, &mut out, 8),
            Some(other) => {
                // Unknown escape: preserve the backslash and the character
                // verbatim rather than silently dropping either.
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Consume `width` hex digits from `chars` and push the decoded scalar value.
/// On a malformed escape the consumed characters are emitted verbatim.
fn push_unicode_escape(chars: &mut std::str::Chars<'_>, out: &mut String, width: usize) {
    let mut hex = String::with_capacity(width);
    for _ in 0..width {
        match chars.next() {
            Some(c) => hex.push(c),
            None => break,
        }
    }
    match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
        Some(decoded) => out.push(decoded),
        None => {
            out.push('\\');
            out.push(if width == 4 { 'u' } else { 'U' });
            out.push_str(&hex);
        }
    }
}

/// Convert a `Term` into a `Subject`, or `None` if it is not a valid subject.
fn term_as_subject(term: &Term) -> Option<Subject> {
    match term {
        Term::NamedNode(n) => Some(Subject::NamedNode(n.clone())),
        Term::BlankNode(b) => Some(Subject::BlankNode(b.clone())),
        Term::QuotedTriple(q) => Some(Subject::QuotedTriple(q.clone())),
        Term::Literal(_) | Term::Variable(_) => None,
    }
}

/// Convert a `Term` into a `Predicate`, or `None` if it is not a valid predicate.
fn term_as_predicate(term: &Term) -> Option<Predicate> {
    match term {
        Term::NamedNode(n) => Some(Predicate::NamedNode(n.clone())),
        _ => None,
    }
}

/// Convert a `Term` into an `Object` (every term kind is a valid object).
fn term_as_object(term: &Term) -> Object {
    match term {
        Term::NamedNode(n) => Object::NamedNode(n.clone()),
        Term::BlankNode(b) => Object::BlankNode(b.clone()),
        Term::Literal(l) => Object::Literal(l.clone()),
        Term::QuotedTriple(q) => Object::QuotedTriple(q.clone()),
        Term::Variable(v) => Object::Variable(v.clone()),
    }
}

/// Lift a matched quad's subject into a binding `Term`.
fn subject_to_term(subject: &Subject) -> Term {
    match subject {
        Subject::NamedNode(n) => Term::NamedNode(n.clone()),
        Subject::BlankNode(b) => Term::BlankNode(b.clone()),
        Subject::Variable(v) => Term::Variable(v.clone()),
        Subject::QuotedTriple(q) => Term::QuotedTriple(q.clone()),
    }
}

/// Lift a matched quad's predicate into a binding `Term`.
fn predicate_to_term(predicate: &Predicate) -> Term {
    match predicate {
        Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
        Predicate::Variable(v) => Term::Variable(v.clone()),
    }
}

/// Lift a matched quad's object into a binding `Term`.
fn object_to_term(object: &Object) -> Term {
    match object {
        Object::NamedNode(n) => Term::NamedNode(n.clone()),
        Object::BlankNode(b) => Term::BlankNode(b.clone()),
        Object::Literal(l) => Term::Literal(l.clone()),
        Object::Variable(v) => Term::Variable(v.clone()),
        Object::QuotedTriple(q) => Term::QuotedTriple(q.clone()),
    }
}

#[cfg(test)]
mod where_update_tests {
    use crate::store::Store;

    /// Regression: `DELETE { <a> <b> <c> } WHERE { <nomatch> … }` must delete
    /// NOTHING when the WHERE clause matches nothing. The old handler discarded
    /// the WHERE clause and deleted the ground template unconditionally.
    #[test]
    fn regression_modify_nonmatching_where_deletes_nothing() {
        let store = Store::new().expect("create store");
        store
            .update(
                "INSERT DATA { <http://example.org/a> <http://example.org/b> <http://example.org/c> . }",
            )
            .expect("seed insert");
        assert_eq!(store.count_triples("default"), 1);

        let result = store
            .update(
                "DELETE { <http://example.org/a> <http://example.org/b> <http://example.org/c> } \
                 WHERE { <http://example.org/nomatch> <http://example.org/type> <http://example.org/X> }",
            )
            .expect("modify with non-matching WHERE should succeed");
        assert_eq!(
            result.stats.quads_deleted, 0,
            "a non-matching WHERE must delete nothing"
        );
        assert_eq!(
            store.count_triples("default"),
            1,
            "the ground DELETE template must NOT be applied unconditionally"
        );
    }

    /// Regression: a variable-bearing `DELETE { ?s ?p ?o } WHERE { ?s ?p ?o }`
    /// must actually evaluate and delete the matched triples (old handler
    /// returned HTTP 500 because it tried to parse `?s` as N-Triples data).
    #[test]
    fn regression_delete_insert_where_with_variables_evaluates() {
        let store = Store::new().expect("create store");
        store
            .update(
                "INSERT DATA {\n\
                   <http://example.org/s1> <http://example.org/p> \"v1\" .\n\
                   <http://example.org/s2> <http://example.org/p> \"v2\" .\n\
                 }",
            )
            .expect("seed insert");
        assert_eq!(store.count_triples("default"), 2);

        // DELETE { ?s <p> ?o } INSERT { ?s <q> ?o } WHERE { ?s <p> ?o }
        let result = store
            .update(
                "DELETE { ?s <http://example.org/p> ?o } \
                 INSERT { ?s <http://example.org/q> ?o } \
                 WHERE { ?s <http://example.org/p> ?o }",
            )
            .expect("variable modify should succeed");
        assert_eq!(result.stats.quads_deleted, 2, "both <p> triples deleted");
        assert_eq!(result.stats.quads_inserted, 2, "both <q> triples inserted");
        assert_eq!(store.count_triples("default"), 2, "net count unchanged");

        // The old predicate must be gone and the new one present.
        let remaining = store
            .update("DELETE WHERE { ?s <http://example.org/q> ?o }")
            .expect("delete rewritten triples");
        assert_eq!(
            remaining.stats.quads_deleted, 2,
            "the INSERT template must have written <q> triples"
        );
        assert_eq!(store.count_triples("default"), 0);
    }

    /// Regression: `INSERT { ?s <derived> ?o } WHERE { ?s <p> ?o }` must
    /// evaluate the WHERE and instantiate the template per solution.
    #[test]
    fn regression_insert_where_evaluates_pattern() {
        let store = Store::new().expect("create store");
        store
            .update(
                "INSERT DATA {\n\
                   <http://example.org/s1> <http://example.org/p> <http://example.org/o1> .\n\
                   <http://example.org/s2> <http://example.org/p> <http://example.org/o2> .\n\
                 }",
            )
            .expect("seed insert");

        let result = store
            .update(
                "INSERT { ?s <http://example.org/derived> ?o } \
                 WHERE { ?s <http://example.org/p> ?o }",
            )
            .expect("insert-where should succeed");
        assert_eq!(
            result.stats.quads_inserted, 2,
            "one derived triple per matched solution"
        );
        assert_eq!(store.count_triples("default"), 4);
    }

    /// Regression: `INSERT { … } WHERE { <nomatch> … }` with an empty solution
    /// set must insert nothing.
    #[test]
    fn regression_insert_where_no_match_inserts_nothing() {
        let store = Store::new().expect("create store");
        let result = store
            .update(
                "INSERT { <http://example.org/a> <http://example.org/b> <http://example.org/c> } \
                 WHERE { ?s <http://example.org/missing> ?o }",
            )
            .expect("insert-where with empty match should succeed");
        assert_eq!(result.stats.quads_inserted, 0);
        assert_eq!(store.count_triples("default"), 0);
    }

    /// Regression: `DELETE WHERE { ?s ?p ?o }` deletes exactly the matched
    /// triples via real pattern evaluation.
    #[test]
    fn regression_delete_where_variables_deletes_matched() {
        let store = Store::new().expect("create store");
        store
            .update(
                "INSERT DATA {\n\
                   <http://example.org/s1> <http://example.org/p1> \"v1\" .\n\
                   <http://example.org/s2> <http://example.org/p2> \"v2\" .\n\
                 }",
            )
            .expect("seed insert");

        let result = store
            .update("DELETE WHERE { ?s <http://example.org/p1> ?o }")
            .expect("delete-where should succeed");
        assert_eq!(
            result.stats.quads_deleted, 1,
            "only the <p1> triple matches"
        );
        assert_eq!(store.count_triples("default"), 1);
    }

    /// A join across two patterns must intersect on the shared variable.
    #[test]
    fn regression_where_join_on_shared_variable() {
        let store = Store::new().expect("create store");
        store
            .update(
                "INSERT DATA {\n\
                   <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .\n\
                   <http://example.org/bob> <http://example.org/age> \"30\" .\n\
                 }",
            )
            .expect("seed insert");

        // Delete alice-knows-bob only when bob has an age.
        let result = store
            .update(
                "DELETE { ?a <http://example.org/knows> ?b } \
                 WHERE { ?a <http://example.org/knows> ?b . ?b <http://example.org/age> ?age }",
            )
            .expect("join modify should succeed");
        assert_eq!(
            result.stats.quads_deleted, 1,
            "the join must match the one knows-triple whose object has an age"
        );
    }
}
