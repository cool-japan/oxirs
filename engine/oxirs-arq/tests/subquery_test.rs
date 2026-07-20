//! SPARQL 1.1 SubSelect (`{ SELECT … }`) — parsing and Serial execution.
//!
//! Before this feature `SELECT … WHERE { { SELECT ?x WHERE { … } } … }` was a
//! hard 400 (known limitation). A SubSelect (§8.2.4) is an independent,
//! **non-correlated** query whose result set joins into the enclosing group on
//! the shared *projected* variables. These tests assert both that such queries
//! now parse and — crucially — that the lowered algebra executes correctly
//! through the same Serial executor the fuseki server forces (no optimizer runs
//! in that path, so the parsed algebra alone determines the answer).
//!
//! The load-bearing case is an inner aggregate (`COUNT … GROUP BY`) whose
//! grouped result is joined into an outer BGP: that exercises a `Group` node
//! nested under a `Join`, the one path the pre-existing modifier tests do not
//! cover.

use std::collections::HashMap;

use oxirs_arq::algebra::{Algebra, PropertyPath, Term, TriplePattern};
use oxirs_arq::executor::ExecutionStrategy;
use oxirs_arq::query::{parse_query, Query};
use oxirs_arq::{Dataset, Literal, QueryExecutor, Variable};
use oxirs_core::model::NamedNode;

// ── a minimal, path-aware in-memory dataset ─────────────────────────────────
//
// The parser encodes a single-IRI predicate as a length-one property path
// (`Term::PropertyPath(PropertyPath::Iri(_))`); the real store's `find_triples`
// normalizes that to a plain IRI when matching, so this fixture does the same.

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn plain(s: &str) -> Term {
    Term::Literal(Literal {
        value: s.to_string(),
        language: None,
        datatype: None,
    })
}

fn normalize_predicate(t: &Term) -> Term {
    match t {
        Term::PropertyPath(PropertyPath::Iri(n)) => Term::Iri(n.clone()),
        other => other.clone(),
    }
}

struct PathDataset {
    triples: Vec<(Term, Term, Term)>,
}

impl PathDataset {
    fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }
    fn add(&mut self, s: Term, p: Term, o: Term) {
        self.triples.push((s, p, o));
    }
}

impl Dataset for PathDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> anyhow::Result<Vec<(Term, Term, Term)>> {
        let want_pred = normalize_predicate(&pattern.predicate);
        let mut out = Vec::new();
        for (s, p, o) in &self.triples {
            let s_ok = matches!(pattern.subject, Term::Variable(_)) || &pattern.subject == s;
            let p_ok = matches!(want_pred, Term::Variable(_)) || &want_pred == p;
            let o_ok = matches!(pattern.object, Term::Variable(_)) || &pattern.object == o;
            if s_ok && p_ok && o_ok {
                out.push((s.clone(), p.clone(), o.clone()));
            }
        }
        Ok(out)
    }

    fn contains_triple(&self, s: &Term, p: &Term, o: &Term) -> anyhow::Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(a, b, c)| a == s && b == p && c == o))
    }

    fn subjects(&self) -> anyhow::Result<Vec<Term>> {
        let mut v: Vec<Term> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        v.sort();
        v.dedup();
        Ok(v)
    }

    fn predicates(&self) -> anyhow::Result<Vec<Term>> {
        let mut v: Vec<Term> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        v.sort();
        v.dedup();
        Ok(v)
    }

    fn objects(&self) -> anyhow::Result<Vec<Term>> {
        let mut v: Vec<Term> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        v.sort();
        v.dedup();
        Ok(v)
    }
}

// ── harness ─────────────────────────────────────────────────────────────────

fn parse(query: &str) -> Query {
    parse_query(query).unwrap_or_else(|e| panic!("query must parse: {e}"))
}

/// Assemble the top-level solution-modifier stack exactly as the fuseki
/// `build_select_algebra` does and evaluate through a Serial executor. The
/// SubSelect's own modifiers are already baked into `where_clause` by the
/// parser, so this only wraps the OUTER query's modifiers.
fn run_like_server(q: &Query, dataset: &PathDataset) -> Vec<HashMap<Variable, Term>> {
    let mut alg = q.where_clause.clone();
    if !q.order_by.is_empty() {
        alg = Algebra::OrderBy {
            pattern: Box::new(alg),
            conditions: q.order_by.clone(),
        };
    }
    if !q.select_variables.is_empty() {
        alg = Algebra::Project {
            pattern: Box::new(alg),
            variables: q.select_variables.clone(),
        };
    }
    if q.limit.is_some() || q.offset.is_some() {
        alg = Algebra::Slice {
            pattern: Box::new(alg),
            offset: q.offset,
            limit: q.limit,
        };
    }
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let (solution, _stats) = executor
        .execute(&alg, dataset)
        .unwrap_or_else(|e| panic!("execution must succeed: {e}"));
    solution
}

fn column(solution: &[HashMap<Variable, Term>], var: &str) -> Vec<Term> {
    let v = Variable::new(var).expect("valid variable");
    solution.iter().filter_map(|b| b.get(&v).cloned()).collect()
}

const IN_SCHEME: &str = "http://example.org/inScheme";
const LABEL: &str = "http://example.org/label";
const OTHER: &str = "http://example.org/other";

/// Three concepts split across two schemes (A: 2, B: 1), each scheme labelled.
fn scheme_dataset() -> PathDataset {
    let mut d = PathDataset::new();
    d.add(
        iri("http://example.org/c1"),
        iri(IN_SCHEME),
        iri("http://example.org/A"),
    );
    d.add(
        iri("http://example.org/c2"),
        iri(IN_SCHEME),
        iri("http://example.org/A"),
    );
    d.add(
        iri("http://example.org/c3"),
        iri(IN_SCHEME),
        iri("http://example.org/B"),
    );
    d.add(iri("http://example.org/A"), iri(LABEL), plain("Alpha"));
    d.add(iri("http://example.org/B"), iri(LABEL), plain("Beta"));
    d
}

// ── parsing: the previously-400 forms now parse ─────────────────────────────

#[test]
fn plain_subselect_parses() {
    // The canonical known-limitation form.
    assert!(
        parse_query("SELECT ?x WHERE { { SELECT ?x WHERE { ?x ?p ?o } } }").is_ok(),
        "`{{ SELECT ?x WHERE {{ … }} }}` must parse (was a hard 400)"
    );
}

#[test]
fn subselect_with_inner_limit_parses() {
    assert!(
        parse_query("SELECT ?x WHERE { { SELECT ?x WHERE { ?x ?p ?o } LIMIT 5 } ?x ?q ?r }")
            .is_ok(),
        "a SubSelect carrying its own LIMIT must parse"
    );
}

#[test]
fn subselect_with_aggregate_parses() {
    assert!(
        parse_query(
            "SELECT ?s ?n WHERE { { SELECT ?s (COUNT(?c) AS ?n) WHERE { ?c ?p ?s } GROUP BY ?s } }"
        )
        .is_ok(),
        "a SubSelect with an aggregate + GROUP BY must parse"
    );
}

#[test]
fn subselect_order_by_terminated_by_brace_parses() {
    // The critical second consumer of the `is_solution_modifier_end` +
    // RightBrace fix: an inner ORDER BY with NO trailing LIMIT is terminated
    // directly by `}`. Without the fix the ORDER BY condition loop reads past
    // the brace and fails with "Expected primary expression".
    assert!(
        parse_query("SELECT ?x WHERE { { SELECT ?x WHERE { ?x ?p ?o } ORDER BY ?x } }").is_ok(),
        "a SubSelect whose ORDER BY is closed by `}}` (no trailing LIMIT) must parse"
    );
}

#[test]
fn subselect_distinct_order_by_limit_parses() {
    assert!(
        parse_query(
            "SELECT ?x WHERE { { SELECT DISTINCT ?x WHERE { ?x ?p ?o } ORDER BY ?x LIMIT 5 } }"
        )
        .is_ok(),
        "a SubSelect combining DISTINCT + ORDER BY + inner LIMIT must parse"
    );
}

#[test]
fn subselect_group_by_having_terminated_by_brace_parses() {
    assert!(
        parse_query(
            "SELECT ?s ?n WHERE { \
               { SELECT ?s (COUNT(?c) AS ?n) WHERE { ?c ?p ?s } GROUP BY ?s HAVING (COUNT(?c) > 1) } \
             }"
        )
        .is_ok(),
        "a SubSelect with GROUP BY + HAVING closed by `}}` must parse"
    );
}

#[test]
fn ask_subquery_parses_without_400() {
    // Non-standard (SubSelect is SELECT-only) but must not 400.
    assert!(
        parse_query("SELECT ?s WHERE { { ASK { ?s ?p ?o } } }").is_ok(),
        "`{{ ASK {{ … }} }}` must parse rather than 400"
    );
}

// ── execution: inner aggregate joined into an outer BGP (Group under Join) ──

#[test]
fn inner_aggregate_joins_outer_bgp() {
    let d = scheme_dataset();
    // Inner: count concepts per scheme → {(A,2),(B,1)}. Outer: attach the label.
    let q = parse(
        "SELECT ?scheme ?n ?lbl WHERE { \
           { SELECT ?scheme (COUNT(?c) AS ?n) WHERE { ?c <http://example.org/inScheme> ?scheme } \
             GROUP BY ?scheme } \
           ?scheme <http://example.org/label> ?lbl \
         }",
    );
    let sol = run_like_server(&q, &d);

    assert_eq!(
        sol.len(),
        2,
        "one row per scheme after joining the grouped counts with labels, got {sol:?}"
    );

    // Pair up (label -> count) to assert the grouped counts survived the join.
    let mut pairs: Vec<(String, String)> = sol
        .iter()
        .map(|b| {
            let lbl = match b.get(&Variable::new("lbl").unwrap()) {
                Some(Term::Literal(l)) => l.value.clone(),
                other => panic!("?lbl must be a literal, got {other:?}"),
            };
            let n = match b.get(&Variable::new("n").unwrap()) {
                Some(Term::Literal(l)) => l.value.clone(),
                other => panic!("?n (COUNT) must be a literal, got {other:?}"),
            };
            (lbl, n)
        })
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("Alpha".to_string(), "2".to_string()),
            ("Beta".to_string(), "1".to_string()),
        ],
        "grouped COUNT must join to the right label"
    );
}

// ── execution: projection restricts what leaves the subquery ────────────────

#[test]
fn unprojected_inner_variable_does_not_leak() {
    let d = scheme_dataset();
    // Inner binds ?c internally but projects only ?scheme; ?c must not appear
    // in any outer binding.
    let q = parse(
        "SELECT * WHERE { { SELECT ?scheme WHERE { ?c <http://example.org/inScheme> ?scheme } } }",
    );
    let sol = run_like_server(&q, &d);

    let c = Variable::new("c").expect("valid variable");
    assert!(
        sol.iter().all(|b| !b.contains_key(&c)),
        "the unprojected inner variable ?c must not leak to the outer solution: {sol:?}"
    );
    // Projection does not deduplicate: three concepts → three ?scheme rows.
    assert_eq!(
        column(&sol, "scheme").len(),
        3,
        "projecting ?scheme keeps one row per inner match (no implicit DISTINCT)"
    );
}

#[test]
fn unprojected_inner_variable_cannot_correlate_outer() {
    let d = scheme_dataset();
    // ?c is projected away inside, so the outer `?c ...` is an INDEPENDENT
    // variable — the subquery must not silently correlate on it. The outer
    // `?c <other> ?z` matches nothing (no :other triples), so the join is empty.
    let q = parse(
        "SELECT ?scheme ?z WHERE { \
           { SELECT ?scheme WHERE { ?c <http://example.org/inScheme> ?scheme } } \
           ?c <http://example.org/other> ?z \
         }",
    );
    let sol = run_like_server(&q, &d);
    assert!(
        sol.is_empty(),
        "no :other triples exist, so the outer join yields nothing: {sol:?}"
    );
    // Sanity: adding one :other triple whose subject (x9) is NOT among the
    // inner ?c values {c1,c2,c3} still joins. The subquery projects only
    // ?scheme (3 rows: A,A,B) and shares no variable with the outer
    // `?c :other ?z` (1 row), so the result is their cross product = 3 rows.
    // A correlated engine would instead constrain outer ?c to the inner values
    // and yield 0 rows — so 3 rows proves non-correlation.
    let mut d2 = scheme_dataset();
    d2.add(iri("http://example.org/x9"), iri(OTHER), plain("z"));
    let sol2 = run_like_server(&q, &d2);
    assert_eq!(
        sol2.len(),
        3,
        "outer ?c ranges freely (cross product of 3 inner ?scheme rows × 1 :other row), \
         proving non-correlation: {sol2:?}"
    );
}

// ── execution: an inner LIMIT is honoured inside the subquery ────────────────

#[test]
fn inner_limit_is_applied_inside_subquery() {
    let d = scheme_dataset();
    let q = parse(
        "SELECT ?c WHERE { { SELECT ?c WHERE { ?c <http://example.org/inScheme> ?scheme } LIMIT 1 } }",
    );
    let sol = run_like_server(&q, &d);
    assert_eq!(
        sol.len(),
        1,
        "the SubSelect's own LIMIT 1 must cap it at one row (3 candidates exist): {sol:?}"
    );
}
