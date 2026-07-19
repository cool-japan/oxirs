//! Regression coverage for two "HTTP-200 wrong answer" parser defects surfaced
//! by the wik.jp GraphQL catalog verification:
//!
//!   1. **OFFSET-before-LIMIT drops the LIMIT.** `… OFFSET n LIMIT m` is legal
//!      SPARQL 1.1 (`LimitOffsetClauses ::= LimitClause OffsetClause?
//!      | OffsetClause LimitClause?`, §18.5), but a fixed LIMIT-then-OFFSET
//!      reader consumed only the OFFSET and silently ignored the trailing LIMIT,
//!      so the query returned every row past the offset instead of `m` rows.
//!
//!   2. **FILTER after `{ A } UNION { B }` must scope the WHOLE group.** A bare
//!      FILTER in a group graph pattern applies to the entire group regardless of
//!      its textual position (§18.2.2 "collect FILTERs"), so
//!      `{ A } UNION { B } FILTER(c)` is `Filter(Union(A,B), c)` — the filter
//!      constrains BOTH union branches, not just the first. FILTER written before
//!      the union is equivalent.
//!
//! Each case asserts the parsed algebra shape AND the executed row set, driving
//! the same Serial executor the fuseki server forces (no optimizer runs in that
//! path, so the parsed algebra alone determines the answer).

use std::collections::HashMap;

use oxirs_arq::algebra::{Algebra, PropertyPath, Term, TriplePattern};
use oxirs_arq::executor::ExecutionStrategy;
use oxirs_arq::query::{parse_query, Query};
use oxirs_arq::{Dataset, Literal, QueryExecutor, Variable};
use oxirs_core::model::NamedNode;

// ── a minimal, path-aware in-memory dataset ─────────────────────────────────
//
// The parser encodes a single-IRI predicate as a length-one property path
// (`Term::PropertyPath(PropertyPath::Iri(_))`) — the real store's `find_triples`
// normalizes that to a plain IRI when matching, so a faithful test fixture must
// do the same. (The shared `tests/common` MockDataset compares terms verbatim
// and would never match a parsed BGP predicate.)

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

/// Collapse a length-one `PropertyPath::Iri` predicate to a plain `Term::Iri`,
/// mirroring the store's predicate handling; other terms pass through.
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

// ── shared harness ──────────────────────────────────────────────────────────

fn parse(query: &str) -> Query {
    parse_query(query).expect("query must parse")
}

/// Assemble the full solution-modifier stack exactly as the fuseki
/// `build_select_algebra` does (WHERE → OrderBy → Project → Slice) and evaluate
/// it through a Serial `QueryExecutor` — the strategy fuseki forces for every
/// query.
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
        .expect("execution must succeed");
    solution
}

/// Values bound to `?var` across a solution, in solution order.
fn column(solution: &[HashMap<Variable, Term>], var: &str) -> Vec<Term> {
    let v = Variable::new(var).expect("valid variable");
    solution.iter().filter_map(|b| b.get(&v).cloned()).collect()
}

// ── Defect 1: LIMIT/OFFSET in either order ───────────────────────────────────

const P: &str = "http://example.org/p";

/// 20 subjects s00..s19, each `sNN <p> "NN"`, so an ASCII ORDER BY ?s is a
/// stable, zero-padded numeric order.
fn twenty_row_dataset() -> PathDataset {
    let mut d = PathDataset::new();
    for n in 0..20 {
        d.add(
            iri(&format!("http://example.org/s{n:02}")),
            iri(P),
            plain(&format!("{n:02}")),
        );
    }
    d
}

#[test]
fn parse_offset_before_limit_keeps_both() {
    // The regression: OFFSET written first must NOT swallow the LIMIT.
    let q = parse("SELECT ?s WHERE { ?s ?p ?o } OFFSET 10 LIMIT 5");
    assert_eq!(
        q.limit,
        Some(5),
        "LIMIT must survive when OFFSET precedes it"
    );
    assert_eq!(q.offset, Some(10), "OFFSET must be captured");
}

#[test]
fn parse_limit_before_offset_keeps_both() {
    let q = parse("SELECT ?s WHERE { ?s ?p ?o } LIMIT 5 OFFSET 10");
    assert_eq!(q.limit, Some(5));
    assert_eq!(q.offset, Some(10));
}

#[test]
fn parse_limit_only_and_offset_only() {
    let only_limit = parse("SELECT ?s WHERE { ?s ?p ?o } LIMIT 7");
    assert_eq!(only_limit.limit, Some(7));
    assert_eq!(only_limit.offset, None);

    let only_offset = parse("SELECT ?s WHERE { ?s ?p ?o } OFFSET 3");
    assert_eq!(only_offset.limit, None);
    assert_eq!(only_offset.offset, Some(3));
}

#[test]
fn exec_offset_before_limit_equals_limit_before_offset() {
    let d = twenty_row_dataset();

    let a = run_like_server(
        &parse("SELECT ?s WHERE { ?s <http://example.org/p> ?o } ORDER BY ?s OFFSET 10 LIMIT 5"),
        &d,
    );
    let b = run_like_server(
        &parse("SELECT ?s WHERE { ?s <http://example.org/p> ?o } ORDER BY ?s LIMIT 5 OFFSET 10"),
        &d,
    );

    // LIMIT honoured in BOTH orders: exactly 5 rows (before the fix, the
    // OFFSET-first form returned all 10 rows past the offset).
    assert_eq!(a.len(), 5, "OFFSET-first must still apply LIMIT 5");
    assert_eq!(b.len(), 5, "LIMIT-first control must apply LIMIT 5");

    // Both orders are semantically identical: same 5 subjects, same order.
    assert_eq!(
        column(&a, "s"),
        column(&b, "s"),
        "OFFSET n LIMIT m and LIMIT m OFFSET n must select the same rows"
    );

    // And the window is the expected s10..s14 (proves OFFSET applied, not ignored).
    let expected: Vec<Term> = (10..15)
        .map(|n| iri(&format!("http://example.org/s{n:02}")))
        .collect();
    assert_eq!(column(&a, "s"), expected, "expected the s10..s14 window");
}

// ── Defect 2: FILTER scopes the whole UNION group ────────────────────────────

const PREF: &str = "http://example.org/pref";
const ALT: &str = "http://example.org/alt";

/// Branch A (pref) and branch B (alt) each contribute one "keep" row and one
/// "drop" row. A correct whole-group FILTER(?l = "keep") retains exactly the two
/// "keep" rows — one from EACH branch.
fn union_filter_dataset() -> PathDataset {
    let mut d = PathDataset::new();
    d.add(iri("http://example.org/s1"), iri(PREF), plain("keep")); // A, kept
    d.add(iri("http://example.org/s2"), iri(PREF), plain("drop")); // A, dropped
    d.add(iri("http://example.org/s3"), iri(ALT), plain("keep")); // B, kept
    d.add(iri("http://example.org/s4"), iri(ALT), plain("drop")); // B, dropped
    d
}

const UNION_FILTER_POSTFIX: &str = "SELECT ?c ?l WHERE { \
     { ?c <http://example.org/pref> ?l } UNION { ?c <http://example.org/alt> ?l } \
     FILTER(?l = \"keep\") } ORDER BY ?c";

const UNION_FILTER_PREFIX: &str = "SELECT ?c ?l WHERE { \
     FILTER(?l = \"keep\") \
     { ?c <http://example.org/pref> ?l } UNION { ?c <http://example.org/alt> ?l } } ORDER BY ?c";

/// The parsed WHERE must be `Filter(Union(_, _), _)` — the FILTER wraps the whole
/// union, never a single branch (`Union(Filter(A), B)`).
fn assert_filter_wraps_union(where_clause: &Algebra) {
    match where_clause {
        Algebra::Filter { pattern, .. } => assert!(
            matches!(**pattern, Algebra::Union { .. }),
            "FILTER must wrap the whole UNION; got Filter over {pattern:?}"
        ),
        other => panic!("expected Filter(Union(..), ..), got {other:?}"),
    }
}

#[test]
fn parse_filter_after_union_wraps_whole_group() {
    assert_filter_wraps_union(&parse(UNION_FILTER_POSTFIX).where_clause);
}

#[test]
fn parse_filter_before_union_wraps_whole_group() {
    assert_filter_wraps_union(&parse(UNION_FILTER_PREFIX).where_clause);
}

#[test]
fn exec_filter_applies_to_both_union_branches() {
    let d = union_filter_dataset();
    let sol = run_like_server(&parse(UNION_FILTER_POSTFIX), &d);

    // Exactly the two "keep" rows — one from each branch.
    assert_eq!(
        sol.len(),
        2,
        "whole-group FILTER over a UNION must keep exactly the two matching rows"
    );
    let subjects = column(&sol, "c");
    assert!(
        subjects.contains(&iri("http://example.org/s1")),
        "branch-A match (s1) must survive the filter"
    );
    assert!(
        subjects.contains(&iri("http://example.org/s3")),
        "branch-B match (s3) must survive the filter — the core regression"
    );
    // No "drop" row from EITHER branch leaked through.
    for l in column(&sol, "l") {
        assert_eq!(l, plain("keep"), "no dropped row may survive the filter");
    }
}

#[test]
fn exec_filter_before_and_after_union_are_equivalent() {
    let d = union_filter_dataset();
    let after = run_like_server(&parse(UNION_FILTER_POSTFIX), &d);
    let before = run_like_server(&parse(UNION_FILTER_PREFIX), &d);
    assert_eq!(
        column(&after, "c"),
        column(&before, "c"),
        "FILTER position within the group must not change the result"
    );
    assert_eq!(column(&after, "c").len(), 2);
}
