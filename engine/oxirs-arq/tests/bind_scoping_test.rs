//! End-to-end tests for group-graph-pattern scoping in the SPARQL parser.
//!
//! Regression coverage for the P1 correctness bug in which `BIND`
//! (`Algebra::Extend`) was deferred to the END of a group exactly like
//! `FILTER`. `FILTER` deferral is correct — a filter constrains the whole
//! group regardless of position — but `BIND` is POSITIONAL: it extends the
//! solution produced by the elements BEFORE it, and elements AFTER it join
//! against the extended solution. For `{ ?s ?p ?o . BIND(?o AS ?x) . ?x ?q ?r }`
//! the correct algebra is
//!   `Join(Extend(BGP(?s ?p ?o), ?x, ?o), BGP(?x ?q ?r))`,
//! whereas the buggy deferral produced
//!   `Extend(Join(BGP(?s ?p ?o), BGP(?x ?q ?r)), ?x, ?o)`,
//! cross-joining the BGPs first and silently mis-joining `?x`.
//!
//! Fixtures are hand-built rather than shared so each row count is fully
//! determined by the algebra shape under test.

mod common;

use common::MockDataset;
use oxirs_arq::algebra::Term;
use oxirs_arq::query::QueryParser;
use oxirs_arq::{Algebra, Expression, Literal, QueryExecutor, TriplePattern, Variable};
use oxirs_core::model::NamedNode;

// ── helpers ─────────────────────────────────────────────────────────────────

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s.to_string()))
}

fn var(name: &str) -> Variable {
    Variable::new(name).expect("valid variable name")
}

fn integer(value: &str) -> Term {
    Term::Literal(Literal {
        value: value.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#integer",
        )),
    })
}

fn bgp_all_vars(s: &str, p: &str, o: &str) -> Algebra {
    Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(var(s)),
        predicate: Term::Variable(var(p)),
        object: Term::Variable(var(o)),
    }])
}

fn parse_where(query: &str) -> Algebra {
    let mut parser = QueryParser::new();
    let q = parser.parse(query).expect("query must parse");
    q.where_clause
}

fn run(algebra: &Algebra, dataset: &MockDataset) -> usize {
    let mut executor = QueryExecutor::new();
    let (solution, _stats) = executor
        .execute(algebra, dataset)
        .expect("execution must succeed");
    solution.len()
}

/// Dataset (`ex:a ex:p ex:b` and `ex:b ex:p ex:c`).
///
/// Under CORRECT positional semantics for
/// `{ ?s ?p ?o . BIND(?o AS ?x) . ?x ?q ?r }`, row `(a,p,b)` binds `?x=b` and
/// `b` is the subject of `(b,p,c)` (one join result), while row `(b,p,c)` binds
/// `?x=c` and `c` is no subject (zero join results) — exactly one row overall.
///
/// Under the OLD cross-join-then-extend semantics the two 2-row BGPs share no
/// variables, so their join is the 2x2 = 4-row cartesian product, which the
/// trailing `Extend` leaves at 4 rows.
fn chain_dataset() -> MockDataset {
    let mut ds = MockDataset::new();
    ds.add_triple(
        iri("http://example.org/a"),
        iri("http://example.org/p"),
        iri("http://example.org/b"),
    );
    ds.add_triple(
        iri("http://example.org/b"),
        iri("http://example.org/p"),
        iri("http://example.org/c"),
    );
    ds
}

// ── (i) the exact P1 shape, end-to-end ──────────────────────────────────────

#[test]
fn bind_positional_p1_end_to_end() {
    let dataset = chain_dataset();
    let where_clause = parse_where("SELECT * WHERE { ?s ?p ?o . BIND(?o AS ?x) . ?x ?q ?r }");

    // Correct positional semantics: exactly one row.
    assert_eq!(
        run(&where_clause, &dataset),
        1,
        "positional BIND must chain ?x=?o into the following pattern (1 row)"
    );

    // The OLD buggy algebra shape (cross-join then extend) over the same data
    // yields a different count, proving the fix is load-bearing end-to-end.
    let buggy = Algebra::Extend {
        pattern: Box::new(Algebra::join(
            bgp_all_vars("s", "p", "o"),
            bgp_all_vars("x", "q", "r"),
        )),
        variable: var("x"),
        expr: Expression::Variable(var("o")),
    };
    assert_eq!(
        run(&buggy, &dataset),
        4,
        "the old cross-join-then-extend shape would have returned 4 rows"
    );
}

// ── (iv) algebra shape of the positional Extend ─────────────────────────────

#[test]
fn bind_positional_algebra_shape() {
    let where_clause = parse_where("SELECT * WHERE { ?s ?p ?o . BIND(?o AS ?x) . ?x ?q ?r }");
    match &where_clause {
        Algebra::Join { left, right } => {
            match left.as_ref() {
                Algebra::Extend {
                    pattern,
                    variable,
                    expr,
                } => {
                    assert!(
                        matches!(pattern.as_ref(), Algebra::Bgp(_)),
                        "Extend must wrap the FIRST BGP, got {pattern:?}"
                    );
                    assert_eq!(variable, &var("x"), "extended variable must be ?x");
                    assert_eq!(
                        expr,
                        &Expression::Variable(var("o")),
                        "extend expression must be ?o"
                    );
                }
                other => panic!("Join.left must be Extend(BGP, ?x, ?o), got {other:?}"),
            }
            assert!(
                matches!(right.as_ref(), Algebra::Bgp(_)),
                "Join.right must be the trailing BGP, got {right:?}"
            );
        }
        other => panic!("expected Join(Extend(BGP, ?x, ?o), BGP), got {other:?}"),
    }
}

// ── (ii) BIND first in group still works (e867401d regression) ──────────────

#[test]
fn bind_first_in_group_extends_unit_table() {
    // Bare BIND over an otherwise empty group must extend the unit table
    // (join identity → one row), not produce zero rows.
    let where_clause = parse_where("SELECT * WHERE { BIND(42 AS ?c) }");
    match &where_clause {
        Algebra::Extend { pattern, .. } => assert!(
            matches!(pattern.as_ref(), Algebra::Table),
            "leading BIND must extend the unit Table, got {pattern:?}"
        ),
        other => panic!("expected Extend(Table, ?c, 42), got {other:?}"),
    }
    // Any dataset works; the unit table yields exactly one row.
    assert_eq!(
        run(&where_clause, &chain_dataset()),
        1,
        "leading BIND over the unit table yields exactly one row"
    );
}

#[test]
fn bind_first_then_pattern_joins_forward() {
    // `{ BIND(42 AS ?c) ?s ?p ?o }` → Join(Extend(Table, ?c, 42), BGP).
    let where_clause = parse_where("SELECT * WHERE { BIND(42 AS ?c) ?s ?p ?o }");
    match &where_clause {
        Algebra::Join { left, right } => {
            match left.as_ref() {
                Algebra::Extend { pattern, .. } => assert!(
                    matches!(pattern.as_ref(), Algebra::Table),
                    "leading BIND must extend the unit Table, got {pattern:?}"
                ),
                other => panic!("Join.left must be Extend(Table, …), got {other:?}"),
            }
            assert!(matches!(right.as_ref(), Algebra::Bgp(_)));
        }
        other => panic!("expected Join(Extend(Table, ?c, 42), BGP), got {other:?}"),
    }
    // Two triples, each extended with ?c=42 → 2 rows.
    assert_eq!(run(&where_clause, &chain_dataset()), 2);
}

// ── (iii) BIND at group end (equivalent under both semantics) ───────────────

#[test]
fn bind_at_group_end_wraps_group() {
    // `{ ?s ?p ?o BIND(?o AS ?x) }` → Extend(BGP, ?x, ?o); with a trailing BIND
    // (nothing joins after it) the positional and deferred shapes coincide.
    let where_clause = parse_where("SELECT * WHERE { ?s ?p ?o BIND(?o AS ?x) }");
    match &where_clause {
        Algebra::Extend { pattern, .. } => assert!(
            matches!(pattern.as_ref(), Algebra::Bgp(_)),
            "trailing BIND must wrap the group's BGP, got {pattern:?}"
        ),
        other => panic!("expected Extend(BGP, ?x, ?o), got {other:?}"),
    }
    // Two triples, each extended with ?x=?o → 2 rows.
    assert_eq!(run(&where_clause, &chain_dataset()), 2);
}

// ── plain SELECT * execution (audit found zero such tests) ───────────────────

#[test]
fn select_star_executes_over_bgp() {
    let mut parser = QueryParser::new();
    let q = parser
        .parse("SELECT * WHERE { ?s ?p ?o }")
        .expect("SELECT * must parse");
    assert!(
        q.select_variables.is_empty(),
        "SELECT * leaves select_variables empty (project all in-scope)"
    );
    assert!(matches!(q.where_clause, Algebra::Bgp(_)));
    assert_eq!(
        run(&q.where_clause, &chain_dataset()),
        2,
        "SELECT * over a 2-triple store returns both rows"
    );
}

// ── group-scoped FILTER still works (e867401d regression) ────────────────────

#[test]
fn filter_stays_group_scoped() {
    // `{ ?person ?p ?age FILTER(?age > 25) }` → Filter(BGP, ?age > 25).
    //
    // A variable predicate keeps the BGP inside the MockDataset matching
    // surface: a bare IRI predicate parses to a `Term::PropertyPath` the mock
    // harness does not resolve, which is orthogonal to FILTER scoping.
    let mut ds = MockDataset::new();
    ds.add_triple(
        iri("http://example.org/alice"),
        iri("http://example.org/age"),
        integer("30"),
    );
    ds.add_triple(
        iri("http://example.org/bob"),
        iri("http://example.org/age"),
        integer("25"),
    );
    let where_clause = parse_where("SELECT * WHERE { ?person ?p ?age FILTER(?age > 25) }");
    match &where_clause {
        Algebra::Filter { pattern, .. } => assert!(
            matches!(pattern.as_ref(), Algebra::Bgp(_)),
            "FILTER must wrap the group's BGP, got {pattern:?}"
        ),
        other => panic!("expected Filter(BGP, ?age > 25), got {other:?}"),
    }
    assert_eq!(
        run(&where_clause, &ds),
        1,
        "FILTER(?age > 25) keeps only Alice (age 30)"
    );
}

// ── P3(a): a trailing FILTER after a top-level UNION is scoped to the group ──

#[test]
fn filter_after_union_is_group_scoped() {
    // Previously the `has_union_pattern` fast path returned early and left the
    // trailing FILTER unconsumed (a parse error). It must now wrap the union.
    let where_clause =
        parse_where("SELECT * WHERE { { ?s ?p ?o } UNION { ?s ?p ?o } FILTER(?o = ?o) }");
    match &where_clause {
        Algebra::Filter { pattern, .. } => assert!(
            matches!(pattern.as_ref(), Algebra::Union { .. }),
            "trailing FILTER must wrap the whole UNION, got {pattern:?}"
        ),
        other => panic!("expected Filter(Union(...), ?o = ?o), got {other:?}"),
    }
}

// ── P3(b): a lone FILTER in a nested group is NOT re-scoped to the outer group

#[test]
fn nested_filter_not_hoisted_to_outer_group() {
    // `{ ?s ?p ?o . { FILTER(?o = ?o) } }` must parse to
    // `Join(BGP, Filter(Table, ?o = ?o))` — the nested `Filter { pattern: Table }`
    // is a JOIN operand, never mistaken for an outer-group modifier.
    let where_clause = parse_where("SELECT * WHERE { ?s ?p ?o . { FILTER(?o = ?o) } }");
    match &where_clause {
        Algebra::Join { left, right } => {
            assert!(
                matches!(left.as_ref(), Algebra::Bgp(_)),
                "Join.left must be the outer BGP, got {left:?}"
            );
            match right.as_ref() {
                Algebra::Filter { pattern, .. } => assert!(
                    matches!(pattern.as_ref(), Algebra::Table),
                    "nested FILTER must stay over its own (Table) scope, got {pattern:?}"
                ),
                other => panic!("Join.right must be Filter(Table, …), got {other:?}"),
            }
        }
        other => panic!("expected Join(BGP, Filter(Table, …)), got {other:?}"),
    }
}
