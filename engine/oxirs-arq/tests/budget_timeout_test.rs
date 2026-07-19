//! Wall-time budget enforcement inside the runaway O(N*M) evaluation loops.
//!
//! These tests pin down the P0 guarantee that a public SPARQL endpoint relies
//! on: a query whose evaluation blows up quadratically (a cross join, or a
//! MINUS between two high-cardinality disjoint patterns) must be aborted by the
//! attached [`ExecutionBudget`] *during* the hot loop, returning the typed
//! [`BudgetExceeded::TimeoutExceeded`] rather than running to completion (or
//! forever). The un-budgeted control proves the same query is otherwise valid
//! and that the abort is caused by the budget, not by a malformed query.

use oxirs_arq::algebra::{Algebra, Literal, Term, TriplePattern, Variable};
use oxirs_arq::executor::{ExecutionStrategy, InMemoryDataset, QueryExecutor};
use oxirs_arq::query_governor::{BudgetExceeded, ExecutionBudget, ResourceBudget};
use oxirs_core::model::NamedNode;
use std::time::{Duration, Instant};

fn var(name: &str) -> Term {
    Term::Variable(Variable::new_unchecked(name))
}

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn lit(s: &str) -> Term {
    Term::Literal(Literal::new(s.to_string(), None, None))
}

/// `n` triples on predicate `<urn:left>` and `n` on `<urn:right>`, using
/// disjoint subjects so `?a <urn:left> ?x` and `?b <urn:right> ?y` each match
/// exactly `n` rows and share no variables.
fn disjoint_dataset(n: usize) -> InMemoryDataset {
    let mut triples = Vec::with_capacity(n * 2);
    for i in 0..n {
        triples.push((
            iri(&format!("urn:s{i}")),
            iri("urn:left"),
            lit(&format!("l{i}")),
        ));
        triples.push((
            iri(&format!("urn:o{i}")),
            iri("urn:right"),
            lit(&format!("r{i}")),
        ));
    }
    InMemoryDataset::from_triples(triples)
}

fn left_bgp() -> Algebra {
    Algebra::Bgp(vec![TriplePattern::new(
        var("a"),
        iri("urn:left"),
        var("x"),
    )])
}

fn right_bgp() -> Algebra {
    Algebra::Bgp(vec![TriplePattern::new(
        var("b"),
        iri("urn:right"),
        var("y"),
    )])
}

/// Run `algebra` on the Serial strategy with a wall-time budget attached.
fn run_budgeted(algebra: &Algebra, ds: &InMemoryDataset, budget: Duration) -> anyhow::Result<()> {
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: Some(budget),
        max_result_rows: None,
        max_triples_scanned: None,
    });
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let mut executor = executor.with_budget(budget);
    executor.execute(algebra, ds).map(|_| ())
}

/// A cross MINUS of two disjoint high-cardinality patterns is O(|L|*|R|); with a
/// 100 ms wall-time budget the throttled `check_time` in `execute_minus`'s inner
/// loop must abort it long before it finishes (the un-budgeted run scans
/// 4000*4000 = 16M inner iterations, each allocating, i.e. seconds of work).
#[test]
fn budget_aborts_minus_cross_product() {
    let ds = disjoint_dataset(4000);
    let algebra = Algebra::Minus {
        left: Box::new(left_bgp()),
        right: Box::new(right_bgp()),
    };
    let started = Instant::now();
    let err = run_budgeted(&algebra, &ds, Duration::from_millis(100))
        .expect_err("100 ms budget must abort the 16M-iteration MINUS");
    let elapsed = started.elapsed();
    let budget_err = err
        .downcast_ref::<BudgetExceeded>()
        .expect("the abort must carry the typed BudgetExceeded (not a stringified error)");
    assert!(
        matches!(budget_err, BudgetExceeded::TimeoutExceeded { .. }),
        "wall-time breach must be TimeoutExceeded, got: {budget_err:?}"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "the query must abort promptly, not hang (took {elapsed:?})"
    );
}

/// The same guarantee on the JOIN / `hash_join` path: a disjoint cross join
/// collapses every build row into one hash bucket, so the inner probe loop runs
/// |L|*|R| times. The budget must abort it in `hash_join`'s inner loop.
#[test]
fn budget_aborts_hash_join_cross_product() {
    let ds = disjoint_dataset(4000);
    let algebra = Algebra::Join {
        left: Box::new(left_bgp()),
        right: Box::new(right_bgp()),
    };
    let started = Instant::now();
    let err = run_budgeted(&algebra, &ds, Duration::from_millis(100))
        .expect_err("100 ms budget must abort the 16M-row cross join");
    let elapsed = started.elapsed();
    let budget_err = err
        .downcast_ref::<BudgetExceeded>()
        .expect("the abort must carry the typed BudgetExceeded");
    assert!(
        matches!(budget_err, BudgetExceeded::TimeoutExceeded { .. }),
        "wall-time breach must be TimeoutExceeded, got: {budget_err:?}"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "the query must abort promptly, not hang (took {elapsed:?})"
    );
}

/// Control: without a budget the identical MINUS completes and, because the two
/// sides are variable-disjoint, MINUS removes nothing — every left row survives.
/// This proves the query is well-formed and that the aborts above are caused by
/// the budget, not by an execution error.
#[test]
fn minus_without_budget_completes_and_keeps_all_left_rows() {
    let n = 300;
    let ds = disjoint_dataset(n);
    let algebra = Algebra::Minus {
        left: Box::new(left_bgp()),
        right: Box::new(right_bgp()),
    };
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let (solution, _stats) = executor
        .execute(&algebra, &ds)
        .expect("un-budgeted disjoint MINUS must succeed");
    assert_eq!(
        solution.len(),
        n,
        "disjoint MINUS removes nothing: all {n} left rows must survive"
    );
}
