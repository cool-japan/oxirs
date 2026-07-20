//! Budget breaches raised *inside* a FILTER (NOT) EXISTS subquery must
//! propagate, not be swallowed into a boolean.
//!
//! `FILTER EXISTS`/`FILTER NOT EXISTS` evaluate an inner pattern per outer row.
//! The two `Err` arms of that evaluation used to collapse *any* failure to
//! `false` (EXISTS) / `true` (NOT EXISTS) — SPARQL's per-row "error ⇒ no match"
//! rule. But a runtime [`BudgetExceeded`] is a whole-query fault: collapsing it
//! yields a silent `200 OK` with a wrong (over-inclusive) result set instead of
//! a timeout. The fix detects the typed error in the EXISTS/NOT-EXISTS arms AND
//! in `apply_filter`'s own `Err` arm (which would otherwise re-swallow the
//! propagated error) and returns it.
//!
//! These are end-to-end `Algebra::Filter { Exists(..) }` executions through the
//! full executor — the path that actually goes through `apply_filter`. A direct
//! `evaluate_exists_subquery` call would bypass `apply_filter` and pass with only
//! half the fix, so the query-level test is the discriminating one.

use oxirs_arq::algebra::{Algebra, Expression, Term, TriplePattern, Variable};
use oxirs_arq::executor::{ExecutionStrategy, InMemoryDataset, QueryExecutor};
use oxirs_arq::query_governor::{BudgetExceeded, ExecutionBudget, ResourceBudget};
use oxirs_core::model::NamedNode;

fn var(name: &str) -> Term {
    Term::Variable(Variable::new_unchecked(name))
}

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

/// `outer_p` triples on `<urn:p>` (the outer pattern) and `inner_q` triples on
/// `<urn:q>` (scanned by the EXISTS subquery). Disjoint subjects so the EXISTS
/// inner pattern (`?other <urn:q> ?z`, `?other` unbound by the outer row) does an
/// unconstrained full scan of all `inner_q` triples.
fn exists_dataset(outer_p: usize, inner_q: usize) -> InMemoryDataset {
    let mut triples = Vec::with_capacity(outer_p + inner_q);
    for i in 0..outer_p {
        triples.push((
            iri(&format!("urn:p{i}")),
            iri("urn:p"),
            iri(&format!("urn:pv{i}")),
        ));
    }
    for i in 0..inner_q {
        triples.push((
            iri(&format!("urn:q{i}")),
            iri("urn:q"),
            iri(&format!("urn:qv{i}")),
        ));
    }
    InMemoryDataset::from_triples(triples)
}

fn outer_bgp() -> Box<Algebra> {
    Box::new(Algebra::Bgp(vec![TriplePattern::new(
        var("s"),
        iri("urn:p"),
        var("o"),
    )]))
}

/// EXISTS/NOT-EXISTS over `{ ?other <urn:q> ?z }` — `?other` is NOT bound by the
/// outer row, so the substituted subquery is an unconstrained scan of `<urn:q>`.
fn inner_scan_pattern() -> Box<Algebra> {
    Box::new(Algebra::Bgp(vec![TriplePattern::new(
        var("other"),
        iri("urn:q"),
        var("z"),
    )]))
}

/// Attach a triple-scan budget and execute on the Serial strategy.
fn run_with_scan_budget(
    algebra: &Algebra,
    ds: &InMemoryDataset,
    max_scan: u64,
) -> anyhow::Result<()> {
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: None,
        max_triples_scanned: Some(max_scan),
    });
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let mut executor = executor.with_budget(budget);
    executor.execute(algebra, ds).map(|_| ())
}

/// A `FILTER EXISTS` whose inner scan blows the triple-scan budget must abort the
/// whole query with the typed `BudgetExceeded`, NOT collapse EXISTS to `false`
/// and return a (wrong) `200`.
#[test]
fn filter_exists_propagates_budget_breach_not_false() {
    // 2 outer rows (2 scanned), inner EXISTS scans 500 `<urn:q>` triples → 502
    // total, over the 100 scan budget. The breach fires inside the first EXISTS.
    let ds = exists_dataset(2, 500);
    let algebra = Algebra::Filter {
        pattern: outer_bgp(),
        condition: Expression::Exists(inner_scan_pattern()),
    };
    let err = run_with_scan_budget(&algebra, &ds, 100)
        .expect_err("EXISTS inner scan over budget must abort the query, not collapse to false");
    let budget_err = err
        .downcast_ref::<BudgetExceeded>()
        .expect("the abort must carry the typed BudgetExceeded (a swallowed EXISTS would be Ok)");
    assert!(
        matches!(budget_err, BudgetExceeded::TriplesScannedExceeded { .. }),
        "scan breach must be TriplesScannedExceeded, got: {budget_err:?}"
    );
}

/// Same for `FILTER NOT EXISTS`: a budget breach in the inner scan must propagate
/// rather than collapse NOT EXISTS to `true`.
#[test]
fn filter_not_exists_propagates_budget_breach_not_true() {
    let ds = exists_dataset(2, 500);
    let algebra = Algebra::Filter {
        pattern: outer_bgp(),
        condition: Expression::NotExists(inner_scan_pattern()),
    };
    let err = run_with_scan_budget(&algebra, &ds, 100)
        .expect_err("NOT EXISTS inner scan over budget must abort the query, not collapse to true");
    let budget_err = err
        .downcast_ref::<BudgetExceeded>()
        .expect("the abort must carry the typed BudgetExceeded");
    assert!(
        matches!(budget_err, BudgetExceeded::TriplesScannedExceeded { .. }),
        "scan breach must be TriplesScannedExceeded, got: {budget_err:?}"
    );
}

/// Control: with an ample budget the identical `FILTER EXISTS` completes. There
/// ARE `<urn:q>` triples, so EXISTS is true for every outer row and all outer
/// rows survive — proving the query is well-formed and the aborts above are the
/// budget, not a malformed query.
#[test]
fn filter_exists_within_budget_keeps_matching_rows() {
    let ds = exists_dataset(2, 500);
    let algebra = Algebra::Filter {
        pattern: outer_bgp(),
        condition: Expression::Exists(inner_scan_pattern()),
    };
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: None,
        max_triples_scanned: Some(1_000_000),
    });
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let mut executor = executor.with_budget(budget);
    let (solution, _stats) = executor
        .execute(&algebra, &ds)
        .expect("ample budget: FILTER EXISTS must complete");
    assert_eq!(
        solution.len(),
        2,
        "EXISTS is true for both outer rows, so both must survive the filter"
    );
}

/// Companion unit fix (`query_governor::record_result_row`): the row checkpoint
/// must surface a triple-scan breach too, symmetric with `record_triple_scan`'s
/// row cross-check. Here a scan overshoots the limit (its own `Err` ignored by a
/// hypothetical caller); the very next row emission must still report the breach.
#[test]
fn record_result_row_surfaces_triple_scan_breach() {
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: None,
        max_result_rows: None,
        max_triples_scanned: Some(10),
    });
    // Overshoot the scan limit; the counter reaches 20 even though this returns Err.
    let scan = budget.record_triple_scan(20);
    assert!(
        matches!(scan, Err(BudgetExceeded::TriplesScannedExceeded { .. })),
        "record_triple_scan(20) over a 10 limit must itself breach"
    );
    // A caller that recorded a row next must ALSO be told the scan limit is blown.
    let row = budget.record_result_row();
    assert!(
        matches!(
            row,
            Err(BudgetExceeded::TriplesScannedExceeded {
                scanned: 20,
                limit: 10
            })
        ),
        "record_result_row must cross-check max_triples_scanned, got: {row:?}"
    );
}
