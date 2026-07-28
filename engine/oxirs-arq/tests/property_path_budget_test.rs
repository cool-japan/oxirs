//! Wall-time budget enforcement *inside* property-path evaluation.
//!
//! A transitive path (`?s <p>* ?o`, `?s <p>+ ?o`, `?s <p>* <o>`) enumerates
//! candidate nodes (`subjects() ∪ objects()`) and runs a BFS from each — an
//! `O(V·(V+E))` blow-up. Before the fix the traversal was a free function with
//! no access to the [`ExecutionBudget`], so the wall-time limit was never
//! consulted *during* the traversal.
//!
//! # Why these tests are discriminating (in-loop vs the post-loop backstop)
//!
//! `QueryExecutor::execute` also runs a *post-execution* row-accounting loop
//! (`record_result_row`, which itself calls `check_time`). For a traversal that
//! terminates, that backstop would eventually raise `BudgetExceeded` too — but
//! only **after the whole traversal has run to completion**. So "did we get a
//! `BudgetExceeded`?" alone does NOT prove the in-path-loop check fired.
//!
//! The discriminator is **time**. The un-budgeted evaluation of the shapes below
//! is very slow (measured on the dev box: `?s next* ?o` over a 2000-chain
//! completes in ~67 s; `?s next* <s0>` over a 10000-chain does not finish inside
//! 120 s). Each test attaches a 100 ms budget and asserts the abort returns in
//! **well under 3 s**. The only way to abort a ≥60 s traversal in <3 s is for the
//! budget check to fire *inside* the candidate / BFS loop — the post-loop
//! backstop could not, because it runs only after the traversal returns. The
//! budgeted runs also stay memory-light precisely because they abort early,
//! before the multi-million-row solution is materialized.

use oxirs_arq::algebra::{Algebra, PropertyPath, Term, Variable};
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

/// A single linked chain `s0 -next-> s1 -next-> … -> s{n}` on `<urn:next>`.
fn chain_dataset(n: usize) -> InMemoryDataset {
    let mut triples = Vec::with_capacity(n);
    for i in 0..n {
        triples.push((
            iri(&format!("urn:s{i}")),
            iri("urn:next"),
            iri(&format!("urn:s{}", i + 1)),
        ));
    }
    InMemoryDataset::from_triples(triples)
}

fn next_path() -> PropertyPath {
    PropertyPath::Iri(NamedNode::new_unchecked("urn:next"))
}

/// Run `algebra` on the Serial strategy (the strategy fuseki forces) with a
/// wall-time budget attached.
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

/// Assert `run` aborted with a typed wall-time `BudgetExceeded` in under `bound`.
/// The tight time bound is what proves the abort happened *inside* the traversal
/// loop rather than via the post-execution backstop (see module docs).
fn assert_mid_traversal_abort(err: anyhow::Error, elapsed: Duration, bound: Duration) {
    let budget_err = err
        .downcast_ref::<BudgetExceeded>()
        .expect("the abort must carry the typed BudgetExceeded (not a stringified error)");
    assert!(
        matches!(budget_err, BudgetExceeded::TimeoutExceeded { .. }),
        "wall-time breach must be TimeoutExceeded, got: {budget_err:?}"
    );
    assert!(
        elapsed < bound,
        "a 100 ms budget must abort the traversal in <{bound:?}, proving the in-loop check \
         fired (the un-budgeted evaluation of this shape takes tens of seconds; the \
         post-execution backstop could only abort after that completes). Took {elapsed:?}"
    );
}

/// `?s <urn:next>* ?o` — both endpoints variable (the finding's literal shape).
/// Un-budgeted over a 2000-chain this completes in ~67 s; a 100 ms budget must
/// abort it in well under 3 s.
#[test]
fn budget_aborts_two_variable_zero_or_more_path() {
    let ds = chain_dataset(2000);
    let algebra = Algebra::PropertyPath {
        subject: var("s"),
        path: PropertyPath::ZeroOrMore(Box::new(next_path())),
        object: var("o"),
    };
    let started = Instant::now();
    let err = run_budgeted(&algebra, &ds, Duration::from_millis(100))
        .expect_err("100 ms budget must abort the O(n^2) `<p>*` two-variable path");
    assert_mid_traversal_abort(err, started.elapsed(), Duration::from_secs(3));
}

/// The same guarantee on the `<p>+` (OneOrMore) BFS loop, a separate code path.
#[test]
fn budget_aborts_two_variable_one_or_more_path() {
    let ds = chain_dataset(2000);
    let algebra = Algebra::PropertyPath {
        subject: var("s"),
        path: PropertyPath::OneOrMore(Box::new(next_path())),
        object: var("o"),
    };
    let started = Instant::now();
    let err = run_budgeted(&algebra, &ds, Duration::from_millis(100))
        .expect_err("100 ms budget must abort the O(n^2) `<p>+` two-variable path");
    assert_mid_traversal_abort(err, started.elapsed(), Duration::from_secs(3));
}

/// `?s <urn:next>* <urn:s0>` — subject variable, concrete object. This drives the
/// same `O(V·(V+E))` candidate/BFS work but with an ~empty result set, so it is
/// the strongest (and most memory-light) runaway: over a 10000-chain the
/// un-budgeted evaluation does not finish within 120 s. A 100 ms budget must
/// still abort it in under 3 s — impossible without an in-loop check, since the
/// traversal never returns for the post-execution backstop to run.
#[test]
fn budget_aborts_subject_variable_star_path_before_completion() {
    let ds = chain_dataset(10_000);
    let algebra = Algebra::PropertyPath {
        subject: var("s"),
        path: PropertyPath::ZeroOrMore(Box::new(next_path())),
        object: iri("urn:s0"),
    };
    let started = Instant::now();
    let err = run_budgeted(&algebra, &ds, Duration::from_millis(100))
        .expect_err("100 ms budget must abort the >120 s subject-variable `<p>*` runaway");
    assert_mid_traversal_abort(err, started.elapsed(), Duration::from_secs(3));
}

/// Control: without a budget a small `<p>*` two-variable path completes and
/// produces the expected reflexive-transitive closure — proving the query is
/// well-formed and the aborts above are caused by the budget, not a malformed
/// query.
#[test]
fn zero_or_more_path_without_budget_completes() {
    let ds = chain_dataset(5);
    let algebra = Algebra::PropertyPath {
        subject: var("s"),
        path: PropertyPath::ZeroOrMore(Box::new(next_path())),
        object: var("o"),
    };
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let (solution, _stats) = executor
        .execute(&algebra, &ds)
        .expect("un-budgeted `<p>*` path must succeed");
    assert!(
        !solution.is_empty(),
        "reflexive-transitive `<next>*` over a non-empty chain must yield rows"
    );
}
