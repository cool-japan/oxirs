//! End-to-end SPARQL 1.1 §8.3.2 MINUS semantics through the public executor.
//!
//! R8-3b hashes the right side of a high-cardinality shared-variable MINUS into
//! an `O(|L|+|R|)` anti-join, keeping the variable-disjoint and partial-binding
//! cases on the exact nested fallback. These tests pin the observable semantics
//! (not the strategy): left/right solutions are injected verbatim with
//! `Algebra::Values`, and `MINUS` is evaluated on the Serial strategy — the same
//! path the fuseki endpoint uses (`arq_exec::dispatch_with_budget`).

use oxirs_arq::algebra::{Algebra, Binding, Literal, Term, Variable};
use oxirs_arq::executor::{ExecutionStrategy, InMemoryDataset, QueryExecutor};

fn var(name: &str) -> Variable {
    Variable::new_unchecked(name)
}

fn lit(s: &str) -> Term {
    Term::Literal(Literal::new(s.to_string(), None, None))
}

/// Build a solution row from `(name, term)` pairs. Omitted variables are unbound.
fn row(pairs: &[(&str, Term)]) -> Binding {
    let mut b = Binding::new();
    for (name, term) in pairs {
        b.insert(var(name), term.clone());
    }
    b
}

/// `Algebra::Values` returns its bindings verbatim as the solution, so it is the
/// cleanest way to feed exact left/right solutions into MINUS.
fn values(rows: Vec<Binding>) -> Algebra {
    let mut vars: Vec<Variable> = Vec::new();
    for b in &rows {
        for k in b.keys() {
            if !vars.contains(k) {
                vars.push(k.clone());
            }
        }
    }
    Algebra::Values {
        variables: vars,
        bindings: rows,
    }
}

/// Evaluate `left MINUS right` on the Serial strategy and return the rows,
/// normalised to a sorted multiset of `(var,value)` tuples for order-independent
/// comparison.
fn run_minus(left: Vec<Binding>, right: Vec<Binding>) -> Vec<Vec<(String, String)>> {
    let algebra = Algebra::Minus {
        left: Box::new(values(left)),
        right: Box::new(values(right)),
    };
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let ds = InMemoryDataset::from_triples(vec![]);
    let (solution, _stats) = executor
        .execute(&algebra, &ds)
        .expect("MINUS evaluation must succeed");
    normalise(&solution)
}

fn normalise(solution: &[Binding]) -> Vec<Vec<(String, String)>> {
    let mut rows: Vec<Vec<(String, String)>> = solution
        .iter()
        .map(|b| {
            let mut kv: Vec<(String, String)> = b
                .iter()
                .map(|(k, v)| (k.to_string(), format!("{v:?}")))
                .collect();
            kv.sort();
            kv
        })
        .collect();
    rows.sort();
    rows
}

fn expect_rows(rows: Vec<Binding>) -> Vec<Vec<(String, String)>> {
    normalise(&rows)
}

// ── shared variable present ────────────────────────────────────────────────

#[test]
fn shared_var_match_excludes_left_row() {
    // { ?s ?p } MINUS { ?s ?q }: a left row whose ?s appears on the right is removed.
    let left = vec![row(&[("s", lit("x"))]), row(&[("s", lit("y"))])];
    let right = vec![row(&[("s", lit("x"))])];
    let got = run_minus(left, right);
    assert_eq!(got, expect_rows(vec![row(&[("s", lit("y"))])]));
}

#[test]
fn shared_var_no_match_keeps_all_left_rows() {
    let left = vec![row(&[("s", lit("x"))]), row(&[("s", lit("y"))])];
    let right = vec![row(&[("s", lit("z"))])];
    let got = run_minus(left.clone(), right);
    assert_eq!(got, expect_rows(left));
}

// ── variable-disjoint MINUS is a no-op (nested fallback) ───────────────────

#[test]
fn disjoint_variables_remove_nothing() {
    // ?a shares no variable with ?b: MINUS removes nothing regardless of values.
    let left = vec![row(&[("a", lit("1"))]), row(&[("a", lit("2"))])];
    let right = vec![row(&[("b", lit("1"))]), row(&[("b", lit("2"))])];
    let got = run_minus(left.clone(), right);
    assert_eq!(got, expect_rows(left));
}

// ── multiple shared variables (all must agree) ─────────────────────────────

#[test]
fn multiple_shared_vars_require_full_agreement() {
    // Shared {a,b}. Only a row matching on BOTH is removed.
    let left = vec![
        row(&[("a", lit("1")), ("b", lit("1"))]), // removed (matches r1)
        row(&[("a", lit("1")), ("b", lit("2"))]), // kept (b differs)
        row(&[("a", lit("2")), ("b", lit("2"))]), // removed (matches r2)
    ];
    let right = vec![
        row(&[("a", lit("1")), ("b", lit("1"))]),
        row(&[("a", lit("2")), ("b", lit("2"))]),
    ];
    let got = run_minus(left, right);
    assert_eq!(
        got,
        expect_rows(vec![row(&[("a", lit("1")), ("b", lit("2"))])])
    );
}

#[test]
fn partial_shared_key_does_not_remove() {
    // Shared {a,b}, all rows homogeneous (hash path). A right row that agrees on
    // a but not b must NOT remove the left row.
    let left = vec![row(&[("a", lit("1")), ("b", lit("9"))])];
    let right = vec![row(&[("a", lit("1")), ("b", lit("8"))])];
    let got = run_minus(left.clone(), right);
    assert_eq!(got, expect_rows(left));
}

// ── partial bindings (heterogeneous -> nested fallback) ────────────────────

#[test]
fn partial_binding_shares_subset_still_removes() {
    // S = {a,b}. Left row L2 leaves b unbound; it shares only {a} with the right
    // row, is compatible there, so SPARQL removes it. (Exact-key hash would keep
    // it — which is why heterogeneous inputs use the nested fallback.)
    let left = vec![
        row(&[("a", lit("1")), ("b", lit("2"))]), // removed
        row(&[("a", lit("1"))]),                  // b unbound -> shares {a} -> removed
        row(&[("a", lit("7"))]),                  // no match -> kept
    ];
    let right = vec![row(&[("a", lit("1")), ("b", lit("2"))])];
    let got = run_minus(left, right);
    assert_eq!(got, expect_rows(vec![row(&[("a", lit("7"))])]));
}

#[test]
fn left_row_with_no_shared_bound_var_is_kept() {
    // S = {a}. A left row that leaves a unbound shares nothing with any right row
    // -> disjoint per-pair -> never removed.
    let left = vec![
        row(&[("a", lit("1"))]), // removed
        row(&[("c", lit("1"))]), // a unbound, only c -> disjoint -> kept
    ];
    let right = vec![row(&[("a", lit("1"))])];
    let got = run_minus(left, right);
    assert_eq!(got, expect_rows(vec![row(&[("c", lit("1"))])]));
}

// ── empty sides ────────────────────────────────────────────────────────────

#[test]
fn empty_right_returns_left_unchanged() {
    let left = vec![row(&[("s", lit("x"))]), row(&[("s", lit("y"))])];
    let got = run_minus(left.clone(), vec![]);
    assert_eq!(got, expect_rows(left));
}

#[test]
fn empty_left_returns_empty() {
    let right = vec![row(&[("s", lit("x"))])];
    let got = run_minus(vec![], right);
    assert!(got.is_empty());
}

// ── duplicate solutions preserved ──────────────────────────────────────────

#[test]
fn duplicate_left_rows_are_each_decided_independently() {
    // Two identical kept rows must both survive; two identical removed rows both go.
    let left = vec![
        row(&[("s", lit("keep"))]),
        row(&[("s", lit("keep"))]),
        row(&[("s", lit("drop"))]),
        row(&[("s", lit("drop"))]),
    ];
    let right = vec![row(&[("s", lit("drop"))])];
    let got = run_minus(left, right);
    assert_eq!(
        got,
        expect_rows(vec![row(&[("s", lit("keep"))]), row(&[("s", lit("keep"))])])
    );
}

#[test]
fn duplicate_right_rows_do_not_change_outcome() {
    let left = vec![row(&[("s", lit("x"))]), row(&[("s", lit("y"))])];
    let right = vec![row(&[("s", lit("x"))]), row(&[("s", lit("x"))])];
    let got = run_minus(left, right);
    assert_eq!(got, expect_rows(vec![row(&[("s", lit("y"))])]));
}

// ── larger homogeneous case exercising the hash path ───────────────────────

#[test]
fn large_homogeneous_shared_var_minus_is_correct() {
    // 2000 left rows on ?s, right removes the even-numbered ?s. This is the shape
    // the hash anti-join accelerates; assert the exact surviving set.
    let left: Vec<Binding> = (0..2000)
        .map(|i| row(&[("s", lit(&format!("s{i}")))]))
        .collect();
    let right: Vec<Binding> = (0..2000)
        .filter(|i| i % 2 == 0)
        .map(|i| row(&[("s", lit(&format!("s{i}")))]))
        .collect();
    let got = run_minus(left, right);
    let expected: Vec<Binding> = (0..2000)
        .filter(|i| i % 2 == 1)
        .map(|i| row(&[("s", lit(&format!("s{i}")))]))
        .collect();
    assert_eq!(got, expect_rows(expected));
}
