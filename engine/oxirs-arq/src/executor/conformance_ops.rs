//! Conformance Operations
//!
//! Implementations for SPARQL algebra operators needed for SPARQL 1.1 conformance:
//! - MINUS (set difference)
//! - EXTEND (BIND)
//! - VALUES (inline data)
//! - Property paths (evaluation via path module)

use crate::algebra::{Algebra, Binding, Solution, Term, Variable};
use anyhow::Result;
use std::collections::{HashSet, VecDeque};

use super::dataset::{convert_property_path, Dataset, DatasetPathAdapter};
use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Execute a MINUS pattern
    ///
    /// SPARQL 1.1 §8.3.2: a left solution μ_l is removed iff there is a right
    /// solution μ_r that is *compatible* with it (every variable bound in **both**
    /// agrees) **and** shares at least one bound variable. When μ_l and μ_r have
    /// no shared bound variable the pair contributes nothing — a variable-disjoint
    /// MINUS removes nothing (a semantic no-op).
    ///
    /// # Evaluation strategy
    ///
    /// Let `S` = (variables bound anywhere on the left) ∩ (variables bound
    /// anywhere on the right). For any left/right pair, `keys(μ_l) ∩ keys(μ_r) ⊆
    /// S`, so `S` bounds which variables can ever participate.
    ///
    /// * **`S` non-empty and every solution binds all of `S` (homogeneous):**
    ///   the per-pair shared set is exactly `S` for every pair, so "compatible on
    ///   the shared vars" collapses to *equality of the `S`-value tuple*. We hash
    ///   the right side's `S`-tuples once and probe the left — an `O(|L|+|R|)`
    ///   anti-join replacing the `O(|L|*|R|)` nested scan that made a
    ///   high-cardinality shared-variable MINUS run for minutes. See
    ///   [`Self::execute_minus_hash`].
    /// * **`S` empty (variable-disjoint) or heterogeneous (some solution leaves
    ///   an `S`-variable unbound):** fall back to the historical nested scan
    ///   ([`Self::execute_minus_nested`]). This keeps two properties intact:
    ///   (a) the exact SPARQL compatibility semantics when partial bindings make
    ///   the per-pair shared set vary (an exact-`S`-key hash would wrongly miss a
    ///   pair that shares only a *subset* of `S`); and (b) the runaway
    ///   disjoint-MINUS remains a real `O(|L|*|R|)` loop that the attached
    ///   wall-time budget aborts at its deadline. The homogeneous hash path and
    ///   this fallback produce byte-for-byte identical result sets on every input.
    pub fn execute_minus(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let left_solutions = self.execute_serial(left, dataset)?;
        let right_solutions = self.execute_serial(right, dataset)?;

        if right_solutions.is_empty() {
            return Ok(left_solutions);
        }

        // S = the variables bound on both sides. `keys(μ_l) ∩ keys(μ_r) ⊆ S`
        // for every pair, so nothing outside `S` can affect compatibility.
        let left_vars: HashSet<&Variable> = left_solutions.iter().flat_map(|b| b.keys()).collect();
        let right_vars: HashSet<&Variable> =
            right_solutions.iter().flat_map(|b| b.keys()).collect();
        let shared_vars: Vec<&Variable> = left_vars.intersection(&right_vars).copied().collect();

        // Homogeneous fast path: ≥1 shared variable and every solution on both
        // sides binds all of `S`. Only then does exact `S`-tuple equality equal
        // the nested-loop compatibility test (see the doc comment above).
        if !shared_vars.is_empty()
            && all_bind_shared(&left_solutions, &shared_vars)
            && all_bind_shared(&right_solutions, &shared_vars)
        {
            return self.execute_minus_hash(&left_solutions, &right_solutions, &shared_vars);
        }

        // Fallback: variable-disjoint (`S` empty) or partial bindings.
        self.execute_minus_nested(&left_solutions, &right_solutions)
    }

    /// Homogeneous hash anti-join for MINUS (see [`Self::execute_minus`]).
    ///
    /// Precondition (checked by the caller): `shared_vars` is non-empty and every
    /// left and right solution binds all of them. The right side's shared-variable
    /// value tuples are hashed once; a left row is removed iff its shared tuple is
    /// present (i.e. some right solution is compatible on the shared variables).
    /// Existence is all an anti-join needs, so a set of keys — not the bindings —
    /// is stored. Keys borrow from the solutions (both live for the call), so the
    /// table holds no cloned terms.
    fn execute_minus_hash(
        &self,
        left_solutions: &Solution,
        right_solutions: &Solution,
        shared_vars: &[&Variable],
    ) -> Result<Solution> {
        // A persistent (not per-row) counter drives the throttled wall-time
        // check across both the build and probe loops, mirroring
        // [`QueryExecutor::hash_join`]. `& 0x3FF` fires at 0 then every 1024
        // iterations; see [`QueryExecutor::budget_check_time`].
        let mut budget_ticks: u64 = 0;

        let mut right_keys: HashSet<Vec<(&Variable, &Term)>> = HashSet::new();
        for right_binding in right_solutions {
            if budget_ticks & 0x3FF == 0 {
                self.budget_check_time()?;
            }
            budget_ticks += 1;
            let key: Vec<(&Variable, &Term)> = shared_vars
                .iter()
                .filter_map(|&var| right_binding.get(var).map(|term| (var, term)))
                .collect();
            right_keys.insert(key);
        }

        let mut result = Solution::new();
        for left_binding in left_solutions {
            if budget_ticks & 0x3FF == 0 {
                self.budget_check_time()?;
            }
            budget_ticks += 1;
            let key: Vec<(&Variable, &Term)> = shared_vars
                .iter()
                .filter_map(|&var| left_binding.get(var).map(|term| (var, term)))
                .collect();
            // No compatible right solution -> the left row survives.
            if !right_keys.contains(&key) {
                result.push(left_binding.clone());
            }
        }
        Ok(result)
    }

    /// Nested-scan fallback for MINUS: the exact SPARQL 1.1 §8.3.2 semantics.
    ///
    /// Used for the variable-disjoint case (which removes nothing) and for
    /// heterogeneous partial bindings where the per-pair shared set varies and an
    /// exact-`S`-key hash would be unsound. It is `O(|L|*|R|)`; for a
    /// high-cardinality *disjoint* MINUS every inner iteration is cheap
    /// (shared-var set empty -> `continue`) but there are `|L|*|R|` of them — the
    /// runaway that the throttled wall-time check aborts at the budget deadline.
    fn execute_minus_nested(
        &self,
        left_solutions: &Solution,
        right_solutions: &Solution,
    ) -> Result<Solution> {
        let mut result = Solution::new();
        let mut budget_ticks: u64 = 0;
        for left_binding in left_solutions {
            let left_vars: HashSet<&Variable> = left_binding.keys().collect();

            let mut should_remove = false;
            for right_binding in right_solutions {
                if budget_ticks & 0x3FF == 0 {
                    self.budget_check_time()?;
                }
                budget_ticks += 1;
                let right_vars: HashSet<&Variable> = right_binding.keys().collect();
                let shared_vars: Vec<&&Variable> = left_vars.intersection(&right_vars).collect();

                if shared_vars.is_empty() {
                    // Disjoint variables: SPARQL spec says MINUS removes nothing
                    continue;
                }

                // Check compatibility on shared variables
                let mut compatible = true;
                for shared_var in &shared_vars {
                    let left_val = left_binding.get(**shared_var);
                    let right_val = right_binding.get(**shared_var);
                    if left_val != right_val {
                        compatible = false;
                        break;
                    }
                }
                if compatible {
                    should_remove = true;
                    break;
                }
            }

            if !should_remove {
                result.push(left_binding.clone());
            }
        }
        Ok(result)
    }

    /// Execute an EXTEND (BIND) pattern
    ///
    /// Extends each solution with a new variable binding computed by evaluating expr.
    pub fn execute_extend(
        &self,
        pattern: &Algebra,
        variable: &Variable,
        expr: &crate::algebra::Expression,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let pattern_solutions = self.execute_serial(pattern, dataset)?;
        let mut result = Solution::new();

        for mut binding in pattern_solutions {
            match self.evaluate_expression(expr, &binding) {
                Ok(value) => {
                    binding.insert(variable.clone(), value);
                }
                Err(_) => {
                    // In SPARQL, BIND with an error: the binding is still included
                    // but the variable is unbound (we just don't add it)
                }
            }
            result.push(binding);
        }
        Ok(result)
    }

    /// Execute a VALUES clause
    ///
    /// Returns the inline data as solutions.
    pub fn execute_values(
        &self,
        _variables: &[Variable],
        bindings: &[Binding],
    ) -> Result<Solution> {
        Ok(bindings.to_vec())
    }

    /// Execute a property path pattern
    ///
    /// Evaluates property paths against the dataset using BFS/DFS traversal.
    pub fn execute_property_path(
        &self,
        subject: &Term,
        path: &crate::algebra::PropertyPath,
        object: &Term,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let adapter = DatasetPathAdapter::new(dataset);
        let path_pp = convert_property_path(path)?;
        // Runtime wall-time budget (if any). A two-variable `<p>*`/`<p>+`
        // endpoint enumerates `subjects() ∪ objects()` and runs a full BFS from
        // every candidate — `O(V·(V+E))`. Both the candidate loop here and the
        // BFS `pop` loops inside `evaluate_path_from` therefore carry a throttled
        // `check_time` so the traversal aborts at the deadline instead of pinning
        // a CPU for minutes. Threading the `Arc` by reference keeps it clone-free.
        let budget = self.budget();

        let mut result = Solution::new();

        match (subject, object) {
            (Term::Variable(s_var), Term::Variable(o_var)) => {
                // Both variable: enumerate all subjects AND objects as potential starting nodes.
                // This is needed for paths like inverse (^pred) where the "subjects" in path
                // evaluation are the objects in the graph, and vice versa.
                let mut candidates: HashSet<Term> = HashSet::new();
                for s in dataset.subjects()? {
                    candidates.insert(s);
                }
                for o in dataset.objects()? {
                    candidates.insert(o);
                }
                let mut seen_pairs: HashSet<(Term, Term)> = HashSet::new();
                // Throttled wall-time check across the candidate fan-out: fires at
                // 0 then every 1024 candidates (see [`Self::budget_check_time`]).
                let mut budget_ticks: u64 = 0;
                for s in candidates {
                    if budget_ticks & 0x3FF == 0 {
                        self.budget_check_time()?;
                    }
                    budget_ticks += 1;
                    let reachable = evaluate_path_from(&path_pp, &s, &adapter, budget)?;
                    for o in reachable {
                        let pair = (s.clone(), o.clone());
                        if seen_pairs.insert(pair) {
                            let mut binding = Binding::new();
                            binding.insert(s_var.clone(), s.clone());
                            binding.insert(o_var.clone(), o);
                            result.push(binding);
                        }
                    }
                }
            }
            (Term::Variable(s_var), concrete_obj) => {
                // Subject variable, concrete object: find all subjects that can reach object
                // We enumerate all subjects AND all objects to handle terminal nodes
                // (nodes that appear only as objects, not as subjects)
                let mut candidates: HashSet<Term> = HashSet::new();
                for s in dataset.subjects()? {
                    candidates.insert(s);
                }
                // Also include all objects so terminal nodes (like leaves in hierarchy)
                // are considered as potential starting points
                for o in dataset.objects()? {
                    candidates.insert(o);
                }
                // Same throttled wall-time check as the two-variable arm above.
                let mut budget_ticks: u64 = 0;
                for s in candidates {
                    if budget_ticks & 0x3FF == 0 {
                        self.budget_check_time()?;
                    }
                    budget_ticks += 1;
                    let reachable = evaluate_path_from(&path_pp, &s, &adapter, budget)?;
                    if reachable.contains(concrete_obj) {
                        let mut binding = Binding::new();
                        binding.insert(s_var.clone(), s);
                        result.push(binding);
                    }
                }
            }
            (concrete_subj, Term::Variable(o_var)) => {
                // Concrete subject, object variable: find all reachable objects
                let reachable = evaluate_path_from(&path_pp, concrete_subj, &adapter, budget)?;
                for o in reachable {
                    let mut binding = Binding::new();
                    binding.insert(o_var.clone(), o);
                    result.push(binding);
                }
            }
            (concrete_subj, concrete_obj) => {
                // Both concrete: check if path exists
                let reachable = evaluate_path_from(&path_pp, concrete_subj, &adapter, budget)?;
                if reachable.contains(concrete_obj) {
                    result.push(Binding::new());
                }
            }
        }
        Ok(result)
    }
}

/// Whether every solution binds all of `shared_vars`.
///
/// The homogeneity precondition of [`QueryExecutor::execute_minus_hash`]: only
/// when it holds for both sides does the per-pair shared set equal the static
/// `S`, making an exact-`S`-tuple hash sound. `Binding` is a `HashMap`, so the
/// membership test is `O(1)` per variable.
fn all_bind_shared(solutions: &Solution, shared_vars: &[&Variable]) -> bool {
    solutions
        .iter()
        .all(|binding| shared_vars.iter().all(|&var| binding.contains_key(var)))
}

/// Throttled wall-clock budget check for property-path evaluation loops.
///
/// A no-op when no budget is attached. When a budget *is* attached it forwards
/// to [`crate::query_governor::ExecutionBudget::check_time`] and, on breach,
/// returns the **typed** [`crate::query_governor::BudgetExceeded`] wrapped via
/// [`anyhow::Error::new`] (not stringified) so a caller up the stack can
/// `downcast_ref` it and map a wall-time timeout to the correct HTTP status.
/// Callers invoke it *throttled* (`& 0x3FF`, i.e. once every 1024 iterations)
/// because the underlying `Instant::now()` is not free at `O(V·(V+E))` scale.
#[inline]
fn path_budget_check_time(
    budget: Option<&std::sync::Arc<crate::query_governor::ExecutionBudget>>,
) -> Result<()> {
    if let Some(b) = budget {
        b.check_time().map_err(anyhow::Error::new)?;
    }
    Ok(())
}

/// Evaluate a property path from a given subject, returning all reachable objects
///
/// `budget` is the optional runtime wall-time budget threaded down from
/// [`QueryExecutor::execute_property_path`]. Every unbounded traversal loop
/// (the two BFS `pop` loops for `*`/`+`, and the complex-inverse fan-out)
/// carries a throttled [`path_budget_check_time`] so a `<p>*` over a large graph
/// is aborted at the deadline rather than running to completion.
fn evaluate_path_from(
    path: &crate::path::PropertyPath,
    subject: &Term,
    adapter: &DatasetPathAdapter<'_>,
    budget: Option<&std::sync::Arc<crate::query_governor::ExecutionBudget>>,
) -> Result<HashSet<Term>> {
    use crate::path::PathDataset;
    use crate::path::PropertyPath as PP;

    let mut result = HashSet::new();

    match path {
        PP::Direct(pred) => {
            let objects = adapter.find_outgoing(subject, pred)?;
            result.extend(objects);
        }
        PP::Inverse(inner) => {
            // For inverse: find nodes x such that inner(x, subject)
            // We need to find all nodes from which subject is reachable via inner
            // For a direct inner path, this is efficient:
            if let PP::Direct(pred) = inner.as_ref() {
                let incoming = adapter.find_incoming(pred, subject)?;
                result.extend(incoming);
            } else {
                // For complex inner paths, we enumerate all predicates. The
                // per-candidate recursion below is the unbounded part, so it is
                // throttled on the wall-time budget.
                let all_preds = adapter.get_predicates()?;
                let mut budget_ticks: u64 = 0;
                for pred in &all_preds {
                    let candidates = adapter.find_incoming(pred, subject)?;
                    for candidate in candidates {
                        if budget_ticks & 0x3FF == 0 {
                            path_budget_check_time(budget)?;
                        }
                        budget_ticks += 1;
                        let forward = evaluate_path_from(inner, &candidate, adapter, budget)?;
                        if forward.contains(subject) {
                            result.insert(candidate);
                        }
                    }
                }
            }
        }
        PP::Sequence(left, right) => {
            let intermediates = evaluate_path_from(left, subject, adapter, budget)?;
            for mid in &intermediates {
                let right_results = evaluate_path_from(right, mid, adapter, budget)?;
                result.extend(right_results);
            }
        }
        PP::Alternative(left, right) => {
            result.extend(evaluate_path_from(left, subject, adapter, budget)?);
            result.extend(evaluate_path_from(right, subject, adapter, budget)?);
        }
        PP::ZeroOrMore(inner) => {
            // Include subject itself (zero hops)
            result.insert(subject.clone());
            // BFS from subject. The `pop` loop is `O(V+E)` and unbounded for a
            // large graph, so it carries a throttled wall-time check (fires at 0
            // then every 1024 pops).
            let mut queue = VecDeque::new();
            queue.push_back(subject.clone());
            let mut seen = HashSet::new();
            seen.insert(subject.clone());

            let mut budget_ticks: u64 = 0;
            while let Some(current) = queue.pop_front() {
                if budget_ticks & 0x3FF == 0 {
                    path_budget_check_time(budget)?;
                }
                budget_ticks += 1;
                let next_nodes = evaluate_path_from(inner, &current, adapter, budget)?;
                for next in next_nodes {
                    if !seen.contains(&next) {
                        seen.insert(next.clone());
                        result.insert(next.clone());
                        queue.push_back(next);
                    }
                }
            }
        }
        PP::OneOrMore(inner) => {
            // BFS without including subject itself
            let mut queue = VecDeque::new();
            let mut seen = HashSet::new();
            seen.insert(subject.clone());

            let immediate = evaluate_path_from(inner, subject, adapter, budget)?;
            for node in immediate {
                if !seen.contains(&node) {
                    seen.insert(node.clone());
                    result.insert(node.clone());
                    queue.push_back(node);
                }
            }

            // Same throttled wall-time check as the `*` BFS above.
            let mut budget_ticks: u64 = 0;
            while let Some(current) = queue.pop_front() {
                if budget_ticks & 0x3FF == 0 {
                    path_budget_check_time(budget)?;
                }
                budget_ticks += 1;
                let next_nodes = evaluate_path_from(inner, &current, adapter, budget)?;
                for next in next_nodes {
                    if !seen.contains(&next) {
                        seen.insert(next.clone());
                        result.insert(next.clone());
                        queue.push_back(next);
                    }
                }
            }
        }
        PP::ZeroOrOne(inner) => {
            // Include subject itself (zero hops)
            result.insert(subject.clone());
            // Include direct successors (one hop)
            result.extend(evaluate_path_from(inner, subject, adapter, budget)?);
        }
        PP::NegatedPropertySet(excluded_preds) => {
            let all_preds = adapter.get_predicates()?;
            let excluded_set: HashSet<&Term> = excluded_preds.iter().collect();

            for pred in &all_preds {
                if excluded_set.is_empty() || !excluded_set.contains(pred) {
                    let objects = adapter.find_outgoing(subject, pred)?;
                    result.extend(objects);
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod minus_diff_tests {
    //! Differential correctness for the two MINUS strategies.
    //!
    //! An independent reference implementation ([`oracle_minus`]) encodes the raw
    //! SPARQL 1.1 §8.3.2 semantics. Every generated case asserts:
    //! * the nested fallback equals the oracle (all inputs), and
    //! * the hash anti-join equals the oracle on **homogeneous** inputs (its
    //!   precondition), i.e. `hash == nested` exactly there.
    //!
    //! Because both the code paths and the oracle preserve left order and push
    //! clones, the result `Vec`s are compared directly (order included).

    use crate::algebra::{Binding, Literal, Solution, Term, Variable};
    use crate::executor::QueryExecutor;

    fn v(name: &str) -> Variable {
        Variable::new_unchecked(name)
    }

    fn t(n: u64) -> Term {
        Term::Literal(Literal::new(n.to_string(), None, None))
    }

    /// Reference MINUS: remove `l` iff some `r` shares ≥1 bound variable and
    /// agrees on every shared bound variable. Independent of the code under test.
    fn oracle_minus(left: &Solution, right: &Solution) -> Solution {
        left.iter()
            .filter(|l| {
                !right.iter().any(|r| {
                    let shared: Vec<&Variable> = l.keys().filter(|k| r.contains_key(*k)).collect();
                    !shared.is_empty() && shared.iter().all(|k| l.get(*k) == r.get(*k))
                })
            })
            .cloned()
            .collect()
    }

    /// Recompute `S` exactly as [`QueryExecutor::execute_minus`] does.
    fn shared_of<'a>(left: &'a Solution, right: &'a Solution) -> Vec<&'a Variable> {
        use std::collections::HashSet;
        let lv: HashSet<&Variable> = left.iter().flat_map(|b| b.keys()).collect();
        let rv: HashSet<&Variable> = right.iter().flat_map(|b| b.keys()).collect();
        lv.intersection(&rv).copied().collect()
    }

    /// Tiny deterministic LCG so the sweep is reproducible without a rand dep.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn upto(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// One homogeneous solution over `vars`, values drawn from `{0..domain}`.
    fn homo_binding(vars: &[&str], rng: &mut Lcg, domain: u64) -> Binding {
        let mut b = Binding::new();
        for name in vars {
            b.insert(v(name), t(rng.upto(domain)));
        }
        b
    }

    #[test]
    fn hash_equals_nested_equals_oracle_on_homogeneous_sweep() {
        let ex = QueryExecutor::new();
        // Left binds {a,b,c}, right binds {a,b,d}: S = {a,b} (bound everywhere),
        // so `execute_minus` would route these to the hash path.
        let left_vars = ["a", "b", "c"];
        let right_vars = ["a", "b", "d"];
        let shared_names = ["a", "b"];
        let mut rng = Lcg(0x1234_5678_9abc_def0);

        for _ in 0..400 {
            let ln = rng.upto(6) as usize; // 0..=5 left rows
            let rn = rng.upto(6) as usize; // 0..=5 right rows
            let domain = 2 + rng.upto(2); // value domain 2 or 3 -> forces collisions
            let left: Solution = (0..ln)
                .map(|_| homo_binding(&left_vars, &mut rng, domain))
                .collect();
            let right: Solution = (0..rn)
                .map(|_| homo_binding(&right_vars, &mut rng, domain))
                .collect();

            let oracle = oracle_minus(&left, &right);
            let nested = ex
                .execute_minus_nested(&left, &right)
                .expect("nested must not error");
            assert_eq!(nested, oracle, "nested must equal the oracle");

            let shared = shared_of(&left, &right);
            // The hash path's precondition (S non-empty + homogeneous) only holds
            // when both sides are non-empty; then S is exactly the two shared vars.
            if !left.is_empty() && !right.is_empty() {
                assert_eq!(shared.len(), shared_names.len(), "S must be {{a,b}}");
                let hash = ex
                    .execute_minus_hash(&left, &right, &shared)
                    .expect("hash must not error");
                assert_eq!(hash, oracle, "hash must equal the oracle (== nested)");
            }
        }
    }

    #[test]
    fn nested_equals_oracle_on_heterogeneous_sweep() {
        // Randomly drop shared-variable bindings so `S`-variables go unbound on
        // some rows (the partial-binding / OPTIONAL shape). `execute_minus`
        // routes these to the nested fallback; assert it matches the oracle. The
        // hash path is intentionally NOT exercised here — its precondition fails.
        let ex = QueryExecutor::new();
        let left_vars = ["a", "b", "c"];
        let right_vars = ["a", "b", "d"];
        let mut rng = Lcg(0x0fed_cba9_8765_4321);

        for _ in 0..400 {
            let ln = rng.upto(6) as usize;
            let rn = rng.upto(6) as usize;
            let domain = 2 + rng.upto(2);
            let make = |vars: &[&str], rng: &mut Lcg| -> Binding {
                let mut b = Binding::new();
                for name in vars {
                    // 25% chance to leave this variable unbound.
                    if rng.upto(4) != 0 {
                        b.insert(v(name), t(rng.upto(domain)));
                    }
                }
                b
            };
            let left: Solution = (0..ln).map(|_| make(&left_vars, &mut rng)).collect();
            let right: Solution = (0..rn).map(|_| make(&right_vars, &mut rng)).collect();

            let oracle = oracle_minus(&left, &right);
            let nested = ex
                .execute_minus_nested(&left, &right)
                .expect("nested must not error");
            assert_eq!(
                nested, oracle,
                "nested must equal the oracle on partial bindings"
            );
        }
    }

    #[test]
    fn hash_would_diverge_on_partial_binding_justifying_the_guard() {
        // Documents WHY `execute_minus` must not send heterogeneous inputs to the
        // hash path. S = {a,b}. L2 binds only a; R1 binds a,b with a matching.
        // Oracle/nested REMOVE L2 (shared = {a}, compatible). An exact-S-tuple
        // hash instead keys L2 as [(a,1)] and R1 as [(a,1),(b,2)] -> miss -> KEEPS
        // L2. So the paths legitimately differ here, which is exactly why the
        // homogeneity guard exists.
        let ex = QueryExecutor::new();
        let mut l1 = Binding::new();
        l1.insert(v("a"), t(1));
        l1.insert(v("b"), t(2));
        let mut l2 = Binding::new();
        l2.insert(v("a"), t(1)); // b unbound
        let left: Solution = vec![l1, l2];
        let mut r1 = Binding::new();
        r1.insert(v("a"), t(1));
        r1.insert(v("b"), t(2));
        let right: Solution = vec![r1];

        let shared = shared_of(&left, &right);
        assert_eq!(shared.len(), 2, "S must be {{a,b}} for this fixture");

        let oracle = oracle_minus(&left, &right);
        assert!(oracle.is_empty(), "oracle removes both left rows");
        let nested = ex.execute_minus_nested(&left, &right).expect("nested ok");
        assert_eq!(nested, oracle, "nested matches the oracle (removes L2)");
        let hash = ex
            .execute_minus_hash(&left, &right, &shared)
            .expect("hash ok");
        assert_eq!(
            hash.len(),
            1,
            "exact-key hash wrongly keeps L2 -> guard needed"
        );
    }

    /// Real-scale timing + equivalence for the shape that motivated R8-3b:
    /// `?a prefLabel ?x MINUS { ?a altLabel ?y }` — shared `{a}`, most left rows
    /// have no matching right row so the nested loop scans all of `right` per
    /// left row (`O(|L|*|R|)`). Builds solutions of the same cardinality as the
    /// wik dataset (172 337 prefLabel rows, 16 849 altLabel rows) and runs the
    /// exact two shipped functions on the identical input, asserting the result
    /// sets are equal and printing both wall times.
    ///
    /// `#[ignore]` (drives a ~10^8+ nested loop). Scale overridable:
    /// `MINUS_BENCH_LEFT` / `MINUS_BENCH_RIGHT`. Run with, e.g.:
    /// `MINUS_BENCH_LEFT=172337 MINUS_BENCH_RIGHT=16849 \
    ///  cargo test -p oxirs-arq --lib minus_bench_nested_vs_hash -- --ignored --nocapture`
    #[test]
    #[ignore = "real-scale MINUS benchmark; run explicitly with --ignored --nocapture"]
    fn minus_bench_nested_vs_hash() {
        use std::time::Instant;
        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }
        // Defaults keep an unattended `--ignored` run bounded (~10^8 pairs);
        // override to the true wik cardinality for the headline measurement.
        let n_left = env_usize("MINUS_BENCH_LEFT", 40_000);
        let n_right = env_usize("MINUS_BENCH_RIGHT", 16_849);
        // ~11% of concepts carry an altLabel in wik, so ~11% of left rows are
        // removed; the rest exercise the full-scan-with-no-match hot path.
        let right_subjects = (n_right as u64).max(1);

        // Left: `(?a=s{i}, ?x=label{i})`, distinct subjects (homogeneous, S={a}).
        let left: Solution = (0..n_left as u64)
            .map(|i| {
                let mut b = Binding::new();
                b.insert(
                    v("a"),
                    Term::Literal(Literal::new(format!("s{i}"), None, None)),
                );
                b.insert(
                    v("x"),
                    Term::Literal(Literal::new(format!("pref{i}"), None, None)),
                );
                b
            })
            .collect();
        // Right: `(?a=s{j}, ?y=alt{j})` for j in the first `n_right` subjects, so
        // a left row is removed iff i < n_right (the low-match-rate shape).
        let right: Solution = (0..right_subjects)
            .map(|j| {
                let mut b = Binding::new();
                b.insert(
                    v("a"),
                    Term::Literal(Literal::new(format!("s{j}"), None, None)),
                );
                b.insert(
                    v("y"),
                    Term::Literal(Literal::new(format!("alt{j}"), None, None)),
                );
                b
            })
            .collect();

        let ex = QueryExecutor::new();
        let shared = shared_of(&left, &right);
        assert_eq!(shared.len(), 1, "S must be {{a}} for the pref/alt shape");

        let t0 = Instant::now();
        let nested = ex
            .execute_minus_nested(&left, &right)
            .expect("nested must not error");
        let nested_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let hash = ex
            .execute_minus_hash(&left, &right, &shared)
            .expect("hash must not error");
        let hash_ms = t1.elapsed().as_secs_f64() * 1e3;

        assert_eq!(nested, hash, "hash result set must equal nested exactly");
        let expected_kept = n_left.saturating_sub(n_right.min(n_left));
        assert_eq!(
            hash.len(),
            expected_kept,
            "kept rows = left without a right match"
        );

        println!(
            "MINUS bench L={n_left} R={n_right} kept={} | nested={nested_ms:.1}ms hash={hash_ms:.1}ms speedup={:.0}x",
            hash.len(),
            if hash_ms > 0.0 { nested_ms / hash_ms } else { f64::INFINITY }
        );
    }
}
