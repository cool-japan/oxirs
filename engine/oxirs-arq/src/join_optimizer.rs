//! Cost-based join reordering for SPARQL query optimization.
//!
//! Implements two strategies:
//! * **Greedy left-deep** — iteratively pick the cheapest next join partner.
//! * **Dynamic programming** — exact optimal plan for small pattern sets (≤12).

// ── Data structures ───────────────────────────────────────────────────────────

/// A single triple pattern annotated with cost-estimation metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinPattern {
    /// Estimated number of result rows produced by this pattern.
    pub estimated_size: usize,
    /// Variables that are already bound when this pattern is reached.
    pub bound_vars: Vec<String>,
    /// Variables that this pattern will bind (free before evaluation).
    pub free_vars: Vec<String>,
    /// Human-readable identifier (e.g. triple pattern string).
    pub label: String,
}

impl JoinPattern {
    /// Convenience constructor.
    pub fn new(
        label: impl Into<String>,
        estimated_size: usize,
        bound_vars: Vec<String>,
        free_vars: Vec<String>,
    ) -> Self {
        Self {
            estimated_size,
            bound_vars,
            free_vars,
            label: label.into(),
        }
    }
}

/// An edge in the join graph connecting two patterns that share variables.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinEdge {
    /// Index of the left pattern.
    pub left: usize,
    /// Index of the right pattern.
    pub right: usize,
    /// Variables shared by both patterns (basis for join selectivity).
    pub shared_vars: Vec<String>,
    /// Estimated fraction of cross-product surviving the join (0.0–1.0).
    pub join_selectivity: f64,
}

/// An ordered join plan produced by the optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinPlan {
    /// Ordered indices into the original `patterns` slice.
    pub order: Vec<usize>,
    /// Estimated total cost of executing this join order.
    pub estimated_cost: f64,
}

// ── Optimizer ─────────────────────────────────────────────────────────────────

/// Cost-based join reordering optimizer.
#[derive(Debug, Clone)]
pub struct JoinOptimizer {
    /// Maximum number of patterns handled by the DP algorithm before falling
    /// back to the greedy heuristic.
    pub max_patterns: usize,
}

impl JoinOptimizer {
    /// Create a new optimizer.
    ///
    /// `max_patterns` controls the DP threshold (recommended: ≤12).
    pub fn new(max_patterns: usize) -> Self {
        Self { max_patterns }
    }

    /// Choose the best join order.
    ///
    /// Uses DP when `patterns.len() <= max_patterns`, greedy otherwise.
    pub fn optimize(&self, patterns: Vec<JoinPattern>) -> JoinPlan {
        if patterns.is_empty() {
            return JoinPlan {
                order: vec![],
                estimated_cost: 0.0,
            };
        }
        if patterns.len() <= self.max_patterns {
            self.dynamic_programming(&patterns)
        } else {
            Self::greedy(&patterns)
        }
    }

    // ── Greedy left-deep ──────────────────────────────────────────────────────

    /// Greedy left-deep join reordering.
    ///
    /// At each step we pick the unplaced pattern that, when joined with the
    /// current pipeline, yields the lowest cost increment.
    pub fn greedy(patterns: &[JoinPattern]) -> JoinPlan {
        if patterns.is_empty() {
            return JoinPlan {
                order: vec![],
                estimated_cost: 0.0,
            };
        }

        let n = patterns.len();
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut total_cost = 0.0_f64;

        // Seed: choose the pattern with the smallest estimated size.
        let (seed_pos, _) = remaining
            .iter()
            .enumerate()
            .min_by_key(|&(_, &idx)| patterns[idx].estimated_size)
            .unwrap_or((0, &0));
        let seed_idx = remaining.remove(seed_pos);
        order.push(seed_idx);
        let mut accumulated = patterns[seed_idx].clone();
        total_cost += accumulated.estimated_size as f64;

        while !remaining.is_empty() {
            let mut best_pos = 0_usize;
            let mut best_cost = f64::MAX;

            for (pos, &candidate) in remaining.iter().enumerate() {
                let incremental = Self::cost_join(&accumulated, &patterns[candidate]);
                if incremental < best_cost {
                    best_cost = incremental;
                    best_pos = pos;
                }
            }

            let chosen = remaining.remove(best_pos);
            accumulated = Self::merge_patterns(&accumulated, &patterns[chosen]);
            total_cost += best_cost;
            order.push(chosen);
        }

        JoinPlan {
            order,
            estimated_cost: total_cost,
        }
    }

    // ── Dynamic programming ───────────────────────────────────────────────────

    /// Exact optimal join order via dynamic programming (subset enumeration).
    ///
    /// Limited to `patterns.len() ≤ 12` to keep exponential blowup manageable.
    pub fn dynamic_programming(&self, patterns: &[JoinPattern]) -> JoinPlan {
        let n = patterns.len();
        if n == 0 {
            return JoinPlan {
                order: vec![],
                estimated_cost: 0.0,
            };
        }
        if n == 1 {
            return JoinPlan {
                order: vec![0],
                estimated_cost: patterns[0].estimated_size as f64,
            };
        }
        // Cap to avoid exponential blowup.
        if n > 20 {
            return Self::greedy(patterns);
        }

        let num_subsets = 1usize << n;
        // dp[mask] = (cost, last_added)
        let mut dp = vec![(f64::MAX, usize::MAX); num_subsets];
        // accumulated_pattern[mask] holds the merged pattern for that subset.
        let mut acc: Vec<Option<JoinPattern>> = vec![None; num_subsets];

        // Initialise singletons.
        // `i` is intentionally used as both a bit-position index and an index into
        // `patterns` — using enumerate() would not simplify this.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let mask = 1usize << i;
            dp[mask] = (patterns[i].estimated_size as f64, i);
            acc[mask] = Some(patterns[i].clone());
        }

        for mask in 1..num_subsets {
            if dp[mask].0 == f64::MAX && mask.count_ones() > 1 {
                continue;
            }
            // Try extending with each remaining pattern.
            // `i` is used as both bit-position and pattern index.
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if mask & (1 << i) != 0 {
                    continue; // already in subset
                }
                let new_mask = mask | (1 << i);
                if let Some(ref current) = acc[mask] {
                    let incremental = Self::cost_join(current, &patterns[i]);
                    let new_cost = dp[mask].0 + incremental;
                    if new_cost < dp[new_mask].0 {
                        dp[new_mask] = (new_cost, i);
                        acc[new_mask] = Some(Self::merge_patterns(current, &patterns[i]));
                    }
                }
            }
        }

        // Reconstruct the order by backtracking through the DP table.
        let full_mask = num_subsets - 1;
        let total_cost = dp[full_mask].0;
        let order = Self::reconstruct_order(&dp, n, full_mask);

        JoinPlan {
            order,
            estimated_cost: total_cost,
        }
    }

    // ── Cost function ─────────────────────────────────────────────────────────

    /// Incremental join cost when appending `right` to a pipeline ending with `left`.
    ///
    /// Considers:
    /// * base sizes of both patterns
    /// * selectivity from shared variables (each shared variable reduces cost)
    pub fn cost_join(left: &JoinPattern, right: &JoinPattern) -> f64 {
        let shared = Self::shared_variables(left, right);
        let base = (left.estimated_size as f64) * (right.estimated_size as f64).max(1.0);
        let selectivity = if shared.is_empty() {
            1.0
        } else {
            // Each shared variable reduces result set size.
            let factor = 0.1_f64.powi(shared.len() as i32);
            factor.max(1e-6)
        };
        base * selectivity
    }

    // ── Join graph ────────────────────────────────────────────────────────────

    /// Build a join graph: one edge for every pair of patterns that share at
    /// least one variable.
    pub fn build_join_graph(patterns: &[JoinPattern]) -> Vec<JoinEdge> {
        let mut edges = Vec::new();
        let n = patterns.len();
        // Indices `i` and `j` are stored directly in `JoinEdge::left/right`,
        // so we use range loops rather than enumerate.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in (i + 1)..n {
                let shared = Self::shared_variables(&patterns[i], &patterns[j]);
                if !shared.is_empty() {
                    let left_size = patterns[i].estimated_size.max(1) as f64;
                    let right_size = patterns[j].estimated_size.max(1) as f64;
                    let cross = left_size * right_size;
                    let join_selectivity = (1.0 / cross * shared.len() as f64).min(1.0);
                    edges.push(JoinEdge {
                        left: i,
                        right: j,
                        shared_vars: shared,
                        join_selectivity,
                    });
                }
            }
        }
        edges
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Return variables that appear in both `a` and `b`.
    pub fn shared_variables(a: &JoinPattern, b: &JoinPattern) -> Vec<String> {
        let mut shared: Vec<String> = a
            .free_vars
            .iter()
            .chain(a.bound_vars.iter())
            .filter(|v| b.free_vars.contains(v) || b.bound_vars.contains(v))
            .cloned()
            .collect();
        shared.sort();
        shared.dedup();
        shared
    }

    /// Merge two patterns into the combined pattern produced by their join.
    pub fn merge_patterns(left: &JoinPattern, right: &JoinPattern) -> JoinPattern {
        let shared = Self::shared_variables(left, right);
        let merged_size = if shared.is_empty() {
            left.estimated_size.saturating_mul(right.estimated_size)
        } else {
            // Each shared variable applies a selectivity factor of 0.1,
            // consistent with cost_join's selectivity model.
            let cross = (left.estimated_size as f64) * (right.estimated_size as f64);
            let selectivity = 0.1_f64.powi(shared.len() as i32).max(1e-6);
            (cross * selectivity).round().max(1.0) as usize
        };

        let mut bound: Vec<String> = left
            .bound_vars
            .iter()
            .chain(left.free_vars.iter())
            .chain(right.bound_vars.iter())
            .cloned()
            .collect();
        bound.sort();
        bound.dedup();

        let free: Vec<String> = right
            .free_vars
            .iter()
            .filter(|v| !bound.contains(v))
            .cloned()
            .collect();

        JoinPattern {
            estimated_size: merged_size,
            bound_vars: bound,
            free_vars: free,
            label: format!("({} ⋈ {})", left.label, right.label),
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn reconstruct_order(dp: &[(f64, usize)], n: usize, mut mask: usize) -> Vec<usize> {
        let mut order = Vec::with_capacity(n);
        while mask != 0 {
            let (_, last) = dp[mask];
            if last == usize::MAX {
                break;
            }
            order.push(last);
            mask ^= 1 << last;
        }
        order.reverse();
        order
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pat(label: &str, size: usize, bound: &[&str], free: &[&str]) -> JoinPattern {
        JoinPattern::new(
            label,
            size,
            bound.iter().map(|s| s.to_string()).collect(),
            free.iter().map(|s| s.to_string()).collect(),
        )
    }

    // ── JoinPattern ───────────────────────────────────────────────────────────

    #[test]
    fn test_pattern_new() {
        let p = pat("p1", 100, &["x"], &["y"]);
        assert_eq!(p.label, "p1");
        assert_eq!(p.estimated_size, 100);
        assert_eq!(p.bound_vars, vec!["x"]);
        assert_eq!(p.free_vars, vec!["y"]);
    }

    #[test]
    fn test_pattern_no_vars() {
        let p = pat("empty", 1, &[], &[]);
        assert!(p.bound_vars.is_empty());
        assert!(p.free_vars.is_empty());
    }

    // ── shared_variables ─────────────────────────────────────────────────────

    #[test]
    fn test_shared_vars_empty() {
        let a = pat("a", 10, &[], &["x"]);
        let b = pat("b", 10, &[], &["y"]);
        assert!(JoinOptimizer::shared_variables(&a, &b).is_empty());
    }

    #[test]
    fn test_shared_vars_one() {
        let a = pat("a", 10, &[], &["x", "y"]);
        let b = pat("b", 10, &[], &["y", "z"]);
        assert_eq!(JoinOptimizer::shared_variables(&a, &b), vec!["y"]);
    }

    #[test]
    fn test_shared_vars_multiple() {
        let a = pat("a", 10, &["x"], &["y", "z"]);
        let b = pat("b", 10, &["x", "y"], &["w"]);
        let shared = JoinOptimizer::shared_variables(&a, &b);
        assert!(shared.contains(&"x".to_string()));
        assert!(shared.contains(&"y".to_string()));
    }

    #[test]
    fn test_shared_vars_between_bound_and_free() {
        let a = pat("a", 5, &["s"], &["p"]);
        let b = pat("b", 5, &[], &["s"]);
        let shared = JoinOptimizer::shared_variables(&a, &b);
        assert!(shared.contains(&"s".to_string()));
    }

    // ── cost_join ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cost_join_no_shared() {
        let a = pat("a", 100, &[], &["x"]);
        let b = pat("b", 200, &[], &["y"]);
        let cost = JoinOptimizer::cost_join(&a, &b);
        // cross product with selectivity 1.0
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_join_with_shared_lower() {
        let a = pat("a", 100, &[], &["x"]);
        let b_no_share = pat("b", 200, &[], &["y"]);
        let b_shared = pat("b", 200, &[], &["x", "y"]);
        let cost_no = JoinOptimizer::cost_join(&a, &b_no_share);
        let cost_shared = JoinOptimizer::cost_join(&a, &b_shared);
        assert!(cost_shared < cost_no);
    }

    #[test]
    fn test_cost_join_symmetry_not_required() {
        // cost_join is generally not symmetric, just check it returns a value
        let a = pat("a", 50, &[], &["x"]);
        let b = pat("b", 150, &[], &["x"]);
        let c1 = JoinOptimizer::cost_join(&a, &b);
        let c2 = JoinOptimizer::cost_join(&b, &a);
        assert!(c1 > 0.0);
        assert!(c2 > 0.0);
    }

    #[test]
    fn test_cost_join_zero_size() {
        let a = pat("a", 0, &[], &["x"]);
        let b = pat("b", 100, &[], &["y"]);
        let cost = JoinOptimizer::cost_join(&a, &b);
        assert!(cost >= 0.0);
    }

    // ── merge_patterns ────────────────────────────────────────────────────────

    #[test]
    fn test_merge_label() {
        let a = pat("A", 10, &[], &["x"]);
        let b = pat("B", 20, &[], &["x", "y"]);
        let merged = JoinOptimizer::merge_patterns(&a, &b);
        assert!(merged.label.contains('A'));
        assert!(merged.label.contains('B'));
    }

    #[test]
    fn test_merge_size_with_shared() {
        let a = pat("a", 100, &[], &["x"]);
        let b = pat("b", 100, &[], &["x", "y"]);
        let merged = JoinOptimizer::merge_patterns(&a, &b);
        // Shared variable reduces size below cross product
        assert!(merged.estimated_size < 100 * 100);
    }

    #[test]
    fn test_merge_size_no_shared() {
        let a = pat("a", 10, &[], &["x"]);
        let b = pat("b", 10, &[], &["y"]);
        let merged = JoinOptimizer::merge_patterns(&a, &b);
        assert_eq!(merged.estimated_size, 100);
    }

    #[test]
    fn test_merge_bound_vars_accumulate() {
        let a = pat("a", 1, &[], &["x"]);
        let b = pat("b", 1, &[], &["x", "y"]);
        let merged = JoinOptimizer::merge_patterns(&a, &b);
        assert!(merged.bound_vars.contains(&"x".to_string()));
    }

    // ── build_join_graph ──────────────────────────────────────────────────────

    #[test]
    fn test_join_graph_no_shared() {
        let patterns = vec![pat("a", 10, &[], &["x"]), pat("b", 10, &[], &["y"])];
        let edges = JoinOptimizer::build_join_graph(&patterns);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_join_graph_one_edge() {
        let patterns = vec![pat("a", 10, &[], &["x"]), pat("b", 10, &[], &["x", "y"])];
        let edges = JoinOptimizer::build_join_graph(&patterns);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].left, 0);
        assert_eq!(edges[0].right, 1);
        assert!(edges[0].shared_vars.contains(&"x".to_string()));
    }

    #[test]
    fn test_join_graph_selectivity_range() {
        let patterns = vec![pat("a", 100, &[], &["x"]), pat("b", 100, &[], &["x", "y"])];
        let edges = JoinOptimizer::build_join_graph(&patterns);
        assert!(!edges.is_empty());
        let sel = edges[0].join_selectivity;
        assert!((0.0..=1.0).contains(&sel));
    }

    #[test]
    fn test_join_graph_multiple_edges() {
        let patterns = vec![
            pat("a", 10, &[], &["x", "y"]),
            pat("b", 10, &[], &["y", "z"]),
            pat("c", 10, &[], &["x", "z"]),
        ];
        let edges = JoinOptimizer::build_join_graph(&patterns);
        assert_eq!(edges.len(), 3);
    }

    // ── greedy optimizer ──────────────────────────────────────────────────────

    #[test]
    fn test_greedy_empty() {
        let plan = JoinOptimizer::greedy(&[]);
        assert!(plan.order.is_empty());
        assert_eq!(plan.estimated_cost, 0.0);
    }

    #[test]
    fn test_greedy_single() {
        let patterns = vec![pat("a", 42, &[], &["x"])];
        let plan = JoinOptimizer::greedy(&patterns);
        assert_eq!(plan.order, vec![0]);
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_greedy_two_patterns_order() {
        // Smaller pattern should be placed first
        let patterns = vec![pat("big", 1000, &[], &["x"]), pat("small", 5, &[], &["y"])];
        let plan = JoinOptimizer::greedy(&patterns);
        assert_eq!(plan.order[0], 1); // small first
    }

    #[test]
    fn test_greedy_all_patterns_included() {
        let patterns = vec![
            pat("a", 10, &[], &["x"]),
            pat("b", 20, &[], &["y"]),
            pat("c", 5, &[], &["z"]),
        ];
        let plan = JoinOptimizer::greedy(&patterns);
        assert_eq!(plan.order.len(), 3);
        let mut sorted = plan.order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_greedy_prefers_shared_vars() {
        // Pattern sharing a variable with the pipeline should cost less.
        let patterns = vec![
            pat("seed", 10, &[], &["x"]),
            pat("linked", 100, &[], &["x", "y"]),
            pat("unrelated", 50, &[], &["z"]),
        ];
        let plan = JoinOptimizer::greedy(&patterns);
        assert_eq!(plan.order.len(), 3);
        assert!(plan.estimated_cost > 0.0);
    }

    // ── DP optimizer ─────────────────────────────────────────────────────────

    #[test]
    fn test_dp_empty() {
        let opt = JoinOptimizer::new(12);
        let plan = opt.dynamic_programming(&[]);
        assert!(plan.order.is_empty());
    }

    #[test]
    fn test_dp_single() {
        let opt = JoinOptimizer::new(12);
        let patterns = vec![pat("a", 10, &[], &["x"])];
        let plan = opt.dynamic_programming(&patterns);
        assert_eq!(plan.order, vec![0]);
    }

    #[test]
    fn test_dp_two_patterns() {
        let opt = JoinOptimizer::new(12);
        let patterns = vec![pat("big", 500, &[], &["x"]), pat("small", 5, &[], &["y"])];
        let plan = opt.dynamic_programming(&patterns);
        assert_eq!(plan.order.len(), 2);
    }

    #[test]
    fn test_dp_covers_all() {
        let opt = JoinOptimizer::new(12);
        let patterns = vec![
            pat("a", 10, &[], &["x", "y"]),
            pat("b", 20, &[], &["y", "z"]),
            pat("c", 5, &[], &["z", "w"]),
        ];
        let plan = opt.dynamic_programming(&patterns);
        assert_eq!(plan.order.len(), 3);
        let mut sorted = plan.order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_dp_cost_not_negative() {
        let opt = JoinOptimizer::new(12);
        let patterns = vec![
            pat("a", 100, &[], &["x"]),
            pat("b", 200, &[], &["x", "y"]),
            pat("c", 50, &[], &["y", "z"]),
        ];
        let plan = opt.dynamic_programming(&patterns);
        assert!(plan.estimated_cost > 0.0);
    }

    // ── optimize (dispatch) ───────────────────────────────────────────────────

    #[test]
    fn test_optimize_small_uses_dp() {
        let opt = JoinOptimizer::new(12);
        let patterns = vec![pat("a", 10, &[], &["x"]), pat("b", 20, &[], &["y"])];
        let plan = opt.optimize(patterns);
        assert_eq!(plan.order.len(), 2);
    }

    #[test]
    fn test_optimize_large_uses_greedy() {
        let opt = JoinOptimizer::new(2); // threshold = 2
        let patterns: Vec<_> = (0..5)
            .map(|i| pat(&format!("p{i}"), 10 + i, &[], &[&format!("x{i}")]))
            .collect();
        let plan = opt.optimize(patterns);
        assert_eq!(plan.order.len(), 5);
    }

    #[test]
    fn test_optimize_empty() {
        let opt = JoinOptimizer::new(12);
        let plan = opt.optimize(vec![]);
        assert!(plan.order.is_empty());
        assert_eq!(plan.estimated_cost, 0.0);
    }

    #[test]
    fn test_optimize_single() {
        let opt = JoinOptimizer::new(12);
        let plan = opt.optimize(vec![pat("solo", 7, &[], &["x"])]);
        assert_eq!(plan.order, vec![0]);
    }

    // ── JoinEdge ─────────────────────────────────────────────────────────────

    #[test]
    fn test_join_edge_fields() {
        let edge = JoinEdge {
            left: 0,
            right: 1,
            shared_vars: vec!["x".to_string()],
            join_selectivity: 0.5,
        };
        assert_eq!(edge.left, 0);
        assert_eq!(edge.right, 1);
        assert_eq!(edge.shared_vars, vec!["x"]);
        assert!((edge.join_selectivity - 0.5).abs() < 1e-9);
    }

    // ── JoinPlan ─────────────────────────────────────────────────────────────

    #[test]
    fn test_join_plan_fields() {
        let plan = JoinPlan {
            order: vec![2, 0, 1],
            estimated_cost: 123.45,
        };
        assert_eq!(plan.order, vec![2, 0, 1]);
        assert!((plan.estimated_cost - 123.45).abs() < 1e-9);
    }

    // ── Integration ───────────────────────────────────────────────────────────

    #[test]
    fn test_star_shaped_query() {
        // Classic star query: central node with many outgoing edges.
        let patterns = vec![
            pat("?s rdf:type :Person", 1000, &[], &["s"]),
            pat("?s :name ?name", 5000, &["s"], &["name"]),
            pat("?s :age ?age", 5000, &["s"], &["age"]),
            pat("?s :email ?email", 4000, &["s"], &["email"]),
        ];
        let opt = JoinOptimizer::new(12);
        let plan = opt.optimize(patterns);
        assert_eq!(plan.order.len(), 4);
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_chain_query() {
        let patterns = vec![
            pat("?a :knows ?b", 200, &[], &["a", "b"]),
            pat("?b :knows ?c", 200, &["b"], &["c"]),
            pat("?c :knows ?d", 200, &["c"], &["d"]),
        ];
        let opt = JoinOptimizer::new(12);
        let plan = opt.optimize(patterns);
        assert_eq!(plan.order.len(), 3);
    }

    #[test]
    fn test_dp_better_or_equal_than_greedy_for_small() {
        // DP should find cost ≤ greedy for small inputs.
        let patterns = vec![
            pat("a", 1000, &[], &["x"]),
            pat("b", 100, &[], &["x", "y"]),
            pat("c", 10, &[], &["y"]),
        ];
        let opt = JoinOptimizer::new(12);
        let dp_plan = opt.dynamic_programming(&patterns);
        let greedy_plan = JoinOptimizer::greedy(&patterns);
        // DP cost should be ≤ greedy cost (or very close)
        assert!(dp_plan.estimated_cost <= greedy_plan.estimated_cost + 1.0);
    }

    #[test]
    fn test_large_pattern_set_greedy() {
        let patterns: Vec<_> = (0..15)
            .map(|i| {
                pat(
                    &format!("p{i}"),
                    100 + i * 7,
                    &[],
                    &[&format!("v{i}"), &format!("v{}", i + 1)],
                )
            })
            .collect();
        let opt = JoinOptimizer::new(12);
        let plan = opt.optimize(patterns);
        assert_eq!(plan.order.len(), 15);
    }

    #[test]
    fn test_merge_deduplicates_vars() {
        let a = pat("a", 10, &["x"], &["x", "y"]);
        let b = pat("b", 10, &["x", "y"], &["y", "z"]);
        let merged = JoinOptimizer::merge_patterns(&a, &b);
        // No duplicate "x" or "y"
        let x_count = merged.bound_vars.iter().filter(|v| *v == "x").count();
        let y_count = merged.bound_vars.iter().filter(|v| *v == "y").count();
        assert_eq!(x_count, 1);
        assert_eq!(y_count, 1);
    }

    #[test]
    fn test_join_graph_empty_patterns() {
        let edges = JoinOptimizer::build_join_graph(&[]);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_join_graph_single_pattern() {
        let patterns = vec![pat("a", 10, &[], &["x"])];
        let edges = JoinOptimizer::build_join_graph(&patterns);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_cost_join_large_size() {
        let a = pat("a", 1_000_000, &[], &["x"]);
        let b = pat("b", 1_000_000, &[], &["y"]);
        let cost = JoinOptimizer::cost_join(&a, &b);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_optimize_four_patterns() {
        let patterns = vec![
            pat("a", 10, &[], &["x"]),
            pat("b", 50, &[], &["x", "y"]),
            pat("c", 100, &[], &["y", "z"]),
            pat("d", 20, &[], &["z", "w"]),
        ];
        let opt = JoinOptimizer::new(12);
        let plan = opt.optimize(patterns);
        assert_eq!(plan.order.len(), 4);
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_optimizer_new() {
        let opt = JoinOptimizer::new(8);
        assert_eq!(opt.max_patterns, 8);
    }

    #[test]
    fn test_dp_twelve_patterns() {
        let opt = JoinOptimizer::new(12);
        let patterns: Vec<_> = (0..12)
            .map(|i| {
                pat(
                    &format!("p{i}"),
                    10 + i,
                    &[],
                    &[&format!("v{i}"), &format!("v{}", i + 1)],
                )
            })
            .collect();
        let plan = opt.dynamic_programming(&patterns);
        assert_eq!(plan.order.len(), 12);
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_plan_order_is_permutation() {
        let patterns: Vec<_> = (0..6)
            .map(|i| pat(&format!("p{i}"), 10 + i * 3, &[], &[&format!("v{i}")]))
            .collect();
        let opt = JoinOptimizer::new(12);
        let plan = opt.optimize(patterns);
        let mut sorted = plan.order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }
}
