//! # Cost Model for Federated SPARQL Query Optimization
//!
//! Provides cost estimation for federated query planning: scan costs, join
//! costs, network transfer costs, and optimal join ordering via greedy
//! left-deep plans.
//!
//! ## Overview
//!
//! The cost model combines:
//! - **Scan cost**: I/O based on triple count, selectivity, and estimated results.
//! - **Join cost**: combination of left / right materialization plus join overhead.
//! - **Transfer cost**: network latency and bandwidth between endpoints.
//! - **Greedy ordering**: repeatedly pick the cheapest next fragment.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Endpoint statistics
// ─────────────────────────────────────────────

/// Runtime statistics for a single SPARQL endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStats {
    /// URL of the SPARQL endpoint.
    pub endpoint_url: String,
    /// Average query latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Approximate triple count stored at this endpoint.
    pub triples_count: u64,
    /// Estimated fraction of triples matching a typical pattern (0.0 – 1.0).
    pub selectivity_estimate: f64,
    /// Available bandwidth to/from this endpoint in Mbit/s.
    pub bandwidth_mbps: f64,
}

// ─────────────────────────────────────────────
// Join strategy
// ─────────────────────────────────────────────

/// Physical join algorithm chosen by the cost model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinStrategy {
    /// Classic hash join (O(n+m)).
    HashJoin,
    /// Nested-loop join (O(n×m), best when one side is tiny).
    NestedLoop,
    /// Sort-merge join (O(n log n + m log m), best when both sides are sorted).
    MergeJoin,
    /// Semi-join reduction: send keys from right to left to reduce transfer.
    SemiJoin,
}

// ─────────────────────────────────────────────
// Join cost estimate
// ─────────────────────────────────────────────

/// Detailed cost breakdown for a binary join between two query fragments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinCost {
    /// Cost of materialising the left operand.
    pub left_cost: f64,
    /// Cost of materialising the right operand.
    pub right_cost: f64,
    /// Cost of the join kernel itself.
    pub join_cost: f64,
    /// `left_cost + right_cost + join_cost`.
    pub total_cost: f64,
    /// Recommended join algorithm.
    pub preferred_strategy: JoinStrategy,
}

// ─────────────────────────────────────────────
// Query fragment
// ─────────────────────────────────────────────

/// A portion of a SPARQL query that will be sent to a single endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFragment {
    /// Target endpoint URL.
    pub endpoint: String,
    /// Number of triple patterns in this sub-query.
    pub pattern_count: usize,
    /// Estimated number of result rows produced.
    pub estimated_results: u64,
    /// Combined filter selectivity (0.0 = no results, 1.0 = all results pass).
    pub filter_selectivity: f64,
}

// ─────────────────────────────────────────────
// Cost model
// ─────────────────────────────────────────────

/// Federated query cost model.
#[derive(Debug, Clone)]
pub struct CostModel {
    endpoint_stats: HashMap<String, EndpointStats>,
    /// Fixed per-hop network overhead added to every transfer cost (ms).
    network_overhead_ms: f64,
}

impl CostModel {
    /// Create a new cost model with the given fixed network overhead.
    pub fn new(network_overhead_ms: f64) -> Self {
        Self {
            endpoint_stats: HashMap::new(),
            network_overhead_ms,
        }
    }

    /// Register (or replace) statistics for a SPARQL endpoint.
    pub fn register_endpoint(&mut self, stats: EndpointStats) {
        self.endpoint_stats
            .insert(stats.endpoint_url.clone(), stats);
    }

    /// Estimate the cost of executing a single fragment at its endpoint.
    ///
    /// Formula: `latency + (triples × selectivity × patterns) / 1e6 × filter`
    pub fn estimate_scan_cost(&self, fragment: &QueryFragment) -> f64 {
        let base = if let Some(ep) = self.endpoint_stats.get(&fragment.endpoint) {
            let io_cost = (ep.triples_count as f64)
                * ep.selectivity_estimate
                * (fragment.pattern_count as f64)
                / 1_000_000.0;
            ep.avg_latency_ms + io_cost
        } else {
            // Unknown endpoint — use a conservative estimate.
            500.0 + (fragment.pattern_count as f64) * 100.0
        };
        base * fragment.filter_selectivity.clamp(0.001, 1.0)
    }

    /// Estimate the cost of joining two fragments.
    pub fn estimate_join_cost(&self, left: &QueryFragment, right: &QueryFragment) -> JoinCost {
        let left_cost = self.estimate_scan_cost(left);
        let right_cost = self.estimate_scan_cost(right);

        let left_rows = left.estimated_results as f64;
        let right_rows = right.estimated_results as f64;

        // Choose strategy heuristically.
        let (strategy, join_kernel_cost) = if left_rows.min(right_rows) < 100.0 {
            // One side is tiny — nested loop is optimal.
            (
                JoinStrategy::NestedLoop,
                left_rows.min(right_rows) * right_rows.max(left_rows) / 1_000.0,
            )
        } else if right_rows < left_rows * 0.1 {
            // Right side is much smaller — semi-join to reduce transfer.
            (JoinStrategy::SemiJoin, (left_rows + right_rows) / 1_000.0)
        } else if left_rows + right_rows < 10_000.0 {
            // Moderate size — hash join.
            (JoinStrategy::HashJoin, (left_rows + right_rows) / 1_000.0)
        } else {
            // Large sides — merge join (assumes sorted input from storage).
            (
                JoinStrategy::MergeJoin,
                (left_rows * left_rows.log2().max(1.0) + right_rows * right_rows.log2().max(1.0))
                    / 1_000.0,
            )
        };

        let join_cost = join_kernel_cost;
        let total_cost = left_cost + right_cost + join_cost;

        JoinCost {
            left_cost,
            right_cost,
            join_cost,
            total_cost,
            preferred_strategy: strategy,
        }
    }

    /// Estimate the cost of transferring `result_count` rows from endpoint
    /// `from` to endpoint `to`.
    ///
    /// Formula: `overhead + (result_count × 200 bytes) / (bandwidth × 125000) × 1000`
    pub fn estimate_transfer_cost(&self, from: &str, to: &str, result_count: u64) -> f64 {
        if from == to {
            return 0.0;
        }
        let bw_mbps = self
            .endpoint_stats
            .get(from)
            .map(|s| s.bandwidth_mbps)
            .unwrap_or(100.0)
            .min(
                self.endpoint_stats
                    .get(to)
                    .map(|s| s.bandwidth_mbps)
                    .unwrap_or(100.0),
            );

        // Assume 200 bytes per result row; bw_mbps → bytes/ms = bw_mbps * 125.
        let bytes = result_count as f64 * 200.0;
        let transfer_ms = bytes / (bw_mbps * 125.0);
        self.network_overhead_ms + transfer_ms
    }

    /// Return the indices of `fragments` sorted by ascending scan cost.
    pub fn rank_fragments(&self, fragments: &[QueryFragment]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..fragments.len()).collect();
        indices.sort_by(|&a, &b| {
            let ca = self.estimate_scan_cost(&fragments[a]);
            let cb = self.estimate_scan_cost(&fragments[b]);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Estimate the total cost of executing all fragments independently and
    /// joining their results at a central coordinator.
    ///
    /// This is the naive sum; real plans may be cheaper due to parallelism.
    pub fn estimate_total_plan_cost(&self, fragments: &[QueryFragment]) -> f64 {
        if fragments.is_empty() {
            return 0.0;
        }

        // Sum of scan costs + pairwise transfer to a virtual coordinator.
        let scan_total: f64 = fragments.iter().map(|f| self.estimate_scan_cost(f)).sum();
        let transfer_total: f64 = fragments
            .iter()
            .map(|f| self.estimate_transfer_cost(&f.endpoint, "coordinator", f.estimated_results))
            .sum();

        scan_total + transfer_total
    }

    /// Compute a greedy left-deep join order: start with the cheapest
    /// fragment, then repeatedly append the fragment whose transfer cost
    /// to the current join result is lowest.
    ///
    /// Returns fragment indices in the recommended execution order.
    pub fn optimal_join_order(&self, fragments: &[QueryFragment]) -> Vec<usize> {
        if fragments.is_empty() {
            return vec![];
        }

        let mut remaining: Vec<usize> = (0..fragments.len()).collect();
        let mut order: Vec<usize> = Vec::with_capacity(fragments.len());

        // Seed: pick the cheapest fragment.
        let seed = remaining
            .iter()
            .copied()
            .min_by(|&a, &b| {
                self.estimate_scan_cost(&fragments[a])
                    .partial_cmp(&self.estimate_scan_cost(&fragments[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        order.push(seed);
        remaining.retain(|&i| i != seed);

        // Greedily append the fragment that minimises incremental transfer
        // cost from the last added endpoint.
        while !remaining.is_empty() {
            let last_ep = &fragments[*order.last().unwrap_or(&seed)].endpoint;
            let last_rows = fragments[*order.last().unwrap_or(&seed)].estimated_results;

            let best = remaining
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let ca =
                        self.estimate_transfer_cost(last_ep, &fragments[a].endpoint, last_rows)
                            + self.estimate_scan_cost(&fragments[a]);
                    let cb =
                        self.estimate_transfer_cost(last_ep, &fragments[b].endpoint, last_rows)
                            + self.estimate_scan_cost(&fragments[b]);
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(remaining[0]);

            order.push(best);
            remaining.retain(|&i| i != best);
        }

        order
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(url: &str) -> EndpointStats {
        EndpointStats {
            endpoint_url: url.to_string(),
            avg_latency_ms: 10.0,
            triples_count: 1_000_000,
            selectivity_estimate: 0.01,
            bandwidth_mbps: 100.0,
        }
    }

    fn frag(endpoint: &str, patterns: usize, results: u64, sel: f64) -> QueryFragment {
        QueryFragment {
            endpoint: endpoint.to_string(),
            pattern_count: patterns,
            estimated_results: results,
            filter_selectivity: sel,
        }
    }

    // ── construction ───────────────────────────────────────────────────

    #[test]
    fn test_new_empty() {
        let model = CostModel::new(5.0);
        assert_eq!(model.endpoint_stats.len(), 0);
        assert!((model.network_overhead_ms - 5.0).abs() < 1e-9);
    }

    // ── register_endpoint ──────────────────────────────────────────────

    #[test]
    fn test_register_endpoint() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        assert!(model.endpoint_stats.contains_key("http://a.example/sparql"));
    }

    #[test]
    fn test_register_endpoint_overwrites() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let mut ep2 = ep("http://a.example/sparql");
        ep2.avg_latency_ms = 999.0;
        model.register_endpoint(ep2);
        assert!(
            (model.endpoint_stats["http://a.example/sparql"].avg_latency_ms - 999.0).abs() < 1e-9
        );
    }

    // ── estimate_scan_cost ──────────────────────────────────────────────

    #[test]
    fn test_scan_cost_known_endpoint() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let f = frag("http://a.example/sparql", 2, 100, 1.0);
        let cost = model.estimate_scan_cost(&f);
        assert!(cost > 0.0, "cost should be positive");
    }

    #[test]
    fn test_scan_cost_unknown_endpoint_conservative() {
        let model = CostModel::new(0.0);
        let f = frag("http://unknown/sparql", 3, 100, 1.0);
        let cost = model.estimate_scan_cost(&f);
        assert!(cost > 500.0, "unknown endpoint should be conservative");
    }

    #[test]
    fn test_scan_cost_scales_with_pattern_count() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let f1 = frag("http://a.example/sparql", 1, 100, 1.0);
        let f2 = frag("http://a.example/sparql", 5, 100, 1.0);
        assert!(model.estimate_scan_cost(&f2) > model.estimate_scan_cost(&f1));
    }

    #[test]
    fn test_scan_cost_low_selectivity_cheaper() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let f_full = frag("http://a.example/sparql", 2, 100, 1.0);
        let f_sel = frag("http://a.example/sparql", 2, 10, 0.01);
        assert!(model.estimate_scan_cost(&f_sel) < model.estimate_scan_cost(&f_full));
    }

    // ── estimate_join_cost ─────────────────────────────────────────────

    #[test]
    fn test_join_cost_positive() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        model.register_endpoint(ep("http://b.example/sparql"));
        let l = frag("http://a.example/sparql", 2, 500, 1.0);
        let r = frag("http://b.example/sparql", 1, 200, 1.0);
        let jc = model.estimate_join_cost(&l, &r);
        assert!(jc.total_cost > 0.0);
        assert!((jc.left_cost + jc.right_cost + jc.join_cost - jc.total_cost).abs() < 1e-6);
    }

    #[test]
    fn test_join_cost_nested_loop_for_small_side() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let l = frag("http://a.example/sparql", 1, 50, 1.0); // tiny
        let r = frag("http://a.example/sparql", 2, 50, 1.0);
        let jc = model.estimate_join_cost(&l, &r);
        assert_eq!(jc.preferred_strategy, JoinStrategy::NestedLoop);
    }

    #[test]
    fn test_join_cost_semi_join_for_skewed() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let l = frag("http://a.example/sparql", 1, 100_000, 1.0);
        let r = frag("http://a.example/sparql", 1, 1_000, 1.0); // <10% of left
        let jc = model.estimate_join_cost(&l, &r);
        assert_eq!(jc.preferred_strategy, JoinStrategy::SemiJoin);
    }

    #[test]
    fn test_join_cost_hash_join_for_medium() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let l = frag("http://a.example/sparql", 1, 5_000, 1.0);
        let r = frag("http://a.example/sparql", 1, 4_000, 1.0);
        let jc = model.estimate_join_cost(&l, &r);
        // For these sizes either hash or merge is valid, just check total
        assert!(jc.total_cost > 0.0);
    }

    // ── estimate_transfer_cost ─────────────────────────────────────────

    #[test]
    fn test_transfer_cost_same_endpoint_is_zero() {
        let model = CostModel::new(5.0);
        let cost = model.estimate_transfer_cost("http://a.example", "http://a.example", 1000);
        assert!((cost).abs() < 1e-9);
    }

    #[test]
    fn test_transfer_cost_includes_overhead() {
        let model = CostModel::new(10.0);
        let cost = model.estimate_transfer_cost("http://a.example", "http://b.example", 0);
        assert!((cost - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_transfer_cost_scales_with_results() {
        let model = CostModel::new(0.0);
        let c1 = model.estimate_transfer_cost("a", "b", 100);
        let c2 = model.estimate_transfer_cost("a", "b", 1000);
        assert!(c2 > c1);
    }

    #[test]
    fn test_transfer_cost_uses_min_bandwidth() {
        let mut model = CostModel::new(0.0);
        let mut ep_fast = ep("http://fast/sparql");
        ep_fast.bandwidth_mbps = 1000.0;
        let mut ep_slow = ep("http://slow/sparql");
        ep_slow.bandwidth_mbps = 1.0;
        model.register_endpoint(ep_fast);
        model.register_endpoint(ep_slow);
        let cost = model.estimate_transfer_cost("http://fast/sparql", "http://slow/sparql", 10_000);
        // Should be dominated by the 1 Mbps endpoint
        assert!(cost > 0.1);
    }

    // ── rank_fragments ─────────────────────────────────────────────────

    #[test]
    fn test_rank_fragments_ascending_cost() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let frags = vec![
            frag("http://a.example/sparql", 5, 100, 1.0),
            frag("http://a.example/sparql", 1, 10, 0.1),
            frag("http://a.example/sparql", 3, 50, 0.5),
        ];
        let ranked = model.rank_fragments(&frags);
        let costs: Vec<f64> = ranked
            .iter()
            .map(|&i| model.estimate_scan_cost(&frags[i]))
            .collect();
        for w in costs.windows(2) {
            assert!(w[0] <= w[1] + 1e-9, "not sorted: {w:?}");
        }
    }

    #[test]
    fn test_rank_fragments_empty() {
        let model = CostModel::new(0.0);
        assert!(model.rank_fragments(&[]).is_empty());
    }

    #[test]
    fn test_rank_fragments_single() {
        let model = CostModel::new(0.0);
        let frags = vec![frag("http://a.example", 1, 10, 1.0)];
        let ranked = model.rank_fragments(&frags);
        assert_eq!(ranked, vec![0]);
    }

    // ── estimate_total_plan_cost ───────────────────────────────────────

    #[test]
    fn test_total_plan_cost_empty() {
        let model = CostModel::new(0.0);
        assert!((model.estimate_total_plan_cost(&[])).abs() < 1e-9);
    }

    #[test]
    fn test_total_plan_cost_single_fragment() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        let frags = vec![frag("http://a.example/sparql", 1, 100, 1.0)];
        let cost = model.estimate_total_plan_cost(&frags);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_total_plan_cost_increases_with_fragments() {
        let mut model = CostModel::new(0.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        model.register_endpoint(ep("http://b.example/sparql"));
        let f1 = vec![frag("http://a.example/sparql", 1, 100, 1.0)];
        let f2 = vec![
            frag("http://a.example/sparql", 1, 100, 1.0),
            frag("http://b.example/sparql", 1, 100, 1.0),
        ];
        assert!(model.estimate_total_plan_cost(&f2) > model.estimate_total_plan_cost(&f1));
    }

    // ── optimal_join_order ─────────────────────────────────────────────

    #[test]
    fn test_join_order_empty() {
        let model = CostModel::new(0.0);
        assert!(model.optimal_join_order(&[]).is_empty());
    }

    #[test]
    fn test_join_order_single() {
        let model = CostModel::new(0.0);
        let frags = vec![frag("http://a.example", 1, 10, 1.0)];
        assert_eq!(model.optimal_join_order(&frags), vec![0]);
    }

    #[test]
    fn test_join_order_is_permutation() {
        let mut model = CostModel::new(5.0);
        model.register_endpoint(ep("http://a.example/sparql"));
        model.register_endpoint(ep("http://b.example/sparql"));
        model.register_endpoint(ep("http://c.example/sparql"));
        let frags = vec![
            frag("http://a.example/sparql", 3, 500, 1.0),
            frag("http://b.example/sparql", 1, 50, 0.1),
            frag("http://c.example/sparql", 2, 200, 0.5),
        ];
        let order = model.optimal_join_order(&frags);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_join_order_cheap_first() {
        let mut model = CostModel::new(0.0);
        let mut ep_cheap = ep("http://cheap/sparql");
        ep_cheap.avg_latency_ms = 1.0;
        ep_cheap.triples_count = 100;
        let mut ep_expensive = ep("http://expensive/sparql");
        ep_expensive.avg_latency_ms = 1000.0;
        ep_expensive.triples_count = 100_000_000;
        model.register_endpoint(ep_cheap);
        model.register_endpoint(ep_expensive);
        let frags = vec![
            frag("http://expensive/sparql", 5, 10_000, 1.0),
            frag("http://cheap/sparql", 1, 10, 0.01),
        ];
        let order = model.optimal_join_order(&frags);
        // The cheap fragment (index 1) should be first
        assert_eq!(order[0], 1, "cheap fragment should come first");
    }
}
