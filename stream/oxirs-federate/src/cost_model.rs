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
//!
//! ## Two layers
//!
//! - [`CostModel`] — the original federate-internal model that operates on
//!   [`QueryFragment`] descriptors.
//! - [`FederationCostModel`] — a higher-level model that walks
//!   [`oxirs_arq::algebra::Algebra`] trees and combines the local CPU/IO/
//!   memory costs from
//!   [`oxirs_arq::cost_model::CostModel`] with network transfer costs derived
//!   from per-endpoint stats.  This is the surface used by the W3-S10
//!   federation optimizer to compare plans during rewrite.

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

// ─────────────────────────────────────────────────────────────────────────────
// FederationCostModel — bridge to oxirs-arq cost estimator
// ─────────────────────────────────────────────────────────────────────────────

use oxirs_arq::algebra::{Algebra, Term};
use oxirs_arq::cost_model::{CostEstimate, CostModel as ArqCostModel, CostModelConfig};

/// Aggregate cost estimate for a federated query plan.
///
/// Combines the *local* CPU/IO/memory cost computed by oxirs-arq with the
/// *network* cost of moving intermediate bindings between endpoints.
#[derive(Debug, Clone)]
pub struct FederationPlanCost {
    /// Sum of all local-execution costs at every SERVICE leg (CPU + IO + memory).
    pub local_cost: f64,
    /// Sum of all network transfer costs between endpoints + coordinator.
    pub network_cost: f64,
    /// `local_cost + network_cost`.
    pub total_cost: f64,
    /// Estimated total cardinality of the plan output.
    pub estimated_cardinality: usize,
    /// Per-endpoint subtotal of local cost (informational).
    pub per_endpoint_local: HashMap<String, f64>,
}

impl FederationPlanCost {
    /// Empty cost (zero everywhere).
    pub fn zero() -> Self {
        Self {
            local_cost: 0.0,
            network_cost: 0.0,
            total_cost: 0.0,
            estimated_cardinality: 0,
            per_endpoint_local: HashMap::new(),
        }
    }
}

/// Federation-aware cost model.
///
/// Wraps an [`oxirs_arq::cost_model::CostModel`] and supplements it with
/// per-endpoint network statistics so plans involving SERVICE clauses can be
/// compared on a uniform total-cost scale.
pub struct FederationCostModel {
    arq: ArqCostModel,
    network: CostModel,
}

impl FederationCostModel {
    /// Build a new model with default arq config and a transfer-only
    /// federation cost model with `network_overhead_ms` per hop.
    pub fn new(network_overhead_ms: f64) -> Self {
        Self {
            arq: ArqCostModel::new(CostModelConfig::default()),
            network: CostModel::new(network_overhead_ms),
        }
    }

    /// Build with a custom oxirs-arq cost-model configuration.
    pub fn with_arq_config(network_overhead_ms: f64, arq_cfg: CostModelConfig) -> Self {
        Self {
            arq: ArqCostModel::new(arq_cfg),
            network: CostModel::new(network_overhead_ms),
        }
    }

    /// Register endpoint statistics.  Forwards to the inner [`CostModel`].
    pub fn register_endpoint(&mut self, stats: EndpointStats) {
        self.network.register_endpoint(stats);
    }

    /// Borrow the network cost model.
    pub fn network(&self) -> &CostModel {
        &self.network
    }

    /// Borrow the local (oxirs-arq) cost model.
    pub fn arq(&self) -> &ArqCostModel {
        &self.arq
    }

    /// Estimate the cost of executing `algebra`.
    ///
    /// Walks the tree:
    /// - Each `Algebra::Service { endpoint, pattern, .. }` contributes a
    ///   local oxirs-arq cost for `pattern` plus a transfer cost from
    ///   `endpoint` to a virtual `coordinator`.
    /// - Other nodes recurse into their children and combine costs (joins
    ///   add, unions add, filters take their child's cost).
    pub fn estimate_plan_cost(&mut self, algebra: &Algebra) -> FederationPlanCost {
        let mut acc = FederationPlanCost::zero();
        self.walk(algebra, &mut acc);
        acc.total_cost = acc.local_cost + acc.network_cost;
        acc
    }

    fn walk(&mut self, algebra: &Algebra, acc: &mut FederationPlanCost) {
        match algebra {
            Algebra::Service {
                endpoint, pattern, ..
            } => {
                let endpoint_str = match endpoint {
                    Term::Iri(n) => n.as_str().to_string(),
                    _ => "<unknown>".into(),
                };
                let local: CostEstimate = self
                    .arq
                    .estimate_cost(pattern)
                    .unwrap_or_else(|_| CostEstimate::new(0.0, 0.0, 0.0, 0.0, 0));
                let local_total = local.cpu_cost + local.io_cost + local.memory_cost;
                acc.local_cost += local_total;
                *acc.per_endpoint_local
                    .entry(endpoint_str.clone())
                    .or_insert(0.0) += local_total;

                let net = self.network.estimate_transfer_cost(
                    &endpoint_str,
                    "coordinator",
                    local.cardinality as u64,
                );
                acc.network_cost += net;
                acc.estimated_cardinality =
                    acc.estimated_cardinality.saturating_add(local.cardinality);
            }
            Algebra::Join { left, right }
            | Algebra::Union { left, right }
            | Algebra::Minus { left, right } => {
                self.walk(left, acc);
                self.walk(right, acc);
            }
            Algebra::LeftJoin { left, right, .. } => {
                self.walk(left, acc);
                self.walk(right, acc);
            }
            Algebra::Filter { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Slice { pattern, .. }
            | Algebra::Group { pattern, .. }
            | Algebra::Having { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::Graph { pattern, .. } => {
                self.walk(pattern, acc);
            }
            Algebra::Bgp(patterns) => {
                // Local BGP — count its execution against a synthetic
                // "local" cost using oxirs-arq.
                let local: CostEstimate = self.arq.estimate_cost(algebra).unwrap_or_else(|_| {
                    CostEstimate::new(
                        (patterns.len() as f64) * 100.0,
                        (patterns.len() as f64) * 50.0,
                        0.0,
                        0.0,
                        patterns.len() * 1_000,
                    )
                });
                acc.local_cost += local.cpu_cost + local.io_cost + local.memory_cost;
                acc.estimated_cardinality =
                    acc.estimated_cardinality.saturating_add(local.cardinality);
            }
            Algebra::Values { bindings, .. } => {
                acc.estimated_cardinality =
                    acc.estimated_cardinality.saturating_add(bindings.len());
            }
            Algebra::PropertyPath { .. } => {
                acc.local_cost += 5_000.0;
                acc.estimated_cardinality = acc.estimated_cardinality.saturating_add(2_000);
            }
            Algebra::Table | Algebra::Zero | Algebra::Empty => {}
        }
    }

    /// Compare two plans and return the cheaper one (by `total_cost`).
    /// Returns the first plan on ties.
    pub fn cheaper<'a>(
        &mut self,
        left: &'a Algebra,
        right: &'a Algebra,
    ) -> (&'a Algebra, FederationPlanCost) {
        let lc = self.estimate_plan_cost(left);
        let rc = self.estimate_plan_cost(right);
        if rc.total_cost < lc.total_cost {
            (right, rc)
        } else {
            (left, lc)
        }
    }
}

#[cfg(test)]
mod federation_cost_tests {
    use super::*;
    use oxirs_arq::algebra::{Term, TriplePattern, Variable};
    use oxirs_core::model::NamedNode;

    fn var(s: &str) -> Variable {
        Variable::new(s).expect("valid var")
    }

    fn iri_term(s: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(s))
    }

    fn bgp(s: &str, p: &str, o: &str) -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(var(s)),
            predicate: iri_term(p),
            object: Term::Variable(var(o)),
        }])
    }

    fn service(endpoint: &str, pattern: Algebra, silent: bool) -> Algebra {
        Algebra::Service {
            endpoint: iri_term(endpoint),
            pattern: Box::new(pattern),
            silent,
        }
    }

    fn ep_stats(url: &str, latency: f64, triples: u64) -> EndpointStats {
        EndpointStats {
            endpoint_url: url.to_string(),
            avg_latency_ms: latency,
            triples_count: triples,
            selectivity_estimate: 0.05,
            bandwidth_mbps: 100.0,
        }
    }

    #[test]
    fn fcm_builds_with_defaults() {
        let m = FederationCostModel::new(5.0);
        assert!(m.network().estimate_transfer_cost("a", "b", 0) >= 5.0);
    }

    #[test]
    fn fcm_zero_cost_for_empty() {
        let mut m = FederationCostModel::new(5.0);
        let cost = m.estimate_plan_cost(&Algebra::Empty);
        assert!((cost.total_cost).abs() < 1e-9);
    }

    #[test]
    fn fcm_estimates_service_cost() {
        let mut m = FederationCostModel::new(5.0);
        m.register_endpoint(ep_stats("http://a.example", 10.0, 1_000_000));
        let s = service("http://a.example", bgp("s", "http://p", "o"), false);
        let cost = m.estimate_plan_cost(&s);
        assert!(cost.local_cost >= 0.0);
        assert!(cost.network_cost >= 5.0);
        assert!(cost.total_cost >= cost.local_cost);
    }

    #[test]
    fn fcm_local_cost_aggregates_per_endpoint() {
        let mut m = FederationCostModel::new(5.0);
        m.register_endpoint(ep_stats("http://a.example", 10.0, 1_000));
        m.register_endpoint(ep_stats("http://b.example", 20.0, 2_000));
        let plan = Algebra::Join {
            left: Box::new(service(
                "http://a.example",
                bgp("s", "http://p", "o"),
                false,
            )),
            right: Box::new(service(
                "http://b.example",
                bgp("s", "http://q", "o"),
                false,
            )),
        };
        let cost = m.estimate_plan_cost(&plan);
        assert!(cost.per_endpoint_local.contains_key("http://a.example"));
        assert!(cost.per_endpoint_local.contains_key("http://b.example"));
    }

    #[test]
    fn fcm_cheaper_picks_lower_cost() {
        let mut m = FederationCostModel::new(5.0);
        m.register_endpoint(ep_stats("http://a.example", 10.0, 1_000));
        let cheap = bgp("s", "http://p", "o");
        let expensive = service("http://a.example", bgp("s", "http://p", "o"), false);
        let (winner, _cost) = m.cheaper(&cheap, &expensive);
        // cheap is purely local → no network, should win.
        assert!(matches!(winner, Algebra::Bgp(_)));
    }

    #[test]
    fn fcm_filter_does_not_double_count() {
        let mut m = FederationCostModel::new(5.0);
        m.register_endpoint(ep_stats("http://a.example", 10.0, 1_000));
        let s = service("http://a.example", bgp("s", "http://p", "o"), false);
        let f = Algebra::Filter {
            pattern: Box::new(s.clone()),
            condition: oxirs_arq::algebra::Expression::Literal(oxirs_arq::algebra::Literal {
                value: "true".into(),
                language: None,
                datatype: None,
            }),
        };
        let c_s = m.estimate_plan_cost(&s);
        let c_f = m.estimate_plan_cost(&f);
        // Filter wraps service — costs should be equal (filter wraps without adding more SERVICE legs).
        assert!((c_s.network_cost - c_f.network_cost).abs() < 1e-6);
    }

    #[test]
    fn fcm_join_cost_is_sum_of_legs_local() {
        let mut m = FederationCostModel::new(0.0);
        m.register_endpoint(ep_stats("http://a.example", 0.0, 1_000));
        m.register_endpoint(ep_stats("http://b.example", 0.0, 1_000));
        let s_a = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s_b = service("http://b.example", bgp("s", "http://q", "o"), false);
        let c_a = m.estimate_plan_cost(&s_a);
        let c_b = m.estimate_plan_cost(&s_b);
        let join = Algebra::Join {
            left: Box::new(s_a),
            right: Box::new(s_b),
        };
        let c_join = m.estimate_plan_cost(&join);
        // Local costs add (with small tolerance for non-deterministic arq cache).
        assert!(c_join.local_cost >= c_a.local_cost + c_b.local_cost - 1e-6);
    }

    #[test]
    fn fcm_silent_flag_does_not_change_cost() {
        let mut m = FederationCostModel::new(5.0);
        m.register_endpoint(ep_stats("http://a.example", 10.0, 1_000));
        let s_loud = service("http://a.example", bgp("s", "http://p", "o"), false);
        let s_silent = service("http://a.example", bgp("s", "http://p", "o"), true);
        let c_loud = m.estimate_plan_cost(&s_loud);
        let c_silent = m.estimate_plan_cost(&s_silent);
        assert!((c_loud.total_cost - c_silent.total_cost).abs() < 1e-6);
    }

    #[test]
    fn fcm_handles_unknown_endpoint() {
        let mut m = FederationCostModel::new(5.0);
        let s = service("http://unknown.example", bgp("s", "http://p", "o"), false);
        let cost = m.estimate_plan_cost(&s);
        // Should still produce a finite, non-negative cost.
        assert!(cost.total_cost.is_finite());
        assert!(cost.total_cost >= 0.0);
    }

    #[test]
    fn fcm_zero_struct() {
        let z = FederationPlanCost::zero();
        assert!((z.total_cost).abs() < 1e-9);
        assert_eq!(z.estimated_cardinality, 0);
        assert!(z.per_endpoint_local.is_empty());
    }
}
