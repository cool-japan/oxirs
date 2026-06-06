//! Cost model for federated query routing.
//!
//! Estimates the execution cost of a query sub-task on a given SPARQL endpoint
//! and ranks a set of candidate endpoints from cheapest to most expensive.

use std::collections::HashMap;

use super::{stats::EndpointStats, EndpointId};

// ---------------------------------------------------------------------------
// Query cost factors
// ---------------------------------------------------------------------------

/// Factors used to estimate query cost on a specific endpoint.
#[derive(Debug, Clone)]
pub struct QueryCostFactors {
    /// Fraction of the graph this query is estimated to touch (0.0-1.0).
    /// Lower selectivity means fewer result rows, lower cost.
    pub selectivity: f64,

    /// Number of triple patterns in the BGP / sub-query.
    pub triple_pattern_count: usize,

    /// Hint for expected result cardinality (number of solution rows).
    pub result_cardinality_hint: u64,
}

impl Default for QueryCostFactors {
    fn default() -> Self {
        Self {
            selectivity: 0.5,
            triple_pattern_count: 1,
            result_cardinality_hint: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

/// Stateless estimator that computes expected query execution cost on a given
/// endpoint, combining query complexity with the endpoint's observed performance.
#[derive(Debug, Clone, Default)]
pub struct EndpointCostEstimator;

impl EndpointCostEstimator {
    /// Estimate the expected cost (in arbitrary units proportional to wall-clock
    /// milliseconds) of executing `factors` on an endpoint with `stats`.
    ///
    /// Formula:
    /// ```text
    /// cost = (patterns × selectivity + 1) × ewma_latency × (1 + error_rate × 10)
    /// ```
    ///
    /// The `+ 1` ensures a non-zero base cost even for a single-pattern query
    /// with zero selectivity.  The error-rate multiplier penalises unreliable
    /// endpoints heavily.
    pub fn estimate_cost(factors: &QueryCostFactors, stats: &EndpointStats) -> f64 {
        let complexity = factors.triple_pattern_count as f64 * factors.selectivity + 1.0;
        let reliability_penalty = 1.0 + stats.ewma_error_rate * 10.0;
        complexity * stats.ewma_latency_ms * reliability_penalty
    }

    /// Rank `endpoints` from lowest to highest estimated cost.
    ///
    /// Endpoints present in the registry but absent from `stats` are skipped.
    /// Returns a `Vec<(EndpointId, cost)>` sorted ascending by cost so callers
    /// can trivially pick the cheapest candidate with `ranked[0]`.
    pub fn rank_endpoints(
        endpoints: &[EndpointId],
        factors: &QueryCostFactors,
        stats: &HashMap<EndpointId, EndpointStats>,
    ) -> Vec<(EndpointId, f64)> {
        let mut ranked: Vec<(EndpointId, f64)> = endpoints
            .iter()
            .filter_map(|ep| {
                stats
                    .get(ep)
                    .map(|s| (ep.clone(), Self::estimate_cost(factors, s)))
            })
            .collect();

        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}
