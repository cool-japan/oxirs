//! Cost estimation for federated query execution.
//!
//! Tracks observed latencies for each SPARQL endpoint and uses these measurements,
//! together with structural properties of a subquery, to produce a numeric cost
//! estimate.  The estimate is used by [`super::optimizer::FederationOptimizer`] to
//! choose optimal join orderings and to decide whether subqueries should be merged.
//!
//! # Design
//!
//! The cost model is intentionally simple:
//!
//! ```text
//! cost(subquery) = latency_weight(endpoint) * complexity_weight(subquery)
//! ```
//!
//! where:
//! * `latency_weight` grows with the observed median latency for that endpoint.
//! * `complexity_weight` is derived from the estimated result cardinality and the
//!   number of unbound variables in the subquery.
//!
//! When no latency history is available for an endpoint, a configurable default
//! is used so that the estimator can still produce meaningful relative costs.

use std::collections::HashMap;
use std::time::Duration;

use super::decomposer::EndpointSubquery;

/// Configurable parameters for the cost model.
#[derive(Debug, Clone)]
pub struct CostModelConfig {
    /// Default latency assumed for endpoints with no measured history.
    pub default_latency: Duration,
    /// Weight applied to the latency component (ms → cost units).
    pub latency_weight_factor: f64,
    /// Weight applied per estimated result row.
    pub result_cardinality_factor: f64,
    /// Weight applied per unbound variable in the subquery.
    pub variable_complexity_factor: f64,
    /// Additional penalty per round-trip to a remote endpoint.
    pub network_round_trip_penalty: f64,
}

impl Default for CostModelConfig {
    fn default() -> Self {
        Self {
            default_latency: Duration::from_millis(200),
            latency_weight_factor: 0.05,
            result_cardinality_factor: 0.01,
            variable_complexity_factor: 5.0,
            network_round_trip_penalty: 10.0,
        }
    }
}

/// Tracks observed latencies per endpoint and estimates execution costs for
/// [`EndpointSubquery`] instances.
///
/// The estimator is intentionally non-async because cost estimation must be
/// fast and synchronous — it runs inside query planning, which is already async.
#[derive(Debug, Clone)]
pub struct CostEstimator {
    /// Map from endpoint URL to a ring-buffer of the most recently observed latencies.
    endpoint_latencies: HashMap<String, Vec<Duration>>,
    /// Maximum number of latency samples retained per endpoint.
    max_history_per_endpoint: usize,
    /// Cost model parameters.
    config: CostModelConfig,
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostEstimator {
    /// Create a new cost estimator with default configuration.
    pub fn new() -> Self {
        Self {
            endpoint_latencies: HashMap::new(),
            max_history_per_endpoint: 50,
            config: CostModelConfig::default(),
        }
    }

    /// Create a cost estimator with a custom cost model configuration.
    pub fn with_config(config: CostModelConfig) -> Self {
        Self {
            endpoint_latencies: HashMap::new(),
            max_history_per_endpoint: 50,
            config,
        }
    }

    /// Record an observed latency for an endpoint.
    ///
    /// Old samples are evicted in FIFO order once the history buffer is full.
    pub fn update_latency(&mut self, endpoint: &str, latency: Duration) {
        let history = self
            .endpoint_latencies
            .entry(endpoint.to_string())
            .or_default();

        history.push(latency);
        if history.len() > self.max_history_per_endpoint {
            history.remove(0);
        }
    }

    /// Retrieve the median observed latency for an endpoint.
    ///
    /// Returns the configured default latency when no samples are available.
    pub fn median_latency(&self, endpoint: &str) -> Duration {
        match self.endpoint_latencies.get(endpoint) {
            None => self.config.default_latency,
            Some(history) if history.is_empty() => self.config.default_latency,
            Some(history) => {
                let mut sorted: Vec<Duration> = history.clone();
                sorted.sort();
                let mid = sorted.len() / 2;
                sorted[mid]
            }
        }
    }

    /// Compute the numeric cost estimate for executing a single subquery at its endpoint.
    ///
    /// The cost is a dimensionless number suitable for comparison and ranking.
    pub fn estimate_cost(&self, subquery: &EndpointSubquery) -> f64 {
        let latency_ms = self.median_latency(&subquery.endpoint_url).as_millis() as f64;
        let latency_component = latency_ms * self.config.latency_weight_factor;

        let cardinality_component =
            subquery.estimated_results as f64 * self.config.result_cardinality_factor;

        let variable_count = count_unbound_variables(&subquery.sparql);
        let complexity_component = variable_count as f64 * self.config.variable_complexity_factor;

        let network_penalty = self.config.network_round_trip_penalty;

        latency_component + cardinality_component + complexity_component + network_penalty
    }

    /// Compute the estimated cost for joining two subquery result sets.
    ///
    /// Uses a simple product model scaled by a join penalty factor.
    pub fn estimate_join_cost(&self, left_cardinality: usize, right_cardinality: usize) -> f64 {
        let product = (left_cardinality as f64) * (right_cardinality as f64);
        // Log-linear model: avoid exploding costs for large cardinalities
        if product > 1.0 {
            product.ln() * 20.0
        } else {
            20.0
        }
    }

    /// Return the set of endpoints for which latency history has been recorded.
    pub fn known_endpoints(&self) -> Vec<&str> {
        self.endpoint_latencies.keys().map(|s| s.as_str()).collect()
    }

    /// Remove all stored latency history (useful in tests or after configuration reset).
    pub fn clear_history(&mut self) {
        self.endpoint_latencies.clear();
    }

    /// Return the number of latency samples stored for a given endpoint.
    pub fn sample_count(&self, endpoint: &str) -> usize {
        self.endpoint_latencies
            .get(endpoint)
            .map(|h| h.len())
            .unwrap_or(0)
    }
}

/// Count the number of SPARQL variable tokens (`?name` or `$name`) in a query string.
///
/// This provides a rough proxy for query complexity that is cheap to compute
/// without a full parse of the SPARQL grammar.
fn count_unbound_variables(sparql: &str) -> usize {
    let mut count = 0usize;
    let mut chars = sparql.chars().peekable();
    while let Some(ch) = chars.next() {
        if (ch == '?' || ch == '$')
            && chars
                .peek()
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
        {
            count += 1;
            // Consume the rest of the variable name
            while chars
                .peek()
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
            {
                chars.next();
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_rewrite::decomposer::EndpointSubquery;

    fn make_subquery(endpoint: &str, sparql: &str, estimated_results: usize) -> EndpointSubquery {
        EndpointSubquery {
            endpoint_url: endpoint.to_string(),
            sparql: sparql.to_string(),
            estimated_results,
            priority: 1.0,
        }
    }

    #[test]
    fn test_default_latency_when_no_history() {
        let estimator = CostEstimator::new();
        let latency = estimator.median_latency("http://unknown/sparql");
        assert_eq!(latency, Duration::from_millis(200));
    }

    #[test]
    fn test_update_latency_and_median() {
        let mut estimator = CostEstimator::new();
        let ep = "http://ep1/sparql";
        estimator.update_latency(ep, Duration::from_millis(100));
        estimator.update_latency(ep, Duration::from_millis(200));
        estimator.update_latency(ep, Duration::from_millis(150));
        // Sorted: [100, 150, 200], median index = 1 → 150ms
        assert_eq!(estimator.median_latency(ep), Duration::from_millis(150));
    }

    #[test]
    fn test_estimate_cost_positive() {
        let estimator = CostEstimator::new();
        let sq = make_subquery("http://ep1/sparql", "SELECT ?s ?p WHERE {?s ?p ?o}", 50);
        let cost = estimator.estimate_cost(&sq);
        assert!(cost > 0.0, "cost must be positive, got {cost}");
    }

    #[test]
    fn test_lower_latency_yields_lower_cost() {
        let mut estimator = CostEstimator::new();
        let ep_fast = "http://fast/sparql";
        let ep_slow = "http://slow/sparql";

        estimator.update_latency(ep_fast, Duration::from_millis(10));
        estimator.update_latency(ep_slow, Duration::from_millis(500));

        let sq_fast = make_subquery(ep_fast, "SELECT ?s WHERE {?s ?p ?o}", 10);
        let sq_slow = make_subquery(ep_slow, "SELECT ?s WHERE {?s ?p ?o}", 10);

        assert!(estimator.estimate_cost(&sq_fast) < estimator.estimate_cost(&sq_slow));
    }

    #[test]
    fn test_higher_cardinality_yields_higher_cost() {
        let estimator = CostEstimator::new();
        let sq_small = make_subquery("http://ep/sparql", "SELECT ?s WHERE {?s ?p ?o}", 10);
        let sq_large = make_subquery("http://ep/sparql", "SELECT ?s WHERE {?s ?p ?o}", 10_000);

        assert!(estimator.estimate_cost(&sq_large) > estimator.estimate_cost(&sq_small));
    }

    #[test]
    fn test_join_cost_positive_and_non_trivial() {
        let estimator = CostEstimator::new();
        let cost = estimator.estimate_join_cost(100, 200);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_join_cost_scales_with_cardinalities() {
        let estimator = CostEstimator::new();
        let small_join = estimator.estimate_join_cost(10, 10);
        let large_join = estimator.estimate_join_cost(1000, 1000);
        assert!(large_join > small_join);
    }

    #[test]
    fn test_history_eviction() {
        let mut estimator = CostEstimator::new();
        let ep = "http://ep/sparql";
        // Fill beyond max_history_per_endpoint (50)
        for i in 0..60u64 {
            estimator.update_latency(ep, Duration::from_millis(i));
        }
        assert_eq!(estimator.sample_count(ep), 50);
    }

    #[test]
    fn test_known_endpoints() {
        let mut estimator = CostEstimator::new();
        estimator.update_latency("http://a/sparql", Duration::from_millis(10));
        estimator.update_latency("http://b/sparql", Duration::from_millis(20));
        let endpoints = estimator.known_endpoints();
        assert_eq!(endpoints.len(), 2);
    }

    #[test]
    fn test_clear_history() {
        let mut estimator = CostEstimator::new();
        estimator.update_latency("http://ep/sparql", Duration::from_millis(50));
        estimator.clear_history();
        assert_eq!(estimator.sample_count("http://ep/sparql"), 0);
        // After clear, default latency should be returned
        assert_eq!(
            estimator.median_latency("http://ep/sparql"),
            Duration::from_millis(200)
        );
    }

    #[test]
    fn test_count_unbound_variables() {
        let sparql = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        // ?s, ?p, ?o appear twice each in WHERE but only once in SELECT — function counts all occurrences
        let count = super::count_unbound_variables(sparql);
        // SELECT ?s ?p ?o WHERE { ?s ?p ?o } → 6 variable tokens
        assert_eq!(count, 6);
    }

    #[test]
    fn test_count_variables_empty_query() {
        let count = super::count_unbound_variables("ASK {}");
        assert_eq!(count, 0);
    }
}
