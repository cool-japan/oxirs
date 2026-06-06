//! Adaptive SPARQL Federation Routing
//!
//! This module implements a production-quality adaptive routing engine for
//! federated SPARQL query execution.  It uses Exponential Weighted Moving
//! Average (EWMA) statistics to track per-endpoint performance and routes
//! query sub-tasks to the lowest-cost available endpoint at each invocation.
//!
//! # Architecture
//!
//! ```text
//! AdaptiveRoutingEngine
//!   ├─ HashMap<EndpointId, EndpointStats>  (live performance data)
//!   ├─ EndpointCostEstimator               (cost formula)
//!   └─ AdaptivePlanner                     (greedy assignment)
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use oxirs_federate::adaptive_routing::{AdaptiveRoutingEngine, AdaptiveRoutingConfig};
//! use oxirs_federate::adaptive_routing::query_planner::{FederatedQuery, SubQuery, TriplePattern};
//!
//! let config = AdaptiveRoutingConfig::default();
//! let mut engine = AdaptiveRoutingEngine::new(config);
//!
//! engine.register_endpoint("https://dbpedia.org/sparql".to_string());
//! engine.register_endpoint("https://wikidata.org/sparql".to_string());
//!
//! let query = FederatedQuery {
//!     sub_queries: vec![SubQuery {
//!         id: 0,
//!         triple_patterns: vec![TriplePattern::Subject("http://example.org/item".to_string())],
//!         estimated_cardinality: 42,
//!     }],
//!     optional_endpoints: Vec::new(),
//! };
//!
//! let plan = engine.route(&query);
//! for (endpoint, sub_query) in &plan.assignments {
//!     println!("SubQuery {} → {}", sub_query.id, endpoint);
//! }
//! ```

use std::collections::HashMap;

pub mod cost_model;
pub mod query_planner;
pub mod stats;
#[cfg(test)]
mod tests;

pub use cost_model::{EndpointCostEstimator, QueryCostFactors};
pub use query_planner::{AdaptivePlanner, FederatedQuery, RoutingPlan, SubQuery, TriplePattern};
pub use stats::EndpointStats;

/// Stable type alias for endpoint identifiers (plain IRI strings).
pub type EndpointId = String;

// ---------------------------------------------------------------------------
// Routing configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`AdaptiveRoutingEngine`].
#[derive(Debug, Clone)]
pub struct AdaptiveRoutingConfig {
    /// EWMA decay factor α ∈ (0, 1).
    ///
    /// Higher values weight recent observations more heavily.
    /// Default: 0.2 (20 % of each new observation blended in).
    pub alpha: f64,

    /// Maximum number of endpoints to consider per sub-query.
    ///
    /// If more are registered, only the top-N by current availability are
    /// considered.  Default: 5.
    pub max_endpoints_per_query: usize,

    /// Maximum cost above which the engine emits a warning and falls back.
    ///
    /// Expressed in the same units as [`EndpointCostEstimator::estimate_cost`].
    /// Default: 50_000.0.
    pub cost_threshold: f64,

    /// Additional latency penalty applied to endpoints that have not been seen
    /// recently.  Expressed in milliseconds.  Default: 500.0.
    pub latency_penalty_ms: f64,
}

impl Default for AdaptiveRoutingConfig {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            max_endpoints_per_query: 5,
            cost_threshold: 50_000.0,
            latency_penalty_ms: 500.0,
        }
    }
}

// Alias for backwards-compatibility / test visibility.
pub use AdaptiveRoutingConfig as RoutingConfig;

// ---------------------------------------------------------------------------
// Routing decision
// ---------------------------------------------------------------------------

/// The output of a routing call: chosen endpoints + estimated total cost.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The chosen routing plan.
    pub plan: RoutingPlan,
    /// Estimated total cost (sum of individual sub-query costs).
    pub estimated_total_cost: f64,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Main adaptive routing engine.
///
/// Owns the endpoint registry, live statistics, cost model, and planner.
/// Thread-safety is *not* built-in; wrap in an `Arc<Mutex<_>>` if shared
/// across threads.
#[derive(Debug)]
pub struct AdaptiveRoutingEngine {
    /// Configuration (alpha, thresholds, …).
    config: AdaptiveRoutingConfig,
    /// Per-endpoint statistics keyed by endpoint ID.
    stats: HashMap<EndpointId, EndpointStats>,
    /// Stateless cost estimator.
    estimator: EndpointCostEstimator,
    /// Planner built from `config`.
    planner: AdaptivePlanner,
}

impl AdaptiveRoutingEngine {
    /// Create a new engine with the supplied configuration.
    pub fn new(config: AdaptiveRoutingConfig) -> Self {
        let planner = AdaptivePlanner::new(config.clone());
        Self {
            config,
            stats: HashMap::new(),
            estimator: EndpointCostEstimator,
            planner,
        }
    }

    /// Register a new endpoint.  If the endpoint is already registered this is
    /// a no-op (existing statistics are preserved).
    pub fn register_endpoint(&mut self, endpoint_id: EndpointId) {
        self.stats.entry(endpoint_id).or_default();
    }

    /// Remove an endpoint from the registry.
    pub fn deregister_endpoint(&mut self, endpoint_id: &str) {
        self.stats.remove(endpoint_id);
    }

    /// Return an immutable reference to the statistics for `endpoint_id`.
    pub fn endpoint_stats(&self, endpoint_id: &str) -> Option<&EndpointStats> {
        self.stats.get(endpoint_id)
    }

    /// Record a successful query completion on `endpoint_id`.
    ///
    /// Updates EWMA latency and decays the error rate.
    pub fn record_success(&mut self, endpoint_id: &str, latency_ms: f64) {
        if let Some(s) = self.stats.get_mut(endpoint_id) {
            s.update_success(latency_ms, self.config.alpha);
        }
    }

    /// Record a query failure on `endpoint_id`.
    ///
    /// Increases EWMA error rate.
    pub fn record_failure(&mut self, endpoint_id: &str) {
        if let Some(s) = self.stats.get_mut(endpoint_id) {
            s.update_failure(self.config.alpha);
        }
    }

    /// Route a federated query and return the routing plan.
    ///
    /// Delegates to the internal [`AdaptivePlanner`] using the current live
    /// statistics.
    pub fn route(&self, query: &FederatedQuery) -> RoutingPlan {
        self.planner.plan(query, &self.stats, &self.estimator)
    }

    /// Route a federated query with fallback assignment.
    pub fn route_with_fallback(&self, query: &FederatedQuery) -> RoutingPlan {
        self.planner
            .plan_with_fallback(query, &self.stats, &self.estimator)
    }

    /// Route a federated query and return both the plan and an estimated
    /// total cost figure.
    pub fn route_with_decision(&self, query: &FederatedQuery) -> RoutingDecision {
        let plan = self.route(query);

        let estimated_total_cost: f64 = plan
            .assignments
            .iter()
            .map(|(ep, sq)| {
                let factors = sq.cost_factors();
                self.stats
                    .get(ep.as_str())
                    .map(|s| EndpointCostEstimator::estimate_cost(&factors, s))
                    .unwrap_or(0.0)
            })
            .sum();

        RoutingDecision {
            plan,
            estimated_total_cost,
        }
    }

    /// Return the number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.stats.len()
    }
}
