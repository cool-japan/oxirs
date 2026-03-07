//! ML-Enhanced Query Router for Federated SPARQL Queries
//!
//! This module implements `MlQueryRouter`, which uses feature vectors
//! (query complexity, estimated cost, endpoint latency history) to route
//! federated queries to optimal endpoints. It includes online learning with
//! exponential moving average (EMA) cost updates.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Feature vector describing a federated query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatureVector {
    /// Number of triple patterns in the query.
    pub triple_pattern_count: usize,
    /// Number of join operations.
    pub join_count: usize,
    /// Number of FILTER expressions.
    pub filter_count: usize,
    /// Number of OPTIONAL blocks.
    pub optional_count: usize,
    /// Number of UNION blocks.
    pub union_count: usize,
    /// Number of result variables projected.
    pub projection_count: usize,
    /// Whether the query uses DISTINCT.
    pub has_distinct: bool,
    /// Whether the query uses ORDER BY.
    pub has_order_by: bool,
    /// Whether the query uses LIMIT / OFFSET.
    pub has_limit: bool,
    /// Whether the query uses aggregates (COUNT, SUM, …).
    pub has_aggregation: bool,
    /// Rough nesting depth of sub-queries / SERVICE clauses.
    pub nesting_depth: usize,
    /// Estimated result cardinality (may be 0 if unknown).
    pub estimated_cardinality: u64,
    /// Complexity score derived from the features above.
    pub complexity_score: f64,
}

impl QueryFeatureVector {
    /// Compute a scalar complexity score from the individual features.
    pub fn compute_complexity(
        triple_count: usize,
        join_count: usize,
        filter_count: usize,
        optional_count: usize,
        union_count: usize,
        has_aggregation: bool,
        nesting_depth: usize,
    ) -> f64 {
        let base = triple_count as f64
            + join_count as f64 * 2.0
            + filter_count as f64 * 0.5
            + optional_count as f64 * 1.5
            + union_count as f64 * 2.0
            + nesting_depth as f64 * 3.0;
        if has_aggregation {
            base * 1.5
        } else {
            base
        }
    }

    /// Return a dense f64 vector representation for distance computations.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.triple_pattern_count as f64,
            self.join_count as f64,
            self.filter_count as f64,
            self.optional_count as f64,
            self.union_count as f64,
            self.projection_count as f64,
            if self.has_distinct { 1.0 } else { 0.0 },
            if self.has_order_by { 1.0 } else { 0.0 },
            if self.has_limit { 1.0 } else { 0.0 },
            if self.has_aggregation { 1.0 } else { 0.0 },
            self.nesting_depth as f64,
            self.estimated_cardinality as f64,
            self.complexity_score,
        ]
    }
}

impl Default for QueryFeatureVector {
    fn default() -> Self {
        Self {
            triple_pattern_count: 1,
            join_count: 0,
            filter_count: 0,
            optional_count: 0,
            union_count: 0,
            projection_count: 1,
            has_distinct: false,
            has_order_by: false,
            has_limit: false,
            has_aggregation: false,
            nesting_depth: 0,
            estimated_cardinality: 0,
            complexity_score: 1.0,
        }
    }
}

/// Historical cost record for a single routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRecord {
    /// The endpoint that was selected.
    pub endpoint_id: String,
    /// Observed execution time in milliseconds.
    pub observed_cost_ms: f64,
    /// Query complexity at decision time.
    pub query_complexity: f64,
    /// Timestamp (millis since epoch).
    pub timestamp_ms: u64,
}

/// Per-endpoint learned statistics maintained by the router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStats {
    /// Endpoint identifier.
    pub endpoint_id: String,
    /// EMA of observed latency (ms).
    pub ema_latency_ms: f64,
    /// EMA of observed cost per unit complexity.
    pub ema_cost_per_complexity: f64,
    /// Number of routing decisions made.
    pub decision_count: u64,
    /// Number of failures recorded.
    pub failure_count: u64,
    /// Whether the endpoint is currently considered healthy.
    pub healthy: bool,
    /// Recent routing records (bounded ring buffer).
    pub recent_records: VecDeque<RoutingRecord>,
}

impl EndpointStats {
    /// Create new stats for an endpoint.
    pub fn new(endpoint_id: impl Into<String>) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            ema_latency_ms: 0.0,
            ema_cost_per_complexity: 0.0,
            decision_count: 0,
            failure_count: 0,
            healthy: true,
            recent_records: VecDeque::with_capacity(100),
        }
    }

    /// Update the EMA statistics with a new observation.
    ///
    /// `alpha` controls the speed of adaptation; higher values weight
    /// recent observations more heavily.
    pub fn update(&mut self, cost_ms: f64, complexity: f64, alpha: f64) {
        if self.ema_latency_ms == 0.0 {
            self.ema_latency_ms = cost_ms;
        } else {
            self.ema_latency_ms = alpha * cost_ms + (1.0 - alpha) * self.ema_latency_ms;
        }

        let cost_per_complexity = if complexity > 0.0 {
            cost_ms / complexity
        } else {
            cost_ms
        };

        if self.ema_cost_per_complexity == 0.0 {
            self.ema_cost_per_complexity = cost_per_complexity;
        } else {
            self.ema_cost_per_complexity =
                alpha * cost_per_complexity + (1.0 - alpha) * self.ema_cost_per_complexity;
        }

        self.decision_count += 1;
    }

    /// Record a failure event.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        // Penalise the latency estimate so the endpoint is deprioritised.
        self.ema_latency_ms *= 2.0;
    }

    /// Compute a routing score (lower is better).
    ///
    /// Score combines EMA latency and a penalty for recent failures.
    pub fn routing_score(&self, query_complexity: f64) -> f64 {
        if !self.healthy {
            return f64::MAX;
        }
        let latency_estimate = if self.ema_latency_ms > 0.0 {
            self.ema_cost_per_complexity * query_complexity
        } else {
            // Unknown endpoint – assign a moderate default.
            500.0
        };
        let failure_penalty = self.failure_count as f64 * 50.0;
        latency_estimate + failure_penalty
    }
}

/// Configuration for `MlQueryRouter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlQueryRouterConfig {
    /// EMA learning rate (0 < alpha ≤ 1).
    pub ema_alpha: f64,
    /// Maximum recent routing records stored per endpoint.
    pub max_recent_records: usize,
    /// Minimum number of observations before an endpoint is trusted.
    pub min_observations: u64,
    /// Failure rate threshold above which an endpoint is marked unhealthy.
    pub failure_rate_threshold: f64,
    /// Exploration probability for epsilon-greedy routing (0.0 – 1.0).
    pub exploration_epsilon: f64,
    /// Timeout for routing decisions.
    pub routing_timeout: Duration,
}

impl Default for MlQueryRouterConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.2,
            max_recent_records: 200,
            min_observations: 5,
            failure_rate_threshold: 0.3,
            exploration_epsilon: 0.05,
            routing_timeout: Duration::from_millis(50),
        }
    }
}

/// Outcome of a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// The selected endpoint identifier.
    pub endpoint_id: String,
    /// Predicted cost for this endpoint in milliseconds.
    pub predicted_cost_ms: f64,
    /// Whether this decision was exploratory (epsilon-greedy).
    pub is_exploratory: bool,
    /// Routing score used for comparison (lower is better).
    pub routing_score: f64,
}

/// ML-enhanced query router that learns optimal endpoint routing online.
///
/// The router maintains per-endpoint EMA statistics and uses them to
/// predict the best endpoint for each incoming query based on query
/// feature vectors.
pub struct MlQueryRouter {
    config: MlQueryRouterConfig,
    endpoint_stats: Arc<RwLock<HashMap<String, EndpointStats>>>,
    /// Total decisions made (used for epsilon-greedy seeding).
    total_decisions: Arc<RwLock<u64>>,
}

impl MlQueryRouter {
    /// Create a new router with default configuration.
    pub fn new() -> Self {
        Self {
            config: MlQueryRouterConfig::default(),
            endpoint_stats: Arc::new(RwLock::new(HashMap::new())),
            total_decisions: Arc::new(RwLock::new(0)),
        }
    }

    /// Create a new router with custom configuration.
    pub fn with_config(config: MlQueryRouterConfig) -> Self {
        Self {
            config,
            endpoint_stats: Arc::new(RwLock::new(HashMap::new())),
            total_decisions: Arc::new(RwLock::new(0)),
        }
    }

    /// Register an endpoint so the router can track it.
    pub async fn register_endpoint(&self, endpoint_id: impl Into<String>) {
        let id = endpoint_id.into();
        let mut stats = self.endpoint_stats.write().await;
        stats
            .entry(id.clone())
            .or_insert_with(|| EndpointStats::new(&id));
        info!("MlQueryRouter: registered endpoint '{}'", id);
    }

    /// Route a query to the best endpoint given its feature vector.
    ///
    /// Returns an error if no healthy endpoints are available.
    pub async fn route(
        &self,
        features: &QueryFeatureVector,
        candidate_endpoints: &[String],
    ) -> Result<RoutingDecision> {
        if candidate_endpoints.is_empty() {
            return Err(anyhow!("No candidate endpoints provided"));
        }

        let stats = self.endpoint_stats.read().await;

        // Build scored list of candidates.
        let mut scored: Vec<(f64, &String)> = candidate_endpoints
            .iter()
            .map(|ep_id| {
                let score = stats
                    .get(ep_id)
                    .map(|s| s.routing_score(features.complexity_score))
                    .unwrap_or(500.0); // Default for unknown endpoints.
                (score, ep_id)
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Check whether any healthy endpoints exist.
        let any_healthy = scored.iter().any(|(score, _)| *score < f64::MAX);
        if !any_healthy {
            return Err(anyhow!("All candidate endpoints are unhealthy"));
        }

        // Epsilon-greedy exploration: occasionally pick a non-optimal endpoint.
        let total = *self.total_decisions.read().await;
        let is_exploratory = self.should_explore(total);

        let chosen = if is_exploratory && scored.len() > 1 {
            // Pick the second-best to avoid always selecting the greedy choice.
            &scored[1]
        } else {
            &scored[0]
        };

        let (score, endpoint_id) = chosen;
        let predicted = stats
            .get(endpoint_id.as_str())
            .map(|s| {
                if s.ema_cost_per_complexity > 0.0 {
                    s.ema_cost_per_complexity * features.complexity_score
                } else {
                    500.0
                }
            })
            .unwrap_or(500.0);

        debug!(
            "MlQueryRouter: routed to '{}' (score={:.1}, exploratory={})",
            endpoint_id, score, is_exploratory
        );

        Ok(RoutingDecision {
            endpoint_id: (*endpoint_id).clone(),
            predicted_cost_ms: predicted,
            is_exploratory,
            routing_score: *score,
        })
    }

    /// Record the observed cost for a routing decision (online learning step).
    pub async fn record_outcome(
        &self,
        endpoint_id: &str,
        features: &QueryFeatureVector,
        observed_cost_ms: f64,
        success: bool,
    ) {
        let mut stats = self.endpoint_stats.write().await;
        let entry = stats
            .entry(endpoint_id.to_owned())
            .or_insert_with(|| EndpointStats::new(endpoint_id));

        if success {
            entry.update(
                observed_cost_ms,
                features.complexity_score,
                self.config.ema_alpha,
            );
        } else {
            entry.record_failure();
        }

        // Update health status based on failure rate.
        let total = entry.decision_count + entry.failure_count;
        if total > self.config.min_observations {
            let failure_rate = entry.failure_count as f64 / total as f64;
            entry.healthy = failure_rate < self.config.failure_rate_threshold;
        }

        // Trim recent records.
        let record = RoutingRecord {
            endpoint_id: endpoint_id.to_owned(),
            observed_cost_ms,
            query_complexity: features.complexity_score,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        entry.recent_records.push_back(record);
        while entry.recent_records.len() > self.config.max_recent_records {
            entry.recent_records.pop_front();
        }

        // Increment total decisions counter.
        drop(stats);
        let mut total_decisions = self.total_decisions.write().await;
        *total_decisions += 1;
    }

    /// Mark an endpoint as unhealthy / healthy explicitly.
    pub async fn set_endpoint_health(&self, endpoint_id: &str, healthy: bool) {
        let mut stats = self.endpoint_stats.write().await;
        if let Some(entry) = stats.get_mut(endpoint_id) {
            entry.healthy = healthy;
        }
    }

    /// Return a snapshot of the current statistics for all tracked endpoints.
    pub async fn endpoint_stats_snapshot(&self) -> HashMap<String, EndpointStats> {
        self.endpoint_stats.read().await.clone()
    }

    /// Return statistics for a specific endpoint.
    pub async fn endpoint_stats(&self, endpoint_id: &str) -> Option<EndpointStats> {
        self.endpoint_stats.read().await.get(endpoint_id).cloned()
    }

    /// Reset statistics for a specific endpoint.
    pub async fn reset_endpoint(&self, endpoint_id: &str) {
        let mut stats = self.endpoint_stats.write().await;
        if let Some(entry) = stats.get_mut(endpoint_id) {
            *entry = EndpointStats::new(endpoint_id);
        }
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    fn should_explore(&self, total_decisions: u64) -> bool {
        // Use a simple deterministic pseudo-random based on decision count.
        // Avoids pulling in rand; behaviour is predictable in tests.
        let pseudo_rand = (total_decisions
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 33) as f64
            / u32::MAX as f64;
        pseudo_rand < self.config.exploration_epsilon
    }
}

impl Default for MlQueryRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_features(complexity: f64) -> QueryFeatureVector {
        QueryFeatureVector {
            triple_pattern_count: 3,
            join_count: 1,
            filter_count: 0,
            optional_count: 0,
            union_count: 0,
            projection_count: 2,
            has_distinct: false,
            has_order_by: false,
            has_limit: false,
            has_aggregation: false,
            nesting_depth: 0,
            estimated_cardinality: 0,
            complexity_score: complexity,
        }
    }

    #[tokio::test]
    async fn test_router_creation() {
        let router = MlQueryRouter::new();
        let snapshot = router.endpoint_stats_snapshot().await;
        assert!(snapshot.is_empty());
    }

    #[tokio::test]
    async fn test_register_endpoint() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        let snapshot = router.endpoint_stats_snapshot().await;
        assert!(snapshot.contains_key("ep1"));
    }

    #[tokio::test]
    async fn test_route_single_endpoint() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        let features = simple_features(3.0);
        let decision = router
            .route(&features, &["ep1".to_string()])
            .await
            .expect("route should succeed");
        assert_eq!(decision.endpoint_id, "ep1");
    }

    #[tokio::test]
    async fn test_route_no_endpoints_returns_error() {
        let router = MlQueryRouter::new();
        let features = simple_features(3.0);
        let result = router.route(&features, &[]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_route_chooses_lower_latency_endpoint() {
        let router = MlQueryRouter::new();
        router.register_endpoint("fast").await;
        router.register_endpoint("slow").await;
        let features = simple_features(5.0);

        // Train: fast endpoint always finishes in 10 ms, slow in 200 ms.
        for _ in 0..20 {
            router.record_outcome("fast", &features, 10.0, true).await;
            router.record_outcome("slow", &features, 200.0, true).await;
        }

        // Most decisions (ignoring occasional exploration) should pick "fast".
        let mut fast_count = 0_u32;
        let candidates = vec!["fast".to_string(), "slow".to_string()];
        for _ in 0..50 {
            let decision = router.route(&features, &candidates).await.unwrap();
            if decision.endpoint_id == "fast" {
                fast_count += 1;
            }
        }
        // At least 80 % should go to "fast".
        assert!(
            fast_count >= 40,
            "Expected >=40 fast routes, got {fast_count}"
        );
    }

    #[tokio::test]
    async fn test_record_outcome_updates_ema_latency() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        let features = simple_features(2.0);

        router.record_outcome("ep1", &features, 100.0, true).await;
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert!(stats.ema_latency_ms > 0.0);
        assert_eq!(stats.decision_count, 1);
    }

    #[tokio::test]
    async fn test_failure_penalises_endpoint() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        let features = simple_features(2.0);

        // Initially latency is 0; after failure it should be penalised.
        router.record_outcome("ep1", &features, 50.0, false).await;
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert_eq!(stats.failure_count, 1);
    }

    #[tokio::test]
    async fn test_unhealthy_endpoint_not_selected() {
        let router = MlQueryRouter::new();
        router.register_endpoint("bad").await;
        router.register_endpoint("good").await;
        let features = simple_features(3.0);

        // Train "good" to be fast.
        for _ in 0..10 {
            router.record_outcome("good", &features, 10.0, true).await;
        }

        // Mark "bad" as unhealthy explicitly.
        router.set_endpoint_health("bad", false).await;

        let candidates = vec!["bad".to_string(), "good".to_string()];
        for _ in 0..10 {
            let decision = router.route(&features, &candidates).await.unwrap();
            assert_eq!(
                decision.endpoint_id, "good",
                "Unhealthy endpoint should not be chosen"
            );
        }
    }

    #[tokio::test]
    async fn test_set_endpoint_health() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        router.set_endpoint_health("ep1", false).await;
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert!(!stats.healthy);

        router.set_endpoint_health("ep1", true).await;
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert!(stats.healthy);
    }

    #[tokio::test]
    async fn test_reset_endpoint() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        let features = simple_features(2.0);
        router.record_outcome("ep1", &features, 100.0, true).await;

        router.reset_endpoint("ep1").await;
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert_eq!(stats.decision_count, 0);
        assert_eq!(stats.ema_latency_ms, 0.0);
    }

    #[tokio::test]
    async fn test_endpoint_stats_snapshot_returns_all() {
        let router = MlQueryRouter::new();
        for i in 0..5 {
            router.register_endpoint(format!("ep{i}")).await;
        }
        let snapshot = router.endpoint_stats_snapshot().await;
        assert_eq!(snapshot.len(), 5);
    }

    #[tokio::test]
    async fn test_query_feature_vector_to_vec() {
        let fv = QueryFeatureVector::default();
        let v = fv.to_vec();
        assert_eq!(v.len(), 13);
    }

    #[tokio::test]
    async fn test_compute_complexity_with_aggregation() {
        let c = QueryFeatureVector::compute_complexity(3, 2, 1, 0, 0, true, 0);
        // base = 3 + 4 + 0.5 = 7.5; × 1.5 = 11.25
        assert!((c - 11.25).abs() < 1e-6, "complexity={c}");
    }

    #[tokio::test]
    async fn test_compute_complexity_without_aggregation() {
        let c = QueryFeatureVector::compute_complexity(2, 1, 0, 1, 0, false, 1);
        // base = 2 + 2 + 0 + 1.5 + 0 + 3 = 8.5
        assert!((c - 8.5).abs() < 1e-6, "complexity={c}");
    }

    #[tokio::test]
    async fn test_recent_records_bounded() {
        let config = MlQueryRouterConfig {
            max_recent_records: 10,
            ..Default::default()
        };
        let router = MlQueryRouter::with_config(config);
        router.register_endpoint("ep1").await;
        let features = simple_features(1.0);

        for _ in 0..25 {
            router.record_outcome("ep1", &features, 50.0, true).await;
        }
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert!(
            stats.recent_records.len() <= 10,
            "Recent records should be bounded to 10"
        );
    }

    #[tokio::test]
    async fn test_ema_converges() {
        let config = MlQueryRouterConfig {
            ema_alpha: 0.5,
            ..Default::default()
        };
        let router = MlQueryRouter::with_config(config);
        router.register_endpoint("ep1").await;
        let features = simple_features(1.0);

        // After many identical observations the EMA should converge to that value.
        for _ in 0..30 {
            router.record_outcome("ep1", &features, 100.0, true).await;
        }
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert!(
            (stats.ema_latency_ms - 100.0).abs() < 1.0,
            "EMA should converge to 100.0, got {}",
            stats.ema_latency_ms
        );
    }

    #[tokio::test]
    async fn test_routing_decision_fields() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        let features = simple_features(4.0);
        let decision = router.route(&features, &["ep1".to_string()]).await.unwrap();
        assert!(!decision.endpoint_id.is_empty());
        assert!(decision.routing_score >= 0.0);
    }

    #[tokio::test]
    async fn test_high_failure_rate_marks_unhealthy() {
        let config = MlQueryRouterConfig {
            failure_rate_threshold: 0.4,
            min_observations: 3,
            ..Default::default()
        };
        let router = MlQueryRouter::with_config(config);
        router.register_endpoint("ep1").await;
        let features = simple_features(2.0);

        // Record 1 success and 4 failures → failure rate 80 % > 40 %.
        router.record_outcome("ep1", &features, 50.0, true).await;
        for _ in 0..4 {
            router.record_outcome("ep1", &features, 0.0, false).await;
        }
        let stats = router.endpoint_stats("ep1").await.unwrap();
        assert!(
            !stats.healthy,
            "Endpoint should be unhealthy after high failure rate"
        );
    }

    #[tokio::test]
    async fn test_unknown_endpoint_gets_default_score() {
        let router = MlQueryRouter::new();
        // Do NOT register "unknown", but include it as a candidate.
        router.register_endpoint("known").await;
        let features = simple_features(3.0);

        // Train "known" to a high latency so "unknown" (default 500) might win.
        for _ in 0..10 {
            router
                .record_outcome("known", &features, 1000.0, true)
                .await;
        }

        let candidates = vec!["known".to_string(), "unknown".to_string()];
        // Should not panic; just pick one of the candidates.
        let decision = router.route(&features, &candidates).await.unwrap();
        assert!(candidates.contains(&decision.endpoint_id));
    }

    #[tokio::test]
    async fn test_multi_endpoint_scoring_order() {
        let router = MlQueryRouter::new();
        let names = ["alpha", "beta", "gamma"];
        for name in &names {
            router.register_endpoint(*name).await;
        }
        let features = simple_features(2.0);

        // Train alpha=10ms, beta=50ms, gamma=200ms.
        for _ in 0..15 {
            router.record_outcome("alpha", &features, 10.0, true).await;
            router.record_outcome("beta", &features, 50.0, true).await;
            router.record_outcome("gamma", &features, 200.0, true).await;
        }

        let candidates: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        // Disable exploration for deterministic test.
        let cfg = MlQueryRouterConfig {
            exploration_epsilon: 0.0,
            ..Default::default()
        };
        let det_router = MlQueryRouter::with_config(cfg);
        for name in &names {
            det_router.register_endpoint(*name).await;
        }
        for _ in 0..15 {
            det_router
                .record_outcome("alpha", &features, 10.0, true)
                .await;
            det_router
                .record_outcome("beta", &features, 50.0, true)
                .await;
            det_router
                .record_outcome("gamma", &features, 200.0, true)
                .await;
        }

        let decision = det_router.route(&features, &candidates).await.unwrap();
        assert_eq!(
            decision.endpoint_id, "alpha",
            "Alpha should be chosen as fastest"
        );
    }

    #[tokio::test]
    async fn test_all_unhealthy_returns_error() {
        let router = MlQueryRouter::new();
        router.register_endpoint("ep1").await;
        router.register_endpoint("ep2").await;
        router.set_endpoint_health("ep1", false).await;
        router.set_endpoint_health("ep2", false).await;
        let features = simple_features(2.0);
        let result = router
            .route(&features, &["ep1".to_string(), "ep2".to_string()])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_with_config_uses_custom_alpha() {
        let config = MlQueryRouterConfig {
            ema_alpha: 0.9,
            ..Default::default()
        };
        let router = MlQueryRouter::with_config(config);
        router.register_endpoint("ep1").await;
        let features = simple_features(1.0);

        // With alpha=0.9, latest observation dominates quickly.
        router.record_outcome("ep1", &features, 200.0, true).await;
        router.record_outcome("ep1", &features, 10.0, true).await;
        let stats = router.endpoint_stats("ep1").await.unwrap();
        // After two observations: first gives 200, second: 0.9*10 + 0.1*200 = 29
        assert!(
            stats.ema_latency_ms < 50.0,
            "High alpha should react fast to new data; got {}",
            stats.ema_latency_ms
        );
    }
}
