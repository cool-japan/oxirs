//! Tests for the adaptive SPARQL federation routing module.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::collections::HashMap;

    use crate::adaptive_routing::{
        cost_model::{EndpointCostEstimator, QueryCostFactors},
        query_planner::{AdaptivePlanner, FederatedQuery, SubQuery, TriplePattern},
        stats::EndpointStats,
        AdaptiveRoutingConfig, AdaptiveRoutingEngine,
    };

    // -----------------------------------------------------------------------
    // EndpointStats — unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ewma_latency_update() {
        let mut stats = EndpointStats::new();
        let initial = stats.ewma_latency_ms;

        // After several updates with a very low latency the EWMA should
        // converge below the initial default.
        for _ in 0..20 {
            stats.update_success(10.0, 0.5);
        }
        assert!(
            stats.ewma_latency_ms < initial,
            "EWMA should have converged downwards: {} vs initial {}",
            stats.ewma_latency_ms,
            initial
        );
    }

    #[test]
    fn test_error_rate_update() {
        let mut stats = EndpointStats::new();
        assert_eq!(stats.ewma_error_rate, 0.0);

        // A series of failures should drive the error rate upward.
        for _ in 0..10 {
            stats.update_failure(0.5);
        }
        assert!(
            stats.ewma_error_rate > 0.0,
            "Error rate should be positive after failures"
        );

        // Recovery: successive successes should drive it back down.
        let rate_after_failures = stats.ewma_error_rate;
        for _ in 0..20 {
            stats.update_success(50.0, 0.5);
        }
        assert!(
            stats.ewma_error_rate < rate_after_failures,
            "Error rate should decline after successes"
        );
    }

    #[test]
    fn test_availability_score() {
        let mut stats = EndpointStats::new();
        stats.ewma_latency_ms = 0.0;
        stats.ewma_error_rate = 0.0;

        // With zero latency and zero error rate the score is
        // (1 - 0) / (1 + 0/1000) = 1.0.
        let score = stats.availability_score();
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect endpoint should score 1.0, got {score}"
        );

        // A degraded endpoint should score lower.
        stats.ewma_error_rate = 0.5;
        stats.ewma_latency_ms = 500.0;
        let degraded = stats.availability_score();
        assert!(
            degraded < 1.0,
            "Degraded endpoint should score below 1.0, got {degraded}"
        );
    }

    // -----------------------------------------------------------------------
    // EndpointCostEstimator — unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cost_estimate_basic() {
        let stats = EndpointStats {
            ewma_latency_ms: 100.0,
            ewma_error_rate: 0.0,
            ..EndpointStats::new()
        };
        let factors = QueryCostFactors {
            selectivity: 1.0,
            triple_pattern_count: 2,
            result_cardinality_hint: 50,
        };
        // cost = (2 * 1.0 + 1) * 100.0 * (1 + 0 * 10) = 3 * 100 = 300
        let cost = EndpointCostEstimator::estimate_cost(&factors, &stats);
        assert!((cost - 300.0).abs() < 1e-9, "Expected 300.0, got {cost}");
    }

    #[test]
    fn test_cost_estimate_high_latency() {
        let stats = EndpointStats {
            ewma_latency_ms: 10_000.0,
            ewma_error_rate: 0.0,
            ..EndpointStats::new()
        };
        let factors = QueryCostFactors {
            selectivity: 0.1,
            triple_pattern_count: 1,
            result_cardinality_hint: 10,
        };
        let cost = EndpointCostEstimator::estimate_cost(&factors, &stats);
        // Should be much larger than for a 100 ms endpoint
        assert!(
            cost > 1_000.0,
            "High-latency endpoint should have large cost, got {cost}"
        );
    }

    #[test]
    fn test_rank_endpoints_ordering() {
        let mut stats_map: HashMap<String, EndpointStats> = HashMap::new();

        // ep_a: high latency
        let mut ep_a = EndpointStats::new();
        ep_a.ewma_latency_ms = 500.0;
        ep_a.ewma_error_rate = 0.0;

        // ep_b: medium latency
        let mut ep_b = EndpointStats::new();
        ep_b.ewma_latency_ms = 200.0;
        ep_b.ewma_error_rate = 0.0;

        // ep_c: low latency
        let mut ep_c = EndpointStats::new();
        ep_c.ewma_latency_ms = 50.0;
        ep_c.ewma_error_rate = 0.0;

        stats_map.insert("ep_a".to_string(), ep_a);
        stats_map.insert("ep_b".to_string(), ep_b);
        stats_map.insert("ep_c".to_string(), ep_c);

        let endpoints: Vec<String> =
            vec!["ep_a".to_string(), "ep_b".to_string(), "ep_c".to_string()];
        let factors = QueryCostFactors::default();

        let ranked = EndpointCostEstimator::rank_endpoints(&endpoints, &factors, &stats_map);

        assert_eq!(ranked.len(), 3);
        // Cheapest should be ep_c (lowest latency)
        assert_eq!(ranked[0].0, "ep_c", "ep_c should be first");
        // Most expensive should be ep_a
        assert_eq!(ranked[2].0, "ep_a", "ep_a should be last");
        // Verify costs are ascending
        assert!(ranked[0].1 <= ranked[1].1);
        assert!(ranked[1].1 <= ranked[2].1);
    }

    // -----------------------------------------------------------------------
    // AdaptivePlanner — unit tests
    // -----------------------------------------------------------------------

    fn make_stats(latency_ms: f64, error_rate: f64) -> EndpointStats {
        EndpointStats {
            ewma_latency_ms: latency_ms,
            ewma_error_rate: error_rate,
            ..EndpointStats::new()
        }
    }

    fn two_endpoint_stats() -> HashMap<String, EndpointStats> {
        let mut m = HashMap::new();
        m.insert("fast".to_string(), make_stats(50.0, 0.0));
        m.insert("slow".to_string(), make_stats(500.0, 0.0));
        m
    }

    fn single_sub_query() -> SubQuery {
        SubQuery {
            id: 0,
            triple_patterns: vec![TriplePattern::Subject("http://example.org/s".to_string())],
            estimated_cardinality: 10,
        }
    }

    #[test]
    fn test_greedy_plan_single_query() {
        let config = AdaptiveRoutingConfig::default();
        let planner = AdaptivePlanner::new(config);
        let stats = two_endpoint_stats();
        let estimator = EndpointCostEstimator;
        let query = FederatedQuery {
            sub_queries: vec![single_sub_query()],
            optional_endpoints: Vec::new(),
        };

        let plan = planner.plan(&query, &stats, &estimator);

        assert_eq!(plan.assignments.len(), 1);
        // The fast endpoint should be chosen
        assert_eq!(plan.assignments[0].0, "fast");
    }

    #[test]
    fn test_greedy_plan_multi_query() {
        let config = AdaptiveRoutingConfig::default();
        let planner = AdaptivePlanner::new(config);

        let mut stats: HashMap<String, EndpointStats> = HashMap::new();
        stats.insert("ep1".to_string(), make_stats(50.0, 0.0));
        stats.insert("ep2".to_string(), make_stats(200.0, 0.0));
        stats.insert("ep3".to_string(), make_stats(1000.0, 0.0));

        let query = FederatedQuery {
            sub_queries: vec![
                SubQuery {
                    id: 0,
                    triple_patterns: vec![TriplePattern::Subject("s0".to_string())],
                    estimated_cardinality: 5,
                },
                SubQuery {
                    id: 1,
                    triple_patterns: vec![TriplePattern::Predicate("p1".to_string())],
                    estimated_cardinality: 20,
                },
                SubQuery {
                    id: 2,
                    triple_patterns: vec![TriplePattern::Object("o2".to_string())],
                    estimated_cardinality: 50,
                },
            ],
            optional_endpoints: Vec::new(),
        };

        let plan = planner.plan(&query, &stats, &EndpointCostEstimator);
        // All three sub-queries should get an assignment
        assert_eq!(plan.assignments.len(), 3);
        // Every assigned endpoint must be registered
        for (ep, _) in &plan.assignments {
            assert!(stats.contains_key(ep.as_str()), "Unknown endpoint: {ep}");
        }
    }

    #[test]
    fn test_plan_with_fallback() {
        let config = AdaptiveRoutingConfig::default();
        let planner = AdaptivePlanner::new(config);
        let stats = two_endpoint_stats();
        let query = FederatedQuery {
            sub_queries: vec![single_sub_query()],
            optional_endpoints: Vec::new(),
        };

        let plan = planner.plan_with_fallback(&query, &stats, &EndpointCostEstimator);

        assert_eq!(plan.assignments.len(), 1);
        // With two endpoints we should get a fallback
        assert!(
            plan.fallback_endpoint.is_some(),
            "Expected a fallback endpoint"
        );
    }

    #[test]
    fn test_availability_below_threshold() {
        let config = AdaptiveRoutingConfig::default();
        let planner = AdaptivePlanner::new(config);

        let mut stats: HashMap<String, EndpointStats> = HashMap::new();
        // ep_bad: 90 % error rate — availability below threshold
        stats.insert("ep_bad".to_string(), make_stats(100.0, 0.9));
        // ep_good: healthy
        stats.insert("ep_good".to_string(), make_stats(200.0, 0.0));

        let query = FederatedQuery {
            sub_queries: vec![single_sub_query()],
            optional_endpoints: Vec::new(),
        };

        let plan = planner.plan(&query, &stats, &EndpointCostEstimator);
        assert_eq!(plan.assignments.len(), 1);
        // ep_good should be preferred despite higher latency
        assert_eq!(plan.assignments[0].0, "ep_good");
    }

    // -----------------------------------------------------------------------
    // AdaptiveRoutingEngine — integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_routing_engine_register() {
        let config = AdaptiveRoutingConfig::default();
        let mut engine = AdaptiveRoutingEngine::new(config);

        engine.register_endpoint("ep1".to_string());

        let stats = engine.endpoint_stats("ep1");
        assert!(stats.is_some(), "ep1 should be registered");
    }

    #[test]
    fn test_routing_engine_route() {
        let config = AdaptiveRoutingConfig::default();
        let mut engine = AdaptiveRoutingEngine::new(config);

        engine.register_endpoint("ep_a".to_string());
        engine.register_endpoint("ep_b".to_string());

        // Prime ep_a with low latency
        engine.record_success("ep_a", 10.0);
        // Prime ep_b with high latency
        engine.record_success("ep_b", 900.0);

        let query = FederatedQuery {
            sub_queries: vec![single_sub_query()],
            optional_endpoints: Vec::new(),
        };

        let plan = engine.route(&query);
        assert_eq!(plan.assignments.len(), 1);
    }

    #[test]
    fn test_routing_engine_update_success() {
        let config = AdaptiveRoutingConfig::default();
        let mut engine = AdaptiveRoutingEngine::new(config);

        engine.register_endpoint("ep1".to_string());
        let initial_latency = engine.endpoint_stats("ep1").unwrap().ewma_latency_ms;

        // Update with a very low latency; EWMA should decline
        engine.record_success("ep1", 1.0);

        let new_latency = engine.endpoint_stats("ep1").unwrap().ewma_latency_ms;
        assert!(
            new_latency < initial_latency,
            "Latency should have decreased: {new_latency} vs {initial_latency}"
        );
    }

    #[test]
    fn test_routing_engine_update_failure() {
        let config = AdaptiveRoutingConfig::default();
        let mut engine = AdaptiveRoutingEngine::new(config);

        engine.register_endpoint("ep1".to_string());
        let initial_error_rate = engine.endpoint_stats("ep1").unwrap().ewma_error_rate;

        engine.record_failure("ep1");

        let new_error_rate = engine.endpoint_stats("ep1").unwrap().ewma_error_rate;
        assert!(
            new_error_rate > initial_error_rate,
            "Error rate should have increased: {new_error_rate} vs {initial_error_rate}"
        );
    }

    #[test]
    fn test_config_default() {
        let config = AdaptiveRoutingConfig::default();
        // Alpha should be a reasonable decay factor in (0, 1)
        assert!(config.alpha > 0.0 && config.alpha < 1.0);
        // max_endpoints_per_query should be a positive integer
        assert!(config.max_endpoints_per_query > 0);
        // cost_threshold should be a positive value
        assert!(config.cost_threshold > 0.0);
        // latency_penalty_ms should be >= 0
        assert!(config.latency_penalty_ms >= 0.0);
    }

    #[test]
    fn test_cost_model_zero_patterns() {
        let stats = EndpointStats {
            ewma_latency_ms: 100.0,
            ewma_error_rate: 0.0,
            ..EndpointStats::new()
        };
        let factors = QueryCostFactors {
            selectivity: 0.5,
            triple_pattern_count: 0,
            result_cardinality_hint: 0,
        };
        // cost = (0 * 0.5 + 1) * 100 * 1 = 100
        let cost = EndpointCostEstimator::estimate_cost(&factors, &stats);
        assert!(
            (cost - 100.0).abs() < 1e-9,
            "Expected 100.0 for zero patterns, got {cost}"
        );
    }

    #[test]
    fn test_planner_empty_endpoints() {
        let config = AdaptiveRoutingConfig::default();
        let planner = AdaptivePlanner::new(config);
        let stats: HashMap<String, EndpointStats> = HashMap::new();
        let query = FederatedQuery {
            sub_queries: vec![single_sub_query()],
            optional_endpoints: Vec::new(),
        };

        let plan = planner.plan(&query, &stats, &EndpointCostEstimator);
        // No endpoints → empty plan (graceful degradation)
        assert!(plan.assignments.is_empty());
        assert!(plan.fallback_endpoint.is_none());
    }

    #[test]
    fn test_planner_single_endpoint() {
        let config = AdaptiveRoutingConfig::default();
        let planner = AdaptivePlanner::new(config);

        let mut stats: HashMap<String, EndpointStats> = HashMap::new();
        stats.insert("only".to_string(), make_stats(100.0, 0.0));

        let query = FederatedQuery {
            sub_queries: vec![
                SubQuery {
                    id: 0,
                    triple_patterns: vec![TriplePattern::Subject("s0".to_string())],
                    estimated_cardinality: 5,
                },
                SubQuery {
                    id: 1,
                    triple_patterns: vec![TriplePattern::Object("o1".to_string())],
                    estimated_cardinality: 15,
                },
                SubQuery {
                    id: 2,
                    triple_patterns: vec![TriplePattern::Full(
                        "s".to_string(),
                        "p".to_string(),
                        "o".to_string(),
                    )],
                    estimated_cardinality: 1,
                },
            ],
            optional_endpoints: Vec::new(),
        };

        let plan = planner.plan(&query, &stats, &EndpointCostEstimator);
        assert_eq!(plan.assignments.len(), 3);
        // All sub-queries must go to the only registered endpoint
        for (ep, _) in &plan.assignments {
            assert_eq!(ep, "only");
        }
    }
}
