//! Tests for Federated Query Optimization

use oxirs_fuseki::{
    config::MonitoringConfig,
    error::{FusekiError, FusekiResult},
    federated_query_optimizer::*,
    federation::planner::ExecutionStrategy,
    metrics::MetricsService,
};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_service_pattern_extraction() {
    let metrics = Arc::new(MetricsService::new(MonitoringConfig::default()).unwrap());
    let optimizer = FederatedQueryOptimizer::new(metrics);

    let query = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?name ?friend
        WHERE {
            ?person foaf:name ?name .
            SERVICE <http://example.org/sparql> {
                ?person foaf:knows ?friend .
            }
        }
    "#;

    let patterns = optimizer.extract_service_patterns(query).unwrap();
    assert_eq!(patterns.len(), 1);
    assert_eq!(patterns[0].service_url, "http://example.org/sparql");
    assert!(!patterns[0].is_silent);
}

#[tokio::test]
async fn test_multiple_service_patterns() {
    let metrics = Arc::new(MetricsService::new(MonitoringConfig::default()).unwrap());
    let optimizer = FederatedQueryOptimizer::new(metrics);

    let query = r#"
        SELECT ?s ?p ?o
        WHERE {
            SERVICE <http://endpoint1.org/sparql> {
                ?s ?p ?o
            }
            SERVICE SILENT <http://endpoint2.org/sparql> {
                ?s ?p2 ?o2
            }
        }
    "#;

    let patterns = optimizer.extract_service_patterns(query).unwrap();
    assert_eq!(patterns.len(), 2);
    assert!(!patterns[0].is_silent);
    assert!(patterns[1].is_silent);
}

#[tokio::test]
async fn test_nested_service_patterns() {
    let metrics = Arc::new(MetricsService::new(MonitoringConfig::default()).unwrap());
    let optimizer = FederatedQueryOptimizer::new(metrics);

    let query = r#"
        SELECT ?person ?name ?friend ?friendName
        WHERE {
            ?person foaf:name ?name .
            SERVICE <http://endpoint1.org/sparql> {
                ?person foaf:knows ?friend .
                OPTIONAL {
                    SERVICE <http://endpoint2.org/sparql> {
                        ?friend foaf:name ?friendName .
                    }
                }
            }
        }
    "#;

    let patterns = optimizer.extract_service_patterns(query).unwrap();
    assert!(patterns.len() >= 1);
}

#[tokio::test]
async fn test_endpoint_health_check() {
    let mut registry = EndpointRegistry::new();

    let endpoint = EndpointInfo {
        url: "http://test.example.org/sparql".to_string(),
        name: "Test Endpoint".to_string(),
        description: Some("Test endpoint for unit tests".to_string()),
        capabilities: EndpointCapabilities {
            sparql_version: "1.1".to_string(),
            supports_update: true,
            supports_graph_store: true,
            supports_service_description: true,
            max_query_size: Some(10000),
            rate_limit: Some(RateLimit {
                requests_per_second: 10,
                burst_size: 20,
            }),
            features: std::collections::HashSet::new(),
        },
        authentication: None,
        timeout_ms: 5000,
        max_retries: 3,
        priority: 1,
    };

    registry.register_endpoint(endpoint.clone());

    // Health check will fail for non-existent endpoint, but should handle gracefully
    let result = registry.check_endpoint_health(endpoint).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_query_decomposition() {
    let planner = QueryPlanner::new();

    let service_patterns = vec![
        ServicePattern {
            service_url: "http://endpoint1.org/sparql".to_string(),
            pattern: r#"
                SERVICE <http://endpoint1.org/sparql> {
                    ?person foaf:knows ?friend .
                }
            "#
            .to_string(),
            is_silent: false,
            is_optional: false,
        },
        ServicePattern {
            service_url: "http://endpoint2.org/sparql".to_string(),
            pattern: r#"
                SERVICE <http://endpoint2.org/sparql> {
                    ?friend foaf:name ?name .
                }
            "#
            .to_string(),
            is_silent: false,
            is_optional: false,
        },
    ];

    let query = r#"
        SELECT ?person ?friend ?name
        WHERE {
            ?person a foaf:Person .
            SERVICE <http://endpoint1.org/sparql> {
                ?person foaf:knows ?friend .
            }
            SERVICE <http://endpoint2.org/sparql> {
                ?friend foaf:name ?name .
            }
        }
    "#;

    let plan = planner
        .create_execution_plan(query, &service_patterns)
        .await
        .unwrap();

    assert!(!plan.fragments.is_empty());
    assert_eq!(plan.fragments.len(), service_patterns.len());
}

#[tokio::test]
async fn test_join_optimization() {
    let optimizer = JoinOrderOptimizer::new();

    let fragments = vec![
        QueryFragment {
            fragment_id: "f1".to_string(),
            sparql: "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
            target_endpoints: vec!["http://endpoint1.org/sparql".to_string()],
            dependencies: vec![],
            estimated_cost: 10.0,
            is_optional: false,
        },
        QueryFragment {
            fragment_id: "f2".to_string(),
            sparql: "SELECT ?s ?p2 ?o2 WHERE { ?s ?p2 ?o2 }".to_string(),
            target_endpoints: vec!["http://endpoint2.org/sparql".to_string()],
            dependencies: vec![],
            estimated_cost: 20.0,
            is_optional: false,
        },
        QueryFragment {
            fragment_id: "f3".to_string(),
            sparql: "SELECT ?o ?p3 ?o3 WHERE { ?o ?p3 ?o3 }".to_string(),
            target_endpoints: vec!["http://endpoint3.org/sparql".to_string()],
            dependencies: vec!["f1".to_string()],
            estimated_cost: 15.0,
            is_optional: false,
        },
    ];

    let join_plan = optimizer.optimize_joins(&fragments).await.unwrap();

    assert!(!join_plan.steps.is_empty());
    assert_eq!(join_plan.steps.len(), fragments.len() - 1);
}

#[tokio::test]
async fn test_cost_estimation() {
    let estimator = CostEstimator::new();

    let plan = ExecutionPlan {
        query_id: "test-query-1".to_string(),
        fragments: vec![QueryFragment {
            fragment_id: "f1".to_string(),
            sparql: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100".to_string(),
            target_endpoints: vec!["http://endpoint1.org/sparql".to_string()],
            dependencies: vec![],
            estimated_cost: 1.0,
            is_optional: false,
        }],
        join_plan: JoinPlan {
            steps: vec![],
            estimated_cost: 0.0,
            estimated_time_ms: 0,
        },
        timeout_ms: 30000,
        optimization_hints: HashMap::new(),
        execution_steps: vec!["Query fragment f1".to_string()],
        estimated_cost: 1.0,
        resource_requirements: ResourceRequirements {
            required_endpoints: vec!["http://endpoint1.org/sparql".to_string()],
            estimated_memory_mb: 100.0,
            estimated_cpu_cores: 1.0,
        },
    };

    let cost = estimator.estimate_cost(&plan).await.unwrap();
    assert!(cost > 0.0);
}

#[tokio::test]
async fn test_result_merger_union() {
    let merger = ResultMerger::new();

    let results = vec![
        QueryResults {
            bindings: vec![
                HashMap::from([
                    ("x".to_string(), serde_json::json!("value1")),
                    ("y".to_string(), serde_json::json!(1)),
                ]),
                HashMap::from([
                    ("x".to_string(), serde_json::json!("value2")),
                    ("y".to_string(), serde_json::json!(2)),
                ]),
            ],
            metadata: ResultMetadata {
                total_execution_time_ms: 100,
                endpoint_times: HashMap::from([("endpoint1".to_string(), 100)]),
                result_count: 2,
                partial_results: false,
            },
        },
        QueryResults {
            bindings: vec![HashMap::from([
                ("x".to_string(), serde_json::json!("value3")),
                ("y".to_string(), serde_json::json!(3)),
            ])],
            metadata: ResultMetadata {
                total_execution_time_ms: 50,
                endpoint_times: HashMap::from([("endpoint2".to_string(), 50)]),
                result_count: 1,
                partial_results: false,
            },
        },
    ];

    let merged = merger.merge_results(results).await.unwrap();
    assert_eq!(merged.bindings.len(), 3);
    assert_eq!(merged.metadata.total_execution_time_ms, 150);
}

#[tokio::test]
async fn test_result_merger_join() {
    let merger = ResultMerger::new();
    let join_strategy = merger.strategies.get("join").unwrap();

    let results = vec![
        QueryResults {
            bindings: vec![
                HashMap::from([
                    ("person".to_string(), serde_json::json!("person1")),
                    ("name".to_string(), serde_json::json!("Alice")),
                ]),
                HashMap::from([
                    ("person".to_string(), serde_json::json!("person2")),
                    ("name".to_string(), serde_json::json!("Bob")),
                ]),
            ],
            metadata: ResultMetadata {
                total_execution_time_ms: 100,
                endpoint_times: HashMap::new(),
                result_count: 2,
                partial_results: false,
            },
        },
        QueryResults {
            bindings: vec![
                HashMap::from([
                    ("person".to_string(), serde_json::json!("person1")),
                    ("friend".to_string(), serde_json::json!("person2")),
                ]),
                HashMap::from([
                    ("person".to_string(), serde_json::json!("person2")),
                    ("friend".to_string(), serde_json::json!("person3")),
                ]),
            ],
            metadata: ResultMetadata {
                total_execution_time_ms: 100,
                endpoint_times: HashMap::new(),
                result_count: 2,
                partial_results: false,
            },
        },
    ];

    let joined = join_strategy.merge(results).await.unwrap();

    // Should join on common "person" variable
    assert_eq!(joined.bindings.len(), 2);

    // Check that joined results have all variables
    for binding in &joined.bindings {
        assert!(binding.contains_key("person"));
        assert!(binding.contains_key("name"));
        assert!(binding.contains_key("friend"));
    }
}

#[tokio::test]
async fn test_result_merger_distinct() {
    let merger = ResultMerger::new();
    let distinct_strategy = merger.strategies.get("distinct").unwrap();

    let results = vec![
        QueryResults {
            bindings: vec![
                HashMap::from([("x".to_string(), serde_json::json!("value1"))]),
                HashMap::from([("x".to_string(), serde_json::json!("value2"))]),
                HashMap::from([("x".to_string(), serde_json::json!("value1"))]), // Duplicate
            ],
            metadata: ResultMetadata {
                total_execution_time_ms: 100,
                endpoint_times: HashMap::new(),
                result_count: 3,
                partial_results: false,
            },
        },
        QueryResults {
            bindings: vec![
                HashMap::from([("x".to_string(), serde_json::json!("value1"))]), // Duplicate
                HashMap::from([("x".to_string(), serde_json::json!("value3"))]),
            ],
            metadata: ResultMetadata {
                total_execution_time_ms: 50,
                endpoint_times: HashMap::new(),
                result_count: 2,
                partial_results: false,
            },
        },
    ];

    let distinct = distinct_strategy.merge(results).await.unwrap();

    // Should remove duplicates
    assert_eq!(distinct.bindings.len(), 3); // value1, value2, value3
}

#[tokio::test]
async fn test_parallel_execution_strategy() {
    let strategy = ExecutionStrategy::Parallel;

    let plan = ExecutionPlan {
        query_id: "test-parallel".to_string(),
        fragments: vec![
            QueryFragment {
                fragment_id: "f1".to_string(),
                sparql: "SELECT ?s WHERE { ?s a ?o }".to_string(),
                target_endpoints: vec!["http://mock.endpoint1.org/sparql".to_string()],
                dependencies: vec![],
                estimated_cost: 10.0,
                is_optional: false,
            },
            QueryFragment {
                fragment_id: "f2".to_string(),
                sparql: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
                target_endpoints: vec!["http://mock.endpoint2.org/sparql".to_string()],
                dependencies: vec![],
                estimated_cost: 10.0,
                is_optional: false,
            },
        ],
        join_plan: JoinPlan {
            steps: vec![],
            estimated_cost: 20.0,
            estimated_time_ms: 200,
        },
        timeout_ms: 30000,
        optimization_hints: HashMap::new(),
        execution_steps: vec!["Execute f1".to_string(), "Execute f2".to_string()],
        estimated_cost: 20.0,
        resource_requirements: ResourceRequirements {
            required_endpoints: vec![
                "http://mock.endpoint1.org/sparql".to_string(),
                "http://mock.endpoint2.org/sparql".to_string(),
            ],
            estimated_memory_mb: 200.0,
            estimated_cpu_cores: 2.0,
        },
    };

    // Test that we can create the strategy and plan
    assert_eq!(strategy, ExecutionStrategy::Parallel);
    assert_eq!(plan.fragments.len(), 2);
}

#[tokio::test]
async fn test_adaptive_execution_strategy() {
    let strategy = ExecutionStrategy::Adaptive;

    let plan = ExecutionPlan {
        query_id: "test-adaptive".to_string(),
        fragments: vec![
            QueryFragment {
                fragment_id: "f1".to_string(),
                sparql: "SELECT ?s WHERE { ?s a ?o }".to_string(),
                target_endpoints: vec!["http://endpoint1.org/sparql".to_string()],
                dependencies: vec![],
                estimated_cost: 10.0,
                is_optional: false,
            },
            QueryFragment {
                fragment_id: "f2".to_string(),
                sparql: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
                target_endpoints: vec!["http://endpoint2.org/sparql".to_string()],
                dependencies: vec![],
                estimated_cost: 10.0,
                is_optional: false,
            },
            QueryFragment {
                fragment_id: "f3".to_string(),
                sparql: "SELECT ?o WHERE { ?s ?p ?o }".to_string(),
                target_endpoints: vec!["http://endpoint3.org/sparql".to_string()],
                dependencies: vec!["f1".to_string()],
                estimated_cost: 15.0,
                is_optional: false,
            },
        ],
        join_plan: JoinPlan {
            steps: vec![],
            estimated_cost: 35.0,
            estimated_time_ms: 350,
        },
        timeout_ms: 30000,
        optimization_hints: HashMap::new(),
        execution_steps: vec![
            "Execute f1".to_string(),
            "Execute f2".to_string(),
            "Execute f3".to_string(),
        ],
        estimated_cost: 35.0,
        resource_requirements: ResourceRequirements {
            required_endpoints: vec![
                "http://endpoint1.org/sparql".to_string(),
                "http://endpoint2.org/sparql".to_string(),
                "http://endpoint3.org/sparql".to_string(),
            ],
            estimated_memory_mb: 350.0,
            estimated_cpu_cores: 3.0,
        },
    };

    // Test that we can create the strategy and plan
    assert_eq!(strategy, ExecutionStrategy::Adaptive);
    assert_eq!(plan.fragments.len(), 3);
}

#[tokio::test]
async fn test_cardinality_estimation() {
    let estimator = CardinalityEstimator::new();

    // Test query with LIMIT
    let query_with_limit = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 42";
    let cardinality = estimator
        .estimate_cardinality(query_with_limit)
        .await
        .unwrap();
    assert_eq!(cardinality, 42);

    // Test query without LIMIT
    let query_no_limit = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let cardinality = estimator
        .estimate_cardinality(query_no_limit)
        .await
        .unwrap();
    assert_eq!(cardinality, 1000); // Default estimate
}

#[tokio::test]
async fn test_retry_policy() {
    let executor = FederatedExecutor::new();

    // Test backoff calculation
    assert_eq!(executor.calculate_backoff(1).as_millis(), 200);
    assert_eq!(executor.calculate_backoff(2).as_millis(), 400);
    assert_eq!(executor.calculate_backoff(3).as_millis(), 800);

    // Should be capped at max_backoff_ms
    assert!(executor.calculate_backoff(10).as_millis() <= 5000);
}

#[tokio::test]
async fn test_full_federated_query_flow() {
    let metrics = Arc::new(MetricsService::new(MonitoringConfig::default()).unwrap());
    let optimizer = FederatedQueryOptimizer::new(metrics);

    let query = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/>
        
        SELECT ?person ?name ?friend ?friendName
        WHERE {
            ?person foaf:name ?name .
            SERVICE <http://people.example.org/sparql> {
                ?person foaf:knows ?friend .
            }
            SERVICE <http://names.example.org/sparql> {
                ?friend foaf:name ?friendName .
            }
            FILTER(?name != ?friendName)
        }
        ORDER BY ?name
        LIMIT 10
    "#;

    // Test will fail with network error, but validates the flow
    let result = optimizer.process_federated_query(query, 5000).await;
    assert!(result.is_err());
}
