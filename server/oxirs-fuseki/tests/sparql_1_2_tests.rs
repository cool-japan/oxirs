//! SPARQL 1.2 enhanced feature tests

use chrono::Utc;
use serde_json;
use std::collections::HashMap;

// Test data structures for SPARQL 1.2 features
use oxirs_fuseki::handlers::sparql::{
    AggregateImplementation, AggregationEngine, BindValuesProcessor, CustomAggregate,
    OptimizedPath, PathExecutionPlan, PathStrategy, PropertyPathOptimizer, ServiceDelegator,
    Sparql12Features, SubqueryOptimizer, TraversalDirection,
};

#[cfg(test)]
mod property_path_tests {
    use super::*;

    #[tokio::test]
    async fn test_property_path_optimization() {
        let optimizer = PropertyPathOptimizer::new();

        // Test simple path optimization
        let simple_path = "foaf:knows";
        let optimized = optimizer.optimize_path(simple_path).await;

        assert!(optimized.is_ok());
        let opt_path = optimized.unwrap();
        assert_eq!(opt_path.original_path, simple_path);
        assert!(matches!(
            opt_path.execution_plan.strategy,
            PathStrategy::IndexLookup
        ));
    }

    #[tokio::test]
    async fn test_complex_property_path() {
        let optimizer = PropertyPathOptimizer::new();

        // Test complex path with inverse
        let complex_path = "foaf:knows/^foaf:knows/foaf:name";
        let optimized = optimizer.optimize_path(complex_path).await;

        assert!(optimized.is_ok());
        let opt_path = optimized.unwrap();
        assert!(matches!(
            opt_path.execution_plan.strategy,
            PathStrategy::BidirectionalMeet
        ));
        assert!(opt_path.execution_plan.estimated_cost > 0.0);
    }

    #[tokio::test]
    async fn test_path_caching() {
        let optimizer = PropertyPathOptimizer::new();

        let path = "rdfs:subClassOf*";

        // First optimization should compute
        let start_time = std::time::Instant::now();
        let _result1 = optimizer.optimize_path(path).await.unwrap();
        let first_duration = start_time.elapsed();

        // Second optimization should use cache (should be faster)
        let start_time = std::time::Instant::now();
        let _result2 = optimizer.optimize_path(path).await.unwrap();
        let second_duration = start_time.elapsed();

        // Cache lookup should be faster (in practice, might not be measurable in tests)
        // This test validates the caching mechanism works
        assert!(second_duration <= first_duration);
    }
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;

    #[tokio::test]
    async fn test_aggregation_engine_initialization() {
        let engine = AggregationEngine::new();

        // Check standard SPARQL 1.1 functions are supported
        assert!(engine.supported_functions.contains("COUNT"));
        assert!(engine.supported_functions.contains("SUM"));
        assert!(engine.supported_functions.contains("AVG"));
        assert!(engine.supported_functions.contains("GROUP_CONCAT"));

        // Check SPARQL 1.2 enhanced functions
        assert!(engine.supported_functions.contains("MEDIAN"));
        assert!(engine.supported_functions.contains("STDDEV"));
        assert!(engine.supported_functions.contains("PERCENTILE"));
        assert!(engine.supported_functions.contains("DISTINCT_COUNT"));
    }

    #[tokio::test]
    async fn test_custom_aggregate_registration() {
        let mut engine = AggregationEngine::new();

        let custom_aggregate = CustomAggregate {
            name: "GEOMETRIC_MEAN".to_string(),
            definition: "PRODUCT(values)^(1/COUNT(values))".to_string(),
            return_type: "xsd:double".to_string(),
            implementation: AggregateImplementation::Computed {
                algorithm: "geometric_mean".to_string(),
            },
        };

        engine.register_custom_aggregate(custom_aggregate);

        assert!(engine.custom_aggregates.contains_key("GEOMETRIC_MEAN"));
    }

    #[tokio::test]
    async fn test_aggregation_optimization() {
        let engine = AggregationEngine::new();

        let query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }";
        let optimized = engine.optimize_aggregation(query).await;

        assert!(optimized.is_ok());
        // In a real implementation, this would apply actual optimizations
        assert_eq!(optimized.unwrap(), query);
    }
}

#[cfg(test)]
mod subquery_tests {
    use super::*;

    #[tokio::test]
    async fn test_subquery_optimizer_initialization() {
        let optimizer = SubqueryOptimizer::new();

        assert!(!optimizer.rewrite_rules.is_empty());
        assert!(optimizer
            .rewrite_rules
            .iter()
            .any(|rule| rule.name == "EXISTS_TO_JOIN"));
        assert!(optimizer
            .rewrite_rules
            .iter()
            .any(|rule| rule.name == "SUBQUERY_PULLUP"));
    }

    #[tokio::test]
    async fn test_subquery_optimization() {
        let optimizer = SubqueryOptimizer::new();

        let query_with_subquery = "SELECT * WHERE { ?s ?p ?o . { SELECT * WHERE { ?s ?p ?o } } }";
        let optimized = optimizer.optimize_subqueries(query_with_subquery).await;

        assert!(optimized.is_ok());
        let opt_query = optimized.unwrap();

        // Should apply SUBQUERY_PULLUP rule
        assert_ne!(opt_query, query_with_subquery);
    }

    #[tokio::test]
    async fn test_exists_to_join_optimization() {
        let optimizer = SubqueryOptimizer::new();

        let query_with_exists = "SELECT ?s WHERE { ?s ?p ?o . EXISTS { ?s ?p ?o } }";
        let optimized = optimizer.optimize_subqueries(query_with_exists).await;

        assert!(optimized.is_ok());
    }
}

#[cfg(test)]
mod bind_values_tests {
    use super::*;

    #[tokio::test]
    async fn test_bind_values_processor() {
        let processor = BindValuesProcessor::new();

        let safe_query = "SELECT ?s WHERE { ?s ?p ?o . BIND(?s as ?subject) }";
        let processed = processor.process_bind_values(safe_query).await;

        assert!(processed.is_ok());
    }

    #[tokio::test]
    async fn test_injection_detection() {
        let processor = BindValuesProcessor::new();

        let dangerous_query = "SELECT ?s WHERE { ?s ?p ?o } ; DROP GRAPH <http://example.org>";
        let result = processor.process_bind_values(dangerous_query).await;

        // Should be blocked by security rules
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_values_optimization() {
        let processor = BindValuesProcessor::new();

        let query_with_values = "SELECT ?s WHERE { VALUES ?s { <http://example.org/1> <http://example.org/2> } ?s ?p ?o }";
        let processed = processor.process_bind_values(query_with_values).await;

        assert!(processed.is_ok());
    }
}

#[cfg(test)]
mod federation_tests {
    use super::*;
    use oxirs_fuseki::federation::{
        EndpointCapabilities, FederationPlanner, HealthStatus, ServiceEndpoint,
    };

    #[tokio::test]
    async fn test_federation_planner_initialization() {
        let planner = FederationPlanner::new();
        let stats = planner.get_statistics().await;

        assert_eq!(stats.total_federated_queries, 0);
    }

    #[tokio::test]
    async fn test_endpoint_management() {
        let planner = FederationPlanner::new();

        let endpoint = ServiceEndpoint {
            url: "https://dbpedia.org/sparql".to_string(),
            name: "DBpedia".to_string(),
            capabilities: EndpointCapabilities::default(),
            statistics: Default::default(),
            health_status: HealthStatus::Healthy,
            authentication: None,
            timeout_ms: 30000,
            priority: 1,
        };

        // Add endpoint
        let add_result = planner.add_endpoint(endpoint.clone()).await;
        assert!(add_result.is_ok());

        // Retrieve endpoint
        let retrieved = planner.get_endpoint(&endpoint.url).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "DBpedia");

        // Remove endpoint
        let remove_result = planner.remove_endpoint(&endpoint.url).await;
        assert!(remove_result.is_ok());
        assert!(remove_result.unwrap());
    }

    #[tokio::test]
    async fn test_federated_query_planning() {
        let planner = FederationPlanner::new();

        // Add test endpoint
        let endpoint = ServiceEndpoint {
            url: "https://example.org/sparql".to_string(),
            name: "Test Endpoint".to_string(),
            capabilities: EndpointCapabilities::default(),
            statistics: Default::default(),
            health_status: HealthStatus::Healthy,
            authentication: None,
            timeout_ms: 30000,
            priority: 1,
        };

        planner.add_endpoint(endpoint).await.unwrap();

        let federated_query = r#"
            SELECT ?person ?name WHERE {
                ?person foaf:name ?name .
                SERVICE <https://example.org/sparql> {
                    ?person foaf:age ?age .
                    FILTER(?age > 18)
                }
            }
        "#;

        let plan = planner.create_execution_plan(federated_query).await;
        assert!(plan.is_ok());

        let exec_plan = plan.unwrap();
        assert!(!exec_plan.execution_steps.is_empty());
        assert!(exec_plan.estimated_cost > 0.0);
        assert!(!exec_plan
            .resource_requirements
            .required_endpoints
            .is_empty());
    }

    #[tokio::test]
    async fn test_parallel_service_execution() {
        let planner = FederationPlanner::new();

        let query_with_multiple_services = r#"
            SELECT ?s ?p WHERE {
                SERVICE <https://endpoint1.org/sparql> {
                    ?s foaf:name ?name .
                }
                SERVICE <https://endpoint2.org/sparql> {
                    ?s foaf:age ?age .
                }
            }
        "#;

        let plan = planner
            .create_execution_plan(query_with_multiple_services)
            .await;
        assert!(plan.is_ok());

        let exec_plan = plan.unwrap();
        // Should identify parallel execution opportunities
        assert!(!exec_plan.parallel_sections.is_empty());
    }
}

#[cfg(test)]
mod websocket_integration_tests {
    use super::*;
    use oxirs_fuseki::handlers::websocket::{
        ChangeNotification, SubscriptionFilters, SubscriptionManager,
    };

    #[tokio::test]
    async fn test_subscription_manager() {
        let manager = SubscriptionManager::new();

        let filters = SubscriptionFilters {
            min_results: None,
            max_results: Some(100),
            graph_filter: None,
            update_threshold_ms: Some(1000),
        };

        let subscription_id = manager
            .add_subscription(
                "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
                Some("test_user".to_string()),
                filters,
            )
            .await;

        assert!(!subscription_id.is_empty());

        // Test subscription retrieval
        let subscription = manager.get_subscription(&subscription_id).await;
        assert!(subscription.is_some());

        // Test subscription removal
        let removed = manager.remove_subscription(&subscription_id).await;
        assert!(removed);
    }

    #[tokio::test]
    async fn test_change_notification() {
        let manager = SubscriptionManager::new();

        let notification = ChangeNotification {
            change_type: "INSERT".to_string(),
            affected_graphs: vec!["http://example.org/test".to_string()],
            timestamp: Utc::now(),
            change_count: 5,
        };

        // This should not fail
        manager.notify_change(notification).await;
    }

    #[tokio::test]
    async fn test_enhanced_subscription_features() {
        let manager = SubscriptionManager::new();

        let enhanced_filters = oxirs_fuseki::handlers::websocket::EnhancedSubscriptionFilters {
            min_results: Some(1),
            max_results: Some(1000),
            graph_filter: Some(vec!["http://example.org/main".to_string()]),
            update_threshold_ms: Some(500),
            result_format: Some("json".to_string()),
            include_provenance: Some(true),
            debounce_ms: Some(200),
            batch_updates: Some(true),
        };

        let subscription_id = manager
            .add_enhanced_subscription(
                "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
                Some("enhanced_user".to_string()),
                enhanced_filters,
            )
            .await;

        assert!(!subscription_id.is_empty());

        // Test pause/resume functionality
        let paused = manager.pause_subscription(&subscription_id).await;
        assert!(paused);

        let resumed = manager.resume_subscription(&subscription_id).await;
        assert!(resumed);

        // Test metrics retrieval
        let metrics = manager.get_subscription_metrics(&subscription_id).await;
        assert!(metrics.is_some());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_sparql_12_features_integration() {
        let features = Sparql12Features::new();

        // Test that all components are properly initialized
        assert!(features.property_path_optimizer.path_cache.read().is_ok());
        assert!(!features.aggregation_engine.supported_functions.is_empty());
        assert!(!features.subquery_optimizer.rewrite_rules.is_empty());
        assert!(features.bind_values_processor.injection_detector.enabled);
    }

    #[tokio::test]
    async fn test_query_processing_pipeline() {
        let features = Sparql12Features::new();

        let test_query = r#"
            SELECT ?person (COUNT(?friend) as ?friendCount) WHERE {
                ?person foaf:knows+ ?friend .
                BIND(?person as ?subject)
                VALUES ?type { foaf:Person }
                ?person a ?type .
            } GROUP BY ?person
        "#;

        // Test property path optimization
        let path_opt = features
            .property_path_optimizer
            .optimize_path("foaf:knows+")
            .await;
        assert!(path_opt.is_ok());

        // Test aggregation optimization
        let agg_opt = features
            .aggregation_engine
            .optimize_aggregation(test_query)
            .await;
        assert!(agg_opt.is_ok());

        // Test subquery optimization
        let sub_opt = features
            .subquery_optimizer
            .optimize_subqueries(test_query)
            .await;
        assert!(sub_opt.is_ok());

        // Test BIND/VALUES processing
        let bind_opt = features
            .bind_values_processor
            .process_bind_values(test_query)
            .await;
        assert!(bind_opt.is_ok());
    }
}

#[cfg(test)]
mod sparql_star_tests {
    use super::*;

    #[tokio::test]
    async fn test_quoted_triple_detection() {
        let queries_with_quoted_triples = vec![
            "SELECT ?s WHERE { << ?s ?p ?o >> :confidence ?value }",
            "SELECT ?stmt WHERE { ?stmt a rdf:Statement ; :hasTriple << :alice :knows :bob >> }",
            "SELECT ?s ?p ?o WHERE { << ?s ?p ?o >> :source :wikipedia }",
            "SELECT ?result WHERE { << << :a :b :c >> :d :e >> :f ?result }",
        ];

        for query in queries_with_quoted_triples {
            assert!(
                oxirs_fuseki::handlers::sparql::contains_sparql_star_features(query),
                "Failed to detect quoted triple in: {}",
                query
            );
        }
    }

    #[tokio::test]
    async fn test_annotation_syntax_detection() {
        let queries_with_annotations = vec![
            "SELECT ?s WHERE { ?s :name ?name {| :confidence 0.9 |} }",
            "SELECT ?s WHERE { ?s :age ?age {| :source :survey ; :date \"2023-01-01\" |} }",
            "SELECT ?s WHERE { :alice :knows :bob {| :since 2020 ; :certainty \"high\" |} }",
        ];

        for query in queries_with_annotations {
            assert!(
                oxirs_fuseki::handlers::sparql::contains_sparql_star_features(query),
                "Failed to detect annotation syntax in: {}",
                query
            );
        }
    }

    #[tokio::test]
    async fn test_sparql_star_functions() {
        let queries_with_functions = vec![
            "SELECT ?s WHERE { ?t a :Statement . BIND(SUBJECT(?t) AS ?s) }",
            "SELECT ?p WHERE { ?t a :Statement . BIND(PREDICATE(?t) AS ?p) }",
            "SELECT ?o WHERE { ?t a :Statement . BIND(OBJECT(?t) AS ?o) }",
            "SELECT ?t WHERE { ?t ?p ?o . FILTER(ISTRIPLE(?t)) }",
        ];

        for query in queries_with_functions {
            assert!(
                oxirs_fuseki::handlers::sparql::contains_sparql_star_features(query),
                "Failed to detect SPARQL-star function in: {}",
                query
            );
        }
    }

    #[tokio::test]
    async fn test_quoted_triple_parsing() {
        // Simple quoted triple
        let result = oxirs_fuseki::handlers::sparql::parse_quoted_triple_value(
            "<< <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> >>"
        );
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.subject, "<http://example.org/alice>");
        assert_eq!(parsed.predicate, "<http://example.org/knows>");
        assert_eq!(parsed.object, "<http://example.org/bob>");

        // Quoted triple with prefixed names
        let result = oxirs_fuseki::handlers::sparql::parse_quoted_triple_value(
            "<< ex:alice foaf:knows ex:bob >>"
        );
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.subject, "ex:alice");
        assert_eq!(parsed.predicate, "foaf:knows");
        assert_eq!(parsed.object, "ex:bob");

        // Quoted triple with literal
        let result = oxirs_fuseki::handlers::sparql::parse_quoted_triple_value(
            "<< ex:alice foaf:age \"30\"^^xsd:integer >>"
        );
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.subject, "ex:alice");
        assert_eq!(parsed.predicate, "foaf:age");
        assert_eq!(parsed.object, "\"30\"^^xsd:integer");

        // Quoted triple with language-tagged literal
        let result = oxirs_fuseki::handlers::sparql::parse_quoted_triple_value(
            "<< ex:alice foaf:name \"Alice\"@en >>"
        );
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.object, "\"Alice\"@en");
    }

    #[tokio::test]
    async fn test_quoted_triple_pattern_extraction() {
        // Single quoted triple pattern
        let query = "SELECT ?s WHERE { << ?s ?p ?o >> :confidence ?value }";
        let patterns = oxirs_fuseki::handlers::sparql::extract_quoted_triple_patterns(query).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0], "<< ?s ?p ?o >>");

        // Multiple quoted triple patterns
        let query = "SELECT ?s WHERE { << ?s ?p ?o >> :confidence ?c . << ?x ?y ?z >> :source ?src }";
        let patterns = oxirs_fuseki::handlers::sparql::extract_quoted_triple_patterns(query).unwrap();
        assert_eq!(patterns.len(), 2);
        assert!(patterns.contains(&"<< ?s ?p ?o >>".to_string()));
        assert!(patterns.contains(&"<< ?x ?y ?z >>".to_string()));

        // Nested quoted triples
        let query = "SELECT ?s WHERE { << << ?a ?b ?c >> ?p ?o >> :confidence ?value }";
        let patterns = oxirs_fuseki::handlers::sparql::extract_quoted_triple_patterns(query).unwrap();
        assert_eq!(patterns.len(), 2);
        assert!(patterns.contains(&"<< ?a ?b ?c >>".to_string()));
        assert!(patterns.contains(&"<< << ?a ?b ?c >> ?p ?o >>".to_string()));
    }

    #[tokio::test]
    async fn test_sparql_star_processing() {
        // Test processing of bindings with quoted triples
        let mut bindings = vec![
            {
                let mut binding = HashMap::new();
                binding.insert(
                    "stmt".to_string(),
                    serde_json::json!("<< ex:alice ex:knows ex:bob >>"),
                );
                binding
            },
        ];

        // Query that uses SUBJECT function
        let query = "SELECT ?stmt ?s WHERE { ?stmt :confidence ?c . BIND(SUBJECT(?stmt) AS ?s) }";
        let result = oxirs_fuseki::handlers::sparql::process_sparql_star_features(query, &mut bindings).await;
        assert!(result.is_ok());

        // Check that subject was extracted
        assert!(bindings[0].contains_key("stmt_subject"));
        assert_eq!(bindings[0]["stmt_subject"], serde_json::json!("ex:alice"));
    }

    #[tokio::test]
    async fn test_annotation_processing() {
        // Test annotation extraction
        let query = "SELECT ?s WHERE { ?s :name ?name {| :confidence 0.9 ; :source :manual |} }";
        let binding = HashMap::new();

        let annotations = oxirs_fuseki::handlers::sparql::extract_annotations(query, &binding).unwrap();
        assert!(!annotations.is_empty());
        
        // Should extract confidence and source annotations
        let annotation_props: Vec<String> = annotations.iter().map(|(k, _)| k.clone()).collect();
        assert!(annotation_props.iter().any(|p| p.contains("confidence")));
        assert!(annotation_props.iter().any(|p| p.contains("source")));
    }

    #[tokio::test]
    async fn test_complex_sparql_star_query() {
        // Complex query combining multiple SPARQL-star features
        let query = r#"
            SELECT ?person ?stmt ?confidence WHERE {
                ?stmt a rdf:Statement ;
                      :hasTriple << ?person foaf:knows ?friend >> ;
                      :confidence ?confidence .
                
                FILTER(?confidence > 0.8)
                
                ?person foaf:name ?name {| :verified true |} .
                
                BIND(OBJECT(?stmt) AS ?friend)
                
                SERVICE <http://example.org/sparql> {
                    << ?friend foaf:age ?age >> :source :external .
                }
            }
        "#;

        // This query should be detected as containing SPARQL-star features
        assert!(oxirs_fuseki::handlers::sparql::contains_sparql_star_features(query));

        // Extract quoted triple patterns
        let patterns = oxirs_fuseki::handlers::sparql::extract_quoted_triple_patterns(query).unwrap();
        assert_eq!(patterns.len(), 2);
        assert!(patterns.contains(&"<< ?person foaf:knows ?friend >>".to_string()));
        assert!(patterns.contains(&"<< ?friend foaf:age ?age >>".to_string()));
    }

    #[tokio::test]
    async fn test_sparql_star_with_aggregation() {
        // Test SPARQL-star with aggregation functions
        let query = r#"
            SELECT ?source (COUNT(DISTINCT ?stmt) as ?count) WHERE {
                ?stmt a rdf:Statement ;
                      :hasTriple << ?s ?p ?o >> ;
                      :source ?source .
                
                FILTER(ISTRIPLE(?stmt))
            }
            GROUP BY ?source
            HAVING(?count > 10)
        "#;

        assert!(oxirs_fuseki::handlers::sparql::contains_sparql_star_features(query));
        assert!(oxirs_fuseki::handlers::sparql::contains_aggregation_functions(query));
    }
}
