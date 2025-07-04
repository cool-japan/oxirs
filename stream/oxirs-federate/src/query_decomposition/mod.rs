//! Advanced Query Decomposition for Federated Execution
//!
//! This module provides sophisticated algorithms for decomposing complex
//! SPARQL and GraphQL queries into optimal execution plans across multiple services.
//!
//! The module is organized into the following components:
//! - `types`: Type definitions and data structures for query decomposition
//! - `core`: Main QueryDecomposer implementation with decompose method and orchestration
//! - `graph_analysis`: Query graph building, connectivity analysis, and component finding
//! - `plan_generation`: Plan generation strategies and advanced distribution algorithms
//! - `cost_estimation`: Cost estimation algorithms for different execution strategies
//! - `pattern_analysis`: Pattern analysis, selectivity estimation, and join pattern detection
//! - `advanced_pattern_analysis`: ML-driven pattern analysis with sophisticated optimization

pub mod advanced_pattern_analysis;
pub mod core;
pub mod cost_estimation;
pub mod graph_analysis;
pub mod pattern_analysis;
pub mod plan_generation;
pub mod types;

// Re-export main types and structs for public API
pub use advanced_pattern_analysis::*;
pub use core::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cost_estimation::OptimizationLevel,
        planner::{QueryInfo, QueryType, TriplePattern},
        FederatedService, ServiceCapability, ServiceRegistry,
    };
    use std::collections::HashSet;

    fn create_test_service() -> FederatedService {
        use crate::service::{ServiceType, ServiceMetadata, ServicePerformance};
        use std::time::Duration;
        
        FederatedService {
            id: "test-service".to_string(),
            name: "Test Service".to_string(),
            endpoint: "http://localhost:8080/sparql".to_string(),
            service_type: ServiceType::Sparql,
            capabilities: [ServiceCapability::SparqlQuery].iter().cloned().collect(),
            data_patterns: vec!["*".to_string()],
            auth: None,
            metadata: ServiceMetadata::default(),
            extended_metadata: None,
            performance: ServicePerformance {
                average_response_time: Some(Duration::from_millis(100)),
                avg_response_time_ms: 100.0,
                reliability_score: 0.95,
                max_concurrent_requests: Some(10),
                rate_limit: None,
                estimated_dataset_size: Some(1000),
                supported_result_formats: vec!["application/json".to_string()],
                success_rate: Some(0.98),
                error_rate: Some(0.02),
                last_updated: None,
            },
            status: None,
        }
    }

    fn create_test_query() -> QueryInfo {
        QueryInfo {
            query_type: QueryType::Select,
            original_query: "SELECT ?s ?p ?o WHERE { ?s rdf:type foaf:Person . ?s foaf:name ?name }".to_string(),
            variables: ["?s".to_string(), "?p".to_string(), "?o".to_string()].iter().cloned().collect(),
            patterns: vec![
                TriplePattern {
                    subject: Some("?s".to_string()),
                    predicate: Some("rdf:type".to_string()),
                    object: Some("foaf:Person".to_string()),
                    pattern_string: "?s rdf:type foaf:Person".to_string(),
                },
                TriplePattern {
                    subject: Some("?s".to_string()),
                    predicate: Some("foaf:name".to_string()),
                    object: Some("?name".to_string()),
                    pattern_string: "?s foaf:name ?name".to_string(),
                },
            ],
            filters: vec![],
            complexity: 2,
            estimated_cost: 1000,
        }
    }

    #[tokio::test]
    async fn test_decomposer_creation() {
        let decomposer = QueryDecomposer::new();
        assert_eq!(
            decomposer.config.optimization_strategy,
            OptimizationStrategy::Balanced
        );
    }

    #[tokio::test]
    async fn test_query_graph_building() {
        let decomposer = QueryDecomposer::new();
        let query = create_test_query();

        let graph = decomposer.build_query_graph(&query).unwrap();
        assert_eq!(graph.variable_nodes.len(), 3); // ?s, ?p, ?o
        assert_eq!(graph.pattern_nodes.len(), 2); // Two patterns
    }

    #[tokio::test]
    async fn test_component_finding() {
        let decomposer = QueryDecomposer::new();
        let query = create_test_query();

        let graph = decomposer.build_query_graph(&query).unwrap();
        let components = decomposer.find_connected_components(&graph);

        assert!(!components.is_empty());
        assert_eq!(components[0].patterns.len(), 2); // Both patterns should be connected
    }

    #[tokio::test]
    async fn test_single_service_plan() {
        let decomposer = QueryDecomposer::new();
        let service = create_test_service();
        let query = create_test_query();

        let graph = decomposer.build_query_graph(&query).unwrap();
        let components = decomposer.find_connected_components(&graph);

        let plan = decomposer
            .create_single_service_plan(&service, &components[0])
            .unwrap();
        assert_eq!(plan.strategy, PlanStrategy::SingleService);
        assert!(!plan.requires_join);
        assert_eq!(plan.steps.len(), 1);
    }

    #[tokio::test]
    async fn test_cost_estimation() {
        let cost_estimator = CostEstimator::new();
        let service = create_test_service();
        let pattern = TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("rdf:type".to_string()),
            object: Some("foaf:Person".to_string()),
            pattern_string: "?s rdf:type foaf:Person".to_string(),
        };

        let cost = cost_estimator.estimate_single_pattern_cost(&service, &pattern);
        assert!(cost > 0.0);
    }

    #[tokio::test]
    async fn test_pattern_analysis() {
        let decomposer = QueryDecomposer::new();
        let pattern = TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("rdf:type".to_string()),
            object: Some("foaf:Person".to_string()),
            pattern_string: "?s rdf:type foaf:Person".to_string(),
        };

        let variables = decomposer.extract_pattern_variables(&pattern);
        assert!(variables.contains("?s"));
        assert_eq!(variables.len(), 1);
    }

    #[test]
    fn test_selectivity_estimation() {
        let decomposer = QueryDecomposer::new();
        let patterns = vec![(
            0,
            TriplePattern {
                subject: Some("?s".to_string()),
                predicate: Some("rdf:type".to_string()),
                object: Some("foaf:Person".to_string()),
                pattern_string: "?s rdf:type foaf:Person".to_string(),
            },
        )];

        let selectivity = decomposer.estimate_pattern_selectivity(&patterns);
        assert!(selectivity > 0.0 && selectivity <= 1.0);
    }

    #[test]
    fn test_star_join_detection() {
        let decomposer = QueryDecomposer::new();
        let mut component = QueryComponent::new();

        // Create a star pattern (multiple patterns sharing ?s)
        component.patterns = vec![
            (
                0,
                TriplePattern {
                    subject: Some("?s".to_string()),
                    predicate: Some("rdf:type".to_string()),
                    object: Some("foaf:Person".to_string()),
                    pattern_string: "?s rdf:type foaf:Person".to_string(),
                },
            ),
            (
                1,
                TriplePattern {
                    subject: Some("?s".to_string()),
                    predicate: Some("foaf:name".to_string()),
                    object: Some("?name".to_string()),
                    pattern_string: "?s foaf:name ?name".to_string(),
                },
            ),
            (
                2,
                TriplePattern {
                    subject: Some("?s".to_string()),
                    predicate: Some("foaf:age".to_string()),
                    object: Some("?age".to_string()),
                    pattern_string: "?s foaf:age ?age".to_string(),
                },
            ),
        ];

        let is_star = decomposer.is_star_join_pattern(&component);
        assert!(is_star);
    }

    #[test]
    fn test_optimization_potential() {
        let cost_estimator = CostEstimator::new();
        let baseline = 1000.0;
        let optimized = 600.0;

        let potential = cost_estimator.estimate_optimization_potential(baseline, optimized);
        assert_eq!(potential.potential_level, OptimizationLevel::Medium);
        assert_eq!(potential.improvement_ratio, 0.4);
    }
}
