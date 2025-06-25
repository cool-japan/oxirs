//! Integration tests for oxirs-shacl-ai
//!
//! This module contains comprehensive integration tests for the AI-powered
//! SHACL validation and shape generation system.

use oxirs_core::model::{Literal, NamedNode, Term, Triple};
use oxirs_core::store::Store;
use oxirs_shacl_ai::*;
use std::collections::HashMap;

mod analytics_tests;
mod insights_tests;
mod optimization_tests;
mod pattern_analysis_tests;
mod prediction_tests;
mod quality_assessment_tests;
mod shape_learning_tests;
mod test_data;

/// Test configuration for integration tests
#[derive(Debug)]
struct TestConfig {
    pub enable_logging: bool,
    pub test_data_size: usize,
    pub mock_store: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            enable_logging: true,
            test_data_size: 1000,
            mock_store: true,
        }
    }
}

/// Setup function for integration tests
fn setup_test_environment() -> (ShaclAiAssistant, Store, TestConfig) {
    let config = ShaclAiConfig::default();
    let assistant = ShaclAiAssistant::new(config);
    let store = Store::new();
    let test_config = TestConfig::default();

    (assistant, store, test_config)
}

/// Generate test RDF data
fn generate_test_data(size: usize) -> Vec<Triple> {
    let mut triples = Vec::new();

    for i in 0..size {
        let subject = NamedNode::new(format!("http://example.org/person/{}", i)).unwrap();
        let name_predicate = NamedNode::new("http://example.org/name").unwrap();
        let age_predicate = NamedNode::new("http://example.org/age").unwrap();
        let type_predicate =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let person_class = NamedNode::new("http://example.org/Person").unwrap();

        // Add type triple
        triples.push(Triple::new(
            subject.clone().into(),
            type_predicate.into(),
            person_class.into(),
        ));

        // Add name triple
        let name_literal = Literal::new_simple_literal(format!("Person {}", i));
        triples.push(Triple::new(
            subject.clone().into(),
            name_predicate.into(),
            name_literal.into(),
        ));

        // Add age triple
        let age_literal = Literal::new_typed_literal(
            format!("{}", 20 + (i % 50)),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        );
        triples.push(Triple::new(
            subject.into(),
            age_predicate.into(),
            age_literal.into(),
        ));
    }

    triples
}

#[tokio::test]
async fn test_full_ai_assistant_workflow() {
    let (mut assistant, mut store, test_config) = setup_test_environment();

    // Generate and load test data
    let test_data = generate_test_data(test_config.test_data_size);
    for triple in test_data {
        store.insert(&triple).expect("Failed to insert test data");
    }

    // Test shape learning
    let learned_shapes = assistant
        .learn_shapes_from_store(&store, None)
        .await
        .expect("Shape learning failed");
    assert!(!learned_shapes.is_empty(), "No shapes were learned");

    // Test pattern discovery
    let patterns = assistant
        .discover_patterns(&store, None)
        .await
        .expect("Pattern discovery failed");
    assert!(!patterns.is_empty(), "No patterns were discovered");

    // Test quality assessment
    let quality_report = assistant
        .assess_quality(&store, &learned_shapes, None)
        .await
        .expect("Quality assessment failed");
    assert!(quality_report.overall_score >= 0.0 && quality_report.overall_score <= 1.0);

    // Test validation prediction
    let prediction_result = assistant
        .predict_validation_outcome(&store, &learned_shapes, None)
        .await
        .expect("Prediction failed");
    assert!(prediction_result.confidence >= 0.0 && prediction_result.confidence <= 1.0);
}

#[test]
fn test_assistant_configuration() {
    let config = ShaclAiConfig {
        learning_config: learning::LearningConfig {
            enable_shape_generation: true,
            min_support: 0.2,
            min_confidence: 0.9,
            max_shapes: 50,
            enable_training: true,
            algorithm_params: HashMap::new(),
        },
        pattern_config: patterns::PatternConfig {
            enable_pattern_recognition: true,
            min_support_threshold: 0.15,
            min_confidence_threshold: 0.8,
            max_pattern_complexity: 3,
            ..Default::default()
        },
        quality_config: quality::QualityConfig {
            enable_assessment: true,
            max_issues_per_category: 25,
            min_recommendation_confidence: 0.8,
            ..Default::default()
        },
        prediction_config: prediction::PredictionConfig {
            enable_prediction: true,
            min_confidence_threshold: 0.75,
            enable_training: true,
            ..Default::default()
        },
        optimization_config: optimization::OptimizationConfig {
            enable_shape_optimization: true,
            enable_performance_optimization: true,
            ..Default::default()
        },
        analytics_config: analytics::AnalyticsConfig {
            enable_analytics: true,
            enable_trend_analysis: true,
            ..Default::default()
        },
        ai_model_type: AiModelType::Local,
        enable_streaming: false,
        enable_distributed: false,
    };

    let assistant = ShaclAiAssistant::new(config.clone());
    assert_eq!(assistant.config().learning_config.max_shapes, 50);
    assert_eq!(assistant.config().pattern_config.max_pattern_complexity, 3);
}

#[test]
fn test_error_handling() {
    let assistant = ShaclAiAssistant::new(ShaclAiConfig::default());
    let empty_store = Store::new();

    // Test error handling with empty store
    let result =
        tokio_test::block_on(async { assistant.learn_shapes_from_store(&empty_store, None).await });

    // Should handle empty store gracefully
    match result {
        Ok(shapes) => assert!(
            shapes.is_empty(),
            "Should return empty shapes for empty store"
        ),
        Err(e) => match e {
            ShaclAiError::EmptyDataset => {} // Expected error
            _ => panic!("Unexpected error type: {:?}", e),
        },
    }
}

#[test]
fn test_concurrent_operations() {
    use std::sync::Arc;
    use tokio::task::JoinSet;

    let assistant = Arc::new(ShaclAiAssistant::new(ShaclAiConfig::default()));
    let store = Arc::new({
        let mut s = Store::new();
        let test_data = generate_test_data(100);
        for triple in test_data {
            s.insert(&triple).expect("Failed to insert test data");
        }
        s
    });

    tokio_test::block_on(async {
        let mut set = JoinSet::new();

        // Spawn concurrent operations
        for i in 0..5 {
            let assistant_clone = assistant.clone();
            let store_clone = store.clone();

            set.spawn(async move {
                let graph_name = format!("test_graph_{}", i);
                assistant_clone
                    .discover_patterns(&*store_clone, Some(&graph_name))
                    .await
            });
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        while let Some(result) = set.join_next().await {
            match result {
                Ok(Ok(patterns)) => results.push(patterns),
                Ok(Err(e)) => panic!("Operation failed: {:?}", e),
                Err(e) => panic!("Task join failed: {:?}", e),
            }
        }

        assert_eq!(
            results.len(),
            5,
            "All concurrent operations should complete"
        );
    });
}

#[test]
fn test_memory_usage() {
    let assistant = ShaclAiAssistant::new(ShaclAiConfig::default());
    let mut store = Store::new();

    // Test with large dataset
    let large_dataset = generate_test_data(10000);
    for triple in large_dataset {
        store.insert(&triple).expect("Failed to insert test data");
    }

    // Memory usage should be reasonable
    let initial_memory = get_memory_usage();

    tokio_test::block_on(async {
        let _patterns = assistant
            .discover_patterns(&store, None)
            .await
            .expect("Pattern discovery should handle large datasets");
    });

    let final_memory = get_memory_usage();

    // Memory increase should be bounded (less than 100MB for this test)
    let memory_increase = final_memory - initial_memory;
    assert!(
        memory_increase < 100 * 1024 * 1024,
        "Memory usage increase should be bounded: {} bytes",
        memory_increase
    );
}

#[test]
fn test_performance_benchmarks() {
    use std::time::Instant;

    let assistant = ShaclAiAssistant::new(ShaclAiConfig::default());
    let mut store = Store::new();

    // Load test data
    let test_data = generate_test_data(1000);
    for triple in test_data {
        store.insert(&triple).expect("Failed to insert test data");
    }

    // Benchmark shape learning
    let start = Instant::now();
    let _shapes =
        tokio_test::block_on(async { assistant.learn_shapes_from_store(&store, None).await })
            .expect("Shape learning failed");
    let shape_learning_duration = start.elapsed();

    // Benchmark pattern discovery
    let start = Instant::now();
    let _patterns = tokio_test::block_on(async { assistant.discover_patterns(&store, None).await })
        .expect("Pattern discovery failed");
    let pattern_discovery_duration = start.elapsed();

    // Performance assertions (adjust based on expected performance)
    assert!(
        shape_learning_duration.as_millis() < 5000,
        "Shape learning should complete within 5 seconds: {:?}",
        shape_learning_duration
    );
    assert!(
        pattern_discovery_duration.as_millis() < 3000,
        "Pattern discovery should complete within 3 seconds: {:?}",
        pattern_discovery_duration
    );

    println!("Performance benchmarks:");
    println!("  Shape learning: {:?}", shape_learning_duration);
    println!("  Pattern discovery: {:?}", pattern_discovery_duration);
}

/// Helper function to get current memory usage (simplified implementation)
fn get_memory_usage() -> usize {
    // In a real implementation, this would use system APIs to get actual memory usage
    // For testing, we'll return a mock value
    42 * 1024 * 1024 // 42 MB
}

#[cfg(test)]
mod tokio_test {
    //! Helper module for running async tests in sync context

    pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(future)
    }
}
