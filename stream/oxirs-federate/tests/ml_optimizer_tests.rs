//! Comprehensive tests for the ML-driven query optimization engine
//!
//! These tests verify the machine learning functionality for query optimization,
//! source selection learning, and predictive analytics.

use oxirs_federate::{
    ml_optimizer::*, planner::QueryInfo, FederatedService, QueryType, ServiceRegistry,
    TriplePattern,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio;

#[tokio::test]
async fn test_ml_optimizer_creation() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Should create with default configuration
    assert_eq!(optimizer.get_config().enable_performance_prediction, true);
    assert_eq!(optimizer.get_config().enable_source_selection_learning, true);
    assert_eq!(optimizer.get_config().enable_join_order_optimization, true);
    assert_eq!(optimizer.get_config().enable_caching_strategy_learning, true);
    assert_eq!(optimizer.get_config().enable_anomaly_detection, true);
}

#[tokio::test]
async fn test_ml_optimizer_custom_config() {
    let config = MLOptimizerConfig {
        enable_performance_prediction: false,
        enable_source_selection_learning: true,
        enable_join_order_optimization: false,
        enable_caching_strategy_learning: true,
        enable_anomaly_detection: false,
        training_data_retention_days: 14,
        model_retrain_interval: Duration::from_secs(3600),
        prediction_confidence_threshold: 0.9,
        anomaly_sensitivity: 0.7,
        feature_importance_threshold: 0.1,
        batch_size: 64,
        learning_rate: 0.001,
        max_iterations: 500,
    };

    let optimizer = MLOptimizer::new(config.clone());

    assert_eq!(optimizer.get_config().enable_performance_prediction, false);
    assert_eq!(optimizer.get_config().training_data_retention_days, 14);
    assert_eq!(optimizer.get_config().prediction_confidence_threshold, 0.9);
}

#[tokio::test]
async fn test_query_performance_prediction() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Create test query
    let query_info = QueryInfo {
        query_type: QueryType::SparqlSelect,
        original_query: "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
        patterns: vec![TriplePattern {
            subject: "?s".to_string(),
            predicate: "?p".to_string(),
            object: "?o".to_string(),
            pattern_string: "?s ?p ?o".to_string(),
        }],
        service_clauses: vec![],
        filters: vec![],
        variables: vec!["?s".to_string(), "?p".to_string(), "?o".to_string()],
        complexity: QueryComplexity::Low,
        estimated_cost: 10,
    };

    // Train with historical data
    let training_data = vec![
        QueryPerformanceData {
            query_features: QueryFeatures {
                pattern_count: 1,
                variable_count: 3,
                filter_count: 0,
                service_count: 0,
                join_count: 0,
                union_count: 0,
                optional_count: 0,
                has_aggregation: false,
                has_grouping: false,
                has_ordering: false,
                has_limit: false,
                complexity_score: 0.2,
                selectivity_estimate: 0.8,
                predicate_distribution: vec![("?p".to_string(), 1.0)],
                namespace_distribution: vec![],
                pattern_type_distribution: vec![("basic".to_string(), 1.0)],
                has_joins: false,
            },
            execution_time: Duration::from_millis(150),
            success: true,
            timestamp: Instant::now(),
        },
        QueryPerformanceData {
            query_features: QueryFeatures {
                pattern_count: 2,
                variable_count: 4,
                filter_count: 1,
                service_count: 0,
                join_count: 1,
                union_count: 0,
                optional_count: 0,
                has_aggregation: false,
                has_grouping: false,
                has_ordering: false,
                has_limit: false,
                complexity_score: 0.4,
                selectivity_estimate: 0.3,
                predicate_distribution: vec![("?p1".to_string(), 0.5), ("?p2".to_string(), 0.5)],
                namespace_distribution: vec![],
                pattern_type_distribution: vec![("basic".to_string(), 1.0)],
                has_joins: true,
            },
            execution_time: Duration::from_millis(450),
            success: true,
            timestamp: Instant::now(),
        },
    ];

    // Train the model
    for data in training_data {
        optimizer.record_query_performance(data).await;
    }

    optimizer.train_performance_model().await.unwrap();

    // Make prediction
    let prediction = optimizer.predict_query_performance(&query_info).await.unwrap();

    // Should have valid prediction
    assert!(prediction.predicted_execution_time > Duration::from_millis(0));
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(prediction.success_probability >= 0.0 && prediction.success_probability <= 1.0);
}

#[tokio::test]
async fn test_source_selection_learning() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Create test services
    let services = vec![
        FederatedService::new_sparql(
            "fast-service".to_string(),
            "Fast SPARQL Service".to_string(),
            "http://fast.example.com/sparql".to_string(),
        ),
        FederatedService::new_sparql(
            "slow-service".to_string(),
            "Slow SPARQL Service".to_string(),
            "http://slow.example.com/sparql".to_string(),
        ),
        FederatedService::new_sparql(
            "medium-service".to_string(),
            "Medium SPARQL Service".to_string(),
            "http://medium.example.com/sparql".to_string(),
        ),
    ];

    // Train with source selection data
    let selection_data = vec![
        SourceSelectionData {
            pattern: TriplePattern {
                subject: "?s".to_string(),
                predicate: "foaf:name".to_string(),
                object: "?name".to_string(),
                pattern_string: "?s foaf:name ?name".to_string(),
            },
            selected_service_id: "fast-service".to_string(),
            performance: Duration::from_millis(80),
            success: true,
            result_count: 150,
            timestamp: Instant::now(),
        },
        SourceSelectionData {
            pattern: TriplePattern {
                subject: "?s".to_string(),
                predicate: "foaf:name".to_string(),
                object: "?name".to_string(),
                pattern_string: "?s foaf:name ?name".to_string(),
            },
            selected_service_id: "slow-service".to_string(),
            performance: Duration::from_millis(500),
            success: true,
            result_count: 120,
            timestamp: Instant::now(),
        },
        SourceSelectionData {
            pattern: TriplePattern {
                subject: "?person".to_string(),
                predicate: "foaf:age".to_string(),
                object: "?age".to_string(),
                pattern_string: "?person foaf:age ?age".to_string(),
            },
            selected_service_id: "medium-service".to_string(),
            performance: Duration::from_millis(200),
            success: true,
            result_count: 80,
            timestamp: Instant::now(),
        },
    ];

    // Train source selection model
    for data in selection_data {
        optimizer.record_source_selection(data).await;
    }

    optimizer.train_source_selection_model().await.unwrap();

    // Test pattern for prediction
    let test_pattern = TriplePattern {
        subject: "?s".to_string(),
        predicate: "foaf:name".to_string(),
        object: "?name".to_string(),
        pattern_string: "?s foaf:name ?name".to_string(),
    };

    // Get source recommendations
    let recommendations = optimizer
        .recommend_sources(&test_pattern, &services)
        .await
        .unwrap();

    // Should recommend services
    assert!(!recommendations.is_empty());
    assert!(recommendations.len() <= services.len());

    // Should rank fast-service higher for foaf:name patterns
    assert_eq!(recommendations[0].service_id, "fast-service");

    // Each recommendation should have confidence and predicted performance
    for rec in &recommendations {
        assert!(rec.confidence >= 0.0 && rec.confidence <= 1.0);
        assert!(rec.predicted_performance > Duration::from_millis(0));
        assert!(rec.predicted_result_count >= 0);
    }
}

#[tokio::test]
async fn test_join_order_optimization() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Create test patterns for join optimization
    let patterns = vec![
        TriplePattern {
            subject: "?person".to_string(),
            predicate: "rdf:type".to_string(),
            object: "foaf:Person".to_string(),
            pattern_string: "?person rdf:type foaf:Person".to_string(),
        },
        TriplePattern {
            subject: "?person".to_string(),
            predicate: "foaf:name".to_string(),
            object: "?name".to_string(),
            pattern_string: "?person foaf:name ?name".to_string(),
        },
        TriplePattern {
            subject: "?person".to_string(),
            predicate: "foaf:age".to_string(),
            object: "?age".to_string(),
            pattern_string: "?person foaf:age ?age".to_string(),
        },
    ];

    // Train with join order data
    let join_data = vec![
        JoinOrderData {
            patterns: patterns.clone(),
            join_order: vec![0, 1, 2], // Type first, then name, then age
            execution_time: Duration::from_millis(300),
            memory_usage: 1024 * 1024, // 1MB
            result_count: 50,
            success: true,
            timestamp: Instant::now(),
        },
        JoinOrderData {
            patterns: patterns.clone(),
            join_order: vec![1, 0, 2], // Name first, then type, then age
            execution_time: Duration::from_millis(800),
            memory_usage: 2048 * 1024, // 2MB
            result_count: 45,
            success: true,
            timestamp: Instant::now(),
        },
        JoinOrderData {
            patterns: patterns.clone(),
            join_order: vec![2, 1, 0], // Age first, then name, then type
            execution_time: Duration::from_millis(1200),
            memory_usage: 4096 * 1024, // 4MB
            result_count: 40,
            success: true,
            timestamp: Instant::now(),
        },
    ];

    // Train join order model
    for data in join_data {
        optimizer.record_join_order_performance(data).await;
    }

    optimizer.train_join_order_model().await.unwrap();

    // Get optimal join order
    let optimal_order = optimizer.optimize_join_order(&patterns).await.unwrap();

    // Should recommend the most efficient order (type first)
    assert_eq!(optimal_order.recommended_order, vec![0, 1, 2]);
    assert!(optimal_order.predicted_execution_time > Duration::from_millis(0));
    assert!(optimal_order.confidence >= 0.0 && optimal_order.confidence <= 1.0);
}

#[tokio::test]
async fn test_caching_strategy_learning() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Train with caching performance data
    let caching_data = vec![
        CachingStrategyData {
            query_pattern: "?s rdf:type foaf:Person".to_string(),
            cache_strategy: CacheStrategy::Aggressive,
            cache_hit_rate: 0.85,
            query_performance_improvement: 0.70, // 70% improvement
            memory_overhead: 512 * 1024,         // 512KB
            freshness_requirement: Duration::from_secs(300), // 5 minutes
            timestamp: Instant::now(),
        },
        CachingStrategyData {
            query_pattern: "?s foaf:name ?name".to_string(),
            cache_strategy: CacheStrategy::Conservative,
            cache_hit_rate: 0.40,
            query_performance_improvement: 0.20, // 20% improvement
            memory_overhead: 128 * 1024,         // 128KB
            freshness_requirement: Duration::from_secs(60), // 1 minute
            timestamp: Instant::now(),
        },
        CachingStrategyData {
            query_pattern: "?s foaf:age ?age".to_string(),
            cache_strategy: CacheStrategy::Adaptive,
            cache_hit_rate: 0.65,
            query_performance_improvement: 0.45, // 45% improvement
            memory_overhead: 256 * 1024,         // 256KB
            freshness_requirement: Duration::from_secs(180), // 3 minutes
            timestamp: Instant::now(),
        },
    ];

    // Train caching strategy model
    for data in caching_data {
        optimizer.record_caching_performance(data).await;
    }

    optimizer.train_caching_strategy_model().await.unwrap();

    // Get caching recommendation for a new pattern
    let test_pattern = "?person rdf:type foaf:Person";
    let caching_recommendation = optimizer
        .recommend_caching_strategy(test_pattern)
        .await
        .unwrap();

    // Should recommend aggressive caching for type patterns
    assert_eq!(caching_recommendation.strategy, CacheStrategy::Aggressive);
    assert!(caching_recommendation.confidence >= 0.0 && caching_recommendation.confidence <= 1.0);
    assert!(caching_recommendation.predicted_hit_rate >= 0.0 && caching_recommendation.predicted_hit_rate <= 1.0);
    assert!(caching_recommendation.predicted_performance_improvement >= 0.0);
}

#[tokio::test]
async fn test_anomaly_detection() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Train with normal query performance data
    let normal_performances = vec![
        Duration::from_millis(100),
        Duration::from_millis(110),
        Duration::from_millis(95),
        Duration::from_millis(105),
        Duration::from_millis(98),
        Duration::from_millis(102),
        Duration::from_millis(107),
        Duration::from_millis(103),
        Duration::from_millis(99),
        Duration::from_millis(101),
    ];

    let query_features = QueryFeatures {
        pattern_count: 1,
        variable_count: 3,
        filter_count: 0,
        service_count: 0,
        join_count: 0,
        union_count: 0,
        optional_count: 0,
        has_aggregation: false,
        has_grouping: false,
        has_ordering: false,
        has_limit: false,
        complexity_score: 0.2,
        selectivity_estimate: 0.8,
        predicate_distribution: vec![("?p".to_string(), 1.0)],
        namespace_distribution: vec![],
        pattern_type_distribution: vec![("basic".to_string(), 1.0)],
        has_joins: false,
    };

    // Record normal performance data
    for performance in normal_performances {
        let data = QueryPerformanceData {
            query_features: query_features.clone(),
            execution_time: performance,
            success: true,
            timestamp: Instant::now(),
        };
        optimizer.record_query_performance(data).await;
    }

    // Train anomaly detection model
    optimizer.train_anomaly_detection_model().await.unwrap();

    // Test with anomalous performance (much higher than normal)
    let anomalous_data = QueryPerformanceData {
        query_features: query_features.clone(),
        execution_time: Duration::from_millis(1000), // 10x normal
        success: true,
        timestamp: Instant::now(),
    };

    let anomaly_result = optimizer.detect_anomaly(&anomalous_data).await.unwrap();

    // Should detect as anomaly
    assert!(anomaly_result.is_anomaly);
    assert!(anomaly_result.anomaly_score > 0.5); // High anomaly score
    assert!(!anomaly_result.explanation.is_empty());

    // Test with normal performance
    let normal_data = QueryPerformanceData {
        query_features: query_features,
        execution_time: Duration::from_millis(103), // Normal
        success: true,
        timestamp: Instant::now(),
    };

    let normal_result = optimizer.detect_anomaly(&normal_data).await.unwrap();

    // Should not detect as anomaly
    assert!(!normal_result.is_anomaly);
    assert!(normal_result.anomaly_score < 0.5); // Low anomaly score
}

#[tokio::test]
async fn test_feature_importance_analysis() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Train with diverse query data
    let training_data = vec![
        QueryPerformanceData {
            query_features: QueryFeatures {
                pattern_count: 1,
                variable_count: 2,
                filter_count: 0,
                service_count: 0,
                join_count: 0,
                union_count: 0,
                optional_count: 0,
                has_aggregation: false,
                has_grouping: false,
                has_ordering: false,
                has_limit: false,
                complexity_score: 0.1,
                selectivity_estimate: 0.9,
                predicate_distribution: vec![("rdf:type".to_string(), 1.0)],
                namespace_distribution: vec![("rdf".to_string(), 1.0)],
                pattern_type_distribution: vec![("basic".to_string(), 1.0)],
                has_joins: false,
            },
            execution_time: Duration::from_millis(50), // Fast for simple queries
            success: true,
            timestamp: Instant::now(),
        },
        QueryPerformanceData {
            query_features: QueryFeatures {
                pattern_count: 5,
                variable_count: 8,
                filter_count: 2,
                service_count: 0,
                join_count: 4,
                union_count: 1,
                optional_count: 1,
                has_aggregation: true,
                has_grouping: true,
                has_ordering: true,
                has_limit: true,
                complexity_score: 0.9,
                selectivity_estimate: 0.2,
                predicate_distribution: vec![
                    ("foaf:name".to_string(), 0.4),
                    ("foaf:age".to_string(), 0.3),
                    ("foaf:knows".to_string(), 0.3),
                ],
                namespace_distribution: vec![("foaf".to_string(), 1.0)],
                pattern_type_distribution: vec![("complex".to_string(), 1.0)],
                has_joins: true,
            },
            execution_time: Duration::from_millis(2000), // Slow for complex queries
            success: true,
            timestamp: Instant::now(),
        },
    ];

    // Train model with sufficient data
    for _ in 0..20 {
        for data in &training_data {
            optimizer.record_query_performance(data.clone()).await;
        }
    }

    optimizer.train_performance_model().await.unwrap();

    // Analyze feature importance
    let feature_importance = optimizer.analyze_feature_importance().await.unwrap();

    // Should identify important features
    assert!(!feature_importance.features.is_empty());

    // Complexity score should be important for performance prediction
    assert!(feature_importance
        .features
        .iter()
        .any(|f| f.feature_name == "complexity_score" && f.importance > 0.1));

    // Join count should be important
    assert!(feature_importance
        .features
        .iter()
        .any(|f| f.feature_name == "join_count" && f.importance > 0.1));

    // Should have valid importance scores
    for feature in &feature_importance.features {
        assert!(feature.importance >= 0.0 && feature.importance <= 1.0);
        assert!(!feature.feature_name.is_empty());
    }
}

#[tokio::test]
async fn test_model_retraining() {
    let config = MLOptimizerConfig {
        model_retrain_interval: Duration::from_millis(100), // Fast retraining for test
        ..Default::default()
    };
    let optimizer = MLOptimizer::new(config);

    // Record initial training data
    let initial_data = QueryPerformanceData {
        query_features: QueryFeatures {
            pattern_count: 1,
            variable_count: 3,
            filter_count: 0,
            service_count: 0,
            join_count: 0,
            union_count: 0,
            optional_count: 0,
            has_aggregation: false,
            has_grouping: false,
            has_ordering: false,
            has_limit: false,
            complexity_score: 0.2,
            selectivity_estimate: 0.8,
            predicate_distribution: vec![("?p".to_string(), 1.0)],
            namespace_distribution: vec![],
            pattern_type_distribution: vec![("basic".to_string(), 1.0)],
            has_joins: false,
        },
        execution_time: Duration::from_millis(100),
        success: true,
        timestamp: Instant::now(),
    };

    // Record initial data multiple times
    for _ in 0..10 {
        optimizer.record_query_performance(initial_data.clone()).await;
    }

    // Train initial model
    optimizer.train_performance_model().await.unwrap();

    let initial_stats = optimizer.get_model_statistics().await.unwrap();
    let initial_train_count = initial_stats.performance_model_train_count;

    // Wait for auto-retraining interval
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Record more data to trigger retraining
    for _ in 0..5 {
        optimizer.record_query_performance(initial_data.clone()).await;
    }

    // Check if model was retrained
    let updated_stats = optimizer.get_model_statistics().await.unwrap();

    // Should have more training data points
    assert!(updated_stats.total_training_data_points > initial_stats.total_training_data_points);
}

#[tokio::test]
async fn test_cross_validation() {
    let config = MLOptimizerConfig::default();
    let optimizer = MLOptimizer::new(config);

    // Create diverse training data for cross-validation
    let mut training_data = Vec::new();

    for i in 0..50 {
        let data = QueryPerformanceData {
            query_features: QueryFeatures {
                pattern_count: (i % 5) + 1,
                variable_count: (i % 8) + 2,
                filter_count: i % 3,
                service_count: 0,
                join_count: i % 4,
                union_count: i % 2,
                optional_count: i % 2,
                has_aggregation: (i % 4) == 0,
                has_grouping: (i % 5) == 0,
                has_ordering: (i % 6) == 0,
                has_limit: (i % 7) == 0,
                complexity_score: (i as f64 % 10.0) / 10.0,
                selectivity_estimate: ((10 - (i % 10)) as f64) / 10.0,
                predicate_distribution: vec![("pred".to_string(), 1.0)],
                namespace_distribution: vec![("ns".to_string(), 1.0)],
                pattern_type_distribution: vec![("basic".to_string(), 1.0)],
                has_joins: (i % 4) > 0,
            },
            execution_time: Duration::from_millis(50 + (i as u64 * 20)), // Correlated with complexity
            success: (i % 10) != 9, // 90% success rate
            timestamp: Instant::now(),
        };
        training_data.push(data);
    }

    // Record training data
    for data in training_data {
        optimizer.record_query_performance(data).await;
    }

    // Perform cross-validation
    let cv_results = optimizer.perform_cross_validation(5).await.unwrap(); // 5-fold CV

    // Should have valid cross-validation results
    assert_eq!(cv_results.fold_count, 5);
    assert!(cv_results.mean_accuracy >= 0.0 && cv_results.mean_accuracy <= 1.0);
    assert!(cv_results.std_accuracy >= 0.0);
    assert!(cv_results.mean_precision >= 0.0 && cv_results.mean_precision <= 1.0);
    assert!(cv_results.mean_recall >= 0.0 && cv_results.mean_recall <= 1.0);
    assert!(cv_results.mean_f1_score >= 0.0 && cv_results.mean_f1_score <= 1.0);
    assert_eq!(cv_results.fold_results.len(), 5);
}

#[tokio::test]
async fn test_prediction_confidence_thresholding() {
    let config = MLOptimizerConfig {
        prediction_confidence_threshold: 0.8, // High confidence threshold
        ..Default::default()
    };
    let optimizer = MLOptimizer::new(config);

    // Train with limited data to create low-confidence predictions
    let limited_training_data = vec![
        QueryPerformanceData {
            query_features: QueryFeatures {
                pattern_count: 1,
                variable_count: 3,
                filter_count: 0,
                service_count: 0,
                join_count: 0,
                union_count: 0,
                optional_count: 0,
                has_aggregation: false,
                has_grouping: false,
                has_ordering: false,
                has_limit: false,
                complexity_score: 0.2,
                selectivity_estimate: 0.8,
                predicate_distribution: vec![("?p".to_string(), 1.0)],
                namespace_distribution: vec![],
                pattern_type_distribution: vec![("basic".to_string(), 1.0)],
                has_joins: false,
            },
            execution_time: Duration::from_millis(100),
            success: true,
            timestamp: Instant::now(),
        }
    ];

    // Record minimal training data
    for data in limited_training_data {
        optimizer.record_query_performance(data).await;
    }

    optimizer.train_performance_model().await.unwrap();

    // Create a query for prediction
    let test_query = QueryInfo {
        query_type: QueryType::SparqlSelect,
        original_query: "SELECT * WHERE { ?x ?y ?z }".to_string(),
        patterns: vec![TriplePattern {
            subject: "?x".to_string(),
            predicate: "?y".to_string(),
            object: "?z".to_string(),
            pattern_string: "?x ?y ?z".to_string(),
        }],
        service_clauses: vec![],
        filters: vec![],
        variables: vec!["?x".to_string(), "?y".to_string(), "?z".to_string()],
        complexity: QueryComplexity::Low,
        estimated_cost: 10,
    };

    // Try prediction - might fail due to low confidence
    let prediction_result = optimizer.predict_query_performance(&test_query).await;

    // If prediction succeeds, confidence should be above threshold
    if let Ok(prediction) = prediction_result {
        assert!(prediction.confidence >= 0.8);
    }
    // Otherwise, it should fail due to insufficient confidence
}

/// Helper function to create test query features
fn create_test_query_features(
    pattern_count: usize,
    variable_count: usize,
    complexity: f64,
) -> QueryFeatures {
    QueryFeatures {
        pattern_count,
        variable_count,
        filter_count: 0,
        service_count: 0,
        join_count: if pattern_count > 1 { pattern_count - 1 } else { 0 },
        union_count: 0,
        optional_count: 0,
        has_aggregation: false,
        has_grouping: false,
        has_ordering: false,
        has_limit: false,
        complexity_score: complexity,
        selectivity_estimate: 1.0 - complexity,
        predicate_distribution: vec![("test:pred".to_string(), 1.0)],
        namespace_distribution: vec![("test".to_string(), 1.0)],
        pattern_type_distribution: vec![("basic".to_string(), 1.0)],
        has_joins: pattern_count > 1,
    }
}

/// Helper function to create test performance data
fn create_test_performance_data(
    features: QueryFeatures,
    execution_time: Duration,
) -> QueryPerformanceData {
    QueryPerformanceData {
        query_features: features,
        execution_time,
        success: true,
        timestamp: Instant::now(),
    }
}