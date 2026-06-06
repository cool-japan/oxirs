//! Tests for the ML predictor (model, feature extraction, histogram statistics).
//!
//! Split out from `ml_predictor.rs` to keep individual source files below the
//! workspace 2000-line refactor threshold.

#![cfg(test)]

use std::time::SystemTime;

use anyhow::Result;

use crate::advanced_optimizer::ml_predictor_features::{QueryCharacteristics, ValueHistogram};
use crate::advanced_optimizer::ml_predictor_model::MLConfig;
use crate::advanced_optimizer::ml_predictor_training::{MLPredictor, TrainingExample};
use crate::algebra::Algebra;

#[test]
fn test_ml_predictor_creation() -> Result<()> {
    let config = MLConfig::default();
    let predictor = MLPredictor::new(config)?;

    assert_eq!(predictor.training_data_count(), 0);
    assert_eq!(predictor.predictions_count(), 0);
    assert!(!predictor.should_use_ml()); // No model trained yet

    Ok(())
}

#[test]
fn test_feature_extraction() -> Result<()> {
    let config = MLConfig::default();
    let predictor = MLPredictor::new(config)?;

    let query = Algebra::Empty;
    let features = predictor.extract_features(&query);

    assert_eq!(features.len(), 13, "Should extract 13 features");

    Ok(())
}

#[test]
fn test_model_serialization() -> Result<()> {
    let config = MLConfig::default();
    let predictor = MLPredictor::new(config)?;

    let serialized = serde_json::to_string(&predictor)?;
    let deserialized: MLPredictor = serde_json::from_str(&serialized)?;

    assert_eq!(predictor.config.model_type, deserialized.config.model_type);

    Ok(())
}

#[test]
fn test_value_histogram_creation() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let histogram = ValueHistogram::from_data(&data, 5);

    assert_eq!(histogram.total_count, 10);
    assert_eq!(histogram.buckets.len(), 5);
    assert_eq!(histogram.min_value, 1.0);
    assert_eq!(histogram.max_value, 10.0);
}

#[test]
fn test_value_histogram_empty() {
    let histogram = ValueHistogram::empty();
    assert_eq!(histogram.total_count, 0);
    assert_eq!(histogram.buckets.len(), 0);
}

#[test]
fn test_value_histogram_uniform() {
    let data = vec![5.0; 100]; // All same value
    let histogram = ValueHistogram::from_data(&data, 10);

    assert_eq!(histogram.total_count, 100);
    assert_eq!(histogram.distinct_count, 1);
    assert_eq!(histogram.buckets.len(), 1);
}

#[test]
fn test_histogram_equality_selectivity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let histogram = ValueHistogram::from_data(&data, 5);

    // Test selectivity for values in the dataset
    let selectivity_5 = histogram.estimate_equality_selectivity(5.0);
    assert!(selectivity_5 > 0.0);
    assert!(selectivity_5 <= 1.0);

    // Test selectivity for value outside range
    let selectivity_100 = histogram.estimate_equality_selectivity(100.0);
    assert_eq!(selectivity_100, 0.0);
}

#[test]
fn test_histogram_range_selectivity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let histogram = ValueHistogram::from_data(&data, 5);

    // Full range should have selectivity ~1.0
    let selectivity_full = histogram.estimate_range_selectivity(1.0, 11.0);
    assert!(selectivity_full >= 0.9);
    assert!(selectivity_full <= 1.0);

    // Half range should have selectivity ~0.5
    let selectivity_half = histogram.estimate_range_selectivity(1.0, 5.5);
    assert!(selectivity_half >= 0.4);
    assert!(selectivity_half <= 0.6);

    // No overlap should have selectivity 0
    let selectivity_none = histogram.estimate_range_selectivity(100.0, 200.0);
    assert_eq!(selectivity_none, 0.0);
}

#[test]
fn test_histogram_cardinality_estimation() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let histogram = ValueHistogram::from_data(&data, 5);

    let selectivity = 0.5;
    let estimated_cardinality = histogram.estimate_cardinality(selectivity);
    assert_eq!(estimated_cardinality, 5);
}

#[test]
fn test_histogram_estimator_build() -> Result<()> {
    let config = MLConfig::default();
    let mut predictor = MLPredictor::new(config)?;

    // Add training examples
    let query_characteristics = QueryCharacteristics {
        triple_pattern_count: 5,
        join_count: 2,
        filter_count: 1,
        optional_count: 0,
        has_aggregation: false,
        has_sorting: false,
        estimated_cardinality: 1000,
        complexity_score: 10.0,
        query_graph_diameter: 2,
        avg_degree: 1.5,
        max_degree: 2,
    };

    for i in 0..100 {
        let features = vec![
            (i % 10) as f64,
            (i % 5) as f64,
            (i % 3) as f64,
            0.0,
            0.0,
            0.0,
            1000.0,
            2.0,
            1.5,
            2.0,
            0.0,
            0.0,
            0.0,
        ];
        let example = TrainingExample {
            features,
            target_cost: (i * 10) as f64,
            actual_cost: (i * 10) as f64,
            query_characteristics: query_characteristics.clone(),
            timestamp: SystemTime::now(),
        };
        predictor.add_training_example(example);
    }

    // Train model (which builds histograms)
    predictor.train_model()?;

    // Check histogram statistics
    let stats = predictor.get_histogram_statistics();
    assert!(stats.num_histograms > 0);
    assert!(stats.total_buckets > 0);

    Ok(())
}

#[test]
fn test_histogram_enhanced_prediction() -> Result<()> {
    let config = MLConfig::default();
    let mut predictor = MLPredictor::new(config)?;

    // Add training examples
    let query_characteristics = QueryCharacteristics {
        triple_pattern_count: 5,
        join_count: 2,
        filter_count: 1,
        optional_count: 0,
        has_aggregation: false,
        has_sorting: false,
        estimated_cardinality: 1000,
        complexity_score: 10.0,
        query_graph_diameter: 2,
        avg_degree: 1.5,
        max_degree: 2,
    };

    for i in 0..100 {
        let features = vec![
            (i % 10) as f64,
            (i % 5) as f64,
            1.0,
            0.0,
            0.0,
            0.0,
            1000.0,
            2.0,
            1.5,
            2.0,
            0.0,
            0.0,
            0.0,
        ];
        let example = TrainingExample {
            features,
            target_cost: (i * 10) as f64,
            actual_cost: (i * 10) as f64,
            query_characteristics: query_characteristics.clone(),
            timestamp: SystemTime::now(),
        };
        predictor.add_training_example(example);
    }

    // Train model
    predictor.train_model()?;

    // Make prediction with histogram enhancement
    let query = Algebra::Empty;
    let prediction = predictor.predict_cost_with_histogram(&query)?;

    assert!(prediction.predicted_cost >= 0.0);
    assert!(prediction.confidence >= 0.0);
    assert!(prediction.confidence <= 1.0);

    Ok(())
}

#[test]
fn test_histogram_cardinality_blending() -> Result<()> {
    let config = MLConfig::default();
    let mut predictor = MLPredictor::new(config)?;

    // Build some training data
    let query_characteristics = QueryCharacteristics {
        triple_pattern_count: 5,
        join_count: 2,
        filter_count: 1,
        optional_count: 0,
        has_aggregation: false,
        has_sorting: false,
        estimated_cardinality: 1000,
        complexity_score: 10.0,
        query_graph_diameter: 2,
        avg_degree: 1.5,
        max_degree: 2,
    };

    for i in 0..150 {
        let features = vec![
            5.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 2.0, 1.5, 2.0, 0.0, 0.0, 0.0,
        ];
        let example = TrainingExample {
            features,
            target_cost: 100.0 + i as f64,
            actual_cost: 100.0 + i as f64,
            query_characteristics: query_characteristics.clone(),
            timestamp: SystemTime::now(),
        };
        predictor.add_training_example(example);
    }

    predictor.train_model()?;

    // Get histogram-based estimate
    let features = vec![
        5.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 2.0, 1.5, 2.0, 0.0, 0.0, 0.0,
    ];
    let histogram_estimate = predictor.estimate_cardinality_with_histogram(&features);

    assert!(histogram_estimate.is_some());
    assert!(histogram_estimate.expect("histogram estimate") > 0.0);

    Ok(())
}
