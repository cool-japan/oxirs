//! Prediction tests for oxirs-shacl-ai
//!
//! This module contains tests for the validation prediction functionality
//! of the AI-powered SHACL system.

use oxirs_core::model::{Literal, NamedNode, Term, Triple};
use oxirs_core::Store;
use oxirs_shacl_ai::*;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_basic() {
        // Basic prediction test
        let store = create_test_store();
        let shapes = create_test_shapes();

        // Create a predictor
        let mut predictor = ValidationPredictor::new();

        // Test prediction
        let result = predictor.predict_validation(&store, &shapes);

        // Basic assertions
        assert!(result.is_ok());
        let prediction = result.unwrap();
        assert!(prediction.confidence >= 0.0);
        assert!(prediction.confidence <= 1.0);
    }

    #[test]
    fn test_outcome_prediction() {
        let mut predictor = ValidationPredictor::new();
        let store = create_test_store();
        let shapes = create_test_shapes();

        let prediction = predictor.predict_validation(&store, &shapes).unwrap();

        assert!(prediction.outcome.success_probability >= 0.0);
        assert!(prediction.outcome.success_probability <= 1.0);
        assert!(prediction.outcome.estimated_violations >= 0);
    }

    #[test]
    fn test_performance_prediction() {
        let mut predictor = ValidationPredictor::new();
        let store = create_test_store();
        let shapes = create_test_shapes();

        let prediction = predictor.predict_validation(&store, &shapes).unwrap();

        assert!(prediction.performance.estimated_duration_ms > 0);
        assert!(prediction.performance.estimated_memory_mb >= 0.0);
    }

    #[test]
    fn test_prediction_caching() {
        let mut predictor = ValidationPredictor::new();
        let store = create_test_store();
        let shapes = create_test_shapes();

        // First prediction
        let start = std::time::Instant::now();
        let _prediction1 = predictor.predict_validation(&store, &shapes).unwrap();
        let first_duration = start.elapsed();

        // Second prediction (should be cached)
        let start = std::time::Instant::now();
        let _prediction2 = predictor.predict_validation(&store, &shapes).unwrap();
        let second_duration = start.elapsed();

        // Cache should make second prediction faster (or at least not significantly slower)
        assert!(second_duration <= first_duration * 2);
    }

    #[test]
    fn test_prediction_config() {
        let config = PredictionConfig::default();
        assert!(config.enable_prediction);
        assert!(config.min_confidence_threshold > 0.0);
        assert!(config.max_cache_size > 0);
    }

    fn create_test_store() -> Store {
        let mut store = Store::new();

        // Add some test data
        let person = NamedNode::new("http://example.org/Person").unwrap();
        let name = NamedNode::new("http://example.org/name").unwrap();
        let age = NamedNode::new("http://example.org/age").unwrap();

        let john = NamedNode::new("http://example.org/john").unwrap();
        store.insert(Triple::new(
            john.clone().into(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                .unwrap()
                .into(),
            person.into(),
        ));
        store.insert(Triple::new(
            john.clone().into(),
            name.into(),
            Literal::new_simple_literal("John Doe").into(),
        ));
        store.insert(Triple::new(
            john.into(),
            age.into(),
            Literal::new_typed_literal("30", oxirs_core::vocab::xsd::INTEGER).into(),
        ));

        store
    }

    fn create_test_shapes() -> Vec<Shape> {
        vec![create_person_shape()]
    }

    fn create_person_shape() -> Shape {
        use oxirs_shacl_ai::shape::*;

        let mut shape = Shape::new("http://example.org/PersonShape".to_string());
        shape.set_target_class("http://example.org/Person".to_string());

        let mut name_constraint = PropertyConstraint::new("http://example.org/name".to_string());
        name_constraint.min_count = Some(1);
        name_constraint.max_count = Some(1);
        shape.add_property_constraint(name_constraint);

        let mut age_constraint = PropertyConstraint::new("http://example.org/age".to_string());
        age_constraint.min_count = Some(1);
        age_constraint.max_count = Some(1);
        age_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#integer".to_string());
        shape.add_property_constraint(age_constraint);

        shape
    }
}
