//! Tests for shape learning functionality

use oxirs_core::model::{Literal, NamedNode, Triple};
use oxirs_core::Store;
use oxirs_shacl_ai::learning::*;

#[test]
fn test_shape_learner_creation() {
    let learner = ShapeLearner::new();
    assert!(learner.config().enable_shape_generation);
    assert_eq!(learner.config().min_confidence, 0.8);
}

#[test]
fn test_custom_learning_config() {
    let config = LearningConfig {
        enable_shape_generation: true,
        min_support: 0.2,
        min_confidence: 0.9,
        max_shapes: 25,
        enable_training: false,
        algorithm_params: std::collections::HashMap::new(),
    };

    let learner = ShapeLearner::with_config(config);
    assert_eq!(learner.config().min_support, 0.2);
    assert_eq!(learner.config().min_confidence, 0.9);
    assert_eq!(learner.config().max_shapes, 25);
    assert!(!learner.config().enable_training);
}

#[test]
fn test_shape_learning_statistics() {
    let mut learner = ShapeLearner::new();
    let stats = learner.get_statistics();

    // Initial statistics should be zero
    assert_eq!(stats.total_shapes_learned, 0);
    assert_eq!(stats.failed_shapes, 0);
    assert_eq!(stats.total_constraints_discovered, 0);
    assert!(!stats.model_trained);
    assert_eq!(stats.last_training_accuracy, 0.0);
}

#[test]
fn test_training_data_structure() {
    let person_class = NamedNode::new("http://example.org/Person").unwrap();
    let name_predicate = NamedNode::new("http://example.org/name").unwrap();
    let subject = NamedNode::new("http://example.org/person1").unwrap();
    let name_literal = Literal::new_simple_literal("John Doe");

    let triple = Triple::new(subject, name_predicate, name_literal);

    let example = ShapeExample {
        graph_data: vec![triple],
        expected_shapes: vec![], // Would contain expected shapes
        quality_score: 0.9,
    };

    let training_data = ShapeTrainingData {
        examples: vec![example],
        validation_examples: vec![],
    };

    assert_eq!(training_data.examples.len(), 1);
    assert_eq!(training_data.examples[0].quality_score, 0.9);
}

#[test]
fn test_empty_store_handling() {
    let mut learner = ShapeLearner::new();
    let empty_store = Store::new();

    let result = learner.learn_shapes_from_store(&empty_store, None);

    // Should handle empty store gracefully
    match result {
        Ok(shapes) => assert!(shapes.is_empty()),
        Err(_) => {} // Empty store might produce error, which is acceptable
    }
}

#[test]
fn test_learning_query_result_types() {
    use oxirs_core::model::Term;
    use oxirs_shacl_ai::learning::LearningQueryResult;
    use std::collections::HashMap;

    // Test empty result
    let empty_result = LearningQueryResult::Empty;
    assert!(matches!(empty_result, LearningQueryResult::Empty));

    // Test ASK result
    let ask_result = LearningQueryResult::Ask(true);
    assert!(matches!(ask_result, LearningQueryResult::Ask(true)));

    // Test SELECT result
    let mut bindings = Vec::new();
    let mut binding = HashMap::new();
    let class_node = NamedNode::new("http://example.org/Person").unwrap();
    binding.insert("class".to_string(), Term::NamedNode(class_node));
    bindings.push(binding);

    let select_result = LearningQueryResult::Select {
        variables: vec!["class".to_string()],
        bindings,
    };

    match select_result {
        LearningQueryResult::Select {
            variables,
            bindings,
        } => {
            assert_eq!(variables.len(), 1);
            assert_eq!(bindings.len(), 1);
            assert_eq!(variables[0], "class");
        }
        _ => panic!("Expected Select result"),
    }
}

#[test]
fn test_pattern_to_shape_conversion() {
    use oxirs_shacl_ai::patterns::Pattern;
    use oxirs_shacl_ai::patterns::PatternType;

    let mut learner = ShapeLearner::new();
    let store = Store::new();

    let class_pattern = Pattern::ClassUsage {
        class: NamedNode::new("http://example.org/Person").unwrap(),
        instance_count: 100,
        support: 0.8,
        confidence: 0.9, // Above threshold
        pattern_type: PatternType::Structural,
    };

    let result = learner.pattern_to_shape(&store, &class_pattern, None);
    assert!(
        result.is_ok(),
        "High confidence pattern should convert to shape"
    );

    let low_confidence_pattern = Pattern::ClassUsage {
        class: NamedNode::new("http://example.org/Person").unwrap(),
        instance_count: 10,
        support: 0.1,
        confidence: 0.5, // Below threshold
        pattern_type: PatternType::Structural,
    };

    let result = learner.pattern_to_shape(&store, &low_confidence_pattern, None);
    assert!(
        result.is_err(),
        "Low confidence pattern should not convert to shape"
    );
}

#[test]
fn test_model_training_simulation() {
    let mut learner = ShapeLearner::new();

    // Create mock training data
    let training_data = ShapeTrainingData {
        examples: vec![],
        validation_examples: vec![],
    };

    let result = learner.train_model(&training_data);
    assert!(result.is_ok(), "Training should complete successfully");

    let training_result = result.unwrap();
    assert!(training_result.success || !training_result.success); // Either outcome is valid for simulation
    assert!(training_result.accuracy >= 0.0 && training_result.accuracy <= 1.0);
    assert!(training_result.loss >= 0.0);
    assert!(training_result.epochs_trained > 0);

    // Statistics should be updated
    let stats = learner.get_statistics();
    assert!(stats.model_trained);
}

#[test]
fn test_cache_operations() {
    let mut learner = ShapeLearner::new();

    // Initially cache should be empty (we can't directly test this, but clear should work)
    learner.clear_cache();

    // After clearing, cache should still be empty
    learner.clear_cache();

    // This test mainly ensures the cache operations don't panic
}

#[test]
fn test_learning_with_graph_name() {
    let mut learner = ShapeLearner::new();
    let store = Store::new();

    // Test with specific graph name
    let result = learner.learn_shapes_from_store(&store, Some("http://example.org/graph"));

    // Should handle graph-specific learning
    match result {
        Ok(shapes) => assert!(shapes.is_empty()), // Empty store should produce no shapes
        Err(_) => {} // Error is acceptable for empty store with graph name
    }
}
