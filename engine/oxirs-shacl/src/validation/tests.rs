//! Tests for the validation module

use super::*;
use crate::{Constraint, PropertyPath, ValidationConfig};
use indexmap::IndexMap;
use oxirs_core::{
    model::{Literal, NamedNode, Term},
    ConcreteStore,
};
use std::time::Duration;

#[test]
fn test_validation_engine_creation() {
    let shapes = IndexMap::new();
    let config = ValidationConfig::default();
    let engine = ValidationEngine::new(&shapes, config.clone());

    // Test that engine was created successfully - since fields are private,
    // we just verify creation works and basic functionality
    assert!(!engine.is_optimization_enabled());
    assert_eq!(engine.get_cache_hit_rate(), 0.0);
}

#[test]
fn test_validation_violation() {
    let focus_node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
    let shape_id = ShapeId::new("http://example.org/PersonShape");
    let component_id = ConstraintComponentId::new("sh:ClassConstraintComponent");

    let violation = ValidationViolation::new(
        focus_node.clone(),
        shape_id.clone(),
        component_id.clone(),
        Severity::Violation,
    )
    .with_message("Test violation".to_string())
    .with_detail("test_key".to_string(), "test_value".to_string());

    assert_eq!(violation.focus_node, focus_node);
    assert_eq!(violation.source_shape, shape_id);
    assert_eq!(violation.source_constraint_component, component_id);
    assert_eq!(violation.result_severity, Severity::Violation);
    assert_eq!(violation.result_message, Some("Test violation".to_string()));
    assert_eq!(
        violation.details.get("test_key"),
        Some(&"test_value".to_string())
    );
}

#[test]
fn test_validation_stats() {
    let mut stats = ValidationStats::default();

    let duration = Duration::from_millis(100);
    stats.record_constraint_evaluation("ClassConstraint".to_string(), duration);
    stats.record_constraint_evaluation("ClassConstraint".to_string(), duration);

    assert_eq!(stats.total_constraint_evaluations, 2);
    assert_eq!(
        stats.constraint_evaluation_times.get("ClassConstraint"),
        Some(&Duration::from_millis(200))
    );
}

#[test]
fn test_constraint_evaluation_result() {
    // Test satisfied result
    let satisfied = ConstraintEvaluationResult::satisfied();
    assert!(satisfied.is_satisfied());
    assert!(!satisfied.is_violated());
    assert!(satisfied.violating_value().is_none());
    assert!(satisfied.message().is_none());

    // Test violated result
    let violating_value = Term::Literal(Literal::new("invalid"));
    let message = "Constraint violated".to_string();
    let violated =
        ConstraintEvaluationResult::violated(Some(violating_value.clone()), Some(message.clone()));

    assert!(!violated.is_satisfied());
    assert!(violated.is_violated());
    assert_eq!(violated.violating_value(), Some(&violating_value));
    assert_eq!(violated.message(), Some(message.as_str()));
}

#[test]
fn test_constraint_cache_key() {
    let focus_node = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
    let shape_id = ShapeId::new("test_shape");
    let component_id = ConstraintComponentId::new("sh:NodeKindConstraintComponent");
    let path = PropertyPath::Predicate(NamedNode::new("http://example.org/prop").unwrap());

    let key1 = ConstraintCacheKey {
        focus_node: focus_node.clone(),
        shape_id: shape_id.clone(),
        constraint_component_id: component_id.clone(),
        property_path: Some(path.clone()),
    };

    let key2 = ConstraintCacheKey {
        focus_node: focus_node.clone(),
        shape_id: shape_id.clone(),
        constraint_component_id: component_id.clone(),
        property_path: Some(path.clone()),
    };

    assert_eq!(key1, key2);
    assert_eq!(key1.clone(), key1);
}

#[test]
fn test_constraint_cache() {
    let cache = ConstraintCache::new();

    let key = ConstraintCacheKey {
        focus_node: Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
        shape_id: ShapeId::new("test_shape"),
        constraint_component_id: ConstraintComponentId::new("sh:NodeKindConstraintComponent"),
        property_path: None,
    };

    let result = ConstraintEvaluationResult::satisfied();

    // Test cache miss
    assert!(cache.get(&key).is_none());
    let (hits, misses) = cache.stats();
    assert_eq!(hits, 0);
    assert_eq!(misses, 1);

    // Insert and test cache hit
    cache.insert(key.clone(), result.clone());
    let cached_result = cache.get(&key);
    assert!(cached_result.is_some());
    assert!(cached_result.unwrap().is_satisfied());

    let (hits, misses) = cache.stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);

    // Test hit rate
    assert_eq!(cache.hit_rate(), 0.5);
}

#[test]
fn test_inheritance_cache() {
    let cache = InheritanceCache::new();
    let shape_id = ShapeId::new("test_shape");
    let mut constraints = IndexMap::new();

    let constraint = Constraint::NodeKind(crate::constraints::NodeKindConstraint {
        node_kind: crate::constraints::NodeKind::Iri,
    });
    constraints.insert(constraint.component_id(), constraint);

    // Test cache miss
    assert!(cache.get(&shape_id).is_none());

    // Insert and test cache hit
    cache.insert(shape_id.clone(), constraints.clone());
    let cached_constraints = cache.get(&shape_id);
    assert!(cached_constraints.is_some());
    assert_eq!(cached_constraints.unwrap().len(), 1);
}

#[test]
fn test_validation_engine_with_empty_shapes() {
    let shapes = IndexMap::new();
    let config = ValidationConfig::default();
    let mut engine = ValidationEngine::new(&shapes, config);

    let store = ConcreteStore::new().unwrap();
    let result = engine.validate_store(&store).unwrap();

    assert!(result.conforms());
    assert_eq!(result.violation_count(), 0);
}

#[test]
fn test_validation_engine_optimization() {
    let shapes = IndexMap::new();
    let config = ValidationConfig::default();
    let mut engine = ValidationEngine::new(&shapes, config);

    // Test optimization state
    assert!(!engine.is_optimization_enabled());

    // Enable optimization
    engine.enable_optimization();
    assert!(engine.is_optimization_enabled());

    // Disable optimization
    engine.disable_optimization();
    assert!(!engine.is_optimization_enabled());
}

#[test]
fn test_qualified_validation_stats() {
    let mut stats = QualifiedValidationStats::new();

    // Record some validations
    stats.record_validation(Duration::from_millis(10), true);
    stats.record_validation(Duration::from_millis(20), false);
    stats.record_validation(Duration::from_millis(15), true);

    assert_eq!(stats.total_validations(), 3);
    assert_eq!(stats.conformance_rate(), 2.0 / 3.0);
    assert!((stats.average_validation_time_ms() - 15.0).abs() < f64::EPSILON);

    // Test summary
    let summary = stats.summary();
    assert!(summary.contains("total: 3"));
    assert!(summary.contains("conformance_rate"));
}

fn create_test_engine() -> ValidationEngine<'static> {
    let shapes = Box::leak(Box::new(IndexMap::new()));
    let config = ValidationConfig::default();
    ValidationEngine::new(shapes, config)
}

#[test]
fn test_cache_manager() {
    let manager = CacheManager::new();

    // Test cache access
    let _constraint_cache = manager.constraint_cache();
    let _inheritance_cache = manager.inheritance_cache();

    // Test statistics
    let stats = manager.statistics();
    assert_eq!(stats.constraint_cache_size, 0);
    assert_eq!(stats.inheritance_cache_size, 0);

    // Test cache clearing
    manager.clear_all();

    let stats_after_clear = manager.statistics();
    assert_eq!(stats_after_clear.constraint_cache_size, 0);
}

#[test]
fn test_cache_statistics_summary() {
    let stats = CacheStatistics {
        constraint_cache_size: 100,
        inheritance_cache_size: 50,
        constraint_cache_hits: 80,
        constraint_cache_misses: 20,
        constraint_cache_hit_rate: 0.8,
    };

    let summary = stats.summary();
    assert!(summary.contains("constraint_size=100"));
    assert!(summary.contains("inheritance_size=50"));
    assert!(summary.contains("hit_rate=80.0%"));
}
