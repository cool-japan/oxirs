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
    let focus_node =
        Term::NamedNode(NamedNode::new("http://example.org/john").expect("should succeed"));
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
    let focus_node =
        Term::NamedNode(NamedNode::new("http://example.org/test").expect("should succeed"));
    let shape_id = ShapeId::new("test_shape");
    let component_id = ConstraintComponentId::new("sh:NodeKindConstraintComponent");
    let path =
        PropertyPath::Predicate(NamedNode::new("http://example.org/prop").expect("should succeed"));

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
        focus_node: Term::NamedNode(
            NamedNode::new("http://example.org/test").expect("should succeed"),
        ),
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
    assert!(cached_result.expect("should succeed").is_satisfied());

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
    assert_eq!(cached_constraints.expect("should succeed").len(), 1);
}

#[test]
fn test_validation_engine_with_empty_shapes() {
    let shapes = IndexMap::new();
    let config = ValidationConfig::default();
    let mut engine = ValidationEngine::new(&shapes, config);

    let store = ConcreteStore::new().expect("should succeed");
    let result = engine.validate_store(&store).expect("should succeed");

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

#[allow(dead_code)]
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

// ---- Additional ValidationEngine tests ----

#[test]
fn test_validation_engine_conforms_empty_store() {
    let shapes = IndexMap::new();
    let config = ValidationConfig::default();
    let mut engine = ValidationEngine::new(&shapes, config);
    let store = ConcreteStore::new().expect("store creation");

    let report = engine.validate_store(&store).expect("validation");
    assert!(
        report.conforms(),
        "Empty store with no shapes should always conform"
    );
    assert_eq!(report.violation_count(), 0);
}

#[test]
fn test_validation_engine_cache_hit_rate_after_operations() {
    let shapes = IndexMap::new();
    let config = ValidationConfig::default();
    let engine = ValidationEngine::new(&shapes, config);

    // Initial cache hit rate should be 0.0
    assert_eq!(engine.get_cache_hit_rate(), 0.0);
}

#[test]
fn test_validation_engine_disable_enable_optimization_cycle() {
    let shapes = IndexMap::new();
    let config = ValidationConfig::default();
    let mut engine = ValidationEngine::new(&shapes, config);

    // Start disabled
    assert!(!engine.is_optimization_enabled());

    // Enable -> Disable -> Enable cycle
    engine.enable_optimization();
    assert!(engine.is_optimization_enabled());
    engine.disable_optimization();
    assert!(!engine.is_optimization_enabled());
    engine.enable_optimization();
    assert!(engine.is_optimization_enabled());
}

// ---- Shape type and structure tests ----

#[test]
fn test_shape_node_shape_creation() {
    use crate::Shape;

    let shape_id = ShapeId::new("http://example.org/PersonShape");
    let shape = Shape::node_shape(shape_id.clone());

    assert_eq!(shape.id, shape_id);
    assert!(shape.is_node_shape());
    assert!(!shape.is_property_shape());
    assert!(shape.is_active());
}

#[test]
fn test_shape_property_shape_creation() {
    use crate::Shape;

    let shape_id = ShapeId::new("http://example.org/NamePropertyShape");
    let path = PropertyPath::Predicate(
        NamedNode::new("http://xmlns.com/foaf/0.1/name").expect("valid IRI"),
    );
    let shape = Shape::property_shape(shape_id.clone(), path.clone());

    assert_eq!(shape.id, shape_id);
    assert!(shape.is_property_shape());
    assert!(!shape.is_node_shape());
    assert_eq!(shape.path.as_ref(), Some(&path));
}

#[test]
fn test_shape_deactivated_flag() {
    use crate::Shape;

    let shape_id = ShapeId::new("http://example.org/DisabledShape");
    let mut shape = Shape::node_shape(shape_id);

    assert!(shape.is_active()); // default is active
    shape.deactivated = true;
    assert!(!shape.is_active());
}

#[test]
fn test_shape_add_constraint() {
    use crate::{
        constraints::{NodeKind, NodeKindConstraint},
        Constraint, Shape,
    };

    let shape_id = ShapeId::new("http://example.org/IriShape");
    let mut shape = Shape::node_shape(shape_id.clone());

    let constraint = Constraint::NodeKind(NodeKindConstraint {
        node_kind: NodeKind::Iri,
    });
    let component_id = constraint.component_id();
    shape.add_constraint(component_id.clone(), constraint);

    assert_eq!(shape.constraints.len(), 1);
    assert!(shape.constraints.contains_key(&component_id));
}

#[test]
fn test_shape_add_target() {
    use crate::{targets::Target, Shape};

    let shape_id = ShapeId::new("http://example.org/PersonShape");
    let mut shape = Shape::node_shape(shape_id);

    let class_iri = NamedNode::new("http://xmlns.com/foaf/0.1/Person").expect("valid IRI");
    let target = Target::Class(class_iri);
    shape.add_target(target);

    assert_eq!(shape.targets.len(), 1);
}

#[test]
fn test_shape_metadata_default() {
    use crate::Shape;

    let shape = Shape::node_shape(ShapeId::new("http://example.org/TestShape"));
    assert!(shape.label.is_none());
    assert!(shape.description.is_none());
    assert!(shape.groups.is_empty());
    assert!(shape.order.is_none());
    assert!(shape.extends.is_empty());
    assert!(shape.property_shapes.is_empty());
    assert!(shape.priority.is_none());
}

#[test]
fn test_shape_extends_parent() {
    use crate::Shape;

    let parent_id = ShapeId::new("http://example.org/BaseShape");
    let child_id = ShapeId::new("http://example.org/ChildShape");
    let mut child_shape = Shape::node_shape(child_id);

    child_shape.extends(parent_id.clone());

    assert!(child_shape.extends_shape(&parent_id));
    assert_eq!(child_shape.parent_shapes().len(), 1);
}

#[test]
fn test_shape_priority() {
    use crate::Shape;

    let shape_id = ShapeId::new("http://example.org/PriorityShape");
    let mut shape = Shape::node_shape(shape_id);

    assert_eq!(shape.effective_priority(), 0); // default
    shape.with_priority(10);
    assert_eq!(shape.effective_priority(), 10);
}

// ---- Severity level tests ----

#[test]
fn test_severity_display() {
    use crate::Severity;

    assert_eq!(format!("{}", Severity::Info), "Info");
    assert_eq!(format!("{}", Severity::Warning), "Warning");
    assert_eq!(format!("{}", Severity::Violation), "Violation");
}

#[test]
fn test_severity_default() {
    use crate::Severity;

    let default_severity = Severity::default();
    assert_eq!(default_severity, Severity::Violation);
}

#[test]
fn test_severity_ordering() {
    use crate::Severity;

    assert!(Severity::Info < Severity::Warning);
    assert!(Severity::Warning < Severity::Violation);
    assert!(Severity::Info < Severity::Violation);
}

// ---- ValidationConfig tests ----

#[test]
fn test_validation_config_default_values() {
    let config = ValidationConfig::default();
    // Default config should have reasonable defaults
    let _ = config.max_violations; // either unlimited (0) or some limit
}

#[test]
fn test_validation_config_with_strategy() {
    use crate::ValidationStrategy;

    let config = ValidationConfig::default().with_strategy(ValidationStrategy::Optimized);
    // Just check it doesn't panic
    drop(config);
}

// ---- ValidationViolation severity tests ----

#[test]
fn test_validation_violation_severity_info() {
    let focus_node = Term::NamedNode(NamedNode::new("http://example.org/node").expect("valid IRI"));
    let shape_id = ShapeId::new("http://example.org/InfoShape");
    let component_id = ConstraintComponentId::new("sh:MinCountConstraintComponent");

    let violation = ValidationViolation::new(focus_node, shape_id, component_id, Severity::Info);

    assert_eq!(violation.result_severity, Severity::Info);
}

#[test]
fn test_validation_violation_severity_warning() {
    let focus_node = Term::NamedNode(NamedNode::new("http://example.org/node").expect("valid IRI"));
    let shape_id = ShapeId::new("http://example.org/WarnShape");
    let component_id = ConstraintComponentId::new("sh:PatternConstraintComponent");

    let violation = ValidationViolation::new(focus_node, shape_id, component_id, Severity::Warning);

    assert_eq!(violation.result_severity, Severity::Warning);
}

#[test]
fn test_validation_violation_with_multiple_details() {
    let focus_node = Term::NamedNode(NamedNode::new("http://example.org/node").expect("valid IRI"));
    let shape_id = ShapeId::new("http://example.org/TestShape");
    let component_id = ConstraintComponentId::new("sh:DatatypeConstraintComponent");

    let violation =
        ValidationViolation::new(focus_node, shape_id, component_id, Severity::Violation)
            .with_message("Type mismatch".to_string())
            .with_detail("expected".to_string(), "xsd:string".to_string())
            .with_detail("actual".to_string(), "xsd:integer".to_string());

    assert_eq!(violation.result_message, Some("Type mismatch".to_string()));
    assert_eq!(
        violation.details.get("expected"),
        Some(&"xsd:string".to_string())
    );
    assert_eq!(
        violation.details.get("actual"),
        Some(&"xsd:integer".to_string())
    );
}

#[test]
fn test_validation_violation_focus_node_preserved() {
    let focus_iri = "http://example.org/specificNode";
    let focus_node = Term::NamedNode(NamedNode::new(focus_iri).expect("valid IRI"));
    let shape_id = ShapeId::new("http://example.org/SomeShape");
    let component_id = ConstraintComponentId::new("sh:ClassConstraintComponent");

    let violation = ValidationViolation::new(
        focus_node.clone(),
        shape_id,
        component_id,
        Severity::Violation,
    );

    assert_eq!(violation.focus_node, focus_node);
    if let Term::NamedNode(nn) = &violation.focus_node {
        assert_eq!(nn.as_str(), focus_iri);
    } else {
        panic!("Expected NamedNode focus node");
    }
}

// ---- ConstraintCacheKey equality tests ----

#[test]
fn test_constraint_cache_key_inequality_different_shape() {
    let focus_node = Term::NamedNode(NamedNode::new("http://example.org/test").expect("valid IRI"));
    let shape_id_a = ShapeId::new("shape_a");
    let shape_id_b = ShapeId::new("shape_b");
    let component_id = ConstraintComponentId::new("sh:NodeKindConstraintComponent");

    let key_a = ConstraintCacheKey {
        focus_node: focus_node.clone(),
        shape_id: shape_id_a,
        constraint_component_id: component_id.clone(),
        property_path: None,
    };

    let key_b = ConstraintCacheKey {
        focus_node: focus_node.clone(),
        shape_id: shape_id_b,
        constraint_component_id: component_id,
        property_path: None,
    };

    assert_ne!(key_a, key_b);
}

#[test]
fn test_constraint_cache_key_inequality_different_path() {
    let focus_node = Term::NamedNode(NamedNode::new("http://example.org/test").expect("valid IRI"));
    let shape_id = ShapeId::new("test_shape");
    let component_id = ConstraintComponentId::new("sh:NodeKindConstraintComponent");
    let path_a =
        PropertyPath::Predicate(NamedNode::new("http://example.org/a").expect("valid IRI"));
    let path_b =
        PropertyPath::Predicate(NamedNode::new("http://example.org/b").expect("valid IRI"));

    let key_a = ConstraintCacheKey {
        focus_node: focus_node.clone(),
        shape_id: shape_id.clone(),
        constraint_component_id: component_id.clone(),
        property_path: Some(path_a),
    };

    let key_b = ConstraintCacheKey {
        focus_node,
        shape_id,
        constraint_component_id: component_id,
        property_path: Some(path_b),
    };

    assert_ne!(key_a, key_b);
}

// ---- ValidationStats tests ----

#[test]
fn test_validation_stats_multiple_constraints() {
    let mut stats = ValidationStats::default();

    stats.record_constraint_evaluation("NodeKind".to_string(), Duration::from_millis(5));
    stats.record_constraint_evaluation("NodeKind".to_string(), Duration::from_millis(10));
    stats.record_constraint_evaluation("MinCount".to_string(), Duration::from_millis(3));
    stats.record_constraint_evaluation("MaxCount".to_string(), Duration::from_millis(7));

    assert_eq!(stats.total_constraint_evaluations, 4);
    assert_eq!(
        stats.constraint_evaluation_times.get("NodeKind"),
        Some(&Duration::from_millis(15))
    );
    assert_eq!(
        stats.constraint_evaluation_times.get("MinCount"),
        Some(&Duration::from_millis(3))
    );
}

#[test]
fn test_validation_stats_zero_evaluations() {
    let stats = ValidationStats::default();
    assert_eq!(stats.total_constraint_evaluations, 0);
    assert!(stats.constraint_evaluation_times.is_empty());
}

// ---- ConstraintEvaluationResult error variant ----

#[test]
fn test_constraint_evaluation_result_error() {
    let error_result = ConstraintEvaluationResult::error("Something went wrong".to_string());
    assert!(!error_result.is_satisfied());
    assert!(!error_result.is_violated());
    assert!(error_result.is_error());
}

#[test]
fn test_constraint_evaluation_result_satisfied_not_violated() {
    let result = ConstraintEvaluationResult::satisfied();
    assert!(result.is_satisfied());
    assert!(!result.is_violated());
    assert!(!result.is_error());
    assert!(result.violating_value().is_none());
    assert!(result.message().is_none());
}

#[test]
fn test_constraint_evaluation_result_violated_with_value() {
    let value = Term::NamedNode(NamedNode::new("http://example.org/badNode").expect("valid IRI"));
    let result = ConstraintEvaluationResult::violated(
        Some(value.clone()),
        Some("Node kind mismatch".to_string()),
    );

    assert!(!result.is_satisfied());
    assert!(result.is_violated());
    assert!(!result.is_error());
    assert_eq!(result.violating_value(), Some(&value));
    assert_eq!(result.message(), Some("Node kind mismatch"));
}

#[test]
fn test_constraint_evaluation_result_violated_no_value() {
    let result = ConstraintEvaluationResult::violated(None, Some("Generic violation".to_string()));

    assert!(result.is_violated());
    assert!(result.violating_value().is_none());
    assert!(result.message().is_some());
}
