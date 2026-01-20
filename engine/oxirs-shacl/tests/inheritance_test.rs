//! Tests for SHACL shape inheritance functionality

use oxirs_core::{model::*, ConcreteStore};
use oxirs_shacl::*;

#[test]
fn test_basic_shape_inheritance() {
    // Create a simple inheritance test case
    let mut validator = Validator::new();

    // Create a parent shape with constraints
    let mut parent_shape = Shape::node_shape(ShapeId::new("http://example.org/ParentShape"));
    parent_shape.add_constraint(
        ConstraintComponentId::new("minCount"),
        Constraint::MinCount(constraints::MinCountConstraint { min_count: 1 }),
    );
    parent_shape.add_constraint(
        ConstraintComponentId::new("datatype"),
        Constraint::Datatype(constraints::DatatypeConstraint {
            datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
        }),
    );

    // Create a child shape that inherits from parent
    let mut child_shape = Shape::node_shape(ShapeId::new("http://example.org/ChildShape"));
    child_shape.extends(ShapeId::new("http://example.org/ParentShape"));
    child_shape.add_constraint(
        ConstraintComponentId::new("maxCount"),
        Constraint::MaxCount(constraints::MaxCountConstraint { max_count: 5 }),
    );

    // Add shapes to validator
    validator.add_shape(parent_shape).unwrap();
    validator.add_shape(child_shape).unwrap();

    // Verify inheritance resolution works
    let mut engine =
        validation::ValidationEngine::new(validator.shapes(), ValidationConfig::default());
    let resolved_constraints = engine
        .resolve_inherited_constraints(&ShapeId::new("http://example.org/ChildShape"))
        .unwrap();

    // Child should have all parent constraints plus its own
    assert_eq!(resolved_constraints.len(), 3);
    assert!(resolved_constraints.contains_key(&ConstraintComponentId::new("minCount")));
    assert!(resolved_constraints.contains_key(&ConstraintComponentId::new("datatype")));
    assert!(resolved_constraints.contains_key(&ConstraintComponentId::new("maxCount")));
}

#[test]
fn test_constraint_override_in_inheritance() {
    // Test that child constraints override parent constraints
    let mut validator = Validator::new();

    // Parent shape with minCount = 1
    let mut parent_shape = Shape::node_shape(ShapeId::new("http://example.org/ParentShape"));
    parent_shape.add_constraint(
        ConstraintComponentId::new("minCount"),
        Constraint::MinCount(constraints::MinCountConstraint { min_count: 1 }),
    );

    // Child shape overrides minCount to 2
    let mut child_shape = Shape::node_shape(ShapeId::new("http://example.org/ChildShape"));
    child_shape.extends(ShapeId::new("http://example.org/ParentShape"));
    child_shape.add_constraint(
        ConstraintComponentId::new("minCount"),
        Constraint::MinCount(constraints::MinCountConstraint { min_count: 2 }),
    );

    validator.add_shape(parent_shape).unwrap();
    validator.add_shape(child_shape).unwrap();

    let mut engine =
        validation::ValidationEngine::new(validator.shapes(), ValidationConfig::default());
    let resolved_constraints = engine
        .resolve_inherited_constraints(&ShapeId::new("http://example.org/ChildShape"))
        .unwrap();

    // Should have only one minCount constraint with the child's value
    assert_eq!(resolved_constraints.len(), 1);
    if let Some(Constraint::MinCount(min_count)) =
        resolved_constraints.get(&ConstraintComponentId::new("minCount"))
    {
        assert_eq!(min_count.min_count, 2); // Child's value should override parent's
    } else {
        panic!("Expected MinCount constraint");
    }
}

#[test]
fn test_priority_based_inheritance() {
    // Test priority-based conflict resolution
    let mut validator = Validator::new();

    // High priority parent
    let mut high_priority_parent =
        Shape::node_shape(ShapeId::new("http://example.org/HighPriorityParent"));
    high_priority_parent.with_priority(10);
    high_priority_parent.add_constraint(
        ConstraintComponentId::new("datatype"),
        Constraint::Datatype(constraints::DatatypeConstraint {
            datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
        }),
    );

    // Low priority parent
    let mut low_priority_parent =
        Shape::node_shape(ShapeId::new("http://example.org/LowPriorityParent"));
    low_priority_parent.with_priority(1);
    low_priority_parent.add_constraint(
        ConstraintComponentId::new("datatype"),
        Constraint::Datatype(constraints::DatatypeConstraint {
            datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        }),
    );

    // Child inherits from both (multiple inheritance)
    let mut child_shape = Shape::node_shape(ShapeId::new("http://example.org/ChildShape"));
    child_shape.extends(ShapeId::new("http://example.org/LowPriorityParent"));
    child_shape.extends(ShapeId::new("http://example.org/HighPriorityParent"));

    validator.add_shape(high_priority_parent).unwrap();
    validator.add_shape(low_priority_parent).unwrap();
    validator.add_shape(child_shape).unwrap();

    let mut engine =
        validation::ValidationEngine::new(validator.shapes(), ValidationConfig::default());
    let resolved_constraints = engine
        .resolve_inherited_constraints(&ShapeId::new("http://example.org/ChildShape"))
        .unwrap();

    // Should have the high priority parent's constraint
    assert_eq!(resolved_constraints.len(), 1);
    if let Some(Constraint::Datatype(datatype)) =
        resolved_constraints.get(&ConstraintComponentId::new("datatype"))
    {
        assert_eq!(
            datatype.datatype_iri.as_str(),
            "http://www.w3.org/2001/XMLSchema#string"
        );
    } else {
        panic!("Expected Datatype constraint");
    }
}

#[test]
fn test_circular_inheritance_prevention() {
    // Test that circular inheritance is prevented
    let mut validator = Validator::new();

    // Create shapes that would form a circle: A -> B -> A
    let mut shape_a = Shape::node_shape(ShapeId::new("http://example.org/ShapeA"));
    shape_a.extends(ShapeId::new("http://example.org/ShapeB"));

    let mut shape_b = Shape::node_shape(ShapeId::new("http://example.org/ShapeB"));
    shape_b.extends(ShapeId::new("http://example.org/ShapeA"));

    // Adding these shapes should either detect the circular dependency or handle it gracefully
    assert!(validator.add_shape(shape_a).is_ok());
    let result = validator.add_shape(shape_b);

    // Should either succeed (with circular prevention) or fail with an appropriate error
    match result {
        Ok(_) => {
            // If it succeeds, inheritance resolution should handle circular references
            let mut engine =
                validation::ValidationEngine::new(validator.shapes(), ValidationConfig::default());
            let resolved =
                engine.resolve_inherited_constraints(&ShapeId::new("http://example.org/ShapeA"));
            assert!(resolved.is_ok());
        }
        Err(e) => {
            // Should be a circular dependency error
            assert!(e.to_string().contains("circular") || e.to_string().contains("cycle"));
        }
    }
}

#[test]
fn test_deep_inheritance_chain() {
    // Test a deep inheritance chain: GrandParent -> Parent -> Child
    let mut validator = Validator::new();

    // Grandparent shape
    let mut grandparent = Shape::node_shape(ShapeId::new("http://example.org/GrandParent"));
    grandparent.add_constraint(
        ConstraintComponentId::new("minLength"),
        Constraint::MinLength(constraints::MinLengthConstraint { min_length: 1 }),
    );

    // Parent shape
    let mut parent = Shape::node_shape(ShapeId::new("http://example.org/Parent"));
    parent.extends(ShapeId::new("http://example.org/GrandParent"));
    parent.add_constraint(
        ConstraintComponentId::new("maxLength"),
        Constraint::MaxLength(constraints::MaxLengthConstraint { max_length: 100 }),
    );

    // Child shape
    let mut child = Shape::node_shape(ShapeId::new("http://example.org/Child"));
    child.extends(ShapeId::new("http://example.org/Parent"));
    child.add_constraint(
        ConstraintComponentId::new("pattern"),
        Constraint::Pattern(constraints::PatternConstraint {
            pattern: "[a-zA-Z]+".to_string(),
            flags: None,
            message: None,
        }),
    );

    validator.add_shape(grandparent).unwrap();
    validator.add_shape(parent).unwrap();
    validator.add_shape(child).unwrap();

    let mut engine =
        validation::ValidationEngine::new(validator.shapes(), ValidationConfig::default());
    let resolved_constraints = engine
        .resolve_inherited_constraints(&ShapeId::new("http://example.org/Child"))
        .unwrap();

    // Child should have all constraints from the inheritance chain
    assert_eq!(resolved_constraints.len(), 3);
    assert!(resolved_constraints.contains_key(&ConstraintComponentId::new("minLength")));
    assert!(resolved_constraints.contains_key(&ConstraintComponentId::new("maxLength")));
    assert!(resolved_constraints.contains_key(&ConstraintComponentId::new("pattern")));
}

#[test]
fn test_inheritance_with_validation() {
    // Test that inheritance works during actual validation
    let mut validator = Validator::new();
    let store = ConcreteStore::new().unwrap();

    // Add test data
    let subject = NamedNode::new("http://example.org/testNode").unwrap();
    let predicate = NamedNode::new("http://example.org/name").unwrap();
    let object = Literal::new_simple_literal("test");
    store
        .insert_quad(Quad::new(
            subject.clone(),
            predicate.clone(),
            object,
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Parent shape with minLength constraint
    let mut parent_shape = Shape::property_shape(
        ShapeId::new("http://example.org/ParentShape"),
        PropertyPath::predicate(predicate.clone()),
    );
    parent_shape.add_target(Target::node(Term::NamedNode(subject.clone())));
    parent_shape.add_constraint(
        ConstraintComponentId::new("minLength"),
        Constraint::MinLength(constraints::MinLengthConstraint { min_length: 1 }),
    );

    // Child shape inheriting from parent, adds maxLength
    let mut child_shape = Shape::property_shape(
        ShapeId::new("http://example.org/ChildShape"),
        PropertyPath::predicate(predicate.clone()),
    );
    child_shape.extends(ShapeId::new("http://example.org/ParentShape"));
    child_shape.add_target(Target::node(Term::NamedNode(subject.clone())));
    child_shape.add_constraint(
        ConstraintComponentId::new("maxLength"),
        Constraint::MaxLength(constraints::MaxLengthConstraint { max_length: 10 }),
    );

    validator.add_shape(parent_shape).unwrap();
    validator.add_shape(child_shape).unwrap();

    // Validate - should apply both inherited and own constraints
    let report = validator.validate_store(&store, None).unwrap();

    // Should pass validation (string "test" is between 1 and 10 characters)
    assert!(report.conforms());
    assert_eq!(report.violation_count(), 0);
}
