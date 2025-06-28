//! Tests for SHACL qualified cardinality constraints

use oxirs_core::{model::*, Store};
use oxirs_shacl::*;

#[test]
fn test_qualified_cardinality_validation() {
    // Test complete qualified cardinality validation
    let mut validator = Validator::new();
    let mut store = Store::new().unwrap();

    // Add test data - some people with different types
    let alice = NamedNode::new("http://example.org/alice").unwrap();
    let bob = NamedNode::new("http://example.org/bob").unwrap();
    let charlie = NamedNode::new("http://example.org/charlie").unwrap();

    let knows_pred = NamedNode::new("http://example.org/knows").unwrap();
    let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    let person_type = NamedNode::new("http://example.org/Person").unwrap();
    let friend_type = NamedNode::new("http://example.org/Friend").unwrap();

    // Alice knows Bob (who is a Friend) and Charlie (who is just a Person)
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            bob.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            charlie.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Type information
    store
        .insert_quad(Quad::new(
            alice.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            bob.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            bob.clone(),
            type_pred.clone(),
            friend_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            charlie.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Create a shape that validates Friends
    let mut friend_shape = Shape::node_shape(ShapeId::new("http://example.org/FriendShape"));
    friend_shape.add_constraint(
        ConstraintComponentId::new("class"),
        Constraint::Class(constraints::ClassConstraint {
            class_iri: friend_type.clone(),
        }),
    );

    // Create a shape that requires at least 1 but at most 1 Friend among the known people
    let mut person_shape = Shape::property_shape(
        ShapeId::new("http://example.org/PersonShape"),
        PropertyPath::predicate(knows_pred.clone()),
    );
    person_shape.add_target(Target::class(person_type.clone()));
    person_shape.add_constraint(
        ConstraintComponentId::new("qualifiedValueShape"),
        Constraint::QualifiedValueShape(constraints::QualifiedValueShapeConstraint {
            qualified_value_shape: ShapeId::new("http://example.org/FriendShape"),
            qualified_min_count: Some(1),
            qualified_max_count: Some(1),
            qualified_value_shapes_disjoint: false,
        }),
    );

    validator.add_shape(friend_shape).unwrap();
    validator.add_shape(person_shape).unwrap();

    // Validate - Alice should conform (knows exactly 1 Friend)
    let report = validator.validate_store(&store, None).unwrap();

    // Should pass validation because Alice knows exactly 1 Friend (Bob)
    assert!(
        report.conforms(),
        "Alice should conform to qualified cardinality constraint"
    );
    assert_eq!(report.violation_count(), 0);
}

#[test]
fn test_qualified_cardinality_min_violation() {
    // Test qualified min count violation
    let mut validator = Validator::new();
    let mut store = Store::new().unwrap();

    let alice = NamedNode::new("http://example.org/alice").unwrap();
    let charlie = NamedNode::new("http://example.org/charlie").unwrap();

    let knows_pred = NamedNode::new("http://example.org/knows").unwrap();
    let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    let person_type = NamedNode::new("http://example.org/Person").unwrap();
    let friend_type = NamedNode::new("http://example.org/Friend").unwrap();

    // Alice knows Charlie (who is NOT a Friend)
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            charlie.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            alice.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            charlie.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    // Charlie is NOT a Friend

    // Friend shape
    let mut friend_shape = Shape::node_shape(ShapeId::new("http://example.org/FriendShape"));
    friend_shape.add_constraint(
        ConstraintComponentId::new("class"),
        Constraint::Class(constraints::ClassConstraint {
            class_iri: friend_type.clone(),
        }),
    );

    // Person shape requiring at least 1 Friend
    let mut person_shape = Shape::property_shape(
        ShapeId::new("http://example.org/PersonShape"),
        PropertyPath::predicate(knows_pred.clone()),
    );
    person_shape.add_target(Target::class(person_type.clone()));
    person_shape.add_constraint(
        ConstraintComponentId::new("qualifiedValueShape"),
        Constraint::QualifiedValueShape(constraints::QualifiedValueShapeConstraint {
            qualified_value_shape: ShapeId::new("http://example.org/FriendShape"),
            qualified_min_count: Some(1),
            qualified_max_count: None,
            qualified_value_shapes_disjoint: false,
        }),
    );

    validator.add_shape(friend_shape).unwrap();
    validator.add_shape(person_shape).unwrap();

    // Validate - should fail because Alice knows 0 Friends
    let report = validator.validate_store(&store, None).unwrap();

    assert!(
        !report.conforms(),
        "Alice should NOT conform - knows 0 Friends but needs at least 1"
    );
    assert!(report.violation_count() > 0);
}

#[test]
fn test_qualified_cardinality_max_violation() {
    // Test qualified max count violation
    let mut validator = Validator::new();
    let mut store = Store::new().unwrap();

    let alice = NamedNode::new("http://example.org/alice").unwrap();
    let bob = NamedNode::new("http://example.org/bob").unwrap();
    let david = NamedNode::new("http://example.org/david").unwrap();

    let knows_pred = NamedNode::new("http://example.org/knows").unwrap();
    let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    let person_type = NamedNode::new("http://example.org/Person").unwrap();
    let friend_type = NamedNode::new("http://example.org/Friend").unwrap();

    // Alice knows Bob and David (both are Friends)
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            bob.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            david.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Type information
    store
        .insert_quad(Quad::new(
            alice.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            bob.clone(),
            type_pred.clone(),
            friend_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            david.clone(),
            type_pred.clone(),
            friend_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Friend shape
    let mut friend_shape = Shape::node_shape(ShapeId::new("http://example.org/FriendShape"));
    friend_shape.add_constraint(
        ConstraintComponentId::new("class"),
        Constraint::Class(constraints::ClassConstraint {
            class_iri: friend_type.clone(),
        }),
    );

    // Person shape allowing at most 1 Friend
    let mut person_shape = Shape::property_shape(
        ShapeId::new("http://example.org/PersonShape"),
        PropertyPath::predicate(knows_pred.clone()),
    );
    person_shape.add_target(Target::class(person_type.clone()));
    person_shape.add_constraint(
        ConstraintComponentId::new("qualifiedValueShape"),
        Constraint::QualifiedValueShape(constraints::QualifiedValueShapeConstraint {
            qualified_value_shape: ShapeId::new("http://example.org/FriendShape"),
            qualified_min_count: None,
            qualified_max_count: Some(1),
            qualified_value_shapes_disjoint: false,
        }),
    );

    validator.add_shape(friend_shape).unwrap();
    validator.add_shape(person_shape).unwrap();

    // Validate - should fail because Alice knows 2 Friends but max is 1
    let report = validator.validate_store(&store, None).unwrap();

    assert!(
        !report.conforms(),
        "Alice should NOT conform - knows 2 Friends but max allowed is 1"
    );
    assert!(report.violation_count() > 0);
}

#[test]
fn test_qualified_cardinality_range_success() {
    // Test qualified cardinality range validation (min and max both specified)
    let mut validator = Validator::new();
    let mut store = Store::new().unwrap();

    let alice = NamedNode::new("http://example.org/alice").unwrap();
    let bob = NamedNode::new("http://example.org/bob").unwrap();
    let charlie = NamedNode::new("http://example.org/charlie").unwrap();
    let david = NamedNode::new("http://example.org/david").unwrap();

    let knows_pred = NamedNode::new("http://example.org/knows").unwrap();
    let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    let person_type = NamedNode::new("http://example.org/Person").unwrap();
    let friend_type = NamedNode::new("http://example.org/Friend").unwrap();

    // Alice knows Bob and David (both Friends) and Charlie (not a Friend)
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            bob.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            david.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            alice.clone(),
            knows_pred.clone(),
            charlie.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Type information
    store
        .insert_quad(Quad::new(
            alice.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            bob.clone(),
            type_pred.clone(),
            friend_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            david.clone(),
            type_pred.clone(),
            friend_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert_quad(Quad::new(
            charlie.clone(),
            type_pred.clone(),
            person_type.clone(),
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Friend shape
    let mut friend_shape = Shape::node_shape(ShapeId::new("http://example.org/FriendShape"));
    friend_shape.add_constraint(
        ConstraintComponentId::new("class"),
        Constraint::Class(constraints::ClassConstraint {
            class_iri: friend_type.clone(),
        }),
    );

    // Person shape requiring 1-3 Friends
    let mut person_shape = Shape::property_shape(
        ShapeId::new("http://example.org/PersonShape"),
        PropertyPath::predicate(knows_pred.clone()),
    );
    person_shape.add_target(Target::class(person_type.clone()));
    person_shape.add_constraint(
        ConstraintComponentId::new("qualifiedValueShape"),
        Constraint::QualifiedValueShape(constraints::QualifiedValueShapeConstraint {
            qualified_value_shape: ShapeId::new("http://example.org/FriendShape"),
            qualified_min_count: Some(1),
            qualified_max_count: Some(3),
            qualified_value_shapes_disjoint: false,
        }),
    );

    validator.add_shape(friend_shape).unwrap();
    validator.add_shape(person_shape).unwrap();

    // Validate - should pass because Alice knows 2 Friends (within range 1-3)
    let report = validator.validate_store(&store, None).unwrap();

    assert!(
        report.conforms(),
        "Alice should conform - knows 2 Friends which is within range 1-3"
    );
    assert_eq!(report.violation_count(), 0);
}
