//! Debug test for qualified cardinality

use oxirs_core::{model::*, Store};
use oxirs_shacl::*;

#[test]
fn debug_qualified_cardinality_min_violation() {
    // Debug version of qualified min count violation test
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
    // Charlie is NOT a Friend - no rdf:type Friend triple

    // Friend shape - requires rdf:type Friend
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
            shape: ShapeId::new("http://example.org/FriendShape"),
            qualified_min_count: Some(1),
            qualified_max_count: None,
            qualified_value_shapes_disjoint: false,
        }),
    );

    validator.add_shape(friend_shape).unwrap();
    validator.add_shape(person_shape).unwrap();

    println!("=== Debug Info ===");
    println!("Shapes in validator: {}", validator.shapes().len());
    for (id, shape) in validator.shapes() {
        println!("Shape: {} (type: {:?})", id.as_str(), shape.shape_type);
        println!("  Targets: {}", shape.targets.len());
        for target in &shape.targets {
            println!("    Target: {:?}", target);
        }
        println!("  Constraints: {}", shape.constraints.len());
        for (cid, constraint) in &shape.constraints {
            println!("    Constraint: {} -> {:?}", cid.as_str(), constraint);
        }
    }

    // Validate
    let report = validator.validate_store(&store, None).unwrap();

    println!("=== Validation Report ===");
    println!("Conforms: {}", report.conforms());
    println!("Violations: {}", report.violation_count());

    for (i, violation) in report.violations.iter().enumerate() {
        println!("Violation {}: {:?}", i, violation);
    }

    // This should fail because Alice knows 0 Friends but needs at least 1
    // But let's see what actually happens
    if report.conforms() {
        println!("WARNING: Report says conforms=true but we expected false!");
        println!(
            "Alice knows Charlie who is NOT a Friend, so qualified min count 1 should be violated"
        );
    } else {
        println!("SUCCESS: Report correctly identifies violation");
    }
}
