//! Advanced SHACL Shapes Example
//!
//! This example demonstrates:
//! - Complex constraint combinations
//! - Property paths
//! - Multiple shape types
//!
//! Run with:
//! ```bash
//! cargo run --example advanced_shapes
//! ```

use oxirs_core::NamedNode;
use oxirs_shacl::{
    constraints::{
        cardinality_constraints::{MaxCountConstraint, MinCountConstraint},
        string_constraints::{MaxLengthConstraint, MinLengthConstraint, PatternConstraint},
        value_constraints::DatatypeConstraint,
        Constraint,
    },
    paths::PropertyPath,
    ConstraintComponentId, Shape, ShapeId, Target,
};

fn main() {
    println!("🚀 OxiRS SHACL - Advanced Shapes Example\n");

    // Example 1: Shape with multiple constraints
    println!("📌 Example 1: Person Shape with Multiple Constraints");
    let person_shape = create_person_shape();
    println!("   Shape ID: {}", person_shape.id.0);
    println!("   Constraints: {}", person_shape.constraints.len());
    println!("   Severity: {:?}\n", person_shape.severity);

    // Example 2: Shape with property paths
    println!("📌 Example 2: Address Shape with Property Paths");
    let address_shape = create_address_shape();
    println!("   Shape ID: {}", address_shape.id.0);
    if address_shape.path.is_some() {
        println!("   Path: complex sequence path");
    }
    println!();

    // Example 3: Organization shape
    println!("📌 Example 3: Organization Shape with Cardinality");
    let org_shape = create_organization_shape();
    println!("   Shape ID: {}", org_shape.id.0);
    println!("   Cardinality constraints: 2");
    println!();

    println!("✨ All advanced shape examples created successfully!");
}

fn create_person_shape() -> Shape {
    let mut shape = Shape::node_shape(ShapeId("http://example.org/PersonShape".to_string()));

    shape.add_target(Target::class(NamedNode::new_unchecked(
        "http://example.org/Person",
    )));

    // String length constraints
    shape.add_constraint(
        ConstraintComponentId::new("sh:minLength"),
        Constraint::MinLength(MinLengthConstraint { min_length: 1 }),
    );
    shape.add_constraint(
        ConstraintComponentId::new("sh:maxLength"),
        Constraint::MaxLength(MaxLengthConstraint { max_length: 100 }),
    );

    // Datatype constraint
    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string"),
        }),
    );

    // Pattern constraint (regex)
    shape.add_constraint(
        ConstraintComponentId::new("sh:pattern"),
        Constraint::Pattern(PatternConstraint {
            pattern: "^[A-Z][a-z]+$".to_string(),
            flags: None,
            message: None,
        }),
    );

    shape.label = Some("Person Shape".to_string());
    shape.description =
        Some("Comprehensive person validation with multiple constraints".to_string());

    shape
}

fn create_address_shape() -> Shape {
    let mut shape = Shape::node_shape(ShapeId("http://example.org/AddressShape".to_string()));

    // Property path: person -> address -> street
    shape.path = Some(PropertyPath::Sequence(vec![
        PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/hasAddress")),
        PropertyPath::Predicate(NamedNode::new_unchecked("http://example.org/street")),
    ]));

    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#string"),
        }),
    );

    shape.label = Some("Address Shape".to_string());

    shape
}

fn create_organization_shape() -> Shape {
    let mut shape = Shape::node_shape(ShapeId("http://example.org/OrganizationShape".to_string()));

    shape.add_target(Target::class(NamedNode::new_unchecked(
        "http://example.org/Organization",
    )));

    // Cardinality constraints
    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );
    shape.add_constraint(
        ConstraintComponentId::new("sh:maxCount"),
        Constraint::MaxCount(MaxCountConstraint { max_count: 10 }),
    );

    shape.label = Some("Organization Shape".to_string());
    shape.description = Some("Organization with employee count constraints".to_string());

    shape
}
