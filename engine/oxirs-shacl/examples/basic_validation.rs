//! Basic SHACL Validation Example
//!
//! This example demonstrates:
//! - Creating a simple SHACL shape
//! - Adding constraints to shapes
//! - Shape properties and metadata
//!
//! Run with:
//! ```bash
//! cargo run --example basic_validation
//! ```

use oxirs_core::NamedNode;
use oxirs_shacl::{
    constraints::{cardinality_constraints::MinCountConstraint, Constraint},
    ConstraintComponentId, Shape, ShapeId, Target,
};

fn main() {
    println!("🔍 OxiRS SHACL - Basic Validation Example\n");

    // Step 1: Create a SHACL shape
    println!("1️⃣  Creating SHACL shape...");
    let mut person_shape = Shape::node_shape(ShapeId("http://example.org/PersonShape".to_string()));

    // Add target: validate all instances of ex:Person
    person_shape.add_target(Target::class(NamedNode::new_unchecked(
        "http://example.org/Person",
    )));

    // Add constraints
    person_shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    person_shape.label = Some("Person Shape".to_string());
    person_shape.description = Some("Validates that persons have required properties".to_string());

    println!("   ✅ Shape created: {}", person_shape.id.0);
    println!("   📋 Constraints: {}", person_shape.constraints.len());
    println!("   🎯 Targets: {}", person_shape.targets.len());
    println!();

    // Step 2: Display shape properties
    println!("2️⃣  Shape Properties:");
    println!(
        "   Label: {}",
        person_shape
            .label
            .as_ref()
            .expect("invariant: value is valid")
    );
    println!(
        "   Description: {}",
        person_shape
            .description
            .as_ref()
            .expect("invariant: value is valid")
    );
    println!("   Severity: {:?}", person_shape.severity);
    println!("   Active: {}", person_shape.is_active());
    println!();

    // Step 3: Display constraint details
    println!("3️⃣  Constraint Details:");
    for (component_id, constraint) in &person_shape.constraints {
        println!("   Component: {}", component_id.0);
        println!("   Type: {:?}", constraint);
    }
    println!();

    println!("✨ Example completed successfully!");
    println!("\n💡 Next steps:");
    println!("   - See examples/advanced_shapes.rs for complex constraints");
    println!("   - See examples/parallel_validation.rs for performance");
    println!("   - Check the LSP server: cargo run --bin shacl_lsp --features lsp");
}
