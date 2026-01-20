//! # Model Transformation Example
//!
//! This example demonstrates the model transformation API for refactoring SAMM models:
//! 1. Create a sample model
//! 2. Apply various transformations (rename, namespace change, optional/required)
//! 3. Chain multiple transformations
//! 4. Track changes and verify results
//!
//! Run with: `cargo run --example model_transformation`

use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, ModelElement, Property};
use oxirs_samm::transformation::{ModelTransformation, TransformationRule};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SAMM Model Transformation Example ===\n");

    // Step 1: Create original model
    println!("Step 1: Creating original model...");
    let aspect = create_original_aspect();
    println!("✓ Original aspect: {}", aspect.name());
    println!("  URN: {}", aspect.metadata.urn);
    println!("  Properties: {}", aspect.properties().len());
    for prop in aspect.properties() {
        println!("    - {} (optional: {})", prop.name(), prop.optional);
    }
    println!();

    // Step 2: Rename a property
    println!("Step 2: Renaming property 'serialNumber' to 'productSerial'...");
    let mut aspect_step2 = aspect;
    let mut transformation = ModelTransformation::new(&mut aspect_step2);
    transformation.rename_property("serialNumber", "productSerial");
    let result = transformation.apply();

    println!("✓ Transformation successful");
    println!(
        "  Transformations applied: {}",
        result.transformations_applied
    );
    println!("  Successful: {}", result.successful_transformations.len());
    for change in &result.successful_transformations {
        println!("    - {}", change);
    }

    let renamed_prop = aspect_step2
        .properties()
        .iter()
        .find(|p| p.name() == "productSerial");
    assert!(renamed_prop.is_some(), "Property rename failed");
    println!("  ✓ Property successfully renamed");
    println!();

    // Step 3: Change namespace
    println!("Step 3: Changing namespace from 'com.example' to 'org.eclipse.esmf'...");
    let mut aspect_step3 = aspect_step2;
    let mut transformation = ModelTransformation::new(&mut aspect_step3);
    transformation.change_namespace(
        "urn:samm:com.example:1.0.0",
        "urn:samm:org.eclipse.esmf:1.0.0",
    );
    let result = transformation.apply();

    println!("✓ Namespace changed");
    println!("  New aspect URN: {}", aspect_step3.metadata.urn);
    println!(
        "  Transformations applied: {}",
        result.transformations_applied
    );
    println!();

    // Step 4: Make all properties optional
    println!("Step 4: Making all properties optional...");
    let mut aspect_step4 = aspect_step3;
    let mut transformation = ModelTransformation::new(&mut aspect_step4);
    transformation.make_all_properties_optional();
    let result = transformation.apply();

    println!("✓ All properties are now optional");
    for prop in aspect_step4.properties() {
        println!("    - {}: optional={}", prop.name(), prop.optional);
        assert!(prop.optional, "Property should be optional");
    }
    println!();

    // Step 5: Chained transformations
    println!("Step 5: Demonstrating chained transformations...");
    let mut aspect_step5 = create_original_aspect();
    let mut transformation = ModelTransformation::new(&mut aspect_step5);
    transformation.rename_property("productName", "title");
    transformation.make_property_optional("title");
    transformation.update_preferred_name("en", "Enhanced Product");
    transformation.update_description("en", "An enhanced product model with improved metadata");
    let result = transformation.apply();

    println!("✓ Applied 4 chained transformations");
    println!(
        "  Total transformations: {}",
        result.transformations_applied
    );
    println!("  Successful transformations:");
    for change in &result.successful_transformations {
        println!("    - {}", change);
    }
    println!();

    // Step 6: String-based URN replacement
    println!("Step 6: Replacing version in all URNs (1.0.0 → 2.0.0)...");
    let mut aspect_step6 = aspect_step5;
    let mut transformation = ModelTransformation::new(&mut aspect_step6);
    transformation.replace_urn_pattern("1.0.0", "2.0.0");
    let result = transformation.apply();

    println!("✓ Version updated in all URNs");
    println!("  New aspect URN: {}", aspect_step6.metadata.urn);
    for prop in aspect_step6.properties() {
        println!("    - Property URN: {}", prop.metadata.urn);
        assert!(prop.metadata.urn.contains("2.0.0"), "Version not updated");
    }
    println!();

    // Step 7: Complex transformation workflow
    println!("Step 7: Complex workflow - Migrating to new namespace and version...");
    let mut aspect_step7 = create_original_aspect();
    let mut transformation = ModelTransformation::new(&mut aspect_step7);
    // Step 1: Update namespace
    transformation.change_namespace("urn:samm:com.example:1.0.0", "urn:samm:org.mycompany:1.0.0");
    // Step 2: Update version
    transformation.replace_urn_pattern("1.0.0", "2.0.0");
    // Step 3: Make some properties optional for backward compatibility
    transformation.make_property_optional("serialNumber");
    transformation.make_property_optional("category");
    // Step 4: Update metadata
    transformation.update_preferred_name("en", "Product v2");
    transformation.update_description("en", "Product model version 2.0 with enhanced flexibility");
    let result = transformation.apply();

    println!("✓ Migration complete");
    println!(
        "  Total transformations: {}",
        result.transformations_applied
    );
    println!("  Final aspect URN: {}", aspect_step7.metadata.urn);
    println!("  Optional properties:");
    for prop in aspect_step7.properties() {
        if prop.optional {
            println!("    - {}", prop.name());
        }
    }
    println!();

    // Summary
    println!("=== Transformation Summary ===");
    println!("Demonstrated transformations:");
    println!("  ✓ Property renaming");
    println!("  ✓ Namespace changes");
    println!("  ✓ Optional/required modifications");
    println!("  ✓ Metadata updates");
    println!("  ✓ String-based URN replacement");
    println!("  ✓ Chained transformations");
    println!("  ✓ Complex migration workflows");
    println!();
    println!("The transformation API provides a fluent interface for refactoring SAMM models");
    println!("with full change tracking and validation.");

    Ok(())
}

/// Create a sample aspect for transformation demonstrations
fn create_original_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#Product".to_string());

    let mut product_name_prop = Property::new("urn:samm:com.example:1.0.0#productName".to_string());
    product_name_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Text".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string()),
    );
    product_name_prop.example_values = vec!["Widget".to_string()];
    product_name_prop.optional = false;

    let mut serial_number_prop =
        Property::new("urn:samm:com.example:1.0.0#serialNumber".to_string());
    serial_number_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#SerialNumber".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string()),
    );
    serial_number_prop.example_values = vec!["SN-12345".to_string()];
    serial_number_prop.optional = false;

    let mut price_prop = Property::new("urn:samm:com.example:1.0.0#price".to_string());
    price_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Price".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:euro".to_string(),
            },
        )
        .with_data_type("xsd:decimal".to_string()),
    );
    price_prop.example_values = vec!["99.99".to_string()];
    price_prop.optional = false;

    let mut category_prop = Property::new("urn:samm:com.example:1.0.0#category".to_string());
    category_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Category".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string()),
    );
    category_prop.example_values = vec!["Electronics".to_string()];
    category_prop.optional = false;

    aspect.properties = vec![
        product_name_prop,
        serial_number_prop,
        price_prop,
        category_prop,
    ];

    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Product".to_string());
    aspect
        .metadata
        .add_description("en".to_string(), "A product in the catalog".to_string());

    aspect
}
