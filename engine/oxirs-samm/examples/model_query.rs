//! # Model Query and Analysis Example
//!
//! This example demonstrates the model query API for introspecting SAMM models:
//! 1. Find properties by various criteria
//! 2. Analyze model complexity
//! 3. Build and analyze dependency graphs
//! 4. Detect circular dependencies
//! 5. Group properties by characteristics
//!
//! Run with: `cargo run --example model_query`

use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Entity, ModelElement, Property,
};
use oxirs_samm::query::ModelQuery;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SAMM Model Query and Analysis Example ===\n");

    // Create a complex sample model
    println!("Creating sample model...");
    let aspect = create_complex_aspect();
    println!("✓ Model created: {}", aspect.name());
    println!("  Total properties: {}", aspect.properties().len());
    println!();

    // Create query instance for the aspect
    let query = ModelQuery::new(&aspect);

    // Step 1: Find optional properties
    println!("Step 1: Finding optional properties...");
    let optional_props = query.find_optional_properties();
    println!("  Found {} optional properties:", optional_props.len());
    for prop in &optional_props {
        println!("    - {}", prop.name());
    }
    println!();

    // Step 2: Find required properties
    println!("Step 2: Finding required properties...");
    let required_props = query.find_required_properties();
    println!("  Found {} required properties:", required_props.len());
    for prop in &required_props {
        println!("    - {}", prop.name());
    }
    println!();

    // Step 3: Find collection properties
    println!("Step 3: Finding collection properties...");
    let collection_props = query.find_properties_with_collection_characteristic();
    println!("  Found {} collection properties:", collection_props.len());
    for prop in &collection_props {
        println!("    - {}", prop.name());
    }
    println!();

    // Step 4: Find properties in specific namespace
    println!("Step 4: Finding properties in 'com.example' namespace...");
    let namespace_props = query.find_properties_in_namespace("urn:samm:com.example");
    println!("  Found {} properties in namespace:", namespace_props.len());
    for prop in &namespace_props {
        println!("    - {} ({})", prop.name(), prop.metadata.urn);
    }
    println!();

    // Step 5: Find properties by name pattern
    println!("Step 5: Finding properties with 'date' in name...");
    let pattern_props = query.find_properties_by_name_pattern(r"(?i)date");
    println!("  Found {} matching properties:", pattern_props.len());
    for prop in &pattern_props {
        println!("    - {}", prop.name());
    }
    println!();

    // Step 6: Find properties by custom predicate
    println!("Step 6: Finding measurement properties...");
    let measurement_props = query
        .find_properties_by_characteristic(|c| matches!(c, CharacteristicKind::Measurement { .. }));
    println!(
        "  Found {} measurement properties:",
        measurement_props.len()
    );
    for prop in &measurement_props {
        if let Some(characteristic) = &prop.characteristic {
            if let CharacteristicKind::Measurement { unit } = &characteristic.kind {
                println!("    - {}: unit={}", prop.name(), unit);
            }
        }
    }
    println!();

    // Step 7: Analyze model complexity
    println!("Step 7: Analyzing model complexity...");
    let metrics = query.complexity_metrics();
    println!("  Complexity Metrics:");
    println!("    Total properties: {}", metrics.total_properties);
    println!("    Optional properties: {}", metrics.optional_properties);
    println!(
        "    Collection properties: {}",
        metrics.collection_properties
    );
    println!("    Total operations: {}", metrics.total_operations);
    println!("    Max nesting depth: {}", metrics.max_nesting_depth);
    println!("    Total referenced entities: {}", metrics.total_entities);
    println!();

    // Interpret complexity
    let complexity_level = if metrics.total_properties < 5 {
        "Simple"
    } else if metrics.total_properties < 15 {
        "Moderate"
    } else {
        "Complex"
    };
    println!("  Complexity Level: {}", complexity_level);
    println!();

    // Step 8: Build dependency graph
    println!("Step 8: Building property dependency graph...");
    let dependencies = query.build_dependency_graph();
    println!("  Dependency graph constructed:");
    println!("    Total dependencies: {}", dependencies.len());
    for dep in &dependencies {
        let from_name = dep.from.split('#').next_back().unwrap_or(&dep.from);
        let to_name = dep.to.split('#').next_back().unwrap_or(&dep.to);
        println!("    {} → {} ({})", from_name, to_name, dep.dependency_type);
    }
    println!();

    // Step 9: Detect circular dependencies
    println!("Step 9: Checking for circular dependencies...");
    let circular_deps = query.detect_circular_dependencies();
    if circular_deps.is_empty() {
        println!("  ✓ No circular dependencies detected");
    } else {
        println!("  ⚠️  Circular dependencies found:");
        for cycle in &circular_deps {
            println!("    Cycle: {}", cycle.join(" → "));
        }
    }
    println!();

    // Step 10: Group properties by characteristic type
    println!("Step 10: Grouping properties by characteristic type...");
    let grouped = query.group_properties_by_characteristic_type();
    println!("  Properties grouped by type:");
    for (char_type, props) in &grouped {
        println!("    {:?}: {} properties", char_type, props.len());
        for prop in props {
            println!("      - {}", prop.name());
        }
    }
    println!();

    // Step 11: Find all referenced entities
    println!("Step 11: Finding all referenced entities...");
    let entity_urns = query.find_all_referenced_entities();
    println!("  Found {} referenced entity URNs:", entity_urns.len());
    for entity_urn in &entity_urns {
        let entity_name = entity_urn.split('#').next_back().unwrap_or(entity_urn);
        println!("    - {}", entity_name);
    }
    println!();

    // Use case: Model validation and quality checks
    println!("=== Use Case: Model Quality Report ===");
    println!();
    println!("Quality Checks:");

    // Check 1: Complexity
    if metrics.total_properties > 20 {
        println!(
            "  ⚠️  Model has {} properties - consider splitting",
            metrics.total_properties
        );
    } else {
        println!("  ✓ Property count is reasonable");
    }

    // Check 2: Required vs Optional balance
    let required_count = query.find_required_properties().len();
    let required_ratio = required_count as f64 / metrics.total_properties as f64;
    if required_ratio > 0.8 {
        println!(
            "  ⚠️  {}% of properties are required - consider making some optional",
            (required_ratio * 100.0) as u32
        );
    } else {
        println!("  ✓ Good balance of required/optional properties");
    }

    // Check 3: Circular dependencies
    if !circular_deps.is_empty() {
        println!("  ⚠️  Circular dependencies detected - may cause issues");
    } else {
        println!("  ✓ No circular dependencies");
    }

    // Check 4: Nesting depth
    if metrics.max_nesting_depth > 3 {
        println!(
            "  ⚠️  Deep nesting ({} levels) - consider flattening",
            metrics.max_nesting_depth
        );
    } else {
        println!("  ✓ Reasonable nesting depth");
    }

    println!();

    // Summary
    println!("=== Query API Summary ===");
    println!("Query capabilities demonstrated:");
    println!("  ✓ Property filtering (optional, required, collections)");
    println!("  ✓ Namespace-based queries");
    println!("  ✓ Pattern matching with regex");
    println!("  ✓ Custom predicate filtering");
    println!("  ✓ Complexity analysis");
    println!("  ✓ Dependency graph construction");
    println!("  ✓ Circular dependency detection");
    println!("  ✓ Property grouping");
    println!("  ✓ Entity discovery");
    println!("  ✓ Quality reporting");
    println!();
    println!("The query API enables deep introspection and analysis of SAMM models.");

    Ok(())
}

/// Create a complex aspect for demonstration
fn create_complex_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#ComplexModel".to_string());

    // Create various property types
    let mut id_prop = Property::new("urn:samm:com.example:1.0.0#id".to_string());
    id_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Id".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string()),
    );
    id_prop.example_values = vec!["ID-12345".to_string()];
    id_prop.optional = false;

    let mut name_prop = Property::new("urn:samm:com.example:1.0.0#name".to_string());
    name_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Text".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string()),
    );
    name_prop.example_values = vec!["Example Name".to_string()];
    name_prop.optional = false;

    let mut creation_date_prop =
        Property::new("urn:samm:com.example:1.0.0#creationDate".to_string());
    creation_date_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Timestamp".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:dateTime".to_string()),
    );
    creation_date_prop.example_values = vec!["2024-01-01T00:00:00Z".to_string()];
    creation_date_prop.optional = false;

    let mut modification_date_prop =
        Property::new("urn:samm:com.example:1.0.0#modificationDate".to_string());
    modification_date_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Timestamp".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:dateTime".to_string()),
    );
    modification_date_prop.example_values = vec!["2024-01-15T10:30:00Z".to_string()];
    modification_date_prop.optional = true;

    let mut temperature_prop = Property::new("urn:samm:com.example:1.0.0#temperature".to_string());
    temperature_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Temperature".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:degreeCelsius".to_string(),
            },
        )
        .with_data_type("xsd:float".to_string()),
    );
    temperature_prop.example_values = vec!["23.5".to_string()];
    temperature_prop.optional = true;

    let mut pressure_prop = Property::new("urn:samm:com.example:1.0.0#pressure".to_string());
    pressure_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Pressure".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:pascal".to_string(),
            },
        )
        .with_data_type("xsd:float".to_string()),
    );
    pressure_prop.example_values = vec!["101325.0".to_string()];
    pressure_prop.optional = true;

    let mut tags_prop = Property::new("urn:samm:com.example:1.0.0#tags".to_string());
    tags_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#TagList".to_string(),
            CharacteristicKind::Collection {
                element_characteristic: None,
            },
        )
        .with_data_type("xsd:string".to_string()),
    );
    tags_prop.example_values = vec!["[\"tag1\", \"tag2\"]".to_string()];
    tags_prop.optional = true;

    let mut metadata_prop = Property::new("urn:samm:com.example:1.0.0#metadata".to_string());
    metadata_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#MetadataMap".to_string(),
            CharacteristicKind::Collection {
                element_characteristic: None,
            },
        )
        .with_data_type("xsd:string".to_string()),
    );
    metadata_prop.example_values = vec!["{\"key\": \"value\"}".to_string()];
    metadata_prop.optional = true;

    let mut status_prop = Property::new("urn:samm:com.example:1.0.0#status".to_string());
    status_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#StatusEnum".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["active".to_string(), "inactive".to_string()],
            },
        )
        .with_data_type("xsd:string".to_string()),
    );
    status_prop.example_values = vec!["active".to_string()];
    status_prop.optional = false;

    let mut description_prop = Property::new("urn:samm:com.example:1.0.0#description".to_string());
    description_prop.characteristic = Some(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#Text".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string()),
    );
    description_prop.example_values = vec!["A detailed description".to_string()];
    description_prop.optional = true;

    aspect.properties = vec![
        id_prop,
        name_prop,
        creation_date_prop,
        modification_date_prop,
        temperature_prop,
        pressure_prop,
        tags_prop,
        metadata_prop,
        status_prop,
        description_prop,
    ];

    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Complex Model".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "A complex model for demonstration purposes".to_string(),
    );

    aspect
}
