//! # Code Generation Pipeline Example
//!
//! This example demonstrates a complete code generation workflow:
//! 1. Parse a SAMM model from TTL
//! 2. Validate the model
//! 3. Generate code in multiple languages
//! 4. Write files to disk with proper organization
//!
//! Run with: `cargo run --example code_generation_pipeline --all-features`

use oxirs_samm::generators::{
    generate_graphql, generate_java, generate_python, generate_scala, generate_sql,
    generate_typescript, GeneratedFile, JavaOptions, MultiFileGenerator, MultiFileOptions,
    OutputLayout, PythonOptions, ScalaOptions, SqlDialect, TsOptions,
};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Entity, ModelElement, Property,
};
use oxirs_samm::parser::ModelResolver;
use oxirs_samm::validator::validate_aspect;
use std::fs;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SAMM Code Generation Pipeline Example ===\n");

    // Step 1: Create a sample SAMM model
    println!("Step 1: Creating sample SAMM model...");
    let aspect = create_sample_aspect();
    println!("✓ Created aspect: {}", aspect.name());
    println!("  Properties: {}", aspect.properties().len());
    println!("  Operations: {}", aspect.operations().len());
    println!();

    // Step 2: Validate the model
    println!("Step 2: Validating model...");
    let resolver = ModelResolver::new();
    let validation_result = validate_aspect(&aspect).await?;

    if validation_result.is_valid {
        println!("✓ Model is valid");
    } else {
        println!("✗ Model validation failed:");
        for error in &validation_result.errors {
            println!("  - {}", error.message);
            if let Some(ref urn) = error.element_urn {
                println!("    Element: {}", urn);
            }
        }
        return Ok(());
    }
    println!();

    // Step 3: Generate code in multiple languages
    println!("Step 3: Generating code in multiple languages...");

    // Create output directory
    let output_dir = PathBuf::from("target/generated");
    fs::create_dir_all(&output_dir)?;

    // Generate TypeScript
    println!("  - Generating TypeScript...");
    let ts_options = TsOptions {
        export_default: true,
        readonly_properties: true,
        strict_null_checks: true,
        ..Default::default()
    };
    let ts_code = generate_typescript(&aspect, ts_options)?;
    fs::write(output_dir.join("Vehicle.ts"), ts_code)?;
    println!("    ✓ Written to target/generated/Vehicle.ts");

    // Generate Python
    println!("  - Generating Python...");
    let py_options = PythonOptions {
        use_pydantic: true,
        add_validation: true,
        generate_docstrings: true,
        ..Default::default()
    };
    let py_code = generate_python(&aspect, py_options)?;
    fs::write(output_dir.join("vehicle.py"), py_code)?;
    println!("    ✓ Written to target/generated/vehicle.py");

    // Generate Java
    println!("  - Generating Java...");
    let java_options = JavaOptions {
        package_name: "com.example.samm".to_string(),
        ..Default::default()
    };
    let java_code = generate_java(&aspect, java_options)?;
    fs::create_dir_all(output_dir.join("java/com/example/samm"))?;
    fs::write(
        output_dir.join("java/com/example/samm/Vehicle.java"),
        java_code,
    )?;
    println!("    ✓ Written to target/generated/java/com/example/samm/Vehicle.java");

    // Generate Scala
    println!("  - Generating Scala...");
    let scala_options = ScalaOptions {
        package_name: "com.example.samm".to_string(),
        ..Default::default()
    };
    let scala_code = generate_scala(&aspect, scala_options)?;
    fs::create_dir_all(output_dir.join("scala/com/example/samm"))?;
    fs::write(
        output_dir.join("scala/com/example/samm/Vehicle.scala"),
        scala_code,
    )?;
    println!("    ✓ Written to target/generated/scala/com/example/samm/Vehicle.scala");

    // Generate GraphQL
    println!("  - Generating GraphQL schema...");
    let graphql_schema = generate_graphql(&aspect)?;
    fs::write(output_dir.join("vehicle.graphql"), graphql_schema)?;
    println!("    ✓ Written to target/generated/vehicle.graphql");

    // Generate SQL
    println!("  - Generating SQL DDL (PostgreSQL)...");
    let sql_ddl = generate_sql(&aspect, SqlDialect::PostgreSql)?;
    fs::write(output_dir.join("vehicle_pg.sql"), sql_ddl)?;
    println!("    ✓ Written to target/generated/vehicle_pg.sql");

    println!("  - Generating SQL DDL (MySQL)...");
    let sql_ddl = generate_sql(&aspect, SqlDialect::MySql)?;
    fs::write(output_dir.join("vehicle_mysql.sql"), sql_ddl)?;
    println!("    ✓ Written to target/generated/vehicle_mysql.sql");

    println!();

    // Step 4: Multi-file generation example
    println!("Step 4: Demonstrating multi-file generation...");

    // TypeScript with barrel exports
    let ts_multifile_dir = output_dir.join("typescript-multifile");
    let ts_multifile_options = MultiFileOptions {
        output_dir: ts_multifile_dir.clone(),
        layout: OutputLayout::OneEntityPerFile,
        generate_index: true,
        language: "typescript".to_string(),
        generate_docs: true,
        custom_naming: None,
        ts_options: Some(TsOptions {
            export_default: true,
            readonly_properties: true,
            strict_null_checks: true,
            ..Default::default()
        }),
        java_options: None,
        python_options: None,
        scala_options: None,
    };

    let ts_generator = MultiFileGenerator::new(ts_multifile_options);
    let ts_files = ts_generator.generate_typescript(&aspect)?;
    ts_generator.write_files(&ts_files)?;
    println!("  ✓ Generated {} TypeScript files", ts_files.len());
    println!("    Location: target/generated/typescript-multifile/");

    // Python with __init__.py
    let py_multifile_dir = output_dir.join("python-multifile");
    let py_multifile_options = MultiFileOptions {
        output_dir: py_multifile_dir.clone(),
        layout: OutputLayout::OneEntityPerFile,
        generate_index: true,
        language: "python".to_string(),
        generate_docs: true,
        custom_naming: None,
        ts_options: None,
        java_options: None,
        python_options: Some(PythonOptions {
            use_pydantic: true,
            add_validation: true,
            generate_docstrings: true,
            ..Default::default()
        }),
        scala_options: None,
    };

    let py_generator = MultiFileGenerator::new(py_multifile_options);
    let py_files = py_generator.generate_python(&aspect)?;
    py_generator.write_files(&py_files)?;
    println!("  ✓ Generated {} Python files", py_files.len());
    println!("    Location: target/generated/python-multifile/");

    println!();

    // Summary
    println!("=== Generation Complete ===");
    println!("Generated code in 7 languages:");
    println!("  • TypeScript (single file + multi-file)");
    println!("  • Python (single file + multi-file)");
    println!("  • Java");
    println!("  • Scala");
    println!("  • GraphQL");
    println!("  • SQL (PostgreSQL + MySQL)");
    println!();
    println!("All files written to: target/generated/");

    Ok(())
}

/// Create a sample SAMM aspect model for demonstration
fn create_sample_aspect() -> Aspect {
    // Create properties with characteristics
    let mut vin_property = Property::new("urn:samm:com.example:1.0.0#vin".to_string());
    vin_property.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:1.0.0#VinCharacteristic".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = vin_property.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    vin_property.example_values = vec!["WBA12345678901234".to_string()];
    vin_property.optional = false;

    let mut manufacturer_property =
        Property::new("urn:samm:com.example:1.0.0#manufacturer".to_string());
    manufacturer_property.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:1.0.0#Text".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = manufacturer_property.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    manufacturer_property.example_values = vec!["BMW".to_string()];
    manufacturer_property.optional = false;

    let mut model_year_property = Property::new("urn:samm:com.example:1.0.0#modelYear".to_string());
    model_year_property.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:1.0.0#Year".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = model_year_property.characteristic {
        char.data_type = Some("xsd:gYear".to_string());
    }
    model_year_property.example_values = vec!["2024".to_string()];
    model_year_property.optional = false;

    let mut mileage_property = Property::new("urn:samm:com.example:1.0.0#mileage".to_string());
    mileage_property.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:1.0.0#Distance".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:kilometre".to_string(),
        },
    ));
    if let Some(ref mut char) = mileage_property.characteristic {
        char.data_type = Some("xsd:float".to_string());
    }
    mileage_property.example_values = vec!["15000.5".to_string()];
    mileage_property.optional = true;

    // Create the aspect
    let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#Vehicle".to_string());
    aspect.properties = vec![
        vin_property,
        manufacturer_property,
        model_year_property,
        mileage_property,
    ];

    // Add metadata
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle".to_string());
    aspect
        .metadata
        .add_preferred_name("de".to_string(), "Fahrzeug".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "Represents a vehicle with identification and status information.".to_string(),
    );
    aspect.metadata.add_description(
        "de".to_string(),
        "Repräsentiert ein Fahrzeug mit Identifikations- und Statusinformationen.".to_string(),
    );

    aspect
}
