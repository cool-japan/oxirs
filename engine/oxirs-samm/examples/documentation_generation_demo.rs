//! Documentation Generation Demonstration
//!
//! This example demonstrates the comprehensive documentation generation capabilities:
//! - HTML documentation with analytics
//! - Markdown documentation for GitHub
//! - JSON structured documentation
//! - Multiple documentation styles
//! - Custom templates and styling
//!
//! Run with: cargo run --example documentation_generation_demo

use oxirs_samm::documentation::{DocumentationFormat, DocumentationGenerator, DocumentationStyle};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Entity, ModelElement, Operation, Property,
};
use std::fs;

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("  OxiRS SAMM - Documentation Generation Demo");
    println!("═══════════════════════════════════════════════════\n");

    // Create a comprehensive test aspect
    let aspect = create_vehicle_aspect();

    // Example 1: Generate HTML Documentation with Analytics
    println!("Example 1: HTML Documentation with Quality Analytics");
    println!("───────────────────────────────────────────────────");
    generate_html_with_analytics(&aspect);

    // Example 2: Generate Markdown Documentation
    println!("\nExample 2: GitHub-Compatible Markdown");
    println!("───────────────────────────────────────────────────");
    generate_markdown_docs(&aspect);

    // Example 3: Generate JSON Structured Documentation
    println!("\nExample 3: JSON Structured Documentation");
    println!("───────────────────────────────────────────────────");
    generate_json_docs(&aspect);

    // Example 4: Multiple Documentation Styles
    println!("\nExample 4: Documentation Styles Comparison");
    println!("───────────────────────────────────────────────────");
    generate_multiple_styles(&aspect);

    // Example 5: Custom Styled Documentation
    println!("\nExample 5: Custom CSS Styling");
    println!("───────────────────────────────────────────────────");
    generate_custom_styled(&aspect);

    println!("\n═══════════════════════════════════════════════════");
    println!("  Documentation generation complete!");
    println!("  Check /tmp/ directory for generated files");
    println!("═══════════════════════════════════════════════════");
}

/// Example 1: Generate HTML documentation with full analytics
fn generate_html_with_analytics(aspect: &Aspect) {
    let generator = DocumentationGenerator::new()
        .with_format(DocumentationFormat::Html)
        .with_style(DocumentationStyle::Complete)
        .with_analytics(true)
        .with_table_of_contents(true)
        .with_examples(true)
        .with_title("Vehicle Aspect - Complete Documentation".to_string())
        .with_footer("Generated with OxiRS SAMM Documentation Generator".to_string());

    match generator.generate(aspect) {
        Ok(html) => {
            let path = "/tmp/vehicle_docs_complete.html";
            if let Err(e) = fs::write(path, html) {
                eprintln!("Failed to write HTML: {}", e);
            } else {
                println!("✓ Generated complete HTML documentation");
                println!("  File: {}", path);
                println!("  Includes: Analytics, TOC, Examples, Quality Score");
            }
        }
        Err(e) => eprintln!("Generation failed: {}", e),
    }
}

/// Example 2: Generate Markdown documentation for GitHub
fn generate_markdown_docs(aspect: &Aspect) {
    let generator = DocumentationGenerator::new()
        .with_format(DocumentationFormat::Markdown)
        .with_style(DocumentationStyle::Technical)
        .with_analytics(true)
        .with_footer("This documentation was automatically generated.".to_string());

    match generator.generate(aspect) {
        Ok(markdown) => {
            let path = "/tmp/vehicle_docs.md";
            if let Err(e) = fs::write(path, markdown) {
                eprintln!("Failed to write Markdown: {}", e);
            } else {
                println!("✓ Generated Markdown documentation");
                println!("  File: {}", path);
                println!("  Format: GitHub-compatible Markdown");
                println!("  Includes: Quality metrics, property tables");
            }
        }
        Err(e) => eprintln!("Generation failed: {}", e),
    }
}

/// Example 3: Generate JSON structured documentation
fn generate_json_docs(aspect: &Aspect) {
    let generator = DocumentationGenerator::new()
        .with_format(DocumentationFormat::Json)
        .with_analytics(true);

    match generator.generate(aspect) {
        Ok(json) => {
            let path = "/tmp/vehicle_docs.json";
            if let Err(e) = fs::write(path, &json) {
                eprintln!("Failed to write JSON: {}", e);
            } else {
                println!("✓ Generated JSON documentation");
                println!("  File: {}", path);
                println!("  Format: Structured JSON");
                println!("  Sample output:");
                // Print first 300 chars as preview
                let preview = if json.len() > 300 {
                    format!("{}...", &json[..300])
                } else {
                    json
                };
                println!("{}", preview);
            }
        }
        Err(e) => eprintln!("Generation failed: {}", e),
    }
}

/// Example 4: Generate multiple documentation styles
fn generate_multiple_styles(aspect: &Aspect) {
    let styles = vec![
        (DocumentationStyle::Technical, "technical"),
        (DocumentationStyle::UserFriendly, "user_friendly"),
        (DocumentationStyle::Api, "api"),
    ];

    for (style, name) in styles {
        let generator = DocumentationGenerator::new()
            .with_format(DocumentationFormat::Html)
            .with_style(style)
            .with_analytics(false)
            .with_table_of_contents(true);

        match generator.generate(aspect) {
            Ok(html) => {
                let path = format!("/tmp/vehicle_docs_{}.html", name);
                if let Err(e) = fs::write(&path, html) {
                    eprintln!("Failed to write {}: {}", name, e);
                } else {
                    println!("✓ Generated {:?} style", style);
                    println!("  File: {}", path);
                }
            }
            Err(e) => eprintln!("Generation failed for {:?}: {}", style, e),
        }
    }
}

/// Example 5: Generate documentation with custom CSS
fn generate_custom_styled(aspect: &Aspect) {
    let custom_css = r#"
body {
    font-family: 'Georgia', serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #2d3748;
}

.container {
    background: white;
    border-radius: 15px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

header h1 {
    color: #5a67d8;
    font-size: 2.5em;
}

.quality-score {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.section h2 {
    color: #667eea;
    font-size: 1.8em;
}

table {
    border-radius: 8px;
    overflow: hidden;
}

th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
"#;

    let generator = DocumentationGenerator::new()
        .with_format(DocumentationFormat::Html)
        .with_style(DocumentationStyle::Complete)
        .with_analytics(true)
        .with_custom_css(custom_css.to_string())
        .with_title("Vehicle Aspect - Premium Edition".to_string());

    match generator.generate(aspect) {
        Ok(html) => {
            let path = "/tmp/vehicle_docs_premium.html";
            if let Err(e) = fs::write(path, html) {
                eprintln!("Failed to write custom styled HTML: {}", e);
            } else {
                println!("✓ Generated premium styled documentation");
                println!("  File: {}", path);
                println!("  Features: Custom gradient theme, rounded corners");
                println!("  Style: Professional with purple gradient");
            }
        }
        Err(e) => eprintln!("Generation failed: {}", e),
    }
}

/// Create a comprehensive vehicle aspect for documentation
fn create_vehicle_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example.automotive:1.0.0#Vehicle".to_string());

    // Add comprehensive metadata
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle".to_string());
    aspect
        .metadata
        .add_preferred_name("de".to_string(), "Fahrzeug".to_string());
    aspect
        .metadata
        .add_preferred_name("fr".to_string(), "Véhicule".to_string());

    aspect.metadata.add_description(
        "en".to_string(),
        "Represents a motor vehicle with its essential characteristics, operational parameters, and identification information.".to_string(),
    );
    aspect.metadata.add_description(
        "de".to_string(),
        "Repräsentiert ein Kraftfahrzeug mit seinen wesentlichen Merkmalen, Betriebsparametern und Identifikationsinformationen.".to_string(),
    );

    // VIN (Vehicle Identification Number)
    let mut vin = Property::new("urn:samm:org.example.automotive:1.0.0#vin".to_string());
    vin.metadata
        .add_preferred_name("en".to_string(), "VIN".to_string());
    vin.metadata.add_description(
        "en".to_string(),
        "Vehicle Identification Number - unique 17-character identifier".to_string(),
    );
    let mut vin_char = Characteristic::new(
        "urn:samm:org.example.automotive:1.0.0#VinTrait".to_string(),
        CharacteristicKind::Trait,
    );
    vin_char.data_type = Some("xsd:string".to_string());
    vin.characteristic = Some(vin_char);
    vin.optional = false;
    aspect.add_property(vin);

    // Manufacturer
    let mut manufacturer =
        Property::new("urn:samm:org.example.automotive:1.0.0#manufacturer".to_string());
    manufacturer
        .metadata
        .add_preferred_name("en".to_string(), "Manufacturer".to_string());
    manufacturer.metadata.add_description(
        "en".to_string(),
        "Vehicle manufacturer name (e.g., BMW, Mercedes-Benz, Toyota)".to_string(),
    );
    let mut mfr_char = Characteristic::new(
        "urn:samm:org.example.automotive:1.0.0#ManufacturerTrait".to_string(),
        CharacteristicKind::Trait,
    );
    mfr_char.data_type = Some("xsd:string".to_string());
    manufacturer.characteristic = Some(mfr_char);
    manufacturer.optional = false;
    aspect.add_property(manufacturer);

    // Model Year
    let mut model_year =
        Property::new("urn:samm:org.example.automotive:1.0.0#modelYear".to_string());
    model_year
        .metadata
        .add_preferred_name("en".to_string(), "Model Year".to_string());
    model_year.metadata.add_description(
        "en".to_string(),
        "Year the vehicle was manufactured".to_string(),
    );
    let mut year_char = Characteristic::new(
        "urn:samm:org.example.automotive:1.0.0#YearTrait".to_string(),
        CharacteristicKind::Trait,
    );
    year_char.data_type = Some("xsd:gYear".to_string());
    model_year.characteristic = Some(year_char);
    model_year.optional = false;
    aspect.add_property(model_year);

    // Mileage
    let mut mileage = Property::new("urn:samm:org.example.automotive:1.0.0#mileage".to_string());
    mileage
        .metadata
        .add_preferred_name("en".to_string(), "Mileage".to_string());
    mileage.metadata.add_description(
        "en".to_string(),
        "Total distance traveled by the vehicle in kilometers".to_string(),
    );
    let mut mileage_char = Characteristic::new(
        "urn:samm:org.example.automotive:1.0.0#MileageMeasurement".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:kilometre".to_string(),
        },
    );
    mileage_char.data_type = Some("xsd:decimal".to_string());
    mileage.characteristic = Some(mileage_char);
    mileage.optional = true;
    aspect.add_property(mileage);

    // Color
    let mut color = Property::new("urn:samm:org.example.automotive:1.0.0#color".to_string());
    color
        .metadata
        .add_preferred_name("en".to_string(), "Color".to_string());
    color.metadata.add_description(
        "en".to_string(),
        "Exterior color of the vehicle".to_string(),
    );
    let mut color_char = Characteristic::new(
        "urn:samm:org.example.automotive:1.0.0#ColorTrait".to_string(),
        CharacteristicKind::Trait,
    );
    color_char.data_type = Some("xsd:string".to_string());
    color.characteristic = Some(color_char);
    color.optional = true;
    aspect.add_property(color);

    // Fuel Type
    let mut fuel_type = Property::new("urn:samm:org.example.automotive:1.0.0#fuelType".to_string());
    fuel_type
        .metadata
        .add_preferred_name("en".to_string(), "Fuel Type".to_string());
    fuel_type.metadata.add_description(
        "en".to_string(),
        "Type of fuel used by the vehicle".to_string(),
    );
    let mut fuel_char = Characteristic::new(
        "urn:samm:org.example.automotive:1.0.0#FuelTypeEnumeration".to_string(),
        CharacteristicKind::Enumeration {
            values: vec![
                "Gasoline".to_string(),
                "Diesel".to_string(),
                "Electric".to_string(),
                "Hybrid".to_string(),
            ],
        },
    );
    fuel_char.data_type = Some("xsd:string".to_string());
    fuel_type.characteristic = Some(fuel_char);
    fuel_type.optional = false;
    aspect.add_property(fuel_type);

    // Registration Date
    let mut registration =
        Property::new("urn:samm:org.example.automotive:1.0.0#registrationDate".to_string());
    registration
        .metadata
        .add_preferred_name("en".to_string(), "Registration Date".to_string());
    registration.metadata.add_description(
        "en".to_string(),
        "Date when the vehicle was first registered".to_string(),
    );
    let mut reg_char = Characteristic::new(
        "urn:samm:org.example.automotive:1.0.0#RegistrationDateTrait".to_string(),
        CharacteristicKind::Trait,
    );
    reg_char.data_type = Some("xsd:date".to_string());
    registration.characteristic = Some(reg_char);
    registration.optional = true;
    aspect.add_property(registration);

    // Add operations
    let start_op = Operation::new("urn:samm:org.example.automotive:1.0.0#startEngine".to_string());
    aspect.add_operation(start_op);

    let stop_op = Operation::new("urn:samm:org.example.automotive:1.0.0#stopEngine".to_string());
    aspect.add_operation(stop_op);

    let refuel_op = Operation::new("urn:samm:org.example.automotive:1.0.0#refuel".to_string());
    aspect.add_operation(refuel_op);

    aspect
}
