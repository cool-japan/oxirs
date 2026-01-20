//! Example: Creating and Using Custom Generator Plugins
//!
//! This example demonstrates how to create custom code generators using the plugin
//! architecture. It shows:
//! 1. Creating a custom generator by implementing the CodeGenerator trait
//! 2. Registering the generator in the registry
//! 3. Using the generator to generate code from a SAMM model
//!
//! Run this example with:
//! ```bash
//! cargo run --example custom_generator_plugin
//! ```

use oxirs_samm::generators::plugin::{CodeGenerator, GeneratorMetadata, GeneratorRegistry};
use oxirs_samm::metamodel::{Aspect, ElementMetadata, ModelElement, Property};
use oxirs_samm::SammError;
use std::collections::HashMap;

/// Example 1: Simple Markdown Generator
///
/// Generates Markdown documentation from a SAMM Aspect model
struct MarkdownGenerator;

impl CodeGenerator for MarkdownGenerator {
    fn name(&self) -> &str {
        "markdown-docs"
    }

    fn description(&self) -> &str {
        "Generates Markdown documentation from SAMM models"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        let mut markdown = String::new();

        // Title
        markdown.push_str(&format!("# {}\n\n", aspect.name()));

        // Description
        if let Some(desc) = aspect.metadata.get_description("en") {
            markdown.push_str(&format!("**Description**: {}\n\n", desc));
        }

        // Properties section
        if !aspect.properties.is_empty() {
            markdown.push_str("## Properties\n\n");
            for property in &aspect.properties {
                markdown.push_str(&format!("### {}\n\n", property.name()));

                if let Some(desc) = property.metadata.get_description("en") {
                    markdown.push_str(&format!("**Description**: {}\n\n", desc));
                }

                if let Some(characteristic) = &property.characteristic {
                    markdown.push_str(&format!(
                        "- **Type**: {:?}\n",
                        characteristic
                            .data_type
                            .as_ref()
                            .unwrap_or(&"Unknown".to_string())
                    ));
                }

                markdown.push_str(&format!("- **Optional**: {}\n", property.optional));

                if !property.example_values.is_empty() {
                    markdown.push_str("- **Examples**:\n");
                    for example in &property.example_values {
                        markdown.push_str(&format!("  - `{}`\n", example));
                    }
                }

                markdown.push('\n');
            }
        }

        // Operations section
        if !aspect.operations.is_empty() {
            markdown.push_str("## Operations\n\n");
            for operation in &aspect.operations {
                markdown.push_str(&format!("### {}\n\n", operation.name()));

                if let Some(desc) = operation.metadata.get_description("en") {
                    markdown.push_str(&format!("**Description**: {}\n\n", desc));
                }
            }
        }

        Ok(markdown)
    }

    fn file_extension(&self) -> &str {
        "md"
    }

    fn mime_type(&self) -> &str {
        "text/markdown"
    }

    fn metadata(&self) -> GeneratorMetadata {
        GeneratorMetadata {
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            license: Some("MIT".to_string()),
            homepage: Some("https://github.com/cool-japan/oxirs".to_string()),
            custom: HashMap::new(),
        }
    }
}

/// Example 2: REST API Specification Generator
///
/// Generates OpenAPI-like REST endpoint specifications
struct RestApiGenerator;

impl CodeGenerator for RestApiGenerator {
    fn name(&self) -> &str {
        "rest-api"
    }

    fn description(&self) -> &str {
        "Generates REST API endpoint specifications"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        let mut api_spec = String::new();

        api_spec.push_str("# REST API Endpoints\n\n");

        let resource_name = aspect.name().to_lowercase();

        // GET endpoint
        api_spec.push_str(&format!("## GET /{}\n\n", resource_name));
        api_spec.push_str(&format!("Retrieves {} information\n\n", aspect.name()));

        // Response schema
        api_spec.push_str("**Response Schema:**\n\n");
        api_spec.push_str("```json\n{\n");
        for property in &aspect.properties {
            let data_type = property
                .characteristic
                .as_ref()
                .and_then(|c| c.data_type.as_ref())
                .map(|dt| dt.as_str())
                .unwrap_or("string");

            let json_type = match data_type {
                s if s.contains("integer") || s.contains("int") => "number",
                s if s.contains("boolean") => "boolean",
                s if s.contains("decimal") || s.contains("double") || s.contains("float") => {
                    "number"
                }
                _ => "string",
            };

            api_spec.push_str(&format!("  \"{}\": {},\n", property.name(), json_type));
        }
        api_spec.push_str("}\n```\n\n");

        // POST endpoint
        api_spec.push_str(&format!("## POST /{}\n\n", resource_name));
        api_spec.push_str(&format!("Creates a new {} instance\n\n", aspect.name()));

        // Operations as custom endpoints
        for operation in &aspect.operations {
            api_spec.push_str(&format!(
                "## POST /{}/{}\n\n",
                resource_name,
                operation.name().to_lowercase()
            ));

            if let Some(desc) = operation.metadata.get_description("en") {
                api_spec.push_str(&format!("{}\n\n", desc));
            } else {
                api_spec.push_str(&format!("Executes {} operation\n\n", operation.name()));
            }
        }

        Ok(api_spec)
    }

    fn file_extension(&self) -> &str {
        "api.md"
    }
}

/// Example 3: CSV Schema Generator
///
/// Generates CSV schema documentation
struct CsvSchemaGenerator;

impl CodeGenerator for CsvSchemaGenerator {
    fn name(&self) -> &str {
        "csv-schema"
    }

    fn description(&self) -> &str {
        "Generates CSV schema documentation"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        let mut csv = String::new();

        // Header
        csv.push_str("Column,Type,Required,Description,Example\n");

        // Property rows
        for property in &aspect.properties {
            let name = property.name();
            let data_type = property
                .characteristic
                .as_ref()
                .and_then(|c| c.data_type.as_ref())
                .map(|dt| dt.as_str())
                .unwrap_or("string");
            let required = if property.optional { "No" } else { "Yes" };
            let description = property
                .metadata
                .get_description("en")
                .unwrap_or_default()
                .replace(',', ";"); // Escape commas in descriptions
            let example = property.example_values.first().cloned().unwrap_or_default();

            csv.push_str(&format!(
                "{},{},{},\"{}\",{}\n",
                name, data_type, required, description, example
            ));
        }

        Ok(csv)
    }

    fn file_extension(&self) -> &str {
        "csv"
    }

    fn mime_type(&self) -> &str {
        "text/csv"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Custom Generator Plugin Example ===\n");

    // Create a sample SAMM Aspect model
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Vehicle".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "A motorized vehicle for transportation".to_string(),
    );

    // Add properties
    let mut vin_property =
        Property::new("urn:samm:org.example:1.0.0#vehicleIdentificationNumber".to_string());
    vin_property
        .metadata
        .add_preferred_name("en".to_string(), "VIN".to_string());
    vin_property.metadata.add_description(
        "en".to_string(),
        "Vehicle Identification Number".to_string(),
    );
    vin_property.example_values = vec!["1HGBH41JXMN109186".to_string()];
    vin_property.optional = false;

    let mut manufacturer_property =
        Property::new("urn:samm:org.example:1.0.0#manufacturer".to_string());
    manufacturer_property
        .metadata
        .add_preferred_name("en".to_string(), "Manufacturer".to_string());
    manufacturer_property
        .metadata
        .add_description("en".to_string(), "Vehicle manufacturer name".to_string());
    manufacturer_property.example_values = vec!["Toyota".to_string(), "Honda".to_string()];
    manufacturer_property.optional = false;

    let mut year_property = Property::new("urn:samm:org.example:1.0.0#manufactureYear".to_string());
    year_property
        .metadata
        .add_preferred_name("en".to_string(), "Year".to_string());
    year_property
        .metadata
        .add_description("en".to_string(), "Year of manufacture".to_string());
    year_property.example_values = vec!["2024".to_string()];
    year_property.optional = false;

    aspect.properties.push(vin_property);
    aspect.properties.push(manufacturer_property);
    aspect.properties.push(year_property);

    // Create a generator registry and register custom generators
    let registry = GeneratorRegistry::new();

    println!("Registering custom generators...");
    registry.register(Box::new(MarkdownGenerator));
    registry.register(Box::new(RestApiGenerator));
    registry.register(Box::new(CsvSchemaGenerator));

    println!("Registered {} generators\n", registry.count());

    // List all available generators
    println!("Available generators:");
    for name in registry.list() {
        if let Some(gen_ref) = registry.get(&name) {
            println!("  - {} ({})", name, gen_ref.description());
        }
    }
    println!();

    // Generate code using each generator
    println!("=== Markdown Documentation ===\n");
    if let Some(gen) = registry.get("markdown-docs") {
        let markdown = gen.generate(&aspect)?;
        println!("{}", markdown);
    }

    println!("\n=== REST API Specification ===\n");
    if let Some(gen) = registry.get("rest-api") {
        let api_spec = gen.generate(&aspect)?;
        println!("{}", api_spec);
    }

    println!("\n=== CSV Schema ===\n");
    if let Some(gen) = registry.get("csv-schema") {
        let csv = gen.generate(&aspect)?;
        println!("{}", csv);
    }

    // Demonstrate metadata access
    println!("\n=== Generator Metadata ===\n");
    if let Some(gen) = registry.get("markdown-docs") {
        let metadata = gen.metadata();
        println!("Generator: markdown-docs");
        println!("  Version: {:?}", metadata.version);
        println!("  Author: {:?}", metadata.author);
        println!("  License: {:?}", metadata.license);
        println!("  Homepage: {:?}", metadata.homepage);
    }

    println!("\n=== Plugin System Demonstration Complete ===");

    Ok(())
}
