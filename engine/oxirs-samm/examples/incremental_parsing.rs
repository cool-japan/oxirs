//! Example: Incremental Parsing for Large SAMM Models
//!
//! This example demonstrates how to use the incremental parser for large SAMM models.
//! It shows:
//! 1. Parsing large files with progress tracking
//! 2. Saving and resuming parser state
//! 3. Event-based parsing with real-time updates
//! 4. Handling errors during incremental parsing
//!
//! Run this example with:
//! ```bash
//! cargo run --example incremental_parsing
//! ```

use futures::StreamExt;
use oxirs_samm::metamodel::ModelElement;
use oxirs_samm::parser::incremental::{IncrementalParser, ParseEvent, ParseState};
use std::io::Write;
use tempfile::NamedTempFile;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Incremental Parsing Example ===\n");

    // Example 1: Basic incremental parsing with progress tracking
    println!("--- Example 1: Basic Incremental Parsing ---\n");
    example_basic_incremental_parsing().await?;

    // Example 2: Resumable parsing with state persistence
    println!("\n--- Example 2: Resumable Parsing with State Persistence ---\n");
    example_resumable_parsing().await?;

    // Example 3: Event-based parsing with real-time updates
    println!("\n--- Example 3: Event-Based Parsing ---\n");
    example_event_based_parsing().await?;

    // Example 4: Parsing with custom chunk size
    println!("\n--- Example 4: Custom Chunk Size ---\n");
    example_custom_chunk_size().await?;

    // Example 5: Parsing with cancellation
    println!("\n--- Example 5: Parsing with Cancellation ---\n");
    example_parsing_with_cancellation().await?;

    Ok(())
}

/// Example 1: Basic incremental parsing with progress tracking
async fn example_basic_incremental_parsing() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample SAMM file
    let mut temp_file = NamedTempFile::new()?;
    let samm_content = create_sample_samm_model();
    write!(temp_file, "{}", samm_content)?;
    temp_file.flush()?;

    println!("Created sample SAMM file with {} bytes", samm_content.len());

    // Create incremental parser
    let mut parser = IncrementalParser::new(temp_file.path());

    println!("Starting incremental parse...");

    // Parse with progress callback
    let aspect = parser
        .parse_with_progress(|bytes_parsed, total_bytes| {
            let percent = (bytes_parsed as f64 / total_bytes as f64) * 100.0;
            print!(
                "\rProgress: {:.1}% ({}/{})",
                percent, bytes_parsed, total_bytes
            );
            std::io::stdout().flush().unwrap();
            true // Continue parsing
        })
        .await?;

    println!("\n\nParsing complete!");
    println!("Aspect name: {}", aspect.name());
    println!("Properties: {}", aspect.properties.len());
    println!("Operations: {}", aspect.operations.len());

    Ok(())
}

/// Example 2: Resumable parsing with state persistence
async fn example_resumable_parsing() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large sample SAMM file
    let mut temp_file = NamedTempFile::new()?;
    let samm_content = create_large_samm_model();
    write!(temp_file, "{}", samm_content)?;
    temp_file.flush()?;

    println!("Created large SAMM file with {} bytes", samm_content.len());

    // Create parser and parse partially
    let mut parser = IncrementalParser::new(temp_file.path());

    // Parse first chunk and save state
    let state_file = std::env::temp_dir().join("parse_state.json");
    println!(
        "Parsing first chunk and saving state to {:?}...",
        state_file
    );

    // We'll simulate partial parsing by using the event stream
    let mut events = parser.parse_with_events().await?;
    let mut event_count = 0;

    // Process only first 3 events to simulate partial parsing
    while let Some(event) = events.next().await {
        event_count += 1;
        match event {
            ParseEvent::Started { total_bytes } => {
                println!("Started parsing {} bytes", total_bytes);
            }
            ParseEvent::Progress {
                bytes_parsed,
                total_bytes,
            } => {
                let percent = (bytes_parsed as f64 / total_bytes as f64) * 100.0;
                println!("Progress: {:.1}%", percent);
            }
            ParseEvent::Completed { aspect } => {
                println!("Parsing completed: {} properties", aspect.properties.len());
            }
            _ => {}
        }

        if event_count >= 3 {
            break; // Simulate stopping mid-parse
        }
    }

    // Save current state
    parser.save_state(&state_file).await?;
    println!("Parser state saved!");

    // Show progress
    let progress = parser.state().progress_percentage().await?;
    println!("Parse progress: {:.1}%", progress);

    // Load state and resume (in practice, this would be in a new session)
    println!("\nResuming parse from saved state...");
    let loaded_state = ParseState::load_from_file(&state_file).await?;
    println!("Loaded state: {} bytes processed", loaded_state.byte_offset);

    let mut resumed_parser = IncrementalParser::from_state(loaded_state);
    let aspect = resumed_parser.parse().await?;

    println!("Resumed parsing complete!");
    println!(
        "Aspect: {} with {} properties",
        aspect.name(),
        aspect.properties.len()
    );

    Ok(())
}

/// Example 3: Event-based parsing with real-time updates
async fn example_event_based_parsing() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample SAMM file
    let mut temp_file = NamedTempFile::new()?;
    let samm_content = create_sample_samm_model_with_multiple_properties();
    write!(temp_file, "{}", samm_content)?;
    temp_file.flush()?;

    println!("Parsing with event stream...\n");

    let mut parser = IncrementalParser::new(temp_file.path());
    let mut events = parser.parse_with_events().await?;

    let mut property_count = 0;
    let mut operation_count = 0;

    while let Some(event) = events.next().await {
        match event {
            ParseEvent::Started { total_bytes } => {
                println!("📦 Started parsing {} bytes", total_bytes);
            }
            ParseEvent::Progress {
                bytes_parsed,
                total_bytes,
            } => {
                let percent = (bytes_parsed as f64 / total_bytes as f64) * 100.0;
                println!("⏳ Progress: {:.1}%", percent);
            }
            ParseEvent::PropertyParsed { property } => {
                property_count += 1;
                println!("✓ Property parsed: {}", property.name());
            }
            ParseEvent::OperationParsed { operation } => {
                operation_count += 1;
                println!("✓ Operation parsed: {}", operation.name());
            }
            ParseEvent::Completed { aspect } => {
                println!("\n✅ Parsing complete!");
                println!("   Aspect: {}", aspect.name());
                println!("   Properties: {}", aspect.properties.len());
                println!("   Operations: {}", aspect.operations.len());
            }
            ParseEvent::Error { error } => {
                println!("❌ Error: {}", error);
            }
            _ => {}
        }
    }

    println!("\nSummary:");
    println!("  Properties parsed: {}", property_count);
    println!("  Operations parsed: {}", operation_count);

    Ok(())
}

/// Example 4: Parsing with custom chunk size
async fn example_custom_chunk_size() -> Result<(), Box<dyn std::error::Error>> {
    let mut temp_file = NamedTempFile::new()?;
    let samm_content = create_sample_samm_model();
    write!(temp_file, "{}", samm_content)?;
    temp_file.flush()?;

    println!("Parsing with custom chunk size (1KB)...");

    // Create parser with custom chunk size
    let mut parser = IncrementalParser::new(temp_file.path()).with_chunk_size(1024);

    let aspect = parser
        .parse_with_progress(|bytes_parsed, total_bytes| {
            let percent = (bytes_parsed as f64 / total_bytes as f64) * 100.0;
            println!("  Progress: {:.1}% ({} bytes)", percent, bytes_parsed);
            true
        })
        .await?;

    println!("Parsing complete!");
    println!("Aspect: {}", aspect.name());

    Ok(())
}

/// Example 5: Parsing with cancellation support
async fn example_parsing_with_cancellation() -> Result<(), Box<dyn std::error::Error>> {
    let mut temp_file = NamedTempFile::new()?;
    let samm_content = create_large_samm_model();
    write!(temp_file, "{}", samm_content)?;
    temp_file.flush()?;

    println!("Parsing with cancellation (will cancel at 50%)...");

    let mut parser = IncrementalParser::new(temp_file.path());

    // Parse with cancellation
    let result = parser
        .parse_with_progress(|bytes_parsed, total_bytes| {
            let percent = (bytes_parsed as f64 / total_bytes as f64) * 100.0;
            println!("  Progress: {:.1}%", percent);

            // Cancel at 50%
            if percent >= 50.0 {
                println!("  Cancelling parse at 50%...");
                return false; // Cancel parsing
            }
            true // Continue
        })
        .await;

    match result {
        Ok(_) => println!("Parsing completed (not expected in this example)"),
        Err(e) => println!("Parsing cancelled: {}", e),
    }

    Ok(())
}

// Helper functions to create sample SAMM models

fn create_sample_samm_model() -> String {
    r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
@prefix : <urn:samm:org.example:1.0.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:VehicleAspect a samm:Aspect ;
    samm:preferredName "Vehicle Aspect"@en ;
    samm:description "Information about a vehicle"@en ;
    samm:properties ( :vin :manufacturer :model ) ;
    samm:operations ( ) .

:vin a samm:Property ;
    samm:preferredName "VIN"@en ;
    samm:description "Vehicle Identification Number"@en ;
    samm:characteristic :VinCharacteristic .

:manufacturer a samm:Property ;
    samm:preferredName "Manufacturer"@en ;
    samm:description "Vehicle manufacturer"@en ;
    samm:characteristic :TextCharacteristic .

:model a samm:Property ;
    samm:preferredName "Model"@en ;
    samm:description "Vehicle model"@en ;
    samm:characteristic :TextCharacteristic .

:VinCharacteristic a samm:Characteristic ;
    samm:dataType xsd:string .

:TextCharacteristic a samm:Characteristic ;
    samm:dataType xsd:string .
"#
    .to_string()
}

fn create_large_samm_model() -> String {
    let mut model = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
@prefix : <urn:samm:org.example.large:1.0.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:LargeAspect a samm:Aspect ;
    samm:preferredName "Large Aspect"@en ;
    samm:description "A large aspect for testing incremental parsing"@en ;
    samm:properties ( "#
        .to_string();

    // Add many properties
    for i in 1..=20 {
        model.push_str(&format!(":property{} ", i));
    }

    model.push_str(") ;\n    samm:operations ( ) .\n\n");

    // Define all properties
    for i in 1..=20 {
        model.push_str(&format!(
            r#"
:property{} a samm:Property ;
    samm:preferredName "Property {}"@en ;
    samm:description "Description for property {}"@en ;
    samm:characteristic :StringCharacteristic .
"#,
            i, i, i
        ));
    }

    model.push_str(
        r#"
:StringCharacteristic a samm:Characteristic ;
    samm:dataType xsd:string .
"#,
    );

    model
}

fn create_sample_samm_model_with_multiple_properties() -> String {
    r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
@prefix : <urn:samm:org.example.sensor:1.0.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:SensorAspect a samm:Aspect ;
    samm:preferredName "Sensor Aspect"@en ;
    samm:description "Information from a sensor"@en ;
    samm:properties ( :temperature :humidity :pressure :timestamp ) ;
    samm:operations ( :reset :calibrate ) .

:temperature a samm:Property ;
    samm:preferredName "Temperature"@en ;
    samm:description "Temperature in Celsius"@en ;
    samm:characteristic :TemperatureCharacteristic .

:humidity a samm:Property ;
    samm:preferredName "Humidity"@en ;
    samm:description "Relative humidity in percent"@en ;
    samm:characteristic :PercentCharacteristic .

:pressure a samm:Property ;
    samm:preferredName "Pressure"@en ;
    samm:description "Atmospheric pressure in hPa"@en ;
    samm:characteristic :PressureCharacteristic .

:timestamp a samm:Property ;
    samm:preferredName "Timestamp"@en ;
    samm:description "Measurement timestamp"@en ;
    samm:characteristic :TimestampCharacteristic .

:TemperatureCharacteristic a samm:Characteristic ;
    samm:dataType xsd:float .

:PercentCharacteristic a samm:Characteristic ;
    samm:dataType xsd:float .

:PressureCharacteristic a samm:Characteristic ;
    samm:dataType xsd:float .

:TimestampCharacteristic a samm:Characteristic ;
    samm:dataType xsd:dateTime .

:reset a samm:Operation ;
    samm:preferredName "Reset"@en ;
    samm:description "Reset the sensor"@en .

:calibrate a samm:Operation ;
    samm:preferredName "Calibrate"@en ;
    samm:description "Calibrate the sensor"@en .
"#
    .to_string()
}
