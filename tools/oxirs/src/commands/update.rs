//! SPARQL update command

use super::CommandResult;
use oxirs_core::rdf_store::RdfStore;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Execute SPARQL update against a dataset
pub async fn run(dataset: String, update: String, file: bool) -> CommandResult {
    println!("Executing SPARQL update on dataset '{dataset}'");

    // Load update from file or use directly
    let sparql_update = if file {
        let update_path = PathBuf::from(&update);
        if !update_path.exists() {
            return Err(format!("Update file '{}' does not exist", update_path.display()).into());
        }
        fs::read_to_string(update_path)?
    } else {
        update
    };

    println!("Update:");
    println!("---");
    println!("{sparql_update}");
    println!("---");

    // Load dataset configuration or use dataset path directly
    let dataset_path = if PathBuf::from(&dataset).join("oxirs.toml").exists() {
        // Dataset with configuration file
        load_dataset_from_config(&dataset)?
    } else {
        // Assume dataset is a directory path
        PathBuf::from(&dataset)
    };

    // Open or create store
    println!("Opening dataset at: {}", dataset_path.display());
    let store = if dataset_path.exists() {
        RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {e}"))?
    } else {
        println!("Creating new dataset at: {}", dataset_path.display());
        RdfStore::open(&dataset_path).map_err(|e| format!("Failed to create dataset: {e}"))?
    };

    // Execute update
    let start_time = Instant::now();
    println!("\nExecuting SPARQL update...");

    execute_update(&store, &sparql_update)?;

    let duration = start_time.elapsed();

    println!(
        "Update executed successfully in {:.3} seconds",
        duration.as_secs_f64()
    );

    Ok(())
}

/// Load dataset configuration from oxirs.toml file
fn load_dataset_from_config(dataset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Use shared configuration loader
    crate::config::load_dataset_from_config(dataset)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Execute SPARQL update operation
fn execute_update(store: &RdfStore, update: &str) -> Result<(), Box<dyn std::error::Error>> {
    use oxirs_core::query::update::{UpdateExecutor, UpdateParser};

    // Step 1: Parse the SPARQL update query
    println!("   [1/2] Parsing SPARQL UPDATE...");
    let parser = UpdateParser::new();
    let parsed_update = parser
        .parse(update)
        .map_err(|e| format!("SPARQL UPDATE parse error: {e}"))?;

    println!("       ✓ Parsed successfully");
    println!("       Operations: {}", parsed_update.operations.len());

    // Step 2: Execute the parsed update
    println!("   [2/2] Executing update operations...");
    let executor = UpdateExecutor::new(store);
    executor
        .execute(&parsed_update)
        .map_err(|e| format!("SPARQL UPDATE execution error: {e}"))?;

    println!("       ✓ All operations completed successfully");

    Ok(())
}
