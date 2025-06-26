//! SPARQL query command

use super::CommandResult;
use super::stubs::Store;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Execute SPARQL query against a dataset
pub async fn run(dataset: String, query: String, file: bool, output: String) -> CommandResult {
    println!("Executing SPARQL query on dataset '{}'", dataset);

    // Load query from file or use directly
    let sparql_query = if file {
        let query_path = PathBuf::from(&query);
        if !query_path.exists() {
            return Err(format!("Query file '{}' does not exist", query_path.display()).into());
        }
        fs::read_to_string(query_path)?
    } else {
        query
    };

    println!("Query:");
    println!("---");
    println!("{}", sparql_query);
    println!("---");

    // Validate output format
    if !is_supported_output_format(&output) {
        return Err(format!(
            "Unsupported output format '{}'. Supported formats: json, csv, tsv, table, xml",
            output
        )
        .into());
    }

    // Load dataset configuration or use dataset path directly
    let dataset_path = if PathBuf::from(&dataset).join("oxirs.toml").exists() {
        // Dataset with configuration file
        load_dataset_from_config(&dataset)?
    } else {
        // Assume dataset is a directory path
        PathBuf::from(&dataset)
    };

    // Open store
    let store = if dataset_path.is_dir() {
        Store::open(&dataset_path)?
    } else {
        return Err(format!(
            "Dataset '{}' not found. Use 'oxide init' to create a dataset.",
            dataset
        )
        .into());
    };

    // Execute query
    let start_time = Instant::now();
    println!("Executing query...");

    let results = store.query(&sparql_query)?;
    let duration = start_time.elapsed();

    // Format and display results
    println!("Query executed in {:.3} seconds", duration.as_secs_f64());
    println!();

    format_results(&results, &output)?;

    Ok(())
}

/// Check if output format is supported
fn is_supported_output_format(format: &str) -> bool {
    matches!(format, "json" | "csv" | "tsv" | "table" | "xml")
}

/// Load dataset configuration from oxirs.toml file
fn load_dataset_from_config(dataset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_path = PathBuf::from(dataset).join("oxirs.toml");

    if !config_path.exists() {
        return Err(format!("Configuration file '{}' not found", config_path.display()).into());
    }

    // For now, just return the dataset directory
    // TODO: Parse TOML configuration and extract actual storage path
    Ok(PathBuf::from(dataset))
}

/// Format and display query results
fn format_results(
    results: &super::stubs::OxirsQueryResults,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        "table" => {
            format_table_results(results)?;
        }
        "json" => {
            format_json_results(results)?;
        }
        "csv" => {
            format_csv_results(results, ",")?;
        }
        "tsv" => {
            format_csv_results(results, "\t")?;
        }
        "xml" => {
            format_xml_results(results)?;
        }
        _ => {
            return Err(format!("Output format '{}' not implemented", format).into());
        }
    }

    Ok(())
}

/// Format results as a table
fn format_table_results(
    _results: &super::stubs::OxirsQueryResults,
) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Implement proper table formatting
    println!("Results (table format):");
    println!("┌─────────────────────────────────────────┐");
    println!("│ No results - implementation pending    │");
    println!("└─────────────────────────────────────────┘");
    Ok(())
}

/// Format results as JSON
fn format_json_results(
    _results: &super::stubs::OxirsQueryResults,
) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Implement proper JSON formatting
    println!("Results (JSON format):");
    println!("{{");
    println!("  \"head\": {{ \"vars\": [] }},");
    println!("  \"results\": {{ \"bindings\": [] }}");
    println!("}}");
    Ok(())
}

/// Format results as CSV/TSV
fn format_csv_results(
    _results: &super::stubs::OxirsQueryResults,
    _separator: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Implement proper CSV/TSV formatting
    println!("Results (CSV/TSV format):");
    println!("# No results - implementation pending");
    Ok(())
}

/// Format results as XML
fn format_xml_results(
    _results: &super::stubs::OxirsQueryResults,
) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Implement proper XML formatting
    println!("Results (XML format):");
    println!("<?xml version=\"1.0\"?>");
    println!("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">");
    println!("  <head></head>");
    println!("  <results></results>");
    println!("</sparql>");
    Ok(())
}
