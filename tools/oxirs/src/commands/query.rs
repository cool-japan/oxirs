//! SPARQL query command

use super::stubs::Store;
use super::CommandResult;
use crate::cli::error::helpers as error_helpers;
use crate::cli::logging::QueryLogger;
use crate::cli::validation::MultiValidator;
use crate::cli::validation::{dataset_validation, query_validation};
use crate::cli::{progress::helpers, ArgumentValidator, CliContext, CliError};
use serde_json;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Execute SPARQL query against a dataset
pub async fn run(dataset: String, query: String, file: bool, output: String) -> CommandResult {
    // Create CLI context for proper output formatting
    let ctx = CliContext::new();

    // Validate arguments using the advanced validation framework
    let mut validator = MultiValidator::new();

    // Validate dataset name
    validator.add(
        ArgumentValidator::new("dataset", Some(&dataset))
            .required()
            .custom(|d| !d.trim().is_empty(), "Dataset name cannot be empty"),
    );
    dataset_validation::validate_dataset_name(&dataset)?;

    // Validate output format
    validator.add(
        ArgumentValidator::new("output", Some(&output))
            .required()
            .one_of(&["json", "csv", "tsv", "table", "xml"]),
    );

    // Validate query file if needed
    if file {
        let query_path = PathBuf::from(&query);
        validator.add(
            ArgumentValidator::new("query_file", Some(query_path.to_str().unwrap_or("")))
                .required()
                .is_file(),
        );
    }

    // Complete validation
    validator.finish()?;

    ctx.info(&format!("Executing SPARQL query on dataset '{dataset}'"));

    // Load query from file or use directly
    let sparql_query = if file {
        let query_path = PathBuf::from(&query);

        let pb = helpers::file_progress(1);
        pb.set_message("Reading query file");
        let content = fs::read_to_string(&query_path)?;
        pb.finish_with_message("Query file loaded");
        content
    } else {
        query
    };

    // Validate SPARQL syntax
    query_validation::validate_sparql_syntax(&sparql_query)?;

    if ctx.should_show_verbose() {
        ctx.info("Query:");
        ctx.verbose(&sparql_query);
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
        return Err(error_helpers::dataset_not_found_error(&dataset));
    };

    // Execute query with progress tracking and logging
    let start_time = Instant::now();

    // Initialize query logger
    let mut query_logger = QueryLogger::new("sparql_query", &dataset);
    query_logger.add_query_text(&sparql_query);

    // Create progress spinner for query execution
    let query_progress = helpers::query_progress();
    query_progress.set_message("Executing SPARQL query");

    let results = match store.query(&sparql_query) {
        Ok(res) => {
            query_logger.complete(res.bindings.len());
            res
        }
        Err(e) => {
            query_logger.error(&e.to_string());
            return Err(CliError::from(e));
        }
    };

    let duration = start_time.elapsed();

    // Format and display results
    query_progress
        .finish_with_message(format!("Query completed in {:.3}s", duration.as_secs_f64()));

    // Display statistics
    ctx.info("Query Results");
    ctx.info(&format!(
        "Execution time: {:.3} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!(
        "Result count: {} bindings",
        results.bindings.len()
    ));

    // Format and display results based on output format
    format_results_enhanced(&results, &output, &ctx)?;

    Ok(())
}

/// Check if output format is supported
#[allow(dead_code)]
fn is_supported_output_format(format: &str) -> bool {
    matches!(format, "json" | "csv" | "tsv" | "table" | "xml")
}

/// Load dataset configuration from oxirs.toml file
fn load_dataset_from_config(dataset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_path = PathBuf::from(dataset).join("oxirs.toml");

    if !config_path.exists() {
        return Err(error_helpers::file_not_found_error(&config_path).into());
    }

    // For now, just return the dataset directory
    // TODO: Parse TOML configuration and extract actual storage path
    Ok(PathBuf::from(dataset))
}

/// Format and display query results
#[allow(dead_code)]
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
            return Err(format!("Output format '{format}' not implemented").into());
        }
    }

    Ok(())
}

/// Format results as a table
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Enhanced format results using CLI context with comprehensive formatters
fn format_results_enhanced(
    results: &super::stubs::OxirsQueryResults,
    output_format: &str,
    ctx: &crate::cli::CliContext,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::cli::formatters::{create_formatter, Binding, QueryResults, RdfTerm, ResultFormatter};
    use std::io;

    // Convert stub results to formatter QueryResults
    let formatter_results = QueryResults {
        variables: results.variables.clone(),
        bindings: results
            .bindings
            .iter()
            .map(|stub_binding| Binding {
                values: stub_binding
                    .values
                    .iter()
                    .map(|opt_val| {
                        opt_val.as_ref().map(|v| RdfTerm::Literal {
                            value: v.clone(),
                            lang: None,
                            datatype: None,
                        })
                    })
                    .collect(),
            })
            .collect(),
    };

    // Use the comprehensive formatter
    if let Some(formatter) = create_formatter(output_format) {
        let mut stdout = io::stdout();
        formatter.format(&formatter_results, &mut stdout)?;
    } else {
        return Err(format!("Unsupported output format: {output_format}").into());
    }

    Ok(())
}
