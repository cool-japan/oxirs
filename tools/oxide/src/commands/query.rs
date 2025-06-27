//! SPARQL query command

use super::CommandResult;
use super::stubs::Store;
use crate::cli::{ArgumentValidator, CliContext, progress::helpers};
use crate::cli::validation::MultiValidator;
use crate::cli::validation::{dataset_validation, query_validation};
use crate::cli::error::helpers as error_helpers;
use crate::cli::logging::QueryLogger;
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
            .custom(|d| !d.trim().is_empty(), "Dataset name cannot be empty")
    );
    dataset_validation::validate_dataset_name(&dataset)?;
    
    // Validate output format
    validator.add(
        ArgumentValidator::new("output", Some(&output))
            .required()
            .one_of(&["json", "csv", "tsv", "table", "xml"])
    );
    
    // Validate query file if needed
    if file {
        let query_path = PathBuf::from(&query);
        validator.add(
            ArgumentValidator::new("query_file", Some(query_path.to_str().unwrap_or("")))
                .required()
                .is_file()
        );
    }
    
    // Complete validation
    validator.finish()?;
    
    ctx.info(&format!("Executing SPARQL query on dataset '{}'", dataset));

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
        return Err(error_helpers::dataset_not_found_error(&dataset).into());
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
        },
        Err(e) => {
            query_logger.error(&e.to_string());
            return Err(e);
        }
    };
    
    let duration = start_time.elapsed();

    // Format and display results
    query_progress.finish_with_message(format!("Query completed in {:.3}s", duration.as_secs_f64()));
    
    // Display statistics
    ctx.info("Query Results");
    ctx.info(&format!("Execution time: {:.3} seconds", duration.as_secs_f64()));
    ctx.info(&format!("Result count: {} bindings", results.bindings.len()));
    
    // Format and display results based on output format
    format_results_enhanced(&results, &output, &ctx)?;

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
        return Err(error_helpers::file_not_found_error(&config_path).into());
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

/// Enhanced format results using CLI context
fn format_results_enhanced(
    results: &super::stubs::OxirsQueryResults,
    output_format: &str,
    ctx: &crate::cli::CliContext,
) -> Result<(), Box<dyn std::error::Error>> {
    match output_format {
        "table" => {
            // Use the table formatter from output module
            let mut table = ctx.output_formatter.create_table();
            if !results.variables.is_empty() {
                use prettytable::{Row, Cell};
                table.set_titles(Row::new(
                    results.variables.iter().map(|v| Cell::new(v)).collect()
                ));
            }
            
            for binding in &results.bindings {
                let cells: Vec<prettytable::Cell> = binding.values.iter()
                    .map(|opt| opt.as_deref().unwrap_or(""))
                    .map(|s| prettytable::Cell::new(s))
                    .collect();
                table.add_row(prettytable::Row::new(cells));
            }
            
            table.printstd();
        },
        "json" => {
            let json_output = serde_json::json!({
                "head": { "vars": results.variables },
                "results": {
                    "bindings": results.bindings.iter().map(|b| {
                        let mut binding_map = serde_json::Map::new();
                        for (i, var) in results.variables.iter().enumerate() {
                            if let Some(Some(value)) = b.values.get(i) {
                                binding_map.insert(var.clone(), serde_json::Value::String(value.clone()));
                            }
                        }
                        binding_map
                    }).collect::<Vec<_>>()
                }
            });
            ctx.output_formatter.json(&json_output)?;
        },
        "csv" | "tsv" => {
            let separator = if output_format == "csv" { "," } else { "\t" };
            
            // Print headers
            println!("{}", results.variables.join(separator));
            
            // Print rows
            for binding in &results.bindings {
                let values: Vec<_> = binding.values.iter()
                    .map(|opt| opt.as_deref().unwrap_or(""))
                    .collect();
                println!("{}", values.join(separator));
            }
        },
        "xml" => {
            format_xml_results(results)?;
        }
        _ => {
            // This should not happen due to validation
            return Err(format!("Unsupported output format: {}", output_format).into());
        }
    }
    
    Ok(())
}
