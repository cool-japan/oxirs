//! SPARQL query command

use super::CommandResult;
use crate::cli::error::helpers as error_helpers;
use crate::cli::logging::QueryLogger;
use crate::cli::validation::MultiValidator;
use crate::cli::validation::{dataset_validation, query_validation};
use crate::cli::{progress::helpers, ArgumentValidator, CliContext};
use oxirs_core::rdf_store::RdfStore;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Execute SPARQL query against a dataset
pub async fn run(dataset: String, query: String, file: bool, output: String) -> CommandResult {
    // Create CLI context for proper output formatting
    let ctx = CliContext::new();

    // Validate arguments using the advanced validation framework
    let mut validator = MultiValidator::new();

    // Validate dataset name (only if it's not a path to an existing directory)
    validator.add(
        ArgumentValidator::new("dataset", Some(&dataset))
            .required()
            .custom(|d| !d.trim().is_empty(), "Dataset name cannot be empty"),
    );

    // Only validate dataset name format if it's not an existing directory path
    if !PathBuf::from(&dataset).exists() {
        dataset_validation::validate_dataset_name(&dataset)?;
    }

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
    let dataset_dir = PathBuf::from(&dataset);
    let dataset_path = if dataset_dir.join("oxirs.toml").exists() {
        // Dataset with configuration file - extract dataset name from directory name
        let dataset_name = dataset_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&dataset);
        let (storage_path, _config) =
            crate::config::load_named_dataset(&dataset_dir, dataset_name)?;
        storage_path
    } else {
        // Assume dataset is a directory path
        dataset_dir
    };

    // Open store
    let store = if dataset_path.is_dir() {
        RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {e}"))?
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
            let binding_count = res.len();
            query_logger.complete(binding_count);
            res
        }
        Err(e) => {
            query_logger.error(&e.to_string());
            return Err(format!("Query execution failed: {e}").into());
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
    ctx.info(&format!("Result count: {} solutions", results.len()));

    // Format and display results based on output format
    format_results_enhanced(&results, &output, &ctx)?;

    Ok(())
}

/// Check if output format is supported
#[allow(dead_code)]
fn is_supported_output_format(format: &str) -> bool {
    matches!(format, "json" | "csv" | "tsv" | "table" | "xml")
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
    results: &oxirs_core::rdf_store::OxirsQueryResults,
    output_format: &str,
    _ctx: &crate::cli::CliContext,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::cli::formatters::{
        create_formatter, Binding, QueryResults, RdfTerm,
    };
    use oxirs_core::rdf_store::QueryResults as CoreQueryResults;
    use std::io;

    // Convert real OxirsQueryResults to formatter QueryResults
    let formatter_results = match results.results() {
        CoreQueryResults::Bindings(bindings) => {
            let variables = results.variables();

            QueryResults {
                variables: variables.to_vec(),
                bindings: bindings
                    .iter()
                    .map(|var_binding| {
                        // Get values in the order of variables
                        let values: Vec<Option<RdfTerm>> = variables
                            .iter()
                            .map(|var| var_binding.get(var).map(|term| term_to_rdf_term(term)))
                            .collect();

                        Binding { values }
                    })
                    .collect(),
            }
        }
        CoreQueryResults::Boolean(value) => {
            // For ASK queries, return a single binding with the boolean result
            QueryResults {
                variables: vec!["result".to_string()],
                bindings: vec![Binding {
                    values: vec![Some(RdfTerm::Literal {
                        value: value.to_string(),
                        lang: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
                    })],
                }],
            }
        }
        CoreQueryResults::Graph(quads) => {
            // For CONSTRUCT/DESCRIBE queries, convert quads to bindings
            QueryResults {
                variables: vec![
                    "subject".to_string(),
                    "predicate".to_string(),
                    "object".to_string(),
                ],
                bindings: quads
                    .iter()
                    .map(|quad| Binding {
                        values: vec![
                            Some(subject_to_rdf_term(quad.subject())),
                            Some(predicate_to_rdf_term(quad.predicate())),
                            Some(object_to_rdf_term(quad.object())),
                        ],
                    })
                    .collect(),
            }
        }
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

/// Convert Term to RdfTerm
fn term_to_rdf_term(term: &oxirs_core::model::Term) -> crate::cli::formatters::RdfTerm {
    use crate::cli::formatters::RdfTerm;
    use oxirs_core::model::Term;

    match term {
        Term::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Term::BlankNode(bnode) => RdfTerm::Bnode {
            value: bnode.as_str().to_string(),
        },
        Term::Literal(lit) => RdfTerm::Literal {
            value: lit.value().to_string(),
            lang: lit.language().map(|l| l.to_string()),
            datatype: Some(lit.datatype().as_str().to_string()),
        },
        Term::Variable(var) => {
            // Variables displayed as special literals
            RdfTerm::Literal {
                value: format!("?{}", var.name()),
                lang: None,
                datatype: None,
            }
        }
        Term::QuotedTriple(triple) => {
            // RDF-star quoted triples displayed as special literals
            RdfTerm::Literal {
                value: format!(
                    "<<{} {} {}>>",
                    triple.subject(),
                    triple.predicate(),
                    triple.object()
                ),
                lang: None,
                datatype: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string()),
            }
        }
    }
}

/// Convert Subject to RdfTerm
fn subject_to_rdf_term(subject: &oxirs_core::model::Subject) -> crate::cli::formatters::RdfTerm {
    use crate::cli::formatters::RdfTerm;
    use oxirs_core::model::Subject;

    match subject {
        Subject::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Subject::BlankNode(bnode) => RdfTerm::Bnode {
            value: bnode.as_str().to_string(),
        },
        Subject::Variable(var) => {
            // Variables displayed as special literals
            RdfTerm::Literal {
                value: format!("?{}", var.name()),
                lang: None,
                datatype: None,
            }
        }
        Subject::QuotedTriple(triple) => {
            // RDF-star quoted triples displayed as special literals
            RdfTerm::Literal {
                value: format!(
                    "<<{} {} {}>>",
                    triple.subject(),
                    triple.predicate(),
                    triple.object()
                ),
                lang: None,
                datatype: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string()),
            }
        }
    }
}

/// Convert Predicate to RdfTerm
fn predicate_to_rdf_term(
    predicate: &oxirs_core::model::Predicate,
) -> crate::cli::formatters::RdfTerm {
    use crate::cli::formatters::RdfTerm;
    use oxirs_core::model::Predicate;

    match predicate {
        Predicate::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Predicate::Variable(var) => {
            // Variables displayed as special literals
            RdfTerm::Literal {
                value: format!("?{}", var.name()),
                lang: None,
                datatype: None,
            }
        }
    }
}

/// Convert Object to RdfTerm
fn object_to_rdf_term(object: &oxirs_core::model::Object) -> crate::cli::formatters::RdfTerm {
    use crate::cli::formatters::RdfTerm;
    use oxirs_core::model::Object;

    match object {
        Object::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Object::BlankNode(bnode) => RdfTerm::Bnode {
            value: bnode.as_str().to_string(),
        },
        Object::Literal(lit) => RdfTerm::Literal {
            value: lit.value().to_string(),
            lang: lit.language().map(|l| l.to_string()),
            datatype: Some(lit.datatype().as_str().to_string()),
        },
        Object::Variable(var) => {
            // Variables displayed as special literals
            RdfTerm::Literal {
                value: format!("?{}", var.name()),
                lang: None,
                datatype: None,
            }
        }
        Object::QuotedTriple(triple) => {
            // RDF-star quoted triples displayed as special literals
            RdfTerm::Literal {
                value: format!(
                    "<<{} {} {}>>",
                    triple.subject(),
                    triple.predicate(),
                    triple.object()
                ),
                lang: None,
                datatype: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string()),
            }
        }
    }
}
