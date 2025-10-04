//! Data import command

use super::CommandResult;
use crate::cli::error::helpers as error_helpers;
use crate::cli::logging::{DataLogger, PerfLogger};
use crate::cli::validation::MultiValidator;
use crate::cli::validation::{dataset_validation, fs_validation, validate_rdf_format};
use crate::cli::{progress::helpers, ArgumentValidator, CliContext};
use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::{GraphName, NamedNode};
use oxirs_core::rdf_store::RdfStore;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Import RDF data into a dataset
pub async fn run(
    dataset: String,
    file: PathBuf,
    format: Option<String>,
    graph: Option<String>,
) -> CommandResult {
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

    // Validate input file
    validator.add(
        ArgumentValidator::new("file", Some(file.to_str().unwrap_or("")))
            .required()
            .is_file(),
    );

    // Complete validation
    validator.finish()?;

    // Validate file size (limit to 1GB for now)
    fs_validation::validate_file_size(&file, Some(1_073_741_824))?;

    // Validate format if specified
    if let Some(ref fmt) = format {
        validate_rdf_format(fmt)?;
    }

    // Validate graph URI if specified
    if let Some(ref g) = graph {
        dataset_validation::validate_graph_uri(g)?;
    }

    ctx.info(&format!("Importing data into dataset '{dataset}'"));
    ctx.info(&format!("Source file: {}", file.display()));

    // Detect format if not specified
    let detected_format = format.unwrap_or_else(|| detect_format(&file));
    ctx.info(&format!("Format: {detected_format}"));

    if let Some(g) = &graph {
        ctx.info(&format!("Target graph: {g}"));
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

    // Start import with progress tracking and logging
    let start_time = Instant::now();
    ctx.info("Import Progress");

    // Initialize data logger
    let mut data_logger = DataLogger::new("import", &dataset);
    let mut perf_logger = PerfLogger::new(format!("import_{detected_format}"));
    perf_logger.add_metadata("file", file.display().to_string());
    perf_logger.add_metadata("format", &detected_format);
    if let Some(ref g) = graph {
        perf_logger.add_metadata("graph", g);
    }

    // Get file size for progress bar
    let file_metadata = fs::metadata(&file)?;
    let file_size = file_metadata.len();

    // Create progress bar for file reading
    let read_progress = helpers::download_progress(file_size, &file.display().to_string());
    read_progress.set_message("Reading file");

    // Open file for reading
    let file_handle = fs::File::open(&file)?;
    read_progress.finish_with_message("File opened");
    data_logger.update_progress(file_size, 0);

    // Create progress spinner for parsing
    let parse_progress = helpers::query_progress();
    parse_progress.set_message("Parsing and importing RDF data");

    // Parse and import with progress
    let (triple_count, error_count) =
        parse_and_import(&store, file_handle, &detected_format, graph.as_deref())?;

    parse_progress.finish_with_message("Import complete");

    let duration = start_time.elapsed();

    // Update data logger with final stats
    data_logger.update_progress(file_size, triple_count as u64);
    data_logger.complete();

    // Complete performance logging
    perf_logger.add_metadata("triple_count", triple_count);
    perf_logger.add_metadata("error_count", error_count);
    perf_logger.complete(Some(5000)); // Log if import takes more than 5 seconds

    // Report statistics with formatted output
    ctx.info("Import Statistics");
    ctx.success(&format!(
        "Import completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Triples imported: {triple_count}"));

    if error_count > 0 {
        ctx.warn(&format!("Errors encountered: {error_count}"));
    }

    ctx.info(&format!(
        "Average rate: {:.0} triples/second",
        triple_count as f64 / duration.as_secs_f64()
    ));

    Ok(())
}

/// Detect RDF format from file extension
fn detect_format(file: &Path) -> String {
    if let Some(ext) = file.extension().and_then(|s| s.to_str()) {
        match ext.to_lowercase().as_str() {
            "ttl" | "turtle" => "turtle".to_string(),
            "nt" | "ntriples" => "ntriples".to_string(),
            "rdf" | "xml" => "rdfxml".to_string(),
            "jsonld" | "json-ld" => "jsonld".to_string(),
            "trig" => "trig".to_string(),
            "nq" | "nquads" => "nquads".to_string(),
            _ => "turtle".to_string(), // Default fallback
        }
    } else {
        "turtle".to_string() // Default fallback
    }
}

/// Check if format is supported
#[allow(dead_code)]
fn is_supported_format(format: &str) -> bool {
    matches!(
        format,
        "turtle" | "ntriples" | "rdfxml" | "jsonld" | "trig" | "nquads"
    )
}

/// Parse RDF content and import into store
fn parse_and_import<S: oxirs_core::Store>(
    store: &S,
    file: fs::File,
    format: &str,
    graph: Option<&str>,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    // Step 1: Determine RDF format from string
    let rdf_format = match format {
        "turtle" | "ttl" => RdfFormat::Turtle,
        "ntriples" | "nt" => RdfFormat::NTriples,
        "nquads" | "nq" => RdfFormat::NQuads,
        "trig" => RdfFormat::TriG,
        "rdfxml" | "rdf" | "xml" => RdfFormat::RdfXml,
        "jsonld" | "json-ld" | "json" => RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        },
        "n3" => RdfFormat::N3,
        _ => {
            return Err(format!("Unsupported import format: {format}").into());
        }
    };

    // Step 2: Determine target graph
    let target_graph = if let Some(graph_iri) = graph {
        if graph_iri == "default" {
            GraphName::DefaultGraph
        } else {
            GraphName::NamedNode(
                NamedNode::new(graph_iri).map_err(|e| format!("Invalid graph IRI: {e}"))?,
            )
        }
    } else {
        GraphName::DefaultGraph
    };

    // Step 3: Parse RDF file
    let reader = BufReader::new(file);
    let parser = RdfParser::new(rdf_format);

    let mut triple_count = 0;
    let mut error_count = 0;

    // Step 4: Parse and insert quads
    for quad_result in parser.for_reader(reader) {
        match quad_result {
            Ok(mut quad) => {
                // If a target graph is specified and the quad is in the default graph,
                // move it to the target graph
                if matches!(target_graph, GraphName::NamedNode(_))
                    && matches!(quad.graph_name(), GraphName::DefaultGraph)
                {
                    quad = oxirs_core::model::Quad::new(
                        quad.subject().clone(),
                        quad.predicate().clone(),
                        quad.object().clone(),
                        target_graph.clone(),
                    );
                }

                // Insert quad into store
                match store.insert_quad(quad) {
                    Ok(_) => {
                        triple_count += 1;
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to insert quad: {e}");
                        error_count += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Parse error: {e}");
                error_count += 1;
            }
        }
    }

    Ok((triple_count, error_count))
}
