//! Batch operations for high-performance data processing

use super::CommandResult;
use crate::cli::error::helpers as error_helpers;
use crate::cli::logging::{DataLogger, PerfLogger};
use crate::cli::validation::{dataset_validation, fs_validation, validate_rdf_format};
use crate::cli::{progress::helpers, CliContext};
use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::{GraphName, NamedNode};
use oxirs_core::rdf_store::RdfStore;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Batch import multiple RDF files into a dataset with parallel processing
pub async fn import_batch(
    dataset: String,
    files: Vec<PathBuf>,
    format: Option<String>,
    graph: Option<String>,
    parallel: usize,
) -> CommandResult {
    // Create CLI context for proper output formatting
    let ctx = CliContext::new();

    // Validate dataset
    dataset_validation::validate_dataset_name(&dataset)?;

    // Validate graph if specified
    if let Some(ref g) = graph {
        dataset_validation::validate_graph_uri(g)?;
    }

    // Validate all files exist
    for file in &files {
        if !file.exists() {
            return Err(format!("File not found: {}", file.display()).into());
        }
        if !file.is_file() {
            return Err(format!("Not a file: {}", file.display()).into());
        }
        // Validate file size (limit to 1GB per file)
        fs_validation::validate_file_size(file, Some(1_073_741_824))?;
    }

    ctx.info(&format!(
        "Batch importing {} files into dataset '{dataset}'",
        files.len()
    ));
    if let Some(ref fmt) = format {
        validate_rdf_format(fmt)?;
        ctx.info(&format!("Format: {fmt}"));
    } else {
        ctx.info("Format: auto-detect from file extensions");
    }
    ctx.info(&format!("Parallel workers: {parallel}"));

    if let Some(g) = &graph {
        ctx.info(&format!("Target graph: {g}"));
    }

    // Load dataset configuration or use dataset path directly
    let dataset_path = if PathBuf::from(&dataset).join("oxirs.toml").exists() {
        crate::config::load_dataset_from_config(&dataset)?
    } else {
        PathBuf::from(&dataset)
    };

    // Open store
    let store = if dataset_path.is_dir() {
        RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {e}"))?
    } else {
        return Err(error_helpers::dataset_not_found_error(&dataset));
    };

    // Wrap store in Arc for thread-safe sharing
    let store = Arc::new(Mutex::new(store));

    // Start batch import
    let start_time = Instant::now();
    ctx.info("Batch Import Progress");

    // Initialize loggers
    let mut data_logger = DataLogger::new("batch_import", &dataset);
    let mut perf_logger = PerfLogger::new(format!("batch_import_{}_files", files.len()));
    perf_logger.add_metadata("file_count", files.len());
    perf_logger.add_metadata("parallel_workers", parallel);

    // Track total progress
    let total_files = files.len();
    let completed_files = Arc::new(Mutex::new(0usize));
    let total_quads = Arc::new(Mutex::new(0usize));
    let total_errors = Arc::new(Mutex::new(0usize));

    // Create progress bar
    let progress = helpers::query_progress();
    progress.set_message("Processing files in parallel");

    // Process files with controlled parallelism
    let chunk_size = parallel.max(1);
    for chunk in files.chunks(chunk_size) {
        // Process chunk of files concurrently
        let mut handles = vec![];

        for file in chunk {
            let file = file.clone();
            let store_clone = Arc::clone(&store);
            let format_clone = format.clone();
            let graph_clone = graph.clone();
            let completed_clone = Arc::clone(&completed_files);
            let quads_clone = Arc::clone(&total_quads);
            let errors_clone = Arc::clone(&total_errors);

            let handle = tokio::spawn(async move {
                process_single_file(
                    store_clone,
                    file,
                    format_clone,
                    graph_clone,
                    completed_clone,
                    quads_clone,
                    errors_clone,
                )
                .await
            });

            handles.push(handle);
        }

        // Wait for all files in chunk to complete
        for handle in handles {
            if let Err(e) = handle.await {
                eprintln!("Warning: Task failed: {e}");
            }
        }
    }

    progress.finish_with_message("Batch import complete");

    let duration = start_time.elapsed();

    // Get final statistics
    let completed = *completed_files.lock().unwrap();
    let quad_count = *total_quads.lock().unwrap();
    let error_count = *total_errors.lock().unwrap();

    // Update loggers
    data_logger.update_progress(0, quad_count as u64);
    data_logger.complete();
    perf_logger.add_metadata("total_quads", quad_count);
    perf_logger.add_metadata("total_errors", error_count);
    perf_logger.add_metadata("completed_files", completed);
    perf_logger.complete(Some(10000)); // Log if batch takes more than 10 seconds

    // Report statistics
    ctx.info("Batch Import Statistics");
    ctx.success(&format!(
        "Batch import completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Files processed: {completed}/{total_files}"));
    ctx.info(&format!("Total quads imported: {quad_count}"));

    if error_count > 0 {
        ctx.warn(&format!("Total errors encountered: {error_count}"));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        quad_count as f64 / duration.as_secs_f64()
    ));
    ctx.info(&format!(
        "Average per file: {:.0} quads",
        quad_count as f64 / completed as f64
    ));

    Ok(())
}

/// Process a single file in the batch
async fn process_single_file(
    store: Arc<Mutex<RdfStore>>,
    file: PathBuf,
    format: Option<String>,
    graph: Option<String>,
    completed_files: Arc<Mutex<usize>>,
    total_quads: Arc<Mutex<usize>>,
    total_errors: Arc<Mutex<usize>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Detect format if not specified
    let detected_format = format.unwrap_or_else(|| detect_format(&file));

    // Determine RDF format
    let rdf_format = match detected_format.as_str() {
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
            eprintln!(
                "Warning: Unsupported format '{}' for file: {}",
                detected_format,
                file.display()
            );
            return Ok(());
        }
    };

    // Determine target graph
    let target_graph = if let Some(graph_iri) = graph {
        if graph_iri == "default" {
            GraphName::DefaultGraph
        } else {
            GraphName::NamedNode(
                NamedNode::new(&graph_iri).map_err(|e| format!("Invalid graph IRI: {e}"))?,
            )
        }
    } else {
        GraphName::DefaultGraph
    };

    // Open and parse file
    let file_handle = fs::File::open(&file)?;
    let reader = BufReader::new(file_handle);
    let parser = RdfParser::new(rdf_format);

    let mut file_quad_count = 0;
    let mut file_error_count = 0;

    // Parse and insert quads
    for quad_result in parser.for_reader(reader) {
        match quad_result {
            Ok(mut quad) => {
                // Move to target graph if specified
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

                // Insert quad into store (with lock)
                let mut store_lock = store.lock().unwrap();
                match store_lock.insert_quad(quad) {
                    Ok(_) => {
                        file_quad_count += 1;
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to insert quad from {}: {e}",
                            file.display()
                        );
                        file_error_count += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Parse error in {}: {e}", file.display());
                file_error_count += 1;
            }
        }
    }

    // Update global counters
    {
        let mut completed = completed_files.lock().unwrap();
        *completed += 1;
    }
    {
        let mut quads = total_quads.lock().unwrap();
        *quads += file_quad_count;
    }
    {
        let mut errors = total_errors.lock().unwrap();
        *errors += file_error_count;
    }

    Ok(())
}

/// Detect RDF format from file extension
fn detect_format(file: &std::path::Path) -> String {
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
