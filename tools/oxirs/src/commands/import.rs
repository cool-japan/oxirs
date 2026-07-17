//! Data import command

use super::tdb_convert;
use super::CommandResult;
use crate::cli::error::helpers as error_helpers;
use crate::cli::logging::{DataLogger, PerfLogger};
use crate::cli::validation::MultiValidator;
use crate::cli::validation::{dataset_validation, fs_validation, validate_rdf_format};
use crate::cli::{progress::helpers, ArgumentValidator, Checkpoint, CheckpointManager, CliContext};
use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::{GraphName, NamedNode, Quad};
use oxirs_core::rdf_store::RdfStore;
use oxirs_tdb::TdbStore;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Supported values for `--dataset-type`.
const SUPPORTED_DATASET_TYPES: &[&str] = &["memory", "tdb2"];

/// Import RDF data into a dataset
///
/// `max_file_size` is the maximum accepted input file size in bytes; `0`
/// (or `None`) means unlimited.
///
/// `dataset_type` selects the storage backend: `"memory"` (default) uses the
/// in-RAM `RdfStore` (N-Quads-log persistence), preserving prior behavior;
/// `"tdb2"` bulk-loads directly into an on-disk `oxirs-tdb` `TdbStore`,
/// which keeps RAM use bounded regardless of dataset size.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    dataset: String,
    file: PathBuf,
    format: Option<String>,
    graph: Option<String>,
    resume: bool,
    max_file_size: Option<u64>,
    dataset_type: String,
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

    // Validate dataset type
    validator.add(
        ArgumentValidator::new("dataset_type", Some(&dataset_type))
            .required()
            .custom(
                |d| SUPPORTED_DATASET_TYPES.contains(&d),
                "Dataset type must be one of: memory, tdb2",
            ),
    );

    // Complete validation
    validator.finish()?;

    // Validate file size. `0` (or the CLI default) means unlimited so that
    // realistic production datasets (multi-GiB N-Quads/Turtle exports) are
    // not rejected outright; pass `--max-file-size` to cap it explicitly.
    let effective_max_size = max_file_size.filter(|&max| max > 0);
    fs_validation::validate_file_size(&file, effective_max_size)?;

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
    ctx.info(&format!("Backend: {dataset_type}"));

    // Detect format if not specified
    let detected_format = format.unwrap_or_else(|| detect_format(&file));
    ctx.info(&format!("Format: {detected_format}"));

    if let Some(g) = &graph {
        ctx.info(&format!("Target graph: {g}"));
    }

    // Initialize checkpoint manager
    let checkpoint_manager = CheckpointManager::new()
        .map_err(|e| format!("Failed to initialize checkpoint manager: {}", e))?;

    // Check for existing checkpoint if resume is enabled
    let mut processed_count = 0usize;

    if resume {
        if let Some(checkpoint) = checkpoint_manager
            .load("import", &dataset, file.to_str().unwrap_or(""))
            .map_err(|e| format!("Failed to load checkpoint: {}", e))?
        {
            ctx.info(&format!(
                "Found checkpoint from {}: {} triples processed ({:.1}% complete)",
                checkpoint.timestamp,
                checkpoint.processed_count,
                checkpoint_manager.progress_percentage(&checkpoint)
            ));

            processed_count = checkpoint.processed_count;

            ctx.info(&format!(
                "Resuming after {} already-imported statements",
                processed_count
            ));
        } else {
            ctx.info("No checkpoint found, starting fresh import");
        }
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

    // Start import with progress tracking and logging
    let start_time = Instant::now();
    ctx.info("Import Progress");

    // Initialize data logger
    let mut data_logger = DataLogger::new("import", &dataset);
    let mut perf_logger = PerfLogger::new(format!("import_{detected_format}"));
    perf_logger.add_metadata("file", file.display().to_string());
    perf_logger.add_metadata("format", &detected_format);
    perf_logger.add_metadata("dataset_type", &dataset_type);
    if let Some(ref g) = graph {
        perf_logger.add_metadata("graph", g);
    }

    // Get file size for progress bar
    let file_metadata = fs::metadata(&file)?;
    let file_size = file_metadata.len();

    // Create progress bar for file reading
    let read_progress = helpers::download_progress(file_size, &file.display().to_string());
    read_progress.set_message("Reading file");

    // Note: We cannot seek to arbitrary byte positions in RDF files
    // Instead, we'll parse from the beginning and skip already-processed triples

    read_progress.finish_with_message("File opened");
    data_logger.update_progress(file_size, 0);

    // Create progress spinner for parsing
    let parse_progress = helpers::query_progress();
    parse_progress.set_message("Parsing and importing RDF data");

    // Parse and import with progress and checkpointing. The backend
    // determines which on-disk store receives the bulk-inserted quads; the
    // parsing/chunking/checkpointing pipeline itself is shared (see
    // `parse_and_import_with_sink`).
    let (triple_count, error_count) = if dataset_type == "tdb2" {
        let mut store = TdbStore::open(&dataset_path).map_err(|e| {
            format!(
                "Failed to open tdb2 dataset at '{}': {e}",
                dataset_path.display()
            )
        })?;
        let file_handle = fs::File::open(&file)?;

        let result = parse_and_import_tdb2(
            &mut store,
            file_handle,
            &detected_format,
            graph.as_deref(),
            resume,
            &checkpoint_manager,
            &dataset,
            &file,
            file_size,
            processed_count,
        )?;

        // Make everything durable before reporting success, even though
        // each chunk already synced at its boundary.
        store
            .sync()
            .map_err(|e| format!("Failed to sync tdb2 store: {e}"))?;

        result
    } else {
        // Open store
        let mut store = if dataset_path.is_dir() {
            RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {e}"))?
        } else {
            return Err(error_helpers::dataset_not_found_error(&dataset));
        };
        let file_handle = fs::File::open(&file)?;

        let result = parse_and_import(
            &mut store,
            file_handle,
            &detected_format,
            graph.as_deref(),
            resume,
            &checkpoint_manager,
            &dataset,
            &file,
            file_size,
            processed_count,
        )?;

        // Make everything durable (fsync the append log / compact deletions)
        // before reporting success.
        store
            .flush()
            .map_err(|e| format!("Failed to flush store: {e}"))?;

        result
    };

    parse_progress.finish_with_message("Import complete");

    // Delete checkpoint on successful completion
    if resume {
        checkpoint_manager
            .delete("import", &dataset, file.to_str().unwrap_or(""))
            .map_err(|e| format!("Failed to delete checkpoint: {}", e))?;
        ctx.info("Checkpoint cleared after successful import");
    }

    let duration = start_time.elapsed();

    // Update data logger with final stats
    data_logger.update_progress(file_size, triple_count as u64);
    data_logger.complete();

    // Complete performance logging
    perf_logger.add_metadata("triple_count", triple_count);
    perf_logger.add_metadata("error_count", error_count);
    perf_logger.complete(Some(5000)); // Log if import takes more than 5 seconds

    // Report statistics with enhanced formatting
    use crate::cli::{format_bytes, format_duration, format_number};

    ctx.info("Import Statistics");
    ctx.success(&format!(
        "✓ Import completed in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Triples imported: {}",
        format_number(triple_count as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));

    if error_count > 0 {
        ctx.warn(&format!(
            "  Errors encountered: {}",
            format_number(error_count as u64)
        ));
    }

    if duration.as_secs_f64() > 0.0 {
        let rate = triple_count as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Import rate: {} triples/second",
            format_number(rate as u64)
        ));

        let throughput = file_size as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Throughput: {}/second",
            format_bytes(throughput as u64)
        ));
    }

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

/// Save an import checkpoint. Shared by both the memory and tdb2 flush
/// paths so resume semantics are identical at every chunk boundary
/// regardless of backend.
#[allow(clippy::too_many_arguments)]
fn save_import_checkpoint(
    checkpoint_manager: &CheckpointManager,
    checkpoint_count: usize,
    dataset: &str,
    file_path: &Path,
    format: &str,
    graph: Option<&str>,
    total_size: u64,
) {
    let checkpoint = Checkpoint {
        operation: "import".to_string(),
        dataset: dataset.to_string(),
        file_path: file_path.to_string_lossy().to_string(),
        processed_count: checkpoint_count,
        last_offset: 0, // Not used for RDF parsing
        timestamp: chrono::Local::now().to_rfc3339(),
        format: format.to_string(),
        graph: graph.map(|s| s.to_string()),
        total_size,
    };

    if let Err(e) = checkpoint_manager.save(&checkpoint) {
        eprintln!("Warning: Failed to save checkpoint: {e}");
    }
}

/// Insert every quad in `batch` in a single batched call (one lock
/// acquisition, one append + one `fsync` on the persistent backend) instead
/// of per-quad round trips, then save a checkpoint at this chunk boundary.
/// This is what keeps `oxirs import` from being O(N^2) on large files:
/// without batching, every single quad would pay its own lock/append
/// overhead.
#[allow(clippy::too_many_arguments)]
fn flush_pending_chunk_memory(
    store: &mut RdfStore,
    batch: Vec<Quad>,
    checkpoint_count: usize,
    enable_checkpointing: bool,
    checkpoint_manager: &CheckpointManager,
    dataset: &str,
    file_path: &Path,
    format: &str,
    graph: Option<&str>,
    total_size: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch.is_empty() {
        return Ok(());
    }

    store
        .bulk_insert_quads(batch)
        .map_err(|e| format!("Failed to bulk-insert quads: {e}"))?;

    if enable_checkpointing {
        save_import_checkpoint(
            checkpoint_manager,
            checkpoint_count,
            dataset,
            file_path,
            format,
            graph,
            total_size,
        );
    }

    Ok(())
}

/// Convert and insert every quad in `batch` into the on-disk tdb2 store,
/// then save a checkpoint at this chunk boundary. Default-graph quads are
/// routed through `insert_triples_bulk` (a single transactional batch,
/// self-syncing); named-graph quads go through `insert_quad` one at a time
/// (the quad API has no bulk variant yet), after which the store is synced
/// explicitly so the whole chunk — triples and quads alike — is durable
/// before the checkpoint is written.
#[allow(clippy::too_many_arguments)]
fn flush_pending_chunk_tdb2(
    store: &mut TdbStore,
    batch: Vec<Quad>,
    checkpoint_count: usize,
    enable_checkpointing: bool,
    checkpoint_manager: &CheckpointManager,
    dataset: &str,
    file_path: &Path,
    format: &str,
    graph: Option<&str>,
    total_size: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch.is_empty() {
        return Ok(());
    }

    let mut default_graph_triples = Vec::with_capacity(batch.len());
    for quad in &batch {
        let subject = tdb_convert::subject_to_tdb_term(quad.subject())
            .map_err(|e| format!("Invalid subject term: {e}"))?;
        let predicate = tdb_convert::predicate_to_tdb_term(quad.predicate())
            .map_err(|e| format!("Invalid predicate term: {e}"))?;
        let object = tdb_convert::object_to_tdb_term(quad.object())
            .map_err(|e| format!("Invalid object term: {e}"))?;
        let graph_term = tdb_convert::graph_name_to_tdb_term(quad.graph_name())
            .map_err(|e| format!("Invalid graph name: {e}"))?;

        match graph_term {
            None => default_graph_triples.push((subject, predicate, object)),
            Some(graph_term) => {
                store
                    .insert_quad(Some(&graph_term), &subject, &predicate, &object)
                    .map_err(|e| {
                        format!("Failed to insert named-graph quad into tdb2 store: {e}")
                    })?;
            }
        }
    }

    if !default_graph_triples.is_empty() {
        store
            .insert_triples_bulk(&default_graph_triples)
            .map_err(|e| format!("Failed to bulk-insert triples into tdb2 store: {e}"))?;
    }

    // `insert_quad` (named-graph path) does not sync on its own, so make the
    // whole chunk durable before the checkpoint records it as processed.
    store
        .sync()
        .map_err(|e| format!("Failed to sync tdb2 store: {e}"))?;

    if enable_checkpointing {
        save_import_checkpoint(
            checkpoint_manager,
            checkpoint_count,
            dataset,
            file_path,
            format,
            graph,
            total_size,
        );
    }

    Ok(())
}

/// Parse an RDF file into chunks of up to `CHUNK_SIZE` quads and hand each
/// chunk (plus the running "quads parsed so far" count, used as the
/// checkpoint's `processed_count`) to `sink`. Shared by every storage
/// backend so the parsing, target-graph rewriting, resume-skip, and
/// chunk-size behavior are identical regardless of which store ultimately
/// receives the data.
fn parse_and_import_with_sink<F>(
    file: fs::File,
    format: &str,
    graph: Option<&str>,
    start_count: usize,
    mut sink: F,
) -> Result<(usize, usize), Box<dyn std::error::Error>>
where
    F: FnMut(Vec<Quad>, usize) -> Result<(), Box<dyn std::error::Error>>,
{
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

    let mut total_parsed = 0; // Total triples parsed (including skipped ones)
    let mut error_count = 0;

    // Accumulate parsed quads into chunks and bulk-insert them, instead of
    // one store round trip per quad (which is O(N^2) on a persistent store
    // that must lock + append on every single insert).
    const CHUNK_SIZE: usize = 10_000;
    let mut pending: Vec<Quad> = Vec::with_capacity(CHUNK_SIZE);

    // Step 4: Parse and accumulate quads
    for quad_result in parser.for_reader(reader) {
        match quad_result {
            Ok(mut quad) => {
                total_parsed += 1;

                // Skip already-processed triples when resuming
                if total_parsed <= start_count {
                    continue;
                }

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

                pending.push(quad);

                // Flush + checkpoint at each chunk boundary.
                if pending.len() >= CHUNK_SIZE {
                    let batch = std::mem::replace(&mut pending, Vec::with_capacity(CHUNK_SIZE));
                    sink(batch, total_parsed)?;
                }
            }
            Err(e) => {
                eprintln!("Warning: Parse error: {e}");
                error_count += 1;
            }
        }
    }

    // Flush the final, possibly partial, chunk.
    if !pending.is_empty() {
        sink(pending, total_parsed)?;
    }

    Ok((total_parsed, error_count))
}

/// Parse RDF content and import into an in-memory `RdfStore`.
#[allow(clippy::too_many_arguments)]
fn parse_and_import(
    store: &mut RdfStore,
    file: fs::File,
    format: &str,
    graph: Option<&str>,
    enable_checkpointing: bool,
    checkpoint_manager: &CheckpointManager,
    dataset: &str,
    file_path: &Path,
    total_size: u64,
    start_count: usize,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    parse_and_import_with_sink(
        file,
        format,
        graph,
        start_count,
        |batch, checkpoint_count| {
            flush_pending_chunk_memory(
                store,
                batch,
                checkpoint_count,
                enable_checkpointing,
                checkpoint_manager,
                dataset,
                file_path,
                format,
                graph,
                total_size,
            )
        },
    )
}

/// Parse RDF content and bulk-load it into an on-disk `oxirs-tdb` `TdbStore`,
/// keeping RAM use bounded to one `CHUNK_SIZE`-sized batch at a time
/// regardless of total dataset size.
#[allow(clippy::too_many_arguments)]
fn parse_and_import_tdb2(
    store: &mut TdbStore,
    file: fs::File,
    format: &str,
    graph: Option<&str>,
    enable_checkpointing: bool,
    checkpoint_manager: &CheckpointManager,
    dataset: &str,
    file_path: &Path,
    total_size: u64,
    start_count: usize,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    parse_and_import_with_sink(
        file,
        format,
        graph,
        start_count,
        |batch, checkpoint_count| {
            flush_pending_chunk_tdb2(
                store,
                batch,
                checkpoint_count,
                enable_checkpointing,
                checkpoint_manager,
                dataset,
                file_path,
                format,
                graph,
                total_size,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the O(N^2) import fix: `parse_and_import` must
    /// accumulate parsed quads into chunks and call `bulk_insert_quads`
    /// rather than round-tripping the store once per quad. This exercises
    /// the real bulk-insert path end to end (parse -> chunk -> bulk insert
    /// -> flush) and confirms every quad lands in the store.
    #[test]
    fn test_parse_and_import_bulk_inserts_all_quads() {
        let unique = std::process::id() as u64 * 1_000_003 + line!() as u64;
        let dataset_dir = std::env::temp_dir().join(format!("oxirs-import-bulk-test-{unique}"));
        let src_dir = std::env::temp_dir().join(format!("oxirs-import-bulk-src-{unique}"));
        std::fs::create_dir_all(&dataset_dir).expect("create temp dataset dir");
        std::fs::create_dir_all(&src_dir).expect("create temp source dir");

        // More than a handful of quads, though well under CHUNK_SIZE, so the
        // final (partial-chunk) flush path is exercised.
        let mut content = String::new();
        for i in 0..25 {
            content.push_str(&format!(
                "<http://ex.org/s{i}> <http://ex.org/p> \"v{i}\" .\n"
            ));
        }
        let nt_file = src_dir.join("data.nt");
        std::fs::write(&nt_file, content).expect("write nt file");

        let mut store = RdfStore::open(&dataset_dir).expect("open store");
        let checkpoint_manager = CheckpointManager::new().expect("create checkpoint manager");
        let file_handle = fs::File::open(&nt_file).expect("open nt file");

        let (parsed, errors) = parse_and_import(
            &mut store,
            file_handle,
            "ntriples",
            None,
            false, // disable checkpointing: avoid touching the real user config dir in a test
            &checkpoint_manager,
            "test-dataset",
            &nt_file,
            0,
            0,
        )
        .expect("parse_and_import should succeed");

        assert_eq!(parsed, 25);
        assert_eq!(errors, 0);
        assert_eq!(
            store.len().expect("store len"),
            25,
            "all 25 quads should have been bulk-inserted into the store"
        );

        std::fs::remove_dir_all(&dataset_dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }

    #[test]
    fn test_max_file_size_zero_means_unlimited() {
        // A `0` (or the CLI's own "unlimited" sentinel) must not be treated
        // as an active byte cap.
        let max_file_size: Option<u64> = Some(0);
        let effective = max_file_size.filter(|&max| max > 0);
        assert_eq!(effective, None);

        let max_file_size: Option<u64> = Some(500);
        let effective = max_file_size.filter(|&max| max > 0);
        assert_eq!(effective, Some(500));
    }

    /// The tdb2 backend must bulk-load default-graph quads (via
    /// `insert_triples_bulk`) and durably persist them: reopening the store
    /// after `parse_and_import_tdb2` returns must see every triple, exactly
    /// mirroring the memory-backend regression test above.
    #[test]
    fn test_parse_and_import_tdb2_bulk_loads_default_graph() {
        let unique = std::process::id() as u64 * 1_000_003 + line!() as u64;
        let dataset_dir = std::env::temp_dir().join(format!("oxirs-import-tdb2-test-{unique}"));
        let src_dir = std::env::temp_dir().join(format!("oxirs-import-tdb2-src-{unique}"));
        std::fs::create_dir_all(&src_dir).expect("create temp source dir");

        let mut content = String::new();
        for i in 0..25 {
            content.push_str(&format!(
                "<http://ex.org/s{i}> <http://ex.org/p> \"v{i}\" .\n"
            ));
        }
        let nt_file = src_dir.join("data.nt");
        std::fs::write(&nt_file, content).expect("write nt file");

        {
            let mut store = TdbStore::open(&dataset_dir).expect("open tdb2 store");
            let checkpoint_manager = CheckpointManager::new().expect("create checkpoint manager");
            let file_handle = fs::File::open(&nt_file).expect("open nt file");

            let (parsed, errors) = parse_and_import_tdb2(
                &mut store,
                file_handle,
                "ntriples",
                None,
                false,
                &checkpoint_manager,
                "test-dataset",
                &nt_file,
                0,
                0,
            )
            .expect("parse_and_import_tdb2 should succeed");

            assert_eq!(parsed, 25);
            assert_eq!(errors, 0);
            store.sync().expect("sync tdb2 store");
        }

        // Reopen from disk to confirm durability, not just an in-memory view.
        let reopened = TdbStore::open(&dataset_dir).expect("reopen tdb2 store");
        let triples = reopened
            .query_triples(None, None, None)
            .expect("query triples");
        assert_eq!(
            triples.len(),
            25,
            "all 25 triples should have survived a close + reopen of the tdb2 store"
        );

        std::fs::remove_dir_all(&dataset_dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }

    /// Quads carrying a named graph must land in the tdb2 store's quad
    /// indexes (not silently dropped into the default graph), and must
    /// round-trip through a reopen.
    #[test]
    fn test_parse_and_import_tdb2_named_graph_quads() {
        let unique = std::process::id() as u64 * 1_000_003 + line!() as u64;
        let dataset_dir =
            std::env::temp_dir().join(format!("oxirs-import-tdb2-graph-test-{unique}"));
        let src_dir = std::env::temp_dir().join(format!("oxirs-import-tdb2-graph-src-{unique}"));
        std::fs::create_dir_all(&src_dir).expect("create temp source dir");

        let content = "<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> <http://ex.org/g> .\n\
             <http://ex.org/s2> <http://ex.org/p> <http://ex.org/o2> .\n";
        let nq_file = src_dir.join("data.nq");
        std::fs::write(&nq_file, content).expect("write nq file");

        {
            let mut store = TdbStore::open(&dataset_dir).expect("open tdb2 store");
            let checkpoint_manager = CheckpointManager::new().expect("create checkpoint manager");
            let file_handle = fs::File::open(&nq_file).expect("open nq file");

            let (parsed, errors) = parse_and_import_tdb2(
                &mut store,
                file_handle,
                "nquads",
                None,
                false,
                &checkpoint_manager,
                "test-dataset",
                &nq_file,
                0,
                0,
            )
            .expect("parse_and_import_tdb2 should succeed");

            assert_eq!(parsed, 2);
            assert_eq!(errors, 0);
            store.sync().expect("sync tdb2 store");
        }

        let reopened = TdbStore::open(&dataset_dir).expect("reopen tdb2 store");
        // One default-graph triple.
        let triples = reopened
            .query_triples(None, None, None)
            .expect("query triples");
        assert_eq!(triples.len(), 1, "default-graph triple should be present");
        // One named-graph quad.
        assert_eq!(
            reopened.quad_count(),
            1,
            "named-graph quad should be present in the quad indexes"
        );

        std::fs::remove_dir_all(&dataset_dir).ok();
        std::fs::remove_dir_all(&src_dir).ok();
    }
}
