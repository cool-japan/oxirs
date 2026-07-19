//! Data export command

use super::tdb_convert;
use super::CommandResult;
use crate::cli::{Checkpoint, CheckpointManager};
use oxirs_core::format::{RdfFormat, RdfSerializer};
use oxirs_core::rdf_store::RdfStore;
use oxirs_tdb::{GraphTarget, TdbStore};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Export RDF data from a dataset
pub async fn run(
    dataset: String,
    file: PathBuf,
    format: String,
    graph: Option<String>,
    resume: bool,
) -> CommandResult {
    println!(
        "Exporting data from dataset '{}' to {}",
        dataset,
        file.display()
    );
    println!("Output format: {format}");

    if let Some(g) = &graph {
        println!("Source graph: {g}");
    }

    // Validate format
    if !is_supported_export_format(&format) {
        return Err(format!(
            "Unsupported export format '{format}'. Supported formats: turtle, ntriples, rdfxml, jsonld, trig, nquads"
        ).into());
    }

    // Initialize checkpoint manager
    let checkpoint_manager = CheckpointManager::new()
        .map_err(|e| format!("Failed to initialize checkpoint manager: {}", e))?;

    // Check for existing checkpoint if resume is enabled
    let mut start_count = 0usize;

    if resume {
        if let Some(checkpoint) = checkpoint_manager
            .load("export", &dataset, file.to_str().unwrap_or(""))
            .map_err(|e| format!("Failed to load checkpoint: {}", e))?
        {
            println!(
                "Found checkpoint from {}: {} triples exported",
                checkpoint.timestamp, checkpoint.processed_count
            );

            start_count = checkpoint.processed_count;
            println!("Resuming from triple {}", start_count);
        } else {
            println!("No checkpoint found, starting fresh export");
        }
    }

    // Check if output file already exists (allow if resuming)
    if file.exists() && !resume {
        return Err(format!("Output file '{}' already exists", file.display()).into());
    }

    // Ensure output directory exists
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent)?;
    }

    // Load dataset configuration or use dataset path directly
    let dataset_path = if PathBuf::from(&dataset).join("oxirs.toml").exists() {
        // Dataset with configuration file
        load_dataset_from_config(&dataset)?
    } else {
        // Assume dataset is a directory path
        PathBuf::from(&dataset)
    };

    if !dataset_path.is_dir() {
        return Err(format!(
            "Dataset '{dataset}' not found. Use 'oxirs init' to create a dataset."
        )
        .into());
    }

    // Start export
    let start_time = Instant::now();
    println!("Starting export...");

    // Detect the on-disk backend (see `tdb_convert::is_tdb2_dataset`) and
    // read through the matching store. The tdb2 path streams quads directly
    // off disk via `TdbStore::quad_iter` instead of materializing the whole
    // dataset into an `RdfStore` first.
    let triple_count = if tdb_convert::is_tdb2_dataset(&dataset_path) {
        let store = TdbStore::open(&dataset_path)
            .map_err(|e| format!("Failed to open tdb2 dataset: {e}"))?;
        export_data_tdb2(
            &store,
            &file,
            &format,
            graph.as_deref(),
            resume,
            &checkpoint_manager,
            &dataset,
            start_count,
        )?
    } else {
        let store =
            RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {e}"))?;
        export_data(
            &store,
            &file,
            &format,
            graph.as_deref(),
            resume,
            &checkpoint_manager,
            &dataset,
            start_count,
        )?
    };

    let duration = start_time.elapsed();

    // Delete checkpoint on successful completion
    if resume {
        checkpoint_manager
            .delete("export", &dataset, file.to_str().unwrap_or(""))
            .map_err(|e| format!("Failed to delete checkpoint: {}", e))?;
        println!("Checkpoint cleared after successful export");
    }

    // Report statistics with enhanced formatting
    use crate::cli::{file_size, format_bytes, format_duration, format_number};

    let file_size_bytes = file_size(&file).unwrap_or(0);

    println!("\n✓ Export completed successfully!");
    println!("  Duration: {}", format_duration(duration));
    println!("  Triples exported: {}", format_number(triple_count as u64));
    println!("  Output file: {}", file.display());
    println!("  File size: {}", format_bytes(file_size_bytes));

    if duration.as_secs_f64() > 0.0 {
        let rate = triple_count as f64 / duration.as_secs_f64();
        println!(
            "  Export rate: {} triples/second",
            format_number(rate as u64)
        );
    }

    Ok(())
}

/// Check if export format is supported
fn is_supported_export_format(format: &str) -> bool {
    matches!(
        format,
        "turtle" | "ntriples" | "rdfxml" | "jsonld" | "trig" | "nquads"
    )
}

/// Load dataset configuration from oxirs.toml file
fn load_dataset_from_config(dataset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Use shared configuration loader with full TOML parsing
    crate::config::load_dataset_from_config(dataset)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Export data from store to file
#[allow(clippy::too_many_arguments)]
fn export_data(
    store: &RdfStore,
    file: &PathBuf,
    format: &str,
    graph: Option<&str>,
    enable_checkpointing: bool,
    checkpoint_manager: &CheckpointManager,
    dataset: &str,
    start_count: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Step 1: Determine RDF format from string
    let rdf_format = match format {
        "turtle" | "ttl" => RdfFormat::Turtle,
        "ntriples" | "nt" => RdfFormat::NTriples,
        "nquads" | "nq" => RdfFormat::NQuads,
        "trig" => RdfFormat::TriG,
        "rdfxml" | "rdf" | "xml" => RdfFormat::RdfXml,
        "jsonld" | "json" => RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        },
        "n3" => RdfFormat::N3,
        _ => {
            return Err(format!("Unsupported export format: {format}").into());
        }
    };

    // Step 2: Query all quads from the store.
    //
    // NOTE (streaming): `RdfStore::quads()` currently returns a fully
    // materialized `Vec<Quad>` — there is no lazy/streaming quad cursor
    // exposed by the store layer yet (the store internals that would need
    // to change, `rdf_store/mod.rs` and `store/mmap_store/**`, are owned by
    // a different work stream). Once a streaming scan API lands there,
    // this should be rewritten to serialize quads directly from that
    // iterator instead of buffering the whole dataset here. Until then we
    // at least avoid a second full-dataset allocation by filtering the
    // graph in place with `retain` rather than collecting into a new Vec.
    println!("   [1/3] Querying triples from store...");
    let mut quads = store
        .quads()
        .map_err(|e| format!("Failed to query quads: {e}"))?;

    if let Some(graph_name) = graph {
        // Filter by specific graph
        use oxirs_core::model::{GraphName, NamedNode};
        let graph_filter = if graph_name == "default" {
            GraphName::DefaultGraph
        } else {
            GraphName::NamedNode(
                NamedNode::new(graph_name).map_err(|e| format!("Invalid graph IRI: {e}"))?,
            )
        };

        quads.retain(|quad| quad.graph_name() == &graph_filter);
    }

    let quad_count = quads.len();
    println!("       ✓ Retrieved {quad_count} quads from store");

    // Step 3: Serialize quads to output format
    println!("   [2/3] Serializing to {format} format...");

    // Open file for writing (create new or append if resuming)
    let output_file = if start_count > 0 {
        fs::OpenOptions::new().append(true).open(file)?
    } else {
        fs::File::create(file)?
    };

    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("owl", "http://www.w3.org/2002/07/owl#")
        .pretty()
        .for_writer(output_file);

    // Checkpoint every 10,000 triples
    const CHECKPOINT_INTERVAL: usize = 10_000;

    for (idx, quad) in quads.iter().enumerate() {
        // Skip already-exported triples when resuming
        if idx < start_count {
            continue;
        }

        serializer
            .serialize_quad(quad.as_ref())
            .map_err(|e| format!("Serialization error: {e}"))?;

        let exported_count = idx + 1;

        // Save checkpoint periodically if checkpointing is enabled
        if enable_checkpointing && exported_count % CHECKPOINT_INTERVAL == 0 {
            let checkpoint = Checkpoint {
                operation: "export".to_string(),
                dataset: dataset.to_string(),
                file_path: file.to_string_lossy().to_string(),
                processed_count: exported_count,
                last_offset: 0, // Not used for export
                timestamp: chrono::Local::now().to_rfc3339(),
                format: format.to_string(),
                graph: graph.map(|s| s.to_string()),
                total_size: quad_count as u64,
            };

            if let Err(e) = checkpoint_manager.save(&checkpoint) {
                eprintln!("Warning: Failed to save checkpoint: {e}");
            }
        }
    }

    serializer
        .finish()
        .map_err(|e| format!("Failed to finalize serialization: {e}"))?;

    println!("       ✓ Serialization completed");

    // Step 4: Write and report
    println!("   [3/3] Writing to file...");
    println!("       ✓ Data written to {}", file.display());

    Ok(quad_count)
}

/// Export data from an on-disk tdb2 `TdbStore` to a file.
///
/// Unlike [`export_data`] (which must materialize the whole `RdfStore` up
/// front — see the NOTE there), this streams quads directly off disk via
/// [`TdbStore::quad_iter`], never buffering more than the current quad in
/// memory. This is the path that keeps `oxirs export` RAM-bounded for large
/// tdb2 datasets (the same bulk-load use case `oxirs import --dataset-type
/// tdb2` targets).
#[allow(clippy::too_many_arguments)]
fn export_data_tdb2(
    store: &TdbStore,
    file: &PathBuf,
    format: &str,
    graph: Option<&str>,
    enable_checkpointing: bool,
    checkpoint_manager: &CheckpointManager,
    dataset: &str,
    start_count: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    use oxirs_tdb::dictionary::Term as TdbTerm;

    // Step 1: Determine RDF format from string
    let rdf_format = match format {
        "turtle" | "ttl" => RdfFormat::Turtle,
        "ntriples" | "nt" => RdfFormat::NTriples,
        "nquads" | "nq" => RdfFormat::NQuads,
        "trig" => RdfFormat::TriG,
        "rdfxml" | "rdf" | "xml" => RdfFormat::RdfXml,
        "jsonld" | "json" => RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        },
        "n3" => RdfFormat::N3,
        _ => {
            return Err(format!("Unsupported export format: {format}").into());
        }
    };

    println!("   [1/3] Streaming quads from tdb2 store...");

    // Step 2: Resolve the graph filter into a `GraphTarget`. `graph_term`
    // must outlive `target` since `GraphTarget::Named` borrows it.
    let graph_term: Option<TdbTerm> = match graph {
        None | Some("default") => None,
        Some(iri) => Some(TdbTerm::Iri(iri.to_string())),
    };
    let target = match graph {
        None => GraphTarget::AnyGraph,
        Some("default") => GraphTarget::DefaultGraph,
        Some(_) => GraphTarget::Named(
            graph_term
                .as_ref()
                .expect("graph_term is Some for a named-graph filter"),
        ),
    };

    // `dataset_len()` is an O(1) counter, not a materializing scan, so this
    // stays cheap even for a multi-million-quad dataset; it is only used for
    // the checkpoint's informational `total_size` field.
    let total_size = store.dataset_len() as u64;

    // Step 3: Serialize quads to output format
    println!("   [2/3] Serializing to {format} format...");

    // Open file for writing (create new or append if resuming)
    let output_file = if start_count > 0 {
        fs::OpenOptions::new().append(true).open(file)?
    } else {
        fs::File::create(file)?
    };

    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("owl", "http://www.w3.org/2002/07/owl#")
        .pretty()
        .for_writer(output_file);

    // Checkpoint every 10,000 quads
    const CHECKPOINT_INTERVAL: usize = 10_000;
    let mut exported_count = 0usize;

    let quad_iter = store
        .quad_iter(target, None, None, None)
        .map_err(|e| format!("Failed to scan tdb2 quads: {e}"))?;

    for (idx, quad_result) in quad_iter.enumerate() {
        // Skip already-exported quads when resuming
        if idx < start_count {
            continue;
        }

        let quad_result =
            quad_result.map_err(|e| format!("Failed to read quad from tdb2 store: {e}"))?;

        let subject = tdb_convert::tdb_term_to_subject(&quad_result.subject)
            .map_err(|e| format!("Invalid tdb2 subject term: {e}"))?;
        let predicate = tdb_convert::tdb_term_to_named_node(&quad_result.predicate)
            .map_err(|e| format!("Invalid tdb2 predicate term: {e}"))?;
        let object = tdb_convert::tdb_term_to_object(&quad_result.object)
            .map_err(|e| format!("Invalid tdb2 object term: {e}"))?;
        let graph_name = tdb_convert::tdb_graph_to_core_graph_name(&quad_result.graph)
            .map_err(|e| format!("Invalid tdb2 graph term: {e}"))?;

        let quad = oxirs_core::model::Quad::new(subject, predicate, object, graph_name);

        serializer
            .serialize_quad(quad.as_ref())
            .map_err(|e| format!("Serialization error: {e}"))?;

        exported_count = idx + 1;

        // Save checkpoint periodically if checkpointing is enabled
        if enable_checkpointing && exported_count % CHECKPOINT_INTERVAL == 0 {
            let checkpoint = Checkpoint {
                operation: "export".to_string(),
                dataset: dataset.to_string(),
                file_path: file.to_string_lossy().to_string(),
                processed_count: exported_count,
                last_offset: 0, // Not used for export
                timestamp: chrono::Local::now().to_rfc3339(),
                format: format.to_string(),
                graph: graph.map(|s| s.to_string()),
                total_size,
            };

            if let Err(e) = checkpoint_manager.save(&checkpoint) {
                eprintln!("Warning: Failed to save checkpoint: {e}");
            }
        }
    }

    serializer
        .finish()
        .map_err(|e| format!("Failed to finalize serialization: {e}"))?;

    println!("       ✓ Serialization completed");

    // Step 4: Write and report
    println!("   [3/3] Writing to file...");
    println!("       ✓ Data written to {}", file.display());

    Ok(exported_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_tdb::dictionary::Term as TdbTerm;

    /// `export_data_tdb2` must stream every default-graph triple and every
    /// named-graph quad currently on disk out to the N-Quads file, proving
    /// the tdb2 read path is real (not empty/simulated).
    #[test]
    fn test_export_data_tdb2_round_trips_default_and_named_graph() {
        let unique = std::process::id() as u64 * 1_000_003 + line!() as u64;
        let dataset_dir = std::env::temp_dir().join(format!("oxirs-export-tdb2-test-{unique}"));
        let out_file = std::env::temp_dir().join(format!("oxirs-export-tdb2-out-{unique}.nq"));

        {
            let mut store = TdbStore::open(&dataset_dir).expect("open tdb2 store");
            store
                .insert_triples_bulk(&[(
                    TdbTerm::Iri("http://ex.org/s".to_string()),
                    TdbTerm::Iri("http://ex.org/p".to_string()),
                    TdbTerm::Iri("http://ex.org/o".to_string()),
                )])
                .expect("bulk insert default-graph triple");
            store
                .insert_quad(
                    Some(&TdbTerm::Iri("http://ex.org/g".to_string())),
                    &TdbTerm::Iri("http://ex.org/s2".to_string()),
                    &TdbTerm::Iri("http://ex.org/p".to_string()),
                    &TdbTerm::Iri("http://ex.org/o2".to_string()),
                )
                .expect("insert named-graph quad");
            store.sync().expect("sync tdb2 store");
        }

        let store = TdbStore::open(&dataset_dir).expect("reopen tdb2 store");
        let checkpoint_manager = CheckpointManager::new().expect("create checkpoint manager");

        let exported = export_data_tdb2(
            &store,
            &out_file,
            "nquads",
            None,
            false, // disable checkpointing: avoid touching the real user config dir in a test
            &checkpoint_manager,
            "test-dataset",
            0,
        )
        .expect("export_data_tdb2 should succeed");

        assert_eq!(
            exported, 2,
            "one default-graph triple + one named-graph quad"
        );

        let contents = std::fs::read_to_string(&out_file).expect("read exported file");
        assert!(
            contents.contains("http://ex.org/s") && contents.contains("http://ex.org/o"),
            "exported file should contain the default-graph triple: {contents}"
        );
        assert!(
            contents.contains("http://ex.org/g"),
            "exported file should contain the named graph: {contents}"
        );

        std::fs::remove_dir_all(&dataset_dir).ok();
        std::fs::remove_file(&out_file).ok();
    }

    /// A graph filter of `"default"` must exclude named-graph quads.
    #[test]
    fn test_export_data_tdb2_default_graph_filter_excludes_named_graph() {
        let unique = std::process::id() as u64 * 1_000_003 + line!() as u64;
        let dataset_dir =
            std::env::temp_dir().join(format!("oxirs-export-tdb2-filter-test-{unique}"));
        let out_file =
            std::env::temp_dir().join(format!("oxirs-export-tdb2-filter-out-{unique}.nq"));

        {
            let mut store = TdbStore::open(&dataset_dir).expect("open tdb2 store");
            store
                .insert_triples_bulk(&[(
                    TdbTerm::Iri("http://ex.org/s".to_string()),
                    TdbTerm::Iri("http://ex.org/p".to_string()),
                    TdbTerm::Iri("http://ex.org/o".to_string()),
                )])
                .expect("bulk insert default-graph triple");
            store
                .insert_quad(
                    Some(&TdbTerm::Iri("http://ex.org/g".to_string())),
                    &TdbTerm::Iri("http://ex.org/s2".to_string()),
                    &TdbTerm::Iri("http://ex.org/p".to_string()),
                    &TdbTerm::Iri("http://ex.org/o2".to_string()),
                )
                .expect("insert named-graph quad");
            store.sync().expect("sync tdb2 store");
        }

        let store = TdbStore::open(&dataset_dir).expect("reopen tdb2 store");
        let checkpoint_manager = CheckpointManager::new().expect("create checkpoint manager");

        let exported = export_data_tdb2(
            &store,
            &out_file,
            "nquads",
            Some("default"),
            false,
            &checkpoint_manager,
            "test-dataset",
            0,
        )
        .expect("export_data_tdb2 should succeed");

        assert_eq!(
            exported, 1,
            "only the default-graph triple should be exported"
        );

        std::fs::remove_dir_all(&dataset_dir).ok();
        std::fs::remove_file(&out_file).ok();
    }
}
