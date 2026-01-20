//! Data export command

use super::CommandResult;
use crate::cli::{Checkpoint, CheckpointManager};
use oxirs_core::format::{RdfFormat, RdfSerializer};
use oxirs_core::rdf_store::RdfStore;
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

    // Open store
    let store = if dataset_path.is_dir() {
        RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {e}"))?
    } else {
        return Err(format!(
            "Dataset '{dataset}' not found. Use 'oxirs init' to create a dataset."
        )
        .into());
    };

    // Start export
    let start_time = Instant::now();
    println!("Starting export...");

    // Export data
    let triple_count = export_data(
        &store,
        &file,
        &format,
        graph.as_deref(),
        resume,
        &checkpoint_manager,
        &dataset,
        start_count,
    )?;

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

    // Step 2: Query all quads from the store
    println!("   [1/3] Querying triples from store...");
    let all_quads = store
        .quads()
        .map_err(|e| format!("Failed to query quads: {e}"))?;

    let quads: Vec<_> = if let Some(graph_name) = graph {
        // Filter by specific graph
        use oxirs_core::model::{GraphName, NamedNode};
        let graph_filter = if graph_name == "default" {
            GraphName::DefaultGraph
        } else {
            GraphName::NamedNode(
                NamedNode::new(graph_name).map_err(|e| format!("Invalid graph IRI: {e}"))?,
            )
        };

        all_quads
            .into_iter()
            .filter(|quad| quad.graph_name() == &graph_filter)
            .collect()
    } else {
        // Export all quads
        all_quads
    };

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
