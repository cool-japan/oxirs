//! Data migration commands - Convert RDF formats and migrate databases

use super::CommandResult;
use crate::cli::logging::{DataLogger, PerfLogger};
use crate::cli::validation::MultiValidator;
use crate::cli::validation::{dataset_validation, fs_validation, validate_rdf_format};
use crate::cli::{progress::helpers, ArgumentValidator, CliContext};
use oxirs_core::format::{RdfFormat, RdfParser, RdfSerializer};
use oxirs_core::model::Quad;
use oxirs_core::rdf_store::RdfStore;
use std::fs;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use std::time::Instant;

/// Convert RDF data from one format to another
pub async fn format(source: PathBuf, target: PathBuf, from: String, to: String) -> CommandResult {
    // Create CLI context for proper output formatting
    let ctx = CliContext::new();

    // Validate arguments using the advanced validation framework
    let mut validator = MultiValidator::new();

    // Validate source file
    validator.add(
        ArgumentValidator::new("source", Some(source.to_str().unwrap_or("")))
            .required()
            .is_file(),
    );

    // Validate formats
    validator.add(
        ArgumentValidator::new("from", Some(&from))
            .required()
            .custom(|f| !f.trim().is_empty(), "Source format cannot be empty"),
    );

    validator.add(
        ArgumentValidator::new("to", Some(&to))
            .required()
            .custom(|t| !t.trim().is_empty(), "Target format cannot be empty"),
    );

    // Complete validation
    validator.finish()?;

    // Validate file size (limit to 1GB for now)
    fs_validation::validate_file_size(&source, Some(1_073_741_824))?;

    // Validate formats
    validate_rdf_format(&from)?;
    validate_rdf_format(&to)?;

    ctx.info(&format!(
        "Migrating RDF data: {} → {}",
        source.display(),
        target.display()
    ));
    ctx.info(&format!("Source format: {from}"));
    ctx.info(&format!("Target format: {to}"));

    // Check if output file already exists
    if target.exists() {
        return Err(format!("Target file '{}' already exists", target.display()).into());
    }

    // Ensure output directory exists
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }

    // Start migration with progress tracking and logging
    let start_time = Instant::now();
    ctx.info("Migration Progress");

    // Initialize loggers
    let mut data_logger = DataLogger::new("migrate", source.to_str().unwrap_or("unknown"));
    let mut perf_logger = PerfLogger::new(format!("migrate_{from}_to_{to}"));
    perf_logger.add_metadata("source", source.display().to_string());
    perf_logger.add_metadata("target", target.display().to_string());
    perf_logger.add_metadata("from_format", &from);
    perf_logger.add_metadata("to_format", &to);

    // Get file size for progress bar
    let file_metadata = fs::metadata(&source)?;
    let file_size = file_metadata.len();

    // Create progress bar for file reading
    let read_progress = helpers::download_progress(file_size, &source.display().to_string());
    read_progress.set_message("Reading source file");

    // Open source file for reading
    let source_file = fs::File::open(&source)?;
    read_progress.finish_with_message("Source file opened");
    data_logger.update_progress(file_size, 0);

    // Create progress spinner for migration
    let migrate_progress = helpers::query_progress();
    migrate_progress.set_message("Converting formats");

    // Perform migration
    let (quad_count, error_count) = migrate_data(source_file, &target, &from, &to)?;

    migrate_progress.finish_with_message("Migration complete");

    let duration = start_time.elapsed();

    // Update data logger with final stats
    data_logger.update_progress(file_size, quad_count as u64);
    data_logger.complete();

    // Complete performance logging
    perf_logger.add_metadata("quad_count", quad_count);
    perf_logger.add_metadata("error_count", error_count);
    perf_logger.complete(Some(5000)); // Log if migration takes more than 5 seconds

    // Report statistics with formatted output
    ctx.info("Migration Statistics");
    ctx.success(&format!(
        "Migration completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Quads migrated: {quad_count}"));

    if error_count > 0 {
        ctx.warn(&format!("Errors encountered: {error_count}"));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        quad_count as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Output written to: {}", target.display()));

    Ok(())
}

/// Migrate RDF data from source file to target file
fn migrate_data(
    source_file: fs::File,
    target: &PathBuf,
    from_format: &str,
    to_format: &str,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    // Step 1: Determine source RDF format
    let source_rdf_format = parse_rdf_format(from_format)?;

    // Step 2: Determine target RDF format
    let target_rdf_format = parse_rdf_format(to_format)?;

    // Step 3: Create parser for source format
    let reader = BufReader::new(source_file);
    let parser = RdfParser::new(source_rdf_format);

    // Step 4: Create serializer for target format
    let output_file = fs::File::create(target)?;
    let mut serializer = RdfSerializer::new(target_rdf_format)
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("owl", "http://www.w3.org/2002/07/owl#")
        .pretty()
        .for_writer(output_file);

    // Step 5: Stream parse and serialize
    let mut quad_count = 0;
    let mut error_count = 0;

    for quad_result in parser.for_reader(reader) {
        match quad_result {
            Ok(quad) => {
                // Serialize quad to target format
                match serializer.serialize_quad(quad.as_ref()) {
                    Ok(_) => {
                        quad_count += 1;
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to serialize quad: {e}");
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

    // Step 6: Finalize serialization
    serializer
        .finish()
        .map_err(|e| format!("Failed to finalize serialization: {e}"))?;

    Ok((quad_count, error_count))
}

/// Parse RDF format string into RdfFormat enum
fn parse_rdf_format(format: &str) -> Result<RdfFormat, Box<dyn std::error::Error>> {
    match format.to_lowercase().as_str() {
        "turtle" | "ttl" => Ok(RdfFormat::Turtle),
        "ntriples" | "nt" => Ok(RdfFormat::NTriples),
        "nquads" | "nq" => Ok(RdfFormat::NQuads),
        "trig" => Ok(RdfFormat::TriG),
        "rdfxml" | "rdf" | "xml" => Ok(RdfFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        }),
        "n3" => Ok(RdfFormat::N3),
        _ => Err(format!("Unsupported RDF format: {format}").into()),
    }
}
/// Migrate from Apache Jena TDB1 database to OxiRS
pub async fn from_tdb1(tdb_dir: PathBuf, dataset: String, skip_validation: bool) -> CommandResult {
    let ctx = CliContext::new();

    ctx.info("Migrating from Jena TDB1 to OxiRS");
    ctx.info(&format!("TDB1 directory: {}", tdb_dir.display()));
    ctx.info(&format!("Target dataset: {dataset}"));

    // Validate dataset name
    dataset_validation::validate_dataset_name(&dataset)?;

    // Check if TDB directory exists
    if !tdb_dir.exists() || !tdb_dir.is_dir() {
        return Err(format!("TDB1 directory not found: {}", tdb_dir.display()).into());
    }

    // Detect TDB version
    let tdb_version = detect_tdb_version(&tdb_dir)?;
    if tdb_version != "TDB1" {
        return Err(format!(
            "Directory appears to be {}, not TDB1. Use the appropriate migration command.",
            tdb_version
        )
        .into());
    }

    ctx.info("TDB1 database detected");

    // Start migration
    let start_time = Instant::now();
    let migration_progress = helpers::query_progress();
    migration_progress.set_message("Reading TDB1 database");

    // Read TDB1 data files
    let quads = read_tdb1_data(&tdb_dir, &ctx)?;

    let quads_count = quads.len();
    ctx.info(&format!("Found {} quads", quads_count));
    migration_progress.set_message("Processing quads");

    // Create target OxiRS dataset
    let dataset_path = PathBuf::from(&dataset);
    if dataset_path.exists() {
        return Err(format!("Dataset '{}' already exists", dataset).into());
    }

    fs::create_dir_all(&dataset_path)?;

    // Open OxiRS store
    let mut store = RdfStore::open(&dataset_path)
        .map_err(|e| format!("Failed to create OxiRS dataset: {}", e))?;

    migration_progress.set_message("Importing quads into OxiRS");

    // Import quads
    let mut imported = 0;
    let mut errors = 0;

    for quad in quads {
        match store.insert_quad(quad) {
            Ok(_) => imported += 1,
            Err(e) => {
                errors += 1;
                eprintln!("Warning: Failed to import quad: {}", e);
            }
        }
    }

    migration_progress.finish_with_message("Migration complete");

    let duration = start_time.elapsed();

    // Validation step
    if !skip_validation {
        ctx.info("Validating migrated data...");
        let validation_progress = helpers::query_progress();
        validation_progress.set_message("Counting triples");

        let oxirs_count = store
            .quads()
            .map_err(|e| format!("Validation failed: {}", e))?
            .len();

        validation_progress.finish_with_message("Validation complete");

        if oxirs_count != imported {
            ctx.warn(&format!(
                "Validation warning: Imported {} quads but found {} in OxiRS",
                imported, oxirs_count
            ));
        } else {
            ctx.success(&format!(
                "Validation passed: {} quads verified",
                oxirs_count
            ));
        }
    }

    // Report statistics
    ctx.info("Migration Statistics");
    ctx.success(&format!(
        "Migration completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Quads imported: {}", imported));

    if errors > 0 {
        ctx.warn(&format!("Errors encountered: {}", errors));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        imported as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Dataset created at: {}", dataset_path.display()));

    Ok(())
}

/// Migrate from Apache Jena TDB2 database to OxiRS
pub async fn from_tdb2(tdb_dir: PathBuf, dataset: String, skip_validation: bool) -> CommandResult {
    let ctx = CliContext::new();

    ctx.info("Migrating from Jena TDB2 to OxiRS");
    ctx.info(&format!("TDB2 directory: {}", tdb_dir.display()));
    ctx.info(&format!("Target dataset: {dataset}"));

    // Validate dataset name
    dataset_validation::validate_dataset_name(&dataset)?;

    // Check if TDB directory exists
    if !tdb_dir.exists() || !tdb_dir.is_dir() {
        return Err(format!("TDB2 directory not found: {}", tdb_dir.display()).into());
    }

    // Detect TDB version
    let tdb_version = detect_tdb_version(&tdb_dir)?;
    if tdb_version != "TDB2" {
        return Err(format!(
            "Directory appears to be {}, not TDB2. Use the appropriate migration command.",
            tdb_version
        )
        .into());
    }

    ctx.info("TDB2 database detected");

    // Start migration
    let start_time = Instant::now();
    let migration_progress = helpers::query_progress();
    migration_progress.set_message("Reading TDB2 database");

    // Read TDB2 data files
    let quads = read_tdb2_data(&tdb_dir, &ctx)?;

    let quads_count = quads.len();
    ctx.info(&format!("Found {} quads", quads_count));
    migration_progress.set_message("Processing quads");

    // Create target OxiRS dataset
    let dataset_path = PathBuf::from(&dataset);
    if dataset_path.exists() {
        return Err(format!("Dataset '{}' already exists", dataset).into());
    }

    fs::create_dir_all(&dataset_path)?;

    // Open OxiRS store
    let mut store = RdfStore::open(&dataset_path)
        .map_err(|e| format!("Failed to create OxiRS dataset: {}", e))?;

    migration_progress.set_message("Importing quads into OxiRS");

    // Import quads
    let mut imported = 0;
    let mut errors = 0;

    for quad in quads {
        match store.insert_quad(quad) {
            Ok(_) => imported += 1,
            Err(e) => {
                errors += 1;
                eprintln!("Warning: Failed to import quad: {}", e);
            }
        }
    }

    migration_progress.finish_with_message("Migration complete");

    let duration = start_time.elapsed();

    // Validation step
    if !skip_validation {
        ctx.info("Validating migrated data...");
        let validation_progress = helpers::query_progress();
        validation_progress.set_message("Counting triples");

        let oxirs_count = store
            .quads()
            .map_err(|e| format!("Validation failed: {}", e))?
            .len();

        validation_progress.finish_with_message("Validation complete");

        if oxirs_count != imported {
            ctx.warn(&format!(
                "Validation warning: Imported {} quads but found {} in OxiRS",
                imported, oxirs_count
            ));
        } else {
            ctx.success(&format!(
                "Validation passed: {} quads verified",
                oxirs_count
            ));
        }
    }

    // Report statistics
    ctx.info("Migration Statistics");
    ctx.success(&format!(
        "Migration completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Quads imported: {}", imported));

    if errors > 0 {
        ctx.warn(&format!("Errors encountered: {}", errors));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        imported as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Dataset created at: {}", dataset_path.display()));

    Ok(())
}

/// Migrate from Virtuoso database to OxiRS
pub async fn from_virtuoso(connection: String, dataset: String, graphs: String) -> CommandResult {
    let ctx = CliContext::new();

    ctx.info("Migrating from Virtuoso to OxiRS");
    ctx.info(&format!("Virtuoso endpoint: {}", connection));
    ctx.info(&format!("Target dataset: {}", dataset));

    // Validate dataset name
    dataset_validation::validate_dataset_name(&dataset)?;

    // Validate connection string (must be HTTP/HTTPS URL)
    if !connection.starts_with("http://") && !connection.starts_with("https://") {
        return Err(
            "Connection string must be an HTTP or HTTPS URL (e.g., http://localhost:8890/sparql)"
                .into(),
        );
    }

    // Parse graphs parameter
    let graph_list: Vec<String> = if graphs.is_empty() || graphs == "all" {
        vec![]
    } else {
        graphs.split(',').map(|s| s.trim().to_string()).collect()
    };

    let start_time = Instant::now();

    // Create target OxiRS dataset
    let dataset_path = PathBuf::from(&dataset);
    if dataset_path.exists() {
        return Err(format!("Dataset '{}' already exists", dataset).into());
    }

    fs::create_dir_all(&dataset_path)?;

    // Open OxiRS store
    let mut store = RdfStore::open(&dataset_path)
        .map_err(|e| format!("Failed to create OxiRS dataset: {}", e))?;

    // If no specific graphs requested, discover all named graphs
    let graphs_to_migrate = if graph_list.is_empty() {
        ctx.info("Discovering named graphs from Virtuoso...");
        discover_virtuoso_graphs(&connection).await?
    } else {
        graph_list
    };

    ctx.info(&format!(
        "Found {} graphs to migrate",
        graphs_to_migrate.len()
    ));

    // Migrate each graph
    let mut total_quads = 0;
    let mut total_errors = 0;

    let migrate_progress = helpers::query_progress();

    for (idx, graph_uri) in graphs_to_migrate.iter().enumerate() {
        migrate_progress.set_message("Migrating graphs");
        ctx.info(&format!(
            "Processing graph {}/{}: {}",
            idx + 1,
            graphs_to_migrate.len(),
            graph_uri
        ));

        match extract_virtuoso_graph(&connection, graph_uri, &mut store).await {
            Ok((quads, errors)) => {
                total_quads += quads;
                total_errors += errors;
                ctx.info(&format!("Graph '{}': {} quads imported", graph_uri, quads));
            }
            Err(e) => {
                ctx.warn(&format!("Failed to migrate graph '{}': {}", graph_uri, e));
                total_errors += 1;
            }
        }
    }

    migrate_progress.finish_with_message("Migration complete");

    let duration = start_time.elapsed();

    // Report statistics
    ctx.info("Migration Statistics");
    ctx.success(&format!(
        "Migration completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!(
        "Total graphs migrated: {}",
        graphs_to_migrate.len()
    ));
    ctx.info(&format!("Total quads imported: {}", total_quads));

    if total_errors > 0 {
        ctx.warn(&format!("Errors encountered: {}", total_errors));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        total_quads as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Dataset created at: {}", dataset_path.display()));

    Ok(())
}

/// Migrate from RDF4J repository to OxiRS
pub async fn from_rdf4j(repo_url: PathBuf, dataset: String) -> CommandResult {
    let ctx = CliContext::new();

    // Treat repo_url as a string (it's actually a URL, not a path)
    let repo_url_str = repo_url.to_str().ok_or("Invalid repository URL")?;

    ctx.info("Migrating from RDF4J to OxiRS");
    ctx.info(&format!("RDF4J repository: {}", repo_url_str));
    ctx.info(&format!("Target dataset: {}", dataset));

    // Validate dataset name
    dataset_validation::validate_dataset_name(&dataset)?;

    // Validate repository URL (must be HTTP/HTTPS URL)
    if !repo_url_str.starts_with("http://") && !repo_url_str.starts_with("https://") {
        return Err("Repository URL must be an HTTP or HTTPS URL (e.g., http://localhost:8080/rdf4j-server/repositories/myrepo)".into());
    }

    let start_time = Instant::now();

    // Create target OxiRS dataset
    let dataset_path = PathBuf::from(&dataset);
    if dataset_path.exists() {
        return Err(format!("Dataset '{}' already exists", dataset).into());
    }

    fs::create_dir_all(&dataset_path)?;

    // Open OxiRS store
    let mut store = RdfStore::open(&dataset_path)
        .map_err(|e| format!("Failed to create OxiRS dataset: {}", e))?;

    // Discover all contexts (named graphs) in RDF4J repository
    ctx.info("Discovering contexts from RDF4J...");
    let contexts = discover_rdf4j_contexts(repo_url_str).await?;

    ctx.info(&format!("Found {} contexts to migrate", contexts.len()));

    // Migrate each context
    let mut total_quads = 0;
    let mut total_errors = 0;

    let migrate_progress = helpers::query_progress();
    migrate_progress.set_message("Migrating contexts");

    for (idx, context) in contexts.iter().enumerate() {
        ctx.info(&format!(
            "Processing context {}/{}: {}",
            idx + 1,
            contexts.len(),
            context
        ));

        match extract_rdf4j_context(repo_url_str, context, &mut store).await {
            Ok((quads, errors)) => {
                total_quads += quads;
                total_errors += errors;
                ctx.info(&format!("Context '{}': {} quads imported", context, quads));
            }
            Err(e) => {
                ctx.warn(&format!("Failed to migrate context '{}': {}", context, e));
                total_errors += 1;
            }
        }
    }

    migrate_progress.finish_with_message("Migration complete");

    let duration = start_time.elapsed();

    // Report statistics
    ctx.info("Migration Statistics");
    ctx.success(&format!(
        "Migration completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Total contexts migrated: {}", contexts.len()));
    ctx.info(&format!("Total quads imported: {}", total_quads));

    if total_errors > 0 {
        ctx.warn(&format!("Errors encountered: {}", total_errors));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        total_quads as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Dataset created at: {}", dataset_path.display()));

    Ok(())
}

/// Migrate from Blazegraph database to OxiRS
pub async fn from_blazegraph(
    endpoint: String,
    dataset: String,
    namespace: String,
) -> CommandResult {
    let ctx = CliContext::new();

    ctx.info("Migrating from Blazegraph to OxiRS");
    ctx.info(&format!("Blazegraph endpoint: {}", endpoint));
    ctx.info(&format!("Namespace: {}", namespace));
    ctx.info(&format!("Target dataset: {}", dataset));

    // Validate dataset name
    dataset_validation::validate_dataset_name(&dataset)?;

    // Validate endpoint URL
    if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        return Err("Endpoint must be an HTTP or HTTPS URL (e.g., http://localhost:9999/blazegraph/namespace/kb/sparql)".into());
    }

    let start_time = Instant::now();

    // Create target OxiRS dataset
    let dataset_path = PathBuf::from(&dataset);
    if dataset_path.exists() {
        return Err(format!("Dataset '{}' already exists", dataset).into());
    }

    fs::create_dir_all(&dataset_path)?;

    // Open OxiRS store
    let mut store = RdfStore::open(&dataset_path)
        .map_err(|e| format!("Failed to create OxiRS dataset: {}", e))?;

    // Discover all named graphs in Blazegraph namespace
    ctx.info("Discovering named graphs from Blazegraph...");
    let graphs = discover_blazegraph_graphs(&endpoint).await?;

    ctx.info(&format!("Found {} graphs to migrate", graphs.len()));

    // Migrate each graph
    let mut total_quads = 0;
    let mut total_errors = 0;

    let migrate_progress = helpers::query_progress();
    migrate_progress.set_message("Migrating graphs");

    for (idx, graph_uri) in graphs.iter().enumerate() {
        ctx.info(&format!(
            "Processing graph {}/{}: {}",
            idx + 1,
            graphs.len(),
            graph_uri
        ));

        match extract_blazegraph_graph(&endpoint, graph_uri, &mut store).await {
            Ok((quads, errors)) => {
                total_quads += quads;
                total_errors += errors;
                ctx.info(&format!("Graph '{}': {} quads imported", graph_uri, quads));
            }
            Err(e) => {
                ctx.warn(&format!("Failed to migrate graph '{}': {}", graph_uri, e));
                total_errors += 1;
            }
        }
    }

    migrate_progress.finish_with_message("Migration complete");

    let duration = start_time.elapsed();

    // Report statistics
    ctx.info("Migration Statistics");
    ctx.success(&format!(
        "Migration completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Total graphs migrated: {}", graphs.len()));
    ctx.info(&format!("Total quads imported: {}", total_quads));

    if total_errors > 0 {
        ctx.warn(&format!("Errors encountered: {}", total_errors));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        total_quads as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Dataset created at: {}", dataset_path.display()));

    Ok(())
}

/// Migrate from Ontotext GraphDB to OxiRS
pub async fn from_graphdb(endpoint: String, dataset: String, repository: String) -> CommandResult {
    let ctx = CliContext::new();

    ctx.info("Migrating from GraphDB to OxiRS");
    ctx.info(&format!("GraphDB endpoint: {}", endpoint));
    ctx.info(&format!("Repository: {}", repository));
    ctx.info(&format!("Target dataset: {}", dataset));

    // Validate dataset name
    dataset_validation::validate_dataset_name(&dataset)?;

    // Validate endpoint URL
    if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        return Err("Endpoint must be an HTTP or HTTPS URL (e.g., http://localhost:7200/repositories/myrepo)".into());
    }

    let start_time = Instant::now();

    // Create target OxiRS dataset
    let dataset_path = PathBuf::from(&dataset);
    if dataset_path.exists() {
        return Err(format!("Dataset '{}' already exists", dataset).into());
    }

    fs::create_dir_all(&dataset_path)?;

    // Open OxiRS store
    let mut store = RdfStore::open(&dataset_path)
        .map_err(|e| format!("Failed to create OxiRS dataset: {}", e))?;

    // Discover all named graphs in GraphDB repository
    ctx.info("Discovering named graphs from GraphDB...");
    let graphs = discover_graphdb_graphs(&endpoint).await?;

    ctx.info(&format!("Found {} graphs to migrate", graphs.len()));

    // Migrate each graph
    let mut total_quads = 0;
    let mut total_errors = 0;

    let migrate_progress = helpers::query_progress();
    migrate_progress.set_message("Migrating graphs");

    for (idx, graph_uri) in graphs.iter().enumerate() {
        ctx.info(&format!(
            "Processing graph {}/{}: {}",
            idx + 1,
            graphs.len(),
            graph_uri
        ));

        match extract_graphdb_graph(&endpoint, graph_uri, &mut store).await {
            Ok((quads, errors)) => {
                total_quads += quads;
                total_errors += errors;
                ctx.info(&format!("Graph '{}': {} quads imported", graph_uri, quads));
            }
            Err(e) => {
                ctx.warn(&format!("Failed to migrate graph '{}': {}", graph_uri, e));
                total_errors += 1;
            }
        }
    }

    migrate_progress.finish_with_message("Migration complete");

    let duration = start_time.elapsed();

    // Report statistics
    ctx.info("Migration Statistics");
    ctx.success(&format!(
        "Migration completed in {:.2} seconds",
        duration.as_secs_f64()
    ));
    ctx.info(&format!("Total graphs migrated: {}", graphs.len()));
    ctx.info(&format!("Total quads imported: {}", total_quads));

    if total_errors > 0 {
        ctx.warn(&format!("Errors encountered: {}", total_errors));
    }

    ctx.info(&format!(
        "Average rate: {:.0} quads/second",
        total_quads as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Dataset created at: {}", dataset_path.display()));

    Ok(())
}

/// Detect TDB version from directory contents
fn detect_tdb_version(tdb_dir: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    // TDB1 has files like: nodes.dat, GSPO.idn, GOSP.idn, etc.
    // TDB2 has files like: nodes-data.bdf, GSPO.bpt, GOSP.bpt, etc.

    let tdb1_indicators = ["nodes.dat", "GSPO.idn", "GOSP.idn"];
    let tdb2_indicators = ["nodes-data.bdf", "GSPO.bpt", "GOSP.bpt"];

    let mut tdb1_score = 0;
    let mut tdb2_score = 0;

    for file in &tdb1_indicators {
        if tdb_dir.join(file).exists() {
            tdb1_score += 1;
        }
    }

    for file in &tdb2_indicators {
        if tdb_dir.join(file).exists() {
            tdb2_score += 1;
        }
    }

    if tdb1_score > tdb2_score && tdb1_score >= 2 {
        Ok("TDB1".to_string())
    } else if tdb2_score > tdb1_score && tdb2_score >= 2 {
        Ok("TDB2".to_string())
    } else if tdb1_score == 0 && tdb2_score == 0 {
        Err("Directory does not appear to be a TDB database".into())
    } else {
        Err("Unable to determine TDB version - directory may be corrupted".into())
    }
}

/// Read TDB1 data files and return quads
fn read_tdb1_data(
    tdb_dir: &std::path::Path,
    ctx: &CliContext,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    ctx.info("Reading TDB1 data files...");

    // TDB1 uses a binary format that we need to parse
    // For now, we'll look for N-Triples/N-Quads dump files that users should create
    // using tdbdump before migration

    let dump_file = tdb_dir.join("dump.nq");
    if !dump_file.exists() {
        return Err(format!(
            "TDB1 migration requires a N-Quads dump file. Please run:\n\
             tdbdump --loc {} --out dump.nq\n\
             Then re-run this migration command.",
            tdb_dir.display()
        )
        .into());
    }

    ctx.info("Found N-Quads dump file");

    // Parse the N-Quads file
    let file = fs::File::open(&dump_file)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(RdfFormat::NQuads);

    let mut quads = Vec::new();
    for quad_result in parser.for_reader(reader) {
        match quad_result {
            Ok(quad) => quads.push(quad),
            Err(e) => {
                eprintln!("Warning: Failed to parse quad: {}", e);
            }
        }
    }

    Ok(quads)
}

/// Read TDB2 data files and return quads
fn read_tdb2_data(
    tdb_dir: &std::path::Path,
    ctx: &CliContext,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    ctx.info("Reading TDB2 data files...");

    // TDB2 also uses a binary format
    // For now, we'll look for N-Quads dump files that users should create
    // using tdb2.tdbdump before migration

    let dump_file = tdb_dir.join("dump.nq");
    if !dump_file.exists() {
        return Err(format!(
            "TDB2 migration requires a N-Quads dump file. Please run:\n\
             tdb2.tdbdump --loc {} --out dump.nq\n\
             Then re-run this migration command.",
            tdb_dir.display()
        )
        .into());
    }

    ctx.info("Found N-Quads dump file");

    // Parse the N-Quads file
    let file = fs::File::open(&dump_file)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(RdfFormat::NQuads);

    let mut quads = Vec::new();
    for quad_result in parser.for_reader(reader) {
        match quad_result {
            Ok(quad) => quads.push(quad),
            Err(e) => {
                eprintln!("Warning: Failed to parse quad: {}", e);
            }
        }
    }

    Ok(quads)
}

/// Discover all named graphs in Virtuoso database
async fn discover_virtuoso_graphs(
    endpoint: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // SPARQL query to get all distinct named graphs
    let query = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }";

    let response = client
        .post(endpoint)
        .header("Accept", "application/sparql-results+json")
        .form(&[("query", query)])
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Virtuoso endpoint: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Virtuoso query failed with status: {}", response.status()).into());
    }

    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse SPARQL results: {}", e))?;

    // Parse SPARQL JSON results
    let bindings = result["results"]["bindings"]
        .as_array()
        .ok_or("Invalid SPARQL results format")?;

    let graphs: Vec<String> = bindings
        .iter()
        .filter_map(|binding| binding["g"]["value"].as_str().map(|s| s.to_string()))
        .collect();

    Ok(graphs)
}

/// Extract all quads from a specific Virtuoso graph
async fn extract_virtuoso_graph(
    endpoint: &str,
    graph_uri: &str,
    store: &mut RdfStore,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // SPARQL CONSTRUCT query to get all quads from the graph
    let query = format!(
        "CONSTRUCT {{ ?s ?p ?o }} WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
        graph_uri
    );

    let response = client
        .post(endpoint)
        .header("Accept", "application/n-quads")
        .form(&[("query", &query)])
        .send()
        .await
        .map_err(|e| format!("Failed to query Virtuoso: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Virtuoso query failed with status: {}", response.status()).into());
    }

    // Get response body as N-Quads
    let nquads_data = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Convert to bytes for parsing
    let nquads_bytes = nquads_data.into_bytes();
    let cursor = Cursor::new(nquads_bytes);

    // Parse N-Quads data
    let parser = RdfParser::new(RdfFormat::NQuads);
    let mut quad_count = 0;
    let mut error_count = 0;

    for quad_result in parser.for_reader(cursor) {
        match quad_result {
            Ok(quad) => match store.insert_quad(quad) {
                Ok(_) => quad_count += 1,
                Err(e) => {
                    eprintln!("Warning: Failed to insert quad: {}", e);
                    error_count += 1;
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to parse quad: {}", e);
                error_count += 1;
            }
        }
    }

    Ok((quad_count, error_count))
}

/// Discover all contexts (named graphs) in RDF4J repository
async fn discover_rdf4j_contexts(
    repo_url: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // RDF4J SPARQL query to get all distinct contexts
    let query = "SELECT DISTINCT ?c WHERE { GRAPH ?c { ?s ?p ?o } }";

    // RDF4J uses /repositories/<repo-id>/query endpoint
    let response = client
        .get(repo_url)
        .query(&[("query", query)])
        .header("Accept", "application/sparql-results+json")
        .send()
        .await
        .map_err(|e| format!("Failed to connect to RDF4J repository: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("RDF4J query failed with status: {}", response.status()).into());
    }

    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse SPARQL results: {}", e))?;

    // Parse SPARQL JSON results
    let bindings = result["results"]["bindings"]
        .as_array()
        .ok_or("Invalid SPARQL results format")?;

    let contexts: Vec<String> = bindings
        .iter()
        .filter_map(|binding| binding["c"]["value"].as_str().map(|s| s.to_string()))
        .collect();

    Ok(contexts)
}

/// Extract all quads from a specific RDF4J context
async fn extract_rdf4j_context(
    repo_url: &str,
    context_uri: &str,
    store: &mut RdfStore,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // RDF4J SPARQL CONSTRUCT query to get all quads from the context
    let query = format!(
        "CONSTRUCT {{ ?s ?p ?o }} WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
        context_uri
    );

    // RDF4J returns statements endpoint for export
    // We'll use the query endpoint with CONSTRUCT
    let response = client
        .get(repo_url)
        .query(&[("query", &query)])
        .header("Accept", "application/n-quads")
        .send()
        .await
        .map_err(|e| format!("Failed to query RDF4J: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("RDF4J query failed with status: {}", response.status()).into());
    }

    // Get response body as N-Quads
    let nquads_data = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Convert to bytes for parsing
    let nquads_bytes = nquads_data.into_bytes();
    let cursor = Cursor::new(nquads_bytes);

    // Parse N-Quads data
    let parser = RdfParser::new(RdfFormat::NQuads);
    let mut quad_count = 0;
    let mut error_count = 0;

    for quad_result in parser.for_reader(cursor) {
        match quad_result {
            Ok(quad) => match store.insert_quad(quad) {
                Ok(_) => quad_count += 1,
                Err(e) => {
                    eprintln!("Warning: Failed to insert quad: {}", e);
                    error_count += 1;
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to parse quad: {}", e);
                error_count += 1;
            }
        }
    }

    Ok((quad_count, error_count))
}

/// Discover all named graphs in Blazegraph database
async fn discover_blazegraph_graphs(
    endpoint: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // SPARQL query to get all distinct named graphs
    let query = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }";

    let response = client
        .post(endpoint)
        .header("Accept", "application/sparql-results+json")
        .form(&[("query", query)])
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Blazegraph endpoint: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Blazegraph query failed with status: {}", response.status()).into());
    }

    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse SPARQL results: {}", e))?;

    // Parse SPARQL JSON results
    let bindings = result["results"]["bindings"]
        .as_array()
        .ok_or("Invalid SPARQL results format")?;

    let graphs: Vec<String> = bindings
        .iter()
        .filter_map(|binding| binding["g"]["value"].as_str().map(|s| s.to_string()))
        .collect();

    Ok(graphs)
}

/// Extract all quads from a specific Blazegraph graph
async fn extract_blazegraph_graph(
    endpoint: &str,
    graph_uri: &str,
    store: &mut RdfStore,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // SPARQL CONSTRUCT query to get all quads from the graph
    let query = format!(
        "CONSTRUCT {{ ?s ?p ?o }} WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
        graph_uri
    );

    let response = client
        .post(endpoint)
        .header("Accept", "application/n-quads")
        .form(&[("query", &query)])
        .send()
        .await
        .map_err(|e| format!("Failed to query Blazegraph: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Blazegraph query failed with status: {}", response.status()).into());
    }

    // Get response body as N-Quads
    let nquads_data = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Convert to bytes for parsing
    let nquads_bytes = nquads_data.into_bytes();
    let cursor = Cursor::new(nquads_bytes);

    // Parse N-Quads data
    let parser = RdfParser::new(RdfFormat::NQuads);
    let mut quad_count = 0;
    let mut error_count = 0;

    for quad_result in parser.for_reader(cursor) {
        match quad_result {
            Ok(quad) => match store.insert_quad(quad) {
                Ok(_) => quad_count += 1,
                Err(e) => {
                    eprintln!("Warning: Failed to insert quad: {}", e);
                    error_count += 1;
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to parse quad: {}", e);
                error_count += 1;
            }
        }
    }

    Ok((quad_count, error_count))
}

/// Discover all named graphs in GraphDB repository
async fn discover_graphdb_graphs(
    endpoint: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // SPARQL query to get all distinct named graphs
    let query = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }";

    let response = client
        .get(endpoint)
        .query(&[("query", query)])
        .header("Accept", "application/sparql-results+json")
        .send()
        .await
        .map_err(|e| format!("Failed to connect to GraphDB endpoint: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("GraphDB query failed with status: {}", response.status()).into());
    }

    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse SPARQL results: {}", e))?;

    // Parse SPARQL JSON results
    let bindings = result["results"]["bindings"]
        .as_array()
        .ok_or("Invalid SPARQL results format")?;

    let graphs: Vec<String> = bindings
        .iter()
        .filter_map(|binding| binding["g"]["value"].as_str().map(|s| s.to_string()))
        .collect();

    Ok(graphs)
}

/// Extract all quads from a specific GraphDB graph
async fn extract_graphdb_graph(
    endpoint: &str,
    graph_uri: &str,
    store: &mut RdfStore,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // SPARQL CONSTRUCT query to get all quads from the graph
    let query = format!(
        "CONSTRUCT {{ ?s ?p ?o }} WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
        graph_uri
    );

    let response = client
        .get(endpoint)
        .query(&[("query", &query)])
        .header("Accept", "application/n-quads")
        .send()
        .await
        .map_err(|e| format!("Failed to query GraphDB: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("GraphDB query failed with status: {}", response.status()).into());
    }

    // Get response body as N-Quads
    let nquads_data = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Convert to bytes for parsing
    let nquads_bytes = nquads_data.into_bytes();
    let cursor = Cursor::new(nquads_bytes);

    // Parse N-Quads data
    let parser = RdfParser::new(RdfFormat::NQuads);
    let mut quad_count = 0;
    let mut error_count = 0;

    for quad_result in parser.for_reader(cursor) {
        match quad_result {
            Ok(quad) => match store.insert_quad(quad) {
                Ok(_) => quad_count += 1,
                Err(e) => {
                    eprintln!("Warning: Failed to insert quad: {}", e);
                    error_count += 1;
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to parse quad: {}", e);
                error_count += 1;
            }
        }
    }

    Ok((quad_count, error_count))
}
