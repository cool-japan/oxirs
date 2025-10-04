//! Data migration command - Convert RDF data between formats

use super::CommandResult;
use crate::cli::logging::{DataLogger, PerfLogger};
use crate::cli::validation::MultiValidator;
use crate::cli::validation::{fs_validation, validate_rdf_format};
use crate::cli::{progress::helpers, ArgumentValidator, CliContext};
use oxirs_core::format::{RdfFormat, RdfParser, RdfSerializer};
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

/// Migrate RDF data from one format to another
pub async fn run(source: PathBuf, target: PathBuf, from: String, to: String) -> CommandResult {
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
