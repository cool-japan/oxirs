//! Public API for RDF dataset generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::{
    super::types::*, domain_data, random_data, schema_detect, schema_owl, schema_rdfs, schema_shacl,
};
use crate::cli::logging::{DataLogger, PerfLogger};
use crate::cli::{format_bytes, format_duration, format_number};
use crate::cli::{progress::helpers, CliContext};
use crate::commands::CommandResult;
use oxirs_core::format::RdfSerializer;
use scirs2_core::random::Random;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Generate a synthetic RDF dataset
pub async fn run(
    output: PathBuf,
    size: String,
    dataset_type: String,
    format: String,
    seed: Option<u64>,
    schema: Option<PathBuf>,
) -> CommandResult {
    let ctx = CliContext::new();
    if let Some(schema_file) = schema {
        return run_schema_based_generation(output, size, schema_file, format, seed).await;
    }
    ctx.info("Generating synthetic RDF dataset");
    ctx.info(&format!("Output file: {}", output.display()));
    let size_enum = DatasetSize::from_string(&size)?;
    let type_enum = DatasetType::from_string(&dataset_type)?;
    let triple_count = size_enum.triple_count();
    ctx.info(&format!("Size: {} ({} triples)", size, triple_count));
    ctx.info(&format!("Type: {}", dataset_type));
    ctx.info(&format!("Format: {}", format));
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let mut data_logger = DataLogger::new("generate", output.to_str().unwrap_or("unknown"));
    let mut perf_logger = PerfLogger::new(format!("generate_{}", dataset_type));
    perf_logger.add_metadata("size", size.clone());
    perf_logger.add_metadata("type", dataset_type.clone());
    perf_logger.add_metadata("format", &format);
    if let Some(s) = seed {
        perf_logger.add_metadata("seed", s.to_string());
    }
    let rdf_format = random_data::parse_rdf_format(&format)?;
    let progress = helpers::query_progress();
    progress.set_message("Generating RDF triples");
    let quads = match type_enum {
        DatasetType::Rdf => random_data::generate_random_rdf(&mut rng, triple_count),
        DatasetType::Graph => random_data::generate_graph_structure(&mut rng, triple_count),
        DatasetType::Semantic => domain_data::generate_semantic_data(&mut rng, triple_count),
        DatasetType::Bibliographic => {
            domain_data::generate_bibliographic_data(&mut rng, triple_count)
        }
        DatasetType::Geographic => domain_data::generate_geographic_data(&mut rng, triple_count),
        DatasetType::Organizational => {
            domain_data::generate_organizational_data(&mut rng, triple_count)
        }
    };
    progress.set_message("Writing to file");
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        match serializer.serialize_quad(quad.as_ref()) {
            Ok(_) => written += 1,
            Err(e) => {
                return Err(format!("Failed to serialize quad: {}", e).into());
            }
        }
    }
    serializer
        .finish()
        .map_err(|e| format!("Failed to finalize serialization: {}", e))?;
    progress.finish_with_message("Dataset generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    data_logger.update_progress(file_size, written as u64);
    data_logger.complete();
    perf_logger.add_metadata("triple_count", written);
    perf_logger.complete(Some(5000));
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Triples generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} triples/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}

/// Generate RDF data conforming to SHACL shapes
pub async fn from_shacl(
    shapes_file: PathBuf,
    output: PathBuf,
    count: usize,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Generating RDF data from SHACL shapes");
    ctx.info(&format!("Shapes file: {}", shapes_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    ctx.info(&format!("Instance count: {}", count));
    ctx.info(&format!("Format: {}", format));
    if !shapes_file.exists() {
        return Err(format!("Shapes file '{}' does not exist", shapes_file.display()).into());
    }
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let progress = helpers::query_progress();
    progress.set_message("Parsing SHACL shapes");
    let shapes = schema_shacl::parse_shacl_shapes(&shapes_file, &ctx)?;
    ctx.info(&format!("Found {} shape definitions", shapes.len()));
    progress.set_message("Generating conforming data");
    let quads = schema_shacl::generate_from_shapes(&mut rng, &shapes, count)?;
    ctx.info(&format!("Generated {} quads", quads.len()));
    progress.set_message("Writing output file");
    let rdf_format = random_data::parse_rdf_format(&format)?;
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("sh", "http://www.w3.org/ns/shacl#")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .with_prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        serializer.serialize_quad(quad.as_ref())?;
        written += 1;
    }
    serializer.finish()?;
    progress.finish_with_message("Generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Quads generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} quads/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}

/// Generate RDF data conforming to RDFS schema
pub async fn from_rdfs(
    schema_file: PathBuf,
    output: PathBuf,
    count: usize,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Generating RDF data from RDFS schema");
    ctx.info(&format!("Schema file: {}", schema_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    ctx.info(&format!("Instance count: {}", count));
    ctx.info(&format!("Format: {}", format));
    if !schema_file.exists() {
        return Err(format!("Schema file '{}' does not exist", schema_file.display()).into());
    }
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let progress = helpers::query_progress();
    progress.set_message("Parsing RDFS schema");
    let schema = schema_rdfs::parse_rdfs_schema(&schema_file, &ctx)?;
    ctx.info(&format!(
        "Found {} classes and {} properties",
        schema.classes.len(),
        schema.properties.len()
    ));
    progress.set_message("Generating conforming data");
    let quads = schema_rdfs::generate_from_rdfs_schema(&mut rng, &schema, count)?;
    ctx.info(&format!("Generated {} quads", quads.len()));
    progress.set_message("Writing output file");
    let rdf_format = random_data::parse_rdf_format(&format)?;
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .with_prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        serializer.serialize_quad(quad.as_ref())?;
        written += 1;
    }
    serializer.finish()?;
    progress.finish_with_message("Generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Quads generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} quads/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}

/// Generate RDF data conforming to OWL ontology
pub async fn from_owl(
    ontology_file: PathBuf,
    output: PathBuf,
    count: usize,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Generating RDF data from OWL ontology");
    ctx.info(&format!("Ontology file: {}", ontology_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    ctx.info(&format!("Instance count: {}", count));
    ctx.info(&format!("Format: {}", format));
    if !ontology_file.exists() {
        return Err(format!("Ontology file '{}' does not exist", ontology_file.display()).into());
    }
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let progress = helpers::query_progress();
    progress.set_message("Parsing OWL ontology");
    let ontology = schema_owl::parse_owl_ontology(&ontology_file, &ctx)?;
    ctx.info(&format!(
        "Found {} classes and {} properties",
        ontology.classes.len(),
        ontology.properties.len()
    ));
    progress.set_message("Generating conforming data");
    let quads = schema_owl::generate_from_owl_ontology(&mut rng, &ontology, count)?;
    ctx.info(&format!("Generated {} quads", quads.len()));
    progress.set_message("Writing output file");
    let rdf_format = random_data::parse_rdf_format(&format)?;
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("owl", "http://www.w3.org/2002/07/owl#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .with_prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        serializer.serialize_quad(quad.as_ref())?;
        written += 1;
    }
    serializer.finish()?;
    progress.finish_with_message("Generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Quads generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} quads/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}

/// Run schema-based RDF data generation (SHACL/RDFS/OWL)
async fn run_schema_based_generation(
    output: PathBuf,
    size: String,
    schema_file: PathBuf,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Schema-based RDF data generation");
    ctx.info(&format!("Schema file: {}", schema_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    let size_enum = DatasetSize::from_string(&size)?;
    let instance_count = size_enum.triple_count();
    ctx.info(&format!("Instances to generate: {}", instance_count));
    ctx.info(&format!("Format: {}", format));
    if !schema_file.exists() {
        return Err(format!("Schema file '{}' not found", schema_file.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let mut data_logger = DataLogger::new("generate_schema", output.to_str().unwrap_or("unknown"));
    let mut perf_logger = PerfLogger::new("generate_schema");
    perf_logger.add_metadata("schema_file", schema_file.display().to_string());
    perf_logger.add_metadata("instance_count", instance_count.to_string());
    perf_logger.add_metadata("format", &format);
    if let Some(s) = seed {
        perf_logger.add_metadata("seed", s.to_string());
    }
    ctx.info("Detecting schema type...");
    let schema_type = schema_detect::detect_schema_type(&schema_file, &ctx)?;
    ctx.info(&format!("Detected schema type: {}", schema_type));
    let quads = match schema_type.as_str() {
        "SHACL" => {
            ctx.info("Parsing SHACL shapes...");
            let shapes = schema_shacl::parse_shacl_shapes(&schema_file, &ctx)?;
            if shapes.is_empty() {
                return Err(
                    "No SHACL shapes found in schema file. Ensure the file contains sh:NodeShape definitions with sh:targetClass."
                        .into(),
                );
            }
            ctx.info(&format!(
                "Found {} shapes with target classes",
                shapes.len()
            ));
            ctx.info("Generating RDF data from SHACL shapes...");
            schema_shacl::generate_from_shapes(&mut rng, &shapes, instance_count)?
        }
        "RDFS" => {
            ctx.info("Parsing RDFS schema...");
            let schema = schema_rdfs::parse_rdfs_schema(&schema_file, &ctx)?;
            if schema.classes.is_empty() {
                return Err(
                    "No RDFS classes found in schema file. Ensure the file contains rdfs:Class definitions."
                        .into(),
                );
            }
            ctx.info(&format!(
                "Found {} RDFS classes and {} properties",
                schema.classes.len(),
                schema.properties.len()
            ));
            ctx.info("Generating RDF data from RDFS schema...");
            schema_rdfs::generate_from_rdfs_schema(&mut rng, &schema, instance_count)?
        }
        "OWL" => {
            ctx.info("Parsing OWL ontology...");
            let ontology = schema_owl::parse_owl_ontology(&schema_file, &ctx)?;

            if ontology.classes.is_empty() {
                return Err("No OWL classes found in ontology file. Ensure the file contains owl:Class definitions.".into());
            }

            ctx.info(&format!(
                "Found {} OWL classes, {} properties",
                ontology.classes.len(),
                ontology.properties.len()
            ));
            ctx.info("Generating RDF data from OWL ontology...");
            schema_owl::generate_from_owl_ontology(&mut rng, &ontology, instance_count)?
        }
        _ => {
            return Err(format!(
                "Unknown schema type: {}. Supported types: SHACL, RDFS, OWL",
                schema_type
            )
            .into());
        }
    };
    ctx.info(&format!("Generated {} RDF quads", quads.len()));
    let rdf_format = random_data::parse_rdf_format(&format)?;
    ctx.info("Writing to file...");
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("sh", "http://www.w3.org/ns/shacl#")
        .with_prefix("ex", "http://example.org/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        match serializer.serialize_quad(quad.as_ref()) {
            Ok(_) => written += 1,
            Err(e) => {
                return Err(format!("Failed to serialize quad: {}", e).into());
            }
        }
    }
    serializer
        .finish()
        .map_err(|e| format!("Failed to finalize serialization: {}", e))?;
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    data_logger.update_progress(file_size, written);
    data_logger.complete();
    perf_logger.add_metadata("quad_count", written);
    perf_logger.complete(Some(5000));
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "Generation completed in {}",
        format_duration(duration)
    ));
    ctx.info(&format!("Quads generated: {}", format_number(written)));
    ctx.info(&format!("File size: {}", format_bytes(file_size)));
    ctx.info(&format!(
        "Generation rate: {:.0} quads/second",
        written as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::*;

    #[test]
    fn test_dataset_size_parsing() {
        assert!(matches!(
            DatasetSize::from_string("tiny"),
            Ok(DatasetSize::Tiny)
        ));
        assert!(matches!(
            DatasetSize::from_string("small"),
            Ok(DatasetSize::Small)
        ));
        assert!(matches!(
            DatasetSize::from_string("medium"),
            Ok(DatasetSize::Medium)
        ));
        assert!(matches!(
            DatasetSize::from_string("1000"),
            Ok(DatasetSize::Custom(1000))
        ));
        assert!(DatasetSize::from_string("invalid").is_err());
    }

    #[test]
    fn test_dataset_type_parsing() {
        assert!(matches!(
            DatasetType::from_string("rdf"),
            Ok(DatasetType::Rdf)
        ));
        assert!(matches!(
            DatasetType::from_string("graph"),
            Ok(DatasetType::Graph)
        ));
        assert!(matches!(
            DatasetType::from_string("semantic"),
            Ok(DatasetType::Semantic)
        ));
        assert!(matches!(
            DatasetType::from_string("bibliographic"),
            Ok(DatasetType::Bibliographic)
        ));
        assert!(matches!(
            DatasetType::from_string("bib"),
            Ok(DatasetType::Bibliographic)
        ));
        assert!(matches!(
            DatasetType::from_string("geographic"),
            Ok(DatasetType::Geographic)
        ));
        assert!(matches!(
            DatasetType::from_string("geo"),
            Ok(DatasetType::Geographic)
        ));
        assert!(matches!(
            DatasetType::from_string("organizational"),
            Ok(DatasetType::Organizational)
        ));
        assert!(matches!(
            DatasetType::from_string("org"),
            Ok(DatasetType::Organizational)
        ));
        assert!(DatasetType::from_string("invalid").is_err());
    }

    #[test]
    fn test_triple_counts() {
        assert_eq!(DatasetSize::Tiny.triple_count(), 100);
        assert_eq!(DatasetSize::Small.triple_count(), 1_000);
        assert_eq!(DatasetSize::Medium.triple_count(), 10_000);
        assert_eq!(DatasetSize::Large.triple_count(), 100_000);
        assert_eq!(DatasetSize::XLarge.triple_count(), 1_000_000);
        assert_eq!(DatasetSize::Custom(500).triple_count(), 500);
    }
}
