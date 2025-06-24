//! Data import command

use std::path::PathBuf;
use std::fs;
use std::time::Instant;
use super::CommandResult;
use oxirs_core::store::Store;

/// Import RDF data into a dataset
pub async fn run(
    dataset: String,
    file: PathBuf,
    format: Option<String>,
    graph: Option<String>,
) -> CommandResult {
    println!("Importing data into dataset '{}'", dataset);
    println!("Source file: {}", file.display());
    
    // Check if source file exists
    if !file.exists() {
        return Err(format!("Source file '{}' does not exist", file.display()).into());
    }
    
    // Detect format if not specified
    let detected_format = format.unwrap_or_else(|| detect_format(&file));
    println!("Format: {}", detected_format);
    
    // Validate format
    if !is_supported_format(&detected_format) {
        return Err(format!(
            "Unsupported format '{}'. Supported formats: turtle, ntriples, rdfxml, jsonld, trig, nquads",
            detected_format
        ).into());
    }
    
    if let Some(g) = &graph {
        println!("Target graph: {}", g);
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
    let mut store = if dataset_path.is_dir() {
        Store::open(&dataset_path)?
    } else {
        return Err(format!("Dataset '{}' not found. Use 'oxide init' to create a dataset.", dataset).into());
    };
    
    // Start import
    let start_time = Instant::now();
    println!("Starting import...");
    
    // Read and parse file
    let content = fs::read_to_string(&file)?;
    let (triple_count, error_count) = parse_and_import(&mut store, &content, &detected_format, graph.as_deref())?;
    
    let duration = start_time.elapsed();
    
    // Report statistics
    println!("Import completed in {:.2} seconds", duration.as_secs_f64());
    println!("Triples imported: {}", triple_count);
    if error_count > 0 {
        println!("Errors encountered: {}", error_count);
    }
    println!("Average rate: {:.0} triples/second", triple_count as f64 / duration.as_secs_f64());
    
    Ok(())
}

/// Detect RDF format from file extension
fn detect_format(file: &PathBuf) -> String {
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
fn is_supported_format(format: &str) -> bool {
    matches!(format, "turtle" | "ntriples" | "rdfxml" | "jsonld" | "trig" | "nquads")
}

/// Load dataset configuration from oxirs.toml file
fn load_dataset_from_config(dataset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_path = PathBuf::from(dataset).join("oxirs.toml");
    
    if !config_path.exists() {
        return Err(format!("Configuration file '{}' not found", config_path.display()).into());
    }
    
    // For now, just return the dataset directory
    // TODO: Parse TOML configuration and extract actual storage path
    Ok(PathBuf::from(dataset))
}

/// Parse RDF content and import into store
fn parse_and_import(
    store: &mut Store,
    content: &str,
    format: &str,
    _graph: Option<&str>,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut triple_count = 0;
    let mut error_count = 0;
    
    // Simple parsing simulation - in reality this would use proper RDF parsers
    match format {
        "turtle" | "ntriples" => {
            // Very basic N-Triples/Turtle parsing simulation
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                
                // Try to parse as simple triple format: <s> <p> <o> .
                if let Some(triple) = parse_simple_triple(line) {
                    match store.insert(&triple.0, &triple.1, &triple.2) {
                        Ok(_) => triple_count += 1,
                        Err(_) => error_count += 1,
                    }
                } else {
                    error_count += 1;
                }
            }
        }
        _ => {
            // For other formats, we'd need proper parsers
            return Err(format!("Format '{}' parsing not yet implemented", format).into());
        }
    }
    
    Ok((triple_count, error_count))
}

/// Parse a simple triple line: <subject> <predicate> <object> .
fn parse_simple_triple(line: &str) -> Option<(String, String, String)> {
    // Very basic parsing - just for demonstration
    // Real implementation would use proper RDF parsing libraries
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 4 && line.ends_with('.') {
        let subject = parts[0].trim_matches('<').trim_matches('>').to_string();
        let predicate = parts[1].trim_matches('<').trim_matches('>').to_string();
        let object = if parts[2].starts_with('<') {
            parts[2].trim_matches('<').trim_matches('>').to_string()
        } else {
            // Handle literal values
            parts[2..parts.len()-1].join(" ").trim_matches('"').to_string()
        };
        Some((subject, predicate, object))
    } else {
        None
    }
}