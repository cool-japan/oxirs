//! TDB Loader - Bulk data loading utility
//!
//! High-performance bulk loading of RDF data into TDB datasets.

use super::{utils, ToolResult, ToolStats};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Run tdbloader command - bulk data loading
pub async fn run(
    location: PathBuf,
    files: Vec<PathBuf>,
    graph: Option<String>,
    progress: bool,
    stats: bool,
) -> ToolResult {
    let mut tool_stats = ToolStats::new();

    println!("TDB Loader - Bulk data loading");
    println!("Dataset location: {}", location.display());
    println!("Input files: {}", files.len());

    if files.is_empty() {
        return Err("No input files specified".into());
    }

    if let Some(ref graph_uri) = graph {
        println!("Target graph: {graph_uri}");
        utils::validate_iri(graph_uri).map_err(|e| format!("Invalid graph URI: {e}"))?;
    }

    // Check if dataset location exists or create it
    if !location.exists() {
        println!("Creating dataset directory: {}", location.display());
        fs::create_dir_all(&location)?;
    } else if !location.is_dir() {
        return Err(format!(
            "Dataset location is not a directory: {}",
            location.display()
        )
        .into());
    }

    // Validate all input files before starting
    println!("Validating input files...");
    for file in &files {
        utils::check_file_readable(file)?;
        let format = utils::detect_rdf_format(file);
        if !utils::is_supported_input_format(&format) {
            return Err(
                format!("Unsupported format for file {}: {}", file.display(), format).into(),
            );
        }
    }

    let mut total_triples = 0;
    let mut total_errors = 0;
    let mut progress_indicator = if progress {
        Some(utils::ProgressIndicator::new())
    } else {
        None
    };

    println!("Starting bulk load...");
    let load_start = Instant::now();

    for (i, file) in files.iter().enumerate() {
        println!(
            "\nLoading file {}/{}: {}",
            i + 1,
            files.len(),
            file.display()
        );

        let file_size = fs::metadata(file)?.len();
        println!("  Size: {}", utils::format_file_size(file_size));

        let format = utils::detect_rdf_format(file);
        println!("  Format: {format}");

        let file_start = Instant::now();
        let result = load_file(&location, file, &format, graph.as_deref())?;
        let file_duration = file_start.elapsed();

        total_triples += result.triples_loaded;
        total_errors += result.errors;
        tool_stats.items_processed += result.triples_loaded;
        tool_stats.errors += result.errors;

        println!("  Triples loaded: {}", result.triples_loaded);
        if result.errors > 0 {
            println!("  Errors: {}", result.errors);
        }
        println!("  Duration: {}", utils::format_duration(file_duration));

        if result.triples_loaded > 0 {
            let rate = result.triples_loaded as f64 / file_duration.as_secs_f64();
            println!("  Rate: {rate:.0} triples/second");
        }

        if let Some(ref mut indicator) = progress_indicator {
            indicator.update(i + 1, Some(files.len()));
        }
    }

    let total_duration = load_start.elapsed();

    if let Some(ref indicator) = progress_indicator {
        indicator.finish(files.len());
    }

    // Final statistics
    println!("\n=== Load Complete ===");
    println!("Total files processed: {}", files.len());
    println!("Total triples loaded: {total_triples}");
    if total_errors > 0 {
        println!("Total errors: {total_errors}");
    }
    println!("Total duration: {}", utils::format_duration(total_duration));

    if total_triples > 0 {
        let overall_rate = total_triples as f64 / total_duration.as_secs_f64();
        println!("Overall rate: {overall_rate:.0} triples/second");
    }

    if stats {
        print_dataset_statistics(&location)?;
    }

    tool_stats.finish();
    tool_stats.print_summary("TDB Loader");

    if total_errors > 0 {
        return Err(format!("Load completed with {total_errors} errors").into());
    }

    Ok(())
}

/// Result of loading a single file
struct LoadResult {
    triples_loaded: usize,
    errors: usize,
}

/// Load a single RDF file into the dataset
fn load_file(
    dataset_location: &Path,
    file_path: &Path,
    format: &str,
    graph_uri: Option<&str>,
) -> ToolResult<LoadResult> {
    // Read file content
    let content = utils::read_input(file_path)?;

    // Parse RDF content
    let (triples, errors) = parse_rdf_for_loading(&content, format)?;

    // Load into dataset
    let loaded_count = store_triples_in_dataset(dataset_location, &triples, graph_uri)?;

    Ok(LoadResult {
        triples_loaded: loaded_count,
        errors,
    })
}

/// Simple triple representation for loading
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LoadTriple {
    subject: String,
    predicate: String,
    object: String,
    graph: Option<String>,
}

/// Parse RDF content for bulk loading
fn parse_rdf_for_loading(content: &str, format: &str) -> ToolResult<(Vec<LoadTriple>, usize)> {
    let mut triples = Vec::new();
    let mut errors = 0;

    match format {
        "ntriples" => {
            for (line_num, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                match parse_ntriples_line_for_loading(line) {
                    Ok(Some(triple)) => triples.push(triple),
                    Ok(None) => {} // Comment or empty line
                    Err(e) => {
                        eprintln!("    Line {}: {}", line_num + 1, e);
                        errors += 1;
                    }
                }
            }
        }
        "nquads" => {
            for (line_num, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                match parse_nquads_line_for_loading(line) {
                    Ok(Some(triple)) => triples.push(triple),
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("    Line {}: {}", line_num + 1, e);
                        errors += 1;
                    }
                }
            }
        }
        "turtle" => {
            // Basic Turtle parsing for loading
            match parse_turtle_for_loading(content) {
                Ok(parsed_triples) => triples.extend(parsed_triples),
                Err(e) => {
                    eprintln!("    Turtle parsing error: {e}");
                    errors += 1;
                }
            }
        }
        _ => {
            return Err(format!("Loading format '{format}' not yet implemented").into());
        }
    }

    Ok((triples, errors))
}

/// Parse N-Triples line for loading
fn parse_ntriples_line_for_loading(line: &str) -> Result<Option<LoadTriple>, String> {
    if !line.ends_with(" .") {
        return Err("N-Triples line must end with ' .'".to_string());
    }

    let line = &line[..line.len() - 2];
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 3 {
        return Err("N-Triples line must have at least 3 parts".to_string());
    }

    let subject = parts[0].to_string();
    let predicate = parts[1].to_string();
    let object = parts[2..].join(" ");

    Ok(Some(LoadTriple {
        subject,
        predicate,
        object,
        graph: None,
    }))
}

/// Parse N-Quads line for loading
fn parse_nquads_line_for_loading(line: &str) -> Result<Option<LoadTriple>, String> {
    if !line.ends_with(" .") {
        return Err("N-Quads line must end with ' .'".to_string());
    }

    let line = &line[..line.len() - 2];
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 3 {
        return Err("N-Quads line must have at least 3 parts".to_string());
    }

    let subject = parts[0].to_string();
    let predicate = parts[1].to_string();

    // Handle potential graph component
    let (object, graph) = if parts.len() >= 4 && parts[parts.len() - 1].starts_with('<') {
        // Last part might be a graph URI
        let obj_parts = &parts[2..parts.len() - 1];
        let object = obj_parts.join(" ");
        let graph = Some(parts[parts.len() - 1].to_string());
        (object, graph)
    } else {
        let object = parts[2..].join(" ");
        (object, None)
    };

    Ok(Some(LoadTriple {
        subject,
        predicate,
        object,
        graph,
    }))
}

/// Parse Turtle content for loading (simplified)
fn parse_turtle_for_loading(content: &str) -> Result<Vec<LoadTriple>, String> {
    let mut triples = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('@') {
            continue;
        }

        if let Some(line) = line.strip_suffix(" .") {
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() >= 3 {
                triples.push(LoadTriple {
                    subject: parts[0].to_string(),
                    predicate: parts[1].to_string(),
                    object: parts[2..].join(" "),
                    graph: None,
                });
            }
        }
    }

    Ok(triples)
}

/// Store triples in the TDB dataset
fn store_triples_in_dataset(
    dataset_location: &Path,
    triples: &[LoadTriple],
    _default_graph: Option<&str>,
) -> ToolResult<usize> {
    use oxirs_tdb::store::TdbStore;

    println!("    Storing {} triples in dataset...", triples.len());

    // Open or create TDB dataset
    let mut store =
        TdbStore::open(dataset_location).map_err(|e| format!("Failed to open TDB store: {}", e))?;

    // Convert all triples to Term format for bulk insertion
    let mut term_triples = Vec::new();
    for load_triple in triples {
        // Parse subject
        let subject = parse_term(&load_triple.subject)
            .map_err(|e| format!("Invalid subject '{}': {}", load_triple.subject, e))?;

        // Parse predicate
        let predicate = parse_term(&load_triple.predicate)
            .map_err(|e| format!("Invalid predicate '{}': {}", load_triple.predicate, e))?;

        // Parse object
        let object = parse_term(&load_triple.object)
            .map_err(|e| format!("Invalid object '{}': {}", load_triple.object, e))?;

        term_triples.push((subject, predicate, object));
    }

    // Use bulk insertion for better performance
    let stored_count = term_triples.len();
    store
        .insert_triples_bulk(&term_triples)
        .map_err(|e| format!("Failed to insert triples: {}", e))?;

    Ok(stored_count)
}

/// Parse a term string into TDB Term
fn parse_term(term_str: &str) -> Result<oxirs_tdb::dictionary::Term, String> {
    use oxirs_tdb::dictionary::Term;

    let term_str = term_str.trim();

    // IRI: <http://example.org/resource>
    if term_str.starts_with('<') && term_str.ends_with('>') {
        let iri = &term_str[1..term_str.len() - 1];
        return Ok(Term::Iri(iri.to_string()));
    }

    // Blank node: _:b1
    if let Some(id) = term_str.strip_prefix("_:") {
        return Ok(Term::BlankNode(id.to_string()));
    }

    // Literal: "value" or "value"@lang or "value"^^<datatype>
    if term_str.starts_with('"') {
        return parse_literal(term_str);
    }

    Err(format!("Invalid term format: {}", term_str))
}

/// Parse a literal term
fn parse_literal(literal_str: &str) -> Result<oxirs_tdb::dictionary::Term, String> {
    use oxirs_tdb::dictionary::Term;
    // Find the closing quote
    let mut in_escape = false;
    let mut quote_pos = None;

    for (i, ch) in literal_str.chars().enumerate().skip(1) {
        if in_escape {
            in_escape = false;
            continue;
        }

        if ch == '\\' {
            in_escape = true;
            continue;
        }

        if ch == '"' {
            quote_pos = Some(i);
            break;
        }
    }

    let quote_pos = quote_pos.ok_or_else(|| "Unclosed literal quote".to_string())?;
    let value = &literal_str[1..quote_pos];

    // Unescape the value
    let value = value
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\"", "\"")
        .replace("\\\\", "\\");

    let rest = &literal_str[quote_pos + 1..];

    // Check for language tag: @lang
    if let Some(lang_tag) = rest.strip_prefix('@') {
        let lang = lang_tag.trim();
        return Ok(Term::Literal {
            value,
            language: Some(lang.to_string()),
            datatype: None,
        });
    }

    // Check for datatype: ^^<datatype>
    if let Some(datatype_str) = rest.strip_prefix("^^") {
        if datatype_str.starts_with('<') && datatype_str.ends_with('>') {
            let datatype = &datatype_str[1..datatype_str.len() - 1];
            return Ok(Term::Literal {
                value,
                language: None,
                datatype: Some(datatype.to_string()),
            });
        }
        return Err(format!("Invalid datatype format: {}", datatype_str));
    }

    // Plain literal (xsd:string)
    Ok(Term::Literal {
        value,
        language: None,
        datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
    })
}

/// Print dataset statistics
fn print_dataset_statistics(dataset_location: &PathBuf) -> ToolResult<()> {
    use oxirs_tdb::store::TdbStore;
    use std::collections::HashSet;

    println!("\n=== Dataset Statistics ===");
    println!("Location: {}", dataset_location.display());

    // Open TDB store
    let store =
        TdbStore::open(dataset_location).map_err(|e| format!("Failed to open TDB store: {}", e))?;

    // Query dataset statistics
    let triples = store
        .query_triples(None, None, None)
        .map_err(|e| format!("Failed to query triples: {}", e))?;

    let triple_count = triples.len();

    // Count unique subjects, predicates, and objects
    let mut subjects = HashSet::new();
    let mut predicates = HashSet::new();
    let mut objects = HashSet::new();

    for (s, p, o) in &triples {
        subjects.insert(format!("{:?}", s));
        predicates.insert(format!("{:?}", p));
        objects.insert(format!("{:?}", o));
    }

    println!("Total triples: {}", format_number(triple_count));
    println!("Unique subjects: {}", format_number(subjects.len()));
    println!("Unique predicates: {}", format_number(predicates.len()));
    println!("Unique objects: {}", format_number(objects.len()));

    // Calculate disk usage
    if let Ok(metadata) = fs::metadata(dataset_location) {
        if metadata.is_dir() {
            let mut total_size = 0u64;
            if let Ok(entries) = fs::read_dir(dataset_location) {
                for entry in entries.flatten() {
                    if let Ok(meta) = entry.metadata() {
                        total_size += meta.len();
                    }
                }
            }
            println!("Disk usage: {}", utils::format_file_size(total_size));
        }
    }

    // Calculate compression ratio if possible
    if triple_count > 0 {
        // Estimate uncompressed size (rough estimate: 150 bytes per triple on average)
        let estimated_uncompressed = (triple_count as u64) * 150;

        if let Ok(metadata) = fs::metadata(dataset_location) {
            if metadata.is_dir() {
                let mut total_size = 0u64;
                if let Ok(entries) = fs::read_dir(dataset_location) {
                    for entry in entries.flatten() {
                        if let Ok(meta) = entry.metadata() {
                            total_size += meta.len();
                        }
                    }
                }

                if total_size > 0 {
                    let ratio = estimated_uncompressed as f64 / total_size as f64;
                    println!("Estimated compression ratio: {:.2}x", ratio);
                }
            }
        }
    }

    println!("==========================");

    Ok(())
}

/// Format number with thousands separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();

    for (count, c) in s.chars().rev().enumerate() {
        if count > 0 && count % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }

    result
}
