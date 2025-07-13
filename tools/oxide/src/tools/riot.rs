//! Riot - RDF parsing and serialization tool
//!
//! Equivalent to Apache Jena's riot command. Parses RDF files and converts
//! between different RDF serialization formats.

use super::{utils, ToolResult, ToolStats};
use std::path::{Path, PathBuf};

/// Run riot command - RDF parsing and serialization
pub async fn run(
    input: Vec<PathBuf>,
    output_format: String,
    output_file: Option<PathBuf>,
    syntax: Option<String>,
    base: Option<String>,
    validate: bool,
    count: bool,
) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("RDF I/O Tool (riot)");
    println!("Input files: {}", input.len());
    println!("Output format: {output_format}");

    if !utils::is_supported_output_format(&output_format) {
        return Err(format!(
            "Unsupported output format '{output_format}'. Supported: turtle, ntriples, rdfxml, jsonld, trig, nquads"
        ).into());
    }

    if let Some(ref base_uri) = base {
        println!("Base URI: {base_uri}");
        // Validate base URI
        utils::validate_iri(base_uri).map_err(|e| format!("Invalid base URI: {e}"))?;
    }

    if validate {
        println!("Mode: Validation only");
    } else if count {
        println!("Mode: Count triples/quads");
    } else {
        println!("Mode: Convert to {output_format}");
    }

    let mut total_triples = 0;
    let mut total_errors = 0;
    let mut all_output = String::new();

    for (i, input_file) in input.iter().enumerate() {
        println!(
            "\nProcessing file {}/{}: {}",
            i + 1,
            input.len(),
            input_file.display()
        );

        // Check file readability
        utils::check_file_readable(input_file)?;

        // Detect input format
        let input_format = syntax
            .clone()
            .unwrap_or_else(|| utils::detect_rdf_format(input_file));
        println!("  Input format: {input_format}");

        if !utils::is_supported_input_format(&input_format) {
            eprintln!("  Warning: Unsupported input format '{input_format}', skipping");
            stats.warnings += 1;
            continue;
        }

        // Read and process file
        match process_rdf_file(
            input_file,
            &input_format,
            &output_format,
            base.as_deref(),
            validate,
            count,
        ) {
            Ok(result) => {
                total_triples += result.triple_count;
                stats.items_processed += result.triple_count;

                if result.errors > 0 {
                    total_errors += result.errors;
                    stats.errors += result.errors;
                    eprintln!("  Errors in file: {}", result.errors);
                }

                println!("  Triples/Quads: {}", result.triple_count);

                if !validate && !count && !result.output.is_empty() {
                    if input.len() > 1 {
                        // Multiple files - add separator comment
                        all_output.push_str(&format!("# File: {}\n", input_file.display()));
                    }
                    all_output.push_str(&result.output);
                    all_output.push('\n');
                }
            }
            Err(e) => {
                eprintln!("  Error processing file: {e}");
                stats.errors += 1;
                continue;
            }
        }
    }

    // Output results
    if validate {
        println!("\nValidation Results:");
        println!("  Total triples/quads: {total_triples}");
        if total_errors > 0 {
            println!("  Total errors: {total_errors}");
            return Err(format!("Validation failed with {total_errors} errors").into());
        } else {
            println!("  All files are valid");
        }
    } else if count {
        println!("\nCount Results:");
        println!("  Total triples/quads: {total_triples}");
    } else {
        // Write output
        if !all_output.is_empty() {
            utils::write_output(&all_output, output_file.as_deref())?;

            if let Some(ref output_path) = output_file {
                println!("Output written to: {}", output_path.display());
            }
        }
    }

    stats.finish();
    stats.print_summary("Riot");

    Ok(())
}

/// Result of processing an RDF file
struct ProcessResult {
    triple_count: usize,
    errors: usize,
    output: String,
}

/// Process a single RDF file
fn process_rdf_file(
    file_path: &Path,
    input_format: &str,
    output_format: &str,
    base_uri: Option<&str>,
    validate_only: bool,
    count_only: bool,
) -> ToolResult<ProcessResult> {
    // Read file content
    let content = utils::read_input(file_path)?;

    // Parse RDF content
    let (triples, errors) = parse_rdf_content(&content, input_format, base_uri)?;

    if validate_only || count_only {
        return Ok(ProcessResult {
            triple_count: triples.len(),
            errors,
            output: String::new(),
        });
    }

    // Serialize to output format
    let output = serialize_rdf_content(&triples, output_format)?;

    Ok(ProcessResult {
        triple_count: triples.len(),
        errors,
        output,
    })
}

/// Simple RDF triple representation
#[derive(Debug, Clone)]
struct RdfTriple {
    subject: String,
    predicate: String,
    object: String,
    graph: Option<String>, // For quads
}

/// Parse RDF content and return triples/quads
fn parse_rdf_content(
    content: &str,
    format: &str,
    base_uri: Option<&str>,
) -> ToolResult<(Vec<RdfTriple>, usize)> {
    let mut triples = Vec::new();
    let mut errors = 0;

    match format {
        "ntriples" => {
            // Parse N-Triples format
            for (line_num, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                match parse_ntriples_line(line, base_uri) {
                    Ok(Some(triple)) => triples.push(triple),
                    Ok(None) => {} // Empty line or comment
                    Err(e) => {
                        eprintln!("    Line {}: {}", line_num + 1, e);
                        errors += 1;
                    }
                }
            }
        }
        "nquads" => {
            // Parse N-Quads format
            for (line_num, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                match parse_nquads_line(line, base_uri) {
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
            // Basic Turtle parsing (simplified)
            match parse_turtle_content(content, base_uri) {
                Ok(parsed_triples) => triples.extend(parsed_triples),
                Err(e) => {
                    eprintln!("    Turtle parsing error: {e}");
                    errors += 1;
                }
            }
        }
        _ => {
            return Err(format!("Parsing for format '{format}' not yet implemented").into());
        }
    }

    Ok((triples, errors))
}

/// Parse a single N-Triples line
fn parse_ntriples_line(line: &str, _base_uri: Option<&str>) -> Result<Option<RdfTriple>, String> {
    if !line.ends_with(" .") {
        return Err("N-Triples line must end with ' .'".to_string());
    }

    let line = &line[..line.len() - 2]; // Remove " ."
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 3 {
        return Err("N-Triples line must have at least 3 parts".to_string());
    }

    let subject = parse_ntriples_term(parts[0])?;
    let predicate = parse_ntriples_term(parts[1])?;
    let object = parse_ntriples_term(&parts[2..].join(" "))?;

    Ok(Some(RdfTriple {
        subject,
        predicate,
        object,
        graph: None,
    }))
}

/// Parse a single N-Quads line
fn parse_nquads_line(line: &str, _base_uri: Option<&str>) -> Result<Option<RdfTriple>, String> {
    if !line.ends_with(" .") {
        return Err("N-Quads line must end with ' .'".to_string());
    }

    let line = &line[..line.len() - 2]; // Remove " ."
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 3 {
        return Err("N-Quads line must have at least 3 parts".to_string());
    }

    let subject = parse_ntriples_term(parts[0])?;
    let predicate = parse_ntriples_term(parts[1])?;

    // For N-Quads, we need to handle the case where object might be multiple tokens (for literals)
    // and there might be a graph component
    let mut object_parts = Vec::new();
    let mut graph = None;
    let mut in_literal = false;

    for (i, part) in parts[2..].iter().enumerate() {
        if part.starts_with('"') && !in_literal {
            in_literal = true;
            object_parts.push(*part);
        } else if part.ends_with('"') && in_literal {
            object_parts.push(*part);
            // in_literal = false; // Not needed as we break

            // Check if there's a graph component after the object
            if i + 3 < parts.len() - 2 {
                // parts[2..] indexing
                let potential_graph = parts[i + 3];
                if potential_graph.starts_with('<') && potential_graph.ends_with('>') {
                    graph = Some(parse_ntriples_term(potential_graph)?);
                }
            }
            break;
        } else if in_literal {
            object_parts.push(*part);
        } else if part.starts_with('<') && part.ends_with('>') {
            // IRI object or graph
            if object_parts.is_empty() {
                object_parts.push(*part);
                // Check if there's a graph after this
                if i + 3 < parts.len() - 2 {
                    let potential_graph = parts[i + 3];
                    if potential_graph.starts_with('<') && potential_graph.ends_with('>') {
                        graph = Some(parse_ntriples_term(potential_graph)?);
                    }
                }
                break;
            } else {
                // This is probably the graph
                graph = Some(parse_ntriples_term(part)?);
                break;
            }
        } else {
            object_parts.push(*part);
        }
    }

    if object_parts.is_empty() {
        return Err("No object found in N-Quads line".to_string());
    }

    let object = parse_ntriples_term(&object_parts.join(" "))?;

    Ok(Some(RdfTriple {
        subject,
        predicate,
        object,
        graph,
    }))
}

/// Parse N-Triples term (IRI, blank node, or literal)
fn parse_ntriples_term(term: &str) -> Result<String, String> {
    let term = term.trim();

    if term.starts_with('<') && term.ends_with('>') {
        // IRI
        let iri = &term[1..term.len() - 1];
        utils::validate_iri(iri)?;
        Ok(term.to_string())
    } else if term.starts_with("_:") {
        // Blank node
        Ok(term.to_string())
    } else if term.starts_with('"') {
        // Literal (simplified parsing)
        Ok(term.to_string())
    } else {
        Err(format!("Invalid N-Triples term: {term}"))
    }
}

/// Basic Turtle parsing (very simplified)
fn parse_turtle_content(content: &str, _base_uri: Option<&str>) -> Result<Vec<RdfTriple>, String> {
    // This is a very basic Turtle parser - a real implementation would be much more complex
    let mut triples = Vec::new();
    let _current_subject: Option<String> = None;
    let _current_predicate: Option<String> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('@') {
            continue;
        }

        // Very basic parsing - just handle simple subject predicate object . patterns
        if let Some(line) = line.strip_suffix(" .") {
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() >= 3 {
                let subject = parts[0].to_string();
                let predicate = parts[1].to_string();
                let object = parts[2..].join(" ");

                triples.push(RdfTriple {
                    subject,
                    predicate,
                    object,
                    graph: None,
                });
            }
        }
    }

    Ok(triples)
}

/// Serialize RDF triples to specified format
fn serialize_rdf_content(triples: &[RdfTriple], format: &str) -> ToolResult<String> {
    let mut output = String::new();

    match format {
        "ntriples" => {
            for triple in triples {
                output.push_str(&format!(
                    "{} {} {} .\n",
                    triple.subject, triple.predicate, triple.object
                ));
            }
        }
        "nquads" => {
            for triple in triples {
                if let Some(ref graph) = triple.graph {
                    output.push_str(&format!(
                        "{} {} {} {} .\n",
                        triple.subject, triple.predicate, triple.object, graph
                    ));
                } else {
                    output.push_str(&format!(
                        "{} {} {} .\n",
                        triple.subject, triple.predicate, triple.object
                    ));
                }
            }
        }
        "turtle" => {
            // Basic Turtle serialization
            output.push_str("# Generated by oxide riot\n\n");
            for triple in triples {
                output.push_str(&format!(
                    "{} {} {} .\n",
                    triple.subject, triple.predicate, triple.object
                ));
            }
        }
        _ => {
            return Err(format!("Serialization for format '{format}' not yet implemented").into());
        }
    }

    Ok(output)
}
