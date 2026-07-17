//! Riot - RDF parsing and serialization tool
//!
//! Equivalent to Apache Jena's riot command. Parses RDF files and converts
//! between different RDF serialization formats using oxirs-core's real,
//! format-aware parser/serializer (`oxirs_core::format::{RdfParser,
//! RdfFormat}` — the same stack `rdfparse.rs`/`rdfcopy.rs` build on for
//! Turtle/N-Triples, extended here to N-Quads/TriG/RDF-XML/JSON-LD/N3 too).
//! Every parsed quad comes from real content; `--validate` counts one real
//! parse error per malformed statement rather than silently skipping it.

use super::{utils, ToolResult, ToolStats};
use oxirs_core::format::serializer::simple::serialize_quads_to_string;
use oxirs_core::format::{JsonLdProfileSet, RdfFormat as CoreRdfFormat, RdfParser};
use oxirs_core::model::Quad;
use std::io::Cursor;
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

    // Parse RDF content via the real oxirs-core format parser
    let (quads, errors) = parse_rdf_content(&content, input_format, base_uri)?;

    if validate_only || count_only {
        return Ok(ProcessResult {
            triple_count: quads.len(),
            errors,
            output: String::new(),
        });
    }

    // Serialize to output format
    let output = serialize_rdf_content(&quads, output_format)?;

    Ok(ProcessResult {
        triple_count: quads.len(),
        errors,
        output,
    })
}

/// Map a `utils::detect_rdf_format`/`--syntax` string to oxirs-core's
/// parser/serializer-facing `RdfFormat`.
fn tool_format_to_core(format: &str) -> ToolResult<CoreRdfFormat> {
    match format {
        "turtle" | "ttl" => Ok(CoreRdfFormat::Turtle),
        "ntriples" | "nt" => Ok(CoreRdfFormat::NTriples),
        "nquads" | "nq" => Ok(CoreRdfFormat::NQuads),
        "trig" => Ok(CoreRdfFormat::TriG),
        "n3" => Ok(CoreRdfFormat::N3),
        "rdfxml" | "rdf" | "xml" => Ok(CoreRdfFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(CoreRdfFormat::JsonLd {
            profile: JsonLdProfileSet::empty(),
        }),
        other => Err(format!("Parsing for format '{other}' is not supported").into()),
    }
}

/// Parse RDF content and return quads plus a real per-statement error count.
///
/// Unlike the previous ad hoc line parser, malformed statements are never
/// silently dropped: every parse failure yielded by the real oxirs-core
/// format parser increments `errors`, so `--validate` reflects the actual
/// state of the file.
fn parse_rdf_content(
    content: &str,
    format: &str,
    base_uri: Option<&str>,
) -> ToolResult<(Vec<Quad>, usize)> {
    let core_format = tool_format_to_core(format)?;

    let mut parser = RdfParser::new(core_format);
    if let Some(base) = base_uri {
        parser = parser.with_base_iri(base.to_string());
    }

    let mut quads = Vec::new();
    let mut errors = 0usize;

    for (idx, result) in parser
        .for_reader(Cursor::new(content.as_bytes().to_vec()))
        .enumerate()
    {
        match result {
            Ok(quad) => quads.push(quad),
            Err(e) => {
                eprintln!("    Statement {}: {e}", idx + 1);
                errors += 1;
            }
        }
    }

    Ok((quads, errors))
}

/// Serialize RDF quads to the specified format via oxirs-core's real
/// format-aware serializer.
fn serialize_rdf_content(quads: &[Quad], format: &str) -> ToolResult<String> {
    let core_format = tool_format_to_core(format)?;
    serialize_quads_to_string(quads, core_format)
        .map_err(|e| format!("Failed to serialize as '{format}': {e}").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_turtle_no_errors() {
        let content =
            "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\nex:s ex:p2 \"lit\" .\n";
        let (quads, errors) = parse_rdf_content(content, "turtle", None).expect("parse");
        assert_eq!(quads.len(), 2);
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_parse_valid_ntriples_no_errors() {
        let content = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let (quads, errors) = parse_rdf_content(content, "ntriples", None).expect("parse");
        assert_eq!(quads.len(), 1);
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_parse_malformed_ntriples_counts_real_error() {
        // Missing closing '>' on the predicate makes this an invalid statement;
        // the old ad hoc parser silently dropped anything not ending in " .",
        // so it would report 0 errors here. The real parser must count it.
        let content = "<http://example.org/s> <http://example.org/p <http://example.org/o> .\n";
        let (_quads, errors) = parse_rdf_content(content, "ntriples", None).expect("parse call");
        assert!(
            errors > 0,
            "malformed N-Triples statement must be counted as a real error"
        );
    }

    #[test]
    fn test_parse_turtle_prefixed_predicate_object_list_not_dropped() {
        // The old ad hoc Turtle parser only understood single-line
        // `subject predicate object .`; predicate/object lists via `;`/`,`
        // and multi-line statements were silently skipped with no error.
        let content = "@prefix ex: <http://example.org/> .\nex:alice ex:knows ex:bob ;\n    ex:knows ex:carol .\n";
        let (quads, errors) = parse_rdf_content(content, "turtle", None).expect("parse");
        assert_eq!(errors, 0);
        assert_eq!(
            quads.len(),
            2,
            "predicate-object list across multiple lines must yield both real triples, not be silently dropped"
        );
    }

    #[test]
    fn test_serialize_and_reparse_roundtrip() {
        let content = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let (quads, errors) = parse_rdf_content(content, "ntriples", None).expect("parse");
        assert_eq!(errors, 0);

        let serialized = serialize_rdf_content(&quads, "turtle").expect("serialize");
        let (reparsed, reparse_errors) =
            parse_rdf_content(&serialized, "turtle", None).expect("reparse");
        assert_eq!(reparse_errors, 0);
        assert_eq!(reparsed.len(), 1);
    }

    #[test]
    fn test_unsupported_format_errors() {
        assert!(parse_rdf_content("", "bogus-format", None).is_err());
        assert!(serialize_rdf_content(&[], "bogus-format").is_err());
    }
}
