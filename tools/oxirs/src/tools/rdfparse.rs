//! RDF Parse tool
//!
//! Parse an RDF file and report triple/quad count with any parse errors.
//! Supports Turtle, N-Triples, N-Quads, TriG and N3 via oxirs-ttl parsers.

use super::{utils, ToolResult};
use oxirs_ttl::formats::nquads::NQuadsParser;
use oxirs_ttl::formats::ntriples::NTriplesParser;
use oxirs_ttl::formats::trig::TriGParser;
use oxirs_ttl::toolkit::Parser;
use oxirs_ttl::turtle::TurtleParser;
use std::io::Cursor;
use std::path::PathBuf;

/// Run rdfparse command — report triple count and parse errors.
pub async fn run(_file: PathBuf, _format: Option<String>, _base: Option<String>) -> ToolResult {
    utils::check_file_readable(&_file)?;

    let content = utils::read_input(&_file)?;

    let format = _format
        .clone()
        .unwrap_or_else(|| utils::detect_rdf_format(&_file));

    println!("File   : {}", _file.display());
    println!("Format : {format}");
    if let Some(ref base) = _base {
        println!("Base   : {base}");
    }
    println!();

    let summary = parse_and_count(&content, &format, _base.as_deref())?;

    println!("Triples/Quads : {}", summary.triple_count);
    println!("Parse errors  : {}", summary.error_count);

    if summary.error_count > 0 {
        for err in &summary.errors {
            eprintln!("  error: {err}");
        }
        return Err(format!("Parse completed with {} error(s)", summary.error_count).into());
    }

    println!("Parse OK");
    Ok(())
}

/// Summary returned by the inner parsing logic.
struct ParseSummary {
    triple_count: usize,
    error_count: usize,
    errors: Vec<String>,
}

/// Parse RDF content and collect statistics.
fn parse_and_count(
    content: &str,
    format: &str,
    base_iri: Option<&str>,
) -> ToolResult<ParseSummary> {
    match format {
        "turtle" | "ttl" => parse_turtle(content, base_iri),
        "ntriples" | "nt" => parse_ntriples(content),
        "nquads" | "nq" => parse_nquads(content),
        "trig" => parse_trig(content),
        "n3" => {
            // N3 shares the Turtle parser for our purposes.
            parse_turtle(content, base_iri)
        }
        other => Err(format!("Unsupported format for rdfparse: '{other}'").into()),
    }
}

fn parse_turtle(content: &str, base_iri: Option<&str>) -> ToolResult<ParseSummary> {
    let mut parser = TurtleParser::new_lenient();
    if let Some(base) = base_iri {
        parser = parser.with_base_iri(base.to_string());
    }

    match parser.parse_document(content) {
        Ok(triples) => Ok(ParseSummary {
            triple_count: triples.len(),
            error_count: 0,
            errors: Vec::new(),
        }),
        Err(e) => Ok(ParseSummary {
            triple_count: 0,
            error_count: 1,
            errors: vec![e.to_string()],
        }),
    }
}

fn parse_ntriples(content: &str) -> ToolResult<ParseSummary> {
    let parser = NTriplesParser::new_lenient();
    let mut triple_count = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        match parser.parse_line(line, idx + 1) {
            Ok(Some(_)) => triple_count += 1,
            Ok(None) => {}
            Err(e) => errors.push(format!("line {}: {e}", idx + 1)),
        }
    }

    let error_count = errors.len();
    Ok(ParseSummary {
        triple_count,
        error_count,
        errors,
    })
}

fn parse_nquads(content: &str) -> ToolResult<ParseSummary> {
    let parser = NQuadsParser::new();
    match parser.parse(Cursor::new(content)) {
        Ok(quads) => Ok(ParseSummary {
            triple_count: quads.len(),
            error_count: 0,
            errors: Vec::new(),
        }),
        Err(e) => Ok(ParseSummary {
            triple_count: 0,
            error_count: 1,
            errors: vec![e.to_string()],
        }),
    }
}

fn parse_trig(content: &str) -> ToolResult<ParseSummary> {
    let parser = TriGParser::new();
    match parser.parse(Cursor::new(content)) {
        Ok(quads) => Ok(ParseSummary {
            triple_count: quads.len(),
            error_count: 0,
            errors: Vec::new(),
        }),
        Err(e) => Ok(ParseSummary {
            triple_count: 0,
            error_count: 1,
            errors: vec![e.to_string()],
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ntriples_valid() {
        let content = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let summary = parse_ntriples(content).expect("parse should succeed");
        assert_eq!(summary.triple_count, 1);
        assert_eq!(summary.error_count, 0);
    }

    #[test]
    fn test_parse_ntriples_with_error() {
        let content = "this is not valid ntriples\n";
        let summary = parse_ntriples(content).expect("function should not propagate error");
        assert_eq!(summary.error_count, 1);
    }

    #[test]
    fn test_parse_turtle_valid() {
        let content =
            "@prefix ex: <http://example.org/> .\nex:s ex:p \"hello\" .\nex:s ex:p2 ex:o .\n";
        let summary = parse_turtle(content, None).expect("parse should succeed");
        assert_eq!(summary.triple_count, 2);
        assert_eq!(summary.error_count, 0);
    }

    #[test]
    fn test_parse_nquads_valid() {
        let content = "<http://s> <http://p> <http://o> <http://g> .\n";
        let summary = parse_nquads(content).expect("parse should succeed");
        assert_eq!(summary.triple_count, 1);
        assert_eq!(summary.error_count, 0);
    }

    #[test]
    fn test_unsupported_format() {
        let result = parse_and_count("", "rdfxml", None);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_missing_file() {
        let result = run(
            std::path::PathBuf::from("/nonexistent/file.ttl"),
            None,
            None,
        )
        .await;
        assert!(result.is_err());
    }
}
