//! RDF Copy tool - Copy RDF datasets with optional format conversion
//!
//! Reads an RDF file (with auto-detected or specified source format) and writes
//! it out in the requested target format, transcoding triples in memory.

use super::{utils, ToolResult};
use oxirs_core::model::Triple;
use oxirs_ttl::formats::nquads::NQuadsParser;
use oxirs_ttl::formats::ntriples::{NTriplesParser, NTriplesSerializer};
use oxirs_ttl::formats::trig::TriGParser;
use oxirs_ttl::toolkit::{Parser, Serializer};
use oxirs_ttl::turtle::{TurtleParser, TurtleSerializer};
use std::io::Cursor;
use std::path::PathBuf;

/// Run rdfcopy command — transcode an RDF file from one format to another.
pub async fn run(
    source: PathBuf,
    target: PathBuf,
    _source_format: Option<String>,
    _target_format: Option<String>,
) -> ToolResult {
    utils::check_file_readable(&source)?;

    let source_fmt = _source_format
        .clone()
        .unwrap_or_else(|| utils::detect_rdf_format(&source));
    let target_fmt = _target_format
        .clone()
        .unwrap_or_else(|| utils::detect_rdf_format(&target));

    println!("Source : {} (format: {source_fmt})", source.display());
    println!("Target : {} (format: {target_fmt})", target.display());

    let content = utils::read_input(&source)?;

    // Parse source into triples
    let triples = parse_to_triples(&content, &source_fmt)?;

    println!("Read   : {} triple(s)", triples.len());

    // Serialize triples to target format
    let serialized = serialize_triples(&triples, &target_fmt)?;

    // Write output
    utils::write_output(&serialized, Some(&target))?;

    println!("Written: {}", target.display());
    Ok(())
}

/// Parse RDF content into a uniform Vec<Triple>, regardless of source format.
fn parse_to_triples(content: &str, format: &str) -> ToolResult<Vec<Triple>> {
    match format {
        "turtle" | "ttl" | "n3" => {
            let parser = TurtleParser::new();
            parser
                .parse_document(content)
                .map_err(|e| format!("Turtle parse error: {e}").into())
        }
        "ntriples" | "nt" => {
            let parser = NTriplesParser::new();
            let mut triples = Vec::new();
            for (idx, line) in content.lines().enumerate() {
                match parser.parse_line(line, idx + 1) {
                    Ok(Some(t)) => triples.push(t),
                    Ok(None) => {}
                    Err(e) => {
                        return Err(format!("N-Triples line {}: {e}", idx + 1).into());
                    }
                }
            }
            Ok(triples)
        }
        "nquads" | "nq" => {
            let parser = NQuadsParser::new();
            let quads = parser
                .parse(Cursor::new(content))
                .map_err(|e| format!("N-Quads parse error: {e}"))?;
            // Convert quads to triples by dropping the graph component.
            Ok(quads
                .into_iter()
                .map(|q| {
                    Triple::new(
                        q.subject().clone(),
                        q.predicate().clone(),
                        q.object().clone(),
                    )
                })
                .collect())
        }
        "trig" => {
            let parser = TriGParser::new();
            let quads = parser
                .parse(Cursor::new(content))
                .map_err(|e| format!("TriG parse error: {e}"))?;
            Ok(quads
                .into_iter()
                .map(|q| {
                    Triple::new(
                        q.subject().clone(),
                        q.predicate().clone(),
                        q.object().clone(),
                    )
                })
                .collect())
        }
        other => Err(format!("Unsupported source format for rdfcopy: '{other}'").into()),
    }
}

/// Serialize triples into a string in the requested target format.
fn serialize_triples(triples: &[Triple], format: &str) -> ToolResult<String> {
    let mut buf = Vec::new();

    match format {
        "turtle" | "ttl" | "n3" => {
            let serializer = TurtleSerializer::new();
            serializer
                .serialize(triples, &mut buf)
                .map_err(|e| format!("Turtle serialize error: {e}"))?;
        }
        "ntriples" | "nt" => {
            let serializer = NTriplesSerializer::new();
            serializer
                .serialize(triples, &mut buf)
                .map_err(|e| format!("N-Triples serialize error: {e}"))?;
        }
        other => {
            return Err(format!("Unsupported target format for rdfcopy: '{other}'").into());
        }
    }

    String::from_utf8(buf).map_err(|e| format!("UTF-8 conversion error: {e}").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_ntriples_to_triples() {
        let content = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let triples = parse_to_triples(content, "ntriples").expect("parse");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_parse_turtle_to_triples() {
        let content = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\n";
        let triples = parse_to_triples(content, "turtle").expect("parse");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_serialize_ntriples() {
        let content = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let triples = parse_to_triples(content, "ntriples").expect("parse");
        let out = serialize_triples(&triples, "ntriples").expect("serialize");
        assert!(out.contains("example.org/s"));
    }

    #[test]
    fn test_unsupported_source() {
        let result = parse_to_triples("", "rdfxml");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_target() {
        let result = serialize_triples(&[], "rdfxml");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_transcode() {
        let src_content = "<http://s> <http://p> <http://o> .\n";
        let mut src = tempfile::NamedTempFile::new().expect("src tempfile");
        write!(src, "{src_content}").expect("write");

        let target_dir = std::env::temp_dir();
        let target_path = target_dir.join("rdfcopy_test_output.ttl");

        let result = run(
            src.path().to_path_buf(),
            target_path.clone(),
            Some("ntriples".to_string()),
            Some("turtle".to_string()),
        )
        .await;

        assert!(result.is_ok(), "rdfcopy should succeed: {result:?}");
        assert!(target_path.exists(), "output file should exist");
        let _ = std::fs::remove_file(target_path);
    }
}
