//! RDF parsers for OxiRS WASM
//!
//! This module provides two parser implementations:
//! - [`streaming_parser`]: Incremental/streaming parser for Turtle, N-Triples, N-Quads
//! - Legacy flat-file parsers for backward compatibility (re-exported from the old parser.rs)

pub mod streaming_parser;

pub use streaming_parser::{ParseError, ParsedStatement, ParsedTerm, RdfFormat, StreamingParser};

use crate::error::{WasmError, WasmResult};
use std::collections::HashMap;

/// Internal triple representation used throughout the WASM crate.
/// Subject, predicate, and object are serialized as plain strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl ParsedTriple {
    /// Construct from a [`ParsedStatement`] (triple variant only)
    pub fn from_statement(stmt: &ParsedStatement) -> Self {
        Self {
            subject: stmt.subject().to_ntriples_string(),
            predicate: stmt.predicate().to_ntriples_string(),
            object: stmt.object().to_ntriples_string(),
        }
    }

    /// Raw subject value (IRI without brackets, literal without quotes)
    pub fn subject_value(&self) -> String {
        term_value_from_ntriples_str(&self.subject)
    }

    /// Raw predicate value
    pub fn predicate_value(&self) -> String {
        term_value_from_ntriples_str(&self.predicate)
    }

    /// Raw object value
    pub fn object_value(&self) -> String {
        term_value_from_ntriples_str(&self.object)
    }
}

/// Strip N-Triples angle brackets or quotes from a serialized term string
fn term_value_from_ntriples_str(s: &str) -> String {
    if s.starts_with('<') && s.ends_with('>') {
        s[1..s.len() - 1].to_string()
    } else if s.starts_with('"') {
        // Strip surrounding quotes for plain literals
        if let Some(end) = s.rfind('"') {
            if end > 0 {
                return s[1..end].to_string();
            }
        }
        s.to_string()
    } else {
        s.to_string()
    }
}

/// Parse Turtle format using the streaming parser
pub fn parse_turtle(turtle: &str) -> WasmResult<Vec<ParsedTriple>> {
    let mut parser = StreamingParser::new(RdfFormat::Turtle);
    let mut stmts = parser
        .feed(turtle)
        .map_err(|e| WasmError::ParseError(e.to_string()))?;
    let mut tail = parser
        .finish()
        .map_err(|e| WasmError::ParseError(e.to_string()))?;
    stmts.append(&mut tail);

    let triples = stmts.iter().map(ParsedTriple::from_statement).collect();

    Ok(triples)
}

/// Parse N-Triples format using the streaming parser
pub fn parse_ntriples(ntriples: &str) -> WasmResult<Vec<ParsedTriple>> {
    let mut parser = StreamingParser::new(RdfFormat::NTriples);
    let mut stmts = parser
        .feed(ntriples)
        .map_err(|e| WasmError::ParseError(e.to_string()))?;
    let mut tail = parser
        .finish()
        .map_err(|e| WasmError::ParseError(e.to_string()))?;
    stmts.append(&mut tail);

    let triples = stmts.iter().map(ParsedTriple::from_statement).collect();

    Ok(triples)
}

/// Build a prefix map from a Turtle document (for store use)
pub fn extract_prefixes(turtle: &str) -> HashMap<String, String> {
    let mut parser = StreamingParser::new(RdfFormat::Turtle);
    let _ = parser.feed(turtle);
    parser.prefixes().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ntriples_compatibility() {
        let nt = r#"
<http://example.org/s> <http://example.org/p> <http://example.org/o> .
<http://example.org/s> <http://example.org/name> "Alice" .
"#;
        let triples = parse_ntriples(nt).expect("parse should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_parse_turtle_compatibility() {
        let ttl = r#"
@prefix : <http://example.org/> .
:alice :knows :bob .
:bob :name "Bob" .
"#;
        let triples = parse_turtle(ttl).expect("parse should succeed");
        assert!(triples.len() >= 2);
    }
}
