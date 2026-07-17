//! N3 format parser
//!
//! N3 (Notation3) is a superset of Turtle that adds support for variables,
//! rules, and formulae. This implementation provides basic support for the
//! Turtle subset of N3: it delegates actual parsing to [`AdvancedN3Parser`]
//! (a real tokenizer/grammar-based parser, see [`crate::formats::n3_parser`])
//! and exposes only the concrete, variable-free statements as plain RDF
//! `Triple`s.
//!
//! Earlier versions of this parser used a line-by-line heuristic that (a)
//! silently dropped any line containing `=` — including perfectly ordinary
//! IRIs with query strings like `?a=b` — under the guise of "skipping N3
//! rules", and (b) could not parse a statement split across multiple lines
//! (e.g. a `;`-continued predicate/object list), silently losing data. Both
//! are why this now routes through the real N3 grammar instead.

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::formats::n3_parser::AdvancedN3Parser;
use crate::toolkit::Parser;
use oxirs_core::model::Triple;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

/// N3 parser with basic Turtle subset support
#[derive(Debug, Clone)]
pub struct N3Parser {
    /// Whether to continue parsing after errors
    pub lenient: bool,
    /// Base IRI for resolving relative IRIs
    pub base_iri: Option<String>,
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
}

impl Default for N3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl N3Parser {
    /// Create a new N3 parser
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add standard prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );

        Self {
            lenient: false,
            base_iri: None,
            prefixes,
        }
    }

    /// Create a lenient N3 parser that continues after errors
    pub fn new_lenient() -> Self {
        Self {
            lenient: true,
            ..Self::new()
        }
    }

    /// Parse N3 content covering the Turtle subset.
    ///
    /// Delegates the actual tokenization/grammar work to [`AdvancedN3Parser`]
    /// so that multi-line statements, `{}` formulas, `=>`/`<=` implications,
    /// and ordinary IRIs containing `=` (e.g. query strings) are all handled
    /// correctly instead of being misparsed by a line-oriented heuristic.
    ///
    /// Only concrete, variable-free top-level statements are returned as
    /// plain RDF `Triple`s (the "basic Turtle subset" this parser exposes).
    /// A statement that uses N3-specific features requiring a variable
    /// (`?x`) or a nested formula (`{ ... }`) as one of its terms cannot be
    /// represented as a plain `Triple`: in strict mode that is a parse
    /// error, in lenient mode the statement is skipped. Implications
    /// (`=>`/`<=`) are parsed (so they no longer break the document) but,
    /// having no triple form at all, are always dropped here — callers that
    /// need full N3 semantics (variables, formulas, rules) should use
    /// [`AdvancedN3Parser`] directly instead of this basic-subset wrapper.
    fn parse_n3_content<R: BufRead>(&self, mut reader: R) -> TurtleResult<Vec<Triple>> {
        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .map_err(TurtleParseError::io)?;

        let mut advanced = AdvancedN3Parser::new(&content)?;
        advanced.lenient = self.lenient;
        // Overlay this parser's prefixes (which may override the standard
        // rdf/rdfs/xsd/owl defaults AdvancedN3Parser also seeds itself with)
        // on top rather than only filling gaps, so an explicit user override
        // always wins.
        for (prefix, iri) in &self.prefixes {
            advanced.prefixes.insert(prefix.clone(), iri.clone());
        }
        if let Some(base) = &self.base_iri {
            advanced.base_iri = Some(base.clone());
        }

        let doc = advanced.parse_document()?;

        let mut triples = Vec::with_capacity(doc.statements.len());
        for statement in doc.statements {
            match statement.as_rdf_triple() {
                Some(triple) => triples.push(triple),
                None => {
                    if !self.lenient {
                        return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                            message:
                                "N3 statement uses a variable or formula and cannot be represented \
                                 as a plain RDF triple; use AdvancedN3Parser for full N3 support"
                                    .to_string(),
                            position: TextPosition::default(),
                        }));
                    }
                    // Lenient mode: skip statements that aren't representable
                    // as a concrete triple rather than failing the parse.
                }
            }
        }

        Ok(triples)
    }
}

impl Parser<Triple> for N3Parser {
    fn parse<R: Read>(&self, reader: R) -> TurtleResult<Vec<Triple>> {
        let buf_reader = BufReader::new(reader);
        self.parse_n3_content(buf_reader)
    }

    /// Parse the given reader and expose the resulting triples as an
    /// iterator.
    ///
    /// Like [`crate::formats::trig::TriGParser::for_reader`], this still
    /// parses the entire document eagerly via [`Self::parse_n3_content`]
    /// before returning; it exists to satisfy the generic [`Parser`] trait
    /// uniformly across formats, not to provide bounded-memory streaming.
    fn for_reader<R: BufRead>(&self, reader: R) -> Box<dyn Iterator<Item = TurtleResult<Triple>>> {
        match self.parse_n3_content(reader) {
            Ok(triples) => Box::new(triples.into_iter().map(Ok)),
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    }
}

/// N3 streaming iterator
pub struct N3Iterator {
    triples: std::vec::IntoIter<Triple>,
}

impl N3Iterator {
    /// Creates a new N3 iterator from a reader and parser
    pub fn new<R: BufRead>(reader: R, parser: &N3Parser) -> TurtleResult<Self> {
        let triples = parser.parse_n3_content(reader)?;
        Ok(Self {
            triples: triples.into_iter(),
        })
    }
}

impl Iterator for N3Iterator {
    type Item = TurtleResult<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        self.triples.next().map(Ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(input: &str) -> TurtleResult<Vec<Triple>> {
        N3Parser::new().parse(Cursor::new(input.as_bytes()))
    }

    #[test]
    fn test_basic_triple() {
        let triples = parse("@prefix ex: <http://example.org/> .\nex:alice ex:knows ex:bob .\n")
            .expect("parsing should succeed");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_iri_with_equals_sign_is_not_dropped() {
        // Regression test: the old line-based parser treated *any* line
        // containing '=' as an N3 rule/implication and silently dropped it,
        // even for a perfectly ordinary IRI with a query string.
        let triples = parse(
            "@prefix ex: <http://example.org/> .\n\
             ex:alice ex:seeAlso <http://example.org/search?a=b&c=d> .\n",
        )
        .expect("parsing should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0].object().to_string(),
            "<http://example.org/search?a=b&c=d>"
        );
    }

    #[test]
    fn test_multiline_semicolon_statement_is_not_lost() {
        // Regression test: the old line-based parser required each line to
        // independently look like a complete `s p o .` triple, so a
        // continuation line after a `;` (very common pretty-printed Turtle)
        // silently produced zero triples for that statement.
        let triples = parse(
            "@prefix ex: <http://example.org/> .\n\
             ex:alice\n  ex:name \"Alice\" ;\n  ex:knows ex:bob .\n",
        )
        .expect("parsing should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_implication_does_not_break_surrounding_triples() {
        // An implication has no plain-triple form, so it contributes zero
        // triples here, but it must not prevent the rest of the document
        // (including triples before and after it) from being parsed.
        let triples = parse(
            "@prefix ex: <http://example.org/> .\n\
             ex:alice ex:knows ex:bob .\n\
             { ?x ex:knows ?y } => { ?y ex:knows ?x } .\n\
             ex:carol ex:knows ex:dave .\n",
        )
        .expect("parsing should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_implication_is_error_in_strict_mode_without_variables() {
        // A statement using a variable cannot become a plain Triple; in
        // strict (non-lenient) mode this must be a clear error rather than
        // silently vanishing.
        let strict = N3Parser::new();
        let result = strict.parse(Cursor::new(
            b"@prefix ex: <http://example.org/> .\n?x ex:knows ex:bob .\n".as_slice(),
        ));
        assert!(result.is_err());
    }

    #[test]
    fn test_variable_statement_skipped_in_lenient_mode() {
        let lenient = N3Parser::new_lenient();
        let triples = lenient
            .parse(Cursor::new(
                b"@prefix ex: <http://example.org/> .\n?x ex:knows ex:bob .\nex:alice ex:knows ex:bob .\n"
                    .as_slice(),
            ))
            .expect("lenient parsing should succeed");
        assert_eq!(triples.len(), 1);
    }
}
