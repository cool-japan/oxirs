//! N-Triples Format Parser and Serializer
//!
//! Extracted and adapted from OxiGraph oxttl with OxiRS enhancements.
//! Based on W3C N-Triples specification: https://www.w3.org/TR/n-triples/

use super::error::SerializeResult;
use super::error::{ParseResult, RdfParseError, TextPosition};
use super::serializer::QuadSerializer;
use crate::model::{BlankNode, Literal, NamedNode, Triple, TripleRef};
use std::io::{Read, Write};

/// Represents a parsed N-Triples term
#[derive(Debug, Clone)]
enum NTriplesTerm {
    Iri(String),
    BlankNode(String),
    Literal(String),
    LanguageLiteral(String, String),
    TypedLiteral(String, String),
}

/// N-Triples parser implementation
#[derive(Debug, Clone)]
pub struct NTriplesParser {
    lenient: bool,
}

impl NTriplesParser {
    /// Create a new N-Triples parser
    pub fn new() -> Self {
        Self { lenient: false }
    }

    /// Enable lenient parsing (skip some validations)
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Parse N-Triples from a reader
    pub fn parse_reader<R: Read>(&self, _reader: R) -> ParseResult<Vec<Triple>> {
        // TODO: Implement actual N-Triples parsing
        Ok(Vec::new())
    }

    /// Parse N-Triples from a byte slice
    pub fn parse_slice(&self, slice: &[u8]) -> ParseResult<Vec<Triple>> {
        let content = std::str::from_utf8(slice)
            .map_err(|e| RdfParseError::syntax(format!("Invalid UTF-8: {}", e)))?;
        self.parse_str(content)
    }

    /// Parse N-Triples from a string
    pub fn parse_str(&self, input: &str) -> ParseResult<Vec<Triple>> {
        let mut triples = Vec::new();
        let mut line_number = 1;

        for line in input.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                line_number += 1;
                continue;
            }

            // Parse triple
            match self.parse_triple_line(trimmed, line_number) {
                Ok(Some(triple)) => triples.push(triple),
                Ok(None) => {} // Valid but empty line
                Err(e) if self.lenient => {
                    // Skip invalid lines in lenient mode
                    eprintln!("Warning: Skipping invalid line {}: {}", line_number, e);
                }
                Err(e) => return Err(e),
            }

            line_number += 1;
        }

        Ok(triples)
    }

    /// Parse a single triple line
    fn parse_triple_line(&self, line: &str, line_number: usize) -> ParseResult<Option<Triple>> {
        // N-Triples format: <subject> <predicate> <object> .
        // - Subject: IRI or blank node
        // - Predicate: IRI
        // - Object: IRI, blank node, or literal

        // Validate line ends with dot
        if !line.ends_with('.') {
            return Err(RdfParseError::syntax_at(
                "N-Triples line must end with '.'",
                TextPosition::new(line_number, line.len(), 0),
            ));
        }

        // Remove the trailing dot and parse terms
        let line_without_dot = line[..line.len() - 1].trim();

        // Parse the three terms
        let mut terms = Vec::new();
        let mut pos = 0;

        // Parse subject
        let (subject_term, new_pos) = self.parse_term(line_without_dot, pos, line_number)?;
        terms.push(subject_term);
        pos = new_pos;

        // Parse predicate
        let (predicate_term, new_pos) = self.parse_term(line_without_dot, pos, line_number)?;
        terms.push(predicate_term);
        pos = new_pos;

        // Parse object
        let (object_term, _) = self.parse_term(line_without_dot, pos, line_number)?;
        terms.push(object_term);

        if terms.len() != 3 {
            return Err(RdfParseError::syntax_at(
                "N-Triples line must have exactly 3 terms",
                TextPosition::new(line_number, 1, 0),
            ));
        }

        // Build the triple
        let subject = self.term_to_subject(&terms[0], line_number)?;
        let predicate = self.term_to_predicate(&terms[1], line_number)?;
        let object = self.term_to_object(&terms[2], line_number)?;

        Ok(Some(Triple::new(subject, predicate, object)))
    }

    /// Check if lenient parsing is enabled
    pub fn is_lenient(&self) -> bool {
        self.lenient
    }

    /// Parse a single term (IRI, blank node, or literal)
    fn parse_term(
        &self,
        input: &str,
        start_pos: usize,
        line_number: usize,
    ) -> ParseResult<(NTriplesTerm, usize)> {
        let trimmed = input[start_pos..].trim_start();
        let actual_start = start_pos + (input.len() - start_pos - trimmed.len());

        if trimmed.is_empty() {
            return Err(RdfParseError::syntax_at(
                "Expected term but found end of line",
                TextPosition::new(line_number, actual_start, 0),
            ));
        }

        if trimmed.starts_with('<') {
            // Parse IRI
            self.parse_iri(trimmed, actual_start, line_number)
        } else if trimmed.starts_with("_:") {
            // Parse blank node
            self.parse_blank_node(trimmed, actual_start, line_number)
        } else if trimmed.starts_with('"') {
            // Parse literal
            self.parse_literal(trimmed, actual_start, line_number)
        } else {
            Err(RdfParseError::syntax_at(
                "Invalid term format. Expected <IRI>, _:blank, or \"literal\"",
                TextPosition::new(line_number, actual_start, 0),
            ))
        }
    }

    /// Parse an IRI term <...>
    fn parse_iri(
        &self,
        input: &str,
        start_pos: usize,
        line_number: usize,
    ) -> ParseResult<(NTriplesTerm, usize)> {
        if let Some(end_pos) = input.find('>') {
            let iri = input[1..end_pos].to_string();
            let new_pos = start_pos + end_pos + 1;
            Ok((NTriplesTerm::Iri(iri), new_pos))
        } else {
            Err(RdfParseError::syntax_at(
                "Unterminated IRI - missing '>'",
                TextPosition::new(line_number, start_pos, 0),
            ))
        }
    }

    /// Parse a blank node term _:...
    fn parse_blank_node(
        &self,
        input: &str,
        start_pos: usize,
        line_number: usize,
    ) -> ParseResult<(NTriplesTerm, usize)> {
        // Find the end of the blank node ID (whitespace or end)
        let mut end_pos = 2; // Start after _:
        for (i, c) in input[2..].char_indices() {
            if c.is_whitespace() {
                end_pos = 2 + i;
                break;
            }
            end_pos = 2 + i + c.len_utf8();
        }

        let blank_id = input[2..end_pos].to_string();
        if blank_id.is_empty() {
            return Err(RdfParseError::syntax_at(
                "Blank node ID cannot be empty",
                TextPosition::new(line_number, start_pos, 0),
            ));
        }

        let new_pos = start_pos + end_pos;
        Ok((NTriplesTerm::BlankNode(blank_id), new_pos))
    }

    /// Parse a literal term "..." with optional language tag or datatype
    fn parse_literal(
        &self,
        input: &str,
        start_pos: usize,
        line_number: usize,
    ) -> ParseResult<(NTriplesTerm, usize)> {
        // Find the closing quote
        let mut end_quote = None;
        let mut i = 1; // Start after opening quote
        let chars: Vec<char> = input.chars().collect();

        while i < chars.len() {
            if chars[i] == '"' {
                // Check if it's escaped
                let mut backslash_count = 0;
                let mut j = i;
                while j > 0 && chars[j - 1] == '\\' {
                    backslash_count += 1;
                    j -= 1;
                }
                if backslash_count % 2 == 0 {
                    // Even number of backslashes means the quote is not escaped
                    end_quote = Some(i);
                    break;
                }
            }
            i += 1;
        }

        let end_quote = end_quote.ok_or_else(|| {
            RdfParseError::syntax_at(
                "Unterminated literal - missing closing quote",
                TextPosition::new(line_number, start_pos, 0),
            )
        })?;

        let literal_value = input[1..end_quote].to_string();
        let mut pos_after_quote = start_pos + end_quote + 1;

        // Check for language tag or datatype
        let remaining = &input[end_quote + 1..];

        if remaining.starts_with('@') {
            // Language tag
            let mut lang_end = 1;
            for (i, c) in remaining[1..].char_indices() {
                if c.is_whitespace() {
                    lang_end = 1 + i;
                    break;
                }
                lang_end = 1 + i + c.len_utf8();
            }

            let language = remaining[1..lang_end].to_string();
            pos_after_quote = start_pos + end_quote + 1 + lang_end;
            Ok((
                NTriplesTerm::LanguageLiteral(literal_value, language),
                pos_after_quote,
            ))
        } else if remaining.starts_with("^^") {
            // Datatype
            if remaining.len() < 3 || !remaining[2..].starts_with('<') {
                return Err(RdfParseError::syntax_at(
                    "Invalid datatype format - expected ^^<datatype>",
                    TextPosition::new(line_number, pos_after_quote, 0),
                ));
            }

            if let Some(datatype_end) = remaining[3..].find('>') {
                let datatype = remaining[3..3 + datatype_end].to_string();
                pos_after_quote = start_pos + end_quote + 1 + 3 + datatype_end + 1;
                Ok((
                    NTriplesTerm::TypedLiteral(literal_value, datatype),
                    pos_after_quote,
                ))
            } else {
                Err(RdfParseError::syntax_at(
                    "Unterminated datatype IRI - missing '>'",
                    TextPosition::new(line_number, pos_after_quote, 0),
                ))
            }
        } else {
            // Simple literal
            Ok((NTriplesTerm::Literal(literal_value), pos_after_quote))
        }
    }

    /// Convert parsed term to Subject
    fn term_to_subject(
        &self,
        term: &NTriplesTerm,
        line_number: usize,
    ) -> ParseResult<crate::model::term::Subject> {
        match term {
            NTriplesTerm::Iri(iri) => {
                let named_node = NamedNode::new(iri).map_err(|e| {
                    RdfParseError::syntax_at(
                        format!("Invalid subject IRI: {e}"),
                        TextPosition::new(line_number, 0, 0),
                    )
                })?;
                Ok(crate::model::term::Subject::NamedNode(named_node))
            }
            NTriplesTerm::BlankNode(id) => {
                let blank_node = BlankNode::new(id).map_err(|e| {
                    RdfParseError::syntax_at(
                        format!("Invalid blank node: {e}"),
                        TextPosition::new(line_number, 0, 0),
                    )
                })?;
                Ok(crate::model::term::Subject::BlankNode(blank_node))
            }
            _ => Err(RdfParseError::syntax_at(
                "Subject must be an IRI or blank node",
                TextPosition::new(line_number, 0, 0),
            )),
        }
    }

    /// Convert parsed term to Predicate
    fn term_to_predicate(
        &self,
        term: &NTriplesTerm,
        line_number: usize,
    ) -> ParseResult<crate::model::term::Predicate> {
        match term {
            NTriplesTerm::Iri(iri) => {
                let named_node = NamedNode::new(iri).map_err(|e| {
                    RdfParseError::syntax_at(
                        format!("Invalid predicate IRI: {e}"),
                        TextPosition::new(line_number, 0, 0),
                    )
                })?;
                Ok(crate::model::term::Predicate::NamedNode(named_node))
            }
            _ => Err(RdfParseError::syntax_at(
                "Predicate must be an IRI",
                TextPosition::new(line_number, 0, 0),
            )),
        }
    }

    /// Convert parsed term to Object
    fn term_to_object(
        &self,
        term: &NTriplesTerm,
        line_number: usize,
    ) -> ParseResult<crate::model::term::Object> {
        match term {
            NTriplesTerm::Iri(iri) => {
                let named_node = NamedNode::new(iri).map_err(|e| {
                    RdfParseError::syntax_at(
                        &format!("Invalid object IRI: {}", e),
                        TextPosition::new(line_number, 0, 0),
                    )
                })?;
                Ok(crate::model::term::Object::NamedNode(named_node))
            }
            NTriplesTerm::BlankNode(id) => {
                let blank_node = BlankNode::new(id).map_err(|e| {
                    RdfParseError::syntax_at(
                        format!("Invalid blank node: {e}"),
                        TextPosition::new(line_number, 0, 0),
                    )
                })?;
                Ok(crate::model::term::Object::BlankNode(blank_node))
            }
            NTriplesTerm::Literal(value) => {
                let literal = Literal::new(value);
                Ok(crate::model::term::Object::Literal(literal))
            }
            NTriplesTerm::LanguageLiteral(value, lang) => {
                let literal = Literal::new_language_tagged_literal(value, lang).map_err(|e| {
                    RdfParseError::syntax_at(
                        &format!("Invalid language tag: {}", e),
                        TextPosition::new(line_number, 0, 0),
                    )
                })?;
                Ok(crate::model::term::Object::Literal(literal))
            }
            NTriplesTerm::TypedLiteral(value, datatype_iri) => {
                let datatype = NamedNode::new(datatype_iri).map_err(|e| {
                    RdfParseError::syntax_at(
                        &format!("Invalid datatype IRI: {}", e),
                        TextPosition::new(line_number, 0, 0),
                    )
                })?;
                let literal = Literal::new_typed_literal(value, datatype);
                Ok(crate::model::term::Object::Literal(literal))
            }
        }
    }
}

impl Default for NTriplesParser {
    fn default() -> Self {
        Self::new()
    }
}

/// N-Triples serializer implementation
#[derive(Debug, Clone)]
pub struct NTriplesSerializer {
    validate: bool,
}

impl NTriplesSerializer {
    /// Create a new N-Triples serializer
    pub fn new() -> Self {
        Self { validate: true }
    }

    /// Disable output validation for performance
    pub fn unvalidated(mut self) -> Self {
        self.validate = false;
        self
    }

    /// Create a writer-based serializer
    pub fn for_writer<W: Write>(self, writer: W) -> WriterNTriplesSerializer<W> {
        WriterNTriplesSerializer::new(writer, self)
    }

    /// Serialize triples to a string
    pub fn serialize_to_string(&self, triples: &[Triple]) -> SerializeResult<String> {
        let mut buffer = Vec::new();
        {
            let mut serializer = self.clone().for_writer(&mut buffer);
            for triple in triples {
                serializer.serialize_triple(triple.as_ref())?;
            }
            serializer.finish()?;
        }
        String::from_utf8(buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Check if validation is enabled
    pub fn is_validating(&self) -> bool {
        self.validate
    }
}

impl Default for NTriplesSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer-based N-Triples serializer
pub struct WriterNTriplesSerializer<W: Write> {
    writer: W,
    config: NTriplesSerializer,
}

impl<W: Write> WriterNTriplesSerializer<W> {
    /// Create a new writer serializer
    pub fn new(writer: W, config: NTriplesSerializer) -> Self {
        Self { writer, config }
    }

    /// Serialize a triple
    pub fn serialize_triple(&mut self, triple: TripleRef<'_>) -> SerializeResult<()> {
        // Format: <subject> <predicate> <object> .

        // Serialize subject
        self.serialize_subject(triple.subject())?;
        write!(self.writer, " ")?;

        // Serialize predicate
        self.serialize_predicate(triple.predicate())?;
        write!(self.writer, " ")?;

        // Serialize object
        self.serialize_object(triple.object())?;
        writeln!(self.writer, " .")?;

        Ok(())
    }

    /// Serialize a subject (IRI or blank node)
    fn serialize_subject(
        &mut self,
        subject: crate::model::triple::SubjectRef<'_>,
    ) -> SerializeResult<()> {
        use crate::model::triple::SubjectRef;
        match subject {
            SubjectRef::NamedNode(node) => {
                write!(self.writer, "<{}>", self.escape_iri(node.as_str()))?;
            }
            SubjectRef::BlankNode(node) => {
                write!(self.writer, "_:{}", node.as_str())?;
            }
            SubjectRef::Variable(var) => {
                write!(self.writer, "?{}", var.as_str())?;
            }
        }
        Ok(())
    }

    /// Serialize a predicate (IRI)
    fn serialize_predicate(
        &mut self,
        predicate: crate::model::triple::PredicateRef<'_>,
    ) -> SerializeResult<()> {
        use crate::model::triple::PredicateRef;
        match predicate {
            PredicateRef::NamedNode(node) => {
                write!(self.writer, "<{}>", self.escape_iri(node.as_str()))?;
            }
            PredicateRef::Variable(var) => {
                write!(self.writer, "?{}", var.as_str())?;
            }
        }
        Ok(())
    }

    /// Serialize an object (IRI, blank node, or literal)
    fn serialize_object(
        &mut self,
        object: crate::model::triple::ObjectRef<'_>,
    ) -> SerializeResult<()> {
        use crate::model::triple::ObjectRef;
        match object {
            ObjectRef::NamedNode(node) => {
                write!(self.writer, "<{}>", self.escape_iri(node.as_str()))?;
            }
            ObjectRef::BlankNode(node) => {
                write!(self.writer, "_:{}", node.as_str())?;
            }
            ObjectRef::Literal(literal) => {
                self.serialize_literal(literal)?;
            }
            ObjectRef::Variable(var) => {
                write!(self.writer, "?{}", var.as_str())?;
            }
        }
        Ok(())
    }

    /// Serialize a literal
    fn serialize_literal(&mut self, literal: &Literal) -> SerializeResult<()> {
        // Write the literal value with proper escaping
        write!(self.writer, "\"{}\"", self.escape_literal(literal.value()))?;

        // Add language tag or datatype if present
        if let Some(language) = literal.language() {
            write!(self.writer, "@{}", language)?;
        } else if literal.datatype().as_str() != crate::vocab::xsd::STRING.as_str() {
            write!(
                self.writer,
                "^^<{}>",
                self.escape_iri(literal.datatype().as_str())
            )?;
        }

        Ok(())
    }

    /// Escape special characters in IRIs
    fn escape_iri(&self, iri: &str) -> String {
        // N-Triples IRIs should already be properly encoded
        // but we can add basic escaping if needed
        iri.to_string()
    }

    /// Escape special characters in literal values
    fn escape_literal(&self, value: &str) -> String {
        value
            .chars()
            .map(|c| match c {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                _ => c.to_string(),
            })
            .collect()
    }

    /// Finish serialization and return the writer
    pub fn finish(self) -> SerializeResult<W> {
        Ok(self.writer)
    }
}

impl<W: Write> QuadSerializer<W> for WriterNTriplesSerializer<W> {
    fn serialize_quad(&mut self, quad: crate::model::QuadRef<'_>) -> SerializeResult<()> {
        // N-Triples only supports default graph, so ignore named graphs
        if quad.graph_name().is_default_graph() {
            self.serialize_triple(quad.triple())
        } else {
            // Could log a warning here about ignoring named graph
            Ok(())
        }
    }

    fn finish(self: Box<Self>) -> SerializeResult<W> {
        Ok(self.writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntriples_parser_creation() {
        let parser = NTriplesParser::new();
        assert!(!parser.is_lenient());
    }

    #[test]
    fn test_ntriples_parser_lenient() {
        let parser = NTriplesParser::new().lenient();
        assert!(parser.is_lenient());
    }

    #[test]
    fn test_ntriples_serializer_creation() {
        let serializer = NTriplesSerializer::new();
        assert!(serializer.is_validating());
    }

    #[test]
    fn test_ntriples_serializer_unvalidated() {
        let serializer = NTriplesSerializer::new().unvalidated();
        assert!(!serializer.is_validating());
    }

    #[test]
    fn test_empty_ntriples_parsing() {
        let parser = NTriplesParser::new();
        let result = parser.parse_str("");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_ntriples_comments() {
        let parser = NTriplesParser::new();
        let ntriples = "# This is a comment\n# Another comment";
        let result = parser.parse_str(ntriples);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_ntriples_line_validation() {
        let parser = NTriplesParser::new();

        // Missing dot should fail
        let result = parser.parse_triple_line(
            "<http://example.org/s> <http://example.org/p> <http://example.org/o>",
            1,
        );
        assert!(result.is_err());

        // Too few components should fail
        let result = parser.parse_triple_line("<http://example.org/s> <http://example.org/p> .", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_ntriples_parsing() {
        let parser = NTriplesParser::new();

        // Test simple triple
        let ntriples = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let result = parser.parse_str(ntriples);
        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);

        // Test with blank node
        let ntriples = "_:s <http://example.org/p> \"literal\" .";
        let result = parser.parse_str(ntriples);
        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);

        // Test with language literal
        let ntriples = "<http://example.org/s> <http://example.org/p> \"hello\"@en .";
        let result = parser.parse_str(ntriples);
        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);

        // Test with typed literal
        let ntriples = "<http://example.org/s> <http://example.org/p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .";
        let result = parser.parse_str(ntriples);
        assert!(result.is_ok());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_ntriples_serialization() {
        let serializer = NTriplesSerializer::new();

        // Create a simple triple
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("test");
        let triple = Triple::new(subject, predicate, object);

        let result = serializer.serialize_to_string(&[triple]);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("<http://example.org/s>"));
        assert!(output.contains("<http://example.org/p>"));
        assert!(output.contains("\"test\""));
        assert!(output.ends_with(" .\n"));
    }
}
