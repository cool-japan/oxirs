//! N-Triples format parser and serializer
//!
//! N-Triples is the simplest RDF serialization format, consisting of one triple per line.
//! This implementation provides both streaming parsing and serialization capabilities.

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::toolkit::{FormattedWriter, Parser, SerializationConfig, Serializer};
use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use std::io::{BufRead, BufReader, Read, Write};

/// N-Triples parser
#[derive(Debug, Clone)]
pub struct NTriplesParser {
    /// Whether to continue parsing after errors
    pub lenient: bool,
}

impl Default for NTriplesParser {
    fn default() -> Self {
        Self::new()
    }
}

impl NTriplesParser {
    /// Create a new N-Triples parser
    pub fn new() -> Self {
        Self { lenient: false }
    }

    /// Create a new lenient N-Triples parser (continues after errors)
    pub fn new_lenient() -> Self {
        Self { lenient: true }
    }

    /// Parse a single N-Triples line
    pub fn parse_line(&self, line: &str, line_number: usize) -> TurtleResult<Option<Triple>> {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        // Must end with a dot
        if !line.ends_with('.') {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "N-Triples line must end with '.'".to_string(),
                position: TextPosition::new(line_number, line.len(), 0),
            }));
        }

        let line = &line[..line.len() - 1].trim(); // Remove dot and trim

        // Split into tokens (simplified tokenization)
        let tokens = self.tokenize_line(line, line_number)?;

        if tokens.len() != 3 {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!(
                    "Expected 3 tokens (subject predicate object), found {}",
                    tokens.len()
                ),
                position: TextPosition::new(line_number, 1, 0),
            }));
        }

        let subject = self.parse_subject(&tokens[0], line_number)?;
        let predicate = self.parse_predicate(&tokens[1], line_number)?;
        let object = self.parse_object(&tokens[2], line_number)?;

        Ok(Some(Triple::new(subject, predicate, object)))
    }

    /// Tokenize a line (simplified - proper implementation would handle quoted strings properly)
    fn tokenize_line(&self, line: &str, line_number: usize) -> TurtleResult<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_quotes = false;
        let mut escaped = false;

        for (_i, ch) in line.char_indices() {
            if escaped {
                current_token.push(ch);
                escaped = false;
                continue;
            }

            match ch {
                '\\' if in_quotes => {
                    escaped = true;
                    current_token.push(ch);
                }
                '"' => {
                    in_quotes = !in_quotes;
                    current_token.push(ch);
                }
                ' ' | '\t' if !in_quotes => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }

        if in_quotes {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Unterminated string literal".to_string(),
                position: TextPosition::new(line_number, line.len(), 0),
            }));
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        Ok(tokens)
    }

    /// Parse a subject (IRI or blank node)
    fn parse_subject(&self, token: &str, line_number: usize) -> TurtleResult<Subject> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri).map_err(TurtleParseError::model)?;
            Ok(Subject::NamedNode(named_node))
        } else if let Some(id) = token.strip_prefix("_:") {
            let blank_node = BlankNode::new(id).map_err(TurtleParseError::model)?;
            Ok(Subject::BlankNode(blank_node))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid subject: {token}"),
                position: TextPosition::new(line_number, 1, 0),
            }))
        }
    }

    /// Parse a predicate (must be an IRI)
    fn parse_predicate(&self, token: &str, line_number: usize) -> TurtleResult<Predicate> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri).map_err(TurtleParseError::model)?;
            Ok(Predicate::NamedNode(named_node))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid predicate: {token}"),
                position: TextPosition::new(line_number, 1, 0),
            }))
        }
    }

    /// Parse an object (IRI, blank node, or literal)
    fn parse_object(&self, token: &str, line_number: usize) -> TurtleResult<Object> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            let named_node = NamedNode::new(iri).map_err(TurtleParseError::model)?;
            Ok(Object::NamedNode(named_node))
        } else if let Some(id) = token.strip_prefix("_:") {
            let blank_node = BlankNode::new(id).map_err(TurtleParseError::model)?;
            Ok(Object::BlankNode(blank_node))
        } else if token.starts_with('"') {
            self.parse_literal(token, line_number)
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid object: {token}"),
                position: TextPosition::new(line_number, 1, 0),
            }))
        }
    }

    /// Parse a literal object
    fn parse_literal(&self, token: &str, line_number: usize) -> TurtleResult<Object> {
        if !token.starts_with('"') {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Literal must start with quote".to_string(),
                position: TextPosition::new(line_number, 1, 0),
            }));
        }

        // Find the end quote (handling escapes)
        let mut end_quote = None;
        let mut escaped = false;
        let chars: Vec<char> = token.chars().collect();

        for (i, &char) in chars.iter().enumerate().skip(1) {
            if escaped {
                escaped = false;
                continue;
            }

            if char == '\\' {
                escaped = true;
            } else if char == '"' {
                end_quote = Some(i);
                break;
            }
        }

        let end_quote = end_quote.ok_or_else(|| {
            TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Unterminated string literal".to_string(),
                position: TextPosition::new(line_number, token.len(), 0),
            })
        })?;

        let value = &token[1..end_quote];
        let remainder = &token[end_quote + 1..];

        // Unescape the string value
        let unescaped_value = self.unescape_string(value)?;

        // Check for language tag or datatype
        if remainder.is_empty() {
            // Simple string literal
            let literal = Literal::new_simple_literal(&unescaped_value);
            Ok(Object::Literal(literal))
        } else if let Some(language) = remainder.strip_prefix('@') {
            // Language-tagged literal
            let literal = Literal::new_language_tagged_literal(&unescaped_value, language)
                .map_err(|e| TurtleParseError::model(e.into()))?;
            Ok(Object::Literal(literal))
        } else if remainder.starts_with("^^<") && remainder.ends_with('>') {
            // Typed literal
            let datatype_iri = &remainder[3..remainder.len() - 1];
            let datatype = NamedNode::new(datatype_iri).map_err(TurtleParseError::model)?;
            let literal = Literal::new_typed_literal(&unescaped_value, datatype);
            Ok(Object::Literal(literal))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid literal suffix: {remainder}"),
                position: TextPosition::new(line_number, end_quote + 1, 0),
            }))
        }
    }

    /// Unescape a string literal
    fn unescape_string(&self, s: &str) -> TurtleResult<String> {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('t') => result.push('\t'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('u') => {
                        // Unicode escape \uXXXX
                        let mut code = String::new();
                        for _ in 0..4 {
                            if let Some(hex_char) = chars.next() {
                                code.push(hex_char);
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("u{code}"),
                                        position: TextPosition::start(),
                                    },
                                ));
                            }
                        }
                        let code_point = u32::from_str_radix(&code, 16).map_err(|_| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidEscape {
                                sequence: format!("u{code}"),
                                position: TextPosition::start(),
                            })
                        })?;
                        let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidUnicode {
                                codepoint: code_point,
                                position: TextPosition::start(),
                            })
                        })?;
                        result.push(unicode_char);
                    }
                    Some('U') => {
                        // Unicode escape \UXXXXXXXX
                        let mut code = String::new();
                        for _ in 0..8 {
                            if let Some(hex_char) = chars.next() {
                                code.push(hex_char);
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("U{code}"),
                                        position: TextPosition::start(),
                                    },
                                ));
                            }
                        }
                        let code_point = u32::from_str_radix(&code, 16).map_err(|_| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidEscape {
                                sequence: format!("U{code}"),
                                position: TextPosition::start(),
                            })
                        })?;
                        let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidUnicode {
                                codepoint: code_point,
                                position: TextPosition::start(),
                            })
                        })?;
                        result.push(unicode_char);
                    }
                    Some(other) => {
                        return Err(TurtleParseError::syntax(TurtleSyntaxError::InvalidEscape {
                            sequence: other.to_string(),
                            position: TextPosition::start(),
                        }));
                    }
                    None => {
                        return Err(TurtleParseError::syntax(TurtleSyntaxError::UnexpectedEof {
                            position: TextPosition::start(),
                        }));
                    }
                }
            } else {
                result.push(ch);
            }
        }

        Ok(result)
    }
}

impl Parser<Triple> for NTriplesParser {
    fn parse<R: Read>(&self, reader: R) -> TurtleResult<Vec<Triple>> {
        let mut triples = Vec::new();
        let mut errors = Vec::new();

        for (line_number, line_result) in BufReader::new(reader).lines().enumerate() {
            let line = line_result.map_err(TurtleParseError::io)?;

            match self.parse_line(&line, line_number + 1) {
                Ok(Some(triple)) => triples.push(triple),
                Ok(None) => {} // Empty line or comment
                Err(e) => {
                    if self.lenient {
                        errors.push(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        if !errors.is_empty() && self.lenient {
            eprintln!(
                "Warning: {} errors encountered during parsing",
                errors.len()
            );
        }

        Ok(triples)
    }

    fn for_reader<R: BufRead + 'static>(
        &self,
        reader: R,
    ) -> Box<dyn Iterator<Item = TurtleResult<Triple>>> {
        Box::new(NTriplesIterator {
            lines: reader.lines().enumerate(),
            parser: self.clone(),
        })
    }
}

/// Iterator for streaming N-Triples parsing
pub struct NTriplesIterator<L> {
    lines: L,
    parser: NTriplesParser,
}

impl<L> Iterator for NTriplesIterator<L>
where
    L: Iterator<Item = (usize, std::io::Result<String>)>,
{
    type Item = TurtleResult<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.lines.next() {
                None => return None,
                Some((line_number, Ok(line))) => {
                    match self.parser.parse_line(&line, line_number + 1) {
                        Ok(Some(triple)) => return Some(Ok(triple)),
                        Ok(None) => continue, // Empty line or comment
                        Err(e) => {
                            if self.parser.lenient {
                                continue; // Skip errors in lenient mode
                            } else {
                                return Some(Err(e));
                            }
                        }
                    }
                }
                Some((_, Err(io_err))) => {
                    return Some(Err(TurtleParseError::io(io_err)));
                }
            }
        }
    }
}

/// N-Triples serializer
#[derive(Debug, Clone)]
pub struct NTriplesSerializer {
    config: SerializationConfig,
}

impl Default for NTriplesSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl NTriplesSerializer {
    /// Create a new N-Triples serializer
    pub fn new() -> Self {
        Self {
            config: SerializationConfig {
                pretty: false,       // N-Triples doesn't support pretty printing
                use_prefixes: false, // N-Triples doesn't support prefixes
                ..SerializationConfig::default()
            },
        }
    }

    /// Serialize a subject
    fn serialize_subject<W: Write>(
        &self,
        subject: &Subject,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        match subject {
            Subject::NamedNode(nn) => {
                writer
                    .write_str(&format!("<{}>", nn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::QuotedTriple(_) => {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "QuotedTriple subjects not supported in N-Triples".to_string(),
                    position: TextPosition::default(),
                }));
            }
        }
        Ok(())
    }

    /// Serialize a predicate
    fn serialize_predicate<W: Write>(
        &self,
        predicate: &Predicate,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        match predicate {
            Predicate::NamedNode(nn) => {
                writer
                    .write_str(&format!("<{}>", nn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Predicate::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
        }
        Ok(())
    }

    /// Serialize an object
    fn serialize_object<W: Write>(
        &self,
        object: &Object,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        match object {
            Object::NamedNode(nn) => {
                writer
                    .write_str(&format!("<{}>", nn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::Literal(literal) => {
                let escaped = writer.escape_string(literal.value());
                writer.write_str(&escaped).map_err(TurtleParseError::io)?;

                if let Some(language) = literal.language() {
                    writer
                        .write_str(&format!("@{language}"))
                        .map_err(TurtleParseError::io)?;
                } else if literal.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    writer
                        .write_str(&format!("^^<{}>", literal.datatype().as_str()))
                        .map_err(TurtleParseError::io)?;
                }
            }
            Object::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::QuotedTriple(_) => {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "QuotedTriple objects not supported in N-Triples".to_string(),
                    position: TextPosition::default(),
                }));
            }
        }
        Ok(())
    }
}

impl Serializer<Triple> for NTriplesSerializer {
    fn serialize<W: Write>(&self, triples: &[Triple], writer: W) -> TurtleResult<()> {
        let mut formatted_writer = FormattedWriter::new(writer, self.config.clone());

        for triple in triples {
            self.serialize_item(triple, &mut formatted_writer)?;
            formatted_writer
                .write_str(" .\n")
                .map_err(TurtleParseError::io)?;
        }

        Ok(())
    }

    fn serialize_item<W: Write>(&self, triple: &Triple, writer: W) -> TurtleResult<()> {
        let mut formatted_writer = FormattedWriter::new(writer, self.config.clone());
        self.serialize_item_formatted(triple, &mut formatted_writer)
    }
}

impl NTriplesSerializer {
    /// Serialize a single triple to a formatted writer
    fn serialize_item_formatted<W: Write>(
        &self,
        triple: &Triple,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        self.serialize_subject(triple.subject(), writer)?;
        writer.write_str(" ").map_err(TurtleParseError::io)?;
        self.serialize_predicate(triple.predicate(), writer)?;
        writer.write_str(" ").map_err(TurtleParseError::io)?;
        self.serialize_object(triple.object(), writer)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_simple_triple() {
        let parser = NTriplesParser::new();
        let input = r#"<http://example.org/subject> <http://example.org/predicate> "object" ."#;

        let triple = parser.parse_line(input, 1).unwrap().unwrap();

        if let Subject::NamedNode(subject) = triple.subject() {
            assert_eq!(subject.as_str(), "http://example.org/subject");
        } else {
            panic!("Expected named node subject");
        }
    }

    #[test]
    fn test_parse_multiple_triples() {
        let parser = NTriplesParser::new();
        let input = r#"<http://example.org/s1> <http://example.org/p1> "o1" .
<http://example.org/s2> <http://example.org/p2> "o2" .
"#;

        let triples = parser.parse(Cursor::new(input)).unwrap();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_serialize_triple() {
        let serializer = NTriplesSerializer::new();
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap());
        let predicate =
            Predicate::NamedNode(NamedNode::new("http://example.org/predicate").unwrap());
        let object = Object::Literal(Literal::new_simple_literal("object"));
        let triple = Triple::new(subject, predicate, object);

        let mut output = Vec::new();
        serializer.serialize(&[triple], &mut output).unwrap();

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("<http://example.org/subject>"));
        assert!(output_str.contains("\"object\""));
    }
}
