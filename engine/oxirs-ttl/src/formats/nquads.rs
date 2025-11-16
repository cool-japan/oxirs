//! N-Quads format parser and serializer (stub implementation)

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::toolkit::{Parser, Serializer};
use oxirs_core::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject,
};
use std::io::{BufRead, BufReader, Read, Write};

/// N-Quads parser
#[derive(Debug, Clone, Default)]
pub struct NQuadsParser;

impl NQuadsParser {
    /// Creates a new N-Quads parser
    pub fn new() -> Self {
        Self
    }

    /// Strip inline comments from a line (# after data, not inside quotes or IRIs)
    fn strip_inline_comment<'a>(&self, line: &'a str) -> &'a str {
        let mut in_string = false;
        let mut in_iri = false;
        let mut escaped = false;

        for (i, ch) in line.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }

            match ch {
                '\\' if in_string => escaped = true,
                '"' => in_string = !in_string,
                '<' if !in_string => in_iri = true,
                '>' if !in_string => in_iri = false,
                '#' if !in_string && !in_iri => return line[..i].trim_end(),
                _ => {}
            }
        }

        line
    }

    /// Parse a single line of N-Quads format
    fn parse_line(&self, line: &str, line_num: usize) -> TurtleResult<Option<Quad>> {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        // Strip inline comments (# after the statement, not inside quotes)
        let line = self.strip_inline_comment(line);

        // Lines must end with '.'
        if !line.ends_with('.') {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "N-Quads line must end with '.'".to_string(),
                position: TextPosition::new(line_num, line.len(), 0),
            }));
        }

        // Remove the trailing '.'
        let content = line[..line.len() - 1].trim();

        // Split into tokens (very simple tokenizer for N-Quads)
        let tokens = self.tokenize(content)?;

        if tokens.len() < 3 || tokens.len() > 4 {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!(
                    "N-Quads line must have 3 or 4 components, found {}",
                    tokens.len()
                ),
                position: TextPosition::new(line_num, 1, 0),
            }));
        }

        let subject = self.parse_subject(&tokens[0])?;
        let predicate = self.parse_predicate(&tokens[1])?;
        let object = self.parse_object(&tokens[2])?;

        let graph_name = if tokens.len() == 4 {
            self.parse_graph(&tokens[3])?
        } else {
            GraphName::DefaultGraph
        };

        Ok(Some(Quad::new(subject, predicate, object, graph_name)))
    }

    /// Simple tokenizer for N-Quads format
    fn tokenize(&self, content: &str) -> TurtleResult<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_string = false;
        let mut in_iri = false;
        let mut escape_next = false;

        for ch in content.chars() {
            if escape_next {
                current_token.push(ch);
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => {
                    current_token.push('\\');
                    escape_next = true;
                }
                '"' => {
                    current_token.push('"');
                    in_string = !in_string;
                }
                '<' if !in_string => {
                    // Check if this is part of a typed literal (^^<IRI>)
                    // If current_token ends with ^^, keep building the same token
                    if !current_token.ends_with("^^") && !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                    current_token.push('<');
                    in_iri = true;
                }
                '>' if in_iri && !in_string => {
                    current_token.push('>');
                    in_iri = false;
                    // Only push token if we're not in the middle of a typed literal
                    // If the token starts with a quote, it's a literal with datatype
                    if !current_token.starts_with('"') {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                }
                ' ' | '\t' if !in_string && !in_iri => {
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

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        Ok(tokens)
    }

    fn parse_subject(&self, token: &str) -> TurtleResult<Subject> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            Ok(Subject::NamedNode(
                NamedNode::new(iri).map_err(TurtleParseError::model)?,
            ))
        } else if let Some(stripped) = token.strip_prefix("_:") {
            Ok(Subject::BlankNode(
                BlankNode::new(stripped).map_err(TurtleParseError::model)?,
            ))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid subject: {token}"),
                position: TextPosition::default(),
            }))
        }
    }

    fn parse_predicate(&self, token: &str) -> TurtleResult<NamedNode> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            Ok(NamedNode::new(iri).map_err(TurtleParseError::model)?)
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid predicate: {token}"),
                position: TextPosition::default(),
            }))
        }
    }

    fn parse_object(&self, token: &str) -> TurtleResult<Object> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            Ok(Object::NamedNode(
                NamedNode::new(iri).map_err(TurtleParseError::model)?,
            ))
        } else if let Some(stripped) = token.strip_prefix("_:") {
            Ok(Object::BlankNode(
                BlankNode::new(stripped).map_err(TurtleParseError::model)?,
            ))
        } else if token.starts_with('"') {
            self.parse_literal(token)
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid object: {token}"),
                position: TextPosition::default(),
            }))
        }
    }

    fn parse_graph(&self, token: &str) -> TurtleResult<GraphName> {
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            Ok(GraphName::NamedNode(
                NamedNode::new(iri).map_err(TurtleParseError::model)?,
            ))
        } else if let Some(stripped) = token.strip_prefix("_:") {
            Ok(GraphName::BlankNode(
                BlankNode::new(stripped).map_err(TurtleParseError::model)?,
            ))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid graph name: {token}"),
                position: TextPosition::default(),
            }))
        }
    }

    fn parse_literal(&self, token: &str) -> TurtleResult<Object> {
        if !token.starts_with('"') {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Literal must start with quote".to_string(),
                position: TextPosition::default(),
            }));
        }

        // Find the closing quote (handling escapes) - use byte indices for proper UTF-8 handling
        let mut end_quote_byte_idx = None;
        let mut char_iter = token.char_indices().skip(1); // Skip opening quote

        while let Some((byte_idx, ch)) = char_iter.next() {
            if ch == '\\' {
                // Skip the escaped character
                char_iter.next();
            } else if ch == '"' {
                end_quote_byte_idx = Some(byte_idx);
                break;
            }
        }

        let end_quote_byte_idx = end_quote_byte_idx.ok_or_else(|| {
            TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Unterminated literal".to_string(),
                position: TextPosition::default(),
            })
        })?;

        // Use byte indices for slicing to properly handle UTF-8 multi-byte characters
        let value_with_escapes = &token[1..end_quote_byte_idx];
        let value = self.unescape_string(value_with_escapes)?;

        let remainder = &token[end_quote_byte_idx + 1..];

        if remainder.is_empty() {
            // Simple literal
            Ok(Object::Literal(Literal::new(value)))
        } else if let Some(lang) = remainder.strip_prefix('@') {
            // Language-tagged literal
            Ok(Object::Literal(
                Literal::new_language_tagged_literal(value, lang).map_err(|e| {
                    TurtleParseError::syntax(TurtleSyntaxError::Generic {
                        message: format!("Invalid language tag: {e}"),
                        position: TextPosition::default(),
                    })
                })?,
            ))
        } else if remainder.starts_with("^^") && remainder.len() > 2 {
            // Typed literal
            let datatype_token = &remainder[2..];
            if datatype_token.starts_with('<') && datatype_token.ends_with('>') {
                let datatype_iri = &datatype_token[1..datatype_token.len() - 1];
                let datatype = NamedNode::new(datatype_iri).map_err(TurtleParseError::model)?;
                Ok(Object::Literal(Literal::new_typed_literal(value, datatype)))
            } else {
                Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "Invalid datatype IRI".to_string(),
                    position: TextPosition::default(),
                }))
            }
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Invalid literal format: {remainder}"),
                position: TextPosition::default(),
            }))
        }
    }

    fn unescape_string(&self, s: &str) -> TurtleResult<String> {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('u') => {
                        // Unicode escape \uXXXX (4 hex digits)
                        let mut hex = String::new();
                        for _ in 0..4 {
                            if let Some(hex_char) = chars.next() {
                                hex.push(hex_char);
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("u{hex}"),
                                        position: TextPosition::default(),
                                    },
                                ));
                            }
                        }

                        let code_point = u32::from_str_radix(&hex, 16).map_err(|_| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidEscape {
                                sequence: format!("u{hex}"),
                                position: TextPosition::default(),
                            })
                        })?;

                        let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidUnicode {
                                codepoint: code_point,
                                position: TextPosition::default(),
                            })
                        })?;

                        result.push(unicode_char);
                    }
                    Some('U') => {
                        // Unicode escape \UXXXXXXXX (8 hex digits)
                        let mut hex = String::new();
                        for _ in 0..8 {
                            if let Some(hex_char) = chars.next() {
                                hex.push(hex_char);
                            } else {
                                return Err(TurtleParseError::syntax(
                                    TurtleSyntaxError::InvalidEscape {
                                        sequence: format!("U{hex}"),
                                        position: TextPosition::default(),
                                    },
                                ));
                            }
                        }

                        let code_point = u32::from_str_radix(&hex, 16).map_err(|_| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidEscape {
                                sequence: format!("U{hex}"),
                                position: TextPosition::default(),
                            })
                        })?;

                        let unicode_char = char::from_u32(code_point).ok_or_else(|| {
                            TurtleParseError::syntax(TurtleSyntaxError::InvalidUnicode {
                                codepoint: code_point,
                                position: TextPosition::default(),
                            })
                        })?;

                        result.push(unicode_char);
                    }
                    Some(other) => {
                        return Err(TurtleParseError::syntax(TurtleSyntaxError::InvalidEscape {
                            sequence: other.to_string(),
                            position: TextPosition::default(),
                        }));
                    }
                    None => {
                        return Err(TurtleParseError::syntax(TurtleSyntaxError::InvalidEscape {
                            sequence: "".to_string(),
                            position: TextPosition::default(),
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

impl Parser<Quad> for NQuadsParser {
    fn parse<R: Read>(&self, reader: R) -> TurtleResult<Vec<Quad>> {
        let mut quads = Vec::new();
        let buf_reader = BufReader::new(reader);

        for (line_num, line_result) in buf_reader.lines().enumerate() {
            let line = line_result.map_err(TurtleParseError::io)?;

            if let Some(quad) = self.parse_line(&line, line_num + 1)? {
                quads.push(quad)
            } // Empty line or comment
        }

        Ok(quads)
    }

    fn for_reader<R: BufRead + 'static>(
        &self,
        reader: R,
    ) -> Box<dyn Iterator<Item = TurtleResult<Quad>>> {
        Box::new(NQuadsIterator::new(reader))
    }
}

/// N-Quads serializer
#[derive(Debug, Clone)]
pub struct NQuadsSerializer;

impl Default for NQuadsSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl NQuadsSerializer {
    /// Creates a new N-Quads serializer
    pub fn new() -> Self {
        Self
    }

    fn format_subject(&self, subject: &Subject) -> String {
        match subject {
            Subject::NamedNode(n) => format!("<{}>", n.as_str()),
            Subject::BlankNode(b) => {
                format!("_:{}", b.as_str().strip_prefix("_:").unwrap_or(b.as_str()))
            }
            Subject::Variable(v) => format!("?{}", v.name()),
            Subject::QuotedTriple(_) => "<<quoted-triple>>".to_string(), // RDF-star support placeholder
        }
    }

    fn format_predicate(&self, predicate: &Predicate) -> String {
        match predicate {
            Predicate::NamedNode(n) => format!("<{}>", n.as_str()),
            Predicate::Variable(v) => format!("?{}", v.name()),
        }
    }

    fn format_object(&self, object: &Object) -> String {
        match object {
            Object::NamedNode(n) => format!("<{}>", n.as_str()),
            Object::BlankNode(b) => {
                format!("_:{}", b.as_str().strip_prefix("_:").unwrap_or(b.as_str()))
            }
            Object::Literal(l) => self.format_literal(l),
            Object::Variable(v) => format!("?{}", v.name()),
            Object::QuotedTriple(_) => "<<quoted-triple>>".to_string(), // RDF-star support placeholder
        }
    }

    fn format_graph(&self, graph: &GraphName) -> String {
        match graph {
            GraphName::NamedNode(n) => format!("<{}>", n.as_str()),
            GraphName::BlankNode(b) => {
                format!("_:{}", b.as_str().strip_prefix("_:").unwrap_or(b.as_str()))
            }
            GraphName::Variable(v) => format!("?{}", v.name()),
            GraphName::DefaultGraph => "<>".to_string(),
        }
    }

    fn format_literal(&self, literal: &Literal) -> String {
        let value = literal.value();
        let escaped = self.escape_string(value);

        if let Some(lang) = literal.language() {
            format!("\"{escaped}\"@{lang}")
        } else {
            let datatype = literal.datatype();
            if datatype.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                format!("\"{escaped}\"")
            } else {
                format!("\"{escaped}\"^^<{}>", datatype.as_str())
            }
        }
    }

    fn escape_string(&self, s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '\\' => result.push_str("\\\\"),
                '\"' => result.push_str("\\\""),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                c if c.is_control() => result.push_str(&format!("\\u{:04X}", c as u32)),
                c => result.push(c),
            }
        }
        result
    }
}

impl Serializer<Quad> for NQuadsSerializer {
    fn serialize<W: Write>(&self, quads: &[Quad], mut writer: W) -> TurtleResult<()> {
        for quad in quads {
            self.serialize_item(quad, &mut writer)?;
        }
        Ok(())
    }

    fn serialize_item<W: Write>(&self, quad: &Quad, mut writer: W) -> TurtleResult<()> {
        // Format: <subject> <predicate> <object> [<graph>] .
        write!(
            writer,
            "{} {} {}",
            self.format_subject(quad.subject()),
            self.format_predicate(quad.predicate()),
            self.format_object(quad.object())
        )
        .map_err(TurtleParseError::io)?;

        // Add graph if not default graph
        if !quad.graph_name().is_default_graph() {
            write!(writer, " {}", self.format_graph(quad.graph_name()))
                .map_err(TurtleParseError::io)?;
        }

        writeln!(writer, " .").map_err(TurtleParseError::io)?;
        Ok(())
    }
}

/// Iterator for parsing N-Quads from a buffered reader
pub struct NQuadsIterator<R: BufRead> {
    reader: R,
    parser: NQuadsParser,
    line_num: usize,
}

impl<R: BufRead> NQuadsIterator<R> {
    /// Creates a new N-Quads iterator from a reader
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            parser: NQuadsParser::new(),
            line_num: 0,
        }
    }
}

impl<R: BufRead> Iterator for NQuadsIterator<R> {
    type Item = TurtleResult<Quad>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None, // EOF
                Ok(_) => {
                    self.line_num += 1;
                    match self.parser.parse_line(&line, self.line_num) {
                        Ok(Some(quad)) => return Some(Ok(quad)),
                        Ok(None) => continue, // Empty line or comment
                        Err(e) => return Some(Err(e)),
                    }
                }
                Err(e) => return Some(Err(TurtleParseError::io(e))),
            }
        }
    }
}
