//! Streaming Turtle/N-Triples/N-Quads parser for WASM
//!
//! Processes RDF data incrementally in chunks, avoiding loading full documents
//! into memory at once. Suitable for streaming downloads in browser environments.

use std::collections::HashMap;
use std::fmt;

/// A parsed RDF term output from the streaming parser
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParsedTerm {
    /// An IRI reference
    Iri(String),
    /// A blank node
    BlankNode(String),
    /// A plain literal (value only)
    PlainLiteral(String),
    /// A language-tagged literal
    LangLiteral { value: String, lang: String },
    /// A datatype-tagged literal
    TypedLiteral { value: String, datatype: String },
}

impl ParsedTerm {
    /// Return the string representation for display/export
    pub fn to_ntriples_string(&self) -> String {
        match self {
            ParsedTerm::Iri(iri) => format!("<{}>", iri),
            ParsedTerm::BlankNode(id) => format!("_:{}", id),
            ParsedTerm::PlainLiteral(v) => format!("\"{}\"", escape_string(v)),
            ParsedTerm::LangLiteral { value, lang } => {
                format!("\"{}\"@{}", escape_string(value), lang)
            }
            ParsedTerm::TypedLiteral { value, datatype } => {
                format!("\"{}\"^^<{}>", escape_string(value), datatype)
            }
        }
    }

    /// Return the raw string value (without IRI brackets or literal quotes)
    pub fn value(&self) -> &str {
        match self {
            ParsedTerm::Iri(s) => s,
            ParsedTerm::BlankNode(s) => s,
            ParsedTerm::PlainLiteral(s) => s,
            ParsedTerm::LangLiteral { value, .. } => value,
            ParsedTerm::TypedLiteral { value, .. } => value,
        }
    }
}

impl fmt::Display for ParsedTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_ntriples_string())
    }
}

/// A parsed RDF triple or quad
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedStatement {
    /// An RDF triple (subject, predicate, object)
    Triple {
        subject: ParsedTerm,
        predicate: ParsedTerm,
        object: ParsedTerm,
    },
    /// An RDF quad (subject, predicate, object, graph)
    Quad {
        subject: ParsedTerm,
        predicate: ParsedTerm,
        object: ParsedTerm,
        graph: ParsedTerm,
    },
}

impl ParsedStatement {
    /// Get the subject term
    pub fn subject(&self) -> &ParsedTerm {
        match self {
            ParsedStatement::Triple { subject, .. } => subject,
            ParsedStatement::Quad { subject, .. } => subject,
        }
    }

    /// Get the predicate term
    pub fn predicate(&self) -> &ParsedTerm {
        match self {
            ParsedStatement::Triple { predicate, .. } => predicate,
            ParsedStatement::Quad { predicate, .. } => predicate,
        }
    }

    /// Get the object term
    pub fn object(&self) -> &ParsedTerm {
        match self {
            ParsedStatement::Triple { object, .. } => object,
            ParsedStatement::Quad { object, .. } => object,
        }
    }

    /// Get the graph term (None for triples)
    pub fn graph(&self) -> Option<&ParsedTerm> {
        match self {
            ParsedStatement::Triple { .. } => None,
            ParsedStatement::Quad { graph, .. } => Some(graph),
        }
    }
}

/// Parse errors with position information
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Unexpected token encountered
    UnexpectedToken {
        line: usize,
        col: usize,
        got: String,
        expected: String,
    },
    /// Invalid IRI
    InvalidIri(String),
    /// Invalid literal
    InvalidLiteral(String),
    /// Unknown prefix in prefixed name
    UnknownPrefix(String),
    /// Unexpected end of input
    UnexpectedEof { line: usize, col: usize },
    /// General I/O error
    IoError(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken {
                line,
                col,
                got,
                expected,
            } => {
                write!(
                    f,
                    "Parse error at {}:{}: got '{}', expected {}",
                    line, col, got, expected
                )
            }
            ParseError::InvalidIri(iri) => write!(f, "Invalid IRI: {}", iri),
            ParseError::InvalidLiteral(lit) => write!(f, "Invalid literal: {}", lit),
            ParseError::UnknownPrefix(prefix) => write!(f, "Unknown prefix: '{}'", prefix),
            ParseError::UnexpectedEof { line, col } => {
                write!(f, "Unexpected end of input at {}:{}", line, col)
            }
            ParseError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

/// Supported RDF serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfFormat {
    /// Turtle (Terse RDF Triple Language)
    Turtle,
    /// N-Triples (line-oriented, no prefixes)
    NTriples,
    /// N-Quads (line-oriented with optional graph name)
    NQuads,
}

/// Internal parser state for incremental parsing
#[derive(Debug, Clone, PartialEq)]
enum TurtleState {
    /// Expecting statement start (subject, @prefix, @base, or end)
    StartStatement,
    /// Parsed subject, expecting predicate
    HaveSubject { subject: ParsedTerm },
    /// Parsed subject and predicate, expecting object
    HavePredicate {
        subject: ParsedTerm,
        predicate: ParsedTerm,
    },
    /// Just parsed an object, expecting `,`, `;`, or `.`
    AfterObject {
        subject: ParsedTerm,
        predicate: ParsedTerm,
    },
}

/// Streaming RDF parser that processes data incrementally in chunks.
///
/// Feed data chunks using [`StreamingParser::feed`] and collect parsed statements.
/// Call [`StreamingParser::finish`] when all input has been provided.
pub struct StreamingParser {
    /// Accumulated unparsed text (incomplete lines/statements)
    buffer: String,
    /// Namespace prefix bindings (@prefix declarations)
    prefixes: HashMap<String, String>,
    /// Base IRI for relative IRI resolution
    base_iri: Option<String>,
    /// Current line number (1-based) for error reporting
    line: usize,
    /// Current column within the current line (1-based)
    col: usize,
    /// Output format being parsed
    format: RdfFormat,
    /// Blank node counter for generating unique IDs
    bnode_counter: u64,
    /// Current Turtle parse state (only used for Turtle format)
    turtle_state: TurtleState,
}

impl StreamingParser {
    /// Create a new streaming parser for the given format
    pub fn new(format: RdfFormat) -> Self {
        Self {
            buffer: String::new(),
            prefixes: HashMap::new(),
            base_iri: None,
            line: 1,
            col: 1,
            format,
            bnode_counter: 0,
            turtle_state: TurtleState::StartStatement,
        }
    }

    /// Set the base IRI for relative IRI resolution
    pub fn with_base_iri(mut self, base: &str) -> Self {
        self.base_iri = Some(base.to_string());
        self
    }

    /// Set a namespace prefix mapping
    pub fn add_prefix(&mut self, prefix: &str, iri: &str) {
        self.prefixes.insert(prefix.to_string(), iri.to_string());
    }

    /// Get a reference to the current prefix map
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Feed a chunk of data to the parser.
    ///
    /// Returns all complete statements parsed from the data so far.
    /// Incomplete statements are buffered and will be completed in
    /// subsequent [`feed`](Self::feed) or [`finish`](Self::finish) calls.
    pub fn feed(&mut self, chunk: &str) -> Result<Vec<ParsedStatement>, ParseError> {
        self.buffer.push_str(chunk);
        self.parse_available()
    }

    /// Signal end of input and flush any remaining buffered data.
    ///
    /// Must be called once all input chunks have been fed.
    /// Returns any remaining complete statements.
    pub fn finish(&mut self) -> Result<Vec<ParsedStatement>, ParseError> {
        // Add a trailing newline to flush the last line
        self.buffer.push('\n');
        let result = self.parse_available();
        self.buffer.clear();
        result
    }

    /// Parse whatever complete statements exist in the buffer
    fn parse_available(&mut self) -> Result<Vec<ParsedStatement>, ParseError> {
        match self.format {
            RdfFormat::NTriples => self.parse_ntriples_available(),
            RdfFormat::NQuads => self.parse_nquads_available(),
            RdfFormat::Turtle => self.parse_turtle_available(),
        }
    }

    // -----------------------------------------------------------------------
    // N-Triples parser
    // -----------------------------------------------------------------------

    fn parse_ntriples_available(&mut self) -> Result<Vec<ParsedStatement>, ParseError> {
        let mut statements = Vec::new();

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line_text: String = self.buffer[..newline_pos].to_string();
            self.buffer.drain(..=newline_pos);

            let line_text = line_text.trim_end_matches('\r');

            if let Some(stmt) = self.parse_ntriples_line(line_text)? {
                statements.push(stmt);
            }
            self.line += 1;
            self.col = 1;
        }

        Ok(statements)
    }

    fn parse_ntriples_line(&self, line: &str) -> Result<Option<ParsedStatement>, ParseError> {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        let mut cursor = 0usize;

        // Subject
        skip_whitespace_in(line, &mut cursor);
        let subject = self.parse_subject_term(line, &mut cursor)?;

        // Predicate
        skip_whitespace_in(line, &mut cursor);
        let predicate = self.parse_iri_term(line, &mut cursor)?;

        // Object
        skip_whitespace_in(line, &mut cursor);
        let object = self.parse_object_term(line, &mut cursor)?;

        // Trailing dot
        skip_whitespace_in(line, &mut cursor);
        if cursor < line.len() && line.as_bytes()[cursor] == b'.' {
            cursor += 1;
        }

        // Ignore trailing comment
        skip_whitespace_in(line, &mut cursor);
        if cursor < line.len() && line.as_bytes()[cursor] != b'#' {
            // Extra content after the dot – not fatal, just ignore
        }

        Ok(Some(ParsedStatement::Triple {
            subject,
            predicate,
            object,
        }))
    }

    // -----------------------------------------------------------------------
    // N-Quads parser
    // -----------------------------------------------------------------------

    fn parse_nquads_available(&mut self) -> Result<Vec<ParsedStatement>, ParseError> {
        let mut statements = Vec::new();

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line_text: String = self.buffer[..newline_pos].to_string();
            self.buffer.drain(..=newline_pos);
            let line_text = line_text.trim_end_matches('\r');

            if let Some(stmt) = self.parse_nquads_line(line_text)? {
                statements.push(stmt);
            }
            self.line += 1;
            self.col = 1;
        }

        Ok(statements)
    }

    fn parse_nquads_line(&self, line: &str) -> Result<Option<ParsedStatement>, ParseError> {
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }

        let mut cursor = 0usize;

        skip_whitespace_in(line, &mut cursor);
        let subject = self.parse_subject_term(line, &mut cursor)?;

        skip_whitespace_in(line, &mut cursor);
        let predicate = self.parse_iri_term(line, &mut cursor)?;

        skip_whitespace_in(line, &mut cursor);
        let object = self.parse_object_term(line, &mut cursor)?;

        skip_whitespace_in(line, &mut cursor);

        // Optional graph name
        if cursor < line.len() && line.as_bytes()[cursor] == b'<' {
            let graph = self.parse_iri_term(line, &mut cursor)?;
            skip_whitespace_in(line, &mut cursor);

            return Ok(Some(ParsedStatement::Quad {
                subject,
                predicate,
                object,
                graph,
            }));
        }

        Ok(Some(ParsedStatement::Triple {
            subject,
            predicate,
            object,
        }))
    }

    // -----------------------------------------------------------------------
    // Turtle parser (incremental, line-buffered)
    // -----------------------------------------------------------------------

    fn parse_turtle_available(&mut self) -> Result<Vec<ParsedStatement>, ParseError> {
        let mut statements = Vec::new();

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line_text: String = self.buffer[..newline_pos].to_string();
            self.buffer.drain(..=newline_pos);
            let line_text = line_text.trim_end_matches('\r');

            let mut new_stmts = self.process_turtle_line(line_text)?;
            statements.append(&mut new_stmts);
            self.line += 1;
            self.col = 1;
        }

        Ok(statements)
    }

    fn process_turtle_line(&mut self, line: &str) -> Result<Vec<ParsedStatement>, ParseError> {
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            return Ok(vec![]);
        }

        // Handle @prefix directive
        if line.starts_with("@prefix") || line.to_uppercase().starts_with("PREFIX") {
            self.handle_prefix_directive(line)?;
            return Ok(vec![]);
        }

        // Handle @base directive
        if line.starts_with("@base") || line.to_uppercase().starts_with("BASE") {
            self.handle_base_directive(line)?;
            return Ok(vec![]);
        }

        // Tokenize and process
        self.process_turtle_tokens(line)
    }

    fn handle_prefix_directive(&mut self, line: &str) -> Result<(), ParseError> {
        // Strip "@prefix" or "PREFIX" keyword
        let rest = if let Some(stripped) = line.strip_prefix("@prefix") {
            stripped.trim()
        } else {
            // Case-insensitive PREFIX
            let upper = line.to_uppercase();
            let start = upper.find("PREFIX").map(|p| p + 6).unwrap_or(0);
            line[start..].trim()
        };

        // Find the colon separating prefix label from IRI
        let colon_pos = rest.find(':').ok_or_else(|| {
            ParseError::InvalidIri(format!("Malformed @prefix directive: {}", line))
        })?;

        let prefix = rest[..colon_pos].trim().to_string();
        let after_colon = rest[colon_pos + 1..].trim();

        // Extract IRI in angle brackets
        let iri = if after_colon.starts_with('<') {
            let end = after_colon.find('>').ok_or_else(|| {
                ParseError::InvalidIri(format!("Unclosed IRI in @prefix: {}", line))
            })?;
            after_colon[1..end].to_string()
        } else {
            after_colon.trim_end_matches('.').trim().to_string()
        };

        self.prefixes.insert(prefix, iri);
        Ok(())
    }

    fn handle_base_directive(&mut self, line: &str) -> Result<(), ParseError> {
        let rest = if let Some(stripped) = line.strip_prefix("@base") {
            stripped.trim()
        } else {
            let upper = line.to_uppercase();
            let start = upper.find("BASE").map(|p| p + 4).unwrap_or(0);
            line[start..].trim()
        };

        let iri = if rest.starts_with('<') {
            let end = rest.find('>').ok_or_else(|| {
                ParseError::InvalidIri(format!("Unclosed IRI in @base: {}", line))
            })?;
            rest[1..end].to_string()
        } else {
            rest.trim_end_matches('.').trim().to_string()
        };

        self.base_iri = Some(iri);
        Ok(())
    }

    fn process_turtle_tokens(&mut self, line: &str) -> Result<Vec<ParsedStatement>, ParseError> {
        let mut statements = Vec::new();
        let tokens = tokenize_turtle(line);

        for token in tokens {
            let token_str = token.as_str();

            match &self.turtle_state {
                TurtleState::StartStatement => {
                    if token_str == "." {
                        // End of statement – already at start, ignore
                        continue;
                    }
                    let subject = self.resolve_turtle_term(token_str)?;
                    self.turtle_state = TurtleState::HaveSubject { subject };
                }

                TurtleState::HaveSubject { subject } => {
                    let subject = subject.clone();
                    if token_str == "a" {
                        let predicate = ParsedTerm::Iri(
                            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                        );
                        self.turtle_state = TurtleState::HavePredicate { subject, predicate };
                    } else {
                        let predicate = self.resolve_turtle_term(token_str)?;
                        self.turtle_state = TurtleState::HavePredicate { subject, predicate };
                    }
                }

                TurtleState::HavePredicate { subject, predicate } => {
                    let (subject, predicate) = (subject.clone(), predicate.clone());
                    let object = self.resolve_turtle_term(token_str)?;
                    // Emit the triple
                    statements.push(ParsedStatement::Triple {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object,
                    });
                    self.turtle_state = TurtleState::AfterObject { subject, predicate };
                }

                TurtleState::AfterObject { subject, predicate } => {
                    let (subject, predicate) = (subject.clone(), predicate.clone());
                    match token_str {
                        "." => {
                            // End of subject block
                            self.turtle_state = TurtleState::StartStatement;
                        }
                        ";" => {
                            // Same subject, new predicate
                            self.turtle_state = TurtleState::HaveSubject {
                                subject: subject.clone(),
                            };
                        }
                        "," => {
                            // Same subject and predicate, new object
                            self.turtle_state = TurtleState::HavePredicate {
                                subject: subject.clone(),
                                predicate: predicate.clone(),
                            };
                        }
                        other => {
                            // Treat as beginning of next triple (implicit .)
                            // This can happen if a line isn't terminated
                            let next_subj = self.resolve_turtle_term(other)?;
                            self.turtle_state = TurtleState::HaveSubject { subject: next_subj };
                        }
                    }
                }
            }
        }

        Ok(statements)
    }

    /// Resolve a Turtle term token into a ParsedTerm
    fn resolve_turtle_term(&mut self, token: &str) -> Result<ParsedTerm, ParseError> {
        // Full IRI in angle brackets
        if token.starts_with('<') && token.ends_with('>') {
            let iri = &token[1..token.len() - 1];
            return Ok(ParsedTerm::Iri(self.resolve_iri(iri)));
        }

        // Blank node
        if let Some(bnode_id) = token.strip_prefix("_:") {
            return Ok(ParsedTerm::BlankNode(bnode_id.to_string()));
        }

        // Auto-generated blank node
        if token == "[]" {
            self.bnode_counter += 1;
            return Ok(ParsedTerm::BlankNode(format!("b{}", self.bnode_counter)));
        }

        // String literal (plain, lang-tagged, or typed)
        if token.starts_with('"') || token.starts_with('\'') {
            return self.parse_turtle_literal(token);
        }

        // Numeric literals
        if token.parse::<i64>().is_ok() {
            return Ok(ParsedTerm::TypedLiteral {
                value: token.to_string(),
                datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
            });
        }
        if token.parse::<f64>().is_ok() {
            return Ok(ParsedTerm::TypedLiteral {
                value: token.to_string(),
                datatype: "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
            });
        }

        // Boolean literals
        if token == "true" || token == "false" {
            return Ok(ParsedTerm::TypedLiteral {
                value: token.to_string(),
                datatype: "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
            });
        }

        // Prefixed name (e.g., rdf:type, :local, foaf:name)
        if let Some(colon_pos) = token.find(':') {
            let prefix = &token[..colon_pos];
            let local = &token[colon_pos + 1..];

            if let Some(ns) = self.prefixes.get(prefix) {
                let full_iri = format!("{}{}", ns, local);
                return Ok(ParsedTerm::Iri(full_iri));
            } else if !prefix.is_empty() {
                return Err(ParseError::UnknownPrefix(prefix.to_string()));
            } else {
                // Empty prefix (:local)
                if let Some(base) = &self.base_iri {
                    return Ok(ParsedTerm::Iri(format!("{}{}", base, local)));
                }
                return Err(ParseError::UnknownPrefix(String::from("")));
            }
        }

        // Relative IRI (no colon, no brackets)
        if let Some(base) = &self.base_iri {
            return Ok(ParsedTerm::Iri(format!("{}{}", base, token)));
        }

        Err(ParseError::InvalidIri(token.to_string()))
    }

    fn parse_turtle_literal(&self, token: &str) -> Result<ParsedTerm, ParseError> {
        // Determine quote character and whether it's triple-quoted
        let quote_char = if token.starts_with('"') { '"' } else { '\'' };
        let triple_quote =
            token.starts_with(&format!("{}{}{}", quote_char, quote_char, quote_char));

        let (value_part, rest) = if triple_quote {
            let end_marker = format!("{}{}{}", quote_char, quote_char, quote_char);
            let start = 3;
            let end = token[start..]
                .find(&end_marker)
                .ok_or_else(|| ParseError::InvalidLiteral(token.to_string()))?;
            (&token[start..start + end], &token[start + end + 3..])
        } else {
            // Find closing quote (not escaped)
            let start = 1;
            let chars: Vec<char> = token[start..].chars().collect();
            let mut pos = 0;
            let mut escaped = false;
            loop {
                if pos >= chars.len() {
                    return Err(ParseError::InvalidLiteral(token.to_string()));
                }
                if escaped {
                    escaped = false;
                    pos += 1;
                    continue;
                }
                if chars[pos] == '\\' {
                    escaped = true;
                    pos += 1;
                    continue;
                }
                if chars[pos] == quote_char {
                    break;
                }
                pos += 1;
            }
            // pos is index of closing quote in chars
            let byte_end = start
                + token[start..]
                    .char_indices()
                    .nth(pos)
                    .map(|(i, _)| i)
                    .unwrap_or(token[start..].len());
            (&token[start..byte_end], &token[byte_end + 1..])
        };

        let value = Self::unescape_string(value_part)
            .map_err(|_| ParseError::InvalidLiteral(token.to_string()))?;

        // Language tag
        if let Some(lang_str) = rest.strip_prefix('@') {
            return Ok(ParsedTerm::LangLiteral {
                value,
                lang: lang_str.to_string(),
            });
        }

        // Datatype
        if let Some(dt_str) = rest.strip_prefix("^^") {
            let datatype = if dt_str.starts_with('<') && dt_str.ends_with('>') {
                dt_str[1..dt_str.len() - 1].to_string()
            } else if let Some(colon_pos) = dt_str.find(':') {
                let dt_prefix = &dt_str[..colon_pos];
                let dt_local = &dt_str[colon_pos + 1..];
                if let Some(ns) = self.prefixes.get(dt_prefix) {
                    format!("{}{}", ns, dt_local)
                } else {
                    dt_str.to_string()
                }
            } else {
                dt_str.to_string()
            };
            return Ok(ParsedTerm::TypedLiteral { value, datatype });
        }

        Ok(ParsedTerm::PlainLiteral(value))
    }

    // -----------------------------------------------------------------------
    // Shared term parsers (for N-Triples / N-Quads)
    // -----------------------------------------------------------------------

    fn parse_subject_term(&self, line: &str, cursor: &mut usize) -> Result<ParsedTerm, ParseError> {
        if *cursor >= line.len() {
            return Err(ParseError::UnexpectedEof {
                line: self.line,
                col: self.col,
            });
        }
        match line.as_bytes()[*cursor] {
            b'<' => self.parse_iri_term(line, cursor),
            b'_' => self.parse_blank_node(line, cursor),
            got => Err(ParseError::UnexpectedToken {
                line: self.line,
                col: *cursor + 1,
                got: (got as char).to_string(),
                expected: "IRI or blank node".to_string(),
            }),
        }
    }

    fn parse_iri_term(&self, line: &str, cursor: &mut usize) -> Result<ParsedTerm, ParseError> {
        if *cursor >= line.len() || line.as_bytes()[*cursor] != b'<' {
            return Err(ParseError::UnexpectedToken {
                line: self.line,
                col: *cursor + 1,
                got: line.get(*cursor..*cursor + 1).unwrap_or("EOF").to_string(),
                expected: "'<'".to_string(),
            });
        }
        *cursor += 1;
        let start = *cursor;
        loop {
            if *cursor >= line.len() {
                return Err(ParseError::UnexpectedEof {
                    line: self.line,
                    col: *cursor + 1,
                });
            }
            if line.as_bytes()[*cursor] == b'>' {
                let iri = line[start..*cursor].to_string();
                *cursor += 1;
                return Ok(ParsedTerm::Iri(self.resolve_iri(&iri)));
            }
            *cursor += 1;
        }
    }

    fn parse_blank_node(&self, line: &str, cursor: &mut usize) -> Result<ParsedTerm, ParseError> {
        // Expect "_:"
        if *cursor + 1 >= line.len()
            || line.as_bytes()[*cursor] != b'_'
            || line.as_bytes()[*cursor + 1] != b':'
        {
            return Err(ParseError::UnexpectedToken {
                line: self.line,
                col: *cursor + 1,
                got: line.get(*cursor..*cursor + 2).unwrap_or("?").to_string(),
                expected: "'_:'".to_string(),
            });
        }
        *cursor += 2;
        let start = *cursor;
        while *cursor < line.len()
            && !line.as_bytes()[*cursor].is_ascii_whitespace()
            && line.as_bytes()[*cursor] != b'.'
        {
            *cursor += 1;
        }
        Ok(ParsedTerm::BlankNode(line[start..*cursor].to_string()))
    }

    fn parse_object_term(&self, line: &str, cursor: &mut usize) -> Result<ParsedTerm, ParseError> {
        if *cursor >= line.len() {
            return Err(ParseError::UnexpectedEof {
                line: self.line,
                col: self.col,
            });
        }
        match line.as_bytes()[*cursor] {
            b'<' => self.parse_iri_term(line, cursor),
            b'_' => self.parse_blank_node(line, cursor),
            b'"' => self.parse_ntriples_literal(line, cursor),
            got => Err(ParseError::UnexpectedToken {
                line: self.line,
                col: *cursor + 1,
                got: (got as char).to_string(),
                expected: "IRI, blank node, or literal".to_string(),
            }),
        }
    }

    fn parse_ntriples_literal(
        &self,
        line: &str,
        cursor: &mut usize,
    ) -> Result<ParsedTerm, ParseError> {
        // Opening quote
        *cursor += 1;
        let start = *cursor;
        let mut value = String::new();
        let mut escaped = false;

        loop {
            if *cursor >= line.len() {
                return Err(ParseError::UnexpectedEof {
                    line: self.line,
                    col: *cursor + 1,
                });
            }
            let byte = line.as_bytes()[*cursor];
            if escaped {
                match byte {
                    b'\\' => value.push('\\'),
                    b'"' => value.push('"'),
                    b'n' => value.push('\n'),
                    b'r' => value.push('\r'),
                    b't' => value.push('\t'),
                    b'u' => {
                        // 4-hex-digit escape
                        if *cursor + 4 < line.len() {
                            let hex = &line[*cursor + 1..*cursor + 5];
                            if let Ok(code_point) = u32::from_str_radix(hex, 16) {
                                if let Some(ch) = char::from_u32(code_point) {
                                    value.push(ch);
                                    *cursor += 4;
                                }
                            }
                        }
                    }
                    _ => {
                        value.push('\\');
                        value.push(byte as char);
                    }
                }
                escaped = false;
                *cursor += 1;
                continue;
            }
            if byte == b'\\' {
                escaped = true;
                *cursor += 1;
                continue;
            }
            if byte == b'"' {
                // End of literal value
                *cursor += 1;
                break;
            }
            // Safe: we slice by byte boundaries for ASCII, use char for non-ASCII
            if byte & 0x80 == 0 {
                value.push(byte as char);
                *cursor += 1;
            } else {
                // Multi-byte UTF-8 character
                let ch_str = &line[*cursor..];
                let ch = ch_str.chars().next().unwrap_or('\u{FFFD}');
                value.push(ch);
                *cursor += ch.len_utf8();
            }
            let _ = start; // suppress unused warning
        }

        // Check for language tag or datatype
        if *cursor < line.len() && line.as_bytes()[*cursor] == b'@' {
            *cursor += 1;
            let lang_start = *cursor;
            while *cursor < line.len()
                && !line.as_bytes()[*cursor].is_ascii_whitespace()
                && line.as_bytes()[*cursor] != b'.'
            {
                *cursor += 1;
            }
            let lang = line[lang_start..*cursor].to_string();
            return Ok(ParsedTerm::LangLiteral { value, lang });
        }

        if *cursor + 1 < line.len()
            && line.as_bytes()[*cursor] == b'^'
            && line.as_bytes()[*cursor + 1] == b'^'
        {
            *cursor += 2;
            // Parse the datatype IRI
            let dt_term = self.parse_iri_term(line, cursor)?;
            let datatype = match dt_term {
                ParsedTerm::Iri(iri) => iri,
                other => other.value().to_string(),
            };
            return Ok(ParsedTerm::TypedLiteral { value, datatype });
        }

        Ok(ParsedTerm::PlainLiteral(value))
    }

    // -----------------------------------------------------------------------
    // IRI resolution
    // -----------------------------------------------------------------------

    /// Resolve an IRI against the base IRI if it is relative
    pub fn resolve_iri(&self, iri: &str) -> String {
        if iri.is_empty() {
            return self.base_iri.clone().unwrap_or_default();
        }
        // Absolute IRI: contains scheme (colon before any slash)
        if let Some(colon_pos) = iri.find(':') {
            let before = &iri[..colon_pos];
            if !before.contains('/') && !before.contains('#') {
                return iri.to_string(); // Already absolute
            }
        }
        // Relative – resolve against base
        match &self.base_iri {
            Some(base) => {
                if iri.starts_with('#') {
                    // Fragment reference
                    let base_no_frag = base.split('#').next().unwrap_or(base);
                    format!("{}{}", base_no_frag, iri)
                } else if iri.starts_with('/') {
                    // Absolute path
                    resolve_absolute_path(base, iri)
                } else {
                    // Relative path
                    let base_dir = base.rfind('/').map(|p| &base[..=p]).unwrap_or(base);
                    format!("{}{}", base_dir, iri)
                }
            }
            None => iri.to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // String unescaping (shared)
    // -----------------------------------------------------------------------

    /// Unescape a Turtle/N-Triples string body (between the delimiters)
    pub fn unescape_string(s: &str) -> Result<String, ParseError> {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars();
        loop {
            match chars.next() {
                None => break,
                Some('\\') => match chars.next() {
                    None => return Err(ParseError::InvalidLiteral(s.to_string())),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('\'') => result.push('\''),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        let code_point = u32::from_str_radix(&hex, 16)
                            .map_err(|_| ParseError::InvalidLiteral(format!("\\u{}", hex)))?;
                        let ch = char::from_u32(code_point)
                            .ok_or_else(|| ParseError::InvalidLiteral(format!("\\u{}", hex)))?;
                        result.push(ch);
                    }
                    Some('U') => {
                        let hex: String = chars.by_ref().take(8).collect();
                        let code_point = u32::from_str_radix(&hex, 16)
                            .map_err(|_| ParseError::InvalidLiteral(format!("\\U{}", hex)))?;
                        let ch = char::from_u32(code_point)
                            .ok_or_else(|| ParseError::InvalidLiteral(format!("\\U{}", hex)))?;
                        result.push(ch);
                    }
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                },
                Some(ch) => result.push(ch),
            }
        }
        Ok(result)
    }
}

// -----------------------------------------------------------------------
// Turtle tokenizer (handles IRIs, literals, blank nodes, punctuation)
// -----------------------------------------------------------------------

/// Tokenize a single Turtle line into discrete tokens.
/// This is a simplified tokenizer that handles common cases.
fn tokenize_turtle(line: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut chars = line.chars().peekable();
    let mut current = String::new();

    macro_rules! push_current {
        () => {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        };
    }

    while let Some(&ch) = chars.peek() {
        // Comment
        if ch == '#' {
            break;
        }

        // IRI in angle brackets
        if ch == '<' {
            push_current!();
            current.push(ch);
            chars.next();
            while let Some(&inner) = chars.peek() {
                current.push(inner);
                chars.next();
                if inner == '>' {
                    break;
                }
            }
            push_current!();
            continue;
        }

        // String literal (double-quoted)
        if ch == '"' {
            push_current!();
            // Check for triple-quote
            let mut buf: Vec<char> = vec!['"'];
            chars.next();
            if chars.peek() == Some(&'"') {
                buf.push('"');
                chars.next();
                if chars.peek() == Some(&'"') {
                    buf.push('"');
                    chars.next();
                    // Triple-quoted string
                    let mut escaped = false;
                    loop {
                        match chars.next() {
                            None => break,
                            Some(c) => {
                                buf.push(c);
                                if escaped {
                                    escaped = false;
                                } else if c == '\\' {
                                    escaped = true;
                                } else if c == '"' && chars.peek() == Some(&'"') {
                                    buf.push('"');
                                    chars.next();
                                    if chars.peek() == Some(&'"') {
                                        buf.push('"');
                                        chars.next();
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Empty string ""  followed by something else
                    tokens.push(buf.iter().collect());
                    continue;
                }
            } else {
                // Regular double-quoted string
                let mut escaped = false;
                loop {
                    match chars.next() {
                        None => break,
                        Some(c) => {
                            buf.push(c);
                            if escaped {
                                escaped = false;
                            } else if c == '\\' {
                                escaped = true;
                            } else if c == '"' {
                                break;
                            }
                        }
                    }
                }
            }
            // Consume optional @lang or ^^type suffix
            if chars.peek() == Some(&'@') {
                buf.push('@');
                chars.next();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '-' {
                        buf.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
            } else if chars.peek() == Some(&'^') {
                buf.push('^');
                chars.next();
                if chars.peek() == Some(&'^') {
                    buf.push('^');
                    chars.next();
                }
                // Read the datatype IRI or prefixed name
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '.' || c == ';' || c == ',' {
                        break;
                    }
                    if c == '<' {
                        buf.push(c);
                        chars.next();
                        while let Some(&inner) = chars.peek() {
                            buf.push(inner);
                            chars.next();
                            if inner == '>' {
                                break;
                            }
                        }
                        break;
                    }
                    buf.push(c);
                    chars.next();
                }
            }
            tokens.push(buf.iter().collect());
            continue;
        }

        // Whitespace
        if ch.is_whitespace() {
            push_current!();
            chars.next();
            continue;
        }

        // Punctuation tokens
        if ch == '.' || ch == ';' || ch == ',' {
            push_current!();
            tokens.push(ch.to_string());
            chars.next();
            continue;
        }

        // All other characters (blank nodes, prefixed names, keywords, numbers)
        current.push(ch);
        chars.next();
    }

    push_current!();
    tokens
}

// -----------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------

/// Advance cursor past ASCII whitespace
fn skip_whitespace_in(s: &str, cursor: &mut usize) {
    while *cursor < s.len() && s.as_bytes()[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
}

/// Escape special characters in a string for N-Triples output
fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

/// Resolve an absolute-path reference against a base IRI
fn resolve_absolute_path(base: &str, path: &str) -> String {
    // Find the authority (scheme://host) portion
    if let Some(scheme_end) = base.find("://") {
        let after_scheme = &base[scheme_end + 3..];
        let authority_end = after_scheme.find('/').unwrap_or(after_scheme.len());
        let authority = &base[..scheme_end + 3 + authority_end];
        format!("{}{}", authority, path)
    } else {
        path.to_string()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntriples_basic() {
        let mut parser = StreamingParser::new(RdfFormat::NTriples);
        let nt = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n";
        let stmts = parser.feed(nt).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
        assert_eq!(
            stmts[0].subject(),
            &ParsedTerm::Iri("http://example.org/s".to_string())
        );
    }

    #[test]
    fn test_ntriples_literal() {
        let mut parser = StreamingParser::new(RdfFormat::NTriples);
        let nt = "<http://example.org/s> <http://example.org/name> \"Alice\" .\n";
        let stmts = parser.feed(nt).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
        assert_eq!(
            stmts[0].object(),
            &ParsedTerm::PlainLiteral("Alice".to_string())
        );
    }

    #[test]
    fn test_ntriples_lang_literal() {
        let mut parser = StreamingParser::new(RdfFormat::NTriples);
        let nt = "<http://s> <http://p> \"Bonjour\"@fr .\n";
        let stmts = parser.feed(nt).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
        assert_eq!(
            stmts[0].object(),
            &ParsedTerm::LangLiteral {
                value: "Bonjour".to_string(),
                lang: "fr".to_string()
            }
        );
    }

    #[test]
    fn test_ntriples_streaming_chunks() {
        let mut parser = StreamingParser::new(RdfFormat::NTriples);

        // Feed partial line
        let s1 = parser.feed("<http://example.org/s> ").expect("feed 1");
        assert_eq!(s1.len(), 0); // Incomplete line

        let s2 = parser
            .feed("<http://example.org/p> <http://example.org/o> .\n")
            .expect("feed 2");
        assert_eq!(s2.len(), 1);
    }

    #[test]
    fn test_ntriples_blank_node() {
        let mut parser = StreamingParser::new(RdfFormat::NTriples);
        let nt = "_:b1 <http://example.org/p> <http://example.org/o> .\n";
        let stmts = parser.feed(nt).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
        assert_eq!(stmts[0].subject(), &ParsedTerm::BlankNode("b1".to_string()));
    }

    #[test]
    fn test_ntriples_skip_comments() {
        let mut parser = StreamingParser::new(RdfFormat::NTriples);
        let nt = "# This is a comment\n<http://s> <http://p> <http://o> .\n# Another comment\n";
        let stmts = parser.feed(nt).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn test_nquads_with_graph() {
        let mut parser = StreamingParser::new(RdfFormat::NQuads);
        let nq = "<http://s> <http://p> <http://o> <http://g> .\n";
        let stmts = parser.feed(nq).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(stmts[0], ParsedStatement::Quad { .. }));
        assert_eq!(
            stmts[0].graph(),
            Some(&ParsedTerm::Iri("http://g".to_string()))
        );
    }

    #[test]
    fn test_turtle_prefix() {
        let mut parser = StreamingParser::new(RdfFormat::Turtle);
        let ttl = "@prefix ex: <http://example.org/> .\n\
                   ex:alice ex:knows ex:bob .\n";
        let stmts = parser.feed(ttl).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
        assert_eq!(
            stmts[0].subject(),
            &ParsedTerm::Iri("http://example.org/alice".to_string())
        );
    }

    #[test]
    fn test_turtle_rdf_type() {
        let mut parser = StreamingParser::new(RdfFormat::Turtle);
        parser.add_prefix("ex", "http://example.org/");
        parser.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        let ttl = "ex:alice a foaf:Person .\n";
        let stmts = parser.feed(ttl).expect("parse should succeed");
        assert_eq!(stmts.len(), 1);
        assert_eq!(
            stmts[0].predicate(),
            &ParsedTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
        );
    }

    #[test]
    fn test_turtle_semicolon() {
        let mut parser = StreamingParser::new(RdfFormat::Turtle);
        parser.add_prefix("ex", "http://example.org/");
        let ttl = "ex:alice ex:knows ex:bob ; ex:name \"Alice\" .\n";
        let stmts = parser.feed(ttl).expect("parse should succeed");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_turtle_comma() {
        let mut parser = StreamingParser::new(RdfFormat::Turtle);
        parser.add_prefix("ex", "http://example.org/");
        let ttl = "ex:alice ex:knows ex:bob , ex:carol .\n";
        let stmts = parser.feed(ttl).expect("parse should succeed");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_turtle_finish_flush() {
        let mut parser = StreamingParser::new(RdfFormat::NTriples);
        // Feed without trailing newline
        let partial = "<http://s> <http://p> <http://o> .";
        let s1 = parser.feed(partial).expect("feed");
        assert_eq!(s1.len(), 0); // Not complete yet

        let s2 = parser.finish().expect("finish");
        assert_eq!(s2.len(), 1);
    }

    #[test]
    fn test_unescape_string() {
        let result = StreamingParser::unescape_string("hello\\nworld").expect("unescape");
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_resolve_iri_absolute() {
        let parser = StreamingParser::new(RdfFormat::NTriples).with_base_iri("http://example.org/");
        assert_eq!(
            parser.resolve_iri("http://other.org/path"),
            "http://other.org/path"
        );
    }

    #[test]
    fn test_resolve_iri_relative() {
        let parser =
            StreamingParser::new(RdfFormat::NTriples).with_base_iri("http://example.org/data/");
        assert_eq!(parser.resolve_iri("item"), "http://example.org/data/item");
    }

    #[test]
    fn test_parsed_term_ntriples_format() {
        let iri = ParsedTerm::Iri("http://example.org/".to_string());
        assert_eq!(iri.to_ntriples_string(), "<http://example.org/>");

        let lit = ParsedTerm::LangLiteral {
            value: "hello".to_string(),
            lang: "en".to_string(),
        };
        assert_eq!(lit.to_ntriples_string(), "\"hello\"@en");

        let typed = ParsedTerm::TypedLiteral {
            value: "42".to_string(),
            datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
        };
        assert_eq!(
            typed.to_ntriples_string(),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }
}
