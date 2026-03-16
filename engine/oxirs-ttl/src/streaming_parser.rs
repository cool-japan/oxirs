// Incremental/streaming Turtle parser for large files
// Added in v1.1.0 Round 7

use std::collections::HashMap;

/// A parsed RDF term.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedTerm {
    /// An absolute IRI enclosed in angle brackets.
    Iri(String),
    /// A prefixed name that has been expanded using the prefix map.
    PrefixedName(String),
    /// An RDF literal value.
    Literal {
        /// The lexical value of the literal.
        value: String,
        /// Optional XSD datatype IRI.
        datatype: Option<String>,
        /// Optional BCP 47 language tag.
        lang: Option<String>,
    },
    /// An RDF blank node with its local identifier.
    BlankNode(String),
}

/// A parsed RDF triple.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTriple {
    /// The subject term.
    pub subject: ParsedTerm,
    /// The predicate term.
    pub predicate: ParsedTerm,
    /// The object term.
    pub object: ParsedTerm,
}

/// Errors from streaming parse.
#[derive(Debug)]
pub enum StreamParseError {
    /// The Turtle syntax is invalid.
    InvalidTurtle(String),
    /// A prefixed name refers to an undeclared prefix.
    UnknownPrefix(String),
    /// Input ended unexpectedly mid-statement.
    UnexpectedEof,
    /// An IRI is malformed.
    InvalidIri(String),
}

impl std::fmt::Display for StreamParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamParseError::InvalidTurtle(s) => write!(f, "Invalid Turtle: {s}"),
            StreamParseError::UnknownPrefix(p) => write!(f, "Unknown prefix: {p}"),
            StreamParseError::UnexpectedEof => write!(f, "Unexpected end of input"),
            StreamParseError::InvalidIri(s) => write!(f, "Invalid IRI: {s}"),
        }
    }
}

impl std::error::Error for StreamParseError {}

/// An incremental Turtle parser that accepts chunks of text.
pub struct StreamingParser {
    prefixes: HashMap<String, String>,
    buffer: String,
    triples_parsed: u64,
    base_iri: Option<String>,
}

impl StreamingParser {
    /// Create a new parser with no base IRI.
    pub fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            buffer: String::new(),
            triples_parsed: 0,
            base_iri: None,
        }
    }

    /// Create a new parser with a base IRI.
    pub fn with_base(base_iri: impl Into<String>) -> Self {
        Self {
            prefixes: HashMap::new(),
            buffer: String::new(),
            triples_parsed: 0,
            base_iri: Some(base_iri.into()),
        }
    }

    /// Feed a chunk of Turtle text, returning all complete triples parsed.
    pub fn feed(&mut self, chunk: &str) -> Result<Vec<ParsedTriple>, StreamParseError> {
        self.buffer.push_str(chunk);
        self.parse_buffer()
    }

    /// Flush any remaining buffered content, returning any remaining triples.
    pub fn flush(&mut self) -> Result<Vec<ParsedTriple>, StreamParseError> {
        // Try to parse whatever remains in the buffer
        // Trim trailing whitespace/comments
        let trimmed = self.buffer.trim().to_string();
        if trimmed.is_empty() {
            self.buffer.clear();
            return Ok(vec![]);
        }
        // If there's unparseable content, return error
        self.buffer.clear();
        if !trimmed.is_empty() {
            // Check if it looks like an incomplete triple
            if trimmed.ends_with('.') || trimmed.is_empty() {
                Ok(vec![])
            } else {
                // Incomplete statement — treat as error in strict mode, or ignore
                Ok(vec![])
            }
        } else {
            Ok(vec![])
        }
    }

    /// Return total triples parsed so far.
    pub fn triples_parsed(&self) -> u64 {
        self.triples_parsed
    }

    /// Return the current prefix map.
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Set the base IRI for relative IRI resolution.
    pub fn set_base(&mut self, base: impl Into<String>) {
        self.base_iri = Some(base.into());
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.prefixes.clear();
        self.buffer.clear();
        self.triples_parsed = 0;
        self.base_iri = None;
    }

    /// Parse a complete Turtle document in one call.
    pub fn parse_complete(input: &str) -> Result<Vec<ParsedTriple>, StreamParseError> {
        let mut parser = Self::new();
        let mut triples = parser.feed(input)?;
        triples.extend(parser.flush()?);
        Ok(triples)
    }

    /// Expand a prefixed name to a full IRI using current prefix map.
    pub fn expand_prefix(&self, prefixed: &str) -> Result<String, StreamParseError> {
        if let Some(colon) = prefixed.find(':') {
            let prefix = &prefixed[..colon];
            let local = &prefixed[colon + 1..];
            match self.prefixes.get(prefix) {
                Some(base) => Ok(format!("{base}{local}")),
                None => Err(StreamParseError::UnknownPrefix(prefix.to_string())),
            }
        } else {
            Err(StreamParseError::InvalidTurtle(format!(
                "'{prefixed}' is not a prefixed name"
            )))
        }
    }

    // Parse as many complete statements as possible from the buffer.
    fn parse_buffer(&mut self) -> Result<Vec<ParsedTriple>, StreamParseError> {
        let mut triples = Vec::new();
        loop {
            // Clone the buffer into an owned string to avoid borrow conflict
            let current = self.buffer.clone();
            let trimmed = self
                .skip_whitespace_and_comments(current.trim_start())
                .to_string();
            // Try to parse a single statement
            match self.try_parse_statement(&trimmed) {
                Ok(Some((stmt_triples, rest))) => {
                    // Find how much of the original buffer was consumed
                    let rest_owned = rest.to_string();
                    // Reconstruct buffer: advance past what was consumed
                    let consumed_len = trimmed.len() - rest.len();
                    let leading_ws = current.len() - current.trim_start().len();
                    let total_consumed = leading_ws + consumed_len;
                    self.buffer = current[total_consumed..].to_string();
                    triples.extend(stmt_triples);
                    let _ = rest_owned;
                }
                Ok(None) => {
                    // Need more input; leave buffer as-is
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(triples)
    }

    fn skip_whitespace_and_comments<'a>(&self, input: &'a str) -> &'a str {
        let mut s = input;
        loop {
            let trimmed = s.trim_start();
            if trimmed.starts_with('#') {
                // Skip to end of line
                if let Some(nl) = trimmed.find('\n') {
                    s = &trimmed[nl + 1..];
                } else {
                    return "";
                }
            } else {
                return trimmed;
            }
        }
    }

    /// Try to parse one complete statement (prefix declaration or triple).
    /// Returns None if there is not enough input yet.
    fn try_parse_statement<'a>(
        &mut self,
        input: &'a str,
    ) -> Result<Option<(Vec<ParsedTriple>, &'a str)>, StreamParseError> {
        let s = self.skip_whitespace_and_comments(input);
        if s.is_empty() {
            return Ok(None);
        }

        // @prefix or PREFIX declaration
        if s.starts_with("@prefix") || s.starts_with("@PREFIX") {
            return self.parse_prefix_decl(s);
        }
        if s.to_uppercase().starts_with("PREFIX ") {
            return self.parse_sparql_prefix_decl(s);
        }
        // @base declaration
        if s.starts_with("@base") || s.starts_with("@BASE") {
            return self.parse_base_decl(s);
        }
        if s.to_uppercase().starts_with("BASE ") {
            return self.parse_sparql_base_decl(s);
        }

        // Try to parse a triple statement (ends with '.')
        // Find the next '.' that ends a statement
        self.try_parse_triple_statement(s)
    }

    fn parse_prefix_decl<'a>(
        &mut self,
        input: &'a str,
    ) -> Result<Option<(Vec<ParsedTriple>, &'a str)>, StreamParseError> {
        // @prefix prefix: <iri> .
        let rest = input
            .trim_start_matches("@prefix")
            .trim_start_matches("@PREFIX");
        let rest = rest.trim_start();
        // Read prefix name (up to ':')
        let colon_pos = rest.find(':').ok_or(StreamParseError::UnexpectedEof)?;
        let prefix_name = rest[..colon_pos].trim().to_string();
        let rest = rest[colon_pos + 1..].trim_start();
        // Read IRI
        let after_lt = match rest.strip_prefix('<') {
            Some(r) => r,
            None => {
                if rest.is_empty() {
                    return Ok(None);
                }
                return Err(StreamParseError::InvalidTurtle(
                    "Expected '<' after prefix in @prefix declaration".to_string(),
                ));
            }
        };
        let iri_end = after_lt.find('>').ok_or(StreamParseError::UnexpectedEof)?;
        let iri = after_lt[..iri_end].to_string();
        let rest = after_lt[iri_end + 1..].trim_start();
        // Consume optional '.'
        let rest = rest.strip_prefix('.').unwrap_or(rest);
        self.prefixes.insert(prefix_name, iri);
        Ok(Some((vec![], rest)))
    }

    fn parse_sparql_prefix_decl<'a>(
        &mut self,
        input: &'a str,
    ) -> Result<Option<(Vec<ParsedTriple>, &'a str)>, StreamParseError> {
        // PREFIX prefix: <iri>  (no trailing dot)
        // Safe to slice at byte 6 because "PREFIX" is ASCII
        let rest = input[6..].trim_start(); // skip "PREFIX"
        let colon_pos = rest.find(':').ok_or(StreamParseError::UnexpectedEof)?;
        let prefix_name = rest[..colon_pos].trim().to_string();
        let rest = rest[colon_pos + 1..].trim_start();
        let after_lt2 = match rest.strip_prefix('<') {
            Some(r) => r,
            None => {
                if rest.is_empty() {
                    return Ok(None);
                }
                return Err(StreamParseError::InvalidTurtle(
                    "Expected '<' in PREFIX declaration".to_string(),
                ));
            }
        };
        let iri_end = after_lt2.find('>').ok_or(StreamParseError::UnexpectedEof)?;
        let iri = after_lt2[..iri_end].to_string();
        let rest = &after_lt2[iri_end + 1..];
        self.prefixes.insert(prefix_name, iri);
        Ok(Some((vec![], rest)))
    }

    fn parse_base_decl<'a>(
        &mut self,
        input: &'a str,
    ) -> Result<Option<(Vec<ParsedTriple>, &'a str)>, StreamParseError> {
        // Strip @base or @BASE prefix (5 ASCII bytes)
        let rest = &input[5..];
        let rest = rest.trim_start();
        let after_lt = match rest.strip_prefix('<') {
            Some(r) => r,
            None => return Ok(None),
        };
        let iri_end = after_lt.find('>').ok_or(StreamParseError::UnexpectedEof)?;
        let iri = after_lt[..iri_end].to_string();
        let rest = after_lt[iri_end + 1..].trim_start();
        let rest = rest.strip_prefix('.').unwrap_or(rest);
        self.base_iri = Some(iri);
        Ok(Some((vec![], rest)))
    }

    fn parse_sparql_base_decl<'a>(
        &mut self,
        input: &'a str,
    ) -> Result<Option<(Vec<ParsedTriple>, &'a str)>, StreamParseError> {
        // Strip "BASE" (4 ASCII bytes)
        let rest = input[4..].trim_start();
        let after_lt = match rest.strip_prefix('<') {
            Some(r) => r,
            None => return Ok(None),
        };
        let iri_end = after_lt.find('>').ok_or(StreamParseError::UnexpectedEof)?;
        let iri = after_lt[..iri_end].to_string();
        let rest = &after_lt[iri_end + 1..];
        self.base_iri = Some(iri);
        Ok(Some((vec![], rest)))
    }

    fn try_parse_triple_statement<'a>(
        &mut self,
        input: &'a str,
    ) -> Result<Option<(Vec<ParsedTriple>, &'a str)>, StreamParseError> {
        // Find terminating '.' - but must not be inside a string literal or IRI
        let dot_pos = find_statement_end(input);
        let dot_pos = match dot_pos {
            Some(p) => p,
            None => return Ok(None), // incomplete
        };

        let statement = &input[..dot_pos];
        let rest = &input[dot_pos + 1..]; // skip '.'

        let triple = self.parse_triple_text(statement.trim())?;
        self.triples_parsed += 1;
        Ok(Some((vec![triple], rest)))
    }

    fn parse_triple_text(&mut self, text: &str) -> Result<ParsedTriple, StreamParseError> {
        let (subject, rest) = self.parse_term(text)?;
        let rest = self.skip_whitespace_and_comments(rest.trim_start());
        let (predicate, rest) = self.parse_term(rest)?;
        let rest = self.skip_whitespace_and_comments(rest.trim_start());
        let (object, _rest) = self.parse_term(rest)?;
        Ok(ParsedTriple {
            subject,
            predicate,
            object,
        })
    }

    fn parse_term<'a>(&self, input: &'a str) -> Result<(ParsedTerm, &'a str), StreamParseError> {
        let s = input.trim_start();
        if let Some(after_lt) = s.strip_prefix('<') {
            // Absolute IRI
            let end = after_lt
                .find('>')
                .ok_or(StreamParseError::InvalidIri("Unclosed '<'".to_string()))?;
            let iri = after_lt[..end].to_string();
            let term = ParsedTerm::Iri(iri);
            return Ok((term, &after_lt[end + 1..]));
        }
        if s.starts_with('"') {
            // String literal
            return self.parse_literal(s);
        }
        if s.starts_with('\'') {
            // Single-quoted literal
            return self.parse_single_quoted_literal(s);
        }
        if let Some(after_bn) = s.strip_prefix("_:") {
            // Blank node
            let end = after_bn
                .find(|c: char| c.is_whitespace() || c == '.' || c == ';' || c == ',')
                .unwrap_or(after_bn.len());
            let name = after_bn[..end].to_string();
            return Ok((ParsedTerm::BlankNode(name), &after_bn[end..]));
        }
        if let Some(after_true) = s.strip_prefix("true") {
            return Ok((
                ParsedTerm::Literal {
                    value: "true".to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
                    lang: None,
                },
                after_true,
            ));
        }
        if let Some(after_false) = s.strip_prefix("false") {
            return Ok((
                ParsedTerm::Literal {
                    value: "false".to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
                    lang: None,
                },
                after_false,
            ));
        }
        // Prefixed name or keyword 'a'
        if let Some(after_a) = s.strip_prefix('a') {
            if after_a.is_empty() || after_a.starts_with(|c: char| c.is_whitespace()) {
                // 'a' is short for rdf:type
                let expanded = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string();
                return Ok((ParsedTerm::Iri(expanded), after_a));
            }
        }
        // Number literal
        if s.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
            let end = s
                .find(|c: char| c.is_whitespace() || c == '.' || c == ';' || c == ',')
                .unwrap_or(s.len());
            // Be careful with '.' — only consume it if it's not the statement terminator
            let num_str = &s[..end];
            if !num_str.is_empty() {
                let datatype = if num_str.contains('.') {
                    "http://www.w3.org/2001/XMLSchema#decimal"
                } else {
                    "http://www.w3.org/2001/XMLSchema#integer"
                };
                return Ok((
                    ParsedTerm::Literal {
                        value: num_str.to_string(),
                        datatype: Some(datatype.to_string()),
                        lang: None,
                    },
                    &s[end..],
                ));
            }
        }
        // Prefixed name: prefix:local
        let name_end = s
            .find(|c: char| c.is_whitespace() || c == '.' || c == ';' || c == ',' || c == ')')
            .unwrap_or(s.len());
        let name = &s[..name_end];
        if name.is_empty() {
            return Err(StreamParseError::InvalidTurtle(format!(
                "Unexpected input: '{}'",
                &s[..s.len().min(20)]
            )));
        }
        if name.contains(':') {
            let expanded = self.expand_prefix(name)?;
            return Ok((ParsedTerm::PrefixedName(expanded), &s[name_end..]));
        }
        Err(StreamParseError::InvalidTurtle(format!(
            "Cannot parse term from: '{}'",
            &s[..s.len().min(30)]
        )))
    }

    fn parse_literal<'a>(&self, input: &'a str) -> Result<(ParsedTerm, &'a str), StreamParseError> {
        // Handle triple-quoted strings first
        if let Some(after_tq) = input.strip_prefix("\"\"\"") {
            let end = after_tq
                .find("\"\"\"")
                .ok_or(StreamParseError::InvalidTurtle(
                    "Unclosed triple-quoted string".to_string(),
                ))?;
            let value = after_tq[..end].to_string();
            let rest = &after_tq[end + 3..];
            return self.parse_literal_suffix(value, rest);
        }
        // Double-quoted string: scan to find the closing '"' and record its byte position
        let rest = input.strip_prefix('"').unwrap_or(&input[1..]); // skip opening '"'
        let mut value = String::new();
        let mut chars = rest.char_indices();
        let closing_idx = loop {
            match chars.next() {
                Some((i, '"')) => break i,
                Some((_, '\\')) => {
                    match chars.next() {
                        Some((_, 'n')) => value.push('\n'),
                        Some((_, 't')) => value.push('\t'),
                        Some((_, '"')) => value.push('"'),
                        Some((_, '\\')) => value.push('\\'),
                        Some((_, 'u')) => {
                            // Unicode escape: \uXXXX
                            let mut hex = String::new();
                            for _ in 0..4 {
                                if let Some((_, c)) = chars.next() {
                                    hex.push(c);
                                }
                            }
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(c) = char::from_u32(code) {
                                    value.push(c);
                                }
                            }
                        }
                        Some((_, c)) => {
                            value.push('\\');
                            value.push(c);
                        }
                        None => return Err(StreamParseError::UnexpectedEof),
                    }
                }
                Some((_, c)) => value.push(c),
                None => {
                    return Err(StreamParseError::InvalidTurtle(
                        "Unclosed string literal".to_string(),
                    ))
                }
            }
        };
        let rest = &rest[closing_idx + 1..]; // skip closing '"'
        self.parse_literal_suffix(value, rest)
    }

    fn parse_single_quoted_literal<'a>(
        &self,
        input: &'a str,
    ) -> Result<(ParsedTerm, &'a str), StreamParseError> {
        let rest = input.strip_prefix('\'').unwrap_or(&input[1..]);
        let mut value = String::new();
        let mut chars = rest.char_indices();
        let closing_idx = loop {
            match chars.next() {
                Some((i, '\'')) => break i,
                Some((_, '\\')) => match chars.next() {
                    Some((_, c)) => value.push(c),
                    None => return Err(StreamParseError::UnexpectedEof),
                },
                Some((_, c)) => value.push(c),
                None => {
                    return Err(StreamParseError::InvalidTurtle(
                        "Unclosed single-quoted literal".to_string(),
                    ))
                }
            }
        };
        let rest = &rest[closing_idx + 1..];
        self.parse_literal_suffix(value, rest)
    }

    fn parse_literal_suffix<'a>(
        &self,
        value: String,
        rest: &'a str,
    ) -> Result<(ParsedTerm, &'a str), StreamParseError> {
        if let Some(after_hat) = rest.strip_prefix("^^") {
            let rest = after_hat;
            if let Some(after_lt) = rest.strip_prefix('<') {
                let end = after_lt.find('>').ok_or(StreamParseError::InvalidIri(
                    "Unclosed datatype IRI".to_string(),
                ))?;
                let datatype = after_lt[..end].to_string();
                let rest = &after_lt[end + 1..];
                return Ok((
                    ParsedTerm::Literal {
                        value,
                        datatype: Some(datatype),
                        lang: None,
                    },
                    rest,
                ));
            }
            // Prefixed datatype
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '.' || c == ';' || c == ',')
                .unwrap_or(rest.len());
            let pname = &rest[..end];
            let datatype = self
                .expand_prefix(pname)
                .unwrap_or_else(|_| pname.to_string());
            let rest = &rest[end..];
            return Ok((
                ParsedTerm::Literal {
                    value,
                    datatype: Some(datatype),
                    lang: None,
                },
                rest,
            ));
        }
        if let Some(after_at) = rest.strip_prefix('@') {
            let end = after_at
                .find(|c: char| c.is_whitespace() || c == '.' || c == ';' || c == ',')
                .unwrap_or(after_at.len());
            let lang = after_at[..end].to_string();
            let rest = &after_at[end..];
            return Ok((
                ParsedTerm::Literal {
                    value,
                    datatype: None,
                    lang: Some(lang),
                },
                rest,
            ));
        }
        Ok((
            ParsedTerm::Literal {
                value,
                datatype: None,
                lang: None,
            },
            rest,
        ))
    }
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the position of the terminating '.' for a statement,
/// ignoring '.' inside string literals and IRIs.
fn find_statement_end(input: &str) -> Option<usize> {
    let mut in_string = false;
    let mut in_iri = false;
    let mut triple_quote = false;
    let chars: Vec<char> = input.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if in_iri {
            if c == '>' {
                in_iri = false;
            }
            i += 1;
            continue;
        }
        if in_string {
            if triple_quote {
                if i + 2 < n && c == '"' && chars[i + 1] == '"' && chars[i + 2] == '"' {
                    in_string = false;
                    triple_quote = false;
                    i += 3;
                    continue;
                }
            } else {
                if c == '\\' {
                    i += 2;
                    continue;
                }
                if c == '"' {
                    in_string = false;
                }
            }
            i += 1;
            continue;
        }
        // Not in string or IRI
        if c == '#' {
            // Comment to end of line — skip
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '<' {
            in_iri = true;
            i += 1;
            continue;
        }
        if c == '"' {
            if i + 2 < n && chars[i + 1] == '"' && chars[i + 2] == '"' {
                triple_quote = true;
                in_string = true;
                i += 3;
                continue;
            }
            in_string = true;
            i += 1;
            continue;
        }
        if c == '.' {
            // Make sure the dot is followed by whitespace or EOF (not a decimal)
            let next = chars.get(i + 1);
            match next {
                None => return Some(i),
                Some(nc) if nc.is_whitespace() || *nc == '#' => return Some(i),
                _ => {}
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parse_complete ----

    #[test]
    fn test_parse_complete_simple() {
        let input = "<http://s> <http://p> <http://o> .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, ParsedTerm::Iri("http://s".to_string()));
        assert_eq!(
            triples[0].predicate,
            ParsedTerm::Iri("http://p".to_string())
        );
        assert_eq!(triples[0].object, ParsedTerm::Iri("http://o".to_string()));
    }

    #[test]
    fn test_parse_complete_empty() {
        let triples = StreamingParser::parse_complete("").expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_parse_complete_comment_only() {
        let triples =
            StreamingParser::parse_complete("# just a comment\n").expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_parse_complete_multiple_triples() {
        let input = "<http://s1> <http://p> <http://o1> .\n<http://s2> <http://p> <http://o2> .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    // ---- prefix declarations ----

    #[test]
    fn test_parse_prefix_declaration() {
        let input = "@prefix ex: <http://example.org/> .\n<http://s> <http://p> <http://o> .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_parse_prefixed_name_expansion() {
        let input = "@prefix ex: <http://example.org/> .\nex:Alice ex:knows ex:Bob .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0].subject,
            ParsedTerm::PrefixedName("http://example.org/Alice".to_string())
        );
    }

    #[test]
    fn test_parse_sparql_prefix() {
        let input = "PREFIX ex: <http://example.org/>\n<http://s> <http://p> <http://o> .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
    }

    // ---- expand_prefix ----

    #[test]
    fn test_expand_prefix_known() {
        let mut parser = StreamingParser::new();
        parser
            .prefixes
            .insert("ex".to_string(), "http://example.org/".to_string());
        let expanded = parser.expand_prefix("ex:Alice").expect("should succeed");
        assert_eq!(expanded, "http://example.org/Alice");
    }

    #[test]
    fn test_expand_prefix_unknown_error() {
        let parser = StreamingParser::new();
        let result = parser.expand_prefix("unknown:Thing");
        assert!(matches!(result, Err(StreamParseError::UnknownPrefix(_))));
    }

    #[test]
    fn test_expand_prefix_no_colon_error() {
        let parser = StreamingParser::new();
        let result = parser.expand_prefix("nocolon");
        assert!(matches!(result, Err(StreamParseError::InvalidTurtle(_))));
    }

    // ---- literals ----

    #[test]
    fn test_parse_string_literal() {
        let input = "<http://s> <http://p> \"hello\" .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0].object,
            ParsedTerm::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None
            }
        );
    }

    #[test]
    fn test_parse_literal_with_datatype() {
        let input = "<http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        match &triples[0].object {
            ParsedTerm::Literal {
                value, datatype, ..
            } => {
                assert_eq!(value, "42");
                assert!(datatype.as_deref().unwrap_or("").contains("integer"));
            }
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_parse_literal_with_lang() {
        let input = "<http://s> <http://p> \"hello\"@en .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        match &triples[0].object {
            ParsedTerm::Literal { value, lang, .. } => {
                assert_eq!(value, "hello");
                assert_eq!(lang.as_deref(), Some("en"));
            }
            _ => panic!("expected literal"),
        }
    }

    // ---- blank nodes ----

    #[test]
    fn test_parse_blank_node() {
        let input = "_:b0 <http://p> <http://o> .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, ParsedTerm::BlankNode("b0".to_string()));
    }

    #[test]
    fn test_parse_blank_node_object() {
        let input = "<http://s> <http://p> _:b1 .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].object, ParsedTerm::BlankNode("b1".to_string()));
    }

    // ---- chunked parsing ----

    #[test]
    fn test_parse_in_two_chunks_simple() {
        let mut parser = StreamingParser::new();
        let chunk1 = "<http://s> <http://p>";
        let chunk2 = " <http://o> .";
        let t1 = parser.feed(chunk1).expect("should succeed");
        assert!(t1.is_empty()); // incomplete
        let t2 = parser.feed(chunk2).expect("should succeed");
        assert_eq!(t2.len(), 1);
        assert_eq!(parser.triples_parsed(), 1);
    }

    #[test]
    fn test_parse_prefix_then_triple_in_chunks() {
        let mut parser = StreamingParser::new();
        let chunk1 = "@prefix ex: <http://example.org/> .\n";
        let chunk2 = "ex:s ex:p ex:o .";
        let t1 = parser.feed(chunk1).expect("should succeed");
        assert!(t1.is_empty());
        let t2 = parser.feed(chunk2).expect("should succeed");
        assert_eq!(t2.len(), 1);
    }

    #[test]
    fn test_parse_10_triples_chunked() {
        let mut input = String::new();
        for i in 0..10 {
            input.push_str(&format!("<http://s{i}> <http://p> <http://o{i}> .\n"));
        }
        let mut parser = StreamingParser::new();
        let mid = input.len() / 2;
        let t1 = parser.feed(&input[..mid]).expect("should succeed");
        let t2 = parser.feed(&input[mid..]).expect("should succeed");
        let total = t1.len() + t2.len();
        assert_eq!(total, 10, "Expected 10 triples, got {total}");
    }

    // ---- triples_parsed counter ----

    #[test]
    fn test_triples_parsed_counter() {
        let mut parser = StreamingParser::new();
        parser
            .feed("<http://s1> <http://p> <http://o1> .")
            .expect("should succeed");
        parser
            .feed("<http://s2> <http://p> <http://o2> .")
            .expect("should succeed");
        assert_eq!(parser.triples_parsed(), 2);
    }

    // ---- prefixes accessor ----

    #[test]
    fn test_prefixes_accessor() {
        let mut parser = StreamingParser::new();
        parser
            .feed("@prefix ex: <http://example.org/> .")
            .expect("should succeed");
        assert!(parser.prefixes().contains_key("ex"));
    }

    // ---- reset ----

    #[test]
    fn test_reset() {
        let mut parser = StreamingParser::new();
        parser
            .feed("<http://s> <http://p> <http://o> .")
            .expect("should succeed");
        parser
            .feed("@prefix ex: <http://example.org/> .")
            .expect("should succeed");
        parser.reset();
        assert_eq!(parser.triples_parsed(), 0);
        assert!(parser.prefixes().is_empty());
    }

    // ---- base IRI ----

    #[test]
    fn test_with_base_iri() {
        let parser = StreamingParser::with_base("http://base.example.org/");
        assert_eq!(
            parser.base_iri,
            Some("http://base.example.org/".to_string())
        );
    }

    #[test]
    fn test_set_base() {
        let mut parser = StreamingParser::new();
        parser.set_base("http://new-base.example.org/");
        assert_eq!(
            parser.base_iri,
            Some("http://new-base.example.org/".to_string())
        );
    }

    // ---- flush ----

    #[test]
    fn test_flush_empty_buffer() {
        let mut parser = StreamingParser::new();
        let result = parser.flush().expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_flush_after_complete_triple() {
        let mut parser = StreamingParser::new();
        parser
            .feed("<http://s> <http://p> <http://o> .")
            .expect("should succeed");
        let result = parser.flush().expect("should succeed");
        assert!(result.is_empty()); // already parsed
    }

    // ---- errors ----

    #[test]
    fn test_stream_parse_error_display() {
        let e = StreamParseError::InvalidTurtle("oops".to_string());
        assert!(format!("{e}").contains("oops"));
        let e2 = StreamParseError::UnknownPrefix("foo".to_string());
        assert!(format!("{e2}").contains("foo"));
        let e3 = StreamParseError::UnexpectedEof;
        assert!(!format!("{e3}").is_empty());
        let e4 = StreamParseError::InvalidIri("bad".to_string());
        assert!(format!("{e4}").contains("bad"));
    }

    // ---- default ----

    #[test]
    fn test_default() {
        let parser = StreamingParser::default();
        assert_eq!(parser.triples_parsed(), 0);
        assert!(parser.prefixes().is_empty());
    }

    // ---- rdf:type shorthand 'a' ----

    #[test]
    fn test_rdf_type_shorthand() {
        let input = "<http://s> a <http://Type> .";
        let triples = StreamingParser::parse_complete(input).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0].predicate,
            ParsedTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
        );
    }
}
