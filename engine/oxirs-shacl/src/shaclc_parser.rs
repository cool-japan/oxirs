//! # SHACL Compact Syntax (SHACLC) Parser
//!
//! Parses the SHACL Compact Syntax — a human-friendly text format for defining
//! SHACL shapes without writing verbose Turtle.
//!
//! ## SHACLC Grammar (simplified)
//!
//! ```text
//! shaclDoc   ::= (prefix | import)* (nodeShape | propertyShape)*
//! prefix     ::= 'PREFIX' PNAME ':' IRI
//! import     ::= 'IMPORTS' IRI
//! nodeShape  ::= 'shape' iri targetDecl? '{' propertyDecl* '}'
//! targetDecl ::= '->' 'targetNode' '(' iri (',' iri)* ')'
//!              | '->' 'targetClass' '(' iri (',' iri)* ')'
//!              | '->' 'targetSubjectsOf' '(' iri ')'
//!              | '->' 'targetObjectsOf' '(' iri ')'
//! propertyDecl ::= path ':' constraintList ';'
//! ```
//!
//! ## Example
//!
//! ```text
//! PREFIX ex: <http://example.org/>
//! PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
//!
//! shape ex:PersonShape -> targetClass(ex:Person) {
//!   ex:name xsd:string [1..1] ;
//!   ex:age  xsd:integer [0..1] minInclusive=0 ;
//! }
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_shacl::shaclc_parser::{ShaclcParser, ShaclcDocument};
//!
//! let input = r#"
//! PREFIX ex: <http://example.org/>
//! shape ex:PersonShape -> targetClass(ex:Person) {
//!   ex:name xsd:string [1..1] ;
//! }
//! "#;
//!
//! let doc = ShaclcParser::parse(input).unwrap();
//! assert_eq!(doc.shapes.len(), 1);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ShaclError;

// ---------------------------------------------------------------------------
// AST types
// ---------------------------------------------------------------------------

/// A parsed SHACLC document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclcDocument {
    /// Prefix declarations (short -> full IRI).
    pub prefixes: HashMap<String, String>,
    /// Import directives.
    pub imports: Vec<String>,
    /// Parsed shapes.
    pub shapes: Vec<ShaclcShape>,
}

/// A parsed SHACL shape from compact syntax.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclcShape {
    /// Shape IRI (expanded).
    pub iri: String,
    /// Target declarations.
    pub targets: Vec<ShaclcTarget>,
    /// Property constraints declared inside the shape.
    pub properties: Vec<ShaclcPropertyDecl>,
    /// Whether this is deactivated.
    pub deactivated: bool,
    /// Severity (optional, defaults to Violation).
    pub severity: Option<String>,
    /// rdfs:label / message
    pub label: Option<String>,
}

/// A target declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShaclcTarget {
    /// sh:targetNode
    TargetNode(Vec<String>),
    /// sh:targetClass
    TargetClass(Vec<String>),
    /// sh:targetSubjectsOf
    TargetSubjectsOf(String),
    /// sh:targetObjectsOf
    TargetObjectsOf(String),
}

/// A property constraint declaration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShaclcPropertyDecl {
    /// Property path IRI (expanded).
    pub path: String,
    /// Datatype constraint (e.g. `xsd:string`).
    pub datatype: Option<String>,
    /// sh:class constraint.
    pub class: Option<String>,
    /// sh:nodeKind constraint.
    pub node_kind: Option<String>,
    /// Cardinality: (minCount, maxCount). `None` means unbounded.
    pub min_count: Option<u32>,
    /// Maximum cardinality. `None` means unbounded.
    pub max_count: Option<u32>,
    /// sh:minInclusive.
    pub min_inclusive: Option<f64>,
    /// sh:maxInclusive.
    pub max_inclusive: Option<f64>,
    /// sh:minExclusive.
    pub min_exclusive: Option<f64>,
    /// sh:maxExclusive.
    pub max_exclusive: Option<f64>,
    /// sh:pattern (regex).
    pub pattern: Option<String>,
    /// sh:minLength.
    pub min_length: Option<u32>,
    /// sh:maxLength.
    pub max_length: Option<u32>,
    /// sh:in (list of allowed values).
    pub in_values: Vec<String>,
    /// sh:hasValue.
    pub has_value: Option<String>,
    /// sh:uniqueLang.
    pub unique_lang: bool,
}

// ---------------------------------------------------------------------------
// Well-known prefixes
// ---------------------------------------------------------------------------

fn default_prefixes() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "xsd".to_string(),
        "http://www.w3.org/2001/XMLSchema#".to_string(),
    );
    m.insert(
        "rdf".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
    );
    m.insert(
        "rdfs".to_string(),
        "http://www.w3.org/2000/01/rdf-schema#".to_string(),
    );
    m.insert("sh".to_string(), "http://www.w3.org/ns/shacl#".to_string());
    m
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// The SHACLC parser.
pub struct ShaclcParser;

impl ShaclcParser {
    /// Parse a complete SHACLC document.
    pub fn parse(input: &str) -> Result<ShaclcDocument, ShaclError> {
        let mut state = ParseState::new(input);
        let mut doc = ShaclcDocument {
            prefixes: default_prefixes(),
            imports: Vec::new(),
            shapes: Vec::new(),
        };

        state.skip_ws_and_comments();

        while !state.is_eof() {
            state.skip_ws_and_comments();
            if state.is_eof() {
                break;
            }

            if state.peek_keyword("PREFIX") || state.peek_keyword("prefix") {
                let (prefix, iri) = Self::parse_prefix(&mut state)?;
                doc.prefixes.insert(prefix, iri);
            } else if state.peek_keyword("IMPORTS") || state.peek_keyword("imports") {
                let iri = Self::parse_import(&mut state)?;
                doc.imports.push(iri);
            } else if state.peek_keyword("shape") || state.peek_keyword("SHAPE") {
                let shape = Self::parse_shape(&mut state, &doc.prefixes)?;
                doc.shapes.push(shape);
            } else {
                let remaining: String = state.remaining().chars().take(30).collect();
                return Err(ShaclError::ShapeParsing(format!(
                    "Unexpected token at position {}: '{remaining}...'",
                    state.pos
                )));
            }
        }

        Ok(doc)
    }

    /// Parse a PREFIX declaration.
    fn parse_prefix(state: &mut ParseState<'_>) -> Result<(String, String), ShaclError> {
        state.expect_keyword_ci("prefix")?;
        state.skip_ws();
        let prefix = state.take_until(':')?;
        state.expect_char(':')?;
        state.skip_ws();
        let iri = state.take_iri()?;
        Ok((prefix, iri))
    }

    /// Parse an IMPORTS directive.
    fn parse_import(state: &mut ParseState<'_>) -> Result<String, ShaclError> {
        state.expect_keyword_ci("imports")?;
        state.skip_ws();
        state.take_iri()
    }

    /// Parse a shape declaration.
    fn parse_shape(
        state: &mut ParseState<'_>,
        prefixes: &HashMap<String, String>,
    ) -> Result<ShaclcShape, ShaclError> {
        state.expect_keyword_ci("shape")?;
        state.skip_ws();

        let iri_raw = state.take_prefixed_or_iri()?;
        let iri = expand_iri(&iri_raw, prefixes)?;

        state.skip_ws();

        // Optional target declarations
        let mut targets = Vec::new();
        while state.peek_str("->") {
            state.advance(2);
            state.skip_ws();
            let target = Self::parse_target(state, prefixes)?;
            targets.push(target);
            state.skip_ws();
        }

        // Parse optional shape-level attributes before '{'
        let mut deactivated = false;
        let mut severity = None;
        let mut label = None;
        state.skip_ws();
        while !state.peek_char('{') && !state.is_eof() {
            if state.peek_keyword("deactivated") {
                state.expect_keyword_ci("deactivated")?;
                deactivated = true;
            } else if state.peek_keyword("severity") {
                state.expect_keyword_ci("severity")?;
                state.skip_ws();
                state.expect_char('=')?;
                state.skip_ws();
                severity = Some(state.take_word()?);
            } else if state.peek_keyword("label") {
                state.expect_keyword_ci("label")?;
                state.skip_ws();
                state.expect_char('=')?;
                state.skip_ws();
                label = Some(state.take_quoted_string()?);
            } else {
                break;
            }
            state.skip_ws();
        }

        state.expect_char('{')?;
        state.skip_ws_and_comments();

        // Property declarations
        let mut properties = Vec::new();
        while !state.peek_char('}') && !state.is_eof() {
            let prop = Self::parse_property_decl(state, prefixes)?;
            properties.push(prop);
            state.skip_ws_and_comments();
        }

        state.expect_char('}')?;

        Ok(ShaclcShape {
            iri,
            targets,
            properties,
            deactivated,
            severity,
            label,
        })
    }

    /// Parse a target declaration.
    fn parse_target(
        state: &mut ParseState<'_>,
        prefixes: &HashMap<String, String>,
    ) -> Result<ShaclcTarget, ShaclError> {
        let kind = state.take_word()?;
        state.skip_ws();
        state.expect_char('(')?;
        state.skip_ws();

        let kind_lower = kind.to_lowercase();
        match kind_lower.as_str() {
            "targetnode" => {
                let iris = Self::parse_iri_list(state, prefixes)?;
                state.expect_char(')')?;
                Ok(ShaclcTarget::TargetNode(iris))
            }
            "targetclass" => {
                let iris = Self::parse_iri_list(state, prefixes)?;
                state.expect_char(')')?;
                Ok(ShaclcTarget::TargetClass(iris))
            }
            "targetsubjectsof" => {
                let iri_raw = state.take_prefixed_or_iri()?;
                let iri = expand_iri(&iri_raw, prefixes)?;
                state.skip_ws();
                state.expect_char(')')?;
                Ok(ShaclcTarget::TargetSubjectsOf(iri))
            }
            "targetobjectsof" => {
                let iri_raw = state.take_prefixed_or_iri()?;
                let iri = expand_iri(&iri_raw, prefixes)?;
                state.skip_ws();
                state.expect_char(')')?;
                Ok(ShaclcTarget::TargetObjectsOf(iri))
            }
            _ => Err(ShaclError::ShapeParsing(format!(
                "Unknown target type: {kind}"
            ))),
        }
    }

    fn parse_iri_list(
        state: &mut ParseState<'_>,
        prefixes: &HashMap<String, String>,
    ) -> Result<Vec<String>, ShaclError> {
        let mut iris = Vec::new();
        let raw = state.take_prefixed_or_iri()?;
        iris.push(expand_iri(&raw, prefixes)?);
        state.skip_ws();
        while state.peek_char(',') {
            state.advance(1);
            state.skip_ws();
            let raw = state.take_prefixed_or_iri()?;
            iris.push(expand_iri(&raw, prefixes)?);
            state.skip_ws();
        }
        Ok(iris)
    }

    /// Parse a single property declaration line.
    fn parse_property_decl(
        state: &mut ParseState<'_>,
        prefixes: &HashMap<String, String>,
    ) -> Result<ShaclcPropertyDecl, ShaclError> {
        let path_raw = state.take_prefixed_or_iri()?;
        let path = expand_iri(&path_raw, prefixes)?;

        let mut decl = ShaclcPropertyDecl {
            path,
            ..Default::default()
        };

        state.skip_ws();

        // Datatype or class or nodeKind (next token before constraints)
        // Skip this block if the next token is a constraint keyword (key=value pair)
        let is_constraint_keyword = state.peek_keyword("minInclusive")
            || state.peek_keyword("maxInclusive")
            || state.peek_keyword("minExclusive")
            || state.peek_keyword("maxExclusive")
            || state.peek_keyword("pattern")
            || state.peek_keyword("minLength")
            || state.peek_keyword("maxLength")
            || state.peek_keyword("in")
            || state.peek_keyword("hasValue")
            || state.peek_keyword("uniqueLang")
            || state.peek_keyword("nodeKind")
            || state.peek_keyword("datatype")
            || state.peek_keyword("class");

        if !is_constraint_keyword
            && !state.peek_char('[')
            && !state.peek_char(';')
            && !state.peek_char('}')
            && !state.is_eof()
        {
            // Check for nodeKind keywords
            if state.peek_keyword("IRI")
                || state.peek_keyword("BlankNode")
                || state.peek_keyword("Literal")
                || state.peek_keyword("BlankNodeOrIRI")
                || state.peek_keyword("BlankNodeOrLiteral")
                || state.peek_keyword("IRIOrLiteral")
            {
                decl.node_kind = Some(state.take_word()?);
            } else {
                let type_raw = state.take_prefixed_or_iri()?;
                let type_expanded = expand_iri(&type_raw, prefixes)?;
                // Heuristic: if it looks like xsd: datatype, set datatype; else set class
                if type_expanded.starts_with("http://www.w3.org/2001/XMLSchema#") {
                    decl.datatype = Some(type_expanded);
                } else {
                    decl.class = Some(type_expanded);
                }
            }
            state.skip_ws();
        }

        // Cardinality: [min..max]
        if state.peek_char('[') {
            state.advance(1);
            state.skip_ws();
            let min_str = state.take_while(|c| c.is_ascii_digit() || c == '*')?;
            let min = if min_str == "*" {
                None
            } else {
                Some(
                    min_str
                        .parse::<u32>()
                        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid min: {e}")))?,
                )
            };
            state.skip_ws();
            state.expect_str("..")?;
            state.skip_ws();
            let max_str = state.take_while(|c| c.is_ascii_digit() || c == '*')?;
            let max = if max_str == "*" {
                None
            } else {
                Some(
                    max_str
                        .parse::<u32>()
                        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid max: {e}")))?,
                )
            };
            state.skip_ws();
            state.expect_char(']')?;
            decl.min_count = min;
            decl.max_count = max;
            state.skip_ws();
        }

        // Inline constraint key=value pairs
        while !state.peek_char(';') && !state.peek_char('}') && !state.is_eof() {
            let key = state.take_word()?;
            state.skip_ws();
            state.expect_char('=')?;
            state.skip_ws();

            match key.as_str() {
                "minInclusive" => {
                    let v = state.take_number()?;
                    decl.min_inclusive = Some(v);
                }
                "maxInclusive" => {
                    let v = state.take_number()?;
                    decl.max_inclusive = Some(v);
                }
                "minExclusive" => {
                    let v = state.take_number()?;
                    decl.min_exclusive = Some(v);
                }
                "maxExclusive" => {
                    let v = state.take_number()?;
                    decl.max_exclusive = Some(v);
                }
                "pattern" => {
                    let p = state.take_quoted_string()?;
                    decl.pattern = Some(p);
                }
                "minLength" => {
                    let v = state.take_uint()?;
                    decl.min_length = Some(v);
                }
                "maxLength" => {
                    let v = state.take_uint()?;
                    decl.max_length = Some(v);
                }
                "in" => {
                    state.expect_char('(')?;
                    state.skip_ws();
                    while !state.peek_char(')') && !state.is_eof() {
                        let val = if state.peek_char('"') {
                            state.take_quoted_string()?
                        } else {
                            let raw = state.take_prefixed_or_iri()?;
                            expand_iri(&raw, prefixes)?
                        };
                        decl.in_values.push(val);
                        state.skip_ws();
                        if state.peek_char(',') {
                            state.advance(1);
                            state.skip_ws();
                        }
                    }
                    state.expect_char(')')?;
                }
                "hasValue" => {
                    let v = if state.peek_char('"') {
                        state.take_quoted_string()?
                    } else {
                        let raw = state.take_prefixed_or_iri()?;
                        expand_iri(&raw, prefixes)?
                    };
                    decl.has_value = Some(v);
                }
                "uniqueLang" => {
                    let v = state.take_word()?;
                    decl.unique_lang = v == "true";
                }
                "nodeKind" => {
                    let v = state.take_word()?;
                    decl.node_kind = Some(v);
                }
                "datatype" => {
                    let raw = state.take_prefixed_or_iri()?;
                    decl.datatype = Some(expand_iri(&raw, prefixes)?);
                }
                "class" => {
                    let raw = state.take_prefixed_or_iri()?;
                    decl.class = Some(expand_iri(&raw, prefixes)?);
                }
                other => {
                    // Skip unknown constraint
                    let _val = state.take_word().unwrap_or_default();
                    tracing::warn!("Unknown constraint key: {other}");
                }
            }
            state.skip_ws();
        }

        // Consume trailing semicolon if present
        if state.peek_char(';') {
            state.advance(1);
        }

        Ok(decl)
    }
}

// ---------------------------------------------------------------------------
// IRI expansion
// ---------------------------------------------------------------------------

fn expand_iri(raw: &str, prefixes: &HashMap<String, String>) -> Result<String, ShaclError> {
    // Already a full IRI
    if raw.starts_with('<') && raw.ends_with('>') {
        return Ok(raw[1..raw.len() - 1].to_string());
    }
    if raw.starts_with("http://") || raw.starts_with("https://") || raw.starts_with("urn:") {
        return Ok(raw.to_string());
    }
    // Prefixed name
    if let Some(colon_pos) = raw.find(':') {
        let prefix = &raw[..colon_pos];
        let local = &raw[colon_pos + 1..];
        if let Some(ns) = prefixes.get(prefix) {
            return Ok(format!("{ns}{local}"));
        }
        return Err(ShaclError::ShapeParsing(format!(
            "Unknown prefix: '{prefix}'"
        )));
    }
    Err(ShaclError::ShapeParsing(format!(
        "Cannot expand IRI: '{raw}'"
    )))
}

// ---------------------------------------------------------------------------
// Lightweight parse state
// ---------------------------------------------------------------------------

struct ParseState<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> ParseState<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn remaining(&self) -> &str {
        &self.input[self.pos..]
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn advance(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.input.len());
    }

    fn peek_char(&self, c: char) -> bool {
        self.remaining().starts_with(c)
    }

    fn peek_str(&self, s: &str) -> bool {
        self.remaining().starts_with(s)
    }

    fn peek_keyword(&self, kw: &str) -> bool {
        let rem = self.remaining();
        if rem.len() < kw.len() {
            return false;
        }
        let candidate = &rem[..kw.len()];
        if candidate.eq_ignore_ascii_case(kw) {
            // Must be followed by non-alphanumeric
            if rem.len() == kw.len() {
                return true;
            }
            let next = rem.as_bytes()[kw.len()];
            !next.is_ascii_alphanumeric() && next != b'_'
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.input.len() {
            let b = self.input.as_bytes()[self.pos];
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            self.skip_ws();
            if self.peek_char('#') {
                // skip to end of line
                while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn expect_char(&mut self, c: char) -> Result<(), ShaclError> {
        if self.peek_char(c) {
            self.advance(c.len_utf8());
            Ok(())
        } else {
            let got: String = self.remaining().chars().take(1).collect();
            Err(ShaclError::ShapeParsing(format!(
                "Expected '{c}' at position {}, got '{got}'",
                self.pos
            )))
        }
    }

    fn expect_str(&mut self, s: &str) -> Result<(), ShaclError> {
        if self.peek_str(s) {
            self.advance(s.len());
            Ok(())
        } else {
            Err(ShaclError::ShapeParsing(format!(
                "Expected '{s}' at position {}",
                self.pos
            )))
        }
    }

    fn expect_keyword_ci(&mut self, kw: &str) -> Result<(), ShaclError> {
        let rem = self.remaining();
        if rem.len() >= kw.len() && rem[..kw.len()].eq_ignore_ascii_case(kw) {
            self.advance(kw.len());
            Ok(())
        } else {
            Err(ShaclError::ShapeParsing(format!(
                "Expected keyword '{kw}' at position {}",
                self.pos
            )))
        }
    }

    fn take_until(&mut self, c: char) -> Result<String, ShaclError> {
        let start = self.pos;
        while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != c as u8 {
            self.pos += 1;
        }
        if self.pos == start {
            return Ok(String::new());
        }
        Ok(self.input[start..self.pos].trim().to_string())
    }

    fn take_iri(&mut self) -> Result<String, ShaclError> {
        self.skip_ws();
        if self.peek_char('<') {
            self.advance(1);
            let start = self.pos;
            while self.pos < self.input.len() && self.input.as_bytes()[self.pos] != b'>' {
                self.pos += 1;
            }
            let iri = self.input[start..self.pos].to_string();
            if self.peek_char('>') {
                self.advance(1);
            }
            Ok(iri)
        } else {
            Err(ShaclError::ShapeParsing(format!(
                "Expected '<' for IRI at position {}",
                self.pos
            )))
        }
    }

    fn take_word(&mut self) -> Result<String, ShaclError> {
        let start = self.pos;
        while self.pos < self.input.len() {
            let b = self.input.as_bytes()[self.pos];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(ShaclError::ShapeParsing(format!(
                "Expected word at position {}",
                self.pos
            )));
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn take_prefixed_or_iri(&mut self) -> Result<String, ShaclError> {
        self.skip_ws();
        if self.peek_char('<') {
            let iri = self.take_iri()?;
            return Ok(format!("<{iri}>"));
        }
        // prefixed name: prefix:local
        let start = self.pos;
        while self.pos < self.input.len() {
            let b = self.input.as_bytes()[self.pos];
            if b.is_ascii_alphanumeric()
                || b == b'_'
                || b == b'-'
                || b == b':'
                || b == b'.'
                || b == b'/'
                || b == b'#'
            {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(ShaclError::ShapeParsing(format!(
                "Expected IRI or prefixed name at position {}",
                self.pos
            )));
        }
        let raw = self.input[start..self.pos].to_string();
        // Trim trailing dots that might be sentence-ending
        let trimmed = raw.trim_end_matches('.');
        if trimmed.len() < raw.len() {
            self.pos -= raw.len() - trimmed.len();
        }
        Ok(trimmed.to_string())
    }

    fn take_while(&mut self, pred: impl Fn(char) -> bool) -> Result<String, ShaclError> {
        let start = self.pos;
        while self.pos < self.input.len() {
            if let Some(c) = self.remaining().chars().next() {
                if pred(c) {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn take_number(&mut self) -> Result<f64, ShaclError> {
        let s = self.take_while(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == '+')?;
        s.parse::<f64>()
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid number '{s}': {e}")))
    }

    fn take_uint(&mut self) -> Result<u32, ShaclError> {
        let s = self.take_while(|c| c.is_ascii_digit())?;
        s.parse::<u32>()
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid integer '{s}': {e}")))
    }

    fn take_quoted_string(&mut self) -> Result<String, ShaclError> {
        self.expect_char('"')?;
        let start = self.pos;
        let mut escaped = false;
        while self.pos < self.input.len() {
            let b = self.input.as_bytes()[self.pos];
            if escaped {
                escaped = false;
                self.pos += 1;
            } else if b == b'\\' {
                escaped = true;
                self.pos += 1;
            } else if b == b'"' {
                break;
            } else {
                self.pos += 1;
            }
        }
        let content = self.input[start..self.pos].to_string();
        self.expect_char('"')?;
        Ok(content)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn parse_ok(input: &str) -> ShaclcDocument {
        ShaclcParser::parse(input).expect("should parse")
    }

    // -- basic document structure --

    #[test]
    fn test_empty_document() {
        let doc = parse_ok("");
        assert!(doc.shapes.is_empty());
        assert!(doc.imports.is_empty());
    }

    #[test]
    fn test_comment_only_document() {
        let doc = parse_ok("# just a comment\n# another one\n");
        assert!(doc.shapes.is_empty());
    }

    #[test]
    fn test_prefix_declaration() {
        let doc = parse_ok("PREFIX ex: <http://example.org/>\n");
        assert_eq!(
            doc.prefixes.get("ex"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_multiple_prefixes() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\n",
        );
        assert!(doc.prefixes.contains_key("ex"));
        assert!(doc.prefixes.contains_key("foaf"));
    }

    #[test]
    fn test_imports() {
        let doc = parse_ok("IMPORTS <http://example.org/shapes>\n");
        assert_eq!(doc.imports, vec!["http://example.org/shapes"]);
    }

    #[test]
    fn test_default_prefixes_present() {
        let doc = parse_ok("");
        assert!(doc.prefixes.contains_key("xsd"));
        assert!(doc.prefixes.contains_key("rdf"));
        assert!(doc.prefixes.contains_key("rdfs"));
        assert!(doc.prefixes.contains_key("sh"));
    }

    // -- shape parsing --

    #[test]
    fn test_empty_shape() {
        let doc = parse_ok("PREFIX ex: <http://example.org/>\nshape ex:MyShape {\n}\n");
        assert_eq!(doc.shapes.len(), 1);
        assert_eq!(doc.shapes[0].iri, "http://example.org/MyShape");
        assert!(doc.shapes[0].properties.is_empty());
    }

    #[test]
    fn test_shape_with_target_class() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:PersonShape -> targetClass(ex:Person) {\n}\n",
        );
        assert_eq!(doc.shapes[0].targets.len(), 1);
        assert_eq!(
            doc.shapes[0].targets[0],
            ShaclcTarget::TargetClass(vec!["http://example.org/Person".to_string()])
        );
    }

    #[test]
    fn test_shape_with_target_node() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S -> targetNode(ex:Alice, ex:Bob) {\n}\n",
        );
        match &doc.shapes[0].targets[0] {
            ShaclcTarget::TargetNode(iris) => {
                assert_eq!(iris.len(), 2);
                assert!(iris[0].ends_with("Alice"));
                assert!(iris[1].ends_with("Bob"));
            }
            other => panic!("Expected TargetNode, got {other:?}"),
        }
    }

    #[test]
    fn test_shape_with_target_subjects_of() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S -> targetSubjectsOf(ex:name) {\n}\n",
        );
        assert_eq!(
            doc.shapes[0].targets[0],
            ShaclcTarget::TargetSubjectsOf("http://example.org/name".to_string())
        );
    }

    #[test]
    fn test_shape_with_target_objects_of() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S -> targetObjectsOf(ex:knows) {\n}\n",
        );
        assert_eq!(
            doc.shapes[0].targets[0],
            ShaclcTarget::TargetObjectsOf("http://example.org/knows".to_string())
        );
    }

    // -- property declarations --

    #[test]
    fn test_property_with_datatype() {
        let doc =
            parse_ok("PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:name xsd:string ;\n}\n");
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.path, "http://example.org/name");
        assert_eq!(
            prop.datatype.as_deref(),
            Some("http://www.w3.org/2001/XMLSchema#string")
        );
    }

    #[test]
    fn test_property_with_cardinality() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:name xsd:string [1..1] ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.min_count, Some(1));
        assert_eq!(prop.max_count, Some(1));
    }

    #[test]
    fn test_property_unbounded_max() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:tag xsd:string [0..*] ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.min_count, Some(0));
        assert_eq!(prop.max_count, None);
    }

    #[test]
    fn test_property_min_inclusive() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:age xsd:integer [0..1] minInclusive=0 ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.min_inclusive, Some(0.0));
    }

    #[test]
    fn test_property_max_inclusive() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:score xsd:decimal maxInclusive=100 ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.max_inclusive, Some(100.0));
    }

    #[test]
    fn test_property_pattern() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:email xsd:string pattern=\"^[^@]+@[^@]+$\" ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.pattern.as_deref(), Some("^[^@]+@[^@]+$"));
    }

    #[test]
    fn test_property_min_max_length() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:code xsd:string minLength=3 maxLength=10 ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.min_length, Some(3));
        assert_eq!(prop.max_length, Some(10));
    }

    #[test]
    fn test_property_in_values() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:status xsd:string in=(\"active\", \"inactive\") ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.in_values, vec!["active", "inactive"]);
    }

    #[test]
    fn test_property_has_value() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:type hasValue=\"Person\" ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.has_value.as_deref(), Some("Person"));
    }

    #[test]
    fn test_property_unique_lang() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:label xsd:string uniqueLang=true ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert!(prop.unique_lang);
    }

    #[test]
    fn test_property_with_class() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:knows ex:Person [0..*] ;\n}\n",
        );
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.class.as_deref(), Some("http://example.org/Person"));
    }

    #[test]
    fn test_property_with_node_kind() {
        let doc =
            parse_ok("PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:ref IRI [1..1] ;\n}\n");
        let prop = &doc.shapes[0].properties[0];
        assert_eq!(prop.node_kind.as_deref(), Some("IRI"));
    }

    // -- multiple shapes --

    #[test]
    fn test_multiple_shapes() {
        let input = r#"
            PREFIX ex: <http://example.org/>
            shape ex:PersonShape -> targetClass(ex:Person) {
                ex:name xsd:string [1..1] ;
            }
            shape ex:CompanyShape -> targetClass(ex:Company) {
                ex:companyName xsd:string [1..1] ;
            }
        "#;
        let doc = parse_ok(input);
        assert_eq!(doc.shapes.len(), 2);
    }

    #[test]
    fn test_multiple_properties_in_shape() {
        let input = r#"
            PREFIX ex: <http://example.org/>
            shape ex:PersonShape {
                ex:name xsd:string [1..1] ;
                ex:age xsd:integer [0..1] ;
                ex:email xsd:string [0..*] ;
            }
        "#;
        let doc = parse_ok(input);
        assert_eq!(doc.shapes[0].properties.len(), 3);
    }

    // -- error cases --

    #[test]
    fn test_error_unknown_prefix() {
        let result = ShaclcParser::parse("shape unknown:Shape {\n}\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_brace() {
        let result = ShaclcParser::parse(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:name xsd:string ;\n",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unknown_target() {
        let result = ShaclcParser::parse(
            "PREFIX ex: <http://example.org/>\nshape ex:S -> unknownTarget(ex:X) {\n}\n",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_cardinality() {
        let result = ShaclcParser::parse(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:name xsd:string [abc..1] ;\n}\n",
        );
        assert!(result.is_err());
    }

    // -- IRI expansion --

    #[test]
    fn test_expand_full_iri() {
        let prefixes = HashMap::new();
        let result = expand_iri("<http://example.org/test>", &prefixes);
        assert_eq!(result.ok(), Some("http://example.org/test".to_string()));
    }

    #[test]
    fn test_expand_prefixed_name() {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());
        let result = expand_iri("ex:test", &prefixes);
        assert_eq!(result.ok(), Some("http://example.org/test".to_string()));
    }

    #[test]
    fn test_expand_unknown_prefix_error() {
        let prefixes = HashMap::new();
        let result = expand_iri("unknown:test", &prefixes);
        assert!(result.is_err());
    }

    // -- complex scenario --

    #[test]
    fn test_full_document() {
        let input = r#"
            PREFIX ex: <http://example.org/>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            IMPORTS <http://example.org/base-shapes>

            # Person shape with validations
            shape ex:PersonShape -> targetClass(ex:Person) {
                ex:name xsd:string [1..1] minLength=1 maxLength=200 ;
                ex:age xsd:integer [0..1] minInclusive=0 maxInclusive=150 ;
                ex:email xsd:string [0..*] pattern="^[^@]+@[^@]+$" ;
                ex:status xsd:string [1..1] in=("active", "inactive", "pending") ;
            }
        "#;
        let doc = parse_ok(input);
        assert_eq!(doc.imports.len(), 1);
        assert_eq!(doc.shapes.len(), 1);
        let shape = &doc.shapes[0];
        assert_eq!(shape.properties.len(), 4);
        let email = &shape.properties[2];
        assert!(email.pattern.is_some());
        let status = &shape.properties[3];
        assert_eq!(status.in_values.len(), 3);
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let input = "PREFIX ex: <http://example.org/>\nSHAPE ex:S {\n}\n";
        let doc = parse_ok(input);
        assert_eq!(doc.shapes.len(), 1);
    }

    #[test]
    fn test_property_min_exclusive() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:val xsd:decimal minExclusive=0 ;\n}\n",
        );
        assert_eq!(doc.shapes[0].properties[0].min_exclusive, Some(0.0));
    }

    #[test]
    fn test_property_max_exclusive() {
        let doc = parse_ok(
            "PREFIX ex: <http://example.org/>\nshape ex:S {\n  ex:val xsd:decimal maxExclusive=100 ;\n}\n",
        );
        assert_eq!(doc.shapes[0].properties[0].max_exclusive, Some(100.0));
    }

    #[test]
    fn test_shape_deactivated() {
        let doc = parse_ok("PREFIX ex: <http://example.org/>\nshape ex:S deactivated {\n}\n");
        assert!(doc.shapes[0].deactivated);
    }

    #[test]
    fn test_property_default_decl() {
        let decl = ShaclcPropertyDecl::default();
        assert!(decl.path.is_empty());
        assert!(decl.datatype.is_none());
        assert!(decl.min_count.is_none());
        assert!(!decl.unique_lang);
    }
}
