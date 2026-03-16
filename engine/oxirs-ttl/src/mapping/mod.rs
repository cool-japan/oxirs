//! RML-inspired RDF Mapping Language support
//!
//! Maps non-RDF data sources (CSV, JSON, inline values) to RDF triples.
//! Inspired by the W3C RDF Mapping Language (RML) specification:
//! <https://rml.io/specs/rml/>
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::mapping::{MappingEngine, MappingRuleBuilder, ObjectSpec};
//!
//! let csv_data = "id,name,age\n1,Alice,30\n2,Bob,25";
//!
//! let rule = MappingRuleBuilder::new("persons")
//!     .csv_source(csv_data)
//!     .subject_template("http://example.org/person/{id}")
//!     .map(
//!         "http://xmlns.com/foaf/0.1/name",
//!         ObjectSpec::Column("name".to_string()),
//!     )
//!     .map(
//!         "http://xmlns.com/foaf/0.1/age",
//!         ObjectSpec::TypedColumn {
//!             column: "age".to_string(),
//!             datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
//!         },
//!     )
//!     .build();
//!
//! let engine = MappingEngine::new();
//! let triples = engine.execute(&rule).expect("should succeed");
//! assert_eq!(triples.len(), 4); // 2 rows × 2 predicates
//! ```

use std::collections::HashMap;
use std::fmt;

use oxirs_core::model::{Literal, NamedNode, Object, Predicate, Subject, Triple};
use thiserror::Error;

// ─── Error Types ─────────────────────────────────────────────────────────────

/// Errors that can occur during RML mapping operations
#[derive(Debug, Error)]
pub enum MappingError {
    /// A required column was not found in the data row
    #[error("Missing column '{column}' in row {row_index}")]
    MissingColumn {
        /// Column name that was not found
        column: String,
        /// Zero-based index of the row where the error occurred
        row_index: usize,
    },

    /// A template contained an unresolvable reference
    #[error("Template '{template}' references unknown column '{column}' in row {row_index}")]
    UnresolvableTemplate {
        /// The template pattern string
        template: String,
        /// Column name referenced in the template
        column: String,
        /// Zero-based index of the row
        row_index: usize,
    },

    /// An IRI generated from a template was syntactically invalid
    #[error("Invalid IRI generated from template '{template}': '{iri}'")]
    InvalidIri {
        /// The template pattern
        template: String,
        /// The invalid IRI that was generated
        iri: String,
    },

    /// A predicate IRI was syntactically invalid
    #[error("Invalid predicate IRI: '{iri}'")]
    InvalidPredicateIri {
        /// The invalid IRI
        iri: String,
    },

    /// An object IRI was syntactically invalid
    #[error("Invalid object IRI: '{iri}'")]
    InvalidObjectIri {
        /// The invalid IRI
        iri: String,
    },

    /// JSON parsing failed
    #[error("JSON parse error: {message}")]
    JsonParseError {
        /// Human-readable description of the parse failure
        message: String,
    },

    /// CSV parsing failed
    #[error("CSV parse error at line {line}: {message}")]
    CsvParseError {
        /// Line number where the error occurred (1-based)
        line: usize,
        /// Human-readable description of the parse failure
        message: String,
    },

    /// A JSON path expression was invalid or matched no data
    #[error("JSON path '{path}' did not match any array in the document")]
    JsonPathNoMatch {
        /// The JSON path that failed to match
        path: String,
    },

    /// No rows could be extracted from the data source
    #[error("Data source produced no rows")]
    EmptyDataSource,

    /// Core RDF model error
    #[error("RDF model error: {0}")]
    RdfModelError(String),
}

/// Convenience type alias for mapping results
pub type MappingResult<T> = Result<T, MappingError>;

// ─── Core Data Types ──────────────────────────────────────────────────────────

/// A data row represented as a map from column name to string value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    /// Map from column (field) name to value
    pub values: HashMap<String, String>,
}

impl Row {
    /// Create a new empty row
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Create a row from an iterator of (key, value) pairs
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            values: pairs.into_iter().collect(),
        }
    }

    /// Get a value by column name
    pub fn get(&self, column: &str) -> Option<&str> {
        self.values.get(column).map(String::as_str)
    }

    /// Check whether a column exists (even if empty)
    pub fn contains(&self, column: &str) -> bool {
        self.values.contains_key(column)
    }

    /// Return an iterator over all (column, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<_> = self.values.iter().collect();
        entries.sort_by_key(|(k, _)| k.as_str());
        write!(f, "{{")?;
        for (i, (k, v)) in entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k}: {v}")?;
        }
        write!(f, "}}")
    }
}

// ─── Template ────────────────────────────────────────────────────────────────

/// A URI template that can be rendered using row values.
///
/// Template syntax: literal text with `{column_name}` placeholders.
/// Column references are URL-percent-encoded to produce valid IRIs.
///
/// # Example
///
/// ```
/// use oxirs_ttl::mapping::{Template, Row};
/// use std::collections::HashMap;
///
/// let tpl = Template::new("http://example.org/{id}/profile");
/// let mut row = Row::new();
/// row.values.insert("id".to_string(), "42".to_string());
/// let iri = tpl.render(&row, 0).expect("should succeed");
/// assert_eq!(iri, "http://example.org/42/profile");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    /// The raw pattern string, e.g. `"http://example.org/{column_name}"`
    pub pattern: String,
}

impl Template {
    /// Create a new template from any string-like value
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
        }
    }

    /// Render the template by substituting `{column}` placeholders with
    /// percent-encoded values from `row`.
    ///
    /// Returns an error if a referenced column is absent from the row.
    pub fn render(&self, row: &Row, row_index: usize) -> MappingResult<String> {
        let mut output = String::with_capacity(self.pattern.len() + 32);
        let mut chars = self.pattern.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                // Collect column name until '}'
                let mut col_name = String::new();
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == '}' {
                        closed = true;
                        break;
                    }
                    col_name.push(inner);
                }
                if !closed {
                    // Treat un-closed brace as literal text
                    output.push('{');
                    output.push_str(&col_name);
                    continue;
                }
                let value =
                    row.get(&col_name)
                        .ok_or_else(|| MappingError::UnresolvableTemplate {
                            template: self.pattern.clone(),
                            column: col_name.clone(),
                            row_index,
                        })?;
                percent_encode_path(value, &mut output);
            } else {
                output.push(ch);
            }
        }
        Ok(output)
    }
}

impl fmt::Display for Template {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pattern)
    }
}

/// Percent-encode characters that are illegal in IRI paths.
/// RFC 3986 unreserved chars and common path chars are kept as-is.
fn percent_encode_path(input: &str, out: &mut String) {
    for byte in input.bytes() {
        match byte {
            // RFC 3986 unreserved characters
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            // Additional IRI-safe characters
            b':' | b'@' | b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+' | b',' | b';'
            | b'=' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push(hex_nibble(byte >> 4));
                out.push(hex_nibble(byte & 0x0F));
            }
        }
    }
}

#[inline]
fn hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + n - 10) as char,
        _ => '0',
    }
}

// ─── ObjectSpec ──────────────────────────────────────────────────────────────

/// Specifies how the object of a triple should be produced from a data row
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectSpec {
    /// Generate an IRI by rendering a template against the row
    Template(Template),

    /// Use the column value as a plain string literal (`xsd:string`)
    Column(String),

    /// Use a constant string — always the same value regardless of the row
    Constant(String),

    /// Use the column value as a typed literal
    TypedColumn {
        /// Column name containing the lexical value
        column: String,
        /// Full XSD datatype IRI (e.g. `"http://www.w3.org/2001/XMLSchema#integer"`)
        datatype: String,
    },

    /// Use the column value as a language-tagged literal; the language tag is
    /// taken from another column.
    LangColumn {
        /// Column name containing the literal text
        column: String,
        /// Column name containing the BCP 47 language tag (e.g. `"en"`)
        lang_column: String,
    },

    /// Use the column value as a language-tagged literal with a fixed language
    LangFixed {
        /// Column name containing the literal text
        column: String,
        /// BCP 47 language tag (e.g. `"en"`)
        lang: String,
    },

    /// Use a constant IRI value (no template substitution)
    ConstantIri(String),
}

// ─── PredicateObjectMap ───────────────────────────────────────────────────────

/// Pairs a predicate IRI with an [`ObjectSpec`] to produce a single
/// predicate-object component of a triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateObjectMap {
    /// Full IRI for the predicate
    pub predicate: String,
    /// Specification for how to produce the object
    pub object_template: ObjectSpec,
}

impl PredicateObjectMap {
    /// Create a new predicate-object map
    pub fn new(predicate: impl Into<String>, object_template: ObjectSpec) -> Self {
        Self {
            predicate: predicate.into(),
            object_template,
        }
    }
}

// ─── DataSource ───────────────────────────────────────────────────────────────

/// The origin of the data to be mapped to RDF
#[derive(Debug, Clone)]
pub enum DataSource {
    /// CSV text with configurable delimiter
    Csv {
        /// The raw CSV content
        content: String,
        /// Field delimiter character (usually `','`)
        delimiter: char,
    },

    /// JSON text, optionally with a dot-separated path to the target array
    Json {
        /// The raw JSON content
        content: String,
        /// Optional dot-separated path to the array of objects (e.g. `"results.bindings"`)
        json_path: Option<String>,
    },

    /// Pre-parsed rows supplied directly as vectors
    InlineValues {
        /// Rows of string values
        rows: Vec<Vec<String>>,
        /// Column headers that correspond positionally to the values in each row
        headers: Vec<String>,
    },
}

// ─── MappingRule ──────────────────────────────────────────────────────────────

/// A complete mapping rule that produces RDF triples from a [`DataSource`]
#[derive(Debug, Clone)]
pub struct MappingRule {
    /// Human-readable name for this rule (used in error messages)
    pub name: String,
    /// The data source to read rows from
    pub source: DataSource,
    /// Template for generating the subject IRI of each triple
    pub subject_template: Template,
    /// List of predicate-object pairs to generate for each row
    pub predicate_object_maps: Vec<PredicateObjectMap>,
    /// Optional named graph to assign all generated triples to
    pub graph_name: Option<String>,
}

impl MappingRule {
    /// Create a minimal mapping rule (use [`MappingRuleBuilder`] for ergonomic construction)
    pub fn new(name: impl Into<String>, source: DataSource, subject_template: Template) -> Self {
        Self {
            name: name.into(),
            source,
            subject_template,
            predicate_object_maps: Vec::new(),
            graph_name: None,
        }
    }

    /// Add a predicate-object map to this rule
    pub fn add_predicate_object_map(&mut self, pom: PredicateObjectMap) {
        self.predicate_object_maps.push(pom);
    }
}

// ─── MappingEngine ────────────────────────────────────────────────────────────

/// Engine that executes [`MappingRule`]s and produces RDF [`Triple`]s
///
/// The engine is stateless and cheap to create.  All configuration is
/// carried by the rules themselves.
#[derive(Debug, Default, Clone)]
pub struct MappingEngine {
    /// Whether to skip rows that produce errors instead of failing fast
    pub skip_errors: bool,
}

impl MappingEngine {
    /// Create a new mapping engine with default settings (fail-fast)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an engine that silently skips rows that produce mapping errors
    pub fn new_lenient() -> Self {
        Self { skip_errors: true }
    }

    /// Execute a single mapping rule and return all produced triples
    pub fn execute(&self, rule: &MappingRule) -> MappingResult<Vec<Triple>> {
        let (headers, rows) = self.extract_rows(&rule.source)?;
        let _ = headers; // headers are embedded inside each Row already
        self.map_rows(rule, &rows)
    }

    /// Execute multiple rules and concatenate all produced triples
    pub fn execute_all(&self, rules: &[MappingRule]) -> MappingResult<Vec<Triple>> {
        let mut all_triples = Vec::new();
        for rule in rules {
            let mut triples = self.execute(rule)?;
            all_triples.append(&mut triples);
        }
        Ok(all_triples)
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    fn extract_rows(&self, source: &DataSource) -> MappingResult<(Vec<String>, Vec<Row>)> {
        match source {
            DataSource::Csv { content, delimiter } => Self::parse_csv(content, *delimiter),
            DataSource::Json { content, json_path } => {
                let rows = Self::parse_json(content, json_path.as_deref())?;
                // headers are implicit in the Row keys; return empty list
                Ok((Vec::new(), rows))
            }
            DataSource::InlineValues { rows, headers } => {
                let parsed_rows: Vec<Row> = rows
                    .iter()
                    .map(|row_values| {
                        let pairs = headers
                            .iter()
                            .zip(row_values.iter())
                            .map(|(h, v)| (h.clone(), v.clone()));
                        Row::from_pairs(pairs)
                    })
                    .collect();
                Ok((headers.clone(), parsed_rows))
            }
        }
    }

    fn map_rows(&self, rule: &MappingRule, rows: &[Row]) -> MappingResult<Vec<Triple>> {
        let mut triples = Vec::with_capacity(rows.len() * rule.predicate_object_maps.len());

        for (row_idx, row) in rows.iter().enumerate() {
            // Generate subject IRI
            let subject_iri = match rule.subject_template.render(row, row_idx) {
                Ok(iri) => iri,
                Err(e) => {
                    if self.skip_errors {
                        continue;
                    }
                    return Err(e);
                }
            };

            let subject_node =
                NamedNode::new(&subject_iri).map_err(|e| MappingError::InvalidIri {
                    template: rule.subject_template.pattern.clone(),
                    iri: format!("{subject_iri} ({e})"),
                })?;
            let subject: Subject = subject_node.into();

            // Generate one triple per predicate-object map
            for pom in &rule.predicate_object_maps {
                let result = self.build_triple(&subject, pom, row, row_idx);
                match result {
                    Ok(triple) => triples.push(triple),
                    Err(e) => {
                        if self.skip_errors {
                            continue;
                        }
                        return Err(e);
                    }
                }
            }
        }
        Ok(triples)
    }

    fn build_triple(
        &self,
        subject: &Subject,
        pom: &PredicateObjectMap,
        row: &Row,
        row_idx: usize,
    ) -> MappingResult<Triple> {
        // Build predicate
        let pred_node =
            NamedNode::new(&pom.predicate).map_err(|_| MappingError::InvalidPredicateIri {
                iri: pom.predicate.clone(),
            })?;
        let predicate: Predicate = pred_node.into();

        // Build object
        let object = self.resolve_object(&pom.object_template, row, row_idx)?;

        Ok(Triple::new(subject.clone(), predicate, object))
    }

    fn resolve_object(
        &self,
        spec: &ObjectSpec,
        row: &Row,
        row_idx: usize,
    ) -> MappingResult<Object> {
        match spec {
            ObjectSpec::Template(tpl) => {
                let iri = tpl.render(row, row_idx)?;
                let node = NamedNode::new(&iri)
                    .map_err(|_| MappingError::InvalidObjectIri { iri: iri.clone() })?;
                Ok(Object::NamedNode(node))
            }

            ObjectSpec::Column(col) => {
                let value = row.get(col).ok_or_else(|| MappingError::MissingColumn {
                    column: col.clone(),
                    row_index: row_idx,
                })?;
                Ok(Object::Literal(Literal::new(value)))
            }

            ObjectSpec::Constant(value) => Ok(Object::Literal(Literal::new(value))),

            ObjectSpec::TypedColumn { column, datatype } => {
                let value = row.get(column).ok_or_else(|| MappingError::MissingColumn {
                    column: column.clone(),
                    row_index: row_idx,
                })?;
                let dt_node =
                    NamedNode::new(datatype).map_err(|_| MappingError::InvalidObjectIri {
                        iri: datatype.clone(),
                    })?;
                Ok(Object::Literal(Literal::new_typed_literal(value, dt_node)))
            }

            ObjectSpec::LangColumn {
                column,
                lang_column,
            } => {
                let value = row.get(column).ok_or_else(|| MappingError::MissingColumn {
                    column: column.clone(),
                    row_index: row_idx,
                })?;
                let lang = row
                    .get(lang_column)
                    .ok_or_else(|| MappingError::MissingColumn {
                        column: lang_column.clone(),
                        row_index: row_idx,
                    })?;
                let lit = Literal::new_language_tagged_literal(value, lang)
                    .map_err(|e| MappingError::RdfModelError(e.to_string()))?;
                Ok(Object::Literal(lit))
            }

            ObjectSpec::LangFixed { column, lang } => {
                let value = row.get(column).ok_or_else(|| MappingError::MissingColumn {
                    column: column.clone(),
                    row_index: row_idx,
                })?;
                let lit = Literal::new_language_tagged_literal(value, lang)
                    .map_err(|e| MappingError::RdfModelError(e.to_string()))?;
                Ok(Object::Literal(lit))
            }

            ObjectSpec::ConstantIri(iri) => {
                let node = NamedNode::new(iri)
                    .map_err(|_| MappingError::InvalidObjectIri { iri: iri.clone() })?;
                Ok(Object::NamedNode(node))
            }
        }
    }

    // ─── CSV parser ──────────────────────────────────────────────────────

    /// Parse CSV content into (headers, rows).
    ///
    /// Handles:
    /// - Configurable delimiter
    /// - Double-quote escaping (`""` inside a quoted field)
    /// - CRLF and LF line endings
    /// - Quoted fields that span multiple lines
    pub fn parse_csv(content: &str, delimiter: char) -> MappingResult<(Vec<String>, Vec<Row>)> {
        let lines = split_csv_lines(content);
        if lines.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Parse header row
        let headers = parse_csv_line(&lines[0], delimiter);
        if headers.is_empty() {
            return Err(MappingError::CsvParseError {
                line: 1,
                message: "empty header row".to_string(),
            });
        }

        let mut rows = Vec::with_capacity(lines.len().saturating_sub(1));
        for (line_idx, line) in lines.iter().enumerate().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let values = parse_csv_line(line, delimiter);
            if values.len() != headers.len() {
                return Err(MappingError::CsvParseError {
                    line: line_idx + 1,
                    message: format!("expected {} fields but got {}", headers.len(), values.len()),
                });
            }
            let row = Row::from_pairs(headers.iter().cloned().zip(values.into_iter()));
            rows.push(row);
        }
        Ok((headers, rows))
    }

    // ─── JSON parser ─────────────────────────────────────────────────────

    /// Parse JSON content into rows.
    ///
    /// Behaviour:
    /// - If `json_path` is `None`, the root must be a JSON array of objects.
    /// - If `json_path` is `Some("a.b.c")`, the engine traverses object keys
    ///   `a` → `b` → `c` and expects to find an array there.
    /// - Each array element must be a JSON object; its key-value pairs become
    ///   the row fields (values are coerced to strings).
    pub fn parse_json(content: &str, json_path: Option<&str>) -> MappingResult<Vec<Row>> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| MappingError::JsonParseError {
                message: e.to_string(),
            })?;

        // Navigate to the target array using dot-separated path
        let array = if let Some(path) = json_path {
            navigate_json_path(&value, path)?
        } else {
            &value
        };

        let arr = array.as_array().ok_or_else(|| {
            let path_desc = json_path.unwrap_or("<root>");
            MappingError::JsonPathNoMatch {
                path: path_desc.to_string(),
            }
        })?;

        let mut rows = Vec::with_capacity(arr.len());
        for element in arr {
            let obj = element
                .as_object()
                .ok_or_else(|| MappingError::JsonParseError {
                    message: "JSON array element is not an object".to_string(),
                })?;
            let row = Row::from_pairs(
                obj.iter()
                    .map(|(k, v)| (k.clone(), json_value_to_string(v))),
            );
            rows.push(row);
        }
        Ok(rows)
    }
}

// ─── JSON helpers ─────────────────────────────────────────────────────────────

fn navigate_json_path<'a>(
    value: &'a serde_json::Value,
    path: &str,
) -> MappingResult<&'a serde_json::Value> {
    let mut current = value;
    for key in path.split('.') {
        current = current
            .get(key)
            .ok_or_else(|| MappingError::JsonPathNoMatch {
                path: path.to_string(),
            })?;
    }
    Ok(current)
}

fn json_value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

// ─── CSV helpers ──────────────────────────────────────────────────────────────

/// Split CSV text into logical lines, handling quoted fields that contain newlines.
fn split_csv_lines(content: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            '\r' => {
                // Handle CRLF
                if chars.peek() == Some(&'\n') {
                    let _ = chars.next();
                }
                if !in_quotes {
                    lines.push(std::mem::take(&mut current));
                } else {
                    current.push('\n');
                }
            }
            '\n' => {
                if !in_quotes {
                    lines.push(std::mem::take(&mut current));
                } else {
                    current.push(ch);
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Parse a single CSV line into a vector of field values.
fn parse_csv_line(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    // Escaped double-quote inside quoted field
                    current.push('"');
                    let _ = chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == delimiter {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

// ─── Builder ──────────────────────────────────────────────────────────────────

/// Fluent builder for constructing [`MappingRule`] instances
///
/// # Example
///
/// ```rust
/// use oxirs_ttl::mapping::{MappingRuleBuilder, ObjectSpec};
///
/// let rule = MappingRuleBuilder::new("employees")
///     .csv_source("id,name\n1,Alice\n2,Bob")
///     .subject_template("http://example.org/employee/{id}")
///     .map("http://xmlns.com/foaf/0.1/name", ObjectSpec::Column("name".to_string()))
///     .build();
/// ```
#[derive(Debug)]
pub struct MappingRuleBuilder {
    rule: MappingRule,
}

impl MappingRuleBuilder {
    /// Start building a new rule with the given name
    pub fn new(name: impl Into<String>) -> Self {
        let name_str = name.into();
        Self {
            rule: MappingRule {
                name: name_str,
                source: DataSource::Csv {
                    content: String::new(),
                    delimiter: ',',
                },
                subject_template: Template::new(""),
                predicate_object_maps: Vec::new(),
                graph_name: None,
            },
        }
    }

    /// Use a CSV string as the data source (comma delimiter)
    pub fn csv_source(mut self, content: impl Into<String>) -> Self {
        self.rule.source = DataSource::Csv {
            content: content.into(),
            delimiter: ',',
        };
        self
    }

    /// Use a CSV string with a custom delimiter
    pub fn csv_source_with_delimiter(
        mut self,
        content: impl Into<String>,
        delimiter: char,
    ) -> Self {
        self.rule.source = DataSource::Csv {
            content: content.into(),
            delimiter,
        };
        self
    }

    /// Use a JSON string as the data source (root must be an array)
    pub fn json_source(mut self, content: impl Into<String>) -> Self {
        self.rule.source = DataSource::Json {
            content: content.into(),
            json_path: None,
        };
        self
    }

    /// Use a JSON string with a dot-separated path to the target array
    pub fn json_source_with_path(
        mut self,
        content: impl Into<String>,
        json_path: impl Into<String>,
    ) -> Self {
        self.rule.source = DataSource::Json {
            content: content.into(),
            json_path: Some(json_path.into()),
        };
        self
    }

    /// Use pre-parsed inline values
    pub fn inline_source(mut self, headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        self.rule.source = DataSource::InlineValues { rows, headers };
        self
    }

    /// Set the subject IRI template
    pub fn subject_template(mut self, template: impl Into<String>) -> Self {
        self.rule.subject_template = Template::new(template);
        self
    }

    /// Add a predicate-object mapping
    pub fn map(mut self, predicate: impl Into<String>, object: ObjectSpec) -> Self {
        self.rule.predicate_object_maps.push(PredicateObjectMap {
            predicate: predicate.into(),
            object_template: object,
        });
        self
    }

    /// Assign all produced triples to a named graph
    pub fn graph(mut self, graph_name: impl Into<String>) -> Self {
        self.rule.graph_name = Some(graph_name.into());
        self
    }

    /// Consume the builder and return the finished [`MappingRule`]
    pub fn build(self) -> MappingRule {
        self.rule
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn engine() -> MappingEngine {
        MappingEngine::new()
    }

    fn lenient_engine() -> MappingEngine {
        MappingEngine::new_lenient()
    }

    fn xsd(local: &str) -> String {
        format!("http://www.w3.org/2001/XMLSchema#{local}")
    }

    fn ex(local: &str) -> String {
        format!("http://example.org/{local}")
    }

    fn foaf(local: &str) -> String {
        format!("http://xmlns.com/foaf/0.1/{local}")
    }

    // ── Template tests ───────────────────────────────────────────────────

    #[test]
    fn test_template_simple_substitution() {
        let tpl = Template::new("http://example.org/{id}");
        let mut row = Row::new();
        row.values.insert("id".to_string(), "42".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/42");
    }

    #[test]
    fn test_template_multiple_placeholders() {
        let tpl = Template::new("http://example.org/{type}/{id}");
        let mut row = Row::new();
        row.values.insert("type".to_string(), "person".to_string());
        row.values.insert("id".to_string(), "7".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/person/7");
    }

    #[test]
    fn test_template_percent_encoding() {
        let tpl = Template::new("http://example.org/{name}");
        let mut row = Row::new();
        row.values
            .insert("name".to_string(), "hello world".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/hello%20world");
    }

    #[test]
    fn test_template_missing_column_error() {
        let tpl = Template::new("http://example.org/{missing}");
        let row = Row::new();
        let err = tpl.render(&row, 3).unwrap_err();
        match err {
            MappingError::UnresolvableTemplate {
                column, row_index, ..
            } => {
                assert_eq!(column, "missing");
                assert_eq!(row_index, 3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_template_no_placeholders() {
        let tpl = Template::new("http://example.org/constant");
        let row = Row::new();
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/constant");
    }

    #[test]
    fn test_template_slash_encoded() {
        let tpl = Template::new("http://example.org/{path}");
        let mut row = Row::new();
        row.values.insert("path".to_string(), "a/b/c".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        // '/' is not in RFC 3986 unreserved, should be encoded
        assert_eq!(result, "http://example.org/a%2Fb%2Fc");
    }

    // ── Row tests ────────────────────────────────────────────────────────

    #[test]
    fn test_row_get() {
        let mut row = Row::new();
        row.values.insert("key".to_string(), "value".to_string());
        assert_eq!(row.get("key"), Some("value"));
        assert_eq!(row.get("absent"), None);
    }

    #[test]
    fn test_row_contains() {
        let row = Row::from_pairs(vec![("x".to_string(), "1".to_string())]);
        assert!(row.contains("x"));
        assert!(!row.contains("y"));
    }

    #[test]
    fn test_row_display() {
        let row = Row::from_pairs(vec![("a".to_string(), "1".to_string())]);
        let s = format!("{row}");
        assert!(s.contains("a: 1"));
    }

    // ── CSV parsing tests ────────────────────────────────────────────────

    #[test]
    fn test_csv_basic_parse() {
        let csv = "id,name,age\n1,Alice,30\n2,Bob,25";
        let (headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(headers, vec!["id", "name", "age"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name"), Some("Alice"));
        assert_eq!(rows[1].get("age"), Some("25"));
    }

    #[test]
    fn test_csv_tab_delimiter() {
        let csv = "id\tvalue\n1\thello\n2\tworld";
        let (_headers, rows) = MappingEngine::parse_csv(csv, '\t').expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("value"), Some("hello"));
    }

    #[test]
    fn test_csv_quoted_fields() {
        let csv = "id,desc\n1,\"hello, world\"\n2,simple";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows[0].get("desc"), Some("hello, world"));
        assert_eq!(rows[1].get("desc"), Some("simple"));
    }

    #[test]
    fn test_csv_escaped_quotes() {
        let csv = "id,text\n1,\"say \"\"hi\"\"\"\n";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows[0].get("text"), Some("say \"hi\""));
    }

    #[test]
    fn test_csv_crlf_endings() {
        let csv = "id,name\r\n1,Alice\r\n2,Bob\r\n";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name"), Some("Alice"));
    }

    #[test]
    fn test_csv_semicolon_delimiter() {
        let csv = "id;value\n1;alpha\n2;beta";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ';').expect("should succeed");
        assert_eq!(rows[0].get("value"), Some("alpha"));
        assert_eq!(rows[1].get("value"), Some("beta"));
    }

    #[test]
    fn test_csv_empty_content_returns_empty() {
        let (headers, rows) = MappingEngine::parse_csv("", ',').expect("should succeed");
        assert!(headers.is_empty());
        assert!(rows.is_empty());
    }

    #[test]
    fn test_csv_field_count_mismatch_error() {
        let csv = "id,name\n1,Alice,extra\n";
        let err = MappingEngine::parse_csv(csv, ',').unwrap_err();
        assert!(matches!(err, MappingError::CsvParseError { .. }));
    }

    #[test]
    fn test_csv_trailing_empty_lines_skipped() {
        let csv = "id,name\n1,Alice\n\n\n";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows.len(), 1);
    }

    // ── JSON parsing tests ───────────────────────────────────────────────

    #[test]
    fn test_json_flat_objects() {
        let json = r#"[{"id":"1","name":"Alice"},{"id":"2","name":"Bob"}]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name"), Some("Alice"));
        assert_eq!(rows[1].get("id"), Some("2"));
    }

    #[test]
    fn test_json_nested_path() {
        let json = r#"{"data":{"people":[{"id":"1","name":"Alice"}]}}"#;
        let rows = MappingEngine::parse_json(json, Some("data.people")).expect("should succeed");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some("Alice"));
    }

    #[test]
    fn test_json_numeric_values_coerced() {
        let json = r#"[{"id":1,"score":9.5,"active":true}]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert_eq!(rows[0].get("id"), Some("1"));
        assert_eq!(rows[0].get("score"), Some("9.5"));
        assert_eq!(rows[0].get("active"), Some("true"));
    }

    #[test]
    fn test_json_null_value_becomes_empty() {
        let json = r#"[{"id":"1","name":null}]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert_eq!(rows[0].get("name"), Some(""));
    }

    #[test]
    fn test_json_invalid_json_error() {
        let err = MappingEngine::parse_json("not json", None).unwrap_err();
        assert!(matches!(err, MappingError::JsonParseError { .. }));
    }

    #[test]
    fn test_json_path_no_match_error() {
        let json = r#"{"a":{}}"#;
        let err = MappingEngine::parse_json(json, Some("a.b.c")).unwrap_err();
        assert!(matches!(err, MappingError::JsonPathNoMatch { .. }));
    }

    #[test]
    fn test_json_root_not_array_error() {
        let json = r#"{"key":"value"}"#;
        let err = MappingEngine::parse_json(json, None).unwrap_err();
        assert!(matches!(err, MappingError::JsonPathNoMatch { .. }));
    }

    #[test]
    fn test_json_empty_array() {
        let json = r#"[]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert!(rows.is_empty());
    }

    // ── Basic CSV mapping tests ──────────────────────────────────────────

    #[test]
    fn test_csv_mapping_single_predicate() {
        let csv = "id,name\n1,Alice";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let t = &triples[0];
        assert_eq!(t.subject().to_string(), format!("<{}>", ex("1")));
        assert_eq!(t.predicate().to_string(), format!("<{}>", foaf("name")));
        assert!(t.object().to_string().contains("Alice"));
    }

    #[test]
    fn test_csv_mapping_two_rows_two_predicates() {
        let csv = "id,name,age\n1,Alice,30\n2,Bob,25";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .map(foaf("age"), ObjectSpec::Column("age".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 4); // 2 rows × 2 predicates
    }

    #[test]
    fn test_csv_mapping_typed_integer() {
        let csv = "id,age\n1,42";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                foaf("age"),
                ObjectSpec::TypedColumn {
                    column: "age".to_string(),
                    datatype: xsd("integer"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("42"), "object should contain 42, got: {obj}");
        assert!(
            obj.contains("integer"),
            "object should contain xsd:integer, got: {obj}"
        );
    }

    #[test]
    fn test_csv_mapping_typed_date() {
        let csv = "id,dob\n1,1990-01-15";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("dob"),
                ObjectSpec::TypedColumn {
                    column: "dob".to_string(),
                    datatype: xsd("date"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("1990-01-15"));
        assert!(obj.contains("date"));
    }

    #[test]
    fn test_csv_mapping_constant_object() {
        let csv = "id\n1\n2";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                ObjectSpec::Constant("Person".to_string()),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
        for t in &triples {
            assert!(t.object().to_string().contains("Person"));
        }
    }

    #[test]
    fn test_csv_mapping_constant_iri_object() {
        let csv = "id\n1";
        let person_class = ex("Person");
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                ObjectSpec::ConstantIri(person_class.clone()),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains(&person_class));
    }

    #[test]
    fn test_csv_mapping_template_object() {
        let csv = "id,dept\n1,sales";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("department"),
                ObjectSpec::Template(Template::new("http://example.org/dept/{dept}")),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("sales"), "got: {obj}");
    }

    #[test]
    fn test_csv_mapping_lang_fixed() {
        let csv = "id,label\n1,Hello";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("label"),
                ObjectSpec::LangFixed {
                    column: "label".to_string(),
                    lang: "en".to_string(),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("Hello"), "got: {obj}");
        assert!(obj.contains("en"), "got: {obj}");
    }

    #[test]
    fn test_csv_mapping_lang_column() {
        let csv = "id,label,lang\n1,Bonjour,fr";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("label"),
                ObjectSpec::LangColumn {
                    column: "label".to_string(),
                    lang_column: "lang".to_string(),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("Bonjour"), "got: {obj}");
        assert!(obj.contains("fr"), "got: {obj}");
    }

    // ── Named graph tests ────────────────────────────────────────────────

    #[test]
    fn test_named_graph_assignment() {
        let csv = "id\n1";
        let graph = "http://example.org/graph1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(ex("type"), ObjectSpec::Constant("X".to_string()))
            .graph(graph)
            .build();
        assert_eq!(rule.graph_name.as_deref(), Some(graph));
        // Engine still produces triples (named graph metadata is on rule)
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
    }

    // ── JSON mapping tests ───────────────────────────────────────────────

    #[test]
    fn test_json_mapping_flat() {
        let json = r#"[{"id":"1","name":"Alice"},{"id":"2","name":"Bob"}]"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_json_mapping_nested_path() {
        let json = r#"{"items":[{"id":"10","val":"x"},{"id":"20","val":"y"}]}"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source_with_path(json, "items")
            .subject_template(ex("{id}"))
            .map(ex("val"), ObjectSpec::Column("val".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_json_mapping_typed_integer_column() {
        let json = r#"[{"id":"1","count":42}]"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(
                ex("count"),
                ObjectSpec::TypedColumn {
                    column: "count".to_string(),
                    datatype: xsd("integer"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("42"));
        assert!(obj.contains("integer"));
    }

    #[test]
    fn test_json_mapping_multi_predicates() {
        let json = r#"[{"id":"1","name":"Alice","age":"30","city":"NYC"}]"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .map(foaf("age"), ObjectSpec::Column("age".to_string()))
            .map(ex("city"), ObjectSpec::Column("city".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 3);
    }

    // ── Inline values tests ──────────────────────────────────────────────

    #[test]
    fn test_inline_values_mapping() {
        let rule = MappingRuleBuilder::new("test")
            .inline_source(
                vec!["id".to_string(), "name".to_string()],
                vec![
                    vec!["1".to_string(), "Alice".to_string()],
                    vec!["2".to_string(), "Bob".to_string()],
                ],
            )
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    // ── Batch execution tests ────────────────────────────────────────────

    #[test]
    fn test_execute_all_multiple_rules() {
        let csv1 = "id,name\n1,Alice";
        let csv2 = "id,name\n100,Bob";
        let rule1 = MappingRuleBuilder::new("r1")
            .csv_source(csv1)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let rule2 = MappingRuleBuilder::new("r2")
            .csv_source(csv2)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine()
            .execute_all(&[rule1, rule2])
            .expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_execute_all_empty_rules() {
        let triples = engine().execute_all(&[]).expect("should succeed");
        assert!(triples.is_empty());
    }

    // ── Error case tests ─────────────────────────────────────────────────

    #[test]
    fn test_missing_column_error() {
        let csv = "id\n1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::MissingColumn { column, .. } if column == "name"));
    }

    #[test]
    fn test_missing_subject_column_error() {
        let csv = "name\nAlice";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{missing_id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::UnresolvableTemplate { .. }));
    }

    #[test]
    fn test_lenient_engine_skips_bad_rows() {
        let csv = "id,name\n1,Alice\n2,Bob";
        // Subject template referencing column that does not exist for a "ghost" row
        // We test this via a bad predicate-object map with a missing column
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            // "score" does not exist; lenient engine should skip those triples
            .map(ex("score"), ObjectSpec::Column("score".to_string()))
            .build();
        let triples = lenient_engine().execute(&rule).expect("should succeed");
        // Both rows fail on the score column; lenient skips them
        assert_eq!(triples.len(), 0);
    }

    #[test]
    fn test_invalid_predicate_iri_error() {
        let csv = "id\n1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map("not a valid iri", ObjectSpec::Constant("x".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::InvalidPredicateIri { .. }));
    }

    #[test]
    fn test_invalid_subject_iri_error() {
        let csv = "id\n1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            // Template produces a string that may not be a valid absolute IRI
            .subject_template("not-an-iri/{id}")
            .map(foaf("name"), ObjectSpec::Constant("x".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::InvalidIri { .. }));
    }

    // ── Builder pattern tests ────────────────────────────────────────────

    #[test]
    fn test_builder_chain() {
        let rule = MappingRuleBuilder::new("chain_test")
            .csv_source("id,x,y\n1,2,3")
            .subject_template(ex("{id}"))
            .map(ex("x"), ObjectSpec::Column("x".to_string()))
            .map(ex("y"), ObjectSpec::Column("y".to_string()))
            .graph("http://example.org/g1")
            .build();
        assert_eq!(rule.name, "chain_test");
        assert_eq!(rule.predicate_object_maps.len(), 2);
        assert_eq!(rule.graph_name.as_deref(), Some("http://example.org/g1"));
    }

    #[test]
    fn test_builder_csv_with_delimiter() {
        let rule = MappingRuleBuilder::new("pipe")
            .csv_source_with_delimiter("id|name\n1|Alice", '|')
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert!(triples[0].object().to_string().contains("Alice"));
    }

    #[test]
    fn test_builder_json_source_with_path() {
        let json = r#"{"list":[{"id":"5","v":"ok"}]}"#;
        let rule = MappingRuleBuilder::new("j")
            .json_source_with_path(json, "list")
            .subject_template(ex("{id}"))
            .map(ex("v"), ObjectSpec::Column("v".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert!(triples[0].object().to_string().contains("ok"));
    }

    // ── IRI generation tests ─────────────────────────────────────────────

    #[test]
    fn test_iri_from_column_value() {
        let csv = "id,related_id\n1,99";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("related"),
                ObjectSpec::Template(Template::new("http://example.org/item/{related_id}")),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("99"), "got: {obj}");
    }

    #[test]
    fn test_iri_generation_with_special_chars() {
        let csv = "id\nhello world";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(ex("self"), ObjectSpec::ConstantIri(ex("x")))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        // Subject should have space encoded as %20
        let subj = triples[0].subject().to_string();
        assert!(subj.contains("%20"), "got: {subj}");
    }

    // ── Multiple rules interaction tests ─────────────────────────────────

    #[test]
    fn test_multiple_rules_different_sources() {
        let csv = "id,label\n1,CSV-item";
        let json = r#"[{"id":"2","label":"JSON-item"}]"#;
        let rule_csv = MappingRuleBuilder::new("r_csv")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(ex("label"), ObjectSpec::Column("label".to_string()))
            .build();
        let rule_json = MappingRuleBuilder::new("r_json")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(ex("label"), ObjectSpec::Column("label".to_string()))
            .build();
        let triples = engine()
            .execute_all(&[rule_csv, rule_json])
            .expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    // ── Typed literal tests ──────────────────────────────────────────────

    #[test]
    fn test_typed_literal_float() {
        let csv = "id,score\n1,3.14";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("score"),
                ObjectSpec::TypedColumn {
                    column: "score".to_string(),
                    datatype: xsd("decimal"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("3.14"));
        assert!(obj.contains("decimal"));
    }

    #[test]
    fn test_typed_literal_boolean() {
        let csv = "id,active\n1,true";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("active"),
                ObjectSpec::TypedColumn {
                    column: "active".to_string(),
                    datatype: xsd("boolean"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("true"));
        assert!(obj.contains("boolean"));
    }

    // ── Edge case tests ──────────────────────────────────────────────────

    #[test]
    fn test_empty_csv_produces_no_triples() {
        let rule = MappingRuleBuilder::new("empty")
            .csv_source("")
            .subject_template(ex("{id}"))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_csv_only_header_produces_no_triples() {
        let rule = MappingRuleBuilder::new("header-only")
            .csv_source("id,name")
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_json_array_empty_produces_no_triples() {
        let rule = MappingRuleBuilder::new("empty-json")
            .json_source("[]")
            .subject_template(ex("{id}"))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_no_predicate_object_maps_produces_no_triples() {
        let csv = "id\n1\n2";
        let rule = MappingRuleBuilder::new("no-pom")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_percent_encode_unicode() {
        let tpl = Template::new("http://example.org/{name}");
        let mut row = Row::new();
        row.values
            .insert("name".to_string(), "こんにちは".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        // Should be percent-encoded
        assert!(result.starts_with("http://example.org/%"));
        assert!(!result.contains("こんにちは"));
    }

    #[test]
    fn test_csv_mapping_pipe_delimiter_multi_row() {
        let csv = "id|label\n10|alpha\n20|beta\n30|gamma";
        let rule = MappingRuleBuilder::new("pipe-multi")
            .csv_source_with_delimiter(csv, '|')
            .subject_template(ex("{id}"))
            .map(ex("label"), ObjectSpec::Column("label".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 3);
    }

    #[test]
    fn test_mapping_engine_default() {
        let engine = MappingEngine::default();
        assert!(!engine.skip_errors);
    }

    #[test]
    fn test_mapping_rule_add_pom() {
        let mut rule = MappingRule::new(
            "r",
            DataSource::Csv {
                content: "id\n1".to_string(),
                delimiter: ',',
            },
            Template::new(ex("{id}")),
        );
        assert!(rule.predicate_object_maps.is_empty());
        rule.add_predicate_object_map(PredicateObjectMap::new(
            ex("p"),
            ObjectSpec::Constant("v".to_string()),
        ));
        assert_eq!(rule.predicate_object_maps.len(), 1);
    }

    #[test]
    fn test_predicate_object_map_construction() {
        let pom = PredicateObjectMap::new(
            "http://example.org/pred",
            ObjectSpec::Column("col".to_string()),
        );
        assert_eq!(pom.predicate, "http://example.org/pred");
    }

    #[test]
    fn test_row_from_pairs() {
        let row = Row::from_pairs(vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string()),
        ]);
        assert_eq!(row.get("a"), Some("1"));
        assert_eq!(row.get("b"), Some("2"));
    }

    #[test]
    fn test_json_deeply_nested_path() {
        let json = r#"{"a":{"b":{"c":[{"id":"1","name":"deep"}]}}}"#;
        let rows = MappingEngine::parse_json(json, Some("a.b.c")).expect("should succeed");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some("deep"));
    }

    #[test]
    fn test_csv_quoted_field_with_newline() {
        let csv = "id,desc\n1,\"line1\nline2\"\n2,simple";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert!(rows[0].get("desc").expect("should succeed").contains('\n'));
        assert_eq!(rows[1].get("desc"), Some("simple"));
    }

    #[test]
    fn test_template_display() {
        let tpl = Template::new("http://example.org/{id}");
        assert_eq!(tpl.to_string(), "http://example.org/{id}");
    }

    #[test]
    fn test_row_iter() {
        let row = Row::from_pairs(vec![
            ("x".to_string(), "1".to_string()),
            ("y".to_string(), "2".to_string()),
        ]);
        let count = row.iter().count();
        assert_eq!(count, 2);
    }
}
