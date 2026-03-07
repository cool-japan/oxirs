//! # SPARQL Results CSV/TSV Serializer
//!
//! Implements the W3C SPARQL Query Results CSV and TSV Formats specification
//! for serializing SPARQL SELECT query results.
//!
//! ## Features
//!
//! - **CSV format**: RFC 4180-compliant CSV output with proper quoting
//! - **TSV format**: Tab-separated output per W3C spec
//! - **Header row**: Variable names as the first row
//! - **Type handling**: Proper serialization of IRIs, literals, blank nodes
//! - **Streaming**: Can serialize row-by-row for large result sets
//!
//! ## Reference
//!
//! - [W3C SPARQL 1.1 Query Results CSV and TSV Formats](https://www.w3.org/TR/sparql11-results-csv-tsv/)

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────

/// An RDF term in a SPARQL result binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RdfTerm {
    /// An IRI reference.
    Iri(String),
    /// A plain literal with optional language tag.
    Literal {
        value: String,
        language: Option<String>,
        datatype: Option<String>,
    },
    /// A blank node.
    BlankNode(String),
    /// Unbound (no value for this variable in this row).
    Unbound,
}

impl RdfTerm {
    pub fn iri(s: impl Into<String>) -> Self {
        RdfTerm::Iri(s.into())
    }

    pub fn literal(value: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            language: None,
            datatype: None,
        }
    }

    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            language: Some(lang.into()),
            datatype: None,
        }
    }

    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        RdfTerm::Literal {
            value: value.into(),
            language: None,
            datatype: Some(datatype.into()),
        }
    }

    pub fn blank_node(id: impl Into<String>) -> Self {
        RdfTerm::BlankNode(id.into())
    }
}

/// A single row of SPARQL result bindings.
#[derive(Debug, Clone)]
pub struct ResultRow {
    /// Values for each variable (same order as variables).
    pub values: Vec<RdfTerm>,
}

impl ResultRow {
    pub fn new(values: Vec<RdfTerm>) -> Self {
        Self { values }
    }
}

/// A complete SPARQL SELECT result set.
#[derive(Debug, Clone)]
pub struct SparqlResultSet {
    /// Variable names (without '?').
    pub variables: Vec<String>,
    /// Result rows.
    pub rows: Vec<ResultRow>,
}

impl SparqlResultSet {
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            variables,
            rows: Vec::new(),
        }
    }

    pub fn add_row(&mut self, row: ResultRow) {
        self.rows.push(row);
    }

    /// Number of variables.
    pub fn width(&self) -> usize {
        self.variables.len()
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.rows.len()
    }
}

// ─────────────────────────────────────────────
// CSV/TSV serialization format
// ─────────────────────────────────────────────

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultFormat {
    Csv,
    Tsv,
}

/// Configuration for the serializer.
#[derive(Debug, Clone)]
pub struct SerializerConfig {
    /// Output format.
    pub format: ResultFormat,
    /// Whether to include the header row (default: true).
    pub include_header: bool,
    /// For TSV: whether to prefix IRIs with '<' and '>' (default: true).
    pub tsv_bracket_iris: bool,
}

impl Default for SerializerConfig {
    fn default() -> Self {
        Self {
            format: ResultFormat::Csv,
            include_header: true,
            tsv_bracket_iris: true,
        }
    }
}

// ─────────────────────────────────────────────
// Serializers
// ─────────────────────────────────────────────

/// Serialize a result set to CSV format (W3C SPARQL Results CSV Format).
pub fn serialize_csv(result_set: &SparqlResultSet) -> String {
    let config = SerializerConfig {
        format: ResultFormat::Csv,
        ..Default::default()
    };
    serialize(result_set, &config)
}

/// Serialize a result set to TSV format (W3C SPARQL Results TSV Format).
pub fn serialize_tsv(result_set: &SparqlResultSet) -> String {
    let config = SerializerConfig {
        format: ResultFormat::Tsv,
        ..Default::default()
    };
    serialize(result_set, &config)
}

/// Serialize a result set with custom configuration.
pub fn serialize(result_set: &SparqlResultSet, config: &SerializerConfig) -> String {
    let separator = match config.format {
        ResultFormat::Csv => ',',
        ResultFormat::Tsv => '\t',
    };
    let mut output = String::new();

    // Header row
    if config.include_header {
        let header: Vec<String> = result_set
            .variables
            .iter()
            .map(|v| match config.format {
                ResultFormat::Csv => csv_escape(v),
                ResultFormat::Tsv => format!("?{v}"),
            })
            .collect();
        output.push_str(&header.join(&separator.to_string()));
        output.push_str("\r\n");
    }

    // Data rows
    for row in &result_set.rows {
        let values: Vec<String> = row
            .values
            .iter()
            .map(|term| format_term(term, config))
            .collect();
        output.push_str(&values.join(&separator.to_string()));
        output.push_str("\r\n");
    }

    output
}

/// Serialize a single row (for streaming).
pub fn serialize_row(row: &ResultRow, config: &SerializerConfig) -> String {
    let separator = match config.format {
        ResultFormat::Csv => ',',
        ResultFormat::Tsv => '\t',
    };
    let values: Vec<String> = row
        .values
        .iter()
        .map(|term| format_term(term, config))
        .collect();
    let mut output = values.join(&separator.to_string());
    output.push_str("\r\n");
    output
}

/// Serialize the header row only.
pub fn serialize_header(variables: &[String], config: &SerializerConfig) -> String {
    let separator = match config.format {
        ResultFormat::Csv => ',',
        ResultFormat::Tsv => '\t',
    };
    let header: Vec<String> = variables
        .iter()
        .map(|v| match config.format {
            ResultFormat::Csv => csv_escape(v),
            ResultFormat::Tsv => format!("?{v}"),
        })
        .collect();
    let mut output = header.join(&separator.to_string());
    output.push_str("\r\n");
    output
}

// ─── Internal formatting ─────────────────────────────────

fn format_term(term: &RdfTerm, config: &SerializerConfig) -> String {
    match config.format {
        ResultFormat::Csv => format_term_csv(term),
        ResultFormat::Tsv => format_term_tsv(term, config.tsv_bracket_iris),
    }
}

/// CSV format: IRIs as bare strings, literals as values, blank nodes as _:id
fn format_term_csv(term: &RdfTerm) -> String {
    match term {
        RdfTerm::Iri(iri) => csv_escape(iri),
        RdfTerm::Literal {
            value,
            language,
            datatype,
        } => {
            if language.is_some() || datatype.is_some() {
                // CSV just uses the lexical form
                csv_escape(value)
            } else {
                csv_escape(value)
            }
        }
        RdfTerm::BlankNode(id) => csv_escape(&format!("_:{id}")),
        RdfTerm::Unbound => String::new(),
    }
}

/// TSV format: IRIs in angle brackets, literals with type annotations
fn format_term_tsv(term: &RdfTerm, bracket_iris: bool) -> String {
    match term {
        RdfTerm::Iri(iri) => {
            if bracket_iris {
                format!("<{iri}>")
            } else {
                iri.clone()
            }
        }
        RdfTerm::Literal {
            value,
            language: Some(lang),
            ..
        } => {
            format!("\"{}\"@{}", tsv_escape_string(value), lang)
        }
        RdfTerm::Literal {
            value,
            datatype: Some(dt),
            ..
        } => {
            format!("\"{}\"^^<{}>", tsv_escape_string(value), dt)
        }
        RdfTerm::Literal { value, .. } => {
            format!("\"{}\"", tsv_escape_string(value))
        }
        RdfTerm::BlankNode(id) => format!("_:{id}"),
        RdfTerm::Unbound => String::new(),
    }
}

/// CSV escaping: quote fields that contain comma, quote, or newline.
fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        let escaped = value.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        value.to_string()
    }
}

/// TSV string escaping: escape tabs, newlines, quotes, backslashes.
fn tsv_escape_string(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\t' => result.push_str("\\t"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            _ => result.push(ch),
        }
    }
    result
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result_set() -> SparqlResultSet {
        let mut rs = SparqlResultSet::new(vec!["name".to_string(), "age".to_string()]);
        rs.add_row(ResultRow::new(vec![
            RdfTerm::literal("Alice"),
            RdfTerm::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ]));
        rs.add_row(ResultRow::new(vec![
            RdfTerm::literal("Bob"),
            RdfTerm::typed_literal("25", "http://www.w3.org/2001/XMLSchema#integer"),
        ]));
        rs
    }

    fn iri_result_set() -> SparqlResultSet {
        let mut rs = SparqlResultSet::new(vec!["s".to_string(), "p".to_string(), "o".to_string()]);
        rs.add_row(ResultRow::new(vec![
            RdfTerm::iri("http://example.org/alice"),
            RdfTerm::iri("http://xmlns.com/foaf/0.1/name"),
            RdfTerm::literal("Alice"),
        ]));
        rs.add_row(ResultRow::new(vec![
            RdfTerm::iri("http://example.org/alice"),
            RdfTerm::iri("http://xmlns.com/foaf/0.1/knows"),
            RdfTerm::iri("http://example.org/bob"),
        ]));
        rs
    }

    // ═══ RdfTerm tests ═══════════════════════════════════

    #[test]
    fn test_rdf_term_iri() {
        let term = RdfTerm::iri("http://example.org");
        assert_eq!(term, RdfTerm::Iri("http://example.org".to_string()));
    }

    #[test]
    fn test_rdf_term_literal() {
        let term = RdfTerm::literal("hello");
        assert!(matches!(term, RdfTerm::Literal { value, .. } if value == "hello"));
    }

    #[test]
    fn test_rdf_term_lang_literal() {
        let term = RdfTerm::lang_literal("hello", "en");
        assert!(matches!(
            term,
            RdfTerm::Literal { language: Some(l), .. } if l == "en"
        ));
    }

    #[test]
    fn test_rdf_term_typed_literal() {
        let term = RdfTerm::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert!(matches!(
            term,
            RdfTerm::Literal { datatype: Some(dt), .. } if dt.contains("integer")
        ));
    }

    #[test]
    fn test_rdf_term_blank_node() {
        let term = RdfTerm::blank_node("b0");
        assert_eq!(term, RdfTerm::BlankNode("b0".to_string()));
    }

    // ═══ Result set tests ════════════════════════════════

    #[test]
    fn test_result_set_dimensions() {
        let rs = sample_result_set();
        assert_eq!(rs.width(), 2);
        assert_eq!(rs.height(), 2);
    }

    #[test]
    fn test_result_set_empty() {
        let rs = SparqlResultSet::new(vec!["x".to_string()]);
        assert_eq!(rs.width(), 1);
        assert_eq!(rs.height(), 0);
    }

    // ═══ CSV serialization tests ═════════════════════════

    #[test]
    fn test_csv_basic() {
        let rs = sample_result_set();
        let csv = serialize_csv(&rs);
        assert!(csv.starts_with("name,age\r\n"));
        assert!(csv.contains("Alice"));
        assert!(csv.contains("30"));
    }

    #[test]
    fn test_csv_header() {
        let rs = sample_result_set();
        let csv = serialize_csv(&rs);
        let first_line = csv.lines().next().expect("first line");
        assert_eq!(first_line, "name,age");
    }

    #[test]
    fn test_csv_iri_values() {
        let rs = iri_result_set();
        let csv = serialize_csv(&rs);
        assert!(csv.contains("http://example.org/alice"));
    }

    #[test]
    fn test_csv_escaping_comma() {
        let mut rs = SparqlResultSet::new(vec!["val".to_string()]);
        rs.add_row(ResultRow::new(vec![RdfTerm::literal("hello, world")]));
        let csv = serialize_csv(&rs);
        assert!(csv.contains("\"hello, world\""));
    }

    #[test]
    fn test_csv_escaping_quote() {
        let mut rs = SparqlResultSet::new(vec!["val".to_string()]);
        rs.add_row(ResultRow::new(vec![RdfTerm::literal("say \"hi\"")]));
        let csv = serialize_csv(&rs);
        assert!(csv.contains("\"say \"\"hi\"\"\""));
    }

    #[test]
    fn test_csv_unbound() {
        let mut rs = SparqlResultSet::new(vec!["a".to_string(), "b".to_string()]);
        rs.add_row(ResultRow::new(vec![
            RdfTerm::literal("x"),
            RdfTerm::Unbound,
        ]));
        let csv = serialize_csv(&rs);
        // Unbound should produce empty field
        assert!(csv.contains("x,\r\n"));
    }

    #[test]
    fn test_csv_blank_node() {
        let mut rs = SparqlResultSet::new(vec!["node".to_string()]);
        rs.add_row(ResultRow::new(vec![RdfTerm::blank_node("b0")]));
        let csv = serialize_csv(&rs);
        assert!(csv.contains("_:b0"));
    }

    #[test]
    fn test_csv_crlf_line_endings() {
        let rs = sample_result_set();
        let csv = serialize_csv(&rs);
        assert!(csv.contains("\r\n"));
    }

    #[test]
    fn test_csv_no_header() {
        let rs = sample_result_set();
        let config = SerializerConfig {
            format: ResultFormat::Csv,
            include_header: false,
            ..Default::default()
        };
        let csv = serialize(&rs, &config);
        // First line should be data, not header
        let first_line = csv.lines().next().expect("first line");
        assert!(!first_line.contains("name"));
        assert!(first_line.contains("Alice"));
    }

    // ═══ TSV serialization tests ═════════════════════════

    #[test]
    fn test_tsv_basic() {
        let rs = sample_result_set();
        let tsv = serialize_tsv(&rs);
        assert!(tsv.starts_with("?name\t?age\r\n"));
    }

    #[test]
    fn test_tsv_header_prefixed() {
        let rs = sample_result_set();
        let tsv = serialize_tsv(&rs);
        let first_line = tsv.lines().next().expect("first line");
        assert!(first_line.starts_with("?name"));
        assert!(first_line.contains("\t?age"));
    }

    #[test]
    fn test_tsv_iri_bracketed() {
        let rs = iri_result_set();
        let tsv = serialize_tsv(&rs);
        assert!(tsv.contains("<http://example.org/alice>"));
    }

    #[test]
    fn test_tsv_literal_quoted() {
        let rs = sample_result_set();
        let tsv = serialize_tsv(&rs);
        assert!(tsv.contains("\"Alice\""));
    }

    #[test]
    fn test_tsv_typed_literal() {
        let rs = sample_result_set();
        let tsv = serialize_tsv(&rs);
        assert!(tsv.contains("^^<http://www.w3.org/2001/XMLSchema#integer>"));
    }

    #[test]
    fn test_tsv_lang_literal() {
        let mut rs = SparqlResultSet::new(vec!["label".to_string()]);
        rs.add_row(ResultRow::new(vec![RdfTerm::lang_literal("chat", "fr")]));
        let tsv = serialize_tsv(&rs);
        assert!(tsv.contains("\"chat\"@fr"));
    }

    #[test]
    fn test_tsv_blank_node() {
        let mut rs = SparqlResultSet::new(vec!["node".to_string()]);
        rs.add_row(ResultRow::new(vec![RdfTerm::blank_node("b1")]));
        let tsv = serialize_tsv(&rs);
        assert!(tsv.contains("_:b1"));
    }

    #[test]
    fn test_tsv_escaping_tab() {
        let mut rs = SparqlResultSet::new(vec!["val".to_string()]);
        rs.add_row(ResultRow::new(vec![RdfTerm::literal("a\tb")]));
        let tsv = serialize_tsv(&rs);
        assert!(tsv.contains("a\\tb"));
    }

    #[test]
    fn test_tsv_escaping_newline() {
        let mut rs = SparqlResultSet::new(vec!["val".to_string()]);
        rs.add_row(ResultRow::new(vec![RdfTerm::literal("line1\nline2")]));
        let tsv = serialize_tsv(&rs);
        assert!(tsv.contains("line1\\nline2"));
    }

    // ═══ Streaming serialization tests ═══════════════════

    #[test]
    fn test_serialize_header_csv() {
        let vars = vec!["name".to_string(), "age".to_string()];
        let config = SerializerConfig {
            format: ResultFormat::Csv,
            ..Default::default()
        };
        let header = serialize_header(&vars, &config);
        assert_eq!(header, "name,age\r\n");
    }

    #[test]
    fn test_serialize_header_tsv() {
        let vars = vec!["name".to_string(), "age".to_string()];
        let config = SerializerConfig {
            format: ResultFormat::Tsv,
            ..Default::default()
        };
        let header = serialize_header(&vars, &config);
        assert_eq!(header, "?name\t?age\r\n");
    }

    #[test]
    fn test_serialize_row_csv() {
        let row = ResultRow::new(vec![RdfTerm::literal("Alice"), RdfTerm::literal("30")]);
        let config = SerializerConfig {
            format: ResultFormat::Csv,
            ..Default::default()
        };
        let output = serialize_row(&row, &config);
        assert_eq!(output, "Alice,30\r\n");
    }

    #[test]
    fn test_serialize_row_tsv() {
        let row = ResultRow::new(vec![
            RdfTerm::iri("http://example.org/a"),
            RdfTerm::literal("hello"),
        ]);
        let config = SerializerConfig {
            format: ResultFormat::Tsv,
            ..Default::default()
        };
        let output = serialize_row(&row, &config);
        assert!(output.contains("<http://example.org/a>"));
        assert!(output.contains("\t"));
    }

    // ═══ Empty result set tests ══════════════════════════

    #[test]
    fn test_csv_empty_results() {
        let rs = SparqlResultSet::new(vec!["x".to_string(), "y".to_string()]);
        let csv = serialize_csv(&rs);
        assert_eq!(csv, "x,y\r\n");
    }

    #[test]
    fn test_tsv_empty_results() {
        let rs = SparqlResultSet::new(vec!["x".to_string()]);
        let tsv = serialize_tsv(&rs);
        assert_eq!(tsv, "?x\r\n");
    }

    // ═══ Config tests ════════════════════════════════════

    #[test]
    fn test_default_config() {
        let config = SerializerConfig::default();
        assert_eq!(config.format, ResultFormat::Csv);
        assert!(config.include_header);
        assert!(config.tsv_bracket_iris);
    }

    #[test]
    fn test_tsv_no_bracket_iris() {
        let rs = iri_result_set();
        let config = SerializerConfig {
            format: ResultFormat::Tsv,
            tsv_bracket_iris: false,
            include_header: true,
        };
        let tsv = serialize(&rs, &config);
        assert!(!tsv.contains("<http://"));
        assert!(tsv.contains("http://example.org/alice"));
    }
}
