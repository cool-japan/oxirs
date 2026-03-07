//! CSV-to-RDF and RDF-to-CSV conversion utilities.
//!
//! Converts tabular CSV data to RDF triple form and back, using a configurable
//! column-to-predicate mapping.  Also provides JSON-LD and N-Quads importers
//! and an N-Quads exporter.

use crate::error::{Result, TdbError};
use std::collections::HashMap;

// ─── RdfColumnType ───────────────────────────────────────────────────────────

/// How to represent a CSV column's values in RDF
#[derive(Debug, Clone)]
pub enum RdfColumnType {
    /// Value is an IRI — wrapped in angle brackets
    Iri,
    /// Value is a plain or datatype-annotated literal. `None` = plain string.
    Literal(Option<String>),
    /// Value is an `xsd:integer` literal
    Integer,
    /// Value is an `xsd:decimal` literal
    Float,
    /// Value is an `xsd:boolean` literal
    Boolean,
    /// Value is an `xsd:dateTime` literal
    DateTime,
}

// ─── CsvRdfMapper ────────────────────────────────────────────────────────────

/// Maps CSV rows to RDF triples and back.
///
/// Given a CSV file, each row produces one subject (taken from
/// `subject_col`) and one triple per other column:
///
/// ```text
/// <{base_iri}{cell}> <{predicate_template_with_col_name}> <cell_value> .
/// ```
pub struct CsvRdfMapper {
    /// Name of the CSV column to use as the subject
    pub subject_col: String,
    /// Template for predicate IRIs; `{column}` is replaced with the column name
    pub predicate_template: String,
    /// Base IRI prepended to relative subject values
    pub base_iri: String,
    /// Per-column type overrides
    pub column_types: HashMap<String, RdfColumnType>,
}

impl CsvRdfMapper {
    /// Create a new mapper with sensible defaults.
    pub fn new(subject_col: &str, base_iri: &str) -> Self {
        Self {
            subject_col: subject_col.to_string(),
            predicate_template: "http://example.org/{column}".to_string(),
            base_iri: base_iri.to_string(),
            column_types: HashMap::new(),
        }
    }

    /// Register a type for `col`. Returns `&mut Self` for chaining.
    pub fn map_column(&mut self, col: &str, col_type: RdfColumnType) -> &mut Self {
        self.column_types.insert(col.to_string(), col_type);
        self
    }

    /// Parse CSV data and return a list of `(subject, predicate, object)` triples.
    ///
    /// The first row is treated as a header.
    pub fn csv_to_triples(&self, csv_data: &str) -> Result<Vec<(String, String, String)>> {
        let mut lines = csv_data.lines();
        let header_line = lines
            .next()
            .ok_or_else(|| TdbError::InvalidInput("Empty CSV".to_string()))?;
        let headers: Vec<&str> = split_csv_line(header_line);

        let subject_idx = headers
            .iter()
            .position(|h| *h == self.subject_col.as_str())
            .ok_or_else(|| {
                TdbError::InvalidInput(format!(
                    "Subject column '{}' not found in CSV headers",
                    self.subject_col
                ))
            })?;

        let mut triples = Vec::new();

        for (line_no, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let cells: Vec<&str> = split_csv_line(line);
            if cells.len() < headers.len() {
                return Err(TdbError::InvalidInput(format!(
                    "Row {} has fewer columns than header",
                    line_no + 2
                )));
            }

            let raw_subject = cells[subject_idx].trim();
            let subject = if raw_subject.starts_with("http://")
                || raw_subject.starts_with("https://")
                || raw_subject.starts_with("urn:")
            {
                raw_subject.to_string()
            } else {
                format!("{}{}", self.base_iri, raw_subject)
            };

            for (col_idx, header) in headers.iter().enumerate() {
                if col_idx == subject_idx {
                    continue;
                }
                let predicate = self.predicate_template.replace("{column}", header);
                let raw_value = cells[col_idx].trim().trim_matches('"');
                let object = self.format_object(header, raw_value);
                triples.push((subject.clone(), predicate, object));
            }
        }

        Ok(triples)
    }

    /// Reconstruct CSV from a list of triples.
    ///
    /// Groups triples by subject, producing one row per subject.
    /// Column headers are derived from predicate names (after the last `/` or `#`).
    pub fn triples_to_csv(&self, triples: &[(String, String, String)]) -> String {
        // Collect predicate → short name mapping (stable order)
        let mut predicate_order: Vec<String> = Vec::new();
        let mut seen_preds: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (_, p, _) in triples {
            if seen_preds.insert(p.clone()) {
                predicate_order.push(p.clone());
            }
        }

        // Group by subject (preserve insertion order)
        let mut subject_order: Vec<String> = Vec::new();
        let mut by_subject: HashMap<String, HashMap<String, String>> = HashMap::new();
        for (s, p, o) in triples {
            if !by_subject.contains_key(s) {
                subject_order.push(s.clone());
            }
            by_subject
                .entry(s.clone())
                .or_default()
                .insert(p.clone(), o.clone());
        }

        // Header row: subject column + predicate short names
        let short_names: Vec<String> = predicate_order
            .iter()
            .map(|p| predicate_short_name(p))
            .collect();

        let mut csv = format!("{},{}\n", self.subject_col, short_names.join(","));

        for subject in &subject_order {
            let preds = &by_subject[subject];
            let mut row = vec![subject.clone()];
            for pred in &predicate_order {
                row.push(preds.get(pred).cloned().unwrap_or_default());
            }
            csv.push_str(&row.join(","));
            csv.push('\n');
        }

        csv
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn format_object(&self, col: &str, value: &str) -> String {
        match self.column_types.get(col) {
            Some(RdfColumnType::Iri) => {
                if value.starts_with("http://")
                    || value.starts_with("https://")
                    || value.starts_with("urn:")
                {
                    value.to_string()
                } else {
                    format!("{}{}", self.base_iri, value)
                }
            }
            Some(RdfColumnType::Integer) => {
                format!("\"{}\"^^xsd:integer", value)
            }
            Some(RdfColumnType::Float) => {
                format!("\"{}\"^^xsd:decimal", value)
            }
            Some(RdfColumnType::Boolean) => {
                format!("\"{}\"^^xsd:boolean", value)
            }
            Some(RdfColumnType::DateTime) => {
                format!("\"{}\"^^xsd:dateTime", value)
            }
            Some(RdfColumnType::Literal(Some(datatype))) => {
                format!("\"{}\"^^{}", value, datatype)
            }
            Some(RdfColumnType::Literal(None)) | None => {
                format!("\"{}\"", value)
            }
        }
    }
}

/// Split a single CSV line on commas, respecting double-quoted fields.
fn split_csv_line(line: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut start = 0usize;
    let mut in_quotes = false;
    let bytes = line.as_bytes();

    for i in 0..bytes.len() {
        match bytes[i] {
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => {
                fields.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    fields.push(&line[start..]);
    fields
}

/// Derive a short display name from a full IRI predicate.
fn predicate_short_name(pred: &str) -> String {
    let after_slash = pred.rfind('/').map(|i| &pred[i + 1..]).unwrap_or(pred);
    let after_hash = after_slash
        .rfind('#')
        .map(|i| &after_slash[i + 1..])
        .unwrap_or(after_slash);
    after_hash.to_string()
}

// ─── JsonLdImporter ──────────────────────────────────────────────────────────

/// Imports triples from a simplified JSON-LD document.
pub struct JsonLdImporter;

impl JsonLdImporter {
    /// Parse a JSON-LD string and return `(subject, predicate, object)` triples.
    ///
    /// Handles:
    /// - Top-level array or single object
    /// - `@id`, `@type`, `@value`, `@language` keywords
    /// - Nested `@graph` arrays
    pub fn import(json_ld: &str) -> Result<Vec<(String, String, String)>> {
        let value: serde_json::Value = serde_json::from_str(json_ld)
            .map_err(|e| TdbError::Deserialization(format!("JSON-LD parse: {}", e)))?;
        let mut triples = Vec::new();
        Self::extract_triples(&value, &mut triples);
        Ok(triples)
    }

    fn extract_triples(value: &serde_json::Value, triples: &mut Vec<(String, String, String)>) {
        match value {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::extract_triples(item, triples);
                }
            }
            serde_json::Value::Object(map) => {
                // Handle @graph
                if let Some(graph) = map.get("@graph") {
                    Self::extract_triples(graph, triples);
                }

                let subject = map
                    .get("@id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("_:anonymous")
                    .to_string();

                for (key, val) in map {
                    match key.as_str() {
                        "@id" | "@context" | "@graph" => continue,
                        "@type" => {
                            let rdf_type =
                                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string();
                            match val {
                                serde_json::Value::String(s) => {
                                    triples.push((subject.clone(), rdf_type, s.clone()));
                                }
                                serde_json::Value::Array(types) => {
                                    for t in types {
                                        if let Some(s) = t.as_str() {
                                            triples.push((
                                                subject.clone(),
                                                rdf_type.clone(),
                                                s.to_string(),
                                            ));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        predicate => {
                            let obj_str = json_ld_value_to_string(val);
                            if let Some(s) = obj_str {
                                triples.push((subject.clone(), predicate.to_string(), s));
                            } else if let serde_json::Value::Array(arr) = val {
                                for item in arr {
                                    let s = json_ld_value_to_string(item);
                                    if let Some(s) = s {
                                        triples.push((subject.clone(), predicate.to_string(), s));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Convert a JSON-LD value node to a string representation.
fn json_ld_value_to_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Object(map) => {
            if let Some(id) = map.get("@id").and_then(|v| v.as_str()) {
                Some(id.to_string())
            } else {
                map.get("@value")
                    .and_then(|v| v.as_str())
                    .map(|val| val.to_string())
            }
        }
        _ => None,
    }
}

// ─── NQuadsImporter ──────────────────────────────────────────────────────────

/// A parsed N-Quad tuple: `(subject, predicate, object, optional_graph)`.
pub type NQuadTuple = (String, String, String, Option<String>);

/// Imports quads from an N-Quads formatted string.
pub struct NQuadsImporter;

impl NQuadsImporter {
    /// Parse N-Quads data and return `(subject, predicate, object, graph?)` tuples.
    ///
    /// Format: `<s> <p> <o> [<g>] .`
    /// Blank nodes (`_:xyz`) are kept as-is.
    pub fn import(nquads: &str) -> Result<Vec<NQuadTuple>> {
        let mut quads = Vec::new();
        for (line_no, line) in nquads.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match parse_nquad_line(line) {
                Ok(quad) => quads.push(quad),
                Err(e) => {
                    return Err(TdbError::InvalidInput(format!(
                        "N-Quads line {}: {}",
                        line_no + 1,
                        e
                    )))
                }
            }
        }
        Ok(quads)
    }
}

/// Parse a single N-Quads line into a (s, p, o, graph?) tuple.
fn parse_nquad_line(line: &str) -> std::result::Result<NQuadTuple, String> {
    let line = line.trim_end_matches('.');
    let line = line.trim();

    let mut tokens = tokenize_nquad(line);

    if tokens.len() < 3 {
        return Err(format!("Expected at least 3 tokens, got {}", tokens.len()));
    }

    let s = tokens.remove(0);
    let p = tokens.remove(0);
    let o = tokens.remove(0);
    let g = if tokens.is_empty() {
        None
    } else {
        Some(tokens.remove(0))
    };

    Ok((s, p, o, g))
}

/// Tokenise a single N-Quads line into IRI, blank-node, or literal tokens.
fn tokenize_nquad(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip whitespace
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }

        if chars[i] == '<' {
            // IRI term
            let mut end = i + 1;
            while end < chars.len() && chars[end] != '>' {
                end += 1;
            }
            tokens.push(chars[i + 1..end].iter().collect());
            i = end + 1;
        } else if chars[i] == '_' && i + 1 < chars.len() && chars[i + 1] == ':' {
            // Blank node
            let mut end = i + 2;
            while end < chars.len() && !chars[end].is_whitespace() {
                end += 1;
            }
            tokens.push(chars[i..end].iter().collect());
            i = end;
        } else if chars[i] == '"' {
            // Literal (possibly with language tag or datatype)
            let mut end = i + 1;
            while end < chars.len() && chars[end] != '"' {
                if chars[end] == '\\' {
                    end += 1; // skip escape
                }
                end += 1;
            }
            let literal: String = chars[i..=end].iter().collect();
            let mut full = literal;
            i = end + 1;
            // Check for language tag or datatype
            if i < chars.len() && chars[i] == '@' {
                let lang_start = i;
                i += 1;
                while i < chars.len() && !chars[i].is_whitespace() {
                    i += 1;
                }
                full.push_str(&chars[lang_start..i].iter().collect::<String>());
            } else if i + 1 < chars.len() && chars[i] == '^' && chars[i + 1] == '^' {
                let dt_start = i;
                i += 2;
                if i < chars.len() && chars[i] == '<' {
                    i += 1;
                    while i < chars.len() && chars[i] != '>' {
                        i += 1;
                    }
                    i += 1;
                }
                full.push_str(&chars[dt_start..i].iter().collect::<String>());
            }
            tokens.push(full);
        } else {
            // Skip unknown token character
            i += 1;
        }
    }
    tokens
}

// ─── NQuadsExporter ──────────────────────────────────────────────────────────

/// Exports triples to N-Quads format.
pub struct NQuadsExporter;

impl NQuadsExporter {
    /// Serialize `triples` to an N-Quads string.
    ///
    /// If `graph` is `Some`, each line includes the named graph IRI.
    pub fn export(triples: &[(String, String, String)], graph: Option<&str>) -> String {
        let mut out = String::new();
        for (s, p, o) in triples {
            let s_term = iri_or_blank(s);
            let p_term = iri_or_blank(p);
            let o_term = if o.starts_with('"') {
                o.clone()
            } else {
                iri_or_blank(o)
            };
            if let Some(g) = graph {
                out.push_str(&format!("{} {} {} <{}> .\n", s_term, p_term, o_term, g));
            } else {
                out.push_str(&format!("{} {} {} .\n", s_term, p_term, o_term));
            }
        }
        out
    }
}

/// Format a term as an IRI or leave blank-node notation unchanged.
fn iri_or_blank(term: &str) -> String {
    if term.starts_with("_:") {
        term.to_string()
    } else {
        format!("<{}>", term)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CsvRdfMapper ─────────────────────────────────────────────────────────

    #[test]
    fn test_csv_rdf_mapper_new() {
        let mapper = CsvRdfMapper::new("id", "http://ex.org/");
        assert_eq!(mapper.subject_col, "id");
        assert_eq!(mapper.base_iri, "http://ex.org/");
    }

    #[test]
    fn test_csv_to_triples_basic() {
        let mut mapper = CsvRdfMapper::new("id", "http://ex.org/");
        mapper.predicate_template = "http://ex.org/{column}".to_string();
        let csv = "id,name,age\nhttp://ex.org/alice,Alice,30\n";
        let triples = mapper.csv_to_triples(csv).unwrap();
        assert_eq!(triples.len(), 2); // name and age columns
        assert!(triples.iter().any(|(_, p, _)| p.contains("name")));
        assert!(triples.iter().any(|(_, p, _)| p.contains("age")));
    }

    #[test]
    fn test_csv_to_triples_relative_subject() {
        let mapper = CsvRdfMapper::new("id", "http://ex.org/");
        let csv = "id,label\nalice,Alice\n";
        let triples = mapper.csv_to_triples(csv).unwrap();
        assert_eq!(triples[0].0, "http://ex.org/alice");
    }

    #[test]
    fn test_csv_to_triples_absolute_subject_preserved() {
        let mapper = CsvRdfMapper::new("id", "http://ex.org/");
        let csv = "id,label\nhttp://other.org/bob,Bob\n";
        let triples = mapper.csv_to_triples(csv).unwrap();
        assert_eq!(triples[0].0, "http://other.org/bob");
    }

    #[test]
    fn test_csv_to_triples_empty_csv_error() {
        let mapper = CsvRdfMapper::new("id", "http://ex.org/");
        let result = mapper.csv_to_triples("");
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_to_triples_missing_subject_col_error() {
        let mapper = CsvRdfMapper::new("nonexistent", "http://ex.org/");
        let result = mapper.csv_to_triples("id,name\nhttp://ex.org/a,Alice\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_map_column_integer() {
        let mut mapper = CsvRdfMapper::new("id", "http://ex.org/");
        mapper.map_column("age", RdfColumnType::Integer);
        let csv = "id,age\nhttp://ex.org/alice,30\n";
        let triples = mapper.csv_to_triples(csv).unwrap();
        let age_triple = triples.iter().find(|(_, p, _)| p.contains("age")).unwrap();
        assert!(
            age_triple.2.contains("xsd:integer"),
            "got: {}",
            age_triple.2
        );
    }

    #[test]
    fn test_map_column_iri() {
        let mut mapper = CsvRdfMapper::new("id", "http://ex.org/");
        mapper.map_column("related", RdfColumnType::Iri);
        let csv = "id,related\nhttp://ex.org/alice,bob\n";
        let triples = mapper.csv_to_triples(csv).unwrap();
        let rel = triples
            .iter()
            .find(|(_, p, _)| p.contains("related"))
            .unwrap();
        // bob → http://ex.org/bob
        assert!(rel.2.contains("bob"), "got: {}", rel.2);
    }

    #[test]
    fn test_triples_to_csv_basic() {
        let mapper = CsvRdfMapper::new("subject", "http://ex.org/");
        let triples = vec![
            (
                "http://ex.org/alice".to_string(),
                "http://ex.org/name".to_string(),
                "Alice".to_string(),
            ),
            (
                "http://ex.org/alice".to_string(),
                "http://ex.org/age".to_string(),
                "30".to_string(),
            ),
        ];
        let csv = mapper.triples_to_csv(&triples);
        assert!(csv.contains("subject"));
        assert!(csv.contains("alice"));
    }

    #[test]
    fn test_triples_to_csv_header_row_present() {
        let mapper = CsvRdfMapper::new("subject", "http://ex.org/");
        let triples = vec![(
            "http://ex.org/s".to_string(),
            "http://ex.org/p".to_string(),
            "o".to_string(),
        )];
        let csv = mapper.triples_to_csv(&triples);
        let first_line = csv.lines().next().unwrap();
        assert!(first_line.contains("subject"));
    }

    #[test]
    fn test_csv_round_trip_multi_row() {
        let mapper = CsvRdfMapper::new("id", "http://ex.org/");
        let csv = "id,name\nhttp://ex.org/a,Alice\nhttp://ex.org/b,Bob\n";
        let triples = mapper.csv_to_triples(csv).unwrap();
        // Two rows × one non-subject column = 2 triples
        assert_eq!(triples.len(), 2);
    }

    // ── JsonLdImporter ───────────────────────────────────────────────────────

    #[test]
    fn test_json_ld_basic_import() {
        let json_ld = r#"[{"@id": "http://ex.org/alice", "http://ex.org/name": "Alice"}]"#;
        let triples = JsonLdImporter::import(json_ld).unwrap();
        assert!(!triples.is_empty());
        assert_eq!(triples[0].0, "http://ex.org/alice");
    }

    #[test]
    fn test_json_ld_type_assertion() {
        let json_ld = r#"[{"@id": "http://ex.org/alice", "@type": "http://ex.org/Person"}]"#;
        let triples = JsonLdImporter::import(json_ld).unwrap();
        let type_triple = triples.iter().find(|(_, p, _)| p.contains("type")).unwrap();
        assert_eq!(type_triple.2, "http://ex.org/Person");
    }

    #[test]
    fn test_json_ld_nested_id() {
        let json_ld = r#"[{"@id": "http://ex.org/alice", "http://ex.org/knows": {"@id": "http://ex.org/bob"}}]"#;
        let triples = JsonLdImporter::import(json_ld).unwrap();
        let knows = triples
            .iter()
            .find(|(_, p, _)| p.contains("knows"))
            .unwrap();
        assert_eq!(knows.2, "http://ex.org/bob");
    }

    #[test]
    fn test_json_ld_graph_import() {
        let json_ld = r#"{"@graph": [{"@id": "http://ex.org/s", "http://ex.org/p": "o"}]}"#;
        let triples = JsonLdImporter::import(json_ld).unwrap();
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_json_ld_invalid_returns_error() {
        let result = JsonLdImporter::import("{bad json");
        assert!(result.is_err());
    }

    // ── NQuadsImporter ───────────────────────────────────────────────────────

    #[test]
    fn test_nquads_basic_import() {
        let nq = "<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> .\n";
        let quads = NQuadsImporter::import(nq).unwrap();
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].0, "http://ex.org/s");
        assert_eq!(quads[0].1, "http://ex.org/p");
        assert_eq!(quads[0].2, "http://ex.org/o");
        assert!(quads[0].3.is_none());
    }

    #[test]
    fn test_nquads_with_graph() {
        let nq = "<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> <http://ex.org/g> .\n";
        let quads = NQuadsImporter::import(nq).unwrap();
        assert_eq!(quads[0].3, Some("http://ex.org/g".to_string()));
    }

    #[test]
    fn test_nquads_skip_comments() {
        let nq = "# This is a comment\n<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> .\n";
        let quads = NQuadsImporter::import(nq).unwrap();
        assert_eq!(quads.len(), 1);
    }

    #[test]
    fn test_nquads_skip_empty_lines() {
        let nq = "\n<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> .\n\n";
        let quads = NQuadsImporter::import(nq).unwrap();
        assert_eq!(quads.len(), 1);
    }

    #[test]
    fn test_nquads_multiple_quads() {
        let nq = "<http://a> <http://b> <http://c> .\n<http://d> <http://e> <http://f> .\n";
        let quads = NQuadsImporter::import(nq).unwrap();
        assert_eq!(quads.len(), 2);
    }

    // ── NQuadsExporter ───────────────────────────────────────────────────────

    #[test]
    fn test_nquads_export_no_graph() {
        let triples = vec![(
            "http://ex.org/s".to_string(),
            "http://ex.org/p".to_string(),
            "http://ex.org/o".to_string(),
        )];
        let out = NQuadsExporter::export(&triples, None);
        assert!(out.contains("<http://ex.org/s>"));
        assert!(out.contains("<http://ex.org/o>"));
        assert!(out.ends_with(".\n"));
    }

    #[test]
    fn test_nquads_export_with_graph() {
        let triples = vec![(
            "http://ex.org/s".to_string(),
            "http://ex.org/p".to_string(),
            "http://ex.org/o".to_string(),
        )];
        let out = NQuadsExporter::export(&triples, Some("http://ex.org/g"));
        assert!(out.contains("<http://ex.org/g>"));
    }

    #[test]
    fn test_nquads_export_blank_node() {
        let triples = vec![(
            "_:b0".to_string(),
            "http://ex.org/p".to_string(),
            "http://ex.org/o".to_string(),
        )];
        let out = NQuadsExporter::export(&triples, None);
        // Blank node should not be wrapped in <>
        assert!(out.contains("_:b0"), "output: {}", out);
    }

    #[test]
    fn test_nquads_export_empty_triples() {
        let triples: Vec<(String, String, String)> = vec![];
        let out = NQuadsExporter::export(&triples, None);
        assert!(out.is_empty());
    }

    #[test]
    fn test_nquads_export_literal_object() {
        let triples = vec![(
            "http://ex.org/s".to_string(),
            "http://ex.org/p".to_string(),
            "\"Alice\"".to_string(), // already a literal
        )];
        let out = NQuadsExporter::export(&triples, None);
        assert!(out.contains("\"Alice\""));
    }
}
