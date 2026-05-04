//! Apache Jena parity matrix data types and TOML catalog parser.
//!
//! The [`JenaParityMatrix`] is a map from [`JenaCategory`] to a list of
//! [`JenaEntry`] records, each representing one documented Apache Jena feature
//! and its current implementation status in OxiRS.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// High-level grouping that mirrors Apache Jena documentation chapters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum JenaCategory {
    /// SPARQL query, update, federation, extensions (ARQ).
    SparqlEngine,
    /// RDF serialization formats (Turtle, N-Triples, N-Quads, TriG, RDF/XML, JSON-LD, Thrift).
    RdfFormats,
    /// Storage backends (TDB2, SDB, in-memory, transactional).
    StorageBackends,
    /// Inference and reasoning (OWL, RDFS, built-in rules).
    Inference,
    /// Validation (SHACL, SHACLC).
    Validation,
    /// Spatial and GeoSPARQL.
    Spatial,
    /// HTTP server and SPARQL protocol (Fuseki).
    HttpServer,
    /// Full-text search integration (JenaText / Lucene / Elasticsearch).
    TextSearch,
    /// Dataset assembly and configuration (Jena Assembler).
    Assembler,
    /// Graph API and model operations (Model, Dataset, Graph interfaces).
    GraphApi,
    /// Security and access control.
    Security,
    /// Utilities and tooling (CLI, riot, rsparql, etc.).
    Tooling,
    /// Any feature category that does not fit the named variants.
    Other,
}

/// Implementation status of a single Apache Jena feature in OxiRS.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum JenaStatus {
    /// The feature is fully implemented and tested.
    Implemented,
    /// The feature is partially implemented; known gaps exist.
    Partial,
    /// The feature is not yet implemented.
    Missing,
    /// Intentionally not planned (e.g., JVM-only or SQL-backed features).
    OutOfScope,
}

/// One row in the parity matrix, describing a single Apache Jena feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JenaEntry {
    /// Human-readable feature name.
    pub name: String,
    /// The Jena Maven artifact that provides this feature, e.g. `"jena-arq"`.
    pub jena_component: String,
    /// The primary Jena class or API entry point.
    pub jena_class_or_api: String,
    /// Rust module path inside OxiRS that implements the feature (if any).
    pub oxirs_module: Option<String>,
    /// Current implementation status in OxiRS.
    pub status: JenaStatus,
    /// One-sentence clarification or scope note.
    pub notes: String,
}

/// The full parity matrix: a mapping from Jena feature category to feature entries.
///
/// Populate with [`parse_catalog`] or [`super::load_catalog`].
pub type JenaParityMatrix = HashMap<JenaCategory, Vec<JenaEntry>>;

/// Parse a TOML catalog string into a [`JenaParityMatrix`].
///
/// The TOML structure uses snake_case section names matching
/// [`JenaCategory`] variants (e.g. `[[sparql_engine]]`).
///
/// # Errors
///
/// Returns an error if the TOML is malformed or contains unexpected types.
pub fn parse_catalog(toml_str: &str) -> Result<JenaParityMatrix, Box<dyn std::error::Error>> {
    // ── Internal deserialization helpers ────────────────────────────────

    #[derive(Deserialize)]
    struct RawEntry {
        name: String,
        jena_component: String,
        jena_class_or_api: String,
        oxirs_module: Option<String>,
        status: String,
        notes: String,
    }

    #[derive(Deserialize)]
    struct CatalogFile {
        sparql_engine: Option<Vec<RawEntry>>,
        rdf_formats: Option<Vec<RawEntry>>,
        storage_backends: Option<Vec<RawEntry>>,
        inference: Option<Vec<RawEntry>>,
        validation: Option<Vec<RawEntry>>,
        spatial: Option<Vec<RawEntry>>,
        http_server: Option<Vec<RawEntry>>,
        text_search: Option<Vec<RawEntry>>,
        assembler: Option<Vec<RawEntry>>,
        graph_api: Option<Vec<RawEntry>>,
        security: Option<Vec<RawEntry>>,
        tooling: Option<Vec<RawEntry>>,
        other: Option<Vec<RawEntry>>,
    }

    let file: CatalogFile = toml::from_str(toml_str)?;
    let mut matrix = JenaParityMatrix::new();

    fn parse_status(s: &str) -> JenaStatus {
        match s {
            "implemented" => JenaStatus::Implemented,
            "partial" => JenaStatus::Partial,
            "out_of_scope" => JenaStatus::OutOfScope,
            _ => JenaStatus::Missing,
        }
    }

    fn convert(raw: Vec<RawEntry>) -> Vec<JenaEntry> {
        raw.into_iter()
            .map(|r| JenaEntry {
                name: r.name,
                jena_component: r.jena_component,
                jena_class_or_api: r.jena_class_or_api,
                oxirs_module: r.oxirs_module,
                status: parse_status(&r.status),
                notes: r.notes,
            })
            .collect()
    }

    if let Some(entries) = file.sparql_engine {
        matrix.insert(JenaCategory::SparqlEngine, convert(entries));
    }
    if let Some(entries) = file.rdf_formats {
        matrix.insert(JenaCategory::RdfFormats, convert(entries));
    }
    if let Some(entries) = file.storage_backends {
        matrix.insert(JenaCategory::StorageBackends, convert(entries));
    }
    if let Some(entries) = file.inference {
        matrix.insert(JenaCategory::Inference, convert(entries));
    }
    if let Some(entries) = file.validation {
        matrix.insert(JenaCategory::Validation, convert(entries));
    }
    if let Some(entries) = file.spatial {
        matrix.insert(JenaCategory::Spatial, convert(entries));
    }
    if let Some(entries) = file.http_server {
        matrix.insert(JenaCategory::HttpServer, convert(entries));
    }
    if let Some(entries) = file.text_search {
        matrix.insert(JenaCategory::TextSearch, convert(entries));
    }
    if let Some(entries) = file.assembler {
        matrix.insert(JenaCategory::Assembler, convert(entries));
    }
    if let Some(entries) = file.graph_api {
        matrix.insert(JenaCategory::GraphApi, convert(entries));
    }
    if let Some(entries) = file.security {
        matrix.insert(JenaCategory::Security, convert(entries));
    }
    if let Some(entries) = file.tooling {
        matrix.insert(JenaCategory::Tooling, convert(entries));
    }
    if let Some(entries) = file.other {
        matrix.insert(JenaCategory::Other, convert(entries));
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_TOML: &str = r#"
[[sparql_engine]]
name = "SPARQL 1.1 SELECT"
jena_component = "jena-arq"
jena_class_or_api = "QueryExecutionFactory.create()"
oxirs_module = "oxirs_arq::executor"
status = "implemented"
notes = "Full SPARQL 1.1 SELECT support."

[[storage_backends]]
name = "SDB (SQL-backed store)"
jena_component = "jena-sdb"
jena_class_or_api = "SDBFactory"
status = "out_of_scope"
notes = "Not planned — Pure Rust TDB2 equivalent preferred."
"#;

    #[test]
    fn test_parse_minimal_catalog() {
        let matrix = parse_catalog(MINIMAL_TOML).expect("should parse");
        assert_eq!(matrix.len(), 2);
        let sparql = matrix
            .get(&JenaCategory::SparqlEngine)
            .expect("sparql_engine");
        assert_eq!(sparql.len(), 1);
        assert_eq!(sparql[0].status, JenaStatus::Implemented);
        assert!(sparql[0].oxirs_module.is_some());
    }

    #[test]
    fn test_parse_out_of_scope_status() {
        let matrix = parse_catalog(MINIMAL_TOML).expect("should parse");
        let backends = matrix
            .get(&JenaCategory::StorageBackends)
            .expect("storage_backends");
        assert_eq!(backends[0].status, JenaStatus::OutOfScope);
        assert!(backends[0].oxirs_module.is_none());
    }

    #[test]
    fn test_unknown_status_becomes_missing() {
        let toml = r#"
[[sparql_engine]]
name = "X"
jena_component = "jena-arq"
jena_class_or_api = "SomeClass"
status = "unknown_value"
notes = "test"
"#;
        let matrix = parse_catalog(toml).expect("should parse");
        let entries = matrix
            .get(&JenaCategory::SparqlEngine)
            .expect("sparql_engine");
        assert_eq!(entries[0].status, JenaStatus::Missing);
    }

    #[test]
    fn test_invalid_toml_returns_error() {
        let result = parse_catalog("this is not valid toml ][}{");
        assert!(result.is_err());
    }

    #[test]
    fn test_jena_entry_fields_populated() {
        let matrix = parse_catalog(MINIMAL_TOML).expect("should parse");
        let sparql = matrix
            .get(&JenaCategory::SparqlEngine)
            .expect("sparql_engine");
        let entry = &sparql[0];
        assert_eq!(entry.jena_component, "jena-arq");
        assert_eq!(entry.jena_class_or_api, "QueryExecutionFactory.create()");
        assert!(!entry.notes.is_empty());
    }
}
