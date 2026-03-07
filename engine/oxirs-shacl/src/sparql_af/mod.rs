//! SHACL Advanced Features — SPARQL-based Targets and Validators
//!
//! This module implements the W3C SHACL-AF specification:
//! <https://www.w3.org/TR/shacl-af/>
//!
//! ## Features
//!
//! - **`sh:SPARQLTarget`** — uses a SPARQL SELECT query to identify focus nodes
//! - **`sh:SPARQLTargetType`** — parameterized SPARQL target type templates
//! - **`sh:SPARQLAskValidator`** — constraint validation via ASK queries
//! - **`sh:SPARQLConstraint`** integration (SELECT-based, existing `SparqlConstraint`
//!   type is enriched with a proper `sh:select` builder API here)
//! - Combined SPARQL + regular constraint validation pipelines

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

pub mod ask_validator;
pub mod sparql_target;
pub mod target_type;

pub use ask_validator::{
    SparqlAskResult, SparqlAskValidator, SparqlAskValidatorBuilder, SparqlAskViolation,
};
pub use sparql_target::{
    ParameterBinding, SparqlAfTarget, SparqlAfTargetResult, SparqlTargetEvaluator, SparqlTargetMock,
};
pub use target_type::{
    SparqlTargetParameter, SparqlTargetType, SparqlTargetTypeInstance, SparqlTargetTypeRegistry,
};

/// Standard SHACL namespace prefix
pub const SHACL_NS: &str = "http://www.w3.org/ns/shacl#";
/// Standard SHACL-AF namespace prefix (same namespace, different types)
pub const SHACL_AF_TARGET_TYPE: &str = "http://www.w3.org/ns/shacl#SPARQLTargetType";
pub const SHACL_AF_SPARQL_TARGET: &str = "http://www.w3.org/ns/shacl#SPARQLTarget";

/// A single row of SPARQL SELECT results, mapping variable names to value strings.
pub type SparqlRow = HashMap<String, String>;

/// Error type for SHACL-AF operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum SparqlAfError {
    #[error("SPARQL query execution error: {0}")]
    QueryExecution(String),

    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    #[error("Invalid parameter value '{value}' for parameter '{param}': {reason}")]
    InvalidParameter {
        param: String,
        value: String,
        reason: String,
    },

    #[error("Target type not found: {0}")]
    TargetTypeNotFound(String),

    #[error("Query building error: {0}")]
    QueryBuild(String),

    #[error("Validator configuration error: {0}")]
    Configuration(String),
}

/// Result type alias for SHACL-AF operations
pub type AfResult<T> = std::result::Result<T, SparqlAfError>;

/// Prefix map for SPARQL queries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PrefixMap(pub HashMap<String, String>);

impl PrefixMap {
    /// Create an empty prefix map
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Add a prefix mapping (builder pattern)
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.0.insert(prefix.into(), iri.into());
        self
    }

    /// Add the default SHACL + RDF prefixes
    pub fn with_shacl_defaults(self) -> Self {
        self.with_prefix("sh", SHACL_NS)
            .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
            .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
    }

    /// Render all prefixes as `PREFIX p: <iri>` declarations
    pub fn render_declarations(&self) -> String {
        self.0
            .iter()
            .map(|(p, iri)| format!("PREFIX {p}: <{iri}>"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Merge another prefix map into this one (other wins on conflict)
    pub fn merge(&mut self, other: &PrefixMap) {
        for (k, v) in &other.0 {
            self.0.insert(k.clone(), v.clone());
        }
    }
}

impl fmt::Display for PrefixMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render_declarations())
    }
}

/// Substitution context for parameterized SPARQL queries
///
/// Holds all parameter bindings that will be injected into the SPARQL query
/// template before execution.
#[derive(Debug, Clone, Default)]
pub struct SubstitutionContext {
    /// Parameter name -> SPARQL term string (e.g., `<http://ex.org/A>` or `"Alice"`)
    pub params: HashMap<String, String>,
    /// The focus node term string (e.g., `<http://ex.org/node1>`)
    pub this_node: Option<String>,
}

impl SubstitutionContext {
    /// Create an empty substitution context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter binding
    pub fn bind(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(name.into(), value.into());
        self
    }

    /// Set the focus node ($this) binding
    pub fn with_this(mut self, this_node: impl Into<String>) -> Self {
        self.this_node = Some(this_node.into());
        self
    }

    /// Apply all substitutions to a query string.
    ///
    /// `$this` is replaced with the focus node.
    /// `$param_name` is replaced with the matching parameter value.
    pub fn apply(&self, query: &str) -> String {
        let mut result = query.to_string();

        // Replace $this first
        if let Some(this) = &self.this_node {
            result = result.replace("$this", this);
        }

        // Replace named parameters
        for (name, value) in &self.params {
            let placeholder = format!("${name}");
            result = result.replace(&placeholder, value);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_map_render() {
        let map = PrefixMap::new()
            .with_prefix("ex", "http://example.org/")
            .with_shacl_defaults();
        let decls = map.render_declarations();
        assert!(decls.contains("PREFIX ex: <http://example.org/>"));
        assert!(decls.contains("PREFIX sh: <http://www.w3.org/ns/shacl#>"));
    }

    #[test]
    fn test_substitution_context_this_replacement() {
        let ctx = SubstitutionContext::new().with_this("<http://example.org/Alice>");
        let query = "SELECT ?this WHERE { $this ?p ?o }";
        let result = ctx.apply(query);
        assert!(result.contains("<http://example.org/Alice>"));
        assert!(!result.contains("$this"));
    }

    #[test]
    fn test_substitution_context_param_replacement() {
        let ctx = SubstitutionContext::new().bind("class", "<http://example.org/Person>");
        let query = "SELECT ?this WHERE { ?this a $class }";
        let result = ctx.apply(query);
        assert!(result.contains("<http://example.org/Person>"));
        assert!(!result.contains("$class"));
    }

    #[test]
    fn test_substitution_context_multiple_params() {
        let ctx = SubstitutionContext::new()
            .with_this("<http://example.org/Bob>")
            .bind("prop", "<http://example.org/age>")
            .bind(
                "minVal",
                "\"18\"^^<http://www.w3.org/2001/XMLSchema#integer>",
            );
        let query = "SELECT ?this WHERE { $this $prop ?v . FILTER(?v >= $minVal) }";
        let result = ctx.apply(query);
        assert!(!result.contains("$this"));
        assert!(!result.contains("$prop"));
        assert!(!result.contains("$minVal"));
        assert!(result.contains("<http://example.org/Bob>"));
        assert!(result.contains("<http://example.org/age>"));
    }

    #[test]
    fn test_prefix_map_merge() {
        let mut base = PrefixMap::new().with_prefix("ex", "http://example.org/");
        let extra = PrefixMap::new().with_prefix("sh", "http://www.w3.org/ns/shacl#");
        base.merge(&extra);
        assert_eq!(base.0.len(), 2);
        assert!(base.0.contains_key("ex"));
        assert!(base.0.contains_key("sh"));
    }

    #[test]
    fn test_prefix_map_display() {
        let map = PrefixMap::new().with_prefix("ex", "http://example.org/");
        let s = map.to_string();
        assert!(s.contains("PREFIX ex: <http://example.org/>"));
    }
}
