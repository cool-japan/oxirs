//! Core data types for SPARQL CONSTRUCT query support.
//!
//! Defines the configuration, parsed-query, template, and statistics types
//! shared by the CONSTRUCT engine, parser, and serializer.

use super::GraphPattern;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for CONSTRUCT query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructConfig {
    /// Whether to deduplicate output triples (default: true).
    pub deduplicate: bool,
    /// Maximum number of output triples (None = unlimited).
    pub max_triples: Option<usize>,
    /// Whether to track construction statistics (default: true).
    pub collect_stats: bool,
    /// Blank node prefix for generated blank nodes.
    pub blank_node_prefix: String,
}

impl Default for ConstructConfig {
    fn default() -> Self {
        Self {
            deduplicate: true,
            max_triples: None,
            collect_stats: true,
            blank_node_prefix: "b".to_string(),
        }
    }
}

// ─────────────────────────────────────────────
// Template types
// ─────────────────────────────────────────────

/// A parsed CONSTRUCT query with template and WHERE clause.
#[derive(Debug, Clone)]
pub struct ConstructQuery {
    /// Template triple patterns to instantiate per solution.
    pub template: Vec<TemplateTriple>,
    /// WHERE clause graph patterns.
    pub(crate) where_patterns: Vec<GraphPattern>,
    /// PREFIX declarations (prefix -> IRI).
    pub prefixes: HashMap<String, String>,
    /// LIMIT modifier.
    pub limit: Option<usize>,
    /// OFFSET modifier.
    pub offset: Option<usize>,
}

/// A triple pattern in the CONSTRUCT template.
#[derive(Debug, Clone)]
pub struct TemplateTriple {
    /// Subject: variable, IRI, or blank node.
    pub subject: TemplateTerm,
    /// Predicate: variable or IRI.
    pub predicate: TemplateTerm,
    /// Object: variable, IRI, blank node, or literal.
    pub object: TemplateTerm,
}

/// A term in a CONSTRUCT template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateTerm {
    /// A SPARQL variable (?name).
    Variable(String),
    /// An IRI reference.
    Iri(String),
    /// A blank node identifier.
    BlankNode(String),
    /// A plain literal.
    Literal(String),
    /// A language-tagged literal.
    LangLiteral { value: String, lang: String },
    /// A datatype-tagged literal.
    TypedLiteral { value: String, datatype: String },
}

impl TemplateTerm {
    /// Instantiate this template term using the given solution mapping.
    ///
    /// Returns None if a variable is unbound (the entire triple is skipped
    /// per SPARQL 1.1 spec section 16.2).
    pub(crate) fn instantiate(
        &self,
        bindings: &HashMap<String, String>,
        blank_scope: &mut HashMap<String, String>,
        blank_counter: &mut u64,
        prefix: &str,
    ) -> Option<String> {
        match self {
            TemplateTerm::Variable(name) => bindings.get(name).cloned(),
            TemplateTerm::Iri(iri) => Some(iri.clone()),
            TemplateTerm::BlankNode(label) => {
                // Scoped blank nodes: each solution mapping gets unique blank node IDs
                let entry = blank_scope.entry(label.clone()).or_insert_with(|| {
                    *blank_counter += 1;
                    format!("_:{}{}", prefix, blank_counter)
                });
                Some(entry.clone())
            }
            TemplateTerm::Literal(val) => Some(format!("\"{}\"", val)),
            TemplateTerm::LangLiteral { value, lang } => Some(format!("\"{}\"@{}", value, lang)),
            TemplateTerm::TypedLiteral { value, datatype } => {
                Some(format!("\"{}\"^^<{}>", value, datatype))
            }
        }
    }
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics from a CONSTRUCT query execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstructStats {
    /// Number of solution mappings from WHERE clause.
    pub solution_count: usize,
    /// Number of template triples per solution.
    pub template_triple_count: usize,
    /// Total triples before deduplication.
    pub raw_triple_count: usize,
    /// Total triples after deduplication.
    pub deduped_triple_count: usize,
    /// Number of triples skipped due to unbound variables.
    pub skipped_unbound: usize,
    /// Number of blank nodes generated.
    pub blank_nodes_generated: u64,
}
