//! RDF data validation utilities
//!
//! This module provides utilities for validating RDF triples and quads,
//! checking for common issues, and ensuring data quality.
//!
//! # Examples
//!
//! ```rust
//! use oxirs_ttl::toolkit::rdf_validator::*;
//! use oxirs_core::model::{NamedNode, Triple, Literal};
//!
//! let triple = Triple::new(
//!     NamedNode::new("http://example.org/subject")?,
//!     NamedNode::new("http://example.org/predicate")?,
//!     Literal::new("value")
//! );
//!
//! // Validate the triple
//! let result = validate_triple(&triple);
//! assert!(result.is_valid);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxirs_core::model::{BlankNode, Literal, NamedNode, Quad, Subject, Triple};
use oxirs_core::RdfTerm;
use std::collections::{HashMap, HashSet};

/// Validation result with issues
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the data is valid
    pub is_valid: bool,
    /// List of validation issues found
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            issues: Vec::new(),
        }
    }

    /// Create an invalid result with issues
    pub fn invalid(issues: Vec<ValidationIssue>) -> Self {
        Self {
            is_valid: false,
            issues,
        }
    }

    /// Add an issue to the result
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        self.is_valid = false;
        self.issues.push(issue);
    }

    /// Check if the result has warnings (non-critical issues)
    pub fn has_warnings(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Warning)
    }

    /// Check if the result has errors
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Error)
    }
}

/// Severity level of a validation issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Informational message
    Info,
    /// Warning (non-critical issue)
    Warning,
    /// Error (critical issue)
    Error,
}

/// A validation issue found during validation
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue
    pub severity: Severity,
    /// Description of the issue
    pub message: String,
    /// Optional suggestion for fixing
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    /// Create a new error issue
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create a new warning issue
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create a new info issue
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            message: message.into(),
            suggestion: None,
        }
    }

    /// Add a suggestion to this issue
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Validate a triple for common issues
pub fn validate_triple(triple: &Triple) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Check subject
    match triple.subject() {
        Subject::NamedNode(node) => {
            if let Some(issue) = validate_named_node(node, "subject") {
                result.add_issue(issue);
            }
        }
        Subject::BlankNode(bnode) => {
            if let Some(issue) = validate_blank_node(bnode, "subject") {
                result.add_issue(issue);
            }
        }
        _ => {}
    }

    // Check predicate (predicates are always NamedNodes or Variables in RDF/N3)
    if triple.predicate().is_named_node() {
        // Extract the IRI from the predicate for validation
        let pred_iri = triple.predicate().to_string();
        if let Ok(node) = NamedNode::new(&pred_iri) {
            if let Some(issue) = validate_named_node(&node, "predicate") {
                result.add_issue(issue);
            }
        }
    }

    // Check object
    match triple.object() {
        oxirs_core::model::Object::NamedNode(node) => {
            if let Some(issue) = validate_named_node(node, "object") {
                result.add_issue(issue);
            }
        }
        oxirs_core::model::Object::BlankNode(bnode) => {
            if let Some(issue) = validate_blank_node(bnode, "object") {
                result.add_issue(issue);
            }
        }
        oxirs_core::model::Object::Literal(lit) => {
            if let Some(issue) = validate_literal(lit) {
                result.add_issue(issue);
            }
        }
        _ => {}
    }

    result
}

/// Validate a quad for common issues
pub fn validate_quad(quad: &Quad) -> ValidationResult {
    let mut result = validate_triple(&Triple::new(
        quad.subject().clone(),
        quad.predicate().clone(),
        quad.object().clone(),
    ));

    // Check graph name
    if let oxirs_core::model::GraphName::NamedNode(node) = quad.graph_name() {
        if let Some(issue) = validate_named_node(node, "graph name") {
            result.add_issue(issue);
        }
    }

    result
}

/// Validate a named node
fn validate_named_node(node: &NamedNode, context: &str) -> Option<ValidationIssue> {
    let iri = node.as_str();

    // Check for empty IRI
    if iri.is_empty() {
        return Some(ValidationIssue::error(format!("Empty IRI in {}", context)));
    }

    // Check for whitespace
    if iri.contains(|c: char| c.is_whitespace()) {
        return Some(
            ValidationIssue::error(format!("IRI contains whitespace in {}", context))
                .with_suggestion("Remove whitespace from IRI"),
        );
    }

    // Check for common IRI issues
    if !iri.starts_with("http://")
        && !iri.starts_with("https://")
        && !iri.starts_with("urn:")
        && !iri.starts_with("file://")
    {
        return Some(
            ValidationIssue::warning(format!("Unusual IRI scheme in {}: {}", context, iri))
                .with_suggestion("Consider using http://, https://, or urn: schemes"),
        );
    }

    None
}

/// Validate a blank node
fn validate_blank_node(bnode: &BlankNode, context: &str) -> Option<ValidationIssue> {
    let id = bnode.as_str();

    // Check for empty blank node ID
    if id.is_empty() {
        return Some(ValidationIssue::error(format!(
            "Empty blank node ID in {}",
            context
        )));
    }

    // Check for whitespace in blank node ID
    if id.contains(|c: char| c.is_whitespace()) {
        return Some(
            ValidationIssue::error(format!("Blank node ID contains whitespace in {}", context))
                .with_suggestion("Remove whitespace from blank node ID"),
        );
    }

    None
}

/// Validate a literal
fn validate_literal(literal: &Literal) -> Option<ValidationIssue> {
    let value = literal.value();

    // Check for extremely long literals
    if value.len() > 1_000_000 {
        return Some(
            ValidationIssue::warning("Literal value is very large (>1MB)")
                .with_suggestion("Consider storing large values externally"),
        );
    }

    // Check language tag format if present
    if let Some(lang) = literal.language() {
        if lang.is_empty() {
            return Some(ValidationIssue::error("Empty language tag"));
        }
        // Basic language tag validation
        if !lang.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return Some(
                ValidationIssue::error(format!("Invalid language tag: {}", lang)).with_suggestion(
                    "Language tags should contain only alphanumeric characters and hyphens",
                ),
            );
        }
    }

    None
}

/// Check for duplicate triples in a collection
pub fn check_duplicates(triples: &[Triple]) -> Vec<Triple> {
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();

    for triple in triples {
        let key = (
            triple.subject().to_string(),
            triple.predicate().to_string(),
            triple.object().to_string(),
        );
        if !seen.insert(key) {
            duplicates.push(triple.clone());
        }
    }

    duplicates
}

/// Check for orphaned blank nodes (blank nodes that appear only once)
pub fn check_orphaned_blank_nodes(triples: &[Triple]) -> Vec<String> {
    let mut bnode_counts: HashMap<String, usize> = HashMap::new();

    for triple in triples {
        if let Subject::BlankNode(bnode) = triple.subject() {
            *bnode_counts.entry(bnode.as_str().to_string()).or_insert(0) += 1;
        }
        if let oxirs_core::model::Object::BlankNode(bnode) = triple.object() {
            *bnode_counts.entry(bnode.as_str().to_string()).or_insert(0) += 1;
        }
    }

    bnode_counts
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(id, _)| id)
        .collect()
}

/// Statistics about an RDF dataset
#[derive(Debug, Clone, Default)]
pub struct DatasetStats {
    /// Total number of triples
    pub triple_count: usize,
    /// Number of unique subjects
    pub unique_subjects: usize,
    /// Number of unique predicates
    pub unique_predicates: usize,
    /// Number of unique objects
    pub unique_objects: usize,
    /// Number of blank nodes
    pub blank_node_count: usize,
    /// Number of literals
    pub literal_count: usize,
    /// Number of language-tagged literals
    pub language_tagged_count: usize,
    /// Number of typed literals
    pub typed_literal_count: usize,
}

/// Compute statistics for a collection of triples
pub fn compute_stats(triples: &[Triple]) -> DatasetStats {
    let mut stats = DatasetStats {
        triple_count: triples.len(),
        ..Default::default()
    };

    let mut subjects = HashSet::new();
    let mut predicates = HashSet::new();
    let mut objects = HashSet::new();

    for triple in triples {
        subjects.insert(triple.subject().to_string());
        predicates.insert(triple.predicate().to_string());
        objects.insert(triple.object().to_string());

        if matches!(triple.subject(), Subject::BlankNode(_)) {
            stats.blank_node_count += 1;
        }

        if let oxirs_core::model::Object::Literal(lit) = triple.object() {
            stats.literal_count += 1;
            if lit.language().is_some() {
                stats.language_tagged_count += 1;
            }
            if lit.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                stats.typed_literal_count += 1;
            }
        }
    }

    stats.unique_subjects = subjects.len();
    stats.unique_predicates = predicates.len();
    stats.unique_objects = objects.len();

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_triple_valid() {
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("value"),
        );

        let result = validate_triple(&triple);
        assert!(result.is_valid);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_validate_literal() {
        let lit = Literal::new_language_tagged_literal("Hello", "en").unwrap();
        let issue = validate_literal(&lit);
        assert!(issue.is_none());
    }

    #[test]
    fn test_check_duplicates() {
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("value"),
        );
        let triple2 = triple1.clone();

        let duplicates = check_duplicates(&[triple1, triple2]);
        assert_eq!(duplicates.len(), 1);
    }

    #[test]
    fn test_compute_stats() {
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/s1").unwrap(),
                NamedNode::new("http://example.org/p").unwrap(),
                Literal::new("value1"),
            ),
            Triple::new(
                NamedNode::new("http://example.org/s2").unwrap(),
                NamedNode::new("http://example.org/p").unwrap(),
                Literal::new("value2"),
            ),
        ];

        let stats = compute_stats(&triples);
        assert_eq!(stats.triple_count, 2);
        assert_eq!(stats.unique_subjects, 2);
        assert_eq!(stats.unique_predicates, 1);
        assert_eq!(stats.literal_count, 2);
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::valid();
        assert!(result.is_valid);

        result.add_issue(ValidationIssue::warning("Test warning"));
        assert!(!result.is_valid);
        assert!(result.has_warnings());
        assert!(!result.has_errors());

        result.add_issue(ValidationIssue::error("Test error"));
        assert!(result.has_errors());
    }
}
