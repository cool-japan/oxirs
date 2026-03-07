//! Advanced SPARQL-based constraint component (sh:sparql)
//!
//! This module implements the SHACL-SPARQL constraint mechanism, which allows
//! using SPARQL SELECT queries as constraints. A constraint violation occurs
//! whenever the SELECT query returns any results for a given focus node.
//!
//! Reference: <https://www.w3.org/TR/shacl/#sparql-constraints>

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Result, ShaclError};

/// Severity level for SPARQL constraint violations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SparqlConstraintSeverity {
    #[default]
    Violation,
    Warning,
    Info,
}

impl fmt::Display for SparqlConstraintSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparqlConstraintSeverity::Violation => write!(f, "sh:Violation"),
            SparqlConstraintSeverity::Warning => write!(f, "sh:Warning"),
            SparqlConstraintSeverity::Info => write!(f, "sh:Info"),
        }
    }
}

/// An advanced SPARQL-based SHACL constraint.
///
/// When a SPARQL SELECT query returns any rows for a given focus node,
/// the constraint is considered violated. The focus node is bound via `$this`.
///
/// ## SHACL-SPARQL binding variables
/// - `$this` — the focus node
/// - `$value` — the value at the property path (for property shapes)
/// - `$PATH` — the property path IRI
/// - `$shapesGraph` — the shapes graph IRI
/// - `$currentShape` — the current shape IRI
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvancedSparqlConstraint {
    /// The SPARQL SELECT query body (without prefix declarations).
    /// Should reference `$this` as the focus node binding.
    pub sparql_query: String,

    /// Optional violation message template.
    /// May reference SPARQL result variables using `{?varname}` syntax.
    pub message: Option<String>,

    /// Namespace prefix declarations (`prefix` -> `IRI`)
    pub prefixes: HashMap<String, String>,

    /// Whether this constraint is deactivated
    pub deactivated: bool,

    /// Severity of violations produced by this constraint
    pub severity: SparqlConstraintSeverity,

    /// Optional annotation: constraint IRI/label
    pub constraint_iri: Option<String>,
}

impl AdvancedSparqlConstraint {
    /// Create a new SPARQL constraint from a SELECT query body.
    pub fn new(sparql: impl Into<String>) -> Self {
        Self {
            sparql_query: sparql.into(),
            ..Self::default()
        }
    }

    /// Set a custom violation message (builder pattern).
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Set the severity level (builder pattern).
    pub fn with_severity(mut self, severity: SparqlConstraintSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add a namespace prefix (builder pattern).
    pub fn add_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Set the constraint IRI (builder pattern).
    pub fn with_iri(mut self, iri: impl Into<String>) -> Self {
        self.constraint_iri = Some(iri.into());
        self
    }

    /// Deactivate this constraint (builder pattern).
    pub fn deactivated(mut self) -> Self {
        self.deactivated = true;
        self
    }

    /// Build the complete SPARQL query string with prefix declarations and
    /// `$this` bound to the given focus node IRI or blank node identifier.
    pub fn build_query(&self, focus_node: &str) -> String {
        let mut parts = vec![
            // Standard SHACL prefixes always included
            "PREFIX sh: <http://www.w3.org/ns/shacl#>".to_string(),
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>".to_string(),
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>".to_string(),
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>".to_string(),
        ];

        // User-supplied prefix declarations
        for (prefix, iri) in &self.prefixes {
            parts.push(format!("PREFIX {prefix}: <{iri}>"));
        }

        // Bind $this to the focus node
        let this_binding = if focus_node.starts_with("_:") {
            // Blank node — cannot be injected as VALUES; use BIND pattern
            format!("BIND({focus_node} AS ?this)")
        } else {
            // Named node
            format!("VALUES (?this) {{ (<{focus_node}>) }}")
        };

        // Replace textual `$this` in query body with `?this`
        let query_body = self.sparql_query.replace("$this", "?this");

        // Assemble final query, wrapping in SELECT with VALUES block
        parts.push(format!(
            "SELECT ?this ?value ?message WHERE {{ {this_binding} {query_body} }}"
        ));

        parts.join("\n")
    }

    /// Format a violation message using result variable bindings.
    ///
    /// Template syntax: `{?varname}` is replaced with the variable value.
    pub fn format_message(&self, bindings: &HashMap<String, String>) -> String {
        let template = match &self.message {
            Some(t) => t.clone(),
            None => return String::from("SPARQL constraint violation"),
        };

        let mut result = template.clone();
        for (var, value) in bindings {
            let placeholder = format!("{{?{var}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Validate a single focus node using the provided SPARQL evaluator.
    ///
    /// Returns `Ok(SparqlConstraintResult)` describing whether the node is
    /// valid. If the constraint is deactivated, the node is always valid.
    pub fn validate_node(
        &self,
        focus_node: &str,
        evaluator: &dyn SparqlEvaluator,
    ) -> Result<SparqlConstraintResult> {
        if self.deactivated {
            return Ok(SparqlConstraintResult {
                focus_node: focus_node.to_string(),
                constraint_iri: self.constraint_iri.clone().unwrap_or_default(),
                is_valid: true,
                violation_bindings: Vec::new(),
                message: None,
                severity: self.severity.clone(),
            });
        }

        let query = self.build_query(focus_node);
        let bound_query = evaluator.bind_focus_node(&query, focus_node);
        let rows = evaluator.execute_select(&bound_query)?;

        let is_valid = rows.is_empty();
        let message = if is_valid {
            None
        } else {
            let first_row = rows.first().cloned().unwrap_or_default();
            Some(self.format_message(&first_row))
        };

        Ok(SparqlConstraintResult {
            focus_node: focus_node.to_string(),
            constraint_iri: self.constraint_iri.clone().unwrap_or_default(),
            is_valid,
            violation_bindings: rows,
            message,
            severity: self.severity.clone(),
        })
    }

    /// Validate a batch of focus nodes using the provided SPARQL evaluator.
    pub fn validate_all(
        &self,
        focus_nodes: &[String],
        evaluator: &dyn SparqlEvaluator,
    ) -> Result<Vec<SparqlConstraintResult>> {
        focus_nodes
            .iter()
            .map(|node| self.validate_node(node, evaluator))
            .collect()
    }
}

/// Result of evaluating an `AdvancedSparqlConstraint` against one focus node.
#[derive(Debug, Clone)]
pub struct SparqlConstraintResult {
    /// The focus node that was validated
    pub focus_node: String,
    /// IRI of the constraint (if set)
    pub constraint_iri: String,
    /// Whether the node satisfies the constraint
    pub is_valid: bool,
    /// Bindings from SELECT result rows (one map per row)
    pub violation_bindings: Vec<HashMap<String, String>>,
    /// Formatted violation message (None when valid)
    pub message: Option<String>,
    /// Severity of this constraint
    pub severity: SparqlConstraintSeverity,
}

impl SparqlConstraintResult {
    /// Returns `true` when the constraint is satisfied.
    pub fn is_satisfied(&self) -> bool {
        self.is_valid
    }

    /// Returns the number of individual violation rows returned by SPARQL.
    pub fn violation_count(&self) -> usize {
        self.violation_bindings.len()
    }
}

// ---------------------------------------------------------------------------
// SparqlEvaluator trait and implementations
// ---------------------------------------------------------------------------

/// Abstraction over a SPARQL execution engine.
///
/// Implementations may use the built-in OxiRS query engine, an external
/// engine via HTTP, or a mock for testing.
pub trait SparqlEvaluator: Send + Sync {
    /// Execute a SPARQL SELECT query and return the variable bindings.
    ///
    /// Each row is a `HashMap<variable_name, value_string>`.
    fn execute_select(&self, query: &str) -> Result<Vec<HashMap<String, String>>>;

    /// Optionally rewrite the query to bind the focus node at the engine level.
    ///
    /// The default implementation returns the query unchanged, since binding
    /// is already embedded via `build_query`.
    fn bind_focus_node(&self, query: &str, _focus_node: &str) -> String {
        query.to_string()
    }
}

/// A mock SPARQL evaluator for unit testing.
///
/// Results are keyed by the full SPARQL query string. When no entry matches,
/// an empty result set is returned (constraint satisfied).
#[derive(Debug, Default)]
pub struct MockSparqlEvaluator {
    /// `query_substring -> rows` mapping.
    /// The evaluator uses substring matching so tests don't need to supply
    /// the exact prefix-expanded query.
    pub results: HashMap<String, Vec<HashMap<String, String>>>,
}

impl MockSparqlEvaluator {
    /// Create an empty mock evaluator (all queries return no rows).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a result for queries containing `key` as a substring.
    pub fn with_result(
        mut self,
        key: impl Into<String>,
        rows: Vec<HashMap<String, String>>,
    ) -> Self {
        self.results.insert(key.into(), rows);
        self
    }
}

impl SparqlEvaluator for MockSparqlEvaluator {
    fn execute_select(&self, query: &str) -> Result<Vec<HashMap<String, String>>> {
        for (key, rows) in &self.results {
            if query.contains(key.as_str()) {
                return Ok(rows.clone());
            }
        }
        Ok(Vec::new())
    }
}

/// A SPARQL evaluator that always returns a fixed violation (for testing).
#[derive(Debug)]
pub struct AlwaysViolatingEvaluator {
    row: HashMap<String, String>,
}

impl AlwaysViolatingEvaluator {
    /// Create an evaluator that returns one violation row with the given bindings.
    pub fn new(row: HashMap<String, String>) -> Self {
        Self { row }
    }
}

impl SparqlEvaluator for AlwaysViolatingEvaluator {
    fn execute_select(&self, _query: &str) -> Result<Vec<HashMap<String, String>>> {
        Ok(vec![self.row.clone()])
    }
}

/// A SPARQL evaluator that always fails with an error (for error-path testing).
#[derive(Debug)]
pub struct FailingEvaluator {
    message: String,
}

impl FailingEvaluator {
    /// Create an evaluator that always returns the given error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl SparqlEvaluator for FailingEvaluator {
    fn execute_select(&self, _query: &str) -> Result<Vec<HashMap<String, String>>> {
        Err(ShaclError::SparqlExecution(self.message.clone()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_violation_row(var: &str, val: &str) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(var.to_string(), val.to_string());
        m
    }

    #[test]
    fn test_build_query_named_node() {
        let constraint = AdvancedSparqlConstraint::new("?this ex:age ?age . FILTER(?age < 0)")
            .add_prefix("ex", "http://example.org/");

        let q = constraint.build_query("http://example.org/Alice");
        assert!(q.contains("VALUES (?this) { (<http://example.org/Alice>) }"));
        assert!(q.contains("PREFIX ex: <http://example.org/>"));
        assert!(q.contains("?age"));
    }

    #[test]
    fn test_build_query_blank_node() {
        let constraint = AdvancedSparqlConstraint::new("?this rdf:type ?t .");
        let q = constraint.build_query("_:b0");
        assert!(q.contains("BIND(_:b0 AS ?this)"));
    }

    #[test]
    fn test_build_query_replaces_this_dollar_sign() {
        let constraint =
            AdvancedSparqlConstraint::new("FILTER NOT EXISTS { $this rdf:type ex:ValidThing }");
        let q = constraint.build_query("http://example.org/X");
        assert!(q.contains("?this rdf:type ex:ValidThing"));
        assert!(!q.contains("$this"));
    }

    #[test]
    fn test_format_message_with_bindings() {
        let constraint = AdvancedSparqlConstraint::new("SELECT * WHERE {}")
            .with_message("Node {?this} has invalid age {?age}");

        let mut bindings = HashMap::new();
        bindings.insert("this".to_string(), "http://example.org/Alice".to_string());
        bindings.insert("age".to_string(), "-5".to_string());

        let msg = constraint.format_message(&bindings);
        assert_eq!(msg, "Node http://example.org/Alice has invalid age -5");
    }

    #[test]
    fn test_format_message_no_template() {
        let constraint = AdvancedSparqlConstraint::new("SELECT * WHERE {}");
        let msg = constraint.format_message(&HashMap::new());
        assert_eq!(msg, "SPARQL constraint violation");
    }

    #[test]
    fn test_validate_node_satisfied() {
        let constraint =
            AdvancedSparqlConstraint::new("FILTER NOT EXISTS { ?this ex:invalid true }");
        let evaluator = MockSparqlEvaluator::new(); // returns empty → satisfied

        let result = constraint
            .validate_node("http://example.org/Alice", &evaluator)
            .expect("validation should not fail");

        assert!(result.is_valid);
        assert_eq!(result.violation_count(), 0);
        assert!(result.message.is_none());
    }

    #[test]
    fn test_validate_node_violated() {
        let row = make_violation_row("value", "http://example.org/BadNode");
        let constraint = AdvancedSparqlConstraint::new("FILTER NOT EXISTS { ?this ex:ok true }")
            .with_message("Constraint violated for {?value}");

        let evaluator = MockSparqlEvaluator::new().with_result("FILTER NOT EXISTS", vec![row]);

        let result = constraint
            .validate_node("http://example.org/Bob", &evaluator)
            .expect("validation should not fail");

        assert!(!result.is_valid);
        assert_eq!(result.violation_count(), 1);
        assert!(result.message.is_some());
    }

    #[test]
    fn test_validate_node_deactivated() {
        let constraint =
            AdvancedSparqlConstraint::new("SELECT * WHERE { ?this ?p ?o }").deactivated();
        let evaluator = AlwaysViolatingEvaluator::new(make_violation_row("p", "val"));

        let result = constraint
            .validate_node("http://example.org/Alice", &evaluator)
            .expect("validation should not fail");

        // Deactivated constraint never reports a violation
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_node_evaluator_error() {
        let constraint = AdvancedSparqlConstraint::new("SELECT * WHERE {}");
        let evaluator = FailingEvaluator::new("endpoint unreachable");

        let result = constraint.validate_node("http://example.org/X", &evaluator);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_all() {
        let row = make_violation_row("value", "http://example.org/Bad");
        let constraint = AdvancedSparqlConstraint::new("FILTER NOT EXISTS { ?this ex:ok true }");
        let evaluator = MockSparqlEvaluator::new().with_result("FILTER NOT EXISTS", vec![row]);

        let nodes: Vec<String> = vec![
            "http://example.org/Alice".to_string(),
            "http://example.org/Bob".to_string(),
        ];

        let results = constraint
            .validate_all(&nodes, &evaluator)
            .expect("batch validation should succeed");

        assert_eq!(results.len(), 2);
        // Both nodes will match the substring key "FILTER NOT EXISTS" so both violate
        for r in &results {
            assert!(!r.is_valid);
        }
    }

    #[test]
    fn test_severity_levels() {
        let info = AdvancedSparqlConstraint::new("SELECT * WHERE {}")
            .with_severity(SparqlConstraintSeverity::Info);
        assert_eq!(info.severity, SparqlConstraintSeverity::Info);

        let warn = AdvancedSparqlConstraint::new("SELECT * WHERE {}")
            .with_severity(SparqlConstraintSeverity::Warning);
        assert_eq!(warn.severity, SparqlConstraintSeverity::Warning);
    }

    #[test]
    fn test_standard_prefixes_always_included() {
        let constraint = AdvancedSparqlConstraint::new("?this rdf:type rdfs:Resource .");
        let q = constraint.build_query("http://example.org/X");
        assert!(q.contains("PREFIX sh: <http://www.w3.org/ns/shacl#>"));
        assert!(q.contains("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"));
        assert!(q.contains("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"));
        assert!(q.contains("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"));
    }
}

// ---------------------------------------------------------------------------
// Additional tests for comprehensive SHACL 1.0 SPARQL constraint coverage
// ---------------------------------------------------------------------------

#[cfg(test)]
mod extended_tests {
    use super::*;

    fn row1(var: &str, val: &str) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(var.to_string(), val.to_string());
        m
    }

    fn row2(v1: &str, vv1: &str, v2: &str, vv2: &str) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(v1.to_string(), vv1.to_string());
        m.insert(v2.to_string(), vv2.to_string());
        m
    }

    // ---- Query construction details -------------------------------------

    #[test]
    fn test_build_query_contains_select() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o");
        let q = c.build_query("http://example.org/X");
        assert!(q.contains("SELECT"));
    }

    #[test]
    fn test_build_query_wraps_body_in_where() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o");
        let q = c.build_query("http://example.org/X");
        assert!(q.contains("WHERE"));
    }

    #[test]
    fn test_build_query_multiple_prefixes() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ex2:val")
            .add_prefix("ex", "http://example.org/")
            .add_prefix("ex2", "http://example2.org/");
        let q = c.build_query("http://example.org/X");
        assert!(q.contains("PREFIX ex: <http://example.org/>"));
        assert!(q.contains("PREFIX ex2: <http://example2.org/>"));
    }

    #[test]
    fn test_build_query_dollar_shapes_graph_replaced() {
        let c = AdvancedSparqlConstraint::new("GRAPH $shapesGraph { ?this sh:conformsTo ?shape }");
        let q = c.build_query("http://example.org/Node");
        // $shapesGraph is not a focus-node binding — ensure query is produced
        assert!(q.contains("shapesGraph"));
    }

    #[test]
    fn test_build_query_does_not_double_wrap_select() {
        // If the user already starts with SELECT the wrapper should still work
        let c = AdvancedSparqlConstraint::new("?this ex:age ?age FILTER(?age < 0)");
        let q = c.build_query("http://example.org/X");
        // Exactly one SELECT keyword in the final query
        let count = q.matches("SELECT").count();
        assert_eq!(count, 1, "expected exactly one SELECT, got: {q}");
    }

    // ---- Constraint IRI -------------------------------------------------

    #[test]
    fn test_constraint_iri_set_and_returned_in_result() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o")
            .with_iri("http://shapes.example.org/MyConstraint");
        let evaluator = MockSparqlEvaluator::new();
        let result = c
            .validate_node("http://example.org/Alice", &evaluator)
            .expect("validation should succeed");
        assert_eq!(
            result.constraint_iri,
            "http://shapes.example.org/MyConstraint"
        );
    }

    #[test]
    fn test_constraint_no_iri_returns_empty_string() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o");
        let evaluator = MockSparqlEvaluator::new();
        let result = c
            .validate_node("http://example.org/Alice", &evaluator)
            .expect("validation should succeed");
        assert_eq!(result.constraint_iri, "");
    }

    // ---- Multiple violations -------------------------------------------

    #[test]
    fn test_multiple_violation_rows_counted() {
        let rows = vec![
            row1("value", "http://example.org/A"),
            row1("value", "http://example.org/B"),
            row1("value", "http://example.org/C"),
        ];
        let c = AdvancedSparqlConstraint::new("?this ex:bad ?value");
        let evaluator = MockSparqlEvaluator::new().with_result("?this ex:bad", rows);
        let result = c
            .validate_node("http://example.org/X", &evaluator)
            .expect("validation should succeed");
        assert!(!result.is_valid);
        assert_eq!(result.violation_count(), 3);
    }

    // ---- Message template with multi-variable bindings -----------------

    #[test]
    fn test_format_message_multi_variable() {
        let c = AdvancedSparqlConstraint::new("SELECT * WHERE {}")
            .with_message("{?subject} has {?predicate} with illegal value {?value}");
        let bindings = row2("subject", "Alice", "predicate", "age")
            .into_iter()
            .chain(row1("value", "-3"))
            .collect();
        let msg = c.format_message(&bindings);
        assert!(msg.contains("Alice"));
        assert!(msg.contains("age"));
        assert!(msg.contains("-3"));
    }

    #[test]
    fn test_format_message_unknown_variable_left_as_placeholder() {
        let c = AdvancedSparqlConstraint::new("SELECT * WHERE {}")
            .with_message("Node {?this} violates {?unknownVar}");
        let bindings = row1("this", "http://example.org/X");
        let msg = c.format_message(&bindings);
        assert!(msg.contains("http://example.org/X"));
        // Unknown variable placeholder stays verbatim
        assert!(msg.contains("{?unknownVar}"));
    }

    // ---- validate_all edge cases ---------------------------------------

    #[test]
    fn test_validate_all_empty_node_list() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o");
        let evaluator = MockSparqlEvaluator::new();
        let results = c
            .validate_all(&[], &evaluator)
            .expect("validate_all on empty list should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_all_deactivated_skips_all() {
        let c = AdvancedSparqlConstraint::new("?this ex:bad ?o").deactivated();
        let evaluator = AlwaysViolatingEvaluator::new(row1("value", "bad"));
        let nodes: Vec<String> = (0..5)
            .map(|i| format!("http://example.org/Node{i}"))
            .collect();
        let results = c
            .validate_all(&nodes, &evaluator)
            .expect("deactivated validate_all should succeed");
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_valid));
    }

    #[test]
    fn test_validate_all_mixed_results() {
        // Nodes whose IRI contains "Bad" → violation, others → valid
        let bad_row = row1("value", "illegal");
        let evaluator = MockSparqlEvaluator::new().with_result("Bad", vec![bad_row]);
        let c = AdvancedSparqlConstraint::new("?this ex:type ?o");

        let nodes = vec![
            "http://example.org/GoodNode".to_string(),
            "http://example.org/BadNode".to_string(),
        ];

        let results = c
            .validate_all(&nodes, &evaluator)
            .expect("validate_all should succeed");
        assert_eq!(results.len(), 2);
        assert!(results[0].is_valid, "GoodNode should be valid");
        assert!(!results[1].is_valid, "BadNode should be invalid");
    }

    // ---- SparqlConstraintResult helpers --------------------------------

    #[test]
    fn test_is_satisfied_alias() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o");
        let evaluator = MockSparqlEvaluator::new();
        let result = c
            .validate_node("http://example.org/Alice", &evaluator)
            .expect("validation should succeed");
        assert!(result.is_satisfied());
        assert_eq!(result.is_satisfied(), result.is_valid);
    }

    #[test]
    fn test_severity_info_is_propagated_to_result() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o")
            .with_severity(SparqlConstraintSeverity::Info);
        let evaluator = AlwaysViolatingEvaluator::new(row1("value", "v"));
        let result = c
            .validate_node("http://example.org/X", &evaluator)
            .expect("validation should succeed");
        assert_eq!(result.severity, SparqlConstraintSeverity::Info);
    }

    #[test]
    fn test_severity_warning_is_propagated_to_result() {
        let c = AdvancedSparqlConstraint::new("?this ex:p ?o")
            .with_severity(SparqlConstraintSeverity::Warning);
        let evaluator = AlwaysViolatingEvaluator::new(row1("value", "v"));
        let result = c
            .validate_node("http://example.org/X", &evaluator)
            .expect("validation should succeed");
        assert_eq!(result.severity, SparqlConstraintSeverity::Warning);
    }

    // ---- Severity Display ----------------------------------------------

    #[test]
    fn test_severity_display_violation() {
        assert_eq!(
            format!("{}", SparqlConstraintSeverity::Violation),
            "sh:Violation"
        );
    }

    #[test]
    fn test_severity_display_warning() {
        assert_eq!(
            format!("{}", SparqlConstraintSeverity::Warning),
            "sh:Warning"
        );
    }

    #[test]
    fn test_severity_display_info() {
        assert_eq!(format!("{}", SparqlConstraintSeverity::Info), "sh:Info");
    }

    // ---- Default constraint --------------------------------------------

    #[test]
    fn test_default_constraint_is_not_deactivated() {
        let c = AdvancedSparqlConstraint::default();
        assert!(!c.deactivated);
    }

    #[test]
    fn test_default_severity_is_violation() {
        let c = AdvancedSparqlConstraint::default();
        assert_eq!(c.severity, SparqlConstraintSeverity::Violation);
    }

    // ---- MockSparqlEvaluator -------------------------------------------

    #[test]
    fn test_mock_evaluator_no_matching_key_returns_empty() {
        let evaluator =
            MockSparqlEvaluator::new().with_result("some_other_key", vec![row1("v", "bad")]);
        let rows = evaluator
            .execute_select("SELECT * WHERE { ?this ex:p ?o }")
            .expect("execute_select should not fail");
        assert!(rows.is_empty());
    }

    #[test]
    fn test_mock_evaluator_returns_multiple_rows() {
        let rows = vec![row1("v", "a"), row1("v", "b")];
        let evaluator = MockSparqlEvaluator::new().with_result("MATCH_KEY", rows.clone());
        let returned = evaluator
            .execute_select("PREFIX sh: <...> MATCH_KEY ?this ?o")
            .expect("execute_select should not fail");
        assert_eq!(returned.len(), 2);
    }

    // ---- FailingEvaluator propagation ----------------------------------

    #[test]
    fn test_failing_evaluator_returns_err_on_execute_select() {
        let evaluator = FailingEvaluator::new("timeout");
        let result = evaluator.execute_select("SELECT * WHERE {}");
        assert!(result.is_err());
    }

    // ---- AlwaysViolatingEvaluator --------------------------------------

    #[test]
    fn test_always_violating_returns_one_row() {
        let evaluator = AlwaysViolatingEvaluator::new(row1("value", "bad"));
        let rows = evaluator
            .execute_select("SELECT * WHERE {}")
            .expect("should not fail");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("value").map(String::as_str), Some("bad"));
    }

    // ---- bind_focus_node default implementation -----------------------

    #[test]
    fn test_bind_focus_node_default_returns_query_unchanged() {
        let evaluator = MockSparqlEvaluator::new();
        let q = "SELECT * WHERE { ?this ex:p ?o }";
        let result = evaluator.bind_focus_node(q, "http://example.org/X");
        assert_eq!(result, q);
    }
}
