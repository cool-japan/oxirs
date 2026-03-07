//! SHACL SPARQL-based constraint validation (sh:SPARQLConstraintComponent).
//!
//! This module implements validation of RDF nodes against SPARQL SELECT queries.
//! A result row from the SELECT query represents a constraint violation at the
//! focus node (`$this`).

use std::collections::HashMap;

// ─────────────────────────────────────────────────
// Severity
// ─────────────────────────────────────────────────

/// The severity level of a SHACL constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Violation,
    Warning,
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Violation => write!(f, "Violation"),
            Severity::Warning => write!(f, "Warning"),
            Severity::Info => write!(f, "Info"),
        }
    }
}

// ─────────────────────────────────────────────────
// SparqlConstraint
// ─────────────────────────────────────────────────

/// A SHACL SPARQL-based constraint.
///
/// The `select_query` is a SPARQL SELECT query where `$this` is the focus node.
/// Any result rows produced by the query represent violations.
#[derive(Debug, Clone)]
pub struct SparqlConstraint {
    pub id: String,
    pub select_query: String,
    pub message: String,
    pub severity: Severity,
    pub prefixes: HashMap<String, String>,
}

impl SparqlConstraint {
    /// Create a new SPARQL constraint with default Violation severity.
    pub fn new(
        id: impl Into<String>,
        select_query: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        SparqlConstraint {
            id: id.into(),
            select_query: select_query.into(),
            message: message.into(),
            severity: Severity::Violation,
            prefixes: HashMap::new(),
        }
    }

    /// Set the severity.
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Add a prefix declaration.
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Evaluate this constraint against a validation input.
    ///
    /// The simulation works as follows:
    /// 1. Substitute `$this` with the focus node IRI in the query.
    /// 2. Parse the query for simple triple patterns: `<s> <p> <o>` or `$this <p> <o>`.
    /// 3. For each triple pattern found, check whether it exists in the graph.
    ///    If ANY pattern matches, it means the constraint is violated.
    pub fn evaluate(&self, input: &ValidationInput) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();
        let substituted = self
            .select_query
            .replace("$this", &format!("<{}>", input.focus_node));

        // Simple simulation: look for lines that contain a triple pattern assertion.
        // A violation occurs when a pattern in the SELECT body matches the graph.
        let violated = self.query_matches_graph(&substituted, input);

        if violated {
            let message = self
                .message
                .replace("$this", &input.focus_node)
                .replace("{$this}", &input.focus_node);
            violations.push(ConstraintViolation {
                focus_node: input.focus_node.clone(),
                constraint_id: self.id.clone(),
                message,
                severity: self.severity.clone(),
                result_path: self.extract_result_path(&substituted),
                value: self.extract_value(&substituted, input),
            });
        }

        violations
    }

    // ── Private helpers ────────────────────────────────────────

    /// Check whether the (substituted) query body matches anything in `input.graph_triples`.
    ///
    /// Simplified simulation rules:
    /// 1. For each graph triple (s, p, o), check if the query "pattern" matches it.
    /// 2. Subject check: the substituted focus-node IRI appears, or the triple's subject IRI appears.
    /// 3. Predicate check: the triple's predicate IRI appears as `<p>` in the query body.
    /// 4. Object check: a SPARQL variable (`?something`) is present in the WHERE clause
    ///    (meaning the object is unconstrained), OR the literal/IRI appears in the query.
    fn query_matches_graph(&self, query: &str, input: &ValidationInput) -> bool {
        let focus_iri = format!("<{}>", input.focus_node);

        // Detect whether the WHERE clause contains an unbound object variable
        // (a SPARQL variable starting with ?)
        let has_object_variable = has_sparql_variable(query);

        for (s, p, o) in &input.graph_triples {
            let s_iri = format!("<{s}>");
            let p_iri = format!("<{p}>");
            let o_iri = format!("<{o}>");

            // Subject match: either the triple subject IRI is in query, or the focus-node IRI matches
            let s_in_query =
                query.contains(&s_iri) || (s == &input.focus_node && query.contains(&focus_iri));

            // Predicate match: the predicate IRI appears in the query as <p>
            let p_in_query = query.contains(&p_iri);

            // Object match: either a variable (wildcard) is in the query, or the specific value is
            let o_in_query = has_object_variable
                || query.contains(&o_iri)
                || query.contains(&format!("\"{o}\""))
                || query.contains(o.as_str());

            if s_in_query && p_in_query && o_in_query {
                return true;
            }
        }
        false
    }

    /// Extract a result path from the query, if any (simplified).
    fn extract_result_path(&self, query: &str) -> Option<String> {
        // Look for ?path or AS ?path in the SELECT clause
        if query.contains("?path") {
            Some("path".to_string())
        } else {
            None
        }
    }

    /// Extract a value from the query / graph (simplified).
    fn extract_value(&self, _query: &str, _input: &ValidationInput) -> Option<String> {
        None
    }
}

// ─────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────

/// Returns `true` if the given SPARQL query text contains a SPARQL variable
/// (a token starting with `?`), which acts as a wildcard in pattern matching.
fn has_sparql_variable(query: &str) -> bool {
    query.contains('?')
}

// ─────────────────────────────────────────────────
// ValidationInput
// ─────────────────────────────────────────────────

/// Input for validating a single focus node.
#[derive(Debug, Clone)]
pub struct ValidationInput {
    /// The IRI of the focus node being validated.
    pub focus_node: String,
    /// All triples in the validation graph.
    pub graph_triples: Vec<(String, String, String)>,
}

impl ValidationInput {
    /// Convenience constructor.
    pub fn new(
        focus_node: impl Into<String>,
        graph_triples: Vec<(String, String, String)>,
    ) -> Self {
        ValidationInput {
            focus_node: focus_node.into(),
            graph_triples,
        }
    }
}

// ─────────────────────────────────────────────────
// ConstraintViolation
// ─────────────────────────────────────────────────

/// A single constraint violation produced by evaluating a SPARQL constraint.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub focus_node: String,
    pub constraint_id: String,
    pub message: String,
    pub severity: Severity,
    pub result_path: Option<String>,
    pub value: Option<String>,
}

// ─────────────────────────────────────────────────
// SparqlConstraintResult
// ─────────────────────────────────────────────────

/// The result of validating one focus node against all constraints.
#[derive(Debug, Clone)]
pub struct SparqlConstraintResult {
    pub violations: Vec<ConstraintViolation>,
    pub conforms: bool,
}

impl SparqlConstraintResult {
    fn new(violations: Vec<ConstraintViolation>) -> Self {
        let conforms = violations.is_empty();
        SparqlConstraintResult {
            violations,
            conforms,
        }
    }
}

// ─────────────────────────────────────────────────
// SparqlConstraintValidator
// ─────────────────────────────────────────────────

/// Validates RDF focus nodes against a set of SPARQL-based SHACL constraints.
#[derive(Debug, Default)]
pub struct SparqlConstraintValidator {
    constraints: Vec<SparqlConstraint>,
}

impl SparqlConstraintValidator {
    /// Create a validator with no constraints.
    pub fn new() -> Self {
        SparqlConstraintValidator {
            constraints: Vec::new(),
        }
    }

    /// Register a constraint.
    pub fn add_constraint(&mut self, constraint: SparqlConstraint) {
        self.constraints.push(constraint);
    }

    /// Validate a single focus node against all registered constraints.
    pub fn validate(&self, input: &ValidationInput) -> SparqlConstraintResult {
        let mut all_violations = Vec::new();
        for constraint in &self.constraints {
            all_violations.extend(constraint.evaluate(input));
        }
        SparqlConstraintResult::new(all_violations)
    }

    /// Validate multiple focus nodes; returns one result per input.
    pub fn validate_all(&self, inputs: &[ValidationInput]) -> Vec<SparqlConstraintResult> {
        inputs.iter().map(|i| self.validate(i)).collect()
    }

    /// Number of registered constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str) -> (String, String, String) {
        (s.to_string(), p.to_string(), o.to_string())
    }

    fn input_with_triple(focus: &str, s: &str, p: &str, o: &str) -> ValidationInput {
        ValidationInput::new(focus, vec![triple(s, p, o)])
    }

    // ── No constraints ─────────────────────────────────────────

    #[test]
    fn test_no_constraints_conforms() {
        let validator = SparqlConstraintValidator::new();
        let input = ValidationInput::new("http://ex.org/alice", vec![]);
        let result = validator.validate(&input);
        assert!(result.conforms);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_no_constraints_count_zero() {
        let validator = SparqlConstraintValidator::new();
        assert_eq!(validator.constraint_count(), 0);
    }

    // ── Single constraint: no violation ───────────────────────

    #[test]
    fn test_single_constraint_no_violation_empty_graph() {
        let mut validator = SparqlConstraintValidator::new();
        let constraint = SparqlConstraint::new(
            "c1",
            "SELECT ?this WHERE { $this <http://ex.org/badProp> ?o }",
            "Bad property present",
        );
        validator.add_constraint(constraint);

        let input = ValidationInput::new("http://ex.org/alice", vec![]);
        let result = validator.validate(&input);
        assert!(result.conforms);
    }

    #[test]
    fn test_single_constraint_no_violation_different_predicate() {
        let mut validator = SparqlConstraintValidator::new();
        let constraint = SparqlConstraint::new(
            "c1",
            "SELECT ?this WHERE { $this <http://ex.org/badProp> ?o }",
            "Bad property present",
        );
        validator.add_constraint(constraint);

        let input = input_with_triple(
            "http://ex.org/alice",
            "http://ex.org/alice",
            "http://ex.org/goodProp",
            "someValue",
        );
        let result = validator.validate(&input);
        assert!(result.conforms);
    }

    // ── Single constraint: violation ──────────────────────────

    #[test]
    fn test_single_constraint_with_violation() {
        let mut validator = SparqlConstraintValidator::new();
        let constraint = SparqlConstraint::new(
            "c_bad",
            "SELECT ?this WHERE { $this <http://ex.org/badProp> ?o }",
            "Node has badProp",
        );
        validator.add_constraint(constraint);

        let input = input_with_triple(
            "http://ex.org/alice",
            "http://ex.org/alice",
            "http://ex.org/badProp",
            "forbiddenValue",
        );
        let result = validator.validate(&input);
        assert!(!result.conforms);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].constraint_id, "c_bad");
        assert_eq!(result.violations[0].focus_node, "http://ex.org/alice");
    }

    // ── Multiple constraints ───────────────────────────────────

    #[test]
    fn test_multiple_constraints_both_fire() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "c1",
            "SELECT ?this WHERE { $this <http://ex.org/p1> ?o }",
            "p1 violation",
        ));
        validator.add_constraint(SparqlConstraint::new(
            "c2",
            "SELECT ?this WHERE { $this <http://ex.org/p2> ?o }",
            "p2 violation",
        ));

        let input = ValidationInput::new(
            "http://ex.org/alice",
            vec![
                triple("http://ex.org/alice", "http://ex.org/p1", "v1"),
                triple("http://ex.org/alice", "http://ex.org/p2", "v2"),
            ],
        );
        let result = validator.validate(&input);
        assert!(!result.conforms);
        assert_eq!(result.violations.len(), 2);
    }

    #[test]
    fn test_multiple_constraints_only_one_fires() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "c1",
            "SELECT ?this WHERE { $this <http://ex.org/p1> ?o }",
            "p1 violation",
        ));
        validator.add_constraint(SparqlConstraint::new(
            "c2",
            "SELECT ?this WHERE { $this <http://ex.org/p2> ?o }",
            "p2 violation",
        ));

        let input = input_with_triple(
            "http://ex.org/alice",
            "http://ex.org/alice",
            "http://ex.org/p1",
            "v",
        );
        let result = validator.validate(&input);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].constraint_id, "c1");
    }

    // ── Severity levels ────────────────────────────────────────

    #[test]
    fn test_severity_violation_default() {
        let c = SparqlConstraint::new("c", "SELECT * WHERE { $this ?p ?o }", "msg");
        assert_eq!(c.severity, Severity::Violation);
    }

    #[test]
    fn test_severity_warning() {
        let c = SparqlConstraint::new("c", "SELECT * WHERE { $this ?p ?o }", "msg")
            .with_severity(Severity::Warning);
        assert_eq!(c.severity, Severity::Warning);
    }

    #[test]
    fn test_severity_info() {
        let c = SparqlConstraint::new("c", "SELECT * WHERE { $this ?p ?o }", "msg")
            .with_severity(Severity::Info);
        assert_eq!(c.severity, Severity::Info);
    }

    #[test]
    fn test_violation_carries_severity() {
        let mut validator = SparqlConstraintValidator::new();
        let c = SparqlConstraint::new(
            "c_warn",
            "SELECT ?this WHERE { $this <http://ex.org/p> ?o }",
            "warning msg",
        )
        .with_severity(Severity::Warning);
        validator.add_constraint(c);

        let input = input_with_triple(
            "http://ex.org/node",
            "http://ex.org/node",
            "http://ex.org/p",
            "val",
        );
        let result = validator.validate(&input);
        assert_eq!(result.violations[0].severity, Severity::Warning);
    }

    // ── validate_all ──────────────────────────────────────────

    #[test]
    fn test_validate_all_empty_inputs() {
        let validator = SparqlConstraintValidator::new();
        let results = validator.validate_all(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_all_multiple_inputs() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "c",
            "SELECT ?this WHERE { $this <http://ex.org/bad> ?o }",
            "msg",
        ));

        let inputs = vec![
            input_with_triple(
                "http://ex.org/a",
                "http://ex.org/a",
                "http://ex.org/bad",
                "v",
            ),
            ValidationInput::new("http://ex.org/b", vec![]),
        ];
        let results = validator.validate_all(&inputs);
        assert_eq!(results.len(), 2);
        assert!(!results[0].conforms);
        assert!(results[1].conforms);
    }

    #[test]
    fn test_validate_all_all_conform() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "c",
            "SELECT ?this WHERE { $this <http://ex.org/bad> ?o }",
            "msg",
        ));
        let inputs = vec![
            ValidationInput::new("http://ex.org/a", vec![]),
            ValidationInput::new("http://ex.org/b", vec![]),
        ];
        let results = validator.validate_all(&inputs);
        assert!(results.iter().all(|r| r.conforms));
    }

    // ── Prefix handling ────────────────────────────────────────

    #[test]
    fn test_constraint_with_prefix() {
        let c = SparqlConstraint::new("c", "SELECT * WHERE {}", "msg")
            .with_prefix("ex", "http://example.org/");
        assert_eq!(
            c.prefixes.get("ex"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_constraint_multiple_prefixes() {
        let c = SparqlConstraint::new("c", "SELECT * WHERE {}", "msg")
            .with_prefix("ex", "http://example.org/")
            .with_prefix("schema", "https://schema.org/");
        assert_eq!(c.prefixes.len(), 2);
    }

    // ── Message template ──────────────────────────────────────

    #[test]
    fn test_message_preserves_text() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "c",
            "SELECT ?this WHERE { $this <http://ex.org/p> ?o }",
            "Node violates constraint",
        ));

        let input = input_with_triple(
            "http://ex.org/node",
            "http://ex.org/node",
            "http://ex.org/p",
            "val",
        );
        let result = validator.validate(&input);
        assert!(result.violations[0].message.contains("constraint"));
    }

    #[test]
    fn test_message_substitutes_focus_node() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "c",
            "SELECT ?this WHERE { $this <http://ex.org/p> ?o }",
            "Node $this has p property",
        ));

        let input = input_with_triple(
            "http://ex.org/alice",
            "http://ex.org/alice",
            "http://ex.org/p",
            "v",
        );
        let result = validator.validate(&input);
        assert!(!result.violations.is_empty());
        assert!(result.violations[0].message.contains("http://ex.org/alice"));
    }

    // ── Constraint count ──────────────────────────────────────

    #[test]
    fn test_add_constraint_increases_count() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new("c1", "SELECT 1", "m1"));
        assert_eq!(validator.constraint_count(), 1);
        validator.add_constraint(SparqlConstraint::new("c2", "SELECT 2", "m2"));
        assert_eq!(validator.constraint_count(), 2);
    }

    // ── Severity Display ──────────────────────────────────────

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Violation.to_string(), "Violation");
        assert_eq!(Severity::Warning.to_string(), "Warning");
        assert_eq!(Severity::Info.to_string(), "Info");
    }

    // ── Constraint id propagation ─────────────────────────────

    #[test]
    fn test_violation_has_correct_constraint_id() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "unique-constraint-id-99",
            "SELECT ?this WHERE { $this <http://ex.org/q> ?o }",
            "msg",
        ));
        let input = input_with_triple("http://ex.org/x", "http://ex.org/x", "http://ex.org/q", "v");
        let result = validator.validate(&input);
        assert_eq!(
            result.violations[0].constraint_id,
            "unique-constraint-id-99"
        );
    }

    // ── Focus node propagation ────────────────────────────────

    #[test]
    fn test_violation_has_correct_focus_node() {
        let mut validator = SparqlConstraintValidator::new();
        validator.add_constraint(SparqlConstraint::new(
            "c",
            "SELECT ?this WHERE { $this <http://ex.org/p> ?o }",
            "msg",
        ));
        let input = input_with_triple(
            "http://ex.org/specificNode",
            "http://ex.org/specificNode",
            "http://ex.org/p",
            "v",
        );
        let result = validator.validate(&input);
        assert_eq!(
            result.violations[0].focus_node,
            "http://ex.org/specificNode"
        );
    }

    // ── ValidationInput ────────────────────────────────────────

    #[test]
    fn test_validation_input_new() {
        let input = ValidationInput::new(
            "http://ex.org/node",
            vec![triple("http://ex.org/node", "http://ex.org/p", "v")],
        );
        assert_eq!(input.focus_node, "http://ex.org/node");
        assert_eq!(input.graph_triples.len(), 1);
    }

    #[test]
    fn test_validation_input_empty_graph() {
        let input = ValidationInput::new("http://ex.org/node", vec![]);
        assert!(input.graph_triples.is_empty());
    }

    // ── SparqlConstraintResult ────────────────────────────────

    #[test]
    fn test_result_conforms_when_no_violations() {
        let result = SparqlConstraintResult::new(vec![]);
        assert!(result.conforms);
    }

    #[test]
    fn test_result_not_conforms_with_violations() {
        let v = ConstraintViolation {
            focus_node: "f".into(),
            constraint_id: "c".into(),
            message: "m".into(),
            severity: Severity::Violation,
            result_path: None,
            value: None,
        };
        let result = SparqlConstraintResult::new(vec![v]);
        assert!(!result.conforms);
    }

    // ── Default impl ──────────────────────────────────────────

    #[test]
    fn test_validator_default() {
        let v = SparqlConstraintValidator::default();
        assert_eq!(v.constraint_count(), 0);
    }
}
