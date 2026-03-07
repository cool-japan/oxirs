//! W3C RDF-star Compliance Checker
//!
//! This module provides a compliance checker that validates RDF-star data against
//! the W3C RDF-star specification. It implements the following compliance rules:
//!
//! - STAR-001: Embedded triple must be ground (no variables)
//! - STAR-002: Annotation consistency (annotation predicates must be named nodes)
//! - STAR-003: No blank node as embedded subject in asserted triples
//! - STAR-004: Nested embedding depth check (prevents excessive nesting)

use crate::model::{StarTerm, StarTriple};
use crate::w3c_compliance::Annotation;

/// Severity level for a compliance violation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational notice — does not prevent conformance.
    Info,
    /// Warning — potential issue but technically permitted.
    Warning,
    /// Error — violates the W3C RDF-star specification; graph is non-conformant.
    Error,
}

/// A single W3C RDF-star compliance violation.
#[derive(Debug, Clone)]
pub struct ComplianceViolation {
    /// Short identifier for the rule that was violated (e.g. "STAR-001").
    pub rule_id: &'static str,
    /// Human-readable description of what went wrong.
    pub message: String,
    /// Severity of the violation.
    pub severity: Severity,
}

impl ComplianceViolation {
    fn new(rule_id: &'static str, message: impl Into<String>, severity: Severity) -> Self {
        Self {
            rule_id,
            message: message.into(),
            severity,
        }
    }

    fn error(rule_id: &'static str, message: impl Into<String>) -> Self {
        Self::new(rule_id, message, Severity::Error)
    }

    fn warning(rule_id: &'static str, message: impl Into<String>) -> Self {
        Self::new(rule_id, message, Severity::Warning)
    }

    #[allow(dead_code)]
    fn info(rule_id: &'static str, message: impl Into<String>) -> Self {
        Self::new(rule_id, message, Severity::Info)
    }
}

/// Summary report produced by [`W3cRdfStarChecker::check_graph`].
#[derive(Debug, Clone, Default)]
pub struct ComplianceReport {
    /// All violations found during the check.
    pub violations: Vec<ComplianceViolation>,
    /// `true` if no `Error`-severity violations were found.
    pub conformant: bool,
}

impl ComplianceReport {
    fn new(violations: Vec<ComplianceViolation>) -> Self {
        let conformant = violations.iter().all(|v| v.severity != Severity::Error);
        Self {
            violations,
            conformant,
        }
    }

    /// Count violations at a given severity level.
    pub fn count_severity(&self, severity: &Severity) -> usize {
        self.violations
            .iter()
            .filter(|v| &v.severity == severity)
            .count()
    }

    /// Return only the error-level violations.
    pub fn errors(&self) -> Vec<&ComplianceViolation> {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect()
    }

    /// Return only the warning-level violations.
    pub fn warnings(&self) -> Vec<&ComplianceViolation> {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Warning)
            .collect()
    }
}

/// Maximum permitted nesting depth for embedded triples (W3C does not impose
/// a hard limit, but excessively deep nesting is flagged as a warning).
const DEFAULT_MAX_NESTING_DEPTH: usize = 20;

/// W3C RDF-star compliance checker.
///
/// Validates RDF-star graphs, individual triples, and annotations against the
/// W3C RDF-star specification rules.
///
/// # Example
/// ```
/// use oxirs_star::compliance::W3cRdfStarChecker;
/// use oxirs_star::model::{StarTerm, StarTriple};
///
/// let checker = W3cRdfStarChecker::new();
/// let triple = StarTriple::new(
///     StarTerm::iri("http://example.org/s").unwrap(),
///     StarTerm::iri("http://example.org/p").unwrap(),
///     StarTerm::iri("http://example.org/o").unwrap(),
/// );
/// let violations = checker.check_embedded_triple(&triple);
/// assert!(violations.is_empty());
/// ```
pub struct W3cRdfStarChecker {
    /// Maximum nesting depth before issuing a STAR-004 warning.
    max_nesting_depth: usize,
}

impl Default for W3cRdfStarChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl W3cRdfStarChecker {
    /// Create a new checker with the default configuration.
    pub fn new() -> Self {
        Self {
            max_nesting_depth: DEFAULT_MAX_NESTING_DEPTH,
        }
    }

    /// Create a checker with a custom maximum nesting depth.
    pub fn with_max_nesting_depth(max_nesting_depth: usize) -> Self {
        Self { max_nesting_depth }
    }

    // -----------------------------------------------------------------------
    // STAR-001: Embedded triple must be ground
    // -----------------------------------------------------------------------

    /// Check that an embedded (quoted) triple is *ground* — that is, it
    /// contains no variables in any position (subject, predicate, or object).
    ///
    /// Rule: **STAR-001** — Embedded triples used as the subject or object of
    /// another triple must be ground according to the W3C RDF-star abstract
    /// syntax.
    pub fn check_embedded_triple(&self, triple: &StarTriple) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();
        self.check_term_ground(&triple.subject, "subject", &mut violations);
        self.check_term_ground(&triple.predicate, "predicate", &mut violations);
        self.check_term_ground(&triple.object, "object", &mut violations);
        violations
    }

    fn check_term_ground(
        &self,
        term: &StarTerm,
        position: &str,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        match term {
            StarTerm::Variable(v) => {
                violations.push(ComplianceViolation::error(
                    "STAR-001",
                    format!(
                        "Embedded triple {} position contains variable '{}'; \
                         embedded triples must be ground (no variables)",
                        position, v.name
                    ),
                ));
            }
            StarTerm::QuotedTriple(inner) => {
                // Recursively check nested quoted triples
                self.check_term_ground(&inner.subject, "subject", violations);
                self.check_term_ground(&inner.predicate, "predicate", violations);
                self.check_term_ground(&inner.object, "object", violations);
            }
            StarTerm::NamedNode(_) | StarTerm::BlankNode(_) | StarTerm::Literal(_) => {}
        }
    }

    // -----------------------------------------------------------------------
    // STAR-002: Annotation consistency
    // -----------------------------------------------------------------------

    /// Validate that an annotation follows the W3C RDF-star annotation pattern:
    /// the annotation predicate must be a named node (IRI), and the base triple
    /// must be well-formed.
    ///
    /// Rule: **STAR-002** — Annotation predicate must be a named node; the base
    /// triple of an annotation must pass embedded-triple validation.
    pub fn check_annotation(&self, annotation: &Annotation) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();

        // Predicate must be a named node
        if !matches!(annotation.predicate, StarTerm::NamedNode(_)) {
            violations.push(ComplianceViolation::error(
                "STAR-002",
                format!(
                    "Annotation predicate is not a named node: {:?}",
                    annotation.predicate
                ),
            ));
        }

        // Check that the annotation object is not a variable
        if matches!(annotation.object, StarTerm::Variable(_)) {
            violations.push(ComplianceViolation::warning(
                "STAR-002",
                "Annotation object is a variable; annotations in data graphs should use ground terms",
            ));
        }

        // Base triple must be ground (it will be embedded)
        let base_violations = self.check_embedded_triple(&annotation.base_triple);
        violations.extend(base_violations);

        violations
    }

    // -----------------------------------------------------------------------
    // STAR-003: No blank node as embedded subject (in asserted context)
    // -----------------------------------------------------------------------

    /// Check that no blank node appears as the subject of an embedded triple
    /// used in an asserted context. The W3C RDF-star specification requires
    /// that embedded triples used as subjects or objects of asserted triples
    /// do not use blank nodes as their subject (to preserve Skolemisation
    /// compatibility).
    ///
    /// Rule: **STAR-003** — Blank nodes as the subject of an embedded triple
    /// are a portability concern and are flagged as a warning.
    pub fn check_embedded_blank_subject(&self, triple: &StarTriple) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();
        self.check_term_blank_as_embedded_subject(triple, &mut violations);
        violations
    }

    fn check_term_blank_as_embedded_subject(
        &self,
        triple: &StarTriple,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        if matches!(triple.subject, StarTerm::BlankNode(_)) {
            violations.push(ComplianceViolation::warning(
                "STAR-003",
                format!(
                    "Embedded triple has blank node as subject: {:?}; \
                     this may cause interoperability issues",
                    triple.subject
                ),
            ));
        }

        // Recurse into nested quoted triples
        if let StarTerm::QuotedTriple(inner) = &triple.subject {
            self.check_term_blank_as_embedded_subject(inner, violations);
        }
        if let StarTerm::QuotedTriple(inner) = &triple.object {
            self.check_term_blank_as_embedded_subject(inner, violations);
        }
    }

    // -----------------------------------------------------------------------
    // STAR-004: Nesting depth check
    // -----------------------------------------------------------------------

    /// Check that the nesting depth of embedded triples does not exceed the
    /// configured maximum. Deeply nested quoted triples are technically valid
    /// but impractical and a likely sign of a modelling error.
    ///
    /// Rule: **STAR-004** — Nesting depth exceeding the threshold is reported
    /// as a warning.
    pub fn check_nesting_depth(&self, triple: &StarTriple) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();
        let depth = triple.nesting_depth();
        if depth > self.max_nesting_depth {
            violations.push(ComplianceViolation::warning(
                "STAR-004",
                format!(
                    "Embedded triple nesting depth {} exceeds maximum {}; \
                     deeply nested quoted triples may indicate a modelling error",
                    depth, self.max_nesting_depth
                ),
            ));
        }
        violations
    }

    // -----------------------------------------------------------------------
    // Full graph check
    // -----------------------------------------------------------------------

    /// Run all compliance rules over every triple in `graph` and return a
    /// [`ComplianceReport`].
    ///
    /// The graph is provided as a slice of [`StarTriple`] references, matching
    /// the interface used throughout the crate.
    pub fn check_graph(&self, graph: &[StarTriple]) -> ComplianceReport {
        let mut violations = Vec::new();

        for triple in graph {
            // STAR-001: all quoted-triple terms must be ground
            self.check_embedded_triple_in_graph(triple, &mut violations);

            // STAR-003: no blank nodes as embedded subjects
            self.check_blank_subjects_in_graph(triple, &mut violations);

            // STAR-004: nesting depth
            violations.extend(self.check_nesting_depth(triple));
        }

        // STAR-002: Check annotation-style triples (subject is quoted triple)
        self.check_graph_annotations(graph, &mut violations);

        ComplianceReport::new(violations)
    }

    /// Check STAR-001 for all embedded (quoted) triple terms in a graph triple.
    fn check_embedded_triple_in_graph(
        &self,
        triple: &StarTriple,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        if let StarTerm::QuotedTriple(inner) = &triple.subject {
            violations.extend(self.check_embedded_triple(inner));
        }
        if let StarTerm::QuotedTriple(inner) = &triple.predicate {
            violations.extend(self.check_embedded_triple(inner));
        }
        if let StarTerm::QuotedTriple(inner) = &triple.object {
            violations.extend(self.check_embedded_triple(inner));
        }
    }

    /// Check STAR-003 for blank subjects in embedded triples within a graph triple.
    fn check_blank_subjects_in_graph(
        &self,
        triple: &StarTriple,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        if let StarTerm::QuotedTriple(inner) = &triple.subject {
            self.check_term_blank_as_embedded_subject(inner, violations);
        }
        if let StarTerm::QuotedTriple(inner) = &triple.object {
            self.check_term_blank_as_embedded_subject(inner, violations);
        }
    }

    /// Check STAR-002 for annotation patterns in the graph.
    ///
    /// An annotation triple is one where the subject is a quoted triple.
    /// The predicate of such a triple must be a named node.
    fn check_graph_annotations(
        &self,
        graph: &[StarTriple],
        violations: &mut Vec<ComplianceViolation>,
    ) {
        for triple in graph {
            // This is an annotation-style triple: subject is a quoted triple
            if matches!(triple.subject, StarTerm::QuotedTriple(_))
                && !matches!(triple.predicate, StarTerm::NamedNode(_))
            {
                violations.push(ComplianceViolation::error(
                    "STAR-002",
                    format!(
                        "Annotation triple has non-IRI predicate: {:?}",
                        triple.predicate
                    ),
                ));
            }
        }
    }

    /// Check a specific triple for all applicable compliance rules.
    /// Returns violations from STAR-001, STAR-003, and STAR-004.
    pub fn check_triple(&self, triple: &StarTriple) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();

        // STAR-001 for embedded term positions
        self.check_embedded_triple_in_graph(triple, &mut violations);

        // STAR-003
        self.check_blank_subjects_in_graph(triple, &mut violations);

        // STAR-004
        violations.extend(self.check_nesting_depth(triple));

        // STAR-002 if annotation-style
        if matches!(triple.subject, StarTerm::QuotedTriple(_))
            && !matches!(triple.predicate, StarTerm::NamedNode(_))
        {
            violations.push(ComplianceViolation::error(
                "STAR-002",
                format!(
                    "Annotation triple has non-IRI predicate: {:?}",
                    triple.predicate
                ),
            ));
        }

        violations
    }
}

// ---------------------------------------------------------------------------
// Utility helpers used in tests
// ---------------------------------------------------------------------------

/// Build a quick named-node term (panics if empty — only for tests).
#[cfg(test)]
fn iri(s: &str) -> StarTerm {
    StarTerm::iri(s).expect("non-empty IRI")
}

#[cfg(test)]
fn lit(s: &str) -> StarTerm {
    StarTerm::literal(s).expect("literal")
}

#[cfg(test)]
fn blank(id: &str) -> StarTerm {
    StarTerm::blank_node(id).expect("non-empty blank node id")
}

#[cfg(test)]
fn var(name: &str) -> StarTerm {
    StarTerm::variable(name).expect("non-empty variable name")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTriple;
    use crate::w3c_compliance::Annotation;

    fn ground_triple() -> StarTriple {
        StarTriple::new(
            iri("http://example.org/s"),
            iri("http://example.org/p"),
            lit("42"),
        )
    }

    // -----------------------------------------------------------------------
    // STAR-001 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_star001_ground_triple_passes() {
        let checker = W3cRdfStarChecker::new();
        let violations = checker.check_embedded_triple(&ground_triple());
        assert!(
            violations.is_empty(),
            "ground triple should have no violations"
        );
    }

    #[test]
    fn test_star001_variable_subject_fails() {
        let checker = W3cRdfStarChecker::new();
        let triple = StarTriple::new(var("s"), iri("http://example.org/p"), lit("42"));
        let violations = checker.check_embedded_triple(&triple);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule_id, "STAR-001");
        assert_eq!(violations[0].severity, Severity::Error);
    }

    #[test]
    fn test_star001_variable_predicate_fails() {
        let checker = W3cRdfStarChecker::new();
        let triple = StarTriple::new(iri("http://example.org/s"), var("p"), lit("42"));
        let violations = checker.check_embedded_triple(&triple);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule_id, "STAR-001");
    }

    #[test]
    fn test_star001_variable_object_fails() {
        let checker = W3cRdfStarChecker::new();
        let triple = StarTriple::new(
            iri("http://example.org/s"),
            iri("http://example.org/p"),
            var("o"),
        );
        let violations = checker.check_embedded_triple(&triple);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule_id, "STAR-001");
    }

    #[test]
    fn test_star001_multiple_variables_all_reported() {
        let checker = W3cRdfStarChecker::new();
        let triple = StarTriple::new(var("s"), var("p"), var("o"));
        let violations = checker.check_embedded_triple(&triple);
        assert_eq!(violations.len(), 3, "all three variable positions reported");
    }

    #[test]
    fn test_star001_nested_variable_fails() {
        let checker = W3cRdfStarChecker::new();
        let inner = StarTriple::new(var("s"), iri("http://example.org/p"), lit("obj"));
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://example.org/p2"),
            lit("val"),
        );
        // The inner triple is embedded and has a variable subject
        let violations = checker.check_embedded_triple(&outer);
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.rule_id == "STAR-001"));
    }

    #[test]
    fn test_star001_blank_node_is_ground() {
        let checker = W3cRdfStarChecker::new();
        let triple = StarTriple::new(blank("b1"), iri("http://example.org/p"), lit("42"));
        // Blank nodes are ground — STAR-001 should not fire
        let violations = checker.check_embedded_triple(&triple);
        assert!(
            violations.iter().all(|v| v.rule_id != "STAR-001"),
            "blank nodes are ground terms"
        );
    }

    // -----------------------------------------------------------------------
    // STAR-002 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_star002_valid_annotation() {
        let checker = W3cRdfStarChecker::new();
        let annotation = Annotation::new(
            ground_triple(),
            iri("http://example.org/certainty"),
            lit("0.9"),
        );
        let violations = checker.check_annotation(&annotation);
        assert!(violations.is_empty(), "valid annotation should pass");
    }

    #[test]
    fn test_star002_literal_predicate_fails() {
        let checker = W3cRdfStarChecker::new();
        let annotation = Annotation::new(ground_triple(), lit("invalid_predicate"), lit("value"));
        let violations = checker.check_annotation(&annotation);
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.rule_id == "STAR-002"));
        assert!(violations.iter().any(|v| v.severity == Severity::Error));
    }

    #[test]
    fn test_star002_blank_predicate_fails() {
        let checker = W3cRdfStarChecker::new();
        let annotation = Annotation::new(ground_triple(), blank("pred_blank"), lit("value"));
        let violations = checker.check_annotation(&annotation);
        assert!(violations
            .iter()
            .any(|v| v.rule_id == "STAR-002" && v.severity == Severity::Error));
    }

    #[test]
    fn test_star002_variable_object_is_warning() {
        let checker = W3cRdfStarChecker::new();
        let annotation =
            Annotation::new(ground_triple(), iri("http://example.org/source"), var("x"));
        let violations = checker.check_annotation(&annotation);
        assert!(violations
            .iter()
            .any(|v| v.rule_id == "STAR-002" && v.severity == Severity::Warning));
        // No errors
        assert!(!violations.iter().any(|v| v.severity == Severity::Error));
    }

    #[test]
    fn test_star002_graph_annotation_triple() {
        let checker = W3cRdfStarChecker::new();
        let inner = ground_triple();
        // Annotation triple: <<s p o>> pred obj — predicate is a named node ✓
        let valid_ann = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://example.org/certainty"),
            lit("high"),
        );
        let report = checker.check_graph(&[valid_ann]);
        assert!(report.conformant);

        // Invalid annotation: predicate is a literal ✗
        let invalid_ann = StarTriple::new(
            StarTerm::quoted_triple(inner),
            lit("not-a-predicate"),
            lit("high"),
        );
        let report = checker.check_graph(&[invalid_ann]);
        assert!(!report.conformant);
        assert!(report.errors().iter().any(|v| v.rule_id == "STAR-002"));
    }

    // -----------------------------------------------------------------------
    // STAR-003 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_star003_blank_subject_in_embedded_triple() {
        let checker = W3cRdfStarChecker::new();
        let inner = StarTriple::new(blank("b1"), iri("http://example.org/p"), lit("val"));
        let violations = checker.check_embedded_blank_subject(&inner);
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.rule_id == "STAR-003"));
        assert!(violations.iter().any(|v| v.severity == Severity::Warning));
    }

    #[test]
    fn test_star003_iri_subject_is_fine() {
        let checker = W3cRdfStarChecker::new();
        let violations = checker.check_embedded_blank_subject(&ground_triple());
        assert!(violations.iter().all(|v| v.rule_id != "STAR-003"));
    }

    #[test]
    fn test_star003_detected_in_graph_check() {
        let checker = W3cRdfStarChecker::new();
        let inner = StarTriple::new(blank("b1"), iri("http://example.org/p"), lit("val"));
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://example.org/meta"),
            lit("x"),
        );
        let report = checker.check_graph(&[outer]);
        assert!(report.violations.iter().any(|v| v.rule_id == "STAR-003"));
    }

    // -----------------------------------------------------------------------
    // STAR-004 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_star004_shallow_nesting_ok() {
        let checker = W3cRdfStarChecker::new();
        let inner = ground_triple();
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://example.org/p"),
            lit("v"),
        );
        let violations = checker.check_nesting_depth(&outer);
        assert!(
            violations.iter().all(|v| v.rule_id != "STAR-004"),
            "single-level nesting should not trigger STAR-004"
        );
    }

    #[test]
    fn test_star004_excessive_nesting_warns() {
        let checker = W3cRdfStarChecker::with_max_nesting_depth(2);

        // Build 4-level nesting (depth = 3, which exceeds max of 2)
        let level1 = ground_triple();
        let level2 = StarTriple::new(
            StarTerm::quoted_triple(level1),
            iri("http://example.org/p"),
            lit("v"),
        );
        let level3 = StarTriple::new(
            StarTerm::quoted_triple(level2),
            iri("http://example.org/p"),
            lit("v"),
        );
        let level4 = StarTriple::new(
            StarTerm::quoted_triple(level3),
            iri("http://example.org/p"),
            lit("v"),
        );

        let violations = checker.check_nesting_depth(&level4);
        assert!(!violations.is_empty(), "excessive nesting should warn");
        assert!(violations
            .iter()
            .any(|v| v.rule_id == "STAR-004" && v.severity == Severity::Warning));
    }

    // -----------------------------------------------------------------------
    // check_graph integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_graph_empty_is_conformant() {
        let checker = W3cRdfStarChecker::new();
        let report = checker.check_graph(&[]);
        assert!(report.conformant);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn test_check_graph_simple_ground_triples_conformant() {
        let checker = W3cRdfStarChecker::new();
        let triples = vec![
            ground_triple(),
            StarTriple::new(
                iri("http://example.org/alice"),
                iri("http://example.org/knows"),
                iri("http://example.org/bob"),
            ),
        ];
        let report = checker.check_graph(&triples);
        assert!(report.conformant);
    }

    #[test]
    fn test_check_graph_mixed_violations_collected() {
        let checker = W3cRdfStarChecker::new();

        // Triple 1: valid annotation
        let annotation_triple = StarTriple::new(
            StarTerm::quoted_triple(ground_triple()),
            iri("http://example.org/confidence"),
            lit("high"),
        );

        // Triple 2: annotation with variable embedded triple (STAR-001 error)
        let bad_inner = StarTriple::new(var("s"), iri("http://example.org/p"), lit("v"));
        let bad_triple = StarTriple::new(
            StarTerm::quoted_triple(bad_inner),
            iri("http://example.org/source"),
            iri("http://example.org/db"),
        );

        let report = checker.check_graph(&[annotation_triple, bad_triple]);
        assert!(!report.conformant);
        assert!(report.count_severity(&Severity::Error) > 0);
    }

    #[test]
    fn test_compliance_report_count_severity() {
        let violations = vec![
            ComplianceViolation::error("STAR-001", "e1"),
            ComplianceViolation::error("STAR-002", "e2"),
            ComplianceViolation::warning("STAR-003", "w1"),
            ComplianceViolation::info("STAR-004", "i1"),
        ];
        let report = ComplianceReport::new(violations);
        assert!(!report.conformant);
        assert_eq!(report.count_severity(&Severity::Error), 2);
        assert_eq!(report.count_severity(&Severity::Warning), 1);
        assert_eq!(report.count_severity(&Severity::Info), 1);
    }

    #[test]
    fn test_compliance_report_no_errors_is_conformant() {
        let violations = vec![
            ComplianceViolation::warning("STAR-003", "w1"),
            ComplianceViolation::info("STAR-004", "i1"),
        ];
        let report = ComplianceReport::new(violations);
        assert!(
            report.conformant,
            "warnings/info alone should be conformant"
        );
    }

    #[test]
    fn test_check_triple_convenience_method() {
        let checker = W3cRdfStarChecker::new();

        // Valid triple with quoted subject
        let triple = StarTriple::new(
            StarTerm::quoted_triple(ground_triple()),
            iri("http://example.org/p"),
            lit("v"),
        );
        let violations = checker.check_triple(&triple);
        assert!(
            violations.is_empty(),
            "valid triple should have no violations"
        );
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn test_checker_custom_depth() {
        // max_nesting_depth=1 means depth > 1 triggers a violation
        // Build a triple with nesting depth 2
        let checker = W3cRdfStarChecker::with_max_nesting_depth(1);
        let inner = ground_triple();
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://example.org/p"),
            lit("v"),
        );
        // outer has nesting_depth = 1 (not > 1) — need one more level
        let double = StarTriple::new(
            StarTerm::quoted_triple(outer),
            iri("http://example.org/p"),
            lit("v"),
        );
        // double has nesting_depth = 2 > max (1) → violation
        let violations = checker.check_nesting_depth(&double);
        assert!(violations.iter().any(|v| v.rule_id == "STAR-004"));
    }

    #[test]
    fn test_report_errors_and_warnings_accessors() {
        let violations = vec![
            ComplianceViolation::error("STAR-001", "err"),
            ComplianceViolation::warning("STAR-003", "warn"),
        ];
        let report = ComplianceReport::new(violations);
        assert_eq!(report.errors().len(), 1);
        assert_eq!(report.warnings().len(), 1);
    }
}
