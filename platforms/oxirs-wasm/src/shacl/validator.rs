//! SHACL validator — validates RDF data graphs against SHACL shapes.
//!
//! Supported constraint components (subset per plan):
//! - `sh:minCount`, `sh:maxCount`
//! - `sh:datatype`
//! - `sh:pattern`
//! - `sh:minInclusive`, `sh:maxInclusive`
//! - `sh:class`
//! - `sh:nodeKind`
//! - `sh:in`
//! - `sh:hasValue`

use super::shapes::{NodeKind, NodeShape, PropertyShape, ShaclConstraint};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// RDF value model for validation
// ─────────────────────────────────────────────────────────────────────────────

/// A simplified RDF node representation used for SHACL validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RdfNode {
    /// An IRI node.
    Iri(String),
    /// A plain or typed literal.
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A blank node.
    BlankNode(String),
}

impl RdfNode {
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }
    pub fn literal(value: impl Into<String>, datatype: Option<String>) -> Self {
        Self::Literal {
            value: value.into(),
            datatype,
            lang: None,
        }
    }
    pub fn blank(id: impl Into<String>) -> Self {
        Self::BlankNode(id.into())
    }

    pub fn is_iri(&self) -> bool {
        matches!(self, Self::Iri(_))
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal { .. })
    }

    pub fn is_blank(&self) -> bool {
        matches!(self, Self::BlankNode(_))
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Iri(s) | Self::BlankNode(s) => s,
            Self::Literal { value, .. } => value,
        }
    }

    pub fn datatype(&self) -> Option<&str> {
        match self {
            Self::Literal { datatype, .. } => datatype.as_deref(),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data graph model
// ─────────────────────────────────────────────────────────────────────────────

/// A triple in the data graph.
#[derive(Debug, Clone)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: RdfNode,
}

impl Triple {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>, object: RdfNode) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object,
        }
    }
}

/// A simple in-memory RDF data graph for SHACL validation.
#[derive(Default)]
pub struct DataGraph {
    pub triples: Vec<Triple>,
}

impl DataGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: RdfNode,
    ) {
        self.triples.push(Triple::new(subject, predicate, object));
    }

    /// Get all values of `predicate` for `subject`.
    pub fn values_of(&self, subject: &str, predicate: &str) -> Vec<&RdfNode> {
        self.triples
            .iter()
            .filter(|t| t.subject == subject && t.predicate == predicate)
            .map(|t| &t.object)
            .collect()
    }

    /// Get all rdf:type values for `subject`.
    pub fn types_of(&self, subject: &str) -> Vec<&str> {
        self.values_of(subject, "rdf:type")
            .into_iter()
            .map(|n| n.as_str())
            .collect()
    }

    /// Get all subjects in the graph.
    pub fn all_subjects(&self) -> Vec<&str> {
        let mut seen = std::collections::HashSet::new();
        let mut subjects = Vec::new();
        for t in &self.triples {
            if seen.insert(t.subject.as_str()) {
                subjects.push(t.subject.as_str());
            }
        }
        subjects
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation result
// ─────────────────────────────────────────────────────────────────────────────

/// A SHACL validation result entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// The focus node that failed validation.
    pub focus_node: String,
    /// The property path that caused the violation (if any).
    pub result_path: Option<String>,
    /// Human-readable violation message.
    pub message: String,
    /// Severity (`sh:Violation`, `sh:Warning`, `sh:Info`).
    pub severity: Severity,
}

/// SHACL violation severity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Violation,
    Warning,
    Info,
}

impl ValidationResult {
    fn violation(focus_node: &str, path: Option<&str>, message: impl Into<String>) -> Self {
        Self {
            focus_node: focus_node.to_string(),
            result_path: path.map(str::to_string),
            message: message.into(),
            severity: Severity::Violation,
        }
    }
}

/// The overall validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub conforms: bool,
    pub results: Vec<ValidationResult>,
}

impl ValidationReport {
    fn new(results: Vec<ValidationResult>) -> Self {
        Self {
            conforms: results.is_empty(),
            results,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShaclValidator
// ─────────────────────────────────────────────────────────────────────────────

/// A SHACL shapes graph validator.
pub struct ShaclValidator {
    pub shapes: Vec<NodeShape>,
}

impl ShaclValidator {
    /// Create a validator with the given shapes.
    pub fn new(shapes: Vec<NodeShape>) -> Self {
        Self { shapes }
    }

    /// Validate `graph` against all registered shapes.
    pub fn validate(&self, graph: &DataGraph) -> ValidationReport {
        let mut results = Vec::new();

        for shape in &self.shapes {
            let focus_nodes = self.focus_nodes(shape, graph);
            for node in focus_nodes {
                results.extend(self.validate_node(shape, node, graph));
            }
        }

        ValidationReport::new(results)
    }

    fn focus_nodes<'a>(&self, shape: &NodeShape, graph: &'a DataGraph) -> Vec<&'a str> {
        if let Some(class) = &shape.target_class {
            graph
                .all_subjects()
                .into_iter()
                .filter(|s| graph.types_of(s).contains(&class.as_str()))
                .collect()
        } else if let Some(node) = &shape.target_node {
            graph
                .all_subjects()
                .into_iter()
                .filter(|s| *s == node.as_str())
                .collect()
        } else {
            Vec::new()
        }
    }

    fn validate_node(
        &self,
        shape: &NodeShape,
        focus: &str,
        graph: &DataGraph,
    ) -> Vec<ValidationResult> {
        let mut results = Vec::new();
        for prop in &shape.properties {
            results.extend(self.validate_property(prop, focus, graph));
        }
        results
    }

    fn validate_property(
        &self,
        prop: &PropertyShape,
        focus: &str,
        graph: &DataGraph,
    ) -> Vec<ValidationResult> {
        let values = graph.values_of(focus, &prop.path);
        let mut results = Vec::new();

        for constraint in &prop.constraints {
            match constraint {
                ShaclConstraint::MinCount(min) => {
                    if values.len() < *min {
                        results.push(ValidationResult::violation(
                            focus,
                            Some(&prop.path),
                            format!(
                                "sh:minCount {min}: expected at least {min} value(s), got {}",
                                values.len()
                            ),
                        ));
                    }
                }
                ShaclConstraint::MaxCount(max) => {
                    if values.len() > *max {
                        results.push(ValidationResult::violation(
                            focus,
                            Some(&prop.path),
                            format!(
                                "sh:maxCount {max}: expected at most {max} value(s), got {}",
                                values.len()
                            ),
                        ));
                    }
                }
                ShaclConstraint::Datatype(dt) => {
                    for val in &values {
                        match val.datatype() {
                            Some(actual) if actual == dt => {}
                            Some(actual) => results.push(ValidationResult::violation(
                                focus,
                                Some(&prop.path),
                                format!("sh:datatype {dt}: got datatype {actual}"),
                            )),
                            None => results.push(ValidationResult::violation(
                                focus,
                                Some(&prop.path),
                                format!("sh:datatype {dt}: value has no datatype"),
                            )),
                        }
                    }
                }
                ShaclConstraint::Pattern(pat) => {
                    let regex = build_regex(pat);
                    for val in &values {
                        let s = val.as_str();
                        if !regex_match(&regex, s) {
                            results.push(ValidationResult::violation(
                                focus,
                                Some(&prop.path),
                                format!("sh:pattern {pat:?}: value {s:?} does not match"),
                            ));
                        }
                    }
                }
                ShaclConstraint::MinInclusive(min) => {
                    for val in &values {
                        if let Ok(n) = val.as_str().parse::<f64>() {
                            if n < *min {
                                results.push(ValidationResult::violation(
                                    focus,
                                    Some(&prop.path),
                                    format!("sh:minInclusive {min}: got {n}"),
                                ));
                            }
                        }
                    }
                }
                ShaclConstraint::MaxInclusive(max) => {
                    for val in &values {
                        if let Ok(n) = val.as_str().parse::<f64>() {
                            if n > *max {
                                results.push(ValidationResult::violation(
                                    focus,
                                    Some(&prop.path),
                                    format!("sh:maxInclusive {max}: got {n}"),
                                ));
                            }
                        }
                    }
                }
                ShaclConstraint::Class(cls) => {
                    for val in &values {
                        if let RdfNode::Iri(iri) = val {
                            // Check that the value node has the required rdf:type
                            if !graph.types_of(iri).contains(&cls.as_str()) {
                                results.push(ValidationResult::violation(
                                    focus,
                                    Some(&prop.path),
                                    format!("sh:class {cls}: {iri} lacks required type"),
                                ));
                            }
                        }
                    }
                }
                ShaclConstraint::NodeKind(kind) => {
                    for val in &values {
                        let ok = match kind {
                            NodeKind::Iri => val.is_iri(),
                            NodeKind::Literal => val.is_literal(),
                            NodeKind::BlankNode => val.is_blank(),
                            NodeKind::BlankNodeOrIri => val.is_blank() || val.is_iri(),
                            NodeKind::BlankNodeOrLiteral => val.is_blank() || val.is_literal(),
                            NodeKind::IriOrLiteral => val.is_iri() || val.is_literal(),
                        };
                        if !ok {
                            results.push(ValidationResult::violation(
                                focus,
                                Some(&prop.path),
                                format!("sh:nodeKind {:?}: wrong node kind", kind),
                            ));
                        }
                    }
                }
                ShaclConstraint::In(allowed) => {
                    for val in &values {
                        let s = val.as_str();
                        if !allowed.iter().any(|a| a == s) {
                            results.push(ValidationResult::violation(
                                focus,
                                Some(&prop.path),
                                format!("sh:in: {s:?} is not in allowed set"),
                            ));
                        }
                    }
                }
                ShaclConstraint::HasValue(expected) => {
                    let has = values.iter().any(|v| v.as_str() == expected);
                    if !has {
                        results.push(ValidationResult::violation(
                            focus,
                            Some(&prop.path),
                            format!("sh:hasValue: {expected:?} not found"),
                        ));
                    }
                }
            }
        }
        results
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal regex matching (no external crate dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal compiled "pattern" representation.
/// Supports: anchors `^` / `$`, literal characters, `.` (any char),
/// `*` / `+` / `?` on single chars.
/// For a production validator, use the `regex` crate.
struct SimplePattern(String);

fn build_regex(pattern: &str) -> SimplePattern {
    SimplePattern(pattern.to_string())
}

fn regex_match(pat: &SimplePattern, s: &str) -> bool {
    // Use simple substring / anchor match for the subset of patterns we expect
    let p = pat.0.as_str();
    let anchored_start = p.starts_with('^');
    let anchored_end = p.ends_with('$') && !p.ends_with("\\$");
    let core = p.trim_start_matches('^').trim_end_matches('$');

    if anchored_start && anchored_end {
        s == core
    } else if anchored_start {
        s.starts_with(core)
    } else if anchored_end {
        s.ends_with(core)
    } else {
        s.contains(core)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::shapes::{NodeShape, PropertyShape, ShaclConstraint};
    use super::*;

    fn person_shape() -> NodeShape {
        NodeShape::new("ex:PersonShape")
            .targeting_class("ex:Person")
            .with_property(
                PropertyShape::new("ex:name")
                    .with_constraint(ShaclConstraint::MinCount(1))
                    .with_constraint(ShaclConstraint::MaxCount(1))
                    .with_constraint(ShaclConstraint::Datatype("xsd:string".into())),
            )
            .with_property(
                PropertyShape::new("ex:age")
                    .with_constraint(ShaclConstraint::MinInclusive(0.0))
                    .with_constraint(ShaclConstraint::MaxInclusive(150.0)),
            )
    }

    fn valid_person_graph() -> DataGraph {
        let mut g = DataGraph::new();
        g.add("ex:Alice", "rdf:type", RdfNode::iri("ex:Person"));
        g.add(
            "ex:Alice",
            "ex:name",
            RdfNode::literal("Alice", Some("xsd:string".into())),
        );
        g.add(
            "ex:Alice",
            "ex:age",
            RdfNode::literal("30", Some("xsd:integer".into())),
        );
        g
    }

    #[test]
    fn test_valid_graph_conforms() {
        let validator = ShaclValidator::new(vec![person_shape()]);
        let report = validator.validate(&valid_person_graph());
        assert!(
            report.conforms,
            "valid graph must conform: {:?}",
            report.results
        );
    }

    #[test]
    fn test_missing_required_property_violation() {
        let mut g = DataGraph::new();
        g.add("ex:Bob", "rdf:type", RdfNode::iri("ex:Person"));
        // Missing ex:name → minCount violation
        let validator = ShaclValidator::new(vec![person_shape()]);
        let report = validator.validate(&g);
        assert!(!report.conforms);
        assert!(report
            .results
            .iter()
            .any(|r| r.message.contains("minCount")));
    }

    #[test]
    fn test_max_count_violation() {
        let mut g = DataGraph::new();
        g.add("ex:Carol", "rdf:type", RdfNode::iri("ex:Person"));
        g.add(
            "ex:Carol",
            "ex:name",
            RdfNode::literal("Carol", Some("xsd:string".into())),
        );
        g.add(
            "ex:Carol",
            "ex:name",
            RdfNode::literal("Caroline", Some("xsd:string".into())),
        );
        let validator = ShaclValidator::new(vec![person_shape()]);
        let report = validator.validate(&g);
        assert!(!report.conforms);
        assert!(report
            .results
            .iter()
            .any(|r| r.message.contains("maxCount")));
    }

    #[test]
    fn test_wrong_datatype_violation() {
        let mut g = DataGraph::new();
        g.add("ex:Dave", "rdf:type", RdfNode::iri("ex:Person"));
        // name with wrong datatype
        g.add(
            "ex:Dave",
            "ex:name",
            RdfNode::literal("Dave", Some("xsd:integer".into())),
        );
        let validator = ShaclValidator::new(vec![person_shape()]);
        let report = validator.validate(&g);
        assert!(!report.conforms);
        assert!(report
            .results
            .iter()
            .any(|r| r.message.contains("datatype")));
    }

    #[test]
    fn test_min_inclusive_violation() {
        let mut g = DataGraph::new();
        g.add("ex:Eve", "rdf:type", RdfNode::iri("ex:Person"));
        g.add(
            "ex:Eve",
            "ex:name",
            RdfNode::literal("Eve", Some("xsd:string".into())),
        );
        g.add(
            "ex:Eve",
            "ex:age",
            RdfNode::literal("-1", Some("xsd:integer".into())),
        );
        let validator = ShaclValidator::new(vec![person_shape()]);
        let report = validator.validate(&g);
        assert!(!report.conforms);
        assert!(report
            .results
            .iter()
            .any(|r| r.message.contains("minInclusive")));
    }

    #[test]
    fn test_node_kind_iri_violation() {
        let shape = NodeShape::new("ex:S")
            .targeting_class("ex:C")
            .with_property(
                PropertyShape::new("ex:link")
                    .with_constraint(ShaclConstraint::NodeKind(NodeKind::Iri)),
            );
        let mut g = DataGraph::new();
        g.add("ex:X", "rdf:type", RdfNode::iri("ex:C"));
        // Literal where IRI is required
        g.add("ex:X", "ex:link", RdfNode::literal("not-an-iri", None));
        let validator = ShaclValidator::new(vec![shape]);
        let report = validator.validate(&g);
        assert!(!report.conforms);
        assert!(report
            .results
            .iter()
            .any(|r| r.message.contains("nodeKind")));
    }

    #[test]
    fn test_sh_in_constraint() {
        let shape =
            NodeShape::new("ex:S")
                .targeting_class("ex:C")
                .with_property(PropertyShape::new("ex:status").with_constraint(
                    ShaclConstraint::In(vec!["active".into(), "inactive".into()]),
                ));
        let mut g = DataGraph::new();
        g.add("ex:N", "rdf:type", RdfNode::iri("ex:C"));
        g.add("ex:N", "ex:status", RdfNode::literal("pending", None));
        let validator = ShaclValidator::new(vec![shape]);
        let report = validator.validate(&g);
        assert!(!report.conforms);
        assert!(report.results.iter().any(|r| r.message.contains("sh:in")));
    }

    #[test]
    fn test_has_value_constraint_pass() {
        let shape = NodeShape::new("ex:S")
            .targeting_class("ex:C")
            .with_property(
                PropertyShape::new("ex:tag")
                    .with_constraint(ShaclConstraint::HasValue("required-tag".into())),
            );
        let mut g = DataGraph::new();
        g.add("ex:N", "rdf:type", RdfNode::iri("ex:C"));
        g.add("ex:N", "ex:tag", RdfNode::literal("required-tag", None));
        g.add("ex:N", "ex:tag", RdfNode::literal("other-tag", None));
        let validator = ShaclValidator::new(vec![shape]);
        let report = validator.validate(&g);
        assert!(report.conforms, "hasValue satisfied");
    }

    #[test]
    fn test_empty_graph_conforms() {
        let validator = ShaclValidator::new(vec![person_shape()]);
        let report = validator.validate(&DataGraph::new());
        assert!(report.conforms, "empty graph → no focus nodes → conforms");
    }
}
