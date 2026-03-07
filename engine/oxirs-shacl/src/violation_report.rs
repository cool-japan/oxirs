//! SHACL violation report builder and formatter.
//!
//! Provides a builder pattern for accumulating SHACL violations and
//! producing structured `ValidationReport` objects serialisable to
//! Turtle and JSON.

/// Severity of a SHACL violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationSeverity {
    /// `sh:Violation` — data does not conform.
    Violation,
    /// `sh:Warning` — advisory issue.
    Warning,
    /// `sh:Info` — informational notice.
    Info,
}

impl std::fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationSeverity::Violation => write!(f, "sh:Violation"),
            ViolationSeverity::Warning => write!(f, "sh:Warning"),
            ViolationSeverity::Info => write!(f, "sh:Info"),
        }
    }
}

/// A single SHACL validation result entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// The focus node of the violation.
    pub focus_node: String,
    /// Property path, if applicable.
    pub path: Option<String>,
    /// The constraint component IRI.
    pub constraint_component: String,
    /// Human-readable violation message.
    pub message: String,
    /// Severity of this violation.
    pub severity: ViolationSeverity,
    /// Source shape IRI, if known.
    pub source_shape: Option<String>,
    /// The value that caused the violation, if applicable.
    pub value: Option<String>,
}

impl Violation {
    /// Convenience constructor.
    pub fn new(
        focus_node: impl Into<String>,
        constraint_component: impl Into<String>,
        message: impl Into<String>,
        severity: ViolationSeverity,
    ) -> Self {
        Self {
            focus_node: focus_node.into(),
            path: None,
            constraint_component: constraint_component.into(),
            message: message.into(),
            severity,
            source_shape: None,
            value: None,
        }
    }
}

/// A SHACL validation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// `true` when there are no `Violation`-severity results.
    pub conforms: bool,
    /// All collected violation/warning/info results.
    pub violations: Vec<Violation>,
}

/// Builder for [`ValidationReport`].
#[derive(Debug, Default)]
pub struct ViolationReportBuilder {
    violations: Vec<Violation>,
}

impl ViolationReportBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a violation to the builder.
    pub fn add(&mut self, v: Violation) -> &mut Self {
        self.violations.push(v);
        self
    }

    /// Consume the builder and produce a [`ValidationReport`].
    pub fn build(self) -> ValidationReport {
        let conforms = !self
            .violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Violation);
        ValidationReport {
            conforms,
            violations: self.violations,
        }
    }

    /// Returns `true` when no `Violation`-severity results have been added yet.
    pub fn conforms(&self) -> bool {
        !self
            .violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Violation)
    }

    /// Count of `Violation`-severity results.
    pub fn violation_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == ViolationSeverity::Violation)
            .count()
    }

    /// Count of `Warning`-severity results.
    pub fn warning_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == ViolationSeverity::Warning)
            .count()
    }

    /// Return all violations whose focus node matches `node`.
    pub fn violations_for_node(&self, node: &str) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.focus_node == node)
            .collect()
    }

    /// Return all violations whose path matches `path`.
    pub fn violations_for_path(&self, path: &str) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.path.as_deref() == Some(path))
            .collect()
    }

    /// Serialise `report` to a minimal Turtle representation.
    pub fn to_turtle(&self, report: &ValidationReport) -> String {
        let mut out = String::new();
        out.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
        out.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");
        out.push_str("[] a sh:ValidationReport ;\n");
        let conforms_str = if report.conforms { "true" } else { "false" };
        out.push_str(&format!("    sh:conforms {conforms_str} "));

        if report.violations.is_empty() {
            out.push_str(".\n");
        } else {
            out.push_str(";\n");
            let last_idx = report.violations.len() - 1;
            for (i, v) in report.violations.iter().enumerate() {
                out.push_str("    sh:result [\n");
                out.push_str("        a sh:ValidationResult ;\n");
                out.push_str(&format!(
                    "        sh:focusNode <{}> ;\n",
                    v.focus_node
                ));
                if let Some(path) = &v.path {
                    out.push_str(&format!("        sh:resultPath <{path}> ;\n"));
                }
                out.push_str(&format!(
                    "        sh:sourceConstraintComponent <{}> ;\n",
                    v.constraint_component
                ));
                out.push_str(&format!(
                    "        sh:resultMessage \"{}\" ;\n",
                    v.message.replace('"', "\\\"")
                ));
                out.push_str(&format!(
                    "        sh:resultSeverity {} ;\n",
                    v.severity
                ));
                if let Some(val) = &v.value {
                    out.push_str(&format!("        sh:value <{val}> ;\n"));
                }
                if i == last_idx {
                    out.push_str("    ] .\n");
                } else {
                    out.push_str("    ] ;\n");
                }
            }
        }
        out
    }

    /// Serialise `report` to a minimal JSON representation.
    pub fn to_json(&self, report: &ValidationReport) -> String {
        let conforms = report.conforms;
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"conforms\": {conforms},\n"));
        out.push_str("  \"results\": [\n");
        for (i, v) in report.violations.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!(
                "      \"focusNode\": \"{}\",\n",
                v.focus_node
            ));
            if let Some(path) = &v.path {
                out.push_str(&format!("      \"path\": \"{path}\",\n"));
            }
            out.push_str(&format!(
                "      \"constraintComponent\": \"{}\",\n",
                v.constraint_component
            ));
            out.push_str(&format!(
                "      \"message\": \"{}\",\n",
                v.message.replace('"', "\\\"")
            ));
            let sev = match v.severity {
                ViolationSeverity::Violation => "Violation",
                ViolationSeverity::Warning => "Warning",
                ViolationSeverity::Info => "Info",
            };
            out.push_str(&format!("      \"severity\": \"{sev}\""));
            if let Some(val) = &v.value {
                out.push_str(&format!(",\n      \"value\": \"{val}\""));
            }
            out.push('\n');
            if i < report.violations.len() - 1 {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  ]\n");
        out.push('}');
        out
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn v_violation(focus: &str) -> Violation {
        Violation::new(focus, "sh:MinCountConstraintComponent", "too few values", ViolationSeverity::Violation)
    }

    fn v_warning(focus: &str) -> Violation {
        Violation::new(focus, "sh:MaxCountConstraintComponent", "advisory", ViolationSeverity::Warning)
    }

    fn v_info(focus: &str) -> Violation {
        Violation::new(focus, "sh:InfoConstraintComponent", "note", ViolationSeverity::Info)
    }

    // ── ViolationSeverity ────────────────────────────────────────────────────

    #[test]
    fn test_severity_display() {
        assert_eq!(ViolationSeverity::Violation.to_string(), "sh:Violation");
        assert_eq!(ViolationSeverity::Warning.to_string(), "sh:Warning");
        assert_eq!(ViolationSeverity::Info.to_string(), "sh:Info");
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(ViolationSeverity::Violation, ViolationSeverity::Violation);
        assert_ne!(ViolationSeverity::Violation, ViolationSeverity::Warning);
    }

    // ── Violation ────────────────────────────────────────────────────────────

    #[test]
    fn test_violation_new() {
        let v = Violation::new(
            "http://ex.org/node1",
            "sh:MinCountConstraintComponent",
            "Too few values",
            ViolationSeverity::Violation,
        );
        assert_eq!(v.focus_node, "http://ex.org/node1");
        assert!(v.path.is_none());
        assert!(v.source_shape.is_none());
        assert!(v.value.is_none());
    }

    #[test]
    fn test_violation_with_path() {
        let mut v = v_violation("n1");
        v.path = Some("ex:name".to_string());
        assert_eq!(v.path.as_deref(), Some("ex:name"));
    }

    #[test]
    fn test_violation_with_value() {
        let mut v = v_violation("n1");
        v.value = Some("bad_val".to_string());
        assert_eq!(v.value.as_deref(), Some("bad_val"));
    }

    #[test]
    fn test_violation_with_source_shape() {
        let mut v = v_violation("n1");
        v.source_shape = Some("ex:PersonShape".to_string());
        assert!(v.source_shape.is_some());
    }

    // ── ViolationReportBuilder ───────────────────────────────────────────────

    #[test]
    fn test_builder_empty_conforms() {
        let b = ViolationReportBuilder::new();
        assert!(b.conforms());
        assert_eq!(b.violation_count(), 0);
        assert_eq!(b.warning_count(), 0);
    }

    #[test]
    fn test_builder_add_violation_not_conforms() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("n1"));
        assert!(!b.conforms());
        assert_eq!(b.violation_count(), 1);
    }

    #[test]
    fn test_builder_warning_still_conforms() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_warning("n1"));
        assert!(b.conforms());
        assert_eq!(b.warning_count(), 1);
    }

    #[test]
    fn test_builder_info_still_conforms() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_info("n1"));
        assert!(b.conforms());
    }

    #[test]
    fn test_builder_mixed_conforms_false() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_warning("n1"));
        b.add(v_violation("n2"));
        b.add(v_info("n3"));
        assert!(!b.conforms());
        assert_eq!(b.violation_count(), 1);
        assert_eq!(b.warning_count(), 1);
    }

    #[test]
    fn test_builder_build_conforms() {
        let b = ViolationReportBuilder::new();
        let report = b.build();
        assert!(report.conforms);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn test_builder_build_not_conforms() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("n1"));
        let report = b.build();
        assert!(!report.conforms);
        assert_eq!(report.violations.len(), 1);
    }

    #[test]
    fn test_violations_for_node_basic() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("n1"));
        b.add(v_warning("n2"));
        b.add(v_violation("n1"));
        let matching = b.violations_for_node("n1");
        assert_eq!(matching.len(), 2);
    }

    #[test]
    fn test_violations_for_node_none() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("n1"));
        let matching = b.violations_for_node("n99");
        assert!(matching.is_empty());
    }

    #[test]
    fn test_violations_for_path_basic() {
        let mut b = ViolationReportBuilder::new();
        let mut v = v_violation("n1");
        v.path = Some("ex:name".to_string());
        b.add(v.clone());
        b.add(v_violation("n2")); // no path
        let matching = b.violations_for_path("ex:name");
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].focus_node, "n1");
    }

    #[test]
    fn test_violations_for_path_none() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("n1"));
        let matching = b.violations_for_path("ex:missing");
        assert!(matching.is_empty());
    }

    // ── to_turtle ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_turtle_conforms_true() {
        let b = ViolationReportBuilder::new();
        let report = b.build();
        let b2 = ViolationReportBuilder::new();
        let ttl = b2.to_turtle(&report);
        assert!(ttl.contains("sh:conforms true"));
    }

    #[test]
    fn test_to_turtle_conforms_false() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("http://ex.org/n1"));
        let report = b.build();
        let ttl = b.to_turtle(&report);
        assert!(ttl.contains("sh:conforms false"));
    }

    #[test]
    fn test_to_turtle_contains_focus_node() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("http://ex.org/node42"));
        let report = b.build();
        let ttl = b.to_turtle(&report);
        assert!(ttl.contains("http://ex.org/node42"));
    }

    #[test]
    fn test_to_turtle_contains_severity() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_warning("n1"));
        let report = b.build();
        let ttl = b.to_turtle(&report);
        assert!(ttl.contains("sh:Warning"));
    }

    #[test]
    fn test_to_turtle_prefix_present() {
        let b = ViolationReportBuilder::new();
        let report = b.build();
        let b2 = ViolationReportBuilder::new();
        let ttl = b2.to_turtle(&report);
        assert!(ttl.contains("@prefix sh:"));
    }

    // ── to_json ──────────────────────────────────────────────────────────────

    #[test]
    fn test_to_json_conforms_true() {
        let b = ViolationReportBuilder::new();
        let report = b.build();
        let b2 = ViolationReportBuilder::new();
        let json = b2.to_json(&report);
        assert!(json.contains("\"conforms\": true"));
    }

    #[test]
    fn test_to_json_conforms_false() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("n1"));
        let report = b.build();
        let json = b.to_json(&report);
        assert!(json.contains("\"conforms\": false"));
    }

    #[test]
    fn test_to_json_contains_focus_node() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("http://ex.org/person1"));
        let report = b.build();
        let json = b.to_json(&report);
        assert!(json.contains("http://ex.org/person1"));
    }

    #[test]
    fn test_to_json_severity_string() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_info("n1"));
        let report = b.build();
        let json = b.to_json(&report);
        assert!(json.contains("\"severity\": \"Info\""));
    }

    #[test]
    fn test_to_json_empty_results() {
        let b = ViolationReportBuilder::new();
        let report = b.build();
        let b2 = ViolationReportBuilder::new();
        let json = b2.to_json(&report);
        assert!(json.contains("\"results\": []") || json.contains("\"results\": [\n  ]"));
    }

    // ── integration ──────────────────────────────────────────────────────────

    #[test]
    fn test_multiple_violations_report() {
        let mut b = ViolationReportBuilder::new();
        for i in 0..5 {
            b.add(v_violation(&format!("node{i}")));
        }
        assert_eq!(b.violation_count(), 5);
        let report = b.build();
        assert_eq!(report.violations.len(), 5);
        assert!(!report.conforms);
    }

    #[test]
    fn test_chaining_add() {
        let mut b = ViolationReportBuilder::new();
        b.add(v_violation("n1")).add(v_warning("n2"));
        assert_eq!(b.violation_count(), 1);
        assert_eq!(b.warning_count(), 1);
    }

    #[test]
    fn test_violation_equality() {
        let v1 = Violation::new("n1", "comp", "msg", ViolationSeverity::Violation);
        let v2 = Violation::new("n1", "comp", "msg", ViolationSeverity::Violation);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_report_equality() {
        let r1 = ViolationReportBuilder::new().build();
        let r2 = ViolationReportBuilder::new().build();
        assert_eq!(r1, r2);
    }
}
