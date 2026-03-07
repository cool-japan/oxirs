//! # SHACL Validation Report Formatter
//!
//! Renders `ValidationReport`s in multiple output formats: Turtle, JSON,
//! plain text, CSV, and HTML.  All serialisation is hand-rolled (no external
//! serde or template dependencies) to keep the module self-contained.

use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Severity
// ─────────────────────────────────────────────────────────────────────────────

/// SHACL result severity level.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational finding — the graph is still considered conformant.
    Info,
    /// Advisory finding — the graph may have issues.
    Warning,
    /// Hard constraint failure — the graph is non-conformant.
    Violation,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Violation => write!(f, "Violation"),
            Severity::Warning => write!(f, "Warning"),
            Severity::Info => write!(f, "Info"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ValidationResult
// ─────────────────────────────────────────────────────────────────────────────

/// A single SHACL validation result.
#[derive(Clone, Debug)]
pub struct ValidationResult {
    /// The RDF node that was validated.
    pub focus_node: String,
    /// The property path that was evaluated (may be absent for node-level constraints).
    pub path: Option<String>,
    /// The offending value, if any.
    pub value: Option<String>,
    /// The SHACL shape that produced this result.
    pub source_shape: String,
    /// Severity of the result.
    pub severity: Severity,
    /// Human-readable message.
    pub message: String,
    /// SHACL constraint component IRI (e.g. `sh:MinCountConstraintComponent`).
    pub constraint_component: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// ValidationReport
// ─────────────────────────────────────────────────────────────────────────────

/// A SHACL validation report containing zero or more results.
#[derive(Clone, Debug)]
pub struct ValidationReport {
    /// `true` when there are no `Violation`-level results.
    pub conforms: bool,
    /// All validation results (may include Info and Warning even when conformant).
    pub results: Vec<ValidationResult>,
}

impl ValidationReport {
    /// Build a report; `conforms` is `true` iff there are no `Violation` results.
    pub fn new(results: Vec<ValidationResult>) -> Self {
        let conforms = results.iter().all(|r| r.severity != Severity::Violation);
        Self { conforms, results }
    }

    /// Return only `Violation`-level results.
    pub fn violations(&self) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.severity == Severity::Violation)
            .collect()
    }

    /// Return only `Warning`-level results.
    pub fn warnings(&self) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.severity == Severity::Warning)
            .collect()
    }

    /// Return only `Info`-level results.
    pub fn infos(&self) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.severity == Severity::Info)
            .collect()
    }

    /// Return all results associated with the given focus node.
    pub fn by_focus_node(&self, node: &str) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.focus_node == node)
            .collect()
    }

    /// Return all results associated with the given source shape.
    pub fn by_shape(&self, shape: &str) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.source_shape == shape)
            .collect()
    }

    /// Total number of results.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Number of `Violation`-level results.
    pub fn violation_count(&self) -> usize {
        self.violations().len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FormatError
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when an unknown format string is requested.
#[derive(Debug, Clone, PartialEq)]
pub struct FormatError(pub String);

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown report format: {}", self.0)
    }
}

impl std::error::Error for FormatError {}

// ─────────────────────────────────────────────────────────────────────────────
// ReportFormatter
// ─────────────────────────────────────────────────────────────────────────────

/// Serialises `ValidationReport` values to various text formats.
#[derive(Debug, Default)]
pub struct ReportFormatter;

impl ReportFormatter {
    /// Create a new formatter.
    pub fn new() -> Self {
        Self
    }

    // ── Turtle ───────────────────────────────────────────────────────────────

    /// Emit the report as a Turtle (`.ttl`) document.
    pub fn to_turtle(report: &ValidationReport) -> String {
        let mut s = String::new();
        s.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
        s.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");
        s.push_str("[] a sh:ValidationReport ;\n");
        let conforms = if report.conforms { "true" } else { "false" };
        s.push_str(&format!("    sh:conforms {conforms} ;\n"));
        if report.results.is_empty() {
            s.push_str("    sh:result () .\n");
        } else {
            for (i, r) in report.results.iter().enumerate() {
                let last = i == report.results.len() - 1;
                let sep = if last { "." } else { ";" };
                s.push_str("    sh:result [\n");
                s.push_str("        a sh:ValidationResult ;\n");
                s.push_str(&format!(
                    "        sh:focusNode <{}> ;\n",
                    escape_iri(&r.focus_node)
                ));
                if let Some(p) = &r.path {
                    s.push_str(&format!("        sh:resultPath <{}> ;\n", escape_iri(p)));
                }
                if let Some(v) = &r.value {
                    s.push_str(&format!("        sh:value \"{}\" ;\n", escape_literal(v)));
                }
                s.push_str(&format!(
                    "        sh:sourceShape <{}> ;\n",
                    escape_iri(&r.source_shape)
                ));
                let sev_iri = match r.severity {
                    Severity::Violation => "sh:Violation",
                    Severity::Warning => "sh:Warning",
                    Severity::Info => "sh:Info",
                };
                s.push_str(&format!("        sh:resultSeverity {sev_iri} ;\n"));
                s.push_str(&format!(
                    "        sh:resultMessage \"{}\" ;\n",
                    escape_literal(&r.message)
                ));
                s.push_str(&format!(
                    "        sh:sourceConstraintComponent <{}>\n",
                    escape_iri(&r.constraint_component)
                ));
                s.push_str(&format!("    ] {sep}\n"));
            }
        }
        s
    }

    // ── JSON ─────────────────────────────────────────────────────────────────

    /// Emit the report as a hand-rolled JSON document.
    pub fn to_json(report: &ValidationReport) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        let conforms = if report.conforms { "true" } else { "false" };
        s.push_str(&format!("  \"conforms\": {conforms},\n"));
        s.push_str("  \"results\": [\n");
        for (i, r) in report.results.iter().enumerate() {
            let last = i == report.results.len() - 1;
            s.push_str("    {\n");
            s.push_str(&format!(
                "      \"focusNode\": \"{}\",\n",
                json_escape(&r.focus_node)
            ));
            let path_val = r.path.as_deref().map(json_escape).unwrap_or_default();
            s.push_str(&format!("      \"path\": \"{path_val}\",\n"));
            let value_val = r.value.as_deref().map(json_escape).unwrap_or_default();
            s.push_str(&format!("      \"value\": \"{value_val}\",\n"));
            s.push_str(&format!(
                "      \"sourceShape\": \"{}\",\n",
                json_escape(&r.source_shape)
            ));
            s.push_str(&format!("      \"severity\": \"{}\",\n", r.severity));
            s.push_str(&format!(
                "      \"message\": \"{}\",\n",
                json_escape(&r.message)
            ));
            s.push_str(&format!(
                "      \"constraintComponent\": \"{}\"",
                json_escape(&r.constraint_component)
            ));
            s.push('\n');
            if last {
                s.push_str("    }\n");
            } else {
                s.push_str("    },\n");
            }
        }
        s.push_str("  ]\n");
        s.push('}');
        s
    }

    // ── Text ─────────────────────────────────────────────────────────────────

    /// Emit the report as a human-readable plain-text summary.
    pub fn to_text(report: &ValidationReport) -> String {
        let mut s = String::new();
        let status = if report.conforms {
            "CONFORMS"
        } else {
            "NON-CONFORMANT"
        };
        s.push_str(&format!("SHACL Validation Report [{status}]\n"));
        s.push_str(&format!("Total results: {}\n", report.results.len()));
        s.push_str(&format!("  Violations : {}\n", report.violation_count()));
        s.push_str(&format!("  Warnings   : {}\n", report.warnings().len()));
        s.push_str(&format!("  Infos      : {}\n\n", report.infos().len()));
        for (i, r) in report.results.iter().enumerate() {
            s.push_str(&format!(
                "[{}] {} | Focus: {} | Shape: {}\n",
                i + 1,
                r.severity,
                r.focus_node,
                r.source_shape
            ));
            if let Some(p) = &r.path {
                s.push_str(&format!("    Path: {p}\n"));
            }
            if let Some(v) = &r.value {
                s.push_str(&format!("    Value: {v}\n"));
            }
            s.push_str(&format!("    Message: {}\n", r.message));
            s.push_str(&format!("    Component: {}\n", r.constraint_component));
        }
        s
    }

    // ── CSV ──────────────────────────────────────────────────────────────────

    /// Emit the report as CSV with the header:
    /// `focusNode,path,value,severity,message,shape`
    pub fn to_csv(report: &ValidationReport) -> String {
        let mut s = String::new();
        s.push_str("focusNode,path,value,severity,message,shape\n");
        for r in &report.results {
            let path = r.path.as_deref().unwrap_or("");
            let value = r.value.as_deref().unwrap_or("");
            s.push_str(&format!(
                "{},{},{},{},{},{}\n",
                csv_field(&r.focus_node),
                csv_field(path),
                csv_field(value),
                csv_field(&r.severity.to_string()),
                csv_field(&r.message),
                csv_field(&r.source_shape),
            ));
        }
        s
    }

    // ── HTML ─────────────────────────────────────────────────────────────────

    /// Emit the report as a simple HTML table.
    pub fn to_html(report: &ValidationReport) -> String {
        let status = if report.conforms {
            "Conforms"
        } else {
            "Non-Conformant"
        };
        let mut s = String::new();
        s.push_str("<!DOCTYPE html>\n<html>\n<head><title>SHACL Validation Report</title></head>\n<body>\n");
        s.push_str(&format!("<h1>SHACL Validation Report: {status}</h1>\n"));
        s.push_str(&format!("<p>Total results: {}</p>\n", report.results.len()));
        if report.results.is_empty() {
            s.push_str("<p>No results.</p>\n");
        } else {
            s.push_str("<table border=\"1\">\n");
            s.push_str("<tr><th>focusNode</th><th>path</th><th>value</th><th>severity</th><th>message</th><th>shape</th></tr>\n");
            for r in &report.results {
                s.push_str("<tr>");
                s.push_str(&format!("<td>{}</td>", html_escape(&r.focus_node)));
                s.push_str(&format!(
                    "<td>{}</td>",
                    html_escape(r.path.as_deref().unwrap_or(""))
                ));
                s.push_str(&format!(
                    "<td>{}</td>",
                    html_escape(r.value.as_deref().unwrap_or(""))
                ));
                s.push_str(&format!(
                    "<td>{}</td>",
                    html_escape(&r.severity.to_string())
                ));
                s.push_str(&format!("<td>{}</td>", html_escape(&r.message)));
                s.push_str(&format!("<td>{}</td>", html_escape(&r.source_shape)));
                s.push_str("</tr>\n");
            }
            s.push_str("</table>\n");
        }
        s.push_str("</body>\n</html>");
        s
    }

    // ── Format dispatch ───────────────────────────────────────────────────────

    /// Dispatch to the appropriate formatter based on `fmt` string.
    ///
    /// Accepted values: `"turtle"`, `"json"`, `"text"`, `"csv"`, `"html"`.
    pub fn format(report: &ValidationReport, fmt: &str) -> Result<String, FormatError> {
        match fmt {
            "turtle" => Ok(Self::to_turtle(report)),
            "json" => Ok(Self::to_json(report)),
            "text" => Ok(Self::to_text(report)),
            "csv" => Ok(Self::to_csv(report)),
            "html" => Ok(Self::to_html(report)),
            other => Err(FormatError(other.to_string())),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// String escaping helpers
// ─────────────────────────────────────────────────────────────────────────────

fn escape_iri(s: &str) -> String {
    s.replace('>', "\\>")
}

fn escape_literal(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Quote a CSV field, wrapping in double-quotes if it contains a comma, quote, or newline.
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn violation(focus: &str, shape: &str, msg: &str) -> ValidationResult {
        ValidationResult {
            focus_node: focus.to_string(),
            path: Some("http://example.org/name".to_string()),
            value: Some("bad-value".to_string()),
            source_shape: shape.to_string(),
            severity: Severity::Violation,
            message: msg.to_string(),
            constraint_component: "sh:MinCountConstraintComponent".to_string(),
        }
    }

    fn warning(focus: &str, shape: &str, msg: &str) -> ValidationResult {
        ValidationResult {
            focus_node: focus.to_string(),
            path: None,
            value: None,
            source_shape: shape.to_string(),
            severity: Severity::Warning,
            message: msg.to_string(),
            constraint_component: "sh:PatternConstraintComponent".to_string(),
        }
    }

    fn info_result(focus: &str, shape: &str, msg: &str) -> ValidationResult {
        ValidationResult {
            focus_node: focus.to_string(),
            path: None,
            value: None,
            source_shape: shape.to_string(),
            severity: Severity::Info,
            message: msg.to_string(),
            constraint_component: "sh:NodeKindConstraintComponent".to_string(),
        }
    }

    fn sample_report() -> ValidationReport {
        ValidationReport::new(vec![
            violation(
                "http://ex.org/alice",
                "http://ex.org/PersonShape",
                "Missing name",
            ),
            warning(
                "http://ex.org/bob",
                "http://ex.org/PersonShape",
                "Unusual email format",
            ),
        ])
    }

    fn empty_report() -> ValidationReport {
        ValidationReport::new(vec![])
    }

    fn conforming_report() -> ValidationReport {
        ValidationReport::new(vec![info_result(
            "http://ex.org/alice",
            "http://ex.org/PersonShape",
            "Note",
        )])
    }

    // ── Severity ─────────────────────────────────────────────────────────────

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Violation.to_string(), "Violation");
        assert_eq!(Severity::Warning.to_string(), "Warning");
        assert_eq!(Severity::Info.to_string(), "Info");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Violation);
    }

    #[test]
    fn test_severity_clone_eq() {
        let s = Severity::Violation;
        assert_eq!(s.clone(), Severity::Violation);
    }

    #[test]
    fn test_severity_hash() {
        let mut map = HashMap::new();
        map.insert(Severity::Violation, 1u32);
        assert_eq!(map.get(&Severity::Violation), Some(&1));
    }

    // ── ValidationReport::new ─────────────────────────────────────────────────

    #[test]
    fn test_report_conforms_true_when_no_violations() {
        let report = conforming_report();
        assert!(report.conforms);
    }

    #[test]
    fn test_report_conforms_false_when_violations() {
        let report = sample_report();
        assert!(!report.conforms);
    }

    #[test]
    fn test_report_empty_conforms() {
        assert!(empty_report().conforms);
    }

    // ── Filter methods ────────────────────────────────────────────────────────

    #[test]
    fn test_violations_returns_only_violations() {
        let report = sample_report();
        assert_eq!(report.violations().len(), 1);
        assert_eq!(report.violations()[0].severity, Severity::Violation);
    }

    #[test]
    fn test_warnings_returns_only_warnings() {
        let report = sample_report();
        assert_eq!(report.warnings().len(), 1);
        assert_eq!(report.warnings()[0].severity, Severity::Warning);
    }

    #[test]
    fn test_infos_returns_only_infos() {
        let report = conforming_report();
        assert_eq!(report.infos().len(), 1);
    }

    #[test]
    fn test_by_focus_node() {
        let report = sample_report();
        let alice = report.by_focus_node("http://ex.org/alice");
        assert_eq!(alice.len(), 1);
    }

    #[test]
    fn test_by_focus_node_not_found() {
        let report = sample_report();
        assert!(report.by_focus_node("http://ex.org/nobody").is_empty());
    }

    #[test]
    fn test_by_shape() {
        let report = sample_report();
        let results = report.by_shape("http://ex.org/PersonShape");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_result_count() {
        assert_eq!(sample_report().result_count(), 2);
        assert_eq!(empty_report().result_count(), 0);
    }

    #[test]
    fn test_violation_count() {
        assert_eq!(sample_report().violation_count(), 1);
    }

    // ── Turtle format ─────────────────────────────────────────────────────────

    #[test]
    fn test_to_turtle_contains_prefix() {
        let ttl = ReportFormatter::to_turtle(&sample_report());
        assert!(ttl.contains("@prefix sh:"));
    }

    #[test]
    fn test_to_turtle_contains_conforms_false() {
        let ttl = ReportFormatter::to_turtle(&sample_report());
        assert!(ttl.contains("sh:conforms false"));
    }

    #[test]
    fn test_to_turtle_contains_conforms_true() {
        let ttl = ReportFormatter::to_turtle(&empty_report());
        assert!(ttl.contains("sh:conforms true"));
    }

    #[test]
    fn test_to_turtle_contains_focus_node() {
        let ttl = ReportFormatter::to_turtle(&sample_report());
        assert!(ttl.contains("alice"));
    }

    #[test]
    fn test_to_turtle_contains_severity_iri() {
        let ttl = ReportFormatter::to_turtle(&sample_report());
        assert!(ttl.contains("sh:Violation"));
    }

    #[test]
    fn test_to_turtle_empty_report() {
        let ttl = ReportFormatter::to_turtle(&empty_report());
        assert!(ttl.contains("sh:result ()"));
    }

    // ── JSON format ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_json_contains_conforms() {
        let json = ReportFormatter::to_json(&sample_report());
        assert!(json.contains("\"conforms\": false"));
    }

    #[test]
    fn test_to_json_contains_results_array() {
        let json = ReportFormatter::to_json(&sample_report());
        assert!(json.contains("\"results\""));
    }

    #[test]
    fn test_to_json_contains_focus_node() {
        let json = ReportFormatter::to_json(&sample_report());
        assert!(json.contains("alice"));
    }

    #[test]
    fn test_to_json_empty_results() {
        let json = ReportFormatter::to_json(&empty_report());
        assert!(json.contains("\"conforms\": true"));
        assert!(json.contains("\"results\": []") || json.contains("\"results\": [\n  ]"));
    }

    #[test]
    fn test_to_json_severity_field() {
        let json = ReportFormatter::to_json(&sample_report());
        assert!(json.contains("\"severity\": \"Violation\""));
    }

    // ── Text format ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_text_header_non_conformant() {
        let text = ReportFormatter::to_text(&sample_report());
        assert!(text.contains("NON-CONFORMANT"));
    }

    #[test]
    fn test_to_text_header_conforms() {
        let text = ReportFormatter::to_text(&empty_report());
        assert!(text.contains("CONFORMS"));
    }

    #[test]
    fn test_to_text_total_results() {
        let text = ReportFormatter::to_text(&sample_report());
        assert!(text.contains("Total results: 2"));
    }

    #[test]
    fn test_to_text_violation_count() {
        let text = ReportFormatter::to_text(&sample_report());
        assert!(text.contains("Violations"));
    }

    #[test]
    fn test_to_text_contains_message() {
        let text = ReportFormatter::to_text(&sample_report());
        assert!(text.contains("Missing name"));
    }

    // ── CSV format ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_csv_header() {
        let csv = ReportFormatter::to_csv(&sample_report());
        assert!(csv.starts_with("focusNode,path,value,severity,message,shape\n"));
    }

    #[test]
    fn test_to_csv_row_count() {
        let csv = ReportFormatter::to_csv(&sample_report());
        let lines: Vec<&str> = csv.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 3); // header + 2 results
    }

    #[test]
    fn test_to_csv_contains_focus_node() {
        let csv = ReportFormatter::to_csv(&sample_report());
        assert!(csv.contains("alice"));
    }

    #[test]
    fn test_to_csv_empty_report_only_header() {
        let csv = ReportFormatter::to_csv(&empty_report());
        let lines: Vec<&str> = csv.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_to_csv_quoting_commas() {
        let mut r = violation(
            "http://ex.org/x",
            "http://ex.org/S",
            "message, with, commas",
        );
        r.path = None;
        r.value = None;
        let report = ValidationReport::new(vec![r]);
        let csv = ReportFormatter::to_csv(&report);
        assert!(csv.contains('"'));
    }

    // ── HTML format ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_html_contains_doctype() {
        let html = ReportFormatter::to_html(&sample_report());
        assert!(html.contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_to_html_contains_table() {
        let html = ReportFormatter::to_html(&sample_report());
        assert!(html.contains("<table"));
    }

    #[test]
    fn test_to_html_contains_focus_node() {
        let html = ReportFormatter::to_html(&sample_report());
        assert!(html.contains("alice"));
    }

    #[test]
    fn test_to_html_empty_report() {
        let html = ReportFormatter::to_html(&empty_report());
        assert!(html.contains("No results."));
    }

    #[test]
    fn test_to_html_escapes_html_entities() {
        let mut r = violation("http://ex.org/<x>", "http://ex.org/S", "msg");
        r.path = None;
        r.value = None;
        let report = ValidationReport::new(vec![r]);
        let html = ReportFormatter::to_html(&report);
        assert!(html.contains("&lt;x&gt;"));
    }

    // ── Format dispatch ───────────────────────────────────────────────────────

    #[test]
    fn test_format_turtle() {
        let r = ReportFormatter::format(&sample_report(), "turtle");
        assert!(r.is_ok());
        assert!(r.unwrap().contains("sh:ValidationReport"));
    }

    #[test]
    fn test_format_json() {
        let r = ReportFormatter::format(&sample_report(), "json");
        assert!(r.is_ok());
    }

    #[test]
    fn test_format_text() {
        let r = ReportFormatter::format(&sample_report(), "text");
        assert!(r.is_ok());
    }

    #[test]
    fn test_format_csv() {
        let r = ReportFormatter::format(&sample_report(), "csv");
        assert!(r.is_ok());
    }

    #[test]
    fn test_format_html() {
        let r = ReportFormatter::format(&sample_report(), "html");
        assert!(r.is_ok());
    }

    #[test]
    fn test_format_unknown() {
        let r = ReportFormatter::format(&sample_report(), "xml");
        assert!(r.is_err());
        let err = r.unwrap_err();
        assert_eq!(err.0, "xml");
    }

    #[test]
    fn test_format_error_display() {
        let e = FormatError("rdf".into());
        assert!(e.to_string().contains("rdf"));
    }

    #[test]
    fn test_format_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(FormatError("x".into()));
        assert!(!e.to_string().is_empty());
    }

    // ── ReportFormatter::new ──────────────────────────────────────────────────

    #[test]
    fn test_report_formatter_new() {
        let _f = ReportFormatter::new();
    }

    // ── Multi-severity report ─────────────────────────────────────────────────

    #[test]
    fn test_mixed_severity_report_counts() {
        let report = ValidationReport::new(vec![
            violation("n1", "s1", "v1"),
            warning("n2", "s1", "w1"),
            info_result("n3", "s1", "i1"),
        ]);
        assert!(!report.conforms);
        assert_eq!(report.violation_count(), 1);
        assert_eq!(report.warnings().len(), 1);
        assert_eq!(report.infos().len(), 1);
        assert_eq!(report.result_count(), 3);
    }
}
