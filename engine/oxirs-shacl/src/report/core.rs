//\! Core ValidationReport structure and implementation

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::{validation::ValidationViolation, Result, Severity, ShapeId};

use super::{ReportConfig, ReportMetadata, ValidationSummary};

/// SHACL validation report according to W3C specification
///
/// The `ValidationReport` contains the results of SHACL validation, including
/// conformance status, detailed violation information, and metadata about
/// the validation process.
///
/// ## Example
///
/// ```rust
/// use oxirs_shacl::{ValidationReport, ValidationViolation, ReportFormat};
/// use oxirs_core::model::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a new validation report
/// let mut report = ValidationReport::new();
///
/// // Add violations if any are found during validation
/// // report.add_violation(violation);
///
/// // Check if data conforms to all shapes
/// if report.conforms() {
///     println\!("✅ Data is valid\!");
/// } else {
///     println\!("❌ Found {} violations", report.violation_count());
///
///     // Print violations
///     for violation in report.violations() {
///         println\!("Violation at {}: {}",
///             violation.focus_node,
///             violation.result_message.as_deref().unwrap_or("No message")
///         );
///     }
/// }
///
/// // Export report in different formats
/// let json_report = report.to_json()?;
/// let html_report = report.to_html()?;
/// let turtle_report = report.to_turtle()?;
/// # Ok(())
/// # }
/// ```
///
/// ## Report Formats
///
/// The validation report can be exported in multiple formats:
///
/// - **JSON**: Machine-readable format for APIs
/// - **HTML**: Human-readable format with styling
/// - **Turtle/RDF**: W3C standard RDF format
/// - **CSV**: Tabular format for spreadsheet analysis
/// - **Text**: Simple plain text summary
///
/// ## Violation Analysis
///
/// Each violation contains detailed information:
///
/// - **Focus node**: The RDF node that failed validation
/// - **Property path**: The path to the violating value (if applicable)
/// - **Constraint component**: Which SHACL constraint was violated
/// - **Severity**: Error, warning, or info level
/// - **Message**: Human-readable explanation
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether the data conforms to all shapes
    pub conforms: bool,

    /// List of validation violations
    pub violations: Vec<ValidationViolation>,

    /// Report metadata
    pub metadata: ReportMetadata,

    /// Summary statistics
    pub summary: ValidationSummary,
}

impl ValidationReport {
    /// Create a new empty validation report
    pub fn new() -> Self {
        Self {
            conforms: true,
            violations: Vec::new(),
            metadata: ReportMetadata::new(),
            summary: ValidationSummary::default(),
        }
    }

    /// Create a new report with initial metadata
    pub fn with_metadata(metadata: ReportMetadata) -> Self {
        Self {
            conforms: true,
            violations: Vec::new(),
            metadata,
            summary: ValidationSummary::default(),
        }
    }

    /// Create a report from violations
    pub fn from_violations(violations: Vec<ValidationViolation>) -> Self {
        let conforms = violations.is_empty();
        let summary = ValidationSummary::from_violations(&violations);

        Self {
            conforms,
            violations,
            metadata: ReportMetadata::new(),
            summary,
        }
    }

    /// Add a violation to the report
    pub fn add_violation(&mut self, violation: ValidationViolation) {
        self.violations.push(violation);
        self.update_conformance();
        self.update_summary();
    }

    /// Add multiple violations to the report
    pub fn add_violations(&mut self, violations: Vec<ValidationViolation>) {
        self.violations.extend(violations);
        self.update_conformance();
        self.update_summary();
    }

    /// Check if the data conforms to all shapes
    pub fn conforms(&self) -> bool {
        self.conforms
    }

    /// Set the conformance status
    pub fn set_conforms(&mut self, conforms: bool) {
        self.conforms = conforms;
    }

    /// Get the violations
    pub fn violations(&self) -> &[ValidationViolation] {
        &self.violations
    }

    /// Get mutable reference to violations
    pub fn violations_mut(&mut self) -> &mut Vec<ValidationViolation> {
        &mut self.violations
    }

    /// Get the number of violations
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Check if there are any violations
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Get violations for a specific shape
    pub fn violations_for_shape(&self, shape_id: &ShapeId) -> Vec<&ValidationViolation> {
        self.violations
            .iter()
            .filter(|v| &v.source_shape == shape_id)
            .collect()
    }

    /// Get violations by severity
    pub fn violations_by_severity(&self, severity: Severity) -> Vec<&ValidationViolation> {
        self.violations
            .iter()
            .filter(|v| v.result_severity == severity)
            .collect()
    }

    /// Get error violations (severity = Violation)
    pub fn error_violations(&self) -> Vec<&ValidationViolation> {
        self.violations_by_severity(Severity::Violation)
    }

    /// Get warning violations
    pub fn warning_violations(&self) -> Vec<&ValidationViolation> {
        self.violations_by_severity(Severity::Warning)
    }

    /// Get info violations
    pub fn info_violations(&self) -> Vec<&ValidationViolation> {
        self.violations_by_severity(Severity::Info)
    }

    /// Filter violations by a predicate
    pub fn filter_violations<F>(&self, predicate: F) -> Vec<&ValidationViolation>
    where
        F: Fn(&ValidationViolation) -> bool,
    {
        self.violations.iter().filter(|v| predicate(v)).collect()
    }

    /// Clear all violations
    pub fn clear_violations(&mut self) {
        self.violations.clear();
        self.update_conformance();
        self.update_summary();
    }

    /// Remove violations that match a predicate
    pub fn remove_violations<F>(&mut self, predicate: F)
    where
        F: Fn(&ValidationViolation) -> bool,
    {
        self.violations.retain(|v| !predicate(v));
        self.update_conformance();
        self.update_summary();
    }

    /// Sort violations by severity (errors first)
    pub fn sort_by_severity(&mut self) {
        self.violations.sort_by(|a, b| {
            use std::cmp::Ordering;
            match (&a.result_severity, &b.result_severity) {
                (Severity::Violation, Severity::Violation) => Ordering::Equal,
                (Severity::Violation, _) => Ordering::Less,
                (_, Severity::Violation) => Ordering::Greater,
                (Severity::Warning, Severity::Warning) => Ordering::Equal,
                (Severity::Warning, _) => Ordering::Less,
                (_, Severity::Warning) => Ordering::Greater,
                _ => Ordering::Equal,
            }
        });
    }

    /// Get the metadata
    pub fn metadata(&self) -> &ReportMetadata {
        &self.metadata
    }

    /// Get mutable reference to metadata
    pub fn metadata_mut(&mut self) -> &mut ReportMetadata {
        &mut self.metadata
    }

    /// Get the summary
    pub fn summary(&self) -> &ValidationSummary {
        &self.summary
    }

    /// Update the summary based on current violations
    pub fn update_summary(&mut self) {
        self.summary.calculate_from_violations(&self.violations);
    }

    /// Set the number of nodes validated for summary calculations
    pub fn set_nodes_validated(&mut self, count: usize) {
        self.summary = self.summary.clone().with_nodes_validated(count);
    }

    /// Set the number of shapes validated for summary calculations
    pub fn set_shapes_validated(&mut self, count: usize) {
        self.summary = self.summary.clone().with_shapes_validated(count);
    }

    /// Get a brief text summary
    pub fn brief_summary(&self) -> String {
        if self.conforms {
            "✅ Validation passed - no violations found".to_string()
        } else {
            format!(
                "❌ Validation failed - {} violations ({} errors, {} warnings)",
                self.violation_count(),
                self.summary.error_count(),
                self.summary.warning_count()
            )
        }
    }

    /// Export to JSON format
    pub fn to_json(&self) -> Result<String> {
        self.to_json_with_config(&ReportConfig::default())
    }

    /// Export to JSON with custom configuration
    pub fn to_json_with_config(&self, config: &ReportConfig) -> Result<String> {
        let filtered_report = self.apply_config(config)?;
        if config.pretty_print {
            serde_json::to_string_pretty(&filtered_report).map_err(|e| {
                crate::ShaclError::ReportError(format!("JSON serialization failed: {e}"))
            })
        } else {
            serde_json::to_string(&filtered_report).map_err(|e| {
                crate::ShaclError::ReportError(format!("JSON serialization failed: {e}"))
            })
        }
    }

    /// Export to HTML format
    pub fn to_html(&self) -> Result<String> {
        self.to_html_with_config(&ReportConfig::default())
    }

    /// Export to HTML with custom configuration
    pub fn to_html_with_config(&self, config: &ReportConfig) -> Result<String> {
        super::serializers::HtmlSerializer::new(config.clone()).serialize(self)
    }

    /// Export to Turtle format
    pub fn to_turtle(&self) -> Result<String> {
        self.to_turtle_with_config(&ReportConfig::default())
    }

    /// Export to Turtle with custom configuration
    pub fn to_turtle_with_config(&self, config: &ReportConfig) -> Result<String> {
        super::serializers::TurtleSerializer::new(config.clone()).serialize(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> Result<String> {
        self.to_csv_with_config(&ReportConfig::default())
    }

    /// Export to CSV with custom configuration
    pub fn to_csv_with_config(&self, config: &ReportConfig) -> Result<String> {
        super::serializers::CsvSerializer::new(config.clone()).serialize(self)
    }

    /// Export to plain text format
    pub fn to_text(&self) -> Result<String> {
        Ok(format!("{self}"))
    }

    /// Export to YAML format
    pub fn to_yaml(&self) -> Result<String> {
        self.to_yaml_with_config(&ReportConfig::default())
    }

    /// Export to YAML with custom configuration
    pub fn to_yaml_with_config(&self, config: &ReportConfig) -> Result<String> {
        let filtered_report = self.apply_config(config)?;
        serde_yaml::to_string(&filtered_report)
            .map_err(|e| crate::ShaclError::ReportError(format!("YAML serialization failed: {e}")))
    }

    /// Apply configuration filters to create a filtered report
    fn apply_config(&self, config: &ReportConfig) -> Result<ValidationReport> {
        let mut filtered_report = self.clone();

        // Limit violations if specified
        if let Some(max_violations) = config.max_violations {
            filtered_report.violations.truncate(max_violations);
        }

        // Remove summary if not included
        if !config.include_summary {
            filtered_report.summary = ValidationSummary::default();
        }

        // Remove metadata if not included
        if !config.include_metadata {
            filtered_report.metadata = ReportMetadata::new();
        }

        // Remove timestamps if not included
        if !config.include_timestamps {
            filtered_report.metadata.timestamp = 0;
        }

        Ok(filtered_report)
    }

    fn update_conformance(&mut self) {
        // Data conforms if there are no error-level violations
        self.conforms = !self
            .violations
            .iter()
            .any(|v| v.result_severity == Severity::Violation);
    }

    /// Merge another report into this one
    pub fn merge(&mut self, other: ValidationReport) {
        self.violations.extend(other.violations);
        self.update_conformance();
        self.update_summary();

        // Update metadata with combined information
        if let Some(other_duration) = other.metadata.validation_duration {
            if let Some(our_duration) = self.metadata.validation_duration {
                self.metadata.validation_duration = Some(our_duration + other_duration);
            } else {
                self.metadata.validation_duration = Some(other_duration);
            }
        }
    }

    /// Merge another report into this one (alias for merge)
    pub fn merge_result(&mut self, other: ValidationReport) {
        self.merge(other);
    }

    /// Create a subset report with only specific severities
    pub fn subset_by_severity(&self, severities: &[Severity]) -> ValidationReport {
        let filtered_violations: Vec<ValidationViolation> = self
            .violations
            .iter()
            .filter(|v| severities.contains(&v.result_severity))
            .cloned()
            .collect();

        ValidationReport::from_violations(filtered_violations)
    }

    /// Create a subset report with only specific shapes
    pub fn subset_by_shapes(&self, shape_ids: &[ShapeId]) -> ValidationReport {
        let filtered_violations: Vec<ValidationViolation> = self
            .violations
            .iter()
            .filter(|v| shape_ids.contains(&v.source_shape))
            .cloned()
            .collect();

        ValidationReport::from_violations(filtered_violations)
    }

    /// Serialize validation report to RDF format (Turtle, N-Triples, etc.)
    pub fn to_rdf(&self, format: &str) -> Result<String> {
        // Basic implementation - in a full implementation this would convert to RDF using SHACL vocabulary
        let report_iri = "http://example.org/validation-report";
        let mut rdf_output = String::new();

        match format.to_lowercase().as_str() {
            "turtle" | "ttl" => {
                rdf_output.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
                rdf_output
                    .push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\n");
                rdf_output.push_str(&format!("<{report_iri}> a sh:ValidationReport ;\n"));
                rdf_output.push_str(&format!("    sh:conforms {} ;\n", self.conforms));

                if !self.violations.is_empty() {
                    rdf_output.push_str("    sh:result ");
                    for (i, violation) in self.violations.iter().enumerate() {
                        if i > 0 {
                            rdf_output.push_str(", ");
                        }
                        rdf_output.push_str("[\n        a sh:ValidationResult ;\n");
                        rdf_output.push_str(&format!(
                            "        sh:focusNode <{}> ;\n",
                            violation.focus_node
                        ));
                        rdf_output.push_str(&format!(
                            "        sh:sourceShape <{}> ;\n",
                            violation.source_shape
                        ));
                        rdf_output.push_str(&format!(
                            "        sh:resultSeverity sh:{}\n",
                            violation.result_severity
                        ));
                        if let Some(message) = &violation.result_message {
                            rdf_output.push_str(&format!(
                                "        sh:resultMessage \"{}\" ;\n",
                                message.replace("\"", "\\\"")
                            ));
                        }
                        rdf_output.push_str("    ]");
                    }
                    rdf_output.push_str(" .\n");
                } else {
                    rdf_output.push_str(" .\n");
                }
            }
            "nt" | "ntriples" => {
                rdf_output.push_str(&format!("<{report_iri}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/ns/shacl#ValidationReport> .\n"));
                rdf_output.push_str(&format!("<{}> <http://www.w3.org/ns/shacl#conforms> \"{}\"^^<http://www.w3.org/2001/XMLSchema#boolean> .\n", report_iri, self.conforms));
            }
            _ => {
                return Err(crate::ShaclError::ValidationEngine(format!(
                    "Unsupported RDF format: {format}"
                )));
            }
        }

        Ok(rdf_output)
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SHACL Validation Report")?;
        writeln!(f, "======================")?;
        writeln!(f, "Conforms: {}", self.conforms)?;
        writeln!(f, "Violations: {}", self.violation_count())?;
        writeln!(f, "Generated: {}", self.metadata.formatted_timestamp())?;

        if !self.violations.is_empty() {
            writeln!(f, "\nViolations:")?;
            for (i, violation) in self.violations.iter().enumerate() {
                writeln!(
                    f,
                    "  {}. {} - {} ({})",
                    i + 1,
                    violation.result_severity,
                    violation.focus_node,
                    violation.source_shape
                )?;
                if let Some(message) = &violation.result_message {
                    writeln!(f, "     {message}")?;
                }
            }
        }

        // Add summary if available
        if self.summary.has_violations() {
            writeln!(f, "\nSummary:")?;
            writeln!(f, "{}", self.summary.text_summary())?;
        }

        Ok(())
    }
}
