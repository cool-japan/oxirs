//! SHACL validation report implementation
//! 
//! This module handles generation and serialization of SHACL validation reports.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{NamedNode, Term, Triple, BlankNode, Literal, RdfTerm},
    OxirsError,
};

use crate::{
    ShaclError, Result, ShapeId, PropertyPath, ConstraintComponentId, Severity,
    validation::ValidationViolation,
    vocabulary::SHACL_PREFIXES,
};

/// SHACL validation report according to W3C specification
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
    
    /// Add a violation to the report
    pub fn add_violation(&mut self, violation: ValidationViolation) {
        self.violations.push(violation);
        self.conforms = false;
        self.update_summary();
    }
    
    /// Add multiple violations to the report
    pub fn add_violations(&mut self, violations: Vec<ValidationViolation>) {
        for violation in violations {
            self.add_violation(violation);
        }
    }
    
    /// Merge another validation report into this one
    pub fn merge_result(&mut self, other: ValidationReport) {
        for violation in other.violations {
            self.add_violation(violation);
        }
        self.update_summary();
    }
    
    /// Check if the data conforms (no violations)
    pub fn conforms(&self) -> bool {
        self.conforms
    }
    
    /// Get the number of violations
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
    
    /// Get violations by severity
    pub fn violations_by_severity(&self, severity: &Severity) -> Vec<&ValidationViolation> {
        self.violations.iter()
            .filter(|v| &v.result_severity == severity)
            .collect()
    }
    
    /// Get violations by shape
    pub fn violations_by_shape(&self, shape_id: &ShapeId) -> Vec<&ValidationViolation> {
        self.violations.iter()
            .filter(|v| &v.source_shape == shape_id)
            .collect()
    }
    
    /// Get violations by constraint component
    pub fn violations_by_constraint(&self, component_id: &ConstraintComponentId) -> Vec<&ValidationViolation> {
        self.violations.iter()
            .filter(|v| &v.source_constraint_component == component_id)
            .collect()
    }
    
    /// Filter violations by severity
    pub fn filter_by_severity(&self, severity: &Severity) -> ValidationReport {
        let filtered_violations: Vec<ValidationViolation> = self.violations.iter()
            .filter(|v| &v.result_severity == severity)
            .cloned()
            .collect();
        
        let mut report = ValidationReport::with_metadata(self.metadata.clone());
        report.add_violations(filtered_violations);
        report
    }
    
    /// Update the summary statistics
    fn update_summary(&mut self) {
        self.summary = ValidationSummary::from_violations(&self.violations);
    }
    
    /// Generate an RDF representation of the report
    pub fn to_rdf(&self, format: &str) -> Result<String> {
        match format.to_lowercase().as_str() {
            "turtle" | "ttl" => self.to_turtle(),
            "json-ld" | "jsonld" => self.to_json_ld(),
            "rdf-xml" | "rdfxml" | "xml" => self.to_rdf_xml(),
            "n-triples" | "nt" => self.to_n_triples(),
            _ => Err(ShaclError::ReportGeneration(format!("Unsupported RDF format: {}", format))),
        }
    }
    
    /// Generate a Turtle representation
    fn to_turtle(&self) -> Result<String> {
        let mut turtle = String::new();
        
        // Add prefixes
        turtle.push_str(SHACL_PREFIXES);
        turtle.push_str("\n");
        
        // Generate report IRI
        let report_iri = format!("_:report_{}", self.metadata.timestamp);
        
        // Main validation report
        turtle.push_str(&format!("{} a sh:ValidationReport ;\n", report_iri));
        turtle.push_str(&format!("    sh:conforms {} ;\n", self.conforms));
        
        if !self.violations.is_empty() {
            turtle.push_str("    sh:result ");
            for (i, violation) in self.violations.iter().enumerate() {
                if i > 0 {
                    turtle.push_str(",\n               ");
                }
                let violation_iri = format!("_:result_{}", i);
                turtle.push_str(&violation_iri);
            }
            turtle.push_str(" .\n\n");
            
            // Individual validation results
            for (i, violation) in self.violations.iter().enumerate() {
                let violation_iri = format!("_:result_{}", i);
                turtle.push_str(&self.violation_to_turtle(violation, &violation_iri)?);
                turtle.push_str("\n");
            }
        } else {
            turtle.push_str(" .\n");
        }
        
        Ok(turtle)
    }
    
    /// Convert a single violation to Turtle
    fn violation_to_turtle(&self, violation: &ValidationViolation, violation_iri: &str) -> Result<String> {
        let mut turtle = String::new();
        
        turtle.push_str(&format!("{} a sh:ValidationResult ;\n", violation_iri));
        turtle.push_str(&format!("    sh:focusNode {} ;\n", self.term_to_turtle(&violation.focus_node)?));
        turtle.push_str(&format!("    sh:sourceShape <{}> ;\n", violation.source_shape.as_str()));
        turtle.push_str(&format!("    sh:sourceConstraintComponent {} ;\n", violation.source_constraint_component.as_str()));
        turtle.push_str(&format!("    sh:resultSeverity sh:{} ", violation.result_severity));
        
        if let Some(path) = &violation.result_path {
            turtle.push_str(&format!(";\n    sh:resultPath {} ", self.path_to_turtle(path)?));
        }
        
        if let Some(value) = &violation.value {
            turtle.push_str(&format!(";\n    sh:value {} ", self.term_to_turtle(value)?));
        }
        
        if let Some(message) = &violation.result_message {
            turtle.push_str(&format!(";\n    sh:resultMessage \"{}\" ", message.replace('"', "\\\"")));
        }
        
        turtle.push_str(" .\n");
        
        Ok(turtle)
    }
    
    /// Convert a term to Turtle representation
    fn term_to_turtle(&self, term: &Term) -> Result<String> {
        match term {
            Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
            Term::BlankNode(node) => Ok(node.as_str().to_string()),
            Term::Literal(literal) => {
                // TODO: Handle datatype and language tags properly
                Ok(format!("\"{}\"", literal.as_str().replace('"', "\\\"")))
            }
            Term::Variable(var) => Ok(format!("?{}", var.name())),
        }
    }
    
    /// Convert a property path to Turtle representation
    fn path_to_turtle(&self, path: &PropertyPath) -> Result<String> {
        match path {
            PropertyPath::Predicate(pred) => Ok(format!("<{}>", pred.as_str())),
            // TODO: Implement complex property path serialization
            _ => Ok("[ ]".to_string()), // Placeholder for complex paths
        }
    }
    
    /// Generate a JSON-LD representation
    fn to_json_ld(&self) -> Result<String> {
        // TODO: Implement JSON-LD serialization
        Err(ShaclError::ReportGeneration("JSON-LD serialization not yet implemented".to_string()))
    }
    
    /// Generate an RDF/XML representation
    fn to_rdf_xml(&self) -> Result<String> {
        // TODO: Implement RDF/XML serialization
        Err(ShaclError::ReportGeneration("RDF/XML serialization not yet implemented".to_string()))
    }
    
    /// Generate an N-Triples representation
    fn to_n_triples(&self) -> Result<String> {
        // TODO: Implement N-Triples serialization
        Err(ShaclError::ReportGeneration("N-Triples serialization not yet implemented".to_string()))
    }
    
    /// Generate a JSON representation (non-RDF)
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ShaclError::ReportGeneration(format!("JSON serialization error: {}", e)))
    }
    
    /// Generate an HTML report
    pub fn to_html(&self) -> Result<String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html>\n<head>\n");
        html.push_str("<title>SHACL Validation Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(include_str!("html_report_style.css"));
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        
        // Header
        html.push_str("<div class=\"header\">\n");
        html.push_str("<h1>SHACL Validation Report</h1>\n");
        html.push_str(&format!("<p>Generated: {}</p>\n", self.metadata.formatted_timestamp()));
        html.push_str("</div>\n");
        
        // Summary
        html.push_str("<div class=\"summary\">\n");
        html.push_str("<h2>Summary</h2>\n");
        html.push_str(&format!("<p><strong>Conforms:</strong> {}</p>\n", 
            if self.conforms { "✅ Yes" } else { "❌ No" }));
        html.push_str(&format!("<p><strong>Total Violations:</strong> {}</p>\n", self.violation_count()));
        html.push_str(&format!("<p><strong>Violations:</strong> {}</p>\n", self.summary.violation_count));
        html.push_str(&format!("<p><strong>Warnings:</strong> {}</p>\n", self.summary.warning_count));
        html.push_str(&format!("<p><strong>Info:</strong> {}</p>\n", self.summary.info_count));
        html.push_str("</div>\n");
        
        // Violations
        if !self.violations.is_empty() {
            html.push_str("<div class=\"violations\">\n");
            html.push_str("<h2>Violations</h2>\n");
            
            for (i, violation) in self.violations.iter().enumerate() {
                html.push_str(&format!("<div class=\"violation violation-{}\">\n", 
                    violation.result_severity.to_string().to_lowercase()));
                html.push_str(&format!("<h3>Violation {} - {}</h3>\n", i + 1, violation.result_severity));
                html.push_str(&format!("<p><strong>Focus Node:</strong> {}</p>\n", violation.focus_node.as_str()));
                html.push_str(&format!("<p><strong>Source Shape:</strong> {}</p>\n", violation.source_shape.as_str()));
                html.push_str(&format!("<p><strong>Constraint Component:</strong> {}</p>\n", violation.source_constraint_component.as_str()));
                
                if let Some(path) = &violation.result_path {
                    html.push_str(&format!("<p><strong>Property Path:</strong> {:?}</p>\n", path));
                }
                
                if let Some(value) = &violation.value {
                    html.push_str(&format!("<p><strong>Value:</strong> {}</p>\n", value.as_str()));
                }
                
                if let Some(message) = &violation.result_message {
                    html.push_str(&format!("<p><strong>Message:</strong> {}</p>\n", message));
                }
                
                html.push_str("</div>\n");
            }
            
            html.push_str("</div>\n");
        }
        
        html.push_str("</body>\n</html>\n");
        
        Ok(html)
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
                writeln!(f, "  {}. {} - {} ({})", 
                    i + 1,
                    violation.result_severity,
                    violation.focus_node.as_str(),
                    violation.source_shape.as_str()
                )?;
                if let Some(message) = &violation.result_message {
                    writeln!(f, "     {}", message)?;
                }
            }
        }
        
        Ok(())
    }
}

/// Metadata about the validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Timestamp when the report was generated
    pub timestamp: u64,
    
    /// Version of the SHACL implementation
    pub shacl_version: String,
    
    /// Version of the validator
    pub validator_version: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ReportMetadata {
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            shacl_version: "1.0".to_string(),
            validator_version: env!("CARGO_PKG_VERSION").to_string(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    pub fn formatted_timestamp(&self) -> String {
        // TODO: Format timestamp properly
        format!("Timestamp: {}", self.timestamp)
    }
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for validation results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_violations: usize,
    pub violation_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub shapes_evaluated: HashSet<ShapeId>,
    pub constraints_evaluated: HashSet<ConstraintComponentId>,
    pub focus_nodes: HashSet<String>,
}

impl ValidationSummary {
    pub fn from_violations(violations: &[ValidationViolation]) -> Self {
        let mut summary = ValidationSummary::default();
        
        summary.total_violations = violations.len();
        
        for violation in violations {
            match violation.result_severity {
                Severity::Violation => summary.violation_count += 1,
                Severity::Warning => summary.warning_count += 1,
                Severity::Info => summary.info_count += 1,
            }
            
            summary.shapes_evaluated.insert(violation.source_shape.clone());
            summary.constraints_evaluated.insert(violation.source_constraint_component.clone());
            summary.focus_nodes.insert(violation.focus_node.as_str().to_string());
        }
        
        summary
    }
}

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Include detailed violation information
    pub include_details: bool,
    
    /// Include performance statistics
    pub include_stats: bool,
    
    /// Maximum number of violations to include
    pub max_violations: Option<usize>,
    
    /// Include only specific severity levels
    pub severity_filter: Option<Vec<Severity>>,
    
    /// Include shape information in reports
    pub include_shape_info: bool,
    
    /// Generate human-readable messages
    pub human_readable: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            include_details: true,
            include_stats: true,
            max_violations: None,
            severity_filter: None,
            include_shape_info: true,
            human_readable: true,
        }
    }
}

/// Report generator for creating validation reports
#[derive(Debug)]
pub struct ReportGenerator {
    config: ReportConfig,
}

impl ReportGenerator {
    pub fn new(config: ReportConfig) -> Self {
        Self { config }
    }
    
    pub fn generate_report(&self, violations: Vec<ValidationViolation>) -> ValidationReport {
        let mut filtered_violations = violations;
        
        // Apply severity filter
        if let Some(ref severity_filter) = self.config.severity_filter {
            filtered_violations.retain(|v| severity_filter.contains(&v.result_severity));
        }
        
        // Apply violation limit
        if let Some(max_violations) = self.config.max_violations {
            filtered_violations.truncate(max_violations);
        }
        
        let mut report = ValidationReport::new();
        report.add_violations(filtered_violations);
        
        report
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new(ReportConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{validation::ValidationViolation, ShapeId, ConstraintComponentId};
    
    #[test]
    fn test_validation_report_creation() {
        let report = ValidationReport::new();
        assert!(report.conforms());
        assert_eq!(report.violation_count(), 0);
        assert!(report.violations.is_empty());
    }
    
    #[test]
    fn test_add_violation() {
        let mut report = ValidationReport::new();
        assert!(report.conforms());
        
        let violation = ValidationViolation::new(
            Term::NamedNode(NamedNode::new("http://example.org/john").unwrap()),
            ShapeId::new("http://example.org/PersonShape"),
            ConstraintComponentId::new("sh:ClassConstraintComponent"),
            Severity::Violation
        );
        
        report.add_violation(violation);
        
        assert!(!report.conforms());
        assert_eq!(report.violation_count(), 1);
    }
    
    #[test]
    fn test_violations_by_severity() {
        let mut report = ValidationReport::new();
        
        let violation1 = ValidationViolation::new(
            Term::NamedNode(NamedNode::new("http://example.org/john").unwrap()),
            ShapeId::new("http://example.org/PersonShape"),
            ConstraintComponentId::new("sh:ClassConstraintComponent"),
            Severity::Violation
        );
        
        let violation2 = ValidationViolation::new(
            Term::NamedNode(NamedNode::new("http://example.org/jane").unwrap()),
            ShapeId::new("http://example.org/PersonShape"),
            ConstraintComponentId::new("sh:MinCountConstraintComponent"),
            Severity::Warning
        );
        
        report.add_violation(violation1);
        report.add_violation(violation2);
        
        let violations = report.violations_by_severity(&Severity::Violation);
        assert_eq!(violations.len(), 1);
        
        let warnings = report.violations_by_severity(&Severity::Warning);
        assert_eq!(warnings.len(), 1);
    }
    
    #[test]
    fn test_filter_by_severity() {
        let mut report = ValidationReport::new();
        
        let violation = ValidationViolation::new(
            Term::NamedNode(NamedNode::new("http://example.org/john").unwrap()),
            ShapeId::new("http://example.org/PersonShape"),
            ConstraintComponentId::new("sh:ClassConstraintComponent"),
            Severity::Violation
        );
        
        let warning = ValidationViolation::new(
            Term::NamedNode(NamedNode::new("http://example.org/jane").unwrap()),
            ShapeId::new("http://example.org/PersonShape"),
            ConstraintComponentId::new("sh:MinCountConstraintComponent"),
            Severity::Warning
        );
        
        report.add_violation(violation);
        report.add_violation(warning);
        
        let violations_only = report.filter_by_severity(&Severity::Violation);
        assert_eq!(violations_only.violation_count(), 1);
        assert!(!violations_only.conforms());
    }
    
    #[test]
    fn test_report_metadata() {
        let metadata = ReportMetadata::new()
            .with_metadata("test_key".to_string(), "test_value".to_string());
        
        assert_eq!(metadata.shacl_version, "1.0");
        assert_eq!(metadata.validator_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(metadata.metadata.get("test_key"), Some(&"test_value".to_string()));
    }
    
    #[test]
    fn test_validation_summary() {
        let violations = vec![
            ValidationViolation::new(
                Term::NamedNode(NamedNode::new("http://example.org/john").unwrap()),
                ShapeId::new("http://example.org/PersonShape"),
                ConstraintComponentId::new("sh:ClassConstraintComponent"),
                Severity::Violation
            ),
            ValidationViolation::new(
                Term::NamedNode(NamedNode::new("http://example.org/jane").unwrap()),
                ShapeId::new("http://example.org/PersonShape"),
                ConstraintComponentId::new("sh:MinCountConstraintComponent"),
                Severity::Warning
            ),
        ];
        
        let summary = ValidationSummary::from_violations(&violations);
        
        assert_eq!(summary.total_violations, 2);
        assert_eq!(summary.violation_count, 1);
        assert_eq!(summary.warning_count, 1);
        assert_eq!(summary.info_count, 0);
        assert_eq!(summary.shapes_evaluated.len(), 1);
        assert_eq!(summary.constraints_evaluated.len(), 2);
        assert_eq!(summary.focus_nodes.len(), 2);
    }
    
    #[test]
    fn test_json_serialization() {
        let report = ValidationReport::new();
        let json = report.to_json().unwrap();
        assert!(json.contains("\"conforms\": true"));
        assert!(json.contains("\"violations\": []"));
    }
}