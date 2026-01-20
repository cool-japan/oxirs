//! Diagnostics generation for SHACL shapes.
//!
//! Generates LSP diagnostics from SHACL validation results and shape parsing errors.

use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, NumberOrString, Position, Range};

use crate::{Severity, Shape, ValidationReport};

/// Diagnostics generator for SHACL shapes
pub struct DiagnosticsGenerator {}

impl DiagnosticsGenerator {
    /// Create a new diagnostics generator
    pub fn new() -> Self {
        Self {}
    }

    /// Generate diagnostics from document text and parsed shapes
    pub fn generate_diagnostics(&self, text: &str, shapes: &[Shape]) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        // Check for common shape issues
        for shape in shapes {
            // Check for missing sh:targetClass or sh:targetNode
            if !self.has_target(shape) {
                diagnostics.push(Diagnostic {
                    range: self.get_shape_range(text, shape),
                    severity: Some(DiagnosticSeverity::WARNING),
                    code: Some(NumberOrString::String("missing-target".to_string())),
                    code_description: None,
                    source: Some("shacl-lsp".to_string()),
                    message: "Shape has no target definition (sh:targetClass, sh:targetNode, etc.)"
                        .to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
            }

            // Check for shapes with no constraints
            if !self.has_node_constraints(shape) {
                diagnostics.push(Diagnostic {
                    range: self.get_shape_range(text, shape),
                    severity: Some(DiagnosticSeverity::INFORMATION),
                    code: Some(NumberOrString::String("empty-shape".to_string())),
                    code_description: None,
                    source: Some("shacl-lsp".to_string()),
                    message: "Shape has no constraints defined".to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                });
            }
        }

        diagnostics
    }

    /// Generate diagnostics from validation report
    pub fn from_validation_report(&self, report: &ValidationReport) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for violation in &report.violations {
            let severity = match violation.result_severity {
                Severity::Violation => DiagnosticSeverity::ERROR,
                Severity::Warning => DiagnosticSeverity::WARNING,
                Severity::Info => DiagnosticSeverity::INFORMATION,
            };

            diagnostics.push(Diagnostic {
                range: Range::new(Position::new(0, 0), Position::new(0, 0)),
                severity: Some(severity),
                code: None,
                code_description: None,
                source: Some("shacl-validation".to_string()),
                message: violation
                    .result_message
                    .clone()
                    .unwrap_or_else(|| "Validation failed".to_string()),
                related_information: None,
                tags: None,
                data: None,
            });
        }

        diagnostics
    }

    /// Check if shape has a target definition
    fn has_target(&self, shape: &Shape) -> bool {
        // Check if shape has any targets defined
        !shape.targets.is_empty() || shape.shape_type == crate::ShapeType::PropertyShape
    }

    /// Check if shape has node-level constraints
    fn has_node_constraints(&self, shape: &Shape) -> bool {
        // Check for constraints like sh:class, sh:datatype, etc.
        !shape.constraints.is_empty()
    }

    /// Get the text range for a shape (simplified - would need position tracking)
    fn get_shape_range(&self, _text: &str, _shape: &Shape) -> Range {
        // In a real implementation, would track line/column positions during parsing
        Range::new(Position::new(0, 0), Position::new(0, 0))
    }
}

impl Default for DiagnosticsGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_generation() {
        let generator = DiagnosticsGenerator::new();
        let shapes = Vec::new();
        let diagnostics = generator.generate_diagnostics("", &shapes);
        assert_eq!(diagnostics.len(), 0);
    }

    #[test]
    fn test_empty_shape_diagnostic() {
        let _generator = DiagnosticsGenerator::new();
        // Test would create a shape with no constraints
        // and verify a diagnostic is generated
    }
}
