//\! Report serializers for different output formats

use super::{ReportConfig, ValidationReport};
use crate::{PropertyPath, Result, ShaclError};

/// HTML serializer for validation reports
pub struct HtmlSerializer {
    config: ReportConfig,
}

impl HtmlSerializer {
    pub fn new(config: ReportConfig) -> Self {
        Self { config }
    }

    pub fn serialize(&self, report: &ValidationReport) -> Result<String> {
        let mut html = String::new();

        // HTML header with styling
        html.push_str(&self.html_header());

        // Report header
        html.push_str(&format!(
            r#"<div class="report-header">
                <h1>SHACL Validation Report</h1>
                <div class="status {}">{}</div>
                <div class="metadata">
                    <span>Generated: {}</span>
                    <span>Violations: {}</span>
                </div>
            </div>"#,
            if report.conforms { "success" } else { "error" },
            if report.conforms {
                "✅ Validation Passed"
            } else {
                "❌ Validation Failed"
            },
            report.metadata.formatted_timestamp(),
            report.violation_count()
        ));

        // Summary section
        if self.config.include_summary && report.summary.has_violations() {
            html.push_str(&self.format_summary(&report.summary));
        }

        // Violations section
        if self.config.include_details && !report.violations.is_empty() {
            html.push_str(&self.format_violations(report));
        }

        // Metadata section
        if self.config.include_metadata {
            html.push_str(&self.format_metadata(&report.metadata));
        }

        html.push_str("</body></html>");
        Ok(html)
    }

    fn html_header(&self) -> String {
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SHACL Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .report-header { border-bottom: 2px solid #ccc; padding-bottom: 20px; margin-bottom: 20px; }
        .status { font-size: 1.2em; font-weight: bold; padding: 10px; border-radius: 5px; }
        .status.success { background-color: #d4edda; color: #155724; }
        .status.error { background-color: #f8d7da; color: #721c24; }
        .metadata { margin-top: 10px; color: #666; }
        .metadata span { margin-right: 20px; }
        .summary { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
        .violations { margin: 20px 0; }
        .violation { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .violation.error { border-left: 4px solid #dc3545; }
        .violation.warning { border-left: 4px solid #ffc107; }
        .violation.info { border-left: 4px solid #17a2b8; }
        .violation-header { font-weight: bold; margin-bottom: 10px; }
        .violation-details { color: #666; font-size: 0.9em; }
        .violation-message { margin-top: 10px; font-style: italic; }
        code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
        .metadata-section { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ccc; }
    </style>
</head>
<body>"#
            .to_string()
    }

    fn format_summary(&self, summary: &super::ValidationSummary) -> String {
        format!(
            r#"<div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Violations:</strong> {}</p>
                <p><strong>Errors:</strong> {}, <strong>Warnings:</strong> {}, <strong>Info:</strong> {}</p>
                <p><strong>Success Rate:</strong> {:.1}%</p>
                <p><strong>Quality Score:</strong> {:.1}%</p>
            </div>"#,
            summary.total_violations(),
            summary.error_count(),
            summary.warning_count(),
            summary.info_count(),
            summary.success_rate * 100.0,
            summary.quality_score() * 100.0
        )
    }

    fn format_violations(&self, report: &ValidationReport) -> String {
        let mut html = String::from(r#"<div class="violations"><h2>Violations</h2>"#);

        let violations = if let Some(max) = self.config.max_violations {
            &report.violations[..report.violations.len().min(max)]
        } else {
            &report.violations
        };

        for (i, violation) in violations.iter().enumerate() {
            let severity_class = match violation.result_severity {
                crate::Severity::Violation => "error",
                crate::Severity::Warning => "warning",
                crate::Severity::Info => "info",
            };

            html.push_str(&format!(
                r#"<div class="violation {}">
                    <div class="violation-header">
                        {} {}. {} at <code>{}</code>
                    </div>
                    <div class="violation-details">
                        <p><strong>Shape:</strong> <code>{}</code></p>"#,
                severity_class,
                match violation.result_severity {
                    crate::Severity::Violation => "❌",
                    crate::Severity::Warning => "⚠️",
                    crate::Severity::Info => "ℹ️",
                },
                i + 1,
                violation.result_severity,
                violation.focus_node,
                violation.source_shape
            ));

            html.push_str(&format!(
                "<p><strong>Constraint:</strong> <code>{}</code></p>",
                violation.source_constraint_component
            ));

            if let Some(path) = &violation.result_path {
                html.push_str(&format!(
                    "<p><strong>Path:</strong> <code>{}</code></p>",
                    self.format_path_for_html(path)
                ));
            }

            if let Some(value) = &violation.value {
                html.push_str(&format!(
                    "<p><strong>Value:</strong> <code>{}</code></p>",
                    value
                ));
            }

            if let Some(message) = &violation.result_message {
                html.push_str(&format!(
                    "<div class=\"violation-message\">{}</div>",
                    message
                ));
            }

            html.push_str("</div>");
        }

        if let Some(max) = self.config.max_violations {
            if report.violations.len() > max {
                html.push_str(&format!(
                    "<p><em>... and {} more violations (showing first {})</em></p>",
                    report.violations.len() - max,
                    max
                ));
            }
        }

        html.push_str("</div>");
        html
    }

    fn format_metadata(&self, metadata: &super::ReportMetadata) -> String {
        let mut html = String::from(r#"<div class="metadata-section"><h2>Metadata</h2>"#);

        html.push_str(&format!(
            "<p><strong>SHACL Version:</strong> {}</p>",
            metadata.shacl_version
        ));
        html.push_str(&format!(
            "<p><strong>Validator Version:</strong> {}</p>",
            metadata.validator_version
        ));

        if let Some(duration) = metadata.validation_duration {
            html.push_str(&format!(
                "<p><strong>Validation Duration:</strong> {:.2?}</p>",
                duration
            ));
        }

        if metadata.has_performance_data() {
            html.push_str(&format!(
                "<p><strong>Performance:</strong> {}</p>",
                metadata.performance_summary()
            ));
        }

        html.push_str("</div>");
        html
    }

    fn format_path_for_html(&self, path: &PropertyPath) -> String {
        // Simplified path formatting for HTML
        format!("{:?}", path)
    }
}

/// CSV serializer for validation reports
pub struct CsvSerializer {
    config: ReportConfig,
}

impl CsvSerializer {
    pub fn new(config: ReportConfig) -> Self {
        Self { config }
    }

    pub fn serialize(&self, report: &ValidationReport) -> Result<String> {
        let mut csv = String::new();

        // CSV header
        csv.push_str("Index,Severity,Focus Node,Shape,Constraint,Path,Value,Message\n");

        let violations = if let Some(max) = self.config.max_violations {
            &report.violations[..report.violations.len().min(max)]
        } else {
            &report.violations
        };

        // CSV rows
        for (i, violation) in violations.iter().enumerate() {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                i + 1,
                self.escape_csv_field(&violation.result_severity.to_string()),
                self.escape_csv_field(&violation.focus_node.to_string()),
                self.escape_csv_field(&violation.source_shape.to_string()),
                self.escape_csv_field(&violation.source_constraint_component.to_string()),
                self.escape_csv_field(
                    &violation
                        .result_path
                        .as_ref()
                        .map(|p| self.format_path_for_csv(p).unwrap_or_default())
                        .unwrap_or_else(|| "".to_string())
                ),
                self.escape_csv_field(
                    &violation
                        .value
                        .as_ref()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "".to_string())
                ),
                self.escape_csv_field(
                    &violation.result_message.as_ref().unwrap_or(&"".to_string())
                )
            ));
        }

        Ok(csv)
    }

    fn format_path_for_csv(&self, path: &PropertyPath) -> Result<String> {
        // Simplified path formatting for CSV
        Ok(format!("{:?}", path))
    }

    fn escape_csv_field(&self, field: &str) -> String {
        if field.contains(',')
            || field.contains('"')
            || field.contains('\n')
            || field.contains('\r')
        {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }
}

/// Turtle serializer for validation reports
pub struct TurtleSerializer {
    config: ReportConfig,
}

impl TurtleSerializer {
    pub fn new(config: ReportConfig) -> Self {
        Self { config }
    }

    pub fn serialize(&self, report: &ValidationReport) -> Result<String> {
        let mut turtle = String::new();

        // Prefixes
        turtle.push_str(&self.turtle_prefixes());

        // Report
        turtle.push_str("[] a sh:ValidationReport ;\n");
        turtle.push_str(&format!("   sh:conforms {} ;\n", report.conforms));

        let violations = if let Some(max) = self.config.max_violations {
            &report.violations[..report.violations.len().min(max)]
        } else {
            &report.violations
        };

        if !violations.is_empty() {
            turtle.push_str("   sh:result\n");
            for (i, violation) in violations.iter().enumerate() {
                turtle.push_str(&format!("      [ a sh:ValidationResult ;\n"));
                turtle.push_str(&format!(
                    "        sh:resultSeverity sh:{} ;\n",
                    match violation.result_severity {
                        crate::Severity::Violation => "Violation",
                        crate::Severity::Warning => "Warning",
                        crate::Severity::Info => "Info",
                    }
                ));
                turtle.push_str(&format!(
                    "        sh:focusNode <{}> ;\n",
                    violation.focus_node
                ));
                turtle.push_str(&format!(
                    "        sh:sourceShape <{}> ;\n",
                    violation.source_shape
                ));

                turtle.push_str(&format!(
                    "        sh:sourceConstraintComponent sh:{} ;\n",
                    violation.source_constraint_component
                ));

                if let Some(message) = &violation.result_message {
                    turtle.push_str(&format!(
                        "        sh:resultMessage \"{}\" ;\n",
                        message.replace('\\', "\\\\").replace('"', "\\\"")
                    ));
                }

                turtle.push_str("      ]");
                if i < violations.len() - 1 {
                    turtle.push_str(" ,\n");
                } else {
                    turtle.push_str(" .\n");
                }
            }
        } else {
            turtle.push_str(" .\n");
        }

        Ok(turtle)
    }

    fn turtle_prefixes(&self) -> String {
        r#"@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ValidationViolation;
    use oxirs_core::model::*;

    #[test]
    fn test_html_serializer() {
        let report = ValidationReport::new();
        let config = ReportConfig::default();
        let serializer = HtmlSerializer::new(config);

        let result = serializer.serialize(&report);
        assert!(result.is_ok());
        let html = result.unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Validation Passed"));
    }

    #[test]
    fn test_csv_serializer() {
        let report = ValidationReport::new();
        let config = ReportConfig::default();
        let serializer = CsvSerializer::new(config);

        let result = serializer.serialize(&report);
        assert!(result.is_ok());
        let csv = result.unwrap();
        assert!(csv.contains("Index,Severity,Focus Node"));
    }

    #[test]
    fn test_turtle_serializer() {
        let report = ValidationReport::new();
        let config = ReportConfig::default();
        let serializer = TurtleSerializer::new(config);

        let result = serializer.serialize(&report);
        assert!(result.is_ok());
        let turtle = result.unwrap();
        assert!(turtle.contains("@prefix sh:"));
        assert!(turtle.contains("sh:ValidationReport"));
    }
}
