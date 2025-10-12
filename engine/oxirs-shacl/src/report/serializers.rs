//\! Report serializers for different output formats

use super::{ReportConfig, ValidationReport};
use crate::{PropertyPath, Result};

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
                    "<p><strong>Value:</strong> <code>{value}</code></p>"
                ));
            }

            if let Some(message) = &violation.result_message {
                html.push_str(&format!("<div class=\"violation-message\">{message}</div>"));
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
                "<p><strong>Validation Duration:</strong> {duration:.2?}</p>"
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
        format!("{path:?}")
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
                self.escape_csv_field(violation.result_message.as_ref().unwrap_or(&"".to_string()))
            ));
        }

        Ok(csv)
    }

    fn format_path_for_csv(&self, path: &PropertyPath) -> Result<String> {
        // Simplified path formatting for CSV
        Ok(format!("{path:?}"))
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
                turtle.push_str("      [ a sh:ValidationResult ;\n");
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

/// Prometheus metrics serializer for validation reports
/// Exports validation metrics in Prometheus text format for monitoring systems
pub struct PrometheusSerializer {
    config: ReportConfig,
}

impl PrometheusSerializer {
    pub fn new(config: ReportConfig) -> Self {
        Self { config }
    }

    pub fn serialize(&self, report: &ValidationReport) -> Result<String> {
        let mut metrics = String::new();

        // HELP and TYPE declarations for each metric
        metrics.push_str(
            "# HELP shacl_validation_conforms Whether the validation passed (1) or failed (0)\n",
        );
        metrics.push_str("# TYPE shacl_validation_conforms gauge\n");
        metrics.push_str(&format!(
            "shacl_validation_conforms {}\n\n",
            if report.conforms { 1 } else { 0 }
        ));

        metrics.push_str(
            "# HELP shacl_validation_violations_total Total number of validation violations\n",
        );
        metrics.push_str("# TYPE shacl_validation_violations_total counter\n");
        metrics.push_str(&format!(
            "shacl_validation_violations_total {}\n\n",
            report.violation_count()
        ));

        // Violations by severity
        metrics.push_str("# HELP shacl_validation_violations_by_severity Number of violations by severity level\n");
        metrics.push_str("# TYPE shacl_validation_violations_by_severity gauge\n");
        metrics.push_str(&format!(
            "shacl_validation_violations_by_severity{{severity=\"violation\"}} {}\n",
            report.summary.error_count()
        ));
        metrics.push_str(&format!(
            "shacl_validation_violations_by_severity{{severity=\"warning\"}} {}\n",
            report.summary.warning_count()
        ));
        metrics.push_str(&format!(
            "shacl_validation_violations_by_severity{{severity=\"info\"}} {}\n\n",
            report.summary.info_count()
        ));

        // Success rate
        metrics.push_str("# HELP shacl_validation_success_rate Percentage of successful validations (0.0 to 1.0)\n");
        metrics.push_str("# TYPE shacl_validation_success_rate gauge\n");
        metrics.push_str(&format!(
            "shacl_validation_success_rate {:.6}\n\n",
            report.summary.success_rate
        ));

        // Quality score
        metrics.push_str(
            "# HELP shacl_validation_quality_score Overall validation quality score (0.0 to 1.0)\n",
        );
        metrics.push_str("# TYPE shacl_validation_quality_score gauge\n");
        metrics.push_str(&format!(
            "shacl_validation_quality_score {:.6}\n\n",
            report.summary.quality_score()
        ));

        // Validation duration if available
        if let Some(duration) = report.metadata.validation_duration {
            metrics.push_str(
                "# HELP shacl_validation_duration_seconds Duration of validation in seconds\n",
            );
            metrics.push_str("# TYPE shacl_validation_duration_seconds gauge\n");
            metrics.push_str(&format!(
                "shacl_validation_duration_seconds {:.6}\n\n",
                duration.as_secs_f64()
            ));
        }

        // Shape validation counts
        metrics.push_str("# HELP shacl_validation_shapes_total Total number of shapes validated\n");
        metrics.push_str("# TYPE shacl_validation_shapes_total counter\n");
        metrics.push_str(&format!(
            "shacl_validation_shapes_total {}\n\n",
            report.summary.shapes_validated
        ));

        // Total violations across all constraint types
        metrics.push_str("# HELP shacl_validation_constraints_total Total number of constraint violations checked\n");
        metrics.push_str("# TYPE shacl_validation_constraints_total counter\n");
        metrics.push_str(&format!(
            "shacl_validation_constraints_total {}\n\n",
            report.summary.violations_by_component.len()
        ));

        // Violations per shape (top violating shapes)
        if self.config.include_details {
            metrics.push_str(
                "# HELP shacl_validation_violations_per_shape Number of violations per shape\n",
            );
            metrics.push_str("# TYPE shacl_validation_violations_per_shape gauge\n");

            use std::collections::HashMap;
            let mut shape_violations: HashMap<String, usize> = HashMap::new();

            for violation in &report.violations {
                *shape_violations
                    .entry(violation.source_shape.to_string())
                    .or_insert(0) += 1;
            }

            // Limit to top shapes if configured
            let max_shapes = self.config.max_violations.unwrap_or(usize::MAX).min(50);
            let mut sorted_shapes: Vec<_> = shape_violations.iter().collect();
            sorted_shapes.sort_by(|a, b| b.1.cmp(a.1));

            for (shape, count) in sorted_shapes.iter().take(max_shapes) {
                let escaped_shape = Self::escape_label_value(shape);
                metrics.push_str(&format!(
                    "shacl_validation_violations_per_shape{{shape=\"{}\"}} {}\n",
                    escaped_shape, count
                ));
            }
            metrics.push('\n');
        }

        // Violations per constraint component
        if self.config.include_details {
            metrics.push_str("# HELP shacl_validation_violations_per_constraint Number of violations per constraint type\n");
            metrics.push_str("# TYPE shacl_validation_violations_per_constraint gauge\n");

            use std::collections::HashMap;
            let mut constraint_violations: HashMap<String, usize> = HashMap::new();

            for violation in &report.violations {
                *constraint_violations
                    .entry(violation.source_constraint_component.to_string())
                    .or_insert(0) += 1;
            }

            for (constraint, count) in constraint_violations.iter() {
                let escaped_constraint = Self::escape_label_value(constraint);
                metrics.push_str(&format!(
                    "shacl_validation_violations_per_constraint{{constraint=\"{}\"}} {}\n",
                    escaped_constraint, count
                ));
            }
            metrics.push('\n');
        }

        // Performance metrics if available
        if report.metadata.has_performance_data() {
            if let Some(nodes_validated) = report.metadata.metadata.get("nodes_validated") {
                if let Ok(count) = nodes_validated.parse::<usize>() {
                    metrics.push_str("# HELP shacl_validation_nodes_validated_total Total number of nodes validated\n");
                    metrics.push_str("# TYPE shacl_validation_nodes_validated_total counter\n");
                    metrics.push_str(&format!(
                        "shacl_validation_nodes_validated_total {}\n\n",
                        count
                    ));
                }
            }

            if let Some(cache_hit_rate) = report.metadata.metadata.get("cache_hit_rate") {
                if let Ok(rate) = cache_hit_rate.parse::<f64>() {
                    metrics.push_str(
                        "# HELP shacl_validation_cache_hit_rate Cache hit rate (0.0 to 1.0)\n",
                    );
                    metrics.push_str("# TYPE shacl_validation_cache_hit_rate gauge\n");
                    metrics.push_str(&format!("shacl_validation_cache_hit_rate {:.6}\n\n", rate));
                }
            }
        }

        // Timestamp of validation
        metrics.push_str("# HELP shacl_validation_timestamp_seconds Unix timestamp when validation was performed\n");
        metrics.push_str("# TYPE shacl_validation_timestamp_seconds gauge\n");
        metrics.push_str(&format!(
            "shacl_validation_timestamp_seconds {}\n\n",
            report.metadata.timestamp
        ));

        Ok(metrics)
    }

    /// Escape label values for Prometheus format
    fn escape_label_value(value: &str) -> String {
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_prometheus_serializer() {
        let report = ValidationReport::new();
        let config = ReportConfig::default();
        let serializer = PrometheusSerializer::new(config);

        let result = serializer.serialize(&report);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.contains("# HELP shacl_validation_conforms"));
        assert!(metrics.contains("# TYPE shacl_validation_conforms gauge"));
        assert!(metrics.contains("shacl_validation_conforms 1"));
        assert!(metrics.contains("shacl_validation_violations_total 0"));
        assert!(metrics.contains("shacl_validation_success_rate"));
        assert!(metrics.contains("shacl_validation_quality_score"));
    }

    #[test]
    fn test_prometheus_escape_label_value() {
        assert_eq!(
            PrometheusSerializer::escape_label_value("test\\value"),
            "test\\\\value"
        );
        assert_eq!(
            PrometheusSerializer::escape_label_value("test\"value"),
            "test\\\"value"
        );
        assert_eq!(
            PrometheusSerializer::escape_label_value("test\nvalue"),
            "test\\nvalue"
        );
    }
}
