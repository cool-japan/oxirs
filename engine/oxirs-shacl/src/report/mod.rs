//! SHACL validation report implementation
//!
//! This module handles generation and serialization of SHACL validation reports.

#![allow(dead_code)]

pub mod advanced_filtering;
pub mod analytics;
pub mod core;
pub mod documentation;
pub mod format;
pub mod generation;
pub mod generator;
pub mod interactive;
pub mod metadata;
pub mod nested_results;
pub mod serializers;
pub mod summary;
pub mod visualizer;

// Re-export key types for convenience
pub use core::ValidationReport;
pub use format::{ReportConfig, ReportFormat};
pub use generator::{generate_report, ReportGenerator};
pub use metadata::ReportMetadata;
pub use summary::ValidationSummary;

// Re-export enhanced generation functions
pub use generation::{
    generate_csv_report, generate_html_report, generate_json_report, generate_jsonld_report,
    generate_ntriples_report, generate_prometheus_report, generate_rdfxml_report,
    generate_text_report, generate_turtle_report, generate_yaml_report,
};

// Re-export analytics types
pub use analytics::{AnalyticsConfig, AnalyzedReport, ReportMetrics, ValidationReportAnalytics};

// Re-export advanced filtering types
pub use advanced_filtering::{
    CustomFilter, FilterConfig, FilterFunction, FilteredReport, QueryCriteria, QueryResult,
    ReportFilterEngine, ReportQueryEngine, ReportTemplate, TemplateConfig, TimeRange,
};

// Re-export interactive viewer types
pub use interactive::{
    ExportConfig, ExportFormat, FilterOptions, InteractiveReportView, InteractiveReportViewer,
    PaginationConfig, SortConfig, SortDirection, SortField, ViewSummary, ViewerConfig, ViewerTheme,
};

// Re-export nested validation result types
pub use nested_results::{
    LogicalConstraintContext, LogicalConstraintType, NestedValidationResults,
    NestedValidationViolation, NestedViolationBuilder, NodeConstraintResult,
    PropertyConstraintResult, QualifiedConstraintContext, RootCause, ShapeConstraintContext,
    ShapeValidationType, ToNestedViolation,
};

// Re-export documentation generator types
pub use documentation::{
    DocumentationBuilder, DocumentationConfig, DocumentationFormat, ShapeDocumentationGenerator,
};

// Re-export visualizer types
pub use visualizer::{
    ChartSettings, ColorTheme, GroupBy, HeatmapEntry, ReportVisualizer, SummaryMetrics,
    TimelineEntry, ViolationDetail, VisualizationData, VisualizationFormat, VisualizerConfig,
    VisualizerConfigBuilder,
};

// Re-export validation types
pub use crate::validation::AsyncValidationStats;

/// Serialize a `ValidationReport` to SHACL-standard Turtle format.
///
/// Emits the SHACL standard output format with:
/// - `sh:conforms` boolean
/// - `sh:result` for each `sh:ValidationResult`
/// - `sh:focusNode`, `sh:resultPath`, `sh:value`, `sh:resultMessage`, `sh:resultSeverity`
///
/// # Errors
///
/// Returns an error if the serializer encounters an internal error.
pub fn serialize_to_turtle(report: &ValidationReport) -> crate::Result<String> {
    report.to_turtle()
}

#[cfg(test)]
mod report_turtle_tests {
    use super::*;
    use crate::{validation::ValidationViolation, ConstraintComponentId, Severity, ShapeId};
    use oxirs_core::model::{NamedNode, Term};
    use std::collections::HashMap;

    fn simple_report() -> ValidationReport {
        ValidationReport::new()
    }

    fn make_focus_node(iri: &str) -> Term {
        Term::NamedNode(NamedNode::new(iri).expect("valid IRI"))
    }

    fn violated_report() -> ValidationReport {
        let mut r = ValidationReport::new();
        let v = ValidationViolation {
            focus_node: make_focus_node("http://example.org/node1"),
            source_shape: ShapeId::new("http://example.org/MyShape"),
            source_constraint_component: ConstraintComponentId(
                "MinCountConstraintComponent".to_string(),
            ),
            result_path: None,
            value: None,
            result_message: Some("Value count below minimum".to_string()),
            result_severity: Severity::Violation,
            details: HashMap::new(),
            nested_results: vec![],
        };
        r.add_violation(v);
        r
    }

    // ---- serialize_to_turtle standalone function tests ----

    #[test]
    fn test_serialize_to_turtle_conforming_report() {
        let report = simple_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        assert!(turtle.contains("@prefix sh:"), "Should have sh: prefix");
        assert!(
            turtle.contains("sh:ValidationReport"),
            "Should declare ValidationReport"
        );
        assert!(turtle.contains("sh:conforms"), "Should include sh:conforms");
        assert!(
            turtle.contains("true"),
            "Conforming report should have true"
        );
    }

    #[test]
    fn test_serialize_to_turtle_violated_report_conforms_false() {
        let report = violated_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        assert!(
            turtle.contains("false"),
            "Violated report should have false for sh:conforms"
        );
    }

    #[test]
    fn test_serialize_to_turtle_violated_report_has_result() {
        let report = violated_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        assert!(
            turtle.contains("sh:result") || turtle.contains("sh:ValidationResult"),
            "Violated report should have sh:result or sh:ValidationResult"
        );
    }

    #[test]
    fn test_serialize_to_turtle_has_standard_prefixes() {
        let report = simple_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        assert!(turtle.contains("@prefix sh:"), "Should have sh: prefix");
        assert!(turtle.contains("@prefix rdf:"), "Should have rdf: prefix");
    }

    #[test]
    fn test_serialize_to_turtle_result_severity() {
        let report = violated_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        // Severity should appear in the output
        assert!(
            turtle.contains("Violation") || turtle.contains("sh:resultSeverity"),
            "Severity should appear in turtle output"
        );
    }

    #[test]
    fn test_serialize_to_turtle_focus_node_in_output() {
        let report = violated_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        assert!(
            turtle.contains("http://example.org/node1") || turtle.contains("sh:focusNode"),
            "Focus node IRI should appear in turtle output"
        );
    }

    #[test]
    fn test_serialize_to_turtle_result_message() {
        let report = violated_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        assert!(
            turtle.contains("Value count below minimum") || turtle.contains("sh:resultMessage"),
            "Result message should appear in turtle output"
        );
    }

    #[test]
    fn test_serialize_to_turtle_empty_report_no_results() {
        let report = simple_report();
        let turtle = serialize_to_turtle(&report).expect("serialize");
        // No violations, so no sh:result entries
        // The turtle should NOT have sh:ValidationResult for empty report
        assert!(
            !turtle.contains("sh:ValidationResult"),
            "Empty report should not have ValidationResult entries"
        );
    }

    #[test]
    fn test_to_turtle_method_matches_serialize_to_turtle() {
        let report = violated_report();
        let method_result = report.to_turtle().expect("method");
        let fn_result = serialize_to_turtle(&report).expect("function");
        assert_eq!(
            method_result, fn_result,
            "Both should produce identical output"
        );
    }

    #[test]
    fn test_serialize_to_turtle_multiple_violations() {
        let mut report = ValidationReport::new();
        for i in 1..=3 {
            let v = ValidationViolation {
                focus_node: make_focus_node(&format!("http://example.org/node{i}")),
                source_shape: ShapeId::new(format!("http://example.org/Shape{i}")),
                source_constraint_component: ConstraintComponentId(
                    "MinCountConstraintComponent".to_string(),
                ),
                result_path: None,
                value: None,
                result_message: Some(format!("Violation {i}")),
                result_severity: Severity::Violation,
                details: HashMap::new(),
                nested_results: vec![],
            };
            report.add_violation(v);
        }
        let turtle = serialize_to_turtle(&report).expect("serialize");
        assert!(
            turtle.contains("false"),
            "Report with 3 violations should not conform"
        );
    }

    // ---- ValidationReport.to_turtle() tests ----

    #[test]
    fn test_to_turtle_conforms_true_for_empty_report() {
        let report = ValidationReport::new();
        let turtle = report.to_turtle().expect("serialize");
        assert!(
            turtle.contains("true"),
            "Empty report should show conforms: true"
        );
    }

    #[test]
    fn test_to_turtle_conforms_false_after_violation() {
        let report = violated_report();
        let turtle = report.to_turtle().expect("serialize");
        assert!(
            turtle.contains("false"),
            "Violated report should show conforms: false"
        );
    }

    #[test]
    fn test_to_turtle_source_shape_in_output() {
        let report = violated_report();
        let turtle = report.to_turtle().expect("serialize");
        assert!(
            turtle.contains("http://example.org/MyShape") || turtle.contains("sh:sourceShape"),
            "Source shape should appear in output"
        );
    }

    #[test]
    fn test_to_rdf_turtle_format() {
        let report = violated_report();
        let rdf = report.to_rdf("turtle").expect("to_rdf");
        assert!(rdf.contains("@prefix sh:"), "Should have sh: prefix");
        assert!(rdf.contains("sh:ValidationReport"), "Should declare type");
        assert!(rdf.contains("sh:conforms"), "Should include conforms");
    }

    #[test]
    fn test_to_rdf_ntriples_format() {
        let report = simple_report();
        let rdf = report.to_rdf("nt").expect("to_rdf nt");
        // N-Triples format uses full IRIs without prefixes
        assert!(
            rdf.contains("shacl#ValidationReport"),
            "Should declare type using full IRI"
        );
        assert!(
            rdf.contains("shacl#conforms"),
            "Should include conforms using full IRI"
        );
    }

    #[test]
    fn test_to_rdf_unsupported_format_error() {
        let report = simple_report();
        assert!(
            report.to_rdf("unknown-format").is_err(),
            "Unknown format should return error"
        );
    }
}
