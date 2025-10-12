//! SHACL validation report implementation
//!
//! This module handles generation and serialization of SHACL validation reports.

#![allow(dead_code)]

pub mod advanced_filtering;
pub mod analytics;
pub mod core;
pub mod format;
pub mod generation;
pub mod generator;
pub mod interactive;
pub mod metadata;
pub mod nested_results;
pub mod serializers;
pub mod summary;

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

// Re-export validation types
pub use crate::validation::AsyncValidationStats;
