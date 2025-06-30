//! SHACL validation report implementation
//!
//! This module handles generation and serialization of SHACL validation reports.

pub mod analytics;
pub mod core;
pub mod format;
pub mod generation;
pub mod generator;
pub mod metadata;
pub mod serializers;
pub mod summary;

// Re-export key types for convenience
pub use core::ValidationReport;
pub use format::ReportFormat;
pub use generator::{generate_report, ReportGenerator};
pub use metadata::ReportMetadata;
pub use summary::ValidationSummary;

// Re-export enhanced generation functions
pub use generation::{
    generate_csv_report, generate_html_report, generate_json_report, generate_jsonld_report,
    generate_ntriples_report, generate_rdfxml_report, generate_text_report, generate_turtle_report,
    generate_yaml_report,
};

// Re-export analytics types
pub use analytics::{
    AnalyticsConfig, AnalyzedReport, AsyncValidationStats, ReportMetrics, ValidationReportAnalytics,
};
