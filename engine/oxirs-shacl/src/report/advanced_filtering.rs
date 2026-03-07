//! Advanced filtering and querying capabilities for validation reports
//!
//! This module provides sophisticated filtering, querying, and interactive analysis
//! capabilities for SHACL validation reports, including SPARQL-based queries and
//! customizable report templates.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use oxirs_core::model::Term;

use crate::{
    report::{ReportFormat, ValidationReport},
    validation::ValidationViolation,
    ConstraintComponentId, Result, Severity, ShaclError, ShapeId,
};

/// Advanced filtering engine for validation reports
#[derive(Debug)]
pub struct ReportFilterEngine {
    /// Current filter configuration
    pub filter_config: FilterConfig,

    /// Query cache for performance
    query_cache: HashMap<String, CachedQueryResult>,

    /// Custom templates
    templates: HashMap<String, ReportTemplate>,
}

/// Configuration for report filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Severity-based filters
    pub severity_filters: Vec<Severity>,

    /// Shape-based filters
    pub shape_filters: Vec<ShapeId>,

    /// Path-based filters (property paths)
    pub path_filters: Vec<String>,

    /// Constraint component filters
    pub component_filters: Vec<ConstraintComponentId>,

    /// Focus node filters
    pub focus_node_filters: Vec<String>,

    /// Time range filter
    pub time_range: Option<TimeRange>,

    /// Maximum violations to include
    pub max_violations: Option<usize>,

    /// Include only violations with messages
    pub require_messages: bool,

    /// Custom filter predicates
    pub custom_filters: Vec<CustomFilter>,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            severity_filters: vec![Severity::Violation, Severity::Warning],
            shape_filters: Vec::new(),
            path_filters: Vec::new(),
            component_filters: Vec::new(),
            focus_node_filters: Vec::new(),
            time_range: None,
            max_violations: None,
            require_messages: false,
            custom_filters: Vec::new(),
        }
    }
}

/// Time range for filtering reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Custom filter predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomFilter {
    pub name: String,
    pub description: String,
    pub sparql_query: Option<String>,
    pub filter_function: FilterFunction,
}

/// Filter function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterFunction {
    /// Contains text in violation message
    ContainsText(String),

    /// Violation count threshold
    ViolationThreshold {
        operator: ComparisonOperator,
        value: usize,
    },

    /// Quality score threshold
    QualityThreshold {
        operator: ComparisonOperator,
        value: f64,
    },

    /// Custom SPARQL query
    SparqlQuery(String),

    /// Regular expression pattern matching
    RegexPattern { field: FilterField, pattern: String },
}

/// Comparison operators for filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Fields that can be filtered by regex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterField {
    ViolationMessage,
    FocusNode,
    ShapeId,
    PropertyPath,
    ConstraintComponent,
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedQueryResult {
    result: FilteredReport,
    timestamp: DateTime<Utc>,
    ttl_seconds: u64,
}

/// Filtered validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteredReport {
    /// Original report metadata
    pub original_report_id: String,

    /// Filtered violations
    pub filtered_violations: Vec<ValidationViolation>,

    /// Filter configuration applied
    pub applied_filters: FilterConfig,

    /// Filtering statistics
    pub filter_stats: FilterStatistics,

    /// Generated timestamp
    pub generated_at: DateTime<Utc>,
}

/// Statistics about filtering operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterStatistics {
    /// Original violation count
    pub original_violation_count: usize,

    /// Filtered violation count
    pub filtered_violation_count: usize,

    /// Filtering efficiency (percentage filtered out)
    pub filtering_efficiency: f64,

    /// Most common filtered constraint types
    pub filtered_constraint_types: HashMap<ConstraintComponentId, usize>,

    /// Most common filtered severities
    pub filtered_severities: HashMap<Severity, usize>,
}

/// SPARQL-based query interface for validation reports
#[derive(Debug)]
pub struct ReportQueryEngine {
    /// Available reports for querying
    reports: Vec<ValidationReport>,

    /// Query execution configuration
    query_config: QueryConfig,
}

/// Configuration for SPARQL query execution
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Maximum query execution time
    pub max_execution_time_ms: u64,

    /// Maximum result size
    pub max_result_size: usize,

    /// Enable query optimization
    pub enable_optimization: bool,

    /// Query cache TTL
    pub cache_ttl_seconds: u64,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 30000, // 30 seconds
            max_result_size: 10000,
            enable_optimization: true,
            cache_ttl_seconds: 3600, // 1 hour
        }
    }
}

/// Customizable report template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,

    /// Template description
    pub description: String,

    /// Output format
    pub format: ReportFormat,

    /// Template configuration
    pub config: TemplateConfig,

    /// Template content/structure
    pub template_content: TemplateContent,
}

/// Template configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// Include violation details
    pub include_violation_details: bool,

    /// Include summary statistics
    pub include_summary: bool,

    /// Include metadata
    pub include_metadata: bool,

    /// Include recommendations
    pub include_recommendations: bool,

    /// Group violations by shape
    pub group_by_shape: bool,

    /// Group violations by severity
    pub group_by_severity: bool,

    /// Custom styling for HTML reports
    pub custom_styling: Option<String>,

    /// Custom headers and footers
    pub custom_headers: HashMap<String, String>,
}

/// Template content structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateContent {
    /// Header template
    pub header: Option<String>,

    /// Body template
    pub body: String,

    /// Footer template
    pub footer: Option<String>,

    /// Custom CSS for HTML templates
    pub css: Option<String>,

    /// Custom JavaScript for interactive templates
    pub javascript: Option<String>,
}

impl ReportFilterEngine {
    /// Create a new filter engine
    pub fn new() -> Self {
        Self {
            filter_config: FilterConfig::default(),
            query_cache: HashMap::new(),
            templates: HashMap::new(),
        }
    }

    /// Create filter engine with configuration
    pub fn with_config(config: FilterConfig) -> Self {
        Self {
            filter_config: config,
            query_cache: HashMap::new(),
            templates: HashMap::new(),
        }
    }

    /// Apply filters to a validation report
    pub fn filter_report(&mut self, report: &ValidationReport) -> Result<FilteredReport> {
        let original_count = report.violations.len();
        let mut filtered_violations = report.violations.clone();

        // Apply severity filters
        if !self.filter_config.severity_filters.is_empty() {
            filtered_violations.retain(|v| {
                self.filter_config
                    .severity_filters
                    .contains(&v.result_severity)
            });
        }

        // Apply shape filters
        if !self.filter_config.shape_filters.is_empty() {
            filtered_violations
                .retain(|v| self.filter_config.shape_filters.contains(&v.source_shape));
        }

        // Apply path filters
        if !self.filter_config.path_filters.is_empty() {
            filtered_violations.retain(|v| {
                if let Some(path) = &v.result_path {
                    self.filter_config
                        .path_filters
                        .iter()
                        .any(|filter_path| path.to_string().contains(filter_path))
                } else {
                    false
                }
            });
        }

        // Apply constraint component filters
        if !self.filter_config.component_filters.is_empty() {
            filtered_violations.retain(|v| {
                self.filter_config
                    .component_filters
                    .contains(&v.source_constraint_component)
            });
        }

        // Apply focus node filters
        if !self.filter_config.focus_node_filters.is_empty() {
            filtered_violations.retain(|v| {
                self.filter_config
                    .focus_node_filters
                    .iter()
                    .any(|filter_node| v.focus_node.to_string().contains(filter_node))
            });
        }

        // Apply message requirement filter
        if self.filter_config.require_messages {
            filtered_violations.retain(|v| v.result_message.is_some());
        }

        // Apply custom filters
        for custom_filter in &self.filter_config.custom_filters {
            filtered_violations = self.apply_custom_filter(&filtered_violations, custom_filter)?;
        }

        // Apply maximum violations limit
        if let Some(max_violations) = self.filter_config.max_violations {
            filtered_violations.truncate(max_violations);
        }

        let filtered_count = filtered_violations.len();
        let filtering_efficiency = if original_count > 0 {
            ((original_count - filtered_count) as f64 / original_count as f64) * 100.0
        } else {
            0.0
        };

        // Compute filter statistics
        let filter_stats = self.compute_filter_statistics(
            original_count,
            filtered_count,
            filtering_efficiency,
            &filtered_violations,
        );

        Ok(FilteredReport {
            original_report_id: "report".to_string(), // In practice, would use actual report ID
            filtered_violations,
            applied_filters: self.filter_config.clone(),
            filter_stats,
            generated_at: Utc::now(),
        })
    }

    /// Apply custom filter to violations
    fn apply_custom_filter(
        &self,
        violations: &[ValidationViolation],
        custom_filter: &CustomFilter,
    ) -> Result<Vec<ValidationViolation>> {
        match &custom_filter.filter_function {
            FilterFunction::ContainsText(text) => Ok(violations
                .iter()
                .filter(|v| {
                    v.result_message
                        .as_ref()
                        .map(|msg| msg.contains(text))
                        .unwrap_or(false)
                })
                .cloned()
                .collect()),
            FilterFunction::ViolationThreshold {
                operator: _,
                value: _,
            } => {
                // This would typically be applied at the report level, not violation level
                // For now, we'll pass through all violations
                Ok(violations.to_vec())
            }
            FilterFunction::QualityThreshold { .. } => {
                // This would typically be applied at the report level
                Ok(violations.to_vec())
            }
            FilterFunction::SparqlQuery(_query) => {
                // SPARQL filtering would require more complex implementation
                // For now, return all violations
                Ok(violations.to_vec())
            }
            FilterFunction::RegexPattern { field, pattern } => {
                let regex = regex::Regex::new(pattern).map_err(|e| {
                    ShaclError::ReportGeneration(format!("Invalid regex pattern: {e}"))
                })?;

                Ok(violations
                    .iter()
                    .filter(|v| match field {
                        FilterField::ViolationMessage => v
                            .result_message
                            .as_ref()
                            .map(|msg| regex.is_match(msg))
                            .unwrap_or(false),
                        FilterField::FocusNode => regex.is_match(&v.focus_node.to_string()),
                        FilterField::ShapeId => regex.is_match(v.source_shape.as_str()),
                        FilterField::PropertyPath => v
                            .result_path
                            .as_ref()
                            .map(|path| regex.is_match(&path.to_string()))
                            .unwrap_or(false),
                        FilterField::ConstraintComponent => {
                            regex.is_match(v.source_constraint_component.as_str())
                        }
                    })
                    .cloned()
                    .collect())
            }
        }
    }

    /// Compute statistics about filtering operation
    fn compute_filter_statistics(
        &self,
        original_count: usize,
        filtered_count: usize,
        filtering_efficiency: f64,
        filtered_violations: &[ValidationViolation],
    ) -> FilterStatistics {
        let mut filtered_constraint_types = HashMap::new();
        let mut filtered_severities = HashMap::new();

        for violation in filtered_violations {
            *filtered_constraint_types
                .entry(violation.source_constraint_component.clone())
                .or_insert(0) += 1;

            *filtered_severities
                .entry(violation.result_severity)
                .or_insert(0) += 1;
        }

        FilterStatistics {
            original_violation_count: original_count,
            filtered_violation_count: filtered_count,
            filtering_efficiency,
            filtered_constraint_types,
            filtered_severities,
        }
    }

    /// Set filter configuration
    pub fn set_filter_config(&mut self, config: FilterConfig) {
        self.filter_config = config;
    }

    /// Add severity filter
    pub fn add_severity_filter(&mut self, severity: Severity) {
        if !self.filter_config.severity_filters.contains(&severity) {
            self.filter_config.severity_filters.push(severity);
        }
    }

    /// Add shape filter
    pub fn add_shape_filter(&mut self, shape_id: ShapeId) {
        if !self.filter_config.shape_filters.contains(&shape_id) {
            self.filter_config.shape_filters.push(shape_id);
        }
    }

    /// Add path filter
    pub fn add_path_filter(&mut self, path: String) {
        if !self.filter_config.path_filters.contains(&path) {
            self.filter_config.path_filters.push(path);
        }
    }

    /// Clear all filters
    pub fn clear_filters(&mut self) {
        self.filter_config = FilterConfig::default();
    }

    /// Install a custom report template
    pub fn install_template(&mut self, template: ReportTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Generate report using custom template
    pub fn generate_report_with_template(
        &self,
        report: &ValidationReport,
        template_name: &str,
    ) -> Result<String> {
        let template = self.templates.get(template_name).ok_or_else(|| {
            ShaclError::ReportGeneration(format!("Template '{template_name}' not found"))
        })?;

        self.apply_template(report, template)
    }

    /// Apply template to generate formatted report
    fn apply_template(
        &self,
        report: &ValidationReport,
        template: &ReportTemplate,
    ) -> Result<String> {
        match template.format {
            ReportFormat::Html => self.generate_html_from_template(report, template),
            ReportFormat::Json => self.generate_json_from_template(report, template),
            ReportFormat::Text => self.generate_text_from_template(report, template),
            _ => Err(ShaclError::ReportGeneration(format!(
                "Template format {:?} not supported",
                template.format
            ))),
        }
    }

    /// Generate HTML report from template
    fn generate_html_from_template(
        &self,
        report: &ValidationReport,
        template: &ReportTemplate,
    ) -> Result<String> {
        let mut html = String::new();

        // Add custom CSS if provided
        if let Some(css) = &template.template_content.css {
            html.push_str(&format!("<style>{css}</style>\n"));
        }

        // Add header if provided
        if let Some(header) = &template.template_content.header {
            html.push_str(&self.interpolate_template_variables(header, report)?);
        }

        // Add body
        html.push_str(
            &self.interpolate_template_variables(&template.template_content.body, report)?,
        );

        // Add footer if provided
        if let Some(footer) = &template.template_content.footer {
            html.push_str(&self.interpolate_template_variables(footer, report)?);
        }

        // Add custom JavaScript if provided
        if let Some(js) = &template.template_content.javascript {
            html.push_str(&format!("<script>{js}</script>\n"));
        }

        Ok(html)
    }

    /// Generate JSON report from template
    fn generate_json_from_template(
        &self,
        report: &ValidationReport,
        _template: &ReportTemplate,
    ) -> Result<String> {
        // For JSON templates, we can use the existing JSON serialization with filters
        serde_json::to_string_pretty(report).map_err(|e| {
            ShaclError::ReportGeneration(format!("JSON template generation failed: {e}"))
        })
    }

    /// Generate text report from template
    fn generate_text_from_template(
        &self,
        report: &ValidationReport,
        template: &ReportTemplate,
    ) -> Result<String> {
        let mut text = String::new();

        // Add header if provided
        if let Some(header) = &template.template_content.header {
            text.push_str(&self.interpolate_template_variables(header, report)?);
            text.push('\n');
        }

        // Add body
        text.push_str(
            &self.interpolate_template_variables(&template.template_content.body, report)?,
        );

        // Add footer if provided
        if let Some(footer) = &template.template_content.footer {
            text.push('\n');
            text.push_str(&self.interpolate_template_variables(footer, report)?);
        }

        Ok(text)
    }

    /// Interpolate template variables with actual report data
    fn interpolate_template_variables(
        &self,
        template: &str,
        report: &ValidationReport,
    ) -> Result<String> {
        let mut result = template.to_string();

        // Replace common template variables
        result = result.replace("{{CONFORMS}}", &report.conforms.to_string());
        result = result.replace("{{VIOLATION_COUNT}}", &report.violations.len().to_string());
        result = result.replace(
            "{{TIMESTAMP}}",
            &Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        );

        // Count violations by severity
        let error_count = report
            .violations
            .iter()
            .filter(|v| v.result_severity == Severity::Violation)
            .count();
        let warning_count = report
            .violations
            .iter()
            .filter(|v| v.result_severity == Severity::Warning)
            .count();

        result = result.replace("{{ERROR_COUNT}}", &error_count.to_string());
        result = result.replace("{{WARNING_COUNT}}", &warning_count.to_string());

        // Add violation details if requested
        if result.contains("{{VIOLATION_DETAILS}}") {
            let mut violation_details = String::new();
            for (i, violation) in report.violations.iter().enumerate() {
                violation_details.push_str(&format!(
                    "{}. {} - {} ({})\n",
                    i + 1,
                    violation.result_severity,
                    violation.focus_node,
                    violation.source_shape
                ));
                if let Some(message) = &violation.result_message {
                    violation_details.push_str(&format!("   {message}\n"));
                }
            }
            result = result.replace("{{VIOLATION_DETAILS}}", &violation_details);
        }

        Ok(result)
    }
}

impl ReportQueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self {
            reports: Vec::new(),
            query_config: QueryConfig::default(),
        }
    }

    /// Add reports to the query engine
    pub fn add_reports(&mut self, reports: Vec<ValidationReport>) {
        self.reports.extend(reports);
    }

    /// Execute a SPARQL query against validation reports
    pub fn execute_sparql_query(&self, _query: &str) -> Result<QueryResult> {
        // This is a simplified implementation
        // In a full implementation, this would convert validation reports to RDF
        // and execute the SPARQL query using a proper SPARQL engine

        // For now, return a placeholder result
        Ok(QueryResult {
            bindings: Vec::new(),
            execution_time_ms: 0,
            result_count: 0,
        })
    }

    /// Query reports by criteria
    pub fn query_reports(&self, criteria: &QueryCriteria) -> Result<Vec<&ValidationReport>> {
        let mut filtered_reports: Vec<&ValidationReport> = self.reports.iter().collect();

        // Apply time range filter
        if let Some(time_range) = &criteria.time_range {
            filtered_reports.retain(|_report| {
                let report_time = Utc::now(); // In practice, would get from report metadata
                report_time >= time_range.start && report_time <= time_range.end
            });
        }

        // Apply violation count filter
        if let Some(min_violations) = criteria.min_violations {
            filtered_reports.retain(|report| report.violations.len() >= min_violations);
        }

        if let Some(max_violations) = criteria.max_violations {
            filtered_reports.retain(|report| report.violations.len() <= max_violations);
        }

        // Apply conformance filter
        if let Some(must_conform) = criteria.must_conform {
            filtered_reports.retain(|report| report.conforms == must_conform);
        }

        Ok(filtered_reports)
    }
}

/// Query criteria for searching reports
#[derive(Debug, Clone)]
pub struct QueryCriteria {
    pub time_range: Option<TimeRange>,
    pub min_violations: Option<usize>,
    pub max_violations: Option<usize>,
    pub must_conform: Option<bool>,
    pub shape_filters: Vec<ShapeId>,
    pub severity_filters: Vec<Severity>,
}

/// Result of SPARQL query execution
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub bindings: Vec<HashMap<String, Term>>,
    pub execution_time_ms: u64,
    pub result_count: usize,
}

impl Default for ReportFilterEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ReportQueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{validation::ValidationViolation, ShapeId};
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_filter_engine_creation() {
        let engine = ReportFilterEngine::new();
        assert_eq!(engine.filter_config.severity_filters.len(), 2);
    }

    #[test]
    fn test_severity_filtering() {
        let mut engine = ReportFilterEngine::new();
        engine.filter_config.severity_filters = vec![Severity::Violation];

        let mut report = ValidationReport::new();
        report.add_violation(ValidationViolation {
            focus_node: oxirs_core::model::Term::NamedNode(
                NamedNode::new("http://example.org/error").expect("valid IRI"),
            ),
            result_severity: Severity::Violation,
            source_shape: ShapeId::new("ErrorShape"),
            source_constraint_component: crate::ConstraintComponentId::new("TestConstraint"),
            result_path: None,
            value: None,
            result_message: Some("Test error".to_string()),
            details: HashMap::new(),
            nested_results: Vec::new(),
        });

        report.add_violation(ValidationViolation {
            focus_node: oxirs_core::model::Term::NamedNode(
                NamedNode::new("http://example.org/warning").expect("valid IRI"),
            ),
            result_severity: Severity::Warning,
            source_shape: ShapeId::new("WarningShape"),
            source_constraint_component: crate::ConstraintComponentId::new("TestConstraint"),
            result_path: None,
            value: None,
            result_message: Some("Test warning".to_string()),
            details: HashMap::new(),
            nested_results: Vec::new(),
        });

        let filtered = engine
            .filter_report(&report)
            .expect("operation should succeed");
        assert_eq!(filtered.filtered_violations.len(), 1);
        assert_eq!(
            filtered.filtered_violations[0].result_severity,
            Severity::Violation
        );
    }

    #[test]
    fn test_template_installation() {
        let mut engine = ReportFilterEngine::new();

        let template = ReportTemplate {
            name: "test_template".to_string(),
            description: "Test template".to_string(),
            format: ReportFormat::Html,
            config: TemplateConfig {
                include_violation_details: true,
                include_summary: true,
                include_metadata: false,
                include_recommendations: false,
                group_by_shape: false,
                group_by_severity: true,
                custom_styling: None,
                custom_headers: HashMap::new(),
            },
            template_content: TemplateContent {
                header: Some("<h1>Validation Report</h1>".to_string()),
                body: "<p>{{VIOLATION_COUNT}} violations found</p>".to_string(),
                footer: Some("<p>Generated at {{TIMESTAMP}}</p>".to_string()),
                css: None,
                javascript: None,
            },
        };

        engine.install_template(template);
        assert!(engine.templates.contains_key("test_template"));
    }

    #[test]
    fn test_query_engine() {
        let mut engine = ReportQueryEngine::new();
        let report = ValidationReport::new();
        engine.add_reports(vec![report]);

        let criteria = QueryCriteria {
            time_range: None,
            min_violations: Some(0),
            max_violations: None,
            must_conform: Some(true),
            shape_filters: Vec::new(),
            severity_filters: Vec::new(),
        };

        let results = engine
            .query_reports(&criteria)
            .expect("query should succeed");
        assert_eq!(results.len(), 1);
    }
}
