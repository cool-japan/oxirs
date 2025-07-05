//! Interactive web-based validation report viewer
//!
//! This module provides comprehensive interactive reporting capabilities including
//! web-based report viewers, real-time filtering, sorting, and export functionality.

use std::collections::{HashMap, BTreeMap};
use std::fmt::Write;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    report::{ValidationReport, ReportFilterEngine, FilterConfig},
    validation::ValidationViolation,
    ConstraintComponentId, Result, Severity, ShaclError, ShapeId,
};

/// Interactive web-based report viewer
#[derive(Debug)]
pub struct InteractiveReportViewer {
    /// Current report being viewed
    current_report: Option<ValidationReport>,
    
    /// Filter engine for interactive filtering
    filter_engine: ReportFilterEngine,
    
    /// Viewer configuration
    config: ViewerConfig,
    
    /// Current sorting configuration
    sort_config: SortConfig,
    
    /// Pagination configuration
    pagination: PaginationConfig,
}

/// Configuration for the interactive viewer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewerConfig {
    /// Enable real-time filtering
    pub enable_real_time_filtering: bool,
    
    /// Enable sorting capabilities
    pub enable_sorting: bool,
    
    /// Enable pagination
    pub enable_pagination: bool,
    
    /// Enable export functionality
    pub enable_export: bool,
    
    /// Enable violation details expansion
    pub enable_details_expansion: bool,
    
    /// Maximum violations per page
    pub max_violations_per_page: usize,
    
    /// Theme configuration
    pub theme: ViewerTheme,
    
    /// Custom CSS styling
    pub custom_css: Option<String>,
    
    /// Enable search functionality
    pub enable_search: bool,
    
    /// Enable charts and visualizations
    pub enable_charts: bool,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            enable_real_time_filtering: true,
            enable_sorting: true,
            enable_pagination: true,
            enable_export: true,
            enable_details_expansion: true,
            max_violations_per_page: 50,
            theme: ViewerTheme::Light,
            custom_css: None,
            enable_search: true,
            enable_charts: true,
        }
    }
}

/// Visual theme for the viewer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewerTheme {
    Light,
    Dark,
    HighContrast,
    Custom(String),
}

/// Sorting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortConfig {
    /// Field to sort by
    pub sort_field: SortField,
    
    /// Sort direction
    pub sort_direction: SortDirection,
    
    /// Secondary sort field
    pub secondary_sort: Option<SortField>,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            sort_field: SortField::Severity,
            sort_direction: SortDirection::Descending,
            secondary_sort: Some(SortField::Shape),
        }
    }
}

/// Fields that can be used for sorting
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortField {
    Severity,
    Shape,
    ConstraintComponent,
    FocusNode,
    Message,
    Path,
}

/// Sort direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Pagination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationConfig {
    /// Current page number (0-based)
    pub current_page: usize,
    
    /// Items per page
    pub items_per_page: usize,
    
    /// Total items available
    pub total_items: usize,
    
    /// Enable page navigation
    pub enable_navigation: bool,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            current_page: 0,
            items_per_page: 25,
            total_items: 0,
            enable_navigation: true,
        }
    }
}

/// Interactive report view state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveReportView {
    /// Filtered and sorted violations
    pub violations: Vec<ValidationViolation>,
    
    /// Summary statistics for current view
    pub view_summary: ViewSummary,
    
    /// Applied filters
    pub active_filters: FilterConfig,
    
    /// Current sorting
    pub sort_config: SortConfig,
    
    /// Pagination state
    pub pagination: PaginationConfig,
    
    /// Search query if any
    pub search_query: Option<String>,
    
    /// Available filter options
    pub filter_options: FilterOptions,
}

/// Summary statistics for the current view
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewSummary {
    /// Total violations in current view
    pub total_violations: usize,
    
    /// Violations by severity in current view
    pub violations_by_severity: BTreeMap<String, usize>,
    
    /// Violations by shape in current view
    pub violations_by_shape: BTreeMap<String, usize>,
    
    /// Violations by constraint component in current view
    pub violations_by_component: BTreeMap<String, usize>,
    
    /// Unique focus nodes in current view
    pub unique_focus_nodes: usize,
    
    /// Conformance status
    pub conforms: bool,
}

/// Available filter options based on current data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterOptions {
    /// Available severity levels
    pub available_severities: Vec<Severity>,
    
    /// Available shapes
    pub available_shapes: Vec<ShapeId>,
    
    /// Available constraint components
    pub available_components: Vec<ConstraintComponentId>,
    
    /// Available property paths
    pub available_paths: Vec<String>,
    
    /// Available focus node patterns
    pub available_focus_node_patterns: Vec<String>,
}

/// Export configuration for interactive reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Export format
    pub format: ExportFormat,
    
    /// Include current filters
    pub include_filters: bool,
    
    /// Include summary statistics
    pub include_summary: bool,
    
    /// Include charts/visualizations
    pub include_charts: bool,
    
    /// Custom filename
    pub filename: Option<String>,
}

/// Available export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Pdf,
    Excel,
    Csv,
    Json,
    Html,
    Xml,
}

impl InteractiveReportViewer {
    /// Create a new interactive report viewer
    pub fn new(config: ViewerConfig) -> Self {
        Self {
            current_report: None,
            filter_engine: ReportFilterEngine::new(),
            config,
            sort_config: SortConfig::default(),
            pagination: PaginationConfig::default(),
        }
    }
    
    /// Load a validation report into the viewer
    pub fn load_report(&mut self, report: ValidationReport) -> Result<()> {
        self.pagination.total_items = report.violations.len();
        self.pagination.current_page = 0;
        self.current_report = Some(report);
        Ok(())
    }
    
    /// Generate the current interactive view
    pub fn generate_view(&mut self) -> Result<InteractiveReportView> {
        let report = self.current_report.as_ref().ok_or_else(|| {
            ShaclError::ReportGeneration("No report loaded".to_string())
        })?;
        
        // Apply filters
        let filtered_report = self.filter_engine.filter_report(report)?;
        let mut violations = filtered_report.filtered_violations;
        
        // Apply sorting
        self.sort_violations(&mut violations);
        
        // Calculate summary for current view
        let view_summary = self.calculate_view_summary(&violations, report.conforms);
        
        // Apply pagination
        let paginated_violations = if self.config.enable_pagination {
            let start = self.pagination.current_page * self.pagination.items_per_page;
            let end = std::cmp::min(start + self.pagination.items_per_page, violations.len());
            violations[start..end].to_vec()
        } else {
            violations
        };
        
        // Generate filter options
        let filter_options = self.generate_filter_options(report);
        
        Ok(InteractiveReportView {
            violations: paginated_violations,
            view_summary,
            active_filters: self.filter_engine.filter_config.clone(),
            sort_config: self.sort_config.clone(),
            pagination: self.pagination.clone(),
            search_query: None,
            filter_options,
        })
    }
    
    /// Generate HTML for interactive web viewer
    pub fn generate_html_viewer(&mut self) -> Result<String> {
        let view = self.generate_view()?;
        
        let mut html = String::new();
        
        // HTML structure
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str("<title>Interactive SHACL Validation Report</title>\n");
        
        // Add CSS
        html.push_str(&self.generate_css());
        
        html.push_str("</head>\n<body>\n");
        
        // Header
        html.push_str("<div class=\"header\">\n");
        html.push_str("<h1>🛡️ SHACL Validation Report</h1>\n");
        html.push_str(&format!("<div class=\"status {}\">{}</div>\n",
            if view.view_summary.conforms { "conforming" } else { "non-conforming" },
            if view.view_summary.conforms { "✅ CONFORMING" } else { "❌ NON-CONFORMING" }
        ));
        html.push_str("</div>\n");
        
        // Summary section
        html.push_str(&self.generate_summary_section(&view.view_summary));
        
        // Charts section (if enabled)
        if self.config.enable_charts {
            html.push_str(&self.generate_charts_section(&view.view_summary));
        }
        
        // Filter section
        if self.config.enable_real_time_filtering {
            html.push_str(&self.generate_filter_section(&view.filter_options));
        }
        
        // Search section
        if self.config.enable_search {
            html.push_str(&self.generate_search_section());
        }
        
        // Violations table
        html.push_str(&self.generate_violations_table(&view.violations));
        
        // Pagination
        if self.config.enable_pagination {
            html.push_str(&self.generate_pagination_section());
        }
        
        // Export section
        if self.config.enable_export {
            html.push_str(&self.generate_export_section());
        }
        
        // JavaScript
        html.push_str(&self.generate_javascript());
        
        html.push_str("</body>\n</html>");
        
        Ok(html)
    }
    
    /// Sort violations based on current sort configuration
    fn sort_violations(&self, violations: &mut Vec<ValidationViolation>) {
        violations.sort_by(|a, b| {
            let primary_cmp = match self.sort_config.sort_field {
                SortField::Severity => a.result_severity.cmp(&b.result_severity),
                SortField::Shape => a.source_shape.cmp(&b.source_shape),
                SortField::ConstraintComponent => a.source_constraint_component.cmp(&b.source_constraint_component),
                SortField::FocusNode => a.focus_node.to_string().cmp(&b.focus_node.to_string()),
                SortField::Message => {
                    a.result_message.as_deref().unwrap_or("").cmp(
                        b.result_message.as_deref().unwrap_or("")
                    )
                }
                SortField::Path => {
                    a.result_path.as_ref().map(|p| p.to_string()).unwrap_or_default().cmp(
                        &b.result_path.as_ref().map(|p| p.to_string()).unwrap_or_default()
                    )
                }
            };
            
            match self.sort_config.sort_direction {
                SortDirection::Ascending => primary_cmp,
                SortDirection::Descending => primary_cmp.reverse(),
            }
        });
    }
    
    /// Calculate summary statistics for current view
    fn calculate_view_summary(&self, violations: &[ValidationViolation], conforms: bool) -> ViewSummary {
        let mut violations_by_severity = BTreeMap::new();
        let mut violations_by_shape = BTreeMap::new();
        let mut violations_by_component = BTreeMap::new();
        let mut unique_focus_nodes = std::collections::HashSet::new();
        
        for violation in violations {
            // Count by severity
            *violations_by_severity
                .entry(violation.result_severity.to_string())
                .or_insert(0) += 1;
                
            // Count by shape
            *violations_by_shape
                .entry(violation.source_shape.to_string())
                .or_insert(0) += 1;
                
            // Count by component
            *violations_by_component
                .entry(violation.source_constraint_component.to_string())
                .or_insert(0) += 1;
                
            // Track unique focus nodes
            unique_focus_nodes.insert(violation.focus_node.to_string());
        }
        
        ViewSummary {
            total_violations: violations.len(),
            violations_by_severity,
            violations_by_shape,
            violations_by_component,
            unique_focus_nodes: unique_focus_nodes.len(),
            conforms,
        }
    }
    
    /// Generate filter options based on available data
    fn generate_filter_options(&self, report: &ValidationReport) -> FilterOptions {
        let mut available_severities = std::collections::HashSet::new();
        let mut available_shapes = std::collections::HashSet::new();
        let mut available_components = std::collections::HashSet::new();
        let mut available_paths = std::collections::HashSet::new();
        let mut focus_node_patterns = std::collections::HashSet::new();
        
        for violation in &report.violations {
            available_severities.insert(violation.result_severity.clone());
            available_shapes.insert(violation.source_shape.clone());
            available_components.insert(violation.source_constraint_component.clone());
            
            if let Some(path) = &violation.result_path {
                available_paths.insert(path.to_string());
            }
            
            // Extract patterns from focus nodes (e.g., domain names)
            let focus_node_str = violation.focus_node.to_string();
            if let Some(domain_start) = focus_node_str.find("://") {
                if let Some(domain_end) = focus_node_str[domain_start + 3..].find('/') {
                    let domain = &focus_node_str[domain_start + 3..domain_start + 3 + domain_end];
                    focus_node_patterns.insert(domain.to_string());
                }
            }
        }
        
        FilterOptions {
            available_severities: available_severities.into_iter().collect(),
            available_shapes: available_shapes.into_iter().collect(),
            available_components: available_components.into_iter().collect(),
            available_paths: available_paths.into_iter().collect(),
            available_focus_node_patterns: focus_node_patterns.into_iter().collect(),
        }
    }
    
    /// Generate CSS for the interactive viewer
    fn generate_css(&self) -> String {
        let theme_colors = match self.config.theme {
            ViewerTheme::Light => ("white", "#333", "#f5f5f5", "#007bff"),
            ViewerTheme::Dark => ("#333", "white", "#444", "#0d6efd"),
            ViewerTheme::HighContrast => ("black", "white", "#666", "yellow"),
            ViewerTheme::Custom(_) => ("white", "#333", "#f5f5f5", "#007bff"),
        };
        
        format!(r#"
        <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {};
            color: {};
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: {};
            border-radius: 8px;
        }}
        
        .status.conforming {{ color: #28a745; }}
        .status.non-conforming {{ color: #dc3545; }}
        
        .summary-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background-color: {};
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .summary-card h3 {{ margin-top: 0; }}
        .summary-number {{ font-size: 2em; font-weight: bold; color: {}; }}
        
        .filter-section {{
            background-color: {};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .filter-row {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        
        .violations-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: {};
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .violations-table th,
        .violations-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .violations-table th {{
            background-color: {};
            cursor: pointer;
        }}
        
        .violations-table tr:hover {{
            background-color: rgba(0,123,255,0.1);
        }}
        
        .severity-violation {{ color: #dc3545; font-weight: bold; }}
        .severity-warning {{ color: #ffc107; font-weight: bold; }}
        .severity-info {{ color: #17a2b8; font-weight: bold; }}
        
        .pagination {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 30px 0;
        }}
        
        .pagination button {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            background-color: {};
            color: {};
            cursor: pointer;
            border-radius: 4px;
        }}
        
        .pagination button:hover {{
            background-color: {};
        }}
        
        .pagination button.active {{
            background-color: {};
            color: white;
        }}
        
        .export-section {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background-color: {};
            border-radius: 8px;
        }}
        
        .btn {{
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background-color: {};
            color: white;
        }}
        
        .btn:hover {{
            opacity: 0.8;
        }}
        
        .search-section {{
            margin-bottom: 20px;
        }}
        
        .search-input {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }}
        
        .charts-section {{
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .chart-container {{
            background-color: {};
            padding: 20px;
            border-radius: 8px;
        }}
        </style>
        "#,
        theme_colors.0, theme_colors.1, theme_colors.2, theme_colors.3,
        theme_colors.2, theme_colors.3, theme_colors.2, theme_colors.0,
        theme_colors.2, theme_colors.0, theme_colors.1, theme_colors.2,
        theme_colors.3, theme_colors.3, theme_colors.2
        )
    }
    
    /// Generate summary section HTML
    fn generate_summary_section(&self, summary: &ViewSummary) -> String {
        format!(r#"
        <div class="summary-section">
            <div class="summary-card">
                <h3>Total Violations</h3>
                <div class="summary-number">{}</div>
            </div>
            <div class="summary-card">
                <h3>Unique Nodes</h3>
                <div class="summary-number">{}</div>
            </div>
            <div class="summary-card">
                <h3>Errors</h3>
                <div class="summary-number">{}</div>
            </div>
            <div class="summary-card">
                <h3>Warnings</h3>
                <div class="summary-number">{}</div>
            </div>
        </div>
        "#,
        summary.total_violations,
        summary.unique_focus_nodes,
        summary.violations_by_severity.get("Violation").unwrap_or(&0),
        summary.violations_by_severity.get("Warning").unwrap_or(&0)
        )
    }
    
    /// Generate charts section HTML
    fn generate_charts_section(&self, summary: &ViewSummary) -> String {
        // This would typically include Chart.js or similar library
        format!(r#"
        <div class="charts-section">
            <div class="chart-container">
                <h3>Violations by Severity</h3>
                <canvas id="severityChart" width="300" height="200"></canvas>
            </div>
            <div class="chart-container">
                <h3>Violations by Shape</h3>
                <canvas id="shapeChart" width="300" height="200"></canvas>
            </div>
        </div>
        "#)
    }
    
    /// Generate filter section HTML
    fn generate_filter_section(&self, options: &FilterOptions) -> String {
        let mut html = String::new();
        html.push_str("<div class=\"filter-section\">\n");
        html.push_str("<h3>🔍 Filters</h3>\n");
        html.push_str("<div class=\"filter-row\">\n");
        
        // Severity filter
        html.push_str("<div>\n");
        html.push_str("<label>Severity:</label><br>\n");
        for severity in &options.available_severities {
            html.push_str(&format!(
                "<input type=\"checkbox\" id=\"severity_{}\" name=\"severity\" value=\"{}\" checked>\n",
                severity, severity
            ));
            html.push_str(&format!("<label for=\"severity_{}\">{}</label><br>\n", severity, severity));
        }
        html.push_str("</div>\n");
        
        html.push_str("</div>\n");
        html.push_str("</div>\n");
        
        html
    }
    
    /// Generate search section HTML
    fn generate_search_section(&self) -> String {
        r#"
        <div class="search-section">
            <input type="text" class="search-input" placeholder="🔍 Search violations..." id="searchInput">
        </div>
        "#.to_string()
    }
    
    /// Generate violations table HTML
    fn generate_violations_table(&self, violations: &[ValidationViolation]) -> String {
        let mut html = String::new();
        
        html.push_str("<table class=\"violations-table\">\n");
        html.push_str("<thead>\n<tr>\n");
        html.push_str("<th onclick=\"sortBy('severity')\">Severity ↕️</th>\n");
        html.push_str("<th onclick=\"sortBy('shape')\">Shape ↕️</th>\n");
        html.push_str("<th onclick=\"sortBy('component')\">Constraint ↕️</th>\n");
        html.push_str("<th onclick=\"sortBy('node')\">Focus Node ↕️</th>\n");
        html.push_str("<th onclick=\"sortBy('message')\">Message ↕️</th>\n");
        html.push_str("</tr>\n</thead>\n<tbody>\n");
        
        for violation in violations {
            html.push_str("<tr>\n");
            
            // Severity
            let severity_class = match violation.result_severity {
                Severity::Violation => "severity-violation",
                Severity::Warning => "severity-warning",
                Severity::Info => "severity-info",
            };
            html.push_str(&format!(
                "<td class=\"{}\">🚨 {}</td>\n",
                severity_class, violation.result_severity
            ));
            
            // Shape
            html.push_str(&format!("<td>{}</td>\n", violation.source_shape));
            
            // Constraint Component
            html.push_str(&format!("<td>{}</td>\n", violation.source_constraint_component));
            
            // Focus Node
            html.push_str(&format!("<td><code>{}</code></td>\n", violation.focus_node));
            
            // Message
            let message = violation.result_message.as_deref().unwrap_or("No message");
            html.push_str(&format!("<td>{}</td>\n", html_escape(message)));
            
            html.push_str("</tr>\n");
        }
        
        html.push_str("</tbody>\n</table>\n");
        html
    }
    
    /// Generate pagination section HTML
    fn generate_pagination_section(&self) -> String {
        let total_pages = (self.pagination.total_items + self.pagination.items_per_page - 1) / self.pagination.items_per_page;
        
        let mut html = String::new();
        html.push_str("<div class=\"pagination\">\n");
        
        // Previous button
        if self.pagination.current_page > 0 {
            html.push_str(&format!(
                "<button onclick=\"goToPage({})\">« Previous</button>\n",
                self.pagination.current_page - 1
            ));
        }
        
        // Page numbers
        for page in 0..total_pages {
            let active_class = if page == self.pagination.current_page { " active" } else { "" };
            html.push_str(&format!(
                "<button class=\"{}\" onclick=\"goToPage({})\">{}</button>\n",
                active_class, page, page + 1
            ));
        }
        
        // Next button
        if self.pagination.current_page < total_pages - 1 {
            html.push_str(&format!(
                "<button onclick=\"goToPage({})\">Next »</button>\n",
                self.pagination.current_page + 1
            ));
        }
        
        html.push_str("</div>\n");
        html
    }
    
    /// Generate export section HTML
    fn generate_export_section(&self) -> String {
        r#"
        <div class="export-section">
            <h3>📁 Export Report</h3>
            <button class="btn" onclick="exportReport('pdf')">📄 Export as PDF</button>
            <button class="btn" onclick="exportReport('excel')">📊 Export as Excel</button>
            <button class="btn" onclick="exportReport('csv')">📋 Export as CSV</button>
            <button class="btn" onclick="exportReport('json')">🔧 Export as JSON</button>
        </div>
        "#.to_string()
    }
    
    /// Generate JavaScript for interactivity
    fn generate_javascript(&self) -> String {
        r#"
        <script>
        let currentSort = { field: 'severity', direction: 'desc' };
        let currentFilters = {};
        
        function sortBy(field) {
            if (currentSort.field === field) {
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            } else {
                currentSort.field = field;
                currentSort.direction = 'asc';
            }
            // In a real implementation, this would trigger a server request
            console.log('Sort by:', currentSort);
        }
        
        function goToPage(page) {
            // In a real implementation, this would trigger a server request
            console.log('Go to page:', page);
        }
        
        function exportReport(format) {
            // In a real implementation, this would trigger export
            console.log('Export as:', format);
            alert('Export functionality would be implemented here');
        }
        
        function applyFilters() {
            // Collect filter values
            const severityFilters = Array.from(document.querySelectorAll('input[name="severity"]:checked'))
                .map(cb => cb.value);
            
            currentFilters.severity = severityFilters;
            
            // In a real implementation, this would trigger filtering
            console.log('Apply filters:', currentFilters);
        }
        
        function searchViolations() {
            const query = document.getElementById('searchInput').value;
            // In a real implementation, this would trigger search
            console.log('Search:', query);
        }
        
        // Add event listeners
        document.addEventListener('DOMContentLoaded', function() {
            // Search as you type
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {
                searchInput.addEventListener('input', searchViolations);
            }
            
            // Filter checkboxes
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.addEventListener('change', applyFilters);
            });
        });
        </script>
        "#.to_string()
    }
    
    /// Update filter configuration
    pub fn update_filters(&mut self, config: FilterConfig) {
        self.filter_engine.set_filter_config(config);
    }
    
    /// Update sort configuration
    pub fn update_sort(&mut self, sort_config: SortConfig) {
        self.sort_config = sort_config;
    }
    
    /// Update pagination
    pub fn update_pagination(&mut self, page: usize, items_per_page: usize) {
        self.pagination.current_page = page;
        self.pagination.items_per_page = items_per_page;
    }
    
    /// Export current view to specified format
    pub fn export_view(&mut self, export_config: ExportConfig) -> Result<Vec<u8>> {
        let view = self.generate_view()?;
        
        match export_config.format {
            ExportFormat::Csv => self.export_to_csv(&view, &export_config),
            ExportFormat::Json => self.export_to_json(&view, &export_config),
            ExportFormat::Html => self.export_to_html(&view, &export_config),
            _ => Err(ShaclError::ReportGeneration(format!(
                "Export format {:?} not yet implemented",
                export_config.format
            ))),
        }
    }
    
    /// Export view to CSV
    fn export_to_csv(&self, view: &InteractiveReportView, _config: &ExportConfig) -> Result<Vec<u8>> {
        let mut csv = String::new();
        csv.push_str("Severity,Shape,Constraint Component,Focus Node,Message\n");
        
        for violation in &view.violations {
            csv.push_str(&format!(
                "{},{},{},{},\"{}\"\n",
                violation.result_severity,
                violation.source_shape,
                violation.source_constraint_component,
                violation.focus_node,
                violation.result_message.as_deref().unwrap_or("").replace("\"", "\"\"")
            ));
        }
        
        Ok(csv.into_bytes())
    }
    
    /// Export view to JSON
    fn export_to_json(&self, view: &InteractiveReportView, _config: &ExportConfig) -> Result<Vec<u8>> {
        let json = serde_json::to_string_pretty(view).map_err(|e| {
            ShaclError::ReportGeneration(format!("JSON export failed: {}", e))
        })?;
        Ok(json.into_bytes())
    }
    
    /// Export view to HTML
    fn export_to_html(&self, view: &InteractiveReportView, _config: &ExportConfig) -> Result<Vec<u8>> {
        let html = self.generate_violations_table(&view.violations);
        Ok(html.into_bytes())
    }
}

/// Escape HTML special characters
fn html_escape(text: &str) -> String {
    text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&#x27;")
}

impl Default for InteractiveReportViewer {
    fn default() -> Self {
        Self::new(ViewerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ShapeId, validation::ValidationViolation};
    use oxirs_core::model::NamedNode;
    
    #[test]
    fn test_interactive_viewer_creation() {
        let viewer = InteractiveReportViewer::new(ViewerConfig::default());
        assert!(viewer.current_report.is_none());
        assert_eq!(viewer.config.max_violations_per_page, 50);
    }
    
    #[test]
    fn test_load_report() {
        let mut viewer = InteractiveReportViewer::new(ViewerConfig::default());
        let report = ValidationReport::new();
        
        viewer.load_report(report).unwrap();
        assert!(viewer.current_report.is_some());
    }
    
    #[test]
    fn test_sorting_configuration() {
        let mut viewer = InteractiveReportViewer::new(ViewerConfig::default());
        
        let sort_config = SortConfig {
            sort_field: SortField::Shape,
            sort_direction: SortDirection::Ascending,
            secondary_sort: None,
        };
        
        viewer.update_sort(sort_config.clone());
        assert_eq!(viewer.sort_config.sort_field, SortField::Shape);
        assert_eq!(viewer.sort_config.sort_direction, SortDirection::Ascending);
    }
    
    #[test]
    fn test_pagination_update() {
        let mut viewer = InteractiveReportViewer::new(ViewerConfig::default());
        
        viewer.update_pagination(2, 25);
        assert_eq!(viewer.pagination.current_page, 2);
        assert_eq!(viewer.pagination.items_per_page, 25);
    }
    
    #[test]
    fn test_html_escape() {
        let input = "<script>alert('test')</script>";
        let escaped = html_escape(input);
        assert_eq!(escaped, "&lt;script&gt;alert(&#x27;test&#x27;)&lt;/script&gt;");
    }
}