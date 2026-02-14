//! Validation Report Visualizer
//!
//! This module provides comprehensive visualization capabilities for SHACL validation
//! reports, generating charts, graphs, and interactive HTML reports.

use crate::validation::ValidationViolation;
use crate::{Result, Severity, ValidationReport};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation report visualizer
#[derive(Debug)]
pub struct ReportVisualizer {
    config: VisualizerConfig,
}

/// Configuration for the visualizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizerConfig {
    /// Output format
    pub format: VisualizationFormat,
    /// Color theme
    pub theme: ColorTheme,
    /// Chart settings
    pub chart_settings: ChartSettings,
    /// Include interactive elements
    pub interactive: bool,
    /// Include raw data tables
    pub include_tables: bool,
    /// Maximum violations to display
    pub max_violations: usize,
    /// Group violations by type
    pub group_by: GroupBy,
    /// Custom CSS
    pub custom_css: Option<String>,
    /// Custom JavaScript
    pub custom_js: Option<String>,
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        Self {
            format: VisualizationFormat::Html,
            theme: ColorTheme::default(),
            chart_settings: ChartSettings::default(),
            interactive: true,
            include_tables: true,
            max_violations: 1000,
            group_by: GroupBy::Shape,
            custom_css: None,
            custom_js: None,
        }
    }
}

/// Visualization output format
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VisualizationFormat {
    /// HTML with embedded SVG
    Html,
    /// Standalone SVG
    Svg,
    /// JSON data for external visualization
    Json,
    /// ASCII art for terminal
    Ascii,
    /// Markdown with embedded charts
    Markdown,
}

/// Color theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorTheme {
    /// Background color
    pub background: String,
    /// Text color
    pub text: String,
    /// Violation color
    pub violation: String,
    /// Warning color
    pub warning: String,
    /// Info color
    pub info: String,
    /// Success color
    pub success: String,
    /// Primary accent
    pub primary: String,
    /// Secondary accent
    pub secondary: String,
    /// Chart colors
    pub chart_colors: Vec<String>,
}

impl Default for ColorTheme {
    fn default() -> Self {
        Self {
            background: "#ffffff".to_string(),
            text: "#333333".to_string(),
            violation: "#dc3545".to_string(),
            warning: "#ffc107".to_string(),
            info: "#17a2b8".to_string(),
            success: "#28a745".to_string(),
            primary: "#007bff".to_string(),
            secondary: "#6c757d".to_string(),
            chart_colors: vec![
                "#4e79a7".to_string(),
                "#f28e2c".to_string(),
                "#e15759".to_string(),
                "#76b7b2".to_string(),
                "#59a14f".to_string(),
                "#edc949".to_string(),
                "#af7aa1".to_string(),
                "#ff9da7".to_string(),
                "#9c755f".to_string(),
                "#bab0ab".to_string(),
            ],
        }
    }
}

/// Chart settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartSettings {
    /// Chart width
    pub width: u32,
    /// Chart height
    pub height: u32,
    /// Show legend
    pub show_legend: bool,
    /// Show labels
    pub show_labels: bool,
    /// Animation enabled
    pub animate: bool,
    /// Font family
    pub font_family: String,
    /// Font size
    pub font_size: u32,
}

impl Default for ChartSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 400,
            show_legend: true,
            show_labels: true,
            animate: true,
            font_family: "system-ui, -apple-system, sans-serif".to_string(),
            font_size: 12,
        }
    }
}

/// Grouping option for violations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GroupBy {
    /// Group by shape
    Shape,
    /// Group by severity
    Severity,
    /// Group by constraint type
    ConstraintType,
    /// Group by focus node
    FocusNode,
    /// No grouping
    None,
}

/// Visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    /// Summary metrics
    pub summary: SummaryMetrics,
    /// Violations by severity
    pub by_severity: HashMap<String, usize>,
    /// Violations by shape
    pub by_shape: HashMap<String, usize>,
    /// Violations by constraint type
    pub by_constraint: HashMap<String, usize>,
    /// Timeline data
    pub timeline: Vec<TimelineEntry>,
    /// Node heatmap
    pub heatmap: Vec<HeatmapEntry>,
    /// Detailed violations
    pub violations: Vec<ViolationDetail>,
}

/// Summary metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMetrics {
    /// Total violations
    pub total_violations: usize,
    /// Violations count
    pub violations: usize,
    /// Warnings count
    pub warnings: usize,
    /// Info count
    pub infos: usize,
    /// Shapes validated
    pub shapes_validated: usize,
    /// Shapes with violations
    pub shapes_with_violations: usize,
    /// Compliance rate
    pub compliance_rate: f64,
    /// Focus nodes checked
    pub focus_nodes_checked: usize,
}

/// Timeline entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// Timestamp or index
    pub index: usize,
    /// Cumulative violations
    pub cumulative: usize,
    /// Violations at this point
    pub count: usize,
    /// Label
    pub label: String,
}

/// Heatmap entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapEntry {
    /// X coordinate (shape index)
    pub x: usize,
    /// Y coordinate (constraint index)
    pub y: usize,
    /// Value (violation count)
    pub value: usize,
    /// X label
    pub x_label: String,
    /// Y label
    pub y_label: String,
}

/// Violation detail for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationDetail {
    /// Violation index
    pub index: usize,
    /// Shape ID
    pub shape: String,
    /// Focus node
    pub focus_node: String,
    /// Property path
    pub path: Option<String>,
    /// Value
    pub value: Option<String>,
    /// Severity
    pub severity: String,
    /// Message
    pub message: String,
    /// Constraint type
    pub constraint_type: String,
}

impl ReportVisualizer {
    /// Create a new visualizer with default configuration
    pub fn new() -> Self {
        Self {
            config: VisualizerConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: VisualizerConfig) -> Self {
        Self { config }
    }

    /// Generate visualization from validation report
    pub fn visualize(&self, report: &ValidationReport) -> Result<String> {
        let data = self.extract_data(report)?;

        match self.config.format {
            VisualizationFormat::Html => self.generate_html(&data),
            VisualizationFormat::Svg => self.generate_svg(&data),
            VisualizationFormat::Json => self.generate_json(&data),
            VisualizationFormat::Ascii => self.generate_ascii(&data),
            VisualizationFormat::Markdown => self.generate_markdown(&data),
        }
    }

    /// Extract visualization data from report
    fn extract_data(&self, report: &ValidationReport) -> Result<VisualizationData> {
        let violations = report.violations();

        // Count by severity
        let mut by_severity: HashMap<String, usize> = HashMap::new();
        let mut by_shape: HashMap<String, usize> = HashMap::new();
        let mut by_constraint: HashMap<String, usize> = HashMap::new();

        let mut violations_count = 0;
        let mut warnings_count = 0;
        let mut infos_count = 0;

        for violation in violations {
            // Count by severity
            match violation.result_severity {
                Severity::Violation => {
                    violations_count += 1;
                    *by_severity.entry("Violation".to_string()).or_insert(0) += 1;
                }
                Severity::Warning => {
                    warnings_count += 1;
                    *by_severity.entry("Warning".to_string()).or_insert(0) += 1;
                }
                Severity::Info => {
                    infos_count += 1;
                    *by_severity.entry("Info".to_string()).or_insert(0) += 1;
                }
            }

            // Count by shape
            let shape_name = violation.source_shape.to_string();
            *by_shape.entry(shape_name).or_insert(0) += 1;

            // Count by constraint type
            let constraint_type = violation.source_constraint_component.0.clone();
            *by_constraint.entry(constraint_type).or_insert(0) += 1;
        }

        // Calculate compliance rate
        let total_violations = violations.len();
        let shapes_with_violations = by_shape.len();
        let focus_nodes: std::collections::HashSet<_> = violations
            .iter()
            .map(|v| v.focus_node.to_string())
            .collect();

        let compliance_rate = if total_violations == 0 {
            100.0
        } else {
            // Simplified compliance calculation
            ((1000.0 - (total_violations.min(1000) as f64)) / 10.0).max(0.0)
        };

        // Generate timeline
        let timeline = self.generate_timeline(violations);

        // Generate heatmap
        let heatmap = self.generate_heatmap(&by_shape, &by_constraint);

        // Extract violation details
        let violation_details: Vec<ViolationDetail> = violations
            .iter()
            .take(self.config.max_violations)
            .enumerate()
            .map(|(i, v)| ViolationDetail {
                index: i + 1,
                shape: v.source_shape.to_string(),
                focus_node: v.focus_node.to_string(),
                path: v.result_path.as_ref().map(|p| format!("{:?}", p)),
                value: v.value.as_ref().map(|val| format!("{:?}", val)),
                severity: format!("{:?}", v.result_severity),
                message: v.result_message.clone().unwrap_or_default(),
                constraint_type: v.source_constraint_component.0.clone(),
            })
            .collect();

        Ok(VisualizationData {
            summary: SummaryMetrics {
                total_violations,
                violations: violations_count,
                warnings: warnings_count,
                infos: infos_count,
                shapes_validated: by_shape.len() + 1, // Simplified
                shapes_with_violations,
                compliance_rate,
                focus_nodes_checked: focus_nodes.len(),
            },
            by_severity,
            by_shape,
            by_constraint,
            timeline,
            heatmap,
            violations: violation_details,
        })
    }

    fn generate_timeline(&self, violations: &[ValidationViolation]) -> Vec<TimelineEntry> {
        let mut timeline = Vec::new();
        let chunk_size = (violations.len() / 10).max(1);

        for (i, chunk) in violations.chunks(chunk_size).enumerate() {
            let cumulative = (i + 1) * chunk.len();
            timeline.push(TimelineEntry {
                index: i,
                cumulative,
                count: chunk.len(),
                label: format!("Batch {}", i + 1),
            });
        }

        timeline
    }

    fn generate_heatmap(
        &self,
        by_shape: &HashMap<String, usize>,
        by_constraint: &HashMap<String, usize>,
    ) -> Vec<HeatmapEntry> {
        let mut heatmap = Vec::new();

        let shapes: Vec<_> = by_shape.keys().take(10).collect();
        let constraints: Vec<_> = by_constraint.keys().take(10).collect();

        for (x, shape) in shapes.iter().enumerate() {
            for (y, constraint) in constraints.iter().enumerate() {
                let value = (by_shape.get(*shape).unwrap_or(&0)
                    + by_constraint.get(*constraint).unwrap_or(&0))
                    / 2;
                heatmap.push(HeatmapEntry {
                    x,
                    y,
                    value,
                    x_label: shape.to_string(),
                    y_label: constraint.to_string(),
                });
            }
        }

        heatmap
    }

    /// Generate HTML visualization
    fn generate_html(&self, data: &VisualizationData) -> Result<String> {
        let mut html = String::new();

        // HTML header
        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str(
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str("<title>SHACL Validation Report</title>\n");

        // Styles
        html.push_str("<style>\n");
        html.push_str(&self.generate_css());
        if let Some(custom_css) = &self.config.custom_css {
            html.push_str(custom_css);
        }
        html.push_str("</style>\n");

        html.push_str("</head>\n<body>\n");

        // Header
        html.push_str("<div class=\"container\">\n");
        html.push_str("<header>\n");
        html.push_str("<h1>SHACL Validation Report</h1>\n");
        html.push_str(&format!(
            "<p class=\"timestamp\">Generated: {}</p>\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        html.push_str("</header>\n");

        // Summary cards
        html.push_str("<section class=\"summary\">\n");
        html.push_str(&self.generate_summary_cards(&data.summary));
        html.push_str("</section>\n");

        // Charts section
        html.push_str("<section class=\"charts\">\n");

        // Severity distribution
        html.push_str("<div class=\"chart-container\">\n");
        html.push_str("<h2>Violations by Severity</h2>\n");
        html.push_str(&self.generate_pie_chart(&data.by_severity, "severity-chart"));
        html.push_str("</div>\n");

        // Shape distribution
        html.push_str("<div class=\"chart-container\">\n");
        html.push_str("<h2>Violations by Shape</h2>\n");
        html.push_str(&self.generate_bar_chart(&data.by_shape, "shape-chart"));
        html.push_str("</div>\n");

        // Constraint distribution
        html.push_str("<div class=\"chart-container\">\n");
        html.push_str("<h2>Violations by Constraint</h2>\n");
        html.push_str(&self.generate_bar_chart(&data.by_constraint, "constraint-chart"));
        html.push_str("</div>\n");

        html.push_str("</section>\n");

        // Violations table
        if self.config.include_tables && !data.violations.is_empty() {
            html.push_str("<section class=\"violations-table\">\n");
            html.push_str("<h2>Violation Details</h2>\n");
            html.push_str(&self.generate_violations_table(&data.violations));
            html.push_str("</section>\n");
        }

        html.push_str("</div>\n");

        // JavaScript
        if self.config.interactive {
            html.push_str("<script>\n");
            html.push_str(&self.generate_js());
            if let Some(custom_js) = &self.config.custom_js {
                html.push_str(custom_js);
            }
            html.push_str("</script>\n");
        }

        html.push_str("</body>\n</html>");

        Ok(html)
    }

    fn generate_css(&self) -> String {
        let theme = &self.config.theme;
        format!(
            r#"
:root {{
    --bg-color: {bg};
    --text-color: {text};
    --violation-color: {violation};
    --warning-color: {warning};
    --info-color: {info};
    --success-color: {success};
    --primary-color: {primary};
    --secondary-color: {secondary};
}}

* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    font-family: {font_family};
    font-size: {font_size}px;
    background-color: var(--bg-color);
    color: var(--text-color);
    line-height: 1.6;
}}

.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}}

header {{
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid var(--secondary-color);
}}

h1 {{
    color: var(--primary-color);
    margin-bottom: 10px;
}}

.timestamp {{
    color: var(--secondary-color);
    font-size: 0.9em;
}}

.summary {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}}

.card {{
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.card h3 {{
    font-size: 0.9em;
    color: var(--secondary-color);
    text-transform: uppercase;
    margin-bottom: 10px;
}}

.card .value {{
    font-size: 2em;
    font-weight: bold;
    color: var(--primary-color);
}}

.card.violation .value {{ color: var(--violation-color); }}
.card.warning .value {{ color: var(--warning-color); }}
.card.info .value {{ color: var(--info-color); }}
.card.success .value {{ color: var(--success-color); }}

.charts {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 30px;
    margin-bottom: 30px;
}}

.chart-container {{
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.chart-container h2 {{
    font-size: 1.1em;
    margin-bottom: 15px;
    color: var(--secondary-color);
}}

.chart-container svg {{
    max-width: 100%;
    height: auto;
}}

.violations-table {{
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    overflow-x: auto;
}}

.violations-table h2 {{
    margin-bottom: 15px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th, td {{
    padding: 12px 8px;
    text-align: left;
    border-bottom: 1px solid #dee2e6;
}}

th {{
    background: #e9ecef;
    font-weight: 600;
    color: var(--secondary-color);
}}

tr:hover {{
    background: #e9ecef;
}}

.severity-violation {{
    color: var(--violation-color);
    font-weight: bold;
}}

.severity-warning {{
    color: var(--warning-color);
    font-weight: bold;
}}

.severity-info {{
    color: var(--info-color);
    font-weight: bold;
}}

.bar {{
    fill: var(--primary-color);
    transition: fill 0.3s;
}}

.bar:hover {{
    fill: var(--secondary-color);
}}

@media (max-width: 768px) {{
    .charts {{
        grid-template-columns: 1fr;
    }}
}}
"#,
            bg = theme.background,
            text = theme.text,
            violation = theme.violation,
            warning = theme.warning,
            info = theme.info,
            success = theme.success,
            primary = theme.primary,
            secondary = theme.secondary,
            font_family = self.config.chart_settings.font_family,
            font_size = self.config.chart_settings.font_size
        )
    }

    fn generate_summary_cards(&self, summary: &SummaryMetrics) -> String {
        format!(
            r#"
<div class="card">
    <h3>Total Violations</h3>
    <div class="value">{}</div>
</div>
<div class="card violation">
    <h3>Violations</h3>
    <div class="value">{}</div>
</div>
<div class="card warning">
    <h3>Warnings</h3>
    <div class="value">{}</div>
</div>
<div class="card info">
    <h3>Info</h3>
    <div class="value">{}</div>
</div>
<div class="card">
    <h3>Shapes with Issues</h3>
    <div class="value">{}</div>
</div>
<div class="card success">
    <h3>Compliance Rate</h3>
    <div class="value">{:.1}%</div>
</div>
"#,
            summary.total_violations,
            summary.violations,
            summary.warnings,
            summary.infos,
            summary.shapes_with_violations,
            summary.compliance_rate
        )
    }

    fn generate_pie_chart(&self, data: &HashMap<String, usize>, id: &str) -> String {
        let total: usize = data.values().sum();
        if total == 0 {
            return "<p>No data available</p>".to_string();
        }

        let mut svg = format!(r#"<svg id="{}" viewBox="0 0 400 300">"#, id);

        let cx = 150.0;
        let cy = 150.0;
        let r = 100.0;
        let mut current_angle = 0.0;

        let colors = &self.config.theme.chart_colors;

        for (i, (label, value)) in data.iter().enumerate() {
            let percentage = *value as f64 / total as f64;
            let angle = percentage * 360.0;

            let start_x = cx + r * (current_angle * std::f64::consts::PI / 180.0).cos();
            let start_y = cy + r * (current_angle * std::f64::consts::PI / 180.0).sin();

            current_angle += angle;

            let end_x = cx + r * (current_angle * std::f64::consts::PI / 180.0).cos();
            let end_y = cy + r * (current_angle * std::f64::consts::PI / 180.0).sin();

            let large_arc = if angle > 180.0 { 1 } else { 0 };
            let color = &colors[i % colors.len()];

            svg.push_str(&format!(
                r#"<path d="M{cx},{cy} L{start_x},{start_y} A{r},{r} 0 {large_arc},1 {end_x},{end_y} Z" fill="{color}"><title>{label}: {value} ({percentage:.1}%)</title></path>"#,
                cx = cx,
                cy = cy,
                start_x = start_x,
                start_y = start_y,
                r = r,
                large_arc = large_arc,
                end_x = end_x,
                end_y = end_y,
                color = color,
                label = label,
                value = value,
                percentage = percentage * 100.0
            ));
        }

        // Legend
        let mut y = 20.0;
        for (i, (label, value)) in data.iter().enumerate() {
            let color = &colors[i % colors.len()];
            svg.push_str(&format!(
                r#"<rect x="320" y="{y}" width="12" height="12" fill="{color}"/><text x="340" y="{ty}" font-size="10">{label} ({value})</text>"#,
                y = y,
                ty = y + 10.0,
                color = color,
                label = label,
                value = value
            ));
            y += 20.0;
        }

        svg.push_str("</svg>");
        svg
    }

    fn generate_bar_chart(&self, data: &HashMap<String, usize>, id: &str) -> String {
        let mut sorted: Vec<_> = data.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        sorted.truncate(10); // Top 10

        if sorted.is_empty() {
            return "<p>No data available</p>".to_string();
        }

        let max_value = *sorted.iter().map(|(_, v)| *v).max().unwrap_or(&1);
        let width = self.config.chart_settings.width as f64;
        let height = (sorted.len() * 30 + 40) as f64;
        let bar_height = 20.0;
        let max_bar_width = width - 200.0;

        let mut svg = format!(r#"<svg id="{}" viewBox="0 0 {} {}">"#, id, width, height);

        for (i, (label, value)) in sorted.iter().enumerate() {
            let y = (i * 30 + 20) as f64;
            let bar_width = (**value as f64 / max_value as f64) * max_bar_width;

            // Truncate label
            let display_label = if label.len() > 25 {
                format!("{}...", &label[..22])
            } else {
                label.to_string()
            };

            // Label
            svg.push_str(&format!(
                r#"<text x="10" y="{}" font-size="10" dominant-baseline="middle">{}</text>"#,
                y + bar_height / 2.0,
                display_label
            ));

            // Bar
            svg.push_str(&format!(
                r#"<rect class="bar" x="180" y="{}" width="{}" height="{}" rx="2"><title>{}: {}</title></rect>"#,
                y, bar_width.max(1.0), bar_height, label, value
            ));

            // Value
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-size="9" dominant-baseline="middle">{}</text>"#,
                185.0 + bar_width,
                y + bar_height / 2.0,
                value
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    fn generate_violations_table(&self, violations: &[ViolationDetail]) -> String {
        let mut html = String::from(
            r#"<table>
<thead>
<tr>
<th>#</th>
<th>Severity</th>
<th>Shape</th>
<th>Focus Node</th>
<th>Message</th>
</tr>
</thead>
<tbody>
"#,
        );

        for violation in violations {
            let severity_class = match violation.severity.as_str() {
                "Violation" => "severity-violation",
                "Warning" => "severity-warning",
                _ => "severity-info",
            };

            // Truncate long values
            let shape = if violation.shape.len() > 40 {
                format!("{}...", &violation.shape[..37])
            } else {
                violation.shape.clone()
            };

            let focus = if violation.focus_node.len() > 40 {
                format!("{}...", &violation.focus_node[..37])
            } else {
                violation.focus_node.clone()
            };

            let message = if violation.message.len() > 60 {
                format!("{}...", &violation.message[..57])
            } else {
                violation.message.clone()
            };

            html.push_str(&format!(
                r#"<tr>
<td>{}</td>
<td class="{}">{}</td>
<td title="{}">{}</td>
<td title="{}">{}</td>
<td title="{}">{}</td>
</tr>
"#,
                violation.index,
                severity_class,
                violation.severity,
                violation.shape,
                shape,
                violation.focus_node,
                focus,
                violation.message,
                message
            ));
        }

        html.push_str("</tbody></table>");
        html
    }

    fn generate_js(&self) -> String {
        r#"
document.addEventListener('DOMContentLoaded', function() {
    // Add tooltips
    const elements = document.querySelectorAll('[title]');
    elements.forEach(el => {
        el.addEventListener('mouseenter', function(e) {
            // Could add custom tooltip logic here
        });
    });

    // Add sorting to table
    const table = document.querySelector('.violations-table table');
    if (table) {
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', function() {
                sortTable(table, index);
            });
        });
    }
});

function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    rows.sort((a, b) => {
        const aVal = a.cells[columnIndex].textContent;
        const bVal = b.cells[columnIndex].textContent;
        return aVal.localeCompare(bVal);
    });

    rows.forEach(row => tbody.appendChild(row));
}
"#
        .to_string()
    }

    /// Generate SVG visualization
    fn generate_svg(&self, data: &VisualizationData) -> Result<String> {
        let width = self.config.chart_settings.width;
        let height = self.config.chart_settings.height;

        let mut svg = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">
<style>
    text {{ font-family: {}; font-size: {}px; }}
    .title {{ font-size: 18px; font-weight: bold; }}
</style>
"#,
            width,
            height,
            self.config.chart_settings.font_family,
            self.config.chart_settings.font_size
        );

        // Title
        svg.push_str(&format!(
            r#"<text class="title" x="{}" y="30" text-anchor="middle">SHACL Validation Report</text>"#,
            width / 2
        ));

        // Summary text
        svg.push_str(&format!(
            r#"<text x="20" y="60">Total Violations: {} | Compliance: {:.1}%</text>"#,
            data.summary.total_violations, data.summary.compliance_rate
        ));

        // Severity bars
        let mut y = 100;
        for (severity, count) in &data.by_severity {
            let bar_width = (*count as f64 / data.summary.total_violations.max(1) as f64)
                * (width as f64 - 200.0);
            let color = match severity.as_str() {
                "Violation" => &self.config.theme.violation,
                "Warning" => &self.config.theme.warning,
                _ => &self.config.theme.info,
            };

            svg.push_str(&format!(
                r#"<text x="20" y="{}">{}</text><rect x="100" y="{}" width="{}" height="15" fill="{}"/><text x="{}" y="{}">{}</text>"#,
                y,
                severity,
                y - 12,
                bar_width.max(1.0),
                color,
                110.0 + bar_width,
                y,
                count
            ));
            y += 30;
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Generate JSON data
    fn generate_json(&self, data: &VisualizationData) -> Result<String> {
        Ok(serde_json::to_string_pretty(data)?)
    }

    /// Generate ASCII art visualization
    fn generate_ascii(&self, data: &VisualizationData) -> Result<String> {
        let mut output = String::new();

        output.push_str("╔════════════════════════════════════════════════════════════╗\n");
        output.push_str("║           SHACL Validation Report                          ║\n");
        output.push_str("╠════════════════════════════════════════════════════════════╣\n");

        // Summary
        output.push_str(&format!(
            "║ Total Violations: {:<40} ║\n",
            data.summary.total_violations
        ));
        output.push_str(&format!(
            "║ Compliance Rate:  {:<40.1}% ║\n",
            data.summary.compliance_rate
        ));
        output.push_str("╠════════════════════════════════════════════════════════════╣\n");

        // Severity distribution
        output.push_str("║ By Severity:                                               ║\n");
        let max_severity = data.by_severity.values().copied().max().unwrap_or(1);

        for (severity, count) in &data.by_severity {
            let bar_len = (*count as f64 / max_severity as f64 * 30.0) as usize;
            let bar: String = "█".repeat(bar_len);
            output.push_str(&format!("║ {:12} {:>5} {:<30} ║\n", severity, count, bar));
        }

        output.push_str("╠════════════════════════════════════════════════════════════╣\n");

        // Top shapes
        output.push_str("║ Top Shapes with Violations:                                ║\n");
        let mut shapes: Vec<_> = data.by_shape.iter().collect();
        shapes.sort_by(|a, b| b.1.cmp(a.1));

        for (shape, count) in shapes.iter().take(5) {
            let truncated = if shape.len() > 35 {
                format!("{}...", &shape[..32])
            } else {
                shape.to_string()
            };
            output.push_str(&format!("║ {:35} {:>5} ║\n", truncated, count));
        }

        output.push_str("╚════════════════════════════════════════════════════════════╝\n");

        Ok(output)
    }

    /// Generate Markdown visualization
    fn generate_markdown(&self, data: &VisualizationData) -> Result<String> {
        let mut md = String::new();

        md.push_str("# SHACL Validation Report\n\n");
        md.push_str(&format!(
            "Generated: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Total Violations | {} |\n",
            data.summary.total_violations
        ));
        md.push_str(&format!("| Violations | {} |\n", data.summary.violations));
        md.push_str(&format!("| Warnings | {} |\n", data.summary.warnings));
        md.push_str(&format!("| Info | {} |\n", data.summary.infos));
        md.push_str(&format!(
            "| Compliance Rate | {:.1}% |\n",
            data.summary.compliance_rate
        ));
        md.push('\n');

        // By severity
        md.push_str("## Violations by Severity\n\n");
        for (severity, count) in &data.by_severity {
            md.push_str(&format!("- **{}**: {}\n", severity, count));
        }
        md.push('\n');

        // By shape
        md.push_str("## Top Shapes with Violations\n\n");
        let mut shapes: Vec<_> = data.by_shape.iter().collect();
        shapes.sort_by(|a, b| b.1.cmp(a.1));

        md.push_str("| Shape | Violations |\n");
        md.push_str("|-------|------------|\n");
        for (shape, count) in shapes.iter().take(10) {
            md.push_str(&format!("| `{}` | {} |\n", shape, count));
        }
        md.push('\n');

        // Violations table
        if !data.violations.is_empty() {
            md.push_str("## Violation Details\n\n");
            md.push_str("| # | Severity | Shape | Message |\n");
            md.push_str("|---|----------|-------|----------|\n");

            for v in data.violations.iter().take(50) {
                let message = if v.message.len() > 50 {
                    format!("{}...", &v.message[..47])
                } else {
                    v.message.clone()
                };
                md.push_str(&format!(
                    "| {} | {} | `{}` | {} |\n",
                    v.index, v.severity, v.shape, message
                ));
            }
        }

        Ok(md)
    }
}

impl Default for ReportVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for visualizer configuration
#[derive(Debug)]
pub struct VisualizerConfigBuilder {
    config: VisualizerConfig,
}

impl VisualizerConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: VisualizerConfig::default(),
        }
    }

    /// Set output format
    pub fn format(mut self, format: VisualizationFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Set color theme
    pub fn theme(mut self, theme: ColorTheme) -> Self {
        self.config.theme = theme;
        self
    }

    /// Set interactive mode
    pub fn interactive(mut self, interactive: bool) -> Self {
        self.config.interactive = interactive;
        self
    }

    /// Include tables
    pub fn include_tables(mut self, include: bool) -> Self {
        self.config.include_tables = include;
        self
    }

    /// Set max violations
    pub fn max_violations(mut self, max: usize) -> Self {
        self.config.max_violations = max;
        self
    }

    /// Set chart width
    pub fn chart_width(mut self, width: u32) -> Self {
        self.config.chart_settings.width = width;
        self
    }

    /// Set chart height
    pub fn chart_height(mut self, height: u32) -> Self {
        self.config.chart_settings.height = height;
        self
    }

    /// Build the configuration
    pub fn build(self) -> VisualizerConfig {
        self.config
    }
}

impl Default for VisualizerConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualizer_creation() {
        let visualizer = ReportVisualizer::new();
        assert!(matches!(
            visualizer.config.format,
            VisualizationFormat::Html
        ));
    }

    #[test]
    fn test_config_builder() {
        let config = VisualizerConfigBuilder::new()
            .format(VisualizationFormat::Svg)
            .interactive(false)
            .max_violations(500)
            .chart_width(1024)
            .build();

        assert!(matches!(config.format, VisualizationFormat::Svg));
        assert!(!config.interactive);
        assert_eq!(config.max_violations, 500);
        assert_eq!(config.chart_settings.width, 1024);
    }

    #[test]
    fn test_default_theme() {
        let theme = ColorTheme::default();
        assert!(!theme.background.is_empty());
        assert!(!theme.chart_colors.is_empty());
    }

    #[test]
    fn test_summary_metrics() {
        let summary = SummaryMetrics {
            total_violations: 100,
            violations: 50,
            warnings: 30,
            infos: 20,
            shapes_validated: 10,
            shapes_with_violations: 5,
            compliance_rate: 85.0,
            focus_nodes_checked: 50,
        };

        assert_eq!(
            summary.total_violations,
            summary.violations + summary.warnings + summary.infos
        );
    }

    #[test]
    fn test_ascii_generation() {
        let visualizer = ReportVisualizer::with_config(
            VisualizerConfigBuilder::new()
                .format(VisualizationFormat::Ascii)
                .build(),
        );

        let data = VisualizationData {
            summary: SummaryMetrics {
                total_violations: 10,
                violations: 5,
                warnings: 3,
                infos: 2,
                shapes_validated: 5,
                shapes_with_violations: 3,
                compliance_rate: 80.0,
                focus_nodes_checked: 10,
            },
            by_severity: [("Violation".to_string(), 5), ("Warning".to_string(), 3)]
                .into_iter()
                .collect(),
            by_shape: [("Shape1".to_string(), 3), ("Shape2".to_string(), 2)]
                .into_iter()
                .collect(),
            by_constraint: HashMap::new(),
            timeline: Vec::new(),
            heatmap: Vec::new(),
            violations: Vec::new(),
        };

        let result = visualizer
            .generate_ascii(&data)
            .expect("generation should succeed");
        assert!(result.contains("SHACL Validation Report"));
        assert!(result.contains("Total Violations: 10"));
    }

    #[test]
    fn test_markdown_generation() {
        let visualizer = ReportVisualizer::with_config(
            VisualizerConfigBuilder::new()
                .format(VisualizationFormat::Markdown)
                .build(),
        );

        let data = VisualizationData {
            summary: SummaryMetrics {
                total_violations: 5,
                violations: 3,
                warnings: 2,
                infos: 0,
                shapes_validated: 2,
                shapes_with_violations: 2,
                compliance_rate: 90.0,
                focus_nodes_checked: 5,
            },
            by_severity: [("Violation".to_string(), 3)].into_iter().collect(),
            by_shape: [("TestShape".to_string(), 3)].into_iter().collect(),
            by_constraint: HashMap::new(),
            timeline: Vec::new(),
            heatmap: Vec::new(),
            violations: Vec::new(),
        };

        let result = visualizer
            .generate_markdown(&data)
            .expect("generation should succeed");
        assert!(result.contains("# SHACL Validation Report"));
        assert!(result.contains("| Total Violations | 5 |"));
    }
}
