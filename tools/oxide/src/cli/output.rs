//! Output formatting with color support and structured display
//!
//! Provides consistent, colored output for different message types.

use colored::*;
use prettytable::{format, Cell, Row, Table};
use serde::Serialize;

/// Output formatter with color and style management
pub struct OutputFormatter {
    no_color: bool,
    _color_scheme: ColorScheme,
}

/// Color scheme for different output types
#[derive(Clone)]
pub struct ColorScheme {
    pub info: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub verbose: Color,
    pub highlight: Color,
    pub muted: Color,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            info: Color::Blue,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            verbose: Color::Cyan,
            highlight: Color::Magenta,
            muted: Color::BrightBlack,
        }
    }
}

impl OutputFormatter {
    /// Create a new output formatter
    pub fn new(format: &str) -> Self {
        // Determine if color should be disabled
        let no_color = format == "json" || format == "csv" || std::env::var("NO_COLOR").is_ok();

        if no_color {
            colored::control::set_override(false);
        }

        Self {
            no_color,
            _color_scheme: ColorScheme::default(),
        }
    }

    /// Create a new output formatter with explicit color setting
    pub fn new_with_color(no_color: bool) -> Self {
        if no_color {
            colored::control::set_override(false);
        }

        Self {
            no_color,
            _color_scheme: ColorScheme::default(),
        }
    }

    /// Print an info message
    pub fn info(&self, message: &str) {
        if self.no_color {
            println!("ℹ {}", message);
        } else {
            println!("{} {}", "ℹ".blue(), message);
        }
    }

    /// Print a success message
    pub fn success(&self, message: &str) {
        if self.no_color {
            println!("✓ {}", message);
        } else {
            println!("{} {}", "✓".green(), message);
        }
    }

    /// Print a warning message
    pub fn warn(&self, message: &str) {
        if self.no_color {
            eprintln!("⚠ {}", message);
        } else {
            eprintln!("{} {}", "⚠".yellow(), message.yellow());
        }
    }

    /// Print an error message
    pub fn error(&self, message: &str) {
        if self.no_color {
            eprintln!("✗ {}", message);
        } else {
            eprintln!("{} {}", "✗".red(), message.red());
        }
    }

    /// Print verbose/debug output
    pub fn verbose(&self, message: &str) {
        if self.no_color {
            println!("  {}", message);
        } else {
            println!("  {}", message.cyan());
        }
    }

    /// Print a section header
    pub fn section(&self, title: &str) {
        if self.no_color {
            println!("\n=== {} ===", title);
        } else {
            println!("\n{}", format!("=== {} ===", title).bright_white().bold());
        }
    }

    /// Print a highlighted value
    pub fn highlight(&self, label: &str, value: &str) {
        if self.no_color {
            println!("{}: {}", label, value);
        } else {
            println!("{}: {}", label.bright_white(), value.magenta());
        }
    }

    /// Print key-value pairs
    pub fn key_value(&self, key: &str, value: &str) {
        if self.no_color {
            println!("{}: {}", key, value);
        } else {
            println!("{}: {}", key.bright_white(), value);
        }
    }

    /// Print a list item
    pub fn list_item(&self, item: &str) {
        if self.no_color {
            println!("  • {}", item);
        } else {
            println!("  {} {}", "•".blue(), item);
        }
    }

    /// Print muted/secondary text
    pub fn muted(&self, text: &str) {
        if self.no_color {
            println!("{}", text);
        } else {
            println!("{}", text.bright_black());
        }
    }

    /// Create a table for structured output
    pub fn create_table(&self) -> Table {
        let mut table = Table::new();
        if self.no_color {
            table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
        } else {
            table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
        }
        table
    }

    /// Format a table row with colors
    pub fn table_row(&self, cells: Vec<&str>) -> Row {
        Row::new(cells.into_iter().map(Cell::new).collect())
    }

    /// Print JSON output (pretty-printed)
    pub fn json<T: Serialize>(&self, data: &T) -> Result<(), serde_json::Error> {
        let json = serde_json::to_string_pretty(data)?;
        println!("{}", json);
        Ok(())
    }

    /// Print a progress update
    pub fn progress(&self, current: usize, total: Option<usize>, message: &str) {
        match total {
            Some(total) => {
                let percent = (current as f64 / total as f64 * 100.0) as u8;
                if self.no_color {
                    print!("\r[{}/{}] {}% - {}", current, total, percent, message);
                } else {
                    print!(
                        "\r{} {}% - {}",
                        format!("[{}/{}]", current, total).blue(),
                        format!("{}%", percent).green(),
                        message
                    );
                }
            }
            None => {
                if self.no_color {
                    print!("\r[{}] {}", current, message);
                } else {
                    print!("\r{} {}", format!("[{}]", current).blue(), message);
                }
            }
        }
        use std::io::{self, Write};
        io::stdout().flush().unwrap_or(());
    }

    /// Clear the current line
    pub fn clear_line(&self) {
        print!("\r{}\r", " ".repeat(80));
        use std::io::{self, Write};
        io::stdout().flush().unwrap_or(());
    }

    /// Print performance report
    pub fn print_performance_report(
        &self,
        report: &crate::tools::performance::PerformanceReport,
    ) -> Result<(), crate::cli::error::CliError> {
        self.section("Performance Report");

        let timestamp = report
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.key_value("Timestamp", &format!("{}", timestamp));
        self.key_value(
            "CPU Usage",
            &format!("{:.1}%", report.current_metrics.cpu_usage),
        );
        self.key_value(
            "Memory Usage",
            &format!(
                "{:.2} GB",
                report.current_metrics.memory_usage as f64 / 1_000_000_000.0
            ),
        );
        self.key_value(
            "Memory Total",
            &format!(
                "{:.2} GB",
                report.current_metrics.memory_total as f64 / 1_000_000_000.0
            ),
        );
        self.key_value(
            "Active Sessions",
            &format!("{}", report.active_profiling_sessions),
        );

        if !report.performance_counters.is_empty() {
            self.section("Performance Counters");
            for (name, value) in &report.performance_counters {
                self.key_value(name, &format!("{}", value));
            }
        }

        Ok(())
    }

    /// Print profiling result
    pub fn print_profiling_result(
        &self,
        result: &crate::tools::performance::ProfilingResult,
    ) -> Result<(), crate::cli::error::CliError> {
        self.section(&format!("Profiling Result: {}", result.operation_name));

        self.key_value("Session ID", &result.session_id);
        self.key_value(
            "Duration",
            &format!("{:.3}s", result.total_duration.as_secs_f64()),
        );
        self.key_value(
            "Efficiency Score",
            &format!("{:.1}/100", result.performance_summary.efficiency_score),
        );
        self.key_value(
            "Memory Delta",
            &format!(
                "{:.2} MB",
                result.performance_summary.memory_delta_bytes as f64 / 1_000_000.0
            ),
        );
        self.key_value(
            "Average CPU",
            &format!("{:.1}%", result.performance_summary.average_cpu_usage),
        );
        self.key_value("Checkpoints", &format!("{}", result.checkpoints.len()));

        if !result.checkpoints.is_empty() {
            self.section("Checkpoints");
            let mut table = self.create_table();
            table.set_titles(self.table_row(vec!["Name", "Time (s)", "Memory Delta (MB)"]));

            for checkpoint in &result.checkpoints {
                table.add_row(self.table_row(vec![
                    &checkpoint.name,
                    &format!("{:.3}", checkpoint.duration_from_start.as_secs_f64()),
                    &format!("{:.2}", checkpoint.memory_delta as f64 / 1_000_000.0),
                ]));
            }

            println!("{}", table);
        }

        Ok(())
    }

    /// Print benchmark comparison
    pub fn print_benchmark_comparison(
        &self,
        comparison: &crate::tools::performance::BenchmarkComparison,
    ) -> Result<(), crate::cli::error::CliError> {
        self.section("Benchmark Comparison");

        self.key_value("Baseline", &comparison.baseline_name);
        self.key_value("Current", &comparison.current_name);

        if comparison.performance_ratio < 0.95 {
            self.success(&comparison.improvement_summary);
        } else if comparison.performance_ratio > 1.05 {
            self.warn(&comparison.improvement_summary);
        } else {
            self.info(&comparison.improvement_summary);
        }

        self.key_value("Time Ratio", &format!("{:.3}x", comparison.time_ratio));
        self.key_value("Memory Ratio", &format!("{:.3}x", comparison.memory_ratio));
        self.key_value(
            "Performance Ratio",
            &format!("{:.3}x", comparison.performance_ratio),
        );

        if !comparison.detailed_metrics.is_empty() {
            self.section("Detailed Metrics");
            let mut table = self.create_table();
            table.set_titles(self.table_row(vec![
                "Metric",
                "Baseline",
                "Current",
                "Improvement %",
                "Significance",
            ]));

            for (metric_name, metric) in &comparison.detailed_metrics {
                let improvement_text = if metric.improvement_percentage > 0.0 {
                    format!("+{:.1}%", metric.improvement_percentage)
                } else {
                    format!("{:.1}%", metric.improvement_percentage)
                };

                table.add_row(self.table_row(vec![
                    metric_name,
                    &format!("{:.3}", metric.baseline_value),
                    &format!("{:.3}", metric.current_value),
                    &improvement_text,
                    &format!("{:?}", metric.significance),
                ]));
            }

            println!("{}", table);
        }

        Ok(())
    }

    /// Print system health
    pub fn print_system_health(
        &self,
        health: &crate::tools::performance::SystemHealth,
    ) -> Result<(), crate::cli::error::CliError> {
        self.section("System Health");

        let status_text = format!("{:?}", health.status);
        match health.status {
            crate::tools::performance::HealthStatus::Healthy => {
                self.success(&format!("Status: {}", status_text))
            }
            crate::tools::performance::HealthStatus::Moderate => {
                self.info(&format!("Status: {}", status_text))
            }
            crate::tools::performance::HealthStatus::Warning => {
                self.warn(&format!("Status: {}", status_text))
            }
            crate::tools::performance::HealthStatus::Critical => {
                self.error(&format!("Status: {}", status_text))
            }
        }

        self.key_value("CPU Usage", &format!("{:.1}%", health.cpu_usage_percentage));
        self.key_value(
            "Memory Usage",
            &format!("{:.1}%", health.memory_usage_percentage),
        );

        if health.disk_space_issues {
            self.warn("Disk space issues detected");
        }

        if health.network_issues {
            self.warn("Network issues detected");
        }

        if !health.recommendations.is_empty() {
            self.section("Health Recommendations");
            for recommendation in &health.recommendations {
                self.list_item(recommendation);
            }
        }

        Ok(())
    }

    /// Print detection result for format detection
    pub fn print_detection_result(
        &self,
        result: &crate::tools::format_detection::DetectionResult,
        path: &std::path::Path,
    ) -> Result<(), crate::cli::error::CliError> {
        self.section("Format Detection Result");

        self.key_value("File", &path.display().to_string());
        self.key_value("Detected Format", &format!("{:?}", result.format));
        self.key_value("Confidence", &format!("{:.1}%", result.confidence * 100.0));
        self.key_value(
            "Detection Method",
            &format!("{:?}", result.detection_method),
        );

        if let Some(encoding) = &result.encoding {
            self.key_value("Encoding", encoding);
        }

        if let Some(compression) = &result.compression {
            self.key_value("Compression", compression);
        }

        if !result.additional_info.is_empty() {
            self.section("Additional Information");
            for (key, value) in &result.additional_info {
                self.key_value(key, value);
            }
        }

        Ok(())
    }
}

/// Result formatter for different output formats
pub struct ResultFormatter;

impl ResultFormatter {
    /// Format SPARQL query results
    pub fn format_sparql_results(
        results: &SparqlResults,
        format: &str,
        formatter: &OutputFormatter,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match format {
            "table" => Ok(Self::format_as_table(results, formatter)),
            "csv" => Ok(Self::format_as_csv(results)),
            "tsv" => Ok(Self::format_as_tsv(results)),
            "json" => Ok(serde_json::to_string_pretty(results)?),
            "xml" => Ok(Self::format_as_xml(results)),
            _ => Err(format!("Unsupported output format: {}", format).into()),
        }
    }

    fn format_as_table(results: &SparqlResults, formatter: &OutputFormatter) -> String {
        let mut table = formatter.create_table();

        // Add headers
        if !results.vars.is_empty() {
            table.set_titles(Row::new(
                results.vars.iter().map(|v| Cell::new(v)).collect(),
            ));
        }

        // Add rows
        for binding in &results.bindings {
            let cells: Vec<Cell> = results
                .vars
                .iter()
                .map(|var| {
                    binding
                        .get(var)
                        .map(|val| Cell::new(val))
                        .unwrap_or_else(|| Cell::new(""))
                })
                .collect();
            table.add_row(Row::new(cells));
        }

        table.to_string()
    }

    fn format_as_csv(results: &SparqlResults) -> String {
        let mut output = String::new();

        // Headers
        output.push_str(&results.vars.join(","));
        output.push('\n');

        // Rows
        for binding in &results.bindings {
            let values: Vec<String> = results
                .vars
                .iter()
                .map(|var| {
                    binding
                        .get(var)
                        .map(|val| Self::escape_csv(val))
                        .unwrap_or_default()
                })
                .collect();
            output.push_str(&values.join(","));
            output.push('\n');
        }

        output
    }

    fn format_as_tsv(results: &SparqlResults) -> String {
        let mut output = String::new();

        // Headers
        output.push_str(&results.vars.join("\t"));
        output.push('\n');

        // Rows
        for binding in &results.bindings {
            let values: Vec<&str> = results
                .vars
                .iter()
                .map(|var| binding.get(var).map(|s| s.as_str()).unwrap_or(""))
                .collect();
            output.push_str(&values.join("\t"));
            output.push('\n');
        }

        output
    }

    fn format_as_xml(results: &SparqlResults) -> String {
        let mut xml = String::from("<?xml version=\"1.0\"?>\n");
        xml.push_str("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n");

        // Head
        xml.push_str("  <head>\n");
        for var in &results.vars {
            xml.push_str(&format!("    <variable name=\"{}\"/>\n", var));
        }
        xml.push_str("  </head>\n");

        // Results
        xml.push_str("  <results>\n");
        for binding in &results.bindings {
            xml.push_str("    <result>\n");
            for var in &results.vars {
                if let Some(value) = binding.get(var) {
                    xml.push_str(&format!("      <binding name=\"{}\">\n", var));
                    xml.push_str(&format!(
                        "        <literal>{}</literal>\n",
                        Self::escape_xml(value)
                    ));
                    xml.push_str("      </binding>\n");
                }
            }
            xml.push_str("    </result>\n");
        }
        xml.push_str("  </results>\n");
        xml.push_str("</sparql>\n");

        xml
    }

    fn escape_csv(value: &str) -> String {
        if value.contains(',') || value.contains('"') || value.contains('\n') {
            format!("\"{}\"", value.replace('"', "\"\""))
        } else {
            value.to_string()
        }
    }

    fn escape_xml(value: &str) -> String {
        value
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

/// Simple SPARQL results structure for formatting
#[derive(Debug, Serialize)]
pub struct SparqlResults {
    pub vars: Vec<String>,
    pub bindings: Vec<std::collections::HashMap<String, String>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_output_formatter() {
        let formatter = OutputFormatter::new_with_color(true); // no color for tests

        // These should not panic
        formatter.info("Information");
        formatter.success("Success");
        formatter.warn("Warning");
        formatter.error("Error");
        formatter.verbose("Debug info");
    }

    #[test]
    fn test_sparql_result_formatting() {
        let mut bindings = Vec::new();
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), "Alice".to_string());
        row1.insert("age".to_string(), "30".to_string());
        bindings.push(row1);

        let results = SparqlResults {
            vars: vec!["name".to_string(), "age".to_string()],
            bindings,
        };

        let csv = ResultFormatter::format_as_csv(&results);
        assert!(csv.contains("name,age"));
        assert!(csv.contains("Alice,30"));

        let tsv = ResultFormatter::format_as_tsv(&results);
        assert!(tsv.contains("name\tage"));
        assert!(tsv.contains("Alice\t30"));
    }

    #[test]
    fn test_csv_escaping() {
        let value = "Hello, \"World\"";
        let escaped = ResultFormatter::escape_csv(value);
        assert_eq!(escaped, "\"Hello, \"\"World\"\"\"");
    }
}
