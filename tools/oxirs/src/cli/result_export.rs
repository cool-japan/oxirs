//! Result Export Module for Interactive REPL
//!
//! Provides functionality to export SPARQL query results to various formats
//! (CSV, JSON, HTML) for further analysis or reporting.

use crate::cli::formatters::{QueryResults, RdfTerm};
use crate::cli::CliResult;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Csv,
    Json,
    Html,
    Xlsx,
}

impl ExportFormat {
    /// Get the file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Csv => "csv",
            ExportFormat::Json => "json",
            ExportFormat::Html => "html",
            ExportFormat::Xlsx => "xlsx",
        }
    }

    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ExportFormat::Csv => "text/csv",
            ExportFormat::Json => "application/json",
            ExportFormat::Html => "text/html",
            ExportFormat::Xlsx => {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
        }
    }

    /// Parse format from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "csv" => Some(ExportFormat::Csv),
            "json" => Some(ExportFormat::Json),
            "html" => Some(ExportFormat::Html),
            "xlsx" | "excel" => Some(ExportFormat::Xlsx),
            _ => None,
        }
    }

    /// Parse format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::parse)
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ExportFormat::Csv => "CSV",
                ExportFormat::Json => "JSON",
                ExportFormat::Html => "HTML",
                ExportFormat::Xlsx => "Excel (XLSX)",
            }
        )
    }
}

/// Configuration for result export
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Format to export to
    pub format: ExportFormat,
    /// Include column headers (CSV only)
    pub include_headers: bool,
    /// Pretty-print JSON output
    pub pretty_json: bool,
    /// Include CSS styling (HTML only)
    pub include_css: bool,
    /// Custom page title (HTML only)
    pub html_title: Option<String>,
    /// Excel worksheet name (XLSX only)
    pub xlsx_sheet_name: Option<String>,
    /// Auto-fit columns in Excel (XLSX only)
    pub xlsx_autofit: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_headers: true,
            pretty_json: true,
            include_css: true,
            html_title: None,
            xlsx_sheet_name: None,
            xlsx_autofit: true,
        }
    }
}

impl ExportConfig {
    /// Create configuration for CSV export
    pub fn csv() -> Self {
        Self {
            format: ExportFormat::Csv,
            ..Default::default()
        }
    }

    /// Create configuration for JSON export
    pub fn json() -> Self {
        Self {
            format: ExportFormat::Json,
            ..Default::default()
        }
    }

    /// Create configuration for HTML export
    pub fn html() -> Self {
        Self {
            format: ExportFormat::Html,
            ..Default::default()
        }
    }

    /// Set whether to include headers (CSV only)
    pub fn with_headers(mut self, include: bool) -> Self {
        self.include_headers = include;
        self
    }

    /// Set whether to pretty-print JSON
    pub fn with_pretty_json(mut self, pretty: bool) -> Self {
        self.pretty_json = pretty;
        self
    }

    /// Set HTML title
    pub fn with_html_title(mut self, title: String) -> Self {
        self.html_title = Some(title);
        self
    }

    /// Create configuration for Excel export
    pub fn xlsx() -> Self {
        Self {
            format: ExportFormat::Xlsx,
            ..Default::default()
        }
    }

    /// Set Excel worksheet name
    pub fn with_xlsx_sheet_name(mut self, name: String) -> Self {
        self.xlsx_sheet_name = Some(name);
        self
    }

    /// Set whether to auto-fit Excel columns
    pub fn with_xlsx_autofit(mut self, autofit: bool) -> Self {
        self.xlsx_autofit = autofit;
        self
    }
}

/// Result exporter
pub struct ResultExporter {
    config: ExportConfig,
}

impl ResultExporter {
    /// Create a new result exporter with default configuration
    pub fn new(format: ExportFormat) -> Self {
        Self {
            config: ExportConfig {
                format,
                ..Default::default()
            },
        }
    }

    /// Create a new result exporter with custom configuration
    pub fn with_config(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export results to a file
    pub fn export_to_file(&self, results: &QueryResults, path: &Path) -> CliResult<()> {
        // XLSX format requires special handling (can't write to a generic writer)
        if self.config.format == ExportFormat::Xlsx {
            #[cfg(feature = "excel-export")]
            {
                return self.export_xlsx(results, path);
            }
            #[cfg(not(feature = "excel-export"))]
            {
                return Err("XLSX export requires the 'excel-export' feature to be enabled".into());
            }
        }

        let mut file = File::create(path)?;
        self.export_to_writer(results, &mut file)
    }

    /// Export results to a writer
    pub fn export_to_writer<W: Write>(
        &self,
        results: &QueryResults,
        writer: &mut W,
    ) -> CliResult<()> {
        match self.config.format {
            ExportFormat::Csv => self.export_csv(results, writer),
            ExportFormat::Json => self.export_json(results, writer),
            ExportFormat::Html => self.export_html(results, writer),
            ExportFormat::Xlsx => Err("XLSX format requires export_to_file() method".into()),
        }
    }

    /// Export results to a string
    pub fn export_to_string(&self, results: &QueryResults) -> CliResult<String> {
        let mut buffer = Vec::new();
        self.export_to_writer(results, &mut buffer)?;
        String::from_utf8(buffer).map_err(|e| e.to_string().into())
    }

    /// Export results to CSV format
    fn export_csv<W: Write>(&self, results: &QueryResults, writer: &mut W) -> CliResult<()> {
        // Write headers
        if self.config.include_headers {
            for (i, var) in results.variables.iter().enumerate() {
                if i > 0 {
                    write!(writer, ",")?;
                }
                write!(writer, "\"{}\"", Self::escape_csv(var))?;
            }
            writeln!(writer)?;
        }

        // Write rows
        for binding in &results.bindings {
            for (i, _var) in results.variables.iter().enumerate() {
                if i > 0 {
                    write!(writer, ",")?;
                }

                if i < binding.values.len() {
                    if let Some(term) = &binding.values[i] {
                        let value = Self::format_term_for_csv(term);
                        write!(writer, "\"{}\"", Self::escape_csv(&value))?;
                    }
                }
            }
            writeln!(writer)?;
        }

        Ok(())
    }

    /// Export results to JSON format
    fn export_json<W: Write>(&self, results: &QueryResults, writer: &mut W) -> CliResult<()> {
        let json_results = self.convert_to_json(results);

        if self.config.pretty_json {
            serde_json::to_writer_pretty(writer, &json_results).map_err(|e| e.to_string())?;
        } else {
            serde_json::to_writer(writer, &json_results).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Export results to HTML format
    fn export_html<W: Write>(&self, results: &QueryResults, writer: &mut W) -> CliResult<()> {
        writeln!(writer, "<!DOCTYPE html>")?;
        writeln!(writer, "<html lang=\"en\">")?;
        writeln!(writer, "<head>")?;
        writeln!(writer, "    <meta charset=\"UTF-8\">")?;
        writeln!(
            writer,
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
        )?;

        let title = self
            .config
            .html_title
            .as_deref()
            .unwrap_or("SPARQL Query Results");
        writeln!(writer, "    <title>{}</title>", Self::escape_html(title))?;

        if self.config.include_css {
            writeln!(writer, "    <style>")?;
            writeln!(
                writer,
                "        body {{ font-family: Arial, sans-serif; margin: 20px; }}"
            )?;
            writeln!(
                writer,
                "        h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}"
            )?;
            writeln!(
                writer,
                "        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}"
            )?;
            writeln!(
                writer,
                "        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}"
            )?;
            writeln!(
                writer,
                "        th {{ background-color: #007acc; color: white; font-weight: bold; }}"
            )?;
            writeln!(
                writer,
                "        tr:nth-child(even) {{ background-color: #f2f2f2; }}"
            )?;
            writeln!(writer, "        tr:hover {{ background-color: #e8f4f8; }}")?;
            writeln!(
                writer,
                "        .summary {{ margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #007acc; }}"
            )?;
            writeln!(writer, "        .uri {{ color: #0066cc; }}")?;
            writeln!(writer, "        .literal {{ color: #009900; }}")?;
            writeln!(
                writer,
                "        .bnode {{ color: #cc6600; font-style: italic; }}"
            )?;
            writeln!(writer, "    </style>")?;
        }

        writeln!(writer, "</head>")?;
        writeln!(writer, "<body>")?;
        writeln!(writer, "    <h1>{}</h1>", Self::escape_html(title))?;

        writeln!(writer, "    <table>")?;
        writeln!(writer, "        <thead>")?;
        writeln!(writer, "            <tr>")?;
        for var in &results.variables {
            writeln!(
                writer,
                "                <th>{}</th>",
                Self::escape_html(var)
            )?;
        }
        writeln!(writer, "            </tr>")?;
        writeln!(writer, "        </thead>")?;
        writeln!(writer, "        <tbody>")?;

        for binding in &results.bindings {
            writeln!(writer, "            <tr>")?;
            for (i, _var) in results.variables.iter().enumerate() {
                write!(writer, "                <td>")?;

                if i < binding.values.len() {
                    if let Some(term) = &binding.values[i] {
                        self.write_html_term(writer, term)?;
                    }
                }

                writeln!(writer, "</td>")?;
            }
            writeln!(writer, "            </tr>")?;
        }

        writeln!(writer, "        </tbody>")?;
        writeln!(writer, "    </table>")?;

        writeln!(writer, "    <div class=\"summary\">")?;
        writeln!(
            writer,
            "        <strong>Total Results:</strong> {}",
            results.bindings.len()
        )?;
        writeln!(writer, "    </div>")?;

        writeln!(writer, "</body>")?;
        writeln!(writer, "</html>")?;

        Ok(())
    }

    /// Convert results to JSON-serializable format
    fn convert_to_json(&self, results: &QueryResults) -> serde_json::Value {
        let bindings: Vec<serde_json::Value> = results
            .bindings
            .iter()
            .map(|binding| {
                let mut obj = serde_json::Map::new();
                for (i, var) in results.variables.iter().enumerate() {
                    if i < binding.values.len() {
                        if let Some(term) = &binding.values[i] {
                            obj.insert(var.clone(), Self::term_to_json(term));
                        }
                    }
                }
                serde_json::Value::Object(obj)
            })
            .collect();

        serde_json::json!({
            "head": {
                "vars": results.variables
            },
            "results": {
                "bindings": bindings
            }
        })
    }

    /// Convert an RDF term to JSON
    fn term_to_json(term: &RdfTerm) -> serde_json::Value {
        match term {
            RdfTerm::Uri { value } => serde_json::json!({
                "type": "uri",
                "value": value
            }),
            RdfTerm::Literal {
                value,
                lang: Some(lang),
                ..
            } => serde_json::json!({
                "type": "literal",
                "value": value,
                "xml:lang": lang
            }),
            RdfTerm::Literal {
                value,
                datatype: Some(datatype),
                ..
            } => serde_json::json!({
                "type": "literal",
                "value": value,
                "datatype": datatype
            }),
            RdfTerm::Literal { value, .. } => serde_json::json!({
                "type": "literal",
                "value": value
            }),
            RdfTerm::Bnode { value } => serde_json::json!({
                "type": "bnode",
                "value": value
            }),
        }
    }

    /// Format an RDF term for CSV export
    fn format_term_for_csv(term: &RdfTerm) -> String {
        match term {
            RdfTerm::Uri { value } => value.clone(),
            RdfTerm::Literal {
                value,
                lang: Some(lang),
                ..
            } => format!("{}@{}", value, lang),
            RdfTerm::Literal {
                value,
                datatype: Some(datatype),
                ..
            } => format!("{}^^{}", value, datatype),
            RdfTerm::Literal { value, .. } => value.clone(),
            RdfTerm::Bnode { value } => format!("_:{}", value),
        }
    }

    /// Write an RDF term as HTML
    fn write_html_term<W: Write>(&self, writer: &mut W, term: &RdfTerm) -> CliResult<()> {
        match term {
            RdfTerm::Uri { value } => {
                write!(
                    writer,
                    "<span class=\"uri\">&lt;{}&gt;</span>",
                    Self::escape_html(value)
                )?;
            }
            RdfTerm::Literal {
                value,
                lang: Some(lang),
                ..
            } => {
                write!(
                    writer,
                    "<span class=\"literal\">\"{}\"@{}</span>",
                    Self::escape_html(value),
                    Self::escape_html(lang)
                )?;
            }
            RdfTerm::Literal {
                value,
                datatype: Some(datatype),
                ..
            } => {
                write!(
                    writer,
                    "<span class=\"literal\">\"{}\"^^&lt;{}&gt;</span>",
                    Self::escape_html(value),
                    Self::escape_html(datatype)
                )?;
            }
            RdfTerm::Literal { value, .. } => {
                write!(
                    writer,
                    "<span class=\"literal\">\"{}\"</span>",
                    Self::escape_html(value)
                )?;
            }
            RdfTerm::Bnode { value } => {
                write!(
                    writer,
                    "<span class=\"bnode\">_:{}</span>",
                    Self::escape_html(value)
                )?;
            }
        }
        Ok(())
    }

    /// Export results to Excel (XLSX) format
    #[cfg(feature = "excel-export")]
    fn export_xlsx(&self, results: &QueryResults, path: &Path) -> CliResult<()> {
        use rust_xlsxwriter::{Format, Workbook};

        // Create a new workbook
        let mut workbook = Workbook::new();

        // Determine sheet name
        let sheet_name = self
            .config
            .xlsx_sheet_name
            .as_deref()
            .unwrap_or("Query Results");

        // Add a worksheet
        let worksheet = workbook.add_worksheet().set_name(sheet_name)?;

        // Create header format (bold, blue background)
        let header_format = Format::new()
            .set_bold()
            .set_background_color(rust_xlsxwriter::Color::RGB(0x0070C0))
            .set_font_color(rust_xlsxwriter::Color::White)
            .set_border(rust_xlsxwriter::FormatBorder::Thin);

        // Create data format with borders
        let data_format = Format::new().set_border(rust_xlsxwriter::FormatBorder::Thin);

        // Write headers
        for (col, var) in results.variables.iter().enumerate() {
            worksheet.write_string_with_format(0, col as u16, var, &header_format)?;
        }

        // Write data rows
        for (row_idx, binding) in results.bindings.iter().enumerate() {
            let excel_row = (row_idx + 1) as u32; // +1 because row 0 is headers

            for (col_idx, value_opt) in binding.values.iter().enumerate() {
                let excel_col = col_idx as u16;

                if let Some(term) = value_opt {
                    let value_str = Self::format_term_for_excel(term);
                    worksheet.write_string_with_format(
                        excel_row,
                        excel_col,
                        &value_str,
                        &data_format,
                    )?;
                } else {
                    // Write empty cell with border
                    worksheet.write_string_with_format(excel_row, excel_col, "", &data_format)?;
                }
            }
        }

        // Auto-fit columns if requested
        if self.config.xlsx_autofit {
            // Auto-fit all columns
            worksheet.autofit();
        }

        // Save the workbook
        workbook.save(path)?;

        Ok(())
    }

    /// Format an RDF term for Excel export
    #[cfg(feature = "excel-export")]
    fn format_term_for_excel(term: &RdfTerm) -> String {
        match term {
            RdfTerm::Uri { value } => value.clone(),
            RdfTerm::Literal {
                value,
                lang,
                datatype,
            } => {
                if let Some(lang_tag) = lang {
                    format!("\"{}\"@{}", value, lang_tag)
                } else if let Some(dt) = datatype {
                    // For common datatypes, just show the value
                    if dt.ends_with("#string")
                        || dt.ends_with("#integer")
                        || dt.ends_with("#decimal")
                        || dt.ends_with("#double")
                        || dt.ends_with("#boolean")
                    {
                        value.clone()
                    } else {
                        format!("\"{}\"^^<{}>", value, dt)
                    }
                } else {
                    format!("\"{}\"", value)
                }
            }
            RdfTerm::Bnode { value } => format!("_:{}", value),
        }
    }

    /// Escape a string for CSV format
    fn escape_csv(s: &str) -> String {
        s.replace('\"', "\"\"")
    }

    /// Escape a string for HTML format
    fn escape_html(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('\"', "&quot;")
            .replace('\'', "&#39;")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::formatters::Binding;

    fn create_test_results() -> QueryResults {
        let variables = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];

        let bindings = vec![
            Binding {
                values: vec![
                    Some(RdfTerm::Uri {
                        value: "http://example.org/alice".to_string(),
                    }),
                    Some(RdfTerm::Uri {
                        value: "http://xmlns.com/foaf/0.1/name".to_string(),
                    }),
                    Some(RdfTerm::Literal {
                        value: "Alice".to_string(),
                        lang: Some("en".to_string()),
                        datatype: None,
                    }),
                ],
            },
            Binding {
                values: vec![
                    Some(RdfTerm::Uri {
                        value: "http://example.org/bob".to_string(),
                    }),
                    Some(RdfTerm::Uri {
                        value: "http://xmlns.com/foaf/0.1/age".to_string(),
                    }),
                    Some(RdfTerm::Literal {
                        value: "30".to_string(),
                        lang: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                    }),
                ],
            },
            Binding {
                values: vec![
                    Some(RdfTerm::Bnode {
                        value: "b1".to_string(),
                    }),
                    Some(RdfTerm::Uri {
                        value: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    }),
                    Some(RdfTerm::Uri {
                        value: "http://xmlns.com/foaf/0.1/Person".to_string(),
                    }),
                ],
            },
        ];

        QueryResults {
            variables,
            bindings,
        }
    }

    #[test]
    fn test_export_format_parsing() {
        assert_eq!(ExportFormat::parse("csv"), Some(ExportFormat::Csv));
        assert_eq!(ExportFormat::parse("xlsx"), Some(ExportFormat::Xlsx));
        assert_eq!(ExportFormat::parse("excel"), Some(ExportFormat::Xlsx));
        assert_eq!(ExportFormat::parse("json"), Some(ExportFormat::Json));
        assert_eq!(ExportFormat::parse("html"), Some(ExportFormat::Html));
        assert_eq!(ExportFormat::parse("CSV"), Some(ExportFormat::Csv));
        assert_eq!(ExportFormat::parse("invalid"), None);
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Csv.extension(), "csv");
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Html.extension(), "html");
    }

    #[test]
    fn test_export_format_mime_type() {
        assert_eq!(ExportFormat::Csv.mime_type(), "text/csv");
        assert_eq!(ExportFormat::Json.mime_type(), "application/json");
        assert_eq!(ExportFormat::Html.mime_type(), "text/html");
    }

    #[test]
    fn test_csv_export() {
        let results = create_test_results();
        let exporter = ResultExporter::new(ExportFormat::Csv);
        let output = exporter.export_to_string(&results).unwrap();

        assert!(output.contains("\"subject\",\"predicate\",\"object\""));
        assert!(output.contains("Alice"));
        assert!(output.contains("30"));
        assert!(output.contains("_:b1"));
    }

    #[test]
    fn test_json_export() {
        let results = create_test_results();
        let exporter = ResultExporter::new(ExportFormat::Json);
        let output = exporter.export_to_string(&results).unwrap();

        assert!(output.contains("\"head\""));
        assert!(output.contains("\"vars\""));
        assert!(output.contains("\"results\""));
        assert!(output.contains("\"bindings\""));
        assert!(output.contains("Alice"));
    }

    #[test]
    fn test_html_export() {
        let results = create_test_results();
        let exporter = ResultExporter::new(ExportFormat::Html);
        let output = exporter.export_to_string(&results).unwrap();

        assert!(output.contains("<!DOCTYPE html>"));
        assert!(output.contains("<table>"));
        assert!(output.contains("<thead>"));
        assert!(output.contains("<tbody>"));
        assert!(output.contains("Alice"));
        assert!(output.contains("Total Results:"));
    }

    #[test]
    fn test_csv_without_headers() {
        let results = create_test_results();
        let config = ExportConfig::csv().with_headers(false);
        let exporter = ResultExporter::with_config(config);
        let output = exporter.export_to_string(&results).unwrap();

        assert!(!output.contains("\"subject\",\"predicate\",\"object\""));
        assert!(output.contains("Alice"));
    }

    #[test]
    fn test_json_compact() {
        let results = create_test_results();
        let config = ExportConfig::json().with_pretty_json(false);
        let exporter = ResultExporter::with_config(config);
        let output = exporter.export_to_string(&results).unwrap();

        // Compact JSON should not have indentation
        assert!(!output.contains("  "));
        assert!(output.contains("\"head\""));
    }

    #[test]
    fn test_html_custom_title() {
        let results = create_test_results();
        let config = ExportConfig::html().with_html_title("Custom Results".to_string());
        let exporter = ResultExporter::with_config(config);
        let output = exporter.export_to_string(&results).unwrap();

        assert!(output.contains("<title>Custom Results</title>"));
        assert!(output.contains("<h1>Custom Results</h1>"));
    }

    #[test]
    fn test_csv_escaping() {
        assert_eq!(ResultExporter::escape_csv("normal"), "normal");
        assert_eq!(ResultExporter::escape_csv("with\"quote"), "with\"\"quote");
    }

    #[test]
    fn test_html_escaping() {
        assert_eq!(ResultExporter::escape_html("normal"), "normal");
        assert_eq!(ResultExporter::escape_html("<tag>"), "&lt;tag&gt;");
        assert_eq!(ResultExporter::escape_html("a & b"), "a &amp; b");
    }

    #[test]
    fn test_empty_results() {
        let results = QueryResults {
            variables: vec!["x".to_string()],
            bindings: vec![],
        };

        let csv_exporter = ResultExporter::new(ExportFormat::Csv);
        let csv_output = csv_exporter.export_to_string(&results).unwrap();
        assert!(csv_output.contains("\"x\""));

        let json_exporter = ResultExporter::new(ExportFormat::Json);
        let json_output = json_exporter.export_to_string(&results).unwrap();
        assert!(json_output.contains("\"bindings\": []"));

        let html_exporter = ResultExporter::new(ExportFormat::Html);
        let html_output = html_exporter.export_to_string(&results).unwrap();
        assert!(html_output.contains("Total Results:"));
    }

    #[test]
    #[cfg(feature = "excel-export")]
    fn test_xlsx_export() {
        use std::env;

        let results = create_test_results();
        let xlsx_exporter = ResultExporter::with_config(ExportConfig::xlsx());

        // Create a temporary file
        let temp_dir = env::temp_dir();
        let temp_file = temp_dir.join("test_export.xlsx");

        // Export to XLSX
        let export_result = xlsx_exporter.export_to_file(&results, &temp_file);
        assert!(
            export_result.is_ok(),
            "Excel export failed: {:?}",
            export_result.err()
        );

        // Verify file was created
        assert!(temp_file.exists(), "XLSX file was not created");

        // Verify file has content (XLSX files should be at least a few KB)
        let metadata = std::fs::metadata(&temp_file).unwrap();
        assert!(
            metadata.len() > 1000,
            "XLSX file is too small: {} bytes",
            metadata.len()
        );

        // Clean up
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    #[cfg(feature = "excel-export")]
    fn test_xlsx_format_term() {
        let uri_term = RdfTerm::Uri {
            value: "http://example.org/test".to_string(),
        };
        assert_eq!(
            ResultExporter::format_term_for_excel(&uri_term),
            "http://example.org/test"
        );

        let literal_term = RdfTerm::Literal {
            value: "Hello".to_string(),
            lang: Some("en".to_string()),
            datatype: None,
        };
        assert_eq!(
            ResultExporter::format_term_for_excel(&literal_term),
            "\"Hello\"@en"
        );

        let bnode_term = RdfTerm::Bnode {
            value: "b1".to_string(),
        };
        assert_eq!(ResultExporter::format_term_for_excel(&bnode_term), "_:b1");
    }

    #[test]
    #[cfg(feature = "excel-export")]
    fn test_xlsx_custom_sheet_name() {
        use std::env;

        let results = create_test_results();
        let config = ExportConfig::xlsx().with_xlsx_sheet_name("My Results".to_string());
        let xlsx_exporter = ResultExporter::with_config(config);

        let temp_dir = env::temp_dir();
        let temp_file = temp_dir.join("test_custom_sheet.xlsx");

        let result = xlsx_exporter.export_to_file(&results, &temp_file);
        assert!(result.is_ok(), "Excel export with custom sheet name failed");

        assert!(temp_file.exists());
        std::fs::remove_file(&temp_file).ok();
    }
}
