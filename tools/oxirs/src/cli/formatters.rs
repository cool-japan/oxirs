//! SPARQL Query Result Formatters
//!
//! Provides comprehensive formatting for SPARQL query results in multiple standard formats:
//! - Table (ASCII/Unicode)
//! - JSON (SPARQL 1.1 Results JSON Format)
//! - CSV/TSV (SPARQL 1.1 Results CSV/TSV Format)
//! - XML (SPARQL Results Format XML)

use prettytable::{format, Cell, Row, Table as PrettyTable};
use serde::{Deserialize, Serialize};
use std::io::Write;

/// SPARQL query results representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResults {
    pub variables: Vec<String>,
    pub bindings: Vec<Binding>,
}

/// Variable binding in a SPARQL result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    pub values: Vec<Option<RdfTerm>>,
}

/// RDF term representation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RdfTerm {
    Uri {
        value: String,
    },
    Literal {
        value: String,
        lang: Option<String>,
        datatype: Option<String>,
    },
    Bnode {
        value: String,
    },
}

impl RdfTerm {
    /// Convert to string representation
    pub fn to_string_repr(&self) -> String {
        match self {
            RdfTerm::Uri { value } => format!("<{}>", value),
            RdfTerm::Literal {
                value,
                lang: Some(lang),
                ..
            } => format!("\"{}\"@{}", value, lang),
            RdfTerm::Literal {
                value,
                datatype: Some(dt),
                ..
            } => format!("\"{}\"^^<{}>", value, dt),
            RdfTerm::Literal { value, .. } => format!("\"{}\"", value),
            RdfTerm::Bnode { value } => format!("_:{}", value),
        }
    }

    /// Convert to plain value (without RDF syntax)
    pub fn to_plain_value(&self) -> String {
        match self {
            RdfTerm::Uri { value } => value.clone(),
            RdfTerm::Literal { value, .. } => value.clone(),
            RdfTerm::Bnode { value } => format!("_:{}", value),
        }
    }
}

/// Result formatter trait
pub trait ResultFormatter {
    /// Format query results
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()>;
}

/// ASCII table formatter
pub struct TableFormatter {
    pub unicode_borders: bool,
    pub max_column_width: usize,
}

impl Default for TableFormatter {
    fn default() -> Self {
        Self {
            unicode_borders: true,
            max_column_width: 80,
        }
    }
}

impl ResultFormatter for TableFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        let mut table = PrettyTable::new();

        // Set table format
        if self.unicode_borders {
            table.set_format(*format::consts::FORMAT_BOX_CHARS);
        } else {
            table.set_format(*format::consts::FORMAT_BORDERS_ONLY);
        }

        // Add header row
        let header_cells: Vec<Cell> = results
            .variables
            .iter()
            .map(|v| Cell::new(&format!("?{}", v)))
            .collect();
        table.set_titles(Row::new(header_cells));

        // Add data rows
        for binding in &results.bindings {
            let cells: Vec<Cell> = binding
                .values
                .iter()
                .map(|opt_term| {
                    let value = match opt_term {
                        Some(term) => {
                            let repr = term.to_string_repr();
                            if repr.len() > self.max_column_width {
                                format!("{}...", &repr[..self.max_column_width - 3])
                            } else {
                                repr
                            }
                        }
                        None => "-".to_string(),
                    };
                    Cell::new(&value)
                })
                .collect();
            table.add_row(Row::new(cells));
        }

        // Write table
        write!(writer, "{}", table)?;

        // Write summary
        writeln!(writer)?;
        writeln!(
            writer,
            "({} result{} returned)",
            results.bindings.len(),
            if results.bindings.len() == 1 { "" } else { "s" }
        )?;

        Ok(())
    }
}

/// JSON formatter (SPARQL 1.1 Results JSON Format)
pub struct JsonFormatter {
    pub pretty: bool,
}

impl Default for JsonFormatter {
    fn default() -> Self {
        Self { pretty: true }
    }
}

impl ResultFormatter for JsonFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        #[derive(Serialize)]
        struct SparqlJsonResults<'a> {
            head: Head<'a>,
            results: Results,
        }

        #[derive(Serialize)]
        struct Head<'a> {
            vars: &'a [String],
        }

        #[derive(Serialize)]
        struct Results {
            bindings: Vec<serde_json::Map<String, serde_json::Value>>,
        }

        let sparql_bindings: Vec<serde_json::Map<String, serde_json::Value>> = results
            .bindings
            .iter()
            .map(|binding| {
                let mut map = serde_json::Map::new();
                for (idx, var) in results.variables.iter().enumerate() {
                    if let Some(Some(term)) = binding.values.get(idx) {
                        let term_json = match term {
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
                                datatype: Some(dt),
                                ..
                            } => serde_json::json!({
                                "type": "literal",
                                "value": value,
                                "datatype": dt
                            }),
                            RdfTerm::Literal { value, .. } => serde_json::json!({
                                "type": "literal",
                                "value": value
                            }),
                            RdfTerm::Bnode { value } => serde_json::json!({
                                "type": "bnode",
                                "value": value
                            }),
                        };
                        map.insert(var.clone(), term_json);
                    }
                }
                map
            })
            .collect();

        let output = SparqlJsonResults {
            head: Head {
                vars: &results.variables,
            },
            results: Results {
                bindings: sparql_bindings,
            },
        };

        let json_string = if self.pretty {
            serde_json::to_string_pretty(&output)
        } else {
            serde_json::to_string(&output)
        }
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        writeln!(writer, "{}", json_string)?;
        Ok(())
    }
}

/// CSV formatter (SPARQL 1.1 Results CSV Format)
pub struct CsvFormatter {
    pub separator: char,
}

impl Default for CsvFormatter {
    fn default() -> Self {
        Self { separator: ',' }
    }
}

impl CsvFormatter {
    pub fn new_tsv() -> Self {
        Self { separator: '\t' }
    }
}

impl ResultFormatter for CsvFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        // Write header
        let header: Vec<String> = results
            .variables
            .iter()
            .map(|v| format!("?{}", v))
            .collect();
        writeln!(writer, "{}", header.join(&self.separator.to_string()))?;

        // Write data rows
        for binding in &results.bindings {
            let row: Vec<String> = binding
                .values
                .iter()
                .map(|opt_term| match opt_term {
                    Some(term) => escape_csv_value(&term.to_plain_value(), self.separator),
                    None => String::new(),
                })
                .collect();
            writeln!(writer, "{}", row.join(&self.separator.to_string()))?;
        }

        Ok(())
    }
}

/// XML formatter (SPARQL Results Format XML)
pub struct XmlFormatter {
    pub pretty: bool,
}

impl Default for XmlFormatter {
    fn default() -> Self {
        Self { pretty: true }
    }
}

impl ResultFormatter for XmlFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        let indent = if self.pretty { "  " } else { "" };
        let _newline = if self.pretty { "\n" } else { "" };

        // XML header
        writeln!(writer, "<?xml version=\"1.0\"?>")?;
        writeln!(
            writer,
            "<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">"
        )?;

        // Head section
        writeln!(writer, "{}<head>", indent)?;
        for var in &results.variables {
            writeln!(
                writer,
                "{}{}<variable name=\"{}\"/>",
                indent,
                indent,
                escape_xml(var)
            )?;
        }
        writeln!(writer, "{}</head>", indent)?;

        // Results section
        writeln!(writer, "{}<results>", indent)?;
        for binding in &results.bindings {
            writeln!(writer, "{}{}<result>", indent, indent)?;
            for (idx, var) in results.variables.iter().enumerate() {
                if let Some(Some(term)) = binding.values.get(idx) {
                    write!(
                        writer,
                        "{}{}{}<binding name=\"{}\">",
                        indent,
                        indent,
                        indent,
                        escape_xml(var)
                    )?;

                    match term {
                        RdfTerm::Uri { value } => {
                            write!(writer, "<uri>{}</uri>", escape_xml(value))?;
                        }
                        RdfTerm::Literal {
                            value,
                            lang: Some(lang),
                            ..
                        } => {
                            write!(
                                writer,
                                "<literal xml:lang=\"{}\">{}</literal>",
                                escape_xml(lang),
                                escape_xml(value)
                            )?;
                        }
                        RdfTerm::Literal {
                            value,
                            datatype: Some(dt),
                            ..
                        } => {
                            write!(
                                writer,
                                "<literal datatype=\"{}\">{}</literal>",
                                escape_xml(dt),
                                escape_xml(value)
                            )?;
                        }
                        RdfTerm::Literal { value, .. } => {
                            write!(writer, "<literal>{}</literal>", escape_xml(value))?;
                        }
                        RdfTerm::Bnode { value } => {
                            write!(writer, "<bnode>{}</bnode>", escape_xml(value))?;
                        }
                    }

                    writeln!(writer, "</binding>")?;
                }
            }
            writeln!(writer, "{}{}</result>", indent, indent)?;
        }
        writeln!(writer, "{}</results>", indent)?;

        writeln!(writer, "</sparql>")?;
        Ok(())
    }
}

/// Escape CSV value
fn escape_csv_value(value: &str, separator: char) -> String {
    if value.contains(separator) || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Escape XML special characters
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Escape HTML special characters
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// HTML table formatter
pub struct HtmlFormatter {
    pub styled: bool,
    pub compact: bool,
}

impl Default for HtmlFormatter {
    fn default() -> Self {
        Self {
            styled: true,
            compact: false,
        }
    }
}

impl ResultFormatter for HtmlFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        let nl = if self.compact { "" } else { "\n" };
        let indent = if self.compact { "" } else { "  " };
        let indent2 = if self.compact { "" } else { "    " };

        // Write HTML header
        writeln!(writer, "<!DOCTYPE html>")?;
        writeln!(writer, "<html>")?;
        writeln!(writer, "<head>")?;
        writeln!(writer, "{}<meta charset=\"UTF-8\">", indent)?;
        writeln!(writer, "{}<title>SPARQL Query Results</title>", indent)?;

        // Add CSS styling if enabled
        if self.styled {
            writeln!(writer, "{}<style>", indent)?;
            writeln!(
                writer,
                "{}body {{ font-family: Arial, sans-serif; margin: 20px; }}",
                indent2
            )?;
            writeln!(
                writer,
                "{}table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}",
                indent2
            )?;
            writeln!(
                writer,
                "{}th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}",
                indent2
            )?;
            writeln!(
                writer,
                "{}th {{ background-color: #4CAF50; color: white; font-weight: bold; }}",
                indent2
            )?;
            writeln!(
                writer,
                "{}tr:nth-child(even) {{ background-color: #f2f2f2; }}",
                indent2
            )?;
            writeln!(writer, "{}tr:hover {{ background-color: #ddd; }}", indent2)?;
            writeln!(writer, "{}.uri {{ color: #0066cc; }}", indent2)?;
            writeln!(writer, "{}.literal {{ color: #006600; }}", indent2)?;
            writeln!(
                writer,
                "{}.bnode {{ color: #cc6600; font-style: italic; }}",
                indent2
            )?;
            writeln!(
                writer,
                "{}.lang {{ font-size: 0.9em; color: #666; }}",
                indent2
            )?;
            writeln!(
                writer,
                "{}.datatype {{ font-size: 0.9em; color: #999; }}",
                indent2
            )?;
            writeln!(writer, "{}</style>", indent)?;
        }

        writeln!(writer, "</head>")?;
        writeln!(writer, "<body>")?;

        // Write header
        writeln!(writer, "{}<h1>SPARQL Query Results</h1>", indent)?;
        writeln!(
            writer,
            "{}<p>Variables: {}</p>",
            indent,
            results.variables.len()
        )?;
        writeln!(
            writer,
            "{}<p>Results: {}</p>",
            indent,
            results.bindings.len()
        )?;

        // Write table
        writeln!(writer, "{}<table>", indent)?;

        // Table header
        write!(writer, "{}<thead><tr>", indent2)?;
        for var in &results.variables {
            write!(writer, "<th>?{}</th>", escape_html(var))?;
        }
        writeln!(writer, "</tr></thead>{}", nl)?;

        // Table body
        writeln!(writer, "{}<tbody>", indent2)?;
        for binding in &results.bindings {
            write!(writer, "{}<tr>", indent2)?;
            for value in &binding.values {
                write!(writer, "<td>")?;
                if let Some(term) = value {
                    match term {
                        RdfTerm::Uri { value } => {
                            if self.styled {
                                write!(
                                    writer,
                                    "<span class=\"uri\">&lt;{}&gt;</span>",
                                    escape_html(value)
                                )?;
                            } else {
                                write!(writer, "&lt;{}&gt;", escape_html(value))?;
                            }
                        }
                        RdfTerm::Literal {
                            value,
                            lang: Some(lang),
                            ..
                        } => {
                            if self.styled {
                                write!(
                                    writer,
                                    "<span class=\"literal\">\"{}\"</span><span class=\"lang\">@{}</span>",
                                    escape_html(value),
                                    escape_html(lang)
                                )?;
                            } else {
                                write!(writer, "\"{}\"@{}", escape_html(value), escape_html(lang))?;
                            }
                        }
                        RdfTerm::Literal {
                            value,
                            datatype: Some(dt),
                            ..
                        } => {
                            if self.styled {
                                write!(
                                    writer,
                                    "<span class=\"literal\">\"{}\"</span><span class=\"datatype\">^^&lt;{}&gt;</span>",
                                    escape_html(value),
                                    escape_html(dt)
                                )?;
                            } else {
                                write!(
                                    writer,
                                    "\"{}\"^^&lt;{}&gt;",
                                    escape_html(value),
                                    escape_html(dt)
                                )?;
                            }
                        }
                        RdfTerm::Literal { value, .. } => {
                            if self.styled {
                                write!(
                                    writer,
                                    "<span class=\"literal\">\"{}\"</span>",
                                    escape_html(value)
                                )?;
                            } else {
                                write!(writer, "\"{}\"", escape_html(value))?;
                            }
                        }
                        RdfTerm::Bnode { value } => {
                            if self.styled {
                                write!(
                                    writer,
                                    "<span class=\"bnode\">_:{}</span>",
                                    escape_html(value)
                                )?;
                            } else {
                                write!(writer, "_:{}", escape_html(value))?;
                            }
                        }
                    }
                }
                write!(writer, "</td>")?;
            }
            writeln!(writer, "</tr>{}", nl)?;
        }
        writeln!(writer, "{}</tbody>", indent2)?;
        writeln!(writer, "{}</table>", indent)?;

        writeln!(writer, "</body>")?;
        writeln!(writer, "</html>")?;
        Ok(())
    }
}

/// Markdown table formatter
pub struct MarkdownFormatter {
    pub aligned: bool,
}

impl Default for MarkdownFormatter {
    fn default() -> Self {
        Self { aligned: true }
    }
}

impl ResultFormatter for MarkdownFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        // Calculate column widths if aligned
        let col_widths: Vec<usize> = if self.aligned {
            results
                .variables
                .iter()
                .enumerate()
                .map(|(idx, var)| {
                    let header_len = var.len() + 1; // +1 for '?'
                    let max_value_len = results
                        .bindings
                        .iter()
                        .filter_map(|b| b.values.get(idx))
                        .filter_map(|v| v.as_ref())
                        .map(|term| term.to_string_repr().len())
                        .max()
                        .unwrap_or(0);
                    std::cmp::max(header_len, max_value_len)
                })
                .collect()
        } else {
            vec![0; results.variables.len()]
        };

        // Write header row
        write!(writer, "|")?;
        for (idx, var) in results.variables.iter().enumerate() {
            let width = if self.aligned { col_widths[idx] } else { 0 };
            if self.aligned {
                write!(writer, " {:width$} |", format!("?{}", var), width = width)?;
            } else {
                write!(writer, " ?{} |", var)?;
            }
        }
        writeln!(writer)?;

        // Write separator row
        write!(writer, "|")?;
        for &width in &col_widths {
            if self.aligned {
                write!(writer, " {} |", "-".repeat(width))?;
            } else {
                write!(writer, " --- |")?;
            }
        }
        writeln!(writer)?;

        // Write data rows
        for binding in &results.bindings {
            write!(writer, "|")?;
            for (idx, value) in binding.values.iter().enumerate() {
                let cell_value = value
                    .as_ref()
                    .map(|t| escape_markdown(&t.to_string_repr()))
                    .unwrap_or_else(String::new);

                let width = if self.aligned { col_widths[idx] } else { 0 };
                if self.aligned {
                    write!(writer, " {:width$} |", cell_value, width = width)?;
                } else {
                    write!(writer, " {} |", cell_value)?;
                }
            }
            writeln!(writer)?;
        }

        // Write summary
        writeln!(writer)?;
        writeln!(
            writer,
            "*{} results for {} variables*",
            results.bindings.len(),
            results.variables.len()
        )?;

        Ok(())
    }
}

/// Escape Markdown special characters
fn escape_markdown(s: &str) -> String {
    s.replace('|', "\\|")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace('*', "\\*")
        .replace('_', "\\_")
        .replace('`', "\\`")
}

/// Excel (.xlsx) spreadsheet formatter
pub struct ExcelFormatter {
    pub sheet_name: String,
    pub auto_filter: bool,
    pub freeze_header: bool,
}

impl Default for ExcelFormatter {
    fn default() -> Self {
        Self {
            sheet_name: "SPARQL Results".to_string(),
            auto_filter: true,
            freeze_header: true,
        }
    }
}

impl ResultFormatter for ExcelFormatter {
    fn format(&self, results: &QueryResults, writer: &mut dyn Write) -> std::io::Result<()> {
        use rust_xlsxwriter::{Format, Workbook};

        // Create workbook in memory
        let mut workbook = Workbook::new();
        let worksheet = workbook
            .add_worksheet()
            .set_name(&self.sheet_name)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Create formats
        let header_format = Format::new()
            .set_bold()
            .set_background_color(rust_xlsxwriter::Color::RGB(0x4472C4))
            .set_font_color(rust_xlsxwriter::Color::White);

        let uri_format = Format::new().set_font_color(rust_xlsxwriter::Color::RGB(0x2E75B6));

        let literal_format = Format::new().set_font_color(rust_xlsxwriter::Color::RGB(0x548235));

        let bnode_format = Format::new().set_font_color(rust_xlsxwriter::Color::RGB(0x7030A0));

        // Write header row
        for (col_idx, var) in results.variables.iter().enumerate() {
            worksheet
                .write_string_with_format(0, col_idx as u16, format!("?{}", var), &header_format)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }

        // Write data rows
        for (row_idx, binding) in results.bindings.iter().enumerate() {
            for (col_idx, value) in binding.values.iter().enumerate() {
                let row = (row_idx + 1) as u32;
                let col = col_idx as u16;

                match value {
                    Some(RdfTerm::Uri { value: uri }) => {
                        worksheet
                            .write_string_with_format(row, col, uri, &uri_format)
                            .map_err(|e| {
                                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                            })?;
                    }
                    Some(RdfTerm::Literal { value: lit, .. }) => {
                        worksheet
                            .write_string_with_format(row, col, lit, &literal_format)
                            .map_err(|e| {
                                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                            })?;
                    }
                    Some(RdfTerm::Bnode { value: bnode }) => {
                        worksheet
                            .write_string_with_format(
                                row,
                                col,
                                format!("_:{}", bnode),
                                &bnode_format,
                            )
                            .map_err(|e| {
                                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                            })?;
                    }
                    None => {
                        worksheet.write_string(row, col, "").map_err(|e| {
                            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                        })?;
                    }
                }
            }
        }

        // Apply auto-filter to header row
        if self.auto_filter && !results.variables.is_empty() {
            let last_col = (results.variables.len() - 1) as u16;
            let last_row = results.bindings.len() as u32;
            worksheet
                .autofilter(0, 0, last_row, last_col)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }

        // Freeze header row
        if self.freeze_header {
            worksheet
                .set_freeze_panes(1, 0)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }

        // Auto-fit columns (approximate)
        for col_idx in 0..results.variables.len() {
            worksheet
                .set_column_width(col_idx as u16, 20.0)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }

        // Save to buffer
        let buffer = workbook
            .save_to_buffer()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Write buffer to writer
        writer.write_all(&buffer)?;

        Ok(())
    }
}

/// Factory function to create formatter by name
pub fn create_formatter(format: &str) -> Option<Box<dyn ResultFormatter>> {
    match format.to_lowercase().as_str() {
        "table" | "text" => Some(Box::new(TableFormatter::default())),
        "table-ascii" => Some(Box::new(TableFormatter {
            unicode_borders: false,
            max_column_width: 80,
        })),
        "json" => Some(Box::new(JsonFormatter::default())),
        "json-compact" => Some(Box::new(JsonFormatter { pretty: false })),
        "csv" => Some(Box::new(CsvFormatter::default())),
        "tsv" => Some(Box::new(CsvFormatter::new_tsv())),
        "xml" => Some(Box::new(XmlFormatter::default())),
        "xml-compact" => Some(Box::new(XmlFormatter { pretty: false })),
        "html" | "html-styled" => Some(Box::new(HtmlFormatter::default())),
        "html-plain" => Some(Box::new(HtmlFormatter {
            styled: false,
            compact: false,
        })),
        "html-compact" => Some(Box::new(HtmlFormatter {
            styled: true,
            compact: true,
        })),
        "markdown" | "md" => Some(Box::new(MarkdownFormatter::default())),
        "markdown-compact" | "md-compact" => Some(Box::new(MarkdownFormatter { aligned: false })),
        "xlsx" | "excel" => Some(Box::new(ExcelFormatter::default())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_results() -> QueryResults {
        QueryResults {
            variables: vec!["s".to_string(), "p".to_string(), "o".to_string()],
            bindings: vec![
                Binding {
                    values: vec![
                        Some(RdfTerm::Uri {
                            value: "http://example.org/resource/1".to_string(),
                        }),
                        Some(RdfTerm::Uri {
                            value: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                        }),
                        Some(RdfTerm::Literal {
                            value: "Example".to_string(),
                            lang: Some("en".to_string()),
                            datatype: None,
                        }),
                    ],
                },
                Binding {
                    values: vec![
                        Some(RdfTerm::Bnode {
                            value: "b0".to_string(),
                        }),
                        Some(RdfTerm::Uri {
                            value: "http://example.org/property/value".to_string(),
                        }),
                        Some(RdfTerm::Literal {
                            value: "42".to_string(),
                            lang: None,
                            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                        }),
                    ],
                },
            ],
        }
    }

    #[test]
    fn test_table_formatter() {
        let results = create_test_results();
        let formatter = TableFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("?s"));
        assert!(output_str.contains("?p"));
        assert!(output_str.contains("?o"));
        assert!(output_str.contains("2 results"));
    }

    #[test]
    fn test_json_formatter() {
        let results = create_test_results();
        let formatter = JsonFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("\"head\""));
        assert!(output_str.contains("\"vars\""));
        assert!(output_str.contains("\"results\""));
        assert!(output_str.contains("\"bindings\""));

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&output_str).unwrap();
        assert!(parsed["head"]["vars"].is_array());
        assert!(parsed["results"]["bindings"].is_array());
    }

    #[test]
    fn test_csv_formatter() {
        let results = create_test_results();
        let formatter = CsvFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 data rows
        assert!(lines[0].contains("?s,?p,?o"));
    }

    #[test]
    fn test_xml_formatter() {
        let results = create_test_results();
        let formatter = XmlFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("<?xml version=\"1.0\"?>"));
        assert!(output_str.contains("<sparql"));
        assert!(output_str.contains("<head>"));
        assert!(output_str.contains("<results>"));
        assert!(output_str.contains("<variable name=\"s\""));
        assert!(output_str.contains("<binding name=\"s\">"));
    }

    #[test]
    fn test_escape_csv() {
        assert_eq!(escape_csv_value("simple", ','), "simple");
        assert_eq!(escape_csv_value("with,comma", ','), "\"with,comma\"");
        assert_eq!(escape_csv_value("with\"quote", ','), "\"with\"\"quote\"");
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("simple"), "simple");
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
    }

    #[test]
    fn test_formatter_factory() {
        assert!(create_formatter("table").is_some());
        assert!(create_formatter("json").is_some());
        assert!(create_formatter("csv").is_some());
        assert!(create_formatter("tsv").is_some());
        assert!(create_formatter("xml").is_some());
        assert!(create_formatter("html").is_some());
        assert!(create_formatter("markdown").is_some());
        assert!(create_formatter("md").is_some());
        assert!(create_formatter("unknown").is_none());
    }

    #[test]
    fn test_html_formatter() {
        let results = create_test_results();
        let formatter = HtmlFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        // Verify HTML structure
        assert!(output_str.contains("<!DOCTYPE html>"));
        assert!(output_str.contains("<html>"));
        assert!(output_str.contains("<head>"));
        assert!(output_str.contains("<title>SPARQL Query Results</title>"));
        assert!(output_str.contains("<table>"));
        assert!(output_str.contains("<thead>"));
        assert!(output_str.contains("<tbody>"));
        assert!(output_str.contains("</html>"));

        // Verify CSS styling
        assert!(output_str.contains("<style>"));
        assert!(output_str.contains("background-color"));

        // Verify headers
        assert!(output_str.contains("?s"));
        assert!(output_str.contains("?p"));
        assert!(output_str.contains("?o"));

        // Verify data content
        assert!(output_str.contains("http://example.org/resource/1"));
        assert!(output_str.contains("Example"));
        assert!(output_str.contains("_:b0"));
    }

    #[test]
    fn test_html_formatter_plain() {
        let results = create_test_results();
        let formatter = HtmlFormatter {
            styled: false,
            compact: false,
        };
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        // Should not have styling
        assert!(!output_str.contains("<style>"));
        assert!(!output_str.contains("class="));

        // But should still have structure
        assert!(output_str.contains("<table>"));
        assert!(output_str.contains("&lt;http://example.org/resource/1&gt;"));
    }

    #[test]
    fn test_html_formatter_compact() {
        let results = create_test_results();
        let formatter = HtmlFormatter {
            styled: false,
            compact: true,
        };
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        // Compact should have less whitespace (no multi-line formatting within table)
        let lines: Vec<&str> = output_str.lines().collect();
        // Exact count depends on implementation, but should be notably fewer lines
        assert!(lines.len() < 30); // Normal is ~50+ lines
    }

    #[test]
    fn test_markdown_formatter() {
        let results = create_test_results();
        let formatter = MarkdownFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        // Verify Markdown table structure
        let lines: Vec<&str> = output_str.lines().collect();

        // Should have header row, separator row, and data rows
        assert!(lines.len() >= 4); // header + separator + 2 data + summary

        // First line should be table header
        assert!(lines[0].contains("?s"));
        assert!(lines[0].contains("?p"));
        assert!(lines[0].contains("?o"));
        assert!(lines[0].starts_with('|'));
        assert!(lines[0].ends_with('|'));

        // Second line should be separator
        assert!(lines[1].contains("---") || lines[1].contains('-'));
        assert!(lines[1].starts_with('|'));

        // Data rows
        assert!(lines[2].contains("http://example.org/resource/1"));
        assert!(lines[3].contains("_:b0"));

        // Summary line
        assert!(output_str.contains("results for"));
        assert!(output_str.contains("variables"));
    }

    #[test]
    fn test_markdown_formatter_compact() {
        let results = create_test_results();
        let formatter = MarkdownFormatter { aligned: false };
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        // Compact version should not align columns (no padding)
        let lines: Vec<&str> = output_str.lines().collect();

        // Header should not have extra spaces for alignment
        assert!(lines[0].contains("| ?s |") || lines[0].contains("| ?s|"));
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("simple"), "simple");
        assert_eq!(escape_html("<tag>"), "&lt;tag&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
        assert_eq!(escape_html("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_html("'single'"), "&#39;single&#39;");
    }

    #[test]
    fn test_escape_markdown() {
        assert_eq!(escape_markdown("simple"), "simple");
        assert_eq!(escape_markdown("a|b"), "a\\|b");
        assert_eq!(escape_markdown("[link]"), "\\[link\\]");
        assert_eq!(escape_markdown("*bold*"), "\\*bold\\*");
        assert_eq!(escape_markdown("_italic_"), "\\_italic\\_");
        assert_eq!(escape_markdown("`code`"), "\\`code\\`");
    }

    #[test]
    fn test_markdown_table_alignment() {
        let results = QueryResults {
            variables: vec!["short".to_string(), "very_long_variable".to_string()],
            bindings: vec![Binding {
                values: vec![
                    Some(RdfTerm::Literal {
                        value: "x".to_string(),
                        lang: None,
                        datatype: None,
                    }),
                    Some(RdfTerm::Literal {
                        value: "test".to_string(),
                        lang: None,
                        datatype: None,
                    }),
                ],
            }],
        };

        let formatter = MarkdownFormatter::default();
        let mut output = Vec::new();
        formatter.format(&results, &mut output).unwrap();
        let output_str = String::from_utf8(output).unwrap();

        let lines: Vec<&str> = output_str.lines().collect();

        // With alignment, all rows should have same width
        let header_len = lines[0].len();
        let separator_len = lines[1].len();

        // Header and separator should be close in length (within reason for separators)
        assert!((header_len as i32 - separator_len as i32).abs() < 10);
    }

    #[test]
    fn test_excel_formatter() {
        let results = create_test_results();
        let formatter = ExcelFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();

        // Verify that output is not empty (Excel file was generated)
        assert!(!output.is_empty());

        // Verify it starts with Excel file signature (PK zip header)
        assert_eq!(&output[0..2], b"PK");

        // Verify reasonable file size (should be at least a few hundred bytes)
        assert!(output.len() > 500);
    }

    #[test]
    fn test_excel_formatter_empty_results() {
        let results = QueryResults {
            variables: vec!["s".to_string()],
            bindings: vec![],
        };
        let formatter = ExcelFormatter::default();
        let mut output = Vec::new();

        formatter.format(&results, &mut output).unwrap();

        // Should still generate a valid Excel file even with no data
        assert!(!output.is_empty());
        assert_eq!(&output[0..2], b"PK");
    }

    #[test]
    fn test_formatter_factory_excel() {
        assert!(create_formatter("xlsx").is_some());
        assert!(create_formatter("excel").is_some());
        assert!(create_formatter("XLSX").is_some());
    }
}
