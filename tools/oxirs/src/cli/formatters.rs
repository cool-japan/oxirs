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
        assert!(create_formatter("unknown").is_none());
    }
}
