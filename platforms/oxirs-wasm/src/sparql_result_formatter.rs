//! SPARQL result set formatting in multiple output formats.
//!
//! Supports W3C SPARQL Results JSON, XML, CSV (RFC 4180), TSV, Markdown, and HTML.
//! Handles SELECT result sets, ASK boolean results, and all SPARQL value types.

use std::collections::HashMap;

// ── Value types ────────────────────────────────────────────────────────────────

/// A single SPARQL term binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparqlValue {
    /// An IRI / URI reference
    Uri(String),
    /// A plain, language-tagged, or datatype literal
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    /// A blank node identifier
    BNode(String),
    /// Unbound variable (absent from result row)
    Unbound,
}

impl SparqlValue {
    /// Return the plain string value without type annotations.
    pub fn as_str(&self) -> &str {
        match self {
            SparqlValue::Uri(s) => s,
            SparqlValue::Literal { value, .. } => value,
            SparqlValue::BNode(id) => id,
            SparqlValue::Unbound => "",
        }
    }
}

// ── Results container ──────────────────────────────────────────────────────────

/// A SPARQL result set (SELECT or ASK).
#[derive(Debug, Clone, Default)]
pub struct SparqlResults {
    /// Projected variable names (in order)
    pub variables: Vec<String>,
    /// Solution mappings for SELECT queries
    pub bindings: Vec<HashMap<String, SparqlValue>>,
    /// Boolean result for ASK queries
    pub boolean: Option<bool>,
}

// ── Output formats ────────────────────────────────────────────────────────────

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Xml,
    Csv,
    Tsv,
    Markdown,
    Html,
}

// ── Formatter options ─────────────────────────────────────────────────────────

/// Fine-grained formatting options.
#[derive(Debug, Clone)]
pub struct FormatterOptions {
    /// Whether to include a header row / header declarations
    pub include_header: bool,
    /// String representation for `SparqlValue::Unbound`
    pub null_representation: String,
}

impl Default for FormatterOptions {
    fn default() -> Self {
        Self {
            include_header: true,
            null_representation: String::new(),
        }
    }
}

// ── Main formatter ────────────────────────────────────────────────────────────

/// Stateless SPARQL result formatter.
pub struct SparqlResultFormatter;

impl SparqlResultFormatter {
    /// Format results using default options.
    pub fn format(results: &SparqlResults, format: OutputFormat) -> String {
        Self::format_with_options(results, format, &FormatterOptions::default())
    }

    /// Format results with custom options.
    pub fn format_with_options(
        results: &SparqlResults,
        format: OutputFormat,
        options: &FormatterOptions,
    ) -> String {
        match format {
            OutputFormat::Json => Self::format_json_impl(results),
            OutputFormat::Xml => Self::format_xml_impl(results),
            OutputFormat::Csv => Self::format_csv_impl(results, options),
            OutputFormat::Tsv => Self::format_tsv_impl(results, options),
            OutputFormat::Markdown => Self::format_markdown_impl(results, options),
            OutputFormat::Html => Self::format_html_impl(results, options),
        }
    }

    // ── Per-format public delegates ───────────────────────────────────────────

    /// Serialize to W3C SPARQL Results JSON.
    pub fn format_json(results: &SparqlResults) -> String {
        Self::format_json_impl(results)
    }

    /// Serialize to W3C SPARQL Results XML.
    pub fn format_xml(results: &SparqlResults) -> String {
        Self::format_xml_impl(results)
    }

    /// Serialize to RFC 4180 CSV.
    pub fn format_csv(results: &SparqlResults) -> String {
        Self::format_csv_impl(results, &FormatterOptions::default())
    }

    /// Serialize to W3C SPARQL Results TSV.
    pub fn format_tsv(results: &SparqlResults) -> String {
        Self::format_tsv_impl(results, &FormatterOptions::default())
    }

    /// Serialize to GitHub-Flavored Markdown table.
    pub fn format_markdown(results: &SparqlResults) -> String {
        Self::format_markdown_impl(results, &FormatterOptions::default())
    }

    /// Serialize to HTML `<table>`.
    pub fn format_html(results: &SparqlResults) -> String {
        Self::format_html_impl(results, &FormatterOptions::default())
    }

    /// Serialize an ASK boolean result to the given format.
    pub fn format_boolean(result: bool, format: OutputFormat) -> String {
        match format {
            OutputFormat::Json => {
                format!(r#"{{"head":{{}},"boolean":{}}}"#, result)
            }
            OutputFormat::Xml => {
                format!(
                    "<?xml version=\"1.0\"?>\n\
                     <sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\
                     <head/><boolean>{}</boolean></sparql>",
                    result
                )
            }
            OutputFormat::Csv | OutputFormat::Tsv => {
                format!("{}", result)
            }
            OutputFormat::Markdown => {
                format!("| result |\n|--------|\n| {} |", result)
            }
            OutputFormat::Html => {
                format!(
                    "<table><thead><tr><th>result</th></tr></thead>\
                     <tbody><tr><td>{}</td></tr></tbody></table>",
                    result
                )
            }
        }
    }

    // ── JSON ──────────────────────────────────────────────────────────────────

    fn format_json_impl(results: &SparqlResults) -> String {
        // Handle ASK
        if let Some(b) = results.boolean {
            return Self::format_boolean(b, OutputFormat::Json);
        }

        let vars_json: Vec<String> = results
            .variables
            .iter()
            .map(|v| format!("\"{}\"", json_escape(v)))
            .collect();

        let mut bindings_json = Vec::new();
        for row in &results.bindings {
            let mut fields = Vec::new();
            for var in &results.variables {
                if let Some(val) = row.get(var) {
                    if let Some(field_json) = sparql_value_to_json(var, val) {
                        fields.push(field_json);
                    }
                }
            }
            bindings_json.push(format!("{{{}}}", fields.join(",")));
        }

        format!(
            "{{\"head\":{{\"vars\":[{}]}},\"results\":{{\"bindings\":[{}]}}}}",
            vars_json.join(","),
            bindings_json.join(","),
        )
    }

    // ── XML ───────────────────────────────────────────────────────────────────

    fn format_xml_impl(results: &SparqlResults) -> String {
        if let Some(b) = results.boolean {
            return Self::format_boolean(b, OutputFormat::Xml);
        }

        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n");
        out.push_str("  <head>\n");
        for v in &results.variables {
            out.push_str(&format!("    <variable name=\"{}\"/>\n", xml_escape(v)));
        }
        out.push_str("  </head>\n");
        out.push_str("  <results>\n");
        for row in &results.bindings {
            out.push_str("    <result>\n");
            for var in &results.variables {
                if let Some(val) = row.get(var) {
                    if let Some(binding_xml) = sparql_value_to_xml(var, val) {
                        out.push_str("      ");
                        out.push_str(&binding_xml);
                        out.push('\n');
                    }
                }
            }
            out.push_str("    </result>\n");
        }
        out.push_str("  </results>\n");
        out.push_str("</sparql>");
        out
    }

    // ── CSV ───────────────────────────────────────────────────────────────────

    fn format_csv_impl(results: &SparqlResults, options: &FormatterOptions) -> String {
        if let Some(b) = results.boolean {
            return format!("{}", b);
        }

        let mut rows: Vec<String> = Vec::new();

        if options.include_header {
            let header: Vec<String> =
                results.variables.iter().map(|v| csv_field(v)).collect();
            rows.push(header.join(","));
        }

        for row in &results.bindings {
            let fields: Vec<String> = results
                .variables
                .iter()
                .map(|v| {
                    let val = row.get(v).unwrap_or(&SparqlValue::Unbound);
                    let raw = match val {
                        SparqlValue::Unbound => options.null_representation.clone(),
                        other => other.as_str().to_string(),
                    };
                    csv_field(&raw)
                })
                .collect();
            rows.push(fields.join(","));
        }

        rows.join("\r\n")
    }

    // ── TSV ───────────────────────────────────────────────────────────────────

    fn format_tsv_impl(results: &SparqlResults, options: &FormatterOptions) -> String {
        if let Some(b) = results.boolean {
            return format!("{}", b);
        }

        let mut rows: Vec<String> = Vec::new();

        if options.include_header {
            let header: Vec<String> = results
                .variables
                .iter()
                .map(|v| format!("?{}", v))
                .collect();
            rows.push(header.join("\t"));
        }

        for row in &results.bindings {
            let fields: Vec<String> = results
                .variables
                .iter()
                .map(|v| {
                    let val = row.get(v).unwrap_or(&SparqlValue::Unbound);
                    sparql_value_to_tsv(val, &options.null_representation)
                })
                .collect();
            rows.push(fields.join("\t"));
        }

        rows.join("\n")
    }

    // ── Markdown ──────────────────────────────────────────────────────────────

    fn format_markdown_impl(results: &SparqlResults, options: &FormatterOptions) -> String {
        if let Some(b) = results.boolean {
            return Self::format_boolean(b, OutputFormat::Markdown);
        }

        let mut out = String::new();

        if options.include_header {
            let header = results
                .variables
                .iter()
                .map(|v| format!(" {} ", md_escape(v)))
                .collect::<Vec<_>>()
                .join("|");
            out.push('|');
            out.push_str(&header);
            out.push_str("|\n");

            // Separator row
            let sep = results
                .variables
                .iter()
                .map(|_| "---")
                .collect::<Vec<_>>()
                .join("|");
            out.push('|');
            out.push_str(&sep);
            out.push_str("|\n");
        }

        for row in &results.bindings {
            let cells = results
                .variables
                .iter()
                .map(|v| {
                    let val = row.get(v).unwrap_or(&SparqlValue::Unbound);
                    let raw = match val {
                        SparqlValue::Unbound => options.null_representation.clone(),
                        other => other.as_str().to_string(),
                    };
                    format!(" {} ", md_escape(&raw))
                })
                .collect::<Vec<_>>()
                .join("|");
            out.push('|');
            out.push_str(&cells);
            out.push_str("|\n");
        }

        out
    }

    // ── HTML ──────────────────────────────────────────────────────────────────

    fn format_html_impl(results: &SparqlResults, options: &FormatterOptions) -> String {
        if let Some(b) = results.boolean {
            return Self::format_boolean(b, OutputFormat::Html);
        }

        let mut out = String::new();
        out.push_str("<table>\n");

        if options.include_header {
            out.push_str("  <thead>\n    <tr>");
            for v in &results.variables {
                out.push_str(&format!("<th>{}</th>", html_escape(v)));
            }
            out.push_str("</tr>\n  </thead>\n");
        }

        out.push_str("  <tbody>\n");
        for row in &results.bindings {
            out.push_str("    <tr>");
            for v in &results.variables {
                let val = row.get(v).unwrap_or(&SparqlValue::Unbound);
                let cell = match val {
                    SparqlValue::Unbound => options.null_representation.clone(),
                    other => other.as_str().to_string(),
                };
                out.push_str(&format!("<td>{}</td>", html_escape(&cell)));
            }
            out.push_str("</tr>\n");
        }
        out.push_str("  </tbody>\n");
        out.push_str("</table>");
        out
    }
}

// ── Escape helpers ─────────────────────────────────────────────────────────────

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn md_escape(s: &str) -> String {
    // Escape pipe characters inside Markdown table cells
    s.replace('|', "\\|")
}

/// RFC 4180 CSV field quoting: quote if the field contains comma, CRLF, or double-quote.
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn sparql_value_to_tsv(val: &SparqlValue, null_repr: &str) -> String {
    match val {
        SparqlValue::Uri(u) => format!("<{}>", u),
        SparqlValue::Literal { value, datatype, lang } => {
            let mut s = format!("\"{}\"", value.replace('"', "\\\""));
            if let Some(l) = lang {
                s.push('@');
                s.push_str(l);
            } else if let Some(dt) = datatype {
                s.push_str("^^<");
                s.push_str(dt);
                s.push('>');
            }
            s
        }
        SparqlValue::BNode(id) => format!("_:{}", id),
        SparqlValue::Unbound => null_repr.to_string(),
    }
}

fn sparql_value_to_json(var: &str, val: &SparqlValue) -> Option<String> {
    let inner = match val {
        SparqlValue::Unbound => return None,
        SparqlValue::Uri(u) => {
            format!("{{\"type\":\"uri\",\"value\":\"{}\"}}", json_escape(u))
        }
        SparqlValue::Literal { value, datatype, lang } => {
            let mut s = format!("{{\"type\":\"literal\",\"value\":\"{}\"", json_escape(value));
            if let Some(l) = lang {
                s.push_str(&format!(",\"xml:lang\":\"{}\"", json_escape(l)));
            } else if let Some(dt) = datatype {
                s.push_str(&format!(",\"datatype\":\"{}\"", json_escape(dt)));
            }
            s.push('}');
            s
        }
        SparqlValue::BNode(id) => {
            format!("{{\"type\":\"bnode\",\"value\":\"{}\"}}", json_escape(id))
        }
    };
    Some(format!("\"{}\":{}", json_escape(var), inner))
}

fn sparql_value_to_xml(var: &str, val: &SparqlValue) -> Option<String> {
    let inner = match val {
        SparqlValue::Unbound => return None,
        SparqlValue::Uri(u) => {
            format!("<binding name=\"{}\"><uri>{}</uri></binding>", xml_escape(var), xml_escape(u))
        }
        SparqlValue::Literal { value, datatype, lang } => {
            let mut attrs = String::new();
            if let Some(l) = lang {
                attrs.push_str(&format!(" xml:lang=\"{}\"", xml_escape(l)));
            } else if let Some(dt) = datatype {
                attrs.push_str(&format!(" datatype=\"{}\"", xml_escape(dt)));
            }
            format!(
                "<binding name=\"{}\"><literal{}>{}</literal></binding>",
                xml_escape(var),
                attrs,
                xml_escape(value)
            )
        }
        SparqlValue::BNode(id) => {
            format!(
                "<binding name=\"{}\"><bnode>{}</bnode></binding>",
                xml_escape(var),
                xml_escape(id)
            )
        }
    };
    Some(inner)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_results() -> SparqlResults {
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), SparqlValue::Literal {
            value: "Alice".to_string(),
            datatype: None,
            lang: Some("en".to_string()),
        });
        row1.insert("uri".to_string(), SparqlValue::Uri("http://example.org/alice".to_string()));

        let mut row2 = HashMap::new();
        row2.insert("name".to_string(), SparqlValue::Literal {
            value: "Bob".to_string(),
            datatype: None,
            lang: None,
        });
        row2.insert("uri".to_string(), SparqlValue::BNode("b0".to_string()));

        SparqlResults {
            variables: vec!["name".to_string(), "uri".to_string()],
            bindings: vec![row1, row2],
            boolean: None,
        }
    }

    fn empty_results() -> SparqlResults {
        SparqlResults {
            variables: vec!["x".to_string()],
            bindings: vec![],
            boolean: None,
        }
    }

    // ── JSON ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_json_contains_head_vars() {
        let json = SparqlResultFormatter::format_json(&simple_results());
        assert!(json.contains("\"vars\""), "json={json}");
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"uri\""));
    }

    #[test]
    fn test_json_contains_bindings() {
        let json = SparqlResultFormatter::format_json(&simple_results());
        assert!(json.contains("\"bindings\""));
        assert!(json.contains("Alice"));
    }

    #[test]
    fn test_json_uri_type() {
        let json = SparqlResultFormatter::format_json(&simple_results());
        assert!(json.contains("\"type\":\"uri\""));
    }

    #[test]
    fn test_json_literal_type() {
        let json = SparqlResultFormatter::format_json(&simple_results());
        assert!(json.contains("\"type\":\"literal\""));
    }

    #[test]
    fn test_json_bnode_type() {
        let json = SparqlResultFormatter::format_json(&simple_results());
        assert!(json.contains("\"type\":\"bnode\""));
    }

    #[test]
    fn test_json_boolean_true() {
        let json = SparqlResultFormatter::format_boolean(true, OutputFormat::Json);
        assert!(json.contains("true"));
        assert!(json.contains("boolean"));
    }

    #[test]
    fn test_json_boolean_false() {
        let json = SparqlResultFormatter::format_boolean(false, OutputFormat::Json);
        assert!(json.contains("false"));
    }

    #[test]
    fn test_json_empty_results() {
        let json = SparqlResultFormatter::format_json(&empty_results());
        assert!(json.contains("\"bindings\":[]"));
    }

    // ── XML ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_xml_has_sparql_root() {
        let xml = SparqlResultFormatter::format_xml(&simple_results());
        assert!(xml.contains("<sparql"));
        assert!(xml.contains("</sparql>"));
    }

    #[test]
    fn test_xml_has_variable_elements() {
        let xml = SparqlResultFormatter::format_xml(&simple_results());
        assert!(xml.contains("<variable name=\"name\""));
        assert!(xml.contains("<variable name=\"uri\""));
    }

    #[test]
    fn test_xml_has_result_elements() {
        let xml = SparqlResultFormatter::format_xml(&simple_results());
        assert!(xml.contains("<result>"));
        assert!(xml.contains("</result>"));
    }

    #[test]
    fn test_xml_has_uri_element() {
        let xml = SparqlResultFormatter::format_xml(&simple_results());
        assert!(xml.contains("<uri>"));
        assert!(xml.contains("</uri>"));
    }

    #[test]
    fn test_xml_has_literal_element() {
        let xml = SparqlResultFormatter::format_xml(&simple_results());
        assert!(xml.contains("<literal"));
    }

    #[test]
    fn test_xml_boolean_true() {
        let xml = SparqlResultFormatter::format_boolean(true, OutputFormat::Xml);
        assert!(xml.contains("<boolean>true</boolean>"));
    }

    #[test]
    fn test_xml_boolean_false() {
        let xml = SparqlResultFormatter::format_boolean(false, OutputFormat::Xml);
        assert!(xml.contains("<boolean>false</boolean>"));
    }

    #[test]
    fn test_xml_empty_results() {
        let xml = SparqlResultFormatter::format_xml(&empty_results());
        assert!(xml.contains("<results>"));
    }

    // ── CSV ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_header_row() {
        let csv = SparqlResultFormatter::format_csv(&simple_results());
        let first_line = csv.lines().next().unwrap_or("");
        assert!(first_line.contains("name"));
        assert!(first_line.contains("uri"));
    }

    #[test]
    fn test_csv_data_rows() {
        let csv = SparqlResultFormatter::format_csv(&simple_results());
        assert!(csv.contains("Alice"));
    }

    #[test]
    fn test_csv_no_header_option() {
        let opts = FormatterOptions { include_header: false, ..Default::default() };
        let csv = SparqlResultFormatter::format_with_options(
            &simple_results(), OutputFormat::Csv, &opts,
        );
        let first_line = csv.lines().next().unwrap_or("");
        assert!(!first_line.contains("name"));
    }

    #[test]
    fn test_csv_special_chars_quoted() {
        let mut row = HashMap::new();
        row.insert("x".to_string(), SparqlValue::Literal {
            value: "hello, world".to_string(),
            datatype: None,
            lang: None,
        });
        let r = SparqlResults {
            variables: vec!["x".to_string()],
            bindings: vec![row],
            boolean: None,
        };
        let csv = SparqlResultFormatter::format_csv(&r);
        assert!(csv.contains("\"hello, world\""), "csv={csv}");
    }

    #[test]
    fn test_csv_quote_in_value_escaped() {
        let mut row = HashMap::new();
        row.insert("x".to_string(), SparqlValue::Literal {
            value: "say \"hi\"".to_string(),
            datatype: None,
            lang: None,
        });
        let r = SparqlResults {
            variables: vec!["x".to_string()],
            bindings: vec![row],
            boolean: None,
        };
        let csv = SparqlResultFormatter::format_csv(&r);
        // RFC 4180 double-quoting
        assert!(csv.contains("\"\""), "csv={csv}");
    }

    // ── TSV ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_tsv_header_with_question_marks() {
        let tsv = SparqlResultFormatter::format_tsv(&simple_results());
        let first_line = tsv.lines().next().unwrap_or("");
        assert!(first_line.contains("?name"));
        assert!(first_line.contains("?uri"));
    }

    #[test]
    fn test_tsv_tab_separated() {
        let tsv = SparqlResultFormatter::format_tsv(&simple_results());
        let second_line = tsv.lines().nth(1).unwrap_or("");
        assert!(second_line.contains('\t'));
    }

    #[test]
    fn test_tsv_uri_wrapped_in_angle_brackets() {
        let tsv = SparqlResultFormatter::format_tsv(&simple_results());
        assert!(tsv.contains("<http://example.org/alice>"));
    }

    #[test]
    fn test_tsv_bnode_prefix() {
        let tsv = SparqlResultFormatter::format_tsv(&simple_results());
        assert!(tsv.contains("_:b0"));
    }

    #[test]
    fn test_tsv_literal_quoted() {
        let tsv = SparqlResultFormatter::format_tsv(&simple_results());
        assert!(tsv.contains("\"Alice\""));
    }

    // ── Markdown ──────────────────────────────────────────────────────────────

    #[test]
    fn test_markdown_pipe_separated() {
        let md = SparqlResultFormatter::format_markdown(&simple_results());
        assert!(md.contains('|'));
    }

    #[test]
    fn test_markdown_header_row() {
        let md = SparqlResultFormatter::format_markdown(&simple_results());
        assert!(md.contains("name"));
    }

    #[test]
    fn test_markdown_separator_row() {
        let md = SparqlResultFormatter::format_markdown(&simple_results());
        assert!(md.contains("---"));
    }

    #[test]
    fn test_markdown_data_present() {
        let md = SparqlResultFormatter::format_markdown(&simple_results());
        assert!(md.contains("Alice"));
    }

    #[test]
    fn test_markdown_boolean_table() {
        let md = SparqlResultFormatter::format_boolean(true, OutputFormat::Markdown);
        assert!(md.contains('|'));
        assert!(md.contains("true"));
    }

    // ── HTML ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_html_table_tag() {
        let html = SparqlResultFormatter::format_html(&simple_results());
        assert!(html.contains("<table>"));
        assert!(html.contains("</table>"));
    }

    #[test]
    fn test_html_thead_tag() {
        let html = SparqlResultFormatter::format_html(&simple_results());
        assert!(html.contains("<thead>"));
        assert!(html.contains("</thead>"));
    }

    #[test]
    fn test_html_th_tags() {
        let html = SparqlResultFormatter::format_html(&simple_results());
        assert!(html.contains("<th>name</th>"));
    }

    #[test]
    fn test_html_tbody_tag() {
        let html = SparqlResultFormatter::format_html(&simple_results());
        assert!(html.contains("<tbody>"));
    }

    #[test]
    fn test_html_td_data() {
        let html = SparqlResultFormatter::format_html(&simple_results());
        assert!(html.contains("<td>Alice</td>"));
    }

    #[test]
    fn test_html_boolean() {
        let html = SparqlResultFormatter::format_boolean(false, OutputFormat::Html);
        assert!(html.contains("<table>"));
        assert!(html.contains("false"));
    }

    #[test]
    fn test_html_xss_escaping() {
        let mut row = HashMap::new();
        row.insert("x".to_string(), SparqlValue::Literal {
            value: "<script>alert(1)</script>".to_string(),
            datatype: None,
            lang: None,
        });
        let r = SparqlResults {
            variables: vec!["x".to_string()],
            bindings: vec![row],
            boolean: None,
        };
        let html = SparqlResultFormatter::format_html(&r);
        assert!(!html.contains("<script>"), "html={html}");
        assert!(html.contains("&lt;script&gt;"));
    }

    // ── Unbound handling ──────────────────────────────────────────────────────

    #[test]
    fn test_unbound_json_omitted() {
        let mut row = HashMap::new();
        row.insert("x".to_string(), SparqlValue::Unbound);
        row.insert("y".to_string(), SparqlValue::Uri("http://a".to_string()));
        let r = SparqlResults {
            variables: vec!["x".to_string(), "y".to_string()],
            bindings: vec![row],
            boolean: None,
        };
        let json = SparqlResultFormatter::format_json(&r);
        // Unbound 'x' should not appear in the JSON binding
        assert!(!json.contains("\"x\":{"), "json={json}");
    }

    #[test]
    fn test_unbound_csv_uses_null_repr() {
        let mut row = HashMap::new();
        row.insert("x".to_string(), SparqlValue::Unbound);
        let r = SparqlResults {
            variables: vec!["x".to_string()],
            bindings: vec![row],
            boolean: None,
        };
        let opts = FormatterOptions {
            include_header: false,
            null_representation: "NULL".to_string(),
        };
        let csv = SparqlResultFormatter::format_with_options(&r, OutputFormat::Csv, &opts);
        assert_eq!(csv.trim(), "NULL");
    }

    // ── format() dispatcher ───────────────────────────────────────────────────

    #[test]
    fn test_format_dispatches_json() {
        let out = SparqlResultFormatter::format(&simple_results(), OutputFormat::Json);
        assert!(out.contains("\"vars\""));
    }

    #[test]
    fn test_format_dispatches_xml() {
        let out = SparqlResultFormatter::format(&simple_results(), OutputFormat::Xml);
        assert!(out.contains("<sparql"));
    }

    #[test]
    fn test_format_dispatches_csv() {
        let out = SparqlResultFormatter::format(&simple_results(), OutputFormat::Csv);
        assert!(out.contains(','));
    }

    #[test]
    fn test_format_dispatches_tsv() {
        let out = SparqlResultFormatter::format(&simple_results(), OutputFormat::Tsv);
        assert!(out.contains('\t'));
    }

    #[test]
    fn test_format_dispatches_markdown() {
        let out = SparqlResultFormatter::format(&simple_results(), OutputFormat::Markdown);
        assert!(out.contains('|'));
    }

    #[test]
    fn test_format_dispatches_html() {
        let out = SparqlResultFormatter::format(&simple_results(), OutputFormat::Html);
        assert!(out.contains("<table>"));
    }
}
