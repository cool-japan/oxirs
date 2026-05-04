//! Content type handling for SPARQL responses
//!
//! This module handles content negotiation and response formatting
//! for SPARQL query results and error responses.

use axum::http::HeaderMap;
use serde::Serialize;

use crate::handlers::sparql::core::QueryResult;

/// Content type constants for SPARQL responses
pub const SPARQL_RESULTS_JSON: &str = "application/sparql-results+json";
pub const SPARQL_RESULTS_XML: &str = "application/sparql-results+xml";
pub const SPARQL_RESULTS_CSV: &str = "text/csv";
pub const SPARQL_RESULTS_TSV: &str = "text/tab-separated-values";
pub const APPLICATION_JSON: &str = "application/json";
pub const APPLICATION_XML: &str = "application/xml";
pub const TEXT_PLAIN: &str = "text/plain";
pub const TEXT_HTML: &str = "text/html";
pub const APPLICATION_RDF_XML: &str = "application/rdf+xml";
pub const TEXT_TURTLE: &str = "text/turtle";
pub const APPLICATION_NTRIPLES: &str = "application/n-triples";
pub const APPLICATION_JSONLD: &str = "application/ld+json";

/// Content negotiation utilities
#[derive(Debug, Clone)]
pub struct ContentNegotiator {
    supported_types: Vec<String>,
    default_type: String,
}

impl Default for ContentNegotiator {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentNegotiator {
    pub fn new() -> Self {
        Self {
            supported_types: vec![
                SPARQL_RESULTS_JSON.to_string(),
                SPARQL_RESULTS_XML.to_string(),
                SPARQL_RESULTS_CSV.to_string(),
                SPARQL_RESULTS_TSV.to_string(),
                APPLICATION_JSON.to_string(),
                TEXT_TURTLE.to_string(),
                APPLICATION_RDF_XML.to_string(),
                APPLICATION_NTRIPLES.to_string(),
                APPLICATION_JSONLD.to_string(),
            ],
            default_type: SPARQL_RESULTS_JSON.to_string(),
        }
    }

    /// Negotiate content type based on Accept header
    pub fn negotiate(&self, headers: &HeaderMap) -> String {
        if let Some(accept_header) = headers.get("accept") {
            if let Ok(accept_str) = accept_header.to_str() {
                return self.parse_accept_header(accept_str);
            }
        }
        self.default_type.clone()
    }

    /// Parse Accept header and return best matching content type
    fn parse_accept_header(&self, accept: &str) -> String {
        let mut types = Vec::new();

        for part in accept.split(',') {
            let part = part.trim();
            let (media_type, quality) = if let Some(q_pos) = part.find(";q=") {
                let media_type = part[..q_pos].trim();
                let quality = part[q_pos + 3..].parse::<f32>().unwrap_or(1.0);
                (media_type, quality)
            } else {
                (part, 1.0)
            };

            types.push((media_type.to_string(), quality));
        }

        // Sort by quality (highest first)
        types.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find first supported type
        for (media_type, _) in types {
            if self.supported_types.contains(&media_type) {
                return media_type;
            }

            // Handle wildcards
            if media_type == "*/*" || media_type == "application/*" {
                return self.default_type.clone();
            }
        }

        self.default_type.clone()
    }

    /// Check if content type is supported
    pub fn is_supported(&self, content_type: &str) -> bool {
        self.supported_types.contains(&content_type.to_string())
    }
}

/// Response formatter for different content types
#[derive(Debug)]
pub struct ResponseFormatter;

impl ResponseFormatter {
    /// Format response data based on content type
    pub fn format<T: Serialize + std::fmt::Debug>(
        data: &T,
        content_type: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match content_type {
            SPARQL_RESULTS_JSON | APPLICATION_JSON => Ok(serde_json::to_string_pretty(data)?),
            SPARQL_RESULTS_XML | APPLICATION_XML => Self::format_sparql_xml(data),
            SPARQL_RESULTS_CSV => Self::format_sparql_csv(data),
            SPARQL_RESULTS_TSV => Self::format_sparql_tsv(data),
            TEXT_TURTLE => Self::format_turtle(data),
            APPLICATION_RDF_XML => Self::format_rdf_xml(data),
            APPLICATION_NTRIPLES => Self::format_ntriples(data),
            APPLICATION_JSONLD => Self::format_jsonld(data),
            _ => {
                // Default to JSON
                Ok(serde_json::to_string_pretty(data)?)
            }
        }
    }

    /// Format SPARQL results as XML according to W3C specification
    fn format_sparql_xml<T: Serialize + std::fmt::Debug>(
        data: &T,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // First serialize to JSON to extract structured data
        let json_str = serde_json::to_string(data)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\"?>\n");
        xml.push_str("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n");

        // Handle different query types
        if let Some(query_type) = parsed.get("query_type").and_then(|v| v.as_str()) {
            match query_type {
                "ASK" => {
                    if let Some(boolean) = parsed.get("boolean").and_then(|v| v.as_bool()) {
                        xml.push_str(&format!("  <boolean>{}</boolean>\n", boolean));
                    }
                }
                "SELECT" => {
                    xml.push_str("  <head>\n");

                    // Extract variable names from first binding if available
                    if let Some(bindings) = parsed.get("bindings").and_then(|v| v.as_array()) {
                        if let Some(first_binding) = bindings.first().and_then(|v| v.as_object()) {
                            for var_name in first_binding.keys() {
                                xml.push_str(&format!("    <variable name=\"{}\"/>\n", var_name));
                            }
                        }
                    }

                    xml.push_str("  </head>\n");
                    xml.push_str("  <results>\n");

                    if let Some(bindings) = parsed.get("bindings").and_then(|v| v.as_array()) {
                        for binding in bindings {
                            if let Some(binding_obj) = binding.as_object() {
                                xml.push_str("    <result>\n");
                                for (var, value) in binding_obj {
                                    xml.push_str(&format!("      <binding name=\"{}\">\n", var));

                                    // Format based on value type (simplified)
                                    if let Some(str_val) = value.as_str() {
                                        if str_val.starts_with("http://")
                                            || str_val.starts_with("https://")
                                        {
                                            xml.push_str(&format!(
                                                "        <uri>{}</uri>\n",
                                                Self::xml_escape(str_val)
                                            ));
                                        } else {
                                            xml.push_str(&format!(
                                                "        <literal>{}</literal>\n",
                                                Self::xml_escape(str_val)
                                            ));
                                        }
                                    } else {
                                        xml.push_str(&format!(
                                            "        <literal>{}</literal>\n",
                                            Self::xml_escape(&value.to_string())
                                        ));
                                    }

                                    xml.push_str("      </binding>\n");
                                }
                                xml.push_str("    </result>\n");
                            }
                        }
                    }

                    xml.push_str("  </results>\n");
                }
                _ => {
                    xml.push_str("  <!-- Unsupported query type for XML format -->\n");
                }
            }
        }

        xml.push_str("</sparql>\n");
        Ok(xml)
    }

    /// Format SPARQL results as CSV
    fn format_sparql_csv<T: Serialize + std::fmt::Debug>(
        data: &T,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let json_str = serde_json::to_string(data)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut csv = String::new();

        if let Some(query_type) = parsed.get("query_type").and_then(|v| v.as_str()) {
            match query_type {
                "ASK" => {
                    csv.push_str("result\n");
                    if let Some(boolean) = parsed.get("boolean").and_then(|v| v.as_bool()) {
                        csv.push_str(&format!("{}\n", boolean));
                    }
                }
                "SELECT" => {
                    if let Some(bindings) = parsed.get("bindings").and_then(|v| v.as_array()) {
                        // Extract headers from first result
                        if let Some(first_binding) = bindings.first().and_then(|v| v.as_object()) {
                            let headers: Vec<&String> = first_binding.keys().collect();
                            csv.push_str(
                                &headers
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join(","),
                            );
                            csv.push('\n');

                            // Write data rows
                            for binding in bindings {
                                if let Some(binding_obj) = binding.as_object() {
                                    let values: Vec<String> = headers
                                        .iter()
                                        .map(|header| {
                                            binding_obj
                                                .get(*header)
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("")
                                                .to_string()
                                        })
                                        .collect();
                                    csv.push_str(&values.join(","));
                                    csv.push('\n');
                                }
                            }
                        }
                    }
                }
                _ => {
                    csv.push_str("# Unsupported query type for CSV format\n");
                }
            }
        }

        Ok(csv)
    }

    /// Format SPARQL results as TSV (Tab-Separated Values)
    fn format_sparql_tsv<T: Serialize + std::fmt::Debug>(
        data: &T,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let csv_result = Self::format_sparql_csv(data)?;
        // Convert CSV to TSV by replacing commas with tabs
        Ok(csv_result.replace(',', "\t"))
    }

    /// Format RDF data as Turtle
    fn format_turtle<T: Serialize + std::fmt::Debug>(
        data: &T,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let json_str = serde_json::to_string(data)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut turtle = String::new();
        turtle.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        turtle.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n");

        // Handle CONSTRUCT/DESCRIBE results
        if let Some(construct_graph) = parsed.get("construct_graph").and_then(|v| v.as_str()) {
            turtle.push_str(construct_graph);
        } else if let Some(describe_graph) = parsed.get("describe_graph").and_then(|v| v.as_str()) {
            turtle.push_str(describe_graph);
        } else {
            turtle.push_str("# No graph data available for Turtle serialization\n");
        }

        Ok(turtle)
    }

    /// Format RDF data as RDF/XML
    fn format_rdf_xml<T: Serialize + std::fmt::Debug>(
        data: &T,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let json_str = serde_json::to_string(data)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut rdf_xml = String::new();
        rdf_xml.push_str("<?xml version=\"1.0\"?>\n");
        rdf_xml.push_str("<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"\n");
        rdf_xml.push_str("         xmlns:rdfs=\"http://www.w3.org/2000/01/rdf-schema#\">\n");

        // Handle CONSTRUCT/DESCRIBE results
        if let Some(construct_graph) = parsed.get("construct_graph").and_then(|v| v.as_str()) {
            rdf_xml.push_str("  <!-- CONSTRUCT result -->\n");
            rdf_xml.push_str(&format!(
                "  <!-- {} -->\n",
                Self::xml_escape(construct_graph)
            ));
        } else if let Some(describe_graph) = parsed.get("describe_graph").and_then(|v| v.as_str()) {
            rdf_xml.push_str("  <!-- DESCRIBE result -->\n");
            rdf_xml.push_str(&format!(
                "  <!-- {} -->\n",
                Self::xml_escape(describe_graph)
            ));
        } else {
            rdf_xml.push_str("  <!-- No graph data available -->\n");
        }

        rdf_xml.push_str("</rdf:RDF>\n");
        Ok(rdf_xml)
    }

    /// Format RDF data as N-Triples
    fn format_ntriples<T: Serialize + std::fmt::Debug>(
        data: &T,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let json_str = serde_json::to_string(data)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut ntriples = String::new();

        // Handle CONSTRUCT/DESCRIBE results
        if let Some(construct_graph) = parsed.get("construct_graph").and_then(|v| v.as_str()) {
            ntriples.push_str(construct_graph);
        } else if let Some(describe_graph) = parsed.get("describe_graph").and_then(|v| v.as_str()) {
            ntriples.push_str(describe_graph);
        } else {
            ntriples.push_str("# No graph data available for N-Triples serialization\n");
        }

        Ok(ntriples)
    }

    /// Format as JSON-LD
    fn format_jsonld<T: Serialize + std::fmt::Debug>(
        data: &T,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let json_str = serde_json::to_string(data)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

        let mut jsonld = serde_json::Map::new();
        jsonld.insert(
            "@context".to_string(),
            serde_json::json!({
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            }),
        );

        // Handle CONSTRUCT/DESCRIBE results
        if let Some(construct_graph) = parsed.get("construct_graph") {
            jsonld.insert("@graph".to_string(), construct_graph.clone());
        } else if let Some(describe_graph) = parsed.get("describe_graph") {
            jsonld.insert("@graph".to_string(), describe_graph.clone());
        } else {
            jsonld.insert("@graph".to_string(), serde_json::json!([]));
        }

        Ok(serde_json::to_string_pretty(&jsonld)?)
    }

    /// Escape XML special characters
    fn xml_escape(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Get appropriate Content-Type header for response
    pub fn get_content_type_header(content_type: &str) -> (&'static str, &str) {
        ("Content-Type", content_type)
    }
}

// ----------------------------------------------------------------------------
// Spec-correct SPARQL Query Results serializers
// ----------------------------------------------------------------------------
//
// These free-standing helpers serialize a QueryResult into the W3C-defined
// SPARQL Query Results formats. They live here (not in core.rs) so the
// formatting logic can be unit-tested in isolation.

/// Serialize a QueryResult to W3C SPARQL Query Results XML.
///
/// See <https://www.w3.org/TR/rdf-sparql-XMLres/>.
pub fn sparql_results_xml(result: &QueryResult) -> String {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n");

    match result.query_type.as_str() {
        "ASK" => {
            xml.push_str("  <head/>\n");
            let value = result.boolean.unwrap_or(false);
            xml.push_str(&format!("  <boolean>{value}</boolean>\n"));
        }
        "SELECT" => {
            let bindings = result.bindings.as_deref().unwrap_or(&[]);
            // Variables: ordered set of keys collected across all bindings
            let mut vars: Vec<String> = Vec::new();
            for binding in bindings {
                for k in binding.keys() {
                    if !vars.iter().any(|existing| existing == k) {
                        vars.push(k.clone());
                    }
                }
            }
            xml.push_str("  <head>\n");
            for v in &vars {
                xml.push_str(&format!("    <variable name=\"{}\"/>\n", xml_escape(v)));
            }
            xml.push_str("  </head>\n");
            xml.push_str("  <results>\n");
            for binding in bindings {
                xml.push_str("    <result>\n");
                for v in &vars {
                    if let Some(value) = binding.get(v) {
                        xml.push_str(&format!("      <binding name=\"{}\">\n", xml_escape(v)));
                        xml.push_str(&serialize_term_xml(value));
                        xml.push_str("      </binding>\n");
                    }
                }
                xml.push_str("    </result>\n");
            }
            xml.push_str("  </results>\n");
        }
        _ => {
            // For CONSTRUCT/DESCRIBE the SPARQL Results XML format does
            // not strictly apply, but emit a well-formed empty results
            // document so clients can detect "no rows".
            xml.push_str("  <head/>\n");
            xml.push_str("  <results/>\n");
        }
    }

    xml.push_str("</sparql>\n");
    xml
}

/// Serialize a QueryResult to W3C SPARQL Query Results CSV.
///
/// See <https://www.w3.org/TR/sparql11-results-csv-tsv/>.
pub fn sparql_results_csv(result: &QueryResult) -> String {
    sparql_results_separated(result, ',', false)
}

/// Serialize a QueryResult to W3C SPARQL Query Results TSV.
///
/// See <https://www.w3.org/TR/sparql11-results-csv-tsv/>.
pub fn sparql_results_tsv(result: &QueryResult) -> String {
    sparql_results_separated(result, '\t', true)
}

/// Build a separated-value SPARQL results document.
///
/// Set `tsv = true` for the W3C TSV variant: variables are prefixed with
/// `?`, fields use a tab separator, lines are terminated with `\n`, and
/// string literals are written using the N-Triples lexical form. The CSV
/// variant uses the comma separator, `\r\n` line terminators, and the
/// bare lexical form with RFC 4180 quoting.
fn sparql_results_separated(result: &QueryResult, sep: char, tsv: bool) -> String {
    let line_end = if tsv { "\n" } else { "\r\n" };
    let mut out = String::new();

    match result.query_type.as_str() {
        "ASK" => {
            // Per spec ASK is not directly supported, but Jena emits a
            // single-column header `result` with `true`/`false`.
            if tsv {
                out.push_str("?result");
            } else {
                out.push_str("result");
            }
            out.push_str(line_end);
            let value = result.boolean.unwrap_or(false);
            out.push_str(if value { "true" } else { "false" });
            out.push_str(line_end);
        }
        "SELECT" => {
            let bindings = result.bindings.as_deref().unwrap_or(&[]);
            let mut vars: Vec<String> = Vec::new();
            for binding in bindings {
                for k in binding.keys() {
                    if !vars.iter().any(|existing| existing == k) {
                        vars.push(k.clone());
                    }
                }
            }
            // Header
            for (i, v) in vars.iter().enumerate() {
                if i > 0 {
                    out.push(sep);
                }
                if tsv {
                    out.push('?');
                    out.push_str(v);
                } else {
                    out.push_str(&csv_escape(v));
                }
            }
            out.push_str(line_end);
            // Rows
            for binding in bindings {
                for (i, v) in vars.iter().enumerate() {
                    if i > 0 {
                        out.push(sep);
                    }
                    if let Some(value) = binding.get(v) {
                        if tsv {
                            out.push_str(&serialize_term_ntriples(value));
                        } else {
                            out.push_str(&serialize_term_csv(value));
                        }
                    }
                }
                out.push_str(line_end);
            }
        }
        _ => {
            // CONSTRUCT/DESCRIBE: write empty header per spec
        }
    }

    out
}

/// Serialize a SPARQL term (in W3C JSON binding shape) as XML for use
/// inside a `<binding>` element.
fn serialize_term_xml(value: &serde_json::Value) -> String {
    let object = value.as_object();
    let term_type = object
        .and_then(|o| o.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("literal");
    let term_value = object
        .and_then(|o| o.get("value"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match term_type {
        "uri" => format!("        <uri>{}</uri>\n", xml_escape(term_value)),
        "bnode" => format!("        <bnode>{}</bnode>\n", xml_escape(term_value)),
        _ => {
            let mut attrs = String::new();
            if let Some(lang) = object
                .and_then(|o| o.get("xml:lang"))
                .and_then(|v| v.as_str())
            {
                attrs.push_str(&format!(" xml:lang=\"{}\"", xml_escape(lang)));
            }
            if let Some(dt) = object
                .and_then(|o| o.get("datatype"))
                .and_then(|v| v.as_str())
            {
                attrs.push_str(&format!(" datatype=\"{}\"", xml_escape(dt)));
            }
            format!(
                "        <literal{}>{}</literal>\n",
                attrs,
                xml_escape(term_value)
            )
        }
    }
}

/// Serialize a SPARQL term in N-Triples lexical form, for the TSV
/// SPARQL Query Results format.
fn serialize_term_ntriples(value: &serde_json::Value) -> String {
    let object = value.as_object();
    let term_type = object
        .and_then(|o| o.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("literal");
    let term_value = object
        .and_then(|o| o.get("value"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match term_type {
        "uri" => format!("<{}>", term_value),
        "bnode" => format!("_:{}", term_value),
        _ => {
            let mut out = format!("\"{}\"", ntriples_escape(term_value));
            if let Some(lang) = object
                .and_then(|o| o.get("xml:lang"))
                .and_then(|v| v.as_str())
            {
                out.push('@');
                out.push_str(lang);
            } else if let Some(dt) = object
                .and_then(|o| o.get("datatype"))
                .and_then(|v| v.as_str())
            {
                out.push_str("^^<");
                out.push_str(dt);
                out.push('>');
            }
            out
        }
    }
}

/// Serialize a SPARQL term in CSV cell form per the SPARQL CSV spec.
///
/// IRIs and blank nodes are written as their string content (Jena
/// behavior). Literals use the lexical value with CSV quoting if it
/// contains commas, quotes, CR, or LF.
fn serialize_term_csv(value: &serde_json::Value) -> String {
    let object = value.as_object();
    let term_type = object
        .and_then(|o| o.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("literal");
    let term_value = object
        .and_then(|o| o.get("value"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match term_type {
        "uri" => csv_escape(term_value),
        "bnode" => csv_escape(&format!("_:{term_value}")),
        _ => csv_escape(term_value),
    }
}

/// Apply CSV quoting per RFC 4180.
fn csv_escape(s: &str) -> String {
    let needs_quoting = s
        .chars()
        .any(|c| c == ',' || c == '"' || c == '\r' || c == '\n');
    if !needs_quoting {
        return s.to_string();
    }
    let escaped = s.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

/// Apply N-Triples literal escaping (control chars, backslash, quote).
fn ntriples_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Apply XML 1.0 escaping for character data and double-quoted attribute values.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Convert an inline CONSTRUCT/DESCRIBE Turtle/N-Triples body into an
/// RDF/XML document. This is a minimal best-effort wrapper: real
/// Turtle parsing is not done here. The body is embedded as a comment
/// alongside an empty `rdf:RDF` envelope.
pub fn rdf_graph_to_rdfxml(body: &str) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n");
    if !body.trim().is_empty() {
        out.push_str("  <!-- Original Turtle/N-Triples graph follows. -->\n");
        out.push_str("  <!-- ");
        out.push_str(&xml_escape(body));
        out.push_str(" -->\n");
    }
    out.push_str("</rdf:RDF>\n");
    out
}

/// Convert an inline CONSTRUCT/DESCRIBE Turtle/N-Triples body into a
/// minimal JSON-LD document with the original graph kept under a
/// `comment` field.
pub fn rdf_graph_to_jsonld(body: &str) -> String {
    let value = serde_json::json!({
        "@context": {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        },
        "@graph": [],
        "comment": body,
    });
    serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    #[test]
    fn test_content_negotiation() {
        let negotiator = ContentNegotiator::new();

        // Test JSON preference
        let mut headers = HeaderMap::new();
        headers.insert("accept", "application/json".parse().unwrap());
        assert_eq!(negotiator.negotiate(&headers), APPLICATION_JSON);

        // Test SPARQL JSON preference
        headers.clear();
        headers.insert("accept", SPARQL_RESULTS_JSON.parse().unwrap());
        assert_eq!(negotiator.negotiate(&headers), SPARQL_RESULTS_JSON);

        // Test wildcard
        headers.clear();
        headers.insert("accept", "*/*".parse().unwrap());
        assert_eq!(negotiator.negotiate(&headers), SPARQL_RESULTS_JSON);

        // Test quality values
        headers.clear();
        headers.insert(
            "accept",
            "application/xml;q=0.8,application/json;q=0.9"
                .parse()
                .unwrap(),
        );
        assert_eq!(negotiator.negotiate(&headers), APPLICATION_JSON);
    }

    #[test]
    fn test_accept_header_parsing() {
        let negotiator = ContentNegotiator::new();

        // Complex Accept header
        let complex_accept = "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.1";
        let result = negotiator.parse_accept_header(complex_accept);
        assert_eq!(result, APPLICATION_JSON);
    }

    #[test]
    fn test_response_formatting() {
        use serde_json::json;

        let data = json!({
            "head": {
                "vars": ["name", "age"]
            },
            "results": {
                "bindings": [
                    {
                        "name": {"type": "literal", "value": "John"},
                        "age": {"type": "literal", "value": "30"}
                    }
                ]
            }
        });

        // Test JSON formatting
        let json_result = ResponseFormatter::format(&data, SPARQL_RESULTS_JSON);
        assert!(json_result.is_ok());
        assert!(json_result.expect("json").contains("John"));

        // Test XML formatting (placeholder)
        let xml_result = ResponseFormatter::format(&data, SPARQL_RESULTS_XML);
        assert!(xml_result.is_ok());
        assert!(xml_result.expect("xml").contains("sparql"));
    }

    fn empty_select_result() -> QueryResult {
        QueryResult {
            query_type: "SELECT".to_string(),
            execution_time_ms: 0,
            result_count: Some(0),
            bindings: Some(Vec::new()),
            boolean: None,
            construct_graph: None,
            describe_graph: None,
        }
    }

    fn one_row_select_result() -> QueryResult {
        let mut binding: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();
        binding.insert(
            "s".to_string(),
            serde_json::json!({"type": "uri", "value": "http://example.org/a"}),
        );
        binding.insert(
            "lit".to_string(),
            serde_json::json!({
                "type": "literal",
                "value": "He said \"hi\"\nLine 2",
                "datatype": "http://www.w3.org/2001/XMLSchema#string",
            }),
        );
        QueryResult {
            query_type: "SELECT".to_string(),
            execution_time_ms: 0,
            result_count: Some(1),
            bindings: Some(vec![binding]),
            boolean: None,
            construct_graph: None,
            describe_graph: None,
        }
    }

    fn ask_result(b: bool) -> QueryResult {
        QueryResult {
            query_type: "ASK".to_string(),
            execution_time_ms: 0,
            result_count: Some(1),
            bindings: None,
            boolean: Some(b),
            construct_graph: None,
            describe_graph: None,
        }
    }

    #[test]
    fn sparql_results_xml_empty_select() {
        let xml = sparql_results_xml(&empty_select_result());
        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">"));
        assert!(xml.contains("<head>"));
        assert!(xml.contains("</head>"));
        assert!(xml.contains("<results>"));
        assert!(xml.contains("</results>"));
        assert!(!xml.contains("<binding"));
    }

    #[test]
    fn sparql_results_xml_ask() {
        let xml_true = sparql_results_xml(&ask_result(true));
        assert!(xml_true.contains("<boolean>true</boolean>"));
        let xml_false = sparql_results_xml(&ask_result(false));
        assert!(xml_false.contains("<boolean>false</boolean>"));
    }

    #[test]
    fn sparql_results_xml_with_uri_and_literal() {
        let xml = sparql_results_xml(&one_row_select_result());
        assert!(xml.contains("<uri>http://example.org/a</uri>"));
        // datatype attribute appears
        assert!(xml.contains("datatype=\"http://www.w3.org/2001/XMLSchema#string\""));
        // XML escaping of quotes inside literal
        assert!(xml.contains("&quot;hi&quot;"));
    }

    #[test]
    fn sparql_results_csv_uses_crlf_terminator() {
        let csv = sparql_results_csv(&one_row_select_result());
        assert!(csv.contains("\r\n"), "CSV must use CRLF line terminators");
        // header includes both vars (order may vary but both present)
        assert!(csv.contains('s'));
        assert!(csv.contains("lit"));
        // CSV quoting on the literal that contains \" and \n
        assert!(csv.contains("\"\""), "embedded quotes are doubled");
    }

    #[test]
    fn sparql_results_tsv_uses_lf_terminator_and_question_prefix() {
        let tsv = sparql_results_tsv(&one_row_select_result());
        // TSV header is `?s\t?lit\n` (or reverse order)
        assert!(tsv.contains("?s") || tsv.contains("?lit"));
        // No CR
        assert!(!tsv.contains('\r'));
        // URIs as <iri>
        assert!(tsv.contains("<http://example.org/a>"));
    }

    #[test]
    fn sparql_results_csv_ask_emits_header() {
        let csv = sparql_results_csv(&ask_result(true));
        assert!(csv.starts_with("result\r\n"));
        assert!(csv.contains("true"));
    }

    #[test]
    fn sparql_results_tsv_ask_uses_question_prefix() {
        let tsv = sparql_results_tsv(&ask_result(false));
        assert!(tsv.starts_with("?result\n"));
        assert!(tsv.contains("false"));
    }

    #[test]
    fn xml_escape_handles_special_chars() {
        let escaped = xml_escape("a<b & c>\"d'e");
        assert_eq!(escaped, "a&lt;b &amp; c&gt;&quot;d&apos;e");
    }

    #[test]
    fn csv_escape_quotes_only_when_needed() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
        assert_eq!(csv_escape("a\nb"), "\"a\nb\"");
    }

    #[test]
    fn ntriples_escape_handles_control_chars() {
        let s = ntriples_escape("a\nb\tc\\d\"e");
        assert_eq!(s, "a\\nb\\tc\\\\d\\\"e");
    }

    #[test]
    fn rdf_graph_to_jsonld_emits_context() {
        let body = "<http://ex/s> <http://ex/p> \"v\" .";
        let jsonld = rdf_graph_to_jsonld(body);
        let value: serde_json::Value = serde_json::from_str(&jsonld).expect("jsonld parses");
        assert!(value.get("@context").is_some());
        assert!(value.get("@graph").is_some());
    }
}
