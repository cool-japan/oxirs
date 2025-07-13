//! Content type handling for SPARQL responses
//!
//! This module handles content negotiation and response formatting
//! for SPARQL query results and error responses.

use axum::http::HeaderMap;
use serde::Serialize;

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
        assert!(json_result.unwrap().contains("John"));

        // Test XML formatting (placeholder)
        let xml_result = ResponseFormatter::format(&data, SPARQL_RESULTS_XML);
        assert!(xml_result.is_ok());
        assert!(xml_result.unwrap().contains("sparql"));
    }
}
