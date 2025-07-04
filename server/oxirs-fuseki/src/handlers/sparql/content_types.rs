//! Content type handling for SPARQL responses
//!
//! This module handles content negotiation and response formatting
//! for SPARQL query results and error responses.

use axum::http::HeaderMap;
use serde::Serialize;

/// Content type constants for SPARQL responses
pub mod content_types {
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
}

/// Content negotiation utilities
#[derive(Debug, Clone)]
pub struct ContentNegotiator {
    supported_types: Vec<String>,
    default_type: String,
}

impl ContentNegotiator {
    pub fn new() -> Self {
        Self {
            supported_types: vec![
                content_types::SPARQL_RESULTS_JSON.to_string(),
                content_types::SPARQL_RESULTS_XML.to_string(),
                content_types::SPARQL_RESULTS_CSV.to_string(),
                content_types::SPARQL_RESULTS_TSV.to_string(),
                content_types::APPLICATION_JSON.to_string(),
                content_types::TEXT_TURTLE.to_string(),
                content_types::APPLICATION_RDF_XML.to_string(),
                content_types::APPLICATION_NTRIPLES.to_string(),
                content_types::APPLICATION_JSONLD.to_string(),
            ],
            default_type: content_types::SPARQL_RESULTS_JSON.to_string(),
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
            content_types::SPARQL_RESULTS_JSON | content_types::APPLICATION_JSON => {
                Ok(serde_json::to_string_pretty(data)?)
            }
            content_types::SPARQL_RESULTS_XML | content_types::APPLICATION_XML => {
                // TODO: Implement XML formatting
                Ok(format!("<?xml version=\"1.0\"?>\n<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n  <!-- XML formatting not yet implemented -->\n  <results>{:?}</results>\n</sparql>", data))
            }
            content_types::SPARQL_RESULTS_CSV => {
                // TODO: Implement CSV formatting
                Ok("CSV formatting not yet implemented".to_string())
            }
            content_types::SPARQL_RESULTS_TSV => {
                // TODO: Implement TSV formatting
                Ok("TSV formatting not yet implemented".to_string())
            }
            content_types::TEXT_TURTLE => {
                // TODO: Implement Turtle formatting
                Ok("# Turtle formatting not yet implemented".to_string())
            }
            content_types::APPLICATION_RDF_XML => {
                // TODO: Implement RDF/XML formatting
                Ok(
                    "<?xml version=\"1.0\"?>\n<!-- RDF/XML formatting not yet implemented -->"
                        .to_string(),
                )
            }
            content_types::APPLICATION_NTRIPLES => {
                // TODO: Implement N-Triples formatting
                Ok("# N-Triples formatting not yet implemented".to_string())
            }
            content_types::APPLICATION_JSONLD => {
                // TODO: Implement JSON-LD formatting
                Ok("{}".to_string())
            }
            _ => {
                // Default to JSON
                Ok(serde_json::to_string_pretty(data)?)
            }
        }
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
        assert_eq!(
            negotiator.negotiate(&headers),
            content_types::APPLICATION_JSON
        );

        // Test SPARQL JSON preference
        headers.clear();
        headers.insert(
            "accept",
            content_types::SPARQL_RESULTS_JSON.parse().unwrap(),
        );
        assert_eq!(
            negotiator.negotiate(&headers),
            content_types::SPARQL_RESULTS_JSON
        );

        // Test wildcard
        headers.clear();
        headers.insert("accept", "*/*".parse().unwrap());
        assert_eq!(
            negotiator.negotiate(&headers),
            content_types::SPARQL_RESULTS_JSON
        );

        // Test quality values
        headers.clear();
        headers.insert(
            "accept",
            "application/xml;q=0.8,application/json;q=0.9"
                .parse()
                .unwrap(),
        );
        assert_eq!(
            negotiator.negotiate(&headers),
            content_types::APPLICATION_JSON
        );
    }

    #[test]
    fn test_accept_header_parsing() {
        let negotiator = ContentNegotiator::new();

        // Complex Accept header
        let complex_accept = "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.1";
        let result = negotiator.parse_accept_header(complex_accept);
        assert_eq!(result, content_types::APPLICATION_JSON);
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
        let json_result = ResponseFormatter::format(&data, content_types::SPARQL_RESULTS_JSON);
        assert!(json_result.is_ok());
        assert!(json_result.unwrap().contains("John"));

        // Test XML formatting (placeholder)
        let xml_result = ResponseFormatter::format(&data, content_types::SPARQL_RESULTS_XML);
        assert!(xml_result.is_ok());
        assert!(xml_result.unwrap().contains("sparql"));
    }
}
