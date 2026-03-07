//! SPARQL 1.1 Protocol Handler
//!
//! Implements the W3C SPARQL 1.1 Protocol specification:
//! - <https://www.w3.org/TR/sparql11-protocol/>
//!
//! Features:
//! - MIME type negotiation for query results (JSON, CSV, TSV, XML, Turtle, N-Triples)
//! - Graph Store Protocol content-type handling
//! - Multipart SPARQL UPDATE bodies (`multipart/form-data`)
//! - Content-type driven result serialisation helpers
//! - `Accept` header parsing and priority-ordered media type selection

use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// SPARQL result MIME types
// ─────────────────────────────────────────────────────────────────────────────

/// SPARQL query result serialisation format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SparqlResultFormat {
    /// `application/sparql-results+json`
    Json,
    /// `application/sparql-results+xml`
    Xml,
    /// `text/csv`
    Csv,
    /// `text/tab-separated-values`
    Tsv,
    /// `text/plain` (N-Triples, for CONSTRUCT / DESCRIBE)
    NTriples,
    /// `text/turtle` (for CONSTRUCT / DESCRIBE)
    Turtle,
    /// `application/n-quads`
    NQuads,
    /// `application/trig`
    TriG,
    /// `application/ld+json`
    JsonLd,
}

impl SparqlResultFormat {
    /// Return the canonical MIME type string.
    pub fn mime_type(&self) -> &'static str {
        match self {
            SparqlResultFormat::Json => "application/sparql-results+json",
            SparqlResultFormat::Xml => "application/sparql-results+xml",
            SparqlResultFormat::Csv => "text/csv",
            SparqlResultFormat::Tsv => "text/tab-separated-values",
            SparqlResultFormat::NTriples => "text/plain",
            SparqlResultFormat::Turtle => "text/turtle",
            SparqlResultFormat::NQuads => "application/n-quads",
            SparqlResultFormat::TriG => "application/trig",
            SparqlResultFormat::JsonLd => "application/ld+json",
        }
    }

    /// Try to construct from a MIME type string (case-insensitive, ignores parameters).
    pub fn from_mime(mime: &str) -> Option<Self> {
        let base = mime.split(';').next().unwrap_or(mime).trim().to_lowercase();
        match base.as_str() {
            "application/sparql-results+json" | "application/json" => {
                Some(SparqlResultFormat::Json)
            }
            "application/sparql-results+xml" | "application/xml" => Some(SparqlResultFormat::Xml),
            "text/csv" => Some(SparqlResultFormat::Csv),
            "text/tab-separated-values" | "text/tsv" => Some(SparqlResultFormat::Tsv),
            "text/plain" | "application/n-triples" => Some(SparqlResultFormat::NTriples),
            "text/turtle" | "application/turtle" | "application/x-turtle" => {
                Some(SparqlResultFormat::Turtle)
            }
            "application/n-quads" => Some(SparqlResultFormat::NQuads),
            "application/trig" => Some(SparqlResultFormat::TriG),
            "application/ld+json" => Some(SparqlResultFormat::JsonLd),
            _ => None,
        }
    }

    /// Return `true` if this format is usable for SELECT / ASK queries.
    pub fn is_select_format(&self) -> bool {
        matches!(
            self,
            SparqlResultFormat::Json
                | SparqlResultFormat::Xml
                | SparqlResultFormat::Csv
                | SparqlResultFormat::Tsv
        )
    }

    /// Return `true` if this format is usable for CONSTRUCT / DESCRIBE queries.
    pub fn is_graph_format(&self) -> bool {
        matches!(
            self,
            SparqlResultFormat::NTriples
                | SparqlResultFormat::Turtle
                | SparqlResultFormat::NQuads
                | SparqlResultFormat::TriG
                | SparqlResultFormat::JsonLd
        )
    }
}

impl fmt::Display for SparqlResultFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mime_type())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Accept-header negotiation
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry from a parsed `Accept` header.
#[derive(Debug, Clone)]
pub struct AcceptEntry {
    /// The MIME type (may contain wildcards)
    pub mime: String,
    /// Quality factor (0.0–1.0)
    pub q: f32,
}

/// Parse an HTTP `Accept` header value into ordered `AcceptEntry` items
/// (sorted descending by quality).
pub fn parse_accept_header(header: &str) -> Vec<AcceptEntry> {
    let mut entries: Vec<AcceptEntry> = header
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return None;
            }
            let mut segments = part.splitn(2, ';');
            let mime = segments.next()?.trim().to_string();
            let q = segments
                .next()
                .and_then(|params| {
                    params
                        .split(';')
                        .find(|p| p.trim().starts_with("q="))
                        .and_then(|p| p.trim().strip_prefix("q="))
                        .and_then(|v| v.parse::<f32>().ok())
                })
                .unwrap_or(1.0);
            Some(AcceptEntry { mime, q })
        })
        .collect();

    // Sort by quality descending, then by specificity (fewer wildcards first)
    entries.sort_by(|a, b| {
        b.q.partial_cmp(&a.q)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let spec_a = if a.mime == "*/*" {
                    0
                } else if a.mime.ends_with("/*") {
                    1
                } else {
                    2
                };
                let spec_b = if b.mime == "*/*" {
                    0
                } else if b.mime.ends_with("/*") {
                    1
                } else {
                    2
                };
                spec_b.cmp(&spec_a)
            })
    });

    entries
}

/// Negotiate the best `SparqlResultFormat` for a SELECT/ASK query given an
/// `Accept` header.  Returns `SparqlResultFormat::Json` if negotiation fails.
pub fn negotiate_select_format(accept: &str) -> SparqlResultFormat {
    let default_formats = [
        SparqlResultFormat::Json,
        SparqlResultFormat::Xml,
        SparqlResultFormat::Csv,
        SparqlResultFormat::Tsv,
    ];
    negotiate_format(accept, &default_formats).unwrap_or(SparqlResultFormat::Json)
}

/// Negotiate the best `SparqlResultFormat` for a CONSTRUCT/DESCRIBE query given
/// an `Accept` header.  Returns `SparqlResultFormat::Turtle` if negotiation fails.
pub fn negotiate_graph_format(accept: &str) -> SparqlResultFormat {
    let default_formats = [
        SparqlResultFormat::Turtle,
        SparqlResultFormat::NTriples,
        SparqlResultFormat::JsonLd,
        SparqlResultFormat::NQuads,
        SparqlResultFormat::TriG,
    ];
    negotiate_format(accept, &default_formats).unwrap_or(SparqlResultFormat::Turtle)
}

fn negotiate_format(accept: &str, supported: &[SparqlResultFormat]) -> Option<SparqlResultFormat> {
    if accept.trim().is_empty() {
        return supported.first().copied();
    }

    let entries = parse_accept_header(accept);
    for entry in &entries {
        // Wildcard: return preferred supported format
        if entry.mime == "*/*" || entry.mime == "application/*" {
            if let Some(fmt) = supported.first() {
                return Some(*fmt);
            }
        }
        if let Some(fmt) = SparqlResultFormat::from_mime(&entry.mime) {
            if supported.contains(&fmt) {
                return Some(fmt);
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Store Protocol
// ─────────────────────────────────────────────────────────────────────────────

/// The HTTP method semantics for the SPARQL Graph Store Protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphStoreOperation {
    /// `GET` — retrieve a named graph
    Get,
    /// `PUT` — replace a named graph
    Put,
    /// `POST` — merge into a named graph
    Post,
    /// `DELETE` — delete a named graph
    Delete,
    /// `HEAD` — check existence without body
    Head,
}

/// Content-type information extracted from the Graph Store Protocol request.
#[derive(Debug, Clone)]
pub struct GraphStoreContentType {
    /// The RDF serialisation format detected from the request Content-Type
    pub format: SparqlResultFormat,
    /// Whether the content type was valid and recognised
    pub recognised: bool,
}

impl GraphStoreContentType {
    /// Parse the `Content-Type` header from a Graph Store request.
    pub fn from_header(content_type: Option<&str>) -> Self {
        match content_type {
            None => GraphStoreContentType {
                format: SparqlResultFormat::Turtle,
                recognised: false,
            },
            Some(ct) => match SparqlResultFormat::from_mime(ct) {
                Some(fmt) if fmt.is_graph_format() => GraphStoreContentType {
                    format: fmt,
                    recognised: true,
                },
                Some(_) | None => GraphStoreContentType {
                    format: SparqlResultFormat::Turtle,
                    recognised: false,
                },
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multipart SPARQL UPDATE
// ─────────────────────────────────────────────────────────────────────────────

/// A single part extracted from a `multipart/form-data` SPARQL Update request.
#[derive(Debug, Clone)]
pub struct MultipartUpdatePart {
    /// `update` (required), `using-graph-uri`, or `using-named-graph-uri`
    pub name: String,
    /// Content-Type of this part (defaults to `application/sparql-update`)
    pub content_type: String,
    /// Raw body bytes of this part
    pub body: Vec<u8>,
}

impl MultipartUpdatePart {
    /// Return the body decoded as UTF-8, or an error.
    pub fn body_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(&self.body)
    }
}

/// Parse a `multipart/form-data` body into individual `MultipartUpdatePart`s.
///
/// This is a minimal implementation sufficient for the SPARQL 1.1 protocol:
/// it handles basic boundary-delimited parts with `Content-Disposition`
/// and `Content-Type` headers.
pub fn parse_multipart_update(
    boundary: &str,
    body: &[u8],
) -> Result<Vec<MultipartUpdatePart>, MultipartError> {
    let delimiter = format!("--{}", boundary);
    let end_delimiter = format!("--{}--", boundary);

    let body_str =
        std::str::from_utf8(body).map_err(|e| MultipartError::InvalidUtf8(e.to_string()))?;

    let mut parts = Vec::new();

    for section in body_str.split(&delimiter) {
        let section = section.trim_start_matches("\r\n").trim_start_matches('\n');

        // Skip the end delimiter and empty sections
        if section.trim().is_empty() || section.starts_with("--") || section == end_delimiter {
            continue;
        }

        // Split headers from body
        let split_pos = section
            .find("\r\n\r\n")
            .or_else(|| section.find("\n\n"))
            .ok_or(MultipartError::MalformedPart)?;

        let headers_str = &section[..split_pos];
        let body_start = if section[split_pos..].starts_with("\r\n\r\n") {
            split_pos + 4
        } else {
            split_pos + 2
        };

        let part_body = if body_start <= section.len() {
            &section[body_start..]
        } else {
            ""
        };
        // Trim trailing boundary markers
        let part_body = part_body
            .trim_end_matches("--")
            .trim_end_matches('\n')
            .trim_end_matches('\r');

        // Parse headers
        let mut name = String::new();
        let mut content_type = "application/sparql-update".to_string();

        for line in headers_str.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let lower = line.to_lowercase();
            if lower.starts_with("content-disposition:") {
                // Extract name="..."
                if let Some(n) = extract_header_param(line, "name") {
                    name = n;
                }
            } else if lower.starts_with("content-type:") {
                if let Some((_key, ct)) = line.split_once(':') {
                    content_type = ct.trim().to_string();
                }
            }
        }

        if name.is_empty() {
            return Err(MultipartError::MissingName);
        }

        parts.push(MultipartUpdatePart {
            name,
            content_type,
            body: part_body.as_bytes().to_vec(),
        });
    }

    Ok(parts)
}

fn extract_header_param(header_line: &str, param_name: &str) -> Option<String> {
    let search = format!("{}=\"", param_name);
    let start = header_line.find(&search)? + search.len();
    let rest = &header_line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract the `boundary` value from a `Content-Type: multipart/form-data; boundary=...` header.
pub fn extract_multipart_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').skip(1).find_map(|param| {
        let param = param.trim();
        param
            .strip_prefix("boundary=")
            .map(|boundary| boundary.trim_matches('"').to_string())
    })
}

/// Error type for multipart parsing.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MultipartError {
    #[error("Invalid UTF-8 in multipart body: {0}")]
    InvalidUtf8(String),
    #[error("Malformed part: missing header/body separator")]
    MalformedPart,
    #[error("Part is missing 'name' in Content-Disposition")]
    MissingName,
}

// ─────────────────────────────────────────────────────────────────────────────
// SparqlProtocolHandler
// ─────────────────────────────────────────────────────────────────────────────

/// Handles SPARQL 1.1 protocol concerns: request dispatching, MIME negotiation,
/// and response construction.
pub struct SparqlProtocolHandler {
    /// Default format for SELECT / ASK results
    pub default_select_format: SparqlResultFormat,
    /// Default format for CONSTRUCT / DESCRIBE results
    pub default_graph_format: SparqlResultFormat,
}

impl Default for SparqlProtocolHandler {
    fn default() -> Self {
        SparqlProtocolHandler {
            default_select_format: SparqlResultFormat::Json,
            default_graph_format: SparqlResultFormat::Turtle,
        }
    }
}

impl SparqlProtocolHandler {
    /// Create a new handler with custom defaults.
    pub fn new(
        default_select_format: SparqlResultFormat,
        default_graph_format: SparqlResultFormat,
    ) -> Self {
        SparqlProtocolHandler {
            default_select_format,
            default_graph_format,
        }
    }

    /// Determine the result format for a SELECT / ASK query from request headers.
    pub fn select_format_from_headers(&self, headers: &HeaderMap) -> SparqlResultFormat {
        match headers.get("accept").and_then(|v| v.to_str().ok()) {
            Some(accept) => negotiate_select_format(accept),
            None => self.default_select_format,
        }
    }

    /// Determine the result format for a CONSTRUCT / DESCRIBE query from request headers.
    pub fn graph_format_from_headers(&self, headers: &HeaderMap) -> SparqlResultFormat {
        match headers.get("accept").and_then(|v| v.to_str().ok()) {
            Some(accept) => negotiate_graph_format(accept),
            None => self.default_graph_format,
        }
    }

    /// Build an HTTP response for a given result body and format.
    pub fn build_query_response(&self, body: String, format: SparqlResultFormat) -> Response {
        use axum::http::{header, HeaderValue};
        use axum::response::IntoResponse;

        let content_type = format.mime_type();
        match HeaderValue::from_str(content_type) {
            Ok(ct_value) => {
                let mut resp = body.into_response();
                resp.headers_mut().insert(header::CONTENT_TYPE, ct_value);
                resp
            }
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Content-Type error").into_response(),
        }
    }

    /// Parse a multipart UPDATE request body, extracting the update statement
    /// and optional named graph URIs.
    ///
    /// Returns `(update_string, using_graphs, using_named_graphs)`.
    pub fn parse_multipart_update_request(
        &self,
        content_type_header: &str,
        body: &[u8],
    ) -> Result<(String, Vec<String>, Vec<String>), MultipartError> {
        let boundary =
            extract_multipart_boundary(content_type_header).ok_or(MultipartError::MalformedPart)?;

        let parts = parse_multipart_update(&boundary, body)?;

        let mut update = String::new();
        let mut using_graphs = Vec::new();
        let mut using_named = Vec::new();

        for part in parts {
            match part.name.as_str() {
                "update" => {
                    update = part
                        .body_str()
                        .map_err(|e| MultipartError::InvalidUtf8(e.to_string()))?
                        .to_string();
                }
                "using-graph-uri" => {
                    let uri = part
                        .body_str()
                        .map_err(|e| MultipartError::InvalidUtf8(e.to_string()))?
                        .trim()
                        .to_string();
                    using_graphs.push(uri);
                }
                "using-named-graph-uri" => {
                    let uri = part
                        .body_str()
                        .map_err(|e| MultipartError::InvalidUtf8(e.to_string()))?
                        .trim()
                        .to_string();
                    using_named.push(uri);
                }
                _ => {} // Ignore unknown parts
            }
        }

        if update.is_empty() {
            return Err(MultipartError::MissingName);
        }

        Ok((update, using_graphs, using_named))
    }

    /// Build a Graph Store Protocol response for the given RDF content and content type.
    pub fn build_graph_store_response(
        &self,
        body: String,
        content_type: &GraphStoreContentType,
    ) -> Response {
        self.build_query_response(body, content_type.format)
    }

    /// Generate a 406 Not Acceptable response when MIME negotiation fails.
    pub fn not_acceptable_response(&self) -> Response {
        use axum::response::IntoResponse;
        (
            StatusCode::NOT_ACCEPTABLE,
            "No supported media type in Accept header",
        )
            .into_response()
    }

    /// Generate a 415 Unsupported Media Type response.
    pub fn unsupported_media_type_response(&self, provided: &str) -> Response {
        use axum::response::IntoResponse;
        (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            format!("Unsupported Content-Type: {provided}"),
        )
            .into_response()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── SparqlResultFormat ───────────────────────────────────────────────

    // 1. MIME type strings are correct
    #[test]
    fn test_mime_type_strings() {
        assert_eq!(
            SparqlResultFormat::Json.mime_type(),
            "application/sparql-results+json"
        );
        assert_eq!(
            SparqlResultFormat::Xml.mime_type(),
            "application/sparql-results+xml"
        );
        assert_eq!(SparqlResultFormat::Csv.mime_type(), "text/csv");
        assert_eq!(
            SparqlResultFormat::Tsv.mime_type(),
            "text/tab-separated-values"
        );
        assert_eq!(SparqlResultFormat::Turtle.mime_type(), "text/turtle");
        assert_eq!(SparqlResultFormat::NTriples.mime_type(), "text/plain");
    }

    // 2. from_mime round-trips
    #[test]
    fn test_from_mime_roundtrip() {
        for fmt in [
            SparqlResultFormat::Json,
            SparqlResultFormat::Xml,
            SparqlResultFormat::Csv,
            SparqlResultFormat::Tsv,
            SparqlResultFormat::Turtle,
            SparqlResultFormat::NTriples,
            SparqlResultFormat::NQuads,
            SparqlResultFormat::TriG,
            SparqlResultFormat::JsonLd,
        ] {
            let mime = fmt.mime_type();
            let decoded = SparqlResultFormat::from_mime(mime);
            assert!(decoded.is_some(), "round-trip failed for {mime}");
        }
    }

    // 3. from_mime accepts aliases
    #[test]
    fn test_from_mime_aliases() {
        assert_eq!(
            SparqlResultFormat::from_mime("application/json"),
            Some(SparqlResultFormat::Json)
        );
        assert_eq!(
            SparqlResultFormat::from_mime("application/xml"),
            Some(SparqlResultFormat::Xml)
        );
        assert_eq!(
            SparqlResultFormat::from_mime("application/turtle"),
            Some(SparqlResultFormat::Turtle)
        );
    }

    // 4. from_mime ignores parameters
    #[test]
    fn test_from_mime_ignores_params() {
        let fmt = SparqlResultFormat::from_mime("text/turtle; charset=utf-8");
        assert_eq!(fmt, Some(SparqlResultFormat::Turtle));
    }

    // 5. is_select_format / is_graph_format
    #[test]
    fn test_select_vs_graph_format() {
        assert!(SparqlResultFormat::Json.is_select_format());
        assert!(!SparqlResultFormat::Json.is_graph_format());
        assert!(SparqlResultFormat::Turtle.is_graph_format());
        assert!(!SparqlResultFormat::Turtle.is_select_format());
    }

    // ─── Accept header parsing ────────────────────────────────────────────

    // 6. Parses single MIME type
    #[test]
    fn test_parse_accept_single() {
        let entries = parse_accept_header("application/sparql-results+json");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].mime, "application/sparql-results+json");
        assert!((entries[0].q - 1.0).abs() < 1e-6);
    }

    // 7. Parses multiple MIME types with q values
    #[test]
    fn test_parse_accept_multiple_with_q() {
        let entries =
            parse_accept_header("application/sparql-results+json, text/csv; q=0.5, */*; q=0.1");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].mime, "application/sparql-results+json");
        // Sorted by q descending
        assert!(entries[0].q >= entries[1].q);
        assert!(entries[1].q >= entries[2].q);
    }

    // 8. Empty Accept header returns empty list
    #[test]
    fn test_parse_accept_empty() {
        assert!(parse_accept_header("").is_empty());
    }

    // ─── Negotiation ──────────────────────────────────────────────────────

    // 9. negotiate_select_format returns JSON by default
    #[test]
    fn test_negotiate_select_default() {
        let fmt = negotiate_select_format("");
        assert_eq!(fmt, SparqlResultFormat::Json);
    }

    // 10. negotiate_select_format picks highest-quality supported format
    #[test]
    fn test_negotiate_select_prefers_quality() {
        let fmt = negotiate_select_format("text/csv; q=0.9, application/sparql-results+xml; q=0.8");
        assert_eq!(fmt, SparqlResultFormat::Csv);
    }

    // 11. negotiate_select_format wildcard falls back to JSON
    #[test]
    fn test_negotiate_select_wildcard() {
        let fmt = negotiate_select_format("*/*");
        assert_eq!(fmt, SparqlResultFormat::Json);
    }

    // 12. negotiate_graph_format returns Turtle by default
    #[test]
    fn test_negotiate_graph_default() {
        let fmt = negotiate_graph_format("");
        assert_eq!(fmt, SparqlResultFormat::Turtle);
    }

    // 13. negotiate_graph_format picks correct graph format
    #[test]
    fn test_negotiate_graph_picks_jsonld() {
        let fmt = negotiate_graph_format("application/ld+json");
        assert_eq!(fmt, SparqlResultFormat::JsonLd);
    }

    // 14. negotiate_select_format returns TSV when requested
    #[test]
    fn test_negotiate_select_tsv() {
        let fmt = negotiate_select_format("text/tab-separated-values");
        assert_eq!(fmt, SparqlResultFormat::Tsv);
    }

    // ─── Graph Store Protocol ─────────────────────────────────────────────

    // 15. GraphStoreContentType recognises Turtle
    #[test]
    fn test_graph_store_ct_turtle() {
        let ct = GraphStoreContentType::from_header(Some("text/turtle"));
        assert_eq!(ct.format, SparqlResultFormat::Turtle);
        assert!(ct.recognised);
    }

    // 16. GraphStoreContentType rejects SELECT formats
    #[test]
    fn test_graph_store_ct_rejects_json_select() {
        let ct = GraphStoreContentType::from_header(Some("application/sparql-results+json"));
        assert!(!ct.recognised, "SELECT format is not valid for graph store");
    }

    // 17. GraphStoreContentType defaults to Turtle when missing
    #[test]
    fn test_graph_store_ct_missing() {
        let ct = GraphStoreContentType::from_header(None);
        assert_eq!(ct.format, SparqlResultFormat::Turtle);
        assert!(!ct.recognised);
    }

    // 18. GraphStoreContentType accepts N-Quads
    #[test]
    fn test_graph_store_ct_nquads() {
        let ct = GraphStoreContentType::from_header(Some("application/n-quads"));
        assert_eq!(ct.format, SparqlResultFormat::NQuads);
        assert!(ct.recognised);
    }

    // ─── Multipart UPDATE ─────────────────────────────────────────────────

    // 19. extract_multipart_boundary parses boundary
    #[test]
    fn test_extract_boundary() {
        let ct = "multipart/form-data; boundary=----WebKitFormBoundary";
        let b = extract_multipart_boundary(ct);
        assert_eq!(b, Some("----WebKitFormBoundary".to_string()));
    }

    // 20. extract_multipart_boundary handles quoted boundary
    #[test]
    fn test_extract_boundary_quoted() {
        let ct = r#"multipart/form-data; boundary="my-boundary-123""#;
        let b = extract_multipart_boundary(ct);
        assert_eq!(b, Some("my-boundary-123".to_string()));
    }

    // 21. parse_multipart_update extracts update part
    #[test]
    fn test_parse_multipart_basic() {
        let boundary = "testboundary";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"update\"\r\nContent-Type: application/sparql-update\r\n\r\nINSERT DATA {{ <s> <p> <o> }}\r\n--{boundary}--",
        );
        let parts = parse_multipart_update(boundary, body.as_bytes()).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].name, "update");
        assert!(parts[0].body_str().unwrap().contains("INSERT DATA"));
    }

    // 22. parse_multipart_update extracts using-graph-uri
    #[test]
    fn test_parse_multipart_with_graph_uri() {
        let boundary = "b1";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"update\"\r\n\r\nINSERT DATA {{ <s> <p> <o> }}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"using-graph-uri\"\r\n\r\nhttp://example.org/graph\r\n--{boundary}--",
        );
        let parts = parse_multipart_update(boundary, body.as_bytes()).unwrap();
        assert_eq!(parts.len(), 2);
        let graph = parts.iter().find(|p| p.name == "using-graph-uri").unwrap();
        assert!(graph
            .body_str()
            .unwrap()
            .contains("http://example.org/graph"));
    }

    // 23. SparqlProtocolHandler parse_multipart_update_request
    #[test]
    fn test_handler_parse_multipart() {
        let handler = SparqlProtocolHandler::default();
        let boundary = "bound42";
        let ct = format!("multipart/form-data; boundary={boundary}");
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"update\"\r\n\r\nDELETE WHERE {{ ?s ?p ?o }}\r\n--{boundary}--",
        );
        let (update, graphs, named) = handler
            .parse_multipart_update_request(&ct, body.as_bytes())
            .unwrap();
        assert!(update.contains("DELETE WHERE"));
        assert!(graphs.is_empty());
        assert!(named.is_empty());
    }

    // 24. SparqlProtocolHandler builds correct Content-Type in response
    #[test]
    fn test_handler_build_response_content_type() {
        let handler = SparqlProtocolHandler::default();
        let resp = handler.build_query_response("{}".to_string(), SparqlResultFormat::Json);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok());
        assert_eq!(ct, Some("application/sparql-results+json"));
    }

    // 25. SparqlProtocolHandler negotiates format from Accept header
    #[test]
    fn test_handler_negotiate_from_headers() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;

        let handler = SparqlProtocolHandler::default();
        let mut headers = HeaderMap::new();
        headers.insert("accept", HeaderValue::from_static("text/csv"));
        let fmt = handler.select_format_from_headers(&headers);
        assert_eq!(fmt, SparqlResultFormat::Csv);
    }

    // 26. not_acceptable_response returns 406
    #[test]
    fn test_not_acceptable_response() {
        let handler = SparqlProtocolHandler::default();
        let resp = handler.not_acceptable_response();
        assert_eq!(resp.status(), StatusCode::NOT_ACCEPTABLE);
    }

    // 27. unsupported_media_type_response returns 415
    #[test]
    fn test_unsupported_media_type_response() {
        let handler = SparqlProtocolHandler::default();
        let resp = handler.unsupported_media_type_response("application/octet-stream");
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    // 28. build_graph_store_response uses correct content type
    #[test]
    fn test_graph_store_response_content_type() {
        let handler = SparqlProtocolHandler::default();
        let ct = GraphStoreContentType {
            format: SparqlResultFormat::NTriples,
            recognised: true,
        };
        let resp = handler.build_graph_store_response("<s> <p> <o> .".to_string(), &ct);
        let header_ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok());
        assert_eq!(header_ct, Some("text/plain"));
    }

    // 29. SparqlResultFormat Display uses mime_type
    #[test]
    fn test_display() {
        let fmt = SparqlResultFormat::Turtle;
        assert_eq!(fmt.to_string(), "text/turtle");
    }

    // 30. parse_multipart_update missing boundary returns error
    #[test]
    fn test_multipart_missing_boundary_returns_error() {
        let handler = SparqlProtocolHandler::default();
        let result = handler.parse_multipart_update_request(
            "multipart/form-data", // No boundary
            b"irrelevant",
        );
        assert!(result.is_err());
    }
}
