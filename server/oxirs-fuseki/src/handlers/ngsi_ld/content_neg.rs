//! Content Negotiation for NGSI-LD API
//!
//! Handles Accept and Content-Type header negotiation for NGSI-LD requests.

use super::types::NgsiError;
use axum::http::HeaderMap;

/// Supported NGSI-LD response formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NgsiFormat {
    /// JSON-LD (application/ld+json)
    #[default]
    JsonLd,
    /// Plain JSON (application/json)
    Json,
    /// GeoJSON (application/geo+json)
    GeoJson,
}

impl NgsiFormat {
    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            NgsiFormat::JsonLd => "application/ld+json",
            NgsiFormat::Json => "application/json",
            NgsiFormat::GeoJson => "application/geo+json",
        }
    }

    /// Parse from media type string
    pub fn from_media_type(media_type: &str) -> Option<Self> {
        let media_type = media_type.split(';').next()?.trim().to_lowercase();
        match media_type.as_str() {
            "application/ld+json" => Some(NgsiFormat::JsonLd),
            "application/json" => Some(NgsiFormat::Json),
            "application/geo+json" => Some(NgsiFormat::GeoJson),
            "*/*" => Some(NgsiFormat::JsonLd), // Default
            _ => None,
        }
    }

    /// Check if this format includes @context
    pub fn includes_context(&self) -> bool {
        matches!(self, NgsiFormat::JsonLd)
    }
}

/// Content negotiator for NGSI-LD requests
pub struct NgsiContentNegotiator {
    /// Supported formats in order of preference
    supported_formats: Vec<NgsiFormat>,
    /// Default format when no Accept header
    default_format: NgsiFormat,
}

impl Default for NgsiContentNegotiator {
    fn default() -> Self {
        Self {
            supported_formats: vec![NgsiFormat::JsonLd, NgsiFormat::Json, NgsiFormat::GeoJson],
            default_format: NgsiFormat::JsonLd,
        }
    }
}

impl NgsiContentNegotiator {
    /// Create a new content negotiator
    pub fn new() -> Self {
        Self::default()
    }

    /// Negotiate the response format based on Accept header
    pub fn negotiate_response(&self, headers: &HeaderMap) -> Result<NgsiFormat, NgsiError> {
        let accept = headers
            .get("accept")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/ld+json");

        self.parse_accept_header(accept)
    }

    /// Validate the request Content-Type header
    pub fn validate_content_type(&self, headers: &HeaderMap) -> Result<NgsiFormat, NgsiError> {
        let content_type = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                NgsiError::UnsupportedMediaType("Content-Type header is required".to_string())
            })?;

        NgsiFormat::from_media_type(content_type).ok_or_else(|| {
            NgsiError::UnsupportedMediaType(format!(
                "Unsupported Content-Type: {}. Supported: application/ld+json, application/json",
                content_type
            ))
        })
    }

    /// Parse Accept header and return the best matching format
    fn parse_accept_header(&self, accept: &str) -> Result<NgsiFormat, NgsiError> {
        // Parse Accept header with quality values
        let mut candidates: Vec<(NgsiFormat, f32)> = Vec::new();

        for part in accept.split(',') {
            let part = part.trim();
            let (media_type, quality) = self.parse_accept_part(part);

            if let Some(format) = NgsiFormat::from_media_type(media_type) {
                candidates.push((format, quality));
            }
        }

        if candidates.is_empty() {
            // No acceptable format found, but we should try default
            if accept.contains("*/*") || accept.is_empty() {
                return Ok(self.default_format);
            }
            return Err(NgsiError::NotAcceptable(format!(
                "No acceptable format found. Requested: {}. Supported: application/ld+json, application/json, application/geo+json",
                accept
            )));
        }

        // Sort by quality (highest first)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return the best match that we support
        for (format, _) in candidates {
            if self.supported_formats.contains(&format) {
                return Ok(format);
            }
        }

        Ok(self.default_format)
    }

    /// Parse a single Accept header part (e.g., "application/json;q=0.9")
    fn parse_accept_part<'a>(&self, part: &'a str) -> (&'a str, f32) {
        let mut iter = part.split(';');
        let media_type = iter.next().unwrap_or("").trim();
        let quality = iter
            .find_map(|param| {
                param
                    .trim()
                    .strip_prefix("q=")
                    .and_then(|q| q.parse::<f32>().ok())
            })
            .unwrap_or(1.0);

        (media_type, quality)
    }

    /// Extract Link header context URI
    pub fn extract_link_context(&self, headers: &HeaderMap) -> Option<String> {
        headers
            .get("link")
            .and_then(|v| v.to_str().ok())
            .and_then(|link| self.parse_link_header(link))
    }

    /// Parse Link header for @context
    fn parse_link_header(&self, link: &str) -> Option<String> {
        // Format: <https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld>;
        //         rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"
        for part in link.split(',') {
            let part = part.trim();

            // Check if this is a context link
            if part.contains("json-ld#context")
                || part.contains("rel=\"http://www.w3.org/ns/json-ld#context\"")
            {
                // Extract URI between < and >
                if let Some(start) = part.find('<') {
                    if let Some(end) = part.find('>') {
                        return Some(part[start + 1..end].to_string());
                    }
                }
            }
        }
        None
    }

    /// Get the NGSILD-Tenant header value
    pub fn extract_tenant(&self, headers: &HeaderMap) -> Option<String> {
        headers
            .get("ngsild-tenant")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    }

    /// Build response headers for NGSI-LD response
    pub fn build_response_headers(&self, format: NgsiFormat) -> Vec<(&'static str, String)> {
        vec![
            ("Content-Type", format.mime_type().to_string()),
            ("NGSILD-Version", super::NGSI_LD_VERSION.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngsi_format_from_media_type() {
        assert_eq!(
            NgsiFormat::from_media_type("application/ld+json"),
            Some(NgsiFormat::JsonLd)
        );
        assert_eq!(
            NgsiFormat::from_media_type("application/json"),
            Some(NgsiFormat::Json)
        );
        assert_eq!(
            NgsiFormat::from_media_type("application/geo+json"),
            Some(NgsiFormat::GeoJson)
        );
        assert_eq!(NgsiFormat::from_media_type("text/html"), None);
    }

    #[test]
    fn test_ngsi_format_mime_type() {
        assert_eq!(NgsiFormat::JsonLd.mime_type(), "application/ld+json");
        assert_eq!(NgsiFormat::Json.mime_type(), "application/json");
        assert_eq!(NgsiFormat::GeoJson.mime_type(), "application/geo+json");
    }

    #[test]
    fn test_content_negotiator_accept() {
        let neg = NgsiContentNegotiator::new();

        // Test simple Accept headers
        assert_eq!(
            neg.parse_accept_header("application/ld+json").unwrap(),
            NgsiFormat::JsonLd
        );
        assert_eq!(
            neg.parse_accept_header("application/json").unwrap(),
            NgsiFormat::Json
        );

        // Test quality values
        assert_eq!(
            neg.parse_accept_header("application/json;q=0.9, application/ld+json;q=1.0")
                .unwrap(),
            NgsiFormat::JsonLd
        );

        // Test wildcard
        assert_eq!(neg.parse_accept_header("*/*").unwrap(), NgsiFormat::JsonLd);
    }

    #[test]
    fn test_link_header_parsing() {
        let neg = NgsiContentNegotiator::new();

        let link = r#"<https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json""#;
        let ctx = neg.parse_link_header(link);
        assert_eq!(
            ctx,
            Some("https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld".to_string())
        );
    }

    #[test]
    fn test_accept_part_parsing() {
        let neg = NgsiContentNegotiator::new();

        let (media, quality) = neg.parse_accept_part("application/json;q=0.8");
        assert_eq!(media, "application/json");
        assert!((quality - 0.8).abs() < f32::EPSILON);

        let (media, quality) = neg.parse_accept_part("application/ld+json");
        assert_eq!(media, "application/ld+json");
        assert!((quality - 1.0).abs() < f32::EPSILON);
    }
}
