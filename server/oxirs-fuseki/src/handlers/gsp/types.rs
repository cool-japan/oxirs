//! GSP Types and Parameters

use serde::{Deserialize, Serialize};
use std::fmt;

/// GSP query parameters
#[derive(Debug, Clone, Deserialize)]
pub struct GspParams {
    /// Target graph parameter: "default", "union", or a graph URI
    pub graph: Option<String>,

    /// Alternative: use ?default for default graph
    pub default: Option<String>,
}

impl GspParams {
    /// Create params for default graph
    pub fn default_graph() -> Self {
        Self {
            graph: None,
            default: Some(String::new()),
        }
    }

    /// Create params for named graph
    pub fn named_graph(uri: impl Into<String>) -> Self {
        Self {
            graph: Some(uri.into()),
            default: None,
        }
    }

    /// Check if this targets the default graph
    pub fn is_default_graph(&self) -> bool {
        self.default.is_some() || self.graph.as_deref() == Some("default")
    }

    /// Check if this targets the union graph
    pub fn is_union_graph(&self) -> bool {
        self.graph.as_deref() == Some("union")
    }

    /// Get the named graph URI
    pub fn graph_uri(&self) -> Option<&str> {
        self.graph
            .as_deref()
            .filter(|g| *g != "default" && *g != "union")
    }
}

/// Graph target for GSP operations
#[derive(Debug, Clone)]
pub enum GraphTarget {
    /// Default graph
    Default,
    /// Union of all named graphs
    Union,
    /// Named graph with specific URI
    Named(String),
}

impl GraphTarget {
    /// Create from GSP parameters
    pub fn from_params(params: &GspParams) -> Result<Self, GspError> {
        if params.is_default_graph() {
            Ok(GraphTarget::Default)
        } else if params.is_union_graph() {
            Ok(GraphTarget::Union)
        } else if let Some(uri) = params.graph_uri() {
            Ok(GraphTarget::Named(uri.to_string()))
        } else {
            // No graph parameter means default graph
            Ok(GraphTarget::Default)
        }
    }

    /// Get a label for this graph target (for logging/errors)
    pub fn label(&self) -> String {
        match self {
            GraphTarget::Default => "default graph".to_string(),
            GraphTarget::Union => "union graph".to_string(),
            GraphTarget::Named(uri) => format!("graph <{}>", uri),
        }
    }

    /// Check if this is a writable target
    pub fn is_writable(&self) -> bool {
        match self {
            GraphTarget::Default | GraphTarget::Named(_) => true,
            GraphTarget::Union => false, // Union graph is read-only
        }
    }
}

impl fmt::Display for GraphTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// RDF format for content negotiation
/// Re-exported from oxirs_core for convenience
pub use oxirs_core::parser::RdfFormat;

/// Extension methods for RdfFormat
pub trait RdfFormatExt {
    /// Parse from media type string (with additional media types)
    fn from_media_type_gsp(media_type: &str) -> Option<RdfFormat>;

    /// Default format preference order for content negotiation
    fn default_order_gsp() -> Vec<RdfFormat>;
}

impl RdfFormatExt for RdfFormat {
    fn from_media_type_gsp(media_type: &str) -> Option<RdfFormat> {
        let media_type = media_type.split(';').next()?.trim().to_lowercase();
        match media_type.as_str() {
            "text/turtle" | "application/x-turtle" => Some(RdfFormat::Turtle),
            "application/n-triples" | "text/plain" => Some(RdfFormat::NTriples),
            "application/n-quads" => Some(RdfFormat::NQuads),
            "application/rdf+xml" | "application/xml" => Some(RdfFormat::RdfXml),
            "application/ld+json" | "application/json" => Some(RdfFormat::JsonLd),
            "application/trig" => Some(RdfFormat::TriG),
            _ => None,
        }
    }

    fn default_order_gsp() -> Vec<RdfFormat> {
        vec![
            RdfFormat::Turtle,
            RdfFormat::JsonLd,
            RdfFormat::NTriples,
            RdfFormat::RdfXml,
            RdfFormat::TriG,
            RdfFormat::NQuads,
        ]
    }
}

/// GSP-specific errors
#[derive(Debug, thiserror::Error)]
pub enum GspError {
    #[error("Graph not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Method not allowed: {0}")]
    MethodNotAllowed(String),

    #[error("Unsupported media type: {0}")]
    UnsupportedMediaType(String),

    #[error("Not acceptable: no suitable content type available")]
    NotAcceptable,

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl GspError {
    /// Get HTTP status code for this error
    pub fn status_code(&self) -> u16 {
        match self {
            GspError::NotFound(_) => 404,
            GspError::BadRequest(_) => 400,
            GspError::MethodNotAllowed(_) => 405,
            GspError::UnsupportedMediaType(_) => 415,
            GspError::NotAcceptable => 406,
            GspError::ParseError(_) => 400,
            GspError::StoreError(_) => 500,
            GspError::Internal(_) => 500,
        }
    }
}

impl axum::response::IntoResponse for GspError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;
        use axum::response::Json;

        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let message = self.to_string();

        (
            status,
            Json(serde_json::json!({
                "error": message,
                "status": status.as_u16(),
            })),
        )
            .into_response()
    }
}

/// GSP operation statistics
#[derive(Debug, Clone, Serialize)]
pub struct GspStats {
    /// Number of triples affected
    pub triples: usize,
    /// Operation duration in milliseconds
    pub duration_ms: u64,
    /// Graph target
    pub graph: String,
    /// Operation type
    pub operation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gsp_params_default_graph() {
        let params = GspParams::default_graph();
        assert!(params.is_default_graph());
        assert!(!params.is_union_graph());
        assert_eq!(params.graph_uri(), None);
    }

    #[test]
    fn test_gsp_params_named_graph() {
        let params = GspParams::named_graph("http://example.org/graph1");
        assert!(!params.is_default_graph());
        assert!(!params.is_union_graph());
        assert_eq!(params.graph_uri(), Some("http://example.org/graph1"));
    }

    #[test]
    fn test_graph_target_from_params() {
        let params = GspParams::default_graph();
        let target = GraphTarget::from_params(&params).unwrap();
        assert!(matches!(target, GraphTarget::Default));

        let params = GspParams {
            graph: Some("union".to_string()),
            default: None,
        };
        let target = GraphTarget::from_params(&params).unwrap();
        assert!(matches!(target, GraphTarget::Union));
    }

    #[test]
    fn test_graph_target_writable() {
        assert!(GraphTarget::Default.is_writable());
        assert!(GraphTarget::Named("http://example.org/g1".to_string()).is_writable());
        assert!(!GraphTarget::Union.is_writable());
    }

    #[test]
    fn test_rdf_format_media_types() {
        assert_eq!(RdfFormat::Turtle.media_type(), "text/turtle");
        assert_eq!(RdfFormat::JsonLd.media_type(), "application/ld+json");
    }

    #[test]
    fn test_rdf_format_from_media_type() {
        assert_eq!(
            RdfFormat::from_media_type_gsp("text/turtle"),
            Some(RdfFormat::Turtle)
        );
        assert_eq!(
            RdfFormat::from_media_type_gsp("application/ld+json; charset=utf-8"),
            Some(RdfFormat::JsonLd)
        );
        assert_eq!(RdfFormat::from_media_type_gsp("unknown/type"), None);
    }
}
