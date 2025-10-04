//! RDF Bulk Upload Handler
//!
//! Provides endpoint for bulk RDF data upload following Apache Jena Fuseki pattern.
//!
//! POST /upload?graph={graph-uri}
//! - Body: RDF data (auto-detect format or use Content-Type)
//! - Returns: Upload statistics (triples inserted, duration, etc.)
//!
//! Supports:
//! - Direct RDF upload with Content-Type header
//! - Multipart form upload with file attachments
//! - Multiple RDF formats (Turtle, N-Triples, RDF/XML, JSON-LD, TriG, N-Quads)
//! - Batch insertion with transaction support

use axum::{
    body::Bytes,
    extract::{Multipart, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Upload query parameters
#[derive(Debug, Clone, Deserialize)]
pub struct UploadParams {
    /// Target graph URI (default if not specified)
    pub graph: Option<String>,

    /// Format hint (auto-detect if not specified)
    pub format: Option<String>,
}

impl UploadParams {
    /// Get target graph name
    pub fn graph_name(&self) -> oxirs_core::model::GraphName {
        match &self.graph {
            Some(uri) if uri != "default" => oxirs_core::model::NamedNode::new(uri)
                .map(oxirs_core::model::GraphName::NamedNode)
                .unwrap_or(oxirs_core::model::GraphName::DefaultGraph),
            _ => oxirs_core::model::GraphName::DefaultGraph,
        }
    }
}

/// Upload error types
#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl UploadError {
    fn status_code(&self) -> StatusCode {
        match self {
            UploadError::BadRequest(_) => StatusCode::BAD_REQUEST,
            UploadError::ParseError(_) => StatusCode::BAD_REQUEST,
            UploadError::StoreError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            UploadError::UnsupportedFormat(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            UploadError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for UploadError {
    fn into_response(self) -> Response {
        let status = self.status_code();
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

/// Upload statistics
#[derive(Debug, Clone, Serialize)]
pub struct UploadStats {
    /// Number of triples successfully inserted
    pub triples_inserted: usize,

    /// Number of quads successfully inserted
    pub quads_inserted: usize,

    /// Target graph
    pub graph: String,

    /// Upload duration in milliseconds
    pub duration_ms: u64,

    /// Format detected/used
    pub format: String,

    /// File size in bytes
    pub bytes_processed: usize,

    /// Number of parse errors encountered
    pub parse_errors: usize,
}

/// Handle RDF upload via direct POST
///
/// POST /upload?graph={graph-uri}&format={format}
/// Content-Type: text/turtle, application/n-triples, etc.
/// Body: RDF data
pub async fn handle_upload<S: Store + Send + Sync + 'static>(
    Query(params): Query<UploadParams>,
    State(store): State<Arc<S>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, UploadError> {
    let start = Instant::now();

    info!(
        "RDF upload request: graph={:?}, size={} bytes",
        params.graph,
        body.len()
    );

    // 1. Determine format
    let format = detect_format(&params, &headers, &body)?;
    debug!("Detected format: {:?}", format);

    // 2. Parse RDF data
    let (triples, quads) = parse_rdf_data(&body, format)?;
    info!("Parsed {} triples, {} quads", triples.len(), quads.len());

    // 3. Insert into store
    let graph_name = params.graph_name();
    let inserted = insert_data(store.as_ref(), &triples, &quads, &graph_name)?;

    let duration = start.elapsed();
    info!(
        "Upload completed: {} items inserted in {:?}",
        inserted, duration
    );

    // 4. Build response
    let stats = UploadStats {
        triples_inserted: triples.len(),
        quads_inserted: quads.len(),
        graph: params
            .graph
            .clone()
            .unwrap_or_else(|| "default".to_string()),
        duration_ms: duration.as_millis() as u64,
        format: format!("{:?}", format),
        bytes_processed: body.len(),
        parse_errors: 0,
    };

    Ok((StatusCode::OK, Json(stats)).into_response())
}

/// Handle multipart file upload
///
/// POST /upload?graph={graph-uri}
/// Content-Type: multipart/form-data
pub async fn handle_multipart_upload<S: Store + Send + Sync + 'static>(
    Query(params): Query<UploadParams>,
    State(store): State<Arc<S>>,
    mut multipart: Multipart,
) -> Result<Response, UploadError> {
    let start = Instant::now();

    info!("Multipart upload request: graph={:?}", params.graph);

    let mut total_triples = 0;
    let mut total_quads = 0;
    let mut total_bytes = 0;
    let mut files_processed = 0;

    // Process each file in multipart request
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| UploadError::BadRequest(format!("Multipart error: {}", e)))?
    {
        let filename = field.file_name().map(|s| s.to_string());
        let content_type = field.content_type().map(|s| s.to_string());

        debug!(
            "Processing field: filename={:?}, content_type={:?}",
            filename, content_type
        );

        // Read field data
        let data = field
            .bytes()
            .await
            .map_err(|e| UploadError::BadRequest(format!("Field read error: {}", e)))?;

        if data.is_empty() {
            continue;
        }

        total_bytes += data.len();

        // Detect format from filename or content-type
        let format = detect_format_from_file(&filename, &content_type, &data)?;

        // Parse and insert
        let (triples, quads) = parse_rdf_data(&data, format)?;
        let graph_name = params.graph_name();
        let inserted = insert_data(store.as_ref(), &triples, &quads, &graph_name)?;

        total_triples += triples.len();
        total_quads += quads.len();
        files_processed += 1;

        info!(
            "Processed file {:?}: {} triples, {} quads",
            filename,
            triples.len(),
            quads.len()
        );
    }

    let duration = start.elapsed();
    info!(
        "Multipart upload completed: {} files, {} triples, {} quads in {:?}",
        files_processed, total_triples, total_quads, duration
    );

    let stats = UploadStats {
        triples_inserted: total_triples,
        quads_inserted: total_quads,
        graph: params
            .graph
            .clone()
            .unwrap_or_else(|| "default".to_string()),
        duration_ms: duration.as_millis() as u64,
        format: format!("{} files", files_processed),
        bytes_processed: total_bytes,
        parse_errors: 0,
    };

    Ok((StatusCode::OK, Json(stats)).into_response())
}

/// Detect RDF format from request
fn detect_format(
    params: &UploadParams,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<oxirs_core::parser::RdfFormat, UploadError> {
    use oxirs_core::parser::RdfFormat;

    // 1. Check format parameter
    if let Some(format_hint) = &params.format {
        return parse_format_hint(format_hint);
    }

    // 2. Check Content-Type header
    if let Some(content_type) = headers.get(header::CONTENT_TYPE) {
        if let Ok(ct) = content_type.to_str() {
            if let Some(format) = format_from_media_type(ct) {
                return Ok(format);
            }
        }
    }

    // 3. Auto-detect from content
    if let Ok(text) = std::str::from_utf8(body) {
        if let Some(format) = oxirs_core::parser::detect_format_from_content(text) {
            return Ok(format);
        }
    }

    // Default to Turtle
    Ok(RdfFormat::Turtle)
}

/// Detect format from filename and content-type
fn detect_format_from_file(
    filename: &Option<String>,
    content_type: &Option<String>,
    data: &[u8],
) -> Result<oxirs_core::parser::RdfFormat, UploadError> {
    use oxirs_core::parser::RdfFormat;

    // 1. Try filename extension
    if let Some(fname) = filename {
        if let Some(ext) = fname.rsplit('.').next() {
            if let Some(format) = RdfFormat::from_extension(ext) {
                return Ok(format);
            }
        }
    }

    // 2. Try content-type
    if let Some(ct) = content_type {
        if let Some(format) = format_from_media_type(ct) {
            return Ok(format);
        }
    }

    // 3. Auto-detect from content
    if let Ok(text) = std::str::from_utf8(data) {
        if let Some(format) = oxirs_core::parser::detect_format_from_content(text) {
            return Ok(format);
        }
    }

    Ok(RdfFormat::Turtle)
}

/// Parse format hint string
fn parse_format_hint(hint: &str) -> Result<oxirs_core::parser::RdfFormat, UploadError> {
    use oxirs_core::parser::RdfFormat;

    match hint.to_lowercase().as_str() {
        "turtle" | "ttl" => Ok(RdfFormat::Turtle),
        "ntriples" | "nt" => Ok(RdfFormat::NTriples),
        "rdfxml" | "rdf" | "xml" => Ok(RdfFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(RdfFormat::JsonLd),
        "trig" => Ok(RdfFormat::TriG),
        "nquads" | "nq" => Ok(RdfFormat::NQuads),
        _ => Err(UploadError::UnsupportedFormat(hint.to_string())),
    }
}

/// Convert media type to RdfFormat
fn format_from_media_type(media_type: &str) -> Option<oxirs_core::parser::RdfFormat> {
    use oxirs_core::parser::RdfFormat;

    let mt = media_type.split(';').next()?.trim().to_lowercase();
    match mt.as_str() {
        "text/turtle" | "application/x-turtle" => Some(RdfFormat::Turtle),
        "application/n-triples" | "text/plain" => Some(RdfFormat::NTriples),
        "application/rdf+xml" | "application/xml" => Some(RdfFormat::RdfXml),
        "application/ld+json" | "application/json" => Some(RdfFormat::JsonLd),
        "application/trig" => Some(RdfFormat::TriG),
        "application/n-quads" => Some(RdfFormat::NQuads),
        _ => None,
    }
}

/// Parse RDF data into triples and quads
fn parse_rdf_data(
    data: &[u8],
    format: oxirs_core::parser::RdfFormat,
) -> Result<(Vec<oxirs_core::model::Triple>, Vec<oxirs_core::model::Quad>), UploadError> {
    use oxirs_core::parser::Parser;

    let parser = Parser::new(format);
    let quads = parser
        .parse_bytes_to_quads(data)
        .map_err(|e| UploadError::ParseError(format!("Parse error: {}", e)))?;

    // Separate into triples (default graph) and quads (named graphs)
    let mut triples = Vec::new();
    let mut named_quads = Vec::new();

    for quad in quads {
        if quad.is_default_graph() {
            triples.push(quad.to_triple());
        } else {
            named_quads.push(quad);
        }
    }

    Ok((triples, named_quads))
}

/// Insert triples and quads into store
fn insert_data<S: Store>(
    store: &S,
    triples: &[oxirs_core::model::Triple],
    quads: &[oxirs_core::model::Quad],
    target_graph: &oxirs_core::model::GraphName,
) -> Result<usize, UploadError> {
    let mut inserted = 0;

    // Insert triples into target graph
    for triple in triples {
        let quad = oxirs_core::model::Quad::new(
            triple.subject().clone(),
            triple.predicate().clone(),
            triple.object().clone(),
            target_graph.clone(),
        );

        store
            .insert_quad(quad)
            .map_err(|e| UploadError::StoreError(e.to_string()))?;
        inserted += 1;
    }

    // Insert quads (preserve their original graph)
    for quad in quads {
        store
            .insert_quad(quad.clone())
            .map_err(|e| UploadError::StoreError(e.to_string()))?;
        inserted += 1;
    }

    Ok(inserted)
}

/// Server-specific handler for direct upload (works with AppState)
pub async fn handle_upload_server(
    Query(params): Query<UploadParams>,
    State(state): State<Arc<crate::server::AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_upload(
        Query(params),
        State(Arc::new(state.store.clone())),
        headers,
        body,
    )
    .await
    {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

/// Server-specific handler for multipart upload (works with AppState)
pub async fn handle_multipart_upload_server(
    Query(params): Query<UploadParams>,
    State(state): State<Arc<crate::server::AppState>>,
    multipart: Multipart,
) -> Response {
    match handle_multipart_upload(
        Query(params),
        State(Arc::new(state.store.clone())),
        multipart,
    )
    .await
    {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::rdf_store::ConcreteStore;

    #[test]
    fn test_upload_params_default_graph() {
        let params = UploadParams {
            graph: None,
            format: None,
        };

        match params.graph_name() {
            oxirs_core::model::GraphName::DefaultGraph => (),
            _ => panic!("Expected default graph"),
        }
    }

    #[test]
    fn test_parse_format_hint() {
        assert!(matches!(
            parse_format_hint("turtle").unwrap(),
            oxirs_core::parser::RdfFormat::Turtle
        ));
        assert!(matches!(
            parse_format_hint("nt").unwrap(),
            oxirs_core::parser::RdfFormat::NTriples
        ));
    }

    #[test]
    fn test_format_from_media_type() {
        assert!(matches!(
            format_from_media_type("text/turtle"),
            Some(oxirs_core::parser::RdfFormat::Turtle)
        ));
        assert!(matches!(
            format_from_media_type("application/ld+json"),
            Some(oxirs_core::parser::RdfFormat::JsonLd)
        ));
    }

    #[tokio::test]
    async fn test_parse_and_insert() {
        let store = Arc::new(ConcreteStore::new().unwrap());

        let turtle_data = r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:name "Alice" .
            ex:Bob ex:name "Bob" .
        "#;

        let (triples, quads) = parse_rdf_data(
            turtle_data.as_bytes(),
            oxirs_core::parser::RdfFormat::Turtle,
        )
        .unwrap();

        assert!(!triples.is_empty());

        let graph_name = oxirs_core::model::GraphName::DefaultGraph;
        let inserted = insert_data(store.as_ref(), &triples, &quads, &graph_name).unwrap();

        assert_eq!(inserted, triples.len() + quads.len());
    }
}
