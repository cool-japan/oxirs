//! SHACL Validation HTTP Handler
//!
//! Provides HTTP endpoint for SHACL validation following Apache Jena Fuseki's pattern.
//!
//! POST /shacl?graph={graph-uri}
//! - Body: SHACL shapes graph (Turtle format)
//! - Returns: SHACL validation report
//!
//! Query parameters:
//! - graph: Target graph URI (or "default"/"union")
//! - target: Optional target node URI for focused validation

use axum::{
    body::Bytes,
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// SHACL validation query parameters
#[derive(Debug, Clone, Deserialize)]
pub struct ShaclParams {
    /// Target graph to validate ("default", "union", or graph URI)
    #[serde(default = "default_graph")]
    pub graph: String,

    /// Optional target node for focused validation
    pub target: Option<String>,
}

fn default_graph() -> String {
    "default".to_string()
}

/// SHACL validation error
#[derive(Debug, thiserror::Error)]
pub enum ShaclError {
    #[error("Graph not found: {0}")]
    GraphNotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl ShaclError {
    fn status_code(&self) -> StatusCode {
        match self {
            ShaclError::GraphNotFound(_) => StatusCode::NOT_FOUND,
            ShaclError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ShaclError::ParseError(_) => StatusCode::BAD_REQUEST,
            ShaclError::ValidationError(_) => StatusCode::BAD_REQUEST,
            ShaclError::StoreError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ShaclError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ShaclError {
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

/// SHACL validation report summary
#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    /// Whether the data conforms to the shapes
    pub conforms: bool,

    /// Number of validation errors
    pub error_count: usize,

    /// Graph validated
    pub graph: String,

    /// Validation duration in milliseconds
    pub duration_ms: u64,

    /// Optional target node
    pub target_node: Option<String>,
}

/// Handle SHACL validation POST request
///
/// POST /shacl?graph={graph-uri}&target={node-uri}
/// Content-Type: text/turtle (SHACL shapes)
///
/// Returns: SHACL validation report in Turtle format
pub async fn handle_shacl_validation<S: Store + Send + Sync + 'static>(
    Query(params): Query<ShaclParams>,
    State(store): State<Arc<S>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ShaclError> {
    let start = Instant::now();

    info!(
        "SHACL validation request: graph={}, target={:?}",
        params.graph, params.target
    );

    // 1. Parse Content-Type (expect Turtle)
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("text/turtle");

    if !content_type.contains("turtle") {
        return Err(ShaclError::BadRequest(format!(
            "Expected text/turtle, got {}",
            content_type
        )));
    }

    // 2. Parse shapes graph from request body
    let shapes_text = std::str::from_utf8(&body)
        .map_err(|e| ShaclError::ParseError(format!("UTF-8 error: {}", e)))?;

    debug!("Parsing SHACL shapes graph ({} bytes)", shapes_text.len());

    let parser = oxirs_core::parser::Parser::new(oxirs_core::parser::RdfFormat::Turtle);
    let shapes_triples: Vec<oxirs_core::model::Triple> = parser
        .parse_str_to_quads(shapes_text)
        .map_err(|e| ShaclError::ParseError(format!("Turtle parse error: {}", e)))?
        .into_iter()
        .map(|quad| quad.to_triple())
        .collect();

    info!("Parsed {} shape triples", shapes_triples.len());

    // 3. Get target data graph
    let data_triples = get_graph_triples(&params.graph, store.as_ref())?;

    info!(
        "Retrieved {} data triples from graph '{}'",
        data_triples.len(),
        params.graph
    );

    // 4. Run SHACL validation
    // TODO: Integrate with oxirs-shacl validator
    // For now, return a placeholder validation report

    let conforms = validate_shapes(&shapes_triples, &data_triples, params.target.as_deref())?;

    let duration = start.elapsed();

    info!(
        "SHACL validation completed: conforms={}, duration={:?}",
        conforms, duration
    );

    // 5. Build validation report
    let report =
        build_validation_report(conforms, &params.graph, params.target.as_deref(), duration);

    // 6. Serialize report as Turtle
    let report_turtle = serialize_validation_report(&report)?;

    // 7. Return response
    let status = if conforms {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };

    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/turtle")
        .header("X-SHACL-Conforms", conforms.to_string())
        .header("X-Duration-Ms", duration.as_millis().to_string())
        .body(axum::body::Body::from(report_turtle))
        .map_err(|e| ShaclError::Internal(format!("Response build error: {}", e)))?)
}

/// Get triples from specified graph
fn get_graph_triples<S: Store>(
    graph_param: &str,
    store: &S,
) -> Result<Vec<oxirs_core::model::Triple>, ShaclError> {
    use oxirs_core::model::GraphName;

    let graph_name = match graph_param {
        "default" => GraphName::DefaultGraph,
        "union" => {
            // Union: get all triples
            return store
                .find_quads(None, None, None, None)
                .map(|quads| {
                    quads
                        .into_iter()
                        .map(|q| {
                            oxirs_core::model::Triple::new(
                                q.subject().clone(),
                                q.predicate().clone(),
                                q.object().clone(),
                            )
                        })
                        .collect()
                })
                .map_err(|e| ShaclError::StoreError(e.to_string()));
        }
        uri => {
            let node = oxirs_core::model::NamedNode::new(uri)
                .map_err(|e| ShaclError::BadRequest(format!("Invalid graph URI: {}", e)))?;
            GraphName::NamedNode(node)
        }
    };

    // Get quads from specified graph
    let quads = store
        .find_quads(None, None, None, Some(&graph_name))
        .map_err(|e| ShaclError::StoreError(e.to_string()))?;

    if quads.is_empty() {
        return Err(ShaclError::GraphNotFound(graph_param.to_string()));
    }

    // Convert quads to triples
    Ok(quads
        .into_iter()
        .map(|q| {
            oxirs_core::model::Triple::new(
                q.subject().clone(),
                q.predicate().clone(),
                q.object().clone(),
            )
        })
        .collect())
}

/// Validate data against shapes
///
/// TODO: Integrate with oxirs-shacl validator
fn validate_shapes(
    shapes_triples: &[oxirs_core::model::Triple],
    data_triples: &[oxirs_core::model::Triple],
    target_node: Option<&str>,
) -> Result<bool, ShaclError> {
    // Placeholder validation logic
    // In production, this should use oxirs-shacl validator

    debug!(
        "Validating {} data triples against {} shape triples",
        data_triples.len(),
        shapes_triples.len()
    );

    if let Some(target) = target_node {
        debug!("Focused validation on target node: {}", target);
    }

    // TODO: Call oxirs-shacl validator
    // For now, return true (conforms)
    Ok(true)
}

/// Build SHACL validation report
fn build_validation_report(
    conforms: bool,
    graph: &str,
    target_node: Option<&str>,
    duration: std::time::Duration,
) -> ValidationSummary {
    ValidationSummary {
        conforms,
        error_count: if conforms { 0 } else { 1 },
        graph: graph.to_string(),
        duration_ms: duration.as_millis() as u64,
        target_node: target_node.map(|s| s.to_string()),
    }
}

/// Serialize validation report as Turtle
fn serialize_validation_report(report: &ValidationSummary) -> Result<String, ShaclError> {
    // Build SHACL validation report in Turtle format
    // Following W3C SHACL specification

    let turtle = format!(
        r#"@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

[]
    a sh:ValidationReport ;
    sh:conforms {} ;
    sh:result [] .
"#,
        if report.conforms { "true" } else { "false" }
    );

    Ok(turtle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::rdf_store::ConcreteStore;

    fn setup_test_store() -> Arc<ConcreteStore> {
        let store = ConcreteStore::new().unwrap();

        // Add test data
        let turtle_data = r#"@prefix ex: <http://example.org/> .
ex:Alice <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ex:Person .
ex:Alice ex:name "Alice" .
ex:Alice ex:age "30" .
"#;

        let triples = oxirs_core::format::turtle::TurtleParser::new().parse_str(turtle_data).unwrap();
        for triple in triples {
            store.insert_triple(triple).unwrap();
        }

        Arc::new(store)
    }

    #[tokio::test]
    async fn test_shacl_params_default() {
        let params = ShaclParams {
            graph: "default".to_string(),
            target: None,
        };

        assert_eq!(params.graph, "default");
        assert!(params.target.is_none());
    }

    #[test]
    fn test_get_graph_triples_default() {
        let store = setup_test_store();
        let result = get_graph_triples("default", store.as_ref());

        assert!(result.is_ok());
        let triples = result.unwrap();
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_validate_shapes_placeholder() {
        let shapes = vec![];
        let data = vec![];

        let result = validate_shapes(&shapes, &data, None);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Placeholder returns true
    }

    #[test]
    fn test_build_validation_report() {
        let report =
            build_validation_report(true, "default", None, std::time::Duration::from_millis(100));

        assert!(report.conforms);
        assert_eq!(report.error_count, 0);
        assert_eq!(report.graph, "default");
    }

    #[test]
    fn test_serialize_validation_report() {
        let report = ValidationSummary {
            conforms: true,
            error_count: 0,
            graph: "default".to_string(),
            duration_ms: 100,
            target_node: None,
        };

        let turtle = serialize_validation_report(&report).unwrap();
        assert!(turtle.contains("sh:ValidationReport"));
        assert!(turtle.contains("sh:conforms true"));
    }
}

/// Server-specific handler that works with AppState
pub async fn handle_shacl_validation_server(
    Query(params): Query<ShaclParams>,
    State(state): State<std::sync::Arc<crate::server::AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_shacl_validation(
        Query(params),
        State(std::sync::Arc::new(state.store.clone())),
        headers,
        body,
    )
    .await
    {
        Ok(response) => response,
        Err(err) => err.into_response(),
    }
}
