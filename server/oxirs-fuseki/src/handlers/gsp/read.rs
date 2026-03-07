//! GSP Read Operations (GET/HEAD)

use super::content_neg::{negotiate_format, serialize_triples};
use super::target::GraphAccess;
use super::types::{GraphTarget, GspError, GspParams, GspStats};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use oxirs_core::Store;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// Handle GET request for a graph
pub async fn handle_gsp_get<S: Store + Send + Sync + 'static>(
    Query(params): Query<GspParams>,
    State(store): State<Arc<S>>,
    headers: HeaderMap,
) -> Result<Response, GspError> {
    let start = Instant::now();

    debug!("GSP GET request: {:?}", params);

    // 1. Determine target graph
    let target = GraphTarget::from_params(&params)?;
    let graph_access = GraphAccess::new(target.clone(), store.as_ref());

    // 2. Check if graph exists
    if !graph_access.exists() {
        info!("GSP GET: Graph not found: {}", graph_access.label());
        return Err(GspError::NotFound(graph_access.label()));
    }

    // 3. Content negotiation
    let accept = headers.get(header::ACCEPT).and_then(|h| h.to_str().ok());
    let format = negotiate_format(accept)?;

    debug!("GSP GET: Negotiated format: {:?}", format);

    // 4. Get triples from graph
    let triples = graph_access.get_triples(store.as_ref())?;

    info!(
        "GSP GET: Retrieved {} triples from {}",
        triples.len(),
        graph_access.label()
    );

    // 5. Serialize triples
    let serialized = serialize_triples(&triples, format)?;

    let duration = start.elapsed();

    // 6. Build response
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.media_type())
        .header("X-Triples-Count", triples.len().to_string())
        .header("X-Duration-Ms", duration.as_millis().to_string())
        .body(Body::from(serialized))
        .map_err(|e| GspError::Internal(format!("Response build error: {}", e)))?;

    Ok(response)
}

/// Handle HEAD request for a graph
pub async fn handle_gsp_head<S: Store + Send + Sync + 'static>(
    Query(params): Query<GspParams>,
    State(store): State<Arc<S>>,
    headers: HeaderMap,
) -> Result<Response, GspError> {
    debug!("GSP HEAD request: {:?}", params);

    // 1. Determine target graph
    let target = GraphTarget::from_params(&params)?;
    let graph_access = GraphAccess::new(target.clone(), store.as_ref());

    // 2. Check if graph exists
    if !graph_access.exists() {
        info!("GSP HEAD: Graph not found: {}", graph_access.label());
        return Err(GspError::NotFound(graph_access.label()));
    }

    // 3. Content negotiation
    let accept = headers.get(header::ACCEPT).and_then(|h| h.to_str().ok());
    let format = negotiate_format(accept)?;

    // 4. Get triple count (without serializing)
    let triples = graph_access.get_triples(store.as_ref())?;

    info!(
        "GSP HEAD: {} contains {} triples",
        graph_access.label(),
        triples.len()
    );

    // 5. Build response (no body)
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.media_type())
        .header("X-Triples-Count", triples.len().to_string())
        .body(Body::empty())
        .map_err(|e| GspError::Internal(format!("Response build error: {}", e)))?;

    Ok(response)
}

/// Handle OPTIONS request for GSP
pub async fn handle_gsp_options() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::ALLOW, "GET, HEAD, PUT, POST, DELETE, OPTIONS")
        .header(
            "Accept-Post",
            "text/turtle, application/rdf+xml, application/n-triples, application/ld+json",
        )
        .header(
            "Accept-Put",
            "text/turtle, application/rdf+xml, application/n-triples, application/ld+json",
        )
        .body(Body::empty())
        .expect("empty body response should be valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode, Triple};
    use oxirs_core::rdf_store::ConcreteStore;

    fn setup_test_store() -> Arc<ConcreteStore> {
        let store = ConcreteStore::new().unwrap();

        // Add test data
        let s = NamedNode::new("http://example.org/subject").unwrap();
        let p = NamedNode::new("http://example.org/predicate").unwrap();
        let o = Literal::new_simple_literal("test value");
        let triple = Triple::new(s, p, o);

        store.insert_triple(triple).unwrap();

        Arc::new(store)
    }

    #[tokio::test]
    async fn test_gsp_get_default_graph() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());
        let headers = HeaderMap::new();

        let result = handle_gsp_get(params, State(store), headers).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_gsp_get_nonexistent_graph() {
        let store = setup_test_store();
        let params = Query(GspParams::named_graph("http://example.org/nonexistent"));
        let headers = HeaderMap::new();

        let result = handle_gsp_get(params, State(store), headers).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GspError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_gsp_get_content_negotiation() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());
        let mut headers = HeaderMap::new();
        headers.insert(header::ACCEPT, "application/n-triples".parse().unwrap());

        let result = handle_gsp_get(params, State(store), headers).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let content_type = response.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(content_type, "application/n-triples");
    }

    #[tokio::test]
    async fn test_gsp_head() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());
        let headers = HeaderMap::new();

        let result = handle_gsp_head(params, State(store), headers).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // HEAD should have no body (verified by HTTP protocol semantics)
        // Note: axum::body::Body doesn't expose size_hint() - the framework ensures HEAD has no body
    }
}
