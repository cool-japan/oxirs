//! GSP Write Operations (PUT/POST/DELETE)

use super::content_neg::{parse_content_type, parse_triples};
use super::target::GraphAccess;
use super::types::{GraphTarget, GspError, GspParams, GspStats};
use axum::{
    body::{Body, Bytes},
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use oxirs_core::Store;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Handle PUT request - replace entire graph
pub async fn handle_gsp_put<S: Store + Send + Sync + 'static>(
    Query(params): Query<GspParams>,
    State(store): State<Arc<S>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GspError> {
    let start = Instant::now();

    debug!("GSP PUT request: {:?}", params);

    // 1. Parse Content-Type
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok());
    let format = parse_content_type(content_type)?;

    debug!("GSP PUT: Content-Type: {:?}", format);

    // 2. Parse RDF data
    let triples = parse_triples(&body, format)?;

    info!("GSP PUT: Parsed {} triples", triples.len());

    // 3. Determine target graph
    let target = GraphTarget::from_params(&params)?;
    let graph_access = GraphAccess::new(target.clone(), store.as_ref());

    // 4. Check if target is writable
    if !graph_access.is_writable() {
        return Err(GspError::MethodNotAllowed(format!(
            "Cannot write to {}",
            graph_access.label()
        )));
    }

    // 5. Replace graph (atomic operation)
    let count = graph_access.replace_triples(store.as_ref(), triples)?;

    let duration = start.elapsed();

    info!(
        "GSP PUT: Replaced {} with {} triples in {:?}",
        graph_access.label(),
        count,
        duration
    );

    // 6. Build response
    let stats = GspStats {
        triples: count,
        duration_ms: duration.as_millis() as u64,
        graph: graph_access.label(),
        operation: "PUT".to_string(),
    };

    Ok((StatusCode::NO_CONTENT, Json(stats)).into_response())
}

/// Handle POST request - add triples to graph
pub async fn handle_gsp_post<S: Store + Send + Sync + 'static>(
    Query(params): Query<GspParams>,
    State(store): State<Arc<S>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GspError> {
    let start = Instant::now();

    debug!("GSP POST request: {:?}", params);

    // 1. Parse Content-Type
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok());
    let format = parse_content_type(content_type)?;

    debug!("GSP POST: Content-Type: {:?}", format);

    // 2. Parse RDF data
    let triples = parse_triples(&body, format)?;

    info!("GSP POST: Parsed {} triples", triples.len());

    // 3. Determine target graph
    let target = GraphTarget::from_params(&params)?;
    let graph_access = GraphAccess::new(target.clone(), store.as_ref());

    // 4. Check if target is writable
    if !graph_access.is_writable() {
        return Err(GspError::MethodNotAllowed(format!(
            "Cannot write to {}",
            graph_access.label()
        )));
    }

    // 5. Add triples to graph (atomic operation)
    let count = graph_access.add_triples(store.as_ref(), triples)?;

    let duration = start.elapsed();

    info!(
        "GSP POST: Added {} triples to {} in {:?}",
        count,
        graph_access.label(),
        duration
    );

    // 6. Build response
    let stats = GspStats {
        triples: count,
        duration_ms: duration.as_millis() as u64,
        graph: graph_access.label(),
        operation: "POST".to_string(),
    };

    Ok((StatusCode::OK, Json(stats)).into_response())
}

/// Handle DELETE request - delete entire graph
pub async fn handle_gsp_delete<S: Store + Send + Sync + 'static>(
    Query(params): Query<GspParams>,
    State(store): State<Arc<S>>,
) -> Result<Response, GspError> {
    let start = Instant::now();

    debug!("GSP DELETE request: {:?}", params);

    // 1. Determine target graph
    let target = GraphTarget::from_params(&params)?;
    let graph_access = GraphAccess::new(target.clone(), store.as_ref());

    // 2. Check if graph exists
    if !graph_access.exists() {
        info!("GSP DELETE: Graph not found: {}", graph_access.label());
        return Err(GspError::NotFound(graph_access.label()));
    }

    // 3. Check if target is writable
    if !graph_access.is_writable() {
        return Err(GspError::MethodNotAllowed(format!(
            "Cannot delete {}",
            graph_access.label()
        )));
    }

    // 4. Delete graph (atomic operation)
    let count = graph_access.delete_graph(store.as_ref())?;

    let duration = start.elapsed();

    info!(
        "GSP DELETE: Deleted {} triples from {} in {:?}",
        count,
        graph_access.label(),
        duration
    );

    // 5. Build response
    let stats = GspStats {
        triples: count,
        duration_ms: duration.as_millis() as u64,
        graph: graph_access.label(),
        operation: "DELETE".to_string(),
    };

    Ok((StatusCode::OK, Json(stats)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode, Triple};
    use oxirs_core::rdf_store::ConcreteStore;

    fn setup_test_store() -> Arc<ConcreteStore> {
        let store = ConcreteStore::new().unwrap();

        // Add initial test data
        let s = NamedNode::new("http://example.org/s1").unwrap();
        let p = NamedNode::new("http://example.org/p1").unwrap();
        let o = Literal::new_simple_literal("value1");
        let triple = Triple::new(s, p, o);

        store.insert_triple(triple).unwrap();

        Arc::new(store)
    }

    #[tokio::test]
    async fn test_gsp_put_default_graph() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());

        let turtle_data = br#"
            @prefix ex: <http://example.org/> .
            ex:subject ex:predicate "new value" .
        "#;

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "text/turtle".parse().unwrap());

        let body = Bytes::from(&turtle_data[..]);

        let result = handle_gsp_put(params, State(store.clone()), headers, body).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);

        // Verify graph was replaced
        let graph_access = GraphAccess::new(GraphTarget::Default, store.as_ref());
        let triples = graph_access.get_triples(store.as_ref()).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[tokio::test]
    async fn test_gsp_post_add_triples() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());

        let turtle_data = br#"
            @prefix ex: <http://example.org/> .
            ex:subject2 ex:predicate2 "value2" .
        "#;

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "text/turtle".parse().unwrap());

        let body = Bytes::from(&turtle_data[..]);

        // Get initial count
        let graph_access = GraphAccess::new(GraphTarget::Default, store.as_ref());
        let initial_count = graph_access.get_triples(store.as_ref()).unwrap().len();

        let result = handle_gsp_post(params, State(store.clone()), headers, body).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Verify triples were added
        let final_count = graph_access.get_triples(store.as_ref()).unwrap().len();
        assert_eq!(final_count, initial_count + 1);
    }

    #[tokio::test]
    async fn test_gsp_delete_graph() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());

        let result = handle_gsp_delete(params, State(store.clone())).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Verify graph was deleted
        let graph_access = GraphAccess::new(GraphTarget::Default, store.as_ref());
        let triples = graph_access.get_triples(store.as_ref()).unwrap();
        assert_eq!(triples.len(), 0);
    }

    #[tokio::test]
    async fn test_gsp_put_union_graph_error() {
        let store = setup_test_store();
        let params = Query(GspParams {
            graph: Some("union".to_string()),
            default: None,
        });

        let turtle_data = br#"
            @prefix ex: <http://example.org/> .
            ex:subject ex:predicate "value" .
        "#;

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "text/turtle".parse().unwrap());

        let body = Bytes::from(&turtle_data[..]);

        let result = handle_gsp_put(params, State(store), headers, body).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GspError::MethodNotAllowed(_)));
    }

    #[tokio::test]
    async fn test_gsp_put_invalid_content_type() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());

        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            "application/octet-stream".parse().unwrap(),
        );

        let body = Bytes::from("invalid data");

        let result = handle_gsp_put(params, State(store), headers, body).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GspError::UnsupportedMediaType(_)
        ));
    }

    #[tokio::test]
    async fn test_gsp_put_parse_error() {
        let store = setup_test_store();
        let params = Query(GspParams::default_graph());

        let invalid_turtle = br#"
            This is not valid Turtle syntax!!!
        "#;

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "text/turtle".parse().unwrap());

        let body = Bytes::from(&invalid_turtle[..]);

        let result = handle_gsp_put(params, State(store), headers, body).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GspError::ParseError(_)));
    }
}
