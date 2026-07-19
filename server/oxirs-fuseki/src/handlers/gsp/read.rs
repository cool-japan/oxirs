//! GSP Read Operations (GET/HEAD)

use super::content_neg::{negotiate_format, serialize_triples};
use super::target::GraphAccess;
use super::types::{GraphTarget, GspError, GspParams, RdfFormat};
use axum::{
    body::{Body, Bytes},
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use oxirs_core::model::{GraphName, NamedNode, Triple};
use oxirs_core::Store;
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info};

/// Number of triples serialized per streamed chunk for line-based formats.
const STREAM_CHUNK: usize = 1000;

/// Translate a resolved GSP graph target into the store scan pattern used by
/// [`Store::for_each_quad`] / [`Store::find_quads`]: the default graph, a single
/// named graph, or the whole-dataset union (unbound).
fn graph_pattern(target: &GraphTarget) -> Result<Option<GraphName>, GspError> {
    match target {
        GraphTarget::Default => Ok(Some(GraphName::DefaultGraph)),
        GraphTarget::Union => Ok(None),
        GraphTarget::Named(uri) => {
            let node = NamedNode::new(uri)
                .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
            Ok(Some(GraphName::NamedNode(node)))
        }
    }
}

/// Serialize one buffered chunk of triples and hand it to the streaming
/// channel. Returns `false` if the receiver was dropped (client hung up), so
/// the producer can stop early. The buffer is drained regardless.
fn serialize_and_send(
    buf: &mut Vec<Triple>,
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    format: RdfFormat,
) -> bool {
    if buf.is_empty() {
        return true;
    }
    let item = match serialize_triples(buf, format) {
        Ok(s) => Ok(Bytes::from(s)),
        Err(e) => Err(std::io::Error::other(e.to_string())),
    };
    buf.clear();
    tx.blocking_send(item).is_ok()
}

/// Stream a graph to the response body for line-based formats (N-Triples /
/// N-Quads) using the store's streaming [`Store::for_each_quad`] scan.
///
/// A background blocking task scans the store and serializes bounded chunks
/// straight into a bounded channel; the response body drains that channel. The
/// whole graph is therefore never materialized as one `Vec<Triple>` or one
/// large `String` — peak memory is a single chunk. Because the scan holds the
/// store's read lock for its duration (see [`Store::for_each_quad`]), a slow
/// client keeps a read lock open for the download; that yields a consistent
/// snapshot and only blocks concurrent writers, not other readers.
fn stream_line_based<S: Store + Send + Sync + 'static>(
    store: Arc<S>,
    pattern: Option<GraphName>,
    format: RdfFormat,
) -> Body {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    tokio::task::spawn_blocking(move || {
        let mut buf: Vec<Triple> = Vec::with_capacity(STREAM_CHUNK);
        let mut send_open = true;
        let scan = store.for_each_quad(None, None, None, pattern.as_ref(), &mut |quad| {
            if !send_open {
                return;
            }
            buf.push(Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            ));
            if buf.len() >= STREAM_CHUNK {
                send_open = serialize_and_send(&mut buf, &tx, format);
            }
        });
        // Flush the trailing partial chunk (only if the client is still there).
        if send_open {
            let _ = serialize_and_send(&mut buf, &tx, format);
        }
        // Surface a mid-scan store error as a stream error so the body aborts
        // rather than silently truncating.
        if let Err(e) = scan {
            let _ = tx.blocking_send(Err(std::io::Error::other(e.to_string())));
        }
    });
    Body::from_stream(ReceiverStream::new(rx))
}

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

    // 4. Build the store scan pattern for the target graph.
    let pattern = graph_pattern(graph_access.target())?;

    // 5. Build the response body.
    //    - Line-based formats (N-Triples / N-Quads) stream straight from the
    //      store's `for_each_quad` scan; the graph is never collected into one
    //      `Vec<Triple>`. The triple count for the header is obtained by a
    //      lightweight counting scan (O(1) peak memory).
    //    - Document-structured formats (Turtle / RDF-XML / JSON-LD / TriG)
    //      cannot be split, so they are materialized and serialized in one pass
    //      (any serialization error surfaces as a clean HTTP error).
    let (body, triple_count) = if matches!(format, RdfFormat::NTriples | RdfFormat::NQuads) {
        let mut triple_count = 0usize;
        store
            .for_each_quad(None, None, None, pattern.as_ref(), &mut |_quad| {
                triple_count += 1;
            })
            .map_err(|e| GspError::StoreError(e.to_string()))?;
        let body = stream_line_based(store.clone(), pattern, format);
        (body, triple_count)
    } else {
        let triples = graph_access.get_triples(store.as_ref())?;
        let triple_count = triples.len();
        let serialized = serialize_triples(&triples, format)?;
        (Body::from(serialized), triple_count)
    };

    info!(
        "GSP GET: Serving {} triples from {}",
        triple_count,
        graph_access.label()
    );

    let duration = start.elapsed();

    // 6. Build response
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.media_type())
        .header("X-Triples-Count", triple_count.to_string())
        .header("X-Duration-Ms", duration.as_millis().to_string())
        .body(body)
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

    // 4. Count triples via a streaming scan (never materializes the graph).
    let pattern = graph_pattern(graph_access.target())?;
    let mut triple_count = 0usize;
    store
        .for_each_quad(None, None, None, pattern.as_ref(), &mut |_quad| {
            triple_count += 1;
        })
        .map_err(|e| GspError::StoreError(e.to_string()))?;

    info!(
        "GSP HEAD: {} contains {} triples",
        graph_access.label(),
        triple_count
    );

    // 5. Build response (no body)
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.media_type())
        .header("X-Triples-Count", triple_count.to_string())
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

    /// Regression: a GET for a line-based format streams the graph in multiple
    /// chunks (> CHUNK triples) via `Body::from_stream` and every triple is
    /// present in the collected body — no data is dropped by chunked
    /// serialization and the count header is accurate.
    #[tokio::test]
    async fn test_gsp_get_streams_all_ntriples() {
        let store = ConcreteStore::new().expect("create store");
        for i in 0..2500 {
            let s = NamedNode::new(format!("http://example.org/s{i}")).expect("subject");
            let p = NamedNode::new("http://example.org/p").expect("predicate");
            let o = Literal::new_simple_literal(format!("v{i}"));
            store
                .insert_triple(Triple::new(s, p, o))
                .expect("insert triple");
        }
        let store = Arc::new(store);

        let params = Query(GspParams::default_graph());
        let mut headers = HeaderMap::new();
        headers.insert(header::ACCEPT, "application/n-triples".parse().unwrap());

        let response = handle_gsp_get(params, State(store), headers)
            .await
            .expect("gsp get");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("X-Triples-Count")
                .and_then(|v| v.to_str().ok()),
            Some("2500")
        );

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("collect streamed body");
        let text = String::from_utf8(bytes.to_vec()).expect("utf8 body");
        let line_count = text.lines().filter(|l| !l.trim().is_empty()).count();
        assert_eq!(line_count, 2500, "all triples must be streamed");
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
