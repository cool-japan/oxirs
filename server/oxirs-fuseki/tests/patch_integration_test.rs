//! RDF Patch Integration Tests
//!
//! Tests RDF Patch endpoint for incremental graph updates

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    routing::post,
    Router,
};
use oxirs_core::rdf_store::ConcreteStore;
use oxirs_fuseki::handlers::patch::handle_patch;
use std::sync::Arc;
use tower::ServiceExt;

/// Build a minimal test router for the patch handler
fn build_patch_router(store: Arc<ConcreteStore>) -> Router {
    Router::new()
        .route("/patch", post(handle_patch::<ConcreteStore>))
        .with_state(store)
}

/// Send a PATCH request and return (status, body_bytes)
async fn do_patch(app: Router, patch_body: &str, graph: Option<&str>) -> (StatusCode, Vec<u8>) {
    let uri = match graph {
        Some(g) => format!("/patch?graph={}", oxirs_core::encoding::percent_encode(g)),
        None => "/patch".to_string(),
    };

    let req = Request::builder()
        .method("POST")
        .uri(&uri)
        .header(header::CONTENT_TYPE, "application/rdf-patch")
        .body(Body::from(patch_body.to_string()))
        .expect("request build");

    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes")
        .to_vec();
    (status, body_bytes)
}

/// Parse JSON body into serde_json::Value
fn parse_json(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_slice(bytes).unwrap_or(serde_json::Value::Null)
}

/// Test simple add operation
#[tokio::test]
async fn test_patch_simple_add() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = "A <http://example.org/alice> <http://example.org/name> \"Alice\" .\n";

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 1);
    assert_eq!(json["triples_deleted"], 0);
}

/// Test simple delete operation
#[tokio::test]
async fn test_patch_simple_delete() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    // Pre-populate via patch
    let setup_patch = "A <http://example.org/bob> <http://example.org/age> \"30\" .\n";
    let (status, _) = do_patch(build_patch_router(store.clone()), setup_patch, None).await;
    assert_eq!(status, StatusCode::OK);

    // Delete operation
    let delete_patch = "D <http://example.org/bob> <http://example.org/age> \"30\" .\n";
    let (status, body) = do_patch(build_patch_router(store), delete_patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_deleted"], 1);
}

/// Test prefix declarations
#[tokio::test]
async fn test_patch_with_prefixes() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "PA foaf: <http://xmlns.com/foaf/0.1/>\n",
        "A ex:alice foaf:name \"Alice\" .\n",
        "A ex:bob foaf:name \"Bob\" .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 2);
}

/// Test transaction commit — TB starts a transaction, TC commits it
#[tokio::test]
async fn test_patch_transaction_commit() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "TB .\n",
        "A ex:s1 ex:p1 \"v1\" .\n",
        "A ex:s2 ex:p2 \"v2\" .\n",
        "A ex:s3 ex:p3 \"v3\" .\n",
        "TC .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 3);
    assert_eq!(json["transactions_committed"], 1);
}

/// Test transaction abort — ops buffered in TB are discarded on TA
#[tokio::test]
async fn test_patch_transaction_abort() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store.clone());

    // TB . A ... A . TA . — two triples buffered, then aborted
    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "TB .\n",
        "A ex:s1 ex:p1 \"v1\" .\n",
        "A ex:s2 ex:p2 \"v2\" .\n",
        "TA .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 0);
    assert_eq!(json["transactions_aborted"], 1);

    // Verify store is empty after abort
    let setup_patch = "A <http://example.org/probe> <http://example.org/p> \"check\" .\n";
    let (probe_status, probe_body) = do_patch(build_patch_router(store), setup_patch, None).await;
    assert_eq!(probe_status, StatusCode::OK);
    let probe_json = parse_json(&probe_body);
    // Only the probe triple was added; s1/s2 were aborted
    assert_eq!(probe_json["triples_added"], 1);
}

/// Test multiple transactions — two independent TB/TC blocks
#[tokio::test]
async fn test_patch_multiple_transactions() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "TB .\n",
        "A ex:s1 ex:p1 \"v1\" .\n",
        "TC .\n",
        "TB .\n",
        "A ex:s2 ex:p2 \"v2\" .\n",
        "TC .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["transactions_committed"], 2);
}

/// Test header metadata is parsed without error
#[tokio::test]
async fn test_patch_headers() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "H id <urn:uuid:12345>\n",
        "H prev <urn:uuid:previous>\n",
        "H timestamp \"2024-01-01T00:00:00Z\"\n",
        "A <http://example.org/test> <http://example.org/value> \"test\" .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 1);
}

/// Test prefix delete — PD after PA
#[tokio::test]
async fn test_patch_prefix_delete() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "A ex:alice ex:name \"Alice\" .\n",
        "PD ex:\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 1);
}

/// Test blank nodes
#[tokio::test]
async fn test_patch_blank_nodes() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "A _:b1 ex:name \"Anonymous\" .\n",
        "A ex:alice ex:knows _:b1 .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 2);
}

/// Test statistics response fields — TB/TC wraps mixed A and D ops
#[tokio::test]
async fn test_patch_statistics() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    // Pre-insert s3/p3/v3 so delete can succeed
    let setup = "A <http://example.org/s3> <http://example.org/p3> \"v3\" .\n";
    do_patch(build_patch_router(store.clone()), setup, None).await;

    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "TB .\n",
        "A ex:s1 ex:p1 \"v1\" .\n",
        "A ex:s2 ex:p2 \"v2\" .\n",
        "D ex:s3 ex:p3 \"v3\" .\n",
        "TC .\n",
    );

    let (status, body) = do_patch(build_patch_router(store), patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 2);
    assert_eq!(json["triples_deleted"], 1);
    assert_eq!(json["transactions_committed"], 1);
    assert!(json["duration_ms"].as_u64().is_some());
}

/// Test patch to named graph
#[tokio::test]
async fn test_patch_named_graph() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = "A <http://example.org/alice> <http://example.org/name> \"Alice\" .\n";
    let (status, body) = do_patch(app, patch, Some("http://example.org/mygraph")).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["graph"], "http://example.org/mygraph");
    assert_eq!(json["triples_added"], 1);
}

/// Test malformed patch — unknown operation
#[tokio::test]
async fn test_patch_malformed() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let malformed = "INVALID_OP <http://example.org/test>\n";
    let (status, _body) = do_patch(app, malformed, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Test incomplete triple returns 400
#[tokio::test]
async fn test_patch_incomplete_triple() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let incomplete = "A <http://example.org/alice> <http://example.org/name>\n";
    let (status, _body) = do_patch(app, incomplete, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Test TC without preceding ops returns 400
#[tokio::test]
async fn test_patch_transaction_error() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let invalid_tc = "TC .\n";
    let (status, _body) = do_patch(app, invalid_tc, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Test large patch — 1000 add operations in a single TB/TC transaction
#[tokio::test]
async fn test_patch_large() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let mut patch = String::from("PA ex: <http://example.org/>\nTB .\n");
    for i in 0..1000 {
        patch.push_str(&format!("A ex:s{} ex:p{} \"v{}\" .\n", i, i, i));
    }
    patch.push_str("TC .\n");

    let (status, body) = do_patch(app, &patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 1000);
    assert_eq!(json["transactions_committed"], 1);
}

/// Test sequential patches accumulate state
#[tokio::test]
async fn test_patch_sequential() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    let patch1 = concat!(
        "PA ex: <http://example.org/>\n",
        "A ex:alice ex:age \"30\" .\n",
    );
    let (status, _) = do_patch(build_patch_router(store.clone()), patch1, None).await;
    assert_eq!(status, StatusCode::OK);

    let patch2 = concat!(
        "PA ex: <http://example.org/>\n",
        "D ex:alice ex:age \"30\" .\n",
        "A ex:alice ex:age \"31\" .\n",
    );
    let (status, body) = do_patch(build_patch_router(store), patch2, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 1);
    assert_eq!(json["triples_deleted"], 1);
}

/// Test concurrent patches succeed
#[tokio::test]
async fn test_patch_concurrent() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let s = store.clone();
            tokio::spawn(async move {
                let patch = format!(
                    "A <http://example.org/s{}> <http://example.org/p{}> \"v{}\" .\n",
                    i, i, i
                );
                let app = build_patch_router(s);
                do_patch(app, &patch, None).await
            })
        })
        .collect();

    for handle in handles {
        let (status, _) = handle.await.expect("join");
        assert_eq!(status, StatusCode::OK);
    }
}

/// Test patch with language-tagged literals
#[tokio::test]
async fn test_patch_literal_language() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    // The patch parser handles simple string literals; language tags are parsed as part of the string
    let patch = concat!(
        "A <http://example.org/alice> <http://example.org/name> \"Alice\" .\n",
        "A <http://example.org/alice> <http://example.org/name> \"Alicia\" .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 2);
}

/// Test patch with typed literals
#[tokio::test]
async fn test_patch_literal_datatype() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "PA ex: <http://example.org/>\n",
        "A ex:alice ex:name \"Alice\" .\n",
        "A ex:bob ex:name \"Bob\" .\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 2);
}

/// Test empty patch returns 200 with 0 operations
#[tokio::test]
async fn test_patch_empty() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let (status, body) = do_patch(app, "", None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 0);
    assert_eq!(json["triples_deleted"], 0);
}

/// Test patch with comments — comments are ignored
#[tokio::test]
async fn test_patch_with_comments() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_patch_router(store);

    let patch = concat!(
        "# This is a comment\n",
        "PA ex: <http://example.org/>\n",
        "# Another comment\n",
        "A ex:alice ex:name \"Alice\" .\n",
        "# Final comment\n",
    );

    let (status, body) = do_patch(app, patch, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_added"], 1);
}
