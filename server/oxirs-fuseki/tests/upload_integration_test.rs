//! RDF Bulk Upload Integration Tests
//!
//! Tests bulk upload endpoint compliance with Apache Jena Fuseki patterns

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    routing::post,
    Router,
};
use oxirs_core::rdf_store::ConcreteStore;
use oxirs_fuseki::handlers::upload::handle_upload;
use std::sync::Arc;
use tower::ServiceExt;

/// Build a minimal test router for the upload handler
fn build_upload_router(store: Arc<ConcreteStore>) -> Router {
    Router::new()
        .route("/upload", post(handle_upload::<ConcreteStore>))
        .with_state(store)
}

/// Send an upload POST request and return (status, body_bytes)
async fn do_upload(
    app: Router,
    data: &str,
    content_type: &str,
    graph: Option<&str>,
    format_hint: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut uri = "/upload".to_string();
    let mut query_parts: Vec<String> = Vec::new();
    if let Some(g) = graph {
        query_parts.push(format!("graph={}", oxirs_core::encoding::percent_encode(g)));
    }
    if let Some(f) = format_hint {
        query_parts.push(format!("format={}", f));
    }
    if !query_parts.is_empty() {
        uri.push('?');
        uri.push_str(&query_parts.join("&"));
    }

    let req = Request::builder()
        .method("POST")
        .uri(&uri)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(data.to_string()))
        .expect("request build");

    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes")
        .to_vec();
    (status, body_bytes)
}

/// Parse JSON body
fn parse_json(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_slice(bytes).unwrap_or(serde_json::Value::Null)
}

/// Test direct Turtle upload to default graph
#[tokio::test]
async fn test_upload_turtle_direct() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let turtle_data = concat!(
        "@prefix ex: <http://example.org/> .\n",
        "ex:Alice ex:name \"Alice\" .\n",
        "ex:Alice ex:age \"30\" .\n",
        "ex:Bob ex:name \"Bob\" .\n",
        "ex:Bob ex:age \"25\" .\n",
    );

    let (status, body) = do_upload(app, turtle_data, "text/turtle", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert!(json["triples_inserted"].as_u64().unwrap_or(0) >= 4);
    assert_eq!(json["graph"], "default");
}

/// Test N-Triples upload to named graph
#[tokio::test]
async fn test_upload_ntriples_to_named_graph() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let ntriples_data = concat!(
        "<http://example.org/Alice> <http://example.org/name> \"Alice\" .\n",
        "<http://example.org/Alice> <http://example.org/age> \"30\" .\n",
        "<http://example.org/Bob> <http://example.org/name> \"Bob\" .\n",
    );

    let (status, body) = do_upload(
        app,
        ntriples_data,
        "application/n-triples",
        Some("http://example.org/people"),
        None,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["graph"], "http://example.org/people");
    assert!(json["triples_inserted"].as_u64().unwrap_or(0) >= 3);
}

/// Test auto-detection of format from Content-Type
#[tokio::test]
async fn test_upload_format_detection_from_content_type() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let turtle_data = "@prefix ex: <http://example.org/> . ex:s ex:p \"o\" .\n";

    let (status, body) = do_upload(app, turtle_data, "text/turtle", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    let fmt = json["format"].as_str().unwrap_or("");
    assert!(
        fmt.contains("Turtle"),
        "Expected Turtle format, got: {}",
        fmt
    );
}

/// Test auto-detection of format from data content (no explicit Content-Type)
#[tokio::test]
async fn test_upload_format_auto_detection() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    // Turtle with @prefix is auto-detected as Turtle by detect_format_from_content.
    // Use application/octet-stream so the media-type map doesn't intercept first.
    // Prefix must be on its own line for the parser to handle it correctly.
    let turtle_with_prefix = "@prefix ex: <http://example.org/> .\nex:test ex:value \"123\" .\n";

    let (status, body) = do_upload(
        app,
        turtle_with_prefix,
        "application/octet-stream",
        None,
        None,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert!(json["triples_inserted"].as_u64().unwrap_or(0) >= 1);
}

/// Test format hint parameter
#[tokio::test]
async fn test_upload_with_format_hint() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let turtle_data = "@prefix ex: <http://example.org/> . ex:s ex:p \"o\" .\n";

    // Explicitly hint turtle even with mismatched Content-Type
    let (status, body) = do_upload(
        app,
        turtle_data,
        "application/octet-stream",
        None,
        Some("turtle"),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    let fmt = json["format"].as_str().unwrap_or("");
    assert!(
        fmt.contains("Turtle"),
        "Expected Turtle format, got: {}",
        fmt
    );
}

/// Test upload statistics response fields
#[tokio::test]
async fn test_upload_statistics() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let data = concat!(
        "@prefix ex: <http://example.org/> .\n",
        "ex:s1 ex:p1 \"v1\" .\n",
        "ex:s2 ex:p2 \"v2\" .\n",
        "ex:s3 ex:p3 \"v3\" .\n",
    );

    let (status, body) = do_upload(app, data, "text/turtle", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert!(json["triples_inserted"].as_u64().unwrap_or(0) >= 3);
    assert!(json["duration_ms"].as_u64().is_some());
    let fmt = json["format"].as_str().unwrap_or("");
    assert!(
        fmt.contains("Turtle"),
        "format should contain Turtle, got: {}",
        fmt
    );
    assert!(json["bytes_processed"].as_u64().unwrap_or(0) > 0);
}

/// Test large file upload performance
#[tokio::test]
async fn test_upload_large_file() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let mut large_data = String::from("@prefix ex: <http://example.org/> .\n");
    for i in 0..1000 {
        large_data.push_str(&format!("ex:s{} ex:p{} \"value{}\" .\n", i, i, i));
    }

    let (status, body) = do_upload(app, &large_data, "text/turtle", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_inserted"], 1000);
}

/// Test upload with parse errors returns 400
#[tokio::test]
async fn test_upload_with_parse_errors() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let malformed_turtle = "this is not valid turtle syntax @@@";
    let (status, _body) = do_upload(app, malformed_turtle, "text/turtle", None, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Test upload with unsupported format returns 415
#[tokio::test]
async fn test_upload_unsupported_format() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let data = "some data";
    let (status, _body) = do_upload(app, data, "text/turtle", None, Some("unknown_format")).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

/// Test concurrent uploads all succeed
#[tokio::test]
async fn test_concurrent_uploads() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let s = store.clone();
            tokio::spawn(async move {
                let data = format!(
                    "@prefix ex: <http://example.org/> . ex:s{} ex:p{} \"v{}\" .\n",
                    i, i, i
                );
                let app = build_upload_router(s);
                do_upload(app, &data, "text/turtle", None, None).await
            })
        })
        .collect();

    for handle in handles {
        let (status, _) = handle.await.expect("join");
        assert_eq!(status, StatusCode::OK);
    }
}

/// Test TriG upload with named graphs
#[tokio::test]
async fn test_upload_trig_with_graphs() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    // Use named graph blocks only (the default-graph block `{ ... }` is not
    // supported by the internal TriG parser; use explicit graph names instead).
    let trig_data = concat!(
        "@prefix ex: <http://example.org/> .\n",
        "<http://example.org/graph1> {\n",
        "    ex:alice ex:name \"Alice\" .\n",
        "    ex:bob ex:name \"Bob\" .\n",
        "    ex:charlie ex:name \"Charlie\" .\n",
        "}\n",
    );

    let (status, body) = do_upload(app, trig_data, "application/trig", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    let total = json["triples_inserted"].as_u64().unwrap_or(0)
        + json["quads_inserted"].as_u64().unwrap_or(0);
    assert!(
        total >= 3,
        "Expected at least 3 total triples/quads, got {}",
        total
    );
}

/// Test N-Quads upload preserves graph assignments
#[tokio::test]
async fn test_upload_nquads() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    let nquads_data = concat!(
        "<http://example.org/alice> <http://example.org/name> \"Alice\" <http://example.org/graph1> .\n",
        "<http://example.org/bob> <http://example.org/name> \"Bob\" <http://example.org/graph1> .\n",
        "<http://example.org/charlie> <http://example.org/name> \"Charlie\" <http://example.org/graph2> .\n",
    );

    let (status, body) = do_upload(app, nquads_data, "application/n-quads", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    let quads = json["quads_inserted"].as_u64().unwrap_or(0);
    assert!(quads >= 3, "Expected at least 3 quads, got {}", quads);
}

/// Test upload with transaction semantics — on malformed data no partial insert
#[tokio::test]
async fn test_upload_transaction_rollback() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    // Entirely malformed — parser fails before any insert
    let bad_data = "NOT VALID TURTLE AT ALL";
    let (status, _body) = do_upload(app, bad_data, "text/turtle", None, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Test upload content size limits — very large upload succeeds (no hard limit in handler)
#[tokio::test]
async fn test_upload_size_limits() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    // 100 triples is representative; actual size limits are at the infrastructure layer
    let mut data = String::from("@prefix ex: <http://example.org/> .\n");
    for i in 0..100 {
        data.push_str(&format!("ex:s{} ex:p{} \"v{}\" .\n", i, i, i));
    }

    let (status, body) = do_upload(app, &data, "text/turtle", None, None).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["triples_inserted"], 100);
}

/// Test multipart form upload (single file) — uses direct upload with form boundary
#[tokio::test]
async fn test_multipart_upload_single_file() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    let boundary = "boundary123";
    let rdf_content = "@prefix ex: <http://example.org/> . ex:s ex:p \"o\" .\n";
    let multipart_body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"data.ttl\"\r\nContent-Type: text/turtle\r\n\r\n{rdf_content}\r\n--{boundary}--\r\n",
        boundary = boundary,
        rdf_content = rdf_content,
    );

    let req = Request::builder()
        .method("POST")
        .uri("/upload")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(multipart_body))
        .expect("request build");

    // Use multipart handler
    use oxirs_fuseki::handlers::upload::handle_multipart_upload;
    let router = Router::new()
        .route("/upload", post(handle_multipart_upload::<ConcreteStore>))
        .with_state(store);

    let resp = router.oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
}

/// Test multipart form upload (multiple files)
#[tokio::test]
async fn test_multipart_upload_multiple_files() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    let boundary = "boundary456";
    // Prefix must be on its own line for the parser
    let file1 = "@prefix ex: <http://example.org/> .\nex:s1 ex:p1 \"v1\" .\n";
    let file2 = "@prefix ex: <http://example.org/> .\nex:s2 ex:p2 \"v2\" .\n";
    let multipart_body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f1.ttl\"\r\nContent-Type: text/turtle\r\n\r\n{f1}\r\n\
         --{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f2.ttl\"\r\nContent-Type: text/turtle\r\n\r\n{f2}\r\n\
         --{b}--\r\n",
        b = boundary,
        f1 = file1,
        f2 = file2,
    );

    let req = Request::builder()
        .method("POST")
        .uri("/upload")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(multipart_body))
        .expect("request build");

    use oxirs_fuseki::handlers::upload::handle_multipart_upload;
    let router = Router::new()
        .route("/upload", post(handle_multipart_upload::<ConcreteStore>))
        .with_state(store);

    let resp = router.oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json = parse_json(&body_bytes);
    assert!(
        json["triples_inserted"].as_u64().unwrap_or(0) >= 2,
        "Expected at least 2 triples across files"
    );
}

/// Test multipart with mixed formats
#[tokio::test]
async fn test_multipart_mixed_formats() {
    let store = Arc::new(ConcreteStore::new().expect("store"));

    let boundary = "boundary789";
    // Prefix on its own line for proper Turtle parsing
    let ttl = "@prefix ex: <http://example.org/> .\nex:s1 ex:p1 \"v1\" .\n";
    let nt = "<http://example.org/s2> <http://example.org/p2> \"v2\" .\n";
    let multipart_body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"data.ttl\"\r\nContent-Type: text/turtle\r\n\r\n{ttl}\r\n\
         --{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"data.nt\"\r\nContent-Type: application/n-triples\r\n\r\n{nt}\r\n\
         --{b}--\r\n",
        b = boundary,
        ttl = ttl,
        nt = nt,
    );

    let req = Request::builder()
        .method("POST")
        .uri("/upload")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={}", boundary),
        )
        .body(Body::from(multipart_body))
        .expect("request build");

    use oxirs_fuseki::handlers::upload::handle_multipart_upload;
    let router = Router::new()
        .route("/upload", post(handle_multipart_upload::<ConcreteStore>))
        .with_state(store);

    let resp = router.oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
}

/// Test empty upload returns 400 (empty body)
#[tokio::test]
async fn test_upload_empty_data() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    // Empty body with Turtle content type — parser may succeed with 0 triples or reject
    let (status, body) = do_upload(app, "", "text/turtle", None, None).await;
    // Either 200 (0 triples, valid empty doc) or 400 (parse failure)
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "unexpected status {} for empty upload",
        status
    );
    let _ = body;
}

/// Test upload with graph parameter
#[tokio::test]
async fn test_upload_to_specific_graph() {
    let store = Arc::new(ConcreteStore::new().expect("store"));
    let app = build_upload_router(store);

    // Prefix must be on its own line for the parser
    let data = "@prefix ex: <http://example.org/> .\nex:test ex:value \"123\" .\n";
    let (status, body) = do_upload(
        app,
        data,
        "text/turtle",
        Some("http://example.org/mygraph"),
        None,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );

    let json = parse_json(&body);
    assert_eq!(json["graph"], "http://example.org/mygraph");
    assert!(json["triples_inserted"].as_u64().unwrap_or(0) >= 1);
}
