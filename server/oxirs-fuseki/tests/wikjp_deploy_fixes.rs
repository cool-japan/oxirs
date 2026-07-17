//! HTTP wire-level regression tests for the sparql.wik.jp deployment fixes
//! (oxirs-fuseki 0.3.4).
//!
//! Each test boots the production SPARQL handler set via
//! [`oxirs_fuseki::server::build_jena_router`] and drives it with real
//! `Request<Body>` values through `tower`'s `oneshot`, so the assertions
//! exercise the exact query/update code path the shipped binary uses.
//!
//! Coverage:
//! * Defect 2 — `read_only` datasets must reject SPARQL UPDATE with HTTP 403
//!   and leave the data untouched.
//! * Defect 4 — `INSERT DATA` with multiple triples on a single line must
//!   insert every triple (not silently drop all but the first), and malformed
//!   data must surface an error instead of a silent zero-row "success".
//! * Defect 3 — the SPARQL query engine must return correct results for
//!   LIMIT/OFFSET/FILTER/ASK (and reject invalid queries with 4xx) instead of
//!   the old silent `200 + empty` fallback. (Aggregation coverage is gated on
//!   the engine wiring; see the aggregation section.)

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use oxirs_fuseki::config::{DatasetConfig, ServerConfig};
use oxirs_fuseki::server::{build_jena_router, build_minimal_app_state};
use oxirs_fuseki::store::Store;
use oxirs_core::rdf_store::ConcreteStore;
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

type Router = axum::Router;

/// Percent-encode a string for an `application/x-www-form-urlencoded` value.
fn pct(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Build a `ServerConfig` whose `default` dataset carries the given
/// `read_only` flag (all other fields are defaults).
fn config_with_default_read_only(read_only: bool) -> ServerConfig {
    let mut config = ServerConfig::default();
    config.datasets.insert(
        "default".to_string(),
        DatasetConfig {
            name: "default".to_string(),
            location: String::new(),
            read_only,
            text_index: None,
            shacl_shapes: vec![],
            services: vec![],
            access_control: None,
            backup: None,
        },
    );
    config
}

/// Build a router over a fresh in-memory store with the given config.
fn router_with_config(config: ServerConfig) -> Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let state = Arc::new(build_minimal_app_state(store, config));
    build_jena_router(state, core_store)
}

/// POST a SPARQL UPDATE (form-urlencoded) and return (status, body).
async fn post_update(router: &Router, update: &str) -> (StatusCode, String) {
    let body = format!("update={}", pct(update));
    let req = Request::builder()
        .method("POST")
        .uri("/update")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from(body))
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// POST a SPARQL query (`application/sparql-query`) and return (status, body).
async fn post_query(router: &Router, query: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/sparql")
        .header("content-type", "application/sparql-query")
        .header("accept", "application/sparql-results+json")
        .body(Body::from(query.to_string()))
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// Number of rows in a SPARQL Results JSON `results.bindings` array.
fn binding_count(json_body: &str) -> usize {
    let v: Value = serde_json::from_str(json_body).unwrap_or(Value::Null);
    v.get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

// 10 sample triples used by several query tests (default graph, N-Triples).
const TEN_TRIPLES: &str = concat!(
    "INSERT DATA {\n",
    "<http://ex/alice> <http://ex/dept> <http://ex/eng> .\n",
    "<http://ex/bob> <http://ex/dept> <http://ex/eng> .\n",
    "<http://ex/carol> <http://ex/dept> <http://ex/sales> .\n",
    "<http://ex/dave> <http://ex/dept> <http://ex/sales> .\n",
    "<http://ex/erin> <http://ex/dept> <http://ex/ops> .\n",
    "<http://ex/alice> <http://ex/age> \"30\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
    "<http://ex/bob> <http://ex/age> \"41\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
    "<http://ex/carol> <http://ex/age> \"37\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
    "<http://ex/dave> <http://ex/age> \"28\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
    "<http://ex/erin> <http://ex/age> \"35\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
    "}"
);

// ---------------------------------------------------------------------------
// Defect 2: read_only enforcement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn read_only_dataset_rejects_update_with_403_and_leaves_data_unchanged() {
    let router = router_with_config(config_with_default_read_only(true));

    let (status, body) = post_update(
        &router,
        "INSERT DATA { <http://ex/s> <http://ex/p> <http://ex/o> . }",
    )
    .await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "read-only dataset must reject UPDATE with 403; got {status}: {body}"
    );

    // Data must be untouched.
    let (qstatus, qbody) = post_query(&router, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }").await;
    assert_eq!(qstatus, StatusCode::OK, "query after rejected update: {qbody}");
    assert_eq!(
        binding_count(&qbody),
        0,
        "no triple must have been written on a read-only dataset; body: {qbody}"
    );
}

#[tokio::test]
async fn writable_dataset_allows_update() {
    let router = router_with_config(config_with_default_read_only(false));

    let (status, body) = post_update(
        &router,
        "INSERT DATA { <http://ex/s> <http://ex/p> <http://ex/o> . }",
    )
    .await;
    assert!(
        status.is_success(),
        "writable dataset must accept UPDATE; got {status}: {body}"
    );

    let (_qs, qbody) = post_query(&router, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }").await;
    assert_eq!(binding_count(&qbody), 1, "inserted triple must be present: {qbody}");
}

// ---------------------------------------------------------------------------
// Defect 4: INSERT DATA line handling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn insert_data_multiple_triples_on_one_line_inserts_all() {
    let router = router_with_config(ServerConfig::default());

    // Two triples on a SINGLE physical line — the old line-by-line parser
    // silently dropped the second.
    let (status, body) = post_update(
        &router,
        "INSERT DATA { <http://ex/a> <http://ex/p> <http://ex/b> . <http://ex/c> <http://ex/p> <http://ex/d> . }",
    )
    .await;
    assert!(status.is_success(), "insert should succeed; got {status}: {body}");

    let (_s, qbody) = post_query(&router, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }").await;
    assert_eq!(
        binding_count(&qbody),
        2,
        "both triples on the single line must be inserted; body: {qbody}"
    );
}

#[tokio::test]
async fn insert_data_malformed_is_not_a_silent_success() {
    let router = router_with_config(ServerConfig::default());

    let (status, body) = post_update(
        &router,
        "INSERT DATA { this is definitely not valid ntriples }",
    )
    .await;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "malformed INSERT DATA must return an error, not a silent success; got {status}: {body}"
    );

    // And nothing must have been written.
    let (_s, qbody) = post_query(&router, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }").await;
    assert_eq!(binding_count(&qbody), 0, "no data should be written: {qbody}");
}

// ---------------------------------------------------------------------------
// Defect 3: correct query results (no silent 200 + empty)
// ---------------------------------------------------------------------------

/// Seed a router with the 10 sample triples and assert the load succeeded.
async fn seeded_router() -> Router {
    let router = router_with_config(ServerConfig::default());
    let (status, body) = post_update(&router, TEN_TRIPLES).await;
    assert!(status.is_success(), "seed failed {status}: {body}");
    // Sanity: a plain SELECT returns all 10.
    let (_s, qbody) = post_query(&router, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }").await;
    assert_eq!(binding_count(&qbody), 10, "seed sanity: {qbody}");
    router
}

/// Extract `results.bindings` as an array from a SPARQL Results JSON body.
fn bindings(json_body: &str) -> Vec<Value> {
    let v: Value = serde_json::from_str(json_body).unwrap_or(Value::Null);
    v.get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
        .cloned()
        .unwrap_or_default()
}

#[tokio::test]
async fn select_count_star_returns_total() {
    let router = seeded_router().await;
    let (status, body) = post_query(&router, "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }").await;
    assert_eq!(status, StatusCode::OK, "count query status: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "COUNT(*) must return one row: {body}");
    let n = rows[0]["n"]["value"].as_str().unwrap_or("");
    assert_eq!(n, "10", "COUNT(*) must equal 10; body: {body}");
}

#[tokio::test]
async fn select_group_by_counts_per_group() {
    let router = seeded_router().await;
    let (status, body) = post_query(
        &router,
        "SELECT ?d (COUNT(?s) AS ?c) WHERE { ?s <http://ex/dept> ?d } GROUP BY ?d",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "group-by status: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 3, "three departments expected; body: {body}");
    let total: i64 = rows
        .iter()
        .map(|r| r["c"]["value"].as_str().unwrap_or("0").parse::<i64>().unwrap_or(0))
        .sum();
    assert_eq!(total, 5, "group counts must sum to the 5 dept triples; body: {body}");
}

#[tokio::test]
async fn select_limit_offset_bounds_rows() {
    let router = seeded_router().await;
    let (status, body) = post_query(
        &router,
        "SELECT ?s WHERE { ?s <http://ex/age> ?a } LIMIT 3 OFFSET 1",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "limit/offset status: {body}");
    assert_eq!(binding_count(&body), 3, "LIMIT 3 OFFSET 1 over 5 rows -> 3; body: {body}");
}

#[tokio::test]
async fn select_filter_is_actually_applied() {
    let router = seeded_router().await;
    // ages: alice=30, bob=41, carol=37, dave=28, erin=35 -> >34: 41,37,35 = 3
    let (status, body) = post_query(
        &router,
        "SELECT ?s ?a WHERE { ?s <http://ex/age> ?a FILTER(?a > 34) }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "filter status: {body}");
    assert_eq!(
        binding_count(&body),
        3,
        "FILTER(?a > 34) must keep only 3 of 5 rows (not all); body: {body}"
    );
}

#[tokio::test]
async fn ask_returns_true_for_present_and_false_for_absent() {
    let router = seeded_router().await;

    let (s1, b1) = post_query(
        &router,
        "ASK { <http://ex/alice> <http://ex/dept> <http://ex/eng> }",
    )
    .await;
    assert_eq!(s1, StatusCode::OK);
    let v1: Value = serde_json::from_str(&b1).unwrap_or(Value::Null);
    assert_eq!(v1["boolean"], Value::Bool(true), "present triple -> true; body: {b1}");

    let (s2, b2) = post_query(
        &router,
        "ASK { <http://ex/alice> <http://ex/dept> <http://ex/sales> }",
    )
    .await;
    assert_eq!(s2, StatusCode::OK);
    let v2: Value = serde_json::from_str(&b2).unwrap_or(Value::Null);
    assert_eq!(v2["boolean"], Value::Bool(false), "absent triple -> false; body: {b2}");
}

#[tokio::test]
async fn invalid_query_returns_4xx_not_silent_empty() {
    let router = seeded_router().await;
    let (status, body) = post_query(&router, "SELECT ¿¿¿ WHERE {").await;
    assert!(
        status.is_client_error(),
        "malformed query must be a 4xx, not 200+empty; got {status}: {body}"
    );
}
