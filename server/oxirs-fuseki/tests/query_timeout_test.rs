//! HTTP-level enforcement of the SPARQL query timeout and WHERE-keyword
//! omission, driven through the real production handler stack
//! (`build_jena_router` -> `handlers::query_handler_post` ->
//! `handlers::sparql::core::sparql_query` -> `execute_sparql_query`) via
//! `tower::oneshot`, mirroring the wiring convention of
//! `insert_data_multibyte_test.rs` (read for reference, not modified).
//!
//! Two guarantees are pinned here:
//!  1. A runaway query (a cross MINUS between two disjoint high-cardinality
//!     patterns, O(N*N)) is aborted by the per-query `ExecutionBudget` and the
//!     endpoint answers `408 Request Timeout` promptly — it neither hangs nor
//!     returns `500`.
//!  2. `SELECT * { ?s ?p ?o } LIMIT 1` — WHERE keyword omitted — is a valid
//!     query and returns `200`.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use oxirs_core::encoding::percent_encode;
use oxirs_core::rdf_store::ConcreteStore;
use oxirs_fuseki::config::ServerConfig;
use oxirs_fuseki::server::{build_jena_router, build_minimal_app_state};
use oxirs_fuseki::store::Store;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

type Router = axum::Router;

/// Build a router over a fresh in-memory store whose query execution is capped
/// at `max_query_time_secs` seconds (the setting this work makes live).
fn router_with_query_cap(max_query_time_secs: u64) -> Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let mut config = ServerConfig::default();
    config.performance.query_optimization.max_query_time_secs = max_query_time_secs;
    let state = Arc::new(build_minimal_app_state(store, config));
    build_jena_router(state, core_store)
}

/// POST a SPARQL Update through the real form-encoded `/update` endpoint.
async fn post_update(router: &Router, update: &str) -> StatusCode {
    let form_body = format!("update={}", percent_encode(update));
    let req = Request::builder()
        .method("POST")
        .uri("/update")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from(form_body))
        .expect("request");
    router.clone().oneshot(req).await.expect("oneshot").status()
}

/// POST a SPARQL query through the real `/sparql` endpoint, returning
/// (status, body).
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

/// `INSERT DATA` block of `n` triples on predicate `<urn:p>`, so both
/// `?a <urn:p> ?x` and `?b <urn:p> ?y` each match `n` rows.
fn insert_n_triples(n: usize) -> String {
    let mut s = String::from("INSERT DATA {");
    for i in 0..n {
        s.push_str(&format!(" <urn:s{i}> <urn:p> \"v{i}\" ."));
    }
    s.push('}');
    s
}

/// A runaway cross MINUS (disjoint variables => removes nothing, but still
/// O(N*N) inner iterations) must be aborted by the 1 s query budget and answer
/// `408`, not hang and not `500`. N = 6000 => 36M inner iterations, several
/// seconds of work un-budgeted, so the 1 s cap fires with margin regardless of
/// host speed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runaway_cross_minus_times_out_with_408() {
    let router = router_with_query_cap(1);
    assert_eq!(
        post_update(&router, &insert_n_triples(6000)).await,
        StatusCode::OK,
        "seed INSERT DATA must succeed"
    );

    // The dot terminates the leading triple before MINUS (valid SPARQL 1.1);
    // omitting it is a parse error (400), not a runaway.
    let runaway = "SELECT * WHERE { ?a <urn:p> ?x . MINUS { ?b <urn:p> ?y } }";
    let started = Instant::now();
    let (status, _body) = post_query(&router, runaway).await;
    let elapsed = started.elapsed();

    assert_eq!(
        status,
        StatusCode::REQUEST_TIMEOUT,
        "runaway query must be aborted with 408, got {status}"
    );
    // Budget = 1 s, outer safety-net = 1 + grace(5) = 6 s. A generous bound that
    // still proves the query did not hang and did not run to completion.
    assert!(
        elapsed < Duration::from_secs(20),
        "408 must return promptly (took {elapsed:?})"
    );
}

/// `SELECT * { ?s ?p ?o } LIMIT 1` with the WHERE keyword omitted must parse and
/// return 200 with one binding.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn where_omitted_select_returns_200() {
    let router = router_with_query_cap(30);
    assert_eq!(
        post_update(&router, &insert_n_triples(3)).await,
        StatusCode::OK,
        "seed INSERT DATA must succeed"
    );

    let (status, body) = post_query(&router, "SELECT * { ?s ?p ?o } LIMIT 1").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "WHERE-omitted SELECT must return 200, got {status} (body: {body})"
    );
    assert!(
        body.contains("\"bindings\""),
        "200 response must be SPARQL Results JSON with a bindings array, got: {body}"
    );
}
