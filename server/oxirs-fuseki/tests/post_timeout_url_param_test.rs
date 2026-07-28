//! POST must honor the SPARQL-protocol `?timeout=` URL parameter (R7.6 gap b).
//!
//! `?timeout=` (seconds) is a *request-line* parameter in the SPARQL 1.1
//! Protocol — it rides the URL query string even on POST, never the body.
//! Previously the routed POST handler (`handlers::query_handler_post`) hardcoded
//! the client timeout to `None`, so `?timeout=` was silently ignored on POST and
//! only the configured `max_query_time_secs` could bound a query.
//!
//! This drives the REAL production stack through `build_jena_router` +
//! `tower::oneshot` (mirroring `query_timeout_test.rs`, read for reference, not
//! modified). The config cap is set HIGH (30 s) so the ONLY thing that can turn
//! a runaway into a prompt `408` is the URL `?timeout=1` being read and applied
//! (`effective = min(requested, config_max)`). With the bug the runaway runs
//! under the 30 s cap and does not return a prompt 408 — the decisive
//! discriminator.

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

/// Router over a fresh in-memory store with a deliberately HIGH query cap, so a
/// prompt 408 can only originate from the client `?timeout=`, not the config.
fn router_with_query_cap(max_query_time_secs: u64) -> Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let mut config = ServerConfig::default();
    config.performance.query_optimization.max_query_time_secs = max_query_time_secs;
    let state = Arc::new(build_minimal_app_state(store, config));
    build_jena_router(state, core_store)
}

/// Seed `n` triples on predicate `<urn:p>` via the real form-encoded `/update`.
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

/// POST a SPARQL query to `uri` (which carries the `?timeout=` under test).
async fn post_query(router: &Router, uri: &str, query: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/sparql-query")
        .header("accept", "application/sparql-results+json")
        .body(Body::from(query.to_string()))
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
    (status, String::from_utf8_lossy(&bytes).to_string())
}

fn insert_n_triples(n: usize) -> String {
    let mut s = String::from("INSERT DATA {");
    for i in 0..n {
        s.push_str(&format!(" <urn:s{i}> <urn:p> \"v{i}\" ."));
    }
    s.push('}');
    s
}

const RUNAWAY: &str = "SELECT * WHERE { ?a <urn:p> ?x . MINUS { ?b <urn:p> ?y } }";

/// POST `?timeout=1` against a 30 s config cap must abort the runaway with a
/// prompt 408 — proving the URL timeout is read on POST and lowers the effective
/// budget below the config ceiling.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn post_url_timeout_is_honored_and_lowers_the_budget() {
    let router = router_with_query_cap(30);
    assert_eq!(
        post_update(&router, &insert_n_triples(6000)).await,
        StatusCode::OK,
        "seed INSERT DATA must succeed"
    );

    let started = Instant::now();
    let (status, body) = post_query(&router, "/sparql?timeout=1", RUNAWAY).await;
    let elapsed = started.elapsed();

    assert_eq!(
        status,
        StatusCode::REQUEST_TIMEOUT,
        "POST ?timeout=1 must abort the runaway with 408 (got {status}, body: {body})"
    );
    // Budget 1 s + grace 5 s = 6 s outer net. A generous 20 s bound still proves
    // the 30 s config cap was NOT what fired (that would be ~30 s+), i.e. the URL
    // timeout was honored.
    assert!(
        elapsed < Duration::from_secs(20),
        "408 from ?timeout=1 must be prompt (took {elapsed:?})"
    );
}

/// Control: the identical runaway with NO `?timeout` and the same 30 s cap must
/// NOT time out at 1 s — it either completes (200) or is bounded only by the
/// 30 s config cap. This pins that the 408 above is attributable to `?timeout=1`
/// and not to some unrelated fast abort.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn post_without_url_timeout_is_not_capped_at_one_second() {
    let router = router_with_query_cap(30);
    assert_eq!(
        post_update(&router, &insert_n_triples(1500)).await,
        StatusCode::OK,
        "seed INSERT DATA must succeed"
    );

    // N=1500 => 2.25M inner iterations: comfortably completes well under the 30 s
    // config cap, so with NO client timeout the answer is a 200, not a 1 s 408.
    let started = Instant::now();
    let (status, _body) = post_query(
        &router,
        "/sparql",
        "SELECT * WHERE { ?a <urn:p> ?x . MINUS { ?b <urn:p> ?y } }",
    )
    .await;
    let elapsed = started.elapsed();

    assert_eq!(
        status,
        StatusCode::OK,
        "runaway with no ?timeout under a 30 s cap must complete with 200, got {status}"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "control query must finish under the config cap (took {elapsed:?})"
    );
}
