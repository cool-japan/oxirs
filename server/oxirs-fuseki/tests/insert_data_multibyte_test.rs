//! HTTP-level regression test for the N-Triples data-block parser
//! char-boundary panic (P0) as seen through `INSERT DATA`.
//!
//! `Store::update`'s `INSERT DATA` / `DELETE DATA` path parses each `.`
//! -terminated statement in the data block with
//! `oxirs_core::parser::Parser::new(RdfFormat::NTriples).parse_str_to_quads`
//! (see `server/oxirs-fuseki/src/store/store_new_group.rs::parse_ntriples_document`,
//! called from `parse_data_block`). Before the fix in
//! `core/oxirs-core/src/parser/mod.rs::parse_literal`, any literal in the
//! data block containing a multi-byte UTF-8 character (e.g. Japanese text)
//! made that parser slice a `&str` at a non-char-boundary byte offset and
//! panic — so `INSERT DATA { <s> <p> "日本語"@ja }` crashed instead of
//! inserting the triple.
//!
//! These tests drive the *production* HTTP handlers (not `Store::update`
//! called directly) via `oxirs_fuseki::server::build_jena_router` +
//! `build_minimal_app_state`, using real `Request<Body>` values through
//! `tower`'s `oneshot` — the same wiring convention as
//! `server/oxirs-fuseki/tests/wikjp_filter_builtins.rs` (read for reference,
//! not modified). That file's own fixture comment notes the N-Triples parser
//! "currently panics on multi-byte literals" and deliberately used ASCII-only
//! label text to route around it; this file is the follow-up that exercises
//! exactly that previously-broken path now that it is fixed.
//!
//! `POST /update` is served by the real
//! `handlers::sparql_refactored::update_handler`, which extracts
//! `Form<SparqlUpdateParams>` — i.e. a genuine
//! `application/x-www-form-urlencoded` body, not a raw
//! `application/sparql-update` body. `oxirs_core::encoding::percent_encode`
//! is used to build that form body correctly for arbitrary (including
//! multi-byte) SPARQL Update text.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use oxirs_core::encoding::percent_encode;
use oxirs_core::rdf_store::ConcreteStore;
use oxirs_fuseki::config::ServerConfig;
use oxirs_fuseki::server::{build_jena_router, build_minimal_app_state};
use oxirs_fuseki::store::Store;
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

type Router = axum::Router;

/// Build a router over a fresh, empty in-memory store.
fn fresh_router() -> Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let state = Arc::new(build_minimal_app_state(store, ServerConfig::default()));
    build_jena_router(state, core_store)
}

/// POST a SPARQL Update through the real `/update` form-encoded endpoint
/// (`Form<SparqlUpdateParams>`), returning (status, body).
async fn post_update(router: &Router, update: &str) -> (StatusCode, String) {
    let form_body = format!("update={}", percent_encode(update));
    let req = Request::builder()
        .method("POST")
        .uri("/update")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from(form_body))
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// POST a SPARQL query (SPARQL Results JSON accept) through the real
/// `/sparql` endpoint and return (status, body). Mirrors
/// `wikjp_filter_builtins.rs::post_query`.
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

/// Extract `results.bindings` as an array from a SPARQL Results JSON body.
fn bindings(json_body: &str) -> Vec<Value> {
    let v: Value = serde_json::from_str(json_body).unwrap_or(Value::Null);
    v.get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
        .cloned()
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Sanity check: the form-encoded /update plumbing itself works for plain
// ASCII, before we lean on it for the multi-byte regression checks below.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn insert_data_ascii_literal_smoke_test() {
    let router = fresh_router();
    let (status, body) = post_update(
        &router,
        r#"INSERT DATA { <http://ex/s> <http://ex/p> "hello"@en }"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "ASCII INSERT DATA must succeed; body: {body}"
    );

    let (status, body) = post_query(
        &router,
        "SELECT ?o WHERE { <http://ex/s> <http://ex/p> ?o }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "query must succeed; body: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "one triple expected; body: {body}");
    assert_eq!(rows[0]["o"]["value"].as_str().unwrap_or(""), "hello");
}

// ---------------------------------------------------------------------------
// The headline acceptance criterion: INSERT DATA with a Japanese
// language-tagged literal must not crash, and the language tag must be
// queryable back out via FILTER(lang(?o)="ja").
// ---------------------------------------------------------------------------

#[tokio::test]
async fn insert_data_japanese_language_tagged_literal_does_not_crash_and_is_queryable() {
    let router = fresh_router();

    let (status, body) = post_update(
        &router,
        r#"INSERT DATA { <http://ex/s> <http://ex/p> "日本語"@ja }"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "INSERT DATA with a Japanese @ja literal must not crash and must return 200; body: {body}"
    );

    let (status, body) = post_query(
        &router,
        r#"SELECT ?o WHERE { <http://ex/s> <http://ex/p> ?o FILTER(lang(?o) = "ja") }"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "FILTER(lang(?o)=\"ja\") query must return 200; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        1,
        "the @ja triple must be found via the lang filter; body: {body}"
    );
    assert_eq!(
        rows[0]["o"]["value"].as_str().unwrap_or(""),
        "日本語",
        "the lexical value must round-trip exactly; body: {body}"
    );
    assert_eq!(
        rows[0]["o"]["xml:lang"].as_str().unwrap_or(""),
        "ja",
        "the language tag must be preserved in the SPARQL JSON results; body: {body}"
    );
}

#[tokio::test]
async fn insert_data_japanese_lang_filter_excludes_other_languages() {
    let router = fresh_router();

    let (status, body) = post_update(
        &router,
        concat!(
            "INSERT DATA { ",
            r#"<http://ex/s> <http://ex/p> "日本語"@ja . "#,
            r#"<http://ex/s> <http://ex/p> "english"@en . "#,
            "}"
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "multi-statement INSERT DATA must succeed; body: {body}"
    );

    let (status, body) = post_query(
        &router,
        r#"SELECT ?o WHERE { <http://ex/s> <http://ex/p> ?o FILTER(lang(?o) = "en") }"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        1,
        "only the @en literal must match; body: {body}"
    );
    assert_eq!(rows[0]["o"]["value"].as_str().unwrap_or(""), "english");
}

// ---------------------------------------------------------------------------
// Plain (no language tag) multi-byte literal, and emoji.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn insert_data_japanese_plain_literal_does_not_crash() {
    let router = fresh_router();
    let (status, body) = post_update(
        &router,
        r#"INSERT DATA { <http://ex/s> <http://ex/p> "日本語の値" }"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "plain multi-byte literal must not crash; body: {body}"
    );

    let (status, body) = post_query(
        &router,
        "SELECT ?o WHERE { <http://ex/s> <http://ex/p> ?o }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "body: {body}");
    assert_eq!(rows[0]["o"]["value"].as_str().unwrap_or(""), "日本語の値");
}

#[tokio::test]
async fn insert_data_emoji_literal_does_not_crash() {
    let router = fresh_router();
    let (status, body) = post_update(
        &router,
        r#"INSERT DATA { <http://ex/s> <http://ex/p> "🎉🎊" }"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "emoji literal must not crash; body: {body}"
    );

    let (status, body) = post_query(
        &router,
        "SELECT ?o WHERE { <http://ex/s> <http://ex/p> ?o }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "body: {body}");
    assert_eq!(rows[0]["o"]["value"].as_str().unwrap_or(""), "🎉🎊");
}

// ---------------------------------------------------------------------------
// Multi-byte characters combined with an escaped quote inside the literal —
// the exact combination that made `end_quote_pos` (a char index) diverge
// from the byte offset it was mistakenly used as.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn insert_data_multibyte_with_escaped_quote_does_not_crash() {
    let router = fresh_router();
    let (status, body) = post_update(
        &router,
        r#"INSERT DATA { <http://ex/s> <http://ex/p> "日\"本"@ja }"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "multi-byte literal with an escaped quote must not crash; body: {body}"
    );

    let (status, body) = post_query(
        &router,
        r#"SELECT ?o WHERE { <http://ex/s> <http://ex/p> ?o FILTER(lang(?o) = "ja") }"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "body: {body}");
    assert_eq!(
        rows[0]["o"]["value"].as_str().unwrap_or(""),
        "日\"本",
        "the escaped quote must be preserved as a literal '\"'; body: {body}"
    );
}
