//! HTTP wire-level tests for the unified oxirs-arq SPARQL query path
//! (oxirs-fuseki 0.4.0).
//!
//! Every query form — SELECT, ASK, CONSTRUCT, DESCRIBE — is parsed once by the
//! real oxirs-arq parser and dispatched on the parsed form (not a substring
//! scan), then executed by the engine against the live store. These tests boot
//! the production SPARQL handler set via
//! [`oxirs_fuseki::server::build_jena_router`] and drive it with real
//! `Request<Body>` values through `tower`'s `oneshot`, exercising the exact
//! code path the shipped binary uses.
//!
//! Coverage:
//! * P1-A — a CONSTRUCT whose literal contains the word "select"/"ask" must
//!   route as CONSTRUCT (the old substring router misrouted it as SELECT).
//! * P1-B — a PREFIX-prologued aggregate SELECT must count correctly (the old
//!   string shim could not handle a prologue).
//! * GROUP BY + HAVING, expression aggregates (`SUM(?a*?b)`).
//! * CONSTRUCT (shorthand + explicit template) and DESCRIBE (with/without
//!   WHERE), serialized as RDF.
//! * `GRAPH <iri>` / `GRAPH ?g` scoping and `FROM` / `FROM NAMED` dataset
//!   construction.
//! * Malformed query -> 4xx (never a silent 200 + empty); SERVICE federation to
//!   an unreachable endpoint fails loud rather than returning 200 + empty.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use oxirs_core::rdf_store::ConcreteStore;
use oxirs_fuseki::config::ServerConfig;
use oxirs_fuseki::server::{build_jena_router, build_minimal_app_state};
use oxirs_fuseki::store::Store;
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

/// Build a router over a fresh in-memory store with a default config.
fn fresh_router() -> Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let state = Arc::new(build_minimal_app_state(store, ServerConfig::default()));
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

/// POST a SPARQL query with a chosen `Accept` header and return (status, ct, body).
async fn post_query_accept(
    router: &Router,
    query: &str,
    accept: &str,
) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/sparql")
        .header("content-type", "application/sparql-query")
        .header("accept", accept)
        .body(Body::from(query.to_string()))
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
    (
        status,
        content_type,
        String::from_utf8_lossy(&bytes).to_string(),
    )
}

/// POST a SPARQL query (SPARQL Results JSON accept) and return (status, body).
async fn post_query(router: &Router, query: &str) -> (StatusCode, String) {
    let (status, _ct, body) =
        post_query_accept(router, query, "application/sparql-results+json").await;
    (status, body)
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

/// Number of rows in a SPARQL Results JSON `results.bindings` array.
fn binding_count(json_body: &str) -> usize {
    bindings(json_body).len()
}

/// Count serialized RDF triples in a Turtle body: lines that begin with a `<`
/// subject and end with a statement terminator (`@prefix` lines are skipped).
fn turtle_triple_count(turtle: &str) -> usize {
    turtle
        .lines()
        .filter(|l| {
            let t = l.trim();
            t.starts_with('<') && t.ends_with('.')
        })
        .count()
}

// 10 default-graph sample triples (5 dept, 5 age) used by several tests.
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

/// Seed a fresh router with the 10 default-graph triples.
async fn seeded_router() -> Router {
    let router = fresh_router();
    let (status, body) = post_update(&router, TEN_TRIPLES).await;
    assert!(status.is_success(), "seed failed {status}: {body}");
    router
}

// ---------------------------------------------------------------------------
// P1-A: query-form routing is by parsed form, not a substring scan
// ---------------------------------------------------------------------------

#[tokio::test]
async fn construct_with_select_and_ask_in_literal_routes_as_construct() {
    let router = seeded_router().await;
    // The object literal contains BOTH "select" and "ask"; the old substring
    // router saw "SELECT" and misrouted this to the SELECT path (empty result).
    let query = concat!(
        "CONSTRUCT { ?s <http://ex/note> \"please select or ask about this\" } ",
        "WHERE { ?s <http://ex/dept> ?d }"
    );
    let (status, ct, body) = post_query_accept(&router, query, "text/turtle").await;
    assert_eq!(status, StatusCode::OK, "construct status: {body}");
    assert!(
        ct.contains("turtle"),
        "expected turtle content-type, got {ct}"
    );
    // One note triple per dept subject (5).
    assert_eq!(
        turtle_triple_count(&body),
        5,
        "CONSTRUCT must emit one note triple per dept subject; body:\n{body}"
    );
    assert!(
        body.contains("<http://ex/note>"),
        "note predicate missing:\n{body}"
    );
    assert!(
        body.contains("<http://ex/alice>"),
        "alice subject missing:\n{body}"
    );
}

#[tokio::test]
async fn ask_with_construct_word_in_iri_routes_as_ask() {
    let router = seeded_router().await;
    // An ASK whose IRI text embeds "construct" must still route as ASK.
    let (status, body) = post_query(
        &router,
        "ASK { <http://ex/alice> <http://ex/construct-like-predicate-unused> ?o }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "ask status: {body}");
    let v: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
    assert_eq!(
        v["boolean"],
        Value::Bool(false),
        "absent predicate -> ASK false (routed as ASK, not empty SELECT); body: {body}"
    );
}

// ---------------------------------------------------------------------------
// P1-B: PREFIX-prologued aggregates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prefixed_count_star_returns_total() {
    let router = seeded_router().await;
    let query = concat!(
        "PREFIX ex: <http://ex/>\n",
        "SELECT (COUNT(*) AS ?n) WHERE { ?s ex:dept ?d }"
    );
    let (status, body) = post_query(&router, query).await;
    assert_eq!(status, StatusCode::OK, "count status: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "COUNT(*) must return one row: {body}");
    assert_eq!(
        rows[0]["n"]["value"].as_str().unwrap_or(""),
        "5",
        "COUNT(*) over 5 dept triples must be 5 (prologue handled); body: {body}"
    );
}

#[tokio::test]
async fn group_by_having_filters_groups() {
    let router = seeded_router().await;
    // eng=2, sales=2, ops=1 -> HAVING(count>1) keeps eng and sales.
    let query = concat!(
        "PREFIX ex: <http://ex/>\n",
        "SELECT ?d (COUNT(?s) AS ?c) WHERE { ?s ex:dept ?d } ",
        "GROUP BY ?d HAVING (COUNT(?s) > 1)"
    );
    let (status, body) = post_query(&router, query).await;
    assert_eq!(status, StatusCode::OK, "group/having status: {body}");
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        2,
        "HAVING(count>1) must keep exactly the 2 groups of size 2; body: {body}"
    );
    for row in &rows {
        assert_eq!(
            row["c"]["value"].as_str().unwrap_or(""),
            "2",
            "each surviving group has count 2; body: {body}"
        );
    }
}

#[tokio::test]
async fn expression_aggregate_sum_of_product() {
    let router = fresh_router();
    let seed = concat!(
        "INSERT DATA {\n",
        "<http://ex/i1> <http://ex/a> \"2\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "<http://ex/i1> <http://ex/b> \"3\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "<http://ex/i2> <http://ex/a> \"4\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "<http://ex/i2> <http://ex/b> \"5\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "}"
    );
    let (status, body) = post_update(&router, seed).await;
    assert!(status.is_success(), "seed failed {status}: {body}");

    // SUM(?a*?b) = 2*3 + 4*5 = 26.
    let (status, body) = post_query(
        &router,
        "SELECT (SUM(?a * ?b) AS ?t) WHERE { ?s <http://ex/a> ?a . ?s <http://ex/b> ?b }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "sum status: {body}");
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "aggregate must return one row: {body}");
    let t: f64 = rows[0]["t"]["value"]
        .as_str()
        .unwrap_or("")
        .parse()
        .unwrap_or(f64::NAN);
    assert_eq!(t, 26.0, "SUM(?a*?b) must equal 26; body: {body}");
}

// ---------------------------------------------------------------------------
// CONSTRUCT and DESCRIBE graph production
// ---------------------------------------------------------------------------

#[tokio::test]
async fn construct_where_shorthand_round_trips() {
    let router = seeded_router().await;
    let (status, ct, body) = post_query_accept(
        &router,
        "CONSTRUCT WHERE { ?s <http://ex/dept> ?d }",
        "text/turtle",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "shorthand status: {body}");
    assert!(ct.contains("turtle"), "content-type: {ct}");
    assert_eq!(
        turtle_triple_count(&body),
        5,
        "shorthand template must reproduce the 5 dept triples; body:\n{body}"
    );
    assert!(
        body.contains("<http://ex/dept>"),
        "dept predicate missing:\n{body}"
    );
}

#[tokio::test]
async fn construct_explicit_template_reshapes() {
    let router = seeded_router().await;
    let (status, _ct, body) = post_query_accept(
        &router,
        "CONSTRUCT { ?s <http://ex/worksIn> ?d } WHERE { ?s <http://ex/dept> ?d }",
        "text/turtle",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "explicit status: {body}");
    assert_eq!(
        turtle_triple_count(&body),
        5,
        "5 reshaped triples; body:\n{body}"
    );
    assert!(
        body.contains("<http://ex/worksIn>"),
        "worksIn predicate missing:\n{body}"
    );
    assert!(
        !body.contains("<http://ex/dept>"),
        "template must not emit dept:\n{body}"
    );
}

#[tokio::test]
async fn construct_empty_template_is_400_not_silent_empty() {
    let router = seeded_router().await;
    let (status, _ct, body) = post_query_accept(
        &router,
        "CONSTRUCT { } WHERE { ?s <http://ex/dept> ?d }",
        "text/turtle",
    )
    .await;
    assert!(
        status.is_client_error(),
        "empty CONSTRUCT template must be a 4xx, not a silent empty graph; got {status}: {body}"
    );
}

#[tokio::test]
async fn describe_iri_without_where_returns_cbd() {
    let router = seeded_router().await;
    let (status, ct, body) =
        post_query_accept(&router, "DESCRIBE <http://ex/alice>", "text/turtle").await;
    assert_eq!(status, StatusCode::OK, "describe status: {body}");
    assert!(ct.contains("turtle"), "content-type: {ct}");
    // alice has exactly two subject triples: dept and age.
    assert_eq!(
        turtle_triple_count(&body),
        2,
        "CBD of alice must be her 2 subject triples; body:\n{body}"
    );
    assert!(body.contains("<http://ex/dept>"), "dept missing:\n{body}");
    assert!(body.contains("<http://ex/age>"), "age missing:\n{body}");
    assert!(
        !body.contains("<http://ex/bob>"),
        "unrelated subject bob must not appear:\n{body}"
    );
}

#[tokio::test]
async fn describe_variable_with_where_binds_and_describes() {
    let router = seeded_router().await;
    let (status, _ct, body) = post_query_accept(
        &router,
        "DESCRIBE ?x WHERE { ?x <http://ex/dept> <http://ex/eng> }",
        "text/turtle",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "describe-where status: {body}");
    // ?x binds to alice and bob; each contributes its 2 subject triples.
    assert_eq!(
        turtle_triple_count(&body),
        4,
        "CBD of {{alice,bob}} must be 4 triples; body:\n{body}"
    );
    assert!(body.contains("<http://ex/alice>"), "alice missing:\n{body}");
    assert!(body.contains("<http://ex/bob>"), "bob missing:\n{body}");
}

#[tokio::test]
async fn describe_star_without_where_is_400() {
    let router = seeded_router().await;
    let (status, _ct, body) = post_query_accept(&router, "DESCRIBE *", "text/turtle").await;
    assert!(
        status.is_client_error(),
        "DESCRIBE * with nothing in scope must be a 4xx, not a silent empty graph; got {status}: {body}"
    );
}

// ---------------------------------------------------------------------------
// GRAPH / FROM / FROM NAMED
// ---------------------------------------------------------------------------

/// Seed a router with one default-graph triple, two triples in <g1> and one in
/// <g2>.
async fn seeded_graphs_router() -> Router {
    let router = fresh_router();
    for stmt in [
        "INSERT DATA { <http://ex/sd> <http://ex/p> <http://ex/od> . }",
        "INSERT DATA { GRAPH <http://ex/g1> { <http://ex/s1> <http://ex/p> <http://ex/o1> . <http://ex/s2> <http://ex/p> <http://ex/o2> . } }",
        "INSERT DATA { GRAPH <http://ex/g2> { <http://ex/s3> <http://ex/p> <http://ex/o3> . } }",
    ] {
        let (status, body) = post_update(&router, stmt).await;
        assert!(status.is_success(), "seed '{stmt}' failed {status}: {body}");
    }
    router
}

#[tokio::test]
async fn plain_bgp_reads_default_graph_only() {
    let router = seeded_graphs_router().await;
    let (status, body) = post_query(&router, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }").await;
    assert_eq!(status, StatusCode::OK, "default status: {body}");
    assert_eq!(
        binding_count(&body),
        1,
        "plain BGP must read only the single default-graph triple; body: {body}"
    );
}

#[tokio::test]
async fn graph_iri_scopes_to_that_graph() {
    let router = seeded_graphs_router().await;
    let (status, body) = post_query(
        &router,
        "SELECT ?s ?p ?o WHERE { GRAPH <http://ex/g1> { ?s ?p ?o } }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "graph-iri status: {body}");
    assert_eq!(
        binding_count(&body),
        2,
        "GRAPH <g1> must return only g1's two triples; body: {body}"
    );
}

#[tokio::test]
async fn graph_variable_enumerates_and_binds() {
    let router = seeded_graphs_router().await;
    let (status, body) = post_query(&router, "SELECT ?g ?s WHERE { GRAPH ?g { ?s ?p ?o } }").await;
    assert_eq!(status, StatusCode::OK, "graph-var status: {body}");
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        3,
        "GRAPH ?g must enumerate all 3 named-graph triples; body: {body}"
    );
    let graphs: Vec<&str> = rows
        .iter()
        .filter_map(|r| r["g"]["value"].as_str())
        .collect();
    assert!(
        graphs.contains(&"http://ex/g1"),
        "?g must bind g1; body: {body}"
    );
    assert!(
        graphs.contains(&"http://ex/g2"),
        "?g must bind g2; body: {body}"
    );
}

#[tokio::test]
async fn from_unions_named_graphs_into_default() {
    let router = seeded_graphs_router().await;
    // FROM <g1> FROM <g2>: default graph becomes g1 ∪ g2 = 3 triples.
    let (status, body) = post_query(&router, "SELECT ?s WHERE { ?s ?p ?o }").await;
    // sanity: without FROM the default graph has 1 triple.
    assert_eq!(binding_count(&body), 1, "sanity default: {body}");
    assert_eq!(status, StatusCode::OK);

    let (status, body) = post_query(
        &router,
        "SELECT ?s FROM <http://ex/g1> FROM <http://ex/g2> WHERE { ?s ?p ?o }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "from-union status: {body}");
    assert_eq!(
        binding_count(&body),
        3,
        "FROM <g1> FROM <g2> must union into a 3-triple default graph; body: {body}"
    );
}

#[tokio::test]
async fn from_named_restricts_visible_graphs() {
    let router = seeded_graphs_router().await;
    // FROM NAMED <g1> only: GRAPH ?g sees g1 (2 triples), not g2.
    let (status, body) = post_query(
        &router,
        "SELECT ?g ?s FROM NAMED <http://ex/g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "from-named status: {body}");
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        2,
        "FROM NAMED <g1> must expose only g1's 2 triples; body: {body}"
    );
    assert!(
        rows.iter()
            .all(|r| r["g"]["value"].as_str() == Some("http://ex/g1")),
        "only g1 may be visible under FROM NAMED <g1>; body: {body}"
    );
}

// ---------------------------------------------------------------------------
// Fail-loud: malformed queries and unexecutable constructs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_query_is_4xx_not_silent_empty() {
    let router = seeded_router().await;
    let (status, body) = post_query(&router, "SELECT ¿¿¿ WHERE {").await;
    assert!(
        status.is_client_error(),
        "malformed query must be a 4xx, not 200+empty; got {status}: {body}"
    );
}

#[tokio::test]
async fn non_grouped_projection_in_aggregate_query_is_400() {
    let router = seeded_router().await;
    // ?s is neither a GROUP BY key nor aggregated -> SPARQL error, not a
    // silently-unbound column.
    let (status, body) = post_query(
        &router,
        "SELECT ?s (COUNT(*) AS ?n) WHERE { ?s <http://ex/dept> ?d }",
    )
    .await;
    assert!(
        status.is_client_error(),
        "ungrouped SELECT variable in an aggregate query must be a 4xx; got {status}: {body}"
    );
}

#[tokio::test]
async fn update_sent_to_query_endpoint_is_rejected() {
    let router = fresh_router();
    let (status, body) = post_query(
        &router,
        "INSERT DATA { <http://ex/s> <http://ex/p> <http://ex/o> }",
    )
    .await;
    assert!(
        status.is_client_error(),
        "a SPARQL UPDATE on the query endpoint must be a 4xx (pointing at /update); got {status}: {body}"
    );
}

/// SERVICE federation to an unreachable endpoint must fail loud (a typed HTTP
/// error), never a silent `200 OK` with an empty body. Marked `#[ignore]`
/// because it performs a real (failing) network connection attempt; run with
/// `cargo nextest run -p oxirs-fuseki -- --ignored` or `--run-ignored all`.
#[tokio::test]
#[ignore = "networked: attempts a real SERVICE connection to an unreachable endpoint"]
async fn service_to_unreachable_endpoint_is_not_silent_empty() {
    let router = seeded_router().await;
    let (status, body) = post_query(
        &router,
        "SELECT * WHERE { SERVICE <http://127.0.0.1:1/sparql> { ?s ?p ?o } }",
    )
    .await;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "SERVICE to an unreachable endpoint must fail loud, not return 200+empty; got {status}: {body}"
    );
}

// ---------------------------------------------------------------------------
// Aggregate arity + HAVING scoping (parse-time validation & fail-loud)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn having_wrong_arity_aggregate_is_400() {
    let router = seeded_router().await;
    // SUM() with zero arguments is an arity error the parser now rejects
    // (parse-time HAVING aggregate-arity validation), so this fails as a 4xx
    // before execution — never a silent 200 with a shrunk result.
    let (status, body) = post_query(
        &router,
        "SELECT ?d (COUNT(?s) AS ?c) WHERE { ?s <http://ex/dept> ?d } \
         GROUP BY ?d HAVING (SUM() > 1)",
    )
    .await;
    assert!(
        status.is_client_error(),
        "HAVING (SUM() > 1) is a parse-time arity error and must be a 4xx; got {status}: {body}"
    );
}

#[tokio::test]
async fn select_projection_wrong_arity_aggregate_is_400() {
    let router = seeded_router().await;
    // Two-argument SUM and zero-argument SUM are both arity errors in the SELECT
    // projection; each must be a 4xx (locks in already-correct behavior).
    let (status, body) = post_query(
        &router,
        "SELECT (SUM(?a, ?b) AS ?t) WHERE { ?s <http://ex/a> ?a . ?s <http://ex/b> ?b }",
    )
    .await;
    assert!(
        status.is_client_error(),
        "SUM(?a, ?b) is an arity error and must be a 4xx; got {status}: {body}"
    );

    let (status, body) = post_query(
        &router,
        "SELECT (SUM() AS ?t) WHERE { ?s <http://ex/dept> ?d }",
    )
    .await;
    assert!(
        status.is_client_error(),
        "SUM() is an arity error and must be a 4xx; got {status}: {body}"
    );
}

#[tokio::test]
async fn unknown_function_in_having_fails_loud() {
    let router = seeded_router().await;
    // FOOBAR is not an implemented function; the arq filter evaluator raises an
    // UnknownFunctionError on the function name (before evaluating its argument),
    // so a HAVING that references it fails loud at execution (5xx) instead of
    // silently dropping rows and returning 200.
    let (status, body) = post_query(
        &router,
        "SELECT ?d (COUNT(?s) AS ?c) WHERE { ?s <http://ex/dept> ?d } \
         GROUP BY ?d HAVING (FOOBAR(?s) > 1)",
    )
    .await;
    assert_ne!(
        status,
        StatusCode::OK,
        "an unknown function in HAVING must not return 200; body: {body}"
    );
    assert!(
        status.is_server_error(),
        "an unknown function in HAVING is an execution error (5xx); got {status}: {body}"
    );
}

#[tokio::test]
async fn having_without_group_by_uses_implicit_single_group() {
    let router = fresh_router();
    // Same fixture as `expression_aggregate_sum_of_product`: SUM(?a*?b) = 26.
    let seed = concat!(
        "INSERT DATA {\n",
        "<http://ex/i1> <http://ex/a> \"2\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "<http://ex/i1> <http://ex/b> \"3\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "<http://ex/i2> <http://ex/a> \"4\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "<http://ex/i2> <http://ex/b> \"5\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
        "}"
    );
    let (status, body) = post_update(&router, seed).await;
    assert!(status.is_success(), "seed failed {status}: {body}");

    // No GROUP BY: SUM(?a*?b)=26 is the single implicit group. A threshold the
    // group clears keeps it (one row); a threshold it fails drops it (zero rows).
    // Both are 200 — an empty aggregate result is a valid answer, not an error.
    let (status, body) = post_query(
        &router,
        "SELECT (SUM(?a * ?b) AS ?t) WHERE { ?s <http://ex/a> ?a . ?s <http://ex/b> ?b } \
         HAVING (SUM(?a * ?b) > 10)",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "keep-threshold status: {body}");
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        1,
        "HAVING(26 > 10) keeps the implicit group: {body}"
    );
    assert_eq!(
        rows[0]["t"]["value"].as_str().unwrap_or(""),
        "26",
        "surviving implicit group has SUM 26: {body}"
    );

    let (status, body) = post_query(
        &router,
        "SELECT (SUM(?a * ?b) AS ?t) WHERE { ?s <http://ex/a> ?a . ?s <http://ex/b> ?b } \
         HAVING (SUM(?a * ?b) > 100)",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "drop-threshold status: {body}");
    assert_eq!(
        binding_count(&body),
        0,
        "HAVING(26 > 100) drops the implicit group -> 200 with zero rows: {body}"
    );
}

// ---------------------------------------------------------------------------
// FILTER scoping inside UNION / nested groups
// ---------------------------------------------------------------------------

#[tokio::test]
async fn union_with_trailing_filter_group_scoped() {
    let router = seeded_router().await;
    // `{ {?s ?p ?o} UNION {?s ?p ?o} FILTER(?p = <age>) }` — the trailing FILTER
    // is scoped to the whole group (the union result), so every returned row must
    // satisfy it. If the filter were dropped or applied to only one branch, dept
    // rows would leak through.
    let (status, body) = post_query(
        &router,
        "SELECT * WHERE { { ?s ?p ?o } UNION { ?s ?p ?o } FILTER(?p = <http://ex/age>) }",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "union+filter status: {body}");
    let rows = bindings(&body);
    assert!(
        !rows.is_empty(),
        "the filtered union must return rows: {body}"
    );
    assert!(
        rows.iter()
            .all(|r| r["p"]["value"].as_str() == Some("http://ex/age")),
        "every row must satisfy the group-scoped FILTER (?p = age); body: {body}"
    );
}

#[tokio::test]
async fn nested_group_filter_not_hoisted() {
    let router = seeded_router().await;
    // `?s ?p ?o . { FILTER(?o = ?o) }` — the FILTER belongs to the nested group
    // and is joined to the BGP (join semantics), not hoisted into the outer
    // group. The query is well-formed and must return 200, not a 4xx/5xx.
    let (status, body) =
        post_query(&router, "SELECT * WHERE { ?s ?p ?o . { FILTER(?o = ?o) } }").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a nested-group FILTER joined to a BGP must return 200; body: {body}"
    );
}

// ---------------------------------------------------------------------------
// DESCRIBE: FROM scoping + symmetric CBD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn describe_honors_from_clause() {
    let router = seeded_graphs_router().await;
    // FROM <g1> makes g1 the default graph, so DESCRIBE <s1> draws s1's CBD from
    // g1 and must include the g1 triple (s1 p o1).
    let (status, ct, body) = post_query_accept(
        &router,
        "DESCRIBE <http://ex/s1> FROM <http://ex/g1>",
        "text/turtle",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "describe-from status: {body}");
    assert!(ct.contains("turtle"), "content-type: {ct}");
    assert!(
        turtle_triple_count(&body) >= 1,
        "DESCRIBE <s1> FROM <g1> must include the g1 triple; body:\n{body}"
    );
    assert!(
        body.contains("<http://ex/s1>"),
        "s1 subject missing:\n{body}"
    );
    assert!(
        body.contains("<http://ex/o1>"),
        "g1 object o1 missing:\n{body}"
    );
}

#[tokio::test]
async fn describe_without_from_reads_default_graph_only() {
    let router = seeded_graphs_router().await;
    // A plain DESCRIBE (no FROM) of the default-graph resource must return exactly
    // its one default-graph triple and must not leak any named-graph data.
    let (status, _ct, body) =
        post_query_accept(&router, "DESCRIBE <http://ex/sd>", "text/turtle").await;
    assert_eq!(status, StatusCode::OK, "describe-default status: {body}");
    assert_eq!(
        turtle_triple_count(&body),
        1,
        "plain DESCRIBE <sd> must be exactly its 1 default-graph triple; body:\n{body}"
    );
    assert!(
        body.contains("<http://ex/sd>"),
        "sd subject missing:\n{body}"
    );
    assert!(
        body.contains("<http://ex/od>"),
        "od object missing:\n{body}"
    );
    for leaked in [
        "<http://ex/o1>",
        "<http://ex/o2>",
        "<http://ex/o3>",
        "<http://ex/s1>",
        "<http://ex/s3>",
    ] {
        assert!(
            !body.contains(leaked),
            "named-graph term {leaked} must not leak into a plain DESCRIBE; body:\n{body}"
        );
    }
}

#[tokio::test]
async fn describe_symmetric_includes_reverse_arcs() {
    let router = fresh_router();
    // `onlyobj` appears solely as an object in the default graph.
    let (status, body) = post_update(
        &router,
        "INSERT DATA { <http://ex/subj> <http://ex/p> <http://ex/onlyobj> . }",
    )
    .await;
    assert!(status.is_success(), "seed failed {status}: {body}");

    // Symmetric CBD: DESCRIBE of an object-only resource must include the incoming
    // arc (subj p onlyobj), so the result is non-empty.
    let (status, _ct, body) =
        post_query_accept(&router, "DESCRIBE <http://ex/onlyobj>", "text/turtle").await;
    assert_eq!(status, StatusCode::OK, "describe-symmetric status: {body}");
    assert!(
        turtle_triple_count(&body) >= 1,
        "symmetric DESCRIBE must include the incoming arc; body:\n{body}"
    );
    assert!(
        body.contains("<http://ex/subj>"),
        "incoming subject subj missing:\n{body}"
    );
    assert!(
        body.contains("<http://ex/onlyobj>"),
        "object onlyobj missing:\n{body}"
    );
}
