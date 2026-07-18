//! HTTP wire-level tests for SPARQL 1.1 FILTER / BIND / SELECT built-in calls
//! on the oxirs-fuseki query path (the sparql.wik.jp deployment target).
//!
//! Before the built-in call fix, a bare keyword such as `lang` / `isIRI` /
//! `regex` was lexed as an empty-prefix `PrefixedName` and lowered to
//! `Function { name: ":lang" }`, which the engine rejected as an unknown
//! function — so `FILTER(lang(?l)="ja")` (the core multilingual-SKOS delivery
//! query for wik.jp) failed with HTTP 500 and `BIND(STR(?s) AS ?x)` silently
//! produced an unbound column.
//!
//! Each test boots the production SPARQL handler set via
//! [`oxirs_fuseki::server::build_jena_router`] and drives it with real
//! `Request<Body>` values through `tower`'s `oneshot`, exercising the exact
//! query code path the shipped binary uses. The multilingual fixture is loaded
//! through the lang-tag-preserving RDF parser (`Store::load_data`), mirroring
//! how wik.jp ingests its SKOS graphs.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use oxirs_core::rdf_store::ConcreteStore;
use oxirs_fuseki::config::ServerConfig;
use oxirs_fuseki::server::{build_jena_router, build_minimal_app_state};
use oxirs_fuseki::store::{RdfSerializationFormat, Store};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

type Router = axum::Router;

const SKOS_PREF_LABEL: &str = "http://www.w3.org/2004/02/skos/core#prefLabel";

/// Multilingual SKOS fixture in N-Triples: two concepts, each with a `@ja` and
/// an `@en` `skos:prefLabel`, plus an IRI-valued `ref` and an `xsd:integer`
/// `age` on `c1`.
///
/// The label lexical forms are ASCII on purpose: the built-in-call behaviour
/// under test is the language-tag / typing distinction (`@ja` vs `@en`,
/// `xsd:integer`), independent of the glyphs, and the oxirs-core N-Triples
/// parser currently panics on multi-byte literals (a separate core-parser
/// defect, out of scope here).
const FIXTURE_NT: &str = concat!(
    "<http://ex/c1> <http://www.w3.org/2004/02/skos/core#prefLabel> \"inu\"@ja .\n",
    "<http://ex/c1> <http://www.w3.org/2004/02/skos/core#prefLabel> \"dog\"@en .\n",
    "<http://ex/c2> <http://www.w3.org/2004/02/skos/core#prefLabel> \"neko\"@ja .\n",
    "<http://ex/c2> <http://www.w3.org/2004/02/skos/core#prefLabel> \"cat\"@en .\n",
    "<http://ex/c1> <http://ex/ref> <http://ex/other> .\n",
    "<http://ex/c1> <http://ex/age> \"30\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
);

/// Build a router over a fresh in-memory store seeded with the multilingual
/// fixture via the lang-tag-preserving RDF loader.
fn seeded_router() -> Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let loaded = store
        .load_data(FIXTURE_NT, RdfSerializationFormat::NTriples, None)
        .expect("fixture must load");
    assert_eq!(loaded, 6, "the fixture has six triples");
    let state = Arc::new(build_minimal_app_state(store, ServerConfig::default()));
    build_jena_router(state, core_store)
}

/// POST a SPARQL query (SPARQL Results JSON accept) and return (status, body).
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
// The headline acceptance criterion: FILTER(lang(?l) = "ja")
// ---------------------------------------------------------------------------

#[tokio::test]
async fn filter_lang_ja_returns_only_japanese_labels() {
    let router = seeded_router();
    let query = concat!(
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n",
        "SELECT ?c ?l WHERE { ?c skos:prefLabel ?l FILTER(lang(?l) = \"ja\") }"
    );
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "FILTER(lang(?l)=\"ja\") must return 200, not the old 500; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        2,
        "exactly the two @ja labels must survive; body: {body}"
    );
    // Every surviving label lexical form is one of the two @ja labels.
    let labels: Vec<&str> = rows
        .iter()
        .filter_map(|r| r["l"]["value"].as_str())
        .collect();
    assert!(
        labels.contains(&"inu") && labels.contains(&"neko"),
        "the @ja rows must be the two Japanese labels; got {labels:?}; body: {body}"
    );
    assert!(
        !labels.iter().any(|l| *l == "dog" || *l == "cat"),
        "no @en label may leak through the ja filter; body: {body}"
    );
}

// ---------------------------------------------------------------------------
// BIND(STR(?s) AS ?x) actually binds (was silently unbound)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bind_str_binds_the_lexical_form() {
    let router = seeded_router();
    let query = "SELECT ?c ?x WHERE { ?c <http://ex/ref> ?o BIND(STR(?c) AS ?x) }";
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "BIND(STR()) must return 200; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "one ref triple; body: {body}");
    assert_eq!(
        rows[0]["x"]["value"].as_str().unwrap_or(""),
        "http://ex/c1",
        "BIND(STR(?c)) must bind ?x to the IRI string (not leave it unbound); body: {body}"
    );
}

// ---------------------------------------------------------------------------
// regex, langMatches, datatype, isIRI
// ---------------------------------------------------------------------------

#[tokio::test]
async fn filter_regex_matches_substring() {
    let router = seeded_router();
    let query =
        format!("SELECT ?l WHERE {{ ?c <{SKOS_PREF_LABEL}> ?l FILTER(regex(?l, \"inu\")) }}");
    let (status, body) = post_query(&router, &query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "regex must return 200; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        1,
        "only the `inu` label matches regex; body: {body}"
    );
}

#[tokio::test]
async fn filter_langmatches_keeps_japanese() {
    let router = seeded_router();
    let query = format!(
        "SELECT ?l WHERE {{ ?c <{SKOS_PREF_LABEL}> ?l FILTER(langMatches(lang(?l), \"ja\")) }}"
    );
    let (status, body) = post_query(&router, &query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "langMatches must return 200; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        2,
        "langMatches(lang(?l), \"ja\") keeps both @ja labels; body: {body}"
    );
}

#[tokio::test]
async fn bind_datatype_of_typed_literal() {
    let router = seeded_router();
    let query = "SELECT ?d WHERE { ?c <http://ex/age> ?a BIND(datatype(?a) AS ?d) }";
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "datatype() must return 200; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "one age triple; body: {body}");
    assert_eq!(
        rows[0]["d"]["value"].as_str().unwrap_or(""),
        "http://www.w3.org/2001/XMLSchema#integer",
        "datatype(?a) must be xsd:integer; body: {body}"
    );
    assert_eq!(
        rows[0]["d"]["type"].as_str().unwrap_or(""),
        "uri",
        "datatype() yields an IRI term; body: {body}"
    );
}

#[tokio::test]
async fn filter_is_iri_keeps_only_iri_objects() {
    let router = seeded_router();
    // Objects across all triples: 4 literal labels + 1 typed literal age + 1 IRI ref.
    let query = "SELECT ?o WHERE { ?c ?p ?o FILTER(isIRI(?o)) }";
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "isIRI must return 200; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(rows.len(), 1, "only the ref object is an IRI; body: {body}");
    assert_eq!(
        rows[0]["o"]["value"].as_str().unwrap_or(""),
        "http://ex/other",
        "the surviving IRI object must be the ref target; body: {body}"
    );
}

// ---------------------------------------------------------------------------
// Round 2: GROUP BY (expr), HAVING(COUNT(*)), bare `a`, lang-literal equality
// ---------------------------------------------------------------------------

/// Fixture for the round-2 wire tests: three concepts (`c1`/`c2` have a `@ja`
/// and a `@en` label, `c3` has one `@en` label), all typed `rdf:type
/// ex:Concept`. Written with full IRIs (N-Triples has no `a` shorthand); the
/// `a` shorthand is exercised on the query side.
const FIXTURE_R2_NT: &str = concat!(
    "<http://ex/c1> <http://www.w3.org/2004/02/skos/core#prefLabel> \"inu\"@ja .\n",
    "<http://ex/c1> <http://www.w3.org/2004/02/skos/core#prefLabel> \"dog\"@en .\n",
    "<http://ex/c2> <http://www.w3.org/2004/02/skos/core#prefLabel> \"neko\"@ja .\n",
    "<http://ex/c2> <http://www.w3.org/2004/02/skos/core#prefLabel> \"cat\"@en .\n",
    "<http://ex/c3> <http://www.w3.org/2004/02/skos/core#prefLabel> \"x\"@en .\n",
    "<http://ex/c1> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex/Concept> .\n",
    "<http://ex/c2> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex/Concept> .\n",
    "<http://ex/c3> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex/Concept> .\n",
);

fn seeded_router_r2() -> Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let loaded = store
        .load_data(FIXTURE_R2_NT, RdfSerializationFormat::NTriples, None)
        .expect("r2 fixture must load");
    assert_eq!(loaded, 8, "the r2 fixture has eight triples");
    let state = Arc::new(build_minimal_app_state(store, ServerConfig::default()));
    build_jena_router(state, core_store)
}

#[tokio::test]
async fn group_by_lang_expression_splits_into_language_groups() {
    let router = seeded_router_r2();
    let query = concat!(
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n",
        "SELECT (COUNT(*) AS ?n) WHERE { ?c skos:prefLabel ?l } GROUP BY (LANG(?l))"
    );
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "GROUP BY (LANG(?l)) must return 200; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        2,
        "GROUP BY (LANG(?l)) must yield 2 groups, not 1 (the silent mis-count); body: {body}"
    );
    let mut counts: Vec<i64> = rows
        .iter()
        .filter_map(|r| r["n"]["value"].as_str().and_then(|s| s.parse().ok()))
        .collect();
    counts.sort();
    assert_eq!(
        counts,
        vec![2, 3],
        "ja=2 (inu,neko) and en=3 (dog,cat,x); body: {body}"
    );
}

#[tokio::test]
async fn having_count_star_filters_groups() {
    let router = seeded_router_r2();
    let query = concat!(
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n",
        "SELECT ?c (COUNT(?l) AS ?n) WHERE { ?c skos:prefLabel ?l } ",
        "GROUP BY ?c HAVING (COUNT(*) > 1)"
    );
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "HAVING (COUNT(*) > 1) must return 200, not 400; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        2,
        "only c1 and c2 (2 labels each) survive; c3 (1 label) is dropped; body: {body}"
    );
}

#[tokio::test]
async fn bare_a_matches_rdf_type() {
    let router = seeded_router_r2();
    let query = "SELECT ?c WHERE { ?c a <http://ex/Concept> }";
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "`?c a <Concept>` must return 200, not the old 400 undefined-prefix error; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        3,
        "all three typed concepts must match the `a` (rdf:type) shorthand; body: {body}"
    );
}

#[tokio::test]
async fn filter_lang_literal_exact_match() {
    let router = seeded_router_r2();
    let query = concat!(
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n",
        "SELECT ?c WHERE { ?c skos:prefLabel ?l FILTER(?l = \"inu\"@ja) }"
    );
    let (status, body) = post_query(&router, query).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a language-tagged literal in a FILTER must return 200, not a 400 `:ja` error; body: {body}"
    );
    let rows = bindings(&body);
    assert_eq!(
        rows.len(),
        1,
        "only c1's inu@ja matches exactly; body: {body}"
    );
    assert_eq!(
        rows[0]["c"]["value"].as_str().unwrap_or(""),
        "http://ex/c1",
        "the exact-match row must be c1; body: {body}"
    );
}
