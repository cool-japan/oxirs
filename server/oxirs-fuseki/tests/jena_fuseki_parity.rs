//! Apache Jena Fuseki HTTP wire-level parity test matrix.
//!
//! This integration test boots the OxiRS Fuseki HTTP router via
//! [`oxirs_fuseki::server::build_jena_router`] (the same handler set
//! used in production for SPARQL query/update, the Graph Store
//! Protocol, bulk upload, RDF Patch, and the standard `$/...` admin
//! surface) and replays a canned matrix of HTTP requests captured from
//! Apache Jena Fuseki. Each fixture is compared against the live OxiRS
//! response under a per-fixture *match mode* declaration:
//!
//! * `structural-json` — parse both bodies as JSON and compare
//!   structurally, optionally allowing some keys (e.g.
//!   `execution_time_ms`) to differ.
//! * `structural-xml` — compare as a sequence of element/text tokens
//!   (no namespaces, no whitespace), optionally allowing fields to be
//!   absent in either side.
//! * `csv` / `tsv` — compare the header line plus a sorted body
//!   (tolerant to row order).
//! * `bytes-prefix` — require the response body to start with the
//!   reference body.
//! * `status` — assert only the status code and (optional) Content-Type.
//!
//! Each fixture is also tagged `spec-required` or `implementation-detail`.
//! `spec-required` fixtures fail the test on mismatch; `implementation-detail`
//! mismatches are surfaced as eprintln warnings without failing.
//!
//! See `tests/fixtures/jena-fuseki-ref/` for the fixture corpus.

use axum::body::{to_bytes, Body};
use axum::http::{HeaderName, HeaderValue, Request, StatusCode};
use oxirs_core::rdf_store::ConcreteStore;
use oxirs_fuseki::config::ServerConfig;
use oxirs_fuseki::server::{build_jena_router, build_minimal_app_state};
use oxirs_fuseki::store::Store;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower::ServiceExt;

/// Directory containing the parity fixture corpus.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("jena-fuseki-ref")
}

/// Classification of a parity fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification {
    /// Fixture corresponds to a wire-level requirement of the SPARQL
    /// Protocol / GSP / etc. Mismatch fails the test.
    SpecRequired,
    /// Fixture corresponds to behaviour that is reasonable but not
    /// strictly required by any spec. Mismatch surfaces as a warning.
    ImplDetail,
}

/// How to compare the live response body with the reference body.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MatchMode {
    /// Structural JSON comparison. Path expressions in `ignore_fields`
    /// are excluded from comparison.
    StructuralJson,
    /// Structural XML comparison: token-based, namespace-agnostic.
    StructuralXml,
    /// CSV: header equal, body lines sorted-equal.
    Csv,
    /// TSV: header equal, body lines sorted-equal.
    Tsv,
    /// Reference body is a prefix of the live body.
    BytesPrefix,
    /// Only status (and Content-Type if reference declares one) checked.
    StatusOnly,
}

/// Parsed fixture metadata.
#[derive(Debug)]
struct Fixture {
    /// Display name (file stem).
    name: String,
    /// Endpoint label, free-form (e.g. `sparql_query`, `gsp_get`).
    #[allow(dead_code)]
    endpoint: String,
    classification: Classification,
    match_mode: MatchMode,
    /// JSON path or XML node names to ignore in comparison.
    ignore_fields: Vec<String>,
    /// Optional description for logs.
    description: String,
    request: HttpRequest,
    response: HttpResponse,
}

/// Parsed request from a fixture.
#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

/// Parsed reference response from a fixture.
#[derive(Debug)]
struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

/// Parse a `key: value, value2` style metadata line.
fn parse_meta_line(line: &str) -> Option<(String, String)> {
    let (k, v) = line.split_once(':')?;
    Some((k.trim().to_string(), v.trim().to_string()))
}

fn parse_classification(s: &str) -> Classification {
    match s {
        "spec-required" => Classification::SpecRequired,
        "implementation-detail" | "impl-detail" => Classification::ImplDetail,
        other => panic!("Unknown classification: {other}"),
    }
}

fn parse_match_mode(s: &str) -> MatchMode {
    match s {
        "structural-json" => MatchMode::StructuralJson,
        "structural-xml" => MatchMode::StructuralXml,
        "csv" => MatchMode::Csv,
        "tsv" => MatchMode::Tsv,
        "bytes-prefix" => MatchMode::BytesPrefix,
        "status" | "status-only" => MatchMode::StatusOnly,
        other => panic!("Unknown match_mode: {other}"),
    }
}

/// Load and parse a fixture from disk.
fn load_fixture(path: &Path) -> Fixture {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read fixture {}: {}", path.display(), e));

    // Three sections: meta (head), request, response. Splitter is `===` lines.
    let mut meta_lines: Vec<&str> = Vec::new();
    let mut request_lines: Vec<&str> = Vec::new();
    let mut response_lines: Vec<&str> = Vec::new();

    let mut current = 0u8; // 0 meta, 1 request, 2 response
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("===") {
            if trimmed.contains("request") {
                current = 1;
                continue;
            }
            if trimmed.contains("response") {
                current = 2;
                continue;
            }
        }
        match current {
            0 => meta_lines.push(line),
            1 => request_lines.push(line),
            _ => response_lines.push(line),
        }
    }

    // Parse meta
    let mut endpoint = String::new();
    let mut classification = Classification::SpecRequired;
    let mut match_mode = MatchMode::StructuralJson;
    let mut ignore_fields: Vec<String> = Vec::new();
    let mut description = String::new();
    for line in meta_lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = parse_meta_line(trimmed) {
            match k.as_str() {
                "endpoint" => endpoint = v,
                "classification" => classification = parse_classification(&v),
                "match_mode" => match_mode = parse_match_mode(&v),
                "ignore_fields" => {
                    ignore_fields = v
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                "description" => description = v,
                _ => {}
            }
        }
    }

    let request = parse_request(&request_lines);
    let response = parse_response(&response_lines);

    Fixture {
        name: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("<noname>")
            .to_string(),
        endpoint,
        classification,
        match_mode,
        ignore_fields,
        description,
        request,
        response,
    }
}

/// Parse a request section. The first non-empty line is `METHOD PATH`.
/// Subsequent non-empty lines until the first blank line are headers.
/// Everything after the blank line is the body.
fn parse_request(lines: &[&str]) -> HttpRequest {
    let mut iter = lines.iter().peekable();
    // Skip leading blanks
    while let Some(line) = iter.peek() {
        if line.trim().is_empty() {
            iter.next();
        } else {
            break;
        }
    }
    let request_line = iter
        .next()
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut body_started = false;
    let mut body = String::new();
    for line in iter {
        if !body_started {
            if line.trim().is_empty() {
                body_started = true;
                continue;
            }
            if let Some((k, v)) = line.split_once(':') {
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    HttpRequest {
        method,
        path,
        headers,
        body: body.trim_end_matches('\n').to_string(),
    }
}

/// Parse a response section. The first non-empty line is the status code.
/// Subsequent lines until a blank line are headers.
/// Everything after is the body.
fn parse_response(lines: &[&str]) -> HttpResponse {
    let mut iter = lines.iter().peekable();
    while let Some(line) = iter.peek() {
        if line.trim().is_empty() {
            iter.next();
        } else {
            break;
        }
    }
    let status_line = iter
        .next()
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut body_started = false;
    let mut body = String::new();
    for line in iter {
        if !body_started {
            if line.trim().is_empty() {
                body_started = true;
                continue;
            }
            if let Some((k, v)) = line.split_once(':') {
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    HttpResponse {
        status,
        headers,
        body: body.trim_end_matches('\n').to_string(),
    }
}

/// Convert a fixture request into an axum `Request<Body>`.
fn build_axum_request(req: &HttpRequest) -> Request<Body> {
    let mut builder = Request::builder()
        .method(req.method.as_str())
        .uri(&req.path);
    for (k, v) in &req.headers {
        builder = builder.header(
            HeaderName::from_bytes(k.as_bytes())
                .unwrap_or_else(|_| HeaderName::from_static("x-bad-header")),
            HeaderValue::from_str(v).unwrap_or_else(|_| HeaderValue::from_static("")),
        );
    }
    builder
        .body(Body::from(req.body.clone()))
        .expect("axum request build")
}

/// Build a fresh router with a fresh in-memory store. Each fixture
/// gets isolated state.
fn fresh_router() -> axum::Router {
    let core_store = Arc::new(ConcreteStore::new().expect("concrete store"));
    let store = Store::new().expect("multi-dataset store");
    let config = ServerConfig::default();
    let state = Arc::new(build_minimal_app_state(store, config));
    build_jena_router(state, core_store)
}

/// Strip ignored fields from a JSON value tree. Each entry in
/// `paths` is a dot-separated path like `head.vars` or
/// `results.bindings`.
fn strip_paths(value: &mut Value, paths: &[String]) {
    for path in paths {
        let segments: Vec<&str> = path.split('.').collect();
        strip_at(value, &segments);
    }
}

fn strip_at(value: &mut Value, segments: &[&str]) {
    if segments.is_empty() {
        return;
    }
    match value {
        Value::Object(map) => {
            if segments.len() == 1 {
                map.remove(segments[0]);
            } else if let Some(child) = map.get_mut(segments[0]) {
                strip_at(child, &segments[1..]);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                strip_at(item, segments);
            }
        }
        _ => {}
    }
}

/// Compare two JSON values structurally with paths in `ignore` removed first.
fn compare_json(reference: &str, actual: &str, ignore: &[String]) -> Result<(), String> {
    let mut ref_val: Value = serde_json::from_str(reference)
        .map_err(|e| format!("reference is not valid JSON: {e}\nreference body:\n{reference}"))?;
    let mut act_val: Value = serde_json::from_str(actual)
        .map_err(|e| format!("live is not valid JSON: {e}\nlive body:\n{actual}"))?;
    strip_paths(&mut ref_val, ignore);
    strip_paths(&mut act_val, ignore);
    if ref_val == act_val {
        Ok(())
    } else {
        Err(format!(
            "JSON mismatch.\nreference (after ignore): {}\nlive (after ignore): {}",
            ref_val, act_val
        ))
    }
}

/// Tokenize an XML document into element/text tokens, ignoring whitespace.
fn xml_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => {
                let txt = buf.trim().to_string();
                if !txt.is_empty() {
                    out.push(format!("TEXT:{txt}"));
                }
                buf.clear();
                in_tag = true;
            }
            '>' if in_tag => {
                let tag = buf.trim().to_string();
                if !tag.is_empty() && !tag.starts_with('?') {
                    out.push(format!("TAG:{tag}"));
                }
                buf.clear();
                in_tag = false;
            }
            _ => buf.push(ch),
        }
    }
    out
}

fn compare_xml(reference: &str, actual: &str, _ignore: &[String]) -> Result<(), String> {
    let r = xml_tokens(reference);
    let a = xml_tokens(actual);
    if r == a {
        Ok(())
    } else {
        Err(format!(
            "XML mismatch.\nreference tokens: {r:?}\nlive tokens: {a:?}"
        ))
    }
}

fn compare_csv(reference: &str, actual: &str, sep: char) -> Result<(), String> {
    let split = |s: &str| -> Vec<String> {
        s.lines()
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect()
    };
    let mut r = split(reference);
    let mut a = split(actual);
    let r_header = r.first().cloned().unwrap_or_default();
    let a_header = a.first().cloned().unwrap_or_default();
    if r_header != a_header {
        return Err(format!(
            "Header mismatch.\nreference: {r_header}\nlive: {a_header}\n(separator={sep})"
        ));
    }
    if !r.is_empty() {
        r.remove(0);
    }
    if !a.is_empty() {
        a.remove(0);
    }
    r.sort();
    a.sort();
    if r == a {
        Ok(())
    } else {
        Err(format!(
            "Body mismatch.\nreference body: {r:?}\nlive body: {a:?}"
        ))
    }
}

/// Outcome from running a single parity fixture.
#[derive(Debug)]
enum Outcome {
    /// Fixture matched. The bool is `true` when the fixture is
    /// spec-required (i.e. counts toward the gate), `false` for an
    /// impl-detail success (informational).
    Match { is_spec: bool },
    /// A spec-required fixture failed (must propagate to test failure).
    Fail(String),
    /// An impl-detail fixture mismatched. The mismatch was already
    /// surfaced via `eprintln`; the message is also returned for
    /// counting purposes.
    Warning(String),
}

/// Top-level fixture runner.
async fn run_fixture(fixture: &Fixture) -> Outcome {
    let app = fresh_router();
    let req = build_axum_request(&fixture.request);
    let response = match app.oneshot(req).await {
        Ok(r) => r,
        Err(e) => return classify(fixture, format!("oneshot error: {e}")),
    };

    let status = response.status();
    let headers = response
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect::<HashMap<String, String>>();
    let body_bytes = match to_bytes(response.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => return classify(fixture, format!("body read: {e}")),
    };
    let body_str = String::from_utf8_lossy(&body_bytes).to_string();

    // Status
    if status.as_u16() != fixture.response.status {
        let msg = format!(
            "status mismatch: expected {}, got {}\nlive body: {body_str}",
            fixture.response.status,
            status.as_u16()
        );
        return classify(fixture, msg);
    }

    // Headers (Content-Type and any reference-listed header)
    for (k, v) in &fixture.response.headers {
        let live_value = headers.get(&k.to_lowercase()).cloned();
        let live_value = match live_value {
            Some(s) => s,
            None => {
                let msg = format!("missing response header: {k}");
                return classify(fixture, msg);
            }
        };
        // Content-Type may carry charset etc.; compare the prefix up to ';'.
        let strip_params = |s: &str| s.split(';').next().unwrap_or(s).trim().to_string();
        if k.eq_ignore_ascii_case("content-type") {
            if strip_params(&live_value) != strip_params(v) {
                let msg = format!(
                    "Content-Type mismatch: expected {v}, got {live_value}\nlive body: {body_str}"
                );
                return classify(fixture, msg);
            }
        } else if live_value != *v {
            let msg = format!("header {k} mismatch: expected {v}, got {live_value}");
            return classify(fixture, msg);
        }
    }

    // Body
    let body_check = match fixture.match_mode {
        MatchMode::StatusOnly => Ok(()),
        MatchMode::BytesPrefix => {
            if body_str.starts_with(&fixture.response.body) {
                Ok(())
            } else {
                Err(format!(
                    "body prefix mismatch.\nreference: {}\nlive: {body_str}",
                    fixture.response.body
                ))
            }
        }
        MatchMode::StructuralJson => {
            compare_json(&fixture.response.body, &body_str, &fixture.ignore_fields)
        }
        MatchMode::StructuralXml => {
            compare_xml(&fixture.response.body, &body_str, &fixture.ignore_fields)
        }
        MatchMode::Csv => compare_csv(&fixture.response.body, &body_str, ','),
        MatchMode::Tsv => compare_csv(&fixture.response.body, &body_str, '\t'),
    };

    if let Err(e) = body_check {
        return classify(fixture, e);
    }

    Outcome::Match {
        is_spec: matches!(fixture.classification, Classification::SpecRequired),
    }
}

/// Classify a mismatch according to the fixture's classification:
/// spec-required mismatches become `Outcome::Fail`; impl-detail
/// mismatches log a warning and become `Outcome::Warning`.
fn classify(fixture: &Fixture, msg: String) -> Outcome {
    let label = if fixture.description.is_empty() {
        fixture.name.clone()
    } else {
        format!("{} ({})", fixture.name, fixture.description)
    };
    match fixture.classification {
        Classification::SpecRequired => Outcome::Fail(format!("[spec-required] {label}: {msg}")),
        Classification::ImplDetail => {
            eprintln!("[impl-detail] {label} mismatch (warning only): {msg}");
            Outcome::Warning(format!("{label}: {msg}"))
        }
    }
}

/// Walk the fixture directory and execute every fixture.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jena_fuseki_parity_matrix() {
    let dir = fixtures_dir();
    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {}", dir.display(), e))
        .filter_map(|res| res.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("txt"))
        .collect();
    entries.sort();

    assert!(
        !entries.is_empty(),
        "no parity fixtures found at {}",
        dir.display()
    );

    let mut spec_total = 0;
    let mut spec_pass = 0;
    let mut impl_total = 0;
    let mut impl_warnings: Vec<String> = Vec::new();
    let mut hard_failures: Vec<String> = Vec::new();

    for path in &entries {
        let fixture = load_fixture(path);
        match fixture.classification {
            Classification::SpecRequired => spec_total += 1,
            Classification::ImplDetail => impl_total += 1,
        }

        let outcome = run_fixture(&fixture).await;
        match outcome {
            Outcome::Match { is_spec: true } => spec_pass += 1,
            Outcome::Match { is_spec: false } => { /* impl-detail OK, no counter */ }
            Outcome::Fail(msg) => hard_failures.push(format!("{}: {}", fixture.name, msg)),
            Outcome::Warning(msg) => impl_warnings.push(format!("{}: {}", fixture.name, msg)),
        }
    }

    let pass_rate = if spec_total == 0 {
        1.0
    } else {
        spec_pass as f64 / spec_total as f64
    };
    eprintln!(
        "Jena Fuseki parity matrix: spec-required={}/{} ({:.1}%), impl-detail={} (warnings={})",
        spec_pass,
        spec_total,
        pass_rate * 100.0,
        impl_total,
        impl_warnings.len(),
    );

    if !hard_failures.is_empty() {
        for f in &hard_failures {
            eprintln!("  FAIL: {f}");
        }
        panic!(
            "{} spec-required fixture(s) failed (pass rate {:.2})",
            hard_failures.len(),
            pass_rate
        );
    }

    // Per spec we require >= 0.95 pass rate on spec-required fixtures.
    assert!(
        pass_rate >= 0.95,
        "Jena Fuseki spec-required pass rate {:.2} below 0.95 threshold",
        pass_rate
    );
}
