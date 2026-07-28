//! Core SPARQL Protocol implementation
//!
//! This module implements the SPARQL 1.1 Protocol for RDF as defined by W3C:
//! <https://www.w3.org/TR/sparql11-protocol/>
//! With SPARQL 1.2 enhancements and advanced optimizations

use crate::{
    auth::{AuthUser, Permission},
    error::{FusekiError, FusekiResult},
    server::AppState,
};
use axum::{
    extract::{Query, State},
    http::{header::ACCEPT, HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
    Form,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, instrument};

// SPARQL query parsing
use oxirs_arq::query::parse_query;
// Runtime resource budget: enforces the effective query timeout *during*
// evaluation (cooperative wall-time checks inside the engine's hot loops).
use oxirs_arq::query_governor::{ExecutionBudget, ResourceBudget};

/// Extra wall-clock slack, beyond the cooperative query budget, before the
/// outer `tokio::time::timeout` safety net gives up on the blocking task and
/// frees the HTTP response. The engine's own budget should abort at
/// ~`effective`; this grace only matters if a budget checkpoint was missed, so
/// it is deliberately small. It also must stay below
/// `server.request_timeout_secs` (the axum `TimeoutLayer`) so the query budget,
/// not the coarse outer layer, is what normally fires — see the startup warning
/// in `Runtime::build_router`.
pub(crate) const QUERY_TIMEOUT_GRACE_SECS: u64 = 5;

/// Fixed, non-revealing client message for a query-execution task failure.
///
/// The internal cause (a panic payload, which may embed an assertion message, a
/// file path, or a fragment of query/engine state) is logged server-side but
/// never returned to the client — see [`internal_execution_error`].
pub(crate) const INTERNAL_EXECUTION_ERROR_MSG: &str = "internal query execution error";

/// Map a `spawn_blocking` join failure (a panicked query task, or the runtime
/// shutting down) to a client-facing 500 **without leaking the internal cause**.
///
/// `FusekiError::internal`'s `Display` is embedded verbatim in the JSON error
/// body (`to_error_response`), so formatting the raw [`tokio::task::JoinError`]
/// (which carries any captured panic message) into it would expose engine
/// internals to every caller. Instead the full join error is logged at `error`
/// level for operators and the client receives only the fixed
/// [`INTERNAL_EXECUTION_ERROR_MSG`] sentence.
fn internal_execution_error(join_err: &tokio::task::JoinError) -> FusekiError {
    error!(
        error = %join_err,
        "SPARQL query execution task failed (panic or runtime shutdown)"
    );
    FusekiError::internal(INTERNAL_EXECUTION_ERROR_MSG)
}

/// Deserializer that accepts either a single string or a sequence of
/// strings, returning `Some(Vec<String>)` in either case.
///
/// SPARQL Protocol query parameters such as `default-graph-uri` may
/// appear once or multiple times in the URL query string. Plain
/// `serde_urlencoded` decodes a single `key=value` as a string and
/// rejects it when bound to `Vec<String>`. This helper bridges both.
fn deserialize_string_or_seq<'de, D>(de: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct V;

    impl<'de> Visitor<'de> for V {
        type Value = Option<Vec<String>>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a string or a sequence of strings")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(vec![v.to_string()]))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(Some(vec![v]))
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(V)
        }

        fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Self::Value, S::Error> {
            let mut out: Vec<String> = Vec::new();
            while let Some(item) = seq.next_element::<String>()? {
                out.push(item);
            }
            Ok(if out.is_empty() { None } else { Some(out) })
        }
    }

    de.deserialize_any(V)
}

/// SPARQL query parameters for GET requests.
///
/// `default-graph-uri` and `named-graph-uri` MAY appear multiple times in
/// a SPARQL Protocol request (per <https://www.w3.org/TR/sparql11-protocol/>).
/// Use `deserialize_string_or_seq` so a single occurrence is also
/// accepted (which is what `serde_urlencoded` produces for a single
/// `default-graph-uri=foo`).
#[derive(Debug, Deserialize)]
pub struct SparqlQueryParams {
    pub query: Option<String>,
    #[serde(
        rename = "default-graph-uri",
        default,
        deserialize_with = "deserialize_string_or_seq"
    )]
    pub default_graph_uri: Option<Vec<String>>,
    #[serde(
        rename = "named-graph-uri",
        default,
        deserialize_with = "deserialize_string_or_seq"
    )]
    pub named_graph_uri: Option<Vec<String>>,
    pub timeout: Option<u32>,
    pub format: Option<String>,
}

/// SPARQL update parameters for POST requests
#[derive(Debug, Deserialize)]
pub struct SparqlUpdateParams {
    pub update: String,
    #[serde(
        rename = "using-graph-uri",
        default,
        deserialize_with = "deserialize_string_or_seq"
    )]
    pub using_graph_uri: Option<Vec<String>>,
    #[serde(
        rename = "using-named-graph-uri",
        default,
        deserialize_with = "deserialize_string_or_seq"
    )]
    pub using_named_graph_uri: Option<Vec<String>>,
}

/// SPARQL query request body for direct POST
#[derive(Debug, Deserialize)]
pub struct SparqlQueryRequest {
    pub query: String,
    #[serde(
        rename = "default-graph-uri",
        default,
        deserialize_with = "deserialize_string_or_seq"
    )]
    pub default_graph_uri: Option<Vec<String>>,
    #[serde(
        rename = "named-graph-uri",
        default,
        deserialize_with = "deserialize_string_or_seq"
    )]
    pub named_graph_uri: Option<Vec<String>>,
    pub timeout: Option<u32>,
}

/// SPARQL query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query_type: String,
    pub execution_time_ms: u64,
    pub result_count: Option<usize>,
    pub bindings: Option<Vec<HashMap<String, serde_json::Value>>>,
    pub boolean: Option<bool>,
    pub construct_graph: Option<String>,
    pub describe_graph: Option<String>,
}

/// SPARQL update execution result
#[derive(Debug, Serialize)]
pub struct UpdateResult {
    pub success: bool,
    pub execution_time_ms: u64,
    pub operations_count: usize,
    pub affected_triples: Option<usize>,
    pub error_message: Option<String>,
}

/// Query execution context with enhanced features
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub user: Option<AuthUser>,
    pub dataset: String,
    pub timeout: Option<Duration>,
    pub max_results: Option<usize>,
    pub enable_optimizations: bool,
    pub enable_federation: bool,
    pub enable_caching: bool,
    pub request_id: String,
    /// SPARQL 1.1 Protocol `default-graph-uri` values (RDF dataset scoping).
    pub default_graph_uris: Vec<String>,
    /// SPARQL 1.1 Protocol `named-graph-uri` values (RDF dataset scoping).
    pub named_graph_uris: Vec<String>,
}

impl Default for QueryContext {
    fn default() -> Self {
        Self {
            user: None,
            dataset: "default".to_string(),
            timeout: Some(Duration::from_secs(30)),
            max_results: Some(10000),
            enable_optimizations: true,
            enable_federation: true,
            enable_caching: true,
            request_id: uuid::Uuid::new_v4().to_string(),
            default_graph_uris: Vec::new(),
            named_graph_uris: Vec::new(),
        }
    }
}

/// Main SPARQL query endpoint handler
#[instrument(skip(state))]
pub async fn sparql_query(
    Query(params): Query<SparqlQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    let query_string = match params.query {
        Some(q) => q,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "missing_query",
                    "message": "Query parameter 'query' is required"
                })),
            )
                .into_response();
        }
    };

    // Create query context. `context.timeout` carries ONLY the client-requested
    // `?timeout=` (seconds), or `None` when the client did not ask for one — in
    // which case the configured `max_query_time_secs` becomes the effective cap
    // (see `execute_sparql_query`). This intentionally overrides the QueryContext
    // default so "no ?timeout" means "server default", not a hardcoded 30 s.
    let mut context = QueryContext {
        user,
        ..Default::default()
    };
    context.timeout = params.timeout.map(|t| Duration::from_secs(t as u64));
    context.default_graph_uris = params.default_graph_uri.clone().unwrap_or_default();
    context.named_graph_uris = params.named_graph_uri.clone().unwrap_or_default();

    // Execute query
    match execute_sparql_query(&query_string, context, &state).await {
        Ok(result) => {
            let execution_time = start_time.elapsed();

            // Record metrics
            if let Some(metrics) = &state.metrics_service {
                let query_type = if query_string.to_uppercase().contains("SELECT") {
                    "SELECT"
                } else if query_string.to_uppercase().contains("CONSTRUCT") {
                    "CONSTRUCT"
                } else if query_string.to_uppercase().contains("ASK") {
                    "ASK"
                } else if query_string.to_uppercase().contains("DESCRIBE") {
                    "DESCRIBE"
                } else {
                    "UNKNOWN"
                };
                metrics
                    .record_sparql_query(execution_time, true, query_type)
                    .await;
            }

            // Determine response format based on Accept header
            let accept_header = headers
                .get(ACCEPT)
                .and_then(|h| h.to_str().ok())
                .unwrap_or("application/sparql-results+json");

            format_query_response(result, accept_header)
        }
        Err(e) => {
            let execution_time = start_time.elapsed();

            // Record metrics for failed query
            if let Some(metrics) = &state.metrics_service {
                let query_type = if query_string.to_uppercase().contains("SELECT") {
                    "SELECT"
                } else if query_string.to_uppercase().contains("CONSTRUCT") {
                    "CONSTRUCT"
                } else if query_string.to_uppercase().contains("ASK") {
                    "ASK"
                } else if query_string.to_uppercase().contains("DESCRIBE") {
                    "DESCRIBE"
                } else {
                    "UNKNOWN"
                };
                metrics
                    .record_sparql_query(execution_time, false, query_type)
                    .await;
            }

            error!("SPARQL query execution failed: {}", e);
            (
                e.status_code(),
                Json(serde_json::json!({
                    "error": e.error_type(),
                    "message": e.to_string()
                })),
            )
                .into_response()
        }
    }
}

/// SPARQL query POST endpoint handler (for form data and direct body)
///
/// `Query(url_params)` exposes the request-line query string so the SPARQL
/// protocol's `?timeout=` (seconds) is read identically to GET — it is a URL
/// parameter even on POST, never part of the body — and is capped the same way
/// (`min(config max)`) in `execute_sparql_query`.
#[instrument(skip(state))]
pub async fn sparql_query_post(
    Query(url_params): Query<SparqlQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let start_time = Instant::now();

    // Determine how to extract the query based on content type
    let content_type = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    let params = if content_type.contains("application/x-www-form-urlencoded") {
        // POST with form data - parse manually from body
        let body_str = String::from_utf8_lossy(&body);
        let mut query = None;
        let mut default_graph_uri = None;
        let mut named_graph_uri = None;

        for part in body_str.split('&') {
            if let Some((key, value)) = part.split_once('=') {
                // application/x-www-form-urlencoded encodes spaces as `+`; decode
                // `+`→space BEFORE percent-decoding so a literal plus (`%2B`)
                // survives. Without this, `query=SELECT+?s+WHERE...` reaches the
                // SPARQL lexer with `+` intact and fails ("found Plus").
                let decoded_value = oxirs_core::encoding::percent_decode(&value.replace('+', " "))
                    .unwrap_or_default()
                    .to_string();
                match key {
                    "query" => query = Some(decoded_value),
                    "default-graph-uri" => {
                        default_graph_uri = Some(vec![decoded_value]);
                    }
                    "named-graph-uri" => {
                        named_graph_uri = Some(vec![decoded_value]);
                    }
                    _ => {}
                }
            }
        }

        SparqlQueryParams {
            query,
            default_graph_uri,
            named_graph_uri,
            timeout: None,
            format: None,
        }
    } else if content_type.contains("application/sparql-query") {
        // POST with SPARQL query directly in body
        let query_string = String::from_utf8_lossy(&body).to_string();
        SparqlQueryParams {
            query: Some(query_string),
            default_graph_uri: None,
            named_graph_uri: None,
            timeout: None,
            format: None,
        }
    } else {
        // Default case - no query found
        SparqlQueryParams {
            query: None,
            default_graph_uri: None,
            named_graph_uri: None,
            timeout: None,
            format: None,
        }
    };

    let query_string = match params.query {
        Some(q) => q,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "missing_query",
                    "message": "Query parameter 'query' is required"
                })),
            )
                .into_response();
        }
    };

    // Create query context. The client `?timeout=` (if any) rides the URL query
    // string on POST just as on GET; `None` means "no client cap", leaving the
    // configured `max_query_time_secs` as the effective ceiling in
    // `execute_sparql_query` (rather than the QueryContext default of 30 s).
    let mut context = QueryContext {
        user,
        ..Default::default()
    };
    context.timeout = url_params.timeout.map(|t| Duration::from_secs(t as u64));
    // Dataset scoping may arrive in the form body (params) or, for a direct
    // `application/sparql-query` body, on the URL (url_params); prefer the body
    // values and fall back to the URL.
    context.default_graph_uris = params
        .default_graph_uri
        .clone()
        .or_else(|| url_params.default_graph_uri.clone())
        .unwrap_or_default();
    context.named_graph_uris = params
        .named_graph_uri
        .clone()
        .or_else(|| url_params.named_graph_uri.clone())
        .unwrap_or_default();

    // Execute the query using the same logic as GET
    match execute_sparql_query(&query_string, context, &state).await {
        Ok(result) => {
            let _execution_time = start_time.elapsed().as_millis() as u64;

            // Determine response format based on Accept header
            let accept_header = headers
                .get(ACCEPT)
                .and_then(|h| h.to_str().ok())
                .unwrap_or("application/sparql-results+json");

            format_query_response(result, accept_header)
        }
        Err(e) => {
            let _execution_time = start_time.elapsed().as_millis() as u64;

            error!("SPARQL query execution failed: {}", e);
            (
                e.status_code(),
                Json(serde_json::json!({
                    "error": e.error_type(),
                    "message": e.to_string()
                })),
            )
                .into_response()
        }
    }
}

/// Main SPARQL update endpoint handler
#[instrument(skip(state))]
pub async fn sparql_update(
    Form(params): Form<SparqlUpdateParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    user: Option<AuthUser>,
) -> impl IntoResponse {
    let start_time = Instant::now();

    // Check permissions — only when authentication is actually configured
    // (`security.auth_required`). With auth disabled (the default) the endpoint
    // serves anonymous callers and write-protection is the dataset `read_only`
    // flag enforced in `execute_sparql_update`. With auth enabled the hardened
    // enforcement stands: no credential → 401, missing permission → 403.
    if state.config.security.auth_required {
        if let Some(ref user) = user {
            if !user.0.permissions.contains(&Permission::SparqlUpdate) {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({
                        "error": "insufficient_permissions",
                        "message": "SPARQL update permission required"
                    })),
                )
                    .into_response();
            }
        } else {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "error": "authentication_required",
                    "message": "Authentication required for SPARQL updates"
                })),
            )
                .into_response();
        }
    }

    // Create update context
    let context = QueryContext {
        user,
        ..Default::default()
    };

    // Execute update
    match execute_sparql_update(&params.update, context, &state).await {
        Ok(result) => {
            let execution_time = start_time.elapsed();

            // Record metrics for successful update
            if let Some(metrics) = &state.metrics_service {
                let update_type = if params.update.to_uppercase().contains("INSERT") {
                    "INSERT"
                } else if params.update.to_uppercase().contains("DELETE") {
                    "DELETE"
                } else if params.update.to_uppercase().contains("LOAD") {
                    "LOAD"
                } else if params.update.to_uppercase().contains("CLEAR") {
                    "CLEAR"
                } else {
                    "UNKNOWN"
                };
                metrics
                    .record_sparql_update(execution_time, true, update_type)
                    .await;
            }

            Json(result).into_response()
        }
        Err(e) => {
            let execution_time = start_time.elapsed();

            // Record metrics for failed update
            if let Some(metrics) = &state.metrics_service {
                let update_type = if params.update.to_uppercase().contains("INSERT") {
                    "INSERT"
                } else if params.update.to_uppercase().contains("DELETE") {
                    "DELETE"
                } else if params.update.to_uppercase().contains("LOAD") {
                    "LOAD"
                } else if params.update.to_uppercase().contains("CLEAR") {
                    "CLEAR"
                } else {
                    "UNKNOWN"
                };
                metrics
                    .record_sparql_update(execution_time, false, update_type)
                    .await;
            }

            error!("SPARQL update execution failed: {}", e);
            (
                e.status_code(),
                Json(serde_json::json!({
                    "error": e.error_type(),
                    "message": e.to_string()
                })),
            )
                .into_response()
        }
    }
}

/// Execute a SPARQL query with a single, authoritative execution path.
///
/// The query is parsed ONCE by the real oxirs-arq parser; the parsed query form
/// (SELECT / ASK / CONSTRUCT / DESCRIBE) — not a substring scan of the text — is
/// the sole authority that decides how it executes. Every form then flows
/// through the oxirs-arq engine
/// ([`crate::handlers::sparql::arq_exec::dispatch`]) so `FILTER`, `LIMIT`/
/// `OFFSET`, `ORDER BY`, `DISTINCT`, joins, aggregation (`GROUP BY`/`HAVING`),
/// `GRAPH`/`FROM` scoping, `SERVICE` federation, `CONSTRUCT` templating and
/// `DESCRIBE` CBD are all actually evaluated.
///
/// There is no silent-empty fallback: a parse failure is an HTTP 400, an
/// execution failure an HTTP 500, and a genuinely unexecutable construct a typed
/// error. The endpoint never answers `200 OK` with a fabricated empty result.
pub async fn execute_sparql_query(
    query: &str,
    context: QueryContext,
    state: &Arc<AppState>,
) -> FusekiResult<QueryResult> {
    // Basic validation first.
    if query.trim().is_empty() {
        return Err(FusekiError::query_parsing("Empty query"));
    }

    // ── Result cache lookup ──────────────────────────────────────────────
    // When the result cache is wired (`performance.caching.query_cache_enabled`),
    // look the query up first. The cache key folds the dataset scope AND any
    // protocol graph URIs into the key so differently-scoped executions of the
    // same text never collide. A hit returns the stored structured result
    // (format-independent, so content negotiation still applies downstream).
    let cache_key = state.sparql_cache.as_ref().map(|_| {
        let mut scope = context.dataset.clone();
        for g in &context.default_graph_uris {
            scope.push_str("|d=");
            scope.push_str(g);
        }
        for g in &context.named_graph_uris {
            scope.push_str("|n=");
            scope.push_str(g);
        }
        crate::cache::QueryCacheKey::new(query, scope)
    });
    if let (Some(cache), Some(key)) = (state.sparql_cache.as_ref(), cache_key.as_ref()) {
        if let Some(hit) = cache.get(key) {
            if let Ok(cached) = serde_json::from_str::<QueryResult>(&hit.result_json) {
                return Ok(cached);
            }
        }
    }

    // ── Effective timeout ────────────────────────────────────────────────
    // The configured `max_query_time_secs` is the ceiling AND the default. A
    // client `?timeout=` can only LOWER it (never raise it above the server
    // cap): effective = min(requested, config_max). A `?timeout=0` (or absent)
    // falls back to the config cap rather than timing out instantly. `.max(1)`
    // guards a pathological config of 0. This is the wiring that finally makes
    // `performance.query_optimization.max_query_time_secs` a live setting.
    let config_max_secs = state
        .config
        .performance
        .query_optimization
        .max_query_time_secs
        .max(1);
    let requested_secs = context.timeout.map(|d| d.as_secs()).filter(|&s| s > 0);
    let effective_secs = requested_secs.map_or(config_max_secs, |s| s.min(config_max_secs));
    let effective = Duration::from_secs(effective_secs);
    let outer_wait = effective + Duration::from_secs(QUERY_TIMEOUT_GRACE_SECS);

    // ── Off-thread execution + hard response deadline ────────────────────
    // The oxirs-arq engine runs synchronously with no `.await`, so running it
    // directly on the async worker would (a) pin a tokio runtime thread for the
    // whole query and (b) make `tokio::time::timeout` and the outer TimeoutLayer
    // structurally unable to fire. `spawn_blocking` moves it to the blocking
    // pool (freeing the async worker) and lets the timeout race it.
    //
    // `spawn_blocking` is NOT cancellable, so the timeout alone cannot stop the
    // CPU work — that is the `ExecutionBudget`'s job (cooperative wall-time
    // checks inside the engine). BOTH are required: the budget halts the
    // computation, the timeout guarantees the client gets a response even if a
    // budget checkpoint is somehow missed. `Store` is a cheap `Arc` clone, which
    // satisfies the `'static` bound on the blocking closure.
    let budget = ExecutionBudget::new(ResourceBudget {
        max_wall_time: Some(effective),
        max_result_rows: None,
        max_triples_scanned: None,
    });
    let store = state.store.clone();
    let query_owned = query.to_string();
    // SPARQL 1.1 Protocol dataset scoping: `default-graph-uri` / `named-graph-uri`
    // supplied out-of-band override any in-query FROM/FROM NAMED (protocol §2.1.4).
    let protocol_default_graphs = context.default_graph_uris.clone();
    let protocol_named_graphs = context.named_graph_uris.clone();
    let join = tokio::task::spawn_blocking(move || {
        // Parse ONCE via the real arq parser. The parsed query form is the single
        // routing authority; a parse failure is a 400, never a silent 200 + empty.
        let mut parsed = match parse_query(&query_owned) {
            Ok(parsed) => parsed,
            Err(e) => {
                // A SPARQL UPDATE sent to the query endpoint will not parse as a
                // query; point the caller at the dedicated /update endpoint
                // instead of emitting a generic parse error.
                if looks_like_update(&query_owned) {
                    return Err(FusekiError::query_parsing(
                        "This is a SPARQL Query endpoint; send SPARQL UPDATE requests to /update",
                    ));
                }
                return Err(FusekiError::query_parsing(format!(
                    "SPARQL parse error: {e}"
                )));
            }
        };
        // Honour the protocol-supplied RDF dataset. If either list is present it
        // replaces the query's own dataset clause; an unparseable IRI is a 400
        // (fail-loud) rather than being silently ignored.
        if !protocol_default_graphs.is_empty() || !protocol_named_graphs.is_empty() {
            parsed.dataset =
                build_protocol_dataset(&protocol_default_graphs, &protocol_named_graphs)?;
        }
        // Dispatch on the parsed form with the wall-time budget attached; a
        // budget breach surfaces as a typed HTTP error (408/503), never a silent
        // empty body.
        crate::handlers::sparql::arq_exec::dispatch_with_budget(&parsed, &store, Some(budget))
    });

    let outcome = match tokio::time::timeout(outer_wait, join).await {
        // Task finished within the deadline: propagate its Ok/Err verbatim (a
        // BudgetExceeded that fired first is already a typed 408/503 here).
        Ok(Ok(result)) => result,
        // The blocking task panicked (or the runtime is shutting down).
        Ok(Err(join_err)) => Err(internal_execution_error(&join_err)),
        // Outer safety net fired: the cooperative budget did not stop the query
        // within `effective + grace`. The blocking thread is still running
        // (uncancellable) but must hit a budget checkpoint shortly and exit. We
        // return 408 for consistency with the budget-timeout and TimeoutLayer.
        Err(_elapsed) => Err(FusekiError::TimeoutWithMessage(format!(
            "query exceeded the {effective_secs}s execution-time limit (server aborted)"
        ))),
    };

    // ── Result cache store ───────────────────────────────────────────────
    // Only successful results are cached. Serialization failures are non-fatal
    // (the result is still returned to the client, just not cached).
    if let (Some(cache), Some(key), Ok(result)) = (state.sparql_cache.as_ref(), cache_key, &outcome)
    {
        if let Ok(serialized) = serde_json::to_string(result) {
            let graphs = result
                .bindings
                .as_ref()
                .map(|_| Vec::<String>::new())
                .unwrap_or_default();
            cache.put(key, serialized, "application/json", graphs);
        }
    }

    outcome
}

/// Build an oxirs-arq [`DatasetClause`](oxirs_arq::query::DatasetClause) from the
/// SPARQL 1.1 Protocol `default-graph-uri` / `named-graph-uri` values.
///
/// Each value must be an absolute IRI; an unparseable value is rejected with a
/// 400 (fail-loud) so a malformed `?default-graph-uri=` never silently degrades
/// to whole-dataset results.
fn build_protocol_dataset(
    default_graph_uris: &[String],
    named_graph_uris: &[String],
) -> FusekiResult<oxirs_arq::query::DatasetClause> {
    use oxirs_core::model::NamedNode;

    let parse_iri = |uri: &str| -> FusekiResult<NamedNode> {
        NamedNode::new(uri).map_err(|e| {
            FusekiError::query_parsing(format!(
                "invalid graph URI '{uri}' in protocol dataset: {e}"
            ))
        })
    };

    let default_graphs = default_graph_uris
        .iter()
        .map(|u| parse_iri(u))
        .collect::<FusekiResult<Vec<_>>>()?;
    let named_graphs = named_graph_uris
        .iter()
        .map(|u| parse_iri(u))
        .collect::<FusekiResult<Vec<_>>>()?;

    Ok(oxirs_arq::query::DatasetClause {
        default_graphs,
        named_graphs,
    })
}

/// Heuristic used only to improve the error message when a request that fails to
/// parse as a query looks like a SPARQL UPDATE (which belongs at `/update`).
///
/// Real routing is by parse form; this only chooses between two 400 messages.
fn looks_like_update(text: &str) -> bool {
    let upper = text.to_uppercase();
    [
        "INSERT ", "DELETE ", "LOAD ", "CLEAR ", "CREATE ", "DROP ", "COPY ", "MOVE ", "ADD ",
    ]
    .iter()
    .any(|kw| upper.contains(*kw))
}

/// Execute SPARQL update with validation
pub async fn execute_sparql_update(
    update: &str,
    context: QueryContext,
    state: &Arc<AppState>,
) -> FusekiResult<UpdateResult> {
    // Enforce read-only datasets before any parsing or mutation. A dataset
    // configured with `read_only = true` must reject every SPARQL UPDATE with
    // HTTP 403 and leave the data untouched — this is the write-protection that
    // a public, query-only endpoint (e.g. sparql.wik.jp) depends on.
    if state.is_dataset_read_only(&context.dataset) {
        return Err(FusekiError::forbidden(format!(
            "Dataset '{}' is read-only; SPARQL UPDATE is not permitted",
            context.dataset
        )));
    }

    // Validate update
    validate_sparql_update(update)?;

    // Execute through store
    let store_result = state.store.update(update)?;

    // Invalidate cached results for the mutated dataset (conservative: any
    // successful UPDATE drops every cached entry for the dataset so a subsequent
    // query never serves a stale answer).
    if let Some(cache) = state.sparql_cache.as_ref() {
        let invalidated = cache.invalidate_dataset(&context.dataset);
        if invalidated > 0 {
            debug!(
                "SPARQL UPDATE invalidated {invalidated} cached result(s) for dataset '{}'",
                context.dataset
            );
        }
    }

    // Convert store::UpdateResult to sparql::core::UpdateResult
    let operations_count = count_update_operations(update);

    let result = UpdateResult {
        success: store_result.stats.success,
        execution_time_ms: store_result.stats.execution_time.as_millis() as u64,
        operations_count,
        affected_triples: Some(
            store_result.stats.quads_inserted + store_result.stats.quads_deleted,
        ),
        error_message: store_result.stats.error_message,
    };

    Ok(result)
}

/// Count the number of update operations in a SPARQL UPDATE query
///
/// Counts distinct operations like INSERT DATA, DELETE DATA, DELETE/INSERT, CLEAR, LOAD, etc.
fn count_update_operations(update: &str) -> usize {
    let update_upper = update.to_uppercase();
    let mut count = 0;

    // Count INSERT DATA operations
    count += update_upper.matches("INSERT DATA").count();

    // Count DELETE DATA operations
    count += update_upper.matches("DELETE DATA").count();

    // Count DELETE/INSERT (or DELETE WHERE/INSERT) patterns
    // This is trickier as DELETE and INSERT might be separate or combined
    let delete_insert_pattern = regex::Regex::new(r"DELETE\s+(?:WHERE\s+)?\{[^}]*\}\s*INSERT\s+\{")
        .expect("regex pattern should be valid");
    count += delete_insert_pattern.find_iter(&update_upper).count();

    // Count standalone DELETE WHERE operations (not part of DELETE/INSERT)
    let standalone_delete = update_upper.matches("DELETE WHERE").count();
    let combined_delete_insert = delete_insert_pattern.find_iter(&update_upper).count();
    count += standalone_delete.saturating_sub(combined_delete_insert);

    // Count standalone INSERT operations (not INSERT DATA or part of DELETE/INSERT)
    let insert_count = update_upper.matches("INSERT").count();
    let insert_data_count = update_upper.matches("INSERT DATA").count();
    let standalone_insert = insert_count.saturating_sub(insert_data_count + combined_delete_insert);
    count += standalone_insert;

    // Count CLEAR operations
    count += update_upper.matches("CLEAR").count();

    // Count LOAD operations
    count += update_upper.matches("LOAD").count();

    // Count DROP operations
    count += update_upper.matches("DROP").count();

    // Count CREATE operations
    count += update_upper.matches("CREATE").count();

    // Count COPY operations
    count += update_upper.matches("COPY").count();

    // Count MOVE operations
    count += update_upper.matches("MOVE").count();

    // Count ADD operations
    count += update_upper.matches("ADD").count();

    // If no operations found, return 1 as a fallback (likely a simple operation we didn't recognize)
    if count == 0 {
        1
    } else {
        count
    }
}

/// Negotiate the preferred SPARQL response format from an Accept header.
///
/// Each comma-separated entry can carry a `q=` quality value (default
/// 1.0). The highest q-value entry whose media type is in our known
/// SPARQL Query Results / RDF Graph format set wins. When no known
/// entry matches, the SPARQL Query Results JSON media type is returned.
fn negotiate_sparql_format(accept: &str) -> String {
    const KNOWN: &[&str] = &[
        "application/sparql-results+json",
        "application/sparql-results+xml",
        "application/json",
        "application/xml",
        "text/csv",
        "text/tab-separated-values",
        "text/turtle",
        "application/x-turtle",
        "application/n-triples",
        "application/rdf+xml",
        "application/ld+json",
    ];
    if accept.trim().is_empty() {
        return "application/sparql-results+json".to_string();
    }
    let mut entries: Vec<(String, f32)> = Vec::new();
    for raw in accept.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let mut parts = raw.split(';');
        let media = parts.next().unwrap_or("").trim().to_lowercase();
        let mut q: f32 = 1.0;
        for param in parts {
            let param = param.trim();
            if let Some(rest) = param.strip_prefix("q=") {
                if let Ok(v) = rest.parse::<f32>() {
                    q = v;
                }
            }
        }
        entries.push((media, q));
    }
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (media, _) in &entries {
        for known in KNOWN {
            if media == known {
                return (*known).to_string();
            }
        }
        if media == "*/*" {
            return "application/sparql-results+json".to_string();
        }
        if media == "application/*" {
            return "application/sparql-results+json".to_string();
        }
        if media == "text/*" {
            return "text/csv".to_string();
        }
    }
    "application/sparql-results+json".to_string()
}

/// Format query response based on content type.
///
/// Performs HTTP Accept-header content negotiation across the W3C
/// SPARQL Query Results JSON / XML / CSV / TSV formats and the common
/// RDF graph media types (Turtle / N-Triples / RDF-XML / JSON-LD).
/// For each media-range entry in `content_type` the q-value is
/// honoured; the highest-scoring supported format wins. If no entry
/// matches, JSON Results is the default.
fn format_query_response(result: QueryResult, content_type: &str) -> Response {
    let primary = negotiate_sparql_format(content_type);

    // A CONSTRUCT/DESCRIBE produces an RDF graph, not a bindings table. If the
    // client negotiated a bindings format (SPARQL Results JSON/XML/CSV/TSV, or an
    // unrecognised type falling through to the JSON default), serving it would
    // emit a silently-empty `bindings` array. Fall back to Turtle instead so a
    // graph result is never a silent empty answer.
    let is_graph = matches!(result.query_type.as_str(), "CONSTRUCT" | "DESCRIBE");
    let is_graph_format = matches!(
        primary.as_str(),
        "text/turtle"
            | "application/x-turtle"
            | "application/n-triples"
            | "application/rdf+xml"
            | "application/ld+json"
    );
    // Symmetric fallback: a SELECT/ASK result (bindings/boolean) negotiated to
    // a graph media type (e.g. Accept: text/turtle, application/rdf+xml,
    // application/ld+json) has no graph to serialize. Falling into the graph
    // arms below would silently emit an empty document instead of the
    // client's actual bindings/boolean. Fall back to SPARQL Results JSON
    // instead, mirroring the CONSTRUCT/DESCRIBE -> Turtle fallback above.
    let primary = if is_graph && !is_graph_format {
        "text/turtle".to_string()
    } else if !is_graph && is_graph_format {
        "application/sparql-results+json".to_string()
    } else {
        primary
    };

    match primary.as_str() {
        "application/sparql-results+json" | "application/json" => {
            let body = build_sparql_results_json(&result);
            let body_text = serde_json::to_string(&body)
                .unwrap_or_else(|_| "{\"head\":{},\"results\":{\"bindings\":[]}}".to_string());
            response_with_content_type(body_text, "application/sparql-results+json")
        }
        "application/sparql-results+xml" | "application/xml" => {
            let body = crate::handlers::sparql::content_types::sparql_results_xml(&result);
            response_with_content_type(body, "application/sparql-results+xml")
        }
        "text/csv" => {
            let body = crate::handlers::sparql::content_types::sparql_results_csv(&result);
            response_with_content_type(body, "text/csv")
        }
        "text/tab-separated-values" => {
            let body = crate::handlers::sparql::content_types::sparql_results_tsv(&result);
            response_with_content_type(body, "text/tab-separated-values")
        }
        "text/turtle" | "application/x-turtle" => {
            let body = result
                .construct_graph
                .clone()
                .or(result.describe_graph.clone())
                .unwrap_or_default();
            response_with_content_type(body, "text/turtle")
        }
        "application/n-triples" => {
            let body = result
                .construct_graph
                .clone()
                .or(result.describe_graph.clone())
                .unwrap_or_default();
            response_with_content_type(body, "application/n-triples")
        }
        "application/rdf+xml" => {
            let graph = result
                .construct_graph
                .as_deref()
                .or(result.describe_graph.as_deref())
                .unwrap_or("");
            match crate::handlers::sparql::content_types::rdf_graph_to_rdfxml(graph) {
                Ok(body) => response_with_content_type(body, "application/rdf+xml"),
                Err(e) => e.into_response(),
            }
        }
        "application/ld+json" => {
            let graph = result
                .construct_graph
                .as_deref()
                .or(result.describe_graph.as_deref())
                .unwrap_or("");
            match crate::handlers::sparql::content_types::rdf_graph_to_jsonld(graph) {
                Ok(body) => response_with_content_type(body, "application/ld+json"),
                Err(e) => e.into_response(),
            }
        }
        _ => {
            // Default: SPARQL Results JSON.
            let body = build_sparql_results_json(&result);
            let body_text = serde_json::to_string(&body)
                .unwrap_or_else(|_| "{\"head\":{},\"results\":{\"bindings\":[]}}".to_string());
            response_with_content_type(body_text, "application/sparql-results+json")
        }
    }
}

/// Build the W3C SPARQL Query Results JSON document for a `QueryResult`.
///
/// See <https://www.w3.org/TR/sparql11-results-json/>.
fn build_sparql_results_json(result: &QueryResult) -> serde_json::Value {
    match result.query_type.as_str() {
        "ASK" => serde_json::json!({
            "head": {},
            "boolean": result.boolean.unwrap_or(false),
        }),
        "SELECT" => {
            let bindings = result.bindings.clone().unwrap_or_default();
            let variables: Vec<String> = bindings
                .first()
                .map(|b| b.keys().cloned().collect())
                .unwrap_or_default();
            serde_json::json!({
                "head": { "vars": variables },
                "results": { "bindings": bindings },
            })
        }
        _ => serde_json::json!({
            "head": {},
            "results": { "bindings": result.bindings.clone().unwrap_or_default() },
        }),
    }
}

/// Build a Response with a fixed `Content-Type` header.
fn response_with_content_type(body: String, content_type: &'static str) -> Response {
    use axum::body::Body;
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .unwrap_or_else(|_| {
            (StatusCode::INTERNAL_SERVER_ERROR, "response build failure").into_response()
        })
}

/// Basic SPARQL query validation
pub fn validate_sparql_query(query: &str) -> FusekiResult<()> {
    if query.trim().is_empty() {
        return Err(FusekiError::query_parsing("Empty query"));
    }

    // Basic syntax validation
    if !query.to_uppercase().contains("SELECT")
        && !query.to_uppercase().contains("CONSTRUCT")
        && !query.to_uppercase().contains("ASK")
        && !query.to_uppercase().contains("DESCRIBE")
    {
        return Err(FusekiError::query_parsing(
            "Query must contain SELECT, CONSTRUCT, ASK, or DESCRIBE",
        ));
    }

    Ok(())
}

/// Basic SPARQL update validation
fn validate_sparql_update(update: &str) -> FusekiResult<()> {
    if update.trim().is_empty() {
        return Err(FusekiError::query_parsing("Empty update"));
    }

    // Basic sanity check: the request must contain at least one recognized
    // SPARQL 1.1 Update keyword. This includes the graph-management operations
    // (CREATE/DROP/COPY/MOVE/ADD), which are valid updates the store implements
    // but which contain none of INSERT/DELETE/LOAD/CLEAR — rejecting them here
    // would make those operations unreachable via HTTP with a spurious 400. The
    // authoritative validation is the store's AST-based dispatch downstream,
    // which rejects genuinely unrecognized operations.
    let upper_update = update.to_uppercase();
    const UPDATE_KEYWORDS: [&str; 9] = [
        "INSERT", "DELETE", "LOAD", "CLEAR", "CREATE", "DROP", "COPY", "MOVE", "ADD",
    ];
    if !UPDATE_KEYWORDS.iter().any(|kw| upper_update.contains(kw)) {
        return Err(FusekiError::query_parsing(
            "Update must contain a SPARQL 1.1 Update operation \
             (INSERT, DELETE, LOAD, CLEAR, CREATE, DROP, COPY, MOVE, or ADD)",
        ));
    }

    Ok(())
}

/// Serialize graph quads to Turtle format
fn serialize_graph_to_turtle(quads: &[oxirs_core::model::Quad]) -> String {
    let mut turtle = String::new();

    // Add common prefixes
    turtle.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
    turtle.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
    turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

    for quad in quads {
        let subject_str =
            format_term_for_turtle(&oxirs_core::model::Term::from_subject(quad.subject()));
        let predicate_str =
            format_term_for_turtle(&oxirs_core::model::Term::from_predicate(quad.predicate()));
        let object_str =
            format_term_for_turtle(&oxirs_core::model::Term::from_object(quad.object()));

        turtle.push_str(&format!(
            "{} {} {} .\n",
            subject_str, predicate_str, object_str
        ));
    }

    turtle
}

/// Serialize triples to Turtle format.
///
/// Shared with [`crate::handlers::sparql::arq_exec`], which serializes CONSTRUCT
/// and DESCRIBE graphs through this same helper after converting arq algebra
/// triples to oxirs-core triples, so the wire format is identical across paths.
pub(crate) fn serialize_triples_to_turtle(triples: &[oxirs_core::model::Triple]) -> String {
    let mut turtle = String::new();

    // Add common prefixes
    turtle.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
    turtle.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
    turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

    for triple in triples {
        let subject_str =
            format_term_for_turtle(&oxirs_core::model::Term::from_subject(triple.subject()));
        let predicate_str =
            format_term_for_turtle(&oxirs_core::model::Term::from_predicate(triple.predicate()));
        let object_str =
            format_term_for_turtle(&oxirs_core::model::Term::from_object(triple.object()));

        turtle.push_str(&format!(
            "{} {} {} .\n",
            subject_str, predicate_str, object_str
        ));
    }

    turtle
}

/// Format RDF term for Turtle serialization
fn format_term_for_turtle(term: &oxirs_core::model::Term) -> String {
    match term {
        oxirs_core::model::Term::NamedNode(iri) => {
            format!("<{}>", iri.as_str())
        }
        oxirs_core::model::Term::BlankNode(bnode) => {
            format!("_:{}", bnode)
        }
        oxirs_core::model::Term::Literal(literal) => {
            let mut formatted = format!("\"{}\"", escape_turtle_string(literal.value()));

            if let Some(language) = literal.language() {
                formatted.push_str(&format!("@{}", language));
            } else {
                let datatype_str = literal.datatype().as_str();
                // Only add datatype if it's not the default string type
                if datatype_str != "http://www.w3.org/2001/XMLSchema#string" {
                    formatted.push_str(&format!("^^<{}>", datatype_str));
                }
            }

            formatted
        }
        oxirs_core::model::Term::Variable(var) => {
            format!("?{}", var.name())
        }
        oxirs_core::model::Term::QuotedTriple(triple) => {
            format!("<< {} >>", triple)
        }
    }
}

/// Escape special characters in Turtle string literals
fn escape_turtle_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod update_validation_tests {
    use super::validate_sparql_update;

    #[test]
    fn regression_graph_management_updates_accepted() {
        // SPARQL 1.1 graph-management operations contain none of
        // INSERT/DELETE/LOAD/CLEAR but are valid updates the store implements.
        for stmt in [
            "CREATE GRAPH <http://example.org/g>",
            "DROP GRAPH <http://example.org/g>",
            "COPY <http://example.org/a> TO <http://example.org/b>",
            "MOVE <http://example.org/a> TO <http://example.org/b>",
            "ADD <http://example.org/a> TO <http://example.org/b>",
            "INSERT DATA { <s> <p> <o> }",
            "DELETE WHERE { ?s ?p ?o }",
        ] {
            assert!(
                validate_sparql_update(stmt).is_ok(),
                "should accept valid update: {stmt}"
            );
        }
    }

    #[test]
    fn regression_non_update_rejected() {
        assert!(validate_sparql_update("").is_err());
        assert!(validate_sparql_update("SELECT ?s WHERE { ?s ?p ?o }").is_err());
    }
}

#[cfg(test)]
mod protocol_dataset_tests {
    use super::build_protocol_dataset;

    #[test]
    fn regression_protocol_dataset_threaded_and_validated() {
        // Valid protocol graph URIs build a dataset clause carrying them.
        let clause = build_protocol_dataset(
            &["http://example.org/g1".to_string()],
            &["http://example.org/n1".to_string()],
        )
        .expect("valid IRIs build a dataset clause");
        assert_eq!(clause.default_graphs.len(), 1);
        assert_eq!(clause.named_graphs.len(), 1);
        assert_eq!(clause.default_graphs[0].as_str(), "http://example.org/g1");

        // A malformed IRI is rejected (fail-loud), not silently dropped.
        assert!(build_protocol_dataset(&["not a valid iri".to_string()], &[]).is_err());
    }
}

#[cfg(test)]
mod internal_error_tests {
    use super::{internal_execution_error, INTERNAL_EXECUTION_ERROR_MSG};

    /// A panicked query task must surface to the client as a fixed, generic 500
    /// message — never the panic payload (which can embed engine internals,
    /// assertion text, or query state). We build a real [`tokio::task::JoinError`]
    /// carrying a distinctive secret and assert it does not reach the
    /// client-facing error body (`FusekiError::Display`, which
    /// `to_error_response` embeds verbatim).
    #[test]
    fn internal_execution_error_hides_panic_payload_from_client_body() {
        let secret = "SENSITIVE_PANIC_9f3c_do_not_leak";
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build current-thread runtime for the test");
        let join_err = rt.block_on(async move {
            tokio::task::spawn_blocking(move || {
                panic!("{secret}");
            })
            .await
            .expect_err("the blocking task panicked, so join must return Err")
        });

        // Sanity: this really is a panic join error (version-robust — some tokio
        // releases omit the message from `Display`, so we check the kind).
        assert!(join_err.is_panic(), "expected a panic JoinError");

        let client_body = internal_execution_error(&join_err).to_string();

        assert!(
            !client_body.contains("SENSITIVE_PANIC"),
            "client body must not leak the panic payload, got: {client_body}"
        );
        assert!(
            client_body.contains(INTERNAL_EXECUTION_ERROR_MSG),
            "client body should carry the fixed sentinel, got: {client_body}"
        );
    }
}
