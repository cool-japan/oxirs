//! Core SPARQL Protocol implementation
//!
//! This module implements the SPARQL 1.1 Protocol for RDF as defined by W3C:
//! https://www.w3.org/TR/sparql11-protocol/
//! With SPARQL 1.2 enhancements and advanced optimizations

use crate::{
    aggregation::{AggregationFactory, EnhancedAggregationProcessor},
    auth::{AuthUser, Permission},
    bind_values_enhanced::{EnhancedBindProcessor, EnhancedValuesProcessor},
    config::ServerConfig,
    error::{FusekiError, FusekiResult},
    federated_query_optimizer::FederatedQueryOptimizer,
    federation::{planner::FederatedQueryPlan, FederationConfig},
    metrics::MetricsService,
    server::AppState,
    store::Store,
    subquery_optimizer::AdvancedSubqueryOptimizer,
};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{
        header::{ACCEPT, CONTENT_TYPE},
        HeaderMap, StatusCode,
    },
    response::{Html, IntoResponse, Json, Response},
    Form,
};
use chrono;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, instrument, warn};

// SPARQL query parsing
use oxirs_arq::query::parse_query;

/// SPARQL query parameters for GET requests
#[derive(Debug, Deserialize)]
pub struct SparqlQueryParams {
    pub query: Option<String>,
    #[serde(rename = "default-graph-uri")]
    pub default_graph_uri: Option<Vec<String>>,
    #[serde(rename = "named-graph-uri")]
    pub named_graph_uri: Option<Vec<String>>,
    pub timeout: Option<u32>,
    pub format: Option<String>,
}

/// SPARQL update parameters for POST requests
#[derive(Debug, Deserialize)]
pub struct SparqlUpdateParams {
    pub update: String,
    #[serde(rename = "using-graph-uri")]
    pub using_graph_uri: Option<Vec<String>>,
    #[serde(rename = "using-named-graph-uri")]
    pub using_named_graph_uri: Option<Vec<String>>,
}

/// SPARQL query request body for direct POST
#[derive(Debug, Deserialize)]
pub struct SparqlQueryRequest {
    pub query: String,
    #[serde(rename = "default-graph-uri")]
    pub default_graph_uri: Option<Vec<String>>,
    #[serde(rename = "named-graph-uri")]
    pub named_graph_uri: Option<Vec<String>>,
    pub timeout: Option<u32>,
}

/// SPARQL query execution result
#[derive(Debug, Clone, Serialize)]
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

    // Extract query from parameters
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

    // Create query context
    let mut context = QueryContext::default();
    context.user = user;
    if let Some(timeout) = params.timeout {
        context.timeout = Some(Duration::from_secs(timeout as u64));
    }

    // Execute query
    match execute_sparql_query(&query_string, context, &state).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;

            // Update metrics
            state.metrics.record_query_execution(execution_time, true);

            // Determine response format based on Accept header
            let accept_header = headers
                .get(ACCEPT)
                .and_then(|h| h.to_str().ok())
                .unwrap_or("application/sparql-results+json");

            format_query_response(result, accept_header).into_response()
        }
        Err(e) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            state.metrics.record_query_execution(execution_time, false);

            error!("SPARQL query execution failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "query_execution_failed",
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

    // Check permissions
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

    // Create update context
    let mut context = QueryContext::default();
    context.user = user;

    // Execute update
    match execute_sparql_update(&params.update, context, &state).await {
        Ok(result) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            state.metrics.record_update_execution(execution_time, true);

            Json(result).into_response()
        }
        Err(e) => {
            let execution_time = start_time.elapsed().as_millis() as u64;
            state.metrics.record_update_execution(execution_time, false);

            error!("SPARQL update execution failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "update_execution_failed",
                    "message": e.to_string()
                })),
            )
                .into_response()
        }
    }
}

/// Execute SPARQL query with enhanced features
async fn execute_sparql_query(
    query: &str,
    context: QueryContext,
    state: &Arc<AppState>,
) -> FusekiResult<QueryResult> {
    // Validate query
    validate_sparql_query(query)?;

    // Parse query for optimization
    let parsed_query = parse_query(query)
        .map_err(|e| FusekiError::query_parsing(format!("Query parsing failed: {}", e)))?;

    // Apply optimizations if enabled
    let optimized_query = if context.enable_optimizations {
        apply_query_optimizations(query, &context, state).await?
    } else {
        query.to_string()
    };

    // Execute through store
    let result = state
        .store
        .execute_query(&optimized_query, &context)
        .await?;

    Ok(result)
}

/// Execute SPARQL update with validation
async fn execute_sparql_update(
    update: &str,
    context: QueryContext,
    state: &Arc<AppState>,
) -> FusekiResult<UpdateResult> {
    // Validate update
    validate_sparql_update(update)?;

    // Execute through store
    let result = state.store.execute_update(update, &context).await?;

    Ok(result)
}

/// Apply various query optimizations
async fn apply_query_optimizations(
    query: &str,
    context: &QueryContext,
    state: &Arc<AppState>,
) -> FusekiResult<String> {
    let mut optimized_query = query.to_string();

    // Apply federation optimization if enabled
    if context.enable_federation {
        let federation_optimizer = FederatedQueryOptimizer::new();
        if let Some(federated_plan) = federation_optimizer.create_execution_plan(query).await? {
            optimized_query = federated_plan.optimized_query;
        }
    }

    // Apply other optimizations...

    Ok(optimized_query)
}

/// Format query response based on content type
fn format_query_response(result: QueryResult, content_type: &str) -> Response {
    match content_type {
        "application/sparql-results+json" | "application/json" => Json(result).into_response(),
        "application/sparql-results+xml" => {
            // TODO: Implement XML formatting
            Html(format!("<result>{:?}</result>", result)).into_response()
        }
        "text/csv" => {
            // TODO: Implement CSV formatting
            format!("CSV format not yet implemented").into_response()
        }
        _ => Json(result).into_response(),
    }
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

    // Basic syntax validation
    let upper_update = update.to_uppercase();
    if !upper_update.contains("INSERT")
        && !upper_update.contains("DELETE")
        && !upper_update.contains("LOAD")
        && !upper_update.contains("CLEAR")
    {
        return Err(FusekiError::query_parsing(
            "Update must contain INSERT, DELETE, LOAD, or CLEAR",
        ));
    }

    Ok(())
}
