//! REST API v2
//!
//! Modern RESTful API with OpenAPI 3.0 specification.
//! Provides comprehensive CRUD operations for datasets, queries, and administration.

use crate::config::DatasetConfig;
use crate::store_ext::StoreExt;
use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query as AxumQuery, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};
use utoipa::{OpenApi, ToSchema};

/// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    paths(
        get_api_info,
        list_datasets,
        get_dataset,
        create_dataset,
        delete_dataset,
        execute_query,
        get_triples,
        insert_triple,
        delete_triple,
        get_statistics,
        get_health
    ),
    components(schemas(
        ApiInfo,
        DatasetList,
        Dataset,
        CreateDatasetRequest,
        QueryRequest,
        QueryResponse,
        TripleList,
        Triple,
        InsertTripleRequest,
        DeleteTripleRequest,
        Statistics,
        HealthStatus,
        ErrorResponse
    )),
    tags(
        (name = "info", description = "API information endpoints"),
        (name = "datasets", description = "Dataset management endpoints"),
        (name = "queries", description = "Query execution endpoints"),
        (name = "triples", description = "Triple manipulation endpoints"),
        (name = "statistics", description = "Statistics and monitoring endpoints"),
        (name = "health", description = "Health check endpoints")
    ),
    info(
        title = "OxiRS Fuseki REST API v2",
        version = "2.0.0",
        description = "Modern RESTful API for SPARQL and RDF data management",
        contact(
            name = "OxiRS Team",
            url = "https://github.com/cool-japan/oxirs",
            email = "team@oxirs.dev"
        ),
        license(
            name = "Apache 2.0",
            url = "https://www.apache.org/licenses/LICENSE-2.0"
        )
    ),
    servers(
        (url = "http://localhost:3030/api/v2", description = "Local development"),
        (url = "https://api.oxirs.dev/v2", description = "Production")
    )
)]
pub struct ApiDoc;

/// API information
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ApiInfo {
    /// API version
    pub version: String,
    /// API name
    pub name: String,
    /// API description
    pub description: String,
    /// Available endpoints
    pub endpoints: Vec<String>,
    /// Supported features
    pub features: Vec<String>,
}

/// Dataset list response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DatasetList {
    /// List of datasets
    pub datasets: Vec<Dataset>,
    /// Total count
    pub total: usize,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Dataset {
    /// Dataset name
    pub name: String,
    /// Number of triples
    pub triple_count: usize,
    /// Dataset description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Creation timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Last modified timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Create dataset request
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateDatasetRequest {
    /// Dataset name
    pub name: String,
    /// Dataset description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Storage type (tdb, memory)
    #[serde(default = "default_storage_type")]
    pub storage_type: String,
}

fn default_storage_type() -> String {
    "tdb".to_string()
}

/// Query request
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QueryRequest {
    /// SPARQL query string
    pub query: String,
    /// Default graph URIs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_graph_uri: Option<Vec<String>>,
    /// Named graph URIs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub named_graph_uri: Option<Vec<String>>,
    /// Query timeout in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
}

/// Query response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QueryResponse {
    /// Variable names (for SELECT queries)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<QueryHead>,
    /// Query results
    pub results: QueryResults,
    /// Query execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Query head with variable names
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QueryHead {
    /// Variable names
    pub vars: Vec<String>,
}

/// Query results
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct QueryResults {
    /// Bindings (for SELECT queries)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bindings: Option<Vec<std::collections::HashMap<String, BindingValue>>>,
    /// Boolean result (for ASK queries)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boolean: Option<bool>,
}

/// Binding value
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BindingValue {
    /// Value type (uri, literal, bnode)
    #[serde(rename = "type")]
    pub value_type: String,
    /// Value
    pub value: String,
    /// Datatype (for literals)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datatype: Option<String>,
    /// Language tag (for literals)
    #[serde(skip_serializing_if = "Option::is_none", rename = "xml:lang")]
    pub lang: Option<String>,
}

/// Triple list response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TripleList {
    /// List of triples
    pub triples: Vec<Triple>,
    /// Total count
    pub total: usize,
    /// Limit
    pub limit: usize,
    /// Offset
    pub offset: usize,
}

/// RDF Triple
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Triple {
    /// Subject URI
    pub subject: String,
    /// Predicate URI
    pub predicate: String,
    /// Object (URI or literal)
    pub object: String,
    /// Object type (uri, literal)
    pub object_type: String,
}

/// Insert triple request
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InsertTripleRequest {
    /// Triples to insert
    pub triples: Vec<Triple>,
    /// Graph URI (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<String>,
}

/// Delete triple request
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeleteTripleRequest {
    /// Subject URI (optional, ? for wildcard)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// Predicate URI (optional, ? for wildcard)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    /// Object (optional, ? for wildcard)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// Graph URI (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<String>,
}

/// System statistics
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Statistics {
    /// Number of datasets
    pub dataset_count: usize,
    /// Total number of triples
    pub total_triples: usize,
    /// Total queries executed
    pub total_queries: u64,
    /// Average query time in milliseconds
    pub avg_query_time_ms: f64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HealthStatus {
    /// Overall status
    pub status: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Component statuses
    pub components: std::collections::HashMap<String, ComponentHealth>,
}

/// Component health
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ComponentHealth {
    /// Status (healthy, degraded, unhealthy)
    pub status: String,
    /// Details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Pagination query parameters
#[derive(Debug, Clone, Deserialize, ToSchema, utoipa::IntoParams)]
pub struct PaginationParams {
    /// Limit (default: 100, max: 10000)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Offset (default: 0)
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    100
}

/// API error type
#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    InternalError(anyhow::Error),
    Unauthorized,
    Forbidden,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg),
            ApiError::InternalError(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                err.to_string(),
            ),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "UNAUTHORIZED",
                "Authentication required".to_string(),
            ),
            ApiError::Forbidden => (
                StatusCode::FORBIDDEN,
                "FORBIDDEN",
                "Access denied".to_string(),
            ),
        };

        let error = ErrorResponse {
            code: code.to_string(),
            message,
            details: None,
            timestamp: chrono::Utc::now(),
        };

        (status, Json(error)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError::InternalError(err)
    }
}

/// Get API information
///
/// Returns information about the API version, available endpoints, and features.
#[utoipa::path(
    get,
    path = "/api/v2",
    responses(
        (status = 200, description = "API information", body = ApiInfo)
    ),
    tag = "info"
)]
pub async fn get_api_info() -> Result<Json<ApiInfo>, ApiError> {
    Ok(Json(ApiInfo {
        version: "2.0.0".to_string(),
        name: "OxiRS Fuseki REST API v2".to_string(),
        description: "Modern RESTful API for SPARQL and RDF data management".to_string(),
        endpoints: vec![
            "/api/v2".to_string(),
            "/api/v2/datasets".to_string(),
            "/api/v2/datasets/{name}".to_string(),
            "/api/v2/datasets/{name}/query".to_string(),
            "/api/v2/datasets/{name}/triples".to_string(),
            "/api/v2/statistics".to_string(),
            "/api/v2/health".to_string(),
        ],
        features: vec![
            "SPARQL 1.1".to_string(),
            "SPARQL 1.2".to_string(),
            "RDF-star".to_string(),
            "Federation".to_string(),
            "GraphQL".to_string(),
            "WebSocket".to_string(),
        ],
    }))
}

/// List all datasets
///
/// Returns a list of all available datasets with their metadata.
#[utoipa::path(
    get,
    path = "/api/v2/datasets",
    responses(
        (status = 200, description = "List of datasets", body = DatasetList)
    ),
    tag = "datasets"
)]
pub async fn list_datasets(
    State(store): State<Arc<crate::store::Store>>,
) -> Result<Json<DatasetList>, ApiError> {
    let datasets = store
        .list_datasets()
        .map_err(|e| ApiError::InternalError(e.into()))?;

    let dataset_infos: Vec<Dataset> = datasets
        .into_iter()
        .map(|name| {
            let triple_count = store.count_triples(&name);
            Dataset {
                name,
                triple_count,
                description: None,
                created_at: None,
                modified_at: None,
            }
        })
        .collect();

    let total = dataset_infos.len();

    Ok(Json(DatasetList {
        datasets: dataset_infos,
        total,
    }))
}

/// Get dataset information
///
/// Returns detailed information about a specific dataset.
#[utoipa::path(
    get,
    path = "/api/v2/datasets/{name}",
    responses(
        (status = 200, description = "Dataset information", body = Dataset),
        (status = 404, description = "Dataset not found", body = ErrorResponse)
    ),
    params(
        ("name" = String, Path, description = "Dataset name")
    ),
    tag = "datasets"
)]
pub async fn get_dataset(
    State(store): State<Arc<crate::store::Store>>,
    Path(name): Path<String>,
) -> Result<Json<Dataset>, ApiError> {
    if !store.dataset_exists(&name) {
        return Err(ApiError::NotFound(format!("Dataset '{}' not found", name)));
    }

    let triple_count = store.count_triples(&name);

    Ok(Json(Dataset {
        name,
        triple_count,
        description: None,
        created_at: None,
        modified_at: None,
    }))
}

/// Create a new dataset
///
/// Creates a new dataset with the specified configuration.
#[utoipa::path(
    post,
    path = "/api/v2/datasets",
    request_body = CreateDatasetRequest,
    responses(
        (status = 201, description = "Dataset created", body = Dataset),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 409, description = "Dataset already exists", body = ErrorResponse)
    ),
    tag = "datasets"
)]
pub async fn create_dataset(
    State(store): State<Arc<crate::store::Store>>,
    Json(request): Json<CreateDatasetRequest>,
) -> Result<(StatusCode, Json<Dataset>), ApiError> {
    if store.dataset_exists(&request.name) {
        return Err(ApiError::BadRequest(format!(
            "Dataset '{}' already exists",
            request.name
        )));
    }

    let config = crate::config::DatasetConfig {
        name: request.name.clone(),
        location: format!("./data/{}", request.name),
        read_only: false,
        text_index: None,
        shacl_shapes: Vec::new(),
        services: Vec::new(),
        access_control: None,
        backup: None,
    };

    store
        .create_dataset(&request.name, config)
        .map_err(|e| ApiError::InternalError(e.into()))?;

    info!("Created dataset: {}", request.name);

    Ok((
        StatusCode::CREATED,
        Json(Dataset {
            name: request.name,
            triple_count: 0,
            description: request.description,
            created_at: Some(chrono::Utc::now()),
            modified_at: Some(chrono::Utc::now()),
        }),
    ))
}

/// Delete a dataset
///
/// Deletes the specified dataset and all its triples.
#[utoipa::path(
    delete,
    path = "/api/v2/datasets/{name}",
    responses(
        (status = 204, description = "Dataset deleted"),
        (status = 404, description = "Dataset not found", body = ErrorResponse)
    ),
    params(
        ("name" = String, Path, description = "Dataset name")
    ),
    tag = "datasets"
)]
pub async fn delete_dataset(
    State(store): State<Arc<crate::store::Store>>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiError> {
    if !store.dataset_exists(&name) {
        return Err(ApiError::NotFound(format!("Dataset '{}' not found", name)));
    }

    store
        .remove_dataset(&name)
        .map_err(|e| ApiError::InternalError(anyhow::anyhow!("{}", e)))?;

    info!("Deleted dataset: {}", name);

    Ok(StatusCode::NO_CONTENT)
}

/// Execute a SPARQL query
///
/// Executes a SPARQL query against the specified dataset.
#[utoipa::path(
    post,
    path = "/api/v2/datasets/{name}/query",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Query results", body = QueryResponse),
        (status = 400, description = "Invalid query", body = ErrorResponse),
        (status = 404, description = "Dataset not found", body = ErrorResponse)
    ),
    params(
        ("name" = String, Path, description = "Dataset name")
    ),
    tag = "queries"
)]
pub async fn execute_query(
    State(store): State<Arc<crate::store::Store>>,
    Path(name): Path<String>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    if !store.dataset_exists(&name) {
        return Err(ApiError::NotFound(format!("Dataset '{}' not found", name)));
    }

    let start = std::time::Instant::now();

    let results = store
        .query_dataset(&request.query, Some(&name))
        .map_err(|e| ApiError::BadRequest(format!("Query execution failed: {}", e)))?;

    let execution_time_ms = start.elapsed().as_millis() as u64;

    // Convert results to API format
    let bindings = match results.inner {
        oxirs_core::query::QueryResult::Select { bindings, .. } => bindings
            .into_iter()
            .map(|binding| {
                binding
                    .into_iter()
                    .map(|(var, value)| {
                        (
                            var,
                            BindingValue {
                                value_type: "uri".to_string(), // Simplified
                                value: value.to_string(),
                                datatype: None,
                                lang: None,
                            },
                        )
                    })
                    .collect()
            })
            .collect(),
        _ => Vec::new(), // Handle other query types
    };

    Ok(Json(QueryResponse {
        head: Some(QueryHead {
            vars: vec![], // Would extract from query
        }),
        results: QueryResults {
            bindings: Some(bindings),
            boolean: None,
        },
        execution_time_ms,
    }))
}

/// Get triples from a dataset
///
/// Retrieves triples matching the specified pattern.
#[utoipa::path(
    get,
    path = "/api/v2/datasets/{name}/triples",
    responses(
        (status = 200, description = "List of triples", body = TripleList),
        (status = 404, description = "Dataset not found", body = ErrorResponse)
    ),
    params(
        ("name" = String, Path, description = "Dataset name"),
        PaginationParams
    ),
    tag = "triples"
)]
pub async fn get_triples(
    State(store): State<Arc<crate::store::Store>>,
    Path(name): Path<String>,
    AxumQuery(pagination): AxumQuery<PaginationParams>,
) -> Result<Json<TripleList>, ApiError> {
    if !store.dataset_exists(&name) {
        return Err(ApiError::NotFound(format!("Dataset '{}' not found", name)));
    }

    // Build SPARQL query to get triples
    let query = format!(
        "SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {} OFFSET {}",
        pagination.limit, pagination.offset
    );

    let results = store
        .query_dataset(&query, Some(&name))
        .map_err(|e| ApiError::InternalError(anyhow::anyhow!("{}", e)))?;

    let triples: Vec<Triple> = match results.inner {
        oxirs_core::query::QueryResult::Select { bindings, .. } => bindings
            .into_iter()
            .map(|binding| Triple {
                subject: binding.get("s").map(|t| t.to_string()).unwrap_or_default(),
                predicate: binding.get("p").map(|t| t.to_string()).unwrap_or_default(),
                object: binding.get("o").map(|t| t.to_string()).unwrap_or_default(),
                object_type: "uri".to_string(), // Simplified
            })
            .collect(),
        _ => Vec::new(), // Handle other query types
    };

    let total = store.count_triples(&name);

    Ok(Json(TripleList {
        triples,
        total,
        limit: pagination.limit,
        offset: pagination.offset,
    }))
}

/// Insert triples into a dataset
///
/// Inserts one or more triples into the specified dataset.
#[utoipa::path(
    post,
    path = "/api/v2/datasets/{name}/triples",
    request_body = InsertTripleRequest,
    responses(
        (status = 201, description = "Triples inserted"),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Dataset not found", body = ErrorResponse)
    ),
    params(
        ("name" = String, Path, description = "Dataset name")
    ),
    tag = "triples"
)]
pub async fn insert_triple(
    State(store): State<Arc<crate::store::Store>>,
    Path(name): Path<String>,
    Json(request): Json<InsertTripleRequest>,
) -> Result<StatusCode, ApiError> {
    if !store.dataset_exists(&name) {
        return Err(ApiError::NotFound(format!("Dataset '{}' not found", name)));
    }

    // Build SPARQL INSERT query
    let triples_str = request
        .triples
        .iter()
        .map(|t| format!("<{}> <{}> <{}> .", t.subject, t.predicate, t.object))
        .collect::<Vec<_>>()
        .join(" ");

    let query = format!("INSERT DATA {{ {} }}", triples_str);

    store
        .update_dataset(&query, Some(&name))
        .map_err(|e| ApiError::BadRequest(format!("Insert failed: {}", e)))?;

    info!(
        "Inserted {} triples into dataset: {}",
        request.triples.len(),
        name
    );

    Ok(StatusCode::CREATED)
}

/// Delete triples from a dataset
///
/// Deletes triples matching the specified pattern.
#[utoipa::path(
    delete,
    path = "/api/v2/datasets/{name}/triples",
    request_body = DeleteTripleRequest,
    responses(
        (status = 204, description = "Triples deleted"),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Dataset not found", body = ErrorResponse)
    ),
    params(
        ("name" = String, Path, description = "Dataset name")
    ),
    tag = "triples"
)]
pub async fn delete_triple(
    State(store): State<Arc<crate::store::Store>>,
    Path(name): Path<String>,
    Json(request): Json<DeleteTripleRequest>,
) -> Result<StatusCode, ApiError> {
    if !store.dataset_exists(&name) {
        return Err(ApiError::NotFound(format!("Dataset '{}' not found", name)));
    }

    // Build SPARQL DELETE query
    let s = request.subject.as_deref().unwrap_or("?s");
    let p = request.predicate.as_deref().unwrap_or("?p");
    let o = request.object.as_deref().unwrap_or("?o");

    let query = format!("DELETE WHERE {{ {} {} {} }}", s, p, o);

    store
        .update_dataset(&query, Some(&name))
        .map_err(|e| ApiError::BadRequest(format!("Delete failed: {}", e)))?;

    info!("Deleted triples from dataset: {}", name);

    Ok(StatusCode::NO_CONTENT)
}

/// Get system statistics
///
/// Returns statistics about the system and its performance.
#[utoipa::path(
    get,
    path = "/api/v2/statistics",
    responses(
        (status = 200, description = "System statistics", body = Statistics)
    ),
    tag = "statistics"
)]
pub async fn get_statistics(
    State(store): State<Arc<crate::store::Store>>,
) -> Result<Json<Statistics>, ApiError> {
    let datasets = store.list_datasets().unwrap_or_default();
    let total_triples: usize = datasets.iter().map(|ds| store.count_triples(ds)).sum();

    Ok(Json(Statistics {
        dataset_count: datasets.len(),
        total_triples,
        total_queries: 0, // Would need metrics integration
        avg_query_time_ms: 0.0,
        uptime_seconds: 0,     // Would need startup time tracking
        memory_usage_bytes: 0, // Would need system metrics
    }))
}

/// Health check
///
/// Returns the health status of the system and its components.
#[utoipa::path(
    get,
    path = "/api/v2/health",
    responses(
        (status = 200, description = "Health status", body = HealthStatus)
    ),
    tag = "health"
)]
pub async fn get_health(
    State(store): State<Arc<crate::store::Store>>,
) -> Result<Json<HealthStatus>, ApiError> {
    let mut components = std::collections::HashMap::new();

    // Check store health
    let store_status = if store.list_datasets().is_ok() {
        ComponentHealth {
            status: "healthy".to_string(),
            details: None,
        }
    } else {
        ComponentHealth {
            status: "unhealthy".to_string(),
            details: Some("Failed to access store".to_string()),
        }
    };
    components.insert("store".to_string(), store_status);

    let overall_status = if components.values().all(|c| c.status == "healthy") {
        "healthy"
    } else {
        "degraded"
    };

    Ok(Json(HealthStatus {
        status: overall_status.to_string(),
        timestamp: chrono::Utc::now(),
        components,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_info_serialization() {
        let info = ApiInfo {
            version: "2.0.0".to_string(),
            name: "Test API".to_string(),
            description: "Test".to_string(),
            endpoints: vec![],
            features: vec![],
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("2.0.0"));
    }

    #[test]
    fn test_error_response() {
        let error = ErrorResponse {
            code: "TEST_ERROR".to_string(),
            message: "Test error".to_string(),
            details: None,
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(error.code, "TEST_ERROR");
    }
}
