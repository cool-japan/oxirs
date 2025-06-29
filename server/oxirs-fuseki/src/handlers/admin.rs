//! Administrative handlers for server and dataset management

use crate::{
    auth::{AuthUser, Permission},
    config::{DatasetConfig, ServerConfig},
    error::{FusekiError, FusekiResult},
    server::AppState,
    store::Store,
};
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, error, info, instrument, warn};

/// Dataset creation request
#[derive(Debug, Deserialize)]
pub struct CreateDatasetRequest {
    pub name: String,
    pub location: Option<String>,
    pub read_only: Option<bool>,
    pub description: Option<String>,
}

/// Dataset information response
#[derive(Debug, Serialize)]
pub struct DatasetInfo {
    pub name: String,
    pub location: String,
    pub read_only: bool,
    pub description: Option<String>,
    pub created_at: String,
    pub last_modified: String,
    pub triple_count: u64,
    pub size_bytes: u64,
    pub services: Vec<ServiceInfo>,
}

/// Service information
#[derive(Debug, Serialize)]
pub struct ServiceInfo {
    pub name: String,
    pub endpoint: String,
    pub service_type: String,
    pub description: String,
}

/// Server statistics
#[derive(Debug, Serialize)]
pub struct ServerStats {
    pub server_name: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub datasets_count: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub active_connections: u32,
    pub last_updated: String,
}

/// Backup request parameters
#[derive(Debug, Deserialize)]
pub struct BackupParams {
    pub format: Option<String>,
    pub compress: Option<bool>,
    pub include_metadata: Option<bool>,
}

/// Compact request parameters
#[derive(Debug, Deserialize)]
pub struct CompactParams {
    pub force: Option<bool>,
}

/// Administrative UI handler
#[instrument(skip(state))]
pub async fn ui_handler(State(state): State<AppState>) -> Result<Html<String>, FusekiError> {
    // Check if admin UI is enabled
    if !state.config.server.admin_ui {
        return Err(FusekiError::not_found("Admin UI is disabled"));
    }

    // Generate basic admin UI HTML
    let html_content = generate_admin_ui_html(&state).await?;

    Ok(Html(html_content))
}

/// List all datasets
#[instrument(skip(state))]
pub async fn list_datasets(
    State(state): State<AppState>,
    // auth_user: AuthUser, // Would be used in full implementation
) -> Result<Json<Vec<DatasetInfo>>, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemMetrics)?;

    let mut datasets = Vec::new();

    for (name, config) in &state.config.datasets {
        let dataset_info = get_dataset_info(name, config, &state.store).await?;
        datasets.push(dataset_info);
    }

    info!("Listed {} datasets", datasets.len());
    Ok(Json(datasets))
}

/// Get specific dataset information
#[instrument(skip(state))]
pub async fn get_dataset(
    State(state): State<AppState>,
    Path(dataset_name): Path<String>,
    // auth_user: AuthUser,
) -> Result<Json<DatasetInfo>, FusekiError> {
    // Check permissions
    // check_dataset_permission(&auth_user, &dataset_name, &Permission::DatasetRead)?;

    let config =
        state.config.datasets.get(&dataset_name).ok_or_else(|| {
            FusekiError::not_found(format!("Dataset '{}' not found", dataset_name))
        })?;

    let dataset_info = get_dataset_info(&dataset_name, config, &state.store).await?;

    Ok(Json(dataset_info))
}

/// Create new dataset
#[instrument(skip(state))]
pub async fn create_dataset(
    State(state): State<AppState>,
    // auth_user: AuthUser,
    Json(request): Json<CreateDatasetRequest>,
) -> Result<Json<DatasetInfo>, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Validate dataset name
    validate_dataset_name(&request.name)?;

    // Check if dataset already exists
    if state.config.datasets.contains_key(&request.name) {
        return Err(FusekiError::conflict(format!(
            "Dataset '{}' already exists",
            request.name
        )));
    }

    // Create dataset configuration
    let dataset_config = DatasetConfig {
        name: request.name.clone(),
        location: request
            .location
            .unwrap_or_else(|| format!("/data/{}", request.name)),
        read_only: request.read_only.unwrap_or(false),
        text_index: None,
        shacl_shapes: Vec::new(),
        services: vec![
            crate::config::ServiceConfig {
                name: "query".to_string(),
                service_type: crate::config::ServiceType::SparqlQuery,
                endpoint: format!("/{}/sparql", request.name),
                auth_required: false,
                rate_limit: None,
            },
            crate::config::ServiceConfig {
                name: "update".to_string(),
                service_type: crate::config::ServiceType::SparqlUpdate,
                endpoint: format!("/{}/update", request.name),
                auth_required: false,
                rate_limit: None,
            },
        ],
        access_control: None,
        backup: None,
    };

    // Create dataset in store
    create_dataset_in_store(&state.store, &request.name, &dataset_config).await?;

    // Generate dataset info
    let dataset_info = get_dataset_info(&request.name, &dataset_config, &state.store).await?;

    info!("Created dataset: {}", request.name);
    Ok(Json(dataset_info))
}

/// Delete dataset
#[instrument(skip(state))]
pub async fn delete_dataset(
    State(state): State<AppState>,
    Path(dataset_name): Path<String>,
    // auth_user: AuthUser,
) -> Result<StatusCode, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Check if dataset exists
    if !state.config.datasets.contains_key(&dataset_name) {
        return Err(FusekiError::not_found(format!(
            "Dataset '{}' not found",
            dataset_name
        )));
    }

    // Delete dataset from store
    delete_dataset_from_store(&state.store, &dataset_name).await?;

    info!("Deleted dataset: {}", dataset_name);
    Ok(StatusCode::NO_CONTENT)
}

/// Get server information
#[instrument(skip(state))]
pub async fn server_info(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, FusekiError> {
    let mut info = HashMap::new();

    info.insert("name", serde_json::json!("OxiRS Fuseki"));
    info.insert("version", serde_json::json!(env!("CARGO_PKG_VERSION")));
    info.insert(
        "description",
        serde_json::json!(
            "SPARQL 1.1/1.2 HTTP protocol server with Fuseki-compatible configuration"
        ),
    );
    info.insert(
        "built_at",
        serde_json::json!(option_env!("VERGEN_BUILD_TIMESTAMP").unwrap_or("unknown")),
    );
    info.insert(
        "features",
        serde_json::json!({
            "authentication": state.config.security.authentication.enabled,
            "metrics": state.config.monitoring.metrics.enabled,
            "admin_ui": state.config.server.admin_ui,
            "cors": state.config.server.cors,
        }),
    );
    info.insert(
        "datasets_count",
        serde_json::json!(state.config.datasets.len()),
    );

    if let Some(metrics_service) = &state.metrics_service {
        let summary = metrics_service.get_summary().await;
        info.insert("uptime_seconds", serde_json::json!(summary.uptime_seconds));
        info.insert("requests_total", serde_json::json!(summary.requests_total));
        info.insert("system_metrics", serde_json::json!(summary.system));
    }

    Ok(Json(serde_json::Value::Object(
        info.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
    )))
}

/// Get server statistics
#[instrument(skip(state))]
pub async fn server_stats(State(state): State<AppState>) -> Result<Json<ServerStats>, FusekiError> {
    let stats = if let Some(metrics_service) = &state.metrics_service {
        let summary = metrics_service.get_summary().await;

        ServerStats {
            server_name: "OxiRS Fuseki".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: summary.uptime_seconds,
            total_requests: summary.requests_total,
            datasets_count: state.config.datasets.len(),
            memory_usage_mb: summary.system.memory_usage_bytes as f64 / 1024.0 / 1024.0,
            cpu_usage_percent: summary.system.cpu_usage_percent,
            active_connections: summary.active_connections as u32,
            last_updated: chrono::Utc::now().to_rfc3339(),
        }
    } else {
        ServerStats {
            server_name: "OxiRS Fuseki".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: 0,
            total_requests: 0,
            datasets_count: state.config.datasets.len(),
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            active_connections: 0,
            last_updated: chrono::Utc::now().to_rfc3339(),
        }
    };

    Ok(Json(stats))
}

/// Compact dataset
#[instrument(skip(state))]
pub async fn compact_dataset(
    State(state): State<AppState>,
    Path(dataset_name): Path<String>,
    Query(params): Query<CompactParams>,
    // auth_user: AuthUser,
) -> Result<Json<serde_json::Value>, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Check if dataset exists
    if !state.config.datasets.contains_key(&dataset_name) {
        return Err(FusekiError::not_found(format!(
            "Dataset '{}' not found",
            dataset_name
        )));
    }

    let start_time = Instant::now();

    // Perform compaction
    let result =
        compact_dataset_in_store(&state.store, &dataset_name, params.force.unwrap_or(false))
            .await?;

    let execution_time = start_time.elapsed();

    info!(
        "Compacted dataset '{}' in {}ms",
        dataset_name,
        execution_time.as_millis()
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "dataset": dataset_name,
        "execution_time_ms": execution_time.as_millis(),
        "size_before_bytes": result.size_before,
        "size_after_bytes": result.size_after,
        "space_saved_bytes": result.size_before - result.size_after,
        "message": "Dataset compaction completed successfully"
    })))
}

/// Backup dataset
#[instrument(skip(state))]
pub async fn backup_dataset(
    State(state): State<AppState>,
    Path(dataset_name): Path<String>,
    Query(params): Query<BackupParams>,
    // auth_user: AuthUser,
) -> Result<Response, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Check if dataset exists
    if !state.config.datasets.contains_key(&dataset_name) {
        return Err(FusekiError::not_found(format!(
            "Dataset '{}' not found",
            dataset_name
        )));
    }

    let format = params.format.as_deref().unwrap_or("turtle");
    let compress = params.compress.unwrap_or(false);
    let include_metadata = params.include_metadata.unwrap_or(true);

    let start_time = Instant::now();

    // Create backup
    let backup_data = create_dataset_backup(
        &state.store,
        &dataset_name,
        format,
        compress,
        include_metadata,
    )
    .await?;

    let execution_time = start_time.elapsed();

    info!(
        "Created backup for dataset '{}' in {}ms",
        dataset_name,
        execution_time.as_millis()
    );

    // Determine content type and filename
    let (content_type, filename) = match format {
        "turtle" => ("text/turtle", format!("{}_backup.ttl", dataset_name)),
        "ntriples" => (
            "application/n-triples",
            format!("{}_backup.nt", dataset_name),
        ),
        "rdfxml" => (
            "application/rdf+xml",
            format!("{}_backup.rdf", dataset_name),
        ),
        _ => ("text/turtle", format!("{}_backup.ttl", dataset_name)),
    };

    let headers = [
        ("content-type", content_type),
        (
            "content-disposition",
            &format!("attachment; filename=\"{}\"", filename),
        ),
    ];

    Ok((StatusCode::OK, headers, backup_data).into_response())
}

// Helper functions

/// Generate basic admin UI HTML
async fn generate_admin_ui_html(state: &AppState) -> FusekiResult<String> {
    let datasets_count = state.config.datasets.len();

    let html = format!(
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OxiRS Fuseki Admin</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007cba; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; padding: 20px; border-left: 4px solid #007cba; background: #f9f9f9; }}
        .endpoint {{ background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 4px; font-family: monospace; }}
        .status {{ padding: 5px 10px; border-radius: 4px; color: white; }}
        .status.enabled {{ background-color: #28a745; }}
        .status.disabled {{ background-color: #dc3545; }}
        a {{ color: #007cba; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🦀 OxiRS Fuseki Server</h1>
        
        <div class="section">
            <h2>Server Information</h2>
            <p><strong>Version:</strong> {}</p>
            <p><strong>Datasets:</strong> {}</p>
            <p><strong>Authentication:</strong> <span class="status {}">{}</span></p>
            <p><strong>Metrics:</strong> <span class="status {}">{}</span></p>
        </div>

        <div class="section">
            <h2>API Endpoints</h2>
            <div class="endpoint">GET <a href="/health">/health</a> - Health check</div>
            <div class="endpoint">GET <a href="/$/server">/$/server</a> - Server information</div>
            <div class="endpoint">GET <a href="/$/stats">/$/stats</a> - Server statistics</div>
            <div class="endpoint">GET <a href="/$/datasets">/$/datasets</a> - List datasets</div>
            <div class="endpoint">GET/POST <a href="/sparql">/sparql</a> - SPARQL Query endpoint</div>
            <div class="endpoint">POST <a href="/update">/update</a> - SPARQL Update endpoint</div>
            {}
        </div>

        <div class="section">
            <h2>Quick Actions</h2>
            <p><a href="/$/ping">Ping Server</a></p>
            <p><a href="/metrics">View Metrics</a></p>
            <p><a href="/health">Check Health</a></p>
        </div>
    </div>
</body>
</html>
    "#,
        env!("CARGO_PKG_VERSION"),
        datasets_count,
        if state.config.security.authentication.enabled {
            "enabled"
        } else {
            "disabled"
        },
        if state.config.security.authentication.enabled {
            "ENABLED"
        } else {
            "DISABLED"
        },
        if state.config.monitoring.metrics.enabled {
            "enabled"
        } else {
            "disabled"
        },
        if state.config.monitoring.metrics.enabled {
            "ENABLED"
        } else {
            "DISABLED"
        },
        if state.config.monitoring.metrics.enabled {
            r#"<div class="endpoint">GET <a href="/metrics">/metrics</a> - Prometheus metrics</div>"#
        } else {
            ""
        }
    );

    Ok(html)
}

/// Get dataset information
async fn get_dataset_info(
    name: &str,
    config: &DatasetConfig,
    store: &Store,
) -> FusekiResult<DatasetInfo> {
    // Get dataset statistics from store
    let stats = get_dataset_stats_from_store(store, name).await?;

    let services = config
        .services
        .iter()
        .map(|service| ServiceInfo {
            name: service.name.clone(),
            endpoint: service.endpoint.clone(),
            service_type: format!("{:?}", service.service_type),
            description: format!("{:?} service", service.service_type),
        })
        .collect();

    Ok(DatasetInfo {
        name: name.to_string(),
        location: config.location.clone(),
        read_only: config.read_only,
        description: None,                              // Could be added to config
        created_at: chrono::Utc::now().to_rfc3339(),    // Mock data
        last_modified: chrono::Utc::now().to_rfc3339(), // Mock data
        triple_count: stats.triple_count,
        size_bytes: stats.size_bytes,
        services,
    })
}

/// Validate dataset name
fn validate_dataset_name(name: &str) -> FusekiResult<()> {
    if name.is_empty() {
        return Err(FusekiError::bad_request("Dataset name cannot be empty"));
    }

    if name.len() > 64 {
        return Err(FusekiError::bad_request(
            "Dataset name too long (max 64 characters)",
        ));
    }

    // Check for valid characters (alphanumeric, dash, underscore)
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(FusekiError::bad_request(
            "Dataset name contains invalid characters",
        ));
    }

    // Cannot start with dash
    if name.starts_with('-') {
        return Err(FusekiError::bad_request(
            "Dataset name cannot start with dash",
        ));
    }

    Ok(())
}

// Mock store operations (to be replaced with actual implementations)

struct DatasetStats {
    triple_count: u64,
    size_bytes: u64,
}

struct CompactionResult {
    size_before: u64,
    size_after: u64,
}

async fn get_dataset_stats_from_store(store: &Store, name: &str) -> FusekiResult<DatasetStats> {
    // Mock implementation
    Ok(DatasetStats {
        triple_count: 1000, // Mock data
        size_bytes: 50000,  // Mock data
    })
}

async fn create_dataset_in_store(
    store: &Store,
    name: &str,
    config: &DatasetConfig,
) -> FusekiResult<()> {
    // Mock implementation
    debug!(
        "Creating dataset '{}' at location '{}'",
        name, config.location
    );
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    Ok(())
}

async fn delete_dataset_from_store(store: &Store, name: &str) -> FusekiResult<()> {
    // Mock implementation
    debug!("Deleting dataset '{}'", name);
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    Ok(())
}

async fn compact_dataset_in_store(
    store: &Store,
    name: &str,
    force: bool,
) -> FusekiResult<CompactionResult> {
    // Mock implementation
    debug!("Compacting dataset '{}' (force: {})", name, force);
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    Ok(CompactionResult {
        size_before: 100000,
        size_after: 75000,
    })
}

async fn create_dataset_backup(
    store: &Store,
    name: &str,
    format: &str,
    compress: bool,
    include_metadata: bool,
) -> FusekiResult<String> {
    // Mock implementation
    debug!(
        "Creating backup for dataset '{}' in format '{}' (compress: {}, metadata: {})",
        name, format, compress, include_metadata
    );

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Return mock backup data
    match format {
        "turtle" => Ok("@prefix ex: <http://example.org/> .\nex:subject ex:predicate \"backup data\" .".to_string()),
        "ntriples" => Ok("<http://example.org/subject> <http://example.org/predicate> \"backup data\" .".to_string()),
        "rdfxml" => Ok("<?xml version=\"1.0\"?>\n<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n</rdf:RDF>".to_string()),
        _ => Ok("# Backup data\n".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_name_validation() {
        // Valid names
        assert!(validate_dataset_name("test").is_ok());
        assert!(validate_dataset_name("test-dataset").is_ok());
        assert!(validate_dataset_name("test_dataset").is_ok());
        assert!(validate_dataset_name("test123").is_ok());

        // Invalid names
        assert!(validate_dataset_name("").is_err());
        assert!(validate_dataset_name("-test").is_err());
        assert!(validate_dataset_name("test/dataset").is_err());
        assert!(validate_dataset_name("test dataset").is_err());
        assert!(validate_dataset_name(&"x".repeat(65)).is_err());
    }
}
