//! Administrative handlers for server and dataset management

use crate::{
    config::DatasetConfig,
    error::{FusekiError, FusekiResult},
    server::AppState,
    store::Store,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};

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
pub async fn ui_handler(State(state): State<Arc<AppState>>) -> Result<Html<String>, FusekiError> {
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
    State(state): State<Arc<AppState>>,
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
    State(state): State<Arc<AppState>>,
    Path(dataset_name): Path<String>,
    // auth_user: AuthUser,
) -> Result<Json<DatasetInfo>, FusekiError> {
    // Check permissions
    // check_dataset_permission(&auth_user, &dataset_name, &Permission::DatasetRead)?;

    let config = state
        .config
        .datasets
        .get(&dataset_name)
        .ok_or_else(|| FusekiError::not_found(format!("Dataset '{dataset_name}' not found")))?;

    let dataset_info = get_dataset_info(&dataset_name, config, &state.store).await?;

    Ok(Json(dataset_info))
}

/// Create new dataset
#[instrument(skip(state))]
pub async fn create_dataset(
    State(state): State<Arc<AppState>>,
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
    State(state): State<Arc<AppState>>,
    Path(dataset_name): Path<String>,
    // auth_user: AuthUser,
) -> Result<StatusCode, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Check if dataset exists
    if !state.config.datasets.contains_key(&dataset_name) {
        return Err(FusekiError::not_found(format!(
            "Dataset '{dataset_name}' not found"
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
    State(state): State<Arc<AppState>>,
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
pub async fn server_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ServerStats>, FusekiError> {
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
    State(state): State<Arc<AppState>>,
    Path(dataset_name): Path<String>,
    Query(params): Query<CompactParams>,
    // auth_user: AuthUser,
) -> Result<Json<serde_json::Value>, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Check if dataset exists
    if !state.config.datasets.contains_key(&dataset_name) {
        return Err(FusekiError::not_found(format!(
            "Dataset '{dataset_name}' not found"
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
    State(state): State<Arc<AppState>>,
    Path(dataset_name): Path<String>,
    Query(params): Query<BackupParams>,
    // auth_user: AuthUser,
) -> Result<Response, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Check if dataset exists
    if !state.config.datasets.contains_key(&dataset_name) {
        return Err(FusekiError::not_found(format!(
            "Dataset '{dataset_name}' not found"
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
        "turtle" => ("text/turtle", format!("{dataset_name}_backup.ttl")),
        "ntriples" => ("application/n-triples", format!("{dataset_name}_backup.nt")),
        "rdfxml" => ("application/rdf+xml", format!("{dataset_name}_backup.rdf")),
        _ => ("text/turtle", format!("{dataset_name}_backup.ttl")),
    };

    let headers = [
        ("content-type", content_type),
        (
            "content-disposition",
            &format!("attachment; filename=\"{filename}\""),
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

async fn get_dataset_stats_from_store(_store: &Store, _name: &str) -> FusekiResult<DatasetStats> {
    // Mock implementation
    Ok(DatasetStats {
        triple_count: 1000, // Mock data
        size_bytes: 50000,  // Mock data
    })
}

async fn create_dataset_in_store(
    _store: &Store,
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

async fn delete_dataset_from_store(_store: &Store, name: &str) -> FusekiResult<()> {
    // Mock implementation
    debug!("Deleting dataset '{}'", name);
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    Ok(())
}

async fn compact_dataset_in_store(
    _store: &Store,
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
    _store: &Store,
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

// ============================================================================
// Fuseki-compatible administrative endpoints
// ============================================================================

/// Backup file information
#[derive(Debug, Serialize)]
pub struct BackupFileInfo {
    /// Backup file name
    pub filename: String,
    /// File size in bytes
    pub size: u64,
    /// Creation timestamp (ISO 8601)
    pub created: String,
    /// Dataset name (if identifiable from filename)
    pub dataset: Option<String>,
    /// Backup format
    pub format: Option<String>,
    /// Whether the backup is compressed
    pub compressed: bool,
}

/// List available backups response
#[derive(Debug, Serialize)]
pub struct BackupListResponse {
    /// List of available backup files
    pub backups: Vec<BackupFileInfo>,
    /// Backup directory path
    pub backup_directory: String,
    /// Total count of backups
    pub count: usize,
    /// Total size of all backups in bytes
    pub total_size: u64,
}

/// List all available backups
///
/// Fuseki-compatible endpoint: GET /$/backups-list
#[instrument(skip(state))]
pub async fn list_backups(
    State(state): State<Arc<AppState>>,
) -> Result<Json<BackupListResponse>, FusekiError> {
    info!("Listing available backups");

    // Get the backup directory from config or use default
    let backup_dir = state
        .config
        .server
        .backup_directory
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("./backups"));

    let mut backups = Vec::new();
    let mut total_size = 0u64;

    // Check if backup directory exists
    if backup_dir.exists() && backup_dir.is_dir() {
        // Read directory entries
        if let Ok(entries) = std::fs::read_dir(&backup_dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                // Only include files, not directories
                if path.is_file() {
                    if let Ok(metadata) = entry.metadata() {
                        let filename = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown")
                            .to_string();

                        let size = metadata.len();
                        total_size += size;

                        // Extract dataset name from filename (e.g., "dataset_2024-01-15T10-30-00.nq.gz")
                        let dataset = extract_dataset_from_filename(&filename);

                        // Detect format from extension
                        let (format, compressed) = detect_backup_format(&filename);

                        // Get creation time
                        let created = metadata
                            .created()
                            .or_else(|_| metadata.modified())
                            .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339())
                            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());

                        backups.push(BackupFileInfo {
                            filename,
                            size,
                            created,
                            dataset,
                            format,
                            compressed,
                        });
                    }
                }
            }
        }
    }

    // Sort by creation time (newest first)
    backups.sort_by(|a, b| b.created.cmp(&a.created));

    let count = backups.len();

    Ok(Json(BackupListResponse {
        backups,
        backup_directory: backup_dir.to_string_lossy().to_string(),
        count,
        total_size,
    }))
}

/// Extract dataset name from backup filename
fn extract_dataset_from_filename(filename: &str) -> Option<String> {
    // Common patterns: "dataset_2024-01-15.nq.gz", "mydb-backup-20240115.ttl"
    let name = filename
        .strip_suffix(".gz")
        .or(Some(filename))
        .unwrap_or(filename);

    let name = name
        .strip_suffix(".nq")
        .or_else(|| name.strip_suffix(".ttl"))
        .or_else(|| name.strip_suffix(".nt"))
        .or_else(|| name.strip_suffix(".rdf"))
        .or_else(|| name.strip_suffix(".xml"))
        .or_else(|| name.strip_suffix(".trig"))
        .unwrap_or(name);

    // Try to extract dataset name before date pattern
    // Pattern: dataset_YYYY-MM-DD or dataset-backup-YYYYMMDD
    if let Some(idx) = name.find('_') {
        let potential_name = &name[..idx];
        if !potential_name.is_empty()
            && potential_name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-')
        {
            return Some(potential_name.to_string());
        }
    }

    if let Some(idx) = name.find("-backup") {
        let potential_name = &name[..idx];
        if !potential_name.is_empty() {
            return Some(potential_name.to_string());
        }
    }

    None
}

/// Detect backup format from filename extension
fn detect_backup_format(filename: &str) -> (Option<String>, bool) {
    let compressed =
        filename.ends_with(".gz") || filename.ends_with(".zip") || filename.ends_with(".zst");

    let name = if compressed {
        filename
            .strip_suffix(".gz")
            .or_else(|| filename.strip_suffix(".zip"))
            .or_else(|| filename.strip_suffix(".zst"))
            .unwrap_or(filename)
    } else {
        filename
    };

    let format = if name.ends_with(".nq") || name.ends_with(".nquads") {
        Some("N-Quads".to_string())
    } else if name.ends_with(".nt") || name.ends_with(".ntriples") {
        Some("N-Triples".to_string())
    } else if name.ends_with(".ttl") || name.ends_with(".turtle") {
        Some("Turtle".to_string())
    } else if name.ends_with(".rdf") || name.ends_with(".xml") || name.ends_with(".rdfxml") {
        Some("RDF/XML".to_string())
    } else if name.ends_with(".trig") {
        Some("TriG".to_string())
    } else if name.ends_with(".jsonld") || name.ends_with(".json") {
        Some("JSON-LD".to_string())
    } else {
        None
    };

    (format, compressed)
}

/// Reload configuration response
#[derive(Debug, Serialize)]
pub struct ReloadResponse {
    /// Whether reload was successful
    pub success: bool,
    /// Status message
    pub message: String,
    /// Configuration file path
    pub config_file: Option<String>,
    /// Changes detected
    pub changes: Vec<String>,
    /// Timestamp of reload
    pub timestamp: String,
}

/// Reload server configuration
///
/// Fuseki-compatible endpoint: POST /$/reload
///
/// This endpoint triggers a hot-reload of the server configuration.
/// Note: Not all configuration changes can be applied without a restart.
#[instrument(skip(state))]
pub async fn reload_config(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ReloadResponse>, FusekiError> {
    info!("Configuration reload requested");

    let config_file = state
        .config
        .server
        .config_file
        .clone()
        .map(|p| p.to_string_lossy().to_string());

    // Check if we have a config file to reload from
    let config_path = match &state.config.server.config_file {
        Some(path) => path.clone(),
        None => {
            return Ok(Json(ReloadResponse {
                success: false,
                message:
                    "No configuration file specified. Server was started without a config file."
                        .to_string(),
                config_file: None,
                changes: vec![],
                timestamp: chrono::Utc::now().to_rfc3339(),
            }));
        }
    };

    // Check if config file exists
    if !config_path.exists() {
        return Ok(Json(ReloadResponse {
            success: false,
            message: format!("Configuration file not found: {}", config_path.display()),
            config_file,
            changes: vec![],
            timestamp: chrono::Utc::now().to_rfc3339(),
        }));
    }

    // Read and parse the new configuration
    let config_content = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(e) => {
            return Ok(Json(ReloadResponse {
                success: false,
                message: format!("Failed to read configuration file: {}", e),
                config_file,
                changes: vec![],
                timestamp: chrono::Utc::now().to_rfc3339(),
            }));
        }
    };

    // Parse the configuration (assuming TOML format)
    let new_config: crate::config::ServerConfig = match toml::from_str(&config_content) {
        Ok(config) => config,
        Err(e) => {
            return Ok(Json(ReloadResponse {
                success: false,
                message: format!("Failed to parse configuration: {}", e),
                config_file,
                changes: vec![],
                timestamp: chrono::Utc::now().to_rfc3339(),
            }));
        }
    };

    // Detect changes between current and new configuration
    let mut changes = Vec::new();

    // Check for dataset changes
    let current_datasets: std::collections::HashSet<_> = state.config.datasets.keys().collect();
    let new_datasets: std::collections::HashSet<_> = new_config.datasets.keys().collect();

    for name in new_datasets.difference(&current_datasets) {
        changes.push(format!("New dataset added: {}", name));
    }

    for name in current_datasets.difference(&new_datasets) {
        changes.push(format!("Dataset removed: {}", name));
    }

    // Check for server config changes
    if state.config.server.port != new_config.server.port {
        changes.push(format!(
            "Port changed: {} -> {} (requires restart)",
            state.config.server.port, new_config.server.port
        ));
    }

    if state.config.server.host != new_config.server.host {
        changes.push(format!(
            "Host changed: {} -> {} (requires restart)",
            state.config.server.host, new_config.server.host
        ));
    }

    // Check security changes
    if state.config.security.authentication.enabled != new_config.security.authentication.enabled {
        changes.push(format!(
            "Authentication: {} -> {}",
            if state.config.security.authentication.enabled {
                "enabled"
            } else {
                "disabled"
            },
            if new_config.security.authentication.enabled {
                "enabled"
            } else {
                "disabled"
            }
        ));
    }

    // Check monitoring changes
    if state.config.monitoring.metrics.enabled != new_config.monitoring.metrics.enabled {
        changes.push(format!(
            "Metrics: {} -> {}",
            if state.config.monitoring.metrics.enabled {
                "enabled"
            } else {
                "disabled"
            },
            if new_config.monitoring.metrics.enabled {
                "enabled"
            } else {
                "disabled"
            }
        ));
    }

    // Note: Actually applying the changes would require mutable access to the config
    // In a real implementation, this would update a RwLock<Config> or similar
    // For now, we report the changes that would be applied

    let message = if changes.is_empty() {
        "Configuration reloaded. No changes detected.".to_string()
    } else {
        format!(
            "Configuration parsed successfully. {} change(s) detected. Some changes may require a server restart to take effect.",
            changes.len()
        )
    };

    info!("Configuration reload completed: {}", message);

    Ok(Json(ReloadResponse {
        success: true,
        message,
        config_file,
        changes,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }))
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

    #[test]
    fn test_extract_dataset_from_filename() {
        // Pattern: dataset_timestamp.format
        assert_eq!(
            extract_dataset_from_filename("mydb_2024-01-15.nq.gz"),
            Some("mydb".to_string())
        );
        assert_eq!(
            extract_dataset_from_filename("test-data_2024-01-15T10-30-00.ttl"),
            Some("test-data".to_string())
        );

        // Pattern: dataset-backup-timestamp
        assert_eq!(
            extract_dataset_from_filename("mydb-backup-20240115.nq"),
            Some("mydb".to_string())
        );

        // No pattern match
        assert_eq!(extract_dataset_from_filename("random-file.txt"), None);
        assert_eq!(extract_dataset_from_filename("nopattern.nq"), None);
    }

    #[test]
    fn test_detect_backup_format() {
        // N-Quads
        assert_eq!(
            detect_backup_format("backup.nq"),
            (Some("N-Quads".to_string()), false)
        );
        assert_eq!(
            detect_backup_format("backup.nq.gz"),
            (Some("N-Quads".to_string()), true)
        );

        // N-Triples
        assert_eq!(
            detect_backup_format("backup.nt"),
            (Some("N-Triples".to_string()), false)
        );

        // Turtle
        assert_eq!(
            detect_backup_format("backup.ttl"),
            (Some("Turtle".to_string()), false)
        );
        assert_eq!(
            detect_backup_format("backup.ttl.gz"),
            (Some("Turtle".to_string()), true)
        );

        // RDF/XML
        assert_eq!(
            detect_backup_format("backup.rdf"),
            (Some("RDF/XML".to_string()), false)
        );

        // TriG
        assert_eq!(
            detect_backup_format("backup.trig"),
            (Some("TriG".to_string()), false)
        );

        // JSON-LD
        assert_eq!(
            detect_backup_format("backup.jsonld"),
            (Some("JSON-LD".to_string()), false)
        );

        // Unknown format
        assert_eq!(detect_backup_format("backup.xyz"), (None, false));

        // Compressed with various algorithms
        assert_eq!(
            detect_backup_format("backup.nq.zip"),
            (Some("N-Quads".to_string()), true)
        );
        assert_eq!(
            detect_backup_format("backup.ttl.zst"),
            (Some("Turtle".to_string()), true)
        );
    }

    #[test]
    fn test_backup_file_info_serialization() {
        let info = BackupFileInfo {
            filename: "test_2024-01-15.nq.gz".to_string(),
            size: 1024,
            created: "2024-01-15T10:30:00Z".to_string(),
            dataset: Some("test".to_string()),
            format: Some("N-Quads".to_string()),
            compressed: true,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"filename\":\"test_2024-01-15.nq.gz\""));
        assert!(json.contains("\"size\":1024"));
        assert!(json.contains("\"compressed\":true"));
    }

    #[test]
    fn test_backup_list_response_serialization() {
        let response = BackupListResponse {
            backups: vec![],
            backup_directory: "./backups".to_string(),
            count: 0,
            total_size: 0,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"backup_directory\":\"./backups\""));
        assert!(json.contains("\"count\":0"));
    }

    #[test]
    fn test_reload_response_serialization() {
        let response = ReloadResponse {
            success: true,
            message: "Configuration reloaded".to_string(),
            config_file: Some("/etc/oxirs/config.toml".to_string()),
            changes: vec!["Port changed: 3030 -> 3031".to_string()],
            timestamp: "2024-01-15T10:30:00Z".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"message\":\"Configuration reloaded\""));
        assert!(json.contains("Port changed"));
    }
}
