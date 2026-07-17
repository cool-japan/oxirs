//! Administrative handlers for server and dataset management

use crate::{
    config::DatasetConfig,
    error::{FusekiError, FusekiResult},
    server::AppState,
    store::{CoreRdfFormat, CoreStore, Serializer, Store},
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
use tracing::{debug, info, instrument, warn};

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
///
/// Unions the statically configured datasets (`state.config.datasets`, from
/// the server's config file) with any datasets registered at runtime through
/// this API (tracked persistently by `state.dataset_manager`), so datasets
/// created via `POST /$/datasets/:name` actually show up afterwards.
#[instrument(skip(state))]
pub async fn list_datasets(
    State(state): State<Arc<AppState>>,
    // auth_user: AuthUser, // Would be used in full implementation
) -> Result<Json<Vec<DatasetInfo>>, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemMetrics)?;

    let mut names: std::collections::BTreeSet<String> =
        state.config.datasets.keys().cloned().collect();
    if let Some(dataset_manager) = &state.dataset_manager {
        for meta in dataset_manager.list_datasets().await {
            names.insert(meta.name);
        }
    }

    let mut datasets = Vec::with_capacity(names.len());
    for name in names {
        datasets.push(get_dataset_info(&state, &name).await?);
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

    let dataset_info = get_dataset_info(&state, &dataset_name).await?;
    Ok(Json(dataset_info))
}

/// Create new dataset
///
/// Delegates to `state.dataset_manager` (real, disk-backed: creates
/// `<base_path>/<name>/` plus `metadata.json`) so the dataset genuinely
/// exists afterwards, then best-effort registers a distinct backing store on
/// `state.store` so it is addressable for SPARQL query/update routing.
#[instrument(skip(state))]
pub async fn create_dataset(
    State(state): State<Arc<AppState>>,
    // auth_user: AuthUser,
    Json(request): Json<CreateDatasetRequest>,
) -> Result<Json<DatasetInfo>, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Mutates dataset existence -- must respect the read-only guard before
    // any further validation or the (durable, disk-touching) creation below.
    state.reject_if_read_only(&request.name, "dataset creation")?;

    // Validate dataset name
    validate_dataset_name(&request.name)?;

    // Check if dataset already exists (statically configured or previously
    // created through this API)
    if state.config.datasets.contains_key(&request.name) {
        return Err(FusekiError::conflict(format!(
            "Dataset '{}' already exists",
            request.name
        )));
    }
    if let Some(dataset_manager) = &state.dataset_manager {
        if dataset_manager.get_dataset(&request.name).await.is_ok() {
            return Err(FusekiError::conflict(format!(
                "Dataset '{}' already exists",
                request.name
            )));
        }
    }

    let dataset_manager = state.dataset_manager.as_ref().ok_or_else(|| {
        FusekiError::service_unavailable(
            "Dataset manager not available; cannot durably create datasets",
        )
    })?;
    dataset_manager
        .create_dataset(request.name.clone(), request.description.clone())
        .await?;

    // Create dataset configuration
    let dataset_config = DatasetConfig {
        name: request.name.clone(),
        location: request.location.clone().unwrap_or_default(),
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

    // Best-effort: also register a distinct backing store under `state.store`
    // so the dataset is queryable via `Store::query_dataset`/`update_dataset`
    // going forward. Not fatal on failure — the dataset manager registration
    // above is the durable source of truth for "does this dataset exist".
    if let Err(e) = state
        .store
        .create_dataset(&request.name, dataset_config.clone())
    {
        warn!(
            "Dataset '{}' was created in the dataset manager but could not be \
             registered as a live backing store: {}",
            request.name, e
        );
    }

    // Generate dataset info
    let dataset_info = get_dataset_info(&state, &request.name).await?;

    info!("Created dataset: {}", request.name);
    Ok(Json(dataset_info))
}

/// Delete dataset
///
/// Real deletion for datasets created through this API: removes the
/// `dataset_manager`-tracked directory on disk and best-effort unregisters
/// the backing store. Datasets that exist only in the static config file
/// (never created through this API) have no manager-tracked storage this
/// process can remove, so that case fails loudly instead of reporting a
/// fake success.
#[instrument(skip(state))]
pub async fn delete_dataset(
    State(state): State<Arc<AppState>>,
    Path(dataset_name): Path<String>,
    // auth_user: AuthUser,
) -> Result<StatusCode, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Mutates dataset existence -- guard before touching the dataset manager
    // or the live backing store.
    state.reject_if_read_only(&dataset_name, "dataset deletion")?;

    if let Some(dataset_manager) = &state.dataset_manager {
        if dataset_manager.get_dataset(&dataset_name).await.is_ok() {
            dataset_manager.delete_dataset(&dataset_name).await?;
            // Best-effort: also drop the live backing store registration, if any.
            let _ = state.store.remove_dataset(&dataset_name);
            info!("Deleted dataset: {}", dataset_name);
            return Ok(StatusCode::NO_CONTENT);
        }
    }

    if state.config.datasets.contains_key(&dataset_name) {
        return Err(FusekiError::server_error(format!(
            "Dataset '{dataset_name}' is declared in the static server configuration file \
             and was never created through this API, so this process has no \
             manager-tracked storage to delete for it. Remove it from the \
             configuration file and restart the server instead."
        )));
    }

    Err(FusekiError::not_found(format!(
        "Dataset '{dataset_name}' not found"
    )))
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

/// Collect the server-wide runtime statistics (uptime, request count,
/// memory/CPU usage, active connections) that back both `GET /$/server`'s
/// sibling admin endpoint and the `"runtime"` object nested in the merged
/// `GET /$/stats` response (see `server::functions::stats_server_handler`).
///
/// Extracted from [`server_stats`] so the collection logic lives in exactly
/// one place — `/$/stats` reuses it rather than re-deriving the same fields
/// a second time (see commit 7b32c39f's removal of the duplicate `/$/stats`
/// route, which silently dropped these fields from that endpoint's shape
/// until this merge restored them additively).
pub async fn collect_runtime_stats(state: &AppState) -> ServerStats {
    if let Some(metrics_service) = &state.metrics_service {
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
    }
}

/// Get server statistics
#[instrument(skip(state))]
pub async fn server_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ServerStats>, FusekiError> {
    Ok(Json(collect_runtime_stats(&state).await))
}

/// Returns `Ok(())` if `name` is a known dataset (statically configured or
/// registered via `state.dataset_manager`), otherwise a 404.
async fn ensure_dataset_known(state: &AppState, name: &str) -> FusekiResult<()> {
    if state.config.datasets.contains_key(name) {
        return Ok(());
    }
    if let Some(dataset_manager) = &state.dataset_manager {
        if dataset_manager.get_dataset(name).await.is_ok() {
            return Ok(());
        }
    }
    Err(FusekiError::not_found(format!(
        "Dataset '{name}' not found"
    )))
}

/// Compact dataset
///
/// The `Store` abstraction exposes each dataset only as a `dyn Store` trait
/// object with no compaction primitive (see `oxirs_core::rdf_store::Store`),
/// so there is nothing to physically reclaim at this layer. Rather than
/// fabricate a "space saved" figure, this reports the real, measured
/// triple/byte size and is honest that no physical compaction ran.
#[instrument(skip(state))]
pub async fn compact_dataset(
    State(state): State<Arc<AppState>>,
    Path(dataset_name): Path<String>,
    Query(params): Query<CompactParams>,
    // auth_user: AuthUser,
) -> Result<Json<serde_json::Value>, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    // Compaction is a store-mutating maintenance operation (even though the
    // current backend has nothing to physically reclaim yet, see the
    // function doc comment) -- guard it the same as any other write.
    state.reject_if_read_only(&dataset_name, "dataset compaction")?;

    ensure_dataset_known(&state, &dataset_name).await?;

    let start_time = Instant::now();

    // Perform compaction
    let result =
        compact_dataset_in_store(&state.store, &dataset_name, params.force.unwrap_or(false))
            .await?;

    let execution_time = start_time.elapsed();

    info!(
        "Compaction requested for dataset '{}' in {}ms: {}",
        dataset_name,
        execution_time.as_millis(),
        result.message
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "dataset": dataset_name,
        "execution_time_ms": execution_time.as_millis(),
        "size_before_bytes": result.size_before,
        "size_after_bytes": result.size_after,
        "space_saved_bytes": result.size_before.saturating_sub(result.size_after),
        "message": result.message,
    })))
}

/// Backup dataset
///
/// Serializes the dataset's actual quads (resolved from the live `Store`,
/// falling back to the default backing store when no distinct named store
/// was registered for `dataset_name`) in the requested RDF format, instead
/// of returning fixed placeholder triples.
///
/// Classification (read_only guard NOT applied): this handler only *reads*
/// the dataset and serializes a snapshot to the HTTP response body -- it
/// never writes to the store. Backup *creation* is intentionally exempt
/// from the read_only write guard (unlike restore-from-backup, which would
/// mutate the store and must be guarded); see `create_dataset`,
/// `delete_dataset`, and `compact_dataset` above for the mutating siblings
/// that are guarded.
#[instrument(skip(state))]
pub async fn backup_dataset(
    State(state): State<Arc<AppState>>,
    Path(dataset_name): Path<String>,
    Query(params): Query<BackupParams>,
    // auth_user: AuthUser,
) -> Result<Response, FusekiError> {
    // Check permissions
    // check_admin_permission(&auth_user, &Permission::SystemConfig)?;

    ensure_dataset_known(&state, &dataset_name).await?;

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
        "Created backup for dataset '{}' ({} bytes) in {}ms",
        dataset_name,
        backup_data.len(),
        execution_time.as_millis()
    );

    // Determine content type and filename
    let (content_type, mut filename) = match format {
        "turtle" => ("text/turtle", format!("{dataset_name}_backup.ttl")),
        "ntriples" => ("application/n-triples", format!("{dataset_name}_backup.nt")),
        "rdfxml" => ("application/rdf+xml", format!("{dataset_name}_backup.rdf")),
        "nquads" => ("application/n-quads", format!("{dataset_name}_backup.nq")),
        "trig" => ("application/trig", format!("{dataset_name}_backup.trig")),
        _ => ("text/turtle", format!("{dataset_name}_backup.ttl")),
    };
    if compress {
        filename.push_str(".gz");
    }
    let content_type = if compress {
        "application/gzip".to_string()
    } else {
        content_type.to_string()
    };

    let headers = [
        ("content-type", content_type),
        (
            "content-disposition",
            format!("attachment; filename=\"{filename}\""),
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

/// Get dataset information, merging the statically configured registry
/// (`state.config.datasets`, name/location/read_only/services) with the
/// dynamic registry (`state.dataset_manager`, description/created_at/
/// modified_at for datasets created through this API) and real triple/byte
/// counts read live from `state.store`.
async fn get_dataset_info(state: &AppState, name: &str) -> FusekiResult<DatasetInfo> {
    let config = state.config.datasets.get(name);
    let dynamic_meta = match &state.dataset_manager {
        Some(dataset_manager) => dataset_manager.get_dataset(name).await.ok(),
        None => None,
    };

    if config.is_none() && dynamic_meta.is_none() {
        return Err(FusekiError::not_found(format!(
            "Dataset '{name}' not found"
        )));
    }

    // Get real dataset statistics from the live store (never fabricated).
    let stats = get_dataset_stats_from_store(&state.store, name).await?;

    let services = config
        .map(|c| {
            c.services
                .iter()
                .map(|service| ServiceInfo {
                    name: service.name.clone(),
                    endpoint: service.endpoint.clone(),
                    service_type: format!("{:?}", service.service_type),
                    description: format!("{:?} service", service.service_type),
                })
                .collect()
        })
        .unwrap_or_default();

    let location = config
        .map(|c| c.location.clone())
        .filter(|l| !l.is_empty())
        .or_else(|| {
            dynamic_meta
                .as_ref()
                .map(|_| format!("./data/datasets/{name}"))
        })
        .unwrap_or_default();
    let read_only = config.map(|c| c.read_only).unwrap_or(false);
    let description = dynamic_meta.as_ref().and_then(|m| m.description.clone());

    let (created_at, last_modified) = match &dynamic_meta {
        Some(meta) => (
            chrono::DateTime::<chrono::Utc>::from(meta.created_at).to_rfc3339(),
            chrono::DateTime::<chrono::Utc>::from(meta.modified_at).to_rfc3339(),
        ),
        // No dynamic-registry record (statically configured dataset that was
        // never created through this API): we genuinely don't know when it
        // was created, so report "now" rather than fabricating a past date.
        None => {
            let now = chrono::Utc::now().to_rfc3339();
            (now.clone(), now)
        }
    };

    Ok(DatasetInfo {
        name: name.to_string(),
        location,
        read_only,
        description,
        created_at,
        last_modified,
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

// Store-backed operations. `resolve_dataset_store` implements the two-tier
// lookup shared by stats/compact/backup: prefer a distinct named store
// registered via `Store::create_dataset`, and fall back to the always-present
// default backing store when no such distinct registration exists. This
// matches the common single-dataset Fuseki deployment shape, where the
// "dataset name" in the config file has no separate `Store::datasets` entry
// and all data actually lives in the default store.

struct DatasetStats {
    triple_count: u64,
    size_bytes: u64,
}

struct CompactionResult {
    size_before: u64,
    size_after: u64,
    message: String,
}

/// Resolve `name` to a live backing store, real triples included.
fn resolve_dataset_store(
    store: &Store,
    name: &str,
) -> FusekiResult<Arc<std::sync::RwLock<dyn CoreStore>>> {
    match store.get_dataset(Some(name)) {
        Ok(dataset) => Ok(dataset),
        Err(_) => store.get_dataset(None),
    }
}

/// Real triple count and a real (serialized-size) byte estimate for
/// `name`, read live from the store — never a fixed placeholder.
async fn get_dataset_stats_from_store(store: &Store, name: &str) -> FusekiResult<DatasetStats> {
    let dataset = resolve_dataset_store(store, name)?;
    let quads = {
        let guard = dataset
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store read lock: {e}")))?;
        guard
            .quads()
            .map_err(|e| FusekiError::store(format!("Failed to enumerate quads: {e}")))?
    };

    let triple_count = quads.len() as u64;

    // Byte size is estimated as the N-Quads-serialized length: a real,
    // reproducible measurement of the dataset's content rather than a
    // fabricated constant.
    let serializer = Serializer::new(CoreRdfFormat::NQuads);
    let mut size_bytes = 0u64;
    for quad in &quads {
        if let Ok(line) = serializer.serialize_quad_to_nquads(quad) {
            size_bytes += line.len() as u64 + 1; // + newline
        }
    }

    Ok(DatasetStats {
        triple_count,
        size_bytes,
    })
}

/// Report the real, measured dataset size honestly rather than fabricate a
/// "space saved" figure — see the `compact_dataset` handler docs for why no
/// physical compaction is performed at this layer.
async fn compact_dataset_in_store(
    store: &Store,
    name: &str,
    force: bool,
) -> FusekiResult<CompactionResult> {
    debug!(
        "Compaction requested for dataset '{}' (force: {})",
        name, force
    );
    let stats = get_dataset_stats_from_store(store, name).await?;
    Ok(CompactionResult {
        size_before: stats.size_bytes,
        size_after: stats.size_bytes,
        message: format!(
            "No physical compaction was performed: the current storage backend \
             (dyn Store trait object) exposes no compaction primitive at this \
             layer. Measured {} triples, {} bytes.",
            stats.triple_count, stats.size_bytes
        ),
    })
}

/// Serialize the dataset's real content (resolved via
/// [`resolve_dataset_store`]) in the requested RDF format, optionally gzip
/// compressed (Pure-Rust `oxiarc-deflate`).
async fn create_dataset_backup(
    store: &Store,
    name: &str,
    format: &str,
    compress: bool,
    include_metadata: bool,
) -> FusekiResult<Vec<u8>> {
    debug!(
        "Creating backup for dataset '{}' in format '{}' (compress: {}, metadata: {})",
        name, format, compress, include_metadata
    );

    let dataset = resolve_dataset_store(store, name)?;
    let quads = {
        let guard = dataset
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store read lock: {e}")))?;
        guard
            .quads()
            .map_err(|e| FusekiError::store(format!("Failed to enumerate quads: {e}")))?
    };

    let rdf_format = match format {
        "ntriples" => CoreRdfFormat::NTriples,
        "rdfxml" => CoreRdfFormat::RdfXml,
        "nquads" => CoreRdfFormat::NQuads,
        "trig" => CoreRdfFormat::TriG,
        _ => CoreRdfFormat::Turtle,
    };
    let serializer = Serializer::new(rdf_format);

    let body = match rdf_format {
        CoreRdfFormat::NQuads | CoreRdfFormat::TriG => {
            let mut dataset_model = oxirs_core::model::Dataset::new();
            for quad in &quads {
                dataset_model.insert(quad.clone());
            }
            serializer.serialize_dataset(&dataset_model).map_err(|e| {
                FusekiError::response_formatting(format!("Failed to serialize backup: {e}"))
            })?
        }
        _ => {
            let triples: Vec<_> = quads.iter().map(|q| q.to_triple()).collect();
            let graph = oxirs_core::model::Graph::from_triples(triples);
            serializer.serialize_graph(&graph).map_err(|e| {
                FusekiError::response_formatting(format!("Failed to serialize backup: {e}"))
            })?
        }
    };

    let mut body = body.into_bytes();
    // RDF/XML must start with its own XML declaration (inserted by the
    // serializer); prepending any banner -- even an XML comment -- ahead of
    // that declaration produces an unparseable document, so metadata is
    // skipped for that format only. All other formats here (Turtle,
    // N-Triples, N-Quads, TriG) support `#`-prefixed line comments.
    if include_metadata && rdf_format != CoreRdfFormat::RdfXml {
        let header = format!(
            "# OxiRS backup of dataset '{name}' -- {} quad(s), generated {}\n",
            quads.len(),
            chrono::Utc::now().to_rfc3339()
        );
        let mut with_header = header.into_bytes();
        with_header.extend_from_slice(&body);
        body = with_header;
    }

    if compress {
        body = oxiarc_deflate::gzip_compress(&body, 6)
            .map_err(|e| FusekiError::internal(format!("Failed to compress backup: {e}")))?;
    }

    Ok(body)
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
    // Reloading can create/delete datasets via `state.dataset_manager` (see
    // the dataset diff below), so it mutates dataset existence just like
    // `POST /$/datasets/{name}` and `DELETE /$/datasets/{name}` and must
    // respect the same read-only guard. Unlike those endpoints this
    // operation is not scoped to a single path-parameter dataset name, so it
    // is guarded on the resolved "default"/sole-dataset key (see
    // `AppState::is_dataset_read_only`) rather than per-affected-dataset.
    state.reject_if_read_only(
        "default",
        "configuration reload (may create or delete datasets)",
    )?;

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

    // Dataset changes are the one category this handler can actually apply:
    // `state.dataset_manager` is a real, mutable, shared registry (unlike
    // `state.config`, which is an immutable per-request snapshot — see
    // `AppState::config` docs). Everything else below is detected but
    // honestly reported as requiring a restart, since this process has no
    // safe way to mutate the rest of the live `ServerConfig` snapshot that
    // every other handler already holds a copy of.
    let current_datasets: std::collections::HashSet<_> = state.config.datasets.keys().collect();
    let new_datasets: std::collections::HashSet<_> = new_config.datasets.keys().collect();

    if let Some(dataset_manager) = &state.dataset_manager {
        for name in new_datasets.difference(&current_datasets) {
            let name = name.as_str();
            match dataset_manager.get_dataset(name).await {
                Ok(_) => changes.push(format!(
                    "Dataset '{name}' already present in the dataset manager (no-op)"
                )),
                Err(_) => match dataset_manager.create_dataset(name.to_string(), None).await {
                    Ok(_) => changes.push(format!("Applied: created new dataset '{name}'")),
                    Err(e) => changes.push(format!("Failed to create new dataset '{name}': {e}")),
                },
            }
        }

        for name in current_datasets.difference(&new_datasets) {
            let name = name.as_str();
            match dataset_manager.get_dataset(name).await {
                Ok(_) => match dataset_manager.delete_dataset(name).await {
                    Ok(_) => changes.push(format!("Applied: deleted dataset '{name}'")),
                    Err(e) => changes.push(format!("Failed to delete dataset '{name}': {e}")),
                },
                Err(_) => changes.push(format!(
                    "Dataset '{name}' removed from config but was never created through \
                     the dataset manager (statically configured); not deleted -- restart \
                     required to fully remove it"
                )),
            }
        }
    } else {
        for name in new_datasets.difference(&current_datasets) {
            changes.push(format!(
                "New dataset detected: {name} (not applied -- dataset manager unavailable, requires restart)"
            ));
        }
        for name in current_datasets.difference(&new_datasets) {
            changes.push(format!(
                "Dataset removed from config: {name} (not applied -- dataset manager unavailable, requires restart)"
            ));
        }
    }

    // Check for server config changes
    if state.config.server.port != new_config.server.port {
        changes.push(format!(
            "Port changed: {} -> {} (not applied -- requires restart)",
            state.config.server.port, new_config.server.port
        ));
    }

    if state.config.server.host != new_config.server.host {
        changes.push(format!(
            "Host changed: {} -> {} (not applied -- requires restart)",
            state.config.server.host, new_config.server.host
        ));
    }

    // Check security changes
    if state.config.security.authentication.enabled != new_config.security.authentication.enabled {
        changes.push(format!(
            "Authentication: {} -> {} (not applied -- requires restart)",
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
            "Metrics: {} -> {} (not applied -- requires restart)",
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

    let applied_count = changes.iter().filter(|c| c.starts_with("Applied:")).count();
    let message = if changes.is_empty() {
        "Configuration reloaded. No changes detected.".to_string()
    } else {
        format!(
            "Configuration parsed successfully. {} change(s) detected, {} actually applied \
             (dataset add/remove only). The rest require a server restart -- see `changes` \
             for details.",
            changes.len(),
            applied_count
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

    // -----------------------------------------------------------------------
    // Real dataset CRUD / stats / backup / compact (regression coverage for
    // the previously mocked implementations).
    // -----------------------------------------------------------------------

    async fn state_with_dataset_manager() -> Arc<AppState> {
        let store = crate::store::Store::new().expect("in-memory store");
        let config = crate::config::ServerConfig::default();
        let mut state = crate::server::build_minimal_app_state(store, config);

        let base_path =
            std::env::temp_dir().join(format!("oxirs_fuseki_admin_test_{}", uuid::Uuid::new_v4()));
        let dm_config = crate::dataset_management::DatasetConfig {
            base_path,
            enable_versioning: false,
            max_snapshots: 5,
            auto_backup: false,
            backup_interval_secs: 3600,
            max_concurrent_ops: 2,
        };
        let dataset_manager = crate::dataset_management::DatasetManager::new(dm_config)
            .await
            .expect("dataset manager should initialize under a fresh temp dir");
        state.dataset_manager = Some(dataset_manager);

        Arc::new(state)
    }

    /// Regression: `create_dataset` previously never inserted into any
    /// mutable registry, so a created dataset could never be listed or
    /// fetched again. It must now be real and durable via `dataset_manager`.
    #[tokio::test]
    async fn test_create_then_get_dataset_roundtrip() {
        let state = state_with_dataset_manager().await;

        let created = create_dataset(
            State(state.clone()),
            Json(CreateDatasetRequest {
                name: "roundtrip".to_string(),
                location: None,
                read_only: None,
                description: Some("test dataset".to_string()),
            }),
        )
        .await
        .expect("create should succeed");
        assert_eq!(created.name, "roundtrip");

        let fetched = get_dataset(State(state.clone()), Path("roundtrip".to_string()))
            .await
            .expect("get should find the just-created dataset");
        assert_eq!(fetched.name, "roundtrip");

        let listed = list_datasets(State(state))
            .await
            .expect("list should succeed");
        assert!(
            listed.0.iter().any(|d| d.name == "roundtrip"),
            "created dataset must appear in the list, not just at creation time"
        );
    }

    /// Regression: `create_dataset` on a duplicate name must fail rather
    /// than silently overwrite (this was mocked before and never checked
    /// the real registry).
    #[tokio::test]
    async fn test_create_dataset_duplicate_rejected() {
        let state = state_with_dataset_manager().await;

        let request = || CreateDatasetRequest {
            name: "dup".to_string(),
            location: None,
            read_only: None,
            description: None,
        };
        let first = create_dataset(State(state.clone()), Json(request()))
            .await
            .expect("first create should succeed");
        assert_eq!(first.name, "dup");
        let second = create_dataset(State(state), Json(request())).await;
        assert!(
            second.is_err(),
            "duplicate dataset creation must be rejected"
        );
    }

    /// Regression: `delete_dataset` previously slept and returned `Ok(())`
    /// unconditionally without touching any storage. It must now actually
    /// remove the dataset from the durable registry.
    #[tokio::test]
    async fn test_delete_dataset_actually_removes_it() {
        let state = state_with_dataset_manager().await;

        let created = create_dataset(
            State(state.clone()),
            Json(CreateDatasetRequest {
                name: "to_delete".to_string(),
                location: None,
                read_only: None,
                description: None,
            }),
        )
        .await
        .expect("create should succeed");
        assert_eq!(created.name, "to_delete");

        delete_dataset(State(state.clone()), Path("to_delete".to_string()))
            .await
            .expect("delete should succeed for a manager-tracked dataset");

        let after = get_dataset(State(state), Path("to_delete".to_string())).await;
        assert!(
            after.is_err(),
            "dataset must actually be gone after DELETE, not just report success"
        );
    }

    /// A dataset that exists only in the static config file (never created
    /// through this API) has no manager-tracked storage to delete; DELETE
    /// must fail loudly rather than report a fabricated success.
    #[tokio::test]
    async fn test_delete_dataset_config_only_fails_honestly() {
        let store = crate::store::Store::new().expect("in-memory store");
        let mut config = crate::config::ServerConfig::default();
        config.datasets.insert(
            "static-ds".to_string(),
            crate::config::DatasetConfig {
                name: "static-ds".to_string(),
                location: String::new(),
                read_only: false,
                text_index: None,
                shacl_shapes: vec![],
                services: vec![],
                access_control: None,
                backup: None,
            },
        );
        let state = Arc::new(crate::server::build_minimal_app_state(store, config));

        let result = delete_dataset(State(state), Path("static-ds".to_string())).await;
        assert!(
            result.is_err(),
            "deleting a config-only dataset must fail loudly, not fake success"
        );
    }

    /// Regression: dataset stats/backup previously returned fixed constants
    /// (`triple_count: 1000`, `size_bytes: 50000`) for every dataset. They
    /// must now reflect the real store content.
    #[tokio::test]
    async fn test_dataset_stats_and_backup_reflect_real_store_content() {
        let state = state_with_dataset_manager().await;
        let created = create_dataset(
            State(state.clone()),
            Json(CreateDatasetRequest {
                name: "withdata".to_string(),
                location: None,
                read_only: None,
                description: None,
            }),
        )
        .await
        .expect("create should succeed");
        assert_eq!(created.name, "withdata");

        state
            .store
            .update_dataset(
                "INSERT DATA { <http://example.org/s> <http://example.org/p> \"real-value\" . }",
                Some("withdata"),
            )
            .expect("insert should succeed against the newly created backing store");

        let info = get_dataset(State(state.clone()), Path("withdata".to_string()))
            .await
            .expect("get should succeed");
        assert_eq!(
            info.triple_count, 1,
            "must report the real triple count, not the old fake constant 1000"
        );
        assert_ne!(
            info.size_bytes, 50000,
            "must not report the old fake size constant"
        );
        assert!(info.size_bytes > 0);

        let backup = backup_dataset(
            State(state),
            Path("withdata".to_string()),
            Query(BackupParams {
                format: Some("ntriples".to_string()),
                compress: Some(false),
                include_metadata: Some(false),
            }),
        )
        .await
        .expect("backup should succeed");
        let body = axum::body::to_bytes(backup.into_body(), usize::MAX)
            .await
            .expect("body should collect");
        let text = String::from_utf8(body.to_vec()).expect("backup body should be UTF-8");
        assert!(
            text.contains("real-value"),
            "backup must contain the dataset's actual inserted data, got: {text}"
        );
        assert!(
            !text.contains("backup data"),
            "backup must not contain the old hardcoded placeholder triple"
        );
    }

    /// Regression: `compact_dataset` previously returned fixed
    /// `size_before: 100000, size_after: 75000` for every call regardless of
    /// the dataset's actual state. It must now report real, measured values.
    #[tokio::test]
    async fn test_compact_dataset_reports_real_measurements_not_fake_constants() {
        let state = state_with_dataset_manager().await;
        let created = create_dataset(
            State(state.clone()),
            Json(CreateDatasetRequest {
                name: "compactme".to_string(),
                location: None,
                read_only: None,
                description: None,
            }),
        )
        .await
        .expect("create should succeed");
        assert_eq!(created.name, "compactme");

        let response = compact_dataset(
            State(state),
            Path("compactme".to_string()),
            Query(CompactParams { force: None }),
        )
        .await
        .expect("compact should succeed");

        let size_before = response.0["size_before_bytes"].as_u64().unwrap();
        let size_after = response.0["size_after_bytes"].as_u64().unwrap();
        assert_ne!(
            size_before, 100000,
            "must not report the old fake size_before constant"
        );
        assert_ne!(
            size_after, 75000,
            "must not report the old fake size_after constant"
        );
        // Empty freshly-created dataset: real measured size is 0 on both sides.
        assert_eq!(size_before, 0);
        assert_eq!(size_after, 0);
    }

    /// Regression: `/$/reload` previously computed a change list and threw it
    /// away with a comment saying mutable config access would be needed. The
    /// dataset-add/remove subset must now actually be applied to the shared
    /// `dataset_manager`.
    #[tokio::test]
    async fn test_reload_config_actually_creates_new_dataset() {
        let base_config = crate::config::ServerConfig::default();
        let mut new_config = base_config.clone();
        new_config.datasets.insert(
            "reloaded".to_string(),
            crate::config::DatasetConfig {
                name: "reloaded".to_string(),
                location: String::new(),
                read_only: false,
                text_index: None,
                shacl_shapes: vec![],
                services: vec![],
                access_control: None,
                backup: None,
            },
        );
        let toml_content = toml::to_string(&new_config).expect("serialize new config");
        let path = std::env::temp_dir().join(format!(
            "oxirs_fuseki_reload_test_{}.toml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, toml_content).expect("write temp config file");

        let store = crate::store::Store::new().expect("in-memory store");
        let mut config = base_config;
        config.server.config_file = Some(path.clone());
        let mut state = crate::server::build_minimal_app_state(store, config);

        let dm_base_path =
            std::env::temp_dir().join(format!("oxirs_fuseki_reload_dm_{}", uuid::Uuid::new_v4()));
        let dm_config = crate::dataset_management::DatasetConfig {
            base_path: dm_base_path,
            enable_versioning: false,
            max_snapshots: 5,
            auto_backup: false,
            backup_interval_secs: 3600,
            max_concurrent_ops: 2,
        };
        let dataset_manager = crate::dataset_management::DatasetManager::new(dm_config)
            .await
            .expect("dataset manager should initialize");
        state.dataset_manager = Some(dataset_manager);
        let state = Arc::new(state);

        let response = reload_config(State(state.clone()))
            .await
            .expect("reload should succeed");
        assert!(
            response
                .changes
                .iter()
                .any(|c| c.contains("Applied: created new dataset 'reloaded'")),
            "expected an 'Applied: created' change, got: {:?}",
            response.changes
        );

        let dataset_manager = state
            .dataset_manager
            .as_ref()
            .expect("dataset manager should still be present");
        assert!(
            dataset_manager.get_dataset("reloaded").await.is_ok(),
            "the dataset detected in the reloaded config must actually have been created"
        );

        let _ = std::fs::remove_file(&path);
    }
}
