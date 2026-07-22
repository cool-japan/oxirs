//! Model management HTTP handlers
//!
//! This module contains handlers for model lifecycle management endpoints.

#[cfg(feature = "api-server")]
use super::super::{ApiState, HealthMetrics, HealthStatus, ModelHealth, ModelInfoResponse};
#[cfg(feature = "api-server")]
use crate::{ModelStats, TrainingStats};
#[cfg(feature = "api-server")]
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
#[cfg(feature = "api-server")]
use chrono::Utc;
#[cfg(feature = "api-server")]
use serde_json::json;
#[cfg(feature = "api-server")]
use std::collections::HashMap;
#[cfg(feature = "api-server")]
use std::sync::Arc;
#[cfg(feature = "api-server")]
use tracing::{debug, error, info};
use uuid::Uuid;

/// List available models
#[cfg(feature = "api-server")]
pub async fn list_models(
    State(state): State<ApiState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    debug!("Listing available models");

    // Get all models from registry
    let models = state.registry.list_models().await;
    let loaded_models = state.models.read().await;

    let mut model_list = Vec::new();
    for model_metadata in models {
        let is_loaded = loaded_models.contains_key(&model_metadata.model_id);

        // Get production version info if available
        let production_version = if let Some(prod_version_id) = model_metadata.production_version {
            match state.registry.get_version(prod_version_id).await {
                Ok(version) => Some(json!({
                    "version_id": version.version_id,
                    "version_number": version.version_number,
                    "created_at": version.created_at,
                    "is_production": version.is_production
                })),
                Err(_) => None,
            }
        } else {
            None
        };

        let model_info = json!({
            "model_id": model_metadata.model_id,
            "name": model_metadata.name,
            "model_type": model_metadata.model_type,
            "created_at": model_metadata.created_at,
            "updated_at": model_metadata.updated_at,
            "owner": model_metadata.owner,
            "description": model_metadata.description,
            "is_loaded": is_loaded,
            "version_count": model_metadata.versions.len(),
            "production_version": production_version,
            "staging_version": model_metadata.staging_version
        });

        model_list.push(model_info);
    }

    // Apply filters if requested
    let detailed = params.get("detailed").map(|v| v == "true").unwrap_or(false);

    let response = if detailed {
        json!({
            "models": model_list,
            "total_count": model_list.len(),
            "loaded_count": loaded_models.len()
        })
    } else {
        json!({
            "models": model_list.iter().map(|m| json!({
                "model_id": m["model_id"],
                "name": m["name"],
                "model_type": m["model_type"],
                "is_loaded": m["is_loaded"]
            })).collect::<Vec<_>>(),
            "total_count": model_list.len()
        })
    };

    Ok(Json(response))
}

/// Get model information
#[cfg(feature = "api-server")]
pub async fn get_model_info(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<ModelInfoResponse>, StatusCode> {
    debug!("Getting model information for: {}", model_id);
    let request_start = std::time::Instant::now();

    // Get model metadata from registry
    let model_metadata = match state.registry.get_model(model_id).await {
        Ok(metadata) => metadata,
        Err(e) => {
            error!("Model not found in registry: {}", e);
            state.metrics.record(request_start.elapsed(), true);
            return Err(StatusCode::NOT_FOUND);
        }
    };

    // Check if model is loaded
    let models = state.models.read().await;
    let is_loaded = models.contains_key(&model_id);

    // Get model stats
    let stats = if let Some(model) = models.get(&model_id) {
        model.get_stats()
    } else {
        ModelStats {
            num_entities: 0,
            num_relations: 0,
            num_triples: 0,
            dimensions: 0,
            is_trained: false,
            model_type: model_metadata.model_type.clone(),
            creation_time: model_metadata.created_at,
            last_training_time: None,
        }
    };

    // Get health status
    let cache_stats = state.cache_manager.get_stats();
    let memory_usage_mb = state.cache_manager.estimate_memory_usage() as f64 / (1024.0 * 1024.0);

    let health_metrics = HealthMetrics {
        // Real, live aggregates from the request-metrics tracker rather than
        // hard-coded placeholders. Before any traffic these are honestly 0.0.
        avg_response_time_ms: state.metrics.avg_response_time_ms(),
        requests_last_hour: cache_stats.total_hits + cache_stats.total_misses,
        error_rate_percent: state.metrics.error_rate_percent(),
        memory_usage_mb,
    };

    let health_status = if is_loaded && stats.is_trained {
        HealthStatus::Healthy
    } else if is_loaded {
        HealthStatus::Degraded
    } else {
        HealthStatus::Unhealthy
    };

    let health = ModelHealth {
        status: health_status,
        last_check: Utc::now(),
        metrics: health_metrics,
    };

    // Get capabilities
    let capabilities = vec![
        "entity_embedding".to_string(),
        "relation_embedding".to_string(),
        "triple_scoring".to_string(),
        "object_prediction".to_string(),
        "subject_prediction".to_string(),
        "relation_prediction".to_string(),
    ];

    // Derive real training statistics from the registry's production (or most
    // recent) version metrics. We never fabricate values: if the recorded
    // version carries no training metrics, `last_training` is reported as None
    // rather than a hard-coded placeholder.
    let last_training = training_stats_from_registry(&state, &model_metadata).await;

    let response = ModelInfoResponse {
        stats,
        health,
        capabilities,
        last_training,
    };

    state.metrics.record(request_start.elapsed(), false);
    Ok(Json(response))
}

/// Build [`TrainingStats`] from real registry version metrics, or `None` when
/// no training metrics were recorded for the model.
///
/// This reads whatever a training/registration flow persisted onto the model's
/// production version (falling back to the latest registered version). Keys are
/// matched permissively (`final_loss`/`loss`, `epochs`/`epochs_completed`, ...)
/// and only the fields actually present are populated; absent fields fall back
/// to neutral, clearly-non-fabricated defaults.
#[cfg(feature = "api-server")]
async fn training_stats_from_registry(
    state: &ApiState,
    model_metadata: &crate::model_registry::ModelMetadata,
) -> Option<TrainingStats> {
    // Prefer the production version; otherwise use the most recently registered.
    let version_id = model_metadata
        .production_version
        .or_else(|| model_metadata.versions.last().copied())?;

    let version = state.registry.get_version(version_id).await.ok()?;
    let metrics = &version.metrics;
    if metrics.is_empty() {
        return None;
    }

    let get = |keys: &[&str]| -> Option<f64> { keys.iter().find_map(|k| metrics.get(*k).copied()) };

    Some(TrainingStats {
        epochs_completed: get(&["epochs_completed", "epochs"])
            .map(|v| v as usize)
            .unwrap_or(0),
        final_loss: get(&["final_loss", "loss"]).unwrap_or(0.0),
        training_time_seconds: get(&["training_time_seconds", "training_time"]).unwrap_or(0.0),
        convergence_achieved: get(&["convergence_achieved", "converged"])
            .map(|v| v != 0.0)
            .unwrap_or(false),
        loss_history: Vec::new(),
    })
}

/// Get model health status
#[cfg(feature = "api-server")]
pub async fn get_model_health(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    debug!("Getting health status for model: {}", model_id);

    // Check if model exists in registry
    if state.registry.get_model(model_id).await.is_err() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Check if model is loaded
    let models = state.models.read().await;
    let is_loaded = models.contains_key(&model_id);

    let health_status = if let Some(model) = models.get(&model_id) {
        if model.is_trained() {
            "healthy"
        } else {
            "degraded"
        }
    } else {
        "unhealthy"
    };

    let response = json!({
        "model_id": model_id,
        "status": health_status,
        "is_loaded": is_loaded,
        "last_check": Utc::now(),
        "details": {
            "loaded": is_loaded,
            "trained": models.get(&model_id).map(|m| m.is_trained()).unwrap_or(false),
            "memory_usage_mb": state.cache_manager.estimate_memory_usage() as f64 / (1024.0 * 1024.0)
        }
    });

    Ok(Json(response))
}

/// Load a model
#[cfg(feature = "api-server")]
pub async fn load_model(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    info!("Loading model: {}", model_id);

    // Check if model exists in registry
    let model_metadata = match state.registry.get_model(model_id).await {
        Ok(metadata) => metadata,
        Err(e) => {
            error!("Model not found in registry: {}", e);
            return Err(StatusCode::NOT_FOUND);
        }
    };

    // Check if model is already loaded
    {
        let models = state.models.read().await;
        if models.contains_key(&model_id) {
            let response = json!({
                "status": "already_loaded",
                "model_id": model_id,
                "message": "Model is already loaded"
            });
            return Ok(Json(response));
        }
    }

    // Attempt to load persisted weights from the registry's storage area and
    // insert the reconstructed model into the loaded-models map. If no
    // persisted artifact exists (or the model type is not reconstructable), we
    // MUST fail loud (per the project's fail-loud contract) rather than return
    // a fabricated 200 "load_initiated" success that leaves the model absent.
    match load_model_from_storage(&state, &model_metadata).await {
        Ok(model) => {
            let mut models = state.models.write().await;
            models.insert(model_id, model);
            let response = json!({
                "status": "loaded",
                "model_id": model_id,
                "model_name": model_metadata.name,
                "model_type": model_metadata.model_type,
                "message": "Model loaded successfully"
            });
            Ok(Json(response))
        }
        Err(e) => {
            error!("Failed to load model {}: {}", model_id, e);
            // 501 when the artifact/type is not supported for reconstruction,
            // otherwise surface an internal error. Either way, never 200.
            Err(StatusCode::NOT_IMPLEMENTED)
        }
    }
}

/// Reconstruct a trained model from the registry's on-disk storage area.
///
/// Layout: `<registry storage_path>/<model_id>/model.bin`, matching the format
/// written by each model's `EmbeddingModel::save` implementation. Returns an
/// error (never a silently-empty success) if the artifact is missing or the
/// model type cannot be reconstructed.
#[cfg(feature = "api-server")]
async fn load_model_from_storage(
    state: &ApiState,
    model_metadata: &crate::model_registry::ModelMetadata,
) -> anyhow::Result<Arc<dyn crate::EmbeddingModel + Send + Sync>> {
    use crate::{
        ComplEx, DistMult, EmbeddingModel, GNNConfig, GNNEmbedding, HoLE, HoLEConfig, ModelConfig,
        RotatE, TransE,
    };

    let model_file = state
        .registry
        .storage_path()
        .join(model_metadata.model_id.to_string())
        .join("model.bin");

    if !model_file.exists() {
        return Err(anyhow::anyhow!(
            "no persisted weights found for model {} at {}",
            model_metadata.model_id,
            model_file.display()
        ));
    }
    let model_file = model_file.to_string_lossy().to_string();

    let mut model: Box<dyn EmbeddingModel + Send + Sync> = match model_metadata.model_type.as_str()
    {
        "TransE" => Box::new(TransE::new(ModelConfig::default())),
        "DistMult" => Box::new(DistMult::new(ModelConfig::default())),
        "ComplEx" => Box::new(ComplEx::new(ModelConfig::default())),
        "RotatE" => Box::new(RotatE::new(ModelConfig::default())),
        "HoLE" | "HolE" => Box::new(HoLE::new(HoLEConfig::default())),
        "GNN" | "GNNEmbedding" => Box::new(GNNEmbedding::new(GNNConfig::default())),
        other => {
            return Err(anyhow::anyhow!(
                "model type '{}' cannot be reconstructed for loading",
                other
            ))
        }
    };

    model.load(&model_file)?;
    Ok(Arc::from(model))
}

/// Unload a model
#[cfg(feature = "api-server")]
pub async fn unload_model(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    info!("Unloading model: {}", model_id);

    // Check if model exists and is loaded
    let was_loaded = {
        let mut models = state.models.write().await;
        models.remove(&model_id).is_some()
    };

    if !was_loaded {
        let response = json!({
            "status": "not_loaded",
            "model_id": model_id,
            "message": "Model was not loaded"
        });
        return Ok(Json(response));
    }

    // Clear any cached data for this model
    state
        .cache_manager
        .clear_computation_cache(&model_id.to_string());

    let response = json!({
        "status": "unloaded",
        "model_id": model_id,
        "message": "Model unloaded successfully"
    });

    Ok(Json(response))
}
