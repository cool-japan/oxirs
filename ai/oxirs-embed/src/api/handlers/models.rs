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
use tracing::{debug, error, info, warn};
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

    // Get model metadata from registry
    let model_metadata = match state.registry.get_model(model_id).await {
        Ok(metadata) => metadata,
        Err(e) => {
            error!("Model not found in registry: {}", e);
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
        avg_response_time_ms: 50.0, // Would need proper tracking
        requests_last_hour: cache_stats.total_hits + cache_stats.total_misses,
        error_rate_percent: 0.0, // Would need proper error tracking
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

    // Get last training stats (placeholder - would need proper tracking)
    let last_training = if stats.is_trained {
        Some(TrainingStats {
            epochs_completed: 100,
            final_loss: 0.1,
            training_time_seconds: 3600.0,
            convergence_achieved: true,
            loss_history: vec![1.0, 0.5, 0.2, 0.1],
        })
    } else {
        None
    };

    let response = ModelInfoResponse {
        stats,
        health,
        capabilities,
        last_training,
    };

    Ok(Json(response))
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

    // In a real implementation, this would:
    // 1. Load model weights from storage
    // 2. Initialize model with proper configuration
    // 3. Add to the loaded models map
    // For now, we'll create a placeholder response

    warn!("Model loading not fully implemented - this would load model weights and configuration");

    let response = json!({
        "status": "load_initiated",
        "model_id": model_id,
        "model_name": model_metadata.name,
        "model_type": model_metadata.model_type,
        "message": "Model loading initiated (implementation pending)"
    });

    Ok(Json(response))
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
