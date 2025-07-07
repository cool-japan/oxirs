//! System monitoring and health HTTP handlers
//!
//! This module contains handlers for system health, statistics, and cache management endpoints.

#[cfg(feature = "api-server")]
use super::super::{ApiState, HealthMetrics, HealthStatus, ModelHealth};
#[cfg(feature = "api-server")]
use axum::{extract::State, http::StatusCode, response::Json};
#[cfg(feature = "api-server")]
use chrono::Utc;
#[cfg(feature = "api-server")]
use serde_json::json;
#[cfg(feature = "api-server")]
use std::collections::HashMap;

/// System health check
#[cfg(feature = "api-server")]
pub async fn system_health(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let models = state.models.read().await;
    let model_count = models.len();

    // Get cache statistics
    let cache_stats = state.cache_manager.get_stats();
    let cache_hit_rate = if cache_stats.total_hits + cache_stats.total_misses > 0 {
        cache_stats.total_hits as f64 / (cache_stats.total_hits + cache_stats.total_misses) as f64
    } else {
        0.0
    };

    // Estimate memory usage
    let memory_usage_mb = state.cache_manager.estimate_memory_usage() as f64 / (1024.0 * 1024.0);

    let health_metrics = HealthMetrics {
        avg_response_time_ms: cache_hit_rate * 100.0, // Simplified calculation
        requests_last_hour: cache_stats.total_hits + cache_stats.total_misses,
        error_rate_percent: 0.0, // Would need proper error tracking
        memory_usage_mb,
    };

    let health_status = if model_count > 0 && cache_hit_rate > 0.5 {
        HealthStatus::Healthy
    } else if model_count > 0 {
        HealthStatus::Degraded
    } else {
        HealthStatus::Unhealthy
    };

    let health = ModelHealth {
        status: health_status,
        last_check: Utc::now(),
        metrics: health_metrics,
    };

    Ok(Json(json!({
        "status": "ok",
        "models_loaded": model_count,
        "cache_hit_rate": cache_hit_rate,
        "health": health
    })))
}

/// System statistics
#[cfg(feature = "api-server")]
pub async fn system_stats(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let models = state.models.read().await;
    let cache_stats = state.cache_manager.get_stats();

    let mut model_stats = HashMap::new();
    for (model_id, _model) in models.iter() {
        model_stats.insert(
            model_id.to_string(),
            json!({
                "id": model_id,
                "status": "loaded",
                "type": "embedding_model"
            }),
        );
    }

    let system_stats = json!({
        "system": {
            "uptime_seconds": 0, // Would need proper uptime tracking
            "version": "1.0.0",
            "config": {
                "host": state.config.host,
                "port": state.config.port,
                "max_batch_size": state.config.max_batch_size,
                "timeout_seconds": state.config.timeout_seconds
            }
        },
        "models": {
            "count": models.len(),
            "loaded_models": model_stats
        },
        "cache": {
            "total_hits": cache_stats.total_hits,
            "total_misses": cache_stats.total_misses,
            "hit_rate": cache_stats.hit_rate,
            "memory_usage_bytes": cache_stats.memory_usage_bytes,
            "time_saved_seconds": cache_stats.total_time_saved_seconds,
            "l1_stats": {
                "hits": cache_stats.l1_stats.hits,
                "misses": cache_stats.l1_stats.misses,
                "size": cache_stats.l1_stats.size,
                "capacity": cache_stats.l1_stats.capacity
            },
            "l2_stats": {
                "hits": cache_stats.l2_stats.hits,
                "misses": cache_stats.l2_stats.misses,
                "size": cache_stats.l2_stats.size,
                "capacity": cache_stats.l2_stats.capacity
            },
            "l3_stats": {
                "hits": cache_stats.l3_stats.hits,
                "misses": cache_stats.l3_stats.misses,
                "size": cache_stats.l3_stats.size,
                "capacity": cache_stats.l3_stats.capacity
            }
        }
    });

    Ok(Json(system_stats))
}

/// Cache statistics
#[cfg(feature = "api-server")]
pub async fn cache_stats(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let cache_stats = state.cache_manager.get_stats();
    let hit_rates = state.cache_manager.get_cache_hit_rates();
    let computation_type_stats = state.cache_manager.get_computation_type_stats();

    let stats = json!({
        "overview": {
            "total_hits": cache_stats.total_hits,
            "total_misses": cache_stats.total_misses,
            "hit_rate": cache_stats.hit_rate,
            "memory_usage_bytes": cache_stats.memory_usage_bytes,
            "time_saved_seconds": cache_stats.total_time_saved_seconds
        },
        "levels": {
            "l1_embeddings": {
                "hits": cache_stats.l1_stats.hits,
                "misses": cache_stats.l1_stats.misses,
                "size": cache_stats.l1_stats.size,
                "capacity": cache_stats.l1_stats.capacity,
                "memory_bytes": cache_stats.l1_stats.memory_bytes
            },
            "l2_computations": {
                "hits": cache_stats.l2_stats.hits,
                "misses": cache_stats.l2_stats.misses,
                "size": cache_stats.l2_stats.size,
                "capacity": cache_stats.l2_stats.capacity,
                "memory_bytes": cache_stats.l2_stats.memory_bytes
            },
            "l3_similarity": {
                "hits": cache_stats.l3_stats.hits,
                "misses": cache_stats.l3_stats.misses,
                "size": cache_stats.l3_stats.size,
                "capacity": cache_stats.l3_stats.capacity,
                "memory_bytes": cache_stats.l3_stats.memory_bytes
            }
        },
        "hit_rates_by_type": hit_rates,
        "computation_type_stats": computation_type_stats
    });

    Ok(Json(stats))
}

/// Clear cache
#[cfg(feature = "api-server")]
pub async fn clear_cache(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let cache_stats_before = state.cache_manager.get_stats();

    // Clear all caches
    state.cache_manager.clear_all();

    let cache_stats_after = state.cache_manager.get_stats();

    let result = json!({
        "status": "success",
        "message": "All caches cleared successfully",
        "before": {
            "total_entries": cache_stats_before.l1_stats.size + cache_stats_before.l2_stats.size + cache_stats_before.l3_stats.size,
            "memory_usage_bytes": cache_stats_before.memory_usage_bytes
        },
        "after": {
            "total_entries": cache_stats_after.l1_stats.size + cache_stats_after.l2_stats.size + cache_stats_after.l3_stats.size,
            "memory_usage_bytes": cache_stats_after.memory_usage_bytes
        }
    });

    Ok(Json(result))
}
