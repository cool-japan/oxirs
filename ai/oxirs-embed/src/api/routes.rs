//! API route definitions
//!
//! This module defines all HTTP routes and their corresponding handlers.

#[cfg(feature = "api-server")]
use super::{
    handlers::{
        embeddings::{embed_batch, embed_single},
        models::{get_model_health, get_model_info, list_models, load_model, unload_model},
        predictions::predict,
        scoring::score_triple,
        system::{cache_stats, clear_cache, system_health, system_stats},
    },
    ApiState,
};
#[cfg(feature = "api-server")]
use axum::{
    routing::{get, post},
    Router,
};

/// Create the API router with all endpoints
#[cfg(feature = "api-server")]
pub fn create_router(state: ApiState) -> Router {
    Router::new()
        // Embedding endpoints
        .route("/api/v1/embed", post(embed_single))
        .route("/api/v1/embed/batch", post(embed_batch))
        // Scoring endpoints
        .route("/api/v1/score", post(score_triple))
        // Prediction endpoints
        .route("/api/v1/predict", post(predict))
        // Model management endpoints
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/models/:model_id", get(get_model_info))
        .route("/api/v1/models/:model_id/health", get(get_model_health))
        .route("/api/v1/models/:model_id/load", post(load_model))
        .route("/api/v1/models/:model_id/unload", post(unload_model))
        // System endpoints
        .route("/api/v1/health", get(system_health))
        .route("/api/v1/stats", get(system_stats))
        .route("/api/v1/cache/stats", get(cache_stats))
        .route("/api/v1/cache/clear", post(clear_cache))
        .with_state(state)
}
