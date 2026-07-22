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
        .route("/api/v1/models/{model_id}", get(get_model_info))
        .route("/api/v1/models/{model_id}/health", get(get_model_health))
        .route("/api/v1/models/{model_id}/load", post(load_model))
        .route("/api/v1/models/{model_id}/unload", post(unload_model))
        // System endpoints
        .route("/api/v1/health", get(system_health))
        .route("/api/v1/stats", get(system_stats))
        .route("/api/v1/cache/stats", get(cache_stats))
        .route("/api/v1/cache/clear", post(clear_cache))
        .with_state(state)
}

#[cfg(all(test, feature = "api-server"))]
mod tests {
    use super::create_router;
    use crate::api::config::{ApiConfig, ApiState};
    use crate::{CacheManager, ModelRegistry};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    /// Regression test for the axum 0.8 route-syntax migration.
    ///
    /// Building the router forces axum/matchit to parse every registered
    /// route path. Under axum 0.8, old-style `:param` / `*wildcard`
    /// segments make `Router::route` PANIC at construction time (the
    /// panic message tells you to use `{param}` instead). This test
    /// invokes the crate's real router-builder function
    /// (`create_router`) with a minimal-but-real `ApiState` and asserts
    /// that construction does not panic, which would have caught the
    /// `/api/v1/models/:model_id...` regression immediately.
    #[tokio::test]
    async fn create_router_does_not_panic_on_route_syntax() {
        let storage_path =
            std::env::temp_dir().join(format!("oxirs-embed-routes-test-{}", uuid::Uuid::new_v4()));

        let state = ApiState {
            registry: Arc::new(ModelRegistry::new(storage_path)),
            cache_manager: Arc::new(CacheManager::new(Default::default())),
            models: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(crate::api::config::ApiMetrics::new()),
            config: ApiConfig::default(),
        };

        // Must not panic: proves every route path uses valid axum 0.8
        // syntax (matchit 0.8's `{param}` / `{*rest}` forms).
        let _router = create_router(state);
    }
}
