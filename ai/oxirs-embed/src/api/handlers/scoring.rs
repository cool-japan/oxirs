//! Knowledge graph triple scoring HTTP handlers
//!
//! This module contains handlers for triple scoring API endpoints.

use super::super::helpers::get_production_model_version;
use super::super::{ApiState, TripleScoreRequest, TripleScoreResponse};
#[cfg(feature = "api-server")]
use crate::CachedEmbeddingModel;
#[cfg(feature = "api-server")]
use axum::{extract::State, http::StatusCode, response::Json};
use std::sync::Arc;

/// Score a triple
#[cfg(feature = "api-server")]
pub async fn score_triple(
    State(state): State<ApiState>,
    Json(request): Json<TripleScoreRequest>,
) -> Result<Json<TripleScoreResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    // Get model version
    let model_version = if let Some(version) = request.model_version {
        version
    } else {
        match get_production_model_version(&state).await {
            Ok(version) => version,
            Err(_) => return Err(StatusCode::NOT_FOUND),
        }
    };

    // Get model
    let models = state.models.read().await;
    let model = match models.get(&model_version) {
        Some(model) => model,
        None => return Err(StatusCode::NOT_FOUND),
    };

    let cached_model =
        CachedEmbeddingModel::new(Box::new(model.as_ref()), Arc::clone(&state.cache_manager));

    // Score triple
    let use_cache = request.use_cache.unwrap_or(true);
    let (score, from_cache) = if use_cache {
        match cached_model.score_triple_cached(
            &request.subject,
            &request.predicate,
            &request.object,
        ) {
            Ok(score) => (score, true), // Simplified - would need to check if actually from cache
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    } else {
        match model.score_triple(&request.subject, &request.predicate, &request.object) {
            Ok(score) => (score, false),
            Err(_) => return Err(StatusCode::NOT_FOUND),
        }
    };

    let scoring_time = start_time.elapsed().as_millis() as f64;

    let response = TripleScoreResponse {
        triple: (request.subject, request.predicate, request.object),
        score,
        model_version,
        from_cache,
        scoring_time_ms: scoring_time,
    };

    Ok(Json(response))
}
