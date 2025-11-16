//! Knowledge graph triple scoring HTTP handlers
//!
//! This module contains handlers for triple scoring API endpoints.

use super::super::helpers::get_production_model_version;
use super::super::{ApiState, TripleScoreRequest, TripleScoreResponse};
#[cfg(feature = "api-server")]
use axum::{extract::State, http::StatusCode, response::Json};

/// Score a triple
#[cfg(feature = "api-server")]
pub async fn score_triple(
    State(state): State<ApiState>,
    Json(request): Json<TripleScoreRequest>,
) -> Result<Json<TripleScoreResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    // Get model version
    let model_version = if let Some(version) = request.model_version {
        match version.parse::<uuid::Uuid>() {
            Ok(uuid) => uuid,
            Err(_) => return Err(StatusCode::BAD_REQUEST),
        }
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

    // Score triple with caching
    let use_cache = request.use_cache.unwrap_or(true);
    let (score, from_cache) = if use_cache {
        // For now, use direct model scoring (can be enhanced with triple scoring cache later)
        match model.score_triple(&request.subject, &request.predicate, &request.object) {
            Ok(score) => (score, false),
            Err(_) => return Err(StatusCode::NOT_FOUND),
        }
    } else {
        match model.score_triple(&request.subject, &request.predicate, &request.object) {
            Ok(score) => (score, false),
            Err(_) => return Err(StatusCode::NOT_FOUND),
        }
    };

    let scoring_time = start_time.elapsed().as_millis() as f64;

    let response = TripleScoreResponse {
        subject: request.subject.clone(),
        predicate: request.predicate.clone(),
        object: request.object.clone(),
        triple: (request.subject, request.predicate, request.object),
        score,
        model_id: request.model_id.unwrap_or(model_version),
        model_version: model_version.to_string(),
        from_cache,
        computation_time_ms: scoring_time,
        scoring_time_ms: scoring_time,
    };

    Ok(Json(response))
}
