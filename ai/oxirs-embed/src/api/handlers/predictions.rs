//! Prediction HTTP handlers
//!
//! This module contains handlers for knowledge graph prediction endpoints.

#[cfg(feature = "api-server")]
use super::super::helpers::get_production_model;
#[cfg(feature = "api-server")]
use super::super::{ApiState, PredictionRequest, PredictionResponse, PredictionType};
#[cfg(feature = "api-server")]
use axum::{extract::State, http::StatusCode, response::Json};
#[cfg(feature = "api-server")]
use std::time::Instant;
#[cfg(feature = "api-server")]
use tracing::{debug, error, warn};

/// Predict entities/relations
#[cfg(feature = "api-server")]
pub async fn predict(
    State(state): State<ApiState>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    let start_time = Instant::now();

    debug!("Received prediction request: {:?}", request);

    // Get model (either specified or best production model)
    let model = if let Some(model_id) = request.model_id {
        // Try to get specific model
        let models = state.models.read().await;
        match models.get(&model_id) {
            Some(model) => model.clone(),
            None => {
                warn!(
                    "Specified model {} not found, falling back to production model",
                    model_id
                );
                match get_production_model(&state.registry).await {
                    Ok(model) => model,
                    Err(e) => {
                        error!("Failed to get production model: {}", e);
                        return Err(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                }
            }
        }
    } else {
        // Get best production model
        match get_production_model(&state.registry).await {
            Ok(model) => model,
            Err(e) => {
                error!("Failed to get production model: {}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    };

    // Check if model is trained
    if !model.is_trained() {
        error!("Model is not trained");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Determine prediction type and perform prediction
    let (prediction_type_str, predictions) = match request.prediction_type {
        PredictionType::Objects { subject, predicate } => {
            let k = request.top_k.unwrap_or(10);

            // Check cache first if enabled
            if request.use_cache.unwrap_or(true) {
                let cache_key = format!("predict_objects:{}:{}:{}", subject, predicate, k);
                if let Some(cached_results) = state.cache_manager.get_similarity_cache(&cache_key) {
                    debug!("Cache hit for prediction: {}", cache_key);
                    let response = PredictionResponse {
                        input: request.entities,
                        prediction_type: "objects".to_string(),
                        predictions: cached_results,
                        model_version: "cached".to_string(),
                        from_cache: true,
                        prediction_time_ms: start_time.elapsed().as_millis() as f64,
                    };
                    return Ok(Json(response));
                }
            }

            match model.predict_objects(&subject, &predicate, k) {
                Ok(predictions) => {
                    // Cache results if enabled
                    if request.use_cache.unwrap_or(true) {
                        let cache_key = format!("predict_objects:{}:{}:{}", subject, predicate, k);
                        state
                            .cache_manager
                            .put_similarity_cache(cache_key, predictions.clone());
                    }
                    ("objects", predictions)
                }
                Err(e) => {
                    error!("Failed to predict objects: {}", e);
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
        }
        PredictionType::Subjects { predicate, object } => {
            let k = request.top_k.unwrap_or(10);

            // Check cache first if enabled
            if request.use_cache.unwrap_or(true) {
                let cache_key = format!("predict_subjects:{}:{}:{}", predicate, object, k);
                if let Some(cached_results) = state.cache_manager.get_similarity_cache(&cache_key) {
                    debug!("Cache hit for prediction: {}", cache_key);
                    let response = PredictionResponse {
                        input: request.entities,
                        prediction_type: "subjects".to_string(),
                        predictions: cached_results,
                        model_version: "cached".to_string(),
                        from_cache: true,
                        prediction_time_ms: start_time.elapsed().as_millis() as f64,
                    };
                    return Ok(Json(response));
                }
            }

            match model.predict_subjects(&predicate, &object, k) {
                Ok(predictions) => {
                    // Cache results if enabled
                    if request.use_cache.unwrap_or(true) {
                        let cache_key = format!("predict_subjects:{}:{}:{}", predicate, object, k);
                        state
                            .cache_manager
                            .put_similarity_cache(cache_key, predictions.clone());
                    }
                    ("subjects", predictions)
                }
                Err(e) => {
                    error!("Failed to predict subjects: {}", e);
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
        }
        PredictionType::Relations { subject, object } => {
            let k = request.top_k.unwrap_or(10);

            // Check cache first if enabled
            if request.use_cache.unwrap_or(true) {
                let cache_key = format!("predict_relations:{}:{}:{}", subject, object, k);
                if let Some(cached_results) = state.cache_manager.get_similarity_cache(&cache_key) {
                    debug!("Cache hit for prediction: {}", cache_key);
                    let response = PredictionResponse {
                        input: request.entities,
                        prediction_type: "relations".to_string(),
                        predictions: cached_results,
                        model_version: "cached".to_string(),
                        from_cache: true,
                        prediction_time_ms: start_time.elapsed().as_millis() as f64,
                    };
                    return Ok(Json(response));
                }
            }

            match model.predict_relations(&subject, &object, k) {
                Ok(predictions) => {
                    // Cache results if enabled
                    if request.use_cache.unwrap_or(true) {
                        let cache_key = format!("predict_relations:{}:{}:{}", subject, object, k);
                        state
                            .cache_manager
                            .put_similarity_cache(cache_key, predictions.clone());
                    }
                    ("relations", predictions)
                }
                Err(e) => {
                    error!("Failed to predict relations: {}", e);
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
        }
    };

    let prediction_time_ms = start_time.elapsed().as_millis() as f64;

    debug!(
        "Prediction completed in {:.2}ms, returned {} results",
        prediction_time_ms,
        predictions.len()
    );

    let response = PredictionResponse {
        input: request.entities,
        prediction_type: prediction_type_str.to_string(),
        predictions,
        model_version: model.model_id().to_string(),
        from_cache: false,
        prediction_time_ms,
    };

    Ok(Json(response))
}
