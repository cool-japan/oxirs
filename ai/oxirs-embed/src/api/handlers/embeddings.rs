//! Embedding generation HTTP handlers
//!
//! This module contains handlers for all embedding-related API endpoints.

use super::super::helpers::get_production_model_version;
use super::super::{
    ApiState, BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
};
#[cfg(feature = "api-server")]
use crate::{CachedEmbeddingModel, EmbeddingModel};
#[cfg(feature = "api-server")]
use axum::{extract::State, http::StatusCode, response::Json};
use std::sync::Arc;

/// Generate embedding for a single entity
#[cfg(feature = "api-server")]
pub async fn embed_single(
    State(state): State<ApiState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    // Get model version
    let model_version = if let Some(version) = request.model_version {
        version
    } else {
        // Use production version
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

    // Create cached model wrapper
    let cached_model = CachedEmbeddingModel::new(
        Box::new(model.as_ref()), // This is not ideal - in real implementation would clone or use Arc
        Arc::clone(&state.cache_manager),
    );

    // Generate embedding
    let use_cache = request.use_cache.unwrap_or(true);
    let (embedding, from_cache) = if use_cache {
        match cached_model.get_entity_embedding_cached(&request.entity) {
            Ok(emb) => (
                emb,
                state.cache_manager.get_embedding(&request.entity).is_some(),
            ),
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    } else {
        match model.get_entity_embedding(&request.entity) {
            Ok(emb) => (emb, false),
            Err(_) => return Err(StatusCode::NOT_FOUND),
        }
    };

    let generation_time = start_time.elapsed().as_millis() as f64;

    let response = EmbeddingResponse {
        entity: request.entity,
        embedding: embedding.values,
        dimensions: embedding.dimensions,
        model_version,
        from_cache,
        generation_time_ms: generation_time,
    };

    Ok(Json(response))
}

/// Generate embeddings for multiple entities
#[cfg(feature = "api-server")]
pub async fn embed_batch(
    State(state): State<ApiState>,
    Json(request): Json<BatchEmbeddingRequest>,
) -> Result<Json<BatchEmbeddingResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    // Validate batch size
    if request.entities.len() > state.config.max_batch_size {
        return Err(StatusCode::BAD_REQUEST);
    }

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

    let use_cache = request.use_cache.unwrap_or(true);
    let mut embeddings = Vec::new();
    let mut cache_hits = 0;
    let mut cache_misses = 0;

    // Process entities
    for entity in &request.entities {
        let entity_start = std::time::Instant::now();

        let (embedding, from_cache) = if use_cache {
            let had_cache = state.cache_manager.get_embedding(&entity).is_some();
            match cached_model.get_entity_embedding_cached(&entity) {
                Ok(emb) => {
                    if had_cache {
                        cache_hits += 1;
                    } else {
                        cache_misses += 1;
                    }
                    (emb, had_cache)
                }
                Err(_) => continue, // Skip failed embeddings
            }
        } else {
            match model.get_entity_embedding(&entity) {
                Ok(emb) => {
                    cache_misses += 1;
                    (emb, false)
                }
                Err(_) => continue,
            }
        };

        let generation_time = entity_start.elapsed().as_millis() as f64;

        embeddings.push(EmbeddingResponse {
            entity: entity.clone(),
            embedding: embedding.values,
            dimensions: embedding.dimensions,
            model_version,
            from_cache,
            generation_time_ms: generation_time,
        });
    }

    let total_time = start_time.elapsed().as_millis() as f64;

    let response = BatchEmbeddingResponse {
        embeddings,
        total_time_ms: total_time,
        cache_hits,
        cache_misses,
    };

    Ok(Json(response))
}
