//! Embedding generation HTTP handlers
//!
//! This module contains handlers for all embedding-related API endpoints.

use super::super::helpers::get_production_model_version;
use super::super::{
    ApiState, BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
};
#[cfg(feature = "api-server")]
use axum::{extract::State, http::StatusCode, response::Json};

/// Generate embedding for a single entity
#[cfg(feature = "api-server")]
pub async fn embed_single(
    State(state): State<ApiState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    // Get model version
    let model_version = if let Some(version) = request.model_version {
        match version.parse::<uuid::Uuid>() {
            Ok(uuid) => uuid,
            Err(_) => return Err(StatusCode::BAD_REQUEST),
        }
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

    // Generate embedding with caching
    let use_cache = request.use_cache.unwrap_or(true);
    let (embedding, from_cache) = if use_cache {
        // Check cache first
        if let Some(cached_embedding) = state.cache_manager.get_embedding(&request.entity) {
            (cached_embedding, true)
        } else {
            // Cache miss - get from model and cache result
            match model.get_entity_embedding(&request.entity) {
                Ok(emb) => {
                    state
                        .cache_manager
                        .put_embedding(request.entity.clone(), emb.clone());
                    (emb, false)
                }
                Err(_) => return Err(StatusCode::NOT_FOUND),
            }
        }
    } else {
        match model.get_entity_embedding(&request.entity) {
            Ok(emb) => (emb, false),
            Err(_) => return Err(StatusCode::NOT_FOUND),
        }
    };

    let generation_time = start_time.elapsed().as_millis() as f64;

    let dimensions = embedding.dimensions;
    let response = EmbeddingResponse {
        entity_id: request.entity_id.clone(),
        entity: request.entity,
        embedding,
        dimensions,
        model_id: request.model_id.unwrap_or(model_version),
        model_version: model_version.to_string(),
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

    let use_cache = request.use_cache.unwrap_or(true);
    let mut embeddings = Vec::new();
    let mut cache_hits = 0;
    let mut cache_misses = 0;

    // Process entities
    for entity in &request.entities {
        let entity_start = std::time::Instant::now();

        let (embedding, from_cache) = if use_cache {
            // Check cache first
            if let Some(cached_embedding) = state.cache_manager.get_embedding(entity) {
                cache_hits += 1;
                (cached_embedding, true)
            } else {
                // Cache miss - get from model and cache result
                match model.get_entity_embedding(entity) {
                    Ok(emb) => {
                        state
                            .cache_manager
                            .put_embedding(entity.clone(), emb.clone());
                        cache_misses += 1;
                        (emb, false)
                    }
                    Err(_) => continue, // Skip failed embeddings
                }
            }
        } else {
            match model.get_entity_embedding(entity) {
                Ok(emb) => {
                    cache_misses += 1;
                    (emb, false)
                }
                Err(_) => continue,
            }
        };

        let generation_time = entity_start.elapsed().as_millis() as f64;

        let dimensions = embedding.dimensions;
        embeddings.push(EmbeddingResponse {
            entity_id: entity.clone(),
            entity: entity.clone(),
            embedding,
            dimensions,
            model_id: request.model_id.unwrap_or(model_version),
            model_version: model_version.to_string(),
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
        model_id: request.model_id.unwrap_or(model_version),
    };

    Ok(Json(response))
}
