//! Admin endpoints for GPU knowledge graph embeddings management
//!
//! This module provides HTTP endpoints for managing and querying
//! GPU-accelerated knowledge graph embeddings.

use crate::error::FusekiResult;
use crate::gpu_kg_embeddings::{
    EmbeddingConfig, EmbeddingModel, GpuBackendType, GpuEmbeddingGenerator, TrainingMetrics,
};
use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Global GPU embedding generator instance
pub type EmbeddingGeneratorState = Arc<RwLock<Option<GpuEmbeddingGenerator>>>;

/// Request to initialize embeddings
#[derive(Debug, Deserialize)]
pub struct InitializeEmbeddingsRequest {
    pub triples: Vec<KnowledgeGraphTriple>,
    pub config: Option<EmbeddingConfigRequest>,
}

/// Knowledge graph triple for API
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KnowledgeGraphTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Embedding configuration for API
#[derive(Debug, Deserialize)]
pub struct EmbeddingConfigRequest {
    pub embedding_dim: Option<usize>,
    pub learning_rate: Option<f32>,
    pub batch_size: Option<usize>,
    pub num_negatives: Option<usize>,
    pub model: Option<String>,
    pub backend: Option<String>,
    pub use_mixed_precision: Option<bool>,
    pub use_tensor_cores: Option<bool>,
}

/// Request to train embeddings
#[derive(Debug, Deserialize)]
pub struct TrainEmbeddingsRequest {
    pub epochs: usize,
}

/// Query parameters for similarity search
#[derive(Debug, Deserialize)]
pub struct SimilarityQueryParams {
    pub entity: String,
    pub top_k: Option<usize>,
}

/// Response from similarity search
#[derive(Debug, Serialize)]
pub struct SimilarityResponse {
    pub entity: String,
    pub similar_entities: Vec<SimilarEntity>,
    pub execution_time_ms: f64,
}

/// Similar entity with score
#[derive(Debug, Serialize)]
pub struct SimilarEntity {
    pub entity: String,
    pub similarity: f32,
}

/// Embedding statistics response
#[derive(Debug, Serialize)]
pub struct EmbeddingStatsResponse {
    pub num_entities: usize,
    pub num_relations: usize,
    pub embedding_dim: usize,
    pub model: String,
    pub gpu_enabled: bool,
    pub tensor_core_enabled: bool,
    pub total_parameters: usize,
}

/// Training metrics response
#[derive(Debug, Serialize)]
pub struct TrainingMetricsResponse {
    pub epochs: usize,
    pub average_loss: f64,
    pub training_time_ms: f64,
    pub gpu_accelerated: bool,
    pub tensor_core_used: bool,
}

/// Entity embedding response
#[derive(Debug, Serialize)]
pub struct EntityEmbeddingResponse {
    pub entity: String,
    pub embedding: Vec<f32>,
    pub embedding_dim: usize,
}

/// Convert string to embedding model
fn parse_embedding_model(s: &str) -> FusekiResult<EmbeddingModel> {
    match s.to_lowercase().as_str() {
        "transe" => Ok(EmbeddingModel::TransE),
        "distmult" => Ok(EmbeddingModel::DistMult),
        "complex" => Ok(EmbeddingModel::ComplEx),
        "rotate" => Ok(EmbeddingModel::RotatE),
        _ => Err(crate::error::FusekiError::bad_request(format!(
            "Unknown embedding model: {}. Valid options: transe, distmult, complex, rotate",
            s
        ))),
    }
}

/// Convert string to GPU backend type
fn parse_backend(s: &str) -> FusekiResult<GpuBackendType> {
    match s.to_lowercase().as_str() {
        "cuda" => Ok(GpuBackendType::Cuda),
        "metal" => Ok(GpuBackendType::Metal),
        "cpu" => Ok(GpuBackendType::Cpu),
        _ => Err(crate::error::FusekiError::bad_request(format!(
            "Unknown backend: {}. Valid options: cuda, metal, cpu",
            s
        ))),
    }
}

/// Handler: POST /$/embeddings/initialize
///
/// Initialize the embedding generator from knowledge graph triples
pub async fn initialize_embeddings(
    State(generator_state): State<EmbeddingGeneratorState>,
    Json(request): Json<InitializeEmbeddingsRequest>,
) -> FusekiResult<impl IntoResponse> {
    info!(
        "Initializing embeddings from {} triples",
        request.triples.len()
    );

    // Build configuration
    let mut config = EmbeddingConfig::default();

    if let Some(cfg) = request.config {
        if let Some(dim) = cfg.embedding_dim {
            config.embedding_dim = dim;
        }
        if let Some(lr) = cfg.learning_rate {
            config.learning_rate = lr;
        }
        if let Some(batch) = cfg.batch_size {
            config.batch_size = batch;
        }
        if let Some(neg) = cfg.num_negatives {
            config.num_negatives = neg;
        }
        if let Some(model) = cfg.model {
            config.model = parse_embedding_model(&model)?;
        }
        if let Some(backend) = cfg.backend {
            config.backend = parse_backend(&backend)?;
        }
        if let Some(mixed) = cfg.use_mixed_precision {
            config.use_mixed_precision = mixed;
        }
        if let Some(tensor) = cfg.use_tensor_cores {
            config.use_tensor_cores = tensor;
        }
    }

    // Create generator
    let mut generator = GpuEmbeddingGenerator::new(config)?;

    // Convert triples
    let triples: Vec<(String, String, String)> = request
        .triples
        .into_iter()
        .map(|t| (t.subject, t.predicate, t.object))
        .collect();

    // Initialize from triples
    generator.initialize_from_triples(&triples)?;

    let stats = generator.get_statistics();

    // Store generator
    let mut gen_state = generator_state.write().await;
    *gen_state = Some(generator);

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "status": "success",
            "num_entities": stats.num_entities,
            "num_relations": stats.num_relations,
            "embedding_dim": stats.embedding_dim,
            "message": "Embeddings initialized successfully"
        })),
    ))
}

/// Handler: POST /$/embeddings/train
///
/// Train the embedding model
pub async fn train_embeddings(
    State(generator_state): State<EmbeddingGeneratorState>,
    Json(request): Json<TrainEmbeddingsRequest>,
) -> FusekiResult<impl IntoResponse> {
    info!("Training embeddings for {} epochs", request.epochs);

    let mut gen_state = generator_state.write().await;

    let generator = gen_state.as_mut().ok_or_else(|| {
        crate::error::FusekiError::bad_request(
            "Embeddings not initialized. Call /$/embeddings/initialize first".to_string(),
        )
    })?;

    // Note: We need to get triples from somewhere - for now, return error
    // In production, we would store the original triples or get them from the store
    Err::<(StatusCode, Json<serde_json::Value>), _>(crate::error::FusekiError::bad_request(
        "Training not yet implemented - triples need to be stored".to_string(),
    ))
}

/// Handler: GET /$/embeddings/similarity
///
/// Find similar entities
pub async fn find_similar(
    State(generator_state): State<EmbeddingGeneratorState>,
    Query(params): Query<SimilarityQueryParams>,
) -> FusekiResult<impl IntoResponse> {
    debug!("Finding similar entities to: {}", params.entity);

    let gen_state = generator_state.read().await;

    let generator = gen_state.as_ref().ok_or_else(|| {
        crate::error::FusekiError::bad_request(
            "Embeddings not initialized. Call /$/embeddings/initialize first".to_string(),
        )
    })?;

    let top_k = params.top_k.unwrap_or(10);

    let start = std::time::Instant::now();
    let similar = generator.find_similar_entities(&params.entity, top_k);
    let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let similar_entities: Vec<SimilarEntity> = similar
        .into_iter()
        .map(|(entity, similarity)| SimilarEntity { entity, similarity })
        .collect();

    Ok(Json(SimilarityResponse {
        entity: params.entity,
        similar_entities,
        execution_time_ms,
    }))
}

/// Handler: GET /$/embeddings/entity/{entity}
///
/// Get embedding for a specific entity
pub async fn get_entity_embedding(
    State(generator_state): State<EmbeddingGeneratorState>,
    axum::extract::Path(entity): axum::extract::Path<String>,
) -> FusekiResult<impl IntoResponse> {
    debug!("Getting embedding for entity: {}", entity);

    let gen_state = generator_state.read().await;

    let generator = gen_state.as_ref().ok_or_else(|| {
        crate::error::FusekiError::bad_request(
            "Embeddings not initialized. Call /$/embeddings/initialize first".to_string(),
        )
    })?;

    let embedding = generator.get_entity_embedding(&entity).ok_or_else(|| {
        crate::error::FusekiError::NotFound {
            resource: format!("Entity: {}", entity),
        }
    })?;

    let embedding_vec = embedding.to_vec();
    let embedding_dim = embedding_vec.len();

    Ok(Json(EntityEmbeddingResponse {
        entity,
        embedding: embedding_vec,
        embedding_dim,
    }))
}

/// Handler: GET /$/embeddings/stats
///
/// Get embedding statistics
pub async fn get_embedding_stats(
    State(generator_state): State<EmbeddingGeneratorState>,
) -> FusekiResult<impl IntoResponse> {
    debug!("Getting embedding statistics");

    let gen_state = generator_state.read().await;

    let generator = gen_state.as_ref().ok_or_else(|| {
        crate::error::FusekiError::bad_request(
            "Embeddings not initialized. Call /$/embeddings/initialize first".to_string(),
        )
    })?;

    let stats = generator.get_statistics();

    let model_str = match stats.model {
        EmbeddingModel::TransE => "TransE",
        EmbeddingModel::DistMult => "DistMult",
        EmbeddingModel::ComplEx => "ComplEx",
        EmbeddingModel::RotatE => "RotatE",
    };

    Ok(Json(EmbeddingStatsResponse {
        num_entities: stats.num_entities,
        num_relations: stats.num_relations,
        embedding_dim: stats.embedding_dim,
        model: model_str.to_string(),
        gpu_enabled: stats.gpu_enabled,
        tensor_core_enabled: stats.tensor_core_enabled,
        total_parameters: stats.total_parameters,
    }))
}

/// Handler: DELETE /$/embeddings/clear
///
/// Clear the embedding generator
pub async fn clear_embeddings(
    State(generator_state): State<EmbeddingGeneratorState>,
) -> FusekiResult<impl IntoResponse> {
    info!("Clearing embeddings");

    let mut gen_state = generator_state.write().await;

    *gen_state = None;

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "success",
            "message": "Embeddings cleared"
        })),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_parse_embedding_model() {
        assert!(matches!(
            parse_embedding_model("transe").unwrap(),
            EmbeddingModel::TransE
        ));
        assert!(matches!(
            parse_embedding_model("TransE").unwrap(),
            EmbeddingModel::TransE
        ));
        assert!(matches!(
            parse_embedding_model("distmult").unwrap(),
            EmbeddingModel::DistMult
        ));
        assert!(parse_embedding_model("invalid").is_err());
    }

    #[tokio::test]
    async fn test_parse_backend() {
        assert!(matches!(
            parse_backend("cuda").unwrap(),
            GpuBackendType::Cuda
        ));
        assert!(matches!(
            parse_backend("METAL").unwrap(),
            GpuBackendType::Metal
        ));
        assert!(matches!(parse_backend("cpu").unwrap(), GpuBackendType::Cpu));
        assert!(parse_backend("invalid").is_err());
    }

    #[tokio::test]
    #[allow(clippy::arc_with_non_send_sync)]
    async fn test_initialize_embeddings() {
        let generator_state = Arc::new(tokio::sync::RwLock::new(None));

        let request = InitializeEmbeddingsRequest {
            triples: vec![
                KnowledgeGraphTriple {
                    subject: "Alice".to_string(),
                    predicate: "knows".to_string(),
                    object: "Bob".to_string(),
                },
                KnowledgeGraphTriple {
                    subject: "Bob".to_string(),
                    predicate: "knows".to_string(),
                    object: "Charlie".to_string(),
                },
            ],
            config: Some(EmbeddingConfigRequest {
                embedding_dim: Some(64),
                backend: Some("cpu".to_string()),
                model: Some("transe".to_string()),
                learning_rate: None,
                batch_size: None,
                num_negatives: None,
                use_mixed_precision: None,
                use_tensor_cores: None,
            }),
        };

        let response = initialize_embeddings(State(generator_state.clone()), Json(request))
            .await
            .unwrap();

        // Verify generator was created
        let gen = generator_state.read().await;
        assert!(gen.is_some());
    }

    #[tokio::test]
    #[allow(clippy::arc_with_non_send_sync)]
    async fn test_get_stats_without_init() {
        let generator_state = Arc::new(tokio::sync::RwLock::new(None));

        let result = get_embedding_stats(State(generator_state)).await;
        assert!(result.is_err());
    }
}
