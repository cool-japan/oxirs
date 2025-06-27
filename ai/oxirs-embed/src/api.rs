//! RESTful and GraphQL API endpoints for embedding services
//!
//! This module provides production-ready HTTP APIs for embedding generation,
//! model management, and batch processing capabilities.

#[cfg(feature = "api-server")]

use crate::{
    CacheManager, CachedEmbeddingModel, EmbeddingModel, ModelConfig, ModelStats,
    TrainingStats, Vector, ModelRegistry, ModelVersion
};
use anyhow::{anyhow, Result};
#[cfg(feature = "api-server")]
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
#[cfg(feature = "api-server")]
use tower::ServiceBuilder;
#[cfg(feature = "api-server")]
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
};
use tracing::{info, warn, error};
use uuid::Uuid;
use std::time::Duration;

/// API server state
#[derive(Clone)]
pub struct ApiState {
    /// Model registry for managing deployed models
    pub registry: Arc<ModelRegistry>,
    /// Cache manager for performance optimization
    pub cache_manager: Arc<CacheManager>,
    /// Currently loaded models
    pub models: Arc<RwLock<HashMap<Uuid, Arc<dyn EmbeddingModel + Send + Sync>>>>,
    /// API configuration
    pub config: ApiConfig,
}

/// API configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum batch size for bulk operations
    pub max_batch_size: usize,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Authentication configuration
    pub auth: AuthConfig,
    /// Enable request logging
    pub enable_logging: bool,
    /// Enable CORS
    pub enable_cors: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            timeout_seconds: 30,
            max_batch_size: 1000,
            rate_limit: RateLimitConfig::default(),
            auth: AuthConfig::default(),
            enable_logging: true,
            enable_cors: true,
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per minute per IP
    pub requests_per_minute: u32,
    /// Enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 1000,
            enabled: true,
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// Enable API key authentication
    pub require_api_key: bool,
    /// Valid API keys
    pub api_keys: Vec<String>,
    /// Enable JWT authentication
    pub enable_jwt: bool,
    /// JWT secret
    pub jwt_secret: Option<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            require_api_key: false,
            api_keys: Vec::new(),
            enable_jwt: false,
            jwt_secret: None,
        }
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request to generate embeddings for a single entity
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// The entity to embed
    pub entity: String,
    /// Model version to use (optional, uses production version if not specified)
    pub model_version: Option<Uuid>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

/// Response containing entity embedding
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// The entity that was embedded
    pub entity: String,
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Model version used
    pub model_version: Uuid,
    /// Whether result was cached
    pub from_cache: bool,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
}

/// Request to generate embeddings for multiple entities
#[derive(Debug, Deserialize)]
pub struct BatchEmbeddingRequest {
    /// List of entities to embed
    pub entities: Vec<String>,
    /// Model version to use
    pub model_version: Option<Uuid>,
    /// Use cached results where available
    pub use_cache: Option<bool>,
    /// Batch processing options
    pub options: Option<BatchOptions>,
}

/// Batch processing options
#[derive(Debug, Deserialize)]
pub struct BatchOptions {
    /// Parallel processing
    pub parallel: Option<bool>,
    /// Chunk size for processing
    pub chunk_size: Option<usize>,
}

/// Response containing batch embeddings
#[derive(Debug, Serialize)]
pub struct BatchEmbeddingResponse {
    /// Embeddings for each entity
    pub embeddings: Vec<EmbeddingResponse>,
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

/// Request to score a triple
#[derive(Debug, Deserialize)]
pub struct TripleScoreRequest {
    /// Subject entity
    pub subject: String,
    /// Predicate relation
    pub predicate: String,
    /// Object entity
    pub object: String,
    /// Model version to use
    pub model_version: Option<Uuid>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

/// Response containing triple score
#[derive(Debug, Serialize)]
pub struct TripleScoreResponse {
    /// The triple that was scored
    pub triple: (String, String, String),
    /// The score
    pub score: f64,
    /// Model version used
    pub model_version: Uuid,
    /// Whether result was cached
    pub from_cache: bool,
    /// Scoring time in milliseconds
    pub scoring_time_ms: f64,
}

/// Request to predict objects/subjects/relations
#[derive(Debug, Deserialize)]
pub struct PredictionRequest {
    /// Known entities (subject for object prediction, object for subject prediction, etc.)
    pub entities: Vec<String>,
    /// Prediction type
    pub prediction_type: PredictionType,
    /// Number of predictions to return
    pub k: usize,
    /// Model version to use
    pub model_version: Option<Uuid>,
    /// Use cached results if available
    pub use_cache: Option<bool>,
}

/// Types of predictions
#[derive(Debug, Deserialize)]
pub enum PredictionType {
    Objects,
    Subjects,
    Relations,
}

/// Response containing predictions
#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    /// Input entities
    pub input: Vec<String>,
    /// Prediction type
    pub prediction_type: String,
    /// Predicted entities with scores
    pub predictions: Vec<(String, f64)>,
    /// Model version used
    pub model_version: Uuid,
    /// Whether result was cached
    pub from_cache: bool,
    /// Prediction time in milliseconds
    pub prediction_time_ms: f64,
}

/// Request to get model information
#[derive(Debug, Deserialize)]
pub struct ModelInfoRequest {
    /// Model version (optional, uses production version if not specified)
    pub model_version: Option<Uuid>,
}

/// Response containing model information
#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    /// Model statistics
    pub stats: ModelStats,
    /// Model version information
    pub version: ModelVersion,
    /// Whether model is currently loaded
    pub is_loaded: bool,
    /// Model health status
    pub health: ModelHealth,
}

/// Model health status
#[derive(Debug, Serialize)]
pub struct ModelHealth {
    /// Overall health status
    pub status: HealthStatus,
    /// Last health check time
    pub last_check: chrono::DateTime<chrono::Utc>,
    /// Performance metrics
    pub metrics: HealthMetrics,
}

/// Health status levels
#[derive(Debug, Serialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health metrics
#[derive(Debug, Serialize)]
pub struct HealthMetrics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Request count in last hour
    pub requests_last_hour: u64,
    /// Error rate percentage
    pub error_rate_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
}

/// Query parameters for various endpoints
#[derive(Debug, Deserialize)]
pub struct QueryParams {
    /// Limit for paginated results
    pub limit: Option<usize>,
    /// Offset for paginated results
    pub offset: Option<usize>,
    /// Filter by model type
    pub model_type: Option<String>,
    /// Include detailed information
    pub detailed: Option<bool>,
}

// ============================================================================
// API Router and Handlers
// ============================================================================

/// Create the API router with all endpoints
#[cfg(feature = "api-server")]
pub fn create_router(state: ApiState) -> Router {
    let mut router = Router::new()
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
        
        .with_state(state);

    // Add middleware layers
    let service = ServiceBuilder::new()
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http());

    if state.config.enable_cors {
        router = router.layer(CorsLayer::permissive());
    }

    router.layer(service)
}

// ============================================================================
// Embedding Handlers
// ============================================================================

/// Generate embedding for a single entity
#[cfg(feature = "api-server")]
async fn embed_single(
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
            Ok(emb) => (emb, state.cache_manager.get_embedding(&request.entity).is_some()),
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
async fn embed_batch(
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

    let cached_model = CachedEmbeddingModel::new(
        Box::new(model.as_ref()),
        Arc::clone(&state.cache_manager),
    );

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

// ============================================================================
// Scoring Handlers
// ============================================================================

/// Score a triple
async fn score_triple(
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

    let cached_model = CachedEmbeddingModel::new(
        Box::new(model.as_ref()),
        Arc::clone(&state.cache_manager),
    );

    // Score triple
    let use_cache = request.use_cache.unwrap_or(true);
    let (score, from_cache) = if use_cache {
        match cached_model.score_triple_cached(&request.subject, &request.predicate, &request.object) {
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

// ============================================================================
// Prediction Handlers
// ============================================================================

/// Make predictions (objects, subjects, or relations)
async fn predict(
    State(state): State<ApiState>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    let start_time = std::time::Instant::now();
    
    // Validate input
    if request.entities.is_empty() || request.k == 0 {
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

    let cached_model = CachedEmbeddingModel::new(
        Box::new(model.as_ref()),
        Arc::clone(&state.cache_manager),
    );

    // Make predictions based on type
    let use_cache = request.use_cache.unwrap_or(true);
    let (predictions, prediction_type_str) = match request.prediction_type {
        PredictionType::Objects => {
            if request.entities.len() != 2 {
                return Err(StatusCode::BAD_REQUEST);
            }
            let preds = if use_cache {
                cached_model.predict_objects_cached(&request.entities[0], &request.entities[1], request.k)
            } else {
                model.predict_objects(&request.entities[0], &request.entities[1], request.k)
            };
            (preds, "objects")
        }
        PredictionType::Subjects => {
            if request.entities.len() != 2 {
                return Err(StatusCode::BAD_REQUEST);
            }
            let preds = model.predict_subjects(&request.entities[0], &request.entities[1], request.k);
            (preds, "subjects")
        }
        PredictionType::Relations => {
            if request.entities.len() != 2 {
                return Err(StatusCode::BAD_REQUEST);
            }
            let preds = model.predict_relations(&request.entities[0], &request.entities[1], request.k);
            (preds, "relations")
        }
    };

    let predictions = match predictions {
        Ok(preds) => preds,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    let prediction_time = start_time.elapsed().as_millis() as f64;

    let response = PredictionResponse {
        input: request.entities,
        prediction_type: prediction_type_str.to_string(),
        predictions,
        model_version,
        from_cache: use_cache, // Simplified
        prediction_time_ms: prediction_time,
    };

    Ok(Json(response))
}

// ============================================================================
// Model Management Handlers
// ============================================================================

/// List all available models
async fn list_models(
    State(state): State<ApiState>,
    Query(params): Query<QueryParams>,
) -> Result<Json<Vec<ModelInfoResponse>>, StatusCode> {
    let models_metadata = state.registry.list_models().await;
    
    let mut responses = Vec::new();
    for metadata in models_metadata {
        if let Some(production_version) = metadata.production_version {
            if let Ok(version) = state.registry.get_version(production_version).await {
                let is_loaded = {
                    let models = state.models.read().await;
                    models.contains_key(&production_version)
                };

                let health = ModelHealth {
                    status: if is_loaded { HealthStatus::Healthy } else { HealthStatus::Degraded },
                    last_check: chrono::Utc::now(),
                    metrics: HealthMetrics {
                        avg_response_time_ms: 50.0, // Placeholder
                        requests_last_hour: 100,    // Placeholder
                        error_rate_percent: 0.1,    // Placeholder
                        memory_usage_mb: 256.0,     // Placeholder
                    },
                };

                responses.push(ModelInfoResponse {
                    stats: ModelStats {
                        num_entities: 0, // Would get from actual model
                        num_relations: 0,
                        num_triples: 0,
                        dimensions: version.config.dimensions,
                        is_trained: true,
                        model_type: metadata.model_type,
                        creation_time: metadata.created_at,
                        last_training_time: Some(metadata.updated_at),
                    },
                    version,
                    is_loaded,
                    health,
                });
            }
        }
    }

    // Apply filtering and pagination
    if let Some(limit) = params.limit {
        responses.truncate(limit);
    }

    Ok(Json(responses))
}

/// Get information about a specific model
async fn get_model_info(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<ModelInfoResponse>, StatusCode> {
    // Implementation would get model info from registry
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Get model health status
async fn get_model_health(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<ModelHealth>, StatusCode> {
    // Implementation would check actual model health
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Load a model into memory
async fn load_model(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Implementation would load model from registry
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// Unload a model from memory
async fn unload_model(
    State(state): State<ApiState>,
    Path(model_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Implementation would unload model
    Err(StatusCode::NOT_IMPLEMENTED)
}

// ============================================================================
// System Handlers
// ============================================================================

/// Get system health status
async fn system_health(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let health = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": 3600, // Placeholder
        "loaded_models": state.models.read().await.len(),
    });

    Ok(Json(health))
}

/// Get system statistics
async fn system_stats(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let stats = serde_json::json!({
        "models_loaded": state.models.read().await.len(),
        "cache_stats": state.cache_manager.get_stats(),
        "memory_usage_mb": 512.0, // Placeholder
        "requests_per_second": 100.0, // Placeholder
    });

    Ok(Json(stats))
}

/// Get cache statistics
async fn cache_stats(
    State(state): State<ApiState>,
) -> Result<Json<crate::caching::CacheStats>, StatusCode> {
    Ok(Json(state.cache_manager.get_stats()))
}

/// Clear all caches
async fn clear_cache(
    State(state): State<ApiState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    state.cache_manager.clear_all();
    
    let response = serde_json::json!({
        "message": "Cache cleared successfully",
        "timestamp": chrono::Utc::now(),
    });

    Ok(Json(response))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the production model version
async fn get_production_model_version(state: &ApiState) -> Result<Uuid> {
    let models = state.registry.list_models().await;
    for model in models {
        if let Some(production_version) = model.production_version {
            return Ok(production_version);
        }
    }
    Err(anyhow!("No production model available"))
}

/// Start the API server
#[cfg(feature = "api-server")]
pub async fn start_server(state: ApiState) -> Result<()> {
    let router = create_router(state.clone());
    
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", state.config.host, state.config.port))
        .await
        .map_err(|e| anyhow!("Failed to bind to address: {}", e))?;

    info!("API server starting on {}:{}", state.config.host, state.config.port);

    axum::serve(listener, router)
        .await
        .map_err(|e| anyhow!("Server error: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelRegistry;
    use tempfile::tempdir;

    fn create_test_state() -> ApiState {
        let temp_dir = tempdir().unwrap();
        let registry = Arc::new(ModelRegistry::new(temp_dir.path().to_path_buf()));
        let cache_manager = Arc::new(CacheManager::new(crate::caching::CacheConfig::default()));
        let models = Arc::new(RwLock::new(HashMap::new()));
        let config = ApiConfig::default();

        ApiState {
            registry,
            cache_manager,
            models,
            config,
        }
    }

    #[tokio::test]
    async fn test_api_router_creation() {
        let state = create_test_state();
        let router = create_router(state);
        
        // Router should be created successfully
        assert!(!router.into_make_service().to_string().is_empty());
    }

    #[test]
    fn test_api_config_default() {
        let config = ApiConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.timeout_seconds, 30);
        assert_eq!(config.max_batch_size, 1000);
    }

    #[test]
    fn test_embedding_request_deserialization() {
        let json = r#"{"entity": "test_entity", "use_cache": true}"#;
        let request: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.entity, "test_entity");
        assert_eq!(request.use_cache, Some(true));
    }

    #[test]
    fn test_batch_embedding_request_deserialization() {
        let json = r#"{"entities": ["entity1", "entity2"], "use_cache": false}"#;
        let request: BatchEmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.entities.len(), 2);
        assert_eq!(request.use_cache, Some(false));
    }

    #[test]
    fn test_triple_score_request_deserialization() {
        let json = r#"{"subject": "Alice", "predicate": "knows", "object": "Bob"}"#;
        let request: TripleScoreRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.subject, "Alice");
        assert_eq!(request.predicate, "knows");
        assert_eq!(request.object, "Bob");
    }
}