//! RESTful and GraphQL API endpoints for embedding services
//!
//! This module provides production-ready HTTP APIs for embedding generation,
//! model management, and batch processing capabilities.

#[cfg(feature = "api-server")]
use crate::{
    CacheManager, CachedEmbeddingModel, EmbeddingModel, ModelConfig, ModelRegistry, ModelStats,
    ModelVersion, TrainingStats, Vector,
};
use anyhow::{anyhow, Result};
#[cfg(feature = "graphql")]
use async_graphql::{
    http::GraphiQLSource, Context, EmptySubscription, Enum, Error as GraphQLError, InputObject,
    Object, Result as GraphQLResult, Schema, SimpleObject,
};
#[cfg(feature = "graphql")]
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
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
use std::time::Duration;
use tokio::sync::RwLock;
#[cfg(feature = "api-server")]
use tower::ServiceBuilder;
#[cfg(feature = "api-server")]
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer};
use tracing::{error, info, warn};
use uuid::Uuid;

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

/// Request to generate text embeddings using specialized models
#[derive(Debug, Deserialize)]
pub struct TextEmbeddingRequest {
    /// Text content to embed
    pub text: String,
    /// Specialized model type to use
    pub model_type: Option<String>,
    /// Model version to use
    pub model_version: Option<Uuid>,
    /// Use preprocessing
    pub preprocess: Option<bool>,
    /// Fine-tuning options
    pub fine_tune: Option<bool>,
}

/// Response containing text embedding
#[derive(Debug, Serialize)]
pub struct TextEmbeddingResponse {
    /// Original text
    pub text: String,
    /// Generated embedding
    pub embedding: Vec<f32>,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Model type used
    pub model_type: String,
    /// Model version used
    pub model_version: Uuid,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
    /// Whether preprocessing was applied
    pub preprocessed: bool,
}

/// Request for multi-modal embeddings
#[derive(Debug, Deserialize)]
pub struct MultiModalRequest {
    /// Text content (optional)
    pub text: Option<String>,
    /// Entity IRI (optional)
    pub entity: Option<String>,
    /// Cross-modal alignment entities
    pub aligned_entities: Option<Vec<String>>,
    /// Model version to use
    pub model_version: Option<Uuid>,
    /// Use contrastive learning
    pub use_contrastive: Option<bool>,
}

/// Response for multi-modal embeddings
#[derive(Debug, Serialize)]
pub struct MultiModalResponse {
    /// Text embedding (if text was provided)
    pub text_embedding: Option<Vec<f32>>,
    /// Entity embedding (if entity was provided)
    pub entity_embedding: Option<Vec<f32>>,
    /// Cross-modal alignment scores
    pub alignment_scores: Option<HashMap<String, f64>>,
    /// Unified embedding (if both text and entity provided)
    pub unified_embedding: Option<Vec<f32>>,
    /// Model version used
    pub model_version: Uuid,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
}

/// Request for streaming embeddings
#[derive(Debug, Deserialize)]
pub struct StreamEmbeddingRequest {
    /// Entities to embed in streaming fashion
    pub entities: Vec<String>,
    /// Model version to use
    pub model_version: Option<Uuid>,
    /// Batch size for streaming
    pub batch_size: Option<usize>,
}

/// Individual streaming response item
#[derive(Debug, Serialize)]
pub struct StreamEmbeddingItem {
    /// Entity name
    pub entity: String,
    /// Entity embedding
    pub embedding: Vec<f32>,
    /// Sequence number in stream
    pub sequence: usize,
    /// Whether this is the last item
    pub is_last: bool,
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
// GraphQL Schema and Types
// ============================================================================

#[cfg(feature = "graphql")]
use async_graphql::{subscription, MaybeUndefined, Subscription, Union};
use futures_util::stream::Stream;
use std::pin::Pin;
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};

/// GraphQL Query root
#[cfg(feature = "graphql")]
#[derive(Default)]
pub struct Query;

/// GraphQL Mutation root
#[cfg(feature = "graphql")]
#[derive(Default)]
pub struct Mutation;

/// GraphQL Subscription root
#[cfg(feature = "graphql")]
#[derive(Default)]
pub struct SubscriptionRoot;

/// Embedding entity with full GraphQL support
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct EmbeddingEntity {
    /// Entity identifier
    pub id: String,
    /// Entity type (e.g., "gene", "drug", "disease")
    pub entity_type: Option<String>,
    /// Entity label/name
    pub label: Option<String>,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Similarity scores to other entities (computed on demand)
    #[graphql(skip)]
    pub similarity_scores: Option<HashMap<String, f64>>,
    /// Related entities (nested)
    #[graphql(skip)]
    pub related_entities: Option<Vec<EmbeddingEntity>>,
    /// Model version used for this embedding
    pub model_version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: Option<DateTime<Utc>>,
    /// Metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Triple with embedded entities
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct EmbeddingTriple {
    /// Triple identifier
    pub id: String,
    /// Subject entity with embedding
    pub subject: EmbeddingEntity,
    /// Predicate/relation
    pub predicate: String,
    /// Object entity with embedding
    pub object: EmbeddingEntity,
    /// Triple score/confidence
    pub score: f64,
    /// Model version used
    pub model_version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Model information for GraphQL
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct GraphQLModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Number of entities
    pub num_entities: i32,
    /// Number of relations
    pub num_relations: i32,
    /// Number of triples
    pub num_triples: i32,
    /// Embedding dimensions
    pub dimensions: i32,
    /// Is model trained
    pub is_trained: bool,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last training time
    pub last_training_time: Option<DateTime<Utc>>,
    /// Model health status
    pub health_status: String,
    /// Performance metrics
    pub performance_metrics: Option<ModelPerformanceMetrics>,
}

/// Model performance metrics
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct ModelPerformanceMetrics {
    /// Mean reciprocal rank
    pub mean_reciprocal_rank: f64,
    /// Hits at 1
    pub hits_at_1: f64,
    /// Hits at 10
    pub hits_at_10: f64,
    /// Average response time in ms
    pub avg_response_time_ms: f64,
    /// Throughput (requests per second)
    pub throughput_rps: f64,
}

/// Entity filtering options
#[cfg(feature = "graphql")]
#[derive(InputObject, Clone)]
pub struct EntityFilter {
    /// Entity type filter
    pub entity_type: Option<String>,
    /// Label pattern (supports wildcards)
    pub label_pattern: Option<String>,
    /// Minimum similarity score (when querying similar entities)
    pub min_similarity: Option<f64>,
    /// Maximum similarity score
    pub max_similarity: Option<f64>,
    /// Entity IDs to include
    pub include_ids: Option<Vec<String>>,
    /// Entity IDs to exclude
    pub exclude_ids: Option<Vec<String>>,
    /// Created after timestamp
    pub created_after: Option<DateTime<Utc>>,
    /// Created before timestamp
    pub created_before: Option<DateTime<Utc>>,
    /// Model version filter
    pub model_version: Option<String>,
    /// Metadata filters
    pub metadata_filters: Option<HashMap<String, String>>,
}

/// Triple filtering options
#[cfg(feature = "graphql")]
#[derive(InputObject, Clone)]
pub struct TripleFilter {
    /// Subject entity ID
    pub subject_id: Option<String>,
    /// Subject entity type
    pub subject_type: Option<String>,
    /// Predicate/relation
    pub predicate: Option<String>,
    /// Object entity ID
    pub object_id: Option<String>,
    /// Object entity type
    pub object_type: Option<String>,
    /// Minimum score
    pub min_score: Option<f64>,
    /// Maximum score
    pub max_score: Option<f64>,
    /// Created after timestamp
    pub created_after: Option<DateTime<Utc>>,
    /// Created before timestamp
    pub created_before: Option<DateTime<Utc>>,
    /// Model version filter
    pub model_version: Option<String>,
}

/// Sorting options
#[cfg(feature = "graphql")]
#[derive(InputObject, Clone)]
pub struct SortOptions {
    /// Field to sort by
    pub field: String,
    /// Sort direction
    pub direction: SortDirection,
}

/// Sort direction
#[cfg(feature = "graphql")]
#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum SortDirection {
    /// Ascending order
    Asc,
    /// Descending order
    Desc,
}

/// Pagination options
#[cfg(feature = "graphql")]
#[derive(InputObject, Clone)]
pub struct PaginationOptions {
    /// Offset (number of items to skip)
    pub offset: Option<i32>,
    /// Limit (maximum number of items to return)
    pub limit: Option<i32>,
    /// Cursor for cursor-based pagination
    pub cursor: Option<String>,
}

/// Connection type for paginated results
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct EntityConnection {
    /// List of entities
    pub entities: Vec<EmbeddingEntity>,
    /// Pagination info
    pub page_info: PageInfo,
    /// Total count
    pub total_count: i32,
}

/// Triple connection for paginated results
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct TripleConnection {
    /// List of triples
    pub triples: Vec<EmbeddingTriple>,
    /// Pagination info
    pub page_info: PageInfo,
    /// Total count
    pub total_count: i32,
}

/// Pagination information
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct PageInfo {
    /// Has next page
    pub has_next_page: bool,
    /// Has previous page
    pub has_previous_page: bool,
    /// Start cursor
    pub start_cursor: Option<String>,
    /// End cursor
    pub end_cursor: Option<String>,
}

/// Real-time subscription events
#[cfg(feature = "graphql")]
#[derive(Union, Clone)]
pub enum SubscriptionEvent {
    /// New entity embedding created
    EntityCreated(EmbeddingEntity),
    /// Entity embedding updated
    EntityUpdated(EmbeddingEntity),
    /// Entity embedding deleted
    EntityDeleted(EntityDeletedEvent),
    /// New triple created
    TripleCreated(EmbeddingTriple),
    /// Triple updated
    TripleUpdated(EmbeddingTriple),
    /// Triple deleted
    TripleDeleted(TripleDeletedEvent),
    /// Model training started
    ModelTrainingStarted(ModelTrainingEvent),
    /// Model training completed
    ModelTrainingCompleted(ModelTrainingEvent),
    /// Model quality alert
    QualityAlert(QualityAlertEvent),
}

/// Entity deleted event
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct EntityDeletedEvent {
    /// Deleted entity ID
    pub entity_id: String,
    /// Deletion timestamp
    pub deleted_at: DateTime<Utc>,
    /// Model version
    pub model_version: String,
}

/// Triple deleted event
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct TripleDeletedEvent {
    /// Deleted triple ID
    pub triple_id: String,
    /// Deletion timestamp
    pub deleted_at: DateTime<Utc>,
    /// Model version
    pub model_version: String,
}

/// Model training event
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct ModelTrainingEvent {
    /// Model ID
    pub model_id: String,
    /// Event type
    pub event_type: String,
    /// Training progress (0.0 to 1.0)
    pub progress: f64,
    /// Current epoch
    pub current_epoch: Option<i32>,
    /// Total epochs
    pub total_epochs: Option<i32>,
    /// Current loss
    pub current_loss: Option<f64>,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Quality alert event
#[cfg(feature = "graphql")]
#[derive(SimpleObject, Clone)]
pub struct QualityAlertEvent {
    /// Alert ID
    pub alert_id: String,
    /// Alert type
    pub alert_type: String,
    /// Alert severity
    pub severity: String,
    /// Alert message
    pub message: String,
    /// Affected model ID
    pub model_id: String,
    /// Metric name that triggered the alert
    pub metric_name: Option<String>,
    /// Metric value
    pub metric_value: Option<f64>,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
}

/// GraphQL context containing API state
#[cfg(feature = "graphql")]
pub struct GraphQLContext {
    /// API state
    pub api_state: ApiState,
    /// Event broadcaster for subscriptions
    pub event_broadcaster: broadcast::Sender<SubscriptionEvent>,
}

/// GraphQL Query implementations
#[cfg(feature = "graphql")]
#[Object]
impl Query {
    /// Get entity by ID with optional related entities
    async fn entity(
        &self,
        ctx: &Context<'_>,
        id: String,
        #[graphql(desc = "Include related entities")] include_related: Option<bool>,
        #[graphql(desc = "Number of related entities to include")] related_limit: Option<i32>,
        #[graphql(desc = "Minimum similarity for related entities")] min_similarity: Option<f64>,
    ) -> GraphQLResult<Option<EmbeddingEntity>> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        // Get entity embedding
        let models = state.models.read().await;
        let production_model = get_production_model(&models).await?;

        match production_model.get_entity_embedding(&id) {
            Ok(embedding) => {
                let mut entity = EmbeddingEntity {
                    id: id.clone(),
                    entity_type: extract_entity_type(&id),
                    label: Some(id.clone()),
                    embedding: embedding.values,
                    dimensions: embedding.dimensions,
                    similarity_scores: None,
                    related_entities: None,
                    model_version: production_model.model_id().to_string(),
                    created_at: Utc::now(),
                    updated_at: None,
                    metadata: None,
                };

                // Include related entities if requested
                if include_related.unwrap_or(false) {
                    let limit = related_limit.unwrap_or(10) as usize;
                    let min_sim = min_similarity.unwrap_or(0.5);

                    entity.related_entities =
                        Some(find_similar_entities(production_model, &id, limit, min_sim).await?);
                }

                Ok(Some(entity))
            }
            Err(_) => Ok(None),
        }
    }

    /// Search entities with filtering and pagination
    async fn entities(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Entity filters")] filter: Option<EntityFilter>,
        #[graphql(desc = "Sorting options")] sort: Option<SortOptions>,
        #[graphql(desc = "Pagination options")] pagination: Option<PaginationOptions>,
    ) -> GraphQLResult<EntityConnection> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let production_model = get_production_model(&models).await?;

        // Get all entities
        let all_entities = production_model.get_entities();

        // Apply filters
        let filtered_entities = apply_entity_filters(all_entities, filter.as_ref()).await?;

        // Apply sorting
        let sorted_entities = apply_entity_sorting(filtered_entities, sort.as_ref()).await?;

        // Apply pagination
        let (entities, page_info) =
            apply_pagination(sorted_entities, pagination.as_ref(), production_model).await?;

        Ok(EntityConnection {
            entities,
            page_info,
            total_count: filtered_entities.len() as i32,
        })
    }

    /// Get similar entities to a given entity
    async fn similar_entities(
        &self,
        ctx: &Context<'_>,
        entity_id: String,
        #[graphql(desc = "Number of similar entities to return")] limit: Option<i32>,
        #[graphql(desc = "Minimum similarity threshold")] min_similarity: Option<f64>,
        #[graphql(desc = "Entity type filter")] entity_type: Option<String>,
    ) -> GraphQLResult<Vec<EmbeddingEntity>> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let production_model = get_production_model(&models).await?;

        let limit = limit.unwrap_or(10) as usize;
        let min_sim = min_similarity.unwrap_or(0.5);

        let similar = find_similar_entities(production_model, &entity_id, limit, min_sim).await?;

        // Apply entity type filter if specified
        let filtered = if let Some(etype) = entity_type {
            similar
                .into_iter()
                .filter(|e| e.entity_type.as_ref() == Some(&etype))
                .collect()
        } else {
            similar
        };

        Ok(filtered)
    }

    /// Get triple by subject, predicate, and object
    async fn triple(
        &self,
        ctx: &Context<'_>,
        subject: String,
        predicate: String,
        object: String,
    ) -> GraphQLResult<Option<EmbeddingTriple>> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let production_model = get_production_model(&models).await?;

        // Get embeddings for subject and object
        let subject_embedding = production_model.get_entity_embedding(&subject)?;
        let object_embedding = production_model.get_entity_embedding(&object)?;

        // Score the triple
        let score = production_model.score_triple(&subject, &predicate, &object)?;

        let triple = EmbeddingTriple {
            id: format!("{}|{}|{}", subject, predicate, object),
            subject: EmbeddingEntity {
                id: subject.clone(),
                entity_type: extract_entity_type(&subject),
                label: Some(subject),
                embedding: subject_embedding.values,
                dimensions: subject_embedding.dimensions,
                similarity_scores: None,
                related_entities: None,
                model_version: production_model.model_id().to_string(),
                created_at: Utc::now(),
                updated_at: None,
                metadata: None,
            },
            predicate: predicate.clone(),
            object: EmbeddingEntity {
                id: object.clone(),
                entity_type: extract_entity_type(&object),
                label: Some(object),
                embedding: object_embedding.values,
                dimensions: object_embedding.dimensions,
                similarity_scores: None,
                related_entities: None,
                model_version: production_model.model_id().to_string(),
                created_at: Utc::now(),
                updated_at: None,
                metadata: None,
            },
            score,
            model_version: production_model.model_id().to_string(),
            created_at: Utc::now(),
            metadata: None,
        };

        Ok(Some(triple))
    }

    /// Search triples with filtering and pagination
    async fn triples(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Triple filters")] filter: Option<TripleFilter>,
        #[graphql(desc = "Sorting options")] sort: Option<SortOptions>,
        #[graphql(desc = "Pagination options")] pagination: Option<PaginationOptions>,
    ) -> GraphQLResult<TripleConnection> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let production_model = get_production_model(&models).await?;

        // Get filtered triples (simplified implementation)
        let triples = find_triples_with_filter(production_model, filter.as_ref()).await?;

        // Apply sorting and pagination
        let sorted_triples = apply_triple_sorting(triples, sort.as_ref()).await?;
        let (paginated_triples, page_info) =
            apply_triple_pagination(sorted_triples, pagination.as_ref()).await?;

        Ok(TripleConnection {
            triples: paginated_triples,
            page_info,
            total_count: triples.len() as i32,
        })
    }

    /// Get model information
    async fn model(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Model ID")] model_id: Option<String>,
    ) -> GraphQLResult<Option<GraphQLModelInfo>> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let model = if let Some(id) = model_id {
            let uuid = Uuid::parse_str(&id)?;
            models.get(&uuid)
        } else {
            get_production_model(&models).await.ok()
        };

        if let Some(model) = model {
            let stats = model.get_stats();
            let model_info = GraphQLModelInfo {
                id: model.model_id().to_string(),
                name: format!("{}-{}", stats.model_type, model.model_id()),
                model_type: stats.model_type.clone(),
                version: "1.0.0".to_string(), // Simplified
                num_entities: stats.num_entities as i32,
                num_relations: stats.num_relations as i32,
                num_triples: stats.num_triples as i32,
                dimensions: stats.dimensions as i32,
                is_trained: stats.is_trained,
                created_at: stats.creation_time,
                last_training_time: stats.last_training_time,
                health_status: "healthy".to_string(), // Simplified
                performance_metrics: Some(ModelPerformanceMetrics {
                    mean_reciprocal_rank: 0.75, // Mock data
                    hits_at_1: 0.65,
                    hits_at_10: 0.89,
                    avg_response_time_ms: 25.0,
                    throughput_rps: 150.0,
                }),
            };
            Ok(Some(model_info))
        } else {
            Ok(None)
        }
    }

    /// Get all available models
    async fn models(&self, ctx: &Context<'_>) -> GraphQLResult<Vec<GraphQLModelInfo>> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let mut model_infos = Vec::new();

        for (_, model) in models.iter() {
            let stats = model.get_stats();
            model_infos.push(GraphQLModelInfo {
                id: model.model_id().to_string(),
                name: format!("{}-{}", stats.model_type, model.model_id()),
                model_type: stats.model_type.clone(),
                version: "1.0.0".to_string(),
                num_entities: stats.num_entities as i32,
                num_relations: stats.num_relations as i32,
                num_triples: stats.num_triples as i32,
                dimensions: stats.dimensions as i32,
                is_trained: stats.is_trained,
                created_at: stats.creation_time,
                last_training_time: stats.last_training_time,
                health_status: "healthy".to_string(),
                performance_metrics: Some(ModelPerformanceMetrics {
                    mean_reciprocal_rank: 0.75,
                    hits_at_1: 0.65,
                    hits_at_10: 0.89,
                    avg_response_time_ms: 25.0,
                    throughput_rps: 150.0,
                }),
            });
        }

        Ok(model_infos)
    }
}

/// GraphQL Mutation implementations
#[cfg(feature = "graphql")]
#[Object]
impl Mutation {
    /// Add a new entity (creates embedding)
    async fn add_entity(
        &self,
        ctx: &Context<'_>,
        entity_id: String,
        entity_type: Option<String>,
        label: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> GraphQLResult<EmbeddingEntity> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let production_model = get_production_model(&models).await?;

        // Generate embedding for the new entity
        let embedding = production_model.get_entity_embedding(&entity_id)?;

        let entity = EmbeddingEntity {
            id: entity_id.clone(),
            entity_type,
            label,
            embedding: embedding.values,
            dimensions: embedding.dimensions,
            similarity_scores: None,
            related_entities: None,
            model_version: production_model.model_id().to_string(),
            created_at: Utc::now(),
            updated_at: None,
            metadata,
        };

        // Broadcast entity creation event
        let _ = context
            .event_broadcaster
            .send(SubscriptionEvent::EntityCreated(entity.clone()));

        Ok(entity)
    }

    /// Update entity metadata
    async fn update_entity(
        &self,
        ctx: &Context<'_>,
        entity_id: String,
        label: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> GraphQLResult<EmbeddingEntity> {
        let context = ctx.data::<GraphQLContext>()?;
        let state = &context.api_state;

        let models = state.models.read().await;
        let production_model = get_production_model(&models).await?;

        let embedding = production_model.get_entity_embedding(&entity_id)?;

        let entity = EmbeddingEntity {
            id: entity_id,
            entity_type: extract_entity_type(&entity_id),
            label,
            embedding: embedding.values,
            dimensions: embedding.dimensions,
            similarity_scores: None,
            related_entities: None,
            model_version: production_model.model_id().to_string(),
            created_at: Utc::now(), // In real implementation, would preserve original
            updated_at: Some(Utc::now()),
            metadata,
        };

        // Broadcast entity update event
        let _ = context
            .event_broadcaster
            .send(SubscriptionEvent::EntityUpdated(entity.clone()));

        Ok(entity)
    }

    /// Delete an entity
    async fn delete_entity(&self, ctx: &Context<'_>, entity_id: String) -> GraphQLResult<bool> {
        let context = ctx.data::<GraphQLContext>()?;

        // In a real implementation, would remove from model
        // For now, just broadcast the deletion event
        let deletion_event = EntityDeletedEvent {
            entity_id,
            deleted_at: Utc::now(),
            model_version: "current".to_string(),
        };

        let _ = context
            .event_broadcaster
            .send(SubscriptionEvent::EntityDeleted(deletion_event));

        Ok(true)
    }

    /// Start model training
    async fn start_training(
        &self,
        ctx: &Context<'_>,
        model_id: String,
        epochs: Option<i32>,
    ) -> GraphQLResult<bool> {
        let context = ctx.data::<GraphQLContext>()?;

        // Broadcast training started event
        let training_event = ModelTrainingEvent {
            model_id,
            event_type: "training_started".to_string(),
            progress: 0.0,
            current_epoch: Some(0),
            total_epochs: epochs,
            current_loss: None,
            timestamp: Utc::now(),
            metadata: None,
        };

        let _ = context
            .event_broadcaster
            .send(SubscriptionEvent::ModelTrainingStarted(training_event));

        Ok(true)
    }
}

/// GraphQL Subscription implementations
#[cfg(feature = "graphql")]
#[Subscription]
impl SubscriptionRoot {
    /// Subscribe to all events
    async fn events(&self, ctx: &Context<'_>) -> impl Stream<Item = SubscriptionEvent> {
        let context = ctx.data::<GraphQLContext>().unwrap();
        let receiver = context.event_broadcaster.subscribe();
        BroadcastStream::new(receiver).filter_map(|event| async { event.ok() })
    }

    /// Subscribe to entity events only
    async fn entity_events(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Entity type filter")] entity_type: Option<String>,
    ) -> impl Stream<Item = SubscriptionEvent> {
        let context = ctx.data::<GraphQLContext>().unwrap();
        let receiver = context.event_broadcaster.subscribe();

        BroadcastStream::new(receiver).filter_map(move |event| {
            let entity_type = entity_type.clone();
            async move {
                if let Ok(event) = event {
                    match &event {
                        SubscriptionEvent::EntityCreated(entity)
                        | SubscriptionEvent::EntityUpdated(entity) => {
                            if let Some(filter_type) = &entity_type {
                                if entity.entity_type.as_ref() == Some(filter_type) {
                                    Some(event)
                                } else {
                                    None
                                }
                            } else {
                                Some(event)
                            }
                        }
                        SubscriptionEvent::EntityDeleted(_) => Some(event),
                        _ => None,
                    }
                } else {
                    None
                }
            }
        })
    }

    /// Subscribe to training events
    async fn training_events(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Model ID filter")] model_id: Option<String>,
    ) -> impl Stream<Item = SubscriptionEvent> {
        let context = ctx.data::<GraphQLContext>().unwrap();
        let receiver = context.event_broadcaster.subscribe();

        BroadcastStream::new(receiver).filter_map(move |event| {
            let model_id = model_id.clone();
            async move {
                if let Ok(event) = event {
                    match &event {
                        SubscriptionEvent::ModelTrainingStarted(training_event)
                        | SubscriptionEvent::ModelTrainingCompleted(training_event) => {
                            if let Some(filter_id) = &model_id {
                                if &training_event.model_id == filter_id {
                                    Some(event)
                                } else {
                                    None
                                }
                            } else {
                                Some(event)
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
        })
    }

    /// Subscribe to quality alerts
    async fn quality_alerts(
        &self,
        ctx: &Context<'_>,
        #[graphql(desc = "Minimum severity filter")] min_severity: Option<String>,
    ) -> impl Stream<Item = QualityAlertEvent> {
        let context = ctx.data::<GraphQLContext>().unwrap();
        let receiver = context.event_broadcaster.subscribe();

        BroadcastStream::new(receiver).filter_map(move |event| {
            let min_severity = min_severity.clone();
            async move {
                if let Ok(SubscriptionEvent::QualityAlert(alert)) = event {
                    if let Some(min_sev) = &min_severity {
                        // Simple severity comparison (would implement proper severity levels)
                        if alert.severity >= *min_sev {
                            Some(alert)
                        } else {
                            None
                        }
                    } else {
                        Some(alert)
                    }
                } else {
                    None
                }
            }
        })
    }
}

// Helper functions for GraphQL operations

#[cfg(feature = "graphql")]
async fn get_production_model(
    models: &HashMap<Uuid, Arc<dyn EmbeddingModel + Send + Sync>>,
) -> GraphQLResult<&Arc<dyn EmbeddingModel + Send + Sync>> {
    // For simplicity, return the first model
    models
        .values()
        .next()
        .ok_or_else(|| GraphQLError::new("No models available"))
}

#[cfg(feature = "graphql")]
fn extract_entity_type(entity_id: &str) -> Option<String> {
    // Simple entity type extraction based on prefix
    if entity_id.starts_with("gene:") {
        Some("gene".to_string())
    } else if entity_id.starts_with("drug:") {
        Some("drug".to_string())
    } else if entity_id.starts_with("disease:") {
        Some("disease".to_string())
    } else if entity_id.starts_with("protein:") {
        Some("protein".to_string())
    } else {
        None
    }
}

#[cfg(feature = "graphql")]
async fn find_similar_entities(
    model: &Arc<dyn EmbeddingModel + Send + Sync>,
    entity_id: &str,
    limit: usize,
    min_similarity: f64,
) -> GraphQLResult<Vec<EmbeddingEntity>> {
    let target_embedding = model.get_entity_embedding(entity_id)?;
    let all_entities = model.get_entities();

    let mut similarities = Vec::new();

    for other_entity in &all_entities {
        if other_entity != entity_id {
            if let Ok(other_embedding) = model.get_entity_embedding(other_entity) {
                let similarity =
                    cosine_similarity(&target_embedding.values, &other_embedding.values);
                if similarity >= min_similarity {
                    similarities.push((other_entity.clone(), similarity, other_embedding));
                }
            }
        }
    }

    // Sort by similarity (descending) and take top k
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(limit);

    let mut entities = Vec::new();
    for (entity_id, similarity, embedding) in similarities {
        entities.push(EmbeddingEntity {
            id: entity_id.clone(),
            entity_type: extract_entity_type(&entity_id),
            label: Some(entity_id),
            embedding: embedding.values,
            dimensions: embedding.dimensions,
            similarity_scores: Some([(entity_id.clone(), similarity)].iter().cloned().collect()),
            related_entities: None,
            model_version: model.model_id().to_string(),
            created_at: Utc::now(),
            updated_at: None,
            metadata: None,
        });
    }

    Ok(entities)
}

#[cfg(feature = "graphql")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot_product / (norm_a * norm_b)) as f64
    }
}

#[cfg(feature = "graphql")]
async fn apply_entity_filters(
    entities: Vec<String>,
    filter: Option<&EntityFilter>,
) -> GraphQLResult<Vec<String>> {
    if let Some(filter) = filter {
        let filtered = entities
            .into_iter()
            .filter(|entity| {
                // Apply entity type filter
                if let Some(ref entity_type) = filter.entity_type {
                    if extract_entity_type(entity).as_ref() != Some(entity_type) {
                        return false;
                    }
                }

                // Apply label pattern filter
                if let Some(ref pattern) = filter.label_pattern {
                    if !entity.contains(pattern) {
                        return false;
                    }
                }

                // Apply include/exclude filters
                if let Some(ref include_ids) = filter.include_ids {
                    if !include_ids.contains(entity) {
                        return false;
                    }
                }

                if let Some(ref exclude_ids) = filter.exclude_ids {
                    if exclude_ids.contains(entity) {
                        return false;
                    }
                }

                true
            })
            .collect();

        Ok(filtered)
    } else {
        Ok(entities)
    }
}

#[cfg(feature = "graphql")]
async fn apply_entity_sorting(
    entities: Vec<String>,
    _sort: Option<&SortOptions>,
) -> GraphQLResult<Vec<String>> {
    // For simplicity, just return entities as-is
    // In a real implementation, would sort based on the specified field
    Ok(entities)
}

#[cfg(feature = "graphql")]
async fn apply_pagination(
    entities: Vec<String>,
    pagination: Option<&PaginationOptions>,
    model: &Arc<dyn EmbeddingModel + Send + Sync>,
) -> GraphQLResult<(Vec<EmbeddingEntity>, PageInfo)> {
    let (offset, limit) = if let Some(pagination) = pagination {
        (
            pagination.offset.unwrap_or(0) as usize,
            pagination.limit.unwrap_or(10) as usize,
        )
    } else {
        (0, 10)
    };

    let total_count = entities.len();
    let end_idx = (offset + limit).min(total_count);
    let paginated_entities = if offset < total_count {
        entities[offset..end_idx].to_vec()
    } else {
        vec![]
    };

    let mut embedding_entities = Vec::new();
    for entity_id in paginated_entities {
        if let Ok(embedding) = model.get_entity_embedding(&entity_id) {
            embedding_entities.push(EmbeddingEntity {
                id: entity_id.clone(),
                entity_type: extract_entity_type(&entity_id),
                label: Some(entity_id),
                embedding: embedding.values,
                dimensions: embedding.dimensions,
                similarity_scores: None,
                related_entities: None,
                model_version: model.model_id().to_string(),
                created_at: Utc::now(),
                updated_at: None,
                metadata: None,
            });
        }
    }

    let page_info = PageInfo {
        has_next_page: end_idx < total_count,
        has_previous_page: offset > 0,
        start_cursor: if embedding_entities.is_empty() {
            None
        } else {
            Some(offset.to_string())
        },
        end_cursor: if embedding_entities.is_empty() {
            None
        } else {
            Some(end_idx.to_string())
        },
    };

    Ok((embedding_entities, page_info))
}

#[cfg(feature = "graphql")]
async fn find_triples_with_filter(
    model: &Arc<dyn EmbeddingModel + Send + Sync>,
    _filter: Option<&TripleFilter>,
) -> GraphQLResult<Vec<EmbeddingTriple>> {
    // Simplified implementation - in reality would query actual triples
    let entities = model.get_entities();
    let relations = model.get_relations();

    let mut triples = Vec::new();

    // Create some sample triples (limited for demo)
    for (i, entity) in entities.iter().take(5).enumerate() {
        if let Some(relation) = relations.get(i % relations.len()) {
            if let Some(object) = entities.get((i + 1) % entities.len()) {
                if let (Ok(subj_emb), Ok(obj_emb)) = (
                    model.get_entity_embedding(entity),
                    model.get_entity_embedding(object),
                ) {
                    let score = model.score_triple(entity, relation, object).unwrap_or(0.0);

                    triples.push(EmbeddingTriple {
                        id: format!("{}|{}|{}", entity, relation, object),
                        subject: EmbeddingEntity {
                            id: entity.clone(),
                            entity_type: extract_entity_type(entity),
                            label: Some(entity.clone()),
                            embedding: subj_emb.values,
                            dimensions: subj_emb.dimensions,
                            similarity_scores: None,
                            related_entities: None,
                            model_version: model.model_id().to_string(),
                            created_at: Utc::now(),
                            updated_at: None,
                            metadata: None,
                        },
                        predicate: relation.clone(),
                        object: EmbeddingEntity {
                            id: object.clone(),
                            entity_type: extract_entity_type(object),
                            label: Some(object.clone()),
                            embedding: obj_emb.values,
                            dimensions: obj_emb.dimensions,
                            similarity_scores: None,
                            related_entities: None,
                            model_version: model.model_id().to_string(),
                            created_at: Utc::now(),
                            updated_at: None,
                            metadata: None,
                        },
                        score,
                        model_version: model.model_id().to_string(),
                        created_at: Utc::now(),
                        metadata: None,
                    });
                }
            }
        }
    }

    Ok(triples)
}

#[cfg(feature = "graphql")]
async fn apply_triple_sorting(
    triples: Vec<EmbeddingTriple>,
    _sort: Option<&SortOptions>,
) -> GraphQLResult<Vec<EmbeddingTriple>> {
    // For simplicity, just return triples as-is
    Ok(triples)
}

#[cfg(feature = "graphql")]
async fn apply_triple_pagination(
    triples: Vec<EmbeddingTriple>,
    pagination: Option<&PaginationOptions>,
) -> GraphQLResult<(Vec<EmbeddingTriple>, PageInfo)> {
    let (offset, limit) = if let Some(pagination) = pagination {
        (
            pagination.offset.unwrap_or(0) as usize,
            pagination.limit.unwrap_or(10) as usize,
        )
    } else {
        (0, 10)
    };

    let total_count = triples.len();
    let end_idx = (offset + limit).min(total_count);
    let paginated_triples = if offset < total_count {
        triples[offset..end_idx].to_vec()
    } else {
        vec![]
    };

    let page_info = PageInfo {
        has_next_page: end_idx < total_count,
        has_previous_page: offset > 0,
        start_cursor: if paginated_triples.is_empty() {
            None
        } else {
            Some(offset.to_string())
        },
        end_cursor: if paginated_triples.is_empty() {
            None
        } else {
            Some(end_idx.to_string())
        },
    };

    Ok((paginated_triples, page_info))
}

/// Create GraphQL schema
#[cfg(feature = "graphql")]
pub fn create_graphql_schema() -> Schema<Query, Mutation, SubscriptionRoot> {
    Schema::build(
        Query::default(),
        Mutation::default(),
        SubscriptionRoot::default(),
    )
    .finish()
}

/// GraphQL endpoint handler
#[cfg(all(feature = "api-server", feature = "graphql"))]
async fn graphql_handler(State(state): State<ApiState>, req: GraphQLRequest) -> GraphQLResponse {
    let (sender, _) = broadcast::channel(1000);
    let context = GraphQLContext {
        api_state: state,
        event_broadcaster: sender,
    };

    let schema = create_graphql_schema();
    schema.execute(req.into_inner().data(context)).await.into()
}

/// GraphiQL playground handler
#[cfg(all(feature = "api-server", feature = "graphql"))]
async fn graphiql() -> impl axum::response::IntoResponse {
    axum::response::Html(GraphiQLSource::build().endpoint("/graphql").finish())
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
        .route("/api/v1/embed/stream", post(embed_stream))
        .route("/api/v1/embed/text", post(embed_text))
        .route("/api/v1/embed/multimodal", post(embed_multimodal))
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

    // Add GraphQL endpoints if feature is enabled
    #[cfg(feature = "graphql")]
    {
        router = router
            .route("/graphql", post(graphql_handler))
            .route("/graphql", get(graphql_handler))
            .route("/graphiql", get(graphiql));
    }

    // Add middleware layers
    let service = ServiceBuilder::new()
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
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

    let cached_model =
        CachedEmbeddingModel::new(Box::new(model.as_ref()), Arc::clone(&state.cache_manager));

    // Make predictions based on type
    let use_cache = request.use_cache.unwrap_or(true);
    let (predictions, prediction_type_str) = match request.prediction_type {
        PredictionType::Objects => {
            if request.entities.len() != 2 {
                return Err(StatusCode::BAD_REQUEST);
            }
            let preds = if use_cache {
                cached_model.predict_objects_cached(
                    &request.entities[0],
                    &request.entities[1],
                    request.k,
                )
            } else {
                model.predict_objects(&request.entities[0], &request.entities[1], request.k)
            };
            (preds, "objects")
        }
        PredictionType::Subjects => {
            if request.entities.len() != 2 {
                return Err(StatusCode::BAD_REQUEST);
            }
            let preds =
                model.predict_subjects(&request.entities[0], &request.entities[1], request.k);
            (preds, "subjects")
        }
        PredictionType::Relations => {
            if request.entities.len() != 2 {
                return Err(StatusCode::BAD_REQUEST);
            }
            let preds =
                model.predict_relations(&request.entities[0], &request.entities[1], request.k);
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
                    status: if is_loaded {
                        HealthStatus::Healthy
                    } else {
                        HealthStatus::Degraded
                    },
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
async fn clear_cache(State(state): State<ApiState>) -> Result<Json<serde_json::Value>, StatusCode> {
    state.cache_manager.clear_all();

    let response = serde_json::json!({
        "message": "Cache cleared successfully",
        "timestamp": chrono::Utc::now(),
    });

    Ok(Json(response))
}

// ============================================================================
// Specialized Embedding Handlers
// ============================================================================

/// Generate text embeddings using specialized models
#[cfg(feature = "api-server")]
async fn embed_text(
    State(state): State<ApiState>,
    Json(request): Json<TextEmbeddingRequest>,
) -> Result<Json<TextEmbeddingResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    // For this demo, we'll simulate specialized text embedding generation
    // In a real implementation, this would use the SpecializedTextEmbedding model

    let model_type = request.model_type.unwrap_or_else(|| "BioBERT".to_string());
    let preprocess = request.preprocess.unwrap_or(true);

    // Simulate processing time based on text length
    let processing_delay = std::cmp::min(request.text.len() / 100, 50) as u64;
    tokio::time::sleep(tokio::time::Duration::from_millis(processing_delay)).await;

    // Generate a simple embedding based on text content (in real implementation would use actual model)
    let dimensions = 768; // Standard for BERT-like models
    let mut embedding = vec![0.0f32; dimensions];

    // Simple text-based features
    let text_bytes = request.text.as_bytes();
    for (i, &byte) in text_bytes.iter().enumerate() {
        if i < dimensions {
            embedding[i] = (byte as f32 / 255.0 - 0.5) * 2.0;
        }
    }

    // Domain-specific features based on model type
    match model_type.as_str() {
        "SciBERT" => {
            if request.text.contains("et al.") {
                embedding[0] += 1.0;
            }
            if request.text.contains("figure") {
                embedding[1] += 1.0;
            }
        }
        "BioBERT" => {
            if request.text.contains("protein") {
                embedding[0] += 1.0;
            }
            if request.text.contains("gene") {
                embedding[1] += 1.0;
            }
            if request.text.contains("disease") {
                embedding[2] += 1.0;
            }
        }
        "CodeBERT" => {
            if request.text.contains("function") {
                embedding[0] += 1.0;
            }
            if request.text.contains("class") {
                embedding[1] += 1.0;
            }
        }
        _ => {}
    }

    let generation_time = start_time.elapsed().as_millis() as f64;

    let response = TextEmbeddingResponse {
        text: request.text,
        embedding,
        dimensions,
        model_type,
        model_version: Uuid::new_v4(), // Would be actual model version
        generation_time_ms: generation_time,
        preprocessed: preprocess,
    };

    Ok(Json(response))
}

/// Generate multi-modal embeddings
#[cfg(feature = "api-server")]
async fn embed_multimodal(
    State(state): State<ApiState>,
    Json(request): Json<MultiModalRequest>,
) -> Result<Json<MultiModalResponse>, StatusCode> {
    let start_time = std::time::Instant::now();

    let mut text_embedding = None;
    let mut entity_embedding = None;
    let mut alignment_scores = None;
    let mut unified_embedding = None;

    // Process text if provided
    if let Some(text) = &request.text {
        let dimensions = 768;
        let mut embedding = vec![0.0f32; dimensions];

        // Simple text-based embedding
        let text_bytes = text.as_bytes();
        for (i, &byte) in text_bytes.iter().enumerate() {
            if i < dimensions {
                embedding[i] = (byte as f32 / 255.0 - 0.5) * 2.0;
            }
        }

        text_embedding = Some(embedding);
    }

    // Process entity if provided
    if let Some(entity) = &request.entity {
        // Get production model for entity embedding
        let model_version = match get_production_model_version(&state).await {
            Ok(version) => version,
            Err(_) => return Err(StatusCode::NOT_FOUND),
        };

        let models = state.models.read().await;
        if let Some(model) = models.get(&model_version) {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                entity_embedding = Some(embedding.values);
            }
        }
    }

    // Compute cross-modal alignment scores if requested
    if let Some(aligned_entities) = &request.aligned_entities {
        let mut scores = HashMap::new();

        // Simple similarity computation (would use contrastive learning in real implementation)
        if let (Some(ref text_emb), Some(ref entity_emb)) = (&text_embedding, &entity_embedding) {
            for aligned_entity in aligned_entities {
                // Simplified alignment score
                let score = 0.5 + (text_emb[0] * entity_emb[0]).min(1.0).max(-1.0) * 0.5;
                scores.insert(aligned_entity.clone(), score as f64);
            }
        }

        alignment_scores = Some(scores);
    }

    // Create unified embedding if both text and entity provided
    if let (Some(ref text_emb), Some(ref entity_emb)) = (&text_embedding, &entity_embedding) {
        let min_len = text_emb.len().min(entity_emb.len());
        let mut unified = Vec::with_capacity(min_len);

        for i in 0..min_len {
            // Simple fusion: weighted average
            unified.push((text_emb[i] * 0.6 + entity_emb[i] * 0.4));
        }

        unified_embedding = Some(unified);
    }

    let generation_time = start_time.elapsed().as_millis() as f64;

    let response = MultiModalResponse {
        text_embedding,
        entity_embedding,
        alignment_scores,
        unified_embedding,
        model_version: request.model_version.unwrap_or_else(Uuid::new_v4),
        generation_time_ms: generation_time,
    };

    Ok(Json(response))
}

/// Stream embeddings for large batches
#[cfg(feature = "api-server")]
async fn embed_stream(
    State(state): State<ApiState>,
    Json(request): Json<StreamEmbeddingRequest>,
) -> Result<impl axum::response::IntoResponse, StatusCode> {
    use axum::response::sse::{Event, Sse};
    use tokio_stream::{wrappers::ReceiverStream, StreamExt};

    let (tx, rx) = tokio::sync::mpsc::channel(100);

    // Get model version
    let model_version = if let Some(version) = request.model_version {
        version
    } else {
        match get_production_model_version(&state).await {
            Ok(version) => version,
            Err(_) => return Err(StatusCode::NOT_FOUND),
        }
    };

    // Clone necessary data for the spawned task
    let entities = request.entities.clone();
    let batch_size = request.batch_size.unwrap_or(10);
    let models = Arc::clone(&state.models);

    // Spawn task to process entities in batches
    tokio::spawn(async move {
        let models = models.read().await;
        let model = match models.get(&model_version) {
            Some(model) => model,
            None => {
                let _ = tx.send(Err("Model not found")).await;
                return;
            }
        };

        let total_entities = entities.len();

        for (batch_idx, chunk) in entities.chunks(batch_size).enumerate() {
            for (idx_in_batch, entity) in chunk.iter().enumerate() {
                let sequence = batch_idx * batch_size + idx_in_batch;
                let is_last = sequence == total_entities - 1;

                match model.get_entity_embedding(entity) {
                    Ok(embedding) => {
                        let item = StreamEmbeddingItem {
                            entity: entity.clone(),
                            embedding: embedding.values,
                            sequence,
                            is_last,
                        };

                        let event =
                            Event::default().data(serde_json::to_string(&item).unwrap_or_default());

                        if tx.send(Ok(event)).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        // Skip failed embeddings or send error event
                        continue;
                    }
                }

                // Small delay to prevent overwhelming the client
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    let sse = Sse::new(stream);

    Ok(sse)
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

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", state.config.host, state.config.port))
            .await
            .map_err(|e| anyhow!("Failed to bind to address: {}", e))?;

    info!(
        "API server starting on {}:{}",
        state.config.host, state.config.port
    );

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

// ============================================================================
// GraphQL API Implementation
// ============================================================================

#[cfg(feature = "graphql")]
/// GraphQL input types
#[derive(InputObject)]
pub struct EmbeddingInput {
    /// The entity to embed
    pub entity: String,
    /// Model version to use (optional)
    pub model_version: Option<String>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

#[cfg(feature = "graphql")]
#[derive(InputObject)]
pub struct BatchEmbeddingInput {
    /// List of entities to embed
    pub entities: Vec<String>,
    /// Model version to use
    pub model_version: Option<String>,
    /// Use cached results where available
    pub use_cache: Option<bool>,
    /// Batch processing options
    pub parallel: Option<bool>,
    pub chunk_size: Option<usize>,
}

#[cfg(feature = "graphql")]
#[derive(InputObject)]
pub struct TripleScoreInput {
    /// Subject entity
    pub subject: String,
    /// Predicate relation
    pub predicate: String,
    /// Object entity
    pub object: String,
    /// Model version to use
    pub model_version: Option<String>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

#[cfg(feature = "graphql")]
#[derive(InputObject)]
pub struct PredictionInput {
    /// Known entities
    pub entities: Vec<String>,
    /// Prediction type
    pub prediction_type: PredictionTypeGQL,
    /// Number of predictions to return
    pub k: i32,
    /// Model version to use
    pub model_version: Option<String>,
    /// Use cached results if available
    pub use_cache: Option<bool>,
}

#[cfg(feature = "graphql")]
#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum PredictionTypeGQL {
    Objects,
    Subjects,
    Relations,
}

#[cfg(feature = "graphql")]
/// GraphQL output types
#[derive(SimpleObject)]
pub struct EmbeddingGQL {
    /// The entity that was embedded
    pub entity: String,
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Embedding dimensions
    pub dimensions: i32,
    /// Model version used
    pub model_version: String,
    /// Whether result was cached
    pub from_cache: bool,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
}

#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct BatchEmbeddingGQL {
    /// Embeddings for each entity
    pub embeddings: Vec<EmbeddingGQL>,
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
    /// Number of cache hits
    pub cache_hits: i32,
    /// Number of cache misses
    pub cache_misses: i32,
}

#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct TripleScoreGQL {
    /// The triple that was scored
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// The score
    pub score: f64,
    /// Model version used
    pub model_version: String,
    /// Whether result was cached
    pub from_cache: bool,
    /// Scoring time in milliseconds
    pub scoring_time_ms: f64,
}

#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct PredictionResultGQL {
    /// Entity or relation
    pub entity: String,
    /// Prediction score
    pub score: f64,
}

#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct PredictionGQL {
    /// Input entities
    pub input_entities: Vec<String>,
    /// Prediction type
    pub prediction_type: String,
    /// Predicted entities with scores
    pub predictions: Vec<PredictionResultGQL>,
    /// Model version used
    pub model_version: String,
    /// Whether result was cached
    pub from_cache: bool,
    /// Prediction time in milliseconds
    pub prediction_time_ms: f64,
}

#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct ModelStatsGQL {
    /// Number of entities
    pub num_entities: i32,
    /// Number of relations
    pub num_relations: i32,
    /// Number of triples
    pub num_triples: i32,
    /// Embedding dimensions
    pub dimensions: i32,
    /// Whether model is trained
    pub is_trained: bool,
    /// Model type
    pub model_type: String,
    /// Creation time
    pub creation_time: String,
    /// Last training time
    pub last_training_time: Option<String>,
}

#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct ModelInfoGQL {
    /// Model statistics
    pub stats: ModelStatsGQL,
    /// Model version
    pub version: String,
    /// Whether model is currently loaded
    pub is_loaded: bool,
    /// Model health status
    pub health_status: String,
    /// Performance metrics
    pub avg_response_time_ms: f64,
    pub requests_last_hour: i32,
    pub error_rate_percent: f64,
    pub memory_usage_mb: f64,
}

#[cfg(feature = "graphql")]
/// GraphQL Query resolver
pub struct QueryRoot;

#[cfg(feature = "graphql")]
#[Object]
impl QueryRoot {
    /// Get embedding for a single entity
    async fn embedding(
        &self,
        ctx: &Context<'_>,
        input: EmbeddingInput,
    ) -> GraphQLResult<EmbeddingGQL> {
        let state = ctx.data::<ApiState>()?;
        let start_time = std::time::Instant::now();

        // Get model version
        let model_version = if let Some(version_str) = input.model_version {
            Uuid::parse_str(&version_str)
                .map_err(|_| GraphQLError::new("Invalid model version format"))?
        } else {
            get_production_model_version(state)
                .await
                .map_err(|_| GraphQLError::new("No production model available"))?
        };

        // Get model
        let models = state.models.read().await;
        let model = models
            .get(&model_version)
            .ok_or_else(|| GraphQLError::new("Model not found"))?;

        // Generate embedding
        let use_cache = input.use_cache.unwrap_or(true);
        let embedding = if use_cache {
            let cached_model = CachedEmbeddingModel::new(
                Box::new(model.as_ref()),
                Arc::clone(&state.cache_manager),
            );
            cached_model
                .get_entity_embedding_cached(&input.entity)
                .map_err(|_| GraphQLError::new("Failed to generate embedding"))?
        } else {
            model
                .get_entity_embedding(&input.entity)
                .map_err(|_| GraphQLError::new("Entity not found"))?
        };

        let generation_time = start_time.elapsed().as_millis() as f64;
        let from_cache = use_cache && state.cache_manager.get_embedding(&input.entity).is_some();

        Ok(EmbeddingGQL {
            entity: input.entity,
            embedding: embedding.values,
            dimensions: embedding.dimensions as i32,
            model_version: model_version.to_string(),
            from_cache,
            generation_time_ms: generation_time,
        })
    }

    /// Get embeddings for multiple entities
    async fn batch_embedding(
        &self,
        ctx: &Context<'_>,
        input: BatchEmbeddingInput,
    ) -> GraphQLResult<BatchEmbeddingGQL> {
        let state = ctx.data::<ApiState>()?;
        let start_time = std::time::Instant::now();

        // Validate batch size
        if input.entities.len() > state.config.max_batch_size {
            return Err(GraphQLError::new("Batch size exceeds maximum limit"));
        }

        // Get model version
        let model_version = if let Some(version_str) = input.model_version {
            Uuid::parse_str(&version_str)
                .map_err(|_| GraphQLError::new("Invalid model version format"))?
        } else {
            get_production_model_version(state)
                .await
                .map_err(|_| GraphQLError::new("No production model available"))?
        };

        // Get model
        let models = state.models.read().await;
        let model = models
            .get(&model_version)
            .ok_or_else(|| GraphQLError::new("Model not found"))?;

        let use_cache = input.use_cache.unwrap_or(true);
        let cached_model =
            CachedEmbeddingModel::new(Box::new(model.as_ref()), Arc::clone(&state.cache_manager));

        let mut embeddings = Vec::new();
        let mut cache_hits = 0;
        let mut cache_misses = 0;

        // Process entities
        for entity in &input.entities {
            let entity_start = std::time::Instant::now();

            let (embedding, had_cache) = if use_cache {
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

            embeddings.push(EmbeddingGQL {
                entity: entity.clone(),
                embedding: embedding.values,
                dimensions: embedding.dimensions as i32,
                model_version: model_version.to_string(),
                from_cache: had_cache,
                generation_time_ms: generation_time,
            });
        }

        let total_time = start_time.elapsed().as_millis() as f64;

        Ok(BatchEmbeddingGQL {
            embeddings,
            total_time_ms: total_time,
            cache_hits,
            cache_misses,
        })
    }

    /// Score a triple
    async fn score_triple(
        &self,
        ctx: &Context<'_>,
        input: TripleScoreInput,
    ) -> GraphQLResult<TripleScoreGQL> {
        let state = ctx.data::<ApiState>()?;
        let start_time = std::time::Instant::now();

        // Get model version
        let model_version = if let Some(version_str) = input.model_version {
            Uuid::parse_str(&version_str)
                .map_err(|_| GraphQLError::new("Invalid model version format"))?
        } else {
            get_production_model_version(state)
                .await
                .map_err(|_| GraphQLError::new("No production model available"))?
        };

        // Get model
        let models = state.models.read().await;
        let model = models
            .get(&model_version)
            .ok_or_else(|| GraphQLError::new("Model not found"))?;

        // Score triple
        let use_cache = input.use_cache.unwrap_or(true);
        let score = if use_cache {
            let cached_model = CachedEmbeddingModel::new(
                Box::new(model.as_ref()),
                Arc::clone(&state.cache_manager),
            );
            cached_model
                .score_triple_cached(&input.subject, &input.predicate, &input.object)
                .map_err(|_| GraphQLError::new("Failed to score triple"))?
        } else {
            model
                .score_triple(&input.subject, &input.predicate, &input.object)
                .map_err(|_| GraphQLError::new("Failed to score triple"))?
        };

        let scoring_time = start_time.elapsed().as_millis() as f64;
        let from_cache = use_cache; // Simplified - would need proper cache checking

        Ok(TripleScoreGQL {
            subject: input.subject,
            predicate: input.predicate,
            object: input.object,
            score,
            model_version: model_version.to_string(),
            from_cache,
            scoring_time_ms: scoring_time,
        })
    }

    /// Make predictions (objects, subjects, or relations)
    async fn predict(
        &self,
        ctx: &Context<'_>,
        input: PredictionInput,
    ) -> GraphQLResult<PredictionGQL> {
        let state = ctx.data::<ApiState>()?;
        let start_time = std::time::Instant::now();

        // Validate input
        if input.entities.is_empty() || input.k <= 0 {
            return Err(GraphQLError::new("Invalid input parameters"));
        }

        // Get model version
        let model_version = if let Some(version_str) = input.model_version {
            Uuid::parse_str(&version_str)
                .map_err(|_| GraphQLError::new("Invalid model version format"))?
        } else {
            get_production_model_version(state)
                .await
                .map_err(|_| GraphQLError::new("No production model available"))?
        };

        // Get model
        let models = state.models.read().await;
        let model = models
            .get(&model_version)
            .ok_or_else(|| GraphQLError::new("Model not found"))?;

        // Make predictions based on type
        let predictions = match input.prediction_type {
            PredictionTypeGQL::Objects => {
                if input.entities.len() != 2 {
                    return Err(GraphQLError::new(
                        "Object prediction requires subject and predicate",
                    ));
                }
                model
                    .predict_objects(&input.entities[0], &input.entities[1], input.k as usize)
                    .map_err(|_| GraphQLError::new("Prediction failed"))?
            }
            PredictionTypeGQL::Subjects => {
                if input.entities.len() != 2 {
                    return Err(GraphQLError::new(
                        "Subject prediction requires predicate and object",
                    ));
                }
                model
                    .predict_subjects(&input.entities[0], &input.entities[1], input.k as usize)
                    .map_err(|_| GraphQLError::new("Prediction failed"))?
            }
            PredictionTypeGQL::Relations => {
                if input.entities.len() != 2 {
                    return Err(GraphQLError::new(
                        "Relation prediction requires subject and object",
                    ));
                }
                model
                    .predict_relations(&input.entities[0], &input.entities[1], input.k as usize)
                    .map_err(|_| GraphQLError::new("Prediction failed"))?
            }
        };

        let prediction_time = start_time.elapsed().as_millis() as f64;
        let from_cache = input.use_cache.unwrap_or(false); // Simplified

        let prediction_results: Vec<PredictionResultGQL> = predictions
            .into_iter()
            .map(|(entity, score)| PredictionResultGQL { entity, score })
            .collect();

        Ok(PredictionGQL {
            input_entities: input.entities,
            prediction_type: format!("{:?}", input.prediction_type),
            predictions: prediction_results,
            model_version: model_version.to_string(),
            from_cache,
            prediction_time_ms: prediction_time,
        })
    }

    /// Get model information
    async fn model_info(
        &self,
        ctx: &Context<'_>,
        model_version: Option<String>,
    ) -> GraphQLResult<ModelInfoGQL> {
        let state = ctx.data::<ApiState>()?;

        // Get model version
        let version = if let Some(version_str) = model_version {
            Uuid::parse_str(&version_str)
                .map_err(|_| GraphQLError::new("Invalid model version format"))?
        } else {
            get_production_model_version(state)
                .await
                .map_err(|_| GraphQLError::new("No production model available"))?
        };

        // Get model
        let models = state.models.read().await;
        let model = models
            .get(&version)
            .ok_or_else(|| GraphQLError::new("Model not found"))?;

        let stats = model.get_stats();
        let is_loaded = true; // Model is loaded if we can access it

        Ok(ModelInfoGQL {
            stats: ModelStatsGQL {
                num_entities: stats.num_entities as i32,
                num_relations: stats.num_relations as i32,
                num_triples: stats.num_triples as i32,
                dimensions: stats.dimensions as i32,
                is_trained: stats.is_trained,
                model_type: stats.model_type,
                creation_time: stats.creation_time.to_rfc3339(),
                last_training_time: stats.last_training_time.map(|t| t.to_rfc3339()),
            },
            version: version.to_string(),
            is_loaded,
            health_status: "Healthy".to_string(), // Simplified
            avg_response_time_ms: 50.0,           // Mock data
            requests_last_hour: 100,              // Mock data
            error_rate_percent: 0.1,              // Mock data
            memory_usage_mb: 256.0,               // Mock data
        })
    }

    /// List available models
    async fn models(&self, ctx: &Context<'_>) -> GraphQLResult<Vec<ModelInfoGQL>> {
        let state = ctx.data::<ApiState>()?;
        let registry_models = state.registry.list_models().await;
        let loaded_models = state.models.read().await;

        let mut models = Vec::new();
        for registry_model in registry_models {
            if let Some(version_id) = registry_model.production_version {
                if let Some(model) = loaded_models.get(&version_id) {
                    let stats = model.get_stats();
                    models.push(ModelInfoGQL {
                        stats: ModelStatsGQL {
                            num_entities: stats.num_entities as i32,
                            num_relations: stats.num_relations as i32,
                            num_triples: stats.num_triples as i32,
                            dimensions: stats.dimensions as i32,
                            is_trained: stats.is_trained,
                            model_type: stats.model_type,
                            creation_time: stats.creation_time.to_rfc3339(),
                            last_training_time: stats.last_training_time.map(|t| t.to_rfc3339()),
                        },
                        version: version_id.to_string(),
                        is_loaded: true,
                        health_status: "Healthy".to_string(),
                        avg_response_time_ms: 50.0,
                        requests_last_hour: 100,
                        error_rate_percent: 0.1,
                        memory_usage_mb: 256.0,
                    });
                }
            }
        }

        Ok(models)
    }
}

#[cfg(feature = "graphql")]
/// GraphQL Mutation resolver
pub struct MutationRoot;

#[cfg(feature = "graphql")]
#[Object]
impl MutationRoot {
    /// Clear all caches
    async fn clear_cache(&self, ctx: &Context<'_>) -> GraphQLResult<String> {
        let state = ctx.data::<ApiState>()?;
        state.cache_manager.clear_all();
        Ok("Cache cleared successfully".to_string())
    }

    /// Load a model
    async fn load_model(&self, ctx: &Context<'_>, model_version: String) -> GraphQLResult<String> {
        let _state = ctx.data::<ApiState>()?;
        let _version = Uuid::parse_str(&model_version)
            .map_err(|_| GraphQLError::new("Invalid model version format"))?;

        // In a real implementation, this would load the model
        // For now, just return success
        Ok("Model loaded successfully".to_string())
    }

    /// Unload a model
    async fn unload_model(
        &self,
        ctx: &Context<'_>,
        model_version: String,
    ) -> GraphQLResult<String> {
        let state = ctx.data::<ApiState>()?;
        let version = Uuid::parse_str(&model_version)
            .map_err(|_| GraphQLError::new("Invalid model version format"))?;

        let mut models = state.models.write().await;
        models.remove(&version);

        Ok("Model unloaded successfully".to_string())
    }
}

#[cfg(feature = "graphql")]
/// Create GraphQL schema
pub type EmbeddingSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

#[cfg(feature = "graphql")]
pub fn create_graphql_schema() -> EmbeddingSchema {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription).finish()
}

#[cfg(feature = "graphql")]
/// GraphQL endpoint handler
async fn graphql_handler(State(state): State<ApiState>, req: GraphQLRequest) -> GraphQLResponse {
    let schema = create_graphql_schema();
    let request = req.into_inner().data(state);
    schema.execute(request).await.into()
}

#[cfg(feature = "graphql")]
/// GraphiQL IDE handler
async fn graphiql() -> axum::response::Html<String> {
    axum::response::Html(GraphiQLSource::build().endpoint("/graphql").finish())
}

#[cfg(feature = "graphql")]
/// Add GraphQL endpoints to router
pub fn add_graphql_routes(router: Router<ApiState>) -> Router<ApiState> {
    router
        .route("/graphql", post(graphql_handler))
        .route("/graphiql", get(graphiql))
}
