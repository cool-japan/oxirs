//! Request and response types for the API
//!
//! This module contains all the data structures used for API requests and responses.

use crate::{ModelStats, TrainingStats, Vector};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Basic embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Entity ID to get embedding for
    pub entity_id: String,
    /// Optional model ID to use
    pub model_id: Option<Uuid>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

/// Basic embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Entity ID
    pub entity_id: String,
    /// Generated embedding vector
    pub embedding: Vector,
    /// Model ID used
    pub model_id: Uuid,
    /// Whether result came from cache
    pub from_cache: bool,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
}

/// Batch embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingRequest {
    /// List of entity IDs
    pub entity_ids: Vec<String>,
    /// Optional model ID to use
    pub model_id: Option<Uuid>,
    /// Batch processing options
    pub options: Option<BatchOptions>,
}

/// Batch processing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOptions {
    /// Use cached results if available
    pub use_cache: Option<bool>,
    /// Parallel processing batch size
    pub batch_size: Option<usize>,
}

/// Batch embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingResponse {
    /// Embedding results
    pub embeddings: Vec<EmbeddingResponse>,
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Model ID used
    pub model_id: Uuid,
}

/// Text embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbeddingRequest {
    /// Text to embed
    pub text: String,
    /// Optional text type hint
    pub text_type: Option<String>,
    /// Optional model ID to use
    pub model_id: Option<Uuid>,
    /// Language hint (ISO 639-1)
    pub language: Option<String>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

/// Text embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbeddingResponse {
    /// Original text
    pub text: String,
    /// Generated embedding vector
    pub embedding: Vector,
    /// Detected language (if applicable)
    pub detected_language: Option<String>,
    /// Model ID used
    pub model_id: Uuid,
    /// Whether result came from cache
    pub from_cache: bool,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
}

/// Multi-modal embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalRequest {
    /// Text content
    pub text: Option<String>,
    /// Knowledge graph entities
    pub entities: Option<Vec<String>>,
    /// Optional model ID to use
    pub model_id: Option<Uuid>,
    /// Fusion strategy
    pub fusion_strategy: Option<String>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

/// Multi-modal embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalResponse {
    /// Generated unified embedding
    pub embedding: Vector,
    /// Individual component embeddings
    pub component_embeddings: HashMap<String, Vector>,
    /// Fusion strategy used
    pub fusion_strategy: String,
    /// Model ID used
    pub model_id: Uuid,
    /// Whether result came from cache
    pub from_cache: bool,
    /// Generation time in milliseconds
    pub generation_time_ms: f64,
}

/// Stream embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEmbeddingRequest {
    /// Stream of items to embed
    pub items: Vec<StreamEmbeddingItem>,
    /// Optional model ID to use
    pub model_id: Option<Uuid>,
    /// Streaming options
    pub options: Option<BatchOptions>,
}

/// Individual item in streaming request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEmbeddingItem {
    /// Item ID
    pub id: String,
    /// Item content (text or entity ID)
    pub content: String,
    /// Content type
    pub content_type: String,
}

/// Triple scoring request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleScoreRequest {
    /// Subject entity
    pub subject: String,
    /// Predicate relation
    pub predicate: String,
    /// Object entity
    pub object: String,
    /// Optional model ID to use
    pub model_id: Option<Uuid>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

/// Triple scoring response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleScoreResponse {
    /// Subject entity
    pub subject: String,
    /// Predicate relation
    pub predicate: String,
    /// Object entity
    pub object: String,
    /// Plausibility score
    pub score: f64,
    /// Model ID used
    pub model_id: Uuid,
    /// Whether result came from cache
    pub from_cache: bool,
    /// Computation time in milliseconds
    pub computation_time_ms: f64,
}

/// Prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    /// Input entities
    pub entities: Vec<String>,
    /// Prediction type
    pub prediction_type: PredictionType,
    /// Number of predictions to return
    pub top_k: Option<usize>,
    /// Optional model ID to use
    pub model_id: Option<Uuid>,
    /// Use cached result if available
    pub use_cache: Option<bool>,
}

/// Types of predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionType {
    /// Predict missing objects
    Objects { subject: String, predicate: String },
    /// Predict missing subjects
    Subjects { predicate: String, object: String },
    /// Predict missing relations
    Relations { subject: String, object: String },
}

/// Prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResponse {
    /// Input entities
    pub input: Vec<String>,
    /// Prediction type
    pub prediction_type: String,
    /// Predictions with scores
    pub predictions: Vec<(String, f64)>,
    /// Model version used
    pub model_version: String,
    /// Whether result came from cache
    pub from_cache: bool,
    /// Prediction time in milliseconds
    pub prediction_time_ms: f64,
}

/// Model information request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfoRequest {
    /// Optional specific model ID
    pub model_id: Option<Uuid>,
}

/// Model information response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfoResponse {
    /// Model statistics
    pub stats: ModelStats,
    /// Model health status
    pub health: ModelHealth,
    /// Available operations
    pub capabilities: Vec<String>,
    /// Last training statistics
    pub last_training: Option<TrainingStats>,
}

/// Model health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealth {
    /// Current health status
    pub status: HealthStatus,
    /// Last health check timestamp
    pub last_check: DateTime<Utc>,
    /// Performance metrics
    pub metrics: HealthMetrics,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Model is healthy and operational
    Healthy,
    /// Model is operational but with degraded performance
    Degraded,
    /// Model is not operational
    Unhealthy,
}

/// Performance metrics for health checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Number of requests in the last hour
    pub requests_last_hour: u64,
    /// Error rate percentage
    pub error_rate_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
}

/// Query parameters for API endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    /// Optional limit for results
    pub limit: Option<usize>,
    /// Optional offset for pagination
    pub offset: Option<usize>,
    /// Optional model ID filter
    pub model_id: Option<Uuid>,
    /// Optional format specification
    pub format: Option<String>,
    /// Include detailed information
    pub detailed: Option<bool>,
}
