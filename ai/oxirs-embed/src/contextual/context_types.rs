//! Context types and configurations for contextual embeddings

use serde::{Deserialize, Serialize};
use crate::ModelConfig;

/// Contextual embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualConfig {
    pub base_config: ModelConfig,
    /// Base embedding dimension
    pub base_dim: usize,
    /// Context-aware output dimension
    pub context_dim: usize,
    /// Context types to consider
    pub context_types: Vec<ContextType>,
    /// Adaptation strategies
    pub adaptation_strategy: AdaptationStrategy,
    /// Context fusion method
    pub fusion_method: ContextFusionMethod,
    /// Temporal context settings
    pub temporal_config: TemporalConfig,
    /// Interactive refinement settings
    pub interactive_config: InteractiveConfig,
    /// Context cache settings
    pub cache_config: ContextCacheConfig,
}

impl Default for ContextualConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            base_dim: 768,
            context_dim: 512,
            context_types: vec![
                ContextType::Query,
                ContextType::User,
                ContextType::Task,
            ],
            adaptation_strategy: AdaptationStrategy::DynamicAttention,
            fusion_method: ContextFusionMethod::MultiHeadAttention,
            temporal_config: TemporalConfig::default(),
            interactive_config: InteractiveConfig::default(),
            cache_config: ContextCacheConfig::default(),
        }
    }
}

/// Types of context for embedding adaptation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ContextType {
    /// Query-specific context from the current request
    Query,
    /// User-specific context from history and preferences
    User,
    /// Task-specific context for domain adaptation
    Task,
    /// Temporal context for time-aware embeddings
    Temporal,
    /// Interactive context from user feedback
    Interactive,
    /// Domain-specific context
    Domain,
    /// Cross-lingual context
    CrossLingual,
}

/// Adaptation strategies for contextual embeddings
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum AdaptationStrategy {
    /// Dynamic attention over context vectors
    DynamicAttention,
    /// Context-conditional layer normalization
    ConditionalLayerNorm,
    /// Adapter layers for context-specific transformation
    AdapterLayers,
    /// Meta-learning for few-shot adaptation
    MetaLearning,
    /// Prompt-based context integration
    PromptBased,
    /// Memory-augmented adaptation
    MemoryAugmented,
}

/// Context fusion methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextFusionMethod {
    /// Simple concatenation of contexts
    Concatenation,
    /// Weighted sum of context vectors
    WeightedSum,
    /// Multi-head attention fusion
    MultiHeadAttention,
    /// Cross-modal transformer fusion
    TransformerFusion,
    /// Graph-based context fusion
    GraphFusion,
    /// Neural module network fusion
    NeuralModules,
}

/// Temporal context configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Enable temporal adaptation
    pub enabled: bool,
    /// Time window for context (hours)
    pub time_window_hours: f64,
    /// Temporal decay factor
    pub decay_factor: f32,
    /// Enable trend analysis
    pub trend_analysis: bool,
    /// Seasonal adjustment
    pub seasonal_adjustment: bool,
    /// Maximum temporal history
    pub max_history_entries: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            time_window_hours: 24.0,
            decay_factor: 0.9,
            trend_analysis: true,
            seasonal_adjustment: false,
            max_history_entries: 1000,
        }
    }
}

/// Interactive refinement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveConfig {
    /// Enable interactive refinement
    pub enabled: bool,
    /// Learning rate for feedback integration
    pub feedback_learning_rate: f32,
    /// Maximum feedback history
    pub max_feedback_history: usize,
    /// Feedback aggregation method
    pub aggregation_method: FeedbackAggregation,
    /// Enable online learning
    pub online_learning: bool,
    /// Confidence threshold for adaptation
    pub confidence_threshold: f32,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            feedback_learning_rate: 0.01,
            max_feedback_history: 500,
            aggregation_method: FeedbackAggregation::WeightedAverage,
            online_learning: false,
            confidence_threshold: 0.7,
        }
    }
}

/// Context cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCacheConfig {
    /// Enable context caching
    pub enabled: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    /// Enable LRU eviction
    pub lru_eviction: bool,
    /// Cache hit ratio threshold for warnings
    pub hit_ratio_threshold: f32,
}

impl Default for ContextCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cache_size: 10000,
            ttl_seconds: 3600,
            lru_eviction: true,
            hit_ratio_threshold: 0.8,
        }
    }
}

/// Feedback aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackAggregation {
    /// Simple average of feedback scores
    Average,
    /// Weighted average with recency weighting
    WeightedAverage,
    /// Exponential moving average
    ExponentialMoving,
    /// Confidence-weighted aggregation
    ConfidenceWeighted,
}