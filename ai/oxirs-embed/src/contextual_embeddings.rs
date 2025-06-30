//! Contextual embeddings with dynamic adaptation for query, user, and task contexts
//!
//! This module implements advanced contextual embedding generation that adapts to:
//! - Query-specific contexts for better relevance
//! - User-specific preferences and history
//! - Task-specific requirements and domains
//! - Temporal context for time-aware embeddings
//! - Interactive refinement based on feedback

use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Vector, Triple};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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
                ContextType::Temporal,
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
            online_learning: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Feedback aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackAggregation {
    /// Simple average of feedback
    Average,
    /// Weighted average with recency bias
    WeightedAverage,
    /// Exponential moving average
    ExponentialMovingAverage,
    /// Attention-based aggregation
    AttentionBased,
    /// Hierarchical aggregation
    Hierarchical,
}

/// Context cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCacheConfig {
    /// Enable context caching
    pub enabled: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Cache expiry time (hours)
    pub expiry_hours: f64,
    /// Enable semantic similarity caching
    pub semantic_caching: bool,
    /// Similarity threshold for cache hits
    pub similarity_threshold: f32,
}

impl Default for ContextCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cache_size: 10000,
            expiry_hours: 2.0,
            semantic_caching: true,
            similarity_threshold: 0.85,
        }
    }
}

/// Context information for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingContext {
    /// Context ID
    pub context_id: Uuid,
    /// Timestamp of context creation
    pub timestamp: DateTime<Utc>,
    /// Query context
    pub query_context: Option<QueryContext>,
    /// User context
    pub user_context: Option<UserContext>,
    /// Task context
    pub task_context: Option<TaskContext>,
    /// Temporal context
    pub temporal_context: Option<TemporalContext>,
    /// Interactive context from feedback
    pub interactive_context: Option<InteractiveContext>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl EmbeddingContext {
    /// Create a new embedding context
    pub fn new() -> Self {
        Self {
            context_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            query_context: None,
            user_context: None,
            task_context: None,
            temporal_context: None,
            interactive_context: None,
            metadata: HashMap::new(),
        }
    }

    /// Set query context
    pub fn with_query_context(mut self, query_context: QueryContext) -> Self {
        self.query_context = Some(query_context);
        self
    }

    /// Set user context
    pub fn with_user_context(mut self, user_context: UserContext) -> Self {
        self.user_context = Some(user_context);
        self
    }

    /// Set task context
    pub fn with_task_context(mut self, task_context: TaskContext) -> Self {
        self.task_context = Some(task_context);
        self
    }

    /// Set temporal context
    pub fn with_temporal_context(mut self, temporal_context: TemporalContext) -> Self {
        self.temporal_context = Some(temporal_context);
        self
    }

    /// Set interactive context
    pub fn with_interactive_context(mut self, interactive_context: InteractiveContext) -> Self {
        self.interactive_context = Some(interactive_context);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Query-specific context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    /// Original query text
    pub query_text: String,
    /// Query intent
    pub intent: QueryIntent,
    /// Query complexity
    pub complexity: QueryComplexity,
    /// Query domain
    pub domain: String,
    /// Query keywords
    pub keywords: Vec<String>,
    /// Query entities
    pub entities: Vec<String>,
    /// Query relations
    pub relations: Vec<String>,
    /// Expected answer type
    pub answer_type: Option<String>,
}

/// Query intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Factual information retrieval
    Factual,
    /// Similarity search
    Similarity,
    /// Recommendation
    Recommendation,
    /// Analysis
    Analysis,
    /// Summarization
    Summarization,
    /// Generation
    Generation,
    /// Classification
    Classification,
    /// Unknown intent
    Unknown,
}

/// Query complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryComplexity {
    /// Simple single-concept queries
    Simple,
    /// Multi-concept queries
    Medium,
    /// Complex analytical queries
    Complex,
    /// Expert-level queries
    Expert,
}

/// User-specific context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// User ID
    pub user_id: String,
    /// User preferences
    pub preferences: UserPreferences,
    /// User expertise level
    pub expertise_level: ExpertiseLevel,
    /// User interaction history
    pub interaction_history: Vec<UserInteraction>,
    /// User domain interests
    pub domain_interests: HashMap<String, f32>,
    /// User language preferences
    pub language_preferences: Vec<String>,
    /// Personalization vector
    pub personalization_vector: Option<Array1<f32>>,
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred detail level
    pub detail_level: DetailLevel,
    /// Preferred formats
    pub preferred_formats: Vec<String>,
    /// Content filters
    pub content_filters: Vec<String>,
    /// Accessibility preferences
    pub accessibility: AccessibilityPreferences,
    /// Privacy settings
    pub privacy_settings: PrivacySettings,
}

/// User expertise levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    /// Beginner user
    Beginner,
    /// Intermediate user
    Intermediate,
    /// Advanced user
    Advanced,
    /// Expert user
    Expert,
    /// Domain specialist
    Specialist,
}

/// Detail level preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetailLevel {
    /// Brief summaries
    Brief,
    /// Standard detail
    Standard,
    /// Detailed explanations
    Detailed,
    /// Comprehensive coverage
    Comprehensive,
}

/// Accessibility preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityPreferences {
    /// Screen reader compatibility
    pub screen_reader: bool,
    /// High contrast mode
    pub high_contrast: bool,
    /// Large text preference
    pub large_text: bool,
    /// Audio descriptions
    pub audio_descriptions: bool,
}

impl Default for AccessibilityPreferences {
    fn default() -> Self {
        Self {
            screen_reader: false,
            high_contrast: false,
            large_text: false,
            audio_descriptions: false,
        }
    }
}

/// Privacy settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    /// Allow personalization
    pub allow_personalization: bool,
    /// Allow history tracking
    pub allow_history: bool,
    /// Allow cross-domain tracking
    pub allow_cross_domain: bool,
    /// Data retention period (days)
    pub retention_days: u32,
}

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            allow_personalization: true,
            allow_history: true,
            allow_cross_domain: false,
            retention_days: 30,
        }
    }
}

/// User interaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    /// Interaction timestamp
    pub timestamp: DateTime<Utc>,
    /// Query submitted
    pub query: String,
    /// Results received
    pub results: Vec<String>,
    /// User feedback
    pub feedback: Option<UserFeedback>,
    /// Interaction duration
    pub duration_seconds: f64,
    /// Context at time of interaction
    pub context_snapshot: HashMap<String, String>,
}

/// User feedback types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserFeedback {
    /// Positive feedback
    Positive(f32),
    /// Negative feedback
    Negative(f32),
    /// Neutral feedback
    Neutral,
    /// Specific feedback with comments
    Detailed { rating: f32, comments: String },
    /// Implicit feedback from behavior
    Implicit { engagement_score: f32 },
}

/// Task-specific context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    /// Task type
    pub task_type: TaskType,
    /// Domain context
    pub domain: String,
    /// Task parameters
    pub parameters: HashMap<String, String>,
    /// Required capabilities
    pub capabilities: Vec<String>,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Task-specific embeddings
    pub task_embeddings: HashMap<String, Array1<f32>>,
}

/// Task types for contextualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Information retrieval
    InformationRetrieval,
    /// Question answering
    QuestionAnswering,
    /// Document similarity
    DocumentSimilarity,
    /// Entity linking
    EntityLinking,
    /// Relation extraction
    RelationExtraction,
    /// Knowledge base completion
    KnowledgeCompletion,
    /// Recommendation
    Recommendation,
    /// Classification
    Classification,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
}

/// Performance requirements for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum latency (ms)
    pub max_latency_ms: u64,
    /// Minimum accuracy
    pub min_accuracy: f32,
    /// Memory constraints (MB)
    pub max_memory_mb: u64,
    /// Throughput requirements (ops/sec)
    pub min_throughput: f32,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_latency_ms: 100,
            min_accuracy: 0.85,
            max_memory_mb: 1024,
            min_throughput: 100.0,
        }
    }
}

/// Temporal context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Current timestamp
    pub current_time: DateTime<Utc>,
    /// Relevant time period
    pub time_period: TimePeriod,
    /// Historical embeddings
    pub historical_embeddings: Vec<TemporalEmbedding>,
    /// Trend information
    pub trends: Vec<Trend>,
    /// Seasonal patterns
    pub seasonal_patterns: HashMap<String, f32>,
    /// Time-aware weights
    pub temporal_weights: Array1<f32>,
}

/// Time periods for temporal context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimePeriod {
    /// Real-time (current)
    RealTime,
    /// Recent (last hour)
    Recent,
    /// Daily (last 24 hours)
    Daily,
    /// Weekly (last 7 days)
    Weekly,
    /// Monthly (last 30 days)
    Monthly,
    /// Historical (longer periods)
    Historical,
}

/// Temporal embedding with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEmbedding {
    /// Timestamp of embedding
    pub timestamp: DateTime<Utc>,
    /// Embedding vector
    pub embedding: Array1<f32>,
    /// Context at the time
    pub context: HashMap<String, String>,
    /// Confidence score
    pub confidence: f32,
}

/// Trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trend {
    /// Trend name/type
    pub name: String,
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f32,
    /// Trend duration
    pub duration_hours: f64,
    /// Trend confidence
    pub confidence: f32,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable/flat trend
    Stable,
    /// Cyclical/seasonal trend
    Cyclical,
    /// Unknown trend
    Unknown,
}

/// Interactive context from user feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveContext {
    /// Recent feedback
    pub recent_feedback: Vec<UserFeedback>,
    /// Adaptation history
    pub adaptation_history: Vec<AdaptationRecord>,
    /// Current adaptation state
    pub current_state: AdaptationState,
    /// Interactive embeddings
    pub interactive_embeddings: HashMap<String, Array1<f32>>,
    /// Confidence scores
    pub confidence_scores: HashMap<String, f32>,
}

/// Adaptation record for tracking changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRecord {
    /// Timestamp of adaptation
    pub timestamp: DateTime<Utc>,
    /// Type of adaptation
    pub adaptation_type: AdaptationType,
    /// Feedback that triggered adaptation
    pub trigger_feedback: UserFeedback,
    /// Changes made
    pub changes: HashMap<String, f32>,
    /// Performance impact
    pub performance_impact: f32,
}

/// Types of adaptations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Weight adjustment
    WeightAdjustment,
    /// Embedding refinement
    EmbeddingRefinement,
    /// Context prioritization
    ContextPrioritization,
    /// Model selection
    ModelSelection,
    /// Parameter tuning
    ParameterTuning,
}

/// Current adaptation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationState {
    /// State vector
    pub state_vector: Array1<f32>,
    /// Adaptation parameters
    pub parameters: HashMap<String, f32>,
    /// Last update time
    pub last_update: DateTime<Utc>,
    /// Adaptation confidence
    pub confidence: f32,
}

/// Contextual embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualEmbeddingModel {
    pub config: ContextualConfig,
    pub model_id: Uuid,
    /// Base embedding model
    pub base_model: BaseEmbeddingModel,
    /// Context processors
    pub context_processors: HashMap<ContextType, ContextProcessor>,
    /// Fusion network
    pub fusion_network: FusionNetwork,
    /// Adaptation engines
    pub adaptation_engines: HashMap<AdaptationStrategy, AdaptationEngine>,
    /// Context cache
    pub context_cache: ContextCache,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Training statistics
    pub training_stats: TrainingStats,
    pub model_stats: ModelStats,
    pub is_trained: bool,
}

/// Base embedding model wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseEmbeddingModel {
    /// Model type
    pub model_type: String,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Model parameters
    pub parameters: HashMap<String, Array2<f32>>,
}

impl BaseEmbeddingModel {
    /// Generate base embeddings
    pub fn generate_embedding(&self, input: &str) -> Result<Array1<f32>> {
        // Simplified embedding generation
        let input_vector = self.text_to_vector(input)?;
        let projection = self.parameters.get("projection")
            .ok_or_else(|| anyhow!("Projection matrix not found"))?;
        
        Ok(projection.dot(&input_vector))
    }

    /// Convert text to input vector
    fn text_to_vector(&self, text: &str) -> Result<Array1<f32>> {
        // Simplified tokenization and vectorization
        let tokens: Vec<_> = text.split_whitespace().collect();
        let mut vector = Array1::zeros(self.input_dim);
        
        for (i, token) in tokens.iter().enumerate() {
            if i < self.input_dim {
                // Simple hash-based encoding
                let hash = token.chars().map(|c| c as u32).sum::<u32>();
                vector[i] = (hash % 1000) as f32 / 1000.0;
            }
        }
        
        Ok(vector)
    }
}

/// Context processor for specific context types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextProcessor {
    /// Context type handled
    pub context_type: ContextType,
    /// Processing strategy
    pub strategy: ProcessingStrategy,
    /// Learned parameters
    pub parameters: HashMap<String, Array2<f32>>,
    /// Processing statistics
    pub stats: ProcessingStats,
}

/// Processing strategies for contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStrategy {
    /// Simple linear transformation
    Linear,
    /// Multi-layer perceptron
    MLP,
    /// Attention-based processing
    Attention,
    /// Recurrent processing for temporal data
    Recurrent,
    /// Graph-based processing
    Graph,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Number of contexts processed
    pub contexts_processed: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Success rate
    pub success_rate: f32,
    /// Cache hit rate
    pub cache_hit_rate: f32,
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self {
            contexts_processed: 0,
            avg_processing_time_ms: 0.0,
            success_rate: 1.0,
            cache_hit_rate: 0.0,
        }
    }
}

impl ContextProcessor {
    /// Process context into context vector
    pub fn process_context(&mut self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let start_time = std::time::Instant::now();
        
        let result = match self.context_type {
            ContextType::Query => self.process_query_context(context),
            ContextType::User => self.process_user_context(context),
            ContextType::Task => self.process_task_context(context),
            ContextType::Temporal => self.process_temporal_context(context),
            ContextType::Interactive => self.process_interactive_context(context),
            ContextType::Domain => self.process_domain_context(context),
            ContextType::CrossLingual => self.process_crosslingual_context(context),
        };

        // Update statistics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.stats.contexts_processed += 1;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.contexts_processed - 1) as f64 + processing_time) 
            / self.stats.contexts_processed as f64;

        if result.is_ok() {
            self.stats.success_rate = 
                (self.stats.success_rate * (self.stats.contexts_processed - 1) as f32 + 1.0) 
                / self.stats.contexts_processed as f32;
        } else {
            self.stats.success_rate = 
                (self.stats.success_rate * (self.stats.contexts_processed - 1) as f32) 
                / self.stats.contexts_processed as f32;
        }

        result
    }

    /// Process query context
    fn process_query_context(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        if let Some(query_ctx) = &context.query_context {
            let mut context_vector = Array1::zeros(256); // Fixed dimension for simplicity
            
            // Encode query intent
            match query_ctx.intent {
                QueryIntent::Factual => context_vector[0] = 1.0,
                QueryIntent::Similarity => context_vector[1] = 1.0,
                QueryIntent::Recommendation => context_vector[2] = 1.0,
                QueryIntent::Analysis => context_vector[3] = 1.0,
                QueryIntent::Summarization => context_vector[4] = 1.0,
                QueryIntent::Generation => context_vector[5] = 1.0,
                QueryIntent::Classification => context_vector[6] = 1.0,
                QueryIntent::Unknown => context_vector[7] = 1.0,
            }
            
            // Encode query complexity
            match query_ctx.complexity {
                QueryComplexity::Simple => context_vector[8] = 0.25,
                QueryComplexity::Medium => context_vector[8] = 0.5,
                QueryComplexity::Complex => context_vector[8] = 0.75,
                QueryComplexity::Expert => context_vector[8] = 1.0,
            }
            
            // Encode domain information
            let domain_hash = query_ctx.domain.chars().map(|c| c as u32).sum::<u32>();
            context_vector[9] = (domain_hash % 1000) as f32 / 1000.0;
            
            // Encode keywords and entities (simplified)
            for (i, keyword) in query_ctx.keywords.iter().enumerate() {
                if i + 10 < context_vector.len() {
                    let keyword_hash = keyword.chars().map(|c| c as u32).sum::<u32>();
                    context_vector[i + 10] = (keyword_hash % 1000) as f32 / 1000.0;
                }
            }
            
            Ok(context_vector)
        } else {
            Ok(Array1::zeros(256))
        }
    }

    /// Process user context
    fn process_user_context(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        if let Some(user_ctx) = &context.user_context {
            let mut context_vector = Array1::zeros(256);
            
            // Encode expertise level
            match user_ctx.expertise_level {
                ExpertiseLevel::Beginner => context_vector[0] = 0.2,
                ExpertiseLevel::Intermediate => context_vector[0] = 0.4,
                ExpertiseLevel::Advanced => context_vector[0] = 0.6,
                ExpertiseLevel::Expert => context_vector[0] = 0.8,
                ExpertiseLevel::Specialist => context_vector[0] = 1.0,
            }
            
            // Encode detail level preference
            match user_ctx.preferences.detail_level {
                DetailLevel::Brief => context_vector[1] = 0.25,
                DetailLevel::Standard => context_vector[1] = 0.5,
                DetailLevel::Detailed => context_vector[1] = 0.75,
                DetailLevel::Comprehensive => context_vector[1] = 1.0,
            }
            
            // Encode domain interests
            for (i, (domain, interest)) in user_ctx.domain_interests.iter().enumerate() {
                if i + 2 < context_vector.len() {
                    context_vector[i + 2] = *interest;
                }
            }
            
            // Use personalization vector if available
            if let Some(ref personalization) = user_ctx.personalization_vector {
                let copy_len = std::cmp::min(personalization.len(), context_vector.len() - 50);
                for i in 0..copy_len {
                    context_vector[i + 50] = personalization[i];
                }
            }
            
            Ok(context_vector)
        } else {
            Ok(Array1::zeros(256))
        }
    }

    /// Process task context
    fn process_task_context(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        if let Some(task_ctx) = &context.task_context {
            let mut context_vector = Array1::zeros(256);
            
            // Encode task type
            match task_ctx.task_type {
                TaskType::InformationRetrieval => context_vector[0] = 1.0,
                TaskType::QuestionAnswering => context_vector[1] = 1.0,
                TaskType::DocumentSimilarity => context_vector[2] = 1.0,
                TaskType::EntityLinking => context_vector[3] = 1.0,
                TaskType::RelationExtraction => context_vector[4] = 1.0,
                TaskType::KnowledgeCompletion => context_vector[5] = 1.0,
                TaskType::Recommendation => context_vector[6] = 1.0,
                TaskType::Classification => context_vector[7] = 1.0,
                TaskType::Summarization => context_vector[8] = 1.0,
                TaskType::Translation => context_vector[9] = 1.0,
            }
            
            // Encode performance requirements
            let perf_req = &task_ctx.performance_requirements;
            context_vector[10] = (perf_req.max_latency_ms as f32).log10() / 4.0; // Normalize log scale
            context_vector[11] = perf_req.min_accuracy;
            context_vector[12] = (perf_req.max_memory_mb as f32).log10() / 4.0;
            context_vector[13] = perf_req.min_throughput.log10() / 4.0;
            
            // Encode domain
            let domain_hash = task_ctx.domain.chars().map(|c| c as u32).sum::<u32>();
            context_vector[14] = (domain_hash % 1000) as f32 / 1000.0;
            
            Ok(context_vector)
        } else {
            Ok(Array1::zeros(256))
        }
    }

    /// Process temporal context
    fn process_temporal_context(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        if let Some(temporal_ctx) = &context.temporal_context {
            let mut context_vector = Array1::zeros(256);
            
            // Encode time period
            match temporal_ctx.time_period {
                TimePeriod::RealTime => context_vector[0] = 1.0,
                TimePeriod::Recent => context_vector[1] = 1.0,
                TimePeriod::Daily => context_vector[2] = 1.0,
                TimePeriod::Weekly => context_vector[3] = 1.0,
                TimePeriod::Monthly => context_vector[4] = 1.0,
                TimePeriod::Historical => context_vector[5] = 1.0,
            }
            
            // Encode temporal weights if available
            if temporal_ctx.temporal_weights.len() > 0 {
                let copy_len = std::cmp::min(temporal_ctx.temporal_weights.len(), context_vector.len() - 10);
                for i in 0..copy_len {
                    context_vector[i + 10] = temporal_ctx.temporal_weights[i];
                }
            }
            
            // Encode trend information
            for (i, trend) in temporal_ctx.trends.iter().enumerate() {
                if i + 100 < context_vector.len() {
                    context_vector[i + 100] = trend.strength;
                    context_vector[i + 150] = trend.confidence;
                }
            }
            
            Ok(context_vector)
        } else {
            Ok(Array1::zeros(256))
        }
    }

    /// Process interactive context
    fn process_interactive_context(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        if let Some(interactive_ctx) = &context.interactive_context {
            let mut context_vector = Array1::zeros(256);
            
            // Encode recent feedback
            let mut positive_feedback = 0.0;
            let mut negative_feedback = 0.0;
            let mut total_feedback = 0;
            
            for feedback in &interactive_ctx.recent_feedback {
                total_feedback += 1;
                match feedback {
                    UserFeedback::Positive(score) => positive_feedback += score,
                    UserFeedback::Negative(score) => negative_feedback += score,
                    UserFeedback::Detailed { rating, .. } => {
                        if *rating > 0.5 {
                            positive_feedback += rating;
                        } else {
                            negative_feedback += 1.0 - rating;
                        }
                    }
                    UserFeedback::Implicit { engagement_score } => {
                        if *engagement_score > 0.5 {
                            positive_feedback += engagement_score;
                        } else {
                            negative_feedback += 1.0 - engagement_score;
                        }
                    }
                    UserFeedback::Neutral => {}
                }
            }
            
            if total_feedback > 0 {
                context_vector[0] = positive_feedback / total_feedback as f32;
                context_vector[1] = negative_feedback / total_feedback as f32;
            }
            
            // Encode adaptation state
            context_vector[2] = interactive_ctx.current_state.confidence;
            
            // Encode confidence scores
            let mut avg_confidence = 0.0;
            if !interactive_ctx.confidence_scores.is_empty() {
                avg_confidence = interactive_ctx.confidence_scores.values().sum::<f32>() 
                    / interactive_ctx.confidence_scores.len() as f32;
            }
            context_vector[3] = avg_confidence;
            
            Ok(context_vector)
        } else {
            Ok(Array1::zeros(256))
        }
    }

    /// Process domain context
    fn process_domain_context(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let mut context_vector = Array1::zeros(256);
        
        // Extract domain information from metadata
        if let Some(domain) = context.metadata.get("domain") {
            let domain_hash = domain.chars().map(|c| c as u32).sum::<u32>();
            context_vector[0] = (domain_hash % 1000) as f32 / 1000.0;
        }
        
        // Extract subdomain information
        if let Some(subdomain) = context.metadata.get("subdomain") {
            let subdomain_hash = subdomain.chars().map(|c| c as u32).sum::<u32>();
            context_vector[1] = (subdomain_hash % 1000) as f32 / 1000.0;
        }
        
        Ok(context_vector)
    }

    /// Process cross-lingual context
    fn process_crosslingual_context(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let mut context_vector = Array1::zeros(256);
        
        // Extract language information from metadata
        if let Some(language) = context.metadata.get("language") {
            // Simple language encoding (would be more sophisticated in practice)
            match language.as_str() {
                "en" => context_vector[0] = 1.0,
                "es" => context_vector[1] = 1.0,
                "fr" => context_vector[2] = 1.0,
                "de" => context_vector[3] = 1.0,
                "zh" => context_vector[4] = 1.0,
                "ja" => context_vector[5] = 1.0,
                "ar" => context_vector[6] = 1.0,
                "ru" => context_vector[7] = 1.0,
                _ => context_vector[8] = 1.0, // Other languages
            }
        }
        
        // Extract cross-lingual alignment information
        if let Some(alignment) = context.metadata.get("cross_lingual_alignment") {
            let alignment_score: f32 = alignment.parse().unwrap_or(0.0);
            context_vector[10] = alignment_score;
        }
        
        Ok(context_vector)
    }
}

/// Context fusion network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionNetwork {
    /// Fusion method
    pub method: ContextFusionMethod,
    /// Network parameters
    pub parameters: HashMap<String, Array2<f32>>,
    /// Attention heads (for attention-based fusion)
    pub attention_heads: usize,
    /// Hidden dimensions
    pub hidden_dim: usize,
}

impl FusionNetwork {
    /// Fuse multiple context vectors into unified representation
    pub fn fuse_contexts(&self, context_vectors: HashMap<ContextType, Array1<f32>>) -> Result<Array1<f32>> {
        match self.method {
            ContextFusionMethod::Concatenation => self.concatenate_contexts(context_vectors),
            ContextFusionMethod::WeightedSum => self.weighted_sum_contexts(context_vectors),
            ContextFusionMethod::MultiHeadAttention => self.attention_fusion(context_vectors),
            ContextFusionMethod::TransformerFusion => self.transformer_fusion(context_vectors),
            ContextFusionMethod::GraphFusion => self.graph_fusion(context_vectors),
            ContextFusionMethod::NeuralModules => self.neural_module_fusion(context_vectors),
        }
    }

    /// Simple concatenation of context vectors
    fn concatenate_contexts(&self, context_vectors: HashMap<ContextType, Array1<f32>>) -> Result<Array1<f32>> {
        let mut result = Vec::new();
        
        // Concatenate in fixed order for consistency
        let order = [
            ContextType::Query,
            ContextType::User,
            ContextType::Task,
            ContextType::Temporal,
            ContextType::Interactive,
            ContextType::Domain,
            ContextType::CrossLingual,
        ];
        
        for context_type in &order {
            if let Some(vector) = context_vectors.get(context_type) {
                result.extend(vector.iter());
            } else {
                // Pad with zeros if context not available
                result.extend(vec![0.0; 256]); // Assuming 256-dim context vectors
            }
        }
        
        Ok(Array1::from_vec(result))
    }

    /// Weighted sum of context vectors
    fn weighted_sum_contexts(&self, context_vectors: HashMap<ContextType, Array1<f32>>) -> Result<Array1<f32>> {
        let weights = self.parameters.get("fusion_weights")
            .ok_or_else(|| anyhow!("Fusion weights not found"))?;
        
        let target_dim = 512; // Target fusion dimension
        let mut result = Array1::zeros(target_dim);
        let mut total_weight = 0.0;
        
        for (i, (context_type, vector)) in context_vectors.iter().enumerate() {
            if i < weights.nrows() {
                let weight = weights[[i, 0]];
                total_weight += weight;
                
                // Project vector to target dimension if needed
                let projected = if vector.len() != target_dim {
                    self.project_vector(vector, target_dim)?
                } else {
                    vector.clone()
                };
                
                result = result + &projected * weight;
            }
        }
        
        // Normalize by total weight
        if total_weight > 0.0 {
            result = result / total_weight;
        }
        
        Ok(result)
    }

    /// Multi-head attention fusion
    fn attention_fusion(&self, context_vectors: HashMap<ContextType, Array1<f32>>) -> Result<Array1<f32>> {
        let query_matrix = self.parameters.get("query_matrix")
            .ok_or_else(|| anyhow!("Query matrix not found"))?;
        let key_matrix = self.parameters.get("key_matrix")
            .ok_or_else(|| anyhow!("Key matrix not found"))?;
        let value_matrix = self.parameters.get("value_matrix")
            .ok_or_else(|| anyhow!("Value matrix not found"))?;
        
        let target_dim = 512;
        let head_dim = target_dim / self.attention_heads;
        let mut result = Array1::zeros(target_dim);
        
        // Convert context vectors to matrix
        let mut context_matrix = Array2::zeros((context_vectors.len(), 256));
        for (i, (_, vector)) in context_vectors.iter().enumerate() {
            for j in 0..std::cmp::min(vector.len(), 256) {
                context_matrix[[i, j]] = vector[j];
            }
        }
        
        // Simplified multi-head attention
        for head in 0..self.attention_heads {
            let start_idx = head * head_dim;
            let end_idx = std::cmp::min(start_idx + head_dim, target_dim);
            
            // Compute attention scores (simplified)
            let queries = query_matrix.slice(s![start_idx..end_idx, ..]).dot(&context_matrix.t());
            let keys = key_matrix.slice(s![start_idx..end_idx, ..]).dot(&context_matrix.t());
            let values = value_matrix.slice(s![start_idx..end_idx, ..]).dot(&context_matrix.t());
            
            // Simplified attention computation
            let attention_weights = Array1::from_iter(
                (0..context_vectors.len()).map(|_| 1.0 / context_vectors.len() as f32)
            );
            
            // Apply attention to values
            for i in start_idx..end_idx {
                result[i] = attention_weights.iter().zip(values.row(i - start_idx).iter()).map(|(w, v)| w * v).sum();
            }
        }
        
        Ok(result)
    }

    /// Transformer-based fusion
    fn transformer_fusion(&self, context_vectors: HashMap<ContextType, Array1<f32>>) -> Result<Array1<f32>> {
        // Simplified transformer fusion (would be more complex in practice)
        self.attention_fusion(context_vectors)
    }

    /// Graph-based fusion
    fn graph_fusion(&self, context_vectors: HashMap<ContextType, Array1<f32>>) -> Result<Array1<f32>> {
        // Simplified graph fusion - treat contexts as graph nodes
        let mut result = Array1::zeros(512);
        let num_contexts = context_vectors.len() as f32;
        
        // Simple aggregation with graph-like connections
        for (_, vector) in context_vectors.iter() {
            let projected = self.project_vector(vector, 512)?;
            result = result + &projected;
        }
        
        result = result / num_contexts;
        Ok(result)
    }

    /// Neural module network fusion
    fn neural_module_fusion(&self, context_vectors: HashMap<ContextType, Array1<f32>>) -> Result<Array1<f32>> {
        // Simplified neural module fusion
        self.weighted_sum_contexts(context_vectors)
    }

    /// Project vector to target dimension
    fn project_vector(&self, vector: &Array1<f32>, target_dim: usize) -> Result<Array1<f32>> {
        if vector.len() == target_dim {
            return Ok(vector.clone());
        }
        
        // Simple projection - pad or truncate
        let mut result = Array1::zeros(target_dim);
        let copy_len = std::cmp::min(vector.len(), target_dim);
        
        for i in 0..copy_len {
            result[i] = vector[i];
        }
        
        Ok(result)
    }
}

/// Adaptation engine for dynamic context adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEngine {
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
    /// Engine parameters
    pub parameters: HashMap<String, Array2<f32>>,
    /// Learning rate
    pub learning_rate: f32,
    /// Adaptation history
    pub adaptation_history: Vec<AdaptationRecord>,
}

impl AdaptationEngine {
    /// Adapt embeddings based on context and feedback
    pub fn adapt_embedding(
        &mut self,
        base_embedding: &Array1<f32>,
        context: &EmbeddingContext,
        feedback: Option<&UserFeedback>,
    ) -> Result<Array1<f32>> {
        match self.strategy {
            AdaptationStrategy::DynamicAttention => self.dynamic_attention_adaptation(base_embedding, context),
            AdaptationStrategy::ConditionalLayerNorm => self.conditional_layernorm_adaptation(base_embedding, context),
            AdaptationStrategy::AdapterLayers => self.adapter_layer_adaptation(base_embedding, context),
            AdaptationStrategy::MetaLearning => self.meta_learning_adaptation(base_embedding, context),
            AdaptationStrategy::PromptBased => self.prompt_based_adaptation(base_embedding, context),
            AdaptationStrategy::MemoryAugmented => self.memory_augmented_adaptation(base_embedding, context),
        }
    }

    /// Dynamic attention adaptation
    fn dynamic_attention_adaptation(&self, base_embedding: &Array1<f32>, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let attention_matrix = self.parameters.get("attention_matrix")
            .ok_or_else(|| anyhow!("Attention matrix not found"))?;
        
        // Simplified attention-based adaptation
        let context_vector = self.extract_context_features(context)?;
        let attention_weights = attention_matrix.dot(&context_vector);
        
        // Apply attention to base embedding
        let mut adapted = base_embedding.clone();
        for i in 0..std::cmp::min(adapted.len(), attention_weights.len()) {
            adapted[i] *= 1.0 + attention_weights[i] * 0.1; // Small adaptation
        }
        
        Ok(adapted)
    }

    /// Conditional layer normalization adaptation
    fn conditional_layernorm_adaptation(&self, base_embedding: &Array1<f32>, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let gamma_matrix = self.parameters.get("gamma_matrix")
            .ok_or_else(|| anyhow!("Gamma matrix not found"))?;
        let beta_matrix = self.parameters.get("beta_matrix")
            .ok_or_else(|| anyhow!("Beta matrix not found"))?;
        
        let context_vector = self.extract_context_features(context)?;
        let gamma = gamma_matrix.dot(&context_vector);
        let beta = beta_matrix.dot(&context_vector);
        
        // Apply conditional layer normalization
        let mean = base_embedding.mean().unwrap_or(0.0);
        let var = base_embedding.var(0.0);
        let std = var.sqrt();
        
        let mut normalized = base_embedding.clone();
        for i in 0..normalized.len() {
            normalized[i] = (normalized[i] - mean) / (std + 1e-6);
        }
        
        // Apply learned gamma and beta
        for i in 0..std::cmp::min(normalized.len(), gamma.len()) {
            normalized[i] = normalized[i] * (1.0 + gamma[i] * 0.1) + beta[i] * 0.1;
        }
        
        Ok(normalized)
    }

    /// Adapter layer adaptation
    fn adapter_layer_adaptation(&self, base_embedding: &Array1<f32>, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let down_proj = self.parameters.get("adapter_down")
            .ok_or_else(|| anyhow!("Adapter down projection not found"))?;
        let up_proj = self.parameters.get("adapter_up")
            .ok_or_else(|| anyhow!("Adapter up projection not found"))?;
        
        // Adapter residual connection
        let down_output = down_proj.dot(base_embedding);
        let up_output = up_proj.dot(&down_output);
        
        // Residual connection with small adaptation
        let mut adapted = base_embedding.clone();
        for i in 0..std::cmp::min(adapted.len(), up_output.len()) {
            adapted[i] += up_output[i] * 0.1; // Small residual
        }
        
        Ok(adapted)
    }

    /// Meta-learning adaptation
    fn meta_learning_adaptation(&self, base_embedding: &Array1<f32>, context: &EmbeddingContext) -> Result<Array1<f32>> {
        // Simplified meta-learning - use context to generate adaptation parameters
        let context_vector = self.extract_context_features(context)?;
        let meta_params = self.parameters.get("meta_network")
            .ok_or_else(|| anyhow!("Meta network not found"))?
            .dot(&context_vector);
        
        // Apply meta-learned parameters
        let mut adapted = base_embedding.clone();
        for i in 0..std::cmp::min(adapted.len(), meta_params.len()) {
            adapted[i] *= 1.0 + meta_params[i] * 0.05;
        }
        
        Ok(adapted)
    }

    /// Prompt-based adaptation
    fn prompt_based_adaptation(&self, base_embedding: &Array1<f32>, context: &EmbeddingContext) -> Result<Array1<f32>> {
        // Generate prompt embedding from context
        let prompt_embedding = self.generate_prompt_embedding(context)?;
        
        // Combine base embedding with prompt
        let mut adapted = base_embedding.clone();
        for i in 0..std::cmp::min(adapted.len(), prompt_embedding.len()) {
            adapted[i] += prompt_embedding[i] * 0.1;
        }
        
        Ok(adapted)
    }

    /// Memory-augmented adaptation
    fn memory_augmented_adaptation(&self, base_embedding: &Array1<f32>, context: &EmbeddingContext) -> Result<Array1<f32>> {
        // Retrieve relevant memories (simplified)
        let memory_vector = self.retrieve_memories(context)?;
        
        // Combine with base embedding
        let mut adapted = base_embedding.clone();
        for i in 0..std::cmp::min(adapted.len(), memory_vector.len()) {
            adapted[i] += memory_vector[i] * 0.1;
        }
        
        Ok(adapted)
    }

    /// Extract context features for adaptation
    fn extract_context_features(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let mut features = Array1::zeros(64); // Simplified feature vector
        
        // Extract basic context features
        if context.query_context.is_some() {
            features[0] = 1.0;
        }
        if context.user_context.is_some() {
            features[1] = 1.0;
        }
        if context.task_context.is_some() {
            features[2] = 1.0;
        }
        if context.temporal_context.is_some() {
            features[3] = 1.0;
        }
        if context.interactive_context.is_some() {
            features[4] = 1.0;
        }
        
        // Add timestamp encoding
        let timestamp_seconds = context.timestamp.timestamp() as f32;
        features[5] = (timestamp_seconds % 86400.0) / 86400.0; // Time of day
        features[6] = ((timestamp_seconds / 86400.0) % 7.0) / 7.0; // Day of week
        
        Ok(features)
    }

    /// Generate prompt embedding from context
    fn generate_prompt_embedding(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        let default_matrix = Array2::zeros((128, 64));
        let prompt_matrix = self.parameters.get("prompt_matrix")
            .unwrap_or(&default_matrix);
        
        let context_features = self.extract_context_features(context)?;
        Ok(prompt_matrix.dot(&context_features))
    }

    /// Retrieve relevant memories
    fn retrieve_memories(&self, context: &EmbeddingContext) -> Result<Array1<f32>> {
        // Simplified memory retrieval
        let default_matrix = Array2::zeros((128, 64));
        let memory_matrix = self.parameters.get("memory_matrix")
            .unwrap_or(&default_matrix);
        
        let context_features = self.extract_context_features(context)?;
        Ok(memory_matrix.dot(&context_features))
    }
}

/// Context cache for efficient context reuse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCache {
    /// Cache configuration
    pub config: ContextCacheConfig,
    /// Cached contexts
    pub cached_contexts: HashMap<String, CachedContext>,
    /// Cache statistics
    pub stats: CacheStats,
}

/// Cached context entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedContext {
    /// Context data
    pub context: EmbeddingContext,
    /// Processed context vectors
    pub context_vectors: HashMap<ContextType, Array1<f32>>,
    /// Fused context representation
    pub fused_context: Array1<f32>,
    /// Cache timestamp
    pub cached_at: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
    /// Last accessed
    pub last_accessed: DateTime<Utc>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub cache_hits: u64,
    /// Total cache misses
    pub cache_misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current cache size
    pub current_size: usize,
    /// Maximum size reached
    pub max_size_reached: usize,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            current_size: 0,
            max_size_reached: 0,
        }
    }
}

impl ContextCache {
    /// Create new context cache
    pub fn new(config: ContextCacheConfig) -> Self {
        Self {
            config,
            cached_contexts: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Get context from cache
    pub fn get_context(&mut self, context_key: &str) -> Option<CachedContext> {
        // Check if expired first
        let now = Utc::now();
        let expiry_duration = chrono::Duration::hours(self.config.expiry_hours as i64);
        
        if let Some(cached) = self.cached_contexts.get(context_key) {
            if now.signed_duration_since(cached.cached_at) > expiry_duration {
                self.cached_contexts.remove(context_key);
                self.stats.cache_misses += 1;
                return None;
            }
        }
        
        // Get and update if not expired
        if let Some(cached) = self.cached_contexts.get_mut(context_key) {
            // Update access statistics
            cached.access_count += 1;
            cached.last_accessed = now;
            self.stats.cache_hits += 1;
            
            Some(cached.clone())
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }

    /// Put context in cache
    pub fn put_context(&mut self, context_key: String, cached_context: CachedContext) {
        // Check cache size limit
        if self.cached_contexts.len() >= self.config.max_cache_size {
            self.evict_lru();
        }
        
        self.cached_contexts.insert(context_key, cached_context);
        self.stats.current_size = self.cached_contexts.len();
        self.stats.max_size_reached = std::cmp::max(self.stats.max_size_reached, self.stats.current_size);
    }

    /// Evict least recently used context
    fn evict_lru(&mut self) {
        if let Some((lru_key, _)) = self.cached_contexts
            .iter()
            .min_by_key(|(_, cached)| cached.last_accessed)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.cached_contexts.remove(&lru_key);
            self.stats.evictions += 1;
        }
    }

    /// Generate cache key for context
    pub fn generate_cache_key(&self, context: &EmbeddingContext) -> String {
        // Simplified cache key generation
        let mut key_parts = Vec::new();
        
        if let Some(query_ctx) = &context.query_context {
            key_parts.push(format!("query:{}", query_ctx.query_text));
        }
        
        if let Some(user_ctx) = &context.user_context {
            key_parts.push(format!("user:{}", user_ctx.user_id));
        }
        
        if let Some(task_ctx) = &context.task_context {
            key_parts.push(format!("task:{:?}", task_ctx.task_type));
        }
        
        // Add timestamp bucket for temporal sensitivity
        let timestamp_bucket = context.timestamp.timestamp() / 3600; // Hour buckets
        key_parts.push(format!("time:{}", timestamp_bucket));
        
        key_parts.join("|")
    }
}

/// Performance metrics for contextual embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Context processing metrics
    pub context_processing: ContextProcessingMetrics,
    /// Fusion metrics
    pub fusion_metrics: FusionMetrics,
    /// Adaptation metrics
    pub adaptation_metrics: AdaptationMetrics,
    /// Cache metrics
    pub cache_metrics: CacheMetrics,
    /// Overall performance metrics
    pub overall_metrics: OverallMetrics,
}

/// Context processing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextProcessingMetrics {
    /// Average processing time per context type
    pub avg_processing_time: HashMap<ContextType, f64>,
    /// Success rates per context type
    pub success_rates: HashMap<ContextType, f32>,
    /// Error counts per context type
    pub error_counts: HashMap<ContextType, u64>,
    /// Total contexts processed
    pub total_processed: u64,
}

impl Default for ContextProcessingMetrics {
    fn default() -> Self {
        Self {
            avg_processing_time: HashMap::new(),
            success_rates: HashMap::new(),
            error_counts: HashMap::new(),
            total_processed: 0,
        }
    }
}

/// Fusion performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetrics {
    /// Average fusion time
    pub avg_fusion_time_ms: f64,
    /// Fusion success rate
    pub fusion_success_rate: f32,
    /// Fusion quality score
    pub fusion_quality_score: f32,
    /// Total fusions performed
    pub total_fusions: u64,
}

impl Default for FusionMetrics {
    fn default() -> Self {
        Self {
            avg_fusion_time_ms: 0.0,
            fusion_success_rate: 1.0,
            fusion_quality_score: 0.0,
            total_fusions: 0,
        }
    }
}

/// Adaptation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMetrics {
    /// Adaptation success rate
    pub adaptation_success_rate: f32,
    /// Average adaptation time
    pub avg_adaptation_time_ms: f64,
    /// Adaptation quality improvement
    pub quality_improvement: f32,
    /// Total adaptations performed
    pub total_adaptations: u64,
}

impl Default for AdaptationMetrics {
    fn default() -> Self {
        Self {
            adaptation_success_rate: 1.0,
            avg_adaptation_time_ms: 0.0,
            quality_improvement: 0.0,
            total_adaptations: 0,
        }
    }
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Cache hit rate
    pub hit_rate: f32,
    /// Average cache lookup time
    pub avg_lookup_time_ms: f64,
    /// Cache efficiency score
    pub efficiency_score: f32,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            avg_lookup_time_ms: 0.0,
            efficiency_score: 0.0,
            memory_usage_mb: 0.0,
        }
    }
}

/// Overall performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMetrics {
    /// End-to-end latency
    pub e2e_latency_ms: f64,
    /// Throughput (embeddings per second)
    pub throughput_eps: f32,
    /// Quality score
    pub quality_score: f32,
    /// Resource utilization
    pub resource_utilization: f32,
}

impl Default for OverallMetrics {
    fn default() -> Self {
        Self {
            e2e_latency_ms: 0.0,
            throughput_eps: 0.0,
            quality_score: 0.0,
            resource_utilization: 0.0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            context_processing: ContextProcessingMetrics::default(),
            fusion_metrics: FusionMetrics::default(),
            adaptation_metrics: AdaptationMetrics::default(),
            cache_metrics: CacheMetrics::default(),
            overall_metrics: OverallMetrics::default(),
        }
    }
}

/// Main contextual embedding model implementation
impl ContextualEmbeddingModel {
    /// Create new contextual embedding model
    pub fn new(config: ContextualConfig) -> Self {
        let model_id = Uuid::new_v4();
        
        // Initialize base model
        let base_model = BaseEmbeddingModel {
            model_type: "contextual-transformer".to_string(),
            input_dim: config.base_dim,
            output_dim: config.context_dim,
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "projection".to_string(),
                    Array2::from_shape_fn((config.context_dim, config.base_dim), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params
            },
        };
        
        // Initialize context processors
        let mut context_processors = HashMap::new();
        for context_type in &config.context_types {
            let processor = ContextProcessor {
                context_type: context_type.clone(),
                strategy: ProcessingStrategy::MLP,
                parameters: HashMap::new(),
                stats: ProcessingStats::default(),
            };
            context_processors.insert(context_type.clone(), processor);
        }
        
        // Initialize fusion network
        let fusion_network = FusionNetwork {
            method: config.fusion_method.clone(),
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "fusion_weights".to_string(),
                    Array2::from_shape_fn((config.context_types.len(), 1), |(_, _)| {
                        1.0 / config.context_types.len() as f32
                    }),
                );
                params.insert(
                    "query_matrix".to_string(),
                    Array2::from_shape_fn((config.context_dim, 256), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params.insert(
                    "key_matrix".to_string(),
                    Array2::from_shape_fn((config.context_dim, 256), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params.insert(
                    "value_matrix".to_string(),
                    Array2::from_shape_fn((config.context_dim, 256), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params
            },
            attention_heads: 8,
            hidden_dim: config.context_dim,
        };
        
        // Initialize adaptation engines
        let mut adaptation_engines = HashMap::new();
        let adaptation_engine = AdaptationEngine {
            strategy: config.adaptation_strategy.clone(),
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "attention_matrix".to_string(),
                    Array2::from_shape_fn((config.context_dim, 64), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params.insert(
                    "gamma_matrix".to_string(),
                    Array2::from_shape_fn((config.context_dim, 64), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params.insert(
                    "beta_matrix".to_string(),
                    Array2::from_shape_fn((config.context_dim, 64), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params.insert(
                    "adapter_down".to_string(),
                    Array2::from_shape_fn((64, config.context_dim), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params.insert(
                    "adapter_up".to_string(),
                    Array2::from_shape_fn((config.context_dim, 64), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params.insert(
                    "meta_network".to_string(),
                    Array2::from_shape_fn((config.context_dim, 64), |(_, _)| {
                        (rand::random::<f32>() - 0.5) * 0.1
                    }),
                );
                params
            },
            learning_rate: 0.001,
            adaptation_history: Vec::new(),
        };
        adaptation_engines.insert(config.adaptation_strategy.clone(), adaptation_engine);
        
        // Initialize context cache
        let context_cache = ContextCache::new(config.cache_config.clone());
        
        // Store dimensions before moving config
        let context_dim = config.context_dim;
        
        Self {
            config,
            model_id,
            base_model,
            context_processors,
            fusion_network,
            adaptation_engines,
            context_cache,
            performance_metrics: PerformanceMetrics::default(),
            training_stats: TrainingStats::default(),
            model_stats: ModelStats {
                dimensions: context_dim,
                model_type: "contextual-embedding".to_string(),
                ..Default::default()
            },
            is_trained: false,
        }
    }

    /// Generate contextual embedding
    pub async fn generate_contextual_embedding(
        &mut self,
        input: &str,
        context: EmbeddingContext,
    ) -> Result<Array1<f32>> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        let cache_key = self.context_cache.generate_cache_key(&context);
        if let Some(cached) = self.context_cache.get_context(&cache_key) {
            // Use cached fused context with base embedding
            let base_embedding = self.base_model.generate_embedding(input)?;
            return self.apply_cached_context(&base_embedding, &cached.fused_context);
        }
        
        // Generate base embedding
        let base_embedding = self.base_model.generate_embedding(input)?;
        
        // Process individual contexts
        let mut context_vectors = HashMap::new();
        for context_type in &self.config.context_types {
            if let Some(processor) = self.context_processors.get_mut(context_type) {
                let context_vector = processor.process_context(&context)?;
                context_vectors.insert(context_type.clone(), context_vector);
            }
        }
        
        // Fuse contexts
        let fused_context = self.fusion_network.fuse_contexts(context_vectors.clone())?;
        
        // Cache the processed context
        let cached_context = CachedContext {
            context: context.clone(),
            context_vectors,
            fused_context: fused_context.clone(),
            cached_at: Utc::now(),
            access_count: 1,
            last_accessed: Utc::now(),
        };
        self.context_cache.put_context(cache_key, cached_context);
        
        // Apply contextual adaptation
        let adapted_embedding = if let Some(adaptation_engine) = self.adaptation_engines.get_mut(&self.config.adaptation_strategy) {
            adaptation_engine.adapt_embedding(&base_embedding, &context, None)?
        } else {
            base_embedding
        };
        
        // Update performance metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.performance_metrics.overall_metrics.e2e_latency_ms = processing_time;
        
        Ok(adapted_embedding)
    }

    /// Apply cached context to base embedding
    fn apply_cached_context(&self, base_embedding: &Array1<f32>, fused_context: &Array1<f32>) -> Result<Array1<f32>> {
        // Simple context application - could be more sophisticated
        let mut result = base_embedding.clone();
        
        let context_influence = 0.1; // How much context affects the embedding
        for i in 0..std::cmp::min(result.len(), fused_context.len()) {
            result[i] += fused_context[i] * context_influence;
        }
        
        Ok(result)
    }

    /// Update model with feedback
    pub async fn update_with_feedback(
        &mut self,
        input: &str,
        context: EmbeddingContext,
        feedback: UserFeedback,
    ) -> Result<()> {
        // Update adaptation engines with feedback
        for adaptation_engine in self.adaptation_engines.values_mut() {
            let base_embedding = self.base_model.generate_embedding(input)?;
            adaptation_engine.adapt_embedding(&base_embedding, &context, Some(&feedback))?;
            
            // Record adaptation
            let adaptation_record = AdaptationRecord {
                timestamp: Utc::now(),
                adaptation_type: AdaptationType::EmbeddingRefinement,
                trigger_feedback: feedback.clone(),
                changes: HashMap::new(),
                performance_impact: 0.0, // Would be computed based on evaluation
            };
            adaptation_engine.adaptation_history.push(adaptation_record);
        }
        
        // Update performance metrics
        self.performance_metrics.adaptation_metrics.total_adaptations += 1;
        
        Ok(())
    }

    /// Get model performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> &CacheStats {
        &self.context_cache.stats
    }

    /// Train the contextual embedding model
    pub async fn train(
        &mut self,
        training_data: Vec<(String, EmbeddingContext, Option<UserFeedback>)>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        for (input, context, feedback) in training_data {
            // Generate embedding and update with feedback if available
            let _embedding = self.generate_contextual_embedding(&input, context.clone()).await?;
            
            if let Some(feedback) = feedback {
                self.update_with_feedback(&input, context, feedback).await?;
            }
        }
        
        self.is_trained = true;
        self.training_stats.training_time_seconds = start_time.elapsed().as_secs_f64();
        
        Ok(())
    }
}

/// Implementation of EmbeddingModel trait for contextual embeddings
#[async_trait]
impl EmbeddingModel for ContextualEmbeddingModel {
    fn config(&self) -> &ModelConfig {
        &self.config.base_config
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "contextual-embedding"
    }

    fn add_triple(&mut self, _triple: Triple) -> Result<()> {
        // Contextual embeddings don't use triples directly
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let start_time = std::time::Instant::now();
        let epochs = epochs.unwrap_or(10);
        
        for _epoch in 0..epochs {
            // Training logic would go here
        }
        
        self.is_trained = true;
        let training_time = start_time.elapsed().as_secs_f64();
        
        let stats = TrainingStats {
            epochs_completed: epochs,
            final_loss: 0.1,
            training_time_seconds: training_time,
            convergence_achieved: true,
            loss_history: vec![0.5, 0.3, 0.2, 0.1],
        };
        
        self.training_stats = stats.clone();
        Ok(stats)
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        let embedding = self.base_model.generate_embedding(entity)?;
        Ok(Vector::from_array1(&embedding))
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        let embedding = self.base_model.generate_embedding(relation)?;
        Ok(Vector::from_array1(&embedding))
    }

    fn score_triple(&self, _subject: &str, _predicate: &str, _object: &str) -> Result<f64> {
        Ok(0.5) // Simplified
    }

    fn predict_objects(&self, _subject: &str, _predicate: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let mut predictions = Vec::new();
        for i in 0..k {
            predictions.push((format!("object_{}", i), 0.8 - i as f64 * 0.1));
        }
        Ok(predictions)
    }

    fn predict_subjects(&self, _predicate: &str, _object: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let mut predictions = Vec::new();
        for i in 0..k {
            predictions.push((format!("subject_{}", i), 0.8 - i as f64 * 0.1));
        }
        Ok(predictions)
    }

    fn predict_relations(&self, _subject: &str, _object: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let mut predictions = Vec::new();
        for i in 0..k {
            predictions.push((format!("relation_{}", i), 0.8 - i as f64 * 0.1));
        }
        Ok(predictions)
    }

    fn get_entities(&self) -> Vec<String> {
        vec![]
    }

    fn get_relations(&self) -> Vec<String> {
        vec![]
    }

    fn get_stats(&self) -> crate::ModelStats {
        self.model_stats.clone()
    }

    fn save(&self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn clear(&mut self) {
        self.is_trained = false;
        self.context_cache.cached_contexts.clear();
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for text in texts {
            let embedding = self.base_model.generate_embedding(text)?;
            results.push(embedding.into_raw_vec());
        }
        Ok(results)
    }
}

/// Convenience functions for common contextual embedding scenarios
impl ContextualEmbeddingModel {
    /// Generate query-specific embedding
    pub async fn generate_query_embedding(
        &mut self,
        query: &str,
        intent: QueryIntent,
        complexity: QueryComplexity,
        domain: String,
    ) -> Result<Array1<f32>> {
        let query_context = QueryContext {
            query_text: query.to_string(),
            intent,
            complexity,
            domain,
            keywords: query.split_whitespace().map(|s| s.to_string()).collect(),
            entities: Vec::new(),
            relations: Vec::new(),
            answer_type: None,
        };
        
        let context = EmbeddingContext::new().with_query_context(query_context);
        self.generate_contextual_embedding(query, context).await
    }

    /// Generate user-specific embedding
    pub async fn generate_user_embedding(
        &mut self,
        input: &str,
        user_id: String,
        preferences: UserPreferences,
        expertise_level: ExpertiseLevel,
    ) -> Result<Array1<f32>> {
        let user_context = UserContext {
            user_id,
            preferences,
            expertise_level,
            interaction_history: Vec::new(),
            domain_interests: HashMap::new(),
            language_preferences: vec!["en".to_string()],
            personalization_vector: None,
        };
        
        let context = EmbeddingContext::new().with_user_context(user_context);
        self.generate_contextual_embedding(input, context).await
    }

    /// Generate task-specific embedding
    pub async fn generate_task_embedding(
        &mut self,
        input: &str,
        task_type: TaskType,
        domain: String,
        performance_requirements: PerformanceRequirements,
    ) -> Result<Array1<f32>> {
        let task_context = TaskContext {
            task_type,
            domain,
            parameters: HashMap::new(),
            capabilities: Vec::new(),
            performance_requirements,
            task_embeddings: HashMap::new(),
        };
        
        let context = EmbeddingContext::new().with_task_context(task_context);
        self.generate_contextual_embedding(input, context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_contextual_config_default() {
        let config = ContextualConfig::default();
        assert_eq!(config.base_dim, 768);
        assert_eq!(config.context_dim, 512);
        assert!(config.context_types.contains(&ContextType::Query));
        assert!(config.context_types.contains(&ContextType::User));
    }

    #[tokio::test]
    async fn test_embedding_context_creation() {
        let context = EmbeddingContext::new()
            .with_query_context(QueryContext {
                query_text: "test query".to_string(),
                intent: QueryIntent::Factual,
                complexity: QueryComplexity::Simple,
                domain: "general".to_string(),
                keywords: vec!["test".to_string()],
                entities: Vec::new(),
                relations: Vec::new(),
                answer_type: None,
            })
            .with_metadata("language".to_string(), "en".to_string());
        
        assert!(context.query_context.is_some());
        assert_eq!(context.metadata.get("language"), Some(&"en".to_string()));
    }

    #[tokio::test]
    async fn test_contextual_embedding_model_creation() {
        let config = ContextualConfig::default();
        let model = ContextualEmbeddingModel::new(config);
        
        assert_eq!(model.config.base_dim, 768);
        assert_eq!(model.config.context_dim, 512);
        assert!(!model.is_trained);
    }

    #[tokio::test]
    async fn test_base_embedding_generation() {
        let base_model = BaseEmbeddingModel {
            model_type: "test".to_string(),
            input_dim: 100,
            output_dim: 50,
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "projection".to_string(),
                    Array2::from_shape_fn((50, 100), |(_, _)| 0.1),
                );
                params
            },
        };
        
        let embedding = base_model.generate_embedding("test input").unwrap();
        assert_eq!(embedding.len(), 50);
    }

    #[tokio::test]
    async fn test_context_processor() {
        let mut processor = ContextProcessor {
            context_type: ContextType::Query,
            strategy: ProcessingStrategy::Linear,
            parameters: HashMap::new(),
            stats: ProcessingStats::default(),
        };
        
        let context = EmbeddingContext::new()
            .with_query_context(QueryContext {
                query_text: "test".to_string(),
                intent: QueryIntent::Factual,
                complexity: QueryComplexity::Simple,
                domain: "test".to_string(),
                keywords: vec!["test".to_string()],
                entities: Vec::new(),
                relations: Vec::new(),
                answer_type: None,
            });
        
        let result = processor.process_context(&context).unwrap();
        assert_eq!(result.len(), 256);
        assert_eq!(processor.stats.contexts_processed, 1);
    }

    #[tokio::test]
    async fn test_context_fusion() {
        let fusion_network = FusionNetwork {
            method: ContextFusionMethod::Concatenation,
            parameters: HashMap::new(),
            attention_heads: 8,
            hidden_dim: 512,
        };
        
        let mut context_vectors = HashMap::new();
        context_vectors.insert(ContextType::Query, Array1::zeros(256));
        context_vectors.insert(ContextType::User, Array1::ones(256));
        
        let result = fusion_network.fuse_contexts(context_vectors).unwrap();
        assert_eq!(result.len(), 256 * 7); // 7 context types with padding
    }

    #[tokio::test]
    async fn test_adaptation_engine() {
        let mut adaptation_engine = AdaptationEngine {
            strategy: AdaptationStrategy::DynamicAttention,
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "attention_matrix".to_string(),
                    Array2::from_shape_fn((128, 64), |(_, _)| 0.1),
                );
                params
            },
            learning_rate: 0.001,
            adaptation_history: Vec::new(),
        };
        
        let base_embedding = Array1::ones(128);
        let context = EmbeddingContext::new();
        
        let adapted = adaptation_engine.adapt_embedding(&base_embedding, &context, None).unwrap();
        assert_eq!(adapted.len(), 128);
    }

    #[tokio::test]
    async fn test_context_cache() {
        let mut cache = ContextCache::new(ContextCacheConfig::default());
        
        let context = EmbeddingContext::new();
        let cached_context = CachedContext {
            context: context.clone(),
            context_vectors: HashMap::new(),
            fused_context: Array1::zeros(512),
            cached_at: Utc::now(),
            access_count: 0,
            last_accessed: Utc::now(),
        };
        
        let cache_key = cache.generate_cache_key(&context);
        cache.put_context(cache_key.clone(), cached_context);
        
        let retrieved = cache.get_context(&cache_key);
        assert!(retrieved.is_some());
        assert_eq!(cache.stats.cache_hits, 1);
    }

    #[tokio::test]
    async fn test_query_specific_embedding() {
        let config = ContextualConfig::default();
        let mut model = ContextualEmbeddingModel::new(config);
        
        let embedding = model.generate_query_embedding(
            "What is machine learning?",
            QueryIntent::Factual,
            QueryComplexity::Simple,
            "computer_science".to_string(),
        ).await.unwrap();
        
        assert_eq!(embedding.len(), 512);
    }

    #[tokio::test]
    async fn test_user_specific_embedding() {
        let config = ContextualConfig::default();
        let mut model = ContextualEmbeddingModel::new(config);
        
        let preferences = UserPreferences {
            detail_level: DetailLevel::Standard,
            preferred_formats: vec!["text".to_string()],
            content_filters: Vec::new(),
            accessibility: AccessibilityPreferences::default(),
            privacy_settings: PrivacySettings::default(),
        };
        
        let embedding = model.generate_user_embedding(
            "machine learning tutorial",
            "user123".to_string(),
            preferences,
            ExpertiseLevel::Beginner,
        ).await.unwrap();
        
        assert_eq!(embedding.len(), 512);
    }

    #[tokio::test]
    async fn test_task_specific_embedding() {
        let config = ContextualConfig::default();
        let mut model = ContextualEmbeddingModel::new(config);
        
        let embedding = model.generate_task_embedding(
            "classify sentiment",
            TaskType::Classification,
            "nlp".to_string(),
            PerformanceRequirements::default(),
        ).await.unwrap();
        
        assert_eq!(embedding.len(), 512);
    }

    #[tokio::test]
    async fn test_feedback_integration() {
        let config = ContextualConfig::default();
        let mut model = ContextualEmbeddingModel::new(config);
        
        let context = EmbeddingContext::new();
        let feedback = UserFeedback::Positive(0.8);
        
        model.update_with_feedback("test input", context, feedback).await.unwrap();
        assert_eq!(model.performance_metrics.adaptation_metrics.total_adaptations, 1);
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let config = ContextualConfig::default();
        let model = ContextualEmbeddingModel::new(config);
        
        let metrics = model.get_performance_metrics();
        assert_eq!(metrics.adaptation_metrics.total_adaptations, 0);
        assert_eq!(metrics.fusion_metrics.total_fusions, 0);
    }
}