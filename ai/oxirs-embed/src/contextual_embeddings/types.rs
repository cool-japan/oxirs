//! Core types and enums for contextual embeddings
//!
//! This module contains all the fundamental data types, enums, and configurations
//! used throughout the contextual embedding system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Types of context to consider for embeddings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ContextType {
    /// Query-specific context
    Query,
    /// User-specific context and preferences
    User,
    /// Task-specific context and requirements
    Task,
    /// Temporal context (time-based)
    Temporal,
    /// Interactive context (feedback-based)
    Interactive,
    /// Domain-specific context
    Domain,
    /// Session context
    Session,
    /// Geographic context
    Geographic,
    /// Device context
    Device,
    /// Environmental context
    Environmental,
    /// Social context
    Social,
    /// Cultural context
    Cultural,
    /// Linguistic context
    Linguistic,
    /// Semantic context
    Semantic,
    /// Syntactic context
    Syntactic,
    /// Pragmatic context
    Pragmatic,
}

/// Adaptation strategies for contextual embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// No adaptation (baseline)
    None,
    /// Linear adaptation
    Linear,
    /// Attention-based adaptation
    Attention,
    /// Multi-layer perceptron adaptation
    MLP,
    /// Transformer-based adaptation
    Transformer,
    /// Mixture of experts
    MixtureOfExperts,
    /// Dynamic routing
    DynamicRouting,
    /// Meta-learning adaptation
    MetaLearning,
    /// Reinforcement learning
    ReinforcementLearning,
    /// Continual learning
    ContinualLearning,
    /// Transfer learning
    TransferLearning,
    /// Few-shot learning
    FewShotLearning,
    /// Zero-shot learning
    ZeroShotLearning,
    /// Self-supervised learning
    SelfSupervisedLearning,
    /// Contrastive learning
    ContrastiveLearning,
    /// Multi-task learning
    MultiTaskLearning,
}

/// Methods for fusing multiple contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextFusionMethod {
    /// Concatenation
    Concatenation,
    /// Element-wise addition
    Addition,
    /// Element-wise multiplication
    Multiplication,
    /// Weighted average
    WeightedAverage,
    /// Attention-based fusion
    Attention,
    /// Gating mechanism
    Gating,
    /// Bilinear pooling
    BilinearPooling,
    /// Compact bilinear pooling
    CompactBilinearPooling,
    /// Multimodal compact bilinear pooling
    MultimodalCompactBilinearPooling,
    /// Low-rank bilinear pooling
    LowRankBilinearPooling,
    /// Hierarchical fusion
    Hierarchical,
    /// Cross-attention
    CrossAttention,
    /// Self-attention
    SelfAttention,
    /// Multi-head attention
    MultiHeadAttention,
    /// Transformer fusion
    TransformerFusion,
    /// Graph-based fusion
    GraphFusion,
    /// Dynamic fusion
    DynamicFusion,
    /// Learned fusion
    LearnedFusion,
    /// Modular fusion
    ModularFusion,
    /// Adaptive fusion
    AdaptiveFusion,
    /// Contextual fusion
    ContextualFusion,
}

/// Feedback aggregation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackAggregation {
    /// Simple averaging
    Average,
    /// Weighted averaging by recency
    RecencyWeighted,
    /// Weighted averaging by confidence
    ConfidenceWeighted,
    /// Exponential moving average
    ExponentialMovingAverage,
    /// Attention-based aggregation
    AttentionBased,
    /// Learned aggregation
    LearnedAggregation,
    /// Hierarchical aggregation
    HierarchicalAggregation,
    /// Contextual aggregation
    ContextualAggregation,
    /// Dynamic aggregation
    DynamicAggregation,
    /// Multi-scale aggregation
    MultiScaleAggregation,
    /// Temporal aggregation
    TemporalAggregation,
    /// Adaptive aggregation
    AdaptiveAggregation,
}

/// Query intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Informational query
    Informational,
    /// Navigational query
    Navigational,
    /// Transactional query
    Transactional,
    /// Comparative query
    Comparative,
    /// Temporal query
    Temporal,
    /// Spatial query
    Spatial,
    /// Analytical query
    Analytical,
    /// Creative query
    Creative,
    /// Factual query
    Factual,
    /// Opinion query
    Opinion,
    /// Procedural query
    Procedural,
    /// Causal query
    Causal,
    /// Hypothetical query
    Hypothetical,
    /// Definitional query
    Definitional,
    /// Classification query
    Classification,
    /// Recommendation query
    Recommendation,
    /// Troubleshooting query
    Troubleshooting,
    /// Exploratory query
    Exploratory,
    /// Confirmatory query
    Confirmatory,
    /// Other
    Other,
}

/// Query complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryComplexity {
    /// Simple single-hop query
    Simple,
    /// Medium complexity multi-hop query
    Medium,
    /// Complex reasoning query
    Complex,
    /// Very complex multi-step query
    VeryComplex,
    /// Expert-level query
    Expert,
    /// Research-level query
    Research,
    /// Ambiguous query
    Ambiguous,
    /// Multi-faceted query
    MultiFaceted,
    /// Interdisciplinary query
    Interdisciplinary,
    /// Open-ended query
    OpenEnded,
}

/// User expertise levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    /// Beginner level
    Beginner,
    /// Intermediate level
    Intermediate,
    /// Advanced level
    Advanced,
    /// Expert level
    Expert,
    /// Professional level
    Professional,
    /// Academic level
    Academic,
    /// Research level
    Research,
    /// Domain expert
    DomainExpert,
}

/// Detail level preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetailLevel {
    /// Minimal details
    Minimal,
    /// Basic details
    Basic,
    /// Moderate details
    Moderate,
    /// Comprehensive details
    Comprehensive,
    /// Exhaustive details
    Exhaustive,
    /// Context-dependent
    ContextDependent,
    /// Adaptive
    Adaptive,
}

/// User feedback types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserFeedback {
    /// Thumbs up/down
    ThumbsUpDown(bool),
    /// Star rating (1-5)
    StarRating(u8),
    /// Relevance score (0.0-1.0)
    RelevanceScore(f64),
    /// Text feedback
    TextFeedback(String),
    /// Click-through behavior
    ClickThrough(bool),
    /// Dwell time (seconds)
    DwellTime(f64),
    /// Bounce rate
    BounceRate(f64),
    /// Conversion rate
    ConversionRate(f64),
    /// Custom feedback
    Custom(HashMap<String, serde_json::Value>),
}

/// Task types for context-aware embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Information retrieval
    InformationRetrieval,
    /// Question answering
    QuestionAnswering,
    /// Text classification
    TextClassification,
    /// Sentiment analysis
    SentimentAnalysis,
    /// Named entity recognition
    NamedEntityRecognition,
    /// Relation extraction
    RelationExtraction,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
    /// Text generation
    TextGeneration,
    /// Dialogue systems
    DialogueSystems,
    /// Recommendation systems
    RecommendationSystems,
    /// Search ranking
    SearchRanking,
    /// Content moderation
    ContentModeration,
    /// Fact checking
    FactChecking,
    /// Knowledge graph completion
    KnowledgeGraphCompletion,
    /// Semantic parsing
    SemanticParsing,
    /// Text similarity
    TextSimilarity,
    /// Document clustering
    DocumentClustering,
    /// Topic modeling
    TopicModeling,
    /// Anomaly detection
    AnomalyDetection,
    /// Trend analysis
    TrendAnalysis,
    /// Time series forecasting
    TimeSeriesForecasting,
    /// Image captioning
    ImageCaptioning,
    /// Visual question answering
    VisualQuestionAnswering,
    /// Cross-modal retrieval
    CrossModalRetrieval,
    /// Multimodal classification
    MultimodalClassification,
    /// Code generation
    CodeGeneration,
    /// Code completion
    CodeCompletion,
    /// Bug detection
    BugDetection,
    /// Test generation
    TestGeneration,
    /// Documentation generation
    DocumentationGeneration,
    /// Academic research
    AcademicResearch,
    /// Literature review
    LiteratureReview,
    /// Patent analysis
    PatentAnalysis,
    /// Legal document analysis
    LegalDocumentAnalysis,
    /// Medical diagnosis
    MedicalDiagnosis,
    /// Drug discovery
    DrugDiscovery,
    /// Financial analysis
    FinancialAnalysis,
    /// Risk assessment
    RiskAssessment,
    /// Fraud detection
    FraudDetection,
    /// Market analysis
    MarketAnalysis,
    /// Customer segmentation
    CustomerSegmentation,
    /// Personalization
    Personalization,
    /// Content recommendation
    ContentRecommendation,
    /// Product recommendation
    ProductRecommendation,
    /// News recommendation
    NewsRecommendation,
    /// Music recommendation
    MusicRecommendation,
    /// Video recommendation
    VideoRecommendation,
    /// Book recommendation
    BookRecommendation,
    /// Job recommendation
    JobRecommendation,
    /// Travel recommendation
    TravelRecommendation,
    /// Restaurant recommendation
    RestaurantRecommendation,
    /// Movie recommendation
    MovieRecommendation,
    /// E-commerce search
    EcommerceSearch,
    /// Academic search
    AcademicSearch,
    /// Enterprise search
    EnterpriseSearch,
    /// Web search
    WebSearch,
    /// Image search
    ImageSearch,
    /// Video search
    VideoSearch,
    /// Audio search
    AudioSearch,
    /// Code search
    CodeSearch,
    /// Custom task
    Custom(String),
}

/// Time periods for temporal context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimePeriod {
    /// Real-time (current moment)
    RealTime,
    /// Recent (last hour)
    Recent,
    /// Today
    Today,
    /// This week
    ThisWeek,
    /// This month
    ThisMonth,
    /// This quarter
    ThisQuarter,
    /// This year
    ThisYear,
    /// Last week
    LastWeek,
    /// Last month
    LastMonth,
    /// Last quarter
    LastQuarter,
    /// Last year
    LastYear,
    /// Historical (more than a year ago)
    Historical,
    /// Seasonal
    Seasonal,
    /// Cyclic
    Cyclic,
    /// Trending
    Trending,
    /// Custom period
    Custom { start: DateTime<Utc>, end: DateTime<Utc> },
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
    /// Volatile trend
    Volatile,
    /// Seasonal trend
    Seasonal,
    /// Cyclic trend
    Cyclic,
    /// No clear trend
    None,
}

/// Adaptation types for contextual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Query-based adaptation
    Query,
    /// User-based adaptation
    User,
    /// Task-based adaptation
    Task,
    /// Temporal adaptation
    Temporal,
    /// Interactive adaptation
    Interactive,
    /// Domain adaptation
    Domain,
    /// Cross-domain adaptation
    CrossDomain,
    /// Multi-task adaptation
    MultiTask,
    /// Continual adaptation
    Continual,
    /// Personalization adaptation
    Personalization,
    /// Contextual adaptation
    Contextual,
    /// Hybrid adaptation
    Hybrid,
}

/// Processing strategies for different contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStrategy {
    /// Eager processing (process all contexts)
    Eager,
    /// Lazy processing (process on demand)
    Lazy,
    /// Selective processing (process based on importance)
    Selective,
    /// Adaptive processing (adjust based on resources)
    Adaptive,
    /// Hierarchical processing (process in stages)
    Hierarchical,
    /// Parallel processing (process contexts in parallel)
    Parallel,
    /// Sequential processing (process contexts sequentially)
    Sequential,
    /// Pipelined processing (process in pipeline)
    Pipelined,
    /// Streaming processing (process continuously)
    Streaming,
    /// Batch processing (process in batches)
    Batch,
    /// Real-time processing (process immediately)
    RealTime,
    /// Offline processing (process offline)
    Offline,
}