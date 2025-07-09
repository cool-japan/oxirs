//! Core types and data structures for advanced pattern mining

use serde::{Deserialize, Serialize};
use std::collections::HashMap;


/// Advanced pattern mining configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPatternMiningConfig {
    /// Minimum support threshold for frequent patterns
    pub min_support: f64,

    /// Minimum confidence for association rules
    pub min_confidence: f64,

    /// Maximum pattern length to consider
    pub max_pattern_length: usize,

    /// Enable temporal pattern analysis
    pub enable_temporal_analysis: bool,

    /// Enable hierarchical pattern discovery
    pub enable_hierarchical_patterns: bool,

    /// Enable parallel processing
    pub enable_parallel_processing: bool,

    /// Window size for sliding window analysis
    pub sliding_window_size: usize,

    /// Quality threshold for patterns
    pub quality_threshold: f64,
}

impl Default for AdvancedPatternMiningConfig {
    fn default() -> Self {
        Self {
            min_support: 0.05,
            min_confidence: 0.7,
            max_pattern_length: 5,
            enable_temporal_analysis: true,
            enable_hierarchical_patterns: true,
            enable_parallel_processing: true,
            sliding_window_size: 1000,
            quality_threshold: 0.8,
        }
    }
}

/// Type of pattern item
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternItemType {
    Property,
    Class,
    ValuePattern,
    Cardinality,
    DataType,
    LanguageTag,
}

/// Role of item in pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemRole {
    Subject,
    Predicate,
    Object,
    Context,
    Modifier,
}

/// Pattern type classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternType {
    /// Structural patterns (graph topology)
    Structural,

    /// Value patterns (data content)
    Value,

    /// Cardinality patterns (quantity constraints)
    Cardinality,

    /// Temporal patterns (time-based)
    Temporal,

    /// Mixed patterns (combination)
    Mixed,
}

/// Trend direction for temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
}

/// Type of suggested constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    MinCount,
    MaxCount,
    ExactCount,
    MinLength,
    MaxLength,
    Pattern,
    DataType,
    NodeKind,
    Class,
    HasValue,
    In,
    MinInclusive,
    MaxInclusive,
    MinExclusive,
    MaxExclusive,
}

/// Access trend for cache analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessTrend {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
    Random,
}

/// Cache access result for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessResult {
    Hit,
    Miss,
    Expired,
    Evicted,
}

/// Time window for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeWindow {
    Last5Minutes,
    Last15Minutes,
    LastHour,
    Last24Hours,
    LastWeek,
    LastMonth,
}

/// Pattern ranking criteria for advanced pattern mining
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternRankingCriteria {
    Frequency,
    Confidence,
    Support,
    Lift,
    Conviction,
    Coverage,
    Novelty,
    Complexity,
}


/// Cache tuning state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum TuningState {
    #[default]
    Idle,
    Monitoring,
    Analyzing,
    Optimizing,
    Testing,
}

/// Type of cache optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    SizeOptimization,
    AccessOptimization,
    QualityOptimization,
    MemoryOptimization,
    PerformanceOptimization,
}

/// Cache eviction algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionAlgorithm {
    LRU,
    LFU,
    QualityBased,
    Intelligent,
    TimeAware,
}

/// Anomaly type for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    Statistical,
    Structural,
    Temporal,
    Quality,
}

/// Anomaly severity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Time granularity for analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeGranularity {
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

/// Placeholder for SPARQL query results
#[derive(Debug)]
pub struct QueryResults {
    /// Bindings returned from query
    pub bindings: Vec<HashMap<String, String>>,
}

/// Pattern mining statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternMiningStats {
    /// Total patterns discovered
    pub total_patterns: usize,

    /// High-quality patterns (above threshold)
    pub high_quality_patterns: usize,

    /// Temporal patterns found
    pub temporal_patterns: usize,

    /// Hierarchical patterns found
    pub hierarchical_patterns: usize,

    /// Processing time
    pub processing_time_ms: u64,

    /// Memory usage peak
    pub peak_memory_mb: f64,

    /// Coverage ratio
    pub coverage_ratio: f64,

    /// Pattern efficiency score
    pub efficiency_score: f64,
}