//! Configuration structures for pattern analysis

use serde::{Deserialize, Serialize};

/// Configuration for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Enable pattern recognition
    pub enable_pattern_recognition: bool,

    /// Enable structural pattern analysis
    pub enable_structural_analysis: bool,

    /// Enable usage pattern analysis
    pub enable_usage_analysis: bool,

    /// Enable temporal pattern analysis
    pub enable_temporal_analysis: bool,

    /// Minimum support threshold for patterns
    pub min_support_threshold: f64,

    /// Minimum confidence threshold for patterns
    pub min_confidence_threshold: f64,

    /// Maximum pattern complexity
    pub max_pattern_complexity: usize,

    /// Pattern analysis algorithms
    pub algorithms: PatternAlgorithms,

    /// Enable training
    pub enable_training: bool,

    /// Pattern cache settings
    pub cache_settings: PatternCacheSettings,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enable_pattern_recognition: true,
            enable_structural_analysis: true,
            enable_usage_analysis: true,
            enable_temporal_analysis: false,
            min_support_threshold: 0.1,
            min_confidence_threshold: 0.7,
            max_pattern_complexity: 5,
            algorithms: PatternAlgorithms::default(),
            enable_training: true,
            cache_settings: PatternCacheSettings::default(),
        }
    }
}

/// Pattern analysis algorithms configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAlgorithms {
    /// Enable frequent itemset mining
    pub enable_frequent_itemsets: bool,

    /// Enable association rule mining
    pub enable_association_rules: bool,

    /// Enable graph pattern mining
    pub enable_graph_patterns: bool,

    /// Enable cluster analysis
    pub enable_clustering: bool,

    /// Enable anomaly detection in patterns
    pub enable_anomaly_detection: bool,

    /// Enable sequential pattern mining
    pub enable_sequential_patterns: bool,
}

impl Default for PatternAlgorithms {
    fn default() -> Self {
        Self {
            enable_frequent_itemsets: true,
            enable_association_rules: true,
            enable_graph_patterns: true,
            enable_clustering: false,
            enable_anomaly_detection: true,
            enable_sequential_patterns: false,
        }
    }
}

/// Pattern cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCacheSettings {
    /// Enable pattern caching
    pub enable_caching: bool,

    /// Maximum cache size
    pub max_cache_size: usize,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,

    /// Enable pattern similarity caching
    pub enable_similarity_cache: bool,
}

impl Default for PatternCacheSettings {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 1000,
            cache_ttl_seconds: 3600,
            enable_similarity_cache: true,
        }
    }
}
