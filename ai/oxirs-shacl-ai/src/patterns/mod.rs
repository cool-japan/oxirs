//! Pattern recognition and analysis for RDF data
//!
//! This module implements AI-powered pattern recognition for discovering
//! data patterns, usage patterns, and structural patterns in RDF graphs.

pub mod algorithms;
pub mod analyzer;
pub mod cache;
pub mod config;
pub mod types;

// Re-export main types for easy access
pub use algorithms::PatternAlgorithms as PatternAlgorithmEngine;
pub use analyzer::PatternAnalyzer;
pub use cache::{CacheStats, PatternCache};
pub use config::{PatternAlgorithms, PatternCacheSettings, PatternConfig};
pub use types::{
    CachedPatternResult, CardinalityType, HierarchyType, Pattern, PatternExample,
    PatternModelState, PatternSimilarity, PatternStatistics, PatternTrainingData, PatternType,
    SimilarityType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_pattern_config_default() {
        let config = PatternConfig::default();
        assert!(config.enable_pattern_recognition);
        assert!(config.enable_structural_analysis);
        assert!(config.enable_usage_analysis);
        assert!(!config.enable_temporal_analysis);
        assert_eq!(config.min_support_threshold, 0.1);
        assert_eq!(config.min_confidence_threshold, 0.7);
    }

    #[test]
    fn test_pattern_analyzer_creation() {
        let analyzer = PatternAnalyzer::new();
        assert!(analyzer.config().enable_pattern_recognition);
    }

    #[test]
    fn test_pattern_support_confidence() {
        let pattern = Pattern::ClassUsage {
            id: "test_class_usage_person".to_string(),
            class: NamedNode::new("http://example.org/Person").unwrap(),
            instance_count: 100,
            support: 0.8,
            confidence: 0.95,
            pattern_type: PatternType::Structural,
        };

        assert_eq!(pattern.support(), 0.8);
        assert_eq!(pattern.confidence(), 0.95);
        assert_eq!(pattern.pattern_type(), &PatternType::Structural);
        assert_eq!(pattern.id(), "test_class_usage_person");
    }

    #[test]
    fn test_pattern_cache() {
        let mut cache = PatternCache::default();
        assert!(cache.is_enabled());

        let patterns = vec![Pattern::ClassUsage {
            id: "test_cache_test_class".to_string(),
            class: NamedNode::new("http://example.org/Test").unwrap(),
            instance_count: 10,
            support: 0.5,
            confidence: 0.8,
            pattern_type: PatternType::Structural,
        }];

        cache.put("test_key".to_string(), patterns.clone());

        let cached = cache.get("test_key");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().patterns.len(), 1);
    }

    #[test]
    fn test_cached_pattern_result_expiry() {
        let patterns = vec![Pattern::ClassUsage {
            id: "test_cached_pattern_test_class".to_string(),
            class: NamedNode::new("http://example.org/Test").unwrap(),
            instance_count: 10,
            support: 0.5,
            confidence: 0.8,
            pattern_type: PatternType::Structural,
        }];

        let cached = CachedPatternResult {
            patterns,
            timestamp: chrono::Utc::now() - chrono::Duration::hours(2),
            ttl: std::time::Duration::from_secs(3600), // 1 hour
        };

        assert!(cached.is_expired());
    }

    #[test]
    fn test_pattern_statistics_default() {
        let stats = PatternStatistics::default();
        assert_eq!(stats.total_analyses, 0);
        assert_eq!(stats.shape_analyses, 0);
        assert_eq!(stats.patterns_discovered, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert!(!stats.model_trained);
    }
}
