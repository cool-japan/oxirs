//! Configuration for AI Orchestrator

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::types::ModelSelectionStrategy;
pub use super::types::PerformanceRequirements;

/// Main configuration for the AI orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeOrchestratorConfig {
    /// Model selection strategy
    pub selection_strategy: ModelSelectionStrategy,
    /// Performance requirements for model selection
    pub performance_requirements: PerformanceRequirements,
    /// Enable automatic model retraining
    pub enable_auto_retraining: bool,
    /// Retraining interval in seconds
    pub retraining_interval_secs: u64,
    /// Maximum number of models to ensemble
    pub max_ensemble_size: usize,
    /// Enable feature caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Minimum confidence threshold for predictions
    pub min_confidence: f64,
    /// Enable adaptive threshold adjustment
    pub adaptive_thresholds: bool,
    /// Maximum concurrent model evaluations
    pub max_concurrent_evaluations: usize,
}

impl Default for ShapeOrchestratorConfig {
    fn default() -> Self {
        Self {
            selection_strategy: ModelSelectionStrategy::PerformanceBased,
            performance_requirements: PerformanceRequirements::default(),
            enable_auto_retraining: false,
            retraining_interval_secs: 3600,
            max_ensemble_size: 5,
            enable_caching: true,
            cache_ttl_secs: 300,
            min_confidence: 0.7,
            adaptive_thresholds: true,
            max_concurrent_evaluations: 4,
        }
    }
}

impl ShapeOrchestratorConfig {
    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            selection_strategy: ModelSelectionStrategy::EnsembleWeighted,
            max_ensemble_size: 10,
            enable_auto_retraining: true,
            retraining_interval_secs: 1800,
            ..Default::default()
        }
    }

    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            selection_strategy: ModelSelectionStrategy::PerformanceBased,
            max_ensemble_size: 2,
            enable_caching: true,
            cache_ttl_secs: 60,
            ..Default::default()
        }
    }

    /// Get the retraining interval as a Duration
    pub fn retraining_interval(&self) -> Duration {
        Duration::from_secs(self.retraining_interval_secs)
    }

    /// Get the cache TTL as a Duration
    pub fn cache_ttl(&self) -> Duration {
        Duration::from_secs(self.cache_ttl_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ShapeOrchestratorConfig::default();
        assert!(config.enable_caching);
        assert_eq!(config.max_ensemble_size, 5);
        assert!((config.min_confidence - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_high_performance_config() {
        let config = ShapeOrchestratorConfig::high_performance();
        assert!(config.enable_auto_retraining);
        assert_eq!(config.max_ensemble_size, 10);
    }

    #[test]
    fn test_low_latency_config() {
        let config = ShapeOrchestratorConfig::low_latency();
        assert_eq!(config.max_ensemble_size, 2);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_retraining_interval_duration() {
        let config = ShapeOrchestratorConfig::default();
        assert_eq!(config.retraining_interval(), Duration::from_secs(3600));
    }

    #[test]
    fn test_cache_ttl_duration() {
        let config = ShapeOrchestratorConfig::default();
        assert_eq!(config.cache_ttl(), Duration::from_secs(300));
    }
}
