//! Configuration for compaction system

use super::strategies::CompactionStrategy;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Compaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Compaction strategy
    pub strategy: CompactionStrategy,
    /// Enable background compaction
    pub enable_background: bool,
    /// Compaction interval for periodic strategy
    pub compaction_interval: Duration,
    /// Fragmentation threshold (0.0 - 1.0) to trigger compaction
    pub fragmentation_threshold: f64,
    /// Batch size (number of vectors to process at once)
    pub batch_size: usize,
    /// Maximum concurrent batches
    pub max_concurrent_batches: usize,
    /// Pause between batches (to avoid impacting queries)
    pub pause_between_batches: Duration,
    /// Maximum compaction duration before pausing
    pub max_compaction_duration: Duration,
    /// Minimum free space to trigger compaction (bytes)
    pub min_free_space_bytes: u64,
    /// Enable verification after compaction
    pub enable_verification: bool,
    /// CPU priority (0-100, lower = more background)
    pub cpu_priority: u8,
    /// Memory limit for compaction (bytes)
    pub memory_limit_bytes: u64,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics retention period
    pub metrics_retention: Duration,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            strategy: CompactionStrategy::Adaptive,
            enable_background: true,
            compaction_interval: Duration::from_secs(3600), // 1 hour
            fragmentation_threshold: 0.3,                   // 30%
            batch_size: 1000,
            max_concurrent_batches: 4,
            pause_between_batches: Duration::from_millis(100),
            max_compaction_duration: Duration::from_secs(300), // 5 minutes
            min_free_space_bytes: 100 * 1024 * 1024,           // 100 MB
            enable_verification: true,
            cpu_priority: 10,                      // Low priority
            memory_limit_bytes: 512 * 1024 * 1024, // 512 MB
            enable_metrics: true,
            metrics_retention: Duration::from_secs(7 * 24 * 3600), // 7 days
        }
    }
}

impl CompactionConfig {
    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.fragmentation_threshold <= 0.0 || self.fragmentation_threshold > 1.0 {
            anyhow::bail!("Fragmentation threshold must be in (0.0, 1.0]");
        }
        if self.batch_size == 0 {
            anyhow::bail!("Batch size must be positive");
        }
        if self.max_concurrent_batches == 0 {
            anyhow::bail!("Max concurrent batches must be positive");
        }
        if self.cpu_priority > 100 {
            anyhow::bail!("CPU priority must be in [0, 100]");
        }
        Ok(())
    }

    /// Create development configuration (aggressive compaction)
    pub fn development() -> Self {
        Self {
            compaction_interval: Duration::from_secs(60), // 1 minute
            fragmentation_threshold: 0.1,                 // 10%
            batch_size: 100,
            pause_between_batches: Duration::from_millis(10),
            ..Default::default()
        }
    }

    /// Create production configuration (conservative compaction)
    pub fn production() -> Self {
        Self {
            compaction_interval: Duration::from_secs(7200), // 2 hours
            fragmentation_threshold: 0.5,                   // 50%
            batch_size: 5000,
            max_concurrent_batches: 8,
            pause_between_batches: Duration::from_millis(500),
            max_compaction_duration: Duration::from_secs(600), // 10 minutes
            memory_limit_bytes: 2 * 1024 * 1024 * 1024,        // 2 GB
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = CompactionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_development_config_is_valid() {
        let config = CompactionConfig::development();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_production_config_is_valid() {
        let config = CompactionConfig::production();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_fragmentation_threshold() {
        let config = CompactionConfig {
            fragmentation_threshold: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_batch_size() {
        let config = CompactionConfig {
            batch_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
