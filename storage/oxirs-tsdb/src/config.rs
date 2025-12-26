use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Time-series database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsdbConfig {
    /// Duration of each time chunk (default: 2 hours)
    #[serde(with = "humantime_serde")]
    pub chunk_duration: Duration,

    /// Enable Gorilla compression
    pub compression_enabled: bool,

    /// In-memory write buffer size (number of data points)
    pub buffer_size: usize,

    /// Enable Write-Ahead Log for durability
    pub wal_enabled: bool,

    /// Sync WAL to disk on every write (slower but more durable)
    pub wal_sync: bool,

    /// Enable background compaction
    pub compaction_enabled: bool,

    /// Compaction interval
    #[serde(with = "humantime_serde")]
    pub compaction_interval: Duration,

    /// Retention policies
    pub retention_policies: Vec<RetentionPolicy>,
}

/// Retention policy for automatic data expiration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Policy name
    pub name: String,

    /// Retention duration
    #[serde(with = "humantime_serde")]
    pub duration: Duration,

    /// Optional downsampling configuration
    pub downsampling: Option<Downsampling>,
}

/// Downsampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Downsampling {
    /// Original resolution
    #[serde(with = "humantime_serde")]
    pub from_resolution: Duration,

    /// Target resolution after downsampling
    #[serde(with = "humantime_serde")]
    pub to_resolution: Duration,

    /// Aggregation function
    pub aggregation: AggregationFunction,
}

/// Aggregation functions for downsampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationFunction {
    /// Average value
    #[serde(rename = "AVG")]
    Average,
    /// Minimum value
    #[serde(rename = "MIN")]
    Min,
    /// Maximum value
    #[serde(rename = "MAX")]
    Max,
    /// Sum of values
    #[serde(rename = "SUM")]
    Sum,
    /// Last value (latest timestamp)
    #[serde(rename = "LAST")]
    Last,
    /// First value (earliest timestamp)
    #[serde(rename = "FIRST")]
    First,
}

impl Default for TsdbConfig {
    fn default() -> Self {
        Self {
            chunk_duration: Duration::from_secs(7200), // 2 hours
            compression_enabled: true,
            buffer_size: 100_000,
            wal_enabled: true,
            wal_sync: false,
            compaction_enabled: true,
            compaction_interval: Duration::from_secs(3600), // 1 hour
            retention_policies: vec![RetentionPolicy {
                name: "raw".to_string(),
                duration: Duration::from_secs(7 * 24 * 3600), // 7 days
                downsampling: None,
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TsdbConfig::default();
        assert!(config.compression_enabled);
        assert_eq!(config.buffer_size, 100_000);
        assert_eq!(config.retention_policies.len(), 1);
    }
}
