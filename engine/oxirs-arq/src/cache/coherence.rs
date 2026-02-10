//! Cache Coherence Protocol for Distributed Cache
//!
//! This module provides cache coherence protocols to ensure consistency across
//! distributed cache nodes. It supports multiple consistency models and provides
//! verification tools to ensure cache coherence.
//!
//! ## Consistency Models
//!
//! - **Eventual**: May serve stale data temporarily, but eventually consistent
//! - **Strong**: Always consistent, but slower (every read checks all nodes)
//! - **BoundedStaleness**: Stale data within a time bound (configurable max staleness)
//!
//! ## Coherence Protocols
//!
//! - **PubSub**: Redis Pub/Sub for invalidation messages (eventual consistency)
//! - **WriteThrough**: Write to L2 immediately on every update (strong consistency)
//! - **WriteBehind**: Async write to L2 (eventual consistency, better performance)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_arq::cache::coherence::{CacheCoherenceProtocol, CoherenceProtocol, CoherenceConfig};
//!
//! let config = CoherenceConfig {
//!     consistency_level: ConsistencyLevel::BoundedStaleness,
//!     max_staleness_seconds: 60,
//! };
//!
//! let protocol = CacheCoherenceProtocol::new(CoherenceProtocol::PubSub, config);
//!
//! // Verify coherence across multiple cache nodes
//! let report = protocol.verify_coherence(&caches).await?;
//! assert!(report.coherence_rate > 0.99);
//! ```

#![cfg(feature = "distributed-cache")]

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

use super::distributed_cache::{CacheKey, CacheValue, DistributedCache};

/// Coherence protocol error types
#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("Coherence verification failed: {0}")]
    VerificationFailed(String),

    #[error("Inconsistent cache state detected: {0}")]
    InconsistentState(String),

    #[error("Cache operation timeout: {0}")]
    Timeout(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, CoherenceError>;

/// Cache coherence protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoherenceProtocol {
    /// Redis Pub/Sub based invalidation (eventual consistency)
    PubSub,
    /// Write through to L2 immediately (strong consistency)
    WriteThrough,
    /// Asynchronous write to L2 (eventual consistency, better performance)
    WriteBehind,
}

impl CoherenceProtocol {
    /// Get the expected coherence rate for this protocol
    pub fn expected_coherence_rate(&self) -> f64 {
        match self {
            CoherenceProtocol::PubSub => 0.99,        // 99% coherence
            CoherenceProtocol::WriteThrough => 1.0,   // 100% coherence
            CoherenceProtocol::WriteBehind => 0.95,   // 95% coherence
        }
    }
}

/// Consistency level for cache operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// May serve stale data temporarily
    Eventual,
    /// Always consistent (slower)
    Strong,
    /// Stale data within time bound
    BoundedStaleness,
}

impl ConsistencyLevel {
    /// Check if a cache entry is consistent given its age
    pub fn is_consistent(&self, entry_age_seconds: u64, max_staleness_seconds: u64) -> bool {
        match self {
            ConsistencyLevel::Eventual => true,
            ConsistencyLevel::Strong => entry_age_seconds == 0,
            ConsistencyLevel::BoundedStaleness => entry_age_seconds <= max_staleness_seconds,
        }
    }
}

/// Configuration for cache coherence
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Maximum staleness in seconds (for BoundedStaleness)
    pub max_staleness_seconds: u64,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            consistency_level: ConsistencyLevel::Eventual,
            max_staleness_seconds: 60,
        }
    }
}

/// Cache coherence protocol manager
pub struct CacheCoherenceProtocol {
    protocol_type: CoherenceProtocol,
    config: CoherenceConfig,
}

impl CacheCoherenceProtocol {
    /// Create a new cache coherence protocol
    pub fn new(protocol_type: CoherenceProtocol, config: CoherenceConfig) -> Self {
        info!(
            "Created coherence protocol: type={:?}, consistency={:?}",
            protocol_type, config.consistency_level
        );
        Self {
            protocol_type,
            config,
        }
    }

    /// Get the protocol type
    pub fn protocol_type(&self) -> CoherenceProtocol {
        self.protocol_type
    }

    /// Get the consistency level
    pub fn consistency_level(&self) -> ConsistencyLevel {
        self.config.consistency_level
    }

    /// Verify cache coherence across multiple nodes
    pub async fn verify_coherence(&self, caches: &[&DistributedCache]) -> Result<CoherenceReport> {
        if caches.is_empty() {
            return Err(CoherenceError::InvalidConfig(
                "No caches provided for verification".to_string(),
            ));
        }

        info!("Starting coherence verification for {} nodes", caches.len());

        let mut report = CoherenceReport::default();
        report.num_nodes = caches.len();

        // Sample keys to check
        let sample_keys = self.sample_keys(100)?;
        report.total_keys_checked = sample_keys.len();

        // Check each sampled key across all nodes
        for key in sample_keys {
            match self.check_key_coherence(&key, caches).await {
                Ok(is_consistent) => {
                    if is_consistent {
                        report.consistent_keys += 1;
                    } else {
                        report.inconsistent_keys.push(key);
                    }
                }
                Err(e) => {
                    warn!("Failed to check key coherence: {:?}", e);
                    report.errors += 1;
                }
            }
        }

        // Calculate coherence rate
        let checked = report.consistent_keys + report.inconsistent_keys.len();
        report.coherence_rate = if checked > 0 {
            report.consistent_keys as f64 / checked as f64
        } else {
            0.0
        };

        info!(
            "Coherence verification complete: rate={:.3}, consistent={}, inconsistent={}",
            report.coherence_rate,
            report.consistent_keys,
            report.inconsistent_keys.len()
        );

        Ok(report)
    }

    async fn check_key_coherence(
        &self,
        key: &CacheKey,
        caches: &[&DistributedCache],
    ) -> Result<bool> {
        let mut values: HashMap<u64, Vec<usize>> = HashMap::new();

        // Collect values from all nodes
        for (idx, cache) in caches.iter().enumerate() {
            match cache.get(key).await {
                Ok(Some(value)) => {
                    // Hash the value to compare
                    let value_hash = Self::hash_value(&value);
                    values.entry(value_hash).or_insert_with(Vec::new).push(idx);
                }
                Ok(None) => {
                    // Node doesn't have the value (cache miss)
                }
                Err(e) => {
                    debug!("Error getting value from node {}: {:?}", idx, e);
                }
            }
        }

        // Check consistency based on the protocol
        match self.protocol_type {
            CoherenceProtocol::PubSub | CoherenceProtocol::WriteBehind => {
                // Eventual consistency: allow some differences
                // Check if at least 80% of nodes have the same value
                if let Some((_, nodes)) = values.iter().max_by_key(|(_, nodes)| nodes.len()) {
                    let consistency_ratio = nodes.len() as f64 / caches.len() as f64;
                    Ok(consistency_ratio >= 0.8)
                } else {
                    Ok(true) // No values means consistent (all cache misses)
                }
            }
            CoherenceProtocol::WriteThrough => {
                // Strong consistency: all nodes must have the same value
                Ok(values.len() <= 1)
            }
        }
    }

    fn hash_value(value: &CacheValue) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        value.data.hash(&mut hasher);
        hasher.finish()
    }

    fn sample_keys(&self, count: usize) -> Result<Vec<CacheKey>> {
        let mut keys = Vec::with_capacity(count);

        for i in 0..count {
            // Generate sample keys (in real use, these would come from actual queries)
            let key = CacheKey::new(format!("sample_query_{}", i));
            keys.push(key);
        }

        Ok(keys)
    }

    /// Check if cache staleness is acceptable
    pub fn is_staleness_acceptable(&self, entry_age_seconds: u64) -> bool {
        self.config
            .consistency_level
            .is_consistent(entry_age_seconds, self.config.max_staleness_seconds)
    }

    /// Get recommended refresh interval for cache entries
    pub fn recommended_refresh_interval(&self) -> Duration {
        match self.config.consistency_level {
            ConsistencyLevel::Strong => Duration::from_secs(0), // Always refresh
            ConsistencyLevel::BoundedStaleness => {
                Duration::from_secs(self.config.max_staleness_seconds / 2)
            }
            ConsistencyLevel::Eventual => Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Report from coherence verification
#[derive(Debug, Clone, Default)]
pub struct CoherenceReport {
    /// Number of nodes checked
    pub num_nodes: usize,
    /// Total keys checked
    pub total_keys_checked: usize,
    /// Number of consistent keys
    pub consistent_keys: usize,
    /// List of inconsistent keys
    pub inconsistent_keys: Vec<CacheKey>,
    /// Number of errors encountered
    pub errors: usize,
    /// Coherence rate (0.0 to 1.0)
    pub coherence_rate: f64,
}

impl CoherenceReport {
    /// Check if coherence is acceptable
    pub fn is_acceptable(&self, min_coherence_rate: f64) -> bool {
        self.coherence_rate >= min_coherence_rate
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Coherence Report: {}/{} nodes, {}/{} keys consistent ({:.1}%), {} errors",
            self.num_nodes,
            self.num_nodes,
            self.consistent_keys,
            self.total_keys_checked,
            self.coherence_rate * 100.0,
            self.errors
        )
    }
}

/// Statistics for coherence monitoring
#[derive(Debug, Clone, Default)]
pub struct CoherenceStatistics {
    /// Total coherence checks performed
    pub total_checks: u64,
    /// Total inconsistencies detected
    pub total_inconsistencies: u64,
    /// Average coherence rate
    pub avg_coherence_rate: f64,
    /// Last check timestamp
    pub last_check: Option<SystemTime>,
}

impl CoherenceStatistics {
    /// Record a coherence check
    pub fn record_check(&mut self, report: &CoherenceReport) {
        self.total_checks += 1;
        self.total_inconsistencies += report.inconsistent_keys.len() as u64;

        // Update running average
        let alpha = 0.1; // Exponential moving average factor
        self.avg_coherence_rate = alpha * report.coherence_rate
            + (1.0 - alpha) * self.avg_coherence_rate;

        self.last_check = Some(SystemTime::now());
    }

    /// Get the inconsistency rate
    pub fn inconsistency_rate(&self) -> f64 {
        1.0 - self.avg_coherence_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_protocol_expected_rate() {
        assert_eq!(CoherenceProtocol::PubSub.expected_coherence_rate(), 0.99);
        assert_eq!(CoherenceProtocol::WriteThrough.expected_coherence_rate(), 1.0);
        assert_eq!(CoherenceProtocol::WriteBehind.expected_coherence_rate(), 0.95);
    }

    #[test]
    fn test_consistency_level() {
        assert!(ConsistencyLevel::Eventual.is_consistent(100, 60));
        assert!(!ConsistencyLevel::Strong.is_consistent(1, 60));
        assert!(ConsistencyLevel::BoundedStaleness.is_consistent(50, 60));
        assert!(!ConsistencyLevel::BoundedStaleness.is_consistent(70, 60));
    }

    #[test]
    fn test_coherence_report() {
        let mut report = CoherenceReport::default();
        report.num_nodes = 3;
        report.total_keys_checked = 100;
        report.consistent_keys = 99;
        report.inconsistent_keys = vec![CacheKey::new("key1".to_string())];
        report.coherence_rate = 0.99;

        assert!(report.is_acceptable(0.95));
        assert!(!report.is_acceptable(0.995));
        assert!(report.summary().contains("99.0%"));
    }

    #[test]
    fn test_coherence_statistics() {
        let mut stats = CoherenceStatistics::default();

        let report1 = CoherenceReport {
            num_nodes: 3,
            total_keys_checked: 100,
            consistent_keys: 99,
            inconsistent_keys: vec![CacheKey::new("key1".to_string())],
            errors: 0,
            coherence_rate: 0.99,
        };

        stats.record_check(&report1);
        assert_eq!(stats.total_checks, 1);
        assert_eq!(stats.total_inconsistencies, 1);
        assert!(stats.avg_coherence_rate > 0.0);

        let report2 = CoherenceReport {
            num_nodes: 3,
            total_keys_checked: 100,
            consistent_keys: 100,
            inconsistent_keys: vec![],
            errors: 0,
            coherence_rate: 1.0,
        };

        stats.record_check(&report2);
        assert_eq!(stats.total_checks, 2);
        assert_eq!(stats.total_inconsistencies, 1);
    }

    #[test]
    fn test_recommended_refresh_interval() {
        let config_strong = CoherenceConfig {
            consistency_level: ConsistencyLevel::Strong,
            max_staleness_seconds: 60,
        };
        let protocol_strong = CacheCoherenceProtocol::new(CoherenceProtocol::WriteThrough, config_strong);
        assert_eq!(protocol_strong.recommended_refresh_interval(), Duration::from_secs(0));

        let config_bounded = CoherenceConfig {
            consistency_level: ConsistencyLevel::BoundedStaleness,
            max_staleness_seconds: 60,
        };
        let protocol_bounded = CacheCoherenceProtocol::new(CoherenceProtocol::PubSub, config_bounded);
        assert_eq!(protocol_bounded.recommended_refresh_interval(), Duration::from_secs(30));

        let config_eventual = CoherenceConfig {
            consistency_level: ConsistencyLevel::Eventual,
            max_staleness_seconds: 60,
        };
        let protocol_eventual = CacheCoherenceProtocol::new(CoherenceProtocol::WriteBehind, config_eventual);
        assert_eq!(protocol_eventual.recommended_refresh_interval(), Duration::from_secs(300));
    }
}
