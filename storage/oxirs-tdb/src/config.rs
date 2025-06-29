//! # TDB Configuration and Performance Tuning
//!
//! Comprehensive configuration system for OxiRS TDB with performance tuning,
//! monitoring integration, and adaptive optimization capabilities.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

// External dependencies
extern crate num_cpus;
extern crate toml;

/// Main TDB configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdbConfig {
    /// Storage configuration
    pub storage: StorageConfig,
    /// Transaction configuration
    pub transactions: TransactionConfig,
    /// Performance tuning
    pub performance: PerformanceConfig,
    /// Monitoring and logging
    pub monitoring: MonitoringConfig,
    /// Advanced features
    pub advanced: AdvancedConfig,
}

impl Default for TdbConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            transactions: TransactionConfig::default(),
            performance: PerformanceConfig::default(),
            monitoring: MonitoringConfig::default(),
            advanced: AdvancedConfig::default(),
        }
    }
}

/// Storage-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Database directory path
    pub database_path: PathBuf,
    /// Page size in bytes (default: 8KB)
    pub page_size: u32,
    /// Maximum file size before splitting (default: 2GB)
    pub max_file_size: u64,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Enable checksum verification
    pub enable_checksums: bool,
    /// Enable memory mapping
    pub enable_memory_mapping: bool,
    /// Sync strategy for durability
    pub sync_strategy: SyncStrategy,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("./tdb-data"),
            page_size: 8192,              // 8KB
            max_file_size: 2_147_483_648, // 2GB
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::LZ4,
            enable_checksums: true,
            enable_memory_mapping: true,
            sync_strategy: SyncStrategy::Async,
        }
    }
}

/// Transaction management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionConfig {
    /// Enable MVCC (Multi-Version Concurrency Control)
    pub enable_mvcc: bool,
    /// Transaction timeout in seconds
    pub transaction_timeout_secs: u64,
    /// Maximum number of concurrent transactions
    pub max_concurrent_transactions: u32,
    /// Lock timeout in milliseconds
    pub lock_timeout_ms: u64,
    /// Enable deadlock detection
    pub enable_deadlock_detection: bool,
    /// WAL (Write-Ahead Logging) configuration
    pub wal_config: WalConfig,
    /// Snapshot isolation level
    pub isolation_level: IsolationLevel,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            enable_mvcc: true,
            transaction_timeout_secs: 300, // 5 minutes
            max_concurrent_transactions: 100,
            lock_timeout_ms: 5000, // 5 seconds
            enable_deadlock_detection: true,
            wal_config: WalConfig::default(),
            isolation_level: IsolationLevel::SnapshotIsolation,
        }
    }
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Buffer pool configuration
    pub buffer_pool: BufferPoolConfig,
    /// Index configuration
    pub index_config: IndexConfig,
    /// Query optimization settings
    pub query_optimization: QueryOptimizationConfig,
    /// Thread pool configuration
    pub thread_pool: ThreadPoolConfig,
    /// Cache configuration
    pub cache_config: CacheConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            buffer_pool: BufferPoolConfig::default(),
            index_config: IndexConfig::default(),
            query_optimization: QueryOptimizationConfig::default(),
            thread_pool: ThreadPoolConfig::default(),
            cache_config: CacheConfig::default(),
        }
    }
}

/// Monitoring and observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable performance metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,
    /// Enable query logging
    pub enable_query_logging: bool,
    /// Enable slow query detection
    pub enable_slow_query_detection: bool,
    /// Slow query threshold in milliseconds
    pub slow_query_threshold_ms: u64,
    /// Enable health checks
    pub enable_health_checks: bool,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Export metrics to external systems
    pub metrics_export: MetricsExportConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_secs: 30,
            enable_query_logging: false,
            enable_slow_query_detection: true,
            slow_query_threshold_ms: 1000,
            enable_health_checks: true,
            health_check_interval_secs: 60,
            metrics_export: MetricsExportConfig::default(),
        }
    }
}

/// Advanced features configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// Enable experimental features
    pub enable_experimental_features: bool,
    /// Enable statistics collection for query optimization
    pub enable_statistics: bool,
    /// Statistics update frequency
    pub statistics_update_frequency: StatisticsFrequency,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Feature flags for optional functionality
    pub feature_flags: HashMap<String, bool>,
    /// Custom properties for extensions
    pub custom_properties: HashMap<String, String>,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        let mut feature_flags = HashMap::new();
        feature_flags.insert("bitmap_indexes".to_string(), true);
        feature_flags.insert("hash_indexes".to_string(), true);
        feature_flags.insert("column_store".to_string(), false);
        feature_flags.insert("distributed_mode".to_string(), false);

        Self {
            enable_experimental_features: false,
            enable_statistics: true,
            statistics_update_frequency: StatisticsFrequency::Adaptive,
            enable_adaptive_optimization: true,
            feature_flags,
            custom_properties: HashMap::new(),
        }
    }
}

/// Buffer pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferPoolConfig {
    /// Size in MB (default: 256MB)
    pub size_mb: u64,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable prefetching
    pub enable_prefetching: bool,
    /// Prefetch size in pages
    pub prefetch_size: u32,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            size_mb: 256,
            eviction_policy: EvictionPolicy::LRU,
            enable_prefetching: true,
            prefetch_size: 8,
        }
    }
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// B+ tree node size
    pub btree_node_size: u32,
    /// Enable bitmap indexes
    pub enable_bitmap_indexes: bool,
    /// Enable hash indexes for equality queries
    pub enable_hash_indexes: bool,
    /// Index rebuild threshold (percentage of changes)
    pub rebuild_threshold_percent: f32,
    /// Enable index compression
    pub enable_index_compression: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            btree_node_size: 512,
            enable_bitmap_indexes: true,
            enable_hash_indexes: true,
            rebuild_threshold_percent: 25.0,
            enable_index_compression: true,
        }
    }
}

/// Query optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimizationConfig {
    /// Enable cost-based optimization
    pub enable_cost_based_optimization: bool,
    /// Enable query plan caching
    pub enable_plan_caching: bool,
    /// Plan cache size (number of plans)
    pub plan_cache_size: u32,
    /// Enable join reordering
    pub enable_join_reordering: bool,
    /// Maximum optimization time in milliseconds
    pub max_optimization_time_ms: u64,
}

impl Default for QueryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_cost_based_optimization: true,
            enable_plan_caching: true,
            plan_cache_size: 1000,
            enable_join_reordering: true,
            max_optimization_time_ms: 100,
        }
    }
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: u32,
    /// Queue size for pending tasks
    pub queue_size: u32,
    /// Thread idle timeout in seconds
    pub idle_timeout_secs: u64,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            worker_threads: 0, // Auto-detect
            queue_size: 10000,
            idle_timeout_secs: 60,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Query result cache size in MB
    pub result_cache_size_mb: u64,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Enable adaptive cache sizing
    pub enable_adaptive_sizing: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            result_cache_size_mb: 64,
            cache_ttl_secs: 300,
            enable_adaptive_sizing: true,
        }
    }
}

/// WAL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfig {
    /// Enable WAL
    pub enabled: bool,
    /// WAL file size in MB
    pub file_size_mb: u64,
    /// Checkpoint interval in seconds
    pub checkpoint_interval_secs: u64,
    /// Force sync on commit
    pub force_sync_on_commit: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            file_size_mb: 64,
            checkpoint_interval_secs: 300,
            force_sync_on_commit: false,
        }
    }
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Enable Prometheus metrics export
    pub enable_prometheus: bool,
    /// Prometheus endpoint
    pub prometheus_endpoint: String,
    /// Enable OpenTelemetry
    pub enable_opentelemetry: bool,
    /// OpenTelemetry endpoint
    pub opentelemetry_endpoint: String,
}

impl Default for MetricsExportConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: false,
            prometheus_endpoint: "localhost:9090".to_string(),
            enable_opentelemetry: false,
            opentelemetry_endpoint: "localhost:4317".to_string(),
        }
    }
}

// Enums for configuration options

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    LZ4,
    Zstd,
    Snappy,
    Gzip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStrategy {
    /// Asynchronous sync (better performance)
    Async,
    /// Sync on transaction commit
    SyncOnCommit,
    /// Sync on every write (highest durability)
    SyncAlways,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
    SnapshotIsolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticsFrequency {
    Never,
    OnStartup,
    Hourly,
    Daily,
    Weekly,
    Adaptive,
}

impl TdbConfig {
    /// Load configuration from a file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path)?;
        let config: TdbConfig = if path.as_ref().extension() == Some(std::ffi::OsStr::new("toml")) {
            toml::from_str(&content)?
        } else {
            serde_json::from_str(&content)?
        };
        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = if path.as_ref().extension() == Some(std::ffi::OsStr::new("toml")) {
            toml::to_string_pretty(self)?
        } else {
            serde_json::to_string_pretty(self)?
        };
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration and return any issues
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Validate storage configuration
        if self.storage.page_size < 1024 {
            warnings.push("Page size below 1KB may impact performance".to_string());
        }
        if self.storage.page_size > 65536 {
            warnings.push("Page size above 64KB may waste memory".to_string());
        }

        // Validate buffer pool size
        if self.performance.buffer_pool.size_mb < 16 {
            warnings.push("Buffer pool size below 16MB may impact performance".to_string());
        }

        // Validate transaction settings
        if self.transactions.max_concurrent_transactions > 1000 {
            warnings.push("High concurrent transaction limit may impact performance".to_string());
        }

        // Check for conflicting settings
        if !self.storage.enable_memory_mapping && self.performance.buffer_pool.size_mb > 1024 {
            warnings
                .push("Large buffer pool without memory mapping may be inefficient".to_string());
        }

        Ok(warnings)
    }

    /// Get optimized configuration for specific workload types
    pub fn optimized_for(workload: WorkloadType) -> Self {
        let mut config = Self::default();

        match workload {
            WorkloadType::AnalyticalQueries => {
                // Optimize for complex analytical queries
                config.performance.buffer_pool.size_mb = 512;
                config.performance.index_config.enable_bitmap_indexes = true;
                config.storage.enable_compression = true;
                config
                    .performance
                    .query_optimization
                    .enable_cost_based_optimization = true;
                config.performance.cache_config.result_cache_size_mb = 128;
            }
            WorkloadType::TransactionalWorkload => {
                // Optimize for high-throughput transactions
                config.transactions.max_concurrent_transactions = 200;
                config.storage.sync_strategy = SyncStrategy::SyncOnCommit;
                config.performance.buffer_pool.size_mb = 256;
                config.transactions.lock_timeout_ms = 1000;
            }
            WorkloadType::ReadHeavy => {
                // Optimize for read-heavy workloads
                config.performance.buffer_pool.size_mb = 1024;
                config.performance.cache_config.result_cache_size_mb = 256;
                config.performance.index_config.enable_hash_indexes = true;
                config.storage.enable_memory_mapping = true;
            }
            WorkloadType::WriteHeavy => {
                // Optimize for write-heavy workloads
                config.storage.sync_strategy = SyncStrategy::Async;
                config.transactions.wal_config.checkpoint_interval_secs = 60;
                config.performance.buffer_pool.size_mb = 128;
                config.storage.enable_compression = false; // Reduce CPU overhead
            }
            WorkloadType::MixedWorkload => {
                // Balanced configuration
                config.performance.buffer_pool.size_mb = 384;
                config.transactions.max_concurrent_transactions = 150;
                config.performance.cache_config.result_cache_size_mb = 96;
                config.storage.sync_strategy = SyncStrategy::SyncOnCommit;
            }
        }

        config
    }

    /// Auto-tune configuration based on system resources
    pub fn auto_tune(&mut self) -> Result<()> {
        // Get system information (simplified - in practice would use system APIs)
        let total_memory_mb = self.get_system_memory_mb()?;
        let cpu_cores = self.get_cpu_cores()?;

        // Auto-tune buffer pool (use 25% of system memory, capped at 2GB)
        self.performance.buffer_pool.size_mb = (total_memory_mb / 4).min(2048);

        // Auto-tune thread pool
        if self.performance.thread_pool.worker_threads == 0 {
            self.performance.thread_pool.worker_threads = cpu_cores;
        }

        // Auto-tune based on available memory
        if total_memory_mb > 8192 {
            // High-memory system - enable more caching
            self.performance.cache_config.result_cache_size_mb = 256;
            self.performance.query_optimization.plan_cache_size = 2000;
        } else if total_memory_mb < 2048 {
            // Low-memory system - reduce caching
            self.performance.cache_config.result_cache_size_mb = 32;
            self.performance.query_optimization.plan_cache_size = 500;
            self.performance.buffer_pool.size_mb = self.performance.buffer_pool.size_mb.min(128);
        }

        Ok(())
    }

    // Helper methods for system information (simplified implementations)
    fn get_system_memory_mb(&self) -> Result<u64> {
        // In practice, would use system APIs to get actual memory
        Ok(8192) // Default to 8GB
    }

    fn get_cpu_cores(&self) -> Result<u32> {
        Ok(num_cpus::get() as u32)
    }
}

/// Workload types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    AnalyticalQueries,
    TransactionalWorkload,
    ReadHeavy,
    WriteHeavy,
    MixedWorkload,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TdbConfig::default();
        assert!(config.storage.enable_compression);
        assert!(config.transactions.enable_mvcc);
        assert_eq!(config.storage.page_size, 8192);
    }

    #[test]
    fn test_config_validation() {
        let config = TdbConfig::default();
        let warnings = config.validate().unwrap();
        assert!(warnings.is_empty()); // Default config should be valid
    }

    #[test]
    fn test_optimized_configurations() {
        let analytical_config = TdbConfig::optimized_for(WorkloadType::AnalyticalQueries);
        assert!(
            analytical_config
                .performance
                .index_config
                .enable_bitmap_indexes
        );
        assert_eq!(analytical_config.performance.buffer_pool.size_mb, 512);

        let transactional_config = TdbConfig::optimized_for(WorkloadType::TransactionalWorkload);
        assert_eq!(
            transactional_config
                .transactions
                .max_concurrent_transactions,
            200
        );
    }

    #[test]
    fn test_auto_tune() {
        let mut config = TdbConfig::default();
        config.auto_tune().unwrap();

        // Should have set reasonable values
        assert!(config.performance.buffer_pool.size_mb > 0);
        assert!(config.performance.thread_pool.worker_threads > 0);
    }

    #[test]
    fn test_serialization() {
        let config = TdbConfig::default();

        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TdbConfig = serde_json::from_str(&json).unwrap();

        // Basic sanity check
        assert_eq!(config.storage.page_size, deserialized.storage.page_size);
    }
}
