//! Store parameters and configuration builder
//!
//! Provides comprehensive configuration management inspired by Apache Jena TDB2's StoreParams.
//! Supports:
//! - Builder pattern for flexible configuration
//! - Parameter validation and constraints
//! - Serialization/deserialization for configuration files
//! - Default presets (development, production, performance)
//! - Parameter inheritance and overlays

use crate::error::{Result, TdbError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Store parameters containing all configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreParams {
    // Storage configuration
    /// Directory for storing RDF data
    pub data_dir: PathBuf,
    /// Page size in bytes (must be power of 2)
    pub page_size: usize,
    /// Buffer pool size (number of pages to cache)
    pub buffer_pool_size: usize,

    // Index configuration
    /// Enable Subject-Predicate-Object index
    pub enable_spo_index: bool,
    /// Enable Predicate-Object-Subject index
    pub enable_pos_index: bool,
    /// Enable Object-Subject-Predicate index
    pub enable_osp_index: bool,
    /// Enable quad (named graph) indexes
    pub enable_quad_indexes: bool,

    // Dictionary configuration
    /// Enable inline storage of small values
    pub enable_inline_values: bool,
    /// Enable prefix compression for URIs
    pub enable_prefix_compression: bool,
    /// Dictionary cache size (number of entries)
    pub dictionary_cache_size: usize,

    // Transaction configuration
    /// Enable write-ahead logging for durability
    pub enable_wal: bool,
    /// Write-ahead log buffer size in bytes
    pub wal_buffer_size: usize,
    /// Maximum transaction size (number of triples)
    pub max_transaction_size: usize,
    /// Transaction timeout in seconds
    pub transaction_timeout_secs: u64,

    // Compression configuration
    /// Enable data compression
    pub enable_compression: bool,
    /// Compression algorithm to use
    pub compression_algorithm: CompressionAlgorithm,
    /// Compression level (algorithm-specific)
    pub compression_level: u32,

    // Bloom filter configuration
    /// Enable bloom filters for indexes
    pub enable_bloom_filters: bool,
    /// Bloom filter false positive rate (0.0 to 1.0)
    pub bloom_filter_fpr: f64,
    /// Bloom filter size per index
    pub bloom_filter_size_per_index: usize,

    // Query optimization
    /// Enable query result caching
    pub enable_query_cache: bool,
    /// Query cache size (number of cached queries)
    pub query_cache_size: usize,
    /// Enable statistics collection
    pub enable_statistics: bool,
    /// Statistics sampling rate (0.0 to 1.0)
    pub statistics_sample_rate: f64,

    // Query monitoring
    /// Enable query performance monitoring
    pub enable_query_monitoring: bool,
    /// Slow query threshold in milliseconds
    pub slow_query_threshold_ms: u64,
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,

    // Spatial indexing
    /// Enable spatial indexing for geospatial queries
    pub enable_spatial_indexing: bool,
    /// Maximum entries per spatial index node
    pub spatial_index_max_entries: usize,

    // Production features
    /// Enable diagnostic logging and error reporting
    pub enable_diagnostics: bool,
    /// Enable metrics collection and export
    pub enable_metrics: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,

    // Connection pooling
    /// Minimum number of connections to maintain
    pub min_connections: usize,
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    // Backup configuration
    /// Enable online (hot) backups
    pub enable_online_backup: bool,
    /// Number of days to retain backups
    pub backup_retention_days: u32,
    /// Enable backup encryption
    pub enable_backup_encryption: bool,

    // Performance tuning
    /// Enable direct I/O bypassing OS cache
    pub enable_direct_io: bool,
    /// Enable asynchronous I/O operations
    pub enable_async_io: bool,
    /// Enable NUMA (Non-Uniform Memory Access) awareness
    pub enable_numa_awareness: bool,
    /// Enable GPU acceleration for computations
    pub enable_gpu_acceleration: bool,

    // Distributed systems
    /// Enable data replication
    pub enable_replication: bool,
    /// Replication mode (master-slave, master-master, etc.)
    pub replication_mode: ReplicationMode,
}

/// Compression algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 - fast compression
    Lz4,
    /// Zstandard - balanced compression
    Zstd,
    /// Brotli - high compression
    Brotli,
    /// Snappy - ultra-fast compression
    Snappy,
}

/// Replication mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationMode {
    /// No replication
    None,
    /// Master-slave replication
    MasterSlave,
    /// Master-master replication
    MasterMaster,
}

impl StoreParams {
    /// Create new store parameters with minimal configuration
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            // Storage configuration
            data_dir: data_dir.as_ref().to_path_buf(),
            page_size: 4096,
            buffer_pool_size: 1000,

            // Index configuration
            enable_spo_index: true,
            enable_pos_index: true,
            enable_osp_index: true,
            enable_quad_indexes: false,

            // Dictionary configuration
            enable_inline_values: true,
            enable_prefix_compression: true,
            dictionary_cache_size: 10000,

            // Transaction configuration
            enable_wal: true,
            wal_buffer_size: 1024 * 1024, // 1MB
            max_transaction_size: 1_000_000,
            transaction_timeout_secs: 60,

            // Compression configuration
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Lz4,
            compression_level: 3,

            // Bloom filter configuration
            enable_bloom_filters: true,
            bloom_filter_fpr: 0.01,
            bloom_filter_size_per_index: 1_000_000,

            // Query optimization
            enable_query_cache: true,
            query_cache_size: 1000,
            enable_statistics: true,
            statistics_sample_rate: 1.0,

            // Query monitoring
            enable_query_monitoring: true,
            slow_query_threshold_ms: 1000,
            query_timeout_ms: 30_000,

            // Spatial indexing
            enable_spatial_indexing: true,
            spatial_index_max_entries: 1000,

            // Production features
            enable_diagnostics: true,
            enable_metrics: true,
            enable_profiling: false,

            // Connection pooling
            min_connections: 2,
            max_connections: 10,
            connection_timeout_secs: 30,

            // Backup configuration
            enable_online_backup: true,
            backup_retention_days: 7,
            enable_backup_encryption: false,

            // Performance tuning
            enable_direct_io: false,
            enable_async_io: false,
            enable_numa_awareness: false,
            enable_gpu_acceleration: false,

            // Distributed systems
            enable_replication: false,
            replication_mode: ReplicationMode::None,
        }
    }

    /// Validate parameters and return errors if invalid
    pub fn validate(&self) -> Result<()> {
        // Page size must be power of 2 and >= 512
        if !self.page_size.is_power_of_two() || self.page_size < 512 {
            return Err(TdbError::InvalidConfiguration(format!(
                "Page size must be power of 2 and >= 512, got {}",
                self.page_size
            )));
        }

        // Buffer pool size must be > 0
        if self.buffer_pool_size == 0 {
            return Err(TdbError::InvalidConfiguration(
                "Buffer pool size must be > 0".to_string(),
            ));
        }

        // Bloom filter FPR must be between 0 and 1
        if self.bloom_filter_fpr <= 0.0 || self.bloom_filter_fpr >= 1.0 {
            return Err(TdbError::InvalidConfiguration(format!(
                "Bloom filter FPR must be between 0 and 1, got {}",
                self.bloom_filter_fpr
            )));
        }

        // Statistics sample rate must be between 0 and 1
        if self.statistics_sample_rate < 0.0 || self.statistics_sample_rate > 1.0 {
            return Err(TdbError::InvalidConfiguration(format!(
                "Statistics sample rate must be between 0 and 1, got {}",
                self.statistics_sample_rate
            )));
        }

        // Connection pool constraints
        if self.min_connections > self.max_connections {
            return Err(TdbError::InvalidConfiguration(format!(
                "Min connections ({}) > max connections ({})",
                self.min_connections, self.max_connections
            )));
        }

        Ok(())
    }

    /// Save parameters to JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TdbError::Serialization(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load parameters from JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let params: StoreParams = serde_json::from_str(&json)
            .map_err(|e| TdbError::Deserialization(format!("Failed to parse config: {}", e)))?;
        params.validate()?;
        Ok(params)
    }
}

/// Builder for store parameters with fluent API
pub struct StoreParamsBuilder {
    params: StoreParams,
}

impl StoreParamsBuilder {
    /// Create new builder with default parameters
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            params: StoreParams::new(data_dir),
        }
    }

    /// Create builder from existing parameters
    pub fn from_params(params: StoreParams) -> Self {
        Self { params }
    }

    /// Set page size (must be power of 2)
    pub fn page_size(mut self, size: usize) -> Self {
        self.params.page_size = size;
        self
    }

    /// Set buffer pool size
    pub fn buffer_pool_size(mut self, size: usize) -> Self {
        self.params.buffer_pool_size = size;
        self
    }

    /// Enable/disable triple indexes
    pub fn with_triple_indexes(mut self, spo: bool, pos: bool, osp: bool) -> Self {
        self.params.enable_spo_index = spo;
        self.params.enable_pos_index = pos;
        self.params.enable_osp_index = osp;
        self
    }

    /// Enable/disable quad indexes
    pub fn with_quad_indexes(mut self, enable: bool) -> Self {
        self.params.enable_quad_indexes = enable;
        self
    }

    /// Enable/disable inline values optimization
    pub fn with_inline_values(mut self, enable: bool) -> Self {
        self.params.enable_inline_values = enable;
        self
    }

    /// Enable/disable prefix compression
    pub fn with_prefix_compression(mut self, enable: bool) -> Self {
        self.params.enable_prefix_compression = enable;
        self
    }

    /// Set dictionary cache size
    pub fn dictionary_cache_size(mut self, size: usize) -> Self {
        self.params.dictionary_cache_size = size;
        self
    }

    /// Enable/disable write-ahead logging
    pub fn with_wal(mut self, enable: bool) -> Self {
        self.params.enable_wal = enable;
        self
    }

    /// Set WAL buffer size
    pub fn wal_buffer_size(mut self, size: usize) -> Self {
        self.params.wal_buffer_size = size;
        self
    }

    /// Set maximum transaction size
    pub fn max_transaction_size(mut self, size: usize) -> Self {
        self.params.max_transaction_size = size;
        self
    }

    /// Set transaction timeout
    pub fn transaction_timeout(mut self, seconds: u64) -> Self {
        self.params.transaction_timeout_secs = seconds;
        self
    }

    /// Configure compression
    pub fn with_compression(mut self, algorithm: CompressionAlgorithm, level: u32) -> Self {
        self.params.enable_compression = algorithm != CompressionAlgorithm::None;
        self.params.compression_algorithm = algorithm;
        self.params.compression_level = level;
        self
    }

    /// Configure bloom filters
    pub fn with_bloom_filters(mut self, enable: bool, fpr: f64, size: usize) -> Self {
        self.params.enable_bloom_filters = enable;
        self.params.bloom_filter_fpr = fpr;
        self.params.bloom_filter_size_per_index = size;
        self
    }

    /// Configure query cache
    pub fn with_query_cache(mut self, enable: bool, size: usize) -> Self {
        self.params.enable_query_cache = enable;
        self.params.query_cache_size = size;
        self
    }

    /// Configure statistics
    pub fn with_statistics(mut self, enable: bool, sample_rate: f64) -> Self {
        self.params.enable_statistics = enable;
        self.params.statistics_sample_rate = sample_rate;
        self
    }

    /// Configure query monitoring
    pub fn with_query_monitoring(
        mut self,
        enable: bool,
        slow_threshold_ms: u64,
        timeout_ms: u64,
    ) -> Self {
        self.params.enable_query_monitoring = enable;
        self.params.slow_query_threshold_ms = slow_threshold_ms;
        self.params.query_timeout_ms = timeout_ms;
        self
    }

    /// Configure spatial indexing
    pub fn with_spatial_indexing(mut self, enable: bool, max_entries: usize) -> Self {
        self.params.enable_spatial_indexing = enable;
        self.params.spatial_index_max_entries = max_entries;
        self
    }

    /// Configure production features
    pub fn with_production_features(
        mut self,
        diagnostics: bool,
        metrics: bool,
        profiling: bool,
    ) -> Self {
        self.params.enable_diagnostics = diagnostics;
        self.params.enable_metrics = metrics;
        self.params.enable_profiling = profiling;
        self
    }

    /// Configure connection pooling
    pub fn with_connection_pool(mut self, min: usize, max: usize, timeout_secs: u64) -> Self {
        self.params.min_connections = min;
        self.params.max_connections = max;
        self.params.connection_timeout_secs = timeout_secs;
        self
    }

    /// Configure backup
    pub fn with_backup(mut self, enable_online: bool, retention_days: u32, encrypt: bool) -> Self {
        self.params.enable_online_backup = enable_online;
        self.params.backup_retention_days = retention_days;
        self.params.enable_backup_encryption = encrypt;
        self
    }

    /// Configure performance features
    pub fn with_performance_features(
        mut self,
        direct_io: bool,
        async_io: bool,
        numa: bool,
        gpu: bool,
    ) -> Self {
        self.params.enable_direct_io = direct_io;
        self.params.enable_async_io = async_io;
        self.params.enable_numa_awareness = numa;
        self.params.enable_gpu_acceleration = gpu;
        self
    }

    /// Configure replication
    pub fn with_replication(mut self, mode: ReplicationMode) -> Self {
        self.params.enable_replication = mode != ReplicationMode::None;
        self.params.replication_mode = mode;
        self
    }

    /// Build and validate parameters
    pub fn build(self) -> Result<StoreParams> {
        self.params.validate()?;
        Ok(self.params)
    }

    /// Build without validation (use with caution)
    pub fn build_unchecked(self) -> StoreParams {
        self.params
    }
}

/// Preset configurations for common use cases
pub struct StorePresets;

impl StorePresets {
    /// Development preset - optimized for fast iteration
    pub fn development<P: AsRef<Path>>(data_dir: P) -> StoreParamsBuilder {
        StoreParamsBuilder::new(data_dir)
            .buffer_pool_size(100)
            .with_compression(CompressionAlgorithm::None, 0)
            .with_statistics(false, 0.0)
            .with_production_features(false, false, false)
            .with_performance_features(false, false, false, false)
    }

    /// Production preset - balanced for production workloads
    pub fn production<P: AsRef<Path>>(data_dir: P) -> StoreParamsBuilder {
        StoreParamsBuilder::new(data_dir)
            .buffer_pool_size(10000)
            .with_compression(CompressionAlgorithm::Lz4, 3)
            .with_statistics(true, 1.0)
            .with_production_features(true, true, false)
            .with_backup(true, 30, true)
            .with_connection_pool(5, 50, 60)
    }

    /// Performance preset - optimized for maximum throughput
    pub fn performance<P: AsRef<Path>>(data_dir: P) -> StoreParamsBuilder {
        StoreParamsBuilder::new(data_dir)
            .buffer_pool_size(50000)
            .with_compression(CompressionAlgorithm::Snappy, 1)
            .with_bloom_filters(true, 0.001, 10_000_000)
            .with_performance_features(true, true, true, true)
            .with_query_cache(true, 10000)
    }

    /// Minimal preset - bare minimum configuration
    pub fn minimal<P: AsRef<Path>>(data_dir: P) -> StoreParamsBuilder {
        StoreParamsBuilder::new(data_dir)
            .buffer_pool_size(10)
            .with_compression(CompressionAlgorithm::None, 0)
            .with_bloom_filters(false, 0.01, 0)
            .with_query_cache(false, 0)
            .with_statistics(false, 0.0)
            .with_production_features(false, false, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_store_params_new() {
        let params = StoreParams::new("/tmp/test");
        assert_eq!(params.page_size, 4096);
        assert_eq!(params.buffer_pool_size, 1000);
        assert!(params.enable_spo_index);
    }

    #[test]
    fn test_store_params_validation() {
        let params = StoreParams::new("/tmp/test");
        assert!(params.validate().is_ok());

        let mut invalid_params = params.clone();
        invalid_params.page_size = 1000; // Not power of 2
        assert!(invalid_params.validate().is_err());

        let mut invalid_params2 = params.clone();
        invalid_params2.bloom_filter_fpr = 1.5; // Out of range
        assert!(invalid_params2.validate().is_err());
    }

    #[test]
    fn test_builder_basic() {
        let params = StoreParamsBuilder::new("/tmp/test")
            .page_size(8192)
            .buffer_pool_size(2000)
            .build()
            .unwrap();

        assert_eq!(params.page_size, 8192);
        assert_eq!(params.buffer_pool_size, 2000);
    }

    #[test]
    fn test_builder_compression() {
        let params = StoreParamsBuilder::new("/tmp/test")
            .with_compression(CompressionAlgorithm::Zstd, 5)
            .build()
            .unwrap();

        assert!(params.enable_compression);
        assert_eq!(params.compression_algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(params.compression_level, 5);
    }

    #[test]
    fn test_builder_bloom_filters() {
        let params = StoreParamsBuilder::new("/tmp/test")
            .with_bloom_filters(true, 0.001, 5_000_000)
            .build()
            .unwrap();

        assert!(params.enable_bloom_filters);
        assert_eq!(params.bloom_filter_fpr, 0.001);
        assert_eq!(params.bloom_filter_size_per_index, 5_000_000);
    }

    #[test]
    fn test_presets_development() {
        let params = StorePresets::development("/tmp/test").build().unwrap();
        assert_eq!(params.buffer_pool_size, 100);
        assert_eq!(params.compression_algorithm, CompressionAlgorithm::None);
        assert!(!params.enable_statistics);
    }

    #[test]
    fn test_presets_production() {
        let params = StorePresets::production("/tmp/test").build().unwrap();
        assert_eq!(params.buffer_pool_size, 10000);
        assert!(params.enable_diagnostics);
        assert!(params.enable_backup_encryption);
    }

    #[test]
    fn test_presets_performance() {
        let params = StorePresets::performance("/tmp/test").build().unwrap();
        assert_eq!(params.buffer_pool_size, 50000);
        assert!(params.enable_direct_io);
        assert!(params.enable_gpu_acceleration);
    }

    #[test]
    fn test_save_load_params() {
        let temp_dir = env::temp_dir();
        let config_file = temp_dir.join("test_store_params.json");

        let params = StoreParamsBuilder::new(&temp_dir)
            .page_size(8192)
            .buffer_pool_size(5000)
            .build()
            .unwrap();

        params.save_to_file(&config_file).unwrap();
        let loaded_params = StoreParams::load_from_file(&config_file).unwrap();

        assert_eq!(params.page_size, loaded_params.page_size);
        assert_eq!(params.buffer_pool_size, loaded_params.buffer_pool_size);
    }

    #[test]
    fn test_replication_config() {
        let params = StoreParamsBuilder::new("/tmp/test")
            .with_replication(ReplicationMode::MasterSlave)
            .build()
            .unwrap();

        assert!(params.enable_replication);
        assert_eq!(params.replication_mode, ReplicationMode::MasterSlave);
    }

    #[test]
    fn test_connection_pool_validation() {
        let result = StoreParamsBuilder::new("/tmp/test")
            .with_connection_pool(10, 5, 30) // min > max (invalid)
            .build();

        assert!(result.is_err());
    }
}
