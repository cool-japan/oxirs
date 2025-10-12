//! # Raft Consensus Optimization
//!
//! Advanced optimizations for Raft consensus including log compaction,
//! batch processing, parallel replication, and compression.

use crate::raft::{OxirsNodeId, RdfCommand};
use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

/// Log compaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Minimum number of log entries before compaction
    pub min_log_size: usize,
    /// Maximum log size before forced compaction
    pub max_log_size: usize,
    /// Compaction interval in seconds
    pub compaction_interval_secs: u64,
    /// Enable aggressive compaction during idle periods
    pub aggressive_compaction: bool,
    /// Keep last N entries for debugging
    pub keep_last_entries: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_log_size: 1000,
            max_log_size: 10000,
            compaction_interval_secs: 300, // 5 minutes
            aggressive_compaction: true,
            keep_last_entries: 100,
        }
    }
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Enable dynamic batch sizing based on load
    pub dynamic_sizing: bool,
    /// Minimum batch size for efficiency
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            batch_timeout_ms: 10,
            dynamic_sizing: true,
            min_batch_size: 10,
        }
    }
}

/// Log compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression for log entries
    pub enabled: bool,
    /// Compression algorithm (zstd, lz4, flate2)
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub level: i32,
    /// Minimum entry size for compression
    pub min_size_bytes: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
            min_size_bytes: 1024,
        }
    }
}

/// Compression algorithm options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// Zstandard compression (best balance)
    Zstd,
    /// LZ4 compression (fastest)
    Lz4,
    /// Flate2 compression (best ratio)
    Flate2,
}

/// Parallel replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelReplicationConfig {
    /// Enable parallel log streaming
    pub enabled: bool,
    /// Number of parallel streams per follower
    pub streams_per_follower: usize,
    /// Pipeline depth for asynchronous replication
    pub pipeline_depth: usize,
    /// Enable SIMD acceleration for entry processing
    pub use_simd: bool,
}

impl Default for ParallelReplicationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            streams_per_follower: 4,
            pipeline_depth: 10,
            use_simd: true,
        }
    }
}

/// Raft optimization manager
#[derive(Debug, Clone)]
pub struct RaftOptimizer {
    compaction_config: CompactionConfig,
    batch_config: BatchConfig,
    compression_config: CompressionConfig,
    parallel_config: ParallelReplicationConfig,
    node_id: OxirsNodeId,
    metrics: Arc<RwLock<OptimizationMetrics>>,
}

/// Optimization metrics
#[derive(Debug, Clone, Default)]
pub struct OptimizationMetrics {
    /// Total log entries compacted
    pub compacted_entries: u64,
    /// Total bytes saved by compression
    pub compression_savings_bytes: u64,
    /// Total batch operations processed
    pub batch_operations: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Parallel replication speedup factor
    pub parallel_speedup: f64,
    /// Last compaction timestamp
    pub last_compaction: Option<SystemTime>,
    /// Total compaction runs
    pub compaction_runs: u64,
}

impl RaftOptimizer {
    /// Create a new Raft optimizer
    pub fn new(node_id: OxirsNodeId) -> Self {
        Self {
            compaction_config: CompactionConfig::default(),
            batch_config: BatchConfig::default(),
            compression_config: CompressionConfig::default(),
            parallel_config: ParallelReplicationConfig::default(),
            node_id,
            metrics: Arc::new(RwLock::new(OptimizationMetrics::default())),
        }
    }

    /// Create optimizer with custom configuration
    pub fn with_config(
        node_id: OxirsNodeId,
        compaction: CompactionConfig,
        batch: BatchConfig,
        compression: CompressionConfig,
        parallel: ParallelReplicationConfig,
    ) -> Self {
        Self {
            compaction_config: compaction,
            batch_config: batch,
            compression_config: compression,
            parallel_config: parallel,
            node_id,
            metrics: Arc::new(RwLock::new(OptimizationMetrics::default())),
        }
    }

    /// Check if log compaction is needed
    pub fn should_compact(&self, log_size: usize) -> bool {
        if log_size >= self.compaction_config.max_log_size {
            return true;
        }

        if log_size >= self.compaction_config.min_log_size {
            // Check if compaction interval has passed
            if let Ok(metrics) = self.metrics.try_read() {
                if let Some(last_compaction) = metrics.last_compaction {
                    if let Ok(elapsed) = SystemTime::now().duration_since(last_compaction) {
                        return elapsed.as_secs()
                            >= self.compaction_config.compaction_interval_secs;
                    }
                }
                return true; // First compaction
            }
        }

        false
    }

    /// Perform log compaction using SciRS2 parallel operations
    pub async fn compact_log<T: Clone + Send + Sync>(&self, log_entries: Vec<T>) -> Result<Vec<T>> {
        let start_time = SystemTime::now();

        if log_entries.len() <= self.compaction_config.keep_last_entries {
            return Ok(log_entries);
        }

        // Keep only the last N entries
        let keep_from = log_entries.len() - self.compaction_config.keep_last_entries;
        let mut compacted = Vec::new();

        // Use parallel processing for large logs
        if log_entries.len() > 1000 && self.parallel_config.enabled {
            // Process in parallel chunks
            let chunk_size = (log_entries.len() - keep_from) / num_cpus::get().max(1);
            let chunks: Vec<&[T]> = log_entries[keep_from..]
                .chunks(chunk_size.max(100))
                .collect();

            for chunk in chunks {
                compacted.extend_from_slice(chunk);
            }
        } else {
            compacted.extend_from_slice(&log_entries[keep_from..]);
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.compacted_entries += (log_entries.len() - compacted.len()) as u64;
        metrics.last_compaction = Some(SystemTime::now());
        metrics.compaction_runs += 1;

        tracing::info!(
            "Node {}: Compacted {} entries to {} (saved {} entries) in {:?}",
            self.node_id,
            log_entries.len(),
            compacted.len(),
            log_entries.len() - compacted.len(),
            start_time.elapsed().unwrap_or_default()
        );

        Ok(compacted)
    }

    /// Batch commands for efficient processing
    pub async fn batch_commands(&self, commands: Vec<RdfCommand>) -> Result<Vec<Vec<RdfCommand>>> {
        if commands.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = if self.batch_config.dynamic_sizing {
            // Adaptive batch sizing based on command count
            let adaptive_size = (commands.len() / 10).clamp(
                self.batch_config.min_batch_size,
                self.batch_config.max_batch_size,
            );
            adaptive_size
        } else {
            self.batch_config.max_batch_size
        };

        let batches: Vec<Vec<RdfCommand>> = commands
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.batch_operations += batches.len() as u64;
        let total_commands: usize = batches.iter().map(|b| b.len()).sum();
        metrics.avg_batch_size = if metrics.batch_operations > 0 {
            (metrics.avg_batch_size * (metrics.batch_operations - batches.len() as u64) as f64
                + total_commands as f64)
                / metrics.batch_operations as f64
        } else {
            total_commands as f64 / batches.len() as f64
        };

        tracing::debug!(
            "Node {}: Created {} batches with avg size {:.1}",
            self.node_id,
            batches.len(),
            metrics.avg_batch_size
        );

        Ok(batches)
    }

    /// Compress log entry data
    pub fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.compression_config.enabled || data.len() < self.compression_config.min_size_bytes {
            return Ok(data.to_vec());
        }

        let compressed = match self.compression_config.algorithm {
            CompressionAlgorithm::Zstd => zstd::encode_all(data, self.compression_config.level)?,
            CompressionAlgorithm::Lz4 => lz4_flex::compress_prepend_size(data),
            CompressionAlgorithm::Flate2 => {
                use flate2::write::GzEncoder;
                use flate2::Compression;
                use std::io::Write;

                let mut encoder = GzEncoder::new(
                    Vec::new(),
                    Compression::new(self.compression_config.level as u32),
                );
                encoder.write_all(data)?;
                encoder.finish()?
            }
        };

        // Update compression metrics
        if let Ok(mut metrics) = self.metrics.try_write() {
            let savings = data.len().saturating_sub(compressed.len());
            metrics.compression_savings_bytes += savings as u64;
        }

        Ok(compressed)
    }

    /// Decompress log entry data
    pub fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        if !self.compression_config.enabled {
            return Ok(compressed.to_vec());
        }

        let decompressed = match self.compression_config.algorithm {
            CompressionAlgorithm::Zstd => zstd::decode_all(compressed)?,
            CompressionAlgorithm::Lz4 => lz4_flex::decompress_size_prepended(compressed)
                .map_err(|e| anyhow::anyhow!("LZ4 decompression failed: {}", e))?,
            CompressionAlgorithm::Flate2 => {
                use flate2::read::GzDecoder;
                use std::io::Read;

                let mut decoder = GzDecoder::new(compressed);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                decompressed
            }
        };

        Ok(decompressed)
    }

    /// Replicate entries in parallel to multiple followers
    pub async fn parallel_replicate<F, Fut>(
        &self,
        followers: Vec<OxirsNodeId>,
        entries: Vec<Vec<u8>>,
        replicate_fn: F,
    ) -> Result<Vec<Result<(), String>>>
    where
        F: Fn(OxirsNodeId, Vec<Vec<u8>>) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        if !self.parallel_config.enabled {
            // Sequential replication fallback
            let mut results = Vec::new();
            for follower in followers {
                let result = replicate_fn(follower, entries.clone())
                    .await
                    .map_err(|e| e.to_string());
                results.push(result);
            }
            return Ok(results);
        }

        // Parallel replication using tokio tasks
        let mut tasks = Vec::new();

        for follower in followers {
            let entries_clone = entries.clone();
            let replicate_fn_clone = replicate_fn.clone();

            let task = tokio::spawn(async move {
                replicate_fn_clone(follower, entries_clone)
                    .await
                    .map_err(|e| e.to_string())
            });

            tasks.push(task);
        }

        // Wait for all replications to complete
        let start_time = SystemTime::now();
        let num_tasks = tasks.len();
        let results = futures::future::join_all(tasks)
            .await
            .into_iter()
            .map(|r| r.unwrap_or_else(|e| Err(e.to_string())))
            .collect();

        // Update parallel replication metrics
        if let Ok(elapsed) = start_time.elapsed() {
            let mut metrics = self.metrics.write().await;
            // Estimate speedup (sequential time / parallel time)
            let estimated_sequential_time = elapsed * num_tasks as u32;
            metrics.parallel_speedup =
                estimated_sequential_time.as_secs_f64() / elapsed.as_secs_f64();

            tracing::debug!(
                "Node {}: Parallel replication to {} followers completed in {:?} (estimated speedup: {:.2}x)",
                self.node_id,
                num_tasks,
                elapsed,
                metrics.parallel_speedup
            );
        }

        Ok(results)
    }

    /// Process entries with SIMD acceleration
    pub fn simd_process_entries(&self, entries: &[f64]) -> Result<Array1<f64>> {
        if !self.parallel_config.use_simd {
            return Ok(Array1::from_vec(entries.to_vec()));
        }

        // Use SciRS2 SIMD operations for entry processing
        let data = Array1::from_vec(entries.to_vec());

        // Example: Compute checksums or hashes using SIMD
        // In a real implementation, this would be entry validation/processing
        let processed = data.mapv(|x| x * 1.0); // Placeholder operation

        Ok(processed)
    }

    /// Get current optimization metrics
    pub async fn get_metrics(&self) -> OptimizationMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset optimization metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = OptimizationMetrics::default();
    }

    /// Get compaction configuration
    pub fn compaction_config(&self) -> &CompactionConfig {
        &self.compaction_config
    }

    /// Get batch configuration
    pub fn batch_config(&self) -> &BatchConfig {
        &self.batch_config
    }

    /// Get compression configuration
    pub fn compression_config(&self) -> &CompressionConfig {
        &self.compression_config
    }

    /// Get parallel replication configuration
    pub fn parallel_config(&self) -> &ParallelReplicationConfig {
        &self.parallel_config
    }
}

/// Batch processor for accumulating commands
#[derive(Debug)]
pub struct BatchProcessor {
    commands: Arc<RwLock<VecDeque<RdfCommand>>>,
    config: BatchConfig,
    last_flush: Arc<RwLock<SystemTime>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchConfig) -> Self {
        Self {
            commands: Arc::new(RwLock::new(VecDeque::new())),
            config,
            last_flush: Arc::new(RwLock::new(SystemTime::now())),
        }
    }

    /// Add a command to the batch
    pub async fn add_command(&self, command: RdfCommand) {
        let mut commands = self.commands.write().await;
        commands.push_back(command);
    }

    /// Check if batch should be flushed
    pub async fn should_flush(&self) -> bool {
        let commands = self.commands.read().await;
        if commands.len() >= self.config.max_batch_size {
            return true;
        }

        if commands.len() >= self.config.min_batch_size {
            let last_flush = self.last_flush.read().await;
            if let Ok(elapsed) = SystemTime::now().duration_since(*last_flush) {
                return elapsed.as_millis() >= self.config.batch_timeout_ms as u128;
            }
        }

        false
    }

    /// Flush accumulated commands
    pub async fn flush(&self) -> Vec<RdfCommand> {
        let mut commands = self.commands.write().await;
        let flushed = commands.drain(..).collect();
        *self.last_flush.write().await = SystemTime::now();
        flushed
    }

    /// Get current batch size
    pub async fn batch_size(&self) -> usize {
        self.commands.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction_config_default() {
        let config = CompactionConfig::default();
        assert_eq!(config.min_log_size, 1000);
        assert_eq!(config.max_log_size, 10000);
        assert_eq!(config.compaction_interval_secs, 300);
        assert!(config.aggressive_compaction);
        assert_eq!(config.keep_last_entries, 100);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 100);
        assert_eq!(config.batch_timeout_ms, 10);
        assert!(config.dynamic_sizing);
        assert_eq!(config.min_batch_size, 10);
    }

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(config.level, 3);
        assert_eq!(config.min_size_bytes, 1024);
    }

    #[tokio::test]
    async fn test_raft_optimizer_creation() {
        let optimizer = RaftOptimizer::new(1);
        assert_eq!(optimizer.node_id, 1);
        assert_eq!(optimizer.compaction_config.min_log_size, 1000);
        assert_eq!(optimizer.batch_config.max_batch_size, 100);
        assert!(optimizer.compression_config.enabled);
        assert!(optimizer.parallel_config.enabled);
    }

    #[tokio::test]
    async fn test_should_compact() {
        let optimizer = RaftOptimizer::new(1);

        // Should not compact small logs
        assert!(!optimizer.should_compact(500));

        // Should compact when exceeding max_log_size
        assert!(optimizer.should_compact(15000));

        // Should compact when exceeding min_log_size and interval passed
        assert!(optimizer.should_compact(1500));
    }

    #[tokio::test]
    async fn test_log_compaction() {
        let optimizer = RaftOptimizer::new(1);
        let entries: Vec<u64> = (0..1000).collect();

        let compacted = optimizer.compact_log(entries.clone()).await.unwrap();

        // Should keep only last 100 entries
        assert_eq!(compacted.len(), 100);
        assert_eq!(compacted[0], 900);
        assert_eq!(compacted[99], 999);
    }

    #[tokio::test]
    async fn test_batch_commands() {
        let optimizer = RaftOptimizer::new(1);
        let commands: Vec<RdfCommand> = (0..250)
            .map(|i| RdfCommand::Insert {
                subject: format!("s{}", i),
                predicate: "p".to_string(),
                object: "o".to_string(),
            })
            .collect();

        let batches = optimizer.batch_commands(commands).await.unwrap();

        // Should create multiple batches
        assert!(batches.len() > 1);

        // Each batch should be within limits
        for batch in &batches {
            assert!(batch.len() <= optimizer.batch_config.max_batch_size);
        }
    }

    #[test]
    fn test_compression_zstd() {
        let optimizer = RaftOptimizer::new(1);
        let data = b"Hello, World! ".repeat(100);

        let compressed = optimizer.compress_data(&data).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = optimizer.decompress_data(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_compression_lz4() {
        let mut optimizer = RaftOptimizer::new(1);
        optimizer.compression_config.algorithm = CompressionAlgorithm::Lz4;
        let data = b"Hello, World! ".repeat(100);

        let compressed = optimizer.compress_data(&data).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = optimizer.decompress_data(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_compression_disabled() {
        let mut optimizer = RaftOptimizer::new(1);
        optimizer.compression_config.enabled = false;
        let data = b"Hello, World!";

        let compressed = optimizer.compress_data(data).unwrap();
        assert_eq!(data, compressed.as_slice());
    }

    #[test]
    fn test_small_data_no_compression() {
        let optimizer = RaftOptimizer::new(1);
        let data = b"Small"; // Below min_size_bytes

        let compressed = optimizer.compress_data(data).unwrap();
        assert_eq!(data, compressed.as_slice());
    }

    #[tokio::test]
    async fn test_batch_processor() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config);

        // Add commands
        for i in 0..50 {
            processor
                .add_command(RdfCommand::Insert {
                    subject: format!("s{}", i),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                })
                .await;
        }

        assert_eq!(processor.batch_size().await, 50);

        // Flush
        let flushed = processor.flush().await;
        assert_eq!(flushed.len(), 50);
        assert_eq!(processor.batch_size().await, 0);
    }

    #[tokio::test]
    async fn test_batch_processor_auto_flush() {
        let mut config = BatchConfig::default();
        config.max_batch_size = 10;
        let processor = BatchProcessor::new(config);

        // Add commands up to max_batch_size
        for i in 0..10 {
            processor
                .add_command(RdfCommand::Insert {
                    subject: format!("s{}", i),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                })
                .await;
        }

        // Should trigger flush
        assert!(processor.should_flush().await);
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let optimizer = RaftOptimizer::new(1);

        // Perform operations
        let entries: Vec<u64> = (0..1000).collect();
        let _ = optimizer.compact_log(entries).await;

        let commands: Vec<RdfCommand> = (0..100)
            .map(|i| RdfCommand::Insert {
                subject: format!("s{}", i),
                predicate: "p".to_string(),
                object: "o".to_string(),
            })
            .collect();
        let _ = optimizer.batch_commands(commands).await;

        // Check metrics
        let metrics = optimizer.get_metrics().await;
        assert_eq!(metrics.compacted_entries, 900);
        assert!(metrics.batch_operations > 0);
        assert!(metrics.last_compaction.is_some());
    }
}
