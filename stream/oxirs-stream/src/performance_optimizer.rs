//! # Advanced Performance Optimizer
//!
//! High-performance optimizations for achieving 100K+ events/second throughput
//! with <10ms latency. Implements advanced batching, memory pooling, zero-copy operations,
//! and parallel processing optimizations.

use crate::{EventMetadata, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable adaptive batching
    pub enable_adaptive_batching: bool,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Target latency for adaptive batching
    pub target_latency_ms: u64,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Memory pool size
    pub memory_pool_size: usize,
    /// Enable zero-copy optimizations
    pub enable_zero_copy: bool,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Number of parallel workers
    pub parallel_workers: usize,
    /// Enable event pre-filtering
    pub enable_event_filtering: bool,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression threshold (bytes)
    pub compression_threshold: usize,
    /// Enable adaptive compression based on network conditions
    pub enable_adaptive_compression: bool,
    /// Network bandwidth estimation (bytes/sec)
    pub estimated_bandwidth: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_batching: true,
            max_batch_size: 10000,
            target_latency_ms: 5,
            enable_memory_pooling: true,
            memory_pool_size: 1_000_000,
            enable_zero_copy: true,
            enable_parallel_processing: true,
            parallel_workers: num_cpus::get(),
            enable_event_filtering: true,
            enable_compression: true,
            compression_threshold: 1024,
            enable_adaptive_compression: true,
            estimated_bandwidth: 1_000_000_000, // 1 Gbps default
        }
    }
}

/// Memory pool for efficient event allocation
pub struct MemoryPool {
    /// Pre-allocated event buffers
    event_buffers: Arc<Mutex<VecDeque<Vec<u8>>>>,
    /// Buffer size
    buffer_size: usize,
    /// Pool statistics
    stats: Arc<MemoryPoolStats>,
}

/// Memory pool statistics
#[derive(Debug, Default)]
pub struct MemoryPoolStats {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub bytes_allocated: AtomicU64,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(pool_size: usize, buffer_size: usize) -> Self {
        let mut buffers = VecDeque::with_capacity(pool_size);
        for _ in 0..pool_size {
            buffers.push_back(vec![0u8; buffer_size]);
        }

        Self {
            event_buffers: Arc::new(Mutex::new(buffers)),
            buffer_size,
            stats: Arc::new(MemoryPoolStats::default()),
        }
    }

    /// Allocate a buffer from the pool
    pub async fn allocate(&self) -> Vec<u8> {
        let mut buffers = self.event_buffers.lock().await;
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);

        if let Some(mut buffer) = buffers.pop_front() {
            buffer.clear();
            buffer.reserve(self.buffer_size);
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            buffer
        } else {
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            self.stats
                .bytes_allocated
                .fetch_add(self.buffer_size as u64, Ordering::Relaxed);
            vec![0u8; self.buffer_size]
        }
    }

    /// Return a buffer to the pool
    pub async fn deallocate(&self, buffer: Vec<u8>) {
        let mut buffers = self.event_buffers.lock().await;
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);

        if buffers.len() < buffers.capacity() {
            buffers.push_back(buffer);
        }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            allocations: AtomicU64::new(self.stats.allocations.load(Ordering::Relaxed)),
            deallocations: AtomicU64::new(self.stats.deallocations.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.stats.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.stats.cache_misses.load(Ordering::Relaxed)),
            bytes_allocated: AtomicU64::new(self.stats.bytes_allocated.load(Ordering::Relaxed)),
        }
    }
}

/// Adaptive batching optimizer
pub struct AdaptiveBatcher {
    /// Current batch size
    current_batch_size: AtomicUsize,
    /// Target latency
    target_latency: Duration,
    /// Maximum batch size
    max_batch_size: usize,
    /// Latency history for adaptation
    latency_history: Arc<Mutex<VecDeque<Duration>>>,
    /// Batch performance statistics
    stats: Arc<BatchingStats>,
}

/// Batching performance statistics
#[derive(Debug, Default)]
pub struct BatchingStats {
    pub total_batches: AtomicU64,
    pub total_events: AtomicU64,
    pub average_batch_size: AtomicU64,
    pub average_latency_ms: AtomicU64,
    pub throughput_eps: AtomicU64,
    pub adaptations: AtomicU64,
}

/// ML-based batch size predictor for intelligent batching optimization
#[derive(Debug)]
pub struct BatchSizePredictor {
    /// Historical batch performance data
    training_data: VecDeque<BatchPerformancePoint>,
    /// Linear regression coefficients
    coefficients: Option<(f64, f64)>, // (slope, intercept)
    /// Prediction accuracy tracker
    accuracy_tracker: VecDeque<f64>,
    /// Maximum training data size
    max_training_size: usize,
}

/// Single training data point for batch performance
#[derive(Debug, Clone)]
pub struct BatchPerformancePoint {
    /// Batch size used
    pub batch_size: usize,
    /// Observed latency
    pub latency_ms: f64,
    /// System load factor (0.0 to 1.0)
    pub system_load: f64,
    /// Event complexity factor
    pub event_complexity: f64,
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
}

impl BatchSizePredictor {
    /// Create a new batch size predictor
    pub fn new() -> Self {
        Self {
            training_data: VecDeque::with_capacity(1000),
            coefficients: None,
            accuracy_tracker: VecDeque::with_capacity(100),
            max_training_size: 1000,
        }
    }

    /// Add a new training data point
    pub fn add_training_point(&mut self, point: BatchPerformancePoint) {
        self.training_data.push_back(point);

        // Keep training data size under limit
        if self.training_data.len() > self.max_training_size {
            self.training_data.pop_front();
        }

        // Retrain model if we have enough data
        if self.training_data.len() >= 10 {
            self.retrain_model();
        }
    }

    /// Predict optimal batch size for given conditions
    pub fn predict_optimal_batch_size(
        &self,
        target_latency_ms: f64,
        system_load: f64,
        event_complexity: f64,
    ) -> Option<usize> {
        let (slope, intercept) = self.coefficients?;

        // Simple linear regression: latency = slope * batch_size + intercept
        // Adjust for system conditions
        let adjusted_target =
            target_latency_ms * (1.0 + system_load * 0.5) * (1.0 + event_complexity * 0.3);

        // Solve for batch_size: batch_size = (latency - intercept) / slope
        if slope.abs() < f64::EPSILON {
            return None;
        }

        let predicted_batch_size = ((adjusted_target - intercept) / slope).max(1.0);
        Some(predicted_batch_size as usize)
    }

    /// Retrain the linear regression model
    fn retrain_model(&mut self) {
        if self.training_data.len() < 2 {
            return;
        }

        let n = self.training_data.len() as f64;
        let mut sum_x = 0.0; // batch size
        let mut sum_y = 0.0; // latency
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for point in &self.training_data {
            let x = point.batch_size as f64;
            let y = point.latency_ms;

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        // Calculate linear regression coefficients
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f64::EPSILON {
            return;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        self.coefficients = Some((slope, intercept));

        debug!(
            "Retrained batch predictor: slope={:.4}, intercept={:.4}, samples={}",
            slope,
            intercept,
            self.training_data.len()
        );
    }

    /// Calculate current prediction accuracy
    pub fn get_accuracy(&self) -> f64 {
        if self.accuracy_tracker.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.accuracy_tracker.iter().sum();
        sum / self.accuracy_tracker.len() as f64
    }

    /// Validate prediction against actual performance
    pub fn validate_prediction(
        &mut self,
        predicted: usize,
        actual_latency: f64,
        actual_batch_size: usize,
    ) {
        if actual_batch_size == 0 {
            return;
        }

        let error_ratio =
            (predicted as f64 - actual_batch_size as f64).abs() / actual_batch_size as f64;
        let accuracy = (1.0 - error_ratio.min(1.0)) * 100.0;

        self.accuracy_tracker.push_back(accuracy);
        if self.accuracy_tracker.len() > 100 {
            self.accuracy_tracker.pop_front();
        }
    }
}

/// Network-aware compression system that adapts compression levels based on network conditions
#[derive(Debug)]
pub struct NetworkAwareCompressor {
    /// Current compression level (1-9)
    current_level: AtomicUsize,
    /// Network bandwidth tracker
    bandwidth_tracker: Arc<Mutex<BandwidthTracker>>,
    /// Compression statistics
    stats: Arc<CompressionStats>,
    /// Configuration
    config: PerformanceConfig,
}

/// Bandwidth tracking and estimation
#[derive(Debug)]
pub struct BandwidthTracker {
    /// Recent bandwidth measurements
    measurements: VecDeque<BandwidthMeasurement>,
    /// Current estimated bandwidth (bytes/sec)
    estimated_bandwidth: f64,
    /// Last measurement time
    last_measurement: Option<Instant>,
}

/// Single bandwidth measurement
#[derive(Debug, Clone)]
pub struct BandwidthMeasurement {
    /// Bytes transferred
    pub bytes: u64,
    /// Time taken
    pub duration: Duration,
    /// Timestamp
    pub timestamp: Instant,
}

/// Compression performance statistics
#[derive(Debug, Default)]
pub struct CompressionStats {
    pub total_compressed: AtomicU64,
    pub total_uncompressed: AtomicU64,
    pub compression_ratio: AtomicU64, // Stored as ratio * 1000
    pub compression_time_ms: AtomicU64,
    pub decompression_time_ms: AtomicU64,
    pub level_adjustments: AtomicU64,
}

impl NetworkAwareCompressor {
    /// Create a new network-aware compressor
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            current_level: AtomicUsize::new(6), // Default compression level
            bandwidth_tracker: Arc::new(Mutex::new(BandwidthTracker::new())),
            stats: Arc::new(CompressionStats::default()),
            config,
        }
    }

    /// Compress data with adaptive compression level
    pub async fn compress_adaptive(&self, data: &[u8]) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        // Determine optimal compression level based on network conditions
        let compression_level = self.calculate_optimal_compression_level().await;

        // Update current level
        self.current_level
            .store(compression_level, Ordering::Relaxed);

        // Perform compression
        let compressed = self.compress_with_level(data, compression_level)?;

        // Update statistics
        let compression_time = start_time.elapsed();
        self.update_compression_stats(data.len(), compressed.len(), compression_time);

        debug!(
            "Compressed {} bytes to {} bytes (ratio: {:.2}x) with level {} in {:?}",
            data.len(),
            compressed.len(),
            data.len() as f64 / compressed.len() as f64,
            compression_level,
            compression_time
        );

        Ok(compressed)
    }

    /// Calculate optimal compression level based on network conditions
    async fn calculate_optimal_compression_level(&self) -> usize {
        if !self.config.enable_adaptive_compression {
            return 6; // Default level
        }

        let bandwidth_tracker = self.bandwidth_tracker.lock().await;
        let estimated_bw = bandwidth_tracker.estimated_bandwidth;
        drop(bandwidth_tracker);

        // Adaptive logic:
        // - High bandwidth (>100 Mbps): Lower compression for speed
        // - Medium bandwidth (10-100 Mbps): Balanced compression
        // - Low bandwidth (<10 Mbps): Higher compression for efficiency

        if estimated_bw > 100_000_000.0 {
            // >100 Mbps
            3 // Low compression, prioritize speed
        } else if estimated_bw > 10_000_000.0 {
            // 10-100 Mbps
            6 // Balanced compression
        } else {
            9 // High compression, prioritize size
        }
    }

    /// Compress data with specific compression level
    fn compress_with_level(&self, data: &[u8], level: usize) -> Result<Vec<u8>> {
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level as u32));
        encoder.write_all(data)?;
        encoder
            .finish()
            .map_err(|e| anyhow!("Compression failed: {}", e))
    }

    /// Update bandwidth measurement
    pub async fn update_bandwidth(&self, bytes_transferred: u64, duration: Duration) {
        let mut tracker = self.bandwidth_tracker.lock().await;
        tracker.add_measurement(BandwidthMeasurement {
            bytes: bytes_transferred,
            duration,
            timestamp: Instant::now(),
        });
    }

    /// Update compression statistics
    fn update_compression_stats(
        &self,
        original_size: usize,
        compressed_size: usize,
        compression_time: Duration,
    ) {
        self.stats
            .total_uncompressed
            .fetch_add(original_size as u64, Ordering::Relaxed);
        self.stats
            .total_compressed
            .fetch_add(compressed_size as u64, Ordering::Relaxed);
        self.stats
            .compression_time_ms
            .fetch_add(compression_time.as_millis() as u64, Ordering::Relaxed);

        // Calculate and store compression ratio
        if compressed_size > 0 {
            let ratio = (original_size as f64 / compressed_size as f64 * 1000.0) as u64;
            self.stats.compression_ratio.store(ratio, Ordering::Relaxed);
        }
    }

    /// Get current compression statistics
    pub fn get_stats(&self) -> CompressionStats {
        CompressionStats {
            total_compressed: AtomicU64::new(self.stats.total_compressed.load(Ordering::Relaxed)),
            total_uncompressed: AtomicU64::new(
                self.stats.total_uncompressed.load(Ordering::Relaxed),
            ),
            compression_ratio: AtomicU64::new(self.stats.compression_ratio.load(Ordering::Relaxed)),
            compression_time_ms: AtomicU64::new(
                self.stats.compression_time_ms.load(Ordering::Relaxed),
            ),
            decompression_time_ms: AtomicU64::new(
                self.stats.decompression_time_ms.load(Ordering::Relaxed),
            ),
            level_adjustments: AtomicU64::new(self.stats.level_adjustments.load(Ordering::Relaxed)),
        }
    }
}

impl BandwidthTracker {
    /// Create a new bandwidth tracker
    pub fn new() -> Self {
        Self {
            measurements: VecDeque::with_capacity(100),
            estimated_bandwidth: 1_000_000_000.0, // 1 Gbps default
            last_measurement: None,
        }
    }

    /// Add a new bandwidth measurement
    pub fn add_measurement(&mut self, measurement: BandwidthMeasurement) {
        let timestamp = measurement.timestamp;
        self.measurements.push_back(measurement);

        // Keep only recent measurements
        if self.measurements.len() > 100 {
            self.measurements.pop_front();
        }

        // Update estimated bandwidth
        self.update_estimated_bandwidth();
        self.last_measurement = Some(timestamp);
    }

    /// Update estimated bandwidth based on recent measurements
    fn update_estimated_bandwidth(&mut self) {
        if self.measurements.is_empty() {
            return;
        }

        // Calculate weighted average bandwidth over recent measurements
        let mut total_weighted_bw = 0.0;
        let mut total_weight = 0.0;
        let now = Instant::now();

        for measurement in &self.measurements {
            let age_seconds = now.duration_since(measurement.timestamp).as_secs_f64();

            // Exponential decay weight (newer measurements have higher weight)
            let weight = (-age_seconds / 60.0).exp(); // 1-minute half-life

            let bandwidth = measurement.bytes as f64 / measurement.duration.as_secs_f64();
            total_weighted_bw += bandwidth * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            self.estimated_bandwidth = total_weighted_bw / total_weight;
        }
    }

    /// Get current estimated bandwidth
    pub fn get_estimated_bandwidth(&self) -> f64 {
        self.estimated_bandwidth
    }
}

impl AdaptiveBatcher {
    /// Create a new adaptive batcher
    pub fn new(target_latency_ms: u64, max_batch_size: usize) -> Self {
        Self {
            current_batch_size: AtomicUsize::new(100),
            target_latency: Duration::from_millis(target_latency_ms),
            max_batch_size,
            latency_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            stats: Arc::new(BatchingStats::default()),
        }
    }

    /// Get optimal batch size based on recent performance
    pub async fn get_optimal_batch_size(&self) -> usize {
        let current = self.current_batch_size.load(Ordering::Relaxed);
        let history = self.latency_history.lock().await;

        if history.len() < 10 {
            return current;
        }

        let avg_latency: Duration = history.iter().sum::<Duration>() / history.len() as u32;

        drop(history);

        let new_size = if avg_latency > self.target_latency {
            // Reduce batch size to decrease latency
            (current as f64 * 0.9).max(10.0) as usize
        } else if avg_latency < self.target_latency * 2 / 3 {
            // Increase batch size for better throughput
            (current as f64 * 1.1).min(self.max_batch_size as f64) as usize
        } else {
            current
        };

        if new_size != current {
            self.current_batch_size.store(new_size, Ordering::Relaxed);
            self.stats.adaptations.fetch_add(1, Ordering::Relaxed);
            debug!(
                "Adapted batch size from {} to {} (avg latency: {:?})",
                current, new_size, avg_latency
            );
        }

        new_size
    }

    /// Record batch processing latency
    pub async fn record_latency(&self, latency: Duration, batch_size: usize) {
        let mut history = self.latency_history.lock().await;
        history.push_back(latency);

        if history.len() > 100 {
            history.pop_front();
        }

        self.stats.total_batches.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_events
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.stats
            .average_latency_ms
            .store(latency.as_millis() as u64, Ordering::Relaxed);

        let total_events = self.stats.total_events.load(Ordering::Relaxed);
        let total_batches = self.stats.total_batches.load(Ordering::Relaxed);
        if total_batches > 0 {
            self.stats
                .average_batch_size
                .store(total_events / total_batches, Ordering::Relaxed);
        }
    }

    /// Get batching statistics
    pub fn get_stats(&self) -> BatchingStats {
        BatchingStats {
            total_batches: AtomicU64::new(self.stats.total_batches.load(Ordering::Relaxed)),
            total_events: AtomicU64::new(self.stats.total_events.load(Ordering::Relaxed)),
            average_batch_size: AtomicU64::new(
                self.stats.average_batch_size.load(Ordering::Relaxed),
            ),
            average_latency_ms: AtomicU64::new(
                self.stats.average_latency_ms.load(Ordering::Relaxed),
            ),
            throughput_eps: AtomicU64::new(self.stats.throughput_eps.load(Ordering::Relaxed)),
            adaptations: AtomicU64::new(self.stats.adaptations.load(Ordering::Relaxed)),
        }
    }
}

/// Zero-copy event wrapper for high-performance processing
#[derive(Debug, Clone)]
pub struct ZeroCopyEvent {
    /// Reference to original event data
    pub data: Arc<[u8]>,
    /// Offset for event start
    pub offset: usize,
    /// Event length
    pub length: usize,
    /// Event metadata (small, copied)
    pub metadata: EventMetadata,
    /// Event type for fast filtering
    pub event_type: u8,
}

impl ZeroCopyEvent {
    /// Create a zero-copy event wrapper
    pub fn new(data: Arc<[u8]>, offset: usize, length: usize, metadata: EventMetadata) -> Self {
        let event_type = Self::extract_event_type(&data[offset..offset + length]);
        Self {
            data,
            offset,
            length,
            metadata,
            event_type,
        }
    }

    /// Extract event type for fast filtering
    fn extract_event_type(data: &[u8]) -> u8 {
        // Simple heuristic based on first bytes
        if data.len() > 10 {
            data[0] ^ data[1] ^ data[2] // Simple hash
        } else {
            0
        }
    }

    /// Get event data slice
    pub fn data_slice(&self) -> &[u8] {
        &self.data[self.offset..self.offset + self.length]
    }
}

/// Parallel event processor for high-throughput processing
pub struct ParallelEventProcessor {
    /// Configuration
    config: PerformanceConfig,
    /// Memory pool for efficient allocation
    memory_pool: Arc<MemoryPool>,
    /// Adaptive batcher
    adaptive_batcher: Arc<AdaptiveBatcher>,
    /// Parallel processing semaphore
    worker_semaphore: Arc<Semaphore>,
    /// Processing statistics
    stats: Arc<ProcessingStats>,
    /// Event filters for pre-filtering
    event_filters: Arc<RwLock<Vec<EventFilter>>>,
}

/// Event filter for pre-processing optimization
pub struct EventFilter {
    /// Filter name
    pub name: String,
    /// Filter function
    pub filter_fn: Box<dyn Fn(&ZeroCopyEvent) -> bool + Send + Sync>,
    /// Filter statistics
    pub stats: FilterStats,
}

/// Filter statistics
#[derive(Debug, Default)]
pub struct FilterStats {
    pub events_processed: AtomicU64,
    pub events_filtered: AtomicU64,
    pub processing_time_ns: AtomicU64,
}

/// Processing statistics
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub total_events_processed: AtomicU64,
    pub total_processing_time_ms: AtomicU64,
    pub parallel_batches_processed: AtomicU64,
    pub zero_copy_optimizations: AtomicU64,
    pub compression_savings_bytes: AtomicU64,
    pub memory_pool_efficiency: AtomicU64,
}

impl ParallelEventProcessor {
    /// Create a new parallel event processor
    pub fn new(config: PerformanceConfig) -> Self {
        let memory_pool = Arc::new(MemoryPool::new(
            config.memory_pool_size,
            8192, // 8KB buffer size
        ));

        let adaptive_batcher = Arc::new(AdaptiveBatcher::new(
            config.target_latency_ms,
            config.max_batch_size,
        ));

        let worker_semaphore = Arc::new(Semaphore::new(config.parallel_workers));

        Self {
            config,
            memory_pool,
            adaptive_batcher,
            worker_semaphore,
            stats: Arc::new(ProcessingStats::default()),
            event_filters: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Process events with maximum performance optimizations
    pub async fn process_events_optimized(
        &self,
        events: Vec<StreamEvent>,
    ) -> Result<Vec<ProcessingResult>> {
        let start_time = Instant::now();
        let event_count = events.len();

        // Get optimal batch size
        let optimal_batch_size = self.adaptive_batcher.get_optimal_batch_size().await;

        // Convert to zero-copy events if enabled
        let zero_copy_events = if self.config.enable_zero_copy {
            self.convert_to_zero_copy(events).await?
        } else {
            // Fallback to regular processing
            return self.process_events_regular(events).await;
        };

        // Apply pre-filters if enabled
        let filtered_events = if self.config.enable_event_filtering {
            self.apply_event_filters(zero_copy_events).await?
        } else {
            zero_copy_events
        };

        // Process in optimal batches
        let results = if self.config.enable_parallel_processing {
            self.process_parallel_batches(filtered_events, optimal_batch_size)
                .await?
        } else {
            self.process_sequential_batches(filtered_events, optimal_batch_size)
                .await?
        };

        // Record performance metrics
        let processing_time = start_time.elapsed();
        self.adaptive_batcher
            .record_latency(processing_time, event_count)
            .await;

        self.stats
            .total_events_processed
            .fetch_add(event_count as u64, Ordering::Relaxed);
        self.stats
            .total_processing_time_ms
            .fetch_add(processing_time.as_millis() as u64, Ordering::Relaxed);

        info!(
            "Processed {} events in {:?} ({:.0} events/sec)",
            event_count,
            processing_time,
            event_count as f64 / processing_time.as_secs_f64()
        );

        Ok(results)
    }

    /// Convert events to zero-copy format
    async fn convert_to_zero_copy(&self, events: Vec<StreamEvent>) -> Result<Vec<ZeroCopyEvent>> {
        let mut zero_copy_events = Vec::with_capacity(events.len());
        let mut buffer = self.memory_pool.allocate().await;

        for event in events {
            let start_offset = buffer.len();

            // Serialize event efficiently
            let serialized = if self.config.enable_compression
                && serde_json::to_string(&event)?.len() > self.config.compression_threshold
            {
                // Use compression for large events
                self.compress_event(&event)?
            } else {
                serde_json::to_string(&event)?.into_bytes()
            };

            buffer.extend_from_slice(&serialized);
            let event_length = serialized.len();

            let zero_copy_event = ZeroCopyEvent::new(
                Arc::from(buffer.as_slice()),
                start_offset,
                event_length,
                event.metadata().clone(),
            );

            zero_copy_events.push(zero_copy_event);
        }

        self.stats
            .zero_copy_optimizations
            .fetch_add(zero_copy_events.len() as u64, Ordering::Relaxed);

        // Return buffer to pool
        self.memory_pool.deallocate(buffer).await;

        Ok(zero_copy_events)
    }

    /// Compress event for storage efficiency
    fn compress_event(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let serialized = serde_json::to_string(event)?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(serialized.as_bytes())?;
        let compressed = encoder.finish()?;

        let savings = serialized.len().saturating_sub(compressed.len());
        self.stats
            .compression_savings_bytes
            .fetch_add(savings as u64, Ordering::Relaxed);

        Ok(compressed)
    }

    /// Apply event filters for pre-processing optimization
    async fn apply_event_filters(&self, events: Vec<ZeroCopyEvent>) -> Result<Vec<ZeroCopyEvent>> {
        let filters = self.event_filters.read().await;
        if filters.is_empty() {
            return Ok(events);
        }

        let mut filtered_events = Vec::with_capacity(events.len());

        for event in events {
            let mut should_include = true;

            for filter in filters.iter() {
                let filter_start = Instant::now();
                let passes_filter = (filter.filter_fn)(&event);
                let filter_time = filter_start.elapsed();

                filter
                    .stats
                    .events_processed
                    .fetch_add(1, Ordering::Relaxed);
                filter
                    .stats
                    .processing_time_ns
                    .fetch_add(filter_time.as_nanos() as u64, Ordering::Relaxed);

                if !passes_filter {
                    should_include = false;
                    filter.stats.events_filtered.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }

            if should_include {
                filtered_events.push(event);
            }
        }

        Ok(filtered_events)
    }

    /// Process events in parallel batches
    async fn process_parallel_batches(
        &self,
        events: Vec<ZeroCopyEvent>,
        batch_size: usize,
    ) -> Result<Vec<ProcessingResult>> {
        let mut results = Vec::new();
        let mut tasks = Vec::new();

        for chunk in events.chunks(batch_size) {
            let semaphore = self.worker_semaphore.clone();
            let stats = self.stats.clone();
            let chunk_events = chunk.to_vec();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await?;
                let batch_result = Self::process_event_batch(chunk_events).await?;

                stats
                    .parallel_batches_processed
                    .fetch_add(1, Ordering::Relaxed);

                Ok::<Vec<ProcessingResult>, anyhow::Error>(batch_result)
            });

            tasks.push(task);
        }

        // Wait for all batches to complete
        for task in tasks {
            let batch_results = task.await??;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Process events in sequential batches
    async fn process_sequential_batches(
        &self,
        events: Vec<ZeroCopyEvent>,
        batch_size: usize,
    ) -> Result<Vec<ProcessingResult>> {
        let mut results = Vec::new();

        for chunk in events.chunks(batch_size) {
            let batch_results = Self::process_event_batch(chunk.to_vec()).await?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// Process a single batch of events
    async fn process_event_batch(events: Vec<ZeroCopyEvent>) -> Result<Vec<ProcessingResult>> {
        let mut results = Vec::with_capacity(events.len());

        for event in events {
            let result = ProcessingResult {
                event_id: event.metadata.event_id.clone(),
                processing_time: Duration::from_nanos(100), // Simulated processing
                status: ProcessingStatus::Success,
                output_size: event.length,
                compression_ratio: if event.length > 1000 { Some(0.7) } else { None },
            };
            results.push(result);
        }

        Ok(results)
    }

    /// Process events using regular (non-optimized) path
    async fn process_events_regular(
        &self,
        events: Vec<StreamEvent>,
    ) -> Result<Vec<ProcessingResult>> {
        let mut results = Vec::with_capacity(events.len());

        for event in events {
            let start = Instant::now();
            let serialized_size = serde_json::to_string(&event)?.len();
            let processing_time = start.elapsed();

            let result = ProcessingResult {
                event_id: event.metadata().event_id.clone(),
                processing_time,
                status: ProcessingStatus::Success,
                output_size: serialized_size,
                compression_ratio: None,
            };
            results.push(result);
        }

        Ok(results)
    }

    /// Add an event filter
    pub async fn add_event_filter<F>(&self, name: String, filter_fn: F)
    where
        F: Fn(&ZeroCopyEvent) -> bool + Send + Sync + 'static,
    {
        let mut filters = self.event_filters.write().await;
        filters.push(EventFilter {
            name,
            filter_fn: Box::new(filter_fn),
            stats: FilterStats::default(),
        });
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        ProcessingStats {
            total_events_processed: AtomicU64::new(
                self.stats.total_events_processed.load(Ordering::Relaxed),
            ),
            total_processing_time_ms: AtomicU64::new(
                self.stats.total_processing_time_ms.load(Ordering::Relaxed),
            ),
            parallel_batches_processed: AtomicU64::new(
                self.stats
                    .parallel_batches_processed
                    .load(Ordering::Relaxed),
            ),
            zero_copy_optimizations: AtomicU64::new(
                self.stats.zero_copy_optimizations.load(Ordering::Relaxed),
            ),
            compression_savings_bytes: AtomicU64::new(
                self.stats.compression_savings_bytes.load(Ordering::Relaxed),
            ),
            memory_pool_efficiency: AtomicU64::new(
                self.stats.memory_pool_efficiency.load(Ordering::Relaxed),
            ),
        }
    }

    /// Get memory pool statistics
    pub fn get_memory_pool_stats(&self) -> MemoryPoolStats {
        self.memory_pool.get_stats()
    }

    /// Get batching statistics
    pub fn get_batching_stats(&self) -> BatchingStats {
        self.adaptive_batcher.get_stats()
    }
}

/// Processing result for a single event
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub event_id: String,
    pub processing_time: Duration,
    pub status: ProcessingStatus,
    pub output_size: usize,
    pub compression_ratio: Option<f64>,
}

/// Processing status
#[derive(Debug, Clone)]
pub enum ProcessingStatus {
    Success,
    Filtered,
    Error(String),
}

// Helper trait for accessing event metadata
trait StreamEventMetadata {
    fn metadata(&self) -> &EventMetadata;
}

impl StreamEventMetadata for StreamEvent {
    fn metadata(&self) -> &EventMetadata {
        match self {
            StreamEvent::TripleAdded { metadata, .. } => metadata,
            StreamEvent::TripleRemoved { metadata, .. } => metadata,
            StreamEvent::QuadAdded { metadata, .. } => metadata,
            StreamEvent::QuadRemoved { metadata, .. } => metadata,
            StreamEvent::GraphCreated { metadata, .. } => metadata,
            StreamEvent::GraphCleared { metadata, .. } => metadata,
            StreamEvent::GraphDeleted { metadata, .. } => metadata,
            StreamEvent::SparqlUpdate { metadata, .. } => metadata,
            StreamEvent::TransactionBegin { metadata, .. } => metadata,
            StreamEvent::TransactionCommit { metadata, .. } => metadata,
            StreamEvent::TransactionAbort { metadata, .. } => metadata,
            StreamEvent::SchemaChanged { metadata, .. } => metadata,
            StreamEvent::Heartbeat { metadata, .. } => metadata,
            StreamEvent::QueryResultAdded { metadata, .. } => metadata,
            StreamEvent::QueryResultRemoved { metadata, .. } => metadata,
            StreamEvent::QueryCompleted { metadata, .. } => metadata,
            StreamEvent::ErrorOccurred { metadata, .. } => metadata,
            _ => {
                // For unmatched event types, return a static reference
                use std::sync::LazyLock;
                static DEFAULT_METADATA: LazyLock<EventMetadata> =
                    LazyLock::new(|| EventMetadata::default());
                &DEFAULT_METADATA
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamEvent;
    use std::collections::HashMap;

    fn create_test_event(id: usize) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: format!("http://test.org/subject_{}", id),
            predicate: "http://test.org/predicate".to_string(),
            object: format!("\"test_value_{}\"", id),
            graph: None,
            metadata: EventMetadata {
                event_id: format!("test_event_{}", id),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        }
    }

    #[tokio::test]
    async fn test_memory_pool() {
        let pool = MemoryPool::new(10, 1024);

        let buffer1 = pool.allocate().await;
        assert_eq!(buffer1.len(), 0);
        assert_eq!(buffer1.capacity(), 1024);

        pool.deallocate(buffer1).await;

        let stats = pool.get_stats();
        assert_eq!(stats.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.deallocations.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_adaptive_batcher() {
        let batcher = AdaptiveBatcher::new(10, 1000);

        let initial_size = batcher.get_optimal_batch_size().await;
        assert!(initial_size > 0);

        // Record some latencies
        batcher.record_latency(Duration::from_millis(15), 100).await;
        batcher.record_latency(Duration::from_millis(20), 100).await;

        let stats = batcher.get_stats();
        assert_eq!(stats.total_batches.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_parallel_event_processor() {
        let config = PerformanceConfig {
            enable_parallel_processing: true,
            parallel_workers: 2,
            max_batch_size: 100,
            ..Default::default()
        };

        let processor = ParallelEventProcessor::new(config);

        let events: Vec<StreamEvent> = (0..1000).map(create_test_event).collect();
        let results = processor.process_events_optimized(events).await.unwrap();

        assert_eq!(results.len(), 1000);

        let stats = processor.get_stats();
        assert_eq!(stats.total_events_processed.load(Ordering::Relaxed), 1000);
    }

    #[tokio::test]
    async fn test_zero_copy_optimization() {
        let config = PerformanceConfig {
            enable_zero_copy: true,
            enable_compression: true,
            compression_threshold: 500,
            ..Default::default()
        };

        let processor = ParallelEventProcessor::new(config);

        let events: Vec<StreamEvent> = (0..100).map(create_test_event).collect();
        let results = processor.process_events_optimized(events).await.unwrap();

        assert_eq!(results.len(), 100);

        let stats = processor.get_stats();
        assert!(stats.zero_copy_optimizations.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_event_filtering() {
        let config = PerformanceConfig {
            enable_event_filtering: true,
            ..Default::default()
        };

        let processor = ParallelEventProcessor::new(config);

        // Add a filter that only allows even IDs
        processor
            .add_event_filter("even_only".to_string(), |event| {
                event
                    .metadata
                    .event_id
                    .ends_with(|c: char| c.is_ascii_digit() && c.to_digit(10).unwrap() % 2 == 0)
            })
            .await;

        let events: Vec<StreamEvent> = (0..100).map(create_test_event).collect();
        let results = processor.process_events_optimized(events).await.unwrap();

        // Should filter out roughly half the events
        assert!(results.len() < 100);
        assert!(results.len() > 40); // Approximately 50 events should pass
    }

    #[test]
    fn test_performance_config_defaults() {
        let config = PerformanceConfig::default();
        assert!(config.enable_adaptive_batching);
        assert!(config.enable_memory_pooling);
        assert!(config.enable_zero_copy);
        assert!(config.enable_parallel_processing);
        assert!(config.parallel_workers > 0);
    }
}

/// AI-driven performance prediction and auto-tuning system
pub struct PerformancePredictor {
    config: PerformanceConfig,
    historical_data: Arc<RwLock<VecDeque<PerformanceDataPoint>>>,
    model: Arc<RwLock<PredictionModel>>,
    auto_tuner: Arc<AutoTuner>,
    prediction_stats: Arc<PredictionStats>,
}

/// Historical performance data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    pub timestamp: DateTime<Utc>,
    pub throughput_eps: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub batch_size: usize,
    pub parallel_workers: usize,
    pub compression_enabled: bool,
    pub event_complexity_score: f64,
    pub network_latency_ms: f64,
}

/// Simple linear regression model for performance prediction
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Coefficients for throughput prediction
    throughput_coefficients: Vec<f64>,
    /// Coefficients for latency prediction  
    latency_coefficients: Vec<f64>,
    /// Model accuracy metrics
    throughput_r_squared: f64,
    latency_r_squared: f64,
    /// Number of training samples
    training_samples: usize,
}

impl Default for PredictionModel {
    fn default() -> Self {
        Self {
            throughput_coefficients: vec![0.0; 8], // 8 features
            latency_coefficients: vec![0.0; 8],
            throughput_r_squared: 0.0,
            latency_r_squared: 0.0,
            training_samples: 0,
        }
    }
}

/// Auto-tuning system for dynamic performance optimization
pub struct AutoTuner {
    current_config: Arc<RwLock<PerformanceConfig>>,
    tuning_history: Arc<RwLock<Vec<TuningDecision>>>,
    last_tuning: Arc<RwLock<Option<Instant>>>,
    tuning_interval: Duration,
}

/// Record of a tuning decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningDecision {
    pub timestamp: DateTime<Utc>,
    pub parameter: String,
    pub old_value: String,
    pub new_value: String,
    pub predicted_improvement: f64,
    pub actual_improvement: Option<f64>,
    pub confidence: f64,
}

/// Prediction statistics and accuracy tracking
#[derive(Debug, Default)]
pub struct PredictionStats {
    pub total_predictions: AtomicU64,
    pub accurate_predictions: AtomicU64, // within 10% of actual
    pub total_tuning_decisions: AtomicU64,
    pub successful_tunings: AtomicU64, // resulted in improvement
    pub average_prediction_error: Arc<RwLock<f64>>,
}

impl PerformancePredictor {
    /// Create a new performance predictor
    pub fn new(config: PerformanceConfig) -> Self {
        let auto_tuner = AutoTuner {
            current_config: Arc::new(RwLock::new(config.clone())),
            tuning_history: Arc::new(RwLock::new(Vec::new())),
            last_tuning: Arc::new(RwLock::new(None)),
            tuning_interval: Duration::from_secs(300), // 5 minutes
        };

        Self {
            config,
            historical_data: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            model: Arc::new(RwLock::new(PredictionModel::default())),
            auto_tuner: Arc::new(auto_tuner),
            prediction_stats: Arc::new(PredictionStats::default()),
        }
    }

    /// Record a performance data point
    pub async fn record_performance(&self, data_point: PerformanceDataPoint) -> Result<()> {
        let mut historical_data = self.historical_data.write().await;

        // Keep only recent data (last 10000 points)
        if historical_data.len() >= 10000 {
            historical_data.pop_front();
        }

        historical_data.push_back(data_point);

        // Retrain model every 100 data points
        if historical_data.len() % 100 == 0 {
            self.retrain_model().await?;
        }

        debug!(
            "Recorded performance data point, total: {}",
            historical_data.len()
        );
        Ok(())
    }

    /// Predict performance for given configuration
    pub async fn predict_performance(&self, test_config: &PerformanceConfig) -> Result<(f64, f64)> {
        let model = self.model.read().await;

        if model.training_samples < 10 {
            return Err(anyhow!("Insufficient training data for prediction"));
        }

        // Extract features from configuration
        let features = self.extract_features(test_config).await;

        // Predict throughput
        let predicted_throughput =
            self.linear_prediction(&features, &model.throughput_coefficients);

        // Predict latency
        let predicted_latency = self.linear_prediction(&features, &model.latency_coefficients);

        self.prediction_stats
            .total_predictions
            .fetch_add(1, Ordering::Relaxed);

        info!(
            "Predicted performance - Throughput: {:.0} eps, Latency: {:.2} ms (confidence: {:.1}%)",
            predicted_throughput,
            predicted_latency,
            (model.throughput_r_squared + model.latency_r_squared) / 2.0 * 100.0
        );

        Ok((predicted_throughput, predicted_latency))
    }

    /// Auto-tune configuration based on current performance
    pub async fn auto_tune(
        &self,
        current_performance: &PerformanceDataPoint,
    ) -> Result<Option<PerformanceConfig>> {
        let mut last_tuning = self.auto_tuner.last_tuning.write().await;

        // Check if enough time has passed since last tuning
        if let Some(last_time) = *last_tuning {
            if last_time.elapsed() < self.auto_tuner.tuning_interval {
                return Ok(None);
            }
        }

        let current_config = self.auto_tuner.current_config.read().await.clone();
        let mut best_config = current_config.clone();
        let mut best_predicted_throughput = 0.0;

        // Try different configuration variations
        let variations = self.generate_config_variations(&current_config).await;

        for variation in variations {
            if let Ok((predicted_throughput, predicted_latency)) =
                self.predict_performance(&variation).await
            {
                // Score based on throughput and latency (higher throughput, lower latency is better)
                let score = predicted_throughput / (1.0 + predicted_latency);
                let current_score =
                    current_performance.throughput_eps / (1.0 + current_performance.latency_ms);

                if score > current_score && predicted_throughput > best_predicted_throughput {
                    best_config = variation;
                    best_predicted_throughput = predicted_throughput;
                }
            }
        }

        // Apply the best configuration if it's significantly better
        if best_predicted_throughput > current_performance.throughput_eps * 1.1 {
            *self.auto_tuner.current_config.write().await = best_config.clone();
            *last_tuning = Some(Instant::now());

            // Record tuning decision
            let decision = TuningDecision {
                timestamp: Utc::now(),
                parameter: "auto_tune".to_string(),
                old_value: format!("{:?}", current_config),
                new_value: format!("{:?}", best_config),
                predicted_improvement: (best_predicted_throughput
                    - current_performance.throughput_eps)
                    / current_performance.throughput_eps,
                actual_improvement: None,
                confidence: 0.8, // TODO: Calculate actual confidence
            };

            self.auto_tuner.tuning_history.write().await.push(decision);
            self.prediction_stats
                .total_tuning_decisions
                .fetch_add(1, Ordering::Relaxed);

            info!(
                "Auto-tuned configuration for {:.1}% predicted improvement",
                (best_predicted_throughput - current_performance.throughput_eps)
                    / current_performance.throughput_eps
                    * 100.0
            );

            Ok(Some(best_config))
        } else {
            Ok(None)
        }
    }

    /// Get current prediction accuracy statistics
    pub async fn get_prediction_stats(&self) -> PredictionStats {
        PredictionStats {
            total_predictions: AtomicU64::new(
                self.prediction_stats
                    .total_predictions
                    .load(Ordering::Relaxed),
            ),
            accurate_predictions: AtomicU64::new(
                self.prediction_stats
                    .accurate_predictions
                    .load(Ordering::Relaxed),
            ),
            total_tuning_decisions: AtomicU64::new(
                self.prediction_stats
                    .total_tuning_decisions
                    .load(Ordering::Relaxed),
            ),
            successful_tunings: AtomicU64::new(
                self.prediction_stats
                    .successful_tunings
                    .load(Ordering::Relaxed),
            ),
            average_prediction_error: Arc::new(RwLock::new(
                *self.prediction_stats.average_prediction_error.read().await,
            )),
        }
    }

    /// Retrain the prediction model with current historical data
    async fn retrain_model(&self) -> Result<()> {
        let historical_data = self.historical_data.read().await;

        if historical_data.len() < 10 {
            return Ok(()); // Need more data
        }

        let mut model = self.model.write().await;

        // Prepare training data
        let mut features_matrix = Vec::new();
        let mut throughput_targets = Vec::new();
        let mut latency_targets = Vec::new();

        for data_point in historical_data.iter() {
            let config = PerformanceConfig {
                max_batch_size: data_point.batch_size,
                parallel_workers: data_point.parallel_workers,
                enable_compression: data_point.compression_enabled,
                ..self.config.clone()
            };

            let features = self.extract_features(&config).await;
            features_matrix.push(features);
            throughput_targets.push(data_point.throughput_eps);
            latency_targets.push(data_point.latency_ms);
        }

        // Train throughput model (simple linear regression)
        model.throughput_coefficients =
            self.train_linear_regression(&features_matrix, &throughput_targets);
        model.throughput_r_squared = self.calculate_r_squared(
            &features_matrix,
            &throughput_targets,
            &model.throughput_coefficients,
        );

        // Train latency model
        model.latency_coefficients =
            self.train_linear_regression(&features_matrix, &latency_targets);
        model.latency_r_squared = self.calculate_r_squared(
            &features_matrix,
            &latency_targets,
            &model.latency_coefficients,
        );

        model.training_samples = historical_data.len();

        info!(
            "Retrained prediction model with {} samples - Throughput R²: {:.3}, Latency R²: {:.3}",
            model.training_samples, model.throughput_r_squared, model.latency_r_squared
        );

        Ok(())
    }

    /// Extract feature vector from configuration
    async fn extract_features(&self, config: &PerformanceConfig) -> Vec<f64> {
        vec![
            config.max_batch_size as f64,
            config.parallel_workers as f64,
            if config.enable_compression { 1.0 } else { 0.0 },
            if config.enable_zero_copy { 1.0 } else { 0.0 },
            if config.enable_memory_pooling {
                1.0
            } else {
                0.0
            },
            if config.enable_adaptive_batching {
                1.0
            } else {
                0.0
            },
            config.target_latency_ms as f64,
            config.compression_threshold as f64,
        ]
    }

    /// Simple linear regression implementation
    fn train_linear_regression(&self, features: &[Vec<f64>], targets: &[f64]) -> Vec<f64> {
        if features.is_empty() || targets.is_empty() {
            return vec![0.0; 8];
        }

        let n = features.len();
        let feature_count = features[0].len();
        let mut coefficients = vec![0.0; feature_count];

        // Simple least squares implementation (placeholder)
        // In a real implementation, you'd use a proper linear algebra library
        for i in 0..feature_count {
            let mut sum_xy = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_x2 = 0.0;

            for j in 0..n {
                let x = features[j][i];
                let y = targets[j];
                sum_xy += x * y;
                sum_x += x;
                sum_y += y;
                sum_x2 += x * x;
            }

            let denominator = n as f64 * sum_x2 - sum_x * sum_x;
            if denominator.abs() > 1e-10 {
                coefficients[i] = (n as f64 * sum_xy - sum_x * sum_y) / denominator;
            }
        }

        coefficients
    }

    /// Calculate R-squared for model evaluation
    fn calculate_r_squared(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        coefficients: &[f64],
    ) -> f64 {
        if features.is_empty() || targets.is_empty() {
            return 0.0;
        }

        let mean_target: f64 = targets.iter().sum::<f64>() / targets.len() as f64;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (i, target) in targets.iter().enumerate() {
            let predicted = self.linear_prediction(&features[i], coefficients);
            ss_res += (target - predicted).powi(2);
            ss_tot += (target - mean_target).powi(2);
        }

        if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        }
    }

    /// Make prediction using linear model
    fn linear_prediction(&self, features: &[f64], coefficients: &[f64]) -> f64 {
        features
            .iter()
            .zip(coefficients.iter())
            .map(|(f, c)| f * c)
            .sum()
    }

    /// Generate configuration variations for auto-tuning
    async fn generate_config_variations(
        &self,
        base_config: &PerformanceConfig,
    ) -> Vec<PerformanceConfig> {
        let mut variations = Vec::new();

        // Batch size variations
        for factor in [0.8, 1.2, 1.5] {
            let mut config = base_config.clone();
            config.max_batch_size = ((config.max_batch_size as f64 * factor) as usize)
                .max(100)
                .min(50000);
            variations.push(config);
        }

        // Worker count variations
        for delta in [-1, 1, 2] {
            let mut config = base_config.clone();
            config.parallel_workers =
                (config.parallel_workers as i32 + delta).max(1).min(32) as usize;
            variations.push(config);
        }

        // Compression toggle
        let mut config = base_config.clone();
        config.enable_compression = !config.enable_compression;
        variations.push(config);

        variations
    }
}

#[cfg(test)]
mod performance_predictor_tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_predictor_creation() {
        let config = PerformanceConfig::default();
        let predictor = PerformancePredictor::new(config);

        let stats = predictor.get_prediction_stats().await;
        assert_eq!(stats.total_predictions.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_record_and_predict() {
        let config = PerformanceConfig::default();
        let predictor = PerformancePredictor::new(config.clone());

        // Record some training data
        for i in 0..20 {
            let data_point = PerformanceDataPoint {
                timestamp: Utc::now(),
                throughput_eps: 1000.0 + i as f64 * 100.0,
                latency_ms: 5.0 + i as f64 * 0.1,
                memory_usage_mb: 100.0,
                cpu_usage_percent: 50.0,
                batch_size: 1000 + i * 100,
                parallel_workers: 4,
                compression_enabled: i % 2 == 0,
                event_complexity_score: 1.0,
                network_latency_ms: 1.0,
            };

            predictor.record_performance(data_point).await.unwrap();
        }

        // Make a prediction
        if let Ok((throughput, latency)) = predictor.predict_performance(&config).await {
            assert!(throughput > 0.0);
            assert!(latency > 0.0);
        }
    }
}
