//! Adaptive batching for performance optimization
//!
//! This module provides adaptive batching capabilities to optimize throughput
//! and latency by dynamically adjusting batch sizes based on system performance.

use super::config::BatchConfig;
use crate::StreamEvent;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Batch size predictor using historical performance data
pub struct BatchSizePredictor {
    config: BatchConfig,
    performance_history: Arc<RwLock<VecDeque<BatchPerformancePoint>>>,
    current_batch_size: AtomicUsize,
    stats: Arc<BatchingStats>,
}

/// Performance data point for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPerformancePoint {
    /// Batch size used
    pub batch_size: usize,
    /// Processing latency in milliseconds
    pub latency_ms: u64,
    /// Throughput in events per second
    pub throughput_eps: f64,
    /// CPU utilization percentage
    pub cpu_usage: f64,
    /// Memory utilization percentage
    pub memory_usage: f64,
    /// Timestamp of measurement (as seconds since Unix epoch)
    pub timestamp: u64,
}

/// Batching statistics
#[derive(Debug)]
pub struct BatchingStats {
    /// Total batches processed
    pub total_batches: AtomicU64,
    /// Total events processed
    pub total_events: AtomicU64,
    /// Average batch size
    pub average_batch_size: AtomicUsize,
    /// Current batch size
    pub current_batch_size: AtomicUsize,
    /// Peak batch size
    pub peak_batch_size: AtomicUsize,
    /// Total processing time
    pub total_processing_time_ms: AtomicU64,
    /// Average processing time per batch
    pub average_processing_time_ms: AtomicU64,
    /// Throughput (events per second)
    pub throughput_eps: AtomicU64,
    /// Peak throughput
    pub peak_throughput_eps: AtomicU64,
    /// Number of adjustments made
    pub adjustments_made: AtomicU64,
    /// Last adjustment timestamp
    pub last_adjustment: AtomicU64,
}

impl Default for BatchingStats {
    fn default() -> Self {
        Self {
            total_batches: AtomicU64::new(0),
            total_events: AtomicU64::new(0),
            average_batch_size: AtomicUsize::new(0),
            current_batch_size: AtomicUsize::new(0),
            peak_batch_size: AtomicUsize::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            average_processing_time_ms: AtomicU64::new(0),
            throughput_eps: AtomicU64::new(0),
            peak_throughput_eps: AtomicU64::new(0),
            adjustments_made: AtomicU64::new(0),
            last_adjustment: AtomicU64::new(0),
        }
    }
}

impl Default for BatchSizePredictor {
    fn default() -> Self {
        Self::new(BatchConfig::default())
    }
}

impl BatchSizePredictor {
    /// Create a new batch size predictor
    pub fn new(config: BatchConfig) -> Self {
        Self {
            current_batch_size: AtomicUsize::new(config.initial_batch_size),
            config,
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(BatchingStats::default()),
        }
    }

    /// Record performance data for a batch
    pub async fn record_performance(&self, performance: BatchPerformancePoint) -> Result<()> {
        let mut history = self.performance_history.write().await;

        // Keep only recent history to avoid memory growth
        if history.len() >= 1000 {
            history.pop_front();
        }

        history.push_back(performance.clone());

        // Update statistics
        self.stats.total_batches.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_events
            .fetch_add(performance.batch_size as u64, Ordering::Relaxed);

        let processing_time_ms = performance.latency_ms;
        self.stats
            .total_processing_time_ms
            .fetch_add(processing_time_ms, Ordering::Relaxed);

        // Update average processing time
        let total_batches = self.stats.total_batches.load(Ordering::Relaxed);
        let total_time = self.stats.total_processing_time_ms.load(Ordering::Relaxed);
        let avg_time = if total_batches > 0 {
            total_time / total_batches
        } else {
            0
        };
        self.stats
            .average_processing_time_ms
            .store(avg_time, Ordering::Relaxed);

        // Update throughput
        let throughput = performance.throughput_eps as u64;
        self.stats
            .throughput_eps
            .store(throughput, Ordering::Relaxed);

        let peak_throughput = self.stats.peak_throughput_eps.load(Ordering::Relaxed);
        if throughput > peak_throughput {
            self.stats
                .peak_throughput_eps
                .store(throughput, Ordering::Relaxed);
        }

        debug!("Recorded batch performance: {:?}", performance);
        Ok(())
    }

    /// Predict optimal batch size based on historical data
    pub async fn predict_batch_size(&self) -> Result<usize> {
        let history = self.performance_history.read().await;

        if history.is_empty() {
            return Ok(self.config.initial_batch_size);
        }

        // Simple adaptive algorithm
        let recent_performance: Vec<_> = history.iter().rev().take(10).collect();
        let avg_latency: f64 = recent_performance
            .iter()
            .map(|p| p.latency_ms as f64)
            .sum::<f64>()
            / recent_performance.len() as f64;

        let current_size = self.current_batch_size.load(Ordering::Relaxed);
        let target_latency = self.config.target_latency_ms as f64;
        let tolerance = self.config.latency_tolerance_ms as f64;

        let new_size = if avg_latency > target_latency + tolerance {
            // Latency too high, decrease batch size
            ((current_size as f64) / self.config.adjustment_factor)
                .max(self.config.min_batch_size as f64) as usize
        } else if avg_latency < target_latency - tolerance {
            // Latency acceptable, try to increase batch size
            ((current_size as f64) * self.config.adjustment_factor)
                .min(self.config.max_batch_size as f64) as usize
        } else {
            // Latency is within tolerance, keep current size
            current_size
        };

        if new_size != current_size {
            self.current_batch_size.store(new_size, Ordering::Relaxed);
            self.stats.adjustments_made.fetch_add(1, Ordering::Relaxed);
            self.stats.last_adjustment.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                Ordering::Relaxed,
            );

            info!(
                "Adjusted batch size from {} to {} (avg_latency: {:.2}ms, target: {}ms)",
                current_size, new_size, avg_latency, target_latency
            );
        }

        Ok(new_size)
    }

    /// Get current batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size.load(Ordering::Relaxed)
    }

    /// Get batching statistics
    pub fn stats(&self) -> BatchingStats {
        BatchingStats {
            total_batches: AtomicU64::new(self.stats.total_batches.load(Ordering::Relaxed)),
            total_events: AtomicU64::new(self.stats.total_events.load(Ordering::Relaxed)),
            average_batch_size: AtomicUsize::new(
                self.stats.average_batch_size.load(Ordering::Relaxed),
            ),
            current_batch_size: AtomicUsize::new(self.current_batch_size.load(Ordering::Relaxed)),
            peak_batch_size: AtomicUsize::new(self.stats.peak_batch_size.load(Ordering::Relaxed)),
            total_processing_time_ms: AtomicU64::new(
                self.stats.total_processing_time_ms.load(Ordering::Relaxed),
            ),
            average_processing_time_ms: AtomicU64::new(
                self.stats
                    .average_processing_time_ms
                    .load(Ordering::Relaxed),
            ),
            throughput_eps: AtomicU64::new(self.stats.throughput_eps.load(Ordering::Relaxed)),
            peak_throughput_eps: AtomicU64::new(
                self.stats.peak_throughput_eps.load(Ordering::Relaxed),
            ),
            adjustments_made: AtomicU64::new(self.stats.adjustments_made.load(Ordering::Relaxed)),
            last_adjustment: AtomicU64::new(self.stats.last_adjustment.load(Ordering::Relaxed)),
        }
    }
}

/// Adaptive batcher for events
pub struct AdaptiveBatcher {
    predictor: BatchSizePredictor,
    buffer: Arc<RwLock<Vec<StreamEvent>>>,
    last_flush: Arc<RwLock<Instant>>,
    flush_interval: Duration,
}

impl AdaptiveBatcher {
    /// Create a new adaptive batcher
    pub fn new(config: BatchConfig, flush_interval: Duration) -> Self {
        Self {
            predictor: BatchSizePredictor::new(config),
            buffer: Arc::new(RwLock::new(Vec::new())),
            last_flush: Arc::new(RwLock::new(Instant::now())),
            flush_interval,
        }
    }

    /// Add an event to the batch
    pub async fn add_event(&self, event: StreamEvent) -> Result<Option<Vec<StreamEvent>>> {
        let mut buffer = self.buffer.write().await;
        buffer.push(event);

        let target_size = self.predictor.predict_batch_size().await?;

        // Check if we should flush based on size or time
        let should_flush_size = buffer.len() >= target_size;
        let should_flush_time = {
            let last_flush = self.last_flush.read().await;
            last_flush.elapsed() >= self.flush_interval
        };

        if should_flush_size || should_flush_time {
            let batch: Vec<StreamEvent> = buffer.drain(..).collect();
            let mut last_flush = self.last_flush.write().await;
            *last_flush = Instant::now();

            debug!(
                "Flushed batch of {} events (size: {}, time: {})",
                batch.len(),
                should_flush_size,
                should_flush_time
            );

            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Force flush the current batch
    pub async fn flush(&self) -> Result<Vec<StreamEvent>> {
        let mut buffer = self.buffer.write().await;
        let batch: Vec<StreamEvent> = buffer.drain(..).collect();

        let mut last_flush = self.last_flush.write().await;
        *last_flush = Instant::now();

        debug!("Force flushed batch of {} events", batch.len());
        Ok(batch)
    }

    /// Record performance for the last batch
    pub async fn record_batch_performance(&self, performance: BatchPerformancePoint) -> Result<()> {
        self.predictor.record_performance(performance).await
    }

    /// Get current buffer size
    pub async fn buffer_size(&self) -> usize {
        self.buffer.read().await.len()
    }

    /// Get batching statistics
    pub fn stats(&self) -> BatchingStats {
        self.predictor.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    #[tokio::test]
    async fn test_batch_size_predictor() {
        let config = BatchConfig::default();
        let predictor = BatchSizePredictor::new(config);

        assert_eq!(predictor.current_batch_size(), 100);

        let batch_size = predictor.predict_batch_size().await.unwrap();
        assert_eq!(batch_size, 100);
    }

    #[tokio::test]
    async fn test_adaptive_batcher() {
        let config = BatchConfig::default();
        let batcher = AdaptiveBatcher::new(config, Duration::from_millis(100));

        let event = StreamEvent::TripleAdded {
            subject: "test".to_string(),
            predicate: "test".to_string(),
            object: "test".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        };

        let result = batcher.add_event(event).await.unwrap();
        assert!(result.is_none());

        assert_eq!(batcher.buffer_size().await, 1);
    }

    #[tokio::test]
    async fn test_batch_flush() {
        let config = BatchConfig::default();
        let batcher = AdaptiveBatcher::new(config, Duration::from_millis(100));

        let event = StreamEvent::TripleAdded {
            subject: "test".to_string(),
            predicate: "test".to_string(),
            object: "test".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        };

        batcher.add_event(event).await.unwrap();

        let batch = batcher.flush().await.unwrap();
        assert_eq!(batch.len(), 1);

        assert_eq!(batcher.buffer_size().await, 0);
    }

    #[tokio::test]
    async fn test_performance_recording() {
        let config = BatchConfig::default();
        let predictor = BatchSizePredictor::new(config);

        let performance = BatchPerformancePoint {
            batch_size: 100,
            latency_ms: 5,
            throughput_eps: 20000.0,
            cpu_usage: 50.0,
            memory_usage: 30.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        predictor.record_performance(performance).await.unwrap();

        let stats = predictor.stats();
        assert_eq!(stats.total_batches.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_events.load(Ordering::Relaxed), 100);
    }
}
