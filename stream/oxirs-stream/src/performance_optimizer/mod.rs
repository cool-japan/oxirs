//! # Advanced Performance Optimizer
//!
//! High-performance optimizations for achieving 100K+ events/second throughput
//! with <10ms latency. Implements advanced batching, memory pooling, zero-copy operations,
//! and parallel processing optimizations.
//!
//! ## Modules
//!
//! - `config`: Configuration structures for performance optimization
//! - `memory`: Memory management and pooling
//! - `batching`: Adaptive batching for throughput optimization
//! - `ml`: Machine learning components for performance prediction
//! - `compression`: Compression and bandwidth optimization
//! - `parallel`: Parallel processing optimizations
//!
//! ## Performance Targets
//!
//! - **Throughput**: 100K+ events/second sustained
//! - **Latency**: P99 <10ms for real-time processing
//! - **Memory**: Efficient memory usage with pooling
//! - **CPU**: Optimal CPU utilization with parallel processing

pub mod batching;
pub mod config;
pub mod memory;
pub mod ml;

// Re-export commonly used types
pub use batching::{AdaptiveBatcher, BatchPerformancePoint, BatchSizePredictor, BatchingStats};
pub use config::{
    BatchConfig, CompressionAlgorithm, CompressionConfig, EnhancedMLConfig, LoadBalancingStrategy,
    MemoryPoolConfig, ParallelConfig, PerformanceConfig,
};
pub use memory::{AllocationStrategy, MemoryHandle, MemoryPool, MemoryPoolStats};
pub use ml::{
    ConfigParams, LinearRegressionModel, ModelStats, PerformanceMetrics, PerformancePredictor,
};

// TODO: The following modules would be created in subsequent refactoring steps:
// pub mod compression;  // Compression and bandwidth optimization
// pub mod parallel;     // Parallel processing optimizations

// Re-export types that are currently in the original file but would be moved
// to appropriate modules in a complete refactoring:

use crate::StreamEvent;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Processing result for batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Number of events processed
    pub events_processed: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Success rate
    pub success_rate: f64,
    /// Errors encountered
    pub errors: Vec<String>,
}

/// Processing statistics
#[derive(Debug)]
pub struct ProcessingStats {
    /// Total events processed
    pub total_events: AtomicU64,
    /// Total processing time
    pub total_processing_time_ms: AtomicU64,
    /// Average processing time per event
    pub avg_processing_time_ms: AtomicU64,
    /// Throughput in events per second
    pub throughput_eps: AtomicU64,
    /// Peak throughput
    pub peak_throughput_eps: AtomicU64,
    /// Error count
    pub error_count: AtomicU64,
    /// Success rate
    pub success_rate: f64,
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self {
            total_events: AtomicU64::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            avg_processing_time_ms: AtomicU64::new(0),
            throughput_eps: AtomicU64::new(0),
            peak_throughput_eps: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            success_rate: 1.0,
        }
    }
}

/// Processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStatus {
    Idle,
    Processing,
    Completed,
    Failed(String),
}

/// Zero-copy event wrapper for efficient processing
#[derive(Debug, Clone)]
pub struct ZeroCopyEvent {
    event: StreamEvent,
    processed: bool,
    processing_start: Option<Instant>,
}

impl ZeroCopyEvent {
    /// Create a new zero-copy event wrapper
    pub fn new(event: StreamEvent) -> Self {
        Self {
            event,
            processed: false,
            processing_start: None,
        }
    }

    /// Mark event as being processed
    pub fn mark_processing(&mut self) {
        self.processing_start = Some(Instant::now());
    }

    /// Mark event as processed
    pub fn mark_processed(&mut self) {
        self.processed = true;
    }

    /// Get processing duration
    pub fn processing_duration(&self) -> Option<Duration> {
        self.processing_start.map(|start| start.elapsed())
    }

    /// Get the underlying event
    pub fn event(&self) -> &StreamEvent {
        &self.event
    }

    /// Check if event is processed
    pub fn is_processed(&self) -> bool {
        self.processed
    }
}

/// Aggregation function for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Count,
    Sum,
    Average,
    Min,
    Max,
    Distinct,
}

/// Tuning decision for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningDecision {
    /// Parameter being tuned
    pub parameter: String,
    /// Old value
    pub old_value: f64,
    /// New value
    pub new_value: f64,
    /// Reason for the change
    pub reason: String,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Confidence level
    pub confidence: f64,
}

/// Auto-tuner for performance parameters
pub struct AutoTuner {
    config: PerformanceConfig,
    performance_history: Vec<ProcessingStats>,
    last_tuning: Option<Instant>,
    tuning_interval: Duration,
}

impl AutoTuner {
    /// Create a new auto-tuner
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            performance_history: Vec::new(),
            last_tuning: None,
            tuning_interval: Duration::from_secs(60), // 1 minute
        }
    }

    /// Record performance data
    pub fn record_performance(&mut self, stats: ProcessingStats) {
        self.performance_history.push(stats);

        // Keep only recent history
        if self.performance_history.len() > 100 {
            self.performance_history.drain(0..50);
        }
    }

    /// Check if tuning is needed
    pub fn needs_tuning(&self) -> bool {
        match self.last_tuning {
            Some(last) => last.elapsed() >= self.tuning_interval,
            None => true,
        }
    }

    /// Perform auto-tuning
    pub fn tune(&mut self) -> Result<Vec<TuningDecision>> {
        if self.performance_history.is_empty() {
            return Ok(Vec::new());
        }

        let mut decisions = Vec::new();

        // Analyze recent performance
        let recent_stats: Vec<_> = self.performance_history.iter().rev().take(10).collect();
        let avg_throughput: f64 = recent_stats
            .iter()
            .map(|s| s.throughput_eps.load(Ordering::Relaxed) as f64)
            .sum::<f64>()
            / recent_stats.len() as f64;

        let avg_latency: f64 = recent_stats
            .iter()
            .map(|s| s.avg_processing_time_ms.load(Ordering::Relaxed) as f64)
            .sum::<f64>()
            / recent_stats.len() as f64;

        // Tune batch size if throughput is low
        if avg_throughput < 50000.0 && self.config.max_batch_size < 2000 {
            let old_batch_size = self.config.max_batch_size as f64;
            let new_batch_size = (old_batch_size * 1.2).min(2000.0);

            decisions.push(TuningDecision {
                parameter: "max_batch_size".to_string(),
                old_value: old_batch_size,
                new_value: new_batch_size,
                reason: "Low throughput detected".to_string(),
                expected_improvement: 0.2,
                confidence: 0.8,
            });
        }

        // Tune parallel workers if latency is high
        if avg_latency > 20.0 && self.config.parallel_workers < num_cpus::get() * 2 {
            let old_workers = self.config.parallel_workers as f64;
            let new_workers = (old_workers + 1.0).min(num_cpus::get() as f64 * 2.0);

            decisions.push(TuningDecision {
                parameter: "parallel_workers".to_string(),
                old_value: old_workers,
                new_value: new_workers,
                reason: "High latency detected".to_string(),
                expected_improvement: 0.15,
                confidence: 0.7,
            });
        }

        Ok(decisions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    #[test]
    fn test_zero_copy_event() {
        let event = StreamEvent::TripleAdded {
            subject: "test".to_string(),
            predicate: "test".to_string(),
            object: "test".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        };

        let mut zero_copy = ZeroCopyEvent::new(event);
        assert!(!zero_copy.is_processed());

        zero_copy.mark_processing();
        zero_copy.mark_processed();
        assert!(zero_copy.is_processed());
    }

    #[test]
    fn test_auto_tuner() {
        let config = PerformanceConfig::default();
        let mut tuner = AutoTuner::new(config);

        assert!(tuner.needs_tuning());

        let stats = ProcessingStats::default();
        tuner.record_performance(stats);

        let decisions = tuner.tune().unwrap();
        assert!(!decisions.is_empty());
    }

    #[test]
    fn test_processing_result() {
        let result = ProcessingResult {
            events_processed: 100,
            processing_time_ms: 50,
            success_rate: 0.95,
            errors: vec!["test error".to_string()],
        };

        assert_eq!(result.events_processed, 100);
        assert_eq!(result.processing_time_ms, 50);
        assert_eq!(result.success_rate, 0.95);
        assert_eq!(result.errors.len(), 1);
    }
}
