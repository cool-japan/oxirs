//! Batch vector processor for efficient GPU operations

use super::{GpuConfig, GpuBuffer, GpuOperationType, GpuOperationResult};
use crate::Vector;
use anyhow::Result;
use std::time::Instant;

/// Batch vector processor for efficient GPU operations
pub struct BatchVectorProcessor {
    config: GpuConfig,
    batch_size: usize,
    processing_queue: Vec<Vector>,
}

impl BatchVectorProcessor {
    /// Create new batch processor
    pub fn new(config: GpuConfig) -> Self {
        let batch_size = config.batch_size;
        Self {
            config,
            batch_size,
            processing_queue: Vec::with_capacity(batch_size),
        }
    }

    /// Add vector to processing queue
    pub fn add_vector(&mut self, vector: Vector) {
        self.processing_queue.push(vector);
    }

    /// Process current batch if full or force process
    pub fn process_batch(&mut self, force: bool) -> Result<Vec<GpuOperationResult>> {
        if self.processing_queue.is_empty() {
            return Ok(Vec::new());
        }

        if !force && self.processing_queue.len() < self.batch_size {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        
        // Process the batch (simplified implementation)
        let batch_size = self.processing_queue.len();
        self.processing_queue.clear();
        
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        let result = GpuOperationResult {
            operation: GpuOperationType::Insert,
            execution_time_ms: execution_time,
            memory_used: batch_size * std::mem::size_of::<Vector>(),
            success: true,
        };

        Ok(vec![result])
    }

    /// Get current queue size
    pub fn queue_size(&self) -> usize {
        self.processing_queue.len()
    }

    /// Check if batch is ready for processing
    pub fn is_batch_ready(&self) -> bool {
        self.processing_queue.len() >= self.batch_size
    }

    /// Clear processing queue
    pub fn clear_queue(&mut self) {
        self.processing_queue.clear();
    }
}

/// Performance report for GPU operations
#[derive(Debug)]
pub struct GpuPerformanceReport {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_execution_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub total_memory_used: usize,
    pub throughput_ops_per_sec: f64,
}

impl GpuPerformanceReport {
    /// Create new performance report from operation results
    pub fn from_results(results: &[GpuOperationResult]) -> Self {
        let total_operations = results.len() as u64;
        let successful_operations = results.iter().filter(|r| r.success).count() as u64;
        let failed_operations = total_operations - successful_operations;
        let total_execution_time_ms: f64 = results.iter().map(|r| r.execution_time_ms).sum();
        let total_memory_used: usize = results.iter().map(|r| r.memory_used).sum();
        
        let average_execution_time_ms = if total_operations > 0 {
            total_execution_time_ms / total_operations as f64
        } else {
            0.0
        };

        let throughput_ops_per_sec = if total_execution_time_ms > 0.0 {
            (total_operations as f64) / (total_execution_time_ms / 1000.0)
        } else {
            0.0
        };

        Self {
            total_operations,
            successful_operations,
            failed_operations,
            total_execution_time_ms,
            average_execution_time_ms,
            total_memory_used,
            throughput_ops_per_sec,
        }
    }

    /// Print performance report
    pub fn print_report(&self) {
        println!("GPU Performance Report:");
        println!("  Total Operations: {}", self.total_operations);
        println!("  Successful: {}", self.successful_operations);
        println!("  Failed: {}", self.failed_operations);
        println!("  Total Execution Time: {:.2}ms", self.total_execution_time_ms);
        println!("  Average Execution Time: {:.2}ms", self.average_execution_time_ms);
        println!("  Total Memory Used: {} bytes", self.total_memory_used);
        println!("  Throughput: {:.2} ops/sec", self.throughput_ops_per_sec);
    }
}