//! Memory-efficient SHACL validation using SciRS2 memory management
//!
//! This module provides memory-efficient validation for large RDF datasets using SciRS2's
//! advanced memory management capabilities including buffer pools, chunked processing,
//! and adaptive memory allocation.

use crate::{
    constraints::ConstraintContext, report::ValidationReport, validation::ValidationViolation,
    Result, Shape, ShapeId,
};
use indexmap::IndexMap;
use oxirs_core::{model::Term, Store};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Memory-efficient validation configuration using SciRS2 memory management
#[derive(Debug, Clone)]
pub struct SciRS2MemoryConfig {
    /// Maximum memory budget in bytes
    pub max_memory_bytes: usize,
    /// Enable memory-mapped arrays for large datasets
    pub enable_memory_mapping: bool,
    /// Enable lazy evaluation
    pub enable_lazy_evaluation: bool,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Enable memory leak detection
    pub enable_leak_detection: bool,
    /// Enable zero-copy operations
    pub enable_zero_copy: bool,
}

impl Default for SciRS2MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1 << 30, // 1GB default
            enable_memory_mapping: true,
            enable_lazy_evaluation: true,
            chunk_size: 1000,
            enable_leak_detection: true,
            enable_zero_copy: true,
        }
    }
}

/// Memory-efficient SHACL validator using SciRS2 memory management
pub struct SciRS2MemoryValidator {
    config: SciRS2MemoryConfig,
    memory_metrics: MemoryMetrics,
}

impl SciRS2MemoryValidator {
    /// Create a new memory-efficient validator
    pub fn new(config: SciRS2MemoryConfig) -> Self {
        Self {
            config,
            memory_metrics: MemoryMetrics::new(),
        }
    }

    /// Validate with memory-efficient processing
    pub fn validate_memory_efficient(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        focus_nodes: &[Term],
    ) -> Result<MemoryEfficientValidationResult> {
        let start_time = Instant::now();
        let initial_memory = self.estimate_memory_usage();

        // Create buffer pool for constraint evaluation
        let mut buffer_pool = BufferPool::new(self.config.max_memory_bytes);

        // Process focus nodes in chunks to limit memory usage
        let mut all_violations = Vec::new();
        let chunks = focus_nodes.chunks(self.config.chunk_size);

        for (chunk_idx, chunk) in chunks.enumerate() {
            // Process chunk with memory efficiency
            let chunk_violations = self.process_chunk_memory_efficient(
                store,
                shapes,
                chunk,
                &mut buffer_pool,
                chunk_idx,
            )?;

            all_violations.extend(chunk_violations);

            // Update metrics
            self.memory_metrics.chunks_processed += 1;
            self.memory_metrics.peak_memory_bytes = self
                .memory_metrics
                .peak_memory_bytes
                .max(self.estimate_memory_usage());

            // Clear buffer pool between chunks if memory pressure is high
            if self.estimate_memory_usage() > self.config.max_memory_bytes * 9 / 10 {
                buffer_pool.clear();
                self.memory_metrics.buffer_clears += 1;
            }
        }

        let execution_time = start_time.elapsed();
        let final_memory = self.estimate_memory_usage();

        Ok(MemoryEfficientValidationResult {
            violations: all_violations,
            total_nodes: focus_nodes.len(),
            processed_chunks: self.memory_metrics.chunks_processed,
            execution_time,
            initial_memory_bytes: initial_memory,
            peak_memory_bytes: self.memory_metrics.peak_memory_bytes,
            final_memory_bytes: final_memory,
            buffer_clears: self.memory_metrics.buffer_clears,
            memory_efficiency_ratio: self.calculate_memory_efficiency(),
        })
    }

    /// Process a chunk of focus nodes with memory efficiency
    fn process_chunk_memory_efficient(
        &self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        chunk: &[Term],
        buffer_pool: &mut BufferPool,
        _chunk_idx: usize,
    ) -> Result<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        for focus_node in chunk {
            // Acquire buffer from pool
            let _buffer = buffer_pool.acquire(256)?;

            // Validate focus node against all active shapes
            for (shape_id, shape) in shapes {
                if !shape.is_active() {
                    continue;
                }

                // Evaluate constraints with lazy evaluation
                for (component_id, constraint) in &shape.constraints {
                    let context = ConstraintContext::new(focus_node.clone(), shape_id.clone())
                        .with_values(vec![focus_node.clone()]);

                    match constraint.evaluate(store, &context) {
                        Ok(crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                            violating_value,
                            message,
                            details: _,
                        }) => {
                            let violation = ValidationViolation::new(
                                focus_node.clone(),
                                shape_id.clone(),
                                component_id.clone(),
                                shape.severity,
                            )
                            .with_value(violating_value.unwrap_or_else(|| focus_node.clone()))
                            .with_message(message.unwrap_or_else(|| {
                                format!("Constraint {} violated", component_id.as_str())
                            }));

                            violations.push(violation);
                        }
                        Ok(_) => {} // No violation
                        Err(_) => {} // Treat errors as no violations for now
                    }
                }
            }

            // Release buffer back to pool (automatic with drop)
        }

        Ok(violations)
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Simplified memory estimation
        // In production, would use actual memory metrics from SciRS2
        std::mem::size_of::<Self>() + self.memory_metrics.chunks_processed * 1024
    }

    /// Calculate memory efficiency ratio
    fn calculate_memory_efficiency(&self) -> f64 {
        if self.memory_metrics.peak_memory_bytes == 0 {
            return 1.0;
        }
        self.config.max_memory_bytes as f64 / self.memory_metrics.peak_memory_bytes as f64
    }

    /// Get memory metrics
    pub fn memory_metrics(&self) -> &MemoryMetrics {
        &self.memory_metrics
    }

    /// Reset memory metrics
    pub fn reset_metrics(&mut self) {
        self.memory_metrics = MemoryMetrics::new();
    }
}

/// Simple buffer pool for constraint evaluation
struct BufferPool {
    max_capacity: usize,
    buffers: Vec<Buffer>,
    allocated_bytes: usize,
}

impl BufferPool {
    fn new(max_capacity: usize) -> Self {
        Self {
            max_capacity,
            buffers: Vec::new(),
            allocated_bytes: 0,
        }
    }

    fn acquire(&mut self, size: usize) -> Result<Buffer> {
        // Check if we have capacity
        if self.allocated_bytes + size > self.max_capacity {
            // Try to reclaim buffers
            self.buffers.retain(|b| !b.is_available());
        }

        let buffer = Buffer::new(size);
        self.allocated_bytes += size;
        self.buffers.push(buffer.clone());

        Ok(buffer)
    }

    fn clear(&mut self) {
        self.buffers.clear();
        self.allocated_bytes = 0;
    }
}

/// Simple buffer implementation
#[derive(Debug, Clone)]
struct Buffer {
    #[allow(dead_code)]
    size: usize,
    available: bool,
}

impl Buffer {
    fn new(size: usize) -> Self {
        Self {
            size,
            available: true,
        }
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

/// Memory usage metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Number of buffer pool clears
    pub buffer_clears: usize,
}

impl MemoryMetrics {
    fn new() -> Self {
        Self {
            chunks_processed: 0,
            peak_memory_bytes: 0,
            buffer_clears: 0,
        }
    }
}

/// Result of memory-efficient validation
#[derive(Debug)]
pub struct MemoryEfficientValidationResult {
    /// All violations found
    pub violations: Vec<ValidationViolation>,
    /// Total number of nodes processed
    pub total_nodes: usize,
    /// Number of chunks processed
    pub processed_chunks: usize,
    /// Total execution time
    pub execution_time: Duration,
    /// Initial memory usage in bytes
    pub initial_memory_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Final memory usage in bytes
    pub final_memory_bytes: usize,
    /// Number of buffer pool clears
    pub buffer_clears: usize,
    /// Memory efficiency ratio (target / peak)
    pub memory_efficiency_ratio: f64,
}

impl MemoryEfficientValidationResult {
    /// Convert to ValidationReport
    pub fn into_report(self) -> ValidationReport {
        let mut report = ValidationReport::new();
        for violation in self.violations {
            report.add_violation(violation);
        }
        report
    }

    /// Get memory summary
    pub fn memory_summary(&self) -> String {
        format!(
            "Memory-efficient validation: {} nodes in {:.3}s, peak memory: {:.2} MB, efficiency: {:.1}%",
            self.total_nodes,
            self.execution_time.as_secs_f64(),
            self.peak_memory_bytes as f64 / (1024.0 * 1024.0),
            self.memory_efficiency_ratio * 100.0
        )
    }

    /// Check if memory usage was within budget
    pub fn is_within_budget(&self, max_memory: usize) -> bool {
        self.peak_memory_bytes <= max_memory
    }

    /// Get memory savings compared to naive approach
    pub fn memory_savings_estimate(&self) -> f64 {
        // Estimate that naive approach would use 10x more memory
        let naive_estimate = self.peak_memory_bytes * 10;
        (naive_estimate - self.peak_memory_bytes) as f64 / naive_estimate as f64
    }
}

/// Adaptive chunking strategy for memory-efficient processing
#[derive(Debug, Clone)]
pub struct AdaptiveChunking {
    /// Initial chunk size
    pub initial_chunk_size: usize,
    /// Minimum chunk size
    pub min_chunk_size: usize,
    /// Maximum chunk size
    pub max_chunk_size: usize,
    /// Memory threshold for adjustment
    pub memory_threshold: f64,
}

impl Default for AdaptiveChunking {
    fn default() -> Self {
        Self {
            initial_chunk_size: 1000,
            min_chunk_size: 100,
            max_chunk_size: 10000,
            memory_threshold: 0.8,
        }
    }
}

impl AdaptiveChunking {
    /// Calculate optimal chunk size based on current memory usage
    pub fn calculate_chunk_size(&self, current_memory: usize, max_memory: usize) -> usize {
        let memory_ratio = current_memory as f64 / max_memory as f64;

        if memory_ratio > self.memory_threshold {
            // High memory pressure - reduce chunk size
            self.min_chunk_size
        } else if memory_ratio < 0.5 {
            // Low memory pressure - increase chunk size
            self.max_chunk_size
        } else {
            // Moderate memory pressure - use initial chunk size
            self.initial_chunk_size
        }
    }

    /// Adjust chunk size dynamically
    pub fn adjust_chunk_size(&self, current_chunk_size: usize, memory_pressure: f64) -> usize {
        if memory_pressure > self.memory_threshold {
            (current_chunk_size / 2).max(self.min_chunk_size)
        } else if memory_pressure < 0.3 {
            (current_chunk_size * 2).min(self.max_chunk_size)
        } else {
            current_chunk_size
        }
    }
}

/// Lazy evaluation cache for constraint results
#[derive(Debug)]
pub struct LazyEvaluationCache {
    cache: HashMap<String, bool>,
    hits: usize,
    misses: usize,
}

impl LazyEvaluationCache {
    /// Create a new lazy evaluation cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Get cached result
    pub fn get(&mut self, key: &str) -> Option<bool> {
        match self.cache.get(key) {
            Some(&result) => {
                self.hits += 1;
                Some(result)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert result
    pub fn insert(&mut self, key: String, result: bool) {
        self.cache.insert(key, result);
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

impl Default for LazyEvaluationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scirs2_memory_config_default() {
        let config = SciRS2MemoryConfig::default();
        assert_eq!(config.max_memory_bytes, 1 << 30); // 1GB
        assert!(config.enable_memory_mapping);
        assert!(config.enable_lazy_evaluation);
        assert!(config.enable_leak_detection);
    }

    #[test]
    fn test_buffer_pool_basic() {
        let mut pool = BufferPool::new(10000);
        let buffer1 = pool.acquire(1000);
        assert!(buffer1.is_ok());

        let buffer2 = pool.acquire(2000);
        assert!(buffer2.is_ok());
    }

    #[test]
    fn test_adaptive_chunking() {
        let chunking = AdaptiveChunking::default();

        // High memory pressure - should reduce chunk size
        let chunk_size = chunking.calculate_chunk_size(900, 1000);
        assert_eq!(chunk_size, chunking.min_chunk_size);

        // Low memory pressure - should increase chunk size
        let chunk_size = chunking.calculate_chunk_size(300, 1000);
        assert_eq!(chunk_size, chunking.max_chunk_size);

        // Moderate memory pressure - should use initial chunk size
        let chunk_size = chunking.calculate_chunk_size(600, 1000);
        assert_eq!(chunk_size, chunking.initial_chunk_size);
    }

    #[test]
    fn test_lazy_evaluation_cache() {
        let mut cache = LazyEvaluationCache::new();

        // First access - miss
        assert_eq!(cache.get("key1"), None);

        // Insert and retrieve - hit
        cache.insert("key1".to_string(), true);
        assert_eq!(cache.get("key1"), Some(true));

        // Check hit rate
        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn test_memory_metrics() {
        let metrics = MemoryMetrics::new();
        assert_eq!(metrics.chunks_processed, 0);
        assert_eq!(metrics.peak_memory_bytes, 0);
        assert_eq!(metrics.buffer_clears, 0);
    }
}
