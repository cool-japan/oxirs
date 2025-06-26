//! Batch builder for accumulating and optimizing RDF operations
//!
//! This module provides a builder pattern for accumulating operations into
//! optimal batches based on system resources and operation types.

use crate::model::{Triple, Subject, Predicate, Object};
use crate::concurrent::parallel_batch::{BatchOperation, BatchConfig};
use crate::OxirsError;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::Mutex;

/// Operation coalescing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalescingStrategy {
    /// No coalescing - operations are kept as-is
    None,
    /// Deduplicate operations (remove duplicates)
    Deduplicate,
    /// Merge compatible operations
    Merge,
    /// Optimize operation order for better cache locality
    OptimizeOrder,
}

/// Batch builder configuration
#[derive(Debug, Clone)]
pub struct BatchBuilderConfig {
    /// Maximum size of a single batch
    pub max_batch_size: usize,
    /// Maximum memory usage in bytes
    pub max_memory_usage: usize,
    /// Coalescing strategy
    pub coalescing_strategy: CoalescingStrategy,
    /// Auto-flush when batch is full
    pub auto_flush: bool,
    /// Group operations by type for better performance
    pub group_by_type: bool,
}

impl Default for BatchBuilderConfig {
    fn default() -> Self {
        let total_memory = sys_info::mem_info()
            .map(|info| info.total * 1024) // Convert to bytes
            .unwrap_or(8 * 1024 * 1024 * 1024); // 8GB default
        
        BatchBuilderConfig {
            max_batch_size: 10000,
            max_memory_usage: total_memory / 10, // Use up to 10% of system memory
            coalescing_strategy: CoalescingStrategy::Deduplicate,
            auto_flush: true,
            group_by_type: true,
        }
    }
}

impl BatchBuilderConfig {
    /// Create configuration optimized for current system
    pub fn auto() -> Self {
        let num_cpus = num_cpus::get();
        let mem_info = sys_info::mem_info().ok();
        
        let (max_batch_size, max_memory_usage) = if let Some(info) = mem_info {
            let total_mb = info.total / 1024;
            if total_mb > 16384 { // > 16GB
                (50000, info.total * 1024 / 8) // Large batches, use 1/8 of memory
            } else if total_mb > 8192 { // > 8GB
                (20000, info.total * 1024 / 10) // Medium batches, use 1/10 of memory
            } else {
                (5000, info.total * 1024 / 20) // Small batches, use 1/20 of memory
            }
        } else {
            (10000, 1024 * 1024 * 1024) // 1GB default
        };
        
        BatchBuilderConfig {
            max_batch_size: max_batch_size * num_cpus / 4, // Scale with CPU count
            max_memory_usage,
            coalescing_strategy: CoalescingStrategy::Merge,
            auto_flush: true,
            group_by_type: true,
        }
    }
}

/// Statistics for batch building
#[derive(Debug, Clone, Default)]
pub struct BatchBuilderStats {
    pub total_operations: usize,
    pub coalesced_operations: usize,
    pub deduplicated_operations: usize,
    pub batches_created: usize,
    pub estimated_memory_usage: usize,
}

/// Batch builder for accumulating operations
pub struct BatchBuilder {
    config: BatchBuilderConfig,
    /// Insert operations
    insert_buffer: Vec<Triple>,
    insert_set: HashSet<Triple>,
    /// Remove operations  
    remove_buffer: Vec<Triple>,
    remove_set: HashSet<Triple>,
    /// Query operations
    query_buffer: Vec<(Option<Subject>, Option<Predicate>, Option<Object>)>,
    /// Transform operations
    transform_buffer: Vec<Arc<dyn Fn(&Triple) -> Option<Triple> + Send + Sync>>,
    /// Current estimated memory usage
    estimated_memory: usize,
    /// Statistics
    stats: BatchBuilderStats,
    /// Flush callback
    flush_callback: Arc<Mutex<Option<Box<dyn Fn(Vec<BatchOperation>) + Send + Sync>>>>,
}

impl BatchBuilder {
    /// Create a new batch builder
    pub fn new(config: BatchBuilderConfig) -> Self {
        BatchBuilder {
            config,
            insert_buffer: Vec::new(),
            insert_set: HashSet::new(),
            remove_buffer: Vec::new(),
            remove_set: HashSet::new(),
            query_buffer: Vec::new(),
            transform_buffer: Vec::new(),
            estimated_memory: 0,
            stats: BatchBuilderStats::default(),
            flush_callback: Arc::new(Mutex::new(None)),
        }
    }

    /// Create a batch builder with automatic configuration
    pub fn auto() -> Self {
        Self::new(BatchBuilderConfig::auto())
    }

    /// Set a callback to be called when batches are flushed
    pub fn on_flush<F>(&mut self, callback: F)
    where
        F: Fn(Vec<BatchOperation>) + Send + Sync + 'static,
    {
        *self.flush_callback.lock() = Some(Box::new(callback));
    }

    /// Add an insert operation
    pub fn insert(&mut self, triple: Triple) -> Result<(), OxirsError> {
        self.stats.total_operations += 1;
        
        // Apply coalescing
        match self.config.coalescing_strategy {
            CoalescingStrategy::None => {
                self.estimated_memory += self.estimate_triple_size(&triple);
                self.insert_buffer.push(triple);
            }
            CoalescingStrategy::Deduplicate | CoalescingStrategy::Merge => {
                if self.insert_set.insert(triple.clone()) {
                    self.insert_buffer.push(triple.clone());
                    self.estimated_memory += self.estimate_triple_size(&triple);
                } else {
                    self.stats.deduplicated_operations += 1;
                }
            }
            CoalescingStrategy::OptimizeOrder => {
                // For optimize order, we'll sort later
                if self.insert_set.insert(triple.clone()) {
                    self.insert_buffer.push(triple.clone());
                    self.estimated_memory += self.estimate_triple_size(&triple);
                }
            }
        }
        
        self.check_flush()?;
        Ok(())
    }

    /// Add multiple insert operations
    pub fn insert_batch(&mut self, triples: Vec<Triple>) -> Result<(), OxirsError> {
        for triple in triples {
            self.insert(triple)?;
        }
        Ok(())
    }

    /// Add a remove operation
    pub fn remove(&mut self, triple: Triple) -> Result<(), OxirsError> {
        self.stats.total_operations += 1;
        
        // Apply coalescing
        match self.config.coalescing_strategy {
            CoalescingStrategy::None => {
                self.estimated_memory += self.estimate_triple_size(&triple);
                self.remove_buffer.push(triple);
            }
            CoalescingStrategy::Deduplicate | CoalescingStrategy::Merge => {
                if self.remove_set.insert(triple.clone()) {
                    self.remove_buffer.push(triple.clone());
                    self.estimated_memory += self.estimate_triple_size(&triple);
                } else {
                    self.stats.deduplicated_operations += 1;
                }
            }
            CoalescingStrategy::OptimizeOrder => {
                if self.remove_set.insert(triple.clone()) {
                    self.remove_buffer.push(triple.clone());
                    self.estimated_memory += self.estimate_triple_size(&triple);
                }
            }
        }
        
        self.check_flush()?;
        Ok(())
    }

    /// Add a query operation
    pub fn query(
        &mut self,
        subject: Option<Subject>,
        predicate: Option<Predicate>,
        object: Option<Object>,
    ) -> Result<(), OxirsError> {
        self.stats.total_operations += 1;
        self.query_buffer.push((subject, predicate, object));
        self.estimated_memory += 128; // Rough estimate for query pattern
        
        self.check_flush()?;
        Ok(())
    }

    /// Add a transform operation
    pub fn transform<F>(&mut self, f: F) -> Result<(), OxirsError>
    where
        F: Fn(&Triple) -> Option<Triple> + Send + Sync + 'static,
    {
        self.stats.total_operations += 1;
        self.transform_buffer.push(Arc::new(f));
        self.estimated_memory += 64; // Rough estimate for closure
        
        self.check_flush()?;
        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> &BatchBuilderStats {
        &self.stats
    }

    /// Get the current number of pending operations
    pub fn pending_operations(&self) -> usize {
        self.insert_buffer.len() + 
        self.remove_buffer.len() + 
        self.query_buffer.len() + 
        self.transform_buffer.len()
    }

    /// Check if we should flush based on size or memory constraints
    fn check_flush(&mut self) -> Result<(), OxirsError> {
        if self.config.auto_flush {
            let should_flush = self.pending_operations() >= self.config.max_batch_size ||
                             self.estimated_memory >= self.config.max_memory_usage;
            
            if should_flush {
                self.flush()?;
            }
        }
        Ok(())
    }

    /// Estimate the memory size of a triple
    fn estimate_triple_size(&self, triple: &Triple) -> usize {
        // Rough estimation: 
        // - Each IRI/blank node: ~100 bytes
        // - Each literal: string length + 50 bytes overhead
        // - Triple structure: 24 bytes
        24 + self.estimate_term_size(triple.subject()) +
             self.estimate_term_size(triple.predicate()) +
             self.estimate_object_size(triple.object())
    }

    fn estimate_term_size(&self, _term: &impl std::fmt::Display) -> usize {
        100 // Simplified estimation
    }

    fn estimate_object_size(&self, _object: &Object) -> usize {
        150 // Simplified estimation, literals can be larger
    }

    /// Flush all pending operations into batch operations
    pub fn flush(&mut self) -> Result<Vec<BatchOperation>, OxirsError> {
        let mut operations = Vec::new();
        
        if self.config.coalescing_strategy == CoalescingStrategy::Merge {
            self.apply_merge_coalescing();
        }
        
        // Group operations by type if configured
        if self.config.group_by_type {
            // Optimize order if requested
            if self.config.coalescing_strategy == CoalescingStrategy::OptimizeOrder {
                self.optimize_operation_order();
            }
            
            // Create batches from buffers
            if !self.insert_buffer.is_empty() {
                operations.extend(self.create_insert_batches());
            }
            
            if !self.remove_buffer.is_empty() {
                operations.extend(self.create_remove_batches());
            }
            
            if !self.query_buffer.is_empty() {
                operations.extend(self.create_query_batches());
            }
            
            if !self.transform_buffer.is_empty() {
                operations.extend(self.create_transform_batches());
            }
        } else {
            // Mix operation types in batches
            operations = self.create_mixed_batches();
        }
        
        // Update statistics
        self.stats.batches_created += operations.len();
        self.stats.estimated_memory_usage = self.estimated_memory;
        
        // Clear buffers
        self.clear();
        
        // Call flush callback if set
        if let Some(callback) = &*self.flush_callback.lock() {
            callback(operations.clone());
        }
        
        Ok(operations)
    }

    /// Apply merge coalescing to combine compatible operations
    fn apply_merge_coalescing(&mut self) {
        // Remove inserts that are immediately removed
        if !self.insert_buffer.is_empty() && !self.remove_buffer.is_empty() {
            let remove_set = &self.remove_set;
            let original_len = self.insert_buffer.len();
            self.insert_buffer.retain(|triple| !remove_set.contains(triple));
            let coalesced = original_len - self.insert_buffer.len();
            
            if coalesced > 0 {
                self.stats.coalesced_operations += coalesced;
                // Also remove from remove buffer
                let insert_set = &self.insert_set;
                self.remove_buffer.retain(|triple| !insert_set.contains(triple));
            }
        }
    }

    /// Optimize operation order for better cache locality
    fn optimize_operation_order(&mut self) {
        // Sort by subject for better cache locality
        self.insert_buffer.sort_by(|a, b| {
            a.subject().to_string().cmp(&b.subject().to_string())
        });
        
        self.remove_buffer.sort_by(|a, b| {
            a.subject().to_string().cmp(&b.subject().to_string())
        });
    }

    /// Create insert batches respecting max batch size
    fn create_insert_batches(&mut self) -> Vec<BatchOperation> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        
        for triple in self.insert_buffer.drain(..) {
            current_batch.push(triple);
            if current_batch.len() >= self.config.max_batch_size {
                batches.push(BatchOperation::Insert(std::mem::take(&mut current_batch)));
            }
        }
        
        if !current_batch.is_empty() {
            batches.push(BatchOperation::Insert(current_batch));
        }
        
        batches
    }

    /// Create remove batches respecting max batch size
    fn create_remove_batches(&mut self) -> Vec<BatchOperation> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        
        for triple in self.remove_buffer.drain(..) {
            current_batch.push(triple);
            if current_batch.len() >= self.config.max_batch_size {
                batches.push(BatchOperation::Remove(std::mem::take(&mut current_batch)));
            }
        }
        
        if !current_batch.is_empty() {
            batches.push(BatchOperation::Remove(current_batch));
        }
        
        batches
    }

    /// Create query batches
    fn create_query_batches(&mut self) -> Vec<BatchOperation> {
        self.query_buffer.drain(..)
            .map(|(s, p, o)| BatchOperation::Query {
                subject: s,
                predicate: p,
                object: o,
            })
            .collect()
    }

    /// Create transform batches
    fn create_transform_batches(&mut self) -> Vec<BatchOperation> {
        self.transform_buffer.drain(..)
            .map(BatchOperation::Transform)
            .collect()
    }

    /// Create mixed batches with different operation types
    fn create_mixed_batches(&mut self) -> Vec<BatchOperation> {
        // This is a simplified implementation
        // In a real scenario, you might want to interleave operations more intelligently
        let mut operations = Vec::new();
        
        operations.extend(self.create_insert_batches());
        operations.extend(self.create_remove_batches());
        operations.extend(self.create_query_batches());
        operations.extend(self.create_transform_batches());
        
        operations
    }

    /// Clear all buffers
    fn clear(&mut self) {
        self.insert_buffer.clear();
        self.insert_set.clear();
        self.remove_buffer.clear();
        self.remove_set.clear();
        self.query_buffer.clear();
        self.transform_buffer.clear();
        self.estimated_memory = 0;
    }
}

/// Create a batch configuration from builder config
impl From<&BatchBuilderConfig> for BatchConfig {
    fn from(builder_config: &BatchBuilderConfig) -> Self {
        BatchConfig {
            batch_size: builder_config.max_batch_size,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NamedNode;

    fn create_test_triple(id: usize) -> Triple {
        Triple::new(
            Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", id)).unwrap()),
            Predicate::NamedNode(NamedNode::new(&format!("http://predicate/{}", id)).unwrap()),
            Object::NamedNode(NamedNode::new(&format!("http://object/{}", id)).unwrap()),
        )
    }

    #[test]
    fn test_batch_builder_basic() {
        let config = BatchBuilderConfig {
            max_batch_size: 10,
            auto_flush: false,
            ..Default::default()
        };
        
        let mut builder = BatchBuilder::new(config);
        
        // Add operations
        for i in 0..25 {
            builder.insert(create_test_triple(i)).unwrap();
        }
        
        assert_eq!(builder.pending_operations(), 25);
        
        // Flush and check batches
        let batches = builder.flush().unwrap();
        assert_eq!(batches.len(), 3); // 10 + 10 + 5
        assert_eq!(builder.pending_operations(), 0);
    }

    #[test]
    fn test_deduplication() {
        let config = BatchBuilderConfig {
            coalescing_strategy: CoalescingStrategy::Deduplicate,
            auto_flush: false,
            ..Default::default()
        };
        
        let mut builder = BatchBuilder::new(config);
        
        // Add duplicate triples
        let triple = create_test_triple(1);
        for _ in 0..5 {
            builder.insert(triple.clone()).unwrap();
        }
        
        assert_eq!(builder.pending_operations(), 1);
        assert_eq!(builder.stats().deduplicated_operations, 4);
    }

    #[test]
    fn test_merge_coalescing() {
        let config = BatchBuilderConfig {
            coalescing_strategy: CoalescingStrategy::Merge,
            auto_flush: false,
            ..Default::default()
        };
        
        let mut builder = BatchBuilder::new(config);
        
        // Add insert then remove same triple
        let triple = create_test_triple(1);
        builder.insert(triple.clone()).unwrap();
        builder.remove(triple).unwrap();
        
        // After merge, both should be eliminated
        let batches = builder.flush().unwrap();
        assert_eq!(batches.len(), 0);
        assert_eq!(builder.stats().coalesced_operations, 1);
    }

    #[test]
    fn test_auto_flush() {
        let config = BatchBuilderConfig {
            max_batch_size: 5,
            auto_flush: true,
            ..Default::default()
        };
        
        let flushed_batches = Arc::new(Mutex::new(Vec::new()));
        let flushed_clone = flushed_batches.clone();
        
        let mut builder = BatchBuilder::new(config);
        builder.on_flush(move |batches| {
            flushed_clone.lock().extend(batches);
        });
        
        // Add operations that trigger auto-flush
        for i in 0..12 {
            builder.insert(create_test_triple(i)).unwrap();
        }
        
        // Should have auto-flushed twice
        assert_eq!(flushed_batches.lock().len(), 2);
        assert_eq!(builder.pending_operations(), 2); // 12 % 5
    }

    #[test]
    fn test_mixed_operations() {
        let config = BatchBuilderConfig {
            group_by_type: true,
            auto_flush: false,
            ..Default::default()
        };
        
        let mut builder = BatchBuilder::new(config);
        
        // Add different operation types
        builder.insert(create_test_triple(1)).unwrap();
        builder.remove(create_test_triple(2)).unwrap();
        builder.query(None, None, None).unwrap();
        
        let batches = builder.flush().unwrap();
        
        // Should have 3 batches (one per type)
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_memory_limits() {
        let config = BatchBuilderConfig {
            max_memory_usage: 1000, // Very small limit
            auto_flush: true,
            ..Default::default()
        };
        
        let mut builder = BatchBuilder::new(config);
        
        // Add operations until memory limit
        let mut added = 0;
        for i in 0..100 {
            builder.insert(create_test_triple(i)).unwrap();
            added += 1;
            if builder.pending_operations() == 0 {
                // Auto-flushed due to memory
                break;
            }
        }
        
        // Should have flushed before adding all 100
        assert!(added < 100);
        assert_eq!(builder.stats().batches_created, 1);
    }
}