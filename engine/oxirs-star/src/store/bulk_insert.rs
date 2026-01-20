//! Bulk insertion configuration for optimized batch operations.
//!
//! This module provides configuration for bulk insertion operations,
//! enabling efficient batch processing of RDF-star triples.

/// Bulk insertion configuration for optimized batch operations
#[derive(Debug, Clone)]
pub struct BulkInsertConfig {
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Disable index updates during bulk insertion
    pub defer_index_updates: bool,
    /// Memory threshold before flushing (bytes)
    pub memory_threshold: usize,
    /// Enable parallel processing for large batches
    pub parallel_processing: bool,
    /// Number of worker threads for parallel insertion
    pub worker_threads: usize,
}

impl Default for BulkInsertConfig {
    fn default() -> Self {
        Self {
            batch_size: 10000,
            defer_index_updates: true,
            memory_threshold: 256 * 1024 * 1024, // 256MB
            parallel_processing: true,
            worker_threads: std::cmp::min(8, 4), // Use 4 as default thread count
        }
    }
}
