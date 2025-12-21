//! Bulk operations module for high-throughput RDF processing
//!
//! This module provides advanced bulk processing capabilities using SciRS2-Core's
//! parallel operations, memory-efficient streaming, and optimized batch processing.

use crate::dictionary::{Dictionary, NodeId, Term};
use crate::error::{Result, TdbError};
use crate::index::{SimdTripleFilter, SimdTriplePattern, Triple, TripleIndexes};
use scirs2_core::parallel_ops::{par_chunks, IntoParallelIterator, ParallelIterator};
use scirs2_core::profiling::{Profiler, Timer};
use std::sync::{Arc, Mutex};

/// Bulk operation statistics
#[derive(Debug, Clone, Default)]
pub struct BulkStats {
    /// Total items processed
    pub items_processed: u64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Throughput (items/sec)
    pub throughput: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Parallel efficiency (0.0-1.0)
    pub parallel_efficiency: f64,
}

impl BulkStats {
    /// Calculate throughput from processed items and time
    pub fn calculate_throughput(&mut self) {
        if self.processing_time_ms > 0 {
            self.throughput =
                (self.items_processed as f64 / self.processing_time_ms as f64) * 1000.0;
        }
    }
}

/// Bulk triple processor with parallel pipeline
///
/// Provides high-throughput triple processing using:
/// - Parallel batch processing with SciRS2-Core
/// - SIMD-accelerated filtering
/// - Memory-efficient streaming
/// - Comprehensive statistics tracking
pub struct BulkTripleProcessor {
    /// Batch size for processing
    batch_size: usize,
    /// Number of parallel workers
    num_workers: usize,
    /// SIMD filter for pattern matching
    simd_filter: Arc<Mutex<SimdTripleFilter>>,
    /// Profiler for performance tracking
    profiler: Profiler,
    /// Statistics
    stats: BulkStats,
}

impl BulkTripleProcessor {
    /// Create a new bulk triple processor
    ///
    /// # Arguments
    /// * `batch_size` - Number of triples per batch (default: 10000)
    /// * `num_workers` - Number of parallel workers (default: num_cpus)
    pub fn new(batch_size: usize, num_workers: usize) -> Self {
        Self {
            batch_size,
            num_workers,
            simd_filter: Arc::new(Mutex::new(SimdTripleFilter::new())),
            profiler: Profiler::new(),
            stats: BulkStats::default(),
        }
    }

    /// Create with default settings
    pub fn default_config() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self::new(10000, num_cpus)
    }

    /// Process triples in parallel batches
    ///
    /// Splits the triple collection into batches and processes them
    /// concurrently for maximum throughput.
    pub fn parallel_process<F>(&mut self, triples: &[Triple], process_fn: F) -> Result<Vec<Triple>>
    where
        F: Fn(&Triple) -> bool + Send + Sync,
    {
        let timer = Timer::start("bulk_parallel_process");
        let start = std::time::Instant::now();

        // Process in parallel using SciRS2-Core parallel operations
        let results: Vec<Triple> = triples
            .into_par_iter()
            .filter(|triple| process_fn(triple))
            .copied()
            .collect();

        // Update statistics
        self.stats.items_processed += triples.len() as u64;
        self.stats.processing_time_ms += start.elapsed().as_millis() as u64;
        self.stats.calculate_throughput();

        timer.stop();
        Ok(results)
    }

    /// Filter triples using multiple patterns in parallel
    ///
    /// Efficiently applies multiple filter patterns using SIMD acceleration
    /// and parallel processing.
    pub fn parallel_multi_filter(
        &mut self,
        triples: &[Triple],
        patterns: &[SimdTriplePattern],
    ) -> Result<Vec<Vec<Triple>>> {
        let timer = Timer::start("bulk_multi_filter");
        let start = std::time::Instant::now();

        // Use SIMD filter for batch pattern matching
        let mut filter = self.simd_filter.lock().map_err(|e| {
            TdbError::Unsupported(format!("Failed to acquire SIMD filter lock: {}", e))
        })?;

        let indices = filter.filter_batch(triples, patterns)?;

        // Convert indices to triples in parallel
        let results: Vec<Vec<Triple>> = indices
            .into_par_iter()
            .map(|idx_vec| idx_vec.iter().map(|&i| triples[i]).collect())
            .collect();

        // Update statistics
        self.stats.items_processed += (triples.len() * patterns.len()) as u64;
        self.stats.processing_time_ms += start.elapsed().as_millis() as u64;
        self.stats.calculate_throughput();

        timer.stop();
        Ok(results)
    }

    /// Batch insert triples into indexes
    ///
    /// Performs parallel batch insertion with optimal throughput.
    pub fn batch_insert(
        &mut self,
        triples: &[Triple],
        indexes: &mut TripleIndexes,
    ) -> Result<usize> {
        let timer = Timer::start("bulk_batch_insert");
        let start = std::time::Instant::now();

        let mut inserted = 0;

        // Process in chunks for memory efficiency
        for chunk in triples.chunks(self.batch_size) {
            for triple in chunk {
                indexes.insert(*triple)?;
                inserted += 1;
            }
        }

        // Update statistics
        self.stats.items_processed += triples.len() as u64;
        self.stats.processing_time_ms += start.elapsed().as_millis() as u64;
        self.stats.calculate_throughput();

        timer.stop();
        Ok(inserted)
    }

    /// Parallel count matching triples
    ///
    /// Efficiently counts triples matching a pattern using parallel processing.
    pub fn parallel_count(
        &mut self,
        triples: &[Triple],
        pattern: &SimdTriplePattern,
    ) -> Result<u64> {
        let timer = Timer::start("bulk_parallel_count");
        let start = std::time::Instant::now();

        // Use SIMD filter for efficient counting
        let mut filter = self.simd_filter.lock().map_err(|e| {
            TdbError::Unsupported(format!("Failed to acquire SIMD filter lock: {}", e))
        })?;

        let count = filter.count_matches(triples, pattern);

        // Update statistics
        self.stats.items_processed += triples.len() as u64;
        self.stats.processing_time_ms += start.elapsed().as_millis() as u64;
        self.stats.calculate_throughput();

        timer.stop();
        Ok(count)
    }

    /// Parallel aggregation over triples
    ///
    /// Performs parallel map-reduce style aggregation.
    pub fn parallel_aggregate<T, M, R>(
        &mut self,
        triples: &[Triple],
        map_fn: M,
        reduce_fn: R,
        initial: T,
    ) -> Result<T>
    where
        T: Send + Sync + Clone,
        M: Fn(&Triple) -> T + Send + Sync,
        R: Fn(T, T) -> T + Send + Sync,
    {
        let timer = Timer::start("bulk_parallel_aggregate");
        let start = std::time::Instant::now();

        // Parallel map-reduce using SciRS2-Core
        let result = triples
            .into_par_iter()
            .map(map_fn)
            .reduce(|| initial.clone(), reduce_fn);

        // Update statistics
        self.stats.items_processed += triples.len() as u64;
        self.stats.processing_time_ms += start.elapsed().as_millis() as u64;
        self.stats.calculate_throughput();

        timer.stop();
        Ok(result)
    }

    /// Parallel group by operation
    ///
    /// Groups triples by a key function in parallel.
    pub fn parallel_group_by<K, F>(
        &mut self,
        triples: &[Triple],
        key_fn: F,
    ) -> Result<std::collections::HashMap<K, Vec<Triple>>>
    where
        K: std::hash::Hash + Eq + Send + Sync + Clone,
        F: Fn(&Triple) -> K + Send + Sync,
    {
        let timer = Timer::start("bulk_parallel_group_by");
        let start = std::time::Instant::now();

        use std::collections::HashMap;

        // Build groups in parallel
        let groups: HashMap<K, Vec<Triple>> = triples
            .into_par_iter()
            .fold(
                || HashMap::new(),
                |mut acc, triple| {
                    let key = key_fn(triple);
                    acc.entry(key).or_insert_with(Vec::new).push(*triple);
                    acc
                },
            )
            .reduce(
                || HashMap::new(),
                |mut acc1, acc2| {
                    for (key, mut values) in acc2 {
                        acc1.entry(key).or_insert_with(Vec::new).append(&mut values);
                    }
                    acc1
                },
            );

        // Update statistics
        self.stats.items_processed += triples.len() as u64;
        self.stats.processing_time_ms += start.elapsed().as_millis() as u64;
        self.stats.calculate_throughput();

        timer.stop();
        Ok(groups)
    }

    /// Get bulk operation statistics
    pub fn stats(&self) -> &BulkStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BulkStats::default();
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

/// Streaming triple iterator for memory-efficient processing
///
/// Processes large triple collections without loading everything into memory.
pub struct StreamingTripleIterator<'a> {
    /// Source triples
    triples: &'a [Triple],
    /// Current position
    position: usize,
    /// Chunk size for streaming
    chunk_size: usize,
}

impl<'a> StreamingTripleIterator<'a> {
    /// Create a new streaming iterator
    pub fn new(triples: &'a [Triple], chunk_size: usize) -> Self {
        Self {
            triples,
            position: 0,
            chunk_size,
        }
    }

    /// Get next chunk of triples
    pub fn next_chunk(&mut self) -> Option<&'a [Triple]> {
        if self.position >= self.triples.len() {
            return None;
        }

        let end = (self.position + self.chunk_size).min(self.triples.len());
        let chunk = &self.triples[self.position..end];
        self.position = end;

        Some(chunk)
    }

    /// Check if there are more chunks
    pub fn has_more(&self) -> bool {
        self.position < self.triples.len()
    }

    /// Get remaining items count
    pub fn remaining(&self) -> usize {
        self.triples.len() - self.position
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// Parallel triple pipeline builder
///
/// Fluent API for building parallel processing pipelines.
pub struct ParallelPipelineBuilder {
    batch_size: usize,
    num_workers: usize,
}

impl ParallelPipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            batch_size: 10000,
            num_workers: num_cpus,
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set number of workers
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.num_workers = workers;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> BulkTripleProcessor {
        BulkTripleProcessor::new(self.batch_size, self.num_workers)
    }
}

impl Default for ParallelPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_triples(count: usize) -> Vec<Triple> {
        (0..count)
            .map(|i| Triple {
                subject: NodeId::new((i / 100) as u64 + 1),
                predicate: NodeId::new((i / 10) as u64 + 1),
                object: NodeId::new(i as u64 + 1),
            })
            .collect()
    }

    #[test]
    fn test_bulk_processor_creation() {
        let processor = BulkTripleProcessor::default_config();
        assert!(processor.batch_size() > 0);
        assert!(processor.num_workers() > 0);
    }

    #[test]
    fn test_parallel_process() {
        let mut processor = BulkTripleProcessor::default_config();
        let triples = create_test_triples(1000);

        let results = processor
            .parallel_process(&triples, |t| t.subject.as_u64() < 5)
            .unwrap();

        assert!(!results.is_empty());
        for triple in &results {
            assert!(triple.subject.as_u64() < 5);
        }
    }

    #[test]
    fn test_parallel_multi_filter() {
        let mut processor = BulkTripleProcessor::default_config();
        let triples = create_test_triples(1000);

        let patterns = vec![
            SimdTriplePattern::with_subject(NodeId::new(1)),
            SimdTriplePattern::with_predicate(NodeId::new(10)),
        ];

        let results = processor
            .parallel_multi_filter(&triples, &patterns)
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parallel_count() {
        let mut processor = BulkTripleProcessor::default_config();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::with_subject(NodeId::new(5));

        let count = processor.parallel_count(&triples, &pattern).unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_parallel_aggregate() {
        let mut processor = BulkTripleProcessor::default_config();
        let triples = create_test_triples(100);

        let sum = processor
            .parallel_aggregate(&triples, |t| t.subject.as_u64(), |a, b| a + b, 0u64)
            .unwrap();

        assert!(sum > 0);
    }

    #[test]
    fn test_parallel_group_by() {
        let mut processor = BulkTripleProcessor::default_config();
        let triples = create_test_triples(100);

        let groups = processor
            .parallel_group_by(&triples, |t| t.subject)
            .unwrap();

        assert!(!groups.is_empty());
    }

    #[test]
    fn test_streaming_iterator() {
        let triples = create_test_triples(1000);
        let mut iterator = StreamingTripleIterator::new(&triples, 100);

        let mut total_processed = 0;
        while let Some(chunk) = iterator.next_chunk() {
            total_processed += chunk.len();
            assert!(chunk.len() <= 100);
        }

        assert_eq!(total_processed, 1000);
    }

    #[test]
    fn test_streaming_iterator_reset() {
        let triples = create_test_triples(100);
        let mut iterator = StreamingTripleIterator::new(&triples, 10);

        iterator.next_chunk();
        iterator.next_chunk();
        assert!(iterator.has_more());

        iterator.reset();
        assert_eq!(iterator.remaining(), 100);
    }

    #[test]
    fn test_bulk_stats_calculation() {
        let mut stats = BulkStats {
            items_processed: 10000,
            processing_time_ms: 1000,
            ..Default::default()
        };

        stats.calculate_throughput();
        assert_eq!(stats.throughput, 10000.0);
    }

    #[test]
    fn test_pipeline_builder() {
        let processor = ParallelPipelineBuilder::new()
            .with_batch_size(5000)
            .with_workers(4)
            .build();

        assert_eq!(processor.batch_size(), 5000);
        assert_eq!(processor.num_workers(), 4);
    }

    #[test]
    fn test_stats_reset() {
        let mut processor = BulkTripleProcessor::default_config();
        let triples = create_test_triples(100);
        let pattern = SimdTriplePattern::any();

        processor.parallel_count(&triples, &pattern).unwrap();
        assert!(processor.stats().items_processed > 0);

        processor.reset_stats();
        assert_eq!(processor.stats().items_processed, 0);
    }
}
