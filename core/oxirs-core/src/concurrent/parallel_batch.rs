//! Parallel batch processing for high-throughput RDF operations
//!
//! This module provides a parallel batch processor with work-stealing queues,
//! configurable thread pools, and progress tracking for efficient RDF data processing.

use crate::model::{Object, Predicate, Subject, Triple};
use crate::OxirsError;
use crossbeam_deque::Injector;
use parking_lot::{Mutex, RwLock};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Batch operation types
#[derive(Clone)]
pub enum BatchOperation {
    /// Insert a collection of triples
    Insert(Vec<Triple>),
    /// Remove a collection of triples
    Remove(Vec<Triple>),
    /// Execute a query with pattern matching
    Query {
        subject: Option<Subject>,
        predicate: Option<Predicate>,
        object: Option<Object>,
    },
    /// Transform triples using a function
    Transform(Arc<dyn Fn(&Triple) -> Option<Triple> + Send + Sync>),
}

impl std::fmt::Debug for BatchOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchOperation::Insert(triples) => write!(f, "Insert({} triples)", triples.len()),
            BatchOperation::Remove(triples) => write!(f, "Remove({} triples)", triples.len()),
            BatchOperation::Query {
                subject,
                predicate,
                object,
            } => {
                write!(f, "Query({subject:?}, {predicate:?}, {object:?})")
            }
            BatchOperation::Transform(_) => write!(f, "Transform(function)"),
        }
    }
}

/// Progress callback for tracking batch operations
pub type ProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Configuration for parallel batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of worker threads (defaults to number of CPU cores)
    pub num_threads: Option<usize>,
    /// Size of each batch for processing
    pub batch_size: usize,
    /// Maximum queue size before applying backpressure
    pub max_queue_size: usize,
    /// Timeout for batch operations
    pub timeout: Option<Duration>,
    /// Enable progress tracking
    pub enable_progress: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        let num_cpus = num_cpus::get();
        BatchConfig {
            num_threads: None,
            batch_size: 1000,
            max_queue_size: num_cpus * 10000,
            timeout: None,
            enable_progress: true,
        }
    }
}

impl BatchConfig {
    /// Create a config optimized for the current system
    pub fn auto() -> Self {
        let num_cpus = num_cpus::get();
        let total_memory = sys_info::mem_info()
            .map(|info| info.total)
            .unwrap_or(8 * 1024 * 1024); // 8GB default

        // Adjust batch size based on available memory
        let batch_size = if total_memory > 16 * 1024 * 1024 {
            5000
        } else if total_memory > 8 * 1024 * 1024 {
            2000
        } else {
            1000
        };

        BatchConfig {
            num_threads: Some(num_cpus),
            batch_size,
            max_queue_size: num_cpus * batch_size * 10,
            timeout: None,
            enable_progress: true,
        }
    }
}

/// Statistics for batch processing
#[derive(Debug, Default)]
pub struct BatchStats {
    pub total_processed: AtomicUsize,
    pub total_succeeded: AtomicUsize,
    pub total_failed: AtomicUsize,
    pub processing_time_ms: AtomicUsize,
}

impl BatchStats {
    /// Get a summary of the statistics
    pub fn summary(&self) -> BatchStatsSummary {
        BatchStatsSummary {
            total_processed: self.total_processed.load(Ordering::Relaxed),
            total_succeeded: self.total_succeeded.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            processing_time_ms: self.processing_time_ms.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchStatsSummary {
    pub total_processed: usize,
    pub total_succeeded: usize,
    pub total_failed: usize,
    pub processing_time_ms: usize,
}

/// Parallel batch processor with work-stealing queues
pub struct ParallelBatchProcessor {
    config: BatchConfig,
    /// Global work queue (injector)
    injector: Arc<Injector<BatchOperation>>,
    /// Cancellation flag
    cancelled: Arc<AtomicBool>,
    /// Processing statistics
    stats: Arc<BatchStats>,
    /// Progress callback
    progress_callback: Arc<Mutex<Option<ProgressCallback>>>,
    /// Error accumulator
    errors: Arc<RwLock<Vec<OxirsError>>>,
}

impl ParallelBatchProcessor {
    /// Create a new parallel batch processor
    pub fn new(config: BatchConfig) -> Self {
        let injector = Arc::new(Injector::new());

        ParallelBatchProcessor {
            config,
            injector,
            cancelled: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(BatchStats::default()),
            progress_callback: Arc::new(Mutex::new(None)),
            errors: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set a progress callback
    pub fn set_progress_callback<F>(&self, callback: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        *self.progress_callback.lock() = Some(Box::new(callback));
    }

    /// Cancel ongoing operations
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if operations are cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Get current statistics
    pub fn stats(&self) -> BatchStatsSummary {
        self.stats.summary()
    }

    /// Get accumulated errors
    pub fn errors(&self) -> Vec<OxirsError> {
        self.errors.read().clone()
    }

    /// Clear accumulated errors
    pub fn clear_errors(&self) {
        self.errors.write().clear();
    }

    /// Submit a batch operation
    pub fn submit(&self, operation: BatchOperation) -> Result<(), OxirsError> {
        // Check queue size for backpressure
        if self.injector.len() > self.config.max_queue_size {
            return Err(OxirsError::Store("Queue is full".to_string()));
        }

        self.injector.push(operation);
        Ok(())
    }

    /// Submit multiple operations
    pub fn submit_batch(&self, operations: Vec<BatchOperation>) -> Result<(), OxirsError> {
        // Check if adding these operations would exceed queue size
        if self.injector.len() + operations.len() > self.config.max_queue_size {
            return Err(OxirsError::Store("Queue would overflow".to_string()));
        }

        for op in operations {
            self.injector.push(op);
        }
        Ok(())
    }

    /// Process operations with the given executor
    pub fn process<E, R>(&self, executor: E) -> Result<Vec<R>, OxirsError>
    where
        E: Fn(BatchOperation) -> Result<R, OxirsError> + Send + Sync + 'static,
        R: Send + 'static,
    {
        let start_time = Instant::now();
        let num_threads = self.config.num_threads.unwrap_or_else(num_cpus::get);
        let barrier = Arc::new(Barrier::new(num_threads + 1));
        let executor = Arc::new(executor);
        let results = Arc::new(Mutex::new(Vec::new()));

        // Reset cancellation flag
        self.cancelled.store(false, Ordering::SeqCst);

        // Spawn worker threads
        let handles: Vec<_> = (0..num_threads)
            .map(|_worker_id| {
                let injector = self.injector.clone();
                let cancelled = self.cancelled.clone();
                let stats = self.stats.clone();
                let executor = executor.clone();
                let results = results.clone();
                let errors = self.errors.clone();
                let barrier = barrier.clone();
                let progress_callback = self.progress_callback.clone();
                let enable_progress = self.config.enable_progress;

                thread::spawn(move || {
                    // Wait for all threads to be ready
                    barrier.wait();

                    loop {
                        // Check for cancellation
                        if cancelled.load(Ordering::SeqCst) {
                            break;
                        }

                        // Try to get work from global queue
                        let task = loop {
                            match injector.steal() {
                                crossbeam_deque::Steal::Success(task) => break Some(task),
                                crossbeam_deque::Steal::Empty => break None,
                                crossbeam_deque::Steal::Retry => continue,
                            }
                        };

                        match task {
                            Some(operation) => {
                                // Process the operation
                                let processed =
                                    stats.total_processed.fetch_add(1, Ordering::Relaxed) + 1;

                                // Report progress
                                if enable_progress && processed % 100 == 0 {
                                    if let Some(callback) = &*progress_callback.lock() {
                                        let total = injector.len() + processed;
                                        callback(processed, total);
                                    }
                                }

                                match executor(operation) {
                                    Ok(result) => {
                                        stats.total_succeeded.fetch_add(1, Ordering::Relaxed);
                                        results.lock().push(result);
                                    }
                                    Err(e) => {
                                        stats.total_failed.fetch_add(1, Ordering::Relaxed);
                                        errors.write().push(e);
                                    }
                                }
                            }
                            None => {
                                // No work available, check if we're done
                                if injector.is_empty() {
                                    break;
                                }
                                // Brief sleep to avoid busy-waiting
                                thread::sleep(Duration::from_micros(10));
                            }
                        }
                    }
                })
            })
            .collect();

        // Signal all threads to start
        barrier.wait();

        // Wait for completion or timeout
        if let Some(timeout) = self.config.timeout {
            let deadline = Instant::now() + timeout;
            for handle in handles {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    self.cancel();
                    return Err(OxirsError::Store("Operation timed out".to_string()));
                }
                // Note: We can't actually join with timeout in std, would need a different approach
                handle
                    .join()
                    .map_err(|_| OxirsError::Store("Worker thread panicked".to_string()))?;
            }
        } else {
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| OxirsError::Store("Worker thread panicked".to_string()))?;
            }
        }

        // Record processing time
        let elapsed = start_time.elapsed();
        self.stats
            .processing_time_ms
            .store(elapsed.as_millis() as usize, Ordering::Relaxed);

        // Check for errors
        let errors = self.errors.read();
        if !errors.is_empty() {
            return Err(OxirsError::Store(format!(
                "Batch processing failed with {} errors",
                errors.len()
            )));
        }

        // Extract results
        let final_results = Arc::try_unwrap(results)
            .map_err(|_| OxirsError::Store("Failed to extract results from Arc".to_string()))?
            .into_inner();

        Ok(final_results)
    }

    /// Process operations in parallel using rayon
    #[cfg(feature = "parallel")]
    pub fn process_rayon<E, R>(&self, executor: E) -> Result<Vec<R>, OxirsError>
    where
        E: Fn(BatchOperation) -> Result<R, OxirsError> + Send + Sync,
        R: Send,
    {
        let start_time = Instant::now();

        // Collect all operations from the queue
        let mut operations = Vec::new();
        loop {
            match self.injector.steal() {
                crossbeam_deque::Steal::Success(op) => {
                    if self.is_cancelled() {
                        return Err(OxirsError::Store("Operation cancelled".to_string()));
                    }
                    operations.push(op);
                }
                crossbeam_deque::Steal::Empty => break,
                crossbeam_deque::Steal::Retry => continue,
            }
        }

        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.num_threads.unwrap_or_else(num_cpus::get))
            .build()
            .map_err(|e| OxirsError::Store(format!("Failed to build thread pool: {}", e)))?;

        // Clone needed references
        let cancelled = self.cancelled.clone();
        let stats = self.stats.clone();
        let errors = self.errors.clone();
        let batch_size = self.config.batch_size;
        let executor = Arc::new(executor);

        // Process in parallel
        let results = pool.install(move || {
            operations
                .into_par_iter()
                .chunks(batch_size)
                .map(move |chunk| {
                    let mut chunk_results = Vec::new();
                    for op in chunk {
                        if cancelled.load(Ordering::SeqCst) {
                            return Err(OxirsError::Store("Operation cancelled".to_string()));
                        }

                        stats.total_processed.fetch_add(1, Ordering::Relaxed);

                        match executor(op) {
                            Ok(result) => {
                                stats.total_succeeded.fetch_add(1, Ordering::Relaxed);
                                chunk_results.push(result);
                            }
                            Err(e) => {
                                stats.total_failed.fetch_add(1, Ordering::Relaxed);
                                errors.write().push(e.clone());
                                return Err(e);
                            }
                        }
                    }
                    Ok(chunk_results)
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        // Flatten results
        let results: Vec<R> = results.into_iter().flatten().collect();

        // Record processing time
        let elapsed = start_time.elapsed();
        self.stats
            .processing_time_ms
            .store(elapsed.as_millis() as usize, Ordering::Relaxed);

        Ok(results)
    }
}

/// Helper functions for creating batch operations
impl BatchOperation {
    /// Create an insert operation
    pub fn insert(triples: Vec<Triple>) -> Self {
        BatchOperation::Insert(triples)
    }

    /// Create a remove operation
    pub fn remove(triples: Vec<Triple>) -> Self {
        BatchOperation::Remove(triples)
    }

    /// Create a query operation
    pub fn query(
        subject: Option<Subject>,
        predicate: Option<Predicate>,
        object: Option<Object>,
    ) -> Self {
        BatchOperation::Query {
            subject,
            predicate,
            object,
        }
    }

    /// Create a transform operation
    pub fn transform<F>(f: F) -> Self
    where
        F: Fn(&Triple) -> Option<Triple> + Send + Sync + 'static,
    {
        BatchOperation::Transform(Arc::new(f))
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
    fn test_parallel_batch_processor() {
        let config = BatchConfig::default();
        let processor = ParallelBatchProcessor::new(config);

        // Submit some operations
        let operations: Vec<_> = (0..1000)
            .map(|i| BatchOperation::insert(vec![create_test_triple(i)]))
            .collect();

        processor.submit_batch(operations).unwrap();

        // Process with a simple executor
        let results = processor
            .process(|op| -> Result<usize, OxirsError> {
                match op {
                    BatchOperation::Insert(triples) => Ok(triples.len()),
                    _ => Ok(0),
                }
            })
            .unwrap();

        assert_eq!(results.len(), 1000);
        assert_eq!(results.iter().sum::<usize>(), 1000);

        let stats = processor.stats();
        assert_eq!(stats.total_processed, 1000);
        assert_eq!(stats.total_succeeded, 1000);
        assert_eq!(stats.total_failed, 0);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_work_stealing() {
        let config = BatchConfig {
            num_threads: Some(4),
            batch_size: 10,
            ..Default::default()
        };

        let processor = ParallelBatchProcessor::new(config);

        // Submit operations
        for i in 0..100 {
            processor
                .submit(BatchOperation::insert(vec![create_test_triple(i)]))
                .unwrap();
        }

        // Process and verify work is distributed
        let results = processor
            .process_rayon(|op| -> Result<usize, OxirsError> {
                // Simulate some work
                thread::sleep(Duration::from_micros(100));
                match op {
                    BatchOperation::Insert(triples) => Ok(triples.len()),
                    _ => Ok(0),
                }
            })
            .unwrap();

        assert_eq!(results.len(), 100);
        let stats = processor.stats();
        assert_eq!(stats.total_processed, 100);
    }

    #[test]
    fn test_error_handling() {
        let config = BatchConfig::default();
        let processor = ParallelBatchProcessor::new(config);

        // Submit operations that will fail
        for i in 0..10 {
            processor
                .submit(BatchOperation::insert(vec![create_test_triple(i)]))
                .unwrap();
        }

        // Process with failing executor
        let result = processor.process(|_op| -> Result<(), OxirsError> {
            Err(OxirsError::Store("Test error".to_string()))
        });

        assert!(result.is_err());
        let stats = processor.stats();
        assert_eq!(stats.total_failed, 10);
        assert_eq!(processor.errors().len(), 10);
    }

    #[test]
    fn test_cancellation() {
        let config = BatchConfig::default();
        let processor = Arc::new(ParallelBatchProcessor::new(config));

        // Submit many operations
        for i in 0..1000 {
            processor
                .submit(BatchOperation::insert(vec![create_test_triple(i)]))
                .unwrap();
        }

        // Start processing in a thread
        let processor_thread = processor.clone();

        let handle = thread::spawn(move || {
            processor_thread.process(|op| -> Result<(), OxirsError> {
                // Simulate slow processing
                thread::sleep(Duration::from_millis(10));
                match op {
                    BatchOperation::Insert(_) => Ok(()),
                    _ => Ok(()),
                }
            })
        });

        // Cancel after a short delay
        thread::sleep(Duration::from_millis(50));
        processor.cancel();

        // Wait for completion
        let result = handle.join().unwrap();

        // Should have processed some but not all
        let stats = processor.stats();
        assert!(stats.total_processed < 1000);
        assert!(processor.is_cancelled());
    }

    #[test]
    fn test_progress_tracking() {
        let config = BatchConfig::default();
        let processor = ParallelBatchProcessor::new(config);

        let progress_count = Arc::new(AtomicUsize::new(0));
        let progress_count_clone = progress_count.clone();

        processor.set_progress_callback(move |current, _total| {
            progress_count_clone.fetch_add(1, Ordering::Relaxed);
            println!("Progress: {}/{}", current, _total);
        });

        // Submit operations
        for i in 0..500 {
            processor
                .submit(BatchOperation::insert(vec![create_test_triple(i)]))
                .unwrap();
        }

        // Process
        processor
            .process(|op| -> Result<(), OxirsError> {
                match op {
                    BatchOperation::Insert(_) => Ok(()),
                    _ => Ok(()),
                }
            })
            .unwrap();

        // Should have received progress updates
        assert!(progress_count.load(Ordering::Relaxed) > 0);
    }
}
