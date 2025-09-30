//! Parallel validation capabilities for SHACL constraint evaluation
//!
//! This module provides thread-safe parallel processing for SHACL validation,
//! with work distribution, load balancing, and performance monitoring.

use crate::{
    constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult},
    optimization::core::{ConstraintCache, ConstraintDependencyAnalyzer},
    report::ValidationReport,
    Result, ShaclError, Shape, ShapeId,
};
use indexmap::IndexMap;
use oxirs_core::{model::Term, Store};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, RwLock,
};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for parallel validation
#[derive(Debug, Clone)]
pub struct ParallelValidationConfig {
    /// Maximum number of worker threads
    pub max_threads: usize,
    /// Work batch size per thread
    pub batch_size: usize,
    /// Enable work stealing between threads
    pub enable_work_stealing: bool,
    /// Thread pool idle timeout
    pub idle_timeout: Duration,
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Maximum queue size per worker
    pub max_queue_size: usize,
}

impl Default for ParallelValidationConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            max_threads: num_cpus,
            batch_size: 100,
            enable_work_stealing: true,
            idle_timeout: Duration::from_secs(30),
            enable_load_balancing: true,
            max_queue_size: 1000,
        }
    }
}

/// Work item for parallel validation
#[derive(Debug, Clone)]
struct ValidationWorkItem {
    /// Constraint to evaluate
    constraint: Constraint,
    /// Evaluation context
    context: ConstraintContext,
    /// Priority (lower values = higher priority)
    priority: usize,
    /// Estimated execution cost
    estimated_cost: f64,
}

/// Result of parallel validation work
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ValidationWorkResult {
    /// Work item that was processed
    work_item: ValidationWorkItem,
    /// Evaluation result
    result: ConstraintEvaluationResult,
    /// Actual execution time
    execution_time: Duration,
    /// Worker thread ID that processed this
    worker_id: usize,
}

/// Statistics for worker thread performance
#[derive(Debug, Clone, Default)]
pub struct WorkerStats {
    /// Number of work items processed
    pub items_processed: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per item
    pub avg_processing_time: Duration,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Work items stolen from other threads
    pub work_stolen: usize,
    /// Work items stolen by other threads
    pub work_given: usize,
}

/// Statistics for the entire worker pool
#[derive(Debug, Clone)]
pub struct WorkerPoolStats {
    /// Statistics per worker thread
    pub worker_stats: HashMap<usize, WorkerStats>,
    /// Total work items processed
    pub total_items_processed: usize,
    /// Total parallel execution time
    pub total_execution_time: Duration,
    /// Parallel efficiency ratio (actual speedup / theoretical max)
    pub parallel_efficiency: f64,
    /// Load balancing effectiveness
    pub load_balance_score: f64,
    /// Cache effectiveness across all workers
    pub overall_cache_hit_rate: f64,
}

/// Parallel validation engine
#[derive(Debug)]
pub struct ParallelValidationEngine {
    /// Configuration
    config: ParallelValidationConfig,
    /// Shared constraint cache
    cache: Arc<ConstraintCache>,
    /// Dependency analyzer for work prioritization
    dependency_analyzer: Arc<ConstraintDependencyAnalyzer>,
    /// Worker pool statistics
    stats: Arc<RwLock<WorkerPoolStats>>,
    /// Active worker count
    active_workers: Arc<AtomicUsize>,
}

impl ParallelValidationEngine {
    /// Create a new parallel validation engine
    pub fn new(config: ParallelValidationConfig) -> Self {
        let cache = Arc::new(ConstraintCache::new(
            50000, // Large cache for parallel processing
            Duration::from_secs(600),
        ));

        let dependency_analyzer = Arc::new(ConstraintDependencyAnalyzer::default());

        let stats = Arc::new(RwLock::new(WorkerPoolStats {
            worker_stats: HashMap::new(),
            total_items_processed: 0,
            total_execution_time: Duration::ZERO,
            parallel_efficiency: 0.0,
            load_balance_score: 0.0,
            overall_cache_hit_rate: 0.0,
        }));

        Self {
            config,
            cache,
            dependency_analyzer,
            stats,
            active_workers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Validate constraints in parallel
    pub fn validate_parallel(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        focus_nodes: &[Term],
    ) -> Result<ParallelValidationResult> {
        let start_time = Instant::now();

        // Prepare work items
        let work_items = self.prepare_work_items(store, shapes, focus_nodes)?;
        let total_work_items = work_items.len();

        if work_items.is_empty() {
            return Ok(ParallelValidationResult {
                validation_report: ValidationReport::new(),
                total_work_items: 0,
                processed_items: 0,
                parallel_execution_time: Duration::ZERO,
                speedup_ratio: 1.0,
                efficiency_score: 1.0,
            });
        }

        // Determine optimal number of threads
        let num_threads = std::cmp::min(
            self.config.max_threads,
            std::cmp::max(1, work_items.len() / self.config.batch_size),
        );

        // Split work among threads
        let work_chunks = self.distribute_work(work_items, num_threads);

        // Execute parallel validation
        let results = self.execute_parallel_validation(store, work_chunks, num_threads)?;

        // Combine results
        let mut validation_report = ValidationReport::new();
        for result in &results {
            if let Some(violation) = result.result.clone().into_violation() {
                validation_report.add_violation(violation);
            }
        }

        let total_execution_time = start_time.elapsed();

        // Calculate performance metrics
        let speedup_ratio = self.calculate_speedup_ratio(total_execution_time, &results);
        let efficiency_score = speedup_ratio / num_threads as f64;

        // Update statistics
        self.update_pool_statistics(total_execution_time, &results)?;

        Ok(ParallelValidationResult {
            validation_report,
            total_work_items,
            processed_items: results.len(),
            parallel_execution_time: total_execution_time,
            speedup_ratio,
            efficiency_score,
        })
    }

    /// Prepare work items from shapes and focus nodes
    fn prepare_work_items(
        &self,
        _store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        focus_nodes: &[Term],
    ) -> Result<Vec<ValidationWorkItem>> {
        let mut work_items = Vec::new();

        for (shape_id, shape) in shapes {
            if !shape.is_active() {
                continue;
            }

            for focus_node in focus_nodes {
                for (_, constraint) in &shape.constraints {
                    let values = vec![focus_node.clone()]; // Simplified for parallel processing

                    let mut context = ConstraintContext::new(focus_node.clone(), shape_id.clone())
                        .with_values(values);
                    if let Some(path) = &shape.path {
                        context = context.with_path(path.clone());
                    }

                    let estimated_cost = self
                        .dependency_analyzer
                        .estimate_constraint_cost(constraint);
                    let priority = self.calculate_priority(constraint, estimated_cost);

                    work_items.push(ValidationWorkItem {
                        constraint: constraint.clone(),
                        context,
                        priority,
                        estimated_cost,
                    });
                }
            }
        }

        // Sort by priority and cost for optimal scheduling
        work_items.sort_by(|a, b| {
            a.priority.cmp(&b.priority).then_with(|| {
                a.estimated_cost
                    .partial_cmp(&b.estimated_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        Ok(work_items)
    }

    /// Calculate priority for a constraint (lower = higher priority)
    fn calculate_priority(&self, constraint: &Constraint, estimated_cost: f64) -> usize {
        let selectivity = self
            .dependency_analyzer
            .estimate_constraint_selectivity(constraint);

        // High selectivity and low cost = high priority (low value)
        let priority_score = selectivity * estimated_cost;
        (priority_score * 1000.0) as usize
    }

    /// Distribute work among threads
    fn distribute_work(
        &self,
        work_items: Vec<ValidationWorkItem>,
        num_threads: usize,
    ) -> Vec<Vec<ValidationWorkItem>> {
        let mut chunks = vec![Vec::new(); num_threads];

        if self.config.enable_load_balancing {
            // Load-balanced distribution based on estimated cost
            let mut thread_loads = vec![0.0; num_threads];

            for item in work_items {
                // Find thread with minimum load
                let min_load_thread = thread_loads
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                thread_loads[min_load_thread] += item.estimated_cost;
                chunks[min_load_thread].push(item);
            }
        } else {
            // Simple round-robin distribution
            for (i, item) in work_items.into_iter().enumerate() {
                chunks[i % num_threads].push(item);
            }
        }

        chunks
    }

    /// Execute parallel validation with worker threads
    fn execute_parallel_validation(
        &self,
        _store: &dyn Store,
        work_chunks: Vec<Vec<ValidationWorkItem>>,
        num_threads: usize,
    ) -> Result<Vec<ValidationWorkResult>> {
        let results = Arc::new(Mutex::new(Vec::new()));
        let work_queues: Arc<Vec<Mutex<Vec<ValidationWorkItem>>>> =
            Arc::new(work_chunks.into_iter().map(Mutex::new).collect());

        let mut handles = Vec::new();

        for worker_id in 0..num_threads {
            let cache = Arc::clone(&self.cache);
            let work_queues = Arc::clone(&work_queues);
            let results = Arc::clone(&results);
            let config = self.config.clone();
            let active_workers = Arc::clone(&self.active_workers);

            // Note: This is a simplified implementation
            // In practice, we'd need a thread-safe Store interface
            let handle = thread::spawn(move || {
                active_workers.fetch_add(1, Ordering::SeqCst);

                let worker_results = Self::worker_thread(worker_id, cache, work_queues, config);

                // Collect results
                if let Ok(mut results_guard) = results.lock() {
                    results_guard.extend(worker_results);
                }

                active_workers.fetch_sub(1, Ordering::SeqCst);
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            handle
                .join()
                .map_err(|_| ShaclError::ValidationEngine("Worker thread panicked".to_string()))?;
        }

        let final_results = results.lock().unwrap().clone();
        Ok(final_results)
    }

    /// Worker thread implementation
    fn worker_thread(
        worker_id: usize,
        cache: Arc<ConstraintCache>,
        work_queues: Arc<Vec<Mutex<Vec<ValidationWorkItem>>>>,
        config: ParallelValidationConfig,
    ) -> Vec<ValidationWorkResult> {
        let mut worker_results = Vec::new();
        let my_queue_id = worker_id;

        loop {
            // Try to get work from own queue first
            let work_item = match work_queues[my_queue_id].lock() {
                Ok(mut my_queue) => my_queue.pop(),
                _ => None,
            };

            let work_item = if let Some(item) = work_item {
                item
            } else if config.enable_work_stealing {
                // Try to steal work from other queues
                let mut stolen_item = None;
                for (queue_id, queue) in work_queues.iter().enumerate() {
                    if queue_id != my_queue_id {
                        if let Ok(mut other_queue) = queue.lock() {
                            if let Some(item) = other_queue.pop() {
                                stolen_item = Some(item);
                                break;
                            }
                        }
                    }
                }
                if let Some(item) = stolen_item {
                    item
                } else {
                    break; // No more work available
                }
            } else {
                break; // No work stealing, exit
            };

            // Process work item
            let start_time = Instant::now();

            // Check cache first
            let result =
                if let Some(cached_result) = cache.get(&work_item.constraint, &work_item.context) {
                    cached_result
                } else {
                    // In practice, this would evaluate the constraint against a thread-safe store
                    // For now, we'll create a placeholder result
                    ConstraintEvaluationResult::satisfied() // Simplified
                };

            let execution_time = start_time.elapsed();

            // Cache the result if not from cache
            cache.put(
                &work_item.constraint,
                &work_item.context,
                result.clone(),
                execution_time,
            );

            worker_results.push(ValidationWorkResult {
                work_item,
                result,
                execution_time,
                worker_id,
            });
        }

        worker_results
    }

    /// Calculate speedup ratio compared to sequential execution
    fn calculate_speedup_ratio(
        &self,
        parallel_time: Duration,
        results: &[ValidationWorkResult],
    ) -> f64 {
        let total_sequential_time: Duration = results.iter().map(|r| r.execution_time).sum();

        if parallel_time.as_nanos() > 0 {
            total_sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64
        } else {
            1.0
        }
    }

    /// Update pool statistics
    fn update_pool_statistics(
        &self,
        total_time: Duration,
        results: &[ValidationWorkResult],
    ) -> Result<()> {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_items_processed = results.len();
            stats.total_execution_time = total_time;

            // Calculate per-worker statistics
            let mut worker_item_counts: HashMap<usize, usize> = HashMap::new();
            for result in results {
                *worker_item_counts.entry(result.worker_id).or_insert(0) += 1;
            }

            // Calculate load balance score (lower variance = better balance)
            if !worker_item_counts.is_empty() {
                let mean = results.len() as f64 / worker_item_counts.len() as f64;
                let variance: f64 = worker_item_counts
                    .values()
                    .map(|&count| (count as f64 - mean).powi(2))
                    .sum::<f64>()
                    / worker_item_counts.len() as f64;

                stats.load_balance_score = 1.0 / (1.0 + variance.sqrt());
            }

            // Update cache statistics
            let cache_stats = self.cache.stats();
            stats.overall_cache_hit_rate = cache_stats.hit_rate();
        }

        Ok(())
    }

    /// Get current worker pool statistics
    pub fn get_stats(&self) -> WorkerPoolStats {
        self.stats.read().unwrap().clone()
    }

    /// Get number of active workers
    pub fn get_active_worker_count(&self) -> usize {
        self.active_workers.load(Ordering::SeqCst)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ParallelValidationConfig) {
        self.config = config;
    }
}

/// Result of parallel validation
#[derive(Debug, Clone)]
pub struct ParallelValidationResult {
    /// The validation report with all violations
    pub validation_report: ValidationReport,
    /// Total number of work items
    pub total_work_items: usize,
    /// Number of items actually processed
    pub processed_items: usize,
    /// Total parallel execution time
    pub parallel_execution_time: Duration,
    /// Speedup ratio compared to sequential
    pub speedup_ratio: f64,
    /// Parallel efficiency (speedup / number of threads)
    pub efficiency_score: f64,
}

impl ParallelValidationResult {
    /// Check if parallel processing was effective
    pub fn is_effective(&self) -> bool {
        self.speedup_ratio > 1.5 && self.efficiency_score > 0.5
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> String {
        format!(
            "Processed {} items with {:.2}x speedup, {:.1}% efficiency",
            self.processed_items,
            self.speedup_ratio,
            self.efficiency_score * 100.0
        )
    }
}
