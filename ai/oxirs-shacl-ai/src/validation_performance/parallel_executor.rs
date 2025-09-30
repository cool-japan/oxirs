//! Parallel validation execution for SHACL validation
//!
//! This module provides parallel execution capabilities for SHACL validation,
//! including task management, thread pool coordination, and result aggregation.

use crate::{ShaclAiError, Shape};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

use super::cache::ValidationCache;
use super::config::{PerformanceConfig, TaskPriority};
use super::resource_monitor::ResourceMonitor;
use super::types::{ValidationResult, ValidationTask, ValidationViolation, ViolationSeverity};

/// Parallel validation executor
#[derive(Debug)]
pub struct ParallelValidationExecutor {
    config: PerformanceConfig,
    thread_pool: Arc<threadpool::ThreadPool>,
    task_queue: Arc<Mutex<VecDeque<ValidationTask>>>,
    result_cache: Arc<Mutex<ValidationCache>>,
    resource_monitor: ResourceMonitor,
}

impl Clone for ParallelValidationExecutor {
    fn clone(&self) -> Self {
        // Create a new instance with the same config
        Self::new(self.config.clone())
    }
}

/// Performance statistics for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelPerformanceStats {
    pub thread_pool_size: usize,
    pub active_tasks: usize,
    pub queued_tasks: usize,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

impl ParallelValidationExecutor {
    pub fn new(config: PerformanceConfig) -> Self {
        let thread_pool = Arc::new(threadpool::ThreadPool::new(config.worker_threads));

        Self {
            config: config.clone(),
            thread_pool,
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            result_cache: Arc::new(Mutex::new(ValidationCache::new(config.cache_size_limit))),
            resource_monitor: ResourceMonitor::new(),
        }
    }

    /// Submit validation task for parallel execution
    pub fn submit_task(&self, task: ValidationTask) -> Result<(), ShaclAiError> {
        let mut queue = self
            .task_queue
            .lock()
            .map_err(|e| ShaclAiError::Optimization(format!("Failed to lock task queue: {e}")))?;

        // Insert task based on priority
        let insert_pos = queue
            .iter()
            .position(|t| t.priority < task.priority)
            .unwrap_or(queue.len());

        queue.insert(insert_pos, task);

        Ok(())
    }

    /// Execute validation tasks in parallel
    pub fn execute_parallel_validation(
        &self,
        shapes: &[Shape],
        batch_size: Option<usize>,
    ) -> Result<Vec<ValidationResult>, ShaclAiError> {
        let batch_size = batch_size.unwrap_or(self.config.batch_size);
        let mut results = Vec::new();

        // Create tasks from shapes
        let tasks = self.create_validation_tasks(shapes, batch_size)?;

        // Submit tasks to thread pool
        let (tx, rx) = std::sync::mpsc::channel();

        for task in tasks {
            let tx_clone = tx.clone();
            let cache_clone = Arc::clone(&self.result_cache);
            let config_clone = self.config.clone();

            self.thread_pool.execute(move || {
                let result = Self::execute_validation_task(&task, &cache_clone, &config_clone);
                tx_clone.send(result).unwrap_or_else(|e| {
                    eprintln!("Failed to send validation result: {e}");
                });
            });
        }

        // Collect results
        drop(tx); // Close sender
        for result in rx {
            results.push(result?);
        }

        Ok(results)
    }

    /// Create validation tasks from shapes
    fn create_validation_tasks(
        &self,
        shapes: &[Shape],
        batch_size: usize,
    ) -> Result<Vec<ValidationTask>, ShaclAiError> {
        let mut tasks = Vec::new();

        for shape in shapes {
            // Extract constraint IDs from shape
            let constraint_ids: Vec<String> = shape
                .property_constraints
                .iter()
                .map(|c| c.path.clone())
                .collect();

            // Create data batches (simplified)
            let data_batches = vec![vec!["sample_data".to_string()]]; // Simplified

            for data_batch in data_batches {
                let task = ValidationTask {
                    task_id: Uuid::new_v4(),
                    constraint_id: format!("{}_{}", shape.id, constraint_ids.join("_")),
                    data_subset: data_batch,
                    priority: TaskPriority::Normal,
                    estimated_duration: Duration::from_millis(100),
                };
                tasks.push(task);
            }
        }

        Ok(tasks)
    }

    /// Execute a single validation task
    fn execute_validation_task(
        task: &ValidationTask,
        cache: &Arc<Mutex<ValidationCache>>,
        config: &PerformanceConfig,
    ) -> Result<ValidationResult, ShaclAiError> {
        let start_time = Instant::now();

        // Check cache first
        if config.enable_caching {
            let cache_key = format!("{}:{:?}", task.constraint_id, task.data_subset);
            if let Ok(mut cache_guard) = cache.lock() {
                if let Some(cached) = cache_guard.get(&cache_key) {
                    return Ok(cached.clone());
                }
            }
        }

        // Simulate validation execution
        let violations = vec![ValidationViolation {
            violation_id: Uuid::new_v4(),
            constraint_id: task.constraint_id.clone(),
            severity: ViolationSeverity::Warning,
            message: "Example violation".to_string(),
            focus_node: "example_node".to_string(),
            result_path: None,
            value: None,
        }];

        let execution_time = start_time.elapsed();
        let result = ValidationResult {
            is_valid: violations.is_empty(),
            violations,
            execution_time,
            memory_usage_mb: 10.0, // Simulated memory usage
            constraint_results: std::collections::HashMap::new(),
        };

        // Cache result
        if config.enable_caching {
            let cache_key = format!("{}:{:?}", task.constraint_id, task.data_subset);
            if let Ok(mut cache_guard) = cache.lock() {
                let cached_result = super::types::CachedValidationResult {
                    key: cache_key.clone(),
                    result: result.clone(),
                    created_at: Instant::now(),
                    ttl: Duration::from_secs(config.cache_ttl_seconds),
                };
                cache_guard.put(cache_key, cached_result);
            }
        }

        Ok(result)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> ParallelPerformanceStats {
        ParallelPerformanceStats {
            thread_pool_size: self.config.worker_threads,
            active_tasks: self.thread_pool.active_count(),
            queued_tasks: self.thread_pool.queued_count(),
            cache_hit_rate: self.get_cache_hit_rate(),
            memory_usage_mb: self.resource_monitor.get_memory_usage_mb(),
            cpu_usage_percent: self.resource_monitor.get_cpu_usage_percent(),
        }
    }

    /// Get cache hit rate
    fn get_cache_hit_rate(&self) -> f64 {
        match self.result_cache.lock() {
            Ok(cache) => cache.get_hit_rate(),
            _ => 0.0,
        }
    }

    /// Execute tasks with priority-based scheduling
    pub fn execute_prioritized_tasks(&self) -> Result<Vec<ValidationResult>, ShaclAiError> {
        let mut results = Vec::new();
        let (tx, rx) = std::sync::mpsc::channel();

        // Process tasks from queue based on priority
        loop {
            let task = {
                let mut queue = self.task_queue.lock().map_err(|e| {
                    ShaclAiError::Optimization(format!("Failed to lock task queue: {e}"))
                })?;
                queue.pop_front()
            };

            match task {
                Some(task) => {
                    let tx_clone = tx.clone();
                    let cache_clone = Arc::clone(&self.result_cache);
                    let config_clone = self.config.clone();

                    self.thread_pool.execute(move || {
                        let result =
                            Self::execute_validation_task(&task, &cache_clone, &config_clone);
                        tx_clone.send(result).unwrap_or_else(|e| {
                            eprintln!("Failed to send validation result: {e}");
                        });
                    });
                }
                None => break, // No more tasks
            }
        }

        // Collect results
        drop(tx);
        for result in rx {
            results.push(result?);
        }

        Ok(results)
    }

    /// Check if executor is idle
    pub fn is_idle(&self) -> bool {
        self.thread_pool.active_count() == 0 && self.thread_pool.queued_count() == 0
    }

    /// Wait for all tasks to complete
    pub fn wait_for_completion(&self) {
        while !self.is_idle() {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Shutdown the executor gracefully
    pub fn shutdown(&self) {
        self.wait_for_completion();
        // ThreadPool doesn't have an explicit shutdown, it will be dropped
    }

    /// Get queue length
    pub fn get_queue_length(&self) -> Result<usize, ShaclAiError> {
        let queue = self
            .task_queue
            .lock()
            .map_err(|e| ShaclAiError::Optimization(format!("Failed to lock task queue: {e}")))?;
        Ok(queue.len())
    }

    /// Clear the task queue
    pub fn clear_queue(&self) -> Result<(), ShaclAiError> {
        let mut queue = self
            .task_queue
            .lock()
            .map_err(|e| ShaclAiError::Optimization(format!("Failed to lock task queue: {e}")))?;
        queue.clear();
        Ok(())
    }

    /// Start resource monitoring
    pub fn start_monitoring(&self) {
        self.resource_monitor.start_monitoring();
    }
}
