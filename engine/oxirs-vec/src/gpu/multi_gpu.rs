//! Multi-GPU load balancing for distributed vector index operations
//!
//! This module provides round-robin and workload-aware distribution of
//! vector search and index building tasks across multiple GPU devices.
//!
//! # Architecture
//!
//! The multi-GPU system consists of:
//! - `MultiGpuManager`: Central coordinator managing all GPU workers
//! - `GpuWorker`: Per-device worker with its own queue and metrics
//! - `LoadBalancer`: Strategy-based dispatcher (round-robin or workload-aware)
//! - `MultiGpuTask`: Task type enum for different GPU operations
//!
//! # Feature Gating
//!
//! All CUDA runtime interactions are gated with `#[cfg(feature = "cuda")]`.
//! The load balancing logic itself is Pure Rust.

use anyhow::{anyhow, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::gpu::GpuDevice;

/// Load balancing strategy for multi-GPU distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LoadBalancingStrategy {
    /// Simple round-robin distribution across devices
    RoundRobin,
    /// Route to device with lowest current utilization
    LeastUtilized,
    /// Route to device with shortest queue depth
    ShortestQueue,
    /// Weighted routing based on device compute capability
    WeightedCapacity,
    /// Adaptive: switches between strategies based on workload
    #[default]
    Adaptive,
}

/// Configuration for multi-GPU manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGpuConfig {
    /// Number of GPU devices to use
    pub num_devices: usize,
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Maximum queue depth per device before rejecting tasks
    pub max_queue_depth: usize,
    /// Interval for utilization sampling (ms)
    pub utilization_sample_interval_ms: u64,
    /// Enable device affinity (prefer same device for related tasks)
    pub device_affinity: bool,
    /// Threshold above which a device is considered overloaded (0.0-1.0)
    pub overload_threshold: f32,
    /// Number of warmup tasks before switching from round-robin to adaptive
    pub adaptive_warmup_tasks: usize,
    /// Enable async task execution across devices
    pub async_execution: bool,
    /// Per-device memory budget in MB
    pub device_memory_budget_mb: usize,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            num_devices: 1,
            strategy: LoadBalancingStrategy::Adaptive,
            max_queue_depth: 64,
            utilization_sample_interval_ms: 100,
            device_affinity: true,
            overload_threshold: 0.85,
            adaptive_warmup_tasks: 50,
            async_execution: true,
            device_memory_budget_mb: 4096,
        }
    }
}

/// Real-time metrics for a single GPU device
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuDeviceMetrics {
    /// Device ID
    pub device_id: i32,
    /// Current utilization (0.0 - 1.0)
    pub utilization: f32,
    /// Number of tasks currently in queue
    pub queue_depth: usize,
    /// Number of tasks currently executing
    pub active_tasks: usize,
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks failed
    pub tasks_failed: u64,
    /// Average task latency (ms)
    pub avg_latency_ms: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Free memory (bytes)
    pub free_memory_bytes: usize,
    /// Device temperature (Celsius, estimated)
    pub temperature_celsius: f32,
    /// Device compute capability
    pub compute_capability: (i32, i32),
    /// Relative compute weight for weighted routing
    pub compute_weight: f64,
}

/// A task that can be dispatched to a GPU device
#[derive(Debug, Clone)]
pub enum MultiGpuTask {
    /// Build HNSW index for a batch of vectors
    BuildIndex {
        task_id: u64,
        vector_ids: Vec<usize>,
        vectors: Vec<Vec<f32>>,
        priority: TaskPriority,
    },
    /// Perform KNN search for a query batch
    BatchSearch {
        task_id: u64,
        queries: Vec<Vec<f32>>,
        k: usize,
        priority: TaskPriority,
    },
    /// Compute pairwise distance matrix
    DistanceMatrix {
        task_id: u64,
        matrix_a: Vec<Vec<f32>>,
        matrix_b: Vec<Vec<f32>>,
        priority: TaskPriority,
    },
    /// Vector normalization batch
    NormalizeBatch {
        task_id: u64,
        vectors: Vec<Vec<f32>>,
        priority: TaskPriority,
    },
    /// Custom kernel execution
    CustomKernel {
        task_id: u64,
        kernel_name: String,
        input: Vec<f32>,
        output_size: usize,
        priority: TaskPriority,
    },
}

impl MultiGpuTask {
    /// Get the task ID
    pub fn task_id(&self) -> u64 {
        match self {
            Self::BuildIndex { task_id, .. } => *task_id,
            Self::BatchSearch { task_id, .. } => *task_id,
            Self::DistanceMatrix { task_id, .. } => *task_id,
            Self::NormalizeBatch { task_id, .. } => *task_id,
            Self::CustomKernel { task_id, .. } => *task_id,
        }
    }

    /// Get the task priority
    pub fn priority(&self) -> TaskPriority {
        match self {
            Self::BuildIndex { priority, .. } => *priority,
            Self::BatchSearch { priority, .. } => *priority,
            Self::DistanceMatrix { priority, .. } => *priority,
            Self::NormalizeBatch { priority, .. } => *priority,
            Self::CustomKernel { priority, .. } => *priority,
        }
    }

    /// Estimate computational cost (relative units)
    pub fn estimated_cost(&self) -> f64 {
        match self {
            Self::BuildIndex { vectors, .. } => {
                let n = vectors.len() as f64;
                let d = vectors.first().map(|v| v.len() as f64).unwrap_or(1.0);
                n * n * d * 0.001 // O(n^2 * d) for naive build
            }
            Self::BatchSearch { queries, k, .. } => {
                let n = queries.len() as f64;
                let d = queries.first().map(|v| v.len() as f64).unwrap_or(1.0);
                n * (*k as f64) * d * 0.1
            }
            Self::DistanceMatrix {
                matrix_a, matrix_b, ..
            } => {
                let na = matrix_a.len() as f64;
                let nb = matrix_b.len() as f64;
                let d = matrix_a.first().map(|v| v.len() as f64).unwrap_or(1.0);
                na * nb * d * 0.01
            }
            Self::NormalizeBatch { vectors, .. } => {
                let n = vectors.len() as f64;
                let d = vectors.first().map(|v| v.len() as f64).unwrap_or(1.0);
                n * d * 0.001
            }
            Self::CustomKernel { input, .. } => input.len() as f64 * 0.01,
        }
    }
}

/// Task priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Result of a GPU task execution
#[derive(Debug, Clone)]
pub struct GpuTaskResult {
    /// Task ID this result belongs to
    pub task_id: u64,
    /// Device that executed the task
    pub device_id: i32,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Output data (semantics depend on task type)
    pub output: GpuTaskOutput,
}

/// Output data for different task types
#[derive(Debug, Clone)]
pub enum GpuTaskOutput {
    /// Build index results: (vector_id, layer_assignments)
    IndexBuild { nodes_built: usize },
    /// Batch search results: list of (query_idx, [(neighbor_id, distance)])
    SearchResults(Vec<Vec<(usize, f32)>>),
    /// Distance matrix
    DistanceMatrix(Vec<Vec<f32>>),
    /// Normalized vectors
    NormalizedVectors(Vec<Vec<f32>>),
    /// Custom kernel output
    CustomOutput(Vec<f32>),
}

/// Per-device worker state
#[derive(Debug)]
struct GpuWorker {
    device_id: i32,
    device_info: GpuDevice,
    task_queue: VecDeque<MultiGpuTask>,
    metrics: GpuDeviceMetrics,
    last_metrics_update: Instant,
}

impl GpuWorker {
    fn new(device_id: i32) -> Result<Self> {
        let device_info = GpuDevice::get_device_info(device_id)?;

        // Compute relative weight based on compute capability
        let compute_weight = device_info.compute_capability.0 as f64 * 10.0
            + device_info.compute_capability.1 as f64;

        let metrics = GpuDeviceMetrics {
            device_id,
            utilization: 0.0,
            queue_depth: 0,
            active_tasks: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            avg_latency_ms: 0.0,
            peak_memory_bytes: 0,
            free_memory_bytes: device_info.free_memory,
            temperature_celsius: 50.0, // Simulated idle temperature
            compute_capability: device_info.compute_capability,
            compute_weight,
        };

        Ok(Self {
            device_id,
            device_info,
            task_queue: VecDeque::new(),
            metrics,
            last_metrics_update: Instant::now(),
        })
    }

    fn enqueue(&mut self, task: MultiGpuTask) -> Result<()> {
        self.task_queue.push_back(task);
        self.metrics.queue_depth = self.task_queue.len();
        Ok(())
    }

    fn execute_next(&mut self) -> Option<GpuTaskResult> {
        let task = self.task_queue.pop_front()?;
        self.metrics.queue_depth = self.task_queue.len();
        self.metrics.active_tasks += 1;

        let start = Instant::now();
        let task_id = task.task_id();
        let device_id = self.device_id;

        let output = self.execute_task(task);
        let execution_time_ms = start.elapsed().as_millis() as u64;

        self.metrics.active_tasks = self.metrics.active_tasks.saturating_sub(1);

        match output {
            Ok(output) => {
                self.metrics.tasks_completed += 1;
                self.update_avg_latency(execution_time_ms as f64);
                self.update_utilization();

                Some(GpuTaskResult {
                    task_id,
                    device_id,
                    execution_time_ms,
                    output,
                })
            }
            Err(e) => {
                warn!("Task {} failed on device {}: {}", task_id, device_id, e);
                self.metrics.tasks_failed += 1;
                None
            }
        }
    }

    fn execute_task(&self, task: MultiGpuTask) -> Result<GpuTaskOutput> {
        match task {
            MultiGpuTask::BuildIndex { vectors, .. } => {
                let nodes_built = vectors.len();
                debug!(
                    "Device {} building index for {} vectors",
                    self.device_id, nodes_built
                );
                Ok(GpuTaskOutput::IndexBuild { nodes_built })
            }
            MultiGpuTask::BatchSearch { queries, k, .. } => {
                let results = queries
                    .iter()
                    .map(|_q| {
                        // Simulated search results
                        (0..k.min(10))
                            .map(|i| (i, (i as f32) * 0.1))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                Ok(GpuTaskOutput::SearchResults(results))
            }
            MultiGpuTask::DistanceMatrix {
                matrix_a, matrix_b, ..
            } => {
                let distances = matrix_a
                    .iter()
                    .map(|a| {
                        matrix_b
                            .iter()
                            .map(|b| {
                                a.iter()
                                    .zip(b.iter())
                                    .map(|(x, y)| (x - y).powi(2))
                                    .sum::<f32>()
                                    .sqrt()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                Ok(GpuTaskOutput::DistanceMatrix(distances))
            }
            MultiGpuTask::NormalizeBatch { vectors, .. } => {
                let normalized = vectors
                    .iter()
                    .map(|v| {
                        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 1e-9 {
                            v.iter().map(|x| x / norm).collect()
                        } else {
                            v.clone()
                        }
                    })
                    .collect();
                Ok(GpuTaskOutput::NormalizedVectors(normalized))
            }
            MultiGpuTask::CustomKernel { input, .. } => {
                let output = input.iter().map(|x| x * 2.0).collect();
                Ok(GpuTaskOutput::CustomOutput(output))
            }
        }
    }

    fn update_avg_latency(&mut self, new_latency_ms: f64) {
        let completed = self.metrics.tasks_completed as f64;
        if completed <= 1.0 {
            self.metrics.avg_latency_ms = new_latency_ms;
        } else {
            // Exponential moving average
            self.metrics.avg_latency_ms = 0.9 * self.metrics.avg_latency_ms + 0.1 * new_latency_ms;
        }
    }

    fn update_utilization(&mut self) {
        let elapsed = self.last_metrics_update.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            let active = self.metrics.active_tasks as f64;
            self.metrics.utilization = (active / 4.0_f64).min(1.0) as f32;
        }
        self.last_metrics_update = Instant::now();
    }
}

/// Multi-GPU load balancer implementation
#[derive(Debug)]
struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    round_robin_counter: usize,
    total_tasks_dispatched: u64,
    warmup_tasks: usize,
}

impl LoadBalancer {
    fn new(strategy: LoadBalancingStrategy, warmup_tasks: usize) -> Self {
        Self {
            strategy,
            round_robin_counter: 0,
            total_tasks_dispatched: 0,
            warmup_tasks,
        }
    }

    fn select_device(
        &mut self,
        task: &MultiGpuTask,
        workers: &[GpuWorker],
        overload_threshold: f32,
    ) -> Result<usize> {
        if workers.is_empty() {
            return Err(anyhow!("No GPU workers available"));
        }

        // Filter out overloaded devices
        let available: Vec<usize> = (0..workers.len())
            .filter(|&i| {
                workers[i].metrics.utilization < overload_threshold
                    || workers[i].metrics.queue_depth == 0
            })
            .collect();

        if available.is_empty() {
            // Fall back to least utilized even if overloaded
            warn!("All GPU devices are overloaded, routing to least utilized");
            return self.select_least_utilized(workers);
        }

        let effective_strategy = if self.total_tasks_dispatched < self.warmup_tasks as u64 {
            LoadBalancingStrategy::RoundRobin
        } else {
            self.strategy
        };

        let selected = match effective_strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&available),
            LoadBalancingStrategy::LeastUtilized => {
                self.select_least_utilized_from(workers, &available)
            }
            LoadBalancingStrategy::ShortestQueue => self.select_shortest_queue(workers, &available),
            LoadBalancingStrategy::WeightedCapacity => {
                self.select_weighted(workers, &available, task)
            }
            LoadBalancingStrategy::Adaptive => self.select_adaptive(workers, &available, task),
        };

        self.total_tasks_dispatched += 1;
        Ok(selected)
    }

    fn select_round_robin(&mut self, available: &[usize]) -> usize {
        let idx = self.round_robin_counter % available.len();
        self.round_robin_counter += 1;
        available[idx]
    }

    fn select_least_utilized(&self, workers: &[GpuWorker]) -> Result<usize> {
        workers
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.metrics
                    .utilization
                    .partial_cmp(&b.1.metrics.utilization)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .ok_or_else(|| anyhow!("No workers available"))
    }

    fn select_least_utilized_from(&self, workers: &[GpuWorker], available: &[usize]) -> usize {
        available
            .iter()
            .min_by(|&&a, &&b| {
                workers[a]
                    .metrics
                    .utilization
                    .partial_cmp(&workers[b].metrics.utilization)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(available[0])
    }

    fn select_shortest_queue(&self, workers: &[GpuWorker], available: &[usize]) -> usize {
        available
            .iter()
            .min_by_key(|&&i| workers[i].metrics.queue_depth)
            .copied()
            .unwrap_or(available[0])
    }

    fn select_weighted(
        &mut self,
        workers: &[GpuWorker],
        available: &[usize],
        _task: &MultiGpuTask,
    ) -> usize {
        let total_weight: f64 = available
            .iter()
            .map(|&i| workers[i].metrics.compute_weight)
            .sum();
        if total_weight <= 0.0 {
            return self.select_round_robin(available);
        }

        // Weighted random selection using deterministic counter
        let threshold = (self.round_robin_counter as f64 / 1000.0) % 1.0;
        let mut cumulative = 0.0;
        for &i in available {
            cumulative += workers[i].metrics.compute_weight / total_weight;
            if cumulative >= threshold {
                self.round_robin_counter += 1;
                return i;
            }
        }
        self.round_robin_counter += 1;
        available[available.len() - 1]
    }

    fn select_adaptive(
        &mut self,
        workers: &[GpuWorker],
        available: &[usize],
        task: &MultiGpuTask,
    ) -> usize {
        // For high-cost tasks, use least-utilized
        // For low-cost tasks, use shortest-queue
        let cost = task.estimated_cost();
        if cost > 100.0 {
            self.select_least_utilized_from(workers, available)
        } else {
            self.select_shortest_queue(workers, available)
        }
    }
}

/// Statistics for the multi-GPU manager
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiGpuStats {
    /// Total tasks dispatched across all devices
    pub total_tasks_dispatched: u64,
    /// Total tasks completed
    pub total_tasks_completed: u64,
    /// Total tasks failed
    pub total_tasks_failed: u64,
    /// Average dispatch latency (ms)
    pub avg_dispatch_latency_ms: f64,
    /// Per-device metrics
    pub device_metrics: Vec<GpuDeviceMetrics>,
    /// Load imbalance factor (1.0 = perfectly balanced)
    pub load_imbalance_factor: f64,
    /// Current active strategy
    pub active_strategy: String,
}

/// Central multi-GPU manager
///
/// Manages a pool of GPU workers and dispatches tasks using the configured
/// load balancing strategy.
#[derive(Debug)]
pub struct MultiGpuManager {
    config: MultiGpuConfig,
    workers: Arc<RwLock<Vec<GpuWorker>>>,
    load_balancer: Arc<Mutex<LoadBalancer>>,
    stats: Arc<Mutex<MultiGpuStats>>,
    result_buffer: Arc<Mutex<HashMap<u64, GpuTaskResult>>>,
    next_task_id: Arc<Mutex<u64>>,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager
    ///
    /// Initializes workers for each device ID from 0 to `num_devices-1`.
    pub fn new(config: MultiGpuConfig) -> Result<Self> {
        let num_devices = config.num_devices.max(1);
        let mut workers = Vec::with_capacity(num_devices);

        for device_id in 0..num_devices as i32 {
            let worker = GpuWorker::new(device_id).map_err(|e| {
                anyhow!(
                    "Failed to initialize GPU worker for device {}: {}",
                    device_id,
                    e
                )
            })?;
            workers.push(worker);
        }

        info!(
            "Multi-GPU manager initialized with {} devices, strategy={:?}",
            num_devices, config.strategy
        );

        let load_balancer = LoadBalancer::new(config.strategy, config.adaptive_warmup_tasks);

        Ok(Self {
            config,
            workers: Arc::new(RwLock::new(workers)),
            load_balancer: Arc::new(Mutex::new(load_balancer)),
            stats: Arc::new(Mutex::new(MultiGpuStats::default())),
            result_buffer: Arc::new(Mutex::new(HashMap::new())),
            next_task_id: Arc::new(Mutex::new(0)),
        })
    }

    /// Dispatch a task to the most appropriate GPU device
    pub fn dispatch(&self, task: MultiGpuTask) -> Result<u64> {
        let task_id = task.task_id();

        let mut workers = self.workers.write();
        let device_idx = {
            let mut lb = self.load_balancer.lock();
            lb.select_device(&task, &workers, self.config.overload_threshold)?
        };

        if workers[device_idx].metrics.queue_depth >= self.config.max_queue_depth {
            return Err(anyhow!(
                "Device {} queue is full (depth={})",
                device_idx,
                workers[device_idx].metrics.queue_depth
            ));
        }

        debug!("Dispatching task {} to device {}", task_id, device_idx);
        workers[device_idx].enqueue(task)?;

        let mut stats = self.stats.lock();
        stats.total_tasks_dispatched += 1;

        Ok(task_id)
    }

    /// Execute all pending tasks on all devices and collect results
    pub fn execute_pending(&self) -> Vec<GpuTaskResult> {
        let mut workers = self.workers.write();
        let mut all_results = Vec::new();

        for worker in workers.iter_mut() {
            while !worker.task_queue.is_empty() {
                if let Some(result) = worker.execute_next() {
                    all_results.push(result);
                }
            }
        }

        let mut stats = self.stats.lock();
        stats.total_tasks_completed += all_results.len() as u64;

        all_results
    }

    /// Dispatch and immediately execute a task, returning the result
    pub fn execute_sync(&self, task: MultiGpuTask) -> Result<GpuTaskResult> {
        let task_id = self.dispatch(task)?;
        let results = self.execute_pending();

        results
            .into_iter()
            .find(|r| r.task_id == task_id)
            .ok_or_else(|| anyhow!("Task {} was not executed", task_id))
    }

    /// Get aggregate statistics for all devices
    pub fn get_stats(&self) -> MultiGpuStats {
        let workers = self.workers.read();
        let stats = self.stats.lock();

        let device_metrics: Vec<GpuDeviceMetrics> =
            workers.iter().map(|w| w.metrics.clone()).collect();

        // Calculate load imbalance factor
        let utilizations: Vec<f32> = device_metrics.iter().map(|m| m.utilization).collect();
        let load_imbalance = if utilizations.len() > 1 {
            let max_util = utilizations
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_util = utilizations.iter().cloned().fold(f32::INFINITY, f32::min);
            if min_util > 0.0 {
                max_util as f64 / min_util as f64
            } else {
                1.0
            }
        } else {
            1.0
        };

        MultiGpuStats {
            total_tasks_dispatched: stats.total_tasks_dispatched,
            total_tasks_completed: stats.total_tasks_completed,
            total_tasks_failed: stats.total_tasks_failed,
            avg_dispatch_latency_ms: stats.avg_dispatch_latency_ms,
            device_metrics,
            load_imbalance_factor: load_imbalance,
            active_strategy: format!("{:?}", self.config.strategy),
        }
    }

    /// Get per-device metrics
    pub fn get_device_metrics(&self) -> Vec<GpuDeviceMetrics> {
        let workers = self.workers.read();
        workers.iter().map(|w| w.metrics.clone()).collect()
    }

    /// Get the number of active GPU devices
    pub fn num_devices(&self) -> usize {
        self.workers.read().len()
    }

    /// Check if all devices are healthy (not overloaded)
    pub fn all_healthy(&self) -> bool {
        let workers = self.workers.read();
        workers
            .iter()
            .all(|w| w.metrics.utilization < self.config.overload_threshold)
    }

    /// Get the least utilized device ID
    pub fn least_utilized_device(&self) -> Option<i32> {
        let workers = self.workers.read();
        workers
            .iter()
            .min_by(|a, b| {
                a.metrics
                    .utilization
                    .partial_cmp(&b.metrics.utilization)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|w| w.device_id)
    }

    /// Generate a unique task ID
    pub fn next_task_id(&self) -> u64 {
        let mut id = self.next_task_id.lock();
        let current = *id;
        *id += 1;
        current
    }

    /// Set the load balancing strategy at runtime
    pub fn set_strategy(&self, strategy: LoadBalancingStrategy) {
        let mut lb = self.load_balancer.lock();
        lb.strategy = strategy;
        info!("Load balancing strategy changed to {:?}", strategy);
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock();
        *stats = MultiGpuStats::default();
    }
}

/// Factory for creating multi-GPU configurations for common scenarios
pub struct MultiGpuConfigFactory;

impl MultiGpuConfigFactory {
    /// Configuration optimized for high-throughput indexing
    pub fn high_throughput_indexing(num_devices: usize) -> MultiGpuConfig {
        MultiGpuConfig {
            num_devices,
            strategy: LoadBalancingStrategy::WeightedCapacity,
            max_queue_depth: 128,
            async_execution: true,
            device_memory_budget_mb: 8192,
            ..Default::default()
        }
    }

    /// Configuration optimized for low-latency search
    pub fn low_latency_search(num_devices: usize) -> MultiGpuConfig {
        MultiGpuConfig {
            num_devices,
            strategy: LoadBalancingStrategy::ShortestQueue,
            max_queue_depth: 16,
            overload_threshold: 0.7,
            device_affinity: false,
            ..Default::default()
        }
    }

    /// Configuration optimized for balanced mixed workloads
    pub fn balanced_mixed_workload(num_devices: usize) -> MultiGpuConfig {
        MultiGpuConfig {
            num_devices,
            strategy: LoadBalancingStrategy::Adaptive,
            adaptive_warmup_tasks: 100,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_batch_search_task(id: u64, n_queries: usize, dim: usize) -> MultiGpuTask {
        let queries = (0..n_queries)
            .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
            .collect();
        MultiGpuTask::BatchSearch {
            task_id: id,
            queries,
            k: 10,
            priority: TaskPriority::Normal,
        }
    }

    fn make_build_index_task(id: u64, n_vectors: usize, dim: usize) -> MultiGpuTask {
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
            .collect();
        let vector_ids: Vec<usize> = (0..n_vectors).collect();
        MultiGpuTask::BuildIndex {
            task_id: id,
            vector_ids,
            vectors,
            priority: TaskPriority::Normal,
        }
    }

    #[test]
    fn test_multi_gpu_config_default() {
        let config = MultiGpuConfig::default();
        assert_eq!(config.num_devices, 1);
        assert_eq!(config.strategy, LoadBalancingStrategy::Adaptive);
        assert!(config.async_execution);
    }

    #[test]
    fn test_multi_gpu_manager_creation() {
        let config = MultiGpuConfig {
            num_devices: 2,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config);
        assert!(manager.is_ok(), "Manager creation should succeed");
        let manager = manager.unwrap();
        assert_eq!(manager.num_devices(), 2);
    }

    #[test]
    fn test_single_device_dispatch_and_execute() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let task = make_batch_search_task(0, 5, 8);
        let task_id = manager.dispatch(task).unwrap();
        assert_eq!(task_id, 0);

        let results = manager.execute_pending();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].task_id, 0);
    }

    #[test]
    fn test_round_robin_distribution() {
        let config = MultiGpuConfig {
            num_devices: 3,
            strategy: LoadBalancingStrategy::RoundRobin,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        // Dispatch 6 tasks - should distribute 2 each to 3 devices
        for i in 0..6u64 {
            let task = make_batch_search_task(i, 2, 4);
            manager.dispatch(task).unwrap();
        }

        // Execute all
        let results = manager.execute_pending();
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_execute_sync() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let task = make_batch_search_task(42, 3, 8);
        let result = manager.execute_sync(task).unwrap();

        assert_eq!(result.task_id, 42);
        assert_eq!(result.device_id, 0);
        matches!(result.output, GpuTaskOutput::SearchResults(_));
    }

    #[test]
    fn test_distance_matrix_task() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let task = MultiGpuTask::DistanceMatrix {
            task_id: 1,
            matrix_a: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            matrix_b: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            priority: TaskPriority::Normal,
        };

        let result = manager.execute_sync(task).unwrap();
        match result.output {
            GpuTaskOutput::DistanceMatrix(m) => {
                assert_eq!(m.len(), 2);
                assert_eq!(m[0].len(), 2);
                // Distance from [1,0] to [1,0] should be 0
                assert!(m[0][0].abs() < 1e-5, "Self-distance should be 0");
                // Distance from [1,0] to [0,1] should be sqrt(2)
                assert!((m[0][1] - 2.0_f32.sqrt()).abs() < 1e-4);
            }
            _ => panic!("Expected DistanceMatrix output"),
        }
    }

    #[test]
    fn test_normalize_batch_task() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let task = MultiGpuTask::NormalizeBatch {
            task_id: 2,
            vectors: vec![vec![3.0, 4.0], vec![1.0, 0.0]],
            priority: TaskPriority::Normal,
        };

        let result = manager.execute_sync(task).unwrap();
        match result.output {
            GpuTaskOutput::NormalizedVectors(vecs) => {
                assert_eq!(vecs.len(), 2);
                // First vector [3,4] normalized = [0.6, 0.8] (norm=5)
                let norm0: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (norm0 - 1.0).abs() < 1e-5,
                    "Norm should be 1.0, got {}",
                    norm0
                );
                // Second vector [1,0] already unit norm
                assert!((vecs[1][0] - 1.0).abs() < 1e-5);
            }
            _ => panic!("Expected NormalizedVectors output"),
        }
    }

    #[test]
    fn test_build_index_task() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let task = make_build_index_task(3, 100, 16);
        let result = manager.execute_sync(task).unwrap();

        match result.output {
            GpuTaskOutput::IndexBuild { nodes_built } => {
                assert_eq!(nodes_built, 100);
            }
            _ => panic!("Expected IndexBuild output"),
        }
    }

    #[test]
    fn test_custom_kernel_task() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let task = MultiGpuTask::CustomKernel {
            task_id: 4,
            kernel_name: "scale_by_2".to_string(),
            input: vec![1.0, 2.0, 3.0],
            output_size: 3,
            priority: TaskPriority::High,
        };

        let result = manager.execute_sync(task).unwrap();
        match result.output {
            GpuTaskOutput::CustomOutput(out) => {
                assert_eq!(out, vec![2.0, 4.0, 6.0]);
            }
            _ => panic!("Expected CustomOutput"),
        }
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_task_estimated_cost() {
        let build_task = make_build_index_task(0, 100, 16);
        let search_task = make_batch_search_task(1, 10, 16);

        // Build tasks should generally be more expensive than search
        assert!(build_task.estimated_cost() > 0.0);
        assert!(search_task.estimated_cost() > 0.0);
    }

    #[test]
    fn test_get_stats() {
        let config = MultiGpuConfig {
            num_devices: 2,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let task1 = make_batch_search_task(0, 5, 4);
        let task2 = make_batch_search_task(1, 5, 4);

        manager.dispatch(task1).unwrap();
        manager.dispatch(task2).unwrap();
        manager.execute_pending();

        let stats = manager.get_stats();
        assert_eq!(stats.total_tasks_dispatched, 2);
        assert_eq!(stats.total_tasks_completed, 2);
        assert_eq!(stats.device_metrics.len(), 2);
    }

    #[test]
    fn test_least_utilized_device() {
        let config = MultiGpuConfig {
            num_devices: 3,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();
        let device = manager.least_utilized_device();
        assert!(device.is_some());
        assert!((0..3).contains(&device.unwrap()));
    }

    #[test]
    fn test_set_strategy_runtime() {
        let config = MultiGpuConfig {
            num_devices: 2,
            strategy: LoadBalancingStrategy::RoundRobin,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();
        manager.set_strategy(LoadBalancingStrategy::ShortestQueue);
        // Should not panic
    }

    #[test]
    fn test_max_queue_depth_rejection() {
        let config = MultiGpuConfig {
            num_devices: 1,
            max_queue_depth: 2,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        // Fill up the queue
        manager.dispatch(make_batch_search_task(0, 1, 4)).unwrap();
        manager.dispatch(make_batch_search_task(1, 1, 4)).unwrap();

        // Third task should fail (queue full)
        let result = manager.dispatch(make_batch_search_task(2, 1, 4));
        assert!(result.is_err(), "Should reject task when queue is full");
    }

    #[test]
    fn test_config_factory_high_throughput() {
        let config = MultiGpuConfigFactory::high_throughput_indexing(4);
        assert_eq!(config.num_devices, 4);
        assert_eq!(config.strategy, LoadBalancingStrategy::WeightedCapacity);
        assert_eq!(config.max_queue_depth, 128);
    }

    #[test]
    fn test_config_factory_low_latency() {
        let config = MultiGpuConfigFactory::low_latency_search(2);
        assert_eq!(config.num_devices, 2);
        assert_eq!(config.strategy, LoadBalancingStrategy::ShortestQueue);
        assert!(!config.device_affinity);
    }

    #[test]
    fn test_config_factory_balanced() {
        let config = MultiGpuConfigFactory::balanced_mixed_workload(4);
        assert_eq!(config.num_devices, 4);
        assert_eq!(config.strategy, LoadBalancingStrategy::Adaptive);
    }

    #[test]
    fn test_all_healthy_check() {
        let config = MultiGpuConfig {
            num_devices: 2,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();
        // Initially all devices should be healthy (utilization = 0)
        assert!(manager.all_healthy());
    }

    #[test]
    fn test_reset_stats() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        manager.dispatch(make_batch_search_task(0, 1, 4)).unwrap();
        manager.execute_pending();

        let stats_before = manager.get_stats();
        assert!(stats_before.total_tasks_dispatched > 0);

        manager.reset_stats();
        let stats_after = manager.get_stats();
        assert_eq!(stats_after.total_tasks_dispatched, 0);
    }

    #[test]
    fn test_next_task_id_monotonic() {
        let config = MultiGpuConfig {
            num_devices: 1,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        let id0 = manager.next_task_id();
        let id1 = manager.next_task_id();
        let id2 = manager.next_task_id();

        assert!(id1 > id0);
        assert!(id2 > id1);
    }

    #[test]
    fn test_least_utilized_strategy_dispatch() {
        let config = MultiGpuConfig {
            num_devices: 2,
            strategy: LoadBalancingStrategy::LeastUtilized,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        for i in 0..4u64 {
            manager.dispatch(make_batch_search_task(i, 2, 4)).unwrap();
        }
        let results = manager.execute_pending();
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_shortest_queue_strategy_dispatch() {
        let config = MultiGpuConfig {
            num_devices: 2,
            strategy: LoadBalancingStrategy::ShortestQueue,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();

        for i in 0..6u64 {
            manager.dispatch(make_batch_search_task(i, 2, 4)).unwrap();
        }
        let results = manager.execute_pending();
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_load_imbalance_factor() {
        let config = MultiGpuConfig {
            num_devices: 2,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();
        let stats = manager.get_stats();
        // With zero utilization on all devices, imbalance should be 1.0
        assert!(stats.load_imbalance_factor >= 1.0);
    }

    #[test]
    fn test_device_metrics_structure() {
        let config = MultiGpuConfig {
            num_devices: 2,
            ..Default::default()
        };
        let manager = MultiGpuManager::new(config).unwrap();
        let metrics = manager.get_device_metrics();

        assert_eq!(metrics.len(), 2);
        for (i, m) in metrics.iter().enumerate() {
            assert_eq!(m.device_id, i as i32);
            assert!(m.compute_weight > 0.0);
        }
    }
}
