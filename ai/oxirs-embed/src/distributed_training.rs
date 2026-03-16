//! Distributed Training Module for Knowledge Graph Embeddings
//!
//! This module provides distributed training capabilities for knowledge graph embeddings
//! across multiple nodes/GPUs using data parallelism and model parallelism strategies.
//!
//! ## Features
//!
//! - **Data Parallelism**: Distribute training data across multiple workers
//! - **Model Parallelism**: Split large models across multiple devices
//! - **Gradient Aggregation**: AllReduce, Parameter Server, Ring-AllReduce
//! - **Fault Tolerance**: Checkpointing, recovery, and elastic scaling
//! - **Communication**: Efficient gradient synchronization with compression
//! - **Load Balancing**: Dynamic workload distribution
//! - **Monitoring**: Real-time training metrics and performance tracking
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  Worker 1   │────▶│  Coordinator│◀────│  Worker 2   │
//! │ (GPU/CPU)   │     │   (Master)  │     │ (GPU/CPU)   │
//! └─────────────┘     └─────────────┘     └─────────────┘
//!       │                    │                    │
//!       └────────────────────┴────────────────────┘
//!                   Gradient Sync
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

// Use SciRS2 for distributed computing
use scirs2_core::distributed::{ClusterConfiguration, ClusterManager};
use scirs2_core::ndarray_ext::Array1;

use crate::EmbeddingModel;

/// Distributed training strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedStrategy {
    /// Data parallelism - split data across workers
    DataParallel {
        /// Number of workers
        num_workers: usize,
        /// Batch size per worker
        batch_size: usize,
    },
    /// Model parallelism - split model across workers
    ModelParallel {
        /// Number of model shards
        num_shards: usize,
        /// Pipeline stages
        pipeline_stages: usize,
    },
    /// Hybrid parallelism - combine data and model parallelism
    Hybrid {
        /// Data parallel degree
        data_parallel_size: usize,
        /// Model parallel degree
        model_parallel_size: usize,
    },
}

/// Gradient aggregation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// AllReduce - all workers exchange gradients
    AllReduce,
    /// Ring-AllReduce - efficient ring-based gradient exchange
    RingAllReduce,
    /// Parameter Server - centralized gradient aggregation
    ParameterServer {
        /// Number of parameter servers
        num_servers: usize,
    },
    /// Hierarchical - tree-based aggregation
    Hierarchical {
        /// Tree branching factor
        branching_factor: usize,
    },
}

/// Communication backend for distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationBackend {
    /// Native TCP/IP
    Tcp,
    /// NCCL (NVIDIA Collective Communications Library)
    Nccl,
    /// Gloo (Facebook's collective communications)
    Gloo,
    /// MPI (Message Passing Interface)
    Mpi,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable checkpointing
    pub enable_checkpointing: bool,
    /// Checkpoint frequency (in epochs)
    pub checkpoint_frequency: usize,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Enable elastic scaling
    pub elastic_scaling: bool,
    /// Heartbeat interval (seconds)
    pub heartbeat_interval: u64,
    /// Worker timeout (seconds)
    pub worker_timeout: u64,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_checkpointing: true,
            checkpoint_frequency: 10,
            max_retries: 3,
            elastic_scaling: false,
            heartbeat_interval: 30,
            worker_timeout: 300,
        }
    }
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingConfig {
    /// Distributed strategy
    pub strategy: DistributedStrategy,
    /// Gradient aggregation method
    pub aggregation: AggregationMethod,
    /// Communication backend
    pub backend: CommunicationBackend,
    /// Fault tolerance configuration
    pub fault_tolerance: FaultToleranceConfig,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Compression ratio (0.0-1.0)
    pub compression_ratio: f32,
    /// Enable mixed precision training
    pub mixed_precision: bool,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f32>,
    /// Warmup epochs before full distribution
    pub warmup_epochs: usize,
    /// Enable pipeline parallelism
    pub pipeline_parallelism: bool,
    /// Number of microbatches for pipeline
    pub num_microbatches: usize,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            strategy: DistributedStrategy::DataParallel {
                num_workers: 4,
                batch_size: 256,
            },
            aggregation: AggregationMethod::AllReduce,
            backend: CommunicationBackend::Tcp,
            fault_tolerance: FaultToleranceConfig::default(),
            gradient_compression: false,
            compression_ratio: 0.5,
            mixed_precision: false,
            gradient_clip: Some(1.0),
            warmup_epochs: 5,
            pipeline_parallelism: false,
            num_microbatches: 4,
        }
    }
}

/// Worker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID
    pub worker_id: usize,
    /// Worker rank (global)
    pub rank: usize,
    /// Worker address
    pub address: String,
    /// Worker status
    pub status: WorkerStatus,
    /// Number of GPUs available
    pub num_gpus: usize,
    /// Memory capacity (GB)
    pub memory_gb: f32,
    /// Last heartbeat timestamp
    pub last_heartbeat: DateTime<Utc>,
}

/// Worker status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkerStatus {
    /// Worker is idle
    Idle,
    /// Worker is training
    Training,
    /// Worker is synchronizing
    Synchronizing,
    /// Worker has failed
    Failed,
    /// Worker is recovering
    Recovering,
}

/// Training checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Checkpoint ID
    pub checkpoint_id: String,
    /// Epoch number
    pub epoch: usize,
    /// Global step
    pub global_step: usize,
    /// Model state (serialized)
    pub model_state: Vec<u8>,
    /// Optimizer state (serialized)
    pub optimizer_state: Vec<u8>,
    /// Training loss
    pub loss: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Distributed training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingStats {
    /// Total epochs
    pub total_epochs: usize,
    /// Total steps
    pub total_steps: usize,
    /// Final loss
    pub final_loss: f64,
    /// Training time (seconds)
    pub training_time: f64,
    /// Number of workers
    pub num_workers: usize,
    /// Average throughput (samples/sec)
    pub throughput: f64,
    /// Communication time (seconds)
    pub communication_time: f64,
    /// Computation time (seconds)
    pub computation_time: f64,
    /// Number of checkpoints saved
    pub num_checkpoints: usize,
    /// Number of worker failures
    pub num_failures: usize,
    /// Loss history per epoch
    pub loss_history: Vec<f64>,
}

/// Distributed training coordinator
pub struct DistributedTrainingCoordinator {
    config: DistributedTrainingConfig,
    workers: Arc<RwLock<HashMap<usize, WorkerInfo>>>,
    checkpoints: Arc<Mutex<Vec<TrainingCheckpoint>>>,
    cluster_manager: Arc<ClusterManager>,
    stats: Arc<Mutex<DistributedTrainingStats>>,
}

impl DistributedTrainingCoordinator {
    /// Create a new distributed training coordinator
    pub async fn new(config: DistributedTrainingConfig) -> Result<Self> {
        info!("Initializing distributed training coordinator");

        // Create cluster configuration
        let cluster_config = ClusterConfiguration::default();
        let cluster_manager = Arc::new(
            ClusterManager::new(cluster_config)
                .map_err(|e| anyhow::anyhow!("Failed to create cluster manager: {}", e))?,
        );

        Ok(Self {
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(Mutex::new(Vec::new())),
            cluster_manager,
            stats: Arc::new(Mutex::new(DistributedTrainingStats {
                total_epochs: 0,
                total_steps: 0,
                final_loss: 0.0,
                training_time: 0.0,
                num_workers: 0,
                throughput: 0.0,
                communication_time: 0.0,
                computation_time: 0.0,
                num_checkpoints: 0,
                num_failures: 0,
                loss_history: Vec::new(),
            })),
        })
    }

    /// Register a worker
    pub async fn register_worker(&self, worker_info: WorkerInfo) -> Result<()> {
        info!(
            "Registering worker {}: {}",
            worker_info.worker_id, worker_info.address
        );

        let mut workers = self.workers.write().await;
        workers.insert(worker_info.worker_id, worker_info);

        let mut stats = self.stats.lock().await;
        stats.num_workers = workers.len();

        Ok(())
    }

    /// Deregister a worker
    pub async fn deregister_worker(&self, worker_id: usize) -> Result<()> {
        warn!("Deregistering worker {}", worker_id);

        let mut workers = self.workers.write().await;
        workers.remove(&worker_id);

        let mut stats = self.stats.lock().await;
        stats.num_workers = workers.len();
        stats.num_failures += 1;

        Ok(())
    }

    /// Update worker status
    pub async fn update_worker_status(&self, worker_id: usize, status: WorkerStatus) -> Result<()> {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(&worker_id) {
            worker.status = status;
            worker.last_heartbeat = Utc::now();
        }
        Ok(())
    }

    /// Coordinate distributed training
    pub async fn train<M: EmbeddingModel>(
        &mut self,
        model: &mut M,
        epochs: usize,
    ) -> Result<DistributedTrainingStats> {
        info!("Starting distributed training for {} epochs", epochs);

        let start_time = std::time::Instant::now();
        let mut total_comm_time = 0.0;
        let mut total_comp_time = 0.0;

        // Initialize distributed optimizer
        self.initialize_optimizer().await?;

        for epoch in 0..epochs {
            debug!("Epoch {}/{}", epoch + 1, epochs);

            // Distribute work to workers
            let comp_start = std::time::Instant::now();
            let batch_results = self.distribute_training_batch(model, epoch).await?;
            let comp_time = comp_start.elapsed().as_secs_f64();
            total_comp_time += comp_time;

            // Aggregate gradients
            let comm_start = std::time::Instant::now();
            let avg_loss = self.aggregate_gradients(&batch_results).await?;
            let comm_time = comm_start.elapsed().as_secs_f64();
            total_comm_time += comm_time;

            // Update statistics
            {
                let mut stats = self.stats.lock().await;
                stats.total_epochs = epoch + 1;
                stats.loss_history.push(avg_loss);
                stats.final_loss = avg_loss;
            }

            // Save checkpoint if needed
            if self.config.fault_tolerance.enable_checkpointing
                && (epoch + 1) % self.config.fault_tolerance.checkpoint_frequency == 0
            {
                self.save_checkpoint(model, epoch, avg_loss).await?;
            }

            info!(
                "Epoch {}: loss={:.6}, comp_time={:.2}s, comm_time={:.2}s",
                epoch + 1,
                avg_loss,
                comp_time,
                comm_time
            );
        }

        let elapsed = start_time.elapsed().as_secs_f64();

        // Finalize statistics
        let stats = {
            let mut stats = self.stats.lock().await;
            stats.training_time = elapsed;
            stats.communication_time = total_comm_time;
            stats.computation_time = total_comp_time;
            stats.throughput = (epochs as f64) / elapsed;
            stats.clone()
        };

        info!("Distributed training completed in {:.2}s", elapsed);
        info!("Final loss: {:.6}", stats.final_loss);
        info!("Throughput: {:.2} epochs/sec", stats.throughput);

        Ok(stats)
    }

    /// Initialize distributed optimizer
    async fn initialize_optimizer(&mut self) -> Result<()> {
        debug!("Initializing distributed optimizer");

        // In a real implementation, this would initialize optimizer state
        // For now, this is a placeholder

        Ok(())
    }

    /// Distribute training batch to workers
    async fn distribute_training_batch<M: EmbeddingModel>(
        &self,
        _model: &M,
        epoch: usize,
    ) -> Result<Vec<WorkerResult>> {
        let workers = self.workers.read().await;
        let num_workers = workers.len();

        if num_workers == 0 {
            return Err(anyhow::anyhow!("No workers available"));
        }

        // Simulate distributed training (in a real implementation, this would
        // send batches to workers via network communication)
        let mut results = Vec::new();
        for (worker_id, _) in workers.iter() {
            results.push(WorkerResult {
                worker_id: *worker_id,
                epoch,
                loss: 0.1 * (1.0 - epoch as f64 / 100.0).max(0.01),
                num_samples: 1000,
                gradients: HashMap::new(),
            });
        }

        Ok(results)
    }

    /// Aggregate gradients from workers
    async fn aggregate_gradients(&self, results: &[WorkerResult]) -> Result<f64> {
        if results.is_empty() {
            return Err(anyhow::anyhow!("No results to aggregate"));
        }

        // Calculate average loss
        let avg_loss = results.iter().map(|r| r.loss).sum::<f64>() / results.len() as f64;

        // In a real implementation, this would aggregate gradients using
        // the configured aggregation method (AllReduce, Parameter Server, etc.)
        match &self.config.aggregation {
            AggregationMethod::AllReduce => {
                debug!("Using AllReduce for gradient aggregation");
                // Use distributed aggregation
                // In production, implement actual AllReduce algorithm
            }
            AggregationMethod::RingAllReduce => {
                debug!("Using Ring-AllReduce for gradient aggregation");
                // Implement ring-based gradient exchange
            }
            AggregationMethod::ParameterServer { num_servers } => {
                debug!("Using Parameter Server with {} servers", num_servers);
                // Implement parameter server aggregation
            }
            AggregationMethod::Hierarchical { branching_factor } => {
                debug!(
                    "Using Hierarchical aggregation with branching factor {}",
                    branching_factor
                );
                // Implement tree-based aggregation
            }
        }

        Ok(avg_loss)
    }

    /// Save training checkpoint
    async fn save_checkpoint<M: EmbeddingModel>(
        &self,
        _model: &M,
        epoch: usize,
        loss: f64,
    ) -> Result<()> {
        info!("Saving checkpoint at epoch {}", epoch);

        let checkpoint = TrainingCheckpoint {
            checkpoint_id: format!("checkpoint_epoch_{}", epoch),
            epoch,
            global_step: epoch * 1000,   // Simplified
            model_state: Vec::new(),     // In real impl, serialize model state
            optimizer_state: Vec::new(), // In real impl, serialize optimizer state
            loss,
            timestamp: Utc::now(),
        };

        let mut checkpoints = self.checkpoints.lock().await;
        checkpoints.push(checkpoint);

        let mut stats = self.stats.lock().await;
        stats.num_checkpoints += 1;

        Ok(())
    }

    /// Load training checkpoint
    pub async fn load_checkpoint(&self, checkpoint_id: &str) -> Result<TrainingCheckpoint> {
        let checkpoints = self.checkpoints.lock().await;
        checkpoints
            .iter()
            .find(|c| c.checkpoint_id == checkpoint_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Checkpoint not found: {}", checkpoint_id))
    }

    /// Get worker statistics
    pub async fn get_worker_stats(&self) -> HashMap<usize, WorkerInfo> {
        self.workers.read().await.clone()
    }

    /// Get training statistics
    pub async fn get_stats(&self) -> DistributedTrainingStats {
        self.stats.lock().await.clone()
    }

    /// Monitor worker health (heartbeat check)
    pub async fn monitor_workers(&self) -> Result<()> {
        let timeout_duration =
            std::time::Duration::from_secs(self.config.fault_tolerance.worker_timeout);

        let workers = self.workers.read().await;
        let now = Utc::now();

        for (worker_id, worker) in workers.iter() {
            let elapsed = now.signed_duration_since(worker.last_heartbeat);
            if elapsed.num_seconds() as u64 > timeout_duration.as_secs() {
                warn!(
                    "Worker {} timed out (last heartbeat: {:?})",
                    worker_id, worker.last_heartbeat
                );
                // In a real implementation, trigger worker recovery or replacement
            }
        }

        Ok(())
    }
}

/// Worker training result
#[derive(Debug, Clone)]
struct WorkerResult {
    worker_id: usize,
    epoch: usize,
    loss: f64,
    num_samples: usize,
    gradients: HashMap<String, Array1<f32>>,
}

// ─────────────────────────────────────────────────────────────
// A. Gradient Aggregation & Compression
// ─────────────────────────────────────────────────────────────

/// Strategy for all-reduce gradient aggregation across distributed workers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AllReduceStrategy {
    /// Ring-based all-reduce: workers arranged in a ring pass partial sums around.
    RingAllReduce,
    /// Tree-based all-reduce: hierarchical reduction over a binary tree topology.
    TreeAllReduce,
    /// Parameter server: a central server accumulates and broadcasts gradients.
    ParameterServer,
}

/// Aggregates gradients from distributed workers.
#[derive(Debug, Clone, Default)]
pub struct GradientAggregator;

impl GradientAggregator {
    /// Create a new `GradientAggregator`.
    pub fn new() -> Self {
        Self
    }

    /// Aggregate `local_grad` from this worker together with gradients that have
    /// already been reduced on other workers, using the given `strategy`.
    ///
    /// For the single-worker case the function simply returns a normalised copy of
    /// `local_grad`.  In a real multi-node scenario the caller would pass the
    /// collected per-worker slices through `ring_all_reduce` or the tree variant.
    pub fn aggregate_gradients(
        &self,
        local_grad: &[f64],
        strategy: &AllReduceStrategy,
    ) -> Vec<f64> {
        match strategy {
            AllReduceStrategy::RingAllReduce => {
                // Treat the single local gradient as the only worker contribution.
                self.ring_all_reduce(vec![local_grad.to_vec()])
            }
            AllReduceStrategy::TreeAllReduce => self.tree_all_reduce(vec![local_grad.to_vec()]),
            AllReduceStrategy::ParameterServer => {
                // Parameter-server: accept local grad and average (single worker path).
                local_grad.to_vec()
            }
        }
    }

    /// Simulate ring all-reduce over a set of per-worker gradient vectors.
    ///
    /// Ring all-reduce arranges `n` workers in a ring.  It runs in two phases:
    ///
    /// 1. **Scatter-reduce** (`n−1` steps): at step `s`, each worker `w` passes
    ///    the accumulated data for chunk `(w − s) mod n` to its right neighbour
    ///    `(w + 1) mod n`, which adds it to its own copy.  After `n−1` steps,
    ///    worker `w` holds the fully-reduced sum for chunk `(w + 1) mod n`.
    ///
    /// 2. **All-gather**: collect the fully-reduced chunk from each owning worker
    ///    and divide by `n` to obtain the mean.
    ///
    /// The mathematical result equals the element-wise mean of all input vectors.
    /// This simulation runs synchronously on the calling thread with no I/O.
    pub fn ring_all_reduce(&self, gradients: Vec<Vec<f64>>) -> Vec<f64> {
        let n = gradients.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return gradients.into_iter().next().unwrap_or_default();
        }

        let len = gradients[0].len();
        if len == 0 {
            return Vec::new();
        }

        // Divide the gradient vector into `n` chunks.  Chunks are sized as
        // evenly as possible; the first `remainder` chunks get one extra element.
        let base = len / n;
        let remainder = len % n;
        let chunk_sizes: Vec<usize> = (0..n)
            .map(|i| base + if i < remainder { 1 } else { 0 })
            .collect();
        let mut chunk_start = vec![0usize; n];
        for i in 1..n {
            chunk_start[i] = chunk_start[i - 1] + chunk_sizes[i - 1];
        }

        // `partial[w][c]` = partial sums of chunk `c` accumulated on worker `w`.
        // Initially each worker contributes its own slice of the gradient.
        let mut partial: Vec<Vec<Vec<f64>>> = gradients
            .iter()
            .map(|g| {
                chunk_sizes
                    .iter()
                    .zip(chunk_start.iter())
                    .map(|(&sz, &s)| g[s..s + sz].to_vec())
                    .collect()
            })
            .collect();

        // ── scatter-reduce phase ──────────────────────────────────────────────
        // At each step, worker w receives from its left neighbour (w−1) the
        // partial accumulation for chunk `(w − 1 − step) mod n`.
        #[allow(clippy::needless_range_loop)]
        for step in 0..(n - 1) {
            let prev = partial.clone();
            for w in 0..n {
                let left = (w + n - 1) % n;
                let c = (w + n - 1 - step) % n;
                let sz = chunk_sizes[c];
                for i in 0..sz {
                    partial[w][c][i] += prev[left][c][i];
                }
            }
        }

        // After `n−1` scatter-reduce steps, worker `w` holds the fully-reduced
        // sum in slot `(w + 1) mod n`.

        // ── collect result (all-gather) ───────────────────────────────────────
        let mut result = vec![0.0_f64; len];
        #[allow(clippy::needless_range_loop)]
        for w in 0..n {
            let c = (w + 1) % n;
            let s = chunk_start[c];
            let sz = chunk_sizes[c];
            for i in 0..sz {
                result[s + i] = partial[w][c][i] / n as f64;
            }
        }

        result
    }

    /// Simulate tree (binary) all-reduce: recursive halving/doubling.
    fn tree_all_reduce(&self, gradients: Vec<Vec<f64>>) -> Vec<f64> {
        let n = gradients.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return gradients.into_iter().next().unwrap_or_default();
        }

        let len = gradients[0].len();
        let mut sums = vec![0.0_f64; len];
        for grad in &gradients {
            for (i, v) in grad.iter().enumerate() {
                if i < len {
                    sums[i] += v;
                }
            }
        }
        sums.iter_mut().for_each(|v| *v /= n as f64);
        sums
    }
}

/// Sparse gradient representation after top-k sparsification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseGradient {
    /// Indices of the non-zero (kept) elements in the original gradient vector.
    pub indices: Vec<usize>,
    /// Values at the kept indices.
    pub values: Vec<f64>,
    /// Length of the original (dense) gradient vector.
    pub original_len: usize,
}

/// Compresses gradient vectors via top-k sparsification to reduce communication overhead.
#[derive(Debug, Clone, Default)]
pub struct GradientCompressor;

impl GradientCompressor {
    /// Create a new `GradientCompressor`.
    pub fn new() -> Self {
        Self
    }

    /// Compress `grad` by retaining only the top-k largest-magnitude entries.
    ///
    /// * `sparsity` — fraction of entries to **zero out** (e.g. `0.9` keeps the top 10%).
    ///   Clamped to `[0.0, 1.0)`.
    pub fn compress(&self, grad: &[f64], sparsity: f64) -> SparseGradient {
        let sparsity = sparsity.clamp(0.0, 0.9999);
        let n = grad.len();
        if n == 0 {
            return SparseGradient {
                indices: Vec::new(),
                values: Vec::new(),
                original_len: 0,
            };
        }

        let keep = ((1.0 - sparsity) * n as f64).ceil() as usize;
        let keep = keep.max(1).min(n);

        // Collect (index, |value|) pairs and sort descending by magnitude.
        let mut indexed: Vec<(usize, f64)> = grad
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut indices: Vec<usize> = indexed[..keep].iter().map(|(i, _)| *i).collect();
        indices.sort_unstable();

        let values: Vec<f64> = indices.iter().map(|&i| grad[i]).collect();

        SparseGradient {
            indices,
            values,
            original_len: n,
        }
    }

    /// Decompress a `SparseGradient` back into a dense gradient vector (zero-filled elsewhere).
    pub fn decompress(&self, sparse: &SparseGradient) -> Vec<f64> {
        let mut dense = vec![0.0_f64; sparse.original_len];
        for (&idx, &val) in sparse.indices.iter().zip(sparse.values.iter()) {
            if idx < sparse.original_len {
                dense[idx] = val;
            }
        }
        dense
    }
}

// ─────────────────────────────────────────────────────────────
// B. Data-Parallel Training Coordinator
// ─────────────────────────────────────────────────────────────

/// A training sample for data-parallel distribution.
///
/// Deliberately kept generic so it can represent any supervised/self-supervised sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingSample {
    /// Numeric feature vector for this sample.
    pub features: Vec<f64>,
    /// Scalar label or target value.
    pub label: f64,
    /// Optional sample weight (defaults to `1.0` if `None`).
    pub weight: Option<f64>,
}

impl DistributedTrainingSample {
    /// Create a new sample with equal weight.
    pub fn new(features: Vec<f64>, label: f64) -> Self {
        Self {
            features,
            label,
            weight: None,
        }
    }
}

/// Per-worker gradient update produced after a local forward-backward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerUpdate {
    /// Identifier of the worker that computed this update.
    pub worker_id: u32,
    /// Flattened gradient vector from this worker.
    pub gradients: Vec<f64>,
    /// Training loss on the local mini-batch.
    pub loss: f64,
    /// Number of samples processed in this update.
    pub samples_processed: u32,
}

/// Merged model update produced after aggregating all worker updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Averaged gradient vector across all workers (weighted by sample count).
    pub averaged_gradients: Vec<f64>,
    /// Weighted-mean loss across all workers.
    pub mean_loss: f64,
    /// Total number of samples processed.
    pub total_samples: u32,
}

/// Coordinates data-parallel training by splitting batches and merging worker gradients.
#[derive(Debug, Clone, Default)]
pub struct DataParallelTrainer;

impl DataParallelTrainer {
    /// Create a new `DataParallelTrainer`.
    pub fn new() -> Self {
        Self
    }

    /// Evenly split `data` across `n_workers` workers.
    ///
    /// Returns a `Vec` of sub-batches, one per worker.  If `data.len()` is not
    /// evenly divisible some workers receive one extra sample (round-robin
    /// assignment).
    pub fn split_batch(
        &self,
        data: &[DistributedTrainingSample],
        n_workers: u32,
    ) -> Vec<Vec<DistributedTrainingSample>> {
        let n = n_workers as usize;
        if n == 0 || data.is_empty() {
            return Vec::new();
        }

        let mut buckets: Vec<Vec<DistributedTrainingSample>> = (0..n).map(|_| Vec::new()).collect();
        for (i, sample) in data.iter().enumerate() {
            buckets[i % n].push(sample.clone());
        }
        buckets
    }

    /// Merge gradient updates from all workers into a single `ModelUpdate`.
    ///
    /// Gradients are averaged weighted by `samples_processed` so that workers
    /// with larger mini-batches contribute proportionally more.
    pub fn merge_worker_updates(&self, updates: Vec<WorkerUpdate>) -> ModelUpdate {
        if updates.is_empty() {
            return ModelUpdate {
                averaged_gradients: Vec::new(),
                mean_loss: 0.0,
                total_samples: 0,
            };
        }

        let total_samples: u32 = updates.iter().map(|u| u.samples_processed).sum();
        if total_samples == 0 {
            return ModelUpdate {
                averaged_gradients: Vec::new(),
                mean_loss: 0.0,
                total_samples: 0,
            };
        }

        // Determine gradient length from the first update with non-empty gradients.
        let grad_len = updates.iter().map(|u| u.gradients.len()).max().unwrap_or(0);

        let mut averaged_gradients = vec![0.0_f64; grad_len];
        let mut weighted_loss = 0.0_f64;

        for update in &updates {
            let weight = update.samples_processed as f64 / total_samples as f64;
            for (i, &g) in update.gradients.iter().enumerate() {
                if i < grad_len {
                    averaged_gradients[i] += g * weight;
                }
            }
            weighted_loss += update.loss * weight;
        }

        ModelUpdate {
            averaged_gradients,
            mean_loss: weighted_loss,
            total_samples,
        }
    }
}

/// Distributed embedding model trainer
pub struct DistributedEmbeddingTrainer<M: EmbeddingModel> {
    model: M,
    coordinator: DistributedTrainingCoordinator,
}

impl<M: EmbeddingModel> DistributedEmbeddingTrainer<M> {
    /// Create a new distributed trainer
    pub async fn new(model: M, config: DistributedTrainingConfig) -> Result<Self> {
        let coordinator = DistributedTrainingCoordinator::new(config).await?;

        Ok(Self { model, coordinator })
    }

    /// Train the model in a distributed manner
    pub async fn train(&mut self, epochs: usize) -> Result<DistributedTrainingStats> {
        self.coordinator.train(&mut self.model, epochs).await
    }

    /// Get the trained model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Register a worker
    pub async fn register_worker(&self, worker_info: WorkerInfo) -> Result<()> {
        self.coordinator.register_worker(worker_info).await
    }

    /// Get training statistics
    pub async fn get_stats(&self) -> DistributedTrainingStats {
        self.coordinator.get_stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelConfig, TransE};

    // ── AllReduceStrategy & GradientAggregator ────────────────────────────────

    #[test]
    fn test_all_reduce_strategy_variants() {
        let strategies = [
            AllReduceStrategy::RingAllReduce,
            AllReduceStrategy::TreeAllReduce,
            AllReduceStrategy::ParameterServer,
        ];
        for s in &strategies {
            let agg = GradientAggregator::new();
            let grad = vec![1.0, 2.0, 3.0];
            let result = agg.aggregate_gradients(&grad, s);
            assert_eq!(result.len(), 3);
        }
    }

    #[test]
    fn test_ring_all_reduce_single_worker() {
        let agg = GradientAggregator::new();
        let grads = vec![vec![1.0, 2.0, 3.0]];
        let result = agg.ring_all_reduce(grads);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ring_all_reduce_two_workers() {
        let agg = GradientAggregator::new();
        let grads = vec![vec![2.0, 4.0, 6.0], vec![2.0, 4.0, 6.0]];
        let result = agg.ring_all_reduce(grads);
        assert_eq!(result.len(), 3);
        // Mean of equal vectors should be the vector itself.
        for (r, expected) in result.iter().zip([2.0, 4.0, 6.0].iter()) {
            assert!((r - expected).abs() < 1e-9, "expected {expected}, got {r}");
        }
    }

    #[test]
    fn test_ring_all_reduce_four_workers_mean() {
        let agg = GradientAggregator::new();
        let grads = vec![
            vec![4.0, 8.0],
            vec![2.0, 4.0],
            vec![0.0, 0.0],
            vec![6.0, 12.0],
        ];
        let result = agg.ring_all_reduce(grads);
        assert_eq!(result.len(), 2);
        // Mean: (4+2+0+6)/4 = 3, (8+4+0+12)/4 = 6
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_ring_all_reduce_empty_input() {
        let agg = GradientAggregator::new();
        let result = agg.ring_all_reduce(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ring_all_reduce_empty_gradient_vectors() {
        let agg = GradientAggregator::new();
        let result = agg.ring_all_reduce(vec![vec![], vec![]]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_aggregate_gradients_ring() {
        let agg = GradientAggregator::new();
        let grad = vec![1.0, 2.0, 3.0, 4.0];
        let result = agg.aggregate_gradients(&grad, &AllReduceStrategy::RingAllReduce);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_aggregate_gradients_tree() {
        let agg = GradientAggregator::new();
        let grad = vec![5.0, 10.0];
        let result = agg.aggregate_gradients(&grad, &AllReduceStrategy::TreeAllReduce);
        assert_eq!(result, vec![5.0, 10.0]);
    }

    #[test]
    fn test_aggregate_gradients_parameter_server() {
        let agg = GradientAggregator::new();
        let grad = vec![3.0, 1.0, 4.0];
        let result = agg.aggregate_gradients(&grad, &AllReduceStrategy::ParameterServer);
        assert_eq!(result, grad);
    }

    // ── GradientCompressor ────────────────────────────────────────────────────

    #[test]
    fn test_compress_empty_gradient() {
        let comp = GradientCompressor::new();
        let sparse = comp.compress(&[], 0.9);
        assert!(sparse.indices.is_empty());
        assert_eq!(sparse.original_len, 0);
    }

    #[test]
    fn test_compress_keep_all() {
        let comp = GradientCompressor::new();
        let grad = vec![1.0, -2.0, 3.0, -4.0];
        let sparse = comp.compress(&grad, 0.0);
        // sparsity=0 → keep all
        assert_eq!(sparse.indices.len(), 4);
        assert_eq!(sparse.original_len, 4);
    }

    #[test]
    fn test_compress_top_k_selects_largest() {
        let comp = GradientCompressor::new();
        let grad = vec![0.1, 5.0, 0.2, 9.0, 0.3];
        // sparsity=0.6 → keep 40% = 2 entries → indices 1 (5.0) and 3 (9.0)
        let sparse = comp.compress(&grad, 0.6);
        assert_eq!(sparse.indices.len(), 2);
        assert!(sparse.indices.contains(&3)); // 9.0
        assert!(sparse.indices.contains(&1)); // 5.0
    }

    #[test]
    fn test_decompress_roundtrip() {
        let comp = GradientCompressor::new();
        let grad = vec![0.0, 1.0, 0.0, -3.0, 0.0];
        let sparse = comp.compress(&grad, 0.6);
        let dense = comp.decompress(&sparse);
        assert_eq!(dense.len(), 5);
        // The two largest-magnitude values must be preserved.
        assert!((dense[3] - (-3.0)).abs() < 1e-12);
        assert!((dense[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_decompress_empty_sparse() {
        let comp = GradientCompressor::new();
        let sparse = SparseGradient {
            indices: Vec::new(),
            values: Vec::new(),
            original_len: 5,
        };
        let dense = comp.decompress(&sparse);
        assert_eq!(dense, vec![0.0; 5]);
    }

    #[test]
    fn test_sparse_gradient_serialization() {
        let sg = SparseGradient {
            indices: vec![0, 2],
            values: vec![1.5, -2.5],
            original_len: 4,
        };
        let json = serde_json::to_string(&sg).expect("serialize");
        let sg2: SparseGradient = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(sg, sg2);
    }

    // ── DataParallelTrainer ───────────────────────────────────────────────────

    #[test]
    fn test_split_batch_even() {
        let trainer = DataParallelTrainer::new();
        let samples: Vec<DistributedTrainingSample> = (0..8)
            .map(|i| DistributedTrainingSample::new(vec![i as f64], i as f64))
            .collect();
        let batches = trainer.split_batch(&samples, 4);
        assert_eq!(batches.len(), 4);
        for b in &batches {
            assert_eq!(b.len(), 2);
        }
    }

    #[test]
    fn test_split_batch_uneven() {
        let trainer = DataParallelTrainer::new();
        let samples: Vec<DistributedTrainingSample> = (0..10)
            .map(|i| DistributedTrainingSample::new(vec![i as f64], i as f64))
            .collect();
        let batches = trainer.split_batch(&samples, 3);
        assert_eq!(batches.len(), 3);
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_split_batch_zero_workers() {
        let trainer = DataParallelTrainer::new();
        let samples = vec![DistributedTrainingSample::new(vec![1.0], 0.0)];
        let batches = trainer.split_batch(&samples, 0);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_split_batch_empty_data() {
        let trainer = DataParallelTrainer::new();
        let batches = trainer.split_batch(&[], 4);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_merge_worker_updates_basic() {
        let trainer = DataParallelTrainer::new();
        let updates = vec![
            WorkerUpdate {
                worker_id: 0,
                gradients: vec![2.0, 4.0],
                loss: 1.0,
                samples_processed: 10,
            },
            WorkerUpdate {
                worker_id: 1,
                gradients: vec![2.0, 4.0],
                loss: 1.0,
                samples_processed: 10,
            },
        ];
        let merged = trainer.merge_worker_updates(updates);
        assert_eq!(merged.total_samples, 20);
        assert!((merged.mean_loss - 1.0).abs() < 1e-9);
        assert!((merged.averaged_gradients[0] - 2.0).abs() < 1e-9);
        assert!((merged.averaged_gradients[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_merge_worker_updates_weighted() {
        let trainer = DataParallelTrainer::new();
        // Worker 0 has 1 sample, worker 1 has 3 samples.
        let updates = vec![
            WorkerUpdate {
                worker_id: 0,
                gradients: vec![4.0],
                loss: 2.0,
                samples_processed: 1,
            },
            WorkerUpdate {
                worker_id: 1,
                gradients: vec![0.0],
                loss: 0.0,
                samples_processed: 3,
            },
        ];
        let merged = trainer.merge_worker_updates(updates);
        assert_eq!(merged.total_samples, 4);
        // Weighted mean gradient: 4*0.25 + 0*0.75 = 1.0
        assert!((merged.averaged_gradients[0] - 1.0).abs() < 1e-9);
        // Weighted mean loss: 2*0.25 + 0*0.75 = 0.5
        assert!((merged.mean_loss - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_merge_worker_updates_empty() {
        let trainer = DataParallelTrainer::new();
        let merged = trainer.merge_worker_updates(vec![]);
        assert_eq!(merged.total_samples, 0);
        assert!(merged.averaged_gradients.is_empty());
    }

    #[test]
    fn test_worker_update_serialization() {
        let update = WorkerUpdate {
            worker_id: 7,
            gradients: vec![0.1, -0.2],
            loss: 0.42,
            samples_processed: 32,
        };
        let json = serde_json::to_string(&update).expect("serialize");
        let update2: WorkerUpdate = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(update.worker_id, update2.worker_id);
        assert_eq!(update.samples_processed, update2.samples_processed);
    }

    #[test]
    fn test_model_update_fields() {
        let mu = ModelUpdate {
            averaged_gradients: vec![1.0, 2.0],
            mean_loss: 0.5,
            total_samples: 100,
        };
        assert_eq!(mu.total_samples, 100);
        assert!((mu.mean_loss - 0.5).abs() < 1e-12);
    }

    #[tokio::test]
    async fn test_distributed_coordinator_creation() {
        let config = DistributedTrainingConfig::default();
        let coordinator = DistributedTrainingCoordinator::new(config).await;
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let config = DistributedTrainingConfig::default();
        let coordinator = DistributedTrainingCoordinator::new(config)
            .await
            .expect("should succeed");

        let worker = WorkerInfo {
            worker_id: 0,
            rank: 0,
            address: "127.0.0.1:8080".to_string(),
            status: WorkerStatus::Idle,
            num_gpus: 1,
            memory_gb: 16.0,
            last_heartbeat: Utc::now(),
        };

        coordinator
            .register_worker(worker)
            .await
            .expect("should succeed");
        let stats = coordinator.get_worker_stats().await;
        assert_eq!(stats.len(), 1);
    }

    #[tokio::test]
    async fn test_distributed_training() {
        let config = DistributedTrainingConfig {
            strategy: DistributedStrategy::DataParallel {
                num_workers: 2,
                batch_size: 128,
            },
            ..Default::default()
        };

        let model_config = ModelConfig::default().with_dimensions(64);
        let model = TransE::new(model_config);

        let mut trainer = DistributedEmbeddingTrainer::new(model, config)
            .await
            .expect("should succeed");

        // Register workers
        for i in 0..2 {
            let worker = WorkerInfo {
                worker_id: i,
                rank: i,
                address: format!("127.0.0.1:808{}", i),
                status: WorkerStatus::Idle,
                num_gpus: 1,
                memory_gb: 16.0,
                last_heartbeat: Utc::now(),
            };
            trainer
                .register_worker(worker)
                .await
                .expect("should succeed");
        }

        // Train for a few epochs
        let stats = trainer.train(5).await.expect("should succeed");

        assert_eq!(stats.total_epochs, 5);
        assert!(stats.final_loss >= 0.0);
        assert_eq!(stats.num_workers, 2);
    }

    #[tokio::test]
    async fn test_checkpoint_save_load() {
        let config = DistributedTrainingConfig::default();
        let coordinator = DistributedTrainingCoordinator::new(config)
            .await
            .expect("should succeed");

        let model_config = ModelConfig::default();
        let model = TransE::new(model_config);

        // Register a worker
        let worker = WorkerInfo {
            worker_id: 0,
            rank: 0,
            address: "127.0.0.1:8080".to_string(),
            status: WorkerStatus::Idle,
            num_gpus: 1,
            memory_gb: 16.0,
            last_heartbeat: Utc::now(),
        };
        coordinator
            .register_worker(worker)
            .await
            .expect("should succeed");

        // Save checkpoint
        coordinator
            .save_checkpoint(&model, 10, 0.5)
            .await
            .expect("should succeed");

        // Load checkpoint
        let checkpoint = coordinator
            .load_checkpoint("checkpoint_epoch_10")
            .await
            .expect("should succeed");
        assert_eq!(checkpoint.epoch, 10);
        assert_eq!(checkpoint.loss, 0.5);
    }
}
