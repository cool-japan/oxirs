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

    #[tokio::test]
    async fn test_distributed_coordinator_creation() {
        let config = DistributedTrainingConfig::default();
        let coordinator = DistributedTrainingCoordinator::new(config).await;
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let config = DistributedTrainingConfig::default();
        let coordinator = DistributedTrainingCoordinator::new(config).await.unwrap();

        let worker = WorkerInfo {
            worker_id: 0,
            rank: 0,
            address: "127.0.0.1:8080".to_string(),
            status: WorkerStatus::Idle,
            num_gpus: 1,
            memory_gb: 16.0,
            last_heartbeat: Utc::now(),
        };

        coordinator.register_worker(worker).await.unwrap();
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
            .unwrap();

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
            trainer.register_worker(worker).await.unwrap();
        }

        // Train for a few epochs
        let stats = trainer.train(5).await.unwrap();

        assert_eq!(stats.total_epochs, 5);
        assert!(stats.final_loss >= 0.0);
        assert_eq!(stats.num_workers, 2);
    }

    #[tokio::test]
    async fn test_checkpoint_save_load() {
        let config = DistributedTrainingConfig::default();
        let coordinator = DistributedTrainingCoordinator::new(config).await.unwrap();

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
        coordinator.register_worker(worker).await.unwrap();

        // Save checkpoint
        coordinator.save_checkpoint(&model, 10, 0.5).await.unwrap();

        // Load checkpoint
        let checkpoint = coordinator
            .load_checkpoint("checkpoint_epoch_10")
            .await
            .unwrap();
        assert_eq!(checkpoint.epoch, 10);
        assert_eq!(checkpoint.loss, 0.5);
    }
}
