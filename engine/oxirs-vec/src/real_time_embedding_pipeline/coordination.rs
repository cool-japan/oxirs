//! Update coordination and synchronization for the real-time embedding pipeline

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tokio::sync::{mpsc, Semaphore};
use uuid::Uuid;

use crate::real_time_embedding_pipeline::{
    config::PipelineConfig,
    types::{UpdateOperation, UpdateBatch, CoordinationState, NodeStatus},
    traits::ProcessingPriority,
};

/// Configuration for update coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Maximum concurrent updates
    pub max_concurrent_updates: usize,
    /// Batch processing timeout
    pub batch_timeout: Duration,
    /// Coordination heartbeat interval
    pub heartbeat_interval: Duration,
    /// Node failure detection timeout
    pub failure_timeout: Duration,
    /// Maximum queue size for updates
    pub max_queue_size: usize,
    /// Enable distributed coordination
    pub enable_distributed: bool,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_updates: 10,
            batch_timeout: Duration::from_millis(100),
            heartbeat_interval: Duration::from_secs(5),
            failure_timeout: Duration::from_secs(30),
            max_queue_size: 10000,
            enable_distributed: false,
        }
    }
}

/// Update coordinator for managing concurrent operations
pub struct UpdateCoordinator {
    /// Configuration
    config: CoordinationConfig,
    /// Semaphore for controlling concurrency
    semaphore: Arc<Semaphore>,
    /// Update queue sender
    update_sender: Arc<RwLock<Option<mpsc::UnboundedSender<UpdateOperation>>>>,
    /// Coordination state
    state: Arc<RwLock<CoordinationState>>,
    /// Running flag
    is_running: AtomicBool,
    /// Processing statistics
    processed_count: AtomicU64,
    /// Failed operations count
    failed_count: AtomicU64,
    /// Active workers
    active_workers: AtomicU64,
}

impl UpdateCoordinator {
    /// Create a new update coordinator
    pub fn new(pipeline_config: &PipelineConfig) -> Result<Self> {
        let config = CoordinationConfig {
            max_concurrent_updates: pipeline_config.max_concurrent_updates,
            batch_timeout: Duration::from_millis(pipeline_config.batch_timeout_ms),
            ..Default::default()
        };

        let node_id = Uuid::new_v4().to_string();
        let state = Arc::new(RwLock::new(CoordinationState {
            node_id: node_id.clone(),
            leader_id: Some(node_id), // Single node initially
            status: NodeStatus::Active,
            last_heartbeat: SystemTime::now(),
            load_factor: 0.0,
            active_tasks: 0,
        }));

        Ok(Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_updates)),
            config,
            update_sender: Arc::new(RwLock::new(None)),
            state,
            is_running: AtomicBool::new(false),
            processed_count: AtomicU64::new(0),
            failed_count: AtomicU64::new(0),
            active_workers: AtomicU64::new(0),
        })
    }

    /// Start the coordinator
    pub async fn start(&self) -> Result<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Err(anyhow::anyhow!("Update coordinator is already running"));
        }

        self.is_running.store(true, Ordering::Release);

        // Create update processing channel
        let (sender, mut receiver) = mpsc::unbounded_channel::<UpdateOperation>();
        
        {
            let mut sender_guard = self.update_sender.write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire sender lock"))?;
            *sender_guard = Some(sender);
        }

        // Start background task for processing updates
        let coordinator = self.clone_for_task();
        tokio::spawn(async move {
            while let Some(operation) = receiver.recv().await {
                if let Err(e) = coordinator.process_update_operation(operation).await {
                    eprintln!("Error processing update operation: {}", e);
                    coordinator.failed_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        // Start heartbeat task
        self.start_heartbeat_task().await;

        Ok(())
    }

    /// Stop the coordinator
    pub async fn stop(&self) -> Result<()> {
        self.is_running.store(false, Ordering::Release);

        // Close the update sender
        {
            let mut sender_guard = self.update_sender.write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire sender lock"))?;
            *sender_guard = None;
        }

        // Wait for active operations to complete
        while self.active_workers.load(Ordering::Acquire) > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Update coordination state
        {
            let mut state = self.state.write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire state lock"))?;
            state.status = NodeStatus::Leaving;
        }

        Ok(())
    }

    /// Submit an update operation for processing
    pub async fn submit_update(&self, operation: UpdateOperation) -> Result<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(anyhow::anyhow!("Update coordinator is not running"));
        }

        let sender_guard = self.update_sender.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire sender lock"))?;
        
        if let Some(sender) = sender_guard.as_ref() {
            sender.send(operation)
                .map_err(|_| anyhow::anyhow!("Failed to send update operation"))?;
        } else {
            return Err(anyhow::anyhow!("Update coordinator not properly initialized"));
        }

        Ok(())
    }

    /// Submit a batch of update operations
    pub async fn submit_batch(&self, batch: UpdateBatch) -> Result<()> {
        for operation in batch.operations {
            self.submit_update(operation).await?;
        }
        Ok(())
    }

    /// Get coordination statistics
    pub fn get_statistics(&self) -> CoordinationStatistics {
        CoordinationStatistics {
            processed_count: self.processed_count.load(Ordering::Acquire),
            failed_count: self.failed_count.load(Ordering::Acquire),
            active_workers: self.active_workers.load(Ordering::Acquire) as usize,
            queue_size: self.get_queue_size(),
            is_running: self.is_running.load(Ordering::Acquire),
        }
    }

    /// Get coordination state
    pub fn get_state(&self) -> Result<CoordinationState> {
        let state = self.state.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire state lock"))?;
        Ok(state.clone())
    }

    /// Check if coordinator is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    // Private helper methods

    fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            semaphore: self.semaphore.clone(),
            update_sender: self.update_sender.clone(),
            state: self.state.clone(),
            is_running: AtomicBool::new(self.is_running.load(Ordering::Acquire)),
            processed_count: AtomicU64::new(self.processed_count.load(Ordering::Acquire)),
            failed_count: AtomicU64::new(self.failed_count.load(Ordering::Acquire)),
            active_workers: AtomicU64::new(self.active_workers.load(Ordering::Acquire)),
        }
    }

    async fn process_update_operation(&self, operation: UpdateOperation) -> Result<()> {
        // Acquire semaphore permit for concurrency control
        let permit = self.semaphore.acquire().await
            .map_err(|_| anyhow::anyhow!("Failed to acquire processing permit"))?;

        self.active_workers.fetch_add(1, Ordering::Relaxed);

        // Process the update operation
        let result = self.execute_update_operation(operation).await;

        self.active_workers.fetch_sub(1, Ordering::Relaxed);
        
        if result.is_ok() {
            self.processed_count.fetch_add(1, Ordering::Relaxed);
        }

        drop(permit);
        result
    }

    async fn execute_update_operation(&self, operation: UpdateOperation) -> Result<()> {
        // Simulate update processing
        match operation {
            UpdateOperation::Insert { id, content } => {
                // Process insert operation
                tokio::time::sleep(Duration::from_millis(1)).await;
                println!("Processed insert for id: {}", id);
            }
            UpdateOperation::Update { id, content, version } => {
                // Process update operation
                tokio::time::sleep(Duration::from_millis(1)).await;
                println!("Processed update for id: {} (version: {:?})", id, version);
            }
            UpdateOperation::Delete { id } => {
                // Process delete operation
                tokio::time::sleep(Duration::from_millis(1)).await;
                println!("Processed delete for id: {}", id);
            }
            UpdateOperation::Batch { operations } => {
                // Process batch operations recursively
                for op in operations {
                    self.execute_update_operation(op).await?;
                }
            }
        }
        Ok(())
    }

    async fn start_heartbeat_task(&self) {
        let state = self.state.clone();
        let heartbeat_interval = self.config.heartbeat_interval;
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            while is_running.load(Ordering::Acquire) {
                {
                    if let Ok(mut state_guard) = state.write() {
                        state_guard.last_heartbeat = SystemTime::now();
                        // Update load factor based on current metrics
                        // This is a simplified calculation
                        state_guard.load_factor = 0.5; // Placeholder
                    }
                }
                tokio::time::sleep(heartbeat_interval).await;
            }
        });
    }

    fn get_queue_size(&self) -> usize {
        // Simplified queue size calculation
        // In a real implementation, this would track the actual queue depth
        0
    }
}

/// Statistics for coordination operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStatistics {
    /// Total operations processed
    pub processed_count: u64,
    /// Total operations failed
    pub failed_count: u64,
    /// Currently active workers
    pub active_workers: usize,
    /// Current queue size
    pub queue_size: usize,
    /// Whether coordinator is running
    pub is_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real_time_embedding_pipeline::config::PipelineConfig;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = PipelineConfig::default();
        let coordinator = UpdateCoordinator::new(&config);
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_coordinator_start_stop() {
        let config = PipelineConfig::default();
        let coordinator = UpdateCoordinator::new(&config).unwrap();
        
        assert!(!coordinator.is_running());
        
        let start_result = coordinator.start().await;
        assert!(start_result.is_ok());
        assert!(coordinator.is_running());
        
        let stop_result = coordinator.stop().await;
        assert!(stop_result.is_ok());
    }

    #[tokio::test]
    async fn test_update_submission() {
        let config = PipelineConfig::default();
        let coordinator = UpdateCoordinator::new(&config).unwrap();
        
        coordinator.start().await.unwrap();
        
        let operation = UpdateOperation::Insert {
            id: "test_id".to_string(),
            content: "test_content".to_string(),
        };
        
        let result = coordinator.submit_update(operation).await;
        assert!(result.is_ok());
        
        coordinator.stop().await.unwrap();
    }
}