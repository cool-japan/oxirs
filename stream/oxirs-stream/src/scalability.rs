//! # Scalability Features for Stream Processing
//!
//! This module provides comprehensive scalability capabilities including
//! horizontal and vertical scaling, adaptive buffering, and dynamic resource management.
//!
//! ## Features
//!
//! - **Horizontal Scaling**: Dynamic partition management and load balancing
//! - **Vertical Scaling**: Adaptive resource allocation
//! - **Adaptive Buffering**: Smart buffer sizing based on load
//! - **Dynamic Partitioning**: Automatic partition assignment and rebalancing
//! - **Resource Optimization**: CPU, memory, and network optimization
//! - **Auto-scaling**: Automatic scaling based on metrics
//!
//! ## Use Cases
//!
//! - **High Throughput**: Handle millions of events per second
//! - **Burst Handling**: Adapt to traffic spikes
//! - **Cost Optimization**: Scale down during low traffic
//! - **Fault Tolerance**: Redistribute load on failures

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Use SciRS2 for scientific computing and statistics

/// Simple moving average calculator
struct MovingAverage {
    window_size: usize,
    values: VecDeque<f64>,
    sum: f64,
}

impl MovingAverage {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
            }
        }
        self.values.push_back(value);
        self.sum += value;
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }
}

/// Scaling mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScalingMode {
    /// Manual scaling - no automatic scaling
    Manual,
    /// Horizontal scaling - add/remove partitions
    Horizontal,
    /// Vertical scaling - adjust resources
    Vertical,
    /// Both horizontal and vertical
    Hybrid,
}

/// Scaling direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingDirection {
    /// Scale up
    ScaleUp { amount: usize },
    /// Scale down
    ScaleDown { amount: usize },
    /// No scaling needed
    NoChange,
}

/// Partition strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Round-robin partitioning
    RoundRobin,
    /// Hash-based partitioning
    Hash { key_field: String },
    /// Range-based partitioning
    Range { ranges: Vec<(i64, i64)> },
    /// Consistent hashing
    ConsistentHash { virtual_nodes: usize },
    /// Custom partitioning
    Custom { strategy_name: String },
}

/// Load balancing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Least loaded
    LeastLoaded,
    /// Weighted distribution
    Weighted { weights: HashMap<String, f64> },
    /// Consistent hashing
    ConsistentHash,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU cores
    pub max_cpu_cores: usize,
    /// Maximum memory (bytes)
    pub max_memory_bytes: u64,
    /// Maximum network bandwidth (bytes/sec)
    pub max_network_bandwidth: u64,
    /// Maximum partitions
    pub max_partitions: usize,
    /// Minimum partitions
    pub min_partitions: usize,
}

/// Scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Scaling mode
    pub mode: ScalingMode,
    /// Partition strategy
    pub partition_strategy: PartitionStrategy,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Scale-up threshold (0.0 to 1.0)
    pub scale_up_threshold: f64,
    /// Scale-down threshold (0.0 to 1.0)
    pub scale_down_threshold: f64,
    /// Cooldown period between scaling operations
    pub cooldown_period: Duration,
    /// Enable adaptive buffering
    pub enable_adaptive_buffering: bool,
    /// Initial buffer size
    pub initial_buffer_size: usize,
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Minimum buffer size
    pub min_buffer_size: usize,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            mode: ScalingMode::Hybrid,
            partition_strategy: PartitionStrategy::RoundRobin,
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            resource_limits: ResourceLimits {
                max_cpu_cores: num_cpus::get(),
                max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                max_network_bandwidth: 1_000_000_000,     // 1 Gbps
                max_partitions: 100,
                min_partitions: 1,
            },
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_period: Duration::from_secs(60),
            enable_adaptive_buffering: true,
            initial_buffer_size: 10000,
            max_buffer_size: 1000000,
            min_buffer_size: 1000,
        }
    }
}

/// Partition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// Partition ID
    pub partition_id: String,
    /// Partition number
    pub partition_number: usize,
    /// Owner node
    pub owner_node: Option<String>,
    /// Replica nodes
    pub replica_nodes: Vec<String>,
    /// Current load (0.0 to 1.0)
    pub load: f64,
    /// Number of events
    pub event_count: u64,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Node metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Node ID
    pub node_id: String,
    /// Node address
    pub address: String,
    /// Assigned partitions
    pub partitions: Vec<usize>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Health status
    pub health: NodeHealth,
    /// Last heartbeat
    pub last_heartbeat: DateTime<Utc>,
}

/// Resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// Network usage (bytes/sec)
    pub network_usage: u64,
    /// Events per second
    pub events_per_second: f64,
}

/// Node health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
}

/// Adaptive buffer
pub struct AdaptiveBuffer<T> {
    /// Buffer configuration
    config: ScalingConfig,
    /// Current buffer
    buffer: Arc<RwLock<VecDeque<T>>>,
    /// Current buffer size limit
    current_size: Arc<RwLock<usize>>,
    /// Load statistics (using SciRS2)
    load_history: Arc<RwLock<VecDeque<f64>>>,
    /// Moving average calculator
    moving_avg: Arc<RwLock<MovingAverage>>,
    /// Last resize time
    last_resize: Arc<RwLock<Instant>>,
}

impl<T> AdaptiveBuffer<T> {
    /// Create a new adaptive buffer
    pub fn new(config: ScalingConfig) -> Self {
        Self {
            current_size: Arc::new(RwLock::new(config.initial_buffer_size)),
            config,
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            load_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            moving_avg: Arc::new(RwLock::new(MovingAverage::new(10))),
            last_resize: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Push an item to the buffer
    pub fn push(&self, item: T) -> Result<()> {
        let mut buffer = self.buffer.write();
        let current_size = *self.current_size.read();

        if buffer.len() >= current_size {
            // Buffer is full - try to resize or reject
            self.try_resize()?;

            let new_size = *self.current_size.read();
            if buffer.len() >= new_size {
                return Err(anyhow!("Buffer full: {}/{}", buffer.len(), new_size));
            }
        }

        buffer.push_back(item);
        self.update_load_metrics(buffer.len(), current_size);
        Ok(())
    }

    /// Pop an item from the buffer
    pub fn pop(&self) -> Option<T> {
        let mut buffer = self.buffer.write();
        let item = buffer.pop_front();

        let current_size = *self.current_size.read();
        self.update_load_metrics(buffer.len(), current_size);
        item
    }

    /// Get current buffer utilization
    pub fn utilization(&self) -> f64 {
        let buffer = self.buffer.read();
        let current_size = *self.current_size.read();
        buffer.len() as f64 / current_size as f64
    }

    /// Update load metrics
    fn update_load_metrics(&self, buffer_len: usize, current_size: usize) {
        let load = buffer_len as f64 / current_size as f64;

        let mut history = self.load_history.write();
        history.push_back(load);
        if history.len() > 100 {
            history.pop_front();
        }

        let mut moving_avg = self.moving_avg.write();
        moving_avg.add(load);
    }

    /// Try to resize the buffer based on load
    fn try_resize(&self) -> Result<()> {
        if !self.config.enable_adaptive_buffering {
            return Ok(());
        }

        let last_resize = *self.last_resize.read();
        if last_resize.elapsed() < Duration::from_secs(10) {
            // Cooldown period
            return Ok(());
        }

        let moving_avg = self.moving_avg.read();
        let avg_load = moving_avg.mean();

        let mut current_size = self.current_size.write();

        if avg_load > self.config.scale_up_threshold {
            // Scale up
            let new_size = (*current_size * 2).min(self.config.max_buffer_size);
            if new_size > *current_size {
                *current_size = new_size;
                *self.last_resize.write() = Instant::now();
                info!("Scaled up buffer to {}", new_size);
            }
        } else if avg_load < self.config.scale_down_threshold {
            // Scale down
            let new_size = (*current_size / 2).max(self.config.min_buffer_size);
            if new_size < *current_size {
                *current_size = new_size;
                *self.last_resize.write() = Instant::now();
                info!("Scaled down buffer to {}", new_size);
            }
        }

        Ok(())
    }

    /// Get current buffer size
    pub fn size(&self) -> usize {
        *self.current_size.read()
    }

    /// Get current buffer length
    pub fn len(&self) -> usize {
        self.buffer.read().len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.read().is_empty()
    }
}

/// Partition manager for horizontal scaling
pub struct PartitionManager {
    /// Configuration
    config: ScalingConfig,
    /// Partitions
    partitions: Arc<DashMap<usize, Partition>>,
    /// Nodes
    nodes: Arc<DashMap<String, Node>>,
    /// Partition assignments
    assignments: Arc<DashMap<usize, String>>,
    /// Last scaling operation
    last_scaling: Arc<RwLock<Instant>>,
    /// Counter for round-robin
    counter: Arc<RwLock<usize>>,
}

impl PartitionManager {
    /// Create a new partition manager
    pub fn new(config: ScalingConfig) -> Self {
        let manager = Self {
            config: config.clone(),
            partitions: Arc::new(DashMap::new()),
            nodes: Arc::new(DashMap::new()),
            assignments: Arc::new(DashMap::new()),
            last_scaling: Arc::new(RwLock::new(Instant::now())),
            counter: Arc::new(RwLock::new(0)),
        };

        // Initialize with minimum partitions
        for i in 0..config.resource_limits.min_partitions {
            manager.create_partition(i);
        }

        manager
    }

    /// Create a new partition
    fn create_partition(&self, partition_number: usize) {
        let partition = Partition {
            partition_id: Uuid::new_v4().to_string(),
            partition_number,
            owner_node: None,
            replica_nodes: Vec::new(),
            load: 0.0,
            event_count: 0,
            created_at: Utc::now(),
            last_updated: Utc::now(),
        };

        self.partitions.insert(partition_number, partition);
        info!("Created partition {}", partition_number);
    }

    /// Add a node to the cluster
    pub fn add_node(&self, node: Node) -> Result<()> {
        let node_id = node.node_id.clone();
        self.nodes.insert(node_id.clone(), node);

        // Rebalance partitions
        self.rebalance_partitions()?;

        info!("Added node {}", node_id);
        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&self, node_id: &str) -> Result<()> {
        self.nodes.remove(node_id);

        // Reassign partitions from this node
        self.rebalance_partitions()?;

        info!("Removed node {}", node_id);
        Ok(())
    }

    /// Rebalance partitions across nodes
    fn rebalance_partitions(&self) -> Result<()> {
        let nodes: Vec<_> = self.nodes.iter().map(|e| e.key().clone()).collect();

        if nodes.is_empty() {
            warn!("No nodes available for partition assignment");
            return Ok(());
        }

        let partitions: Vec<_> = self.partitions.iter().map(|e| *e.key()).collect();

        // Simple round-robin assignment
        for (idx, partition_num) in partitions.iter().enumerate() {
            let node_id = &nodes[idx % nodes.len()];
            self.assignments.insert(*partition_num, node_id.clone());

            // Update partition owner
            if let Some(mut partition) = self.partitions.get_mut(partition_num) {
                partition.owner_node = Some(node_id.clone());
                partition.last_updated = Utc::now();
            }

            // Update node partitions
            if let Some(mut node) = self.nodes.get_mut(node_id) {
                if !node.partitions.contains(partition_num) {
                    node.partitions.push(*partition_num);
                }
            }
        }

        debug!(
            "Rebalanced {} partitions across {} nodes",
            partitions.len(),
            nodes.len()
        );
        Ok(())
    }

    /// Evaluate scaling needs
    pub fn evaluate_scaling(&self) -> ScalingDirection {
        if !matches!(
            self.config.mode,
            ScalingMode::Horizontal | ScalingMode::Hybrid
        ) {
            return ScalingDirection::NoChange;
        }

        // Check cooldown
        if self.last_scaling.read().elapsed() < self.config.cooldown_period {
            return ScalingDirection::NoChange;
        }

        // Calculate average load across partitions
        let partitions: Vec<_> = self.partitions.iter().map(|e| e.clone()).collect();

        if partitions.is_empty() {
            return ScalingDirection::NoChange;
        }

        let avg_load = partitions.iter().map(|p| p.load).sum::<f64>() / partitions.len() as f64;

        if avg_load > self.config.scale_up_threshold
            && partitions.len() < self.config.resource_limits.max_partitions
        {
            // Scale up
            let amount = ((partitions.len() as f64 * 0.5).ceil() as usize)
                .min(self.config.resource_limits.max_partitions - partitions.len())
                .max(1);
            ScalingDirection::ScaleUp { amount }
        } else if avg_load < self.config.scale_down_threshold
            && partitions.len() > self.config.resource_limits.min_partitions
        {
            // Scale down
            let amount = ((partitions.len() as f64 * 0.25).ceil() as usize)
                .min(partitions.len() - self.config.resource_limits.min_partitions)
                .max(1);
            ScalingDirection::ScaleDown { amount }
        } else {
            ScalingDirection::NoChange
        }
    }

    /// Apply scaling decision
    pub fn apply_scaling(&self, direction: &ScalingDirection) -> Result<()> {
        match direction {
            ScalingDirection::ScaleUp { amount } => {
                let current_max = self.partitions.iter().map(|e| *e.key()).max().unwrap_or(0);

                for i in 1..=*amount {
                    let partition_num = current_max + i;
                    if partition_num < self.config.resource_limits.max_partitions {
                        self.create_partition(partition_num);
                    }
                }

                self.rebalance_partitions()?;
                *self.last_scaling.write() = Instant::now();

                info!("Scaled up by {} partitions", amount);
            }
            ScalingDirection::ScaleDown { amount } => {
                let partition_nums: Vec<_> = self.partitions.iter().map(|e| *e.key()).collect();

                // Remove the highest numbered partitions
                let mut removed = 0;
                for partition_num in partition_nums.iter().rev() {
                    if removed >= *amount {
                        break;
                    }
                    if partition_nums.len() - removed > self.config.resource_limits.min_partitions {
                        self.partitions.remove(partition_num);
                        self.assignments.remove(partition_num);
                        removed += 1;
                    }
                }

                self.rebalance_partitions()?;
                *self.last_scaling.write() = Instant::now();

                info!("Scaled down by {} partitions", removed);
            }
            ScalingDirection::NoChange => {}
        }

        Ok(())
    }

    /// Get partition for a key
    pub fn get_partition_for_key(&self, key: &str) -> usize {
        match &self.config.partition_strategy {
            PartitionStrategy::RoundRobin => {
                // Use counter for round-robin
                let mut counter = self.counter.write();
                let partition = *counter % self.partitions.len();
                *counter = counter.wrapping_add(1);
                partition
            }
            PartitionStrategy::Hash { .. } => {
                // Simple hash-based partitioning
                let hash = self.hash_key(key);
                (hash as usize) % self.partitions.len()
            }
            PartitionStrategy::ConsistentHash { .. } => {
                // Simplified consistent hashing
                let hash = self.hash_key(key);
                (hash as usize) % self.partitions.len()
            }
            _ => 0, // Default to first partition
        }
    }

    /// Simple hash function
    fn hash_key(&self, key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Update partition load
    pub fn update_partition_load(&self, partition_num: usize, load: f64) {
        if let Some(mut partition) = self.partitions.get_mut(&partition_num) {
            partition.load = load;
            partition.last_updated = Utc::now();
        }
    }

    /// Get partition count
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Auto-scaler that monitors metrics and applies scaling decisions
pub struct AutoScaler {
    /// Configuration
    config: ScalingConfig,
    /// Partition manager
    partition_manager: Arc<PartitionManager>,
    /// Monitoring interval
    monitoring_interval: Duration,
    /// Command channel
    command_tx: mpsc::UnboundedSender<ScalingCommand>,
    /// Background task
    _background_task: Option<tokio::task::JoinHandle<()>>,
}

/// Scaling command
enum ScalingCommand {
    Evaluate,
    Stop,
}

impl AutoScaler {
    /// Create a new auto-scaler
    pub fn new(config: ScalingConfig, partition_manager: Arc<PartitionManager>) -> Self {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel();

        let monitoring_interval = Duration::from_secs(30);

        let partition_manager_clone = partition_manager.clone();
        let background_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitoring_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let decision = partition_manager_clone.evaluate_scaling();
                        if !matches!(decision, ScalingDirection::NoChange) {
                            info!("Auto-scaler decision: {:?}", decision);
                            if let Err(e) = partition_manager_clone.apply_scaling(&decision) {
                                error!("Failed to apply scaling: {}", e);
                            }
                        }
                    }
                    Some(cmd) = command_rx.recv() => {
                        match cmd {
                            ScalingCommand::Evaluate => {
                                let decision = partition_manager_clone.evaluate_scaling();
                                if !matches!(decision, ScalingDirection::NoChange) {
                                    if let Err(e) = partition_manager_clone.apply_scaling(&decision) {
                                        error!("Failed to apply scaling: {}", e);
                                    }
                                }
                            }
                            ScalingCommand::Stop => break,
                        }
                    }
                }
            }
        });

        Self {
            config,
            partition_manager,
            monitoring_interval,
            command_tx,
            _background_task: Some(background_task),
        }
    }

    /// Trigger immediate evaluation
    pub fn evaluate_now(&self) -> Result<()> {
        self.command_tx
            .send(ScalingCommand::Evaluate)
            .map_err(|e| anyhow!("Failed to send command: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_buffer() {
        let config = ScalingConfig::default();
        let buffer: AdaptiveBuffer<u64> = AdaptiveBuffer::new(config);

        // Push items
        for i in 0..100 {
            buffer.push(i).unwrap();
        }

        assert_eq!(buffer.len(), 100);

        // Pop items
        for _ in 0..50 {
            assert!(buffer.pop().is_some());
        }

        assert_eq!(buffer.len(), 50);
    }

    #[test]
    fn test_partition_manager() {
        let config = ScalingConfig::default();
        let manager = PartitionManager::new(config);

        // Add a node
        let node = Node {
            node_id: "node-1".to_string(),
            address: "localhost:8001".to_string(),
            partitions: Vec::new(),
            resource_usage: ResourceUsage {
                cpu_usage: 0.5,
                memory_usage: 1024 * 1024 * 1024,
                network_usage: 1000000,
                events_per_second: 1000.0,
            },
            health: NodeHealth::Healthy,
            last_heartbeat: Utc::now(),
        };

        manager.add_node(node).unwrap();

        assert_eq!(manager.node_count(), 1);
        assert!(manager.partition_count() >= 1);
    }

    #[test]
    fn test_partition_assignment() {
        let config = ScalingConfig::default();
        let manager = PartitionManager::new(config);

        let partition = manager.get_partition_for_key("test-key");
        assert!(partition < manager.partition_count());
    }
}
