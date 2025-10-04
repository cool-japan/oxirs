//! Revolutionary Cluster Configuration Types

use serde::{Deserialize, Serialize};
/// Revolutionary cluster optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevolutionaryClusterConfig {
    /// Enable AI-powered consensus optimization
    pub enable_ai_consensus_optimization: bool,
    /// Enable intelligent data distribution
    pub enable_intelligent_data_distribution: bool,
    /// Enable adaptive replication strategies
    pub enable_adaptive_replication: bool,
    /// Enable quantum-enhanced networking
    pub enable_quantum_networking: bool,
    /// Enable advanced cluster analytics
    pub enable_advanced_cluster_analytics: bool,
    /// Enable predictive scaling
    pub enable_predictive_scaling: bool,
    /// Consensus optimization configuration
    pub consensus_config: ConsensusOptimizationConfig,
    /// Data distribution configuration
    pub data_distribution_config: DataDistributionConfig,
    /// Replication strategy configuration
    pub replication_config: AdaptiveReplicationConfig,
    /// Network optimization configuration
    pub network_config: NetworkOptimizationConfig,
    /// Performance targets
    pub performance_targets: ClusterPerformanceTargets,
}

impl Default for RevolutionaryClusterConfig {
    fn default() -> Self {
        Self {
            enable_ai_consensus_optimization: true,
            enable_intelligent_data_distribution: true,
            enable_adaptive_replication: true,
            enable_quantum_networking: true,
            enable_advanced_cluster_analytics: true,
            enable_predictive_scaling: true,
            consensus_config: ConsensusOptimizationConfig::default(),
            data_distribution_config: DataDistributionConfig::default(),
            replication_config: AdaptiveReplicationConfig::default(),
            network_config: NetworkOptimizationConfig::default(),
            performance_targets: ClusterPerformanceTargets::default(),
        }
    }
}

/// Consensus optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusOptimizationConfig {
    /// Enable AI-powered leader selection
    pub enable_ai_leader_selection: bool,
    /// Enable quantum consensus algorithms
    pub enable_quantum_consensus: bool,
    /// Enable adaptive timeout management
    pub enable_adaptive_timeouts: bool,
    /// Consensus algorithm optimization strategy
    pub optimization_strategy: ConsensusOptimizationStrategy,
    /// Leader election prediction window
    pub leader_prediction_window_ms: u64,
    /// Consensus performance monitoring frequency
    pub monitoring_frequency_ms: u64,
}

impl Default for ConsensusOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_ai_leader_selection: true,
            enable_quantum_consensus: true,
            enable_adaptive_timeouts: true,
            optimization_strategy: ConsensusOptimizationStrategy::AIAdaptive,
            leader_prediction_window_ms: 5000,
            monitoring_frequency_ms: 100,
        }
    }
}

/// Consensus optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusOptimizationStrategy {
    Traditional,
    AIAssisted,
    AIAdaptive,
    QuantumEnhanced,
    HybridQuantumAI,
}

/// Data distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDistributionConfig {
    /// Enable ML-powered data placement
    pub enable_ml_data_placement: bool,
    /// Enable dynamic sharding
    pub enable_dynamic_sharding: bool,
    /// Enable load balancing optimization
    pub enable_load_balancing_optimization: bool,
    /// Data distribution strategy
    pub distribution_strategy: DataDistributionStrategy,
    /// Shard rebalancing threshold
    pub rebalancing_threshold: f64,
    /// Data placement prediction interval
    pub placement_prediction_interval_ms: u64,
}

impl Default for DataDistributionConfig {
    fn default() -> Self {
        Self {
            enable_ml_data_placement: true,
            enable_dynamic_sharding: true,
            enable_load_balancing_optimization: true,
            distribution_strategy: DataDistributionStrategy::MLOptimized,
            rebalancing_threshold: 0.2,
            placement_prediction_interval_ms: 10000,
        }
    }
}

/// Data distribution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataDistributionStrategy {
    RoundRobin,
    HashBased,
    LoadAware,
    MLOptimized,
    QuantumInspired,
}

/// Adaptive replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveReplicationConfig {
    /// Enable intelligent replication factor adjustment
    pub enable_intelligent_replication_factor: bool,
    /// Enable cross-region replication optimization
    pub enable_cross_region_optimization: bool,
    /// Enable failure prediction and preemptive replication
    pub enable_failure_prediction: bool,
    /// Replication strategy
    pub replication_strategy: AdaptiveReplicationStrategy,
    /// Base replication factor
    pub base_replication_factor: usize,
    /// Maximum replication factor
    pub max_replication_factor: usize,
    /// Failure prediction window
    pub failure_prediction_window_ms: u64,
}

impl Default for AdaptiveReplicationConfig {
    fn default() -> Self {
        Self {
            enable_intelligent_replication_factor: true,
            enable_cross_region_optimization: true,
            enable_failure_prediction: true,
            replication_strategy: AdaptiveReplicationStrategy::AIAdaptive,
            base_replication_factor: 3,
            max_replication_factor: 7,
            failure_prediction_window_ms: 30000,
        }
    }
}

/// Adaptive replication strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptiveReplicationStrategy {
    Static,
    LoadBased,
    LatencyOptimized,
    AIAdaptive,
    QuantumOptimized,
}

/// Network optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimizationConfig {
    /// Enable quantum network optimization
    pub enable_quantum_network_optimization: bool,
    /// Enable adaptive routing
    pub enable_adaptive_routing: bool,
    /// Enable network congestion prediction
    pub enable_congestion_prediction: bool,
    /// Network optimization strategy
    pub optimization_strategy: NetworkOptimizationStrategy,
    /// Network monitoring interval
    pub monitoring_interval_ms: u64,
    /// Congestion prediction window
    pub congestion_prediction_window_ms: u64,
}

impl Default for NetworkOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_quantum_network_optimization: true,
            enable_adaptive_routing: true,
            enable_congestion_prediction: true,
            optimization_strategy: NetworkOptimizationStrategy::QuantumEnhanced,
            monitoring_interval_ms: 500,
            congestion_prediction_window_ms: 10000,
        }
    }
}

/// Network optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkOptimizationStrategy {
    Traditional,
    LoadBalanced,
    LatencyOptimized,
    AIOptimized,
    QuantumEnhanced,
}

/// Cluster performance targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPerformanceTargets {
    /// Target consensus latency in milliseconds
    pub target_consensus_latency_ms: u64,
    /// Target query throughput (queries per second)
    pub target_query_throughput_qps: f64,
    /// Target cluster availability (0.0-1.0)
    pub target_availability: f64,
    /// Target data consistency level (0.0-1.0)
    pub target_consistency: f64,
    /// Target network utilization (0.0-1.0)
    pub target_network_utilization: f64,
    /// Target memory efficiency (MB per node)
    pub target_memory_efficiency_mb: f64,
}

impl Default for ClusterPerformanceTargets {
    fn default() -> Self {
        Self {
            target_consensus_latency_ms: 100,
            target_query_throughput_qps: 10000.0,
            target_availability: 0.999,
            target_consistency: 0.95,
            target_network_utilization: 0.8,
            target_memory_efficiency_mb: 1024.0,
        }
    }
}
