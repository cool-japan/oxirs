//! Revolutionary Cluster Optimization Framework
//!
//! AI-powered distributed cluster optimization integrating advanced consensus,
//! intelligent data distribution, adaptive replication, and quantum-enhanced networking.
//!
//! ## Module Organization
//!
//! - `config` - Configuration structures and types
//! - `types` - Shared types and state structures
//! - `consensus_optimizer` - AI-powered consensus optimization
//! - `data_distribution` - Intelligent data distribution engine
//! - `replication_manager` - Adaptive replication management
//! - `network_optimizer` - Quantum network optimization
//! - `analytics` - Advanced cluster analytics
//! - `scaling` - Predictive scaling engine
//! - `coordinator` - Unified cluster coordinator
//! - `optimizer` - Main revolutionary cluster optimizer

pub mod analytics;
pub mod config;
pub mod consensus_optimizer;
pub mod coordinator;
pub mod data_distribution;
pub mod network_optimizer;
pub mod optimizer;
pub mod replication_manager;
pub mod scaling;
pub mod types;

// Re-export main types
pub use config::*;
pub use optimizer::RevolutionaryClusterOptimizer;
pub use types::*;

// Re-export component types for convenience
pub use analytics::AdvancedClusterAnalytics;
pub use consensus_optimizer::AIConsensusOptimizer;
pub use coordinator::ClusterUnifiedCoordinator;
pub use data_distribution::IntelligentDataDistributionEngine;
pub use network_optimizer::QuantumNetworkOptimizer;
pub use replication_manager::AdaptiveReplicationManager;
pub use scaling::PredictiveScalingEngine;
