//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use super::types::{ClusterMetrics, ElasticScalingConfig, NodeInstance, ScalingEvent};

/// Elastic scaling manager for cloud-based auto-scaling
pub struct ElasticScalingManager {
    pub(super) config: ElasticScalingConfig,
    pub(super) current_nodes: Arc<RwLock<Vec<NodeInstance>>>,
    pub(super) metrics_history: Arc<RwLock<VecDeque<ClusterMetrics>>>,
    pub(super) last_scaling_time: Arc<RwLock<Instant>>,
    pub(super) scaling_events: Arc<RwLock<VecDeque<ScalingEvent>>>,
}
