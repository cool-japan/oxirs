//! # ElasticScalingManager - update_metrics_group Methods
//!
//! This module contains method implementations for `ElasticScalingManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ClusterMetrics;

use super::elasticscalingmanager_type::ElasticScalingManager;

impl ElasticScalingManager {
    /// Update cluster metrics
    pub async fn update_metrics(&self, metrics: ClusterMetrics) {
        let mut history = self.metrics_history.write().await;
        history.push_back(metrics);
        while history.len() > 3600 {
            history.pop_front();
        }
    }
}
