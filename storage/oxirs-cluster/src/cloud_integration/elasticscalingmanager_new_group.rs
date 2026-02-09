//! # ElasticScalingManager - new_group Methods
//!
//! This module contains method implementations for `ElasticScalingManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use super::types::{CostOptimization, CostRecommendation, ElasticScalingConfig};

use super::elasticscalingmanager_type::ElasticScalingManager;

impl ElasticScalingManager {
    /// Create new elastic scaling manager
    pub fn new(config: ElasticScalingConfig) -> Self {
        Self {
            config,
            current_nodes: Arc::new(RwLock::new(Vec::new())),
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            last_scaling_time: Arc::new(RwLock::new(Instant::now())),
            scaling_events: Arc::new(RwLock::new(VecDeque::new())),
        }
    }
    /// Calculate cost optimization recommendations
    pub async fn get_cost_optimization(&self) -> CostOptimization {
        let nodes = self.current_nodes.read().await;
        let mut on_demand_count = 0;
        let mut spot_count = 0;
        let mut total_hourly_cost = 0.0;
        let mut potential_savings = 0.0;
        for node in nodes.iter() {
            if let Some(instance_type) = self
                .config
                .instance_types
                .iter()
                .find(|t| t.name == node.instance_type)
            {
                if node.is_spot {
                    spot_count += 1;
                    total_hourly_cost += instance_type.spot_hourly_cost;
                } else {
                    on_demand_count += 1;
                    total_hourly_cost += instance_type.hourly_cost;
                    potential_savings += instance_type.hourly_cost - instance_type.spot_hourly_cost;
                }
            }
        }
        let current_spot_ratio = if nodes.is_empty() {
            0.0
        } else {
            spot_count as f64 / nodes.len() as f64
        };
        let mut recommendations = Vec::new();
        if current_spot_ratio < self.config.max_spot_ratio && on_demand_count > 0 {
            recommendations.push(CostRecommendation {
                action: "Increase spot instance usage".to_string(),
                estimated_savings: potential_savings * 0.5,
                risk_level: "Medium".to_string(),
                description: format!(
                    "Current spot ratio: {:.1}%. Can safely increase to {:.1}%",
                    current_spot_ratio * 100.0,
                    self.config.max_spot_ratio * 100.0
                ),
            });
        }
        let history = self.metrics_history.read().await;
        if let Some(recent) = history.back() {
            if recent.avg_cpu_utilization < 0.3 && recent.avg_memory_utilization < 0.3 {
                recommendations.push(CostRecommendation {
                    action: "Consider smaller instance types".to_string(),
                    estimated_savings: total_hourly_cost * 0.3,
                    risk_level: "Low".to_string(),
                    description: "Low utilization suggests over-provisioning".to_string(),
                });
            }
        }
        CostOptimization {
            current_hourly_cost: total_hourly_cost,
            current_monthly_cost: total_hourly_cost * 24.0 * 30.0,
            on_demand_count,
            spot_count,
            potential_monthly_savings: potential_savings * 24.0 * 30.0,
            recommendations,
        }
    }
}
