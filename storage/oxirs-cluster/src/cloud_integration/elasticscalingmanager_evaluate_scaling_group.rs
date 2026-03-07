//! # ElasticScalingManager - evaluate_scaling_group Methods
//!
//! This module contains method implementations for `ElasticScalingManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{ClusterMetrics, NodeInstance, ScalingDecision};

use super::elasticscalingmanager_type::ElasticScalingManager;

impl ElasticScalingManager {
    /// Evaluate scaling decision
    pub async fn evaluate_scaling(&self) -> ScalingDecision {
        let history = self.metrics_history.read().await;
        let nodes = self.current_nodes.read().await;
        let last_scaling = *self.last_scaling_time.read().await;
        if last_scaling.elapsed() < Duration::from_secs(self.config.cooldown_seconds as u64) {
            return ScalingDecision::NoAction {
                reason: "In cooldown period".to_string(),
            };
        }
        let recent_metrics: Vec<&ClusterMetrics> = history.iter().rev().take(300).collect();
        if recent_metrics.is_empty() {
            return ScalingDecision::NoAction {
                reason: "Insufficient metrics data".to_string(),
            };
        }
        let avg_cpu: f64 = recent_metrics
            .iter()
            .map(|m| m.avg_cpu_utilization)
            .sum::<f64>()
            / recent_metrics.len() as f64;
        let avg_mem: f64 = recent_metrics
            .iter()
            .map(|m| m.avg_memory_utilization)
            .sum::<f64>()
            / recent_metrics.len() as f64;
        let current_count = nodes.len() as u32;
        if avg_cpu > self.config.scale_up_threshold || avg_mem > self.config.scale_up_threshold {
            if current_count < self.config.max_nodes {
                let scale_count = self.calculate_scale_up_count(avg_cpu, avg_mem);
                let use_spot = self.should_use_spot(&nodes);
                let instance_type = self.select_instance_type(avg_cpu, avg_mem);
                return ScalingDecision::ScaleUp {
                    count: scale_count,
                    instance_type,
                    use_spot,
                    reason: format!(
                        "High utilization - CPU: {:.1}%, Memory: {:.1}%",
                        avg_cpu * 100.0,
                        avg_mem * 100.0
                    ),
                };
            } else {
                return ScalingDecision::NoAction {
                    reason: "Already at maximum nodes".to_string(),
                };
            }
        }
        if avg_cpu < self.config.scale_down_threshold && avg_mem < self.config.scale_down_threshold
        {
            if current_count > self.config.min_nodes {
                let scale_count = self.calculate_scale_down_count(avg_cpu, avg_mem, current_count);
                let instance_ids = self.select_nodes_to_terminate(&nodes, scale_count);
                return ScalingDecision::ScaleDown {
                    count: scale_count,
                    instance_ids,
                    reason: format!(
                        "Low utilization - CPU: {:.1}%, Memory: {:.1}%",
                        avg_cpu * 100.0,
                        avg_mem * 100.0
                    ),
                };
            } else {
                return ScalingDecision::NoAction {
                    reason: "Already at minimum nodes".to_string(),
                };
            }
        }
        ScalingDecision::NoAction {
            reason: "Utilization within target range".to_string(),
        }
    }
    /// Helper: Calculate scale up count
    fn calculate_scale_up_count(&self, avg_cpu: f64, avg_mem: f64) -> u32 {
        let max_util = avg_cpu.max(avg_mem);
        let scale_factor = max_util / self.config.target_cpu_utilization;
        let current = self.current_nodes.try_read().map(|n| n.len()).unwrap_or(3) as u32;
        let additional = ((current as f64 * (scale_factor - 1.0)).ceil() as u32).max(1);
        additional.min(self.config.max_nodes - current)
    }
    /// Helper: Calculate scale down count
    fn calculate_scale_down_count(&self, avg_cpu: f64, avg_mem: f64, current: u32) -> u32 {
        let max_util = avg_cpu.max(avg_mem);
        let target_nodes =
            ((current as f64 * max_util) / self.config.target_cpu_utilization).ceil() as u32;
        let reduction = current - target_nodes.max(self.config.min_nodes);
        reduction.min(current - self.config.min_nodes)
    }
    /// Helper: Decide whether to use spot instances
    fn should_use_spot(&self, nodes: &[NodeInstance]) -> bool {
        if !self.config.use_spot_instances {
            return false;
        }
        let spot_count = nodes.iter().filter(|n| n.is_spot).count();
        let total = nodes.len();
        if total == 0 {
            return true;
        }
        (spot_count as f64 / total as f64) < self.config.max_spot_ratio
    }
    /// Helper: Select appropriate instance type
    fn select_instance_type(&self, avg_cpu: f64, avg_mem: f64) -> String {
        if avg_cpu > 0.7 || avg_mem > 0.7 {
            "large".to_string()
        } else if avg_cpu > 0.4 || avg_mem > 0.4 {
            "medium".to_string()
        } else {
            "small".to_string()
        }
    }
    /// Helper: Select nodes to terminate (prefer spot instances)
    fn select_nodes_to_terminate(&self, nodes: &[NodeInstance], count: u32) -> Vec<String> {
        let mut candidates: Vec<&NodeInstance> = nodes.iter().collect();
        candidates.sort_by(|a, b| {
            if a.is_spot != b.is_spot {
                b.is_spot.cmp(&a.is_spot)
            } else {
                a.launch_time.cmp(&b.launch_time)
            }
        });
        candidates
            .iter()
            .take(count as usize)
            .map(|n| n.instance_id.clone())
            .collect()
    }
}
