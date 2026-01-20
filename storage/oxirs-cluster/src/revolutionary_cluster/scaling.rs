//! Predictive Scaling Engine

use anyhow::Result;
use scirs2_core::ml_pipeline::MLPipeline;
use scirs2_core::ndarray_ext::Array1;
use std::collections::VecDeque;
use std::time::Instant;
use super::types::*;
pub struct PredictiveScalingEngine {
    scaling_ml: MLPipeline,
    scaling_history: VecDeque<ScalingEvent>,
    capacity_predictor: CapacityPredictor,
}

impl PredictiveScalingEngine {
    async fn new() -> Result<Self> {
        Ok(Self {
            scaling_ml: MLPipeline::new(),
            scaling_history: VecDeque::with_capacity(1000),
            capacity_predictor: CapacityPredictor::new(),
        })
    }

    async fn predict_scaling_requirements(&self, cluster_state: &ClusterState) -> Result<ScalingPrediction> {
        // Extract features for scaling prediction
        let features = self.extract_scaling_features(cluster_state);

        // Use ML to predict scaling needs
        let scaling_scores = self.scaling_ml.predict(&features).await?;

        // Predict capacity requirements
        let capacity_prediction = self.capacity_predictor.predict_capacity_needs(cluster_state).await?;

        Ok(ScalingPrediction {
            scale_up_probability: scaling_scores.get(0).copied().unwrap_or(0.3),
            scale_down_probability: scaling_scores.get(1).copied().unwrap_or(0.1),
            optimal_node_count: capacity_prediction.optimal_node_count,
            scaling_timeline: capacity_prediction.scaling_timeline,
            scaling_recommendations: self.generate_scaling_recommendations(&capacity_prediction),
            cost_impact_analysis: self.analyze_cost_impact(&capacity_prediction),
        })
    }

    fn extract_scaling_features(&self, cluster_state: &ClusterState) -> Vec<f64> {
        vec![
            cluster_state.performance_metrics.query_throughput_qps / 1000.0, // Normalize
            cluster_state.performance_metrics.consensus_latency_ms as f64 / 100.0, // Normalize
            cluster_state.performance_metrics.cpu_utilization,
            cluster_state.performance_metrics.memory_utilization,
            cluster_state.nodes.len() as f64,
        ]
    }

    fn generate_scaling_recommendations(&self, capacity_prediction: &CapacityPrediction) -> Vec<String> {
        let mut recommendations = Vec::new();

        if capacity_prediction.optimal_node_count > capacity_prediction.current_node_count {
            recommendations.push(format!(
                "Scale up: Add {} nodes within {}",
                capacity_prediction.optimal_node_count - capacity_prediction.current_node_count,
                format_duration(capacity_prediction.scaling_timeline)
            ));
        } else if capacity_prediction.optimal_node_count < capacity_prediction.current_node_count {
            recommendations.push(format!(
                "Scale down: Remove {} nodes to optimize costs",
                capacity_prediction.current_node_count - capacity_prediction.optimal_node_count
            ));
        }

        recommendations.push("Monitor cluster performance for next 24 hours".to_string());
        recommendations.push("Prepare auto-scaling policies for peak traffic".to_string());

        recommendations
    }

    fn analyze_cost_impact(&self, capacity_prediction: &CapacityPrediction) -> CostImpactAnalysis {
        let cost_per_node = 100.0; // Monthly cost per node
        let current_cost = capacity_prediction.current_node_count as f64 * cost_per_node;
        let projected_cost = capacity_prediction.optimal_node_count as f64 * cost_per_node;

        CostImpactAnalysis {
            current_monthly_cost: current_cost,
            projected_monthly_cost: projected_cost,
            cost_change: projected_cost - current_cost,
            cost_per_performance_unit: projected_cost / capacity_prediction.expected_performance_improvement,
            roi_analysis: if projected_cost > current_cost {
                format!("Cost increase of ${:.2}/month for {:.1}% performance improvement",
                       projected_cost - current_cost,
                       (capacity_prediction.expected_performance_improvement - 1.0) * 100.0)
            } else {
                format!("Cost savings of ${:.2}/month", current_cost - projected_cost)
            },
        }
    }
}

/// Capacity predictor for scaling decisions
#[derive(Debug)]
pub struct CapacityPredictor {
    prediction_model: MLPipeline,
}

impl CapacityPredictor {
    fn new() -> Self {
        Self {
            prediction_model: MLPipeline::new(),
        }
    }

    async fn predict_capacity_needs(&self, cluster_state: &ClusterState) -> Result<CapacityPrediction> {
        let current_node_count = cluster_state.nodes.len();
        let current_throughput = cluster_state.performance_metrics.query_throughput_qps;

        // Simple capacity prediction based on current utilization
        let avg_cpu_utilization = cluster_state.performance_metrics.cpu_utilization;
        let avg_memory_utilization = cluster_state.performance_metrics.memory_utilization;

        let optimal_node_count = if avg_cpu_utilization > 0.8 || avg_memory_utilization > 0.8 {
            // Need to scale up
            current_node_count + ((avg_cpu_utilization.max(avg_memory_utilization) - 0.6) / 0.2).ceil() as usize
        } else if avg_cpu_utilization < 0.3 && avg_memory_utilization < 0.3 {
            // Could scale down
            (current_node_count as f64 * 0.8).max(3.0) as usize // Keep minimum of 3 nodes
        } else {
            current_node_count
        };

        let scaling_timeline = if optimal_node_count != current_node_count {
            Duration::from_secs(1800) // 30 minutes
        } else {
            Duration::from_secs(3600) // 1 hour for monitoring
        };

        Ok(CapacityPrediction {
            current_node_count,
            optimal_node_count,
            current_throughput,
            expected_throughput: current_throughput * (optimal_node_count as f64 / current_node_count as f64),
            scaling_timeline,
            expected_performance_improvement: optimal_node_count as f64 / current_node_count as f64,
        })
    }
}
