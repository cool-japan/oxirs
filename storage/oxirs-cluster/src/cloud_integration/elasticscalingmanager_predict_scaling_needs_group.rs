//! # ElasticScalingManager - predict_scaling_needs_group Methods
//!
//! This module contains method implementations for `ElasticScalingManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::{ScalingPrediction, Trend};

use super::elasticscalingmanager_type::ElasticScalingManager;

impl ElasticScalingManager {
    /// Predict future scaling needs using statistical analysis
    pub async fn predict_scaling_needs(&self, horizon_minutes: u32) -> ScalingPrediction {
        let history = self.metrics_history.read().await;
        if history.len() < 60 {
            return ScalingPrediction {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time should be after UNIX_EPOCH")
                    .as_secs(),
                horizon_minutes,
                predicted_cpu: 0.5,
                predicted_memory: 0.5,
                predicted_nodes_needed: self.config.min_nodes,
                confidence: 0.0,
                trend: Trend::Stable,
            };
        }
        let cpu_values: Vec<f64> = history.iter().map(|m| m.avg_cpu_utilization).collect();
        let mem_values: Vec<f64> = history.iter().map(|m| m.avg_memory_utilization).collect();
        let (predicted_cpu, cpu_trend) = self.exponential_smoothing_forecast(&cpu_values);
        let (predicted_memory, mem_trend) = self.exponential_smoothing_forecast(&mem_values);
        let cpu_variance = self.calculate_variance(&cpu_values);
        let confidence = (1.0 - cpu_variance.min(1.0)).max(0.0);
        let trend = if cpu_trend > 0.1 || mem_trend > 0.1 {
            Trend::Increasing
        } else if cpu_trend < -0.1 || mem_trend < -0.1 {
            Trend::Decreasing
        } else {
            Trend::Stable
        };
        let max_util = predicted_cpu.max(predicted_memory);
        let predicted_nodes = ((max_util / self.config.target_cpu_utilization)
            * history.back().map(|m| m.node_count as f64).unwrap_or(3.0))
        .ceil() as u32;
        let predicted_nodes = predicted_nodes
            .max(self.config.min_nodes)
            .min(self.config.max_nodes);
        ScalingPrediction {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_secs(),
            horizon_minutes,
            predicted_cpu,
            predicted_memory,
            predicted_nodes_needed: predicted_nodes,
            confidence,
            trend,
        }
    }
    /// Helper: Exponential smoothing forecast
    fn exponential_smoothing_forecast(&self, values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.5, 0.0);
        }
        let alpha = 0.3;
        let mut level = values[0];
        let mut trend = 0.0;
        for (_i, &value) in values.iter().enumerate().skip(1) {
            let prev_level = level;
            level = alpha * value + (1.0 - alpha) * (level + trend);
            trend = 0.1 * (level - prev_level) + 0.9 * trend;
        }
        let forecast = level + trend;
        (forecast.clamp(0.0, 1.0), trend)
    }
    /// Helper: Calculate variance
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}
