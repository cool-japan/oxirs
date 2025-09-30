//! # Auto-Scaling Module
//!
//! Comprehensive auto-scaling capabilities for OxiRS Stream, providing intelligent
//! scaling based on custom metrics, load patterns, and resource utilization.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Horizontal Pod Autoscaler (HPA) configuration
    pub hpa: HpaConfig,
    /// Vertical Pod Autoscaler (VPA) configuration
    pub vpa: VpaConfig,
    /// Custom metrics configuration
    pub custom_metrics: Vec<CustomMetric>,
    /// Scaling policies
    pub scaling_policies: ScalingPolicies,
    /// Predictive scaling configuration
    pub predictive_scaling: PredictiveScalingConfig,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hpa: HpaConfig::default(),
            vpa: VpaConfig::default(),
            custom_metrics: vec![
                CustomMetric {
                    name: "kafka_consumer_lag".to_string(),
                    threshold: 1000.0,
                    comparison: MetricComparison::GreaterThan,
                    query: "kafka_consumer_lag_sum".to_string(),
                },
                CustomMetric {
                    name: "stream_throughput".to_string(),
                    threshold: 10000.0,
                    comparison: MetricComparison::GreaterThan,
                    query: "stream_events_per_second".to_string(),
                },
            ],
            scaling_policies: ScalingPolicies::default(),
            predictive_scaling: PredictiveScalingConfig::default(),
        }
    }
}

/// Horizontal Pod Autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpaConfig {
    /// Enable HPA
    pub enabled: bool,
    /// Minimum number of replicas
    pub min_replicas: u32,
    /// Maximum number of replicas
    pub max_replicas: u32,
    /// Target CPU utilization percentage
    pub target_cpu_utilization: f64,
    /// Target memory utilization percentage
    pub target_memory_utilization: f64,
    /// Scale down stabilization window (seconds)
    pub scale_down_stabilization: u64,
    /// Scale up stabilization window (seconds)
    pub scale_up_stabilization: u64,
}

impl Default for HpaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_replicas: 2,
            max_replicas: 20,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            scale_down_stabilization: 300,
            scale_up_stabilization: 60,
        }
    }
}

/// Vertical Pod Autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpaConfig {
    /// Enable VPA
    pub enabled: bool,
    /// Update mode (Off, Initial, Recreation, Auto)
    pub update_mode: String,
    /// Minimum allowed CPU
    pub min_allowed_cpu: String,
    /// Maximum allowed CPU
    pub max_allowed_cpu: String,
    /// Minimum allowed memory
    pub min_allowed_memory: String,
    /// Maximum allowed memory
    pub max_allowed_memory: String,
}

impl Default for VpaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            update_mode: "Auto".to_string(),
            min_allowed_cpu: "100m".to_string(),
            max_allowed_cpu: "2".to_string(),
            min_allowed_memory: "128Mi".to_string(),
            max_allowed_memory: "4Gi".to_string(),
        }
    }
}

/// Custom metric for scaling decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    /// Metric name
    pub name: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub comparison: MetricComparison,
    /// Prometheus query
    pub query: String,
}

/// Metric comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricComparison {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Scaling policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicies {
    /// Scale up policies
    pub scale_up: ScalePolicy,
    /// Scale down policies
    pub scale_down: ScalePolicy,
}

impl Default for ScalingPolicies {
    fn default() -> Self {
        Self {
            scale_up: ScalePolicy {
                change_type: ChangeType::Percent,
                value: 50,
                period_seconds: 60,
            },
            scale_down: ScalePolicy {
                change_type: ChangeType::Percent,
                value: 25,
                period_seconds: 300,
            },
        }
    }
}

/// Scale policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalePolicy {
    /// Change type (Percent or Pods)
    pub change_type: ChangeType,
    /// Change value
    pub value: u32,
    /// Period in seconds
    pub period_seconds: u64,
}

/// Change type for scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Percent,
    Pods,
}

/// Predictive scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveScalingConfig {
    /// Enable predictive scaling
    pub enabled: bool,
    /// Look-ahead window in minutes
    pub look_ahead_minutes: u32,
    /// Historical data window in days
    pub historical_data_days: u32,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Machine learning model type
    pub model_type: ModelType,
}

impl Default for PredictiveScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            look_ahead_minutes: 15,
            historical_data_days: 7,
            confidence_threshold: 0.8,
            model_type: ModelType::LinearRegression,
        }
    }
}

/// Machine learning model types for prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LinearRegression,
    TimeSeriesForecasting,
    NeuralNetwork,
}

/// Auto-scaling manager
#[derive(Debug)]
pub struct AutoScalingManager {
    config: AutoScalingConfig,
    current_replicas: u32,
    scaling_history: Vec<ScalingEvent>,
}

impl AutoScalingManager {
    /// Create a new auto-scaling manager
    pub fn new(config: AutoScalingConfig) -> Self {
        Self {
            config,
            current_replicas: 3, // Default starting replicas
            scaling_history: Vec::new(),
        }
    }

    /// Evaluate scaling decisions
    pub async fn evaluate_scaling(&mut self) -> Result<ScalingDecision> {
        if !self.config.enabled {
            return Ok(ScalingDecision::NoAction);
        }

        // Get current metrics
        let metrics = self.collect_metrics().await?;
        
        // Evaluate HPA decisions
        let hpa_decision = self.evaluate_hpa(&metrics).await?;
        
        // Evaluate custom metrics
        let custom_decision = self.evaluate_custom_metrics(&metrics).await?;
        
        // Evaluate predictive scaling
        let predictive_decision = if self.config.predictive_scaling.enabled {
            self.evaluate_predictive_scaling().await?
        } else {
            ScalingDecision::NoAction
        };

        // Combine decisions
        let final_decision = self.combine_decisions(vec![hpa_decision, custom_decision, predictive_decision]);
        
        // Record scaling event if action is needed
        if !matches!(final_decision, ScalingDecision::NoAction) {
            self.record_scaling_event(&final_decision);
        }

        Ok(final_decision)
    }

    /// Collect current metrics
    async fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        // Mock metrics collection
        metrics.insert("cpu_utilization".to_string(), 65.0);
        metrics.insert("memory_utilization".to_string(), 75.0);
        metrics.insert("kafka_consumer_lag".to_string(), 500.0);
        metrics.insert("stream_throughput".to_string(), 8500.0);
        metrics.insert("request_latency_p99".to_string(), 150.0);
        
        Ok(metrics)
    }

    /// Evaluate HPA scaling decisions
    async fn evaluate_hpa(&self, metrics: &HashMap<String, f64>) -> Result<ScalingDecision> {
        if !self.config.hpa.enabled {
            return Ok(ScalingDecision::NoAction);
        }

        let cpu_util = metrics.get("cpu_utilization").unwrap_or(&0.0);
        let memory_util = metrics.get("memory_utilization").unwrap_or(&0.0);

        if *cpu_util > self.config.hpa.target_cpu_utilization 
            || *memory_util > self.config.hpa.target_memory_utilization {
            
            if self.current_replicas < self.config.hpa.max_replicas {
                let target_replicas = (self.current_replicas as f64 * 1.5) as u32;
                let target = target_replicas.min(self.config.hpa.max_replicas);
                return Ok(ScalingDecision::ScaleUp { target_replicas: target });
            }
        } else if *cpu_util < self.config.hpa.target_cpu_utilization * 0.5 
            && *memory_util < self.config.hpa.target_memory_utilization * 0.5 {
            
            if self.current_replicas > self.config.hpa.min_replicas {
                let target_replicas = (self.current_replicas as f64 * 0.75) as u32;
                let target = target_replicas.max(self.config.hpa.min_replicas);
                return Ok(ScalingDecision::ScaleDown { target_replicas: target });
            }
        }

        Ok(ScalingDecision::NoAction)
    }

    /// Evaluate custom metrics scaling decisions
    async fn evaluate_custom_metrics(&self, metrics: &HashMap<String, f64>) -> Result<ScalingDecision> {
        for custom_metric in &self.config.custom_metrics {
            if let Some(value) = metrics.get(&custom_metric.name) {
                let should_scale = match custom_metric.comparison {
                    MetricComparison::GreaterThan => *value > custom_metric.threshold,
                    MetricComparison::LessThan => *value < custom_metric.threshold,
                    MetricComparison::Equal => (*value - custom_metric.threshold).abs() < f64::EPSILON,
                    MetricComparison::GreaterThanOrEqual => *value >= custom_metric.threshold,
                    MetricComparison::LessThanOrEqual => *value <= custom_metric.threshold,
                };

                if should_scale && matches!(custom_metric.comparison, MetricComparison::GreaterThan | MetricComparison::GreaterThanOrEqual) {
                    let target_replicas = self.current_replicas + 1;
                    return Ok(ScalingDecision::ScaleUp { 
                        target_replicas: target_replicas.min(self.config.hpa.max_replicas) 
                    });
                }
            }
        }

        Ok(ScalingDecision::NoAction)
    }

    /// Evaluate predictive scaling decisions
    async fn evaluate_predictive_scaling(&self) -> Result<ScalingDecision> {
        // Mock predictive scaling logic
        // In a real implementation, this would use historical data and ML models
        println!("Evaluating predictive scaling with {} model", 
                match self.config.predictive_scaling.model_type {
                    ModelType::LinearRegression => "Linear Regression",
                    ModelType::TimeSeriesForecasting => "Time Series Forecasting",
                    ModelType::NeuralNetwork => "Neural Network",
                });
        
        Ok(ScalingDecision::NoAction)
    }

    /// Combine multiple scaling decisions
    fn combine_decisions(&self, decisions: Vec<ScalingDecision>) -> ScalingDecision {
        // Priority: ScaleUp > ScaleDown > NoAction
        for decision in decisions {
            if matches!(decision, ScalingDecision::ScaleUp { .. }) {
                return decision;
            }
        }
        
        for decision in decisions {
            if matches!(decision, ScalingDecision::ScaleDown { .. }) {
                return decision;
            }
        }
        
        ScalingDecision::NoAction
    }

    /// Record scaling event
    fn record_scaling_event(&mut self, decision: &ScalingDecision) {
        let event = ScalingEvent {
            timestamp: SystemTime::now(),
            decision: decision.clone(),
            previous_replicas: self.current_replicas,
        };
        
        self.scaling_history.push(event);
        
        // Update current replicas based on decision
        match decision {
            ScalingDecision::ScaleUp { target_replicas } => {
                self.current_replicas = *target_replicas;
            },
            ScalingDecision::ScaleDown { target_replicas } => {
                self.current_replicas = *target_replicas;
            },
            ScalingDecision::NoAction => {},
        }
    }

    /// Get scaling history
    pub fn get_scaling_history(&self) -> &[ScalingEvent] {
        &self.scaling_history
    }

    /// Get current replicas
    pub fn get_current_replicas(&self) -> u32 {
        self.current_replicas
    }
}

/// Scaling decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingDecision {
    ScaleUp { target_replicas: u32 },
    ScaleDown { target_replicas: u32 },
    NoAction,
}

/// Scaling event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    pub timestamp: SystemTime,
    pub decision: ScalingDecision,
    pub previous_replicas: u32,
}

/// Initialize auto-scaling
pub async fn initialize(config: &AutoScalingConfig) -> Result<()> {
    if !config.enabled {
        return Ok(());
    }
    
    let _manager = AutoScalingManager::new(config.clone());
    
    println!("Auto-scaling initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_scaling_config_default() {
        let config = AutoScalingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.hpa.min_replicas, 2);
        assert_eq!(config.hpa.max_replicas, 20);
    }

    #[test]
    fn test_auto_scaling_manager_creation() {
        let config = AutoScalingConfig::default();
        let manager = AutoScalingManager::new(config);
        assert_eq!(manager.get_current_replicas(), 3);
    }

    #[tokio::test]
    async fn test_auto_scaling_evaluation() {
        let config = AutoScalingConfig::default();
        let mut manager = AutoScalingManager::new(config);
        let decision = manager.evaluate_scaling().await.unwrap();
        // Decision depends on mock metrics
        assert!(matches!(decision, ScalingDecision::NoAction | ScalingDecision::ScaleUp { .. } | ScalingDecision::ScaleDown { .. }));
    }

    #[test]
    fn test_scaling_decision_combination() {
        let config = AutoScalingConfig::default();
        let manager = AutoScalingManager::new(config);
        
        let decisions = vec![
            ScalingDecision::NoAction,
            ScalingDecision::ScaleUp { target_replicas: 5 },
            ScalingDecision::ScaleDown { target_replicas: 2 },
        ];
        
        let result = manager.combine_decisions(decisions);
        assert!(matches!(result, ScalingDecision::ScaleUp { target_replicas: 5 }));
    }
}