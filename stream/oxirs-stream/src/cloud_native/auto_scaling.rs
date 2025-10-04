//! Auto-Scaling Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Horizontal Pod Autoscaler
    pub hpa: HPAConfig,
    /// Vertical Pod Autoscaler
    pub vpa: VPAConfig,
    /// Cluster autoscaler
    pub cluster_autoscaler: ClusterAutoscalerConfig,
    /// Custom metrics for scaling
    pub custom_metrics: Vec<CustomScalingMetric>,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hpa: HPAConfig::default(),
            vpa: VPAConfig::default(),
            cluster_autoscaler: ClusterAutoscalerConfig::default(),
            custom_metrics: vec![
                CustomScalingMetric {
                    name: "stream_events_per_second".to_string(),
                    target_value: 1000.0,
                    metric_type: ScalingMetricType::Value,
                },
                CustomScalingMetric {
                    name: "memory_utilization".to_string(),
                    target_value: 70.0,
                    metric_type: ScalingMetricType::Utilization,
                },
            ],
        }
    }
}

/// Horizontal Pod Autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HPAConfig {
    /// Enable HPA
    pub enabled: bool,
    /// Minimum replicas
    pub min_replicas: u32,
    /// Maximum replicas
    pub max_replicas: u32,
    /// Target CPU utilization
    pub target_cpu_utilization: f64,
    /// Target memory utilization
    pub target_memory_utilization: f64,
    /// Scale down stabilization window
    pub scale_down_stabilization_window: Duration,
    /// Scale up stabilization window
    pub scale_up_stabilization_window: Duration,
}

impl Default for HPAConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_replicas: 2,
            max_replicas: 100,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            scale_down_stabilization_window: Duration::from_secs(300),
            scale_up_stabilization_window: Duration::from_secs(60),
        }
    }
}

/// Vertical Pod Autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAConfig {
    /// Enable VPA
    pub enabled: bool,
    /// Update mode
    pub update_mode: VPAUpdateMode,
    /// Resource policy
    pub resource_policy: VPAResourcePolicy,
}

impl Default for VPAConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default due to potential conflicts with HPA
            update_mode: VPAUpdateMode::Auto,
            resource_policy: VPAResourcePolicy::default(),
        }
    }
}

/// VPA update modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VPAUpdateMode {
    Off,
    Initial,
    Auto,
}

/// VPA resource policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPAResourcePolicy {
    /// Minimum allowed resources
    pub min_allowed: ResourceRequirements,
    /// Maximum allowed resources
    pub max_allowed: ResourceRequirements,
}

impl Default for VPAResourcePolicy {
    fn default() -> Self {
        Self {
            min_allowed: ResourceRequirements {
                cpu: "100m".to_string(),
                memory: "128Mi".to_string(),
            },
            max_allowed: ResourceRequirements {
                cpu: "4".to_string(),
                memory: "8Gi".to_string(),
            },
        }
    }
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu: String,
    pub memory: String,
}

/// Cluster autoscaler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAutoscalerConfig {
    /// Enable cluster autoscaler
    pub enabled: bool,
    /// Minimum nodes
    pub min_nodes: u32,
    /// Maximum nodes
    pub max_nodes: u32,
    /// Scale down delay after add
    pub scale_down_delay_after_add: Duration,
    /// Scale down unneeded time
    pub scale_down_unneeded_time: Duration,
}

impl Default for ClusterAutoscalerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_nodes: 3,
            max_nodes: 100,
            scale_down_delay_after_add: Duration::from_secs(600),
            scale_down_unneeded_time: Duration::from_secs(600),
        }
    }
}

/// Custom scaling metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScalingMetric {
    pub name: String,
    pub target_value: f64,
    pub metric_type: ScalingMetricType,
}

/// Scaling metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingMetricType {
    Value,
    AverageValue,
    Utilization,
}
