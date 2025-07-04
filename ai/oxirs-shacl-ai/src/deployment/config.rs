//! Deployment Configuration Types and Settings
//!
//! This module contains all configuration types for deployment strategies,
//! resource management, monitoring, security, and environment settings.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for deployment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Enable containerization
    pub enable_containerization: bool,

    /// Enable auto-scaling
    pub enable_auto_scaling: bool,

    /// Enable load balancing
    pub enable_load_balancing: bool,

    /// Enable health monitoring
    pub enable_health_monitoring: bool,

    /// Deployment strategy type
    pub deployment_strategy: DeploymentStrategy,

    /// Environment configuration
    pub environment: EnvironmentType,

    /// Resource limits
    pub resource_limits: ResourceLimits,

    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,

    /// Monitoring configuration
    pub monitoring: MonitoringConfig,

    /// Update strategy
    pub update_strategy: UpdateStrategy,

    /// Security configuration
    pub security: SecurityConfig,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            enable_containerization: true,
            enable_auto_scaling: true,
            enable_load_balancing: true,
            enable_health_monitoring: true,
            deployment_strategy: DeploymentStrategy::BlueGreen,
            environment: EnvironmentType::Production,
            resource_limits: ResourceLimits::default(),
            auto_scaling: AutoScalingConfig::default(),
            monitoring: MonitoringConfig::default(),
            update_strategy: UpdateStrategy::RollingUpdate,
            security: SecurityConfig::default(),
        }
    }
}

/// Deployment strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    BlueGreen,
    RollingUpdate,
    Canary,
    Recreation,
    CustomStrategy(String),
}

/// Environment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentType {
    Development,
    Testing,
    Staging,
    Production,
    DisasterRecovery,
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_limit: f64,
    pub memory_limit_mb: usize,
    pub disk_limit_gb: usize,
    pub network_bandwidth_mbps: f64,
    pub max_concurrent_validations: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu_limit: 4.0,
            memory_limit_mb: 8192,
            disk_limit_gb: 100,
            network_bandwidth_mbps: 1000.0,
            max_concurrent_validations: 1000,
        }
    }
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    pub min_instances: u32,
    pub max_instances: u32,
    pub target_cpu_utilization: f64,
    pub target_memory_utilization: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_up_cooldown: Duration,
    pub scale_down_cooldown: Duration,
    pub metrics_window: Duration,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            min_instances: 2,
            max_instances: 10,
            target_cpu_utilization: 0.7,
            target_memory_utilization: 0.8,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            scale_up_cooldown: Duration::from_secs(300),
            scale_down_cooldown: Duration::from_secs(600),
            metrics_window: Duration::from_secs(300),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics_collection: bool,
    pub enable_distributed_tracing: bool,
    pub enable_log_aggregation: bool,
    pub enable_alerting: bool,
    pub metrics_retention_days: u32,
    pub log_retention_days: u32,
    pub health_check_interval: Duration,
    pub alert_thresholds: AlertThresholds,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics_collection: true,
            enable_distributed_tracing: true,
            enable_log_aggregation: true,
            enable_alerting: true,
            metrics_retention_days: 30,
            log_retention_days: 7,
            health_check_interval: Duration::from_secs(30),
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_utilization_warning: f64,
    pub cpu_utilization_critical: f64,
    pub memory_utilization_warning: f64,
    pub memory_utilization_critical: f64,
    pub error_rate_warning: f64,
    pub error_rate_critical: f64,
    pub response_time_warning_ms: f64,
    pub response_time_critical_ms: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_utilization_warning: 0.8,
            cpu_utilization_critical: 0.9,
            memory_utilization_warning: 0.85,
            memory_utilization_critical: 0.95,
            error_rate_warning: 0.05,
            error_rate_critical: 0.1,
            response_time_warning_ms: 1000.0,
            response_time_critical_ms: 2000.0,
        }
    }
}

/// Update strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateStrategy {
    RollingUpdate,
    BlueGreenUpdate,
    CanaryUpdate,
    ImmediateUpdate,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_tls: bool,
    pub enable_authentication: bool,
    pub enable_authorization: bool,
    pub enable_network_policies: bool,
    pub certificate_auto_renewal: bool,
    pub secret_management: SecretManagementConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_tls: true,
            enable_authentication: true,
            enable_authorization: true,
            enable_network_policies: true,
            certificate_auto_renewal: true,
            secret_management: SecretManagementConfig::default(),
        }
    }
}

/// Secret management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretManagementConfig {
    pub provider: SecretProvider,
    pub auto_rotation: bool,
    pub encryption_at_rest: bool,
    pub encryption_in_transit: bool,
}

impl Default for SecretManagementConfig {
    fn default() -> Self {
        Self {
            provider: SecretProvider::Kubernetes,
            auto_rotation: true,
            encryption_at_rest: true,
            encryption_in_transit: true,
        }
    }
}

/// Secret providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecretProvider {
    Kubernetes,
    HashiCorpVault,
    AWSSecretsManager,
    AzureKeyVault,
    GoogleSecretManager,
}
