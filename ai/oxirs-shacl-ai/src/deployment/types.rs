//! Shared Types and Data Structures
//!
//! This module contains shared types used across all deployment modules,
//! including result types, specifications, and common data structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::config::{DeploymentStrategy, EnvironmentType, UpdateStrategy};
use super::containerization::Protocol;

/// Deployment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentSpec {
    pub name: String,
    pub version: String,
    pub environment: EnvironmentType,
    pub resources: ResourceRequirements,
    pub replicas: u32,
    pub configuration: HashMap<String, String>,
    pub volumes: Vec<VolumeSpec>,
    pub networking: NetworkingSpec,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu: Option<String>,
    pub memory: Option<String>,
}

/// Volume specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeSpec {
    pub name: String,
    pub volume_type: VolumeType,
    pub size: String,
    pub mount_path: String,
}

/// Volume types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolumeType {
    EmptyDir,
    PersistentVolume,
    ConfigMap,
    Secret,
    HostPath,
}

/// Networking specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkingSpec {
    pub service_type: ServiceType,
    pub ports: Vec<ServicePort>,
    pub ingress: Option<IngressSpec>,
}

/// Service types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceType {
    ClusterIP,
    NodePort,
    LoadBalancer,
    ExternalName,
}

/// Service port
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePort {
    pub name: String,
    pub port: u16,
    pub target_port: u16,
    pub protocol: Protocol,
}

/// Ingress specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressSpec {
    pub host: String,
    pub path: String,
    pub tls_enabled: bool,
    pub annotations: HashMap<String, String>,
}

/// Deployment result
#[derive(Debug, Clone)]
pub struct DeploymentResult {
    pub deployment_id: String,
    pub status: DeploymentStatus,
    pub deployment_time: Duration,
    pub image_info: Option<ImageInfo>,
    pub orchestration_result: OrchestrationResult,
    pub deployment_info: DeploymentInfo,
    pub endpoints: Vec<ServiceEndpoint>,
    pub monitoring_urls: Vec<MonitoringUrl>,
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    InProgress,
    Successful,
    Failed,
    RolledBack,
    Cancelled,
}

/// Image information
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub image_tag: String,
    pub image_size: u64,
    pub build_time: Duration,
    pub vulnerabilities: Vec<String>,
}

/// Orchestration result
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    pub cluster_name: String,
    pub namespace: String,
    pub node_count: u32,
    pub setup_time: Duration,
}

/// Deployment information
#[derive(Debug, Clone)]
pub struct DeploymentInfo {
    pub deployment_id: String,
    pub namespace: String,
    pub services: Vec<String>,
    pub pods: Vec<String>,
    pub replicas: u32,
}

/// Service endpoint
#[derive(Debug, Clone)]
pub struct ServiceEndpoint {
    pub service_name: String,
    pub endpoint_url: String,
    pub port: u16,
    pub protocol: String,
}

/// Monitoring URL
#[derive(Debug, Clone)]
pub struct MonitoringUrl {
    pub service_name: String,
    pub url: String,
}

/// Scaling request
#[derive(Debug, Clone)]
pub struct ScalingRequest {
    pub target_service: String,
    pub scaling_type: ScalingType,
    pub target_replicas: Option<u32>,
    pub resource_adjustment: Option<ResourceRequirements>,
    pub auto_triggered: bool,
    pub reason: String,
}

/// Scaling types
#[derive(Debug, Clone)]
pub enum ScalingType {
    HorizontalUp,
    HorizontalDown,
    VerticalUp,
    VerticalDown,
}

/// Scaling result
#[derive(Debug, Clone)]
pub struct ScalingResult {
    pub success: bool,
    pub previous_replicas: u32,
    pub new_replicas: u32,
    pub scaling_time: Duration,
    pub resource_changes: Option<ResourceRequirements>,
}

/// Update specification
#[derive(Debug, Clone)]
pub struct UpdateSpec {
    pub target_version: String,
    pub update_strategy: UpdateStrategy,
    pub rollback_on_failure: bool,
    pub health_check_timeout: Duration,
}

/// Update result
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub success: bool,
    pub previous_version: String,
    pub new_version: String,
    pub update_time: Duration,
    pub rollback_performed: bool,
}

/// Deployment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRecord {
    pub deployment_id: String,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub strategy: DeploymentStrategy,
    pub status: DeploymentStatus,
    pub rollback_info: Option<RollbackInfo>,
}

/// Rollback information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackInfo {
    pub rollback_reason: String,
    pub rollback_timestamp: chrono::DateTime<chrono::Utc>,
    pub previous_version: String,
}

/// Deployment statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeploymentStatistics {
    pub total_deployments: usize,
    pub successful_deployments: usize,
    pub failed_deployments: usize,
    pub rollbacks_performed: usize,
    pub average_deployment_time: Duration,
    pub uptime_percentage: f64,
    pub scaling_events: usize,
    pub auto_scaling_triggered: usize,
}
