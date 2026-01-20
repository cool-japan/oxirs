//! Cloud-Native Configuration Types

use serde::{Deserialize, Serialize};
use super::kubernetes::KubernetesConfig;
use super::service_mesh::ServiceMeshConfig;
use super::auto_scaling::AutoScalingConfig;
use super::observability::ObservabilityConfig;
use super::multi_cloud::MultiCloudConfig;
use super::gitops::GitOpsConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudNativeConfig {
    /// Kubernetes configuration
    pub kubernetes: KubernetesConfig,
    /// Service mesh configuration
    pub service_mesh: ServiceMeshConfig,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Observability configuration
    pub observability: ObservabilityConfig,
    /// Multi-cloud configuration
    pub multi_cloud: MultiCloudConfig,
    /// GitOps configuration
    pub gitops: GitOpsConfig,
}

impl Default for CloudNativeConfig {
    fn default() -> Self {
        Self {
            kubernetes: KubernetesConfig::default(),
            service_mesh: ServiceMeshConfig::default(),
            auto_scaling: AutoScalingConfig::default(),
            observability: ObservabilityConfig::default(),
            multi_cloud: MultiCloudConfig::default(),
            gitops: GitOpsConfig::default(),
        }
    }
}
