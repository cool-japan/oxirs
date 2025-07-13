//! # Cloud-Native Integration Module
//!
//! Comprehensive Kubernetes and service mesh integration for OxiRS Stream,
//! providing enterprise-grade cloud-native deployment, scaling, and management capabilities.
//!
//! This module provides:
//! - Kubernetes Custom Resource Definitions (CRDs) and Operators
//! - Service mesh integration (Istio, Linkerd, Consul Connect)
//! - Auto-scaling with custom metrics
//! - Health checks and observability
//! - Multi-cloud deployment strategies
//! - GitOps integration and CI/CD pipelines

pub mod kubernetes;
pub mod service_mesh;
pub mod auto_scaling;
pub mod observability;
pub mod multi_cloud;
pub mod gitops;

use anyhow::Result;
use serde::{Deserialize, Serialize};

// Re-export commonly used types
pub use kubernetes::KubernetesConfig;
pub use service_mesh::ServiceMeshConfig;
pub use auto_scaling::AutoScalingConfig;
pub use observability::ObservabilityConfig;
pub use multi_cloud::MultiCloudConfig;
pub use gitops::GitOpsConfig;

/// Cloud-native configuration
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

/// Cloud-native manager for coordinating all cloud-native functionality
#[derive(Debug)]
pub struct CloudNativeManager {
    config: CloudNativeConfig,
}

impl CloudNativeManager {
    /// Create a new cloud-native manager
    pub fn new(config: CloudNativeConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &CloudNativeConfig {
        &self.config
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: CloudNativeConfig) {
        self.config = config;
    }

    /// Initialize cloud-native components
    pub async fn initialize(&self) -> Result<()> {
        // Initialize each component
        if self.config.kubernetes.enabled {
            kubernetes::initialize(&self.config.kubernetes).await?;
        }
        
        if self.config.service_mesh.enabled {
            service_mesh::initialize(&self.config.service_mesh).await?;
        }
        
        if self.config.auto_scaling.enabled {
            auto_scaling::initialize(&self.config.auto_scaling).await?;
        }
        
        if self.config.observability.enabled {
            observability::initialize(&self.config.observability).await?;
        }
        
        if self.config.multi_cloud.enabled {
            multi_cloud::initialize(&self.config.multi_cloud).await?;
        }
        
        if self.config.gitops.enabled {
            gitops::initialize(&self.config.gitops).await?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_native_config_default() {
        let config = CloudNativeConfig::default();
        assert!(config.kubernetes.enabled);
        assert_eq!(config.kubernetes.namespace, "oxirs");
    }

    #[test]
    fn test_cloud_native_manager_creation() {
        let config = CloudNativeConfig::default();
        let manager = CloudNativeManager::new(config);
        assert_eq!(manager.config().kubernetes.namespace, "oxirs");
    }

    #[tokio::test]
    async fn test_cloud_native_manager_initialization() {
        let config = CloudNativeConfig::default();
        let manager = CloudNativeManager::new(config);
        // Note: This would fail in a real test without proper setup
        // but we're testing the structure
        assert!(manager.initialize().await.is_ok());
    }
}