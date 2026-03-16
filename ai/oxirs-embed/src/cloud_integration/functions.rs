//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::{
        AWSSageMakerService, CloudIntegrationConfig, CloudIntegrationManager, CloudService,
        DeploymentConfig, DeploymentStatus, NetworkingConfig, ResourceRequirements,
    };
    use std::collections::HashMap;
    #[tokio::test]
    async fn test_cloud_integration_manager_creation() {
        let config = CloudIntegrationConfig::default();
        let manager = CloudIntegrationManager::new(config);
        let providers = manager.providers.read().await;
        assert!(providers.is_empty());
    }
    #[tokio::test]
    async fn test_aws_sagemaker_service() {
        let service = AWSSageMakerService::new(
            "us-east-1".to_string(),
            "test_key".to_string(),
            "test_secret".to_string(),
            None,
        );
        let config = DeploymentConfig {
            model_name: "test-model".to_string(),
            model_version: "1.0".to_string(),
            instance_type: "ml.m5.large".to_string(),
            initial_instance_count: 1,
            auto_scaling_enabled: true,
            environment_variables: HashMap::new(),
            resource_requirements: ResourceRequirements {
                cpu_cores: 2.0,
                memory_gb: 8.0,
                gpu_count: 0,
                storage_gb: 50,
            },
            networking: NetworkingConfig {
                vpc_config: None,
                enable_network_isolation: false,
                custom_security_groups: vec![],
            },
            data_capture: None,
        };
        let result = service.deploy_model(&config).await.expect("should succeed");
        assert!(matches!(result.status, DeploymentStatus::Creating));
        assert!(result.cost_estimate.is_some());
    }
    #[tokio::test]
    async fn test_cost_estimation() {
        let service = AWSSageMakerService::new(
            "us-east-1".to_string(),
            "test_key".to_string(),
            "test_secret".to_string(),
            None,
        );
        let config = DeploymentConfig {
            model_name: "test-model".to_string(),
            model_version: "1.0".to_string(),
            instance_type: "ml.m5.large".to_string(),
            initial_instance_count: 2,
            auto_scaling_enabled: true,
            environment_variables: HashMap::new(),
            resource_requirements: ResourceRequirements {
                cpu_cores: 2.0,
                memory_gb: 8.0,
                gpu_count: 0,
                storage_gb: 50,
            },
            networking: NetworkingConfig {
                vpc_config: None,
                enable_network_isolation: false,
                custom_security_groups: vec![],
            },
            data_capture: None,
        };
        let estimate = service
            .estimate_costs(&config, 24)
            .await
            .expect("should succeed");
        assert!(estimate.hourly_cost_usd > 0.0);
        assert!(estimate.estimated_monthly_cost_usd > 0.0);
    }
}
