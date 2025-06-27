//! Model Registry and Versioning System
//!
//! This module provides a comprehensive model lifecycle management system including
//! versioning, deployment, performance tracking, and A/B testing capabilities.

use crate::{EmbeddingModel, ModelConfig, ModelStats};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version_id: Uuid,
    pub model_id: Uuid,
    pub version_number: String,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub description: String,
    pub tags: Vec<String>,
    pub metrics: HashMap<String, f64>,
    pub config: ModelConfig,
    pub is_production: bool,
    pub is_deprecated: bool,
}

/// Model deployment status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DeploymentStatus {
    NotDeployed,
    Deploying,
    Deployed,
    Failed,
    Retiring,
    Retired,
}

/// Model deployment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeployment {
    pub deployment_id: Uuid,
    pub version_id: Uuid,
    pub status: DeploymentStatus,
    pub deployed_at: Option<DateTime<Utc>>,
    pub endpoint: Option<String>,
    pub resource_allocation: ResourceAllocation,
    pub health_check_url: Option<String>,
    pub rollback_version: Option<Uuid>,
}

/// Resource allocation for model deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_cores: f32,
    pub memory_gb: f32,
    pub gpu_count: u32,
    pub gpu_memory_gb: f32,
    pub max_concurrent_requests: usize,
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            cpu_cores: 2.0,
            memory_gb: 4.0,
            gpu_count: 0,
            gpu_memory_gb: 0.0,
            max_concurrent_requests: 100,
        }
    }
}

/// A/B test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    pub test_id: Uuid,
    pub name: String,
    pub description: String,
    pub version_a: Uuid,
    pub version_b: Uuid,
    pub traffic_split: f32, // Percentage going to version B (0.0 - 1.0)
    pub started_at: DateTime<Utc>,
    pub ends_at: Option<DateTime<Utc>>,
    pub metrics_to_track: Vec<String>,
    pub is_active: bool,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub throughput_qps: f64,
    pub error_rate: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: Option<f64>,
    pub cache_hit_rate: f64,
}

/// Model registry for managing model lifecycle
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<Uuid, ModelMetadata>>>,
    versions: Arc<RwLock<HashMap<Uuid, ModelVersion>>>,
    deployments: Arc<RwLock<HashMap<Uuid, ModelDeployment>>>,
    ab_tests: Arc<RwLock<HashMap<Uuid, ABTestConfig>>>,
    performance_history: Arc<RwLock<HashMap<Uuid, Vec<PerformanceMetrics>>>>,
    storage_path: PathBuf,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: Uuid,
    pub name: String,
    pub model_type: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub owner: String,
    pub description: String,
    pub versions: Vec<Uuid>,
    pub production_version: Option<Uuid>,
    pub staging_version: Option<Uuid>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(storage_path: PathBuf) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
            deployments: Arc::new(RwLock::new(HashMap::new())),
            ab_tests: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            storage_path,
        }
    }

    /// Register a new model
    pub async fn register_model(
        &self,
        name: String,
        model_type: String,
        owner: String,
        description: String,
    ) -> Result<Uuid> {
        let model_id = Uuid::new_v4();
        let metadata = ModelMetadata {
            model_id,
            name,
            model_type,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            owner,
            description,
            versions: Vec::new(),
            production_version: None,
            staging_version: None,
        };

        self.models.write().await.insert(model_id, metadata);
        Ok(model_id)
    }

    /// Register a new model version
    pub async fn register_version(
        &self,
        model_id: Uuid,
        version_number: String,
        created_by: String,
        description: String,
        config: ModelConfig,
        metrics: HashMap<String, f64>,
    ) -> Result<Uuid> {
        let version_id = Uuid::new_v4();
        
        // Verify model exists
        let mut models = self.models.write().await;
        let model = models.get_mut(&model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;

        let version = ModelVersion {
            version_id,
            model_id,
            version_number,
            created_at: Utc::now(),
            created_by,
            description,
            tags: Vec::new(),
            metrics,
            config,
            is_production: false,
            is_deprecated: false,
        };

        model.versions.push(version_id);
        model.updated_at = Utc::now();
        
        self.versions.write().await.insert(version_id, version);
        Ok(version_id)
    }

    /// Deploy a model version
    pub async fn deploy_version(
        &self,
        version_id: Uuid,
        resource_allocation: ResourceAllocation,
    ) -> Result<Uuid> {
        // Verify version exists
        if !self.versions.read().await.contains_key(&version_id) {
            return Err(anyhow!("Version not found: {}", version_id));
        }

        let deployment_id = Uuid::new_v4();
        let deployment = ModelDeployment {
            deployment_id,
            version_id,
            status: DeploymentStatus::Deploying,
            deployed_at: None,
            endpoint: None,
            resource_allocation,
            health_check_url: None,
            rollback_version: None,
        };

        self.deployments.write().await.insert(deployment_id, deployment);
        
        // Start deployment process (in real implementation)
        self.start_deployment(deployment_id).await?;
        
        Ok(deployment_id)
    }

    /// Start deployment process
    async fn start_deployment(&self, deployment_id: Uuid) -> Result<()> {
        // In a real implementation, this would:
        // 1. Allocate resources
        // 2. Load model weights
        // 3. Start serving infrastructure
        // 4. Configure load balancer
        // 5. Run health checks
        
        // For now, simulate deployment
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        let mut deployments = self.deployments.write().await;
        if let Some(deployment) = deployments.get_mut(&deployment_id) {
            deployment.status = DeploymentStatus::Deployed;
            deployment.deployed_at = Some(Utc::now());
            deployment.endpoint = Some(format!("https://api.oxirs.ai/v1/embed/{}", deployment_id));
            deployment.health_check_url = Some(format!("https://api.oxirs.ai/v1/embed/{}/health", deployment_id));
        }
        
        Ok(())
    }

    /// Promote version to production
    pub async fn promote_to_production(&self, version_id: Uuid) -> Result<()> {
        let versions = self.versions.read().await;
        let version = versions.get(&version_id)
            .ok_or_else(|| anyhow!("Version not found: {}", version_id))?;
        
        let model_id = version.model_id;
        drop(versions);
        
        let mut models = self.models.write().await;
        let model = models.get_mut(&model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;
        
        // Mark previous production version as non-production
        if let Some(prev_prod) = model.production_version {
            let mut versions = self.versions.write().await;
            if let Some(prev_version) = versions.get_mut(&prev_prod) {
                prev_version.is_production = false;
            }
        }
        
        model.production_version = Some(version_id);
        model.updated_at = Utc::now();
        
        let mut versions = self.versions.write().await;
        if let Some(version) = versions.get_mut(&version_id) {
            version.is_production = true;
        }
        
        Ok(())
    }

    /// Create A/B test
    pub async fn create_ab_test(
        &self,
        name: String,
        description: String,
        version_a: Uuid,
        version_b: Uuid,
        traffic_split: f32,
        duration_hours: Option<u32>,
    ) -> Result<Uuid> {
        // Verify both versions exist
        let versions = self.versions.read().await;
        if !versions.contains_key(&version_a) {
            return Err(anyhow!("Version A not found: {}", version_a));
        }
        if !versions.contains_key(&version_b) {
            return Err(anyhow!("Version B not found: {}", version_b));
        }
        drop(versions);
        
        if traffic_split < 0.0 || traffic_split > 1.0 {
            return Err(anyhow!("Traffic split must be between 0.0 and 1.0"));
        }
        
        let test_id = Uuid::new_v4();
        let ab_test = ABTestConfig {
            test_id,
            name,
            description,
            version_a,
            version_b,
            traffic_split,
            started_at: Utc::now(),
            ends_at: duration_hours.map(|h| Utc::now() + chrono::Duration::hours(h as i64)),
            metrics_to_track: vec![
                "latency_p95".to_string(),
                "accuracy".to_string(),
                "error_rate".to_string(),
            ],
            is_active: true,
        };
        
        self.ab_tests.write().await.insert(test_id, ab_test);
        Ok(test_id)
    }

    /// Record performance metrics
    pub async fn record_performance(
        &self,
        version_id: Uuid,
        metrics: PerformanceMetrics,
    ) -> Result<()> {
        let mut history = self.performance_history.write().await;
        history.entry(version_id)
            .or_insert_with(Vec::new)
            .push(metrics);
        
        // Keep only last 1000 metrics per version
        if let Some(vec) = history.get_mut(&version_id) {
            if vec.len() > 1000 {
                vec.drain(0..vec.len() - 1000);
            }
        }
        
        Ok(())
    }

    /// Get model metadata
    pub async fn get_model(&self, model_id: Uuid) -> Result<ModelMetadata> {
        self.models.read().await
            .get(&model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))
    }

    /// Get version info
    pub async fn get_version(&self, version_id: Uuid) -> Result<ModelVersion> {
        self.versions.read().await
            .get(&version_id)
            .cloned()
            .ok_or_else(|| anyhow!("Version not found: {}", version_id))
    }

    /// Get deployment info
    pub async fn get_deployment(&self, deployment_id: Uuid) -> Result<ModelDeployment> {
        self.deployments.read().await
            .get(&deployment_id)
            .cloned()
            .ok_or_else(|| anyhow!("Deployment not found: {}", deployment_id))
    }

    /// Get performance history
    pub async fn get_performance_history(
        &self,
        version_id: Uuid,
        limit: Option<usize>,
    ) -> Result<Vec<PerformanceMetrics>> {
        let history = self.performance_history.read().await;
        let metrics = history.get(&version_id)
            .ok_or_else(|| anyhow!("No performance history for version: {}", version_id))?;
        
        let limit = limit.unwrap_or(100);
        let start = metrics.len().saturating_sub(limit);
        
        Ok(metrics[start..].to_vec())
    }

    /// Rollback deployment
    pub async fn rollback_deployment(&self, deployment_id: Uuid) -> Result<()> {
        let deployments = self.deployments.read().await;
        let deployment = deployments.get(&deployment_id)
            .ok_or_else(|| anyhow!("Deployment not found: {}", deployment_id))?;
        
        if let Some(rollback_version) = deployment.rollback_version {
            drop(deployments);
            
            // Deploy the rollback version
            self.deploy_version(rollback_version, deployment.resource_allocation.clone()).await?;
            
            // Mark current deployment as retired
            let mut deployments = self.deployments.write().await;
            if let Some(deployment) = deployments.get_mut(&deployment_id) {
                deployment.status = DeploymentStatus::Retired;
            }
        } else {
            return Err(anyhow!("No rollback version configured"));
        }
        
        Ok(())
    }

    /// List all models
    pub async fn list_models(&self) -> Vec<ModelMetadata> {
        self.models.read().await.values().cloned().collect()
    }

    /// List versions for a model
    pub async fn list_versions(&self, model_id: Uuid) -> Result<Vec<ModelVersion>> {
        let models = self.models.read().await;
        let model = models.get(&model_id)
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;
        
        let version_ids = model.versions.clone();
        drop(models);
        
        let versions = self.versions.read().await;
        let mut result = Vec::new();
        
        for version_id in version_ids {
            if let Some(version) = versions.get(&version_id) {
                result.push(version.clone());
            }
        }
        
        Ok(result)
    }

    /// Get active A/B tests
    pub async fn get_active_ab_tests(&self) -> Vec<ABTestConfig> {
        self.ab_tests.read().await
            .values()
            .filter(|test| test.is_active)
            .cloned()
            .collect()
    }

    /// End A/B test
    pub async fn end_ab_test(&self, test_id: Uuid) -> Result<()> {
        let mut ab_tests = self.ab_tests.write().await;
        let test = ab_tests.get_mut(&test_id)
            .ok_or_else(|| anyhow!("A/B test not found: {}", test_id))?;
        
        test.is_active = false;
        test.ends_at = Some(Utc::now());
        
        Ok(())
    }
}

/// Model serving infrastructure
pub struct ModelServer {
    registry: Arc<ModelRegistry>,
    loaded_models: Arc<RwLock<HashMap<Uuid, Box<dyn EmbeddingModel>>>>,
    warm_up_cache: Arc<RwLock<HashMap<Uuid, Vec<String>>>>,
}

impl ModelServer {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self {
            registry,
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            warm_up_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load model into memory
    pub async fn load_model(&self, version_id: Uuid) -> Result<()> {
        // In real implementation, this would load the actual model
        // For now, we just mark it as loaded
        Ok(())
    }

    /// Warm up model with sample inputs
    pub async fn warm_up_model(&self, version_id: Uuid, samples: Vec<String>) -> Result<()> {
        self.warm_up_cache.write().await.insert(version_id, samples);
        
        // In real implementation, run inference on samples to warm up caches
        Ok(())
    }

    /// Get model for inference
    pub async fn get_model(&self, version_id: Uuid) -> Result<Arc<Box<dyn EmbeddingModel>>> {
        // In real implementation, return loaded model
        Err(anyhow!("Model loading not implemented"))
    }

    /// Route request based on A/B test
    pub async fn route_request(&self, test_id: Uuid) -> Result<Uuid> {
        let ab_tests = self.registry.ab_tests.read().await;
        let test = ab_tests.get(&test_id)
            .ok_or_else(|| anyhow!("A/B test not found: {}", test_id))?;
        
        // Simple random routing based on traffic split
        let random = rand::random::<f32>();
        Ok(if random < test.traffic_split {
            test.version_b
        } else {
            test.version_a
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_registry_lifecycle() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path().to_path_buf());
        
        // Register model
        let model_id = registry.register_model(
            "test-model".to_string(),
            "TransformerEmbedding".to_string(),
            "test-user".to_string(),
            "Test model".to_string(),
        ).await.unwrap();
        
        // Register version
        let config = ModelConfig::default();
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        
        let version_id = registry.register_version(
            model_id,
            "1.0.0".to_string(),
            "test-user".to_string(),
            "Initial version".to_string(),
            config,
            metrics,
        ).await.unwrap();
        
        // Deploy version
        let deployment_id = registry.deploy_version(
            version_id,
            ResourceAllocation::default(),
        ).await.unwrap();
        
        // Wait for deployment
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        // Check deployment status
        let deployment = registry.get_deployment(deployment_id).await.unwrap();
        assert_eq!(deployment.status, DeploymentStatus::Deployed);
        assert!(deployment.endpoint.is_some());
        
        // Promote to production
        registry.promote_to_production(version_id).await.unwrap();
        
        let model = registry.get_model(model_id).await.unwrap();
        assert_eq!(model.production_version, Some(version_id));
    }

    #[tokio::test]
    async fn test_ab_testing() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path().to_path_buf());
        
        // Register model and two versions
        let model_id = registry.register_model(
            "ab-test-model".to_string(),
            "GNNEmbedding".to_string(),
            "test-user".to_string(),
            "AB test model".to_string(),
        ).await.unwrap();
        
        let version_a = registry.register_version(
            model_id,
            "1.0.0".to_string(),
            "test-user".to_string(),
            "Version A".to_string(),
            ModelConfig::default(),
            HashMap::new(),
        ).await.unwrap();
        
        let version_b = registry.register_version(
            model_id,
            "1.1.0".to_string(),
            "test-user".to_string(),
            "Version B".to_string(),
            ModelConfig::default(),
            HashMap::new(),
        ).await.unwrap();
        
        // Create A/B test
        let test_id = registry.create_ab_test(
            "Performance test".to_string(),
            "Testing new model version".to_string(),
            version_a,
            version_b,
            0.3, // 30% traffic to version B
            Some(24), // 24 hour test
        ).await.unwrap();
        
        // Check active tests
        let active_tests = registry.get_active_ab_tests().await;
        assert_eq!(active_tests.len(), 1);
        assert_eq!(active_tests[0].test_id, test_id);
        
        // End test
        registry.end_ab_test(test_id).await.unwrap();
        
        let active_tests = registry.get_active_ab_tests().await;
        assert_eq!(active_tests.len(), 0);
    }
}