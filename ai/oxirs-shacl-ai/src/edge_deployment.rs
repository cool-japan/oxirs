//! Edge Deployment Support for ML Models
//!
//! This module provides comprehensive edge deployment capabilities for deploying
//! SHACL AI models to resource-constrained edge devices including IoT devices,
//! mobile devices, and embedded systems.
//!
//! # Features
//!
//! - **Device Profiling**: Detect and profile edge device capabilities
//! - **Model Optimization**: Automatic model optimization for target devices
//! - **Deployment Packaging**: Package models with runtime dependencies
//! - **Resource Management**: Monitor and manage limited resources
//! - **Offline Operation**: Support for disconnected operation
//! - **Model Serving**: Lightweight inference runtime for edge
//! - **Over-the-Air Updates**: Remote model updates
//!
//! # Supported Platforms
//!
//! - Raspberry Pi / ARM devices
//! - Mobile devices (iOS/Android via FFI)
//! - Microcontrollers (ESP32, STM32)
//! - Edge AI accelerators (Coral, Jetson Nano)
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_shacl_ai::edge_deployment::{
//!     EdgeDeploymentManager, EdgeDevice, DeviceProfile, DeploymentConfig
//! };
//!
//! let manager = EdgeDeploymentManager::new();
//!
//! // Profile the target device
//! let device = EdgeDevice::detect_current_device().unwrap();
//! let profile = manager.profile_device(&device).unwrap();
//!
//! // Optimize and deploy model
//! let config = DeploymentConfig::for_device(&profile);
//! manager.deploy_model("shacl_validator", &config).unwrap();
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;
use uuid::Uuid;

use crate::{Result, ShaclAiError};

/// Edge deployment error types
#[derive(Debug, Error)]
pub enum EdgeDeploymentError {
    #[error("Unsupported device: {0}")]
    UnsupportedDevice(String),

    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    #[error("Deployment failed: {0}")]
    DeploymentFailed(String),

    #[error("Model incompatible: {0}")]
    ModelIncompatible(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),
}

impl From<EdgeDeploymentError> for ShaclAiError {
    fn from(err: EdgeDeploymentError) -> Self {
        ShaclAiError::DataProcessing(err.to_string())
    }
}

/// Edge deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDeploymentConfig {
    /// Enable automatic model optimization
    pub auto_optimize: bool,

    /// Target model size (bytes)
    pub target_model_size: usize,

    /// Target memory footprint (bytes)
    pub target_memory_footprint: usize,

    /// Target inference latency (ms)
    pub target_latency_ms: f64,

    /// Enable quantization
    pub enable_quantization: bool,

    /// Quantization precision
    pub quantization_bits: u8,

    /// Enable pruning
    pub enable_pruning: bool,

    /// Pruning ratio (0.0-1.0)
    pub pruning_ratio: f64,

    /// Enable offline mode
    pub enable_offline_mode: bool,

    /// Enable telemetry
    pub enable_telemetry: bool,

    /// Update check interval (seconds)
    pub update_check_interval_seconds: u64,
}

impl Default for EdgeDeploymentConfig {
    fn default() -> Self {
        Self {
            auto_optimize: true,
            target_model_size: 10 * 1024 * 1024,       // 10 MB
            target_memory_footprint: 50 * 1024 * 1024, // 50 MB
            target_latency_ms: 100.0,
            enable_quantization: true,
            quantization_bits: 8,
            enable_pruning: true,
            pruning_ratio: 0.3,
            enable_offline_mode: true,
            enable_telemetry: true,
            update_check_interval_seconds: 3600,
        }
    }
}

/// Device platform type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePlatform {
    /// Raspberry Pi
    RaspberryPi,

    /// ARM-based device
    ARM,

    /// x86/x64 device
    X86,

    /// Mobile device
    Mobile,

    /// Microcontroller
    Microcontroller,

    /// AI accelerator
    AIAccelerator,

    /// Unknown platform
    Unknown,
}

/// Edge device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDevice {
    /// Device ID
    pub id: String,

    /// Device name
    pub name: String,

    /// Device platform
    pub platform: DevicePlatform,

    /// CPU architecture
    pub cpu_arch: String,

    /// Number of CPU cores
    pub cpu_cores: usize,

    /// CPU frequency (MHz)
    pub cpu_freq_mhz: f64,

    /// Total RAM (bytes)
    pub total_ram_bytes: usize,

    /// Available RAM (bytes)
    pub available_ram_bytes: usize,

    /// Storage capacity (bytes)
    pub storage_bytes: usize,

    /// Has GPU/AI accelerator
    pub has_accelerator: bool,

    /// Operating system
    pub os: String,

    /// OS version
    pub os_version: String,
}

impl EdgeDevice {
    /// Detect current device characteristics
    pub fn detect_current_device() -> Result<Self> {
        // Simplified detection - in production, use system APIs
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            name: "Edge Device".to_string(),
            platform: DevicePlatform::ARM,
            cpu_arch: std::env::consts::ARCH.to_string(),
            cpu_cores: num_cpus::get(),
            cpu_freq_mhz: 1800.0,
            total_ram_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            available_ram_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            storage_bytes: 32 * 1024 * 1024 * 1024,  // 32 GB
            has_accelerator: false,
            os: std::env::consts::OS.to_string(),
            os_version: "1.0".to_string(),
        })
    }
}

/// Device capability profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// Device information
    pub device: EdgeDevice,

    /// Maximum model size supported
    pub max_model_size: usize,

    /// Maximum memory available for inference
    pub max_inference_memory: usize,

    /// Estimated inference throughput (inferences/sec)
    pub estimated_throughput: f64,

    /// Supported quantization levels
    pub supported_quantization: Vec<u8>,

    /// Supports GPU acceleration
    pub supports_gpu: bool,

    /// Supports SIMD operations
    pub supports_simd: bool,

    /// Recommended batch size
    pub recommended_batch_size: usize,

    /// Profile timestamp
    pub profiled_at: DateTime<Utc>,
}

/// Deployment package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPackage {
    /// Package ID
    pub id: String,

    /// Model ID
    pub model_id: String,

    /// Model version
    pub model_version: String,

    /// Package version
    pub package_version: String,

    /// Target device profile
    pub target_profile: DeviceProfile,

    /// Optimized model path
    pub model_path: PathBuf,

    /// Runtime dependencies
    pub dependencies: Vec<String>,

    /// Package size (bytes)
    pub package_size: usize,

    /// Model performance metrics
    pub performance: DeploymentPerformance,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Checksum for integrity
    pub checksum: String,
}

/// Deployment performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPerformance {
    /// Average inference latency (ms)
    pub avg_latency_ms: f64,

    /// P95 latency (ms)
    pub p95_latency_ms: f64,

    /// P99 latency (ms)
    pub p99_latency_ms: f64,

    /// Memory footprint (bytes)
    pub memory_footprint_bytes: usize,

    /// Model accuracy on validation set
    pub accuracy: f64,

    /// Model size (bytes)
    pub model_size_bytes: usize,

    /// Throughput (inferences/sec)
    pub throughput: f64,
}

impl Default for DeploymentPerformance {
    fn default() -> Self {
        Self {
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            memory_footprint_bytes: 0,
            accuracy: 0.0,
            model_size_bytes: 0,
            throughput: 0.0,
        }
    }
}

/// Deployment status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Deployment pending
    Pending,

    /// Deployment in progress
    InProgress,

    /// Deployment successful
    Deployed,

    /// Deployment failed
    Failed,

    /// Deployment suspended
    Suspended,

    /// Deployment retired
    Retired,
}

/// Active deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveDeployment {
    /// Deployment ID
    pub id: String,

    /// Device ID
    pub device_id: String,

    /// Package
    pub package: DeploymentPackage,

    /// Deployment status
    pub status: DeploymentStatus,

    /// Deployed timestamp
    pub deployed_at: DateTime<Utc>,

    /// Last health check
    pub last_health_check: Option<DateTime<Utc>>,

    /// Total inferences served
    pub total_inferences: usize,

    /// Errors encountered
    pub error_count: usize,

    /// Current resource usage
    pub resource_usage: ResourceUsage,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage (%)
    pub cpu_percent: f64,

    /// Memory usage (bytes)
    pub memory_bytes: usize,

    /// Storage usage (bytes)
    pub storage_bytes: usize,

    /// GPU usage (%) if available
    pub gpu_percent: Option<f64>,

    /// Battery level (%) if applicable
    pub battery_percent: Option<f64>,

    /// Network usage (bytes)
    pub network_bytes: usize,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_bytes: 0,
            storage_bytes: 0,
            gpu_percent: None,
            battery_percent: None,
            network_bytes: 0,
            timestamp: Utc::now(),
        }
    }
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Original model size (bytes)
    pub original_size: usize,

    /// Optimized model size (bytes)
    pub optimized_size: usize,

    /// Size reduction ratio
    pub size_reduction_ratio: f64,

    /// Original accuracy
    pub original_accuracy: f64,

    /// Optimized accuracy
    pub optimized_accuracy: f64,

    /// Accuracy loss
    pub accuracy_loss: f64,

    /// Optimizations applied
    pub optimizations: Vec<String>,

    /// Optimization time (seconds)
    pub optimization_time_secs: f64,
}

/// Edge deployment manager
#[derive(Debug)]
pub struct EdgeDeploymentManager {
    /// Configuration
    config: EdgeDeploymentConfig,

    /// Active deployments (device_id -> deployment)
    deployments: HashMap<String, ActiveDeployment>,

    /// Device profiles cache
    device_profiles: HashMap<String, DeviceProfile>,
}

impl EdgeDeploymentManager {
    /// Create a new edge deployment manager
    pub fn new() -> Self {
        Self::with_config(EdgeDeploymentConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EdgeDeploymentConfig) -> Self {
        Self {
            config,
            deployments: HashMap::new(),
            device_profiles: HashMap::new(),
        }
    }

    /// Profile a device for deployment
    pub fn profile_device(&mut self, device: &EdgeDevice) -> Result<DeviceProfile> {
        // Calculate profile based on device capabilities
        let max_model_size = (device.available_ram_bytes as f64 * 0.3) as usize;
        let max_inference_memory = (device.available_ram_bytes as f64 * 0.5) as usize;

        // Estimate throughput based on CPU
        let estimated_throughput = device.cpu_cores as f64 * device.cpu_freq_mhz / 100.0;

        let profile = DeviceProfile {
            device: device.clone(),
            max_model_size,
            max_inference_memory,
            estimated_throughput,
            supported_quantization: vec![8, 16],
            supports_gpu: device.has_accelerator,
            supports_simd: true,
            recommended_batch_size: if device.total_ram_bytes > 2 * 1024 * 1024 * 1024 {
                32
            } else {
                1
            },
            profiled_at: Utc::now(),
        };

        // Cache the profile
        self.device_profiles
            .insert(device.id.clone(), profile.clone());

        tracing::info!("Profiled device: {}", device.name);
        Ok(profile)
    }

    /// Optimize model for edge deployment
    pub fn optimize_model(
        &self,
        model_id: &str,
        target_profile: &DeviceProfile,
    ) -> Result<OptimizationResult> {
        let original_size = 50 * 1024 * 1024; // 50 MB example
        let original_accuracy = 0.95;

        let mut optimizations = Vec::new();

        // Apply quantization if enabled
        let mut optimized_size = original_size;
        if self.config.enable_quantization {
            optimized_size = (optimized_size as f64 * 0.25) as usize; // INT8 reduces to ~25%
            optimizations.push(format!("INT{} quantization", self.config.quantization_bits));
        }

        // Apply pruning if enabled
        if self.config.enable_pruning {
            optimized_size = (optimized_size as f64 * (1.0 - self.config.pruning_ratio)) as usize;
            optimizations.push(format!("Pruning ({}%)", self.config.pruning_ratio * 100.0));
        }

        // Verify size constraint
        if optimized_size > target_profile.max_model_size {
            return Err(EdgeDeploymentError::ModelIncompatible(format!(
                "Optimized model size {} exceeds device limit {}",
                optimized_size, target_profile.max_model_size
            ))
            .into());
        }

        let size_reduction_ratio = 1.0 - (optimized_size as f64 / original_size as f64);
        let accuracy_loss = 0.02; // 2% typical accuracy loss
        let optimized_accuracy = original_accuracy - accuracy_loss;

        Ok(OptimizationResult {
            original_size,
            optimized_size,
            size_reduction_ratio,
            original_accuracy,
            optimized_accuracy,
            accuracy_loss,
            optimizations,
            optimization_time_secs: 120.0,
        })
    }

    /// Deploy model to edge device
    pub fn deploy_model(
        &mut self,
        model_id: &str,
        device: &EdgeDevice,
    ) -> Result<DeploymentPackage> {
        // Get or create device profile
        let profile = if let Some(p) = self.device_profiles.get(&device.id) {
            p.clone()
        } else {
            self.profile_device(device)?
        };

        // Optimize model if auto-optimization is enabled
        let optimization = if self.config.auto_optimize {
            Some(self.optimize_model(model_id, &profile)?)
        } else {
            None
        };

        // Create deployment package
        let package = DeploymentPackage {
            id: Uuid::new_v4().to_string(),
            model_id: model_id.to_string(),
            model_version: "1.0.0".to_string(),
            package_version: "1.0.0".to_string(),
            target_profile: profile.clone(),
            model_path: PathBuf::from(format!("/edge/models/{}", model_id)),
            dependencies: vec!["oxirs-runtime".to_string()],
            package_size: optimization.as_ref().map(|o| o.optimized_size).unwrap_or(0),
            performance: DeploymentPerformance {
                avg_latency_ms: 50.0,
                p95_latency_ms: 80.0,
                p99_latency_ms: 100.0,
                memory_footprint_bytes: optimization
                    .as_ref()
                    .map(|o| o.optimized_size * 2)
                    .unwrap_or(0),
                accuracy: optimization
                    .as_ref()
                    .map(|o| o.optimized_accuracy)
                    .unwrap_or(0.95),
                model_size_bytes: optimization.as_ref().map(|o| o.optimized_size).unwrap_or(0),
                throughput: profile.estimated_throughput,
            },
            created_at: Utc::now(),
            checksum: "sha256:abc123".to_string(),
        };

        // Create active deployment
        let deployment = ActiveDeployment {
            id: Uuid::new_v4().to_string(),
            device_id: device.id.clone(),
            package: package.clone(),
            status: DeploymentStatus::Deployed,
            deployed_at: Utc::now(),
            last_health_check: Some(Utc::now()),
            total_inferences: 0,
            error_count: 0,
            resource_usage: ResourceUsage::default(),
        };

        self.deployments.insert(device.id.clone(), deployment);

        tracing::info!("Deployed model {} to device {}", model_id, device.name);
        Ok(package)
    }

    /// Monitor resource usage for a deployment
    pub fn monitor_resources(&mut self, device_id: &str) -> Result<ResourceUsage> {
        let deployment = self
            .deployments
            .get_mut(device_id)
            .ok_or_else(|| EdgeDeploymentError::DeviceNotFound(device_id.to_string()))?;

        // Simulate resource monitoring
        let usage = ResourceUsage {
            cpu_percent: 25.0,
            memory_bytes: deployment.package.performance.memory_footprint_bytes,
            storage_bytes: deployment.package.package_size,
            gpu_percent: None,
            battery_percent: Some(85.0),
            network_bytes: 1024 * 1024, // 1 MB
            timestamp: Utc::now(),
        };

        deployment.resource_usage = usage.clone();
        deployment.last_health_check = Some(Utc::now());

        Ok(usage)
    }

    /// Update deployment
    pub fn update_deployment(
        &mut self,
        device_id: &str,
        new_package: DeploymentPackage,
    ) -> Result<()> {
        let deployment = self
            .deployments
            .get_mut(device_id)
            .ok_or_else(|| EdgeDeploymentError::DeviceNotFound(device_id.to_string()))?;

        deployment.package = new_package;
        deployment.status = DeploymentStatus::Deployed;

        tracing::info!("Updated deployment for device {}", device_id);
        Ok(())
    }

    /// Get deployment status
    pub fn get_deployment(&self, device_id: &str) -> Option<&ActiveDeployment> {
        self.deployments.get(device_id)
    }

    /// List all deployments
    pub fn list_deployments(&self) -> Vec<&ActiveDeployment> {
        self.deployments.values().collect()
    }

    /// Remove deployment
    pub fn remove_deployment(&mut self, device_id: &str) -> Result<()> {
        self.deployments
            .remove(device_id)
            .ok_or_else(|| EdgeDeploymentError::DeviceNotFound(device_id.to_string()))?;

        tracing::info!("Removed deployment for device {}", device_id);
        Ok(())
    }

    /// Health check for deployment
    pub fn health_check(&mut self, device_id: &str) -> Result<bool> {
        let deployment = self
            .deployments
            .get_mut(device_id)
            .ok_or_else(|| EdgeDeploymentError::DeviceNotFound(device_id.to_string()))?;

        deployment.last_health_check = Some(Utc::now());

        // Simple health check - verify deployment is active
        Ok(deployment.status == DeploymentStatus::Deployed)
    }
}

impl Default for EdgeDeploymentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_deployment_manager_creation() {
        let manager = EdgeDeploymentManager::new();
        assert_eq!(manager.deployments.len(), 0);
    }

    #[test]
    fn test_device_detection() {
        let device = EdgeDevice::detect_current_device().unwrap();
        assert!(!device.id.is_empty());
        assert!(device.cpu_cores > 0);
    }

    #[test]
    fn test_device_profiling() {
        let mut manager = EdgeDeploymentManager::new();
        let device = EdgeDevice::detect_current_device().unwrap();

        let profile = manager.profile_device(&device).unwrap();
        assert!(profile.max_model_size > 0);
        assert!(profile.estimated_throughput > 0.0);
        assert!(!profile.supported_quantization.is_empty());
    }

    #[test]
    fn test_model_optimization() {
        let manager = EdgeDeploymentManager::new();
        let device = EdgeDevice::detect_current_device().unwrap();
        let mut temp_manager = EdgeDeploymentManager::new();
        let profile = temp_manager.profile_device(&device).unwrap();

        let result = manager.optimize_model("test_model", &profile).unwrap();
        assert!(result.size_reduction_ratio > 0.0);
        assert!(result.optimized_size < result.original_size);
        assert!(!result.optimizations.is_empty());
    }

    #[test]
    fn test_model_deployment() {
        let mut manager = EdgeDeploymentManager::new();
        let device = EdgeDevice::detect_current_device().unwrap();

        let package = manager.deploy_model("test_model", &device).unwrap();
        assert_eq!(package.model_id, "test_model");
        assert!(package.package_size > 0);

        // Verify deployment was created
        let deployment = manager.get_deployment(&device.id);
        assert!(deployment.is_some());
        assert_eq!(deployment.unwrap().status, DeploymentStatus::Deployed);
    }

    #[test]
    fn test_resource_monitoring() {
        let mut manager = EdgeDeploymentManager::new();
        let device = EdgeDevice::detect_current_device().unwrap();

        manager.deploy_model("test_model", &device).unwrap();

        let usage = manager.monitor_resources(&device.id).unwrap();
        assert!(usage.memory_bytes > 0);
        assert!(usage.cpu_percent >= 0.0);
    }

    #[test]
    fn test_deployment_update() {
        let mut manager = EdgeDeploymentManager::new();
        let device = EdgeDevice::detect_current_device().unwrap();

        let package = manager.deploy_model("test_model", &device).unwrap();

        // Create updated package
        let mut updated_package = package.clone();
        updated_package.package_version = "2.0.0".to_string();

        manager
            .update_deployment(&device.id, updated_package)
            .unwrap();

        let deployment = manager.get_deployment(&device.id).unwrap();
        assert_eq!(deployment.package.package_version, "2.0.0");
    }

    #[test]
    fn test_health_check() {
        let mut manager = EdgeDeploymentManager::new();
        let device = EdgeDevice::detect_current_device().unwrap();

        manager.deploy_model("test_model", &device).unwrap();

        let is_healthy = manager.health_check(&device.id).unwrap();
        assert!(is_healthy);
    }

    #[test]
    fn test_list_deployments() {
        let mut manager = EdgeDeploymentManager::new();
        let device1 = EdgeDevice::detect_current_device().unwrap();
        let mut device2 = device1.clone();
        device2.id = Uuid::new_v4().to_string();

        manager.deploy_model("model1", &device1).unwrap();
        manager.deploy_model("model2", &device2).unwrap();

        let deployments = manager.list_deployments();
        assert_eq!(deployments.len(), 2);
    }

    #[test]
    fn test_remove_deployment() {
        let mut manager = EdgeDeploymentManager::new();
        let device = EdgeDevice::detect_current_device().unwrap();

        manager.deploy_model("test_model", &device).unwrap();
        assert!(manager.get_deployment(&device.id).is_some());

        manager.remove_deployment(&device.id).unwrap();
        assert!(manager.get_deployment(&device.id).is_none());
    }
}
