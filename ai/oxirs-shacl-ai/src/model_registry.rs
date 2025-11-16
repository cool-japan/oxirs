//! Model Registry and Versioning System
//!
//! This module provides comprehensive model lifecycle management including versioning,
//! storage, retrieval, and performance tracking for AI models.

use crate::{Result, ShaclAiError};

use chrono::{DateTime, Utc};
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Central model registry for managing all AI models
#[derive(Debug)]
pub struct ModelRegistry {
    /// Registered models by name and version
    models: Arc<Mutex<HashMap<String, BTreeMap<Version, RegisteredModel>>>>,

    /// Model metadata
    metadata: Arc<Mutex<HashMap<String, ModelMetadata>>>,

    /// Storage backend
    storage: Arc<Mutex<ModelStorage>>,

    /// Configuration
    config: RegistryConfig,

    /// Performance tracker
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Storage path for models
    pub storage_path: PathBuf,

    /// Enable automatic versioning
    pub auto_versioning: bool,

    /// Maximum versions to keep per model
    pub max_versions_per_model: usize,

    /// Enable performance tracking
    pub enable_performance_tracking: bool,

    /// Enable model compression
    pub enable_compression: bool,

    /// Compression level (1-9)
    pub compression_level: u8,

    /// Enable model validation on registration
    pub enable_validation: bool,

    /// Enable automatic cleanup of old models
    pub enable_auto_cleanup: bool,

    /// Days to keep old models
    pub retention_days: u32,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            storage_path: std::env::temp_dir().join("oxirs_model_registry"),
            auto_versioning: true,
            max_versions_per_model: 10,
            enable_performance_tracking: true,
            enable_compression: true,
            compression_level: 6,
            enable_validation: true,
            enable_auto_cleanup: true,
            retention_days: 90,
        }
    }
}

/// Builder for model registration to avoid too many function arguments
#[derive(Debug, Clone)]
pub struct ModelRegistrationBuilder {
    pub name: String,
    pub model_type: ModelType,
    pub parameters: ModelParameters,
    pub config: serde_json::Value,
    pub training_metrics: TrainingMetrics,
    pub description: String,
    pub author: String,
    pub tags: Vec<String>,
}

impl ModelRegistrationBuilder {
    pub fn new(name: String, model_type: ModelType) -> Self {
        Self {
            name,
            model_type,
            parameters: ModelParameters::default(),
            config: serde_json::Value::Null,
            training_metrics: TrainingMetrics::default(),
            description: String::new(),
            author: String::from("unknown"),
            tags: Vec::new(),
        }
    }

    pub fn with_parameters(mut self, parameters: ModelParameters) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = config;
        self
    }

    pub fn with_training_metrics(mut self, metrics: TrainingMetrics) -> Self {
        self.training_metrics = metrics;
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    pub fn with_author(mut self, author: String) -> Self {
        self.author = author;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Model version
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn increment_major(&mut self) {
        self.major += 1;
        self.minor = 0;
        self.patch = 0;
    }

    pub fn increment_minor(&mut self) {
        self.minor += 1;
        self.patch = 0;
    }

    pub fn increment_patch(&mut self) {
        self.patch += 1;
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::str::FromStr for Version {
    type Err = ShaclAiError;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(ShaclAiError::Configuration(format!(
                "Invalid version format: {}",
                s
            )));
        }

        Ok(Version::new(
            parts[0]
                .parse()
                .map_err(|_| ShaclAiError::Configuration("Invalid major version".to_string()))?,
            parts[1]
                .parse()
                .map_err(|_| ShaclAiError::Configuration("Invalid minor version".to_string()))?,
            parts[2]
                .parse()
                .map_err(|_| ShaclAiError::Configuration("Invalid patch version".to_string()))?,
        ))
    }
}

/// Registered model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    /// Model ID
    pub id: String,

    /// Model name
    pub name: String,

    /// Version
    pub version: Version,

    /// Model type
    pub model_type: ModelType,

    /// Registration timestamp
    pub registered_at: DateTime<Utc>,

    /// Model size in bytes
    pub size_bytes: usize,

    /// Model parameters (weights, etc.)
    pub parameters: ModelParameters,

    /// Model configuration
    pub config: serde_json::Value,

    /// Training metrics
    pub training_metrics: TrainingMetrics,

    /// Status
    pub status: ModelStatus,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Description
    pub description: String,

    /// Author
    pub author: String,
}

/// Model type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    ShapeLearner,
    ConstraintGenerator,
    AnomalyDetector,
    QualityAssessor,
    PatternRecognizer,
    Transformer,
    Ensemble,
    Custom(String),
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::ShapeLearner => write!(f, "ShapeLearner"),
            ModelType::ConstraintGenerator => write!(f, "ConstraintGenerator"),
            ModelType::AnomalyDetector => write!(f, "AnomalyDetector"),
            ModelType::QualityAssessor => write!(f, "QualityAssessor"),
            ModelType::PatternRecognizer => write!(f, "PatternRecognizer"),
            ModelType::Transformer => write!(f, "Transformer"),
            ModelType::Ensemble => write!(f, "Ensemble"),
            ModelType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Number of parameters
    pub num_parameters: usize,

    /// Parameter shapes
    pub parameter_shapes: Vec<Vec<usize>>,

    /// Storage path
    pub storage_path: PathBuf,

    /// Checksum for integrity verification
    pub checksum: String,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            num_parameters: 0,
            parameter_shapes: Vec::new(),
            storage_path: std::env::temp_dir(),
            checksum: String::new(),
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Final training loss
    pub final_loss: f64,

    /// Final validation accuracy
    pub validation_accuracy: f64,

    /// Number of training epochs
    pub epochs: usize,

    /// Training duration in seconds
    pub training_duration_secs: f64,

    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            final_loss: 0.0,
            validation_accuracy: 0.0,
            epochs: 0,
            training_duration_secs: 0.0,
            metrics: HashMap::new(),
        }
    }
}

/// Model status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model is being trained
    Training,

    /// Model is registered but not validated
    Registered,

    /// Model is validated and ready for use
    Validated,

    /// Model is deployed in production
    Production,

    /// Model is deprecated
    Deprecated,

    /// Model is archived
    Archived,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_name: String,
    pub latest_version: Version,
    pub total_versions: usize,
    pub production_version: Option<Version>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(config: RegistryConfig) -> Result<Self> {
        // Create storage directory if it doesn't exist
        if !config.storage_path.exists() {
            std::fs::create_dir_all(&config.storage_path).map_err(|e| {
                ShaclAiError::Configuration(format!("Failed to create storage directory: {}", e))
            })?;
        }

        Ok(Self {
            models: Arc::new(Mutex::new(HashMap::new())),
            metadata: Arc::new(Mutex::new(HashMap::new())),
            storage: Arc::new(Mutex::new(ModelStorage::new(config.storage_path.clone()))),
            config,
            performance_tracker: Arc::new(Mutex::new(PerformanceTracker::new())),
        })
    }

    /// Register a new model using a builder
    pub fn register_model(&self, builder: ModelRegistrationBuilder) -> Result<RegisteredModel> {
        let name = builder.name;
        let model_type = builder.model_type;
        let parameters = builder.parameters;
        let config = builder.config;
        let training_metrics = builder.training_metrics;
        let description = builder.description;
        let author = builder.author;
        let tags = builder.tags;
        let mut models = self
            .models
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock models: {}", e)))?;

        // Determine version
        let version = if self.config.auto_versioning {
            self.get_next_version(&name, &models)?
        } else {
            Version::new(1, 0, 0)
        };

        let model = RegisteredModel {
            id: format!("{}_{}", name, version),
            name: name.clone(),
            version: version.clone(),
            model_type,
            registered_at: Utc::now(),
            size_bytes: 0, // Will be calculated during storage
            parameters,
            config,
            training_metrics,
            status: ModelStatus::Registered,
            tags,
            description,
            author,
        };

        // Store model
        let mut storage = self
            .storage
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock storage: {}", e)))?;
        storage.store_model(&model)?;

        // Add to registry
        models
            .entry(name.clone())
            .or_insert_with(BTreeMap::new)
            .insert(version.clone(), model.clone());

        // Update metadata
        self.update_metadata(&name, &version)?;

        // Cleanup old versions if needed
        if self.config.enable_auto_cleanup {
            self.cleanup_old_versions(&name, &mut models)?;
        }

        Ok(model)
    }

    /// Get a specific model version
    pub fn get_model(&self, name: &str, version: &Version) -> Result<Option<RegisteredModel>> {
        let models = self
            .models
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock models: {}", e)))?;

        Ok(models
            .get(name)
            .and_then(|versions| versions.get(version))
            .cloned())
    }

    /// Get the latest version of a model
    pub fn get_latest_model(&self, name: &str) -> Result<Option<RegisteredModel>> {
        let models = self
            .models
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock models: {}", e)))?;

        Ok(models
            .get(name)
            .and_then(|versions| versions.iter().last())
            .map(|(_, model)| model.clone()))
    }

    /// Get the production version of a model
    pub fn get_production_model(&self, name: &str) -> Result<Option<RegisteredModel>> {
        let models = self
            .models
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock models: {}", e)))?;

        Ok(models.get(name).and_then(|versions| {
            versions
                .values()
                .find(|m| m.status == ModelStatus::Production)
                .cloned()
        }))
    }

    /// List all versions of a model
    pub fn list_model_versions(&self, name: &str) -> Result<Vec<RegisteredModel>> {
        let models = self
            .models
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock models: {}", e)))?;

        Ok(models
            .get(name)
            .map(|versions| versions.values().cloned().collect())
            .unwrap_or_default())
    }

    /// Promote a model to production
    pub fn promote_to_production(&self, name: &str, version: &Version) -> Result<()> {
        let mut models = self
            .models
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock models: {}", e)))?;

        // Demote current production model
        if let Some(versions) = models.get_mut(name) {
            for (_, model) in versions.iter_mut() {
                if model.status == ModelStatus::Production {
                    model.status = ModelStatus::Validated;
                }
            }

            // Promote new version
            if let Some(model) = versions.get_mut(version) {
                model.status = ModelStatus::Production;

                // Update metadata
                drop(models); // Release lock before updating metadata
                self.update_production_version(name, version)?;
            } else {
                return Err(ShaclAiError::VersionNotFound(format!(
                    "Model {}:{} not found",
                    name, version
                )));
            }
        } else {
            return Err(ShaclAiError::VersionNotFound(format!(
                "Model {} not found",
                name
            )));
        }

        Ok(())
    }

    /// Compare model versions
    pub fn compare_versions(
        &self,
        name: &str,
        version1: &Version,
        version2: &Version,
    ) -> Result<ModelComparison> {
        let model1 = self
            .get_model(name, version1)?
            .ok_or_else(|| ShaclAiError::VersionNotFound(format!("{}:{}", name, version1)))?;

        let model2 = self
            .get_model(name, version2)?
            .ok_or_else(|| ShaclAiError::VersionNotFound(format!("{}:{}", name, version2)))?;

        Ok(ModelComparison {
            model1_version: version1.clone(),
            model2_version: version2.clone(),
            accuracy_diff: model2.training_metrics.validation_accuracy
                - model1.training_metrics.validation_accuracy,
            loss_diff: model2.training_metrics.final_loss - model1.training_metrics.final_loss,
            size_diff: model2.size_bytes as i64 - model1.size_bytes as i64,
            recommended_version: if model2.training_metrics.validation_accuracy
                > model1.training_metrics.validation_accuracy
            {
                version2.clone()
            } else {
                version1.clone()
            },
        })
    }

    /// Track model performance
    pub fn record_performance(
        &self,
        name: &str,
        version: &Version,
        metrics: PerformanceMetrics,
    ) -> Result<()> {
        let mut tracker = self.performance_tracker.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock performance tracker: {}", e))
        })?;

        tracker.record(name, version, metrics);
        Ok(())
    }

    /// Get performance history
    pub fn get_performance_history(
        &self,
        name: &str,
        version: &Version,
    ) -> Result<Vec<PerformanceMetrics>> {
        let tracker = self.performance_tracker.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock performance tracker: {}", e))
        })?;

        Ok(tracker.get_history(name, version))
    }

    /// Get next version
    fn get_next_version(
        &self,
        name: &str,
        models: &HashMap<String, BTreeMap<Version, RegisteredModel>>,
    ) -> Result<Version> {
        if let Some(versions) = models.get(name) {
            if let Some((latest_version, _)) = versions.iter().last() {
                let mut next_version = latest_version.clone();
                next_version.increment_patch();
                return Ok(next_version);
            }
        }

        Ok(Version::new(1, 0, 0))
    }

    /// Update metadata
    fn update_metadata(&self, name: &str, version: &Version) -> Result<()> {
        let mut metadata = self
            .metadata
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock metadata: {}", e)))?;

        metadata
            .entry(name.to_string())
            .and_modify(|m| {
                m.latest_version = version.clone();
                m.total_versions += 1;
                m.updated_at = Utc::now();
            })
            .or_insert(ModelMetadata {
                model_name: name.to_string(),
                latest_version: version.clone(),
                total_versions: 1,
                production_version: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            });

        Ok(())
    }

    /// Update production version in metadata
    fn update_production_version(&self, name: &str, version: &Version) -> Result<()> {
        let mut metadata = self
            .metadata
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock metadata: {}", e)))?;

        if let Some(meta) = metadata.get_mut(name) {
            meta.production_version = Some(version.clone());
            meta.updated_at = Utc::now();
        }

        Ok(())
    }

    /// Cleanup old versions
    fn cleanup_old_versions(
        &self,
        name: &str,
        models: &mut HashMap<String, BTreeMap<Version, RegisteredModel>>,
    ) -> Result<()> {
        if let Some(versions) = models.get_mut(name) {
            while versions.len() > self.config.max_versions_per_model {
                // Remove oldest version (but keep production)
                if let Some((oldest_version, oldest_model)) = versions.iter().next() {
                    if oldest_model.status != ModelStatus::Production {
                        let version_to_remove = oldest_version.clone();
                        versions.remove(&version_to_remove);
                    } else {
                        break; // Don't remove production model
                    }
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get model metadata
    pub fn get_metadata(&self, name: &str) -> Result<Option<ModelMetadata>> {
        let metadata = self
            .metadata
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock metadata: {}", e)))?;

        Ok(metadata.get(name).cloned())
    }

    /// Get configuration
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }
}

/// Model storage backend
#[derive(Debug)]
struct ModelStorage {
    base_path: PathBuf,
}

impl ModelStorage {
    fn new(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    fn store_model(&mut self, model: &RegisteredModel) -> Result<()> {
        let model_path = self.base_path.join(&model.id);
        std::fs::create_dir_all(&model_path).map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to create model directory: {}", e))
        })?;

        // Store model metadata
        let metadata_path = model_path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(model)?;
        std::fs::write(metadata_path, metadata_json)?;

        Ok(())
    }
}

/// Performance tracker
#[derive(Debug)]
struct PerformanceTracker {
    records: HashMap<String, Vec<(Version, PerformanceMetrics)>>,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    fn record(&mut self, name: &str, version: &Version, metrics: PerformanceMetrics) {
        self.records
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push((version.clone(), metrics));
    }

    fn get_history(&self, name: &str, version: &Version) -> Vec<PerformanceMetrics> {
        self.records
            .get(name)
            .map(|records| {
                records
                    .iter()
                    .filter(|(v, _)| v == version)
                    .map(|(_, m)| m.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub inference_time_ms: f64,
    pub throughput: f64,
    pub accuracy: f64,
    pub memory_usage_mb: f64,
}

/// Model comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model1_version: Version,
    pub model2_version: Version,
    pub accuracy_diff: f64,
    pub loss_diff: f64,
    pub size_diff: i64,
    pub recommended_version: Version,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let version: Version = "1.2.3".parse().unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert_eq!(version.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_increment() {
        let mut version = Version::new(1, 2, 3);
        version.increment_patch();
        assert_eq!(version, Version::new(1, 2, 4));

        version.increment_minor();
        assert_eq!(version, Version::new(1, 3, 0));

        version.increment_major();
        assert_eq!(version, Version::new(2, 0, 0));
    }

    #[test]
    fn test_model_registry_creation() {
        let config = RegistryConfig::default();
        let registry = ModelRegistry::new(config).unwrap();
        assert!(registry.config.auto_versioning);
    }

    #[test]
    fn test_model_registration() {
        let config = RegistryConfig::default();
        let registry = ModelRegistry::new(config).unwrap();

        let builder =
            ModelRegistrationBuilder::new("test_model".to_string(), ModelType::ShapeLearner)
                .with_parameters(ModelParameters {
                    num_parameters: 1000,
                    parameter_shapes: vec![vec![10, 10]],
                    storage_path: PathBuf::from("/tmp/model"),
                    checksum: "abc123".to_string(),
                })
                .with_config(serde_json::json!({}))
                .with_training_metrics(TrainingMetrics {
                    final_loss: 0.1,
                    validation_accuracy: 0.95,
                    epochs: 100,
                    training_duration_secs: 3600.0,
                    metrics: HashMap::new(),
                })
                .with_description("Test model".to_string())
                .with_author("Test Author".to_string())
                .with_tags(vec!["test".to_string()]);

        let model = registry.register_model(builder).unwrap();

        assert_eq!(model.name, "test_model");
        assert_eq!(model.version, Version::new(1, 0, 0));
    }
}
