//! Configuration types for AI orchestration
//!
//! This module contains all configuration structures and settings
//! for the AI orchestration system.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::ml::VotingStrategy;

/// Advanced model selection strategies for dynamic orchestration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ModelSelectionStrategy {
    /// Performance-based selection using historical metrics
    #[default]
    PerformanceBased,
    /// Adaptive selection based on data characteristics
    DataAdaptive,
    /// Ensemble-weighted selection with dynamic weights
    EnsembleWeighted,
    /// Reinforcement learning-based selection
    ReinforcementLearning,
    /// Meta-learning approach for model selection
    MetaLearning,
    /// Hybrid approach combining multiple strategies
    Hybrid(Vec<ModelSelectionStrategy>),
}

/// Performance requirements for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub min_accuracy: f64,
    pub min_precision: f64,
    pub min_recall: f64,
    pub max_inference_time: Duration,
    pub max_memory_usage: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            min_accuracy: 0.8,
            min_precision: 0.75,
            min_recall: 0.7,
            max_inference_time: Duration::from_millis(100),
            max_memory_usage: 500.0,
        }
    }
}

/// Configuration for AI orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiOrchestratorConfig {
    /// Enable ensemble learning
    pub enable_ensemble_learning: bool,

    /// Ensemble voting strategy
    pub ensemble_voting: VotingStrategy,

    /// Enable multi-stage learning
    pub enable_multi_stage_learning: bool,

    /// Enable quality-driven optimization
    pub enable_quality_optimization: bool,

    /// Enable predictive validation
    pub enable_predictive_validation: bool,

    /// Confidence threshold for shape generation
    pub min_shape_confidence: f64,

    /// Maximum number of shapes to generate
    pub max_shapes_generated: usize,

    /// Enable adaptive learning
    pub enable_adaptive_learning: bool,

    /// Learning rate adaptation factor
    pub learning_rate_adaptation: f64,

    /// Enable continuous improvement
    pub enable_continuous_improvement: bool,

    /// Model selection strategy for dynamic orchestration
    pub model_selection_strategy: ModelSelectionStrategy,

    /// Enable advanced model selection
    pub enable_advanced_model_selection: bool,

    /// Performance requirements for model selection
    pub performance_requirements: PerformanceRequirements,

    /// Enable neural pattern recognition
    pub enable_neural_patterns: bool,

    /// Enable advanced analytics
    pub enable_advanced_analytics: bool,

    /// Enable real-time optimization
    pub enable_real_time_optimization: bool,

    /// Enable quality assessment
    pub enable_quality_assessment: bool,

    /// Maximum number of concurrent learning tasks
    pub max_concurrent_tasks: usize,

    /// Learning timeout in seconds
    pub learning_timeout_secs: u64,

    /// Cache configuration
    pub cache_config: CacheConfig,

    /// Resource limits
    pub resource_limits: ResourceLimits,

    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,

    /// Advanced features
    pub advanced_features: AdvancedFeatures,
}

impl Default for AiOrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_ensemble_learning: true,
            ensemble_voting: VotingStrategy::Weighted,
            enable_multi_stage_learning: true,
            enable_quality_optimization: true,
            enable_predictive_validation: true,
            min_shape_confidence: 0.7,
            max_shapes_generated: 100,
            enable_adaptive_learning: true,
            learning_rate_adaptation: 0.95,
            enable_continuous_improvement: true,
            model_selection_strategy: ModelSelectionStrategy::Hybrid(vec![
                ModelSelectionStrategy::PerformanceBased,
                ModelSelectionStrategy::DataAdaptive,
                ModelSelectionStrategy::EnsembleWeighted,
            ]),
            enable_advanced_model_selection: true,
            performance_requirements: PerformanceRequirements::default(),
            enable_neural_patterns: true,
            enable_advanced_analytics: true,
            enable_real_time_optimization: true,
            enable_quality_assessment: true,
            max_concurrent_tasks: 4,
            learning_timeout_secs: 300,
            cache_config: CacheConfig::default(),
            resource_limits: ResourceLimits::default(),
            quality_thresholds: QualityThresholds::default(),
            advanced_features: AdvancedFeatures::default(),
        }
    }
}

impl AiOrchestratorConfig {
    /// Validate the configuration and return any errors
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate confidence threshold
        if self.min_shape_confidence < 0.0 || self.min_shape_confidence > 1.0 {
            return Err(ConfigValidationError::InvalidConfidenceThreshold(
                self.min_shape_confidence,
            ));
        }

        // Validate max shapes generated
        if self.max_shapes_generated == 0 {
            return Err(ConfigValidationError::InvalidMaxShapes(
                self.max_shapes_generated,
            ));
        }

        // Validate learning rate adaptation
        if self.learning_rate_adaptation <= 0.0 || self.learning_rate_adaptation > 1.0 {
            return Err(ConfigValidationError::InvalidLearningRate(
                self.learning_rate_adaptation,
            ));
        }

        // Validate concurrent tasks
        if self.max_concurrent_tasks == 0 {
            return Err(ConfigValidationError::InvalidConcurrentTasks(
                self.max_concurrent_tasks,
            ));
        }

        // Validate learning timeout
        if self.learning_timeout_secs == 0 {
            return Err(ConfigValidationError::InvalidTimeout(
                self.learning_timeout_secs,
            ));
        }

        // Validate performance requirements
        self.performance_requirements.validate()?;

        // Validate resource limits
        self.resource_limits.validate()?;

        // Validate quality thresholds
        self.quality_thresholds.validate()?;

        Ok(())
    }

    /// Create a configuration builder
    pub fn builder() -> AiOrchestratorConfigBuilder {
        AiOrchestratorConfigBuilder::new()
    }
}

impl PerformanceRequirements {
    /// Validate performance requirements
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        // Validate accuracy range
        if self.min_accuracy < 0.0 || self.min_accuracy > 1.0 {
            return Err(ConfigValidationError::InvalidAccuracy(self.min_accuracy));
        }

        // Validate precision range
        if self.min_precision < 0.0 || self.min_precision > 1.0 {
            return Err(ConfigValidationError::InvalidPrecision(self.min_precision));
        }

        // Validate recall range
        if self.min_recall < 0.0 || self.min_recall > 1.0 {
            return Err(ConfigValidationError::InvalidRecall(self.min_recall));
        }

        // Validate memory usage
        if self.max_memory_usage <= 0.0 {
            return Err(ConfigValidationError::InvalidMemoryUsage(
                self.max_memory_usage,
            ));
        }

        Ok(())
    }
}

impl ResourceLimits {
    /// Validate resource limits
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.max_memory_mb == 0 {
            return Err(ConfigValidationError::InvalidMemoryLimit(
                self.max_memory_mb,
            ));
        }

        if self.max_cpu_percent <= 0.0 || self.max_cpu_percent > 100.0 {
            return Err(ConfigValidationError::InvalidCpuLimit(self.max_cpu_percent));
        }

        if self.max_disk_mb == 0 {
            return Err(ConfigValidationError::InvalidDiskLimit(self.max_disk_mb));
        }

        if self.max_threads == 0 {
            return Err(ConfigValidationError::InvalidThreadLimit(self.max_threads));
        }

        if self.max_training_time_secs == 0 {
            return Err(ConfigValidationError::InvalidTrainingTime(
                self.max_training_time_secs,
            ));
        }

        Ok(())
    }
}

impl QualityThresholds {
    /// Validate quality thresholds
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        let thresholds = [
            ("accuracy", self.min_accuracy),
            ("precision", self.min_precision),
            ("recall", self.min_recall),
            ("f1_score", self.min_f1_score),
            ("confidence", self.min_confidence),
        ];

        for (name, value) in thresholds.iter() {
            if *value < 0.0 || *value > 1.0 {
                return Err(ConfigValidationError::InvalidQualityThreshold {
                    metric: name.to_string(),
                    value: *value,
                });
            }
        }

        if self.max_error_rate < 0.0 || self.max_error_rate > 1.0 {
            return Err(ConfigValidationError::InvalidErrorRate(self.max_error_rate));
        }

        Ok(())
    }
}

/// Configuration validation errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigValidationError {
    #[error("Invalid confidence threshold: {0} (must be between 0.0 and 1.0)")]
    InvalidConfidenceThreshold(f64),

    #[error("Invalid max shapes: {0} (must be greater than 0)")]
    InvalidMaxShapes(usize),

    #[error("Invalid learning rate: {0} (must be between 0.0 and 1.0)")]
    InvalidLearningRate(f64),

    #[error("Invalid concurrent tasks: {0} (must be greater than 0)")]
    InvalidConcurrentTasks(usize),

    #[error("Invalid timeout: {0} (must be greater than 0)")]
    InvalidTimeout(u64),

    #[error("Invalid accuracy: {0} (must be between 0.0 and 1.0)")]
    InvalidAccuracy(f64),

    #[error("Invalid precision: {0} (must be between 0.0 and 1.0)")]
    InvalidPrecision(f64),

    #[error("Invalid recall: {0} (must be between 0.0 and 1.0)")]
    InvalidRecall(f64),

    #[error("Invalid memory usage: {0} (must be greater than 0.0)")]
    InvalidMemoryUsage(f64),

    #[error("Invalid memory limit: {0} (must be greater than 0)")]
    InvalidMemoryLimit(usize),

    #[error("Invalid CPU limit: {0} (must be between 0.0 and 100.0)")]
    InvalidCpuLimit(f64),

    #[error("Invalid disk limit: {0} (must be greater than 0)")]
    InvalidDiskLimit(usize),

    #[error("Invalid thread limit: {0} (must be greater than 0)")]
    InvalidThreadLimit(usize),

    #[error("Invalid training time: {0} (must be greater than 0)")]
    InvalidTrainingTime(u64),

    #[error("Invalid quality threshold for {metric}: {value} (must be between 0.0 and 1.0)")]
    InvalidQualityThreshold { metric: String, value: f64 },

    #[error("Invalid error rate: {0} (must be between 0.0 and 1.0)")]
    InvalidErrorRate(f64),
}

/// Builder for AiOrchestratorConfig
#[derive(Debug, Default)]
pub struct AiOrchestratorConfigBuilder {
    config: AiOrchestratorConfig,
}

impl AiOrchestratorConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: AiOrchestratorConfig::default(),
        }
    }

    /// Enable or disable ensemble learning
    pub fn ensemble_learning(mut self, enable: bool) -> Self {
        self.config.enable_ensemble_learning = enable;
        self
    }

    /// Set ensemble voting strategy
    pub fn ensemble_voting(mut self, strategy: VotingStrategy) -> Self {
        self.config.ensemble_voting = strategy;
        self
    }

    /// Enable or disable multi-stage learning
    pub fn multi_stage_learning(mut self, enable: bool) -> Self {
        self.config.enable_multi_stage_learning = enable;
        self
    }

    /// Enable or disable quality optimization
    pub fn quality_optimization(mut self, enable: bool) -> Self {
        self.config.enable_quality_optimization = enable;
        self
    }

    /// Enable or disable predictive validation
    pub fn predictive_validation(mut self, enable: bool) -> Self {
        self.config.enable_predictive_validation = enable;
        self
    }

    /// Set minimum shape confidence threshold
    pub fn min_shape_confidence(mut self, confidence: f64) -> Self {
        self.config.min_shape_confidence = confidence;
        self
    }

    /// Set maximum number of shapes to generate
    pub fn max_shapes_generated(mut self, max_shapes: usize) -> Self {
        self.config.max_shapes_generated = max_shapes;
        self
    }

    /// Enable or disable adaptive learning
    pub fn adaptive_learning(mut self, enable: bool) -> Self {
        self.config.enable_adaptive_learning = enable;
        self
    }

    /// Set learning rate adaptation factor
    pub fn learning_rate_adaptation(mut self, rate: f64) -> Self {
        self.config.learning_rate_adaptation = rate;
        self
    }

    /// Set model selection strategy
    pub fn model_selection_strategy(mut self, strategy: ModelSelectionStrategy) -> Self {
        self.config.model_selection_strategy = strategy;
        self
    }

    /// Set performance requirements
    pub fn performance_requirements(mut self, requirements: PerformanceRequirements) -> Self {
        self.config.performance_requirements = requirements;
        self
    }

    /// Set maximum concurrent tasks
    pub fn max_concurrent_tasks(mut self, max_tasks: usize) -> Self {
        self.config.max_concurrent_tasks = max_tasks;
        self
    }

    /// Set learning timeout in seconds
    pub fn learning_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.learning_timeout_secs = timeout;
        self
    }

    /// Set cache configuration
    pub fn cache_config(mut self, cache_config: CacheConfig) -> Self {
        self.config.cache_config = cache_config;
        self
    }

    /// Set resource limits
    pub fn resource_limits(mut self, limits: ResourceLimits) -> Self {
        self.config.resource_limits = limits;
        self
    }

    /// Set quality thresholds
    pub fn quality_thresholds(mut self, thresholds: QualityThresholds) -> Self {
        self.config.quality_thresholds = thresholds;
        self
    }

    /// Set advanced features
    pub fn advanced_features(mut self, features: AdvancedFeatures) -> Self {
        self.config.advanced_features = features;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> Result<AiOrchestratorConfig, ConfigValidationError> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build the configuration without validation
    pub fn build_unchecked(self) -> AiOrchestratorConfig {
        self.config
    }
}

/// Cache configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,

    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,

    /// Enable result caching
    pub enable_result_caching: bool,

    /// Enable model caching
    pub enable_model_caching: bool,

    /// Maximum number of cached models
    pub max_cached_models: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size_mb: 512,
            cache_ttl_secs: 3600, // 1 hour
            enable_result_caching: true,
            enable_model_caching: true,
            max_cached_models: 10,
        }
    }
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,

    /// Maximum CPU usage percentage
    pub max_cpu_percent: f64,

    /// Maximum disk usage in MB
    pub max_disk_mb: usize,

    /// Maximum number of threads
    pub max_threads: usize,

    /// Maximum training time per model in seconds
    pub max_training_time_secs: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048, // 2GB
            max_cpu_percent: 80.0,
            max_disk_mb: 1024, // 1GB
            max_threads: num_cpus::get(),
            max_training_time_secs: 1800, // 30 minutes
        }
    }
}

/// Quality threshold settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum acceptable accuracy
    pub min_accuracy: f64,

    /// Minimum acceptable precision
    pub min_precision: f64,

    /// Minimum acceptable recall
    pub min_recall: f64,

    /// Minimum acceptable F1 score
    pub min_f1_score: f64,

    /// Minimum confidence threshold
    pub min_confidence: f64,

    /// Maximum acceptable error rate
    pub max_error_rate: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_accuracy: 0.8,
            min_precision: 0.75,
            min_recall: 0.7,
            min_f1_score: 0.72,
            min_confidence: 0.6,
            max_error_rate: 0.2,
        }
    }
}

/// Advanced feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedFeatures {
    /// Enable meta-learning capabilities
    pub enable_meta_learning: bool,

    /// Enable transfer learning
    pub enable_transfer_learning: bool,

    /// Enable automated hyperparameter tuning
    pub enable_auto_hyperparameter_tuning: bool,

    /// Enable model interpretation
    pub enable_model_interpretation: bool,

    /// Enable federated learning
    pub enable_federated_learning: bool,

    /// Enable online learning
    pub enable_online_learning: bool,

    /// Enable uncertainty quantification
    pub enable_uncertainty_quantification: bool,

    /// Enable adversarial training
    pub enable_adversarial_training: bool,
}

impl Default for AdvancedFeatures {
    fn default() -> Self {
        Self {
            enable_meta_learning: false,
            enable_transfer_learning: false,
            enable_auto_hyperparameter_tuning: false,
            enable_model_interpretation: true,
            enable_federated_learning: false,
            enable_online_learning: false,
            enable_uncertainty_quantification: false,
            enable_adversarial_training: false,
        }
    }
}
