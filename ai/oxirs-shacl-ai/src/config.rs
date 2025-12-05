//! Configuration types for SHACL-AI operations
//!
//! This module contains all configuration structures used throughout the oxirs-shacl-ai crate,
//! including model configurations, training parameters, and global AI settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::learning::LearningConfig;
use crate::optimization;
use crate::patterns::PatternConfig;
use crate::quality::QualityConfig;
use crate::{AnalyticsConfig, PredictionConfig};

/// AI model types for different learning tasks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiModelType {
    /// Pattern recognition for constraint discovery
    PatternRecognition,

    /// Classification for data quality assessment
    QualityClassification,

    /// Regression for performance prediction
    PerformancePrediction,

    /// Clustering for shape grouping
    ShapeClustering,

    /// Anomaly detection for data issues
    AnomalyDetection,

    /// Reinforcement learning for validation optimization
    ValidationOptimization,
}

/// Machine learning model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiModelConfig {
    /// Type of model
    pub model_type: AiModelType,

    /// Model parameters
    pub parameters: HashMap<String, f64>,

    /// Training configuration
    pub training: TrainingConfig,

    /// Feature engineering settings
    pub features: FeatureConfig,

    /// Model versioning
    pub version: String,

    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
}

/// Training configuration for AI models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Training data split ratio
    pub train_split: f64,

    /// Validation data split ratio
    pub validation_split: f64,

    /// Maximum training epochs
    pub max_epochs: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Batch size for training
    pub batch_size: usize,

    /// Early stopping patience
    pub patience: usize,

    /// Model checkpointing interval
    pub checkpoint_interval: usize,
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Include graph structure features
    pub include_graph_structure: bool,

    /// Include cardinality features
    pub include_cardinality: bool,

    /// Include type distribution features
    pub include_type_distribution: bool,

    /// Include pattern frequency features
    pub include_pattern_frequency: bool,

    /// Include temporal features
    pub include_temporal: bool,

    /// Maximum feature dimension
    pub max_features: usize,

    /// Feature normalization method
    pub normalization: FeatureNormalization,
}

/// Feature normalization methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureNormalization {
    None,
    MinMax,
    StandardScore,
    RobustScaler,
}

/// Performance thresholds for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Minimum accuracy for classification models
    pub min_accuracy: f64,

    /// Minimum precision for pattern recognition
    pub min_precision: f64,

    /// Minimum recall for anomaly detection
    pub min_recall: f64,

    /// Maximum mean squared error for regression
    pub max_mse: f64,

    /// Minimum F1 score
    pub min_f1_score: f64,
}

/// Configuration for SHACL-AI operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShaclAiConfig {
    /// Shape learning configuration
    pub learning: LearningConfig,

    /// Quality assessment configuration
    pub quality: QualityConfig,

    /// Validation prediction configuration
    pub prediction: PredictionConfig,

    /// Optimization configuration
    pub optimization: optimization::OptimizationConfig,

    /// Pattern analysis configuration
    pub patterns: PatternConfig,

    /// Analytics configuration
    pub analytics: AnalyticsConfig,

    /// Global AI settings
    pub global: GlobalAiConfig,
}

/// Global AI configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalAiConfig {
    /// Enable parallel processing
    pub enable_parallel_processing: bool,

    /// Maximum memory usage for AI operations (in MB)
    pub max_memory_mb: usize,

    /// Enable caching of AI results
    pub enable_caching: bool,

    /// Cache size limit
    pub cache_size_limit: usize,

    /// Enable model checkpointing
    pub enable_checkpointing: bool,

    /// Logging level for AI operations
    pub log_level: String,

    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

impl Default for GlobalAiConfig {
    fn default() -> Self {
        Self {
            enable_parallel_processing: true,
            max_memory_mb: 1024,
            enable_caching: true,
            cache_size_limit: 10000,
            enable_checkpointing: true,
            log_level: "info".to_string(),
            enable_monitoring: true,
        }
    }
}

/// Comprehensive statistics about SHACL-AI operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclAiStatistics {
    pub shapes_learned: usize,
    pub quality_assessments: usize,
    pub predictions_made: usize,
    pub optimizations_performed: usize,
    pub patterns_analyzed: usize,
    pub insights_generated: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_ai_config_default() {
        let config = GlobalAiConfig::default();
        assert!(config.enable_parallel_processing);
        assert_eq!(config.max_memory_mb, 1024);
        assert!(config.enable_caching);
        assert_eq!(config.cache_size_limit, 10000);
    }

    #[test]
    fn test_shacl_ai_config_default() {
        let config = ShaclAiConfig::default();
        assert!(config.global.enable_parallel_processing);
        assert_eq!(config.global.max_memory_mb, 1024);
        assert!(config.global.enable_caching);
        assert_eq!(config.global.cache_size_limit, 10000);
    }

    #[test]
    fn test_ai_model_config() {
        let config = AiModelConfig {
            model_type: AiModelType::PatternRecognition,
            parameters: HashMap::new(),
            training: TrainingConfig {
                train_split: 0.8,
                validation_split: 0.2,
                max_epochs: 100,
                learning_rate: 0.001,
                batch_size: 32,
                patience: 10,
                checkpoint_interval: 10,
            },
            features: FeatureConfig {
                include_graph_structure: true,
                include_cardinality: true,
                include_type_distribution: true,
                include_pattern_frequency: true,
                include_temporal: false,
                max_features: 1000,
                normalization: FeatureNormalization::StandardScore,
            },
            version: "1.0.0".to_string(),
            thresholds: PerformanceThresholds {
                min_accuracy: 0.85,
                min_precision: 0.80,
                min_recall: 0.75,
                max_mse: 0.1,
                min_f1_score: 0.8,
            },
        };

        assert_eq!(config.model_type, AiModelType::PatternRecognition);
        assert_eq!(config.training.max_epochs, 100);
        assert_eq!(config.features.max_features, 1000);
        assert_eq!(config.thresholds.min_accuracy, 0.85);
    }
}
