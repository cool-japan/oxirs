//! Configuration Management for Neural-Symbolic Bridge
//!
//! Provides comprehensive configuration management for hybrid AI processing,
//! supporting both neural and symbolic components with adaptive tuning.

use crate::neural_symbolic_bridge::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::Duration;

/// Main configuration for neural-symbolic bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSymbolicConfig {
    /// Vector processing configuration
    pub vector_config: VectorProcessingConfig,
    /// Symbolic reasoning configuration
    pub symbolic_config: SymbolicReasoningConfig,
    /// Integration strategy configuration
    pub integration_config: IntegrationConfig,
    /// Performance and resource limits
    pub performance_config: PerformanceConfig,
    /// Temporal reasoning configuration
    pub temporal_config: TemporalConfig,
    /// Learning and adaptation settings
    pub learning_config: LearningConfig,
}

/// Vector processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorProcessingConfig {
    /// Embedding model path or identifier
    pub embedding_model: String,
    /// Vector dimension
    pub vector_dimension: usize,
    /// Similarity threshold for vector matching
    pub similarity_threshold: f32,
    /// Maximum number of similar vectors to consider
    pub max_similar_vectors: usize,
    /// Enable GPU acceleration for vector operations
    pub enable_gpu: bool,
    /// Vector index type (HNSW, IVF, etc.)
    pub index_type: VectorIndexType,
    /// Quantization settings
    pub quantization: QuantizationConfig,
}

/// Symbolic reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicReasoningConfig {
    /// SPARQL endpoint URL
    pub sparql_endpoint: Option<String>,
    /// Default reasoning depth
    pub default_reasoning_depth: u32,
    /// Maximum inference steps
    pub max_inference_steps: usize,
    /// Enable rule-based reasoning
    pub enable_rules: bool,
    /// Ontology paths for reasoning
    pub ontology_paths: Vec<PathBuf>,
    /// Inference engine type
    pub inference_engine: InferenceEngineType,
}

/// Integration strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// How to combine neural and symbolic results
    pub integration_strategy: IntegrationStrategy,
    /// Weights for combining scores
    pub score_weights: ScoreWeights,
    /// Confidence thresholds
    pub confidence_thresholds: ConfidenceThresholds,
    /// Explanation generation settings
    pub explanation_config: ExplanationConfig,
}

/// Performance and resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Maximum CPU threads
    pub max_cpu_threads: usize,
    /// Request timeout duration
    pub request_timeout: Duration,
    /// Cache size for intermediate results
    pub cache_size: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Batch processing size
    pub batch_size: usize,
}

/// Temporal reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Enable temporal reasoning
    pub enable_temporal: bool,
    /// Default time horizon for predictions
    pub default_horizon: Duration,
    /// Temporal granularity
    pub time_granularity: TimeGranularity,
    /// Historical data retention period
    pub history_retention: Duration,
    /// Enable temporal caching
    pub enable_temporal_cache: bool,
}

/// Learning and adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable adaptive learning
    pub enable_learning: bool,
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Minimum samples for learning
    pub min_samples: usize,
    /// Model update frequency
    pub update_frequency: Duration,
    /// Enable online learning
    pub enable_online_learning: bool,
    /// Feature importance tracking
    pub track_feature_importance: bool,
}

/// Vector index types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorIndexType {
    HNSW,
    IVF,
    LSH,
    PQ,
    FAISS,
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Enable quantization
    pub enabled: bool,
    /// Quantization bits
    pub bits: u8,
    /// Quantization method
    pub method: QuantizationMethod,
}

/// Quantization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMethod {
    Product,
    Scalar,
    Binary,
}

/// Inference engine types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceEngineType {
    Forward,
    Backward,
    Hybrid,
    RETE,
}

/// Score combination weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreWeights {
    /// Neural similarity weight
    pub neural_weight: f32,
    /// Symbolic reasoning weight
    pub symbolic_weight: f32,
    /// Temporal relevance weight
    pub temporal_weight: f32,
    /// Confidence weight
    pub confidence_weight: f32,
}

/// Confidence thresholds for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholds {
    /// Minimum confidence for neural results
    pub neural_min: f32,
    /// Minimum confidence for symbolic results
    pub symbolic_min: f32,
    /// Minimum confidence for combined results
    pub combined_min: f32,
    /// Threshold for explanation generation
    pub explanation_threshold: f32,
}

/// Explanation generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationConfig {
    /// Enable explanation generation
    pub enabled: bool,
    /// Maximum explanation depth
    pub max_depth: u32,
    /// Include provenance information
    pub include_provenance: bool,
    /// Include confidence scores
    pub include_confidence: bool,
    /// Explanation format
    pub format: ExplanationFormat,
}

/// Explanation formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationFormat {
    Natural,
    Structured,
    Graph,
    JSON,
}

/// Time granularity for temporal reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeGranularity {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

impl Default for NeuralSymbolicConfig {
    fn default() -> Self {
        Self {
            vector_config: VectorProcessingConfig::default(),
            symbolic_config: SymbolicReasoningConfig::default(),
            integration_config: IntegrationConfig::default(),
            performance_config: PerformanceConfig::default(),
            temporal_config: TemporalConfig::default(),
            learning_config: LearningConfig::default(),
        }
    }
}

impl Default for VectorProcessingConfig {
    fn default() -> Self {
        Self {
            embedding_model: "all-MiniLM-L6-v2".to_string(),
            vector_dimension: 384,
            similarity_threshold: 0.7,
            max_similar_vectors: 100,
            enable_gpu: false,
            index_type: VectorIndexType::HNSW,
            quantization: QuantizationConfig::default(),
        }
    }
}

impl Default for SymbolicReasoningConfig {
    fn default() -> Self {
        Self {
            sparql_endpoint: None,
            default_reasoning_depth: 3,
            max_inference_steps: 1000,
            enable_rules: true,
            ontology_paths: Vec::new(),
            inference_engine: InferenceEngineType::Hybrid,
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            integration_strategy: IntegrationStrategy::Sequential,
            score_weights: ScoreWeights::default(),
            confidence_thresholds: ConfidenceThresholds::default(),
            explanation_config: ExplanationConfig::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 4096,
            max_cpu_threads: num_cpus::get(),
            request_timeout: Duration::seconds(30),
            cache_size: 10000,
            enable_monitoring: true,
            batch_size: 32,
        }
    }
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enable_temporal: false,
            default_horizon: Duration::days(30),
            time_granularity: TimeGranularity::Day,
            history_retention: Duration::days(365),
            enable_temporal_cache: true,
        }
    }
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enable_learning: true,
            learning_rate: 0.01,
            min_samples: 100,
            update_frequency: Duration::hours(24),
            enable_online_learning: false,
            track_feature_importance: true,
        }
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bits: 8,
            method: QuantizationMethod::Product,
        }
    }
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            neural_weight: 0.4,
            symbolic_weight: 0.4,
            temporal_weight: 0.1,
            confidence_weight: 0.1,
        }
    }
}

impl Default for ConfidenceThresholds {
    fn default() -> Self {
        Self {
            neural_min: 0.5,
            symbolic_min: 0.6,
            combined_min: 0.7,
            explanation_threshold: 0.8,
        }
    }
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 5,
            include_provenance: true,
            include_confidence: true,
            format: ExplanationFormat::Structured,
        }
    }
}

/// Configuration builder for easy setup
pub struct ConfigBuilder {
    config: NeuralSymbolicConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: NeuralSymbolicConfig::default(),
        }
    }

    /// Set vector model
    pub fn vector_model(mut self, model: &str) -> Self {
        self.config.vector_config.embedding_model = model.to_string();
        self
    }

    /// Set similarity threshold
    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.config.vector_config.similarity_threshold = threshold;
        self
    }

    /// Enable GPU acceleration
    pub fn enable_gpu(mut self, enable: bool) -> Self {
        self.config.vector_config.enable_gpu = enable;
        self
    }

    /// Set reasoning depth
    pub fn reasoning_depth(mut self, depth: u32) -> Self {
        self.config.symbolic_config.default_reasoning_depth = depth;
        self
    }

    /// Set integration strategy
    pub fn integration_strategy(mut self, strategy: IntegrationStrategy) -> Self {
        self.config.integration_config.integration_strategy = strategy;
        self
    }

    /// Enable temporal reasoning
    pub fn enable_temporal(mut self, enable: bool) -> Self {
        self.config.temporal_config.enable_temporal = enable;
        self
    }

    /// Enable learning
    pub fn enable_learning(mut self, enable: bool) -> Self {
        self.config.learning_config.enable_learning = enable;
        self
    }

    /// Build the configuration
    pub fn build(self) -> NeuralSymbolicConfig {
        self.config
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration validation
impl NeuralSymbolicConfig {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        // Validate vector config
        if self.vector_config.vector_dimension == 0 {
            return Err("Vector dimension must be greater than 0".to_string());
        }

        if self.vector_config.similarity_threshold < 0.0 || self.vector_config.similarity_threshold > 1.0 {
            return Err("Similarity threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate score weights sum to reasonable range
        let total_weight = self.integration_config.score_weights.neural_weight
            + self.integration_config.score_weights.symbolic_weight
            + self.integration_config.score_weights.temporal_weight
            + self.integration_config.score_weights.confidence_weight;

        if (total_weight - 1.0).abs() > 0.1 {
            return Err("Score weights should sum to approximately 1.0".to_string());
        }

        // Validate confidence thresholds
        let thresholds = &self.integration_config.confidence_thresholds;
        if thresholds.neural_min < 0.0 || thresholds.neural_min > 1.0 {
            return Err("Neural confidence threshold must be between 0.0 and 1.0".to_string());
        }

        if thresholds.symbolic_min < 0.0 || thresholds.symbolic_min > 1.0 {
            return Err("Symbolic confidence threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate learning config
        if self.learning_config.learning_rate <= 0.0 || self.learning_config.learning_rate > 1.0 {
            return Err("Learning rate must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }

    /// Load configuration from file
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: NeuralSymbolicConfig = toml::from_str(&content)?;
        config.validate().map_err(|e| e.into())?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = NeuralSymbolicConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .vector_model("bert-base")
            .similarity_threshold(0.8)
            .enable_gpu(true)
            .reasoning_depth(5)
            .enable_temporal(true)
            .build();

        assert_eq!(config.vector_config.embedding_model, "bert-base");
        assert_eq!(config.vector_config.similarity_threshold, 0.8);
        assert!(config.vector_config.enable_gpu);
        assert_eq!(config.symbolic_config.default_reasoning_depth, 5);
        assert!(config.temporal_config.enable_temporal);
    }

    #[test]
    fn test_invalid_vector_dimension() {
        let mut config = NeuralSymbolicConfig::default();
        config.vector_config.vector_dimension = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_similarity_threshold() {
        let mut config = NeuralSymbolicConfig::default();
        config.vector_config.similarity_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_learning_rate() {
        let mut config = NeuralSymbolicConfig::default();
        config.learning_config.learning_rate = 2.0;
        assert!(config.validate().is_err());
    }
}