//! Configuration types for neural cost estimation

use serde::{Deserialize, Serialize};

/// Configuration for neural cost estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCostEstimationConfig {
    /// Neural network architecture
    pub network_architecture: NetworkArchitecture,

    /// Feature extraction configuration
    pub feature_extraction: FeatureExtractionConfig,

    /// Historical data configuration
    pub historical_data: HistoricalDataConfig,

    /// Ensemble configuration
    pub ensemble: EnsembleConfig,

    /// Real-time adaptation settings
    pub realtime_adaptation: RealTimeAdaptationConfig,

    /// Uncertainty quantification settings
    pub uncertainty_quantification: UncertaintyConfig,

    /// Performance profiling settings
    pub performance_profiling: PerformanceProfilingConfig,
}

/// Neural network architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkArchitecture {
    /// Input dimension
    pub input_dim: usize,

    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,

    /// Output dimension
    pub output_dim: usize,

    /// Activation function
    pub activation: ActivationFunction,

    /// Dropout rate
    pub dropout_rate: f64,

    /// Use batch normalization
    pub use_batch_norm: bool,

    /// Use residual connections
    pub use_residual: bool,

    /// Use attention mechanism
    pub use_attention: bool,

    /// Learning rate
    pub learning_rate: f64,

    /// L2 regularization
    pub l2_regularization: f64,
}

/// Activation function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU { alpha: f64 },
    ELU { alpha: f64 },
    Swish,
    GELU,
    Tanh,
    Sigmoid,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    /// Pattern structure features
    pub pattern_structure: bool,

    /// Index usage features
    pub index_usage: bool,

    /// Join complexity features
    pub join_complexity: bool,

    /// Selectivity features
    pub selectivity_features: bool,

    /// Historical performance features
    pub historical_performance: bool,

    /// Context features
    pub context_features: bool,

    /// Temporal features
    pub temporal_features: bool,

    /// System resource features
    pub system_resource: bool,

    /// Data characteristics features
    pub data_characteristics: bool,

    /// Query complexity features
    pub query_complexity: bool,

    /// Feature dimension
    pub total_feature_dim: usize,
}

/// Historical data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDataConfig {
    /// Maximum history size
    pub max_history_size: usize,

    /// Data retention period (days)
    pub retention_period_days: usize,

    /// Similarity threshold for pattern matching
    pub similarity_threshold: f64,

    /// Enable data compression
    pub enable_compression: bool,

    /// Enable periodic cleanup
    pub enable_cleanup: bool,

    /// Cleanup interval (hours)
    pub cleanup_interval_hours: usize,
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Number of base models
    pub num_base_models: usize,

    /// Ensemble strategy
    pub ensemble_strategy: EnsembleStrategy,

    /// Model diversity requirement
    pub diversity_threshold: f64,

    /// Enable model selection
    pub enable_model_selection: bool,

    /// Model update frequency
    pub model_update_frequency: usize,
}

/// Ensemble strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    Averaging,
    WeightedAveraging,
    Voting,
    Stacking,
    Bagging,
    Boosting,
    AdaptiveWeighting,
}

/// Real-time adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeAdaptationConfig {
    /// Enable online learning
    pub enable_online_learning: bool,

    /// Adaptation learning rate
    pub adaptation_rate: f64,

    /// Minimum samples for adaptation
    pub min_samples_for_adaptation: usize,

    /// Adaptation frequency
    pub adaptation_frequency: usize,

    /// Performance degradation threshold
    pub degradation_threshold: f64,
}

/// Uncertainty quantification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyConfig {
    /// Enable uncertainty estimation
    pub enable_uncertainty: bool,

    /// Uncertainty method
    pub uncertainty_method: UncertaintyMethod,

    /// Confidence intervals
    pub confidence_levels: Vec<f64>,

    /// Bootstrap samples for uncertainty
    pub bootstrap_samples: usize,
}

/// Uncertainty estimation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyMethod {
    Bootstrap,
    Bayesian,
    Ensemble,
    Dropout,
    Gaussian,
}

/// Performance profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfilingConfig {
    /// Enable detailed profiling
    pub enable_profiling: bool,

    /// Profiling granularity
    pub granularity: ProfilingGranularity,

    /// Resource monitoring
    pub monitor_resources: bool,

    /// Cache analysis
    pub analyze_cache: bool,

    /// I/O analysis
    pub analyze_io: bool,
}

/// Profiling granularity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfilingGranularity {
    Coarse,
    Medium,
    Fine,
    UltraFine,
}

impl Default for NeuralCostEstimationConfig {
    fn default() -> Self {
        Self {
            network_architecture: NetworkArchitecture {
                input_dim: 100,
                hidden_dims: vec![512, 256, 128, 64],
                output_dim: 1,
                activation: ActivationFunction::ReLU,
                dropout_rate: 0.1,
                use_batch_norm: true,
                use_residual: true,
                use_attention: true,
                learning_rate: 0.001,
                l2_regularization: 0.0001,
            },
            feature_extraction: FeatureExtractionConfig {
                pattern_structure: true,
                index_usage: true,
                join_complexity: true,
                selectivity_features: true,
                historical_performance: true,
                context_features: true,
                temporal_features: true,
                system_resource: true,
                data_characteristics: true,
                query_complexity: true,
                total_feature_dim: 100,
            },
            historical_data: HistoricalDataConfig {
                max_history_size: 10000,
                retention_period_days: 30,
                similarity_threshold: 0.8,
                enable_compression: true,
                enable_cleanup: true,
                cleanup_interval_hours: 24,
            },
            ensemble: EnsembleConfig {
                num_base_models: 5,
                ensemble_strategy: EnsembleStrategy::AdaptiveWeighting,
                diversity_threshold: 0.1,
                enable_model_selection: true,
                model_update_frequency: 100,
            },
            realtime_adaptation: RealTimeAdaptationConfig {
                enable_online_learning: true,
                adaptation_rate: 0.01,
                min_samples_for_adaptation: 50,
                adaptation_frequency: 10,
                degradation_threshold: 0.1,
            },
            uncertainty_quantification: UncertaintyConfig {
                enable_uncertainty: true,
                uncertainty_method: UncertaintyMethod::Ensemble,
                confidence_levels: vec![0.95, 0.99],
                bootstrap_samples: 1000,
            },
            performance_profiling: PerformanceProfilingConfig {
                enable_profiling: true,
                granularity: ProfilingGranularity::Medium,
                monitor_resources: true,
                analyze_cache: true,
                analyze_io: true,
            },
        }
    }
}