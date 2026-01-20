//! Common types and configurations for meta-learning algorithms

use crate::{EmbeddingModel, Vector};
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use uuid::Uuid;
use scirs2_core::random::{thread_rng, Rng};

/// Configuration for meta-learning systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// MAML configuration
    pub maml_config: MAMLConfig,
    /// Reptile configuration
    pub reptile_config: ReptileConfig,
    /// Prototypical Networks configuration
    pub prototypical_config: PrototypicalConfig,
    /// Matching Networks configuration
    pub matching_config: MatchingConfig,
    /// Relation Networks configuration
    pub relation_config: RelationConfig,
    /// MANN configuration
    pub mann_config: MANNConfig,
    /// Task sampling configuration
    pub task_config: TaskSamplingConfig,
    /// Global meta-learning settings
    pub global_settings: GlobalMetaSettings,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            maml_config: MAMLConfig::default(),
            reptile_config: ReptileConfig::default(),
            prototypical_config: PrototypicalConfig::default(),
            matching_config: MatchingConfig::default(),
            relation_config: RelationConfig::default(),
            mann_config: MANNConfig::default(),
            task_config: TaskSamplingConfig::default(),
            global_settings: GlobalMetaSettings::default(),
        }
    }
}

/// Global meta-learning settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetaSettings {
    /// Number of meta-training episodes
    pub meta_episodes: usize,
    /// Meta-learning rate (outer loop)
    pub meta_learning_rate: f32,
    /// Task-specific learning rate (inner loop)
    pub task_learning_rate: f32,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
    /// Evaluation frequency
    pub eval_frequency: usize,
    /// Enable gradient clipping
    pub enable_gradient_clipping: bool,
    /// Gradient clipping threshold
    pub gradient_clip_value: f32,
    /// Enable early stopping
    pub enable_early_stopping: bool,
    /// Patience for early stopping
    pub early_stopping_patience: usize,
}

impl Default for GlobalMetaSettings {
    fn default() -> Self {
        Self {
            meta_episodes: 1000,
            meta_learning_rate: 0.001,
            task_learning_rate: 0.01,
            adaptation_steps: 5,
            eval_frequency: 100,
            enable_gradient_clipping: true,
            gradient_clip_value: 0.5,
            enable_early_stopping: true,
            early_stopping_patience: 50,
        }
    }
}

/// MAML Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MAMLConfig {
    /// Model architecture
    pub model_architecture: ModelArchitecture,
    /// Inner loop learning rate
    pub inner_lr: f32,
    /// Outer loop learning rate
    pub outer_lr: f32,
    /// Number of inner loop steps
    pub inner_steps: usize,
    /// Use first-order approximation
    pub first_order: bool,
    /// Allow unused parameters in backward pass
    pub allow_unused: bool,
}

impl Default for MAMLConfig {
    fn default() -> Self {
        Self {
            model_architecture: ModelArchitecture::default(),
            inner_lr: 0.01,
            outer_lr: 0.001,
            inner_steps: 5,
            first_order: false,
            allow_unused: true,
        }
    }
}

/// Model architecture specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: String,
    /// Use batch normalization
    pub use_batch_norm: bool,
    /// Dropout rate
    pub dropout_rate: f32,
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self {
            input_dim: 128,
            hidden_dims: vec![256, 256],
            output_dim: 128,
            activation: "relu".to_string(),
            use_batch_norm: true,
            dropout_rate: 0.1,
        }
    }
}

/// Model parameters container
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Weight matrices for each layer
    pub weights: Vec<Array2<f32>>,
    /// Bias vectors for each layer
    pub biases: Vec<Array1<f32>>,
    /// Batch normalization parameters
    pub batch_norm_params: Option<BatchNormParameters>,
}

/// Batch normalization parameters
#[derive(Debug, Clone)]
pub struct BatchNormParameters {
    /// Scale parameters
    pub scale: Vec<Array1<f32>>,
    /// Shift parameters
    pub shift: Vec<Array1<f32>>,
    /// Running mean
    pub running_mean: Vec<Array1<f32>>,
    /// Running variance
    pub running_var: Vec<Array1<f32>>,
}

/// Meta-optimizer for parameter updates
#[derive(Debug)]
pub struct MetaOptimizer {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum coefficient
    pub momentum: f32,
    /// Momentum buffers
    pub momentum_buffers: Vec<Array2<f32>>,
    /// Bias momentum buffers
    pub bias_momentum_buffers: Vec<Array1<f32>>,
    /// Adam beta1 parameter
    pub beta1: f32,
    /// Adam beta2 parameter
    pub beta2: f32,
    /// Adam epsilon parameter
    pub epsilon: f32,
    /// Adam first moment estimates
    pub m_weights: Vec<Array2<f32>>,
    /// Adam second moment estimates
    pub v_weights: Vec<Array2<f32>>,
    /// Adam bias first moment estimates
    pub m_biases: Vec<Array1<f32>>,
    /// Adam bias second moment estimates
    pub v_biases: Vec<Array1<f32>>,
    /// Time step for Adam
    pub time_step: usize,
}

/// Adaptation result for tracking meta-learning progress
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Task identifier
    pub task_id: Uuid,
    /// Initial loss before adaptation
    pub initial_loss: f32,
    /// Final loss after adaptation
    pub final_loss: f32,
    /// Adaptation steps taken
    pub adaptation_steps: usize,
    /// Adaptation duration
    pub duration: Duration,
    /// Task metadata
    pub task_metadata: TaskMetadata,
}

/// Task metadata for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Task domain
    pub domain: String,
    /// Task difficulty
    pub difficulty: f32,
    /// Number of support examples
    pub support_size: usize,
    /// Number of query examples
    pub query_size: usize,
    /// Task creation timestamp
    pub created_at: Instant,
}

/// Reptile Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReptileConfig {
    /// Model architecture
    pub model_architecture: ModelArchitecture,
    /// Inner loop learning rate
    pub inner_lr: f32,
    /// Outer loop learning rate
    pub outer_lr: f32,
    /// Number of inner loop steps
    pub inner_steps: usize,
    /// Number of tasks per batch
    pub tasks_per_batch: usize,
}

impl Default for ReptileConfig {
    fn default() -> Self {
        Self {
            model_architecture: ModelArchitecture::default(),
            inner_lr: 0.01,
            outer_lr: 0.001,
            inner_steps: 10,
            tasks_per_batch: 5,
        }
    }
}

/// Prototypical Networks Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypicalConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of support examples per class
    pub support_size: usize,
    /// Number of query examples per class
    pub query_size: usize,
    /// Number of classes per episode
    pub num_classes: usize,
    /// Distance metric
    pub distance_metric: String,
    /// Temperature parameter for softmax
    pub temperature: f32,
}

impl Default for PrototypicalConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            support_size: 5,
            query_size: 15,
            num_classes: 5,
            distance_metric: "euclidean".to_string(),
            temperature: 1.0,
        }
    }
}

/// Matching Networks Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// LSTM hidden dimension
    pub lstm_hidden_dim: usize,
    /// Number of processing steps
    pub processing_steps: usize,
    /// Use full context embeddings
    pub use_full_context: bool,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            lstm_hidden_dim: 256,
            processing_steps: 5,
            use_full_context: true,
        }
    }
}

/// Relation Networks Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationConfig {
    /// Feature dimension
    pub feature_dim: usize,
    /// Relation module hidden dimensions
    pub relation_hidden_dims: Vec<usize>,
    /// Embedding module hidden dimensions
    pub embedding_hidden_dims: Vec<usize>,
    /// Dropout rate
    pub dropout_rate: f32,
}

impl Default for RelationConfig {
    fn default() -> Self {
        Self {
            feature_dim: 128,
            relation_hidden_dims: vec![256, 128, 64, 1],
            embedding_hidden_dims: vec![128, 128, 128, 128],
            dropout_rate: 0.1,
        }
    }
}

/// Memory-Augmented Neural Networks Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MANNConfig {
    /// Memory size
    pub memory_size: usize,
    /// Memory vector dimension
    pub memory_dim: usize,
    /// Controller hidden dimension
    pub controller_hidden_dim: usize,
    /// Number of read heads
    pub num_read_heads: usize,
    /// Number of write heads
    pub num_write_heads: usize,
    /// Memory initialization strategy
    pub memory_init: String,
}

impl Default for MANNConfig {
    fn default() -> Self {
        Self {
            memory_size: 128,
            memory_dim: 64,
            controller_hidden_dim: 256,
            num_read_heads: 4,
            num_write_heads: 1,
            memory_init: "random".to_string(),
        }
    }
}

/// Task sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSamplingConfig {
    /// Minimum number of support examples
    pub min_support: usize,
    /// Maximum number of support examples
    pub max_support: usize,
    /// Minimum number of query examples
    pub min_query: usize,
    /// Maximum number of query examples
    pub max_query: usize,
    /// Task difficulty sampling strategy
    pub difficulty_sampling: String,
    /// Domain distribution weights
    pub domain_weights: HashMap<String, f32>,
}

impl Default for TaskSamplingConfig {
    fn default() -> Self {
        Self {
            min_support: 1,
            max_support: 10,
            min_query: 5,
            max_query: 20,
            difficulty_sampling: "uniform".to_string(),
            domain_weights: HashMap::new(),
        }
    }
}

/// Meta-learning history tracking
#[derive(Debug)]
pub struct MetaLearningHistory {
    /// Episode results
    pub episodes: Vec<EpisodeResult>,
    /// Performance metrics over time
    pub performance_history: Vec<PerformanceSnapshot>,
    /// Task distribution statistics
    pub task_statistics: TaskStatistics,
}

/// Episode result for tracking
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    /// Episode number
    pub episode: usize,
    /// Average loss across tasks
    pub avg_loss: f32,
    /// Average accuracy across tasks
    pub avg_accuracy: f32,
    /// Task results
    pub task_results: Vec<TaskResult>,
    /// Episode duration
    pub duration: Duration,
}

/// Individual task result
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: Uuid,
    /// Task loss
    pub loss: f32,
    /// Task accuracy
    pub accuracy: f32,
    /// Adaptation steps
    pub adaptation_steps: usize,
    /// Task metadata
    pub metadata: TaskMetadata,
}

/// Performance snapshot for tracking
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Average loss
    pub avg_loss: f32,
    /// Average accuracy
    pub avg_accuracy: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Memory usage
    pub memory_usage: usize,
}

/// Task statistics
#[derive(Debug)]
pub struct TaskStatistics {
    /// Domain distribution
    pub domain_distribution: HashMap<String, usize>,
    /// Difficulty distribution
    pub difficulty_distribution: HashMap<String, usize>,
    /// Success rate by domain
    pub success_rate_by_domain: HashMap<String, f32>,
    /// Average adaptation time by domain
    pub avg_adaptation_time: HashMap<String, Duration>,
}

/// Meta-performance metrics
#[derive(Debug)]
pub struct MetaPerformanceMetrics {
    /// Current meta-learning rate
    pub current_meta_lr: f32,
    /// Best validation accuracy
    pub best_validation_accuracy: f32,
    /// Best validation loss
    pub best_validation_loss: f32,
    /// Episodes without improvement
    pub episodes_without_improvement: usize,
    /// Total training time
    pub total_training_time: Duration,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Memory usage statistics
#[derive(Debug)]
pub struct MemoryStats {
    /// Current memory usage
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Average memory usage
    pub avg_usage: f32,
    /// Memory allocation count
    pub allocation_count: usize,
}

/// Optimizer types for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Momentum,
}

/// Difficulty distribution for task sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyDistribution {
    Uniform,
    Exponential,
    Gaussian,
    Beta,
}

/// Data generators for synthetic tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataGenerator {
    Sinusoidal,
    Linear,
    Polynomial,
    Gaussian,
    Categorical,
}

/// Data point for meta-learning tasks
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Input features
    pub input: Array1<f32>,
    /// Target output
    pub target: Array1<f32>,
    /// Optional metadata
    pub metadata: Option<DataPointMetadata>,
}

/// Metadata for data points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPointMetadata {
    /// Data point identifier
    pub id: Uuid,
    /// Source information
    pub source: String,
    /// Quality score
    pub quality: f32,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// Task structure for meta-learning
#[derive(Debug, Clone)]
pub struct Task {
    /// Task identifier
    pub id: Uuid,
    /// Task type description
    pub task_type: String,
    /// Support set (training examples)
    pub support_set: Vec<DataPoint>,
    /// Query set (test examples)
    pub query_set: Vec<DataPoint>,
    /// Task metadata
    pub metadata: TaskMetadata,
}

/// Meta-learning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningResult {
    /// Average loss across tasks
    pub average_loss: f32,
    /// Individual adaptation results
    pub adaptation_results: Vec<AdaptationResult>,
    /// Convergence metric
    pub convergence_metric: f32,
}

/// Model gradients
#[derive(Debug, Clone)]
pub struct ModelGradients {
    /// Weight gradients
    pub weight_gradients: Vec<Array2<f32>>,
    /// Bias gradients
    pub bias_gradients: Vec<Array1<f32>>,
    /// Batch normalization gradients
    pub batch_norm_gradients: Option<BatchNormGradients>,
}

/// Batch normalization gradients
#[derive(Debug, Clone)]
pub struct BatchNormGradients {
    /// Gamma gradients
    pub gamma_gradients: Vec<Array1<f32>>,
    /// Beta gradients
    pub beta_gradients: Vec<Array1<f32>>,
}

impl ModelParameters {
    pub fn new(architecture: &ModelArchitecture) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = rand::rng();
        
        let mut dims = vec![architecture.input_dim];
        dims.extend(&architecture.hidden_dims);
        dims.push(architecture.output_dim);
        
        for i in 0..dims.len() - 1 {
            let input_dim = dims[i];
            let output_dim = dims[i + 1];
            
            // Xavier initialization
            let std = (2.0 / (input_dim + output_dim) as f32).sqrt();
            let weight = Array2::from_shape_fn((output_dim, input_dim), |_| {
                rng.uniform(-std, std)
            });
            let bias = Array1::zeros(output_dim);
            
            weights.push(weight);
            biases.push(bias);
        }
        
        let batch_norm_params = if architecture.use_batch_norm {
            let mut scale = Vec::new();
            let mut shift = Vec::new();
            let mut running_mean = Vec::new();
            let mut running_var = Vec::new();
            
            for &dim in &architecture.hidden_dims {
                scale.push(Array1::ones(dim));
                shift.push(Array1::zeros(dim));
                running_mean.push(Array1::zeros(dim));
                running_var.push(Array1::ones(dim));
            }
            
            Some(BatchNormParameters {
                scale,
                shift,
                running_mean,
                running_var,
            })
        } else {
            None
        };
        
        Self {
            weights,
            biases,
            batch_norm_params,
        }
    }

    /// Clone parameters for adaptation
    pub fn clone_for_adaptation(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            batch_norm_params: self.batch_norm_params.clone(),
        }
    }

    /// Update parameters with gradients
    pub fn update_with_gradients(&mut self, gradients: &ModelGradients, learning_rate: f32) {
        for (weight, grad_weight) in self.weights.iter_mut().zip(&gradients.weight_gradients) {
            *weight = &*weight - learning_rate * grad_weight;
        }
        
        for (bias, grad_bias) in self.biases.iter_mut().zip(&gradients.bias_gradients) {
            *bias = &*bias - learning_rate * grad_bias;
        }
        
        if let (Some(bn_params), Some(bn_grads)) = (&mut self.batch_norm_params, &gradients.batch_norm_gradients) {
            for (scale, grad_scale) in bn_params.scale.iter_mut().zip(&bn_grads.gamma_gradients) {
                *scale = &*scale - learning_rate * grad_scale;
            }
            
            for (shift, grad_shift) in bn_params.shift.iter_mut().zip(&bn_grads.beta_gradients) {
                *shift = &*shift - learning_rate * grad_shift;
            }
        }
    }
}