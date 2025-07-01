//! Architecture definitions and related structures

use crate::neural_architecture_search::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Architecture representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    /// Unique identifier
    pub id: Uuid,
    /// Network layers
    pub layers: Vec<LayerConfig>,
    /// Global architecture parameters
    pub global_config: GlobalArchConfig,
    /// Performance metrics
    pub performance: Option<PerformanceMetrics>,
    /// Generation number
    pub generation: usize,
    /// Parent architectures (for tracking lineage)
    pub parents: Vec<Uuid>,
}

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer type and parameters
    pub layer_type: LayerType,
    /// Activation function
    pub activation: ActivationType,
    /// Normalization
    pub normalization: NormalizationType,
    /// Skip connections
    pub skip_pattern: SkipPattern,
    /// Layer-specific hyperparameters
    pub hyperparameters: HashMap<String, f64>,
}

/// Global architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalArchConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output embedding dimension
    pub output_dim: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Regularization parameters
    pub regularization: RegularizationConfig,
    /// Training configuration
    pub training_config: TrainingConfig,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizerType {
    Adam { beta1: f64, beta2: f64, eps: f64 },
    AdamW { beta1: f64, beta2: f64, eps: f64, weight_decay: f64 },
    SGD { momentum: f64 },
    RMSprop { alpha: f64, eps: f64 },
    Lion { beta1: f64, beta2: f64 },
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization weight
    pub l1_weight: f64,
    /// L2 regularization weight
    pub l2_weight: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Label smoothing
    pub label_smoothing: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Validation split
    pub validation_split: f64,
    /// Learning rate schedule
    pub lr_schedule: LRScheduleType,
    /// Loss function
    pub loss_function: LossFunction,
}

/// Learning rate schedule types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LRScheduleType {
    Constant,
    StepLR { step_size: usize, gamma: f64 },
    ExponentialLR { gamma: f64 },
    CosineAnnealingLR { t_max: usize },
    ReduceLROnPlateau { factor: f64, patience: usize },
    WarmupCosine { warmup_epochs: usize },
}

/// Loss function types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LossFunction {
    MSE,
    CosineSimilarity,
    TripletLoss { margin: f64 },
    ContrastiveLoss { margin: f64 },
    InfoNCE { temperature: f64 },
    ArcFace { scale: f64, margin: f64 },
}

/// Performance metrics for architecture evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Embedding quality score
    pub embedding_quality: f64,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Inference latency in milliseconds
    pub inference_latency_ms: f64,
    /// Model size in parameters
    pub model_size_params: usize,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// FLOPs count
    pub flops: u64,
    /// Training time in minutes
    pub training_time_minutes: f64,
    /// Energy consumption
    pub energy_consumption: f64,
    /// Task-specific metrics
    pub task_metrics: HashMap<String, f64>,
}

/// Architecture search space definition
#[derive(Debug, Clone)]
pub struct ArchitectureSearchSpace {
    /// Available layer types
    pub layer_types: Vec<LayerType>,
    /// Depth range (min, max)
    pub depth_range: (usize, usize),
    /// Width range (min, max)
    pub width_range: (usize, usize),
    /// Available activation functions
    pub activations: Vec<ActivationType>,
    /// Available normalization types
    pub normalizations: Vec<NormalizationType>,
    /// Available attention mechanisms
    pub attention_types: Vec<AttentionType>,
    /// Skip connection patterns
    pub skip_patterns: Vec<SkipPattern>,
    /// Embedding dimension options
    pub embedding_dims: Vec<usize>,
}

impl Architecture {
    /// Create a new architecture
    pub fn new(layers: Vec<LayerConfig>, global_config: GlobalArchConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            layers,
            global_config,
            performance: None,
            generation: 0,
            parents: Vec::new(),
        }
    }

    /// Estimate model complexity
    pub fn estimate_complexity(&self) -> usize {
        self.layers.iter().map(|layer| {
            match &layer.layer_type {
                LayerType::Linear { input_dim, output_dim } => input_dim * output_dim,
                LayerType::Conv1D { filters, kernel_size, .. } => filters * kernel_size,
                LayerType::LSTM { hidden_size, num_layers } => hidden_size * num_layers * 4,
                LayerType::GRU { hidden_size, num_layers } => hidden_size * num_layers * 3,
                LayerType::Transformer { d_model, num_heads, num_layers } => {
                    d_model * d_model * num_heads * num_layers
                },
                _ => 1000, // Default complexity estimate
            }
        }).sum()
    }
}

impl Default for ArchitectureSearchSpace {
    fn default() -> Self {
        Self {
            layer_types: vec![
                LayerType::Linear { input_dim: 512, output_dim: 256 },
                LayerType::LSTM { hidden_size: 256, num_layers: 2 },
                LayerType::Transformer { d_model: 256, num_heads: 8, num_layers: 4 },
            ],
            depth_range: (2, 12),
            width_range: (64, 1024),
            activations: vec![ActivationType::ReLU, ActivationType::GELU, ActivationType::Swish],
            normalizations: vec![NormalizationType::LayerNorm, NormalizationType::BatchNorm],
            attention_types: vec![AttentionType::MultiHead { num_heads: 8 }],
            skip_patterns: vec![SkipPattern::None, SkipPattern::Residual],
            embedding_dims: vec![128, 256, 512, 768, 1024],
        }
    }
}