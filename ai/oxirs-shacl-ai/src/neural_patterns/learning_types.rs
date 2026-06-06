//! Learning config, model types, dataset types, training state types.

use scirs2_core::ndarray_ext::{Array1, Array2};
use std::collections::HashMap;

/// Neural network weights and biases
#[derive(Debug)]
pub struct NetworkWeights {
    /// Embedding layer weights
    pub embedding_weights: Array2<f64>,
    /// Attention layer weights
    pub attention_weights: HashMap<String, Array2<f64>>,
    /// Classification layer weights
    pub classification_weights: Array2<f64>,
    /// Layer biases
    pub biases: HashMap<String, Array1<f64>>,
}

/// Optimizer state for training
#[derive(Debug)]
pub struct OptimizerState {
    /// Momentum vectors for SGD with momentum
    pub momentum: HashMap<String, Array2<f64>>,
    /// Squared gradients for Adam optimizer
    pub squared_gradients: HashMap<String, Array2<f64>>,
    /// Bias correction terms
    pub bias_correction_1: f64,
    pub bias_correction_2: f64,
    /// Current training step
    pub step: usize,
}

/// Training history and metrics
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Loss values per epoch
    pub loss_history: Vec<f64>,
    /// Accuracy values per epoch
    pub accuracy_history: Vec<f64>,
    /// Validation loss per epoch
    pub validation_loss_history: Vec<f64>,
    /// Learning rate schedule
    pub learning_rate_history: Vec<f64>,
    /// Training time per epoch
    pub epoch_times: Vec<std::time::Duration>,
}

impl NetworkWeights {
    /// Initialize network weights using Xavier initialization
    pub fn new(config: &super::types::NeuralPatternConfig) -> Self {
        let embedding_weights = Self::xavier_init(config.embedding_dim, config.embedding_dim);
        let classification_weights = Self::xavier_init(config.embedding_dim, 10);

        let mut attention_weights = HashMap::new();
        for head in 0..config.attention_heads {
            let head_name = format!("attention_head_{head}");
            let head_dim = config.embedding_dim / config.attention_heads;
            attention_weights.insert(head_name, Self::xavier_init(head_dim, head_dim));
        }

        Self {
            embedding_weights,
            attention_weights,
            classification_weights,
            biases: HashMap::new(),
        }
    }

    /// Xavier weight initialization
    pub fn xavier_init(input_dim: usize, output_dim: usize) -> Array2<f64> {
        use scirs2_core::random::{Random, RngExt};

        let bound = (6.0 / (input_dim + output_dim) as f64).sqrt();
        let mut rng = Random::default();
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-bound..bound))
    }

    /// Clone network weights for meta-learning
    pub fn clone_weights(&self) -> NetworkWeights {
        NetworkWeights {
            embedding_weights: self.embedding_weights.clone(),
            attention_weights: self.attention_weights.clone(),
            classification_weights: self.classification_weights.clone(),
            biases: self.biases.clone(),
        }
    }
}

impl OptimizerState {
    /// Initialize optimizer state
    pub fn new() -> Self {
        Self {
            momentum: HashMap::new(),
            squared_gradients: HashMap::new(),
            bias_correction_1: 0.9,
            bias_correction_2: 0.999,
            step: 0,
        }
    }
}
