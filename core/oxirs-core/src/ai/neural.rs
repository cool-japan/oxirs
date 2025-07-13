//! Neural Network Utilities and Components
//!
//! This module provides common neural network utilities, activation functions,
//! and building blocks used across different AI models.

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Neural network layer trait
pub trait NeuralLayer: Send + Sync {
    /// Forward pass
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>>;

    /// Get layer parameters
    fn parameters(&self) -> Vec<Array2<f32>>;

    /// Set layer parameters
    fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()>;

    /// Get layer name
    fn name(&self) -> &str;
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU { negative_slope: f32 },
    ELU { alpha: f32 },
    SELU,
    GELU,
    Swish,
    Mish,
    Tanh,
    Sigmoid,
    Softmax,
    Softplus,
    Softsign,
    HardTanh,
    Identity,
}

/// Apply activation function to array
pub fn apply_activation(x: &Array2<f32>, activation: &ActivationFunction) -> Array2<f32> {
    match activation {
        ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
        ActivationFunction::LeakyReLU { negative_slope } => {
            x.mapv(|v| if v > 0.0 { v } else { v * negative_slope })
        }
        ActivationFunction::ELU { alpha } => {
            x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        }
        ActivationFunction::SELU => {
            let alpha = 1.673_263_2;
            let scale = 1.050_701;
            x.mapv(|v| scale * if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        }
        ActivationFunction::GELU => {
            x.mapv(|v| 0.5 * v * (1.0 + (v * 0.797_884_6 * (1.0 + 0.044715 * v * v)).tanh()))
        }
        ActivationFunction::Swish => x.mapv(|v| v * (1.0 / (1.0 + (-v).exp()))),
        ActivationFunction::Mish => x.mapv(|v| v * (1.0 + (-v).exp()).ln().tanh()),
        ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
        ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
        ActivationFunction::Softmax => {
            let mut result = x.clone();
            for mut row in result.axis_iter_mut(Axis(0)) {
                let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                row.mapv_inplace(|v| (v - max_val).exp());
                let sum = row.sum();
                row.mapv_inplace(|v| v / sum);
            }
            result
        }
        ActivationFunction::Softplus => x.mapv(|v| (1.0 + v.exp()).ln()),
        ActivationFunction::Softsign => x.mapv(|v| v / (1.0 + v.abs())),
        ActivationFunction::HardTanh => x.mapv(|v| v.clamp(-1.0, 1.0)),
        ActivationFunction::Identity => x.clone(),
    }
}

/// Linear layer (fully connected)
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Layer name
    name: String,

    /// Weight matrix
    weight: Array2<f32>,

    /// Bias vector
    bias: Array1<f32>,

    /// Input dimension
    #[allow(dead_code)]
    input_dim: usize,

    /// Output dimension
    output_dim: usize,
}

impl LinearLayer {
    /// Create new linear layer
    pub fn new(name: String, input_dim: usize, output_dim: usize) -> Self {
        // Xavier initialization
        let bound = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let weight = Array2::from_shape_simple_fn((input_dim, output_dim), || {
            rand::random::<f32>() * 2.0 * bound - bound
        });
        let bias = Array1::zeros(output_dim);

        Self {
            name,
            weight,
            bias,
            input_dim,
            output_dim,
        }
    }
}

impl NeuralLayer for LinearLayer {
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let output = input.dot(&self.weight) + &self.bias;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Array2<f32>> {
        vec![
            self.weight.clone(),
            self.bias
                .clone()
                .to_shape((self.output_dim, 1))
                .unwrap()
                .to_owned(),
        ]
    }

    fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()> {
        if params.len() != 2 {
            return Err(anyhow::anyhow!("Linear layer expects 2 parameters"));
        }

        self.weight = params[0].clone();
        self.bias = params[1].clone().to_shape(self.output_dim)?.to_owned();

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Dropout layer
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    name: String,
    dropout_rate: f32,
    training: bool,
}

impl DropoutLayer {
    pub fn new(name: String, dropout_rate: f32) -> Self {
        Self {
            name,
            dropout_rate,
            training: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl NeuralLayer for DropoutLayer {
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.training || self.dropout_rate <= 0.0 {
            return Ok(input.clone());
        }

        let keep_prob = 1.0 - self.dropout_rate;
        let output = input.mapv(|v| {
            if rand::random::<f32>() < keep_prob {
                v / keep_prob
            } else {
                0.0
            }
        });

        Ok(output)
    }

    fn parameters(&self) -> Vec<Array2<f32>> {
        vec![] // Dropout has no parameters
    }

    fn set_parameters(&mut self, _params: &[Array2<f32>]) -> Result<()> {
        Ok(()) // No parameters to set
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Batch normalization layer
#[derive(Debug, Clone)]
pub struct BatchNormLayer {
    name: String,
    num_features: usize,
    gamma: Array1<f32>,
    beta: Array1<f32>,
    running_mean: Array1<f32>,
    running_var: Array1<f32>,
    #[allow(dead_code)]
    momentum: f32,
    eps: f32,
    training: bool,
}

impl BatchNormLayer {
    pub fn new(name: String, num_features: usize) -> Self {
        Self {
            name,
            num_features,
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl NeuralLayer for BatchNormLayer {
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (mean, var) = if self.training {
            // Compute batch statistics
            let batch_mean = input.mean_axis(Axis(0)).unwrap();
            let batch_var = input.var_axis(Axis(0), 0.0);

            // Update running statistics (in a real implementation)
            // self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean
            // self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * batch_var

            (batch_mean, batch_var)
        } else {
            // Use running statistics
            (self.running_mean.clone(), self.running_var.clone())
        };

        // Normalize
        let normalized = (input - &mean) / &var.mapv(|v| (v + self.eps).sqrt());

        // Scale and shift
        let output = &normalized * &self.gamma + &self.beta;

        Ok(output)
    }

    fn parameters(&self) -> Vec<Array2<f32>> {
        vec![
            self.gamma
                .clone()
                .to_shape((self.num_features, 1))
                .unwrap()
                .to_owned(),
            self.beta
                .clone()
                .to_shape((self.num_features, 1))
                .unwrap()
                .to_owned(),
        ]
    }

    fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()> {
        if params.len() != 2 {
            return Err(anyhow::anyhow!("BatchNorm layer expects 2 parameters"));
        }

        self.gamma = params[0].clone().to_shape(self.num_features)?.to_owned();
        self.beta = params[1].clone().to_shape(self.num_features)?.to_owned();

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Multi-head attention layer
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionLayer {
    name: String,
    #[allow(dead_code)]
    embed_dim: usize,
    #[allow(dead_code)]
    num_heads: usize,
    head_dim: usize,
    query_proj: LinearLayer,
    key_proj: LinearLayer,
    value_proj: LinearLayer,
    output_proj: LinearLayer,
    dropout: DropoutLayer,
}

impl MultiHeadAttentionLayer {
    pub fn new(name: String, embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert_eq!(
            embed_dim % num_heads,
            0,
            "embed_dim must be divisible by num_heads"
        );

        let head_dim = embed_dim / num_heads;

        Self {
            query_proj: LinearLayer::new(format!("{name}_query"), embed_dim, embed_dim),
            key_proj: LinearLayer::new(format!("{name}_key"), embed_dim, embed_dim),
            value_proj: LinearLayer::new(format!("{name}_value"), embed_dim, embed_dim),
            output_proj: LinearLayer::new(format!("{name}_output"), embed_dim, embed_dim),
            dropout: DropoutLayer::new(format!("{name}_dropout"), dropout),
            name,
            embed_dim,
            num_heads,
            head_dim,
        }
    }

    fn scaled_dot_product_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        // Compute attention scores
        let scores = query.dot(&key.t()) / (self.head_dim as f32).sqrt();

        // Apply softmax
        let attention_weights = apply_activation(&scores, &ActivationFunction::Softmax);

        // Apply dropout to attention weights
        let attention_weights = self.dropout.forward(&attention_weights)?;

        // Apply attention to values
        let output = attention_weights.dot(value);

        Ok(output)
    }
}

impl NeuralLayer for MultiHeadAttentionLayer {
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let _batch_size = input.nrows();

        // Project to query, key, value
        let query = self.query_proj.forward(input)?;
        let key = self.key_proj.forward(input)?;
        let value = self.value_proj.forward(input)?;

        // Reshape for multi-head attention
        // In a real implementation, would properly reshape for multiple heads
        let attention_output = self.scaled_dot_product_attention(&query, &key, &value)?;

        // Final output projection
        let output = self.output_proj.forward(&attention_output)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<Array2<f32>> {
        let mut params = Vec::new();
        params.extend(self.query_proj.parameters());
        params.extend(self.key_proj.parameters());
        params.extend(self.value_proj.parameters());
        params.extend(self.output_proj.parameters());
        params
    }

    fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()> {
        if params.len() != 8 {
            // 4 layers * 2 params each
            return Err(anyhow::anyhow!("MultiHeadAttention expects 8 parameters"));
        }

        self.query_proj.set_parameters(&params[0..2])?;
        self.key_proj.set_parameters(&params[2..4])?;
        self.value_proj.set_parameters(&params[4..6])?;
        self.output_proj.set_parameters(&params[6..8])?;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Neural network builder
pub struct NeuralNetworkBuilder {
    layers: Vec<Box<dyn NeuralLayer>>,
    name: String,
}

impl NeuralNetworkBuilder {
    pub fn new(name: String) -> Self {
        Self {
            layers: Vec::new(),
            name,
        }
    }

    pub fn add_linear(mut self, input_dim: usize, output_dim: usize) -> Self {
        let layer_name = format!("{}_linear_{}", self.name, self.layers.len());
        self.layers.push(Box::new(LinearLayer::new(
            layer_name, input_dim, output_dim,
        )));
        self
    }

    pub fn add_dropout(mut self, dropout_rate: f32) -> Self {
        let layer_name = format!("{}_dropout_{}", self.name, self.layers.len());
        self.layers
            .push(Box::new(DropoutLayer::new(layer_name, dropout_rate)));
        self
    }

    pub fn add_batch_norm(mut self, num_features: usize) -> Self {
        let layer_name = format!("{}_batchnorm_{}", self.name, self.layers.len());
        self.layers
            .push(Box::new(BatchNormLayer::new(layer_name, num_features)));
        self
    }

    pub fn add_attention(mut self, embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        let layer_name = format!("{}_attention_{}", self.name, self.layers.len());
        self.layers.push(Box::new(MultiHeadAttentionLayer::new(
            layer_name, embed_dim, num_heads, dropout,
        )));
        self
    }

    pub fn build(self) -> NeuralNetwork {
        NeuralNetwork {
            layers: self.layers,
            name: self.name,
        }
    }
}

/// Neural network container
pub struct NeuralNetwork {
    layers: Vec<Box<dyn NeuralLayer>>,
    name: String,
}

impl NeuralNetwork {
    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output)?;
        }

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<Array2<f32>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    pub fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()> {
        let mut param_idx = 0;

        for layer in &mut self.layers {
            let layer_params = layer.parameters();
            let num_params = layer_params.len();

            if param_idx + num_params > params.len() {
                return Err(anyhow::anyhow!("Not enough parameters provided"));
            }

            layer.set_parameters(&params[param_idx..param_idx + num_params])?;
            param_idx += num_params;
        }

        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Weight initialization strategies
#[derive(Debug, Clone)]
pub enum WeightInitialization {
    Xavier,
    Kaiming,
    Normal { mean: f32, std: f32 },
    Uniform { low: f32, high: f32 },
    Zeros,
    Ones,
}

/// Initialize weights according to strategy
pub fn initialize_weights(shape: (usize, usize), init: &WeightInitialization) -> Array2<f32> {
    match init {
        WeightInitialization::Xavier => {
            let bound = (6.0 / (shape.0 + shape.1) as f32).sqrt();
            Array2::from_shape_simple_fn(shape, || rand::random::<f32>() * 2.0 * bound - bound)
        }
        WeightInitialization::Kaiming => {
            let std = (2.0 / shape.0 as f32).sqrt();
            Array2::from_shape_simple_fn(shape, || {
                // Box-Muller transform for normal distribution
                let u1: f32 = rand::random();
                let u2: f32 = rand::random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                z * std
            })
        }
        WeightInitialization::Normal { mean, std } => Array2::from_shape_simple_fn(shape, || {
            let u1: f32 = rand::random();
            let u2: f32 = rand::random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z * std + mean
        }),
        WeightInitialization::Uniform { low, high } => {
            Array2::from_shape_simple_fn(shape, || rand::random::<f32>() * (high - low) + low)
        }
        WeightInitialization::Zeros => Array2::zeros(shape),
        WeightInitialization::Ones => Array2::ones(shape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        let input = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();

        let relu = apply_activation(&input, &ActivationFunction::ReLU);
        assert_eq!(relu[[0, 0]], 0.0);
        assert_eq!(relu[[1, 1]], 2.0);

        let sigmoid = apply_activation(&input, &ActivationFunction::Sigmoid);
        assert!(sigmoid[[0, 0]] > 0.0 && sigmoid[[0, 0]] < 1.0);
    }

    #[test]
    fn test_linear_layer() {
        let layer = LinearLayer::new("test".to_string(), 3, 2);
        let input = Array2::ones((4, 3)); // batch_size=4, input_dim=3

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 2]);
    }

    #[test]
    fn test_neural_network_builder() {
        let network = NeuralNetworkBuilder::new("test_network".to_string())
            .add_linear(10, 20)
            .add_dropout(0.1)
            .add_linear(20, 5)
            .build();

        let input = Array2::ones((4, 10));
        let output = network.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 5]);
    }

    #[test]
    fn test_weight_initialization() {
        let weights = initialize_weights((10, 20), &WeightInitialization::Xavier);
        assert_eq!(weights.shape(), &[10, 20]);

        let zeros = initialize_weights((5, 5), &WeightInitialization::Zeros);
        assert!(zeros.iter().all(|&x| x == 0.0));
    }
}
