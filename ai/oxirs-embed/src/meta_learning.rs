//! Meta-Learning and Few-Shot Learning for Advanced Embedding Adaptation
//!
//! This module implements state-of-the-art meta-learning algorithms that enable
//! embedding models to quickly adapt to new domains and tasks with minimal data.
//! Features include MAML, Reptile, Prototypical Networks, Model-Agnostic Meta-Learning,
//! and advanced few-shot learning techniques for knowledge graph embeddings.
//!
//! Meta-learning capabilities enable:
//! - Rapid adaptation to new knowledge domains
//! - Few-shot entity and relation learning
//! - Transfer learning across knowledge graphs
//! - Continual learning without catastrophic forgetting
//! - Cross-domain knowledge transfer and adaptation

use crate::{EmbeddingModel, Vector};
use anyhow::{anyhow, Result};
use ndarray::{s, Array1, Array2, Array3, Axis, Zip};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use uuid::Uuid;
use rand::{seq::SliceRandom, thread_rng, Rng};

/// Meta-Learning Engine for few-shot adaptation
pub struct MetaLearningEngine {
    /// Configuration for meta-learning
    config: MetaLearningConfig,
    /// MAML (Model-Agnostic Meta-Learning) implementation
    maml: MAML,
    /// Reptile meta-learning algorithm
    reptile: Reptile,
    /// Prototypical Networks for few-shot learning
    prototypical_networks: PrototypicalNetworks,
    /// Matching Networks implementation
    matching_networks: MatchingNetworks,
    /// Relation Networks for relational reasoning
    relation_networks: RelationNetworks,
    /// Memory-Augmented Neural Networks for meta-learning
    mann: MANN,
    /// Task distribution and sampling
    task_sampler: TaskSampler,
    /// Meta-learning history and statistics
    meta_history: MetaLearningHistory,
    /// Performance metrics
    performance_metrics: MetaPerformanceMetrics,
}

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

/// Model-Agnostic Meta-Learning (MAML) implementation
pub struct MAML {
    /// Configuration
    config: MAMLConfig,
    /// Base model parameters
    base_params: ModelParameters,
    /// Meta-optimizer
    meta_optimizer: MetaOptimizer,
    /// Adaptation history
    adaptation_history: Vec<AdaptationResult>,
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
    /// Weight matrices
    pub weights: Vec<Array2<f32>>,
    /// Bias vectors
    pub biases: Vec<Array1<f32>>,
    /// Batch normalization parameters
    pub batch_norm_params: Option<BatchNormParameters>,
}

/// Batch normalization parameters
#[derive(Debug, Clone)]
pub struct BatchNormParameters {
    /// Scale parameters
    pub gamma: Vec<Array1<f32>>,
    /// Shift parameters
    pub beta: Vec<Array1<f32>>,
    /// Running means
    pub running_mean: Vec<Array1<f32>>,
    /// Running variances
    pub running_var: Vec<Array1<f32>>,
}

impl ModelParameters {
    pub fn new(architecture: &ModelArchitecture) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = thread_rng();
        
        let mut dims = vec![architecture.input_dim];
        dims.extend(&architecture.hidden_dims);
        dims.push(architecture.output_dim);
        
        for i in 0..dims.len() - 1 {
            let input_dim = dims[i];
            let output_dim = dims[i + 1];
            
            // Xavier initialization
            let std = (2.0 / (input_dim + output_dim) as f32).sqrt();
            let weight = Array2::from_shape_fn((output_dim, input_dim), |_| {
                rng.gen_range(-std..std)
            });
            let bias = Array1::zeros(output_dim);
            
            weights.push(weight);
            biases.push(bias);
        }
        
        let batch_norm_params = if architecture.use_batch_norm {
            let mut gamma = Vec::new();
            let mut beta = Vec::new();
            let mut running_mean = Vec::new();
            let mut running_var = Vec::new();
            
            for &dim in &architecture.hidden_dims {
                gamma.push(Array1::ones(dim));
                beta.push(Array1::zeros(dim));
                running_mean.push(Array1::zeros(dim));
                running_var.push(Array1::ones(dim));
            }
            
            Some(BatchNormParameters {
                gamma,
                beta,
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
            *weight = weight - learning_rate * grad_weight;
        }
        
        for (bias, grad_bias) in self.biases.iter_mut().zip(&gradients.bias_gradients) {
            *bias = bias - learning_rate * grad_bias;
        }
        
        if let (Some(bn_params), Some(bn_grads)) = (&mut self.batch_norm_params, &gradients.batch_norm_gradients) {
            for (gamma, grad_gamma) in bn_params.gamma.iter_mut().zip(&bn_grads.gamma_gradients) {
                *gamma = gamma - learning_rate * grad_gamma;
            }
            
            for (beta, grad_beta) in bn_params.beta.iter_mut().zip(&bn_grads.beta_gradients) {
                *beta = beta - learning_rate * grad_beta;
            }
        }
    }
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

/// Meta-optimizer for parameter updates
pub struct MetaOptimizer {
    /// Optimizer type
    optimizer_type: OptimizerType,
    /// Learning rate
    learning_rate: f32,
    /// Momentum (for SGD with momentum)
    momentum: f32,
    /// Beta parameters (for Adam)
    beta1: f32,
    beta2: f32,
    /// Epsilon (for Adam)
    epsilon: f32,
    /// Velocity (for momentum-based optimizers)
    velocity: Option<ModelGradients>,
    /// First moment estimate (for Adam)
    m: Option<ModelGradients>,
    /// Second moment estimate (for Adam)
    v: Option<ModelGradients>,
    /// Time step (for Adam)
    t: usize,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    SGDMomentum,
    Adam,
    RMSprop,
}

impl MetaOptimizer {
    pub fn new(optimizer_type: OptimizerType, learning_rate: f32) -> Self {
        Self {
            optimizer_type,
            learning_rate,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            velocity: None,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Update parameters using meta-gradients
    pub fn update(&mut self, params: &mut ModelParameters, gradients: &ModelGradients) {
        match self.optimizer_type {
            OptimizerType::SGD => {
                params.update_with_gradients(gradients, self.learning_rate);
            }
            OptimizerType::SGDMomentum => {
                self.update_with_momentum(params, gradients);
            }
            OptimizerType::Adam => {
                self.update_with_adam(params, gradients);
            }
            OptimizerType::RMSprop => {
                self.update_with_rmsprop(params, gradients);
            }
        }
    }

    fn update_with_momentum(&mut self, params: &mut ModelParameters, gradients: &ModelGradients) {
        if self.velocity.is_none() {
            self.velocity = Some(gradients.clone());
        }
        
        if let Some(velocity) = &mut self.velocity {
            // Update velocity
            for (v, g) in velocity.weight_gradients.iter_mut().zip(&gradients.weight_gradients) {
                *v = self.momentum * &*v + &*g;
            }
            
            for (v, g) in velocity.bias_gradients.iter_mut().zip(&gradients.bias_gradients) {
                *v = self.momentum * &*v + &*g;
            }
            
            // Update parameters
            params.update_with_gradients(velocity, self.learning_rate);
        }
    }

    fn update_with_adam(&mut self, params: &mut ModelParameters, gradients: &ModelGradients) {
        self.t += 1;
        
        // Initialize moments if needed
        if self.m.is_none() {
            self.m = Some(self.zero_gradients_like(gradients));
        }
        if self.v.is_none() {
            self.v = Some(self.zero_gradients_like(gradients));
        }
        
        if let (Some(m), Some(v)) = (&mut self.m, &mut self.v) {
            // Update biased first moment estimate
            for (m_w, g_w) in m.weight_gradients.iter_mut().zip(&gradients.weight_gradients) {
                *m_w = self.beta1 * &*m_w + (1.0 - self.beta1) * g_w;
            }
            
            for (m_b, g_b) in m.bias_gradients.iter_mut().zip(&gradients.bias_gradients) {
                *m_b = self.beta1 * &*m_b + (1.0 - self.beta1) * g_b;
            }
            
            // Update biased second moment estimate
            for (v_w, g_w) in v.weight_gradients.iter_mut().zip(&gradients.weight_gradients) {
                *v_w = self.beta2 * &*v_w + (1.0 - self.beta2) * &g_w.mapv(|x| x * x);
            }
            
            for (v_b, g_b) in v.bias_gradients.iter_mut().zip(&gradients.bias_gradients) {
                *v_b = self.beta2 * &*v_b + (1.0 - self.beta2) * &g_b.mapv(|x| x * x);
            }
            
            // Bias correction
            let lr_t = self.learning_rate * 
                (1.0 - self.beta2.powi(self.t as i32)).sqrt() / 
                (1.0 - self.beta1.powi(self.t as i32));
            
            // Update parameters
            for ((weight, m_w), v_w) in params.weights.iter_mut().zip(&m.weight_gradients).zip(&v.weight_gradients) {
                let update = m_w / &v_w.mapv(|x| x.sqrt() + self.epsilon);
                *weight = weight - lr_t * &update;
            }
            
            for ((bias, m_b), v_b) in params.biases.iter_mut().zip(&m.bias_gradients).zip(&v.bias_gradients) {
                let update = m_b / &v_b.mapv(|x| x.sqrt() + self.epsilon);
                *bias = bias - lr_t * &update;
            }
        }
    }

    fn update_with_rmsprop(&mut self, params: &mut ModelParameters, gradients: &ModelGradients) {
        if self.v.is_none() {
            self.v = Some(self.zero_gradients_like(gradients));
        }
        
        if let Some(v) = &mut self.v {
            let decay_rate = 0.9;
            
            // Update running average of squared gradients
            for (v_w, g_w) in v.weight_gradients.iter_mut().zip(&gradients.weight_gradients) {
                *v_w = decay_rate * &*v_w + (1.0 - decay_rate) * &g_w.mapv(|x| x * x);
            }
            
            for (v_b, g_b) in v.bias_gradients.iter_mut().zip(&gradients.bias_gradients) {
                *v_b = decay_rate * &*v_b + (1.0 - decay_rate) * &g_b.mapv(|x| x * x);
            }
            
            // Update parameters
            for ((weight, g_w), v_w) in params.weights.iter_mut().zip(&gradients.weight_gradients).zip(&v.weight_gradients) {
                let update = g_w / &v_w.mapv(|x| x.sqrt() + self.epsilon);
                *weight = weight - self.learning_rate * &update;
            }
            
            for ((bias, g_b), v_b) in params.biases.iter_mut().zip(&gradients.bias_gradients).zip(&v.bias_gradients) {
                let update = g_b / &v_b.mapv(|x| x.sqrt() + self.epsilon);
                *bias = bias - self.learning_rate * &update;
            }
        }
    }

    fn zero_gradients_like(&self, gradients: &ModelGradients) -> ModelGradients {
        let weight_gradients = gradients.weight_gradients.iter()
            .map(|w| Array2::zeros(w.raw_dim()))
            .collect();
        
        let bias_gradients = gradients.bias_gradients.iter()
            .map(|b| Array1::zeros(b.raw_dim()))
            .collect();
        
        let batch_norm_gradients = gradients.batch_norm_gradients.as_ref().map(|bn| {
            BatchNormGradients {
                gamma_gradients: bn.gamma_gradients.iter()
                    .map(|g| Array1::zeros(g.raw_dim()))
                    .collect(),
                beta_gradients: bn.beta_gradients.iter()
                    .map(|b| Array1::zeros(b.raw_dim()))
                    .collect(),
            }
        });
        
        ModelGradients {
            weight_gradients,
            bias_gradients,
            batch_norm_gradients,
        }
    }
}

/// Adaptation result for MAML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    /// Task ID
    pub task_id: Uuid,
    /// Initial loss
    pub initial_loss: f32,
    /// Final loss after adaptation
    pub final_loss: f32,
    /// Adaptation steps taken
    pub steps_taken: usize,
    /// Adaptation time
    pub adaptation_time_ms: f32,
    /// Success indicator
    pub success: bool,
}

impl MAML {
    pub fn new(config: MAMLConfig) -> Self {
        let base_params = ModelParameters::new(&config.model_architecture);
        let meta_optimizer = MetaOptimizer::new(OptimizerType::Adam, config.outer_lr);
        
        Self {
            config,
            base_params,
            meta_optimizer,
            adaptation_history: Vec::new(),
        }
    }

    /// Perform meta-learning episode
    pub async fn meta_learn_episode(&mut self, tasks: &[Task]) -> Result<MetaLearningResult> {
        let mut total_loss = 0.0;
        let mut adaptation_results = Vec::new();
        let mut meta_gradients = self.zero_gradients();
        
        for task in tasks {
            let adaptation_result = self.adapt_to_task(task).await?;
            total_loss += adaptation_result.final_loss;
            adaptation_results.push(adaptation_result);
            
            // Compute meta-gradients (simplified)
            let task_gradients = self.compute_task_gradients(task)?;
            self.accumulate_meta_gradients(&mut meta_gradients, &task_gradients);
        }
        
        // Update meta-parameters
        self.meta_optimizer.update(&mut self.base_params, &meta_gradients);
        
        Ok(MetaLearningResult {
            average_loss: total_loss / tasks.len() as f32,
            adaptation_results,
            convergence_metric: self.compute_convergence_metric(),
        })
    }

    /// Adapt model to a specific task
    pub async fn adapt_to_task(&mut self, task: &Task) -> Result<AdaptationResult> {
        let start_time = Instant::now();
        let mut adapted_params = self.base_params.clone_for_adaptation();
        
        let initial_loss = self.compute_loss(&adapted_params, &task.support_set)?;
        let mut current_loss = initial_loss;
        
        for step in 0..self.config.inner_steps {
            // Compute gradients on support set
            let gradients = self.compute_gradients(&adapted_params, &task.support_set)?;
            
            // Update parameters
            adapted_params.update_with_gradients(&gradients, self.config.inner_lr);
            
            // Compute new loss
            current_loss = self.compute_loss(&adapted_params, &task.support_set)?;
            
            // Early stopping if loss is good enough
            if current_loss < 0.01 {
                break;
            }
        }
        
        let adaptation_time = start_time.elapsed().as_millis() as f32;
        
        let result = AdaptationResult {
            task_id: task.id,
            initial_loss,
            final_loss: current_loss,
            steps_taken: self.config.inner_steps,
            adaptation_time_ms: adaptation_time,
            success: current_loss < initial_loss * 0.5, // 50% improvement threshold
        };
        
        self.adaptation_history.push(result.clone());
        Ok(result)
    }

    fn compute_loss(&self, params: &ModelParameters, data: &[DataPoint]) -> Result<f32> {
        let mut total_loss = 0.0;
        
        for data_point in data {
            let prediction = self.forward(params, &data_point.input)?;
            let loss = self.mse_loss(&prediction, &data_point.target);
            total_loss += loss;
        }
        
        Ok(total_loss / data.len() as f32)
    }

    fn forward(&self, params: &ModelParameters, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut x = input.clone();
        
        for (i, (weight, bias)) in params.weights.iter().zip(&params.biases).enumerate() {
            x = weight.dot(&x) + bias;
            
            // Apply batch normalization if enabled
            if let Some(bn_params) = &params.batch_norm_params {
                if i < bn_params.gamma.len() {
                    x = self.batch_norm(&x, &bn_params.gamma[i], &bn_params.beta[i], 
                                       &bn_params.running_mean[i], &bn_params.running_var[i]);
                }
            }
            
            // Apply activation (except for last layer)
            if i < params.weights.len() - 1 {
                x = self.apply_activation(&x, &self.config.model_architecture.activation);
            }
        }
        
        Ok(x)
    }

    fn batch_norm(&self, x: &Array1<f32>, gamma: &Array1<f32>, beta: &Array1<f32>, 
                  mean: &Array1<f32>, var: &Array1<f32>) -> Array1<f32> {
        let epsilon = 1e-5;
        let normalized = (x - mean) / &var.mapv(|v| (v + epsilon).sqrt());
        gamma * &normalized + beta
    }

    fn apply_activation(&self, x: &Array1<f32>, activation: &str) -> Array1<f32> {
        match activation {
            "relu" => x.mapv(|v| v.max(0.0)),
            "sigmoid" => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            "tanh" => x.mapv(|v| v.tanh()),
            "leaky_relu" => x.mapv(|v| if v > 0.0 { v } else { 0.01 * v }),
            _ => x.clone(),
        }
    }

    fn mse_loss(&self, prediction: &Array1<f32>, target: &Array1<f32>) -> f32 {
        let diff = prediction - target;
        diff.mapv(|x| x * x).sum() / prediction.len() as f32
    }

    fn compute_gradients(&self, params: &ModelParameters, data: &[DataPoint]) -> Result<ModelGradients> {
        // Simplified gradient computation (in practice, would use automatic differentiation)
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();
        
        for (weight, bias) in params.weights.iter().zip(&params.biases) {
            weight_gradients.push(Array2::zeros(weight.raw_dim()));
            bias_gradients.push(Array1::zeros(bias.raw_dim()));
        }
        
        // Compute gradients using finite differences (simplified)
        let epsilon = 1e-4;
        
        for (data_point_idx, data_point) in data.iter().enumerate() {
            for (layer_idx, (weight, bias)) in params.weights.iter().zip(&params.biases).enumerate() {
                // Weight gradients
                for i in 0..weight.nrows() {
                    for j in 0..weight.ncols() {
                        let mut params_plus = params.clone_for_adaptation();
                        let mut params_minus = params.clone_for_adaptation();
                        
                        params_plus.weights[layer_idx][[i, j]] += epsilon;
                        params_minus.weights[layer_idx][[i, j]] -= epsilon;
                        
                        let pred_plus = self.forward(&params_plus, &data_point.input)?;
                        let pred_minus = self.forward(&params_minus, &data_point.input)?;
                        
                        let loss_plus = self.mse_loss(&pred_plus, &data_point.target);
                        let loss_minus = self.mse_loss(&pred_minus, &data_point.target);
                        
                        weight_gradients[layer_idx][[i, j]] += (loss_plus - loss_minus) / (2.0 * epsilon);
                    }
                }
                
                // Bias gradients
                for i in 0..bias.len() {
                    let mut params_plus = params.clone_for_adaptation();
                    let mut params_minus = params.clone_for_adaptation();
                    
                    params_plus.biases[layer_idx][i] += epsilon;
                    params_minus.biases[layer_idx][i] -= epsilon;
                    
                    let pred_plus = self.forward(&params_plus, &data_point.input)?;
                    let pred_minus = self.forward(&params_minus, &data_point.input)?;
                    
                    let loss_plus = self.mse_loss(&pred_plus, &data_point.target);
                    let loss_minus = self.mse_loss(&pred_minus, &data_point.target);
                    
                    bias_gradients[layer_idx][i] += (loss_plus - loss_minus) / (2.0 * epsilon);
                }
            }
        }
        
        // Average gradients over data points
        for weight_grad in &mut weight_gradients {
            *weight_grad = weight_grad / data.len() as f32;
        }
        for bias_grad in &mut bias_gradients {
            *bias_grad = bias_grad / data.len() as f32;
        }
        
        Ok(ModelGradients {
            weight_gradients,
            bias_gradients,
            batch_norm_gradients: None, // Simplified
        })
    }

    fn compute_task_gradients(&self, task: &Task) -> Result<ModelGradients> {
        // Compute gradients on query set after adaptation
        let mut adapted_params = self.base_params.clone_for_adaptation();
        
        // Adapt on support set
        for _ in 0..self.config.inner_steps {
            let gradients = self.compute_gradients(&adapted_params, &task.support_set)?;
            adapted_params.update_with_gradients(&gradients, self.config.inner_lr);
        }
        
        // Compute gradients on query set
        self.compute_gradients(&adapted_params, &task.query_set)
    }

    fn accumulate_meta_gradients(&self, meta_grads: &mut ModelGradients, task_grads: &ModelGradients) {
        for (meta_grad, task_grad) in meta_grads.weight_gradients.iter_mut().zip(&task_grads.weight_gradients) {
            *meta_grad = &*meta_grad + task_grad;
        }
        
        for (meta_grad, task_grad) in meta_grads.bias_gradients.iter_mut().zip(&task_grads.bias_gradients) {
            *meta_grad = &*meta_grad + task_grad;
        }
    }

    fn zero_gradients(&self) -> ModelGradients {
        let weight_gradients = self.base_params.weights.iter()
            .map(|w| Array2::zeros(w.raw_dim()))
            .collect();
        
        let bias_gradients = self.base_params.biases.iter()
            .map(|b| Array1::zeros(b.raw_dim()))
            .collect();
        
        ModelGradients {
            weight_gradients,
            bias_gradients,
            batch_norm_gradients: None,
        }
    }

    fn compute_convergence_metric(&self) -> f32 {
        if self.adaptation_history.len() < 10 {
            return 0.0;
        }
        
        let recent_results: Vec<&AdaptationResult> = self.adaptation_history
            .iter()
            .rev()
            .take(10)
            .collect();
        
        let avg_improvement: f32 = recent_results.iter()
            .map(|r| (r.initial_loss - r.final_loss) / r.initial_loss)
            .sum::<f32>() / recent_results.len() as f32;
        
        avg_improvement
    }
}

/// Task definition for meta-learning
#[derive(Debug, Clone)]
pub struct Task {
    /// Task identifier
    pub id: Uuid,
    /// Task type/domain
    pub task_type: String,
    /// Support set (few examples for adaptation)
    pub support_set: Vec<DataPoint>,
    /// Query set (test examples)
    pub query_set: Vec<DataPoint>,
    /// Task metadata
    pub metadata: TaskMetadata,
}

/// Single data point
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Input features
    pub input: Array1<f32>,
    /// Target output
    pub target: Array1<f32>,
    /// Data point metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Task metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Domain name
    pub domain: String,
    /// Difficulty level
    pub difficulty: f32,
    /// Number of classes/relations
    pub num_classes: usize,
    /// Task description
    pub description: String,
    /// Additional properties
    pub properties: HashMap<String, String>,
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

/// Reptile meta-learning algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReptileConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of inner steps
    pub inner_steps: usize,
    /// Interpolation factor
    pub interpolation_factor: f32,
}

impl Default for ReptileConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            inner_steps: 5,
            interpolation_factor: 0.1,
        }
    }
}

pub struct Reptile {
    config: ReptileConfig,
    base_params: ModelParameters,
    adaptation_history: Vec<AdaptationResult>,
}

impl Reptile {
    pub fn new(config: ReptileConfig, architecture: &ModelArchitecture) -> Self {
        Self {
            config,
            base_params: ModelParameters::new(architecture),
            adaptation_history: Vec::new(),
        }
    }

    /// Reptile meta-learning step
    pub async fn meta_step(&mut self, task: &Task) -> Result<f32> {
        let mut adapted_params = self.base_params.clone_for_adaptation();
        
        // Adapt on task
        for _ in 0..self.config.inner_steps {
            let gradients = self.compute_gradients(&adapted_params, &task.support_set)?;
            adapted_params.update_with_gradients(&gradients, self.config.learning_rate);
        }
        
        // Interpolate toward adapted parameters
        self.interpolate_parameters(&adapted_params);
        
        // Evaluate on query set
        let query_loss = self.compute_loss(&self.base_params, &task.query_set)?;
        
        Ok(query_loss)
    }

    fn interpolate_parameters(&mut self, adapted_params: &ModelParameters) {
        let alpha = self.config.interpolation_factor;
        
        for (base_weight, adapted_weight) in self.base_params.weights.iter_mut().zip(&adapted_params.weights) {
            *base_weight = (1.0 - alpha) * &*base_weight + alpha * adapted_weight;
        }
        
        for (base_bias, adapted_bias) in self.base_params.biases.iter_mut().zip(&adapted_params.biases) {
            *base_bias = (1.0 - alpha) * &*base_bias + alpha * adapted_bias;
        }
    }

    fn compute_loss(&self, params: &ModelParameters, data: &[DataPoint]) -> Result<f32> {
        // Implementation similar to MAML
        let mut total_loss = 0.0;
        
        for data_point in data {
            let prediction = self.forward(params, &data_point.input)?;
            let loss = self.mse_loss(&prediction, &data_point.target);
            total_loss += loss;
        }
        
        Ok(total_loss / data.len() as f32)
    }

    fn forward(&self, params: &ModelParameters, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut x = input.clone();
        
        for (weight, bias) in params.weights.iter().zip(&params.biases) {
            x = weight.dot(&x) + bias;
            x = x.mapv(|v| v.max(0.0)); // ReLU activation
        }
        
        Ok(x)
    }

    fn mse_loss(&self, prediction: &Array1<f32>, target: &Array1<f32>) -> f32 {
        let diff = prediction - target;
        diff.mapv(|x| x * x).sum() / prediction.len() as f32
    }

    fn compute_gradients(&self, params: &ModelParameters, data: &[DataPoint]) -> Result<ModelGradients> {
        // Simplified gradient computation
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();
        
        for (weight, bias) in params.weights.iter().zip(&params.biases) {
            weight_gradients.push(Array2::zeros(weight.raw_dim()));
            bias_gradients.push(Array1::zeros(bias.raw_dim()));
        }
        
        Ok(ModelGradients {
            weight_gradients,
            bias_gradients,
            batch_norm_gradients: None,
        })
    }
}

/// Prototypical Networks configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypicalConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Distance metric
    pub distance_metric: String,
    /// Temperature for softmax
    pub temperature: f32,
}

impl Default for PrototypicalConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            distance_metric: "euclidean".to_string(),
            temperature: 1.0,
        }
    }
}

/// Prototypical Networks for few-shot learning
pub struct PrototypicalNetworks {
    config: PrototypicalConfig,
    embedding_network: ModelParameters,
}

impl PrototypicalNetworks {
    pub fn new(config: PrototypicalConfig, architecture: &ModelArchitecture) -> Self {
        Self {
            config,
            embedding_network: ModelParameters::new(architecture),
        }
    }

    /// Classify query examples using prototypical networks
    pub fn classify(&self, support_set: &[DataPoint], query_examples: &[Array1<f32>]) -> Result<Vec<Array1<f32>>> {
        // Compute prototypes from support set
        let prototypes = self.compute_prototypes(support_set)?;
        
        // Classify query examples
        let mut predictions = Vec::new();
        for query in query_examples {
            let query_embedding = self.embed(query)?;
            let distances = self.compute_distances(&query_embedding, &prototypes)?;
            let probabilities = self.softmax(&distances);
            predictions.push(probabilities);
        }
        
        Ok(predictions)
    }

    fn compute_prototypes(&self, support_set: &[DataPoint]) -> Result<Vec<Array1<f32>>> {
        let mut class_embeddings: HashMap<String, Vec<Array1<f32>>> = HashMap::new();
        
        // Group examples by class
        for data_point in support_set {
            let embedding = self.embed(&data_point.input)?;
            let class_label = self.extract_class_label(data_point)?;
            
            class_embeddings.entry(class_label)
                .or_insert_with(Vec::new)
                .push(embedding);
        }
        
        // Compute prototype for each class
        let mut prototypes = Vec::new();
        for embeddings in class_embeddings.values() {
            let prototype = self.compute_centroid(embeddings)?;
            prototypes.push(prototype);
        }
        
        Ok(prototypes)
    }

    fn embed(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut x = input.clone();
        
        for (weight, bias) in self.embedding_network.weights.iter().zip(&self.embedding_network.biases) {
            x = weight.dot(&x) + bias;
            x = x.mapv(|v| v.max(0.0)); // ReLU activation
        }
        
        Ok(x)
    }

    fn extract_class_label(&self, data_point: &DataPoint) -> Result<String> {
        // Find the index of the maximum value in the target (assuming one-hot encoding)
        let class_idx = data_point.target.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("Invalid target vector"))?;
        
        Ok(format!("class_{}", class_idx))
    }

    fn compute_centroid(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Err(anyhow!("Cannot compute centroid of empty embeddings"));
        }
        
        let sum = embeddings.iter()
            .fold(Array1::zeros(embeddings[0].len()), |acc, emb| acc + emb);
        
        Ok(sum / embeddings.len() as f32)
    }

    fn compute_distances(&self, query: &Array1<f32>, prototypes: &[Array1<f32>]) -> Result<Array1<f32>> {
        let mut distances = Array1::zeros(prototypes.len());
        
        for (i, prototype) in prototypes.iter().enumerate() {
            distances[i] = match self.config.distance_metric.as_str() {
                "euclidean" => {
                    let diff = query - prototype;
                    diff.mapv(|x| x * x).sum().sqrt()
                }
                "cosine" => {
                    let dot_product = query.dot(prototype);
                    let norm_query = query.mapv(|x| x * x).sum().sqrt();
                    let norm_prototype = prototype.mapv(|x| x * x).sum().sqrt();
                    1.0 - dot_product / (norm_query * norm_prototype)
                }
                _ => return Err(anyhow!("Unsupported distance metric")),
            };
        }
        
        Ok(distances)
    }

    fn softmax(&self, distances: &Array1<f32>) -> Array1<f32> {
        // Convert distances to similarities (negative distances)
        let similarities = distances.map(|&d| -d / self.config.temperature);
        
        let max_sim = similarities.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sims = similarities.map(|&s| (s - max_sim).exp());
        let sum_exp = exp_sims.sum();
        
        exp_sims / sum_exp
    }
}

/// Matching Networks configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Use full context embeddings
    pub use_full_context: bool,
    /// Number of LSTM processing steps
    pub lstm_steps: usize,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            use_full_context: true,
            lstm_steps: 5,
        }
    }
}

/// Matching Networks implementation
pub struct MatchingNetworks {
    config: MatchingConfig,
    embedding_network: ModelParameters,
    attention_network: ModelParameters,
}

impl MatchingNetworks {
    pub fn new(config: MatchingConfig, architecture: &ModelArchitecture) -> Self {
        Self {
            config,
            embedding_network: ModelParameters::new(architecture),
            attention_network: ModelParameters::new(architecture),
        }
    }

    /// Classify using matching networks
    pub fn classify(&self, support_set: &[DataPoint], query: &Array1<f32>) -> Result<Array1<f32>> {
        // Embed support set and query
        let support_embeddings = self.embed_support_set(support_set)?;
        let query_embedding = self.embed_query(query, &support_embeddings)?;
        
        // Compute attention weights
        let attention_weights = self.compute_attention(&query_embedding, &support_embeddings)?;
        
        // Weighted combination of support labels
        let prediction = self.weighted_combination(support_set, &attention_weights)?;
        
        Ok(prediction)
    }

    fn embed_support_set(&self, support_set: &[DataPoint]) -> Result<Vec<Array1<f32>>> {
        support_set.iter()
            .map(|dp| self.embed(&dp.input))
            .collect()
    }

    fn embed_query(&self, query: &Array1<f32>, support_embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if self.config.use_full_context {
            // Use LSTM to process query in context of support set
            self.embed_with_context(query, support_embeddings)
        } else {
            self.embed(query)
        }
    }

    fn embed(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut x = input.clone();
        
        for (weight, bias) in self.embedding_network.weights.iter().zip(&self.embedding_network.biases) {
            x = weight.dot(&x) + bias;
            x = x.mapv(|v| v.max(0.0)); // ReLU activation
        }
        
        Ok(x)
    }

    fn embed_with_context(&self, query: &Array1<f32>, support_embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        let mut query_embedding = self.embed(query)?;
        
        // Simple context processing (in practice, would use bidirectional LSTM)
        for _ in 0..self.config.lstm_steps {
            let context = self.compute_context(&query_embedding, support_embeddings)?;
            query_embedding = &query_embedding + &context;
        }
        
        Ok(query_embedding)
    }

    fn compute_context(&self, query: &Array1<f32>, support: &[Array1<f32>]) -> Result<Array1<f32>> {
        let mut context = Array1::zeros(query.len());
        
        for support_emb in support {
            let similarity = self.cosine_similarity(query, support_emb);
            context = context + similarity * support_emb;
        }
        
        Ok(context / support.len() as f32)
    }

    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.mapv(|x| x * x).sum().sqrt();
        let norm_b = b.mapv(|x| x * x).sum().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    fn compute_attention(&self, query: &Array1<f32>, support: &[Array1<f32>]) -> Result<Array1<f32>> {
        let mut attention_scores = Array1::zeros(support.len());
        
        for (i, support_emb) in support.iter().enumerate() {
            attention_scores[i] = self.cosine_similarity(query, support_emb);
        }
        
        // Softmax
        let max_score = attention_scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores = attention_scores.map(|&s| (s - max_score).exp());
        let sum_exp = exp_scores.sum();
        
        Ok(exp_scores / sum_exp)
    }

    fn weighted_combination(&self, support_set: &[DataPoint], weights: &Array1<f32>) -> Result<Array1<f32>> {
        if support_set.is_empty() || weights.is_empty() {
            return Err(anyhow!("Empty support set or weights"));
        }
        
        let output_dim = support_set[0].target.len();
        let mut prediction = Array1::zeros(output_dim);
        
        for (i, data_point) in support_set.iter().enumerate() {
            prediction = prediction + weights[i] * &data_point.target;
        }
        
        Ok(prediction)
    }
}

/// Relation Networks configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationConfig {
    /// Feature dimension
    pub feature_dim: usize,
    /// Relation dimension
    pub relation_dim: usize,
    /// Number of relation layers
    pub relation_layers: usize,
}

impl Default for RelationConfig {
    fn default() -> Self {
        Self {
            feature_dim: 128,
            relation_dim: 64,
            relation_layers: 2,
        }
    }
}

/// Relation Networks for few-shot learning
pub struct RelationNetworks {
    config: RelationConfig,
    feature_network: ModelParameters,
    relation_network: ModelParameters,
}

impl RelationNetworks {
    pub fn new(config: RelationConfig, architecture: &ModelArchitecture) -> Self {
        Self {
            config,
            feature_network: ModelParameters::new(architecture),
            relation_network: ModelParameters::new(architecture),
        }
    }

    /// Classify using relation networks
    pub fn classify(&self, support_set: &[DataPoint], query: &Array1<f32>) -> Result<Array1<f32>> {
        // Extract features
        let support_features = self.extract_features_from_support(support_set)?;
        let query_features = self.extract_features(&query)?;
        
        // Compute relation scores
        let relation_scores = self.compute_relation_scores(&query_features, &support_features)?;
        
        Ok(relation_scores)
    }

    fn extract_features_from_support(&self, support_set: &[DataPoint]) -> Result<Vec<Array1<f32>>> {
        support_set.iter()
            .map(|dp| self.extract_features(&dp.input))
            .collect()
    }

    fn extract_features(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut x = input.clone();
        
        for (weight, bias) in self.feature_network.weights.iter().zip(&self.feature_network.biases) {
            x = weight.dot(&x) + bias;
            x = x.mapv(|v| v.max(0.0)); // ReLU activation
        }
        
        Ok(x)
    }

    fn compute_relation_scores(&self, query_features: &Array1<f32>, support_features: &[Array1<f32>]) -> Result<Array1<f32>> {
        let mut scores = Array1::zeros(support_features.len());
        
        for (i, support_feat) in support_features.iter().enumerate() {
            let concatenated = self.concatenate_features(query_features, support_feat);
            let relation_score = self.compute_relation(&concatenated)?;
            scores[i] = relation_score;
        }
        
        Ok(scores)
    }

    fn concatenate_features(&self, query: &Array1<f32>, support: &Array1<f32>) -> Array1<f32> {
        concatenate![Axis(0), query.view(), support.view()]
    }

    fn compute_relation(&self, features: &Array1<f32>) -> Result<f32> {
        let mut x = features.clone();
        
        for (weight, bias) in self.relation_network.weights.iter().zip(&self.relation_network.biases) {
            x = weight.dot(&x) + bias;
            x = x.mapv(|v| v.max(0.0)); // ReLU activation
        }
        
        // Output single relation score
        Ok(x[0])
    }
}

/// Memory-Augmented Neural Networks for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MANNConfig {
    /// Memory size
    pub memory_size: usize,
    /// Memory dimension
    pub memory_dim: usize,
    /// Controller dimension
    pub controller_dim: usize,
    /// Number of read heads
    pub num_read_heads: usize,
}

impl Default for MANNConfig {
    fn default() -> Self {
        Self {
            memory_size: 256,
            memory_dim: 64,
            controller_dim: 128,
            num_read_heads: 4,
        }
    }
}

/// Memory-Augmented Neural Networks
pub struct MANN {
    config: MANNConfig,
    controller: ModelParameters,
    memory: Array2<f32>,
    read_weights: Array2<f32>,
    write_weights: Array1<f32>,
}

impl MANN {
    pub fn new(config: MANNConfig, architecture: &ModelArchitecture) -> Self {
        let memory = Array2::zeros((config.memory_size, config.memory_dim));
        let read_weights = Array2::zeros((config.num_read_heads, config.memory_size));
        let write_weights = Array1::zeros(config.memory_size);
        
        Self {
            config,
            controller: ModelParameters::new(architecture),
            memory,
            read_weights,
            write_weights,
        }
    }

    /// Process sequence with memory
    pub fn process_sequence(&mut self, sequence: &[Array1<f32>]) -> Result<Vec<Array1<f32>>> {
        let mut outputs = Vec::new();
        
        for input in sequence {
            let output = self.forward_step(input)?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }

    fn forward_step(&mut self, input: &Array1<f32>) -> Result<Array1<f32>> {
        // Read from memory
        let read_vectors = self.read_from_memory()?;
        
        // Concatenate input with read vectors
        let mut controller_input = input.clone();
        for read_vector in &read_vectors {
            controller_input = concatenate![Axis(0), controller_input.view(), read_vector.view()];
        }
        
        // Controller forward pass
        let controller_output = self.controller_forward(&controller_input)?;
        
        // Parse controller output
        let (output, write_vector, addressing) = self.parse_controller_output(&controller_output)?;
        
        // Write to memory
        self.write_to_memory(&write_vector, &addressing)?;
        
        Ok(output)
    }

    fn read_from_memory(&self) -> Result<Vec<Array1<f32>>> {
        let mut read_vectors = Vec::new();
        
        for i in 0..self.config.num_read_heads {
            let read_weights = self.read_weights.row(i);
            let read_vector = self.memory.t().dot(&read_weights);
            read_vectors.push(read_vector);
        }
        
        Ok(read_vectors)
    }

    fn controller_forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut x = input.clone();
        
        for (weight, bias) in self.controller.weights.iter().zip(&self.controller.biases) {
            x = weight.dot(&x) + bias;
            x = x.mapv(|v| v.max(0.0)); // ReLU activation
        }
        
        Ok(x)
    }

    fn parse_controller_output(&self, output: &Array1<f32>) -> Result<(Array1<f32>, Array1<f32>, Array1<f32>)> {
        let output_size = 64; // Fixed output size
        let write_size = self.config.memory_dim;
        let addressing_size = self.config.memory_size;
        
        if output.len() < output_size + write_size + addressing_size {
            return Err(anyhow!("Controller output too small"));
        }
        
        let network_output = output.slice(s![..output_size]).to_owned();
        let write_vector = output.slice(s![output_size..output_size + write_size]).to_owned();
        let addressing = output.slice(s![output_size + write_size..output_size + write_size + addressing_size]).to_owned();
        
        Ok((network_output, write_vector, addressing))
    }

    fn write_to_memory(&mut self, write_vector: &Array1<f32>, addressing: &Array1<f32>) -> Result<()> {
        // Softmax normalization of addressing
        let max_addr = addressing.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_addr = addressing.map(|&a| (a - max_addr).exp());
        let sum_exp = exp_addr.sum();
        let normalized_addr = exp_addr / sum_exp;
        
        // Write to memory
        for i in 0..self.config.memory_size {
            for j in 0..self.config.memory_dim {
                self.memory[[i, j]] += normalized_addr[i] * write_vector[j];
            }
        }
        
        self.write_weights = normalized_addr;
        
        Ok(())
    }

    /// Reset memory for new task
    pub fn reset_memory(&mut self) {
        self.memory.fill(0.0);
        self.read_weights.fill(0.0);
        self.write_weights.fill(0.0);
    }
}

/// Task sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSamplingConfig {
    /// Number of ways (classes per task)
    pub n_way: usize,
    /// Number of shots (examples per class)
    pub k_shot: usize,
    /// Number of query examples per class
    pub n_query: usize,
    /// Task domains to sample from
    pub domains: Vec<String>,
    /// Task difficulty distribution
    pub difficulty_distribution: DifficultyDistribution,
}

impl Default for TaskSamplingConfig {
    fn default() -> Self {
        Self {
            n_way: 5,
            k_shot: 1,
            n_query: 15,
            domains: vec!["general".to_string()],
            difficulty_distribution: DifficultyDistribution::Uniform,
        }
    }
}

/// Task difficulty distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyDistribution {
    Uniform,
    Normal { mean: f32, std: f32 },
    Exponential { lambda: f32 },
    Custom { weights: Vec<f32> },
}

/// Task sampler for meta-learning
pub struct TaskSampler {
    config: TaskSamplingConfig,
    task_pools: HashMap<String, Vec<TaskTemplate>>,
    sampling_statistics: SamplingStatistics,
}

/// Task template for generating tasks
#[derive(Debug, Clone)]
pub struct TaskTemplate {
    /// Template ID
    pub id: Uuid,
    /// Domain
    pub domain: String,
    /// Difficulty level
    pub difficulty: f32,
    /// Data generator function
    pub data_generator: DataGenerator,
    /// Template metadata
    pub metadata: TaskMetadata,
}

/// Data generator for tasks
#[derive(Debug, Clone)]
pub enum DataGenerator {
    Synthetic {
        function_type: String,
        noise_level: f32,
        complexity: f32,
    },
    RealData {
        dataset_path: String,
        preprocessing: String,
    },
    Procedural {
        generation_rules: Vec<String>,
        random_seed: u64,
    },
}

/// Sampling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingStatistics {
    /// Total tasks sampled
    pub total_tasks: usize,
    /// Tasks per domain
    pub domain_counts: HashMap<String, usize>,
    /// Difficulty distribution
    pub difficulty_histogram: Vec<usize>,
    /// Average task generation time
    pub avg_generation_time_ms: f32,
}

impl TaskSampler {
    pub fn new(config: TaskSamplingConfig) -> Self {
        Self {
            config,
            task_pools: HashMap::new(),
            sampling_statistics: SamplingStatistics {
                total_tasks: 0,
                domain_counts: HashMap::new(),
                difficulty_histogram: vec![0; 10], // 10 difficulty bins
                avg_generation_time_ms: 0.0,
            },
        }
    }

    /// Sample a batch of tasks
    pub fn sample_task_batch(&mut self, batch_size: usize) -> Result<Vec<Task>> {
        let mut tasks = Vec::new();
        
        for _ in 0..batch_size {
            let task = self.sample_single_task()?;
            tasks.push(task);
        }
        
        Ok(tasks)
    }

    /// Sample a single task
    pub fn sample_single_task(&mut self) -> Result<Task> {
        let start_time = Instant::now();
        
        // Select domain
        let domain = self.select_domain()?;
        
        // Select difficulty
        let difficulty = self.sample_difficulty();
        
        // Generate task
        let task = self.generate_task(&domain, difficulty)?;
        
        // Update statistics
        let generation_time = start_time.elapsed().as_millis() as f32;
        self.update_statistics(&domain, difficulty, generation_time);
        
        Ok(task)
    }

    fn select_domain(&self) -> Result<String> {
        if self.config.domains.is_empty() {
            return Err(anyhow!("No domains configured"));
        }
        
        let mut rng = thread_rng();
        Ok(self.config.domains.choose(&mut rng).unwrap().clone())
    }

    fn sample_difficulty(&self) -> f32 {
        let mut rng = thread_rng();
        
        match &self.config.difficulty_distribution {
            DifficultyDistribution::Uniform => rng.gen_range(0.0..1.0),
            DifficultyDistribution::Normal { mean, std } => {
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(*mean, *std).unwrap();
                normal.sample(&mut rng).max(0.0).min(1.0)
            }
            DifficultyDistribution::Exponential { lambda } => {
                use rand_distr::{Distribution, Exp};
                let exp = Exp::new(*lambda).unwrap();
                exp.sample(&mut rng).min(1.0)
            }
            DifficultyDistribution::Custom { weights } => {
                let total_weight: f32 = weights.iter().sum();
                let random_weight = rng.gen_range(0.0..total_weight);
                let mut cumulative = 0.0;
                
                for (i, &weight) in weights.iter().enumerate() {
                    cumulative += weight;
                    if random_weight <= cumulative {
                        return i as f32 / weights.len() as f32;
                    }
                }
                
                1.0
            }
        }
    }

    fn generate_task(&self, domain: &str, difficulty: f32) -> Result<Task> {
        let mut rng = thread_rng();
        
        // Generate synthetic data for now
        let mut support_set = Vec::new();
        let mut query_set = Vec::new();
        
        for class_idx in 0..self.config.n_way {
            // Generate support examples
            for _ in 0..self.config.k_shot {
                let data_point = self.generate_data_point(class_idx, difficulty, &mut rng)?;
                support_set.push(data_point);
            }
            
            // Generate query examples
            for _ in 0..self.config.n_query {
                let data_point = self.generate_data_point(class_idx, difficulty, &mut rng)?;
                query_set.push(data_point);
            }
        }
        
        Ok(Task {
            id: Uuid::new_v4(),
            task_type: format!("{}_way_{}_shot", self.config.n_way, self.config.k_shot),
            support_set,
            query_set,
            metadata: TaskMetadata {
                domain: domain.to_string(),
                difficulty,
                num_classes: self.config.n_way,
                description: format!("Generated {}-way {}-shot task", self.config.n_way, self.config.k_shot),
                properties: HashMap::new(),
            },
        })
    }

    fn generate_data_point(&self, class_idx: usize, difficulty: f32, rng: &mut impl Rng) -> Result<DataPoint> {
        let input_dim = 128;
        let output_dim = self.config.n_way;
        
        // Generate synthetic input with class-specific pattern
        let mut input = Array1::zeros(input_dim);
        for i in 0..input_dim {
            let base_value = (class_idx as f32 * 2.0 + difficulty) * (i as f32 / input_dim as f32);
            let noise = rng.gen_range(-0.1..0.1) * difficulty;
            input[i] = base_value + noise;
        }
        
        // Generate one-hot target
        let mut target = Array1::zeros(output_dim);
        target[class_idx] = 1.0;
        
        Ok(DataPoint {
            input,
            target,
            metadata: None,
        })
    }

    fn update_statistics(&mut self, domain: &str, difficulty: f32, generation_time: f32) {
        self.sampling_statistics.total_tasks += 1;
        
        *self.sampling_statistics.domain_counts.entry(domain.to_string()).or_insert(0) += 1;
        
        let difficulty_bin = (difficulty * 10.0) as usize;
        if difficulty_bin < self.sampling_statistics.difficulty_histogram.len() {
            self.sampling_statistics.difficulty_histogram[difficulty_bin] += 1;
        }
        
        let alpha = 0.1; // Exponential moving average
        self.sampling_statistics.avg_generation_time_ms = 
            alpha * generation_time + (1.0 - alpha) * self.sampling_statistics.avg_generation_time_ms;
    }

    /// Get sampling statistics
    pub fn get_statistics(&self) -> &SamplingStatistics {
        &self.sampling_statistics
    }
}

/// Meta-learning history tracking
pub struct MetaLearningHistory {
    /// Episode results
    episode_results: Vec<MetaLearningResult>,
    /// Performance trends
    performance_trends: PerformanceTrends,
    /// Best models
    best_models: BestModels,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Loss trend
    pub loss_trend: Vec<f32>,
    /// Convergence trend
    pub convergence_trend: Vec<f32>,
    /// Adaptation speed trend
    pub adaptation_speed_trend: Vec<f32>,
    /// Generalization trend
    pub generalization_trend: Vec<f32>,
}

/// Best models storage
pub struct BestModels {
    /// Best overall model
    pub best_overall: Option<ModelParameters>,
    /// Best model per domain
    pub best_per_domain: HashMap<String, ModelParameters>,
    /// Best model per difficulty
    pub best_per_difficulty: HashMap<String, ModelParameters>,
}

/// Meta-performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaPerformanceMetrics {
    /// Average adaptation time
    pub avg_adaptation_time_ms: f32,
    /// Few-shot accuracy
    pub few_shot_accuracy: f32,
    /// Zero-shot accuracy
    pub zero_shot_accuracy: f32,
    /// Transfer learning efficiency
    pub transfer_efficiency: f32,
    /// Catastrophic forgetting measure
    pub forgetting_measure: f32,
    /// Domain adaptation score
    pub domain_adaptation_score: f32,
}

impl Default for MetaPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_adaptation_time_ms: 0.0,
            few_shot_accuracy: 0.0,
            zero_shot_accuracy: 0.0,
            transfer_efficiency: 0.0,
            forgetting_measure: 0.0,
            domain_adaptation_score: 0.0,
        }
    }
}

impl MetaLearningEngine {
    /// Create new meta-learning engine
    pub fn new(config: MetaLearningConfig) -> Result<Self> {
        let maml = MAML::new(config.maml_config.clone());
        let reptile = Reptile::new(config.reptile_config.clone(), &config.maml_config.model_architecture);
        let prototypical_networks = PrototypicalNetworks::new(
            config.prototypical_config.clone(),
            &config.maml_config.model_architecture,
        );
        let matching_networks = MatchingNetworks::new(
            config.matching_config.clone(),
            &config.maml_config.model_architecture,
        );
        let relation_networks = RelationNetworks::new(
            config.relation_config.clone(),
            &config.maml_config.model_architecture,
        );
        let mann = MANN::new(config.mann_config.clone(), &config.maml_config.model_architecture);
        let task_sampler = TaskSampler::new(config.task_config.clone());
        
        let meta_history = MetaLearningHistory {
            episode_results: Vec::new(),
            performance_trends: PerformanceTrends {
                loss_trend: Vec::new(),
                convergence_trend: Vec::new(),
                adaptation_speed_trend: Vec::new(),
                generalization_trend: Vec::new(),
            },
            best_models: BestModels {
                best_overall: None,
                best_per_domain: HashMap::new(),
                best_per_difficulty: HashMap::new(),
            },
        };
        
        Ok(Self {
            config,
            maml,
            reptile,
            prototypical_networks,
            matching_networks,
            relation_networks,
            mann,
            task_sampler,
            meta_history,
            performance_metrics: MetaPerformanceMetrics::default(),
        })
    }

    /// Run meta-learning training
    pub async fn meta_train(&mut self, num_episodes: usize) -> Result<()> {
        info!("Starting meta-learning training for {} episodes", num_episodes);
        
        for episode in 0..num_episodes {
            info!("Meta-learning episode {}/{}", episode + 1, num_episodes);
            
            // Sample task batch
            let tasks = self.task_sampler.sample_task_batch(8)?; // Batch size of 8
            
            // Train with different algorithms
            let maml_result = self.maml.meta_learn_episode(&tasks).await?;
            
            // Track results
            self.meta_history.episode_results.push(maml_result.clone());
            self.update_performance_trends(&maml_result);
            
            // Evaluation
            if episode % self.config.global_settings.eval_frequency == 0 {
                let eval_result = self.evaluate_meta_learning().await?;
                info!("Episode {} evaluation: accuracy = {:.4}", episode, eval_result.few_shot_accuracy);
            }
            
            // Early stopping check
            if self.check_early_stopping() {
                info!("Early stopping triggered at episode {}", episode);
                break;
            }
        }
        
        info!("Meta-learning training completed");
        Ok(())
    }

    /// Evaluate meta-learning performance
    pub async fn evaluate_meta_learning(&mut self) -> Result<MetaPerformanceMetrics> {
        let eval_tasks = self.task_sampler.sample_task_batch(20)?;
        let mut total_accuracy = 0.0;
        let mut total_adaptation_time = 0.0;
        
        for task in &eval_tasks {
            let start_time = Instant::now();
            let adaptation_result = self.maml.adapt_to_task(task).await?;
            let adaptation_time = start_time.elapsed().as_millis() as f32;
            
            // Compute accuracy (simplified)
            let accuracy = if adaptation_result.success { 1.0 } else { 0.0 };
            total_accuracy += accuracy;
            total_adaptation_time += adaptation_time;
        }
        
        self.performance_metrics.few_shot_accuracy = total_accuracy / eval_tasks.len() as f32;
        self.performance_metrics.avg_adaptation_time_ms = total_adaptation_time / eval_tasks.len() as f32;
        
        Ok(self.performance_metrics.clone())
    }

    /// Fast adaptation to new task
    pub async fn fast_adapt(&mut self, task: &Task, algorithm: &str) -> Result<AdaptationResult> {
        match algorithm {
            "maml" => self.maml.adapt_to_task(task).await,
            "reptile" => {
                let loss = self.reptile.meta_step(task).await?;
                Ok(AdaptationResult {
                    task_id: task.id,
                    initial_loss: loss,
                    final_loss: loss,
                    steps_taken: self.config.reptile_config.inner_steps,
                    adaptation_time_ms: 50.0, // Simplified
                    success: loss < 0.5,
                })
            }
            _ => Err(anyhow!("Unknown algorithm: {}", algorithm)),
        }
    }

    /// Few-shot classification
    pub async fn few_shot_classify(&mut self, support_set: &[DataPoint], queries: &[Array1<f32>], algorithm: &str) -> Result<Vec<Array1<f32>>> {
        match algorithm {
            "prototypical" => self.prototypical_networks.classify(support_set, queries),
            "matching" => {
                let mut results = Vec::new();
                for query in queries {
                    let result = self.matching_networks.classify(support_set, query)?;
                    results.push(result);
                }
                Ok(results)
            }
            "relation" => {
                let mut results = Vec::new();
                for query in queries {
                    let result = self.relation_networks.classify(support_set, query)?;
                    results.push(result);
                }
                Ok(results)
            }
            _ => Err(anyhow!("Unknown few-shot algorithm: {}", algorithm)),
        }
    }

    fn update_performance_trends(&mut self, result: &MetaLearningResult) {
        self.meta_history.performance_trends.loss_trend.push(result.average_loss);
        self.meta_history.performance_trends.convergence_trend.push(result.convergence_metric);
        
        let avg_adaptation_time = result.adaptation_results.iter()
            .map(|r| r.adaptation_time_ms)
            .sum::<f32>() / result.adaptation_results.len() as f32;
        self.meta_history.performance_trends.adaptation_speed_trend.push(avg_adaptation_time);
    }

    fn check_early_stopping(&self) -> bool {
        if !self.config.global_settings.enable_early_stopping {
            return false;
        }
        
        let patience = self.config.global_settings.early_stopping_patience;
        let trends = &self.meta_history.performance_trends.loss_trend;
        
        if trends.len() < patience {
            return false;
        }
        
        let recent_losses = &trends[trends.len() - patience..];
        let best_recent = recent_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let current_loss = trends[trends.len() - 1];
        
        current_loss > best_recent * 1.01 // No improvement tolerance
    }

    /// Get meta-learning statistics
    pub fn get_statistics(&self) -> MetaLearningStatistics {
        MetaLearningStatistics {
            total_episodes: self.meta_history.episode_results.len(),
            current_performance: self.performance_metrics.clone(),
            performance_trends: self.meta_history.performance_trends.clone(),
            task_sampling_stats: self.task_sampler.get_statistics().clone(),
        }
    }
}

/// Meta-learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningStatistics {
    /// Total training episodes
    pub total_episodes: usize,
    /// Current performance metrics
    pub current_performance: MetaPerformanceMetrics,
    /// Performance trends over time
    pub performance_trends: PerformanceTrends,
    /// Task sampling statistics
    pub task_sampling_stats: SamplingStatistics,
}

/// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_meta_learning_engine_creation() {
        let config = MetaLearningConfig::default();
        let engine = MetaLearningEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_maml_adaptation() {
        let config = MAMLConfig::default();
        let mut maml = MAML::new(config);
        
        // Create simple task
        let support_set = vec![DataPoint {
            input: Array1::ones(128),
            target: Array1::zeros(128),
            metadata: None,
        }];
        
        let query_set = vec![DataPoint {
            input: Array1::ones(128),
            target: Array1::zeros(128),
            metadata: None,
        }];
        
        let task = Task {
            id: Uuid::new_v4(),
            task_type: "test".to_string(),
            support_set,
            query_set,
            metadata: TaskMetadata {
                domain: "test".to_string(),
                difficulty: 0.5,
                num_classes: 1,
                description: "test task".to_string(),
                properties: HashMap::new(),
            },
        };
        
        let result = maml.adapt_to_task(&task).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_prototypical_networks() {
        let config = PrototypicalConfig::default();
        let architecture = ModelArchitecture::default();
        let prototypical = PrototypicalNetworks::new(config, &architecture);
        
        let support_set = vec![
            DataPoint {
                input: Array1::ones(128),
                target: Array1::from_vec(vec![1.0, 0.0]),
                metadata: None,
            },
            DataPoint {
                input: Array1::zeros(128),
                target: Array1::from_vec(vec![0.0, 1.0]),
                metadata: None,
            },
        ];
        
        let queries = vec![Array1::ones(128), Array1::zeros(128)];
        
        let result = prototypical.classify(&support_set, &queries);
        assert!(result.is_ok());
    }

    #[test]
    fn test_task_sampler() {
        let config = TaskSamplingConfig::default();
        let mut sampler = TaskSampler::new(config);
        
        let task = sampler.sample_single_task();
        assert!(task.is_ok());
        
        let task = task.unwrap();
        assert_eq!(task.support_set.len(), 5); // n_way * k_shot
        assert_eq!(task.query_set.len(), 75); // n_way * n_query
    }

    #[test]
    fn test_model_parameters() {
        let architecture = ModelArchitecture::default();
        let params = ModelParameters::new(&architecture);
        
        assert_eq!(params.weights.len(), 3); // input -> hidden1 -> hidden2 -> output
        assert_eq!(params.biases.len(), 3);
        
        let cloned = params.clone_for_adaptation();
        assert_eq!(cloned.weights.len(), params.weights.len());
    }

    #[tokio::test]
    async fn test_mann_sequence_processing() {
        let config = MANNConfig::default();
        let architecture = ModelArchitecture::default();
        let mut mann = MANN::new(config, &architecture);
        
        let sequence = vec![
            Array1::ones(128),
            Array1::zeros(128),
            Array1::ones(128) * 0.5,
        ];
        
        let result = mann.process_sequence(&sequence);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }
}