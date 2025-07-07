//! Model-Agnostic Meta-Learning (MAML) implementation

use super::types::*;
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use std::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

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
        
        let duration = start_time.elapsed();
        
        let result = AdaptationResult {
            task_id: task.id,
            initial_loss,
            final_loss: current_loss,
            adaptation_steps: self.config.inner_steps,
            duration,
            task_metadata: task.metadata.clone(),
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
                if i < bn_params.scale.len() {
                    x = self.batch_norm(&x, &bn_params.scale[i], &bn_params.shift[i], 
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

    fn batch_norm(&self, x: &Array1<f32>, scale: &Array1<f32>, shift: &Array1<f32>, 
                  mean: &Array1<f32>, var: &Array1<f32>) -> Array1<f32> {
        let epsilon = 1e-5;
        let normalized = (x - mean) / &var.mapv(|v| (v + epsilon).sqrt());
        scale * &normalized + shift
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
        
        for data_point in data {
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
            *weight_grad = &*weight_grad / data.len() as f32;
        }
        for bias_grad in &mut bias_gradients {
            *bias_grad = &*bias_grad / data.len() as f32;
        }
        
        Ok(ModelGradients {
            weight_gradients,
            bias_gradients,
            batch_norm_gradients: None, // Simplified for now
        })
    }

    fn zero_gradients(&self) -> ModelGradients {
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();
        
        for (weight, bias) in self.base_params.weights.iter().zip(&self.base_params.biases) {
            weight_gradients.push(Array2::zeros(weight.raw_dim()));
            bias_gradients.push(Array1::zeros(bias.raw_dim()));
        }
        
        ModelGradients {
            weight_gradients,
            bias_gradients,
            batch_norm_gradients: None,
        }
    }

    fn compute_task_gradients(&self, task: &Task) -> Result<ModelGradients> {
        // Simplified task gradient computation
        self.compute_gradients(&self.base_params, &task.query_set)
    }

    fn accumulate_meta_gradients(&self, meta_gradients: &mut ModelGradients, task_gradients: &ModelGradients) {
        for (meta_grad, task_grad) in meta_gradients.weight_gradients.iter_mut().zip(&task_gradients.weight_gradients) {
            *meta_grad = &*meta_grad + task_grad;
        }
        
        for (meta_grad, task_grad) in meta_gradients.bias_gradients.iter_mut().zip(&task_gradients.bias_gradients) {
            *meta_grad = &*meta_grad + task_grad;
        }
    }

    fn compute_convergence_metric(&self) -> f32 {
        if self.adaptation_history.len() < 2 {
            return 0.0;
        }
        
        let recent_losses: Vec<f32> = self.adaptation_history
            .iter()
            .rev()
            .take(10)
            .map(|result| result.final_loss)
            .collect();
        
        if recent_losses.len() < 2 {
            return 0.0;
        }
        
        let mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let variance = recent_losses.iter()
            .map(|&loss| (loss - mean).powi(2))
            .sum::<f32>() / recent_losses.len() as f32;
        
        // Return negative variance as convergence metric (lower variance = better convergence)
        -variance
    }

    /// Get adaptation history
    pub fn get_adaptation_history(&self) -> &[AdaptationResult] {
        &self.adaptation_history
    }

    /// Get current base parameters
    pub fn get_base_parameters(&self) -> &ModelParameters {
        &self.base_params
    }

    /// Evaluate on a specific task
    pub async fn evaluate_task(&self, task: &Task) -> Result<f32> {
        let adapted_params = self.base_params.clone_for_adaptation();
        self.compute_loss(&adapted_params, &task.query_set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_maml_creation() {
        let config = MAMLConfig::default();
        let maml = MAML::new(config);
        assert_eq!(maml.adaptation_history.len(), 0);
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
                support_size: 1,
                query_size: 1,
                created_at: Instant::now(),
            },
        };
        
        let result = maml.adapt_to_task(&task).await;
        assert!(result.is_ok());
        
        let adaptation_result = result.unwrap();
        assert_eq!(adaptation_result.task_id, task.id);
        assert!(adaptation_result.final_loss >= 0.0);
    }

    #[test]
    fn test_forward_pass() {
        let config = MAMLConfig::default();
        let maml = MAML::new(config);
        
        let input = Array1::ones(128);
        let result = maml.forward(&maml.base_params, &input);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 128); // Output dimension should match config
    }

    #[test]
    fn test_activation_functions() {
        let config = MAMLConfig::default();
        let maml = MAML::new(config);
        
        let input = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        
        let relu_output = maml.apply_activation(&input, "relu");
        assert_eq!(relu_output[0], 0.0);
        assert_eq!(relu_output[1], 0.0);
        assert_eq!(relu_output[2], 1.0);
        
        let sigmoid_output = maml.apply_activation(&input, "sigmoid");
        assert!(sigmoid_output[0] < 0.5);
        assert_eq!(sigmoid_output[1], 0.5);
        assert!(sigmoid_output[2] > 0.5);
    }
}