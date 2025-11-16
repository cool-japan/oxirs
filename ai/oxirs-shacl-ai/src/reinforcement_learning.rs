//! Reinforcement Learning for SHACL Validation Optimization
//!
//! This module provides RL-based optimization for SHACL validation strategies,
//! constraint ordering, and resource allocation decisions.

use crate::{Result, ShaclAiError};

use chrono::{DateTime, Utc};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use scirs2_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Reinforcement learning agent for SHACL validation optimization
#[derive(Debug)]
pub struct RlValidationAgent {
    /// Q-learning network
    q_network: Arc<Mutex<QNetwork>>,

    /// Experience replay buffer
    replay_buffer: Arc<Mutex<ReplayBuffer>>,

    /// Configuration
    config: RlConfig,

    /// Training statistics
    stats: RlStatistics,

    /// Policy (epsilon-greedy, softmax, etc.)
    policy: Policy,

    /// Random number generator
    rng: Random,
}

/// Configuration for RL agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlConfig {
    /// State dimension
    pub state_dim: usize,

    /// Action dimension (number of possible actions)
    pub action_dim: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Discount factor (gamma)
    pub discount_factor: f64,

    /// Exploration rate (epsilon)
    pub epsilon: f64,

    /// Epsilon decay rate
    pub epsilon_decay: f64,

    /// Minimum epsilon
    pub epsilon_min: f64,

    /// Replay buffer size
    pub replay_buffer_size: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Target network update frequency
    pub target_update_freq: usize,

    /// Reward scaling factor
    pub reward_scale: f64,

    /// Enable double DQN
    pub enable_double_dqn: bool,

    /// Enable prioritized experience replay
    pub enable_prioritized_replay: bool,

    /// Enable dueling DQN architecture
    pub enable_dueling_dqn: bool,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            state_dim: 64,
            action_dim: 10,
            learning_rate: 0.001,
            discount_factor: 0.99,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            replay_buffer_size: 10000,
            batch_size: 32,
            target_update_freq: 100,
            reward_scale: 1.0,
            enable_double_dqn: true,
            enable_prioritized_replay: true,
            enable_dueling_dqn: true,
        }
    }
}

/// Action for SHACL validation optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationAction {
    /// Prioritize constraint type
    PrioritizeConstraint(usize),

    /// Adjust batch size
    AdjustBatchSize(usize),

    /// Enable/disable parallel processing
    ToggleParallel(bool),

    /// Adjust cache size
    AdjustCacheSize(usize),

    /// Change validation order
    ChangeValidationOrder(usize),

    /// Skip low-priority validations
    SkipLowPriority,

    /// Use fast validation mode
    UseFastMode,

    /// Use thorough validation mode
    UseThoroughMode,

    /// Allocate more resources
    AllocateResources(usize),

    /// No action (continue with current strategy)
    NoAction,
}

/// State representation for validation environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationState {
    /// Current validation statistics
    pub validation_stats: ValidationStats,

    /// Resource usage
    pub resource_usage: ResourceUsage,

    /// Queue depth
    pub queue_depth: usize,

    /// Average processing time
    pub avg_processing_time_ms: f64,

    /// Error rate
    pub error_rate: f64,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Validation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_validations: usize,
    pub successful_validations: usize,
    pub failed_validations: usize,
    pub avg_time_ms: f64,
    pub throughput: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f64,
    pub memory_mb: f64,
    pub disk_io_mb: f64,
    pub network_mb: f64,
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Array1<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
    pub priority: f64,
}

/// Q-Network for approximating action values
#[derive(Debug)]
pub struct QNetwork {
    /// Network weights
    weights: Vec<Array2<f64>>,

    /// Network biases
    biases: Vec<Array1<f64>>,

    /// Target network weights (for stable learning)
    target_weights: Vec<Array2<f64>>,

    /// Target network biases
    target_biases: Vec<Array1<f64>>,

    /// Network architecture
    layer_sizes: Vec<usize>,
}

impl QNetwork {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(hidden_dims);
        layer_sizes.push(output_dim);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Initialize weights with Xavier/He initialization
        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];
            let scale = (2.0 / cols as f64).sqrt();

            let weight = Array2::from_shape_fn((rows, cols), |_| {
                let mut rng = Random::default();
                (rng.random::<f64>() - 0.5) * scale
            });

            let bias = Array1::zeros(rows);

            weights.push(weight);
            biases.push(bias);
        }

        // Clone for target network
        let target_weights = weights.clone();
        let target_biases = biases.clone();

        Self {
            weights,
            biases,
            target_weights,
            target_biases,
            layer_sizes,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, state: &Array1<f64>) -> Result<Array1<f64>> {
        let mut activation = state.clone();

        for (i, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            // Linear transformation: activation = W * x + b
            let mut output = bias.clone();
            for (j, row) in weight.outer_iter().enumerate() {
                let dot_product: f64 = row.iter().zip(activation.iter()).map(|(w, a)| w * a).sum();
                output[j] += dot_product;
            }

            // Apply ReLU activation (except last layer)
            if i < self.weights.len() - 1 {
                activation = output.mapv(|x| x.max(0.0));
            } else {
                activation = output;
            }
        }

        Ok(activation)
    }

    /// Forward pass using target network
    pub fn forward_target(&self, state: &Array1<f64>) -> Result<Array1<f64>> {
        let mut activation = state.clone();

        for (i, (weight, bias)) in self
            .target_weights
            .iter()
            .zip(&self.target_biases)
            .enumerate()
        {
            let mut output = bias.clone();
            for (j, row) in weight.outer_iter().enumerate() {
                let dot_product: f64 = row.iter().zip(activation.iter()).map(|(w, a)| w * a).sum();
                output[j] += dot_product;
            }

            if i < self.target_weights.len() - 1 {
                activation = output.mapv(|x| x.max(0.0));
            } else {
                activation = output;
            }
        }

        Ok(activation)
    }

    /// Update target network to match current network
    pub fn update_target_network(&mut self) {
        self.target_weights = self.weights.clone();
        self.target_biases = self.biases.clone();
    }

    /// Train the network on a batch of experiences
    pub fn train_batch(
        &mut self,
        batch: &[Experience],
        learning_rate: f64,
        discount_factor: f64,
    ) -> Result<f64> {
        let mut total_loss = 0.0;

        for experience in batch {
            // Compute target Q-value
            let next_q_values = self.forward_target(&experience.next_state)?;
            let max_next_q = next_q_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let target_q = if experience.done {
                experience.reward
            } else {
                experience.reward + discount_factor * max_next_q
            };

            // Compute current Q-values
            let current_q_values = self.forward(&experience.state)?;
            let current_q = current_q_values[experience.action];

            // Compute TD error
            let td_error = target_q - current_q;
            total_loss += td_error * td_error;

            // Simplified gradient descent update (in practice, use backpropagation)
            // This is a placeholder for proper backprop implementation
            let gradient_scale = learning_rate * td_error;

            // Update output layer weights (simplified)
            if let Some(last_weight) = self.weights.last_mut() {
                for row in last_weight.outer_iter_mut() {
                    for val in row {
                        *val += gradient_scale * 0.001; // Simplified update
                    }
                }
            }
        }

        Ok(total_loss / batch.len() as f64)
    }
}

/// Experience replay buffer
#[derive(Debug)]
pub struct ReplayBuffer {
    /// Buffer storage
    buffer: VecDeque<Experience>,

    /// Maximum buffer size
    max_size: usize,

    /// Prioritized replay enabled
    prioritized: bool,
}

impl ReplayBuffer {
    pub fn new(max_size: usize, prioritized: bool) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            prioritized,
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample a batch of experiences
    pub fn sample(&self, batch_size: usize, rng: &mut Random) -> Vec<Experience> {
        let sample_size = batch_size.min(self.buffer.len());
        let mut samples = Vec::with_capacity(sample_size);

        if self.prioritized {
            // Prioritized sampling based on TD error
            let total_priority: f64 = self.buffer.iter().map(|e| e.priority.abs()).sum();

            for _ in 0..sample_size {
                let mut target = rng.random::<f64>() * total_priority;
                for exp in &self.buffer {
                    target -= exp.priority.abs();
                    if target <= 0.0 {
                        samples.push(exp.clone());
                        break;
                    }
                }
            }
        } else {
            // Uniform random sampling
            for _ in 0..sample_size {
                let idx =
                    (rng.random::<f64>() * self.buffer.len() as f64) as usize % self.buffer.len();
                if let Some(exp) = self.buffer.get(idx) {
                    samples.push(exp.clone());
                }
            }
        }

        samples
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Policy for action selection
#[derive(Debug, Clone)]
pub enum Policy {
    /// Epsilon-greedy policy
    EpsilonGreedy { epsilon: f64 },

    /// Softmax (Boltzmann) policy
    Softmax { temperature: f64 },

    /// Greedy policy (always best action)
    Greedy,
}

impl RlValidationAgent {
    /// Create a new RL validation agent
    pub fn new(config: RlConfig) -> Result<Self> {
        let q_network = QNetwork::new(
            config.state_dim,
            vec![128, 128], // Hidden layers
            config.action_dim,
        );

        Ok(Self {
            q_network: Arc::new(Mutex::new(q_network)),
            replay_buffer: Arc::new(Mutex::new(ReplayBuffer::new(
                config.replay_buffer_size,
                config.enable_prioritized_replay,
            ))),
            config: config.clone(),
            stats: RlStatistics::default(),
            policy: Policy::EpsilonGreedy {
                epsilon: config.epsilon,
            },
            rng: Random::default(),
        })
    }

    /// Select an action based on current state
    pub fn select_action(&mut self, state: &ValidationState) -> Result<ValidationAction> {
        let state_vector = self.state_to_vector(state)?;

        let q_values = {
            let q_network = self.q_network.lock().map_err(|e| {
                ShaclAiError::ProcessingError(format!("Failed to lock Q-network: {}", e))
            })?;

            q_network.forward(&state_vector)?
        }; // q_network lock released here

        let action_idx = match &self.policy {
            Policy::EpsilonGreedy { epsilon } => {
                if self.rng.random::<f64>() < *epsilon {
                    // Explore: random action
                    (self.rng.random::<f64>() * self.config.action_dim as f64) as usize
                        % self.config.action_dim
                } else {
                    // Exploit: best action
                    self.argmax(&q_values)
                }
            }
            Policy::Softmax { temperature } => {
                // Softmax action selection
                self.softmax_sample(&q_values, *temperature)?
            }
            Policy::Greedy => {
                // Always best action
                self.argmax(&q_values)
            }
        };

        Ok(self.idx_to_action(action_idx))
    }

    /// Train the agent on collected experience
    pub fn train(&mut self, experience: Experience) -> Result<()> {
        // Add experience to replay buffer
        let mut replay_buffer = self.replay_buffer.lock().map_err(|e| {
            ShaclAiError::ProcessingError(format!("Failed to lock replay buffer: {}", e))
        })?;

        replay_buffer.add(experience);

        // Train if enough samples
        if replay_buffer.len() >= self.config.batch_size {
            let batch = replay_buffer.sample(self.config.batch_size, &mut self.rng);
            drop(replay_buffer); // Release lock

            let mut q_network = self.q_network.lock().map_err(|e| {
                ShaclAiError::ProcessingError(format!("Failed to lock Q-network: {}", e))
            })?;

            let loss = q_network.train_batch(
                &batch,
                self.config.learning_rate,
                self.config.discount_factor,
            )?;

            self.stats.total_training_steps += 1;
            self.stats.total_loss += loss;
            self.stats.avg_loss = self.stats.total_loss / self.stats.total_training_steps as f64;

            // Update target network periodically
            if self.stats.total_training_steps % self.config.target_update_freq == 0 {
                q_network.update_target_network();
            }
        }

        // Decay epsilon
        if let Policy::EpsilonGreedy { epsilon } = &mut self.policy {
            *epsilon = (*epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);
        }

        Ok(())
    }

    /// Compute reward based on validation performance
    pub fn compute_reward(&self, prev_state: &ValidationState, new_state: &ValidationState) -> f64 {
        let mut reward = 0.0;

        // Reward for improved throughput
        let throughput_improvement =
            new_state.validation_stats.throughput - prev_state.validation_stats.throughput;
        reward += throughput_improvement * 10.0;

        // Penalty for increased error rate
        let error_rate_increase = new_state.error_rate - prev_state.error_rate;
        reward -= error_rate_increase * 100.0;

        // Reward for improved cache hit rate
        let cache_improvement = new_state.cache_hit_rate - prev_state.cache_hit_rate;
        reward += cache_improvement * 50.0;

        // Penalty for increased resource usage
        let cpu_increase =
            new_state.resource_usage.cpu_percent - prev_state.resource_usage.cpu_percent;
        reward -= cpu_increase * 0.5;

        // Reward for reduced processing time
        let time_reduction = prev_state.avg_processing_time_ms - new_state.avg_processing_time_ms;
        reward += time_reduction * 0.1;

        reward * self.config.reward_scale
    }

    /// Convert state to vector representation
    fn state_to_vector(&self, state: &ValidationState) -> Result<Array1<f64>> {
        let mut vector = Array1::zeros(self.config.state_dim);

        // Normalize and pack state features
        let idx = 0;
        if idx < self.config.state_dim {
            vector[idx] = (state.validation_stats.total_validations as f64).ln() / 10.0;
        }
        let idx = 1;
        if idx < self.config.state_dim {
            vector[idx] = state.validation_stats.throughput / 1000.0;
        }
        let idx = 2;
        if idx < self.config.state_dim {
            vector[idx] = state.error_rate;
        }
        let idx = 3;
        if idx < self.config.state_dim {
            vector[idx] = state.cache_hit_rate;
        }
        let idx = 4;
        if idx < self.config.state_dim {
            vector[idx] = state.resource_usage.cpu_percent / 100.0;
        }
        let idx = 5;
        if idx < self.config.state_dim {
            vector[idx] = state.resource_usage.memory_mb / 1024.0;
        }
        let idx = 6;
        if idx < self.config.state_dim {
            vector[idx] = (state.queue_depth as f64).ln() / 10.0;
        }
        let idx = 7;
        if idx < self.config.state_dim {
            vector[idx] = state.avg_processing_time_ms / 1000.0;
        }

        Ok(vector)
    }

    /// Convert action index to action
    fn idx_to_action(&self, idx: usize) -> ValidationAction {
        match idx % 10 {
            0 => ValidationAction::PrioritizeConstraint(idx / 10),
            1 => ValidationAction::AdjustBatchSize(32 * (1 + idx / 10)),
            2 => ValidationAction::ToggleParallel(idx % 2 == 0),
            3 => ValidationAction::AdjustCacheSize(1024 * (1 + idx / 10)),
            4 => ValidationAction::ChangeValidationOrder(idx / 10),
            5 => ValidationAction::SkipLowPriority,
            6 => ValidationAction::UseFastMode,
            7 => ValidationAction::UseThoroughMode,
            8 => ValidationAction::AllocateResources(idx / 10),
            _ => ValidationAction::NoAction,
        }
    }

    /// Find index of maximum value
    fn argmax(&self, values: &Array1<f64>) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Softmax sampling
    fn softmax_sample(&mut self, q_values: &Array1<f64>, temperature: f64) -> Result<usize> {
        // Compute softmax probabilities
        let max_q = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = q_values
            .iter()
            .map(|&q| ((q - max_q) / temperature).exp())
            .collect();
        let sum_exp: f64 = exp_values.iter().sum();
        let probabilities: Vec<f64> = exp_values.iter().map(|e| e / sum_exp).collect();

        // Sample from distribution
        let mut target = self.rng.random::<f64>();
        for (idx, &prob) in probabilities.iter().enumerate() {
            target -= prob;
            if target <= 0.0 {
                return Ok(idx);
            }
        }

        Ok(probabilities.len() - 1)
    }

    /// Get training statistics
    pub fn get_stats(&self) -> &RlStatistics {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &RlConfig {
        &self.config
    }
}

/// Training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RlStatistics {
    pub total_training_steps: usize,
    pub total_loss: f64,
    pub avg_loss: f64,
    pub total_episodes: usize,
    pub total_rewards: f64,
    pub avg_reward: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rl_agent_creation() {
        let config = RlConfig::default();
        let agent = RlValidationAgent::new(config).unwrap();

        assert_eq!(agent.config.state_dim, 64);
        assert_eq!(agent.config.action_dim, 10);
    }

    #[test]
    fn test_q_network_forward() {
        let network = QNetwork::new(10, vec![20, 20], 5);
        let state = Array1::zeros(10);

        let output = network.forward(&state).unwrap();
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100, false);
        let experience = Experience {
            state: Array1::zeros(10),
            action: 0,
            reward: 1.0,
            next_state: Array1::zeros(10),
            done: false,
            priority: 1.0,
        };

        buffer.add(experience);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_action_selection() {
        let config = RlConfig::default();
        let mut agent = RlValidationAgent::new(config).unwrap();

        let state = ValidationState {
            validation_stats: ValidationStats::default(),
            resource_usage: ResourceUsage::default(),
            queue_depth: 10,
            avg_processing_time_ms: 50.0,
            error_rate: 0.01,
            cache_hit_rate: 0.8,
            timestamp: Utc::now(),
        };

        let action = agent.select_action(&state).unwrap();
        // Action should be one of the valid variants (any of the 10 types)
        // Just verify it's a valid enum variant by pattern matching
        match action {
            ValidationAction::PrioritizeConstraint(_)
            | ValidationAction::AdjustBatchSize(_)
            | ValidationAction::ToggleParallel(_)
            | ValidationAction::AdjustCacheSize(_)
            | ValidationAction::ChangeValidationOrder(_)
            | ValidationAction::SkipLowPriority
            | ValidationAction::UseFastMode
            | ValidationAction::UseThoroughMode
            | ValidationAction::AllocateResources(_)
            | ValidationAction::NoAction => { /* Valid action */ }
        }
    }

    #[test]
    fn test_reward_computation() {
        let config = RlConfig::default();
        let agent = RlValidationAgent::new(config).unwrap();

        let prev_state = ValidationState {
            validation_stats: ValidationStats {
                throughput: 100.0,
                ..Default::default()
            },
            resource_usage: ResourceUsage::default(),
            queue_depth: 10,
            avg_processing_time_ms: 100.0,
            error_rate: 0.05,
            cache_hit_rate: 0.7,
            timestamp: Utc::now(),
        };

        let new_state = ValidationState {
            validation_stats: ValidationStats {
                throughput: 150.0, // Improved
                ..Default::default()
            },
            resource_usage: ResourceUsage::default(),
            queue_depth: 5,
            avg_processing_time_ms: 80.0, // Improved
            error_rate: 0.03,             // Improved
            cache_hit_rate: 0.85,         // Improved
            timestamp: Utc::now(),
        };

        let reward = agent.compute_reward(&prev_state, &new_state);
        assert!(reward > 0.0); // Should be positive for improvements
    }
}
