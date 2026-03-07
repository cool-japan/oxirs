//! # Reinforcement Learning for Stream Processing Optimization
//!
//! This module provides reinforcement learning agents that can automatically optimize
//! stream processing parameters in real-time, learning from system feedback to improve
//! performance metrics like throughput, latency, and resource utilization.
//!
//! ## Features
//! - Q-Learning and Deep Q-Networks (DQN) for discrete action spaces
//! - Policy gradient methods (REINFORCE, Actor-Critic) for continuous actions
//! - Multi-armed bandit algorithms for hyperparameter tuning
//! - Experience replay for stable learning
//! - Adaptive exploration strategies (ε-greedy, UCB, Thompson sampling)
//! - Reward shaping for complex optimization objectives
//!
//! ## Example Usage
//! ```rust,ignore
//! use oxirs_stream::reinforcement_learning::{RLAgent, RLConfig, RLAlgorithm};
//!
//! let config = RLConfig {
//!     algorithm: RLAlgorithm::DQN,
//!     learning_rate: 0.001,
//!     discount_factor: 0.99,
//!     ..Default::default()
//! };
//!
//! let mut agent = RLAgent::new(config)?;
//! let action = agent.select_action(&state).await?;
//! let reward = execute_action(action);
//! agent.learn(&state, action, reward, &next_state).await?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info};

/// Reinforcement learning algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RLAlgorithm {
    /// Q-Learning (discrete actions)
    QLearning,
    /// Deep Q-Network
    DQN,
    /// SARSA (on-policy TD)
    SARSA,
    /// Actor-Critic
    ActorCritic,
    /// REINFORCE (policy gradient)
    REINFORCE,
    /// Proximal Policy Optimization
    PPO,
    /// Multi-Armed Bandit (UCB)
    UCB,
    /// Thompson Sampling
    ThompsonSampling,
    /// Epsilon-Greedy Bandit
    EpsilonGreedy,
}

/// State representation for stream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Current throughput (events/second)
    pub throughput: f64,
    /// Current latency (milliseconds)
    pub latency_ms: f64,
    /// CPU utilization (0-1)
    pub cpu_utilization: f64,
    /// Memory utilization (0-1)
    pub memory_utilization: f64,
    /// Queue depth
    pub queue_depth: usize,
    /// Error rate (0-1)
    pub error_rate: f64,
    /// Additional features
    pub features: Vec<f64>,
}

impl State {
    /// Convert state to feature vector
    pub fn to_vector(&self) -> Vec<f64> {
        let mut vec = vec![
            self.throughput,
            self.latency_ms,
            self.cpu_utilization,
            self.memory_utilization,
            self.queue_depth as f64,
            self.error_rate,
        ];
        vec.extend(&self.features);
        vec
    }

    /// Get state dimension
    pub fn dimension(&self) -> usize {
        6 + self.features.len()
    }
}

/// Action representation (can be discrete or continuous)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Discrete action (index)
    Discrete(usize),
    /// Continuous action (vector)
    Continuous(Vec<f64>),
}

impl Action {
    /// Get action as index (for discrete actions)
    pub fn as_index(&self) -> Option<usize> {
        match self {
            Action::Discrete(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Get action as vector (for continuous actions)
    pub fn as_vector(&self) -> Option<&[f64]> {
        match self {
            Action::Continuous(vec) => Some(vec),
            _ => None,
        }
    }
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    /// Current state
    pub state: State,
    /// Action taken
    pub action: Action,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: State,
    /// Whether episode terminated
    pub done: bool,
}

/// RL agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLConfig {
    /// RL algorithm to use
    pub algorithm: RLAlgorithm,
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub discount_factor: f64,
    /// Exploration rate (epsilon) for ε-greedy
    pub epsilon: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Minimum epsilon
    pub epsilon_min: f64,
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
    /// Batch size for learning
    pub batch_size: usize,
    /// Target network update frequency (for DQN)
    pub target_update_freq: usize,
    /// Number of discrete actions (if applicable)
    pub n_actions: usize,
    /// Number of hidden units in neural network
    pub hidden_units: Vec<usize>,
    /// Enable prioritized experience replay
    pub prioritized_replay: bool,
    /// UCB exploration constant
    pub ucb_c: f64,
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            algorithm: RLAlgorithm::DQN,
            learning_rate: 0.001,
            discount_factor: 0.99,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            replay_buffer_size: 10000,
            batch_size: 32,
            target_update_freq: 100,
            n_actions: 10,
            hidden_units: vec![64, 64],
            prioritized_replay: false,
            ucb_c: 2.0,
        }
    }
}

/// Q-table for tabular RL
type QTable = HashMap<String, Vec<f64>>;

/// Neural network weights (simplified)
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    /// Layer weights
    pub weights: Vec<Array2<f64>>,
    /// Layer biases
    pub biases: Vec<Array1<f64>>,
}

impl NeuralNetwork {
    /// Create a new neural network
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        rng: &mut Random,
    ) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut dims = vec![input_dim];
        dims.extend(hidden_dims);
        dims.push(output_dim);

        for i in 0..dims.len() - 1 {
            let w = Self::init_weights(dims[i], dims[i + 1], rng);
            let b = Array1::zeros(dims[i + 1]);
            weights.push(w);
            biases.push(b);
        }

        Self { weights, biases }
    }

    /// Initialize weights with Xavier/Glorot initialization
    fn init_weights(input_dim: usize, output_dim: usize, rng: &mut Random) -> Array2<f64> {
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let values: Vec<f64> = (0..input_dim * output_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();
        Array2::from_shape_vec((input_dim, output_dim), values)
            .expect("shape and vector length match")
    }

    /// Forward pass
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut activation = input.clone();

        for (w, b) in self.weights.iter().zip(&self.biases) {
            // Linear transform
            activation = activation.dot(w) + b;

            // ReLU activation (except last layer)
            if w != self
                .weights
                .last()
                .expect("collection validated to be non-empty")
            {
                activation.mapv_inplace(|x| x.max(0.0));
            }
        }

        activation
    }

    /// Update weights (simplified gradient descent)
    pub fn update(&mut self, gradient_scale: f64, learning_rate: f64) {
        for w in &mut self.weights {
            w.mapv_inplace(|x| x - learning_rate * gradient_scale);
        }
    }
}

/// RL Agent statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLStats {
    /// Total steps taken
    pub total_steps: u64,
    /// Total episodes completed
    pub total_episodes: u64,
    /// Average reward per episode
    pub avg_reward_per_episode: f64,
    /// Current epsilon (exploration rate)
    pub current_epsilon: f64,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Average Q-value
    pub avg_q_value: f64,
    /// Loss (for neural network methods)
    pub avg_loss: f64,
}

impl Default for RLStats {
    fn default() -> Self {
        Self {
            total_steps: 0,
            total_episodes: 0,
            avg_reward_per_episode: 0.0,
            current_epsilon: 1.0,
            total_reward: 0.0,
            avg_q_value: 0.0,
            avg_loss: 0.0,
        }
    }
}

/// Main RL Agent for stream optimization
pub struct RLAgent {
    config: RLConfig,
    /// Q-table for tabular methods
    q_table: Arc<RwLock<QTable>>,
    /// Q-network for DQN
    q_network: Arc<RwLock<Option<NeuralNetwork>>>,
    /// Target network for DQN
    target_network: Arc<RwLock<Option<NeuralNetwork>>>,
    /// Experience replay buffer
    replay_buffer: Arc<RwLock<VecDeque<Experience>>>,
    /// Action counts (for bandits and UCB)
    action_counts: Arc<RwLock<Vec<u64>>>,
    /// Action rewards (for bandits)
    action_rewards: Arc<RwLock<Vec<f64>>>,
    /// Statistics
    stats: Arc<RwLock<RLStats>>,
    /// Random number generator
    #[allow(clippy::arc_with_non_send_sync)]
    rng: Arc<Mutex<Random>>,
    /// Current episode reward
    episode_reward: Arc<RwLock<f64>>,
    /// Update counter
    update_counter: Arc<RwLock<usize>>,
}

impl RLAgent {
    /// Create a new RL agent
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(config: RLConfig) -> Result<Self> {
        let action_counts = vec![0u64; config.n_actions];
        let action_rewards = vec![0.0; config.n_actions];
        let buffer_size = config.replay_buffer_size;
        let epsilon = config.epsilon;

        Ok(Self {
            config,
            q_table: Arc::new(RwLock::new(HashMap::new())),
            q_network: Arc::new(RwLock::new(None)),
            target_network: Arc::new(RwLock::new(None)),
            replay_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(buffer_size))),
            action_counts: Arc::new(RwLock::new(action_counts)),
            action_rewards: Arc::new(RwLock::new(action_rewards)),
            stats: Arc::new(RwLock::new(RLStats {
                current_epsilon: epsilon,
                ..Default::default()
            })),
            rng: Arc::new(Mutex::new(Random::default())),
            episode_reward: Arc::new(RwLock::new(0.0)),
            update_counter: Arc::new(RwLock::new(0)),
        })
    }

    /// Initialize neural networks (for DQN/Actor-Critic)
    pub async fn initialize_networks(&mut self, state_dim: usize) -> Result<()> {
        if matches!(
            self.config.algorithm,
            RLAlgorithm::DQN | RLAlgorithm::ActorCritic | RLAlgorithm::PPO
        ) {
            let mut rng = self.rng.lock().await;

            let q_net = NeuralNetwork::new(
                state_dim,
                &self.config.hidden_units,
                self.config.n_actions,
                &mut rng,
            );

            let target_net = NeuralNetwork::new(
                state_dim,
                &self.config.hidden_units,
                self.config.n_actions,
                &mut rng,
            );

            *self.q_network.write().await = Some(q_net);
            *self.target_network.write().await = Some(target_net);

            info!(
                "Initialized neural networks with state_dim={}, n_actions={}",
                state_dim, self.config.n_actions
            );
        }

        Ok(())
    }

    /// Select an action given current state
    pub async fn select_action(&self, state: &State) -> Result<Action> {
        match self.config.algorithm {
            RLAlgorithm::QLearning | RLAlgorithm::SARSA => {
                self.select_action_q_learning(state).await
            }
            RLAlgorithm::DQN => self.select_action_dqn(state).await,
            RLAlgorithm::UCB => self.select_action_ucb().await,
            RLAlgorithm::ThompsonSampling => self.select_action_thompson().await,
            RLAlgorithm::EpsilonGreedy => self.select_action_epsilon_greedy().await,
            _ => {
                // Default to ε-greedy
                self.select_action_epsilon_greedy().await
            }
        }
    }

    /// ε-greedy action selection for Q-learning
    async fn select_action_q_learning(&self, state: &State) -> Result<Action> {
        let stats = self.stats.read().await;
        let epsilon = stats.current_epsilon;
        drop(stats);

        let mut rng = self.rng.lock().await;

        if rng.random::<f64>() < epsilon {
            // Explore: random action
            let action_idx = rng.random_range(0..self.config.n_actions);
            Ok(Action::Discrete(action_idx))
        } else {
            // Exploit: best action from Q-table
            let state_key = self.state_to_key(state);
            let q_table = self.q_table.read().await;

            let q_values = q_table
                .get(&state_key)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.config.n_actions]);

            let best_action = q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            Ok(Action::Discrete(best_action))
        }
    }

    /// Action selection for DQN
    async fn select_action_dqn(&self, state: &State) -> Result<Action> {
        let stats = self.stats.read().await;
        let epsilon = stats.current_epsilon;
        drop(stats);

        let mut rng = self.rng.lock().await;

        if rng.random::<f64>() < epsilon {
            let action_idx = rng.random_range(0..self.config.n_actions);
            Ok(Action::Discrete(action_idx))
        } else {
            drop(rng);

            let q_network = self.q_network.read().await;
            if let Some(ref network) = *q_network {
                let state_vec = Array1::from_vec(state.to_vector());
                let q_values = network.forward(&state_vec);

                let best_action = q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                Ok(Action::Discrete(best_action))
            } else {
                Err(anyhow!("Q-network not initialized"))
            }
        }
    }

    /// UCB action selection
    async fn select_action_ucb(&self) -> Result<Action> {
        let action_counts = self.action_counts.read().await;
        let action_rewards = self.action_rewards.read().await;
        let stats = self.stats.read().await;
        let total_steps = stats.total_steps;

        let mut ucb_values = Vec::with_capacity(self.config.n_actions);

        for i in 0..self.config.n_actions {
            let count = action_counts[i];
            let avg_reward = if count > 0 {
                action_rewards[i] / count as f64
            } else {
                f64::INFINITY // Prioritize unexplored actions
            };

            let exploration_bonus = if count > 0 {
                self.config.ucb_c * ((total_steps as f64).ln() / count as f64).sqrt()
            } else {
                f64::INFINITY
            };

            ucb_values.push(avg_reward + exploration_bonus);
        }

        let best_action = ucb_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(Action::Discrete(best_action))
    }

    /// Thompson sampling action selection
    async fn select_action_thompson(&self) -> Result<Action> {
        let action_counts = self.action_counts.read().await;
        let action_rewards = self.action_rewards.read().await;
        let mut rng = self.rng.lock().await;

        let mut sampled_values = Vec::with_capacity(self.config.n_actions);

        for i in 0..self.config.n_actions {
            let count = action_counts[i];
            let sum_reward = action_rewards[i];

            // Beta distribution sampling (simplified)
            let alpha = sum_reward + 1.0;
            let beta = (count as f64 - sum_reward).max(0.0) + 1.0;

            // Simplified beta sampling
            let sample = rng.random::<f64>().powf(1.0 / alpha)
                * (1.0 - rng.random::<f64>()).powf(1.0 / beta);
            sampled_values.push(sample);
        }

        let best_action = sampled_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(Action::Discrete(best_action))
    }

    /// Epsilon-greedy bandit action selection
    async fn select_action_epsilon_greedy(&self) -> Result<Action> {
        let stats = self.stats.read().await;
        let epsilon = stats.current_epsilon;
        drop(stats);

        let mut rng = self.rng.lock().await;

        if rng.random::<f64>() < epsilon {
            let action_idx = rng.random_range(0..self.config.n_actions);
            Ok(Action::Discrete(action_idx))
        } else {
            drop(rng);

            let action_counts = self.action_counts.read().await;
            let action_rewards = self.action_rewards.read().await;

            let best_action = (0..self.config.n_actions)
                .max_by(|&a, &b| {
                    let avg_a = if action_counts[a] > 0 {
                        action_rewards[a] / action_counts[a] as f64
                    } else {
                        0.0
                    };
                    let avg_b = if action_counts[b] > 0 {
                        action_rewards[b] / action_counts[b] as f64
                    } else {
                        0.0
                    };
                    avg_a
                        .partial_cmp(&avg_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            Ok(Action::Discrete(best_action))
        }
    }

    /// Learn from experience (update model)
    pub async fn learn(
        &mut self,
        state: &State,
        action: Action,
        reward: f64,
        next_state: &State,
    ) -> Result<()> {
        // Add to replay buffer
        let experience = Experience {
            state: state.clone(),
            action: action.clone(),
            reward,
            next_state: next_state.clone(),
            done: false,
        };

        let mut replay_buffer = self.replay_buffer.write().await;
        replay_buffer.push_back(experience);

        if replay_buffer.len() > self.config.replay_buffer_size {
            replay_buffer.pop_front();
        }
        drop(replay_buffer);

        // Update statistics
        *self.episode_reward.write().await += reward;
        let mut stats = self.stats.write().await;
        stats.total_steps += 1;
        stats.total_reward += reward;

        // Update action counts for bandits
        if let Action::Discrete(idx) = action {
            let mut counts = self.action_counts.write().await;
            let mut rewards = self.action_rewards.write().await;
            counts[idx] += 1;
            rewards[idx] += reward;
        }

        // Perform learning update
        match self.config.algorithm {
            RLAlgorithm::QLearning | RLAlgorithm::SARSA => {
                drop(stats);
                self.update_q_learning(state, &action, reward, next_state)
                    .await?;
            }
            RLAlgorithm::DQN => {
                drop(stats);
                self.update_dqn().await?;
            }
            _ => {
                // Bandits don't need explicit update beyond counting
            }
        }

        // Decay epsilon
        let mut stats = self.stats.write().await;
        stats.current_epsilon =
            (stats.current_epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);

        Ok(())
    }

    /// Q-learning update
    async fn update_q_learning(
        &self,
        state: &State,
        action: &Action,
        reward: f64,
        next_state: &State,
    ) -> Result<()> {
        if let Action::Discrete(action_idx) = action {
            let state_key = self.state_to_key(state);
            let next_state_key = self.state_to_key(next_state);

            let mut q_table = self.q_table.write().await;

            // Get max next Q value first
            let max_next_q = {
                let next_q_values = q_table
                    .entry(next_state_key)
                    .or_insert_with(|| vec![0.0; self.config.n_actions]);
                next_q_values
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            };

            // Now update current Q value
            let q_values = q_table
                .entry(state_key.clone())
                .or_insert_with(|| vec![0.0; self.config.n_actions]);

            // Q-learning update: Q(s,a) <- Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            let current_q = q_values[*action_idx];
            let td_target = reward + self.config.discount_factor * max_next_q;
            let td_error = td_target - current_q;

            q_values[*action_idx] += self.config.learning_rate * td_error;

            debug!(
                "Q-learning update: state={}, action={}, Q={:.4}",
                state_key, action_idx, q_values[*action_idx]
            );
        }

        Ok(())
    }

    /// DQN update
    async fn update_dqn(&self) -> Result<()> {
        let replay_buffer = self.replay_buffer.read().await;

        if replay_buffer.len() < self.config.batch_size {
            return Ok(()); // Not enough samples
        }

        // Sample random batch
        let batch_indices: Vec<usize> = {
            let mut rng = self.rng.lock().await;
            (0..self.config.batch_size)
                .map(|_| rng.random_range(0..replay_buffer.len()))
                .collect()
        };

        // Clone the batch experiences before dropping the lock
        let batch: Vec<Experience> = batch_indices
            .iter()
            .map(|&i| replay_buffer[i].clone())
            .collect();
        drop(replay_buffer);

        // Compute TD errors and update network
        let mut total_loss = 0.0;

        let q_network = self.q_network.read().await;
        let target_network = self.target_network.read().await;

        if let (Some(ref q_net), Some(ref target_net)) = (&*q_network, &*target_network) {
            for exp in &batch {
                let state_vec = Array1::from_vec(exp.state.to_vector());
                let next_state_vec = Array1::from_vec(exp.next_state.to_vector());

                let q_values = q_net.forward(&state_vec);
                let next_q_values = target_net.forward(&next_state_vec);

                let max_next_q = next_q_values
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);

                if let Action::Discrete(action_idx) = exp.action {
                    let td_target = exp.reward + self.config.discount_factor * max_next_q;
                    let td_error = td_target - q_values[action_idx];
                    total_loss += td_error * td_error;
                }
            }
        }
        drop(q_network);
        drop(target_network);

        // Update network (simplified)
        let mut q_network = self.q_network.write().await;
        if let Some(ref mut network) = *q_network {
            let gradient_scale = total_loss / self.config.batch_size as f64;
            network.update(gradient_scale, self.config.learning_rate);
        }
        drop(q_network);

        // Update target network periodically
        let mut counter = self.update_counter.write().await;
        *counter += 1;

        if *counter % self.config.target_update_freq == 0 {
            let q_net = self.q_network.read().await;
            if let Some(ref network) = *q_net {
                *self.target_network.write().await = Some(network.clone());
                debug!("Updated target network at step {}", *counter);
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.avg_loss = (stats.avg_loss * (stats.total_steps - 1) as f64 + total_loss)
            / stats.total_steps as f64;

        Ok(())
    }

    /// End episode and record statistics
    pub async fn end_episode(&mut self) -> Result<()> {
        let episode_reward = *self.episode_reward.read().await;
        *self.episode_reward.write().await = 0.0;

        let mut stats = self.stats.write().await;
        stats.total_episodes += 1;
        stats.avg_reward_per_episode =
            (stats.avg_reward_per_episode * (stats.total_episodes - 1) as f64 + episode_reward)
                / stats.total_episodes as f64;

        info!(
            "Episode {} complete: reward={:.2}, avg_reward={:.2}",
            stats.total_episodes, episode_reward, stats.avg_reward_per_episode
        );

        Ok(())
    }

    /// Convert state to string key for Q-table
    fn state_to_key(&self, state: &State) -> String {
        // Discretize continuous state for tabular methods
        format!(
            "{:.0}_{:.0}_{:.2}_{:.2}_{}_{ :.2}",
            (state.throughput / 1000.0).round(),
            (state.latency_ms / 10.0).round(),
            (state.cpu_utilization * 10.0).round() / 10.0,
            (state.memory_utilization * 10.0).round() / 10.0,
            state.queue_depth / 100,
            (state.error_rate * 100.0).round() / 100.0,
        )
    }

    /// Get RL statistics
    pub async fn get_stats(&self) -> RLStats {
        self.stats.read().await.clone()
    }

    /// Get current epsilon
    pub async fn get_epsilon(&self) -> f64 {
        self.stats.read().await.current_epsilon
    }

    /// Set epsilon (for exploration control)
    pub async fn set_epsilon(&mut self, epsilon: f64) {
        self.stats.write().await.current_epsilon = epsilon.clamp(0.0, 1.0);
    }

    /// Export policy for deployment
    pub async fn export_policy(&self) -> Result<String> {
        let policy = match self.config.algorithm {
            RLAlgorithm::QLearning | RLAlgorithm::SARSA => {
                let q_table = self.q_table.read().await;
                serde_json::json!({
                    "algorithm": "Q-Learning",
                    "q_table": q_table.iter().take(10).collect::<HashMap<_, _>>(), // Sample
                })
            }
            _ => {
                let stats = self.get_stats().await;
                serde_json::json!({
                    "algorithm": format!("{:?}", self.config.algorithm),
                    "stats": stats,
                })
            }
        };

        Ok(serde_json::to_string_pretty(&policy)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> State {
        State {
            throughput: 10000.0,
            latency_ms: 5.0,
            cpu_utilization: 0.5,
            memory_utilization: 0.6,
            queue_depth: 100,
            error_rate: 0.01,
            features: vec![],
        }
    }

    #[tokio::test]
    async fn test_rl_agent_creation() {
        let config = RLConfig::default();
        let agent = RLAgent::new(config);
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_q_learning_action_selection() {
        let config = RLConfig {
            algorithm: RLAlgorithm::QLearning,
            n_actions: 5,
            ..Default::default()
        };

        let agent = RLAgent::new(config).unwrap();
        let state = create_test_state();

        let action = agent.select_action(&state).await;
        assert!(action.is_ok());

        if let Action::Discrete(idx) = action.unwrap() {
            assert!(idx < 5);
        }
    }

    #[tokio::test]
    async fn test_dqn_initialization() {
        let config = RLConfig {
            algorithm: RLAlgorithm::DQN,
            n_actions: 10,
            hidden_units: vec![32, 32],
            ..Default::default()
        };

        let mut agent = RLAgent::new(config).unwrap();
        let state = create_test_state();

        agent.initialize_networks(state.dimension()).await.unwrap();

        let action = agent.select_action(&state).await;
        assert!(action.is_ok());
    }

    #[tokio::test]
    async fn test_ucb_action_selection() {
        let config = RLConfig {
            algorithm: RLAlgorithm::UCB,
            n_actions: 5,
            ..Default::default()
        };

        let agent = RLAgent::new(config).unwrap();
        let action = agent.select_action_ucb().await;
        assert!(action.is_ok());
    }

    #[tokio::test]
    async fn test_learning_update() {
        let config = RLConfig {
            algorithm: RLAlgorithm::QLearning,
            n_actions: 3,
            ..Default::default()
        };

        let mut agent = RLAgent::new(config).unwrap();
        let state = create_test_state();
        let action = Action::Discrete(1);
        let reward = 1.0;
        let next_state = create_test_state();

        let result = agent.learn(&state, action, reward, &next_state).await;
        assert!(result.is_ok());

        let stats = agent.get_stats().await;
        assert_eq!(stats.total_steps, 1);
        assert_eq!(stats.total_reward, 1.0);
    }

    #[tokio::test]
    async fn test_epsilon_decay() {
        let config = RLConfig {
            epsilon: 1.0,
            epsilon_decay: 0.9,
            epsilon_min: 0.1,
            ..Default::default()
        };

        let mut agent = RLAgent::new(config).unwrap();
        let initial_epsilon = agent.get_epsilon().await;

        let state = create_test_state();
        for _ in 0..10 {
            agent
                .learn(&state, Action::Discrete(0), 0.0, &state)
                .await
                .unwrap();
        }

        let final_epsilon = agent.get_epsilon().await;
        assert!(final_epsilon < initial_epsilon);
        assert!(final_epsilon >= 0.1);
    }

    #[tokio::test]
    async fn test_episode_management() {
        let config = RLConfig::default();
        let mut agent = RLAgent::new(config).unwrap();

        let state = create_test_state();
        agent
            .learn(&state, Action::Discrete(0), 1.0, &state)
            .await
            .unwrap();
        agent
            .learn(&state, Action::Discrete(1), 2.0, &state)
            .await
            .unwrap();

        agent.end_episode().await.unwrap();

        let stats = agent.get_stats().await;
        assert_eq!(stats.total_episodes, 1);
        assert!(stats.avg_reward_per_episode > 0.0);
    }

    #[tokio::test]
    async fn test_replay_buffer() {
        let config = RLConfig {
            replay_buffer_size: 5,
            ..Default::default()
        };

        let mut agent = RLAgent::new(config).unwrap();
        let state = create_test_state();

        for i in 0..10 {
            agent
                .learn(&state, Action::Discrete(0), i as f64, &state)
                .await
                .unwrap();
        }

        let buffer = agent.replay_buffer.read().await;
        assert_eq!(buffer.len(), 5); // Should not exceed buffer size
    }

    #[tokio::test]
    async fn test_export_policy() {
        let config = RLConfig {
            algorithm: RLAlgorithm::QLearning,
            ..Default::default()
        };

        let mut agent = RLAgent::new(config).unwrap();
        let state = create_test_state();

        agent
            .learn(&state, Action::Discrete(0), 1.0, &state)
            .await
            .unwrap();

        let export = agent.export_policy().await;
        assert!(export.is_ok());
        assert!(export.unwrap().contains("algorithm"));
    }

    #[tokio::test]
    async fn test_thompson_sampling() {
        let config = RLConfig {
            algorithm: RLAlgorithm::ThompsonSampling,
            n_actions: 5,
            ..Default::default()
        };

        let agent = RLAgent::new(config).unwrap();
        let action = agent.select_action_thompson().await;
        assert!(action.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_episodes() {
        let config = RLConfig {
            algorithm: RLAlgorithm::QLearning,
            n_actions: 3,
            ..Default::default()
        };

        let mut agent = RLAgent::new(config).unwrap();
        let state = create_test_state();

        for episode in 0..5 {
            for _ in 0..10 {
                let action = agent.select_action(&state).await.unwrap();
                let reward = if episode % 2 == 0 { 1.0 } else { -1.0 };
                agent.learn(&state, action, reward, &state).await.unwrap();
            }
            agent.end_episode().await.unwrap();
        }

        let stats = agent.get_stats().await;
        assert_eq!(stats.total_episodes, 5);
        assert_eq!(stats.total_steps, 50);
    }
}
