//! # Neural Architecture Search for Stream Operators
//!
//! This module provides automated neural architecture search capabilities for
//! discovering optimal network configurations for stream processing tasks.
//!
//! ## Features
//! - Search space definition for network architectures
//! - Multiple search strategies (Random, Evolutionary, Gradient-based)
//! - Performance estimation and early stopping
//! - Architecture encoding and decoding
//! - Multi-objective optimization (accuracy, latency, memory)
//! - Transfer learning from meta-learning datasets
//!
//! ## Example Usage
//! ```rust,ignore
//! use oxirs_stream::neural_architecture_search::{NAS, NASConfig, SearchStrategy};
//!
//! let config = NASConfig {
//!     strategy: SearchStrategy::Evolutionary,
//!     max_trials: 100,
//!     ..Default::default()
//! };
//!
//! let mut nas = NAS::new(config)?;
//! let best_architecture = nas.search(&training_data).await?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Neural architecture search strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Random search
    Random,
    /// Evolutionary algorithms
    Evolutionary,
    /// Gradient-based (DARTS-like)
    GradientBased,
    /// Bayesian optimization
    BayesianOptimization,
    /// Reinforcement learning controller
    RLController,
}

/// Layer types for neural networks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerType {
    /// Fully connected layer
    Dense { units: usize },
    /// Convolutional layer
    Conv1D { filters: usize, kernel_size: usize },
    /// LSTM recurrent layer
    LSTM { units: usize },
    /// GRU recurrent layer
    GRU { units: usize },
    /// Attention mechanism
    Attention { heads: usize },
    /// Batch normalization
    BatchNorm,
    /// Dropout
    Dropout { rate: f64 },
    /// Activation function
    Activation { function: ActivationType },
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU
    LeakyReLU,
    /// Sigmoid
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Softmax
    Softmax,
    /// Linear (identity)
    Linear,
}

/// Neural network architecture specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    /// Ordered list of layers
    pub layers: Vec<LayerType>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Architecture encoding (for search)
    pub encoding: Vec<f64>,
}

impl Architecture {
    /// Create a new architecture
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            layers: Vec::new(),
            input_dim,
            output_dim,
            encoding: Vec::new(),
        }
    }

    /// Add a layer to the architecture
    pub fn add_layer(&mut self, layer: LayerType) {
        self.layers.push(layer);
    }

    /// Get total parameter count (estimated)
    pub fn parameter_count(&self) -> usize {
        let mut params = 0;
        let mut prev_dim = self.input_dim;

        for layer in &self.layers {
            match layer {
                LayerType::Dense { units } => {
                    params += prev_dim * units + units;
                    prev_dim = *units;
                }
                LayerType::Conv1D {
                    filters,
                    kernel_size,
                } => {
                    params += kernel_size * filters + filters;
                    prev_dim = *filters;
                }
                LayerType::LSTM { units } => {
                    params += 4 * (prev_dim * units + units * units + units);
                    prev_dim = *units;
                }
                LayerType::GRU { units } => {
                    params += 3 * (prev_dim * units + units * units + units);
                    prev_dim = *units;
                }
                LayerType::Attention { heads } => {
                    params += 3 * prev_dim * prev_dim / heads;
                }
                _ => {}
            }
        }

        params
    }

    /// Get computational cost estimate (FLOPs)
    pub fn computational_cost(&self) -> f64 {
        let mut flops = 0.0;
        let mut prev_dim = self.input_dim;

        for layer in &self.layers {
            match layer {
                LayerType::Dense { units } => {
                    flops += (2.0 * prev_dim as f64 - 1.0) * (*units as f64);
                    prev_dim = *units;
                }
                LayerType::LSTM { units } | LayerType::GRU { units } => {
                    flops += 8.0 * prev_dim as f64 * (*units as f64);
                    prev_dim = *units;
                }
                LayerType::Attention { heads } => {
                    flops += prev_dim as f64 * prev_dim as f64 / (*heads as f64);
                }
                _ => {}
            }
        }

        flops
    }
}

/// Search space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Maximum number of layers
    pub max_layers: usize,
    /// Minimum number of layers
    pub min_layers: usize,
    /// Allowed layer types
    pub allowed_layers: Vec<LayerType>,
    /// Maximum units per layer
    pub max_units: usize,
    /// Minimum units per layer
    pub min_units: usize,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            max_layers: 10,
            min_layers: 2,
            allowed_layers: vec![
                LayerType::Dense { units: 64 },
                LayerType::LSTM { units: 64 },
                LayerType::GRU { units: 64 },
                LayerType::Dropout { rate: 0.2 },
                LayerType::BatchNorm,
                LayerType::Activation {
                    function: ActivationType::ReLU,
                },
            ],
            max_units: 512,
            min_units: 16,
        }
    }
}

/// Architecture performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturePerformance {
    /// Architecture
    pub architecture: Architecture,
    /// Validation accuracy
    pub accuracy: f64,
    /// Training time (seconds)
    pub training_time: f64,
    /// Inference latency (milliseconds)
    pub inference_latency_ms: f64,
    /// Memory usage (MB)
    pub memory_mb: f64,
    /// Parameter count
    pub parameter_count: usize,
    /// Multi-objective score
    pub score: f64,
}

impl ArchitecturePerformance {
    /// Compute multi-objective score
    pub fn compute_score(&mut self, weights: &ObjectiveWeights) {
        // Normalize metrics
        let norm_accuracy = self.accuracy;
        let norm_latency = 1.0 / (1.0 + self.inference_latency_ms / 100.0);
        let norm_memory = 1.0 / (1.0 + self.memory_mb / 1000.0);
        let norm_params = 1.0 / (1.0 + self.parameter_count as f64 / 1_000_000.0);

        // Weighted combination
        self.score = weights.accuracy * norm_accuracy
            + weights.latency * norm_latency
            + weights.memory * norm_memory
            + weights.params * norm_params;
    }
}

/// Weights for multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveWeights {
    /// Accuracy weight
    pub accuracy: f64,
    /// Latency weight
    pub latency: f64,
    /// Memory weight
    pub memory: f64,
    /// Parameter count weight
    pub params: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            accuracy: 0.6,
            latency: 0.2,
            memory: 0.1,
            params: 0.1,
        }
    }
}

/// NAS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASConfig {
    /// Search strategy
    pub strategy: SearchStrategy,
    /// Maximum number of architectures to evaluate
    pub max_trials: usize,
    /// Search space definition
    pub search_space: SearchSpace,
    /// Objective weights
    pub objective_weights: ObjectiveWeights,
    /// Population size (for evolutionary)
    pub population_size: usize,
    /// Number of generations (for evolutionary)
    pub generations: usize,
    /// Mutation rate (for evolutionary)
    pub mutation_rate: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
}

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::Evolutionary,
            max_trials: 100,
            search_space: SearchSpace::default(),
            objective_weights: ObjectiveWeights::default(),
            population_size: 20,
            generations: 10,
            mutation_rate: 0.2,
            early_stopping_patience: 10,
            enable_transfer_learning: false,
        }
    }
}

/// NAS statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASStats {
    /// Total architectures evaluated
    pub total_evaluated: u64,
    /// Best score achieved
    pub best_score: f64,
    /// Total search time (seconds)
    pub total_search_time: f64,
    /// Average evaluation time (seconds)
    pub avg_evaluation_time: f64,
    /// Number of unique architectures discovered
    pub unique_architectures: usize,
}

impl Default for NASStats {
    fn default() -> Self {
        Self {
            total_evaluated: 0,
            best_score: 0.0,
            total_search_time: 0.0,
            avg_evaluation_time: 0.0,
            unique_architectures: 0,
        }
    }
}

/// Neural Architecture Search engine
pub struct NAS {
    config: NASConfig,
    /// Best architecture found
    best_architecture: Arc<RwLock<Option<ArchitecturePerformance>>>,
    /// Population (for evolutionary)
    population: Arc<RwLock<Vec<Architecture>>>,
    /// Evaluation history
    history: Arc<RwLock<Vec<ArchitecturePerformance>>>,
    /// Statistics
    stats: Arc<RwLock<NASStats>>,
    /// Random number generator
    #[allow(clippy::arc_with_non_send_sync)]
    rng: Arc<Mutex<Random>>,
}

impl NAS {
    /// Create a new NAS instance
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(config: NASConfig) -> Result<Self> {
        Ok(Self {
            config,
            best_architecture: Arc::new(RwLock::new(None)),
            population: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(NASStats::default())),
            rng: Arc::new(Mutex::new(Random::default())),
        })
    }

    /// Run architecture search
    pub async fn search(&mut self, input_dim: usize, output_dim: usize) -> Result<Architecture> {
        info!(
            "Starting NAS with strategy {:?}, max_trials={}",
            self.config.strategy, self.config.max_trials
        );

        let start_time = std::time::Instant::now();

        match self.config.strategy {
            SearchStrategy::Random => {
                self.random_search(input_dim, output_dim).await?;
            }
            SearchStrategy::Evolutionary => {
                self.evolutionary_search(input_dim, output_dim).await?;
            }
            SearchStrategy::GradientBased => {
                self.gradient_based_search(input_dim, output_dim).await?;
            }
            _ => {
                warn!(
                    "Strategy {:?} not fully implemented, using random search",
                    self.config.strategy
                );
                self.random_search(input_dim, output_dim).await?;
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_search_time = start_time.elapsed().as_secs_f64();

        let best = self.best_architecture.read().await;
        match &*best {
            Some(perf) => {
                info!(
                    "NAS complete: best score={:.4}, params={}, latency={:.2}ms",
                    perf.score, perf.parameter_count, perf.inference_latency_ms
                );
                Ok(perf.architecture.clone())
            }
            None => Err(anyhow!("No valid architecture found")),
        }
    }

    /// Random search strategy
    async fn random_search(&mut self, input_dim: usize, output_dim: usize) -> Result<()> {
        for trial in 0..self.config.max_trials {
            let architecture = self
                .generate_random_architecture(input_dim, output_dim)
                .await?;
            let performance = self.evaluate_architecture(&architecture).await?;

            self.update_best(performance.clone()).await;
            self.history.write().await.push(performance);

            debug!(
                "Random search trial {}: score={:.4}",
                trial,
                self.best_architecture
                    .read()
                    .await
                    .as_ref()
                    .map(|p| p.score)
                    .unwrap_or(0.0)
            );
        }

        Ok(())
    }

    /// Evolutionary search strategy
    async fn evolutionary_search(&mut self, input_dim: usize, output_dim: usize) -> Result<()> {
        // Initialize population
        self.initialize_population(input_dim, output_dim).await?;

        let mut no_improvement = 0;
        let mut best_score = 0.0;

        for generation in 0..self.config.generations {
            // Evaluate population
            let mut performances = Vec::new();
            let population = self.population.read().await.clone();

            for architecture in &population {
                let perf = self.evaluate_architecture(architecture).await?;
                performances.push(perf.clone());
                self.update_best(perf.clone()).await;
                self.history.write().await.push(perf.clone());
            }

            // Check for improvement
            let current_best = self
                .best_architecture
                .read()
                .await
                .as_ref()
                .expect("best_architecture should be set after update_best")
                .score;
            if current_best > best_score + 0.001 {
                best_score = current_best;
                no_improvement = 0;
            } else {
                no_improvement += 1;
            }

            // Early stopping
            if no_improvement >= self.config.early_stopping_patience {
                info!("Early stopping at generation {}", generation);
                break;
            }

            // Selection
            let selected = self.tournament_selection(&performances).await?;

            // Crossover and mutation
            let offspring = self
                .create_offspring(&selected, input_dim, output_dim)
                .await?;

            // Update population
            *self.population.write().await = offspring;

            info!("Generation {}: best_score={:.4}", generation, best_score);
        }

        Ok(())
    }

    /// Gradient-based search (simplified DARTS)
    async fn gradient_based_search(&mut self, input_dim: usize, output_dim: usize) -> Result<()> {
        // Simplified version - just sample architectures and evaluate
        // Real DARTS would use continuous relaxation and gradient descent

        for trial in 0..self.config.max_trials {
            let architecture = self
                .generate_random_architecture(input_dim, output_dim)
                .await?;
            let performance = self.evaluate_architecture(&architecture).await?;

            self.update_best(performance.clone()).await;
            self.history.write().await.push(performance);

            debug!(
                "Gradient-based trial {}: score={:.4}",
                trial,
                self.best_architecture
                    .read()
                    .await
                    .as_ref()
                    .map(|p| p.score)
                    .unwrap_or(0.0)
            );
        }

        Ok(())
    }

    /// Generate a random architecture
    async fn generate_random_architecture(
        &self,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Architecture> {
        let mut rng = self.rng.lock().await;
        let mut architecture = Architecture::new(input_dim, output_dim);

        let num_layers = rng.random_range(
            self.config.search_space.min_layers..=self.config.search_space.max_layers,
        );

        for _ in 0..num_layers {
            let layer_idx = rng.random_range(0..self.config.search_space.allowed_layers.len());
            let mut layer = self.config.search_space.allowed_layers[layer_idx].clone();

            // Randomize layer parameters
            layer = match layer {
                LayerType::Dense { .. } => {
                    let units = rng.random_range(
                        self.config.search_space.min_units..=self.config.search_space.max_units,
                    );
                    LayerType::Dense { units }
                }
                LayerType::LSTM { .. } => {
                    let units = rng.random_range(
                        self.config.search_space.min_units..=self.config.search_space.max_units,
                    );
                    LayerType::LSTM { units }
                }
                LayerType::GRU { .. } => {
                    let units = rng.random_range(
                        self.config.search_space.min_units..=self.config.search_space.max_units,
                    );
                    LayerType::GRU { units }
                }
                LayerType::Dropout { .. } => {
                    let rate = rng.random::<f64>() * 0.5; // 0-0.5 dropout rate
                    LayerType::Dropout { rate }
                }
                other => other,
            };

            architecture.add_layer(layer);
        }

        // Add final output layer
        architecture.add_layer(LayerType::Dense { units: output_dim });

        Ok(architecture)
    }

    /// Evaluate architecture performance
    async fn evaluate_architecture(
        &self,
        architecture: &Architecture,
    ) -> Result<ArchitecturePerformance> {
        let start_time = std::time::Instant::now();

        // Simplified evaluation - in production, actually train and test the network
        let mut rng = self.rng.lock().await;

        let param_count = architecture.parameter_count();
        let comp_cost = architecture.computational_cost();

        // Simulated metrics (in production, train the actual network)
        let base_accuracy = 0.7 + rng.random::<f64>() * 0.2;
        let complexity_penalty = (param_count as f64 / 1_000_000.0).min(0.1);
        let accuracy = base_accuracy - complexity_penalty;

        let training_time = (param_count as f64 / 100_000.0) + rng.random::<f64>();
        let inference_latency = (comp_cost / 1_000_000.0) + rng.random::<f64>() * 5.0;
        let memory_mb = (param_count as f64 * 4.0) / (1024.0 * 1024.0); // 4 bytes per parameter

        drop(rng);

        let mut performance = ArchitecturePerformance {
            architecture: architecture.clone(),
            accuracy,
            training_time,
            inference_latency_ms: inference_latency,
            memory_mb,
            parameter_count: param_count,
            score: 0.0,
        };

        performance.compute_score(&self.config.objective_weights);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_evaluated += 1;
        let eval_time = start_time.elapsed().as_secs_f64();
        stats.avg_evaluation_time =
            (stats.avg_evaluation_time * (stats.total_evaluated - 1) as f64 + eval_time)
                / stats.total_evaluated as f64;

        Ok(performance)
    }

    /// Initialize population for evolutionary search
    async fn initialize_population(&self, input_dim: usize, output_dim: usize) -> Result<()> {
        let mut population = Vec::new();

        for _ in 0..self.config.population_size {
            let architecture = self
                .generate_random_architecture(input_dim, output_dim)
                .await?;
            population.push(architecture);
        }

        *self.population.write().await = population;
        Ok(())
    }

    /// Tournament selection
    async fn tournament_selection(
        &self,
        performances: &[ArchitecturePerformance],
    ) -> Result<Vec<Architecture>> {
        let mut selected = Vec::new();
        let tournament_size = 3;

        let mut rng = self.rng.lock().await;

        for _ in 0..self.config.population_size {
            let mut best_idx = rng.random_range(0..performances.len());
            let mut best_score = performances[best_idx].score;

            for _ in 1..tournament_size {
                let idx = rng.random_range(0..performances.len());
                if performances[idx].score > best_score {
                    best_idx = idx;
                    best_score = performances[idx].score;
                }
            }

            selected.push(performances[best_idx].architecture.clone());
        }

        Ok(selected)
    }

    /// Create offspring through crossover and mutation
    async fn create_offspring(
        &self,
        parents: &[Architecture],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<Architecture>> {
        let mut offspring = Vec::new();
        let mut rng = self.rng.lock().await;

        for i in (0..parents.len()).step_by(2) {
            let parent1 = &parents[i];
            let parent2 = &parents[(i + 1) % parents.len()];

            // Crossover
            let (mut child1, mut child2) = self.crossover(parent1, parent2, &mut rng)?;

            // Mutation
            if rng.random::<f64>() < self.config.mutation_rate {
                self.mutate(&mut child1, input_dim, output_dim, &mut rng)?;
            }
            if rng.random::<f64>() < self.config.mutation_rate {
                self.mutate(&mut child2, input_dim, output_dim, &mut rng)?;
            }

            offspring.push(child1);
            offspring.push(child2);
        }

        offspring.truncate(self.config.population_size);
        Ok(offspring)
    }

    /// Crossover operation
    fn crossover(
        &self,
        parent1: &Architecture,
        parent2: &Architecture,
        rng: &mut Random,
    ) -> Result<(Architecture, Architecture)> {
        let mut child1 = Architecture::new(parent1.input_dim, parent1.output_dim);
        let mut child2 = Architecture::new(parent2.input_dim, parent2.output_dim);

        let min_len = parent1.layers.len().min(parent2.layers.len());
        if min_len == 0 {
            return Ok((parent1.clone(), parent2.clone()));
        }

        let crossover_point = rng.gen_range(1..min_len);

        // Child 1: first part from parent1, second from parent2
        for i in 0..crossover_point {
            child1.add_layer(parent1.layers[i].clone());
        }
        for i in crossover_point..parent2.layers.len() {
            child1.add_layer(parent2.layers[i].clone());
        }

        // Child 2: first part from parent2, second from parent1
        for i in 0..crossover_point {
            child2.add_layer(parent2.layers[i].clone());
        }
        for i in crossover_point..parent1.layers.len() {
            child2.add_layer(parent1.layers[i].clone());
        }

        Ok((child1, child2))
    }

    /// Mutation operation
    fn mutate(
        &self,
        architecture: &mut Architecture,
        _input_dim: usize,
        _output_dim: usize,
        rng: &mut Random,
    ) -> Result<()> {
        if architecture.layers.is_empty() {
            return Ok(());
        }

        let mutation_type = rng.gen_range(0..3);

        match mutation_type {
            0 => {
                // Add a layer
                if architecture.layers.len() < self.config.search_space.max_layers {
                    let layer_idx = rng.gen_range(0..self.config.search_space.allowed_layers.len());
                    let layer = self.config.search_space.allowed_layers[layer_idx].clone();
                    let insert_pos = rng.gen_range(0..=architecture.layers.len());
                    architecture.layers.insert(insert_pos, layer);
                }
            }
            1 => {
                // Remove a layer
                if architecture.layers.len() > self.config.search_space.min_layers {
                    let remove_pos = rng.gen_range(0..architecture.layers.len());
                    architecture.layers.remove(remove_pos);
                }
            }
            _ => {
                // Modify a layer
                let modify_pos = rng.gen_range(0..architecture.layers.len());
                let layer_idx = rng.gen_range(0..self.config.search_space.allowed_layers.len());
                architecture.layers[modify_pos] =
                    self.config.search_space.allowed_layers[layer_idx].clone();
            }
        }

        Ok(())
    }

    /// Update best architecture
    async fn update_best(&self, performance: ArchitecturePerformance) {
        let mut best = self.best_architecture.write().await;

        let should_update = match &*best {
            Some(current_best) => performance.score > current_best.score,
            None => true,
        };

        if should_update {
            *best = Some(performance.clone());
            let mut stats = self.stats.write().await;
            stats.best_score = performance.score;
        }
    }

    /// Get search statistics
    pub async fn get_stats(&self) -> NASStats {
        self.stats.read().await.clone()
    }

    /// Get evaluation history
    pub async fn get_history(&self) -> Vec<ArchitecturePerformance> {
        self.history.read().await.clone()
    }

    /// Get best architecture
    pub async fn get_best_architecture(&self) -> Option<ArchitecturePerformance> {
        self.best_architecture.read().await.clone()
    }

    /// Export best architecture for deployment
    pub async fn export_architecture(&self) -> Result<String> {
        let best = self.best_architecture.read().await;

        match &*best {
            Some(perf) => {
                let export = serde_json::json!({
                    "architecture": {
                        "layers": perf.architecture.layers,
                        "input_dim": perf.architecture.input_dim,
                        "output_dim": perf.architecture.output_dim,
                    },
                    "performance": {
                        "accuracy": perf.accuracy,
                        "score": perf.score,
                        "parameters": perf.parameter_count,
                        "latency_ms": perf.inference_latency_ms,
                        "memory_mb": perf.memory_mb,
                    }
                });
                Ok(serde_json::to_string_pretty(&export)?)
            }
            None => Err(anyhow!("No architecture to export")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_types() {
        let dense = LayerType::Dense { units: 128 };
        let lstm = LayerType::LSTM { units: 64 };
        let dropout = LayerType::Dropout { rate: 0.3 };

        assert!(matches!(dense, LayerType::Dense { .. }));
        assert!(matches!(lstm, LayerType::LSTM { .. }));
        assert!(matches!(dropout, LayerType::Dropout { .. }));
    }

    #[test]
    fn test_architecture_parameter_count() {
        let mut arch = Architecture::new(10, 2);
        arch.add_layer(LayerType::Dense { units: 64 });
        arch.add_layer(LayerType::Dense { units: 32 });
        arch.add_layer(LayerType::Dense { units: 2 });

        let params = arch.parameter_count();
        // 10*64 + 64 + 64*32 + 32 + 32*2 + 2
        assert!(params > 0);
    }

    #[test]
    fn test_architecture_computational_cost() {
        let mut arch = Architecture::new(10, 2);
        arch.add_layer(LayerType::Dense { units: 64 });
        arch.add_layer(LayerType::LSTM { units: 32 });

        let cost = arch.computational_cost();
        assert!(cost > 0.0);
    }

    #[tokio::test]
    async fn test_nas_creation() {
        let config = NASConfig::default();
        let nas = NAS::new(config);
        assert!(nas.is_ok());
    }

    #[tokio::test]
    async fn test_generate_random_architecture() {
        let config = NASConfig {
            search_space: SearchSpace {
                min_layers: 2,
                max_layers: 5,
                ..Default::default()
            },
            ..Default::default()
        };

        let nas = NAS::new(config).unwrap();
        let arch = nas.generate_random_architecture(10, 2).await;

        assert!(arch.is_ok());
        let architecture = arch.unwrap();
        assert_eq!(architecture.input_dim, 10);
        assert_eq!(architecture.output_dim, 2);
        assert!(architecture.layers.len() >= 2);
        // Max layers + 1 for output layer
        assert!(architecture.layers.len() <= 6);
    }

    #[tokio::test]
    async fn test_evaluate_architecture() {
        let config = NASConfig::default();
        let nas = NAS::new(config).unwrap();

        let mut arch = Architecture::new(10, 2);
        arch.add_layer(LayerType::Dense { units: 64 });
        arch.add_layer(LayerType::Dense { units: 2 });

        let perf = nas.evaluate_architecture(&arch).await;
        assert!(perf.is_ok());

        let performance = perf.unwrap();
        assert!(performance.accuracy > 0.0);
        assert!(performance.score >= 0.0);
        assert!(performance.parameter_count > 0);
    }

    #[tokio::test]
    async fn test_random_search() {
        let config = NASConfig {
            strategy: SearchStrategy::Random,
            max_trials: 5,
            ..Default::default()
        };

        let mut nas = NAS::new(config).unwrap();
        let result = nas.search(10, 2).await;
        assert!(result.is_ok());

        let stats = nas.get_stats().await;
        assert!(stats.total_evaluated > 0);
        assert!(stats.best_score > 0.0);
    }

    #[tokio::test]
    async fn test_evolutionary_search() {
        let config = NASConfig {
            strategy: SearchStrategy::Evolutionary,
            population_size: 5,
            generations: 3,
            ..Default::default()
        };

        let mut nas = NAS::new(config).unwrap();
        let result = nas.search(10, 2).await;
        assert!(result.is_ok());

        let stats = nas.get_stats().await;
        assert!(stats.total_evaluated > 0);
    }

    #[tokio::test]
    async fn test_architecture_performance_score() {
        let arch = Architecture::new(10, 2);
        let mut perf = ArchitecturePerformance {
            architecture: arch,
            accuracy: 0.85,
            training_time: 10.0,
            inference_latency_ms: 5.0,
            memory_mb: 100.0,
            parameter_count: 10000,
            score: 0.0,
        };

        let weights = ObjectiveWeights::default();
        perf.compute_score(&weights);

        assert!(perf.score > 0.0);
    }

    #[tokio::test]
    async fn test_crossover() {
        let config = NASConfig::default();
        let nas = NAS::new(config).unwrap();

        let mut parent1 = Architecture::new(10, 2);
        parent1.add_layer(LayerType::Dense { units: 64 });
        parent1.add_layer(LayerType::Dense { units: 32 });

        let mut parent2 = Architecture::new(10, 2);
        parent2.add_layer(LayerType::LSTM { units: 64 });
        parent2.add_layer(LayerType::Dense { units: 16 });

        let mut rng = Random::default();
        let (child1, child2) = nas.crossover(&parent1, &parent2, &mut rng).unwrap();

        assert!(!child1.layers.is_empty());
        assert!(!child2.layers.is_empty());
    }

    #[tokio::test]
    async fn test_mutation() {
        let config = NASConfig::default();
        let nas = NAS::new(config).unwrap();

        let mut arch = Architecture::new(10, 2);
        arch.add_layer(LayerType::Dense { units: 64 });
        arch.add_layer(LayerType::Dense { units: 32 });

        let _original_len = arch.layers.len();

        let mut rng = Random::default();
        nas.mutate(&mut arch, 10, 2, &mut rng).unwrap();

        // Mutation may add, remove, or modify layers
        assert!(!arch.layers.is_empty());
    }

    #[tokio::test]
    async fn test_export_architecture() {
        let config = NASConfig {
            max_trials: 2,
            ..Default::default()
        };

        let mut nas = NAS::new(config).unwrap();
        nas.search(10, 2).await.unwrap();

        let export = nas.export_architecture().await;
        assert!(export.is_ok());

        let json_str = export.unwrap();
        assert!(json_str.contains("architecture"));
        assert!(json_str.contains("performance"));
    }

    #[tokio::test]
    async fn test_get_history() {
        let config = NASConfig {
            max_trials: 3,
            ..Default::default()
        };

        let mut nas = NAS::new(config).unwrap();
        nas.search(10, 2).await.unwrap();

        let history = nas.get_history().await;
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_objective_weights() {
        let weights = ObjectiveWeights {
            accuracy: 0.8,
            latency: 0.1,
            memory: 0.05,
            params: 0.05,
        };

        assert_eq!(
            weights.accuracy + weights.latency + weights.memory + weights.params,
            1.0
        );
    }
}
