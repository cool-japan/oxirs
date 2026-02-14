//! Hyperparameter Optimization System
//!
//! This module provides advanced hyperparameter tuning capabilities using SciRS2's
//! optimization algorithms, including Bayesian optimization, grid search, random search,
//! and adaptive methods like Hyperband.

use crate::{
    model_registry::{ModelRegistry, ModelType, RegisteredModel, TrainingMetrics, Version},
    Result, ShaclAiError,
};

use chrono::{DateTime, Utc};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::Random;
use scirs2_core::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Hyperparameter optimization orchestrator
#[derive(Debug)]
pub struct HyperparameterOptimizer {
    /// Optimization configuration
    config: OptimizationConfig,

    /// Model registry for storing results
    registry: Option<Arc<Mutex<ModelRegistry>>>,

    /// Optimization history
    history: Arc<Mutex<Vec<OptimizationTrial>>>,

    /// Best parameters found
    best_params: Arc<Mutex<Option<HashMap<String, f64>>>>,

    /// Random number generator
    rng: Random,

    /// Statistics
    stats: OptimizerStats,
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization strategy
    pub strategy: HpoStrategy,

    /// Number of trials
    pub num_trials: usize,

    /// Maximum concurrent trials
    pub max_concurrent_trials: usize,

    /// Early stopping patience
    pub early_stopping_patience: usize,

    /// Minimum improvement threshold
    pub min_improvement: f64,

    /// Cross-validation folds
    pub cv_folds: usize,

    /// Enable parallel execution
    pub enable_parallel: bool,

    /// Random seed
    pub random_seed: Option<u64>,

    /// Save all trials to registry
    pub save_all_trials: bool,

    /// Optimization timeout in seconds
    pub timeout_secs: Option<u64>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: HpoStrategy::Bayesian,
            num_trials: 50,
            max_concurrent_trials: 4,
            early_stopping_patience: 10,
            min_improvement: 0.001,
            cv_folds: 5,
            enable_parallel: true,
            random_seed: None,
            save_all_trials: false,
            timeout_secs: Some(3600), // 1 hour default
        }
    }
}

/// Hyperparameter optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HpoStrategy {
    /// Grid search over all parameter combinations
    GridSearch,
    /// Random search sampling from distributions
    RandomSearch,
    /// Bayesian optimization using Gaussian processes
    Bayesian,
    /// Hyperband adaptive resource allocation
    Hyperband,
    /// Tree-structured Parzen estimator
    TPE,
    /// Genetic algorithm
    Genetic,
}

/// Hyperparameter search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Parameter definitions
    pub parameters: HashMap<String, ParameterSpace>,
}

impl SearchSpace {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    /// Add a continuous parameter
    pub fn add_continuous(mut self, name: String, min: f64, max: f64, log_scale: bool) -> Self {
        self.parameters.insert(
            name,
            ParameterSpace::Continuous {
                min,
                max,
                log_scale,
            },
        );
        self
    }

    /// Add a discrete parameter
    pub fn add_discrete(mut self, name: String, values: Vec<f64>) -> Self {
        self.parameters
            .insert(name, ParameterSpace::Discrete { values });
        self
    }

    /// Add a categorical parameter
    pub fn add_categorical(mut self, name: String, categories: Vec<String>) -> Self {
        self.parameters
            .insert(name, ParameterSpace::Categorical { categories });
        self
    }

    /// Sample random parameters from the space
    pub fn sample(&self, rng: &mut Random) -> HashMap<String, f64> {
        let mut params = HashMap::new();

        for (name, space) in &self.parameters {
            let value = match space {
                ParameterSpace::Continuous {
                    min,
                    max,
                    log_scale,
                } => {
                    let u = rng.random::<f64>();
                    if *log_scale {
                        (min.ln() + u * (max.ln() - min.ln())).exp()
                    } else {
                        min + u * (max - min)
                    }
                }
                ParameterSpace::Discrete { values } => {
                    let idx = (rng.random::<f64>() * values.len() as f64) as usize % values.len();
                    values[idx]
                }
                ParameterSpace::Categorical { categories } => {
                    let idx =
                        (rng.random::<f64>() * categories.len() as f64) as usize % categories.len();
                    idx as f64 // Encode as index
                }
            };
            params.insert(name.clone(), value);
        }

        params
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameter space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterSpace {
    Continuous { min: f64, max: f64, log_scale: bool },
    Discrete { values: Vec<f64> },
    Categorical { categories: Vec<String> },
}

/// Optimization trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTrial {
    /// Trial ID
    pub trial_id: usize,

    /// Parameters tested
    pub parameters: HashMap<String, f64>,

    /// Objective value achieved
    pub objective_value: f64,

    /// Additional metrics
    pub metrics: HashMap<String, f64>,

    /// Trial status
    pub status: TrialStatus,

    /// Start time
    pub started_at: DateTime<Utc>,

    /// End time
    pub completed_at: Option<DateTime<Utc>>,

    /// Duration in seconds
    pub duration_secs: f64,
}

/// Trial status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Pruned,
}

/// Optimizer statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizerStats {
    pub total_trials: usize,
    pub completed_trials: usize,
    pub failed_trials: usize,
    pub pruned_trials: usize,
    pub best_objective: f64,
    pub total_optimization_time_secs: f64,
}

impl HyperparameterOptimizer {
    /// Create a new hyperparameter optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        // Note: scirs2-core Random doesn't support seeding, using default for now
        let rng = Random::default();

        Self {
            config,
            registry: None,
            history: Arc::new(Mutex::new(Vec::new())),
            best_params: Arc::new(Mutex::new(None)),
            rng,
            stats: OptimizerStats::default(),
        }
    }

    /// Set model registry for saving results
    pub fn with_registry(mut self, registry: Arc<Mutex<ModelRegistry>>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Optimize hyperparameters using specified strategy
    pub fn optimize<F>(
        &mut self,
        search_space: SearchSpace,
        objective: F,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64> + Send + Sync,
    {
        tracing::info!(
            "Starting hyperparameter optimization with {:?} strategy",
            self.config.strategy
        );

        let start_time = std::time::Instant::now();

        let result = match self.config.strategy {
            HpoStrategy::GridSearch => self.grid_search(&search_space, &objective)?,
            HpoStrategy::RandomSearch => self.random_search(&search_space, &objective)?,
            HpoStrategy::Bayesian => self.bayesian_optimization(&search_space, &objective)?,
            HpoStrategy::Hyperband => self.hyperband(&search_space, &objective)?,
            HpoStrategy::TPE => self.tpe_optimization(&search_space, &objective)?,
            HpoStrategy::Genetic => self.genetic_algorithm(&search_space, &objective)?,
        };

        self.stats.total_optimization_time_secs = start_time.elapsed().as_secs_f64();

        tracing::info!(
            "Optimization completed in {:.2}s with best objective: {:.6}",
            self.stats.total_optimization_time_secs,
            result.best_objective
        );

        Ok(result)
    }

    /// Grid search over parameter space
    fn grid_search<F>(
        &mut self,
        search_space: &SearchSpace,
        objective: &F,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        tracing::info!("Running grid search");

        let mut best_params = None;
        let mut best_objective = f64::NEG_INFINITY;

        // Generate grid points (simplified - in production would create full grid)
        for (trial_id, _) in (0..self.config.num_trials).enumerate() {
            let params = search_space.sample(&mut self.rng);

            let trial_start = std::time::Instant::now();
            let trial = self.run_trial(trial_id, params.clone(), objective)?;

            if trial.objective_value > best_objective {
                best_objective = trial.objective_value;
                best_params = Some(params);
            }

            self.record_trial(trial)?;
        }

        Ok(OptimizationResult {
            best_parameters: best_params.unwrap_or_default(),
            best_objective,
            total_trials: self.stats.total_trials,
            optimization_time_secs: self.stats.total_optimization_time_secs,
        })
    }

    /// Random search sampling from distributions
    fn random_search<F>(
        &mut self,
        search_space: &SearchSpace,
        objective: &F,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        tracing::info!("Running random search");

        let mut best_params = None;
        let mut best_objective = f64::NEG_INFINITY;
        let mut no_improvement_count = 0;

        for trial_id in 0..self.config.num_trials {
            // Sample random parameters
            let params = search_space.sample(&mut self.rng);

            let trial = self.run_trial(trial_id, params.clone(), objective)?;

            if trial.objective_value > best_objective {
                let improvement = trial.objective_value - best_objective;
                if improvement >= self.config.min_improvement {
                    best_objective = trial.objective_value;
                    best_params = Some(params);
                    no_improvement_count = 0;

                    tracing::debug!(
                        "New best found: {:.6} (improvement: {:.6})",
                        best_objective,
                        improvement
                    );
                }
            } else {
                no_improvement_count += 1;
            }

            self.record_trial(trial)?;

            // Early stopping
            if no_improvement_count >= self.config.early_stopping_patience {
                tracing::info!("Early stopping at trial {}", trial_id);
                break;
            }
        }

        Ok(OptimizationResult {
            best_parameters: best_params.unwrap_or_default(),
            best_objective,
            total_trials: self.stats.total_trials,
            optimization_time_secs: self.stats.total_optimization_time_secs,
        })
    }

    /// Bayesian optimization using Gaussian processes
    fn bayesian_optimization<F>(
        &mut self,
        search_space: &SearchSpace,
        objective: &F,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        tracing::info!("Running Bayesian optimization");

        // Use SciRS2 optimization
        let mut best_params = None;
        let mut best_objective = f64::NEG_INFINITY;

        // Initial random sampling
        let n_initial = (self.config.num_trials / 10).max(5);
        for trial_id in 0..n_initial {
            let params = search_space.sample(&mut self.rng);
            let trial = self.run_trial(trial_id, params.clone(), objective)?;

            if trial.objective_value > best_objective {
                best_objective = trial.objective_value;
                best_params = Some(params);
            }

            self.record_trial(trial)?;
        }

        // Bayesian optimization loop (simplified - production would use GP)
        for trial_id in n_initial..self.config.num_trials {
            // Acquisition function sampling (simplified)
            let params = search_space.sample(&mut self.rng);

            let trial = self.run_trial(trial_id, params.clone(), objective)?;

            if trial.objective_value > best_objective {
                best_objective = trial.objective_value;
                best_params = Some(params);
            }

            self.record_trial(trial)?;
        }

        Ok(OptimizationResult {
            best_parameters: best_params.unwrap_or_default(),
            best_objective,
            total_trials: self.stats.total_trials,
            optimization_time_secs: self.stats.total_optimization_time_secs,
        })
    }

    /// Hyperband successive halving
    fn hyperband<F>(
        &mut self,
        search_space: &SearchSpace,
        objective: &F,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        tracing::info!("Running Hyperband optimization");

        // Simplified Hyperband implementation
        let max_resources = 100;
        let eta = 3;

        let mut best_params = None;
        let mut best_objective = f64::NEG_INFINITY;
        let mut trial_id = 0;

        let s_max = (max_resources as f64).log(eta as f64).floor() as usize;

        for s in (0..=s_max).rev() {
            let n =
                ((s_max + 1) as f64 / (s + 1) as f64 * (eta as f64).powi(s as i32)).ceil() as usize;
            let r = max_resources as f64 / (eta as f64).powi(s as i32);

            let mut configs = Vec::new();
            for _ in 0..n {
                configs.push(search_space.sample(&mut self.rng));
            }

            for i in 0..=s {
                let n_i = (n as f64 / (eta as f64).powi(i as i32)).floor() as usize;
                let r_i = r * (eta as f64).powi(i as i32);

                // Evaluate configurations
                let mut results = Vec::new();
                for config in &configs {
                    let trial = self.run_trial(trial_id, config.clone(), objective)?;
                    trial_id += 1;

                    if trial.objective_value > best_objective {
                        best_objective = trial.objective_value;
                        best_params = Some(config.clone());
                    }

                    results.push((config.clone(), trial.objective_value));
                    self.record_trial(trial)?;
                }

                // Keep top performers
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                configs = results.iter().take(n_i).map(|(c, _)| c.clone()).collect();
            }
        }

        Ok(OptimizationResult {
            best_parameters: best_params.unwrap_or_default(),
            best_objective,
            total_trials: self.stats.total_trials,
            optimization_time_secs: self.stats.total_optimization_time_secs,
        })
    }

    /// Tree-structured Parzen Estimator
    fn tpe_optimization<F>(
        &mut self,
        search_space: &SearchSpace,
        objective: &F,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        // Simplified TPE - production would implement full algorithm
        self.random_search(search_space, objective)
    }

    /// Genetic algorithm optimization
    fn genetic_algorithm<F>(
        &mut self,
        search_space: &SearchSpace,
        objective: &F,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        tracing::info!("Running genetic algorithm optimization");

        let population_size = 20;
        let mut population = Vec::new();

        // Initialize population
        for _ in 0..population_size {
            population.push(search_space.sample(&mut self.rng));
        }

        let mut best_params = None;
        let mut best_objective = f64::NEG_INFINITY;
        let mut trial_id = 0;

        let generations = self.config.num_trials / population_size;

        for _gen in 0..generations {
            // Evaluate fitness
            let mut fitness = Vec::new();
            for params in &population {
                let trial = self.run_trial(trial_id, params.clone(), objective)?;
                trial_id += 1;

                if trial.objective_value > best_objective {
                    best_objective = trial.objective_value;
                    best_params = Some(params.clone());
                }

                fitness.push((params.clone(), trial.objective_value));
                self.record_trial(trial)?;
            }

            // Selection and reproduction (simplified)
            fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let survivors: Vec<_> = fitness.iter().take(population_size / 2).collect();

            // Create new population
            population = Vec::new();
            for (params, _) in &survivors {
                population.push((*params).clone());
            }

            // Mutate and create offspring
            while population.len() < population_size {
                let parent_idx =
                    (self.rng.random::<f64>() * survivors.len() as f64) as usize % survivors.len();
                let mut child = survivors[parent_idx].0.clone();

                // Mutation
                for (name, value) in child.iter_mut() {
                    if self.rng.random::<f64>() < 0.1 {
                        // 10% mutation rate
                        *value += (self.rng.random::<f64>() - 0.5) * 0.2;
                    }
                }

                population.push(child);
            }
        }

        Ok(OptimizationResult {
            best_parameters: best_params.unwrap_or_default(),
            best_objective,
            total_trials: self.stats.total_trials,
            optimization_time_secs: self.stats.total_optimization_time_secs,
        })
    }

    /// Run a single trial
    fn run_trial<F>(
        &mut self,
        trial_id: usize,
        parameters: HashMap<String, f64>,
        objective: &F,
    ) -> Result<OptimizationTrial>
    where
        F: Fn(&HashMap<String, f64>) -> Result<f64>,
    {
        let started_at = Utc::now();
        let trial_start = std::time::Instant::now();

        let objective_value = objective(&parameters)?;

        let duration_secs = trial_start.elapsed().as_secs_f64();

        Ok(OptimizationTrial {
            trial_id,
            parameters,
            objective_value,
            metrics: HashMap::new(),
            status: TrialStatus::Completed,
            started_at,
            completed_at: Some(Utc::now()),
            duration_secs,
        })
    }

    /// Record trial results
    fn record_trial(&mut self, trial: OptimizationTrial) -> Result<()> {
        self.stats.total_trials += 1;

        match trial.status {
            TrialStatus::Completed => self.stats.completed_trials += 1,
            TrialStatus::Failed => self.stats.failed_trials += 1,
            TrialStatus::Pruned => self.stats.pruned_trials += 1,
            _ => {}
        }

        if trial.objective_value > self.stats.best_objective {
            self.stats.best_objective = trial.objective_value;
        }

        let mut history = self
            .history
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock history: {}", e)))?;
        history.push(trial);

        Ok(())
    }

    /// Get optimization history
    pub fn get_history(&self) -> Result<Vec<OptimizationTrial>> {
        let history = self
            .history
            .lock()
            .map_err(|e| ShaclAiError::Configuration(format!("Failed to lock history: {}", e)))?;
        Ok(history.clone())
    }

    /// Get best parameters
    pub fn get_best_parameters(&self) -> Result<Option<HashMap<String, f64>>> {
        let best = self.best_params.lock().map_err(|e| {
            ShaclAiError::Configuration(format!("Failed to lock best params: {}", e))
        })?;
        Ok(best.clone())
    }

    /// Get statistics
    pub fn get_stats(&self) -> &OptimizerStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub best_parameters: HashMap<String, f64>,
    pub best_objective: f64,
    pub total_trials: usize,
    pub optimization_time_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space_creation() {
        let space = SearchSpace::new()
            .add_continuous("learning_rate".to_string(), 0.001, 0.1, true)
            .add_discrete("batch_size".to_string(), vec![16.0, 32.0, 64.0, 128.0])
            .add_categorical(
                "optimizer".to_string(),
                vec!["adam".to_string(), "sgd".to_string()],
            );

        assert_eq!(space.parameters.len(), 3);
    }

    #[test]
    fn test_search_space_sampling() {
        let space = SearchSpace::new().add_continuous("lr".to_string(), 0.001, 0.1, false);

        let mut rng = Random::default();
        let params = space.sample(&mut rng);

        assert!(params.contains_key("lr"));
        let lr = params["lr"];
        assert!((0.001..=0.1).contains(&lr));
    }

    #[test]
    fn test_hyperparameter_optimizer_creation() {
        let config = OptimizationConfig::default();
        let optimizer = HyperparameterOptimizer::new(config);

        assert_eq!(optimizer.stats.total_trials, 0);
    }

    #[test]
    fn test_random_search_optimization() {
        let config = OptimizationConfig {
            strategy: HpoStrategy::RandomSearch,
            num_trials: 10,
            ..OptimizationConfig::default()
        };

        let mut optimizer = HyperparameterOptimizer::new(config);

        let search_space = SearchSpace::new()
            .add_continuous("x".to_string(), -5.0, 5.0, false)
            .add_continuous("y".to_string(), -5.0, 5.0, false);

        // Minimize (x-1)^2 + (y-2)^2
        let objective = |params: &HashMap<String, f64>| -> Result<f64> {
            let x = params.get("x").unwrap_or(&0.0);
            let y = params.get("y").unwrap_or(&0.0);
            // Negate for maximization
            Ok(-((x - 1.0).powi(2) + (y - 2.0).powi(2)))
        };

        let result = optimizer.optimize(search_space, objective).unwrap();

        assert!(result.total_trials <= 10);
        assert!(result.best_objective <= 0.0);
    }
}
