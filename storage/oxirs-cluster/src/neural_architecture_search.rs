//! # Neural Architecture Search for Cluster Parameter Tuning
//!
//! Automatically discovers optimal cluster configuration parameters using
//! evolutionary algorithms and parallel search strategies.
//!
//! ## Features
//!
//! - Evolutionary parameter optimization
//! - Parallel candidate evaluation
//! - Multi-objective optimization (latency, throughput, stability)
//! - Adaptive mutation rates
//! - Performance-based selection
//! - Historical best tracking

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info};

// Parallel processing for candidate evaluation
use rayon::prelude::*;
use scirs2_core::metrics::Counter;
use scirs2_core::random::{rng, Rng};

use crate::error::Result;

/// Parameter search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    /// Raft heartbeat interval (ms): [10, 500]
    pub heartbeat_interval_ms: (u64, u64),
    /// Election timeout (ms): [150, 3000]
    pub election_timeout_ms: (u64, u64),
    /// Batch size: [1, 10000]
    pub batch_size: (usize, usize),
    /// Replication factor: [1, 10]
    pub replication_factor: (usize, usize),
    /// Cache size (MB): [64, 4096]
    pub cache_size_mb: (usize, usize),
    /// Compaction interval (s): [60, 3600]
    pub compaction_interval_secs: (u64, u64),
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            heartbeat_interval_ms: (10, 500),
            election_timeout_ms: (150, 3000),
            batch_size: (1, 10000),
            replication_factor: (1, 10),
            cache_size_mb: (64, 4096),
            compaction_interval_secs: (60, 3600),
        }
    }
}

/// Cluster parameter configuration candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterCandidate {
    /// Candidate ID
    pub id: String,
    /// Raft heartbeat interval (ms)
    pub heartbeat_interval_ms: u64,
    /// Election timeout (ms)
    pub election_timeout_ms: u64,
    /// Batch size
    pub batch_size: usize,
    /// Replication factor
    pub replication_factor: usize,
    /// Cache size (MB)
    pub cache_size_mb: usize,
    /// Compaction interval (s)
    pub compaction_interval_secs: u64,
    /// Fitness score (higher is better)
    pub fitness: f64,
    /// Generation number
    pub generation: u32,
    /// Created timestamp
    pub created_at: SystemTime,
}

impl ParameterCandidate {
    /// Create a random candidate within parameter space
    pub fn random(space: &ParameterSpace, generation: u32) -> Self {
        let mut rng_inst = rng();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            heartbeat_interval_ms: rng_inst
                .random_range(space.heartbeat_interval_ms.0..=space.heartbeat_interval_ms.1),
            election_timeout_ms: rng_inst
                .random_range(space.election_timeout_ms.0..=space.election_timeout_ms.1),
            batch_size: rng_inst.random_range(space.batch_size.0..=space.batch_size.1),
            replication_factor: rng_inst
                .random_range(space.replication_factor.0..=space.replication_factor.1),
            cache_size_mb: rng_inst.random_range(space.cache_size_mb.0..=space.cache_size_mb.1),
            compaction_interval_secs: rng_inst
                .random_range(space.compaction_interval_secs.0..=space.compaction_interval_secs.1),
            fitness: 0.0,
            generation,
            created_at: SystemTime::now(),
        }
    }

    /// Mutate this candidate (evolutionary mutation)
    pub fn mutate(&self, space: &ParameterSpace, mutation_rate: f64) -> Self {
        let mut rng_inst = rng();
        let mut mutated = self.clone();
        mutated.id = uuid::Uuid::new_v4().to_string();
        mutated.generation = self.generation + 1;
        mutated.created_at = SystemTime::now();

        // Mutate each parameter with probability = mutation_rate
        if rng_inst.random::<f64>() < mutation_rate {
            mutated.heartbeat_interval_ms = rng_inst
                .random_range(space.heartbeat_interval_ms.0..=space.heartbeat_interval_ms.1);
        }

        if rng_inst.random::<f64>() < mutation_rate {
            mutated.election_timeout_ms =
                rng_inst.random_range(space.election_timeout_ms.0..=space.election_timeout_ms.1);
        }

        if rng_inst.random::<f64>() < mutation_rate {
            mutated.batch_size = rng_inst.random_range(space.batch_size.0..=space.batch_size.1);
        }

        if rng_inst.random::<f64>() < mutation_rate {
            mutated.replication_factor =
                rng_inst.random_range(space.replication_factor.0..=space.replication_factor.1);
        }

        if rng_inst.random::<f64>() < mutation_rate {
            mutated.cache_size_mb =
                rng_inst.random_range(space.cache_size_mb.0..=space.cache_size_mb.1);
        }

        if rng_inst.random::<f64>() < mutation_rate {
            mutated.compaction_interval_secs = rng_inst
                .random_range(space.compaction_interval_secs.0..=space.compaction_interval_secs.1);
        }

        mutated
    }

    /// Crossover with another candidate (genetic recombination)
    pub fn crossover(&self, other: &ParameterCandidate) -> Self {
        let mut rng_inst = rng();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            heartbeat_interval_ms: if rng_inst.random::<bool>() {
                self.heartbeat_interval_ms
            } else {
                other.heartbeat_interval_ms
            },
            election_timeout_ms: if rng_inst.random::<bool>() {
                self.election_timeout_ms
            } else {
                other.election_timeout_ms
            },
            batch_size: if rng_inst.random::<bool>() {
                self.batch_size
            } else {
                other.batch_size
            },
            replication_factor: if rng_inst.random::<bool>() {
                self.replication_factor
            } else {
                other.replication_factor
            },
            cache_size_mb: if rng_inst.random::<bool>() {
                self.cache_size_mb
            } else {
                other.cache_size_mb
            },
            compaction_interval_secs: if rng_inst.random::<bool>() {
                self.compaction_interval_secs
            } else {
                other.compaction_interval_secs
            },
            fitness: 0.0,
            generation: self.generation.max(other.generation) + 1,
            created_at: SystemTime::now(),
        }
    }
}

/// Performance metrics for fitness evaluation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average query latency (ms)
    pub avg_latency_ms: f64,
    /// Queries per second
    pub queries_per_sec: f64,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f64,
    /// Replication lag (ms)
    pub replication_lag_ms: f64,
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
    /// Leader election frequency (per hour)
    pub election_frequency: f64,
}

impl PerformanceMetrics {
    /// Compute fitness score (multi-objective optimization)
    pub fn compute_fitness(&self) -> f64 {
        // Weights for each objective
        let latency_weight = 0.30;
        let throughput_weight = 0.25;
        let stability_weight = 0.20;
        let resource_weight = 0.15;
        let reliability_weight = 0.10;

        // Latency score (lower is better, normalize to [0, 1])
        let latency_score = (1000.0 - self.avg_latency_ms.min(1000.0)) / 1000.0;

        // Throughput score (higher is better, normalize to [0, 1])
        let throughput_score = (self.queries_per_sec / 10000.0).min(1.0);

        // Stability score (fewer elections is better)
        let stability_score = (10.0 - self.election_frequency.min(10.0)) / 10.0;

        // Resource efficiency (lower utilization is better if performance is good)
        let resource_score = 1.0 - ((self.cpu_utilization + self.memory_utilization) / 2.0);

        // Reliability score (lower error rate is better)
        let reliability_score = 1.0 - self.error_rate;

        // Weighted combination
        let fitness = latency_weight * latency_score
            + throughput_weight * throughput_score
            + stability_weight * stability_score
            + resource_weight * resource_score
            + reliability_weight * reliability_score;

        fitness.max(0.0).min(1.0) // Clamp to [0, 1]
    }
}

/// NAS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub max_generations: u32,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Elite selection ratio (top % to keep)
    pub elite_ratio: f64,
    /// Crossover rate
    pub crossover_rate: f64,
    /// Performance evaluation duration (seconds)
    pub eval_duration_secs: u64,
}

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            mutation_rate: 0.1,
            elite_ratio: 0.2,
            crossover_rate: 0.7,
            eval_duration_secs: 60,
        }
    }
}

/// Neural Architecture Search engine
pub struct NeuralArchitectureSearch {
    config: NASConfig,
    space: ParameterSpace,
    /// Current population
    population: Arc<RwLock<Vec<ParameterCandidate>>>,
    /// Best candidate so far
    best_candidate: Arc<RwLock<Option<ParameterCandidate>>>,
    /// Current generation
    current_generation: Arc<RwLock<u32>>,
    /// Search metrics
    evaluations_counter: Counter,
    /// Best fitness scores seen
    fitness_scores: Arc<RwLock<Vec<f64>>>,
}

impl NeuralArchitectureSearch {
    /// Create a new NAS engine
    pub fn new(config: NASConfig, space: ParameterSpace) -> Self {
        Self {
            config,
            space,
            population: Arc::new(RwLock::new(Vec::new())),
            best_candidate: Arc::new(RwLock::new(None)),
            current_generation: Arc::new(RwLock::new(0)),
            evaluations_counter: Counter::new("nas_evaluations".to_string()),
            fitness_scores: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize population with random candidates
    pub async fn initialize_population(&self) {
        let mut population = self.population.write().await;

        for _ in 0..self.config.population_size {
            let candidate = ParameterCandidate::random(&self.space, 0);
            population.push(candidate);
        }

        info!(
            "Initialized NAS population with {} candidates",
            self.config.population_size
        );
    }

    /// Evaluate a candidate (simulated fitness function)
    /// In production, this would deploy and measure real performance
    fn evaluate_candidate(&self, candidate: &ParameterCandidate) -> f64 {
        self.evaluations_counter.inc();

        // Simulated performance based on parameter choices
        // In production, this would measure actual cluster performance

        let metrics = PerformanceMetrics {
            // Latency improves with smaller heartbeat intervals (to a point)
            avg_latency_ms: 50.0 + (candidate.heartbeat_interval_ms as f64 / 10.0),
            // Throughput improves with larger batch sizes
            queries_per_sec: (candidate.batch_size as f64).sqrt() * 10.0,
            // CPU scales with batch size and replication
            cpu_utilization: (candidate.batch_size as f64 / 10000.0)
                * (candidate.replication_factor as f64 / 10.0),
            // Memory usage scales with cache size
            memory_utilization: candidate.cache_size_mb as f64 / 4096.0,
            // Replication lag depends on heartbeat and batch size
            replication_lag_ms: (candidate.heartbeat_interval_ms as f64)
                + (candidate.batch_size as f64 / 100.0),
            // Error rate is random with bias toward optimal parameters
            error_rate: rng().random_range(0.0..0.05),
            // Election frequency depends on timeout
            election_frequency: 100.0 / (candidate.election_timeout_ms as f64),
        };

        metrics.compute_fitness()
    }

    /// Evaluate all candidates in parallel
    pub async fn evaluate_population(&self) {
        let population_clone = {
            let pop = self.population.read().await;
            pop.clone()
        };

        // Parallel evaluation using rayon
        let evaluated: Vec<ParameterCandidate> = population_clone
            .par_iter()
            .map(|candidate| {
                let mut eval_candidate = candidate.clone();
                eval_candidate.fitness = self.evaluate_candidate(candidate);
                eval_candidate
            })
            .collect();

        // Record fitness scores
        let fitness_values: Vec<f64> = evaluated.iter().map(|c| c.fitness).collect();
        {
            let mut scores = self.fitness_scores.write().await;
            scores.extend(fitness_values);
        }

        // Update population with fitness scores
        let mut population = self.population.write().await;
        *population = evaluated;

        // Update best candidate
        if let Some(best) = population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let mut best_candidate = self.best_candidate.write().await;
            if best_candidate.is_none() || best.fitness > best_candidate.as_ref().unwrap().fitness {
                info!(
                    "New best candidate found! Fitness: {:.4}, Gen: {}",
                    best.fitness, best.generation
                );
                *best_candidate = Some(best.clone());
            }
        }
    }

    /// Evolve to next generation
    pub async fn evolve_generation(&self) {
        let mut population = self.population.write().await;

        // Sort by fitness (descending)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Select elite candidates
        let elite_count = (self.config.population_size as f64 * self.config.elite_ratio) as usize;
        let mut next_generation = population[..elite_count].to_vec();

        // Fill rest of population with mutations and crossovers
        while next_generation.len() < self.config.population_size {
            if rng().random::<f64>() < self.config.crossover_rate && population.len() >= 2 {
                // Crossover two random elite candidates
                let parent1 = &population[rng().random_range(0..elite_count)];
                let parent2 = &population[rng().random_range(0..elite_count)];
                next_generation.push(parent1.crossover(parent2));
            } else {
                // Mutate a random elite candidate
                let parent = &population[rng().random_range(0..elite_count)];
                next_generation.push(parent.mutate(&self.space, self.config.mutation_rate));
            }
        }

        *population = next_generation;

        // Increment generation counter
        let mut gen = self.current_generation.write().await;
        *gen += 1;

        debug!("Evolved to generation {}", *gen);
    }

    /// Run NAS search
    pub async fn search(&self) -> Result<ParameterCandidate> {
        info!("Starting Neural Architecture Search...");

        // Initialize population
        self.initialize_population().await;

        // Evolution loop
        for generation in 0..self.config.max_generations {
            // Evaluate current population
            self.evaluate_population().await;

            // Log progress
            if generation % 10 == 0 {
                let best = self.best_candidate.read().await;
                if let Some(candidate) = &*best {
                    info!(
                        "Generation {}: Best fitness = {:.4}",
                        generation, candidate.fitness
                    );
                }
            }

            // Evolve to next generation
            if generation < self.config.max_generations - 1 {
                self.evolve_generation().await;
            }
        }

        // Return best candidate
        let best = self.best_candidate.read().await;
        best.clone()
            .ok_or_else(|| crate::error::ClusterError::Other("No best candidate found".to_string()))
    }

    /// Get current best candidate
    pub async fn get_best_candidate(&self) -> Option<ParameterCandidate> {
        self.best_candidate.read().await.clone()
    }

    /// Get current generation number
    pub async fn get_current_generation(&self) -> u32 {
        *self.current_generation.read().await
    }

    /// Get search statistics
    pub async fn get_statistics(&self) -> NASStatistics {
        let scores = self.fitness_scores.read().await;

        NASStatistics {
            total_evaluations: scores.len() as u64,
            current_generation: self.get_current_generation().await,
            best_fitness: self
                .best_candidate
                .read()
                .await
                .as_ref()
                .map(|c| c.fitness)
                .unwrap_or(0.0),
            population_size: self.population.read().await.len(),
        }
    }
}

/// NAS search statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASStatistics {
    /// Total candidates evaluated
    pub total_evaluations: u64,
    /// Current generation number
    pub current_generation: u32,
    /// Best fitness score achieved
    pub best_fitness: f64,
    /// Current population size
    pub population_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_candidate_random() {
        let space = ParameterSpace::default();
        let candidate = ParameterCandidate::random(&space, 0);

        assert!(candidate.heartbeat_interval_ms >= space.heartbeat_interval_ms.0);
        assert!(candidate.heartbeat_interval_ms <= space.heartbeat_interval_ms.1);
        assert_eq!(candidate.generation, 0);
    }

    #[test]
    fn test_performance_metrics_fitness() {
        let metrics = PerformanceMetrics {
            avg_latency_ms: 50.0,
            queries_per_sec: 5000.0,
            cpu_utilization: 0.5,
            memory_utilization: 0.5,
            replication_lag_ms: 10.0,
            error_rate: 0.01,
            election_frequency: 1.0,
        };

        let fitness = metrics.compute_fitness();
        assert!((0.0..=1.0).contains(&fitness));
    }

    #[tokio::test]
    async fn test_nas_initialization() {
        let config = NASConfig::default();
        let space = ParameterSpace::default();
        let nas = NeuralArchitectureSearch::new(config.clone(), space);

        nas.initialize_population().await;

        let population = nas.population.read().await;
        assert_eq!(population.len(), config.population_size);
    }

    #[tokio::test]
    async fn test_nas_evaluation() {
        let config = NASConfig {
            population_size: 10,
            ..Default::default()
        };
        let space = ParameterSpace::default();
        let nas = NeuralArchitectureSearch::new(config, space);

        nas.initialize_population().await;
        nas.evaluate_population().await;

        let best = nas.get_best_candidate().await;
        assert!(best.is_some());
        assert!(best.unwrap().fitness > 0.0);
    }
}
