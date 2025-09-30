//! Configuration types for Neural Architecture Search

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Neural Architecture Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NASConfig {
    /// Population size for evolutionary algorithms
    pub population_size: usize,
    /// Maximum number of generations/iterations
    pub max_generations: usize,
    /// Mutation rate for evolutionary algorithms
    pub mutation_rate: f64,
    /// Crossover rate for evolutionary algorithms
    pub crossover_rate: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Target performance threshold
    pub target_performance: f64,
    /// Maximum training epochs per architecture
    pub max_training_epochs: usize,
    /// Evaluation timeout in seconds
    pub evaluation_timeout: u64,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Multi-objective optimization weights
    pub multi_objective_weights: MultiObjectiveWeights,
    /// Progressive search strategy
    pub progressive_strategy: ProgressiveStrategy,
    /// Hardware-aware optimization
    pub hardware_aware: bool,
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
}

/// Multi-objective optimization weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveWeights {
    /// Weight for accuracy/performance
    pub performance_weight: f64,
    /// Weight for model efficiency (memory, FLOPs)
    pub efficiency_weight: f64,
    /// Weight for inference latency
    pub latency_weight: f64,
    /// Weight for energy consumption
    pub energy_weight: f64,
    /// Weight for model size
    pub size_weight: f64,
    /// Custom objective weights
    pub custom_weights: HashMap<String, f64>,
}

/// Progressive search strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveStrategy {
    /// Enable progressive search
    pub enabled: bool,
    /// Starting search complexity
    pub initial_complexity: SearchComplexity,
    /// Complexity increase schedule
    pub complexity_schedule: ComplexitySchedule,
    /// Performance thresholds for progression
    pub progression_thresholds: Vec<f64>,
    /// Knowledge transfer between stages
    pub enable_knowledge_transfer: bool,
}

/// Search complexity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum SearchComplexity {
    Simple,
    Medium,
    Complex,
    VeryComplex,
}

/// Complexity increase schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexitySchedule {
    Linear { steps: usize },
    Exponential { base: f64 },
    Adaptive { threshold: f64 },
    Manual { schedule: Vec<(usize, SearchComplexity)> },
}

/// Resource constraints for architecture evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum model parameters
    pub max_parameters: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Maximum training time in minutes
    pub max_training_time_minutes: usize,
    /// Maximum inference latency in milliseconds
    pub max_inference_latency_ms: f64,
    /// Target compression ratio
    pub target_compression_ratio: f64,
}

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            early_stopping_patience: 10,
            target_performance: 0.95,
            max_training_epochs: 100,
            evaluation_timeout: 3600, // 1 hour
            resource_constraints: ResourceConstraints::default(),
            random_seed: 42,
            multi_objective_weights: MultiObjectiveWeights::default(),
            progressive_strategy: ProgressiveStrategy::default(),
            hardware_aware: true,
            enable_transfer_learning: false,
        }
    }
}

impl Default for MultiObjectiveWeights {
    fn default() -> Self {
        Self {
            performance_weight: 0.6,
            efficiency_weight: 0.2,
            latency_weight: 0.1,
            energy_weight: 0.05,
            size_weight: 0.05,
            custom_weights: HashMap::new(),
        }
    }
}

impl Default for ProgressiveStrategy {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_complexity: SearchComplexity::Simple,
            complexity_schedule: ComplexitySchedule::Linear { steps: 10 },
            progression_thresholds: vec![0.8, 0.85, 0.9],
            enable_knowledge_transfer: true,
        }
    }
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_parameters: 10_000_000,  // 10M parameters
            max_memory_mb: 2000,         // 2GB
            max_training_time_minutes: 120, // 2 hours
            max_inference_latency_ms: 100.0, // 100ms
            target_compression_ratio: 0.1,   // 10x compression
        }
    }
}