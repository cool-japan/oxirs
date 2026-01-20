//! Type definitions for evolutionary neural architecture system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for evolutionary neural architecture system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryConfig {
    /// Population size for genetic algorithm
    pub population_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Evolution interval in milliseconds
    pub evolution_interval_ms: u64,
    /// Performance improvement threshold for deployment
    pub deployment_improvement_threshold: f64,
    /// Resource constraints
    pub max_memory_mb: usize,
    pub max_compute_units: usize,
    pub max_inference_latency_ms: u64,
    pub energy_efficiency_target: f64,
    /// Diversity requirements
    pub diversity_requirements: DiversityRequirements,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
    /// Neural architecture search strategy
    pub nas_strategy: NASSearchStrategy,
    /// Search space configuration
    pub search_space_config: SearchSpaceConfig,
    /// Performance predictor configuration
    pub predictor_config: PredictorConfig,
    /// Architecture encoding configuration
    pub encoding_config: EncodingConfig,
    /// Genetic operators
    pub genetic_operators: Vec<GeneticOperator>,
    /// Mutation strategies
    pub mutation_strategies: Vec<MutationStrategy>,
    /// Crossover strategies
    pub crossover_strategies: Vec<CrossoverStrategy>,
    /// Selection strategies
    pub selection_strategies: Vec<SelectionStrategy>,
}

impl Default for EvolutionaryConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 50,
            evolution_interval_ms: 60000,           // 1 minute
            deployment_improvement_threshold: 0.05, // 5% improvement
            max_memory_mb: 8192,                    // 8GB
            max_compute_units: 16,
            max_inference_latency_ms: 100,
            energy_efficiency_target: 0.9,
            diversity_requirements: DiversityRequirements::default(),
            performance_targets: PerformanceTargets::default(),
            nas_strategy: NASSearchStrategy::EvolutionarySearch,
            search_space_config: SearchSpaceConfig,
            predictor_config: PredictorConfig,
            encoding_config: EncodingConfig,
            genetic_operators: vec![GeneticOperator::default()],
            mutation_strategies: vec![MutationStrategy::default()],
            crossover_strategies: vec![CrossoverStrategy],
            selection_strategies: vec![SelectionStrategy],
        }
    }
}

/// Neural Architecture Search strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NASSearchStrategy {
    RandomSearch,
    BayesianOptimization,
    EvolutionarySearch,
    GradientBased,
    ReinforcementLearning,
}

/// Diversity requirements for evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityRequirements {
    pub min_structural_diversity: f64,
    pub min_functional_diversity: f64,
    pub min_performance_diversity: f64,
    pub diversity_preservation_ratio: f64,
}

impl Default for DiversityRequirements {
    fn default() -> Self {
        Self {
            min_structural_diversity: 0.3,
            min_functional_diversity: 0.4,
            min_performance_diversity: 0.2,
            diversity_preservation_ratio: 0.2,
        }
    }
}

/// Performance targets for evolved architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub min_accuracy: f64,
    pub max_latency_ms: u64,
    pub max_memory_mb: usize,
    pub min_throughput: f64,
    pub min_energy_efficiency: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            min_accuracy: 0.9,
            max_latency_ms: 100,
            max_memory_mb: 1024,
            min_throughput: 1000.0,
            min_energy_efficiency: 0.8,
        }
    }
}

/// Resource constraints for evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_memory_mb: usize,
    pub max_compute_units: usize,
    pub max_inference_latency_ms: u64,
    pub energy_efficiency_target: f64,
}

/// Neural architecture representation
#[derive(Debug, Clone, Default)]
pub struct NeuralArchitecture {
    pub layers: Vec<LayerSpec>,
    pub connections: Vec<ConnectionSpec>,
    pub topology: TopologySpec,
    pub hyperparameters: HyperparameterSpec,
}

/// Layer specification for neural architecture
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub layer_type: LayerType,
    pub size: usize,
    pub activation: ActivationType,
    pub regularization: RegularizationType,
    pub parameters: HashMap<String, f64>,
}

/// Layer types for neural architectures
#[derive(Debug, Clone)]
pub enum LayerType {
    Dense,
    Convolutional,
    Recurrent,
    Attention,
    Normalization,
    Dropout,
    Pooling,
    Embedding,
}

/// Activation function types
#[derive(Debug, Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    LeakyReLU,
    ELU,
}

/// Regularization types
#[derive(Debug, Clone)]
pub enum RegularizationType {
    None,
    L1,
    L2,
    Dropout,
    BatchNorm,
    LayerNorm,
}

/// Connection specification between layers
#[derive(Debug, Clone)]
pub struct ConnectionSpec {
    pub from_layer: usize,
    pub to_layer: usize,
    pub connection_type: ConnectionType,
    pub weight: f64,
}

/// Connection types between layers
#[derive(Debug, Clone)]
pub enum ConnectionType {
    FullyConnected,
    Convolutional,
    Residual,
    Skip,
    Attention,
}

/// Topology specification
#[derive(Debug, Clone)]
pub struct TopologySpec {
    pub topology_type: TopologyType,
    pub depth: usize,
    pub width: usize,
    pub branching_factor: usize,
}

impl Default for TopologySpec {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::Sequential,
            depth: 3,
            width: 128,
            branching_factor: 1,
        }
    }
}

/// Topology types for neural architectures
#[derive(Debug, Clone)]
pub enum TopologyType {
    Sequential,
    Residual,
    DenseNet,
    Inception,
    AttentionBased,
    Custom,
}

/// Hyperparameter specification
#[derive(Debug, Clone)]
pub struct HyperparameterSpec {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub optimizer: OptimizerType,
    pub scheduler: SchedulerType,
    pub regularization_strength: f64,
}

impl Default for HyperparameterSpec {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            optimizer: OptimizerType::Adam,
            scheduler: SchedulerType::StepLR,
            regularization_strength: 0.01,
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
}

/// Learning rate scheduler types
#[derive(Debug, Clone)]
pub enum SchedulerType {
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    WarmupLR,
}

/// Performance metrics for architectures
#[derive(Debug, Clone)]
pub struct ArchitecturePerformanceMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub inference_latency_ms: f64,
    pub training_time_ms: f64,
    pub memory_efficiency: f64,
    pub energy_efficiency: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsageMetrics {
    pub memory_usage_mb: f64,
    pub compute_units_used: f64,
    pub inference_ops_per_second: f64,
    pub energy_consumption_joules: f64,
    pub storage_requirements_mb: f64,
}

/// Validation task for architecture evolution
#[derive(Debug, Clone)]
pub struct ValidationTask {
    pub task_id: String,
    pub complexity_score: f64,
    pub resource_requirements: ResourceConstraints,
    pub performance_requirements: PerformanceTargets,
    pub specialization_hints: Vec<SpecializationHint>,
}

/// Specialization hint for architecture design
#[derive(Debug, Clone)]
pub struct SpecializationHint {
    pub hint_type: String,
    pub weight: f64,
    pub description: String,
}

/// Context for evolutionary validation
#[derive(Debug)]
pub struct EvolutionaryValidationContext {
    pub current_validation_tasks: Vec<ValidationTask>,
    pub performance_targets: PerformanceTargets,
    pub resource_constraints: ResourceConstraints,
    pub diversity_requirements: DiversityRequirements,
    pub specialization_hints: Vec<SpecializationHint>,
}

/// Result of evolutionary validation
#[derive(Debug)]
pub struct EvolutionaryValidationResult {
    pub evolved_architectures: Vec<EvolvedArchitecture>,
    pub fitness_improvements: FitnessImprovements,
    pub generation_statistics: GenerationStatistics,
    pub self_modification_results: SelfModificationResults,
    pub pareto_front: ParetoFront,
    pub evolution_time: Duration,
    pub convergence_metrics: ConvergenceMetrics,
}

/// Evolved neural architecture
#[derive(Debug, Clone)]
pub struct EvolvedArchitecture {
    pub architecture_id: String,
    pub neural_architecture: NeuralArchitecture,
    pub fitness_score: f64,
    pub performance_metrics: ArchitecturePerformanceMetrics,
    pub resource_usage: ResourceUsageMetrics,
    pub generation: usize,
    pub parent_lineage: Vec<String>,
    pub mutation_history: Vec<MutationRecord>,
}

/// Fitness improvements from evolution
#[derive(Debug, Default)]
pub struct FitnessImprovements {
    pub average_improvement: f64,
    pub best_improvement: f64,
    pub improvement_distribution: Vec<f64>,
    pub convergence_rate: f64,
}

/// Generation statistics
#[derive(Debug, Default)]
pub struct GenerationStatistics {
    pub generation_number: usize,
    pub population_size: usize,
    pub average_fitness: f64,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub fitness_variance: f64,
    pub diversity_metrics: DiversityMetrics,
}

/// Diversity metrics for population
#[derive(Debug, Clone, Default)]
pub struct DiversityMetrics {
    pub structural_diversity: f64,
    pub functional_diversity: f64,
    pub performance_diversity: f64,
    pub overall_diversity: f64,
}

/// Self-modification results
#[derive(Debug)]
pub struct SelfModificationResults {
    pub modifications_applied: usize,
    pub modification_success_rate: f64,
    pub performance_improvements: Vec<f64>,
    pub modification_insights: Vec<ModificationInsight>,
}

/// Modification insight
#[derive(Debug)]
pub struct ModificationInsight {
    pub insight_type: String,
    pub confidence: f64,
    pub description: String,
}

/// Pareto front for multi-objective optimization
#[derive(Debug, Default, Clone)]
pub struct ParetoFront {
    pub solutions: Vec<EvolvedArchitecture>,
    pub convergence_metrics: ParetoConvergenceMetrics,
}

/// Pareto convergence metrics
#[derive(Debug, Default, Clone)]
pub struct ParetoConvergenceMetrics {
    pub hypervolume: f64,
    pub spacing: f64,
    pub spread: f64,
}

/// Convergence metrics
#[derive(Debug, Default)]
pub struct ConvergenceMetrics {
    pub generations_to_converge: usize,
    pub final_diversity: f64,
    pub convergence_rate: f64,
}

/// Mutation record
#[derive(Debug, Clone)]
pub struct MutationRecord {
    pub mutation_type: String,
    pub generation: usize,
    pub fitness_change: f64,
}

// Placeholder types for compilation compatibility
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchSpaceConfig;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictorConfig;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncodingConfig;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneticOperator {
    pub mutation_probability: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MutationStrategy {
    pub mutation_probability: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossoverStrategy;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SelectionStrategy;

// Initialization result types
#[derive(Debug, Clone, Default)]
pub struct NASInitResult {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct EvolutionaryInitResult {
    pub nas_engine: NASInitResult,
    pub genetic_system: GeneticInitResult,
    pub population: PopulationInitResult,
    pub evolution_process: EvolutionInitResult2,
    pub optimization: OptimizationInitResult,
    pub performance_evaluator: PerformanceEvaluatorInitResult,
    pub timestamp: std::time::SystemTime,
}

// Rename the one to avoid conflict
#[derive(Debug, Clone, Default)]
pub struct EvolutionInitResult2 {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct GeneticInitResult {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct PopulationInitResult {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct EvolutionInitResult {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceEvaluatorInitResult {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct OptimizationInitResult {
    pub success: bool,
    pub message: String,
}

// Missing types referenced in core.rs
#[derive(Debug, Clone, Default)]
pub struct OffspringGeneration {
    pub offspring: Vec<EvolvedArchitecture>,
    pub generation_number: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MutationResults {
    pub successful_mutations: usize,
    pub failed_mutations: usize,
    pub mutation_rate: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ParetoOptimization {
    pub front_size: usize,
    pub convergence_rate: f64,
    pub pareto_optimal_set: Vec<EvolvedArchitecture>,
    pub current_pareto_front: ParetoFront,
}

// Metrics type
#[derive(Debug, Clone, Default)]
pub struct EvolutionaryMetrics {
    pub generations_completed: usize,
    pub best_fitness: f64,
    pub average_fitness: f64,
    pub population_diversity: f64,
}

// Missing types for genetic programming
#[derive(Debug, Clone, Default)]
pub struct ParentSelection {
    pub selected_parents: Vec<String>,
    pub selection_strategy: String,
}

// Removed duplicate MutationResults definition - first one is used
