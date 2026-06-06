//! Evolutionary NAS — Types
//!
//! Chromosome/genome representation, mutation operators, fitness types, and population config.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for evolutionary neural architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub elite_percentage: f32,
    pub tournament_size: usize,
    pub crossover_probability: f32,
    pub mutation_probability: f32,
    pub diversity_strength: f32,
    pub objective_weights: ObjectiveWeights,
    pub target_hardware: HardwareTarget,
    pub progressive_config: ProgressiveConfig,
}

impl Default for EvolutionaryConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            elite_percentage: 0.1,
            tournament_size: 5,
            crossover_probability: 0.8,
            mutation_probability: 0.1,
            diversity_strength: 0.3,
            objective_weights: ObjectiveWeights::default(),
            target_hardware: HardwareTarget::default(),
            progressive_config: ProgressiveConfig::default(),
        }
    }
}

/// Multi-objective optimization weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveWeights {
    pub accuracy_weight: f32,
    pub efficiency_weight: f32,
    pub memory_weight: f32,
    pub simplicity_weight: f32,
    pub novelty_weight: f32,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            accuracy_weight: 0.4,
            efficiency_weight: 0.3,
            memory_weight: 0.15,
            simplicity_weight: 0.1,
            novelty_weight: 0.05,
        }
    }
}

/// Target hardware configuration for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareTarget {
    HighPerformanceGPU {
        gpu_memory_gb: f32,
        compute_capability: f32,
        parallelism_factor: f32,
    },
    EdgeDevice {
        cpu_cores: usize,
        memory_mb: f32,
        power_budget_watts: f32,
    },
    CloudDeployment {
        instance_type: String,
        cost_per_hour: f32,
        scaling_factor: f32,
    },
    NeuromorphicChip {
        neuron_count: usize,
        synapse_count: usize,
        spike_rate_khz: f32,
    },
}

impl Default for HardwareTarget {
    fn default() -> Self {
        Self::HighPerformanceGPU {
            gpu_memory_gb: 16.0,
            compute_capability: 8.0,
            parallelism_factor: 1.0,
        }
    }
}

/// Progressive complexification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    pub start_complexity: usize,
    pub max_complexity: usize,
    pub complexity_increase_rate: f32,
    pub enable_modular_building: bool,
    pub enable_module_library: bool,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            start_complexity: 3,
            max_complexity: 20,
            complexity_increase_rate: 0.1,
            enable_modular_building: true,
            enable_module_library: true,
        }
    }
}

// ── Genome ────────────────────────────────────────────────────────────────────

/// Architecture candidate in the population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureCandidate {
    pub id: Uuid,
    pub genome: ArchitectureGenome,
    pub fitness: FitnessScores,
    pub performance: Option<PerformanceMetrics>,
    pub generation: usize,
    pub parents: Vec<Uuid>,
    pub novelty_score: f32,
    pub hardware_metrics: HardwareMetrics,
}

/// Architecture genome representation using graph-based encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureGenome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub global_params: GlobalParameters,
    pub modules: Vec<ModuleDefinition>,
}

/// Node gene representing a layer or operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: usize,
    pub operation: OperationType,
    pub parameters: HashMap<String, f32>,
    pub active: bool,
    pub innovation_number: usize,
}

/// Connection gene representing data flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub from_node: usize,
    pub to_node: usize,
    pub weight: f32,
    pub active: bool,
    pub innovation_number: usize,
}

/// Types of operations available for architecture building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Linear {
        input_dim: usize,
        output_dim: usize,
    },
    Convolution {
        filters: usize,
        kernel_size: usize,
    },
    GraphConv {
        channels: usize,
        aggregation: String,
    },
    Attention {
        heads: usize,
        embed_dim: usize,
    },
    Transformer {
        layers: usize,
        heads: usize,
    },
    Embedding {
        vocab_size: usize,
        embed_dim: usize,
    },
    Activation {
        function: String,
    },
    Normalization {
        method: String,
    },
    Dropout {
        rate: f32,
    },
    SkipConnection,
    Pooling {
        method: String,
        size: usize,
    },
    Custom {
        operation_id: String,
        params: HashMap<String, f32>,
    },
}

/// Global parameters affecting the entire architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalParameters {
    pub learning_rate: f32,
    pub optimizer: String,
    pub regularization: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

impl Default for GlobalParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            optimizer: "adam".to_string(),
            regularization: 0.01,
            batch_size: 32,
            epochs: 100,
        }
    }
}

/// Module definition for modular architecture building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDefinition {
    pub id: String,
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub interface: ModuleInterface,
    pub characteristics: ModuleCharacteristics,
}

/// Module interface specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInterface {
    pub input_dim: usize,
    pub output_dim: usize,
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
}

/// Module performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleCharacteristics {
    pub computational_cost: f64,
    pub memory_cost: f64,
    pub accuracy_contribution: f32,
    pub stability: f32,
}

// ── Fitness & Performance ─────────────────────────────────────────────────────

/// Fitness scores for multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessScores {
    pub overall_fitness: f32,
    pub accuracy: f32,
    pub efficiency: f32,
    pub memory_efficiency: f32,
    pub simplicity: f32,
    pub novelty: f32,
    pub hardware_compatibility: f32,
    pub pareto_rank: usize,
    pub crowding_distance: f32,
}

impl Default for FitnessScores {
    fn default() -> Self {
        Self {
            overall_fitness: 0.0,
            accuracy: 0.0,
            efficiency: 0.0,
            memory_efficiency: 0.0,
            simplicity: 0.0,
            novelty: 0.0,
            hardware_compatibility: 0.0,
            pareto_rank: 0,
            crowding_distance: 0.0,
        }
    }
}

/// Performance metrics for architecture evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub training_accuracy: f32,
    pub validation_accuracy: f32,
    pub test_accuracy: Option<f32>,
    pub training_time: f64,
    pub inference_time_ms: f32,
    pub memory_usage_mb: f32,
    pub energy_consumption: Option<f32>,
    pub model_size: usize,
    pub flops: u64,
}

/// Hardware-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    pub gpu_utilization: f32,
    pub memory_utilization: f32,
    pub throughput: f32,
    pub power_consumption: f32,
    pub efficiency_score: f32,
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self {
            gpu_utilization: 0.0,
            memory_utilization: 0.0,
            throughput: 0.0,
            power_consumption: 0.0,
            efficiency_score: 0.0,
        }
    }
}

// ── History & Tracking ────────────────────────────────────────────────────────

/// Statistics for each generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStatistics {
    pub generation: usize,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub fitness_std: f32,
    pub diversity_score: f32,
    pub new_innovations: usize,
    pub timestamp: DateTime<Utc>,
}

/// Innovation tracking for genetic operators
pub struct InnovationTracker {
    pub(crate) next_innovation: usize,
    pub(crate) innovation_history: HashMap<String, usize>,
    pub(crate) innovation_fitness: HashMap<usize, f32>,
}

impl InnovationTracker {
    pub fn new() -> Self {
        Self {
            next_innovation: 1,
            innovation_history: HashMap::new(),
            innovation_fitness: HashMap::new(),
        }
    }

    pub fn get_innovation_number(&mut self, innovation_key: &str) -> usize {
        if let Some(&innovation) = self.innovation_history.get(innovation_key) {
            innovation
        } else {
            let innovation = self.next_innovation;
            self.next_innovation += 1;
            self.innovation_history
                .insert(innovation_key.to_string(), innovation);
            innovation
        }
    }
}

impl Default for InnovationTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Convergence metrics for evolution monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    pub improvement_rate: f32,
    pub stagnation_count: usize,
    pub diversity_trend: Vec<f32>,
    pub convergence_probability: f32,
}

/// Diversity metrics for population analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    pub genotypic_diversity: f32,
    pub phenotypic_diversity: f32,
    pub novelty_distribution: Vec<f32>,
    pub population_entropy: f32,
}

// ── Evaluation support ────────────────────────────────────────────────────────

/// Dataset for architecture evaluation
#[derive(Debug, Clone)]
pub struct EvaluationDataset {
    pub name: String,
    pub train_triples: Vec<(String, String, String)>,
    pub val_triples: Vec<(String, String, String)>,
    pub test_triples: Option<Vec<(String, String, String)>>,
    pub entity_vocab: std::collections::HashSet<String>,
    pub relation_vocab: std::collections::HashSet<String>,
}

/// Profiling result for hardware measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingResult {
    pub architecture_id: Uuid,
    pub hardware_metrics: HardwareMetrics,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
}

/// Hardware optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub performance_improvement: f32,
    pub efficiency_gain: f32,
    pub confidence: f32,
    pub modifications: Vec<String>,
}
