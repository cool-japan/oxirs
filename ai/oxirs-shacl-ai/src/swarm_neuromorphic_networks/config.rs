//! Configuration structures for swarm neuromorphic networks

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for swarm neuromorphic network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmNetworkConfig {
    /// Target swarm size (number of nodes)
    pub target_swarm_size: usize,
    /// Node configuration
    pub node_configuration: SwarmNodeConfig,
    /// Swarm intelligence parameters
    pub swarm_intelligence_config: SwarmIntelligenceConfig,
    /// Network topology settings
    pub topology_config: NetworkTopologyConfig,
    /// Communication protocol settings
    pub communication_config: SwarmCommunicationConfig,
    /// Distributed learning parameters
    pub learning_config: DistributedLearningConfig,
    /// Resilience and fault tolerance settings
    pub resilience_config: SwarmResilienceConfig,
    /// Bio-inspired optimization parameters
    pub bio_optimization_config: BioOptimizationConfig,
}

impl Default for SwarmNetworkConfig {
    fn default() -> Self {
        Self {
            target_swarm_size: 100,
            node_configuration: SwarmNodeConfig::default(),
            swarm_intelligence_config: SwarmIntelligenceConfig::default(),
            topology_config: NetworkTopologyConfig::default(),
            communication_config: SwarmCommunicationConfig::default(),
            learning_config: DistributedLearningConfig::default(),
            resilience_config: SwarmResilienceConfig::default(),
            bio_optimization_config: BioOptimizationConfig::default(),
        }
    }
}

/// Swarm node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmNodeConfig {
    /// Processing capacity per node
    pub processing_capacity: f64,
    /// Memory capacity per node
    pub memory_capacity_mb: usize,
    /// Communication range
    pub communication_range: f64,
    /// Energy consumption model
    pub energy_consumption_model: EnergyConsumptionModel,
}

impl Default for SwarmNodeConfig {
    fn default() -> Self {
        Self {
            processing_capacity: 1e6, // 1 million operations per second
            memory_capacity_mb: 1024,
            communication_range: 100.0,
            energy_consumption_model: EnergyConsumptionModel::Linear,
        }
    }
}

/// Energy consumption models for swarm nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnergyConsumptionModel {
    Linear,
    Logarithmic,
    Exponential,
    BiologicallyInspired,
}

/// Swarm intelligence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmIntelligenceConfig {
    /// Particle swarm optimization parameters
    pub pso_config: PSOConfig,
    /// Ant colony optimization parameters
    pub aco_config: ACOConfig,
    /// Bee algorithm parameters
    pub bee_config: BeeAlgorithmConfig,
    /// Fish schooling parameters
    pub fish_schooling_config: FishSchoolingConfig,
}

impl Default for SwarmIntelligenceConfig {
    fn default() -> Self {
        Self {
            pso_config: PSOConfig::default(),
            aco_config: ACOConfig::default(),
            bee_config: BeeAlgorithmConfig::default(),
            fish_schooling_config: FishSchoolingConfig::default(),
        }
    }
}

/// Particle Swarm Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PSOConfig {
    pub inertia_weight: f64,
    pub cognitive_coefficient: f64,
    pub social_coefficient: f64,
    pub max_velocity: f64,
}

impl Default for PSOConfig {
    fn default() -> Self {
        Self {
            inertia_weight: 0.9,
            cognitive_coefficient: 2.0,
            social_coefficient: 2.0,
            max_velocity: 10.0,
        }
    }
}

/// Ant Colony Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ACOConfig {
    pub pheromone_evaporation_rate: f64,
    pub pheromone_deposit_strength: f64,
    pub alpha: f64, // Pheromone importance
    pub beta: f64,  // Heuristic importance
}

impl Default for ACOConfig {
    fn default() -> Self {
        Self {
            pheromone_evaporation_rate: 0.1,
            pheromone_deposit_strength: 1.0,
            alpha: 1.0,
            beta: 2.0,
        }
    }
}

/// Bee Algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeeAlgorithmConfig {
    pub scout_bee_percentage: f64,
    pub elite_site_count: usize,
    pub selected_site_count: usize,
    pub recruitment_radius: f64,
}

impl Default for BeeAlgorithmConfig {
    fn default() -> Self {
        Self {
            scout_bee_percentage: 0.1,
            elite_site_count: 10,
            selected_site_count: 20,
            recruitment_radius: 5.0,
        }
    }
}

/// Fish Schooling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FishSchoolingConfig {
    pub separation_radius: f64,
    pub alignment_radius: f64,
    pub cohesion_radius: f64,
    pub separation_weight: f64,
    pub alignment_weight: f64,
    pub cohesion_weight: f64,
}

impl Default for FishSchoolingConfig {
    fn default() -> Self {
        Self {
            separation_radius: 2.0,
            alignment_radius: 5.0,
            cohesion_radius: 8.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
        }
    }
}

/// Network topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopologyConfig {
    pub topology_type: TopologyType,
    pub connection_density: f64,
    pub reconfiguration_frequency: Duration,
    pub adaptive_threshold: f64,
}

impl Default for NetworkTopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::SmallWorld,
            connection_density: 0.3,
            reconfiguration_frequency: Duration::from_secs(60),
            adaptive_threshold: 0.8,
        }
    }
}

/// Network topology types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyType {
    FullyConnected,
    Ring,
    Star,
    Mesh,
    SmallWorld,
    ScaleFree,
    Random,
}

/// Swarm communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCommunicationConfig {
    pub message_propagation_speed: f64,
    pub message_reliability: f64,
    pub bandwidth_per_node_mbps: f64,
    pub pheromone_diffusion_rate: f64,
}

impl Default for SwarmCommunicationConfig {
    fn default() -> Self {
        Self {
            message_propagation_speed: 100.0, // units per second
            message_reliability: 0.95,
            bandwidth_per_node_mbps: 10.0,
            pheromone_diffusion_rate: 0.1,
        }
    }
}

/// Distributed learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedLearningConfig {
    pub federated_learning_rounds: usize,
    pub local_learning_epochs: usize,
    pub knowledge_sharing_frequency: Duration,
    pub consensus_threshold: f64,
}

impl Default for DistributedLearningConfig {
    fn default() -> Self {
        Self {
            federated_learning_rounds: 10,
            local_learning_epochs: 5,
            knowledge_sharing_frequency: Duration::from_secs(30),
            consensus_threshold: 0.8,
        }
    }
}

/// Swarm resilience configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResilienceConfig {
    pub fault_detection_sensitivity: f64,
    pub self_healing_response_time: Duration,
    pub redundancy_factor: f64,
    pub recovery_strategy: RecoveryStrategy,
}

impl Default for SwarmResilienceConfig {
    fn default() -> Self {
        Self {
            fault_detection_sensitivity: 0.95,
            self_healing_response_time: Duration::from_secs(5),
            redundancy_factor: 2.0,
            recovery_strategy: RecoveryStrategy::GradualRecovery,
        }
    }
}

/// Recovery strategies for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    ImmediateRecovery,
    GradualRecovery,
    AdaptiveRecovery,
    CollectiveRecovery,
}

/// Bio-inspired optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioOptimizationConfig {
    pub genetic_algorithm_config: GeneticAlgorithmConfig,
    pub neural_evolution_config: NeuralEvolutionConfig,
    pub evolutionary_programming_config: EvolutionaryProgrammingConfig,
}

impl Default for BioOptimizationConfig {
    fn default() -> Self {
        Self {
            genetic_algorithm_config: GeneticAlgorithmConfig::default(),
            neural_evolution_config: NeuralEvolutionConfig::default(),
            evolutionary_programming_config: EvolutionaryProgrammingConfig::default(),
        }
    }
}

/// Genetic algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticAlgorithmConfig {
    pub population_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub selection_pressure: f64,
}

impl Default for GeneticAlgorithmConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection_pressure: 2.0,
        }
    }
}

/// Neural evolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEvolutionConfig {
    pub network_topology_evolution: bool,
    pub weight_evolution: bool,
    pub activation_function_evolution: bool,
    pub complexity_penalty: f64,
}

impl Default for NeuralEvolutionConfig {
    fn default() -> Self {
        Self {
            network_topology_evolution: true,
            weight_evolution: true,
            activation_function_evolution: true,
            complexity_penalty: 0.01,
        }
    }
}

/// Evolutionary programming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryProgrammingConfig {
    pub tournament_size: usize,
    pub survival_rate: f64,
    pub diversity_pressure: f64,
    pub elitism_percentage: f64,
}

impl Default for EvolutionaryProgrammingConfig {
    fn default() -> Self {
        Self {
            tournament_size: 5,
            survival_rate: 0.5,
            diversity_pressure: 0.1,
            elitism_percentage: 0.1,
        }
    }
}
