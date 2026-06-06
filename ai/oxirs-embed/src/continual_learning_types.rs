use crate::{ModelConfig, TrainingStats};
use chrono::{DateTime, Utc};
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContinualLearningConfig {
    pub base_config: ModelConfig,
    pub memory_config: MemoryConfig,
    pub regularization_config: RegularizationConfig,
    pub architecture_config: ArchitectureConfig,
    pub task_config: TaskConfig,
    pub replay_config: ReplayConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub memory_type: MemoryType,
    pub memory_capacity: usize,
    pub update_strategy: MemoryUpdateStrategy,
    pub consolidation: ConsolidationConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_type: MemoryType::EpisodicMemory,
            memory_capacity: 10000,
            update_strategy: MemoryUpdateStrategy::ReservoirSampling,
            consolidation: ConsolidationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    ProceduralMemory,
    HybridMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryUpdateStrategy {
    FIFO,
    Random,
    ReservoirSampling,
    ImportanceBased,
    GradientBased,
    ClusteringBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    pub enabled: bool,
    pub frequency: usize,
    pub strength: f32,
    pub sleep_consolidation: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: 1000,
            strength: 0.1,
            sleep_consolidation: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    pub methods: Vec<RegularizationMethod>,
    pub ewc_config: EWCConfig,
    pub si_config: SynapticIntelligenceConfig,
    pub lwf_config: LwFConfig,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                RegularizationMethod::EWC,
                RegularizationMethod::SynapticIntelligence,
            ],
            ewc_config: EWCConfig::default(),
            si_config: SynapticIntelligenceConfig::default(),
            lwf_config: LwFConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RegularizationMethod {
    EWC,
    SynapticIntelligence,
    LwF,
    MAS,
    RiemannianWalk,
    PackNet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCConfig {
    pub lambda: f32,
    pub fisher_method: FisherMethod,
    pub online: bool,
    pub gamma: f32,
}

impl Default for EWCConfig {
    fn default() -> Self {
        Self {
            lambda: 0.4,
            fisher_method: FisherMethod::Empirical,
            online: true,
            gamma: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FisherMethod {
    Empirical,
    True,
    Diagonal,
    BlockDiagonal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticIntelligenceConfig {
    pub c: f32,
    pub xi: f32,
    pub damping: f32,
}

impl Default for SynapticIntelligenceConfig {
    fn default() -> Self {
        Self {
            c: 0.1,
            xi: 1.0,
            damping: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwFConfig {
    pub alpha: f32,
    pub temperature: f32,
    pub attention_transfer: bool,
}

impl Default for LwFConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            temperature: 4.0,
            attention_transfer: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub adaptation_method: ArchitectureAdaptation,
    pub progressive_config: ProgressiveConfig,
    pub dynamic_config: DynamicConfig,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            adaptation_method: ArchitectureAdaptation::Progressive,
            progressive_config: ProgressiveConfig::default(),
            dynamic_config: DynamicConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureAdaptation {
    Progressive,
    Dynamic,
    PackNet,
    HAT,
    Supermasks,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    pub columns_per_task: usize,
    pub lateral_strength: f32,
    pub column_capacity: usize,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            columns_per_task: 1,
            lateral_strength: 0.5,
            column_capacity: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicConfig {
    pub expansion_threshold: f32,
    pub pruning_threshold: f32,
    pub growth_rate: f32,
    pub max_size: usize,
}

impl Default for DynamicConfig {
    fn default() -> Self {
        Self {
            expansion_threshold: 0.9,
            pruning_threshold: 0.1,
            growth_rate: 0.1,
            max_size: 100000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub detection_method: TaskDetection,
    pub boundary_detection: BoundaryDetection,
    pub switching_strategy: TaskSwitching,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            detection_method: TaskDetection::Automatic,
            boundary_detection: BoundaryDetection::ChangePoint,
            switching_strategy: TaskSwitching::Soft,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskDetection {
    Manual,
    Automatic,
    Oracle,
    Clustering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryDetection {
    ChangePoint,
    DistributionShift,
    LossBased,
    GradientBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskSwitching {
    Hard,
    Soft,
    Attention,
    Gating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    pub methods: Vec<ReplayMethod>,
    pub buffer_size: usize,
    pub replay_ratio: f32,
    pub generative_config: GenerativeReplayConfig,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                ReplayMethod::ExperienceReplay,
                ReplayMethod::GenerativeReplay,
            ],
            buffer_size: 5000,
            replay_ratio: 0.5,
            generative_config: GenerativeReplayConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReplayMethod {
    ExperienceReplay,
    GenerativeReplay,
    PseudoRehearsal,
    MetaReplay,
    GradientEpisodicMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeReplayConfig {
    pub generator_type: GeneratorType,
    pub quality_threshold: f32,
    pub diversity_weight: f32,
}

impl Default for GenerativeReplayConfig {
    fn default() -> Self {
        Self {
            generator_type: GeneratorType::VAE,
            quality_threshold: 0.8,
            diversity_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratorType {
    VAE,
    GAN,
    Flow,
    Diffusion,
}

#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: String,
    pub task_type: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub examples_seen: usize,
    pub performance: f32,
    pub task_embedding: Option<Array1<f32>>,
}

impl TaskInfo {
    pub fn new(task_id: String, task_type: String) -> Self {
        Self {
            task_id,
            task_type,
            start_time: Utc::now(),
            end_time: None,
            examples_seen: 0,
            performance: 0.0,
            task_embedding: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub data: Array1<f32>,
    pub target: Array1<f32>,
    pub task_id: String,
    pub timestamp: DateTime<Utc>,
    pub importance: f32,
    pub access_count: usize,
}

impl MemoryEntry {
    pub fn new(data: Array1<f32>, target: Array1<f32>, task_id: String) -> Self {
        Self {
            data,
            target,
            task_id,
            timestamp: Utc::now(),
            importance: 1.0,
            access_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EWCState {
    pub fisher_information: Array2<f32>,
    pub optimal_parameters: Array2<f32>,
    pub task_id: String,
    pub importance: f32,
}

#[derive(Debug)]
pub struct ContinualLearningModel {
    pub config: ContinualLearningConfig,
    pub model_id: Uuid,
    pub embeddings: Array2<f32>,
    pub task_specific_embeddings: HashMap<String, Array2<f32>>,
    pub episodic_memory: VecDeque<MemoryEntry>,
    pub semantic_memory: HashMap<String, Array1<f32>>,
    pub ewc_states: Vec<EWCState>,
    pub synaptic_importance: Array2<f32>,
    pub parameter_trajectory: Array2<f32>,
    pub current_task: Option<TaskInfo>,
    pub task_history: Vec<TaskInfo>,
    pub task_boundaries: Vec<usize>,
    pub network_columns: Vec<Array2<f32>>,
    pub lateral_connections: Vec<Array2<f32>>,
    pub generator: Option<Array2<f32>>,
    pub discriminator: Option<Array2<f32>>,
    pub entities: HashMap<String, usize>,
    pub relations: HashMap<String, usize>,
    pub examples_seen: usize,
    pub training_stats: Option<TrainingStats>,
    pub is_trained: bool,
}
