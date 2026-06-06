//! Evolutionary Neural Architecture Search — facade
//!
//! Re-exports all public items from the sibling modules.

pub use crate::evolutionary_nas_eval::{
    compute_proxy_accuracy, estimate_flops, estimate_memory_mb, should_stop_early,
    FitnessEvaluator, HardwareProfiler,
};
pub use crate::evolutionary_nas_evolution::{
    CrossoverOperator, DiversityStrategy, EvolutionaryNAS, HardwareOptimizationStrategy,
    MutationOperator, PerformanceModel, SelectionOperator,
};
pub use crate::evolutionary_nas_types::{
    ArchitectureCandidate, ArchitectureGenome, ConnectionGene, ConvergenceMetrics,
    DiversityMetrics, EvaluationDataset, EvolutionaryConfig, FitnessScores, GenerationStatistics,
    GlobalParameters, HardwareMetrics, HardwareTarget, InnovationTracker, ModuleCharacteristics,
    ModuleDefinition, ModuleInterface, NodeGene, ObjectiveWeights, OperationType,
    OptimizationResult, PerformanceMetrics, ProfilingResult, ProgressiveConfig,
};
