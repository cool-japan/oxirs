//! Consciousness-Inspired Computing Module
//!
//! This module implements artificial consciousness concepts for enhanced
//! query optimization and data processing, including:
//!
//! - Intuitive query planning using pattern memory and gut feelings
//! - Creative optimization strategies inspired by human creativity
//! - Emotional context for data relations and processing
//! - Dream-state graph processing for memory consolidation
//! - Quantum-inspired consciousness states for enhanced processing
//! - Emotional learning networks for empathetic decision-making
//! - Advanced dream processing for pattern discovery and insight generation
//!
//! These features represent the cutting edge of consciousness-inspired
//! computing in the semantic web domain.
//!
//! # Module layout
//!
//! The implementation is split across sibling modules:
//! - [`consciousness_module`]: the integrated [`ConsciousnessModule`] and its
//!   supporting insight/approach/metric types.
//! - [`meta_consciousness`]: [`MetaConsciousness`] self-awareness and
//!   cross-component communication.
//! - The remaining sibling modules each implement a single consciousness
//!   subsystem (intuitive planner, quantum consciousness, emotional learning,
//!   dream processing, and so on).

#![allow(dead_code)]

pub mod consciousness_module;
pub mod dream_processing;
pub mod emotional_learning;
pub mod enhanced_coordinator;
pub mod intuitive_planner;
pub mod meta_consciousness;
pub mod quantum_consciousness;
pub mod quantum_genetic_optimizer;
pub mod temporal_consciousness;

#[cfg(test)]
mod consciousness_tests;

pub use intuitive_planner::{
    ComplexityLevel, CreativeTechnique, CreativityEngine, DatasetSize, ExecutionResults,
    GutFeelingEngine, IntuitionNetwork, IntuitiveExecutionPlan, IntuitiveQueryPlanner,
    PatternCharacteristic, PatternMemory, PerformanceRequirement, QueryContext,
};

pub use quantum_consciousness::{
    BellMeasurement, BellState, PatternEntanglement, QuantumConsciousnessState,
    QuantumErrorCorrection, QuantumMeasurement, QuantumMetrics, QuantumSuperposition,
};

pub use emotional_learning::{
    CompassionResponse, CompassionType, EmotionalApproach, EmotionalAssociation,
    EmotionalExperience, EmotionalInsights, EmotionalLearningNetwork, EmotionalMemory,
    EmotionalPrediction, MoodState, MoodTracker, RegulationOutcome,
};

pub use dream_processing::{
    DreamProcessor, DreamQuality, DreamSequence, DreamState, MemoryConsolidator, MemoryContent,
    MemoryTrace, MemoryType, ProcessingSummary, SequenceType, StepResult, WakeupReport,
    WorkingMemory,
};

pub use quantum_genetic_optimizer::{
    BellStateType, ConsciousnessEvolutionInsight, InsightType, OptimizationStrategy,
    QuantumEntanglementLevel, QuantumEvolutionResult, QuantumGeneticOptimizer,
    QuantumOptimizationSuperposition,
};

pub use enhanced_coordinator::{
    ActivationCondition, ConditionType, ConsciousnessOptimizer, CoordinationResult,
    EnhancedConsciousnessCoordinator, EvolutionCheckpoint, IntegrationPattern, OptimizationResult,
    PatternAnalysis, PatternPerformanceMetrics, PerformanceImprovement, SyncRequirements,
    SynchronizationMonitor,
};

pub use temporal_consciousness::{
    EmotionalContextResult, EmotionalTrend, EvolutionSnapshot, FutureProjection,
    HistoricalContextResult, PatternEvolutionTracker, PredictionResult, RecommendationType,
    SequenceAnalysisResult, SequenceStep, TemporalAnalysisResult, TemporalConsciousness,
    TemporalExperience, TemporalRecommendation, TemporalSequence, TrendAnalysis, TrendDirection,
};

pub use consciousness_module::{
    ConsciousnessApproach, ConsciousnessInsights, ConsciousnessMetadata, ConsciousnessModule,
    ConsciousnessPerformanceMetrics, EmotionalState, ExperienceFeedback, OptimizedConsciousPlan,
    QueryExecutionMetrics,
};

pub use meta_consciousness::{
    AdaptiveRecommendations, ConsciousnessMessage, IntegrationSyncState, MessageType,
    MetaConsciousness, PerformanceMetric,
};
