//! Advanced Consciousness-Aware Response Generation
//!
//! Modular consciousness system with:
//! - Neural-inspired consciousness simulation
//! - Advanced emotional state tracking with valence/arousal dynamics
//! - Multi-layered memory systems (working, episodic, semantic)
//! - Metacognitive assessment with self-reflection capabilities
//! - Attention mechanism with weighted focus distribution
//! - Temporal pattern recognition and future projection
//! - Dream-state memory consolidation
//! - Dynamic state machine for adaptive behavior
//!
//! ## Module Organization
//!
//! - `model` - Core ConsciousnessModel orchestrating all components
//! - `config` - Configuration structures
//! - `responses` - Response types and metadata
//! - `metrics` - Performance monitoring and snapshots
//! - `attention` - Attention mechanism and focus distribution
//! - `memory` - Multi-layer memory system
//! - `emotional` - Emotional state tracking
//! - `neural` - Neural correlates and activation
//! - `stream` - Consciousness stream management
//! - `metacognitive` - Self-reflection and metacognitive assessment
//! - `state_machine` - State transitions and rules
//! - `dream` - Dream-state processing
//! - `temporal` - Temporal consciousness and projection

pub mod attention;
pub mod config;
pub mod dream;
pub mod emotional;
pub mod memory;
pub mod metacognitive;
pub mod metrics;
pub mod model;
pub mod neural;
pub mod responses;
pub mod state_machine;
pub mod stream;
pub mod temporal;

// Re-export main types
pub use attention::AttentionMechanism;
pub use config::{ConsciousnessConfig, ConsciousnessModelConfig};
pub use dream::DreamStateProcessor;
pub use emotional::AdvancedEmotionalState;
pub use memory::MultiLayerMemorySystem;
pub use metacognitive::{
    ConsciousnessIntegration, EnhancedMetacognitiveLayer, MetacognitiveAssessment,
    MetacognitiveLayer,
};
pub use metrics::{ConsciousnessMetrics, ConsciousnessSnapshot};
pub use model::{ConsciousnessModel, EmotionalState, MemoryTrace};
pub use neural::NeuralCorrelates;
pub use responses::{
    AdvancedConsciousInsight, AdvancedConsciousResponse, AdvancedConsciousnessMetadata,
    AdvancedInsightType, ConsciousInsight, ConsciousResponse, ConsciousnessMetadata, InsightType,
};
pub use state_machine::{ConsciousnessState, ConsciousnessStateMachine, StateTransition};
pub use stream::ConsciousnessStream;
pub use temporal::{TemporalConsciousness, TemporalEvent, TemporalPatternRecognition};
