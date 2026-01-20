//! Consciousness Response Types
//!
//! Defines response structures for consciousness-enhanced query processing.

use super::super::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Basic conscious response with metadata
pub struct ConsciousResponse {
    pub base_response: AssembledContext,
    pub consciousness_metadata: ConsciousnessMetadata,
    pub enhanced_insights: Vec<ConsciousInsight>,
}

/// Basic consciousness metadata
#[derive(Debug, Clone)]
pub struct ConsciousnessMetadata {
    pub awareness_level: f64,
    pub attention_focus: Vec<String>,
    pub emotional_resonance: f64,
    pub metacognitive_confidence: f64,
    pub memory_integration_score: f64,
}

/// Basic conscious insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousInsight {
    pub insight_type: InsightType,
    pub content: String,
    pub confidence: f64,
    pub implications: Vec<String>,
}

/// Basic insight type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    PatternRecognition,
    EmotionalResonance,
    MemoryIntegration,
    ContextualUnderstanding,
    StrategicPlanning,
}

/// Advanced conscious response with enhanced features
pub struct AdvancedConsciousResponse {
    pub base_response: AssembledContext,
    pub consciousness_metadata: AdvancedConsciousnessMetadata,
    pub enhanced_insights: Vec<AdvancedConsciousInsight>,
    pub consciousness_stream_context: Vec<ConsciousnessExperience>,
}

/// Advanced consciousness metadata with detailed tracking
#[derive(Debug, Clone)]
pub struct AdvancedConsciousnessMetadata {
    pub awareness_level: f64,
    pub neural_activation: NeuralActivation,
    pub attention_allocation: AttentionAllocation,
    pub memory_integration: MemoryIntegrationResult,
    pub emotional_response: EmotionalResponse,
    pub metacognitive_result: MetacognitiveResult,
    pub processing_time: Duration,
    pub consciousness_health_score: f64,
}

/// Advanced conscious insight with detailed analysis
#[derive(Debug, Clone)]
pub struct AdvancedConsciousInsight {
    pub insight_type: AdvancedInsightType,
    pub content: String,
    pub confidence: f64,
    pub implications: Vec<String>,
    pub supporting_evidence: Vec<String>,
    pub consciousness_correlation: f64,
}

/// Advanced insight type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdvancedInsightType {
    NeuralPattern,
    MemoryIntegration,
    AttentionFocus,
    StreamCoherence,
    EmotionalResonance,
    MetacognitiveAssessment,
}
