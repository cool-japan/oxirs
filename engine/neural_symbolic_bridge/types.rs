//! Type definitions for Neural-Symbolic Bridge
//!
//! Contains all the core data structures, enums, and configuration types
//! used throughout the neural-symbolic bridge system.

use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

/// Neural-symbolic query types
#[derive(Debug, Clone)]
pub enum HybridQuery {
    /// Semantic similarity with symbolic constraints
    SimilarityWithConstraints {
        text_query: String,
        sparql_constraints: String,
        threshold: f32,
        limit: usize,
    },
    /// Reasoning-guided vector search
    ReasoningGuidedSearch {
        concept_hierarchy: String,
        search_terms: Vec<String>,
        inference_depth: u32,
    },
    /// Multimodal knowledge graph completion
    KnowledgeCompletion {
        incomplete_triples: Vec<String>,
        context_embeddings: Vec<f32>,
        confidence_threshold: f32,
    },
    /// Explainable similarity reasoning
    ExplainableSimilarity {
        query: String,
        explanation_depth: u32,
        include_provenance: bool,
    },
    /// Temporal reasoning with time-aware constraints
    TemporalReasoning {
        query: String,
        temporal_constraints: TemporalConstraints,
        reasoning_horizon: Duration,
        include_predictions: bool,
    },
    /// Advanced query optimization with learning
    AdaptiveOptimization {
        query: String,
        optimization_history: Vec<QueryPerformance>,
        learning_mode: LearningMode,
        adaptation_rate: f32,
    },
}

/// Hybrid query result
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub symbolic_matches: Vec<SymbolicMatch>,
    pub vector_matches: Vec<VectorMatch>,
    pub combined_score: f32,
    pub explanation: Option<Explanation>,
    pub confidence: f32,
}

/// Consciousness-enhanced hybrid query result with advanced AI insights
#[derive(Debug, Clone)]
pub struct ConsciousnessEnhancedResult {
    /// Base hybrid query result
    pub hybrid_result: HybridResult,
    /// Consciousness insights used in processing
    pub consciousness_insights: Option<crate::consciousness::ConsciousnessInsights>,
    /// Quantum enhancement details if applied
    pub quantum_enhancements: Option<QuantumEnhancements>,
    /// Emotional context during processing
    pub emotional_context: Option<EmotionalContext>,
    /// Dream processing discoveries
    pub dream_discoveries: Option<DreamDiscoveries>,
    /// Performance prediction for similar future queries
    pub performance_prediction: f64,
}

/// Quantum enhancement details
#[derive(Debug, Clone)]
pub struct QuantumEnhancements {
    /// Level of quantum enhancement applied (0.0 to 1.0)
    pub enhancement_level: f32,
    /// Quantum advantage achieved
    pub quantum_advantage: f64,
    /// Whether quantum coherence was maintained
    pub coherence_maintained: bool,
}

/// Emotional context during query processing
#[derive(Debug, Clone)]
pub struct EmotionalContext {
    /// Current emotional state of the system
    pub emotional_state: crate::consciousness::EmotionalState,
    /// Emotional influence on processing
    pub emotional_influence: f64,
    /// Empathy level during processing
    pub empathy_level: f64,
}

/// Dream processing discoveries
#[derive(Debug, Clone)]
pub struct DreamDiscoveries {
    /// Number of insights generated
    pub insights_generated: u32,
    /// Number of patterns discovered
    pub pattern_discoveries: u32,
    /// Creative suggestions from dream processing
    pub creative_suggestions: u32,
    /// Overall quality of dream processing
    pub dream_quality: f64,
}

/// Symbolic match from SPARQL/reasoning
#[derive(Debug, Clone)]
pub struct SymbolicMatch {
    pub resource_uri: String,
    pub properties: HashMap<String, String>,
    pub reasoning_path: Vec<String>,
    pub certainty: f32,
}

/// Vector match from similarity search
#[derive(Debug, Clone)]
pub struct VectorMatch {
    pub resource_uri: String,
    pub similarity_score: f32,
    pub embedding_source: String,
    pub content_snippet: Option<String>,
}

/// Explanation for hybrid results
#[derive(Debug, Clone)]
pub struct Explanation {
    pub reasoning_steps: Vec<ReasoningStep>,
    pub similarity_factors: Vec<SimilarityFactor>,
    pub confidence_breakdown: ConfidenceBreakdown,
}

/// Individual reasoning step
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_type: ReasoningStepType,
    pub premise: String,
    pub conclusion: String,
    pub rule_applied: Option<String>,
    pub confidence: f32,
}

/// Types of reasoning steps
#[derive(Debug, Clone)]
pub enum ReasoningStepType {
    Deduction,
    Induction,
    Abduction,
    Similarity,
    VectorAlignment,
    RuleApplication,
}

/// Temporal constraints for time-aware reasoning
#[derive(Debug, Clone)]
pub struct TemporalConstraints {
    pub time_window: TimeWindow,
    pub temporal_relations: Vec<TemporalRelation>,
    pub causality_requirements: Vec<CausalityConstraint>,
    pub prediction_targets: Vec<String>,
}

/// Time window specification
#[derive(Debug, Clone)]
pub struct TimeWindow {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
    pub duration: Option<Duration>,
    pub granularity: TemporalGranularity,
}

/// Temporal granularity levels
#[derive(Debug, Clone)]
pub enum TemporalGranularity {
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

/// Temporal relation types
#[derive(Debug, Clone)]
pub struct TemporalRelation {
    pub relation_type: TemporalRelationType,
    pub entity_a: String,
    pub entity_b: String,
    pub confidence: f32,
}

/// Types of temporal relations
#[derive(Debug, Clone)]
pub enum TemporalRelationType {
    Before,
    After,
    During,
    Overlaps,
    Meets,
    StartedBy,
    FinishedBy,
    Equals,
}

/// Causality constraint
#[derive(Debug, Clone)]
pub struct CausalityConstraint {
    pub cause: String,
    pub effect: String,
    pub min_delay: Duration,
    pub max_delay: Duration,
    pub strength: f32,
}

/// Query performance metrics for learning
#[derive(Debug, Clone)]
pub struct QueryPerformance {
    pub query_id: String,
    pub execution_time: Duration,
    pub accuracy_score: f32,
    pub resource_usage: ResourceUsage,
    pub timestamp: DateTime<Utc>,
    pub optimization_strategy: String,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_utilization: f32,
    pub memory_usage: u64,
    pub io_operations: u64,
    pub network_requests: u32,
}

/// Learning modes for adaptive optimization
#[derive(Debug, Clone)]
pub enum LearningMode {
    Passive,    // Learn from observations only
    Active,     // Actively experiment with strategies
    Hybrid,     // Combine passive and active learning
    Transfer,   // Transfer learning from similar queries
}

/// Similarity contributing factors
#[derive(Debug, Clone)]
pub struct SimilarityFactor {
    pub factor_type: String,
    pub contribution: f32,
    pub description: String,
}

/// Confidence score breakdown
#[derive(Debug, Clone)]
pub struct ConfidenceBreakdown {
    pub symbolic_confidence: f32,
    pub vector_confidence: f32,
    pub integration_confidence: f32,
    pub overall_confidence: f32,
}

/// Configuration for the neural-symbolic bridge with consciousness integration
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub max_reasoning_depth: u32,
    pub similarity_threshold: f32,
    pub confidence_threshold: f32,
    pub explanation_detail_level: ExplanationLevel,
    pub enable_cross_modal: bool,
    pub enable_temporal_reasoning: bool,
    pub enable_uncertainty_handling: bool,
    /// Enable consciousness-inspired processing enhancements
    pub enable_consciousness_integration: bool,
    /// Weight for consciousness insights in final scoring (0.0 to 1.0)
    pub consciousness_weight: f32,
    /// Threshold for enabling quantum enhancement (complexity > threshold)
    pub quantum_enhancement_threshold: f32,
    /// Weight for emotional context in similarity scoring (0.0 to 1.0)
    pub emotional_context_weight: f32,
    /// Complexity threshold for triggering dream processing (0.0 to 1.0)
    pub dream_processing_complexity_threshold: f32,
}

/// Explanation detail levels
#[derive(Debug, Clone, Copy)]
pub enum ExplanationLevel {
    Minimal,
    Standard,
    Detailed,
    Comprehensive,
}

/// Performance metrics for the bridge
#[derive(Debug, Default)]
pub struct BridgeMetrics {
    pub total_queries: u64,
    pub successful_integrations: u64,
    pub failed_integrations: u64,
    pub average_processing_time: std::time::Duration,
    pub symbolic_accuracy: f32,
    pub vector_accuracy: f32,
    pub hybrid_accuracy: f32,
}

/// Temporal entity extracted from query
#[derive(Debug, Clone)]
pub struct TemporalEntity {
    pub entity_name: String,
    pub entity_type: String,
    pub temporal_reference: String,
    pub confidence: f32,
}

/// Causal analysis result
#[derive(Debug, Clone)]
pub struct CausalAnalysis {
    pub causal_chains: Vec<CausalChain>,
    pub strength_matrix: HashMap<String, HashMap<String, f32>>,
    pub confidence_score: f32,
}

/// Causal chain representation
#[derive(Debug, Clone)]
pub struct CausalChain {
    pub cause_event: String,
    pub effect_event: String,
    pub intermediate_steps: Vec<String>,
    pub total_strength: f32,
    pub temporal_delay: Duration,
}

/// Hybrid match combining symbolic and vector results
#[derive(Debug, Clone)]
pub struct HybridMatch {
    pub symbolic_match: Option<SymbolicMatch>,
    pub vector_match: Option<VectorMatch>,
    pub combined_score: f32,
}

/// Query features for optimization
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    pub complexity_score: f32,
    pub entity_count: usize,
    pub temporal_aspects: Vec<String>,
    pub semantic_categories: Vec<String>,
    pub linguistic_patterns: Vec<String>,
}

/// Performance analysis result
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub optimal_strategies: Vec<String>,
    pub performance_trends: HashMap<String, f32>,
    pub bottleneck_identification: Vec<String>,
    pub improvement_opportunities: Vec<String>,
}

/// Optimization strategy
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub strategy_name: String,
    pub priority_weights: HashMap<String, f32>,
    pub execution_order: Vec<String>,
    pub resource_allocation: HashMap<String, f32>,
    pub expected_improvement: f32,
}

/// Adaptive weights for query components
#[derive(Debug, Clone)]
pub struct AdaptiveWeights {
    pub symbolic_weight: f32,
    pub vector_weight: f32,
    pub temporal_weight: f32,
    pub similarity_threshold: f32,
    pub confidence_boost: f32,
}

/// Completion task for knowledge completion
#[derive(Debug, Clone)]
pub struct CompletionTask {
    pub triple_pattern: String,
    pub missing_component: String,
    pub context_requirements: Vec<String>,
}

/// Completion candidate
#[derive(Debug, Clone)]
pub struct CompletionCandidate {
    pub candidate_value: String,
    pub confidence: f32,
    pub reasoning_support: Vec<String>,
}

impl CompletionCandidate {
    pub fn to_symbolic_match(&self) -> SymbolicMatch {
        let mut properties = HashMap::new();
        properties.insert("completion".to_string(), self.candidate_value.clone());
        properties.insert("confidence".to_string(), self.confidence.to_string());

        SymbolicMatch {
            resource_uri: format!("completion:{}", self.candidate_value),
            properties,
            reasoning_path: self.reasoning_support.clone(),
            certainty: self.confidence,
        }
    }

    pub fn to_vector_match(&self) -> VectorMatch {
        VectorMatch {
            resource_uri: format!("completion:{}", self.candidate_value),
            similarity_score: self.confidence,
            embedding_source: "knowledge_completion".to_string(),
            content_snippet: Some(self.candidate_value.clone()),
        }
    }
}

/// Related concept for hierarchical reasoning
#[derive(Debug, Clone)]
pub struct RelatedConcept {
    pub term: String,
    pub relevance_score: f32,
    pub relation_type: String,
}

/// Query pattern for SPARQL analysis
#[derive(Debug, Clone)]
pub struct QueryPattern {
    pub triple_pattern: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_type: String,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            max_reasoning_depth: 5,
            similarity_threshold: 0.7,
            confidence_threshold: 0.6,
            explanation_detail_level: ExplanationLevel::Standard,
            enable_cross_modal: true,
            enable_temporal_reasoning: false,
            enable_uncertainty_handling: true,
            enable_consciousness_integration: true,
            consciousness_weight: 0.3,
            quantum_enhancement_threshold: 0.8,
            emotional_context_weight: 0.2,
            dream_processing_complexity_threshold: 0.9,
        }
    }
}