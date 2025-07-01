//! Type definitions for explainable AI system
//!
//! This module contains all the data structures and enums used throughout
//! the explainable AI system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Configuration for explainable AI system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainableAIConfig {
    pub enable_natural_language: bool,
    pub cache_explanations: bool,
    pub explanation_depth: ExplanationDepth,
    pub generate_visualizations: bool,
    pub track_all_decisions: bool,
    pub compliance_mode: bool,
}

impl Default for ExplainableAIConfig {
    fn default() -> Self {
        Self {
            enable_natural_language: true,
            cache_explanations: true,
            explanation_depth: ExplanationDepth::Detailed,
            generate_visualizations: true,
            track_all_decisions: false,
            compliance_mode: false,
        }
    }
}

/// Depth of explanation to generate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationDepth {
    Brief,
    Standard,
    Detailed,
    Comprehensive,
}

/// Input data for explanation generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationData {
    pub input_type: String,
    pub input_data: serde_json::Value,
    pub context: serde_json::Value,
    pub timestamp: SystemTime,
}

/// Raw explanation before processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawExplanation {
    pub explanation_id: Uuid,
    pub explanation_type: String,
    pub technical_details: serde_json::Value,
    pub confidence_score: f64,
    pub generation_time: Duration,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Processed and formatted explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedExplanation {
    pub explanation_id: Uuid,
    pub natural_language: Option<String>,
    pub technical_summary: String,
    pub visual_elements: Vec<VisualizationElement>,
    pub confidence_score: f64,
    pub supporting_evidence: Vec<EvidenceItem>,
    pub alternative_explanations: Vec<AlternativeExplanation>,
    pub compliance_info: Option<ComplianceInfo>,
}

/// Types of decisions being tracked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    PatternRecognition,
    ValidationRule,
    AdaptationChoice,
    QuantumPattern,
    NeuralActivation,
    ReasoningStep,
}

/// Context for a decision being made
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    pub decision_id: Uuid,
    pub decision_type: DecisionType,
    pub input_data: serde_json::Value,
    pub output_data: serde_json::Value,
    pub confidence: f64,
    pub timestamp: SystemTime,
    pub reasoning_chain: Vec<ReasoningStep>,
}

/// A single step in reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_id: Uuid,
    pub description: String,
    pub input_state: serde_json::Value,
    pub output_state: serde_json::Value,
    pub confidence: f64,
    pub applied_rules: Vec<String>,
}

/// Cached explanation for performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedExplanation {
    pub explanation: ProcessedExplanation,
    pub cache_timestamp: SystemTime,
    pub access_count: u64,
    pub last_accessed: SystemTime,
}

/// Feature importance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceAnalysis {
    pub feature_scores: HashMap<String, f64>,
    pub top_features: Vec<(String, f64)>,
    pub analysis_method: String,
    pub confidence: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Visualization element for explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationElement {
    pub element_type: String,
    pub data: serde_json::Value,
    pub rendering_hints: HashMap<String, String>,
}

/// Supporting evidence for explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub evidence_type: String,
    pub description: String,
    pub strength: f64,
    pub source: String,
    pub data: serde_json::Value,
}

/// Alternative explanation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeExplanation {
    pub explanation: String,
    pub probability: f64,
    pub supporting_evidence: Vec<EvidenceItem>,
}

/// Compliance information for regulatory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceInfo {
    pub regulation_type: String,
    pub compliance_level: String,
    pub audit_trail: Vec<String>,
    pub verification_timestamp: SystemTime,
}

/// Validation-specific explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationExplanation {
    pub decision_id: Uuid,
    pub validation_outcome: bool,
    pub natural_language_explanation: String,
    pub decision_tree: DecisionTree,
    pub key_factors: Vec<KeyFactor>,
    pub confidence_score: f64,
    pub reasoning_steps: Vec<String>,
    pub supporting_evidence: Vec<String>,
    pub alternative_explanations: Vec<String>,
    pub timestamp: SystemTime,
}

/// Pattern recognition explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExplanation {
    pub explanation_id: Uuid,
    pub recognized_patterns: Vec<NeuralPattern>,
    pub recognition_confidence: f64,
    pub feature_importance: FeatureImportanceAnalysis,
    pub visual_explanations: Vec<VisualExplanation>,
    pub pattern_similarities: Vec<PatternSimilarity>,
    pub decision_boundary_analysis: DecisionBoundaryAnalysis,
    pub counterfactual_examples: Vec<CounterfactualExample>,
    pub timestamp: SystemTime,
}

/// Quantum pattern explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumExplanation {
    pub explanation_id: Uuid,
    pub quantum_states: Vec<String>,
    pub superposition_analysis: SuperpositionAnalysis,
    pub entanglement_effects: EntanglementAnalysis,
    pub measurement_impact: MeasurementImpact,
    pub quantum_advantage_explanation: String,
    pub classical_comparison: ClassicalComparison,
    pub timestamp: SystemTime,
}

/// Adaptation logic explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationExplanation {
    pub explanation_id: Uuid,
    pub strategy_chosen: String,
    pub strategy_rationale: String,
    pub performance_triggers: Vec<PerformanceTrigger>,
    pub alternative_strategies: Vec<AlternativeStrategy>,
    pub expected_outcomes: Vec<PredictedOutcome>,
    pub risk_assessment: RiskAssessment,
    pub rollback_plan: RollbackPlan,
    pub timestamp: SystemTime,
}

/// Comprehensive interpretability report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityReport {
    pub report_id: Uuid,
    pub system_state: SystemState,
    pub model_behavior_analysis: ModelBehaviorAnalysis,
    pub global_feature_importance: GlobalFeatureImportance,
    pub system_decision_flow: SystemDecisionFlow,
    pub bias_and_fairness_analysis: BiasAnalysis,
    pub performance_interpretability: PerformanceInterpretability,
    pub reliability_analysis: ReliabilityAnalysis,
    pub natural_language_summary: String,
    pub recommendations: Vec<InterpretabilityRecommendation>,
    pub timestamp: SystemTime,
}

/// Context for validation explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationContext {
    pub shape_id: String,
    pub target_node: String,
    pub constraint_type: String,
}

/// Context for pattern recognition explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognitionContext {
    pub input_size: usize,
    pub model_version: String,
    pub confidence_threshold: f64,
}

/// Context for quantum pattern explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPatternContext {
    pub quantum_backend: String,
    pub num_qubits: usize,
    pub coherence_time: Duration,
}

// Supporting types for explanations

/// Decision tree representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub root: DecisionNode,
    pub depth: usize,
    pub total_nodes: usize,
}

/// Decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    pub node_id: String,
    pub condition: String,
    pub outcome: Option<String>,
    pub children: Vec<DecisionNode>,
    pub confidence: f64,
}

/// Key factor in decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFactor {
    pub factor_name: String,
    pub importance_score: f64,
    pub description: String,
    pub impact_direction: String, // "positive", "negative", "neutral"
}

/// Visual explanation element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualExplanation {
    pub visualization_type: String,
    pub data: serde_json::Value,
    pub title: String,
    pub description: String,
}

/// Pattern similarity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSimilarity {
    pub pattern_id: String,
    pub similarity_score: f64,
    pub comparison_method: String,
}

/// Decision boundary analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBoundaryAnalysis {
    pub boundary_type: String,
    pub complexity_score: f64,
    pub key_features: Vec<String>,
    pub confidence_regions: Vec<ConfidenceRegion>,
}

/// Confidence region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceRegion {
    pub region_id: String,
    pub confidence_level: f64,
    pub bounds: HashMap<String, (f64, f64)>,
}

/// Counterfactual example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualExample {
    pub example_id: String,
    pub original_input: serde_json::Value,
    pub modified_input: serde_json::Value,
    pub outcome_change: String,
    pub minimal_changes: Vec<String>,
}

/// Superposition analysis for quantum explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperpositionAnalysis {
    pub superposition_states: Vec<String>,
    pub amplitudes: Vec<f64>,
    pub phase_information: Vec<f64>,
}

/// Entanglement analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementAnalysis {
    pub entangled_qubits: Vec<(usize, usize)>,
    pub entanglement_strength: Vec<f64>,
    pub correlation_effects: Vec<String>,
}

/// Measurement impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementImpact {
    pub pre_measurement_state: String,
    pub post_measurement_state: String,
    pub information_gain: f64,
    pub decoherence_time: Duration,
}

/// Classical comparison for quantum explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalComparison {
    pub classical_algorithm: String,
    pub quantum_advantage: f64,
    pub complexity_comparison: String,
    pub accuracy_comparison: f64,
}

/// Performance trigger for adaptations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrigger {
    pub trigger_type: String,
    pub threshold_value: f64,
    pub current_value: f64,
    pub trigger_time: SystemTime,
}

/// Alternative strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeStrategy {
    pub strategy_name: String,
    pub expected_performance: f64,
    pub implementation_cost: f64,
    pub risk_level: String,
}

/// Predicted outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedOutcome {
    pub outcome_type: String,
    pub probability: f64,
    pub confidence_interval: (f64, f64),
    pub time_horizon: Duration,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk_level: String,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_strategies: Vec<String>,
    pub acceptable_risk_threshold: f64,
}

/// Risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor_name: String,
    pub probability: f64,
    pub impact_severity: String,
    pub mitigation_available: bool,
}

/// Rollback plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    pub rollback_triggers: Vec<String>,
    pub rollback_steps: Vec<String>,
    pub estimated_rollback_time: Duration,
    pub data_preservation_strategy: String,
}

/// System state for interpretability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub state_id: String,
    pub timestamp: SystemTime,
    pub active_components: Vec<String>,
    pub performance_metrics: HashMap<String, f64>,
    pub resource_utilization: HashMap<String, f64>,
}

/// Model behavior analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBehaviorAnalysis {
    pub behavior_patterns: Vec<String>,
    pub consistency_score: f64,
    pub anomaly_detection: Vec<String>,
    pub stability_metrics: HashMap<String, f64>,
}

/// Global feature importance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalFeatureImportance {
    pub feature_rankings: Vec<(String, f64)>,
    pub importance_method: String,
    pub stability_across_samples: f64,
    pub interaction_effects: HashMap<String, f64>,
}

/// System decision flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemDecisionFlow {
    pub decision_path: Vec<String>,
    pub branching_points: Vec<String>,
    pub critical_decisions: Vec<String>,
    pub flow_efficiency: f64,
}

/// Bias analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasAnalysis {
    pub bias_metrics: HashMap<String, f64>,
    pub fairness_criteria: Vec<String>,
    pub protected_attributes: Vec<String>,
    pub bias_mitigation_suggestions: Vec<String>,
}

/// Performance interpretability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInterpretability {
    pub performance_drivers: Vec<String>,
    pub bottleneck_analysis: Vec<String>,
    pub optimization_opportunities: Vec<String>,
    pub trade_off_analysis: HashMap<String, f64>,
}

/// Reliability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityAnalysis {
    pub reliability_score: f64,
    pub failure_modes: Vec<String>,
    pub robustness_metrics: HashMap<String, f64>,
    pub error_propagation_analysis: Vec<String>,
}

/// Interpretability recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityRecommendation {
    pub recommendation_id: String,
    pub recommendation_type: String,
    pub description: String,
    pub priority_level: String,
    pub implementation_effort: String,
    pub expected_impact: f64,
}

// Import necessary types for references
use crate::neural_patterns::NeuralPattern;
