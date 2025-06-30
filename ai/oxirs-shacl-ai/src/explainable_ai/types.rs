//! Core types and data structures for explainable AI

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use oxirs_core::{
    model::{NamedNode, Term, Triple},
    Store,
};
use oxirs_shacl::{
    constraints::*, Shape, ShapeId, ValidationConfig, ValidationReport, ValidationViolation,
};
use crate::{Result, ShaclAiError};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationDepth {
    Brief,
    Standard,
    Detailed,
    Comprehensive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    Validation,
    PatternRecognition,
    Adaptation,
    QuantumProcessing,
    Learning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurrogateModel {
    DecisionTree,
    LinearRegression,
    SimpleNN,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineStrategy {
    Zero,
    Mean,
    Random,
    Custom(Vec<f64>),
}

/// Core data structures for explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationData {
    pub input_type: String,
    pub input_data: serde_json::Value,
    pub context: serde_json::Value,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawExplanation {
    pub explanation_id: Uuid,
    pub explanation_type: String,
    pub source_component: String,
    pub data: serde_json::Value,
    pub confidence: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedExplanation {
    pub explanation_id: Uuid,
    pub explanation_type: String,
    pub source_component: String,
    pub natural_language: Option<String>,
    pub structured_data: serde_json::Value,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
    pub limitations: Vec<String>,
    pub related_explanations: Vec<Uuid>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedExplanation {
    pub explanation: ProcessedExplanation,
    pub access_count: usize,
    pub last_accessed: SystemTime,
    pub cache_key: String,
}

/// Feature importance and analysis structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceAnalysis {
    pub features: Vec<FeatureImportance>,
    pub global_importance: HashMap<String, f64>,
    pub local_importance: HashMap<String, f64>,
    pub feature_interactions: Vec<FeatureInteraction>,
    pub analysis_timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub feature_name: String,
    pub importance_score: f64,
    pub confidence_interval: (f64, f64),
    pub rank: usize,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInteraction {
    pub feature_1: String,
    pub feature_2: String,
    pub interaction_strength: f64,
    pub interaction_type: String,
}

/// Attention analysis structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionAnalysis {
    pub attention_patterns: Vec<AttentionPattern>,
    pub head_importance: Vec<f64>,
    pub layer_importance: Vec<f64>,
    pub token_attention: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPattern {
    pub pattern_id: Uuid,
    pub pattern_type: String,
    pub attention_weights: Vec<Vec<f64>>,
    pub tokens: Vec<String>,
    pub strength: f64,
}

/// Decision path analysis structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPathAnalysis {
    pub decision_tree: DecisionTree,
    pub critical_paths: Vec<DecisionPath>,
    pub path_statistics: PathStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub tree_id: Uuid,
    pub root_node: DecisionNode,
    pub depth: usize,
    pub total_nodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    pub node_id: Uuid,
    pub condition: String,
    pub decision_value: f64,
    pub children: Vec<DecisionNode>,
    pub leaf_explanation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPath {
    pub path_id: Uuid,
    pub conditions: Vec<String>,
    pub outcome: String,
    pub probability: f64,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathStatistics {
    pub average_depth: f64,
    pub max_depth: usize,
    pub most_frequent_conditions: Vec<String>,
    pub decision_distribution: HashMap<String, usize>,
}

/// Model behavior analysis structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBehaviorAnalysis {
    pub prediction_confidence: ConfidenceAnalysis,
    pub uncertainty_analysis: UncertaintyAnalysis,
    pub sensitivity_analysis: SensitivityAnalysis,
    pub robustness_analysis: RobustnessAnalysis,
    pub calibration_analysis: CalibrationAnalysis,
    pub bias_analysis: BiasAnalysis,
    pub performance_interpretability: PerformanceInterpretability,
    pub reliability_analysis: ReliabilityAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceAnalysis {
    pub confidence_score: f64,
    pub confidence_bounds: (f64, f64),
    pub uncertainty_sources: Vec<String>,
    pub calibration_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyAnalysis {
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub total_uncertainty: f64,
    pub uncertainty_breakdown: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    pub input_sensitivities: HashMap<String, f64>,
    pub parameter_sensitivities: HashMap<String, f64>,
    pub perturbation_analysis: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessAnalysis {
    pub adversarial_robustness: f64,
    pub noise_robustness: f64,
    pub distribution_shift_robustness: f64,
    pub robustness_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAnalysis {
    pub calibration_error: f64,
    pub reliability_diagram: Vec<(f64, f64)>,
    pub brier_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasAnalysis {
    pub detected_biases: Vec<String>,
    pub fairness_metrics: HashMap<String, f64>,
    pub bias_mitigation_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInterpretability {
    pub performance_drivers: Vec<String>,
    pub bottleneck_analysis: Vec<String>,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityAnalysis {
    pub reliability_score: f64,
    pub failure_modes: Vec<String>,
    pub redundancy_analysis: String,
}

/// Audit trail and compliance structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    pub audit_id: Uuid,
    pub decision_id: String,
    pub decision_type: DecisionType,
    pub decision_timeline: Vec<DecisionStep>,
    pub input_data_fingerprint: String,
    pub processing_steps: Vec<ProcessingStep>,
    pub model_versions: HashMap<String, String>,
    pub configuration_snapshot: HashMap<String, String>,
    pub output_verification: Option<OutputVerification>,
    pub compliance_markers: Vec<ComplianceMarker>,
    pub created_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStep {
    pub step_id: Uuid,
    pub step_type: String,
    pub timestamp: SystemTime,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStep {
    pub step_name: String,
    pub duration: Duration,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputVerification {
    pub verification_method: String,
    pub verification_result: bool,
    pub verification_details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMarker {
    pub regulation: String,
    pub compliance_status: String,
    pub verification_timestamp: SystemTime,
}

#[derive(Debug)]
pub struct DecisionTracker {
    pub decision_id: String,
    pub timeline: Vec<DecisionStep>,
    pub input_fingerprint: String,
    pub processing_steps: Vec<ProcessingStep>,
    pub model_versions: HashMap<String, String>,
    pub configuration_snapshot: HashMap<String, String>,
    pub output_verification: Option<OutputVerification>,
    pub compliance_markers: Vec<ComplianceMarker>,
    pub created_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityRecommendation {
    pub recommendation_id: Uuid,
    pub category: String,
    pub priority: String,
    pub description: String,
    pub implementation_effort: String,
}

/// Attention visualization structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionWeights {
    pub weights: Vec<Vec<Vec<f64>>>, // [layer][head][token_i][token_j]
    pub tokens: Vec<String>,
    pub num_layers: usize,
    pub num_heads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionVisualization {
    pub head_view: Vec<HeadVisualization>,
    pub attention_rollout: Option<Vec<Vec<f64>>>,
    pub attention_flow: Option<AttentionFlow>,
    pub summary_statistics: AttentionStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadVisualization {
    pub layer: usize,
    pub head: usize,
    pub attention_matrix: Vec<Vec<f64>>,
    pub tokens: Vec<String>,
    pub head_interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlow {
    pub token_flows: HashMap<String, Vec<FlowConnection>>,
    pub aggregated_flows: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConnection {
    pub from_token: String,
    pub to_token: String,
    pub flow_strength: f64,
    pub layer: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionStatistics {
    pub entropy: Vec<f64>,
    pub sparsity: Vec<f64>,
    pub locality: Vec<f64>,
    pub head_diversity: f64,
}