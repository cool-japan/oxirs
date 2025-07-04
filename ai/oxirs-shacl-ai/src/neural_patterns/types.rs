//! Type definitions for neural pattern recognition
//!
//! This module contains all shared data structures, enums, and type aliases
//! used across the neural pattern recognition modules.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::ml::{EdgeFeatures, GlobalFeatures, NodeFeatures};

/// Configuration for advanced pattern correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysisConfig {
    /// Minimum correlation threshold for significance
    pub min_correlation_threshold: f64,
    /// Maximum correlation depth to analyze
    pub max_correlation_depth: usize,
    /// Enable cross-modal correlation analysis
    pub enable_cross_modal_analysis: bool,
    /// Enable temporal correlation analysis
    pub enable_temporal_correlation: bool,
    /// Enable hierarchical pattern discovery
    pub enable_hierarchical_discovery: bool,
    /// Enable causal inference
    pub enable_causal_inference: bool,
    /// Number of correlation clusters to discover
    pub num_correlation_clusters: usize,
    /// Enable advanced similarity metrics
    pub enable_advanced_similarity: bool,
    /// Correlation confidence threshold
    pub correlation_confidence_threshold: f64,
}

impl Default for CorrelationAnalysisConfig {
    fn default() -> Self {
        Self {
            min_correlation_threshold: 0.3,
            max_correlation_depth: 5,
            enable_cross_modal_analysis: true,
            enable_temporal_correlation: true,
            enable_hierarchical_discovery: true,
            enable_causal_inference: true,
            num_correlation_clusters: 10,
            enable_advanced_similarity: true,
            correlation_confidence_threshold: 0.8,
        }
    }
}

/// Types of pattern correlations that can be discovered
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum CorrelationType {
    /// Structural similarity correlations
    Structural,
    /// Semantic similarity correlations
    Semantic,
    /// Temporal co-occurrence correlations
    Temporal,
    /// Causal relationships between patterns
    Causal,
    /// Hierarchical parent-child relationships
    Hierarchical,
    /// Functional dependency correlations
    Functional,
    /// Contextual similarity correlations
    Contextual,
    /// Cross-domain correlations
    CrossDomain,
}

/// Pattern relationship graph representing complex pattern interactions
#[derive(Debug, Clone)]
pub struct PatternRelationshipGraph {
    /// Nodes representing patterns
    pub pattern_nodes: HashMap<String, PatternNode>,
    /// Edges representing relationships
    pub relationship_edges: Vec<RelationshipEdge>,
    /// Graph statistics
    pub graph_stats: GraphStatistics,
}

/// Node in the pattern relationship graph
#[derive(Debug, Clone)]
pub struct PatternNode {
    pub pattern_id: String,
    pub node_features: NodeFeatures,
    pub centrality_scores: CentralityScores,
    pub cluster_membership: Vec<usize>,
}

/// Edge representing a relationship between patterns
#[derive(Debug, Clone)]
pub struct RelationshipEdge {
    pub source_pattern: String,
    pub target_pattern: String,
    pub relationship_type: CorrelationType,
    pub correlation_strength: f64,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
    pub temporal_dynamics: Option<TemporalDynamics>,
}

/// Centrality scores for pattern importance
#[derive(Debug, Clone)]
pub struct CentralityScores {
    pub degree_centrality: f64,
    pub betweenness_centrality: f64,
    pub closeness_centrality: f64,
    pub eigenvector_centrality: f64,
    pub pagerank_score: f64,
}

/// Pattern hierarchy representing discovered structural relationships
#[derive(Debug, Clone)]
pub struct PatternHierarchy {
    pub hierarchy_id: String,
    pub root_patterns: Vec<String>,
    pub hierarchy_levels: Vec<HierarchyLevel>,
    pub hierarchy_metrics: HierarchyMetrics,
}

/// Level in the pattern hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyLevel {
    pub level: usize,
    pub patterns: Vec<String>,
    pub level_coherence: f64,
    pub inter_level_connections: Vec<(String, String, f64)>,
}

/// Metrics for evaluating hierarchy quality
#[derive(Debug, Clone)]
pub struct HierarchyMetrics {
    pub hierarchy_depth: usize,
    pub branching_factor: f64,
    pub coherence_score: f64,
    pub coverage_percentage: f64,
    pub stability_measure: f64,
}

/// Temporal dynamics of pattern relationships
#[derive(Debug, Clone)]
pub struct TemporalDynamics {
    pub relationship_strength_over_time: Vec<(u64, f64)>,
    pub emergence_timestamp: u64,
    pub decay_rate: f64,
    pub periodicity: Option<f64>,
    pub trend_direction: TrendDirection,
}

/// Direction of temporal trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}

/// Graph statistics for pattern relationships
#[derive(Debug, Clone, Default)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f64,
    pub clustering_coefficient: f64,
    pub graph_density: f64,
    pub connected_components: usize,
}

/// Statistics for correlation analysis
#[derive(Debug, Clone, Default)]
pub struct CorrelationAnalysisStats {
    pub correlations_analyzed: usize,
    pub significant_correlations_found: usize,
    pub analysis_execution_time: Duration,
    pub average_correlation_strength: f64,
    pub correlation_type_distribution: HashMap<CorrelationType, usize>,
    pub hierarchies_discovered: usize,
    pub clusters_formed: usize,
    pub causal_relationships_identified: usize,
}

/// Result of correlation analysis
#[derive(Debug)]
pub struct CorrelationAnalysisResult {
    pub discovered_correlations: Vec<PatternCorrelation>,
    pub correlation_clusters: Vec<CorrelationCluster>,
    pub pattern_hierarchies: Vec<PatternHierarchy>,
    pub attention_insights: AttentionAnalysisResult,
    pub causal_relationships: Vec<CausalRelationship>,
    pub analysis_metadata: CorrelationAnalysisMetadata,
}

/// Individual pattern correlation
#[derive(Debug, Clone)]
pub struct PatternCorrelation {
    pub pattern1_id: String,
    pub pattern2_id: String,
    pub correlation_type: CorrelationType,
    pub correlation_coefficient: f64,
    pub statistical_significance: f64,
    pub confidence_interval: (f64, f64),
    pub supporting_features: Vec<String>,
    pub temporal_context: Option<TemporalContext>,
}

/// Cluster of correlated patterns
#[derive(Debug, Clone)]
pub struct CorrelationCluster {
    pub cluster_id: String,
    pub member_patterns: Vec<String>,
    pub cluster_centroid: Vec<f64>,
    pub intra_cluster_correlation: f64,
    pub cluster_coherence: f64,
    pub representative_pattern: String,
}

/// Causal relationship between patterns
#[derive(Debug, Clone)]
pub struct CausalRelationship {
    pub cause_pattern: String,
    pub effect_pattern: String,
    pub causal_strength: f64,
    pub confidence_level: f64,
    pub evidence_type: CausalEvidenceType,
    pub temporal_lag: Option<Duration>,
}

/// Types of causal evidence
#[derive(Debug, Clone)]
pub enum CausalEvidenceType {
    Temporal,
    Interventional,
    Observational,
    Counterfactual,
}

/// Temporal context for correlations
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub time_window: Duration,
    pub temporal_resolution: Duration,
    pub seasonality_detected: bool,
    pub trend_strength: f64,
}

/// Result of attention analysis
#[derive(Debug)]
pub struct AttentionAnalysisResult {
    pub attention_weights: HashMap<String, Array2<f64>>,
    pub attention_patterns: Vec<AttentionPattern>,
    pub cross_pattern_influences: Vec<CrossPatternInfluence>,
}

/// Discovered attention pattern
#[derive(Debug)]
pub struct AttentionPattern {
    pub pattern_id: String,
    pub attention_distribution: Array1<f64>,
    pub focus_regions: Vec<AttentionFocus>,
}

/// Region of focused attention
#[derive(Debug)]
pub struct AttentionFocus {
    pub focus_type: AttentionFocusType,
    pub intensity: f64,
    pub spatial_extent: Option<(usize, usize)>,
}

/// Types of attention focus
#[derive(Debug)]
pub enum AttentionFocusType {
    Local,
    Global,
    Contextual,
    Relational,
}

/// Cross-pattern influence detected by attention
#[derive(Debug)]
pub struct CrossPatternInfluence {
    pub source_pattern: String,
    pub target_pattern: String,
    pub influence_strength: f64,
    pub influence_type: InfluenceType,
}

/// Types of cross-pattern influence
#[derive(Debug)]
pub enum InfluenceType {
    Excitatory,
    Inhibitory,
    Modulatory,
    Competitive,
}

/// Metadata about correlation analysis
#[derive(Debug)]
pub struct CorrelationAnalysisMetadata {
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
    pub analysis_duration: Duration,
    pub patterns_analyzed: usize,
    pub correlation_methods_used: Vec<String>,
    pub confidence_threshold: f64,
}

impl Default for CorrelationAnalysisMetadata {
    fn default() -> Self {
        Self {
            analysis_timestamp: chrono::Utc::now(),
            analysis_duration: Duration::default(),
            patterns_analyzed: 0,
            correlation_methods_used: Vec::new(),
            confidence_threshold: 0.8,
        }
    }
}

/// Configuration for neural pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPatternConfig {
    pub embedding_dim: usize,
    pub attention_heads: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub dropout_rate: f64,
    pub regularization_strength: f64,
    pub enable_batch_norm: bool,
    pub enable_residual_connections: bool,
    pub activation_function: ActivationFunction,
}

impl Default for NeuralPatternConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 256,
            attention_heads: 8,
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            dropout_rate: 0.1,
            regularization_strength: 0.01,
            enable_batch_norm: true,
            enable_residual_connections: true,
            activation_function: ActivationFunction::ReLU,
        }
    }
}

/// Activation functions for neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU,
    ELU,
    Swish,
    GELU,
    Tanh,
    Sigmoid,
}

/// Learning rate schedule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    Constant,
    Linear,
    Cosine,
    Exponential,
    StepLR,
    ReduceOnPlateau,
}

/// Core neural pattern representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPattern {
    /// Unique identifier for the pattern
    pub id: String,
    /// Pattern features as a vector
    pub features: Vec<f64>,
    /// Pattern type classification
    pub pattern_type: PatternType,
    /// Confidence score for the pattern
    pub confidence: f64,
    /// Semantic embedding of the pattern
    pub embedding: Vec<f64>,
    /// Associated SHACL constraints
    pub constraints: Vec<String>,
    /// Pattern metadata
    pub metadata: HashMap<String, String>,
}

/// Types of neural patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternType {
    /// Structural patterns in RDF graphs
    Structural,
    /// Semantic patterns based on meaning
    Semantic,
    /// Temporal patterns over time
    Temporal,
    /// Spatial patterns in data
    Spatial,
    /// Hierarchical patterns
    Hierarchical,
    /// Causal patterns
    Causal,
    /// Constraint patterns for SHACL
    Constraint,
    /// Custom pattern type
    Custom(String),
}

/// Attention configuration for neural patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention dimension
    pub attention_dim: usize,
    /// Dropout rate for attention
    pub dropout_rate: f64,
    /// Enable scaled dot-product attention
    pub scaled_attention: bool,
}

/// Analysis quality metrics for pattern evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisQualityMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confidence: f64,
}

/// Attention flow dynamics between pattern layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlowDynamics {
    pub flow_vectors: Vec<Array1<f64>>,
    pub temporal_evolution: Vec<f64>,
    pub intensity_patterns: HashMap<String, f64>,
}

/// Attention hotspot in pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHotspot {
    pub hotspot_type: HotspotType,
    pub position: (f64, f64),
    pub intensity: f64,
    pub influence_radius: f64,
}

/// Types of attention hotspots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotspotType {
    HighActivity,
    PatternBoundary,
    FeatureInteraction,
    AnomalyDetection,
}

/// Attention insights from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionInsights {
    pub key_patterns: Vec<String>,
    pub attention_distribution: HashMap<String, f64>,
    pub temporal_trends: Vec<f64>,
    pub recommendations: Vec<String>,
}

/// Attention pathway in neural processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPathway {
    pub source_layer: String,
    pub target_layer: String,
    pub pathway_strength: f64,
    pub information_flow: Vec<f64>,
}

/// Causal mechanism discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalMechanism {
    pub mechanism_type: MechanismType,
    pub cause_variables: Vec<String>,
    pub effect_variables: Vec<String>,
    pub strength: f64,
    pub confidence: f64,
}

/// Types of causal mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MechanismType {
    DirectCausation,
    IndirectCausation,
    ConditionalCausation,
    BidirectionalCausation,
}

/// Cluster characteristics in pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCharacteristics {
    pub cluster_id: String,
    pub size: usize,
    pub density: f64,
    pub cohesion: f64,
    pub separation: f64,
    pub representative_patterns: Vec<String>,
}

/// Evidence for correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEvidence {
    pub evidence_type: String,
    pub strength: f64,
    pub confidence: f64,
    pub supporting_data: Vec<String>,
}

/// Cross-scale interaction patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossScaleInteraction {
    pub interaction_type: InteractionType,
    pub scales: Vec<String>,
    pub interaction_strength: f64,
    pub bidirectional: bool,
}

/// Types of interactions between scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Hierarchical,
    Emergent,
    Regulatory,
    Competitive,
}

/// Emergence pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencePattern {
    pub emergence_level: usize,
    pub source_patterns: Vec<String>,
    pub emergent_properties: Vec<String>,
    pub emergence_strength: f64,
}

/// Learned constraint pattern from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedConstraintPattern {
    pub constraint_type: String,
    pub pattern_expression: String,
    pub confidence: f64,
    pub support_count: usize,
    pub applicability_score: f64,
}

/// Multi-scale finding from hierarchical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleFinding {
    pub finding_id: String,
    pub scales_involved: Vec<String>,
    pub finding_description: String,
    pub significance: f64,
    pub cross_scale_validity: f64,
}

/// Temporal behavior patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBehavior {
    pub behavior_type: String,
    pub time_series: Vec<f64>,
    pub periodicity: Option<Duration>,
    pub trend_direction: String,
    pub volatility: f64,
}
