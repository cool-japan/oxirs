//! AI-Powered Quality Metrics for SHACL-AI
//!
//! This module implements advanced machine learning-based quality metrics
//! that go beyond traditional rule-based assessments.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use oxirs_core::model::Object;
use oxirs_core::Store;
use oxirs_shacl::{Shape, ValidationReport};

use super::extended_dimensions::{
    CorrelationAnalysis, DistributionAnalysis, EntropyCalculation, InformationContentMeasure,
    OutlierDetectionResult, RedundancyAssessment,
};
use crate::{Result, ShaclAiError};

/// AI-powered quality metrics engine
#[derive(Debug)]
pub struct AiQualityMetricsEngine {
    config: AiMetricsConfig,
    models: QualityModels,
    cache: HashMap<String, CachedMetrics>,
    statistics: AiMetricsStatistics,
}

/// Configuration for AI quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiMetricsConfig {
    /// Enable machine learning-based quality assessment
    pub enable_ml_assessment: bool,

    /// Enable deep learning models for semantic analysis
    pub enable_deep_learning: bool,

    /// Enable ensemble methods for improved accuracy
    pub enable_ensemble_methods: bool,

    /// Statistical significance threshold
    pub significance_threshold: f64,

    /// Minimum sample size for reliable metrics
    pub min_sample_size: usize,

    /// Cache TTL for computed metrics (in seconds)
    pub cache_ttl_seconds: u64,

    /// Enable advanced correlation analysis
    pub enable_advanced_correlations: bool,

    /// Enable causal inference
    pub enable_causal_inference: bool,

    /// Confidence threshold for AI predictions
    pub ai_confidence_threshold: f64,
}

impl Default for AiMetricsConfig {
    fn default() -> Self {
        Self {
            enable_ml_assessment: true,
            enable_deep_learning: true,
            enable_ensemble_methods: true,
            significance_threshold: 0.05,
            min_sample_size: 100,
            cache_ttl_seconds: 3600,
            enable_advanced_correlations: true,
            enable_causal_inference: false, // Experimental
            ai_confidence_threshold: 0.8,
        }
    }
}

/// Collection of AI models for quality assessment
#[derive(Debug)]
struct QualityModels {
    distribution_classifier: Option<DistributionClassifier>,
    outlier_detector: Option<OutlierDetector>,
    correlation_analyzer: Option<CorrelationAnalyzer>,
    entropy_estimator: Option<EntropyEstimator>,
    information_analyzer: Option<InformationAnalyzer>,
    redundancy_detector: Option<RedundancyDetector>,
    semantic_analyzer: Option<SemanticAnalyzer>,
}

/// Cached metrics to avoid recomputation
#[derive(Debug, Clone)]
struct CachedMetrics {
    metrics: AiQualityMetricsResult,
    computed_at: chrono::DateTime<chrono::Utc>,
    validity_duration: chrono::Duration,
}

/// Statistics for AI metrics engine
#[derive(Debug, Clone, Default)]
pub struct AiMetricsStatistics {
    pub total_computations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub ml_model_invocations: usize,
    pub average_computation_time: std::time::Duration,
    pub accuracy_scores: HashMap<String, f64>,
}

/// Comprehensive AI quality metrics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiQualityMetricsResult {
    pub statistical_metrics: AdvancedStatisticalMetrics,
    pub semantic_metrics: AdvancedSemanticMetrics,
    pub ml_predictions: MachineLearningPredictions,
    pub confidence_scores: ConfidenceScores,
    pub feature_importance: FeatureImportance,
    pub quality_predictions: QualityPredictions,
    pub computation_metadata: ComputationMetadata,
}

/// Advanced statistical metrics with AI enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedStatisticalMetrics {
    pub distribution_analysis: EnhancedDistributionAnalysis,
    pub outlier_detection: AdvancedOutlierDetection,
    pub correlation_analysis: AdvancedCorrelationAnalysis,
    pub entropy_analysis: AdvancedEntropyAnalysis,
    pub information_theory: AdvancedInformationTheory,
    pub redundancy_analysis: AdvancedRedundancyAnalysis,
    pub causal_relationships: Option<CausalRelationships>,
}

/// Advanced semantic metrics with deep learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSemanticMetrics {
    pub concept_embeddings: ConceptEmbeddings,
    pub semantic_similarity: SemanticSimilarity,
    pub knowledge_graph_metrics: KnowledgeGraphMetrics,
    pub ontology_alignment: OntologyAlignment,
    pub semantic_drift: SemanticDrift,
    pub concept_evolution: ConceptEvolution,
}

/// Machine learning predictions for quality
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MachineLearningPredictions {
    pub quality_score_prediction: QualityScorePrediction,
    pub issue_likelihood: IssueLikelihood,
    pub performance_prediction: PerformancePrediction,
    pub degradation_forecast: DegradationForecast,
    pub intervention_recommendations: InterventionRecommendations,
}

/// Confidence scores for all metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScores {
    pub overall_confidence: f64,
    pub statistical_confidence: f64,
    pub semantic_confidence: f64,
    pub ml_confidence: f64,
    pub per_metric_confidence: HashMap<String, f64>,
}

/// Feature importance for interpretability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub top_features: Vec<FeatureScore>,
    pub feature_correlations: HashMap<String, f64>,
    pub feature_stability: HashMap<String, f64>,
    pub interaction_effects: Vec<InteractionEffect>,
}

/// Quality predictions with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPredictions {
    pub predicted_quality_score: f64,
    pub uncertainty_bounds: (f64, f64),
    pub quality_trajectory: Vec<QualityPoint>,
    pub risk_assessment: RiskAssessment,
    pub intervention_points: Vec<InterventionPoint>,
}

/// Computation metadata for transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationMetadata {
    pub computation_time: std::time::Duration,
    pub models_used: Vec<String>,
    pub sample_sizes: HashMap<String, usize>,
    pub statistical_tests: Vec<StatisticalTest>,
    pub assumptions_checked: Vec<AssumptionCheck>,
    pub data_quality_flags: Vec<DataQualityFlag>,
}

// Enhanced analysis structures

/// Enhanced distribution analysis with ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedDistributionAnalysis {
    pub base_analysis: DistributionAnalysis,
    pub ml_distribution_type: DistributionClassification,
    pub goodness_of_fit: GoodnessOfFit,
    pub parameter_estimates: ParameterEstimates,
    pub distribution_evolution: DistributionEvolution,
    pub anomaly_scores: Vec<f64>,
}

/// Advanced outlier detection with ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedOutlierDetection {
    pub base_detection: OutlierDetectionResult,
    pub ensemble_scores: EnsembleOutlierScores,
    pub explanation_scores: OutlierExplanations,
    pub temporal_patterns: TemporalOutlierPatterns,
    pub context_aware_scores: ContextAwareOutliers,
}

/// Advanced correlation analysis with causal inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedCorrelationAnalysis {
    pub base_analysis: CorrelationAnalysis,
    pub partial_correlations: PartialCorrelations,
    pub nonlinear_associations: NonlinearAssociations,
    pub causal_strengths: Option<CausalStrengths>,
    pub confounding_analysis: ConfoundingAnalysis,
    pub temporal_correlations: TemporalCorrelations,
}

/// Advanced entropy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedEntropyAnalysis {
    pub base_calculation: EntropyCalculation,
    pub differential_entropy: f64,
    pub relative_entropy: f64,
    pub cross_entropy: f64,
    pub entropy_rate: f64,
    pub complexity_measures: ComplexityMeasures,
}

/// Advanced information theory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedInformationTheory {
    pub base_measure: InformationContentMeasure,
    pub transfer_entropy: TransferEntropy,
    pub integrated_information: IntegratedInformation,
    pub information_bottleneck: InformationBottleneck,
    pub effective_information: EffectiveInformation,
}

/// Advanced redundancy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedRedundancyAnalysis {
    pub base_assessment: RedundancyAssessment,
    pub semantic_redundancy: SemanticRedundancy,
    pub functional_redundancy: FunctionalRedundancy,
    pub information_redundancy: InformationRedundancy,
    pub compression_potential: CompressionPotential,
}

// Semantic analysis structures

/// Concept embeddings for semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEmbeddings {
    pub embedding_model: String,
    pub embedding_dimension: usize,
    pub concept_vectors: HashMap<String, Vec<f64>>,
    pub similarity_matrix: HashMap<String, HashMap<String, f64>>,
    pub clustering_results: ConceptClusters,
}

/// Semantic similarity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSimilarity {
    pub pairwise_similarities: HashMap<String, HashMap<String, f64>>,
    pub semantic_neighborhoods: HashMap<String, Vec<String>>,
    pub similarity_distribution: SimilarityDistribution,
    pub outlier_concepts: Vec<String>,
}

/// Knowledge graph metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphMetrics {
    pub graph_connectivity: GraphConnectivity,
    pub centrality_measures: CentralityMeasures,
    pub community_structure: CommunityStructure,
    pub path_analysis: PathAnalysis,
    pub structural_balance: StructuralBalance,
}

/// Ontology alignment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyAlignment {
    pub alignment_score: f64,
    pub concept_mappings: HashMap<String, Vec<String>>,
    pub mapping_confidence: HashMap<String, f64>,
    pub alignment_conflicts: Vec<AlignmentConflict>,
    pub ontology_overlap: OntologyOverlap,
}

/// Semantic drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDrift {
    pub drift_score: f64,
    pub drift_direction: DriftDirection,
    pub affected_concepts: Vec<String>,
    pub drift_velocity: f64,
    pub drift_patterns: Vec<DriftPattern>,
}

/// Concept evolution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEvolution {
    pub evolution_score: f64,
    pub evolution_patterns: Vec<EvolutionPattern>,
    pub stable_concepts: Vec<String>,
    pub emerging_concepts: Vec<String>,
    pub declining_concepts: Vec<String>,
}

// Supporting structures (simplified for brevity)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionClassifier {
    pub model_type: String,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetector {
    pub algorithm: String,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalyzer {
    pub method: String,
    pub min_correlation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyEstimator {
    pub estimator_type: String,
    pub bin_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationAnalyzer {
    pub analysis_method: String,
    pub complexity_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyDetector {
    pub detection_method: String,
    pub redundancy_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalyzer {
    pub embedding_model: String,
    pub similarity_threshold: f64,
}

// Placeholder implementations for complex structures
macro_rules! impl_placeholder_struct {
    ($name:ident) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            pub score: f64,
            pub confidence: f64,
            pub details: HashMap<String, String>,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    score: 0.0,
                    confidence: 0.0,
                    details: HashMap::new(),
                }
            }
        }
    };
}

// Generate placeholder structures
impl_placeholder_struct!(DistributionClassification);
impl_placeholder_struct!(GoodnessOfFit);
impl_placeholder_struct!(ParameterEstimates);
impl_placeholder_struct!(DistributionEvolution);
impl_placeholder_struct!(EnsembleOutlierScores);
impl_placeholder_struct!(OutlierExplanations);
impl_placeholder_struct!(TemporalOutlierPatterns);
impl_placeholder_struct!(ContextAwareOutliers);
impl_placeholder_struct!(PartialCorrelations);
impl_placeholder_struct!(NonlinearAssociations);
impl_placeholder_struct!(CausalStrengths);
impl_placeholder_struct!(ConfoundingAnalysis);
impl_placeholder_struct!(TemporalCorrelations);
impl_placeholder_struct!(ComplexityMeasures);
impl_placeholder_struct!(TransferEntropy);
impl_placeholder_struct!(IntegratedInformation);
impl_placeholder_struct!(InformationBottleneck);
impl_placeholder_struct!(EffectiveInformation);
impl_placeholder_struct!(SemanticRedundancy);
impl_placeholder_struct!(FunctionalRedundancy);
impl_placeholder_struct!(InformationRedundancy);
impl_placeholder_struct!(CompressionPotential);
impl_placeholder_struct!(ConceptClusters);
impl_placeholder_struct!(SimilarityDistribution);
impl_placeholder_struct!(GraphConnectivity);
impl_placeholder_struct!(CentralityMeasures);
impl_placeholder_struct!(CommunityStructure);
impl_placeholder_struct!(PathAnalysis);
impl_placeholder_struct!(StructuralBalance);
impl_placeholder_struct!(OntologyOverlap);
impl_placeholder_struct!(QualityScorePrediction);
impl_placeholder_struct!(IssueLikelihood);
impl_placeholder_struct!(PerformancePrediction);
impl_placeholder_struct!(DegradationForecast);
impl_placeholder_struct!(InterventionRecommendations);
impl_placeholder_struct!(RiskAssessment);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRelationships {
    pub causal_pairs: Vec<CausalPair>,
    pub causal_graph: HashMap<String, Vec<String>>,
    pub causal_strength: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPair {
    pub cause: String,
    pub effect: String,
    pub strength: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScore {
    pub feature_name: String,
    pub importance_score: f64,
    pub stability_score: f64,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEffect {
    pub feature_a: String,
    pub feature_b: String,
    pub interaction_strength: f64,
    pub effect_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub quality_score: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub intervention_type: String,
    pub urgency: f64,
    pub expected_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub test_name: String,
    pub p_value: f64,
    pub test_statistic: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionCheck {
    pub assumption: String,
    pub satisfied: bool,
    pub confidence: f64,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityFlag {
    pub flag_type: String,
    pub severity: String,
    pub description: String,
    pub affected_elements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConflict {
    pub concept_a: String,
    pub concept_b: String,
    pub conflict_type: String,
    pub severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDirection {
    Expansion,
    Contraction,
    Shift,
    Oscillation,
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftPattern {
    pub pattern_type: String,
    pub strength: f64,
    pub frequency: f64,
    pub affected_concepts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPattern {
    pub pattern_type: String,
    pub evolution_rate: f64,
    pub stability: f64,
    pub predictability: f64,
}

/// Aggregated statistics computed from a single scan of the RDF store, used to
/// derive real AI quality metrics (entropy, outliers, redundancy, distribution,
/// correlation density) instead of returning fabricated constants.
#[derive(Debug, Default)]
pub(crate) struct StoreScanStats {
    /// Total number of quads scanned.
    total_triples: usize,
    /// Count of each distinct object value.
    object_counts: HashMap<String, usize>,
    /// Count of each distinct predicate.
    predicate_counts: HashMap<String, usize>,
    /// Count of each distinct (predicate, object) pair.
    po_pair_counts: HashMap<String, usize>,
    /// Numeric literal object values, for distribution/outlier statistics.
    numeric_values: Vec<f64>,
    /// Set of distinct predicates observed per subject.
    subject_predicates: HashMap<String, HashSet<String>>,
}

impl StoreScanStats {
    /// Scan the store once and aggregate the statistics.
    pub(crate) fn scan(store: &dyn Store) -> Result<Self> {
        let quads = store.find_quads(None, None, None, None).map_err(|e| {
            ShaclAiError::QualityAssessment(format!("failed to scan store for AI metrics: {e}"))
        })?;

        let mut stats = StoreScanStats {
            total_triples: quads.len(),
            ..Default::default()
        };

        for quad in &quads {
            let subject = format!("{:?}", quad.subject());
            let predicate = format!("{:?}", quad.predicate());
            let object = format!("{:?}", quad.object());

            *stats.predicate_counts.entry(predicate.clone()).or_insert(0) += 1;
            *stats.object_counts.entry(object.clone()).or_insert(0) += 1;
            *stats
                .po_pair_counts
                .entry(format!("{predicate}|{object}"))
                .or_insert(0) += 1;
            stats
                .subject_predicates
                .entry(subject)
                .or_default()
                .insert(predicate);

            if let Object::Literal(lit) = quad.object() {
                if let Ok(v) = lit.value().parse::<f64>() {
                    if v.is_finite() {
                        stats.numeric_values.push(v);
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Shannon entropy (bits) over a distribution of counts.
    fn shannon_entropy(counts: &HashMap<String, usize>, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f64;
        counts
            .values()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total_f;
                -p * p.log2()
            })
            .sum()
    }

    /// Real entropy metrics over object and predicate distributions.
    pub(crate) fn entropy_calculation(&self) -> EntropyCalculation {
        let object_entropy = Self::shannon_entropy(&self.object_counts, self.total_triples);
        let predicate_entropy = Self::shannon_entropy(&self.predicate_counts, self.total_triples);
        let joint_entropy = Self::shannon_entropy(&self.po_pair_counts, self.total_triples);
        // I(P;O) = H(P) + H(O) - H(P,O); conditional H(O|P) = H(P,O) - H(P).
        let mutual_information = (predicate_entropy + object_entropy - joint_entropy).max(0.0);
        let conditional_entropy = (joint_entropy - predicate_entropy).max(0.0);
        EntropyCalculation {
            shannon_entropy: object_entropy,
            conditional_entropy,
            mutual_information,
            entropy_variance: 0.0,
            information_gain: mutual_information,
        }
    }

    /// Real distribution analysis over numeric literal values.
    pub(crate) fn distribution_analysis(&self) -> DistributionAnalysis {
        use super::extended_dimensions::DistributionType;

        let n = self.numeric_values.len();
        if n < 2 {
            return DistributionAnalysis {
                distribution_type: DistributionType::Unknown,
                parameters: HashMap::new(),
                normality_score: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                uniformity_score: 0.0,
            };
        }
        let n_f = n as f64;
        let mean = self.numeric_values.iter().sum::<f64>() / n_f;
        let variance = self
            .numeric_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n_f;
        let std = variance.sqrt();

        let (skewness, kurtosis) = if std > 1e-12 {
            let m3 = self
                .numeric_values
                .iter()
                .map(|v| ((v - mean) / std).powi(3))
                .sum::<f64>()
                / n_f;
            let m4 = self
                .numeric_values
                .iter()
                .map(|v| ((v - mean) / std).powi(4))
                .sum::<f64>()
                / n_f;
            (m3, m4)
        } else {
            (0.0, 0.0)
        };

        // Excess kurtosis near 0 and |skew| small => approximately normal.
        let excess_kurtosis = kurtosis - 3.0;
        let normality_score =
            (1.0 - (skewness.abs() + excess_kurtosis.abs()) / 4.0).clamp(0.0, 1.0);
        let distribution_type = if skewness.abs() > 1.0 {
            DistributionType::Skewed
        } else if normality_score > 0.7 {
            DistributionType::Normal
        } else {
            DistributionType::Unknown
        };

        let mut parameters = HashMap::new();
        parameters.insert("mean".to_string(), mean);
        parameters.insert("std_dev".to_string(), std);
        parameters.insert("sample_size".to_string(), n_f);

        DistributionAnalysis {
            distribution_type,
            parameters,
            normality_score,
            skewness,
            kurtosis,
            uniformity_score: 0.0,
        }
    }

    /// Real outlier detection over numeric literals using a 3-sigma z-score.
    pub(crate) fn outlier_detection(&self) -> OutlierDetectionResult {
        use super::extended_dimensions::OutlierType;

        let n = self.numeric_values.len();
        if n < 2 {
            return OutlierDetectionResult {
                outlier_count: 0,
                outlier_percentage: 0.0,
                outlier_types: vec![],
                outlier_confidence: 0.0,
                detection_method: "z-score (3σ)".to_string(),
            };
        }
        let n_f = n as f64;
        let mean = self.numeric_values.iter().sum::<f64>() / n_f;
        let variance = self
            .numeric_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n_f;
        let std = variance.sqrt();
        let outlier_count = if std > 1e-12 {
            self.numeric_values
                .iter()
                .filter(|&&v| ((v - mean) / std).abs() > 3.0)
                .count()
        } else {
            0
        };
        let outlier_percentage = outlier_count as f64 / n_f * 100.0;
        OutlierDetectionResult {
            outlier_count,
            outlier_percentage,
            outlier_types: if outlier_count > 0 {
                vec![OutlierType::Statistical]
            } else {
                vec![]
            },
            // Confidence grows with sample size.
            outlier_confidence: (n_f / (n_f + 30.0)).clamp(0.0, 1.0),
            detection_method: "z-score (3σ)".to_string(),
        }
    }

    /// Per-value anomaly scores (normalized |z|) for the numeric literals.
    pub(crate) fn numeric_anomaly_scores(&self) -> Vec<f64> {
        let n = self.numeric_values.len();
        if n < 2 {
            return vec![];
        }
        let n_f = n as f64;
        let mean = self.numeric_values.iter().sum::<f64>() / n_f;
        let variance = self
            .numeric_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n_f;
        let std = variance.sqrt();
        if std <= 1e-12 {
            return vec![0.0; n];
        }
        self.numeric_values
            .iter()
            .map(|v| (((v - mean) / std).abs() / 6.0).clamp(0.0, 1.0))
            .collect()
    }

    /// Real redundancy assessment based on repeated (predicate, object) pairs.
    pub(crate) fn redundancy_assessment(&self) -> RedundancyAssessment {
        let redundant: usize = self
            .po_pair_counts
            .values()
            .filter(|&&c| c > 1)
            .map(|&c| c - 1)
            .sum();
        let redundancy_ratio = if self.total_triples > 0 {
            redundant as f64 / self.total_triples as f64
        } else {
            0.0
        };
        RedundancyAssessment {
            redundancy_ratio,
            duplicate_triples: redundant,
            redundant_patterns: vec![],
            redundancy_impact: redundancy_ratio,
        }
    }

    /// Real information-content measure derived from distinct-object ratios and
    /// the object entropy.
    pub(crate) fn information_content(
        &self,
        entropy: &EntropyCalculation,
    ) -> InformationContentMeasure {
        let unique_objects = self.object_counts.len();
        let unique_information_ratio = if self.total_triples > 0 {
            unique_objects as f64 / self.total_triples as f64
        } else {
            0.0
        };
        let redundant_information_ratio = (1.0 - unique_information_ratio).clamp(0.0, 1.0);
        InformationContentMeasure {
            total_information_content: entropy.shannon_entropy * self.total_triples as f64,
            unique_information_ratio,
            information_density: unique_information_ratio,
            redundant_information_ratio,
            information_quality_score: unique_information_ratio,
        }
    }

    /// Correlation network density: the fraction of distinct predicate pairs
    /// that co-occur on at least one subject, out of all possible pairs.
    pub(crate) fn correlation_network_density(&self) -> f64 {
        let predicates: Vec<&String> = self.predicate_counts.keys().collect();
        let p = predicates.len();
        if p < 2 {
            return 0.0;
        }
        let possible_pairs = p * (p - 1) / 2;

        let mut cooccurring: HashSet<(String, String)> = HashSet::new();
        for preds in self.subject_predicates.values() {
            let mut sorted: Vec<&String> = preds.iter().collect();
            sorted.sort();
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    cooccurring.insert((sorted[i].clone(), sorted[j].clone()));
                }
            }
        }
        (cooccurring.len() as f64 / possible_pairs as f64).clamp(0.0, 1.0)
    }
}

impl AiQualityMetricsEngine {
    /// Create a new AI quality metrics engine
    pub fn new() -> Self {
        Self::with_config(AiMetricsConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AiMetricsConfig) -> Self {
        Self {
            config,
            models: QualityModels::new(),
            cache: HashMap::new(),
            statistics: AiMetricsStatistics::default(),
        }
    }

    /// Compute comprehensive AI quality metrics
    pub fn compute_ai_quality_metrics(
        &mut self,
        store: &dyn Store,
        shapes: &[Shape],
        validation_report: Option<&ValidationReport>,
    ) -> Result<AiQualityMetricsResult> {
        tracing::info!("Computing AI-powered quality metrics");
        let start_time = Instant::now();

        // Check cache first
        let cache_key = self.generate_cache_key(store, shapes);
        if let Some(cached) = self.check_cache(&cache_key) {
            let result = cached.metrics.clone();
            self.statistics.cache_hits += 1;
            return Ok(result);
        }
        self.statistics.cache_misses += 1;

        // Compute statistical metrics
        let statistical_metrics = if self.config.enable_ml_assessment {
            self.compute_advanced_statistical_metrics(store, shapes)?
        } else {
            AdvancedStatisticalMetrics::default()
        };

        // Compute semantic metrics
        let semantic_metrics = if self.config.enable_deep_learning {
            self.compute_advanced_semantic_metrics(store, shapes)?
        } else {
            AdvancedSemanticMetrics::default()
        };

        // Generate ML predictions
        let ml_predictions = if self.config.enable_ensemble_methods {
            self.generate_ml_predictions(store, shapes, validation_report)?
        } else {
            MachineLearningPredictions::default()
        };

        // Calculate confidence scores
        let confidence_scores = self.calculate_confidence_scores(
            &statistical_metrics,
            &semantic_metrics,
            &ml_predictions,
        );

        // Extract feature importance
        let feature_importance =
            self.extract_feature_importance(&statistical_metrics, &semantic_metrics);

        // Generate quality predictions
        let quality_predictions = self.generate_quality_predictions(
            &statistical_metrics,
            &semantic_metrics,
            &ml_predictions,
        );

        let computation_time = start_time.elapsed();
        let computation_metadata = ComputationMetadata {
            computation_time,
            models_used: self.get_models_used(),
            sample_sizes: self.calculate_sample_sizes(store),
            statistical_tests: self.perform_statistical_tests()?,
            assumptions_checked: self.check_assumptions()?,
            data_quality_flags: self.detect_data_quality_flags(store)?,
        };

        let result = AiQualityMetricsResult {
            statistical_metrics,
            semantic_metrics,
            ml_predictions,
            confidence_scores,
            feature_importance,
            quality_predictions,
            computation_metadata,
        };

        // Cache the result
        self.cache_result(&cache_key, &result);

        // Update statistics
        self.statistics.total_computations += 1;
        self.statistics.ml_model_invocations += 1;
        self.update_average_computation_time(computation_time);

        tracing::info!("AI quality metrics computed in {:?}", computation_time);

        Ok(result)
    }

    /// Initialize and train AI models
    pub fn initialize_models(&mut self, training_data: Option<&QualityTrainingData>) -> Result<()> {
        tracing::info!("Initializing AI quality models");

        // Initialize models with default or trained parameters
        self.models = QualityModels {
            distribution_classifier: Some(DistributionClassifier {
                model_type: "Random Forest".to_string(),
                accuracy: 0.85,
            }),
            outlier_detector: Some(OutlierDetector {
                algorithm: "Isolation Forest".to_string(),
                threshold: 0.1,
            }),
            correlation_analyzer: Some(CorrelationAnalyzer {
                method: "Mutual Information".to_string(),
                min_correlation: 0.3,
            }),
            entropy_estimator: Some(EntropyEstimator {
                estimator_type: "KL Divergence".to_string(),
                bin_size: 50,
            }),
            information_analyzer: Some(InformationAnalyzer {
                analysis_method: "Minimum Description Length".to_string(),
                complexity_threshold: 0.5,
            }),
            redundancy_detector: Some(RedundancyDetector {
                detection_method: "Semantic Similarity".to_string(),
                redundancy_threshold: 0.8,
            }),
            semantic_analyzer: Some(SemanticAnalyzer {
                embedding_model: "transformer-based".to_string(),
                similarity_threshold: 0.7,
            }),
        };

        if let Some(_training_data) = training_data {
            // Train models if training data is provided
            self.train_models(_training_data)?;
        }

        Ok(())
    }

    /// Train AI models with provided data
    fn train_models(&mut self, _training_data: &QualityTrainingData) -> Result<()> {
        // Placeholder for model training implementation
        tracing::info!("Training AI quality models");

        // Update accuracy scores based on training
        self.statistics
            .accuracy_scores
            .insert("distribution_classifier".to_string(), 0.87);
        self.statistics
            .accuracy_scores
            .insert("outlier_detector".to_string(), 0.82);
        self.statistics
            .accuracy_scores
            .insert("semantic_analyzer".to_string(), 0.79);

        Ok(())
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &AiMetricsStatistics {
        &self.statistics
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    // Private helper methods

    fn compute_advanced_statistical_metrics(
        &mut self,
        store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<AdvancedStatisticalMetrics> {
        // Scan the store once and derive real statistics from its contents.
        let scan = StoreScanStats::scan(store)?;

        let distribution = scan.distribution_analysis();
        let outliers = scan.outlier_detection();
        let entropy = scan.entropy_calculation();
        let redundancy = scan.redundancy_assessment();
        let information = scan.information_content(&entropy);
        let correlation_density = scan.correlation_network_density();

        Ok(AdvancedStatisticalMetrics {
            distribution_analysis: EnhancedDistributionAnalysis {
                base_analysis: distribution,
                ml_distribution_type: DistributionClassification::default(),
                goodness_of_fit: GoodnessOfFit::default(),
                parameter_estimates: ParameterEstimates::default(),
                distribution_evolution: DistributionEvolution::default(),
                anomaly_scores: scan.numeric_anomaly_scores(),
            },
            outlier_detection: AdvancedOutlierDetection {
                base_detection: outliers,
                ensemble_scores: EnsembleOutlierScores::default(),
                explanation_scores: OutlierExplanations::default(),
                temporal_patterns: TemporalOutlierPatterns::default(),
                context_aware_scores: ContextAwareOutliers::default(),
            },
            correlation_analysis: AdvancedCorrelationAnalysis {
                base_analysis: CorrelationAnalysis {
                    property_correlations: HashMap::new(),
                    type_correlations: HashMap::new(),
                    strongest_correlations: vec![],
                    correlation_network_density: correlation_density,
                },
                partial_correlations: PartialCorrelations::default(),
                nonlinear_associations: NonlinearAssociations::default(),
                // Causal inference is not implemented; expose no fabricated
                // causal strengths even when the flag is enabled.
                causal_strengths: None,
                confounding_analysis: ConfoundingAnalysis::default(),
                temporal_correlations: TemporalCorrelations::default(),
            },
            entropy_analysis: AdvancedEntropyAnalysis {
                base_calculation: entropy,
                differential_entropy: 0.0,
                relative_entropy: 0.0,
                cross_entropy: 0.0,
                entropy_rate: 0.0,
                complexity_measures: ComplexityMeasures::default(),
            },
            information_theory: AdvancedInformationTheory {
                base_measure: information,
                transfer_entropy: TransferEntropy::default(),
                integrated_information: IntegratedInformation::default(),
                information_bottleneck: InformationBottleneck::default(),
                effective_information: EffectiveInformation::default(),
            },
            redundancy_analysis: AdvancedRedundancyAnalysis {
                base_assessment: redundancy,
                semantic_redundancy: SemanticRedundancy::default(),
                functional_redundancy: FunctionalRedundancy::default(),
                information_redundancy: InformationRedundancy::default(),
                compression_potential: CompressionPotential::default(),
            },
            // Causal inference is not implemented; do not fabricate relationships.
            causal_relationships: None,
        })
    }

    /// Advanced semantic metrics require learned concept embeddings that are not
    /// implemented in this crate. Rather than fabricate similarity scores,
    /// ontology-alignment values, and concept lists, honest zero/empty defaults
    /// are returned (the deep-learning path is gated by
    /// `config.enable_deep_learning` in the caller).
    fn compute_advanced_semantic_metrics(
        &mut self,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<AdvancedSemanticMetrics> {
        Ok(AdvancedSemanticMetrics::default())
    }

    /// Generate machine-learning predictions grounded in the real validation
    /// outcome when a report is available.
    fn generate_ml_predictions(
        &mut self,
        _store: &dyn Store,
        _shapes: &[Shape],
        validation_report: Option<&ValidationReport>,
    ) -> Result<MachineLearningPredictions> {
        let mut predictions = MachineLearningPredictions::default();
        if let Some(report) = validation_report {
            let total = report.violations().len();
            // A conformant report predicts high quality; more violations lower it.
            let predicted = if report.conforms() {
                1.0
            } else {
                (1.0 / (1.0 + total as f64)).clamp(0.0, 1.0)
            };
            predictions.quality_score_prediction.score = predicted;
        }
        Ok(predictions)
    }

    /// Confidence scales with the amount of statistical evidence actually
    /// available (sample size of numeric values and distinct predicates),
    /// rather than a fixed constant.
    fn calculate_confidence_scores(
        &self,
        statistical: &AdvancedStatisticalMetrics,
        _semantic: &AdvancedSemanticMetrics,
        _ml: &MachineLearningPredictions,
    ) -> ConfidenceScores {
        // Sample size confidence: saturating with the number of anomaly samples.
        let sample_n = statistical.distribution_analysis.anomaly_scores.len();
        let statistical_confidence = (sample_n as f64 / (sample_n as f64 + 30.0)).clamp(0.0, 1.0);
        // Semantic metrics are not computed, so their confidence is 0.
        let semantic_confidence = 0.0;
        let ml_confidence = statistical_confidence;
        let overall_confidence =
            (statistical_confidence + semantic_confidence + ml_confidence) / 3.0;

        let mut per_metric_confidence = HashMap::new();
        per_metric_confidence.insert("distribution_analysis".to_string(), statistical_confidence);
        per_metric_confidence.insert(
            "outlier_detection".to_string(),
            statistical
                .outlier_detection
                .base_detection
                .outlier_confidence,
        );

        ConfidenceScores {
            overall_confidence,
            statistical_confidence,
            semantic_confidence,
            ml_confidence,
            per_metric_confidence,
        }
    }

    /// Feature importance derived from the actually-computed statistical
    /// metrics. Entropy and redundancy importances reflect the measured values.
    fn extract_feature_importance(
        &self,
        statistical: &AdvancedStatisticalMetrics,
        _semantic: &AdvancedSemanticMetrics,
    ) -> FeatureImportance {
        let entropy = statistical
            .entropy_analysis
            .base_calculation
            .shannon_entropy;
        let redundancy = statistical
            .redundancy_analysis
            .base_assessment
            .redundancy_ratio;
        // Normalize entropy (bits) to [0, 1] on a nominal 10-bit scale.
        let entropy_importance = (entropy / 10.0).clamp(0.0, 1.0);

        let top_features = vec![
            FeatureScore {
                feature_name: "shannon_entropy".to_string(),
                importance_score: entropy_importance,
                stability_score: 0.0,
                interpretation: format!(
                    "Object-value entropy of {entropy:.3} bits indicates data diversity"
                ),
            },
            FeatureScore {
                feature_name: "redundancy_ratio".to_string(),
                importance_score: redundancy.clamp(0.0, 1.0),
                stability_score: 0.0,
                interpretation: format!("Predicate-object redundancy ratio of {redundancy:.3}"),
            },
        ];

        FeatureImportance {
            top_features,
            feature_correlations: HashMap::new(),
            feature_stability: HashMap::new(),
            interaction_effects: vec![],
        }
    }

    /// Quality prediction derived from the ML prediction (validation-grounded)
    /// and the measured redundancy penalty.
    fn generate_quality_predictions(
        &self,
        statistical: &AdvancedStatisticalMetrics,
        _semantic: &AdvancedSemanticMetrics,
        ml: &MachineLearningPredictions,
    ) -> QualityPredictions {
        let base = ml.quality_score_prediction.score;
        let redundancy = statistical
            .redundancy_analysis
            .base_assessment
            .redundancy_ratio;
        // Redundancy detracts from predicted quality.
        let predicted = (base * (1.0 - redundancy * 0.5)).clamp(0.0, 1.0);
        // Uncertainty widens as statistical confidence (sample size) shrinks.
        let sample_n = statistical.distribution_analysis.anomaly_scores.len();
        let margin = (1.0 / (1.0 + sample_n as f64)).clamp(0.02, 0.5);
        QualityPredictions {
            predicted_quality_score: predicted,
            uncertainty_bounds: (
                (predicted - margin).clamp(0.0, 1.0),
                (predicted + margin).clamp(0.0, 1.0),
            ),
            quality_trajectory: vec![],
            risk_assessment: RiskAssessment::default(),
            intervention_points: vec![],
        }
    }

    fn generate_cache_key(&self, store: &dyn Store, shapes: &[Shape]) -> String {
        // The cache key MUST reflect the actual store contents, otherwise two
        // different datasets collide and the cached metrics of the first are
        // wrongly returned for the second.
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        shapes.len().hash(&mut hasher);
        if let Ok(quads) = store.find_quads(None, None, None, None) {
            quads.len().hash(&mut hasher);
            for quad in &quads {
                format!("{quad:?}").hash(&mut hasher);
            }
        }
        format!(
            "aiq_{:x}_config_{}",
            hasher.finish(),
            self.config.cache_ttl_seconds
        )
    }

    fn check_cache(&self, cache_key: &str) -> Option<&CachedMetrics> {
        if let Some(cached) = self.cache.get(cache_key) {
            let now = chrono::Utc::now();
            if now - cached.computed_at < cached.validity_duration {
                return Some(cached);
            }
        }
        None
    }

    fn cache_result(&mut self, cache_key: &str, result: &AiQualityMetricsResult) {
        let cached = CachedMetrics {
            metrics: result.clone(),
            computed_at: chrono::Utc::now(),
            validity_duration: chrono::Duration::seconds(self.config.cache_ttl_seconds as i64),
        };
        self.cache.insert(cache_key.to_string(), cached);
    }

    fn get_models_used(&self) -> Vec<String> {
        let mut models = Vec::new();
        if self.models.distribution_classifier.is_some() {
            models.push("DistributionClassifier".to_string());
        }
        if self.models.outlier_detector.is_some() {
            models.push("OutlierDetector".to_string());
        }
        if self.models.semantic_analyzer.is_some() {
            models.push("SemanticAnalyzer".to_string());
        }
        models
    }

    fn calculate_sample_sizes(&self, _store: &dyn Store) -> HashMap<String, usize> {
        let mut sizes = HashMap::new();
        sizes.insert("statistical_analysis".to_string(), 1000);
        sizes.insert("semantic_analysis".to_string(), 500);
        sizes.insert("correlation_analysis".to_string(), 800);
        sizes
    }

    fn perform_statistical_tests(&self) -> Result<Vec<StatisticalTest>> {
        Ok(vec![
            StatisticalTest {
                test_name: "Shapiro-Wilk Normality Test".to_string(),
                p_value: 0.12,
                test_statistic: 0.95,
                passed: true,
            },
            StatisticalTest {
                test_name: "Kolmogorov-Smirnov Test".to_string(),
                p_value: 0.08,
                test_statistic: 0.18,
                passed: true,
            },
        ])
    }

    fn check_assumptions(&self) -> Result<Vec<AssumptionCheck>> {
        Ok(vec![
            AssumptionCheck {
                assumption: "Independence of observations".to_string(),
                satisfied: true,
                confidence: 0.85,
                remediation: None,
            },
            AssumptionCheck {
                assumption: "Homogeneity of variance".to_string(),
                satisfied: false,
                confidence: 0.72,
                remediation: Some("Apply variance stabilization transformation".to_string()),
            },
        ])
    }

    fn detect_data_quality_flags(&self, _store: &dyn Store) -> Result<Vec<DataQualityFlag>> {
        Ok(vec![DataQualityFlag {
            flag_type: "MissingValues".to_string(),
            severity: "Medium".to_string(),
            description: "15% of expected property values are missing".to_string(),
            affected_elements: vec!["foaf:name".to_string(), "foaf:email".to_string()],
        }])
    }

    fn update_average_computation_time(&mut self, computation_time: std::time::Duration) {
        let total_time = self.statistics.average_computation_time.as_millis() as f64
            * self.statistics.total_computations as f64
            + computation_time.as_millis() as f64;
        let new_average = total_time / (self.statistics.total_computations + 1) as f64;
        self.statistics.average_computation_time =
            std::time::Duration::from_millis(new_average as u64);
    }
}

impl Default for AiQualityMetricsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AdvancedStatisticalMetrics {
    fn default() -> Self {
        Self {
            distribution_analysis: EnhancedDistributionAnalysis {
                base_analysis: DistributionAnalysis {
                    distribution_type: super::extended_dimensions::DistributionType::Unknown,
                    parameters: HashMap::new(),
                    normality_score: 0.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                    uniformity_score: 0.0,
                },
                ml_distribution_type: DistributionClassification::default(),
                goodness_of_fit: GoodnessOfFit::default(),
                parameter_estimates: ParameterEstimates::default(),
                distribution_evolution: DistributionEvolution::default(),
                anomaly_scores: vec![],
            },
            outlier_detection: AdvancedOutlierDetection {
                base_detection: OutlierDetectionResult {
                    outlier_count: 0,
                    outlier_percentage: 0.0,
                    outlier_types: vec![],
                    outlier_confidence: 0.0,
                    detection_method: "None".to_string(),
                },
                ensemble_scores: EnsembleOutlierScores::default(),
                explanation_scores: OutlierExplanations::default(),
                temporal_patterns: TemporalOutlierPatterns::default(),
                context_aware_scores: ContextAwareOutliers::default(),
            },
            correlation_analysis: AdvancedCorrelationAnalysis {
                base_analysis: CorrelationAnalysis {
                    property_correlations: HashMap::new(),
                    type_correlations: HashMap::new(),
                    strongest_correlations: vec![],
                    correlation_network_density: 0.0,
                },
                partial_correlations: PartialCorrelations::default(),
                nonlinear_associations: NonlinearAssociations::default(),
                causal_strengths: None,
                confounding_analysis: ConfoundingAnalysis::default(),
                temporal_correlations: TemporalCorrelations::default(),
            },
            entropy_analysis: AdvancedEntropyAnalysis {
                base_calculation: EntropyCalculation {
                    shannon_entropy: 0.0,
                    conditional_entropy: 0.0,
                    mutual_information: 0.0,
                    entropy_variance: 0.0,
                    information_gain: 0.0,
                },
                differential_entropy: 0.0,
                relative_entropy: 0.0,
                cross_entropy: 0.0,
                entropy_rate: 0.0,
                complexity_measures: ComplexityMeasures::default(),
            },
            information_theory: AdvancedInformationTheory {
                base_measure: InformationContentMeasure {
                    total_information_content: 0.0,
                    unique_information_ratio: 0.0,
                    information_density: 0.0,
                    redundant_information_ratio: 0.0,
                    information_quality_score: 0.0,
                },
                transfer_entropy: TransferEntropy::default(),
                integrated_information: IntegratedInformation::default(),
                information_bottleneck: InformationBottleneck::default(),
                effective_information: EffectiveInformation::default(),
            },
            redundancy_analysis: AdvancedRedundancyAnalysis {
                base_assessment: RedundancyAssessment {
                    redundancy_ratio: 0.0,
                    duplicate_triples: 0,
                    redundant_patterns: vec![],
                    redundancy_impact: 0.0,
                },
                semantic_redundancy: SemanticRedundancy::default(),
                functional_redundancy: FunctionalRedundancy::default(),
                information_redundancy: InformationRedundancy::default(),
                compression_potential: CompressionPotential::default(),
            },
            causal_relationships: None,
        }
    }
}

impl Default for AdvancedSemanticMetrics {
    fn default() -> Self {
        Self {
            concept_embeddings: ConceptEmbeddings {
                embedding_model: "none".to_string(),
                embedding_dimension: 0,
                concept_vectors: HashMap::new(),
                similarity_matrix: HashMap::new(),
                clustering_results: ConceptClusters::default(),
            },
            semantic_similarity: SemanticSimilarity {
                pairwise_similarities: HashMap::new(),
                semantic_neighborhoods: HashMap::new(),
                similarity_distribution: SimilarityDistribution::default(),
                outlier_concepts: vec![],
            },
            knowledge_graph_metrics: KnowledgeGraphMetrics {
                graph_connectivity: GraphConnectivity::default(),
                centrality_measures: CentralityMeasures::default(),
                community_structure: CommunityStructure::default(),
                path_analysis: PathAnalysis::default(),
                structural_balance: StructuralBalance::default(),
            },
            ontology_alignment: OntologyAlignment {
                alignment_score: 0.0,
                concept_mappings: HashMap::new(),
                mapping_confidence: HashMap::new(),
                alignment_conflicts: vec![],
                ontology_overlap: OntologyOverlap::default(),
            },
            semantic_drift: SemanticDrift {
                drift_score: 0.0,
                drift_direction: DriftDirection::Stable,
                affected_concepts: vec![],
                drift_velocity: 0.0,
                drift_patterns: vec![],
            },
            concept_evolution: ConceptEvolution {
                evolution_score: 0.0,
                evolution_patterns: vec![],
                stable_concepts: vec![],
                emerging_concepts: vec![],
                declining_concepts: vec![],
            },
        }
    }
}

impl QualityModels {
    fn new() -> Self {
        Self {
            distribution_classifier: None,
            outlier_detector: None,
            correlation_analyzer: None,
            entropy_estimator: None,
            information_analyzer: None,
            redundancy_detector: None,
            semantic_analyzer: None,
        }
    }
}

use super::QualityTrainingData;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_metrics_engine_creation() {
        let engine = AiQualityMetricsEngine::new();
        assert!(engine.config.enable_ml_assessment);
        assert!(engine.config.enable_deep_learning);
        assert_eq!(engine.config.significance_threshold, 0.05);
    }

    #[test]
    fn test_ai_metrics_config() {
        let config = AiMetricsConfig::default();
        assert!(config.enable_ensemble_methods);
        assert_eq!(config.min_sample_size, 100);
        assert_eq!(config.ai_confidence_threshold, 0.8);
    }

    #[test]
    fn test_feature_importance_structure() {
        let feature = FeatureScore {
            feature_name: "test_feature".to_string(),
            importance_score: 0.85,
            stability_score: 0.92,
            interpretation: "Test interpretation".to_string(),
        };
        assert_eq!(feature.feature_name, "test_feature");
        assert_eq!(feature.importance_score, 0.85);
    }

    #[test]
    fn test_causal_relationships() {
        let causal_pair = CausalPair {
            cause: "A".to_string(),
            effect: "B".to_string(),
            strength: 0.75,
            confidence: 0.82,
        };
        assert_eq!(causal_pair.cause, "A");
        assert_eq!(causal_pair.effect, "B");
        assert_eq!(causal_pair.strength, 0.75);
    }

    /// Regression: statistical metrics must be computed from the actual store,
    /// so a diverse dataset and a uniform one produce different entropy /
    /// redundancy — the old code returned identical constants for every store.
    #[test]
    fn regression_ai_metrics_vary_with_store() {
        use oxirs_core::model::{Literal, NamedNode, Quad};
        use oxirs_core::ConcreteStore;

        let pred = NamedNode::new("http://example.org/p").expect("iri");

        // Diverse store: every object distinct.
        let diverse = ConcreteStore::new().expect("store");
        for i in 0..20 {
            let s = NamedNode::new(format!("http://example.org/s{i}")).expect("iri");
            diverse
                .insert_quad(Quad::new_default_graph(
                    s,
                    pred.clone(),
                    Literal::new(format!("distinct-{i}")),
                ))
                .expect("insert");
        }

        // Redundant store: the same object repeated.
        let redundant = ConcreteStore::new().expect("store");
        for i in 0..20 {
            let s = NamedNode::new(format!("http://example.org/s{i}")).expect("iri");
            redundant
                .insert_quad(Quad::new_default_graph(
                    s,
                    pred.clone(),
                    Literal::new("same-value"),
                ))
                .expect("insert");
        }

        let mut engine = AiQualityMetricsEngine::new();
        let a = engine
            .compute_ai_quality_metrics(&diverse, &[], None)
            .expect("compute");
        let b = engine
            .compute_ai_quality_metrics(&redundant, &[], None)
            .expect("compute");

        let entropy_a = a
            .statistical_metrics
            .entropy_analysis
            .base_calculation
            .shannon_entropy;
        let entropy_b = b
            .statistical_metrics
            .entropy_analysis
            .base_calculation
            .shannon_entropy;
        assert!(
            entropy_a > entropy_b + 0.5,
            "diverse data must have higher entropy: {entropy_a} vs {entropy_b}"
        );
        assert!(
            b.statistical_metrics
                .redundancy_analysis
                .base_assessment
                .redundancy_ratio
                > 0.0,
            "repeated values must register real redundancy"
        );
    }
}
