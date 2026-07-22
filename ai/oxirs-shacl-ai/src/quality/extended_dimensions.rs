//! Extended Quality Dimensions for SHACL-AI
//!
//! This module implements multi-dimensional quality assessment including
//! intrinsic and contextual quality dimensions with AI-powered metrics.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use oxirs_core::model::Object;
use oxirs_core::{RdfTerm, Store};
use oxirs_shacl::{Severity, Shape, ValidationReport};

use super::ai_metrics::StoreScanStats;
use crate::{Result, ShaclAiError};

/// Extended quality dimensions assessor
#[derive(Debug)]
pub struct ExtendedQualityDimensionsAssessor {
    config: ExtendedQualityConfig,
    cache: HashMap<String, QualityDimensionResult>,
}

/// Configuration for extended quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedQualityConfig {
    /// Enable intrinsic quality assessment
    pub enable_intrinsic_quality: bool,

    /// Enable contextual quality assessment
    pub enable_contextual_quality: bool,

    /// Enable statistical quality measures
    pub enable_statistical_measures: bool,

    /// Enable semantic quality measures
    pub enable_semantic_measures: bool,

    /// Precision measurement threshold
    pub precision_threshold: f64,

    /// Currency analysis window (in days)
    pub currency_window_days: u32,

    /// Relevance assessment method
    pub relevance_method: RelevanceMethod,

    /// Minimum confidence for quality measures
    pub min_confidence: f64,
}

impl Default for ExtendedQualityConfig {
    fn default() -> Self {
        Self {
            enable_intrinsic_quality: true,
            enable_contextual_quality: true,
            enable_statistical_measures: true,
            enable_semantic_measures: true,
            precision_threshold: 0.8,
            currency_window_days: 30,
            relevance_method: RelevanceMethod::Statistical,
            min_confidence: 0.7,
        }
    }
}

/// Methods for relevance assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelevanceMethod {
    Statistical,
    Semantic,
    MachineLearning,
    Hybrid,
}

/// Multi-dimensional quality assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDimensionalQualityAssessment {
    pub intrinsic_quality: IntrinsicQualityAssessment,
    pub contextual_quality: ContextualQualityAssessment,
    pub statistical_measures: StatisticalQualityMeasures,
    pub semantic_measures: SemanticQualityMeasures,
    pub overall_quality_score: f64,
    pub assessment_timestamp: chrono::DateTime<chrono::Utc>,
    pub confidence_score: f64,
}

/// Intrinsic quality assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntrinsicQualityAssessment {
    pub accuracy: QualityDimensionResult,
    pub consistency: QualityDimensionResult,
    pub completeness: QualityDimensionResult,
    pub validity: QualityDimensionResult,
    pub precision: QualityDimensionResult,
    pub currency: QualityDimensionResult,
}

/// Contextual quality assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextualQualityAssessment {
    pub relevance: QualityDimensionResult,
    pub timeliness: QualityDimensionResult,
    pub accessibility: QualityDimensionResult,
    pub compliance: QualityDimensionResult,
    pub security: QualityDimensionResult,
    pub usability: QualityDimensionResult,
}

/// Statistical quality measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalQualityMeasures {
    pub distribution_analysis: DistributionAnalysis,
    pub outlier_detection: OutlierDetectionResult,
    pub correlation_analysis: CorrelationAnalysis,
    pub entropy_calculation: EntropyCalculation,
    pub information_content: InformationContentMeasure,
    pub redundancy_assessment: RedundancyAssessment,
}

/// Semantic quality measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticQualityMeasures {
    pub concept_coherence: ConceptCoherence,
    pub relationship_validity: RelationshipValidity,
    pub taxonomy_consistency: TaxonomyConsistency,
    pub semantic_density: SemanticDensity,
    pub knowledge_completeness: KnowledgeCompleteness,
    pub logical_consistency: LogicalConsistency,
}

/// Quality dimension result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDimensionResult {
    pub score: f64,
    pub confidence: f64,
    pub details: QualityDimensionDetails,
    pub issues: Vec<QualityDimensionIssue>,
    pub recommendations: Vec<QualityDimensionRecommendation>,
}

/// Quality dimension details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDimensionDetails {
    pub measurement_method: String,
    pub sample_size: usize,
    pub measurement_timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Quality dimension issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDimensionIssue {
    pub issue_type: String,
    pub severity: QualityIssueSeverity,
    pub description: String,
    pub affected_elements: Vec<String>,
    pub impact_score: f64,
}

/// Quality dimension recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDimensionRecommendation {
    pub action_type: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub estimated_improvement: f64,
    pub implementation_effort: ImplementationEffort,
}

/// Quality issue severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Urgent,
    High,
    Medium,
    Low,
}

/// Implementation effort
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Extensive,
}

/// Distribution analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    pub distribution_type: DistributionType,
    pub parameters: HashMap<String, f64>,
    pub normality_score: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub uniformity_score: f64,
}

/// Distribution type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    Uniform,
    Skewed,
    Bimodal,
    Unknown,
}

/// Outlier detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetectionResult {
    pub outlier_count: usize,
    pub outlier_percentage: f64,
    pub outlier_types: Vec<OutlierType>,
    pub outlier_confidence: f64,
    pub detection_method: String,
}

/// Outlier type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierType {
    Statistical,
    Semantic,
    Structural,
    Temporal,
}

/// Correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub property_correlations: HashMap<String, f64>,
    pub type_correlations: HashMap<String, f64>,
    pub strongest_correlations: Vec<CorrelationPair>,
    pub correlation_network_density: f64,
}

/// Correlation pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPair {
    pub element_a: String,
    pub element_b: String,
    pub correlation_coefficient: f64,
    pub significance: f64,
}

/// Entropy calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyCalculation {
    pub shannon_entropy: f64,
    pub conditional_entropy: f64,
    pub mutual_information: f64,
    pub entropy_variance: f64,
    pub information_gain: f64,
}

/// Information content measure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationContentMeasure {
    pub total_information_content: f64,
    pub unique_information_ratio: f64,
    pub information_density: f64,
    pub redundant_information_ratio: f64,
    pub information_quality_score: f64,
}

/// Redundancy assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyAssessment {
    pub redundancy_ratio: f64,
    pub duplicate_triples: usize,
    pub redundant_patterns: Vec<RedundantPattern>,
    pub redundancy_impact: f64,
}

/// Redundant pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundantPattern {
    pub pattern_type: String,
    pub occurrences: usize,
    pub redundancy_score: f64,
    pub suggested_consolidation: String,
}

/// Concept coherence assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCoherence {
    pub coherence_score: f64,
    pub concept_clusters: Vec<ConceptCluster>,
    pub incoherent_concepts: Vec<IncoherentConcept>,
    pub semantic_similarity_matrix: HashMap<String, HashMap<String, f64>>,
}

/// Concept cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCluster {
    pub cluster_id: String,
    pub concepts: Vec<String>,
    pub coherence_score: f64,
    pub centroid_concept: String,
}

/// Incoherent concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncoherentConcept {
    pub concept: String,
    pub incoherence_score: f64,
    pub conflicting_relationships: Vec<String>,
    pub recommended_action: String,
}

/// Relationship validity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipValidity {
    pub validity_score: f64,
    pub valid_relationships: usize,
    pub invalid_relationships: usize,
    pub suspicious_relationships: Vec<SuspiciousRelationship>,
}

/// Suspicious relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousRelationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub suspicion_score: f64,
    pub suspicion_reasons: Vec<String>,
}

/// Taxonomy consistency assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyConsistency {
    pub consistency_score: f64,
    pub hierarchy_violations: Vec<HierarchyViolation>,
    pub circular_dependencies: Vec<CircularDependency>,
    pub taxonomy_depth_analysis: TaxonomyDepthAnalysis,
}

/// Hierarchy violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyViolation {
    pub violation_type: String,
    pub involved_concepts: Vec<String>,
    pub severity: QualityIssueSeverity,
    pub description: String,
}

/// Circular dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularDependency {
    pub dependency_chain: Vec<String>,
    pub dependency_type: String,
    pub severity: QualityIssueSeverity,
}

/// Taxonomy depth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyDepthAnalysis {
    pub max_depth: usize,
    pub average_depth: f64,
    pub depth_variance: f64,
    pub optimal_depth_recommendation: usize,
}

/// Semantic density assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDensity {
    pub density_score: f64,
    pub relationship_density: f64,
    pub property_density: f64,
    pub type_density: f64,
    pub sparse_areas: Vec<SparseArea>,
    pub dense_areas: Vec<DenseArea>,
}

/// Sparse area in knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseArea {
    pub area_identifier: String,
    pub sparsity_score: f64,
    pub missing_relationships: Vec<String>,
    pub enrichment_suggestions: Vec<String>,
}

/// Dense area in knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseArea {
    pub area_identifier: String,
    pub density_score: f64,
    pub relationship_types: Vec<String>,
    pub potential_redundancies: Vec<String>,
}

/// Knowledge completeness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeCompleteness {
    pub completeness_score: f64,
    pub domain_coverage: DomainCoverage,
    pub missing_knowledge_areas: Vec<MissingKnowledgeArea>,
    pub knowledge_gaps: Vec<KnowledgeGap>,
}

/// Domain coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCoverage {
    pub covered_domains: Vec<String>,
    pub coverage_percentage: f64,
    pub domain_completeness: HashMap<String, f64>,
    pub interdomain_connections: usize,
}

/// Missing knowledge area
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingKnowledgeArea {
    pub domain: String,
    pub missing_concepts: Vec<String>,
    pub missing_relationships: Vec<String>,
    pub importance_score: f64,
}

/// Knowledge gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub gap_type: String,
    pub description: String,
    pub affected_concepts: Vec<String>,
    pub impact_score: f64,
    pub filling_priority: RecommendationPriority,
}

/// Logical consistency assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalConsistency {
    pub consistency_score: f64,
    pub logical_violations: Vec<LogicalViolation>,
    pub inconsistent_axioms: Vec<InconsistentAxiom>,
    pub reasoning_errors: Vec<ReasoningError>,
}

/// Logical violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalViolation {
    pub violation_type: String,
    pub involved_statements: Vec<String>,
    pub contradiction_explanation: String,
    pub severity: QualityIssueSeverity,
}

/// Inconsistent axiom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InconsistentAxiom {
    pub axiom: String,
    pub conflicting_axioms: Vec<String>,
    pub inconsistency_type: String,
    pub resolution_suggestions: Vec<String>,
}

/// Reasoning error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningError {
    pub error_type: String,
    pub problematic_inferences: Vec<String>,
    pub error_explanation: String,
    pub corrective_actions: Vec<String>,
}

impl ExtendedQualityDimensionsAssessor {
    /// Create a new extended quality dimensions assessor
    pub fn new() -> Self {
        Self::with_config(ExtendedQualityConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ExtendedQualityConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    /// Assess multi-dimensional quality
    pub fn assess_multi_dimensional_quality(
        &mut self,
        store: &dyn Store,
        shapes: &[Shape],
        validation_report: Option<&ValidationReport>,
    ) -> Result<MultiDimensionalQualityAssessment> {
        tracing::info!("Starting multi-dimensional quality assessment");
        let start_time = Instant::now();

        // Scan the store once and derive shared evidence for every dimension.
        let evidence = DimensionEvidence::scan(store, shapes, validation_report)?;

        let intrinsic_quality = if self.config.enable_intrinsic_quality {
            self.assess_intrinsic_quality(&evidence)
        } else {
            IntrinsicQualityAssessment::default()
        };

        let contextual_quality = if self.config.enable_contextual_quality {
            self.assess_contextual_quality(&evidence)
        } else {
            ContextualQualityAssessment::default()
        };

        let statistical_measures = if self.config.enable_statistical_measures {
            self.compute_statistical_measures(store)?
        } else {
            StatisticalQualityMeasures::default()
        };

        let semantic_measures = if self.config.enable_semantic_measures {
            self.compute_semantic_measures(&evidence)
        } else {
            SemanticQualityMeasures::default()
        };

        let overall_quality_score = self.calculate_overall_quality_score(
            &intrinsic_quality,
            &contextual_quality,
            &statistical_measures,
            &semantic_measures,
        );

        let confidence_score = self.calculate_confidence_score(
            &intrinsic_quality,
            &contextual_quality,
            &statistical_measures,
            &semantic_measures,
        );

        tracing::info!(
            "Multi-dimensional quality assessment completed in {:?}. Overall score: {:.3}",
            start_time.elapsed(),
            overall_quality_score
        );

        Ok(MultiDimensionalQualityAssessment {
            intrinsic_quality,
            contextual_quality,
            statistical_measures,
            semantic_measures,
            overall_quality_score,
            assessment_timestamp: chrono::Utc::now(),
            confidence_score,
        })
    }

    /// Assess intrinsic quality dimensions from real store evidence.
    fn assess_intrinsic_quality(&self, evidence: &DimensionEvidence) -> IntrinsicQualityAssessment {
        tracing::debug!("Assessing intrinsic quality dimensions");
        IntrinsicQualityAssessment {
            accuracy: evidence.accuracy_dimension(),
            consistency: evidence.consistency_dimension(),
            completeness: evidence.completeness_dimension(),
            validity: evidence.validity_dimension(),
            precision: evidence.precision_dimension(),
            currency: evidence.currency_dimension(self.config.currency_window_days),
        }
    }

    /// Assess contextual quality dimensions from real store evidence.
    fn assess_contextual_quality(
        &self,
        evidence: &DimensionEvidence,
    ) -> ContextualQualityAssessment {
        tracing::debug!("Assessing contextual quality dimensions");
        ContextualQualityAssessment {
            relevance: evidence.relevance_dimension(),
            timeliness: evidence.timeliness_dimension(self.config.currency_window_days),
            accessibility: evidence.accessibility_dimension(),
            compliance: evidence.compliance_dimension(),
            security: evidence.security_dimension(),
            usability: evidence.usability_dimension(),
        }
    }

    /// Compute statistical quality measures from a single real store scan.
    fn compute_statistical_measures(
        &mut self,
        store: &dyn Store,
    ) -> Result<StatisticalQualityMeasures> {
        tracing::debug!("Computing statistical quality measures");

        let scan = StoreScanStats::scan(store)?;
        let entropy_calculation = scan.entropy_calculation();
        let information_content = scan.information_content(&entropy_calculation);
        let correlation_analysis = CorrelationAnalysis {
            property_correlations: HashMap::new(),
            type_correlations: HashMap::new(),
            strongest_correlations: vec![],
            correlation_network_density: scan.correlation_network_density(),
        };

        Ok(StatisticalQualityMeasures {
            distribution_analysis: scan.distribution_analysis(),
            outlier_detection: scan.outlier_detection(),
            correlation_analysis,
            entropy_calculation,
            information_content,
            redundancy_assessment: scan.redundancy_assessment(),
        })
    }

    /// Compute semantic quality measures from real store evidence.
    fn compute_semantic_measures(&self, evidence: &DimensionEvidence) -> SemanticQualityMeasures {
        tracing::debug!("Computing semantic quality measures");
        SemanticQualityMeasures {
            concept_coherence: evidence.concept_coherence(),
            relationship_validity: evidence.relationship_validity(),
            taxonomy_consistency: evidence.taxonomy_consistency(),
            semantic_density: evidence.semantic_density(),
            knowledge_completeness: evidence.knowledge_completeness(),
            logical_consistency: evidence.logical_consistency(),
        }
    }

    /// Calculate overall quality score
    fn calculate_overall_quality_score(
        &self,
        intrinsic: &IntrinsicQualityAssessment,
        contextual: &ContextualQualityAssessment,
        statistical: &StatisticalQualityMeasures,
        semantic: &SemanticQualityMeasures,
    ) -> f64 {
        let intrinsic_score = (intrinsic.accuracy.score
            + intrinsic.consistency.score
            + intrinsic.completeness.score
            + intrinsic.validity.score
            + intrinsic.precision.score
            + intrinsic.currency.score)
            / 6.0;

        let contextual_score = (contextual.relevance.score
            + contextual.timeliness.score
            + contextual.accessibility.score
            + contextual.compliance.score
            + contextual.security.score
            + contextual.usability.score)
            / 6.0;

        let statistical_score = (statistical.distribution_analysis.normality_score
            + (1.0 - statistical.outlier_detection.outlier_percentage / 100.0)
            + statistical.correlation_analysis.correlation_network_density
            + (statistical.entropy_calculation.shannon_entropy / 10.0)
            + statistical.information_content.information_quality_score
            + (1.0 - statistical.redundancy_assessment.redundancy_ratio))
            / 6.0;

        let semantic_score = (semantic.concept_coherence.coherence_score
            + semantic.relationship_validity.validity_score
            + semantic.taxonomy_consistency.consistency_score
            + semantic.semantic_density.density_score
            + semantic.knowledge_completeness.completeness_score
            + semantic.logical_consistency.consistency_score)
            / 6.0;

        // Weighted combination
        intrinsic_score * 0.4
            + contextual_score * 0.25
            + statistical_score * 0.2
            + semantic_score * 0.15
    }

    /// Calculate confidence score
    fn calculate_confidence_score(
        &self,
        intrinsic: &IntrinsicQualityAssessment,
        contextual: &ContextualQualityAssessment,
        _statistical: &StatisticalQualityMeasures,
        _semantic: &SemanticQualityMeasures,
    ) -> f64 {
        let intrinsic_confidence = (intrinsic.accuracy.confidence
            + intrinsic.consistency.confidence
            + intrinsic.completeness.confidence
            + intrinsic.validity.confidence
            + intrinsic.precision.confidence
            + intrinsic.currency.confidence)
            / 6.0;

        let contextual_confidence = (contextual.relevance.confidence
            + contextual.timeliness.confidence
            + contextual.accessibility.confidence
            + contextual.compliance.confidence
            + contextual.security.confidence
            + contextual.usability.confidence)
            / 6.0;

        (intrinsic_confidence + contextual_confidence) / 2.0
    }
}

impl Default for ExtendedQualityDimensionsAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations for assessment results

impl Default for StatisticalQualityMeasures {
    fn default() -> Self {
        Self {
            distribution_analysis: DistributionAnalysis {
                distribution_type: DistributionType::Unknown,
                parameters: HashMap::new(),
                normality_score: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                uniformity_score: 0.0,
            },
            outlier_detection: OutlierDetectionResult {
                outlier_count: 0,
                outlier_percentage: 0.0,
                outlier_types: vec![],
                outlier_confidence: 0.0,
                detection_method: "None".to_string(),
            },
            correlation_analysis: CorrelationAnalysis {
                property_correlations: HashMap::new(),
                type_correlations: HashMap::new(),
                strongest_correlations: vec![],
                correlation_network_density: 0.0,
            },
            entropy_calculation: EntropyCalculation {
                shannon_entropy: 0.0,
                conditional_entropy: 0.0,
                mutual_information: 0.0,
                entropy_variance: 0.0,
                information_gain: 0.0,
            },
            information_content: InformationContentMeasure {
                total_information_content: 0.0,
                unique_information_ratio: 0.0,
                information_density: 0.0,
                redundant_information_ratio: 0.0,
                information_quality_score: 0.0,
            },
            redundancy_assessment: RedundancyAssessment {
                redundancy_ratio: 0.0,
                duplicate_triples: 0,
                redundant_patterns: vec![],
                redundancy_impact: 0.0,
            },
        }
    }
}

impl Default for SemanticQualityMeasures {
    fn default() -> Self {
        Self {
            concept_coherence: ConceptCoherence {
                coherence_score: 0.0,
                concept_clusters: vec![],
                incoherent_concepts: vec![],
                semantic_similarity_matrix: HashMap::new(),
            },
            relationship_validity: RelationshipValidity {
                validity_score: 0.0,
                valid_relationships: 0,
                invalid_relationships: 0,
                suspicious_relationships: vec![],
            },
            taxonomy_consistency: TaxonomyConsistency {
                consistency_score: 0.0,
                hierarchy_violations: vec![],
                circular_dependencies: vec![],
                taxonomy_depth_analysis: TaxonomyDepthAnalysis {
                    max_depth: 0,
                    average_depth: 0.0,
                    depth_variance: 0.0,
                    optimal_depth_recommendation: 0,
                },
            },
            semantic_density: SemanticDensity {
                density_score: 0.0,
                relationship_density: 0.0,
                property_density: 0.0,
                type_density: 0.0,
                sparse_areas: vec![],
                dense_areas: vec![],
            },
            knowledge_completeness: KnowledgeCompleteness {
                completeness_score: 0.0,
                domain_coverage: DomainCoverage {
                    covered_domains: vec![],
                    coverage_percentage: 0.0,
                    domain_completeness: HashMap::new(),
                    interdomain_connections: 0,
                },
                missing_knowledge_areas: vec![],
                knowledge_gaps: vec![],
            },
            logical_consistency: LogicalConsistency {
                consistency_score: 0.0,
                logical_violations: vec![],
                inconsistent_axioms: vec![],
                reasoning_errors: vec![],
            },
        }
    }
}

impl Default for QualityDimensionResult {
    fn default() -> Self {
        Self {
            score: 0.0,
            confidence: 0.0,
            details: QualityDimensionDetails {
                measurement_method: "Not assessed".to_string(),
                sample_size: 0,
                measurement_timestamp: chrono::Utc::now(),
                metadata: HashMap::new(),
            },
            issues: vec![],
            recommendations: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Real, store-derived evidence for quality-dimension assessment
// ---------------------------------------------------------------------------

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const SKOS_PREFLABEL: &str = "http://www.w3.org/2004/02/skos/core#prefLabel";
const RDFS_SUBCLASSOF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
const RDF_LANGSTRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

/// Confidence that scales with the amount of supporting evidence (`n` samples).
fn evidence_confidence(n: usize) -> f64 {
    (n as f64 / (n as f64 + 30.0)).clamp(0.0, 1.0)
}

/// Build a [`QualityDimensionResult`] with real measurement metadata.
fn dim_result(
    score: f64,
    confidence: f64,
    method: &str,
    sample_size: usize,
) -> QualityDimensionResult {
    QualityDimensionResult {
        score: score.clamp(0.0, 1.0),
        confidence: confidence.clamp(0.0, 1.0),
        details: QualityDimensionDetails {
            measurement_method: method.to_string(),
            sample_size,
            measurement_timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        },
        issues: vec![],
        recommendations: vec![],
    }
}

/// Aggregated real evidence from a single scan of the store, shared across all
/// quality-dimension assessors.
#[derive(Debug, Default)]
struct DimensionEvidence {
    total_triples: usize,
    subjects: HashSet<String>,
    named_subjects: HashSet<String>,
    blank_subjects: HashSet<String>,
    subjects_with_type: HashSet<String>,
    subjects_with_label: HashSet<String>,
    subject_predicate_counts: HashMap<String, usize>,
    predicate_counts: HashMap<String, usize>,
    /// Per-predicate datatype -> count, for datatype-consistency measurement.
    predicate_datatypes: HashMap<String, HashMap<String, usize>>,
    total_literals: usize,
    typed_literals: usize,
    typed_valid: usize,
    numeric_decimals: Vec<usize>,
    datetimes: Vec<chrono::DateTime<chrono::Utc>>,
    pii_literals: usize,
    relationship_total: usize,
    /// Blank-node object references (for referential-integrity checks).
    blank_object_refs: Vec<String>,
    iri_object_refs: usize,
    /// (child, parent) rdfs:subClassOf edges.
    subclass_edges: Vec<(String, String)>,
    type_values: HashMap<String, usize>,
    /// Validation signal: (conforms, total_violations, error_violations).
    validation: Option<(bool, usize, usize)>,
}

impl DimensionEvidence {
    fn scan(
        store: &dyn Store,
        _shapes: &[Shape],
        validation_report: Option<&ValidationReport>,
    ) -> Result<Self> {
        let quads = store.find_quads(None, None, None, None).map_err(|e| {
            ShaclAiError::QualityAssessment(format!("failed to scan store for dimensions: {e}"))
        })?;

        let mut ev = DimensionEvidence {
            total_triples: quads.len(),
            ..Default::default()
        };

        for quad in &quads {
            let subject = quad.subject().as_str().to_string();
            let predicate = quad.predicate().as_str().to_string();

            ev.subjects.insert(subject.clone());
            if quad.subject().is_blank_node() {
                ev.blank_subjects.insert(subject.clone());
            } else {
                ev.named_subjects.insert(subject.clone());
            }
            *ev.subject_predicate_counts
                .entry(subject.clone())
                .or_insert(0) += 1;
            *ev.predicate_counts.entry(predicate.clone()).or_insert(0) += 1;

            match predicate.as_str() {
                RDF_TYPE => {
                    ev.subjects_with_type.insert(subject.clone());
                    if let Object::NamedNode(_) = quad.object() {
                        *ev.type_values
                            .entry(quad.object().as_str().to_string())
                            .or_insert(0) += 1;
                    }
                }
                RDFS_LABEL | SKOS_PREFLABEL => {
                    ev.subjects_with_label.insert(subject.clone());
                }
                RDFS_SUBCLASSOF => {
                    if let Object::NamedNode(_) = quad.object() {
                        ev.subclass_edges
                            .push((subject.clone(), quad.object().as_str().to_string()));
                    }
                }
                _ => {}
            }

            match quad.object() {
                Object::Literal(lit) => {
                    ev.total_literals += 1;
                    let datatype = lit.datatype().as_str().to_string();
                    let value = lit.value();

                    let is_plain = datatype == XSD_STRING || datatype == RDF_LANGSTRING;
                    if !is_plain {
                        ev.typed_literals += 1;
                        if is_valid_typed(&datatype, value) {
                            ev.typed_valid += 1;
                        }
                    }
                    *ev.predicate_datatypes
                        .entry(predicate.clone())
                        .or_default()
                        .entry(datatype.clone())
                        .or_insert(0) += 1;

                    if let Ok(num) = value.parse::<f64>() {
                        if num.is_finite() {
                            ev.numeric_decimals.push(decimal_places(value));
                        }
                    }
                    if let Some(dt) = parse_datetime(value) {
                        ev.datetimes.push(dt);
                    }
                    if looks_like_pii(value) {
                        ev.pii_literals += 1;
                    }
                }
                Object::NamedNode(_) => {
                    ev.relationship_total += 1;
                    ev.iri_object_refs += 1;
                }
                Object::BlankNode(_) => {
                    ev.relationship_total += 1;
                    ev.blank_object_refs
                        .push(quad.object().as_str().to_string());
                }
                _ => {}
            }
        }

        ev.validation = validation_report.map(|report| {
            let total = report.violations().len();
            let errors = report
                .violations()
                .iter()
                .filter(|v| v.result_severity == Severity::Violation)
                .count();
            (report.conforms(), total, errors)
        });

        Ok(ev)
    }

    fn distinct_subjects(&self) -> usize {
        self.subjects.len()
    }

    // --- Intrinsic dimensions ---

    fn accuracy_dimension(&self) -> QualityDimensionResult {
        let n = self.distinct_subjects();
        match self.validation {
            Some((conforms, _total, errors)) => {
                let score = if conforms {
                    1.0
                } else {
                    (1.0 - errors as f64 / n.max(1) as f64).clamp(0.0, 1.0)
                };
                dim_result(
                    score,
                    evidence_confidence(n),
                    "SHACL validation error ratio",
                    n,
                )
            }
            None => {
                // Without a validation report, fall back to datatype validity.
                let score = if self.typed_literals > 0 {
                    self.typed_valid as f64 / self.typed_literals as f64
                } else {
                    0.5
                };
                let conf = if self.typed_literals > 0 {
                    0.5 * evidence_confidence(self.typed_literals)
                } else {
                    0.0
                };
                dim_result(
                    score,
                    conf,
                    "datatype-validity proxy (no validation report)",
                    self.typed_literals,
                )
            }
        }
    }

    fn validity_dimension(&self) -> QualityDimensionResult {
        match self.validation {
            Some((conforms, total, _errors)) => {
                let n = self.distinct_subjects();
                let score = if conforms {
                    1.0
                } else {
                    (1.0 - total as f64 / n.max(1) as f64).clamp(0.0, 1.0)
                };
                dim_result(
                    score,
                    evidence_confidence(n),
                    "SHACL validation violation ratio",
                    n,
                )
            }
            None => {
                let score = if self.typed_literals > 0 {
                    self.typed_valid as f64 / self.typed_literals as f64
                } else {
                    0.5
                };
                dim_result(
                    score,
                    evidence_confidence(self.typed_literals),
                    "literal datatype validity",
                    self.typed_literals,
                )
            }
        }
    }

    fn consistency_dimension(&self) -> QualityDimensionResult {
        // For each predicate with literal values, the fraction sharing the
        // dominant datatype; weighted by literal count.
        let mut weighted = 0.0;
        let mut total = 0usize;
        for datatypes in self.predicate_datatypes.values() {
            let pred_total: usize = datatypes.values().sum();
            if pred_total == 0 {
                continue;
            }
            let dominant = datatypes.values().copied().max().unwrap_or(0);
            weighted += dominant as f64;
            total += pred_total;
        }
        let score = if total > 0 {
            weighted / total as f64
        } else {
            0.5
        };
        dim_result(
            score,
            evidence_confidence(self.predicate_datatypes.len()),
            "per-predicate datatype consistency",
            total,
        )
    }

    fn completeness_dimension(&self) -> QualityDimensionResult {
        // Uniformity of description: mean predicates-per-subject over the max.
        let counts: Vec<usize> = self.subject_predicate_counts.values().copied().collect();
        let score = if counts.is_empty() {
            0.5
        } else {
            let max = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
            let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
            (mean / max).clamp(0.0, 1.0)
        };
        dim_result(
            score,
            evidence_confidence(counts.len()),
            "predicate-coverage uniformity across subjects",
            counts.len(),
        )
    }

    fn precision_dimension(&self) -> QualityDimensionResult {
        if self.numeric_decimals.is_empty() {
            return dim_result(0.5, 0.0, "unmeasured (no numeric literals)", 0);
        }
        let mean_decimals =
            self.numeric_decimals.iter().sum::<usize>() as f64 / self.numeric_decimals.len() as f64;
        // More decimal places -> higher granularity; saturates around 4 places.
        let score = (mean_decimals / 4.0).clamp(0.0, 1.0);
        dim_result(
            score,
            evidence_confidence(self.numeric_decimals.len()),
            "numeric-literal decimal granularity",
            self.numeric_decimals.len(),
        )
    }

    fn currency_dimension(&self, window_days: u32) -> QualityDimensionResult {
        if self.datetimes.is_empty() {
            return dim_result(0.5, 0.0, "unmeasured (no temporal literals)", 0);
        }
        let now = chrono::Utc::now();
        let window = chrono::Duration::days(window_days as i64);
        let fresh = self
            .datetimes
            .iter()
            .filter(|dt| now.signed_duration_since(**dt) <= window)
            .count();
        let score = fresh as f64 / self.datetimes.len() as f64;
        dim_result(
            score,
            evidence_confidence(self.datetimes.len()),
            "fraction of timestamps within currency window",
            self.datetimes.len(),
        )
    }

    // --- Contextual dimensions ---

    fn relevance_dimension(&self) -> QualityDimensionResult {
        // Triples using predicates that appear only once are treated as low
        // relevance signal; relevance = 1 - their fraction.
        if self.total_triples == 0 {
            return dim_result(0.5, 0.0, "unmeasured (empty store)", 0);
        }
        let rare_triples: usize = self.predicate_counts.values().filter(|&&c| c == 1).count();
        let score = 1.0 - (rare_triples as f64 / self.total_triples as f64);
        dim_result(
            score,
            evidence_confidence(self.total_triples),
            "statistical predicate-frequency relevance",
            self.total_triples,
        )
    }

    fn timeliness_dimension(&self, window_days: u32) -> QualityDimensionResult {
        if self.datetimes.is_empty() {
            return dim_result(0.5, 0.0, "unmeasured (no temporal literals)", 0);
        }
        let now = chrono::Utc::now();
        let newest = self.datetimes.iter().max().copied().unwrap_or(now);
        let age_days = now.signed_duration_since(newest).num_days().max(0) as f64;
        let score = (1.0 - age_days / (window_days.max(1) as f64)).clamp(0.0, 1.0);
        dim_result(
            score,
            evidence_confidence(self.datetimes.len()),
            "recency of the most recent timestamp",
            self.datetimes.len(),
        )
    }

    fn accessibility_dimension(&self) -> QualityDimensionResult {
        let n = self.distinct_subjects();
        if n == 0 {
            return dim_result(0.5, 0.0, "unmeasured (empty store)", 0);
        }
        // Dereferenceable IRIs vs blank nodes.
        let score = self.named_subjects.len() as f64 / n as f64;
        dim_result(
            score,
            evidence_confidence(n),
            "fraction of dereferenceable (IRI) subjects",
            n,
        )
    }

    fn compliance_dimension(&self) -> QualityDimensionResult {
        if self.total_literals == 0 {
            return dim_result(0.5, 0.0, "unmeasured (no literals)", 0);
        }
        // Explicitly-typed literals as a standards-compliance proxy.
        let score = self.typed_literals as f64 / self.total_literals as f64;
        dim_result(
            score,
            evidence_confidence(self.total_literals),
            "fraction of explicitly-typed literals",
            self.total_literals,
        )
    }

    fn security_dimension(&self) -> QualityDimensionResult {
        if self.total_literals == 0 {
            return dim_result(0.5, 0.0, "unmeasured (no literals)", 0);
        }
        // Lower exposure of PII-looking literals => higher security score.
        let score = 1.0 - (self.pii_literals as f64 / self.total_literals as f64);
        dim_result(
            score,
            evidence_confidence(self.total_literals),
            "PII-exposure heuristic over literals",
            self.total_literals,
        )
    }

    fn usability_dimension(&self) -> QualityDimensionResult {
        let n = self.named_subjects.len();
        if n == 0 {
            return dim_result(0.5, 0.0, "unmeasured (no named subjects)", 0);
        }
        let labeled = self
            .named_subjects
            .iter()
            .filter(|s| self.subjects_with_label.contains(*s))
            .count();
        let score = labeled as f64 / n as f64;
        dim_result(
            score,
            evidence_confidence(n),
            "fraction of subjects with a human-readable label",
            n,
        )
    }

    // --- Semantic measures ---

    fn concept_coherence(&self) -> ConceptCoherence {
        // Coherence proxy: fraction of subjects grounded by an explicit type.
        let n = self.distinct_subjects();
        let coherence_score = if n > 0 {
            self.subjects_with_type.len() as f64 / n as f64
        } else {
            0.0
        };
        ConceptCoherence {
            coherence_score,
            concept_clusters: vec![],
            incoherent_concepts: vec![],
            semantic_similarity_matrix: HashMap::new(),
        }
    }

    fn relationship_validity(&self) -> RelationshipValidity {
        // Referential integrity: blank-node object references must resolve to a
        // subject; IRI references are considered valid (may be external).
        let mut invalid = 0usize;
        for target in &self.blank_object_refs {
            if !self.subjects.contains(target) {
                invalid += 1;
            }
        }
        let valid = self.relationship_total.saturating_sub(invalid);
        let validity_score = if self.relationship_total > 0 {
            valid as f64 / self.relationship_total as f64
        } else {
            1.0
        };
        RelationshipValidity {
            validity_score,
            valid_relationships: valid,
            invalid_relationships: invalid,
            suspicious_relationships: vec![],
        }
    }

    fn taxonomy_consistency(&self) -> TaxonomyConsistency {
        // Build child -> parents adjacency and detect cycles + depth.
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        let mut nodes: HashSet<String> = HashSet::new();
        for (child, parent) in &self.subclass_edges {
            adjacency
                .entry(child.clone())
                .or_default()
                .push(parent.clone());
            nodes.insert(child.clone());
            nodes.insert(parent.clone());
        }

        let (cycle_count, max_depth, avg_depth) = analyze_taxonomy(&adjacency, &nodes);
        let consistency_score = if nodes.is_empty() {
            1.0
        } else {
            (1.0 - cycle_count as f64 / nodes.len() as f64).clamp(0.0, 1.0)
        };

        TaxonomyConsistency {
            consistency_score,
            hierarchy_violations: vec![],
            circular_dependencies: vec![],
            taxonomy_depth_analysis: TaxonomyDepthAnalysis {
                max_depth,
                average_depth: avg_depth,
                depth_variance: 0.0,
                optimal_depth_recommendation: max_depth.min(6),
            },
        }
    }

    fn semantic_density(&self) -> SemanticDensity {
        let n = self.distinct_subjects().max(1) as f64;
        let relationship_density = if self.total_triples > 0 {
            self.relationship_total as f64 / self.total_triples as f64
        } else {
            0.0
        };
        let property_density = (self.predicate_counts.len() as f64 / n).clamp(0.0, 1.0);
        let type_density = self.subjects_with_type.len() as f64 / n;
        let density_score =
            ((relationship_density + property_density + type_density) / 3.0).clamp(0.0, 1.0);
        SemanticDensity {
            density_score,
            relationship_density,
            property_density,
            type_density,
            sparse_areas: vec![],
            dense_areas: vec![],
        }
    }

    fn knowledge_completeness(&self) -> KnowledgeCompleteness {
        let n = self.distinct_subjects();
        let completeness_score = if n > 0 {
            self.subjects_with_type.len() as f64 / n as f64
        } else {
            0.0
        };
        let mut covered_domains: Vec<String> = self.type_values.keys().cloned().collect();
        covered_domains.sort();
        let coverage_percentage = completeness_score * 100.0;
        KnowledgeCompleteness {
            completeness_score,
            domain_coverage: DomainCoverage {
                covered_domains,
                coverage_percentage,
                domain_completeness: HashMap::new(),
                interdomain_connections: self.relationship_total,
            },
            missing_knowledge_areas: vec![],
            knowledge_gaps: vec![],
        }
    }

    fn logical_consistency(&self) -> LogicalConsistency {
        // Combine validation conformance with taxonomy acyclicity.
        let validation_score = match self.validation {
            Some((conforms, total, _)) => {
                if conforms {
                    1.0
                } else {
                    (1.0 - total as f64 / self.distinct_subjects().max(1) as f64).clamp(0.0, 1.0)
                }
            }
            None => 1.0,
        };
        let taxonomy_score = self.taxonomy_consistency().consistency_score;
        let consistency_score = (validation_score + taxonomy_score) / 2.0;
        LogicalConsistency {
            consistency_score,
            logical_violations: vec![],
            inconsistent_axioms: vec![],
            reasoning_errors: vec![],
        }
    }
}

/// Whether a typed literal value is well-formed for its datatype.
fn is_valid_typed(datatype: &str, value: &str) -> bool {
    let local = datatype.rsplit('#').next().unwrap_or(datatype);
    match local {
        "integer" | "int" | "long" | "short" | "byte" | "nonNegativeInteger"
        | "positiveInteger" | "negativeInteger" | "nonPositiveInteger" | "unsignedInt"
        | "unsignedLong" => value.trim().parse::<i64>().is_ok(),
        "decimal" | "double" | "float" => value.trim().parse::<f64>().is_ok(),
        "boolean" => matches!(value.trim(), "true" | "false" | "0" | "1"),
        "dateTime" | "date" => parse_datetime(value).is_some(),
        // Strings, IRIs and unknown datatypes accept any lexical form.
        _ => true,
    }
}

/// Count of digits after the decimal point in a numeric lexical form.
fn decimal_places(value: &str) -> usize {
    match value.split_once('.') {
        Some((_, frac)) => frac.chars().take_while(|c| c.is_ascii_digit()).count(),
        None => 0,
    }
}

/// Parse an xsd:dateTime / xsd:date lexical form into a UTC timestamp.
fn parse_datetime(value: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let v = value.trim();
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(v) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(v, "%Y-%m-%d") {
        return date.and_hms_opt(0, 0, 0).map(|ndt| {
            chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt, chrono::Utc)
        });
    }
    None
}

/// Very small heuristic for PII-looking literals (email / phone number).
fn looks_like_pii(value: &str) -> bool {
    let v = value.trim();
    // Email: has '@' with a dotted domain after it.
    if let Some((_, domain)) = v.split_once('@') {
        if domain.contains('.') && !domain.starts_with('.') {
            return true;
        }
    }
    // Phone: mostly digits, 7..=15 of them.
    let digits = v.chars().filter(|c| c.is_ascii_digit()).count();
    let non_space = v.chars().filter(|c| !c.is_whitespace()).count();
    if (7..=15).contains(&digits) && digits * 2 >= non_space {
        return true;
    }
    false
}

/// Detect cycles and depth over a child->parents subclass adjacency.
/// Returns (cycle_node_count, max_depth, average_depth).
fn analyze_taxonomy(
    adjacency: &HashMap<String, Vec<String>>,
    nodes: &HashSet<String>,
) -> (usize, usize, f64) {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        Visiting,
        Done,
    }

    let mut depth_cache: HashMap<String, usize> = HashMap::new();
    let mut cycle_nodes: HashSet<String> = HashSet::new();

    // Iterative DFS computing depth (longest path to a root) with cycle marking.
    fn dfs(
        node: &str,
        adjacency: &HashMap<String, Vec<String>>,
        state: &mut HashMap<String, State>,
        depth_cache: &mut HashMap<String, usize>,
        cycle_nodes: &mut HashSet<String>,
    ) -> usize {
        if let Some(d) = depth_cache.get(node) {
            return *d;
        }
        match state.get(node) {
            Some(State::Visiting) => {
                cycle_nodes.insert(node.to_string());
                return 0;
            }
            Some(State::Done) => return depth_cache.get(node).copied().unwrap_or(0),
            None => {}
        }
        state.insert(node.to_string(), State::Visiting);
        let mut best = 0;
        if let Some(parents) = adjacency.get(node) {
            for parent in parents {
                let d = dfs(parent, adjacency, state, depth_cache, cycle_nodes) + 1;
                best = best.max(d);
            }
        }
        state.insert(node.to_string(), State::Done);
        depth_cache.insert(node.to_string(), best);
        best
    }

    // Convert the borrow-friendly &str map into owned-key form for recursion.
    let mut owned_state: HashMap<String, State> = HashMap::new();
    let mut max_depth = 0;
    let mut depth_sum = 0usize;
    for node in nodes {
        let d = dfs(
            node,
            adjacency,
            &mut owned_state,
            &mut depth_cache,
            &mut cycle_nodes,
        );
        max_depth = max_depth.max(d);
        depth_sum += d;
    }
    let avg_depth = if nodes.is_empty() {
        0.0
    } else {
        depth_sum as f64 / nodes.len() as f64
    };
    (cycle_nodes.len(), max_depth, avg_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_quality_assessor_creation() {
        let assessor = ExtendedQualityDimensionsAssessor::new();
        assert!(assessor.config.enable_intrinsic_quality);
        assert!(assessor.config.enable_contextual_quality);
        assert_eq!(assessor.config.precision_threshold, 0.8);
    }

    #[test]
    fn test_quality_dimension_result_default() {
        let result = QualityDimensionResult::default();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_multi_dimensional_assessment_structure() {
        let config = ExtendedQualityConfig::default();
        assert_eq!(config.currency_window_days, 30);
        assert_eq!(config.min_confidence, 0.7);
    }

    /// Regression: dimension scores must be derived from the real store, so a
    /// well-labeled dataset and a bare one produce different usability /
    /// accessibility scores — the old code returned fixed constants for every
    /// store.
    #[test]
    fn regression_dimensions_vary_with_store() {
        use oxirs_core::model::{Literal, NamedNode, Quad};
        use oxirs_core::ConcreteStore;

        let type_p = NamedNode::new(RDF_TYPE).expect("iri");
        let label_p = NamedNode::new(RDFS_LABEL).expect("iri");
        let person = NamedNode::new("http://example.org/Person").expect("iri");

        // Rich store: every subject is a typed, labeled IRI.
        let rich = ConcreteStore::new().expect("store");
        for i in 0..10 {
            let s = NamedNode::new(format!("http://example.org/s{i}")).expect("iri");
            rich.insert_quad(Quad::new_default_graph(
                s.clone(),
                type_p.clone(),
                person.clone(),
            ))
            .expect("insert");
            rich.insert_quad(Quad::new_default_graph(
                s,
                label_p.clone(),
                Literal::new(format!("Subject {i}")),
            ))
            .expect("insert");
        }

        // Bare store: subjects have a single opaque property, no type/label.
        let bare = ConcreteStore::new().expect("store");
        let p = NamedNode::new("http://example.org/p").expect("iri");
        for i in 0..10 {
            let s = NamedNode::new(format!("http://example.org/b{i}")).expect("iri");
            bare.insert_quad(Quad::new_default_graph(s, p.clone(), Literal::new("x")))
                .expect("insert");
        }

        let mut assessor = ExtendedQualityDimensionsAssessor::new();
        let rich_result = assessor
            .assess_multi_dimensional_quality(&rich, &[], None)
            .expect("assessment");
        let bare_result = assessor
            .assess_multi_dimensional_quality(&bare, &[], None)
            .expect("assessment");

        // Labels present -> usability is measured and positive for the rich store.
        assert!(
            rich_result.contextual_quality.usability.score > 0.9,
            "rich store usability should be high, got {}",
            rich_result.contextual_quality.usability.score
        );
        assert_eq!(
            bare_result.contextual_quality.usability.score, 0.0,
            "bare store has no labels -> zero usability"
        );
        // Typed subjects -> higher semantic coherence for the rich store.
        assert!(
            rich_result
                .semantic_measures
                .concept_coherence
                .coherence_score
                > bare_result
                    .semantic_measures
                    .concept_coherence
                    .coherence_score,
            "typed subjects should raise concept coherence"
        );
    }
}
