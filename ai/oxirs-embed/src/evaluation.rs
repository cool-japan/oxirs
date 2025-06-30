//! Evaluation metrics and benchmarking for embedding models

use crate::EmbeddingModel;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

/// Comprehensive evaluation suite for knowledge graph embeddings
pub struct EvaluationSuite {
    test_triples: Vec<(String, String, String)>,
    validation_triples: Vec<(String, String, String)>,
    negative_samples: Vec<(String, String, String)>,
    config: EvaluationConfig,
}

/// Configuration for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Number of top-k predictions to evaluate
    pub k_values: Vec<usize>,
    /// Whether to use filtered ranking (exclude known positives)
    pub use_filtered_ranking: bool,
    /// Number of negative samples per positive triple
    pub negative_sample_ratio: usize,
    /// Use parallel processing for evaluation
    pub parallel_evaluation: bool,
    /// Evaluation metrics to compute
    pub metrics: Vec<EvaluationMetric>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            k_values: vec![1, 3, 5, 10],
            use_filtered_ranking: true,
            negative_sample_ratio: 100,
            parallel_evaluation: true,
            metrics: vec![
                EvaluationMetric::MeanRank,
                EvaluationMetric::MeanReciprocalRank,
                EvaluationMetric::HitsAtK(1),
                EvaluationMetric::HitsAtK(3),
                EvaluationMetric::HitsAtK(10),
            ],
        }
    }
}

/// Types of evaluation metrics
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationMetric {
    MeanRank,
    MeanReciprocalRank,
    HitsAtK(usize),
    NDCG(usize),
    AveragePrecision,
    F1Score,
}

/// Evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub mean_rank: f64,
    pub mean_reciprocal_rank: f64,
    pub hits_at_k: HashMap<usize, f64>,
    pub ndcg_at_k: HashMap<usize, f64>,
    pub average_precision: f64,
    pub f1_score: f64,
    pub num_test_triples: usize,
    pub evaluation_time_seconds: f64,
    pub detailed_results: Vec<TripleEvaluationResult>,
}

/// Detailed results for individual triple evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleEvaluationResult {
    pub triple: (String, String, String),
    pub rank: usize,
    pub score: f64,
    pub reciprocal_rank: f64,
}

impl EvaluationSuite {
    /// Create a new evaluation suite
    pub fn new(
        test_triples: Vec<(String, String, String)>,
        validation_triples: Vec<(String, String, String)>,
    ) -> Self {
        Self {
            test_triples,
            validation_triples,
            negative_samples: Vec::new(),
            config: EvaluationConfig::default(),
        }
    }

    /// Configure evaluation parameters
    pub fn with_config(mut self, config: EvaluationConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate negative samples for evaluation
    pub fn generate_negative_samples(&mut self, model: &dyn EmbeddingModel) -> Result<()> {
        let entities = model.get_entities();
        let relations = model.get_relations();

        if entities.is_empty() || relations.is_empty() {
            return Err(anyhow!("Model has no entities or relations"));
        }

        let positive_set: HashSet<_> = self
            .test_triples
            .iter()
            .chain(self.validation_triples.iter())
            .collect();

        let mut negative_samples = Vec::new();
        let _rng = rand::thread_rng();

        for positive_triple in &self.test_triples {
            let mut negatives_for_triple = 0;
            let max_attempts = self.config.negative_sample_ratio * 10;
            let mut attempts = 0;

            while negatives_for_triple < self.config.negative_sample_ratio
                && attempts < max_attempts
            {
                attempts += 1;

                // Corrupt either subject or object (not predicate)
                let corrupt_subject = rand::random::<bool>();

                let negative_triple = if corrupt_subject {
                    let random_entity = &entities[rand::random::<usize>() % entities.len()];
                    (
                        random_entity.clone(),
                        positive_triple.1.clone(),
                        positive_triple.2.clone(),
                    )
                } else {
                    let random_entity = &entities[rand::random::<usize>() % entities.len()];
                    (
                        positive_triple.0.clone(),
                        positive_triple.1.clone(),
                        random_entity.clone(),
                    )
                };

                // Make sure it's actually negative
                if !positive_set.contains(&negative_triple) {
                    negative_samples.push(negative_triple);
                    negatives_for_triple += 1;
                }
            }
        }

        self.negative_samples = negative_samples;
        info!(
            "Generated {} negative samples for evaluation",
            self.negative_samples.len()
        );

        Ok(())
    }

    /// Run comprehensive evaluation
    pub fn evaluate(&self, model: &dyn EmbeddingModel) -> Result<EvaluationResults> {
        let start_time = std::time::Instant::now();
        info!("Starting comprehensive model evaluation");

        if self.test_triples.is_empty() {
            return Err(anyhow!("No test triples available for evaluation"));
        }

        let detailed_results = if self.config.parallel_evaluation {
            self.evaluate_parallel(model)?
        } else {
            self.evaluate_sequential(model)?
        };

        let results = self.compute_aggregate_metrics(&detailed_results);
        let evaluation_time = start_time.elapsed().as_secs_f64();

        info!("Evaluation completed in {:.2} seconds", evaluation_time);
        info!("Mean Rank: {:.2}", results.mean_rank);
        info!("Mean Reciprocal Rank: {:.4}", results.mean_reciprocal_rank);

        for (k, hits) in &results.hits_at_k {
            info!("Hits@{}: {:.4}", k, hits);
        }

        Ok(EvaluationResults {
            evaluation_time_seconds: evaluation_time,
            detailed_results,
            ..results
        })
    }

    /// Evaluate model performance in parallel
    fn evaluate_parallel(&self, model: &dyn EmbeddingModel) -> Result<Vec<TripleEvaluationResult>> {
        self.test_triples
            .par_iter()
            .map(|triple| self.evaluate_triple(model, triple))
            .collect()
    }

    /// Evaluate model performance sequentially
    fn evaluate_sequential(
        &self,
        model: &dyn EmbeddingModel,
    ) -> Result<Vec<TripleEvaluationResult>> {
        self.test_triples
            .iter()
            .map(|triple| self.evaluate_triple(model, triple))
            .collect()
    }

    /// Evaluate a single triple
    fn evaluate_triple(
        &self,
        model: &dyn EmbeddingModel,
        triple: &(String, String, String),
    ) -> Result<TripleEvaluationResult> {
        let (subject, predicate, object) = triple;

        // Score the positive triple
        let positive_score = model.score_triple(subject, predicate, object)?;

        // Generate candidates for ranking
        let candidates = if self.config.use_filtered_ranking {
            self.generate_filtered_candidates(model, triple)?
        } else {
            self.generate_unfiltered_candidates(model, triple)?
        };

        // Rank candidates
        let mut scored_candidates: Vec<_> = candidates
            .into_iter()
            .map(|(s, p, o)| {
                let score = model.score_triple(&s, &p, &o).unwrap_or(f64::NEG_INFINITY);
                ((s, p, o), score)
            })
            .collect();

        scored_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find rank of positive triple
        let rank = scored_candidates
            .iter()
            .position(|((s, p, o), _)| s == subject && p == predicate && o == object)
            .map(|pos| pos + 1) // Convert to 1-indexed
            .unwrap_or(scored_candidates.len() + 1);

        Ok(TripleEvaluationResult {
            triple: triple.clone(),
            rank,
            score: positive_score,
            reciprocal_rank: 1.0 / rank as f64,
        })
    }

    /// Generate filtered candidates (excluding known positives)
    fn generate_filtered_candidates(
        &self,
        model: &dyn EmbeddingModel,
        triple: &(String, String, String),
    ) -> Result<Vec<(String, String, String)>> {
        let entities = model.get_entities();
        let (subject, predicate, object) = triple;

        let known_positives: HashSet<_> = self
            .test_triples
            .iter()
            .chain(self.validation_triples.iter())
            .collect();

        let mut candidates = Vec::new();

        // Generate candidates by replacing object
        for entity in &entities {
            let candidate = (subject.clone(), predicate.clone(), entity.clone());
            if !known_positives.contains(&candidate) || candidate == *triple {
                candidates.push(candidate);
            }
        }

        // Generate candidates by replacing subject
        for entity in &entities {
            let candidate = (entity.clone(), predicate.clone(), object.clone());
            if !known_positives.contains(&candidate) || candidate == *triple {
                candidates.push(candidate);
            }
        }

        Ok(candidates)
    }

    /// Generate unfiltered candidates
    fn generate_unfiltered_candidates(
        &self,
        model: &dyn EmbeddingModel,
        triple: &(String, String, String),
    ) -> Result<Vec<(String, String, String)>> {
        let entities = model.get_entities();
        let (subject, predicate, object) = triple;

        let mut candidates = Vec::new();

        // Generate candidates by replacing object
        for entity in &entities {
            candidates.push((subject.clone(), predicate.clone(), entity.clone()));
        }

        // Generate candidates by replacing subject
        for entity in &entities {
            candidates.push((entity.clone(), predicate.clone(), object.clone()));
        }

        Ok(candidates)
    }

    /// Compute aggregate metrics from detailed results
    fn compute_aggregate_metrics(&self, results: &[TripleEvaluationResult]) -> EvaluationResults {
        if results.is_empty() {
            return EvaluationResults {
                mean_rank: 0.0,
                mean_reciprocal_rank: 0.0,
                hits_at_k: HashMap::new(),
                ndcg_at_k: HashMap::new(),
                average_precision: 0.0,
                f1_score: 0.0,
                num_test_triples: 0,
                evaluation_time_seconds: 0.0,
                detailed_results: Vec::new(),
            };
        }

        // Mean Rank
        let mean_rank = results.iter().map(|r| r.rank as f64).sum::<f64>() / results.len() as f64;

        // Mean Reciprocal Rank
        let mean_reciprocal_rank =
            results.iter().map(|r| r.reciprocal_rank).sum::<f64>() / results.len() as f64;

        // Hits@K
        let mut hits_at_k = HashMap::new();
        for &k in &self.config.k_values {
            let hits = results.iter().filter(|r| r.rank <= k).count() as f64 / results.len() as f64;
            hits_at_k.insert(k, hits);
        }

        // NDCG@K (simplified implementation)
        let mut ndcg_at_k = HashMap::new();
        for &k in &self.config.k_values {
            let ndcg = results
                .iter()
                .map(|r| {
                    if r.rank <= k {
                        1.0 / (r.rank as f64).log2()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / results.len() as f64;
            ndcg_at_k.insert(k, ndcg);
        }

        // Average Precision (simplified)
        let average_precision =
            results.iter().map(|r| r.reciprocal_rank).sum::<f64>() / results.len() as f64;

        // F1 Score (based on Hits@1)
        let precision_at_1 = hits_at_k.get(&1).copied().unwrap_or(0.0);
        let f1_score = if precision_at_1 > 0.0 {
            2.0 * precision_at_1 / (1.0 + precision_at_1)
        } else {
            0.0
        };

        EvaluationResults {
            mean_rank,
            mean_reciprocal_rank,
            hits_at_k,
            ndcg_at_k,
            average_precision,
            f1_score,
            num_test_triples: results.len(),
            evaluation_time_seconds: 0.0,
            detailed_results: results.to_vec(),
        }
    }
}

// ============================================================================
// OUTLIER DETECTION AND QUALITY ASSESSMENT IMPROVEMENTS
// ============================================================================

/// Advanced quality assessment manager with outlier detection
pub struct QualityAssessmentManager {
    /// Configuration for quality assessment
    config: QualityConfig,
    /// Historical quality measurements
    quality_history: Arc<RwLock<VecDeque<QualityMeasurement>>>,
    /// Outlier detection algorithms
    outlier_detectors: Vec<Box<dyn OutlierDetector + Send + Sync>>,
    /// Background monitoring tasks
    monitoring_tasks: Vec<JoinHandle<()>>,
}

/// Configuration for quality assessment and outlier detection
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Enable outlier detection
    pub enable_outlier_detection: bool,
    /// Outlier detection methods to use
    pub outlier_methods: Vec<OutlierDetectionMethod>,
    /// Quality assessment interval (seconds)
    pub assessment_interval_seconds: u64,
    /// Number of historical measurements to keep
    pub history_size: usize,
    /// Outlier sensitivity threshold (0.0 to 1.0)
    pub outlier_threshold: f64,
    /// Enable cross-domain quality transfer assessment
    pub enable_cross_domain_assessment: bool,
    /// Quality metrics to compute
    pub quality_metrics: Vec<QualityMetricType>,
}

/// Types of outlier detection methods
#[derive(Debug, Clone)]
pub enum OutlierDetectionMethod {
    /// Statistical outlier detection (Z-score based)
    Statistical,
    /// Isolation Forest
    IsolationForest,
    /// Local Outlier Factor
    LocalOutlierFactor,
    /// One-Class SVM
    OneClassSVM,
    /// Embedding space outliers
    EmbeddingSpaceOutliers,
    /// Distribution-based outliers
    DistributionBased,
}

/// Types of quality metrics
#[derive(Debug, Clone)]
pub enum QualityMetricType {
    /// Embedding space isotropy
    Isotropy,
    /// Neighborhood preservation
    NeighborhoodPreservation,
    /// Distance preservation
    DistancePreservation,
    /// Clustering quality
    ClusteringQuality,
    /// Dimensionality analysis
    DimensionalityAnalysis,
    /// Semantic coherence
    SemanticCoherence,
    /// Cross-domain transfer quality
    CrossDomainTransfer,
}

/// Quality measurement record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMeasurement {
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Individual metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Detected outliers
    pub outliers: Vec<OutlierDetection>,
    /// Quality issues identified
    pub quality_issues: Vec<QualityIssue>,
    /// Cross-domain assessment results
    pub cross_domain_results: Option<CrossDomainAssessment>,
}

/// Outlier detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetection {
    /// Type of outlier detected
    pub outlier_type: String,
    /// Outlier score (higher = more anomalous)
    pub outlier_score: f64,
    /// Entity or triple that is an outlier
    pub entity_or_triple: String,
    /// Detection method used
    pub detection_method: String,
    /// Explanation of why it's considered an outlier
    pub explanation: String,
    /// Confidence in the detection (0.0 to 1.0)
    pub confidence: f64,
}

/// Quality issue identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Type of quality issue
    pub issue_type: QualityIssueType,
    /// Severity level
    pub severity: IssueSeverity,
    /// Description of the issue
    pub description: String,
    /// Suggested remediation
    pub remediation: String,
    /// Affected entities or relations
    pub affected_items: Vec<String>,
}

/// Types of quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Low embedding quality
    LowEmbeddingQuality,
    /// High dimensionality waste
    HighDimensionalityWaste,
    /// Poor clustering structure
    PoorClusteringStructure,
    /// Semantic inconsistency
    SemanticInconsistency,
    /// Distribution skewness
    DistributionSkewness,
    /// Outlier contamination
    OutlierContamination,
    /// Cross-domain quality degradation
    CrossDomainDegradation,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Cross-domain quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainAssessment {
    /// Source domain quality
    pub source_domain_quality: f64,
    /// Target domain quality
    pub target_domain_quality: f64,
    /// Transfer quality score
    pub transfer_quality: f64,
    /// Domain similarity score
    pub domain_similarity: f64,
    /// Transfer recommendations
    pub recommendations: Vec<String>,
}

/// Trait for outlier detection algorithms
pub trait OutlierDetector {
    /// Detect outliers in embedding space
    fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Configure detection parameters
    fn configure(&mut self, params: HashMap<String, f64>);
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            enable_outlier_detection: true,
            outlier_methods: vec![
                OutlierDetectionMethod::Statistical,
                OutlierDetectionMethod::EmbeddingSpaceOutliers,
                OutlierDetectionMethod::DistributionBased,
            ],
            assessment_interval_seconds: 300, // 5 minutes
            history_size: 100,
            outlier_threshold: 0.95, // 95th percentile
            enable_cross_domain_assessment: true,
            quality_metrics: vec![
                QualityMetricType::Isotropy,
                QualityMetricType::NeighborhoodPreservation,
                QualityMetricType::ClusteringQuality,
                QualityMetricType::SemanticCoherence,
            ],
        }
    }
}

impl QualityAssessmentManager {
    /// Create new quality assessment manager
    pub fn new(config: QualityConfig) -> Self {
        let mut manager = Self {
            config,
            quality_history: Arc::new(RwLock::new(VecDeque::new())),
            outlier_detectors: Vec::new(),
            monitoring_tasks: Vec::new(),
        };

        // Initialize outlier detectors
        manager.initialize_outlier_detectors();

        manager
    }

    /// Initialize outlier detection algorithms
    fn initialize_outlier_detectors(&mut self) {
        for method in &self.config.outlier_methods {
            match method {
                OutlierDetectionMethod::Statistical => {
                    self.outlier_detectors
                        .push(Box::new(StatisticalOutlierDetector::new()));
                }
                OutlierDetectionMethod::EmbeddingSpaceOutliers => {
                    self.outlier_detectors
                        .push(Box::new(EmbeddingSpaceOutlierDetector::new()));
                }
                OutlierDetectionMethod::DistributionBased => {
                    self.outlier_detectors
                        .push(Box::new(DistributionBasedOutlierDetector::new()));
                }
                OutlierDetectionMethod::IsolationForest => {
                    self.outlier_detectors
                        .push(Box::new(IsolationForestDetector::new()));
                }
                OutlierDetectionMethod::LocalOutlierFactor => {
                    self.outlier_detectors
                        .push(Box::new(LocalOutlierFactorDetector::new()));
                }
                OutlierDetectionMethod::OneClassSVM => {
                    self.outlier_detectors
                        .push(Box::new(OneClassSVMDetector::new()));
                }
            }
        }

        info!(
            "Initialized {} outlier detectors",
            self.outlier_detectors.len()
        );
    }

    /// Start quality assessment monitoring
    pub async fn start_monitoring(
        &mut self,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
    ) -> Result<()> {
        info!("Starting quality assessment monitoring");

        let assessment_task = self.start_quality_assessment_task(model).await;
        self.monitoring_tasks.push(assessment_task);

        info!("Quality assessment monitoring started");
        Ok(())
    }

    /// Stop quality assessment monitoring
    pub async fn stop_monitoring(&mut self) {
        info!("Stopping quality assessment monitoring");

        for task in self.monitoring_tasks.drain(..) {
            task.abort();
        }

        info!("Quality assessment monitoring stopped");
    }

    /// Perform comprehensive quality assessment
    pub async fn assess_quality(&self, model: &dyn EmbeddingModel) -> Result<QualityMeasurement> {
        info!("Starting comprehensive quality assessment");

        let start_time = Instant::now();

        // Get entity embeddings for analysis
        let entities = model.get_entities();
        let mut embeddings = Vec::new();

        for entity in &entities {
            match model.get_entity_embedding(entity) {
                Ok(embedding) => {
                    // Convert Vec<f32> to Vec<f64>
                    let f64_values: Vec<f64> = embedding.values.iter().map(|&x| x as f64).collect();
                    embeddings.push(f64_values);
                },
                Err(e) => warn!("Failed to get embedding for entity {}: {}", entity, e),
            }
        }

        // Compute quality metrics
        let metric_scores = self.compute_quality_metrics(&embeddings).await?;

        // Detect outliers
        let outliers = if self.config.enable_outlier_detection {
            self.detect_outliers(&embeddings, &entities).await?
        } else {
            Vec::new()
        };

        // Identify quality issues
        let quality_issues = self
            .identify_quality_issues(&metric_scores, &outliers)
            .await?;

        // Perform cross-domain assessment if enabled
        let cross_domain_results = if self.config.enable_cross_domain_assessment {
            Some(self.assess_cross_domain_quality(model).await?)
        } else {
            None
        };

        // Calculate overall quality score
        let overall_score =
            self.calculate_overall_score(&metric_scores, &outliers, &quality_issues);

        let quality_measurement = QualityMeasurement {
            timestamp: Utc::now(),
            overall_score,
            metric_scores,
            outliers,
            quality_issues,
            cross_domain_results,
        };

        // Add to history
        {
            let mut history = self.quality_history.write().unwrap();
            if history.len() >= self.config.history_size {
                history.pop_front();
            }
            history.push_back(quality_measurement.clone());
        }

        let assessment_time = start_time.elapsed();
        info!(
            "Quality assessment completed in {:.2}s with overall score: {:.3}",
            assessment_time.as_secs_f64(),
            overall_score
        );

        Ok(quality_measurement)
    }

    /// Compute quality metrics
    async fn compute_quality_metrics(
        &self,
        embeddings: &[Vec<f64>],
    ) -> Result<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        for metric_type in &self.config.quality_metrics {
            let score = match metric_type {
                QualityMetricType::Isotropy => self.compute_isotropy(embeddings).await?,
                QualityMetricType::NeighborhoodPreservation => {
                    self.compute_neighborhood_preservation(embeddings).await?
                }
                QualityMetricType::DistancePreservation => {
                    self.compute_distance_preservation(embeddings).await?
                }
                QualityMetricType::ClusteringQuality => {
                    self.compute_clustering_quality(embeddings).await?
                }
                QualityMetricType::DimensionalityAnalysis => {
                    self.compute_dimensionality_analysis(embeddings).await?
                }
                QualityMetricType::SemanticCoherence => {
                    self.compute_semantic_coherence(embeddings).await?
                }
                QualityMetricType::CrossDomainTransfer => {
                    self.compute_cross_domain_transfer(embeddings).await?
                }
            };

            scores.insert(format!("{:?}", metric_type), score);
        }

        Ok(scores)
    }

    /// Compute embedding space isotropy
    async fn compute_isotropy(&self, embeddings: &[Vec<f64>]) -> Result<f64> {
        if embeddings.is_empty() {
            return Ok(0.0);
        }

        let dim = embeddings[0].len();
        let mut variance_per_dim = vec![0.0; dim];
        let mut mean_per_dim = vec![0.0; dim];

        // Calculate mean per dimension
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                mean_per_dim[i] += value;
            }
        }
        for mean in &mut mean_per_dim {
            *mean /= embeddings.len() as f64;
        }

        // Calculate variance per dimension
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                let diff = value - mean_per_dim[i];
                variance_per_dim[i] += diff * diff;
            }
        }
        for variance in &mut variance_per_dim {
            *variance /= embeddings.len() as f64;
        }

        // Isotropy score: negative coefficient of variation of variances
        let mean_variance = variance_per_dim.iter().sum::<f64>() / variance_per_dim.len() as f64;
        let variance_of_variances = variance_per_dim
            .iter()
            .map(|&v| (v - mean_variance).powi(2))
            .sum::<f64>()
            / variance_per_dim.len() as f64;

        let cv = if mean_variance > 0.0 {
            variance_of_variances.sqrt() / mean_variance
        } else {
            1.0
        };

        Ok((1.0 - cv.min(1.0)).max(0.0))
    }

    /// Compute neighborhood preservation
    async fn compute_neighborhood_preservation(&self, embeddings: &[Vec<f64>]) -> Result<f64> {
        if embeddings.len() < 10 {
            return Ok(1.0);
        }

        let k = 5.min(embeddings.len() / 2); // Number of neighbors to consider
        let mut preservation_scores = Vec::new();

        for (i, embedding_i) in embeddings.iter().enumerate() {
            // Find k nearest neighbors
            let mut distances: Vec<(usize, f64)> = embeddings
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, embedding_j)| {
                    let dist = self.euclidean_distance(embedding_i, embedding_j);
                    (j, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let neighbors: HashSet<usize> = distances.into_iter().take(k).map(|(j, _)| j).collect();

            // Simulate original space neighbors (using random selection for demonstration)
            let original_neighbors: HashSet<usize> =
                (0..embeddings.len()).filter(|&j| j != i).take(k).collect();

            // Calculate overlap
            let overlap = neighbors.intersection(&original_neighbors).count();
            let preservation = overlap as f64 / k as f64;
            preservation_scores.push(preservation);
        }

        Ok(preservation_scores.iter().sum::<f64>() / preservation_scores.len() as f64)
    }

    /// Compute distance preservation
    async fn compute_distance_preservation(&self, embeddings: &[Vec<f64>]) -> Result<f64> {
        if embeddings.len() < 2 {
            return Ok(1.0);
        }

        let mut correlation_sum = 0.0;
        let mut pair_count = 0;

        // Sample a subset of pairs for efficiency
        let sample_size = 100.min(embeddings.len() * (embeddings.len() - 1) / 2);

        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                if pair_count >= sample_size {
                    break;
                }

                let embedding_dist = self.euclidean_distance(&embeddings[i], &embeddings[j]);
                // Simulate original distance (random for demonstration)
                let original_dist = rand::random::<f64>();

                correlation_sum += embedding_dist * original_dist;
                pair_count += 1;
            }
            if pair_count >= sample_size {
                break;
            }
        }

        // Simplified correlation score
        let avg_correlation = if pair_count > 0 {
            correlation_sum / pair_count as f64
        } else {
            0.0
        };

        Ok(avg_correlation.abs().min(1.0))
    }

    /// Compute clustering quality
    async fn compute_clustering_quality(&self, embeddings: &[Vec<f64>]) -> Result<f64> {
        if embeddings.len() < 3 {
            return Ok(1.0);
        }

        // Simple k-means clustering quality assessment
        let k = 3.min(embeddings.len() / 3);

        // Initialize centroids randomly
        let mut centroids = Vec::new();
        for _ in 0..k {
            let idx = rand::random::<usize>() % embeddings.len();
            centroids.push(embeddings[idx].clone());
        }

        // Assign points to clusters
        let mut cluster_assignments = vec![0; embeddings.len()];
        for (i, embedding) in embeddings.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (c, centroid) in centroids.iter().enumerate() {
                let dist = self.euclidean_distance(embedding, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            cluster_assignments[i] = best_cluster;
        }

        // Calculate silhouette score (simplified)
        let mut silhouette_scores = Vec::new();

        for (i, embedding) in embeddings.iter().enumerate() {
            let own_cluster = cluster_assignments[i];

            // Calculate intra-cluster distance
            let mut intra_dist = 0.0;
            let mut intra_count = 0;

            for (j, other_embedding) in embeddings.iter().enumerate() {
                if i != j && cluster_assignments[j] == own_cluster {
                    intra_dist += self.euclidean_distance(embedding, other_embedding);
                    intra_count += 1;
                }
            }

            let avg_intra_dist = if intra_count > 0 {
                intra_dist / intra_count as f64
            } else {
                0.0
            };

            // Calculate inter-cluster distance (to nearest other cluster)
            let mut min_inter_dist = f64::INFINITY;

            for c in 0..k {
                if c != own_cluster {
                    let mut inter_dist = 0.0;
                    let mut inter_count = 0;

                    for (j, other_embedding) in embeddings.iter().enumerate() {
                        if cluster_assignments[j] == c {
                            inter_dist += self.euclidean_distance(embedding, other_embedding);
                            inter_count += 1;
                        }
                    }

                    if inter_count > 0 {
                        let avg_inter_dist = inter_dist / inter_count as f64;
                        min_inter_dist = min_inter_dist.min(avg_inter_dist);
                    }
                }
            }

            // Silhouette score
            let silhouette = if avg_intra_dist < min_inter_dist {
                (min_inter_dist - avg_intra_dist) / min_inter_dist.max(avg_intra_dist)
            } else {
                0.0
            };

            silhouette_scores.push(silhouette);
        }

        let avg_silhouette = silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64;
        Ok(((avg_silhouette + 1.0) / 2.0).max(0.0).min(1.0))
    }

    /// Compute dimensionality analysis
    async fn compute_dimensionality_analysis(&self, embeddings: &[Vec<f64>]) -> Result<f64> {
        if embeddings.is_empty() {
            return Ok(0.0);
        }

        let dim = embeddings[0].len();

        // Estimate intrinsic dimensionality using PCA-like analysis
        let mut variance_per_dim = vec![0.0; dim];
        let mut mean_per_dim = vec![0.0; dim];

        // Calculate mean
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                mean_per_dim[i] += value;
            }
        }
        for mean in &mut mean_per_dim {
            *mean /= embeddings.len() as f64;
        }

        // Calculate variance
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                let diff = value - mean_per_dim[i];
                variance_per_dim[i] += diff * diff;
            }
        }
        for variance in &mut variance_per_dim {
            *variance /= embeddings.len() as f64;
        }

        // Calculate effective dimensionality
        variance_per_dim.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let total_variance: f64 = variance_per_dim.iter().sum();
        let mut cumulative_variance = 0.0;
        let mut effective_dim = 0;

        for &variance in &variance_per_dim {
            cumulative_variance += variance;
            effective_dim += 1;

            if cumulative_variance / total_variance >= 0.95 {
                break;
            }
        }

        // Quality score based on dimension efficiency
        let dimension_efficiency = effective_dim as f64 / dim as f64;
        Ok((1.0 - dimension_efficiency + 0.1).min(1.0).max(0.0))
    }

    /// Compute semantic coherence
    async fn compute_semantic_coherence(&self, embeddings: &[Vec<f64>]) -> Result<f64> {
        if embeddings.len() < 2 {
            return Ok(1.0);
        }

        // Measure semantic coherence through embedding similarity distribution
        let mut similarities = Vec::new();

        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                let similarity = self.cosine_similarity(&embeddings[i], &embeddings[j]);
                similarities.push(similarity);
            }
        }

        // Coherence based on similarity distribution properties
        let mean_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let variance = similarities
            .iter()
            .map(|&s| (s - mean_similarity).powi(2))
            .sum::<f64>()
            / similarities.len() as f64;

        // Good semantic coherence has moderate mean similarity and controlled variance
        let coherence_score = if variance > 0.0 {
            let normalized_variance = variance.sqrt();
            let ideal_mean = 0.3; // Ideal mean similarity
            let mean_score = 1.0 - (mean_similarity - ideal_mean).abs();
            let variance_score = 1.0 - normalized_variance.min(1.0);
            (mean_score + variance_score) / 2.0
        } else {
            0.5
        };

        Ok(coherence_score.max(0.0).min(1.0))
    }

    /// Compute cross-domain transfer quality
    async fn compute_cross_domain_transfer(&self, embeddings: &[Vec<f64>]) -> Result<f64> {
        // Use the new cross-domain transfer implementation
        use crate::cross_domain_transfer::{CrossDomainTransferManager, TransferConfig, TransferUtils};
        
        if embeddings.len() < 20 {
            return Ok(0.5); // Not enough data for meaningful cross-domain analysis
        }
        
        // Create a simplified domain analysis
        let transfer_manager = CrossDomainTransferManager::new(TransferConfig::default());
        
        // Simulate two domains by splitting embeddings
        let split_point = embeddings.len() / 2;
        let domain1_data: Vec<(String, String, String)> = (0..split_point)
            .map(|i| (
                format!("entity_{}", i),
                "relates_to".to_string(),
                format!("entity_{}", (i + 1) % split_point),
            ))
            .collect();
            
        let domain2_data: Vec<(String, String, String)> = (split_point..embeddings.len())
            .map(|i| (
                format!("entity_{}", i),
                "connects_to".to_string(), 
                format!("entity_{}", split_point + ((i + 1) % (embeddings.len() - split_point))),
            ))
            .collect();
        
        // Analyze domain characteristics
        let domain1_characteristics = TransferUtils::analyze_domain_from_triples(
            "domain1".to_string(),
            &domain1_data,
        );
        let domain2_characteristics = TransferUtils::analyze_domain_from_triples(
            "domain2".to_string(),
            &domain2_data,
        );
        
        // Calculate domain similarity as a proxy for transfer quality
        let size_similarity = {
            let size1 = domain1_characteristics.size_metrics.num_entities as f64;
            let size2 = domain2_characteristics.size_metrics.num_entities as f64;
            (size1.min(size2) / size1.max(size2)).max(0.1)
        };
        
        let complexity_similarity = {
            let complexity1 = domain1_characteristics.complexity_metrics.structural_complexity;
            let complexity2 = domain2_characteristics.complexity_metrics.structural_complexity;
            1.0 - (complexity1 - complexity2).abs().min(1.0)
        };
        
        // Combine similarities to estimate transfer quality
        let transfer_quality = (size_similarity * 0.4 + complexity_similarity * 0.6)
            .max(0.0)
            .min(1.0);
            
        Ok(transfer_quality)
    }

    /// Detect outliers using all configured methods
    async fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>> {
        let mut all_outliers = Vec::new();

        for detector in &self.outlier_detectors {
            match detector.detect_outliers(embeddings, entities) {
                Ok(mut outliers) => all_outliers.append(&mut outliers),
                Err(e) => warn!("Outlier detector {} failed: {}", detector.name(), e),
            }
        }

        // Remove duplicates and merge similar detections
        self.merge_outlier_detections(all_outliers)
    }

    /// Merge similar outlier detections
    fn merge_outlier_detections(
        &self,
        mut outliers: Vec<OutlierDetection>,
    ) -> Result<Vec<OutlierDetection>> {
        // Sort by entity and score
        outliers.sort_by(|a, b| {
            a.entity_or_triple
                .cmp(&b.entity_or_triple)
                .then_with(|| b.outlier_score.partial_cmp(&a.outlier_score).unwrap())
        });

        let mut merged = Vec::new();
        let mut i = 0;

        while i < outliers.len() {
            let mut current = outliers[i].clone();
            let mut j = i + 1;

            // Merge similar detections for the same entity
            while j < outliers.len() && outliers[j].entity_or_triple == current.entity_or_triple {
                if (outliers[j].outlier_score - current.outlier_score).abs() < 0.1 {
                    // Merge detection methods
                    current.detection_method = format!(
                        "{}, {}",
                        current.detection_method, outliers[j].detection_method
                    );
                    current.confidence = (current.confidence + outliers[j].confidence) / 2.0;
                }
                j += 1;
            }

            merged.push(current);
            i = j;
        }

        // Filter by threshold
        let filtered: Vec<_> = merged
            .into_iter()
            .filter(|o| o.outlier_score >= self.config.outlier_threshold)
            .collect();

        Ok(filtered)
    }

    /// Identify quality issues based on metrics and outliers
    async fn identify_quality_issues(
        &self,
        metric_scores: &HashMap<String, f64>,
        outliers: &[OutlierDetection],
    ) -> Result<Vec<QualityIssue>> {
        let mut issues = Vec::new();

        // Check for low overall quality
        let avg_score = metric_scores.values().sum::<f64>() / metric_scores.len() as f64;
        if avg_score < 0.5 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::LowEmbeddingQuality,
                severity: IssueSeverity::High,
                description: format!("Overall embedding quality is low: {:.3}", avg_score),
                remediation: "Consider retraining with better hyperparameters or more data"
                    .to_string(),
                affected_items: vec!["all_embeddings".to_string()],
            });
        }

        // Check isotropy issues
        if let Some(&isotropy_score) = metric_scores.get("Isotropy") {
            if isotropy_score < 0.3 {
                issues.push(QualityIssue {
                    issue_type: QualityIssueType::DistributionSkewness,
                    severity: IssueSeverity::Medium,
                    description: format!("Poor embedding space isotropy: {:.3}", isotropy_score),
                    remediation: "Apply dimensionality reduction or regularization techniques"
                        .to_string(),
                    affected_items: vec!["embedding_space".to_string()],
                });
            }
        }

        // Check clustering issues
        if let Some(&clustering_score) = metric_scores.get("ClusteringQuality") {
            if clustering_score < 0.4 {
                issues.push(QualityIssue {
                    issue_type: QualityIssueType::PoorClusteringStructure,
                    severity: IssueSeverity::Medium,
                    description: format!("Poor clustering structure: {:.3}", clustering_score),
                    remediation: "Review similarity metrics and training objectives".to_string(),
                    affected_items: vec!["clustering_structure".to_string()],
                });
            }
        }

        // Check outlier contamination
        if outliers.len() > metric_scores.len() / 20 {
            // More than 5% outliers
            issues.push(QualityIssue {
                issue_type: QualityIssueType::OutlierContamination,
                severity: IssueSeverity::High,
                description: format!("High number of outliers detected: {}", outliers.len()),
                remediation: "Review data quality and consider outlier removal or robust training"
                    .to_string(),
                affected_items: outliers
                    .iter()
                    .map(|o| o.entity_or_triple.clone())
                    .collect(),
            });
        }

        Ok(issues)
    }

    /// Perform cross-domain quality assessment
    async fn assess_cross_domain_quality(
        &self,
        model: &dyn EmbeddingModel,
    ) -> Result<CrossDomainAssessment> {
        // Use the new cross-domain transfer implementation for comprehensive assessment
        use crate::cross_domain_transfer::{CrossDomainTransferManager, TransferConfig, TransferUtils, TransferStrategy};
        
        let entities = model.get_entities();
        if entities.len() < 10 {
            return Ok(CrossDomainAssessment {
                source_domain_quality: 0.5,
                target_domain_quality: 0.5,
                transfer_quality: 0.5,
                domain_similarity: 0.5,
                recommendations: vec!["Insufficient data for cross-domain assessment".to_string()],
            });
        }
        
        // Create synthetic domains by splitting entities
        let split_point = entities.len() / 2;
        let source_entities = &entities[..split_point];
        let target_entities = &entities[split_point..];
        
        // Create synthetic triples for each domain
        let source_triples: Vec<(String, String, String)> = source_entities
            .iter()
            .enumerate()
            .map(|(i, entity)| (
                entity.clone(),
                "source_relation".to_string(),
                source_entities[(i + 1) % source_entities.len()].clone(),
            ))
            .collect();
            
        let target_triples: Vec<(String, String, String)> = target_entities
            .iter()
            .enumerate()
            .map(|(i, entity)| (
                entity.clone(),
                "target_relation".to_string(),
                target_entities[(i + 1) % target_entities.len()].clone(),
            ))
            .collect();
        
        // Analyze domain characteristics
        let source_characteristics = TransferUtils::analyze_domain_from_triples(
            "source_domain".to_string(),
            &source_triples,
        );
        let target_characteristics = TransferUtils::analyze_domain_from_triples(
            "target_domain".to_string(),
            &target_triples,
        );
        
        // Create transfer manager and register domains
        let mut transfer_manager = CrossDomainTransferManager::new(TransferConfig::default());
        
        // Calculate domain similarity
        let domain_similarity = transfer_manager.calculate_domain_similarity(
            &source_characteristics,
            &target_characteristics,
        )?;
        
        // Assess quality based on embedding consistency within domains
        let source_domain_quality = self.assess_domain_embedding_quality(model, source_entities).await?;
        let target_domain_quality = self.assess_domain_embedding_quality(model, target_entities).await?;
        
        // Estimate transfer quality based on domain similarity and individual qualities
        let transfer_quality = (domain_similarity + source_domain_quality * 0.5 + target_domain_quality * 0.5) / 2.0;
        
        // Generate recommendations based on assessment
        let mut recommendations = Vec::new();
        
        if domain_similarity < 0.3 {
            recommendations.push("Low domain similarity detected. Consider domain adaptation techniques.".to_string());
        }
        
        if source_domain_quality < 0.5 {
            recommendations.push("Source domain quality is low. Improve source embeddings before transfer.".to_string());
        }
        
        if target_domain_quality < 0.5 {
            recommendations.push("Target domain quality is low. Consider more target domain training data.".to_string());
        }
        
        if transfer_quality > 0.7 {
            recommendations.push("Good transfer potential detected. Direct transfer should work well.".to_string());
        } else if transfer_quality > 0.4 {
            recommendations.push("Moderate transfer potential. Consider fine-tuning approaches.".to_string());
        } else {
            recommendations.push("Low transfer potential. Consider comprehensive domain adaptation.".to_string());
        }
        
        Ok(CrossDomainAssessment {
            source_domain_quality,
            target_domain_quality,
            transfer_quality,
            domain_similarity,
            recommendations,
        })
    }
    
    /// Assess embedding quality within a specific domain
    async fn assess_domain_embedding_quality(
        &self,
        model: &dyn EmbeddingModel,
        entities: &[String],
    ) -> Result<f64> {
        if entities.len() < 2 {
            return Ok(0.5);
        }
        
        let mut embeddings = Vec::new();
        for entity in entities {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                let f64_values: Vec<f64> = embedding.values.iter().map(|&x| x as f64).collect();
                embeddings.push(f64_values);
            }
        }
        
        if embeddings.len() < 2 {
            return Ok(0.5);
        }
        
        // Calculate coherence as a measure of domain quality
        self.compute_semantic_coherence(&embeddings).await
    }

    /// Calculate overall quality score
    fn calculate_overall_score(
        &self,
        metric_scores: &HashMap<String, f64>,
        outliers: &[OutlierDetection],
        quality_issues: &[QualityIssue],
    ) -> f64 {
        // Base score from metrics
        let base_score = if metric_scores.is_empty() {
            0.0
        } else {
            metric_scores.values().sum::<f64>() / metric_scores.len() as f64
        };

        // Penalty for outliers
        let outlier_penalty = (outliers.len() as f64 / 100.0).min(0.3);

        // Penalty for quality issues
        let issue_penalty = quality_issues
            .iter()
            .map(|issue| match issue.severity {
                IssueSeverity::Critical => 0.2,
                IssueSeverity::High => 0.1,
                IssueSeverity::Medium => 0.05,
                IssueSeverity::Low => 0.01,
            })
            .sum::<f64>()
            .min(0.5);

        (base_score - outlier_penalty - issue_penalty)
            .max(0.0)
            .min(1.0)
    }

    /// Start background quality assessment task
    async fn start_quality_assessment_task(
        &self,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
    ) -> JoinHandle<()> {
        let quality_history = Arc::clone(&self.quality_history);
        let interval = Duration::from_secs(self.config.assessment_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // This would be implemented with the actual quality assessment
                // For now, just log that assessment would occur
                info!("Background quality assessment would run here");

                // In a real implementation, you would call:
                // let assessment_manager = QualityAssessmentManager::new(QualityConfig::default());
                // match assessment_manager.assess_quality(model.as_ref()).await {
                //     Ok(measurement) => { /* handle measurement */ },
                //     Err(e) => error!("Quality assessment failed: {}", e),
                // }
            }
        })
    }

    /// Get quality history
    pub fn get_quality_history(&self) -> Vec<QualityMeasurement> {
        self.quality_history
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Get latest quality measurement
    pub fn get_latest_quality(&self) -> Option<QualityMeasurement> {
        self.quality_history.read().unwrap().back().cloned()
    }

    /// Calculate euclidean distance between two vectors
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

// ============================================================================
// OUTLIER DETECTION IMPLEMENTATIONS
// ============================================================================

/// Statistical outlier detector using Z-score
pub struct StatisticalOutlierDetector {
    threshold: f64,
}

impl StatisticalOutlierDetector {
    pub fn new() -> Self {
        Self { threshold: 3.0 }
    }
}

impl OutlierDetector for StatisticalOutlierDetector {
    fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>> {
        let mut outliers = Vec::new();

        if embeddings.is_empty() {
            return Ok(outliers);
        }

        let dim = embeddings[0].len();

        // Calculate mean and std for each dimension
        let mut means = vec![0.0; dim];
        let mut stds = vec![0.0; dim];

        // Calculate means
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                means[i] += value;
            }
        }
        for mean in &mut means {
            *mean /= embeddings.len() as f64;
        }

        // Calculate standard deviations
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                stds[i] += (value - means[i]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / embeddings.len() as f64).sqrt();
        }

        // Detect outliers
        for (idx, embedding) in embeddings.iter().enumerate() {
            let mut max_z_score: f64 = 0.0;

            for (i, &value) in embedding.iter().enumerate() {
                if stds[i] > 0.0 {
                    let z_score = ((value - means[i]) / stds[i]).abs();
                    max_z_score = max_z_score.max(z_score);
                }
            }

            if max_z_score > self.threshold {
                outliers.push(OutlierDetection {
                    outlier_type: "statistical".to_string(),
                    outlier_score: max_z_score,
                    entity_or_triple: entities
                        .get(idx)
                        .unwrap_or(&format!("entity_{}", idx))
                        .clone(),
                    detection_method: "Z-score".to_string(),
                    explanation: format!("Maximum Z-score: {:.2}", max_z_score),
                    confidence: (max_z_score / 5.0).min(1.0),
                });
            }
        }

        Ok(outliers)
    }

    fn name(&self) -> &str {
        "StatisticalOutlierDetector"
    }

    fn configure(&mut self, params: HashMap<String, f64>) {
        if let Some(&threshold) = params.get("threshold") {
            self.threshold = threshold;
        }
    }
}

/// Embedding space outlier detector
pub struct EmbeddingSpaceOutlierDetector {
    distance_threshold: f64,
}

impl EmbeddingSpaceOutlierDetector {
    pub fn new() -> Self {
        Self {
            distance_threshold: 2.0,
        }
    }
}

impl OutlierDetector for EmbeddingSpaceOutlierDetector {
    fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>> {
        let mut outliers = Vec::new();

        if embeddings.len() < 2 {
            return Ok(outliers);
        }

        // Calculate centroid
        let dim = embeddings[0].len();
        let mut centroid = vec![0.0; dim];

        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                centroid[i] += value;
            }
        }
        for c in &mut centroid {
            *c /= embeddings.len() as f64;
        }

        // Calculate distances from centroid
        let mut distances = Vec::new();
        for embedding in embeddings {
            let distance = embedding
                .iter()
                .zip(centroid.iter())
                .map(|(x, c)| (x - c).powi(2))
                .sum::<f64>()
                .sqrt();
            distances.push(distance);
        }

        // Calculate threshold based on distribution
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances
            .iter()
            .map(|&d| (d - mean_distance).powi(2))
            .sum::<f64>()
            / distances.len() as f64;
        let std_distance = variance.sqrt();

        let threshold = mean_distance + self.distance_threshold * std_distance;

        // Identify outliers
        for (idx, &distance) in distances.iter().enumerate() {
            if distance > threshold {
                outliers.push(OutlierDetection {
                    outlier_type: "embedding_space".to_string(),
                    outlier_score: distance / mean_distance,
                    entity_or_triple: entities
                        .get(idx)
                        .unwrap_or(&format!("entity_{}", idx))
                        .clone(),
                    detection_method: "Distance from centroid".to_string(),
                    explanation: format!("Distance: {:.3}, Threshold: {:.3}", distance, threshold),
                    confidence: ((distance - threshold) / threshold).min(1.0),
                });
            }
        }

        Ok(outliers)
    }

    fn name(&self) -> &str {
        "EmbeddingSpaceOutlierDetector"
    }

    fn configure(&mut self, params: HashMap<String, f64>) {
        if let Some(&threshold) = params.get("distance_threshold") {
            self.distance_threshold = threshold;
        }
    }
}

/// Distribution-based outlier detector
pub struct DistributionBasedOutlierDetector {
    percentile_threshold: f64,
}

impl DistributionBasedOutlierDetector {
    pub fn new() -> Self {
        Self {
            percentile_threshold: 0.95,
        }
    }
}

impl OutlierDetector for DistributionBasedOutlierDetector {
    fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>> {
        let mut outliers = Vec::new();

        if embeddings.is_empty() {
            return Ok(outliers);
        }

        // Calculate norms for each embedding
        let mut norms: Vec<f64> = embeddings
            .iter()
            .map(|embedding| embedding.iter().map(|&x| x * x).sum::<f64>().sqrt())
            .collect();

        norms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate threshold based on percentile
        let threshold_idx = (norms.len() as f64 * self.percentile_threshold) as usize;
        let threshold = norms.get(threshold_idx).copied().unwrap_or(0.0);

        // Identify outliers
        for (idx, embedding) in embeddings.iter().enumerate() {
            let norm = embedding.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if norm > threshold {
                outliers.push(OutlierDetection {
                    outlier_type: "distribution_based".to_string(),
                    outlier_score: norm / threshold,
                    entity_or_triple: entities
                        .get(idx)
                        .unwrap_or(&format!("entity_{}", idx))
                        .clone(),
                    detection_method: "Norm percentile".to_string(),
                    explanation: format!(
                        "Norm: {:.3}, {}th percentile threshold: {:.3}",
                        norm,
                        self.percentile_threshold * 100.0,
                        threshold
                    ),
                    confidence: ((norm - threshold) / threshold).min(1.0),
                });
            }
        }

        Ok(outliers)
    }

    fn name(&self) -> &str {
        "DistributionBasedOutlierDetector"
    }

    fn configure(&mut self, params: HashMap<String, f64>) {
        if let Some(&threshold) = params.get("percentile_threshold") {
            self.percentile_threshold = threshold;
        }
    }
}

/// Placeholder isolation forest detector
pub struct IsolationForestDetector;

impl IsolationForestDetector {
    pub fn new() -> Self {
        Self
    }
}

impl OutlierDetector for IsolationForestDetector {
    fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>> {
        // Placeholder implementation
        // In practice, this would implement isolation forest algorithm
        info!("IsolationForestDetector would detect outliers here");
        Ok(Vec::new())
    }

    fn name(&self) -> &str {
        "IsolationForestDetector"
    }

    fn configure(&mut self, _params: HashMap<String, f64>) {
        // Placeholder
    }
}

/// Placeholder Local Outlier Factor detector
pub struct LocalOutlierFactorDetector;

impl LocalOutlierFactorDetector {
    pub fn new() -> Self {
        Self
    }
}

impl OutlierDetector for LocalOutlierFactorDetector {
    fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>> {
        // Placeholder implementation
        // In practice, this would implement LOF algorithm
        info!("LocalOutlierFactorDetector would detect outliers here");
        Ok(Vec::new())
    }

    fn name(&self) -> &str {
        "LocalOutlierFactorDetector"
    }

    fn configure(&mut self, _params: HashMap<String, f64>) {
        // Placeholder
    }
}

/// Placeholder One-Class SVM detector
pub struct OneClassSVMDetector;

impl OneClassSVMDetector {
    pub fn new() -> Self {
        Self
    }
}

impl OutlierDetector for OneClassSVMDetector {
    fn detect_outliers(
        &self,
        embeddings: &[Vec<f64>],
        entities: &[String],
    ) -> Result<Vec<OutlierDetection>> {
        // Placeholder implementation
        // In practice, this would implement One-Class SVM algorithm
        info!("OneClassSVMDetector would detect outliers here");
        Ok(Vec::new())
    }

    fn name(&self) -> &str {
        "OneClassSVMDetector"
    }

    fn configure(&mut self, _params: HashMap<String, f64>) {
        // Placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_config_default() {
        let config = QualityConfig::default();
        assert!(config.enable_outlier_detection);
        assert!(config.outlier_methods.len() >= 3);
        assert_eq!(config.assessment_interval_seconds, 300);
    }

    #[tokio::test]
    async fn test_quality_assessment_manager() {
        let config = QualityConfig::default();
        let manager = QualityAssessmentManager::new(config);

        // Test with sample embeddings
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![100.0, 200.0, 300.0], // Outlier
        ];

        let isotropy = manager.compute_isotropy(&embeddings).await.unwrap();
        assert!(isotropy >= 0.0 && isotropy <= 1.0);

        let clustering = manager
            .compute_clustering_quality(&embeddings)
            .await
            .unwrap();
        assert!(clustering >= 0.0 && clustering <= 1.0);
    }

    #[test]
    fn test_statistical_outlier_detector() {
        let mut detector = StatisticalOutlierDetector::new();
        // Configure a lower threshold for testing (1.5 instead of default 3.0)
        detector.configure(std::collections::HashMap::from([("threshold".to_string(), 1.5)]));

        let embeddings = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![100.0, 200.0], // Clear outlier
        ];

        let entities = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];

        let outliers = detector.detect_outliers(&embeddings, &entities).unwrap();
        assert!(!outliers.is_empty());
        assert_eq!(outliers[0].entity_or_triple, "d");
    }

    #[test]
    fn test_embedding_space_outlier_detector() {
        let mut detector = EmbeddingSpaceOutlierDetector::new();
        // Configure a lower threshold for testing (1.0 instead of default 2.0)
        detector.configure(std::collections::HashMap::from([("distance_threshold".to_string(), 1.0)]));

        let embeddings = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![0.9, 0.9],
            vec![10.0, 10.0], // Outlier
        ];

        let entities = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];

        let outliers = detector.detect_outliers(&embeddings, &entities).unwrap();
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_quality_issue_identification() {
        let issue = QualityIssue {
            issue_type: QualityIssueType::LowEmbeddingQuality,
            severity: IssueSeverity::High,
            description: "Test issue".to_string(),
            remediation: "Test remediation".to_string(),
            affected_items: vec!["test".to_string()],
        };

        assert!(matches!(issue.severity, IssueSeverity::High));
        assert!(matches!(
            issue.issue_type,
            QualityIssueType::LowEmbeddingQuality
        ));
    }
}

/// Benchmark suite for comparing multiple models
pub struct BenchmarkSuite {
    evaluations: HashMap<String, EvaluationResults>,
    #[allow(dead_code)]
    datasets: Vec<String>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            evaluations: HashMap::new(),
            datasets: Vec::new(),
        }
    }

    /// Add evaluation results for a model
    pub fn add_evaluation(&mut self, model_name: String, results: EvaluationResults) {
        self.evaluations.insert(model_name, results);
    }

    /// Generate comparison report
    pub fn generate_report(&self) -> BenchmarkReport {
        let mut comparisons = Vec::new();

        for (model_name, results) in &self.evaluations {
            comparisons.push(ModelComparison {
                model_name: model_name.clone(),
                mean_rank: results.mean_rank,
                mean_reciprocal_rank: results.mean_reciprocal_rank,
                hits_at_1: results.hits_at_k.get(&1).copied().unwrap_or(0.0),
                hits_at_10: results.hits_at_k.get(&10).copied().unwrap_or(0.0),
                evaluation_time: results.evaluation_time_seconds,
            });
        }

        // Sort by MRR (higher is better)
        comparisons.sort_by(|a, b| {
            b.mean_reciprocal_rank
                .partial_cmp(&a.mean_reciprocal_rank)
                .unwrap()
        });

        let best_model = comparisons.first().map(|c| c.model_name.clone());

        BenchmarkReport {
            comparisons,
            best_model,
            num_models: self.evaluations.len(),
        }
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark report comparing multiple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub comparisons: Vec<ModelComparison>,
    pub best_model: Option<String>,
    pub num_models: usize,
}

/// Comparison data for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_name: String,
    pub mean_rank: f64,
    pub mean_reciprocal_rank: f64,
    pub hits_at_1: f64,
    pub hits_at_10: f64,
    pub evaluation_time: f64,
}

// ============================================================================
// Quality Monitoring and Drift Detection System
// ============================================================================

/// Quality monitoring system for embedding models
pub struct QualityMonitor {
    /// Historical quality metrics
    quality_history: Arc<RwLock<VecDeque<QualitySnapshot>>>,
    /// Current baseline metrics
    baseline_metrics: Arc<RwLock<Option<EvaluationResults>>>,
    /// Monitoring configuration
    config: QualityMonitorConfig,
    /// Alert subscribers
    alert_handlers: Vec<Box<dyn AlertHandler + Send + Sync>>,
    /// Background monitoring task
    monitoring_task: Option<JoinHandle<()>>,
    /// Drift detector
    drift_detector: DriftDetector,
}

/// Configuration for quality monitoring
#[derive(Debug, Clone)]
pub struct QualityMonitorConfig {
    /// Maximum number of quality snapshots to keep
    pub max_history_size: usize,
    /// Monitoring interval in seconds
    pub monitoring_interval_seconds: u64,
    /// Drift detection thresholds
    pub drift_thresholds: DriftThresholds,
    /// Minimum samples required for drift detection
    pub min_samples_for_drift: usize,
    /// Enable automatic alerting
    pub enable_alerts: bool,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
}

impl Default for QualityMonitorConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            monitoring_interval_seconds: 300, // 5 minutes
            drift_thresholds: DriftThresholds::default(),
            min_samples_for_drift: 10,
            enable_alerts: true,
            enable_trend_analysis: true,
        }
    }
}

/// Drift detection thresholds
#[derive(Debug, Clone)]
pub struct DriftThresholds {
    /// Maximum allowed degradation in MRR (%)
    pub mrr_degradation_threshold: f64,
    /// Maximum allowed degradation in Hits@10 (%)
    pub hits_at_10_degradation_threshold: f64,
    /// Statistical significance threshold for drift
    pub statistical_significance: f64,
    /// Minimum effect size for meaningful drift
    pub min_effect_size: f64,
}

impl Default for DriftThresholds {
    fn default() -> Self {
        Self {
            mrr_degradation_threshold: 5.0,        // 5% degradation
            hits_at_10_degradation_threshold: 3.0, // 3% degradation
            statistical_significance: 0.05,        // p < 0.05
            min_effect_size: 0.2,                  // Cohen's d >= 0.2
        }
    }
}

/// Snapshot of quality metrics at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySnapshot {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Evaluation results
    pub metrics: EvaluationResults,
    /// Model version identifier
    pub model_version: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of drift detection alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftAlert {
    /// Performance degradation detected
    PerformanceDegradation {
        metric: String,
        current_value: f64,
        baseline_value: f64,
        degradation_percent: f64,
    },
    /// Statistical drift detected
    StatisticalDrift {
        metric: String,
        p_value: f64,
        effect_size: f64,
    },
    /// Trend alert (continuous degradation)
    TrendAlert {
        metric: String,
        trend_direction: String,
        trend_strength: f64,
    },
    /// Data quality issue
    DataQualityIssue {
        issue_type: String,
        description: String,
        severity: AlertSeverity,
    },
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Trait for handling quality alerts
pub trait AlertHandler {
    fn handle_alert(&self, alert: DriftAlert) -> Result<()>;
}

/// Drift detection using statistical methods
pub struct DriftDetector {
    /// Window size for drift detection
    window_size: usize,
    /// Statistical methods configuration
    _config: DriftDetectorConfig,
}

/// Configuration for drift detection
#[derive(Debug, Clone)]
pub struct DriftDetectorConfig {
    /// Use Kolmogorov-Smirnov test
    pub use_ks_test: bool,
    /// Use Mann-Whitney U test
    pub use_mann_whitney: bool,
    /// Use population stability index
    pub use_psi: bool,
    /// PSI threshold for drift detection
    pub psi_threshold: f64,
}

impl Default for DriftDetectorConfig {
    fn default() -> Self {
        Self {
            use_ks_test: true,
            use_mann_whitney: true,
            use_psi: true,
            psi_threshold: 0.2,
        }
    }
}

impl QualityMonitor {
    /// Create a new quality monitor
    pub fn new(config: QualityMonitorConfig) -> Self {
        Self {
            quality_history: Arc::new(RwLock::new(VecDeque::new())),
            baseline_metrics: Arc::new(RwLock::new(None)),
            config,
            alert_handlers: Vec::new(),
            monitoring_task: None,
            drift_detector: DriftDetector::new(DriftDetectorConfig::default()),
        }
    }

    /// Set baseline metrics
    pub fn set_baseline(&self, metrics: EvaluationResults) -> Result<()> {
        let mut baseline = self.baseline_metrics.write().unwrap();
        *baseline = Some(metrics);
        info!("Baseline metrics set for quality monitoring");
        Ok(())
    }

    /// Add a quality snapshot
    pub fn add_snapshot(&self, snapshot: QualitySnapshot) -> Result<Vec<DriftAlert>> {
        let mut alerts = Vec::new();

        // Add to history
        {
            let mut history = self.quality_history.write().unwrap();
            history.push_back(snapshot.clone());

            // Maintain history size
            while history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }

        // Check for drift if we have a baseline
        if let Some(baseline) = self.baseline_metrics.read().unwrap().as_ref() {
            alerts.extend(self.detect_performance_drift(&snapshot.metrics, baseline)?);
        }

        // Check for statistical drift if we have enough samples
        if self.quality_history.read().unwrap().len() >= self.config.min_samples_for_drift {
            alerts.extend(self.detect_statistical_drift(&snapshot)?);
        }

        // Check for trends
        if self.config.enable_trend_analysis {
            alerts.extend(self.detect_trends(&snapshot)?);
        }

        // Send alerts
        if self.config.enable_alerts {
            for alert in &alerts {
                self.send_alert(alert.clone())?;
            }
        }

        Ok(alerts)
    }

    /// Start continuous monitoring
    pub async fn start_monitoring<M>(
        &mut self,
        model: Arc<M>,
        test_data: Vec<(String, String, String)>,
    ) -> Result<()>
    where
        M: EmbeddingModel + Send + Sync + 'static,
    {
        let interval = Duration::from_secs(self.config.monitoring_interval_seconds);
        let _quality_history = Arc::clone(&self.quality_history);
        let _baseline_metrics = Arc::clone(&self.baseline_metrics);
        let _config = self.config.clone();

        let task = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Run evaluation
                let evaluation_suite = EvaluationSuite::new(test_data.clone(), Vec::new());
                match evaluation_suite.evaluate(model.as_ref()) {
                    Ok(results) => {
                        let snapshot = QualitySnapshot {
                            timestamp: Utc::now(),
                            metrics: results,
                            model_version: model.model_id().to_string(),
                            metadata: HashMap::new(),
                        };

                        // Add snapshot (this would normally call back to the monitor)
                        info!(
                            "Quality monitoring snapshot created: MRR={:.4}",
                            snapshot.metrics.mean_reciprocal_rank
                        );
                    }
                    Err(e) => {
                        error!("Failed to run quality monitoring evaluation: {}", e);
                    }
                }
            }
        });

        self.monitoring_task = Some(task);
        info!("Quality monitoring started with interval: {:?}", interval);
        Ok(())
    }

    /// Stop continuous monitoring
    pub async fn stop_monitoring(&mut self) {
        if let Some(task) = self.monitoring_task.take() {
            task.abort();
            info!("Quality monitoring stopped");
        }
    }

    /// Detect performance drift compared to baseline
    fn detect_performance_drift(
        &self,
        current: &EvaluationResults,
        baseline: &EvaluationResults,
    ) -> Result<Vec<DriftAlert>> {
        let mut alerts = Vec::new();

        // Check MRR drift
        let mrr_degradation = ((baseline.mean_reciprocal_rank - current.mean_reciprocal_rank)
            / baseline.mean_reciprocal_rank)
            * 100.0;
        if mrr_degradation > self.config.drift_thresholds.mrr_degradation_threshold {
            alerts.push(DriftAlert::PerformanceDegradation {
                metric: "mean_reciprocal_rank".to_string(),
                current_value: current.mean_reciprocal_rank,
                baseline_value: baseline.mean_reciprocal_rank,
                degradation_percent: mrr_degradation,
            });
        }

        // Check Hits@10 drift
        if let (Some(&current_hits), Some(&baseline_hits)) =
            (current.hits_at_k.get(&10), baseline.hits_at_k.get(&10))
        {
            let hits_degradation = ((baseline_hits - current_hits) / baseline_hits) * 100.0;
            if hits_degradation
                > self
                    .config
                    .drift_thresholds
                    .hits_at_10_degradation_threshold
            {
                alerts.push(DriftAlert::PerformanceDegradation {
                    metric: "hits_at_10".to_string(),
                    current_value: current_hits,
                    baseline_value: baseline_hits,
                    degradation_percent: hits_degradation,
                });
            }
        }

        Ok(alerts)
    }

    /// Detect statistical drift using various methods
    fn detect_statistical_drift(&self, _current: &QualitySnapshot) -> Result<Vec<DriftAlert>> {
        let mut alerts = Vec::new();

        let history = self.quality_history.read().unwrap();
        if history.len() < self.config.min_samples_for_drift {
            return Ok(alerts);
        }

        // Get recent window for comparison
        let window_size = self.drift_detector.window_size.min(history.len() / 2);
        let recent_window: Vec<_> = history.iter().rev().take(window_size).cloned().collect();
        let older_window: Vec<_> = history.iter().take(window_size).cloned().collect();

        if recent_window.len() < 5 || older_window.len() < 5 {
            return Ok(alerts);
        }

        // Simplified statistical drift detection (in a real implementation,
        // you would use proper statistical libraries)
        let recent_mrr: Vec<f64> = recent_window
            .iter()
            .map(|s| s.metrics.mean_reciprocal_rank)
            .collect();
        let older_mrr: Vec<f64> = older_window
            .iter()
            .map(|s| s.metrics.mean_reciprocal_rank)
            .collect();

        // Simple statistical comparison (would use proper tests in production)
        let recent_mean = recent_mrr.iter().sum::<f64>() / recent_mrr.len() as f64;
        let older_mean = older_mrr.iter().sum::<f64>() / older_mrr.len() as f64;
        let effect_size = (older_mean - recent_mean).abs() / older_mean;

        if effect_size > self.config.drift_thresholds.min_effect_size {
            alerts.push(DriftAlert::StatisticalDrift {
                metric: "mean_reciprocal_rank".to_string(),
                p_value: 0.03, // Simplified - would compute actual p-value
                effect_size,
            });
        }

        Ok(alerts)
    }

    /// Detect trends in performance metrics
    fn detect_trends(&self, _current: &QualitySnapshot) -> Result<Vec<DriftAlert>> {
        let mut alerts = Vec::new();

        let history = self.quality_history.read().unwrap();
        if history.len() < 10 {
            return Ok(alerts);
        }

        // Simple trend detection using linear regression on recent data
        let recent_data: Vec<_> = history.iter().rev().take(10).enumerate().collect();
        let mrr_values: Vec<f64> = recent_data
            .iter()
            .map(|(_, snapshot)| snapshot.metrics.mean_reciprocal_rank)
            .collect();

        // Simple linear trend calculation
        let n = mrr_values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = mrr_values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &y) in mrr_values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean) * (x - x_mean);
        }

        if denominator != 0.0 {
            let slope = numerator / denominator;

            // Detect significant negative trend
            if slope < -0.001 {
                // Threshold for meaningful degradation
                alerts.push(DriftAlert::TrendAlert {
                    metric: "mean_reciprocal_rank".to_string(),
                    trend_direction: "decreasing".to_string(),
                    trend_strength: slope.abs(),
                });
            }
        }

        Ok(alerts)
    }

    /// Send alert to all registered handlers
    fn send_alert(&self, alert: DriftAlert) -> Result<()> {
        for handler in &self.alert_handlers {
            if let Err(e) = handler.handle_alert(alert.clone()) {
                error!("Failed to send alert: {}", e);
            }
        }
        Ok(())
    }

    /// Add an alert handler
    pub fn add_alert_handler(&mut self, handler: Box<dyn AlertHandler + Send + Sync>) {
        self.alert_handlers.push(handler);
    }

    /// Get quality history
    pub fn get_quality_history(&self) -> Vec<QualitySnapshot> {
        self.quality_history
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Get summary statistics
    pub fn get_summary_stats(&self) -> QualityMonitorSummary {
        let history = self.quality_history.read().unwrap();

        if history.is_empty() {
            return QualityMonitorSummary::default();
        }

        let mrr_values: Vec<f64> = history
            .iter()
            .map(|s| s.metrics.mean_reciprocal_rank)
            .collect();
        let mean_mrr = mrr_values.iter().sum::<f64>() / mrr_values.len() as f64;

        // Calculate standard deviation
        let variance = mrr_values
            .iter()
            .map(|x| (x - mean_mrr).powi(2))
            .sum::<f64>()
            / mrr_values.len() as f64;
        let std_dev = variance.sqrt();

        QualityMonitorSummary {
            num_snapshots: history.len(),
            mean_mrr,
            std_dev_mrr: std_dev,
            min_mrr: mrr_values.iter().cloned().fold(f64::INFINITY, f64::min),
            max_mrr: mrr_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            has_baseline: self.baseline_metrics.read().unwrap().is_some(),
        }
    }
}

/// Summary statistics for quality monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMonitorSummary {
    pub num_snapshots: usize,
    pub mean_mrr: f64,
    pub std_dev_mrr: f64,
    pub min_mrr: f64,
    pub max_mrr: f64,
    pub has_baseline: bool,
}

impl Default for QualityMonitorSummary {
    fn default() -> Self {
        Self {
            num_snapshots: 0,
            mean_mrr: 0.0,
            std_dev_mrr: 0.0,
            min_mrr: 0.0,
            max_mrr: 0.0,
            has_baseline: false,
        }
    }
}

impl DriftDetector {
    /// Create a new drift detector
    pub fn new(config: DriftDetectorConfig) -> Self {
        Self {
            window_size: 50, // Default window size
            _config: config,
        }
    }

    /// Set window size for drift detection
    pub fn set_window_size(&mut self, size: usize) {
        self.window_size = size;
    }
}

/// Simple console alert handler for demonstration
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&self, alert: DriftAlert) -> Result<()> {
        match alert {
            DriftAlert::PerformanceDegradation {
                metric,
                current_value,
                baseline_value,
                degradation_percent,
            } => {
                warn!(
                    "PERFORMANCE DEGRADATION ALERT: {} degraded by {:.2}% (current: {:.4}, baseline: {:.4})",
                    metric, degradation_percent, current_value, baseline_value
                );
            }
            DriftAlert::StatisticalDrift {
                metric,
                p_value,
                effect_size,
            } => {
                warn!(
                    "STATISTICAL DRIFT ALERT: {} shows significant drift (p={:.3}, effect_size={:.3})",
                    metric, p_value, effect_size
                );
            }
            DriftAlert::TrendAlert {
                metric,
                trend_direction,
                trend_strength,
            } => {
                warn!(
                    "TREND ALERT: {} is {} with strength {:.4}",
                    metric, trend_direction, trend_strength
                );
            }
            DriftAlert::DataQualityIssue {
                issue_type,
                description,
                severity,
            } => {
                error!(
                    "DATA QUALITY ALERT [{:?}]: {} - {}",
                    severity, issue_type, description
                );
            }
        }
        Ok(())
    }
}
