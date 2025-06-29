//! Evaluation metrics and benchmarking for embedding models

use crate::EmbeddingModel;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{info, warn, error};

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
            let ndcg = self.compute_ndcg(results, k);
            ndcg_at_k.insert(k, ndcg);
        }

        // Average Precision (simplified)
        let average_precision =
            results.iter().map(|r| r.reciprocal_rank).sum::<f64>() / results.len() as f64;

        // F1 Score (using Hits@1 as precision/recall approximation)
        let hits_at_1 = hits_at_k.get(&1).copied().unwrap_or(0.0);
        let f1_score = 2.0 * hits_at_1 * hits_at_1 / (hits_at_1 + hits_at_1 + 1e-10);

        EvaluationResults {
            mean_rank,
            mean_reciprocal_rank,
            hits_at_k,
            ndcg_at_k,
            average_precision,
            f1_score,
            num_test_triples: results.len(),
            evaluation_time_seconds: 0.0, // Will be set by caller
            detailed_results: results.to_vec(),
        }
    }

    /// Compute NDCG@K (simplified implementation)
    fn compute_ndcg(&self, results: &[TripleEvaluationResult], k: usize) -> f64 {
        // Simplified NDCG calculation
        // In practice, this would use proper relevance scores
        let dcg: f64 = results
            .iter()
            .filter(|r| r.rank <= k)
            .map(|r| 1.0 / (r.rank as f64).log2())
            .sum();

        let idcg = 1.0; // Ideal DCG for binary relevance

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
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
    pub async fn start_monitoring<M>(&mut self, model: Arc<M>, test_data: Vec<(String, String, String)>) -> Result<()>
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
                        info!("Quality monitoring snapshot created: MRR={:.4}", snapshot.metrics.mean_reciprocal_rank);
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
    fn detect_performance_drift(&self, current: &EvaluationResults, baseline: &EvaluationResults) -> Result<Vec<DriftAlert>> {
        let mut alerts = Vec::new();

        // Check MRR drift
        let mrr_degradation = ((baseline.mean_reciprocal_rank - current.mean_reciprocal_rank) / baseline.mean_reciprocal_rank) * 100.0;
        if mrr_degradation > self.config.drift_thresholds.mrr_degradation_threshold {
            alerts.push(DriftAlert::PerformanceDegradation {
                metric: "mean_reciprocal_rank".to_string(),
                current_value: current.mean_reciprocal_rank,
                baseline_value: baseline.mean_reciprocal_rank,
                degradation_percent: mrr_degradation,
            });
        }

        // Check Hits@10 drift
        if let (Some(&current_hits), Some(&baseline_hits)) = (
            current.hits_at_k.get(&10),
            baseline.hits_at_k.get(&10),
        ) {
            let hits_degradation = ((baseline_hits - current_hits) / baseline_hits) * 100.0;
            if hits_degradation > self.config.drift_thresholds.hits_at_10_degradation_threshold {
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
        let recent_mrr: Vec<f64> = recent_window.iter().map(|s| s.metrics.mean_reciprocal_rank).collect();
        let older_mrr: Vec<f64> = older_window.iter().map(|s| s.metrics.mean_reciprocal_rank).collect();

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
        let mrr_values: Vec<f64> = recent_data.iter().map(|(_, snapshot)| snapshot.metrics.mean_reciprocal_rank).collect();

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
            if slope < -0.001 { // Threshold for meaningful degradation
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
        self.quality_history.read().unwrap().iter().cloned().collect()
    }

    /// Get summary statistics
    pub fn get_summary_stats(&self) -> QualityMonitorSummary {
        let history = self.quality_history.read().unwrap();
        
        if history.is_empty() {
            return QualityMonitorSummary::default();
        }

        let mrr_values: Vec<f64> = history.iter().map(|s| s.metrics.mean_reciprocal_rank).collect();
        let mean_mrr = mrr_values.iter().sum::<f64>() / mrr_values.len() as f64;
        
        // Calculate standard deviation
        let variance = mrr_values.iter().map(|x| (x - mean_mrr).powi(2)).sum::<f64>() / mrr_values.len() as f64;
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
            DriftAlert::PerformanceDegradation { metric, current_value, baseline_value, degradation_percent } => {
                warn!(
                    "PERFORMANCE DEGRADATION ALERT: {} degraded by {:.2}% (current: {:.4}, baseline: {:.4})",
                    metric, degradation_percent, current_value, baseline_value
                );
            }
            DriftAlert::StatisticalDrift { metric, p_value, effect_size } => {
                warn!(
                    "STATISTICAL DRIFT ALERT: {} shows significant drift (p={:.3}, effect_size={:.3})",
                    metric, p_value, effect_size
                );
            }
            DriftAlert::TrendAlert { metric, trend_direction, trend_strength } => {
                warn!(
                    "TREND ALERT: {} is {} with strength {:.4}",
                    metric, trend_direction, trend_strength
                );
            }
            DriftAlert::DataQualityIssue { issue_type, description, severity } => {
                error!(
                    "DATA QUALITY ALERT [{:?}]: {} - {}",
                    severity, issue_type, description
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_suite() {
        let test_triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            (
                "bob".to_string(),
                "knows".to_string(),
                "charlie".to_string(),
            ),
        ];

        let validation_triples =
            vec![("alice".to_string(), "likes".to_string(), "bob".to_string())];

        let suite = EvaluationSuite::new(test_triples, validation_triples);
        assert_eq!(suite.test_triples.len(), 2);
        assert_eq!(suite.validation_triples.len(), 1);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();

        let results1 = EvaluationResults {
            mean_rank: 10.0,
            mean_reciprocal_rank: 0.5,
            hits_at_k: [(1, 0.3), (10, 0.8)].iter().cloned().collect(),
            ndcg_at_k: HashMap::new(),
            average_precision: 0.4,
            f1_score: 0.35,
            num_test_triples: 100,
            evaluation_time_seconds: 5.0,
            detailed_results: Vec::new(),
        };

        suite.add_evaluation("TransE".to_string(), results1);

        let report = suite.generate_report();
        assert_eq!(report.num_models, 1);
        assert_eq!(report.best_model, Some("TransE".to_string()));
    }
}
