//! Real-time inference pipeline for streaming RDF validation
//!
//! Processes incoming triples and predicts constraint violations with
//! configurable batching, caching and latency budgets.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::ensemble::{
    EdgeFeature, GraphFeatures, GraphStats, NodeFeature, ShapeLearnerEnsemble, ShapePrediction,
};
use crate::ShaclAiError;

// -------------------------------------------------------------------------
// Public data types
// -------------------------------------------------------------------------

/// The result of running inference on a single triple
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The triple that was evaluated
    pub triple: (String, String, String),
    /// Constraint violations predicted for this triple
    pub predicted_violations: Vec<PredictedViolation>,
    /// Wall-clock time taken to produce this result
    pub inference_time: Duration,
    /// Overall confidence in the violation set (0.0 = no confidence, 1.0 = certain)
    pub confidence: f64,
}

/// A single predicted constraint violation
#[derive(Debug, Clone)]
pub struct PredictedViolation {
    /// SHACL shape that owns the violated constraint
    pub shape_id: String,
    /// SHACL constraint component IRI (e.g. "sh:minCount")
    pub constraint_type: String,
    /// Severity score in [0.0, 1.0] – higher means more severe
    pub severity: f64,
    /// Human-readable explanation of why this violation was predicted
    pub explanation: String,
}

/// Configuration for the real-time inference pipeline
#[derive(Debug, Clone)]
pub struct InferencePipelineConfig {
    /// Maximum number of triples per inference batch
    pub batch_size: usize,
    /// Maximum acceptable latency in milliseconds before a partial batch is flushed
    pub max_latency_ms: u64,
    /// Minimum confidence required for a violation to be reported
    pub confidence_threshold: f64,
    /// Whether to cache inference results (keyed on the triple string)
    pub enable_caching: bool,
    /// How long a cached result remains valid
    pub cache_ttl: Duration,
}

impl Default for InferencePipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_latency_ms: 100,
            confidence_threshold: 0.7,
            enable_caching: true,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

/// Summary statistics for a running inference pipeline
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total number of triples processed since construction
    pub total_inferences: u64,
    /// Fraction of requests served from cache (0.0 – 1.0)
    pub cache_hit_rate: f64,
    /// Mean inference latency in milliseconds
    pub avg_latency_ms: f64,
    /// 95th-percentile latency in milliseconds (approximated)
    pub p95_latency_ms: f64,
    /// Total predicted violations emitted
    pub violations_predicted: u64,
}

// -------------------------------------------------------------------------
// Pipeline implementation
// -------------------------------------------------------------------------

/// Real-time inference pipeline that wraps a `ShapeLearnerEnsemble`
///
/// Triples are buffered into micro-batches.  When the batch reaches
/// `config.batch_size` triples, or when `flush()` is called, the batch is
/// sent to the ensemble for inference.  Results that pass the confidence
/// threshold are returned to the caller.
pub struct RealTimeInferencePipeline {
    config: InferencePipelineConfig,
    ensemble: Arc<Mutex<ShapeLearnerEnsemble>>,
    /// Buffered triples waiting for the next batch
    pending_batch: Vec<(String, String, String)>,
    /// Cache: triple_key -> (violations, expiry instant)
    inference_cache: HashMap<String, (Vec<PredictedViolation>, Instant)>,
    /// Timestamp of the oldest triple in the current pending batch
    batch_started_at: Option<Instant>,
    // Statistics accumulators
    total_inferences: u64,
    cache_hits: u64,
    violations_predicted: u64,
    latency_samples: Vec<f64>,
}

impl RealTimeInferencePipeline {
    /// Create a new pipeline wrapping the provided ensemble
    pub fn new(ensemble: ShapeLearnerEnsemble, config: InferencePipelineConfig) -> Self {
        Self {
            config,
            ensemble: Arc::new(Mutex::new(ensemble)),
            pending_batch: Vec::new(),
            inference_cache: HashMap::new(),
            batch_started_at: None,
            total_inferences: 0,
            cache_hits: 0,
            violations_predicted: 0,
            latency_samples: Vec::new(),
        }
    }

    /// Submit a single triple for inference.
    ///
    /// Returns `Some(InferenceResult)` immediately when the cache is hit or
    /// the batch fills up.  Returns `None` when the triple has been buffered
    /// but the batch is not yet full and the latency budget has not been
    /// exceeded.
    pub fn infer_triple(
        &mut self,
        s: &str,
        p: &str,
        o: &str,
    ) -> Result<Option<InferenceResult>, ShaclAiError> {
        let start = Instant::now();

        // --- Cache lookup ---
        if self.config.enable_caching {
            let cache_key = Self::cache_key(s, p, o);
            // Clone cached data first to release the immutable borrow before
            // we call mutable methods (record_latency, increment counters).
            let cached = self
                .inference_cache
                .get(&cache_key)
                .and_then(|(violations, expiry)| {
                    if Instant::now() < *expiry {
                        Some(violations.clone())
                    } else {
                        None
                    }
                });
            if let Some(cached_violations) = cached {
                self.cache_hits += 1;
                self.total_inferences += 1;
                let conf = Self::aggregate_confidence(&cached_violations);
                let elapsed = start.elapsed();
                self.record_latency(elapsed);
                return Ok(Some(InferenceResult {
                    triple: (s.to_string(), p.to_string(), o.to_string()),
                    predicted_violations: cached_violations,
                    inference_time: elapsed,
                    confidence: conf,
                }));
            }
        }

        // --- Buffer the triple ---
        if self.batch_started_at.is_none() {
            self.batch_started_at = Some(Instant::now());
        }
        self.pending_batch
            .push((s.to_string(), p.to_string(), o.to_string()));

        // --- Decide whether to flush ---
        let batch_full = self.pending_batch.len() >= self.config.batch_size;
        let latency_exceeded = self
            .batch_started_at
            .is_some_and(|t| t.elapsed() >= Duration::from_millis(self.config.max_latency_ms));

        if batch_full || latency_exceeded {
            let batch = std::mem::take(&mut self.pending_batch);
            self.batch_started_at = None;
            let mut results = self.process_batch(batch)?;
            // Return the result for the triple we just added (last in batch)
            return Ok(results.pop());
        }

        Ok(None)
    }

    /// Flush any pending triples in the buffer and run inference on them.
    ///
    /// Returns results for all flushed triples.
    pub fn flush(&mut self) -> Result<Vec<InferenceResult>, ShaclAiError> {
        if self.pending_batch.is_empty() {
            return Ok(Vec::new());
        }
        let batch = std::mem::take(&mut self.pending_batch);
        self.batch_started_at = None;
        self.process_batch(batch)
    }

    /// Return a snapshot of pipeline statistics
    pub fn pipeline_stats(&self) -> PipelineStats {
        let total = self.total_inferences;
        let cache_hit_rate = if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        };

        let avg_latency_ms = if self.latency_samples.is_empty() {
            0.0
        } else {
            self.latency_samples.iter().sum::<f64>() / self.latency_samples.len() as f64
        };

        let p95_latency_ms = self.approximate_p95();

        PipelineStats {
            total_inferences: total,
            cache_hit_rate,
            avg_latency_ms,
            p95_latency_ms,
            violations_predicted: self.violations_predicted,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Process a batch of triples through the ensemble
    fn process_batch(
        &mut self,
        batch: Vec<(String, String, String)>,
    ) -> Result<Vec<InferenceResult>, ShaclAiError> {
        let mut results = Vec::with_capacity(batch.len());

        let ensemble = Arc::clone(&self.ensemble);

        for triple in batch {
            let start = Instant::now();
            let (s, p, o) = &triple;

            let features = self.extract_triple_features(s, p, o);
            let predictions = {
                let guard = ensemble.lock().map_err(|e| {
                    ShaclAiError::ModelTraining(format!("Ensemble lock error: {e}"))
                })?;
                guard.predict(&features)?
            };

            let violations = self.predictions_to_violations(&predictions);
            let filtered: Vec<PredictedViolation> = violations
                .into_iter()
                .filter(|v| v.severity >= self.config.confidence_threshold)
                .collect();

            let conf = Self::aggregate_confidence(&filtered);
            let elapsed = start.elapsed();

            // Update cache
            if self.config.enable_caching {
                let key = Self::cache_key(s, p, o);
                let expiry = Instant::now() + self.config.cache_ttl;
                self.inference_cache.insert(key, (filtered.clone(), expiry));
            }

            self.violations_predicted += filtered.len() as u64;
            self.total_inferences += 1;
            self.record_latency(elapsed);

            results.push(InferenceResult {
                triple: (s.clone(), p.clone(), o.clone()),
                predicted_violations: filtered,
                inference_time: elapsed,
                confidence: conf,
            });
        }

        Ok(results)
    }

    /// Build a minimal `GraphFeatures` from a single (subject, predicate, object) triple
    fn extract_triple_features(&self, s: &str, p: &str, o: &str) -> GraphFeatures {
        // Determine if the object looks like a literal or an IRI
        let obj_is_literal = o.starts_with('"') || o.starts_with('\'');

        let subject_node = NodeFeature {
            node_id: s.to_string(),
            type_vec: vec![1.0, 0.0], // [is_subject, is_object]
            predicate_histogram: vec![1.0],
            degree_in: 0,
            degree_out: 1,
        };

        let object_node = NodeFeature {
            node_id: o.to_string(),
            type_vec: vec![0.0, 1.0],
            predicate_histogram: vec![1.0],
            degree_in: 1,
            degree_out: 0,
        };

        let edge = EdgeFeature {
            predicate: p.to_string(),
            source_type: "Subject".to_string(),
            target_type: if obj_is_literal { "Literal" } else { "IRI" }.to_string(),
            frequency: 1.0,
        };

        GraphFeatures {
            node_features: vec![subject_node, object_node],
            edge_features: vec![edge],
            graph_stats: GraphStats {
                node_count: 2,
                edge_count: 1,
                type_count: 1,
                predicate_count: 1,
                avg_degree: 1.0,
                density: 1.0,
            },
        }
    }

    /// Convert shape predictions to predicted violations
    fn predictions_to_violations(
        &self,
        predictions: &[ShapePrediction],
    ) -> Vec<PredictedViolation> {
        predictions
            .iter()
            .map(|pred| PredictedViolation {
                shape_id: pred.shape_id.clone(),
                constraint_type: pred.constraint_type.clone(),
                severity: pred.confidence,
                explanation: format!(
                    "Predicted violation of {} in {} (confidence: {:.2})",
                    pred.constraint_type, pred.shape_id, pred.confidence
                ),
            })
            .collect()
    }

    /// Cache key for a triple
    fn cache_key(s: &str, p: &str, o: &str) -> String {
        format!("{s}\x1F{p}\x1F{o}")
    }

    /// Average severity as overall confidence (0 if no violations)
    fn aggregate_confidence(violations: &[PredictedViolation]) -> f64 {
        if violations.is_empty() {
            return 0.0;
        }
        violations.iter().map(|v| v.severity).sum::<f64>() / violations.len() as f64
    }

    /// Record a latency sample in milliseconds
    fn record_latency(&mut self, elapsed: Duration) {
        self.latency_samples.push(elapsed.as_secs_f64() * 1000.0);
        // Keep only the last 10_000 samples to bound memory usage
        if self.latency_samples.len() > 10_000 {
            self.latency_samples.remove(0);
        }
    }

    /// Approximate the 95th percentile from recorded samples
    fn approximate_p95(&self) -> f64 {
        if self.latency_samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.latency_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::{EnsembleStrategy, ShapeLearner, TrainingExample, TrainingMetrics};

    // -----------------------------------------------------------------------
    // Mock learner that returns configurable predictions
    // -----------------------------------------------------------------------

    struct MockLearner {
        preds: Vec<ShapePrediction>,
    }

    impl ShapeLearner for MockLearner {
        fn name(&self) -> &str {
            "mock"
        }

        fn predict(&self, _f: &GraphFeatures) -> Result<Vec<ShapePrediction>, ShaclAiError> {
            Ok(self.preds.clone())
        }

        fn train(
            &mut self,
            _examples: &[TrainingExample],
        ) -> Result<TrainingMetrics, ShaclAiError> {
            Ok(TrainingMetrics {
                model_name: "mock".to_string(),
                epochs: 1,
                final_loss: 0.0,
                precision: 1.0,
                recall: 1.0,
                f1_score: 1.0,
            })
        }
    }

    fn build_pipeline(predictions: Vec<ShapePrediction>) -> RealTimeInferencePipeline {
        let mut ensemble =
            ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting).with_min_confidence(0.0);
        ensemble.add_learner(Box::new(MockLearner { preds: predictions }), 1.0);
        RealTimeInferencePipeline::new(ensemble, InferencePipelineConfig::default())
    }

    #[test]
    fn test_flush_empty_batch() {
        let mut pipeline = build_pipeline(Vec::new());
        let results = pipeline.flush().expect("flush failed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_flush_on_full() {
        let config = InferencePipelineConfig {
            batch_size: 2,
            max_latency_ms: 10_000,
            confidence_threshold: 0.0,
            enable_caching: false,
            cache_ttl: Duration::from_secs(60),
        };
        let mut ensemble =
            ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting).with_min_confidence(0.0);
        ensemble.add_learner(Box::new(MockLearner { preds: Vec::new() }), 1.0);
        let mut pipeline = RealTimeInferencePipeline::new(ensemble, config);

        // First triple should buffer
        let r1 = pipeline.infer_triple("s", "p", "o1").expect("infer failed");
        assert!(r1.is_none(), "Should buffer until batch is full");

        // Second triple should trigger flush
        let r2 = pipeline.infer_triple("s", "p", "o2").expect("infer failed");
        assert!(r2.is_some(), "Second triple should complete the batch");
    }

    #[test]
    fn test_flush_returns_all_pending() {
        let config = InferencePipelineConfig {
            batch_size: 100,
            max_latency_ms: 10_000,
            confidence_threshold: 0.0,
            enable_caching: false,
            cache_ttl: Duration::from_secs(60),
        };
        let mut ensemble =
            ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting).with_min_confidence(0.0);
        ensemble.add_learner(Box::new(MockLearner { preds: Vec::new() }), 1.0);
        let mut pipeline = RealTimeInferencePipeline::new(ensemble, config);

        // Buffer 5 triples without filling the batch
        for i in 0_u32..5 {
            let _ = pipeline
                .infer_triple("s", "p", &format!("o{i}"))
                .expect("infer failed");
        }

        let results = pipeline.flush().expect("flush failed");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_cache_hit() {
        let violation_pred = ShapePrediction {
            shape_id: "TestShape".to_string(),
            constraint_type: "sh:minCount".to_string(),
            confidence: 0.9,
            supporting_triples: Vec::new(),
        };

        let config = InferencePipelineConfig {
            batch_size: 1, // flush immediately
            enable_caching: true,
            confidence_threshold: 0.0,
            ..Default::default()
        };
        let mut ensemble =
            ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting).with_min_confidence(0.0);
        ensemble.add_learner(
            Box::new(MockLearner {
                preds: vec![violation_pred],
            }),
            1.0,
        );
        let mut pipeline = RealTimeInferencePipeline::new(ensemble, config);

        // First call runs inference
        let r1 = pipeline.infer_triple("s", "p", "o").expect("infer failed");
        assert!(r1.is_some());

        // Second call should hit cache
        let before = pipeline.cache_hits;
        let r2 = pipeline.infer_triple("s", "p", "o").expect("infer failed");
        assert!(r2.is_some());
        assert_eq!(
            pipeline.cache_hits,
            before + 1,
            "Second call should increment cache_hits"
        );
    }

    #[test]
    fn test_pipeline_stats_tracking() {
        let config = InferencePipelineConfig {
            batch_size: 1,
            confidence_threshold: 0.0,
            enable_caching: false,
            ..Default::default()
        };
        let mut ensemble =
            ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting).with_min_confidence(0.0);
        ensemble.add_learner(Box::new(MockLearner { preds: Vec::new() }), 1.0);
        let mut pipeline = RealTimeInferencePipeline::new(ensemble, config);

        for i in 0_u32..5 {
            let _ = pipeline
                .infer_triple("s", "p", &format!("o{i}"))
                .expect("infer failed");
        }

        let stats = pipeline.pipeline_stats();
        assert_eq!(stats.total_inferences, 5);
        assert!(stats.avg_latency_ms >= 0.0);
    }
}
