//! # Adaptive Recall Tuner for Vector Search
//!
//! Dynamically adjusts vector search parameters (ef_search, num_candidates,
//! re-ranking depth, over-fetch ratio) to achieve target recall@k while
//! minimizing latency.
//!
//! ## Problem
//!
//! Fixed search parameters lead to either:
//! - Under-fetching: Missing relevant results (low recall)
//! - Over-fetching: Wasting compute on unnecessary candidates (high latency)
//!
//! ## Solution
//!
//! The adaptive tuner observes query feedback (user clicks, explicit relevance
//! judgments, or ground-truth evaluations) and uses a PID-like control loop
//! to adjust parameters in real-time.
//!
//! ## Architecture
//!
//! ```text
//! Query --> SearchEngine --> Results --> FeedbackCollector
//!   ^                                         |
//!   |                                         v
//!   +--- ParameterAdjuster <--- RecallEstimator
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Search parameters
// ---------------------------------------------------------------------------

/// Tunable search parameters for vector index queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchParams {
    /// HNSW ef_search: number of candidates to explore during search.
    pub ef_search: usize,
    /// Number of candidates to pre-fetch before re-ranking.
    pub num_candidates: usize,
    /// Over-fetch ratio: fetch this many times more candidates than k.
    pub over_fetch_ratio: f64,
    /// Re-ranking depth: how many candidates to re-rank with exact distances.
    pub rerank_depth: usize,
    /// Whether to enable approximate early termination.
    pub early_termination: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            ef_search: 64,
            num_candidates: 100,
            over_fetch_ratio: 2.0,
            rerank_depth: 50,
            early_termination: true,
        }
    }
}

impl SearchParams {
    /// Create parameters optimized for high recall.
    pub fn high_recall() -> Self {
        Self {
            ef_search: 256,
            num_candidates: 500,
            over_fetch_ratio: 5.0,
            rerank_depth: 200,
            early_termination: false,
        }
    }

    /// Create parameters optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            ef_search: 32,
            num_candidates: 50,
            over_fetch_ratio: 1.5,
            rerank_depth: 20,
            early_termination: true,
        }
    }

    /// Clamp all parameters to valid ranges.
    pub fn clamp(&mut self) {
        self.ef_search = self.ef_search.clamp(8, 1024);
        self.num_candidates = self.num_candidates.clamp(10, 5000);
        self.over_fetch_ratio = self.over_fetch_ratio.clamp(1.0, 20.0);
        self.rerank_depth = self.rerank_depth.clamp(0, self.num_candidates);
    }
}

// ---------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------

/// Feedback from a single query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeedback {
    /// The parameters used for this query.
    pub params: SearchParams,
    /// Number of results requested (k).
    pub k: usize,
    /// Number of truly relevant results in the top-k (from ground truth or clicks).
    pub relevant_in_top_k: usize,
    /// Total number of known relevant results (if available).
    pub total_relevant: Option<usize>,
    /// Query latency.
    pub latency: Duration,
    /// Timestamp of the feedback.
    #[serde(skip, default = "std::time::Instant::now")]
    pub timestamp: Instant,
}

impl QueryFeedback {
    /// Compute recall@k for this feedback.
    pub fn recall_at_k(&self) -> f64 {
        match self.total_relevant {
            Some(total) if total > 0 => self.relevant_in_top_k as f64 / total as f64,
            _ => {
                // If total_relevant is unknown, estimate from k
                if self.k == 0 {
                    return 0.0;
                }
                self.relevant_in_top_k as f64 / self.k as f64
            }
        }
    }

    /// Compute precision@k.
    pub fn precision_at_k(&self) -> f64 {
        if self.k == 0 {
            return 0.0;
        }
        self.relevant_in_top_k as f64 / self.k as f64
    }
}

// ---------------------------------------------------------------------------
// Tuner configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive recall tuner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerConfig {
    /// Target recall@k (e.g., 0.95 for 95%).
    pub target_recall: f64,
    /// Maximum acceptable latency.
    pub max_latency: Duration,
    /// Number of recent feedback samples to consider.
    pub window_size: usize,
    /// Proportional gain for the control loop.
    pub kp: f64,
    /// Integral gain for the control loop.
    pub ki: f64,
    /// Derivative gain for the control loop.
    pub kd: f64,
    /// Minimum number of samples before adjusting.
    pub min_samples: usize,
    /// How often to recalculate (in number of queries).
    pub adjust_interval: usize,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            target_recall: 0.95,
            max_latency: Duration::from_millis(100),
            window_size: 100,
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,
            min_samples: 10,
            adjust_interval: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Tuner statistics
// ---------------------------------------------------------------------------

/// Statistics from the adaptive tuner.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TunerStats {
    /// Total number of feedback samples received.
    pub total_feedbacks: u64,
    /// Number of parameter adjustments made.
    pub adjustments_made: u64,
    /// Current estimated recall.
    pub current_recall: f64,
    /// Current average latency in milliseconds.
    pub current_avg_latency_ms: f64,
    /// Whether the tuner is currently meeting its target.
    pub target_met: bool,
    /// Running average precision.
    pub avg_precision: f64,
    /// Historical recall values (last N).
    pub recall_history: Vec<f64>,
}

impl TunerStats {
    /// Check if recall is within tolerance of target.
    pub fn is_near_target(&self, target: f64, tolerance: f64) -> bool {
        (self.current_recall - target).abs() < tolerance
    }
}

// ---------------------------------------------------------------------------
// Adaptive Recall Tuner
// ---------------------------------------------------------------------------

/// The main adaptive recall tuner.
///
/// Collects query feedback and adjusts search parameters to converge on
/// the target recall while respecting latency constraints.
pub struct AdaptiveRecallTuner {
    config: TunerConfig,
    current_params: SearchParams,
    feedback_window: VecDeque<QueryFeedback>,
    stats: TunerStats,
    /// PID controller state
    integral_error: f64,
    prev_error: f64,
    query_count: u64,
}

impl AdaptiveRecallTuner {
    /// Create a new tuner with default parameters and configuration.
    pub fn new(config: TunerConfig) -> Self {
        Self {
            config,
            current_params: SearchParams::default(),
            feedback_window: VecDeque::new(),
            stats: TunerStats::default(),
            integral_error: 0.0,
            prev_error: 0.0,
            query_count: 0,
        }
    }

    /// Create with specific initial parameters.
    pub fn with_initial_params(config: TunerConfig, initial: SearchParams) -> Self {
        Self {
            config,
            current_params: initial,
            feedback_window: VecDeque::new(),
            stats: TunerStats::default(),
            integral_error: 0.0,
            prev_error: 0.0,
            query_count: 0,
        }
    }

    /// Get the current recommended search parameters.
    pub fn current_params(&self) -> &SearchParams {
        &self.current_params
    }

    /// Get tuner statistics.
    pub fn stats(&self) -> &TunerStats {
        &self.stats
    }

    /// Record a query feedback observation.
    ///
    /// Returns `true` if parameters were adjusted as a result.
    pub fn record_feedback(&mut self, feedback: QueryFeedback) -> bool {
        // Add to window
        self.feedback_window.push_back(feedback);
        while self.feedback_window.len() > self.config.window_size {
            self.feedback_window.pop_front();
        }

        self.stats.total_feedbacks += 1;
        self.query_count += 1;

        // Update running statistics
        self.update_stats();

        // Check if we should adjust
        if self.feedback_window.len() >= self.config.min_samples
            && self.query_count % self.config.adjust_interval as u64 == 0
        {
            self.adjust_parameters();
            return true;
        }

        false
    }

    /// Force a parameter adjustment regardless of interval.
    pub fn force_adjust(&mut self) {
        if self.feedback_window.len() >= self.config.min_samples {
            self.adjust_parameters();
        }
    }

    /// Reset the tuner state (keeps configuration).
    pub fn reset(&mut self) {
        self.current_params = SearchParams::default();
        self.feedback_window.clear();
        self.stats = TunerStats::default();
        self.integral_error = 0.0;
        self.prev_error = 0.0;
        self.query_count = 0;
    }

    // ── Internal methods ──────────────────────────────────────────────────

    fn update_stats(&mut self) {
        if self.feedback_window.is_empty() {
            return;
        }

        let recalls: Vec<f64> = self
            .feedback_window
            .iter()
            .map(|f| f.recall_at_k())
            .collect();
        let precisions: Vec<f64> = self
            .feedback_window
            .iter()
            .map(|f| f.precision_at_k())
            .collect();
        let latencies: Vec<f64> = self
            .feedback_window
            .iter()
            .map(|f| f.latency.as_millis() as f64)
            .collect();

        let n = recalls.len() as f64;
        self.stats.current_recall = recalls.iter().sum::<f64>() / n;
        self.stats.avg_precision = precisions.iter().sum::<f64>() / n;
        self.stats.current_avg_latency_ms = latencies.iter().sum::<f64>() / n;
        self.stats.target_met = self.stats.current_recall >= self.config.target_recall;

        // Record recall history (keep last 50)
        self.stats.recall_history.push(self.stats.current_recall);
        if self.stats.recall_history.len() > 50 {
            self.stats.recall_history.remove(0);
        }
    }

    fn adjust_parameters(&mut self) {
        let error = self.config.target_recall - self.stats.current_recall;

        // PID control
        self.integral_error += error;
        // Clamp integral to prevent windup
        self.integral_error = self.integral_error.clamp(-10.0, 10.0);

        let derivative = error - self.prev_error;
        self.prev_error = error;

        let adjustment = self.config.kp * error
            + self.config.ki * self.integral_error
            + self.config.kd * derivative;

        // Apply adjustment to parameters
        // Positive adjustment means recall is too low -> increase search effort
        // Negative adjustment means recall is high enough -> can reduce effort

        let scale = 1.0 + adjustment;

        self.current_params.ef_search =
            ((self.current_params.ef_search as f64 * scale) as usize).max(8);
        self.current_params.num_candidates =
            ((self.current_params.num_candidates as f64 * scale) as usize).max(10);
        self.current_params.over_fetch_ratio =
            (self.current_params.over_fetch_ratio * scale).max(1.0);
        self.current_params.rerank_depth =
            ((self.current_params.rerank_depth as f64 * scale) as usize).max(1);

        // If latency is too high, pull back
        if self.stats.current_avg_latency_ms > self.config.max_latency.as_millis() as f64 {
            let latency_ratio =
                self.config.max_latency.as_millis() as f64 / self.stats.current_avg_latency_ms;
            self.current_params.ef_search =
                ((self.current_params.ef_search as f64 * latency_ratio) as usize).max(8);
            self.current_params.num_candidates =
                ((self.current_params.num_candidates as f64 * latency_ratio) as usize).max(10);
        }

        self.current_params.clamp();
        self.stats.adjustments_made += 1;
    }
}

impl fmt::Debug for AdaptiveRecallTuner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdaptiveRecallTuner")
            .field("config", &self.config)
            .field("current_params", &self.current_params)
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Recall evaluator (ground-truth based)
// ---------------------------------------------------------------------------

/// Evaluates recall by comparing search results against ground-truth.
pub struct RecallEvaluator;

/// A single recall evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallEvaluation {
    /// Recall at the specified k.
    pub recall_at_k: f64,
    /// Precision at the specified k.
    pub precision_at_k: f64,
    /// F1 score.
    pub f1_score: f64,
    /// Average precision (AP).
    pub average_precision: f64,
    /// Normalized discounted cumulative gain (nDCG).
    pub ndcg: f64,
    /// Number of queries evaluated.
    pub num_queries: usize,
}

impl RecallEvaluator {
    /// Evaluate recall for a set of query results against ground truth.
    ///
    /// - `results`: For each query, the IDs returned by the search engine.
    /// - `ground_truth`: For each query, the set of truly relevant IDs.
    /// - `k`: The cutoff for evaluation.
    pub fn evaluate(
        results: &[Vec<String>],
        ground_truth: &[Vec<String>],
        k: usize,
    ) -> RecallEvaluation {
        if results.is_empty() || ground_truth.is_empty() {
            return RecallEvaluation {
                recall_at_k: 0.0,
                precision_at_k: 0.0,
                f1_score: 0.0,
                average_precision: 0.0,
                ndcg: 0.0,
                num_queries: 0,
            };
        }

        let n = results.len().min(ground_truth.len());
        let mut total_recall = 0.0;
        let mut total_precision = 0.0;
        let mut total_ap = 0.0;
        let mut total_ndcg = 0.0;

        for i in 0..n {
            let result_k: Vec<_> = results[i].iter().take(k).cloned().collect();
            let truth: std::collections::HashSet<_> = ground_truth[i].iter().cloned().collect();

            if truth.is_empty() {
                continue;
            }

            // Recall@k
            let relevant_found = result_k.iter().filter(|r| truth.contains(*r)).count();
            let recall = relevant_found as f64 / truth.len() as f64;
            total_recall += recall;

            // Precision@k
            let precision = if result_k.is_empty() {
                0.0
            } else {
                relevant_found as f64 / result_k.len() as f64
            };
            total_precision += precision;

            // Average Precision (AP)
            let mut running_relevant = 0.0;
            let mut ap_sum = 0.0;
            for (pos, item) in result_k.iter().enumerate() {
                if truth.contains(item) {
                    running_relevant += 1.0;
                    ap_sum += running_relevant / (pos + 1) as f64;
                }
            }
            total_ap += if truth.is_empty() {
                0.0
            } else {
                ap_sum / truth.len() as f64
            };

            // nDCG
            let dcg: f64 = result_k
                .iter()
                .enumerate()
                .map(|(pos, item)| {
                    let rel = if truth.contains(item) { 1.0 } else { 0.0 };
                    rel / ((pos + 2) as f64).ln()
                })
                .sum();
            let ideal_k = truth.len().min(k);
            let ideal_dcg: f64 = (0..ideal_k).map(|pos| 1.0 / ((pos + 2) as f64).ln()).sum();
            let ndcg = if ideal_dcg > 0.0 {
                dcg / ideal_dcg
            } else {
                0.0
            };
            total_ndcg += ndcg;
        }

        let n_f = n as f64;
        let avg_recall = total_recall / n_f;
        let avg_precision = total_precision / n_f;
        let f1 = if (avg_recall + avg_precision) > 0.0 {
            2.0 * avg_recall * avg_precision / (avg_recall + avg_precision)
        } else {
            0.0
        };

        RecallEvaluation {
            recall_at_k: avg_recall,
            precision_at_k: avg_precision,
            f1_score: f1,
            average_precision: total_ap / n_f,
            ndcg: total_ndcg / n_f,
            num_queries: n,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SearchParams tests ────────────────────────────────────────────────

    #[test]
    fn test_search_params_default() {
        let p = SearchParams::default();
        assert_eq!(p.ef_search, 64);
        assert_eq!(p.num_candidates, 100);
        assert!((p.over_fetch_ratio - 2.0).abs() < 0.01);
        assert_eq!(p.rerank_depth, 50);
        assert!(p.early_termination);
    }

    #[test]
    fn test_search_params_high_recall() {
        let p = SearchParams::high_recall();
        assert!(p.ef_search >= 256);
        assert!(!p.early_termination);
    }

    #[test]
    fn test_search_params_low_latency() {
        let p = SearchParams::low_latency();
        assert!(p.ef_search <= 32);
        assert!(p.early_termination);
    }

    #[test]
    fn test_search_params_clamp() {
        let mut p = SearchParams {
            ef_search: 0,
            num_candidates: 0,
            over_fetch_ratio: 0.1,
            rerank_depth: 99999,
            early_termination: false,
        };
        p.clamp();
        assert_eq!(p.ef_search, 8);
        assert_eq!(p.num_candidates, 10);
        assert!((p.over_fetch_ratio - 1.0).abs() < 0.01);
        assert_eq!(p.rerank_depth, 10); // clamped to num_candidates
    }

    // ── QueryFeedback tests ───────────────────────────────────────────────

    #[test]
    fn test_feedback_recall_with_ground_truth() {
        let fb = QueryFeedback {
            params: SearchParams::default(),
            k: 10,
            relevant_in_top_k: 8,
            total_relevant: Some(10),
            latency: Duration::from_millis(50),
            timestamp: Instant::now(),
        };
        assert!((fb.recall_at_k() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_feedback_recall_without_ground_truth() {
        let fb = QueryFeedback {
            params: SearchParams::default(),
            k: 10,
            relevant_in_top_k: 7,
            total_relevant: None,
            latency: Duration::from_millis(50),
            timestamp: Instant::now(),
        };
        assert!((fb.recall_at_k() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_feedback_precision() {
        let fb = QueryFeedback {
            params: SearchParams::default(),
            k: 10,
            relevant_in_top_k: 5,
            total_relevant: Some(20),
            latency: Duration::from_millis(50),
            timestamp: Instant::now(),
        };
        assert!((fb.precision_at_k() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_feedback_k_zero() {
        let fb = QueryFeedback {
            params: SearchParams::default(),
            k: 0,
            relevant_in_top_k: 0,
            total_relevant: None,
            latency: Duration::from_millis(1),
            timestamp: Instant::now(),
        };
        assert_eq!(fb.recall_at_k(), 0.0);
        assert_eq!(fb.precision_at_k(), 0.0);
    }

    // ── TunerConfig tests ─────────────────────────────────────────────────

    #[test]
    fn test_tuner_config_default() {
        let c = TunerConfig::default();
        assert!((c.target_recall - 0.95).abs() < 0.01);
        assert_eq!(c.window_size, 100);
        assert_eq!(c.min_samples, 10);
    }

    // ── Tuner core tests ──────────────────────────────────────────────────

    fn make_feedback(recall_ratio: f64, k: usize, latency_ms: u64) -> QueryFeedback {
        let relevant = (k as f64 * recall_ratio) as usize;
        QueryFeedback {
            params: SearchParams::default(),
            k,
            relevant_in_top_k: relevant,
            total_relevant: Some(k),
            latency: Duration::from_millis(latency_ms),
            timestamp: Instant::now(),
        }
    }

    #[test]
    fn test_tuner_initial_params() {
        let tuner = AdaptiveRecallTuner::new(TunerConfig::default());
        assert_eq!(tuner.current_params().ef_search, 64);
    }

    #[test]
    fn test_tuner_with_initial_params() {
        let initial = SearchParams::high_recall();
        let tuner =
            AdaptiveRecallTuner::with_initial_params(TunerConfig::default(), initial.clone());
        assert_eq!(tuner.current_params().ef_search, initial.ef_search);
    }

    #[test]
    fn test_tuner_no_adjust_before_min_samples() {
        let config = TunerConfig {
            min_samples: 10,
            adjust_interval: 1,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        for _ in 0..5 {
            let adjusted = tuner.record_feedback(make_feedback(0.5, 10, 50));
            assert!(!adjusted);
        }
        assert_eq!(tuner.stats().adjustments_made, 0);
    }

    #[test]
    fn test_tuner_adjusts_after_min_samples() {
        let config = TunerConfig {
            min_samples: 5,
            adjust_interval: 5,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        for i in 0..10 {
            tuner.record_feedback(make_feedback(0.5, 10, 50));
            if i >= 4 && (i + 1) % 5 == 0 {
                // Should have adjusted
            }
        }
        assert!(tuner.stats().adjustments_made > 0);
    }

    #[test]
    fn test_tuner_increases_params_for_low_recall() {
        let config = TunerConfig {
            min_samples: 5,
            adjust_interval: 1,
            target_recall: 0.95,
            kp: 0.5,
            ki: 0.0,
            kd: 0.0,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);
        let initial_ef = tuner.current_params().ef_search;

        // Feed low recall data
        for _ in 0..10 {
            tuner.record_feedback(make_feedback(0.3, 10, 30));
        }

        // Parameters should have increased
        assert!(tuner.current_params().ef_search > initial_ef);
    }

    #[test]
    fn test_tuner_decreases_params_for_high_recall() {
        let config = TunerConfig {
            min_samples: 5,
            adjust_interval: 1,
            target_recall: 0.5,
            kp: 0.5,
            ki: 0.0,
            kd: 0.0,
            ..Default::default()
        };
        let initial = SearchParams::high_recall();
        let mut tuner = AdaptiveRecallTuner::with_initial_params(config, initial.clone());

        // Feed high recall data (above target)
        for _ in 0..10 {
            tuner.record_feedback(make_feedback(0.99, 10, 30));
        }

        // Parameters should have decreased (or stayed) since recall exceeds target
        assert!(tuner.current_params().ef_search <= initial.ef_search);
    }

    #[test]
    fn test_tuner_respects_latency_constraint() {
        let config = TunerConfig {
            min_samples: 5,
            adjust_interval: 1,
            max_latency: Duration::from_millis(50),
            target_recall: 0.99,
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        // Feed data with high latency -> tuner should pull back despite low recall
        for _ in 0..20 {
            tuner.record_feedback(make_feedback(0.3, 10, 200));
        }

        // ef_search should not grow unboundedly
        assert!(tuner.current_params().ef_search < 1024);
    }

    #[test]
    fn test_tuner_stats_tracking() {
        let config = TunerConfig {
            min_samples: 3,
            adjust_interval: 1,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        for _ in 0..5 {
            tuner.record_feedback(make_feedback(0.8, 10, 40));
        }

        assert_eq!(tuner.stats().total_feedbacks, 5);
        assert!(tuner.stats().current_recall > 0.0);
        assert!(tuner.stats().current_avg_latency_ms > 0.0);
    }

    #[test]
    fn test_tuner_reset() {
        let config = TunerConfig {
            min_samples: 3,
            adjust_interval: 1,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        for _ in 0..5 {
            tuner.record_feedback(make_feedback(0.8, 10, 40));
        }
        tuner.reset();

        assert_eq!(tuner.stats().total_feedbacks, 0);
        assert_eq!(tuner.stats().adjustments_made, 0);
        assert_eq!(tuner.current_params().ef_search, 64);
    }

    #[test]
    fn test_tuner_force_adjust() {
        let config = TunerConfig {
            min_samples: 3,
            adjust_interval: 100, // Very high interval
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        for _ in 0..5 {
            tuner.record_feedback(make_feedback(0.5, 10, 40));
        }
        assert_eq!(tuner.stats().adjustments_made, 0);

        tuner.force_adjust();
        assert_eq!(tuner.stats().adjustments_made, 1);
    }

    #[test]
    fn test_stats_near_target() {
        let stats = TunerStats {
            current_recall: 0.94,
            ..Default::default()
        };
        assert!(stats.is_near_target(0.95, 0.02));
        assert!(!stats.is_near_target(0.95, 0.005));
    }

    #[test]
    fn test_recall_history() {
        let config = TunerConfig {
            min_samples: 3,
            adjust_interval: 1,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        for _ in 0..5 {
            tuner.record_feedback(make_feedback(0.8, 10, 40));
        }

        assert!(!tuner.stats().recall_history.is_empty());
    }

    // ── RecallEvaluator tests ─────────────────────────────────────────────

    #[test]
    fn test_evaluator_perfect_recall() {
        let results = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let truth = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let eval = RecallEvaluator::evaluate(&results, &truth, 3);
        assert!((eval.recall_at_k - 1.0).abs() < 0.01);
        assert!((eval.precision_at_k - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluator_partial_recall() {
        let results = vec![vec!["a".to_string(), "b".to_string(), "x".to_string()]];
        let truth = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ]];
        let eval = RecallEvaluator::evaluate(&results, &truth, 3);
        assert!((eval.recall_at_k - 0.5).abs() < 0.01); // 2/4
        assert!((eval.precision_at_k - 2.0 / 3.0).abs() < 0.01); // 2/3
    }

    #[test]
    fn test_evaluator_zero_recall() {
        let results = vec![vec!["x".to_string(), "y".to_string(), "z".to_string()]];
        let truth = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let eval = RecallEvaluator::evaluate(&results, &truth, 3);
        assert_eq!(eval.recall_at_k, 0.0);
        assert_eq!(eval.precision_at_k, 0.0);
    }

    #[test]
    fn test_evaluator_empty() {
        let eval = RecallEvaluator::evaluate(&[], &[], 10);
        assert_eq!(eval.num_queries, 0);
        assert_eq!(eval.recall_at_k, 0.0);
    }

    #[test]
    fn test_evaluator_multiple_queries() {
        let results = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
        ];
        let truth = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "x".to_string()],
        ];
        let eval = RecallEvaluator::evaluate(&results, &truth, 2);
        assert_eq!(eval.num_queries, 2);
        // Query 1: recall=1.0, Query 2: recall=0.5 -> avg=0.75
        assert!((eval.recall_at_k - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_evaluator_k_less_than_results() {
        let results = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ]];
        let truth = vec![vec!["a".to_string(), "b".to_string()]];
        let eval = RecallEvaluator::evaluate(&results, &truth, 2);
        // Only top-2 considered: a, b — both relevant
        assert!((eval.recall_at_k - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluator_ndcg() {
        let results = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let truth = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let eval = RecallEvaluator::evaluate(&results, &truth, 3);
        // Perfect ranking -> nDCG should be 1.0
        assert!((eval.ndcg - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluator_f1_score() {
        let results = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let truth = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let eval = RecallEvaluator::evaluate(&results, &truth, 3);
        // P=1.0, R=1.0 -> F1=1.0
        assert!((eval.f1_score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluator_average_precision() {
        let results = vec![vec!["a".to_string(), "x".to_string(), "b".to_string()]];
        let truth = vec![vec!["a".to_string(), "b".to_string()]];
        let eval = RecallEvaluator::evaluate(&results, &truth, 3);
        // AP: (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        assert!(eval.average_precision > 0.0);
    }

    // ── Integration test ──────────────────────────────────────────────────

    #[test]
    fn test_tuner_convergence_simulation() {
        let config = TunerConfig {
            min_samples: 5,
            adjust_interval: 1,
            target_recall: 0.9,
            kp: 0.3,
            ki: 0.05,
            kd: 0.02,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        // Simulate recall improving as parameters are tuned
        for i in 0..50 {
            let recall = 0.5 + (i as f64 * 0.01).min(0.45);
            tuner.record_feedback(make_feedback(recall, 10, 30));
        }

        // After 50 iterations, should have made adjustments
        assert!(tuner.stats().adjustments_made > 0);
        assert!(tuner.stats().total_feedbacks == 50);
    }

    #[test]
    fn test_integral_windup_prevention() {
        let config = TunerConfig {
            min_samples: 3,
            adjust_interval: 1,
            target_recall: 0.99,
            kp: 0.1,
            ki: 1.0, // Very high integral gain
            kd: 0.0,
            ..Default::default()
        };
        let mut tuner = AdaptiveRecallTuner::new(config);

        // Feed consistently low recall
        for _ in 0..100 {
            tuner.record_feedback(make_feedback(0.1, 10, 20));
        }

        // Despite high integral error, parameters should be clamped
        assert!(tuner.current_params().ef_search <= 1024);
        assert!(tuner.current_params().num_candidates <= 5000);
    }
}
