//! Advanced Query Statistics and Pattern Analysis
//!
//! This module provides sophisticated statistical analysis for query optimization,
//! including histogram-based cardinality estimation, correlation detection,
//! and adaptive learning from query execution history.

use crate::query::algebra::{AlgebraTriplePattern, GraphPattern, TermPattern};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Advanced statistics collector with histogram support
#[derive(Debug)]
pub struct AdvancedStatisticsCollector {
    /// Subject cardinality histogram
    subject_histogram: Arc<RwLock<CardinalityHistogram>>,
    /// Predicate cardinality histogram
    predicate_histogram: Arc<RwLock<CardinalityHistogram>>,
    /// Object cardinality histogram
    object_histogram: Arc<RwLock<CardinalityHistogram>>,
    /// Join selectivity estimates
    join_selectivity: Arc<RwLock<JoinSelectivityEstimator>>,
    /// Pattern execution history
    execution_history: Arc<RwLock<ExecutionHistory>>,
    /// Total queries analyzed
    queries_analyzed: AtomicU64,
}

impl AdvancedStatisticsCollector {
    /// Create a new advanced statistics collector
    pub fn new() -> Self {
        Self {
            subject_histogram: Arc::new(RwLock::new(CardinalityHistogram::new())),
            predicate_histogram: Arc::new(RwLock::new(CardinalityHistogram::new())),
            object_histogram: Arc::new(RwLock::new(CardinalityHistogram::new())),
            join_selectivity: Arc::new(RwLock::new(JoinSelectivityEstimator::new())),
            execution_history: Arc::new(RwLock::new(ExecutionHistory::new(1000))),
            queries_analyzed: AtomicU64::new(0),
        }
    }

    /// Record execution of a triple pattern
    pub fn record_pattern_execution(
        &self,
        pattern: &AlgebraTriplePattern,
        actual_cardinality: usize,
        execution_time_ms: u64,
    ) {
        // Update histograms based on pattern terms
        self.update_histograms(pattern, actual_cardinality);

        // Record in execution history
        let mut history = self
            .execution_history
            .write()
            .expect("execution_history lock poisoned");
        history.record(PatternExecution {
            pattern: pattern.clone(),
            cardinality: actual_cardinality,
            execution_time_ms,
            timestamp: std::time::SystemTime::now(),
        });

        self.queries_analyzed.fetch_add(1, Ordering::Relaxed);
    }

    /// Update cardinality histograms
    fn update_histograms(&self, pattern: &AlgebraTriplePattern, cardinality: usize) {
        // Update subject histogram if bound
        if let TermPattern::NamedNode(node) = &pattern.subject {
            let mut hist = self
                .subject_histogram
                .write()
                .expect("subject_histogram lock poisoned");
            hist.record(node.as_str(), cardinality);
        }

        // Update predicate histogram if bound
        if let TermPattern::NamedNode(node) = &pattern.predicate {
            let mut hist = self
                .predicate_histogram
                .write()
                .expect("predicate_histogram lock poisoned");
            hist.record(node.as_str(), cardinality);
        }

        // Update object histogram if bound
        if let TermPattern::NamedNode(node) = &pattern.object {
            let mut hist = self
                .object_histogram
                .write()
                .expect("object_histogram lock poisoned");
            hist.record(node.as_str(), cardinality);
        }
    }

    /// Estimate cardinality for a pattern using histograms
    pub fn estimate_cardinality(&self, pattern: &AlgebraTriplePattern) -> Option<usize> {
        // Use histogram data if available
        let subject_est = if let TermPattern::NamedNode(node) = &pattern.subject {
            self.subject_histogram
                .read()
                .expect("subject_histogram lock poisoned")
                .estimate(node.as_str())
        } else {
            None
        };

        let predicate_est = if let TermPattern::NamedNode(node) = &pattern.predicate {
            self.predicate_histogram
                .read()
                .expect("predicate_histogram lock poisoned")
                .estimate(node.as_str())
        } else {
            None
        };

        let object_est = if let TermPattern::NamedNode(node) = &pattern.object {
            self.object_histogram
                .read()
                .expect("object_histogram lock poisoned")
                .estimate(node.as_str())
        } else {
            None
        };

        // Combine estimates using minimum (most selective)
        [subject_est, predicate_est, object_est]
            .iter()
            .filter_map(|&x| x)
            .min()
    }

    /// Record join execution for selectivity learning
    pub fn record_join_execution(
        &self,
        _left_pattern: &GraphPattern,
        _right_pattern: &GraphPattern,
        left_cardinality: usize,
        right_cardinality: usize,
        result_cardinality: usize,
    ) {
        let mut estimator = self
            .join_selectivity
            .write()
            .expect("join_selectivity lock poisoned");
        estimator.record_join(left_cardinality, right_cardinality, result_cardinality);
    }

    /// Estimate join selectivity
    pub fn estimate_join_selectivity(&self, left_card: usize, right_card: usize) -> f64 {
        self.join_selectivity
            .read()
            .expect("join_selectivity lock poisoned")
            .estimate(left_card, right_card)
    }

    /// Get execution history for a pattern
    pub fn get_pattern_history(&self, pattern: &AlgebraTriplePattern) -> Vec<PatternExecution> {
        self.execution_history
            .read()
            .expect("execution_history lock poisoned")
            .get_similar_patterns(pattern)
    }

    /// Get overall statistics
    pub fn get_statistics(&self) -> AdvancedStatistics {
        AdvancedStatistics {
            queries_analyzed: self.queries_analyzed.load(Ordering::Relaxed),
            subject_histogram_size: self
                .subject_histogram
                .read()
                .expect("subject_histogram lock poisoned")
                .size(),
            predicate_histogram_size: self
                .predicate_histogram
                .read()
                .expect("predicate_histogram lock poisoned")
                .size(),
            object_histogram_size: self
                .object_histogram
                .read()
                .expect("object_histogram lock poisoned")
                .size(),
            join_samples: self
                .join_selectivity
                .read()
                .expect("join_selectivity lock poisoned")
                .sample_count(),
            history_size: self
                .execution_history
                .read()
                .expect("execution_history lock poisoned")
                .size(),
        }
    }

    /// Clear all statistics (useful for testing)
    pub fn clear(&self) {
        self.subject_histogram
            .write()
            .expect("subject_histogram lock poisoned")
            .clear();
        self.predicate_histogram
            .write()
            .expect("predicate_histogram lock poisoned")
            .clear();
        self.object_histogram
            .write()
            .expect("object_histogram lock poisoned")
            .clear();
        self.join_selectivity
            .write()
            .expect("join_selectivity lock poisoned")
            .clear();
        self.execution_history
            .write()
            .expect("execution_history lock poisoned")
            .clear();
        self.queries_analyzed.store(0, Ordering::Relaxed);
    }
}

impl Default for AdvancedStatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Cardinality histogram for specific terms
#[derive(Debug)]
struct CardinalityHistogram {
    /// Term -> observed cardinalities
    data: HashMap<String, Vec<usize>>,
    /// Maximum samples per term
    max_samples: usize,
}

impl CardinalityHistogram {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
            max_samples: 100, // Keep last 100 observations per term
        }
    }

    fn record(&mut self, term: &str, cardinality: usize) {
        let samples = self.data.entry(term.to_string()).or_default();
        samples.push(cardinality);

        // Keep only recent samples
        if samples.len() > self.max_samples {
            samples.remove(0);
        }
    }

    fn estimate(&self, term: &str) -> Option<usize> {
        self.data.get(term).and_then(|samples| {
            if samples.is_empty() {
                None
            } else {
                // Use median for robust estimation
                let mut sorted = samples.clone();
                sorted.sort_unstable();
                Some(sorted[sorted.len() / 2])
            }
        })
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

/// Join selectivity estimator
#[derive(Debug)]
struct JoinSelectivityEstimator {
    /// Observed join results: (left_card, right_card) -> result_card
    observations: Vec<JoinObservation>,
    /// Maximum observations to keep
    max_observations: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct JoinObservation {
    left_cardinality: usize,
    right_cardinality: usize,
    result_cardinality: usize,
    selectivity: f64,
}

impl JoinSelectivityEstimator {
    fn new() -> Self {
        Self {
            observations: Vec::new(),
            max_observations: 1000,
        }
    }

    fn record_join(&mut self, left_card: usize, right_card: usize, result_card: usize) {
        let product = (left_card as f64) * (right_card as f64);
        let selectivity = if product > 0.0 {
            (result_card as f64) / product
        } else {
            0.0
        };

        self.observations.push(JoinObservation {
            left_cardinality: left_card,
            right_cardinality: right_card,
            result_cardinality: result_card,
            selectivity,
        });

        // Keep only recent observations
        if self.observations.len() > self.max_observations {
            self.observations.remove(0);
        }
    }

    fn estimate(&self, left_card: usize, right_card: usize) -> f64 {
        if self.observations.is_empty() {
            return 0.1; // Default selectivity
        }

        // Find similar observations (within 2x range)
        let similar: Vec<f64> = self
            .observations
            .iter()
            .filter(|obs| {
                let left_ratio = (obs.left_cardinality as f64) / (left_card.max(1) as f64);
                let right_ratio = (obs.right_cardinality as f64) / (right_card.max(1) as f64);
                (0.5..=2.0).contains(&left_ratio) && (0.5..=2.0).contains(&right_ratio)
            })
            .map(|obs| obs.selectivity)
            .collect();

        if similar.is_empty() {
            // Use global average
            let avg: f64 = self.observations.iter().map(|o| o.selectivity).sum::<f64>()
                / self.observations.len() as f64;
            avg
        } else {
            // Use median of similar observations
            let mut sorted = similar;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        }
    }

    fn sample_count(&self) -> usize {
        self.observations.len()
    }

    fn clear(&mut self) {
        self.observations.clear();
    }
}

/// Execution history tracker
#[derive(Debug)]
struct ExecutionHistory {
    /// Recent pattern executions
    executions: Vec<PatternExecution>,
    /// Maximum history size
    max_size: usize,
}

#[derive(Debug, Clone)]
pub struct PatternExecution {
    pub pattern: AlgebraTriplePattern,
    pub cardinality: usize,
    pub execution_time_ms: u64,
    pub timestamp: std::time::SystemTime,
}

impl ExecutionHistory {
    fn new(max_size: usize) -> Self {
        Self {
            executions: Vec::new(),
            max_size,
        }
    }

    fn record(&mut self, execution: PatternExecution) {
        self.executions.push(execution);

        // Keep only recent history
        if self.executions.len() > self.max_size {
            self.executions.remove(0);
        }
    }

    fn get_similar_patterns(&self, pattern: &AlgebraTriplePattern) -> Vec<PatternExecution> {
        self.executions
            .iter()
            .filter(|exec| Self::patterns_similar(&exec.pattern, pattern))
            .cloned()
            .collect()
    }

    fn patterns_similar(p1: &AlgebraTriplePattern, p2: &AlgebraTriplePattern) -> bool {
        // Patterns are similar if they have the same structure (bound/unbound positions)
        Self::term_pattern_type(&p1.subject) == Self::term_pattern_type(&p2.subject)
            && Self::term_pattern_type(&p1.predicate) == Self::term_pattern_type(&p2.predicate)
            && Self::term_pattern_type(&p1.object) == Self::term_pattern_type(&p2.object)
    }

    fn term_pattern_type(term: &TermPattern) -> &'static str {
        match term {
            TermPattern::Variable(_) => "var",
            TermPattern::NamedNode(_) => "node",
            TermPattern::BlankNode(_) => "blank",
            TermPattern::Literal(_) => "literal",
            TermPattern::QuotedTriple(_) => "quoted",
        }
    }

    fn size(&self) -> usize {
        self.executions.len()
    }

    fn clear(&mut self) {
        self.executions.clear();
    }
}

/// Summary statistics
#[derive(Debug, Clone)]
pub struct AdvancedStatistics {
    pub queries_analyzed: u64,
    pub subject_histogram_size: usize,
    pub predicate_histogram_size: usize,
    pub object_histogram_size: usize,
    pub join_samples: usize,
    pub history_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Variable};

    fn create_test_pattern() -> AlgebraTriplePattern {
        AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("s").unwrap()),
            predicate: TermPattern::NamedNode(
                NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap(),
            ),
            object: TermPattern::Variable(Variable::new("o").unwrap()),
        }
    }

    #[test]
    fn test_collector_creation() {
        let collector = AdvancedStatisticsCollector::new();
        let stats = collector.get_statistics();

        assert_eq!(stats.queries_analyzed, 0);
        assert_eq!(stats.history_size, 0);
    }

    #[test]
    fn test_pattern_recording() {
        let collector = AdvancedStatisticsCollector::new();
        let pattern = create_test_pattern();

        collector.record_pattern_execution(&pattern, 100, 50);

        let stats = collector.get_statistics();
        assert_eq!(stats.queries_analyzed, 1);
        assert_eq!(stats.history_size, 1);
    }

    #[test]
    fn test_histogram_estimation() {
        let collector = AdvancedStatisticsCollector::new();
        let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();

        let pattern = AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("s").unwrap()),
            predicate: TermPattern::NamedNode(foaf_name.clone()),
            object: TermPattern::Variable(Variable::new("o").unwrap()),
        };

        // Record multiple observations
        for i in 1..=10 {
            collector.record_pattern_execution(&pattern, 100 * i, 10);
        }

        // Estimate should be based on histogram
        let estimate = collector.estimate_cardinality(&pattern);
        assert!(estimate.is_some());
        let est = estimate.unwrap();
        // Should be around median (500-600)
        assert!((400..=700).contains(&est));
    }

    #[test]
    fn test_join_selectivity() {
        let collector = AdvancedStatisticsCollector::new();

        // Record several joins
        collector.record_join_execution(
            &GraphPattern::Bgp(vec![]),
            &GraphPattern::Bgp(vec![]),
            1000,
            1000,
            100,
        );
        collector.record_join_execution(
            &GraphPattern::Bgp(vec![]),
            &GraphPattern::Bgp(vec![]),
            2000,
            2000,
            400,
        );

        // Estimate should be around 0.0001 (100/1M or 400/4M)
        let selectivity = collector.estimate_join_selectivity(1500, 1500);
        assert!(selectivity > 0.00005 && selectivity < 0.002);
    }

    #[test]
    fn test_history_limit() {
        let collector = AdvancedStatisticsCollector::new();
        let pattern = create_test_pattern();

        // Record more than max_size executions
        for _ in 0..1500 {
            collector.record_pattern_execution(&pattern, 100, 10);
        }

        let stats = collector.get_statistics();
        assert!(stats.history_size <= 1000);
    }

    #[test]
    fn test_clear_statistics() {
        let collector = AdvancedStatisticsCollector::new();
        let pattern = create_test_pattern();

        collector.record_pattern_execution(&pattern, 100, 10);
        collector.clear();

        let stats = collector.get_statistics();
        assert_eq!(stats.queries_analyzed, 0);
        assert_eq!(stats.history_size, 0);
    }
}
