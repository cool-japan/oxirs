//! Query statistics collection and management using SciRS2-core
//!
//! This module provides comprehensive statistics collection for query optimization,
//! including cardinality estimation, selectivity tracking, and execution metrics.

use crate::model::pattern::TriplePattern;
use crate::model::Triple;
use crate::query::algebra::{AlgebraTriplePattern, TermPattern};
use crate::OxirsError;
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Graph statistics for query optimization
///
/// Collects and maintains statistical information about the RDF graph
/// for use in cost-based query optimization.
#[derive(Clone)]
pub struct GraphStatistics {
    /// Total number of triples in the graph
    total_triples: Arc<AtomicU64>,
    /// Total number of distinct subjects
    distinct_subjects: Arc<AtomicU64>,
    /// Total number of distinct predicates
    distinct_predicates: Arc<AtomicU64>,
    /// Total number of distinct objects
    distinct_objects: Arc<AtomicU64>,
    /// Statistics per predicate
    predicate_stats: Arc<RwLock<HashMap<String, PredicateStatistics>>>,
    /// Pattern selectivity history
    pattern_selectivity: Arc<RwLock<HashMap<String, SelectivityInfo>>>,
    /// Metrics registry (for future use)
    #[allow(dead_code)]
    metrics: Arc<MetricsRegistry>,
    /// Statistics collection timestamp
    last_updated: Arc<RwLock<Instant>>,
}

/// Statistics for a specific predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateStatistics {
    /// Number of triples with this predicate
    pub count: u64,
    /// Number of distinct subjects
    pub distinct_subjects: u64,
    /// Number of distinct objects
    pub distinct_objects: u64,
    /// Average objects per subject
    pub avg_objects_per_subject: f64,
    /// Average subjects per object (inverse property)
    pub avg_subjects_per_object: f64,
    /// Minimum cardinality observed
    pub min_cardinality: u64,
    /// Maximum cardinality observed
    pub max_cardinality: u64,
}

/// Selectivity information for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectivityInfo {
    /// Pattern signature (hash of pattern structure)
    pub pattern_signature: String,
    /// Observed selectivity (0.0 to 1.0)
    pub observed_selectivity: f64,
    /// Number of observations
    pub observation_count: u64,
    /// Last observed timestamp (as milliseconds)
    pub last_observed_ms: u128,
    /// Estimated result size
    pub estimated_result_size: u64,
}

/// Query execution statistics for feedback loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExecutionStats {
    /// Query signature
    pub query_signature: String,
    /// Actual execution time
    pub execution_time: Duration,
    /// Estimated execution time
    pub estimated_time: Duration,
    /// Actual result count
    pub actual_results: u64,
    /// Estimated result count
    pub estimated_results: u64,
    /// Memory used in bytes
    pub memory_bytes: u64,
    /// CPU time used
    pub cpu_time: Duration,
}

impl GraphStatistics {
    /// Create a new graph statistics collector
    pub fn new() -> Self {
        let metrics = MetricsRegistry::new();

        Self {
            total_triples: Arc::new(AtomicU64::new(0)),
            distinct_subjects: Arc::new(AtomicU64::new(0)),
            distinct_predicates: Arc::new(AtomicU64::new(0)),
            distinct_objects: Arc::new(AtomicU64::new(0)),
            predicate_stats: Arc::new(RwLock::new(HashMap::new())),
            pattern_selectivity: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(metrics),
            last_updated: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Update statistics after inserting a triple
    pub fn record_insert(&self, triple: &Triple) -> Result<(), OxirsError> {
        self.total_triples.fetch_add(1, Ordering::Relaxed);

        // Update predicate-specific statistics
        if let crate::model::Predicate::NamedNode(predicate) = triple.predicate() {
            let pred_str = predicate.as_str().to_string();

            let mut stats = self.predicate_stats.write().map_err(|e| {
                OxirsError::Store(format!("Failed to write predicate stats: {}", e))
            })?;

            let pred_stat = stats
                .entry(pred_str.clone())
                .or_insert_with(|| PredicateStatistics {
                    count: 0,
                    distinct_subjects: 0,
                    distinct_objects: 0,
                    avg_objects_per_subject: 0.0,
                    avg_subjects_per_object: 0.0,
                    min_cardinality: u64::MAX,
                    max_cardinality: 0,
                });

            pred_stat.count += 1;

            // Update metrics
            let counter = Counter::new("graph.triples.total".to_string());
            counter.add(1);

            let pred_counter = Counter::new(format!("graph.predicate.{}.count", pred_str));
            pred_counter.add(1);
        }

        // Update last updated timestamp
        if let Ok(mut last) = self.last_updated.write() {
            *last = Instant::now();
        }

        Ok(())
    }

    /// Update statistics after removing a triple
    pub fn record_remove(&self, triple: &Triple) -> Result<(), OxirsError> {
        let current = self.total_triples.load(Ordering::Relaxed);
        if current > 0 {
            self.total_triples.fetch_sub(1, Ordering::Relaxed);
        }

        // Update predicate-specific statistics
        if let crate::model::Predicate::NamedNode(predicate) = triple.predicate() {
            let pred_str = predicate.as_str().to_string();

            let mut stats = self.predicate_stats.write().map_err(|e| {
                OxirsError::Store(format!("Failed to write predicate stats: {}", e))
            })?;

            if let Some(pred_stat) = stats.get_mut(&pred_str) {
                if pred_stat.count > 0 {
                    pred_stat.count -= 1;
                }
            }

            // Update metrics
            let counter = Counter::new("graph.triples.removed".to_string());
            counter.add(1);
        }

        // Update last updated timestamp
        if let Ok(mut last) = self.last_updated.write() {
            *last = Instant::now();
        }

        Ok(())
    }

    /// Get total number of triples
    pub fn total_triples(&self) -> u64 {
        self.total_triples.load(Ordering::Relaxed)
    }

    /// Get statistics for a specific predicate
    pub fn get_predicate_stats(&self, predicate: &str) -> Option<PredicateStatistics> {
        self.predicate_stats.read().ok()?.get(predicate).cloned()
    }

    /// Estimate cardinality for a triple pattern
    pub fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> u64 {
        let total = self.total_triples() as f64;
        if total == 0.0 {
            return 0;
        }

        let mut selectivity = 1.0;

        // Adjust selectivity based on bound terms
        if pattern.subject().is_some() {
            selectivity *= 0.001; // Bound subject is very selective
        } else {
            selectivity *= 0.5;
        }

        if let Some(crate::model::pattern::PredicatePattern::NamedNode(pred)) = pattern.predicate()
        {
            // Use actual predicate statistics if available
            if let Some(stats) = self.get_predicate_stats(pred.as_str()) {
                let pred_selectivity = stats.count as f64 / total;
                selectivity *= pred_selectivity;
            } else {
                selectivity *= 0.1; // Default predicate selectivity
            }
        } else {
            selectivity *= 0.5;
        }

        if pattern.object().is_some() {
            selectivity *= 0.001; // Bound object is very selective
        } else {
            selectivity *= 0.5;
        }

        (total * selectivity).max(1.0) as u64
    }

    /// Estimate cardinality for algebra triple pattern
    pub fn estimate_algebra_pattern_cardinality(&self, pattern: &AlgebraTriplePattern) -> u64 {
        let total = self.total_triples() as f64;
        if total == 0.0 {
            return 0;
        }

        let mut selectivity = 1.0;

        // Adjust based on bound terms
        match &pattern.subject {
            TermPattern::Variable(_) => selectivity *= 0.5,
            _ => selectivity *= 0.001,
        }

        match &pattern.predicate {
            TermPattern::Variable(_) => selectivity *= 0.5,
            TermPattern::NamedNode(pred) => {
                if let Some(stats) = self.get_predicate_stats(pred.as_str()) {
                    selectivity *= stats.count as f64 / total;
                } else {
                    selectivity *= 0.1;
                }
            }
            _ => selectivity *= 0.1,
        }

        match &pattern.object {
            TermPattern::Variable(_) => selectivity *= 0.5,
            _ => selectivity *= 0.001,
        }

        (total * selectivity).max(1.0) as u64
    }

    /// Record actual query execution for adaptive learning
    pub fn record_query_execution(&self, stats: QueryExecutionStats) -> Result<(), OxirsError> {
        // Record metrics using counters
        let exec_counter = Counter::new("query.execution.total".to_string());
        exec_counter.add(1);

        let time_counter = Counter::new("query.execution.time_ms".to_string());
        time_counter.add(stats.execution_time.as_millis() as u64);

        let accuracy_ratio = if stats.estimated_results > 0 {
            stats.actual_results as f64 / stats.estimated_results as f64
        } else {
            1.0
        };

        let histogram = Histogram::new("query.estimation.accuracy".to_string());
        histogram.observe(accuracy_ratio);

        // Update selectivity information based on actual results
        let observed_selectivity = if self.total_triples() > 0 {
            stats.actual_results as f64 / self.total_triples() as f64
        } else {
            0.0
        };

        let mut pattern_sel = self
            .pattern_selectivity
            .write()
            .map_err(|e| OxirsError::Query(format!("Failed to write selectivity: {}", e)))?;

        let selectivity_info = pattern_sel
            .entry(stats.query_signature.clone())
            .or_insert_with(|| SelectivityInfo {
                pattern_signature: stats.query_signature.clone(),
                observed_selectivity: 0.0,
                observation_count: 0,
                last_observed_ms: 0,
                estimated_result_size: 0,
            });

        // Update with exponential moving average
        let alpha = 0.3; // Weight for new observation
        selectivity_info.observed_selectivity =
            alpha * observed_selectivity + (1.0 - alpha) * selectivity_info.observed_selectivity;
        selectivity_info.observation_count += 1;
        selectivity_info.last_observed_ms = Instant::now().elapsed().as_millis();
        selectivity_info.estimated_result_size = stats.actual_results;

        Ok(())
    }

    /// Get learned selectivity for a pattern
    pub fn get_learned_selectivity(&self, pattern_signature: &str) -> Option<f64> {
        self.pattern_selectivity
            .read()
            .ok()?
            .get(pattern_signature)
            .map(|info| info.observed_selectivity)
    }

    /// Export statistics to JSON for persistence
    pub fn export_to_json(&self) -> Result<String, OxirsError> {
        let stats = self
            .predicate_stats
            .read()
            .map_err(|e| OxirsError::Serialize(format!("Failed to read stats: {}", e)))?;

        serde_json::to_string_pretty(&*stats).map_err(|e| OxirsError::Serialize(e.to_string()))
    }

    /// Import statistics from JSON
    pub fn import_from_json(&self, json: &str) -> Result<(), OxirsError> {
        let stats: HashMap<String, PredicateStatistics> =
            serde_json::from_str(json).map_err(|e| OxirsError::Parse(e.to_string()))?;

        let mut current_stats = self
            .predicate_stats
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to write stats: {}", e)))?;

        *current_stats = stats;

        // Recalculate total triples
        let total: u64 = current_stats.values().map(|s| s.count).sum();
        self.total_triples.store(total, Ordering::Relaxed);

        Ok(())
    }

    /// Full statistics recomputation (expensive operation)
    pub fn recompute_from_triples(&self, triples: &[Triple]) -> Result<(), OxirsError> {
        tracing::info!("Recomputing statistics from {} triples", triples.len());

        let start = Instant::now();

        // Reset counters
        self.total_triples
            .store(triples.len() as u64, Ordering::Relaxed);

        let mut predicate_counts: HashMap<String, PredicateStatistics> = HashMap::new();
        let mut subject_counts: HashMap<String, u64> = HashMap::new();
        let mut object_counts: HashMap<String, u64> = HashMap::new();

        // First pass: count triples per predicate
        for triple in triples {
            if let crate::model::Predicate::NamedNode(pred) = triple.predicate() {
                let pred_str = pred.as_str().to_string();

                let stat = predicate_counts.entry(pred_str.clone()).or_insert_with(|| {
                    PredicateStatistics {
                        count: 0,
                        distinct_subjects: 0,
                        distinct_objects: 0,
                        avg_objects_per_subject: 0.0,
                        avg_subjects_per_object: 0.0,
                        min_cardinality: u64::MAX,
                        max_cardinality: 0,
                    }
                });

                stat.count += 1;

                // Track subjects and objects for this predicate
                if let crate::model::Subject::NamedNode(subj) = triple.subject() {
                    *subject_counts
                        .entry(format!("{}:{}", pred_str, subj.as_str()))
                        .or_insert(0) += 1;
                }

                if let crate::model::Object::NamedNode(obj) = triple.object() {
                    *object_counts
                        .entry(format!("{}:{}", pred_str, obj.as_str()))
                        .or_insert(0) += 1;
                }
            }
        }

        // Second pass: calculate distinct counts and averages
        for (pred_str, stat) in predicate_counts.iter_mut() {
            let prefix = format!("{}:", pred_str);

            stat.distinct_subjects = subject_counts
                .keys()
                .filter(|k| k.starts_with(&prefix))
                .count() as u64;

            stat.distinct_objects = object_counts
                .keys()
                .filter(|k| k.starts_with(&prefix))
                .count() as u64;

            if stat.distinct_subjects > 0 {
                stat.avg_objects_per_subject = stat.count as f64 / stat.distinct_subjects as f64;
            }

            if stat.distinct_objects > 0 {
                stat.avg_subjects_per_object = stat.count as f64 / stat.distinct_objects as f64;
            }
        }

        // Update stored statistics
        let mut stats = self
            .predicate_stats
            .write()
            .map_err(|e| OxirsError::Store(format!("Failed to write stats: {}", e)))?;
        *stats = predicate_counts;

        // Update distinct counts
        self.distinct_predicates
            .store(stats.len() as u64, Ordering::Relaxed);

        let elapsed = start.elapsed();
        tracing::info!("Statistics recomputation completed in {:?}", elapsed);

        Ok(())
    }

    /// Get all statistics as a summary
    pub fn summary(&self) -> StatisticsSummary {
        StatisticsSummary {
            total_triples: self.total_triples(),
            distinct_subjects: self.distinct_subjects.load(Ordering::Relaxed),
            distinct_predicates: self.distinct_predicates.load(Ordering::Relaxed),
            distinct_objects: self.distinct_objects.load(Ordering::Relaxed),
            predicate_count: self
                .predicate_stats
                .read()
                .ok()
                .map(|s| s.len())
                .unwrap_or(0),
            last_updated: self.last_updated.read().ok().map(|t| *t),
        }
    }
}

impl Default for GraphStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsSummary {
    pub total_triples: u64,
    pub distinct_subjects: u64,
    pub distinct_predicates: u64,
    pub distinct_objects: u64,
    pub predicate_count: usize,
    #[serde(skip)]
    pub last_updated: Option<Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_statistics_creation() {
        let stats = GraphStatistics::new();
        assert_eq!(stats.total_triples(), 0);
    }

    #[test]
    fn test_record_insert() {
        let stats = GraphStatistics::new();

        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("value");

        let triple = Triple::new(subject, predicate, object);

        stats.record_insert(&triple).unwrap();
        assert_eq!(stats.total_triples(), 1);
    }

    #[test]
    fn test_record_remove() {
        let stats = GraphStatistics::new();

        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("value");

        let triple = Triple::new(subject, predicate, object);

        stats.record_insert(&triple).unwrap();
        assert_eq!(stats.total_triples(), 1);

        stats.record_remove(&triple).unwrap();
        assert_eq!(stats.total_triples(), 0);
    }

    #[test]
    fn test_predicate_statistics() {
        let stats = GraphStatistics::new();

        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("value");

        let triple = Triple::new(subject, predicate.clone(), object);

        stats.record_insert(&triple).unwrap();

        let pred_stats = stats.get_predicate_stats(predicate.as_str());
        assert!(pred_stats.is_some());
        assert_eq!(pred_stats.unwrap().count, 1);
    }

    #[test]
    fn test_pattern_cardinality_estimation() {
        let stats = GraphStatistics::new();

        // Add some triples
        for i in 0..100 {
            let subject = NamedNode::new(format!("http://example.org/s{}", i)).unwrap();
            let predicate = NamedNode::new("http://example.org/p").unwrap();
            let object = Literal::new(format!("value{}", i));

            let triple = Triple::new(subject, predicate, object);
            stats.record_insert(&triple).unwrap();
        }

        // Estimate cardinality for a pattern with bound predicate
        let pattern = TriplePattern::new(
            None,
            Some(crate::model::pattern::PredicatePattern::NamedNode(
                NamedNode::new("http://example.org/p").unwrap(),
            )),
            None,
        );

        let estimated = stats.estimate_pattern_cardinality(&pattern);
        assert!(estimated > 0);
        assert!(estimated <= 100);
    }

    #[test]
    fn test_query_execution_recording() {
        let stats = GraphStatistics::new();

        let exec_stats = QueryExecutionStats {
            query_signature: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            execution_time: Duration::from_millis(50),
            estimated_time: Duration::from_millis(100),
            actual_results: 42,
            estimated_results: 50,
            memory_bytes: 1024 * 1024,
            cpu_time: Duration::from_millis(30),
        };

        stats.record_query_execution(exec_stats).unwrap();

        let learned = stats.get_learned_selectivity("SELECT ?s WHERE { ?s ?p ?o }");
        assert!(learned.is_some());
    }

    #[test]
    fn test_statistics_export_import() {
        let stats = GraphStatistics::new();

        // Add some data
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("value");

        let triple = Triple::new(subject, predicate, object);
        stats.record_insert(&triple).unwrap();

        // Export
        let json = stats.export_to_json().unwrap();
        assert!(!json.is_empty());

        // Import to new instance
        let stats2 = GraphStatistics::new();
        stats2.import_from_json(&json).unwrap();

        assert_eq!(stats2.total_triples(), 1);
    }

    #[test]
    fn test_statistics_summary() {
        let stats = GraphStatistics::new();

        let summary = stats.summary();
        assert_eq!(summary.total_triples, 0);
        assert_eq!(summary.predicate_count, 0);
    }
}
