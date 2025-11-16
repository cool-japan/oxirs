//! Statistics collection for cost-based query optimization
//!
//! This module collects and maintains statistics about the stored triples
//! to enable cost-based query optimization. Statistics include:
//! - Cardinality estimates for subjects, predicates, and objects
//! - Selectivity estimates for different query patterns
//! - Distribution of triples across indexes

use crate::dictionary::NodeId;
use crate::error::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for statistics collection
#[derive(Debug, Clone)]
pub struct StatisticsConfig {
    /// Whether to enable statistics collection
    pub enabled: bool,
    /// How often to update statistics (in number of modifications)
    pub update_threshold: usize,
    /// Whether to use sampling for large datasets
    pub use_sampling: bool,
    /// Sample rate (0.0 to 1.0)
    pub sample_rate: f64,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_threshold: 1000,
            use_sampling: true,
            sample_rate: 0.1, // 10% sampling for very large datasets
        }
    }
}

/// Cardinality statistics for a specific position
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CardinalityStats {
    /// Total number of distinct values
    pub distinct_count: u64,
    /// Most frequent values (top 10)
    pub most_frequent: Vec<(NodeId, u64)>,
    /// Least frequent values (bottom 10)
    pub least_frequent: Vec<(NodeId, u64)>,
}

/// Pattern selectivity estimate
#[derive(Debug, Clone, Copy)]
pub struct SelectivityEstimate {
    /// Estimated number of results
    pub estimated_count: u64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// Triple store statistics
pub struct TripleStatistics {
    /// Configuration
    config: StatisticsConfig,
    /// Total number of triples
    total_triples: AtomicU64,
    /// Subject cardinality
    subject_stats: Arc<DashMap<NodeId, u64>>,
    /// Predicate cardinality
    predicate_stats: Arc<DashMap<NodeId, u64>>,
    /// Object cardinality
    object_stats: Arc<DashMap<NodeId, u64>>,
    /// Subject-Predicate pairs
    sp_pairs: Arc<DashMap<(NodeId, NodeId), u64>>,
    /// Predicate-Object pairs
    po_pairs: Arc<DashMap<(NodeId, NodeId), u64>>,
    /// Subject-Object pairs
    so_pairs: Arc<DashMap<(NodeId, NodeId), u64>>,
    /// Number of modifications since last update
    modifications: AtomicU64,
}

impl TripleStatistics {
    /// Create a new statistics collector
    pub fn new(config: StatisticsConfig) -> Self {
        Self {
            config,
            total_triples: AtomicU64::new(0),
            subject_stats: Arc::new(DashMap::new()),
            predicate_stats: Arc::new(DashMap::new()),
            object_stats: Arc::new(DashMap::new()),
            sp_pairs: Arc::new(DashMap::new()),
            po_pairs: Arc::new(DashMap::new()),
            so_pairs: Arc::new(DashMap::new()),
            modifications: AtomicU64::new(0),
        }
    }

    /// Record an insert operation
    pub fn record_insert(&self, subject: NodeId, predicate: NodeId, object: NodeId) {
        if !self.config.enabled {
            return;
        }

        // Increment total
        self.total_triples.fetch_add(1, Ordering::Relaxed);

        // Update per-position stats
        *self.subject_stats.entry(subject).or_insert(0) += 1;
        *self.predicate_stats.entry(predicate).or_insert(0) += 1;
        *self.object_stats.entry(object).or_insert(0) += 1;

        // Update pair stats
        *self.sp_pairs.entry((subject, predicate)).or_insert(0) += 1;
        *self.po_pairs.entry((predicate, object)).or_insert(0) += 1;
        *self.so_pairs.entry((subject, object)).or_insert(0) += 1;

        // Track modifications
        self.modifications.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a delete operation
    pub fn record_delete(&self, subject: NodeId, predicate: NodeId, object: NodeId) {
        if !self.config.enabled {
            return;
        }

        // Decrement total
        self.total_triples.fetch_sub(1, Ordering::Relaxed);

        // Update per-position stats
        if let Some(mut count) = self.subject_stats.get_mut(&subject) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                drop(count);
                self.subject_stats.remove(&subject);
            }
        }

        if let Some(mut count) = self.predicate_stats.get_mut(&predicate) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                drop(count);
                self.predicate_stats.remove(&predicate);
            }
        }

        if let Some(mut count) = self.object_stats.get_mut(&object) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                drop(count);
                self.object_stats.remove(&object);
            }
        }

        // Update pair stats
        let sp_key = (subject, predicate);
        if let Some(mut count) = self.sp_pairs.get_mut(&sp_key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                drop(count);
                self.sp_pairs.remove(&sp_key);
            }
        }

        let po_key = (predicate, object);
        if let Some(mut count) = self.po_pairs.get_mut(&po_key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                drop(count);
                self.po_pairs.remove(&po_key);
            }
        }

        let so_key = (subject, object);
        if let Some(mut count) = self.so_pairs.get_mut(&so_key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                drop(count);
                self.so_pairs.remove(&so_key);
            }
        }

        // Track modifications
        self.modifications.fetch_add(1, Ordering::Relaxed);
    }

    /// Estimate selectivity for a query pattern
    ///
    /// Returns an estimate of how many triples match the given pattern.
    /// Uses various heuristics based on available statistics.
    pub fn estimate_selectivity(
        &self,
        subject: Option<NodeId>,
        predicate: Option<NodeId>,
        object: Option<NodeId>,
    ) -> SelectivityEstimate {
        let total = self.total_triples.load(Ordering::Relaxed);

        // (?, ?, ?) - all triples
        if subject.is_none() && predicate.is_none() && object.is_none() {
            return SelectivityEstimate {
                estimated_count: total,
                confidence: 1.0,
            };
        }

        // (S, P, O) - exact match (0 or 1)
        if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
            // Check all three positions
            let s_exists = self.subject_stats.contains_key(&s);
            let p_exists = self.predicate_stats.contains_key(&p);
            let o_exists = self.object_stats.contains_key(&o);

            if s_exists && p_exists && o_exists {
                return SelectivityEstimate {
                    estimated_count: 1,
                    confidence: 0.9, // High confidence for exact matches
                };
            } else {
                return SelectivityEstimate {
                    estimated_count: 0,
                    confidence: 1.0,
                };
            }
        }

        // (S, P, ?) - subject and predicate bound
        if let (Some(s), Some(p), None) = (subject, predicate, object) {
            if let Some(count) = self.sp_pairs.get(&(s, p)) {
                return SelectivityEstimate {
                    estimated_count: *count,
                    confidence: 0.95,
                };
            }
        }

        // (?, P, O) - predicate and object bound
        if let (None, Some(p), Some(o)) = (subject, predicate, object) {
            if let Some(count) = self.po_pairs.get(&(p, o)) {
                return SelectivityEstimate {
                    estimated_count: *count,
                    confidence: 0.95,
                };
            }
        }

        // (S, ?, O) - subject and object bound
        if let (Some(s), None, Some(o)) = (subject, predicate, object) {
            if let Some(count) = self.so_pairs.get(&(s, o)) {
                return SelectivityEstimate {
                    estimated_count: *count,
                    confidence: 0.95,
                };
            }
        }

        // (S, ?, ?) - subject only
        if let (Some(s), None, None) = (subject, predicate, object) {
            if let Some(count) = self.subject_stats.get(&s) {
                return SelectivityEstimate {
                    estimated_count: *count,
                    confidence: 0.9,
                };
            }
        }

        // (?, P, ?) - predicate only
        if let (None, Some(p), None) = (subject, predicate, object) {
            if let Some(count) = self.predicate_stats.get(&p) {
                return SelectivityEstimate {
                    estimated_count: *count,
                    confidence: 0.9,
                };
            }
        }

        // (?, ?, O) - object only
        if let (None, None, Some(o)) = (subject, predicate, object) {
            if let Some(count) = self.object_stats.get(&o) {
                return SelectivityEstimate {
                    estimated_count: *count,
                    confidence: 0.9,
                };
            }
        }

        // Fallback: no stats available, use rough estimate
        SelectivityEstimate {
            estimated_count: total / 10, // Wild guess: 10% of data
            confidence: 0.1,             // Very low confidence
        }
    }

    /// Get subject cardinality statistics
    pub fn subject_cardinality(&self) -> CardinalityStats {
        self.compute_cardinality_stats(&self.subject_stats)
    }

    /// Get predicate cardinality statistics
    pub fn predicate_cardinality(&self) -> CardinalityStats {
        self.compute_cardinality_stats(&self.predicate_stats)
    }

    /// Get object cardinality statistics
    pub fn object_cardinality(&self) -> CardinalityStats {
        self.compute_cardinality_stats(&self.object_stats)
    }

    /// Compute cardinality statistics from a frequency map
    fn compute_cardinality_stats(&self, stats_map: &DashMap<NodeId, u64>) -> CardinalityStats {
        let distinct_count = stats_map.len() as u64;

        // Collect all values
        let mut values: Vec<(NodeId, u64)> = stats_map
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();

        // Sort by frequency (descending)
        values.sort_by(|a, b| b.1.cmp(&a.1));

        let most_frequent = values.iter().take(10).cloned().collect();

        values.reverse(); // Sort ascending for least frequent
        let least_frequent = values.iter().take(10).cloned().collect();

        CardinalityStats {
            distinct_count,
            most_frequent,
            least_frequent,
        }
    }

    /// Get total number of triples
    pub fn total_triples(&self) -> u64 {
        self.total_triples.load(Ordering::Relaxed)
    }

    /// Get number of distinct subjects
    pub fn distinct_subjects(&self) -> usize {
        self.subject_stats.len()
    }

    /// Get number of distinct predicates
    pub fn distinct_predicates(&self) -> usize {
        self.predicate_stats.len()
    }

    /// Get number of distinct objects
    pub fn distinct_objects(&self) -> usize {
        self.object_stats.len()
    }

    /// Check if statistics need updating
    pub fn needs_update(&self) -> bool {
        self.modifications.load(Ordering::Relaxed) >= self.config.update_threshold as u64
    }

    /// Reset modification counter (call after statistics update)
    pub fn reset_modifications(&self) {
        self.modifications.store(0, Ordering::Relaxed);
    }

    /// Clear all statistics
    pub fn clear(&self) {
        self.total_triples.store(0, Ordering::Relaxed);
        self.subject_stats.clear();
        self.predicate_stats.clear();
        self.object_stats.clear();
        self.sp_pairs.clear();
        self.po_pairs.clear();
        self.so_pairs.clear();
        self.modifications.store(0, Ordering::Relaxed);
    }

    /// Export statistics to a serializable format
    pub fn export(&self) -> StatisticsSnapshot {
        StatisticsSnapshot {
            total_triples: self.total_triples.load(Ordering::Relaxed),
            distinct_subjects: self.distinct_subjects() as u64,
            distinct_predicates: self.distinct_predicates() as u64,
            distinct_objects: self.distinct_objects() as u64,
            subject_cardinality: self.subject_cardinality(),
            predicate_cardinality: self.predicate_cardinality(),
            object_cardinality: self.object_cardinality(),
        }
    }
}

/// Serializable snapshot of statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsSnapshot {
    /// Total number of triples
    pub total_triples: u64,
    /// Number of distinct subjects
    pub distinct_subjects: u64,
    /// Number of distinct predicates
    pub distinct_predicates: u64,
    /// Number of distinct objects
    pub distinct_objects: u64,
    /// Subject cardinality statistics
    pub subject_cardinality: CardinalityStats,
    /// Predicate cardinality statistics
    pub predicate_cardinality: CardinalityStats,
    /// Object cardinality statistics
    pub object_cardinality: CardinalityStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_creation() {
        let config = StatisticsConfig::default();
        let stats = TripleStatistics::new(config);

        assert_eq!(stats.total_triples(), 0);
        assert_eq!(stats.distinct_subjects(), 0);
        assert_eq!(stats.distinct_predicates(), 0);
        assert_eq!(stats.distinct_objects(), 0);
    }

    #[test]
    fn test_record_insert() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        let s1 = NodeId::new(1);
        let p1 = NodeId::new(2);
        let o1 = NodeId::new(3);

        stats.record_insert(s1, p1, o1);

        assert_eq!(stats.total_triples(), 1);
        assert_eq!(stats.distinct_subjects(), 1);
        assert_eq!(stats.distinct_predicates(), 1);
        assert_eq!(stats.distinct_objects(), 1);
    }

    #[test]
    fn test_record_delete() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        let s1 = NodeId::new(1);
        let p1 = NodeId::new(2);
        let o1 = NodeId::new(3);

        stats.record_insert(s1, p1, o1);
        assert_eq!(stats.total_triples(), 1);

        stats.record_delete(s1, p1, o1);
        assert_eq!(stats.total_triples(), 0);
        assert_eq!(stats.distinct_subjects(), 0);
    }

    #[test]
    fn test_multiple_inserts() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        let s1 = NodeId::new(1);
        let s2 = NodeId::new(2);
        let p1 = NodeId::new(3);
        let o1 = NodeId::new(4);
        let o2 = NodeId::new(5);

        stats.record_insert(s1, p1, o1);
        stats.record_insert(s2, p1, o2);

        assert_eq!(stats.total_triples(), 2);
        assert_eq!(stats.distinct_subjects(), 2);
        assert_eq!(stats.distinct_predicates(), 1);
        assert_eq!(stats.distinct_objects(), 2);
    }

    #[test]
    fn test_selectivity_all_wildcards() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        // Insert 100 triples
        for i in 0..100 {
            stats.record_insert(NodeId::new(i), NodeId::new(1), NodeId::new(i + 100));
        }

        let estimate = stats.estimate_selectivity(None, None, None);
        assert_eq!(estimate.estimated_count, 100);
        assert_eq!(estimate.confidence, 1.0);
    }

    #[test]
    fn test_selectivity_exact_match() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        let s1 = NodeId::new(1);
        let p1 = NodeId::new(2);
        let o1 = NodeId::new(3);

        stats.record_insert(s1, p1, o1);

        let estimate = stats.estimate_selectivity(Some(s1), Some(p1), Some(o1));
        assert_eq!(estimate.estimated_count, 1);
        assert!(estimate.confidence >= 0.9);
    }

    #[test]
    fn test_selectivity_subject_predicate() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        let s1 = NodeId::new(1);
        let p1 = NodeId::new(2);

        // Insert 5 triples with same S and P
        for i in 0..5 {
            stats.record_insert(s1, p1, NodeId::new(i + 100));
        }

        let estimate = stats.estimate_selectivity(Some(s1), Some(p1), None);
        assert_eq!(estimate.estimated_count, 5);
        assert!(estimate.confidence >= 0.9);
    }

    #[test]
    fn test_selectivity_predicate_only() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        let p1 = NodeId::new(2);

        // Insert 10 triples with same predicate
        for i in 0..10 {
            stats.record_insert(NodeId::new(i), p1, NodeId::new(i + 100));
        }

        let estimate = stats.estimate_selectivity(None, Some(p1), None);
        assert_eq!(estimate.estimated_count, 10);
        assert!(estimate.confidence >= 0.9);
    }

    #[test]
    fn test_cardinality_stats() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        let p1 = NodeId::new(1);
        let p2 = NodeId::new(2);

        // p1 appears 10 times
        for i in 0..10 {
            stats.record_insert(NodeId::new(i), p1, NodeId::new(i + 100));
        }

        // p2 appears 5 times
        for i in 0..5 {
            stats.record_insert(NodeId::new(i), p2, NodeId::new(i + 200));
        }

        let card = stats.predicate_cardinality();
        assert_eq!(card.distinct_count, 2);
        assert!(!card.most_frequent.is_empty());
    }

    #[test]
    fn test_needs_update() {
        let config = StatisticsConfig {
            update_threshold: 10,
            ..Default::default()
        };
        let stats = TripleStatistics::new(config);

        assert!(!stats.needs_update());

        // Insert 10 triples
        for i in 0..10 {
            stats.record_insert(NodeId::new(i), NodeId::new(1), NodeId::new(i + 100));
        }

        assert!(stats.needs_update());

        stats.reset_modifications();
        assert!(!stats.needs_update());
    }

    #[test]
    fn test_clear() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        for i in 0..10 {
            stats.record_insert(NodeId::new(i), NodeId::new(1), NodeId::new(i + 100));
        }

        assert_eq!(stats.total_triples(), 10);

        stats.clear();

        assert_eq!(stats.total_triples(), 0);
        assert_eq!(stats.distinct_subjects(), 0);
    }

    #[test]
    fn test_export_snapshot() {
        let stats = TripleStatistics::new(StatisticsConfig::default());

        for i in 0..10 {
            stats.record_insert(NodeId::new(i), NodeId::new(1), NodeId::new(i + 100));
        }

        let snapshot = stats.export();
        assert_eq!(snapshot.total_triples, 10);
        assert_eq!(snapshot.distinct_predicates, 1);
    }

    #[test]
    fn test_disabled_statistics() {
        let config = StatisticsConfig {
            enabled: false,
            ..Default::default()
        };
        let stats = TripleStatistics::new(config);

        stats.record_insert(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        // Should not record anything when disabled
        assert_eq!(stats.total_triples(), 0);
    }
}
