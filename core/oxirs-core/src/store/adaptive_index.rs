//! Adaptive indexing for RDF graphs that automatically adjusts based on query patterns
//!
//! This module provides indexes that learn from query patterns and automatically
//! create, update, or remove indexes to optimize query performance.

use crate::model::{Object, Predicate, Subject, Triple};
use crate::store::{IndexType, IndexedGraph};
use crate::OxirsError;
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Query pattern types for tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryPattern {
    /// Subject-based query (? ? ?)
    SubjectQuery,
    /// Predicate-based query (? p ?)
    PredicateQuery,
    /// Object-based query (? ? o)
    ObjectQuery,
    /// Subject-Predicate query (s p ?)
    SubjectPredicate,
    /// Subject-Object query (s ? o)
    SubjectObject,
    /// Predicate-Object query (? p o)
    PredicateObject,
    /// Specific triple query (s p o)
    SpecificTriple,
    /// Full scan query (? ? ?)
    FullScan,
}

impl QueryPattern {
    /// Determine pattern from optional components
    pub fn from_components(
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Self {
        match (subject.is_some(), predicate.is_some(), object.is_some()) {
            (true, true, true) => QueryPattern::SpecificTriple,
            (true, true, false) => QueryPattern::SubjectPredicate,
            (true, false, true) => QueryPattern::SubjectObject,
            (false, true, true) => QueryPattern::PredicateObject,
            (true, false, false) => QueryPattern::SubjectQuery,
            (false, true, false) => QueryPattern::PredicateQuery,
            (false, false, true) => QueryPattern::ObjectQuery,
            (false, false, false) => QueryPattern::FullScan,
        }
    }

    /// Get recommended index type for this pattern
    pub fn recommended_index(&self) -> Option<IndexType> {
        match self {
            QueryPattern::SubjectQuery | QueryPattern::SubjectPredicate => Some(IndexType::SPO),
            QueryPattern::PredicateQuery | QueryPattern::PredicateObject => Some(IndexType::POS),
            QueryPattern::ObjectQuery | QueryPattern::SubjectObject => Some(IndexType::OSP),
            QueryPattern::SpecificTriple => Some(IndexType::SPO), // Any index works
            QueryPattern::FullScan => None,                       // No specific index helps
        }
    }
}

/// Statistics for a specific query pattern
#[derive(Debug, Clone)]
pub struct PatternStats {
    /// Number of times this pattern was queried
    pub query_count: u64,
    /// Total time spent on queries of this pattern
    pub total_time: Duration,
    /// Average result set size
    pub avg_result_size: f64,
    /// Last query timestamp
    pub last_queried: Instant,
    /// Moving average of query frequency (queries per second)
    pub query_frequency: f64,
}

impl Default for PatternStats {
    fn default() -> Self {
        Self {
            query_count: 0,
            total_time: Duration::ZERO,
            avg_result_size: 0.0,
            last_queried: Instant::now(),
            query_frequency: 0.0,
        }
    }
}

/// Adaptive index configuration
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Minimum queries before considering index creation
    pub min_queries_for_index: u64,
    /// Minimum query frequency for index creation (queries/sec)
    pub min_frequency_for_index: f64,
    /// Maximum number of adaptive indexes
    pub max_adaptive_indexes: usize,
    /// Time window for query pattern analysis
    pub analysis_window: Duration,
    /// Index maintenance interval
    pub maintenance_interval: Duration,
    /// Cost threshold for index creation (relative to full scan)
    pub index_cost_threshold: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_queries_for_index: 100,
            min_frequency_for_index: 0.1,
            max_adaptive_indexes: 5,
            analysis_window: Duration::from_secs(300), // 5 minutes
            maintenance_interval: Duration::from_secs(60), // 1 minute
            index_cost_threshold: 0.5,                 // Index if cost < 50% of full scan
        }
    }
}

/// Adaptive index manager
pub struct AdaptiveIndexManager {
    /// Base indexed graph
    base_graph: Arc<RwLock<IndexedGraph>>,
    /// Query pattern statistics
    pattern_stats: Arc<DashMap<QueryPattern, PatternStats>>,
    /// Currently active adaptive indexes
    adaptive_indexes: Arc<RwLock<HashMap<QueryPattern, Box<dyn AdaptiveIndex>>>>,
    /// Configuration
    config: AdaptiveConfig,
    /// Last maintenance timestamp
    last_maintenance: Arc<Mutex<Instant>>,
    /// Query history for pattern analysis
    query_history: Arc<Mutex<VecDeque<(QueryPattern, Instant, Duration)>>>,
}

impl AdaptiveIndexManager {
    /// Create a new adaptive index manager
    pub fn new(base_graph: IndexedGraph, config: AdaptiveConfig) -> Self {
        Self {
            base_graph: Arc::new(RwLock::new(base_graph)),
            pattern_stats: Arc::new(DashMap::new()),
            adaptive_indexes: Arc::new(RwLock::new(HashMap::new())),
            config,
            last_maintenance: Arc::new(Mutex::new(Instant::now())),
            query_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Execute a query with adaptive indexing
    pub fn query(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>, OxirsError> {
        let start = Instant::now();
        let pattern = QueryPattern::from_components(subject, predicate, object);

        // Check if we have an adaptive index for this pattern
        let result = {
            let indexes = self.adaptive_indexes.read();
            if let Some(index) = indexes.get(&pattern) {
                // Use adaptive index
                index.query(subject, predicate, object)
            } else {
                // Fall back to base graph
                let graph = self.base_graph.read();
                Ok(graph.match_pattern(subject, predicate, object))
            }
        }?;

        let duration = start.elapsed();

        // Update statistics
        self.update_pattern_stats(pattern, duration, result.len());

        // Record in history
        {
            let mut history = self.query_history.lock();
            history.push_back((pattern, Instant::now(), duration));

            // Keep only recent history
            let cutoff = Instant::now() - self.config.analysis_window;
            while let Some((_, timestamp, _)) = history.front() {
                if *timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        // Check if maintenance is needed
        self.maybe_run_maintenance();

        Ok(result)
    }

    /// Update statistics for a query pattern
    fn update_pattern_stats(&self, pattern: QueryPattern, duration: Duration, result_size: usize) {
        let mut stats = self.pattern_stats.entry(pattern).or_default();

        let now = Instant::now();
        let time_since_last = now.duration_since(stats.last_queried).as_secs_f64();

        // Update basic stats
        stats.query_count += 1;
        stats.total_time += duration;
        stats.avg_result_size = (stats.avg_result_size * (stats.query_count - 1) as f64
            + result_size as f64)
            / stats.query_count as f64;

        // Update frequency with exponential moving average
        if time_since_last > 0.0 {
            let instant_frequency = 1.0 / time_since_last;
            stats.query_frequency = stats.query_frequency * 0.9 + instant_frequency * 0.1;
        }

        stats.last_queried = now;
    }

    /// Run maintenance if needed
    fn maybe_run_maintenance(&self) {
        let mut last_maintenance = self.last_maintenance.lock();
        if last_maintenance.elapsed() >= self.config.maintenance_interval {
            *last_maintenance = Instant::now();
            drop(last_maintenance);

            // Run maintenance in background
            let self_clone = self.clone();
            std::thread::spawn(move || {
                self_clone.run_maintenance_internal();
            });
        }
    }

    /// Run index maintenance (public for testing only)
    ///
    /// # Note
    /// This method is only intended for use in tests to trigger maintenance
    /// manually. In production, maintenance runs automatically based on the
    /// configured maintenance interval.
    pub fn run_maintenance(&self) {
        self.run_maintenance_internal();
    }

    /// Run index maintenance (internal)
    fn run_maintenance_internal(&self) {
        // Analyze patterns and create/remove indexes
        let patterns_to_index = self.analyze_patterns();

        // Create new indexes
        for pattern in patterns_to_index {
            self.create_adaptive_index(pattern);
        }

        // Remove underused indexes
        self.cleanup_indexes();
    }

    /// Analyze query patterns to determine which indexes to create
    fn analyze_patterns(&self) -> Vec<QueryPattern> {
        let mut candidates = Vec::new();

        for entry in self.pattern_stats.iter() {
            let (pattern, stats) = entry.pair();

            // Skip if already indexed
            if self.adaptive_indexes.read().contains_key(pattern) {
                continue;
            }

            // Check if pattern qualifies for indexing
            if stats.query_count >= self.config.min_queries_for_index
                && stats.query_frequency >= self.config.min_frequency_for_index
            {
                // Estimate cost benefit
                if let Some(benefit) = self.estimate_index_benefit(*pattern, stats) {
                    if benefit >= self.config.index_cost_threshold {
                        candidates.push((*pattern, benefit));
                    }
                }
            }
        }

        // Sort by benefit and take top candidates
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(self.config.max_adaptive_indexes);

        candidates.into_iter().map(|(pattern, _)| pattern).collect()
    }

    /// Estimate the benefit of creating an index for a pattern
    fn estimate_index_benefit(&self, _pattern: QueryPattern, stats: &PatternStats) -> Option<f64> {
        // Simple cost model: benefit = (scan_cost - index_cost) / scan_cost
        let graph = self.base_graph.read();
        let total_triples = graph.len() as f64;

        // Estimate scan cost (proportional to total triples)
        let scan_cost = total_triples;

        // Estimate index cost (proportional to result size)
        let index_cost = stats.avg_result_size;

        if scan_cost > 0.0 {
            Some((scan_cost - index_cost) / scan_cost)
        } else {
            None
        }
    }

    /// Create an adaptive index for a pattern
    fn create_adaptive_index(&self, pattern: QueryPattern) {
        let mut indexes = self.adaptive_indexes.write();

        // Check capacity
        if indexes.len() >= self.config.max_adaptive_indexes {
            return;
        }

        // Create appropriate index type
        let index: Box<dyn AdaptiveIndex> = match pattern {
            QueryPattern::PredicateQuery => Box::new(PredicateIndex::new(self.base_graph.clone())),
            QueryPattern::SubjectPredicate => {
                Box::new(SubjectPredicateIndex::new(self.base_graph.clone()))
            }
            _ => return, // Add more index types as needed
        };

        indexes.insert(pattern, index);
    }

    /// Remove underused indexes
    fn cleanup_indexes(&self) {
        let mut indexes = self.adaptive_indexes.write();
        let stats = self.pattern_stats.clone();

        indexes.retain(|pattern, _| {
            match stats.get(pattern) {
                Some(pattern_stats) => {
                    // Keep if still meeting frequency threshold
                    pattern_stats.query_frequency >= self.config.min_frequency_for_index * 0.5
                }
                _ => false,
            }
        });
    }

    /// Get current statistics
    pub fn get_stats(&self) -> AdaptiveIndexStats {
        let pattern_stats: HashMap<QueryPattern, PatternStats> = self
            .pattern_stats
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();

        let active_indexes: Vec<QueryPattern> =
            self.adaptive_indexes.read().keys().copied().collect();

        let total_queries = pattern_stats.values().map(|s| s.query_count).sum();

        AdaptiveIndexStats {
            pattern_stats,
            active_indexes,
            total_queries,
        }
    }

    /// Insert a triple and update indexes
    pub fn insert(&self, triple: Triple) -> Result<bool, OxirsError> {
        // Insert into base graph
        let inserted = self.base_graph.write().insert(&triple);

        if inserted {
            // Update adaptive indexes
            let indexes = self.adaptive_indexes.read();
            for index in indexes.values() {
                index.insert(&triple)?;
            }
        }

        Ok(inserted)
    }

    /// Remove a triple and update indexes
    pub fn remove(&self, triple: &Triple) -> Result<bool, OxirsError> {
        // Remove from base graph
        let removed = self.base_graph.write().remove(triple);

        if removed {
            // Update adaptive indexes
            let indexes = self.adaptive_indexes.read();
            for index in indexes.values() {
                index.remove(triple)?;
            }
        }

        Ok(removed)
    }
}

// Implement Clone manually to avoid trait object issues
impl Clone for AdaptiveIndexManager {
    fn clone(&self) -> Self {
        Self {
            base_graph: self.base_graph.clone(),
            pattern_stats: self.pattern_stats.clone(),
            adaptive_indexes: self.adaptive_indexes.clone(),
            config: self.config.clone(),
            last_maintenance: Arc::new(Mutex::new(*self.last_maintenance.lock())),
            query_history: self.query_history.clone(),
        }
    }
}

/// Trait for adaptive index implementations
trait AdaptiveIndex: Send + Sync {
    /// Query the index
    fn query(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>, OxirsError>;

    /// Insert a triple
    fn insert(&self, triple: &Triple) -> Result<(), OxirsError>;

    /// Remove a triple
    fn remove(&self, triple: &Triple) -> Result<(), OxirsError>;
}

/// Predicate-based adaptive index
struct PredicateIndex {
    base_graph: Arc<RwLock<IndexedGraph>>,
    predicate_map: Arc<DashMap<Predicate, Vec<Triple>>>,
}

impl PredicateIndex {
    fn new(base_graph: Arc<RwLock<IndexedGraph>>) -> Self {
        let index = Self {
            base_graph: base_graph.clone(),
            predicate_map: Arc::new(DashMap::new()),
        };

        // Build initial index
        let graph = base_graph.read();
        for triple in graph.iter() {
            index
                .predicate_map
                .entry(triple.predicate().clone())
                .or_default()
                .push(triple);
        }

        index
    }
}

impl AdaptiveIndex for PredicateIndex {
    fn query(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>, OxirsError> {
        if let Some(pred) = predicate {
            if let Some(triples) = self.predicate_map.get(pred) {
                let results: Vec<Triple> = triples
                    .iter()
                    .filter(|t| {
                        subject.map_or(true, |s| t.subject() == s)
                            && object.map_or(true, |o| t.object() == o)
                    })
                    .cloned()
                    .collect();
                return Ok(results);
            }
        }

        // Fall back to base graph
        let graph = self.base_graph.read();
        Ok(graph.match_pattern(subject, predicate, object))
    }

    fn insert(&self, triple: &Triple) -> Result<(), OxirsError> {
        self.predicate_map
            .entry(triple.predicate().clone())
            .or_default()
            .push(triple.clone());
        Ok(())
    }

    fn remove(&self, triple: &Triple) -> Result<(), OxirsError> {
        if let Some(mut triples) = self.predicate_map.get_mut(triple.predicate()) {
            triples.retain(|t| t != triple);
        }
        Ok(())
    }
}

/// Subject-Predicate composite index
struct SubjectPredicateIndex {
    base_graph: Arc<RwLock<IndexedGraph>>,
    sp_map: Arc<DashMap<(Subject, Predicate), Vec<Object>>>,
}

impl SubjectPredicateIndex {
    fn new(base_graph: Arc<RwLock<IndexedGraph>>) -> Self {
        let index = Self {
            base_graph: base_graph.clone(),
            sp_map: Arc::new(DashMap::new()),
        };

        // Build initial index
        let graph = base_graph.read();
        for triple in graph.iter() {
            index
                .sp_map
                .entry((triple.subject().clone(), triple.predicate().clone()))
                .or_default()
                .push(triple.object().clone());
        }

        index
    }
}

impl AdaptiveIndex for SubjectPredicateIndex {
    fn query(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>, OxirsError> {
        if let (Some(subj), Some(pred)) = (subject, predicate) {
            if let Some(objects) = self.sp_map.get(&(subj.clone(), pred.clone())) {
                let results: Vec<Triple> = objects
                    .iter()
                    .filter(|o| object.map_or(true, |obj| *o == obj))
                    .map(|o| Triple::new(subj.clone(), pred.clone(), o.clone()))
                    .collect();
                return Ok(results);
            }
        }

        // Fall back to base graph
        let graph = self.base_graph.read();
        Ok(graph.match_pattern(subject, predicate, object))
    }

    fn insert(&self, triple: &Triple) -> Result<(), OxirsError> {
        self.sp_map
            .entry((triple.subject().clone(), triple.predicate().clone()))
            .or_default()
            .push(triple.object().clone());
        Ok(())
    }

    fn remove(&self, triple: &Triple) -> Result<(), OxirsError> {
        if let Some(mut objects) = self
            .sp_map
            .get_mut(&(triple.subject().clone(), triple.predicate().clone()))
        {
            objects.retain(|o| o != triple.object());
        }
        Ok(())
    }
}

/// Statistics for adaptive indexing
#[derive(Debug, Clone)]
pub struct AdaptiveIndexStats {
    pub pattern_stats: HashMap<QueryPattern, PatternStats>,
    pub active_indexes: Vec<QueryPattern>,
    pub total_queries: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NamedNode;

    #[test]
    fn test_query_pattern_detection() {
        let s = Subject::NamedNode(NamedNode::new("http://s").unwrap());
        let p = Predicate::NamedNode(NamedNode::new("http://p").unwrap());
        let o = Object::NamedNode(NamedNode::new("http://o").unwrap());

        assert_eq!(
            QueryPattern::from_components(Some(&s), Some(&p), Some(&o)),
            QueryPattern::SpecificTriple
        );
        assert_eq!(
            QueryPattern::from_components(Some(&s), Some(&p), None),
            QueryPattern::SubjectPredicate
        );
        assert_eq!(
            QueryPattern::from_components(None, Some(&p), None),
            QueryPattern::PredicateQuery
        );
        assert_eq!(
            QueryPattern::from_components(None, None, None),
            QueryPattern::FullScan
        );
    }

    #[test]
    fn test_adaptive_index_creation() {
        let graph = IndexedGraph::new();
        let config = AdaptiveConfig {
            min_queries_for_index: 2,
            min_frequency_for_index: 0.01,
            ..Default::default()
        };

        let manager = AdaptiveIndexManager::new(graph, config);

        // Insert test data
        for i in 0..10 {
            let triple = Triple::new(
                NamedNode::new(format!("http://s{i}")).unwrap(),
                NamedNode::new("http://p").unwrap(),
                NamedNode::new(format!("http://o{i}")).unwrap(),
            );
            manager.insert(triple).unwrap();
        }

        // Query the same pattern multiple times
        let pred = Predicate::NamedNode(NamedNode::new("http://p").unwrap());
        for _ in 0..3 {
            let results = manager.query(None, Some(&pred), None).unwrap();
            assert_eq!(results.len(), 10);
        }

        // Force maintenance
        manager.run_maintenance();

        // Check if index was created
        let stats = manager.get_stats();
        assert!(stats.total_queries >= 3);
    }

    #[test]
    fn test_predicate_index() {
        let graph = Arc::new(RwLock::new(IndexedGraph::new()));

        // Insert test data
        for i in 0..5 {
            let triple = Triple::new(
                NamedNode::new(format!("http://s{i}")).unwrap(),
                NamedNode::new("http://p1").unwrap(),
                NamedNode::new(format!("http://o{i}")).unwrap(),
            );
            graph.write().insert(&triple);
        }

        for i in 0..3 {
            let triple = Triple::new(
                NamedNode::new(format!("http://s{i}")).unwrap(),
                NamedNode::new("http://p2").unwrap(),
                NamedNode::new(format!("http://o{i}")).unwrap(),
            );
            graph.write().insert(&triple);
        }

        let index = PredicateIndex::new(graph.clone());

        // Query by predicate
        let p1 = Predicate::NamedNode(NamedNode::new("http://p1").unwrap());
        let results = index.query(None, Some(&p1), None).unwrap();
        assert_eq!(results.len(), 5);

        let p2 = Predicate::NamedNode(NamedNode::new("http://p2").unwrap());
        let results = index.query(None, Some(&p2), None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_subject_predicate_index() {
        let graph = Arc::new(RwLock::new(IndexedGraph::new()));

        // Insert test data
        let s1 = Subject::NamedNode(NamedNode::new("http://s1").unwrap());
        let p1 = Predicate::NamedNode(NamedNode::new("http://p1").unwrap());

        for i in 0..5 {
            let triple = Triple::new(
                s1.clone(),
                p1.clone(),
                Object::NamedNode(NamedNode::new(format!("http://o{i}")).unwrap()),
            );
            graph.write().insert(&triple);
        }

        let index = SubjectPredicateIndex::new(graph.clone());

        // Query by subject and predicate
        let results = index.query(Some(&s1), Some(&p1), None).unwrap();
        assert_eq!(results.len(), 5);
    }
}
