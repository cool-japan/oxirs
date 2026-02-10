//! Cache Invalidation Engine
//!
//! Core system for tracking cache dependencies and automatically invalidating stale entries
//! when RDF updates occur. Provides multiple invalidation strategies with <1% overhead target.

use crate::algebra::{Term, TriplePattern};
use anyhow::{Context, Result};
use dashmap::DashMap;
use scirs2_core::error::CoreError;
use scirs2_core::metrics::{Counter, Histogram, HistogramStats, Timer};
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cache invalidation strategy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InvalidationStrategy {
    /// Invalidate immediately when updates occur (safest, highest overhead)
    Immediate,
    /// Buffer invalidations and flush periodically (balanced)
    Batched {
        /// Batch size before flush
        batch_size: usize,
        /// Maximum time before flush (milliseconds)
        max_delay_ms: u64,
    },
    /// Use Bloom filter for fast "may be affected" check
    BloomFilter {
        /// Expected number of elements
        expected_elements: usize,
        /// False positive rate (0.0 to 1.0)
        false_positive_rate: f64,
    },
    /// Invalidate only if re-execution cost > invalidation cost
    CostBased {
        /// Threshold ratio: invalidate if cost_ratio > threshold
        threshold: f64,
    },
}

impl Default for InvalidationStrategy {
    fn default() -> Self {
        Self::Batched {
            batch_size: 100,
            max_delay_ms: 50,
        }
    }
}

/// Triple pattern hash for efficient lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TriplePatternHash(u64);

impl TriplePatternHash {
    /// Create from triple pattern
    pub fn from_pattern(pattern: &TriplePattern) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Get hash value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Cache key identifier
pub type CacheKey = String;

/// Dependency graph tracking which cache entries depend on which triple patterns
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Map: TriplePattern → Set of cache entries that depend on it
    pattern_to_entries: Arc<DashMap<TriplePatternHash, HashSet<CacheKey>>>,
    /// Map: CacheEntry → Set of triple patterns it depends on
    entry_to_patterns: Arc<DashMap<CacheKey, HashSet<TriplePatternHash>>>,
    /// Statistics
    stats: Arc<DependencyGraphStats>,
}

#[derive(Debug, Default)]
struct DependencyGraphStats {
    /// Total patterns tracked
    pattern_count: AtomicUsize,
    /// Total cache entries tracked
    entry_count: AtomicUsize,
    /// Total edges in bipartite graph
    edge_count: AtomicUsize,
    /// Average dependencies per entry
    avg_deps_per_entry: AtomicU64,
}

impl DependencyGraph {
    /// Create new dependency graph
    pub fn new() -> Self {
        Self {
            pattern_to_entries: Arc::new(DashMap::new()),
            entry_to_patterns: Arc::new(DashMap::new()),
            stats: Arc::new(DependencyGraphStats::default()),
        }
    }

    /// Register dependencies for a cache entry
    pub fn register_dependencies(
        &self,
        cache_key: CacheKey,
        patterns: Vec<TriplePattern>,
    ) -> Result<()> {
        if patterns.is_empty() {
            return Ok(());
        }

        let pattern_hashes: HashSet<TriplePatternHash> = patterns
            .iter()
            .map(|p| TriplePatternHash::from_pattern(p))
            .collect();

        // Update entry → patterns mapping
        let is_new_entry = !self.entry_to_patterns.contains_key(&cache_key);
        self.entry_to_patterns
            .insert(cache_key.clone(), pattern_hashes.clone());

        // Update pattern → entries mapping
        for pattern_hash in &pattern_hashes {
            self.pattern_to_entries
                .entry(*pattern_hash)
                .or_insert_with(HashSet::new)
                .insert(cache_key.clone());
        }

        // Update statistics
        if is_new_entry {
            self.stats.entry_count.fetch_add(1, Ordering::Relaxed);
        }
        self.stats
            .edge_count
            .fetch_add(pattern_hashes.len(), Ordering::Relaxed);
        self.update_avg_deps();

        Ok(())
    }

    /// Remove a cache entry and its dependencies
    pub fn remove_entry(&self, cache_key: &CacheKey) -> Result<()> {
        // Get patterns this entry depends on
        if let Some((_, patterns)) = self.entry_to_patterns.remove(cache_key) {
            // Remove entry from all pattern mappings
            for pattern_hash in &patterns {
                if let Some(mut entries) = self.pattern_to_entries.get_mut(pattern_hash) {
                    entries.remove(cache_key);
                    if entries.is_empty() {
                        drop(entries);
                        self.pattern_to_entries.remove(pattern_hash);
                        self.stats.pattern_count.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }

            self.stats.entry_count.fetch_sub(1, Ordering::Relaxed);
            self.stats
                .edge_count
                .fetch_sub(patterns.len(), Ordering::Relaxed);
            self.update_avg_deps();
        }

        Ok(())
    }

    /// Find all cache entries affected by a triple pattern
    pub fn find_affected_entries(&self, pattern: &TriplePattern) -> HashSet<CacheKey> {
        let pattern_hash = TriplePatternHash::from_pattern(pattern);

        // Check exact match
        let mut affected = self
            .pattern_to_entries
            .get(&pattern_hash)
            .map(|entries| entries.clone())
            .unwrap_or_default();

        // Check for pattern subsumption (pattern with variables can match multiple)
        // For now, use simple matching; can be enhanced with more sophisticated logic
        for entry in self.pattern_to_entries.iter() {
            if self.pattern_matches(*entry.key(), pattern) {
                affected.extend(entry.value().iter().cloned());
            }
        }

        affected
    }

    /// Check if a pattern matches (considering variables)
    fn pattern_matches(&self, stored_hash: TriplePatternHash, query_pattern: &TriplePattern) -> bool {
        // This is a simplified version
        // In practice, you'd need to reconstruct the pattern or store metadata
        // For now, we use exact hash matching
        stored_hash == TriplePatternHash::from_pattern(query_pattern)
    }

    /// Get statistics
    pub fn statistics(&self) -> DependencyGraphStatistics {
        DependencyGraphStatistics {
            pattern_count: self.stats.pattern_count.load(Ordering::Relaxed),
            entry_count: self.stats.entry_count.load(Ordering::Relaxed),
            edge_count: self.stats.edge_count.load(Ordering::Relaxed),
            avg_deps_per_entry: f64::from_bits(self.stats.avg_deps_per_entry.load(Ordering::Relaxed)),
        }
    }

    /// Update average dependencies per entry
    fn update_avg_deps(&self) {
        let entries = self.stats.entry_count.load(Ordering::Relaxed);
        if entries > 0 {
            let edges = self.stats.edge_count.load(Ordering::Relaxed);
            let avg = edges as f64 / entries as f64;
            self.stats
                .avg_deps_per_entry
                .store(avg.to_bits(), Ordering::Relaxed);
        }
    }

    /// Clear all dependencies
    pub fn clear(&self) {
        self.pattern_to_entries.clear();
        self.entry_to_patterns.clear();
        self.stats.pattern_count.store(0, Ordering::Relaxed);
        self.stats.entry_count.store(0, Ordering::Relaxed);
        self.stats.edge_count.store(0, Ordering::Relaxed);
        self.stats.avg_deps_per_entry.store(0, Ordering::Relaxed);
    }

    /// Get memory usage estimate (bytes)
    pub fn memory_usage(&self) -> usize {
        let pattern_count = self.stats.pattern_count.load(Ordering::Relaxed);
        let entry_count = self.stats.entry_count.load(Ordering::Relaxed);
        let edge_count = self.stats.edge_count.load(Ordering::Relaxed);

        // Rough estimate:
        // - 24 bytes per HashMap entry (key + value pointer)
        // - 8 bytes per hash
        // - 40 bytes per String (average cache key)
        pattern_count * 24 + entry_count * 24 + edge_count * (8 + 40)
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DependencyGraphStatistics {
    pub pattern_count: usize,
    pub entry_count: usize,
    pub edge_count: usize,
    pub avg_deps_per_entry: f64,
}

/// Bloom filter for efficient pattern matching
struct BloomFilter {
    bits: Vec<AtomicU64>,
    num_hash_functions: usize,
    bit_count: usize,
}

impl BloomFilter {
    /// Create new Bloom filter
    fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal bit count and hash functions
        let m = Self::optimal_bit_count(expected_elements, false_positive_rate);
        let k = Self::optimal_hash_count(expected_elements, m);

        let num_u64s = (m + 63) / 64;
        let bits = (0..num_u64s).map(|_| AtomicU64::new(0)).collect();

        Self {
            bits,
            num_hash_functions: k,
            bit_count: m,
        }
    }

    /// Calculate optimal bit count
    fn optimal_bit_count(n: usize, p: f64) -> usize {
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        (-(n as f64 * p.ln()) / ln2_squared).ceil() as usize
    }

    /// Calculate optimal hash function count
    fn optimal_hash_count(n: usize, m: usize) -> usize {
        ((m as f64 / n as f64) * std::f64::consts::LN_2).ceil() as usize
    }

    /// Add pattern to filter
    fn add(&self, pattern_hash: TriplePatternHash) {
        for i in 0..self.num_hash_functions {
            let bit_index = self.hash_i(pattern_hash, i) % self.bit_count;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            self.bits[word_index].fetch_or(1u64 << bit_offset, Ordering::Relaxed);
        }
    }

    /// Check if pattern might be in set
    fn might_contain(&self, pattern_hash: TriplePatternHash) -> bool {
        for i in 0..self.num_hash_functions {
            let bit_index = self.hash_i(pattern_hash, i) % self.bit_count;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            let word = self.bits[word_index].load(Ordering::Relaxed);
            if (word & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Hash function with index
    fn hash_i(&self, pattern_hash: TriplePatternHash, i: usize) -> usize {
        // Simple double hashing
        let h1 = pattern_hash.value() as usize;
        let h2 = (pattern_hash.value().wrapping_mul(2654435761)) as usize;
        h1.wrapping_add(i.wrapping_mul(h2))
    }

    /// Clear filter
    fn clear(&self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
    }
}

/// Batch of pending invalidations
#[derive(Debug)]
struct InvalidationBatch {
    entries: Vec<CacheKey>,
    timestamp: Instant,
}

/// Core invalidation engine
pub struct InvalidationEngine {
    /// Dependency graph
    dependency_graph: DependencyGraph,
    /// Invalidation strategy
    strategy: InvalidationStrategy,
    /// Bloom filter (if using BloomFilter strategy)
    bloom_filter: Option<Arc<BloomFilter>>,
    /// Pending invalidations (for batched strategy)
    pending_invalidations: Arc<RwLock<VecDeque<InvalidationBatch>>>,
    /// Metrics
    metrics: InvalidationMetrics,
    /// Configuration
    config: InvalidationConfig,
}

#[derive(Clone)]
struct InvalidationMetrics {
    /// Total invalidations triggered
    total_invalidations: Arc<Counter>,
    /// Time spent in invalidation
    invalidation_time: Arc<Timer>,
    /// Invalidation overhead ratio
    overhead_ratio: Arc<Histogram>,
    /// Cache entries invalidated per update
    entries_per_update: Arc<Histogram>,
}

impl InvalidationMetrics {
    fn new() -> Self {
        Self {
            total_invalidations: Arc::new(Counter::new("invalidation_total".to_string())),
            invalidation_time: Arc::new(Timer::new("invalidation_time".to_string())),
            overhead_ratio: Arc::new(Histogram::new("invalidation_overhead".to_string())),
            entries_per_update: Arc::new(Histogram::new("invalidation_entries_per_update".to_string())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationConfig {
    /// Enable metrics tracking
    pub enable_metrics: bool,
    /// Maximum pending batch size
    pub max_pending_batches: usize,
    /// Enable aggressive pattern matching
    pub aggressive_matching: bool,
}

impl Default for InvalidationConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            max_pending_batches: 100,
            aggressive_matching: false,
        }
    }
}

impl InvalidationEngine {
    /// Create new invalidation engine
    pub fn new(strategy: InvalidationStrategy) -> Self {
        Self::with_config(strategy, InvalidationConfig::default())
    }

    /// Create with configuration
    pub fn with_config(strategy: InvalidationStrategy, config: InvalidationConfig) -> Self {
        let bloom_filter = match strategy {
            InvalidationStrategy::BloomFilter {
                expected_elements,
                false_positive_rate,
            } => Some(Arc::new(BloomFilter::new(
                expected_elements,
                false_positive_rate,
            ))),
            _ => None,
        };

        Self {
            dependency_graph: DependencyGraph::new(),
            strategy,
            bloom_filter,
            pending_invalidations: Arc::new(RwLock::new(VecDeque::new())),
            metrics: InvalidationMetrics::new(),
            config,
        }
    }

    /// Register dependencies for a cache entry
    pub fn register_dependencies(
        &self,
        cache_key: CacheKey,
        patterns: Vec<TriplePattern>,
    ) -> Result<()> {
        // Add to dependency graph
        self.dependency_graph
            .register_dependencies(cache_key, patterns.clone())?;

        // Add to bloom filter if using that strategy
        if let Some(bloom) = &self.bloom_filter {
            for pattern in &patterns {
                bloom.add(TriplePatternHash::from_pattern(pattern));
            }
        }

        Ok(())
    }

    /// Remove cache entry and its dependencies
    pub fn remove_entry(&self, cache_key: &CacheKey) -> Result<()> {
        self.dependency_graph.remove_entry(cache_key)
    }

    /// Find entries that should be invalidated due to a triple update
    pub fn find_affected_entries(&self, triple: &TriplePattern) -> Result<HashSet<CacheKey>> {
        let start_time = Instant::now();

        let affected = match self.strategy {
            InvalidationStrategy::BloomFilter { .. } => {
                // Use Bloom filter for fast check
                if let Some(bloom) = &self.bloom_filter {
                    let pattern_hash = TriplePatternHash::from_pattern(triple);
                    if bloom.might_contain(pattern_hash) {
                        self.dependency_graph.find_affected_entries(triple)
                    } else {
                        HashSet::new()
                    }
                } else {
                    self.dependency_graph.find_affected_entries(triple)
                }
            }
            _ => self.dependency_graph.find_affected_entries(triple),
        };

        // Track metrics
        if self.config.enable_metrics {
            let elapsed = start_time.elapsed();
            self.metrics
                .invalidation_time
                .observe(elapsed);
            self.metrics
                .entries_per_update
                .observe(affected.len() as f64);
        }

        Ok(affected)
    }

    /// Invalidate cache entries (strategy-dependent)
    pub fn invalidate<F>(&self, triple: &TriplePattern, mut invalidate_fn: F) -> Result<()>
    where
        F: FnMut(&CacheKey) -> Result<()>,
    {
        let affected = self.find_affected_entries(triple)?;
        let affected_count = affected.len();

        match self.strategy {
            InvalidationStrategy::Immediate => {
                // Invalidate immediately
                for cache_key in &affected {
                    invalidate_fn(cache_key)?;
                    self.dependency_graph.remove_entry(cache_key)?;
                }
            }
            InvalidationStrategy::Batched {
                batch_size,
                max_delay_ms,
            } => {
                // Add to batch
                self.add_to_batch(affected, batch_size, max_delay_ms, &mut invalidate_fn)?;
            }
            InvalidationStrategy::BloomFilter { .. } => {
                // Same as immediate for actual invalidation
                for cache_key in &affected {
                    invalidate_fn(cache_key)?;
                    self.dependency_graph.remove_entry(cache_key)?;
                }
            }
            InvalidationStrategy::CostBased { threshold } => {
                // Only invalidate if beneficial
                for cache_key in &affected {
                    if self.should_invalidate_cost_based(cache_key, threshold)? {
                        invalidate_fn(cache_key)?;
                        self.dependency_graph.remove_entry(cache_key)?;
                    }
                }
            }
        }

        // Update metrics
        if self.config.enable_metrics {
            self.metrics
                .total_invalidations
                .add(affected_count as u64);
        }

        Ok(())
    }

    /// Add entries to batch for later invalidation
    fn add_to_batch<F>(
        &self,
        entries: HashSet<CacheKey>,
        batch_size: usize,
        max_delay_ms: u64,
        invalidate_fn: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&CacheKey) -> Result<()>,
    {
        let mut pending = self
            .pending_invalidations
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        // Add new batch
        pending.push_back(InvalidationBatch {
            entries: entries.into_iter().collect(),
            timestamp: Instant::now(),
        });

        // Flush if batch size exceeded or max delay reached
        let should_flush = pending.len() >= batch_size
            || pending
                .front()
                .map(|b| b.timestamp.elapsed().as_millis() as u64 >= max_delay_ms)
                .unwrap_or(false);

        if should_flush {
            self.flush_batches(&mut pending, invalidate_fn)?;
        }

        Ok(())
    }

    /// Flush pending invalidation batches
    fn flush_batches<F>(
        &self,
        pending: &mut VecDeque<InvalidationBatch>,
        invalidate_fn: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&CacheKey) -> Result<()>,
    {
        while let Some(batch) = pending.pop_front() {
            for cache_key in &batch.entries {
                invalidate_fn(cache_key)?;
                self.dependency_graph.remove_entry(cache_key)?;
            }
        }
        Ok(())
    }

    /// Force flush all pending invalidations
    pub fn flush_pending<F>(&self, mut invalidate_fn: F) -> Result<()>
    where
        F: FnMut(&CacheKey) -> Result<()>,
    {
        let mut pending = self
            .pending_invalidations
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        self.flush_batches(&mut pending, &mut invalidate_fn)
    }

    /// Check if entry should be invalidated (cost-based strategy)
    fn should_invalidate_cost_based(&self, _cache_key: &CacheKey, _threshold: f64) -> Result<bool> {
        // Simplified: In practice, you'd compare:
        // - Cost of re-executing query
        // - Cost of invalidating and warming cache
        // For now, always invalidate (conservative)
        Ok(true)
    }

    /// Get invalidation statistics
    pub fn statistics(&self) -> InvalidationStatistics {
        let graph_stats = self.dependency_graph.statistics();
        let time_stats = self.metrics.invalidation_time.get_stats();
        let overhead_stats = self.metrics.overhead_ratio.get_stats();
        let entries_stats = self.metrics.entries_per_update.get_stats();

        InvalidationStatistics {
            strategy: self.strategy,
            total_invalidations: self.metrics.total_invalidations.get(),
            avg_invalidation_time_us: time_stats.mean,
            overhead_ratio: overhead_stats.mean,
            avg_entries_per_update: entries_stats.mean,
            dependency_graph: graph_stats,
            memory_usage_bytes: self.dependency_graph.memory_usage(),
        }
    }

    /// Clear all state
    pub fn clear(&self) -> Result<()> {
        self.dependency_graph.clear();
        if let Some(bloom) = &self.bloom_filter {
            bloom.clear();
        }
        let mut pending = self
            .pending_invalidations
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        pending.clear();
        Ok(())
    }
}

/// Invalidation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationStatistics {
    pub strategy: InvalidationStrategy,
    pub total_invalidations: u64,
    pub avg_invalidation_time_us: f64,
    pub overhead_ratio: f64,
    pub avg_entries_per_update: f64,
    pub dependency_graph: DependencyGraphStatistics,
    pub memory_usage_bytes: usize,
}

/// RDF update listener trait
pub trait RdfUpdateListener {
    /// Called when an RDF triple is inserted
    fn on_insert(&mut self, triple: &TriplePattern) -> Result<()>;

    /// Called when an RDF triple is deleted
    fn on_delete(&mut self, triple: &TriplePattern) -> Result<()>;

    /// Called when multiple triples are inserted
    fn on_batch_insert(&mut self, triples: &[TriplePattern]) -> Result<()> {
        for triple in triples {
            self.on_insert(triple)?;
        }
        Ok(())
    }

    /// Called when multiple triples are deleted
    fn on_batch_delete(&mut self, triples: &[TriplePattern]) -> Result<()> {
        for triple in triples {
            self.on_delete(triple)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Variable;

    fn create_test_pattern(s: &str, p: &str, o: &str) -> TriplePattern {
        TriplePattern {
            subject: Term::Variable(Variable::new(s).expect("valid variable")),
            predicate: Term::Variable(Variable::new(p).expect("valid variable")),
            object: Term::Variable(Variable::new(o).expect("valid variable")),
        }
    }

    #[test]
    fn test_dependency_graph_basic() {
        let graph = DependencyGraph::new();

        let pattern1 = create_test_pattern("s", "p", "o");
        let pattern2 = create_test_pattern("x", "y", "z");

        graph
            .register_dependencies("key1".to_string(), vec![pattern1.clone(), pattern2.clone()])
            .unwrap();

        let stats = graph.statistics();
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.edge_count, 2);
    }

    #[test]
    fn test_invalidation_engine_immediate() {
        let engine = InvalidationEngine::new(InvalidationStrategy::Immediate);

        let pattern = create_test_pattern("s", "p", "o");
        engine
            .register_dependencies("key1".to_string(), vec![pattern.clone()])
            .unwrap();

        let affected = engine.find_affected_entries(&pattern).unwrap();
        assert_eq!(affected.len(), 1);
        assert!(affected.contains("key1"));
    }

    #[test]
    fn test_invalidation_engine_batched() {
        let engine = InvalidationEngine::new(InvalidationStrategy::Batched {
            batch_size: 10,
            max_delay_ms: 100,
        });

        let pattern = create_test_pattern("s", "p", "o");
        engine
            .register_dependencies("key1".to_string(), vec![pattern.clone()])
            .unwrap();

        let mut invalidated = Vec::new();
        engine
            .invalidate(&pattern, |key| {
                invalidated.push(key.clone());
                Ok(())
            })
            .unwrap();

        // Force flush
        engine
            .flush_pending(|key| {
                invalidated.push(key.clone());
                Ok(())
            })
            .unwrap();

        // Should have invalidated key1 (possibly twice if batched)
        assert!(!invalidated.is_empty());
    }

    #[test]
    fn test_bloom_filter() {
        let filter = BloomFilter::new(1000, 0.01);

        let pattern = create_test_pattern("s", "p", "o");
        let hash = TriplePatternHash::from_pattern(&pattern);

        // Should not contain before adding
        assert!(!filter.might_contain(hash));

        // Add and check
        filter.add(hash);
        assert!(filter.might_contain(hash));
    }

    #[test]
    fn test_remove_entry() {
        let graph = DependencyGraph::new();

        let pattern = create_test_pattern("s", "p", "o");
        graph
            .register_dependencies("key1".to_string(), vec![pattern.clone()])
            .unwrap();

        assert_eq!(graph.statistics().entry_count, 1);

        graph.remove_entry(&"key1".to_string()).unwrap();

        assert_eq!(graph.statistics().entry_count, 0);
    }

    #[test]
    fn test_multiple_dependencies() {
        let engine = InvalidationEngine::new(InvalidationStrategy::Immediate);

        let pattern1 = create_test_pattern("s", "p", "o");
        let pattern2 = create_test_pattern("x", "y", "z");

        engine
            .register_dependencies("key1".to_string(), vec![pattern1.clone()])
            .unwrap();
        engine
            .register_dependencies("key2".to_string(), vec![pattern1.clone(), pattern2.clone()])
            .unwrap();

        // Update pattern1 should affect both entries
        let affected = engine.find_affected_entries(&pattern1).unwrap();
        assert_eq!(affected.len(), 2);

        // Update pattern2 should only affect key2
        let affected2 = engine.find_affected_entries(&pattern2).unwrap();
        assert_eq!(affected2.len(), 1);
        assert!(affected2.contains("key2"));
    }
}
