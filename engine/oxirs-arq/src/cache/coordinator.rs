//! Cache Coordinator
//!
//! Unified coordinator for managing 3-level cache hierarchy and ensuring
//! cache consistency through coordinated invalidations.

use super::invalidation_engine::{
    InvalidationEngine, InvalidationStatistics, InvalidationStrategy, RdfUpdateListener,
};
use crate::algebra::TriplePattern;
use crate::query_plan_cache::{QueryPlanCache, CacheStats as PlanCacheStats};
use crate::query_result_cache::{QueryResultCache, CacheStatistics as ResultCacheStatistics};
use anyhow::{Context, Result};
use scirs2_core::error::CoreError;
use scirs2_core::metrics::{Counter, HistogramStats, Timer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Cache level identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheLevel {
    /// Query result cache (L1)
    Result,
    /// Query plan cache (L2)
    Plan,
    /// Optimizer/statistics cache (L3)
    Optimizer,
}

/// Cache invalidation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationConfig {
    /// Invalidation strategy
    pub strategy: InvalidationStrategy,
    /// Enable cross-level invalidation propagation
    pub propagate_invalidations: bool,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Maximum batch size for batched strategy
    pub max_batch_size: usize,
    /// Flush interval for batched strategy (milliseconds)
    pub flush_interval_ms: u64,
}

impl Default for InvalidationConfig {
    fn default() -> Self {
        Self {
            strategy: InvalidationStrategy::Batched {
                batch_size: 100,
                max_delay_ms: 50,
            },
            propagate_invalidations: true,
            enable_metrics: true,
            max_batch_size: 1000,
            flush_interval_ms: 50,
        }
    }
}

/// Unified cache coordinator
pub struct CacheCoordinator {
    /// Result cache
    result_cache: Option<Arc<QueryResultCache>>,
    /// Plan cache
    plan_cache: Option<Arc<QueryPlanCache>>,
    /// Optimizer cache (placeholder for now)
    optimizer_cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// Invalidation engine
    invalidation_engine: Arc<InvalidationEngine>,
    /// Configuration
    config: InvalidationConfig,
    /// Coordinator metrics
    metrics: CoordinatorMetrics,
    /// Cache entry metadata
    entry_metadata: Arc<RwLock<HashMap<String, CacheEntryMetadata>>>,
}

#[derive(Debug, Clone)]
struct CacheEntryMetadata {
    level: CacheLevel,
    cache_key: String,
    created_at: Instant,
    last_accessed: Instant,
    access_count: usize,
    size_bytes: usize,
}

#[derive(Clone)]
struct CoordinatorMetrics {
    /// Total invalidations across all levels
    total_invalidations: Arc<Counter>,
    /// Invalidations per level
    result_invalidations: Arc<Counter>,
    plan_invalidations: Arc<Counter>,
    optimizer_invalidations: Arc<Counter>,
    /// Coordination overhead
    coordination_overhead: Arc<Timer>,
    /// Cache coherence checks
    coherence_checks: Arc<Counter>,
    /// Coherence violations detected
    coherence_violations: Arc<Counter>,
}

impl CoordinatorMetrics {
    fn new() -> Self {
        Self {
            total_invalidations: Arc::new(Counter::new("cache_total_invalidations".to_string())),
            result_invalidations: Arc::new(Counter::new("cache_result_invalidations".to_string())),
            plan_invalidations: Arc::new(Counter::new("cache_plan_invalidations".to_string())),
            optimizer_invalidations: Arc::new(Counter::new("cache_optimizer_invalidations".to_string())),
            coordination_overhead: Arc::new(Timer::new("cache_coordination_overhead".to_string())),
            coherence_checks: Arc::new(Counter::new("cache_coherence_checks".to_string())),
            coherence_violations: Arc::new(Counter::new("cache_coherence_violations".to_string())),
        }
    }
}

impl CacheCoordinator {
    /// Create new cache coordinator
    pub fn new(config: InvalidationConfig) -> Self {
        let invalidation_engine = Arc::new(InvalidationEngine::with_config(
            config.strategy,
            super::invalidation_engine::InvalidationConfig {
                enable_metrics: config.enable_metrics,
                max_pending_batches: config.max_batch_size,
                aggressive_matching: false,
            },
        ));

        Self {
            result_cache: None,
            plan_cache: None,
            optimizer_cache: Arc::new(RwLock::new(HashMap::new())),
            invalidation_engine,
            config,
            metrics: CoordinatorMetrics::new(),
            entry_metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Attach result cache
    pub fn attach_result_cache(&mut self, cache: Arc<QueryResultCache>) {
        self.result_cache = Some(cache);
    }

    /// Attach plan cache
    pub fn attach_plan_cache(&mut self, cache: Arc<QueryPlanCache>) {
        self.plan_cache = Some(cache);
    }

    /// Register a cache entry with its dependencies
    pub fn register_cache_entry(
        &self,
        level: CacheLevel,
        cache_key: String,
        patterns: Vec<TriplePattern>,
        size_bytes: usize,
    ) -> Result<()> {
        // Register with invalidation engine
        self.invalidation_engine
            .register_dependencies(cache_key.clone(), patterns)?;

        // Store metadata
        let mut metadata = self
            .entry_metadata
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        metadata.insert(
            cache_key.clone(),
            CacheEntryMetadata {
                level,
                cache_key,
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                access_count: 0,
                size_bytes,
            },
        );

        Ok(())
    }

    /// Invalidate caches based on RDF update
    pub fn invalidate_on_update(&self, triple: &TriplePattern) -> Result<()> {
        let start_time = Instant::now();

        // Find affected entries
        let affected = self
            .invalidation_engine
            .find_affected_entries(triple)
            .context("Failed to find affected entries")?;

        // Group by cache level
        let mut result_keys = Vec::new();
        let mut plan_keys = Vec::new();
        let mut optimizer_keys = Vec::new();

        {
            let metadata = self
                .entry_metadata
                .read()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

            for cache_key in &affected {
                if let Some(entry) = metadata.get(cache_key) {
                    match entry.level {
                        CacheLevel::Result => result_keys.push(cache_key.clone()),
                        CacheLevel::Plan => plan_keys.push(cache_key.clone()),
                        CacheLevel::Optimizer => optimizer_keys.push(cache_key.clone()),
                    }
                }
            }
        }

        // Invalidate at each level
        self.invalidate_result_entries(&result_keys)?;
        self.invalidate_plan_entries(&plan_keys)?;
        self.invalidate_optimizer_entries(&optimizer_keys)?;

        // Propagate invalidations if configured
        if self.config.propagate_invalidations {
            self.propagate_invalidations(&result_keys, &plan_keys, &optimizer_keys)?;
        }

        // Update metrics
        if self.config.enable_metrics {
            let elapsed = start_time.elapsed();
            self.metrics
                .coordination_overhead
                .observe(elapsed);
            self.metrics
                .total_invalidations
                .add(affected.len() as u64);
        }

        Ok(())
    }

    /// Invalidate result cache entries
    fn invalidate_result_entries(&self, keys: &[String]) -> Result<()> {
        if let Some(cache) = &self.result_cache {
            for key in keys {
                cache
                    .invalidate(key)
                    .context("Failed to invalidate result cache entry")?;
                self.remove_metadata(key)?;
            }
            self.metrics
                .result_invalidations
                .add(keys.len() as u64);
        }
        Ok(())
    }

    /// Invalidate plan cache entries
    fn invalidate_plan_entries(&self, keys: &[String]) -> Result<()> {
        if let Some(cache) = &self.plan_cache {
            for _key in keys {
                // QueryPlanCache uses pattern-based invalidation
                // We'd need to extract pattern from key or store mapping
                // For now, we can clear the entire cache as a conservative approach
                // In production, you'd maintain a mapping of keys to patterns
                cache.clear();
            }
            self.metrics.plan_invalidations.add(keys.len() as u64);
        }
        Ok(())
    }

    /// Invalidate optimizer cache entries
    fn invalidate_optimizer_entries(&self, keys: &[String]) -> Result<()> {
        let mut cache = self
            .optimizer_cache
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        for key in keys {
            cache.remove(key);
            self.remove_metadata(key)?;
        }

        self.metrics
            .optimizer_invalidations
            .add(keys.len() as u64);
        Ok(())
    }

    /// Propagate invalidations across cache levels
    fn propagate_invalidations(
        &self,
        _result_keys: &[String],
        plan_keys: &[String],
        optimizer_keys: &[String],
    ) -> Result<()> {
        // If plan cache entries are invalidated, also invalidate dependent result cache entries
        if !plan_keys.is_empty() && self.result_cache.is_some() {
            // Find result cache entries that depend on these plans
            // This would require additional tracking; simplified for now
        }

        // If optimizer cache entries are invalidated, invalidate dependent plan cache entries
        if !optimizer_keys.is_empty() && self.plan_cache.is_some() {
            // Optimizer changes affect plan cache
            // For now, clear plan cache conservatively
            if let Some(cache) = &self.plan_cache {
                cache.clear();
            }
        }

        Ok(())
    }

    /// Remove entry metadata
    fn remove_metadata(&self, cache_key: &str) -> Result<()> {
        let mut metadata = self
            .entry_metadata
            .write()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        metadata.remove(cache_key);
        Ok(())
    }

    /// Check cache coherence (for testing/verification)
    pub fn check_coherence(&self) -> Result<CoherenceReport> {
        self.metrics.coherence_checks.inc();

        let violations = Vec::new();

        // Check if any result cache entries reference invalidated plans
        // This would require additional tracking

        // Check if any plan cache entries reference invalidated optimizer state
        // This would require additional tracking

        let has_violations = !violations.is_empty();
        if has_violations {
            self.metrics
                .coherence_violations
                .add(violations.len() as u64);
        }

        Ok(CoherenceReport {
            is_coherent: !has_violations,
            violations,
            check_time: Instant::now(),
        })
    }

    /// Get coordinator statistics
    pub fn statistics(&self) -> CoordinatorStatistics {
        let invalidation_stats = self.invalidation_engine.statistics();

        let result_cache_stats = self
            .result_cache
            .as_ref()
            .map(|c| c.statistics())
            .unwrap_or_default();

        let plan_cache_stats = self
            .plan_cache
            .as_ref()
            .map(|c| c.statistics())
            .unwrap_or_else(|| PlanCacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                invalidations: 0,
                size: 0,
                capacity: 0,
                hit_rate: 0.0,
            });

        let metadata = self.entry_metadata.read().ok();
        let entry_count_by_level = metadata.as_ref().map(|m| {
            let mut counts = HashMap::new();
            for entry in m.values() {
                *counts.entry(entry.level).or_insert(0) += 1;
            }
            counts
        });

        let overhead_stats = self.metrics.coordination_overhead.get_stats();

        CoordinatorStatistics {
            total_invalidations: self.metrics.total_invalidations.get(),
            result_invalidations: self.metrics.result_invalidations.get(),
            plan_invalidations: self.metrics.plan_invalidations.get(),
            optimizer_invalidations: self.metrics.optimizer_invalidations.get(),
            avg_coordination_overhead_us: overhead_stats.mean,
            coherence_checks: self.metrics.coherence_checks.get(),
            coherence_violations: self.metrics.coherence_violations.get(),
            invalidation_engine_stats: invalidation_stats,
            result_cache_stats,
            plan_cache_stats,
            entry_count_by_level: entry_count_by_level.unwrap_or_default(),
        }
    }

    /// Clear all caches
    pub fn clear_all(&self) -> Result<()> {
        if let Some(cache) = &self.result_cache {
            cache.invalidate_all()?;
        }

        if let Some(cache) = &self.plan_cache {
            cache.clear();
        }

        {
            let mut optimizer_cache = self
                .optimizer_cache
                .write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            optimizer_cache.clear();
        }

        {
            let mut metadata = self
                .entry_metadata
                .write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            metadata.clear();
        }

        self.invalidation_engine.clear()?;

        Ok(())
    }

    /// Force flush any pending invalidations
    pub fn flush_pending(&self) -> Result<()> {
        self.invalidation_engine.flush_pending(|key| {
            // Invalidate across all caches
            if let Some(cache) = &self.result_cache {
                let _ = cache.invalidate(key);
            }
            // Plan cache would need key-to-pattern mapping
            // Optimizer cache
            let mut optimizer_cache = self
                .optimizer_cache
                .write()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            optimizer_cache.remove(key);
            Ok(())
        })
    }

    /// Get invalidation engine
    pub fn invalidation_engine(&self) -> Arc<InvalidationEngine> {
        Arc::clone(&self.invalidation_engine)
    }
}

/// Coherence check report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceReport {
    pub is_coherent: bool,
    pub violations: Vec<CoherenceViolation>,
    #[serde(skip, default = "Instant::now")]
    pub check_time: Instant,
}

/// Coherence violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceViolation {
    pub level: CacheLevel,
    pub cache_key: String,
    pub violation_type: ViolationType,
    pub details: String,
}

/// Types of coherence violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Result cache references invalidated plan
    StaleResultReference,
    /// Plan cache references invalidated optimizer state
    StalePlanReference,
    /// Cross-level inconsistency
    CrossLevelInconsistency,
}

/// Coordinator statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorStatistics {
    pub total_invalidations: u64,
    pub result_invalidations: u64,
    pub plan_invalidations: u64,
    pub optimizer_invalidations: u64,
    pub avg_coordination_overhead_us: f64,
    pub coherence_checks: u64,
    pub coherence_violations: u64,
    pub invalidation_engine_stats: InvalidationStatistics,
    pub result_cache_stats: ResultCacheStatistics,
    pub plan_cache_stats: PlanCacheStats,
    pub entry_count_by_level: HashMap<CacheLevel, usize>,
}

/// Implement RdfUpdateListener for CacheCoordinator
impl RdfUpdateListener for CacheCoordinator {
    fn on_insert(&mut self, triple: &TriplePattern) -> Result<()> {
        self.invalidate_on_update(triple)
    }

    fn on_delete(&mut self, triple: &TriplePattern) -> Result<()> {
        self.invalidate_on_update(triple)
    }

    fn on_batch_insert(&mut self, triples: &[TriplePattern]) -> Result<()> {
        for triple in triples {
            self.invalidate_on_update(triple)?;
        }
        // Flush any pending batched invalidations
        self.flush_pending()?;
        Ok(())
    }

    fn on_batch_delete(&mut self, triples: &[TriplePattern]) -> Result<()> {
        for triple in triples {
            self.invalidate_on_update(triple)?;
        }
        // Flush any pending batched invalidations
        self.flush_pending()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Term, Variable};
    use crate::query_result_cache::CacheConfig;

    fn create_test_pattern(s: &str, p: &str, o: &str) -> TriplePattern {
        TriplePattern {
            subject: Term::Variable(Variable::new(s).expect("valid variable")),
            predicate: Term::Variable(Variable::new(p).expect("valid variable")),
            object: Term::Variable(Variable::new(o).expect("valid variable")),
        }
    }

    #[test]
    fn test_coordinator_creation() {
        let config = InvalidationConfig::default();
        let coordinator = CacheCoordinator::new(config);

        let stats = coordinator.statistics();
        assert_eq!(stats.total_invalidations, 0);
    }

    #[test]
    fn test_register_and_invalidate() {
        let config = InvalidationConfig::default();
        let coordinator = CacheCoordinator::new(config);

        let pattern = create_test_pattern("s", "p", "o");
        let cache_key = "test_key".to_string();

        coordinator
            .register_cache_entry(CacheLevel::Result, cache_key.clone(), vec![pattern.clone()], 100)
            .unwrap();

        coordinator.invalidate_on_update(&pattern).unwrap();

        let stats = coordinator.statistics();
        assert_eq!(stats.total_invalidations, 1);
    }

    #[test]
    fn test_attach_caches() {
        let mut coordinator = CacheCoordinator::new(InvalidationConfig::default());

        let result_cache = Arc::new(QueryResultCache::new(CacheConfig::default()));
        let plan_cache = Arc::new(QueryPlanCache::new());

        coordinator.attach_result_cache(result_cache);
        coordinator.attach_plan_cache(plan_cache);

        // Coordinator should now have caches attached
        assert!(coordinator.result_cache.is_some());
        assert!(coordinator.plan_cache.is_some());
    }

    #[test]
    fn test_clear_all() {
        let config = InvalidationConfig::default();
        let coordinator = CacheCoordinator::new(config);

        let pattern = create_test_pattern("s", "p", "o");
        coordinator
            .register_cache_entry(CacheLevel::Result, "key1".to_string(), vec![pattern], 100)
            .unwrap();

        coordinator.clear_all().unwrap();

        let stats = coordinator.statistics();
        assert_eq!(stats.entry_count_by_level.len(), 0);
    }

    #[test]
    fn test_multi_level_invalidation() {
        let config = InvalidationConfig {
            propagate_invalidations: true,
            ..Default::default()
        };
        let coordinator = CacheCoordinator::new(config);

        let pattern = create_test_pattern("s", "p", "o");

        // Register entries at different levels
        coordinator
            .register_cache_entry(CacheLevel::Result, "result_key".to_string(), vec![pattern.clone()], 100)
            .unwrap();
        coordinator
            .register_cache_entry(CacheLevel::Plan, "plan_key".to_string(), vec![pattern.clone()], 50)
            .unwrap();

        // Invalidate
        coordinator.invalidate_on_update(&pattern).unwrap();

        let stats = coordinator.statistics();
        assert!(stats.result_invalidations > 0 || stats.plan_invalidations > 0);
    }
}
