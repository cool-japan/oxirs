//! Optimization Cache for Query Plans
//!
//! This module provides caching of optimization decisions and query plans
//! to improve performance of repeated optimization operations.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::algebra::Algebra;

/// Cache for optimization decisions and plans
#[derive(Clone)]
pub struct OptimizationCache {
    plan_cache: HashMap<u64, CachedPlan>,
    decision_cache: HashMap<u64, CachedDecision>,
    config: CacheConfig,
    statistics: CacheStatistics,
}

/// Configuration for optimization cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_plan_entries: usize,
    pub max_decision_entries: usize,
    pub ttl_seconds: u64,
    pub enable_statistics: bool,
}

/// Cached query plan
#[derive(Debug, Clone)]
pub struct CachedPlan {
    pub optimized_plan: Algebra,
    pub estimated_cost: f64,
    pub creation_time: Instant,
    pub access_count: usize,
    pub last_access: Instant,
}

/// Cached optimization decision
#[derive(Debug, Clone)]
pub struct CachedDecision {
    pub decision_type: DecisionType,
    pub confidence: f64,
    pub estimated_benefit: f64,
    pub context_hash: u64,
    pub creation_time: Instant,
}

/// Types of optimization decisions
#[derive(Debug, Clone)]
pub enum DecisionType {
    IndexSelection(String),
    JoinAlgorithm(String),
    StreamingStrategy(String),
    ParallelismDegree(usize),
    MaterializeView(String),
}

/// Cache performance statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStatistics {
    pub plan_hits: usize,
    pub plan_misses: usize,
    pub decision_hits: usize,
    pub decision_misses: usize,
    pub evictions: usize,
    pub total_lookups: usize,
}

impl OptimizationCache {
    /// Create a new optimization cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            plan_cache: HashMap::new(),
            decision_cache: HashMap::new(),
            config,
            statistics: CacheStatistics::default(),
        }
    }

    /// Cache an optimized plan
    pub fn cache_plan(&mut self, query_hash: u64, plan: Algebra, cost: f64) {
        if self.plan_cache.len() >= self.config.max_plan_entries {
            self.evict_least_recently_used_plan();
        }

        let cached_plan = CachedPlan {
            optimized_plan: plan,
            estimated_cost: cost,
            creation_time: Instant::now(),
            access_count: 0,
            last_access: Instant::now(),
        };

        self.plan_cache.insert(query_hash, cached_plan);
    }

    /// Get cached plan if available and not expired
    pub fn get_cached_plan(&mut self, query_hash: u64) -> Option<Algebra> {
        self.statistics.total_lookups += 1;

        // Check if entry exists and is not expired
        let should_remove = if let Some(cached) = self.plan_cache.get(&query_hash) {
            self.is_expired(cached.creation_time)
        } else {
            false
        };

        if should_remove {
            self.plan_cache.remove(&query_hash);
            self.statistics.plan_misses += 1;
            return None;
        }

        if let Some(cached) = self.plan_cache.get_mut(&query_hash) {
            cached.access_count += 1;
            cached.last_access = Instant::now();
            self.statistics.plan_hits += 1;
            Some(cached.optimized_plan.clone())
        } else {
            self.statistics.plan_misses += 1;
            None
        }
    }

    /// Cache an optimization decision
    pub fn cache_decision(&mut self, context_hash: u64, decision: CachedDecision) {
        if self.decision_cache.len() >= self.config.max_decision_entries {
            self.evict_oldest_decision();
        }

        self.decision_cache.insert(context_hash, decision);
    }

    /// Get cached decision if available
    pub fn get_cached_decision(&mut self, context_hash: u64) -> Option<CachedDecision> {
        self.statistics.total_lookups += 1;

        // Check if entry exists and is not expired
        let should_remove = if let Some(decision) = self.decision_cache.get(&context_hash) {
            self.is_expired(decision.creation_time)
        } else {
            false
        };

        if should_remove {
            self.decision_cache.remove(&context_hash);
            self.statistics.decision_misses += 1;
            return None;
        }

        if let Some(decision) = self.decision_cache.get(&context_hash) {
            self.statistics.decision_hits += 1;
            Some(decision.clone())
        } else {
            self.statistics.decision_misses += 1;
            None
        }
    }

    /// Get cache statistics
    pub fn statistics(&self) -> &CacheStatistics {
        &self.statistics
    }

    /// Clear all cached entries
    pub fn clear(&mut self) {
        self.plan_cache.clear();
        self.decision_cache.clear();
        self.statistics = CacheStatistics::default();
    }

    /// Get cache hit ratio for plans
    pub fn plan_hit_ratio(&self) -> f64 {
        let total = self.statistics.plan_hits + self.statistics.plan_misses;
        if total == 0 {
            0.0
        } else {
            self.statistics.plan_hits as f64 / total as f64
        }
    }

    /// Get cache hit ratio for decisions
    pub fn decision_hit_ratio(&self) -> f64 {
        let total = self.statistics.decision_hits + self.statistics.decision_misses;
        if total == 0 {
            0.0
        } else {
            self.statistics.decision_hits as f64 / total as f64
        }
    }

    /// Get overall cache hit ratio (combining plans and decisions)
    pub fn hit_ratio(&self) -> f64 {
        let total_hits = self.statistics.plan_hits + self.statistics.decision_hits;
        let total_misses = self.statistics.plan_misses + self.statistics.decision_misses;
        let total = total_hits + total_misses;
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// Get total number of requests made to the cache
    pub fn total_requests(&self) -> usize {
        self.statistics.total_lookups
    }

    fn is_expired(&self, creation_time: Instant) -> bool {
        creation_time.elapsed() > Duration::from_secs(self.config.ttl_seconds)
    }

    fn evict_least_recently_used_plan(&mut self) {
        if let Some((&key, _)) = self
            .plan_cache
            .iter()
            .min_by_key(|(_, cached)| cached.last_access)
        {
            self.plan_cache.remove(&key);
            self.statistics.evictions += 1;
        }
    }

    fn evict_oldest_decision(&mut self) {
        if let Some((&key, _)) = self
            .decision_cache
            .iter()
            .min_by_key(|(_, decision)| decision.creation_time)
        {
            self.decision_cache.remove(&key);
            self.statistics.evictions += 1;
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_plan_entries: 1000,
            max_decision_entries: 5000,
            ttl_seconds: 3600, // 1 hour
            enable_statistics: true,
        }
    }
}
