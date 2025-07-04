//! Intelligent Query Optimization Cache Module
//!
//! This module provides an advanced caching system for query optimization plans,
//! execution strategies, and performance metrics to significantly improve federation
//! query performance through intelligent plan reuse and adaptation.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::planner::planning::types::{ExecutionPlan, QueryInfo, QueryType};

/// Configuration for the optimization cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCacheConfig {
    /// Maximum number of cached plans
    pub max_cached_plans: usize,
    /// Plan cache TTL (time to live)
    pub plan_ttl: Duration,
    /// Enable adaptive caching based on performance
    pub enable_adaptive_caching: bool,
    /// Performance improvement threshold for caching
    pub performance_threshold: f64,
    /// Enable plan similarity matching
    pub enable_similarity_matching: bool,
    /// Similarity threshold for plan matching
    pub similarity_threshold: f64,
    /// Maximum age for performance data
    pub max_performance_age: Duration,
    /// Enable cache warming
    pub enable_cache_warming: bool,
    /// Cache warming interval
    pub cache_warming_interval: Duration,
}

impl Default for OptimizationCacheConfig {
    fn default() -> Self {
        Self {
            max_cached_plans: 1000,
            plan_ttl: Duration::from_secs(3600), // 1 hour
            enable_adaptive_caching: true,
            performance_threshold: 0.15, // 15% improvement
            enable_similarity_matching: true,
            similarity_threshold: 0.8,
            max_performance_age: Duration::from_secs(7200), // 2 hours
            enable_cache_warming: true,
            cache_warming_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Query fingerprint for efficient cache lookups
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryFingerprint {
    /// Query type
    pub query_type: QueryType,
    /// Pattern count
    pub pattern_count: usize,
    /// Variable count
    pub variable_count: usize,
    /// Filter count
    pub filter_count: usize,
    /// Complexity score (bucketed)
    pub complexity_bucket: u8,
    /// Service count
    pub service_count: usize,
    /// Query structure hash
    pub structure_hash: u64,
}

impl QueryFingerprint {
    /// Create a fingerprint from query information
    pub fn from_query_info(query_info: &QueryInfo) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query_info.original_query.hash(&mut hasher);

        Self {
            query_type: query_info.query_type,
            pattern_count: query_info.patterns.len(),
            variable_count: query_info.variables.len(),
            filter_count: query_info.filters.len(),
            complexity_bucket: Self::bucket_complexity(query_info.complexity),
            service_count: 1, // Will be updated based on planning
            structure_hash: hasher.finish(),
        }
    }

    /// Bucket complexity score for better cache hit rates
    fn bucket_complexity(complexity: u64) -> u8 {
        match complexity {
            0..=10 => 1,
            11..=50 => 2,
            51..=100 => 3,
            101..=500 => 4,
            501..=1000 => 5,
            _ => 6,
        }
    }

    /// Calculate similarity score with another fingerprint
    pub fn similarity(&self, other: &QueryFingerprint) -> f64 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        // Query type match (high weight)
        if self.query_type == other.query_type {
            score += 3.0;
        }
        total_weight += 3.0;

        // Pattern count similarity
        let pattern_similarity = 1.0
            - ((self.pattern_count as f64 - other.pattern_count as f64).abs()
                / (self.pattern_count.max(other.pattern_count) as f64 + 1.0));
        score += pattern_similarity * 2.0;
        total_weight += 2.0;

        // Variable count similarity
        let variable_similarity = 1.0
            - ((self.variable_count as f64 - other.variable_count as f64).abs()
                / (self.variable_count.max(other.variable_count) as f64 + 1.0));
        score += variable_similarity * 1.5;
        total_weight += 1.5;

        // Complexity bucket match
        if self.complexity_bucket == other.complexity_bucket {
            score += 1.0;
        }
        total_weight += 1.0;

        // Service count similarity
        let service_similarity = 1.0
            - ((self.service_count as f64 - other.service_count as f64).abs()
                / (self.service_count.max(other.service_count) as f64 + 1.0));
        score += service_similarity * 1.0;
        total_weight += 1.0;

        score / total_weight
    }
}

/// Cached execution plan with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPlan {
    /// The execution plan
    pub plan: ExecutionPlan,
    /// When the plan was cached
    pub cached_at: SystemTime,
    /// Number of times this plan was used
    pub usage_count: u32,
    /// Average execution time when using this plan
    pub avg_execution_time: Duration,
    /// Success rate for this plan
    pub success_rate: f64,
    /// Services used in this plan
    pub services: Vec<String>,
    /// Performance improvement over baseline
    pub performance_improvement: Option<f64>,
    /// Query fingerprint for similarity matching
    pub fingerprint: QueryFingerprint,
}

impl CachedPlan {
    /// Create a new cached plan
    pub fn new(plan: ExecutionPlan, fingerprint: QueryFingerprint) -> Self {
        Self {
            plan,
            cached_at: SystemTime::now(),
            usage_count: 0,
            avg_execution_time: Duration::from_millis(0),
            success_rate: 1.0,
            services: Vec::new(),
            performance_improvement: None,
            fingerprint,
        }
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self, execution_time: Duration, success: bool) {
        self.usage_count += 1;

        // Update average execution time
        let current_avg_ms = self.avg_execution_time.as_millis() as f64;
        let new_time_ms = execution_time.as_millis() as f64;
        let new_avg_ms = (current_avg_ms * (self.usage_count - 1) as f64 + new_time_ms)
            / self.usage_count as f64;
        self.avg_execution_time = Duration::from_millis(new_avg_ms as u64);

        // Update success rate
        let current_successes = (self.success_rate * (self.usage_count - 1) as f64).round() as u32;
        let new_successes = if success {
            current_successes + 1
        } else {
            current_successes
        };
        self.success_rate = new_successes as f64 / self.usage_count as f64;
    }

    /// Check if the plan is still valid (not expired)
    pub fn is_valid(&self, ttl: Duration) -> bool {
        match self.cached_at.elapsed() {
            Ok(age) => age < ttl,
            Err(_) => false,
        }
    }

    /// Calculate plan score for ranking
    pub fn calculate_score(&self) -> f64 {
        let time_factor = 1.0 / (self.avg_execution_time.as_millis() as f64 + 1.0);
        let usage_factor = (self.usage_count as f64).ln() + 1.0;
        let success_factor = self.success_rate;
        let improvement_factor = self.performance_improvement.unwrap_or(0.0) + 1.0;

        time_factor * usage_factor * success_factor * improvement_factor
    }
}

/// Performance data for cache optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    /// Execution time
    pub execution_time: Duration,
    /// Whether execution was successful
    pub success: bool,
    /// Memory usage
    pub memory_usage: u64,
    /// Services contacted
    pub services_contacted: Vec<String>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Statistics for the optimization cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Average plan reuse count
    pub avg_reuse_count: f64,
    /// Cache effectiveness (hit rate)
    pub hit_rate: f64,
    /// Performance improvement from caching
    pub performance_improvement: f64,
    /// Last updated
    pub last_updated: SystemTime,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            avg_reuse_count: 0.0,
            hit_rate: 0.0,
            performance_improvement: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

/// Intelligent query optimization cache
pub struct OptimizationCache {
    config: OptimizationCacheConfig,
    cached_plans: Arc<RwLock<HashMap<QueryFingerprint, CachedPlan>>>,
    performance_history: Arc<RwLock<VecDeque<PerformanceData>>>,
    statistics: Arc<RwLock<CacheStatistics>>,
    last_warming: Arc<RwLock<Instant>>,
}

impl OptimizationCache {
    /// Create a new optimization cache
    pub fn new(config: OptimizationCacheConfig) -> Self {
        Self {
            config,
            cached_plans: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            statistics: Arc::new(RwLock::new(CacheStatistics::default())),
            last_warming: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Look up a cached plan for a query
    pub async fn get_plan(&self, fingerprint: &QueryFingerprint) -> Option<CachedPlan> {
        let cached_plans = self.cached_plans.read().await;

        // Direct lookup first
        if let Some(plan) = cached_plans.get(fingerprint) {
            if plan.is_valid(self.config.plan_ttl) {
                self.record_hit().await;
                debug!("Cache hit for fingerprint: {:?}", fingerprint);
                return Some(plan.clone());
            }
        }

        // Similarity-based lookup if enabled
        if self.config.enable_similarity_matching {
            let mut best_match: Option<CachedPlan> = None;
            let mut best_similarity = 0.0;

            for plan in cached_plans.values() {
                if !plan.is_valid(self.config.plan_ttl) {
                    continue;
                }

                let similarity = fingerprint.similarity(&plan.fingerprint);
                if similarity > self.config.similarity_threshold && similarity > best_similarity {
                    best_similarity = similarity;
                    best_match = Some(plan.clone());
                }
            }

            if let Some(plan) = best_match {
                self.record_hit().await;
                debug!("Similarity cache hit with score: {:.3}", best_similarity);
                return Some(plan);
            }
        }

        self.record_miss().await;
        None
    }

    /// Cache an execution plan
    pub async fn cache_plan(&self, fingerprint: QueryFingerprint, plan: ExecutionPlan) {
        let mut cached_plans = self.cached_plans.write().await;

        // Check cache size limit
        if cached_plans.len() >= self.config.max_cached_plans {
            self.evict_least_valuable(&mut cached_plans).await;
        }

        let cached_plan = CachedPlan::new(plan, fingerprint.clone());
        cached_plans.insert(fingerprint.clone(), cached_plan);

        info!(
            "Cached new execution plan with fingerprint: {:?}",
            fingerprint
        );
    }

    /// Update performance metrics for a cached plan
    pub async fn update_performance(
        &self,
        fingerprint: &QueryFingerprint,
        execution_time: Duration,
        success: bool,
    ) {
        let mut cached_plans = self.cached_plans.write().await;

        if let Some(plan) = cached_plans.get_mut(fingerprint) {
            plan.update_metrics(execution_time, success);
            debug!("Updated performance metrics for plan: {:?}", fingerprint);
        }

        // Also update performance history
        let performance_data = PerformanceData {
            execution_time,
            success,
            memory_usage: 0,                // Would be provided by caller
            services_contacted: Vec::new(), // Would be provided by caller
            timestamp: SystemTime::now(),
        };

        let mut history = self.performance_history.write().await;
        history.push_back(performance_data);

        // Limit history size
        while history.len() > 10000 {
            history.pop_front();
        }
    }

    /// Evict the least valuable cached plan
    async fn evict_least_valuable(&self, cached_plans: &mut HashMap<QueryFingerprint, CachedPlan>) {
        if cached_plans.is_empty() {
            return;
        }

        let mut lowest_score = f64::INFINITY;
        let mut evict_key: Option<QueryFingerprint> = None;

        for (fingerprint, plan) in cached_plans.iter() {
            let score = plan.calculate_score();
            if score < lowest_score {
                lowest_score = score;
                evict_key = Some(fingerprint.clone());
            }
        }

        if let Some(key) = evict_key {
            cached_plans.remove(&key);
            self.record_eviction().await;
            debug!("Evicted plan with lowest score: {:.3}", lowest_score);
        }
    }

    /// Record a cache hit
    async fn record_hit(&self) {
        let mut stats = self.statistics.write().await;
        stats.hits += 1;
        stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
        stats.last_updated = SystemTime::now();
    }

    /// Record a cache miss
    async fn record_miss(&self) {
        let mut stats = self.statistics.write().await;
        stats.misses += 1;
        stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
        stats.last_updated = SystemTime::now();
    }

    /// Record an eviction
    async fn record_eviction(&self) {
        let mut stats = self.statistics.write().await;
        stats.evictions += 1;
        stats.last_updated = SystemTime::now();
    }

    /// Get cache statistics
    pub async fn get_statistics(&self) -> CacheStatistics {
        let stats = self.statistics.read().await;
        stats.clone()
    }

    /// Warm the cache with commonly used query patterns
    pub async fn warm_cache(&self) -> Result<()> {
        let now = Instant::now();
        let last_warming = self.last_warming.read().await;

        if now.duration_since(*last_warming) < self.config.cache_warming_interval {
            return Ok(());
        }

        info!("Starting cache warming process");

        // Analyze performance history to identify patterns
        let history = self.performance_history.read().await;
        let mut pattern_performance: HashMap<String, Vec<Duration>> = HashMap::new();

        for data in history.iter() {
            if data.success {
                // Group by execution time buckets for pattern analysis
                let bucket = Self::get_time_bucket(data.execution_time);
                pattern_performance
                    .entry(bucket)
                    .or_default()
                    .push(data.execution_time);
            }
        }

        // Pre-create fingerprints for common patterns
        let common_patterns = vec![
            ("simple_select", QueryType::Select, 1, 1, 0),
            ("complex_select", QueryType::Select, 5, 3, 2),
            ("construct", QueryType::Construct, 3, 2, 1),
            ("ask", QueryType::Ask, 1, 0, 0),
        ];

        for (name, query_type, patterns, vars, filters) in common_patterns {
            let fingerprint = QueryFingerprint {
                query_type,
                pattern_count: patterns,
                variable_count: vars,
                filter_count: filters,
                complexity_bucket: 2,
                service_count: 1,
                structure_hash: name
                    .as_bytes()
                    .iter()
                    .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64)),
            };

            // Check if we should pre-warm this pattern
            if !self.cached_plans.read().await.contains_key(&fingerprint) {
                debug!("Would pre-warm pattern: {}", name);
                // In a real implementation, we would create optimized plans here
            }
        }

        // Update last warming time
        *self.last_warming.write().await = now;

        info!("Cache warming completed");
        Ok(())
    }

    /// Get time bucket for performance analysis
    fn get_time_bucket(duration: Duration) -> String {
        let ms = duration.as_millis();
        match ms {
            0..=100 => "fast".to_string(),
            101..=1000 => "medium".to_string(),
            1001..=5000 => "slow".to_string(),
            _ => "very_slow".to_string(),
        }
    }

    /// Analyze cache effectiveness and provide recommendations
    pub async fn analyze_effectiveness(&self) -> Result<CacheAnalysis> {
        let stats = self.statistics.read().await;
        let cached_plans = self.cached_plans.read().await;
        let performance_history = self.performance_history.read().await;

        let total_requests = stats.hits + stats.misses;
        let hit_rate = if total_requests > 0 {
            stats.hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let mut avg_reuse = 0.0;
        if !cached_plans.is_empty() {
            avg_reuse = cached_plans
                .values()
                .map(|p| p.usage_count as f64)
                .sum::<f64>()
                / cached_plans.len() as f64;
        }

        let recommendations = self
            .generate_recommendations(hit_rate, avg_reuse, &cached_plans)
            .await;

        Ok(CacheAnalysis {
            hit_rate,
            avg_reuse_count: avg_reuse,
            total_cached_plans: cached_plans.len(),
            memory_usage_estimate: cached_plans.len() * 1024, // Rough estimate
            recommendations,
            effectiveness_score: self.calculate_effectiveness_score(hit_rate, avg_reuse),
        })
    }

    /// Generate optimization recommendations
    async fn generate_recommendations(
        &self,
        hit_rate: f64,
        avg_reuse: f64,
        cached_plans: &HashMap<QueryFingerprint, CachedPlan>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if hit_rate < 0.3 {
            recommendations.push(
                "Consider increasing cache size or adjusting similarity threshold".to_string(),
            );
        }

        if avg_reuse < 2.0 {
            recommendations
                .push("Query patterns show low reuse - consider query optimization".to_string());
        }

        if cached_plans.len() < self.config.max_cached_plans / 10 {
            recommendations
                .push("Cache utilization is low - consider more aggressive caching".to_string());
        }

        let expired_count = cached_plans
            .values()
            .filter(|p| !p.is_valid(self.config.plan_ttl))
            .count();

        if expired_count > cached_plans.len() / 4 {
            recommendations.push("Many cached plans are expired - consider longer TTL".to_string());
        }

        recommendations
    }

    /// Calculate overall effectiveness score
    fn calculate_effectiveness_score(&self, hit_rate: f64, avg_reuse: f64) -> f64 {
        let hit_score = hit_rate;
        let reuse_score = (avg_reuse - 1.0).max(0.0) / 10.0; // Normalize to 0-1 range
        let combined_score = (hit_score * 0.7 + reuse_score * 0.3).min(1.0);
        combined_score
    }

    /// Clean expired entries from the cache
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let mut cached_plans = self.cached_plans.write().await;
        let initial_count = cached_plans.len();

        cached_plans.retain(|_, plan| plan.is_valid(self.config.plan_ttl));

        let removed_count = initial_count - cached_plans.len();
        if removed_count > 0 {
            info!("Cleaned up {} expired cache entries", removed_count);
        }

        Ok(removed_count)
    }
}

/// Cache effectiveness analysis result
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheAnalysis {
    pub hit_rate: f64,
    pub avg_reuse_count: f64,
    pub total_cached_plans: usize,
    pub memory_usage_estimate: usize,
    pub recommendations: Vec<String>,
    pub effectiveness_score: f64,
}

impl Default for OptimizationCache {
    fn default() -> Self {
        Self::new(OptimizationCacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_fingerprint_similarity() {
        let fp1 = QueryFingerprint {
            query_type: QueryType::Select,
            pattern_count: 3,
            variable_count: 2,
            filter_count: 1,
            complexity_bucket: 2,
            service_count: 1,
            structure_hash: 12345,
        };

        let fp2 = QueryFingerprint {
            query_type: QueryType::Select,
            pattern_count: 3,
            variable_count: 2,
            filter_count: 1,
            complexity_bucket: 2,
            service_count: 1,
            structure_hash: 54321,
        };

        let similarity = fp1.similarity(&fp2);
        assert!(
            similarity > 0.8,
            "Similar queries should have high similarity score"
        );
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let cache = OptimizationCache::default();

        let fingerprint = QueryFingerprint {
            query_type: QueryType::Select,
            pattern_count: 1,
            variable_count: 1,
            filter_count: 0,
            complexity_bucket: 1,
            service_count: 1,
            structure_hash: 67890,
        };

        // Test cache miss
        assert!(cache.get_plan(&fingerprint).await.is_none());

        // Test cache storage and hit
        let plan = ExecutionPlan {
            query_id: "test-query".to_string(),
            steps: Vec::new(),
            estimated_total_cost: 100.0,
            max_parallelism: 4,
            planning_time: Duration::from_millis(50),
            cache_key: None,
            metadata: HashMap::new(),
            parallelizable_steps: Vec::new(),
        };

        cache.cache_plan(fingerprint.clone(), plan).await;
        assert!(cache.get_plan(&fingerprint).await.is_some());
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let cache = OptimizationCache::default();
        let fingerprint = QueryFingerprint {
            query_type: QueryType::Select,
            pattern_count: 1,
            variable_count: 1,
            filter_count: 0,
            complexity_bucket: 1,
            service_count: 1,
            structure_hash: 11111,
        };

        // Generate some hits and misses
        cache.get_plan(&fingerprint).await; // miss
        cache.get_plan(&fingerprint).await; // miss

        let stats = cache.get_statistics().await;
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hit_rate, 0.0);
    }
}
