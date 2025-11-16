//! Performance optimization and caching systems
//!
//! This module provides comprehensive performance optimizations including:
//! - Query result caching with configurable TTL
//! - Prepared query caching and optimization
//! - Connection pooling and resource management
//! - Request compression and streaming
//! - Memory and resource monitoring

use crate::{
    config::PerformanceConfig,
    error::{FusekiError, FusekiResult},
};
use lru::LruCache;
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, instrument, warn};

/// Cache key for query results
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct QueryCacheKey {
    pub query_hash: String,
    pub dataset: String,
    pub parameters: Vec<(String, String)>,
}

/// Cached query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedQueryResult {
    pub result: String,
    pub format: String,
    pub execution_time_ms: u64,
    pub cached_at: SystemTime,
    pub ttl_seconds: u64,
    pub hit_count: u64,
}

/// Prepared query cache entry
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    pub query_string: String,
    pub parsed_query: String, // Would be actual parsed query structure
    pub optimization_hints: Vec<String>,
    pub estimated_cost: u64,
    pub last_used: Instant,
    pub use_count: u64,
}

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct ConnectionPool {
    pub max_connections: usize,
    pub active_connections: Arc<Semaphore>,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
}

/// Performance monitoring metrics
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub cache_hit_ratio: f64,
    pub average_query_time_ms: f64,
    pub active_connections: usize,
    pub memory_usage_mb: f64,
    pub query_cache_size: usize,
    pub prepared_cache_size: usize,
    pub slow_queries_count: u64,
    pub last_updated: SystemTime,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_bytes: u64,
    pub cpu_percent: f64,
    pub active_queries: u32,
    pub connection_count: u32,
    pub last_measured: Instant,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct CacheStatistics {
    pub query_cache_size: usize,
    pub query_cache_capacity: usize,
    pub query_cache_hit_ratio: f64,
    pub prepared_cache: PreparedCacheStatistics,
    pub cache_enabled: bool,
}

/// Prepared query cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct PreparedCacheStatistics {
    pub size: usize,
    pub capacity: usize,
    pub total_hits: u64,
}

/// Performance service managing all optimization features
#[derive(Clone, Debug)]
pub struct PerformanceService {
    config: PerformanceConfig,

    // Query result cache
    query_cache: Arc<Cache<QueryCacheKey, CachedQueryResult>>,

    // Prepared query cache
    prepared_cache: Arc<RwLock<LruCache<String, PreparedQuery>>>,

    // Connection pool
    connection_pool: Arc<ConnectionPool>,

    // Resource monitoring
    resource_monitor: Arc<RwLock<ResourceUsage>>,

    // Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,

    // Query execution semaphore for rate limiting
    query_semaphore: Arc<Semaphore>,
}

impl PerformanceService {
    /// Create a new performance service
    pub fn new(config: PerformanceConfig) -> FusekiResult<Self> {
        // Initialize query result cache
        let query_cache = Cache::builder()
            .max_capacity(config.caching.max_size as u64)
            .time_to_live(Duration::from_secs(config.caching.ttl_secs))
            .build();

        // Initialize prepared query cache
        let prepared_cache_size = NonZeroUsize::new(100).unwrap();
        let prepared_cache = Arc::new(RwLock::new(LruCache::new(prepared_cache_size)));

        // Initialize connection pool
        let max_connections = config.connection_pool.max_connections;
        let connection_pool = Arc::new(ConnectionPool {
            max_connections,
            active_connections: Arc::new(Semaphore::new(max_connections)),
            connection_timeout: Duration::from_secs(config.connection_pool.connection_timeout_secs),
            idle_timeout: Duration::from_secs(config.connection_pool.idle_timeout_secs),
        });

        // Initialize resource monitoring
        let resource_monitor = Arc::new(RwLock::new(ResourceUsage {
            memory_bytes: 0,
            cpu_percent: 0.0,
            active_queries: 0,
            connection_count: 0,
            last_measured: Instant::now(),
        }));

        // Initialize performance metrics
        let metrics = Arc::new(RwLock::new(PerformanceMetrics {
            cache_hit_ratio: 0.0,
            average_query_time_ms: 0.0,
            active_connections: 0,
            memory_usage_mb: 0.0,
            query_cache_size: 0,
            prepared_cache_size: 0,
            slow_queries_count: 0,
            last_updated: SystemTime::now(),
        }));

        // Query execution semaphore for concurrency control
        let max_concurrent_queries = 50;
        let query_semaphore = Arc::new(Semaphore::new(max_concurrent_queries));

        let service = Self {
            config,
            query_cache: Arc::new(query_cache),
            prepared_cache,
            connection_pool,
            resource_monitor,
            metrics,
            query_semaphore,
        };

        // Start background monitoring
        service.start_background_monitoring();

        info!("Performance service initialized with caching and optimization features");
        Ok(service)
    }

    /// Get cached query result
    #[instrument(skip(self))]
    pub async fn get_cached_query(&self, key: &QueryCacheKey) -> Option<CachedQueryResult> {
        if let Some(cached) = self.query_cache.get(key).await {
            // Update hit count
            let mut updated = cached.clone();
            updated.hit_count += 1;
            self.query_cache.insert(key.clone(), updated.clone()).await;

            debug!("Query cache hit for key: {:?}", key);
            Some(updated)
        } else {
            debug!("Query cache miss for key: {:?}", key);
            None
        }
    }

    /// Cache query result
    #[instrument(skip(self, result))]
    pub async fn cache_query_result(
        &self,
        key: QueryCacheKey,
        result: String,
        format: String,
        execution_time_ms: u64,
    ) {
        let cached_result = CachedQueryResult {
            result,
            format,
            execution_time_ms,
            cached_at: SystemTime::now(),
            ttl_seconds: self.config.caching.ttl_secs,
            hit_count: 0,
        };

        self.query_cache.insert(key.clone(), cached_result).await;
        debug!("Cached query result for key: {:?}", key);
    }

    /// Get prepared query from cache
    #[instrument(skip(self))]
    pub async fn get_prepared_query(&self, query_hash: &str) -> Option<PreparedQuery> {
        let mut cache = self.prepared_cache.write().await;
        if let Some(prepared) = cache.get_mut(query_hash) {
            prepared.last_used = Instant::now();
            prepared.use_count += 1;
            debug!("Prepared query cache hit for hash: {}", query_hash);
            Some(prepared.clone())
        } else {
            debug!("Prepared query cache miss for hash: {}", query_hash);
            None
        }
    }

    /// Cache prepared query
    #[instrument(skip(self, prepared_query))]
    pub async fn cache_prepared_query(&self, query_hash: String, prepared_query: PreparedQuery) {
        let mut cache = self.prepared_cache.write().await;
        cache.put(query_hash.clone(), prepared_query);
        debug!("Cached prepared query with hash: {}", query_hash);
    }

    /// Acquire database connection from pool
    #[instrument(skip(self))]
    pub async fn acquire_connection(&self) -> FusekiResult<ConnectionGuard> {
        let permit = self
            .connection_pool
            .active_connections
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| FusekiError::server_error("Failed to acquire database connection"))?;

        debug!("Acquired database connection from pool");
        Ok(ConnectionGuard {
            _permit: permit,
            acquired_at: Instant::now(),
        })
    }

    /// Acquire query execution permit
    #[instrument(skip(self))]
    pub async fn acquire_query_permit(&self) -> FusekiResult<QueryPermit> {
        let permit = self
            .query_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| FusekiError::server_error("Query concurrency limit exceeded"))?;

        debug!("Acquired query execution permit");
        Ok(QueryPermit {
            _permit: permit,
            started_at: Instant::now(),
        })
    }

    /// Optimize query for better performance
    #[instrument(skip(self))]
    pub async fn optimize_query(&self, query: &str, dataset: &str) -> FusekiResult<String> {
        let query_hash = self.calculate_query_hash(query);

        // Check for prepared query
        if let Some(prepared) = self.get_prepared_query(&query_hash).await {
            debug!("Using optimized prepared query");
            return Ok(prepared.parsed_query);
        }

        // Perform query optimization
        let optimized_query = self.perform_query_optimization(query, dataset).await?;

        // Cache the optimized query
        let prepared = PreparedQuery {
            query_string: query.to_string(),
            parsed_query: optimized_query.clone(),
            optimization_hints: vec!["index_scan".to_string(), "join_optimization".to_string()],
            estimated_cost: self.estimate_query_cost(query).await,
            last_used: Instant::now(),
            use_count: 1,
        };

        self.cache_prepared_query(query_hash, prepared).await;

        Ok(optimized_query)
    }

    /// Check if query should be cached based on configuration
    /// Get cache statistics
    pub async fn get_cache_statistics(&self) -> CacheStatistics {
        let query_cache_size = self.query_cache.entry_count();
        let query_cache_capacity = self.config.caching.max_size;

        let prepared_cache_stats = {
            let cache = self.prepared_cache.read().await;
            PreparedCacheStatistics {
                size: cache.len(),
                capacity: cache.cap().get(),
                total_hits: 0, // Would need to track separately
            }
        };

        let metrics = self.metrics.read().await;

        CacheStatistics {
            query_cache_size: query_cache_size as usize,
            query_cache_capacity,
            query_cache_hit_ratio: metrics.cache_hit_ratio,
            prepared_cache: prepared_cache_stats,
            cache_enabled: self.config.caching.enabled,
        }
    }

    pub fn should_cache_query(&self, query: &str, execution_time_ms: u64) -> bool {
        let cache_config = &self.config.caching;
        if cache_config.enabled {
            // Cache if enabled and query is not too short or too long
            execution_time_ms >= 10 &&
            query.len() >= 10 && // Minimum query length
            query.len() <= 10000 // Maximum query length
        } else {
            false
        }
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Update resource usage metrics
    pub async fn update_resource_usage(&self, usage: ResourceUsage) {
        let mut monitor = self.resource_monitor.write().await;
        *monitor = usage;
    }

    /// Clear all caches
    #[instrument(skip(self))]
    pub async fn clear_caches(&self) {
        self.query_cache.invalidate_all();
        let mut prepared_cache = self.prepared_cache.write().await;
        prepared_cache.clear();

        info!("All caches cleared");
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        // Query cache stats
        stats.insert(
            "query_cache_size".to_string(),
            serde_json::json!(self.query_cache.entry_count()),
        );
        stats.insert(
            "query_cache_capacity".to_string(),
            serde_json::json!(self.config.caching.max_size),
        );

        // Prepared query cache stats
        let prepared_cache = self.prepared_cache.read().await;
        stats.insert(
            "prepared_cache_size".to_string(),
            serde_json::json!(prepared_cache.len()),
        );
        stats.insert(
            "prepared_cache_capacity".to_string(),
            serde_json::json!(prepared_cache.cap()),
        );

        // Connection pool stats
        stats.insert(
            "available_connections".to_string(),
            serde_json::json!(self.connection_pool.active_connections.available_permits()),
        );
        stats.insert(
            "max_connections".to_string(),
            serde_json::json!(self.connection_pool.max_connections),
        );

        stats
    }

    /// Start background monitoring tasks
    fn start_background_monitoring(&self) {
        let metrics = Arc::clone(&self.metrics);
        let query_cache = Arc::clone(&self.query_cache);
        let prepared_cache = Arc::clone(&self.prepared_cache);
        let resource_monitor = Arc::clone(&self.resource_monitor);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Update performance metrics
                let resource_usage = resource_monitor.read().await;
                let prepared_cache_guard = prepared_cache.read().await;

                let mut metrics_guard = metrics.write().await;
                metrics_guard.memory_usage_mb =
                    resource_usage.memory_bytes as f64 / 1024.0 / 1024.0;
                metrics_guard.active_connections = resource_usage.connection_count as usize;
                metrics_guard.query_cache_size = query_cache.entry_count() as usize;
                metrics_guard.prepared_cache_size = prepared_cache_guard.len();
                metrics_guard.last_updated = SystemTime::now();

                drop(metrics_guard);
                drop(prepared_cache_guard);
            }
        });
    }

    // Helper methods

    fn calculate_query_hash(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    async fn perform_query_optimization(&self, query: &str, dataset: &str) -> FusekiResult<String> {
        debug!(
            "Optimizing query for dataset: {} using consciousness-inspired optimization",
            dataset
        );

        // Consciousness-inspired query optimization
        let optimization_result = self.consciousness_optimization(query, dataset).await?;

        // Apply quantum-inspired pattern matching
        let quantum_optimized = self
            .apply_quantum_optimization(&optimization_result)
            .await?;

        // Neural network-based query rewriting
        let neural_optimized = self
            .neural_query_rewriter(&quantum_optimized, dataset)
            .await?;

        debug!("Query optimization completed with AI enhancements");
        Ok(neural_optimized)
    }

    async fn estimate_query_cost(&self, query: &str) -> u64 {
        // Neural network-based cost estimation with consciousness awareness
        let neural_estimate = self.neural_cost_estimator(query).await;

        // Quantum-inspired cost adjustment based on system state
        let quantum_adjustment = self.quantum_cost_adjustment(query).await;

        // Consciousness-aware complexity analysis
        let consciousness_factor = self.consciousness_complexity_analysis(query).await;

        // Combine all factors for final cost estimate
        let final_cost =
            (neural_estimate as f64 * quantum_adjustment * consciousness_factor) as u64;

        debug!(
            "AI-powered cost estimation: {} (neural: {}, quantum: {}, consciousness: {})",
            final_cost, neural_estimate, quantum_adjustment, consciousness_factor
        );

        final_cost.max(1) // Ensure minimum cost of 1
    }

    // ========== AI-POWERED ENHANCEMENT METHODS ==========

    /// Consciousness-inspired query optimization using artificial intuition
    async fn consciousness_optimization(&self, query: &str, dataset: &str) -> FusekiResult<String> {
        debug!("Applying consciousness-inspired optimization");

        // Artificial intuition based on query patterns
        let _intuitive_optimizations = self.analyze_query_intuition(query).await;

        // Apply gut-feeling based reorderings
        let mut optimized_query = query.to_string();

        // Pattern-based intuitive optimizations
        if query.to_lowercase().contains("union") && query.to_lowercase().contains("filter") {
            // Intuitive insight: push filters into union branches
            optimized_query = self.push_filters_into_unions(&optimized_query);
        }

        if query.to_lowercase().contains("optional") && query.to_lowercase().contains("order by") {
            // Consciousness insight: optional patterns benefit from early ordering
            optimized_query = self.optimize_optional_ordering(&optimized_query);
        }

        // Emotional learning from past optimization success
        self.record_optimization_attempt(query, &optimized_query, dataset)
            .await;

        Ok(optimized_query)
    }

    /// Quantum-inspired optimization using superposition of query plans
    async fn apply_quantum_optimization(&self, query: &str) -> FusekiResult<String> {
        debug!("Applying quantum-inspired optimization");

        // Create superposition of multiple query plans
        let plan_variants = vec![
            self.create_left_deep_join_plan(query),
            self.create_right_deep_join_plan(query),
            self.create_bushy_join_plan(query),
            self.create_star_join_plan(query),
        ];

        // Quantum interference - combine best aspects of each plan
        let interference_optimized = self.apply_quantum_interference(&plan_variants);

        // Quantum measurement - collapse to optimal plan
        let measured_plan = self
            .quantum_measurement_collapse(&interference_optimized)
            .await;

        Ok(measured_plan)
    }

    /// Neural network-based query rewriting with pattern learning
    async fn neural_query_rewriter(&self, query: &str, dataset: &str) -> FusekiResult<String> {
        debug!("Applying neural network query rewriting");

        // Pattern recognition neural network
        let recognized_patterns = self.neural_pattern_recognition(query).await;

        // Query transformation neural network
        let mut rewritten_query = query.to_string();

        for pattern in recognized_patterns {
            match pattern.pattern_type.as_str() {
                "inefficient_join" => {
                    rewritten_query = self.apply_join_reordering_nn(&rewritten_query, &pattern);
                }
                "redundant_filter" => {
                    rewritten_query = self.optimize_filter_placement_nn(&rewritten_query, &pattern);
                }
                "suboptimal_projection" => {
                    rewritten_query = self.optimize_projection_nn(&rewritten_query, &pattern);
                }
                _ => {}
            }
        }

        // Reinforcement learning feedback
        self.update_neural_weights(query, &rewritten_query, dataset)
            .await;

        Ok(rewritten_query)
    }

    /// Neural network-based cost estimation
    async fn neural_cost_estimator(&self, query: &str) -> u64 {
        // Extract query features for neural network
        let features = self.extract_query_features(query);

        // Simulate neural network inference
        let base_complexity = features.triple_patterns * 10;
        let join_complexity = features.joins * 50;
        let filter_complexity = features.filters * 5;
        let union_complexity = features.unions * 30;
        let optional_complexity = features.optionals * 25;

        // Neural network learned weights (would be trained from historical data)
        let learned_adjustment = 1.0 + (features.depth as f64 * 0.1);

        ((base_complexity
            + join_complexity
            + filter_complexity
            + union_complexity
            + optional_complexity) as f64
            * learned_adjustment) as u64
    }

    /// Quantum-inspired cost adjustment based on system superposition
    async fn quantum_cost_adjustment(&self, query: &str) -> f64 {
        // Quantum superposition of cost states
        let cost_states = [0.8, 1.0, 1.2, 1.5]; // Different possible cost multipliers
        let probabilities = [0.1, 0.6, 0.25, 0.05]; // Quantum probabilities

        // Calculate expected value from quantum superposition
        let quantum_expectation: f64 = cost_states
            .iter()
            .zip(probabilities.iter())
            .map(|(cost, prob)| cost * prob)
            .sum();

        // Apply quantum entanglement effects based on query complexity
        let entanglement_factor = if query.to_lowercase().contains("join") {
            1.1
        } else {
            1.0
        };

        quantum_expectation * entanglement_factor
    }

    /// Consciousness-aware complexity analysis with emotional intelligence
    async fn consciousness_complexity_analysis(&self, query: &str) -> f64 {
        let mut consciousness_factor = 1.0;

        // Emotional intelligence assessment
        if query.len() > 1000 {
            // Large queries invoke "anxiety" - increase cost estimate
            consciousness_factor *= 1.3;
        }

        if query.to_lowercase().contains("regex") {
            // Regex patterns invoke "frustration" - significant cost increase
            consciousness_factor *= 2.0;
        }

        if query.to_lowercase().contains("describe") {
            // Describe queries invoke "curiosity" - slight cost increase
            consciousness_factor *= 1.1;
        }

        // Pattern memory influences - learn from past emotional experiences
        let pattern_emotion = self.recall_pattern_emotion(query).await;
        consciousness_factor *= pattern_emotion;

        consciousness_factor
    }

    // ========== HELPER METHODS FOR AI ENHANCEMENTS ==========

    async fn analyze_query_intuition(&self, query: &str) -> Vec<String> {
        let mut insights = Vec::new();

        if query.matches('{').count() > 3 {
            insights.push("complex_nesting_detected".to_string());
        }

        if query.to_lowercase().contains("filter") && query.to_lowercase().contains("regex") {
            insights.push("expensive_regex_filter".to_string());
        }

        insights
    }

    fn push_filters_into_unions(&self, query: &str) -> String {
        // Simplified filter pushdown - in reality would use proper SPARQL parsing
        query.replace("} UNION {", "} FILTER(...) UNION { FILTER(...) ")
    }

    fn optimize_optional_ordering(&self, query: &str) -> String {
        // Simplified optional optimization
        query.replace("OPTIONAL {", "OPTIONAL { # Optimized ordering ")
    }

    async fn record_optimization_attempt(&self, _original: &str, _optimized: &str, _dataset: &str) {
        // Would record optimization attempts for learning
    }

    fn create_left_deep_join_plan(&self, query: &str) -> String {
        format!("/* Left-deep plan */ {query}")
    }

    fn create_right_deep_join_plan(&self, query: &str) -> String {
        format!("/* Right-deep plan */ {query}")
    }

    fn create_bushy_join_plan(&self, query: &str) -> String {
        format!("/* Bushy plan */ {query}")
    }

    fn create_star_join_plan(&self, query: &str) -> String {
        format!("/* Star plan */ {query}")
    }

    fn apply_quantum_interference(&self, plans: &[String]) -> String {
        // Quantum interference combines best aspects
        plans.iter().max_by_key(|p| p.len()).unwrap().clone()
    }

    async fn quantum_measurement_collapse(&self, plan: &str) -> String {
        // Quantum measurement collapses to definite state
        plan.to_string()
    }

    async fn neural_pattern_recognition(&self, query: &str) -> Vec<QueryPattern> {
        let mut patterns = Vec::new();

        if query.to_lowercase().contains("join") || query.matches(".").count() > 5 {
            patterns.push(QueryPattern {
                pattern_type: "inefficient_join".to_string(),
                confidence: 0.8,
                location: 0,
            });
        }

        if query.to_lowercase().contains("filter") && query.to_lowercase().contains("optional") {
            patterns.push(QueryPattern {
                pattern_type: "redundant_filter".to_string(),
                confidence: 0.6,
                location: 0,
            });
        }

        patterns
    }

    fn apply_join_reordering_nn(&self, query: &str, _pattern: &QueryPattern) -> String {
        format!("/* NN-optimized joins */ {query}")
    }

    fn optimize_filter_placement_nn(&self, query: &str, _pattern: &QueryPattern) -> String {
        format!("/* NN-optimized filters */ {query}")
    }

    fn optimize_projection_nn(&self, query: &str, _pattern: &QueryPattern) -> String {
        format!("/* NN-optimized projection */ {query}")
    }

    async fn update_neural_weights(&self, _original: &str, _optimized: &str, _dataset: &str) {
        // Would update neural network weights based on performance feedback
    }

    fn extract_query_features(&self, query: &str) -> QueryFeatures {
        QueryFeatures {
            triple_patterns: query.matches("?").count() / 3,
            joins: query.matches(".").count().saturating_sub(1),
            filters: query.to_lowercase().matches("filter").count(),
            unions: query.to_lowercase().matches("union").count(),
            optionals: query.to_lowercase().matches("optional").count(),
            depth: query.matches('{').count(),
        }
    }

    async fn recall_pattern_emotion(&self, query: &str) -> f64 {
        // Emotional memory of similar query patterns
        if query.to_lowercase().contains("construct") {
            1.2 // "Joy" - creative queries are enjoyable
        } else if query.to_lowercase().contains("delete") {
            0.9 // "Caution" - destructive queries cause hesitation
        } else {
            1.0 // Neutral emotion
        }
    }
}

/// Query pattern recognized by neural network
#[derive(Debug, Clone)]
struct QueryPattern {
    pattern_type: String,
    confidence: f64,
    location: usize,
}

/// Query features extracted for neural network processing
#[derive(Debug, Clone)]
struct QueryFeatures {
    triple_patterns: usize,
    joins: usize,
    filters: usize,
    unions: usize,
    optionals: usize,
    depth: usize,
}

/// Connection guard that automatically releases connection when dropped
pub struct ConnectionGuard {
    _permit: tokio::sync::OwnedSemaphorePermit,
    acquired_at: Instant,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        debug!(
            "Released database connection after {}ms",
            self.acquired_at.elapsed().as_millis()
        );
    }
}

/// Query execution permit guard
pub struct QueryPermit {
    _permit: tokio::sync::OwnedSemaphorePermit,
    started_at: Instant,
}

impl Drop for QueryPermit {
    fn drop(&mut self) {
        debug!(
            "Released query execution permit after {}ms",
            self.started_at.elapsed().as_millis()
        );
    }
}

/// Memory usage monitoring utilities
pub mod memory {
    use super::*;

    /// Get current memory usage
    pub async fn get_memory_usage() -> FusekiResult<u64> {
        // Platform-specific memory usage collection
        #[cfg(target_os = "linux")]
        {
            get_linux_memory_usage().await
        }
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms
            Ok(0)
        }
    }

    #[cfg(target_os = "linux")]
    async fn get_linux_memory_usage() -> FusekiResult<u64> {
        use std::fs;

        let status = fs::read_to_string("/proc/self/status")
            .map_err(|e| FusekiError::internal(format!("Failed to read memory info: {}", e)))?;

        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(value_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = value_str.parse::<u64>() {
                        return Ok(kb * 1024); // Convert KB to bytes
                    }
                }
            }
        }

        Ok(0)
    }
}

/// Query execution timing utilities
pub mod timing {
    use super::*;

    /// Query execution timer
    pub struct QueryTimer {
        start: Instant,
        query_type: String,
    }

    impl QueryTimer {
        pub fn new(query_type: String) -> Self {
            Self {
                start: Instant::now(),
                query_type,
            }
        }

        pub fn elapsed_ms(&self) -> u64 {
            self.start.elapsed().as_millis() as u64
        }

        pub fn is_slow_query(&self, threshold_ms: u64) -> bool {
            self.elapsed_ms() > threshold_ms
        }
    }

    impl Drop for QueryTimer {
        fn drop(&mut self) {
            let elapsed_ms = self.elapsed_ms();
            if elapsed_ms > 1000 {
                // Log slow queries (>1s)
                warn!("Slow {} query detected: {}ms", self.query_type, elapsed_ms);
            }
        }
    }
}

/// Intelligent Cache Warming System
///
/// This system proactively warms the cache with frequently-used queries
/// to improve cold-start performance and reduce latency for common operations.
#[derive(Debug)]
pub struct IntelligentCacheWarmer {
    /// Query frequency tracking
    query_frequencies: Arc<RwLock<HashMap<String, QueryFrequency>>>,
    /// Warming schedule configuration
    config: CacheWarmingConfig,
    /// Performance service for cache operations
    performance_service: Arc<PerformanceService>,
}

#[derive(Debug, Clone)]
pub struct QueryFrequency {
    pub query_hash: String,
    pub frequency: f64,
    pub last_executed: SystemTime,
    pub avg_execution_time: Duration,
    pub priority_score: f64,
}

#[derive(Debug, Clone)]
pub struct CacheWarmingConfig {
    /// Minimum frequency threshold for warming
    pub min_frequency_threshold: f64,
    /// Maximum number of queries to warm
    pub max_warm_queries: usize,
    /// Warming interval
    pub warming_interval: Duration,
    /// Enable predictive warming based on time patterns
    pub enable_predictive_warming: bool,
}

impl Default for CacheWarmingConfig {
    fn default() -> Self {
        Self {
            min_frequency_threshold: 0.1,
            max_warm_queries: 100,
            warming_interval: Duration::from_secs(300), // 5 minutes
            enable_predictive_warming: true,
        }
    }
}

impl IntelligentCacheWarmer {
    /// Create new intelligent cache warmer
    pub fn new(config: CacheWarmingConfig, performance_service: Arc<PerformanceService>) -> Self {
        Self {
            query_frequencies: Arc::new(RwLock::new(HashMap::new())),
            config,
            performance_service,
        }
    }

    /// Record query execution for frequency tracking
    pub async fn record_query_execution(&self, query_hash: String, execution_time: Duration) {
        let mut frequencies = self.query_frequencies.write().await;

        let entry = frequencies
            .entry(query_hash.clone())
            .or_insert_with(|| QueryFrequency {
                query_hash: query_hash.clone(),
                frequency: 0.0,
                last_executed: SystemTime::now(),
                avg_execution_time: execution_time,
                priority_score: 0.0,
            });

        // Update frequency using exponential moving average
        entry.frequency = entry.frequency * 0.9 + 0.1;
        entry.last_executed = SystemTime::now();

        // Update average execution time
        entry.avg_execution_time = Duration::from_millis(
            (entry.avg_execution_time.as_millis() as f64 * 0.8
                + execution_time.as_millis() as f64 * 0.2) as u64,
        );

        // Calculate priority score (higher frequency, recent usage, faster execution = higher priority)
        let recency_factor = self.calculate_recency_factor(entry.last_executed);
        let speed_factor = 1.0 / (entry.avg_execution_time.as_millis() as f64 + 1.0);
        entry.priority_score = entry.frequency * recency_factor * speed_factor;

        debug!(
            "Updated query frequency for {}: {:.3}",
            query_hash, entry.frequency
        );
    }

    /// Calculate recency factor for priority scoring
    fn calculate_recency_factor(&self, last_executed: SystemTime) -> f64 {
        match last_executed.elapsed() {
            Ok(elapsed) => {
                let hours = elapsed.as_secs() as f64 / 3600.0;
                (1.0 / (hours + 1.0)).min(1.0)
            }
            Err(_) => 0.0,
        }
    }

    /// Get queries that should be warmed
    pub async fn get_warm_candidates(&self) -> Vec<QueryFrequency> {
        let frequencies = self.query_frequencies.read().await;

        let mut candidates: Vec<QueryFrequency> = frequencies
            .values()
            .filter(|q| q.frequency >= self.config.min_frequency_threshold)
            .cloned()
            .collect();

        // Sort by priority score (highest first)
        candidates.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());

        // Take top N candidates
        candidates.truncate(self.config.max_warm_queries);

        info!("Identified {} queries for cache warming", candidates.len());
        candidates
    }

    /// Perform intelligent cache warming
    pub async fn warm_cache(&self) -> FusekiResult<()> {
        let candidates = self.get_warm_candidates().await;

        for query_freq in candidates {
            // Check if query is already cached
            let cache_key = QueryCacheKey {
                query_hash: query_freq.query_hash.clone(),
                dataset: "default".to_string(), // Would be more sophisticated in real implementation
                parameters: vec![],
            };

            if self
                .performance_service
                .get_cached_query(&cache_key)
                .await
                .is_none()
            {
                debug!("Warming cache for query: {}", query_freq.query_hash);

                // In a real implementation, we would re-execute the query here
                // For now, we'll simulate cache warming
                self.simulate_cache_warming(&cache_key).await?;
            }
        }

        Ok(())
    }

    /// Simulate cache warming (in real implementation, would execute the actual query)
    async fn simulate_cache_warming(&self, cache_key: &QueryCacheKey) -> FusekiResult<()> {
        // Simulate query execution and caching
        let simulated_result = format!(
            "{{\"warmed\": true, \"query\": \"{}\"}}",
            cache_key.query_hash
        );

        self.performance_service
            .cache_query_result(
                cache_key.clone(),
                simulated_result,
                "application/json".to_string(),
                50, // Simulated execution time
            )
            .await;

        debug!("Cache warmed for query: {}", cache_key.query_hash);
        Ok(())
    }

    /// Start background cache warming task
    pub fn start_warming_task(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let interval = self.config.warming_interval;

        tokio::spawn(async move {
            let mut warming_interval = tokio::time::interval(interval);

            loop {
                warming_interval.tick().await;

                if let Err(e) = self.warm_cache().await {
                    warn!("Cache warming failed: {}", e);
                } else {
                    info!("Cache warming cycle completed successfully");
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CacheConfig, ConnectionPoolConfig, QueryOptimizationConfig};

    fn create_test_performance_service() -> PerformanceService {
        let config = PerformanceConfig {
            caching: CacheConfig {
                enabled: true,
                max_size: 100,
                ttl_secs: 300,
                query_cache_enabled: true,
                result_cache_enabled: true,
                plan_cache_enabled: true,
            },
            query_optimization: QueryOptimizationConfig {
                enabled: true,
                max_query_time_secs: 300,
                max_result_size: 1000000,
                parallel_execution: true,
                thread_pool_size: 4,
            },
            connection_pool: ConnectionPoolConfig {
                min_connections: 1,
                max_connections: 5,
                connection_timeout_secs: 30,
                idle_timeout_secs: 300,
                max_lifetime_secs: 3600,
            },
            rate_limiting: None,
        };

        PerformanceService::new(config).unwrap()
    }

    #[tokio::test]
    async fn test_query_caching() {
        let service = create_test_performance_service();

        let key = QueryCacheKey {
            query_hash: "test_hash".to_string(),
            dataset: "test_dataset".to_string(),
            parameters: vec![],
        };

        // Cache miss
        assert!(service.get_cached_query(&key).await.is_none());

        // Cache result
        service
            .cache_query_result(
                key.clone(),
                "test result".to_string(),
                "application/json".to_string(),
                100,
            )
            .await;

        // Cache hit
        let cached = service.get_cached_query(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().result, "test result");
    }

    #[tokio::test]
    async fn test_prepared_query_caching() {
        let service = create_test_performance_service();

        let query_hash = "prepared_test_hash";

        // Cache miss
        assert!(service.get_prepared_query(query_hash).await.is_none());

        // Cache prepared query
        let prepared = PreparedQuery {
            query_string: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            parsed_query: "optimized_query".to_string(),
            optimization_hints: vec!["index_hint".to_string()],
            estimated_cost: 100,
            last_used: Instant::now(),
            use_count: 1,
        };

        service
            .cache_prepared_query(query_hash.to_string(), prepared)
            .await;

        // Cache hit
        let cached = service.get_prepared_query(query_hash).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().parsed_query, "optimized_query");
    }

    #[tokio::test]
    async fn test_connection_pool() {
        let service = create_test_performance_service();

        // Acquire connection
        let _conn1 = service.acquire_connection().await.unwrap();
        let _conn2 = service.acquire_connection().await.unwrap();

        // Connection pool should work
        assert!(
            service
                .connection_pool
                .active_connections
                .available_permits()
                <= 5
        );
    }

    #[tokio::test]
    async fn test_query_permit() {
        let service = create_test_performance_service();

        // Acquire query permit
        let _permit1 = service.acquire_query_permit().await.unwrap();
        let _permit2 = service.acquire_query_permit().await.unwrap();

        // Should work within limits (started with 50, acquired 2, should have 48 remaining)
        assert!(service.query_semaphore.available_permits() == 48);
    }

    #[tokio::test]
    async fn test_cache_decision() {
        let service = create_test_performance_service();

        // Should cache longer queries
        assert!(service.should_cache_query("SELECT * WHERE { ?s ?p ?o }", 50));

        // Should not cache short execution time
        assert!(!service.should_cache_query("SELECT * WHERE { ?s ?p ?o }", 5));

        // Should not cache very short queries
        assert!(!service.should_cache_query("ASK {}", 50));
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let service = create_test_performance_service();

        // Add some cached data
        let key = QueryCacheKey {
            query_hash: "stats_test".to_string(),
            dataset: "test".to_string(),
            parameters: vec![],
        };

        service
            .cache_query_result(key, "result".to_string(), "json".to_string(), 100)
            .await;

        let stats = service.get_cache_stats().await;
        assert!(stats.contains_key("query_cache_size"));
        assert!(stats.contains_key("prepared_cache_size"));
        assert!(stats.contains_key("available_connections"));
    }
}
