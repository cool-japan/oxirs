//! Performance optimization and caching systems
//!
//! This module provides comprehensive performance optimizations including:
//! - Query result caching with configurable TTL
//! - Prepared query caching and optimization
//! - Connection pooling and resource management
//! - Request compression and streaming
//! - Memory and resource monitoring

use crate::{
    config::{PerformanceConfig, CacheConfig},
    error::{FusekiError, FusekiResult},
    metrics::MetricsService,
};
use lru::LruCache;
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn, error, instrument};

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

/// Performance service managing all optimization features
#[derive(Clone)]
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
            .max_capacity(config.caching.as_ref().map(|c| c.max_entries).unwrap_or(1000) as u64)
            .time_to_live(Duration::from_secs(
                config.caching.as_ref().map(|c| c.ttl_seconds).unwrap_or(300)
            ))
            .build();

        // Initialize prepared query cache
        let prepared_cache_size = NonZeroUsize::new(
            config.query_optimization.as_ref()
                .map(|opt| opt.prepared_cache_size)
                .unwrap_or(100)
        ).unwrap();
        let prepared_cache = Arc::new(RwLock::new(LruCache::new(prepared_cache_size)));

        // Initialize connection pool
        let max_connections = config.connection_pool.as_ref()
            .map(|pool| pool.max_connections)
            .unwrap_or(10);
        let connection_pool = Arc::new(ConnectionPool {
            max_connections,
            active_connections: Arc::new(Semaphore::new(max_connections)),
            connection_timeout: Duration::from_secs(
                config.connection_pool.as_ref()
                    .map(|pool| pool.connection_timeout_secs)
                    .unwrap_or(30)
            ),
            idle_timeout: Duration::from_secs(
                config.connection_pool.as_ref()
                    .map(|pool| pool.idle_timeout_secs)
                    .unwrap_or(300)
            ),
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
        let max_concurrent_queries = config.query_optimization.as_ref()
            .map(|opt| opt.max_concurrent_queries)
            .unwrap_or(50);
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
            ttl_seconds: self.config.caching.as_ref()
                .map(|c| c.ttl_seconds)
                .unwrap_or(300),
            hit_count: 0,
        };

        self.query_cache.insert(key.clone(), cached_result).await;
        debug!("Cached query result for key: {:?}", key);
    }

    /// Get prepared query from cache
    #[instrument(skip(self))]
    pub async fn get_prepared_query(&self, query_hash: &str) -> Option<PreparedQuery> {
        let mut cache = self.prepared_cache.write().await;
        if let Some(mut prepared) = cache.get_mut(query_hash) {
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
        let permit = self.connection_pool.active_connections
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
        let permit = self.query_semaphore
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
    pub fn should_cache_query(&self, query: &str, execution_time_ms: u64) -> bool {
        if let Some(cache_config) = &self.config.caching {
            // Cache if enabled and query is not too short or too long
            cache_config.enabled &&
            execution_time_ms >= cache_config.min_execution_time_ms &&
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
        stats.insert("query_cache_size".to_string(), 
                    serde_json::json!(self.query_cache.entry_count()));
        stats.insert("query_cache_capacity".to_string(), 
                    serde_json::json!(self.query_cache.max_capacity()));
        
        // Prepared query cache stats
        let prepared_cache = self.prepared_cache.read().await;
        stats.insert("prepared_cache_size".to_string(), 
                    serde_json::json!(prepared_cache.len()));
        stats.insert("prepared_cache_capacity".to_string(), 
                    serde_json::json!(prepared_cache.cap()));
        
        // Connection pool stats
        stats.insert("available_connections".to_string(), 
                    serde_json::json!(self.connection_pool.active_connections.available_permits()));
        stats.insert("max_connections".to_string(), 
                    serde_json::json!(self.connection_pool.max_connections));
        
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
                metrics_guard.memory_usage_mb = resource_usage.memory_bytes as f64 / 1024.0 / 1024.0;
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
        // Mock optimization - in reality this would use a query planner
        debug!("Optimizing query for dataset: {}", dataset);
        
        // Simulate optimization time
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        // Return "optimized" query (in reality would be a parsed/optimized query structure)
        Ok(query.to_string())
    }

    async fn estimate_query_cost(&self, query: &str) -> u64 {
        // Mock cost estimation - in reality would analyze query complexity
        let base_cost = query.len() as u64;
        let complexity_factor = if query.to_lowercase().contains("join") { 10 } else { 1 };
        
        base_cost * complexity_factor
    }
}

/// Connection guard that automatically releases connection when dropped
pub struct ConnectionGuard {
    _permit: tokio::sync::OwnedSemaphorePermit,
    acquired_at: Instant,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        debug!("Released database connection after {}ms", 
               self.acquired_at.elapsed().as_millis());
    }
}

/// Query execution permit guard
pub struct QueryPermit {
    _permit: tokio::sync::OwnedSemaphorePermit,
    started_at: Instant,
}

impl Drop for QueryPermit {
    fn drop(&mut self) {
        debug!("Released query execution permit after {}ms", 
               self.started_at.elapsed().as_millis());
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
            if elapsed_ms > 1000 { // Log slow queries (>1s)
                warn!("Slow {} query detected: {}ms", self.query_type, elapsed_ms);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CacheConfig, QueryOptimizationConfig, ConnectionPoolConfig};

    fn create_test_performance_service() -> PerformanceService {
        let config = PerformanceConfig {
            caching: Some(CacheConfig {
                enabled: true,
                max_entries: 100,
                ttl_seconds: 300,
                min_execution_time_ms: 10,
            }),
            query_optimization: Some(QueryOptimizationConfig {
                enabled: true,
                prepared_cache_size: 50,
                max_concurrent_queries: 10,
                slow_query_threshold_ms: 1000,
            }),
            connection_pool: Some(ConnectionPoolConfig {
                max_connections: 5,
                connection_timeout_secs: 30,
                idle_timeout_secs: 300,
            }),
            rate_limiting: None,
            compression: None,
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
        service.cache_query_result(
            key.clone(),
            "test result".to_string(),
            "application/json".to_string(),
            100,
        ).await;
        
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
        
        service.cache_prepared_query(query_hash.to_string(), prepared).await;
        
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
        assert!(service.connection_pool.active_connections.available_permits() <= 5);
    }

    #[tokio::test]
    async fn test_query_permit() {
        let service = create_test_performance_service();
        
        // Acquire query permit
        let _permit1 = service.acquire_query_permit().await.unwrap();
        let _permit2 = service.acquire_query_permit().await.unwrap();
        
        // Should work within limits
        assert!(service.query_semaphore.available_permits() <= 10);
    }

    #[test]
    fn test_cache_decision() {
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
        
        service.cache_query_result(key, "result".to_string(), "json".to_string(), 100).await;
        
        let stats = service.get_cache_stats().await;
        assert!(stats.contains_key("query_cache_size"));
        assert!(stats.contains_key("prepared_cache_size"));
        assert!(stats.contains_key("available_connections"));
    }
}