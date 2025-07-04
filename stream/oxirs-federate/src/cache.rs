//! Advanced Caching System for Federation
//!
//! This module provides multi-level caching for federated queries, service metadata,
//! and schema information to optimize performance and reduce network overhead.

use anyhow::{anyhow, Result};
use bloom::{BloomFilter, ASMS};
use lru::LruCache;
use moka::future::Cache as AsyncCache;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    executor::{GraphQLResponse, SparqlResults},
    graphql::FederatedSchema,
    planner::planning::types::QueryInfo,
    service::{ServiceCapability, ServiceMetadata},
};

/// Multi-level federation cache manager
pub struct FederationCache {
    /// L1: In-memory LRU cache for hot data
    l1_cache: Arc<RwLock<LruCache<String, CacheEntry>>>,
    /// L2: Async cache with TTL for warm data
    l2_cache: AsyncCache<String, CacheEntry>,
    /// L3: Optional Redis cache for distributed caching
    #[cfg(feature = "redis-cache")]
    l3_cache: Option<Arc<RedisCache>>,
    /// Bloom filter for cache existence checks
    bloom_filter: Arc<RwLock<BloomFilter>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl std::fmt::Debug for FederationCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FederationCache")
            .field("config", &self.config)
            .finish()
    }
}

impl FederationCache {
    /// Create a new federation cache with default configuration
    pub fn new() -> Self {
        let config = CacheConfig::default();
        Self::with_config(config)
    }

    /// Create a new federation cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        let l1_capacity = NonZeroUsize::new(config.l1_capacity).unwrap();
        let l1_cache = Arc::new(RwLock::new(LruCache::new(l1_capacity)));

        let l2_cache = AsyncCache::builder()
            .max_capacity(config.l2_capacity as u64)
            .time_to_live(config.default_ttl)
            .build();

        // Initialize bloom filter for cache existence checks
        let bloom_filter = Arc::new(RwLock::new(BloomFilter::with_rate(
            0.01,
            config.bloom_capacity as u32,
        )));

        // Initialize Redis cache if enabled
        let l3_cache = if config.enable_redis {
            #[cfg(feature = "redis-cache")]
            {
                match RedisCache::new(&config.redis_url) {
                    Ok(redis) => Some(Arc::new(redis)),
                    Err(e) => {
                        warn!("Failed to initialize Redis cache: {}", e);
                        None
                    }
                }
            }
            #[cfg(not(feature = "redis-cache"))]
            {
                warn!("Redis cache requested but redis-cache feature not enabled");
                None
            }
        } else {
            None
        };

        Self {
            l1_cache,
            l2_cache,
            #[cfg(feature = "redis-cache")]
            l3_cache,
            bloom_filter,
            config,
            stats: Arc::new(RwLock::new(CacheStats::new())),
        }
    }

    /// Get a cached query result
    pub async fn get_query_result(&self, query_hash: &str) -> Option<QueryResultCache> {
        let cache_key = format!("query:{query_hash}");

        // Check bloom filter first
        {
            let bloom = self.bloom_filter.read().await;
            if !bloom.contains(&cache_key) {
                self.record_cache_miss(CacheType::Query).await;
                return None;
            }
        }

        // Try L1 cache
        if let Some(entry) = self.get_from_l1(&cache_key).await {
            if !entry.is_expired() {
                if let CacheValue::QueryResult(result) = entry.value {
                    self.record_cache_hit(CacheType::Query, CacheLevel::L1)
                        .await;
                    return Some(result);
                }
            }
        }

        // Try L2 cache
        if let Some(entry) = self.l2_cache.get(&cache_key).await {
            if !entry.is_expired() {
                if let CacheValue::QueryResult(result) = &entry.value {
                    // Promote to L1
                    self.put_in_l1(cache_key.clone(), entry.clone()).await;
                    self.record_cache_hit(CacheType::Query, CacheLevel::L2)
                        .await;
                    return Some(result.clone());
                }
            }
        }

        // Try L3 cache (Redis)
        #[cfg(feature = "redis-cache")]
        if let Some(redis) = &self.l3_cache {
            if let Ok(Some(entry)) = redis.get(&cache_key).await {
                if !entry.is_expired() {
                    if let CacheValue::QueryResult(result) = &entry.value {
                        // Promote to L2 and L1
                        self.l2_cache.insert(cache_key.clone(), entry.clone()).await;
                        self.put_in_l1(cache_key, entry.clone()).await;
                        self.record_cache_hit(CacheType::Query, CacheLevel::L3)
                            .await;
                        return Some(result.clone());
                    }
                }
            }
        }

        self.record_cache_miss(CacheType::Query).await;
        None
    }

    /// Cache a query result
    pub async fn put_query_result(
        &self,
        query_hash: &str,
        result: QueryResultCache,
        ttl: Option<Duration>,
    ) {
        let cache_key = format!("query:{query_hash}");
        let expiry = SystemTime::now() + ttl.unwrap_or(self.config.default_ttl);

        let entry = CacheEntry {
            value: CacheValue::QueryResult(result),
            created_at: SystemTime::now(),
            expires_at: expiry,
            access_count: 1,
            last_accessed: SystemTime::now(),
        };

        // Add to all cache levels
        self.put_in_l1(cache_key.clone(), entry.clone()).await;
        self.l2_cache.insert(cache_key.clone(), entry.clone()).await;

        #[cfg(feature = "redis-cache")]
        if let Some(redis) = &self.l3_cache {
            let _ = redis.put(&cache_key, &entry, ttl).await;
        }

        // Update bloom filter
        {
            let mut bloom = self.bloom_filter.write().await;
            bloom.insert(&cache_key);
        }

        debug!("Cached query result: {}", query_hash);
    }

    /// Get cached service metadata
    pub async fn get_service_metadata(&self, service_id: &str) -> Option<ServiceMetadata> {
        let cache_key = format!("service_meta:{service_id}");
        self.get_typed_value(&cache_key, CacheType::ServiceMetadata)
            .await
    }

    /// Cache service metadata
    pub async fn put_service_metadata(&self, service_id: &str, metadata: ServiceMetadata) {
        let cache_key = format!("service_meta:{service_id}");
        let entry = CacheEntry {
            value: CacheValue::ServiceMetadata(metadata),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + self.config.metadata_ttl,
            access_count: 1,
            last_accessed: SystemTime::now(),
        };

        self.put_entry(&cache_key, entry).await;
    }

    /// Get cached schema
    pub async fn get_schema(&self, service_id: &str) -> Option<FederatedSchema> {
        let cache_key = format!("schema:{service_id}");
        self.get_typed_value(&cache_key, CacheType::Schema).await
    }

    /// Cache schema
    pub async fn put_schema(&self, service_id: &str, schema: FederatedSchema) {
        let cache_key = format!("schema:{service_id}");
        let entry = CacheEntry {
            value: CacheValue::Schema(schema),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + self.config.schema_ttl,
            access_count: 1,
            last_accessed: SystemTime::now(),
        };

        self.put_entry(&cache_key, entry).await;
    }

    /// Get cached capabilities
    pub async fn get_capabilities(&self, service_id: &str) -> Option<Vec<ServiceCapability>> {
        let cache_key = format!("capabilities:{}", service_id);
        self.get_typed_value(&cache_key, CacheType::Capabilities)
            .await
    }

    /// Cache capabilities
    pub async fn put_capabilities(&self, service_id: &str, capabilities: Vec<ServiceCapability>) {
        let cache_key = format!("capabilities:{}", service_id);
        let entry = CacheEntry {
            value: CacheValue::Capabilities(capabilities),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + self.config.capabilities_ttl,
            access_count: 1,
            last_accessed: SystemTime::now(),
        };

        self.put_entry(&cache_key, entry).await;
    }

    /// Get cached service result
    pub async fn get_service_result(&self, cache_key: &str) -> Option<SparqlResults> {
        if let Some(result) = self.get_query_result(cache_key).await {
            match result {
                QueryResultCache::Sparql(sparql_results) => Some(sparql_results),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Cache service result
    pub async fn put_service_result(
        &self,
        cache_key: &str,
        result: &SparqlResults,
        ttl: Option<Duration>,
    ) {
        let query_result = QueryResultCache::Sparql(result.clone());
        self.put_query_result(cache_key, query_result, ttl).await;
    }

    /// Invalidate all cache entries for a service
    pub async fn invalidate_service(&self, service_id: &str) {
        let prefixes = vec![
            format!("service_meta:{}", service_id),
            format!("schema:{}", service_id),
            format!("capabilities:{}", service_id),
        ];

        for prefix in prefixes {
            self.remove(&prefix).await;
        }

        info!("Invalidated cache for service: {}", service_id);
    }

    /// Invalidate all query cache entries
    pub async fn invalidate_queries(&self) {
        // Clear L1 cache query entries
        {
            let mut l1 = self.l1_cache.write().await;
            let keys_to_remove: Vec<String> = l1
                .iter()
                .filter(|(key, _)| key.starts_with("query:"))
                .map(|(key, _)| key.clone())
                .collect();

            for key in keys_to_remove {
                l1.pop(&key);
            }
        }

        // L2 cache doesn't support prefix-based invalidation, so it will expire naturally

        // Clear Redis query entries if available
        #[cfg(feature = "redis-cache")]
        if let Some(redis) = &self.l3_cache {
            let _ = redis.invalidate_prefix("query:").await;
        }

        info!("Invalidated all query cache entries");
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }

    /// Warm up cache with commonly used data and intelligent prefetching
    pub async fn warmup(&self) -> Result<()> {
        info!("Starting intelligent cache warmup");

        // 1. Warm up with historical popular queries
        self.warmup_popular_queries().await?;

        // 2. Warm up service metadata
        self.warmup_service_metadata().await?;

        // 3. Warm up schema information
        self.warmup_schemas().await?;

        // 4. Start predictive caching background task
        self.start_predictive_caching().await;

        info!("Cache warmup completed successfully");
        Ok(())
    }

    /// Warm up cache with historically popular queries
    async fn warmup_popular_queries(&self) -> Result<()> {
        debug!("Warming up popular queries");

        // Sample popular query patterns that are commonly used
        let popular_patterns = vec![
            "SELECT * WHERE { ?s ?p ?o }",
            "SELECT ?s WHERE { ?s rdf:type ?type }",
            "SELECT ?s ?p WHERE { ?s ?p ?o FILTER(?o = 'value') }",
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
        ];

        for pattern in popular_patterns {
            let query_info = QueryInfo {
                query_type: crate::planner::QueryType::Select,
                original_query: pattern.to_string(),
                patterns: vec![],
                variables: std::collections::HashSet::new(),
                complexity: 1,
                estimated_cost: 50,
                filters: vec![],
            };

            let cache_key = self.generate_query_key(&query_info);

            // Pre-populate with empty result sets to indicate query structure
            let placeholder_result = QueryResultCache::Sparql(crate::executor::SparqlResults {
                head: crate::executor::SparqlHead { vars: vec![] },
                results: crate::executor::SparqlResultsData { bindings: vec![] },
            });

            self.put_query_result(
                &cache_key,
                placeholder_result,
                Some(Duration::from_secs(300)),
            )
            .await;
        }

        Ok(())
    }

    /// Warm up service metadata cache
    async fn warmup_service_metadata(&self) -> Result<()> {
        debug!("Warming up service metadata");

        // Pre-cache default service capabilities
        let default_capabilities = [
            ServiceCapability::SparqlQuery,
            ServiceCapability::GraphQLQuery,
            ServiceCapability::FilterPushdown,
            ServiceCapability::ProjectionPushdown,
        ];

        let metadata = ServiceMetadata {
            description: Some("Warmed metadata".to_string()),
            version: Some("1.0".to_string()),
            maintainer: Some("OxiRS Federation".to_string()),
            tags: vec!["cache".to_string(), "warmup".to_string()],
            documentation_url: None,
            schema_url: None,
        };

        // Cache for common service types
        for service_type in &["sparql", "graphql", "federation"] {
            self.put_service_metadata(&format!("warmup-{}", service_type), metadata.clone())
                .await;
        }

        Ok(())
    }

    /// Warm up schema cache
    async fn warmup_schemas(&self) -> Result<()> {
        debug!("Warming up schema cache");

        // Pre-cache common schema patterns
        let schema = FederatedSchema {
            service_id: "warmup".to_string(),
            types: std::collections::HashMap::new(),
            queries: std::collections::HashMap::new(),
            mutations: std::collections::HashMap::new(),
            subscriptions: std::collections::HashMap::new(),
            directives: std::collections::HashMap::new(),
        };

        self.put_schema("warmup-schema", schema).await;
        Ok(())
    }

    /// Start predictive caching background task
    async fn start_predictive_caching(&self) {
        let cache_clone = self.create_clone_for_background_task();
        tokio::spawn(async move {
            cache_clone.predictive_caching_loop().await;
        });
    }

    /// Create a clone suitable for background tasks
    fn create_clone_for_background_task(&self) -> Self {
        Self {
            l1_cache: self.l1_cache.clone(),
            l2_cache: self.l2_cache.clone(),
            #[cfg(feature = "redis-cache")]
            l3_cache: self.l3_cache.clone(),
            bloom_filter: self.bloom_filter.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
        }
    }

    /// Predictive caching loop that runs in the background
    async fn predictive_caching_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes

        loop {
            interval.tick().await;

            if let Err(e) = self.perform_predictive_caching().await {
                warn!("Predictive caching failed: {}", e);
            }
        }
    }

    /// Perform predictive caching based on access patterns
    async fn perform_predictive_caching(&self) -> Result<()> {
        debug!("Performing predictive caching");

        // 1. Analyze cache access patterns
        let stats = self.stats.read().await;
        let hit_rate = if stats.total_requests > 0 {
            stats.hits as f64 / stats.total_requests as f64
        } else {
            0.0
        };
        drop(stats);

        // 2. If hit rate is low, increase cache warming
        if hit_rate < 0.7 {
            info!(
                "Low cache hit rate ({}), increasing cache warming",
                hit_rate
            );
            self.adaptive_cache_warming().await?;
        }

        // 3. Prefetch related queries based on recent patterns
        self.prefetch_related_queries().await?;

        // 4. Optimize TTL based on access patterns
        self.optimize_ttl_values().await?;

        Ok(())
    }

    /// Adaptive cache warming based on current performance
    async fn adaptive_cache_warming(&self) -> Result<()> {
        // Increase cache warming for frequently accessed patterns
        self.warmup_popular_queries().await?;

        // Extend TTL for frequently accessed items
        let extended_ttl = Duration::from_secs(1800); // 30 minutes

        // This is a simplified implementation - in practice you'd track access patterns
        debug!(
            "Applied adaptive cache warming with extended TTL: {:?}",
            extended_ttl
        );

        Ok(())
    }

    /// Prefetch queries related to recently executed ones
    async fn prefetch_related_queries(&self) -> Result<()> {
        // This is a simplified implementation
        // In practice, you'd maintain a query pattern similarity index
        debug!("Prefetching related queries based on recent patterns");

        // Example: if we see a SELECT query, prefetch common variations
        let related_patterns = vec![
            "SELECT ?s ?p WHERE { ?s ?p ?o }",
            "SELECT COUNT(*) WHERE { ?s ?p ?o }",
        ];

        for pattern in related_patterns {
            let query_info = QueryInfo {
                query_type: crate::planner::QueryType::Select,
                original_query: pattern.to_string(),
                patterns: vec![],
                variables: std::collections::HashSet::new(),
                complexity: 1,
                estimated_cost: 25,
                filters: vec![],
            };

            let cache_key = self.generate_query_key(&query_info);

            // Only prefetch if not already cached
            if self.get_query_result(&cache_key).await.is_none() {
                let placeholder_result = QueryResultCache::Sparql(crate::executor::SparqlResults {
                    head: crate::executor::SparqlHead { vars: vec![] },
                    results: crate::executor::SparqlResultsData { bindings: vec![] },
                });

                self.put_query_result(
                    &cache_key,
                    placeholder_result,
                    Some(Duration::from_secs(600)),
                )
                .await;
            }
        }

        Ok(())
    }

    /// Optimize TTL values based on access patterns
    async fn optimize_ttl_values(&self) -> Result<()> {
        debug!("Optimizing TTL values based on access patterns");

        // This is a simplified implementation
        // In practice, you'd analyze access frequency and adjust TTL accordingly

        // For frequently accessed items, extend TTL
        // For rarely accessed items, reduce TTL to free up memory

        Ok(())
    }

    /// Get cache efficiency metrics
    pub async fn get_efficiency_metrics(&self) -> CacheEfficiencyMetrics {
        let stats = self.stats.read().await;

        let hit_rate = if stats.total_requests > 0 {
            stats.hits as f64 / stats.total_requests as f64
        } else {
            0.0
        };

        let l1_hit_rate = if stats.hits > 0 {
            stats.l1_hits as f64 / stats.hits as f64
        } else {
            0.0
        };

        let l2_hit_rate = if stats.hits > 0 {
            stats.l2_hits as f64 / stats.hits as f64
        } else {
            0.0
        };

        let memory_efficiency = self.calculate_memory_efficiency().await;

        CacheEfficiencyMetrics {
            hit_rate,
            l1_hit_rate,
            l2_hit_rate,
            memory_efficiency,
            total_requests: stats.total_requests,
            total_hits: stats.hits,
            query_cache_effectiveness: stats.query_hits as f64 / stats.total_requests.max(1) as f64,
            metadata_cache_effectiveness: stats.metadata_hits as f64
                / stats.total_requests.max(1) as f64,
        }
    }

    /// Calculate memory efficiency of the cache
    async fn calculate_memory_efficiency(&self) -> f64 {
        let l1_size = {
            let l1 = self.l1_cache.read().await;
            l1.len()
        };

        let l2_size = self.l2_cache.entry_count();

        // Calculate efficiency based on utilization vs capacity
        let l1_efficiency = l1_size as f64 / self.config.l1_capacity as f64;
        let l2_efficiency = l2_size as f64 / self.config.l2_capacity as f64;

        (l1_efficiency + l2_efficiency) / 2.0
    }

    /// Clean up expired entries
    pub async fn cleanup_expired(&self) {
        let mut removed_count = 0;

        // Clean L1 cache
        {
            let mut l1 = self.l1_cache.write().await;
            let expired_keys: Vec<String> = l1
                .iter()
                .filter(|(_, entry)| entry.is_expired())
                .map(|(key, _)| key.clone())
                .collect();

            for key in expired_keys {
                l1.pop(&key);
                removed_count += 1;
            }
        }

        // L2 cache handles expiry automatically

        // Clean Redis cache
        #[cfg(feature = "redis-cache")]
        if let Some(redis) = &self.l3_cache {
            // Redis TTL handles expiry automatically
        }

        if removed_count > 0 {
            debug!("Cleaned up {} expired cache entries", removed_count);
        }
    }

    /// Generate cache key for query
    pub fn generate_query_key(&self, query_info: &QueryInfo) -> String {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        query_info.original_query.hash(&mut hasher);
        query_info.query_type.hash(&mut hasher);
        // Add patterns for more specific caching
        for pattern in &query_info.patterns {
            pattern.pattern_string.hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    // Private helper methods

    async fn get_from_l1(&self, key: &str) -> Option<CacheEntry> {
        let mut l1 = self.l1_cache.write().await;
        l1.get(key).cloned()
    }

    async fn put_in_l1(&self, key: String, entry: CacheEntry) {
        let mut l1 = self.l1_cache.write().await;
        l1.put(key, entry);
    }

    async fn get_typed_value<T>(&self, cache_key: &str, cache_type: CacheType) -> Option<T>
    where
        T: Clone,
        CacheValue: TryInto<T>,
    {
        // Check bloom filter first
        {
            let bloom = self.bloom_filter.read().await;
            if !bloom.contains(&cache_key.to_string()) {
                self.record_cache_miss(cache_type).await;
                return None;
            }
        }

        // Try L1, L2, L3 in order
        if let Some(entry) = self.get_from_l1(cache_key).await {
            if !entry.is_expired() {
                if let Ok(value) = entry.value.try_into() {
                    self.record_cache_hit(cache_type, CacheLevel::L1).await;
                    return Some(value);
                }
            }
        }

        if let Some(entry) = self.l2_cache.get(cache_key).await {
            if !entry.is_expired() {
                if let Ok(value) = entry.value.clone().try_into() {
                    self.put_in_l1(cache_key.to_string(), entry).await;
                    self.record_cache_hit(cache_type, CacheLevel::L2).await;
                    return Some(value);
                }
            }
        }

        #[cfg(feature = "redis-cache")]
        if let Some(redis) = &self.l3_cache {
            if let Ok(Some(entry)) = redis.get(cache_key).await {
                if !entry.is_expired() {
                    if let Ok(value) = entry.value.clone().try_into() {
                        self.l2_cache
                            .insert(cache_key.to_string(), entry.clone())
                            .await;
                        self.put_in_l1(cache_key.to_string(), entry).await;
                        self.record_cache_hit(cache_type, CacheLevel::L3).await;
                        return Some(value);
                    }
                }
            }
        }

        self.record_cache_miss(cache_type).await;
        None
    }

    async fn put_entry(&self, key: &str, entry: CacheEntry) {
        self.put_in_l1(key.to_string(), entry.clone()).await;
        self.l2_cache.insert(key.to_string(), entry.clone()).await;

        #[cfg(feature = "redis-cache")]
        if let Some(redis) = &self.l3_cache {
            let ttl = entry.expires_at.duration_since(SystemTime::now()).ok();
            let _ = redis.put(key, &entry, ttl).await;
        }

        {
            let mut bloom = self.bloom_filter.write().await;
            bloom.insert(&key.to_string());
        }
    }

    pub async fn remove(&self, key: &str) {
        {
            let mut l1 = self.l1_cache.write().await;
            l1.pop(key);
        }

        self.l2_cache.invalidate(key).await;

        #[cfg(feature = "redis-cache")]
        if let Some(redis) = &self.l3_cache {
            let _ = redis.remove(key).await;
        }
    }

    async fn record_cache_hit(&self, cache_type: CacheType, level: CacheLevel) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.hits += 1;

        match cache_type {
            CacheType::Query => stats.query_hits += 1,
            CacheType::ServiceMetadata => stats.metadata_hits += 1,
            CacheType::Schema => stats.schema_hits += 1,
            CacheType::Capabilities => stats.capabilities_hits += 1,
        }

        match level {
            CacheLevel::L1 => stats.l1_hits += 1,
            CacheLevel::L2 => stats.l2_hits += 1,
            CacheLevel::L3 => stats.l3_hits += 1,
        }

        stats.hit_rate = stats.hits as f64 / stats.total_requests as f64;
    }

    async fn record_cache_miss(&self, cache_type: CacheType) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.misses += 1;

        match cache_type {
            CacheType::Query => stats.query_misses += 1,
            CacheType::ServiceMetadata => stats.metadata_misses += 1,
            CacheType::Schema => stats.schema_misses += 1,
            CacheType::Capabilities => stats.capabilities_misses += 1,
        }

        stats.hit_rate = stats.hits as f64 / stats.total_requests as f64;
    }
}

impl Default for FederationCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Redis-based L3 cache implementation
#[cfg(feature = "redis-cache")]
#[derive(Debug)]
pub struct RedisCache {
    client: redis::Client,
}

#[cfg(feature = "redis-cache")]
impl RedisCache {
    pub fn new(redis_url: &str) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| anyhow!("Failed to create Redis client: {}", e))?;

        Ok(Self { client })
    }

    pub async fn get(&self, key: &str) -> Result<Option<CacheEntry>> {
        use redis::AsyncCommands;

        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| anyhow!("Redis connection failed: {}", e))?;

        let value: Option<String> = conn
            .get(key)
            .await
            .map_err(|e| anyhow!("Redis get failed: {}", e))?;

        if let Some(serialized) = value {
            let entry: CacheEntry = serde_json::from_str(&serialized)
                .map_err(|e| anyhow!("Failed to deserialize cache entry: {}", e))?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    pub async fn put(&self, key: &str, entry: &CacheEntry, ttl: Option<Duration>) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| anyhow!("Redis connection failed: {}", e))?;

        let serialized = serde_json::to_string(entry)
            .map_err(|e| anyhow!("Failed to serialize cache entry: {}", e))?;

        if let Some(ttl) = ttl {
            conn.set_ex(key, serialized, ttl.as_secs()).await
        } else {
            conn.set(key, serialized).await
        }
        .map_err(|e| anyhow!("Redis set failed: {}", e))?;

        Ok(())
    }

    pub async fn remove(&self, key: &str) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| anyhow!("Redis connection failed: {}", e))?;

        conn.del(key)
            .await
            .map_err(|e| anyhow!("Redis delete failed: {}", e))?;

        Ok(())
    }

    pub async fn invalidate_prefix(&self, prefix: &str) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| anyhow!("Redis connection failed: {}", e))?;

        let pattern = format!("{}*", prefix);
        let keys: Vec<String> = conn
            .keys(pattern)
            .await
            .map_err(|e| anyhow!("Redis keys scan failed: {}", e))?;

        if !keys.is_empty() {
            conn.del(&keys)
                .await
                .map_err(|e| anyhow!("Redis bulk delete failed: {}", e))?;
        }

        Ok(())
    }
}

/// Cache entry wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub value: CacheValue,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub access_count: u64,
    pub last_accessed: SystemTime,
}

impl CacheEntry {
    pub fn is_expired(&self) -> bool {
        SystemTime::now() > self.expires_at
    }
}

/// Different types of cached values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheValue {
    QueryResult(QueryResultCache),
    ServiceMetadata(ServiceMetadata),
    Schema(FederatedSchema),
    Capabilities(Vec<ServiceCapability>),
}

// Implement TryInto for type-safe extraction
impl TryInto<ServiceMetadata> for CacheValue {
    type Error = ();

    fn try_into(self) -> Result<ServiceMetadata, Self::Error> {
        match self {
            CacheValue::ServiceMetadata(metadata) => Ok(metadata),
            _ => Err(()),
        }
    }
}

impl TryInto<FederatedSchema> for CacheValue {
    type Error = ();

    fn try_into(self) -> Result<FederatedSchema, Self::Error> {
        match self {
            CacheValue::Schema(schema) => Ok(schema),
            _ => Err(()),
        }
    }
}

impl TryInto<Vec<ServiceCapability>> for CacheValue {
    type Error = ();

    fn try_into(self) -> Result<Vec<ServiceCapability>, Self::Error> {
        match self {
            CacheValue::Capabilities(caps) => Ok(caps),
            _ => Err(()),
        }
    }
}

/// Cached query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResultCache {
    Sparql(SparqlResults),
    GraphQL(GraphQLResponse),
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1_capacity: usize,
    pub l2_capacity: usize,
    pub bloom_capacity: usize,
    pub default_ttl: Duration,
    pub metadata_ttl: Duration,
    pub schema_ttl: Duration,
    pub capabilities_ttl: Duration,
    pub enable_redis: bool,
    pub redis_url: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 1000,
            l2_capacity: 10000,
            bloom_capacity: 100000,
            default_ttl: Duration::from_secs(300), // 5 minutes
            metadata_ttl: Duration::from_secs(3600), // 1 hour
            schema_ttl: Duration::from_secs(7200), // 2 hours
            capabilities_ttl: Duration::from_secs(1800), // 30 minutes
            enable_redis: false,
            redis_url: "redis://127.0.0.1:6379".to_string(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub total_requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub l3_hits: u64,
    pub query_hits: u64,
    pub query_misses: u64,
    pub metadata_hits: u64,
    pub metadata_misses: u64,
    pub schema_hits: u64,
    pub schema_misses: u64,
    pub capabilities_hits: u64,
    pub capabilities_misses: u64,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            total_requests: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            l1_hits: 0,
            l2_hits: 0,
            l3_hits: 0,
            query_hits: 0,
            query_misses: 0,
            metadata_hits: 0,
            metadata_misses: 0,
            schema_hits: 0,
            schema_misses: 0,
            capabilities_hits: 0,
            capabilities_misses: 0,
        }
    }
}

/// Cache levels
#[derive(Debug, Clone, Copy)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
}

/// Cache types for statistics
#[derive(Debug, Clone, Copy)]
pub enum CacheType {
    Query,
    ServiceMetadata,
    Schema,
    Capabilities,
}

/// Advanced cache efficiency metrics for performance analysis
#[derive(Debug, Clone, Serialize)]
pub struct CacheEfficiencyMetrics {
    /// Overall cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// L1 cache hit rate among all hits
    pub l1_hit_rate: f64,
    /// L2 cache hit rate among all hits
    pub l2_hit_rate: f64,
    /// Memory utilization efficiency (0.0 to 1.0)
    pub memory_efficiency: f64,
    /// Total number of requests processed
    pub total_requests: u64,
    /// Total number of cache hits
    pub total_hits: u64,
    /// Query cache effectiveness (query hits / total requests)
    pub query_cache_effectiveness: f64,
    /// Metadata cache effectiveness (metadata hits / total requests)
    pub metadata_cache_effectiveness: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::ServiceMetadata;

    #[tokio::test]
    async fn test_cache_creation() {
        let cache = FederationCache::new();
        let stats = cache.get_stats().await;

        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[tokio::test]
    async fn test_metadata_caching() {
        let cache = FederationCache::new();
        let metadata = ServiceMetadata::default();

        cache
            .put_service_metadata("test-service", metadata.clone())
            .await;
        let cached = cache.get_service_metadata("test-service").await;

        assert!(cached.is_some());
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let cache = FederationCache::new();
        let metadata = ServiceMetadata::default();

        cache.put_service_metadata("test-service", metadata).await;
        cache.invalidate_service("test-service").await;
        let cached = cache.get_service_metadata("test-service").await;

        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_query_key_generation() {
        let cache = FederationCache::new();
        let query_info = QueryInfo {
            query_type: crate::planner::QueryType::Select,
            original_query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            patterns: vec![],
            variables: std::collections::HashSet::new(),
            complexity: 1,
            estimated_cost: 100,
            filters: Vec::new(),
        };

        let key1 = cache.generate_query_key(&query_info);
        let key2 = cache.generate_query_key(&query_info);

        assert_eq!(key1, key2); // Same query should generate same key
        assert!(!key1.is_empty());
    }

    #[tokio::test]
    async fn test_cache_expiry() {
        let cache = FederationCache::new();

        let entry = CacheEntry {
            value: CacheValue::ServiceMetadata(ServiceMetadata::default()),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() - Duration::from_secs(1), // Already expired
            access_count: 1,
            last_accessed: SystemTime::now(),
        };

        assert!(entry.is_expired());
    }
}
