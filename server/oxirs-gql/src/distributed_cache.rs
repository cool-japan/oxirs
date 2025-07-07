//! Distributed Caching with Redis Integration
//!
//! This module provides high-performance distributed caching for GraphQL queries
//! with Redis backend, intelligent cache strategies, and federation support.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use redis::{cmd, Client};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub redis_urls: Vec<String>,
    pub default_ttl: Duration,
    pub max_cache_size: u64,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub cluster_mode: bool,
    pub sharding_strategy: ShardingStrategy,
    pub eviction_policy: EvictionPolicy,
    pub consistency_level: ConsistencyLevel,
    pub replication_factor: usize,
    pub local_cache_size: usize,
    pub prefetch_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_urls: vec!["redis://localhost:6379".to_string()],
            default_ttl: Duration::from_secs(3600),
            max_cache_size: 1024 * 1024 * 1024, // 1GB
            compression_enabled: true,
            encryption_enabled: false,
            cluster_mode: false,
            sharding_strategy: ShardingStrategy::ConsistentHashing,
            eviction_policy: EvictionPolicy::LRU,
            consistency_level: ConsistencyLevel::Eventual,
            replication_factor: 2,
            local_cache_size: 10000,
            prefetch_enabled: true,
        }
    }
}

/// Sharding strategies for distributed cache
#[derive(Debug, Clone)]
pub enum ShardingStrategy {
    ConsistentHashing,
    Range,
    ModuloHash,
    QueryType,
    ServiceAffinity,
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    TTL,
    Adaptive,
}

/// Consistency levels for distributed caching
#[derive(Debug, Clone)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Session,
    Bounded,
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub access_count: u64,
    pub last_accessed: SystemTime,
    pub size_bytes: usize,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Cache operation statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub sets: u64,
    pub deletes: u64,
    pub evictions: u64,
    pub total_size_bytes: u64,
    pub entry_count: u64,
    pub average_response_time: Duration,
}

/// Cache invalidation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationEvent {
    pub keys: Vec<String>,
    pub tags: Vec<String>,
    pub timestamp: SystemTime,
    pub source: String,
    pub reason: InvalidationReason,
}

/// Reasons for cache invalidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidationReason {
    SchemaChange,
    DataUpdate,
    Manual,
    TTLExpired,
    MemoryPressure,
    ErrorRecovery,
}

/// GraphQL query context for caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub query_hash: String,
    pub variables_hash: String,
    pub operation_name: Option<String>,
    pub user_id: Option<String>,
    pub service_ids: Vec<String>,
    pub schema_version: String,
    pub requested_fields: Vec<String>,
}

impl QueryContext {
    /// Generate cache key from query context
    pub fn cache_key(&self) -> String {
        format!(
            "gql:{}:{}:{}:{}",
            self.query_hash,
            self.variables_hash,
            self.schema_version,
            self.service_ids.join(",")
        )
    }

    /// Generate tags for cache invalidation
    pub fn tags(&self) -> Vec<String> {
        let mut tags = vec![
            format!("query:{}", self.query_hash),
            format!("schema:{}", self.schema_version),
        ];

        for service_id in &self.service_ids {
            tags.push(format!("service:{service_id}"));
        }

        for field in &self.requested_fields {
            tags.push(format!("field:{field}"));
        }

        if let Some(user_id) = &self.user_id {
            tags.push(format!("user:{user_id}"));
        }

        tags
    }
}

/// Distributed cache trait
#[async_trait]
pub trait DistributedCache: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    async fn set(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn exists(&self, key: &str) -> Result<bool>;
    async fn invalidate_by_tags(&self, tags: &[String]) -> Result<u64>;
    async fn get_stats(&self) -> Result<CacheStats>;
    async fn health_check(&self) -> Result<bool>;
    async fn clear(&self) -> Result<()>;
}

/// Redis-based distributed cache implementation
pub struct RedisDistributedCache {
    config: CacheConfig,
    redis_pool: Arc<RwLock<Vec<Client>>>,
    local_cache: Arc<RwLock<lru::LruCache<String, CacheEntry>>>,
    stats: Arc<RwLock<CacheStats>>,
    compression: Option<Arc<dyn CompressionStrategy>>,
    encryption: Option<Arc<dyn EncryptionStrategy>>,
}

impl RedisDistributedCache {
    /// Create a new Redis-based distributed cache
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let mut redis_clients = Vec::new();

        for redis_url in &config.redis_urls {
            let client = Client::open(redis_url.as_str())
                .map_err(|e| anyhow!("Failed to create Redis client: {}", e))?;
            redis_clients.push(client);
        }

        let local_cache = lru::LruCache::new(
            std::num::NonZeroUsize::new(config.local_cache_size)
                .unwrap_or(std::num::NonZeroUsize::new(1000).unwrap()),
        );

        let compression = if config.compression_enabled {
            Some(Arc::new(GzipCompressionStrategy::new()) as Arc<dyn CompressionStrategy>)
        } else {
            None
        };

        let encryption = if config.encryption_enabled {
            Some(Arc::new(AesEncryptionStrategy::new()) as Arc<dyn EncryptionStrategy>)
        } else {
            None
        };

        Ok(Self {
            config,
            redis_pool: Arc::new(RwLock::new(redis_clients)),
            local_cache: Arc::new(RwLock::new(local_cache)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            compression,
            encryption,
        })
    }

    /// Get Redis client for a given key
    async fn get_redis_client(&self, key: &str) -> Result<Client> {
        let clients = self.redis_pool.read().await;

        if clients.is_empty() {
            return Err(anyhow!("No Redis clients available"));
        }

        let index = match self.config.sharding_strategy {
            ShardingStrategy::ConsistentHashing => self.consistent_hash(key, clients.len()),
            ShardingStrategy::ModuloHash => self.modulo_hash(key, clients.len()),
            _ => 0, // Default to first client for other strategies
        };

        Ok(clients[index].clone())
    }

    /// Consistent hashing for key distribution
    fn consistent_hash(&self, key: &str, num_nodes: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % num_nodes
    }

    /// Simple modulo hashing
    fn modulo_hash(&self, key: &str, num_nodes: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % num_nodes
    }

    /// Process data through compression/encryption pipeline
    async fn process_data(&self, data: &[u8], encode: bool) -> Result<Vec<u8>> {
        let mut processed_data = data.to_vec();

        if encode {
            // Apply compression first
            if let Some(compression) = &self.compression {
                processed_data = compression.compress(&processed_data).await?;
            }

            // Then encryption
            if let Some(encryption) = &self.encryption {
                processed_data = encryption.encrypt(&processed_data).await?;
            }
        } else {
            // Reverse order for decoding: decrypt first
            if let Some(encryption) = &self.encryption {
                processed_data = encryption.decrypt(&processed_data).await?;
            }

            // Then decompress
            if let Some(compression) = &self.compression {
                processed_data = compression.decompress(&processed_data).await?;
            }
        }

        Ok(processed_data)
    }

    /// Update cache statistics
    async fn update_stats<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut CacheStats),
    {
        let mut stats = self.stats.write().await;
        update_fn(&mut stats);
    }
}

#[async_trait]
impl DistributedCache for RedisDistributedCache {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start_time = std::time::Instant::now();

        // Check local cache first
        {
            let mut local_cache = self.local_cache.write().await;
            if let Some(entry) = local_cache.get(key) {
                if entry.expires_at > SystemTime::now() {
                    self.update_stats(|stats| {
                        stats.hits += 1;
                        stats.average_response_time =
                            (stats.average_response_time + start_time.elapsed()) / 2;
                    })
                    .await;

                    return Ok(Some(entry.value.clone()));
                } else {
                    // Entry expired, remove it
                    local_cache.pop(key);
                }
            }
        }

        // Check Redis
        let client = self.get_redis_client(key).await?;
        let mut connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| anyhow!("Failed to get Redis connection: {}", e))?;

        let redis_result: Option<Vec<u8>> = cmd("GET")
            .arg(key)
            .query_async(&mut connection)
            .await
            .map_err(|e| anyhow!("Redis GET failed: {}", e))?;

        if let Some(raw_data) = redis_result {
            // Process data (decrypt/decompress)
            let processed_data = self.process_data(&raw_data, false).await?;

            // Store in local cache
            let entry = CacheEntry {
                key: key.to_string(),
                value: processed_data.clone(),
                created_at: SystemTime::now(),
                expires_at: SystemTime::now() + self.config.default_ttl,
                access_count: 1,
                last_accessed: SystemTime::now(),
                size_bytes: processed_data.len(),
                tags: Vec::new(),
                metadata: HashMap::new(),
            };

            {
                let mut local_cache = self.local_cache.write().await;
                local_cache.put(key.to_string(), entry);
            }

            self.update_stats(|stats| {
                stats.hits += 1;
                stats.average_response_time =
                    (stats.average_response_time + start_time.elapsed()) / 2;
            })
            .await;

            Ok(Some(processed_data))
        } else {
            self.update_stats(|stats| {
                stats.misses += 1;
                stats.average_response_time =
                    (stats.average_response_time + start_time.elapsed()) / 2;
            })
            .await;

            Ok(None)
        }
    }

    async fn set(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Result<()> {
        let ttl = ttl.unwrap_or(self.config.default_ttl);

        // Process data (compress/encrypt)
        let processed_data = self.process_data(&value, true).await?;

        // Store in Redis
        let client = self.get_redis_client(key).await?;
        let mut connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| anyhow!("Failed to get Redis connection: {}", e))?;

        cmd("SETEX")
            .arg(key)
            .arg(ttl.as_secs())
            .arg(&processed_data)
            .exec_async(&mut connection)
            .await
            .map_err(|e| anyhow!("Redis SETEX failed: {}", e))?;

        // Store in local cache
        let entry = CacheEntry {
            key: key.to_string(),
            value,
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + ttl,
            access_count: 0,
            last_accessed: SystemTime::now(),
            size_bytes: processed_data.len(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };

        {
            let mut local_cache = self.local_cache.write().await;
            local_cache.put(key.to_string(), entry);
        }

        self.update_stats(|stats| {
            stats.sets += 1;
            stats.total_size_bytes += processed_data.len() as u64;
            stats.entry_count += 1;
        })
        .await;

        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        // Remove from local cache
        {
            let mut local_cache = self.local_cache.write().await;
            local_cache.pop(key);
        }

        // Remove from Redis
        let client = self.get_redis_client(key).await?;
        let mut connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| anyhow!("Failed to get Redis connection: {}", e))?;

        cmd("DEL")
            .arg(key)
            .query_async::<()>(&mut connection)
            .await
            .map_err(|e| anyhow!("Redis DEL failed: {}", e))?;

        self.update_stats(|stats| {
            stats.deletes += 1;
        })
        .await;

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        // Check local cache first
        {
            let mut local_cache = self.local_cache.write().await;
            if let Some(entry) = local_cache.get(key) {
                if entry.expires_at > SystemTime::now() {
                    return Ok(true);
                } else {
                    local_cache.pop(key);
                }
            }
        }

        // Check Redis
        let client = self.get_redis_client(key).await?;
        let mut connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| anyhow!("Failed to get Redis connection: {}", e))?;

        let exists: bool = cmd("EXISTS")
            .arg(key)
            .query_async(&mut connection)
            .await
            .map_err(|e| anyhow!("Redis EXISTS failed: {}", e))?;

        Ok(exists)
    }

    async fn invalidate_by_tags(&self, tags: &[String]) -> Result<u64> {
        // This is a simplified implementation
        // A production implementation would use Redis sets to track keys by tags
        let mut invalidated = 0;

        for tag in tags {
            // Create a pattern to match keys with this tag
            let pattern = format!("*{tag}*");

            let clients = self.redis_pool.read().await;
            for client in clients.iter() {
                let mut connection = client.get_multiplexed_async_connection().await?;

                let keys: Vec<String> = cmd("KEYS")
                    .arg(&pattern)
                    .query_async(&mut connection)
                    .await?;

                for key in keys {
                    self.delete(&key).await?;
                    invalidated += 1;
                }
            }
        }

        Ok(invalidated)
    }

    async fn get_stats(&self) -> Result<CacheStats> {
        Ok(self.stats.read().await.clone())
    }

    async fn health_check(&self) -> Result<bool> {
        let clients = self.redis_pool.read().await;

        for client in clients.iter() {
            match client.get_multiplexed_async_connection().await {
                Ok(mut connection) => {
                    let result: Result<String, _> = cmd("PING").query_async(&mut connection).await;
                    if result.is_err() {
                        return Ok(false);
                    }
                }
                Err(_) => return Ok(false),
            }
        }

        Ok(true)
    }

    async fn clear(&self) -> Result<()> {
        // Clear local cache
        {
            let mut local_cache = self.local_cache.write().await;
            local_cache.clear();
        }

        // Clear Redis
        let clients = self.redis_pool.read().await;
        for client in clients.iter() {
            let mut connection = client.get_multiplexed_async_connection().await?;
            cmd("FLUSHDB").query_async::<()>(&mut connection).await?;
        }

        // Reset stats
        {
            let mut stats = self.stats.write().await;
            *stats = CacheStats::default();
        }

        Ok(())
    }
}

/// Compression strategy trait
#[async_trait]
pub trait CompressionStrategy: Send + Sync {
    async fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;
    async fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// Gzip compression strategy
pub struct GzipCompressionStrategy;

impl Default for GzipCompressionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl GzipCompressionStrategy {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CompressionStrategy for GzipCompressionStrategy {
    async fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    async fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
}

/// Encryption strategy trait
#[async_trait]
pub trait EncryptionStrategy: Send + Sync {
    async fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>>;
    async fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// AES encryption strategy (stub implementation)
pub struct AesEncryptionStrategy;

impl Default for AesEncryptionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl AesEncryptionStrategy {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl EncryptionStrategy for AesEncryptionStrategy {
    async fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Stub implementation - would use actual AES encryption
        Ok(data.to_vec())
    }

    async fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Stub implementation - would use actual AES decryption
        Ok(data.to_vec())
    }
}

/// GraphQL query cache manager
pub struct GraphQLQueryCache {
    cache: Arc<dyn DistributedCache>,
    config: CacheConfig,
}

impl GraphQLQueryCache {
    /// Create a new GraphQL query cache
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let cache = Arc::new(RedisDistributedCache::new(config.clone()).await?);

        Ok(Self { cache, config })
    }

    /// Cache a GraphQL query result
    pub async fn cache_query_result(
        &self,
        context: &QueryContext,
        result: &serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let key = context.cache_key();
        let value = serde_json::to_vec(result)?;

        self.cache.set(&key, value, ttl).await?;

        info!("Cached GraphQL query result: {}", key);
        Ok(())
    }

    /// Get cached GraphQL query result
    pub async fn get_cached_result(
        &self,
        context: &QueryContext,
    ) -> Result<Option<serde_json::Value>> {
        let key = context.cache_key();

        if let Some(cached_data) = self.cache.get(&key).await? {
            let result: serde_json::Value = serde_json::from_slice(&cached_data)?;
            debug!("Cache hit for GraphQL query: {}", key);
            return Ok(Some(result));
        }

        debug!("Cache miss for GraphQL query: {}", key);
        Ok(None)
    }

    /// Invalidate cache entries based on schema changes
    pub async fn invalidate_on_schema_change(&self, schema_version: &str) -> Result<u64> {
        let tags = vec![format!("schema:{}", schema_version)];
        self.cache.invalidate_by_tags(&tags).await
    }

    /// Invalidate cache entries for specific services
    pub async fn invalidate_for_services(&self, service_ids: &[String]) -> Result<u64> {
        let tags: Vec<String> = service_ids
            .iter()
            .map(|id| format!("service:{id}"))
            .collect();
        self.cache.invalidate_by_tags(&tags).await
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> Result<CacheStats> {
        self.cache.get_stats().await
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool> {
        self.cache.health_check().await
    }

    /// Raw cache get for internal use
    pub async fn raw_get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.cache.get(key).await
    }

    /// Raw cache set for internal use  
    pub async fn raw_set(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Result<()> {
        self.cache.set(key, value, ttl).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_context_cache_key() {
        let context = QueryContext {
            query_hash: "abc123".to_string(),
            variables_hash: "def456".to_string(),
            operation_name: Some("GetUser".to_string()),
            user_id: Some("user123".to_string()),
            service_ids: vec!["service1".to_string(), "service2".to_string()],
            schema_version: "v1.0".to_string(),
            requested_fields: vec!["name".to_string(), "email".to_string()],
        };

        let cache_key = context.cache_key();
        assert!(cache_key.contains("abc123"));
        assert!(cache_key.contains("def456"));
        assert!(cache_key.contains("v1.0"));
    }

    #[tokio::test]
    async fn test_gzip_compression() {
        let compression = GzipCompressionStrategy::new();
        // Use a larger, more repetitive string that will actually compress well
        let original_data = b"This is a test string for compression. ".repeat(100);

        let compressed = compression.compress(&original_data).await.unwrap();
        let decompressed = compression.decompress(&compressed).await.unwrap();

        assert_eq!(original_data.as_slice(), decompressed.as_slice());
        assert!(compressed.len() < original_data.len()); // Should be compressed
    }
}
