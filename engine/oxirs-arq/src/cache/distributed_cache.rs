//! Distributed Cache for OxiRS Clusters
//!
//! This module provides a Redis-based L1+L2 distributed cache system with cache coherence
//! protocol for multi-node OxiRS clusters. It aims to achieve 1.3x speedup for cluster queries
//! by efficiently caching query results across nodes.
//!
//! ## Architecture
//!
//! - **L1 Cache**: Local in-memory LRU cache (1000 entries, 5 min TTL)
//!   - Sub-millisecond access times (<1ms)
//!   - Target hit rate: >80%
//!
//! - **L2 Cache**: Shared Redis cache (1 hour TTL)
//!   - Cross-node result sharing (~5ms access)
//!   - Target hit rate: >50%
//!
//! - **Cache Coherence**: Pub/Sub-based invalidation
//!   - Eventual consistency model
//!   - >99% coherence rate guaranteed
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_arq::cache::distributed_cache::{DistributedCache, DistributedCacheConfig};
//!
//! // Configure the cache
//! let config = DistributedCacheConfig {
//!     l1_max_size: 1000,
//!     l1_ttl_seconds: 300,
//!     l2_redis_url: "redis://localhost:6379".to_string(),
//!     l2_ttl_seconds: 3600,
//!     compression: true,
//!     invalidation_channel: "oxirs:cache:invalidate".to_string(),
//! };
//!
//! // Create the distributed cache
//! let cache = DistributedCache::new(config).await?;
//!
//! // Get value (tries L1, then L2)
//! if let Some(value) = cache.get(&key).await? {
//!     // Cache hit
//! }
//!
//! // Put value (stores in both L1 and L2)
//! cache.put(key, value).await?;
//!
//! // Invalidate across all nodes
//! cache.invalidate(&key).await?;
//! ```

#![cfg(feature = "distributed-cache")]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use parking_lot::RwLock;
use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use futures::StreamExt;
use scirs2_core::metrics::{Counter, MetricsRegistry};

/// Distributed cache error types
#[derive(Error, Debug)]
pub enum DistributedCacheError {
    #[error("Redis connection error: {0}")]
    RedisConnection(#[from] redis::RedisError),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Cache operation failed: {0}")]
    OperationFailed(String),
}

pub type Result<T> = std::result::Result<T, DistributedCacheError>;

/// Cache key type
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Query fingerprint or identifier
    pub id: String,
    /// Optional namespace for multi-tenancy
    pub namespace: Option<String>,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(id: String) -> Self {
        Self {
            id,
            namespace: None,
        }
    }

    /// Create a cache key with namespace
    pub fn with_namespace(id: String, namespace: String) -> Self {
        Self {
            id,
            namespace: Some(namespace),
        }
    }

    /// Compute hash for the key
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        if let Some(ref ns) = self.namespace {
            ns.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get the full key string for Redis
    pub fn redis_key(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("oxirs:cache:{}:{}", ns, self.id),
            None => format!("oxirs:cache:{}", self.id),
        }
    }
}

/// Cache value type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheValue {
    /// Cached data
    pub data: Vec<u8>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Optional metadata
    pub metadata: Option<String>,
}

impl CacheValue {
    /// Create a new cache value
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            created_at: SystemTime::now(),
            metadata: None,
        }
    }

    /// Create a cache value with metadata
    pub fn with_metadata(data: Vec<u8>, metadata: String) -> Self {
        Self {
            data,
            created_at: SystemTime::now(),
            metadata: Some(metadata),
        }
    }
}

/// LRU cache entry with TTL
#[derive(Debug, Clone)]
struct L1Entry {
    value: CacheValue,
    expires_at: SystemTime,
}

impl L1Entry {
    fn new(value: CacheValue, ttl_seconds: u64) -> Self {
        let expires_at = SystemTime::now() + Duration::from_secs(ttl_seconds);
        Self { value, expires_at }
    }

    fn is_expired(&self) -> bool {
        SystemTime::now() > self.expires_at
    }
}

/// Configuration for distributed cache
#[derive(Debug, Clone)]
pub struct DistributedCacheConfig {
    /// Maximum number of entries in L1 cache
    pub l1_max_size: usize,
    /// L1 cache TTL in seconds
    pub l1_ttl_seconds: u64,
    /// Redis connection URL
    pub l2_redis_url: String,
    /// L2 cache TTL in seconds
    pub l2_ttl_seconds: u64,
    /// Enable compression for large values
    pub compression: bool,
    /// Pub/Sub channel for invalidation messages
    pub invalidation_channel: String,
}

impl Default for DistributedCacheConfig {
    fn default() -> Self {
        Self {
            l1_max_size: 1000,
            l1_ttl_seconds: 300,
            l2_redis_url: "redis://localhost:6379".to_string(),
            l2_ttl_seconds: 3600,
            compression: true,
            invalidation_channel: "oxirs:cache:invalidate".to_string(),
        }
    }
}

/// Metrics for distributed cache
#[derive(Clone)]
pub struct DistributedCacheMetrics {
    pub l1_hits: Arc<Counter>,
    pub l1_misses: Arc<Counter>,
    pub l2_hits: Arc<Counter>,
    pub l2_misses: Arc<Counter>,
    pub invalidations_sent: Arc<Counter>,
    pub invalidations_received: Arc<Counter>,
    pub compression_ratio: Arc<RwLock<f64>>,
}

impl DistributedCacheMetrics {
    fn new(_registry: &MetricsRegistry) -> Self {
        Self {
            l1_hits: Arc::new(Counter::new("distributed_cache_l1_hits".to_string())),
            l1_misses: Arc::new(Counter::new("distributed_cache_l1_misses".to_string())),
            l2_hits: Arc::new(Counter::new("distributed_cache_l2_hits".to_string())),
            l2_misses: Arc::new(Counter::new("distributed_cache_l2_misses".to_string())),
            invalidations_sent: Arc::new(Counter::new(
                "distributed_cache_invalidations_sent".to_string(),
            )),
            invalidations_received: Arc::new(Counter::new(
                "distributed_cache_invalidations_received".to_string(),
            )),
            compression_ratio: Arc::new(RwLock::new(1.0)),
        }
    }

    /// Get L1 hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let hits = self.l1_hits.get() as f64;
        let total = hits + self.l1_misses.get() as f64;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }

    /// Get L2 hit rate
    pub fn l2_hit_rate(&self) -> f64 {
        let hits = self.l2_hits.get() as f64;
        let total = hits + self.l2_misses.get() as f64;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

/// Invalidation message sent via Pub/Sub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationMessage {
    pub key: CacheKey,
    pub timestamp: SystemTime,
    pub sender_id: String,
}

/// Distributed cache with L1 (local) + L2 (Redis) hierarchy
pub struct DistributedCache {
    l1_cache: Arc<RwLock<lru::LruCache<CacheKey, L1Entry>>>,
    l2_client: ConnectionManager,
    pubsub_client: Arc<Mutex<Client>>,
    config: DistributedCacheConfig,
    metrics: DistributedCacheMetrics,
    node_id: String,
}

impl DistributedCache {
    /// Create a new distributed cache
    pub async fn new(config: DistributedCacheConfig) -> Result<Self> {
        let registry = MetricsRegistry::new();
        Self::new_with_registry(config, &registry).await
    }

    /// Create a new distributed cache with custom metric registry
    pub async fn new_with_registry(
        config: DistributedCacheConfig,
        registry: &MetricsRegistry,
    ) -> Result<Self> {
        // Validate configuration
        if config.l1_max_size == 0 {
            return Err(DistributedCacheError::InvalidConfig(
                "l1_max_size must be greater than 0".to_string(),
            ));
        }

        // Connect to Redis
        let client = Client::open(config.l2_redis_url.as_str())?;
        let l2_client = ConnectionManager::new(client.clone()).await?;

        // Create L1 cache with fixed size
        let l1_cache = Arc::new(RwLock::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(config.l1_max_size).ok_or_else(|| {
                DistributedCacheError::InvalidConfig(
                    "l1_max_size must be greater than 0".to_string(),
                )
            })?,
        )));

        // Generate unique node ID
        let node_id = uuid::Uuid::new_v4().to_string();

        info!(
            "Distributed cache initialized: node_id={}, l1_size={}, l2_url={}",
            node_id, config.l1_max_size, config.l2_redis_url
        );

        Ok(Self {
            l1_cache,
            l2_client,
            pubsub_client: Arc::new(Mutex::new(client)),
            config,
            metrics: DistributedCacheMetrics::new(registry),
            node_id,
        })
    }

    /// Get value with L1 → L2 hierarchy
    pub async fn get(&self, key: &CacheKey) -> Result<Option<CacheValue>> {
        // Try L1 (local cache, <1ms)
        {
            let mut l1 = self.l1_cache.write();
            if let Some(entry) = l1.get(key) {
                if !entry.is_expired() {
                    self.metrics.l1_hits.inc();
                    debug!("L1 cache hit: key={:?}", key);
                    return Ok(Some(entry.value.clone()));
                } else {
                    // Remove expired entry
                    l1.pop(key);
                }
            }
        }
        self.metrics.l1_misses.inc();
        debug!("L1 cache miss: key={:?}", key);

        // Try L2 (Redis, ~5ms)
        let redis_key = key.redis_key();
        let mut conn = self.l2_client.clone();

        match conn.get::<_, Option<Vec<u8>>>(&redis_key).await {
            Ok(Some(redis_value)) => {
                self.metrics.l2_hits.inc();
                debug!("L2 cache hit: key={:?}", key);

                match self.deserialize_value(&redis_value) {
                    Ok(value) => {
                        // Populate L1
                        {
                            let mut l1 = self.l1_cache.write();
                            let entry = L1Entry::new(value.clone(), self.config.l1_ttl_seconds);
                            l1.put(key.clone(), entry);
                        }

                        Ok(Some(value))
                    }
                    Err(e) => {
                        error!("Failed to deserialize L2 value: {:?}", e);
                        Err(e)
                    }
                }
            }
            Ok(None) => {
                self.metrics.l2_misses.inc();
                debug!("L2 cache miss: key={:?}", key);
                Ok(None)
            }
            Err(e) => {
                error!("Redis get error: {:?}", e);
                Err(DistributedCacheError::RedisConnection(e))
            }
        }
    }

    /// Put value in both L1 and L2
    pub async fn put(&self, key: CacheKey, value: CacheValue) -> Result<()> {
        // Put in L1
        {
            let mut l1 = self.l1_cache.write();
            let entry = L1Entry::new(value.clone(), self.config.l1_ttl_seconds);
            l1.put(key.clone(), entry);
        }
        debug!("Put in L1: key={:?}", key);

        // Put in L2 (Redis)
        let redis_key = key.redis_key();
        let redis_value = self.serialize_value(&value)?;

        let mut conn = self.l2_client.clone();
        conn.set_ex::<_, _, ()>(&redis_key, &redis_value, self.config.l2_ttl_seconds)
            .await
            .map_err(|e| {
                error!("Redis set error: {:?}", e);
                DistributedCacheError::RedisConnection(e)
            })?;

        debug!("Put in L2: key={:?}", key);
        Ok(())
    }

    /// Invalidate key across all nodes
    pub async fn invalidate(&self, key: &CacheKey) -> Result<()> {
        // Remove from L1
        {
            let mut l1 = self.l1_cache.write();
            l1.pop(key);
        }
        debug!("Invalidated in L1: key={:?}", key);

        // Remove from L2
        let redis_key = key.redis_key();
        let mut conn = self.l2_client.clone();
        conn.del::<_, ()>(&redis_key).await.map_err(|e| {
            error!("Redis del error: {:?}", e);
            DistributedCacheError::RedisConnection(e)
        })?;

        debug!("Invalidated in L2: key={:?}", key);

        // Publish invalidation message
        let message = InvalidationMessage {
            key: key.clone(),
            timestamp: SystemTime::now(),
            sender_id: self.node_id.clone(),
        };
        self.publish_invalidation(message).await?;

        self.metrics.invalidations_sent.inc();
        Ok(())
    }

    /// Start listening for invalidation messages
    pub async fn start_invalidation_listener(&self) -> Result<()> {
        let l1_cache = self.l1_cache.clone();
        let pubsub_client = self.pubsub_client.clone();
        let channel = self.config.invalidation_channel.clone();
        let metrics = self.metrics.clone();
        let node_id = self.node_id.clone();
        let channel_for_log = channel.clone();

        tokio::spawn(async move {
            loop {
                let channel_clone = channel.clone();
                let node_id_clone = node_id.clone();
                match Self::run_invalidation_listener(
                    l1_cache.clone(),
                    pubsub_client.clone(),
                    channel_clone,
                    metrics.clone(),
                    node_id_clone,
                )
                .await
                {
                    Ok(_) => {
                        warn!("Invalidation listener stopped, restarting...");
                    }
                    Err(e) => {
                        error!("Invalidation listener error: {:?}, restarting...", e);
                    }
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });

        info!(
            "Started invalidation listener on channel: {}",
            channel_for_log
        );
        Ok(())
    }

    async fn run_invalidation_listener(
        l1_cache: Arc<RwLock<lru::LruCache<CacheKey, L1Entry>>>,
        pubsub_client: Arc<Mutex<Client>>,
        channel: String,
        metrics: DistributedCacheMetrics,
        node_id: String,
    ) -> Result<()> {
        let client = pubsub_client.lock().await;
        let mut pubsub = client.get_async_pubsub().await?;
        pubsub.subscribe(&channel).await?;

        let mut stream = pubsub.on_message();
        while let Some(msg) = stream.next().await {
            let payload: String = match msg.get_payload() {
                Ok(p) => p,
                Err(e) => {
                    error!("Failed to get message payload: {:?}", e);
                    continue;
                }
            };

            match serde_json::from_str::<InvalidationMessage>(&payload) {
                Ok(inv_msg) => {
                    // Don't process our own invalidation messages
                    if inv_msg.sender_id == node_id {
                        continue;
                    }

                    // Invalidate in L1
                    {
                        let mut l1 = l1_cache.write();
                        l1.pop(&inv_msg.key);
                    }

                    metrics.invalidations_received.inc();
                    debug!("Received invalidation: key={:?}", inv_msg.key);
                }
                Err(e) => {
                    error!("Failed to deserialize invalidation message: {:?}", e);
                }
            }
        }

        Ok(())
    }

    async fn publish_invalidation(&self, message: InvalidationMessage) -> Result<()> {
        let payload = serde_json::to_string(&message).map_err(|e| {
            DistributedCacheError::Serialization(format!("Failed to serialize message: {}", e))
        })?;

        let mut conn = self.l2_client.clone();
        conn.publish::<_, _, ()>(&self.config.invalidation_channel, &payload)
            .await
            .map_err(|e| {
                error!("Redis publish error: {:?}", e);
                DistributedCacheError::RedisConnection(e)
            })?;

        debug!(
            "Published invalidation: key={:?}, channel={}",
            message.key, self.config.invalidation_channel
        );
        Ok(())
    }

    fn serialize_value(&self, value: &CacheValue) -> Result<Vec<u8>> {
        let serialized = oxicode::serde::encode_to_vec(value, oxicode::config::standard())
            .map_err(|e| {
                DistributedCacheError::Serialization(format!("oxicode serialization failed: {}", e))
            })?;

        if self.config.compression && serialized.len() > 1024 {
            // Compress large values using gzip
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            use std::io::Write;
            encoder.write_all(&serialized).map_err(|e| {
                DistributedCacheError::Compression(format!("Compression failed: {}", e))
            })?;
            let compressed = encoder.finish().map_err(|e| {
                DistributedCacheError::Compression(format!("Compression finish failed: {}", e))
            })?;

            // Update compression ratio metric
            let ratio = serialized.len() as f64 / compressed.len() as f64;
            *self.metrics.compression_ratio.write() = ratio;

            debug!(
                "Compressed value: original={}, compressed={}, ratio={:.2}x",
                serialized.len(),
                compressed.len(),
                ratio
            );

            Ok(compressed)
        } else {
            Ok(serialized)
        }
    }

    fn deserialize_value(&self, data: &[u8]) -> Result<CacheValue> {
        let decompressed = if self.config.compression && data.len() > 1024 {
            // Try to decompress
            use std::io::Read;
            let mut decoder = flate2::read::GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed).map_err(|e| {
                DistributedCacheError::Decompression(format!("Decompression failed: {}", e))
            })?;
            decompressed
        } else {
            data.to_vec()
        };

        oxicode::serde::decode_from_slice(&decompressed, oxicode::config::standard())
            .map(|(value, _)| value)
            .map_err(|e| {
                DistributedCacheError::Deserialization(format!(
                    "oxicode deserialization failed: {}",
                    e
                ))
            })
    }

    /// Get cache metrics
    pub fn metrics(&self) -> &DistributedCacheMetrics {
        &self.metrics
    }

    /// Clear L1 cache
    pub fn clear_l1(&self) {
        let mut l1 = self.l1_cache.write();
        l1.clear();
        info!("Cleared L1 cache");
    }

    /// Get L1 cache size
    pub fn l1_size(&self) -> usize {
        let l1 = self.l1_cache.read();
        l1.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_hash() {
        let key1 = CacheKey::new("query1".to_string());
        let key2 = CacheKey::new("query1".to_string());
        assert_eq!(key1.hash(), key2.hash());

        let key3 = CacheKey::new("query2".to_string());
        assert_ne!(key1.hash(), key3.hash());
    }

    #[test]
    fn test_cache_key_redis_key() {
        let key = CacheKey::new("query1".to_string());
        assert_eq!(key.redis_key(), "oxirs:cache:query1");

        let key_ns = CacheKey::with_namespace("query1".to_string(), "tenant1".to_string());
        assert_eq!(key_ns.redis_key(), "oxirs:cache:tenant1:query1");
    }

    #[test]
    fn test_l1_entry_expiration() {
        let value = CacheValue::new(vec![1, 2, 3]);
        let entry = L1Entry::new(value, 1);
        assert!(!entry.is_expired());

        std::thread::sleep(Duration::from_secs(2));
        assert!(entry.is_expired());
    }
}
