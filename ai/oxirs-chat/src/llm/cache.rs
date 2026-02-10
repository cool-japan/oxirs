//! Response Cache for LLM Requests
//!
//! Provides LRU caching for LLM responses to reduce API costs and improve latency.

use anyhow::Result;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;

use super::types::{LLMRequest, LLMResponse};

/// Hash key for request caching
type RequestHash = u64;

/// Cached response with metadata
#[derive(Debug, Clone)]
pub struct CachedResponse {
    pub response: LLMResponse,
    pub provider_id: String,
    pub timestamp: SystemTime,
    pub ttl: Duration,
    pub access_count: u32,
}

impl CachedResponse {
    pub fn is_expired(&self) -> bool {
        SystemTime::now()
            .duration_since(self.timestamp)
            .map(|elapsed| elapsed > self.ttl)
            .unwrap_or(true)
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_size: usize,
    pub ttl_seconds: u64,
    pub enable_metrics: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            ttl_seconds: 3600, // 1 hour
            enable_metrics: true,
        }
    }
}

/// Cache metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_requests: u64,
}

impl CacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_requests as f64
        }
    }
}

/// LRU Cache entry for ordering
#[derive(Debug, Clone)]
struct LruEntry {
    cached_response: CachedResponse,
    last_accessed: SystemTime,
}

/// Response cache with LRU eviction
pub struct ResponseCache {
    cache: Arc<RwLock<HashMap<RequestHash, LruEntry>>>,
    config: CacheConfig,
    metrics: Arc<RwLock<CacheMetrics>>,
}

impl ResponseCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    /// Generate hash for request
    fn hash_request(request: &LLMRequest) -> RequestHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash messages
        for msg in &request.messages {
            msg.role.hash(&mut hasher);
            msg.content.hash(&mut hasher);
        }

        // Hash system prompt
        if let Some(ref prompt) = request.system_prompt {
            prompt.hash(&mut hasher);
        }

        // Hash temperature (convert to bits for consistent hashing)
        ((request.temperature * 1000.0) as u64).hash(&mut hasher);

        // Hash max_tokens
        request.max_tokens.hash(&mut hasher);

        hasher.finish()
    }

    /// Get cached response if available
    pub async fn get(&self, request: &LLMRequest) -> Option<LLMResponse> {
        let hash = Self::hash_request(request);
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(&hash) {
            // Check if expired
            if entry.cached_response.is_expired() {
                cache.remove(&hash);
                self.record_miss().await;
                return None;
            }

            // Update access time and count
            entry.last_accessed = SystemTime::now();
            entry.cached_response.access_count += 1;

            self.record_hit().await;
            Some(entry.cached_response.response.clone())
        } else {
            self.record_miss().await;
            None
        }
    }

    /// Store response in cache
    pub async fn put(&self, request: &LLMRequest, response: LLMResponse, provider_id: String) {
        let hash = Self::hash_request(request);
        let mut cache = self.cache.write().await;

        // Check if we need to evict
        if cache.len() >= self.config.max_size && !cache.contains_key(&hash) {
            self.evict_lru(&mut cache).await;
        }

        let cached_response = CachedResponse {
            response,
            provider_id,
            timestamp: SystemTime::now(),
            ttl: Duration::from_secs(self.config.ttl_seconds),
            access_count: 1,
        };

        let entry = LruEntry {
            cached_response,
            last_accessed: SystemTime::now(),
        };

        cache.insert(hash, entry);
    }

    /// Evict least recently used entry
    async fn evict_lru(&self, cache: &mut HashMap<RequestHash, LruEntry>) {
        if cache.is_empty() {
            return;
        }

        // Find LRU entry
        let lru_key = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, _)| *k);

        if let Some(key) = lru_key {
            cache.remove(&key);

            // Record eviction
            if self.config.enable_metrics {
                let mut metrics = self.metrics.write().await;
                metrics.evictions += 1;
            }
        }
    }

    /// Remove expired entries
    pub async fn invalidate_expired(&self) -> Result<usize> {
        let mut cache = self.cache.write().await;
        let now = SystemTime::now();

        let expired_keys: Vec<RequestHash> = cache
            .iter()
            .filter(|(_, entry)| {
                now.duration_since(entry.cached_response.timestamp)
                    .map(|elapsed| elapsed > entry.cached_response.ttl)
                    .unwrap_or(true)
            })
            .map(|(k, _)| *k)
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            cache.remove(&key);
        }

        Ok(count)
    }

    /// Clear all cached entries
    pub async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        Ok(())
    }

    /// Get current cache size
    pub async fn size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }

    /// Get cache metrics
    pub async fn get_metrics(&self) -> CacheMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Calculate hit rate
    pub async fn hit_rate(&self) -> f64 {
        let metrics = self.metrics.read().await;
        metrics.hit_rate()
    }

    async fn record_hit(&self) {
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.hits += 1;
            metrics.total_requests += 1;
        }
    }

    async fn record_miss(&self) {
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().await;
            metrics.misses += 1;
            metrics.total_requests += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatMessage, ChatRole, Priority, Usage, UseCase};

    fn create_test_request(content: &str) -> LLMRequest {
        LLMRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: content.to_string(),
                metadata: None,
            }],
            system_prompt: Some("Test prompt".to_string()),
            temperature: 0.7,
            max_tokens: Some(100),
            use_case: UseCase::Conversation,
            priority: Priority::Normal,
            timeout: None,
        }
    }

    fn create_test_response(content: &str) -> LLMResponse {
        LLMResponse {
            content: content.to_string(),
            model_used: "test-model".to_string(),
            provider_used: "test-provider".to_string(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                cost: 0.001,
            },
            latency: Duration::from_millis(100),
            quality_score: Some(0.9),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let cache = ResponseCache::new(CacheConfig::default());
        let request = create_test_request("test query");
        let response = create_test_response("test response");

        // Store response
        cache
            .put(&request, response.clone(), "test-provider".to_string())
            .await;

        // Retrieve response
        let cached = cache.get(&request).await;
        assert!(cached.is_some());
        assert_eq!(cached.as_ref().unwrap().content, response.content);

        // Check metrics
        let metrics = cache.get_metrics().await;
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 0);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = ResponseCache::new(CacheConfig::default());
        let request = create_test_request("test query");

        // Try to retrieve non-existent response
        let cached = cache.get(&request).await;
        assert!(cached.is_none());

        // Check metrics
        let metrics = cache.get_metrics().await;
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 1);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let mut config = CacheConfig::default();
        config.ttl_seconds = 1; // 1 second TTL

        let cache = ResponseCache::new(config);
        let request = create_test_request("test query");
        let response = create_test_response("test response");

        // Store response
        cache
            .put(&request, response.clone(), "test-provider".to_string())
            .await;

        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Try to retrieve expired response
        let cached = cache.get(&request).await;
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let mut config = CacheConfig::default();
        config.max_size = 2;

        let cache = ResponseCache::new(config);

        // Fill cache
        let req1 = create_test_request("query 1");
        let req2 = create_test_request("query 2");
        let req3 = create_test_request("query 3");

        cache
            .put(&req1, create_test_response("response 1"), "provider1".to_string())
            .await;
        cache
            .put(&req2, create_test_response("response 2"), "provider2".to_string())
            .await;

        // Access req1 to make it more recent
        let _ = cache.get(&req1).await;

        // Add req3, should evict req2 (least recently used)
        cache
            .put(&req3, create_test_response("response 3"), "provider3".to_string())
            .await;

        // req1 and req3 should exist, req2 should be evicted
        assert!(cache.get(&req1).await.is_some());
        assert!(cache.get(&req2).await.is_none());
        assert!(cache.get(&req3).await.is_some());
    }

    #[tokio::test]
    async fn test_hit_rate_calculation() {
        let cache = ResponseCache::new(CacheConfig::default());
        let request = create_test_request("test query");
        let response = create_test_response("test response");

        // Store response
        cache
            .put(&request, response.clone(), "test-provider".to_string())
            .await;

        // 3 hits, 2 misses = 60% hit rate
        let _ = cache.get(&request).await; // hit
        let _ = cache.get(&request).await; // hit
        let _ = cache.get(&request).await; // hit
        let _ = cache.get(&create_test_request("other query")).await; // miss
        let _ = cache.get(&create_test_request("another query")).await; // miss

        let hit_rate = cache.hit_rate().await;
        assert!((hit_rate - 0.6).abs() < 0.01); // 3/5 = 0.6
    }
}
