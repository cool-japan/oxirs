//! Apollo Federation v2 Entity Batch Loader with Caching
//!
//! Provides efficient entity resolution for Apollo Federation by:
//! - Batching entity requests together to minimise round-trips.
//! - Caching resolved entities with TTL and LRU eviction.
//! - Tracking cache statistics per entity type.
//! - Supporting per-tenant entity isolation.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A serialisable entity representation.
///
/// Entities are keyed by `(type_name, key_json)` where `key_json` is the
/// JSON-encoded representation of the entity's `@key` fields.
#[derive(Debug, Clone)]
pub struct ResolvedEntity {
    /// GraphQL type name (e.g. `"Product"`).
    pub type_name: String,
    /// JSON-encoded entity key fields.
    pub key_json: String,
    /// JSON-encoded resolved entity data.
    pub data_json: String,
    /// When this entity was resolved.
    pub resolved_at: Instant,
    /// How long this entity should be cached.
    pub ttl: Duration,
}

impl ResolvedEntity {
    /// Returns `true` if this entity has expired.
    pub fn is_expired(&self) -> bool {
        self.resolved_at.elapsed() >= self.ttl
    }

    /// Remaining time-to-live (saturating at zero).
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl.saturating_sub(self.resolved_at.elapsed())
    }
}

/// Cache key for entity entries.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct EntityCacheKey {
    /// Optional tenant identifier for isolation.
    pub tenant_id: Option<String>,
    /// The GraphQL type name.
    pub type_name: String,
    /// JSON-encoded entity key (normalised / sorted).
    pub key_json: String,
}

impl EntityCacheKey {
    /// Create a new entity cache key.
    pub fn new(
        tenant_id: Option<&str>,
        type_name: impl Into<String>,
        key_json: impl Into<String>,
    ) -> Self {
        Self {
            tenant_id: tenant_id.map(|s| s.to_string()),
            type_name: type_name.into(),
            key_json: key_json.into(),
        }
    }
}

/// Inner mutable cache store.
struct EntityCacheStore {
    entries: HashMap<EntityCacheKey, ResolvedEntity>,
    lru: VecDeque<EntityCacheKey>,
    max_entries: usize,
    /// Per-type stats: (hits, misses).
    type_stats: HashMap<String, (u64, u64)>,
}

impl EntityCacheStore {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            max_entries,
            type_stats: HashMap::new(),
        }
    }

    fn insert(&mut self, key: EntityCacheKey, entity: ResolvedEntity) {
        // Remove existing slot from LRU order
        if self.entries.contains_key(&key) {
            self.lru.retain(|k| k != &key);
        }

        self.entries.insert(key.clone(), entity);
        self.lru.push_back(key);

        // Evict LRU if over capacity
        while self.entries.len() > self.max_entries {
            if let Some(oldest) = self.lru.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }

    fn get_mut(&mut self, key: &EntityCacheKey) -> Option<&mut ResolvedEntity> {
        self.entries.get_mut(key)
    }

    fn touch(&mut self, key: &EntityCacheKey) {
        self.lru.retain(|k| k != key);
        self.lru.push_back(key.clone());
    }

    fn remove(&mut self, key: &EntityCacheKey) {
        self.entries.remove(key);
        self.lru.retain(|k| k != key);
    }

    fn evict_expired(&mut self) -> usize {
        let expired: Vec<EntityCacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired.len();
        for key in expired {
            self.remove(&key);
        }
        count
    }

    fn record_hit(&mut self, type_name: &str) {
        let entry = self
            .type_stats
            .entry(type_name.to_string())
            .or_insert((0, 0));
        entry.0 += 1;
    }

    fn record_miss(&mut self, type_name: &str) {
        let entry = self
            .type_stats
            .entry(type_name.to_string())
            .or_insert((0, 0));
        entry.1 += 1;
    }
}

/// Statistics for the entity cache.
#[derive(Debug, Clone)]
pub struct EntityCacheStats {
    /// Total cache hits.
    pub total_hits: u64,
    /// Total cache misses.
    pub total_misses: u64,
    /// Total LRU + TTL evictions.
    pub total_evictions: u64,
    /// Current number of cached entities.
    pub current_size: usize,
    /// Per-type hit/miss counts.
    pub by_type: HashMap<String, (u64, u64)>,
}

impl EntityCacheStats {
    /// Returns the overall hit rate `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = (self.total_hits + self.total_misses) as f64;
        if total == 0.0 {
            0.0
        } else {
            self.total_hits as f64 / total
        }
    }
}

/// Thread-safe entity cache with LRU eviction and TTL.
pub struct EntityCache {
    store: Arc<Mutex<EntityCacheStore>>,
    default_ttl: Duration,
    total_hits: Arc<AtomicU64>,
    total_misses: Arc<AtomicU64>,
    total_evictions: Arc<AtomicU64>,
}

impl std::fmt::Debug for EntityCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntityCache")
            .field("default_ttl", &self.default_ttl)
            .field("hits", &self.total_hits.load(Ordering::Relaxed))
            .field("misses", &self.total_misses.load(Ordering::Relaxed))
            .finish()
    }
}

impl EntityCache {
    /// Create a new entity cache.
    pub fn new(max_entries: usize, default_ttl: Duration) -> Self {
        Self {
            store: Arc::new(Mutex::new(EntityCacheStore::new(max_entries))),
            default_ttl,
            total_hits: Arc::new(AtomicU64::new(0)),
            total_misses: Arc::new(AtomicU64::new(0)),
            total_evictions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Look up a cached entity.
    ///
    /// Returns `Some(data_json)` on a hit, or `None` on a miss / expired entry.
    pub fn get(&self, key: &EntityCacheKey) -> Option<String> {
        let mut store = self.store.lock().unwrap_or_else(|p| p.into_inner());

        // Check if entity exists and gather needed data before modifying the store.
        // We clone the required fields so the mutable borrow is released before
        // we call other &mut methods on `store`.
        let entity_info: Option<(bool, String, String)> = store.get_mut(key).map(|entity| {
            (
                entity.is_expired(),
                entity.type_name.clone(),
                entity.data_json.clone(),
            )
        });

        match entity_info {
            Some((true, ref type_name, _)) => {
                // Entity is expired: remove it
                store.remove(key);
                store.record_miss(type_name);
                self.total_misses.fetch_add(1, Ordering::Relaxed);
                self.total_evictions.fetch_add(1, Ordering::Relaxed);
                None
            }
            Some((false, ref type_name, ref data)) => {
                // Entity is valid: record hit and return data
                let data_clone = data.clone();
                store.touch(key);
                store.record_hit(type_name);
                self.total_hits.fetch_add(1, Ordering::Relaxed);
                Some(data_clone)
            }
            None => {
                let type_name = key.type_name.clone();
                store.record_miss(&type_name);
                self.total_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store a resolved entity with the default TTL.
    pub fn put(&self, key: EntityCacheKey, entity: ResolvedEntity) {
        self.put_with_ttl(key, entity, self.default_ttl);
    }

    /// Store a resolved entity with an explicit TTL.
    pub fn put_with_ttl(&self, key: EntityCacheKey, mut entity: ResolvedEntity, ttl: Duration) {
        entity.ttl = ttl;
        entity.resolved_at = Instant::now();
        if let Ok(mut store) = self.store.lock() {
            store.insert(key, entity);
        }
    }

    /// Remove expired entries.
    pub fn evict_expired(&self) -> usize {
        let count = self
            .store
            .lock()
            .map(|mut s| s.evict_expired())
            .unwrap_or(0);
        self.total_evictions
            .fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Clear all entries.
    pub fn clear(&self) -> usize {
        if let Ok(mut store) = self.store.lock() {
            let count = store.entries.len();
            store.entries.clear();
            store.lru.clear();
            count
        } else {
            0
        }
    }

    /// Current number of entries.
    pub fn size(&self) -> usize {
        self.store.lock().map(|s| s.entries.len()).unwrap_or(0)
    }

    /// Return cache statistics.
    pub fn stats(&self) -> EntityCacheStats {
        let by_type = self
            .store
            .lock()
            .map(|s| s.type_stats.clone())
            .unwrap_or_default();

        EntityCacheStats {
            total_hits: self.total_hits.load(Ordering::Relaxed),
            total_misses: self.total_misses.load(Ordering::Relaxed),
            total_evictions: self.total_evictions.load(Ordering::Relaxed),
            current_size: self.size(),
            by_type,
        }
    }

    /// Hit rate `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.total_hits.load(Ordering::Relaxed) as f64;
        let misses = self.total_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }
}

/// A pending batch of entity resolution requests.
#[derive(Debug)]
pub struct EntityBatch {
    /// Type name for this batch.
    pub type_name: String,
    /// Entity key JSONs to resolve.
    pub keys: Vec<String>,
    /// Optional tenant ID.
    pub tenant_id: Option<String>,
}

impl EntityBatch {
    /// Create a new batch for a given entity type.
    pub fn new(type_name: impl Into<String>, tenant_id: Option<&str>) -> Self {
        Self {
            type_name: type_name.into(),
            keys: Vec::new(),
            tenant_id: tenant_id.map(|s| s.to_string()),
        }
    }

    /// Add a key JSON to this batch.
    pub fn add_key(&mut self, key_json: impl Into<String>) {
        self.keys.push(key_json.into());
    }

    /// Number of keys in this batch.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether this batch is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Resolver function type for resolving a batch of entities.
///
/// Receives the batch and should return a Vec of `(key_json, data_json)` pairs.
type BatchResolverFn = Box<
    dyn Fn(EntityBatch) -> Pin<Box<dyn Future<Output = Result<Vec<(String, String)>>> + Send>>
        + Send
        + Sync,
>;

/// Entity batch loader that integrates with `EntityCache`.
///
/// Collects entity requests, checks the cache first, and issues batch calls
/// to the underlying subgraph for cache misses.
pub struct EntityBatchLoader {
    cache: Arc<EntityCache>,
    resolver: Arc<BatchResolverFn>,
    /// Maximum number of keys per batch call.
    max_batch_size: usize,
    default_ttl: Duration,
}

impl std::fmt::Debug for EntityBatchLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntityBatchLoader")
            .field("max_batch_size", &self.max_batch_size)
            .field("default_ttl", &self.default_ttl)
            .finish()
    }
}

impl EntityBatchLoader {
    /// Create a new batch loader.
    pub fn new(
        cache: Arc<EntityCache>,
        resolver: impl Fn(EntityBatch) -> Pin<Box<dyn Future<Output = Result<Vec<(String, String)>>> + Send>>
            + Send
            + Sync
            + 'static,
        max_batch_size: usize,
        default_ttl: Duration,
    ) -> Self {
        Self {
            cache,
            resolver: Arc::new(Box::new(resolver)),
            max_batch_size,
            default_ttl,
        }
    }

    /// Resolve a list of entity keys for a given type, using the cache.
    ///
    /// Returns a map from key JSON to resolved data JSON.
    pub async fn load_many(
        &self,
        type_name: &str,
        key_jsons: Vec<String>,
        tenant_id: Option<&str>,
    ) -> Result<HashMap<String, String>> {
        let mut result: HashMap<String, String> = HashMap::new();
        let mut cache_misses: Vec<String> = Vec::new();

        // Check cache first
        for key_json in &key_jsons {
            let cache_key = EntityCacheKey::new(tenant_id, type_name, key_json.as_str());
            if let Some(data) = self.cache.get(&cache_key) {
                result.insert(key_json.clone(), data);
            } else {
                cache_misses.push(key_json.clone());
            }
        }

        // Resolve misses in batches
        for chunk in cache_misses.chunks(self.max_batch_size) {
            let mut batch = EntityBatch::new(type_name, tenant_id);
            for key in chunk {
                batch.add_key(key.clone());
            }

            let resolved = (self.resolver)(batch).await?;

            for (key_json, data_json) in resolved {
                let cache_key = EntityCacheKey::new(tenant_id, type_name, key_json.as_str());
                let entity = ResolvedEntity {
                    type_name: type_name.to_string(),
                    key_json: key_json.clone(),
                    data_json: data_json.clone(),
                    resolved_at: Instant::now(),
                    ttl: self.default_ttl,
                };
                self.cache.put(cache_key, entity);
                result.insert(key_json, data_json);
            }
        }

        Ok(result)
    }

    /// Load a single entity.
    pub async fn load_one(
        &self,
        type_name: &str,
        key_json: &str,
        tenant_id: Option<&str>,
    ) -> Result<Option<String>> {
        let mut results = self
            .load_many(type_name, vec![key_json.to_string()], tenant_id)
            .await?;
        Ok(results.remove(key_json))
    }

    /// Return a reference to the underlying entity cache.
    pub fn cache(&self) -> &Arc<EntityCache> {
        &self.cache
    }

    /// Invalidate all cached entities of a given type.
    pub fn invalidate_type(&self, type_name: &str) -> usize {
        // We use the full clear as a conservative strategy since we can't
        // efficiently filter by type without an extra index. In production this
        // would be extended with a type index.
        let mut to_remove: Vec<EntityCacheKey> = Vec::new();
        if let Ok(store) = self.cache.store.lock() {
            for key in store.entries.keys() {
                if key.type_name == type_name {
                    to_remove.push(key.clone());
                }
            }
        }
        let count = to_remove.len();
        if let Ok(mut store) = self.cache.store.lock() {
            for key in &to_remove {
                store.remove(key);
            }
        }
        count
    }
}

/// Create a simple mock batch resolver for testing.
///
/// The mock resolver returns `{"id": "<key>"}` for every requested key.
#[allow(clippy::type_complexity)]
pub fn mock_batch_resolver(
) -> impl Fn(EntityBatch) -> Pin<Box<dyn Future<Output = Result<Vec<(String, String)>>> + Send>>
       + Send
       + Sync
       + 'static {
    |batch: EntityBatch| {
        Box::pin(async move {
            let results: Vec<(String, String)> = batch
                .keys
                .into_iter()
                .map(|k| {
                    let data = format!(r#"{{"id": {k}, "type": "{}"}}"#, batch.type_name);
                    (k, data)
                })
                .collect();
            Ok(results)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    fn make_cache() -> Arc<EntityCache> {
        Arc::new(EntityCache::new(100, Duration::from_secs(60)))
    }

    fn entity(type_name: &str, key: &str, data: &str) -> ResolvedEntity {
        ResolvedEntity {
            type_name: type_name.to_string(),
            key_json: key.to_string(),
            data_json: data.to_string(),
            resolved_at: Instant::now(),
            ttl: Duration::from_secs(60),
        }
    }

    fn ekey(type_name: &str, key: &str) -> EntityCacheKey {
        EntityCacheKey::new(None, type_name, key)
    }

    fn ekey_tenant(tenant: &str, type_name: &str, key: &str) -> EntityCacheKey {
        EntityCacheKey::new(Some(tenant), type_name, key)
    }

    // ---- EntityCacheKey tests -----------------------------------------------

    #[test]
    fn test_entity_cache_key_equality() {
        let k1 = ekey("Product", r#"{"id":1}"#);
        let k2 = ekey("Product", r#"{"id":1}"#);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_entity_cache_key_different_types() {
        let k1 = ekey("Product", r#"{"id":1}"#);
        let k2 = ekey("User", r#"{"id":1}"#);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_entity_cache_key_different_tenants() {
        let k1 = ekey_tenant("acme", "Product", r#"{"id":1}"#);
        let k2 = ekey_tenant("corp", "Product", r#"{"id":1}"#);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_entity_cache_key_none_vs_some_tenant() {
        let k1 = ekey("Product", r#"{"id":1}"#);
        let k2 = ekey_tenant("acme", "Product", r#"{"id":1}"#);
        assert_ne!(k1, k2);
    }

    // ---- ResolvedEntity tests -----------------------------------------------

    #[test]
    fn test_resolved_entity_not_expired_immediately() {
        let e = entity("Product", r#"{"id":1}"#, r#"{"name":"Foo"}"#);
        assert!(!e.is_expired());
    }

    #[test]
    fn test_resolved_entity_expired_after_ttl() {
        let mut e = entity("Product", r#"{"id":1}"#, r#"{"name":"Foo"}"#);
        e.ttl = Duration::from_nanos(1);
        e.resolved_at = Instant::now();
        std::thread::sleep(Duration::from_millis(5));
        assert!(e.is_expired());
    }

    #[test]
    fn test_resolved_entity_remaining_ttl() {
        let e = entity("Product", r#"{"id":1}"#, r#"{"name":"Foo"}"#);
        assert!(e.remaining_ttl() > Duration::ZERO);
    }

    // ---- EntityCache tests --------------------------------------------------

    #[test]
    fn test_entity_cache_put_and_get() {
        let cache = make_cache();
        let key = ekey("Product", r#"{"id":1}"#);
        let e = entity("Product", r#"{"id":1}"#, r#"{"name":"Widget"}"#);
        cache.put(key.clone(), e);

        let result = cache.get(&key);
        assert_eq!(result.as_deref(), Some(r#"{"name":"Widget"}"#));
    }

    #[test]
    fn test_entity_cache_miss_returns_none() {
        let cache = make_cache();
        let key = ekey("Product", r#"{"id":999}"#);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_entity_cache_expired_entry_removed_on_get() {
        let cache = make_cache();
        let key = ekey("Product", r#"{"id":1}"#);
        let mut e = entity("Product", r#"{"id":1}"#, r#"{"name":"Old"}"#);
        e.ttl = Duration::from_nanos(1);
        cache.put_with_ttl(key.clone(), e, Duration::from_nanos(1));

        std::thread::sleep(Duration::from_millis(5));

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_entity_cache_lru_eviction() {
        let cache = Arc::new(EntityCache::new(2, Duration::from_secs(60)));

        let k1 = ekey("T", "k1");
        let k2 = ekey("T", "k2");
        let k3 = ekey("T", "k3");

        cache.put(k1.clone(), entity("T", "k1", "d1"));
        cache.put(k2.clone(), entity("T", "k2", "d2"));
        // Touch k1 to make k2 the LRU
        cache.get(&k1);
        // Insert k3 — should evict k2
        cache.put(k3.clone(), entity("T", "k3", "d3"));

        assert_eq!(cache.size(), 2);
        assert!(cache.get(&k2).is_none(), "k2 should have been evicted");
        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn test_entity_cache_evict_expired() {
        let cache = Arc::new(EntityCache::new(100, Duration::from_nanos(1)));
        let key = ekey("Product", "exp");
        cache.put_with_ttl(
            key,
            entity("Product", "exp", "old"),
            Duration::from_nanos(1),
        );
        std::thread::sleep(Duration::from_millis(5));

        let removed = cache.evict_expired();
        assert_eq!(removed, 1);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_entity_cache_clear() {
        let cache = make_cache();
        cache.put(ekey("T", "k1"), entity("T", "k1", "d1"));
        cache.put(ekey("T", "k2"), entity("T", "k2", "d2"));

        let removed = cache.clear();
        assert_eq!(removed, 2);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_entity_cache_stats() {
        let cache = make_cache();
        let key = ekey("Product", "k1");
        cache.put(key.clone(), entity("Product", "k1", "d1"));
        cache.get(&key); // hit
        cache.get(&ekey("Product", "miss")); // miss

        let stats = cache.stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.current_size, 1);
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entity_cache_per_type_stats() {
        let cache = make_cache();
        let key = ekey("Product", "k1");
        cache.put(key.clone(), entity("Product", "k1", "d1"));
        cache.get(&key); // hit for Product
        cache.get(&ekey("User", "u1")); // miss for User

        let stats = cache.stats();
        let product_stats = stats.by_type.get("Product");
        assert!(product_stats.is_some());
        // Hit should be recorded for Product
        assert_eq!(product_stats.map(|(h, _)| *h).unwrap_or(0), 1);
    }

    #[test]
    fn test_entity_cache_tenant_isolation() {
        let cache = make_cache();
        let k1 = ekey_tenant("acme", "Product", r#"{"id":1}"#);
        let k2 = ekey_tenant("corp", "Product", r#"{"id":1}"#);

        cache.put(
            k1.clone(),
            entity("Product", r#"{"id":1}"#, r#"{"name":"ACME Widget"}"#),
        );
        cache.put(
            k2.clone(),
            entity("Product", r#"{"id":1}"#, r#"{"name":"Corp Widget"}"#),
        );

        assert_ne!(cache.get(&k1), cache.get(&k2));
        assert_eq!(cache.size(), 2);
    }

    // ---- EntityBatch tests --------------------------------------------------

    #[test]
    fn test_entity_batch_add_key() {
        let mut batch = EntityBatch::new("Product", None);
        assert!(batch.is_empty());

        batch.add_key(r#"{"id":1}"#);
        batch.add_key(r#"{"id":2}"#);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_entity_batch_tenant() {
        let batch = EntityBatch::new("Product", Some("acme"));
        assert_eq!(batch.tenant_id.as_deref(), Some("acme"));
    }

    // ---- EntityBatchLoader tests --------------------------------------------

    #[tokio::test]
    async fn test_batch_loader_load_one_hit() {
        let cache = make_cache();
        let loader = EntityBatchLoader::new(
            Arc::clone(&cache),
            mock_batch_resolver(),
            50,
            Duration::from_secs(60),
        );

        let result = loader.load_one("Product", r#"{"id":1}"#, None).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_batch_loader_caches_result() {
        let cache = make_cache();
        let loader = EntityBatchLoader::new(
            Arc::clone(&cache),
            mock_batch_resolver(),
            50,
            Duration::from_secs(60),
        );

        // First call resolves and caches
        let _ = loader.load_one("Product", r#"{"id":42}"#, None).await;
        assert_eq!(cache.size(), 1);

        // Second call should be a cache hit (no additional resolver call needed)
        let result = loader.load_one("Product", r#"{"id":42}"#, None).await;
        assert!(result.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_batch_loader_load_many() {
        let cache = make_cache();
        let loader = EntityBatchLoader::new(
            Arc::clone(&cache),
            mock_batch_resolver(),
            50,
            Duration::from_secs(60),
        );

        let keys = vec![
            r#"{"id":1}"#.to_string(),
            r#"{"id":2}"#.to_string(),
            r#"{"id":3}"#.to_string(),
        ];

        let results = loader.load_many("Product", keys.clone(), None).await;
        assert!(results.is_ok());
        let map = results.unwrap();
        assert_eq!(map.len(), 3);
        for key in &keys {
            assert!(map.contains_key(key.as_str()));
        }
    }

    #[tokio::test]
    async fn test_batch_loader_invalidate_type() {
        let cache = make_cache();
        let loader = EntityBatchLoader::new(
            Arc::clone(&cache),
            mock_batch_resolver(),
            50,
            Duration::from_secs(60),
        );

        let _ = loader.load_one("Product", r#"{"id":1}"#, None).await;
        let _ = loader.load_one("Product", r#"{"id":2}"#, None).await;
        let _ = loader.load_one("User", r#"{"id":1}"#, None).await;

        assert_eq!(cache.size(), 3);

        let removed = loader.invalidate_type("Product");
        assert_eq!(removed, 2);
        assert_eq!(cache.size(), 1);
    }

    #[tokio::test]
    async fn test_batch_loader_batches_large_requests() {
        // Verify that chunking works correctly for large key lists
        let cache = make_cache();
        let loader = EntityBatchLoader::new(
            Arc::clone(&cache),
            mock_batch_resolver(),
            2, // small batch size to test chunking
            Duration::from_secs(60),
        );

        let keys: Vec<String> = (1..=5).map(|i| format!(r#"{{"id":{i}}}"#)).collect();
        let results = loader.load_many("Product", keys, None).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 5);
    }

    #[test]
    fn test_entity_cache_size_after_multiple_puts() {
        let cache = make_cache();
        for i in 0..10 {
            cache.put(
                ekey("T", &format!("k{i}")),
                entity("T", &format!("k{i}"), &format!("d{i}")),
            );
        }
        assert_eq!(cache.size(), 10);
    }

    #[test]
    fn test_entity_cache_hit_rate_zero_when_no_requests() {
        let cache = make_cache();
        assert_eq!(cache.hit_rate(), 0.0);
    }
}
