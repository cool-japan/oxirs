//! LRU embedding cache with memory-bounded eviction and per-model invalidation.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// Cache key: content hash + model identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Hash of the input content (e.g. FNV-1a or SipHash).
    pub content_hash: u64,
    /// Identifier of the embedding model used.
    pub model_id: String,
}

impl CacheKey {
    /// Create a cache key.
    pub fn new(content_hash: u64, model_id: impl Into<String>) -> Self {
        Self {
            content_hash,
            model_id: model_id.into(),
        }
    }
}

/// A single cached embedding entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Number of times this entry has been accessed (cache hits).
    pub access_count: u64,
    /// Approximate size in bytes (4 × dimensions).
    pub size_bytes: usize,
}

impl CacheEntry {
    fn new(embedding: Vec<f32>) -> Self {
        let size_bytes = embedding.len() * std::mem::size_of::<f32>();
        Self {
            embedding,
            access_count: 0,
            size_bytes,
        }
    }
}

/// Cache statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total successful lookups.
    pub hits: u64,
    /// Total unsuccessful lookups.
    pub misses: u64,
    /// Total entries evicted due to capacity or memory limits.
    pub evictions: u64,
    /// `hits / (hits + misses)`, or 0.0 when no lookups have occurred.
    pub hit_rate: f64,
    /// Sum of `size_bytes` across all entries currently held.
    pub total_size_bytes: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal LRU node list
// ──────────────────────────────────────────────────────────────────────────────

/// An intrusive doubly-linked list node.
struct LruNode {
    key: CacheKey,
    entry: CacheEntry,
    prev: Option<usize>, // index into slab
    next: Option<usize>,
}

/// Simple slab-allocated doubly-linked list for LRU tracking.
///
/// `head` is the most-recently-used end; `tail` is the least-recently-used.
struct LruList {
    nodes: Vec<Option<LruNode>>,
    free: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl LruList {
    fn new(capacity: usize) -> Self {
        Self {
            nodes: (0..capacity).map(|_| None).collect(),
            free: (0..capacity).rev().collect(),
            head: None,
            tail: None,
        }
    }

    fn allocate(&mut self, key: CacheKey, entry: CacheEntry) -> Option<usize> {
        let idx = self.free.pop()?;
        self.nodes[idx] = Some(LruNode {
            key,
            entry,
            prev: None,
            next: None,
        });
        Some(idx)
    }

    fn reclaim(&mut self, idx: usize) {
        self.nodes[idx] = None;
        self.free.push(idx);
    }

    /// Move `idx` to the head (most-recently-used position).
    fn move_to_head(&mut self, idx: usize) {
        self.detach(idx);
        self.attach_head(idx);
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let node = self.nodes[idx].as_ref().expect("node must exist");
            (node.prev, node.next)
        };
        if let Some(p) = prev {
            self.nodes[p].as_mut().expect("prev must exist").next = next;
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            self.nodes[n].as_mut().expect("next must exist").prev = prev;
        } else {
            self.tail = prev;
        }
        let node = self.nodes[idx].as_mut().expect("node must exist");
        node.prev = None;
        node.next = None;
    }

    fn attach_head(&mut self, idx: usize) {
        let node = self.nodes[idx].as_mut().expect("node must exist");
        node.next = self.head;
        node.prev = None;
        let old_head = self.head;
        self.head = Some(idx);
        if let Some(h) = old_head {
            self.nodes[h].as_mut().expect("old head must exist").prev = Some(idx);
        } else {
            self.tail = Some(idx);
        }
    }

    /// Remove and return the LRU (tail) node's key and entry.
    fn evict_lru(&mut self) -> Option<(CacheKey, CacheEntry)> {
        let tail_idx = self.tail?;
        self.detach(tail_idx);
        let node = self.nodes[tail_idx].take().expect("tail node must exist");
        self.free.push(tail_idx);
        Some((node.key, node.entry))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// EmbeddingCache
// ──────────────────────────────────────────────────────────────────────────────

/// LRU embedding cache with a fixed entry-count capacity.
pub struct EmbeddingCache {
    capacity: usize,
    map: HashMap<CacheKey, usize>, // key → slab index
    list: LruList,
    hits: u64,
    misses: u64,
    evictions: u64,
    total_size_bytes: usize,
}

impl EmbeddingCache {
    /// Create a cache that holds at most `capacity` embeddings.
    ///
    /// Capacity is clamped to at least 1.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            list: LruList::new(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
            total_size_bytes: 0,
        }
    }

    /// Look up an embedding. Promotes the entry to MRU on hit.
    pub fn get(&mut self, key: &CacheKey) -> Option<&[f32]> {
        if let Some(&idx) = self.map.get(key) {
            self.list.move_to_head(idx);
            self.hits += 1;
            let node = self.list.nodes[idx].as_mut().expect("node must exist");
            node.entry.access_count += 1;
            // Safety: we return a reference into the node which lives inside the
            // Vec<Option<LruNode>>. We need to go through the map again to avoid
            // holding `&mut self`.
            let idx2 = *self.map.get(key).expect("just inserted");
            let node2 = self.list.nodes[idx2].as_ref().expect("node must exist");
            return Some(&node2.entry.embedding);
        }
        self.misses += 1;
        None
    }

    /// Insert an embedding. Evicts the LRU entry when at capacity.
    pub fn insert(&mut self, key: CacheKey, embedding: Vec<f32>) {
        // If already present, update in place.
        if let Some(&idx) = self.map.get(&key) {
            let node = self.list.nodes[idx].as_mut().expect("node must exist");
            let old_size = node.entry.size_bytes;
            node.entry = CacheEntry::new(embedding);
            let new_size = node.entry.size_bytes;
            self.total_size_bytes = self.total_size_bytes - old_size + new_size;
            self.list.move_to_head(idx);
            return;
        }

        // Evict LRU if full.
        if self.map.len() >= self.capacity {
            if let Some((evicted_key, evicted_entry)) = self.list.evict_lru() {
                self.total_size_bytes -= evicted_entry.size_bytes;
                self.map.remove(&evicted_key);
                self.evictions += 1;
            }
        }

        let entry = CacheEntry::new(embedding);
        self.total_size_bytes += entry.size_bytes;

        if let Some(idx) = self.list.allocate(key.clone(), entry) {
            self.list.attach_head(idx);
            self.map.insert(key, idx);
        }
    }

    /// Manually evict the LRU entry.
    pub fn evict_lru(&mut self) -> Option<(CacheKey, CacheEntry)> {
        let (k, e) = self.list.evict_lru()?;
        self.map.remove(&k);
        self.total_size_bytes -= e.size_bytes;
        self.evictions += 1;
        Some((k, e))
    }

    /// Remove a specific entry. Returns `true` if it was present.
    pub fn invalidate(&mut self, key: &CacheKey) -> bool {
        if let Some(idx) = self.map.remove(key) {
            self.list.detach(idx);
            let node = self.list.nodes[idx].take().expect("node must exist");
            self.list.free.push(idx);
            self.total_size_bytes -= node.entry.size_bytes;
            true
        } else {
            false
        }
    }

    /// Remove all entries for a given model.
    pub fn invalidate_model(&mut self, model_id: &str) -> usize {
        let keys_to_remove: Vec<CacheKey> = self
            .map
            .keys()
            .filter(|k| k.model_id == model_id)
            .cloned()
            .collect();
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.invalidate(&key);
        }
        count
    }

    /// Current statistics snapshot.
    pub fn stats(&self) -> CacheStats {
        let total = self.hits + self.misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        };
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate,
            total_size_bytes: self.total_size_bytes,
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Maximum entry capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MemoryBoundedCache
// ──────────────────────────────────────────────────────────────────────────────

/// An `EmbeddingCache` variant that evicts entries when total memory exceeds a
/// byte limit.
pub struct MemoryBoundedCache {
    inner: EmbeddingCache,
    max_bytes: usize,
}

impl MemoryBoundedCache {
    /// Create a memory-bounded cache with a maximum of `max_bytes`.
    ///
    /// The internal entry capacity is set to a generous upper bound so that the
    /// byte limit is always the binding constraint.
    pub fn new(max_bytes: usize) -> Self {
        // Assume max 4 bytes/dim × 1024 dims → 4 KiB per entry.
        // Use a large entry-count capacity so byte limit governs.
        let capacity = (max_bytes / (4 * 128)).max(4);
        Self {
            inner: EmbeddingCache::new(capacity),
            max_bytes,
        }
    }

    /// Insert an embedding, evicting LRU entries until within the memory limit.
    pub fn insert(&mut self, key: CacheKey, embedding: Vec<f32>) {
        self.inner.insert(key, embedding);
        // Evict until within bounds.
        while self.inner.total_size_bytes > self.max_bytes {
            if self.inner.evict_lru().is_none() {
                break;
            }
        }
    }

    /// Delegate to inner cache.
    pub fn get(&mut self, key: &CacheKey) -> Option<&[f32]> {
        self.inner.get(key)
    }

    /// Current byte usage.
    pub fn total_size_bytes(&self) -> usize {
        self.inner.total_size_bytes
    }

    /// Maximum byte limit.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Statistics.
    pub fn stats(&self) -> CacheStats {
        self.inner.stats()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn key(hash: u64, model: &str) -> CacheKey {
        CacheKey::new(hash, model)
    }

    fn emb(dims: usize) -> Vec<f32> {
        (0..dims).map(|i| i as f32 * 0.1).collect()
    }

    // ── CacheKey ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_key_equality() {
        assert_eq!(key(1, "model-a"), key(1, "model-a"));
        assert_ne!(key(1, "model-a"), key(1, "model-b"));
        assert_ne!(key(1, "model-a"), key(2, "model-a"));
    }

    #[test]
    fn test_cache_key_clone() {
        let k = key(42, "bert");
        let k2 = k.clone();
        assert_eq!(k, k2);
    }

    // ── CacheEntry ────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_entry_size_bytes() {
        let e = CacheEntry::new(vec![0.0f32; 128]);
        assert_eq!(e.size_bytes, 128 * 4);
    }

    #[test]
    fn test_cache_entry_access_count_starts_zero() {
        let e = CacheEntry::new(vec![1.0; 4]);
        assert_eq!(e.access_count, 0);
    }

    // ── EmbeddingCache basic ──────────────────────────────────────────────────

    #[test]
    fn test_cache_empty_initially() {
        let c = EmbeddingCache::new(10);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_cache_capacity() {
        let c = EmbeddingCache::new(5);
        assert_eq!(c.capacity(), 5);
    }

    #[test]
    fn test_cache_capacity_min_one() {
        let c = EmbeddingCache::new(0);
        assert_eq!(c.capacity(), 1);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut c = EmbeddingCache::new(10);
        let k = key(1, "m");
        let e = emb(4);
        c.insert(k.clone(), e.clone());
        let got = c.get(&k).expect("should be cached");
        assert_eq!(got, e.as_slice());
    }

    #[test]
    fn test_cache_miss() {
        let mut c = EmbeddingCache::new(10);
        assert!(c.get(&key(99, "m")).is_none());
    }

    #[test]
    fn test_cache_len_after_insert() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), emb(4));
        c.insert(key(2, "m"), emb(4));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn test_cache_stats_hits_misses() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), emb(4));
        c.get(&key(1, "m")); // hit
        c.get(&key(99, "m")); // miss
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cache_stats_no_lookups_hit_rate_zero() {
        let c = EmbeddingCache::new(10);
        assert_eq!(c.stats().hit_rate, 0.0);
    }

    #[test]
    fn test_cache_stats_total_size_bytes() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), vec![0.0f32; 64]);
        c.insert(key(2, "m"), vec![0.0f32; 128]);
        assert_eq!(c.stats().total_size_bytes, (64 + 128) * 4);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    #[test]
    fn test_cache_evicts_lru_when_full() {
        let mut c = EmbeddingCache::new(3);
        c.insert(key(1, "m"), emb(4));
        c.insert(key(2, "m"), emb(4));
        c.insert(key(3, "m"), emb(4));
        // key(1) is LRU; inserting key(4) should evict it
        c.insert(key(4, "m"), emb(4));
        assert!(c.get(&key(1, "m")).is_none(), "key(1) should be evicted");
        assert!(c.get(&key(4, "m")).is_some());
    }

    #[test]
    fn test_cache_get_promotes_to_mru() {
        let mut c = EmbeddingCache::new(3);
        c.insert(key(1, "m"), emb(4));
        c.insert(key(2, "m"), emb(4));
        c.insert(key(3, "m"), emb(4));
        // Promote key(1) so key(2) becomes LRU
        c.get(&key(1, "m"));
        c.insert(key(4, "m"), emb(4));
        assert!(c.get(&key(2, "m")).is_none(), "key(2) should be evicted");
        assert!(c.get(&key(1, "m")).is_some());
    }

    #[test]
    fn test_cache_evictions_stat_incremented() {
        let mut c = EmbeddingCache::new(2);
        c.insert(key(1, "m"), emb(4));
        c.insert(key(2, "m"), emb(4));
        c.insert(key(3, "m"), emb(4)); // evicts key(1)
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn test_cache_manual_evict_lru() {
        let mut c = EmbeddingCache::new(3);
        c.insert(key(1, "m"), emb(4));
        c.insert(key(2, "m"), emb(4));
        let (evicted_key, _) = c.evict_lru().expect("should evict");
        // The LRU is key(1) since key(2) was inserted last.
        assert_eq!(evicted_key, key(1, "m"));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_cache_manual_evict_lru_empty() {
        let mut c = EmbeddingCache::new(3);
        assert!(c.evict_lru().is_none());
    }

    // ── Invalidation ─────────────────────────────────────────────────────────

    #[test]
    fn test_cache_invalidate_present() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), emb(4));
        assert!(c.invalidate(&key(1, "m")));
        assert!(c.is_empty());
    }

    #[test]
    fn test_cache_invalidate_absent() {
        let mut c = EmbeddingCache::new(10);
        assert!(!c.invalidate(&key(99, "m")));
    }

    #[test]
    fn test_cache_invalidate_model() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "bert"), emb(4));
        c.insert(key(2, "bert"), emb(4));
        c.insert(key(3, "gpt"), emb(4));
        let removed = c.invalidate_model("bert");
        assert_eq!(removed, 2);
        assert!(c.get(&key(1, "bert")).is_none());
        assert!(c.get(&key(2, "bert")).is_none());
        assert!(c.get(&key(3, "gpt")).is_some());
    }

    #[test]
    fn test_cache_invalidate_model_none() {
        let mut c = EmbeddingCache::new(10);
        assert_eq!(c.invalidate_model("unknown"), 0);
    }

    #[test]
    fn test_cache_size_decreases_on_invalidate() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), vec![0.0f32; 64]);
        let before = c.stats().total_size_bytes;
        c.invalidate(&key(1, "m"));
        assert_eq!(c.stats().total_size_bytes, before - 64 * 4);
    }

    // ── Update in-place ───────────────────────────────────────────────────────

    #[test]
    fn test_cache_insert_same_key_updates() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), emb(4));
        c.insert(key(1, "m"), vec![99.0; 4]);
        let got = c.get(&key(1, "m")).expect("should exist");
        assert_eq!(got[0], 99.0);
        assert_eq!(c.len(), 1);
    }

    // ── MemoryBoundedCache ────────────────────────────────────────────────────

    #[test]
    fn test_memory_bounded_cache_empty() {
        let c = MemoryBoundedCache::new(1024);
        assert!(c.is_empty());
        assert_eq!(c.total_size_bytes(), 0);
    }

    #[test]
    fn test_memory_bounded_cache_max_bytes() {
        let c = MemoryBoundedCache::new(4096);
        assert_eq!(c.max_bytes(), 4096);
    }

    #[test]
    fn test_memory_bounded_cache_stays_within_limit() {
        // 512 bytes limit; each 64-dim embedding = 256 bytes
        let mut c = MemoryBoundedCache::new(512);
        for i in 0..10u64 {
            c.insert(key(i, "m"), vec![0.0f32; 64]);
        }
        assert!(c.total_size_bytes() <= 512);
    }

    #[test]
    fn test_memory_bounded_cache_insert_and_get() {
        let mut c = MemoryBoundedCache::new(1 << 20); // 1 MiB
        let k = key(1, "m");
        c.insert(k.clone(), vec![1.0; 128]);
        let got = c.get(&k).expect("should be present");
        assert_eq!(got.len(), 128);
        assert!((got[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_memory_bounded_cache_stats() {
        let mut c = MemoryBoundedCache::new(1 << 20);
        c.insert(key(1, "m"), vec![0.0; 32]);
        c.get(&key(1, "m")); // hit
        c.get(&key(2, "m")); // miss
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    // ── Stress ────────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_stress_insert_get() {
        let mut c = EmbeddingCache::new(100);
        for i in 0u64..200 {
            c.insert(key(i, "m"), emb(32));
        }
        assert_eq!(c.len(), 100);
        let s = c.stats();
        assert_eq!(s.evictions, 100);
    }

    #[test]
    fn test_cache_all_hits() {
        let mut c = EmbeddingCache::new(50);
        for i in 0u64..50 {
            c.insert(key(i, "m"), emb(8));
        }
        for i in 0u64..50 {
            assert!(c.get(&key(i, "m")).is_some());
        }
        assert_eq!(c.stats().hits, 50);
        assert_eq!(c.stats().misses, 0);
        assert!((c.stats().hit_rate - 1.0).abs() < 1e-9);
    }

    // ── Round 7 additional tests ──────────────────────────────────────────────

    #[test]
    fn test_cache_access_count_increments_on_get() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), emb(4));
        // Access the entry three times and verify via a fresh insert (same key).
        c.get(&key(1, "m"));
        c.get(&key(1, "m"));
        c.get(&key(1, "m"));
        // After three hits, stats.hits should reflect them.
        assert_eq!(c.stats().hits, 3);
    }

    #[test]
    fn test_cache_multiple_models_isolated() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "bert"), emb(4));
        c.insert(key(1, "gpt"), emb(8));
        assert!(c.get(&key(1, "bert")).is_some());
        assert!(c.get(&key(1, "gpt")).is_some());
    }

    #[test]
    fn test_cache_key_hash_collision_different_models() {
        // Same hash, different model → different keys.
        let k1 = key(100, "modelA");
        let k2 = key(100, "modelB");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_evict_only_one_on_overflow() {
        let mut c = EmbeddingCache::new(3);
        c.insert(key(1, "m"), emb(4));
        c.insert(key(2, "m"), emb(4));
        c.insert(key(3, "m"), emb(4));
        c.insert(key(4, "m"), emb(4)); // evicts key(1)
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn test_cache_get_returns_correct_embedding() {
        let mut c = EmbeddingCache::new(5);
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        c.insert(key(42, "m"), v.clone());
        let got = c.get(&key(42, "m")).expect("should be present");
        assert_eq!(got, v.as_slice());
    }

    #[test]
    fn test_cache_is_not_empty_after_insert() {
        let mut c = EmbeddingCache::new(5);
        c.insert(key(1, "m"), emb(4));
        assert!(!c.is_empty());
    }

    #[test]
    fn test_cache_len_zero_initially() {
        let c = EmbeddingCache::new(5);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_cache_invalidate_all_via_model() {
        let mut c = EmbeddingCache::new(10);
        for i in 0u64..5 {
            c.insert(key(i, "bert"), emb(4));
        }
        c.insert(key(99, "gpt"), emb(4));
        let removed = c.invalidate_model("bert");
        assert_eq!(removed, 5);
        assert_eq!(c.len(), 1); // only gpt remains
    }

    #[test]
    fn test_cache_stats_evictions_multiple() {
        let mut c = EmbeddingCache::new(2);
        for i in 0u64..6 {
            c.insert(key(i, "m"), emb(4));
        }
        // Each of the 4 inserts beyond capacity evicts one entry.
        assert_eq!(c.stats().evictions, 4);
    }

    #[test]
    fn test_cache_size_zero_after_all_invalidated() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), emb(32));
        c.insert(key(2, "m"), emb(32));
        c.invalidate(&key(1, "m"));
        c.invalidate(&key(2, "m"));
        assert_eq!(c.stats().total_size_bytes, 0);
    }

    #[test]
    fn test_memory_bounded_cache_evicts_to_stay_within_limit() {
        // 256 bytes = 1 entry of 64 dims * 4 bytes each.
        let mut c = MemoryBoundedCache::new(256);
        for i in 0u64..5 {
            c.insert(key(i, "m"), vec![0.0f32; 64]);
        }
        assert!(c.total_size_bytes() <= 256);
    }

    #[test]
    fn test_memory_bounded_cache_get_returns_none_for_missing() {
        let mut c = MemoryBoundedCache::new(1024);
        assert!(c.get(&key(99, "m")).is_none());
    }

    #[test]
    fn test_memory_bounded_cache_len_tracks_inserts() {
        let mut c = MemoryBoundedCache::new(1 << 20);
        assert_eq!(c.len(), 0);
        c.insert(key(1, "m"), emb(4));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_cache_insert_updates_size_correctly() {
        let mut c = EmbeddingCache::new(10);
        c.insert(key(1, "m"), vec![0.0f32; 10]);
        assert_eq!(c.stats().total_size_bytes, 10 * 4);
        c.insert(key(2, "m"), vec![0.0f32; 20]);
        assert_eq!(c.stats().total_size_bytes, 30 * 4);
    }
}
