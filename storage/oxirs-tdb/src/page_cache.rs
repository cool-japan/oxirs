//! LRU page cache for TDB storage.
//!
//! [`PageCache`] evicts the least-recently-used page when capacity is
//! exceeded, tracks hit/miss/eviction statistics, and supports pinning pages
//! to prevent them from being evicted.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Page ────────────────────────────────────────────────────────────────────

/// A single cached page.
#[derive(Debug, Clone)]
pub struct Page {
    /// The unique page identifier.
    pub page_id: u64,
    /// Raw page data.
    pub data: Vec<u8>,
    /// Whether this page has been modified since it was loaded.
    pub dirty: bool,
    /// Number of times this page has been accessed.
    pub access_count: u64,
}

impl Page {
    fn new(page_id: u64, data: Vec<u8>) -> Self {
        Self {
            page_id,
            data,
            dirty: false,
            access_count: 0,
        }
    }
}

// ── Stats ────────────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`PageCache`] instance.
#[derive(Debug, Clone, Default)]
pub struct PageCacheStats {
    /// Successful lookups where the page was already in cache.
    pub hits: u64,
    /// Lookups where the page was not found in cache.
    pub misses: u64,
    /// Total number of pages evicted (clean or dirty).
    pub evictions: u64,
    /// Number of dirty pages that were evicted (these must be written back).
    pub dirty_evictions: u64,
}

// ── PageCache ────────────────────────────────────────────────────────────────

/// An LRU page cache of fixed capacity.
///
/// The cache uses a [`VecDeque`] as the LRU ordering list and a [`HashMap`]
/// for O(1) page lookups.  Pinned pages are excluded from eviction.
pub struct PageCache {
    capacity: usize,
    pages: HashMap<u64, Page>,
    /// LRU order: front = most-recently-used, back = least-recently-used.
    lru_order: VecDeque<u64>,
    stats: PageCacheStats,
    /// Pages that cannot be evicted.
    pinned: HashSet<u64>,
}

impl PageCache {
    /// Create a new cache with the given capacity (in pages).
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "PageCache capacity must be > 0");
        Self {
            capacity,
            pages: HashMap::new(),
            lru_order: VecDeque::new(),
            stats: PageCacheStats::default(),
            pinned: HashSet::new(),
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Move `page_id` to the front of the LRU queue (most-recently used).
    fn touch(&mut self, page_id: u64) {
        if let Some(pos) = self.lru_order.iter().position(|&id| id == page_id) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_front(page_id);
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Look up a page by id, recording a hit or miss.
    pub fn get(&mut self, page_id: u64) -> Option<&Page> {
        if self.pages.contains_key(&page_id) {
            self.stats.hits += 1;
            let page = self.pages.get_mut(&page_id)?;
            page.access_count += 1;
            self.touch(page_id);
            self.pages.get(&page_id)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Look up a page by id for mutation, recording a hit or miss.
    pub fn get_mut(&mut self, page_id: u64) -> Option<&mut Page> {
        if self.pages.contains_key(&page_id) {
            self.stats.hits += 1;
            self.touch(page_id);
            let page = self.pages.get_mut(&page_id)?;
            page.access_count += 1;
            Some(page)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a page into the cache.
    ///
    /// If the cache is at capacity an LRU eviction is performed first.
    /// Returns the evicted [`Page`] if one was displaced (the caller is
    /// responsible for flushing dirty evictions).
    pub fn insert(&mut self, page_id: u64, data: Vec<u8>) -> Option<Page> {
        // If the page is already cached, update it in-place.
        if self.pages.contains_key(&page_id) {
            self.touch(page_id);
            let p = self.pages.get_mut(&page_id)?;
            p.data = data;
            return None;
        }

        // Evict if needed.
        let evicted = if self.pages.len() >= self.capacity {
            self.evict()
        } else {
            None
        };

        self.pages.insert(page_id, Page::new(page_id, data));
        self.lru_order.push_front(page_id);

        evicted
    }

    /// Mark a cached page as dirty (needs write-back).
    ///
    /// No-op if the page is not cached.
    pub fn mark_dirty(&mut self, page_id: u64) {
        if let Some(p) = self.pages.get_mut(&page_id) {
            p.dirty = true;
        }
    }

    /// Collect all dirty pages, clearing their dirty flag.
    ///
    /// Returns a `Vec` of the pages that need to be written back to storage.
    pub fn flush_dirty(&mut self) -> Vec<Page> {
        let dirty_ids: Vec<u64> = self
            .pages
            .values()
            .filter(|p| p.dirty)
            .map(|p| p.page_id)
            .collect();

        let mut result = Vec::new();
        for id in dirty_ids {
            if let Some(p) = self.pages.get_mut(&id) {
                p.dirty = false;
                result.push(p.clone());
            }
        }
        result
    }

    /// Manually evict the LRU (un-pinned) page.
    ///
    /// Returns the evicted page, or `None` if all pages are pinned or the
    /// cache is empty.
    pub fn evict(&mut self) -> Option<Page> {
        // Find the LRU un-pinned page id.
        let evict_id = self
            .lru_order
            .iter()
            .rev()
            .find(|&&id| !self.pinned.contains(&id))
            .copied()?;

        // Remove from LRU queue.
        if let Some(pos) = self.lru_order.iter().position(|&id| id == evict_id) {
            self.lru_order.remove(pos);
        }

        let page = self.pages.remove(&evict_id)?;
        self.stats.evictions += 1;
        if page.dirty {
            self.stats.dirty_evictions += 1;
        }
        Some(page)
    }

    /// Returns `true` if the page is currently cached.
    pub fn contains(&self, page_id: u64) -> bool {
        self.pages.contains_key(&page_id)
    }

    /// Current number of pages in cache.
    pub fn size(&self) -> usize {
        self.pages.len()
    }

    /// Maximum number of pages the cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> &PageCacheStats {
        &self.stats
    }

    /// Number of dirty (un-flushed) pages currently in cache.
    pub fn dirty_count(&self) -> usize {
        self.pages.values().filter(|p| p.dirty).count()
    }

    /// Cache hit rate as a value in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no accesses have been recorded.
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        }
    }

    /// Remove all pages from the cache and reset statistics.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.lru_order.clear();
        self.pinned.clear();
        self.stats = PageCacheStats::default();
    }

    /// Prevent `page_id` from being evicted.
    ///
    /// No-op if the page is not in cache.
    pub fn pin(&mut self, page_id: u64) {
        if self.pages.contains_key(&page_id) {
            self.pinned.insert(page_id);
        }
    }

    /// Allow `page_id` to be evicted again.
    pub fn unpin(&mut self, page_id: u64) {
        self.pinned.remove(&page_id);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(seed: u8) -> Vec<u8> {
        vec![seed; 64]
    }

    // ── construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_cache_empty() {
        let c = PageCache::new(4);
        assert_eq!(c.size(), 0);
        assert_eq!(c.capacity(), 4);
    }

    #[test]
    #[should_panic]
    fn test_zero_capacity_panics() {
        let _ = PageCache::new(0);
    }

    // ── insert / contains ───────────────────────────────────────────────────

    #[test]
    fn test_insert_and_contains() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        assert!(c.contains(1));
        assert!(!c.contains(2));
    }

    #[test]
    fn test_insert_returns_none_when_below_capacity() {
        let mut c = PageCache::new(4);
        let ev = c.insert(1, make_data(1));
        assert!(ev.is_none());
    }

    #[test]
    fn test_insert_update_existing_no_eviction() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        let ev = c.insert(1, make_data(2));
        assert!(ev.is_none());
        assert_eq!(c.size(), 1);
    }

    // ── LRU eviction order ──────────────────────────────────────────────────

    #[test]
    fn test_capacity_enforced() {
        let mut c = PageCache::new(3);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.insert(3, make_data(3));
        let ev = c.insert(4, make_data(4));
        assert_eq!(c.size(), 3);
        // Oldest (page 1) should have been evicted.
        assert!(ev.is_some());
        assert_eq!(ev.unwrap().page_id, 1);
        assert!(!c.contains(1));
        assert!(c.contains(4));
    }

    #[test]
    fn test_lru_evict_oldest() {
        let mut c = PageCache::new(3);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.insert(3, make_data(3));
        // Access page 1 to make it recently used.
        c.get(1);
        // Page 2 is now the LRU.
        let ev = c.insert(4, make_data(4));
        assert_eq!(ev.unwrap().page_id, 2);
    }

    #[test]
    fn test_get_touches_lru() {
        let mut c = PageCache::new(2);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        // Touch page 1 — page 2 becomes LRU.
        c.get(1);
        let ev = c.insert(3, make_data(3));
        assert_eq!(ev.unwrap().page_id, 2);
        assert!(c.contains(1));
        assert!(c.contains(3));
    }

    // ── get / get_mut ────────────────────────────────────────────────────────

    #[test]
    fn test_get_hit() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        assert!(c.get(1).is_some());
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 0);
    }

    #[test]
    fn test_get_miss() {
        let mut c = PageCache::new(4);
        assert!(c.get(99).is_none());
        assert_eq!(c.stats().misses, 1);
        assert_eq!(c.stats().hits, 0);
    }

    #[test]
    fn test_get_mut_modifies_data() {
        let mut c = PageCache::new(4);
        c.insert(1, vec![0u8; 4]);
        {
            let p = c.get_mut(1).unwrap();
            p.data[0] = 42;
        }
        let p = c.get(1).unwrap();
        assert_eq!(p.data[0], 42);
    }

    #[test]
    fn test_access_count_increments() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.get(1);
        c.get(1);
        assert_eq!(c.get(1).unwrap().access_count, 3);
    }

    // ── dirty tracking ───────────────────────────────────────────────────────

    #[test]
    fn test_mark_dirty() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.mark_dirty(1);
        assert_eq!(c.dirty_count(), 1);
        assert!(c.get(1).unwrap().dirty);
    }

    #[test]
    fn test_mark_dirty_noop_missing() {
        let mut c = PageCache::new(4);
        c.mark_dirty(99); // Should not panic.
        assert_eq!(c.dirty_count(), 0);
    }

    #[test]
    fn test_flush_dirty_clears_flag() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.mark_dirty(1);
        c.mark_dirty(2);
        let flushed = c.flush_dirty();
        assert_eq!(flushed.len(), 2);
        assert_eq!(c.dirty_count(), 0);
    }

    #[test]
    fn test_flush_dirty_returns_only_dirty() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.mark_dirty(1);
        let flushed = c.flush_dirty();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].page_id, 1);
    }

    #[test]
    fn test_dirty_eviction_counted() {
        let mut c = PageCache::new(2);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.mark_dirty(1);
        // Access page 2 so page 1 is LRU.
        c.get(2);
        // Insert page 3 — page 1 (dirty) gets evicted.
        c.insert(3, make_data(3));
        assert_eq!(c.stats().dirty_evictions, 1);
        assert_eq!(c.stats().evictions, 1);
    }

    // ── evict ────────────────────────────────────────────────────────────────

    #[test]
    fn test_evict_manual() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        let ev = c.evict();
        assert!(ev.is_some());
        assert_eq!(c.size(), 1);
    }

    #[test]
    fn test_evict_empty_returns_none() {
        let mut c = PageCache::new(4);
        assert!(c.evict().is_none());
    }

    // ── pin / unpin ──────────────────────────────────────────────────────────

    #[test]
    fn test_pin_prevents_eviction() {
        let mut c = PageCache::new(2);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.pin(1);
        // With page 1 pinned, inserting page 3 must evict page 2.
        let ev = c.insert(3, make_data(3));
        assert_eq!(ev.unwrap().page_id, 2);
        assert!(c.contains(1));
    }

    #[test]
    fn test_unpin_allows_eviction() {
        let mut c = PageCache::new(2);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.pin(1);
        c.unpin(1);
        // Now page 1 or 2 can be evicted; page 2 is LRU after we touch 1.
        c.get(1);
        let ev = c.insert(3, make_data(3));
        assert_eq!(ev.unwrap().page_id, 2);
    }

    #[test]
    fn test_pin_noop_when_not_cached() {
        let mut c = PageCache::new(4);
        c.pin(99); // Should not panic.
        assert!(!c.pinned.contains(&99));
    }

    #[test]
    fn test_all_pinned_evict_returns_none() {
        let mut c = PageCache::new(2);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.pin(1);
        c.pin(2);
        assert!(c.evict().is_none());
    }

    // ── hit_rate ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hit_rate_no_accesses() {
        let c = PageCache::new(4);
        assert_eq!(c.hit_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.get(1);
        c.get(1);
        assert!((c.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_mixed() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.get(1); // hit
        c.get(2); // miss
        assert!((c.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_empties_cache() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.mark_dirty(1);
        c.get(1);
        c.clear();
        assert_eq!(c.size(), 0);
        assert_eq!(c.dirty_count(), 0);
        assert_eq!(c.stats().hits, 0);
        assert_eq!(c.stats().misses, 0);
    }

    #[test]
    fn test_reuse_after_clear() {
        let mut c = PageCache::new(2);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.clear();
        c.insert(3, make_data(3));
        assert!(c.contains(3));
        assert_eq!(c.size(), 1);
    }

    // ── size / capacity ───────────────────────────────────────────────────────

    #[test]
    fn test_size_tracks_insertions() {
        let mut c = PageCache::new(10);
        for i in 0..5u64 {
            c.insert(i, make_data(i as u8));
        }
        assert_eq!(c.size(), 5);
    }

    #[test]
    fn test_capacity_unchanged() {
        let c = PageCache::new(42);
        assert_eq!(c.capacity(), 42);
    }

    // ── stat accuracy ─────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_stat_increments() {
        let mut c = PageCache::new(1);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2)); // evicts page 1
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn test_multiple_evictions_counted() {
        let mut c = PageCache::new(1);
        for i in 0..5u64 {
            c.insert(i, make_data(i as u8));
        }
        assert_eq!(c.stats().evictions, 4);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_page_data_stored_correctly() {
        let mut c = PageCache::new(4);
        let data = vec![1u8, 2, 3, 4];
        c.insert(10, data.clone());
        assert_eq!(c.get(10).unwrap().data, data);
    }

    #[test]
    fn test_page_id_matches() {
        let mut c = PageCache::new(4);
        c.insert(42, make_data(7));
        assert_eq!(c.get(42).unwrap().page_id, 42);
    }

    #[test]
    fn test_page_not_dirty_initially() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        assert!(!c.get(1).unwrap().dirty);
    }

    #[test]
    fn test_dirty_count_zero_initially() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        assert_eq!(c.dirty_count(), 0);
    }

    #[test]
    fn test_dirty_count_tracks_multiple() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.insert(3, make_data(3));
        c.mark_dirty(1);
        c.mark_dirty(3);
        assert_eq!(c.dirty_count(), 2);
    }

    #[test]
    fn test_evict_returns_lru_page() {
        let mut c = PageCache::new(3);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.insert(3, make_data(3));
        // page 1 is LRU (oldest); touch 2 and 3 to confirm.
        c.get(2);
        c.get(3);
        let ev = c.evict().unwrap();
        assert_eq!(ev.page_id, 1);
    }

    #[test]
    fn test_contains_after_eviction() {
        let mut c = PageCache::new(1);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2)); // evicts 1
        assert!(!c.contains(1));
        assert!(c.contains(2));
    }

    #[test]
    fn test_stats_hits_misses_after_multiple_ops() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.get(1); // hit
        c.get(1); // hit
        c.get(2); // miss
        c.get(3); // miss
        assert_eq!(c.stats().hits, 2);
        assert_eq!(c.stats().misses, 2);
    }

    #[test]
    fn test_large_capacity_cache() {
        let mut c = PageCache::new(1000);
        for i in 0..500u64 {
            c.insert(i, make_data((i % 256) as u8));
        }
        assert_eq!(c.size(), 500);
        assert!(c.stats().evictions == 0);
    }

    #[test]
    fn test_flush_dirty_returns_clones_not_removed() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.mark_dirty(1);
        let flushed = c.flush_dirty();
        assert_eq!(flushed.len(), 1);
        // Page should still be in cache.
        assert!(c.contains(1));
    }

    #[test]
    fn test_pin_after_clear_does_nothing() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.clear();
        c.pin(1); // Page no longer exists — no-op.
        // After clear, pinned set should be empty.
        assert!(!c.pinned.contains(&1));
    }

    #[test]
    fn test_get_mut_hit_increments_access_count() {
        let mut c = PageCache::new(4);
        c.insert(1, make_data(1));
        c.get_mut(1);
        assert_eq!(c.get(1).unwrap().access_count, 2);
    }

    #[test]
    fn test_get_mut_miss_increments_miss_stat() {
        let mut c = PageCache::new(4);
        assert!(c.get_mut(99).is_none());
        assert_eq!(c.stats().misses, 1);
    }

    #[test]
    fn test_insert_at_exact_capacity_triggers_eviction() {
        let mut c = PageCache::new(3);
        c.insert(1, make_data(1));
        c.insert(2, make_data(2));
        c.insert(3, make_data(3));
        // Cache is full; inserting one more should evict.
        let ev = c.insert(4, make_data(4));
        assert!(ev.is_some());
        assert_eq!(c.size(), 3);
    }

    #[test]
    fn test_dirty_eviction_stat_not_incremented_for_clean() {
        let mut c = PageCache::new(1);
        c.insert(1, make_data(1));
        // Page 1 is clean.
        c.insert(2, make_data(2));
        assert_eq!(c.stats().dirty_evictions, 0);
        assert_eq!(c.stats().evictions, 1);
    }
}
