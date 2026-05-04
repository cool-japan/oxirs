//! LRU eviction policy for the algebra-level plan cache.
//!
//! [`LruEviction`] maintains an ordered access list using a [`VecDeque<u64>`].
//! The front of the deque is the least-recently-used key; the back is the
//! most-recently-used.  On every access or insert, the key is moved to the
//! back.  When the capacity is exceeded the front is popped and returned as
//! the eviction candidate.
//!
//! This module is intentionally kept small and allocation-light.  The `O(n)`
//! retain call is acceptable for cache sizes ≤ 4096 (the practical upper bound
//! for an in-process algebra plan cache).

use std::collections::VecDeque;

/// Least-recently-used eviction tracker for `u64` cache keys.
///
/// The caller is responsible for actually removing the evicted entry from the
/// backing map — this type only tracks the access order.
pub struct LruEviction {
    capacity: usize,
    access_order: VecDeque<u64>,
}

impl LruEviction {
    /// Create a new `LruEviction` with the given `capacity`.
    ///
    /// Capacity must be at least 1; passing `0` results in a capacity of `1`.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            access_order: VecDeque::with_capacity(cap + 1),
        }
    }

    /// Record a cache access for `key` (cache hit path).
    ///
    /// Moves `key` to the MRU position.  If the tracker is already at capacity
    /// and `key` is a new entry, returns the key that should be evicted;
    /// otherwise returns `None`.
    ///
    /// For cache hits (`key` already tracked), this never evicts.
    pub fn on_access(&mut self, key: u64) -> Option<u64> {
        self.access_order.retain(|&k| k != key);
        self.access_order.push_back(key);
        if self.access_order.len() > self.capacity {
            self.access_order.pop_front()
        } else {
            None
        }
    }

    /// Record a new key insertion (cache miss path).
    ///
    /// Equivalent to [`on_access`](Self::on_access) — the new key is placed at
    /// the MRU position and the LRU key is returned if capacity is exceeded.
    pub fn on_insert(&mut self, key: u64) -> Option<u64> {
        self.on_access(key)
    }

    /// Number of keys currently tracked.
    pub fn len(&self) -> usize {
        self.access_order.len()
    }

    /// Returns `true` if no keys are tracked.
    pub fn is_empty(&self) -> bool {
        self.access_order.is_empty()
    }

    /// The configured maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_eviction_within_capacity() {
        let mut lru = LruEviction::new(3);
        assert!(lru.on_insert(1).is_none());
        assert!(lru.on_insert(2).is_none());
        assert!(lru.on_insert(3).is_none());
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn evicts_lru_on_overflow() {
        let mut lru = LruEviction::new(3);
        lru.on_insert(1);
        lru.on_insert(2);
        lru.on_insert(3);
        let evicted = lru.on_insert(4);
        assert_eq!(evicted, Some(1));
    }

    #[test]
    fn access_refreshes_mru_position() {
        let mut lru = LruEviction::new(3);
        lru.on_insert(1);
        lru.on_insert(2);
        lru.on_insert(3);
        lru.on_access(1); // 1 → MRU; LRU is now 2
        let evicted = lru.on_insert(4);
        assert_eq!(evicted, Some(2));
    }

    #[test]
    fn duplicate_insert_does_not_grow() {
        let mut lru = LruEviction::new(3);
        lru.on_insert(1);
        lru.on_insert(1);
        lru.on_insert(1);
        assert_eq!(lru.len(), 1);
    }

    #[test]
    fn is_empty_on_new() {
        let lru = LruEviction::new(10);
        assert!(lru.is_empty());
    }

    #[test]
    fn capacity_one_always_evicts_on_second_insert() {
        let mut lru = LruEviction::new(1);
        assert!(lru.on_insert(1).is_none());
        let evicted = lru.on_insert(2);
        assert_eq!(evicted, Some(1));
    }
}
