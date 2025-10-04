//! Cache eviction policies

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::types::{CacheItem, CacheKey, EvictionStatistics};

/// Eviction policy trait for different eviction strategies
pub trait EvictionPolicy: Send + Sync + std::fmt::Debug {
    /// Determine which items to evict
    fn evict(&mut self, current_size: u64, target_size: u64, items: &[CacheItem]) -> Vec<CacheKey>;

    /// Update access information for an item
    fn on_access(&mut self, key: &CacheKey, access_time: Instant);

    /// Update when an item is stored
    fn on_store(&mut self, key: &CacheKey, size: u64, store_time: Instant);

    /// Get policy-specific statistics
    fn statistics(&self) -> EvictionStatistics;
}

/// LRU (Least Recently Used) eviction policy
#[derive(Debug)]
pub struct LRUEvictionPolicy {
    access_order: VecDeque<CacheKey>,
}

impl Default for LRUEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LRUEvictionPolicy {
    pub fn new() -> Self {
        Self {
            access_order: VecDeque::new(),
        }
    }
}

impl EvictionPolicy for LRUEvictionPolicy {
    fn evict(
        &mut self,
        current_size: u64,
        target_size: u64,
        _items: &[CacheItem],
    ) -> Vec<CacheKey> {
        let bytes_to_evict = current_size.saturating_sub(target_size);
        let items_to_evict = (bytes_to_evict / 1024).max(1) as usize; // Estimate items to evict

        self.access_order
            .iter()
            .take(items_to_evict)
            .cloned()
            .collect()
    }

    fn on_access(&mut self, key: &CacheKey, _access_time: Instant) {
        // Move to back (most recently used)
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            let key = self.access_order.remove(pos).unwrap();
            self.access_order.push_back(key);
        }
    }

    fn on_store(&mut self, key: &CacheKey, _size: u64, _store_time: Instant) {
        self.access_order.push_back(key.clone());
    }

    fn statistics(&self) -> EvictionStatistics {
        EvictionStatistics::default()
    }
}

/// LFU (Least Frequently Used) eviction policy
#[derive(Debug)]
pub struct LFUEvictionPolicy {
    frequency_map: HashMap<CacheKey, u64>,
}

impl Default for LFUEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LFUEvictionPolicy {
    pub fn new() -> Self {
        Self {
            frequency_map: HashMap::new(),
        }
    }
}

impl EvictionPolicy for LFUEvictionPolicy {
    fn evict(
        &mut self,
        current_size: u64,
        target_size: u64,
        _items: &[CacheItem],
    ) -> Vec<CacheKey> {
        let bytes_to_evict = current_size.saturating_sub(target_size);
        let items_to_evict = (bytes_to_evict / 1024).max(1) as usize;

        let mut frequency_pairs: Vec<_> = self.frequency_map.iter().collect();
        frequency_pairs.sort_by_key(|&(_, &freq)| freq);

        frequency_pairs
            .iter()
            .take(items_to_evict)
            .map(|(key, _)| (*key).clone())
            .collect()
    }

    fn on_access(&mut self, key: &CacheKey, _access_time: Instant) {
        *self.frequency_map.entry(key.clone()).or_insert(0) += 1;
    }

    fn on_store(&mut self, key: &CacheKey, _size: u64, _store_time: Instant) {
        self.frequency_map.insert(key.clone(), 0);
    }

    fn statistics(&self) -> EvictionStatistics {
        EvictionStatistics::default()
    }
}

/// Adaptive eviction policy that combines LRU and LFU
#[derive(Debug)]
pub struct AdaptiveEvictionPolicy {
    lru_component: LRUEvictionPolicy,
    lfu_component: LFUEvictionPolicy,
    lru_weight: f64,
}

impl Default for AdaptiveEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveEvictionPolicy {
    pub fn new() -> Self {
        Self {
            lru_component: LRUEvictionPolicy::new(),
            lfu_component: LFUEvictionPolicy::new(),
            lru_weight: 0.5,
        }
    }
}

impl EvictionPolicy for AdaptiveEvictionPolicy {
    fn evict(&mut self, current_size: u64, target_size: u64, items: &[CacheItem]) -> Vec<CacheKey> {
        // Combine LRU and LFU decisions
        let lru_candidates = self.lru_component.evict(current_size, target_size, items);
        let lfu_candidates = self.lfu_component.evict(current_size, target_size, items);

        // For simplicity, interleave the results based on weights
        let lru_count = (lru_candidates.len() as f64 * self.lru_weight) as usize;
        let mut result = Vec::new();
        result.extend(lru_candidates.into_iter().take(lru_count));
        result.extend(lfu_candidates.into_iter().take(items.len() - lru_count));

        result
    }

    fn on_access(&mut self, key: &CacheKey, access_time: Instant) {
        self.lru_component.on_access(key, access_time);
        self.lfu_component.on_access(key, access_time);
    }

    fn on_store(&mut self, key: &CacheKey, size: u64, store_time: Instant) {
        self.lru_component.on_store(key, size, store_time);
        self.lfu_component.on_store(key, size, store_time);
    }

    fn statistics(&self) -> EvictionStatistics {
        EvictionStatistics::default()
    }
}
