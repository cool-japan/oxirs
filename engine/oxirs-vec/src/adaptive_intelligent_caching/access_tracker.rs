//! Access tracking for cache items

use std::collections::{BTreeMap, HashMap};
use std::time::{Instant, SystemTime};

use super::types::{AccessTracker, CacheKey};

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessTracker {
    pub fn new() -> Self {
        Self {
            access_counts: HashMap::new(),
            access_times: HashMap::new(),
            hot_keys: BTreeMap::new(),
        }
    }

    pub fn on_access(&mut self, key: &CacheKey, _access_time: Instant) {
        *self.access_counts.entry(key.clone()).or_insert(0) += 1;
        self.access_times
            .entry(key.clone())
            .or_default()
            .push_back(SystemTime::now());
    }

    pub fn on_store(&mut self, key: &CacheKey) {
        // Record that an item was stored
        self.access_times.entry(key.clone()).or_default();
    }

    pub fn on_remove(&mut self, key: &CacheKey) {
        self.access_counts.remove(key);
        self.access_times.remove(key);
    }
}
