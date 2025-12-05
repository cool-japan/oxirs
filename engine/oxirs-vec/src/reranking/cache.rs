//! Re-ranking result caching

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingCacheConfig {
    pub max_size: usize,
}

pub struct RerankingCache {
    cache: RwLock<HashMap<String, f32>>,
    max_size: usize,
}

impl RerankingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size,
        }
    }

    pub fn get(&self, key: &str) -> Option<f32> {
        self.cache.read().ok()?.get(key).copied()
    }

    pub fn put(&self, key: String, value: f32) {
        if let Ok(mut cache) = self.cache.write() {
            if cache.len() >= self.max_size {
                // Simple eviction: remove first entry
                if let Some(first_key) = cache.keys().next().cloned() {
                    cache.remove(&first_key);
                }
            }
            cache.insert(key, value);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    pub fn stats(&self) -> (usize, usize) {
        if let Ok(cache) = self.cache.read() {
            (cache.len(), self.max_size)
        } else {
            (0, self.max_size)
        }
    }
}
