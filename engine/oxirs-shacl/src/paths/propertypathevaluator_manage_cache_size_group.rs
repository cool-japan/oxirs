//! # PropertyPathEvaluator - manage_cache_size_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Manage cache size with intelligent eviction
    pub(super) fn manage_cache_size(&mut self, cache_key: String, cached_result: CachedPathResult) {
        if self.cache.len() >= self.cache_config.max_cache_entries {
            if self.cache_config.intelligent_eviction {
                self.intelligent_cache_eviction();
            } else if let Some((oldest_key, _)) = self
                .cache
                .iter()
                .min_by_key(|(_, result)| result.last_accessed)
            {
                let oldest_key = oldest_key.clone();
                self.cache.remove(&oldest_key);
            }
        }
        self.cache.insert(cache_key, cached_result);
    }
    /// Intelligent cache eviction based on access patterns and freshness
    fn intelligent_cache_eviction(&mut self) {
        if self.cache.is_empty() {
            return;
        }
        let mut eviction_candidates: Vec<(String, f64)> = self
            .cache
            .iter()
            .map(|(key, result)| {
                let age_factor = result.cached_at.elapsed().as_secs() as f64 / 3600.0;
                let access_factor = 1.0 / (result.access_count as f64 + 1.0);
                let size_factor = result.estimated_size_bytes as f64 / 1024.0;
                let freshness_penalty = 1.0 - result.freshness_score;
                let eviction_score = age_factor * 0.3
                    + access_factor * 0.4
                    + size_factor * 0.1
                    + freshness_penalty * 0.2;
                (key.clone(), eviction_score)
            })
            .collect();
        eviction_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((key_to_remove, _)) = eviction_candidates.first() {
            self.cache.remove(key_to_remove);
        }
    }
}
