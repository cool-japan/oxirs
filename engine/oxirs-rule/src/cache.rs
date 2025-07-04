//! # Advanced Caching System for Rule Engines
//!
//! This module provides sophisticated caching mechanisms to improve rule engine performance:
//! - Rule result caching with LRU eviction
//! - Fact derivation memoization
//! - Query result caching
//! - Intelligent cache warming and prefetching
//! - Cache statistics and monitoring

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::{Rule, RuleAtom, Term};

/// Cache key for rule execution results
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuleCacheKey {
    pub rule_name: String,
    pub input_facts: Vec<RuleAtom>,
}

/// Cache key for fact derivation queries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DerivationCacheKey {
    pub goal_fact: RuleAtom,
    pub context_facts: Vec<RuleAtom>,
}

/// Cached result entry
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub timestamp: Instant,
    pub access_count: usize,
    pub last_access: Instant,
    pub ttl: Option<Duration>,
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,      // Least Recently Used
    LFU,      // Least Frequently Used
    FIFO,     // First In, First Out
    TTL,      // Time To Live
    Adaptive, // Combines multiple strategies
}

/// Cache statistics and monitoring
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub insertions: usize,
    pub memory_usage: usize,
    pub hit_rate: f64,
    pub average_access_time: Duration,
    pub cache_size: usize,
    pub max_size: usize,
}

/// LRU cache implementation with advanced features
#[derive(Debug)]
pub struct SmartCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    entries: HashMap<K, CacheEntry<V>>,
    access_order: VecDeque<K>,
    max_size: usize,
    policy: EvictionPolicy,
    stats: CacheStatistics,
    default_ttl: Option<Duration>,
}

/// Rule-specific caching system
#[derive(Debug)]
pub struct RuleCache {
    rule_results: Arc<RwLock<SmartCache<RuleCacheKey, Vec<RuleAtom>>>>,
    derivation_results: Arc<RwLock<SmartCache<DerivationCacheKey, bool>>>,
    unification_cache: Arc<RwLock<SmartCache<String, HashMap<String, String>>>>,
    pattern_cache: Arc<RwLock<SmartCache<String, Vec<RuleAtom>>>>,
    enabled: bool,
}

impl<K, V> SmartCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    /// Create a new smart cache with specified capacity and policy
    pub fn new(max_size: usize, policy: EvictionPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_size,
            policy,
            stats: CacheStatistics::default(),
            default_ttl: None,
        }
    }

    /// Create cache with time-to-live
    pub fn with_ttl(max_size: usize, policy: EvictionPolicy, ttl: Duration) -> Self {
        let mut cache = Self::new(max_size, policy);
        cache.default_ttl = Some(ttl);
        cache
    }

    /// Get value from cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        let now = Instant::now();

        // Check if entry exists and get value without holding the mutable borrow
        let result = if let Some(entry) = self.entries.get(key) {
            // Check TTL expiration
            if let Some(ttl) = entry.ttl {
                if now.duration_since(entry.timestamp) > ttl {
                    None // Will be removed below
                } else {
                    Some(entry.value.clone())
                }
            } else {
                Some(entry.value.clone())
            }
        } else {
            None
        };

        match result {
            Some(value) => {
                // Update access statistics - now we can safely get mutable access
                if let Some(entry) = self.entries.get_mut(key) {
                    entry.access_count += 1;
                    entry.last_access = now;
                }

                // Update access order for LRU
                self.update_access_order(key);

                self.stats.hits += 1;
                self.update_hit_rate();

                Some(value)
            }
            None => {
                // Check if we need to remove expired entry
                if self.entries.contains_key(key) {
                    self.entries.remove(key);
                    self.remove_from_access_order(key);
                }

                self.stats.misses += 1;
                self.update_hit_rate();
                None
            }
        }
    }

    /// Insert value into cache
    pub fn insert(&mut self, key: K, value: V) {
        self.insert_with_ttl(key, value, self.default_ttl);
    }

    /// Insert value with custom TTL
    pub fn insert_with_ttl(&mut self, key: K, value: V, ttl: Option<Duration>) {
        let now = Instant::now();

        // Check if we need to evict
        if self.entries.len() >= self.max_size && !self.entries.contains_key(&key) {
            self.evict();
        }

        let entry = CacheEntry {
            value,
            timestamp: now,
            access_count: 1,
            last_access: now,
            ttl,
        };

        // If key already exists, remove from access order
        if self.entries.contains_key(&key) {
            self.remove_from_access_order(&key);
        } else {
            self.stats.insertions += 1;
        }

        self.entries.insert(key.clone(), entry);
        self.access_order.push_back(key);

        self.update_memory_usage();
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.remove_from_access_order(key);
            self.update_memory_usage();
            Some(entry.value)
        } else {
            None
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.stats = CacheStatistics::default();
        self.stats.max_size = self.max_size;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStatistics {
        &self.stats
    }

    /// Clean expired entries
    pub fn cleanup_expired(&mut self) {
        let now = Instant::now();
        let mut expired_keys = Vec::new();

        for (key, entry) in &self.entries {
            if let Some(ttl) = entry.ttl {
                if now.duration_since(entry.timestamp) > ttl {
                    expired_keys.push(key.clone());
                }
            }
        }

        for key in expired_keys {
            self.remove(&key);
        }
    }

    /// Evict entries based on policy
    fn evict(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let key_to_evict = match self.policy {
            EvictionPolicy::LRU => self.access_order.front().cloned(),
            EvictionPolicy::LFU => self.find_least_frequently_used(),
            EvictionPolicy::FIFO => self.access_order.front().cloned(),
            EvictionPolicy::TTL => self.find_expired_entry(),
            EvictionPolicy::Adaptive => self.adaptive_eviction(),
        };

        if let Some(key) = key_to_evict {
            self.remove(&key);
            self.stats.evictions += 1;
        }
    }

    fn find_least_frequently_used(&self) -> Option<K> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone())
    }

    fn find_expired_entry(&self) -> Option<K> {
        let now = Instant::now();
        for (key, entry) in &self.entries {
            if let Some(ttl) = entry.ttl {
                if now.duration_since(entry.timestamp) > ttl {
                    return Some(key.clone());
                }
            }
        }
        // If no expired entries, fall back to LRU
        self.access_order.front().cloned()
    }

    fn adaptive_eviction(&self) -> Option<K> {
        // Adaptive strategy: consider both frequency and recency
        let now = Instant::now();
        let mut best_score = f64::INFINITY;
        let mut best_key = None;

        for (key, entry) in &self.entries {
            let recency_score = now.duration_since(entry.last_access).as_secs_f64();
            let frequency_score = 1.0 / (entry.access_count as f64 + 1.0);
            let combined_score = recency_score * frequency_score;

            if combined_score < best_score {
                best_score = combined_score;
                best_key = Some(key.clone());
            }
        }

        best_key
    }

    fn update_access_order(&mut self, key: &K) {
        self.remove_from_access_order(key);
        self.access_order.push_back(key.clone());
    }

    fn remove_from_access_order(&mut self, key: &K) {
        if let Some(pos) = self.access_order.iter().position(|x| x == key) {
            self.access_order.remove(pos);
        }
    }

    fn update_hit_rate(&mut self) {
        let total_requests = self.stats.hits + self.stats.misses;
        if total_requests > 0 {
            self.stats.hit_rate = self.stats.hits as f64 / total_requests as f64;
        }
    }

    fn update_memory_usage(&mut self) {
        self.stats.memory_usage = self.entries.len() * 128; // Rough estimate
        self.stats.cache_size = self.entries.len();
    }
}

impl RuleCache {
    /// Create a new rule cache system
    pub fn new() -> Self {
        Self {
            rule_results: Arc::new(RwLock::new(SmartCache::new(1000, EvictionPolicy::Adaptive))),
            derivation_results: Arc::new(RwLock::new(SmartCache::new(500, EvictionPolicy::LRU))),
            unification_cache: Arc::new(RwLock::new(SmartCache::new(200, EvictionPolicy::LFU))),
            pattern_cache: Arc::new(RwLock::new(SmartCache::with_ttl(
                300,
                EvictionPolicy::TTL,
                Duration::from_secs(300),
            ))),
            enabled: true,
        }
    }

    /// Create cache with custom sizes
    pub fn with_sizes(
        rule_cache_size: usize,
        derivation_cache_size: usize,
        unification_cache_size: usize,
        pattern_cache_size: usize,
    ) -> Self {
        Self {
            rule_results: Arc::new(RwLock::new(SmartCache::new(
                rule_cache_size,
                EvictionPolicy::Adaptive,
            ))),
            derivation_results: Arc::new(RwLock::new(SmartCache::new(
                derivation_cache_size,
                EvictionPolicy::LRU,
            ))),
            unification_cache: Arc::new(RwLock::new(SmartCache::new(
                unification_cache_size,
                EvictionPolicy::LFU,
            ))),
            pattern_cache: Arc::new(RwLock::new(SmartCache::with_ttl(
                pattern_cache_size,
                EvictionPolicy::TTL,
                Duration::from_secs(300),
            ))),
            enabled: true,
        }
    }

    /// Enable or disable caching
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Cache rule execution result
    pub fn cache_rule_result(
        &self,
        rule_name: &str,
        input_facts: &[RuleAtom],
        result: Vec<RuleAtom>,
    ) {
        if !self.enabled {
            return;
        }

        let key = RuleCacheKey {
            rule_name: rule_name.to_string(),
            input_facts: input_facts.to_vec(),
        };

        if let Ok(mut cache) = self.rule_results.write() {
            cache.insert(key, result);
        }
    }

    /// Get cached rule result
    pub fn get_rule_result(
        &self,
        rule_name: &str,
        input_facts: &[RuleAtom],
    ) -> Option<Vec<RuleAtom>> {
        if !self.enabled {
            return None;
        }

        let key = RuleCacheKey {
            rule_name: rule_name.to_string(),
            input_facts: input_facts.to_vec(),
        };

        if let Ok(mut cache) = self.rule_results.write() {
            cache.get(&key)
        } else {
            None
        }
    }

    /// Cache derivation result
    pub fn cache_derivation(&self, goal: &RuleAtom, context: &[RuleAtom], result: bool) {
        if !self.enabled {
            return;
        }

        let key = DerivationCacheKey {
            goal_fact: goal.clone(),
            context_facts: context.to_vec(),
        };

        if let Ok(mut cache) = self.derivation_results.write() {
            cache.insert(key, result);
        }
    }

    /// Get cached derivation result
    pub fn get_derivation(&self, goal: &RuleAtom, context: &[RuleAtom]) -> Option<bool> {
        if !self.enabled {
            return None;
        }

        let key = DerivationCacheKey {
            goal_fact: goal.clone(),
            context_facts: context.to_vec(),
        };

        if let Ok(mut cache) = self.derivation_results.write() {
            cache.get(&key)
        } else {
            None
        }
    }

    /// Cache unification result
    pub fn cache_unification(&self, pattern: &str, bindings: HashMap<String, String>) {
        if !self.enabled {
            return;
        }

        if let Ok(mut cache) = self.unification_cache.write() {
            cache.insert(pattern.to_string(), bindings);
        }
    }

    /// Get cached unification result
    pub fn get_unification(&self, pattern: &str) -> Option<HashMap<String, String>> {
        if !self.enabled {
            return None;
        }

        if let Ok(mut cache) = self.unification_cache.write() {
            cache.get(&pattern.to_string())
        } else {
            None
        }
    }

    /// Cache pattern matching result
    pub fn cache_pattern(&self, pattern: &str, matches: Vec<RuleAtom>) {
        if !self.enabled {
            return;
        }

        if let Ok(mut cache) = self.pattern_cache.write() {
            cache.insert(pattern.to_string(), matches);
        }
    }

    /// Get cached pattern result
    pub fn get_pattern(&self, pattern: &str) -> Option<Vec<RuleAtom>> {
        if !self.enabled {
            return None;
        }

        if let Ok(mut cache) = self.pattern_cache.write() {
            cache.get(&pattern.to_string())
        } else {
            None
        }
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        if let Ok(mut cache) = self.rule_results.write() {
            cache.clear();
        }
        if let Ok(mut cache) = self.derivation_results.write() {
            cache.clear();
        }
        if let Ok(mut cache) = self.unification_cache.write() {
            cache.clear();
        }
        if let Ok(mut cache) = self.pattern_cache.write() {
            cache.clear();
        }
    }

    /// Clean expired entries from all caches
    pub fn cleanup_expired(&self) {
        if let Ok(mut cache) = self.rule_results.write() {
            cache.cleanup_expired();
        }
        if let Ok(mut cache) = self.derivation_results.write() {
            cache.cleanup_expired();
        }
        if let Ok(mut cache) = self.unification_cache.write() {
            cache.cleanup_expired();
        }
        if let Ok(mut cache) = self.pattern_cache.write() {
            cache.cleanup_expired();
        }
    }

    /// Get combined cache statistics
    pub fn get_statistics(&self) -> CachingStatistics {
        let rule_stats = if let Ok(cache) = self.rule_results.read() {
            cache.stats().clone()
        } else {
            CacheStatistics::default()
        };

        let derivation_stats = if let Ok(cache) = self.derivation_results.read() {
            cache.stats().clone()
        } else {
            CacheStatistics::default()
        };

        let unification_stats = if let Ok(cache) = self.unification_cache.read() {
            cache.stats().clone()
        } else {
            CacheStatistics::default()
        };

        let pattern_stats = if let Ok(cache) = self.pattern_cache.read() {
            cache.stats().clone()
        } else {
            CacheStatistics::default()
        };

        CachingStatistics {
            rule_cache: rule_stats,
            derivation_cache: derivation_stats,
            unification_cache: unification_stats,
            pattern_cache: pattern_stats,
            enabled: self.enabled,
        }
    }

    /// Warm cache with common patterns
    pub fn warm_cache(&self, rules: &[Rule], common_facts: &[RuleAtom]) {
        if !self.enabled {
            return;
        }

        // Pre-populate pattern cache with rule patterns
        for rule in rules {
            for atom in &rule.body {
                let pattern = format!("{:?}", atom);
                self.cache_pattern(&pattern, vec![atom.clone()]);
            }
        }

        // Pre-populate with common fact patterns
        for fact in common_facts {
            let pattern = format!("{:?}", fact);
            self.cache_pattern(&pattern, vec![fact.clone()]);
        }
    }
}

/// Combined caching statistics
#[derive(Debug, Clone)]
pub struct CachingStatistics {
    pub rule_cache: CacheStatistics,
    pub derivation_cache: CacheStatistics,
    pub unification_cache: CacheStatistics,
    pub pattern_cache: CacheStatistics,
    pub enabled: bool,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            insertions: 0,
            memory_usage: 0,
            hit_rate: 0.0,
            average_access_time: Duration::from_nanos(0),
            cache_size: 0,
            max_size: 0,
        }
    }
}

impl Default for RuleCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_cache_basic_operations() {
        let mut cache = SmartCache::new(3, EvictionPolicy::LRU);

        cache.insert("key1".to_string(), "value1".to_string());
        cache.insert("key2".to_string(), "value2".to_string());

        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        assert_eq!(cache.get(&"key2".to_string()), Some("value2".to_string()));
        assert_eq!(cache.get(&"key3".to_string()), None);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = SmartCache::new(2, EvictionPolicy::LRU);

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        cache.insert("key3", "value3"); // Should evict key1

        assert_eq!(cache.get(&"key1"), None);
        assert_eq!(cache.get(&"key2"), Some("value2"));
        assert_eq!(cache.get(&"key3"), Some("value3"));
    }

    #[test]
    fn test_ttl_expiration() {
        let mut cache = SmartCache::with_ttl(5, EvictionPolicy::TTL, Duration::from_millis(10));

        cache.insert("key1", "value1");
        assert_eq!(cache.get(&"key1"), Some("value1"));

        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(cache.get(&"key1"), None);
    }

    #[test]
    fn test_rule_cache_operations() {
        let cache = RuleCache::new();

        let rule_name = "test_rule";
        let input_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("test".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("entity".to_string()),
        }];
        let result = vec![RuleAtom::Triple {
            subject: Term::Constant("test".to_string()),
            predicate: Term::Constant("derived".to_string()),
            object: Term::Constant("property".to_string()),
        }];

        // Cache and retrieve
        cache.cache_rule_result(rule_name, &input_facts, result.clone());
        let cached_result = cache.get_rule_result(rule_name, &input_facts);

        assert_eq!(cached_result, Some(result));
    }

    #[test]
    fn test_cache_statistics() {
        let mut cache = SmartCache::new(10, EvictionPolicy::LRU);

        // Generate some activity
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        cache.get(&"key1");
        cache.get(&"key3"); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.insertions, 2);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_cache_warm_up() {
        let cache = RuleCache::new();

        let rules = vec![Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("test".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("derived".to_string()),
                object: Term::Constant("property".to_string()),
            }],
        }];

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("entity1".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("test".to_string()),
        }];

        cache.warm_cache(&rules, &facts);

        // Should have populated pattern cache
        let stats = cache.get_statistics();
        assert!(stats.pattern_cache.cache_size > 0);
    }
}
