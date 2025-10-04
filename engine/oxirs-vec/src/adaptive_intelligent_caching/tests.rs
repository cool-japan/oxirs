//! Tests for the adaptive intelligent caching system

#![cfg(test)]

use crate::similarity::SimilarityMetric;
use std::collections::HashMap;
use std::time::{Instant, SystemTime};

use super::cache::AdaptiveIntelligentCache;
use super::config::CacheConfiguration;
use super::eviction::LRUEvictionPolicy;
use super::types::{CacheKey, CacheMetadata, CacheValue, ExportFormat};

#[test]
fn test_adaptive_cache_creation() {
    let config = CacheConfiguration::default();
    let cache = AdaptiveIntelligentCache::new(config).unwrap();
    // Cache should be created successfully with default config (3 tiers)
    let stats = cache.get_statistics();
    assert_eq!(stats.tier_statistics.len(), 3);
}

#[test]
fn test_cache_store_and_retrieve() {
    let config = CacheConfiguration::default();
    let mut cache = AdaptiveIntelligentCache::new(config).unwrap();

    let key = CacheKey {
        query_vector: vec![1, 2, 3, 4],
        similarity_metric: SimilarityMetric::Cosine,
        parameters: HashMap::new(),
    };

    let value = CacheValue {
        results: vec![("vec1".to_string(), 0.95)],
        metadata: CacheMetadata {
            size_bytes: 1024,
            computation_cost: 0.5,
            quality_score: 0.9,
            staleness_factor: 0.1,
        },
        created_at: SystemTime::now(),
        last_accessed: SystemTime::now(),
        access_count: 1,
    };

    cache.store(key.clone(), value.clone()).unwrap();
    let retrieved = cache.retrieve(&key);

    assert!(retrieved.is_some());
    let retrieved_value = retrieved.unwrap();
    assert_eq!(retrieved_value.results, value.results);
}

#[test]
fn test_cache_statistics() {
    let config = CacheConfiguration::default();
    let cache = AdaptiveIntelligentCache::new(config).unwrap();
    let stats = cache.get_statistics();

    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.hit_rate, 0.0);
}

#[test]
fn test_cache_optimization() {
    let config = CacheConfiguration::default();
    let mut cache = AdaptiveIntelligentCache::new(config).unwrap();

    let result = cache.optimize().unwrap();
    assert!(result.improvement_score >= 0.0);
}

#[test]
fn test_performance_data_export() {
    let config = CacheConfiguration::default();
    let cache = AdaptiveIntelligentCache::new(config).unwrap();

    let json_export = cache.export_performance_data(ExportFormat::Json).unwrap();
    assert!(!json_export.is_empty());

    let prometheus_export = cache
        .export_performance_data(ExportFormat::Prometheus)
        .unwrap();
    assert!(!prometheus_export.is_empty());
}

#[test]
fn test_eviction_policies() {
    let mut lru = LRUEvictionPolicy::new();
    let key = CacheKey {
        query_vector: vec![1, 2, 3],
        similarity_metric: SimilarityMetric::Cosine,
        parameters: HashMap::new(),
    };

    use super::eviction::EvictionPolicy;
    lru.on_store(&key, 1024, Instant::now());
    lru.on_access(&key, Instant::now());

    let items = vec![];
    let evicted = lru.evict(2048, 1024, &items);
    assert!(!evicted.is_empty());
}
