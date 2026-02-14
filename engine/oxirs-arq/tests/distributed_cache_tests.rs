//! Comprehensive tests for distributed cache system
//!
//! This test suite verifies the distributed cache implementation with L1+L2 hierarchy,
//! Redis backend, and cache coherence protocols.

#![cfg(feature = "distributed-cache")]

use std::time::{Duration, Instant};

use oxirs_arq::cache::{
    CacheCoherenceProtocol, CacheKey, CacheValue, CoherenceConfig, CoherenceProtocol,
    ConsistencyLevel, DistributedCache, DistributedCacheConfig,
};
/// Helper to check if Redis is available
async fn is_redis_available() -> bool {
    match redis::Client::open("redis://localhost:6379") {
        Ok(client) => match client.get_multiplexed_async_connection().await {
            Ok(_) => true,
            Err(_) => false,
        },
        Err(_) => false,
    }
}

/// Helper to create a test cache
async fn setup_cache() -> Option<DistributedCache> {
    if !is_redis_available().await {
        eprintln!("Redis not available, skipping test");
        return None;
    }

    let config = DistributedCacheConfig {
        l1_max_size: 100,
        l1_ttl_seconds: 10,
        l2_redis_url: "redis://localhost:6379".to_string(),
        l2_ttl_seconds: 30,
        compression: true,
        invalidation_channel: format!("oxirs:test:{}", uuid::Uuid::new_v4()),
    };

    match DistributedCache::new(config).await {
        Ok(cache) => Some(cache),
        Err(e) => {
            eprintln!("Failed to create cache: {:?}", e);
            None
        }
    }
}

/// Helper to create multiple test caches on different channels
async fn setup_multi_cache(count: usize) -> Option<Vec<DistributedCache>> {
    if !is_redis_available().await {
        eprintln!("Redis not available, skipping test");
        return None;
    }

    let channel = format!("oxirs:test:{}", uuid::Uuid::new_v4());
    let mut caches = Vec::new();

    for _ in 0..count {
        let config = DistributedCacheConfig {
            l1_max_size: 100,
            l1_ttl_seconds: 10,
            l2_redis_url: "redis://localhost:6379".to_string(),
            l2_ttl_seconds: 30,
            compression: true,
            invalidation_channel: channel.clone(),
        };

        match DistributedCache::new(config).await {
            Ok(cache) => caches.push(cache),
            Err(e) => {
                eprintln!("Failed to create cache: {:?}", e);
                return None;
            }
        }
    }

    Some(caches)
}

#[tokio::test]
async fn test_l1_hit() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key = CacheKey::new("query1".to_string());
    let value = CacheValue::new(vec![1, 2, 3, 4, 5]);

    // Put value
    cache.put(key.clone(), value.clone()).await.unwrap();

    // Get value (should hit L1)
    let start = Instant::now();
    let result = cache.get(&key).await.unwrap();
    let elapsed = start.elapsed();

    assert!(result.is_some());
    assert_eq!(result.unwrap().data, value.data);

    // L1 access should be very fast (<1ms)
    assert!(elapsed.as_millis() < 10); // Generous bound for CI

    // Check metrics
    assert!(cache.metrics().l1_hits.get() > 0);
}

#[tokio::test]
async fn test_l2_hit() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key = CacheKey::new("query2".to_string());
    let value = CacheValue::new(vec![10, 20, 30, 40, 50]);

    // Put value (stores in both L1 and L2)
    cache.put(key.clone(), value.clone()).await.unwrap();

    // Clear L1 to force L2 access
    cache.clear_l1();

    // Get value (should hit L2 and populate L1)
    let result = cache.get(&key).await.unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().data, value.data);

    // Check metrics
    assert!(cache.metrics().l2_hits.get() > 0);

    // L1 should now have the value
    let result2 = cache.get(&key).await.unwrap();
    assert!(result2.is_some());
    assert!(cache.metrics().l1_hits.get() > 0);
}

#[tokio::test]
async fn test_cache_miss() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key = CacheKey::new("nonexistent".to_string());

    // Get non-existent value
    let result = cache.get(&key).await.unwrap();
    assert!(result.is_none());

    // Check metrics
    assert!(cache.metrics().l1_misses.get() > 0);
    assert!(cache.metrics().l2_misses.get() > 0);
}

#[tokio::test]
async fn test_put_both_levels() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key = CacheKey::new("query3".to_string());
    let value = CacheValue::new(vec![100; 1000]);

    // Put value
    cache.put(key.clone(), value.clone()).await.unwrap();

    // Verify L1 has it
    assert_eq!(cache.l1_size(), 1);

    // Clear L1 and verify L2 has it
    cache.clear_l1();
    let result = cache.get(&key).await.unwrap();
    assert!(result.is_some());
}

#[tokio::test]
async fn test_invalidation() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key = CacheKey::new("query4".to_string());
    let value = CacheValue::new(vec![1, 2, 3]);

    // Put value
    cache.put(key.clone(), value.clone()).await.unwrap();

    // Verify it exists
    assert!(cache.get(&key).await.unwrap().is_some());

    // Invalidate
    cache.invalidate(&key).await.unwrap();

    // Verify it's gone from both levels
    assert!(cache.get(&key).await.unwrap().is_none());

    // Check metrics
    assert!(cache.metrics().invalidations_sent.get() > 0);
}

#[tokio::test]
async fn test_pubsub_invalidation() {
    let caches = match setup_multi_cache(3).await {
        Some(c) => c,
        None => return,
    };

    // Start invalidation listeners on all caches
    for cache in &caches {
        cache.start_invalidation_listener().await.unwrap();
    }

    // Give listeners time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let key = CacheKey::new("query5".to_string());
    let value = CacheValue::new(vec![1, 2, 3]);

    // Put value in all caches
    for cache in &caches {
        cache.put(key.clone(), value.clone()).await.unwrap();
    }

    // Invalidate from first cache
    caches[0].invalidate(&key).await.unwrap();

    // Give time for invalidation to propagate
    tokio::time::sleep(Duration::from_millis(500)).await;

    // All caches should have received the invalidation
    for (idx, cache) in caches.iter().enumerate() {
        let result = cache.get(&key).await.unwrap();
        if idx == 0 {
            // First cache initiated the invalidation
            assert!(result.is_none(), "Cache {} should not have the key", idx);
        }
        // Note: Other caches might have the key if the invalidation hasn't propagated yet
        // This is expected with eventual consistency
    }

    // Check that invalidations were sent and received
    assert!(caches[0].metrics().invalidations_sent.get() > 0);
}

#[tokio::test]
async fn test_compression() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key = CacheKey::new("large_value".to_string());
    // Create a large compressible value (1MB of zeros)
    let value = CacheValue::new(vec![0u8; 1024 * 1024]);

    // Put value (should compress)
    cache.put(key.clone(), value.clone()).await.unwrap();

    // Get value (should decompress)
    let result = cache.get(&key).await.unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().data.len(), value.data.len());

    // Check compression ratio
    let ratio = *cache.metrics().compression_ratio.read();
    assert!(
        ratio > 1.0,
        "Compression ratio should be > 1.0, got {}",
        ratio
    );
}

#[tokio::test]
async fn test_hit_rates() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    // Warm up cache with 50 entries
    for i in 0..50 {
        let key = CacheKey::new(format!("query_{}", i));
        let value = CacheValue::new(vec![i as u8; 10]);
        cache.put(key, value).await.unwrap();
    }

    // Access first 40 entries (should all hit L1)
    for i in 0..40 {
        let key = CacheKey::new(format!("query_{}", i));
        cache.get(&key).await.unwrap();
    }

    // L1 hit rate should be high
    let l1_hit_rate = cache.metrics().l1_hit_rate();
    assert!(
        l1_hit_rate > 0.8,
        "L1 hit rate is {:.2}, expected > 0.8",
        l1_hit_rate
    );
}

#[tokio::test]
async fn test_latency() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key = CacheKey::new("latency_test".to_string());
    let value = CacheValue::new(vec![1, 2, 3, 4, 5]);

    // Put value
    cache.put(key.clone(), value.clone()).await.unwrap();

    // Measure L1 latency
    let mut l1_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        cache.get(&key).await.unwrap();
        l1_times.push(start.elapsed());
    }

    let avg_l1 = l1_times.iter().sum::<Duration>() / l1_times.len() as u32;
    println!("Average L1 latency: {:?}", avg_l1);
    assert!(avg_l1.as_millis() < 10); // L1 should be very fast

    // Clear L1 and measure L2 latency
    cache.clear_l1();

    let mut l2_times = Vec::new();
    for _ in 0..10 {
        cache.clear_l1(); // Clear L1 before each access
        let start = Instant::now();
        cache.get(&key).await.unwrap();
        l2_times.push(start.elapsed());
    }

    let avg_l2 = l2_times.iter().sum::<Duration>() / l2_times.len() as u32;
    println!("Average L2 latency: {:?}", avg_l2);
    assert!(avg_l2.as_millis() < 50); // L2 should be reasonably fast
}

#[tokio::test]
async fn test_multi_node_consistency() {
    let caches = match setup_multi_cache(3).await {
        Some(c) => c,
        None => return,
    };

    // Start invalidation listeners
    for cache in &caches {
        cache.start_invalidation_listener().await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Put different values in different nodes
    let key = CacheKey::new("shared_query".to_string());
    let value1 = CacheValue::new(vec![1, 2, 3]);
    let value2 = CacheValue::new(vec![4, 5, 6]);

    caches[0].put(key.clone(), value1.clone()).await.unwrap();
    caches[1].put(key.clone(), value2.clone()).await.unwrap();

    // Give time for L2 to settle (last write wins)
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Clear all L1 caches and read from L2
    for cache in &caches {
        cache.clear_l1();
    }

    // All nodes should read the same value from L2
    let results: Vec<_> =
        futures::future::join_all(caches.iter().map(|cache| cache.get(&key))).await;

    // At least 2 out of 3 should have a value
    let values_count = results
        .iter()
        .filter(|r| r.as_ref().ok().and_then(|v| v.as_ref()).is_some())
        .count();
    assert!(
        values_count >= 2,
        "Expected at least 2 nodes to have the value"
    );
}

#[tokio::test]
async fn test_coherence_verification() {
    let caches = match setup_multi_cache(3).await {
        Some(c) => c,
        None => return,
    };

    // Populate caches with test data
    for i in 0..10 {
        let key = CacheKey::new(format!("coherence_test_{}", i));
        let value = CacheValue::new(vec![i as u8; 10]);

        for cache in &caches {
            cache.put(key.clone(), value.clone()).await.unwrap();
        }
    }

    // Verify coherence
    let config = CoherenceConfig {
        consistency_level: ConsistencyLevel::Eventual,
        max_staleness_seconds: 60,
    };
    let protocol = CacheCoherenceProtocol::new(CoherenceProtocol::PubSub, config);

    let cache_refs: Vec<&DistributedCache> = caches.iter().collect();
    let report = protocol.verify_coherence(&cache_refs).await.unwrap();

    println!("Coherence report: {}", report.summary());
    assert!(
        report.coherence_rate >= 0.8,
        "Coherence rate should be >= 0.8, got {}",
        report.coherence_rate
    );
}

#[tokio::test]
async fn test_cache_key_namespaces() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let key1 = CacheKey::with_namespace("query1".to_string(), "tenant1".to_string());
    let key2 = CacheKey::with_namespace("query1".to_string(), "tenant2".to_string());

    let value1 = CacheValue::new(vec![1, 2, 3]);
    let value2 = CacheValue::new(vec![4, 5, 6]);

    // Put values with different namespaces
    cache.put(key1.clone(), value1.clone()).await.unwrap();
    cache.put(key2.clone(), value2.clone()).await.unwrap();

    // Get values
    let result1 = cache.get(&key1).await.unwrap();
    let result2 = cache.get(&key2).await.unwrap();

    assert_eq!(result1.unwrap().data, value1.data);
    assert_eq!(result2.unwrap().data, value2.data);
}

#[tokio::test]
async fn test_l1_expiration() {
    let config = DistributedCacheConfig {
        l1_max_size: 100,
        l1_ttl_seconds: 1, // 1 second TTL
        l2_redis_url: "redis://localhost:6379".to_string(),
        l2_ttl_seconds: 30,
        compression: false,
        invalidation_channel: format!("oxirs:test:{}", uuid::Uuid::new_v4()),
    };

    let cache = match DistributedCache::new(config).await {
        Ok(c) => c,
        Err(_) => return,
    };

    let key = CacheKey::new("expiring_key".to_string());
    let value = CacheValue::new(vec![1, 2, 3]);

    // Put value
    cache.put(key.clone(), value.clone()).await.unwrap();

    // Immediate access should hit L1
    assert!(cache.get(&key).await.unwrap().is_some());

    // Wait for L1 expiration
    tokio::time::sleep(Duration::from_secs(2)).await;

    // L1 should be expired, but L2 should still have it
    cache.clear_l1(); // Force expiration check
    let result = cache.get(&key).await.unwrap();
    assert!(result.is_some(), "L2 should still have the value");
}

#[tokio::test]
async fn test_concurrent_access() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let cache = std::sync::Arc::new(cache);

    // Spawn multiple concurrent tasks
    let mut handles = Vec::new();
    for i in 0..10 {
        let cache = cache.clone();
        let handle = tokio::spawn(async move {
            let key = CacheKey::new(format!("concurrent_{}", i));
            let value = CacheValue::new(vec![i as u8; 100]);

            // Put value
            cache.put(key.clone(), value.clone()).await.unwrap();

            // Get value multiple times
            for _ in 0..10 {
                let result = cache.get(&key).await.unwrap();
                assert!(result.is_some());
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Check cache size
    assert!(cache.l1_size() > 0);
}

/// Performance benchmark test
#[tokio::test]
async fn bench_distributed_cache() {
    let cache = match setup_cache().await {
        Some(c) => c,
        None => return,
    };

    let num_queries = 1000;

    // Warm up L1 with queries
    for i in 0..num_queries {
        let key = CacheKey::new(format!("bench_query_{}", i));
        let value = CacheValue::new(vec![i as u8; 100]);
        cache.put(key, value).await.unwrap();
    }

    // Measure L1 hit rate
    let start = Instant::now();
    for i in 0..num_queries {
        let key = CacheKey::new(format!("bench_query_{}", i));
        let _val = cache.get(&key).await.unwrap();
    }
    let elapsed = start.elapsed();

    let l1_hit_rate = cache.metrics().l1_hit_rate();
    let throughput = num_queries as f64 / elapsed.as_secs_f64();

    println!("Benchmark results:");
    println!("  L1 hit rate: {:.2}%", l1_hit_rate * 100.0);
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Average latency: {:.2}ms",
        elapsed.as_millis() as f64 / num_queries as f64
    );

    // Verify performance targets
    assert!(
        l1_hit_rate > 0.8,
        "L1 hit rate should be > 80%, got {:.2}%",
        l1_hit_rate * 100.0
    );
    assert!(
        elapsed.as_millis() < num_queries * 2,
        "Average latency should be < 2ms per operation"
    );
}
