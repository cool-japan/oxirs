//! Demonstration of the Query Result Cache
//!
//! This example shows how to use the QueryResultCache for caching SPARQL query results
//! with TTL expiration and LRU eviction.

use oxirs_core::query::result_cache::{CacheConfig, QueryResultCache};
use std::time::Duration;

fn main() {
    println!("🚀 OxiRS Query Result Cache Demo\n");
    println!("{}", "=".repeat(60));

    // Create a cache with custom configuration
    let config = CacheConfig {
        max_entries: 100,
        max_memory_bytes: 10 * 1024 * 1024,   // 10 MB
        default_ttl: Duration::from_secs(60), // 1 minute
        enable_lru: true,
    };

    let cache = QueryResultCache::<Vec<String>>::new(config);

    println!("\n📊 Cache Configuration:");
    println!("  - Max entries: 100");
    println!("  - Max memory: 10 MB");
    println!("  - Default TTL: 60 seconds");
    println!("  - LRU eviction: enabled");

    // Example 1: Basic caching
    println!("\n\n1️⃣ Basic Query Caching");
    println!("{}", "=".repeat(60));

    let query1 = "SELECT * WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }".to_string();
    let results1 = vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
    ];

    cache.put(query1.clone(), results1.clone());
    println!("✅ Cached query: SELECT * WHERE {{ ?s foaf:name ?name }}");
    println!("   Results: {:?}", results1);

    // Retrieve from cache
    if let Some(cached_results) = cache.get(&query1) {
        println!("✅ Cache hit! Retrieved {} results", cached_results.len());
        println!("   Results: {:?}", cached_results);
    } else {
        println!("❌ Cache miss");
    }

    // Example 2: Cache statistics
    println!("\n\n2️⃣ Cache Statistics");
    println!("{}", "=".repeat(60));

    let stats = cache.stats();
    println!("📈 Cache Stats:");
    println!(
        "  - Hits: {}",
        stats.hits.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  - Misses: {}",
        stats.misses.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!("  - Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    println!("  - Cache size: {} entries", cache.len());
    println!("  - Memory usage: {} bytes", cache.memory_usage());

    // Example 3: Custom TTL
    println!("\n\n3️⃣ Custom TTL (Time-To-Live)");
    println!("{}", "=".repeat(60));

    let query2 = "SELECT * WHERE { ?s a <http://xmlns.com/foaf/0.1/Person> }".to_string();
    let results2 = vec!["Person1".to_string(), "Person2".to_string()];

    cache.put_with_ttl(query2.clone(), results2.clone(), Duration::from_millis(100));
    println!("✅ Cached query with TTL=100ms");

    // Immediate retrieval should work
    if cache.get(&query2).is_some() {
        println!("✅ Immediate retrieval successful");
    }

    // Wait for expiration
    println!("⏳ Waiting for expiration (150ms)...");
    std::thread::sleep(Duration::from_millis(150));

    if cache.get(&query2).is_none() {
        println!("✅ Entry expired as expected");
    }

    let expirations = cache
        .stats()
        .expirations
        .load(std::sync::atomic::Ordering::Relaxed);
    println!("📊 Total expirations: {}", expirations);

    // Example 4: LRU Eviction
    println!("\n\n4️⃣ LRU Eviction");
    println!("{}", "=".repeat(60));

    // Create a smaller cache
    let small_config = CacheConfig {
        max_entries: 3,
        max_memory_bytes: 10 * 1024 * 1024,
        default_ttl: Duration::from_secs(300),
        enable_lru: true,
    };
    let small_cache = QueryResultCache::<String>::new(small_config);

    // Fill the cache
    small_cache.put("query1".to_string(), "result1".to_string());
    small_cache.put("query2".to_string(), "result2".to_string());
    small_cache.put("query3".to_string(), "result3".to_string());
    println!("✅ Filled cache with 3 entries");

    // Access query1 to make it most recently used
    small_cache.get("query1");
    println!("✅ Accessed query1 (now most recently used)");

    // Add query4, should evict query2 (least recently used)
    small_cache.put("query4".to_string(), "result4".to_string());
    println!("✅ Added query4 (triggers eviction)");

    // Check what's in the cache
    println!("\n📋 Cache contents after eviction:");
    println!(
        "  - query1: {}",
        if small_cache.get("query1").is_some() {
            "✅ present"
        } else {
            "❌ evicted"
        }
    );
    println!(
        "  - query2: {}",
        if small_cache.get("query2").is_some() {
            "✅ present"
        } else {
            "❌ evicted (LRU)"
        }
    );
    println!(
        "  - query3: {}",
        if small_cache.get("query3").is_some() {
            "✅ present"
        } else {
            "❌ evicted"
        }
    );
    println!(
        "  - query4: {}",
        if small_cache.get("query4").is_some() {
            "✅ present"
        } else {
            "❌ evicted"
        }
    );

    let evictions = small_cache
        .stats()
        .evictions
        .load(std::sync::atomic::Ordering::Relaxed);
    println!("\n📊 Total evictions: {}", evictions);

    // Example 5: Cache invalidation
    println!("\n\n5️⃣ Manual Cache Invalidation");
    println!("{}", "=".repeat(60));

    cache.put("to_invalidate".to_string(), vec!["data".to_string()]);
    println!("✅ Added entry to cache");

    if cache.invalidate("to_invalidate") {
        println!("✅ Successfully invalidated entry");
    }

    if cache.get("to_invalidate").is_none() {
        println!("✅ Entry no longer in cache");
    }

    let invalidations = cache
        .stats()
        .invalidations
        .load(std::sync::atomic::Ordering::Relaxed);
    println!("📊 Total invalidations: {}", invalidations);

    // Example 6: Clear cache
    println!("\n\n6️⃣ Clear Cache");
    println!("{}", "=".repeat(60));

    println!("📊 Cache size before clear: {} entries", cache.len());
    cache.clear();
    println!("✅ Cache cleared");
    println!("📊 Cache size after clear: {} entries", cache.len());

    // Final statistics
    println!("\n\n📊 Final Cache Statistics");
    println!("{}", "=".repeat(60));
    let final_stats = cache.stats();
    println!(
        "  - Total hits: {}",
        final_stats.hits.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  - Total misses: {}",
        final_stats
            .misses
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  - Total puts: {}",
        final_stats.puts.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  - Total evictions: {}",
        final_stats
            .evictions
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  - Total expirations: {}",
        final_stats
            .expirations
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  - Total invalidations: {}",
        final_stats
            .invalidations
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "  - Overall hit rate: {:.2}%",
        final_stats.hit_rate() * 100.0
    );

    println!("\n✅ Demo completed successfully!");
}
