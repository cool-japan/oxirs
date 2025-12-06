//! Query Result Caching Demo
//!
//! Demonstrates the query result caching capabilities introduced in Beta.2++++:
//! - Fingerprint-based caching with structural query hashing
//! - LRU eviction policy with configurable cache size
//! - TTL-based expiration for cache freshness
//! - Optional compression to reduce memory footprint
//! - Comprehensive statistics tracking (hit rate, evictions, size)
//!
//! Run with:
//! ```bash
//! cargo run --example query_caching_demo --all-features
//! ```

use oxirs_arq::query_fingerprinting::{FingerprintConfig, QueryFingerprinter};
use oxirs_arq::query_result_cache::{CacheConfig, QueryResultCache};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Mock query result for demonstration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct QueryResult {
    bindings: Vec<(String, String)>,
    row_count: usize,
}

impl QueryResult {
    fn new(bindings: Vec<(String, String)>) -> Self {
        let row_count = bindings.len();
        Self {
            bindings,
            row_count,
        }
    }

    fn mock_query_execution(query: &str) -> Self {
        // Simulate query execution with different result sizes
        let binding_count = (query.len() % 50) + 1;
        let bindings: Vec<_> = (0..binding_count)
            .map(|i| {
                (
                    format!("?var{}", i),
                    format!("http://example.org/resource/{}", i),
                )
            })
            .collect();

        Self::new(bindings)
    }

    fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).unwrap()
    }
}

fn main() {
    println!("🚀 OxiRS ARQ - Query Result Caching Demo (Beta.2++++)\n");
    println!("This demo showcases the new query result caching capabilities.\n");

    // Demo 1: Basic Caching
    println!("═══════════════════════════════════════════════════════════");
    println!("Demo 1: Basic Query Result Caching");
    println!("═══════════════════════════════════════════════════════════\n");
    demo_basic_caching();

    // Demo 2: Cache Configuration and Statistics
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Demo 2: Cache Configuration and Statistics");
    println!("═══════════════════════════════════════════════════════════\n");
    demo_cache_configuration();

    // Demo 3: TTL-based Expiration
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Demo 3: TTL-based Cache Expiration");
    println!("═══════════════════════════════════════════════════════════\n");
    demo_ttl_expiration();

    // Demo 4: LRU Eviction
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Demo 4: LRU Eviction Policy");
    println!("═══════════════════════════════════════════════════════════\n");
    demo_lru_eviction();

    // Demo 5: Cache Invalidation
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Demo 5: Cache Invalidation");
    println!("═══════════════════════════════════════════════════════════\n");
    demo_invalidation();

    // Demo 6: Performance Comparison
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Demo 6: Performance Comparison");
    println!("═══════════════════════════════════════════════════════════\n");
    demo_performance();

    println!("\n✅ All demos completed successfully!");
    println!("\n💡 Key Takeaways:");
    println!("   • Query result caching dramatically improves performance");
    println!("   • Fingerprint-based caching handles query equivalence");
    println!("   • TTL ensures cache freshness for dynamic data");
    println!("   • LRU eviction prevents unbounded memory growth");
    println!("   • Statistics help monitor cache effectiveness");
}

fn demo_basic_caching() {
    // Initialize cache with default configuration
    let cache = QueryResultCache::new(CacheConfig::default());
    let fingerprinter = QueryFingerprinter::new(FingerprintConfig::default());

    // Simulate query execution
    let query1 = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100";
    let fingerprint1 = fingerprinter.fingerprint(query1).unwrap();
    let fp_hash = fingerprint1.hash.clone();

    println!("Executing query:\n  {}", query1);
    println!("Query fingerprint: {}", fp_hash);

    // First execution - cache miss
    println!("\nFirst execution:");
    let result1 = match cache.get(&fp_hash) {
        Some(cached) => {
            println!("✓ Cache hit!");
            QueryResult::from_bytes(&cached)
        }
        None => {
            println!("✗ Cache miss - executing query...");
            let result = QueryResult::mock_query_execution(query1);
            println!("  Fetched {} bindings", result.row_count);

            // Store in cache
            cache.put(fp_hash.clone(), result.to_bytes()).unwrap();
            println!("✓ Result cached");

            result
        }
    };

    // Second execution - cache hit
    println!("\nSecond execution (same query):");
    let result2 = match cache.get(&fp_hash) {
        Some(cached) => {
            println!("✓ Cache hit! Retrieved from cache (no query execution)");
            QueryResult::from_bytes(&cached)
        }
        None => {
            println!("✗ Unexpected cache miss");
            QueryResult::mock_query_execution(query1)
        }
    };

    assert_eq!(result1, result2, "Cached result should match original");
    println!("✓ Results verified to match");

    print_stats(&cache);
}

fn demo_cache_configuration() {
    // Create cache with custom configuration
    let config = CacheConfig {
        max_entries: 50,
        ttl: Duration::from_secs(300), // 5 minutes
        enable_compression: true,
        max_result_size: 10 * 1024 * 1024, // 10 MB
        enable_stats: true,
        eviction_batch_size: 10,
    };

    println!("Custom Cache Configuration:");
    println!("  • Max entries: {}", config.max_entries);
    println!("  • TTL: {} seconds", config.ttl.as_secs());
    println!("  • Compression: enabled");
    println!(
        "  • Max result size: {} MB",
        config.max_result_size / (1024 * 1024)
    );

    let cache = QueryResultCache::new(config);
    let fingerprinter = QueryFingerprinter::new(FingerprintConfig::default());

    // Execute multiple queries
    println!("\nCaching 10 different queries...");
    for i in 0..10 {
        let query = format!(
            "SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER(?x = {}) }} LIMIT 100",
            i
        );
        let fingerprint = fingerprinter.fingerprint(&query).unwrap();

        if cache.get(&fingerprint.hash).is_none() {
            let result = QueryResult::mock_query_execution(&query);
            cache
                .put(fingerprint.hash.clone(), result.to_bytes())
                .unwrap();
        }
    }

    println!("✓ All queries cached");
    print_stats(&cache);
}

fn demo_ttl_expiration() {
    // Create cache with short TTL for demonstration
    let config = CacheConfig {
        max_entries: 100,
        ttl: Duration::from_secs(2), // 2 seconds TTL
        enable_compression: false,
        max_result_size: 10 * 1024 * 1024,
        enable_stats: true,
        eviction_batch_size: 10,
    };

    let cache = QueryResultCache::new(config);
    let fingerprinter = QueryFingerprinter::new(FingerprintConfig::default());

    let query = "SELECT ?s WHERE { ?s a :Person } LIMIT 50";
    let fingerprint = fingerprinter.fingerprint(query).unwrap();

    // First execution
    println!("Caching query result with 2-second TTL...");
    let result = QueryResult::mock_query_execution(query);
    cache
        .put(fingerprint.hash.clone(), result.to_bytes())
        .unwrap();
    println!("✓ Result cached");

    // Immediate retrieval - should hit
    println!("\nImmediate retrieval:");
    if cache.get(&fingerprint.hash).is_some() {
        println!("✓ Cache hit (entry still fresh)");
    } else {
        println!("✗ Unexpected cache miss");
    }

    // Wait for TTL to expire
    println!("\nWaiting for TTL to expire (2 seconds)...");
    std::thread::sleep(Duration::from_millis(2500));

    // Retrieval after TTL - should miss
    println!("Retrieval after TTL:");
    match cache.get(&fingerprint.hash) {
        Some(_) => println!("✗ Unexpected cache hit (TTL should have expired)"),
        None => println!("✓ Cache miss (entry expired as expected)"),
    }

    print_stats(&cache);
}

fn demo_lru_eviction() {
    // Create cache with small capacity to trigger eviction
    let config = CacheConfig {
        max_entries: 5, // Very small to demonstrate eviction
        ttl: Duration::from_secs(3600),
        enable_compression: false,
        max_result_size: 10 * 1024 * 1024,
        enable_stats: true,
        eviction_batch_size: 1,
    };

    let cache = QueryResultCache::new(config);
    let fingerprinter = QueryFingerprinter::new(FingerprintConfig::default());

    println!("Cache capacity: 5 entries");
    println!("\nAdding 7 queries to trigger LRU eviction...");

    let queries = (0..7)
        .map(|i| format!("SELECT ?s WHERE {{ ?s a :Type{} }}", i))
        .collect::<Vec<_>>();

    let fingerprints: Vec<_> = queries
        .iter()
        .map(|q| fingerprinter.fingerprint(q).unwrap())
        .collect();

    for (i, (query, fp)) in queries.iter().zip(fingerprints.iter()).enumerate() {
        let result = QueryResult::mock_query_execution(query);
        cache.put(fp.hash.clone(), result.to_bytes()).unwrap();
        println!(
            "  Added query {} (fingerprint: {}...)",
            i + 1,
            &fp.hash[..16]
        );
    }

    println!("\n✓ Added 7 queries (5 max capacity)");
    print_stats(&cache);

    // Check which entries survived
    println!("\nChecking which entries are still cached:");
    for (i, fp) in fingerprints.iter().enumerate() {
        if cache.contains(&fp.hash) {
            println!("  ✓ Query {} still in cache", i + 1);
        } else {
            println!("  ✗ Query {} evicted", i + 1);
        }
    }
}

fn demo_invalidation() {
    let cache = QueryResultCache::new(CacheConfig::default());
    let fingerprinter = QueryFingerprinter::new(FingerprintConfig::default());

    // Add multiple queries to cache
    println!("Caching 5 different queries...");
    let queries = [
        "SELECT ?s WHERE { ?s a :Person }",
        "SELECT ?s WHERE { ?s a :Organization }",
        "SELECT ?s WHERE { ?s a :Place }",
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
        "SELECT ?s WHERE { ?s :name ?name } ORDER BY ?name",
    ];

    let fingerprints: Vec<_> = queries
        .iter()
        .map(|q| fingerprinter.fingerprint(q).unwrap())
        .collect();

    for (query, fp) in queries.iter().zip(fingerprints.iter()) {
        let result = QueryResult::mock_query_execution(query);
        cache.put(fp.hash.clone(), result.to_bytes()).unwrap();
    }

    println!("✓ All queries cached");
    println!("Cache size: {} entries", cache.size());

    // Selective invalidation
    println!("\nInvalidating first query...");
    cache.invalidate(&fingerprints[0].hash).unwrap();
    println!("✓ Invalidated. Cache size: {} entries", cache.size());

    // Verify invalidation
    if cache.contains(&fingerprints[0].hash) {
        println!("✗ Entry still present (unexpected)");
    } else {
        println!("✓ Entry successfully removed");
    }

    // Global clear
    println!("\nClearing entire cache...");
    cache.invalidate_all().unwrap();
    println!("✓ Cache cleared. Size: {} entries", cache.size());

    print_stats(&cache);
}

fn demo_performance() {
    let cache = QueryResultCache::new(CacheConfig::default());
    let fingerprinter = QueryFingerprinter::new(FingerprintConfig::default());

    let query =
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o . ?s :created ?date } ORDER BY DESC(?date) LIMIT 1000";
    let fingerprint = fingerprinter.fingerprint(query).unwrap();

    // Measure uncached execution time (simulated)
    println!("Simulating query execution without cache...");
    let start = std::time::Instant::now();
    let result = QueryResult::mock_query_execution(query);
    let uncached_time = start.elapsed();
    println!(
        "  Execution time: {}ms (simulated slow query)",
        uncached_time.as_millis()
    );
    println!("  Result size: {} bindings", result.row_count);

    // Cache the result
    cache
        .put(fingerprint.hash.clone(), result.to_bytes())
        .unwrap();
    println!("✓ Result cached");

    // Measure cached retrieval time
    println!("\nRetrieving from cache...");
    let start = std::time::Instant::now();
    let cached_data = cache.get(&fingerprint.hash);
    let cached_time = start.elapsed();

    assert!(cached_data.is_some(), "Cache should have the result");
    let cached_result = QueryResult::from_bytes(&cached_data.unwrap());
    println!("  Retrieval time: {}µs", cached_time.as_micros());
    println!("  Result size: {} bindings", cached_result.row_count);

    // Calculate improvement
    let speedup = uncached_time.as_micros() as f64 / cached_time.as_micros().max(1) as f64;
    println!("\n📊 Performance Improvement:");
    println!("  • Uncached: {}ms", uncached_time.as_millis());
    println!("  • Cached: {}µs", cached_time.as_micros());
    println!("  • Speedup: {:.1}x faster", speedup);

    print_stats(&cache);
}

fn print_stats(cache: &QueryResultCache) {
    let stats = cache.statistics();

    println!("\n📊 Cache Statistics:");
    let total_requests = stats.hits + stats.misses;
    println!("  • Total requests: {}", total_requests);
    println!("  • Cache hits: {}", stats.hits);
    println!("  • Cache misses: {}", stats.misses);
    println!(
        "  • Hit rate: {:.2}%",
        if total_requests > 0 {
            (stats.hits as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("  • Current entries: {}", stats.entry_count);
    println!(
        "  • Cache size: {} bytes ({:.2} KB)",
        stats.size_bytes,
        stats.size_bytes as f64 / 1024.0
    );
    println!("  • Total evictions: {}", stats.evictions);
    if stats.avg_result_size > 0 {
        println!("  • Avg result size: {} bytes", stats.avg_result_size);
    }
    if stats.compression_ratio > 0.0 {
        println!("  • Compression ratio: {:.2}x", stats.compression_ratio);
    }
}
