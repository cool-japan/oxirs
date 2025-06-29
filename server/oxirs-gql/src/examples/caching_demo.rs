//! Distributed Caching Demo
//!
//! This example demonstrates the distributed caching capabilities with Redis
//! integration, compression, encryption, and performance monitoring.

use anyhow::Result;
use oxirs_gql::{
    distributed_cache::{
        GraphQLQueryCache, CacheConfig, CompressionType, EncryptionType,
        ShardingStrategy, EvictionPolicy, CacheStats,
    },
    GraphQLConfig, GraphQLServer, RdfStore,
};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting OxiRS GraphQL Distributed Caching Demo");

    // Create RDF store
    let store = Arc::new(RdfStore::new()?);
    load_cache_test_data(&store).await?;

    // Configure distributed caching
    let cache_config = CacheConfig {
        redis_urls: vec![
            "redis://localhost:6379".to_string(),
            "redis://localhost:6380".to_string(), // Optional second Redis instance
        ],
        default_ttl: Duration::from_secs(3600), // 1 hour default TTL
        max_local_cache_size: 1000,
        compression_type: CompressionType::Gzip,
        compression_level: 6,
        encryption_type: EncryptionType::AES256,
        encryption_key: Some("my-secret-key-32-bytes-long!!!".to_string()),
        sharding_strategy: ShardingStrategy::ConsistentHash,
        eviction_policy: EvictionPolicy::LRU,
        enable_local_cache: true,
        enable_write_through: true,
        enable_write_behind: false,
        enable_metrics: true,
        circuit_breaker_enabled: true,
        circuit_breaker_failure_threshold: 5,
        circuit_breaker_recovery_time: Duration::from_secs(30),
        batch_size: 100,
        connection_pool_size: 10,
        connection_timeout: Duration::from_secs(5),
        operation_timeout: Duration::from_secs(2),
    };

    // Create distributed cache
    let cache = Arc::new(GraphQLQueryCache::new(cache_config.clone()).await?);

    // Demo 1: Basic caching operations
    demo_basic_caching(&cache).await?;

    // Demo 2: Query result caching
    demo_query_caching(&cache, &store).await?;

    // Demo 3: Performance comparison with/without cache
    demo_performance_comparison(&cache, &store).await?;

    // Demo 4: Cache statistics and monitoring
    demo_cache_monitoring(&cache).await?;

    // Demo 5: Advanced caching strategies
    demo_advanced_strategies(&cache).await?;

    // Create GraphQL server with caching enabled
    let server_config = GraphQLConfig {
        enable_introspection: true,
        enable_playground: true,
        max_query_depth: Some(10),
        max_query_complexity: Some(1000),
        enable_query_validation: true,
        distributed_cache_config: Some(cache_config),
    };

    let server = GraphQLServer::new(store.clone())
        .with_config(server_config);

    info!("GraphQL server with distributed caching configured");
    info!("GraphQL Playground available at http://127.0.0.1:4000/playground");
    info!("Cache statistics endpoint at http://127.0.0.1:4000/cache/stats");

    // Start the server
    server.start("127.0.0.1:4000").await?;

    Ok(())
}

/// Load test data for caching demonstrations
async fn load_cache_test_data(store: &Arc<RdfStore>) -> Result<()> {
    info!("Loading cache test dataset...");
    
    let mut store_mut = RdfStore::new()?;

    // Add sample data that will be frequently queried
    for i in 1..=50 {
        store_mut.insert_triple(
            &format!("http://example.org/user/{}", i),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/User",
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/user/{}", i),
            "http://example.org/name",
            &format!("\"User {}\"", i),
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/user/{}", i),
            "http://example.org/email",
            &format!("\"user{}@example.org\"", i),
        )?;
        
        // Add some users to groups
        let group_id = (i % 5) + 1;
        store_mut.insert_triple(
            &format!("http://example.org/user/{}", i),
            "http://example.org/memberOf",
            &format!("http://example.org/group/{}", group_id),
        )?;
    }

    // Add groups
    for i in 1..=5 {
        store_mut.insert_triple(
            &format!("http://example.org/group/{}", i),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Group",
        )?;
        
        store_mut.insert_triple(
            &format!("http://example.org/group/{}", i),
            "http://example.org/name",
            &format!("\"Group {}\"", i),
        )?;
    }

    info!("Cache test dataset loaded: 50 users, 5 groups");
    Ok(())
}

/// Demonstrate basic caching operations
async fn demo_basic_caching(cache: &GraphQLQueryCache) -> Result<()> {
    info!("=== Demo 1: Basic Caching Operations ===");

    // Test basic set/get operations
    let test_key = "demo:basic:test";
    let test_value = b"Hello, distributed cache!";

    info!("Setting cache value...");
    cache.set(test_key, test_value.to_vec(), Some(Duration::from_secs(300))).await?;

    info!("Getting cache value...");
    if let Some(cached_value) = cache.get(test_key).await? {
        let cached_str = String::from_utf8_lossy(&cached_value);
        info!("Retrieved from cache: {}", cached_str);
    } else {
        warn!("Value not found in cache");
    }

    // Test TTL
    info!("Testing TTL with short expiration...");
    cache.set("demo:ttl:test", b"short-lived".to_vec(), Some(Duration::from_secs(2))).await?;
    
    // Immediate retrieval should succeed
    if cache.get("demo:ttl:test").await?.is_some() {
        info!("✓ Value retrieved immediately after setting");
    }

    // Wait for expiration
    tokio::time::sleep(Duration::from_secs(3)).await;
    
    if cache.get("demo:ttl:test").await?.is_none() {
        info!("✓ Value correctly expired after TTL");
    }

    // Test deletion
    cache.set("demo:delete:test", b"to-be-deleted".to_vec(), None).await?;
    cache.delete("demo:delete:test").await?;
    
    if cache.get("demo:delete:test").await?.is_none() {
        info!("✓ Value correctly deleted");
    }

    Ok(())
}

/// Demonstrate query result caching
async fn demo_query_caching(cache: &GraphQLQueryCache, store: &Arc<RdfStore>) -> Result<()> {
    info!("=== Demo 2: Query Result Caching ===");

    let test_query = r#"
    query CachedUserQuery {
        users(limit: 10) {
            id
            name
            email
            groups {
                id
                name
            }
        }
    }
    "#;

    // Simulate query execution and caching
    let query_key = cache.generate_query_key(test_query, &std::collections::HashMap::new());
    
    info!("Executing query (first time - will be cached)...");
    let start_time = std::time::Instant::now();
    
    // Simulate query execution
    let query_result = simulate_query_execution(store, test_query).await?;
    let execution_time = start_time.elapsed();
    
    // Cache the result
    let serialized_result = serde_json::to_vec(&query_result)?;
    cache.set(&query_key, serialized_result, Some(Duration::from_secs(600))).await?;
    
    info!("Query executed and cached in {:?}", execution_time);

    // Retrieve from cache
    info!("Retrieving query result from cache...");
    let cache_start = std::time::Instant::now();
    
    if let Some(cached_result) = cache.get(&query_key).await? {
        let cache_time = cache_start.elapsed();
        let _result: serde_json::Value = serde_json::from_slice(&cached_result)?;
        
        info!("Query result retrieved from cache in {:?}", cache_time);
        info!("Cache retrieval was {:.1}x faster", 
              execution_time.as_nanos() as f64 / cache_time.as_nanos() as f64);
    }

    Ok(())
}

/// Demonstrate performance comparison
async fn demo_performance_comparison(cache: &GraphQLQueryCache, store: &Arc<RdfStore>) -> Result<()> {
    info!("=== Demo 3: Performance Comparison ===");

    let queries = vec![
        "query Q1 { users(limit: 5) { id name } }",
        "query Q2 { users { id email groups { name } } }",
        "query Q3 { groups { id name users(limit: 3) { name } } }",
    ];

    for (i, query) in queries.iter().enumerate() {
        info!("Testing query {}: {}", i + 1, query);
        
        // First execution (no cache)
        let start = std::time::Instant::now();
        let result = simulate_query_execution(store, query).await?;
        let uncached_time = start.elapsed();
        
        // Cache the result
        let query_key = cache.generate_query_key(query, &std::collections::HashMap::new());
        let serialized = serde_json::to_vec(&result)?;
        cache.set(&query_key, serialized, Some(Duration::from_secs(300))).await?;
        
        // Second execution (with cache)
        let start = std::time::Instant::now();
        let _cached_result = cache.get(&query_key).await?;
        let cached_time = start.elapsed();
        
        info!("  Uncached: {:?}, Cached: {:?}, Speedup: {:.1}x",
              uncached_time, cached_time,
              uncached_time.as_nanos() as f64 / cached_time.as_nanos() as f64);
    }

    Ok(())
}

/// Demonstrate cache monitoring and statistics
async fn demo_cache_monitoring(cache: &GraphQLQueryCache) -> Result<()> {
    info!("=== Demo 4: Cache Monitoring ===");

    // Perform some cache operations to generate statistics
    for i in 0..20 {
        let key = format!("demo:stats:key:{}", i);
        let value = format!("value_{}", i).into_bytes();
        
        cache.set(&key, value, Some(Duration::from_secs(60))).await?;
        
        // Randomly access some keys to generate hits/misses
        if i % 3 == 0 {
            let _ = cache.get(&format!("demo:stats:key:{}", i / 2)).await?;
        }
        
        // Try to access non-existent keys
        if i % 4 == 0 {
            let _ = cache.get(&format!("demo:stats:nonexistent:{}", i)).await?;
        }
    }

    // Get cache statistics
    let stats = cache.get_stats().await?;
    
    info!("Cache Statistics:");
    info!("  Total operations: {}", stats.total_operations);
    info!("  Cache hits: {}", stats.hits);
    info!("  Cache misses: {}", stats.misses);
    info!("  Hit rate: {:.2}%", stats.hit_rate * 100.0);
    info!("  Local cache size: {}", stats.local_cache_size);
    info!("  Memory usage: {} bytes", stats.memory_usage_bytes);
    info!("  Average operation time: {:.2}ms", stats.average_operation_time_ms);
    
    if let Some(redis_stats) = &stats.redis_stats {
        info!("  Redis connections: {}", redis_stats.active_connections);
        info!("  Redis operations: {}", redis_stats.total_operations);
        info!("  Redis errors: {}", redis_stats.connection_errors);
    }

    Ok(())
}

/// Demonstrate advanced caching strategies
async fn demo_advanced_strategies(cache: &GraphQLQueryCache) -> Result<()> {
    info!("=== Demo 5: Advanced Caching Strategies ===");

    // Test batch operations
    info!("Testing batch operations...");
    let batch_keys: Vec<String> = (0..10).map(|i| format!("batch:key:{}", i)).collect();
    let batch_values: Vec<Vec<u8>> = (0..10).map(|i| format!("batch_value_{}", i).into_bytes()).collect();
    
    // Batch set
    let pairs: Vec<(String, Vec<u8>)> = batch_keys.iter().cloned()
        .zip(batch_values.iter().cloned()).collect();
    
    cache.set_batch(pairs, Some(Duration::from_secs(300))).await?;
    info!("✓ Batch set completed");
    
    // Batch get
    let retrieved = cache.get_batch(&batch_keys).await?;
    info!("✓ Batch get completed: {}/{} keys retrieved", retrieved.len(), batch_keys.len());

    // Test cache warming
    info!("Testing cache warming...");
    let warm_keys = vec![
        ("warm:frequent:1", "frequently_accessed_data_1"),
        ("warm:frequent:2", "frequently_accessed_data_2"),
        ("warm:frequent:3", "frequently_accessed_data_3"),
    ];
    
    for (key, value) in warm_keys {
        cache.set(key, value.as_bytes().to_vec(), Some(Duration::from_secs(3600))).await?;
    }
    info!("✓ Cache warming completed");

    // Test invalidation patterns
    info!("Testing cache invalidation...");
    
    // Set some data with tags
    cache.set_with_tags("tagged:user:1", b"user_data_1".to_vec(), 
                       vec!["user".to_string(), "profile".to_string()], 
                       Some(Duration::from_secs(300))).await?;
    
    cache.set_with_tags("tagged:user:2", b"user_data_2".to_vec(), 
                       vec!["user".to_string(), "profile".to_string()], 
                       Some(Duration::from_secs(300))).await?;
    
    // Invalidate by tag
    cache.invalidate_by_tag("user").await?;
    info!("✓ Tag-based invalidation completed");

    // Test circuit breaker (simulate failures)
    info!("Testing circuit breaker resilience...");
    
    // This would normally test failure scenarios, but for demo purposes
    // we'll just show the circuit breaker is configured
    if let Some(cb_stats) = cache.get_circuit_breaker_stats().await? {
        info!("  Circuit breaker state: {:?}", cb_stats.state);
        info!("  Failure count: {}", cb_stats.failure_count);
        info!("  Success count: {}", cb_stats.success_count);
        info!("✓ Circuit breaker is active and monitoring");
    }

    Ok(())
}

/// Simulate query execution (in real implementation, this would execute against the store)
async fn simulate_query_execution(_store: &Arc<RdfStore>, query: &str) -> Result<serde_json::Value> {
    // Simulate some processing time
    tokio::time::sleep(Duration::from_millis(10 + query.len() as u64)).await;
    
    // Return mock query result
    Ok(serde_json::json!({
        "data": {
            "query": query,
            "results": [
                {"id": "1", "name": "Sample Result 1"},
                {"id": "2", "name": "Sample Result 2"}
            ]
        }
    }))
}