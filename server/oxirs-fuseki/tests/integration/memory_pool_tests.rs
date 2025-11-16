// Integration tests for memory pooling and optimization (memory_pool.rs)

use oxirs_fuseki::memory_pool::{
    MemoryPool, BufferPool, QueryContextPool, MemoryPressureMonitor,
    ChunkedArrayManager, GarbageCollector, PoolStatistics,
};
use scirs2_core::memory::{BufferPool as SciRSBufferPool, GlobalBufferPool};
use scirs2_core::memory_efficient::{ChunkedArray, AdaptiveChunking};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Duration;

#[tokio::test]
async fn test_buffer_pool_basic() {
    let pool = BufferPool::new(10, 1024);

    // Acquire buffer
    let buffer1 = pool.acquire().await.unwrap();
    assert_eq!(buffer1.len(), 1024);

    // Release buffer
    pool.release(buffer1).await;

    // Acquire again - should reuse
    let buffer2 = pool.acquire().await.unwrap();
    assert_eq!(buffer2.len(), 1024);

    let stats = pool.statistics().await;
    assert!(stats.total_allocated > 0);
    assert!(stats.pool_hits > 0);
}

#[tokio::test]
async fn test_buffer_pool_concurrent_access() {
    let pool = Arc::new(BufferPool::new(10, 1024));

    // Spawn concurrent tasks
    let mut handles = Vec::new();
    for i in 0..20 {
        let p = pool.clone();
        let handle = tokio::spawn(async move {
            let buffer = p.acquire().await.unwrap();
            // Simulate work
            tokio::time::sleep(Duration::from_millis(10)).await;
            p.release(buffer).await;
            i
        });
        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        handle.await.unwrap();
    }

    let stats = pool.statistics().await;
    assert!(stats.total_allocated >= 10);
    assert!(stats.pool_hits > 0);
}

#[tokio::test]
async fn test_query_context_pool() {
    let pool = QueryContextPool::new(5);

    // Acquire contexts
    let ctx1 = pool.acquire().await.unwrap();
    let ctx2 = pool.acquire().await.unwrap();

    // Verify they're different
    assert_ne!(ctx1.id(), ctx2.id());

    // Release
    pool.release(ctx1).await;
    pool.release(ctx2).await;

    // Acquire again - should reuse
    let ctx3 = pool.acquire().await.unwrap();

    let stats = pool.statistics().await;
    assert_eq!(stats.total_contexts, 3);
    assert!(stats.reuse_count > 0);
}

#[tokio::test]
async fn test_memory_pressure_monitor() {
    let monitor = MemoryPressureMonitor::new(0.8);

    // Initially no pressure
    assert!(!monitor.is_under_pressure().await);

    // Simulate memory usage increase
    monitor.update_memory_usage(0.85).await;

    // Should detect pressure
    assert!(monitor.is_under_pressure().await);

    // Get recommendations
    let recommendations = monitor.get_recommendations().await;
    assert!(!recommendations.is_empty());
}

#[tokio::test]
async fn test_memory_pressure_adaptive_behavior() {
    let monitor = Arc::new(MemoryPressureMonitor::new(0.7));
    let pool = Arc::new(BufferPool::new(100, 1024));

    // Start background pressure monitoring
    let m = monitor.clone();
    let p = pool.clone();
    tokio::spawn(async move {
        loop {
            if m.is_under_pressure().await {
                // Reduce pool size under pressure
                p.shrink(50).await;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    // Simulate high memory usage
    monitor.update_memory_usage(0.85).await;

    // Wait for adaptive response
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stats = pool.statistics().await;
    // Pool should have been shrunk
    assert!(stats.current_size < 100);
}

#[tokio::test]
async fn test_chunked_array_manager() {
    let manager = ChunkedArrayManager::new(1024, 1024 * 1024); // 1KB chunks, 1MB max

    // Create chunked array
    let array_id = manager.create_array(10000).await.unwrap();

    // Write data
    let data = vec![1.0f32; 1000];
    manager.write_chunk(array_id, 0, &data).await.unwrap();

    // Read data
    let read_data = manager.read_chunk(array_id, 0, 1000).await.unwrap();
    assert_eq!(read_data.len(), 1000);
    assert_eq!(read_data[0], 1.0);

    // Get statistics
    let stats = manager.statistics().await;
    assert_eq!(stats.active_arrays, 1);
    assert!(stats.total_memory_bytes > 0);

    // Release array
    manager.release_array(array_id).await.unwrap();
}

#[tokio::test]
async fn test_adaptive_chunking() {
    let chunking = AdaptiveChunking::new()
        .with_memory_limit(1024 * 1024) // 1MB
        .with_min_chunk_size(512)
        .with_max_chunk_size(8192)
        .build()
        .unwrap();

    // Small data - should use larger chunks
    let chunks_small = chunking.calculate_chunks(1000).await;
    assert!(chunks_small.chunk_size >= 512);

    // Large data - should use smaller chunks
    let chunks_large = chunking.calculate_chunks(1_000_000).await;
    assert!(chunks_large.chunk_size <= 8192);
    assert!(chunks_large.num_chunks > 1);
}

#[tokio::test]
async fn test_garbage_collector() {
    let gc = GarbageCollector::new(Duration::from_millis(100));
    let pool = Arc::new(BufferPool::new(100, 1024));

    // Register pool with GC
    gc.register_pool(pool.clone()).await;

    // Start GC
    gc.start().await;

    // Create garbage
    for _ in 0..50 {
        let buffer = pool.acquire().await.unwrap();
        // Don't release - becomes garbage
        std::mem::forget(buffer);
    }

    // Wait for GC cycle
    tokio::time::sleep(Duration::from_millis(200)).await;

    // GC should have cleaned up some objects
    let stats = pool.statistics().await;
    assert!(stats.gc_collections > 0);

    gc.stop().await;
}

#[tokio::test]
async fn test_scirs2_integration() {
    use scirs2_core::memory::GlobalBufferPool;

    // Use SciRS2's global buffer pool
    let buffer = GlobalBufferPool::acquire(1024).await.unwrap();
    assert_eq!(buffer.len(), 1024);

    // Return to pool
    GlobalBufferPool::release(buffer).await;

    // Verify reuse
    let buffer2 = GlobalBufferPool::acquire(1024).await.unwrap();
    assert_eq!(buffer2.len(), 1024);
}

#[tokio::test]
async fn test_memory_pool_with_scirs2_chunked_array() {
    use scirs2_core::memory_efficient::ChunkedArray;

    let manager = ChunkedArrayManager::new(1024, 1024 * 1024);

    // Create large array using SciRS2's ChunkedArray
    let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    let chunked = ChunkedArray::from_vec(data, 1024).unwrap();

    // Process chunks
    for (i, chunk) in chunked.chunks().enumerate() {
        assert!(chunk.len() <= 1024);
        assert!(chunk[0] >= (i * 1024) as f64);
    }
}

#[tokio::test]
async fn test_pool_statistics_accuracy() {
    let pool = BufferPool::new(10, 1024);

    // Track operations
    let mut acquired = 0;
    let mut released = 0;

    for _ in 0..20 {
        let buffer = pool.acquire().await.unwrap();
        acquired += 1;
        pool.release(buffer).await;
        released += 1;
    }

    let stats = pool.statistics().await;
    assert_eq!(stats.total_acquires, acquired);
    assert_eq!(stats.total_releases, released);
    assert!(stats.pool_hit_ratio > 0.0);
}

#[tokio::test]
async fn test_memory_leak_detection() {
    let pool = Arc::new(BufferPool::new(10, 1024));
    let monitor = MemoryPressureMonitor::new(0.8);

    // Create potential leak
    let mut leaked_buffers = Vec::new();
    for _ in 0..5 {
        let buffer = pool.acquire().await.unwrap();
        leaked_buffers.push(buffer);
        // Don't release
    }

    // Monitor should detect increasing memory
    let initial_usage = monitor.current_memory_usage().await;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let final_usage = monitor.current_memory_usage().await;
    assert!(final_usage >= initial_usage);

    // Check for leaks
    let has_leaks = pool.check_for_leaks().await;
    assert!(has_leaks);

    // Cleanup
    for buffer in leaked_buffers {
        pool.release(buffer).await;
    }
}

#[tokio::test]
async fn test_memory_pool_performance() {
    let pool = Arc::new(BufferPool::new(100, 4096));

    let start = std::time::Instant::now();

    // Benchmark acquire/release
    let mut handles = Vec::new();
    for _ in 0..1000 {
        let p = pool.clone();
        let handle = tokio::spawn(async move {
            let buffer = p.acquire().await.unwrap();
            // Simulate minimal work
            tokio::task::yield_now().await;
            p.release(buffer).await;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let duration = start.elapsed();
    let ops_per_sec = 1000.0 / duration.as_secs_f64();

    println!("Memory pool performance: {:.2} ops/sec", ops_per_sec);

    // Should be reasonably fast (at least 10k ops/sec)
    assert!(ops_per_sec > 10_000.0);

    let stats = pool.statistics().await;
    assert!(stats.pool_hit_ratio > 0.5); // Should have good hit ratio
}

#[tokio::test]
async fn test_automatic_gc_under_pressure() {
    let monitor = Arc::new(MemoryPressureMonitor::new(0.7));
    let pool = Arc::new(BufferPool::new(100, 1024));
    let gc = GarbageCollector::new(Duration::from_millis(50));

    gc.register_pool(pool.clone()).await;
    gc.start().await;

    // Simulate memory pressure
    monitor.update_memory_usage(0.85).await;

    // Create objects
    for _ in 0..50 {
        let buffer = pool.acquire().await.unwrap();
        std::mem::forget(buffer); // Leak intentionally
    }

    // Wait for automatic GC
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stats = pool.statistics().await;
    assert!(stats.gc_collections > 0);
    assert!(stats.gc_freed_objects > 0);

    gc.stop().await;
}
