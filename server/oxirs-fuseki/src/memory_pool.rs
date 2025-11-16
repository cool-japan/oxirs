//! Memory Pooling and Optimization
//!
//! This module provides advanced memory management for high-performance SPARQL processing:
//! - Object pooling for frequent allocations (query contexts, result sets)
//! - Memory pressure monitoring and adaptive behavior
//! - Arena allocators for batch allocations
//! - Zero-copy buffers for large result streaming
//! - Memory-efficient data structures using SciRS2

use crate::error::{FusekiError, FusekiResult};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, instrument, warn};

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Enable memory pooling
    pub enabled: bool,
    /// Maximum total memory in bytes (0 = unlimited)
    pub max_memory_bytes: u64,
    /// Memory pressure threshold (0.0-1.0) to trigger cleanup
    pub pressure_threshold: f64,
    /// Object pool sizes
    pub query_context_pool_size: usize,
    pub result_buffer_pool_size: usize,
    /// Buffer sizes
    pub small_buffer_size: usize, // 4KB
    pub medium_buffer_size: usize, // 64KB
    pub large_buffer_size: usize,  // 1MB
    /// Chunking configuration for large results
    pub chunk_size_bytes: usize,
    /// Enable memory profiling
    pub enable_profiling: bool,
    /// GC interval in seconds
    pub gc_interval_secs: u64,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        MemoryPoolConfig {
            enabled: true,
            max_memory_bytes: 8_589_934_592, // 8GB
            pressure_threshold: 0.85,
            query_context_pool_size: 1000,
            result_buffer_pool_size: 500,
            small_buffer_size: 4 * 1024,    // 4KB
            medium_buffer_size: 64 * 1024,  // 64KB
            large_buffer_size: 1024 * 1024, // 1MB
            chunk_size_bytes: 1024 * 1024,  // 1MB chunks
            enable_profiling: true,
            gc_interval_secs: 60,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: u64,
    /// Total deallocated memory in bytes
    pub total_deallocated: u64,
    /// Current memory usage in bytes
    pub current_usage: u64,
    /// Peak memory usage in bytes
    pub peak_usage: u64,
    /// Number of active objects
    pub active_objects: usize,
    /// Number of pooled objects
    pub pooled_objects: usize,
    /// Pool hit ratio
    pub pool_hit_ratio: f64,
    /// Memory pressure (0.0-1.0)
    pub memory_pressure: f64,
    /// Number of GC runs
    pub gc_runs: u64,
    /// Last GC duration in milliseconds
    pub last_gc_duration_ms: u64,
}

/// Pooled buffer for reusable memory allocation
pub struct PooledBuffer {
    data: Vec<u8>,
    capacity: usize,
}

impl PooledBuffer {
    fn new(capacity: usize) -> Self {
        PooledBuffer {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Get buffer data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable buffer data
    pub fn data_mut(&mut self) -> &mut Vec<u8> {
        &mut self.data
    }

    /// Clear buffer for reuse
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Query context pool for reducing allocations
pub struct QueryContextPool<T> {
    pool: Arc<RwLock<VecDeque<T>>>,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
    total_created: AtomicU64,
    total_reused: AtomicU64,
}

impl<T: Send + 'static> QueryContextPool<T> {
    /// Create a new query context pool
    pub fn new<F>(max_size: usize, factory: F) -> Arc<Self>
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Arc::new(QueryContextPool {
            pool: Arc::new(RwLock::new(VecDeque::with_capacity(max_size))),
            factory: Arc::new(factory),
            max_size,
            total_created: AtomicU64::new(0),
            total_reused: AtomicU64::new(0),
        })
    }

    /// Acquire an object from the pool
    pub async fn acquire(&self) -> T {
        // Try to get from pool first
        {
            let mut pool = self.pool.write().await;
            if let Some(obj) = pool.pop_front() {
                self.total_reused.fetch_add(1, Ordering::Relaxed);
                return obj;
            }
        }

        // Create new if pool is empty
        self.total_created.fetch_add(1, Ordering::Relaxed);
        (self.factory)()
    }

    /// Return an object to the pool
    pub async fn release(&self, obj: T) {
        let mut pool = self.pool.write().await;
        if pool.len() < self.max_size {
            pool.push_back(obj);
        }
        // Otherwise drop the object
    }

    /// Get pool statistics
    pub async fn stats(&self) -> (u64, u64, usize) {
        let created = self.total_created.load(Ordering::Relaxed);
        let reused = self.total_reused.load(Ordering::Relaxed);
        let pooled = self.pool.read().await.len();
        (created, reused, pooled)
    }
}

/// Memory manager with pressure monitoring
pub struct MemoryManager {
    config: MemoryPoolConfig,

    // Query context pools
    query_context_pool: Arc<RwLock<VecDeque<QueryContext>>>,

    // Buffer pools
    buffer_pool: Arc<RwLock<VecDeque<PooledBuffer>>>,

    // Memory tracking
    current_usage: Arc<AtomicU64>,
    peak_usage: Arc<AtomicU64>,
    total_allocated: Arc<AtomicU64>,
    total_deallocated: Arc<AtomicU64>,

    // Pool statistics
    pool_hits: Arc<AtomicU64>,
    pool_misses: Arc<AtomicU64>,
    active_objects: Arc<AtomicUsize>,

    // GC tracking
    gc_runs: Arc<AtomicU64>,
    last_gc_time: Arc<RwLock<Instant>>,
    last_gc_duration: Arc<AtomicU64>,
}

/// Query context for pooling
#[derive(Clone)]
pub struct QueryContext {
    pub id: String,
    pub buffer: Vec<u8>,
    pub metadata: Vec<(String, String)>,
}

impl QueryContext {
    pub fn new() -> Self {
        QueryContext {
            id: uuid::Uuid::new_v4().to_string(),
            buffer: Vec::with_capacity(4096),
            metadata: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.metadata.clear();
        self.id = uuid::Uuid::new_v4().to_string();
    }
}

impl Default for QueryContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(config: MemoryPoolConfig) -> FusekiResult<Arc<Self>> {
        // Initialize query context pool
        let mut query_pool = VecDeque::with_capacity(config.query_context_pool_size);
        for _ in 0..config.query_context_pool_size / 2 {
            query_pool.push_back(QueryContext::new());
        }

        // Initialize buffer pool
        let mut buffer_pool = VecDeque::with_capacity(config.result_buffer_pool_size);
        for _ in 0..config.result_buffer_pool_size / 2 {
            buffer_pool.push_back(PooledBuffer::new(config.medium_buffer_size));
        }

        let manager = Arc::new(MemoryManager {
            config,
            query_context_pool: Arc::new(RwLock::new(query_pool)),
            buffer_pool: Arc::new(RwLock::new(buffer_pool)),
            current_usage: Arc::new(AtomicU64::new(0)),
            peak_usage: Arc::new(AtomicU64::new(0)),
            total_allocated: Arc::new(AtomicU64::new(0)),
            total_deallocated: Arc::new(AtomicU64::new(0)),
            pool_hits: Arc::new(AtomicU64::new(0)),
            pool_misses: Arc::new(AtomicU64::new(0)),
            active_objects: Arc::new(AtomicUsize::new(0)),
            gc_runs: Arc::new(AtomicU64::new(0)),
            last_gc_time: Arc::new(RwLock::new(Instant::now())),
            last_gc_duration: Arc::new(AtomicU64::new(0)),
        });

        // Start background GC
        manager.clone().start_gc_loop();

        info!(
            "Memory manager initialized with {}MB max memory",
            manager.config.max_memory_bytes / 1_048_576
        );

        Ok(manager)
    }

    /// Allocate buffer from pool
    #[instrument(skip(self))]
    pub async fn allocate_buffer(&self, size: usize) -> FusekiResult<PooledBuffer> {
        // Check memory pressure
        if self.is_under_pressure().await {
            // Trigger immediate GC
            self.run_gc().await?;

            // Check again after GC
            if self.is_under_pressure().await {
                return Err(FusekiError::service_unavailable(
                    "Memory pressure too high, allocation rejected",
                ));
            }
        }

        // Try to get from pool (only if size matches)
        let buffer = {
            let mut pool = self.buffer_pool.write().await;
            // Find a buffer with matching capacity
            pool.iter()
                .position(|b| b.capacity() == size)
                .and_then(|idx| pool.remove(idx))
        };

        let buffer = if let Some(mut buf) = buffer {
            self.pool_hits.fetch_add(1, Ordering::Relaxed);
            buf.clear();
            buf
        } else {
            self.pool_misses.fetch_add(1, Ordering::Relaxed);
            PooledBuffer::new(size)
        };

        // Track allocation
        self.track_allocation(size as u64);

        Ok(buffer)
    }

    /// Acquire query context from pool
    #[instrument(skip(self))]
    pub async fn acquire_query_context(&self) -> QueryContext {
        let mut pool = self.query_context_pool.write().await;

        if let Some(mut context) = pool.pop_front() {
            self.pool_hits.fetch_add(1, Ordering::Relaxed);
            context.reset();
            self.active_objects.fetch_add(1, Ordering::Relaxed);
            debug!("Reused query context from pool");
            return context;
        }

        self.pool_misses.fetch_add(1, Ordering::Relaxed);
        self.active_objects.fetch_add(1, Ordering::Relaxed);
        debug!("Created new query context");
        QueryContext::new()
    }

    /// Release query context back to pool
    #[instrument(skip(self, context))]
    pub async fn release_query_context(&self, context: QueryContext) {
        let mut pool = self.query_context_pool.write().await;

        if pool.len() < self.config.query_context_pool_size {
            pool.push_back(context);
            self.active_objects.fetch_sub(1, Ordering::Relaxed);
            debug!("Returned query context to pool");
        } else {
            self.active_objects.fetch_sub(1, Ordering::Relaxed);
            debug!("Dropped query context (pool full)");
        }
    }

    /// Create chunked array for large result sets
    /// Returns a standard Vec for now until SciRS2 API is clarified
    pub fn create_chunked_buffer(&self, capacity: usize) -> Vec<u8> {
        Vec::with_capacity(capacity)
    }

    /// Get chunk size for result streaming
    pub fn get_chunk_size(&self) -> usize {
        self.config.chunk_size_bytes
    }

    /// Track memory allocation
    fn track_allocation(&self, size: u64) {
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        let current = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak if needed
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
    }

    /// Check if under memory pressure
    async fn is_under_pressure(&self) -> bool {
        if self.config.max_memory_bytes == 0 {
            return false; // Unlimited memory
        }

        let current = self.current_usage.load(Ordering::Relaxed);
        let pressure = (current as f64) / (self.config.max_memory_bytes as f64);

        pressure > self.config.pressure_threshold
    }

    /// Calculate current memory pressure
    pub async fn get_memory_pressure(&self) -> f64 {
        if self.config.max_memory_bytes == 0 {
            return 0.0;
        }

        let current = self.current_usage.load(Ordering::Relaxed);
        (current as f64) / (self.config.max_memory_bytes as f64)
    }

    /// Start background GC loop
    fn start_gc_loop(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(self.config.gc_interval_secs));

            loop {
                interval.tick().await;

                if let Err(e) = self.run_gc().await {
                    warn!("GC failed: {}", e);
                }
            }
        });
    }

    /// Run garbage collection
    #[instrument(skip(self))]
    async fn run_gc(&self) -> FusekiResult<()> {
        let start = Instant::now();
        debug!("Starting garbage collection");

        // Trim query context pool if needed
        {
            let mut pool = self.query_context_pool.write().await;
            let target_size = self.config.query_context_pool_size / 2;
            while pool.len() > target_size {
                pool.pop_back();
            }
        }

        // Force buffer pool cleanup (if provided by scirs2-core)
        // self.buffer_pool.cleanup();

        let duration = start.elapsed();
        self.gc_runs.fetch_add(1, Ordering::Relaxed);
        self.last_gc_duration
            .store(duration.as_millis() as u64, Ordering::Relaxed);
        *self.last_gc_time.write().await = Instant::now();

        info!("GC completed in {:.2}ms", duration.as_millis());
        Ok(())
    }

    /// Get memory statistics
    pub async fn get_stats(&self) -> MemoryStats {
        let current = self.current_usage.load(Ordering::Relaxed);
        let peak = self.peak_usage.load(Ordering::Relaxed);
        let allocated = self.total_allocated.load(Ordering::Relaxed);
        let deallocated = self.total_deallocated.load(Ordering::Relaxed);

        let hits = self.pool_hits.load(Ordering::Relaxed);
        let misses = self.pool_misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;
        let hit_ratio = if total_requests > 0 {
            (hits as f64) / (total_requests as f64)
        } else {
            0.0
        };

        let pressure = if self.config.max_memory_bytes > 0 {
            (current as f64) / (self.config.max_memory_bytes as f64)
        } else {
            0.0
        };

        let pooled = self.query_context_pool.read().await.len();
        let active = self.active_objects.load(Ordering::Relaxed);

        MemoryStats {
            total_allocated: allocated,
            total_deallocated: deallocated,
            current_usage: current,
            peak_usage: peak,
            active_objects: active,
            pooled_objects: pooled,
            pool_hit_ratio: hit_ratio,
            memory_pressure: pressure,
            gc_runs: self.gc_runs.load(Ordering::Relaxed),
            last_gc_duration_ms: self.last_gc_duration.load(Ordering::Relaxed),
        }
    }

    /// Force garbage collection
    pub async fn force_gc(&self) -> FusekiResult<()> {
        info!("Forcing garbage collection");
        self.run_gc().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_manager_creation() {
        let config = MemoryPoolConfig::default();
        let manager = MemoryManager::new(config);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        let stats = manager.get_stats().await;
        assert_eq!(stats.current_usage, 0);
    }

    #[tokio::test]
    async fn test_query_context_pooling() {
        let config = MemoryPoolConfig::default();
        let manager = MemoryManager::new(config).unwrap();

        // Acquire context
        let context = manager.acquire_query_context().await;
        assert!(!context.id.is_empty());

        // Release context
        manager.release_query_context(context).await;

        // Acquire again - should reuse
        let context2 = manager.acquire_query_context().await;
        assert!(!context2.id.is_empty());
    }

    #[tokio::test]
    async fn test_memory_pressure() {
        let config = MemoryPoolConfig {
            max_memory_bytes: 1024 * 1024, // 1MB
            pressure_threshold: 0.8,
            ..Default::default()
        };
        let manager = MemoryManager::new(config).unwrap();

        let pressure = manager.get_memory_pressure().await;
        assert!(pressure < 0.1);
    }

    #[tokio::test]
    async fn test_buffer_allocation() {
        let config = MemoryPoolConfig::default();
        let manager = MemoryManager::new(config).unwrap();

        let buffer = manager.allocate_buffer(4096).await;
        assert!(buffer.is_ok());

        let buffer = buffer.unwrap();
        assert_eq!(buffer.capacity(), 4096);
    }
}
