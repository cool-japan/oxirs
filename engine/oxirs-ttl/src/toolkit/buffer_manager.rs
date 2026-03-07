//! Buffer management for efficient RDF parsing
//!
//! This module provides memory pooling and buffer management optimized for RDF parsing workloads.
//! It uses scirs2-core's advanced memory management to minimize allocations during parsing.

use std::sync::{Arc, Mutex};

/// Buffer manager for RDF parsing operations
///
/// This manages a pool of reusable buffers to minimize allocations during parsing.
/// Particularly useful for:
/// - Temporary string buffers during tokenization
/// - Blank node ID generation
/// - IRI resolution and prefix expansion
/// - Literal value accumulation
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::BufferManager;
///
/// let mut manager = BufferManager::new();
///
/// // Acquire a buffer for temporary string operations
/// let mut buffer = manager.acquire_string_buffer();
/// buffer.push_str("temporary content");
///
/// // Use the buffer...
/// let content = buffer.clone();
///
/// // Release it back to the pool when done
/// manager.release_string_buffer(buffer);
/// ```
#[derive(Debug)]
pub struct BufferManager {
    /// Pool of string buffers for reuse
    string_buffers: Vec<String>,
    /// Maximum number of buffers to keep in the pool
    max_pooled_buffers: usize,
    /// Statistics for monitoring buffer usage
    stats: BufferStats,
}

/// Statistics about buffer pool usage
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Total number of buffer acquisitions
    pub total_acquisitions: usize,
    /// Number of times a buffer was reused from pool (hit)
    pub pool_hits: usize,
    /// Number of times a new buffer was allocated (miss)
    pub pool_misses: usize,
    /// Total number of buffer releases
    pub total_releases: usize,
    /// Current number of buffers in the pool
    pub current_pool_size: usize,
}

impl BufferManager {
    /// Create a new buffer manager with default capacity
    pub fn new() -> Self {
        Self::with_capacity(32)
    }

    /// Create a new buffer manager with specified pool capacity
    pub fn with_capacity(max_pooled_buffers: usize) -> Self {
        Self {
            string_buffers: Vec::with_capacity(max_pooled_buffers),
            max_pooled_buffers,
            stats: BufferStats::default(),
        }
    }

    /// Acquire a string buffer from the pool
    ///
    /// Returns a cleared buffer from the pool if available,
    /// otherwise allocates a new buffer.
    pub fn acquire_string_buffer(&mut self) -> String {
        self.stats.total_acquisitions += 1;

        if let Some(mut buffer) = self.string_buffers.pop() {
            self.stats.pool_hits += 1;
            self.stats.current_pool_size = self.string_buffers.len();
            buffer.clear();
            buffer
        } else {
            self.stats.pool_misses += 1;
            String::with_capacity(256) // Pre-allocate reasonable capacity
        }
    }

    /// Acquire a string buffer with specific capacity
    pub fn acquire_string_buffer_with_capacity(&mut self, capacity: usize) -> String {
        self.stats.total_acquisitions += 1;

        // Try to find a buffer with sufficient capacity
        for (i, buffer) in self.string_buffers.iter().enumerate() {
            if buffer.capacity() >= capacity {
                let mut buffer = self.string_buffers.swap_remove(i);
                self.stats.pool_hits += 1;
                self.stats.current_pool_size = self.string_buffers.len();
                buffer.clear();
                return buffer;
            }
        }

        // No suitable buffer found, allocate new
        self.stats.pool_misses += 1;
        String::with_capacity(capacity)
    }

    /// Release a string buffer back to the pool
    ///
    /// The buffer will be cleared and reused for future acquisitions.
    /// If the pool is full, the buffer is dropped.
    pub fn release_string_buffer(&mut self, buffer: String) {
        self.stats.total_releases += 1;

        if self.string_buffers.len() < self.max_pooled_buffers {
            // Only keep buffers with reasonable capacity to avoid memory bloat
            if buffer.capacity() <= 4096 {
                self.string_buffers.push(buffer);
                self.stats.current_pool_size = self.string_buffers.len();
            }
        }
        // Otherwise, drop the buffer (it goes out of scope)
    }

    /// Generate a blank node ID efficiently using a pooled buffer
    pub fn generate_blank_node_id(&mut self, counter: usize) -> String {
        let mut buffer = self.acquire_string_buffer();
        buffer.push_str("_:b");
        buffer.push_str(&counter.to_string());
        // Don't release - we're returning this string
        buffer
    }

    /// Expand a prefixed name to a full IRI using a pooled buffer
    pub fn expand_prefixed_name(&mut self, _prefix: &str, local: &str, namespace: &str) -> String {
        let total_len = namespace.len() + local.len();
        let mut buffer = self.acquire_string_buffer_with_capacity(total_len);
        buffer.push_str(namespace);
        buffer.push_str(local);
        // Don't release - we're returning this string
        buffer
    }

    /// Get buffer pool statistics
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }

    /// Get the buffer pool hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.stats.total_acquisitions == 0 {
            return 0.0;
        }
        self.stats.pool_hits as f64 / self.stats.total_acquisitions as f64
    }

    /// Clear all buffers from the pool and reset statistics
    pub fn clear(&mut self) {
        self.string_buffers.clear();
        self.stats = BufferStats::default();
    }

    /// Shrink the buffer pool to fit current usage
    pub fn shrink_to_fit(&mut self) {
        self.string_buffers.shrink_to_fit();
    }

    /// Get the current number of buffers in the pool
    pub fn pool_size(&self) -> usize {
        self.string_buffers.len()
    }
}

impl Default for BufferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferStats {
    /// Get a human-readable report of buffer statistics
    pub fn report(&self) -> String {
        let hit_rate = if self.total_acquisitions > 0 {
            (self.pool_hits as f64 / self.total_acquisitions as f64) * 100.0
        } else {
            0.0
        };

        format!(
            "Buffer Pool Statistics:\n\
             - Total acquisitions: {}\n\
             - Pool hits: {} ({:.1}%)\n\
             - Pool misses: {}\n\
             - Total releases: {}\n\
             - Current pool size: {}",
            self.total_acquisitions,
            self.pool_hits,
            hit_rate,
            self.pool_misses,
            self.total_releases,
            self.current_pool_size
        )
    }
}

/// Thread-safe global buffer manager
pub struct GlobalBufferManager {
    inner: Arc<Mutex<BufferManager>>,
}

impl GlobalBufferManager {
    /// Create a new global buffer manager
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(BufferManager::new())),
        }
    }

    /// Create with specific capacity
    pub fn with_capacity(max_pooled_buffers: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BufferManager::with_capacity(max_pooled_buffers))),
        }
    }

    /// Acquire a string buffer
    pub fn acquire_string_buffer(&self) -> String {
        self.inner
            .lock()
            .expect("lock should not be poisoned")
            .acquire_string_buffer()
    }

    /// Release a string buffer
    pub fn release_string_buffer(&self, buffer: String) {
        self.inner
            .lock()
            .expect("lock should not be poisoned")
            .release_string_buffer(buffer);
    }

    /// Generate blank node ID
    pub fn generate_blank_node_id(&self, counter: usize) -> String {
        self.inner
            .lock()
            .expect("lock should not be poisoned")
            .generate_blank_node_id(counter)
    }

    /// Get statistics
    pub fn stats(&self) -> BufferStats {
        self.inner
            .lock()
            .expect("lock should not be poisoned")
            .stats()
            .clone()
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        self.inner
            .lock()
            .expect("lock should not be poisoned")
            .hit_rate()
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.inner
            .lock()
            .expect("lock should not be poisoned")
            .clear();
    }
}

impl Default for GlobalBufferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GlobalBufferManager {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_acquisition_and_release() {
        let mut manager = BufferManager::new();

        // Acquire a buffer
        let buffer1 = manager.acquire_string_buffer();
        assert_eq!(manager.stats().total_acquisitions, 1);
        assert_eq!(manager.stats().pool_misses, 1); // First acquisition is always a miss

        // Release it back
        manager.release_string_buffer(buffer1);
        assert_eq!(manager.stats().total_releases, 1);
        assert_eq!(manager.pool_size(), 1);

        // Acquire again - should reuse
        let buffer2 = manager.acquire_string_buffer();
        assert_eq!(manager.stats().total_acquisitions, 2);
        assert_eq!(manager.stats().pool_hits, 1); // Should have hit the pool
        assert_eq!(manager.pool_size(), 0);

        manager.release_string_buffer(buffer2);
    }

    #[test]
    fn test_buffer_pool_limit() {
        let mut manager = BufferManager::with_capacity(2);

        // Fill the pool
        let buf1 = manager.acquire_string_buffer();
        let buf2 = manager.acquire_string_buffer();
        let buf3 = manager.acquire_string_buffer();

        manager.release_string_buffer(buf1);
        manager.release_string_buffer(buf2);
        manager.release_string_buffer(buf3);

        // Pool should be limited to 2
        assert_eq!(manager.pool_size(), 2);
    }

    #[test]
    fn test_buffer_cleared_on_reuse() {
        let mut manager = BufferManager::new();

        let mut buffer = manager.acquire_string_buffer();
        buffer.push_str("old content");
        manager.release_string_buffer(buffer);

        let reused_buffer = manager.acquire_string_buffer();
        assert_eq!(reused_buffer.len(), 0); // Should be cleared
        assert_eq!(reused_buffer, "");

        manager.release_string_buffer(reused_buffer);
    }

    #[test]
    fn test_hit_rate() {
        let mut manager = BufferManager::new();

        let buf1 = manager.acquire_string_buffer();
        manager.release_string_buffer(buf1);

        let buf2 = manager.acquire_string_buffer(); // Hit
        manager.release_string_buffer(buf2);

        let buf3 = manager.acquire_string_buffer(); // Hit
        manager.release_string_buffer(buf3);

        // 3 acquisitions: 1 miss, 2 hits = 66.7% hit rate
        let hit_rate = manager.hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_blank_node_id_generation() {
        let mut manager = BufferManager::new();

        let id1 = manager.generate_blank_node_id(0);
        assert_eq!(id1, "_:b0");

        let id2 = manager.generate_blank_node_id(42);
        assert_eq!(id2, "_:b42");

        let id3 = manager.generate_blank_node_id(999);
        assert_eq!(id3, "_:b999");
    }

    #[test]
    fn test_prefixed_name_expansion() {
        let mut manager = BufferManager::new();

        let iri = manager.expand_prefixed_name("ex", "Person", "http://example.org/");
        assert_eq!(iri, "http://example.org/Person");

        let iri2 = manager.expand_prefixed_name(
            "rdf",
            "type",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        );
        assert_eq!(iri2, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    }

    #[test]
    fn test_buffer_capacity_hint() {
        let mut manager = BufferManager::new();

        let buffer = manager.acquire_string_buffer_with_capacity(1024);
        assert!(buffer.capacity() >= 1024);

        manager.release_string_buffer(buffer);

        // Acquire again - should get the same high-capacity buffer
        let buffer2 = manager.acquire_string_buffer_with_capacity(512);
        assert!(buffer2.capacity() >= 512); // Should reuse the 1024-capacity buffer

        manager.release_string_buffer(buffer2);
    }

    #[test]
    fn test_stats_report() {
        let mut manager = BufferManager::new();

        let buf = manager.acquire_string_buffer();
        manager.release_string_buffer(buf);

        let report = manager.stats().report();
        assert!(report.contains("Total acquisitions: 1"));
        assert!(report.contains("Pool hits: 0"));
        assert!(report.contains("Pool misses: 1"));
        assert!(report.contains("Total releases: 1"));
    }

    #[test]
    fn test_clear() {
        let mut manager = BufferManager::new();

        let buf = manager.acquire_string_buffer();
        manager.release_string_buffer(buf);

        assert_eq!(manager.pool_size(), 1);
        assert_eq!(manager.stats().total_acquisitions, 1);

        manager.clear();

        assert_eq!(manager.pool_size(), 0);
        assert_eq!(manager.stats().total_acquisitions, 0);
    }

    #[test]
    fn test_global_buffer_manager() {
        let manager = GlobalBufferManager::new();

        let buf1 = manager.acquire_string_buffer();
        manager.release_string_buffer(buf1);

        let buf2 = manager.acquire_string_buffer();
        manager.release_string_buffer(buf2);

        // Should have hit the pool
        assert!(manager.hit_rate() > 0.0);
    }

    #[test]
    fn test_global_buffer_manager_clone() {
        let manager1 = GlobalBufferManager::new();
        let manager2 = manager1.clone();

        let buf = manager1.acquire_string_buffer();
        manager2.release_string_buffer(buf);

        // Both should share the same pool
        assert_eq!(manager1.stats().total_releases, 1);
        assert_eq!(manager2.stats().total_releases, 1);
    }
}
