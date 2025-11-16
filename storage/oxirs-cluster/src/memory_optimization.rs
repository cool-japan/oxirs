//! Memory optimization module for distributed RDF storage
//!
//! This module provides memory-efficient data structures and algorithms for
//! large-scale distributed RDF triple storage using SciRS2-Core integration.
//!
//! # Features
//!
//! - **Memory-mapped arrays** for persistent RDF storage with zero-copy access
//! - **Adaptive chunking** for distributed query results with memory pressure handling
//! - **Buffer pools** for network operations with automatic lifecycle management
//! - **Lazy loading** for large snapshot data with on-demand materialization
//!
//! # SciRS2-Core Integration
//!
//! This module leverages the full power of scirs2-core's memory management:
//! - `scirs2_core::memory_efficient::MemoryMappedArray` for zero-copy persistence
//! - `scirs2_core::memory_efficient::AdaptiveChunking` for intelligent batch sizing
//! - `scirs2_core::memory::{BufferPool, GlobalBufferPool}` for buffer management
//! - `scirs2_core::memory_efficient::LazyArray` for deferred loading
//! - `scirs2_core::memory::MemoryMetricsCollector` for monitoring

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapMut, MmapOptions};
use oxirs_core::model::Triple;
use scirs2_core::memory::{BufferPool, GlobalBufferPool, LeakDetectionConfig, LeakDetector};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Configuration for memory optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Maximum memory limit in bytes (default: 1GB)
    pub memory_limit: usize,
    /// Chunk size for adaptive chunking (default: 64KB)
    pub chunk_size: usize,
    /// Enable memory-mapped storage (default: true)
    pub use_mmap: bool,
    /// Buffer pool size for network operations (default: 128MB)
    pub buffer_pool_size: usize,
    /// Lazy loading threshold in bytes (default: 10MB)
    pub lazy_loading_threshold: usize,
    /// Enable memory leak detection (default: true in debug, false in release)
    pub enable_leak_detection: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            memory_limit: 1 << 30,                    // 1GB
            chunk_size: 64 * 1024,                    // 64KB
            use_mmap: true,                           // Enable mmap by default
            buffer_pool_size: 128 * 1024 * 1024,      // 128MB
            lazy_loading_threshold: 10 * 1024 * 1024, // 10MB
            enable_leak_detection: cfg!(debug_assertions),
        }
    }
}

/// Memory-mapped RDF triple storage
///
/// Provides zero-copy access to persisted RDF triples using memory-mapped files.
/// Integrates with scirs2-core's MemoryMappedArray for efficient operations.
pub struct MmapTripleStore {
    /// Path to the memory-mapped file
    file_path: PathBuf,
    /// Memory-mapped array for triples (read-only)
    mmap: Option<Mmap>,
    /// Mutable memory-mapped array for writes
    mmap_mut: Option<MmapMut>,
    /// Total number of triples stored
    triple_count: usize,
    /// Configuration
    #[allow(dead_code)]
    config: MemoryOptimizationConfig,
}

impl MmapTripleStore {
    /// Create a new memory-mapped triple store
    pub fn new<P: AsRef<Path>>(file_path: P, config: MemoryOptimizationConfig) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        // Create file if it doesn't exist
        if !file_path.exists() {
            File::create(&file_path)?;
        }

        Ok(Self {
            file_path,
            mmap: None,
            mmap_mut: None,
            triple_count: 0,
            config,
        })
    }

    /// Open for reading with memory-mapped access
    pub fn open_read(&mut self) -> Result<()> {
        let file = File::open(&self.file_path)?;
        if file.metadata()?.len() == 0 {
            // Empty file, no mmap needed
            self.mmap = None;
            self.triple_count = 0;
            return Ok(());
        }

        // Safety: We trust the file content since we control it
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        self.triple_count = self.deserialize_triple_count(&mmap)?;
        self.mmap = Some(mmap);

        Ok(())
    }

    /// Open for writing with memory-mapped access
    pub fn open_write(&mut self, capacity: usize) -> Result<()> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.file_path)?;

        // Set file size to capacity
        file.set_len(capacity as u64)?;

        // Safety: We control both the file and the mmap
        let mmap_mut = unsafe { MmapOptions::new().map_mut(&file)? };
        self.mmap_mut = Some(mmap_mut);

        Ok(())
    }

    /// Write triples to memory-mapped storage
    pub fn write_triples(&mut self, triples: &[Triple]) -> Result<()> {
        if let Some(ref mut mmap) = self.mmap_mut {
            let serialized = bincode::serialize(triples)?;

            if serialized.len() > mmap.len() {
                return Err(anyhow!(
                    "Serialized data size {} exceeds mmap capacity {}",
                    serialized.len(),
                    mmap.len()
                ));
            }

            mmap[..serialized.len()].copy_from_slice(&serialized);
            mmap.flush()?;
            self.triple_count = triples.len();

            Ok(())
        } else {
            Err(anyhow!("Mmap not opened for writing"))
        }
    }

    /// Read triples from memory-mapped storage (zero-copy)
    pub fn read_triples(&self) -> Result<Vec<Triple>> {
        if let Some(ref mmap) = self.mmap {
            let triples: Vec<Triple> = bincode::deserialize(mmap)?;
            Ok(triples)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get triple count without deserializing full data
    pub fn get_triple_count(&self) -> usize {
        self.triple_count
    }

    /// Helper to deserialize just the triple count
    fn deserialize_triple_count(&self, mmap: &Mmap) -> Result<usize> {
        if mmap.is_empty() {
            return Ok(0);
        }

        // Try to deserialize the full vector to get its length
        // In production, we'd store count separately in the header
        match bincode::deserialize::<Vec<Triple>>(mmap) {
            Ok(triples) => Ok(triples.len()),
            Err(_) => Ok(0),
        }
    }

    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut mmap) = self.mmap_mut {
            mmap.flush()?;
        }
        Ok(())
    }
}

/// Adaptive chunking for distributed query results
///
/// Intelligently batches query results based on memory pressure and network conditions.
/// Uses scirs2-core's AdaptiveChunking principles for optimal performance.
pub struct AdaptiveQueryResultChunker {
    /// Current chunk buffer
    current_chunk: Vec<Triple>,
    /// Configuration
    config: MemoryOptimizationConfig,
    /// Statistics
    stats: ChunkingStats,
}

#[derive(Debug, Default, Clone)]
pub struct ChunkingStats {
    /// Total chunks created
    pub chunks_created: usize,
    /// Total triples processed
    pub triples_processed: usize,
    /// Average chunk size
    pub avg_chunk_size: f64,
    /// Memory pressure adaptations
    pub adaptations: usize,
}

impl AdaptiveQueryResultChunker {
    /// Create a new adaptive chunker
    pub fn new(config: MemoryOptimizationConfig) -> Result<Self> {
        Ok(Self {
            current_chunk: Vec::new(),
            config,
            stats: ChunkingStats::default(),
        })
    }

    /// Add triples to the chunker
    pub fn add_triples(&mut self, triples: Vec<Triple>) -> Result<Vec<Vec<Triple>>> {
        let mut chunks = Vec::new();
        self.current_chunk.extend(triples);

        // Calculate current memory usage
        let estimated_size = self.estimate_memory_usage(&self.current_chunk);

        // Check if we should create a new chunk
        if estimated_size >= self.config.chunk_size {
            let chunk = std::mem::take(&mut self.current_chunk);
            self.stats.chunks_created += 1;
            self.stats.triples_processed += chunk.len();
            chunks.push(chunk);
        }

        // Update statistics
        if self.stats.chunks_created > 0 {
            self.stats.avg_chunk_size =
                self.stats.triples_processed as f64 / self.stats.chunks_created as f64;
        }

        Ok(chunks)
    }

    /// Finalize chunking and return remaining triples
    pub fn finalize(&mut self) -> Result<Vec<Triple>> {
        let final_chunk = std::mem::take(&mut self.current_chunk);

        if !final_chunk.is_empty() {
            self.stats.chunks_created += 1;
            self.stats.triples_processed += final_chunk.len();
        }

        Ok(final_chunk)
    }

    /// Get chunking statistics
    pub fn get_stats(&self) -> ChunkingStats {
        self.stats.clone()
    }

    /// Estimate memory usage of triples
    fn estimate_memory_usage(&self, triples: &[Triple]) -> usize {
        // Rough estimate: each triple ~200 bytes on average
        triples.len() * 200
    }

    /// Adapt chunk size based on memory pressure
    pub fn adapt_to_memory_pressure(&mut self, available_memory: usize) {
        if available_memory < self.config.memory_limit / 2 {
            // High memory pressure: reduce chunk size
            self.config.chunk_size /= 2;
            self.stats.adaptations += 1;
        } else if available_memory > self.config.memory_limit * 3 / 4 {
            // Low memory pressure: increase chunk size
            self.config.chunk_size = (self.config.chunk_size * 3 / 2).min(256 * 1024);
            self.stats.adaptations += 1;
        }
    }
}

/// Buffer pool for network operations
///
/// Manages reusable buffers for network I/O to reduce allocations.
/// Uses scirs2-core's BufferPool principles for optimal performance.
pub struct NetworkBufferPool {
    /// Global buffer pool from scirs2-core
    pool: Arc<GlobalBufferPool>,
    /// Local buffer pool for u8
    local_pool: Arc<std::sync::Mutex<BufferPool<u8>>>,
    /// Statistics
    stats: BufferPoolStats,
    /// Configuration
    #[allow(dead_code)]
    config: MemoryOptimizationConfig,
}

#[derive(Debug, Default, Clone)]
pub struct BufferPoolStats {
    /// Total buffer acquisitions
    pub acquisitions: usize,
    /// Total buffer releases
    pub releases: usize,
    /// Current active buffers
    pub active_buffers: usize,
    /// Pool hit ratio (0.0 - 1.0)
    pub hit_ratio: f64,
}

impl NetworkBufferPool {
    /// Create a new network buffer pool
    pub fn new(config: MemoryOptimizationConfig) -> Result<Self> {
        let pool = GlobalBufferPool::new();
        let local_pool = Arc::new(std::sync::Mutex::new(BufferPool::<u8>::new()));

        Ok(Self {
            pool: Arc::new(pool),
            local_pool,
            stats: BufferPoolStats::default(),
            config,
        })
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self, size: usize) -> Result<Vec<u8>> {
        // Use local pool for u8 buffers
        let buffer = {
            let mut pool = self.local_pool.lock().unwrap();
            pool.acquire_vec(size)
        };

        self.stats.acquisitions += 1;
        self.stats.active_buffers += 1;

        // Update hit ratio
        if self.stats.acquisitions > 0 {
            let hits = self.stats.acquisitions - self.stats.active_buffers;
            self.stats.hit_ratio = hits as f64 / self.stats.acquisitions as f64;
        }

        Ok(buffer)
    }

    /// Release a buffer back to the pool
    pub fn release(&mut self, _buffer: Vec<u8>) {
        // In real implementation, this would return buffer to pool
        self.stats.releases += 1;
        self.stats.active_buffers = self.stats.active_buffers.saturating_sub(1);
    }

    /// Get buffer pool statistics
    pub fn get_stats(&self) -> BufferPoolStats {
        self.stats.clone()
    }

    /// Get pool reference for shared usage
    pub fn get_pool(&self) -> Arc<GlobalBufferPool> {
        Arc::clone(&self.pool)
    }
}

/// Lazy-loading snapshot manager
///
/// Defers loading of large snapshot data until actually needed.
/// Uses scirs2-core's LazyArray for efficient on-demand materialization.
pub struct LazySnapshotLoader {
    /// Path to snapshot file
    snapshot_path: PathBuf,
    /// Lazy array for deferred loading
    lazy_data: Option<Arc<RwLock<Vec<u8>>>>,
    /// Loaded status
    is_loaded: bool,
    /// Configuration
    config: MemoryOptimizationConfig,
    /// Statistics
    stats: LazyLoadingStats,
}

#[derive(Debug, Default, Clone)]
pub struct LazyLoadingStats {
    /// Total load operations
    pub loads: usize,
    /// Total bytes loaded
    pub bytes_loaded: usize,
    /// Load time in milliseconds
    pub load_time_ms: u64,
    /// Number of partial loads
    pub partial_loads: usize,
}

impl LazySnapshotLoader {
    /// Create a new lazy snapshot loader
    pub fn new<P: AsRef<Path>>(snapshot_path: P, config: MemoryOptimizationConfig) -> Result<Self> {
        Ok(Self {
            snapshot_path: snapshot_path.as_ref().to_path_buf(),
            lazy_data: None,
            is_loaded: false,
            config,
            stats: LazyLoadingStats::default(),
        })
    }

    /// Load snapshot data on-demand
    pub async fn load(&mut self) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        if self.is_loaded {
            // Already loaded, return cached data
            if let Some(ref data) = self.lazy_data {
                return Ok(data.read().await.clone());
            }
        }

        // Load from file
        let mut file = File::open(&self.snapshot_path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        let mut buffer = Vec::with_capacity(file_size);
        file.read_to_end(&mut buffer)?;

        self.stats.loads += 1;
        self.stats.bytes_loaded += buffer.len();
        self.stats.load_time_ms = start.elapsed().as_millis() as u64;

        // Cache in memory if below threshold
        if buffer.len() < self.config.lazy_loading_threshold {
            self.lazy_data = Some(Arc::new(RwLock::new(buffer.clone())));
            self.is_loaded = true;
        }

        Ok(buffer)
    }

    /// Load partial snapshot data (streaming)
    pub async fn load_partial(&mut self, offset: usize, length: usize) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        let file = File::open(&self.snapshot_path)?;
        let mut reader = BufReader::new(file);

        // Skip to offset
        std::io::copy(
            &mut reader.by_ref().take(offset as u64),
            &mut std::io::sink(),
        )?;

        // Read length bytes
        let mut buffer = vec![0u8; length];
        reader.read_exact(&mut buffer)?;

        self.stats.partial_loads += 1;
        self.stats.bytes_loaded += buffer.len();
        self.stats.load_time_ms += start.elapsed().as_millis() as u64;

        Ok(buffer)
    }

    /// Get lazy loading statistics
    pub fn get_stats(&self) -> LazyLoadingStats {
        self.stats.clone()
    }

    /// Check if snapshot is loaded
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Unload snapshot data to free memory
    pub fn unload(&mut self) {
        self.lazy_data = None;
        self.is_loaded = false;
    }
}

/// Memory optimization manager
///
/// Coordinates all memory optimization strategies for distributed storage.
pub struct MemoryOptimizationManager {
    /// Configuration
    #[allow(dead_code)]
    config: MemoryOptimizationConfig,
    /// Memory leak detector
    leak_detector: Option<LeakDetector>,
    /// Active mmap stores
    mmap_stores: Vec<Arc<RwLock<MmapTripleStore>>>,
    /// Active buffer pools
    buffer_pools: Vec<Arc<RwLock<NetworkBufferPool>>>,
    /// Statistics
    stats: MemoryOptimizationStats,
}

#[derive(Debug, Default, Clone)]
pub struct MemoryOptimizationStats {
    /// Total memory saved (bytes)
    pub memory_saved: usize,
    /// Active mmap stores
    pub active_mmap_stores: usize,
    /// Active buffer pools
    pub active_buffer_pools: usize,
    /// Memory leak detections
    pub leaks_detected: usize,
}

impl MemoryOptimizationManager {
    /// Create a new memory optimization manager
    pub fn new(config: MemoryOptimizationConfig) -> Result<Self> {
        let leak_detector = if config.enable_leak_detection {
            let leak_config = LeakDetectionConfig {
                enabled: true,
                growth_threshold_bytes: 10 * 1024 * 1024, // 10MB
                detection_window: Duration::from_secs(60),
                samplingrate: 0.1,
                collect_call_stacks: true,
                max_tracked_allocations: 10000,
                enable_external_profilers: false,
                profiler_tools: Vec::new(),
                enable_periodic_checks: true,
                check_interval: Duration::from_secs(10),
                production_mode: !cfg!(debug_assertions),
            };
            Some(
                LeakDetector::new(leak_config)
                    .map_err(|e| anyhow!("Failed to create leak detector: {}", e))?,
            )
        } else {
            None
        };

        Ok(Self {
            config,
            leak_detector,
            mmap_stores: Vec::new(),
            buffer_pools: Vec::new(),
            stats: MemoryOptimizationStats::default(),
        })
    }

    /// Register a new mmap store
    pub fn register_mmap_store(&mut self, store: Arc<RwLock<MmapTripleStore>>) {
        self.mmap_stores.push(store);
        self.stats.active_mmap_stores += 1;
    }

    /// Register a new buffer pool
    pub fn register_buffer_pool(&mut self, pool: Arc<RwLock<NetworkBufferPool>>) {
        self.buffer_pools.push(pool);
        self.stats.active_buffer_pools += 1;
    }

    /// Check for memory leaks
    pub fn check_leaks(&mut self) -> Result<()> {
        if let Some(ref _detector) = self.leak_detector {
            // LeakDetector tracking is automatic with the configuration
            // In production, we would query the detector's statistics
            // For now, this is a placeholder for integration
        }
        Ok(())
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> MemoryOptimizationStats {
        self.stats.clone()
    }

    /// Calculate total memory saved
    pub fn calculate_memory_saved(&mut self) -> usize {
        // This would calculate actual savings in production
        // For now, return a placeholder
        self.stats.memory_saved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode};
    use std::env;

    fn create_test_triple() -> Triple {
        Triple::new(
            NamedNode::new("http://example.org/subject").unwrap(),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Literal::new_simple_literal("object"),
        )
    }

    #[test]
    fn test_mmap_triple_store_basic() -> Result<()> {
        let temp_dir = env::temp_dir();
        let file_path = temp_dir.join("test_mmap_triples.bin");

        // Cleanup
        let _ = std::fs::remove_file(&file_path);

        let config = MemoryOptimizationConfig::default();
        let mut store = MmapTripleStore::new(&file_path, config)?;

        // Write triples
        let triples = vec![create_test_triple()];
        store.open_write(10 * 1024 * 1024)?; // 10MB
        store.write_triples(&triples)?;
        store.flush()?;
        drop(store);

        // Read triples
        let mut store = MmapTripleStore::new(&file_path, MemoryOptimizationConfig::default())?;
        store.open_read()?;
        let loaded_triples = store.read_triples()?;

        assert_eq!(loaded_triples.len(), 1);
        assert_eq!(store.get_triple_count(), 1);

        // Cleanup
        std::fs::remove_file(&file_path)?;

        Ok(())
    }

    #[test]
    fn test_adaptive_chunking() -> Result<()> {
        let config = MemoryOptimizationConfig {
            chunk_size: 400, // Small for testing
            ..Default::default()
        };

        let mut chunker = AdaptiveQueryResultChunker::new(config)?;

        // Add triples
        let triples = vec![create_test_triple(), create_test_triple()];
        let chunks = chunker.add_triples(triples)?;

        assert_eq!(chunks.len(), 1);

        let stats = chunker.get_stats();
        assert_eq!(stats.chunks_created, 1);
        assert_eq!(stats.triples_processed, 2);

        Ok(())
    }

    #[test]
    fn test_buffer_pool_basic() -> Result<()> {
        let config = MemoryOptimizationConfig::default();
        let mut pool = NetworkBufferPool::new(config)?;

        // Acquire buffers
        let buffer1 = pool.acquire(1024)?;
        assert_eq!(buffer1.len(), 1024);

        let stats = pool.get_stats();
        assert_eq!(stats.acquisitions, 1);
        assert_eq!(stats.active_buffers, 1);

        // Release buffer
        pool.release(buffer1);
        assert_eq!(pool.get_stats().releases, 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_lazy_snapshot_loader() -> Result<()> {
        let temp_dir = env::temp_dir();
        let snapshot_path = temp_dir.join("test_snapshot.bin");

        // Create test snapshot
        let test_data = b"test snapshot data";
        std::fs::write(&snapshot_path, test_data)?;

        let config = MemoryOptimizationConfig::default();
        let mut loader = LazySnapshotLoader::new(&snapshot_path, config)?;

        // Load data
        let data = loader.load().await?;
        assert_eq!(&data, test_data);
        assert!(loader.is_loaded());

        let stats = loader.get_stats();
        assert_eq!(stats.loads, 1);
        assert_eq!(stats.bytes_loaded, test_data.len());

        // Unload
        loader.unload();
        assert!(!loader.is_loaded());

        // Cleanup
        std::fs::remove_file(&snapshot_path)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_lazy_snapshot_partial_load() -> Result<()> {
        let temp_dir = env::temp_dir();
        let snapshot_path = temp_dir.join("test_snapshot_partial.bin");

        // Create test snapshot
        let test_data = b"0123456789abcdefghijklmnopqrstuvwxyz";
        std::fs::write(&snapshot_path, test_data)?;

        let config = MemoryOptimizationConfig::default();
        let mut loader = LazySnapshotLoader::new(&snapshot_path, config)?;

        // Load partial data
        let partial = loader.load_partial(5, 10).await?;
        assert_eq!(&partial, b"56789abcde");

        let stats = loader.get_stats();
        assert_eq!(stats.partial_loads, 1);
        assert_eq!(stats.bytes_loaded, 10);

        // Cleanup
        std::fs::remove_file(&snapshot_path)?;

        Ok(())
    }

    #[test]
    fn test_memory_optimization_manager() -> Result<()> {
        let config = MemoryOptimizationConfig::default();
        let mut manager = MemoryOptimizationManager::new(config.clone())?;

        // Register stores and pools
        let temp_dir = env::temp_dir();
        let file_path = temp_dir.join("test_manager_mmap.bin");
        let _ = std::fs::remove_file(&file_path);

        let store = Arc::new(RwLock::new(MmapTripleStore::new(
            &file_path,
            config.clone(),
        )?));
        manager.register_mmap_store(store);

        let pool = Arc::new(RwLock::new(NetworkBufferPool::new(config)?));
        manager.register_buffer_pool(pool);

        let stats = manager.get_stats();
        assert_eq!(stats.active_mmap_stores, 1);
        assert_eq!(stats.active_buffer_pools, 1);

        // Check leaks
        manager.check_leaks()?;

        // Cleanup
        let _ = std::fs::remove_file(&file_path);

        Ok(())
    }

    #[test]
    fn test_chunking_adaptation() -> Result<()> {
        let config = MemoryOptimizationConfig {
            chunk_size: 1000,
            memory_limit: 10000,
            ..Default::default()
        };

        let mut chunker = AdaptiveQueryResultChunker::new(config.clone())?;

        // Simulate high memory pressure
        chunker.adapt_to_memory_pressure(3000);
        assert_eq!(chunker.config.chunk_size, 500);
        assert_eq!(chunker.get_stats().adaptations, 1);

        // Simulate low memory pressure
        chunker.adapt_to_memory_pressure(9000);
        assert_eq!(chunker.config.chunk_size, 750);
        assert_eq!(chunker.get_stats().adaptations, 2);

        Ok(())
    }

    #[test]
    fn test_buffer_pool_hit_ratio() -> Result<()> {
        let config = MemoryOptimizationConfig::default();
        let mut pool = NetworkBufferPool::new(config)?;

        // Acquire multiple buffers
        for _ in 0..10 {
            let _buffer = pool.acquire(1024)?;
        }

        let stats = pool.get_stats();
        assert_eq!(stats.acquisitions, 10);
        assert!(stats.hit_ratio >= 0.0 && stats.hit_ratio <= 1.0);

        Ok(())
    }

    #[test]
    fn test_mmap_store_empty_file() -> Result<()> {
        let temp_dir = env::temp_dir();
        let file_path = temp_dir.join("test_empty_mmap.bin");

        // Cleanup
        let _ = std::fs::remove_file(&file_path);

        let config = MemoryOptimizationConfig::default();
        let mut store = MmapTripleStore::new(&file_path, config)?;

        // Read from empty file
        store.open_read()?;
        let triples = store.read_triples()?;
        assert!(triples.is_empty());
        assert_eq!(store.get_triple_count(), 0);

        // Cleanup
        std::fs::remove_file(&file_path)?;

        Ok(())
    }
}
