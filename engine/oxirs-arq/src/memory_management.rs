//! Professional-Grade Memory Management using SciRS2
//!
//! This module provides advanced memory management capabilities for the OxiRS query engine,
//! leveraging SciRS2's comprehensive memory management features for optimal performance
//! with large RDF datasets and complex SPARQL queries.

use crate::algebra::{Binding, Solution, Term};
use anyhow::Result;
use scirs2_core::array;  // Beta.3 array macro convenience
use scirs2_core::error::CoreError;
use scirs2_core::memory::BufferPool;
use scirs2_core::memory_efficient::ChunkedArray;
// Native SciRS2 APIs (beta.4+)
use scirs2_core::metrics::{Counter, Gauge, Histogram, Timer};
use scirs2_core::profiling::Profiler;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, ThreadLocalRngPool};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory management configuration for query execution
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Total memory limit for query execution (bytes)
    pub memory_limit: usize,
    /// Memory threshold for triggering cleanup (percentage)
    pub cleanup_threshold: f64,
    /// Chunk size for memory-mapped operations
    pub chunk_size: usize,
    /// Enable aggressive memory optimization
    pub aggressive_optimization: bool,
    /// Temporary directory for disk-backed operations
    pub temp_dir: Option<PathBuf>,
    /// Memory pressure management strategy
    pub pressure_strategy: MemoryPressureStrategy,
    /// Buffer pool configuration
    pub buffer_pool_config: BufferPoolConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB default
            cleanup_threshold: 0.8,                // 80% threshold
            chunk_size: 64 * 1024 * 1024,          // 64MB chunks
            aggressive_optimization: true,
            temp_dir: Some(std::env::temp_dir().join("oxirs_memory")),
            pressure_strategy: MemoryPressureStrategy::Adaptive,
            buffer_pool_config: BufferPoolConfig::default(),
        }
    }
}

/// Buffer pool configuration
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Initial buffer pool size
    pub initial_size: usize,
    /// Maximum buffer pool size
    pub max_size: usize,
    /// Buffer size for individual allocations
    pub buffer_size: usize,
    /// Enable buffer pool warming
    pub enable_warming: bool,
    /// Buffer pool growth strategy
    pub growth_strategy: BufferGrowthStrategy,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024, // 256MB
            max_size: 2 * 1024 * 1024 * 1024, // 2GB
            buffer_size: 4096,                // 4KB buffers
            enable_warming: true,
            growth_strategy: BufferGrowthStrategy::Exponential,
        }
    }
}

/// Memory pressure management strategies
#[derive(Debug, Clone, Copy)]
pub enum MemoryPressureStrategy {
    /// No automatic pressure management
    None,
    /// Conservative approach with early cleanup
    Conservative,
    /// Adaptive strategy based on system conditions
    Adaptive,
    /// Aggressive optimization with proactive management
    Aggressive,
}

/// Buffer pool growth strategies
#[derive(Debug, Clone, Copy)]
pub enum BufferGrowthStrategy {
    /// Linear growth
    Linear,
    /// Exponential growth
    Exponential,
    /// Adaptive growth based on usage patterns
    Adaptive,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage (bytes)
    pub current_usage: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Buffer pool hit ratio
    pub buffer_pool_hit_ratio: f64,
    /// Average allocation size
    pub avg_allocation_size: usize,
    /// Memory pressure level (0.0 to 1.0)
    pub pressure_level: f64,
}

/// Memory-managed query execution context
pub struct MemoryManagedContext {
    config: MemoryConfig,
    buffer_pool: Arc<RwLock<BufferPool<u8>>>,
    global_pool: Arc<GlobalBufferPool>,
    leak_detector: LeakDetector,
    metrics_collector: MemoryMetricsCollector,
    profiler: Profiler,

    // Memory tracking
    current_usage: Arc<Mutex<usize>>,
    peak_usage: Arc<Mutex<usize>>,
    allocation_count: Arc<Mutex<u64>>,
    deallocation_count: Arc<Mutex<u64>>,

    // Performance metrics
    memory_allocation_timer: Timer,
    memory_deallocation_timer: Timer,
    buffer_pool_hit_counter: Counter,
    buffer_pool_miss_counter: Counter,
    memory_pressure_gauge: Gauge,

    // Memory-managed data structures
    solution_cache: Arc<RwLock<MemoryManagedSolutionCache>>,
    term_dictionary: Arc<RwLock<MemoryManagedTermDictionary>>,
}

impl MemoryManagedContext {
    /// Create new memory-managed context
    pub fn new(config: MemoryConfig) -> Result<Self> {
        // Initialize buffer pools
        let buffer_pool = Arc::new(RwLock::new(BufferPool::new(
            config.buffer_pool_config.initial_size,
        )?));

        let global_pool = Arc::new(GlobalBufferPool::new(config.memory_limit)?);

        // Initialize monitoring
        let leak_detector = LeakDetector::new();
        let metrics_collector = MemoryMetricsCollector::new();
        let profiler = Profiler::new();

        // Create temporary directory if needed
        if let Some(ref temp_dir) = config.temp_dir {
            std::fs::create_dir_all(temp_dir)?;
        }

        Ok(Self {
            config,
            buffer_pool,
            global_pool,
            leak_detector,
            metrics_collector,
            profiler,
            current_usage: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
            allocation_count: Arc::new(Mutex::new(0)),
            deallocation_count: Arc::new(Mutex::new(0)),
            memory_allocation_timer: Timer::new("memory_allocation".to_string()),
            memory_deallocation_timer: Timer::new("memory_deallocation".to_string()),
            buffer_pool_hit_counter: Counter::new("buffer_pool_hits".to_string()),
            buffer_pool_miss_counter: Counter::new("buffer_pool_misses".to_string()),
            memory_pressure_gauge: Gauge::new("memory_pressure".to_string()),
            solution_cache: Arc::new(RwLock::new(MemoryManagedSolutionCache::new(
                config.memory_limit / 4,
            )?)),
            term_dictionary: Arc::new(RwLock::new(MemoryManagedTermDictionary::new(
                config.memory_limit / 8,
            )?)),
        })
    }

    /// Allocate memory with tracking and optimization
    pub fn allocate(&self, size: usize) -> Result<MemoryManagedBuffer> {
        self.profiler.start("memory_allocation");
        let start_time = Instant::now();

        // Check memory pressure before allocation
        self.check_memory_pressure()?;

        // Try buffer pool first for common sizes
        if size <= self.config.buffer_pool_config.buffer_size {
            if let Ok(buffer_pool) = self.buffer_pool.read() {
                if let Ok(buffer) = buffer_pool.acquire(size) {
                    self.buffer_pool_hit_counter.increment();
                    self.update_allocation_stats(size);
                    self.memory_allocation_timer.observe(start_time.elapsed());
                    self.profiler.stop("memory_allocation");

                    return Ok(MemoryManagedBuffer::BufferPool(buffer));
                }
            }
            self.buffer_pool_miss_counter.increment();
        }

        // Fall back to global pool
        let buffer = self.global_pool.allocate(size)?;
        self.update_allocation_stats(size);
        self.memory_allocation_timer.observe(start_time.elapsed());
        self.profiler.stop("memory_allocation");

        Ok(MemoryManagedBuffer::Global(buffer))
    }

    /// Deallocate memory with tracking
    pub fn deallocate(&self, buffer: MemoryManagedBuffer) -> Result<()> {
        self.profiler.start("memory_deallocation");
        let start_time = Instant::now();

        let size = buffer.size();

        match buffer {
            MemoryManagedBuffer::BufferPool(buffer) => {
                if let Ok(buffer_pool) = self.buffer_pool.read() {
                    buffer_pool.release(buffer)?;
                }
            }
            MemoryManagedBuffer::Global(buffer) => {
                self.global_pool.deallocate(buffer)?;
            }
            MemoryManagedBuffer::MemoryMapped(_) => {
                // Memory-mapped buffers are automatically cleaned up
            }
            MemoryManagedBuffer::DiskBacked(_) => {
                // Disk-backed buffers handle their own cleanup
            }
        }

        self.update_deallocation_stats(size);
        self.memory_deallocation_timer.observe(start_time.elapsed());
        self.profiler.stop("memory_deallocation");

        Ok(())
    }

    /// Create memory-mapped array for large datasets
    pub fn create_memory_mapped_array<T>(&self, path: &PathBuf, size: usize) -> Result<MemoryMappedArray<T>>
    where
        T: Clone + Default,
    {
        self.profiler.start("memory_mapping");

        let mmap_array = MemoryMappedArray::create(path, size)?;

        self.profiler.stop("memory_mapping");
        Ok(mmap_array)
    }

    /// Create disk-backed array for very large datasets
    pub fn create_disk_backed_array<T>(&self, capacity: usize) -> Result<DiskBackedArray<T>>
    where
        T: Clone + Default + serde::Serialize + serde::de::DeserializeOwned,
    {
        let temp_path = self
            .config
            .temp_dir
            .as_ref()
            .unwrap_or(&std::env::temp_dir())
            .join(format!("disk_backed_{}", uuid::Uuid::new_v4()));

        DiskBackedArray::new(temp_path, capacity)
    }

    /// Create lazy array for deferred loading
    pub fn create_lazy_array<T, F>(&self, size: usize, generator: F) -> LazyArray<T>
    where
        T: Clone,
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        LazyArray::new(size, generator)
    }

    /// Create adaptive chunking for large data processing
    pub fn create_adaptive_chunking(&self) -> Result<AdaptiveChunking> {
        AdaptiveChunking::new()
            .with_memory_limit(self.config.memory_limit)
            .with_target_chunk_size(self.config.chunk_size)
            .with_temp_dir(
                self.config
                    .temp_dir
                    .clone()
                    .unwrap_or_else(|| std::env::temp_dir()),
            )
            .build()
    }

    /// Process large solution sets with memory management
    pub fn process_large_solution(&self, solution: Solution) -> Result<ProcessedSolution> {
        if solution.len() * std::mem::size_of::<Binding>() > self.config.memory_limit / 2 {
            // Use disk-backed processing for very large solutions
            self.process_solution_disk_backed(solution)
        } else if solution.len() > 10000 {
            // Use chunked processing for moderately large solutions
            self.process_solution_chunked(solution)
        } else {
            // Use in-memory processing for small solutions
            Ok(ProcessedSolution::InMemory(solution))
        }
    }

    /// Process solution using disk-backed storage
    fn process_solution_disk_backed(&self, solution: Solution) -> Result<ProcessedSolution> {
        let mut disk_backed = self.create_disk_backed_array(solution.len())?;

        for binding in solution {
            disk_backed.push(binding)?;
        }

        Ok(ProcessedSolution::DiskBacked(disk_backed))
    }

    /// Process solution using chunked storage
    fn process_solution_chunked(&self, solution: Solution) -> Result<ProcessedSolution> {
        let chunking = self.create_adaptive_chunking()?;
        let chunk_processor = ChunkProcessor::new(chunking);

        let processed_chunks = chunk_processor.process_data(solution)?;

        Ok(ProcessedSolution::Chunked(processed_chunks))
    }

    /// Check and handle memory pressure
    fn check_memory_pressure(&self) -> Result<()> {
        let current_usage = *self.current_usage.lock().expect("lock should not be poisoned");
        let pressure_ratio = current_usage as f64 / self.config.memory_limit as f64;

        self.memory_pressure_gauge.set(pressure_ratio);

        if pressure_ratio > self.config.cleanup_threshold {
            match self.config.pressure_strategy {
                MemoryPressureStrategy::None => {
                    // Do nothing
                }
                MemoryPressureStrategy::Conservative => {
                    self.trigger_conservative_cleanup()?;
                }
                MemoryPressureStrategy::Adaptive => {
                    self.trigger_adaptive_cleanup(pressure_ratio)?;
                }
                MemoryPressureStrategy::Aggressive => {
                    self.trigger_aggressive_cleanup()?;
                }
            }
        }

        Ok(())
    }

    /// Trigger conservative memory cleanup
    fn trigger_conservative_cleanup(&self) -> Result<()> {
        // Clear solution cache
        if let Ok(mut cache) = self.solution_cache.write() {
            cache.clear_expired()?;
        }

        // Run garbage collection on buffer pools
        if let Ok(buffer_pool) = self.buffer_pool.write() {
            buffer_pool.garbage_collect()?;
        }

        Ok(())
    }

    /// Trigger adaptive memory cleanup based on pressure
    fn trigger_adaptive_cleanup(&self, pressure_ratio: f64) -> Result<()> {
        self.trigger_conservative_cleanup()?;

        if pressure_ratio > 0.9 {
            // Very high pressure - aggressive cleanup
            self.trigger_aggressive_cleanup()?;
        } else if pressure_ratio > 0.85 {
            // High pressure - clear more caches
            if let Ok(mut term_dict) = self.term_dictionary.write() {
                term_dict.compact()?;
            }
        }

        Ok(())
    }

    /// Trigger aggressive memory cleanup
    fn trigger_aggressive_cleanup(&self) -> Result<()> {
        // Clear all caches
        if let Ok(mut cache) = self.solution_cache.write() {
            cache.clear_all()?;
        }

        if let Ok(mut term_dict) = self.term_dictionary.write() {
            term_dict.clear_cache()?;
        }

        // Force buffer pool compaction
        if let Ok(mut buffer_pool) = self.buffer_pool.write() {
            buffer_pool.compact()?;
        }

        // Global garbage collection
        self.global_pool.force_garbage_collect()?;

        Ok(())
    }

    /// Update allocation statistics
    fn update_allocation_stats(&self, size: usize) {
        if let (Ok(mut current), Ok(mut peak), Ok(mut count)) = (
            self.current_usage.lock(),
            self.peak_usage.lock(),
            self.allocation_count.lock(),
        ) {
            *current += size;
            if *current > *peak {
                *peak = *current;
            }
            *count += 1;
        }
    }

    /// Update deallocation statistics
    fn update_deallocation_stats(&self, size: usize) {
        if let (Ok(mut current), Ok(mut count)) = (
            self.current_usage.lock(),
            self.deallocation_count.lock(),
        ) {
            *current = current.saturating_sub(size);
            *count += 1;
        }
    }

    /// Get comprehensive memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let current_usage = *self.current_usage.lock().expect("lock should not be poisoned");
        let peak_usage = *self.peak_usage.lock().expect("lock should not be poisoned");
        let allocation_count = *self.allocation_count.lock().expect("lock should not be poisoned");
        let deallocation_count = *self.deallocation_count.lock().expect("lock should not be poisoned");

        let buffer_pool_hits = self.buffer_pool_hit_counter.get();
        let buffer_pool_misses = self.buffer_pool_miss_counter.get();
        let buffer_pool_hit_ratio = if buffer_pool_hits + buffer_pool_misses > 0 {
            buffer_pool_hits as f64 / (buffer_pool_hits + buffer_pool_misses) as f64
        } else {
            0.0
        };

        let avg_allocation_size = if allocation_count > 0 {
            current_usage / allocation_count as usize
        } else {
            0
        };

        let pressure_level = current_usage as f64 / self.config.memory_limit as f64;

        // Calculate fragmentation ratio (simplified)
        let fragmentation_ratio = if allocation_count > deallocation_count {
            0.1 * (allocation_count - deallocation_count) as f64 / allocation_count as f64
        } else {
            0.0
        };

        MemoryStats {
            current_usage,
            peak_usage,
            allocation_count,
            deallocation_count,
            fragmentation_ratio,
            buffer_pool_hit_ratio,
            avg_allocation_size,
            pressure_level,
        }
    }

    /// Perform comprehensive memory leak detection
    pub fn detect_memory_leaks(&self) -> Result<MemoryLeakReport> {
        self.leak_detector.check()?;

        let stats = self.get_memory_stats();
        let leak_indicators = vec![];

        // Check for potential leaks
        let mut potential_leaks = Vec::new();

        if stats.allocation_count > stats.deallocation_count + 1000 {
            potential_leaks.push(format!(
                "Allocation/deallocation imbalance: {} allocations vs {} deallocations",
                stats.allocation_count, stats.deallocation_count
            ));
        }

        if stats.fragmentation_ratio > 0.3 {
            potential_leaks.push(format!(
                "High memory fragmentation: {:.2}%",
                stats.fragmentation_ratio * 100.0
            ));
        }

        if stats.pressure_level > 0.9 {
            potential_leaks.push(format!(
                "High memory pressure: {:.2}%",
                stats.pressure_level * 100.0
            ));
        }

        Ok(MemoryLeakReport {
            has_leaks: !potential_leaks.is_empty(),
            potential_leaks,
            stats,
        })
    }

    /// Optimize memory layout for better performance
    pub fn optimize_memory_layout(&self) -> Result<()> {
        self.profiler.start("memory_optimization");

        // Compact buffer pools
        if let Ok(mut buffer_pool) = self.buffer_pool.write() {
            buffer_pool.optimize_layout()?;
        }

        // Compact caches
        if let Ok(mut cache) = self.solution_cache.write() {
            cache.optimize()?;
        }

        if let Ok(mut term_dict) = self.term_dictionary.write() {
            term_dict.optimize()?;
        }

        self.profiler.stop("memory_optimization");
        Ok(())
    }

    /// Get memory management performance report
    pub fn get_performance_report(&self) -> MemoryPerformanceReport {
        let stats = self.get_memory_stats();

        MemoryPerformanceReport {
            total_allocations: stats.allocation_count,
            total_deallocations: stats.deallocation_count,
            peak_memory_usage_mb: stats.peak_usage / (1024 * 1024),
            current_memory_usage_mb: stats.current_usage / (1024 * 1024),
            buffer_pool_efficiency: stats.buffer_pool_hit_ratio,
            memory_fragmentation_percent: stats.fragmentation_ratio * 100.0,
            avg_allocation_time_ns: self.memory_allocation_timer.average().as_nanos() as u64,
            avg_deallocation_time_ns: self.memory_deallocation_timer.average().as_nanos() as u64,
            memory_pressure_level: stats.pressure_level,
        }
    }
}

/// Memory-managed buffer types
pub enum MemoryManagedBuffer {
    BufferPool(scirs2_core::memory::Buffer),
    Global(scirs2_core::memory::GlobalBuffer),
    MemoryMapped(MemoryMappedArray<u8>),
    DiskBacked(DiskBackedArray<u8>),
}

impl MemoryManagedBuffer {
    /// Get buffer size
    pub fn size(&self) -> usize {
        match self {
            Self::BufferPool(buf) => buf.len(),
            Self::Global(buf) => buf.len(),
            Self::MemoryMapped(mmap) => mmap.len(),
            Self::DiskBacked(disk) => disk.len(),
        }
    }
}

/// Processed solution with memory management
pub enum ProcessedSolution {
    InMemory(Solution),
    Chunked(Vec<ChunkedArray<Binding>>),
    DiskBacked(DiskBackedArray<Binding>),
}

/// Memory-managed solution cache
struct MemoryManagedSolutionCache {
    cache: HashMap<String, (Solution, Instant)>,
    memory_limit: usize,
    current_usage: usize,
}

impl MemoryManagedSolutionCache {
    fn new(memory_limit: usize) -> Result<Self> {
        Ok(Self {
            cache: HashMap::new(),
            memory_limit,
            current_usage: 0,
        })
    }

    fn clear_expired(&mut self) -> Result<()> {
        let now = Instant::now();
        let expiry_duration = Duration::from_secs(300); // 5 minutes

        self.cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp) < expiry_duration
        });

        Ok(())
    }

    fn clear_all(&mut self) -> Result<()> {
        self.cache.clear();
        self.current_usage = 0;
        Ok(())
    }

    fn optimize(&mut self) -> Result<()> {
        self.clear_expired()?;

        // Additional optimization logic could go here
        Ok(())
    }
}

/// Memory-managed term dictionary
struct MemoryManagedTermDictionary {
    dictionary: HashMap<Term, u64>,
    reverse_dict: HashMap<u64, Term>,
    memory_limit: usize,
    current_usage: usize,
}

impl MemoryManagedTermDictionary {
    fn new(memory_limit: usize) -> Result<Self> {
        Ok(Self {
            dictionary: HashMap::new(),
            reverse_dict: HashMap::new(),
            memory_limit,
            current_usage: 0,
        })
    }

    fn compact(&mut self) -> Result<()> {
        // Remove unused entries (simplified)
        if self.current_usage > self.memory_limit {
            let target_size = self.memory_limit / 2;
            let remove_count = self.dictionary.len() - target_size;

            let keys_to_remove: Vec<_> = self.dictionary.keys().take(remove_count).cloned().collect();
            for key in keys_to_remove {
                if let Some(id) = self.dictionary.remove(&key) {
                    self.reverse_dict.remove(&id);
                }
            }
        }

        Ok(())
    }

    fn clear_cache(&mut self) -> Result<()> {
        self.dictionary.clear();
        self.reverse_dict.clear();
        self.current_usage = 0;
        Ok(())
    }

    fn optimize(&mut self) -> Result<()> {
        self.compact()?;
        Ok(())
    }
}

/// Memory leak detection report
#[derive(Debug)]
pub struct MemoryLeakReport {
    pub has_leaks: bool,
    pub potential_leaks: Vec<String>,
    pub stats: MemoryStats,
}

/// Memory management performance report
#[derive(Debug, Clone)]
pub struct MemoryPerformanceReport {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub peak_memory_usage_mb: usize,
    pub current_memory_usage_mb: usize,
    pub buffer_pool_efficiency: f64,
    pub memory_fragmentation_percent: f64,
    pub avg_allocation_time_ns: u64,
    pub avg_deallocation_time_ns: u64,
    pub memory_pressure_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_managed_context_creation() {
        let config = MemoryConfig::default();
        let context = MemoryManagedContext::new(config);
        assert!(context.is_ok());
    }

    #[test]
    fn test_memory_allocation_deallocation() {
        let config = MemoryConfig::default();
        let context = MemoryManagedContext::new(config).unwrap();

        let buffer = context.allocate(1024).unwrap();
        assert_eq!(buffer.size(), 1024);

        assert!(context.deallocate(buffer).is_ok());
    }

    #[test]
    fn test_memory_stats() {
        let config = MemoryConfig::default();
        let context = MemoryManagedContext::new(config).unwrap();

        let _buffer = context.allocate(1024).unwrap();

        let stats = context.get_memory_stats();
        assert!(stats.allocation_count > 0);
        assert!(stats.current_usage > 0);
    }

    #[test]
    fn test_memory_leak_detection() {
        let config = MemoryConfig::default();
        let context = MemoryManagedContext::new(config).unwrap();

        let leak_report = context.detect_memory_leaks().unwrap();
        assert!(!leak_report.has_leaks); // Should not have leaks in test
    }

    #[test]
    fn test_adaptive_chunking() {
        let config = MemoryConfig::default();
        let context = MemoryManagedContext::new(config).unwrap();

        let chunking = context.create_adaptive_chunking();
        assert!(chunking.is_ok());
    }
}