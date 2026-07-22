//! Advanced memory mapping features for large datasets
//!
//! This module provides advanced memory mapping capabilities including:
//! - Lazy loading with page-level access
//! - Smart caching and eviction policies
//! - NUMA-aware memory allocation
//! - Swapping policies for memory pressure

use anyhow::{bail, Result};
use lru::LruCache;
use memmap2::Mmap;
use oxirs_core::parallel::*;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::warn;

/// Page size for lazy loading (16KB for better vector alignment)
const VECTOR_PAGE_SIZE: usize = 16384;

/// Maximum number of pages to keep in memory
const DEFAULT_MAX_PAGES: usize = 10000;

/// NUMA node information
#[cfg(target_os = "linux")]
mod numa {
    use libc::{c_ulong, c_void};

    extern "C" {
        fn numa_available() -> i32;
        fn numa_max_node() -> i32;
        fn numa_node_of_cpu(cpu: i32) -> i32;
        fn numa_alloc_onnode(size: usize, node: i32) -> *mut c_void;
        fn numa_free(ptr: *mut c_void, size: usize);
        fn mbind(
            addr: *mut c_void,
            len: c_ulong,
            mode: i32,
            nodemask: *const c_ulong,
            maxnode: c_ulong,
            flags: u32,
        ) -> i32;
    }

    pub const MPOL_BIND: i32 = 2;
    pub const MPOL_INTERLEAVE: i32 = 3;

    pub fn is_available() -> bool {
        unsafe { numa_available() >= 0 }
    }

    pub fn max_node() -> i32 {
        unsafe { numa_max_node() }
    }

    pub fn node_of_cpu(cpu: i32) -> i32 {
        unsafe { numa_node_of_cpu(cpu) }
    }
}

#[cfg(not(target_os = "linux"))]
mod numa {
    pub fn is_available() -> bool {
        false
    }
    pub fn max_node() -> i32 {
        0
    }
    pub fn node_of_cpu(_cpu: i32) -> i32 {
        0
    }
}

/// Page access pattern for predictive prefetching
#[derive(Debug, Clone)]
struct AccessPattern {
    page_id: usize,
    access_time: Instant,
    access_count: usize,
}

/// Page cache entry with metadata
#[derive(Debug)]
pub struct PageCacheEntry {
    data: Vec<u8>,
    page_id: usize,
    last_access: Instant,
    /// When the page was first inserted into the cache. Drives FIFO eviction
    /// (oldest insertion evicted first), independent of subsequent accesses.
    inserted_at: Instant,
    access_count: AtomicUsize,
    /// Clock-algorithm reference bit: set on every access, cleared to grant a
    /// "second chance" during a Clock eviction sweep.
    reference_bit: AtomicBool,
    dirty: bool,
    numa_node: i32,
}

impl PageCacheEntry {
    /// Get the data slice
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the NUMA node
    pub fn numa_node(&self) -> i32 {
        self.numa_node
    }
}

/// Eviction policy for page cache
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    LRU,   // Least Recently Used
    LFU,   // Least Frequently Used
    FIFO,  // First In First Out
    Clock, // Clock algorithm
    ARC,   // Adaptive Replacement Cache
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

/// Advanced memory-mapped vector storage
pub struct AdvancedMemoryMap {
    /// Base file mapping
    mmap: Option<Mmap>,

    /// Path to the backing file for dirty-page write-back
    file_path: Option<std::path::PathBuf>,

    /// Page cache
    page_cache: Arc<RwLock<LruCache<usize, Arc<PageCacheEntry>>>>,

    /// Access pattern tracking
    access_patterns: Arc<RwLock<VecDeque<AccessPattern>>>,

    /// Page access frequency
    page_frequency: Arc<RwLock<HashMap<usize, usize>>>,

    /// Eviction policy
    eviction_policy: EvictionPolicy,

    /// Memory statistics
    total_memory: AtomicUsize,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,

    /// NUMA configuration
    numa_enabled: bool,
    numa_nodes: Vec<i32>,

    /// Memory pressure monitor
    memory_pressure: Arc<RwLock<MemoryPressure>>,

    /// Configuration
    max_pages: usize,
    page_size: usize,
    prefetch_distance: usize,
}

impl AdvancedMemoryMap {
    /// Create a new advanced memory map
    pub fn new(mmap: Option<Mmap>, max_pages: usize) -> Self {
        let numa_enabled = numa::is_available();
        let numa_nodes = if numa_enabled {
            (0..=numa::max_node()).collect()
        } else {
            vec![0]
        };

        let cache_size = NonZeroUsize::new(max_pages)
            .unwrap_or(NonZeroUsize::new(1).expect("constant 1 is non-zero"));

        Self {
            mmap,
            file_path: None,
            page_cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            access_patterns: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            page_frequency: Arc::new(RwLock::new(HashMap::new())),
            eviction_policy: EvictionPolicy::ARC,
            total_memory: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            numa_enabled,
            numa_nodes,
            memory_pressure: Arc::new(RwLock::new(MemoryPressure::Low)),
            max_pages,
            page_size: VECTOR_PAGE_SIZE,
            prefetch_distance: 3,
        }
    }

    /// Create a new advanced memory map with a backing file path for dirty-page write-back
    pub fn new_with_path(
        mmap: Option<Mmap>,
        max_pages: usize,
        file_path: Option<std::path::PathBuf>,
    ) -> Self {
        let mut s = Self::new(mmap, max_pages);
        s.file_path = file_path;
        s
    }

    /// Get a page with lazy loading
    pub fn get_page(&self, page_id: usize) -> Result<Arc<PageCacheEntry>> {
        // Check cache first
        {
            let mut cache = self.page_cache.write();
            if let Some(entry) = cache.get(&page_id) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                // Mark the page as recently referenced for the Clock algorithm.
                entry.reference_bit.store(true, Ordering::Relaxed);
                self.record_access(page_id);
                return Ok(Arc::clone(entry));
            }
        }

        // Cache miss - load from mmap
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.load_page(page_id)
    }

    /// Load a page from memory-mapped file
    fn load_page(&self, page_id: usize) -> Result<Arc<PageCacheEntry>> {
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No memory mapping available"))?;

        let start = page_id * self.page_size;
        let end = (start + self.page_size).min(mmap.len());

        if start >= mmap.len() {
            bail!("Page {} out of bounds", page_id);
        }

        // Copy page data
        let page_data = mmap[start..end].to_vec();

        // Determine NUMA node for allocation
        let numa_node = if self.numa_enabled {
            let cpu = sched_getcpu();
            numa::node_of_cpu(cpu)
        } else {
            0
        };

        let now = Instant::now();
        let entry = Arc::new(PageCacheEntry {
            data: page_data,
            page_id,
            last_access: now,
            inserted_at: now,
            access_count: AtomicUsize::new(1),
            reference_bit: AtomicBool::new(true),
            dirty: false,
            numa_node,
        });

        // Check memory pressure and evict if needed
        self.check_memory_pressure();
        if *self.memory_pressure.read() >= MemoryPressure::High {
            self.evict_pages(1)?;
        }

        // Insert into cache
        {
            let mut cache = self.page_cache.write();
            cache.put(page_id, Arc::clone(&entry));
        }

        self.total_memory
            .fetch_add(entry.data.len(), Ordering::Relaxed);
        self.record_access(page_id);

        // Predictive prefetching
        self.prefetch_pages(page_id);

        Ok(entry)
    }

    /// Record page access for pattern analysis
    fn record_access(&self, page_id: usize) {
        let mut patterns = self.access_patterns.write();
        patterns.push_back(AccessPattern {
            page_id,
            access_time: Instant::now(),
            access_count: 1,
        });

        // Keep only recent patterns
        while patterns.len() > 1000 {
            patterns.pop_front();
        }

        // Update frequency map
        let mut freq = self.page_frequency.write();
        *freq.entry(page_id).or_insert(0) += 1;
    }

    /// Predictive prefetching based on access patterns
    fn prefetch_pages(&self, current_page: usize) {
        let patterns = self.access_patterns.read();
        let freq = self.page_frequency.read();

        // Analyze recent access patterns for intelligent prefetching
        let recent_patterns: Vec<_> = patterns.iter().rev().take(10).collect();

        // Check for sequential access pattern
        let is_sequential = recent_patterns
            .windows(2)
            .all(|w| w[0].page_id > 0 && w[0].page_id == w[1].page_id + 1);

        // Check for strided access pattern
        let stride = if recent_patterns.len() >= 3 {
            let diff1 = recent_patterns[0]
                .page_id
                .saturating_sub(recent_patterns[1].page_id);
            let diff2 = recent_patterns[1]
                .page_id
                .saturating_sub(recent_patterns[2].page_id);
            if diff1 == diff2 && diff1 > 0 && diff1 <= 10 {
                Some(diff1)
            } else {
                None
            }
        } else {
            None
        };

        // Adaptive prefetching based on patterns
        if is_sequential {
            // Aggressive sequential prefetching
            for i in 1..=(self.prefetch_distance * 2) {
                let prefetch_page = current_page + i;
                self.async_prefetch(prefetch_page);
            }
        } else if let Some(stride) = stride {
            // Strided prefetching
            for i in 1..=self.prefetch_distance {
                let prefetch_page = current_page + (i * stride);
                self.async_prefetch(prefetch_page);
            }
        } else {
            // Conservative prefetching with frequency-based hints
            for i in 1..=self.prefetch_distance {
                let prefetch_page = current_page + i;

                // Check if this page has been accessed frequently
                let frequency = *freq.get(&prefetch_page).unwrap_or(&0);
                if frequency > 0 {
                    self.async_prefetch(prefetch_page);
                }
            }
        }

        // Prefetch frequently accessed pages near current page
        let nearby_range = current_page.saturating_sub(3)..=(current_page + 3);
        for page_id in nearby_range {
            let frequency = *freq.get(&page_id).unwrap_or(&0);
            if frequency > 2 && page_id != current_page {
                self.async_prefetch(page_id);
            }
        }
    }

    /// Asynchronous prefetch with throttling
    pub fn async_prefetch(&self, page_id: usize) {
        // Check if page is already in cache
        {
            let cache = self.page_cache.read();
            if cache.contains(&page_id) {
                return;
            }
        }

        // Check memory pressure before prefetching
        if *self.memory_pressure.read() >= MemoryPressure::High {
            return;
        }

        let self_clone = self.clone_ref();
        spawn(move || {
            let _ = self_clone.get_page(page_id);
        });
    }

    /// Check system memory pressure
    fn check_memory_pressure(&self) {
        let total_memory = self.total_memory.load(Ordering::Relaxed);
        let max_memory = self.max_pages * self.page_size;

        let pressure = if total_memory < max_memory / 2 {
            MemoryPressure::Low
        } else if total_memory < max_memory * 3 / 4 {
            MemoryPressure::Medium
        } else if total_memory < max_memory * 9 / 10 {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        };

        *self.memory_pressure.write() = pressure;
    }

    /// Evict pages based on eviction policy
    fn evict_pages(&self, num_pages: usize) -> Result<()> {
        match self.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(num_pages),
            EvictionPolicy::LFU => self.evict_lfu(num_pages),
            EvictionPolicy::FIFO => self.evict_fifo(num_pages),
            EvictionPolicy::Clock => self.evict_clock(num_pages),
            EvictionPolicy::ARC => self.evict_arc(num_pages),
        }
    }

    /// LRU eviction
    fn evict_lru(&self, num_pages: usize) -> Result<()> {
        let mut cache = self.page_cache.write();

        // LruCache automatically evicts least recently used
        for _ in 0..num_pages {
            if let Some((_, entry)) = cache.pop_lru() {
                self.total_memory
                    .fetch_sub(entry.data.len(), Ordering::Relaxed);

                // Write back if dirty
                if entry.dirty {
                    if let Err(e) = self.write_back_page(entry.page_id, &entry.data) {
                        warn!("Failed to write back page {}: {}", entry.page_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// LFU eviction
    fn evict_lfu(&self, num_pages: usize) -> Result<()> {
        let cache = self.page_cache.read();
        let freq = self.page_frequency.read();

        // Sort pages by frequency
        let mut pages_by_freq: Vec<(usize, usize)> = cache
            .iter()
            .map(|(page_id, _)| (*page_id, *freq.get(page_id).unwrap_or(&0)))
            .collect();
        pages_by_freq.sort_by_key(|(_, freq)| *freq);

        // Evict least frequently used
        drop(cache);
        drop(freq);

        let mut cache = self.page_cache.write();
        for (page_id, _) in pages_by_freq.iter().take(num_pages) {
            if let Some(entry) = cache.pop(page_id) {
                self.total_memory
                    .fetch_sub(entry.data.len(), Ordering::Relaxed);
                if entry.dirty {
                    if let Err(e) = self.write_back_page(entry.page_id, &entry.data) {
                        warn!("Failed to write back dirty page {}: {}", entry.page_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// FIFO eviction: evict pages in insertion order (oldest `inserted_at`
    /// first), regardless of how recently they were accessed. This is the key
    /// behavioral difference from LRU and avoids LRU thrashing under scan-heavy
    /// workloads.
    fn evict_fifo(&self, num_pages: usize) -> Result<()> {
        // Snapshot (page_id, inserted_at) under a read lock, then evict the
        // oldest under a write lock.
        let mut pages_by_age: Vec<(usize, Instant)> = {
            let cache = self.page_cache.read();
            cache
                .iter()
                .map(|(page_id, entry)| (*page_id, entry.inserted_at))
                .collect()
        };
        pages_by_age.sort_by_key(|(_, inserted_at)| *inserted_at);

        let mut cache = self.page_cache.write();
        for (page_id, _) in pages_by_age.iter().take(num_pages) {
            if let Some(entry) = cache.pop(page_id) {
                self.total_memory
                    .fetch_sub(entry.data.len(), Ordering::Relaxed);
                if entry.dirty {
                    if let Err(e) = self.write_back_page(entry.page_id, &entry.data) {
                        warn!("Failed to write back page {}: {}", entry.page_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Clock (second-chance) eviction: sweep pages in a stable circular order;
    /// a page whose reference bit is set is given a second chance (bit cleared,
    /// page retained), a page whose bit is clear is evicted. Bounded to a few
    /// sweeps so it always terminates even if every page was recently touched.
    fn evict_clock(&self, num_pages: usize) -> Result<()> {
        if num_pages == 0 {
            return Ok(());
        }

        let mut cache = self.page_cache.write();

        // Stable circular order by page_id so the "clock hand" is deterministic.
        let mut order: Vec<usize> = cache.iter().map(|(page_id, _)| *page_id).collect();
        order.sort_unstable();
        if order.is_empty() {
            return Ok(());
        }

        let mut to_evict: Vec<usize> = Vec::with_capacity(num_pages);
        // At most 2 full sweeps: pass 1 may clear reference bits, pass 2 then
        // finds victims with cleared bits. A tiny extra margin guards rounding.
        let max_steps = order.len() * 3 + num_pages;
        let mut hand = 0usize;
        let mut steps = 0usize;

        while to_evict.len() < num_pages && steps < max_steps {
            let page_id = order[hand % order.len()];
            hand += 1;
            steps += 1;

            if let Some(entry) = cache.peek(&page_id) {
                if entry.reference_bit.swap(false, Ordering::Relaxed) {
                    // Reference bit was set: grant a second chance (now cleared).
                    continue;
                }
                // Reference bit clear: this page is a victim.
                to_evict.push(page_id);
            }
        }

        for page_id in to_evict {
            if let Some(entry) = cache.pop(&page_id) {
                self.total_memory
                    .fetch_sub(entry.data.len(), Ordering::Relaxed);
                if entry.dirty {
                    if let Err(e) = self.write_back_page(entry.page_id, &entry.data) {
                        warn!("Failed to write back page {}: {}", entry.page_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// ARC (Adaptive Replacement Cache) eviction
    fn evict_arc(&self, num_pages: usize) -> Result<()> {
        // Simplified ARC - combines recency and frequency
        let cache = self.page_cache.read();
        let freq = self.page_frequency.read();

        // Score = recency * 0.5 + frequency * 0.5
        let now = Instant::now();
        let mut scored_pages: Vec<(usize, f64)> = cache
            .iter()
            .map(|(page_id, entry)| {
                let recency_score =
                    1.0 / (now.duration_since(entry.last_access).as_secs_f64() + 1.0);
                let frequency_score = *freq.get(page_id).unwrap_or(&0) as f64;
                let combined_score = recency_score * 0.5 + frequency_score * 0.5;
                (*page_id, combined_score)
            })
            .collect();

        scored_pages.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        drop(cache);
        drop(freq);

        let mut cache = self.page_cache.write();
        for (page_id, _) in scored_pages.iter().take(num_pages) {
            if let Some(entry) = cache.pop(page_id) {
                self.total_memory
                    .fetch_sub(entry.data.len(), Ordering::Relaxed);
                if entry.dirty {
                    if let Err(e) = self.write_back_page(entry.page_id, &entry.data) {
                        warn!("Failed to write back dirty page {}: {}", entry.page_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> MemoryMapStats {
        let cache = self.page_cache.read();

        MemoryMapStats {
            total_pages: cache.len(),
            total_memory: self.total_memory.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            hit_rate: self.calculate_hit_rate(),
            memory_pressure: *self.memory_pressure.read(),
            numa_enabled: self.numa_enabled,
        }
    }

    fn calculate_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }

    fn clone_ref(&self) -> Self {
        Self {
            mmap: None, // Don't clone the mmap
            file_path: self.file_path.clone(),
            page_cache: Arc::clone(&self.page_cache),
            access_patterns: Arc::clone(&self.access_patterns),
            page_frequency: Arc::clone(&self.page_frequency),
            eviction_policy: self.eviction_policy,
            total_memory: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            numa_enabled: self.numa_enabled,
            numa_nodes: self.numa_nodes.clone(),
            memory_pressure: Arc::clone(&self.memory_pressure),
            max_pages: self.max_pages,
            page_size: self.page_size,
            prefetch_distance: self.prefetch_distance,
        }
    }

    /// Write a dirty page back to the backing file.
    fn write_back_page(&self, page_id: usize, data: &[u8]) -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let path = match &self.file_path {
            Some(p) => p,
            None => return Ok(()), // No file path configured — skip write-back
        };
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open file for write-back: {}", e))?;
        let offset = (page_id * self.page_size) as u64;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| anyhow::anyhow!("Failed to seek to page {}: {}", page_id, e))?;
        file.write_all(data)
            .map_err(|e| anyhow::anyhow!("Failed to write page {}: {}", page_id, e))?;
        Ok(())
    }

    /// Flush all dirty pages back to the backing file.
    pub fn flush_dirty_pages(&self) -> Result<()> {
        if self.file_path.is_none() {
            return Ok(());
        }
        let cache = self.page_cache.read();
        for (_, entry) in cache.iter() {
            if entry.dirty {
                self.write_back_page(entry.page_id, &entry.data)?;
            }
        }
        Ok(())
    }
}

/// Statistics for memory-mapped storage
#[derive(Debug, Clone)]
pub struct MemoryMapStats {
    pub total_pages: usize,
    pub total_memory: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f64,
    pub memory_pressure: MemoryPressure,
    pub numa_enabled: bool,
}

/// Get current CPU for NUMA operations
#[cfg(target_os = "linux")]
fn sched_getcpu() -> i32 {
    unsafe { libc::sched_getcpu() }
}

#[cfg(not(target_os = "linux"))]
fn sched_getcpu() -> i32 {
    0
}

/// NUMA-aware vector allocator
pub struct NumaVectorAllocator {
    numa_nodes: Vec<i32>,
    current_node: AtomicUsize,
}

impl Default for NumaVectorAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl NumaVectorAllocator {
    pub fn new() -> Self {
        let numa_nodes = if numa::is_available() {
            (0..=numa::max_node()).collect()
        } else {
            vec![0]
        };

        Self {
            numa_nodes,
            current_node: AtomicUsize::new(0),
        }
    }

    /// Allocate vector memory on specific NUMA node
    pub fn allocate_on_node(&self, size: usize, node: Option<i32>) -> Vec<u8> {
        if !numa::is_available() {
            return vec![0u8; size];
        }

        let _target_node = node.unwrap_or_else(|| {
            // Round-robin allocation across NUMA nodes
            let idx = self.current_node.fetch_add(1, Ordering::Relaxed) % self.numa_nodes.len();
            self.numa_nodes[idx]
        });

        // For now, just use standard allocation
        // TODO: Implement actual NUMA allocation when libc bindings are available
        vec![0u8; size]
    }

    /// Allocate optimized vector with NUMA awareness (specialized for f32 vectors)
    pub fn allocate_vector_on_node(&self, dimensions: usize, node: Option<i32>) -> Vec<f32> {
        if !numa::is_available() {
            // Pre-allocate with optimal alignment for SIMD operations
            let mut vec = Vec::with_capacity(dimensions);
            vec.resize(dimensions, 0.0f32);
            return vec;
        }

        let _target_node = node.unwrap_or_else(|| {
            // Use current CPU's NUMA node for better locality
            self.preferred_node()
        });

        // For better performance, use aligned allocation
        let mut vec = Vec::with_capacity(dimensions);
        vec.resize(dimensions, 0.0f32);

        // TODO: When NUMA bindings are available, use numa_alloc_onnode
        // and bind the memory to the specific node

        vec
    }

    /// Get preferred NUMA node for current thread
    pub fn preferred_node(&self) -> i32 {
        if numa::is_available() {
            numa::node_of_cpu(sched_getcpu())
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pressure() {
        let mmap = AdvancedMemoryMap::new(None, 100);

        assert_eq!(*mmap.memory_pressure.read(), MemoryPressure::Low);

        // Simulate memory usage
        mmap.total_memory
            .store(50 * VECTOR_PAGE_SIZE, Ordering::Relaxed);
        mmap.check_memory_pressure();
        assert_eq!(*mmap.memory_pressure.read(), MemoryPressure::Medium);

        mmap.total_memory
            .store(90 * VECTOR_PAGE_SIZE, Ordering::Relaxed);
        mmap.check_memory_pressure();
        assert_eq!(*mmap.memory_pressure.read(), MemoryPressure::Critical);
    }

    #[test]
    fn test_cache_stats() {
        let mmap = AdvancedMemoryMap::new(None, 100);

        mmap.cache_hits.store(75, Ordering::Relaxed);
        mmap.cache_misses.store(25, Ordering::Relaxed);

        let stats = mmap.stats();
        assert_eq!(stats.cache_hits, 75);
        assert_eq!(stats.cache_misses, 25);
        assert_eq!(stats.hit_rate, 0.75);
    }

    /// Insert a synthetic (clean) page directly into the cache for eviction
    /// tests, controlling its reference bit.
    fn insert_test_page(map: &AdvancedMemoryMap, page_id: usize, referenced: bool) {
        let now = Instant::now();
        let entry = Arc::new(PageCacheEntry {
            data: vec![0u8; 8],
            page_id,
            last_access: now,
            inserted_at: now,
            access_count: AtomicUsize::new(1),
            reference_bit: AtomicBool::new(referenced),
            dirty: false,
            numa_node: 0,
        });
        map.page_cache.write().put(page_id, entry);
    }

    #[test]
    fn regression_fifo_evicts_oldest_not_lru() {
        let map = AdvancedMemoryMap::new(None, 100);
        // Insert in order 0,1,2 -> page 0 is the oldest by insertion time.
        insert_test_page(&map, 0, false);
        insert_test_page(&map, 1, false);
        insert_test_page(&map, 2, false);

        // "Recently use" page 0 the way LRU would track (bump its recency in the
        // LruCache). FIFO must still evict page 0 because it was inserted first.
        {
            let mut cache = map.page_cache.write();
            let _ = cache.get(&0);
        }

        map.evict_fifo(1).expect("fifo eviction");

        let cache = map.page_cache.read();
        assert!(
            cache.peek(&0).is_none(),
            "FIFO must evict the first-inserted page (0)"
        );
        assert!(cache.peek(&1).is_some());
        assert!(cache.peek(&2).is_some());
    }

    #[test]
    fn regression_clock_gives_second_chance() {
        let map = AdvancedMemoryMap::new(None, 100);
        // Sweep order is by page_id ascending: [0, 1, 2].
        // page 0 is referenced (gets a second chance), page 1 is not (victim).
        insert_test_page(&map, 0, true);
        insert_test_page(&map, 1, false);
        insert_test_page(&map, 2, false);

        map.evict_clock(1).expect("clock eviction");

        let cache = map.page_cache.read();
        assert!(
            cache.peek(&0).is_some(),
            "referenced page 0 must survive one Clock sweep (second chance)"
        );
        assert!(
            cache.peek(&1).is_none(),
            "unreferenced page 1 must be the Clock victim"
        );
        // Page 0's reference bit must have been cleared by the sweep.
        assert!(
            !cache
                .peek(&0)
                .expect("page 0 present")
                .reference_bit
                .load(Ordering::Relaxed),
            "Clock sweep must clear the reference bit it consumed"
        );
    }
}
