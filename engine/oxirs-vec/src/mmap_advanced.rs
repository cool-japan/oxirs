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
use tracing::{trace, warn};

/// Page size for lazy loading (16KB for better vector alignment)
const VECTOR_PAGE_SIZE: usize = 16384;

/// Maximum number of pages to keep in memory
const DEFAULT_MAX_PAGES: usize = 10000;

/// NUMA topology discovery and memory-policy binding (Pure Rust).
///
/// This module deliberately avoids `libnuma`: the topology is read straight
/// from sysfs (`/sys/devices/system/node`, `/sys/devices/system/cpu`) and the
/// memory policy is applied with the raw `mbind(2)` system call through
/// `libc::syscall`. That keeps `oxirs-vec` free of any C link-time dependency
/// per the COOLJAPAN Pure Rust Policy, while providing exactly the three
/// topology queries and the one policy call the crate actually needs.
///
/// NOTE: CUDA detection/linking was likewise removed from `oxirs-vec`; real
/// NVIDIA CUDA acceleration lives in the quarantined `oxirs-vec-adapter-cuda`
/// crate (`publish = false`), keeping this crate's published `--all-features`
/// surface free of `cuda-runtime-sys`. Together with the removal of the
/// `libnuma` link directive above, `oxirs-vec` no longer needs a build script.
#[cfg(target_os = "linux")]
mod numa {
    use libc::{c_int, c_uint, c_ulong, c_void};
    use std::sync::OnceLock;

    /// sysfs root exposing the NUMA node topology.
    const NODE_SYSFS_ROOT: &str = "/sys/devices/system/node";

    /// sysfs root exposing the per-CPU topology.
    const CPU_SYSFS_ROOT: &str = "/sys/devices/system/cpu";

    /// Upper bound on node/CPU ids we are willing to materialise from a sysfs
    /// range such as `0-3`. The kernel's `MAX_NUMNODES` is at most 1024 and
    /// `CONFIG_NR_CPUS` tops out well below this, so anything larger indicates
    /// corrupt input rather than a real machine; clamping keeps a bogus
    /// `0-4294967295` from turning into a huge allocation.
    const MAX_SUPPORTED_ID: i32 = 65535;

    /// Number of bits carried by one `c_ulong` word of an `mbind` node mask.
    const NODEMASK_BITS: usize = std::mem::size_of::<c_ulong>() * 8;

    /// `mbind(2)` policy: allocate strictly from the nodes in the mask.
    pub const MPOL_BIND: c_int = libc::MPOL_BIND;

    /// `mbind(2)` policy: spread page allocations round-robin over the mask.
    pub const MPOL_INTERLEAVE: c_int = libc::MPOL_INTERLEAVE;

    /// Parse a kernel "cpuset list" such as `0`, `0-1` or `0-1,4` into the
    /// sorted, de-duplicated set of ids it denotes.
    ///
    /// The format is used verbatim by `node/possible`, `node/online` and
    /// `node*/cpulist`. Unparseable or inverted entries are skipped rather than
    /// treated as fatal: sysfs is advisory here and a partial answer is always
    /// better than a panic. Never panics, never allocates unboundedly.
    pub fn parse_cpuset_list(raw: &str) -> Vec<i32> {
        let mut ids: Vec<i32> = Vec::new();

        for entry in raw.trim().split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }

            // An entry is either `N` or `N-M`. Some kernels append a stride
            // (`N-M:S/T`); we only need the plain range, so anything after a
            // ':' is ignored conservatively by taking the range part.
            let range_part = entry.split(':').next().unwrap_or(entry);
            let (start, end) = match range_part.split_once('-') {
                Some((lo, hi)) => {
                    let (lo, hi) = (lo.trim(), hi.trim());
                    match (lo.parse::<i32>(), hi.parse::<i32>()) {
                        (Ok(lo), Ok(hi)) => (lo, hi),
                        _ => continue,
                    }
                }
                None => match range_part.parse::<i32>() {
                    Ok(single) => (single, single),
                    Err(_) => continue,
                },
            };

            // Reject inverted ranges, negative ids and absurd upper bounds.
            if start < 0 || end < start || start > MAX_SUPPORTED_ID {
                continue;
            }
            let end = end.min(MAX_SUPPORTED_ID);
            ids.extend(start..=end);
        }

        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Read a sysfs file and trim it, returning `None` on any I/O failure.
    fn read_sysfs(path: &str) -> Option<String> {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
    }

    /// Discover the NUMA nodes of this machine from sysfs.
    ///
    /// `node/possible` is preferred (it covers hot-pluggable nodes that are
    /// currently offline); `node/online` is the fallback. A machine without a
    /// NUMA topology is reported as the single node 0 so that callers always
    /// have a non-empty list to index.
    fn discover_nodes() -> Vec<i32> {
        let raw = read_sysfs(&format!("{NODE_SYSFS_ROOT}/possible"))
            .or_else(|| read_sysfs(&format!("{NODE_SYSFS_ROOT}/online")));

        let nodes = raw.map(|raw| parse_cpuset_list(&raw)).unwrap_or_default();
        if nodes.is_empty() {
            vec![0]
        } else {
            nodes
        }
    }

    /// Cached NUMA node list. Topology is fixed for the lifetime of the
    /// process for every practical purpose, and `is_available` /
    /// `node_of_cpu` sit on the per-allocation hot path, so the sysfs walk
    /// must happen exactly once.
    fn topology() -> &'static [i32] {
        static TOPOLOGY: OnceLock<Vec<i32>> = OnceLock::new();
        TOPOLOGY.get_or_init(discover_nodes)
    }

    /// Build the CPU-id -> node-id table by scanning every node's `cpulist`.
    ///
    /// One pass over the (few) nodes answers every CPU, which is far cheaper
    /// than the per-CPU `readdir` probe used as a fallback below.
    fn build_cpu_node_map() -> Vec<i32> {
        let mut map: Vec<i32> = Vec::new();

        for &node in topology() {
            let path = format!("{NODE_SYSFS_ROOT}/node{node}/cpulist");
            let Some(raw) = read_sysfs(&path) else {
                continue;
            };
            for cpu in parse_cpuset_list(&raw) {
                let idx = cpu as usize;
                if idx >= map.len() {
                    map.resize(idx + 1, 0);
                }
                map[idx] = node;
            }
        }

        map
    }

    /// Cached CPU-id -> node-id table.
    fn cpu_node_map() -> &'static [i32] {
        static CPU_NODE_MAP: OnceLock<Vec<i32>> = OnceLock::new();
        CPU_NODE_MAP.get_or_init(build_cpu_node_map)
    }

    /// Probe `/sys/devices/system/cpu/cpu{cpu}/` for the `node{N}` symlink the
    /// kernel places there, returning `N`.
    ///
    /// This is the authoritative per-CPU mapping and is used whenever the
    /// cached `cpulist` table cannot answer (unreadable `cpulist`, or a CPU
    /// hot-plugged after the table was built).
    fn probe_node_of_cpu(cpu: i32) -> Option<i32> {
        let dir = std::fs::read_dir(format!("{CPU_SYSFS_ROOT}/cpu{cpu}")).ok()?;
        for entry in dir.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("node") {
                if let Ok(node) = rest.parse::<i32>() {
                    return Some(node);
                }
            }
        }
        None
    }

    /// Whether this machine exposes a NUMA topology at all.
    ///
    /// Mirrors `numa_available()`: node 0's sysfs directory exists exactly when
    /// the kernel was built with NUMA support and enumerated the topology.
    pub fn is_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE
            .get_or_init(|| std::path::Path::new(&format!("{NODE_SYSFS_ROOT}/node0")).exists())
    }

    /// The set of NUMA nodes on this machine, ascending.
    ///
    /// Unlike `0..=max_node()` this is exact for sparse topologies such as
    /// `0-1,4`, where nodes 2 and 3 do not exist.
    pub fn nodes() -> &'static [i32] {
        topology()
    }

    /// Highest NUMA node id on this machine (0 when NUMA is unavailable).
    pub fn max_node() -> i32 {
        topology().last().copied().unwrap_or(0)
    }

    /// NUMA node owning `cpu`, or 0 when it cannot be determined.
    pub fn node_of_cpu(cpu: i32) -> i32 {
        if cpu < 0 {
            return 0;
        }
        if let Some(node) = cpu_node_map().get(cpu as usize) {
            return *node;
        }
        probe_node_of_cpu(cpu).unwrap_or(0)
    }

    /// Build the `(nodemask, maxnode)` pair `mbind(2)` expects for `nodes`.
    ///
    /// The mask is sized from the machine's node count (as libnuma does) plus
    /// one spare word, because the kernel's `get_nodes()` decrements `maxnode`
    /// before deriving the word count; the spare word makes the highest node id
    /// addressable regardless of where it falls relative to a word boundary.
    /// Returns `None` when no valid node bit would be set, which `mbind` would
    /// reject with `EINVAL` anyway.
    pub fn nodemask_from_nodes(nodes: &[i32], max_node: i32) -> Option<(Vec<c_ulong>, c_ulong)> {
        let highest = nodes
            .iter()
            .copied()
            .chain(std::iter::once(max_node))
            .max()
            .unwrap_or(0)
            .clamp(0, MAX_SUPPORTED_ID);

        let words = highest as usize / NODEMASK_BITS + 2;
        let mut mask = vec![0 as c_ulong; words];
        let mut any = false;

        for &node in nodes {
            if node < 0 || node > highest {
                continue;
            }
            let idx = node as usize / NODEMASK_BITS;
            let bit = node as usize % NODEMASK_BITS;
            mask[idx] |= (1 as c_ulong) << bit;
            any = true;
        }

        if !any {
            return None;
        }

        Some((mask, (words * NODEMASK_BITS) as c_ulong))
    }

    /// Apply an `mbind(2)` memory policy to `[addr, addr + len)`.
    ///
    /// `flags` is left at 0 so that only *future* faults in the range are
    /// steered; already-resident pages are deliberately not migrated, which is
    /// what makes this cheap enough to run on an allocation path.
    ///
    /// # Safety
    ///
    /// `addr` must be page-aligned and `[addr, addr + len)` must lie entirely
    /// within a live mapping owned by the calling process. The call does not
    /// read or write the range; it only changes the kernel's NUMA policy for
    /// it.
    pub unsafe fn mbind(
        addr: *mut c_void,
        len: usize,
        mode: c_int,
        nodes: &[i32],
    ) -> std::io::Result<()> {
        let Some((mask, maxnode)) = nodemask_from_nodes(nodes, max_node()) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "empty NUMA node mask",
            ));
        };

        // SAFETY: `SYS_mbind` takes (addr, len, mode, nodemask, maxnode,
        // flags). `mask` outlives the call and holds `maxnode / NODEMASK_BITS`
        // words, matching what the kernel reads. The address range is valid by
        // this function's own safety contract.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                addr,
                len as c_ulong,
                mode,
                mask.as_ptr(),
                maxnode,
                0 as c_uint,
            )
        };

        if rc == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

/// Non-Linux stub with the same surface as the Linux module above. NUMA memory
/// policies are a Linux concept; everywhere else the machine is reported as a
/// single node and `mbind` is unsupported.
#[cfg(not(target_os = "linux"))]
mod numa {
    use std::ffi::c_void;

    /// `mbind(2)` policy placeholder (Linux `MPOL_BIND`).
    pub const MPOL_BIND: i32 = 2;

    /// `mbind(2)` policy placeholder (Linux `MPOL_INTERLEAVE`).
    pub const MPOL_INTERLEAVE: i32 = 3;

    pub fn is_available() -> bool {
        false
    }

    pub fn nodes() -> &'static [i32] {
        &[0]
    }

    pub fn max_node() -> i32 {
        0
    }

    pub fn node_of_cpu(_cpu: i32) -> i32 {
        0
    }

    /// # Safety
    ///
    /// Always a no-op returning `Unsupported`; retained so that the
    /// cross-platform allocation path compiles unchanged.
    pub unsafe fn mbind(
        _addr: *mut c_void,
        _len: usize,
        _mode: i32,
        _nodes: &[i32],
    ) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "mbind is only available on Linux",
        ))
    }
}

/// System page size, queried once.
fn page_size() -> usize {
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        #[cfg(unix)]
        {
            // SAFETY: `sysconf` is thread-safe, takes no pointers and has no
            // preconditions beyond a valid name constant.
            let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if value > 0 {
                return value as usize;
            }
        }
        // Conservative default for platforms without `sysconf` (or if it
        // fails): every target we support uses at least 4 KiB pages.
        4096
    })
}

/// Largest page-aligned subrange `[start, end)` contained in the buffer that
/// begins at `start_addr` and spans `len` bytes.
///
/// `mbind(2)` requires a page-aligned start and operates on whole pages, so a
/// buffer must be trimmed to its page-aligned interior before it can be bound.
/// Returns `None` when the buffer contains no complete page — the common case
/// for small allocations served from a malloc arena.
fn page_aligned_subrange(
    start_addr: usize,
    len: usize,
    page_size: usize,
) -> Option<(usize, usize)> {
    if len == 0 || page_size == 0 || !page_size.is_power_of_two() {
        return None;
    }

    let end_addr = start_addr.checked_add(len)?;
    // Round the start up and the end down to page boundaries.
    let aligned_start = start_addr.checked_add(page_size - 1)? & !(page_size - 1);
    let aligned_end = end_addr & !(page_size - 1);

    if aligned_end > aligned_start {
        Some((aligned_start, aligned_end - aligned_start))
    } else {
        None
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
            numa::nodes().to_vec()
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
            numa::nodes().to_vec()
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
        if !numa::is_available() || size == 0 {
            return vec![0u8; size];
        }

        // Reserve the capacity *without* touching it: an untouched capacity has
        // no physical pages yet, so the memory policy installed below decides
        // where the first-touch faults land.
        let mut buffer: Vec<u8> = Vec::with_capacity(size);
        self.apply_memory_policy(buffer.as_mut_ptr().cast(), size, node);
        buffer.resize(size, 0u8);
        buffer
    }

    /// Allocate optimized vector with NUMA awareness (specialized for f32 vectors)
    pub fn allocate_vector_on_node(&self, dimensions: usize, node: Option<i32>) -> Vec<f32> {
        let mut vec: Vec<f32> = Vec::with_capacity(dimensions);

        if numa::is_available() && dimensions > 0 {
            if let Some(byte_len) = dimensions.checked_mul(std::mem::size_of::<f32>()) {
                // Fall back to the current CPU's node when the caller has no
                // usable preference, so a vector is faulted in next to the
                // thread that is about to read it.
                let target = self
                    .explicit_node(node)
                    .or_else(|| Some(self.preferred_node()));
                self.apply_memory_policy(vec.as_mut_ptr().cast(), byte_len, target);
            }
        }

        vec.resize(dimensions, 0.0f32);
        vec
    }

    /// Install a NUMA memory policy over the page-aligned interior of a freshly
    /// reserved, not-yet-touched buffer.
    ///
    /// Best effort by design: `mbind` legitimately fails for small buffers that
    /// live inside a malloc arena (no whole page to bind) or when the policy is
    /// restricted by cgroups, and the correct response is simply to let the
    /// allocator place the pages itself.
    fn apply_memory_policy(&self, ptr: *mut std::ffi::c_void, byte_len: usize, node: Option<i32>) {
        let page = page_size();
        let Some((addr, len)) = page_aligned_subrange(ptr as usize, byte_len, page) else {
            // Buffer smaller than a page, or not spanning a whole page: nothing
            // the kernel can bind. Extremely common and not worth reporting.
            return;
        };

        // `rr` backs the single-node mask; it is only read on the branches that
        // initialise it.
        let rr;
        let (mode, target_nodes): (i32, &[i32]) = match self.explicit_node(node) {
            Some(explicit) => {
                rr = [explicit];
                (numa::MPOL_BIND, &rr)
            }
            // No caller preference and more than one page to place across more
            // than one node: interleave, so a large buffer draws on the memory
            // bandwidth of every node instead of saturating one.
            None if self.numa_nodes.len() > 1 && len > page => {
                (numa::MPOL_INTERLEAVE, self.numa_nodes.as_slice())
            }
            None => {
                rr = [self.next_round_robin_node()];
                (numa::MPOL_BIND, &rr)
            }
        };

        // SAFETY: `addr`/`len` are the page-aligned interior of a live buffer
        // owned by this process (the caller's `Vec` allocation), so the range
        // lies inside a single valid mapping. `mbind` neither reads nor writes
        // the range, it only records a policy for future faults.
        if let Err(err) =
            unsafe { numa::mbind(addr as *mut std::ffi::c_void, len, mode, target_nodes) }
        {
            trace!(
                "mbind({} bytes, mode {}) failed, falling back to default placement: {}",
                len,
                mode,
                err
            );
        }
    }

    /// Validate a caller-supplied node hint against the real topology.
    ///
    /// Hints travel through `PageCacheEntry::numa_node`, so a stale or bogus id
    /// can reach us; screening it here avoids handing the kernel a mask it
    /// would reject with `EINVAL`.
    fn explicit_node(&self, node: Option<i32>) -> Option<i32> {
        let node = node?;
        if node >= 0 && node <= numa::max_node() && self.numa_nodes.contains(&node) {
            Some(node)
        } else {
            None
        }
    }

    /// Next node in the round-robin rotation used when no hint is available.
    fn next_round_robin_node(&self) -> i32 {
        if self.numa_nodes.is_empty() {
            return 0;
        }
        let idx = self.current_node.fetch_add(1, Ordering::Relaxed) % self.numa_nodes.len();
        self.numa_nodes[idx]
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

    // ---------------------------------------------------------------------
    // NUMA topology / memory-policy tests (Pure Rust replacement for libnuma)
    // ---------------------------------------------------------------------

    #[test]
    fn test_numa_topology_is_sane() {
        let nodes = numa::nodes();
        assert!(!nodes.is_empty(), "node list must never be empty");
        assert!(
            nodes.iter().all(|&n| n >= 0),
            "node ids must be non-negative, got {nodes:?}"
        );
        assert!(
            nodes.windows(2).all(|w| w[0] < w[1]),
            "node list must be sorted and de-duplicated, got {nodes:?}"
        );
        assert_eq!(
            numa::max_node(),
            nodes.iter().copied().max().unwrap_or(0),
            "max_node must agree with the node list"
        );

        // The allocator derives its own view of the topology; it must match.
        let allocator = NumaVectorAllocator::new();
        assert!(!allocator.numa_nodes.is_empty());
        assert!(allocator.numa_nodes.iter().all(|&n| n >= 0));
    }

    #[test]
    fn test_node_of_cpu_within_topology() {
        let node = numa::node_of_cpu(0);
        assert!(
            (0..=numa::max_node()).contains(&node),
            "node_of_cpu(0) = {node} must lie in 0..={}",
            numa::max_node()
        );

        // Negative and absurd CPU ids must degrade to node 0, not panic.
        assert_eq!(numa::node_of_cpu(-1), 0);
        assert_eq!(numa::node_of_cpu(i32::MAX), 0);

        // The allocator's preferred node must be a real node too.
        let allocator = NumaVectorAllocator::new();
        let preferred = allocator.preferred_node();
        assert!((0..=numa::max_node()).contains(&preferred));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpuset_list() {
        use super::numa::parse_cpuset_list;

        assert_eq!(parse_cpuset_list("0"), vec![0]);
        assert_eq!(parse_cpuset_list("0-1"), vec![0, 1]);
        assert_eq!(parse_cpuset_list("0-1,4"), vec![0, 1, 4]);
        assert_eq!(parse_cpuset_list("0-3\n"), vec![0, 1, 2, 3]);
        // Out-of-order and overlapping entries are normalised.
        assert_eq!(parse_cpuset_list("4,0-1,1"), vec![0, 1, 4]);
        // Stride notation (`N-M:S/T`) degrades to the plain range.
        assert_eq!(parse_cpuset_list("0-2:1/2"), vec![0, 1, 2]);

        // Empty / whitespace-only input.
        assert!(parse_cpuset_list("").is_empty());
        assert!(parse_cpuset_list("   \n").is_empty());
        assert!(parse_cpuset_list(",,").is_empty());

        // Garbage must be skipped, never panic.
        assert!(parse_cpuset_list("abc").is_empty());
        assert!(parse_cpuset_list("-").is_empty());
        assert!(parse_cpuset_list("3-1").is_empty(), "inverted range");
        assert!(parse_cpuset_list("-5").is_empty(), "negative id");
        assert!(parse_cpuset_list("99999999999999999999").is_empty());
        // Mixed valid + garbage keeps the valid part.
        assert_eq!(parse_cpuset_list("0,bogus,2"), vec![0, 2]);
        // Absurd upper bounds are clamped instead of allocating unboundedly.
        assert!(parse_cpuset_list("0-4294967295").is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_nodemask_from_nodes() {
        use super::numa::nodemask_from_nodes;

        let (mask, maxnode) = nodemask_from_nodes(&[0], 0).expect("node 0 mask");
        assert_eq!(mask[0] & 1, 1, "bit 0 must be set");
        assert!(
            maxnode as usize >= 64,
            "mask must be at least one word wide"
        );
        assert_eq!(
            maxnode as usize,
            mask.len() * std::mem::size_of::<libc::c_ulong>() * 8,
            "maxnode must describe the whole mask in bits"
        );

        let (mask, _) = nodemask_from_nodes(&[1, 3], 3).expect("sparse mask");
        assert_eq!(mask[0] & 0b1111, 0b1010);

        // A node id beyond the last word must still be addressable.
        let (mask, maxnode) = nodemask_from_nodes(&[65], 65).expect("wide mask");
        assert!(mask.len() >= 2);
        assert_eq!(mask[1] & 0b10, 0b10);
        assert!((maxnode as usize) > 65);

        // Nothing to bind -> None rather than an EINVAL syscall.
        assert!(nodemask_from_nodes(&[], 0).is_none());
        assert!(nodemask_from_nodes(&[-1], 0).is_none());
    }

    #[test]
    fn test_page_aligned_subrange() {
        let page = 4096usize;

        // Already aligned, whole number of pages.
        assert_eq!(
            page_aligned_subrange(page, 2 * page, page),
            Some((page, 2 * page))
        );

        // Unaligned start: round the start up, the end down.
        assert_eq!(
            page_aligned_subrange(page + 100, 3 * page, page),
            Some((2 * page, 2 * page))
        );

        // Aligned start, ragged end: trim the tail.
        assert_eq!(
            page_aligned_subrange(page, page + 7, page),
            Some((page, page))
        );

        // Buffer smaller than a page -> no bindable range.
        assert_eq!(page_aligned_subrange(page + 1, 16, page), None);
        assert_eq!(page_aligned_subrange(page, page - 1, page), None);

        // Exactly one page but straddling a boundary -> no whole page inside.
        assert_eq!(page_aligned_subrange(page + 1, page, page), None);

        // Degenerate inputs must not panic.
        assert_eq!(page_aligned_subrange(0, 0, page), None);
        assert_eq!(page_aligned_subrange(page, page, 0), None);
        assert_eq!(
            page_aligned_subrange(page, page, 4095),
            None,
            "not a power of two"
        );
        assert_eq!(page_aligned_subrange(usize::MAX, 4, page), None, "overflow");

        // The real page size must be usable with the same helper.
        let real = page_size();
        assert!(real.is_power_of_two() && real >= 4096);
        assert!(page_aligned_subrange(real, 4 * real, real).is_some());
    }

    #[test]
    fn test_numa_allocation_shapes_and_contents() {
        let allocator = NumaVectorAllocator::new();

        // Byte allocation: exact length, zero-initialised, node hint honoured
        // without changing the observable result.
        for node in [None, Some(0), Some(numa::max_node()), Some(-7), Some(9999)] {
            // Large enough to contain whole pages, exercising the mbind path.
            let buf = allocator.allocate_on_node(3 * page_size(), node);
            assert_eq!(buf.len(), 3 * page_size());
            assert!(buf.iter().all(|&b| b == 0));
        }
        assert!(allocator.allocate_on_node(0, None).is_empty());

        // f32 allocation: exact dimensions, zero-initialised.
        for node in [None, Some(0), Some(-1)] {
            let vec = allocator.allocate_vector_on_node(2048, node);
            assert_eq!(vec.len(), 2048);
            assert!(vec.iter().all(|&v| v == 0.0));
        }
        assert!(allocator.allocate_vector_on_node(0, None).is_empty());
    }

    #[test]
    fn test_explicit_node_validation_and_round_robin() {
        let allocator = NumaVectorAllocator::new();
        let valid = allocator.numa_nodes[0];

        assert_eq!(allocator.explicit_node(Some(valid)), Some(valid));
        assert_eq!(allocator.explicit_node(None), None);
        assert_eq!(allocator.explicit_node(Some(-1)), None);
        assert_eq!(
            allocator.explicit_node(Some(numa::max_node() + 1)),
            None,
            "out-of-range hints must be rejected, not passed to the kernel"
        );

        // Round-robin must always yield a node that exists.
        for _ in 0..(allocator.numa_nodes.len() * 3 + 1) {
            let node = allocator.next_round_robin_node();
            assert!(allocator.numa_nodes.contains(&node));
        }
    }
}
