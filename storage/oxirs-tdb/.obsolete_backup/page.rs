//! # Page Management System for TDB Storage
//!
//! Implements a sophisticated page management system with buffer pool,
//! LRU replacement, dirty page tracking, and efficient I/O operations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;
use tracing::info;

#[cfg(target_os = "linux")]
use libc::{cpu_set_t, sched_getaffinity, sched_setaffinity, CPU_ISSET, CPU_SET, CPU_ZERO};
#[cfg(target_os = "linux")]
use std::ffi::CString;

/// Standard page size for TDB (8KB)
pub const PAGE_SIZE: usize = 8192;

/// Page identifier type
pub type PageId = u64;

/// Invalid page ID constant
pub const INVALID_PAGE_ID: PageId = u64::MAX;

/// Page header structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHeader {
    /// Page ID
    pub page_id: PageId,
    /// Page type
    pub page_type: PageType,
    /// Number of records in the page
    pub record_count: u32,
    /// Free space in bytes
    pub free_space: u32,
    /// Next page in chain (for overflow pages)
    pub next_page: PageId,
    /// Previous page in chain
    pub prev_page: PageId,
    /// Checksum for integrity
    pub checksum: u64,
    /// Last modification timestamp
    pub modified: u64,
    /// Reserved for future use
    pub reserved: [u8; 16],
}

impl PageHeader {
    /// Header size in bytes
    pub const SIZE: usize = 72;

    /// Create a new page header
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        Self {
            page_id,
            page_type,
            record_count: 0,
            free_space: (PAGE_SIZE - Self::SIZE) as u32,
            next_page: INVALID_PAGE_ID,
            prev_page: INVALID_PAGE_ID,
            checksum: 0,
            modified: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            reserved: [0; 16],
        }
    }

    /// Calculate checksum for the page
    pub fn calculate_checksum(&self, data: &[u8]) -> u64 {
        // Simple checksum implementation - in production, use CRC32 or similar
        let mut checksum = 0u64;
        for byte in data {
            checksum = checksum.wrapping_add(*byte as u64);
        }
        checksum
    }

    /// Serialize header to bytes (fixed size)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(Self::SIZE);

        // Fixed-size serialization to ensure consistent header size
        bytes.extend_from_slice(&self.page_id.to_le_bytes()); // 8 bytes
        bytes.push(self.page_type as u8); // 1 byte
        bytes.extend_from_slice(&self.record_count.to_le_bytes()); // 4 bytes
        bytes.extend_from_slice(&self.free_space.to_le_bytes()); // 4 bytes
        bytes.extend_from_slice(&self.next_page.to_le_bytes()); // 8 bytes
        bytes.extend_from_slice(&self.prev_page.to_le_bytes()); // 8 bytes
        bytes.extend_from_slice(&self.checksum.to_le_bytes()); // 8 bytes
        bytes.extend_from_slice(&self.modified.to_le_bytes()); // 8 bytes
        bytes.extend_from_slice(&self.reserved); // 16 bytes

        // Total: 8+1+4+4+8+8+8+8+16 = 65 bytes, pad to 72 for alignment
        bytes.resize(Self::SIZE, 0);
        Ok(bytes)
    }

    /// Deserialize header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(anyhow!(
                "Insufficient data for page header: {} < {}",
                data.len(),
                Self::SIZE
            ));
        }

        let mut offset = 0;

        let page_id = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let page_type = match data[offset] {
            1 => PageType::Data,
            2 => PageType::Index,
            3 => PageType::FreeSpace,
            4 => PageType::Metadata,
            5 => PageType::Overflow,
            _ => return Err(anyhow!("Invalid page type: {}", data[offset])),
        };
        offset += 1;

        let record_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        let free_space = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        let next_page = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let prev_page = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let checksum = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let modified = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let mut reserved = [0u8; 16];
        reserved.copy_from_slice(&data[offset..offset + 16]);

        Ok(Self {
            page_id,
            page_type,
            record_count,
            free_space,
            next_page,
            prev_page,
            checksum,
            modified,
            reserved,
        })
    }
}

/// Page type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageType {
    /// Data page containing records
    Data = 1,
    /// Index page (B+ tree node)
    Index = 2,
    /// Free space management page
    FreeSpace = 3,
    /// Metadata page
    Metadata = 4,
    /// Overflow page for large records
    Overflow = 5,
    /// Transaction log page
    Transaction = 6,
}

/// Page structure
#[derive(Debug, Clone)]
pub struct Page {
    /// Page header
    pub header: PageHeader,
    /// Page data (without header)
    pub data: Vec<u8>,
    /// Dirty flag
    pub is_dirty: bool,
    /// Last access time
    pub last_access: SystemTime,
    /// Reference count
    pub ref_count: u32,
}

impl Page {
    /// Create a new page
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        let header = PageHeader::new(page_id, page_type);
        let data = vec![0; PAGE_SIZE - PageHeader::SIZE];

        Self {
            header,
            data,
            is_dirty: true,
            last_access: SystemTime::now(),
            ref_count: 0,
        }
    }

    /// Create a page from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() != PAGE_SIZE {
            return Err(anyhow!(
                "Invalid page size: expected {}, got {}",
                PAGE_SIZE,
                data.len()
            ));
        }

        let header = PageHeader::from_bytes(data)?;
        let page_data = data[PageHeader::SIZE..].to_vec();

        Ok(Self {
            header,
            data: page_data,
            is_dirty: false,
            last_access: SystemTime::now(),
            ref_count: 0,
        })
    }

    /// Convert page to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(PAGE_SIZE);
        result.extend_from_slice(&self.header.to_bytes()?);
        result.extend_from_slice(&self.data);

        // Ensure exact page size
        result.resize(PAGE_SIZE, 0);
        Ok(result)
    }

    /// Mark page as dirty
    pub fn mark_dirty(&mut self) {
        self.is_dirty = true;
        self.header.modified = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Update access time
    pub fn touch(&mut self) {
        self.last_access = SystemTime::now();
    }

    /// Get available space in the page
    pub fn available_space(&self) -> usize {
        self.header.free_space as usize
    }

    /// Validate page integrity
    pub fn validate(&self) -> Result<bool> {
        let calculated_checksum = self.header.calculate_checksum(&self.data);
        Ok(calculated_checksum == self.header.checksum)
    }
}

/// NUMA topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub numa_nodes: usize,
    /// CPU cores per NUMA node
    pub cores_per_node: Vec<Vec<usize>>,
    /// Memory per NUMA node in bytes
    pub memory_per_node: Vec<usize>,
    /// Current NUMA node
    pub current_node: usize,
    /// NUMA distance matrix
    pub distance_matrix: Vec<Vec<u8>>,
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self {
            numa_nodes: 1,
            cores_per_node: vec![vec![0]],
            memory_per_node: vec![0],
            current_node: 0,
            distance_matrix: vec![vec![10]], // Self distance is typically 10
        }
    }
}

/// NUMA allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NumaAllocationStrategy {
    /// Allocate on current NUMA node
    #[default]
    Local,
    /// Interleave allocation across all NUMA nodes
    Interleave,
    /// Round-robin allocation across NUMA nodes
    RoundRobin,
    /// Allocate on preferred NUMA node
    Preferred(usize),
    /// Bind allocation to specific NUMA node
    Bind(usize),
}

/// NUMA memory pool configuration
#[derive(Debug, Clone)]
pub struct NumaMemoryPool {
    /// NUMA node ID
    pub node_id: usize,
    /// Pool size in bytes
    pub pool_size: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Allocated pages count
    pub allocated_pages: usize,
    /// Maximum pages per pool
    pub max_pages: usize,
    /// Page allocation bitmap
    pub page_bitmap: Vec<bool>,
}

impl NumaMemoryPool {
    pub fn new(node_id: usize, pool_size: usize, max_pages: usize) -> Self {
        Self {
            node_id,
            pool_size,
            available_memory: pool_size,
            allocated_pages: 0,
            max_pages,
            page_bitmap: vec![false; max_pages],
        }
    }

    pub fn can_allocate(&self) -> bool {
        self.allocated_pages < self.max_pages && self.available_memory >= PAGE_SIZE
    }

    pub fn allocate_page(&mut self) -> Option<usize> {
        if !self.can_allocate() {
            return None;
        }

        for (i, &allocated) in self.page_bitmap.iter().enumerate() {
            if !allocated {
                self.page_bitmap[i] = true;
                self.allocated_pages += 1;
                self.available_memory = self.available_memory.saturating_sub(PAGE_SIZE);
                return Some(i);
            }
        }
        None
    }

    pub fn deallocate_page(&mut self, page_index: usize) {
        if page_index < self.page_bitmap.len() && self.page_bitmap[page_index] {
            self.page_bitmap[page_index] = false;
            self.allocated_pages = self.allocated_pages.saturating_sub(1);
            self.available_memory += PAGE_SIZE;
        }
    }

    pub fn utilization(&self) -> f64 {
        if self.max_pages == 0 {
            0.0
        } else {
            self.allocated_pages as f64 / self.max_pages as f64
        }
    }
}

/// Buffer pool configuration
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Maximum number of pages in buffer pool
    pub max_pages: usize,
    /// Enable LRU eviction
    pub enable_lru: bool,
    /// Write-behind delay in milliseconds
    pub write_behind_delay_ms: u64,
    /// Sync frequency in seconds
    pub sync_frequency_seconds: u64,
    /// Enable page compression
    pub enable_compression: bool,
    /// Enable NUMA-aware allocation
    pub enable_numa: bool,
    /// NUMA allocation strategy
    pub numa_strategy: NumaAllocationStrategy,
    /// NUMA topology detection
    pub auto_detect_numa: bool,
    /// Memory pools per NUMA node
    pub memory_pools_per_node: usize,
    /// NUMA memory binding
    pub numa_memory_binding: bool,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_pages: 1024, // 8MB buffer pool
            enable_lru: true,
            write_behind_delay_ms: 5000, // 5 seconds
            sync_frequency_seconds: 30,
            enable_compression: false,
            enable_numa: true,
            numa_strategy: NumaAllocationStrategy::default(),
            auto_detect_numa: true,
            memory_pools_per_node: 4,
            numa_memory_binding: true,
        }
    }
}

/// Buffer pool statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct BufferPoolStats {
    /// Total page requests
    pub total_requests: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Pages read from disk
    pub disk_reads: u64,
    /// Pages written to disk
    pub disk_writes: u64,
    /// Current pages in buffer
    pub current_pages: usize,
    /// Dirty pages count
    pub dirty_pages: usize,
    /// Evicted pages count
    pub evicted_pages: u64,
    /// Average access time
    pub avg_access_time_ms: f64,
    /// NUMA allocation statistics
    pub numa_allocations_per_node: HashMap<usize, u64>,
    /// NUMA memory utilization per node
    pub numa_memory_utilization: HashMap<usize, f64>,
    /// NUMA page migrations
    pub numa_page_migrations: u64,
    /// NUMA allocation failures
    pub numa_allocation_failures: u64,
}

impl BufferPoolStats {
    /// Calculate cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }
}

/// LRU node for buffer pool
#[derive(Debug)]
struct LruNode {
    #[allow(dead_code)]
    page_id: PageId,
    prev: Option<PageId>,
    next: Option<PageId>,
}

/// Buffer pool implementation
pub struct BufferPool {
    /// Configuration
    config: BufferPoolConfig,
    /// Cached pages
    pages: Arc<RwLock<HashMap<PageId, Arc<Mutex<Page>>>>>,
    /// LRU list head
    lru_head: Arc<Mutex<Option<PageId>>>,
    /// LRU list tail
    lru_tail: Arc<Mutex<Option<PageId>>>,
    /// LRU nodes
    lru_nodes: Arc<RwLock<HashMap<PageId, LruNode>>>,
    /// Dirty pages queue
    dirty_pages: Arc<Mutex<VecDeque<PageId>>>,
    /// Statistics
    stats: Arc<Mutex<BufferPoolStats>>,
    /// Page file
    page_file: Arc<Mutex<File>>,
    /// File path
    #[allow(dead_code)]
    file_path: PathBuf,
    /// Next page ID counter
    next_page_id: Arc<Mutex<PageId>>,
    /// NUMA topology
    #[allow(dead_code)]
    numa_topology: Arc<RwLock<NumaTopology>>,
    /// NUMA memory pools
    #[allow(dead_code)]
    numa_memory_pools: Arc<RwLock<HashMap<usize, NumaMemoryPool>>>,
    /// Page to NUMA node mapping
    #[allow(dead_code)]
    page_numa_mapping: Arc<RwLock<HashMap<PageId, usize>>>,
    /// Current NUMA allocation node (for round-robin)
    #[allow(dead_code)]
    current_numa_node: Arc<Mutex<usize>>,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        Self::with_config(file_path, BufferPoolConfig::default())
    }

    /// Create a new buffer pool with configuration
    pub fn with_config<P: AsRef<Path>>(file_path: P, config: BufferPoolConfig) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        // Create or open the page file
        let page_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&file_path)
            .map_err(|e| anyhow!("Failed to open page file: {}", e))?;

        let numa_topology = if config.enable_numa && config.auto_detect_numa {
            Self::detect_numa_topology()
        } else {
            NumaTopology::default()
        };

        let numa_memory_pools = Self::initialize_numa_memory_pools(&numa_topology, &config);

        Ok(Self {
            config,
            pages: Arc::new(RwLock::new(HashMap::new())),
            lru_head: Arc::new(Mutex::new(None)),
            lru_tail: Arc::new(Mutex::new(None)),
            lru_nodes: Arc::new(RwLock::new(HashMap::new())),
            dirty_pages: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(BufferPoolStats::default())),
            page_file: Arc::new(Mutex::new(page_file)),
            file_path,
            next_page_id: Arc::new(Mutex::new(0)),
            numa_topology: Arc::new(RwLock::new(numa_topology)),
            numa_memory_pools: Arc::new(RwLock::new(numa_memory_pools)),
            page_numa_mapping: Arc::new(RwLock::new(HashMap::new())),
            current_numa_node: Arc::new(Mutex::new(0)),
        })
    }

    /// Get a page from the buffer pool
    pub fn get_page(&self, page_id: PageId) -> Result<Arc<Mutex<Page>>> {
        self.update_stats_request();

        // Check if page is already in buffer
        if let Some(page) = self.get_cached_page(page_id) {
            self.update_lru(page_id)?;
            self.update_stats_hit();
            return Ok(page);
        }

        // Page not in buffer, need to load from disk
        self.update_stats_miss();
        let page = self.load_page_from_disk(page_id)?;

        // Add to buffer pool
        self.add_to_buffer(page_id, page.clone())?;

        Ok(page)
    }

    /// Create a new page
    pub fn create_page(&self, page_type: PageType) -> Result<(PageId, Arc<Mutex<Page>>)> {
        let page_id = self.allocate_page_id()?;
        let page = Arc::new(Mutex::new(Page::new(page_id, page_type)));

        self.add_to_buffer(page_id, page.clone())?;
        self.mark_page_dirty(page_id)?;

        Ok((page_id, page))
    }

    /// Flush a specific page to disk
    pub fn flush_page(&self, page_id: PageId) -> Result<()> {
        if let Some(page_arc) = self.get_cached_page(page_id) {
            let page = page_arc
                .lock()
                .map_err(|_| anyhow!("Failed to lock page"))?;

            if page.is_dirty {
                self.write_page_to_disk(&page)?;
                self.update_stats_write();

                // Remove from dirty pages queue
                let mut dirty_pages = self
                    .dirty_pages
                    .lock()
                    .map_err(|_| anyhow!("Failed to lock dirty pages"))?;
                dirty_pages.retain(|&id| id != page_id);
            }
        }

        Ok(())
    }

    /// Flush all dirty pages to disk
    pub fn flush_all(&self) -> Result<()> {
        let dirty_page_ids = {
            let dirty_pages = self
                .dirty_pages
                .lock()
                .map_err(|_| anyhow!("Failed to lock dirty pages"))?;
            dirty_pages.iter().copied().collect::<Vec<_>>()
        };

        for page_id in dirty_page_ids {
            self.flush_page(page_id)?;
        }

        // Sync file to disk
        let file = self
            .page_file
            .lock()
            .map_err(|_| anyhow!("Failed to lock page file"))?;
        file.sync_all()
            .map_err(|e| anyhow!("Failed to sync file: {}", e))?;

        Ok(())
    }

    /// Evict a page from the buffer pool
    pub fn evict_page(&self, page_id: PageId) -> Result<()> {
        // Flush if dirty
        self.flush_page(page_id)?;

        // Remove from buffer
        let mut pages = self
            .pages
            .write()
            .map_err(|_| anyhow!("Failed to lock pages"))?;
        pages.remove(&page_id);

        // Remove from LRU
        self.remove_from_lru(page_id)?;

        self.update_stats_eviction();
        Ok(())
    }

    /// Get buffer pool statistics
    pub fn get_stats(&self) -> Result<BufferPoolStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| anyhow!("Failed to lock stats"))?;
        Ok(stats.clone())
    }

    /// Compact the buffer pool by evicting least recently used pages
    pub fn compact(&self) -> Result<usize> {
        let mut evicted = 0;
        let max_pages = self.config.max_pages;

        let current_count = {
            let pages = self
                .pages
                .read()
                .map_err(|_| anyhow!("Failed to lock pages"))?;
            pages.len()
        };

        if current_count > max_pages {
            let to_evict = current_count - max_pages;
            let mut tail = self
                .lru_tail
                .lock()
                .map_err(|_| anyhow!("Failed to lock LRU tail"))?;

            for _ in 0..to_evict {
                if let Some(page_id) = *tail {
                    let next_tail = {
                        let lru_nodes = self
                            .lru_nodes
                            .read()
                            .map_err(|_| anyhow!("Failed to lock LRU nodes"))?;
                        lru_nodes.get(&page_id).and_then(|node| node.prev)
                    };

                    self.evict_page(page_id)?;
                    *tail = next_tail;
                    evicted += 1;
                } else {
                    break;
                }
            }
        }

        info!("Compacted buffer pool, evicted {} pages", evicted);
        Ok(evicted)
    }

    /// Evict the least recently used page
    fn evict_lru(&self) -> Result<()> {
        let tail_page_id = {
            let tail = self
                .lru_tail
                .lock()
                .map_err(|_| anyhow!("Failed to lock LRU tail"))?;
            *tail
        };

        if let Some(page_id) = tail_page_id {
            let next_tail = {
                let lru_nodes = self
                    .lru_nodes
                    .read()
                    .map_err(|_| anyhow!("Failed to lock LRU nodes"))?;
                lru_nodes.get(&page_id).and_then(|node| node.prev)
            };

            self.evict_page(page_id)?;

            let mut tail = self
                .lru_tail
                .lock()
                .map_err(|_| anyhow!("Failed to lock LRU tail"))?;
            *tail = next_tail;
        }

        Ok(())
    }

    // Private helper methods

    fn get_cached_page(&self, page_id: PageId) -> Option<Arc<Mutex<Page>>> {
        let pages = self.pages.read().ok()?;
        pages.get(&page_id).cloned()
    }

    fn load_page_from_disk(&self, page_id: PageId) -> Result<Arc<Mutex<Page>>> {
        let mut file = self
            .page_file
            .lock()
            .map_err(|_| anyhow!("Failed to lock page file"))?;

        let offset = page_id * PAGE_SIZE as u64;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| anyhow!("Failed to seek to page {}: {}", page_id, e))?;

        let mut buffer = vec![0; PAGE_SIZE];
        file.read_exact(&mut buffer)
            .map_err(|e| anyhow!("Failed to read page {}: {}", page_id, e))?;

        let page = Page::from_bytes(&buffer)?;
        self.update_stats_read();

        Ok(Arc::new(Mutex::new(page)))
    }

    fn write_page_to_disk(&self, page: &Page) -> Result<()> {
        let mut file = self
            .page_file
            .lock()
            .map_err(|_| anyhow!("Failed to lock page file"))?;

        let offset = page.header.page_id * PAGE_SIZE as u64;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| anyhow!("Failed to seek to page {}: {}", page.header.page_id, e))?;

        let data = page.to_bytes()?;
        file.write_all(&data)
            .map_err(|e| anyhow!("Failed to write page {}: {}", page.header.page_id, e))?;

        Ok(())
    }

    fn add_to_buffer(&self, page_id: PageId, page: Arc<Mutex<Page>>) -> Result<()> {
        // Check if we need to evict pages first
        if self.config.enable_lru {
            let current_count = {
                let pages = self
                    .pages
                    .read()
                    .map_err(|_| anyhow!("Failed to lock pages"))?;
                pages.len()
            };

            // If adding one more page would exceed capacity, evict LRU pages
            if current_count + 1 > self.config.max_pages {
                let to_evict = (current_count + 1) - self.config.max_pages;
                for _ in 0..to_evict {
                    self.evict_lru()?;
                }
            }
        }

        // Add to pages map
        let mut pages = self
            .pages
            .write()
            .map_err(|_| anyhow!("Failed to lock pages"))?;
        pages.insert(page_id, page);

        // Add to LRU
        self.add_to_lru(page_id)?;

        Ok(())
    }

    fn add_to_lru(&self, page_id: PageId) -> Result<()> {
        let mut lru_nodes = self
            .lru_nodes
            .write()
            .map_err(|_| anyhow!("Failed to lock LRU nodes"))?;

        let mut head = self
            .lru_head
            .lock()
            .map_err(|_| anyhow!("Failed to lock LRU head"))?;

        let node = LruNode {
            page_id,
            prev: None,
            next: *head,
        };

        if let Some(old_head) = *head {
            if let Some(old_head_node) = lru_nodes.get_mut(&old_head) {
                old_head_node.prev = Some(page_id);
            }
        } else {
            // First node, also set as tail
            let mut tail = self
                .lru_tail
                .lock()
                .map_err(|_| anyhow!("Failed to lock LRU tail"))?;
            *tail = Some(page_id);
        }

        lru_nodes.insert(page_id, node);
        *head = Some(page_id);

        Ok(())
    }

    fn update_lru(&self, page_id: PageId) -> Result<()> {
        // Move to head if not already there
        let head = self
            .lru_head
            .lock()
            .map_err(|_| anyhow!("Failed to lock LRU head"))?;

        if *head != Some(page_id) {
            self.remove_from_lru(page_id)?;
            drop(head);
            self.add_to_lru(page_id)?;
        }

        Ok(())
    }

    fn remove_from_lru(&self, page_id: PageId) -> Result<()> {
        let mut lru_nodes = self
            .lru_nodes
            .write()
            .map_err(|_| anyhow!("Failed to lock LRU nodes"))?;

        if let Some(node) = lru_nodes.remove(&page_id) {
            // Update prev node's next pointer
            if let Some(prev_id) = node.prev {
                if let Some(prev_node) = lru_nodes.get_mut(&prev_id) {
                    prev_node.next = node.next;
                }
            } else {
                // This was the head
                let mut head = self
                    .lru_head
                    .lock()
                    .map_err(|_| anyhow!("Failed to lock LRU head"))?;
                *head = node.next;
            }

            // Update next node's prev pointer
            if let Some(next_id) = node.next {
                if let Some(next_node) = lru_nodes.get_mut(&next_id) {
                    next_node.prev = node.prev;
                }
            } else {
                // This was the tail
                let mut tail = self
                    .lru_tail
                    .lock()
                    .map_err(|_| anyhow!("Failed to lock LRU tail"))?;
                *tail = node.prev;
            }
        }

        Ok(())
    }

    fn mark_page_dirty(&self, page_id: PageId) -> Result<()> {
        let mut dirty_pages = self
            .dirty_pages
            .lock()
            .map_err(|_| anyhow!("Failed to lock dirty pages"))?;

        if !dirty_pages.contains(&page_id) {
            dirty_pages.push_back(page_id);
        }

        Ok(())
    }

    fn allocate_page_id(&self) -> Result<PageId> {
        let mut next_id = self
            .next_page_id
            .lock()
            .map_err(|_| anyhow!("Failed to lock next_page_id"))?;

        let id = *next_id;
        *next_id += 1;
        Ok(id)
    }

    fn update_stats_request(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_requests += 1;
        }
    }

    fn update_stats_hit(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_hits += 1;
        }
    }

    fn update_stats_miss(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_misses += 1;
        }
    }

    fn update_stats_read(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.disk_reads += 1;
        }
    }

    fn update_stats_write(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.disk_writes += 1;
        }
    }

    fn update_stats_eviction(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.evicted_pages += 1;
        }
    }

    /// Detect NUMA topology of the current system
    fn detect_numa_topology() -> NumaTopology {
        #[cfg(target_os = "linux")]
        {
            // Try to detect actual NUMA topology on Linux
            use std::fs;

            if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
                let mut numa_nodes = 0;
                let mut cores_per_node = Vec::new();
                let mut memory_per_node = Vec::new();

                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name() {
                        if let Some(name_str) = name.to_str() {
                            if name_str.starts_with("node") {
                                if let Ok(node_id) = name_str[4..].parse::<usize>() {
                                    numa_nodes = numa_nodes.max(node_id + 1);

                                    // Read CPU list for this node
                                    let cpulist_path = path.join("cpulist");
                                    if let Ok(cpulist) = fs::read_to_string(cpulist_path) {
                                        let cores = Self::parse_cpu_list(&cpulist.trim());
                                        while cores_per_node.len() <= node_id {
                                            cores_per_node.push(Vec::new());
                                        }
                                        cores_per_node[node_id] = cores;
                                    }

                                    // Read memory info for this node
                                    let meminfo_path = path.join("meminfo");
                                    if let Ok(meminfo) = fs::read_to_string(meminfo_path) {
                                        let memory = Self::parse_memory_info(&meminfo);
                                        while memory_per_node.len() <= node_id {
                                            memory_per_node.push(0);
                                        }
                                        memory_per_node[node_id] = memory;
                                    }
                                }
                            }
                        }
                    }
                }

                if numa_nodes > 0 {
                    // Create a simple distance matrix (real implementation would read from /sys)
                    let mut distance_matrix = vec![vec![255u8; numa_nodes]; numa_nodes];
                    for i in 0..numa_nodes {
                        for j in 0..numa_nodes {
                            distance_matrix[i][j] = if i == j { 10 } else { 20 };
                        }
                    }

                    return NumaTopology {
                        numa_nodes,
                        cores_per_node,
                        memory_per_node,
                        current_node: 0, // Would detect current node in real implementation
                        distance_matrix,
                    };
                }
            }
        }

        // Fallback to default topology for non-Linux or detection failure
        NumaTopology::default()
    }

    /// Parse CPU list string (e.g., "0-3,8-11")
    #[allow(dead_code)]
    fn parse_cpu_list(cpulist: &str) -> Vec<usize> {
        let mut cores = Vec::new();
        for range in cpulist.split(',') {
            if range.contains('-') {
                let parts: Vec<&str> = range.split('-').collect();
                if parts.len() == 2 {
                    if let (Ok(start), Ok(end)) =
                        (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                    {
                        for cpu in start..=end {
                            cores.push(cpu);
                        }
                    }
                }
            } else if let Ok(cpu) = range.parse::<usize>() {
                cores.push(cpu);
            }
        }
        cores
    }

    /// Parse memory info from NUMA node meminfo
    #[allow(dead_code)]
    fn parse_memory_info(meminfo: &str) -> usize {
        for line in meminfo.lines() {
            if line.starts_with("Node") && line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(kb) = parts[3].parse::<usize>() {
                        return kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }
        0
    }

    /// Initialize NUMA memory pools based on topology
    fn initialize_numa_memory_pools(
        numa_topology: &NumaTopology,
        config: &BufferPoolConfig,
    ) -> HashMap<usize, NumaMemoryPool> {
        let mut pools = HashMap::new();

        for node_id in 0..numa_topology.numa_nodes {
            let pool_size = if node_id < numa_topology.memory_per_node.len() {
                numa_topology.memory_per_node[node_id]
            } else {
                1024 * 1024 * 1024 // 1GB default
            };

            let max_pages = config.max_pages / numa_topology.numa_nodes.max(1);
            let pool = NumaMemoryPool::new(node_id, pool_size, max_pages);
            pools.insert(node_id, pool);
        }

        // If no NUMA nodes detected, create a default pool
        if pools.is_empty() {
            let pool = NumaMemoryPool::new(0, 1024 * 1024 * 1024, config.max_pages);
            pools.insert(0, pool);
        }

        pools
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_page_creation() {
        let page = Page::new(1, PageType::Data);
        assert_eq!(page.header.page_id, 1);
        assert_eq!(page.header.page_type, PageType::Data);
        assert!(page.is_dirty);
        assert_eq!(page.data.len(), PAGE_SIZE - PageHeader::SIZE);
    }

    #[test]
    fn test_page_serialization() {
        let page = Page::new(42, PageType::Index);
        let bytes = page.to_bytes().unwrap();
        assert_eq!(bytes.len(), PAGE_SIZE);

        let deserialized = Page::from_bytes(&bytes).unwrap();
        assert_eq!(deserialized.header.page_id, 42);
        assert_eq!(deserialized.header.page_type, PageType::Index);
    }

    #[test]
    fn test_buffer_pool_basic() {
        let temp_file = NamedTempFile::new().unwrap();
        let buffer_pool = BufferPool::new(temp_file.path()).unwrap();

        // Create a new page
        let (page_id, page) = buffer_pool.create_page(PageType::Data).unwrap();
        assert_eq!(page.lock().unwrap().header.page_id, page_id);

        // Get the same page (should be cached)
        let page2 = buffer_pool.get_page(page_id).unwrap();
        assert_eq!(page2.lock().unwrap().header.page_id, page_id);

        // Flush the page
        buffer_pool.flush_page(page_id).unwrap();

        let stats = buffer_pool.get_stats().unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_buffer_pool_eviction() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = BufferPoolConfig {
            max_pages: 2,
            ..Default::default()
        };
        let buffer_pool = BufferPool::with_config(temp_file.path(), config).unwrap();

        // Create pages up to capacity
        let (_page1_id, _) = buffer_pool.create_page(PageType::Data).unwrap();
        let (_page2_id, _) = buffer_pool.create_page(PageType::Data).unwrap();

        // Check initial stats
        let initial_stats = buffer_pool.get_stats().unwrap();
        assert_eq!(initial_stats.evicted_pages, 0);

        // Creating a third page should trigger automatic eviction
        let (_page3_id, _) = buffer_pool.create_page(PageType::Data).unwrap();

        // Check that eviction occurred during page creation
        let stats = buffer_pool.get_stats().unwrap();
        assert!(stats.evicted_pages > 0);

        // Verify buffer pool is at capacity
        let current_count = {
            let pages = buffer_pool.pages.read().unwrap();
            pages.len()
        };
        assert_eq!(current_count, 2); // Should be at max capacity
    }
}
