//! Buffer pool with LRU caching
//!
//! This module implements an in-memory buffer pool for caching disk pages
//! with LRU (Least Recently Used) eviction policy.

use crate::error::{Result, TdbError};
use crate::storage::file_manager::FileManager;
use crate::storage::page::{Page, PageId, PageType};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Frame ID within the buffer pool
pub type FrameId = usize;

/// A buffer frame holding a cached page
struct BufferFrame {
    /// The cached page (None if frame is empty)
    page: RwLock<Option<Page>>,
    /// Access counter for LRU (higher = more recently used)
    access_count: AtomicU64,
    /// Whether this frame is currently pinned
    pin_count: AtomicU64,
}

impl BufferFrame {
    fn new() -> Self {
        Self {
            page: RwLock::new(None),
            access_count: AtomicU64::new(0),
            pin_count: AtomicU64::new(0),
        }
    }

    fn is_empty(&self) -> bool {
        self.page.read().is_none()
    }

    fn is_pinned(&self) -> bool {
        self.pin_count.load(Ordering::Acquire) > 0
    }

    fn pin(&self) {
        self.pin_count.fetch_add(1, Ordering::AcqRel);
    }

    fn unpin(&self) {
        let prev = self.pin_count.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "Unpin called on unpinned frame");
    }

    fn access(&self) {
        self.access_count.fetch_add(1, Ordering::AcqRel);
    }

    fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::Acquire)
    }
}

/// Buffer pool for caching pages in memory
///
/// The buffer pool maintains a fixed-size cache of pages loaded from disk.
/// When the cache is full, it uses LRU eviction to make room for new pages.
pub struct BufferPool {
    /// Fixed-size array of buffer frames
    frames: Vec<BufferFrame>,
    /// Map from PageId to FrameId
    page_table: DashMap<PageId, FrameId>,
    /// File manager for disk I/O
    file_manager: Arc<FileManager>,
    /// Next victim for eviction (clock hand)
    clock_hand: AtomicU64,
    /// Statistics
    stats: BufferPoolStats,
}

/// Buffer pool statistics
#[derive(Debug, Default)]
pub struct BufferPoolStats {
    /// Total number of page fetches
    pub total_fetches: AtomicU64,
    /// Number of cache hits
    pub cache_hits: AtomicU64,
    /// Number of cache misses
    pub cache_misses: AtomicU64,
    /// Number of page evictions
    pub evictions: AtomicU64,
    /// Number of pages written to disk
    pub writes: AtomicU64,
}

impl BufferPoolStats {
    /// Get cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_fetches.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            let hits = self.cache_hits.load(Ordering::Relaxed);
            hits as f64 / total as f64
        }
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.total_fetches.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.writes.store(0, Ordering::Relaxed);
    }

    /// Clone statistics (manual implementation)
    pub fn snapshot(&self) -> Self {
        Self {
            total_fetches: AtomicU64::new(self.total_fetches.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(Ordering::Relaxed)),
            evictions: AtomicU64::new(self.evictions.load(Ordering::Relaxed)),
            writes: AtomicU64::new(self.writes.load(Ordering::Relaxed)),
        }
    }
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(pool_size: usize, file_manager: Arc<FileManager>) -> Self {
        let mut frames = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            frames.push(BufferFrame::new());
        }

        Self {
            frames,
            page_table: DashMap::new(),
            file_manager,
            clock_hand: AtomicU64::new(0),
            stats: BufferPoolStats::default(),
        }
    }

    /// Fetch a page (from cache or disk)
    pub fn fetch_page(&self, page_id: PageId) -> Result<PageGuard<'_>> {
        self.stats.total_fetches.fetch_add(1, Ordering::Relaxed);

        // Check if page is already in buffer pool
        if let Some(frame_entry) = self.page_table.get(&page_id) {
            let frame_id = *frame_entry;
            let frame = &self.frames[frame_id];
            frame.pin();
            frame.access();
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(PageGuard {
                frame_id,
                page_id,
                buffer_pool: self,
            });
        }

        // Cache miss - need to load from disk
        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Find a free or evictable frame
        let frame_id = self.find_victim_frame()?;
        let frame = &self.frames[frame_id];

        // Evict old page if frame is occupied
        {
            let mut page_guard = frame.page.write();
            if let Some(old_page) = page_guard.take() {
                // Write dirty page back to disk
                if old_page.is_dirty() {
                    let old_page_id = old_page.page_id();
                    self.page_table.remove(&old_page_id);
                    drop(page_guard); // Release lock before I/O

                    let mut write_page = old_page;
                    self.file_manager.write_page(&mut write_page)?;
                    self.stats.writes.fetch_add(1, Ordering::Relaxed);

                    page_guard = frame.page.write();
                } else {
                    let old_page_id = old_page.page_id();
                    self.page_table.remove(&old_page_id);
                }
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }

            // Load new page from disk
            let page = self.file_manager.read_page(page_id)?;
            *page_guard = Some(page);
        }

        // Update page table
        self.page_table.insert(page_id, frame_id);
        frame.pin();
        frame.access();

        Ok(PageGuard {
            frame_id,
            page_id,
            buffer_pool: self,
        })
    }

    /// Create a new page
    pub fn new_page(&self, page_type: PageType) -> Result<PageGuard<'_>> {
        // Allocate new page on disk
        let page_id = self.file_manager.allocate_page()?;

        // Find a free frame
        let frame_id = self.find_victim_frame()?;
        let frame = &self.frames[frame_id];

        // Evict old page if needed
        {
            let mut page_guard = frame.page.write();
            if let Some(old_page) = page_guard.take() {
                if old_page.is_dirty() {
                    let old_page_id = old_page.page_id();
                    self.page_table.remove(&old_page_id);
                    drop(page_guard);

                    let mut write_page = old_page;
                    self.file_manager.write_page(&mut write_page)?;
                    self.stats.writes.fetch_add(1, Ordering::Relaxed);

                    page_guard = frame.page.write();
                } else {
                    let old_page_id = old_page.page_id();
                    self.page_table.remove(&old_page_id);
                }
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }

            // Create new page
            let page = Page::new(page_id, page_type);
            *page_guard = Some(page);
        }

        self.page_table.insert(page_id, frame_id);
        frame.pin();
        frame.access();

        Ok(PageGuard {
            frame_id,
            page_id,
            buffer_pool: self,
        })
    }

    /// Flush a specific page to disk
    pub fn flush_page(&self, page_id: PageId) -> Result<()> {
        if let Some(frame_entry) = self.page_table.get(&page_id) {
            let frame_id = *frame_entry;
            let frame = &self.frames[frame_id];
            let mut page_guard = frame.page.write();

            if let Some(page) = page_guard.as_mut() {
                if page.is_dirty() {
                    self.file_manager.write_page(page)?;
                    self.stats.writes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }

    /// Flush all dirty pages to disk
    pub fn flush_all(&self) -> Result<()> {
        for entry in self.page_table.iter() {
            let frame_id = *entry.value();
            let frame = &self.frames[frame_id];
            let mut page_guard = frame.page.write();

            if let Some(page) = page_guard.as_mut() {
                if page.is_dirty() {
                    self.file_manager.write_page(page)?;
                    self.stats.writes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.file_manager.flush()?;
        Ok(())
    }

    /// Unpin a page
    fn unpin_page(&self, frame_id: FrameId) {
        self.frames[frame_id].unpin();
    }

    /// Find a victim frame for eviction (Clock algorithm)
    fn find_victim_frame(&self) -> Result<FrameId> {
        let pool_size = self.frames.len();
        let mut iterations = 0;
        let max_iterations = pool_size * 2; // Two full sweeps

        loop {
            if iterations >= max_iterations {
                return Err(TdbError::BufferPoolFull);
            }

            let hand = self.clock_hand.fetch_add(1, Ordering::AcqRel) as usize % pool_size;
            let frame = &self.frames[hand];

            // Skip pinned frames
            if frame.is_pinned() {
                iterations += 1;
                continue;
            }

            // Check if frame is empty
            if frame.is_empty() {
                return Ok(hand);
            }

            // LRU-based eviction: check access count
            let access_count = frame.get_access_count();
            if access_count == 0 {
                // Frame hasn't been accessed recently, evict it
                return Ok(hand);
            }

            // Decrement access count (second chance)
            frame.access_count.fetch_sub(1, Ordering::AcqRel);
            iterations += 1;
        }
    }

    /// Get buffer pool statistics
    pub fn stats(&self) -> BufferPoolStats {
        self.stats.snapshot()
    }

    /// Get pool size
    pub fn pool_size(&self) -> usize {
        self.frames.len()
    }

    /// Get number of cached pages
    pub fn cached_pages(&self) -> usize {
        self.page_table.len()
    }
}

/// RAII guard for a pinned page
///
/// Automatically unpins the page when dropped.
pub struct PageGuard<'a> {
    frame_id: FrameId,
    page_id: PageId,
    buffer_pool: &'a BufferPool,
}

impl<'a> PageGuard<'a> {
    /// Get immutable reference to the page
    pub fn page(&self) -> parking_lot::RwLockReadGuard<'_, Option<Page>> {
        self.buffer_pool.frames[self.frame_id].page.read()
    }

    /// Get mutable reference to the page
    pub fn page_mut(&self) -> parking_lot::RwLockWriteGuard<'_, Option<Page>> {
        self.buffer_pool.frames[self.frame_id].page.write()
    }

    /// Get page ID
    pub fn page_id(&self) -> PageId {
        self.page_id
    }
}

impl<'a> Drop for PageGuard<'a> {
    fn drop(&mut self) {
        self.buffer_pool.unpin_page(self.frame_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_buffer_pool_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(10, fm);

        assert_eq!(bp.pool_size(), 10);
        assert_eq!(bp.cached_pages(), 0);
    }

    #[test]
    fn test_buffer_pool_new_page() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(10, fm);

        let guard = bp.new_page(PageType::BTreeLeaf).unwrap();
        assert_eq!(guard.page_id(), 0);
        assert_eq!(bp.cached_pages(), 1);
    }

    #[test]
    fn test_buffer_pool_fetch_page() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(10, fm.clone());

        // Create and write a page
        let page_id = {
            let guard = bp.new_page(PageType::BTreeLeaf).unwrap();
            let mut page = guard.page_mut();
            page.as_mut().unwrap().write_at(0, b"test data").unwrap();
            guard.page_id()
        };

        // Flush to disk
        bp.flush_page(page_id).unwrap();

        // Fetch the page
        let guard = bp.fetch_page(page_id).unwrap();
        let page = guard.page();
        let data = page.as_ref().unwrap().read_at(0, 9).unwrap();
        assert_eq!(data, b"test data");
    }

    #[test]
    fn test_buffer_pool_cache_hit() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(10, fm);

        let guard1 = bp.new_page(PageType::BTreeLeaf).unwrap();
        let page_id = guard1.page_id();
        drop(guard1);

        // Fetch same page - should be cache hit
        let _guard2 = bp.fetch_page(page_id).unwrap();

        let stats = bp.stats();
        assert!(stats.cache_hits.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_buffer_pool_eviction() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(3, fm); // Small pool

        // Fill the pool
        let _g1 = bp.new_page(PageType::BTreeLeaf).unwrap();
        let _g2 = bp.new_page(PageType::BTreeLeaf).unwrap();
        let _g3 = bp.new_page(PageType::BTreeLeaf).unwrap();

        assert_eq!(bp.cached_pages(), 3);

        // Drop guards to allow eviction
        drop(_g1);
        drop(_g2);
        drop(_g3);

        // Allocate one more - should trigger eviction
        let _g4 = bp.new_page(PageType::BTreeLeaf).unwrap();

        let stats = bp.stats();
        assert!(stats.evictions.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_buffer_pool_flush_all() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(10, fm);

        // Create some dirty pages by writing data to them
        {
            let g1 = bp.new_page(PageType::BTreeLeaf).unwrap();
            let mut page1 = g1.page_mut();
            page1.as_mut().unwrap().write_at(0, b"data1").unwrap();
        }
        {
            let g2 = bp.new_page(PageType::BTreeLeaf).unwrap();
            let mut page2 = g2.page_mut();
            page2.as_mut().unwrap().write_at(0, b"data2").unwrap();
        }

        bp.flush_all().unwrap();

        let stats = bp.stats();
        assert!(stats.writes.load(Ordering::Relaxed) >= 2);
    }

    #[test]
    fn test_buffer_pool_hit_rate() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(10, fm);

        let guard = bp.new_page(PageType::BTreeLeaf).unwrap();
        let page_id = guard.page_id();
        drop(guard);

        // Multiple fetches of same page
        for _ in 0..5 {
            let _g = bp.fetch_page(page_id).unwrap();
        }

        let stats = bp.stats();
        assert!(stats.hit_rate() > 0.8); // Should be high
    }

    #[test]
    fn test_buffer_pool_pin_prevents_eviction() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(2, fm); // Very small pool

        let g1 = bp.new_page(PageType::BTreeLeaf).unwrap();
        let page_id1 = g1.page_id();
        // Keep g1 pinned

        let g2 = bp.new_page(PageType::BTreeLeaf).unwrap();
        drop(g2);

        // Try to allocate one more - should not evict pinned page
        let result = bp.new_page(PageType::BTreeLeaf);

        // Should succeed (evicts g2, not g1)
        assert!(result.is_ok());

        // g1 should still be cached
        assert!(bp.page_table.contains_key(&page_id1));
    }
}
