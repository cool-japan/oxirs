//! Buffer pool with LRU caching
//!
//! This module implements an in-memory buffer pool for caching disk pages
//! with LRU (Least Recently Used) eviction policy.

use crate::error::{Result, TdbError};
use crate::storage::file_manager::FileManager;
use crate::storage::page::{Page, PageId, PageType};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Frame ID within the buffer pool
pub type FrameId = usize;

/// Where the contents of a freshly-loaded frame come from.
enum LoadSource {
    /// Read an existing page from disk.
    Disk,
    /// Initialise a brand-new (just-allocated) page in memory.
    New(PageType),
}

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
    /// Serializes the miss path (victim selection + eviction + load) so two
    /// concurrent misses can never select and evict the same frame — the root
    /// cause of the previous cross-page corruption (a TOCTOU between victim
    /// selection and pinning). Cache hits do not take this lock.
    load_lock: Mutex<()>,
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

impl Clone for BufferPoolStats {
    fn clone(&self) -> Self {
        Self {
            total_fetches: AtomicU64::new(self.total_fetches.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(Ordering::Relaxed)),
            evictions: AtomicU64::new(self.evictions.load(Ordering::Relaxed)),
            writes: AtomicU64::new(self.writes.load(Ordering::Relaxed)),
        }
    }
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
            load_lock: Mutex::new(()),
            stats: BufferPoolStats::default(),
        }
    }

    /// Access the underlying file manager (used by the store to persist the
    /// superblock directly, outside the cached page space).
    pub fn file_manager(&self) -> &Arc<FileManager> {
        &self.file_manager
    }

    /// Fetch a page (from cache or disk)
    pub fn fetch_page(&self, page_id: PageId) -> Result<PageGuard<'_>> {
        self.stats.total_fetches.fetch_add(1, Ordering::Relaxed);

        // Fast path: page already cached. `try_hit` validates the frame's
        // identity under its latch and pins it, so it cannot return a guard to
        // a frame that a concurrent eviction has re-assigned.
        if let Some(guard) = self.try_hit(page_id) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(guard);
        }

        // Miss: serialize the load so two misses never race on the same victim.
        let _load = self.load_lock.lock();

        // Double-check under the load lock — another thread may have loaded the
        // page while we waited.
        if let Some(guard) = self.try_hit(page_id) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(guard);
        }

        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        let frame_id = self.evict_and_load(page_id, LoadSource::Disk)?;
        Ok(PageGuard {
            frame_id,
            page_id,
            buffer_pool: self,
        })
    }

    /// Create a new page
    pub fn new_page(&self, page_type: PageType) -> Result<PageGuard<'_>> {
        // Serialize with the miss path: allocation + frame claiming must be
        // atomic with respect to concurrent evictions.
        let _load = self.load_lock.lock();
        let page_id = self.file_manager.allocate_page()?;
        let frame_id = self.evict_and_load(page_id, LoadSource::New(page_type))?;
        Ok(PageGuard {
            frame_id,
            page_id,
            buffer_pool: self,
        })
    }

    /// Fast cache-hit path.
    ///
    /// Returns a pinned [`PageGuard`] only if `page_id` is resident and the
    /// frame still holds it. The pin is taken while holding the frame's read
    /// latch, which is mutually exclusive with the write latch that
    /// [`Self::evict_and_load`] holds across an eviction, so a hit can never be
    /// granted on a frame mid-reassignment.
    fn try_hit(&self, page_id: PageId) -> Option<PageGuard<'_>> {
        let frame_id = *self.page_table.get(&page_id)?;
        let frame = &self.frames[frame_id];

        // Latch the frame's identity, then verify it before pinning.
        let read = frame.page.read();
        let holds_page = read.as_ref().map(|p| p.page_id()) == Some(page_id);
        let still_mapped = self.page_table.get(&page_id).map(|e| *e) == Some(frame_id);
        if holds_page && still_mapped {
            frame.pin();
            frame.access();
            drop(read);
            Some(PageGuard {
                frame_id,
                page_id,
                buffer_pool: self,
            })
        } else {
            None
        }
    }

    /// Select a victim frame, evict its resident page (writing it back if
    /// dirty), load `page_id` into it, register it in the page table, and pin
    /// it — all while holding the frame's write latch so the frame's identity
    /// change is atomic. Returns the pinned frame id.
    ///
    /// Must be called with `load_lock` held.
    fn evict_and_load(&self, page_id: PageId, source: LoadSource) -> Result<FrameId> {
        let pool_size = self.frames.len();
        let mut checked = 0usize;
        let max_checks = pool_size * 4; // enough sweeps to give every frame a second chance

        loop {
            if checked >= max_checks {
                return Err(TdbError::BufferPoolFull);
            }

            let hand = self.clock_hand.fetch_add(1, Ordering::AcqRel) as usize % pool_size;
            let frame = &self.frames[hand];

            // Cheap pre-check: skip pinned frames without latching.
            if frame.is_pinned() {
                checked += 1;
                continue;
            }

            // Latch the frame exclusively; this excludes `try_hit` readers.
            let mut page_guard = frame.page.write();

            // A concurrent hit may have pinned the frame between the pre-check
            // and acquiring the latch — re-check under the latch.
            if frame.is_pinned() {
                drop(page_guard);
                checked += 1;
                continue;
            }

            if page_guard.is_some() {
                // Second-chance: skip recently-accessed occupied frames.
                if frame.get_access_count() > 0 {
                    frame.access_count.fetch_sub(1, Ordering::AcqRel);
                    drop(page_guard);
                    checked += 1;
                    continue;
                }

                // Evict the resident page.
                let is_dirty = page_guard.as_ref().map(|p| p.is_dirty()).unwrap_or(false);
                let old_id = page_guard.as_ref().map(|p| p.page_id());
                if is_dirty {
                    if let Some(mut old_page) = page_guard.take() {
                        self.file_manager.write_page(&mut old_page)?;
                        self.stats.writes.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    *page_guard = None;
                }
                if let Some(old_id) = old_id {
                    self.page_table.remove(&old_id);
                }
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }

            // Install the new page content.
            let new_page = match source {
                LoadSource::Disk => self.file_manager.read_page(page_id)?,
                LoadSource::New(page_type) => Page::new(page_id, page_type),
            };
            *page_guard = Some(new_page);

            // Register + pin BEFORE releasing the latch so no concurrent
            // eviction or hit can observe an inconsistent (mapped-but-unpinned
            // or reassigned) state.
            self.page_table.insert(page_id, hand);
            frame.pin();
            frame.access();
            drop(page_guard);
            return Ok(hand);
        }
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

    /// Drop a page from the cache WITHOUT writing it back to disk.
    ///
    /// This is required before a page is returned to the file manager's free
    /// list (see [`crate::storage::FileManager::free_page`]): `free_page` writes a
    /// free-list node directly through the file manager, bypassing the buffer
    /// pool. If a stale (possibly dirty) copy of that page remained cached, a
    /// later [`Self::flush_all`] would write it back over the free-list node,
    /// corrupting the free list. Discarding the cached frame first closes that
    /// cache-coherence hazard.
    ///
    /// The caller must ensure the page is no longer pinned (no live
    /// [`PageGuard`] to it); a pinned frame is left in place and `false` is
    /// returned so the caller can detect the misuse. Returns `true` when the page
    /// was resident and has been evicted (or was not cached at all).
    pub fn discard_page(&self, page_id: PageId) -> bool {
        // Serialize with the miss/eviction path so we never race a concurrent
        // load that is re-assigning a frame.
        let _load = self.load_lock.lock();

        let frame_id = match self.page_table.get(&page_id).map(|e| *e) {
            Some(fid) => fid,
            None => return true, // not cached: nothing to do
        };

        let frame = &self.frames[frame_id];
        if frame.is_pinned() {
            // A live guard still references this page; refuse to discard.
            return false;
        }

        let mut page_guard = frame.page.write();
        // Re-check identity under the latch.
        if page_guard.as_ref().map(|p| p.page_id()) == Some(page_id) {
            *page_guard = None;
            frame.access_count.store(0, Ordering::Release);
            self.page_table.remove(&page_id);
        }
        true
    }

    /// Unpin a page
    fn unpin_page(&self, frame_id: FrameId) {
        self.frames[frame_id].unpin();
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
    /// Get immutable reference to the page.
    ///
    /// In debug builds this asserts the frame still holds the page this guard
    /// was created for; for a fallible, always-on identity check use
    /// [`PageGuard::page_checked`] (which the B+Tree hot path uses).
    pub fn page(&self) -> parking_lot::RwLockReadGuard<'_, Option<Page>> {
        let guard = self.buffer_pool.frames[self.frame_id].page.read();
        debug_assert!(
            guard.as_ref().map(|p| p.page_id()) == Some(self.page_id),
            "PageGuard identity mismatch on frame {}: expected page {}",
            self.frame_id,
            self.page_id
        );
        guard
    }

    /// Get mutable reference to the page (see [`PageGuard::page`]).
    pub fn page_mut(&self) -> parking_lot::RwLockWriteGuard<'_, Option<Page>> {
        let guard = self.buffer_pool.frames[self.frame_id].page.write();
        debug_assert!(
            guard.as_ref().map(|p| p.page_id()) == Some(self.page_id),
            "PageGuard identity mismatch on frame {}: expected page {}",
            self.frame_id,
            self.page_id
        );
        guard
    }

    /// Get an immutable reference to the page, verifying that the frame still
    /// holds the expected page id.
    ///
    /// Returns [`TdbError::PageIdMismatch`] if the buffer frame has been
    /// re-assigned to a different page, turning what would otherwise be a
    /// silent wrong-page read into a detectable error.
    pub fn page_checked(&self) -> Result<parking_lot::RwLockReadGuard<'_, Option<Page>>> {
        let guard = self.buffer_pool.frames[self.frame_id].page.read();
        match guard.as_ref().map(|p| p.page_id()) {
            Some(actual) if actual == self.page_id => Ok(guard),
            Some(actual) => Err(TdbError::PageIdMismatch {
                expected: self.page_id,
                actual,
            }),
            None => Err(TdbError::PageNotFound(self.page_id)),
        }
    }

    /// Get a mutable reference to the page, verifying that the frame still
    /// holds the expected page id (see [`PageGuard::page_checked`]).
    pub fn page_mut_checked(&self) -> Result<parking_lot::RwLockWriteGuard<'_, Option<Page>>> {
        let guard = self.buffer_pool.frames[self.frame_id].page.write();
        match guard.as_ref().map(|p| p.page_id()) {
            Some(actual) if actual == self.page_id => Ok(guard),
            Some(actual) => Err(TdbError::PageIdMismatch {
                expected: self.page_id,
                actual,
            }),
            None => Err(TdbError::PageNotFound(self.page_id)),
        }
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

    #[test]
    fn test_page_guard_checked_matches_identity() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        let bp = BufferPool::new(10, fm);

        let guard = bp.new_page(PageType::BTreeLeaf).unwrap();
        // A guard over a resident, correctly-mapped frame validates cleanly.
        // Each temporary guard is dropped at the end of its statement, so the
        // read latch is released before the write latch is requested.
        assert!(guard.page_checked().is_ok());
        assert!(guard.page_mut_checked().is_ok());
    }

    #[test]
    fn test_buffer_pool_concurrent_no_cross_page_corruption() {
        use std::thread;

        let temp_file = NamedTempFile::new().unwrap();
        let fm = Arc::new(FileManager::open(temp_file.path(), false).unwrap());
        // Deliberately tiny pool vs. page count to force heavy eviction and
        // exercise the miss/eviction race that previously corrupted frames.
        let bp = Arc::new(BufferPool::new(8, fm));
        let num_pages: u64 = 64;

        // Stamp each page's own id into its first 8 bytes.
        for _ in 0..num_pages {
            let guard = bp.new_page(PageType::BTreeLeaf).unwrap();
            let pid = guard.page_id();
            {
                let mut page = guard.page_mut();
                page.as_mut()
                    .unwrap()
                    .write_at(0, &pid.to_le_bytes())
                    .unwrap();
            }
        }
        bp.flush_all().unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let bp = bp.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000u64 {
                    let pid = i % num_pages;
                    let guard = bp.fetch_page(pid).expect("fetch");
                    let page_lock = guard.page_checked().expect("frame identity");
                    let page = page_lock.as_ref().expect("resident");
                    let bytes = page.read_at(0, 8).expect("read");
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(bytes);
                    assert_eq!(
                        u64::from_le_bytes(arr),
                        pid,
                        "buffer frame returned bytes belonging to a different page"
                    );
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked (corruption detected)");
        }
    }
}
