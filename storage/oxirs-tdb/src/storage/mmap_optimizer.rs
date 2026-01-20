//! Memory-mapped file optimization
//!
//! This module provides advanced memory-mapped file optimizations for high-performance
//! RDF storage, including OS-level caching hints, huge pages support, and intelligent
//! prefetching.

use crate::error::{Result, TdbError};
use crate::storage::page::{PageId, PAGE_SIZE};
use memmap2::{MmapMut, MmapOptions};
use parking_lot::RwLock;
use std::fs::File;
use std::sync::Arc;

/// Memory access pattern hint for OS-level optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Normal access pattern (default)
    Normal,
    /// Sequential access (forward scan)
    Sequential,
    /// Random access (point queries)
    Random,
    /// Will need soon (prefetch hint)
    WillNeed,
    /// Won't need (eviction hint)
    DontNeed,
}

/// Configuration for memory-mapped file optimization
#[derive(Debug, Clone)]
pub struct MmapOptimizerConfig {
    /// Use huge pages (2MB or 1GB pages on Linux)
    pub use_huge_pages: bool,
    /// Enable transparent huge pages (Linux kernel feature)
    pub enable_thp: bool,
    /// Prefetch window size in pages
    pub prefetch_window: usize,
    /// Maximum memory regions to manage
    pub max_regions: usize,
    /// Lock pages in memory (prevent swapping)
    pub lock_memory: bool,
    /// Default access pattern
    pub default_pattern: AccessPattern,
}

impl Default for MmapOptimizerConfig {
    fn default() -> Self {
        Self {
            use_huge_pages: true,
            enable_thp: true,
            prefetch_window: 64, // 64 pages = 256KB
            max_regions: 16,
            lock_memory: false,
            default_pattern: AccessPattern::Normal,
        }
    }
}

/// Statistics for memory-mapped file usage
#[derive(Debug, Clone, Default)]
pub struct MmapStats {
    /// Total bytes mapped
    pub bytes_mapped: u64,
    /// Number of memory regions
    pub num_regions: usize,
    /// Number of prefetch operations
    pub prefetch_count: u64,
    /// Number of advise calls
    pub advise_count: u64,
    /// Number of huge pages used (if supported)
    pub huge_pages_used: usize,
    /// Number of page faults (approximation)
    pub page_faults: u64,
}

/// Memory region tracking
#[derive(Debug)]
struct MmapRegion {
    /// Memory-mapped region
    mmap: MmapMut,
    /// Start page ID in this region
    start_page: PageId,
    /// Number of pages in this region
    num_pages: u64,
    /// Current access pattern
    access_pattern: AccessPattern,
}

/// Optimized memory-mapped file manager
///
/// Provides advanced features:
/// - OS-level caching hints (madvise)
/// - Huge pages support for reduced TLB misses
/// - Intelligent prefetching based on access patterns
/// - Multi-region management for large files
/// - Statistics tracking
pub struct MmapOptimizer {
    /// Configuration
    config: MmapOptimizerConfig,
    /// Memory regions
    regions: Arc<RwLock<Vec<MmapRegion>>>,
    /// Statistics
    stats: Arc<RwLock<MmapStats>>,
}

impl MmapOptimizer {
    /// Create a new memory-mapped file optimizer
    pub fn new(config: MmapOptimizerConfig) -> Self {
        Self {
            config,
            regions: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(MmapStats::default())),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(MmapOptimizerConfig::default())
    }

    /// Map a file region
    pub fn map_region(&self, file: &File, start_page: PageId, num_pages: u64) -> Result<()> {
        let offset = start_page * PAGE_SIZE as u64;
        let length = num_pages * PAGE_SIZE as u64;

        // Create memory-mapped region
        // Note: `mut` is required on Linux for apply_huge_pages but unused on macOS
        #[allow(unused_mut)]
        let mut mmap = unsafe {
            MmapOptions::new()
                .offset(offset)
                .len(length as usize)
                .map_mut(file)?
        };

        // Apply huge pages hint if enabled
        #[cfg(target_os = "linux")]
        if self.config.use_huge_pages || self.config.enable_thp {
            self.apply_huge_pages(&mut mmap)?;
        }

        // Apply default access pattern
        self.apply_access_pattern_internal(&mmap, self.config.default_pattern)?;

        // Lock memory if requested
        if self.config.lock_memory {
            self.lock_memory_internal(&mmap)?;
        }

        // Store region
        let region = MmapRegion {
            mmap,
            start_page,
            num_pages,
            access_pattern: self.config.default_pattern,
        };

        let mut regions = self.regions.write();
        if regions.len() >= self.config.max_regions {
            return Err(TdbError::Other(
                "Maximum number of memory regions reached".to_string(),
            ));
        }
        regions.push(region);

        // Update statistics
        let mut stats = self.stats.write();
        stats.bytes_mapped += length;
        stats.num_regions = regions.len();

        Ok(())
    }

    /// Set access pattern hint for a page range
    pub fn set_access_pattern(
        &self,
        start_page: PageId,
        num_pages: u64,
        pattern: AccessPattern,
    ) -> Result<()> {
        let regions = self.regions.read();

        // Find the region containing this page range
        for region in regions.iter() {
            if start_page >= region.start_page && start_page < region.start_page + region.num_pages
            {
                let offset = (start_page - region.start_page) * PAGE_SIZE as u64;
                let length = num_pages * PAGE_SIZE as u64;

                self.apply_access_pattern_range(
                    &region.mmap,
                    offset as usize,
                    length as usize,
                    pattern,
                )?;

                // Update statistics
                let mut stats = self.stats.write();
                stats.advise_count += 1;

                return Ok(());
            }
        }

        Err(TdbError::PageNotFound(start_page))
    }

    /// Prefetch a range of pages
    pub fn prefetch(&self, start_page: PageId, num_pages: u64) -> Result<()> {
        self.set_access_pattern(start_page, num_pages, AccessPattern::WillNeed)?;

        let mut stats = self.stats.write();
        stats.prefetch_count += 1;

        Ok(())
    }

    /// Advise pages won't be needed (allow eviction)
    pub fn evict_hint(&self, start_page: PageId, num_pages: u64) -> Result<()> {
        self.set_access_pattern(start_page, num_pages, AccessPattern::DontNeed)
    }

    /// Read data from a mapped page
    pub fn read_page(&self, page_id: PageId, buffer: &mut [u8; PAGE_SIZE]) -> Result<()> {
        let regions = self.regions.read();

        // Find the region containing this page
        for region in regions.iter() {
            if page_id >= region.start_page && page_id < region.start_page + region.num_pages {
                let offset = (page_id - region.start_page) * PAGE_SIZE as u64;
                let start = offset as usize;
                let end = start + PAGE_SIZE;

                buffer.copy_from_slice(&region.mmap[start..end]);
                return Ok(());
            }
        }

        Err(TdbError::PageNotFound(page_id))
    }

    /// Write data to a mapped page
    pub fn write_page(&self, page_id: PageId, data: &[u8; PAGE_SIZE]) -> Result<()> {
        let mut regions = self.regions.write();

        // Find the region containing this page
        for region in regions.iter_mut() {
            if page_id >= region.start_page && page_id < region.start_page + region.num_pages {
                let offset = (page_id - region.start_page) * PAGE_SIZE as u64;
                let start = offset as usize;
                let end = start + PAGE_SIZE;

                region.mmap[start..end].copy_from_slice(data);
                return Ok(());
            }
        }

        Err(TdbError::PageNotFound(page_id))
    }

    /// Flush a specific page range to disk
    pub fn flush_range(&self, start_page: PageId, num_pages: u64) -> Result<()> {
        let regions = self.regions.read();

        for region in regions.iter() {
            if start_page >= region.start_page && start_page < region.start_page + region.num_pages
            {
                let offset = (start_page - region.start_page) * PAGE_SIZE as u64;
                let length = num_pages * PAGE_SIZE as u64;

                region.mmap.flush_range(offset as usize, length as usize)?;
                return Ok(());
            }
        }

        Ok(())
    }

    /// Flush all mapped regions to disk
    pub fn flush_all(&self) -> Result<()> {
        let regions = self.regions.read();

        for region in regions.iter() {
            region.mmap.flush()?;
        }

        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> MmapStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = MmapStats::default();
    }

    // Internal helper methods

    #[cfg(target_os = "linux")]
    fn apply_huge_pages(&self, mmap: &mut MmapMut) -> Result<()> {
        use std::io;

        // MADV_HUGEPAGE - Use transparent huge pages if available
        let ptr = mmap.as_ptr() as *mut std::ffi::c_void;
        let len = mmap.len();

        unsafe {
            // MADV_HUGEPAGE = 14 on Linux
            let ret = libc::madvise(ptr, len, 14);
            if ret != 0 {
                let err = io::Error::last_os_error();
                // Don't fail if huge pages aren't supported, just log
                log::debug!("Could not enable huge pages: {}", err);
            } else {
                let mut stats = self.stats.write();
                // Approximate number of huge pages (2MB each)
                stats.huge_pages_used = (len + 2 * 1024 * 1024 - 1) / (2 * 1024 * 1024);
            }
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn apply_huge_pages(&self, _mmap: &mut MmapMut) -> Result<()> {
        // Huge pages are Linux-specific
        Ok(())
    }

    fn apply_access_pattern_internal(&self, mmap: &MmapMut, pattern: AccessPattern) -> Result<()> {
        self.apply_access_pattern_range(mmap, 0, mmap.len(), pattern)
    }

    #[cfg(unix)]
    fn apply_access_pattern_range(
        &self,
        mmap: &MmapMut,
        offset: usize,
        length: usize,
        pattern: AccessPattern,
    ) -> Result<()> {
        use std::io;

        let ptr = unsafe { mmap.as_ptr().add(offset) as *mut std::ffi::c_void };

        let advice = match pattern {
            AccessPattern::Normal => libc::MADV_NORMAL,
            AccessPattern::Sequential => libc::MADV_SEQUENTIAL,
            AccessPattern::Random => libc::MADV_RANDOM,
            AccessPattern::WillNeed => libc::MADV_WILLNEED,
            AccessPattern::DontNeed => libc::MADV_DONTNEED,
        };

        unsafe {
            let ret = libc::madvise(ptr, length, advice);
            if ret != 0 {
                let err = io::Error::last_os_error();
                return Err(TdbError::Io(err));
            }
        }

        Ok(())
    }

    #[cfg(not(unix))]
    fn apply_access_pattern_range(
        &self,
        _mmap: &MmapMut,
        _offset: usize,
        _length: usize,
        _pattern: AccessPattern,
    ) -> Result<()> {
        // madvise is Unix-specific
        Ok(())
    }

    #[cfg(unix)]
    fn lock_memory_internal(&self, mmap: &MmapMut) -> Result<()> {
        use std::io;

        let ptr = mmap.as_ptr() as *const std::ffi::c_void;
        let len = mmap.len();

        unsafe {
            let ret = libc::mlock(ptr, len);
            if ret != 0 {
                let err = io::Error::last_os_error();
                log::warn!("Could not lock memory: {}", err);
            }
        }

        Ok(())
    }

    #[cfg(not(unix))]
    fn lock_memory_internal(&self, _mmap: &MmapMut) -> Result<()> {
        // mlock is Unix-specific
        Ok(())
    }
}

impl Drop for MmapOptimizer {
    fn drop(&mut self) {
        // Flush all regions before dropping
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_optimizer_creation() {
        let config = MmapOptimizerConfig::default();
        let optimizer = MmapOptimizer::new(config);

        let stats = optimizer.stats();
        assert_eq!(stats.bytes_mapped, 0);
        assert_eq!(stats.num_regions, 0);
    }

    #[test]
    fn test_map_region() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 10 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let optimizer = MmapOptimizer::default_config();
        let file = temp_file.reopen().unwrap();

        optimizer.map_region(&file, 0, 10).unwrap();

        let stats = optimizer.stats();
        assert_eq!(stats.bytes_mapped, file_size as u64);
        assert_eq!(stats.num_regions, 1);
    }

    #[test]
    fn test_read_write_page() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 10 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let optimizer = MmapOptimizer::default_config();
        let file = temp_file.reopen().unwrap();

        optimizer.map_region(&file, 0, 10).unwrap();

        // Write test data
        let mut write_data = [0u8; PAGE_SIZE];
        write_data[0..9].copy_from_slice(b"test data");
        optimizer.write_page(0, &write_data).unwrap();

        // Read it back
        let mut read_data = [0u8; PAGE_SIZE];
        optimizer.read_page(0, &mut read_data).unwrap();

        assert_eq!(&read_data[0..9], b"test data");
    }

    #[test]
    fn test_access_patterns() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 100 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let optimizer = MmapOptimizer::default_config();
        let file = temp_file.reopen().unwrap();

        optimizer.map_region(&file, 0, 100).unwrap();

        // Test different access patterns
        optimizer
            .set_access_pattern(0, 10, AccessPattern::Sequential)
            .unwrap();
        optimizer
            .set_access_pattern(10, 10, AccessPattern::Random)
            .unwrap();
        optimizer.prefetch(20, 10).unwrap();
        optimizer.evict_hint(30, 10).unwrap();

        let stats = optimizer.stats();
        assert!(stats.advise_count > 0);
        assert_eq!(stats.prefetch_count, 1);
    }

    #[test]
    fn test_flush_range() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 10 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let optimizer = MmapOptimizer::default_config();
        let file = temp_file.reopen().unwrap();

        optimizer.map_region(&file, 0, 10).unwrap();

        // Write and flush
        let mut data = [0u8; PAGE_SIZE];
        data[0..5].copy_from_slice(b"flush");
        optimizer.write_page(0, &data).unwrap();
        optimizer.flush_range(0, 1).unwrap();
    }

    #[test]
    fn test_flush_all() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 10 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let optimizer = MmapOptimizer::default_config();
        let file = temp_file.reopen().unwrap();

        optimizer.map_region(&file, 0, 10).unwrap();

        // Write to multiple pages
        let mut data = [0u8; PAGE_SIZE];
        for i in 0..5u64 {
            data[0] = i as u8;
            optimizer.write_page(i, &data).unwrap();
        }

        optimizer.flush_all().unwrap();
    }

    #[test]
    fn test_multiple_regions() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 100 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let optimizer = MmapOptimizer::default_config();
        let file = temp_file.reopen().unwrap();

        // Map multiple regions
        optimizer.map_region(&file, 0, 30).unwrap();
        optimizer.map_region(&file, 30, 30).unwrap();
        optimizer.map_region(&file, 60, 40).unwrap();

        let stats = optimizer.stats();
        assert_eq!(stats.num_regions, 3);
        assert_eq!(stats.bytes_mapped, file_size as u64);
    }

    #[test]
    fn test_page_not_found() {
        let optimizer = MmapOptimizer::default_config();

        let mut buffer = [0u8; PAGE_SIZE];
        let result = optimizer.read_page(999, &mut buffer);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_regions_limit() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 200 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let config = MmapOptimizerConfig {
            max_regions: 2,
            ..Default::default()
        };
        let optimizer = MmapOptimizer::new(config);
        let file = temp_file.reopen().unwrap();

        optimizer.map_region(&file, 0, 50).unwrap();
        optimizer.map_region(&file, 50, 50).unwrap();

        // This should fail (exceeds max_regions)
        let result = optimizer.map_region(&file, 100, 50);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_reset() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_size = 10 * PAGE_SIZE;
        temp_file.write_all(&vec![0u8; file_size]).unwrap();
        temp_file.flush().unwrap();

        let optimizer = MmapOptimizer::default_config();
        let file = temp_file.reopen().unwrap();

        optimizer.map_region(&file, 0, 10).unwrap();
        optimizer.prefetch(0, 5).unwrap();

        let stats = optimizer.stats();
        assert!(stats.prefetch_count > 0);

        optimizer.reset_stats();
        let stats = optimizer.stats();
        assert_eq!(stats.prefetch_count, 0);
    }

    #[test]
    fn test_huge_pages_config() {
        let config = MmapOptimizerConfig {
            use_huge_pages: true,
            enable_thp: true,
            ..Default::default()
        };

        let optimizer = MmapOptimizer::new(config);
        assert!(optimizer.config.use_huge_pages);
        assert!(optimizer.config.enable_thp);
    }

    #[test]
    fn test_lock_memory_config() {
        let config = MmapOptimizerConfig {
            lock_memory: true,
            ..Default::default()
        };

        let optimizer = MmapOptimizer::new(config);
        assert!(optimizer.config.lock_memory);
    }
}
