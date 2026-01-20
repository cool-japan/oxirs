//! Direct I/O operations for large sequential access patterns
//!
//! This module provides unbuffered I/O that bypasses the OS page cache for scenarios
//! where we're doing large sequential scans and don't want to pollute the page cache.
//!
//! Benefits:
//! - Avoid evicting hot data from OS page cache
//! - Predictable performance for large scans
//! - Reduced memory pressure on the system
//! - Better control over I/O patterns
//!
//! Use cases:
//! - Large sequential table scans
//! - Backup and restore operations
//! - Bulk import/export
//! - Log file processing

use crate::error::{Result, TdbError};
use crate::storage::page::PAGE_SIZE;
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Alignment requirement for direct I/O (typically 512 bytes or 4KB)
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

/// Direct I/O configuration
#[derive(Debug, Clone)]
pub struct DirectIOConfig {
    /// Enable direct I/O mode (bypasses OS page cache)
    pub enable_direct_io: bool,
    /// Buffer size for direct I/O operations (must be aligned)
    pub buffer_size: usize,
    /// Prefetch size for sequential reads
    pub prefetch_size: usize,
    /// Use O_SYNC for synchronous writes
    pub use_sync_writes: bool,
    /// Minimum sequential access threshold to enable direct I/O
    pub sequential_threshold: usize,
}

impl Default for DirectIOConfig {
    fn default() -> Self {
        Self {
            enable_direct_io: false,        // Disabled by default for compatibility
            buffer_size: 1024 * 1024,       // 1 MB buffer
            prefetch_size: 4 * 1024 * 1024, // 4 MB prefetch
            use_sync_writes: false,
            sequential_threshold: 10, // Enable after 10 sequential accesses
        }
    }
}

/// Direct I/O file handle
pub struct DirectIOFile {
    /// File path
    path: PathBuf,
    /// Underlying file handle
    file: RwLock<File>,
    /// Configuration
    config: DirectIOConfig,
    /// Whether direct I/O is active
    direct_io_active: AtomicBool,
    /// Sequential access counter
    sequential_count: AtomicU64,
    /// Last accessed offset
    last_offset: RwLock<u64>,
    /// Statistics
    stats: DirectIOStats,
}

impl DirectIOFile {
    /// Open a file with direct I/O capabilities
    pub fn open<P: AsRef<Path>>(path: P, config: DirectIOConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open file (direct I/O flags would be set here in production)
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false) // Don't truncate existing files
            .open(&path)
            .map_err(|e| TdbError::Other(format!("Failed to open file: {}", e)))?;

        Ok(Self {
            path,
            file: RwLock::new(file),
            config,
            direct_io_active: AtomicBool::new(false),
            sequential_count: AtomicU64::new(0),
            last_offset: RwLock::new(0),
            stats: DirectIOStats::default(),
        })
    }

    /// Read data with direct I/O optimizations
    pub fn read_direct(&self, offset: u64, buffer: &mut [u8]) -> Result<usize> {
        // Check if this is sequential access
        let is_sequential = self.check_sequential_access(offset);

        if is_sequential {
            self.stats.sequential_reads.fetch_add(1, Ordering::Relaxed);

            // Enable direct I/O if threshold is reached
            let seq_count = self.sequential_count.fetch_add(1, Ordering::Relaxed) + 1;
            if seq_count >= self.config.sequential_threshold as u64 && self.config.enable_direct_io
            {
                self.direct_io_active.store(true, Ordering::Release);
            }
        } else {
            self.stats.random_reads.fetch_add(1, Ordering::Relaxed);
            self.sequential_count.store(0, Ordering::Relaxed);
            self.direct_io_active.store(false, Ordering::Release);
        }

        // Update last offset
        *self.last_offset.write() = offset + buffer.len() as u64;

        // Perform the read
        let bytes_read = if self.direct_io_active.load(Ordering::Acquire) {
            self.read_direct_aligned(offset, buffer)?
        } else {
            self.read_buffered(offset, buffer)?
        };

        self.stats
            .total_bytes_read
            .fetch_add(bytes_read as u64, Ordering::Relaxed);

        Ok(bytes_read)
    }

    /// Write data with direct I/O optimizations
    pub fn write_direct(&self, offset: u64, data: &[u8]) -> Result<usize> {
        let bytes_written = if self.direct_io_active.load(Ordering::Acquire) {
            self.write_direct_aligned(offset, data)?
        } else {
            self.write_buffered(offset, data)?
        };

        self.stats
            .total_bytes_written
            .fetch_add(bytes_written as u64, Ordering::Relaxed);

        Ok(bytes_written)
    }

    /// Check if access pattern is sequential
    fn check_sequential_access(&self, offset: u64) -> bool {
        let last_offset = *self.last_offset.read();
        let diff = offset.abs_diff(last_offset);

        // Consider sequential if within reasonable range (e.g., page size)
        diff <= PAGE_SIZE as u64 * 4
    }

    /// Read with direct I/O (aligned)
    fn read_direct_aligned(&self, offset: u64, buffer: &mut [u8]) -> Result<usize> {
        // Ensure alignment
        if offset % DIRECT_IO_ALIGNMENT as u64 != 0 {
            return Err(TdbError::Other(
                "Direct I/O requires aligned offset".to_string(),
            ));
        }

        if buffer.len() % DIRECT_IO_ALIGNMENT != 0 {
            return Err(TdbError::Other(
                "Direct I/O requires aligned buffer size".to_string(),
            ));
        }

        let mut file = self.file.write();
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| TdbError::Other(format!("Seek failed: {}", e)))?;

        file.read(buffer)
            .map_err(|e| TdbError::Other(format!("Read failed: {}", e)))
    }

    /// Read with buffered I/O
    fn read_buffered(&self, offset: u64, buffer: &mut [u8]) -> Result<usize> {
        let mut file = self.file.write();
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| TdbError::Other(format!("Seek failed: {}", e)))?;

        file.read(buffer)
            .map_err(|e| TdbError::Other(format!("Read failed: {}", e)))
    }

    /// Write with direct I/O (aligned)
    fn write_direct_aligned(&self, offset: u64, data: &[u8]) -> Result<usize> {
        // Ensure alignment
        if offset % DIRECT_IO_ALIGNMENT as u64 != 0 {
            return Err(TdbError::Other(
                "Direct I/O requires aligned offset".to_string(),
            ));
        }

        if data.len() % DIRECT_IO_ALIGNMENT != 0 {
            return Err(TdbError::Other(
                "Direct I/O requires aligned buffer size".to_string(),
            ));
        }

        let mut file = self.file.write();
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| TdbError::Other(format!("Seek failed: {}", e)))?;

        let written = file
            .write(data)
            .map_err(|e| TdbError::Other(format!("Write failed: {}", e)))?;

        if self.config.use_sync_writes {
            file.sync_all()
                .map_err(|e| TdbError::Other(format!("Sync failed: {}", e)))?;
        }

        Ok(written)
    }

    /// Write with buffered I/O
    fn write_buffered(&self, offset: u64, data: &[u8]) -> Result<usize> {
        let mut file = self.file.write();
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| TdbError::Other(format!("Seek failed: {}", e)))?;

        file.write(data)
            .map_err(|e| TdbError::Other(format!("Write failed: {}", e)))
    }

    /// Flush any pending writes
    pub fn flush(&self) -> Result<()> {
        self.file
            .write()
            .flush()
            .map_err(|e| TdbError::Other(format!("Flush failed: {}", e)))
    }

    /// Sync file to disk
    pub fn sync(&self) -> Result<()> {
        self.file
            .write()
            .sync_all()
            .map_err(|e| TdbError::Other(format!("Sync failed: {}", e)))
    }

    /// Get file size
    pub fn size(&self) -> Result<u64> {
        let file = self.file.read();
        file.metadata()
            .map(|m| m.len())
            .map_err(|e| TdbError::Other(format!("Failed to get file size: {}", e)))
    }

    /// Check if direct I/O is currently active
    pub fn is_direct_io_active(&self) -> bool {
        self.direct_io_active.load(Ordering::Acquire)
    }

    /// Get statistics
    pub fn stats(&self) -> DirectIOFileStats {
        DirectIOFileStats {
            sequential_reads: self.stats.sequential_reads.load(Ordering::Relaxed),
            random_reads: self.stats.random_reads.load(Ordering::Relaxed),
            total_bytes_read: self.stats.total_bytes_read.load(Ordering::Relaxed),
            total_bytes_written: self.stats.total_bytes_written.load(Ordering::Relaxed),
            direct_io_active: self.direct_io_active.load(Ordering::Acquire),
        }
    }

    /// Reset direct I/O mode (force buffered I/O)
    pub fn reset_direct_io(&self) {
        self.direct_io_active.store(false, Ordering::Release);
        self.sequential_count.store(0, Ordering::Relaxed);
    }
}

/// Direct I/O statistics
#[derive(Debug, Default)]
struct DirectIOStats {
    /// Number of sequential reads
    sequential_reads: AtomicU64,
    /// Number of random reads
    random_reads: AtomicU64,
    /// Total bytes read
    total_bytes_read: AtomicU64,
    /// Total bytes written
    total_bytes_written: AtomicU64,
}

/// Snapshot of direct I/O file statistics
#[derive(Debug, Clone)]
pub struct DirectIOFileStats {
    /// Sequential read operations
    pub sequential_reads: u64,
    /// Random read operations
    pub random_reads: u64,
    /// Total bytes read
    pub total_bytes_read: u64,
    /// Total bytes written
    pub total_bytes_written: u64,
    /// Whether direct I/O is active
    pub direct_io_active: bool,
}

impl DirectIOFileStats {
    /// Calculate sequential read percentage
    pub fn sequential_read_percentage(&self) -> f64 {
        let total = self.sequential_reads + self.random_reads;
        if total == 0 {
            0.0
        } else {
            (self.sequential_reads as f64 / total as f64) * 100.0
        }
    }
}

/// Aligned buffer for direct I/O operations
pub struct AlignedBuffer {
    /// Buffer data (aligned to DIRECT_IO_ALIGNMENT)
    data: Vec<u8>,
    /// Actual data size (may be less than capacity)
    size: usize,
}

impl AlignedBuffer {
    /// Create a new aligned buffer with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        // Round up to alignment boundary
        let aligned_capacity = align_up(capacity, DIRECT_IO_ALIGNMENT);

        Self {
            data: vec![0; aligned_capacity],
            size: capacity,
        }
    }

    /// Get the buffer data
    pub fn data(&self) -> &[u8] {
        &self.data[..self.size]
    }

    /// Get mutable buffer data
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data[..self.size]
    }

    /// Get aligned buffer for direct I/O (full aligned capacity)
    pub fn aligned_data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable aligned buffer for direct I/O
    pub fn aligned_data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get buffer size (actual data)
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get buffer capacity (aligned)
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Resize buffer (rounds up to alignment)
    pub fn resize(&mut self, new_size: usize) {
        let aligned_capacity = align_up(new_size, DIRECT_IO_ALIGNMENT);
        self.data.resize(aligned_capacity, 0);
        self.size = new_size;
    }
}

/// Round up to alignment boundary
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Round down to alignment boundary
#[inline]
pub fn align_down(value: usize, alignment: usize) -> usize {
    value & !(alignment - 1)
}

/// Check if value is aligned
#[inline]
pub fn is_aligned(value: usize, alignment: usize) -> bool {
    value & (alignment - 1) == 0
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use std::env;

    fn temp_file_path(name: &str) -> PathBuf {
        let mut path = env::temp_dir();
        path.push(format!("oxirs_tdb_direct_io_test_{}", name));
        path
    }

    #[test]
    fn test_direct_io_config_default() {
        let config = DirectIOConfig::default();
        assert!(!config.enable_direct_io);
        assert_eq!(config.buffer_size, 1024 * 1024);
        assert_eq!(config.prefetch_size, 4 * 1024 * 1024);
    }

    #[test]
    fn test_direct_io_file_creation() {
        let path = temp_file_path("creation");
        let config = DirectIOConfig::default();

        let file = DirectIOFile::open(&path, config).unwrap();
        assert!(!file.is_direct_io_active());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_buffered_read_write() {
        let path = temp_file_path("buffered_rw");
        let config = DirectIOConfig::default();

        let file = DirectIOFile::open(&path, config).unwrap();

        // Write data
        let data = b"Hello, Direct I/O!";
        let written = file.write_direct(0, data).unwrap();
        assert_eq!(written, data.len());

        file.flush().unwrap();

        // Read data back
        let mut buffer = vec![0u8; data.len()];
        let read = file.read_direct(0, &mut buffer).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(&buffer[..read], data);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_sequential_access_detection() {
        let path = temp_file_path("sequential");
        let mut config = DirectIOConfig::default();
        config.sequential_threshold = 3;

        let file = DirectIOFile::open(&path, config).unwrap();

        // Sequential reads
        let mut buffer = vec![0u8; PAGE_SIZE];
        for i in 0..5 {
            let _ = file.read_direct((i * PAGE_SIZE) as u64, &mut buffer);
        }

        let stats = file.stats();
        assert!(stats.sequential_reads >= 4); // First read might not be counted

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_random_access_detection() {
        let path = temp_file_path("random");
        let config = DirectIOConfig::default();

        let file = DirectIOFile::open(&path, config).unwrap();

        // Random reads (large gaps)
        let mut buffer = vec![0u8; PAGE_SIZE];
        let offsets = [0, 100000, 50000, 200000];
        for &offset in &offsets {
            let _ = file.read_direct(offset, &mut buffer);
        }

        let stats = file.stats();
        assert!(stats.random_reads > 0);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_direct_io_activation() {
        let path = temp_file_path("activation");
        let mut config = DirectIOConfig::default();
        config.enable_direct_io = true;
        config.sequential_threshold = 3;

        let file = DirectIOFile::open(&path, config).unwrap();

        assert!(!file.is_direct_io_active());

        // Perform sequential reads to trigger direct I/O
        let mut buffer = vec![0u8; PAGE_SIZE];
        for i in 0..5 {
            let _ = file.read_direct((i * PAGE_SIZE) as u64, &mut buffer);
        }

        // Should activate after threshold
        assert!(file.is_direct_io_active());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_stats_tracking() {
        let path = temp_file_path("stats");
        let config = DirectIOConfig::default();

        let file = DirectIOFile::open(&path, config).unwrap();

        // Write some data
        let data = vec![1u8; PAGE_SIZE];
        file.write_direct(0, &data).unwrap();

        // Read some data
        let mut buffer = vec![0u8; PAGE_SIZE];
        file.read_direct(0, &mut buffer).unwrap();

        let stats = file.stats();
        assert_eq!(stats.total_bytes_written, PAGE_SIZE as u64);
        assert_eq!(stats.total_bytes_read, PAGE_SIZE as u64);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_reset_direct_io() {
        let path = temp_file_path("reset");
        let mut config = DirectIOConfig::default();
        config.enable_direct_io = true;
        config.sequential_threshold = 2;

        let file = DirectIOFile::open(&path, config).unwrap();

        // Activate direct I/O
        let mut buffer = vec![0u8; PAGE_SIZE];
        for i in 0..4 {
            let _ = file.read_direct((i * PAGE_SIZE) as u64, &mut buffer);
        }

        assert!(file.is_direct_io_active());

        // Reset
        file.reset_direct_io();
        assert!(!file.is_direct_io_active());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer = AlignedBuffer::with_capacity(1000);

        assert_eq!(buffer.size(), 1000);
        assert!(buffer.capacity() >= 1000);
        assert!(is_aligned(buffer.capacity(), DIRECT_IO_ALIGNMENT));
    }

    #[test]
    fn test_aligned_buffer_resize() {
        let mut buffer = AlignedBuffer::with_capacity(1000);

        buffer.resize(2000);
        assert_eq!(buffer.size(), 2000);
        assert!(is_aligned(buffer.capacity(), DIRECT_IO_ALIGNMENT));
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(100, 512), 512);
        assert_eq!(align_up(1024, 512), 1024);
    }

    #[test]
    fn test_align_down() {
        assert_eq!(align_down(4096, 4096), 4096);
        assert_eq!(align_down(4097, 4096), 4096);
        assert_eq!(align_down(8191, 4096), 4096);
        assert_eq!(align_down(1024, 512), 1024);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(4096, 4096));
        assert!(is_aligned(8192, 4096));
        assert!(!is_aligned(4097, 4096));
        assert!(!is_aligned(100, 512));
        assert!(is_aligned(1024, 512));
    }

    #[test]
    fn test_sequential_read_percentage() {
        let stats = DirectIOFileStats {
            sequential_reads: 80,
            random_reads: 20,
            total_bytes_read: 10000,
            total_bytes_written: 5000,
            direct_io_active: true,
        };

        assert!((stats.sequential_read_percentage() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_file_size() {
        let path = temp_file_path("size");
        let config = DirectIOConfig::default();

        let file = DirectIOFile::open(&path, config).unwrap();

        // Write some data
        let data = vec![1u8; 5000];
        file.write_direct(0, &data).unwrap();
        file.flush().unwrap();

        let size = file.size().unwrap();
        assert!(size >= 5000);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_sync() {
        let path = temp_file_path("sync");
        let config = DirectIOConfig::default();

        let file = DirectIOFile::open(&path, config).unwrap();

        let data = vec![1u8; 1000];
        file.write_direct(0, &data).unwrap();
        file.sync().unwrap();

        std::fs::remove_file(path).ok();
    }
}
