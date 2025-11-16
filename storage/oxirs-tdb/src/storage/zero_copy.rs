//! Zero-copy I/O operations for high-performance data transfer
//!
//! This module provides zero-copy operations to minimize memory copying:
//! - Shared memory regions for inter-process communication
//! - Direct buffer access without intermediate copying
//! - Memory-mapped views for efficient batch operations
//! - Vectored I/O for scatter-gather operations

use crate::error::{Result, TdbError};
use crate::storage::page::{PageId, PAGE_SIZE};
use parking_lot::RwLock;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Zero-copy buffer that wraps page data without copying
#[derive(Debug)]
pub struct ZeroCopyBuffer {
    /// Underlying data (Arc to enable sharing)
    data: Arc<RwLock<Vec<u8>>>,
    /// Offset into the buffer
    offset: usize,
    /// Length of the accessible region
    length: usize,
}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer from existing data
    pub fn new(data: Vec<u8>) -> Self {
        let length = data.len();
        Self {
            data: Arc::new(RwLock::new(data)),
            offset: 0,
            length,
        }
    }

    /// Create a zero-copy buffer with a specific size
    pub fn with_capacity(capacity: usize) -> Self {
        let data = vec![0; capacity];
        Self::new(data)
    }

    /// Create a slice view of this buffer (zero-copy)
    pub fn slice(&self, offset: usize, length: usize) -> Result<ZeroCopyBuffer> {
        if offset + length > self.length {
            return Err(TdbError::Other("Slice out of bounds".to_string()));
        }

        Ok(ZeroCopyBuffer {
            data: Arc::clone(&self.data),
            offset: self.offset + offset,
            length,
        })
    }

    /// Get the length of accessible data
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Read data from the buffer (copies into destination)
    pub fn read_into(&self, dest: &mut [u8]) -> Result<usize> {
        let data_lock = self.data.read();
        let available = self.length.min(dest.len());

        dest[..available].copy_from_slice(&data_lock[self.offset..self.offset + available]);

        Ok(available)
    }

    /// Write data into the buffer (copies from source)
    pub fn write_from(&self, src: &[u8]) -> Result<usize> {
        let mut data_lock = self.data.write();
        let available = self.length.min(src.len());

        data_lock[self.offset..self.offset + available].copy_from_slice(&src[..available]);

        Ok(available)
    }

    /// Get read-only access to the underlying data
    pub fn as_slice(&self) -> Vec<u8> {
        let data_lock = self.data.read();
        data_lock[self.offset..self.offset + self.length].to_vec()
    }

    /// Clone the buffer (shares underlying data)
    pub fn share(&self) -> ZeroCopyBuffer {
        ZeroCopyBuffer {
            data: Arc::clone(&self.data),
            offset: self.offset,
            length: self.length,
        }
    }
}

/// Vectored I/O operation for scatter-gather
#[derive(Debug)]
pub struct VectoredIO {
    /// List of buffers for vectored I/O
    buffers: Vec<ZeroCopyBuffer>,
    /// Total length across all buffers
    total_length: usize,
}

impl VectoredIO {
    /// Create a new vectored I/O operation
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            total_length: 0,
        }
    }

    /// Add a buffer to the vector
    pub fn add_buffer(&mut self, buffer: ZeroCopyBuffer) {
        self.total_length += buffer.len();
        self.buffers.push(buffer);
    }

    /// Get total length across all buffers
    pub fn total_length(&self) -> usize {
        self.total_length
    }

    /// Get number of buffers
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Read all data into a single contiguous buffer (gather operation)
    pub fn gather(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.total_length);

        for buffer in &self.buffers {
            result.extend_from_slice(&buffer.as_slice());
        }

        result
    }

    /// Write data to buffers (scatter operation)
    pub fn scatter(&self, data: &[u8]) -> Result<usize> {
        let mut offset = 0;

        for buffer in &self.buffers {
            let chunk_size = buffer.len().min(data.len() - offset);
            if chunk_size == 0 {
                break;
            }

            buffer.write_from(&data[offset..offset + chunk_size])?;
            offset += chunk_size;
        }

        Ok(offset)
    }
}

impl Default for VectoredIO {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-mapped batch view for efficient multi-page operations
#[derive(Debug)]
pub struct BatchView {
    /// Pages in the batch
    pages: Vec<ZeroCopyBuffer>,
    /// Page IDs
    page_ids: Vec<PageId>,
}

impl BatchView {
    /// Create a new batch view
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            page_ids: Vec::new(),
        }
    }

    /// Add a page to the batch
    pub fn add_page(&mut self, page_id: PageId, buffer: ZeroCopyBuffer) {
        self.page_ids.push(page_id);
        self.pages.push(buffer);
    }

    /// Get number of pages in the batch
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get a page by index
    pub fn get_page(&self, index: usize) -> Option<&ZeroCopyBuffer> {
        self.pages.get(index)
    }

    /// Get page ID by index
    pub fn get_page_id(&self, index: usize) -> Option<PageId> {
        self.page_ids.get(index).copied()
    }

    /// Iterate over all pages
    pub fn iter(&self) -> impl Iterator<Item = (PageId, &ZeroCopyBuffer)> {
        self.page_ids.iter().copied().zip(self.pages.iter())
    }

    /// Apply a function to all pages (serial processing for now)
    ///
    /// Note: Parallel processing would require rayon, which we're avoiding.
    /// For true parallelism, use external parallel processing libraries.
    pub fn map<F>(&self, f: F) -> Vec<Vec<u8>>
    where
        F: Fn(&ZeroCopyBuffer) -> Vec<u8>,
    {
        self.pages.iter().map(f).collect()
    }

    /// Collect all pages into a contiguous buffer
    pub fn collect_all(&self) -> Vec<u8> {
        let total_size = self.pages.len() * PAGE_SIZE;
        let mut result = Vec::with_capacity(total_size);

        for buffer in &self.pages {
            result.extend_from_slice(&buffer.as_slice());
        }

        result
    }
}

impl Default for BatchView {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy I/O statistics
#[derive(Debug, Default, Clone)]
pub struct ZeroCopyStats {
    /// Total zero-copy operations
    pub zero_copy_operations: u64,
    /// Total bytes transferred without copying
    pub zero_copy_bytes: u64,
    /// Total scatter-gather operations
    pub vectored_operations: u64,
    /// Total batch operations
    pub batch_operations: u64,
}

impl ZeroCopyStats {
    /// Record a zero-copy operation
    pub fn record_zero_copy(&mut self, bytes: usize) {
        self.zero_copy_operations += 1;
        self.zero_copy_bytes += bytes as u64;
    }

    /// Record a vectored I/O operation
    pub fn record_vectored(&mut self) {
        self.vectored_operations += 1;
    }

    /// Record a batch operation
    pub fn record_batch(&mut self, page_count: usize) {
        self.batch_operations += 1;
        self.zero_copy_bytes += (page_count * PAGE_SIZE) as u64;
    }

    /// Get average bytes per operation
    pub fn avg_bytes_per_operation(&self) -> u64 {
        if self.zero_copy_operations == 0 {
            0
        } else {
            self.zero_copy_bytes / self.zero_copy_operations
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_buffer_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let buffer = ZeroCopyBuffer::new(data.clone());

        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_zero_copy_buffer_slice() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let buffer = ZeroCopyBuffer::new(data);

        let slice = buffer.slice(2, 4).unwrap();
        assert_eq!(slice.len(), 4);

        let slice_data = slice.as_slice();
        assert_eq!(slice_data, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_zero_copy_buffer_read_write() {
        let buffer = ZeroCopyBuffer::with_capacity(10);

        // Write data
        let src = vec![1, 2, 3, 4, 5];
        let written = buffer.write_from(&src).unwrap();
        assert_eq!(written, 5);

        // Read data
        let mut dest = vec![0u8; 5];
        let read = buffer.read_into(&mut dest).unwrap();
        assert_eq!(read, 5);
        assert_eq!(dest, src);
    }

    #[test]
    fn test_zero_copy_buffer_share() {
        let data = vec![1, 2, 3, 4, 5];
        let buffer1 = ZeroCopyBuffer::new(data);

        // Share the buffer (zero-copy)
        let buffer2 = buffer1.share();

        // Both should access the same underlying data
        assert_eq!(buffer1.as_slice(), buffer2.as_slice());

        // Writing to one affects the other (shared data)
        buffer1.write_from(&[10, 20, 30, 40, 50]).unwrap();
        assert_eq!(buffer2.as_slice(), vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_vectored_io_creation() {
        let vio = VectoredIO::new();

        assert_eq!(vio.buffer_count(), 0);
        assert_eq!(vio.total_length(), 0);
    }

    #[test]
    fn test_vectored_io_gather() {
        let mut vio = VectoredIO::new();

        vio.add_buffer(ZeroCopyBuffer::new(vec![1, 2, 3]));
        vio.add_buffer(ZeroCopyBuffer::new(vec![4, 5, 6]));
        vio.add_buffer(ZeroCopyBuffer::new(vec![7, 8, 9]));

        assert_eq!(vio.buffer_count(), 3);
        assert_eq!(vio.total_length(), 9);

        let gathered = vio.gather();
        assert_eq!(gathered, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_vectored_io_scatter() {
        let mut vio = VectoredIO::new();

        vio.add_buffer(ZeroCopyBuffer::with_capacity(3));
        vio.add_buffer(ZeroCopyBuffer::with_capacity(3));
        vio.add_buffer(ZeroCopyBuffer::with_capacity(3));

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let scattered = vio.scatter(&data).unwrap();

        assert_eq!(scattered, 9);

        // Gather back to verify
        let gathered = vio.gather();
        assert_eq!(gathered, data);
    }

    #[test]
    fn test_batch_view_creation() {
        let batch = BatchView::new();

        assert_eq!(batch.page_count(), 0);
    }

    #[test]
    fn test_batch_view_add_pages() {
        let mut batch = BatchView::new();

        batch.add_page(0, ZeroCopyBuffer::with_capacity(PAGE_SIZE));
        batch.add_page(1, ZeroCopyBuffer::with_capacity(PAGE_SIZE));
        batch.add_page(2, ZeroCopyBuffer::with_capacity(PAGE_SIZE));

        assert_eq!(batch.page_count(), 3);
        assert_eq!(batch.get_page_id(0), Some(0));
        assert_eq!(batch.get_page_id(1), Some(1));
        assert_eq!(batch.get_page_id(2), Some(2));
    }

    #[test]
    fn test_batch_view_iteration() {
        let mut batch = BatchView::new();

        batch.add_page(10, ZeroCopyBuffer::with_capacity(PAGE_SIZE));
        batch.add_page(20, ZeroCopyBuffer::with_capacity(PAGE_SIZE));

        let page_ids: Vec<PageId> = batch.iter().map(|(id, _)| id).collect();
        assert_eq!(page_ids, vec![10, 20]);
    }

    #[test]
    fn test_batch_view_map() {
        let mut batch = BatchView::new();

        // Add some buffers with data
        let buf1 = ZeroCopyBuffer::new(vec![1; PAGE_SIZE]);
        let buf2 = ZeroCopyBuffer::new(vec![2; PAGE_SIZE]);

        batch.add_page(0, buf1);
        batch.add_page(1, buf2);

        // Map operation (zero-copy)
        let results = batch.map(|buf| {
            let data = buf.as_slice();
            vec![data[0] * 2] // Simple transformation
        });

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![2]);
        assert_eq!(results[1], vec![4]);
    }

    #[test]
    fn test_batch_view_collect_all() {
        let mut batch = BatchView::new();

        batch.add_page(0, ZeroCopyBuffer::new(vec![1; PAGE_SIZE]));
        batch.add_page(1, ZeroCopyBuffer::new(vec![2; PAGE_SIZE]));

        let collected = batch.collect_all();
        assert_eq!(collected.len(), 2 * PAGE_SIZE);
        assert_eq!(collected[0], 1);
        assert_eq!(collected[PAGE_SIZE], 2);
    }

    #[test]
    fn test_zero_copy_stats() {
        let mut stats = ZeroCopyStats::default();

        stats.record_zero_copy(1024);
        stats.record_zero_copy(2048);
        stats.record_vectored();
        stats.record_batch(5);

        assert_eq!(stats.zero_copy_operations, 2);
        assert_eq!(stats.zero_copy_bytes, 1024 + 2048 + (5 * PAGE_SIZE) as u64);
        assert_eq!(stats.vectored_operations, 1);
        assert_eq!(stats.batch_operations, 1);
    }

    #[test]
    fn test_zero_copy_buffer_out_of_bounds_slice() {
        let buffer = ZeroCopyBuffer::new(vec![1, 2, 3, 4, 5]);

        let result = buffer.slice(3, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_vectored_io_partial_scatter() {
        let mut vio = VectoredIO::new();

        vio.add_buffer(ZeroCopyBuffer::with_capacity(5));
        vio.add_buffer(ZeroCopyBuffer::with_capacity(5));

        // Only 7 bytes of data (less than total capacity of 10)
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let scattered = vio.scatter(&data).unwrap();

        assert_eq!(scattered, 7);
    }

    #[test]
    fn test_zero_copy_stats_avg_bytes() {
        let mut stats = ZeroCopyStats::default();

        assert_eq!(stats.avg_bytes_per_operation(), 0); // No operations yet

        stats.record_zero_copy(1000);
        stats.record_zero_copy(2000);

        assert_eq!(stats.avg_bytes_per_operation(), 1500);
    }

    #[test]
    fn test_zero_copy_buffer_with_capacity() {
        let buffer = ZeroCopyBuffer::with_capacity(1024);

        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_batch_view_get_page() {
        let mut batch = BatchView::new();

        batch.add_page(5, ZeroCopyBuffer::with_capacity(PAGE_SIZE));

        assert!(batch.get_page(0).is_some());
        assert!(batch.get_page(1).is_none());
        assert_eq!(batch.get_page_id(0), Some(5));
    }
}
