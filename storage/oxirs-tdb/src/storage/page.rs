//! Page management for disk-based storage
//!
//! This module provides fixed-size page abstraction (4KB) for efficient disk I/O.
//! Pages are the fundamental unit of storage in TDB.

use crate::error::{Result, TdbError};
use bincode::{Decode, Encode};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Page size (4KB) - standard OS page size for efficient I/O
pub const PAGE_SIZE: usize = 4096;

/// Page header size (reserved for metadata)
const PAGE_HEADER_SIZE: usize = 32;

/// Usable space in a page (PAGE_SIZE - header)
pub const PAGE_USABLE_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

/// Page identifier (unique within a file)
pub type PageId = u64;

/// Page type indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[repr(u8)]
pub enum PageType {
    /// Free page (not in use)
    Free = 0,
    /// B+Tree internal node
    BTreeInternal = 1,
    /// B+Tree leaf node
    BTreeLeaf = 2,
    /// Dictionary data page
    Dictionary = 3,
    /// String data page
    StringData = 4,
    /// Metadata page
    Metadata = 5,
}

/// Page header (32 bytes)
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PageHeader {
    /// Page ID
    pub page_id: PageId,
    /// Page type
    pub page_type: PageType,
    /// Number of bytes used in this page
    pub used_size: u16,
    /// Checksum for integrity verification (CRC32)
    pub checksum: u32,
    /// Reserved for future use
    _reserved: [u8; 14],
}

impl PageHeader {
    /// Create a new page header
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        Self {
            page_id,
            page_type,
            used_size: 0,
            checksum: 0,
            _reserved: [0; 14],
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, bincode::config::standard())
            .expect("Failed to serialize page header")
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::decode_from_slice(bytes, bincode::config::standard())
            .map(|(header, _)| header)
            .map_err(|e| TdbError::Deserialization(e.to_string()))
    }

    /// Calculate checksum for page data
    pub fn calculate_checksum(data: &[u8]) -> u32 {
        crc32fast::hash(data)
    }
}

/// A single page in the storage system
///
/// Pages are 4KB fixed-size blocks that serve as the fundamental unit
/// of disk I/O. Each page has:
/// - A unique ID within the file
/// - A type indicator (B+Tree node, dictionary, etc.)
/// - Metadata (size, checksum)
/// - Data payload (4064 bytes usable)
pub struct Page {
    /// Page header
    header: PageHeader,
    /// Page data (PAGE_SIZE bytes total)
    data: Box<[u8; PAGE_SIZE]>,
    /// Whether this page has been modified
    dirty: bool,
    /// Number of pins (active references)
    pin_count: AtomicUsize,
}

impl Page {
    /// Create a new empty page
    pub fn new(page_id: PageId, page_type: PageType) -> Self {
        let header = PageHeader::new(page_id, page_type);
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Write header to the first bytes
        let header_bytes = header.to_bytes();
        data[..header_bytes.len()].copy_from_slice(&header_bytes);

        Self {
            header,
            data,
            dirty: false,
            pin_count: AtomicUsize::new(0),
        }
    }

    /// Create page from raw bytes (loaded from disk)
    pub fn from_bytes(bytes: &[u8; PAGE_SIZE]) -> Result<Self> {
        // Parse header
        let header = PageHeader::from_bytes(&bytes[..PAGE_HEADER_SIZE])?;

        let mut data = Box::new([0u8; PAGE_SIZE]);
        data.copy_from_slice(bytes);

        Ok(Self {
            header,
            data,
            dirty: false,
            pin_count: AtomicUsize::new(0),
        })
    }

    /// Get page ID
    pub fn page_id(&self) -> PageId {
        self.header.page_id
    }

    /// Get page type
    pub fn page_type(&self) -> PageType {
        self.header.page_type
    }

    /// Check if page is dirty (modified)
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark page as dirty
    pub fn set_dirty(&mut self, dirty: bool) {
        self.dirty = dirty;
    }

    /// Get pin count
    pub fn pin_count(&self) -> usize {
        self.pin_count.load(Ordering::Acquire)
    }

    /// Pin this page (increment reference count)
    pub fn pin(&self) {
        self.pin_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Unpin this page (decrement reference count)
    pub fn unpin(&self) {
        let prev = self.pin_count.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "Unpin called on unpinned page");
    }

    /// Check if page can be evicted (pin_count == 0)
    pub fn can_evict(&self) -> bool {
        self.pin_count.load(Ordering::Acquire) == 0
    }

    /// Get mutable reference to page data (excluding header)
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.dirty = true;
        &mut self.data[PAGE_HEADER_SIZE..]
    }

    /// Get immutable reference to page data (excluding header)
    pub fn data(&self) -> &[u8] {
        &self.data[PAGE_HEADER_SIZE..]
    }

    /// Get mutable reference to raw page bytes (including header)
    pub fn raw_data_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        self.dirty = true;
        &mut self.data
    }

    /// Get immutable reference to raw page bytes (including header)
    pub fn raw_data(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Update header and write to page
    pub fn update_header(&mut self) {
        // Update checksum
        let data_slice = &self.data[PAGE_HEADER_SIZE..];
        self.header.checksum = PageHeader::calculate_checksum(data_slice);

        // Serialize and write header
        let header_bytes = self.header.to_bytes();
        self.data[..header_bytes.len()].copy_from_slice(&header_bytes);
        self.dirty = true;
    }

    /// Verify page checksum
    pub fn verify_checksum(&self) -> bool {
        let data_slice = &self.data[PAGE_HEADER_SIZE..];
        let calculated = PageHeader::calculate_checksum(data_slice);
        calculated == self.header.checksum
    }

    /// Write data to page at offset
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > PAGE_USABLE_SIZE {
            return Err(TdbError::Other(format!(
                "Write exceeds page boundary: offset={}, len={}, max={}",
                offset,
                data.len(),
                PAGE_USABLE_SIZE
            )));
        }

        let start = PAGE_HEADER_SIZE + offset;
        let end = start + data.len();
        self.data[start..end].copy_from_slice(data);
        self.header.used_size = self.header.used_size.max((offset + data.len()) as u16);
        self.dirty = true;

        Ok(())
    }

    /// Read data from page at offset
    pub fn read_at(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset + len > PAGE_USABLE_SIZE {
            return Err(TdbError::Other(format!(
                "Read exceeds page boundary: offset={}, len={}, max={}",
                offset, len, PAGE_USABLE_SIZE
            )));
        }

        let start = PAGE_HEADER_SIZE + offset;
        let end = start + len;
        Ok(&self.data[start..end])
    }

    /// Get used size (excluding header)
    pub fn used_size(&self) -> usize {
        self.header.used_size as usize
    }

    /// Get remaining free space
    pub fn free_space(&self) -> usize {
        PAGE_USABLE_SIZE - self.used_size()
    }

    /// Clear page data (reset to empty)
    pub fn clear(&mut self) {
        self.data[PAGE_HEADER_SIZE..].fill(0);
        self.header.used_size = 0;
        self.header.checksum = 0;
        self.dirty = true;
    }
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("page_id", &self.header.page_id)
            .field("page_type", &self.header.page_type)
            .field("used_size", &self.header.used_size)
            .field("dirty", &self.dirty)
            .field("pin_count", &self.pin_count.load(Ordering::Relaxed))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_creation() {
        let page = Page::new(1, PageType::BTreeLeaf);
        assert_eq!(page.page_id(), 1);
        assert_eq!(page.page_type(), PageType::BTreeLeaf);
        assert_eq!(page.used_size(), 0);
        assert_eq!(page.free_space(), PAGE_USABLE_SIZE);
        assert!(!page.is_dirty());
    }

    #[test]
    fn test_page_write_read() {
        let mut page = Page::new(1, PageType::BTreeLeaf);
        let data = b"Hello, TDB!";

        page.write_at(0, data).unwrap();
        assert!(page.is_dirty());
        assert_eq!(page.used_size(), data.len());

        let read_data = page.read_at(0, data.len()).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_page_pin_unpin() {
        let page = Page::new(1, PageType::BTreeLeaf);
        assert_eq!(page.pin_count(), 0);
        assert!(page.can_evict());

        page.pin();
        assert_eq!(page.pin_count(), 1);
        assert!(!page.can_evict());

        page.pin();
        assert_eq!(page.pin_count(), 2);

        page.unpin();
        assert_eq!(page.pin_count(), 1);

        page.unpin();
        assert_eq!(page.pin_count(), 0);
        assert!(page.can_evict());
    }

    #[test]
    fn test_page_checksum() {
        let mut page = Page::new(1, PageType::BTreeLeaf);
        page.write_at(0, b"test data").unwrap();
        page.update_header();

        assert!(page.verify_checksum());

        // Corrupt data
        page.data_mut()[0] = 0xFF;
        assert!(!page.verify_checksum());
    }

    #[test]
    fn test_page_serialization() {
        let mut page = Page::new(42, PageType::Dictionary);
        page.write_at(0, b"serialization test").unwrap();
        page.update_header();

        let bytes = page.raw_data();
        let restored = Page::from_bytes(bytes).unwrap();

        assert_eq!(restored.page_id(), 42);
        assert_eq!(restored.page_type(), PageType::Dictionary);
        assert_eq!(restored.used_size(), 18);
    }

    #[test]
    fn test_page_boundary_check() {
        let mut page = Page::new(1, PageType::BTreeLeaf);
        let large_data = vec![0u8; PAGE_USABLE_SIZE + 1];

        let result = page.write_at(0, &large_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_page_clear() {
        let mut page = Page::new(1, PageType::BTreeLeaf);
        page.write_at(0, b"some data").unwrap();
        assert!(page.used_size() > 0);

        page.clear();
        assert_eq!(page.used_size(), 0);
        assert_eq!(page.free_space(), PAGE_USABLE_SIZE);
        assert!(page.is_dirty());
    }
}
