//! Free space allocator using bitmap
//!
//! This module tracks which pages are free and which are in use,
//! enabling efficient page allocation and deallocation.

use crate::error::{Result, TdbError};
use crate::storage::page::PageId;
use parking_lot::RwLock;
use std::collections::VecDeque;

/// Bitmap-based free page allocator
///
/// Tracks free pages using a bitmap where each bit represents one page:
/// - 0 = free
/// - 1 = in use
pub struct Allocator {
    /// Bitmap of allocated pages (1 = in use, 0 = free)
    bitmap: RwLock<Vec<u64>>,
    /// Total number of pages tracked
    total_pages: RwLock<u64>,
    /// Queue of recently freed pages (for reuse)
    free_list: RwLock<VecDeque<PageId>>,
}

const BITS_PER_WORD: u64 = 64;

impl Allocator {
    /// Create a new allocator
    pub fn new() -> Self {
        Self {
            bitmap: RwLock::new(Vec::new()),
            total_pages: RwLock::new(0),
            free_list: RwLock::new(VecDeque::new()),
        }
    }

    /// Create allocator with initial capacity
    pub fn with_capacity(num_pages: u64) -> Self {
        let num_words = ((num_pages + BITS_PER_WORD - 1) / BITS_PER_WORD) as usize;
        Self {
            bitmap: RwLock::new(vec![0; num_words]),
            total_pages: RwLock::new(num_pages),
            free_list: RwLock::new(VecDeque::new()),
        }
    }

    /// Allocate a new page (returns page ID)
    pub fn allocate(&self) -> Result<PageId> {
        // Try to reuse a freed page first
        if let Some(page_id) = self.free_list.write().pop_front() {
            self.mark_allocated(page_id)?;
            return Ok(page_id);
        }

        // Otherwise, allocate a new page
        let mut total_guard = self.total_pages.write();
        let page_id = *total_guard;
        
        // Extend bitmap if needed
        let required_words = ((page_id + 1 + BITS_PER_WORD - 1) / BITS_PER_WORD) as usize;
        let mut bitmap_guard = self.bitmap.write();
        if bitmap_guard.len() < required_words {
            bitmap_guard.resize(required_words, 0);
        }

        // Mark as allocated
        let word_index = (page_id / BITS_PER_WORD) as usize;
        let bit_index = page_id % BITS_PER_WORD;
        bitmap_guard[word_index] |= 1u64 << bit_index;

        *total_guard = page_id + 1;
        Ok(page_id)
    }

    /// Free a page (add to free list)
    pub fn free(&self, page_id: PageId) -> Result<()> {
        // Mark as free in bitmap
        let mut bitmap_guard = self.bitmap.write();
        let word_index = (page_id / BITS_PER_WORD) as usize;
        let bit_index = page_id % BITS_PER_WORD;

        if word_index >= bitmap_guard.len() {
            return Err(TdbError::Other(format!(
                "Invalid page ID {}: out of range",
                page_id
            )));
        }

        bitmap_guard[word_index] &= !(1u64 << bit_index);

        // Add to free list for reuse
        self.free_list.write().push_back(page_id);

        Ok(())
    }

    /// Mark a page as allocated
    pub fn mark_allocated(&self, page_id: PageId) -> Result<()> {
        let mut bitmap_guard = self.bitmap.write();
        let word_index = (page_id / BITS_PER_WORD) as usize;
        let bit_index = page_id % BITS_PER_WORD;

        // Extend bitmap if needed
        let required_words = word_index + 1;
        if bitmap_guard.len() < required_words {
            bitmap_guard.resize(required_words, 0);
        }

        bitmap_guard[word_index] |= 1u64 << bit_index;

        // Update total pages if this is beyond current total
        let mut total_guard = self.total_pages.write();
        if page_id >= *total_guard {
            *total_guard = page_id + 1;
        }

        Ok(())
    }

    /// Check if a page is allocated
    pub fn is_allocated(&self, page_id: PageId) -> bool {
        let bitmap_guard = self.bitmap.read();
        let word_index = (page_id / BITS_PER_WORD) as usize;
        let bit_index = page_id % BITS_PER_WORD;

        if word_index >= bitmap_guard.len() {
            return false;
        }

        (bitmap_guard[word_index] & (1u64 << bit_index)) != 0
    }

    /// Get total number of pages tracked
    pub fn total_pages(&self) -> u64 {
        *self.total_pages.read()
    }

    /// Get number of allocated pages
    pub fn allocated_pages(&self) -> u64 {
        let bitmap_guard = self.bitmap.read();
        bitmap_guard.iter().map(|word| word.count_ones() as u64).sum()
    }

    /// Get number of free pages
    pub fn free_pages(&self) -> u64 {
        self.total_pages() - self.allocated_pages()
    }

    /// Get fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fn fragmentation_ratio(&self) -> f64 {
        let free_list_len = self.free_list.read().len() as u64;
        let total_free = self.free_pages();
        
        if total_free == 0 {
            0.0
        } else {
            free_list_len as f64 / total_free as f64
        }
    }

    /// Compact free list (remove duplicate entries)
    pub fn compact_free_list(&self) {
        let mut free_list_guard = self.free_list.write();
        let mut seen = std::collections::HashSet::new();
        free_list_guard.retain(|&page_id| seen.insert(page_id));
    }
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocator")
            .field("total_pages", &self.total_pages())
            .field("allocated_pages", &self.allocated_pages())
            .field("free_pages", &self.free_pages())
            .field("free_list_len", &self.free_list.read().len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let allocator = Allocator::new();
        assert_eq!(allocator.total_pages(), 0);
        assert_eq!(allocator.allocated_pages(), 0);
        assert_eq!(allocator.free_pages(), 0);
    }

    #[test]
    fn test_allocator_allocate() {
        let allocator = Allocator::new();
        
        let page1 = allocator.allocate().unwrap();
        assert_eq!(page1, 0);
        assert_eq!(allocator.allocated_pages(), 1);
        
        let page2 = allocator.allocate().unwrap();
        assert_eq!(page2, 1);
        assert_eq!(allocator.allocated_pages(), 2);
    }

    #[test]
    fn test_allocator_free_reuse() {
        let allocator = Allocator::new();
        
        let page1 = allocator.allocate().unwrap();
        let page2 = allocator.allocate().unwrap();
        let page3 = allocator.allocate().unwrap();
        
        assert_eq!(allocator.allocated_pages(), 3);
        
        // Free page 1
        allocator.free(page1).unwrap();
        assert_eq!(allocator.allocated_pages(), 2);
        
        // Allocate again should reuse page 1
        let page4 = allocator.allocate().unwrap();
        assert_eq!(page4, page1);
        assert_eq!(allocator.allocated_pages(), 3);
    }

    #[test]
    fn test_allocator_is_allocated() {
        let allocator = Allocator::new();
        
        let page = allocator.allocate().unwrap();
        assert!(allocator.is_allocated(page));
        
        allocator.free(page).unwrap();
        assert!(!allocator.is_allocated(page));
    }

    #[test]
    fn test_allocator_with_capacity() {
        let allocator = Allocator::with_capacity(100);
        assert_eq!(allocator.total_pages(), 100);
        assert_eq!(allocator.allocated_pages(), 0);
    }

    #[test]
    fn test_allocator_mark_allocated() {
        let allocator = Allocator::new();
        
        allocator.mark_allocated(5).unwrap();
        assert!(allocator.is_allocated(5));
        assert_eq!(allocator.allocated_pages(), 1);
    }

    #[test]
    fn test_allocator_fragmentation() {
        let allocator = Allocator::new();
        
        // Allocate several pages
        for _ in 0..10 {
            allocator.allocate().unwrap();
        }
        
        // Free some pages
        allocator.free(2).unwrap();
        allocator.free(5).unwrap();
        allocator.free(8).unwrap();
        
        let frag_ratio = allocator.fragmentation_ratio();
        assert!(frag_ratio > 0.0 && frag_ratio <= 1.0);
    }

    #[test]
    fn test_allocator_compact_free_list() {
        let allocator = Allocator::new();
        
        for _ in 0..5 {
            allocator.allocate().unwrap();
        }
        
        allocator.free(1).unwrap();
        allocator.free(1).unwrap(); // Duplicate
        allocator.free(3).unwrap();
        
        assert_eq!(allocator.free_list.read().len(), 3);
        
        allocator.compact_free_list();
        
        assert_eq!(allocator.free_list.read().len(), 2);
    }

    #[test]
    fn test_allocator_large_scale() {
        let allocator = Allocator::new();
        
        // Allocate 10000 pages
        let mut pages = Vec::new();
        for _ in 0..10000 {
            pages.push(allocator.allocate().unwrap());
        }
        
        assert_eq!(allocator.allocated_pages(), 10000);
        
        // Free every other page
        for i in (0..10000).step_by(2) {
            allocator.free(pages[i]).unwrap();
        }
        
        assert_eq!(allocator.allocated_pages(), 5000);
        assert_eq!(allocator.free_pages(), 5000);
    }
}
