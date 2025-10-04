//! File manager with memory-mapped I/O
//!
//! This module manages disk files using memory-mapped I/O (mmap) for
//! high-performance random access to pages.

use crate::error::{Result, TdbError};
use crate::storage::page::{Page, PageId, PageType, PAGE_SIZE};
use memmap2::{MmapMut, MmapOptions};
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// File manager for memory-mapped file I/O
///
/// Manages a single data file with memory-mapped access for efficient
/// random page reads/writes.
pub struct FileManager {
    /// Path to the data file
    path: PathBuf,
    /// File handle
    file: Arc<RwLock<File>>,
    /// Memory-mapped region (optional, for mmap mode)
    mmap: Arc<RwLock<Option<MmapMut>>>,
    /// Current file size in bytes
    file_size: Arc<RwLock<u64>>,
    /// Whether to use mmap (true) or direct I/O (false)
    use_mmap: bool,
}

impl FileManager {
    /// Open or create a file
    pub fn open<P: AsRef<Path>>(path: P, use_mmap: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open or create file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let file_size = file.metadata()?.len();
        let file = Arc::new(RwLock::new(file));

        // Create mmap if requested and file is non-empty
        let mmap = if use_mmap && file_size > 0 {
            let file_guard = file.read();
            let mmap = unsafe { MmapOptions::new().map_mut(&*file_guard)? };
            Arc::new(RwLock::new(Some(mmap)))
        } else {
            Arc::new(RwLock::new(None))
        };

        Ok(Self {
            path,
            file,
            mmap,
            file_size: Arc::new(RwLock::new(file_size)),
            use_mmap,
        })
    }

    /// Get total number of pages
    pub fn num_pages(&self) -> u64 {
        let size = *self.file_size.read();
        size / PAGE_SIZE as u64
    }

    /// Extend file to accommodate more pages
    pub fn extend_to(&self, num_pages: u64) -> Result<()> {
        let new_size = num_pages * PAGE_SIZE as u64;
        let mut size_guard = self.file_size.write();
        
        if new_size > *size_guard {
            let mut file_guard = self.file.write();
            file_guard.set_len(new_size)?;
            file_guard.flush()?;
            *size_guard = new_size;

            // Recreate mmap with new size
            if self.use_mmap {
                drop(file_guard); // Release write lock
                let file_guard = self.file.read();
                let new_mmap = unsafe { MmapOptions::new().map_mut(&*file_guard)? };
                *self.mmap.write() = Some(new_mmap);
            }
        }

        Ok(())
    }

    /// Allocate a new page (extend file and return page ID)
    pub fn allocate_page(&self) -> Result<PageId> {
        let page_id = self.num_pages();
        self.extend_to(page_id + 1)?;
        Ok(page_id)
    }

    /// Read a page from disk
    pub fn read_page(&self, page_id: PageId) -> Result<Page> {
        if page_id >= self.num_pages() {
            return Err(TdbError::PageNotFound(page_id));
        }

        let offset = page_id * PAGE_SIZE as u64;
        let mut page_data = [0u8; PAGE_SIZE];

        if self.use_mmap {
            // Read from mmap
            let mmap_guard = self.mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                let start = offset as usize;
                let end = start + PAGE_SIZE;
                page_data.copy_from_slice(&mmap[start..end]);
            } else {
                return Err(TdbError::Other("Mmap not initialized".to_string()));
            }
        } else {
            // Direct I/O
            use std::io::Read;
            let mut file_guard = self.file.write();
            file_guard.seek(SeekFrom::Start(offset))?;
            file_guard.read_exact(&mut page_data)?;
        }

        Page::from_bytes(&page_data)
    }

    /// Write a page to disk
    pub fn write_page(&self, page: &mut Page) -> Result<()> {
        let page_id = page.page_id();
        
        if page_id >= self.num_pages() {
            return Err(TdbError::PageNotFound(page_id));
        }

        // Update header before writing
        page.update_header();
        
        let offset = page_id * PAGE_SIZE as u64;
        let page_data = page.raw_data();

        if self.use_mmap {
            // Write to mmap
            let mut mmap_guard = self.mmap.write();
            if let Some(ref mut mmap) = *mmap_guard {
                let start = offset as usize;
                let end = start + PAGE_SIZE;
                mmap[start..end].copy_from_slice(page_data);
                mmap.flush_range(start, PAGE_SIZE)?;
            } else {
                return Err(TdbError::Other("Mmap not initialized".to_string()));
            }
        } else {
            // Direct I/O
            let mut file_guard = self.file.write();
            file_guard.seek(SeekFrom::Start(offset))?;
            file_guard.write_all(page_data)?;
            file_guard.flush()?;
        }

        page.set_dirty(false);
        Ok(())
    }

    /// Flush all changes to disk
    pub fn flush(&self) -> Result<()> {
        if self.use_mmap {
            let mmap_guard = self.mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                mmap.flush()?;
            }
        } else {
            let mut file_guard = self.file.write();
            file_guard.flush()?;
        }
        Ok(())
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get file size in bytes
    pub fn file_size(&self) -> u64 {
        *self.file_size.read()
    }
}

impl std::fmt::Debug for FileManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileManager")
            .field("path", &self.path)
            .field("file_size", &self.file_size())
            .field("num_pages", &self.num_pages())
            .field("use_mmap", &self.use_mmap)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_file_manager_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();
        
        assert_eq!(fm.num_pages(), 0);
        assert_eq!(fm.file_size(), 0);
    }

    #[test]
    fn test_file_manager_extend() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();
        
        fm.extend_to(10).unwrap();
        assert_eq!(fm.num_pages(), 10);
        assert_eq!(fm.file_size(), 10 * PAGE_SIZE as u64);
    }

    #[test]
    fn test_file_manager_allocate() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();
        
        let page_id = fm.allocate_page().unwrap();
        assert_eq!(page_id, 0);
        assert_eq!(fm.num_pages(), 1);
        
        let page_id = fm.allocate_page().unwrap();
        assert_eq!(page_id, 1);
        assert_eq!(fm.num_pages(), 2);
    }

    #[test]
    fn test_file_manager_read_write() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();
        
        // Allocate and write page
        let page_id = fm.allocate_page().unwrap();
        let mut page = Page::new(page_id, PageType::BTreeLeaf);
        page.write_at(0, b"test data").unwrap();
        
        fm.write_page(&mut page).unwrap();
        assert!(!page.is_dirty());
        
        // Read page back
        let read_page = fm.read_page(page_id).unwrap();
        let data = read_page.read_at(0, 9).unwrap();
        assert_eq!(data, b"test data");
    }

    #[test]
    fn test_file_manager_mmap() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), true).unwrap();
        
        // Allocate pages to create file
        fm.allocate_page().unwrap();
        
        // Write with mmap
        let mut page = Page::new(0, PageType::BTreeLeaf);
        page.write_at(0, b"mmap test").unwrap();
        fm.write_page(&mut page).unwrap();
        
        // Read with mmap
        let read_page = fm.read_page(0).unwrap();
        let data = read_page.read_at(0, 9).unwrap();
        assert_eq!(data, b"mmap test");
    }

    #[test]
    fn test_file_manager_flush() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();
        
        let page_id = fm.allocate_page().unwrap();
        let mut page = Page::new(page_id, PageType::BTreeLeaf);
        page.write_at(0, b"flush test").unwrap();
        fm.write_page(&mut page).unwrap();
        
        fm.flush().unwrap();
        
        // Verify data persisted
        let read_page = fm.read_page(page_id).unwrap();
        assert_eq!(read_page.read_at(0, 10).unwrap(), b"flush test");
    }

    #[test]
    fn test_file_manager_invalid_page() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();
        
        let result = fm.read_page(999);
        assert!(result.is_err());
    }
}
