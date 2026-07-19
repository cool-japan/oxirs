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
    /// Head of the on-disk free-page list (`None` when there are no free pages).
    ///
    /// Freed pages form a singly-linked list on disk: each free page stores the
    /// [`PageId`] of the next free page in its first 8 bytes (`0` = end of list,
    /// since page 0 is permanently reserved for the superblock). The head is
    /// persisted in the superblock and restored via [`FileManager::set_free_list_head`]
    /// when the store is reopened, so [`FileManager::allocate_page`] can reuse
    /// freed pages across restarts.
    free_list_head: Arc<RwLock<Option<PageId>>>,
}

/// Sentinel stored in a free page's "next" slot meaning "end of list".
///
/// Page 0 is permanently reserved for the superblock, so it can never be a
/// legitimate free page and is therefore a safe "no next" marker.
const FREE_LIST_NULL: u64 = 0;

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
            .truncate(false)
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
            free_list_head: Arc::new(RwLock::new(None)),
        })
    }

    /// Restore the head of the persisted free-page list (called on open after
    /// the superblock has been read).
    pub fn set_free_list_head(&self, head: Option<PageId>) {
        *self.free_list_head.write() = head;
    }

    /// Current head of the persisted free-page list, to be recorded in the
    /// superblock on `sync()`.
    pub fn free_list_head(&self) -> Option<PageId> {
        *self.free_list_head.read()
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

    /// Allocate a page, reusing a previously freed page if one is available.
    ///
    /// Reused pages come from the persisted free-page list (see
    /// [`FileManager::free_page`]); only when that list is empty is the file
    /// physically extended.
    pub fn allocate_page(&self) -> Result<PageId> {
        // Reuse a freed page first so the file does not grow unbounded.
        {
            let mut head_guard = self.free_list_head.write();
            if let Some(free_id) = *head_guard {
                let next = self.read_free_next(free_id)?;
                *head_guard = next;
                return Ok(free_id);
            }
        }

        // No free pages: physically extend the file.
        let page_id = self.num_pages();
        self.extend_to(page_id + 1)?;
        Ok(page_id)
    }

    /// Return a page to the persisted free-page list so it can be reused by a
    /// later [`FileManager::allocate_page`].
    ///
    /// The caller must ensure the page is no longer referenced by any live
    /// structure and is not currently pinned in the buffer pool: this writes a
    /// free-list node directly to the page on disk.
    pub fn free_page(&self, page_id: PageId) -> Result<()> {
        if page_id >= self.num_pages() {
            return Err(TdbError::PageNotFound(page_id));
        }
        let mut head_guard = self.free_list_head.write();
        let old_head = *head_guard;
        // Chain the current head off the newly freed page, then make it the head.
        self.write_free_next(page_id, old_head)?;
        *head_guard = Some(page_id);
        Ok(())
    }

    /// Read the "next free page" pointer stored in a free page's first 8 bytes.
    fn read_free_next(&self, page_id: PageId) -> Result<Option<PageId>> {
        let page = self.read_page(page_id)?;
        let bytes = page.read_at(0, 8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        let next = u64::from_le_bytes(arr);
        if next == FREE_LIST_NULL {
            Ok(None)
        } else {
            Ok(Some(next))
        }
    }

    /// Write the "next free page" pointer into a free page's first 8 bytes.
    fn write_free_next(&self, page_id: PageId, next: Option<PageId>) -> Result<()> {
        let mut page = Page::new(page_id, PageType::Free);
        let next_raw = next.unwrap_or(FREE_LIST_NULL);
        page.write_at(0, &next_raw.to_le_bytes())?;
        page.update_header();
        self.write_page(&mut page)?;
        Ok(())
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
            // Direct I/O. Persist the page's bytes to the physical device:
            // std::fs::File::flush() is a documented no-op, so an explicit
            // sync_data() is required for the write to be crash-durable.
            let mut file_guard = self.file.write();
            file_guard.seek(SeekFrom::Start(offset))?;
            file_guard.write_all(page_data)?;
            file_guard.sync_data()?;
        }

        page.set_dirty(false);
        Ok(())
    }

    /// Flush all changes to disk, issuing a full fsync so that everything
    /// written so far (including any file-length changes from `extend_to`) is
    /// crash-durable. This is the durability barrier relied on by the buffer
    /// pool's `flush_all()` checkpoint path and by `TdbStore::sync()`.
    pub fn flush(&self) -> Result<()> {
        if self.use_mmap {
            let mmap_guard = self.mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                mmap.flush()?;
            }
        } else {
            let file_guard = self.file.write();
            file_guard.sync_all()?;
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

    #[test]
    fn test_file_manager_free_page_reuse() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();

        let p0 = fm.allocate_page().expect("alloc 0");
        let p1 = fm.allocate_page().expect("alloc 1");
        let p2 = fm.allocate_page().expect("alloc 2");
        assert_eq!((p0, p1, p2), (0, 1, 2));
        assert_eq!(fm.num_pages(), 3);

        // Free p1 then p2: the free list is now [p2 -> p1].
        fm.free_page(p1).expect("free p1");
        fm.free_page(p2).expect("free p2");
        assert_eq!(fm.free_list_head(), Some(p2));

        // Allocation reuses freed pages LIFO without growing the file.
        let a = fm.allocate_page().expect("realloc a");
        let b = fm.allocate_page().expect("realloc b");
        assert_eq!(a, p2);
        assert_eq!(b, p1);
        assert_eq!(fm.num_pages(), 3, "no file growth while free pages exist");
        assert_eq!(fm.free_list_head(), None);

        // The list is exhausted: the next allocation extends the file.
        let c = fm.allocate_page().expect("realloc c");
        assert_eq!(c, 3);
        assert_eq!(fm.num_pages(), 4);
    }

    #[test]
    fn test_file_manager_free_list_head_restore() {
        let temp_file = NamedTempFile::new().unwrap();
        let fm = FileManager::open(temp_file.path(), false).unwrap();
        for _ in 0..4 {
            fm.allocate_page().expect("alloc");
        }
        fm.free_page(2).expect("free 2");

        // Simulate reopen: a fresh handle with the persisted head restored.
        let head = fm.free_list_head();
        let fm2 = FileManager::open(temp_file.path(), false).unwrap();
        fm2.set_free_list_head(head);
        assert_eq!(fm2.allocate_page().expect("reuse across reopen"), 2);
    }

    #[test]
    fn test_file_manager_write_page_is_durable() {
        // A page written via write_page() must survive a reopen (fsynced).
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();
        {
            let fm = FileManager::open(&path, false).unwrap();
            let page_id = fm.allocate_page().unwrap();
            let mut page = Page::new(page_id, PageType::BTreeLeaf);
            page.write_at(0, b"durable").unwrap();
            fm.write_page(&mut page).unwrap();
            fm.flush().unwrap();
        }
        let fm = FileManager::open(&path, false).unwrap();
        let page = fm.read_page(0).unwrap();
        assert_eq!(page.read_at(0, 7).unwrap(), b"durable");
    }
}
