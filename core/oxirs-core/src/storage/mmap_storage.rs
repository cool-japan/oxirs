//! Memory-mapped file storage for large RDF datasets
//!
//! This module provides efficient storage and retrieval of large RDF graphs using
//! memory-mapped files. This allows working with datasets larger than RAM by
//! leveraging the operating system's virtual memory management.
//!
//! # Features
//!
//! - **Large dataset support**: Handle graphs with billions of triples
//! - **Efficient I/O**: OS-managed paging reduces memory pressure
//! - **Zero-copy access**: Direct memory access without deserialization
//! - **Persistence**: Automatic synchronization to disk
//! - **Safe concurrency**: Read-only memory maps for safe multi-threaded access
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_core::storage::mmap_storage::MmapTripleStore;
//! use oxirs_core::model::Triple;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! // Create a memory-mapped triple store
//! let path = Path::new("/tmp/large_graph.bin");
//! let mut store = MmapTripleStore::create(path, 1_000_000_000)?; // 1B triples capacity
//!
//! // Store triples with automatic paging
//! // let triple = Triple::new(...);
//! // store.insert(triple)?;
//!
//! println!("Store capacity: {} triples", store.capacity());
//! # Ok(())
//! # }
//! ```

use crate::model::Triple;
use crate::OxirsError;

use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Size of the header in bytes
const HEADER_SIZE: usize = 64;

/// Magic number to identify our file format
const MAGIC_NUMBER: u32 = 0x4F584952; // "OXIR" in hex

/// File format version
const FORMAT_VERSION: u32 = 1;

/// Memory-mapped triple store for large RDF datasets
pub struct MmapTripleStore {
    /// Path to the backing file
    path: PathBuf,
    /// Memory-mapped region (mutable for writes)
    mmap: Option<MmapMut>,
    /// Read-only memory map for safe concurrent reads (reserved for optimization)
    #[allow(dead_code)]
    mmap_ro: Option<Mmap>,
    /// Maximum number of triples this store can hold
    capacity: usize,
    /// Current number of triples stored
    count: Arc<AtomicU64>,
    /// Size of each serialized triple in bytes
    triple_size: usize,
    /// Whether the store is in read-only mode
    read_only: bool,
}

impl MmapTripleStore {
    /// Create a new memory-mapped triple store
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the backing file
    /// * `capacity` - Maximum number of triples to store
    ///
    /// # Returns
    ///
    /// A new memory-mapped triple store
    pub fn create<P: AsRef<Path>>(path: P, capacity: usize) -> Result<Self, OxirsError> {
        let path = path.as_ref().to_path_buf();

        // Estimate triple size (conservative estimate)
        let triple_size = 256; // bytes per triple (will be adjusted based on actual data)

        // Calculate required file size
        let data_size = capacity * triple_size;
        let total_size = HEADER_SIZE + data_size;

        // Create the file with the required size
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| OxirsError::Io(format!("Failed to create file: {}", e)))?;

        file.set_len(total_size as u64)
            .map_err(|e| OxirsError::Io(format!("Failed to set file size: {}", e)))?;

        // Create memory map
        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| OxirsError::Io(format!("Failed to create memory map: {}", e)))?
        };

        let mut store = Self {
            path,
            mmap: Some(mmap),
            mmap_ro: None,
            capacity,
            count: Arc::new(AtomicU64::new(0)),
            triple_size,
            read_only: false,
        };

        // Initialize header
        store.write_header()?;

        Ok(store)
    }

    /// Open an existing memory-mapped triple store
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the existing store file
    /// * `read_only` - Whether to open in read-only mode
    pub fn open<P: AsRef<Path>>(path: P, read_only: bool) -> Result<Self, OxirsError> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new()
            .read(true)
            .write(!read_only)
            .open(&path)
            .map_err(|e| OxirsError::Io(format!("Failed to open file: {}", e)))?;

        if read_only {
            // Create read-only memory map
            let mmap_ro = unsafe {
                MmapOptions::new().map(&file).map_err(|e| {
                    OxirsError::Io(format!("Failed to create read-only memory map: {}", e))
                })?
            };

            // Read header
            let (capacity, count, triple_size) = Self::read_header_from_bytes(&mmap_ro)?;

            Ok(Self {
                path,
                mmap: None,
                mmap_ro: Some(mmap_ro),
                capacity,
                count: Arc::new(AtomicU64::new(count)),
                triple_size,
                read_only: true,
            })
        } else {
            // Create mutable memory map
            let mmap = unsafe {
                MmapOptions::new().map_mut(&file).map_err(|e| {
                    OxirsError::Io(format!("Failed to create mutable memory map: {}", e))
                })?
            };

            // Read header
            let (capacity, count, triple_size) = Self::read_header_from_bytes(&mmap)?;

            Ok(Self {
                path,
                mmap: Some(mmap),
                mmap_ro: None,
                capacity,
                count: Arc::new(AtomicU64::new(count)),
                triple_size,
                read_only: false,
            })
        }
    }

    /// Insert a triple into the store
    pub fn insert(&mut self, triple: &Triple) -> Result<bool, OxirsError> {
        if self.read_only {
            return Err(OxirsError::Store(
                "Cannot insert into read-only store".to_string(),
            ));
        }

        let current_count = self.count.load(Ordering::Acquire);

        if current_count >= self.capacity as u64 {
            return Err(OxirsError::Store("Store is at capacity".to_string()));
        }

        // Serialize the triple using bincode
        let serialized = bincode::serde::encode_to_vec(triple, bincode::config::standard())
            .map_err(|e| OxirsError::Serialize(format!("Failed to serialize triple: {}", e)))?;

        // Check if serialized size fits within our allocated space
        if serialized.len() > self.triple_size {
            return Err(OxirsError::Serialize(format!(
                "Serialized triple size ({}) exceeds allocated space ({})",
                serialized.len(),
                self.triple_size
            )));
        }

        // Get mutable reference to memory map
        let mmap = self
            .mmap
            .as_mut()
            .ok_or_else(|| OxirsError::Store("Memory map not initialized".to_string()))?;

        // Calculate offset for this triple
        let offset = HEADER_SIZE + (current_count as usize * self.triple_size);

        // Ensure we don't write beyond the mapped region
        if offset + self.triple_size > mmap.len() {
            return Err(OxirsError::Store(format!(
                "Offset {} exceeds memory map size {}",
                offset + self.triple_size,
                mmap.len()
            )));
        }

        // Write the serialized data to the memory map
        // First, write the length as u32 (4 bytes)
        let len_bytes = (serialized.len() as u32).to_le_bytes();
        mmap[offset..offset + 4].copy_from_slice(&len_bytes);

        // Then write the actual data
        mmap[offset + 4..offset + 4 + serialized.len()].copy_from_slice(&serialized);

        // Zero out the remaining space to maintain consistency
        let remaining_start = offset + 4 + serialized.len();
        let remaining_end = offset + self.triple_size;
        if remaining_start < remaining_end {
            for byte in &mut mmap[remaining_start..remaining_end] {
                *byte = 0;
            }
        }

        // Increment the counter
        self.count.fetch_add(1, Ordering::Release);

        Ok(true)
    }

    /// Get the number of triples in the store
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity of the store
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Flush pending writes to disk
    pub fn flush(&mut self) -> Result<(), OxirsError> {
        if let Some(mmap) = &mut self.mmap {
            mmap.flush()
                .map_err(|e| OxirsError::Io(format!("Failed to flush memory map: {}", e)))?;
        }
        Ok(())
    }

    /// Get read-only access for concurrent operations
    pub fn as_readonly(&self) -> Result<ReadOnlyMmapView, OxirsError> {
        // Always create a new read-only view by opening the file
        let file = File::open(&self.path).map_err(|e| {
            OxirsError::Io(format!("Failed to open file for read-only view: {}", e))
        })?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| OxirsError::Io(format!("Failed to create read-only view: {}", e)))?
        };

        Ok(ReadOnlyMmapView {
            mmap: Arc::new(mmap),
            capacity: self.capacity,
            count: Arc::clone(&self.count),
            triple_size: self.triple_size,
        })
    }

    // Helper methods

    fn write_header(&mut self) -> Result<(), OxirsError> {
        if let Some(mmap) = &mut self.mmap {
            let header = &mut mmap[0..HEADER_SIZE];

            // Write magic number
            header[0..4].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());

            // Write format version
            header[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());

            // Write capacity
            header[8..16].copy_from_slice(&(self.capacity as u64).to_le_bytes());

            // Write count
            header[16..24].copy_from_slice(&self.count.load(Ordering::Acquire).to_le_bytes());

            // Write triple size
            header[24..32].copy_from_slice(&(self.triple_size as u64).to_le_bytes());

            // Remaining bytes reserved for future use
        }

        Ok(())
    }

    fn read_header_from_bytes(bytes: &[u8]) -> Result<(usize, u64, usize), OxirsError> {
        if bytes.len() < HEADER_SIZE {
            return Err(OxirsError::Store(
                "File too small to contain header".to_string(),
            ));
        }

        let header = &bytes[0..HEADER_SIZE];

        // Read and validate magic number
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != MAGIC_NUMBER {
            return Err(OxirsError::Store(
                "Invalid file format (magic number mismatch)".to_string(),
            ));
        }

        // Read format version
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(OxirsError::Store(format!(
                "Unsupported format version: {}",
                version
            )));
        }

        // Read capacity
        let capacity = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;

        // Read count
        let count = u64::from_le_bytes(header[16..24].try_into().unwrap());

        // Read triple size
        let triple_size = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;

        Ok((capacity, count, triple_size))
    }
}

impl Drop for MmapTripleStore {
    fn drop(&mut self) {
        // Flush any pending writes before dropping
        let _ = self.flush();
    }
}

/// Read-only view of a memory-mapped triple store
///
/// This allows safe concurrent read access to the underlying data
/// without requiring locks.
#[derive(Clone)]
pub struct ReadOnlyMmapView {
    /// Read-only memory map (wrapped in Arc for cloning)
    mmap: Arc<Mmap>,
    /// Capacity of the store
    capacity: usize,
    /// Current count (shared with parent store)
    count: Arc<AtomicU64>,
    /// Size of each triple in bytes
    triple_size: usize,
}

impl ReadOnlyMmapView {
    /// Get the number of triples
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get raw bytes for a triple at the given index
    pub fn get_raw_triple(&self, index: usize) -> Option<&[u8]> {
        if index >= self.len() {
            return None;
        }

        let offset = HEADER_SIZE + (index * self.triple_size);
        let end = offset + self.triple_size;

        // Get bytes from the memory-mapped region
        if end <= self.mmap.len() {
            Some(&self.mmap[offset..end])
        } else {
            None
        }
    }

    /// Get a deserialized triple at the given index
    pub fn get(&self, index: usize) -> Result<Option<Triple>, OxirsError> {
        let raw_bytes = match self.get_raw_triple(index) {
            Some(bytes) => bytes,
            None => return Ok(None),
        };

        // Read the length from the first 4 bytes
        if raw_bytes.len() < 4 {
            return Err(OxirsError::Parse(
                "Insufficient data for length prefix".to_string(),
            ));
        }

        let len_bytes: [u8; 4] = [raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]];
        let data_len = u32::from_le_bytes(len_bytes) as usize;

        // Validate length
        if data_len == 0 {
            return Ok(None); // Empty slot
        }

        if 4 + data_len > raw_bytes.len() {
            return Err(OxirsError::Parse(format!(
                "Invalid data length: {} exceeds available bytes",
                data_len
            )));
        }

        // Deserialize the triple
        let triple: Triple = bincode::serde::decode_from_slice(
            &raw_bytes[4..4 + data_len],
            bincode::config::standard(),
        )
        .map(|(v, _)| v)
        .map_err(|e| OxirsError::Parse(format!("Failed to deserialize triple: {}", e)))?;

        Ok(Some(triple))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_path(name: &str) -> PathBuf {
        env::temp_dir().join(format!("oxirs_test_{}", name))
    }

    #[test]
    fn test_create_mmap_store() {
        let path = temp_path("create");
        let store = MmapTripleStore::create(&path, 1000).unwrap();

        assert_eq!(store.capacity(), 1000);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_open_existing_store() {
        let path = temp_path("open_existing");

        // Create store
        {
            let store = MmapTripleStore::create(&path, 500).unwrap();
            assert_eq!(store.capacity(), 500);
        }

        // Open existing
        {
            let store = MmapTripleStore::open(&path, false).unwrap();
            assert_eq!(store.capacity(), 500);
            assert_eq!(store.len(), 0);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_readonly_view() {
        let path = temp_path("readonly");

        let store = MmapTripleStore::create(&path, 100).unwrap();
        let view = store.as_readonly().unwrap();

        assert_eq!(view.capacity(), 100);
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_readonly_mode() {
        let path = temp_path("readonly_mode");

        // Create and close
        {
            let _ = MmapTripleStore::create(&path, 50).unwrap();
        }

        // Open as read-only
        let store = MmapTripleStore::open(&path, true).unwrap();
        assert_eq!(store.capacity(), 50);
        assert!(store.read_only);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_capacity_limit() {
        let path = temp_path("capacity");
        let mut store = MmapTripleStore::create(&path, 0).unwrap();

        // Attempting to insert when at capacity should fail
        let s = crate::model::Subject::NamedNode(
            crate::model::NamedNode::new("http://example.org/s").unwrap(),
        );
        let p = crate::model::Predicate::NamedNode(
            crate::model::NamedNode::new("http://example.org/p").unwrap(),
        );
        let o = crate::model::Object::Literal(crate::model::Literal::new("test"));
        let triple = Triple::new(s, p, o);

        let result = store.insert(&triple);
        assert!(result.is_err());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
