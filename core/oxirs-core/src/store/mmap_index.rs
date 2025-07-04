//! Memory-mapped B-tree index implementation
//!
//! This module provides an on-disk B-tree implementation that uses memory mapping
//! for efficient access to large indexes without loading them entirely into memory.

use anyhow::{bail, Context, Result};
use lru::LruCache;
use memmap2::{Mmap, MmapOptions};
use parking_lot::{Mutex, RwLock};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// B-tree node size (4KB)
const NODE_SIZE: usize = 4096;

/// Maximum keys per node (calculated to fit in NODE_SIZE)
const MAX_KEYS: usize = 100;

/// Minimum keys per node (except root)
const MIN_KEYS: usize = MAX_KEYS / 2;

/// Cache size for frequently accessed nodes
const CACHE_SIZE: usize = 1024;

/// Index file header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct IndexHeader {
    magic: [u8; 8],
    version: u32,
    flags: u32,
    root_offset: u64,
    free_offset: u64,
    node_count: u64,
    entry_count: u64,
    height: u32,
    reserved: [u8; 28],
}

impl IndexHeader {
    fn new() -> Self {
        Self {
            magic: *b"OXIRIDX\0",
            version: 1,
            flags: 0,
            root_offset: std::mem::size_of::<IndexHeader>() as u64,
            free_offset: 0,
            node_count: 0,
            entry_count: 0,
            height: 0,
            reserved: [0; 28],
        }
    }

    fn validate(&self) -> Result<()> {
        if self.magic != *b"OXIRIDX\0" {
            bail!("Invalid index magic number");
        }
        if self.version != 1 {
            bail!("Unsupported index version: {}", self.version);
        }
        Ok(())
    }
}

/// Index entry stored in the B-tree
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IndexEntry {
    pub offset: u64,
    pub quad_id: u64,
}

/// B-tree node types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
    Leaf = 0,
    Internal = 1,
}

/// On-disk B-tree node structure
#[repr(C)]
struct DiskNode {
    node_type: NodeType,
    key_count: u16,
    reserved: [u8; 5],
    // Followed by:
    // - Keys: [u8; 48] * key_count (48-byte keys)
    // - For leaf nodes: IndexEntry * key_count
    // - For internal nodes: u64 * (key_count + 1) (child offsets)
}

/// In-memory representation of a B-tree node
#[derive(Debug, Clone)]
struct Node {
    offset: u64,
    node_type: NodeType,
    keys: Vec<String>,
    entries: Vec<IndexEntry>, // For leaf nodes
    children: Vec<u64>,       // For internal nodes
    dirty: bool,
}

impl Node {
    fn new_leaf() -> Self {
        Self {
            offset: 0,
            node_type: NodeType::Leaf,
            keys: Vec::with_capacity(MAX_KEYS),
            entries: Vec::with_capacity(MAX_KEYS),
            children: Vec::new(),
            dirty: true,
        }
    }

    fn new_internal() -> Self {
        Self {
            offset: 0,
            node_type: NodeType::Internal,
            keys: Vec::with_capacity(MAX_KEYS),
            entries: Vec::new(),
            children: Vec::with_capacity(MAX_KEYS + 1),
            dirty: true,
        }
    }

    fn is_full(&self) -> bool {
        self.keys.len() >= MAX_KEYS
    }

    fn is_underflow(&self) -> bool {
        self.keys.len() < MIN_KEYS
    }
}

/// Memory-mapped B-tree index
pub struct MmapIndex {
    path: PathBuf,
    file: Arc<Mutex<File>>,
    header: Arc<RwLock<IndexHeader>>,
    mmap: Arc<RwLock<Option<Mmap>>>,
    cache: Arc<Mutex<LruCache<u64, Node>>>,
    write_lock: Arc<Mutex<()>>,
}

impl MmapIndex {
    /// Create a new index
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open or create index file
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .context("Failed to open index file")?;

        // Initialize or load header
        let file_len = file.metadata()?.len();
        let header = if file_len == 0 {
            // New file, write header and root node
            let header = IndexHeader::new();
            file.write_all(unsafe {
                std::slice::from_raw_parts(
                    &header as *const _ as *const u8,
                    std::mem::size_of::<IndexHeader>(),
                )
            })?;

            // Write empty root node
            let root = Node::new_leaf();
            Self::write_node(&mut file, header.root_offset, &root)?;

            file.flush()?;

            let mut header = header;
            header.node_count = 1;
            header
        } else if file_len >= std::mem::size_of::<IndexHeader>() as u64 {
            // Existing file, read header
            let mut header_bytes = vec![0u8; std::mem::size_of::<IndexHeader>()];
            file.seek(SeekFrom::Start(0))?;
            std::io::Read::read_exact(&mut file, &mut header_bytes)?;
            let header: IndexHeader =
                unsafe { std::ptr::read(header_bytes.as_ptr() as *const IndexHeader) };
            header.validate()?;
            header
        } else {
            bail!("Corrupted index file: invalid size");
        };

        // Create memory map if file is large enough
        let mmap = if file_len > std::mem::size_of::<IndexHeader>() as u64 {
            Some(unsafe { MmapOptions::new().map(&file)? })
        } else {
            None
        };

        // Create cache
        let cache = LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap());

        Ok(Self {
            path,
            file: Arc::new(Mutex::new(file)),
            header: Arc::new(RwLock::new(header)),
            mmap: Arc::new(RwLock::new(mmap)),
            cache: Arc::new(Mutex::new(cache)),
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Insert a key-value pair
    pub fn insert(&self, key: &str, entry: IndexEntry) -> Result<()> {
        let _lock = self.write_lock.lock();

        // Start from root
        let header = self.header.read();
        let root_offset = header.root_offset;
        drop(header);

        // Load root node
        let mut root = self.load_node(root_offset)?;

        // If root is full, split it
        if root.is_full() {
            let mut new_root = Node::new_internal();
            new_root.children.push(root_offset);

            // Split root
            let (median_key, new_node) = self.split_node(&mut root)?;
            new_root.keys.push(median_key);
            new_root.children.push(new_node.offset);

            // Update root offset
            let new_root_offset = self.allocate_node()?;
            new_root.offset = new_root_offset;
            self.save_node(&new_root)?;

            let mut header = self.header.write();
            header.root_offset = new_root_offset;
            header.height += 1;
            drop(header);

            // Continue insertion from new root
            self.insert_non_full(&mut new_root, key, entry)?;
        } else {
            self.insert_non_full(&mut root, key, entry)?;
        }

        // Update header
        let mut header = self.header.write();
        header.entry_count += 1;
        self.save_header(&header)?;

        Ok(())
    }

    /// Insert into a non-full node
    fn insert_non_full(&self, node: &mut Node, key: &str, entry: IndexEntry) -> Result<()> {
        // Find insertion position
        let pos = node
            .keys
            .binary_search_by(|k| k.as_str().cmp(key))
            .unwrap_or_else(|p| p);

        if node.node_type == NodeType::Leaf {
            // Insert into leaf
            node.keys.insert(pos, key.to_string());
            node.entries.insert(pos, entry);
            node.dirty = true;
            self.save_node(node)?;
        } else {
            // Insert into internal node
            let child_offset = node.children[pos];
            let mut child = self.load_node(child_offset)?;

            if child.is_full() {
                // Split child
                let (median_key, new_node) = self.split_node(&mut child)?;

                // Insert median key into parent
                node.keys.insert(pos, median_key.clone());
                node.children.insert(pos + 1, new_node.offset);
                node.dirty = true;
                self.save_node(node)?;

                // Determine which child to insert into
                if key < median_key.as_str() {
                    self.insert_non_full(&mut child, key, entry)?;
                } else {
                    let mut new_child = self.load_node(new_node.offset)?;
                    self.insert_non_full(&mut new_child, key, entry)?;
                }
            } else {
                self.insert_non_full(&mut child, key, entry)?;
            }
        }

        Ok(())
    }

    /// Split a full node
    fn split_node(&self, node: &mut Node) -> Result<(String, Node)> {
        let mid = node.keys.len() / 2;
        let median_key = node.keys[mid].clone();

        let mut new_node = if node.node_type == NodeType::Leaf {
            let mut n = Node::new_leaf();
            n.keys = node.keys.split_off(mid + 1);
            n.entries = node.entries.split_off(mid + 1);
            n
        } else {
            let mut n = Node::new_internal();
            n.keys = node.keys.split_off(mid + 1);
            n.children = node.children.split_off(mid + 1);
            n
        };

        // Remove median from original node (for internal nodes)
        if node.node_type == NodeType::Internal {
            node.keys.pop();
        }

        // Allocate and save new node
        new_node.offset = self.allocate_node()?;
        new_node.dirty = true;
        self.save_node(&new_node)?;

        // Mark original node as dirty
        node.dirty = true;
        self.save_node(node)?;

        Ok((median_key, new_node))
    }

    /// Search for entries with a given key prefix
    pub fn search_prefix(&self, prefix: &str) -> Result<Vec<(String, IndexEntry)>> {
        let header = self.header.read();
        let root_offset = header.root_offset;
        drop(header);

        let mut results = Vec::new();
        self.search_prefix_recursive(root_offset, prefix, &mut results)?;
        Ok(results)
    }

    /// Recursive prefix search
    fn search_prefix_recursive(
        &self,
        node_offset: u64,
        prefix: &str,
        results: &mut Vec<(String, IndexEntry)>,
    ) -> Result<()> {
        let node = self.load_node(node_offset)?;

        if node.node_type == NodeType::Leaf {
            // Search leaf node
            for (i, key) in node.keys.iter().enumerate() {
                if key.starts_with(prefix) {
                    results.push((key.clone(), node.entries[i]));
                } else if key.as_str() > prefix {
                    break;
                }
            }
        } else {
            // Search internal node
            for (i, key) in node.keys.iter().enumerate() {
                if key.as_str() >= prefix {
                    self.search_prefix_recursive(node.children[i], prefix, results)?;
                }
                if (key.starts_with(prefix) || key.as_str() > prefix) && i + 1 < node.children.len()
                {
                    self.search_prefix_recursive(node.children[i + 1], prefix, results)?;
                }
                if !key.starts_with(prefix) && key.as_str() > prefix {
                    break;
                }
            }

            // Check last child if needed
            if let Some(last_key) = node.keys.last() {
                if prefix > last_key.as_str() {
                    if let Some(&last_child) = node.children.last() {
                        self.search_prefix_recursive(last_child, prefix, results)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Load a node from disk or cache
    fn load_node(&self, offset: u64) -> Result<Node> {
        // Check cache first
        {
            let mut cache = self.cache.lock();
            if let Some(node) = cache.get(&offset) {
                return Ok(node.clone());
            }
        }

        // Load from disk
        let node = self.read_node(offset)?;

        // Add to cache
        {
            let mut cache = self.cache.lock();
            cache.put(offset, node.clone());
        }

        Ok(node)
    }

    /// Read a node from disk
    fn read_node(&self, offset: u64) -> Result<Node> {
        let mmap = self.mmap.read();
        let mmap = mmap.as_ref().context("No memory map available")?;

        if offset + NODE_SIZE as u64 > mmap.len() as u64 {
            bail!("Node offset out of bounds");
        }

        // Read node header
        let disk_node = unsafe { &*(mmap.as_ptr().add(offset as usize) as *const DiskNode) };

        let mut node = Node {
            offset,
            node_type: disk_node.node_type,
            keys: Vec::with_capacity(disk_node.key_count as usize),
            entries: Vec::new(),
            children: Vec::new(),
            dirty: false,
        };

        // Read keys
        let key_data = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr()
                    .add(offset as usize + std::mem::size_of::<DiskNode>()),
                48 * disk_node.key_count as usize,
            )
        };

        for i in 0..disk_node.key_count as usize {
            let key_bytes = &key_data[i * 48..(i + 1) * 48];
            let key_len = key_bytes.iter().position(|&b| b == 0).unwrap_or(48);
            let key = std::str::from_utf8(&key_bytes[..key_len])?.to_string();
            node.keys.push(key);
        }

        // Read entries or children
        let data_offset =
            offset as usize + std::mem::size_of::<DiskNode>() + 48 * disk_node.key_count as usize;

        if node.node_type == NodeType::Leaf {
            // Read entries
            node.entries.reserve(disk_node.key_count as usize);
            let entries = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(data_offset) as *const IndexEntry,
                    disk_node.key_count as usize,
                )
            };
            node.entries.extend_from_slice(entries);
        } else {
            // Read children
            node.children.reserve(disk_node.key_count as usize + 1);
            let children = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(data_offset) as *const u64,
                    disk_node.key_count as usize + 1,
                )
            };
            node.children.extend_from_slice(children);
        }

        Ok(node)
    }

    /// Save a node to disk
    fn save_node(&self, node: &Node) -> Result<()> {
        if !node.dirty {
            return Ok(());
        }

        let mut file = self.file.lock();
        Self::write_node(&mut file, node.offset, node)?;

        // Update cache
        let mut cache = self.cache.lock();
        cache.put(node.offset, node.clone());

        Ok(())
    }

    /// Write a node to disk
    fn write_node(file: &mut File, offset: u64, node: &Node) -> Result<()> {
        // Prepare node buffer
        let mut buffer = vec![0u8; NODE_SIZE];

        // Write node header
        let disk_node = DiskNode {
            node_type: node.node_type,
            key_count: node.keys.len() as u16,
            reserved: [0; 5],
        };

        unsafe {
            std::ptr::write(buffer.as_mut_ptr() as *mut DiskNode, disk_node);
        }

        // Write keys
        let key_offset = std::mem::size_of::<DiskNode>();
        for (i, key) in node.keys.iter().enumerate() {
            let key_bytes = key.as_bytes();
            let len = key_bytes.len().min(48);
            buffer[key_offset + i * 48..key_offset + i * 48 + len]
                .copy_from_slice(&key_bytes[..len]);
        }

        // Write entries or children
        let data_offset = key_offset + 48 * node.keys.len();

        if node.node_type == NodeType::Leaf {
            // Write entries
            let entries_bytes = unsafe {
                std::slice::from_raw_parts(
                    node.entries.as_ptr() as *const u8,
                    node.entries.len() * std::mem::size_of::<IndexEntry>(),
                )
            };
            buffer[data_offset..data_offset + entries_bytes.len()].copy_from_slice(entries_bytes);
        } else {
            // Write children
            let children_bytes = unsafe {
                std::slice::from_raw_parts(
                    node.children.as_ptr() as *const u8,
                    node.children.len() * std::mem::size_of::<u64>(),
                )
            };
            buffer[data_offset..data_offset + children_bytes.len()].copy_from_slice(children_bytes);
        }

        // Write to file
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(&buffer)?;

        Ok(())
    }

    /// Allocate a new node
    fn allocate_node(&self) -> Result<u64> {
        let mut header = self.header.write();
        let offset =
            std::mem::size_of::<IndexHeader>() as u64 + header.node_count * NODE_SIZE as u64;
        header.node_count += 1;
        Ok(offset)
    }

    /// Save header to disk
    fn save_header(&self, header: &IndexHeader) -> Result<()> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(unsafe {
            std::slice::from_raw_parts(
                header as *const _ as *const u8,
                std::mem::size_of::<IndexHeader>(),
            )
        })?;
        file.flush()?;

        // Update memory map
        self.update_mmap()?;

        Ok(())
    }

    /// Update memory map after writes
    fn update_mmap(&self) -> Result<()> {
        let file = self.file.lock();
        let _file_len = file.metadata()?.len();

        let mut mmap = self.mmap.write();
        *mmap = Some(unsafe { MmapOptions::new().map(&*file)? });

        Ok(())
    }

    /// Flush all changes to disk
    pub fn flush(&self) -> Result<()> {
        let _lock = self.write_lock.lock();

        // Flush cached nodes
        {
            let cache = self.cache.lock();
            let dirty_nodes: Vec<Node> = cache
                .iter()
                .filter(|(_, node)| node.dirty)
                .map(|(_, node)| node.clone())
                .collect();
            drop(cache);

            for node in dirty_nodes {
                self.save_node(&node)?;
            }
        }

        // Save header
        let header = self.header.read();
        self.save_header(&header)?;

        Ok(())
    }
}

impl Drop for MmapIndex {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_index() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let index = MmapIndex::new(temp_file.path())?;
        index.flush()?;
        Ok(())
    }

    #[test]
    fn test_insert_search() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let index = MmapIndex::new(temp_file.path())?;

        // Insert entries
        for i in 0..100 {
            let key = format!("key{i:04}");
            let entry = IndexEntry {
                offset: i * 100,
                quad_id: i,
            };
            index.insert(&key, entry)?;
        }

        // Search for entries
        let results = index.search_prefix("key00")?;
        assert_eq!(results.len(), 10); // key0000 through key0009

        Ok(())
    }

    #[test]
    fn test_large_index() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let index = MmapIndex::new(temp_file.path())?;

        // Insert many entries to trigger splits
        for i in 0..1000 {
            let key = format!("{i:064x}"); // 64-character hex key
            let entry = IndexEntry {
                offset: i * 32,
                quad_id: i,
            };
            index.insert(&key, entry)?;
        }

        index.flush()?;

        // Verify all entries
        for i in 0..1000 {
            let key = format!("{i:064x}");
            let results = index.search_prefix(&key)?;
            assert!(!results.is_empty());
            assert_eq!(results[0].1.quad_id, i);
        }

        Ok(())
    }
}
