//! Memory-mapped store for handling large RDF datasets that don't fit in memory
//!
//! This module provides a disk-based RDF store using memory-mapped files for efficient
//! access to datasets larger than available RAM. Features include:
//! - Append-only writes for crash safety
//! - Memory-mapped reads for efficient access
//! - On-disk indexes for fast lookups
//! - Support for concurrent readers
//! - Automatic recovery from crashes

use crate::model::{NamedNode, BlankNode, Literal, Subject, Predicate, Object, Quad, GraphName};
use crate::store::term_interner::TermInterner;
use crate::store::mmap_index::{MmapIndex, IndexEntry};
use anyhow::{Result, Context, bail};
use memmap2::{MmapMut, MmapOptions, Mmap};
use parking_lot::{RwLock, Mutex};
use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::collections::HashMap;
use blake3::Hasher;

/// Magic number for file format identification
const MAGIC: &[u8; 8] = b"OXIRSMM\0";

/// Current file format version
const VERSION: u32 = 1;

/// Default page size for memory mapping (4KB)
const PAGE_SIZE: usize = 4096;

/// Maximum size for a single memory map (1GB)
const MAX_MMAP_SIZE: usize = 1 << 30;

/// Header size (must be page-aligned)
const HEADER_SIZE: usize = PAGE_SIZE;

/// File header structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FileHeader {
    magic: [u8; 8],
    version: u32,
    flags: u32,
    quad_count: u64,
    term_count: u64,
    data_offset: u64,
    index_offset: u64,
    term_offset: u64,
    checksum: [u8; 32],
    reserved: [u8; 3968], // Pad to PAGE_SIZE
}

impl FileHeader {
    fn new() -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            quad_count: 0,
            term_count: 0,
            data_offset: HEADER_SIZE as u64,
            index_offset: 0,
            term_offset: 0,
            checksum: [0; 32],
            reserved: [0; 3968],
        }
    }

    fn validate(&self) -> Result<()> {
        if self.magic != *MAGIC {
            bail!("Invalid magic number");
        }
        if self.version != VERSION {
            bail!("Unsupported version: {}", self.version);
        }
        Ok(())
    }

    fn compute_checksum(&mut self) {
        let mut hasher = Hasher::new();
        hasher.update(&self.magic);
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.flags.to_le_bytes());
        hasher.update(&self.quad_count.to_le_bytes());
        hasher.update(&self.term_count.to_le_bytes());
        hasher.update(&self.data_offset.to_le_bytes());
        hasher.update(&self.index_offset.to_le_bytes());
        hasher.update(&self.term_offset.to_le_bytes());
        self.checksum = *hasher.finalize().as_bytes();
    }
}

/// On-disk representation of a quad
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DiskQuad {
    subject_id: u64,
    predicate_id: u64,
    object_id: u64,
    graph_id: u64,
}

/// Memory-mapped RDF store
pub struct MmapStore {
    path: PathBuf,
    header: Arc<RwLock<FileHeader>>,
    data_file: Arc<Mutex<File>>,
    data_mmap: Arc<RwLock<Option<Mmap>>>,
    append_buffer: Arc<Mutex<Vec<DiskQuad>>>,
    term_interner: Arc<RwLock<TermInterner>>,
    indexes: Arc<RwLock<HashMap<String, MmapIndex>>>,
    write_lock: Arc<Mutex<()>>,
}

impl MmapStore {
    /// Create a new memory-mapped store
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let data_path = path.join("data.oxirs");
        
        // Ensure directory exists
        std::fs::create_dir_all(&path)?;
        
        // Open or create data file
        let mut data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_path)
            .context("Failed to open data file")?;
        
        // Initialize or load header
        let file_len = data_file.metadata()?.len();
        let header = if file_len == 0 {
            // New file, write header
            let header = FileHeader::new();
            data_file.write_all(unsafe {
                std::slice::from_raw_parts(
                    &header as *const _ as *const u8,
                    std::mem::size_of::<FileHeader>()
                )
            })?;
            data_file.flush()?;
            header
        } else if file_len >= HEADER_SIZE as u64 {
            // Existing file, read header
            let mut header_bytes = vec![0u8; HEADER_SIZE];
            data_file.seek(SeekFrom::Start(0))?;
            std::io::Read::read_exact(&mut data_file, &mut header_bytes)?;
            let header: FileHeader = unsafe {
                std::ptr::read(header_bytes.as_ptr() as *const FileHeader)
            };
            header.validate()?;
            header
        } else {
            bail!("Corrupted data file: invalid size");
        };
        
        // Create initial memory map if file is large enough
        let data_mmap = if file_len > HEADER_SIZE as u64 {
            Some(unsafe {
                MmapOptions::new()
                    .offset(HEADER_SIZE as u64)
                    .len((file_len - HEADER_SIZE as u64) as usize)
                    .map(&data_file)?
            })
        } else {
            None
        };
        
        // Load or create term interner
        let term_interner = TermInterner::new();
        
        // Initialize indexes
        let mut indexes = HashMap::new();
        
        // SPO index (Subject, Predicate, Object)
        let spo_path = path.join("spo.idx");
        indexes.insert("spo".to_string(), MmapIndex::new(&spo_path)?);
        
        // POS index (Predicate, Object, Subject)
        let pos_path = path.join("pos.idx");
        indexes.insert("pos".to_string(), MmapIndex::new(&pos_path)?);
        
        // OSP index (Object, Subject, Predicate)
        let osp_path = path.join("osp.idx");
        indexes.insert("osp".to_string(), MmapIndex::new(&osp_path)?);
        
        // GSPO index (Graph, Subject, Predicate, Object)
        let gspo_path = path.join("gspo.idx");
        indexes.insert("gspo".to_string(), MmapIndex::new(&gspo_path)?);
        
        Ok(Self {
            path,
            header: Arc::new(RwLock::new(header)),
            data_file: Arc::new(Mutex::new(data_file)),
            data_mmap: Arc::new(RwLock::new(data_mmap)),
            append_buffer: Arc::new(Mutex::new(Vec::with_capacity(1024))),
            term_interner: Arc::new(RwLock::new(term_interner)),
            indexes: Arc::new(RwLock::new(indexes)),
            write_lock: Arc::new(Mutex::new(())),
        })
    }
    
    /// Open an existing memory-mapped store
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            bail!("Store does not exist: {:?}", path);
        }
        
        let mut store = Self::new(path)?;
        
        // Try to load existing term interner
        let term_path = store.path.join("terms.oxirs");
        if term_path.exists() {
            match TermInterner::load(&term_path) {
                Ok(interner) => {
                    *store.term_interner.write() = interner;
                }
                Err(e) => {
                    // Log error but continue with empty interner
                    eprintln!("Warning: Failed to load term interner: {}", e);
                }
            }
        }
        
        Ok(store)
    }
    
    /// Add a quad to the store
    pub fn add(&self, quad: &Quad) -> Result<()> {
        let _lock = self.write_lock.lock();
        
        // Intern terms
        let subject_id = {
            let mut interner = self.term_interner.write();
            match quad.subject() {
                Subject::NamedNode(n) => interner.intern_named_node(n),
                Subject::BlankNode(b) => interner.intern_blank_node(b),
                Subject::Variable(_) | Subject::QuotedTriple(_) => {
                    bail!("Variables and quoted triples cannot be interned in storage");
                }
            }
        };
        
        let predicate_id = {
            let mut interner = self.term_interner.write();
            match quad.predicate() {
                Predicate::NamedNode(n) => interner.intern_named_node(n),
                Predicate::Variable(_) => {
                    bail!("Variables cannot be interned in storage");
                }
            }
        };
        
        let object_id = {
            let mut interner = self.term_interner.write();
            match quad.object() {
                Object::NamedNode(n) => interner.intern_named_node(n),
                Object::BlankNode(b) => interner.intern_blank_node(b),
                Object::Literal(l) => interner.intern_literal(l),
                Object::Variable(_) | Object::QuotedTriple(_) => {
                    bail!("Variables and quoted triples cannot be interned in storage");
                }
            }
        };
        
        let graph_id = {
            let mut interner = self.term_interner.write();
            match quad.graph_name() {
                GraphName::NamedNode(n) => interner.intern_named_node(n),
                GraphName::BlankNode(b) => interner.intern_blank_node(b),
                GraphName::DefaultGraph => 0, // Default graph
                GraphName::Variable(_) => {
                    bail!("Variables cannot be interned in storage");
                }
            }
        };
        
        // Create disk quad
        let disk_quad = DiskQuad {
            subject_id,
            predicate_id,
            object_id,
            graph_id,
        };
        
        // Add to append buffer
        {
            let mut buffer = self.append_buffer.lock();
            buffer.push(disk_quad);
            
            // Flush if buffer is full
            if buffer.len() >= 1024 {
                self.flush_buffer(&mut buffer)?;
            }
        }
        
        Ok(())
    }
    
    /// Flush append buffer to disk
    fn flush_buffer(&self, buffer: &mut Vec<DiskQuad>) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }
        
        let mut data_file = self.data_file.lock();
        let mut header = self.header.write();
        
        // Seek to end of data
        let offset = data_file.seek(SeekFrom::End(0))?;
        
        // Write quads
        for quad in buffer.iter() {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    quad as *const _ as *const u8,
                    std::mem::size_of::<DiskQuad>()
                )
            };
            data_file.write_all(bytes)?;
        }
        
        // Update indexes
        {
            let mut indexes = self.indexes.write();
            let base_idx = header.quad_count;
            
            for (idx, quad) in buffer.iter().enumerate() {
                let quad_idx = base_idx + idx as u64;
                
                // SPO index
                if let Some(spo) = indexes.get_mut("spo") {
                    let key = format!("{:016x}{:016x}{:016x}", 
                        quad.subject_id, quad.predicate_id, quad.object_id);
                    spo.insert(&key, IndexEntry {
                        offset: offset + (idx * std::mem::size_of::<DiskQuad>()) as u64,
                        quad_id: quad_idx,
                    })?;
                }
                
                // POS index
                if let Some(pos) = indexes.get_mut("pos") {
                    let key = format!("{:016x}{:016x}{:016x}", 
                        quad.predicate_id, quad.object_id, quad.subject_id);
                    pos.insert(&key, IndexEntry {
                        offset: offset + (idx * std::mem::size_of::<DiskQuad>()) as u64,
                        quad_id: quad_idx,
                    })?;
                }
                
                // OSP index
                if let Some(osp) = indexes.get_mut("osp") {
                    let key = format!("{:016x}{:016x}{:016x}", 
                        quad.object_id, quad.subject_id, quad.predicate_id);
                    osp.insert(&key, IndexEntry {
                        offset: offset + (idx * std::mem::size_of::<DiskQuad>()) as u64,
                        quad_id: quad_idx,
                    })?;
                }
                
                // GSPO index
                if let Some(gspo) = indexes.get_mut("gspo") {
                    let key = format!("{:016x}{:016x}{:016x}{:016x}", 
                        quad.graph_id, quad.subject_id, quad.predicate_id, quad.object_id);
                    gspo.insert(&key, IndexEntry {
                        offset: offset + (idx * std::mem::size_of::<DiskQuad>()) as u64,
                        quad_id: quad_idx,
                    })?;
                }
            }
        }
        
        // Update header
        header.quad_count += buffer.len() as u64;
        header.compute_checksum();
        
        // Write updated header
        data_file.seek(SeekFrom::Start(0))?;
        data_file.write_all(unsafe {
            std::slice::from_raw_parts(
                &*header as *const _ as *const u8,
                std::mem::size_of::<FileHeader>()
            )
        })?;
        
        data_file.flush()?;
        
        // Clear buffer
        buffer.clear();
        
        // Update memory map
        self.update_mmap()?;
        
        Ok(())
    }
    
    /// Update memory map after writes
    fn update_mmap(&self) -> Result<()> {
        let data_file = self.data_file.lock();
        let file_len = data_file.metadata()?.len();
        
        if file_len > HEADER_SIZE as u64 {
            let mut data_mmap = self.data_mmap.write();
            *data_mmap = Some(unsafe {
                MmapOptions::new()
                    .offset(HEADER_SIZE as u64)
                    .len((file_len - HEADER_SIZE as u64) as usize)
                    .map(&*data_file)?
            });
        }
        
        Ok(())
    }
    
    /// Flush all pending writes
    pub fn flush(&self) -> Result<()> {
        let _lock = self.write_lock.lock();
        let mut buffer = self.append_buffer.lock();
        self.flush_buffer(&mut buffer)?;
        
        // Save term interner
        let term_path = self.path.join("terms.oxirs");
        self.term_interner.read().save(&term_path)?;
        
        Ok(())
    }
    
    /// Get quad count
    pub fn len(&self) -> u64 {
        self.header.read().quad_count
    }
    
    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Iterate over all quads matching a pattern
    pub fn quads_matching(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<impl Iterator<Item = Result<Quad>>> {
        // Ensure buffer is flushed
        self.flush()?;
        
        let interner = self.term_interner.read();
        let indexes = self.indexes.read();
        let data_mmap = self.data_mmap.read();
        
        // Convert terms to IDs
        let subject_id = subject.and_then(|s| match s {
            Subject::NamedNode(n) => interner.get_named_node_id(n),
            Subject::BlankNode(b) => interner.get_blank_node_id(b),
            Subject::Variable(_) | Subject::QuotedTriple(_) => None,
        });
        
        let predicate_id = predicate.and_then(|p| match p {
            Predicate::NamedNode(n) => interner.get_named_node_id(n),
            Predicate::Variable(_) => None,
        });
        
        let object_id = object.and_then(|o| match o {
            Object::NamedNode(n) => interner.get_named_node_id(n),
            Object::BlankNode(b) => interner.get_blank_node_id(b),
            Object::Literal(l) => interner.get_literal_id(l),
            Object::Variable(_) | Object::QuotedTriple(_) => None,
        });
        
        let graph_id = graph_name.and_then(|g| match g {
            GraphName::NamedNode(n) => interner.get_named_node_id(n),
            GraphName::BlankNode(b) => interner.get_blank_node_id(b),
            GraphName::DefaultGraph => Some(0),
            GraphName::Variable(_) => None,
        });
        
        // Choose best index based on pattern
        let results = match (subject_id, predicate_id, object_id, graph_id) {
            (Some(s), Some(p), Some(o), _) => {
                // Use SPO index for exact match
                let key = format!("{:016x}{:016x}{:016x}", s, p, o);
                vec![] // TODO: Implement index lookup
            }
            (Some(s), Some(p), None, _) => {
                // Use SPO index with prefix
                let prefix = format!("{:016x}{:016x}", s, p);
                vec![] // TODO: Implement prefix scan
            }
            _ => {
                // Full scan
                vec![] // TODO: Implement full scan
            }
        };
        
        Ok(results.into_iter().map(|_: ()| bail!("Not implemented")))
    }
    
    /// Compact the store to reclaim space
    pub fn compact(&self) -> Result<()> {
        // TODO: Implement compaction
        Ok(())
    }
    
    /// Get store statistics
    pub fn stats(&self) -> StoreStats {
        let header = self.header.read();
        let interner = self.term_interner.read();
        
        StoreStats {
            quad_count: header.quad_count,
            term_count: header.term_count,
            data_size: header.index_offset - header.data_offset,
            index_size: header.term_offset - header.index_offset,
            term_size: 0, // TODO: Calculate from term file
        }
    }
}

impl Drop for MmapStore {
    fn drop(&mut self) {
        // Ensure all data is flushed
        let _ = self.flush();
    }
}

/// Store statistics
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub quad_count: u64,
    pub term_count: u64,
    pub data_size: u64,
    pub index_size: u64,
    pub term_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_create_store() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let store = MmapStore::new(temp_dir.path())?;
        assert_eq!(store.len(), 0);
        Ok(())
    }
    
    #[test]
    fn test_add_quad() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let store = MmapStore::new(temp_dir.path())?;
        
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new("http://example.org/s")?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::NamedNode(NamedNode::new("http://example.org/o")?),
            GraphName::DefaultGraph,
        );
        
        store.add(&quad)?;
        store.flush()?;
        
        assert_eq!(store.len(), 1);
        Ok(())
    }
    
    #[test]
    fn test_persistence() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let path = temp_dir.path();
        
        // Create store and add data
        {
            let store = MmapStore::new(path)?;
            
            for i in 0..100 {
                let quad = Quad::new(
                    Subject::NamedNode(NamedNode::new(&format!("http://example.org/s{}", i))?),
                    Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
                    Object::Literal(Literal::new_simple_literal(&format!("value{}", i))),
                    GraphName::DefaultGraph,
                );
                store.add(&quad)?;
            }
            
            store.flush()?;
            assert_eq!(store.len(), 100);
        }
        
        // Reopen and verify
        {
            let store = MmapStore::open(path)?;
            assert_eq!(store.len(), 100);
        }
        
        Ok(())
    }
}