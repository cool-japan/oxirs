//! Memory-mapped store for handling large RDF datasets that don't fit in memory
//!
//! This module provides a disk-based RDF store using memory-mapped files for efficient
//! access to datasets larger than available RAM. Features include:
//! - Append-only writes for crash safety
//! - Memory-mapped reads for efficient access
//! - On-disk indexes for fast lookups
//! - Support for concurrent readers
//! - Automatic recovery from crashes
//! - Full and incremental backup support

mod backup;
mod types;

pub use types::{AccessStats, BackupConfig, BackupMetadata, PerformanceStats, StoreStats};

use crate::model::{GraphName, Object, Predicate, Quad, Subject};
use crate::store::mmap_index::{IndexEntry, MmapIndex};
use crate::store::term_interner::TermInterner;
use anyhow::{bail, Context, Result};
use memmap2::{Mmap, MmapOptions};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use types::{DiskQuad, FileHeader, HEADER_SIZE};

/// Memory-mapped RDF store with performance optimizations
pub struct MmapStore {
    pub(super) path: PathBuf,
    pub(super) header: Arc<RwLock<FileHeader>>,
    pub(super) data_file: Arc<Mutex<File>>,
    pub(super) data_mmap: Arc<RwLock<Option<Mmap>>>,
    pub(super) append_buffer: Arc<Mutex<Vec<DiskQuad>>>,
    pub(super) term_interner: Arc<RwLock<TermInterner>>,
    pub(super) indexes: Arc<RwLock<HashMap<String, MmapIndex>>>,
    pub(super) write_lock: Arc<Mutex<()>>,

    // Performance optimization fields
    pub(super) term_cache: Arc<RwLock<HashMap<String, u64>>>,
    pub(super) batch_buffer: Arc<Mutex<Vec<Quad>>>,
    pub(super) performance_stats: Arc<Mutex<PerformanceStats>>,

    // Deletion tracking for compaction
    pub(super) deleted_quads: Arc<RwLock<HashSet<u64>>>,

    // Access statistics for query optimization
    pub(super) access_stats: Arc<Mutex<AccessStats>>,
    pub(super) subject_access_counts: Arc<RwLock<HashMap<u64, u64>>>,
    pub(super) predicate_access_counts: Arc<RwLock<HashMap<u64, u64>>>,

    // Backup tracking
    pub(super) last_backup_offset: Arc<RwLock<u64>>,
    pub(super) backup_history: Arc<RwLock<Vec<BackupMetadata>>>,
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
            .truncate(false)
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
                    std::mem::size_of::<FileHeader>(),
                )
            })?;
            data_file.flush()?;
            header
        } else if file_len >= HEADER_SIZE as u64 {
            // Existing file, read header
            let mut header_bytes = vec![0u8; HEADER_SIZE];
            data_file.seek(SeekFrom::Start(0))?;
            std::io::Read::read_exact(&mut data_file, &mut header_bytes)?;
            let header: FileHeader =
                unsafe { std::ptr::read(header_bytes.as_ptr() as *const FileHeader) };
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

        // Initialize indexes lazily for better performance
        let indexes = HashMap::new();

        Ok(Self {
            path,
            header: Arc::new(RwLock::new(header)),
            data_file: Arc::new(Mutex::new(data_file)),
            data_mmap: Arc::new(RwLock::new(data_mmap)),
            append_buffer: Arc::new(Mutex::new(Vec::with_capacity(8192))),
            term_interner: Arc::new(RwLock::new(term_interner)),
            indexes: Arc::new(RwLock::new(indexes)),
            write_lock: Arc::new(Mutex::new(())),

            // Initialize performance optimization fields
            term_cache: Arc::new(RwLock::new(HashMap::with_capacity(10000))),
            batch_buffer: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            performance_stats: Arc::new(Mutex::new(PerformanceStats::default())),

            // Initialize deletion tracking
            deleted_quads: Arc::new(RwLock::new(HashSet::new())),

            // Initialize access statistics
            access_stats: Arc::new(Mutex::new(AccessStats::default())),
            subject_access_counts: Arc::new(RwLock::new(HashMap::new())),
            predicate_access_counts: Arc::new(RwLock::new(HashMap::new())),

            // Initialize backup tracking
            last_backup_offset: Arc::new(RwLock::new(0)),
            backup_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Open an existing memory-mapped store
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            bail!("Store does not exist: {:?}", path);
        }

        let store = Self::new(path)?;

        // Try to load existing term interner
        let term_path = store.path.join("terms.oxirs");
        if term_path.exists() {
            match TermInterner::load(&term_path) {
                Ok(interner) => {
                    *store.term_interner.write() = interner;
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load term interner: {e}");
                }
            }
        }

        Ok(store)
    }

    /// Add a quad to the store with optimized batching and caching
    pub fn add(&self, quad: &Quad) -> Result<()> {
        // Update performance stats
        {
            let mut stats = self.performance_stats.lock();
            stats.add_operations += 1;
        }

        // Use adaptive batch buffer for optimal performance
        {
            let mut batch_buffer = self.batch_buffer.lock();
            batch_buffer.push(quad.clone());

            // Calculate optimal batch size based on performance statistics
            let optimal_batch_size = {
                let stats = self.performance_stats.lock();
                if stats.batch_operations > 10 {
                    let avg_batch_size = stats.average_batch_size as usize;
                    if avg_batch_size > 200 {
                        std::cmp::min(1000, avg_batch_size + 50)
                    } else {
                        std::cmp::max(50, avg_batch_size)
                    }
                } else {
                    100
                }
            };

            // Process batch when buffer reaches optimal size
            if batch_buffer.len() >= optimal_batch_size {
                let quads_to_process: Vec<Quad> = batch_buffer.drain(..).collect();
                drop(batch_buffer);
                self.add_batch_optimized(&quads_to_process)?;
            }
        }

        Ok(())
    }

    /// Optimized add with term caching to reduce interner lock contention
    fn _add_single_optimized(&self, quad: &Quad) -> Result<()> {
        let start = std::time::Instant::now();

        let (subject_id, predicate_id, object_id, graph_id) = self._get_or_intern_terms(quad)?;

        let disk_quad = DiskQuad {
            subject_id,
            predicate_id,
            object_id,
            graph_id,
        };

        {
            let mut buffer = self.append_buffer.lock();
            buffer.push(disk_quad);

            let buffer_size = if buffer.capacity() > 16384 {
                16384
            } else {
                8192
            };
            if buffer.len() >= buffer_size {
                self.flush_buffer(&mut buffer)?;
            }
        }

        {
            let mut stats = self.performance_stats.lock();
            stats.total_flush_time_ms += start.elapsed().as_millis() as u64;
        }

        Ok(())
    }

    /// Get term IDs with caching optimization
    fn _get_or_intern_terms(&self, quad: &Quad) -> Result<(u64, u64, u64, u64)> {
        let subject_key = self.term_to_cache_key(quad.subject());
        let predicate_key = self.term_to_cache_key_predicate(quad.predicate());
        let object_key = self.term_to_cache_key_object(quad.object());
        let graph_key = self.term_to_cache_key_graph(quad.graph_name());

        let (cached_subject, cached_predicate, cached_object, cached_graph) = {
            let cache = self.term_cache.read();
            (
                cache.get(&subject_key).copied(),
                cache.get(&predicate_key).copied(),
                cache.get(&object_key).copied(),
                cache.get(&graph_key).copied(),
            )
        };

        let cache_hits = [
            cached_subject,
            cached_predicate,
            cached_object,
            cached_graph,
        ]
        .iter()
        .filter(|id| id.is_some())
        .count();

        {
            let mut stats = self.performance_stats.lock();
            stats.cache_hits += cache_hits as u64;
            stats.cache_misses += (4 - cache_hits) as u64;
        }

        let subject_id = if let Some(id) = cached_subject {
            id
        } else {
            let interner = self.term_interner.write();
            let id = match quad.subject() {
                Subject::NamedNode(n) => interner.intern_named_node(n),
                Subject::BlankNode(b) => interner.intern_blank_node(b),
                Subject::Variable(_) | Subject::QuotedTriple(_) => {
                    bail!("Variables and quoted triples cannot be interned in storage");
                }
            };
            let mut cache = self.term_cache.write();
            cache.insert(subject_key, id);
            id
        };

        let predicate_id = if let Some(id) = cached_predicate {
            id
        } else {
            let interner = self.term_interner.write();
            let id = match quad.predicate() {
                Predicate::NamedNode(n) => interner.intern_named_node(n),
                Predicate::Variable(_) => {
                    bail!("Variables cannot be interned in storage");
                }
            };
            let mut cache = self.term_cache.write();
            cache.insert(predicate_key, id);
            id
        };

        let object_id = if let Some(id) = cached_object {
            id
        } else {
            let interner = self.term_interner.write();
            let id = match quad.object() {
                Object::NamedNode(n) => interner.intern_named_node(n),
                Object::BlankNode(b) => interner.intern_blank_node(b),
                Object::Literal(l) => interner.intern_literal(l),
                Object::Variable(_) | Object::QuotedTriple(_) => {
                    bail!("Variables and quoted triples cannot be interned in storage");
                }
            };
            let mut cache = self.term_cache.write();
            cache.insert(object_key, id);
            id
        };

        let graph_id = if let Some(id) = cached_graph {
            id
        } else {
            let interner = self.term_interner.write();
            let id = match quad.graph_name() {
                GraphName::NamedNode(n) => interner.intern_named_node(n),
                GraphName::BlankNode(b) => interner.intern_blank_node(b),
                GraphName::DefaultGraph => 0,
                GraphName::Variable(_) => {
                    bail!("Variables cannot be interned in storage");
                }
            };
            let mut cache = self.term_cache.write();
            cache.insert(graph_key, id);
            id
        };

        Ok((subject_id, predicate_id, object_id, graph_id))
    }

    fn term_to_cache_key(&self, subject: &Subject) -> String {
        match subject {
            Subject::NamedNode(n) => format!("nn:{}", n.as_str()),
            Subject::BlankNode(b) => format!("bn:{}", b.as_str()),
            Subject::Variable(v) => format!("var:{}", v.as_str()),
            Subject::QuotedTriple(_) => "qt:unsupported".to_string(),
        }
    }

    fn term_to_cache_key_predicate(&self, predicate: &Predicate) -> String {
        match predicate {
            Predicate::NamedNode(n) => format!("pred_nn:{}", n.as_str()),
            Predicate::Variable(v) => format!("pred_var:{}", v.as_str()),
        }
    }

    fn term_to_cache_key_object(&self, object: &Object) -> String {
        match object {
            Object::NamedNode(n) => format!("obj_nn:{}", n.as_str()),
            Object::BlankNode(b) => format!("obj_bn:{}", b.as_str()),
            Object::Literal(l) => format!("obj_lit:{l}"),
            Object::Variable(v) => format!("obj_var:{}", v.as_str()),
            Object::QuotedTriple(_) => "obj_qt:unsupported".to_string(),
        }
    }

    fn term_to_cache_key_graph(&self, graph: &GraphName) -> String {
        match graph {
            GraphName::NamedNode(n) => format!("graph_nn:{}", n.as_str()),
            GraphName::BlankNode(b) => format!("graph_bn:{}", b.as_str()),
            GraphName::DefaultGraph => "graph_default".to_string(),
            GraphName::Variable(v) => format!("graph_var:{}", v.as_str()),
        }
    }

    /// Ultra-optimized batch processing method with pre-allocated caching
    pub fn add_batch_optimized(&self, quads: &[Quad]) -> Result<()> {
        if quads.is_empty() {
            return Ok(());
        }

        let start = std::time::Instant::now();
        let _lock = self.write_lock.lock();

        {
            let mut stats = self.performance_stats.lock();
            stats.batch_operations += 1;
            stats.average_batch_size = (stats.average_batch_size
                * (stats.batch_operations - 1) as f64
                + quads.len() as f64)
                / stats.batch_operations as f64;
        }

        let mut disk_quads = Vec::with_capacity(quads.len());
        let mut local_term_cache: HashMap<String, u64> = HashMap::with_capacity(quads.len() * 4);

        {
            let interner = self.term_interner.write();
            let mut cache = self.term_cache.write();

            for quad in quads {
                let subject_key = self.term_to_cache_key(quad.subject());
                let subject_id = if let Some(&id) = local_term_cache.get(&subject_key) {
                    id
                } else if let Some(&id) = cache.get(&subject_key) {
                    local_term_cache.insert(subject_key.clone(), id);
                    id
                } else {
                    let id = match quad.subject() {
                        Subject::NamedNode(n) => interner.intern_named_node(n),
                        Subject::BlankNode(b) => interner.intern_blank_node(b),
                        Subject::Variable(_) | Subject::QuotedTriple(_) => {
                            bail!("Variables and quoted triples cannot be interned in storage");
                        }
                    };
                    cache.insert(subject_key.clone(), id);
                    local_term_cache.insert(subject_key, id);
                    id
                };

                let predicate_key = self.term_to_cache_key_predicate(quad.predicate());
                let predicate_id = if let Some(&id) = local_term_cache.get(&predicate_key) {
                    id
                } else if let Some(&id) = cache.get(&predicate_key) {
                    local_term_cache.insert(predicate_key.clone(), id);
                    id
                } else {
                    let id = match quad.predicate() {
                        Predicate::NamedNode(n) => interner.intern_named_node(n),
                        Predicate::Variable(_) => {
                            bail!("Variables cannot be interned in storage");
                        }
                    };
                    cache.insert(predicate_key.clone(), id);
                    local_term_cache.insert(predicate_key, id);
                    id
                };

                let object_key = self.term_to_cache_key_object(quad.object());
                let object_id = if let Some(&id) = local_term_cache.get(&object_key) {
                    id
                } else if let Some(&id) = cache.get(&object_key) {
                    local_term_cache.insert(object_key.clone(), id);
                    id
                } else {
                    let id = match quad.object() {
                        Object::NamedNode(n) => interner.intern_named_node(n),
                        Object::BlankNode(b) => interner.intern_blank_node(b),
                        Object::Literal(l) => interner.intern_literal(l),
                        Object::Variable(_) | Object::QuotedTriple(_) => {
                            bail!("Variables and quoted triples cannot be interned in storage");
                        }
                    };
                    cache.insert(object_key.clone(), id);
                    local_term_cache.insert(object_key, id);
                    id
                };

                let graph_key = self.term_to_cache_key_graph(quad.graph_name());
                let graph_id = if let Some(&id) = local_term_cache.get(&graph_key) {
                    id
                } else if let Some(&id) = cache.get(&graph_key) {
                    local_term_cache.insert(graph_key.clone(), id);
                    id
                } else {
                    let id = match quad.graph_name() {
                        GraphName::NamedNode(n) => interner.intern_named_node(n),
                        GraphName::BlankNode(b) => interner.intern_blank_node(b),
                        GraphName::DefaultGraph => 0,
                        GraphName::Variable(_) => {
                            bail!("Variables cannot be interned in storage");
                        }
                    };
                    cache.insert(graph_key.clone(), id);
                    local_term_cache.insert(graph_key, id);
                    id
                };

                disk_quads.push(DiskQuad {
                    subject_id,
                    predicate_id,
                    object_id,
                    graph_id,
                });
            }
        }

        {
            let mut buffer = self.append_buffer.lock();
            buffer.extend(disk_quads);

            if buffer.len() >= 8192 {
                self.flush_buffer(&mut buffer)?;
            }
        }

        {
            let mut stats = self.performance_stats.lock();
            stats.total_flush_time_ms += start.elapsed().as_millis() as u64;
        }

        Ok(())
    }

    fn ensure_index(&self, index_name: &str) -> Result<()> {
        let mut indexes = self.indexes.write();

        if !indexes.contains_key(index_name) {
            let index_path = match index_name {
                "spo" => self.path.join("spo.idx"),
                "pos" => self.path.join("pos.idx"),
                "osp" => self.path.join("osp.idx"),
                "gspo" => self.path.join("gspo.idx"),
                _ => bail!("Unknown index: {}", index_name),
            };

            indexes.insert(index_name.to_string(), MmapIndex::new(&index_path)?);
        }

        Ok(())
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_stats.lock().clone()
    }

    /// Flush any remaining batches and optimize cache
    pub fn finalize(&self) -> Result<()> {
        {
            let mut batch_buffer = self.batch_buffer.lock();
            if !batch_buffer.is_empty() {
                let quads_to_process: Vec<Quad> = batch_buffer.drain(..).collect();
                drop(batch_buffer);
                self.add_batch_optimized(&quads_to_process)?;
            }
        }

        {
            let mut buffer = self.append_buffer.lock();
            if !buffer.is_empty() {
                self.flush_buffer(&mut buffer)?;
            }
        }

        {
            let mut cache = self.term_cache.write();
            if cache.len() > 50000 {
                let keys_to_remove: Vec<_> = cache.keys().skip(30000).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }

        Ok(())
    }

    /// Add multiple quads efficiently in batch
    pub fn add_batch(&self, quads: &[Quad]) -> Result<()> {
        if quads.is_empty() {
            return Ok(());
        }

        let _lock = self.write_lock.lock();
        let mut disk_quads = Vec::with_capacity(quads.len());

        {
            let interner = self.term_interner.write();

            for quad in quads {
                let subject_id = match quad.subject() {
                    Subject::NamedNode(n) => interner.intern_named_node(n),
                    Subject::BlankNode(b) => interner.intern_blank_node(b),
                    Subject::Variable(_) | Subject::QuotedTriple(_) => {
                        bail!("Variables and quoted triples cannot be interned in storage");
                    }
                };

                let predicate_id = match quad.predicate() {
                    Predicate::NamedNode(n) => interner.intern_named_node(n),
                    Predicate::Variable(_) => {
                        bail!("Variables cannot be interned in storage");
                    }
                };

                let object_id = match quad.object() {
                    Object::NamedNode(n) => interner.intern_named_node(n),
                    Object::BlankNode(b) => interner.intern_blank_node(b),
                    Object::Literal(l) => interner.intern_literal(l),
                    Object::Variable(_) | Object::QuotedTriple(_) => {
                        bail!("Variables and quoted triples cannot be interned in storage");
                    }
                };

                let graph_id = match quad.graph_name() {
                    GraphName::NamedNode(n) => interner.intern_named_node(n),
                    GraphName::BlankNode(b) => interner.intern_blank_node(b),
                    GraphName::DefaultGraph => 0,
                    GraphName::Variable(_) => {
                        bail!("Variables cannot be interned in storage");
                    }
                };

                disk_quads.push(DiskQuad {
                    subject_id,
                    predicate_id,
                    object_id,
                    graph_id,
                });
            }
        }

        {
            let mut buffer = self.append_buffer.lock();
            buffer.extend_from_slice(&disk_quads);

            if buffer.len() >= 8192 {
                self.flush_buffer(&mut buffer)?;
            }
        }

        Ok(())
    }

    fn flush_buffer(&self, buffer: &mut Vec<DiskQuad>) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        let mut data_file = self.data_file.lock();
        let mut header = self.header.write();

        let offset = data_file.seek(SeekFrom::End(0))?;

        let quad_size = std::mem::size_of::<DiskQuad>();
        let total_bytes = buffer.len() * quad_size;
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, total_bytes) };
        data_file.write_all(bytes)?;

        if buffer.len() > 100 {
            self.ensure_index("spo")?;
            self.ensure_index("pos")?;
            self.ensure_index("osp")?;
            self.ensure_index("gspo")?;

            let mut indexes = self.indexes.write();
            let base_idx = header.quad_count;

            let mut spo_entries: Vec<([u8; 24], IndexEntry)> = Vec::with_capacity(buffer.len());
            let mut pos_entries: Vec<([u8; 24], IndexEntry)> = Vec::with_capacity(buffer.len());
            let mut osp_entries: Vec<([u8; 24], IndexEntry)> = Vec::with_capacity(buffer.len());
            let mut gspo_entries: Vec<([u8; 32], IndexEntry)> = Vec::with_capacity(buffer.len());

            for (idx, quad) in buffer.iter().enumerate() {
                let quad_idx = base_idx + idx as u64;
                let entry_offset = offset + (idx * quad_size) as u64;

                let entry = IndexEntry {
                    offset: entry_offset,
                    quad_id: quad_idx,
                };

                spo_entries.push((
                    Self::make_binary_key_3(quad.subject_id, quad.predicate_id, quad.object_id),
                    entry,
                ));
                pos_entries.push((
                    Self::make_binary_key_3(quad.predicate_id, quad.object_id, quad.subject_id),
                    entry,
                ));
                osp_entries.push((
                    Self::make_binary_key_3(quad.object_id, quad.subject_id, quad.predicate_id),
                    entry,
                ));
                gspo_entries.push((
                    Self::make_binary_key_4(
                        quad.graph_id,
                        quad.subject_id,
                        quad.predicate_id,
                        quad.object_id,
                    ),
                    entry,
                ));
            }

            if let Some(spo) = indexes.get_mut("spo") {
                self.bulk_insert_index(spo, &spo_entries)?;
            }
            if let Some(pos) = indexes.get_mut("pos") {
                self.bulk_insert_index(pos, &pos_entries)?;
            }
            if let Some(osp) = indexes.get_mut("osp") {
                self.bulk_insert_index(osp, &osp_entries)?;
            }
            if let Some(gspo) = indexes.get_mut("gspo") {
                self.bulk_insert_index_4(gspo, &gspo_entries)?;
            }
        }

        header.quad_count += buffer.len() as u64;
        header.compute_checksum();

        data_file.seek(SeekFrom::Start(0))?;
        data_file.write_all(unsafe {
            std::slice::from_raw_parts(
                &*header as *const _ as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        })?;

        data_file.flush()?;
        buffer.clear();
        self.update_mmap()?;

        Ok(())
    }

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
    ) -> Result<QuadIterator<'_>> {
        let query_start = std::time::Instant::now();

        self.flush()?;

        let subject_id = subject.and_then(|s| match s {
            Subject::NamedNode(n) => self.term_interner.read().get_named_node_id(n),
            Subject::BlankNode(b) => self.term_interner.read().get_blank_node_id(b),
            Subject::Variable(_) | Subject::QuotedTriple(_) => None,
        });

        let predicate_id = predicate.and_then(|p| match p {
            Predicate::NamedNode(n) => self.term_interner.read().get_named_node_id(n),
            Predicate::Variable(_) => None,
        });

        let object_id = object.and_then(|o| match o {
            Object::NamedNode(n) => self.term_interner.read().get_named_node_id(n),
            Object::BlankNode(b) => self.term_interner.read().get_blank_node_id(b),
            Object::Literal(l) => self.term_interner.read().get_literal_id(l),
            Object::Variable(_) | Object::QuotedTriple(_) => None,
        });

        let graph_id = graph_name.and_then(|g| match g {
            GraphName::NamedNode(n) => self.term_interner.read().get_named_node_id(n),
            GraphName::BlankNode(b) => self.term_interner.read().get_blank_node_id(b),
            GraphName::DefaultGraph => Some(0),
            GraphName::Variable(_) => None,
        });

        let mut offsets = Vec::new();

        let index_type;
        match (subject_id, predicate_id, object_id, graph_id) {
            (Some(s), Some(p), Some(o), g) => {
                index_type = "spo";
                let key = format!("{s:016x}{p:016x}{o:016x}");
                if let Some(spo_index) = self.indexes.read().get("spo") {
                    let results = spo_index.search_prefix(&key)?;
                    for (_, entry) in results {
                        if g.is_none()
                            || self.check_graph_match(
                                entry.offset,
                                g.expect("g is Some when not is_none"),
                            )?
                        {
                            offsets.push(entry.offset);
                        }
                    }
                }
            }
            (Some(s), Some(p), None, g) => {
                index_type = "spo";
                let prefix = format!("{s:016x}{p:016x}");
                if let Some(spo_index) = self.indexes.read().get("spo") {
                    let results = spo_index.search_prefix(&prefix)?;
                    for (_, entry) in results {
                        if g.is_none()
                            || self.check_graph_match(
                                entry.offset,
                                g.expect("g is Some when not is_none"),
                            )?
                        {
                            offsets.push(entry.offset);
                        }
                    }
                }
            }
            (Some(s), None, None, g) => {
                index_type = "spo";
                let prefix = format!("{s:016x}");
                if let Some(spo_index) = self.indexes.read().get("spo") {
                    let results = spo_index.search_prefix(&prefix)?;
                    for (_, entry) in results {
                        if g.is_none()
                            || self.check_graph_match(
                                entry.offset,
                                g.expect("g is Some when not is_none"),
                            )?
                        {
                            offsets.push(entry.offset);
                        }
                    }
                }
            }
            (None, Some(p), Some(o), g) => {
                index_type = "pos";
                let key = format!("{p:016x}{o:016x}");
                if let Some(pos_index) = self.indexes.read().get("pos") {
                    let results = pos_index.search_prefix(&key)?;
                    for (_, entry) in results {
                        if g.is_none()
                            || self.check_graph_match(
                                entry.offset,
                                g.expect("g is Some when not is_none"),
                            )?
                        {
                            offsets.push(entry.offset);
                        }
                    }
                }
            }
            (None, None, Some(o), g) => {
                index_type = "osp";
                let prefix = format!("{o:016x}");
                if let Some(osp_index) = self.indexes.read().get("osp") {
                    let results = osp_index.search_prefix(&prefix)?;
                    for (_, entry) in results {
                        if g.is_none()
                            || self.check_graph_match(
                                entry.offset,
                                g.expect("g is Some when not is_none"),
                            )?
                        {
                            offsets.push(entry.offset);
                        }
                    }
                }
            }
            (None, None, None, Some(g)) => {
                index_type = "gspo";
                let prefix = format!("{g:016x}");
                if let Some(gspo_index) = self.indexes.read().get("gspo") {
                    let results = gspo_index.search_prefix(&prefix)?;
                    for (_, entry) in results {
                        offsets.push(entry.offset);
                    }
                }
            }
            _ => {
                index_type = "full_scan";
                let quad_count = self.header.read().quad_count;
                let quad_size = std::mem::size_of::<DiskQuad>() as u64;
                for i in 0..quad_count {
                    let offset = HEADER_SIZE as u64 + i * quad_size;
                    if self.check_pattern_match(
                        offset,
                        subject_id,
                        predicate_id,
                        object_id,
                        graph_id,
                    )? {
                        offsets.push(offset);
                    }
                }
            }
        }

        let latency_us = query_start.elapsed().as_micros() as u64;
        self.record_query_access(index_type, subject_id, predicate_id, latency_us);

        Ok(QuadIterator {
            store: self,
            offsets,
            current: 0,
        })
    }

    fn check_graph_match(&self, offset: u64, graph_id: u64) -> Result<bool> {
        let mmap = self.data_mmap.read();
        if let Some(mmap) = mmap.as_ref() {
            if offset + std::mem::size_of::<DiskQuad>() as u64
                <= HEADER_SIZE as u64 + mmap.len() as u64
            {
                let disk_quad = unsafe {
                    &*((mmap.as_ptr() as usize + (offset - HEADER_SIZE as u64) as usize)
                        as *const DiskQuad)
                };
                return Ok(disk_quad.graph_id == graph_id);
            }
        }
        Ok(false)
    }

    fn check_pattern_match(
        &self,
        offset: u64,
        subject_id: Option<u64>,
        predicate_id: Option<u64>,
        object_id: Option<u64>,
        graph_id: Option<u64>,
    ) -> Result<bool> {
        let mmap = self.data_mmap.read();
        if let Some(mmap) = mmap.as_ref() {
            if offset >= HEADER_SIZE as u64
                && offset + std::mem::size_of::<DiskQuad>() as u64
                    <= HEADER_SIZE as u64 + mmap.len() as u64
            {
                let disk_quad = unsafe {
                    &*((mmap.as_ptr() as usize + (offset - HEADER_SIZE as u64) as usize)
                        as *const DiskQuad)
                };

                if let Some(s) = subject_id {
                    if disk_quad.subject_id != s {
                        return Ok(false);
                    }
                }
                if let Some(p) = predicate_id {
                    if disk_quad.predicate_id != p {
                        return Ok(false);
                    }
                }
                if let Some(o) = object_id {
                    if disk_quad.object_id != o {
                        return Ok(false);
                    }
                }
                if let Some(g) = graph_id {
                    if disk_quad.graph_id != g {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Remove a quad from the store (marks as deleted for later compaction)
    pub fn remove_quad(&self, quad: &Quad) -> Result<bool> {
        let _write_lock = self.write_lock.lock();

        self.flush()?;

        let (subject_id, predicate_id, object_id, graph_id) = {
            let interner = self.term_interner.read();

            let subject_id = match quad.subject() {
                Subject::NamedNode(n) => interner.get_named_node_id(n),
                Subject::BlankNode(b) => interner.get_blank_node_id(b),
                _ => None,
            };

            let predicate_id = match quad.predicate() {
                Predicate::NamedNode(n) => interner.get_named_node_id(n),
                _ => None,
            };

            let object_id = match quad.object() {
                Object::NamedNode(n) => interner.get_named_node_id(n),
                Object::BlankNode(b) => interner.get_blank_node_id(b),
                Object::Literal(l) => interner.get_literal_id(l),
                _ => None,
            };

            let graph_id = match quad.graph_name() {
                GraphName::NamedNode(n) => interner.get_named_node_id(n),
                GraphName::BlankNode(b) => interner.get_blank_node_id(b),
                GraphName::DefaultGraph => Some(0),
                _ => None,
            };

            (subject_id, predicate_id, object_id, graph_id)
        };

        let (Some(sid), Some(pid), Some(oid), Some(gid)) =
            (subject_id, predicate_id, object_id, graph_id)
        else {
            return Ok(false);
        };

        let mmap = self.data_mmap.read();

        if let Some(mmap) = mmap.as_ref() {
            let data_size = mmap.len();
            let quad_size = std::mem::size_of::<DiskQuad>();
            let num_quads = data_size / quad_size;

            for i in 0..num_quads {
                let offset = HEADER_SIZE as u64 + (i * quad_size) as u64;

                {
                    let deleted = self.deleted_quads.read();
                    if deleted.contains(&offset) {
                        continue;
                    }
                }

                let disk_quad =
                    unsafe { &*((mmap.as_ptr() as usize + (i * quad_size)) as *const DiskQuad) };

                if disk_quad.subject_id == sid
                    && disk_quad.predicate_id == pid
                    && disk_quad.object_id == oid
                    && disk_quad.graph_id == gid
                {
                    let mut deleted = self.deleted_quads.write();
                    deleted.insert(offset);
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Check if a quad exists in the store
    pub fn contains_quad(&self, quad: &Quad) -> Result<bool> {
        let (subject_id, predicate_id, object_id, graph_id) = {
            let interner = self.term_interner.read();

            let subject_id = match quad.subject() {
                Subject::NamedNode(n) => interner.get_named_node_id(n),
                Subject::BlankNode(b) => interner.get_blank_node_id(b),
                _ => None,
            };

            let predicate_id = match quad.predicate() {
                Predicate::NamedNode(n) => interner.get_named_node_id(n),
                _ => None,
            };

            let object_id = match quad.object() {
                Object::NamedNode(n) => interner.get_named_node_id(n),
                Object::BlankNode(b) => interner.get_blank_node_id(b),
                Object::Literal(l) => interner.get_literal_id(l),
                _ => None,
            };

            let graph_id = match quad.graph_name() {
                GraphName::NamedNode(n) => interner.get_named_node_id(n),
                GraphName::BlankNode(b) => interner.get_blank_node_id(b),
                GraphName::DefaultGraph => Some(0),
                _ => None,
            };

            (subject_id, predicate_id, object_id, graph_id)
        };

        let (Some(sid), Some(pid), Some(oid), Some(gid)) =
            (subject_id, predicate_id, object_id, graph_id)
        else {
            return Ok(false);
        };

        let mmap = self.data_mmap.read();

        if let Some(mmap) = mmap.as_ref() {
            let data_size = mmap.len();
            let quad_size = std::mem::size_of::<DiskQuad>();
            let num_quads = data_size / quad_size;

            let deleted = self.deleted_quads.read();

            for i in 0..num_quads {
                let offset = HEADER_SIZE as u64 + (i * quad_size) as u64;

                if deleted.contains(&offset) {
                    continue;
                }

                let disk_quad =
                    unsafe { &*((mmap.as_ptr() as usize + (i * quad_size)) as *const DiskQuad) };

                if disk_quad.subject_id == sid
                    && disk_quad.predicate_id == pid
                    && disk_quad.object_id == oid
                    && disk_quad.graph_id == gid
                {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get the count of deleted quads pending compaction
    pub fn deleted_count(&self) -> usize {
        self.deleted_quads.read().len()
    }

    /// Compact the store to reclaim space by removing deleted entries
    pub fn compact(&self) -> Result<()> {
        let _write_lock = self.write_lock.lock();

        self.flush()?;

        let deleted_count = self.deleted_quads.read().len();
        if deleted_count == 0 {
            let term_path = self.path.join("terms.oxirs");
            if let Err(e) = self.term_interner.read().save(&term_path) {
                eprintln!("Warning: Failed to save term interner during compaction: {e}");
            }
            println!("Compaction completed (no deleted entries)");
            return Ok(());
        }

        println!(
            "Starting compaction: {} deleted entries to remove",
            deleted_count
        );

        let temp_data_path = self.path.join("data.oxirs.tmp");
        let mut temp_data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_data_path)
            .context("Failed to create temp data file")?;

        let mut new_header = FileHeader::new();
        temp_data_file.write_all(unsafe {
            std::slice::from_raw_parts(
                &new_header as *const _ as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        })?;

        let mut new_quad_count = 0u64;
        let deleted = self.deleted_quads.read();
        let mmap = self.data_mmap.read();

        if let Some(mmap) = mmap.as_ref() {
            let data_size = mmap.len();
            let quad_size = std::mem::size_of::<DiskQuad>();
            let num_quads = data_size / quad_size;

            for i in 0..num_quads {
                let offset = HEADER_SIZE as u64 + (i * quad_size) as u64;

                if deleted.contains(&offset) {
                    continue;
                }

                let disk_quad =
                    unsafe { &*((mmap.as_ptr() as usize + (i * quad_size)) as *const DiskQuad) };

                temp_data_file.write_all(unsafe {
                    std::slice::from_raw_parts(disk_quad as *const _ as *const u8, quad_size)
                })?;

                new_quad_count += 1;
            }
        }
        drop(mmap);
        drop(deleted);

        new_header.quad_count = new_quad_count;
        {
            let interner = self.term_interner.read();
            new_header.term_count = interner.stats().total_terms() as u64;
        }
        new_header.compute_checksum();

        temp_data_file.seek(SeekFrom::Start(0))?;
        temp_data_file.write_all(unsafe {
            std::slice::from_raw_parts(
                &new_header as *const _ as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        })?;
        temp_data_file.flush()?;
        temp_data_file.sync_all()?;

        let term_path = self.path.join("terms.oxirs");
        if let Err(e) = self.term_interner.read().save(&term_path) {
            eprintln!("Warning: Failed to save term interner during compaction: {e}");
        }

        let data_path = self.path.join("data.oxirs");

        {
            let mut data_mmap = self.data_mmap.write();
            *data_mmap = None;
        }

        drop(temp_data_file);

        fs::rename(&temp_data_path, &data_path)
            .context("Failed to atomically replace data file")?;

        let new_data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&data_path)
            .context("Failed to reopen data file after compaction")?;

        let file_len = new_data_file.metadata()?.len();
        let new_mmap = if file_len > HEADER_SIZE as u64 {
            Some(unsafe {
                MmapOptions::new()
                    .offset(HEADER_SIZE as u64)
                    .len((file_len - HEADER_SIZE as u64) as usize)
                    .map(&new_data_file)?
            })
        } else {
            None
        };

        {
            let mut data_file = self.data_file.lock();
            *data_file = new_data_file;
        }
        {
            let mut data_mmap = self.data_mmap.write();
            *data_mmap = new_mmap;
        }
        {
            let mut header = self.header.write();
            *header = new_header;
        }

        {
            let mut deleted = self.deleted_quads.write();
            deleted.clear();
        }

        {
            let mut indexes = self.indexes.write();
            indexes.clear();
        }

        println!(
            "Compaction completed: {} quads retained, {} quads removed",
            new_quad_count, deleted_count
        );
        Ok(())
    }

    fn make_binary_key_3(a: u64, b: u64, c: u64) -> [u8; 24] {
        let mut key = [0u8; 24];
        key[0..8].copy_from_slice(&a.to_be_bytes());
        key[8..16].copy_from_slice(&b.to_be_bytes());
        key[16..24].copy_from_slice(&c.to_be_bytes());
        key
    }

    fn make_binary_key_4(a: u64, b: u64, c: u64, d: u64) -> [u8; 32] {
        let mut key = [0u8; 32];
        key[0..8].copy_from_slice(&a.to_be_bytes());
        key[8..16].copy_from_slice(&b.to_be_bytes());
        key[16..24].copy_from_slice(&c.to_be_bytes());
        key[24..32].copy_from_slice(&d.to_be_bytes());
        key
    }

    fn bulk_insert_index(
        &self,
        index: &mut MmapIndex,
        entries: &[([u8; 24], IndexEntry)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let string_entries: Vec<(String, IndexEntry)> = entries
            .iter()
            .map(|(key_bytes, entry)| (String::from_utf8_lossy(key_bytes).to_string(), *entry))
            .collect();

        index.bulk_insert(&string_entries)?;
        Ok(())
    }

    fn bulk_insert_index_4(
        &self,
        index: &mut MmapIndex,
        entries: &[([u8; 32], IndexEntry)],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let string_entries: Vec<(String, IndexEntry)> = entries
            .iter()
            .map(|(key_bytes, entry)| (String::from_utf8_lossy(key_bytes).to_string(), *entry))
            .collect();

        index.bulk_insert(&string_entries)?;
        Ok(())
    }

    /// Get store statistics
    pub fn stats(&self) -> StoreStats {
        let header = self.header.read();

        let term_size = {
            let term_path = self.path.join("terms.oxirs");
            if term_path.exists() {
                term_path.metadata().map(|m| m.len()).unwrap_or(0)
            } else {
                header.term_count * 50
            }
        };

        StoreStats {
            quad_count: header.quad_count,
            term_count: header.term_count,
            data_size: header.index_offset - header.data_offset,
            index_size: header.term_offset - header.index_offset,
            term_size,
        }
    }

    /// Get access statistics for query optimization
    pub fn get_access_stats(&self) -> AccessStats {
        let mut stats = self.access_stats.lock().clone();

        let subject_counts = self.subject_access_counts.read();
        let mut subject_vec: Vec<_> = subject_counts.iter().map(|(&k, &v)| (k, v)).collect();
        subject_vec.sort_by(|a, b| b.1.cmp(&a.1));
        stats.hot_subjects = subject_vec.into_iter().take(10).collect();

        let predicate_counts = self.predicate_access_counts.read();
        let mut predicate_vec: Vec<_> = predicate_counts.iter().map(|(&k, &v)| (k, v)).collect();
        predicate_vec.sort_by(|a, b| b.1.cmp(&a.1));
        stats.hot_predicates = predicate_vec.into_iter().take(10).collect();

        stats
    }

    fn record_query_access(
        &self,
        index_type: &str,
        subject_id: Option<u64>,
        predicate_id: Option<u64>,
        latency_us: u64,
    ) {
        let mut stats = self.access_stats.lock();

        match index_type {
            "spo" => stats.spo_queries += 1,
            "pos" => stats.pos_queries += 1,
            "osp" => stats.osp_queries += 1,
            "gspo" => stats.gspo_queries += 1,
            "full_scan" => stats.full_scans += 1,
            _ => {}
        }

        stats.total_queries += 1;
        stats.avg_query_latency_us =
            (stats.avg_query_latency_us * (stats.total_queries - 1) as f64 + latency_us as f64)
                / stats.total_queries as f64;

        drop(stats);

        if let Some(sid) = subject_id {
            let mut counts = self.subject_access_counts.write();
            *counts.entry(sid).or_insert(0) += 1;
        }

        if let Some(pid) = predicate_id {
            let mut counts = self.predicate_access_counts.write();
            *counts.entry(pid).or_insert(0) += 1;
        }
    }

    /// Reset access statistics
    pub fn reset_access_stats(&self) {
        *self.access_stats.lock() = AccessStats::default();
        self.subject_access_counts.write().clear();
        self.predicate_access_counts.write().clear();
    }
}

impl Drop for MmapStore {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Iterator over quads in the store
pub struct QuadIterator<'a> {
    store: &'a MmapStore,
    offsets: Vec<u64>,
    current: usize,
}

impl<'a> Iterator for QuadIterator<'a> {
    type Item = Result<Quad>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.offsets.len() {
            return None;
        }

        let offset = self.offsets[self.current];
        self.current += 1;

        let mmap = self.store.data_mmap.read();
        let mmap = match mmap.as_ref() {
            Some(m) => m,
            None => return Some(Err(anyhow::anyhow!("No memory map available"))),
        };

        if offset < HEADER_SIZE as u64
            || offset + std::mem::size_of::<DiskQuad>() as u64
                > HEADER_SIZE as u64 + mmap.len() as u64
        {
            return Some(Err(anyhow::anyhow!("Invalid quad offset")));
        }

        let disk_quad = unsafe {
            &*((mmap.as_ptr() as usize + (offset - HEADER_SIZE as u64) as usize) as *const DiskQuad)
        };

        let interner = self.store.term_interner.read();

        let subject = match interner.get_subject(disk_quad.subject_id as u32) {
            Some(s) => s,
            None => {
                return Some(Err(anyhow::anyhow!(
                    "Invalid subject ID: {}",
                    disk_quad.subject_id
                )))
            }
        };

        let predicate = match interner.get_predicate(disk_quad.predicate_id as u32) {
            Some(p) => p,
            None => {
                return Some(Err(anyhow::anyhow!(
                    "Invalid predicate ID: {}",
                    disk_quad.predicate_id
                )))
            }
        };

        let object = match interner.get_object(disk_quad.object_id as u32) {
            Some(o) => o,
            None => {
                return Some(Err(anyhow::anyhow!(
                    "Invalid object ID: {}",
                    disk_quad.object_id
                )))
            }
        };

        let graph_name = if disk_quad.graph_id == 0 {
            GraphName::DefaultGraph
        } else {
            match interner.get_subject(disk_quad.graph_id as u32) {
                Some(Subject::NamedNode(n)) => GraphName::NamedNode(n),
                Some(Subject::BlankNode(b)) => GraphName::BlankNode(b),
                _ => {
                    return Some(Err(anyhow::anyhow!(
                        "Invalid graph ID: {}",
                        disk_quad.graph_id
                    )))
                }
            }
        };

        Some(Ok(Quad::new(subject, predicate, object, graph_name)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.offsets.len() - self.current;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests;
