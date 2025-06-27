//! Memory-mapped vector index for efficient disk-based storage
//!
//! This module provides a disk-based vector index using memory-mapped files for
//! efficient access to large vector datasets that don't fit in memory.

use crate::{Vector, VectorIndex, index::{DistanceMetric, IndexConfig, SearchResult}};
use crate::mmap_advanced::{AdvancedMemoryMap, MemoryMapStats, NumaVectorAllocator};
use anyhow::{Result, Context, bail};
use memmap2::{MmapMut, MmapOptions, Mmap};
use parking_lot::{RwLock, Mutex};
use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::collections::{HashMap, BinaryHeap};
use blake3::Hasher;
use oxirs_core::parallel::*;

/// Magic number for file format identification
const MAGIC: &[u8; 8] = b"OXIRSVEC";

/// Current file format version
const VERSION: u32 = 1;

/// Default page size for memory mapping (4KB)
const PAGE_SIZE: usize = 4096;

/// Header size (must be page-aligned)
const HEADER_SIZE: usize = PAGE_SIZE;

/// File header structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FileHeader {
    magic: [u8; 8],
    version: u32,
    flags: u32,
    vector_count: u64,
    dimensions: u32,
    vector_size: u32, // Size of each vector in bytes
    data_offset: u64,
    index_offset: u64,
    uri_offset: u64,
    checksum: [u8; 32],
    reserved: [u8; 3968], // Pad to PAGE_SIZE
}

impl FileHeader {
    fn new(dimensions: u32) -> Self {
        let vector_size = dimensions * std::mem::size_of::<f32>() as u32;
        Self {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            vector_count: 0,
            dimensions,
            vector_size,
            data_offset: HEADER_SIZE as u64,
            index_offset: 0,
            uri_offset: 0,
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
        hasher.update(&self.vector_count.to_le_bytes());
        hasher.update(&self.dimensions.to_le_bytes());
        hasher.update(&self.vector_size.to_le_bytes());
        hasher.update(&self.data_offset.to_le_bytes());
        hasher.update(&self.index_offset.to_le_bytes());
        hasher.update(&self.uri_offset.to_le_bytes());
        self.checksum = *hasher.finalize().as_bytes();
    }
}

/// Memory-mapped vector index for large datasets
pub struct MemoryMappedVectorIndex {
    config: IndexConfig,
    path: PathBuf,
    header: Arc<RwLock<FileHeader>>,
    data_file: Arc<Mutex<File>>,
    data_mmap: Arc<RwLock<Option<Mmap>>>,
    uri_map: Arc<RwLock<HashMap<String, u64>>>, // URI to vector ID
    uri_store: Arc<RwLock<Vec<String>>>, // Vector ID to URI
    write_buffer: Arc<Mutex<Vec<(String, Vector)>>>,
    buffer_size: usize,
    
    // Advanced memory mapping features
    advanced_mmap: Option<Arc<AdvancedMemoryMap>>,
    numa_allocator: Arc<NumaVectorAllocator>,
    enable_lazy_loading: bool,
}

impl MemoryMappedVectorIndex {
    /// Create a new memory-mapped vector index
    pub fn new<P: AsRef<Path>>(path: P, config: IndexConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Create or open the data file
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .context("Failed to open data file")?;

        // Initialize or load header
        let header = if data_file.metadata()?.len() == 0 {
            // New file, write header
            let header = FileHeader::new(0);
            data_file.set_len(HEADER_SIZE as u64)?;
            let mut header_bytes = vec![0u8; HEADER_SIZE];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &header as *const _ as *const u8,
                    header_bytes.as_mut_ptr(),
                    std::mem::size_of::<FileHeader>(),
                );
            }
            (&data_file).write_all(&header_bytes)?;
            header
        } else {
            // Existing file, read header
            let mut mmap = unsafe { MmapOptions::new().map(&data_file)? };
            let header = unsafe {
                std::ptr::read(mmap.as_ptr() as *const FileHeader)
            };
            header.validate()?;
            header
        };

        Ok(Self {
            config,
            path,
            header: Arc::new(RwLock::new(header)),
            data_file: Arc::new(Mutex::new(data_file)),
            data_mmap: Arc::new(RwLock::new(None)),
            uri_map: Arc::new(RwLock::new(HashMap::new())),
            uri_store: Arc::new(RwLock::new(Vec::new())),
            write_buffer: Arc::new(Mutex::new(Vec::new())),
            buffer_size: 1000, // Buffer 1000 vectors before flushing
            advanced_mmap: None,
            numa_allocator: Arc::new(NumaVectorAllocator::new()),
            enable_lazy_loading: true,
        })
    }

    /// Load an existing memory-mapped index
    pub fn load<P: AsRef<Path>>(path: P, config: IndexConfig) -> Result<Self> {
        let mut index = Self::new(path, config)?;
        index.reload_mmap()?;
        index.load_uri_mappings()?;
        Ok(index)
    }

    /// Reload memory mapping
    fn reload_mmap(&mut self) -> Result<()> {
        let file = self.data_file.lock();
        let file_len = file.metadata()?.len();
        
        if file_len > HEADER_SIZE as u64 {
            let mmap = unsafe { MmapOptions::new().map(&*file)? };
            
            // Create advanced memory map if lazy loading is enabled
            if self.enable_lazy_loading {
                // Note: We don't pass the mmap to AdvancedMemoryMap since it can't be cloned
                // The advanced mmap will create its own mapping
                let advanced = AdvancedMemoryMap::new(None, 10000);
                self.advanced_mmap = Some(Arc::new(advanced));
            }
            
            *self.data_mmap.write() = Some(mmap);
        }
        
        Ok(())
    }

    /// Load URI mappings from disk
    fn load_uri_mappings(&self) -> Result<()> {
        let header = self.header.read();
        let uri_offset = header.uri_offset as usize;
        
        if uri_offset > 0 {
            if let Some(ref mmap) = *self.data_mmap.read() {
                // Parse URI mappings from memory-mapped region
                let uri_data = &mmap[uri_offset..];
                let mut offset = 0;
                let mut uri_map = self.uri_map.write();
                let mut uri_store = self.uri_store.write();
                
                for id in 0..header.vector_count {
                    if offset + 4 > uri_data.len() {
                        break;
                    }
                    
                    let uri_len = u32::from_le_bytes([
                        uri_data[offset],
                        uri_data[offset + 1],
                        uri_data[offset + 2],
                        uri_data[offset + 3],
                    ]) as usize;
                    offset += 4;
                    
                    if offset + uri_len > uri_data.len() {
                        break;
                    }
                    
                    let uri = String::from_utf8_lossy(&uri_data[offset..offset + uri_len])
                        .into_owned();
                    offset += uri_len;
                    
                    uri_map.insert(uri.clone(), id);
                    uri_store.push(uri);
                }
            }
        }
        
        Ok(())
    }

    /// Flush write buffer to disk
    fn flush_buffer(&self) -> Result<()> {
        let mut buffer = self.write_buffer.lock();
        if buffer.is_empty() {
            return Ok(());
        }

        let mut file = self.data_file.lock();
        let mut header = self.header.write();
        
        // Calculate required space
        let vectors_to_write = buffer.len();
        let vector_data_size = vectors_to_write * header.vector_size as usize;
        
        // Extend file if needed
        let current_data_end = header.data_offset + 
            (header.vector_count * header.vector_size as u64);
        let new_data_end = current_data_end + vector_data_size as u64;
        
        file.set_len(new_data_end)?;
        file.seek(SeekFrom::Start(current_data_end))?;
        
        // Write vectors
        let mut uri_map = self.uri_map.write();
        let mut uri_store = self.uri_store.write();
        
        for (uri, vector) in buffer.drain(..) {
            // Validate dimensions
            if header.dimensions == 0 {
                header.dimensions = vector.dimensions as u32;
                header.vector_size = vector.dimensions as u32 * std::mem::size_of::<f32>() as u32;
            } else if vector.dimensions != header.dimensions as usize {
                bail!(
                    "Vector dimensions ({}) don't match index dimensions ({})",
                    vector.dimensions,
                    header.dimensions
                );
            }
            
            // Write vector data
            let vector_f32 = vector.as_f32();
            let vector_bytes: Vec<u8> = vector_f32.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            file.write_all(&vector_bytes)?;
            
            // Update mappings
            let vector_id = header.vector_count;
            uri_map.insert(uri.clone(), vector_id);
            uri_store.push(uri);
            header.vector_count += 1;
        }
        
        // Update header
        header.compute_checksum();
        file.seek(SeekFrom::Start(0))?;
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &*header as *const _ as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        };
        file.write_all(header_bytes)?;
        file.sync_all()?;
        
        // Reload memory mapping
        drop(file);
        drop(header);
        drop(uri_map);
        drop(uri_store);
        
        // For now, just reload the basic mmap without advanced features in flush
        // Advanced features will be enabled on load
        let file = self.data_file.lock();
        let file_len = file.metadata()?.len();
        if file_len > HEADER_SIZE as u64 {
            let mmap = unsafe { MmapOptions::new().map(&*file)? };
            *self.data_mmap.write() = Some(mmap);
        }
        
        Ok(())
    }

    /// Get vector by ID from memory-mapped region
    fn get_vector_by_id(&self, id: u64) -> Result<Option<Vector>> {
        let header = self.header.read();
        
        if id >= header.vector_count {
            return Ok(None);
        }
        
        if let Some(ref mmap) = *self.data_mmap.read() {
            let offset = header.data_offset as usize + (id as usize * header.vector_size as usize);
            let end = offset + header.vector_size as usize;
            
            if end <= mmap.len() {
                let vector_bytes = &mmap[offset..end];
                let mut values = Vec::with_capacity(header.dimensions as usize);
                
                for i in 0..header.dimensions as usize {
                    let byte_offset = i * std::mem::size_of::<f32>();
                    let bytes = [
                        vector_bytes[byte_offset],
                        vector_bytes[byte_offset + 1],
                        vector_bytes[byte_offset + 2],
                        vector_bytes[byte_offset + 3],
                    ];
                    values.push(f32::from_le_bytes(bytes));
                }
                
                return Ok(Some(Vector::new(values)));
            }
        }
        
        Ok(None)
    }

    /// Search using brute force with memory-mapped vectors
    fn search_mmap(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        let header = self.header.read();
        let distance_metric = self.config.distance_metric;
        
        if header.vector_count == 0 {
            return Ok(Vec::new());
        }

        // Check if we should use parallel search
        if self.config.parallel && header.vector_count > 1000 {
            self.search_mmap_parallel(query, k, distance_metric)
        } else {
            self.search_mmap_sequential(query, k, distance_metric)
        }
    }

    /// Sequential search through memory-mapped vectors
    fn search_mmap_sequential(
        &self,
        query: &Vector,
        k: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        let header = self.header.read();
        let uri_store = self.uri_store.read();
        let mut heap = BinaryHeap::new();

        for id in 0..header.vector_count {
            if let Some(vector) = self.get_vector_by_id(id)? {
                let distance = distance_metric.distance_vectors(query, &vector);
                
                if heap.len() < k {
                    heap.push(std::cmp::Reverse(SearchResult {
                        uri: uri_store.get(id as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("vector_{}", id)),
                        distance,
                        metadata: None,
                    }));
                } else if let Some(std::cmp::Reverse(worst)) = heap.peek() {
                    if distance < worst.distance {
                        heap.pop();
                        heap.push(std::cmp::Reverse(SearchResult {
                            uri: uri_store.get(id as usize)
                                .cloned()
                                .unwrap_or_else(|| format!("vector_{}", id)),
                            distance,
                            metadata: None,
                        }));
                    }
                }
            }
        }

        let mut results: Vec<SearchResult> = heap.into_iter().map(|r| r.0).collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(results)
    }

    /// Parallel search through memory-mapped vectors
    fn search_mmap_parallel(
        &self,
        query: &Vector,
        k: usize,
        distance_metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        let header = self.header.read();
        let uri_store = self.uri_store.read();
        let vector_count = header.vector_count;
        let chunk_size = (vector_count / num_threads() as u64).max(100);
        
        // Process chunks in parallel
        let partial_results: Vec<Vec<SearchResult>> = (0..vector_count)
            .step_by(chunk_size as usize)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&start_id| {
                let end_id = (start_id + chunk_size).min(vector_count);
                let mut local_heap = BinaryHeap::new();
                
                for id in start_id..end_id {
                    if let Ok(Some(vector)) = self.get_vector_by_id(id) {
                        let distance = distance_metric.distance_vectors(query, &vector);
                        
                        if local_heap.len() < k {
                            local_heap.push(std::cmp::Reverse(SearchResult {
                                uri: uri_store.get(id as usize)
                                    .cloned()
                                    .unwrap_or_else(|| format!("vector_{}", id)),
                                distance,
                                metadata: None,
                            }));
                        } else if let Some(std::cmp::Reverse(worst)) = local_heap.peek() {
                            if distance < worst.distance {
                                local_heap.pop();
                                local_heap.push(std::cmp::Reverse(SearchResult {
                                    uri: uri_store.get(id as usize)
                                        .cloned()
                                        .unwrap_or_else(|| format!("vector_{}", id)),
                                    distance,
                                    metadata: None,
                                }));
                            }
                        }
                    }
                }
                
                local_heap.into_sorted_vec().into_iter().map(|r| r.0).collect()
            })
            .collect();
        
        // Merge results from all chunks
        let mut final_heap = BinaryHeap::new();
        for partial in partial_results {
            for result in partial {
                if final_heap.len() < k {
                    final_heap.push(std::cmp::Reverse(result));
                } else if let Some(std::cmp::Reverse(worst)) = final_heap.peek() {
                    if result.distance < worst.distance {
                        final_heap.pop();
                        final_heap.push(std::cmp::Reverse(result));
                    }
                }
            }
        }
        
        let mut results: Vec<SearchResult> = final_heap.into_iter().map(|r| r.0).collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(results)
    }

    /// Save URI mappings to disk
    pub fn save_uri_mappings(&self) -> Result<()> {
        let mut file = self.data_file.lock();
        let mut header = self.header.write();
        let uri_store = self.uri_store.read();
        
        // Calculate size needed for URI data
        let mut uri_data_size = 0;
        for uri in uri_store.iter() {
            uri_data_size += 4 + uri.len(); // 4 bytes for length + URI bytes
        }
        
        // Set URI offset after vector data
        let data_end = header.data_offset + (header.vector_count * header.vector_size as u64);
        header.uri_offset = data_end;
        
        // Extend file and write URI data
        file.set_len(data_end + uri_data_size as u64)?;
        file.seek(SeekFrom::Start(header.uri_offset))?;
        
        for uri in uri_store.iter() {
            let len_bytes = (uri.len() as u32).to_le_bytes();
            file.write_all(&len_bytes)?;
            file.write_all(uri.as_bytes())?;
        }
        
        // Update header
        header.compute_checksum();
        file.seek(SeekFrom::Start(0))?;
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &*header as *const _ as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        };
        file.write_all(header_bytes)?;
        file.sync_all()?;
        
        Ok(())
    }

    /// Compact the index file by removing deleted entries
    pub fn compact(&self) -> Result<()> {
        // This would rewrite the file removing any gaps
        // For now, we don't support deletion, so nothing to compact
        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> MemoryMappedIndexStats {
        let header = self.header.read();
        let file_size = self.data_file.lock()
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);
        
        MemoryMappedIndexStats {
            vector_count: header.vector_count,
            dimensions: header.dimensions,
            file_size,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let uri_map_size = self.uri_map.read().len() * 
            (std::mem::size_of::<String>() + std::mem::size_of::<u64>());
        let uri_store_size = self.uri_store.read().iter()
            .map(|s| s.capacity())
            .sum::<usize>();
        let buffer_size = self.write_buffer.lock().len() * 
            (std::mem::size_of::<String>() + std::mem::size_of::<Vector>());
        
        uri_map_size + uri_store_size + buffer_size + HEADER_SIZE
    }
    
    /// Enable or disable lazy loading
    pub fn set_lazy_loading(&mut self, enabled: bool) {
        self.enable_lazy_loading = enabled;
    }
    
    /// Get advanced memory mapping statistics
    pub fn advanced_stats(&self) -> Option<MemoryMapStats> {
        self.advanced_mmap.as_ref().map(|mmap| mmap.stats())
    }
    
    /// Configure NUMA allocation preferences
    pub fn configure_numa(&mut self, numa_enabled: bool) {
        if numa_enabled {
            self.numa_allocator = Arc::new(NumaVectorAllocator::new());
        }
    }
}

impl VectorIndex for MemoryMappedVectorIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Add to write buffer
        self.write_buffer.lock().push((uri, vector));
        
        // Flush if buffer is full
        if self.write_buffer.lock().len() >= self.buffer_size {
            self.flush_buffer()?;
        }
        
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        // Flush any pending writes
        if !self.write_buffer.lock().is_empty() {
            self.flush_buffer()?;
        }
        
        let results = self.search_mmap(query, k)?;
        Ok(results.into_iter().map(|r| (r.uri, r.distance)).collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        // Flush any pending writes
        if !self.write_buffer.lock().is_empty() {
            self.flush_buffer()?;
        }
        
        let header = self.header.read();
        let uri_store = self.uri_store.read();
        let distance_metric = self.config.distance_metric;
        let mut results = Vec::new();

        for id in 0..header.vector_count {
            if let Some(vector) = self.get_vector_by_id(id)? {
                let distance = distance_metric.distance_vectors(query, &vector);
                if distance <= threshold {
                    let uri = uri_store.get(id as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("vector_{}", id));
                    results.push((uri, distance));
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }
    
    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        // Memory-mapped index doesn't store vectors in memory
        // We would need to read from disk, which doesn't fit the API
        // that returns a reference. Return None for now.
        None
    }
}

impl Drop for MemoryMappedVectorIndex {
    fn drop(&mut self) {
        // Flush any remaining vectors
        if let Err(e) = self.flush_buffer() {
            eprintln!("Error flushing buffer on drop: {}", e);
        }
        // Save URI mappings
        if let Err(e) = self.save_uri_mappings() {
            eprintln!("Error saving URI mappings on drop: {}", e);
        }
    }
}

/// Statistics for memory-mapped index
#[derive(Debug, Clone)]
pub struct MemoryMappedIndexStats {
    pub vector_count: u64,
    pub dimensions: u32,
    pub file_size: u64,
    pub memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_mapped_index_basic() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_vectors.idx");
        
        let config = IndexConfig::default();
        let mut index = MemoryMappedVectorIndex::new(&path, config)?;
        
        // Insert some vectors
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let v3 = Vector::new(vec![7.0, 8.0, 9.0]);
        
        index.insert("vec1".to_string(), v1.clone())?;
        index.insert("vec2".to_string(), v2.clone())?;
        index.insert("vec3".to_string(), v3.clone())?;
        
        // Force flush
        index.flush_buffer()?;
        
        // Search
        let query = Vector::new(vec![3.0, 4.0, 5.0]);
        let results = index.search_knn(&query, 2)?;
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "vec2");
        
        Ok(())
    }

    #[test]
    fn test_memory_mapped_index_persistence() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test_persist.idx");
        
        // Create and populate index
        {
            let config = IndexConfig::default();
            let mut index = MemoryMappedVectorIndex::new(&path, config)?;
            
            for i in 0..10 {
                let vec = Vector::new(vec![i as f32, (i + 1) as f32, (i + 2) as f32]);
                index.insert(format!("vec{}", i), vec)?;
            }
        }
        
        // Load existing index
        {
            let config = IndexConfig::default();
            let index = MemoryMappedVectorIndex::load(&path, config)?;
            
            let stats = index.stats();
            assert_eq!(stats.vector_count, 10);
            assert_eq!(stats.dimensions, 3);
            
            let query = Vector::new(vec![5.0, 6.0, 7.0]);
            let results = index.search_knn(&query, 3)?;
            
            assert_eq!(results.len(), 3);
            assert_eq!(results[0].0, "vec5");
        }
        
        Ok(())
    }
}