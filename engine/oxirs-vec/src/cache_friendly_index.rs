//! Cache-friendly memory layouts for vector index
//!
//! This module provides optimized data structures for better cache performance:
//! - Structure of Arrays (SoA) layout for better vectorization
//! - Hot/cold data separation
//! - Cache-line aligned storage
//! - Prefetching hints for predictable access patterns

use crate::{similarity::SimilarityConfig, Vector, VectorIndex};
use anyhow::Result;
use oxirs_core::parallel::*;
use std::alloc::{alloc, dealloc, Layout};
use std::cmp::Ordering as CmpOrdering;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Wrapper for f32 to implement Ord trait for use in BinaryHeap
#[derive(Debug, Clone, Copy)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.partial_cmp(other).unwrap_or(CmpOrdering::Equal)
    }
}

/// Cache line size (typically 64 bytes on modern CPUs)
const CACHE_LINE_SIZE: usize = 64;

/// Align to cache line boundary
#[repr(C, align(64))]
struct CacheAligned<T>(T);

/// Cache-friendly vector index with optimized memory layout
pub struct CacheFriendlyVectorIndex {
    // Hot data - frequently accessed during search
    hot_data: HotData,

    // Cold data - accessed less frequently
    cold_data: ColdData,

    // Configuration
    config: IndexConfig,

    // Statistics for adaptive optimization
    stats: IndexStats,
}

/// Hot data layout - optimized for search operations
struct HotData {
    // Vector data in Structure of Arrays format
    vectors_soa: VectorsSoA,

    // Precomputed norms for fast similarity computation
    norms: AlignedVec<f32>,

    // Compact URI indices (4 bytes instead of full strings)
    uri_indices: AlignedVec<u32>,
}

/// Cold data - accessed during result retrieval
struct ColdData {
    // Full URI strings
    uris: Vec<String>,

    // Optional metadata
    metadata: Vec<Option<std::collections::HashMap<String, String>>>,
}

/// Structure of Arrays for vector data
struct VectorsSoA {
    // Transposed vector data for better SIMD access
    // data[dimension][vector_index]
    data: Vec<AlignedVec<f32>>,

    // Number of vectors
    count: AtomicUsize,

    // Dimensionality
    dimensions: usize,
}

/// Cache-line aligned vector storage
struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T: Copy> AlignedVec<T> {
    fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: ptr::null_mut(),
                len: 0,
                capacity: 0,
            };
        }

        let layout =
            Layout::from_size_align(capacity * std::mem::size_of::<T>(), CACHE_LINE_SIZE).unwrap();

        unsafe {
            let ptr = alloc(layout) as *mut T;
            Self {
                ptr,
                len: 0,
                capacity,
            }
        }
    }

    fn push(&mut self, value: T) {
        if self.len >= self.capacity {
            self.grow();
        }

        unsafe {
            ptr::write(self.ptr.add(self.len), value);
        }
        self.len += 1;
    }

    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 {
            16
        } else {
            self.capacity * 2
        };
        let new_layout =
            Layout::from_size_align(new_capacity * std::mem::size_of::<T>(), CACHE_LINE_SIZE)
                .unwrap();

        unsafe {
            let new_ptr = alloc(new_layout) as *mut T;

            if !self.ptr.is_null() {
                ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len);

                let old_layout = Layout::from_size_align(
                    self.capacity * std::mem::size_of::<T>(),
                    CACHE_LINE_SIZE,
                )
                .unwrap();
                dealloc(self.ptr as *mut u8, old_layout);
            }

            self.ptr = new_ptr;
            self.capacity = new_capacity;
        }
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.capacity > 0 {
            let layout =
                Layout::from_size_align(self.capacity * std::mem::size_of::<T>(), CACHE_LINE_SIZE)
                    .unwrap();
            unsafe {
                dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

/// Configuration for cache-friendly index
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Expected number of vectors (for preallocation)
    pub expected_vectors: usize,

    /// Enable prefetching hints
    pub enable_prefetch: bool,

    /// Similarity configuration
    pub similarity_config: SimilarityConfig,

    /// Enable parallel search
    pub parallel_search: bool,

    /// Minimum vectors for parallel processing
    pub parallel_threshold: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            expected_vectors: 10_000,
            enable_prefetch: true,
            similarity_config: SimilarityConfig::default(),
            parallel_search: true,
            parallel_threshold: 1000,
        }
    }
}

/// Statistics for adaptive optimization
#[derive(Debug, Default)]
struct IndexStats {
    searches: AtomicUsize,
    cache_misses: AtomicUsize,
    total_search_time: AtomicUsize,
}

impl CacheFriendlyVectorIndex {
    pub fn new(config: IndexConfig) -> Self {
        let dimensions = 0; // Will be set on first insert

        Self {
            hot_data: HotData {
                vectors_soa: VectorsSoA {
                    data: Vec::new(),
                    count: AtomicUsize::new(0),
                    dimensions,
                },
                norms: AlignedVec::new(config.expected_vectors),
                uri_indices: AlignedVec::new(config.expected_vectors),
            },
            cold_data: ColdData {
                uris: Vec::with_capacity(config.expected_vectors),
                metadata: Vec::with_capacity(config.expected_vectors),
            },
            config,
            stats: IndexStats::default(),
        }
    }

    /// Initialize SoA structure for given dimensions
    fn initialize_soa(&mut self, dimensions: usize) {
        self.hot_data.vectors_soa.dimensions = dimensions;
        self.hot_data.vectors_soa.data = (0..dimensions)
            .map(|_| AlignedVec::new(self.config.expected_vectors))
            .collect();
    }

    /// Add vector data to SoA structure
    fn add_to_soa(&mut self, vector: &[f32]) {
        for (dim, value) in vector.iter().enumerate() {
            self.hot_data.vectors_soa.data[dim].push(*value);
        }
    }

    /// Compute L2 norm for caching
    fn compute_norm(vector: &[f32]) -> f32 {
        use oxirs_core::simd::SimdOps;
        f32::norm(vector)
    }

    /// Prefetch data for upcoming access
    #[inline(always)]
    fn prefetch_vector(&self, index: usize) {
        if !self.config.enable_prefetch {
            return;
        }

        // Prefetch vector data for next few vectors
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;

            for i in 0..4 {
                let next_idx = index + i;
                if next_idx < self.hot_data.vectors_soa.count.load(Ordering::Relaxed) {
                    // Prefetch first few dimensions
                    for dim in 0..self.hot_data.vectors_soa.dimensions.min(8) {
                        let ptr = self.hot_data.vectors_soa.data[dim].ptr.add(next_idx);
                        _mm_prefetch(ptr as *const i8, 1); // _MM_HINT_T1
                    }
                }
            }
        }
    }

    /// Sequential search with cache-friendly access pattern
    fn search_sequential(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        use oxirs_core::simd::SimdOps;

        let count = self.hot_data.vectors_soa.count.load(Ordering::Relaxed);
        let metric = self.config.similarity_config.primary_metric;

        // Precompute query norm for cosine similarity
        let query_norm = Self::compute_norm(query);

        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(OrderedFloat, usize)>> =
            std::collections::BinaryHeap::new();

        // Process vectors in chunks for better cache utilization
        const CHUNK_SIZE: usize = 16;

        for chunk_start in (0..count).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(count);

            // Prefetch next chunk
            if chunk_end < count {
                self.prefetch_vector(chunk_end);
            }

            // Process current chunk
            for idx in chunk_start..chunk_end {
                // Compute similarity using SoA layout
                let similarity = match metric {
                    crate::similarity::SimilarityMetric::Cosine => {
                        let mut dot_product = 0.0f32;

                        // Process dimensions in groups for better vectorization
                        for dim in 0..self.hot_data.vectors_soa.dimensions {
                            let vec_val =
                                unsafe { *self.hot_data.vectors_soa.data[dim].ptr.add(idx) };
                            dot_product += query[dim] * vec_val;
                        }

                        let vec_norm = self.hot_data.norms.as_slice()[idx];
                        dot_product / (query_norm * vec_norm + 1e-8)
                    }
                    _ => {
                        // For other metrics, reconstruct vector (less efficient)
                        let mut vector = vec![0.0f32; self.hot_data.vectors_soa.dimensions];
                        for dim in 0..self.hot_data.vectors_soa.dimensions {
                            vector[dim] =
                                unsafe { *self.hot_data.vectors_soa.data[dim].ptr.add(idx) };
                        }
                        metric.similarity(query, &vector).unwrap_or(0.0)
                    }
                };

                // Maintain top-k heap
                if heap.len() < k {
                    heap.push(std::cmp::Reverse((OrderedFloat(similarity), idx)));
                } else if let Some(&std::cmp::Reverse((OrderedFloat(min_sim), _))) = heap.peek() {
                    if similarity > min_sim {
                        heap.pop();
                        heap.push(std::cmp::Reverse((OrderedFloat(similarity), idx)));
                    }
                }
            }
        }

        // Extract results
        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(sim), idx))| (idx, sim))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Parallel search for large datasets
    fn search_parallel(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let count = self.hot_data.vectors_soa.count.load(Ordering::Relaxed);
        let chunk_size = (count / num_threads()).max(100);

        // Parallel search across chunks
        let partial_results: Vec<Vec<(usize, f32)>> = (0..count)
            .collect::<Vec<_>>()
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk.len()).min(count);

                let mut local_results = Vec::with_capacity(k);

                for idx in start..end {
                    // Similar to sequential but for this chunk
                    let similarity = self.compute_similarity_at(query, idx);

                    if local_results.len() < k {
                        local_results.push((idx, similarity));
                        if local_results.len() == k {
                            local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        }
                    } else if similarity > local_results[k - 1].1 {
                        local_results[k - 1] = (idx, similarity);
                        local_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    }
                }

                local_results
            })
            .collect();

        // Merge partial results
        let mut final_results = Vec::with_capacity(k);
        for partial in partial_results {
            for (idx, sim) in partial {
                if final_results.len() < k {
                    final_results.push((idx, sim));
                    if final_results.len() == k {
                        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    }
                } else if sim > final_results[k - 1].1 {
                    final_results[k - 1] = (idx, sim);
                    final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                }
            }
        }

        final_results
    }

    /// Compute similarity for a specific index
    fn compute_similarity_at(&self, query: &[f32], idx: usize) -> f32 {
        use oxirs_core::simd::SimdOps;

        let metric = self.config.similarity_config.primary_metric;

        match metric {
            crate::similarity::SimilarityMetric::Cosine => {
                let mut dot_product = 0.0f32;

                for dim in 0..self.hot_data.vectors_soa.dimensions {
                    let vec_val = unsafe { *self.hot_data.vectors_soa.data[dim].ptr.add(idx) };
                    dot_product += query[dim] * vec_val;
                }

                let query_norm = Self::compute_norm(query);
                let vec_norm = self.hot_data.norms.as_slice()[idx];
                dot_product / (query_norm * vec_norm + 1e-8)
            }
            _ => {
                // Reconstruct vector for other metrics
                let mut vector = vec![0.0f32; self.hot_data.vectors_soa.dimensions];
                for dim in 0..self.hot_data.vectors_soa.dimensions {
                    vector[dim] = unsafe { *self.hot_data.vectors_soa.data[dim].ptr.add(idx) };
                }
                metric.similarity(query, &vector).unwrap_or(0.0)
            }
        }
    }
}

impl VectorIndex for CacheFriendlyVectorIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        let vector_f32 = vector.as_f32();

        // Initialize SoA on first insert
        if self.hot_data.vectors_soa.dimensions == 0 {
            self.initialize_soa(vector_f32.len());
        } else if vector_f32.len() != self.hot_data.vectors_soa.dimensions {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        // Add to hot data
        self.add_to_soa(&vector_f32);
        let norm = Self::compute_norm(&vector_f32);
        self.hot_data.norms.push(norm);

        let uri_idx = self.cold_data.uris.len() as u32;
        self.hot_data.uri_indices.push(uri_idx);

        // Add to cold data
        self.cold_data.uris.push(uri);
        self.cold_data.metadata.push(vector.metadata);

        // Update count
        self.hot_data
            .vectors_soa
            .count
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();

        // Update statistics
        self.stats.searches.fetch_add(1, Ordering::Relaxed);

        // Choose search strategy based on dataset size
        let count = self.hot_data.vectors_soa.count.load(Ordering::Relaxed);
        let results = if self.config.parallel_search && count > self.config.parallel_threshold {
            self.search_parallel(&query_f32, k)
        } else {
            self.search_sequential(&query_f32, k)
        };

        // Convert indices to URIs
        Ok(results
            .into_iter()
            .map(|(idx, sim)| {
                let uri_idx = self.hot_data.uri_indices.as_slice()[idx] as usize;
                (self.cold_data.uris[uri_idx].clone(), sim)
            })
            .collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();
        let count = self.hot_data.vectors_soa.count.load(Ordering::Relaxed);

        let mut results = Vec::new();

        for idx in 0..count {
            let similarity = self.compute_similarity_at(&query_f32, idx);

            if similarity >= threshold {
                let uri_idx = self.hot_data.uri_indices.as_slice()[idx] as usize;
                results.push((self.cold_data.uris[uri_idx].clone(), similarity));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        // This requires reconstructing the vector from SoA layout
        // For now, return None as this is primarily an optimization for search
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec() {
        let mut vec = AlignedVec::<f32>::new(10);

        for i in 0..20 {
            vec.push(i as f32);
        }

        assert_eq!(vec.len, 20);
        assert!(vec.capacity >= 20);

        let slice = vec.as_slice();
        for (i, &val) in slice.iter().enumerate() {
            assert_eq!(val, i as f32);
        }
    }

    #[test]
    fn test_cache_friendly_index() {
        let mut config = IndexConfig::default();
        // Use Euclidean distance for this test since all vectors have same cosine similarity
        config.similarity_config.primary_metric = crate::similarity::SimilarityMetric::Euclidean;
        let mut index = CacheFriendlyVectorIndex::new(config);

        // Insert test vectors
        for i in 0..100 {
            let vector = Vector::new(vec![i as f32; 128]);
            index.insert(format!("vec_{}", i), vector).unwrap();
        }

        // Search for nearest neighbors
        let query = Vector::new(vec![50.0; 128]);
        let results = index.search_knn(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // The most similar should be vec_50 (exact match with Euclidean distance)
        assert_eq!(results[0].0, "vec_50");
    }
}
