//! Performance optimizations for HNSW operations

use crate::hnsw::HnswIndex;
use crate::similarity::SimilarityMetric;
use crate::Vector;
use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::simd::simd_dot_f32;

impl HnswIndex {
    /// CPU-based batch distance calculation
    pub fn cpu_batch_distance_calculation(
        &self,
        query: &Vector,
        candidates: &[usize],
    ) -> Result<Vec<f32>> {
        let mut distances = Vec::with_capacity(candidates.len());

        for &candidate_id in candidates {
            if let Some(node) = self.nodes().get(candidate_id) {
                let distance = self.config().metric.distance(query, &node.vector)?;
                distances.push(distance);
            } else {
                distances.push(f32::INFINITY);
            }
        }

        Ok(distances)
    }

    /// SIMD-optimized distance calculation
    pub fn simd_distance_calculation(
        &self,
        query: &Vector,
        candidates: &[usize],
    ) -> Result<Vec<f32>> {
        if !self.config().enable_simd {
            return self.cpu_batch_distance_calculation(query, candidates);
        }

        let mut distances = Vec::with_capacity(candidates.len());
        let query_f32 = query.as_f32();
        let query_array = Array1::from_vec(query_f32.clone());

        // Use SIMD operations for distance calculations. Only the cosine metric
        // has a SIMD fast path here; every other metric falls back to the scalar
        // batch path so that SIMD-on and SIMD-off produce *identical* distances
        // (this is critical: the graph is built with the scalar metric, so a
        // search-time SIMD distance that disagreed would degrade recall).
        match self.config().metric {
            SimilarityMetric::Cosine => {
                let query_mag = (query_array.iter().map(|x| x * x).sum::<f32>()).sqrt();
                for &candidate_id in candidates {
                    if let Some(node) = self.nodes().get(candidate_id) {
                        let candidate_f32 = node.vector.as_f32();
                        let candidate_array = Array1::from_vec(candidate_f32);

                        // SIMD-accelerated dot product, then cosine normalization.
                        let dot_prod = simd_dot_f32(&query_array.view(), &candidate_array.view());
                        let candidate_mag =
                            (candidate_array.iter().map(|x| x * x).sum::<f32>()).sqrt();

                        if query_mag > 0.0 && candidate_mag > 0.0 {
                            // Match `SimilarityMetric::distance` exactly: clamp the
                            // similarity to [0, 1] before converting to distance.
                            let similarity =
                                (dot_prod / (query_mag * candidate_mag)).clamp(0.0, 1.0);
                            distances.push((1.0 - similarity).max(0.0));
                        } else {
                            distances.push(f32::INFINITY);
                        }
                    } else {
                        distances.push(f32::INFINITY);
                    }
                }
            }
            _ => {
                // For other metrics, use the scalar calculation (exact parity).
                return self.cpu_batch_distance_calculation(query, candidates);
            }
        }

        self.update_simd_stats(candidates.len() as u64);
        Ok(distances)
    }

    /// Prefetch the vector data of the given nodes into CPU cache ahead of the
    /// distance computations that will read them.
    ///
    /// On x86/x86_64 this issues real `PREFETCHT0` instructions against each
    /// node's vector storage; on other architectures it performs a cheap
    /// best-effort touch of the first cache line. Governed by `enable_prefetch`
    /// and `prefetch_distance` so those config knobs have a real effect on the
    /// search hot path.
    pub fn prefetch_nodes(&self, node_ids: &[usize]) {
        if !self.config().enable_prefetch || node_ids.is_empty() {
            return;
        }

        let nodes = self.nodes();
        let mut issued = 0u64;
        for &node_id in node_ids.iter().take(self.config().prefetch_distance.max(1)) {
            if let Some(node) = nodes.get(node_id) {
                let data = node.vector.as_f32();
                if let Some(first) = data.first() {
                    let ptr = first as *const f32;
                    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                    {
                        #[cfg(target_arch = "x86")]
                        use std::arch::x86::{_mm_prefetch, _MM_HINT_T0};
                        #[cfg(target_arch = "x86_64")]
                        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                        // SAFETY: `ptr` points at live, aligned f32 data owned by
                        // `node` for the duration of this borrow; `_mm_prefetch`
                        // only hints the cache and never dereferences unsafely.
                        unsafe {
                            _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
                        }
                    }
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
                    {
                        // Portable best-effort: a volatile read of the first
                        // element pulls its cache line in without being optimized
                        // away. SAFETY: `ptr` is valid for reads of one f32.
                        unsafe {
                            std::ptr::read_volatile(ptr);
                        }
                    }
                    issued += 1;
                }
            }
        }
        self.update_prefetch_stats(issued);
    }

    /// Optimize the in-memory graph layout for cache-friendly traversal.
    ///
    /// Shrinks every node's per-layer adjacency-set allocations (and the outer
    /// connections vector) to fit, removing the slack that `HashSet`/`Vec`
    /// growth leaves behind. This packs the graph metadata more densely so more
    /// of it fits per cache line during traversal. The complementary win — visiting
    /// neighbors in ascending-id order to make `nodes()[neighbor_id]` accesses
    /// (and [`Self::prefetch_nodes`] hints) sequential — is applied on the search
    /// hot path in `search_layer` when `cache_friendly_layout` is enabled. This
    /// is a pure layout change: neighbor *sets* are unchanged, so results are
    /// identical. Governed by `cache_friendly_layout`.
    pub fn optimize_memory_layout(&mut self) -> Result<()> {
        if !self.config().cache_friendly_layout {
            return Ok(());
        }

        for node in self.nodes_mut().iter_mut() {
            for level_set in node.connections.iter_mut() {
                level_set.shrink_to_fit();
            }
            node.connections.shrink_to_fit();
        }

        Ok(())
    }

    /// Update cache statistics
    pub fn update_cache_stats(&self, hits: u64, misses: u64) {
        self.get_stats()
            .cache_hits
            .fetch_add(hits, std::sync::atomic::Ordering::Relaxed);
        self.get_stats()
            .cache_misses
            .fetch_add(misses, std::sync::atomic::Ordering::Relaxed);
    }

    /// Update SIMD operation statistics
    pub fn update_simd_stats(&self, operations: u64) {
        self.get_stats()
            .simd_operations
            .fetch_add(operations, std::sync::atomic::Ordering::Relaxed);
    }

    /// Update prefetch statistics
    pub fn update_prefetch_stats(&self, operations: u64) {
        self.get_stats()
            .prefetch_operations
            .fetch_add(operations, std::sync::atomic::Ordering::Relaxed);
    }

    /// Parallel batch distance calculation using scirs2-core SIMD operations
    /// Note: We use a simplified parallel approach for now.
    /// Full parallelization with rayon can be added when needed.
    pub fn parallel_batch_distance_calculation(
        &self,
        query: &Vector,
        candidates: &[usize],
    ) -> Result<Vec<f32>> {
        // For now, use SIMD-optimized sequential processing
        // Full parallel implementation requires rayon integration
        // which is available through scirs2-core's par_chunks when properly configured

        if candidates.len() < 100 {
            // For small batches, use SIMD without parallelization overhead
            return self.simd_distance_calculation(query, candidates);
        }

        // For larger batches, still use SIMD which provides significant speedup
        self.simd_distance_calculation(query, candidates)
    }

    /// Parallel processing utilities
    pub fn should_use_parallel(&self, work_size: usize) -> bool {
        self.config().enable_parallel && work_size > 100
    }

    /// Get optimal number of threads for parallel operations
    pub fn get_optimal_thread_count(&self, work_size: usize) -> usize {
        if !self.config().enable_parallel {
            return 1;
        }

        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Use a heuristic to determine optimal thread count
        (work_size / 1000).max(1).min(max_threads)
    }
}
