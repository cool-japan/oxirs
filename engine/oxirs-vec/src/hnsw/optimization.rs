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

        // Use SIMD operations for distance calculations
        match self.config().metric {
            SimilarityMetric::Cosine | SimilarityMetric::DotProduct => {
                // Use scirs2-core's SIMD dot product for cosine similarity
                for &candidate_id in candidates {
                    if let Some(node) = self.nodes().get(candidate_id) {
                        let candidate_f32 = node.vector.as_f32();
                        let candidate_array = Array1::from_vec(candidate_f32);

                        // Use SIMD-accelerated dot product
                        let dot_prod = simd_dot_f32(&query_array.view(), &candidate_array.view());

                        if self.config().metric == SimilarityMetric::Cosine {
                            // For cosine similarity, normalize
                            let query_mag = (query_array.iter().map(|x| x * x).sum::<f32>()).sqrt();
                            let candidate_mag =
                                (candidate_array.iter().map(|x| x * x).sum::<f32>()).sqrt();

                            if query_mag > 0.0 && candidate_mag > 0.0 {
                                let similarity = dot_prod / (query_mag * candidate_mag);
                                distances.push(1.0 - similarity); // Convert to distance
                            } else {
                                distances.push(f32::INFINITY);
                            }
                        } else {
                            // For dot product, use directly as distance
                            distances.push(-dot_prod); // Negate for distance semantics
                        }
                    } else {
                        distances.push(f32::INFINITY);
                    }
                }
            }
            _ => {
                // For other metrics, use standard calculation
                return self.cpu_batch_distance_calculation(query, candidates);
            }
        }

        self.update_simd_stats(candidates.len() as u64);
        Ok(distances)
    }

    /// Prefetch memory for improved cache performance
    pub fn prefetch_nodes(&self, node_ids: &[usize]) {
        if !self.config().enable_prefetch {
            return;
        }

        // Placeholder for memory prefetching
        // Real implementation would use:
        // - CPU cache prefetch instructions
        // - Strategic memory access patterns
        // - Prefetch lookahead based on search patterns

        for &node_id in node_ids.iter().take(self.config().prefetch_distance) {
            if node_id < self.nodes().len() {
                // Prefetch would happen here
                // std::arch::x86_64::_mm_prefetch(...) on x86_64
            }
        }
    }

    /// Optimize memory layout for cache-friendly access
    pub fn optimize_memory_layout(&mut self) -> Result<()> {
        if !self.config().cache_friendly_layout {
            return Ok(());
        }

        // Placeholder for memory layout optimization
        // Real implementation would:
        // - Reorder nodes based on access patterns
        // - Pack frequently accessed data together
        // - Align data structures to cache line boundaries

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
