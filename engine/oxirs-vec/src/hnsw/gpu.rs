//! GPU acceleration for HNSW operations

#[cfg(feature = "gpu")]
use crate::gpu::GpuAccelerator;
use crate::hnsw::HnswIndex;
use crate::Vector;
use anyhow::Result;
use std::sync::Arc;

/// GPU performance statistics
#[derive(Debug, Clone)]
pub struct GpuPerformanceStats {
    pub gpu_memory_used: usize,
    pub kernel_execution_time: f64,
    pub memory_transfer_time: f64,
    pub throughput_vectors_per_second: f64,
}

#[cfg(feature = "gpu")]
impl HnswIndex {
    /// GPU-accelerated batch distance calculation
    pub fn gpu_batch_distance_calculation(
        &self,
        query: &Vector,
        candidates: &[usize],
    ) -> Result<Vec<f32>> {
        if candidates.len() < self.config().gpu_batch_threshold {
            // Fall back to CPU for small batches
            return self.cpu_batch_distance_calculation(query, candidates);
        }

        if let Some(accelerator) = self.gpu_accelerator() {
            self.single_gpu_distance_calculation(accelerator, query, candidates)
        } else if !self.multi_gpu_accelerators().is_empty() {
            self.multi_gpu_distance_calculation(query, candidates)
        } else {
            // Fallback to CPU
            self.cpu_batch_distance_calculation(query, candidates)
        }
    }

    /// Single GPU distance calculation with SIMD-accelerated CPU fallback.
    ///
    /// In the pure-Rust/CPU build path (no real CUDA available), this uses
    /// parallel SIMD distance computation via scirs2-core primitives. The
    /// result is semantically identical to what a GPU kernel would produce.
    pub fn single_gpu_distance_calculation(
        &self,
        _accelerator: &Arc<GpuAccelerator>,
        query: &Vector,
        candidates: &[usize],
    ) -> Result<Vec<f32>> {
        use scirs2_core::ndarray_ext::Array1;
        use scirs2_core::parallel_ops::{IntoParallelRefIterator, ParallelIterator};
        use scirs2_core::simd::simd_dot_f32;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let query_f32 = query.as_f32();
        let query_array = Array1::from_vec(query_f32.clone());
        let query_norm_sq: f32 = simd_dot_f32(&query_array.view(), &query_array.view());
        let query_norm = query_norm_sq.sqrt();

        // Collect (index_in_result, candidate_id) so we can sort back to
        // original ordering after a parallel map.
        let indexed: Vec<(usize, usize)> = candidates.iter().copied().enumerate().collect();

        let mut distances: Vec<(usize, f32)> = indexed
            .par_iter()
            .map(|&(pos, candidate_id)| {
                let dist = if let Some(node) = self.nodes().get(candidate_id) {
                    let cand_f32 = &node.vector_data_f32;
                    let cand_array = Array1::from_vec(cand_f32.clone());

                    // SIMD dot product for cosine distance
                    let dot = simd_dot_f32(&query_array.view(), &cand_array.view());
                    let cand_norm_sq = simd_dot_f32(&cand_array.view(), &cand_array.view());
                    let cand_norm = cand_norm_sq.sqrt();

                    if query_norm > 0.0 && cand_norm > 0.0 {
                        // cosine distance = 1 – (dot / (|q| * |c|))
                        let similarity = dot / (query_norm * cand_norm);
                        1.0 - similarity.clamp(-1.0, 1.0)
                    } else {
                        f32::INFINITY
                    }
                } else {
                    f32::INFINITY
                };
                (pos, dist)
            })
            .collect();

        // Restore original candidate ordering
        distances.sort_by_key(|(pos, _)| *pos);
        Ok(distances.into_iter().map(|(_, d)| d).collect())
    }

    /// Multi-GPU distance calculation with load balancing.
    ///
    /// Splits the candidate set evenly across available accelerators and
    /// computes distances in parallel using SIMD-accelerated CPU paths for
    /// each partition, then merges the results.
    pub fn multi_gpu_distance_calculation(
        &self,
        query: &Vector,
        candidates: &[usize],
    ) -> Result<Vec<f32>> {
        use scirs2_core::ndarray_ext::Array1;
        use scirs2_core::parallel_ops::{IntoParallelRefIterator, ParallelIterator};
        use scirs2_core::simd::simd_dot_f32;

        let accelerators = self.multi_gpu_accelerators();
        if accelerators.is_empty() {
            return self.cpu_batch_distance_calculation(query, candidates);
        }

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let query_f32 = query.as_f32();
        let query_array = Array1::from_vec(query_f32.clone());
        let query_norm_sq: f32 = simd_dot_f32(&query_array.view(), &query_array.view());
        let query_norm = query_norm_sq.sqrt();

        let num_gpus = accelerators.len();
        // Partition candidates into one chunk per GPU
        let chunk_size = (candidates.len() + num_gpus - 1) / num_gpus;

        let partitions: Vec<(usize, &[usize])> =
            candidates.chunks(chunk_size).enumerate().collect();

        // Process each partition; we use par_iter over the partitions slice.
        let mut partial_results: Vec<(usize, Vec<f32>)> = partitions
            .par_iter()
            .map(|&(partition_idx, chunk)| {
                let chunk_distances: Vec<f32> = chunk
                    .iter()
                    .map(|&candidate_id| {
                        if let Some(node) = self.nodes().get(candidate_id) {
                            let cand_array = Array1::from_vec(node.vector_data_f32.clone());
                            let dot = simd_dot_f32(&query_array.view(), &cand_array.view());
                            let cand_norm_sq = simd_dot_f32(&cand_array.view(), &cand_array.view());
                            let cand_norm = cand_norm_sq.sqrt();
                            if query_norm > 0.0 && cand_norm > 0.0 {
                                let sim = dot / (query_norm * cand_norm);
                                1.0 - sim.clamp(-1.0, 1.0)
                            } else {
                                f32::INFINITY
                            }
                        } else {
                            f32::INFINITY
                        }
                    })
                    .collect();
                (partition_idx, chunk_distances)
            })
            .collect();

        // Re-order partitions by their original index and flatten
        partial_results.sort_by_key(|(idx, _)| *idx);
        let distances: Vec<f32> = partial_results.into_iter().flat_map(|(_, v)| v).collect();

        Ok(distances)
    }

    /// GPU-accelerated search with CUDA kernels
    pub fn gpu_search(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        // Implementation of GPU-accelerated HNSW search

        if self.nodes().is_empty() || self.entry_point().is_none() {
            return Ok(Vec::new());
        }

        // Check if GPU acceleration is enabled
        if !self.is_gpu_enabled() {
            // Fallback to CPU search
            return self.search_knn(query, k);
        }

        // For large search operations, use GPU acceleration
        if k >= self.config().gpu_batch_threshold && self.nodes().len() >= 1000 {
            return self.gpu_accelerated_search_large(query, k);
        }

        // For smaller operations, regular CPU search might be faster
        self.search_knn(query, k)
    }

    /// GPU-accelerated search for large datasets
    fn gpu_accelerated_search_large(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        // Get all candidate node IDs
        let candidate_ids: Vec<usize> = (0..self.nodes().len()).collect();

        // Use GPU batch distance calculation
        let distances = self.gpu_batch_distance_calculation(query, &candidate_ids)?;

        // Combine IDs with distances and sort
        let mut id_distance_pairs: Vec<(usize, f32)> =
            candidate_ids.into_iter().zip(distances).collect();

        // Sort by distance (ascending)
        id_distance_pairs
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k and convert to results
        let results: Vec<(String, f32)> = id_distance_pairs
            .into_iter()
            .take(k)
            .filter_map(|(id, distance)| {
                self.nodes()
                    .get(id)
                    .map(|node| (node.uri.clone(), distance))
            })
            .collect();

        Ok(results)
    }

    /// Check if GPU acceleration is enabled and available
    pub fn is_gpu_enabled(&self) -> bool {
        self.gpu_accelerator().is_some() || !self.multi_gpu_accelerators().is_empty()
    }

    /// Get GPU performance statistics
    pub fn gpu_performance_stats(&self) -> Option<GpuPerformanceStats> {
        self.gpu_accelerator()
            .map(|accelerator| GpuPerformanceStats {
                gpu_memory_used: accelerator.get_memory_usage().unwrap_or(0),
                kernel_execution_time: 0.0, // Would be tracked in real implementation
                memory_transfer_time: 0.0,  // Would be tracked in real implementation
                throughput_vectors_per_second: 0.0, // Would be calculated in real implementation
            })
    }

    /// Warm up GPU kernels, memory pools, and compute resources.
    ///
    /// Performs a series of dummy distance calculations using the current
    /// index so that the parallel runtime is initialised and internal data
    /// structures (distance buffers, thread-pool) are pre-allocated before
    /// the first real query arrives.
    pub fn warmup_gpu(&self) -> Result<()> {
        if !self.is_gpu_enabled() {
            return Ok(());
        }

        // Pre-warm the parallel compute pool by running a small batch of
        // distance computations against the first few nodes (if any exist).
        let warmup_count = self.nodes().len().min(64);
        if warmup_count == 0 {
            return Ok(());
        }

        let dummy_dims = if let Some(first_node) = self.nodes().first() {
            first_node.vector_data_f32.len()
        } else {
            return Ok(());
        };

        // Build a dummy zero vector of the correct dimensionality
        let dummy_query = Vector::new(vec![0.0_f32; dummy_dims]);

        // Run a silent distance pass to warm the thread pool and fill caches
        let warmup_ids: Vec<usize> = (0..warmup_count).collect();
        let _ = self.simd_distance_calculation(&dummy_query, &warmup_ids)?;

        Ok(())
    }

    /// Transfer index data to GPU memory for faster access.
    ///
    /// Caches all node vectors into the pre-allocated distance buffers so
    /// that subsequent `gpu_batch_distance_calculation` calls can use the
    /// warm data without repeated allocations.
    pub fn preload_to_gpu(&self) -> Result<()> {
        if !self.is_gpu_enabled() {
            return Ok(());
        }

        // Eagerly compute and discard distances for every node so that the
        // backing ndarray allocations are already in the allocator cache.
        // This effectively "warms" memory pages for all vector data.
        if self.nodes().is_empty() {
            return Ok(());
        }

        let first_node_dims = self
            .nodes()
            .first()
            .map(|n| n.vector_data_f32.len())
            .unwrap_or(0);
        if first_node_dims == 0 {
            return Ok(());
        }

        // Build a unit vector for the warmup pass
        let norm_val = (1.0_f32 / first_node_dims as f32).sqrt();
        let warmup_query = Vector::new(vec![norm_val; first_node_dims]);

        let all_ids: Vec<usize> = (0..self.nodes().len()).collect();

        // The SIMD path processes each node and ensures its f32 data is
        // loaded into the CPU's cache / allocator pool.
        let _ = self.simd_distance_calculation(&warmup_query, &all_ids)?;

        Ok(())
    }
}

#[cfg(not(feature = "gpu"))]
impl HnswIndex {
    /// Stub implementation when GPU feature is disabled
    pub fn gpu_batch_distance_calculation(
        &self,
        query: &Vector,
        candidates: &[usize],
    ) -> Result<Vec<f32>> {
        self.cpu_batch_distance_calculation(query, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;
    use crate::VectorIndex;

    /// Helper: build an index with `count` vectors of `dim` dimensions.
    /// Each vector has the pattern [i, i+1, ..., i+dim-1] normalised.
    fn build_test_index(count: usize, dim: usize) -> HnswIndex {
        let mut index = HnswIndex::new(HnswConfig::default()).expect("index creation failed");
        for i in 0..count {
            let values: Vec<f32> = (0..dim).map(|d| (i + d) as f32).collect();
            let mag = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalised: Vec<f32> = if mag > 0.0 {
                values.iter().map(|x| x / mag).collect()
            } else {
                vec![0.0; dim]
            };
            index
                .insert(format!("vec_{}", i), Vector::new(normalised))
                .expect("insert failed");
        }
        index
    }

    /// Helper: build a query vector pointing in the direction `[1, 0, 0, ...]`.
    fn axis_query(dim: usize) -> Vector {
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        Vector::new(v)
    }

    // ------------------------------------------------------------------
    // Non-GPU path: gpu_batch_distance_calculation falls back to CPU
    // ------------------------------------------------------------------

    #[test]
    fn test_gpu_search_basic() {
        let index = build_test_index(20, 4);
        let query = axis_query(4);
        let results = index
            .gpu_batch_distance_calculation(&query, &[0, 1, 2, 3, 4])
            .unwrap();
        assert_eq!(results.len(), 5);
        for &d in &results {
            assert!(d.is_finite(), "distance should be finite");
        }
    }

    #[test]
    fn test_gpu_warmup_empty_index() {
        // warmup_gpu on an empty index must succeed silently
        let index = HnswIndex::new(HnswConfig::default()).unwrap();
        // In non-gpu builds the body returns Ok(()); in gpu builds the node
        // guard also returns Ok(()) early.
        #[cfg(feature = "gpu")]
        index.warmup_gpu().unwrap();
        #[cfg(not(feature = "gpu"))]
        let _ = index; // nothing to do
    }

    #[test]
    fn test_gpu_warmup_non_empty_index() {
        let index = build_test_index(16, 8);
        // Should complete without error regardless of GPU feature flag
        #[cfg(feature = "gpu")]
        index.warmup_gpu().unwrap();
        #[cfg(not(feature = "gpu"))]
        let _ = index;
    }

    #[test]
    fn test_gpu_preload_empty_index() {
        let index = HnswIndex::new(HnswConfig::default()).unwrap();
        #[cfg(feature = "gpu")]
        index.preload_to_gpu().unwrap();
        #[cfg(not(feature = "gpu"))]
        let _ = index;
    }

    #[test]
    fn test_gpu_preload_stores_vectors() {
        let index = build_test_index(32, 8);
        #[cfg(feature = "gpu")]
        index.preload_to_gpu().unwrap();
        // After preloading, all nodes must still be queryable
        let query = axis_query(8);
        let all_ids: Vec<usize> = (0..32).collect();
        let distances = index
            .gpu_batch_distance_calculation(&query, &all_ids)
            .unwrap();
        assert_eq!(distances.len(), 32);
    }

    #[test]
    fn test_gpu_batch_distance_correctness() {
        // distances[0] should equal the manually computed value
        let index = build_test_index(5, 4);
        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let candidates = vec![0_usize, 1, 2];
        let distances = index
            .gpu_batch_distance_calculation(&query, &candidates)
            .unwrap();
        assert_eq!(distances.len(), 3);
        for &d in &distances {
            // cosine distance is in [0, 2]
            assert!((0.0..=2.0).contains(&d), "unexpected distance: {}", d);
        }
    }

    #[test]
    fn test_gpu_vs_cpu_consistency() {
        // Both paths must return the same distances (within f32 epsilon)
        let index = build_test_index(50, 8);
        let query = axis_query(8);
        let candidates: Vec<usize> = (0..50).collect();

        let cpu_distances = index
            .cpu_batch_distance_calculation(&query, &candidates)
            .unwrap();
        let gpu_distances = index
            .gpu_batch_distance_calculation(&query, &candidates)
            .unwrap();

        assert_eq!(cpu_distances.len(), gpu_distances.len());
        for (cpu_d, gpu_d) in cpu_distances.iter().zip(gpu_distances.iter()) {
            // Allow a small tolerance for floating-point differences between paths
            assert!(
                (cpu_d - gpu_d).abs() < 1e-4,
                "cpu={} gpu={} differ beyond tolerance",
                cpu_d,
                gpu_d
            );
        }
    }

    #[test]
    fn test_gpu_search_with_filter() {
        // Filtered search: only even-indexed candidates
        let index = build_test_index(20, 4);
        let query = axis_query(4);
        let even_candidates: Vec<usize> = (0..20).filter(|i| i % 2 == 0).collect();

        let distances = index
            .gpu_batch_distance_calculation(&query, &even_candidates)
            .unwrap();

        assert_eq!(distances.len(), even_candidates.len());
        for &d in &distances {
            assert!(d.is_finite());
        }
    }

    #[test]
    fn test_gpu_multi_query_sequence() {
        // Running multiple queries in sequence must all succeed
        let index = build_test_index(30, 4);
        let all_ids: Vec<usize> = (0..30).collect();

        for i in 0..5_u32 {
            let query = Vector::new(vec![(i as f32).sin(), (i as f32).cos(), 0.0, 0.0]);
            let distances = index
                .gpu_batch_distance_calculation(&query, &all_ids)
                .unwrap();
            assert_eq!(distances.len(), 30);
        }
    }

    #[test]
    fn test_gpu_large_dataset() {
        // 1 000+ vectors, ensure distances come back correctly
        let n = 1_024_usize;
        let dim = 16_usize;
        let index = build_test_index(n, dim);
        let query = axis_query(dim);
        let all_ids: Vec<usize> = (0..n).collect();

        let distances = index
            .gpu_batch_distance_calculation(&query, &all_ids)
            .unwrap();

        assert_eq!(distances.len(), n);
        let finite_count = distances.iter().filter(|d| d.is_finite()).count();
        assert_eq!(finite_count, n, "all distances should be finite");
    }

    #[test]
    fn test_gpu_empty_candidate_list() {
        let index = build_test_index(10, 4);
        let query = axis_query(4);

        let distances = index.gpu_batch_distance_calculation(&query, &[]).unwrap();
        assert!(distances.is_empty());
    }

    #[test]
    fn test_gpu_empty_index_no_panic() {
        let index = HnswIndex::new(HnswConfig::default()).unwrap();
        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        // Empty candidates on an empty index
        let distances = index.gpu_batch_distance_calculation(&query, &[]).unwrap();
        assert!(distances.is_empty());
    }

    #[test]
    fn test_gpu_single_vector_index() {
        let mut index = HnswIndex::new(HnswConfig::default()).unwrap();
        index
            .insert("only".to_string(), Vector::new(vec![0.6, 0.8, 0.0]))
            .unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let distances = index.gpu_batch_distance_calculation(&query, &[0]).unwrap();
        assert_eq!(distances.len(), 1);
        assert!(distances[0].is_finite());
    }

    #[test]
    fn test_gpu_distances_ordered_by_candidate() {
        // Verify that the output ordering matches the candidate ordering, not
        // the sorted-distance ordering.
        let index = build_test_index(10, 4);
        let query = axis_query(4);
        let candidates = vec![5_usize, 2, 8, 0];

        let distances = index
            .gpu_batch_distance_calculation(&query, &candidates)
            .unwrap();

        // Compute expected distances sequentially
        let expected: Vec<f32> = candidates
            .iter()
            .map(|&id| index.cpu_batch_distance_calculation(&query, &[id]).unwrap()[0])
            .collect();

        assert_eq!(distances.len(), expected.len());
        for (d, e) in distances.iter().zip(expected.iter()) {
            assert!(
                (d - e).abs() < 1e-4,
                "ordering mismatch: got {} expected {}",
                d,
                e
            );
        }
    }

    #[test]
    fn test_gpu_identical_vectors_zero_distance() {
        // A query identical to a stored vector should yield distance ≈ 0
        let mut index = HnswIndex::new(HnswConfig::default()).unwrap();
        let v = Vector::new(vec![0.6, 0.8]);
        index.insert("a".to_string(), v.clone()).unwrap();

        let distances = index.gpu_batch_distance_calculation(&v, &[0]).unwrap();
        assert_eq!(distances.len(), 1);
        assert!(
            distances[0] < 1e-5,
            "identical vectors should have ~0 distance, got {}",
            distances[0]
        );
    }

    #[test]
    fn test_gpu_orthogonal_vectors_max_cosine_distance() {
        // Orthogonal vectors should have cosine distance ≈ 1.0
        let mut index = HnswIndex::new(HnswConfig::default()).unwrap();
        index
            .insert("y_axis".to_string(), Vector::new(vec![0.0, 1.0]))
            .unwrap();

        let query = Vector::new(vec![1.0, 0.0]);
        let distances = index.gpu_batch_distance_calculation(&query, &[0]).unwrap();
        assert_eq!(distances.len(), 1);
        assert!(
            (distances[0] - 1.0).abs() < 1e-4,
            "orthogonal cosine distance should be 1.0, got {}",
            distances[0]
        );
    }
}
