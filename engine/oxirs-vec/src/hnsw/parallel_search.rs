//! Parallel search operations for HNSW using SciRS2-Core
//!
//! This module provides parallel search capabilities using SciRS2's
//! parallel processing primitives for improved throughput.

use crate::hnsw::{Candidate, HnswIndex};
use crate::Vector;
use anyhow::Result;
use scirs2_core::parallel_ops::{IntoParallelRefIterator, ParallelIterator};

impl HnswIndex {
    /// Parallel batch search for multiple queries
    /// Uses Rayon's parallel iterators for improved throughput
    pub fn parallel_batch_search(
        &self,
        queries: &[Vector],
        k: usize,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        // Process queries in parallel using Rayon
        let results: Vec<Vec<(String, f32)>> = queries
            .par_iter()
            .map(|query| self.search_knn(query, k).unwrap_or_default())
            .collect();

        Ok(results)
    }

    /// Parallel candidate evaluation
    /// Evaluates multiple candidates in parallel for faster distance computation
    pub fn parallel_evaluate_candidates(
        &self,
        query: &Vector,
        candidate_ids: &[usize],
    ) -> Result<Vec<Candidate>> {
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Process candidates in parallel using Rayon
        let candidates: Vec<Candidate> = candidate_ids
            .par_iter()
            .filter_map(|&id| {
                self.calculate_distance(query, id)
                    .ok()
                    .map(|distance| Candidate::new(id, distance))
            })
            .collect();

        Ok(candidates)
    }

    /// Parallel range search across multiple query vectors
    pub fn parallel_range_search(
        &self,
        queries: &[Vector],
        radius: f32,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        // Process queries in parallel using Rayon
        let results: Vec<Vec<(String, f32)>> = queries
            .par_iter()
            .map(|query| self.range_search(query, radius).unwrap_or_default())
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;
    use crate::VectorIndex;

    #[test]
    fn test_parallel_batch_search() {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config).unwrap();

        // Add some vectors
        for i in 0..100 {
            let vector = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            index.insert(format!("vec_{}", i), vector).unwrap();
        }

        // Create multiple queries
        let queries = vec![
            Vector::new(vec![1.0, 2.0, 3.0]),
            Vector::new(vec![10.0, 20.0, 30.0]),
            Vector::new(vec![50.0, 100.0, 150.0]),
        ];

        // Perform parallel search
        let results = index.parallel_batch_search(&queries, 5).unwrap();

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.len() <= 5);
        }
    }

    #[test]
    fn test_parallel_range_search() {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new(config).unwrap();

        // Add some vectors
        for i in 0..50 {
            let vector = Vector::new(vec![i as f32, 0.0, 0.0]);
            index.insert(format!("vec_{}", i), vector).unwrap();
        }

        // Create multiple queries
        let queries = vec![
            Vector::new(vec![0.0, 0.0, 0.0]),
            Vector::new(vec![25.0, 0.0, 0.0]),
        ];

        // Perform parallel range search
        let results = index.parallel_range_search(&queries, 10.0).unwrap();

        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
    }
}
