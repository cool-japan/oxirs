//! SIMD-optimized distance calculations for HNSW using SciRS2-Core
//!
//! This module provides hardware-accelerated distance metric calculations
//! using SciRS2's SIMD operations for maximum performance.

use anyhow::Result;
use scirs2_core::ndarray_ext::ArrayView1;
use scirs2_core::simd::{simd_dot_f32, simd_sub_f32};

/// SIMD-accelerated cosine distance calculation
pub fn simd_cosine_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow::anyhow!(
            "Vector dimensions must match: {} != {}",
            a.len(),
            b.len()
        ));
    }

    // Convert slices to ArrayView1 for scirs2-core SIMD operations
    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Use SciRS2's SIMD-optimized dot product
    let dot = simd_dot_f32(&a_view, &b_view);

    // Calculate norms using SIMD
    let norm_a_sq = simd_dot_f32(&a_view, &a_view);
    let norm_b_sq = simd_dot_f32(&b_view, &b_view);

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(1.0); // Maximum distance for zero vectors
    }

    // Cosine similarity = dot / (norm_a * norm_b)
    // Cosine distance = 1 - similarity
    Ok(1.0 - (dot / (norm_a * norm_b)))
}

/// SIMD-accelerated Euclidean distance calculation
pub fn simd_euclidean(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow::anyhow!(
            "Vector dimensions must match: {} != {}",
            a.len(),
            b.len()
        ));
    }

    // Convert slices to ArrayView1
    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute difference using SIMD
    let diff = simd_sub_f32(&a_view, &b_view);
    let diff_view = diff.view();

    // Compute squared Euclidean distance using SIMD dot product
    let dist_sq = simd_dot_f32(&diff_view, &diff_view);

    Ok(dist_sq.sqrt())
}

/// SIMD-accelerated Manhattan distance calculation
pub fn simd_manhattan(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow::anyhow!(
            "Vector dimensions must match: {} != {}",
            a.len(),
            b.len()
        ));
    }

    // Convert slices to ArrayView1
    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // Compute difference using SIMD
    let diff = simd_sub_f32(&a_view, &b_view);

    // Compute sum of absolute values
    let dist: f32 = diff.iter().map(|x| x.abs()).sum();

    Ok(dist)
}

/// SIMD-accelerated dot product (inner product) distance
/// For normalized vectors, this is equivalent to cosine similarity
pub fn simd_dot_product_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow::anyhow!(
            "Vector dimensions must match: {} != {}",
            a.len(),
            b.len()
        ));
    }

    // Convert slices to ArrayView1
    let a_view = ArrayView1::from(a);
    let b_view = ArrayView1::from(b);

    // For distance metric, we return negative dot product
    // (smaller distance = more similar for positive dot products)
    let dot = simd_dot_f32(&a_view, &b_view);
    Ok(-dot)
}

/// Batch SIMD-accelerated distance calculations
/// Computes distances from a query vector to multiple vectors simultaneously
pub fn simd_batch_cosine_distances(query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut distances = Vec::with_capacity(vectors.len());

    for vector in vectors {
        let dist = simd_cosine_distance(query, vector)?;
        distances.push(dist);
    }

    Ok(distances)
}

/// Batch SIMD-accelerated Euclidean distance calculations
pub fn simd_batch_euclidean_distances(query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut distances = Vec::with_capacity(vectors.len());

    for vector in vectors {
        let dist = simd_euclidean(query, vector)?;
        distances.push(dist);
    }

    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = simd_cosine_distance(&a, &b).unwrap();
        assert!((dist - 0.0).abs() < 1e-6); // Identical vectors have distance 0

        let c = vec![0.0, 1.0, 0.0];
        let dist = simd_cosine_distance(&a, &c).unwrap();
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal vectors have distance 1
    }

    #[test]
    fn test_simd_euclidean() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let dist = simd_euclidean(&a, &b).unwrap();
        let expected = 3.0_f32.sqrt();
        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_simd_manhattan() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let dist = simd_manhattan(&a, &b).unwrap();
        assert!((dist - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_batch_distances() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let distances = simd_batch_cosine_distances(&query, &vectors).unwrap();
        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 0.0).abs() < 1e-6); // Same vector
        assert!((distances[1] - 1.0).abs() < 1e-6); // Orthogonal
        assert!((distances[2] - 1.0).abs() < 1e-6); // Orthogonal
    }
}
