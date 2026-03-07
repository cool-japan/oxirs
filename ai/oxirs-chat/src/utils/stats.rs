//! Statistical utility functions for RAG and ranking
//!
//! This module provides statistical functions used throughout the chat system,
//! particularly for document ranking and relevance scoring.

use anyhow::Result;

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        anyhow::bail!("Vectors must have the same length");
    }

    if a.is_empty() {
        return Ok(0.0);
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return Ok(0.0);
    }

    Ok(dot_product / (norm_a * norm_b))
}

/// Compute the mean (average) of a slice of values
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute the standard deviation of a slice of values
pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean_val = mean(values);
    let variance = values
        .iter()
        .map(|x| {
            let diff = x - mean_val;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;

    variance.sqrt()
}

/// Normalize values to [0, 1] range using min-max normalization
pub fn normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let min_val = values
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("collection should not be empty");
    let max_val = values
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("collection should not be empty");

    if (max_val - min_val).abs() < 1e-10 {
        // All values are the same
        return vec![0.5; values.len()];
    }

    values
        .iter()
        .map(|x| (x - min_val) / (max_val - min_val))
        .collect()
}

/// Z-score normalization (standardization)
pub fn standardize(values: &[f64]) -> Vec<f64> {
    if values.len() < 2 {
        return values.to_vec();
    }

    let mean_val = mean(values);
    let std_val = std_dev(values);

    if std_val < 1e-10 {
        return vec![0.0; values.len()];
    }

    values.iter().map(|x| (x - mean_val) / std_val).collect()
}

/// Compute Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        anyhow::bail!("Vectors must have the same length");
    }

    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    Ok(sum_sq.sqrt())
}

/// Compute Manhattan distance between two vectors
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        anyhow::bail!("Vectors must have the same length");
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim > 0.9 && sim < 1.0);

        // Orthogonal vectors
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        let sim2 = cosine_similarity(&c, &d).unwrap();
        assert!((sim2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_std_dev() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = std_dev(&values);
        assert!((std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_normalize() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize(&values);
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[4], 1.0);
        assert!((normalized[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b).unwrap();
        assert_eq!(dist, 5.0);
    }
}
