//! Advanced distance and similarity metrics for vectors.
//!
//! This module provides statistical and domain-specific metrics including:
//! - Pearson correlation coefficient
//! - Spearman rank correlation
//! - Jaccard similarity for binary vectors
//! - Hamming distance for binary vectors
//! - Jensen-Shannon divergence
//! - Earth Mover's Distance (EMD)
//! - Wasserstein distance
//! - KL divergence
//! - Hellinger distance
//! - Mahalanobis distance

use std::collections::HashSet;

use anyhow::Result;

use crate::{Vector, VectorData};

impl Vector {
    /// Calculate Pearson correlation coefficient
    pub fn pearson_correlation(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let x = self.as_f32();
        let y = other.as_f32();
        let n = x.len() as f32;

        let sum_x: f32 = x.iter().sum();
        let sum_y: f32 = y.iter().sum();
        let sum_x2: f32 = x.iter().map(|v| v * v).sum();
        let sum_y2: f32 = y.iter().map(|v| v * v).sum();
        let sum_xy: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate Spearman rank correlation
    pub fn spearman_correlation(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let x = self.as_f32();
        let y = other.as_f32();

        // Convert to ranks
        let x_ranks = to_ranks(&x);
        let y_ranks = to_ranks(&y);

        // Calculate Pearson correlation on ranks
        let rank_vec_x = Vector::new(x_ranks);
        let rank_vec_y = Vector::new(y_ranks);

        rank_vec_x.pearson_correlation(&rank_vec_y)
    }

    /// Calculate Jaccard similarity for binary vectors
    pub fn jaccard_similarity(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        match (&self.values, &other.values) {
            (VectorData::Binary(a), VectorData::Binary(b)) => {
                let mut intersection = 0u32;
                let mut union = 0u32;

                for (byte_a, byte_b) in a.iter().zip(b) {
                    let and_result = byte_a & byte_b;
                    let or_result = byte_a | byte_b;
                    
                    intersection += and_result.count_ones();
                    union += or_result.count_ones();
                }

                if union == 0 {
                    Ok(0.0)
                } else {
                    Ok(intersection as f32 / union as f32)
                }
            }
            _ => {
                // Convert to binary using threshold of 0.5
                let a_binary = Vector::to_binary(&self.as_f32(), 0.5);
                let b_binary = Vector::to_binary(&other.as_f32(), 0.5);
                
                let vec_a = Vector::binary(a_binary);
                let vec_b = Vector::binary(b_binary);
                
                vec_a.jaccard_similarity(&vec_b)
            }
        }
    }

    /// Calculate Hamming distance for binary vectors
    pub fn hamming_distance(&self, other: &Vector) -> Result<u32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        match (&self.values, &other.values) {
            (VectorData::Binary(a), VectorData::Binary(b)) => {
                let mut distance = 0u32;
                
                for (byte_a, byte_b) in a.iter().zip(b) {
                    let xor_result = byte_a ^ byte_b;
                    distance += xor_result.count_ones();
                }
                
                Ok(distance)
            }
            _ => {
                // Convert to binary and calculate
                let a_binary = Vector::to_binary(&self.as_f32(), 0.5);
                let b_binary = Vector::to_binary(&other.as_f32(), 0.5);
                
                let vec_a = Vector::binary(a_binary);
                let vec_b = Vector::binary(b_binary);
                
                vec_a.hamming_distance(&vec_b)
            }
        }
    }

    /// Calculate Jensen-Shannon divergence
    pub fn jensen_shannon_divergence(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let p = normalize_to_probability(&self.as_f32())?;
        let q = normalize_to_probability(&other.as_f32())?;

        // Calculate average distribution
        let m: Vec<f32> = p.iter().zip(&q).map(|(a, b)| (a + b) / 2.0).collect();

        // Calculate KL divergences
        let kl_pm = kl_divergence_raw(&p, &m)?;
        let kl_qm = kl_divergence_raw(&q, &m)?;

        Ok((kl_pm + kl_qm) / 2.0)
    }

    /// Calculate Kullback-Leibler divergence
    pub fn kl_divergence(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let p = normalize_to_probability(&self.as_f32())?;
        let q = normalize_to_probability(&other.as_f32())?;

        kl_divergence_raw(&p, &q)
    }

    /// Calculate Hellinger distance
    pub fn hellinger_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let p = normalize_to_probability(&self.as_f32())?;
        let q = normalize_to_probability(&other.as_f32())?;

        let sum: f32 = p.iter()
            .zip(&q)
            .map(|(a, b)| (a.sqrt() - b.sqrt()).powi(2))
            .sum();

        Ok((sum / 2.0).sqrt())
    }

    /// Calculate Earth Mover's Distance (1D approximation)
    pub fn earth_movers_distance(&self, other: &Vector) -> Result<f32> {
        if self.dimensions != other.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        // Normalize to probability distributions
        let p = normalize_to_probability(&self.as_f32())?;
        let q = normalize_to_probability(&other.as_f32())?;

        // Calculate cumulative distributions
        let mut cdf_p = vec![0.0; p.len()];
        let mut cdf_q = vec![0.0; q.len()];
        
        cdf_p[0] = p[0];
        cdf_q[0] = q[0];
        
        for i in 1..p.len() {
            cdf_p[i] = cdf_p[i - 1] + p[i];
            cdf_q[i] = cdf_q[i - 1] + q[i];
        }

        // EMD is the L1 distance between CDFs
        let emd: f32 = cdf_p.iter()
            .zip(&cdf_q)
            .map(|(a, b)| (a - b).abs())
            .sum();

        Ok(emd)
    }

    /// Calculate Wasserstein distance (1-Wasserstein, same as EMD for 1D)
    pub fn wasserstein_distance(&self, other: &Vector) -> Result<f32> {
        self.earth_movers_distance(other)
    }

    /// Calculate Mahalanobis distance (simplified version without covariance matrix)
    /// This version assumes diagonal covariance (independent dimensions)
    pub fn mahalanobis_distance(&self, other: &Vector, variance: &[f32]) -> Result<f32> {
        if self.dimensions != other.dimensions || self.dimensions != variance.len() {
            return Err(anyhow::anyhow!("Dimensions must match"));
        }

        let x = self.as_f32();
        let y = other.as_f32();

        let distance_sq: f32 = x.iter()
            .zip(&y)
            .zip(variance)
            .map(|((a, b), &var)| {
                if var > 0.0 {
                    (a - b).powi(2) / var
                } else {
                    0.0
                }
            })
            .sum();

        Ok(distance_sq.sqrt())
    }
}

/// Convert values to ranks (1-based)
fn to_ranks(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    
    while i < indexed.len() {
        let mut j = i;
        // Find all equal values
        while j < indexed.len() && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        
        // Assign average rank to all equal values
        let avg_rank = (i + j) as f32 / 2.0 + 0.5;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        
        i = j;
    }
    
    ranks
}

/// Normalize vector to probability distribution
fn normalize_to_probability(values: &[f32]) -> Result<Vec<f32>> {
    // Ensure non-negative values
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let shifted: Vec<f32> = if min_val < 0.0 {
        values.iter().map(|v| v - min_val).collect()
    } else {
        values.to_vec()
    };

    let sum: f32 = shifted.iter().sum();
    
    if sum == 0.0 {
        // Uniform distribution if all zeros
        Ok(vec![1.0 / values.len() as f32; values.len()])
    } else {
        Ok(shifted.iter().map(|v| v / sum).collect())
    }
}

/// Calculate KL divergence between two probability distributions
fn kl_divergence_raw(p: &[f32], q: &[f32]) -> Result<f32> {
    if p.len() != q.len() {
        return Err(anyhow::anyhow!("Distributions must have same length"));
    }

    let mut kl = 0.0;
    for (pi, qi) in p.iter().zip(q) {
        if *pi > 0.0 && *qi > 0.0 {
            kl += pi * (pi / qi).ln();
        } else if *pi > 0.0 && *qi == 0.0 {
            return Ok(f32::INFINITY);
        }
    }
    
    Ok(kl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        let vec1 = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let vec2 = Vector::new(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        
        let correlation = vec1.pearson_correlation(&vec2).unwrap();
        assert!((correlation - 1.0).abs() < 1e-6); // Perfect correlation
    }

    #[test]
    fn test_spearman_correlation() {
        let vec1 = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let vec2 = Vector::new(vec![1.0, 4.0, 9.0, 16.0, 25.0]); // Monotonic but not linear
        
        let correlation = vec1.spearman_correlation(&vec2).unwrap();
        assert!((correlation - 1.0).abs() < 1e-6); // Perfect rank correlation
    }

    #[test]
    fn test_jaccard_similarity() {
        let vec1 = Vector::binary(vec![0b11110000]);
        let vec2 = Vector::binary(vec![0b11001100]);
        
        let similarity = vec1.jaccard_similarity(&vec2).unwrap();
        // Intersection: 0b11000000 (2 bits)
        // Union: 0b11111100 (6 bits)
        assert!((similarity - 2.0/6.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        let vec1 = Vector::binary(vec![0b11110000]);
        let vec2 = Vector::binary(vec![0b11001100]);
        
        let distance = vec1.hamming_distance(&vec2).unwrap();
        // XOR: 0b00111100 (4 different bits)
        assert_eq!(distance, 4);
    }

    #[test]
    fn test_jensen_shannon_divergence() {
        let vec1 = Vector::new(vec![0.25, 0.25, 0.25, 0.25]);
        let vec2 = Vector::new(vec![0.5, 0.3, 0.1, 0.1]);
        
        let jsd = vec1.jensen_shannon_divergence(&vec2).unwrap();
        assert!(jsd >= 0.0 && jsd <= 1.0);
        
        // Same distribution should have JSD = 0
        let jsd_same = vec1.jensen_shannon_divergence(&vec1).unwrap();
        assert!(jsd_same.abs() < 1e-6);
    }

    #[test]
    fn test_hellinger_distance() {
        let vec1 = Vector::new(vec![0.25, 0.25, 0.25, 0.25]);
        let vec2 = Vector::new(vec![0.25, 0.25, 0.25, 0.25]);
        
        let distance = vec1.hellinger_distance(&vec2).unwrap();
        assert!(distance.abs() < 1e-6); // Same distribution
        
        let vec3 = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let distance2 = vec1.hellinger_distance(&vec3).unwrap();
        assert!(distance2 > 0.0);
    }

    #[test]
    fn test_earth_movers_distance() {
        let vec1 = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let vec2 = Vector::new(vec![0.0, 0.0, 0.0, 1.0]);
        
        let emd = vec1.earth_movers_distance(&vec2).unwrap();
        assert!(emd > 0.0); // Mass needs to be moved
        
        // Same distribution
        let emd_same = vec1.earth_movers_distance(&vec1).unwrap();
        assert!(emd_same.abs() < 1e-6);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let vec1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let vec2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let variance = vec![1.0, 2.0, 3.0];
        
        let distance = vec1.mahalanobis_distance(&vec2, &variance).unwrap();
        assert!(distance > 0.0);
        
        // Distance to self should be 0
        let self_distance = vec1.mahalanobis_distance(&vec1, &variance).unwrap();
        assert!(self_distance.abs() < 1e-6);
    }
}