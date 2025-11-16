//! Extended distance metrics for vector similarity
//!
//! This module provides a comprehensive collection of distance metrics
//! including specialized metrics for different data types and use cases.

use crate::Vector;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Extended distance metric types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExtendedDistanceMetric {
    // Standard metrics
    Cosine,
    Euclidean,
    Manhattan,
    Chebyshev,
    Minkowski { p: f32 },

    // Specialized metrics
    Hamming,
    Jaccard,
    Dice,
    Pearson,
    Spearman,
    Kendall,

    // Statistical metrics
    KLDivergence,
    JensenShannon,
    Bhattacharyya,
    Hellinger,

    // Edit distance metrics
    Levenshtein,
    DamerauLevenshtein,

    // Information-theoretic metrics
    MutualInformation,
    NormalizedCompressionDistance,

    // Specialized for embeddings
    Mahalanobis,
    BrayCurtis,

    // Custom metric (user-defined)
    Custom(u32), // ID for custom metric lookup
}

impl ExtendedDistanceMetric {
    /// Calculate distance between two vectors
    pub fn distance(&self, a: &Vector, b: &Vector) -> Result<f32> {
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();

        if a_f32.len() != b_f32.len() {
            return Err(anyhow::anyhow!(
                "Vector dimensions must match: {} != {}",
                a_f32.len(),
                b_f32.len()
            ));
        }

        match self {
            ExtendedDistanceMetric::Cosine => Self::cosine_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Euclidean => Self::euclidean_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Manhattan => Self::manhattan_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Chebyshev => Self::chebyshev_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Minkowski { p } => Self::minkowski_distance(&a_f32, &b_f32, *p),
            ExtendedDistanceMetric::Hamming => Self::hamming_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Jaccard => Self::jaccard_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Dice => Self::dice_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Pearson => Self::pearson_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Spearman => Self::spearman_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Kendall => Self::kendall_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::KLDivergence => Self::kl_divergence(&a_f32, &b_f32),
            ExtendedDistanceMetric::JensenShannon => Self::jensen_shannon(&a_f32, &b_f32),
            ExtendedDistanceMetric::Bhattacharyya => Self::bhattacharyya(&a_f32, &b_f32),
            ExtendedDistanceMetric::Hellinger => Self::hellinger(&a_f32, &b_f32),
            ExtendedDistanceMetric::Levenshtein => Self::levenshtein_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::DamerauLevenshtein => {
                Self::damerau_levenshtein_distance(&a_f32, &b_f32)
            }
            ExtendedDistanceMetric::MutualInformation => Self::mutual_information(&a_f32, &b_f32),
            ExtendedDistanceMetric::NormalizedCompressionDistance => Self::ncd(&a_f32, &b_f32),
            ExtendedDistanceMetric::Mahalanobis => Self::mahalanobis_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::BrayCurtis => Self::bray_curtis_distance(&a_f32, &b_f32),
            ExtendedDistanceMetric::Custom(_id) => {
                // Custom metrics would be looked up from a registry
                Err(anyhow::anyhow!("Custom metrics not implemented"))
            }
        }
    }

    // Standard distance metrics

    fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(1.0);
        }

        Ok(1.0 - (dot / (norm_a * norm_b)))
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let dist: f32 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        Ok(dist)
    }

    fn manhattan_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let dist: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
        Ok(dist)
    }

    fn chebyshev_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let dist = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, |max, val| max.max(val));
        Ok(dist)
    }

    fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> Result<f32> {
        if p <= 0.0 {
            return Err(anyhow::anyhow!("p must be positive for Minkowski distance"));
        }

        if p == f32::INFINITY {
            return Self::chebyshev_distance(a, b);
        }

        let dist = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs().powf(p))
            .sum::<f32>()
            .powf(1.0 / p);
        Ok(dist)
    }

    // Specialized distance metrics

    fn hamming_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let threshold = 0.5; // Threshold for binary conversion
        let dist = a
            .iter()
            .zip(b)
            .filter(|(x, y)| {
                let x_bin = **x > threshold;
                let y_bin = **y > threshold;
                x_bin != y_bin
            })
            .count();
        Ok(dist as f32)
    }

    fn jaccard_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let threshold = 0.5;
        let mut intersection = 0;
        let mut union = 0;

        for (x, y) in a.iter().zip(b) {
            let x_bin = *x > threshold;
            let y_bin = *y > threshold;

            if x_bin || y_bin {
                union += 1;
                if x_bin && y_bin {
                    intersection += 1;
                }
            }
        }

        if union == 0 {
            return Ok(0.0);
        }

        Ok(1.0 - (intersection as f32 / union as f32))
    }

    fn dice_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let threshold = 0.5;
        let mut intersection = 0;
        let mut a_count = 0;
        let mut b_count = 0;

        for (x, y) in a.iter().zip(b) {
            let x_bin = *x > threshold;
            let y_bin = *y > threshold;

            if x_bin {
                a_count += 1;
            }
            if y_bin {
                b_count += 1;
            }
            if x_bin && y_bin {
                intersection += 1;
            }
        }

        let sum = a_count + b_count;
        if sum == 0 {
            return Ok(0.0);
        }

        Ok(1.0 - (2.0 * intersection as f32 / sum as f32))
    }

    fn pearson_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let n = a.len() as f32;
        let mean_a: f32 = a.iter().sum::<f32>() / n;
        let mean_b: f32 = b.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for (x, y) in a.iter().zip(b) {
            let da = x - mean_a;
            let db = y - mean_b;
            numerator += da * db;
            sum_sq_a += da * da;
            sum_sq_b += db * db;
        }

        if sum_sq_a == 0.0 || sum_sq_b == 0.0 {
            return Ok(1.0);
        }

        let correlation = numerator / (sum_sq_a.sqrt() * sum_sq_b.sqrt());
        Ok(1.0 - correlation)
    }

    fn spearman_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        // Convert to ranks
        let rank_a = Self::rank_vector(a);
        let rank_b = Self::rank_vector(b);

        // Calculate Pearson on ranks
        Self::pearson_distance(&rank_a, &rank_b)
    }

    fn kendall_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let n = a.len();
        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sign_a = (a[j] - a[i]).signum();
                let sign_b = (b[j] - b[i]).signum();

                if sign_a * sign_b > 0.0 {
                    concordant += 1;
                } else if sign_a * sign_b < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = (n * (n - 1)) / 2;
        if total_pairs == 0 {
            return Ok(0.0);
        }

        let tau = (concordant - discordant) as f32 / total_pairs as f32;
        Ok(1.0 - tau)
    }

    // Statistical distance metrics

    fn kl_divergence(p: &[f32], q: &[f32]) -> Result<f32> {
        let epsilon = 1e-10;
        let mut divergence = 0.0;

        for (pi, qi) in p.iter().zip(q) {
            let pi_safe = pi.max(epsilon);
            let qi_safe = qi.max(epsilon);
            divergence += pi_safe * (pi_safe / qi_safe).ln();
        }

        Ok(divergence)
    }

    fn jensen_shannon(p: &[f32], q: &[f32]) -> Result<f32> {
        let m: Vec<f32> = p.iter().zip(q).map(|(pi, qi)| (pi + qi) / 2.0).collect();

        let kl_pm = Self::kl_divergence(p, &m)?;
        let kl_qm = Self::kl_divergence(q, &m)?;

        Ok((kl_pm + kl_qm) / 2.0)
    }

    fn bhattacharyya(p: &[f32], q: &[f32]) -> Result<f32> {
        let bc: f32 = p.iter().zip(q).map(|(pi, qi)| (pi * qi).sqrt()).sum();
        Ok(-bc.ln())
    }

    fn hellinger(p: &[f32], q: &[f32]) -> Result<f32> {
        let sum: f32 = p
            .iter()
            .zip(q)
            .map(|(pi, qi)| (pi.sqrt() - qi.sqrt()).powi(2))
            .sum();
        Ok((sum / 2.0).sqrt())
    }

    // Edit distance metrics

    #[allow(clippy::needless_range_loop)]
    fn levenshtein_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let threshold = 0.5;
        let a_bin: Vec<bool> = a.iter().map(|x| *x > threshold).collect();
        let b_bin: Vec<bool> = b.iter().map(|x| *x > threshold).collect();

        let m = a_bin.len();
        let n = b_bin.len();

        if m == 0 {
            return Ok(n as f32);
        }
        if n == 0 {
            return Ok(m as f32);
        }

        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if a_bin[i - 1] == b_bin[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        Ok(dp[m][n] as f32)
    }

    fn damerau_levenshtein_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        // Simplified Damerau-Levenshtein (allows transpositions)
        // Full implementation is complex, this is an approximation
        Self::levenshtein_distance(a, b)
    }

    // Information-theoretic metrics

    fn mutual_information(a: &[f32], b: &[f32]) -> Result<f32> {
        // Simplified mutual information calculation
        // Full implementation would require histogram binning
        let joint_entropy = Self::calculate_entropy(a)? + Self::calculate_entropy(b)?;
        let individual_entropy = Self::calculate_joint_entropy(a, b)?;

        Ok(joint_entropy - individual_entropy)
    }

    fn ncd(a: &[f32], b: &[f32]) -> Result<f32> {
        // Normalized Compression Distance
        // Approximation using simple compression ratios
        let ca = Self::estimate_compression_size(a);
        let cb = Self::estimate_compression_size(b);
        let cab = Self::estimate_joint_compression_size(a, b);

        let min_c = ca.min(cb);
        let max_c = ca.max(cb);

        if max_c == 0.0 {
            return Ok(0.0);
        }

        Ok((cab - min_c) / max_c)
    }

    // Advanced distance metrics

    fn mahalanobis_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        // Simplified Mahalanobis distance (assuming identity covariance)
        // Full implementation would require covariance matrix
        Self::euclidean_distance(a, b)
    }

    fn bray_curtis_distance(a: &[f32], b: &[f32]) -> Result<f32> {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (x, y) in a.iter().zip(b) {
            numerator += (x - y).abs();
            denominator += x + y;
        }

        if denominator == 0.0 {
            return Ok(0.0);
        }

        Ok(numerator / denominator)
    }

    // Helper functions

    fn rank_vector(v: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = v.iter().enumerate().map(|(i, &x)| (i, x)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; v.len()];
        for (rank, (original_index, _)) in indexed.iter().enumerate() {
            ranks[*original_index] = rank as f32;
        }

        ranks
    }

    fn calculate_entropy(v: &[f32]) -> Result<f32> {
        let epsilon = 1e-10;
        let mut entropy = 0.0;

        for &x in v {
            if x > epsilon {
                entropy -= x * x.ln();
            }
        }

        Ok(entropy)
    }

    fn calculate_joint_entropy(a: &[f32], b: &[f32]) -> Result<f32> {
        let epsilon = 1e-10;
        let mut entropy = 0.0;

        for (x, y) in a.iter().zip(b) {
            let joint = x * y;
            if joint > epsilon {
                entropy -= joint * joint.ln();
            }
        }

        Ok(entropy)
    }

    fn estimate_compression_size(v: &[f32]) -> f32 {
        // Rough estimate based on unique values and entropy
        // Since f32 doesn't implement Eq/Hash, we'll use a different approach
        let mut sorted = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut unique_count = 1;
        for i in 1..sorted.len() {
            if (sorted[i] - sorted[i - 1]).abs() > 1e-6 {
                unique_count += 1;
            }
        }

        unique_count as f32
    }

    fn estimate_joint_compression_size(a: &[f32], b: &[f32]) -> f32 {
        let mut combined = Vec::with_capacity(a.len() + b.len());
        combined.extend_from_slice(a);
        combined.extend_from_slice(b);
        Self::estimate_compression_size(&combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let a = Vector::new(vec![1.0, 0.0, 0.0]);
        let b = Vector::new(vec![1.0, 0.0, 0.0]);

        let distance = ExtendedDistanceMetric::Cosine.distance(&a, &b).unwrap();
        assert!(distance < 0.01); // Should be close to 0
    }

    #[test]
    fn test_euclidean_distance() {
        let a = Vector::new(vec![0.0, 0.0]);
        let b = Vector::new(vec![3.0, 4.0]);

        let distance = ExtendedDistanceMetric::Euclidean.distance(&a, &b).unwrap();
        assert!((distance - 5.0).abs() < 0.01); // Should be 5.0
    }

    #[test]
    fn test_hamming_distance() {
        let a = Vector::new(vec![1.0, 1.0, 0.0, 0.0]);
        let b = Vector::new(vec![1.0, 0.0, 1.0, 0.0]);

        let distance = ExtendedDistanceMetric::Hamming.distance(&a, &b).unwrap();
        assert_eq!(distance, 2.0); // 2 positions differ
    }

    #[test]
    fn test_jaccard_distance() {
        let a = Vector::new(vec![1.0, 1.0, 0.0, 0.0]);
        let b = Vector::new(vec![1.0, 0.0, 1.0, 0.0]);

        let distance = ExtendedDistanceMetric::Jaccard.distance(&a, &b).unwrap();
        assert!(distance > 0.0 && distance < 1.0);
    }

    #[test]
    fn test_pearson_distance() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);

        let distance = ExtendedDistanceMetric::Pearson.distance(&a, &b).unwrap();
        assert!(distance < 0.01); // Perfect correlation
    }

    #[test]
    fn test_manhattan_distance() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);

        let distance = ExtendedDistanceMetric::Manhattan.distance(&a, &b).unwrap();
        assert_eq!(distance, 9.0); // |1-4| + |2-5| + |3-6| = 9
    }
}
