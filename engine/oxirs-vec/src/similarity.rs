//! Advanced similarity algorithms and semantic matching for vectors

use crate::Vector;
use anyhow::{anyhow, Result};
use oxirs_core::simd::SimdOps;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// Similarity measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct SimilarityConfig {
    /// Primary similarity metric
    pub primary_metric: SimilarityMetric,
    /// Secondary metrics for ensemble scoring
    pub ensemble_metrics: Vec<SimilarityMetric>,
    /// Weights for ensemble metrics
    pub ensemble_weights: Vec<f32>,
    /// Threshold for considering vectors similar
    pub similarity_threshold: f32,
    /// Enable semantic boosting
    pub semantic_boost: bool,
    /// Enable temporal decay
    pub temporal_decay: bool,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            primary_metric: SimilarityMetric::Cosine,
            ensemble_metrics: vec![
                SimilarityMetric::Cosine,
                SimilarityMetric::Pearson,
                SimilarityMetric::Jaccard,
            ],
            ensemble_weights: vec![0.5, 0.3, 0.2],
            similarity_threshold: 0.7,
            semantic_boost: true,
            temporal_decay: false,
        }
    }
}

/// Available similarity metrics
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, oxicode::Encode, oxicode::Decode,
)]
pub enum SimilarityMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance (converted to similarity)
    Euclidean,
    /// Manhattan distance (converted to similarity)
    Manhattan,
    /// Minkowski distance (general Lp norm)
    Minkowski(f32),
    /// Pearson correlation coefficient
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Jaccard similarity (for sparse vectors)
    Jaccard,
    /// Dice coefficient
    Dice,
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Bhattacharyya distance
    Bhattacharyya,
    /// Mahalanobis distance (requires covariance matrix)
    Mahalanobis,
    /// Hamming distance (for binary vectors)
    Hamming,
    /// Canberra distance
    Canberra,
    /// Angular distance
    Angular,
    /// Chebyshev distance (L∞ norm)
    Chebyshev,
    /// Dot product (inner product)
    DotProduct,
}

impl SimilarityMetric {
    /// Calculate similarity between two vectors
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow!("Vector dimensions must match"));
        }

        let similarity = match self {
            SimilarityMetric::Cosine => cosine_similarity(a, b),
            SimilarityMetric::Euclidean => euclidean_similarity(a, b),
            SimilarityMetric::Manhattan => manhattan_similarity(a, b),
            SimilarityMetric::Minkowski(p) => minkowski_similarity(a, b, *p),
            SimilarityMetric::Pearson => pearson_correlation(a, b)?,
            SimilarityMetric::Spearman => spearman_correlation(a, b)?,
            SimilarityMetric::Jaccard => jaccard_similarity(a, b),
            SimilarityMetric::Dice => dice_coefficient(a, b),
            SimilarityMetric::JensenShannon => jensen_shannon_similarity(a, b)?,
            SimilarityMetric::Bhattacharyya => bhattacharyya_similarity(a, b)?,
            SimilarityMetric::Mahalanobis => {
                // Mahalanobis requires a covariance matrix, which this stateless
                // metric API does not carry. Do not silently return Euclidean
                // (Mahalanobis with Σ = I) mislabeled as Mahalanobis — fail loud.
                // Use `SemanticSimilarity::set_covariance_matrix` +
                // `SemanticSimilarity::mahalanobis_similarity` for the real metric.
                return Err(anyhow!(
                    "Mahalanobis similarity requires a covariance matrix; use \
                     SemanticSimilarity::set_covariance_matrix() rather than the \
                     stateless SimilarityMetric::Mahalanobis"
                ));
            }
            SimilarityMetric::Hamming => hamming_similarity(a, b),
            SimilarityMetric::Canberra => canberra_similarity(a, b),
            SimilarityMetric::Angular => angular_similarity(a, b),
            SimilarityMetric::Chebyshev => chebyshev_similarity(a, b),
            SimilarityMetric::DotProduct => dot_product_similarity(a, b),
        };

        Ok(similarity.clamp(0.0, 1.0))
    }

    /// Calculate distance between two vectors (lower is more similar)
    pub fn distance(&self, a: &Vector, b: &Vector) -> Result<f32> {
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();

        if a_f32.len() != b_f32.len() {
            return Err(anyhow!("Vector dimensions must match"));
        }

        let distance = match self {
            // Distance metrics - use direct calculation
            SimilarityMetric::Euclidean => euclidean_distance(&a_f32, &b_f32),
            SimilarityMetric::Manhattan => manhattan_distance(&a_f32, &b_f32),
            SimilarityMetric::Minkowski(p) => minkowski_distance(&a_f32, &b_f32, *p),
            SimilarityMetric::Hamming => hamming_distance(&a_f32, &b_f32),
            SimilarityMetric::Canberra => canberra_distance(&a_f32, &b_f32),
            SimilarityMetric::Chebyshev => chebyshev_distance(&a_f32, &b_f32),

            // Similarity metrics - convert to distance (1 - similarity)
            _ => {
                let similarity = self.similarity(&a_f32, &b_f32)?;
                1.0 - similarity
            }
        };

        Ok(distance.max(0.0))
    }

    /// Compute similarity between two vectors (alias for similarity method)
    pub fn compute(&self, a: &Vector, b: &Vector) -> Result<f32> {
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();
        self.similarity(&a_f32, &b_f32)
    }
}

/// Semantic similarity computer with multiple algorithms
pub struct SemanticSimilarity {
    config: SimilarityConfig,
    feature_weights: Option<Vec<f32>>,
    covariance_matrix: Option<Vec<Vec<f32>>>,
}

impl SemanticSimilarity {
    pub fn new(config: SimilarityConfig) -> Self {
        Self {
            config,
            feature_weights: None,
            covariance_matrix: None,
        }
    }

    /// Set feature importance weights
    pub fn set_feature_weights(&mut self, weights: Vec<f32>) {
        self.feature_weights = Some(weights);
    }

    /// Set covariance matrix for Mahalanobis distance
    pub fn set_covariance_matrix(&mut self, matrix: Vec<Vec<f32>>) {
        self.covariance_matrix = Some(matrix);
    }

    /// Compute the true Mahalanobis distance `sqrt((a-b)ᵀ · Σ⁻¹ · (a-b))` using
    /// the configured covariance matrix Σ (set via
    /// [`Self::set_covariance_matrix`]).
    ///
    /// Fails loudly if no covariance matrix has been configured or if it is not
    /// a square matrix matching the vector dimensionality / is singular — never
    /// silently degrades to Euclidean distance.
    pub fn mahalanobis_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow!("Vector dimensions must match"));
        }
        let cov = self.covariance_matrix.as_ref().ok_or_else(|| {
            anyhow!(
                "Mahalanobis distance requires a covariance matrix; call \
                 SemanticSimilarity::set_covariance_matrix() first"
            )
        })?;
        let n = a.len();
        if cov.len() != n || cov.iter().any(|row| row.len() != n) {
            return Err(anyhow!(
                "Covariance matrix must be {n}x{n} to match the vector dimensionality"
            ));
        }
        let inv = invert_matrix(cov)
            .ok_or_else(|| anyhow!("Covariance matrix is singular and cannot be inverted"))?;

        // diff = a - b
        let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        // tmp = Σ⁻¹ · diff
        let mut quadratic = 0.0f32;
        for (i, &di) in diff.iter().enumerate() {
            let mut row_sum = 0.0f32;
            for (j, &dj) in diff.iter().enumerate() {
                row_sum += inv[i][j] * dj;
            }
            quadratic += di * row_sum;
        }
        // Numerical guard: a valid inverse-covariance is PSD so quadratic >= 0,
        // but clamp tiny negative round-off to 0 before sqrt.
        Ok(quadratic.max(0.0).sqrt())
    }

    /// Mahalanobis *similarity* in `[0, 1]` derived from the distance as
    /// `1 / (1 + d)` (larger = more similar), consistent with the crate's
    /// distance→similarity convention.
    pub fn mahalanobis_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let d = self.mahalanobis_distance(a, b)?;
        Ok(1.0 / (1.0 + d))
    }

    /// Calculate similarity using primary metric
    pub fn similarity(&self, a: &Vector, b: &Vector) -> Result<f32> {
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();

        // Mahalanobis is stateful (needs the covariance matrix), so it is
        // handled here rather than via the stateless `SimilarityMetric` path,
        // which deliberately fails loudly for this variant.
        let mut similarity = if matches!(self.config.primary_metric, SimilarityMetric::Mahalanobis)
        {
            self.mahalanobis_similarity(&a_f32, &b_f32)?
        } else {
            self.config.primary_metric.similarity(&a_f32, &b_f32)?
        };

        // Apply feature weighting if available
        if let Some(ref weights) = self.feature_weights {
            similarity = self.apply_feature_weights(&a_f32, &b_f32, weights);
        }

        // Apply semantic boosting
        if self.config.semantic_boost {
            similarity = self.apply_semantic_boost(similarity, a, b);
        }

        Ok(similarity)
    }

    /// Calculate ensemble similarity using multiple metrics
    pub fn ensemble_similarity(&self, a: &Vector, b: &Vector) -> Result<f32> {
        if self.config.ensemble_metrics.len() != self.config.ensemble_weights.len() {
            return Err(anyhow!("Ensemble metrics and weights length mismatch"));
        }

        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (metric, weight) in self
            .config
            .ensemble_metrics
            .iter()
            .zip(&self.config.ensemble_weights)
        {
            let similarity = metric.similarity(&a_f32, &b_f32)?;
            weighted_sum += similarity * weight;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        let ensemble_score = weighted_sum / total_weight;

        // Apply semantic boosting
        if self.config.semantic_boost {
            Ok(self.apply_semantic_boost(ensemble_score, a, b))
        } else {
            Ok(ensemble_score)
        }
    }

    /// Calculate similarity matrix for a set of vectors
    pub fn similarity_matrix(&self, vectors: &[Vector]) -> Result<Vec<Vec<f32>>> {
        let n = vectors.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                let similarity = if i == j {
                    1.0
                } else {
                    self.similarity(&vectors[i], &vectors[j])?
                };

                matrix[i][j] = similarity;
                matrix[j][i] = similarity;
            }
        }

        Ok(matrix)
    }

    /// Find most similar vectors to a query
    pub fn find_similar(
        &self,
        query: &Vector,
        candidates: &[(String, Vector)],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let mut similarities: Vec<(String, f32)> = candidates
            .iter()
            .map(|(uri, vector)| {
                let sim = self.similarity(query, vector).unwrap_or(0.0);
                (uri.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);

        Ok(similarities)
    }

    /// Calculate semantic clusters based on similarity
    pub fn cluster_by_similarity(
        &self,
        vectors: &[(String, Vector)],
        threshold: f32,
    ) -> Result<Vec<Vec<String>>> {
        let mut clusters: Vec<Vec<String>> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; vectors.len()];

        for i in 0..vectors.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![vectors[i].0.clone()];
            assigned[i] = true;

            for j in (i + 1)..vectors.len() {
                if assigned[j] {
                    continue;
                }

                let similarity = self.similarity(&vectors[i].1, &vectors[j].1)?;
                if similarity >= threshold {
                    cluster.push(vectors[j].0.clone());
                    assigned[j] = true;
                }
            }

            clusters.push(cluster);
        }

        Ok(clusters)
    }

    fn apply_feature_weights(&self, a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
        let weighted_a: Vec<f32> = a.iter().zip(weights).map(|(x, w)| x * w).collect();
        let weighted_b: Vec<f32> = b.iter().zip(weights).map(|(x, w)| x * w).collect();

        cosine_similarity(&weighted_a, &weighted_b)
    }

    fn apply_semantic_boost(&self, similarity: f32, a: &Vector, b: &Vector) -> f32 {
        // Simple semantic boosting based on vector magnitude similarity
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();
        let mag_a = vector_magnitude(&a_f32);
        let mag_b = vector_magnitude(&b_f32);
        let magnitude_similarity = 1.0 - (mag_a - mag_b).abs() / (mag_a + mag_b + f32::EPSILON);

        // Weighted combination
        0.8 * similarity + 0.2 * magnitude_similarity
    }
}

/// Adaptive similarity that learns from user feedback
pub struct AdaptiveSimilarity {
    base_similarity: SemanticSimilarity,
    feedback_weights: HashMap<String, f32>,
    learning_rate: f32,
}

impl AdaptiveSimilarity {
    pub fn new(config: SimilarityConfig, learning_rate: f32) -> Self {
        Self {
            base_similarity: SemanticSimilarity::new(config),
            feedback_weights: HashMap::new(),
            learning_rate,
        }
    }

    /// Provide feedback on similarity result
    pub fn add_feedback(&mut self, uri: &str, expected_similarity: f32, actual_similarity: f32) {
        let error = expected_similarity - actual_similarity;
        let adjustment = self.learning_rate * error;

        *self.feedback_weights.entry(uri.to_string()).or_insert(0.0) += adjustment;
    }

    /// Calculate similarity with learned adjustments
    pub fn adaptive_similarity(
        &self,
        a: &Vector,
        b: &Vector,
        uri_a: &str,
        uri_b: &str,
    ) -> Result<f32> {
        let base_sim = self.base_similarity.similarity(a, b)?;

        let weight_a = self.feedback_weights.get(uri_a).unwrap_or(&0.0);
        let weight_b = self.feedback_weights.get(uri_b).unwrap_or(&0.0);
        let adjustment = (weight_a + weight_b) / 2.0;

        Ok((base_sim + adjustment).clamp(0.0, 1.0))
    }

    /// Get learned weights for analysis
    pub fn get_feedback_weights(&self) -> &HashMap<String, f32> {
        &self.feedback_weights
    }
}

/// Temporal similarity that considers time decay
pub struct TemporalSimilarity {
    base_similarity: SemanticSimilarity,
    decay_rate: f32,
    time_weights: HashMap<String, f32>,
}

impl TemporalSimilarity {
    pub fn new(config: SimilarityConfig, decay_rate: f32) -> Self {
        Self {
            base_similarity: SemanticSimilarity::new(config),
            decay_rate,
            time_weights: HashMap::new(),
        }
    }

    /// Set time weight for a URI (higher = more recent)
    pub fn set_time_weight(&mut self, uri: &str, time_weight: f32) {
        self.time_weights.insert(uri.to_string(), time_weight);
    }

    /// Calculate similarity with temporal decay
    pub fn temporal_similarity(
        &self,
        a: &Vector,
        b: &Vector,
        uri_a: &str,
        uri_b: &str,
    ) -> Result<f32> {
        let base_sim = self.base_similarity.similarity(a, b)?;

        let time_a = self.time_weights.get(uri_a).unwrap_or(&1.0);
        let time_b = self.time_weights.get(uri_b).unwrap_or(&1.0);

        let time_factor = (time_a + time_b) / 2.0;
        let decay = (-self.decay_rate * (1.0 - time_factor)).exp();

        Ok(base_sim * decay)
    }
}

// Individual similarity function implementations

/// Compute similarity between two vectors using the specified metric
pub fn compute_similarity(a: &[f32], b: &[f32], metric: SimilarityMetric) -> Result<f32> {
    metric.similarity(a, b)
}

/// Normalize a vector to unit length (in-place)
pub fn normalize_vector(vector: &mut [f32]) -> Result<()> {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for value in vector.iter_mut() {
            *value /= magnitude;
        }
    }
    Ok(())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Use oxirs-core SIMD operations
    1.0 - f32::cosine_distance(a, b)
}

fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Use oxirs-core SIMD operations
    let distance = f32::euclidean_distance(a, b);
    1.0 / (1.0 + distance)
}

fn manhattan_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Use oxirs-core SIMD operations
    let distance = f32::manhattan_distance(a, b);
    1.0 / (1.0 + distance)
}

fn minkowski_similarity(a: &[f32], b: &[f32], p: f32) -> f32 {
    if p <= 0.0 {
        // Handle edge case
        return euclidean_similarity(a, b);
    }

    let distance: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f32>()
        .powf(1.0 / p);
    1.0 / (1.0 + distance)
}

fn chebyshev_similarity(a: &[f32], b: &[f32]) -> f32 {
    let distance: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |acc, diff| acc.max(diff));
    1.0 / (1.0 + distance)
}

fn pearson_correlation(a: &[f32], b: &[f32]) -> Result<f32> {
    let n = a.len() as f32;
    if n == 0.0 {
        return Ok(0.0);
    }

    // Use oxirs-core SIMD operations for mean calculation
    let mean_a = f32::mean(a);
    let mean_b = f32::mean(b);

    let numerator: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    let sum_sq_a: f32 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
    let sum_sq_b: f32 = b.iter().map(|x| (x - mean_b).powi(2)).sum();

    let denominator = (sum_sq_a * sum_sq_b).sqrt();

    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

fn spearman_correlation(a: &[f32], b: &[f32]) -> Result<f32> {
    let ranks_a = compute_ranks(a);
    let ranks_b = compute_ranks(b);
    pearson_correlation(&ranks_a, &ranks_b)
}

fn compute_ranks(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; values.len()];
    for (rank, (original_index, _)) in indexed.iter().enumerate() {
        ranks[*original_index] = rank as f32 + 1.0;
    }

    ranks
}

fn jaccard_similarity(a: &[f32], b: &[f32]) -> f32 {
    let threshold = 0.01; // Consider values above this as "present"
    let set_a: Vec<bool> = a.iter().map(|&x| x > threshold).collect();
    let set_b: Vec<bool> = b.iter().map(|&x| x > threshold).collect();

    let intersection: usize = set_a
        .iter()
        .zip(&set_b)
        .map(|(x, y)| (*x && *y) as usize)
        .sum();
    let union: usize = set_a
        .iter()
        .zip(&set_b)
        .map(|(x, y)| (*x || *y) as usize)
        .sum();

    if union == 0 {
        1.0 // Both empty sets
    } else {
        intersection as f32 / union as f32
    }
}

fn dice_coefficient(a: &[f32], b: &[f32]) -> f32 {
    let threshold = 0.01;
    let set_a: Vec<bool> = a.iter().map(|&x| x > threshold).collect();
    let set_b: Vec<bool> = b.iter().map(|&x| x > threshold).collect();

    let intersection: usize = set_a
        .iter()
        .zip(&set_b)
        .map(|(x, y)| (*x && *y) as usize)
        .sum();
    let size_a: usize = set_a.iter().map(|&x| x as usize).sum();
    let size_b: usize = set_b.iter().map(|&x| x as usize).sum();

    if size_a + size_b == 0 {
        1.0
    } else {
        2.0 * intersection as f32 / (size_a + size_b) as f32
    }
}

fn jensen_shannon_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    // Normalize to probability distributions
    let sum_a: f32 = a.iter().sum();
    let sum_b: f32 = b.iter().sum();

    if sum_a == 0.0 || sum_b == 0.0 {
        return Ok(0.0);
    }

    let p: Vec<f32> = a.iter().map(|x| x / sum_a).collect();
    let q: Vec<f32> = b.iter().map(|x| x / sum_b).collect();

    // Compute average distribution
    let m: Vec<f32> = p.iter().zip(&q).map(|(x, y)| (x + y) / 2.0).collect();

    // Compute KL divergences
    let kl_pm = kl_divergence(&p, &m);
    let kl_qm = kl_divergence(&q, &m);

    let js_distance = (kl_pm + kl_qm) / 2.0;
    Ok(1.0 - js_distance.sqrt()) // Convert distance to similarity
}

fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q)
        .map(|(pi, qi)| {
            if *pi > 0.0 && *qi > 0.0 {
                pi * (pi / qi).ln()
            } else {
                0.0
            }
        })
        .sum()
}

fn bhattacharyya_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    let sum_a: f32 = a.iter().sum();
    let sum_b: f32 = b.iter().sum();

    if sum_a == 0.0 || sum_b == 0.0 {
        return Ok(0.0);
    }

    let p: Vec<f32> = a.iter().map(|x| x / sum_a).collect();
    let q: Vec<f32> = b.iter().map(|x| x / sum_b).collect();

    let bc: f32 = p.iter().zip(&q).map(|(x, y)| (x * y).sqrt()).sum();
    Ok(bc)
}

fn hamming_similarity(a: &[f32], b: &[f32]) -> f32 {
    let threshold = 0.5;
    let matches = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x > threshold) == (**y > threshold))
        .count();

    matches as f32 / a.len() as f32
}

fn canberra_similarity(a: &[f32], b: &[f32]) -> f32 {
    let distance: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let numerator = (x - y).abs();
            let denominator = x.abs() + y.abs();
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        })
        .sum();

    1.0 / (1.0 + distance)
}

fn angular_similarity(a: &[f32], b: &[f32]) -> f32 {
    let cosine_sim = cosine_similarity(a, b);
    let angle = cosine_sim.acos();
    1.0 - (angle / std::f32::consts::PI)
}

fn dot_product_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Simple dot product implementation
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vector_magnitude(vector: &[f32]) -> f32 {
    // Calculate vector magnitude (L2 norm)
    vector.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Invert a square matrix via Gauss-Jordan elimination with partial pivoting.
///
/// Returns `None` if the matrix is singular (or numerically indistinguishable
/// from singular). Used to obtain Σ⁻¹ for the Mahalanobis distance. Pure Rust,
/// no external linear-algebra dependency.
fn invert_matrix(matrix: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    // Build the augmented matrix [A | I] in f64 for numerical stability.
    let mut aug: Vec<Vec<f64>> = matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r: Vec<f64> = row.iter().map(|&v| v as f64).collect();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot: pick the row with the largest absolute value in `col`.
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            let v = aug_row[col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return None; // Singular.
        }
        aug.swap(col, pivot);

        // Normalize the pivot row.
        let pivot_val = aug[col][col];
        for x in aug[col].iter_mut() {
            *x /= pivot_val;
        }

        // Eliminate `col` from every other row. Clone the (already normalized)
        // pivot row so we can iterate the target row mutably without a second
        // index into `aug`.
        let pivot_row = aug[col].clone();
        for (row, target_row) in aug.iter_mut().enumerate() {
            if row == col {
                continue;
            }
            let factor = target_row[col];
            if factor != 0.0 {
                for (target, &pv) in target_row.iter_mut().zip(pivot_row.iter()) {
                    *target -= factor * pv;
                }
            }
        }
    }

    // Extract the right half (the inverse), back into f32.
    Some(
        aug.into_iter()
            .map(|row| row[n..].iter().map(|&v| v as f32).collect())
            .collect(),
    )
}

// Distance function implementations (lower values mean more similar)

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    // Use oxirs-core SIMD operations
    f32::euclidean_distance(a, b)
}

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    // Use oxirs-core SIMD operations
    f32::manhattan_distance(a, b)
}

fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> f32 {
    if p <= 0.0 {
        return euclidean_distance(a, b);
    }

    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f32>()
        .powf(1.0 / p)
}

fn chebyshev_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |acc, diff| acc.max(diff))
}

fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    let threshold = 0.5;
    let mismatches = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x > threshold) != (**y > threshold))
        .count();

    mismatches as f32 / a.len() as f32
}

fn canberra_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let numerator = (x - y).abs();
            let denominator = x.abs() + y.abs();
            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        })
        .sum()
}

/// Similarity search result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub id: String,
    pub uri: String,
    pub similarity: f32,
    pub metrics: HashMap<String, f32>,
    pub metadata: Option<HashMap<String, String>>,
}

/// Batch similarity processor for efficient computation
pub struct BatchSimilarityProcessor {
    similarity: SemanticSimilarity,
    cache: HashMap<(String, String), f32>,
    max_cache_size: usize,
}

impl BatchSimilarityProcessor {
    pub fn new(config: SimilarityConfig, max_cache_size: usize) -> Self {
        Self {
            similarity: SemanticSimilarity::new(config),
            cache: HashMap::new(),
            max_cache_size,
        }
    }

    /// Process batch of similarity computations with caching
    pub fn process_batch(
        &mut self,
        queries: &[(String, Vector)],
        candidates: &[(String, Vector)],
    ) -> Result<Vec<Vec<SimilarityResult>>> {
        let mut results = Vec::new();

        for (query_uri, query_vec) in queries {
            let mut query_results = Vec::new();

            for (candidate_uri, candidate_vec) in candidates {
                let cache_key = if query_uri < candidate_uri {
                    (query_uri.clone(), candidate_uri.clone())
                } else {
                    (candidate_uri.clone(), query_uri.clone())
                };

                let similarity = if let Some(&cached_sim) = self.cache.get(&cache_key) {
                    cached_sim
                } else {
                    let sim = self.similarity.similarity(query_vec, candidate_vec)?;

                    // Cache management
                    if self.cache.len() >= self.max_cache_size {
                        // Simple eviction: remove first entry
                        if let Some(key) = self.cache.keys().next().cloned() {
                            self.cache.remove(&key);
                        }
                    }

                    self.cache.insert(cache_key, sim);
                    sim
                };

                query_results.push(SimilarityResult {
                    id: generate_similarity_id(candidate_uri, similarity),
                    uri: candidate_uri.clone(),
                    similarity,
                    metrics: HashMap::new(),
                    metadata: None,
                });
            }

            // Sort by similarity (descending)
            query_results.sort_by(|a, b| {
                b.similarity
                    .partial_cmp(&a.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.push(query_results);
        }

        Ok(results)
    }

    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.max_cache_size)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Generate a unique ID for similarity results
fn generate_similarity_id(uri: &str, similarity: f32) -> String {
    let mut hasher = DefaultHasher::new();
    uri.hash(&mut hasher);
    similarity.to_bits().hash(&mut hasher);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    timestamp.hash(&mut hasher);

    format!("sim_{:x}", hasher.finish())
}

#[cfg(test)]
mod mahalanobis_tests {
    use super::*;
    use crate::distance_metrics::ExtendedDistanceMetric;

    #[test]
    fn regression_stateless_mahalanobis_fails_loud() {
        // The stateless metric APIs must NOT silently return Euclidean disguised
        // as Mahalanobis — they must error.
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!(SimilarityMetric::Mahalanobis.similarity(&a, &b).is_err());
        let va = Vector::new(a.to_vec());
        let vb = Vector::new(b.to_vec());
        assert!(SimilarityMetric::Mahalanobis.distance(&va, &vb).is_err());
        assert!(ExtendedDistanceMetric::Mahalanobis
            .distance(&va, &vb)
            .is_err());
    }

    #[test]
    fn regression_mahalanobis_identity_equals_euclidean() -> Result<()> {
        // With Σ = I, Mahalanobis distance reduces to Euclidean distance.
        let mut sem = SemanticSimilarity::new(SimilarityConfig {
            primary_metric: SimilarityMetric::Mahalanobis,
            ..Default::default()
        });
        sem.set_covariance_matrix(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0, 3.0];
        let maha = sem.mahalanobis_distance(&a, &b)?;
        let eucl = ((3.0f32).powi(2) + (4.0f32).powi(2)).sqrt(); // = 5.0
        assert!((maha - eucl).abs() < 1e-4, "maha={maha}, eucl={eucl}");
        Ok(())
    }

    #[test]
    fn regression_mahalanobis_uses_covariance() -> Result<()> {
        // A diagonal covariance with a large variance on axis 0 should shrink
        // that axis's contribution, so it must differ from plain Euclidean.
        let mut sem = SemanticSimilarity::new(SimilarityConfig {
            primary_metric: SimilarityMetric::Mahalanobis,
            ..Default::default()
        });
        sem.set_covariance_matrix(vec![vec![100.0, 0.0], vec![0.0, 1.0]]);
        let a = [0.0f32, 0.0];
        let b = [10.0f32, 0.0];
        // d = sqrt((10)^2 / 100) = 1.0, NOT the Euclidean 10.0.
        let maha = sem.mahalanobis_distance(&a, &b)?;
        assert!((maha - 1.0).abs() < 1e-4, "maha={maha}");
        Ok(())
    }

    #[test]
    fn regression_mahalanobis_requires_covariance() {
        let sem = SemanticSimilarity::new(SimilarityConfig {
            primary_metric: SimilarityMetric::Mahalanobis,
            ..Default::default()
        });
        assert!(sem.mahalanobis_distance(&[1.0, 2.0], &[3.0, 4.0]).is_err());
    }

    #[test]
    fn regression_mahalanobis_singular_covariance_errs() {
        let mut sem = SemanticSimilarity::new(SimilarityConfig::default());
        // A zero matrix is singular -> must error, not silently proceed.
        sem.set_covariance_matrix(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert!(sem.mahalanobis_distance(&[1.0, 2.0], &[3.0, 4.0]).is_err());
    }
}
