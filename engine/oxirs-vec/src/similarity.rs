//! Advanced similarity algorithms and semantic matching for vectors

use crate::Vector;
use anyhow::{anyhow, Result};
use oxirs_core::simd::SimdOps;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Similarity measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
                // Requires covariance matrix - use identity for now
                euclidean_similarity(a, b)
            }
            SimilarityMetric::Hamming => hamming_similarity(a, b),
            SimilarityMetric::Canberra => canberra_similarity(a, b),
            SimilarityMetric::Angular => angular_similarity(a, b),
            SimilarityMetric::Chebyshev => chebyshev_similarity(a, b),
        };

        Ok(similarity.clamp(0.0, 1.0))
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

    /// Calculate similarity using primary metric
    pub fn similarity(&self, a: &Vector, b: &Vector) -> Result<f32> {
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();

        let mut similarity = self.config.primary_metric.similarity(&a_f32, &b_f32)?;

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

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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

fn vector_magnitude(vector: &[f32]) -> f32 {
    // Use oxirs-core SIMD operations
    f32::norm(vector)
}

/// Similarity search result with metadata
#[derive(Debug, Clone)]
pub struct SimilarityResult {
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
                    uri: candidate_uri.clone(),
                    similarity,
                    metrics: HashMap::new(),
                    metadata: None,
                });
            }

            // Sort by similarity (descending)
            query_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
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
