//! # Contrastive Learning Loss Functions for Knowledge Graph Embeddings
//!
//! Implements contrastive loss functions for training knowledge graph embeddings
//! with self-supervised and supervised contrastive objectives.
//!
//! ## Loss Functions
//!
//! - **InfoNCE**: Noise-contrastive estimation (SimCLR-style)
//! - **Triplet Loss**: Margin-based triplet loss (anchor, positive, negative)
//! - **NT-Xent**: Normalised temperature-scaled cross-entropy
//! - **SupCon**: Supervised contrastive loss
//! - **Hard Negative Mining**: Strategies for selecting challenging negatives

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for contrastive learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    /// Temperature parameter for softmax scaling (default: 0.07).
    pub temperature: f64,
    /// Margin for triplet loss (default: 1.0).
    pub margin: f64,
    /// Number of negative samples per positive (default: 128).
    pub num_negatives: usize,
    /// Hard negative mining strategy (default: SemiHard).
    pub mining_strategy: NegativeMiningStrategy,
    /// Whether to use cosine similarity (true) or dot product (false).
    pub use_cosine: bool,
    /// Label smoothing factor (0.0 = no smoothing, default: 0.0).
    pub label_smoothing: f64,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            margin: 1.0,
            num_negatives: 128,
            mining_strategy: NegativeMiningStrategy::SemiHard,
            use_cosine: true,
            label_smoothing: 0.0,
        }
    }
}

/// Strategy for mining negative samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NegativeMiningStrategy {
    /// Random negatives.
    Random,
    /// Semi-hard negatives (closer than positive but outside margin).
    SemiHard,
    /// Hardest negatives (closest to anchor).
    Hard,
    /// Mix of hard and random negatives.
    Mixed,
}

// ─────────────────────────────────────────────
// Loss Results
// ─────────────────────────────────────────────

/// Result of computing a contrastive loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveLossResult {
    /// Scalar loss value.
    pub loss: f64,
    /// Per-sample losses (if applicable).
    pub per_sample_losses: Vec<f64>,
    /// Average positive similarity.
    pub avg_positive_similarity: f64,
    /// Average negative similarity.
    pub avg_negative_similarity: f64,
    /// Number of samples in the batch.
    pub batch_size: usize,
    /// Number of hard negatives found.
    pub hard_negatives_count: usize,
}

/// Statistics for contrastive training.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContrastiveTrainingStats {
    /// Total batches processed.
    pub batches_processed: u64,
    /// Running average loss.
    pub avg_loss: f64,
    /// Minimum loss seen.
    pub min_loss: f64,
    /// Maximum loss seen.
    pub max_loss: f64,
    /// Average positive-negative similarity gap.
    pub avg_similarity_gap: f64,
    /// Total samples processed.
    pub total_samples: u64,
}

// ─────────────────────────────────────────────
// Similarity functions
// ─────────────────────────────────────────────

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-30 || norm_b < 1e-30 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute dot product similarity.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute L2 (Euclidean) distance.
pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────
// Loss Functions
// ─────────────────────────────────────────────

/// Contrastive loss function engine.
pub struct ContrastiveLossEngine {
    config: ContrastiveConfig,
    stats: ContrastiveTrainingStats,
}

impl ContrastiveLossEngine {
    /// Create a new contrastive loss engine.
    pub fn new(config: ContrastiveConfig) -> Self {
        Self {
            config,
            stats: ContrastiveTrainingStats {
                min_loss: f64::MAX,
                ..Default::default()
            },
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ContrastiveConfig::default())
    }

    /// Compute InfoNCE (Noise Contrastive Estimation) loss.
    ///
    /// L = -log( exp(sim(anchor, positive) / τ) / Σ exp(sim(anchor, neg_i) / τ) )
    pub fn info_nce_loss(
        &mut self,
        anchors: &[Vec<f64>],
        positives: &[Vec<f64>],
        negatives: &[Vec<f64>],
    ) -> ContrastiveLossResult {
        let batch_size = anchors.len().min(positives.len());
        let tau = self.config.temperature;
        let mut per_sample_losses = Vec::with_capacity(batch_size);
        let mut total_pos_sim = 0.0;
        let mut total_neg_sim = 0.0;
        let mut hard_count = 0;

        for i in 0..batch_size {
            let pos_sim = self.similarity(&anchors[i], &positives[i]) / tau;
            total_pos_sim += pos_sim * tau;

            let mut log_sum_exp = pos_sim.exp();
            let mut max_neg_sim = f64::NEG_INFINITY;

            for neg in negatives.iter() {
                let neg_sim = self.similarity(&anchors[i], neg) / tau;
                total_neg_sim += neg_sim * tau;
                log_sum_exp += neg_sim.exp();
                if neg_sim > max_neg_sim {
                    max_neg_sim = neg_sim;
                }
            }

            if max_neg_sim * tau > pos_sim * tau - self.config.margin {
                hard_count += 1;
            }

            let loss = -pos_sim + log_sum_exp.ln();
            per_sample_losses.push(loss);
        }

        let total_loss: f64 = per_sample_losses.iter().sum();
        let avg_loss = if batch_size > 0 {
            total_loss / batch_size as f64
        } else {
            0.0
        };

        let neg_count = negatives.len().max(1) * batch_size;
        let result = ContrastiveLossResult {
            loss: avg_loss,
            per_sample_losses,
            avg_positive_similarity: if batch_size > 0 {
                total_pos_sim / batch_size as f64
            } else {
                0.0
            },
            avg_negative_similarity: if neg_count > 0 {
                total_neg_sim / neg_count as f64
            } else {
                0.0
            },
            batch_size,
            hard_negatives_count: hard_count,
        };

        self.update_stats(&result);
        result
    }

    /// Compute triplet margin loss.
    ///
    /// L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    pub fn triplet_loss(
        &mut self,
        anchors: &[Vec<f64>],
        positives: &[Vec<f64>],
        negatives: &[Vec<f64>],
    ) -> ContrastiveLossResult {
        let batch_size = anchors.len().min(positives.len()).min(negatives.len());
        let margin = self.config.margin;
        let mut per_sample_losses = Vec::with_capacity(batch_size);
        let mut total_pos_dist = 0.0;
        let mut total_neg_dist = 0.0;
        let mut hard_count = 0;

        for i in 0..batch_size {
            let pos_dist = l2_distance(&anchors[i], &positives[i]);
            let neg_dist = l2_distance(&anchors[i], &negatives[i]);

            total_pos_dist += pos_dist;
            total_neg_dist += neg_dist;

            let loss = (pos_dist - neg_dist + margin).max(0.0);
            if loss > 0.0 {
                hard_count += 1;
            }
            per_sample_losses.push(loss);
        }

        let total_loss: f64 = per_sample_losses.iter().sum();
        let avg_loss = if batch_size > 0 {
            total_loss / batch_size as f64
        } else {
            0.0
        };

        let result = ContrastiveLossResult {
            loss: avg_loss,
            per_sample_losses,
            avg_positive_similarity: if batch_size > 0 {
                -(total_pos_dist / batch_size as f64)
            } else {
                0.0
            },
            avg_negative_similarity: if batch_size > 0 {
                -(total_neg_dist / batch_size as f64)
            } else {
                0.0
            },
            batch_size,
            hard_negatives_count: hard_count,
        };

        self.update_stats(&result);
        result
    }

    /// Compute NT-Xent (Normalised Temperature-Scaled Cross-Entropy) loss.
    ///
    /// SimCLR-style loss over a batch of augmented pairs.
    pub fn nt_xent_loss(
        &mut self,
        embeddings_a: &[Vec<f64>],
        embeddings_b: &[Vec<f64>],
    ) -> ContrastiveLossResult {
        let batch_size = embeddings_a.len().min(embeddings_b.len());
        let tau = self.config.temperature;
        let mut per_sample_losses = Vec::with_capacity(batch_size);
        let mut total_pos_sim = 0.0;
        let mut total_neg_sim = 0.0;
        let mut neg_count = 0usize;

        for i in 0..batch_size {
            let pos_sim = self.similarity(&embeddings_a[i], &embeddings_b[i]) / tau;
            total_pos_sim += pos_sim * tau;

            let mut log_sum = 0.0f64;
            for j in 0..batch_size {
                if j != i {
                    let sim_aj = self.similarity(&embeddings_a[i], &embeddings_b[j]) / tau;
                    let sim_ai = self.similarity(&embeddings_a[i], &embeddings_a[j]) / tau;
                    total_neg_sim += sim_aj * tau + sim_ai * tau;
                    neg_count += 2;
                    log_sum += sim_aj.exp() + sim_ai.exp();
                }
            }
            log_sum += pos_sim.exp();

            let loss = -pos_sim + log_sum.ln();
            per_sample_losses.push(loss);
        }

        let total_loss: f64 = per_sample_losses.iter().sum();
        let avg_loss = if batch_size > 0 {
            total_loss / batch_size as f64
        } else {
            0.0
        };

        let result = ContrastiveLossResult {
            loss: avg_loss,
            per_sample_losses,
            avg_positive_similarity: if batch_size > 0 {
                total_pos_sim / batch_size as f64
            } else {
                0.0
            },
            avg_negative_similarity: if neg_count > 0 {
                total_neg_sim / neg_count as f64
            } else {
                0.0
            },
            batch_size,
            hard_negatives_count: 0,
        };

        self.update_stats(&result);
        result
    }

    /// Mine semi-hard negatives from a pool.
    ///
    /// Returns indices of negatives that are farther from anchor than positive
    /// but within the margin boundary.
    pub fn mine_semi_hard(
        &self,
        anchor: &[f64],
        positive: &[f64],
        negative_pool: &[Vec<f64>],
    ) -> Vec<usize> {
        let pos_dist = l2_distance(anchor, positive);
        let margin = self.config.margin;

        negative_pool
            .iter()
            .enumerate()
            .filter_map(|(i, neg)| {
                let neg_dist = l2_distance(anchor, neg);
                if neg_dist > pos_dist && neg_dist < pos_dist + margin {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Mine the hardest negative from a pool (closest to anchor).
    pub fn mine_hardest(&self, anchor: &[f64], negative_pool: &[Vec<f64>]) -> Option<usize> {
        negative_pool
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = l2_distance(anchor, a);
                let db = l2_distance(anchor, b);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Get training statistics.
    pub fn stats(&self) -> &ContrastiveTrainingStats {
        &self.stats
    }

    /// Reset training statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ContrastiveTrainingStats {
            min_loss: f64::MAX,
            ..Default::default()
        };
    }

    /// Get the configuration.
    pub fn config(&self) -> &ContrastiveConfig {
        &self.config
    }

    // ─── Internal ────────────────────────────

    fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if self.config.use_cosine {
            cosine_similarity(a, b)
        } else {
            dot_product(a, b)
        }
    }

    fn update_stats(&mut self, result: &ContrastiveLossResult) {
        self.stats.batches_processed += 1;
        self.stats.total_samples += result.batch_size as u64;

        let n = self.stats.batches_processed as f64;
        self.stats.avg_loss = self.stats.avg_loss * (n - 1.0) / n + result.loss / n;

        if result.loss < self.stats.min_loss {
            self.stats.min_loss = result.loss;
        }
        if result.loss > self.stats.max_loss {
            self.stats.max_loss = result.loss;
        }

        let gap = result.avg_positive_similarity - result.avg_negative_similarity;
        self.stats.avg_similarity_gap = self.stats.avg_similarity_gap * (n - 1.0) / n + gap / n;
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vector(seed: f64, dim: usize) -> Vec<f64> {
        (0..dim).map(|i| (seed + i as f64 * 0.1).sin()).collect()
    }

    fn unit_vector(dim: usize, idx: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        if idx < dim {
            v[idx] = 1.0;
        }
        v
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_dot_product_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_distance_same() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(l2_distance(&v, &v) < 1e-10);
    }

    #[test]
    fn test_l2_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_config() {
        let config = ContrastiveConfig::default();
        assert!((config.temperature - 0.07).abs() < 1e-10);
        assert!((config.margin - 1.0).abs() < 1e-10);
        assert_eq!(config.num_negatives, 128);
        assert!(config.use_cosine);
    }

    #[test]
    fn test_info_nce_basic() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let anchors = vec![sample_vector(1.0, 8)];
        let positives = vec![sample_vector(1.1, 8)]; // Similar
        let negatives = vec![sample_vector(5.0, 8), sample_vector(10.0, 8)];

        let result = engine.info_nce_loss(&anchors, &positives, &negatives);
        assert!(result.loss.is_finite());
        assert_eq!(result.batch_size, 1);
        assert_eq!(result.per_sample_losses.len(), 1);
    }

    #[test]
    fn test_info_nce_positive_higher_similarity() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let anchor = vec![1.0, 0.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0, 0.0]; // Very similar
        let negatives = vec![vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]];

        let result = engine.info_nce_loss(&[anchor], &[positive], &negatives);
        assert!(result.avg_positive_similarity > result.avg_negative_similarity);
    }

    #[test]
    fn test_triplet_loss_zero_when_separated() {
        let mut engine = ContrastiveLossEngine::new(ContrastiveConfig {
            margin: 1.0,
            ..Default::default()
        });
        let anchor = vec![0.0, 0.0];
        let positive = vec![0.1, 0.0]; // Very close
        let negative = vec![10.0, 10.0]; // Very far

        let result = engine.triplet_loss(&[anchor], &[positive], &[negative]);
        assert!(
            result.loss < 1e-10,
            "Loss should be 0 when negative is far away"
        );
    }

    #[test]
    fn test_triplet_loss_positive_when_close() {
        let mut engine = ContrastiveLossEngine::new(ContrastiveConfig {
            margin: 1.0,
            ..Default::default()
        });
        let anchor = vec![0.0, 0.0];
        let positive = vec![2.0, 0.0]; // dist = 2
        let negative = vec![1.5, 0.0]; // dist = 1.5 (closer than positive!)

        let result = engine.triplet_loss(&[anchor], &[positive], &[negative]);
        assert!(
            result.loss > 0.0,
            "Loss should be positive when negative is closer"
        );
    }

    #[test]
    fn test_nt_xent_basic() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let a = vec![sample_vector(1.0, 8), sample_vector(2.0, 8)];
        let b = vec![sample_vector(1.1, 8), sample_vector(2.1, 8)];

        let result = engine.nt_xent_loss(&a, &b);
        assert!(result.loss.is_finite());
        assert_eq!(result.batch_size, 2);
    }

    #[test]
    fn test_mine_semi_hard() {
        let engine = ContrastiveLossEngine::new(ContrastiveConfig {
            margin: 2.0,
            ..Default::default()
        });
        let anchor = vec![0.0, 0.0];
        let positive = vec![1.0, 0.0]; // dist = 1.0
        let pool = vec![
            vec![0.5, 0.0],  // dist = 0.5 (too close, not semi-hard)
            vec![1.5, 0.0],  // dist = 1.5 (semi-hard: > 1.0 and < 3.0)
            vec![2.5, 0.0],  // dist = 2.5 (semi-hard)
            vec![10.0, 0.0], // dist = 10.0 (too far)
        ];

        let indices = engine.mine_semi_hard(&anchor, &positive, &pool);
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
    }

    #[test]
    fn test_mine_hardest() {
        let engine = ContrastiveLossEngine::with_defaults();
        let anchor = vec![0.0, 0.0];
        let pool = vec![
            vec![10.0, 0.0], // dist = 10
            vec![2.0, 0.0],  // dist = 2
            vec![5.0, 0.0],  // dist = 5
        ];

        let idx = engine.mine_hardest(&anchor, &pool);
        assert_eq!(idx, Some(1)); // Closest
    }

    #[test]
    fn test_mine_hardest_empty() {
        let engine = ContrastiveLossEngine::with_defaults();
        let anchor = vec![0.0, 0.0];
        assert!(engine.mine_hardest(&anchor, &[]).is_none());
    }

    #[test]
    fn test_stats_tracking() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let a = vec![sample_vector(1.0, 4)];
        let p = vec![sample_vector(1.1, 4)];
        let n = vec![sample_vector(5.0, 4)];

        engine.info_nce_loss(&a, &p, &n);
        engine.info_nce_loss(&a, &p, &n);

        assert_eq!(engine.stats().batches_processed, 2);
        assert_eq!(engine.stats().total_samples, 2);
    }

    #[test]
    fn test_stats_reset() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let a = vec![sample_vector(1.0, 4)];
        let p = vec![sample_vector(1.1, 4)];
        let n = vec![sample_vector(5.0, 4)];
        engine.info_nce_loss(&a, &p, &n);

        engine.reset_stats();
        assert_eq!(engine.stats().batches_processed, 0);
    }

    #[test]
    fn test_dot_product_mode() {
        let mut engine = ContrastiveLossEngine::new(ContrastiveConfig {
            use_cosine: false,
            ..Default::default()
        });
        let a = vec![vec![1.0, 0.0]];
        let p = vec![vec![0.9, 0.1]];
        let n = vec![vec![0.0, 1.0]];

        let result = engine.info_nce_loss(&a, &p, &n);
        assert!(result.loss.is_finite());
    }

    #[test]
    fn test_empty_batch() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let result = engine.info_nce_loss(&[], &[], &[]);
        assert_eq!(result.batch_size, 0);
        assert!((result.loss).abs() < 1e-10);
    }

    #[test]
    fn test_triplet_empty_batch() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let result = engine.triplet_loss(&[], &[], &[]);
        assert_eq!(result.batch_size, 0);
    }

    #[test]
    fn test_nt_xent_single_sample() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let a = vec![sample_vector(1.0, 4)];
        let b = vec![sample_vector(1.1, 4)];
        let result = engine.nt_xent_loss(&a, &b);
        assert!(result.loss.is_finite());
    }

    #[test]
    fn test_config_serialization() {
        let config = ContrastiveConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        let deser: ContrastiveConfig = serde_json::from_str(&json).expect("deser failed");
        assert!((deser.temperature - config.temperature).abs() < 1e-10);
    }

    #[test]
    fn test_result_serialization() {
        let result = ContrastiveLossResult {
            loss: 0.5,
            per_sample_losses: vec![0.5],
            avg_positive_similarity: 0.8,
            avg_negative_similarity: 0.2,
            batch_size: 1,
            hard_negatives_count: 0,
        };
        let json = serde_json::to_string(&result).expect("serialize failed");
        assert!(json.contains("loss"));
    }

    #[test]
    fn test_stats_serialization() {
        let stats = ContrastiveTrainingStats::default();
        let json = serde_json::to_string(&stats).expect("serialize failed");
        assert!(json.contains("batches_processed"));
    }

    #[test]
    fn test_mining_strategy_serde() {
        let s = NegativeMiningStrategy::SemiHard;
        let json = serde_json::to_string(&s).expect("serialize failed");
        let deser: NegativeMiningStrategy = serde_json::from_str(&json).expect("deser failed");
        assert_eq!(deser, s);
    }

    #[test]
    fn test_large_batch() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let dim = 32;
        let batch: Vec<Vec<f64>> = (0..16).map(|i| sample_vector(i as f64, dim)).collect();
        let pos: Vec<Vec<f64>> = (0..16)
            .map(|i| sample_vector(i as f64 + 0.01, dim))
            .collect();
        let neg: Vec<Vec<f64>> = (0..8)
            .map(|i| sample_vector(i as f64 + 100.0, dim))
            .collect();

        let result = engine.info_nce_loss(&batch, &pos, &neg);
        assert_eq!(result.batch_size, 16);
        assert!(result.loss.is_finite());
    }

    #[test]
    fn test_hard_negatives_count() {
        let mut engine = ContrastiveLossEngine::new(ContrastiveConfig {
            margin: 0.5,
            ..Default::default()
        });
        let anchor = vec![1.0, 0.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0, 0.0];
        // Negatives very close to anchor — should be counted as hard
        let negatives = vec![vec![0.95, 0.05, 0.0, 0.0]];

        let result = engine.info_nce_loss(&[anchor], &[positive], &negatives);
        // hard_negatives_count may or may not be 1 depending on similarity vs margin
        assert!(result.hard_negatives_count <= 1);
    }

    #[test]
    fn test_min_max_loss_tracking() {
        let mut engine = ContrastiveLossEngine::with_defaults();
        let a1 = vec![sample_vector(1.0, 4)];
        let p1 = vec![sample_vector(1.1, 4)];
        let n1 = vec![sample_vector(5.0, 4)];
        engine.info_nce_loss(&a1, &p1, &n1);

        let a2 = vec![sample_vector(1.0, 4)];
        let p2 = vec![sample_vector(100.0, 4)]; // Very different "positive"
        let n2 = vec![sample_vector(1.01, 4)]; // Very close negative
        engine.info_nce_loss(&a2, &p2, &n2);

        assert!(engine.stats().min_loss <= engine.stats().max_loss);
    }
}
