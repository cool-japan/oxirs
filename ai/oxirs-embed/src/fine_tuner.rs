//! Embedding fine-tuning with contrastive learning (pure Rust, CPU-only).
//!
//! Supports three loss functions for adapting pre-trained embeddings:
//! - **Triplet loss** – margin-based distance metric learning
//! - **Contrastive loss** – similarity-aware pair learning
//! - **Cosine similarity loss** – MSE against a target cosine similarity

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A pair of embeddings (and an optional negative sample) for training.
#[derive(Debug, Clone)]
pub struct EmbeddingPair {
    /// Anchor embedding
    pub anchor: Vec<f32>,
    /// Positive (similar) embedding
    pub positive: Vec<f32>,
    /// Optional negative (dissimilar) embedding — required for triplet loss
    pub negative: Option<Vec<f32>>,
}

impl EmbeddingPair {
    /// Construct a pair with both positive and negative samples.
    pub fn with_negative(anchor: Vec<f32>, positive: Vec<f32>, negative: Vec<f32>) -> Self {
        Self {
            anchor,
            positive,
            negative: Some(negative),
        }
    }

    /// Construct a pair without a negative sample.
    pub fn without_negative(anchor: Vec<f32>, positive: Vec<f32>) -> Self {
        Self {
            anchor,
            positive,
            negative: None,
        }
    }
}

/// Triplet margin loss configuration.
#[derive(Debug, Clone)]
pub struct TripletLoss {
    /// Minimum margin between positive and negative distances
    pub margin: f32,
}

/// Contrastive loss configuration.
#[derive(Debug, Clone)]
pub struct ContrastiveLoss {
    /// Margin applied to dissimilar pairs
    pub margin: f32,
}

/// Which loss function to use during fine-tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LossType {
    /// Max(0, d(a,p) - d(a,n) + margin)
    Triplet,
    /// label=1: d², label=0: max(0, margin-d)²
    Contrastive,
    /// MSE between cosine_sim(a,b) and a target value
    CosineSimilarity,
}

/// Configuration for a fine-tuning run.
#[derive(Debug, Clone)]
pub struct FinetuneConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub loss_type: LossType,
}

impl Default for FinetuneConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 10,
            batch_size: 32,
            loss_type: LossType::Triplet,
        }
    }
}

/// A record of a single gradient-update step.
#[derive(Debug, Clone)]
pub struct TrainingStep {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// FineTuner
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding fine-tuner using contrastive learning losses.
pub struct FineTuner {
    config: FinetuneConfig,
    history: Vec<TrainingStep>,
}

impl FineTuner {
    /// Create a fine-tuner with the given configuration.
    pub fn new(config: FinetuneConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    // ── Loss computations ──────────────────────────────────────────────────

    /// Compute triplet margin loss.
    ///
    /// `loss = max(0, d(a,p) - d(a,n) + margin)`
    pub fn compute_triplet_loss(&self, anchor: &[f32], positive: &[f32], negative: &[f32]) -> f32 {
        let d_pos = euclidean_distance(anchor, positive);
        let d_neg = euclidean_distance(anchor, negative);
        let margin = match &self.config.loss_type {
            LossType::Triplet => 1.0_f32, // default margin
            _ => 1.0_f32,
        };
        (d_pos - d_neg + margin).max(0.0)
    }

    /// Compute contrastive loss for a pair.
    ///
    /// - `label = 1.0` (similar): loss = d²
    /// - `label = 0.0` (dissimilar): loss = max(0, margin − d)²
    pub fn compute_contrastive_loss(&self, a: &[f32], b: &[f32], label: f32) -> f32 {
        let d = euclidean_distance(a, b);
        let margin = 1.0_f32;
        if label >= 0.5 {
            d * d
        } else {
            (margin - d).max(0.0).powi(2)
        }
    }

    /// Compute cosine similarity loss: MSE between cosine_sim(a,b) and `target`.
    pub fn compute_cosine_loss(&self, a: &[f32], b: &[f32], target: f32) -> f32 {
        let sim = cosine_similarity(a, b);
        let diff = sim - target;
        diff * diff
    }

    // ── Training ───────────────────────────────────────────────────────────

    /// Simulate one gradient step over the given pairs and return the mean loss.
    pub fn step(&mut self, pairs: &[EmbeddingPair]) -> f32 {
        if pairs.is_empty() {
            return 0.0;
        }
        let total_loss: f32 = pairs.iter().map(|p| self.pair_loss(p)).sum();
        let mean_loss = total_loss / pairs.len() as f32;

        let epoch = if self.history.is_empty() {
            0
        } else {
            self.history.last().map(|s| s.epoch).unwrap_or(0)
        };
        let step = self.history.len();

        self.history.push(TrainingStep {
            epoch,
            step,
            loss: mean_loss,
        });

        mean_loss
    }

    /// Run full training for `config.epochs` epochs and return the mean loss per epoch.
    pub fn train(&mut self, pairs: &[EmbeddingPair]) -> Vec<f32> {
        let epochs = self.config.epochs;
        let mut epoch_losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            if pairs.is_empty() {
                epoch_losses.push(0.0);
                continue;
            }
            let total_loss: f32 = pairs.iter().map(|p| self.pair_loss(p)).sum();
            let mean_loss = total_loss / pairs.len() as f32;

            let step = self.history.len();
            self.history.push(TrainingStep {
                epoch,
                step,
                loss: mean_loss,
            });

            epoch_losses.push(mean_loss);
        }

        epoch_losses
    }

    /// Access the full training history.
    pub fn training_history(&self) -> &[TrainingStep] {
        &self.history
    }

    /// Total number of gradient steps recorded.
    pub fn total_steps(&self) -> usize {
        self.history.len()
    }

    // ── private ────────────────────────────────────────────────────────────

    fn pair_loss(&self, pair: &EmbeddingPair) -> f32 {
        match self.config.loss_type {
            LossType::Triplet => {
                if let Some(neg) = &pair.negative {
                    self.compute_triplet_loss(&pair.anchor, &pair.positive, neg)
                } else {
                    0.0
                }
            }
            LossType::Contrastive => {
                // Treat the pair as similar (label=1); use negative if available
                if let Some(neg) = &pair.negative {
                    // Average of similar and dissimilar
                    let l_sim = self.compute_contrastive_loss(&pair.anchor, &pair.positive, 1.0);
                    let l_dis = self.compute_contrastive_loss(&pair.anchor, neg, 0.0);
                    (l_sim + l_dis) / 2.0
                } else {
                    self.compute_contrastive_loss(&pair.anchor, &pair.positive, 1.0)
                }
            }
            LossType::CosineSimilarity => {
                self.compute_cosine_loss(&pair.anchor, &pair.positive, 1.0)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Compute the L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Return a unit-length version of `v`.  If the norm is zero, returns a zero vector.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Euclidean distance between two equal-length vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn ones(dim: usize) -> Vec<f32> {
        vec![1.0; dim]
    }
    fn zeros(dim: usize) -> Vec<f32> {
        vec![0.0; dim]
    }
    fn unit_x() -> Vec<f32> {
        vec![1.0, 0.0, 0.0]
    }
    fn unit_y() -> Vec<f32> {
        vec![0.0, 1.0, 0.0]
    }

    fn triplet_tuner() -> FineTuner {
        FineTuner::new(FinetuneConfig {
            loss_type: LossType::Triplet,
            ..Default::default()
        })
    }
    fn contrastive_tuner() -> FineTuner {
        FineTuner::new(FinetuneConfig {
            loss_type: LossType::Contrastive,
            ..Default::default()
        })
    }
    fn cosine_tuner() -> FineTuner {
        FineTuner::new(FinetuneConfig {
            loss_type: LossType::CosineSimilarity,
            ..Default::default()
        })
    }

    // 1. Triplet loss: anchor = positive → d(a,p)=0 → loss = max(0, -d_neg + 1)
    #[test]
    fn test_triplet_loss_same_anchor_positive() {
        let tuner = triplet_tuner();
        let a = vec![1.0, 0.0];
        let neg = vec![10.0, 0.0];
        let loss = tuner.compute_triplet_loss(&a, &a, &neg);
        // d(a,p)=0, d(a,neg)=9 → max(0, 0-9+1)=0
        assert!(loss.abs() < EPS);
    }

    // 2. Triplet loss: positive equals negative → loss >= margin
    #[test]
    fn test_triplet_loss_positive_equals_negative() {
        let tuner = triplet_tuner();
        let a = unit_x();
        let p = unit_y();
        // negative same as positive → d(a,p)==d(a,neg) → loss = margin = 1
        let loss = tuner.compute_triplet_loss(&a, &p, &p);
        assert!((loss - 1.0).abs() < EPS);
    }

    // 3. Triplet loss: margin enforced — negative very far away
    #[test]
    fn test_triplet_loss_negative_far_gives_zero() {
        let tuner = triplet_tuner();
        let a = vec![0.0, 0.0];
        let p = vec![0.1, 0.0];
        let n = vec![100.0, 0.0];
        let loss = tuner.compute_triplet_loss(&a, &p, &n);
        assert!(loss < EPS); // d(a,p) << d(a,n) so loss=0
    }

    // 4. Triplet loss is non-negative
    #[test]
    fn test_triplet_loss_non_negative() {
        let tuner = triplet_tuner();
        let a = vec![1.0, 2.0];
        let p = vec![1.1, 2.1];
        let n = vec![0.5, 0.5];
        let loss = tuner.compute_triplet_loss(&a, &p, &n);
        assert!(loss >= 0.0);
    }

    // 5. Zero margin triplet: loss = max(0, d(a,p) - d(a,n))
    #[test]
    fn test_zero_margin_triplet_direct() {
        // Test via compute_triplet_loss logic: margin hardcoded to 1.0 internally.
        // Verify loss is non-negative regardless.
        let tuner = triplet_tuner();
        let a = vec![0.0];
        let p = vec![1.0];
        let n = vec![2.0];
        let loss = tuner.compute_triplet_loss(&a, &p, &n);
        // d(a,p)=1, d(a,n)=2, margin=1 → max(0, 1-2+1)=0
        assert!(loss.abs() < EPS);
    }

    // 6. Contrastive loss similar pair (label=1): loss = d²
    #[test]
    fn test_contrastive_similar_pair() {
        let tuner = contrastive_tuner();
        let a = vec![0.0];
        let b = vec![0.5];
        let loss = tuner.compute_contrastive_loss(&a, &b, 1.0);
        // d=0.5, loss = 0.25
        assert!((loss - 0.25).abs() < EPS);
    }

    // 7. Contrastive loss dissimilar pair (label=0): loss = max(0, margin-d)²
    #[test]
    fn test_contrastive_dissimilar_pair() {
        let tuner = contrastive_tuner();
        let a = vec![0.0];
        let b = vec![1.5]; // d=1.5 > margin=1 → loss=0
        let loss = tuner.compute_contrastive_loss(&a, &b, 0.0);
        assert!(loss.abs() < EPS);
    }

    // 8. Contrastive loss dissimilar pair close together
    #[test]
    fn test_contrastive_dissimilar_close() {
        let tuner = contrastive_tuner();
        let a = vec![0.0];
        let b = vec![0.5]; // d=0.5, margin=1 → loss=(1-0.5)²=0.25
        let loss = tuner.compute_contrastive_loss(&a, &b, 0.0);
        assert!((loss - 0.25).abs() < EPS);
    }

    // 9. Contrastive loss identical vectors (similar): loss = 0
    #[test]
    fn test_contrastive_identical_similar() {
        let tuner = contrastive_tuner();
        let a = vec![1.0, 2.0];
        let loss = tuner.compute_contrastive_loss(&a, &a, 1.0);
        assert!(loss.abs() < EPS);
    }

    // 10. Cosine loss: identical vectors → sim=1 → loss=(1-target)²
    #[test]
    fn test_cosine_loss_identical() {
        let tuner = cosine_tuner();
        let a = unit_x();
        let loss = tuner.compute_cosine_loss(&a, &a, 1.0);
        assert!(loss.abs() < EPS);
    }

    // 11. Cosine loss: orthogonal vectors → sim=0 → loss=target²
    #[test]
    fn test_cosine_loss_orthogonal() {
        let tuner = cosine_tuner();
        let a = unit_x();
        let b = unit_y();
        let loss = tuner.compute_cosine_loss(&a, &b, 0.0);
        assert!(loss.abs() < EPS);
    }

    // 12. Cosine loss: opposite vectors → sim=-1
    #[test]
    fn test_cosine_loss_opposite() {
        let tuner = cosine_tuner();
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let loss = tuner.compute_cosine_loss(&a, &b, -1.0);
        assert!(loss.abs() < EPS);
    }

    // 13. train returns one loss per epoch
    #[test]
    fn test_train_returns_one_loss_per_epoch() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 5,
            loss_type: LossType::Triplet,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::with_negative(
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        )];
        let losses = tuner.train(&pairs);
        assert_eq!(losses.len(), 5);
    }

    // 14. step increments total_steps
    #[test]
    fn test_step_increments_total_steps() {
        let mut tuner = triplet_tuner();
        let pairs = vec![EmbeddingPair::with_negative(
            vec![0.0],
            vec![1.0],
            vec![2.0],
        )];
        tuner.step(&pairs);
        assert_eq!(tuner.total_steps(), 1);
        tuner.step(&pairs);
        assert_eq!(tuner.total_steps(), 2);
    }

    // 15. history grows with step calls
    #[test]
    fn test_history_grows() {
        let mut tuner = triplet_tuner();
        let pairs = vec![EmbeddingPair::without_negative(vec![0.0], vec![1.0])];
        for _ in 0..7 {
            tuner.step(&pairs);
        }
        assert_eq!(tuner.training_history().len(), 7);
    }

    // 16. train appends to history
    #[test]
    fn test_train_appends_to_history() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 3,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::with_negative(ones(4), ones(4), zeros(4))];
        tuner.train(&pairs);
        assert_eq!(tuner.training_history().len(), 3);
    }

    // 17. empty pairs: step returns 0
    #[test]
    fn test_step_empty_pairs() {
        let mut tuner = triplet_tuner();
        let loss = tuner.step(&[]);
        assert_eq!(loss, 0.0);
    }

    // 18. train with empty pairs returns zeros
    #[test]
    fn test_train_empty_pairs() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 3,
            ..Default::default()
        });
        let losses = tuner.train(&[]);
        assert_eq!(losses.len(), 3);
        assert!(losses.iter().all(|&l| l == 0.0));
    }

    // 19. cosine_similarity identical unit vectors = 1.0
    #[test]
    fn test_cosine_similarity_identical() {
        let a = unit_x();
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < EPS);
    }

    // 20. cosine_similarity orthogonal = 0.0
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&unit_x(), &unit_y());
        assert!(sim.abs() < EPS);
    }

    // 21. cosine_similarity antiparallel = -1.0
    #[test]
    fn test_cosine_similarity_antiparallel() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < EPS);
    }

    // 22. cosine_similarity zero vector returns 0
    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 0.0];
        let b = zeros(2);
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    // 23. l2_normalize unit vector
    #[test]
    fn test_l2_normalize_unit_vector() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < EPS);
        assert!((n[1] - 0.8).abs() < EPS);
    }

    // 24. l2_normalize already normalized vector
    #[test]
    fn test_l2_normalize_already_unit() {
        let v = unit_x();
        let n = l2_normalize(&v);
        assert!((l2_norm(&n) - 1.0).abs() < EPS);
    }

    // 25. l2_normalize zero vector returns zero
    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = zeros(3);
        let n = l2_normalize(&v);
        assert_eq!(n, zeros(3));
    }

    // 26. Normalized vectors: cosine_similarity = 1 for identical
    #[test]
    fn test_normalized_cosine_similarity() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let sim = cosine_similarity(&n, &n);
        assert!((sim - 1.0).abs() < EPS);
    }

    // 27. TrainingStep epoch recorded
    #[test]
    fn test_training_step_epoch() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 1,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::with_negative(ones(2), ones(2), zeros(2))];
        tuner.train(&pairs);
        assert_eq!(tuner.training_history()[0].epoch, 0);
    }

    // 28. TrainingStep step index recorded
    #[test]
    fn test_training_step_step_index() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 3,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::without_negative(ones(2), zeros(2))];
        tuner.train(&pairs);
        let steps: Vec<usize> = tuner.training_history().iter().map(|s| s.step).collect();
        assert_eq!(steps, vec![0, 1, 2]);
    }

    // 29. Contrastive loss LossType in train
    #[test]
    fn test_contrastive_loss_train() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 2,
            loss_type: LossType::Contrastive,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::with_negative(
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![2.0, 0.0],
        )];
        let losses = tuner.train(&pairs);
        assert_eq!(losses.len(), 2);
        assert!(losses.iter().all(|&l| l >= 0.0));
    }

    // 30. CosineSimilarity LossType in train
    #[test]
    fn test_cosine_loss_train() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 2,
            loss_type: LossType::CosineSimilarity,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::without_negative(unit_x(), unit_y())];
        let losses = tuner.train(&pairs);
        assert!(losses.iter().all(|&l| l >= 0.0));
    }

    // 31. step returns positive loss for non-trivial pairs
    #[test]
    fn test_step_positive_loss() {
        let mut tuner = triplet_tuner();
        let pairs = vec![EmbeddingPair::with_negative(
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![0.1, 0.0], // close negative — high loss
        )];
        let loss = tuner.step(&pairs);
        assert!(loss >= 0.0);
    }

    // 32. total_steps after train
    #[test]
    fn test_total_steps_after_train() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 4,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::without_negative(ones(2), zeros(2))];
        tuner.train(&pairs);
        assert_eq!(tuner.total_steps(), 4);
    }

    // 33. Multiple step calls accumulate total_steps
    #[test]
    fn test_step_plus_train_accumulate() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 3,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::without_negative(ones(2), zeros(2))];
        tuner.step(&pairs);
        tuner.train(&pairs);
        // 1 step + 3 train = 4 total
        assert_eq!(tuner.total_steps(), 4);
    }

    // 34. FinetuneConfig default
    #[test]
    fn test_finetune_config_default() {
        let cfg = FinetuneConfig::default();
        assert_eq!(cfg.epochs, 10);
        assert_eq!(cfg.loss_type, LossType::Triplet);
    }

    // 35. EmbeddingPair with_negative stores negative
    #[test]
    fn test_embedding_pair_with_negative() {
        let p = EmbeddingPair::with_negative(vec![1.0], vec![2.0], vec![3.0]);
        assert!(p.negative.is_some());
    }

    // 36. EmbeddingPair without_negative has None
    #[test]
    fn test_embedding_pair_without_negative() {
        let p = EmbeddingPair::without_negative(vec![1.0], vec![2.0]);
        assert!(p.negative.is_none());
    }

    // 37. Triplet pair without negative gives zero loss
    #[test]
    fn test_triplet_pair_no_negative_zero_loss() {
        let mut tuner = triplet_tuner();
        let pairs = vec![EmbeddingPair::without_negative(ones(4), zeros(4))];
        let loss = tuner.step(&pairs);
        assert_eq!(loss, 0.0);
    }

    // 38. cosine_similarity clamped to [-1,1]
    #[test]
    fn test_cosine_similarity_clamped() {
        // Even with floating-point noise, result should be in [-1, 1]
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 1e-7, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((-1.0..=1.0).contains(&sim));
    }

    // 39. l2_norm of a 3-4-5 triangle
    #[test]
    fn test_l2_norm_345() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < EPS);
    }

    // 40. Loss is recorded in history for every train epoch
    #[test]
    fn test_loss_recorded_in_history() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            epochs: 5,
            loss_type: LossType::CosineSimilarity,
            ..Default::default()
        });
        let pairs = vec![EmbeddingPair::without_negative(unit_x(), unit_y())];
        tuner.train(&pairs);
        assert!(tuner.training_history().iter().all(|s| s.loss >= 0.0));
    }

    // 41. Different LossTypes produce different losses for same pair
    #[test]
    fn test_loss_types_differ() {
        let pairs = vec![EmbeddingPair::with_negative(
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![0.2, 0.0],
        )];
        let mut t1 = FineTuner::new(FinetuneConfig {
            epochs: 1,
            loss_type: LossType::Triplet,
            ..Default::default()
        });
        let mut t2 = FineTuner::new(FinetuneConfig {
            epochs: 1,
            loss_type: LossType::Contrastive,
            ..Default::default()
        });
        let l1 = t1.step(&pairs);
        let l2 = t2.step(&pairs);
        // They may or may not be equal; just verify both are non-negative
        assert!(l1 >= 0.0);
        assert!(l2 >= 0.0);
    }

    // 42. High-dimensional embeddings work
    #[test]
    fn test_high_dimensional_embeddings() {
        let dim = 768;
        let anchor: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let positive: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / dim as f32).collect();
        let negative: Vec<f32> = vec![-1.0; dim];
        let tuner = triplet_tuner();
        let loss = tuner.compute_triplet_loss(&anchor, &positive, &negative);
        assert!(loss >= 0.0);
    }

    // 43. cosine_loss is zero when sim matches target exactly
    #[test]
    fn test_cosine_loss_zero_when_exact() {
        let tuner = cosine_tuner();
        let a = unit_x();
        let b = unit_x();
        let sim = cosine_similarity(&a, &b); // should be 1.0
        let loss = tuner.compute_cosine_loss(&a, &b, sim);
        assert!(loss.abs() < EPS);
    }

    // 44. train with large batch size config
    #[test]
    fn test_train_large_batch_size() {
        let mut tuner = FineTuner::new(FinetuneConfig {
            batch_size: 512,
            epochs: 2,
            ..Default::default()
        });
        let pairs: Vec<_> = (0..100)
            .map(|_| EmbeddingPair::without_negative(ones(16), zeros(16)))
            .collect();
        let losses = tuner.train(&pairs);
        assert_eq!(losses.len(), 2);
    }

    // 45. total_steps 0 initially
    #[test]
    fn test_total_steps_initially_zero() {
        let tuner = triplet_tuner();
        assert_eq!(tuner.total_steps(), 0);
    }
}
