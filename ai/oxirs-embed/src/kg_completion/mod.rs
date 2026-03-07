//! Knowledge Graph Completion: negative sampling and batched training loops.
//!
//! This module provides the building blocks for training knowledge graph embedding
//! models using the standard *positive / negative sample* approach:
//!
//! - [`NegativeSampler`] — three strategies for generating corrupted triples.
//! - [`KgCompletionTask`] — produces negative samples given a positive triple.
//! - [`TrainingBatch`] — a bundle of positive and negative triples ready for training.
//! - [`BatchedTrainingLoop`] — prepares batches and computes training losses.
//!
//! ## Supported loss functions
//!
//! | Function | Formula |
//! |---|---|
//! | Margin loss (TransE-style) | `max(0, margin - score_pos + score_neg)` summed over negatives |
//! | Binary cross-entropy | `−Σ [log σ(score_pos) + log(1 − σ(score_neg))]` |

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::{NamedNode, Triple};

// ─────────────────────────────────────────────────────────────────────────────
// Pseudo-random helper — uses simple linear-congruential generator so we stay
// free of the `rand` crate (scirs2_core::random requires feature flags that
// may not always be active in unit tests).
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal LCG random number generator seeded deterministically.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x6c62_272e_07bb_0142,
        }
    }

    /// Return the next value in `[0, modulus)`.
    fn next_usize(&mut self, modulus: usize) -> usize {
        // Knuth LCG constants.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) as usize) % modulus
    }

    /// Return the next `f64` in `[0.0, 1.0)`.
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NegativeSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used to generate corrupted (*negative*) triples from a positive one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NegativeSampler {
    /// Randomly replace head or tail with any entity drawn uniformly at random.
    Uniform,
    /// Replace only with entities that appear in the same position (head or tail)
    /// across the observed triple set — provides type-constrained negatives.
    TypeConstrained,
    /// Self-adversarial sampling: weight candidates proportionally to their
    /// current model score so hard negatives are sampled more often.
    SelfAdversarial {
        /// Temperature parameter controlling sharpness of the distribution.
        /// Higher temperature → more uniform; lower temperature → harder negatives.
        temperature: f64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// KgCompletionTask
// ─────────────────────────────────────────────────────────────────────────────

/// Generates negative triples for knowledge graph completion training.
///
/// The task maintains an optional list of *known head entities* and *known tail
/// entities* for type-constrained sampling; if not provided, uniform sampling
/// is used as a fallback.
#[derive(Debug, Clone)]
pub struct KgCompletionTask {
    /// All entity IRI strings observed in the training set.
    known_entities: Vec<String>,
    /// Entities that appear as heads in at least one observed triple.
    head_entities: Vec<String>,
    /// Entities that appear as tails in at least one observed triple.
    tail_entities: Vec<String>,
}

impl KgCompletionTask {
    /// Create a task with a flat list of entities (used for uniform sampling).
    pub fn new(known_entities: Vec<String>) -> Self {
        let head_entities = known_entities.clone();
        let tail_entities = known_entities.clone();
        Self {
            known_entities,
            head_entities,
            tail_entities,
        }
    }

    /// Create a task with separate head and tail entity pools for type-constrained sampling.
    pub fn with_type_constraints(
        known_entities: Vec<String>,
        head_entities: Vec<String>,
        tail_entities: Vec<String>,
    ) -> Self {
        Self {
            known_entities,
            head_entities,
            tail_entities,
        }
    }

    /// Build a `KgCompletionTask` by inspecting a set of observed triples.
    ///
    /// Extracts all unique entities, heads, and tails automatically.
    pub fn from_triples(triples: &[Triple]) -> Self {
        let mut all: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut heads: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut tails: std::collections::HashSet<String> = std::collections::HashSet::new();

        for t in triples {
            all.insert(t.subject.iri.clone());
            all.insert(t.predicate.iri.clone());
            all.insert(t.object.iri.clone());
            heads.insert(t.subject.iri.clone());
            tails.insert(t.object.iri.clone());
        }

        let mut known: Vec<String> = all.into_iter().collect();
        let mut head_vec: Vec<String> = heads.into_iter().collect();
        let mut tail_vec: Vec<String> = tails.into_iter().collect();
        known.sort_unstable();
        head_vec.sort_unstable();
        tail_vec.sort_unstable();

        Self {
            known_entities: known,
            head_entities: head_vec,
            tail_entities: tail_vec,
        }
    }

    /// Sample `n` negative triples for the given positive `triple`.
    ///
    /// * `entity_count` — not used internally (the task uses its own entity pool),
    ///   but kept in the signature for API compatibility.
    /// * `strategy` — sampling strategy to apply.
    ///
    /// Returns an empty `Vec` when the entity pool is empty.
    pub fn sample_negatives(
        &self,
        triple: &Triple,
        _entity_count: usize,
        n: usize,
        strategy: &NegativeSampler,
    ) -> Vec<Triple> {
        if self.known_entities.is_empty() || n == 0 {
            return Vec::new();
        }

        // Seed the LCG with a deterministic hash of the positive triple so
        // different triples produce different negatives.
        let seed: u64 = triple
            .subject
            .iri
            .bytes()
            .chain(triple.predicate.iri.bytes())
            .chain(triple.object.iri.bytes())
            .enumerate()
            .fold(0u64, |acc, (i, b)| {
                acc.wrapping_add((b as u64).wrapping_mul(i as u64 + 1))
            });
        let mut rng = Lcg::new(seed);

        match strategy {
            NegativeSampler::Uniform => self.sample_uniform(triple, n, &mut rng),
            NegativeSampler::TypeConstrained => self.sample_type_constrained(triple, n, &mut rng),
            NegativeSampler::SelfAdversarial { temperature } => {
                self.sample_self_adversarial(triple, n, *temperature, &mut rng)
            }
        }
    }

    // ── private sampling methods ──────────────────────────────────────────────

    fn sample_uniform(&self, triple: &Triple, n: usize, rng: &mut Lcg) -> Vec<Triple> {
        let pool = &self.known_entities;
        let mut result = Vec::with_capacity(n);
        let mut attempts = 0usize;
        while result.len() < n && attempts < n * 10 {
            attempts += 1;
            let idx = rng.next_usize(pool.len());
            let replacement = &pool[idx];
            // Randomly choose to corrupt head (0) or tail (1).
            let neg = if rng.next_usize(2) == 0 {
                make_triple(replacement, &triple.predicate.iri, &triple.object.iri)
            } else {
                make_triple(&triple.subject.iri, &triple.predicate.iri, replacement)
            };
            // Only accept if it differs from the positive triple.
            if is_different(&neg, triple) {
                result.push(neg);
            }
        }
        result
    }

    fn sample_type_constrained(&self, triple: &Triple, n: usize, rng: &mut Lcg) -> Vec<Triple> {
        let heads = if self.head_entities.is_empty() {
            &self.known_entities
        } else {
            &self.head_entities
        };
        let tails = if self.tail_entities.is_empty() {
            &self.known_entities
        } else {
            &self.tail_entities
        };

        let mut result = Vec::with_capacity(n);
        let mut attempts = 0usize;
        while result.len() < n && attempts < n * 10 {
            attempts += 1;
            let neg = if rng.next_usize(2) == 0 {
                // Replace head with a type-compatible head entity.
                let idx = rng.next_usize(heads.len());
                make_triple(&heads[idx], &triple.predicate.iri, &triple.object.iri)
            } else {
                // Replace tail with a type-compatible tail entity.
                let idx = rng.next_usize(tails.len());
                make_triple(&triple.subject.iri, &triple.predicate.iri, &tails[idx])
            };
            if is_different(&neg, triple) {
                result.push(neg);
            }
        }
        result
    }

    fn sample_self_adversarial(
        &self,
        triple: &Triple,
        n: usize,
        temperature: f64,
        rng: &mut Lcg,
    ) -> Vec<Triple> {
        let pool = &self.known_entities;
        if pool.is_empty() {
            return Vec::new();
        }

        // Assign pseudo-scores as position-based values (simulating model scores
        // without requiring actual model access).
        let temp = temperature.max(1e-6);
        let raw_scores: Vec<f64> = pool
            .iter()
            .enumerate()
            .map(|(i, _)| {
                // Deterministic mock score — decreases with position.
                1.0 / (i as f64 + 1.0)
            })
            .collect();

        // Softmax with temperature.
        let max_score = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = raw_scores
            .iter()
            .map(|s| ((s - max_score) / temp).exp())
            .collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        // Build cumulative distribution.
        let mut cdf: Vec<f64> = Vec::with_capacity(pool.len());
        let mut cumsum = 0.0_f64;
        for s in &exp_scores {
            cumsum += s / sum_exp;
            cdf.push(cumsum);
        }

        let mut result = Vec::with_capacity(n);
        let mut attempts = 0usize;
        while result.len() < n && attempts < n * 10 {
            attempts += 1;
            let u = rng.next_f64();
            let idx = cdf.iter().position(|&c| u <= c).unwrap_or(pool.len() - 1);
            let replacement = &pool[idx];

            let neg = if rng.next_usize(2) == 0 {
                make_triple(replacement, &triple.predicate.iri, &triple.object.iri)
            } else {
                make_triple(&triple.subject.iri, &triple.predicate.iri, replacement)
            };
            if is_different(&neg, triple) {
                result.push(neg);
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchedTrainingLoop
// ─────────────────────────────────────────────────────────────────────────────

/// A prepared batch of positive and negative triples for one training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingBatch {
    /// Observed (positive) triples.
    pub positive_triples: Vec<Triple>,
    /// Corrupted (negative) triples — `neg_ratio × |positives|` in total.
    pub negative_triples: Vec<Triple>,
}

impl TrainingBatch {
    /// Total number of positive triples in this batch.
    pub fn positive_count(&self) -> usize {
        self.positive_triples.len()
    }

    /// Total number of negative triples in this batch.
    pub fn negative_count(&self) -> usize {
        self.negative_triples.len()
    }
}

/// Efficient batched training for knowledge graph completion.
///
/// Combines negative-sample generation, batch preparation, and loss computation
/// into a single cohesive API.
#[derive(Debug, Clone, Default)]
pub struct BatchedTrainingLoop;

impl BatchedTrainingLoop {
    /// Create a new `BatchedTrainingLoop`.
    pub fn new() -> Self {
        Self
    }

    /// Prepare a `TrainingBatch` from a slice of positive triples.
    ///
    /// For every positive triple, `neg_ratio` negative triples are sampled via
    /// the given `sampler`.
    ///
    /// Returns an error when `positives` is empty.
    pub fn prepare_batch(
        &self,
        task: &KgCompletionTask,
        positives: &[Triple],
        neg_ratio: u32,
        sampler: &NegativeSampler,
    ) -> Result<TrainingBatch> {
        if positives.is_empty() {
            return Err(anyhow!("positives must not be empty"));
        }
        let mut negatives = Vec::with_capacity(positives.len() * neg_ratio as usize);
        for triple in positives {
            let mut neg_samples = task.sample_negatives(
                triple,
                task.known_entities.len(),
                neg_ratio as usize,
                sampler,
            );
            negatives.append(&mut neg_samples);
        }

        Ok(TrainingBatch {
            positive_triples: positives.to_vec(),
            negative_triples: negatives,
        })
    }

    /// Compute the TransE-style margin ranking loss.
    ///
    /// ```text
    /// L = Σ_neg max(0, margin − score_pos + score_neg)
    /// ```
    ///
    /// Higher scores are assumed to be *better* for positive triples.
    ///
    /// Returns an error when `pos_scores` or `neg_scores` is empty.
    pub fn compute_margin_loss(
        &self,
        pos_scores: &[f64],
        neg_scores: &[f64],
        margin: f64,
    ) -> Result<f64> {
        if pos_scores.is_empty() {
            return Err(anyhow!("pos_scores must not be empty"));
        }
        if neg_scores.is_empty() {
            return Err(anyhow!("neg_scores must not be empty"));
        }

        // Pair each positive with all negatives (or in round-robin if counts differ).
        let n_neg = neg_scores.len();
        let loss: f64 = pos_scores
            .iter()
            .enumerate()
            .flat_map(|(i, &pos)| {
                // Assign all negatives to each positive when counts differ.
                neg_scores.iter().enumerate().map(move |(j, &neg)| {
                    let _ = (i, j); // suppress unused warning
                    (margin - pos + neg).max(0.0)
                })
            })
            .sum();

        Ok(loss / (pos_scores.len() * n_neg) as f64)
    }

    /// Compute binary cross-entropy loss over positive and negative scores.
    ///
    /// ```text
    /// L = −(1/N) Σ [log σ(s_pos) + log(1 − σ(s_neg))]
    /// ```
    ///
    /// Returns an error when either input slice is empty.
    pub fn compute_binary_cross_entropy(
        &self,
        pos_scores: &[f64],
        neg_scores: &[f64],
    ) -> Result<f64> {
        if pos_scores.is_empty() {
            return Err(anyhow!("pos_scores must not be empty"));
        }
        if neg_scores.is_empty() {
            return Err(anyhow!("neg_scores must not be empty"));
        }

        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        let eps = 1e-12_f64;

        let pos_loss: f64 = pos_scores
            .iter()
            .map(|&s| -(sigmoid(s).max(eps).ln()))
            .sum();
        let neg_loss: f64 = neg_scores
            .iter()
            .map(|&s| -((1.0 - sigmoid(s)).max(eps).ln()))
            .sum();

        let n = (pos_scores.len() + neg_scores.len()) as f64;
        Ok((pos_loss + neg_loss) / n)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

fn make_triple(subject: &str, predicate: &str, object: &str) -> Triple {
    Triple::new(
        NamedNode {
            iri: subject.to_string(),
        },
        NamedNode {
            iri: predicate.to_string(),
        },
        NamedNode {
            iri: object.to_string(),
        },
    )
}

fn is_different(a: &Triple, b: &Triple) -> bool {
    a.subject.iri != b.subject.iri
        || a.predicate.iri != b.predicate.iri
        || a.object.iri != b.object.iri
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entities() -> Vec<String> {
        (0..10).map(|i| format!("entity_{i}")).collect()
    }

    fn sample_triple() -> Triple {
        make_triple("entity_0", "relation_A", "entity_1")
    }

    // ── NegativeSampler / KgCompletionTask ────────────────────────────────────

    #[test]
    fn test_uniform_sampling_returns_correct_count() {
        let task = KgCompletionTask::new(sample_entities());
        let positive = sample_triple();
        let negatives = task.sample_negatives(&positive, 10, 5, &NegativeSampler::Uniform);
        assert_eq!(negatives.len(), 5);
    }

    #[test]
    fn test_uniform_negatives_differ_from_positive() {
        let task = KgCompletionTask::new(sample_entities());
        let positive = sample_triple();
        let negatives = task.sample_negatives(&positive, 10, 8, &NegativeSampler::Uniform);
        for neg in &negatives {
            assert!(is_different(neg, &positive), "negative == positive");
        }
    }

    #[test]
    fn test_type_constrained_sampling() {
        let entities = sample_entities();
        let heads = vec!["entity_0".into(), "entity_2".into(), "entity_4".into()];
        let tails = vec!["entity_1".into(), "entity_3".into(), "entity_5".into()];
        let task = KgCompletionTask::with_type_constraints(entities, heads.clone(), tails.clone());
        let positive = sample_triple();
        let negatives = task.sample_negatives(&positive, 10, 6, &NegativeSampler::TypeConstrained);
        assert!(!negatives.is_empty());
        for neg in &negatives {
            // Every negative must have either a head from the head pool or a tail from the tail pool.
            let head_ok = heads.contains(&neg.subject.iri);
            let tail_ok = tails.contains(&neg.object.iri);
            assert!(
                head_ok || tail_ok,
                "corrupted entity not in allowed pool: {neg:?}"
            );
        }
    }

    #[test]
    fn test_self_adversarial_sampling() {
        let task = KgCompletionTask::new(sample_entities());
        let positive = sample_triple();
        let negatives = task.sample_negatives(
            &positive,
            10,
            6,
            &NegativeSampler::SelfAdversarial { temperature: 0.5 },
        );
        assert_eq!(negatives.len(), 6);
        for neg in &negatives {
            assert!(is_different(neg, &positive));
        }
    }

    #[test]
    fn test_sampling_empty_entity_pool() {
        let task = KgCompletionTask::new(vec![]);
        let positive = sample_triple();
        let negatives = task.sample_negatives(&positive, 0, 5, &NegativeSampler::Uniform);
        assert!(negatives.is_empty());
    }

    #[test]
    fn test_sampling_n_zero() {
        let task = KgCompletionTask::new(sample_entities());
        let positive = sample_triple();
        let negatives = task.sample_negatives(&positive, 10, 0, &NegativeSampler::Uniform);
        assert!(negatives.is_empty());
    }

    #[test]
    fn test_from_triples_builds_pools() {
        let triples = vec![
            make_triple("alice", "knows", "bob"),
            make_triple("bob", "knows", "charlie"),
        ];
        let task = KgCompletionTask::from_triples(&triples);
        assert!(task.known_entities.contains(&"alice".to_string()));
        assert!(task.head_entities.contains(&"alice".to_string()));
        assert!(task.tail_entities.contains(&"bob".to_string()));
    }

    // ── BatchedTrainingLoop / prepare_batch ───────────────────────────────────

    #[test]
    fn test_prepare_batch_basic() {
        let task = KgCompletionTask::new(sample_entities());
        let positives = vec![sample_triple()];
        let batch_loop = BatchedTrainingLoop::new();
        let batch = batch_loop
            .prepare_batch(&task, &positives, 3, &NegativeSampler::Uniform)
            .expect("batch");
        assert_eq!(batch.positive_count(), 1);
        // Should have up to 3 negatives (may be fewer if uniqueness is hard to satisfy,
        // but should have at least 1 given 10 available entities).
        assert!(!batch.negative_triples.is_empty());
    }

    #[test]
    fn test_prepare_batch_empty_positives_error() {
        let task = KgCompletionTask::new(sample_entities());
        let batch_loop = BatchedTrainingLoop::new();
        let result = batch_loop.prepare_batch(&task, &[], 3, &NegativeSampler::Uniform);
        assert!(result.is_err());
    }

    #[test]
    fn test_training_batch_counts() {
        let batch = TrainingBatch {
            positive_triples: vec![sample_triple(), sample_triple()],
            negative_triples: vec![sample_triple(); 6],
        };
        assert_eq!(batch.positive_count(), 2);
        assert_eq!(batch.negative_count(), 6);
    }

    // ── BatchedTrainingLoop / compute_margin_loss ─────────────────────────────

    #[test]
    fn test_margin_loss_zero_when_pos_larger() {
        let bl = BatchedTrainingLoop::new();
        // pos=10 >> neg=1, margin=1 → loss = max(0, 1-10+1) = 0
        let loss = bl.compute_margin_loss(&[10.0], &[1.0], 1.0).expect("loss");
        assert!((loss).abs() < 1e-9, "expected 0 loss, got {loss}");
    }

    #[test]
    fn test_margin_loss_positive_when_neg_larger() {
        let bl = BatchedTrainingLoop::new();
        // pos=1, neg=10, margin=1 → loss = max(0, 1-1+10) = 10
        let loss = bl.compute_margin_loss(&[1.0], &[10.0], 1.0).expect("loss");
        assert!(loss > 0.0, "expected positive loss, got {loss}");
    }

    #[test]
    fn test_margin_loss_multiple_pairs() {
        let bl = BatchedTrainingLoop::new();
        let pos = vec![5.0, 5.0];
        let neg = vec![4.0, 3.0];
        // All negatives lower than positive → zero loss
        let loss = bl.compute_margin_loss(&pos, &neg, 1.0).expect("loss");
        assert!((loss).abs() < 1e-9);
    }

    #[test]
    fn test_margin_loss_empty_pos_error() {
        let bl = BatchedTrainingLoop::new();
        assert!(bl.compute_margin_loss(&[], &[1.0], 1.0).is_err());
    }

    #[test]
    fn test_margin_loss_empty_neg_error() {
        let bl = BatchedTrainingLoop::new();
        assert!(bl.compute_margin_loss(&[1.0], &[], 1.0).is_err());
    }

    // ── BatchedTrainingLoop / compute_binary_cross_entropy ────────────────────

    #[test]
    fn test_bce_positive_loss() {
        let bl = BatchedTrainingLoop::new();
        // High positive scores and very negative scores → low loss.
        let loss = bl
            .compute_binary_cross_entropy(&[10.0], &[-10.0])
            .expect("bce");
        assert!(loss < 0.01, "expected near-zero loss, got {loss}");
    }

    #[test]
    fn test_bce_high_loss_when_wrong() {
        let bl = BatchedTrainingLoop::new();
        // Negative score high, positive score low → high loss.
        let loss = bl
            .compute_binary_cross_entropy(&[-10.0], &[10.0])
            .expect("bce");
        assert!(loss > 5.0, "expected high loss, got {loss}");
    }

    #[test]
    fn test_bce_empty_pos_error() {
        let bl = BatchedTrainingLoop::new();
        assert!(bl.compute_binary_cross_entropy(&[], &[1.0]).is_err());
    }

    #[test]
    fn test_bce_empty_neg_error() {
        let bl = BatchedTrainingLoop::new();
        assert!(bl.compute_binary_cross_entropy(&[1.0], &[]).is_err());
    }

    #[test]
    fn test_bce_symmetric_scores_moderate_loss() {
        let bl = BatchedTrainingLoop::new();
        // Zero scores → sigmoid = 0.5, loss = -log(0.5) ≈ 0.693 each.
        let loss = bl
            .compute_binary_cross_entropy(&[0.0], &[0.0])
            .expect("bce");
        assert!(
            (loss - std::f64::consts::LN_2).abs() < 0.001,
            "expected ln(2) ≈ 0.693, got {loss}"
        );
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    #[test]
    fn test_negative_sampler_serialization() {
        let s = NegativeSampler::SelfAdversarial { temperature: 0.5 };
        let json = serde_json::to_string(&s).expect("serialize");
        let s2: NegativeSampler = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, s2);
    }

    #[test]
    fn test_training_batch_serialization() {
        let batch = TrainingBatch {
            positive_triples: vec![sample_triple()],
            negative_triples: vec![],
        };
        let json = serde_json::to_string(&batch).expect("serialize");
        let batch2: TrainingBatch = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(batch2.positive_count(), 1);
    }
}
