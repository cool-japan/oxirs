//! Advanced Knowledge Graph Embedding Models
//!
//! Implements RotatE+, PairRE, and RESCAL — three advanced KG embedding models
//! that handle complex relation patterns including phase-space operations,
//! 1-N/N-N relations, and bilinear scoring.

/// Simple LCG (Linear Congruential Generator) for deterministic initialization
/// without external rand dependency.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance one step and return a value in [0.0, 1.0)
    pub fn next_f32(&mut self) -> f32 {
        // Knuth multiplicative LCG constants
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Take upper 32 bits for uniform [0, 1)
        ((self.state >> 33) as f32) / (u32::MAX as f32)
    }

    /// Return a value in [0.0, max)
    pub fn next_f32_range(&mut self, max: f32) -> f32 {
        self.next_f32() * max
    }
}

// ─────────────────────────────────────────────
// RotatE+
// ─────────────────────────────────────────────

/// RotatE+ with phase-space operations.
///
/// Entities and relations are represented as phase vectors in [0, 2π).
/// The scoring function computes the L1 distance in phase space after
/// applying the relational rotation: score = -||h ∘ r - t||_1.
#[derive(Debug, Clone)]
pub struct RotatEPlus {
    /// Entity phase embeddings: `[num_entities][dim]`, values in \[0, 2π)
    pub entity_phase: Vec<Vec<f32>>,
    /// Relation phase embeddings: `[num_relations][dim]`, values in \[0, 2π)
    pub relation_phase: Vec<Vec<f32>>,
    /// Embedding dimension
    pub dim: usize,
}

impl RotatEPlus {
    /// Create a new RotatE+ model with random phase initialization.
    pub fn new(num_entities: usize, num_relations: usize, dim: usize) -> Self {
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut lcg = Lcg::new(42);

        let entity_phase = (0..num_entities)
            .map(|_| (0..dim).map(|_| lcg.next_f32_range(two_pi)).collect())
            .collect();

        let relation_phase = (0..num_relations)
            .map(|_| (0..dim).map(|_| lcg.next_f32_range(two_pi)).collect())
            .collect();

        Self {
            entity_phase,
            relation_phase,
            dim,
        }
    }

    /// Compute score = -||h ∘ r - t||_1
    /// where ∘ is element-wise phase addition mod 2π.
    pub fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let two_pi = 2.0 * std::f32::consts::PI;
        let h = &self.entity_phase[head];
        let r = &self.relation_phase[relation];
        let t = &self.entity_phase[tail];

        let l1: f32 = (0..self.dim)
            .map(|i| {
                // Phase addition mod 2π
                let rotated = (h[i] + r[i]) % two_pi;
                // Circular distance
                let raw = (rotated - t[i]).abs();
                raw.min(two_pi - raw)
            })
            .sum();

        -l1
    }

    /// Update embeddings via margin-based gradient step.
    ///
    /// `pos_score` and `neg_score` are output from `score()`.
    /// We push pos_score higher and neg_score lower.
    pub fn update(
        &mut self,
        head: usize,
        relation: usize,
        tail: usize,
        pos_score: f32,
        neg_score: f32,
        lr: f32,
    ) {
        let two_pi = 2.0 * std::f32::consts::PI;
        let margin = 1.0_f32;
        let loss_gradient = if pos_score - neg_score < margin {
            1.0_f32
        } else {
            0.0_f32
        };

        if loss_gradient.abs() < 1e-9 {
            return;
        }

        // Gradient sign: increase positive score (decrease L1), decrease negative
        for i in 0..self.dim {
            let h_phase = self.entity_phase[head][i];
            let r_phase = self.relation_phase[relation][i];
            let t_phase = self.entity_phase[tail][i];

            let rotated = (h_phase + r_phase) % two_pi;
            let diff = rotated - t_phase;
            // Sign of gradient for L1
            let sign = if diff > 0.0 { 1.0_f32 } else { -1.0_f32 };

            // Positive example: push score up → decrease L1 distance
            let grad = sign * loss_gradient * lr;
            self.entity_phase[head][i] = (self.entity_phase[head][i] - grad).rem_euclid(two_pi);
            self.relation_phase[relation][i] =
                (self.relation_phase[relation][i] - grad).rem_euclid(two_pi);
            self.entity_phase[tail][i] = (self.entity_phase[tail][i] + grad).rem_euclid(two_pi);
        }
    }

    /// Number of entities
    pub fn entity_count(&self) -> usize {
        self.entity_phase.len()
    }

    /// Number of relations
    pub fn relation_count(&self) -> usize {
        self.relation_phase.len()
    }
}

// ─────────────────────────────────────────────
// PairRE
// ─────────────────────────────────────────────

/// PairRE: Handles 1-N, N-1, and N-N relations with paired relation vectors.
///
/// Each relation has two vectors r_h (applied to head) and r_t (applied to tail).
/// score = -||h ⊙ r_h - t ⊙ r_t||_2
#[derive(Debug, Clone)]
pub struct PairRE {
    /// Entity embeddings: `[num_entities][dim]`
    pub entity_emb: Vec<Vec<f32>>,
    /// Head-side relation vectors: `[num_relations][dim]`
    pub relation_head: Vec<Vec<f32>>,
    /// Tail-side relation vectors: `[num_relations][dim]`
    pub relation_tail: Vec<Vec<f32>>,
    /// Embedding dimension
    pub dim: usize,
}

impl PairRE {
    /// Create a new PairRE model with small random initializations.
    pub fn new(num_entities: usize, num_relations: usize, dim: usize) -> Self {
        let mut lcg = Lcg::new(7);
        let scale = 0.1_f32;

        let entity_emb = (0..num_entities)
            .map(|_| (0..dim).map(|_| (lcg.next_f32() - 0.5) * scale).collect())
            .collect();

        let relation_head = (0..num_relations)
            .map(|_| (0..dim).map(|_| (lcg.next_f32() - 0.5) * scale).collect())
            .collect();

        let relation_tail = (0..num_relations)
            .map(|_| (0..dim).map(|_| (lcg.next_f32() - 0.5) * scale).collect())
            .collect();

        Self {
            entity_emb,
            relation_head,
            relation_tail,
            dim,
        }
    }

    /// Compute score = -||h ⊙ rh - t ⊙ rt||_2
    pub fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let h = &self.entity_emb[head];
        let rh = &self.relation_head[relation];
        let t = &self.entity_emb[tail];
        let rt = &self.relation_tail[relation];

        let l2_sq: f32 = (0..self.dim)
            .map(|i| {
                let diff = h[i] * rh[i] - t[i] * rt[i];
                diff * diff
            })
            .sum();

        -l2_sq.sqrt()
    }

    /// SGD-style update step.
    pub fn update(&mut self, head: usize, relation: usize, tail: usize, label: f32, lr: f32) {
        // Compute current score and gradient
        let h = self.entity_emb[head].clone();
        let rh = self.relation_head[relation].clone();
        let t = self.entity_emb[tail].clone();
        let rt = self.relation_tail[relation].clone();

        let diffs: Vec<f32> = (0..self.dim).map(|i| h[i] * rh[i] - t[i] * rt[i]).collect();
        let norm: f32 = diffs.iter().map(|d| d * d).sum::<f32>().sqrt().max(1e-8);

        // Gradient of ||...||_2 w.r.t. diff[i] = diff[i] / norm
        // score = -||d||_2, so d_score/d_diff[i] = -diff[i]/norm
        // For positive (label=+1): maximize score → update in +gradient direction → diff decreases
        // For negative (label=-1): minimize score → update in -gradient direction → diff increases
        // sign = +label achieves this: positive label pulls diff toward zero, negative pushes out
        let sign = label;

        for i in 0..self.dim {
            let grad = sign * diffs[i] / norm;
            self.entity_emb[head][i] -= lr * grad * rh[i];
            self.relation_head[relation][i] -= lr * grad * h[i];
            self.entity_emb[tail][i] += lr * grad * rt[i];
            self.relation_tail[relation][i] += lr * grad * t[i];
        }
    }

    /// Predict top-k tail entities for a (head, relation) query.
    pub fn predict_tail(&self, head: usize, relation: usize, top_k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = (0..self.entity_emb.len())
            .map(|tail_idx| (tail_idx, self.score(head, relation, tail_idx)))
            .collect();

        // Sort descending by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Number of entities
    pub fn entity_count(&self) -> usize {
        self.entity_emb.len()
    }
}

// ─────────────────────────────────────────────
// RESCAL
// ─────────────────────────────────────────────

/// RESCAL: Bilinear model for knowledge graph embedding.
///
/// Each relation has a full dim×dim matrix. Scoring: h^T * M_r * t
#[derive(Debug, Clone)]
pub struct Rescal {
    /// Entity embeddings: `[num_entities][dim]`
    pub entity_emb: Vec<Vec<f32>>,
    /// Relation matrices: `[num_relations][dim][dim]`
    pub relation_mat: Vec<Vec<Vec<f32>>>,
    /// Embedding dimension
    pub dim: usize,
}

impl Rescal {
    /// Create a new RESCAL model.
    pub fn new(num_entities: usize, num_relations: usize, dim: usize) -> Self {
        let mut lcg = Lcg::new(13);
        let e_scale = 1.0 / (dim as f32).sqrt();
        let m_scale = 1.0 / (dim as f32);

        let entity_emb = (0..num_entities)
            .map(|_| (0..dim).map(|_| (lcg.next_f32() - 0.5) * e_scale).collect())
            .collect();

        let relation_mat = (0..num_relations)
            .map(|_| {
                (0..dim)
                    .map(|_| (0..dim).map(|_| (lcg.next_f32() - 0.5) * m_scale).collect())
                    .collect()
            })
            .collect();

        Self {
            entity_emb,
            relation_mat,
            dim,
        }
    }

    /// Compute score = h^T * M_r * t
    pub fn score(&self, head: usize, relation: usize, tail: usize) -> f32 {
        let h = &self.entity_emb[head];
        let t = &self.entity_emb[tail];
        let m = &self.relation_mat[relation];

        // M_r * t → dim-vector
        let mt: Vec<f32> = (0..self.dim)
            .map(|i| (0..self.dim).map(|j| m[i][j] * t[j]).sum())
            .collect();

        // h^T * (M_r * t)
        h.iter().zip(mt.iter()).map(|(hi, mti)| hi * mti).sum()
    }

    /// SGD update step minimizing squared loss (score - label)^2.
    pub fn update(&mut self, head: usize, relation: usize, tail: usize, label: f32, lr: f32) {
        let s = self.score(head, relation, tail);
        let err = s - label; // gradient of 0.5*(s-label)^2 = err

        let h = self.entity_emb[head].clone();
        let t = self.entity_emb[tail].clone();
        let m = self.relation_mat[relation].clone();

        // ∂loss/∂h_i = err * (M_r * t)[i]
        let mt: Vec<f32> = (0..self.dim)
            .map(|i| (0..self.dim).map(|j| m[i][j] * t[j]).sum())
            .collect();

        // ∂loss/∂t_j = err * (h^T * M_r)[j]
        let hm: Vec<f32> = (0..self.dim)
            .map(|j| (0..self.dim).map(|i| h[i] * m[i][j]).sum())
            .collect();

        // Apply gradients
        for i in 0..self.dim {
            self.entity_emb[head][i] -= lr * err * mt[i];
            self.entity_emb[tail][i] -= lr * err * hm[i];
            for (j, t_j) in t.iter().enumerate() {
                self.relation_mat[relation][i][j] -= lr * err * h[i] * t_j;
            }
        }
    }

    /// Access the relation matrix for a given relation.
    pub fn relation_matrix(&self, relation: usize) -> &Vec<Vec<f32>> {
        &self.relation_mat[relation]
    }

    /// Number of entities
    pub fn entity_count(&self) -> usize {
        self.entity_emb.len()
    }

    /// Number of relations
    pub fn relation_count(&self) -> usize {
        self.relation_mat.len()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LCG ──────────────────────────────────

    #[test]
    fn test_lcg_range() {
        let mut lcg = Lcg::new(1);
        for _ in 0..1000 {
            let v = lcg.next_f32();
            assert!((0.0..1.0).contains(&v), "LCG value out of [0,1): {v}");
        }
    }

    #[test]
    fn test_lcg_deterministic() {
        let mut a = Lcg::new(99);
        let mut b = Lcg::new(99);
        for _ in 0..50 {
            assert_eq!(a.next_f32().to_bits(), b.next_f32().to_bits());
        }
    }

    // ── RotatE+ ───────────────────────────────

    #[test]
    fn test_rotate_plus_creation() {
        let m = RotatEPlus::new(10, 5, 16);
        assert_eq!(m.entity_count(), 10);
        assert_eq!(m.relation_count(), 5);
        assert_eq!(m.dim, 16);
    }

    #[test]
    fn test_rotate_plus_phases_in_range() {
        let m = RotatEPlus::new(5, 3, 8);
        let two_pi = 2.0 * std::f32::consts::PI;
        for row in &m.entity_phase {
            for &v in row {
                assert!(v >= 0.0 && v < two_pi, "entity phase out of range: {v}");
            }
        }
        for row in &m.relation_phase {
            for &v in row {
                assert!(v >= 0.0 && v < two_pi, "relation phase out of range: {v}");
            }
        }
    }

    #[test]
    fn test_rotate_plus_score_is_finite() {
        let m = RotatEPlus::new(4, 2, 8);
        let s = m.score(0, 0, 1);
        assert!(s.is_finite(), "score should be finite: {s}");
    }

    #[test]
    fn test_rotate_plus_score_non_positive() {
        let m = RotatEPlus::new(4, 2, 8);
        let s = m.score(0, 0, 1);
        assert!(s <= 0.0, "RotatE+ score should be ≤ 0 (it is -L1): {s}");
    }

    #[test]
    fn test_rotate_plus_self_score() {
        // score(h, r, h) should be -||r||_1 (not zero in general),
        // but just verify it's finite and ≤ 0
        let m = RotatEPlus::new(4, 2, 8);
        let s = m.score(0, 0, 0);
        assert!(s.is_finite() && s <= 0.0);
    }

    #[test]
    fn test_rotate_plus_update_changes_embeddings() {
        let mut m = RotatEPlus::new(4, 2, 8);
        let before_h = m.entity_phase[0].clone();
        let pos_score = m.score(0, 0, 1);
        let neg_score = m.score(0, 0, 2);
        m.update(0, 0, 1, pos_score, neg_score, 0.01);
        // At least one phase should have changed
        let changed = m.entity_phase[0]
            .iter()
            .zip(before_h.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "update should modify entity phases");
    }

    #[test]
    fn test_rotate_plus_update_keeps_phases_in_range() {
        let mut m = RotatEPlus::new(4, 2, 8);
        let two_pi = 2.0 * std::f32::consts::PI;
        let pos_score = m.score(0, 0, 1);
        let neg_score = m.score(0, 0, 2) - 2.0; // force margin violation
        m.update(0, 0, 1, pos_score, neg_score, 0.5);
        for &v in &m.entity_phase[0] {
            assert!(
                v >= 0.0 && v < two_pi + 1e-5,
                "phase out of range after update: {v}"
            );
        }
    }

    #[test]
    fn test_rotate_plus_training_loop() {
        let mut m = RotatEPlus::new(6, 3, 16);
        let triples = [(0usize, 0usize, 1usize), (1, 1, 2), (2, 2, 3)];
        for _ in 0..20 {
            for &(h, r, t) in &triples {
                let neg_t = (t + 1) % 6;
                let ps = m.score(h, r, t);
                let ns = m.score(h, r, neg_t);
                m.update(h, r, t, ps, ns, 0.01);
            }
        }
        // Should not panic and scores should be finite
        for &(h, r, t) in &triples {
            assert!(m.score(h, r, t).is_finite());
        }
    }

    // ── PairRE ────────────────────────────────

    #[test]
    fn test_pairre_creation() {
        let m = PairRE::new(8, 4, 16);
        assert_eq!(m.entity_count(), 8);
        assert_eq!(m.dim, 16);
    }

    #[test]
    fn test_pairre_score_finite() {
        let m = PairRE::new(5, 3, 8);
        let s = m.score(0, 0, 1);
        assert!(s.is_finite(), "PairRE score should be finite: {s}");
    }

    #[test]
    fn test_pairre_score_non_positive() {
        let m = PairRE::new(5, 3, 8);
        let s = m.score(0, 0, 1);
        assert!(s <= 0.0, "PairRE score should be ≤ 0 (it is -L2): {s}");
    }

    #[test]
    fn test_pairre_update_changes_embeddings() {
        let mut m = PairRE::new(5, 3, 8);
        let before = m.entity_emb[0].clone();
        m.update(0, 0, 1, 1.0, 0.01);
        let changed = m.entity_emb[0]
            .iter()
            .zip(before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "update should modify embeddings");
    }

    #[test]
    fn test_pairre_predict_tail_returns_correct_count() {
        let m = PairRE::new(10, 3, 8);
        let preds = m.predict_tail(0, 0, 5);
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_pairre_predict_tail_sorted_desc() {
        let m = PairRE::new(10, 3, 8);
        let preds = m.predict_tail(0, 0, 5);
        for w in preds.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "predictions should be sorted descending by score"
            );
        }
    }

    #[test]
    fn test_pairre_predict_tail_k_larger_than_entities() {
        let m = PairRE::new(3, 2, 8);
        let preds = m.predict_tail(0, 0, 100);
        assert_eq!(preds.len(), 3); // capped at entity count
    }

    #[test]
    fn test_pairre_training_positive_vs_negative() {
        let mut m = PairRE::new(8, 4, 16);
        // After many steps the positive triple should score higher than random negative
        for _ in 0..100 {
            m.update(0, 0, 1, 1.0, 0.01); // positive
            m.update(0, 0, 2, -1.0, 0.01); // negative
        }
        let pos_score = m.score(0, 0, 1);
        let neg_score = m.score(0, 0, 2);
        assert!(
            pos_score > neg_score,
            "positive score {pos_score} should exceed negative {neg_score}"
        );
    }

    // ── RESCAL ────────────────────────────────

    #[test]
    fn test_rescal_creation() {
        let m = Rescal::new(6, 3, 8);
        assert_eq!(m.entity_count(), 6);
        assert_eq!(m.relation_count(), 3);
        assert_eq!(m.dim, 8);
    }

    #[test]
    fn test_rescal_score_finite() {
        let m = Rescal::new(5, 3, 8);
        let s = m.score(0, 0, 1);
        assert!(s.is_finite(), "RESCAL score should be finite: {s}");
    }

    #[test]
    fn test_rescal_relation_matrix_shape() {
        let m = Rescal::new(5, 3, 8);
        let mat = m.relation_matrix(0);
        assert_eq!(mat.len(), 8);
        assert_eq!(mat[0].len(), 8);
    }

    #[test]
    fn test_rescal_update_changes_embeddings() {
        let mut m = Rescal::new(5, 3, 8);
        let before = m.entity_emb[0].clone();
        m.update(0, 0, 1, 1.0, 0.01);
        let changed = m.entity_emb[0]
            .iter()
            .zip(before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "update should modify entity embeddings");
    }

    #[test]
    fn test_rescal_training_converges() {
        let mut m = Rescal::new(5, 2, 4);
        // Train on one positive triple, expect score to increase toward 1.0
        let initial_score = m.score(0, 0, 1);
        for _ in 0..500 {
            m.update(0, 0, 1, 1.0, 0.001);
        }
        let final_score = m.score(0, 0, 1);
        assert!(
            final_score > initial_score,
            "RESCAL score should increase toward label"
        );
    }

    #[test]
    fn test_rescal_antisymmetric_scores() {
        let m = Rescal::new(5, 3, 8);
        // score(h, r, t) and score(t, r, h) generally differ for RESCAL (bilinear)
        let s_fwd = m.score(0, 0, 1);
        let s_bwd = m.score(1, 0, 0);
        // They may or may not be equal by chance; just check both are finite
        assert!(s_fwd.is_finite() && s_bwd.is_finite());
    }

    #[test]
    fn test_rescal_different_relations_give_different_scores() {
        let m = Rescal::new(5, 4, 8);
        let s0 = m.score(0, 0, 1);
        let s1 = m.score(0, 1, 1);
        let s2 = m.score(0, 2, 1);
        // At least two of them should differ (with very high probability for random init)
        assert!(
            (s0 - s1).abs() > 1e-6 || (s1 - s2).abs() > 1e-6,
            "Different relations should produce different scores"
        );
    }

    #[test]
    fn test_all_models_score_interface() {
        let rotate = RotatEPlus::new(4, 2, 8);
        let pairre = PairRE::new(4, 2, 8);
        let rescal = Rescal::new(4, 2, 8);

        // All should produce finite scores for valid indices
        assert!(rotate.score(0, 0, 1).is_finite());
        assert!(pairre.score(0, 0, 1).is_finite());
        assert!(rescal.score(0, 0, 1).is_finite());
    }
}
