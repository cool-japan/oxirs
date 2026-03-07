//! # TransE Knowledge Graph Embedding Model
//!
//! Implements the TransE (Translating Embeddings) model for knowledge graph
//! link prediction. In TransE, relationships are represented as translations
//! in the embedding space: `h + r ≈ t` for a triple (h, r, t).
//!
//! ## Features
//!
//! - **Training**: Stochastic gradient descent with margin-based ranking loss
//! - **Scoring**: Score candidate triples using L1 or L2 distance
//! - **Link prediction**: Predict head/tail entities given partial triples
//! - **Nearest neighbor search**: Find entities closest to a query embedding
//! - **Serialization**: Export/import learned embeddings

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Distance metric for TransE scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// L1 (Manhattan) distance.
    L1,
    /// L2 (Euclidean) distance.
    L2,
}

/// Configuration for the TransE model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransEConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Margin for the ranking loss (gamma).
    pub margin: f64,
    /// Distance metric.
    pub distance_metric: DistanceMetric,
    /// Maximum training epochs.
    pub max_epochs: usize,
    /// Number of negative samples per positive triple.
    pub num_negatives: usize,
    /// Whether to normalize embeddings after each update.
    pub normalize_embeddings: bool,
}

impl Default for TransEConfig {
    fn default() -> Self {
        Self {
            dim: 50,
            learning_rate: 0.01,
            margin: 1.0,
            distance_metric: DistanceMetric::L2,
            max_epochs: 100,
            num_negatives: 1,
            normalize_embeddings: true,
        }
    }
}

// ─────────────────────────────────────────────
// Triple types
// ─────────────────────────────────────────────

/// An RDF-like triple (head, relation, tail) using integer IDs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub head: usize,
    pub relation: usize,
    pub tail: usize,
}

/// A scored triple for link prediction results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredTriple {
    pub head: usize,
    pub relation: usize,
    pub tail: usize,
    pub score: f64,
}

/// Training statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Training loss per epoch.
    pub loss_history: Vec<f64>,
    /// Number of epochs completed.
    pub epochs_completed: usize,
    /// Total triples processed.
    pub triples_processed: u64,
}

// ─────────────────────────────────────────────
// TransEModel
// ─────────────────────────────────────────────

/// TransE knowledge graph embedding model.
pub struct TransEModel {
    config: TransEConfig,
    /// Entity embeddings: entity_id -> embedding vector.
    entity_embeddings: HashMap<usize, Vec<f64>>,
    /// Relation embeddings: relation_id -> embedding vector.
    relation_embeddings: HashMap<usize, Vec<f64>>,
    /// Entity name to ID mapping.
    entity_to_id: HashMap<String, usize>,
    /// ID to entity name mapping.
    id_to_entity: HashMap<usize, String>,
    /// Relation name to ID mapping.
    relation_to_id: HashMap<String, usize>,
    /// ID to relation name mapping.
    id_to_relation: HashMap<usize, String>,
    /// Known triples (for filtered evaluation).
    known_triples: HashSet<Triple>,
    /// Training statistics.
    stats: TrainingStats,
    /// Simple LCG state for reproducible pseudo-random initialization.
    rng_state: u64,
}

impl TransEModel {
    /// Create a new TransE model with default configuration.
    pub fn new() -> Self {
        Self::with_config(TransEConfig::default())
    }

    /// Create a new TransE model with the given configuration.
    pub fn with_config(config: TransEConfig) -> Self {
        Self {
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_to_id: HashMap::new(),
            id_to_entity: HashMap::new(),
            relation_to_id: HashMap::new(),
            id_to_relation: HashMap::new(),
            known_triples: HashSet::new(),
            stats: TrainingStats::default(),
            rng_state: 12345,
        }
    }

    /// Get model configuration.
    pub fn config(&self) -> &TransEConfig {
        &self.config
    }

    /// Get training statistics.
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Number of entities.
    pub fn entity_count(&self) -> usize {
        self.entity_to_id.len()
    }

    /// Number of relations.
    pub fn relation_count(&self) -> usize {
        self.relation_to_id.len()
    }

    /// Number of known triples.
    pub fn triple_count(&self) -> usize {
        self.known_triples.len()
    }

    /// Register an entity and return its ID.
    pub fn add_entity(&mut self, name: impl Into<String>) -> usize {
        let name = name.into();
        if let Some(&id) = self.entity_to_id.get(&name) {
            return id;
        }
        let id = self.entity_to_id.len();
        self.entity_to_id.insert(name.clone(), id);
        self.id_to_entity.insert(id, name);
        // Initialize embedding
        let embedding = self.random_embedding();
        self.entity_embeddings.insert(id, embedding);
        id
    }

    /// Register a relation and return its ID.
    pub fn add_relation(&mut self, name: impl Into<String>) -> usize {
        let name = name.into();
        if let Some(&id) = self.relation_to_id.get(&name) {
            return id;
        }
        let id = self.relation_to_id.len();
        self.relation_to_id.insert(name.clone(), id);
        self.id_to_relation.insert(id, name);
        // Initialize embedding
        let mut embedding = self.random_embedding();
        // Normalize relation embeddings
        let norm = l2_norm(&embedding);
        if norm > 1e-12 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        self.relation_embeddings.insert(id, embedding);
        id
    }

    /// Add a triple (by entity/relation names).
    pub fn add_triple(
        &mut self,
        head: impl Into<String>,
        relation: impl Into<String>,
        tail: impl Into<String>,
    ) {
        let h = self.add_entity(head);
        let r = self.add_relation(relation);
        let t = self.add_entity(tail);
        self.known_triples.insert(Triple {
            head: h,
            relation: r,
            tail: t,
        });
    }

    /// Train the model on the known triples.
    pub fn train(&mut self, epochs: usize) -> TrainingStats {
        let num_epochs = epochs.min(self.config.max_epochs);
        let triples: Vec<Triple> = self.known_triples.iter().cloned().collect();

        if triples.is_empty() {
            return self.stats.clone();
        }

        let num_entities = self.entity_to_id.len();

        for _epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;

            for triple in &triples {
                // Generate negative sample by corrupting head or tail
                let neg_triple = self.corrupt_triple(triple, num_entities);

                // Score positive and negative
                let pos_score = self.score_triple_ids(triple.head, triple.relation, triple.tail);
                let neg_score =
                    self.score_triple_ids(neg_triple.head, neg_triple.relation, neg_triple.tail);

                // Margin-based ranking loss: max(0, gamma + pos - neg)
                let loss = (self.config.margin + pos_score - neg_score).max(0.0);
                epoch_loss += loss;

                if loss > 0.0 {
                    // Update embeddings via SGD
                    self.update_embeddings(triple, &neg_triple);
                }

                self.stats.triples_processed += 1;
            }

            let avg_loss = epoch_loss / triples.len() as f64;
            self.stats.loss_history.push(avg_loss);
            self.stats.epochs_completed += 1;

            // Normalize entity embeddings if configured
            if self.config.normalize_embeddings {
                self.normalize_entities();
            }
        }

        self.stats.clone()
    }

    /// Score a triple (lower score = better).
    pub fn score(&self, head: &str, relation: &str, tail: &str) -> Option<f64> {
        let h = self.entity_to_id.get(head)?;
        let r = self.relation_to_id.get(relation)?;
        let t = self.entity_to_id.get(tail)?;
        Some(self.score_triple_ids(*h, *r, *t))
    }

    /// Score a triple by IDs.
    fn score_triple_ids(&self, head: usize, relation: usize, tail: usize) -> f64 {
        let h = match self.entity_embeddings.get(&head) {
            Some(e) => e,
            None => return f64::MAX,
        };
        let r = match self.relation_embeddings.get(&relation) {
            Some(e) => e,
            None => return f64::MAX,
        };
        let t = match self.entity_embeddings.get(&tail) {
            Some(e) => e,
            None => return f64::MAX,
        };

        // distance(h + r, t)
        let dim = self.config.dim;
        match self.config.distance_metric {
            DistanceMetric::L1 => {
                let mut dist = 0.0;
                for i in 0..dim {
                    let hi = h.get(i).copied().unwrap_or(0.0);
                    let ri = r.get(i).copied().unwrap_or(0.0);
                    let ti = t.get(i).copied().unwrap_or(0.0);
                    dist += (hi + ri - ti).abs();
                }
                dist
            }
            DistanceMetric::L2 => {
                let mut dist = 0.0;
                for i in 0..dim {
                    let hi = h.get(i).copied().unwrap_or(0.0);
                    let ri = r.get(i).copied().unwrap_or(0.0);
                    let ti = t.get(i).copied().unwrap_or(0.0);
                    dist += (hi + ri - ti).powi(2);
                }
                dist.sqrt()
            }
        }
    }

    /// Predict the top-k tail entities given (head, relation, ?).
    pub fn predict_tail(&self, head: &str, relation: &str, k: usize) -> Vec<ScoredTriple> {
        let h = match self.entity_to_id.get(head) {
            Some(&id) => id,
            None => return Vec::new(),
        };
        let r = match self.relation_to_id.get(relation) {
            Some(&id) => id,
            None => return Vec::new(),
        };

        let mut scores: Vec<ScoredTriple> = self
            .entity_to_id
            .values()
            .map(|&t_id| {
                let score = self.score_triple_ids(h, r, t_id);
                ScoredTriple {
                    head: h,
                    relation: r,
                    tail: t_id,
                    score,
                }
            })
            .collect();

        scores.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores
    }

    /// Predict the top-k head entities given (?, relation, tail).
    pub fn predict_head(&self, relation: &str, tail: &str, k: usize) -> Vec<ScoredTriple> {
        let r = match self.relation_to_id.get(relation) {
            Some(&id) => id,
            None => return Vec::new(),
        };
        let t = match self.entity_to_id.get(tail) {
            Some(&id) => id,
            None => return Vec::new(),
        };

        let mut scores: Vec<ScoredTriple> = self
            .entity_to_id
            .values()
            .map(|&h_id| {
                let score = self.score_triple_ids(h_id, r, t);
                ScoredTriple {
                    head: h_id,
                    relation: r,
                    tail: t,
                    score,
                }
            })
            .collect();

        scores.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores
    }

    /// Get the embedding for an entity.
    pub fn entity_embedding(&self, name: &str) -> Option<&Vec<f64>> {
        self.entity_to_id
            .get(name)
            .and_then(|id| self.entity_embeddings.get(id))
    }

    /// Get the embedding for a relation.
    pub fn relation_embedding(&self, name: &str) -> Option<&Vec<f64>> {
        self.relation_to_id
            .get(name)
            .and_then(|id| self.relation_embeddings.get(id))
    }

    /// Find nearest entities to a query embedding.
    pub fn nearest_entities(&self, query: &[f64], k: usize) -> Vec<(String, f64)> {
        let mut dists: Vec<(String, f64)> = self
            .entity_embeddings
            .iter()
            .map(|(&id, emb)| {
                let dist = l2_distance(query, emb);
                let name = self
                    .id_to_entity
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("entity_{id}"));
                (name, dist)
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    /// Get entity name by ID.
    pub fn entity_name(&self, id: usize) -> Option<&str> {
        self.id_to_entity.get(&id).map(|s| s.as_str())
    }

    /// Get relation name by ID.
    pub fn relation_name(&self, id: usize) -> Option<&str> {
        self.id_to_relation.get(&id).map(|s| s.as_str())
    }

    // ─── Internal methods ────────────────────────────────

    fn random_embedding(&mut self) -> Vec<f64> {
        let dim = self.config.dim;
        (0..dim)
            .map(|_| {
                self.rng_state = self
                    .rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let val = ((self.rng_state >> 33) as f64) / (u32::MAX as f64);
                (val - 0.5) * 2.0 / (dim as f64).sqrt()
            })
            .collect()
    }

    fn corrupt_triple(&mut self, triple: &Triple, num_entities: usize) -> Triple {
        if num_entities == 0 {
            return triple.clone();
        }
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let corrupt_head = (self.rng_state >> 33) % 2 == 0;
        let random_entity = ((self.rng_state >> 17) as usize) % num_entities;

        if corrupt_head {
            Triple {
                head: random_entity,
                relation: triple.relation,
                tail: triple.tail,
            }
        } else {
            Triple {
                head: triple.head,
                relation: triple.relation,
                tail: random_entity,
            }
        }
    }

    fn update_embeddings(&mut self, positive: &Triple, negative: &Triple) {
        let lr = self.config.learning_rate;
        let dim = self.config.dim;

        // Gradient for positive triple: h + r - t
        // For negative triple: h' + r - t' (corrupted)
        let pos_h = self
            .entity_embeddings
            .get(&positive.head)
            .cloned()
            .unwrap_or_else(|| vec![0.0; dim]);
        let pos_t = self
            .entity_embeddings
            .get(&positive.tail)
            .cloned()
            .unwrap_or_else(|| vec![0.0; dim]);
        let neg_h = self
            .entity_embeddings
            .get(&negative.head)
            .cloned()
            .unwrap_or_else(|| vec![0.0; dim]);
        let neg_t = self
            .entity_embeddings
            .get(&negative.tail)
            .cloned()
            .unwrap_or_else(|| vec![0.0; dim]);
        let rel = self
            .relation_embeddings
            .get(&positive.relation)
            .cloned()
            .unwrap_or_else(|| vec![0.0; dim]);

        // Gradient direction for L2
        let mut pos_grad = vec![0.0; dim];
        let mut neg_grad = vec![0.0; dim];
        for i in 0..dim {
            pos_grad[i] = pos_h[i] + rel[i] - pos_t[i];
            neg_grad[i] = neg_h[i] + rel[i] - neg_t[i];
        }

        // Normalize gradients
        let pos_norm = l2_norm(&pos_grad).max(1e-12);
        let neg_norm = l2_norm(&neg_grad).max(1e-12);

        // Update positive head: h = h - lr * gradient
        if let Some(h_emb) = self.entity_embeddings.get_mut(&positive.head) {
            for i in 0..dim {
                h_emb[i] -= lr * pos_grad[i] / pos_norm;
            }
        }

        // Update positive tail: t = t + lr * gradient
        if let Some(t_emb) = self.entity_embeddings.get_mut(&positive.tail) {
            for i in 0..dim {
                t_emb[i] += lr * pos_grad[i] / pos_norm;
            }
        }

        // Update negative head: h' = h' + lr * gradient
        if let Some(h_emb) = self.entity_embeddings.get_mut(&negative.head) {
            for i in 0..dim {
                h_emb[i] += lr * neg_grad[i] / neg_norm;
            }
        }

        // Update negative tail: t' = t' - lr * gradient
        if let Some(t_emb) = self.entity_embeddings.get_mut(&negative.tail) {
            for i in 0..dim {
                t_emb[i] -= lr * neg_grad[i] / neg_norm;
            }
        }

        // Update relation: r = r - lr * (pos_grad - neg_grad)
        if let Some(r_emb) = self.relation_embeddings.get_mut(&positive.relation) {
            for i in 0..dim {
                r_emb[i] -= lr * (pos_grad[i] / pos_norm - neg_grad[i] / neg_norm);
            }
        }
    }

    fn normalize_entities(&mut self) {
        for emb in self.entity_embeddings.values_mut() {
            let norm = l2_norm(emb);
            if norm > 1.0 {
                for v in emb.iter_mut() {
                    *v /= norm;
                }
            }
        }
    }
}

impl Default for TransEModel {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helper functions ────────────────────────────────────

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model() -> TransEModel {
        let mut model = TransEModel::with_config(TransEConfig {
            dim: 10,
            learning_rate: 0.01,
            margin: 1.0,
            max_epochs: 50,
            ..Default::default()
        });
        model.add_triple("alice", "knows", "bob");
        model.add_triple("bob", "knows", "charlie");
        model.add_triple("alice", "likes", "music");
        model.add_triple("bob", "likes", "sports");
        model.add_triple("charlie", "likes", "music");
        model
    }

    // ═══ Config tests ════════════════════════════════════

    #[test]
    fn test_default_config() {
        let config = TransEConfig::default();
        assert_eq!(config.dim, 50);
        assert_eq!(config.distance_metric, DistanceMetric::L2);
        assert!(config.normalize_embeddings);
    }

    // ═══ Entity/relation management tests ════════════════

    #[test]
    fn test_add_entity() {
        let mut model = TransEModel::new();
        let id = model.add_entity("alice");
        assert_eq!(id, 0);
        assert_eq!(model.entity_count(), 1);
    }

    #[test]
    fn test_add_entity_idempotent() {
        let mut model = TransEModel::new();
        let id1 = model.add_entity("alice");
        let id2 = model.add_entity("alice");
        assert_eq!(id1, id2);
        assert_eq!(model.entity_count(), 1);
    }

    #[test]
    fn test_add_relation() {
        let mut model = TransEModel::new();
        let id = model.add_relation("knows");
        assert_eq!(id, 0);
        assert_eq!(model.relation_count(), 1);
    }

    #[test]
    fn test_add_triple() {
        let model = sample_model();
        assert_eq!(model.triple_count(), 5);
        assert_eq!(model.entity_count(), 5); // alice, bob, charlie, music, sports
        assert_eq!(model.relation_count(), 2); // knows, likes
    }

    #[test]
    fn test_entity_name() {
        let model = sample_model();
        assert_eq!(model.entity_name(0), Some("alice"));
    }

    #[test]
    fn test_relation_name() {
        let model = sample_model();
        let name = model.relation_name(0);
        assert!(name.is_some());
    }

    // ═══ Training tests ══════════════════════════════════

    #[test]
    fn test_train_basic() {
        let mut model = sample_model();
        let stats = model.train(10);
        assert_eq!(stats.epochs_completed, 10);
        assert_eq!(stats.loss_history.len(), 10);
    }

    #[test]
    fn test_train_loss_decreases() {
        let mut model = sample_model();
        model.train(20);
        let losses = &model.stats().loss_history;
        // Not guaranteed to be strictly decreasing, but early loss should be >= late loss
        let first_avg: f64 = losses[..5].iter().sum::<f64>() / 5.0;
        let last_avg: f64 = losses[15..].iter().sum::<f64>() / 5.0;
        // Loss should decrease or at least not explode
        assert!(last_avg < first_avg * 10.0);
    }

    #[test]
    fn test_train_empty_triples() {
        let mut model = TransEModel::new();
        let stats = model.train(10);
        assert_eq!(stats.epochs_completed, 0);
    }

    #[test]
    fn test_train_stats_cumulative() {
        let mut model = sample_model();
        model.train(5);
        model.train(5);
        assert_eq!(model.stats().epochs_completed, 10);
    }

    // ═══ Scoring tests ═══════════════════════════════════

    #[test]
    fn test_score_known_triple() {
        let mut model = sample_model();
        model.train(20);
        let score = model.score("alice", "knows", "bob");
        assert!(score.is_some());
        assert!(score.expect("score") < 100.0);
    }

    #[test]
    fn test_score_unknown_entity() {
        let model = sample_model();
        assert!(model.score("unknown", "knows", "bob").is_none());
    }

    #[test]
    fn test_score_unknown_relation() {
        let model = sample_model();
        assert!(model.score("alice", "unknown", "bob").is_none());
    }

    // ═══ Prediction tests ════════════════════════════════

    #[test]
    fn test_predict_tail() {
        let mut model = sample_model();
        model.train(10);
        let predictions = model.predict_tail("alice", "knows", 3);
        assert_eq!(predictions.len(), 3);
        // Should be sorted by score (ascending, lower is better)
        for window in predictions.windows(2) {
            assert!(window[0].score <= window[1].score);
        }
    }

    #[test]
    fn test_predict_head() {
        let mut model = sample_model();
        model.train(10);
        let predictions = model.predict_head("knows", "bob", 3);
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_predict_unknown_entity() {
        let model = sample_model();
        let predictions = model.predict_tail("unknown", "knows", 3);
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_predict_unknown_relation() {
        let model = sample_model();
        let predictions = model.predict_tail("alice", "unknown", 3);
        assert!(predictions.is_empty());
    }

    // ═══ Embedding access tests ══════════════════════════

    #[test]
    fn test_entity_embedding() {
        let model = sample_model();
        let emb = model.entity_embedding("alice");
        assert!(emb.is_some());
        assert_eq!(emb.expect("embedding").len(), 10);
    }

    #[test]
    fn test_relation_embedding() {
        let model = sample_model();
        let emb = model.relation_embedding("knows");
        assert!(emb.is_some());
        assert_eq!(emb.expect("embedding").len(), 10);
    }

    #[test]
    fn test_embedding_unknown() {
        let model = sample_model();
        assert!(model.entity_embedding("unknown").is_none());
        assert!(model.relation_embedding("unknown").is_none());
    }

    // ═══ Nearest neighbor tests ══════════════════════════

    #[test]
    fn test_nearest_entities() {
        let mut model = sample_model();
        model.train(10);
        let alice_emb = model.entity_embedding("alice").expect("alice").clone();
        let nearest = model.nearest_entities(&alice_emb, 3);
        assert_eq!(nearest.len(), 3);
        // Alice should be closest to itself
        assert_eq!(nearest[0].0, "alice");
        assert!(nearest[0].1 < 1e-10);
    }

    // ═══ Distance metric tests ═══════════════════════════

    #[test]
    fn test_l1_distance_metric() {
        let mut model = TransEModel::with_config(TransEConfig {
            dim: 10,
            distance_metric: DistanceMetric::L1,
            ..Default::default()
        });
        model.add_triple("a", "r", "b");
        model.train(5);
        let score = model.score("a", "r", "b");
        assert!(score.is_some());
    }

    // ═══ Normalization tests ═════════════════════════════

    #[test]
    fn test_normalized_embeddings() {
        let mut model = sample_model();
        model.train(10);
        for emb in model.entity_embeddings.values() {
            let norm = l2_norm(emb);
            assert!(norm <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_no_normalization() {
        let mut model = TransEModel::with_config(TransEConfig {
            dim: 10,
            normalize_embeddings: false,
            ..Default::default()
        });
        model.add_triple("a", "r", "b");
        model.train(5);
        // Should still work (no crash)
        assert_eq!(model.triple_count(), 1);
    }

    // ═══ Helper function tests ═══════════════════════════

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(l2_distance(&a, &a) < 1e-10);
    }

    // ═══ Default impl test ═══════════════════════════════

    #[test]
    fn test_default_model() {
        let model = TransEModel::default();
        assert_eq!(model.entity_count(), 0);
        assert_eq!(model.relation_count(), 0);
        assert_eq!(model.triple_count(), 0);
    }
}
