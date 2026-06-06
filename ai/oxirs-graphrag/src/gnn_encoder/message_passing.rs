//! Message-passing GNN encoder for knowledge-graph entity embeddings.
//!
//! Implements a simple mean-aggregation multi-layer GNN with Xavier weight
//! initialisation, ReLU + L2-normalisation activations, and a margin-based
//! link-prediction training objective.  All random numbers are generated
//! via an internal Linear Congruential Generator so the crate does not depend
//! on the `rand` crate.

use std::collections::HashMap;

use crate::GraphRAGError;

use super::adjacency::AdjacencyGraph;

// ─────────────────────────────────────────────────────────────────────────────
// Linear Congruential Generator (internal; no rand dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple Linear Congruential Generator — Numerical Recipes parameters.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform sample in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform sample in [-scale, scale)
    fn next_f64_range(&mut self, scale: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * scale
    }

    /// Random usize in [0, n)
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the message-passing GNN encoder.
#[derive(Debug, Clone)]
pub struct GnnEncoderConfig {
    /// Number of message-passing layers (default: 2)
    pub num_layers: usize,
    /// Dimensionality of hidden and output embeddings (default: 64)
    pub hidden_dim: usize,
    /// Training epochs for the link-prediction objective (default: 50)
    pub num_epochs: usize,
    /// SGD learning rate (default: 0.01)
    pub learning_rate: f64,
    /// Margin for the triplet margin loss (default: 1.0)
    pub margin: f64,
}

impl Default for GnnEncoderConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            num_epochs: 50,
            learning_rate: 0.01,
            margin: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GnnEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-layer mean-aggregation GNN encoder.
///
/// After calling [`GnnEncoder::fit`] on a set of RDF triples the encoder can
/// produce a fixed-dimensional embedding for any entity seen during training
/// via [`GnnEncoder::embed_entity`].  Unknown entities return a zero vector.
pub struct GnnEncoder {
    /// Encoder configuration
    config: GnnEncoderConfig,
    /// Entity embeddings: `entity_embeddings[i]` is the embedding for node `i`
    entity_embeddings: Vec<Vec<f64>>,
    /// Weight matrices for each layer: `weight_matrices[l][i][j]`
    weight_matrices: Vec<Vec<Vec<f64>>>,
    /// Map from entity name → integer index
    entity_index: HashMap<String, usize>,
}

impl GnnEncoder {
    /// Create a new, untrained encoder.
    pub fn new(config: GnnEncoderConfig) -> Self {
        Self {
            config,
            entity_embeddings: Vec::new(),
            weight_matrices: Vec::new(),
            entity_index: HashMap::new(),
        }
    }

    /// Fit the encoder to the provided RDF triples.
    ///
    /// Builds the adjacency graph, initialises embeddings and weight matrices,
    /// then runs stochastic gradient descent with a margin-based
    /// link-prediction loss.
    pub fn fit(&mut self, triples: &[(String, String, String)]) -> Result<(), GraphRAGError> {
        if triples.is_empty() {
            return Err(GraphRAGError::EmbeddingError(
                "Cannot fit GnnEncoder on empty triple set".into(),
            ));
        }

        let graph = AdjacencyGraph::from_triples(triples);
        let n = graph.entity_count();
        let d = self.config.hidden_dim;

        // Store entity index map
        self.entity_index = graph.entity_to_idx.clone();

        let mut rng = Lcg::new(42);

        // Initialise entity embeddings with Xavier uniform
        self.entity_embeddings = Self::xavier_init(n, d, &mut rng);

        // Initialise one weight matrix per layer (d × d)
        self.weight_matrices = (0..self.config.num_layers)
            .map(|_| Self::xavier_init(d, d, &mut rng))
            .collect();

        // Training loop
        for _epoch in 0..self.config.num_epochs {
            // For each triple, run a positive + negative sample update
            for (s_str, _p_str, o_str) in triples {
                let Some(&s_idx) = self.entity_index.get(s_str.as_str()) else {
                    continue;
                };
                let Some(&o_idx) = self.entity_index.get(o_str.as_str()) else {
                    continue;
                };

                // Sample a random negative entity
                let neg_idx = loop {
                    let candidate = rng.next_usize(n);
                    if candidate != o_idx {
                        break candidate;
                    }
                };

                // Forward pass: compute embeddings for s, o, neg
                let emb_s = self.forward_entity(s_idx, &graph);
                let emb_o = self.forward_entity(o_idx, &graph);
                let emb_neg = self.forward_entity(neg_idx, &graph);

                let loss = Self::margin_loss(&emb_s, &emb_o, &emb_neg, self.config.margin);

                // Only update if loss is positive (violated margin)
                if loss > 0.0 {
                    self.sgd_update(s_idx, o_idx, neg_idx, &graph);
                }
            }
        }

        // Final forward pass to store steady-state embeddings
        for i in 0..n {
            self.entity_embeddings[i] = self.forward_entity(i, &graph);
        }

        Ok(())
    }

    /// Return the embedding vector for a named entity.
    /// Returns a zero vector of length `hidden_dim` if the entity is unknown.
    pub fn embed_entity(&self, entity: &str) -> Vec<f64> {
        match self.entity_index.get(entity) {
            Some(&idx) if idx < self.entity_embeddings.len() => self.entity_embeddings[idx].clone(),
            _ => vec![0.0; self.config.hidden_dim],
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Xavier/Glorot uniform initialisation for an (rows × cols) weight matrix.
    fn xavier_init(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<Vec<f64>> {
        let scale = (6.0 / (rows + cols) as f64).sqrt();
        (0..rows)
            .map(|_| (0..cols).map(|_| rng.next_f64_range(scale)).collect())
            .collect()
    }

    /// Run a single forward pass for node `idx`, producing a `hidden_dim`-dimensional
    /// embedding by iterating over each message-passing layer.
    fn forward_entity(&self, idx: usize, graph: &AdjacencyGraph) -> Vec<f64> {
        let d = self.config.hidden_dim;
        let mut h = if idx < self.entity_embeddings.len() {
            self.entity_embeddings[idx].clone()
        } else {
            vec![0.0; d]
        };

        for layer in 0..self.config.num_layers {
            // Collect neighbour embeddings for mean aggregation
            let neighbors = graph.neighbors(idx);
            let neighbor_embs: Vec<&Vec<f64>> = neighbors
                .iter()
                .filter_map(|&nidx| self.entity_embeddings.get(nidx))
                .collect();

            let aggregated = if neighbor_embs.is_empty() {
                h.clone()
            } else {
                // Mean of self + neighbours
                let mut combined = neighbor_embs.clone();
                combined.push(&h);
                Self::mean_aggregate(&combined)
            };

            // Apply weight matrix for this layer: h_new = W * aggregated
            let w = &self.weight_matrices[layer];
            let mut new_h = vec![0.0; d];
            for (i, row) in w.iter().enumerate() {
                let dot: f64 = row.iter().zip(aggregated.iter()).map(|(a, b)| a * b).sum();
                new_h[i] = dot;
            }

            Self::relu_and_normalize(&mut new_h);
            h = new_h;
        }

        h
    }

    /// One SGD step pushing the positive pair (s, o) closer and the negative
    /// pair (s, neg) further apart in embedding space.
    fn sgd_update(&mut self, s_idx: usize, o_idx: usize, neg_idx: usize, graph: &AdjacencyGraph) {
        let lr = self.config.learning_rate;
        let d = self.config.hidden_dim;

        let emb_s = self.forward_entity(s_idx, graph);
        let emb_o = self.forward_entity(o_idx, graph);
        let emb_neg = self.forward_entity(neg_idx, graph);

        // Gradient for entity embeddings:
        //   push s closer to o: emb_s -= lr * (emb_s - emb_o)
        //   push s away from neg: emb_s += lr * (emb_s - emb_neg)
        for j in 0..d {
            if s_idx < self.entity_embeddings.len() {
                let grad_pos = emb_s[j] - emb_o[j];
                let grad_neg = emb_s[j] - emb_neg[j];
                self.entity_embeddings[s_idx][j] -= lr * (grad_pos - grad_neg);
            }
        }

        // Normalise updated embedding
        if s_idx < self.entity_embeddings.len() {
            let v = &mut self.entity_embeddings[s_idx];
            Self::relu_and_normalize(v);
        }
    }

    /// Compute the mean embedding of a non-empty slice of embedding vectors.
    pub fn mean_aggregate(embeddings: &[&Vec<f64>]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }
        let d = embeddings[0].len();
        let mut mean = vec![0.0_f64; d];
        for emb in embeddings {
            for (j, &val) in emb.iter().enumerate() {
                if j < mean.len() {
                    mean[j] += val;
                }
            }
        }
        let n = embeddings.len() as f64;
        for v in &mut mean {
            *v /= n;
        }
        mean
    }

    /// Apply ReLU activation then L2-normalise the vector in-place.
    /// If the L2 norm is near zero the vector is left unchanged.
    pub fn relu_and_normalize(v: &mut [f64]) {
        // ReLU
        for x in v.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        // L2 normalise
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Triplet margin loss: max(0, d(s, o) - d(s, neg) + margin)
    /// where d is squared Euclidean distance.
    pub fn margin_loss(pos_s: &[f64], pos_o: &[f64], neg_o: &[f64], margin: f64) -> f64 {
        let d_pos: f64 = pos_s
            .iter()
            .zip(pos_o.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let d_neg: f64 = pos_s
            .iter()
            .zip(neg_o.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (d_pos - d_neg + margin).max(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triples() -> Vec<(String, String, String)> {
        vec![
            ("Alice".into(), "knows".into(), "Bob".into()),
            ("Bob".into(), "worksAt".into(), "Acme".into()),
            ("Carol".into(), "worksAt".into(), "Acme".into()),
            ("Alice".into(), "friendOf".into(), "Carol".into()),
            ("Dave".into(), "knows".into(), "Alice".into()),
        ]
    }

    #[test]
    fn test_fit_completes() {
        let mut encoder = GnnEncoder::new(GnnEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_epochs: 5,
            ..Default::default()
        });
        encoder.fit(&triples()).expect("fit should succeed");
    }

    #[test]
    fn test_embed_shape_correct() {
        let mut encoder = GnnEncoder::new(GnnEncoderConfig {
            num_layers: 2,
            hidden_dim: 32,
            num_epochs: 3,
            ..Default::default()
        });
        encoder.fit(&triples()).expect("fit should succeed");
        let emb = encoder.embed_entity("Alice");
        assert_eq!(emb.len(), 32, "Embedding dimension must match hidden_dim");
    }

    #[test]
    fn test_unseen_entity_returns_zero_vec() {
        let mut encoder = GnnEncoder::new(GnnEncoderConfig {
            num_layers: 1,
            hidden_dim: 8,
            num_epochs: 2,
            ..Default::default()
        });
        encoder.fit(&triples()).expect("fit should succeed");
        let emb = encoder.embed_entity("UnknownEntity_XYZ");
        assert_eq!(emb.len(), 8);
        assert!(
            emb.iter().all(|&x| x == 0.0),
            "Unknown entity must map to zero vector"
        );
    }

    #[test]
    fn test_loss_is_non_negative() {
        // The margin loss must always be ≥ 0
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![0.0_f64, 1.0, 0.0];
        let c = vec![0.0_f64, 0.0, 1.0];
        let loss = GnnEncoder::margin_loss(&a, &b, &c, 1.0);
        assert!(loss >= 0.0, "Margin loss must be non-negative");
    }

    #[test]
    fn test_embeddings_l2_normalized() {
        let mut encoder = GnnEncoder::new(GnnEncoderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_epochs: 5,
            ..Default::default()
        });
        encoder.fit(&triples()).expect("fit should succeed");

        for entity in &["Alice", "Bob", "Acme"] {
            let emb = encoder.embed_entity(entity);
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            // After relu + l2-normalise, norm should be close to 1
            // (or zero if all activations are zero)
            assert!(
                (norm - 1.0).abs() < 1e-6 || norm < 1e-10,
                "Entity {} norm={} should be 1 or 0 (all-zero)",
                entity,
                norm
            );
        }
    }

    #[test]
    fn test_mean_aggregation_correct() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![3.0_f64, 4.0, 5.0];
        let result = GnnEncoder::mean_aggregate(&[&a, &b]);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }
}
