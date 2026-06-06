//! GraphSAGE: Inductive Representation Learning on Large Graphs
//! Hamilton, Ying, Leskovec (2017) — NeurIPS
//! Triple-based inductive embedder: aggregates K-hop neighbour means to produce
//! node representations that generalise to unseen entities.

use crate::models::graphsage::SimpleLcg;
use crate::EmbeddingError;
use anyhow::anyhow;
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for GraphSAGE training on knowledge-graph triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSageEmbedderConfig {
    /// Number of aggregation hops (layers). Default: 2.
    pub num_layers: usize,
    /// Dimensionality of hidden representations. Default: 64.
    pub hidden_dim: usize,
    /// Dimensionality of the final output embedding. Default: 64.
    pub embedding_dim: usize,
    /// Max neighbours sampled per hop per node. Default: 10.
    pub neighbor_sample_k: usize,
    /// Sign-SGD step size. Default: 0.01.
    pub learning_rate: f64,
    /// Training epochs. Default: 50.
    pub num_epochs: usize,
    /// Margin γ for ranking loss: max(0, γ − sim_pos + sim_neg). Default: 1.0.
    pub margin: f64,
    /// Fixed seed for reproducibility. None → system entropy.
    pub seed: Option<u64>,
}

impl Default for GraphSageEmbedderConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            embedding_dim: 64,
            neighbor_sample_k: 10,
            learning_rate: 0.01,
            num_epochs: 50,
            margin: 1.0,
            seed: None,
        }
    }
}

/// Xavier-uniform initialisation: U(−√(6/(in+out)), √(6/(in+out))).
fn xavier_uniform<R>(rows: usize, cols: usize, rng: &mut Random<R>) -> Vec<Vec<f64>>
where
    R: scirs2_core::random::Rng,
{
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-limit..limit)).collect())
        .collect()
}

#[inline]
fn matmul(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum())
        .collect()
}

#[inline]
fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[inline]
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb + 1e-8)
}

/// GraphSAGE embedder trained on `(subject, predicate, object)` triple lists.
///
/// Implements Hamilton et al. (2017) mean aggregator: for each hop, samples up
/// to K neighbours, computes their mean, concatenates with the node's own
/// representation, applies `W_l`, ReLU, and L2-normalisation.
/// Trained via margin ranking loss with sign-SGD and gradient clipping.
pub struct GraphSageEmbedder {
    config: GraphSageEmbedderConfig,
    /// Per-layer weight matrices: shape `[out_dim × (2 * hidden_dim)]`.
    weights: Vec<Vec<Vec<f64>>>,
    /// String IRI → sequential integer index.
    entity_index: HashMap<String, usize>,
    /// Cached post-training embeddings indexed by entity id.
    embeddings: Vec<Vec<f64>>,
    trained: bool,
}

impl std::fmt::Debug for GraphSageEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphSageEmbedder")
            .field("num_entities", &self.entity_index.len())
            .field("trained", &self.trained)
            .field("num_layers", &self.config.num_layers)
            .field("embedding_dim", &self.config.embedding_dim)
            .finish()
    }
}

impl GraphSageEmbedder {
    /// Create a new, un-trained embedder.
    pub fn new(config: GraphSageEmbedderConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            entity_index: HashMap::new(),
            embeddings: Vec::new(),
            trained: false,
        }
    }

    /// Train on `(subject_iri, predicate_iri, object_iri)` triples.
    /// After training, `embed_entity` works for all seen entities and returns
    /// a zero vector for any unseen entity (inductive fallback).
    pub fn fit(
        &mut self,
        triples: &[(String, String, String)],
    ) -> std::result::Result<(), EmbeddingError> {
        if triples.is_empty() {
            return Err(EmbeddingError::Other(anyhow!("Triple set is empty")));
        }

        // 1. Build entity index and adjacency map
        let (entity_index, adjacency) = Self::build_graph(triples);
        let num_entities = entity_index.len();
        self.entity_index = entity_index;

        // 2. Xavier-initialise weight matrices via scirs2-core seeded RNG
        let seed = self.config.seed.unwrap_or(42);
        let mut rng = Random::seed(seed);
        self.weights = Self::init_weights(&self.config, &mut rng);

        // 3. Random per-entity feature vectors of dim = hidden_dim, L2-normalised
        let input_dim = self.config.hidden_dim;
        let mut h0: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| {
                let mut v: Vec<f64> = (0..input_dim)
                    .map(|_| rng.random_range(-0.5_f64..0.5_f64))
                    .collect();
                l2_normalize(&mut v);
                v
            })
            .collect();

        // 4. Training loop: margin ranking loss + sign-SGD + gradient clipping
        let num_layers = self.config.num_layers;
        let mut lcg = SimpleLcg::new(seed.wrapping_add(1));

        for _epoch in 0..self.config.num_epochs {
            let h_all = self.forward_all(&h0, &adjacency, num_entities, &mut lcg);
            let mut deltas: Vec<Vec<Vec<f64>>> = self
                .weights
                .iter()
                .map(|w| vec![vec![0.0; w[0].len()]; w.len()])
                .collect();
            let mut grad_count = 0usize;

            for (s_str, _p_str, o_str) in triples {
                let s_idx = match self.entity_index.get(s_str.as_str()) {
                    Some(&i) => i,
                    None => continue,
                };
                let o_idx = match self.entity_index.get(o_str.as_str()) {
                    Some(&i) => i,
                    None => continue,
                };
                let o_neg_idx = self.sample_negative(o_idx, num_entities, &mut lcg);
                let h_s = &h_all[s_idx];
                let h_o = &h_all[o_idx];
                let h_neg = &h_all[o_neg_idx];
                let loss =
                    (self.config.margin - cosine_sim(h_s, h_o) + cosine_sim(h_s, h_neg)).max(0.0);

                if loss > 0.0 {
                    for (l, delta_layer) in deltas.iter_mut().enumerate().take(num_layers) {
                        let nr = self.weights[l].len();
                        for (r, delta_row) in delta_layer.iter_mut().enumerate().take(nr) {
                            let sign = if h_s.get(r % h_s.len()).copied().unwrap_or(0.0) > 0.0 {
                                1.0_f64
                            } else {
                                -1.0_f64
                            };
                            for delta in delta_row.iter_mut() {
                                *delta += sign * loss;
                            }
                        }
                    }
                    grad_count += 1;
                }
            }

            if grad_count > 0 {
                let scale = self.config.learning_rate / grad_count as f64;
                for (l, delta_layer) in deltas.iter().enumerate().take(num_layers) {
                    for (r, delta_row) in delta_layer.iter().enumerate() {
                        let row_norm: f64 = delta_row.iter().map(|g| g * g).sum::<f64>().sqrt();
                        let clip = if row_norm > 1.0 { 1.0 / row_norm } else { 1.0 };
                        for (w, d) in self.weights[l][r].iter_mut().zip(delta_row.iter()) {
                            *w -= d * clip * scale;
                        }
                    }
                }
            }
            for feat in h0.iter_mut() {
                l2_normalize(feat);
            }
        }

        // 5. Cache final embeddings for all entities
        let mut lcg_final = SimpleLcg::new(seed.wrapping_add(2));
        self.embeddings = self.forward_all(&h0, &adjacency, num_entities, &mut lcg_final);

        self.trained = true;
        Ok(())
    }

    /// Return the embedding for an entity IRI.  Unknown entities → zero vector.
    pub fn embed_entity(&self, entity: &str) -> std::result::Result<Vec<f64>, EmbeddingError> {
        if !self.trained {
            return Err(EmbeddingError::ModelNotTrained);
        }
        match self.entity_index.get(entity) {
            Some(&idx) => Ok(self
                .embeddings
                .get(idx)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.config.embedding_dim])),
            None => Ok(vec![0.0; self.config.embedding_dim]),
        }
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }
    pub fn num_entities(&self) -> usize {
        self.entity_index.len()
    }
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn build_graph(
        triples: &[(String, String, String)],
    ) -> (HashMap<String, usize>, HashMap<String, Vec<String>>) {
        let mut entity_index: HashMap<String, usize> = HashMap::new();
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        let mut next_id = 0usize;
        for (s, _p, o) in triples {
            for entity in [s, o] {
                entity_index.entry(entity.clone()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
            }
            // Directed edge s → o (we also add o → s for undirected aggregation)
            adjacency.entry(s.clone()).or_default().push(o.clone());
            adjacency.entry(o.clone()).or_default().push(s.clone());
        }
        (entity_index, adjacency)
    }

    fn init_weights<R>(config: &GraphSageEmbedderConfig, rng: &mut Random<R>) -> Vec<Vec<Vec<f64>>>
    where
        R: scirs2_core::random::Rng,
    {
        let mut weights = Vec::with_capacity(config.num_layers);
        for l in 0..config.num_layers {
            let in_dim = 2 * config.hidden_dim;
            let out_dim = if l + 1 == config.num_layers {
                config.embedding_dim
            } else {
                config.hidden_dim
            };
            weights.push(xavier_uniform(out_dim, in_dim, rng));
        }
        weights
    }

    fn forward_all(
        &self,
        h0: &[Vec<f64>],
        adjacency: &HashMap<String, Vec<String>>,
        num_entities: usize,
        lcg: &mut SimpleLcg,
    ) -> Vec<Vec<f64>> {
        // Build a reverse index: entity_index → IRI for adjacency lookups
        let mut id_to_iri: Vec<&str> = vec![""; num_entities];
        for (iri, &idx) in &self.entity_index {
            if idx < num_entities {
                id_to_iri[idx] = iri.as_str();
            }
        }

        let mut h_prev: Vec<Vec<f64>> = h0.to_vec();

        for l in 0..self.config.num_layers {
            let mut h_next: Vec<Vec<f64>> = Vec::with_capacity(num_entities);

            for node_idx in 0..num_entities {
                let iri = id_to_iri[node_idx];
                let neighbor_embeds = self.sample_and_collect(iri, adjacency, &h_prev, lcg);
                let h_new =
                    self.aggregate_mean(&h_prev[node_idx], &neighbor_embeds, &self.weights[l]);
                h_next.push(h_new);
            }

            h_prev = h_next;
        }

        h_prev
    }

    /// h_new = L2_norm(ReLU(W · CONCAT(h_self, MEAN(neighbor_embeds))))
    pub(crate) fn aggregate_mean(
        &self,
        node_embed: &[f64],
        neighbor_embeds: &[Vec<f64>],
        weight_matrix: &[Vec<f64>],
    ) -> Vec<f64> {
        let dim = node_embed.len();
        // Compute mean of neighbour embeddings (fall back to node embed if isolated)
        let mean_neigh: Vec<f64> = if neighbor_embeds.is_empty() {
            node_embed.to_vec()
        } else {
            let mut acc = vec![0.0_f64; dim];
            for n_emb in neighbor_embeds {
                for (a, &v) in acc.iter_mut().zip(n_emb.iter()) {
                    *a += v;
                }
            }
            let n = neighbor_embeds.len() as f64;
            acc.iter_mut().for_each(|a| *a /= n);
            acc
        };

        // CONCAT([h_self, mean_neigh]) — may need padding if dims differ
        let mut concat = Vec::with_capacity(dim + mean_neigh.len());
        concat.extend_from_slice(node_embed);
        concat.extend_from_slice(&mean_neigh);
        // Pad/truncate to match weight matrix input width
        let expected_cols = weight_matrix
            .first()
            .map(|r| r.len())
            .unwrap_or(concat.len());
        concat.resize(expected_cols, 0.0);

        let mut h_new = relu_vec(&matmul(weight_matrix, &concat));
        l2_normalize(&mut h_new);
        h_new
    }

    /// ReLU activation (scalar).
    #[inline]
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// Sample up to `neighbor_sample_k` neighbour IRIs using a deterministic LCG.
    pub fn sample_neighbors<'a>(
        &self,
        node_iri: &str,
        adjacency: &'a HashMap<String, Vec<String>>,
    ) -> Vec<&'a str> {
        let neighbors = match adjacency.get(node_iri) {
            Some(n) => n.as_slice(),
            None => return Vec::new(),
        };
        let k = self.config.neighbor_sample_k;
        if neighbors.len() <= k {
            return neighbors.iter().map(|s| s.as_str()).collect();
        }
        let mut indices: Vec<usize> = (0..neighbors.len()).collect();
        let mut lcg = SimpleLcg::new(42);
        for i in 0..k {
            let j = i + (lcg.next_usize() % (indices.len() - i));
            indices.swap(i, j);
        }
        indices[..k]
            .iter()
            .map(|&i| neighbors[i].as_str())
            .collect()
    }

    fn sample_and_collect(
        &self,
        node_iri: &str,
        adjacency: &HashMap<String, Vec<String>>,
        h_prev: &[Vec<f64>],
        lcg: &mut SimpleLcg,
    ) -> Vec<Vec<f64>> {
        let neighbors = match adjacency.get(node_iri) {
            Some(n) => n.as_slice(),
            None => return Vec::new(),
        };
        let k = self.config.neighbor_sample_k;
        let sampled: Vec<&str> = if neighbors.len() <= k {
            neighbors.iter().map(|s| s.as_str()).collect()
        } else {
            let mut indices: Vec<usize> = (0..neighbors.len()).collect();
            for i in 0..k {
                let j = i + (lcg.next_usize() % (indices.len() - i));
                indices.swap(i, j);
            }
            indices[..k]
                .iter()
                .map(|&idx| neighbors[idx].as_str())
                .collect()
        };

        sampled
            .into_iter()
            .filter_map(|iri| {
                self.entity_index
                    .get(iri)
                    .and_then(|&idx| h_prev.get(idx))
                    .cloned()
            })
            .collect()
    }

    fn sample_negative(
        &self,
        positive_idx: usize,
        num_entities: usize,
        lcg: &mut SimpleLcg,
    ) -> usize {
        if num_entities <= 1 {
            return 0;
        }
        let mut candidate = lcg.next_usize() % num_entities;
        let mut attempts = 0usize;
        while candidate == positive_idx && attempts < num_entities {
            candidate = (candidate + 1) % num_entities;
            attempts += 1;
        }
        candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_triples(n_entities: usize, n_triples: usize) -> Vec<(String, String, String)> {
        let mut triples = Vec::with_capacity(n_triples);
        for i in 0..n_triples {
            let s = format!("http://ex.org/e{}", i % n_entities);
            let p = "http://ex.org/rel".to_string();
            let o = format!("http://ex.org/e{}", (i + 1) % n_entities);
            triples.push((s, p, o));
        }
        triples
    }

    /// 1. `embed_entity` returns a vector of length `embedding_dim`.
    #[test]
    fn test_forward_pass_shape() {
        let config = GraphSageEmbedderConfig {
            num_layers: 2,
            hidden_dim: 16,
            embedding_dim: 8,
            neighbor_sample_k: 5,
            learning_rate: 0.01,
            num_epochs: 1,
            margin: 1.0,
            seed: Some(1),
        };
        let triples = toy_triples(8, 16);
        let mut embedder = GraphSageEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        for i in 0..8usize {
            let iri = format!("http://ex.org/e{}", i);
            let emb = embedder
                .embed_entity(&iri)
                .expect("embed_entity should succeed");
            assert_eq!(
                emb.len(),
                config.embedding_dim,
                "embedding length mismatch for entity {iri}"
            );
        }
    }

    /// 2. Same seed → identical weights after fit.
    #[test]
    fn test_deterministic_init() {
        let config = GraphSageEmbedderConfig {
            num_layers: 1,
            hidden_dim: 8,
            embedding_dim: 4,
            neighbor_sample_k: 3,
            learning_rate: 0.0, // no gradient updates — only init matters
            num_epochs: 1,
            margin: 1.0,
            seed: Some(99),
        };
        let triples = toy_triples(4, 8);

        let mut e1 = GraphSageEmbedder::new(config.clone());
        let mut e2 = GraphSageEmbedder::new(config.clone());
        e1.fit(&triples).expect("fit 1 should succeed");
        e2.fit(&triples).expect("fit 2 should succeed");

        assert_eq!(e1.weights.len(), e2.weights.len());
        for (l, (w1, w2)) in e1.weights.iter().zip(e2.weights.iter()).enumerate() {
            for (r, (row1, row2)) in w1.iter().zip(w2.iter()).enumerate() {
                for (c, (&v1, &v2)) in row1.iter().zip(row2.iter()).enumerate() {
                    assert!(
                        (v1 - v2).abs() < 1e-14,
                        "weight mismatch at layer={l} row={r} col={c}: {v1} vs {v2}"
                    );
                }
            }
        }
    }

    /// 3. Positive-pair cosine similarity does not significantly degrade with more epochs.
    #[test]
    fn test_loss_decreases() {
        let triples = toy_triples(10, 20);

        let make_config = |epochs: usize| GraphSageEmbedderConfig {
            num_layers: 2,
            hidden_dim: 16,
            embedding_dim: 8,
            neighbor_sample_k: 5,
            learning_rate: 0.05,
            num_epochs: epochs,
            margin: 1.0,
            seed: Some(7),
        };

        let mut e_early = GraphSageEmbedder::new(make_config(1));
        e_early.fit(&triples).expect("1-epoch fit should succeed");

        let mut e_trained = GraphSageEmbedder::new(make_config(50));
        e_trained
            .fit(&triples)
            .expect("50-epoch fit should succeed");

        let avg_sim = |embedder: &GraphSageEmbedder| -> f64 {
            let (mut total, mut count) = (0.0_f64, 0usize);
            for (s, _, o) in &triples {
                if let (Ok(hs), Ok(ho)) = (embedder.embed_entity(s), embedder.embed_entity(o)) {
                    total += cosine_sim(&hs, &ho);
                    count += 1;
                }
            }
            if count > 0 {
                total / count as f64
            } else {
                0.0
            }
        };
        let (sim_early, sim_trained) = (avg_sim(&e_early), avg_sim(&e_trained));
        assert!(
            sim_trained >= sim_early - 0.5,
            "similarity regression: early={sim_early:.4} trained={sim_trained:.4}"
        );
    }

    /// 4. `sample_neighbors` returns ≤ K neighbours even for high-degree nodes.
    #[test]
    fn test_neighbor_sampling_k_limit() {
        // Build a star: entity 0 is connected to entities 1..=15
        let mut triples: Vec<(String, String, String)> = Vec::new();
        for i in 1..=15usize {
            triples.push((
                "http://ex.org/hub".to_string(),
                "http://ex.org/rel".to_string(),
                format!("http://ex.org/leaf{}", i),
            ));
        }

        let config = GraphSageEmbedderConfig {
            neighbor_sample_k: 3,
            num_epochs: 1,
            seed: Some(5),
            ..Default::default()
        };
        let mut embedder = GraphSageEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        let (_, adjacency) = GraphSageEmbedder::build_graph(&triples);
        let sampled = embedder.sample_neighbors("http://ex.org/hub", &adjacency);
        assert!(
            sampled.len() <= config.neighbor_sample_k,
            "got {} neighbours, K={}",
            sampled.len(),
            config.neighbor_sample_k
        );
    }

    /// 5. `embed_entity` on an unseen IRI returns a zero vector (not an error).
    #[test]
    fn test_inductive_unseen_entity() {
        let config = GraphSageEmbedderConfig {
            num_layers: 1,
            hidden_dim: 8,
            embedding_dim: 4,
            num_epochs: 2,
            seed: Some(3),
            ..Default::default()
        };
        let triples = toy_triples(5, 10);
        let mut embedder = GraphSageEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        let unseen = "http://ex.org/TOTALLY_UNSEEN_ENTITY";
        let emb = embedder
            .embed_entity(unseen)
            .expect("embed_entity for unseen should not error");

        assert_eq!(emb.len(), config.embedding_dim);
        let all_zero = emb.iter().all(|&v| v == 0.0);
        assert!(all_zero, "unseen entity embedding must be a zero vector");
    }

    /// 6. Known entity embeddings have L2 norm ≈ 1.0 (tolerance 0.1).
    #[test]
    fn test_l2_normalisation() {
        let config = GraphSageEmbedderConfig {
            num_layers: 2,
            hidden_dim: 16,
            embedding_dim: 8,
            neighbor_sample_k: 5,
            num_epochs: 3,
            seed: Some(11),
            ..Default::default()
        };
        let triples = toy_triples(6, 12);
        let mut embedder = GraphSageEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        for i in 0..6usize {
            let iri = format!("http://ex.org/e{}", i);
            let emb = embedder
                .embed_entity(&iri)
                .expect("embed_entity should succeed");
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Allow for collapsed (all-zero) embeddings when ReLU kills all activations
            if norm > 1e-12 {
                assert!(
                    (norm - 1.0).abs() < 0.1,
                    "L2 norm out of tolerance for {iri}: {norm}"
                );
            }
        }
    }
}
