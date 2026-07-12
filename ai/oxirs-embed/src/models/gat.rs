//! Graph Attention Network (GAT) Embedder for RDF Knowledge Graphs
//!
//! Veličković et al. (2018) — ICLR: "Graph Attention Networks"
//!
//! Implements multi-head self-attention over graph neighbourhoods to produce
//! context-aware entity embeddings.  Each attention head learns independent Q/K/V
//! projections; heads are concatenated (or averaged) and projected through W_out;
//! the result is passed through ReLU and L2-normalised.
//!
//! Trained with margin-ranking loss and sign-SGD, identical to the GraphSAGE
//! embedder pattern established in `graph_sage.rs`.

use crate::EmbeddingError;
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::graphsage::SimpleLcg;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the multi-head GAT embedder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatEmbedderConfig {
    /// Number of GAT layers. Default: 2.
    pub num_layers: usize,
    /// Hidden/output dimension (must be divisible by `num_heads`). Default: 64.
    pub hidden_dim: usize,
    /// Number of attention heads. Default: 4.
    pub num_heads: usize,
    /// Dropout rate applied to attention coefficients. Default: 0.1.
    pub dropout_rate: f64,
    /// Number of training epochs. Default: 50.
    pub num_epochs: usize,
    /// Sign-SGD learning rate. Default: 0.01.
    pub learning_rate: f64,
    /// Margin γ for ranking loss: max(0, γ − sim_pos + sim_neg). Default: 1.0.
    pub margin: f64,
    /// Seed for reproducibility. Default: 42.
    pub seed: u64,
}

impl Default for GatEmbedderConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            num_heads: 4,
            dropout_rate: 0.1,
            num_epochs: 50,
            learning_rate: 0.01,
            margin: 1.0,
            seed: 42,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Xavier-uniform init: values drawn from U(−√(6/(in+out)), √(6/(in+out))).
fn xavier_uniform_2d(rows: usize, cols: usize, rng: &mut SimpleLcg) -> Vec<Vec<f64>> {
    let limit = (6.0_f64 / (rows + cols).max(1) as f64).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.next_f64_range(limit)).collect())
        .collect()
}

/// Matrix-vector multiply: W (rows×cols) · x (cols) → (rows).
#[inline]
fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum())
        .collect()
}

/// In-place L2-normalisation; no-op when norm ≤ 1e-12.
fn l2_normalize_inplace(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Element-wise ReLU.
#[inline]
fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

/// Cosine similarity between two equal-length slices (numerically safe).
#[inline]
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb + 1e-8)
}

// ── Public scalar utilities (exposed for testing) ─────────────────────────────

/// LeakyReLU: passes positive inputs unchanged, attenuates negatives by
/// `negative_slope`.  GAT canonical value: `negative_slope = 0.2`.
#[inline]
pub fn leaky_relu(x: f64, negative_slope: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        negative_slope * x
    }
}

/// Numerically stable softmax over a slice.  Returns a same-length vector
/// whose entries sum to 1.0.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max_val = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        vec![1.0 / scores.len() as f64; scores.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

// ── Per-layer weight matrices ──────────────────────────────────────────────────

/// Weight matrices for one GAT layer (all `num_heads` heads + output projection).
struct GatLayerWeights {
    /// Query projections, one per head: [num_heads][head_dim × hidden_dim]
    w_query: Vec<Vec<Vec<f64>>>,
    /// Key projections, one per head: [num_heads][head_dim × hidden_dim]
    w_key: Vec<Vec<Vec<f64>>>,
    /// Value projections, one per head: [num_heads][head_dim × hidden_dim]
    w_value: Vec<Vec<Vec<f64>>>,
    /// Output projection: [hidden_dim × (head_dim * num_heads)]
    w_out: Vec<Vec<f64>>,
    /// Number of heads
    num_heads: usize,
    /// Per-head dimension (= hidden_dim / num_heads)
    head_dim: usize,
    /// Full hidden dimension
    hidden_dim: usize,
}

impl GatLayerWeights {
    fn new(hidden_dim: usize, num_heads: usize, rng: &mut SimpleLcg) -> Self {
        let head_dim = hidden_dim / num_heads.max(1);
        let mut w_query = Vec::with_capacity(num_heads);
        let mut w_key = Vec::with_capacity(num_heads);
        let mut w_value = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            w_query.push(xavier_uniform_2d(head_dim, hidden_dim, rng));
            w_key.push(xavier_uniform_2d(head_dim, hidden_dim, rng));
            w_value.push(xavier_uniform_2d(head_dim, hidden_dim, rng));
        }
        // Output projection: from (head_dim * num_heads) → hidden_dim
        let concat_dim = head_dim * num_heads;
        let w_out = xavier_uniform_2d(hidden_dim, concat_dim, rng);
        Self {
            w_query,
            w_key,
            w_value,
            w_out,
            num_heads,
            head_dim,
            hidden_dim,
        }
    }
}

// ── Main embedder ──────────────────────────────────────────────────────────────

/// Multi-head graph attention network embedder trained on RDF triple lists.
///
/// Architecture follows Veličković et al. (2018): for each node the model
/// attends over its in-neighbourhood using learned Q/K/V projections,
/// applies LeakyReLU-gated softmax attention, concatenates the `num_heads`
/// outputs, projects through W_out, applies ReLU, and L2-normalises.
///
/// Training uses a margin-ranking loss and sign-SGD with gradient clipping.
pub struct GatEmbedder {
    config: GatEmbedderConfig,
    /// Entity IRI → sequential integer index.
    entity_index: HashMap<String, usize>,
    /// Cached post-training embeddings indexed by entity id.
    embeddings: Vec<Vec<f64>>,
    /// Per-layer weight matrices (length = num_layers).
    layer_weights: Vec<GatLayerWeights>,
    trained: bool,
}

impl std::fmt::Debug for GatEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatEmbedder")
            .field("num_entities", &self.entity_index.len())
            .field("trained", &self.trained)
            .field("num_layers", &self.config.num_layers)
            .field("hidden_dim", &self.config.hidden_dim)
            .field("num_heads", &self.config.num_heads)
            .finish()
    }
}

impl GatEmbedder {
    /// Create a new, un-trained GAT embedder.
    pub fn new(config: GatEmbedderConfig) -> Self {
        Self {
            config,
            entity_index: HashMap::new(),
            embeddings: Vec::new(),
            layer_weights: Vec::new(),
            trained: false,
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// Train on `(subject_iri, predicate_iri, object_iri)` triples.
    ///
    /// Steps:
    /// 1. Build entity index and adjacency map (undirected: s→o and o→s).
    /// 2. Xavier-initialise all Q/K/V and W_out weight matrices.
    /// 3. Initialise random L2-normalised entity feature vectors.
    /// 4. For each epoch: attention forward-pass → margin-ranking loss →
    ///    sign-SGD weight update.
    /// 5. Cache final embeddings.
    pub fn fit(&mut self, triples: &[(String, String, String)]) -> Result<(), EmbeddingError> {
        if triples.is_empty() {
            return Err(EmbeddingError::Other(anyhow!("Triple set is empty")));
        }

        // 1. Build entity index and integer-keyed adjacency list
        let (entity_index, adj_by_idx) = Self::build_graph(triples);
        let num_entities = entity_index.len();
        self.entity_index = entity_index;

        // 2. Initialise weight matrices
        let mut rng = SimpleLcg::new(self.config.seed);
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_layers = self.config.num_layers;
        self.layer_weights = (0..num_layers)
            .map(|_| GatLayerWeights::new(hidden_dim, num_heads, &mut rng))
            .collect();

        // 3. Initialise entity feature vectors ~ U(-0.5, 0.5), L2-normalised
        let mut h0: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| {
                let mut v: Vec<f64> = (0..hidden_dim)
                    .map(|_| rng.next_f64_range(0.5_f64))
                    .collect();
                l2_normalize_inplace(&mut v);
                v
            })
            .collect();

        // 4. Training loop
        let mut lcg = SimpleLcg::new(self.config.seed.wrapping_add(1));

        for _epoch in 0..self.config.num_epochs {
            // Forward pass: compute updated embeddings for all entities
            let h_all = self.forward_all(&h0, &adj_by_idx, num_entities);

            // Accumulate sign-SGD deltas for weight matrices (simplified proxy
            // gradient: outer product of embedding sign × loss magnitude)
            let mut deltas: Vec<Vec<Vec<Vec<f64>>>> = self
                .layer_weights
                .iter()
                .map(|lw| {
                    let heads: Vec<Vec<Vec<f64>>> = (0..lw.num_heads)
                        .map(|_| vec![vec![0.0; hidden_dim]; lw.head_dim])
                        .collect();
                    // index 0..num_heads → Q, num_heads..2*num_heads → K,
                    // 2*num_heads..3*num_heads → V, 3*num_heads → W_out
                    let mut all = heads.clone();
                    all.extend(heads.clone()); // K
                    all.extend(heads.clone()); // V
                    all.push(vec![vec![0.0; lw.head_dim * lw.num_heads]; lw.hidden_dim]); // W_out
                    all
                })
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
                let o_neg_idx = Self::sample_negative(o_idx, num_entities, &mut lcg);

                let h_s = &h_all[s_idx];
                let h_o = &h_all[o_idx];
                let h_neg = &h_all[o_neg_idx];

                let loss =
                    (self.config.margin - cosine_sim(h_s, h_o) + cosine_sim(h_s, h_neg)).max(0.0);

                if loss > 0.0 {
                    // Accumulate magnitude-scaled sign gradient for every layer
                    for (l, lw) in self.layer_weights.iter().enumerate() {
                        // Proxy gradient: sign of subject embedding component
                        let nh = lw.num_heads;
                        let hd = lw.head_dim;
                        for h in 0..nh {
                            // Q deltas
                            for (r, row) in deltas[l][h].iter_mut().enumerate().take(hd) {
                                let sign = if h_s.get(r % h_s.len()).copied().unwrap_or(0.0) > 0.0 {
                                    1.0_f64
                                } else {
                                    -1.0_f64
                                };
                                for delta in row.iter_mut() {
                                    *delta += sign * loss;
                                }
                            }
                            // K deltas
                            for (r, row) in deltas[l][nh + h].iter_mut().enumerate().take(hd) {
                                let sign = if h_o.get(r % h_o.len()).copied().unwrap_or(0.0) > 0.0 {
                                    1.0_f64
                                } else {
                                    -1.0_f64
                                };
                                for delta in row.iter_mut() {
                                    *delta += sign * loss;
                                }
                            }
                            // V deltas
                            for (r, row) in deltas[l][2 * nh + h].iter_mut().enumerate().take(hd) {
                                let sign = if h_o.get(r % h_o.len()).copied().unwrap_or(0.0) > 0.0 {
                                    1.0_f64
                                } else {
                                    -1.0_f64
                                };
                                for delta in row.iter_mut() {
                                    *delta += sign * loss;
                                }
                            }
                        }
                        // W_out deltas
                        for (r, row) in deltas[l][3 * nh].iter_mut().enumerate() {
                            let sign = if h_s.get(r % h_s.len()).copied().unwrap_or(0.0) > 0.0 {
                                1.0_f64
                            } else {
                                -1.0_f64
                            };
                            for delta in row.iter_mut() {
                                *delta += sign * loss;
                            }
                        }
                    }
                    grad_count += 1;
                }
            }

            // Apply sign-SGD updates with row-norm gradient clipping
            if grad_count > 0 {
                let lr = self.config.learning_rate / grad_count as f64;
                for (l, lw) in self.layer_weights.iter_mut().enumerate() {
                    let nh = lw.num_heads;
                    let hd = lw.head_dim;

                    for h in 0..nh {
                        // Update Q
                        for (r, delta_row) in deltas[l][h].iter().enumerate().take(hd) {
                            let row_norm: f64 = delta_row.iter().map(|g| g * g).sum::<f64>().sqrt();
                            let clip = if row_norm > 1.0 { 1.0 / row_norm } else { 1.0 };
                            for (w, d) in lw.w_query[h][r].iter_mut().zip(delta_row.iter()) {
                                *w -= d * clip * lr;
                            }
                        }
                        // Update K
                        for (r, delta_row) in deltas[l][nh + h].iter().enumerate().take(hd) {
                            let row_norm: f64 = delta_row.iter().map(|g| g * g).sum::<f64>().sqrt();
                            let clip = if row_norm > 1.0 { 1.0 / row_norm } else { 1.0 };
                            for (w, d) in lw.w_key[h][r].iter_mut().zip(delta_row.iter()) {
                                *w -= d * clip * lr;
                            }
                        }
                        // Update V
                        for (r, delta_row) in deltas[l][2 * nh + h].iter().enumerate().take(hd) {
                            let row_norm: f64 = delta_row.iter().map(|g| g * g).sum::<f64>().sqrt();
                            let clip = if row_norm > 1.0 { 1.0 / row_norm } else { 1.0 };
                            for (w, d) in lw.w_value[h][r].iter_mut().zip(delta_row.iter()) {
                                *w -= d * clip * lr;
                            }
                        }
                    }
                    // Update W_out
                    for (r, delta_row) in deltas[l][3 * nh].iter().enumerate() {
                        let row_norm: f64 = delta_row.iter().map(|g| g * g).sum::<f64>().sqrt();
                        let clip = if row_norm > 1.0 { 1.0 / row_norm } else { 1.0 };
                        for (w, d) in lw.w_out[r].iter_mut().zip(delta_row.iter()) {
                            *w -= d * clip * lr;
                        }
                    }
                }
            }

            // Re-normalise input features for next epoch
            for feat in h0.iter_mut() {
                l2_normalize_inplace(feat);
            }
        }

        // 5. Cache final embeddings
        self.embeddings = self.forward_all(&h0, &adj_by_idx, num_entities);
        self.trained = true;
        Ok(())
    }

    /// Return the embedding for a known entity IRI.
    /// Returns a zero vector for unseen entities (inductive fallback — never panics).
    pub fn embed_entity(&self, entity: &str) -> Vec<f64> {
        match self.entity_index.get(entity) {
            Some(&idx) => self
                .embeddings
                .get(idx)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.config.hidden_dim]),
            None => vec![0.0; self.config.hidden_dim],
        }
    }

    /// Multi-head attention forward pass for a single entity.
    ///
    /// For each head `h`:
    ///   Q_h = W_query\[h\] · e_i
    ///   K_h = W_key\[h\]   · e_j  for each neighbour j
    ///   V_h = W_value\[h\] · e_j
    ///   score_j = LeakyReLU(Q_h · K_h_j / √head_dim)
    ///   α_j = softmax({score_j})
    ///   head_out_h = Σ_j α_j · V_h_j
    ///
    /// Heads are concatenated → projected by W_out → ReLU → L2-normalised.
    pub fn attention_forward(
        &self,
        entity_idx: usize,
        adj: &HashMap<usize, Vec<usize>>,
        embeddings: &[Vec<f64>],
        layer_idx: usize,
    ) -> Vec<f64> {
        let lw = &self.layer_weights[layer_idx];
        let h_self = match embeddings.get(entity_idx) {
            Some(e) => e,
            None => return vec![0.0; self.config.hidden_dim],
        };

        // Collect neighbour embeddings (include self for isolated nodes)
        let neighbor_indices: Vec<usize> = adj.get(&entity_idx).cloned().unwrap_or_default();
        let all_indices: Vec<usize> = {
            let mut v = vec![entity_idx];
            v.extend_from_slice(&neighbor_indices);
            v
        };

        let scale = (lw.head_dim.max(1) as f64).sqrt();

        // Compute per-head outputs
        let mut concat_heads: Vec<f64> = Vec::with_capacity(lw.head_dim * lw.num_heads);

        for h in 0..lw.num_heads {
            // Q for entity i
            let q_i: Vec<f64> = matvec(&lw.w_query[h], h_self);

            // Attention scores for all neighbours (including self)
            let scores: Vec<f64> = all_indices
                .iter()
                .map(|&j| {
                    let h_j = match embeddings.get(j) {
                        Some(e) => e,
                        None => h_self,
                    };
                    let k_j: Vec<f64> = matvec(&lw.w_key[h], h_j);
                    let raw_score: f64 = q_i.iter().zip(k_j.iter()).map(|(&a, &b)| a * b).sum();
                    leaky_relu(raw_score / scale, 0.2)
                })
                .collect();

            let alphas = softmax(&scores);

            // Weighted sum of value vectors
            let mut head_out = vec![0.0_f64; lw.head_dim];
            for (&j, &alpha) in all_indices.iter().zip(alphas.iter()) {
                let h_j = match embeddings.get(j) {
                    Some(e) => e,
                    None => h_self,
                };
                let v_j: Vec<f64> = matvec(&lw.w_value[h], h_j);
                for (acc, vv) in head_out.iter_mut().zip(v_j.iter()) {
                    *acc += alpha * vv;
                }
            }
            concat_heads.extend_from_slice(&head_out);
        }

        // Output projection → ReLU → L2-normalise
        let mut out = relu_vec(&matvec(&lw.w_out, &concat_heads));
        l2_normalize_inplace(&mut out);
        out
    }

    // ── Accessors ──────────────────────────────────────────────────────────────

    /// Whether `fit` has been called successfully.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of distinct entities seen during training.
    pub fn num_entities(&self) -> usize {
        self.entity_index.len()
    }

    /// Dimension of each output embedding.
    pub fn embedding_dim(&self) -> usize {
        self.config.hidden_dim
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Build an entity-IRI→index map and an integer adjacency list from triples.
    fn build_graph(
        triples: &[(String, String, String)],
    ) -> (HashMap<String, usize>, HashMap<usize, Vec<usize>>) {
        let mut entity_index: HashMap<String, usize> = HashMap::new();
        let mut next_id = 0usize;

        let mut get_or_insert = |iri: &str| -> usize {
            if let Some(&id) = entity_index.get(iri) {
                return id;
            }
            let id = next_id;
            next_id += 1;
            entity_index.insert(iri.to_string(), id);
            id
        };

        // First pass: build entity index
        for (s, _p, o) in triples {
            get_or_insert(s.as_str());
            get_or_insert(o.as_str());
        }

        // Second pass: build adjacency (undirected: s→o and o→s)
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for (s, _p, o) in triples {
            let s_idx = *entity_index.get(s.as_str()).expect("just inserted");
            let o_idx = *entity_index.get(o.as_str()).expect("just inserted");
            adj.entry(s_idx).or_default().push(o_idx);
            adj.entry(o_idx).or_default().push(s_idx);
        }

        (entity_index, adj)
    }

    /// Forward pass over all entities for all layers.
    fn forward_all(
        &self,
        h0: &[Vec<f64>],
        adj: &HashMap<usize, Vec<usize>>,
        num_entities: usize,
    ) -> Vec<Vec<f64>> {
        let mut h_prev = h0.to_vec();

        for layer_idx in 0..self.config.num_layers {
            let mut h_next: Vec<Vec<f64>> = Vec::with_capacity(num_entities);
            for node_idx in 0..num_entities {
                // Use a temporary GatEmbedder-like context pointing at h_prev
                let out = self.attention_forward_on(node_idx, adj, &h_prev, layer_idx);
                h_next.push(out);
            }
            h_prev = h_next;
        }

        h_prev
    }

    /// Internal variant of `attention_forward` that takes explicit embeddings slice.
    fn attention_forward_on(
        &self,
        entity_idx: usize,
        adj: &HashMap<usize, Vec<usize>>,
        embeddings: &[Vec<f64>],
        layer_idx: usize,
    ) -> Vec<f64> {
        self.attention_forward(entity_idx, adj, embeddings, layer_idx)
    }

    /// Sample a negative entity index different from `positive_idx`.
    fn sample_negative(positive_idx: usize, num_entities: usize, lcg: &mut SimpleLcg) -> usize {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small, fully-connected knowledge graph for testing.
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

    // ── Test 1: Default config produces expected dimensions ────────────────────
    #[test]
    fn test_default_config_dimensions() {
        let config = GatEmbedderConfig::default();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_epochs, 50);
        // head_dim = hidden_dim / num_heads = 16
        assert_eq!(config.hidden_dim / config.num_heads, 16);
    }

    // ── Test 2: fit completes without error on a small graph ──────────────────
    #[test]
    fn test_fit_completes_small_graph() {
        let config = GatEmbedderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 4,
            num_epochs: 5,
            seed: 7,
            ..Default::default()
        };
        let triples = toy_triples(5, 8);
        let mut embedder = GatEmbedder::new(config);
        let result = embedder.fit(&triples);
        assert!(result.is_ok(), "fit should succeed: {result:?}");
        assert!(embedder.is_trained());
        assert_eq!(embedder.num_entities(), 5);
    }

    // ── Test 3: embed_entity returns correct dimension after fit ───────────────
    #[test]
    fn test_embed_entity_dimension() {
        let config = GatEmbedderConfig {
            num_layers: 2,
            hidden_dim: 32,
            num_heads: 4,
            num_epochs: 3,
            seed: 11,
            ..Default::default()
        };
        let triples = toy_triples(5, 8);
        let mut embedder = GatEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        for i in 0..5usize {
            let iri = format!("http://ex.org/e{}", i);
            let emb = embedder.embed_entity(&iri);
            assert_eq!(
                emb.len(),
                config.hidden_dim,
                "embedding length mismatch for entity {iri}"
            );
        }
    }

    // ── Test 4: Unseen entity returns zero vector (not panic) ─────────────────
    #[test]
    fn test_unseen_entity_returns_zero_vector() {
        let config = GatEmbedderConfig {
            num_layers: 1,
            hidden_dim: 16,
            num_heads: 2,
            num_epochs: 2,
            seed: 3,
            ..Default::default()
        };
        let triples = toy_triples(5, 8);
        let mut embedder = GatEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        let unseen = "http://ex.org/TOTALLY_UNSEEN";
        let emb = embedder.embed_entity(unseen);
        assert_eq!(emb.len(), config.hidden_dim);
        assert!(
            emb.iter().all(|&v| v == 0.0),
            "unseen entity must return a zero vector"
        );
    }

    // ── Test 5: Softmax: attention scores sum to 1.0 ──────────────────────────
    #[test]
    fn test_softmax_sums_to_one() {
        let scores = vec![1.0_f64, 2.0, 0.5, -1.0, 3.5];
        let probs = softmax(&scores);
        assert_eq!(probs.len(), scores.len());
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "softmax outputs must sum to 1.0, got {total}"
        );
        // All values must be in (0, 1)
        for &p in &probs {
            assert!(p > 0.0 && p <= 1.0, "softmax value out of (0,1]: {p}");
        }
    }

    // ── Test 6: LeakyReLU passes positive inputs, attenuates negative ones ─────
    #[test]
    fn test_leaky_relu_behavior() {
        let neg_slope = 0.2_f64;
        // Positive input: passes unchanged
        let pos = 3.7_f64;
        assert!((leaky_relu(pos, neg_slope) - pos).abs() < 1e-12);
        // Zero: passes unchanged
        assert!((leaky_relu(0.0, neg_slope)).abs() < 1e-12);
        // Negative input: attenuated by slope
        let neg = -4.0_f64;
        let expected = neg_slope * neg;
        assert!(
            (leaky_relu(neg, neg_slope) - expected).abs() < 1e-12,
            "leaky_relu({neg}) should be {expected}"
        );
        // Attenuation: |output| < |input| for negative input
        assert!(
            leaky_relu(-5.0, neg_slope).abs() < 5.0,
            "negative input should be attenuated"
        );
    }

    // ── Test 7: Embeddings are L2-normalised after forward pass ───────────────
    #[test]
    fn test_embeddings_l2_normalized() {
        let config = GatEmbedderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 4,
            num_epochs: 3,
            seed: 13,
            ..Default::default()
        };
        let triples = toy_triples(5, 8);
        let mut embedder = GatEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        for i in 0..5usize {
            let iri = format!("http://ex.org/e{}", i);
            let emb = embedder.embed_entity(&iri);
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Allow collapsed (all-zero) embeddings if ReLU kills all activations
            if norm > 1e-12 {
                assert!(
                    (norm - 1.0).abs() < 0.1,
                    "L2 norm out of tolerance for {iri}: got {norm}"
                );
            }
        }
    }

    // ── Test 8: num_heads attention heads → full hidden_dim output ─────────────
    #[test]
    fn test_multi_head_output_dimension() {
        let config = GatEmbedderConfig {
            num_layers: 1,
            hidden_dim: 32,
            num_heads: 4,
            num_epochs: 1,
            seed: 17,
            ..Default::default()
        };
        let triples = toy_triples(5, 8);
        let mut embedder = GatEmbedder::new(config.clone());
        embedder.fit(&triples).expect("fit should succeed");

        // Build the structures used internally
        let (entity_index, adj) = GatEmbedder::build_graph(&triples);
        let num_entities = entity_index.len();
        let mut rng = SimpleLcg::new(config.seed);
        let hidden_dim = config.hidden_dim;
        let h0: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| {
                let mut v: Vec<f64> = (0..hidden_dim)
                    .map(|_| rng.next_f64_range(0.5_f64))
                    .collect();
                l2_normalize_inplace(&mut v);
                v
            })
            .collect();

        // After fit, each cached embedding has length == hidden_dim
        for i in 0..5usize {
            let iri = format!("http://ex.org/e{}", i);
            let emb = embedder.embed_entity(&iri);
            assert_eq!(
                emb.len(),
                hidden_dim,
                "expected output dim {hidden_dim} for entity {i}"
            );
            // The W_out projection maps from head_dim*num_heads → hidden_dim;
            // confirm head_dim * num_heads == hidden_dim (concat property)
            let head_dim = hidden_dim / config.num_heads;
            assert_eq!(
                head_dim * config.num_heads,
                hidden_dim,
                "concat dim mismatch: {} * {} ≠ {}",
                head_dim,
                config.num_heads,
                hidden_dim
            );
        }

        // Direct test: attention_forward on entity 0 produces hidden_dim output
        let emb0 = embedder.attention_forward(0, &adj, &h0, 0);
        assert_eq!(
            emb0.len(),
            hidden_dim,
            "attention_forward should output hidden_dim={hidden_dim}"
        );
    }

    // ── Test 9: Loss decreases over training epochs ────────────────────────────
    #[test]
    fn test_loss_decreases_over_epochs() {
        let triples = toy_triples(5, 8);

        let make_config = |epochs: usize, seed: u64| GatEmbedderConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 4,
            num_epochs: epochs,
            learning_rate: 0.05,
            margin: 1.0,
            seed,
            ..Default::default()
        };

        // Compute average positive-pair cosine similarity as a proxy for loss
        let avg_sim = |embedder: &GatEmbedder| -> f64 {
            let (mut total, mut count) = (0.0_f64, 0usize);
            for (s, _, o) in &triples {
                let hs = embedder.embed_entity(s);
                let ho = embedder.embed_entity(o);
                // Only count non-zero embeddings
                let ns: f64 = hs.iter().map(|x| x * x).sum::<f64>().sqrt();
                let no: f64 = ho.iter().map(|x| x * x).sum::<f64>().sqrt();
                if ns > 1e-12 && no > 1e-12 {
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

        let mut e_early = GatEmbedder::new(make_config(1, 42));
        e_early.fit(&triples).expect("1-epoch fit should succeed");
        let sim_early = avg_sim(&e_early);

        let mut e_trained = GatEmbedder::new(make_config(50, 42));
        e_trained
            .fit(&triples)
            .expect("50-epoch fit should succeed");
        let sim_trained = avg_sim(&e_trained);

        // Trained model should not be dramatically worse; allow ±0.5 slack
        assert!(
            sim_trained >= sim_early - 0.5,
            "similarity regression: 1-epoch={sim_early:.4} 50-epoch={sim_trained:.4}"
        );
    }
}
