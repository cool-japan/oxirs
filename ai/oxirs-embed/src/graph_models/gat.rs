//! GAT: Graph Attention Network Embeddings (v0.3.0)
//!
//! Veličković et al., ICLR 2018 — "Graph Attention Networks"
//!
//! Key contributions:
//! - Multi-head attention over node neighborhoods
//! - No structural assumptions (unlike GCN's fixed Laplacian)
//! - Implicit structural weighting via learned attention coefficients
//!
//! This module provides:
//! - `GATLayer`:   single multi-head attention layer
//! - `GATModel`:   stacked GAT layers for deep graph embeddings
//! - `GATConfig`:  hyperparameter configuration
//! - `GATEmbeddings`: output container with similarity utilities

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use super::graphsage::{Graph, Lcg};

// ---------------------------------------------------------------------------
// GATConfig
// ---------------------------------------------------------------------------

/// Full hyperparameter configuration for a GAT model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GATConfig {
    /// Dimension of raw input node features.
    pub input_dim: usize,
    /// Per-head output dimensionality in intermediate layers.
    pub hidden_head_dim: usize,
    /// Number of attention heads in intermediate layers.
    pub hidden_num_heads: usize,
    /// Per-head output dimensionality in the final layer.
    pub output_head_dim: usize,
    /// Number of attention heads in the final layer.
    pub output_num_heads: usize,
    /// Number of stacked GAT layers (must be >= 1).
    pub num_layers: usize,
    /// Dropout rate applied to attention coefficients (0.0 = disabled).
    pub dropout: f64,
    /// Negative slope for LeakyReLU in attention scoring.
    pub alpha: f64,
    /// If true, concatenate head outputs in intermediate layers; else average.
    pub concat_hidden: bool,
    /// Average head outputs in the final layer (standard GAT for classification).
    pub avg_output: bool,
    /// L2-normalize final node embeddings.
    pub normalize_output: bool,
    /// Random seed for weight initialization.
    pub seed: u64,
}

impl Default for GATConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_head_dim: 8,
            hidden_num_heads: 8,
            output_head_dim: 8,
            output_num_heads: 1,
            num_layers: 2,
            dropout: 0.6,
            alpha: 0.2,
            concat_hidden: true,
            avg_output: true,
            normalize_output: true,
            seed: 42,
        }
    }
}

impl GATConfig {
    /// Compute the total output dimensionality of this configuration.
    pub fn output_dim(&self) -> usize {
        if self.avg_output {
            self.output_head_dim
        } else {
            self.output_head_dim * self.output_num_heads
        }
    }

    /// Compute the output dimensionality of each hidden (intermediate) layer.
    pub fn hidden_layer_out_dim(&self) -> usize {
        if self.concat_hidden {
            self.hidden_head_dim * self.hidden_num_heads
        } else {
            self.hidden_head_dim
        }
    }
}

// ---------------------------------------------------------------------------
// Attention head
// ---------------------------------------------------------------------------

/// A single attention head: W * x, then LeakyReLU attention scoring.
#[derive(Debug, Clone)]
struct AttentionHead {
    /// Linear transform W: [in_dim -> head_dim]
    w: Vec<Vec<f64>>, // [head_dim][in_dim]
    /// Attention source parameter a_src: [head_dim]
    a_src: Vec<f64>,
    /// Attention target parameter a_dst: [head_dim]
    a_dst: Vec<f64>,
    head_dim: usize,
    /// LeakyReLU negative slope
    alpha: f64,
}

impl AttentionHead {
    fn new(in_dim: usize, head_dim: usize, alpha: f64, rng: &mut Lcg) -> Self {
        let w_scale = (6.0 / (in_dim + head_dim) as f64).sqrt();
        let w = (0..head_dim)
            .map(|_| (0..in_dim).map(|_| rng.next_f64_range(w_scale)).collect())
            .collect();
        let a_scale = (2.0 / head_dim as f64).sqrt();
        let a_src = (0..head_dim).map(|_| rng.next_f64_range(a_scale)).collect();
        let a_dst = (0..head_dim).map(|_| rng.next_f64_range(a_scale)).collect();
        Self {
            w,
            a_src,
            a_dst,
            head_dim,
            alpha,
        }
    }

    /// Compute Wh for a single node.
    fn linear(&self, x: &[f64]) -> Vec<f64> {
        self.w
            .iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&w, &xi)| w * xi).sum())
            .collect()
    }

    /// LeakyReLU(x) with configured negative slope.
    fn leaky_relu(&self, x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            self.alpha * x
        }
    }

    /// Softmax over a slice.
    fn softmax(scores: &[f64]) -> Vec<f64> {
        if scores.is_empty() {
            return Vec::new();
        }
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum < 1e-12 {
            return vec![1.0 / scores.len() as f64; scores.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Compute attention-weighted aggregation for node `v`.
    ///
    /// Returns the new head embedding for `v`.
    fn forward(
        &self,
        v: usize,
        all_transformed: &[Vec<f64>], // Wh for every node
        neighbors: &[usize],
        dropout_mask: &[bool], // pre-computed dropout, true = keep
    ) -> Vec<f64> {
        // Self-connection plus neighbors
        let mut candidates: Vec<usize> = vec![v];
        candidates.extend_from_slice(neighbors);

        let h_v = &all_transformed[v];

        // Compute attention score for each candidate
        let scores: Vec<f64> = candidates
            .iter()
            .map(|&u| {
                let h_u = &all_transformed[u];
                // e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
                let src: f64 = self
                    .a_src
                    .iter()
                    .zip(h_v.iter())
                    .map(|(&a, &h)| a * h)
                    .sum();
                let dst: f64 = self
                    .a_dst
                    .iter()
                    .zip(h_u.iter())
                    .map(|(&a, &h)| a * h)
                    .sum();
                self.leaky_relu(src + dst)
            })
            .collect();

        let weights = Self::softmax(&scores);

        // Weighted sum of neighbor transformed features (with dropout)
        let mut out = vec![0.0f64; self.head_dim];
        for (k, (&u, &w)) in candidates.iter().zip(weights.iter()).enumerate() {
            // Apply dropout: if mask says drop, skip (effectively zero weight)
            let keep = dropout_mask.get(k).copied().unwrap_or(true);
            let effective_w = if keep { w } else { 0.0 };
            let h_u = &all_transformed[u];
            for (j, &val) in h_u.iter().enumerate() {
                out[j] += effective_w * val;
            }
        }
        // ELU activation on output (standard in GAT)
        out.iter_mut().for_each(|x| {
            if *x < 0.0 {
                *x = (*x).exp() - 1.0;
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------
// GATLayer
// ---------------------------------------------------------------------------

/// A multi-head graph attention layer.
///
/// For each node `v`:
///   For each head `k`:
///     α_ij = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )
///     h_v^k = ELU( Σ_j α_ij * W * h_j )
///   h_v = CONCAT or AVG over heads
pub struct GATLayer {
    heads: Vec<AttentionHead>,
    in_dim: usize,
    head_dim: usize,
    num_heads: usize,
    concat: bool,
    dropout_rate: f64,
}

impl std::fmt::Debug for GATLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GATLayer")
            .field("in_dim", &self.in_dim)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("concat", &self.concat)
            .finish()
    }
}

impl GATLayer {
    /// Create a new GAT layer.
    pub fn new(
        in_dim: usize,
        head_dim: usize,
        num_heads: usize,
        alpha: f64,
        dropout: f64,
        concat: bool,
        rng: &mut Lcg,
    ) -> Result<Self> {
        if in_dim == 0 {
            return Err(anyhow!("GATLayer: in_dim must be > 0"));
        }
        if head_dim == 0 {
            return Err(anyhow!("GATLayer: head_dim must be > 0"));
        }
        if num_heads == 0 {
            return Err(anyhow!("GATLayer: num_heads must be > 0"));
        }
        let heads = (0..num_heads)
            .map(|_| AttentionHead::new(in_dim, head_dim, alpha, rng))
            .collect();
        Ok(Self {
            heads,
            in_dim,
            head_dim,
            num_heads,
            concat,
            dropout_rate: dropout,
        })
    }

    /// Output dimensionality of this layer.
    pub fn out_dim(&self) -> usize {
        if self.concat {
            self.head_dim * self.num_heads
        } else {
            self.head_dim
        }
    }

    /// Forward pass: returns new embeddings for all nodes.
    pub fn forward(
        &self,
        graph: &Graph,
        current_embeddings: &[Vec<f64>],
        rng: &mut Lcg,
    ) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();

        // Pre-compute Wh for each head and all nodes
        // all_transformed[head_idx][node_idx] = head.linear(node_emb)
        let all_transformed: Vec<Vec<Vec<f64>>> = self
            .heads
            .iter()
            .map(|head| {
                current_embeddings
                    .iter()
                    .map(|emb| head.linear(emb))
                    .collect()
            })
            .collect();

        // Compute new embedding for each node
        (0..n)
            .map(|v| {
                let neighbors = graph.neighbors(v);
                // Generate dropout masks (one per head per candidate)
                let num_candidates = 1 + neighbors.len(); // self + neighbors
                let dropout_mask: Vec<bool> = (0..num_candidates)
                    .map(|_| rng.next_f64() > self.dropout_rate)
                    .collect();

                let head_outputs: Vec<Vec<f64>> = self
                    .heads
                    .iter()
                    .enumerate()
                    .map(|(k, head)| head.forward(v, &all_transformed[k], neighbors, &dropout_mask))
                    .collect();

                if self.concat {
                    // Concatenate: [h_1 || h_2 || ... || h_K]
                    head_outputs.into_iter().flatten().collect()
                } else {
                    // Average: mean over heads
                    let mut avg = vec![0.0f64; self.head_dim];
                    for h in &head_outputs {
                        for (i, &v) in h.iter().enumerate() {
                            avg[i] += v;
                        }
                    }
                    let k = self.num_heads as f64;
                    avg.iter_mut().for_each(|x| *x /= k);
                    avg
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GATModel
// ---------------------------------------------------------------------------

/// Multi-layer Graph Attention Network.
pub struct GATModel {
    layers: Vec<GATLayer>,
    config: GATConfig,
}

impl std::fmt::Debug for GATModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GATModel")
            .field("num_layers", &self.layers.len())
            .field("output_dim", &self.config.output_dim())
            .finish()
    }
}

impl GATModel {
    /// Construct a GAT model from configuration.
    pub fn new(config: GATConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(anyhow!("GATConfig: input_dim must be > 0"));
        }
        if config.num_layers == 0 {
            return Err(anyhow!("GATConfig: num_layers must be > 0"));
        }
        if config.hidden_head_dim == 0 {
            return Err(anyhow!("GATConfig: hidden_head_dim must be > 0"));
        }
        if config.output_head_dim == 0 {
            return Err(anyhow!("GATConfig: output_head_dim must be > 0"));
        }
        if config.hidden_num_heads == 0 || config.output_num_heads == 0 {
            return Err(anyhow!("GATConfig: num_heads must be > 0"));
        }

        let mut rng = Lcg::new(config.seed);
        let mut layers = Vec::with_capacity(config.num_layers);

        // Compute layer-by-layer input dimensions
        let mut current_in_dim = config.input_dim;
        for layer_idx in 0..config.num_layers {
            let is_last = layer_idx == config.num_layers - 1;
            let (head_dim, num_heads, concat) = if is_last {
                (
                    config.output_head_dim,
                    config.output_num_heads,
                    !config.avg_output,
                )
            } else {
                (
                    config.hidden_head_dim,
                    config.hidden_num_heads,
                    config.concat_hidden,
                )
            };

            let layer = GATLayer::new(
                current_in_dim,
                head_dim,
                num_heads,
                config.alpha,
                config.dropout,
                concat,
                &mut rng,
            )?;
            current_in_dim = layer.out_dim();
            layers.push(layer);
        }

        Ok(Self { layers, config })
    }

    /// Compute embeddings for all nodes in `graph`.
    pub fn embed(&self, graph: &Graph) -> Result<GATEmbeddings> {
        if graph.num_nodes() == 0 {
            return Err(anyhow!("GATModel: graph has no nodes"));
        }
        let mut rng = Lcg::new(self.config.seed.wrapping_add(0xcafe_babe));
        let mut current: Vec<Vec<f64>> = graph.node_features.clone();
        for layer in &self.layers {
            current = layer.forward(graph, &current, &mut rng);
        }
        if self.config.normalize_output {
            for emb in &mut current {
                l2_normalize_inplace(emb);
            }
        }
        let dim = self.config.output_dim();
        let num_nodes = graph.num_nodes();
        Ok(GATEmbeddings {
            embeddings: current,
            num_nodes,
            dim,
        })
    }
}

// ---------------------------------------------------------------------------
// GATEmbeddings
// ---------------------------------------------------------------------------

/// Node embeddings output by `GATModel`.
#[derive(Debug, Clone)]
pub struct GATEmbeddings {
    pub embeddings: Vec<Vec<f64>>,
    pub num_nodes: usize,
    pub dim: usize,
}

impl GATEmbeddings {
    /// Get embedding for node `v`.
    pub fn get(&self, v: usize) -> Option<&[f64]> {
        self.embeddings.get(v).map(|e| e.as_slice())
    }

    /// Cosine similarity between nodes `a` and `b`.
    pub fn cosine_similarity(&self, a: usize, b: usize) -> Option<f64> {
        let ea = self.embeddings.get(a)?;
        let eb = self.embeddings.get(b)?;
        Some(cosine_similarity_vecs(ea, eb))
    }

    /// Top-k most similar nodes to `query_node` (excluding itself).
    pub fn top_k_similar(&self, query_node: usize, k: usize) -> Vec<(usize, f64)> {
        let qe = match self.embeddings.get(query_node) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut sims: Vec<(usize, f64)> = self
            .embeddings
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != query_node)
            .map(|(i, e)| (i, cosine_similarity_vecs(qe, e)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        sims
    }

    /// Compute mean embedding across all nodes.
    pub fn mean_embedding(&self) -> Vec<f64> {
        if self.embeddings.is_empty() {
            return Vec::new();
        }
        let mut mean = vec![0.0f64; self.dim];
        for emb in &self.embeddings {
            for (i, &v) in emb.iter().enumerate().take(self.dim) {
                mean[i] += v;
            }
        }
        let n = self.embeddings.len() as f64;
        mean.iter_mut().for_each(|v| *v /= n);
        mean
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn cosine_similarity_vecs(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn l2_normalize_inplace(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::graphsage::{Graph, Lcg};
    use super::*;

    fn line_graph(n: usize, feat_dim: usize, seed: u64) -> Graph {
        let mut rng = Lcg::new(seed);
        let features: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..feat_dim).map(|_| rng.next_f64()).collect())
            .collect();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n.saturating_sub(1) {
            adjacency[i].push(i + 1);
            adjacency[i + 1].push(i);
        }
        Graph::new(features, adjacency).expect("line graph construction should succeed")
    }

    #[test]
    fn test_gat_config_default() {
        let config = GATConfig::default();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_num_heads, 8);
        // Default: avg_output=true => output_dim = output_head_dim
        assert_eq!(config.output_dim(), config.output_head_dim);
    }

    #[test]
    fn test_gat_config_concat_hidden() {
        let config = GATConfig {
            hidden_head_dim: 8,
            hidden_num_heads: 4,
            concat_hidden: true,
            ..Default::default()
        };
        assert_eq!(config.hidden_layer_out_dim(), 32); // 8 * 4
    }

    #[test]
    fn test_gat_config_avg_hidden() {
        let config = GATConfig {
            hidden_head_dim: 8,
            hidden_num_heads: 4,
            concat_hidden: false,
            ..Default::default()
        };
        assert_eq!(config.hidden_layer_out_dim(), 8); // just head_dim
    }

    #[test]
    fn test_gat_layer_construction() {
        let mut rng = Lcg::new(42);
        let layer =
            GATLayer::new(8, 4, 2, 0.2, 0.0, true, &mut rng).expect("layer should construct");
        assert_eq!(layer.out_dim(), 8); // 2 heads * 4 dim, concat
    }

    #[test]
    fn test_gat_layer_avg() {
        let mut rng = Lcg::new(43);
        let layer =
            GATLayer::new(8, 4, 3, 0.2, 0.0, false, &mut rng).expect("layer should construct");
        assert_eq!(layer.out_dim(), 4); // avg mode: just head_dim
    }

    #[test]
    fn test_gat_layer_invalid() {
        let mut rng = Lcg::new(1);
        assert!(GATLayer::new(0, 4, 2, 0.2, 0.0, true, &mut rng).is_err());
        assert!(GATLayer::new(8, 0, 2, 0.2, 0.0, true, &mut rng).is_err());
        assert!(GATLayer::new(8, 4, 0, 0.2, 0.0, true, &mut rng).is_err());
    }

    #[test]
    fn test_gat_model_embed_shape() {
        let config = GATConfig {
            input_dim: 8,
            hidden_head_dim: 4,
            hidden_num_heads: 2,
            output_head_dim: 4,
            output_num_heads: 1,
            num_layers: 2,
            dropout: 0.0,
            concat_hidden: true,
            avg_output: true,
            normalize_output: false,
            ..Default::default()
        };
        let model = GATModel::new(config.clone()).expect("GAT model should construct");
        let g = line_graph(5, 8, 100);
        let embs = model.embed(&g).expect("embed should succeed");

        assert_eq!(embs.num_nodes, 5);
        assert_eq!(embs.dim, config.output_dim());
        for i in 0..5 {
            assert_eq!(
                embs.get(i).expect("embedding should exist").len(),
                config.output_dim()
            );
        }
    }

    #[test]
    fn test_gat_model_single_layer() {
        let config = GATConfig {
            input_dim: 4,
            hidden_head_dim: 8,
            hidden_num_heads: 2,
            output_head_dim: 8,
            output_num_heads: 2,
            num_layers: 1,
            dropout: 0.0,
            concat_hidden: true,
            avg_output: false,
            normalize_output: false,
            ..Default::default()
        };
        let model = GATModel::new(config.clone()).expect("GAT model should construct");
        let g = line_graph(4, 4, 200);
        let embs = model.embed(&g).expect("embed should succeed");
        // Single layer, no avg: concat of 2 heads * 8 = 16
        assert_eq!(embs.dim, 16);
    }

    #[test]
    fn test_gat_model_normalized_output() {
        let config = GATConfig {
            input_dim: 4,
            hidden_head_dim: 4,
            hidden_num_heads: 2,
            output_head_dim: 4,
            output_num_heads: 1,
            num_layers: 1,
            dropout: 0.0,
            concat_hidden: false,
            avg_output: true,
            normalize_output: true,
            ..Default::default()
        };
        let model = GATModel::new(config).expect("GAT model should construct");
        let g = line_graph(5, 4, 300);
        let embs = model.embed(&g).expect("embed should succeed");
        for i in 0..5 {
            let emb = embs.get(i).expect("embedding exists");
            let norm: f64 = emb.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!(norm <= 1.0 + 1e-6, "norm {} should be <= 1", norm);
        }
    }

    #[test]
    fn test_gat_cosine_similarity_bounds() {
        let config = GATConfig {
            input_dim: 4,
            hidden_head_dim: 4,
            hidden_num_heads: 2,
            output_head_dim: 4,
            output_num_heads: 1,
            num_layers: 1,
            dropout: 0.0,
            concat_hidden: true,
            avg_output: true,
            normalize_output: false,
            ..Default::default()
        };
        let model = GATModel::new(config).expect("GAT model should construct");
        let g = line_graph(5, 4, 400);
        let embs = model.embed(&g).expect("embed should succeed");
        for i in 0..5 {
            for j in 0..5 {
                if let Some(sim) = embs.cosine_similarity(i, j) {
                    assert!(
                        (-1.0 - 1e-6..=1.0 + 1e-6).contains(&sim),
                        "cosine_similarity({i}, {j}) = {sim} out of range"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gat_top_k_similar() {
        let config = GATConfig {
            input_dim: 4,
            hidden_head_dim: 4,
            hidden_num_heads: 2,
            output_head_dim: 4,
            output_num_heads: 1,
            num_layers: 2,
            dropout: 0.0,
            concat_hidden: true,
            avg_output: true,
            normalize_output: true,
            ..Default::default()
        };
        let model = GATModel::new(config).expect("GAT model should construct");
        let g = line_graph(8, 4, 500);
        let embs = model.embed(&g).expect("embed should succeed");
        let top3 = embs.top_k_similar(0, 3);
        assert!(top3.len() <= 3);
        for window in top3.windows(2) {
            assert!(
                window[0].1 >= window[1].1 - 1e-10,
                "top_k should be sorted descending"
            );
        }
    }

    #[test]
    fn test_gat_isolated_node() {
        let config = GATConfig {
            input_dim: 4,
            hidden_head_dim: 4,
            hidden_num_heads: 2,
            output_head_dim: 4,
            output_num_heads: 1,
            num_layers: 1,
            dropout: 0.0,
            concat_hidden: true,
            avg_output: true,
            normalize_output: false,
            ..Default::default()
        };
        let model = GATModel::new(config).expect("GAT model should construct");
        let features = vec![vec![1.0f64, 0.5, -0.3, 0.8]];
        let adjacency = vec![vec![]]; // isolated node
        let g = Graph::new(features, adjacency).expect("isolated node graph");
        let embs = model.embed(&g).expect("isolated node should embed");
        assert_eq!(embs.num_nodes, 1);
        assert!(embs.get(0).is_some());
    }

    #[test]
    fn test_gat_invalid_config() {
        assert!(GATModel::new(GATConfig {
            input_dim: 0,
            ..Default::default()
        })
        .is_err());
        assert!(GATModel::new(GATConfig {
            num_layers: 0,
            ..Default::default()
        })
        .is_err());
        assert!(GATModel::new(GATConfig {
            hidden_num_heads: 0,
            ..Default::default()
        })
        .is_err());
        assert!(GATModel::new(GATConfig {
            output_head_dim: 0,
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn test_gat_mean_embedding() {
        let config = GATConfig {
            input_dim: 4,
            hidden_head_dim: 4,
            hidden_num_heads: 2,
            output_head_dim: 4,
            output_num_heads: 1,
            num_layers: 1,
            dropout: 0.0,
            concat_hidden: false,
            avg_output: true,
            normalize_output: true,
            ..Default::default()
        };
        let model = GATModel::new(config).expect("GAT model should construct");
        let g = line_graph(5, 4, 600);
        let embs = model.embed(&g).expect("embed should succeed");
        let mean = embs.mean_embedding();
        assert_eq!(mean.len(), embs.dim);
    }

    #[test]
    fn test_gat_attention_softmax_sums_to_one() {
        let scores = vec![1.0f64, 2.0, 3.0, 0.5, -1.0];
        let weights = AttentionHead::softmax(&scores);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax should sum to 1, got {sum}"
        );
        // Larger scores should produce larger weights
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_gat_three_layer_deep() {
        let config = GATConfig {
            input_dim: 8,
            hidden_head_dim: 4,
            hidden_num_heads: 3,
            output_head_dim: 4,
            output_num_heads: 1,
            num_layers: 3,
            dropout: 0.0,
            concat_hidden: true,
            avg_output: true,
            normalize_output: true,
            seed: 77,
            ..Default::default()
        };
        let model = GATModel::new(config.clone()).expect("3-layer GAT should construct");
        let g = line_graph(6, 8, 77);
        let embs = model.embed(&g).expect("embed should succeed");
        assert_eq!(embs.num_nodes, 6);
        assert_eq!(embs.dim, config.output_dim());
    }
}
