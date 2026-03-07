//! Graph Attention Networks (GAT) - Basic Implementation
//!
//! Veličković et al. (2018) - ICLR
//! "Graph Attention Networks"
//!
//! Key innovation: learn attention coefficients between nodes and their neighbors,
//! enabling the model to selectively focus on relevant structural information.
//! Multi-head attention provides stability and richer representations.

use crate::EmbeddingError;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use super::graphsage::{cosine_similarity_vecs, dot_product, GraphData, SimpleLcg};

/// Configuration for a Graph Attention Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatConfig {
    /// Dimensionality of input node features
    pub input_dim: usize,
    /// Dimensionality of output per attention head
    pub head_output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate (applied to attention coefficients)
    pub dropout: f64,
    /// LeakyReLU negative slope for attention scoring
    pub alpha: f64,
    /// If true, concatenate head outputs; else average them
    pub concat_heads: bool,
    /// L2-normalize output embeddings
    pub normalize_output: bool,
    /// Random seed for parameter initialization
    pub seed: u64,
}

impl Default for GatConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            head_output_dim: 8,
            num_heads: 8,
            dropout: 0.6,
            alpha: 0.2,
            concat_heads: true,
            normalize_output: true,
            seed: 42,
        }
    }
}

impl GatConfig {
    /// Compute the final output dimensionality
    pub fn output_dim(&self) -> usize {
        if self.concat_heads {
            self.head_output_dim * self.num_heads
        } else {
            self.head_output_dim
        }
    }
}

/// A single attention head in the GAT layer
///
/// Computes e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) for each edge (i,j),
/// then normalizes with softmax over the neighborhood.
#[derive(Debug, Clone)]
struct AttentionHead {
    /// Linear transform W: [input_dim x head_output_dim]
    w: Vec<Vec<f64>>,
    /// Attention source weights a_src: [head_output_dim]
    a_src: Vec<f64>,
    /// Attention target weights a_dst: [head_output_dim]
    a_dst: Vec<f64>,
    /// Output dimensionality (head_output_dim)
    output_dim: usize,
    /// LeakyReLU negative slope
    alpha: f64,
}

impl AttentionHead {
    /// Create a new attention head with Xavier initialization
    fn new(input_dim: usize, output_dim: usize, alpha: f64, rng: &mut SimpleLcg) -> Self {
        let scale = (6.0 / (input_dim + output_dim) as f64).sqrt();
        let w = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| rng.next_f64_range(scale)).collect())
            .collect();

        let attn_scale = (2.0 / output_dim as f64).sqrt();
        let a_src = (0..output_dim)
            .map(|_| rng.next_f64_range(attn_scale))
            .collect();
        let a_dst = (0..output_dim)
            .map(|_| rng.next_f64_range(attn_scale))
            .collect();

        Self {
            w,
            a_src,
            a_dst,
            output_dim,
            alpha,
        }
    }

    /// Apply the linear transform W to a feature vector
    fn transform(&self, feat: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; self.output_dim];
        for (i, row) in self.w.iter().enumerate() {
            for (j, &wv) in row.iter().enumerate() {
                if j < feat.len() {
                    out[i] += wv * feat[j];
                }
            }
        }
        out
    }

    /// Compute unnormalized attention coefficient e_ij
    ///
    /// e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
    fn attention_coeff(&self, h_i: &[f64], h_j: &[f64]) -> f64 {
        let src_score = dot_product(&self.a_src, h_i);
        let dst_score = dot_product(&self.a_dst, h_j);
        Self::leaky_relu(src_score + dst_score, self.alpha)
    }

    /// LeakyReLU: max(alpha*x, x)
    fn leaky_relu(x: f64, alpha: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            alpha * x
        }
    }

    /// Softmax over a slice of scores, returns normalized attention weights
    fn softmax(scores: &[f64]) -> Vec<f64> {
        if scores.is_empty() {
            return Vec::new();
        }
        // Numerical stability: subtract max
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum < 1e-12 {
            // Uniform if all exps are ~0
            return vec![1.0 / scores.len() as f64; scores.len()];
        }
        exps.iter().map(|e| e / sum).collect()
    }

    /// Forward pass for a single node.
    ///
    /// Computes: h'_i = sigma(sum_{j in N(i)} alpha_ij * W*h_j)
    /// where alpha_ij are the softmax-normalized attention coefficients.
    fn forward(&self, node_feat: &[f64], neighbor_feats: &[Vec<f64>]) -> Vec<f64> {
        // Transform self and neighbors
        let h_self = self.transform(node_feat);

        if neighbor_feats.is_empty() {
            // No neighbors: return self-transformed feature with self-attention
            return h_self;
        }

        let neighbor_transformed: Vec<Vec<f64>> =
            neighbor_feats.iter().map(|f| self.transform(f)).collect();

        // Compute attention coefficients (including self-loop)
        let mut all_feats = vec![&h_self as &Vec<f64>];
        all_feats.extend(neighbor_transformed.iter());

        let scores: Vec<f64> = all_feats
            .iter()
            .map(|h_j| self.attention_coeff(&h_self, h_j))
            .collect();

        let weights = Self::softmax(&scores);

        // Weighted sum of transformed features
        let mut output = vec![0.0f64; self.output_dim];
        for (weight, h_j) in weights.iter().zip(all_feats.iter()) {
            for (o, &v) in output.iter_mut().zip(h_j.iter()) {
                *o += weight * v;
            }
        }

        // Apply ELU activation (approximated as LeakyReLU here)
        output
            .into_iter()
            .map(|x| Self::leaky_relu(x, self.alpha))
            .collect()
    }
}

/// Graph Attention Network embedding model
///
/// Implements multi-head graph attention as described in Veličković et al. (2018).
/// Each attention head independently computes attention-weighted aggregations,
/// and the results are either concatenated or averaged.
#[derive(Debug, Clone)]
pub struct Gat {
    /// Model configuration
    config: GatConfig,
    /// Attention heads
    heads: Vec<AttentionHead>,
}

impl Gat {
    /// Create a new GAT model with the given configuration
    pub fn new(config: GatConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(anyhow!("input_dim must be > 0"));
        }
        if config.num_heads == 0 {
            return Err(anyhow!("num_heads must be > 0"));
        }
        if config.head_output_dim == 0 {
            return Err(anyhow!("head_output_dim must be > 0"));
        }

        let mut rng = SimpleLcg::new(config.seed);
        let heads = (0..config.num_heads)
            .map(|_| {
                AttentionHead::new(
                    config.input_dim,
                    config.head_output_dim,
                    config.alpha,
                    &mut rng,
                )
            })
            .collect();

        Ok(Self { config, heads })
    }

    /// Generate embeddings for all nodes using multi-head attention
    pub fn embed(&self, graph: &GraphData) -> Result<GatEmbeddings> {
        if graph.num_nodes() == 0 {
            return Err(anyhow!("Graph has no nodes"));
        }
        if graph.feature_dim() != self.config.input_dim {
            return Err(anyhow!(
                "Graph feature_dim {} != GAT input_dim {}",
                graph.feature_dim(),
                self.config.input_dim
            ));
        }

        let embeddings: Vec<Vec<f64>> = (0..graph.num_nodes())
            .map(|node| self.forward_node(node, graph))
            .collect();

        let embeddings = if self.config.normalize_output {
            embeddings.into_iter().map(|e| normalize_l2(&e)).collect()
        } else {
            embeddings
        };

        let output_dim = self.config.output_dim();
        let num_nodes = graph.num_nodes();

        Ok(GatEmbeddings {
            embeddings,
            config: self.config.clone(),
            num_nodes,
            dim: output_dim,
        })
    }

    /// Compute embedding for a single node using all attention heads
    fn forward_node(&self, node: usize, graph: &GraphData) -> Vec<f64> {
        let node_feat = match graph.node_features.get(node) {
            Some(f) => f.as_slice(),
            None => return vec![0.0; self.config.output_dim()],
        };

        let neighbors = graph.neighbors(node);
        let neighbor_feats: Vec<Vec<f64>> = neighbors
            .iter()
            .filter_map(|&n| graph.node_features.get(n).cloned())
            .collect();

        // Run each attention head
        let head_outputs: Vec<Vec<f64>> = self
            .heads
            .iter()
            .map(|head| head.forward(node_feat, &neighbor_feats))
            .collect();

        if self.config.concat_heads {
            // Concatenate all head outputs
            let mut concat = Vec::with_capacity(self.config.output_dim());
            for head_out in &head_outputs {
                concat.extend(head_out.iter().copied());
            }
            concat
        } else {
            // Average across heads
            let dim = self.config.head_output_dim;
            let mut avg = vec![0.0f64; dim];
            for head_out in &head_outputs {
                for (a, &v) in avg.iter_mut().zip(head_out.iter()) {
                    *a += v;
                }
            }
            let n = self.heads.len() as f64;
            avg.iter_mut().for_each(|v| *v /= n);
            avg
        }
    }
}

/// Output embeddings from GAT inference
#[derive(Debug, Clone)]
pub struct GatEmbeddings {
    /// Embedding vectors indexed by node ID
    pub embeddings: Vec<Vec<f64>>,
    /// Configuration used
    pub config: GatConfig,
    /// Number of nodes
    pub num_nodes: usize,
    /// Embedding dimensionality
    pub dim: usize,
}

impl GatEmbeddings {
    /// Get embedding for a specific node
    pub fn get(&self, node: usize) -> Option<&[f64]> {
        self.embeddings.get(node).map(|v| v.as_slice())
    }

    /// Compute cosine similarity between two nodes
    pub fn cosine_similarity(&self, a: usize, b: usize) -> Option<f64> {
        let va = self.get(a)?;
        let vb = self.get(b)?;
        Some(cosine_similarity_vecs(va, vb))
    }

    /// Get the top-k most similar nodes to a given node
    pub fn top_k_similar(&self, node: usize, k: usize) -> Vec<(usize, f64)> {
        let query = match self.get(node) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut similarities: Vec<(usize, f64)> = (0..self.num_nodes)
            .filter(|&i| i != node)
            .filter_map(|i| self.get(i).map(|v| (i, cosine_similarity_vecs(query, v))))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }
}

/// L2 normalize a vector
fn normalize_l2(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Softmax over a slice (re-exported for use in ab_test module)
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    AttentionHead::softmax(scores)
}

/// Convert to EmbeddingError
pub fn gat_err(msg: impl Into<String>) -> EmbeddingError {
    EmbeddingError::Other(anyhow!(msg.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line_graph(n: usize, feat_dim: usize, seed: u64) -> GraphData {
        let mut rng = SimpleLcg::new(seed);
        let features: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..feat_dim).map(|_| rng.next_f64()).collect())
            .collect();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n.saturating_sub(1) {
            adjacency[i].push(i + 1);
            adjacency[i + 1].push(i);
        }
        GraphData::new(features, adjacency).expect("line graph construction should succeed")
    }

    #[test]
    fn test_gat_config_default() {
        let config = GatConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_output_dim, 8);
        assert_eq!(config.output_dim(), 64); // concat: 8 * 8
    }

    #[test]
    fn test_gat_config_avg() {
        let config = GatConfig {
            concat_heads: false,
            num_heads: 4,
            head_output_dim: 16,
            ..Default::default()
        };
        assert_eq!(config.output_dim(), 16); // average: head_output_dim
    }

    #[test]
    fn test_gat_embed_shape() {
        let config = GatConfig {
            input_dim: 8,
            head_output_dim: 4,
            num_heads: 2,
            concat_heads: true,
            normalize_output: false,
            ..Default::default()
        };
        let model = Gat::new(config.clone()).expect("GAT construction should succeed");
        let graph = make_line_graph(5, 8, 100);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        assert_eq!(embeddings.num_nodes, 5);
        assert_eq!(embeddings.dim, 8); // 2 heads * 4 per head
        for i in 0..5 {
            assert_eq!(embeddings.get(i).expect("embedding should exist").len(), 8);
        }
    }

    #[test]
    fn test_gat_embed_avg_heads() {
        let config = GatConfig {
            input_dim: 8,
            head_output_dim: 4,
            num_heads: 3,
            concat_heads: false,
            normalize_output: false,
            ..Default::default()
        };
        let model = Gat::new(config.clone()).expect("GAT should construct");
        let graph = make_line_graph(4, 8, 200);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        assert_eq!(embeddings.dim, 4); // avg: head_output_dim
        for i in 0..4 {
            assert_eq!(embeddings.get(i).expect("embedding exists").len(), 4);
        }
    }

    #[test]
    fn test_gat_normalized_output() {
        let config = GatConfig {
            input_dim: 4,
            head_output_dim: 4,
            num_heads: 2,
            concat_heads: false,
            normalize_output: true,
            ..Default::default()
        };
        let model = Gat::new(config).expect("GAT should construct");
        let graph = make_line_graph(5, 4, 300);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        for i in 0..5 {
            let emb = embeddings.get(i).expect("embedding exists");
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Either 0 (all ReLU killed) or ~1
            assert!(norm <= 1.0 + 1e-6, "norm {} should be <= 1", norm);
        }
    }

    #[test]
    fn test_gat_cosine_similarity() {
        let config = GatConfig {
            input_dim: 4,
            head_output_dim: 4,
            num_heads: 1,
            concat_heads: true,
            normalize_output: false,
            ..Default::default()
        };
        let model = Gat::new(config).expect("GAT should construct");
        let graph = make_line_graph(5, 4, 400);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        // Cosine similarity should be in [-1, 1]
        for i in 0..5 {
            for j in 0..5 {
                if let Some(sim) = embeddings.cosine_similarity(i, j) {
                    assert!(
                        (-1.0 - 1e-6..=1.0 + 1e-6).contains(&sim),
                        "cosine_similarity({}, {}) = {} out of range",
                        i,
                        j,
                        sim
                    );
                }
            }
        }
    }

    #[test]
    fn test_gat_top_k_similar() {
        let config = GatConfig {
            input_dim: 4,
            head_output_dim: 4,
            num_heads: 2,
            concat_heads: true,
            normalize_output: true,
            ..Default::default()
        };
        let model = Gat::new(config).expect("GAT should construct");
        let graph = make_line_graph(6, 4, 500);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        let top3 = embeddings.top_k_similar(0, 3);
        assert!(top3.len() <= 3);
        // Results should be in descending similarity order
        for window in top3.windows(2) {
            assert!(
                window[0].1 >= window[1].1 - 1e-10,
                "top_k should be sorted descending"
            );
        }
    }

    #[test]
    fn test_attention_head_softmax() {
        // Verify softmax sums to 1
        let scores = vec![1.0, 2.0, 3.0, 0.5, -1.0];
        let weights = AttentionHead::softmax(&scores);
        assert_eq!(weights.len(), scores.len());
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax should sum to 1, got {}",
            sum
        );
        // Larger scores should have larger weights
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_gat_invalid_config() {
        assert!(Gat::new(GatConfig {
            num_heads: 0,
            ..Default::default()
        })
        .is_err());
        assert!(Gat::new(GatConfig {
            input_dim: 0,
            ..Default::default()
        })
        .is_err());
        assert!(Gat::new(GatConfig {
            head_output_dim: 0,
            ..Default::default()
        })
        .is_err());
    }

    #[test]
    fn test_gat_isolated_node() {
        // A single isolated node should still produce an embedding
        let config = GatConfig {
            input_dim: 4,
            head_output_dim: 4,
            num_heads: 2,
            concat_heads: true,
            normalize_output: false,
            ..Default::default()
        };
        let model = Gat::new(config).expect("GAT should construct");
        let features = vec![vec![1.0, 0.5, -0.5, 0.2]];
        let adjacency = vec![vec![]]; // no neighbors
        let graph = GraphData::new(features, adjacency).expect("graph should construct");
        let embeddings = model.embed(&graph).expect("should embed isolated node");
        assert_eq!(embeddings.num_nodes, 1);
        assert!(embeddings.get(0).is_some());
    }
}
