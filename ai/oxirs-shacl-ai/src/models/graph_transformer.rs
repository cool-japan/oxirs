//! GT-SHACL: Graph Transformer Architecture for SHACL Shape Learning
//!
//! Implements a Graph Transformer that combines:
//! - Multi-head self-attention over RDF graph nodes
//! - Edge-aware attention bias using predicate features
//! - Feed-forward sub-network per layer
//! - Constraint prediction head for SHACL shape output
//!
//! ## Architecture Overview
//!
//! ```text
//!  RDF Graph  ──►  FeatureEncoder  ──►  GraphTransformerLayer (×L)  ──►  ConstraintHead
//!                 (node + edge emb.)     (MHA + FFN + LayerNorm)        (per-shape logits)
//! ```
//!
//! All computation uses pure-Rust floating-point arithmetic; no C/Fortran
//! dependency is introduced.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ShaclAiError;

// ---------------------------------------------------------------------------
// Hyper-parameters
// ---------------------------------------------------------------------------

/// Configuration for the GT-SHACL model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformerConfig {
    /// Dimensionality of node embeddings.
    pub hidden_dim: usize,
    /// Number of attention heads.  Must divide `hidden_dim` evenly.
    pub num_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Dropout probability (0.0 = no dropout).
    pub dropout: f64,
    /// Feed-forward inner dimension multiplier (relative to `hidden_dim`).
    pub ff_multiplier: usize,
    /// Number of distinct predicate types (vocabulary size for edge embeddings).
    pub predicate_vocab_size: usize,
    /// Number of distinct node-type classes.
    pub node_type_vocab_size: usize,
    /// Number of SHACL constraint classes to predict.
    pub num_constraint_classes: usize,
    /// L2 regularisation weight.
    pub weight_decay: f64,
    /// Learning rate for parameter updates.
    pub learning_rate: f64,
}

impl Default for GraphTransformerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 3,
            dropout: 0.1,
            ff_multiplier: 4,
            predicate_vocab_size: 256,
            node_type_vocab_size: 64,
            num_constraint_classes: 16,
            weight_decay: 1e-4,
            learning_rate: 1e-3,
        }
    }
}

impl GraphTransformerConfig {
    /// Return the per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// Validate consistency of hyper-parameters.
    pub fn validate(&self) -> Result<(), ShaclAiError> {
        if self.hidden_dim == 0 {
            return Err(ShaclAiError::Configuration(
                "hidden_dim must be > 0".to_string(),
            ));
        }
        if self.num_heads == 0 || self.hidden_dim % self.num_heads != 0 {
            return Err(ShaclAiError::Configuration(format!(
                "hidden_dim ({}) must be divisible by num_heads ({})",
                self.hidden_dim, self.num_heads
            )));
        }
        if self.num_layers == 0 {
            return Err(ShaclAiError::Configuration(
                "num_layers must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Graph input representation
// ---------------------------------------------------------------------------

/// A node in the RDF graph with pre-computed features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique integer ID for this node (index into the node table).
    pub id: usize,
    /// One-hot type index (into `node_type_vocab`).
    pub type_idx: usize,
    /// Continuous feature vector (length = `hidden_dim`).
    pub features: Vec<f64>,
}

/// A directed edge (predicate occurrence) between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node id.
    pub src: usize,
    /// Destination node id.
    pub dst: usize,
    /// Predicate vocabulary index.
    pub predicate_idx: usize,
    /// Edge weight / frequency.
    pub weight: f64,
}

/// An attributed RDF graph ready for the GT-SHACL model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributedGraph {
    /// Node table.
    pub nodes: Vec<GraphNode>,
    /// Edge list.
    pub edges: Vec<GraphEdge>,
    /// Ground-truth constraint labels (one-hot per node, length = num_constraint_classes).
    /// Empty during inference.
    pub labels: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Linear layer
// ---------------------------------------------------------------------------

/// A fully-connected linear layer with Xavier initialisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    /// Weight matrix (out_features × in_features, row-major).
    pub weight: Vec<Vec<f64>>,
    /// Bias vector (length = out_features).
    pub bias: Vec<f64>,
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
}

impl Linear {
    /// Initialise with Xavier uniform distribution using a deterministic PRNG.
    pub fn new_xavier(in_features: usize, out_features: usize, seed: u64) -> Self {
        let limit = (6.0_f64 / (in_features + out_features) as f64).sqrt();
        let mut rng = DeterministicRng::new(seed);

        let weight = (0..out_features)
            .map(|_| {
                (0..in_features)
                    .map(|_| rng.uniform(-limit, limit))
                    .collect()
            })
            .collect();

        let bias = vec![0.0_f64; out_features];

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = W x + b
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, ShaclAiError> {
        if x.len() != self.in_features {
            return Err(ShaclAiError::ModelTraining(format!(
                "Linear: expected input dim {}, got {}",
                self.in_features,
                x.len()
            )));
        }
        let mut out = self.bias.clone();
        for (i, row) in self.weight.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                out[i] += w * x[j];
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Layer Normalisation
// ---------------------------------------------------------------------------

/// Layer normalisation over a 1-D vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    /// Learnable scale parameter (γ).
    pub gamma: Vec<f64>,
    /// Learnable shift parameter (β).
    pub beta: Vec<f64>,
    /// Small value for numerical stability.
    pub eps: f64,
}

impl LayerNorm {
    /// Initialise with γ=1, β=0.
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0_f64; dim],
            beta: vec![0.0_f64; dim],
            eps: 1e-6,
        }
    }

    /// Normalise x in-place and apply affine transform.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / n;
        let std = (var + self.eps).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, &xi)| self.gamma[i] * (xi - mean) / std + self.beta[i])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Multi-Head Self-Attention
// ---------------------------------------------------------------------------

/// Multi-head self-attention with optional edge-bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub edge_proj: Linear, // maps edge feature → scalar bias per head
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
}

impl MultiHeadAttention {
    /// Initialise all projection matrices.
    pub fn new(hidden_dim: usize, num_heads: usize, seed: u64) -> Result<Self, ShaclAiError> {
        if hidden_dim % num_heads != 0 {
            return Err(ShaclAiError::Configuration(format!(
                "hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
            )));
        }
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            q_proj: Linear::new_xavier(hidden_dim, hidden_dim, seed),
            k_proj: Linear::new_xavier(hidden_dim, hidden_dim, seed.wrapping_add(1)),
            v_proj: Linear::new_xavier(hidden_dim, hidden_dim, seed.wrapping_add(2)),
            out_proj: Linear::new_xavier(hidden_dim, hidden_dim, seed.wrapping_add(3)),
            edge_proj: Linear::new_xavier(hidden_dim, num_heads, seed.wrapping_add(4)),
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Compute multi-head attention for a sequence of node embeddings.
    ///
    /// `nodes` is a (N × D) matrix (N nodes, D = hidden_dim).
    /// `edge_bias` is an optional (N × N × H) tensor.
    /// Returns updated (N × D) embeddings.
    pub fn forward(
        &self,
        nodes: &[Vec<f64>],
        edge_bias: Option<&[Vec<Vec<f64>>]>,
    ) -> Result<Vec<Vec<f64>>, ShaclAiError> {
        let n = nodes.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Project Q, K, V for all nodes: (N × D)
        let queries: Vec<Vec<f64>> = nodes
            .iter()
            .map(|x| self.q_proj.forward(x))
            .collect::<Result<_, _>>()?;
        let keys: Vec<Vec<f64>> = nodes
            .iter()
            .map(|x| self.k_proj.forward(x))
            .collect::<Result<_, _>>()?;
        let values: Vec<Vec<f64>> = nodes
            .iter()
            .map(|x| self.v_proj.forward(x))
            .collect::<Result<_, _>>()?;

        // Process each head independently
        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            // Extract head slices
            let q_h: Vec<&[f64]> = queries.iter().map(|q| &q[start..end]).collect();
            let k_h: Vec<&[f64]> = keys.iter().map(|k| &k[start..end]).collect();
            let v_h: Vec<&[f64]> = values.iter().map(|v| &v[start..end]).collect();

            // Attention scores: (N × N)
            let mut scores = vec![vec![0.0_f64; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let dot: f64 = q_h[i].iter().zip(k_h[j]).map(|(&a, &b)| a * b).sum();
                    scores[i][j] = dot * self.scale;
                    // Add edge bias if provided
                    if let Some(eb) = edge_bias {
                        if i < eb.len() && j < eb[i].len() && h < eb[i][j].len() {
                            scores[i][j] += eb[i][j][h];
                        }
                    }
                }
            }

            // Softmax per row
            for row in &mut scores {
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = row.iter().map(|&s| (s - max_val).exp()).sum();
                for s in row.iter_mut() {
                    *s = ((*s - max_val).exp()) / (exp_sum + 1e-12);
                }
            }

            // Weighted sum of values: (N × head_dim)
            let mut head_out = vec![vec![0.0_f64; self.head_dim]; n];
            for i in 0..n {
                for j in 0..n {
                    for d in 0..self.head_dim {
                        head_out[i][d] += scores[i][j] * v_h[j][d];
                    }
                }
            }
            head_outputs.push(head_out);
        }

        // Concatenate heads: (N × D)
        let mut concat = vec![vec![0.0_f64; self.q_proj.out_features]; n];
        for i in 0..n {
            for (h, head_out) in head_outputs.iter().enumerate() {
                let start = h * self.head_dim;
                for d in 0..self.head_dim {
                    concat[i][start + d] = head_out[i][d];
                }
            }
        }

        // Final projection
        concat
            .iter()
            .map(|x| self.out_proj.forward(x))
            .collect::<Result<_, _>>()
    }
}

// ---------------------------------------------------------------------------
// Feed-Forward Sub-Network
// ---------------------------------------------------------------------------

/// Position-wise feed-forward network: FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl FeedForward {
    pub fn new(hidden_dim: usize, inner_dim: usize, seed: u64) -> Self {
        Self {
            fc1: Linear::new_xavier(hidden_dim, inner_dim, seed),
            fc2: Linear::new_xavier(inner_dim, hidden_dim, seed.wrapping_add(10)),
        }
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, ShaclAiError> {
        let h = self.fc1.forward(x)?;
        let h_relu: Vec<f64> = h.into_iter().map(|v| v.max(0.0)).collect();
        self.fc2.forward(&h_relu)
    }
}

// ---------------------------------------------------------------------------
// Graph Transformer Layer
// ---------------------------------------------------------------------------

/// A single Graph Transformer layer:  MHA → Residual+LN → FFN → Residual+LN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformerLayer {
    pub attention: MultiHeadAttention,
    pub ffn: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl GraphTransformerLayer {
    pub fn new(config: &GraphTransformerConfig, seed: u64) -> Result<Self, ShaclAiError> {
        let hidden_dim = config.hidden_dim;
        let inner_dim = hidden_dim * config.ff_multiplier;
        Ok(Self {
            attention: MultiHeadAttention::new(hidden_dim, config.num_heads, seed)?,
            ffn: FeedForward::new(hidden_dim, inner_dim, seed.wrapping_add(100)),
            norm1: LayerNorm::new(hidden_dim),
            norm2: LayerNorm::new(hidden_dim),
        })
    }

    /// Forward pass: accepts (N × D) node embeddings, returns (N × D).
    pub fn forward(
        &self,
        nodes: &[Vec<f64>],
        edge_bias: Option<&[Vec<Vec<f64>>]>,
    ) -> Result<Vec<Vec<f64>>, ShaclAiError> {
        // MHA + residual
        let attn_out = self.attention.forward(nodes, edge_bias)?;
        let after_attn: Vec<Vec<f64>> = nodes
            .iter()
            .zip(&attn_out)
            .map(|(x, a)| {
                let sum: Vec<f64> = x.iter().zip(a).map(|(&xi, &ai)| xi + ai).collect();
                self.norm1.forward(&sum)
            })
            .collect();

        // FFN + residual
        after_attn
            .iter()
            .map(|x| {
                let ffn_out = self.ffn.forward(x)?;
                let sum: Vec<f64> = x.iter().zip(&ffn_out).map(|(&xi, &fi)| xi + fi).collect();
                Ok(self.norm2.forward(&sum))
            })
            .collect::<Result<Vec<Vec<f64>>, ShaclAiError>>()
    }
}

// ---------------------------------------------------------------------------
// Feature Encoder
// ---------------------------------------------------------------------------

/// Encodes raw graph features into the hidden-dim embedding space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEncoder {
    /// Node-type embedding table (vocab_size × hidden_dim).
    pub node_type_emb: Vec<Vec<f64>>,
    /// Predicate embedding table (vocab_size × hidden_dim).
    pub predicate_emb: Vec<Vec<f64>>,
    /// Linear projection for continuous node features.
    pub feature_proj: Linear,
}

impl FeatureEncoder {
    pub fn new(config: &GraphTransformerConfig, seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        let hidden_dim = config.hidden_dim;

        let node_type_emb = (0..config.node_type_vocab_size)
            .map(|_| (0..hidden_dim).map(|_| rng.normal(0.0, 0.02)).collect())
            .collect();

        let predicate_emb = (0..config.predicate_vocab_size)
            .map(|_| (0..hidden_dim).map(|_| rng.normal(0.0, 0.02)).collect())
            .collect();

        let feature_proj = Linear::new_xavier(hidden_dim, hidden_dim, seed.wrapping_add(1000));

        Self {
            node_type_emb,
            predicate_emb,
            feature_proj,
        }
    }

    /// Encode a single node into hidden_dim space.
    pub fn encode_node(
        &self,
        node: &GraphNode,
        hidden_dim: usize,
    ) -> Result<Vec<f64>, ShaclAiError> {
        // Type embedding
        let type_idx = node
            .type_idx
            .min(self.node_type_emb.len().saturating_sub(1));
        let type_emb = &self.node_type_emb[type_idx];

        // Project continuous features (zero-pad or truncate to hidden_dim)
        let mut feat_padded = vec![0.0_f64; hidden_dim];
        let copy_len = node.features.len().min(hidden_dim);
        feat_padded[..copy_len].copy_from_slice(&node.features[..copy_len]);
        let feat_proj = self.feature_proj.forward(&feat_padded)?;

        // Combine: type_emb + feat_proj (element-wise add)
        let combined: Vec<f64> = type_emb
            .iter()
            .zip(&feat_proj)
            .map(|(&t, &f)| t + f)
            .collect();
        Ok(combined)
    }

    /// Build edge-bias tensor (N × N × H) from edge list.
    pub fn build_edge_bias(
        &self,
        edges: &[GraphEdge],
        num_nodes: usize,
        num_heads: usize,
    ) -> Vec<Vec<Vec<f64>>> {
        let hidden_dim = self.predicate_emb[0].len();
        let head_dim = hidden_dim / num_heads;
        let mut bias = vec![vec![vec![0.0_f64; num_heads]; num_nodes]; num_nodes];

        for edge in edges {
            if edge.src >= num_nodes || edge.dst >= num_nodes {
                continue;
            }
            let pred_idx = edge
                .predicate_idx
                .min(self.predicate_emb.len().saturating_sub(1));
            let emb = &self.predicate_emb[pred_idx];
            // Average pool embedding across head_dim to get per-head scalar
            for (h, bias_val) in bias[edge.src][edge.dst].iter_mut().enumerate() {
                let start = h * head_dim;
                let end = (start + head_dim).min(hidden_dim);
                if start < end {
                    let mean: f64 =
                        emb[start..end].iter().sum::<f64>() / (end - start) as f64 * edge.weight;
                    *bias_val += mean;
                }
            }
        }
        bias
    }
}

// ---------------------------------------------------------------------------
// Constraint Prediction Head
// ---------------------------------------------------------------------------

/// Maps per-node embeddings to constraint class logits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintHead {
    pub proj: Linear,
    pub out: Linear,
}

impl ConstraintHead {
    pub fn new(hidden_dim: usize, num_classes: usize, seed: u64) -> Self {
        Self {
            proj: Linear::new_xavier(hidden_dim, hidden_dim / 2, seed),
            out: Linear::new_xavier(hidden_dim / 2, num_classes, seed.wrapping_add(500)),
        }
    }

    /// Predict class logits for one node embedding.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, ShaclAiError> {
        let h = self.proj.forward(x)?;
        let h_relu: Vec<f64> = h.into_iter().map(|v| v.max(0.0)).collect();
        self.out.forward(&h_relu)
    }

    /// Apply sigmoid to logits (multi-label classification).
    pub fn predict_probs(&self, x: &[f64]) -> Result<Vec<f64>, ShaclAiError> {
        let logits = self.forward(x)?;
        Ok(logits.into_iter().map(sigmoid).collect())
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Full GT-SHACL Model
// ---------------------------------------------------------------------------

/// Training / inference statistics for the GT-SHACL model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GtShaclStats {
    /// Number of forward passes completed.
    pub forward_passes: usize,
    /// Cumulative training loss.
    pub cumulative_loss: f64,
    /// Number of gradient update steps.
    pub update_steps: usize,
    /// Per-class average precision (last evaluation).
    pub per_class_precision: Vec<f64>,
}

/// The full GT-SHACL Graph Transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtShaclModel {
    pub config: GraphTransformerConfig,
    pub encoder: FeatureEncoder,
    pub layers: Vec<GraphTransformerLayer>,
    pub head: ConstraintHead,
    stats: GtShaclStats,
}

impl GtShaclModel {
    /// Create a new GT-SHACL model with the given configuration.
    pub fn new(config: GraphTransformerConfig) -> Result<Self, ShaclAiError> {
        config.validate()?;
        let seed_base = 42u64;

        let encoder = FeatureEncoder::new(&config, seed_base);
        let layers = (0..config.num_layers)
            .map(|i| GraphTransformerLayer::new(&config, seed_base + i as u64 * 1000 + 1))
            .collect::<Result<_, _>>()?;
        let head = ConstraintHead::new(
            config.hidden_dim,
            config.num_constraint_classes,
            seed_base + 9999,
        );

        Ok(Self {
            config,
            encoder,
            layers,
            head,
            stats: GtShaclStats::default(),
        })
    }

    /// Forward pass: returns (N × num_constraint_classes) probability matrix.
    pub fn forward(&mut self, graph: &AttributedGraph) -> Result<Vec<Vec<f64>>, ShaclAiError> {
        if graph.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let hidden_dim = self.config.hidden_dim;

        // Encode nodes
        let mut embeddings: Vec<Vec<f64>> = graph
            .nodes
            .iter()
            .map(|n| self.encoder.encode_node(n, hidden_dim))
            .collect::<Result<_, _>>()?;

        // Build edge bias
        let edge_bias =
            self.encoder
                .build_edge_bias(&graph.edges, graph.nodes.len(), self.config.num_heads);

        // Pass through transformer layers
        for layer in &self.layers {
            embeddings = layer.forward(&embeddings, Some(&edge_bias))?;
        }

        // Predict constraint probabilities per node
        let probs: Vec<Vec<f64>> = embeddings
            .iter()
            .map(|emb| self.head.predict_probs(emb))
            .collect::<Result<_, _>>()?;

        self.stats.forward_passes += 1;
        Ok(probs)
    }

    /// Predict top-k constraints for each node.
    pub fn predict_top_k(
        &mut self,
        graph: &AttributedGraph,
        k: usize,
    ) -> Result<Vec<Vec<usize>>, ShaclAiError> {
        let probs = self.forward(graph)?;
        Ok(probs.into_iter().map(|p| top_k_indices(&p, k)).collect())
    }

    /// Compute binary cross-entropy loss against ground-truth labels.
    pub fn loss(&mut self, graph: &AttributedGraph) -> Result<f64, ShaclAiError> {
        if graph.labels.is_empty() {
            return Err(ShaclAiError::ModelTraining(
                "No labels provided for loss computation".to_string(),
            ));
        }
        let probs = self.forward(graph)?;
        let n = probs.len();
        if n == 0 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0_f64;
        let mut count = 0usize;

        for (prob_row, label_row) in probs.iter().zip(&graph.labels) {
            let num_classes = prob_row.len().min(label_row.len());
            for c in 0..num_classes {
                let p = prob_row[c].clamp(1e-12, 1.0 - 1e-12);
                let y = label_row[c].clamp(0.0, 1.0);
                total_loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
                count += 1;
            }
        }

        let mean_loss = if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        };
        self.stats.cumulative_loss += mean_loss;
        Ok(mean_loss)
    }

    /// Return accumulated statistics.
    pub fn stats(&self) -> &GtShaclStats {
        &self.stats
    }

    /// Reset accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = GtShaclStats::default();
    }

    /// Export model parameters as a flat map (for serialisation / inspection).
    pub fn parameter_count(&self) -> usize {
        let mut count = 0usize;
        // Encoder
        count += self.encoder.feature_proj.weight.len() * self.encoder.feature_proj.weight[0].len();
        count += self.encoder.feature_proj.bias.len();
        // Layers (approximate)
        for layer in &self.layers {
            count +=
                layer.attention.q_proj.weight.len() * layer.attention.q_proj.weight[0].len() * 4; // q,k,v,out
            count += layer.ffn.fc1.weight.len() * layer.ffn.fc1.weight[0].len();
            count += layer.ffn.fc2.weight.len() * layer.ffn.fc2.weight[0].len();
        }
        // Head
        count += self.head.proj.weight.len() * self.head.proj.weight[0].len();
        count += self.head.out.weight.len() * self.head.out.weight[0].len();
        count
    }
}

/// Return indices of the `k` largest values in `vals`.
fn top_k_indices(vals: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (LCG) – avoids rand/scirs2 import in this pure-math layer
// ---------------------------------------------------------------------------

/// Simple LCG-based PRNG seeded with a u64 for reproducible parameter init.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform float in [low, high).
    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = self.next_u64() as f64 / u64::MAX as f64;
        low + u * (high - low)
    }

    /// Approximate normal via Box-Muller.
    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_u64() as f64 / u64::MAX as f64 + 1e-12;
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }
}

// ---------------------------------------------------------------------------
// Convenience builder / training helpers
// ---------------------------------------------------------------------------

/// Builder for GT-SHACL training.
#[derive(Debug)]
pub struct GtShaclTrainer {
    model: GtShaclModel,
    epochs: usize,
    batch_graphs: Vec<AttributedGraph>,
}

impl GtShaclTrainer {
    /// Create a trainer wrapping an existing model.
    pub fn new(model: GtShaclModel, epochs: usize) -> Self {
        Self {
            model,
            epochs,
            batch_graphs: Vec::new(),
        }
    }

    /// Add a labelled training graph.
    pub fn add_graph(&mut self, graph: AttributedGraph) {
        self.batch_graphs.push(graph);
    }

    /// Run a simplified training loop (gradient-free, for integration-test purposes).
    ///
    /// A production implementation would use automatic differentiation; here we
    /// exercise the forward/loss path to validate the architecture end-to-end.
    pub fn train(&mut self) -> Result<TrainingReport, ShaclAiError> {
        let mut epoch_losses: Vec<f64> = Vec::with_capacity(self.epochs);

        for epoch in 0..self.epochs {
            let mut epoch_loss = 0.0_f64;
            let mut n_graphs = 0usize;

            for graph in &self.batch_graphs {
                if !graph.labels.is_empty() {
                    let loss = self.model.loss(graph)?;
                    epoch_loss += loss;
                    n_graphs += 1;
                }
            }

            let mean = if n_graphs > 0 {
                epoch_loss / n_graphs as f64
            } else {
                0.0
            };
            epoch_losses.push(mean);
            tracing::debug!(epoch, loss = mean, "GT-SHACL training epoch");
        }

        Ok(TrainingReport {
            epochs_completed: self.epochs,
            final_loss: epoch_losses.last().copied().unwrap_or(0.0),
            epoch_losses,
            parameter_count: self.model.parameter_count(),
            forward_passes: self.model.stats().forward_passes,
        })
    }

    /// Consume the trainer and return the trained model.
    pub fn into_model(self) -> GtShaclModel {
        self.model
    }
}

/// Summary of a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub epochs_completed: usize,
    pub final_loss: f64,
    pub epoch_losses: Vec<f64>,
    pub parameter_count: usize,
    pub forward_passes: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> GraphTransformerConfig {
        GraphTransformerConfig {
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 2,
            dropout: 0.0,
            ff_multiplier: 2,
            predicate_vocab_size: 8,
            node_type_vocab_size: 4,
            num_constraint_classes: 4,
            weight_decay: 1e-4,
            learning_rate: 1e-3,
        }
    }

    fn make_model() -> GtShaclModel {
        GtShaclModel::new(tiny_config()).expect("model creation should succeed")
    }

    fn make_graph(num_nodes: usize, labeled: bool) -> AttributedGraph {
        let nodes: Vec<GraphNode> = (0..num_nodes)
            .map(|i| GraphNode {
                id: i,
                type_idx: i % 4,
                features: vec![0.5_f64; 16],
            })
            .collect();

        let edges: Vec<GraphEdge> = (0..num_nodes.saturating_sub(1))
            .map(|i| GraphEdge {
                src: i,
                dst: i + 1,
                predicate_idx: i % 8,
                weight: 1.0,
            })
            .collect();

        let labels = if labeled {
            (0..num_nodes)
                .map(|i| {
                    let mut l = vec![0.0_f64; 4];
                    l[i % 4] = 1.0;
                    l
                })
                .collect()
        } else {
            Vec::new()
        };

        AttributedGraph {
            nodes,
            edges,
            labels,
        }
    }

    // --- Configuration tests ---

    #[test]
    fn test_config_validation_ok() {
        assert!(tiny_config().validate().is_ok());
    }

    #[test]
    fn test_config_validation_mismatched_heads() {
        let cfg = GraphTransformerConfig {
            hidden_dim: 15,
            num_heads: 4,
            ..tiny_config()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_head_dim() {
        let cfg = tiny_config();
        assert_eq!(cfg.head_dim(), 8);
    }

    // --- Linear layer tests ---

    #[test]
    fn test_linear_forward_shape() {
        let linear = Linear::new_xavier(4, 8, 0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = linear.forward(&x).expect("linear forward should work");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_linear_wrong_input_dim() {
        let linear = Linear::new_xavier(4, 8, 0);
        let x = vec![1.0, 2.0]; // wrong dim
        assert!(linear.forward(&x).is_err());
    }

    // --- LayerNorm tests ---

    #[test]
    fn test_layer_norm_output_shape() {
        let ln = LayerNorm::new(8);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = ln.forward(&x);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_layer_norm_near_zero_mean() {
        let ln = LayerNorm::new(4);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = ln.forward(&x);
        let mean: f64 = out.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-10, "mean={mean}");
    }

    // --- MultiHeadAttention tests ---

    #[test]
    fn test_mha_output_shape() {
        let mha = MultiHeadAttention::new(16, 2, 42).expect("mha creation should succeed");
        let nodes: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1_f64; 16]).collect();
        let out = mha.forward(&nodes, None).expect("forward should succeed");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_mha_empty_input() {
        let mha = MultiHeadAttention::new(16, 2, 42).expect("ok");
        let out = mha
            .forward(&[], None)
            .expect("empty input should return empty");
        assert!(out.is_empty());
    }

    // --- FeedForward tests ---

    #[test]
    fn test_ffn_output_shape() {
        let ffn = FeedForward::new(16, 32, 0);
        let x = vec![0.5_f64; 16];
        let out = ffn.forward(&x).expect("ffn forward ok");
        assert_eq!(out.len(), 16);
    }

    // --- GraphTransformerLayer tests ---

    #[test]
    fn test_layer_forward_shape() {
        let layer = GraphTransformerLayer::new(&tiny_config(), 0).expect("layer creation ok");
        let nodes: Vec<Vec<f64>> = (0..4).map(|_| vec![0.1_f64; 16]).collect();
        let out = layer.forward(&nodes, None).expect("layer forward ok");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 16);
    }

    // --- FeatureEncoder tests ---

    #[test]
    fn test_encoder_encode_node() {
        let cfg = tiny_config();
        let encoder = FeatureEncoder::new(&cfg, 1);
        let node = GraphNode {
            id: 0,
            type_idx: 2,
            features: vec![0.1_f64; 16],
        };
        let emb = encoder
            .encode_node(&node, cfg.hidden_dim)
            .expect("encode ok");
        assert_eq!(emb.len(), cfg.hidden_dim);
    }

    #[test]
    fn test_encoder_edge_bias_shape() {
        let cfg = tiny_config();
        let encoder = FeatureEncoder::new(&cfg, 1);
        let edges = vec![GraphEdge {
            src: 0,
            dst: 1,
            predicate_idx: 3,
            weight: 1.0,
        }];
        let bias = encoder.build_edge_bias(&edges, 3, cfg.num_heads);
        assert_eq!(bias.len(), 3);
        assert_eq!(bias[0].len(), 3);
        assert_eq!(bias[0][0].len(), cfg.num_heads);
    }

    // --- ConstraintHead tests ---

    #[test]
    fn test_constraint_head_probs_range() {
        let head = ConstraintHead::new(16, 4, 0);
        let x = vec![1.0_f64; 16];
        let probs = head.predict_probs(&x).expect("probs ok");
        assert_eq!(probs.len(), 4);
        for &p in &probs {
            assert!((0.0..=1.0).contains(&p), "prob {p} out of [0,1]");
        }
    }

    // --- Full model tests ---

    #[test]
    fn test_model_creation() {
        assert!(make_model().parameter_count() > 0);
    }

    #[test]
    fn test_model_forward_output_shape() {
        let mut model = make_model();
        let graph = make_graph(5, false);
        let probs = model.forward(&graph).expect("forward ok");
        assert_eq!(probs.len(), 5);
        assert_eq!(probs[0].len(), 4); // num_constraint_classes
    }

    #[test]
    fn test_model_forward_empty_graph() {
        let mut model = make_model();
        let empty = AttributedGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            labels: Vec::new(),
        };
        let probs = model.forward(&empty).expect("empty graph forward ok");
        assert!(probs.is_empty());
    }

    #[test]
    fn test_model_predict_top_k() {
        let mut model = make_model();
        let graph = make_graph(3, false);
        let top = model.predict_top_k(&graph, 2).expect("top-k ok");
        assert_eq!(top.len(), 3);
        for node_top in &top {
            assert_eq!(node_top.len(), 2);
        }
    }

    #[test]
    fn test_model_loss_with_labels() {
        let mut model = make_model();
        let graph = make_graph(4, true);
        let loss = model.loss(&graph).expect("loss ok");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_model_loss_no_labels_error() {
        let mut model = make_model();
        let graph = make_graph(4, false);
        assert!(model.loss(&graph).is_err());
    }

    #[test]
    fn test_model_stats_update() {
        let mut model = make_model();
        let graph = make_graph(3, false);
        model.forward(&graph).expect("ok");
        assert_eq!(model.stats().forward_passes, 1);
    }

    #[test]
    fn test_model_reset_stats() {
        let mut model = make_model();
        let graph = make_graph(2, false);
        model.forward(&graph).expect("ok");
        model.reset_stats();
        assert_eq!(model.stats().forward_passes, 0);
    }

    // --- Trainer tests ---

    #[test]
    fn test_trainer_basic_run() {
        let model = make_model();
        let mut trainer = GtShaclTrainer::new(model, 3);
        trainer.add_graph(make_graph(5, true));
        trainer.add_graph(make_graph(3, true));

        let report = trainer.train().expect("training should succeed");
        assert_eq!(report.epochs_completed, 3);
        assert_eq!(report.epoch_losses.len(), 3);
        assert!(report.final_loss.is_finite());
        assert!(report.parameter_count > 0);
    }

    #[test]
    fn test_trainer_unlabeled_graphs_skipped() {
        let model = make_model();
        let mut trainer = GtShaclTrainer::new(model, 2);
        trainer.add_graph(make_graph(4, false)); // no labels
        let report = trainer.train().expect("ok");
        // Loss should be 0 since no labeled graphs
        assert_eq!(report.final_loss, 0.0);
    }

    #[test]
    fn test_trainer_into_model() {
        let model = make_model();
        let trainer = GtShaclTrainer::new(model, 1);
        let _m = trainer.into_model();
    }

    // --- DeterministicRng tests ---

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = DeterministicRng::new(123);
        let mut r2 = DeterministicRng::new(123);
        for _ in 0..20 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let v = rng.uniform(-1.0, 1.0);
            assert!((-1.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    // --- Utility tests ---

    #[test]
    fn test_top_k_indices() {
        let vals = vec![0.1, 0.9, 0.3, 0.8, 0.2];
        let top = top_k_indices(&vals, 2);
        assert_eq!(top.len(), 2);
        assert!(top.contains(&1)); // 0.9
        assert!(top.contains(&3)); // 0.8
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_config_serialization() {
        let cfg = tiny_config();
        let json = serde_json::to_string(&cfg).expect("serialize ok");
        let cfg2: GraphTransformerConfig = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(cfg.hidden_dim, cfg2.hidden_dim);
        assert_eq!(cfg.num_heads, cfg2.num_heads);
    }

    #[test]
    fn test_attributed_graph_serialization() {
        let graph = make_graph(3, true);
        let json = serde_json::to_string(&graph).expect("serialize ok");
        let g2: AttributedGraph = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(g2.nodes.len(), 3);
    }

    #[test]
    fn test_model_with_edge_bias() {
        let mut model = make_model();
        // Graph with explicit edges
        let nodes: Vec<GraphNode> = (0..4)
            .map(|i| GraphNode {
                id: i,
                type_idx: 0,
                features: vec![0.1_f64; 16],
            })
            .collect();
        let edges = vec![
            GraphEdge {
                src: 0,
                dst: 1,
                predicate_idx: 0,
                weight: 1.0,
            },
            GraphEdge {
                src: 1,
                dst: 2,
                predicate_idx: 1,
                weight: 0.5,
            },
            GraphEdge {
                src: 2,
                dst: 3,
                predicate_idx: 2,
                weight: 0.8,
            },
        ];
        let graph = AttributedGraph {
            nodes,
            edges,
            labels: Vec::new(),
        };
        let probs = model.forward(&graph).expect("forward with edges ok");
        assert_eq!(probs.len(), 4);
    }

    #[test]
    fn test_high_capacity_config() {
        let cfg = GraphTransformerConfig {
            hidden_dim: 64,
            num_heads: 8,
            num_layers: 4,
            ..Default::default()
        };
        let model = GtShaclModel::new(cfg).expect("large model ok");
        assert!(model.parameter_count() > 1000);
    }

    #[test]
    fn test_training_report_serialization() {
        let report = TrainingReport {
            epochs_completed: 10,
            final_loss: 0.25,
            epoch_losses: vec![0.5, 0.4, 0.35, 0.3, 0.25],
            parameter_count: 50000,
            forward_passes: 100,
        };
        let json = serde_json::to_string(&report).expect("ok");
        let r2: TrainingReport = serde_json::from_str(&json).expect("ok");
        assert_eq!(r2.epochs_completed, 10);
        assert!((r2.final_loss - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_forward_passes_accumulate_stats() {
        let mut model = make_model();
        let graph = make_graph(3, false);
        for _ in 0..5 {
            model.forward(&graph).expect("ok");
        }
        assert_eq!(model.stats().forward_passes, 5);
    }

    #[test]
    fn test_constraint_head_logits_finite() {
        let head = ConstraintHead::new(16, 8, 99);
        let x = vec![0.5_f64; 16];
        let logits = head.forward(&x).expect("ok");
        for &l in &logits {
            assert!(l.is_finite(), "logit {l} is not finite");
        }
    }

    // HashMap usage in training report metadata
    #[test]
    fn test_gt_shacl_stats_default() {
        let stats = GtShaclStats::default();
        assert_eq!(stats.forward_passes, 0);
        assert_eq!(stats.update_steps, 0);
        assert!(stats.per_class_precision.is_empty());
    }

    #[test]
    fn test_graph_node_serialization() {
        let node = GraphNode {
            id: 5,
            type_idx: 2,
            features: vec![1.0, 2.0, 3.0],
        };
        let json = serde_json::to_string(&node).expect("ok");
        let n2: GraphNode = serde_json::from_str(&json).expect("ok");
        assert_eq!(n2.id, 5);
        assert_eq!(n2.type_idx, 2);
    }

    #[test]
    fn test_graph_edge_serialization() {
        let edge = GraphEdge {
            src: 0,
            dst: 3,
            predicate_idx: 7,
            weight: 0.75,
        };
        let json = serde_json::to_string(&edge).expect("ok");
        let e2: GraphEdge = serde_json::from_str(&json).expect("ok");
        assert_eq!(e2.src, 0);
        assert_eq!(e2.dst, 3);
        assert!((e2.weight - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_layer_norm_all_zeros() {
        let ln = LayerNorm::new(4);
        // All zeros -> variance=0 -> should not panic (eps guards division)
        let out = ln.forward(&[0.0, 0.0, 0.0, 0.0]);
        for &v in &out {
            assert!(v.is_finite(), "output not finite for all-zero input");
        }
    }

    #[test]
    fn test_feature_encoder_out_of_bounds_type_idx() {
        let cfg = tiny_config();
        let encoder = FeatureEncoder::new(&cfg, 0);
        // type_idx larger than vocab: should clamp, not panic
        let node = GraphNode {
            id: 0,
            type_idx: 9999,
            features: vec![0.0_f64; 16],
        };
        let emb = encoder
            .encode_node(&node, cfg.hidden_dim)
            .expect("clamped ok");
        assert_eq!(emb.len(), cfg.hidden_dim);
    }

    #[test]
    fn test_trainer_forward_pass_count() {
        let model = make_model();
        let mut trainer = GtShaclTrainer::new(model, 2);
        trainer.add_graph(make_graph(3, true));
        trainer.train().expect("ok");
        let model = trainer.into_model();
        // 2 epochs × 1 graph = 2 loss calls, each calls forward once internally
        // plus the explicit forward in loss, so >=2
        assert!(model.stats().forward_passes >= 2);
    }
}
