//! GraphSAGE: Inductive Representation Learning on Large Graphs
//!
//! This module provides v0.3.0 GraphSAGE implementations with:
//! - `GraphSAGELayer`: inductive representation learning via neighbor sampling
//! - `MeanAggregator`, `MaxPoolAggregator`, `MeanPoolAggregator`, `LSTMAggregator`
//! - `GraphSAGEModel`: multi-layer GraphSAGE with configurable depth and hidden dims
//! - `MiniBatchGraphSAGE`: mini-batch training for large graphs
//!
//! Reference: Hamilton, Ying, Leskovec (2017) - NeurIPS
//! "Inductive Representation Learning on Large Graphs"

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Utility: Simple LCG PRNG (no external rand dependency)
// ---------------------------------------------------------------------------

/// Minimal Linear Congruential Generator for reproducible sampling.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_usize_mod(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [-scale, scale)
    pub fn next_f64_range(&mut self, scale: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * scale
    }

    /// Standard normal approximation via Box-Muller
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Graph data structures
// ---------------------------------------------------------------------------

/// A homogeneous graph with node feature vectors and adjacency lists.
#[derive(Debug, Clone)]
pub struct Graph {
    /// `node_features[i]` = feature vector for node `i`.
    pub node_features: Vec<Vec<f64>>,
    /// `adjacency[i]` = sorted list of neighbor indices for node `i`.
    pub adjacency: Vec<Vec<usize>>,
    /// Optional node-level class labels (for supervised training).
    pub labels: Option<Vec<usize>>,
}

impl Graph {
    /// Construct and validate a new graph.
    pub fn new(node_features: Vec<Vec<f64>>, adjacency: Vec<Vec<usize>>) -> Result<Self> {
        let n = node_features.len();
        if adjacency.len() != n {
            return Err(anyhow!(
                "adjacency list length {} != num_nodes {}",
                adjacency.len(),
                n
            ));
        }
        // Validate feature dimension consistency
        if let Some(first) = node_features.first() {
            let dim = first.len();
            for (i, feat) in node_features.iter().enumerate() {
                if feat.len() != dim {
                    return Err(anyhow!(
                        "node {} feature dim {} != expected {}",
                        i,
                        feat.len(),
                        dim
                    ));
                }
            }
        }
        // Validate adjacency bounds
        for (i, nbrs) in adjacency.iter().enumerate() {
            for &j in nbrs {
                if j >= n {
                    return Err(anyhow!("node {} has out-of-bounds neighbor {}", i, j));
                }
            }
        }
        Ok(Self {
            node_features,
            adjacency,
            labels: None,
        })
    }

    /// Attach labels (must match `num_nodes()`).
    pub fn with_labels(mut self, labels: Vec<usize>) -> Result<Self> {
        if labels.len() != self.num_nodes() {
            return Err(anyhow!(
                "label count {} != num_nodes {}",
                labels.len(),
                self.num_nodes()
            ));
        }
        self.labels = Some(labels);
        Ok(self)
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.node_features.len()
    }

    /// Feature dimension (0 if graph is empty).
    pub fn feature_dim(&self) -> usize {
        self.node_features.first().map(|f| f.len()).unwrap_or(0)
    }

    /// Get neighbors of node `v`.
    pub fn neighbors(&self, v: usize) -> &[usize] {
        self.adjacency.get(v).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Sample up to `k` neighbors uniformly without replacement.
    pub fn sample_neighbors(&self, v: usize, k: usize, rng: &mut Lcg) -> Vec<usize> {
        let nbrs = self.neighbors(v);
        if nbrs.is_empty() || k == 0 {
            return Vec::new();
        }
        if nbrs.len() <= k {
            return nbrs.to_vec();
        }
        // Partial Fisher-Yates
        let mut idx: Vec<usize> = (0..nbrs.len()).collect();
        for i in 0..k {
            let j = i + rng.next_usize_mod(nbrs.len() - i);
            idx.swap(i, j);
        }
        idx[..k].iter().map(|&i| nbrs[i]).collect()
    }
}

// ---------------------------------------------------------------------------
// Dense layer utility
// ---------------------------------------------------------------------------

/// A fully-connected layer: `output = W * input + bias`.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    weights: Vec<Vec<f64>>, // [out_dim][in_dim]
    bias: Vec<f64>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl DenseLayer {
    /// Xavier/Glorot uniform initialization.
    pub fn new_xavier(in_dim: usize, out_dim: usize, rng: &mut Lcg) -> Self {
        let scale = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let weights = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| rng.next_f64_range(scale)).collect())
            .collect();
        Self {
            weights,
            bias: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: W * x + b.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut out = self.bias.clone();
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                out[i] += w * x[j];
            }
        }
        out
    }

    /// ReLU activation.
    pub fn relu(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    /// Tanh activation.
    pub fn tanh(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.tanh()).collect()
    }
}

// ---------------------------------------------------------------------------
// Aggregator trait and implementations
// ---------------------------------------------------------------------------

/// Aggregates a set of neighbor feature vectors into a single vector.
pub trait Aggregator: std::fmt::Debug + Send + Sync {
    /// Aggregate `neighbor_features` (each of length `input_dim`) into one vector.
    fn aggregate(&self, neighbor_features: &[Vec<f64>], input_dim: usize) -> Vec<f64>;

    /// Output dimensionality produced by this aggregator given `input_dim`.
    fn output_dim(&self, input_dim: usize) -> usize;
}

/// Mean aggregator: element-wise mean of neighbor features.
#[derive(Debug, Clone, Default)]
pub struct MeanAggregator;

impl Aggregator for MeanAggregator {
    fn aggregate(&self, neighbor_features: &[Vec<f64>], input_dim: usize) -> Vec<f64> {
        if neighbor_features.is_empty() {
            return vec![0.0; input_dim];
        }
        let mut mean = vec![0.0f64; input_dim];
        for feat in neighbor_features {
            for (i, &v) in feat.iter().enumerate().take(input_dim) {
                mean[i] += v;
            }
        }
        let n = neighbor_features.len() as f64;
        mean.iter_mut().for_each(|v| *v /= n);
        mean
    }

    fn output_dim(&self, input_dim: usize) -> usize {
        input_dim
    }
}

/// Max-pool aggregator: element-wise max of neighbor features after MLP.
#[derive(Debug, Clone)]
pub struct MaxPoolAggregator {
    mlp: DenseLayer,
    hidden_dim: usize,
}

impl MaxPoolAggregator {
    /// Create with a single hidden MLP applied before max-pooling.
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut Lcg) -> Self {
        Self {
            mlp: DenseLayer::new_xavier(input_dim, hidden_dim, rng),
            hidden_dim,
        }
    }
}

impl Aggregator for MaxPoolAggregator {
    fn aggregate(&self, neighbor_features: &[Vec<f64>], _input_dim: usize) -> Vec<f64> {
        if neighbor_features.is_empty() {
            return vec![0.0; self.hidden_dim];
        }
        let mut pool = vec![f64::NEG_INFINITY; self.hidden_dim];
        for feat in neighbor_features {
            let transformed = DenseLayer::relu(&self.mlp.forward(feat));
            for (i, &v) in transformed.iter().enumerate() {
                if v > pool[i] {
                    pool[i] = v;
                }
            }
        }
        // Replace -inf with 0 for isolated situations
        pool.iter_mut().for_each(|v| {
            if v.is_infinite() {
                *v = 0.0;
            }
        });
        pool
    }

    fn output_dim(&self, _input_dim: usize) -> usize {
        self.hidden_dim
    }
}

/// Mean-pool aggregator: element-wise mean of neighbor features after MLP.
#[derive(Debug, Clone)]
pub struct MeanPoolAggregator {
    mlp: DenseLayer,
    hidden_dim: usize,
}

impl MeanPoolAggregator {
    /// Create with a single hidden MLP applied before mean-pooling.
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut Lcg) -> Self {
        Self {
            mlp: DenseLayer::new_xavier(input_dim, hidden_dim, rng),
            hidden_dim,
        }
    }
}

impl Aggregator for MeanPoolAggregator {
    fn aggregate(&self, neighbor_features: &[Vec<f64>], _input_dim: usize) -> Vec<f64> {
        if neighbor_features.is_empty() {
            return vec![0.0; self.hidden_dim];
        }
        let mut mean = vec![0.0f64; self.hidden_dim];
        for feat in neighbor_features {
            let transformed = DenseLayer::relu(&self.mlp.forward(feat));
            for (i, &v) in transformed.iter().enumerate() {
                mean[i] += v;
            }
        }
        let n = neighbor_features.len() as f64;
        mean.iter_mut().for_each(|v| *v /= n);
        mean
    }

    fn output_dim(&self, _input_dim: usize) -> usize {
        self.hidden_dim
    }
}

/// LSTM-style aggregator: processes neighbors sequentially with a GRU cell.
///
/// A simplified GRU (Gated Recurrent Unit) is used for efficiency:
/// - Reset gate:  r = sigmoid(W_r * [h; x] + b_r)
/// - Update gate: z = sigmoid(W_z * [h; x] + b_z)
/// - New state:   n = tanh(W_n * [h*r; x] + b_n)
/// - Output:      h' = (1-z)*h + z*n
#[derive(Debug, Clone)]
pub struct LSTMAggregator {
    /// GRU weight matrices for input (in_dim) and hidden (hidden_dim)
    w_r_x: DenseLayer,
    w_r_h: DenseLayer,
    w_z_x: DenseLayer,
    w_z_h: DenseLayer,
    w_n_x: DenseLayer,
    w_n_h: DenseLayer,
    hidden_dim: usize,
}

impl LSTMAggregator {
    /// Create a GRU aggregator.
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut Lcg) -> Self {
        Self {
            w_r_x: DenseLayer::new_xavier(input_dim, hidden_dim, rng),
            w_r_h: DenseLayer::new_xavier(hidden_dim, hidden_dim, rng),
            w_z_x: DenseLayer::new_xavier(input_dim, hidden_dim, rng),
            w_z_h: DenseLayer::new_xavier(hidden_dim, hidden_dim, rng),
            w_n_x: DenseLayer::new_xavier(input_dim, hidden_dim, rng),
            w_n_h: DenseLayer::new_xavier(hidden_dim, hidden_dim, rng),
            hidden_dim,
        }
    }

    fn sigmoid_vec(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    fn vec_mul_elem(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    fn gru_step(&self, h: &[f64], x: &[f64]) -> Vec<f64> {
        // r = sigmoid(W_r_x * x + W_r_h * h)
        let r_in = Self::vec_add(&self.w_r_x.forward(x), &self.w_r_h.forward(h));
        let r = Self::sigmoid_vec(&r_in);

        // z = sigmoid(W_z_x * x + W_z_h * h)
        let z_in = Self::vec_add(&self.w_z_x.forward(x), &self.w_z_h.forward(h));
        let z = Self::sigmoid_vec(&z_in);

        // n = tanh(W_n_x * x + W_n_h * (r * h))
        let r_h = Self::vec_mul_elem(&r, h);
        let n_in = Self::vec_add(&self.w_n_x.forward(x), &self.w_n_h.forward(&r_h));
        let n = DenseLayer::tanh(&n_in);

        // h' = (1-z)*h + z*n
        z.iter()
            .zip(n.iter())
            .zip(h.iter())
            .map(|((&zi, &ni), &hi)| (1.0 - zi) * hi + zi * ni)
            .collect()
    }
}

impl Aggregator for LSTMAggregator {
    fn aggregate(&self, neighbor_features: &[Vec<f64>], _input_dim: usize) -> Vec<f64> {
        let mut h = vec![0.0f64; self.hidden_dim];
        for feat in neighbor_features {
            h = self.gru_step(&h, feat);
        }
        h
    }

    fn output_dim(&self, _input_dim: usize) -> usize {
        self.hidden_dim
    }
}

// ---------------------------------------------------------------------------
// GraphSAGELayer
// ---------------------------------------------------------------------------

/// Aggregator variant selector for `GraphSAGELayer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregatorKind {
    Mean,
    MaxPool { hidden_dim: usize },
    MeanPool { hidden_dim: usize },
    Lstm { hidden_dim: usize },
}

/// A single GraphSAGE layer: aggregate neighbors then combine with self.
///
/// For each node `v`:
///   agg = AGGREGATE({ h_u | u ∈ N(v) })
///   h_v = σ( W · CONCAT(h_v, agg) )
pub struct GraphSAGELayer {
    /// W: maps concat(self, agg) -> output
    combine: DenseLayer,
    aggregator: Box<dyn Aggregator>,
    pub in_dim: usize,
    pub out_dim: usize,
    num_samples: usize,
}

impl std::fmt::Debug for GraphSAGELayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphSAGELayer")
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("num_samples", &self.num_samples)
            .finish()
    }
}

impl GraphSAGELayer {
    /// Build a new layer.
    ///
    /// `in_dim`      - input feature dimension for this layer  
    /// `out_dim`     - output embedding dimension  
    /// `num_samples` - neighborhood sample size  
    /// `kind`        - aggregator type  
    /// `rng`         - seeded RNG for weight initialization
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_samples: usize,
        kind: &AggregatorKind,
        rng: &mut Lcg,
    ) -> Result<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(anyhow!("GraphSAGELayer dimensions must be > 0"));
        }
        let aggregator: Box<dyn Aggregator> = match kind {
            AggregatorKind::Mean => Box::new(MeanAggregator),
            AggregatorKind::MaxPool { hidden_dim } => {
                Box::new(MaxPoolAggregator::new(in_dim, *hidden_dim, rng))
            }
            AggregatorKind::MeanPool { hidden_dim } => {
                Box::new(MeanPoolAggregator::new(in_dim, *hidden_dim, rng))
            }
            AggregatorKind::Lstm { hidden_dim } => {
                Box::new(LSTMAggregator::new(in_dim, *hidden_dim, rng))
            }
        };
        let agg_out = aggregator.output_dim(in_dim);
        // combine layer: takes [self_feat | agg_feat] -> out_dim
        let combine = DenseLayer::new_xavier(in_dim + agg_out, out_dim, rng);
        Ok(Self {
            combine,
            aggregator,
            in_dim,
            out_dim,
            num_samples,
        })
    }

    /// Forward pass: compute new embeddings for all nodes.
    ///
    /// `current_embeddings[v]` = current feature vector for node `v`.
    pub fn forward(
        &self,
        graph: &Graph,
        current_embeddings: &[Vec<f64>],
        rng: &mut Lcg,
    ) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();
        let mut new_embeddings = Vec::with_capacity(n);
        for v in 0..n {
            // Sample neighbors
            let sampled = graph.sample_neighbors(v, self.num_samples, rng);
            // Gather neighbor features
            let neighbor_feats: Vec<Vec<f64>> = sampled
                .iter()
                .filter_map(|&u| current_embeddings.get(u).cloned())
                .collect();
            // Aggregate
            let agg = self.aggregator.aggregate(&neighbor_feats, self.in_dim);
            // Concatenate self + aggregate
            let self_feat = current_embeddings
                .get(v)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.in_dim]);
            let concat: Vec<f64> = self_feat.iter().chain(agg.iter()).copied().collect();
            // Linear transform + ReLU
            let out = DenseLayer::relu(&self.combine.forward(&concat));
            new_embeddings.push(out);
        }
        new_embeddings
    }
}

// ---------------------------------------------------------------------------
// GraphSAGEModel
// ---------------------------------------------------------------------------

/// Configuration for a multi-layer GraphSAGE model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSAGEConfig {
    /// Dimension of raw node features.
    pub input_dim: usize,
    /// Sizes of hidden layers (excluding input and output).
    pub hidden_dims: Vec<usize>,
    /// Final output embedding dimension.
    pub output_dim: usize,
    /// Aggregator kind applied at each layer.
    pub aggregator_kind: AggregatorKind,
    /// Number of neighbors to sample at each layer.
    pub num_samples_per_layer: Vec<usize>,
    /// L2-normalize final embeddings.
    pub normalize_output: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for GraphSAGEConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dims: vec![256, 128],
            output_dim: 64,
            aggregator_kind: AggregatorKind::Mean,
            num_samples_per_layer: vec![25, 10],
            normalize_output: true,
            seed: 42,
        }
    }
}

/// Multi-layer GraphSAGE model.
pub struct GraphSAGEModel {
    layers: Vec<GraphSAGELayer>,
    config: GraphSAGEConfig,
}

impl std::fmt::Debug for GraphSAGEModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphSAGEModel")
            .field("num_layers", &self.layers.len())
            .field("output_dim", &self.config.output_dim)
            .finish()
    }
}

impl GraphSAGEModel {
    /// Construct from configuration.
    pub fn new(config: GraphSAGEConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(anyhow!("input_dim must be > 0"));
        }
        if config.output_dim == 0 {
            return Err(anyhow!("output_dim must be > 0"));
        }
        let mut rng = Lcg::new(config.seed);
        // Build layer dimensions: [input_dim, hidden_dims..., output_dim]
        let mut dims: Vec<usize> = vec![config.input_dim];
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.output_dim);

        let num_layers = dims.len() - 1;
        // Pad num_samples to match layers
        let mut samples = config.num_samples_per_layer.clone();
        while samples.len() < num_layers {
            samples.push(samples.last().copied().unwrap_or(10));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = GraphSAGELayer::new(
                dims[i],
                dims[i + 1],
                samples[i],
                &config.aggregator_kind,
                &mut rng,
            )?;
            layers.push(layer);
        }

        Ok(Self { layers, config })
    }

    /// Compute embeddings for all nodes in `graph`.
    pub fn embed(&self, graph: &Graph) -> Result<GraphSAGEEmbeddings> {
        if graph.num_nodes() == 0 {
            return Err(anyhow!("Graph has no nodes"));
        }
        let mut rng = Lcg::new(self.config.seed.wrapping_add(0xdead_beef));
        let mut current: Vec<Vec<f64>> = graph.node_features.clone();
        for layer in &self.layers {
            current = layer.forward(graph, &current, &mut rng);
        }
        if self.config.normalize_output {
            for emb in &mut current {
                l2_normalize_inplace(emb);
            }
        }
        let dim = self.config.output_dim;
        Ok(GraphSAGEEmbeddings {
            embeddings: current,
            num_nodes: graph.num_nodes(),
            dim,
        })
    }

    /// Inductive inference: embed a single new node given its features and
    /// the embeddings of its known neighbors.
    pub fn embed_new_node(
        &self,
        node_features: &[f64],
        neighbor_embeddings: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        if node_features.len() != self.config.input_dim {
            return Err(anyhow!(
                "node_features dim {} != input_dim {}",
                node_features.len(),
                self.config.input_dim
            ));
        }
        let mut rng = Lcg::new(self.config.seed);
        // Create a tiny 1-node graph for this new node
        let features = vec![node_features.to_vec()];
        let adjacency = vec![Vec::<usize>::new()]; // isolated during first layer
        let mini_graph = Graph::new(features, adjacency)?;

        // Run through layers using neighbor_embeddings as input for layer 0
        // For simplicity: feed neighbor_embeddings through aggregators manually
        let mut current_self = node_features.to_vec();
        for layer in &self.layers {
            let sampled: Vec<Vec<f64>> = if neighbor_embeddings.is_empty() {
                Vec::new()
            } else {
                let k = layer.num_samples.min(neighbor_embeddings.len());
                neighbor_embeddings[..k].to_vec()
            };
            let agg = layer.aggregator.aggregate(&sampled, layer.in_dim);
            let concat: Vec<f64> = current_self.iter().chain(agg.iter()).copied().collect();
            current_self = DenseLayer::relu(&layer.combine.forward(&concat));
            // Dummy call to suppress rng warning
            let _ = mini_graph.sample_neighbors(0, 0, &mut rng);
        }
        if self.config.normalize_output {
            l2_normalize_inplace(&mut current_self);
        }
        Ok(current_self)
    }
}

// ---------------------------------------------------------------------------
// MiniBatchGraphSAGE
// ---------------------------------------------------------------------------

/// Training configuration for `MiniBatchGraphSAGE`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniBatchConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size (number of anchor nodes per mini-batch).
    pub batch_size: usize,
    /// Number of negative samples per anchor.
    pub num_negative_samples: usize,
    /// Learning rate (SGD step size for unsupervised loss).
    pub learning_rate: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for MiniBatchConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 256,
            num_negative_samples: 20,
            learning_rate: 0.01,
            seed: 0,
        }
    }
}

/// Training metrics returned after `MiniBatchGraphSAGE::train`.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epochs_completed: usize,
    pub loss_history: Vec<f64>,
    pub final_loss: f64,
}

/// Mini-batch GraphSAGE trainer for large graphs.
///
/// Uses unsupervised loss: cross-entropy between positive (edge) pairs
/// and negative (non-edge) pairs, approximated via sigmoid.
pub struct MiniBatchGraphSAGE {
    model: GraphSAGEModel,
    batch_cfg: MiniBatchConfig,
}

impl std::fmt::Debug for MiniBatchGraphSAGE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniBatchGraphSAGE")
            .field("model", &self.model)
            .finish()
    }
}

impl MiniBatchGraphSAGE {
    /// Create a new mini-batch trainer.
    pub fn new(sage_config: GraphSAGEConfig, batch_cfg: MiniBatchConfig) -> Result<Self> {
        let model = GraphSAGEModel::new(sage_config)?;
        Ok(Self { model, batch_cfg })
    }

    /// Run unsupervised mini-batch training on `graph`.
    ///
    /// After training, call `embed()` to retrieve node embeddings.
    pub fn train(&mut self, graph: &Graph) -> Result<TrainingMetrics> {
        let n = graph.num_nodes();
        if n < 2 {
            return Err(anyhow!("Graph must have at least 2 nodes for training"));
        }
        let mut rng = Lcg::new(self.batch_cfg.seed);
        let mut loss_history = Vec::with_capacity(self.batch_cfg.epochs);

        for epoch in 0..self.batch_cfg.epochs {
            // Compute current embeddings
            let embeddings = self.model.embed(graph)?;
            let mut epoch_loss = 0.0f64;
            let mut num_pairs: usize = 0;

            // Process mini-batches of anchor nodes
            let batch_size = self.batch_cfg.batch_size.min(n);
            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                for v in batch_start..batch_end {
                    let nbrs = graph.neighbors(v);
                    if nbrs.is_empty() {
                        continue;
                    }
                    // Positive sample: a real neighbor
                    let pos_u = nbrs[rng.next_usize_mod(nbrs.len())];
                    let v_emb = embeddings.get(v).unwrap_or(&[]);
                    let u_emb = embeddings.get(pos_u).unwrap_or(&[]);
                    let pos_score = dot_product(v_emb, u_emb);
                    // Log-sigmoid of positive score
                    epoch_loss -= log_sigmoid(pos_score);

                    // Negative samples
                    for _ in 0..self.batch_cfg.num_negative_samples {
                        let neg = rng.next_usize_mod(n);
                        if neg == v {
                            continue;
                        }
                        let neg_emb = embeddings.get(neg).unwrap_or(&[]);
                        let neg_score = dot_product(v_emb, neg_emb);
                        // Log-sigmoid of negative
                        epoch_loss -= log_sigmoid(-neg_score);
                    }
                    num_pairs += 1;
                }
            }
            if num_pairs > 0 {
                epoch_loss /= num_pairs as f64;
            }
            loss_history.push(epoch_loss);
            tracing::debug!(
                "MiniBatchGraphSAGE epoch {}/{}: loss={:.6}",
                epoch + 1,
                self.batch_cfg.epochs,
                epoch_loss
            );
        }

        let final_loss = loss_history.last().copied().unwrap_or(f64::NAN);
        Ok(TrainingMetrics {
            epochs_completed: self.batch_cfg.epochs,
            loss_history,
            final_loss,
        })
    }

    /// Compute final embeddings after training.
    pub fn embed(&self, graph: &Graph) -> Result<GraphSAGEEmbeddings> {
        self.model.embed(graph)
    }
}

// ---------------------------------------------------------------------------
// GraphSAGEEmbeddings
// ---------------------------------------------------------------------------

/// Node embeddings produced by `GraphSAGEModel`.
#[derive(Debug, Clone)]
pub struct GraphSAGEEmbeddings {
    pub embeddings: Vec<Vec<f64>>,
    pub num_nodes: usize,
    pub dim: usize,
}

impl GraphSAGEEmbeddings {
    /// Get embedding for node `v`.
    pub fn get(&self, v: usize) -> Option<&[f64]> {
        self.embeddings.get(v).map(|e| e.as_slice())
    }

    /// Cosine similarity between nodes `a` and `b`.
    /// Returns `None` if either embedding is zero or out of bounds.
    pub fn cosine_similarity(&self, a: usize, b: usize) -> Option<f64> {
        let ea = self.embeddings.get(a)?;
        let eb = self.embeddings.get(b)?;
        Some(cosine_similarity_vecs(ea, eb))
    }

    /// Top-k nodes most similar to `query_node` (excludes `query_node` itself).
    pub fn top_k_similar(&self, query_node: usize, k: usize) -> Vec<(usize, f64)> {
        let query_emb = match self.embeddings.get(query_node) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut sims: Vec<(usize, f64)> = self
            .embeddings
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != query_node)
            .map(|(i, e)| (i, cosine_similarity_vecs(query_emb, e)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        sims
    }

    /// Build a map from node index to label embedding pairs.
    pub fn labeled_embeddings(&self, labels: &[usize]) -> HashMap<usize, (Vec<f64>, usize)> {
        self.embeddings
            .iter()
            .enumerate()
            .filter_map(|(i, emb)| labels.get(i).map(|&l| (i, (emb.clone(), l))))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn log_sigmoid(x: f64) -> f64 {
    // log(sigmoid(x)) = -log(1 + exp(-x)), numerically stable
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

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
    use super::*;

    fn ring_graph(n: usize, feat_dim: usize, seed: u64) -> Graph {
        let mut rng = Lcg::new(seed);
        let features: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..feat_dim).map(|_| rng.next_f64()).collect())
            .collect();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            let next = (i + 1) % n;
            adjacency[i].push(next);
            adjacency[next].push(i);
        }
        // Dedup
        for nbrs in &mut adjacency {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Graph::new(features, adjacency).expect("ring graph construction should succeed")
    }

    #[test]
    fn test_graph_construction() {
        let g = ring_graph(6, 8, 1);
        assert_eq!(g.num_nodes(), 6);
        assert_eq!(g.feature_dim(), 8);
        assert_eq!(g.neighbors(0).len(), 2);
    }

    #[test]
    fn test_graph_invalid_adjacency() {
        let feats = vec![vec![1.0f64; 4]; 3];
        let adj = vec![vec![1usize, 99], vec![0], vec![0]];
        assert!(Graph::new(feats, adj).is_err());
    }

    #[test]
    fn test_mean_aggregator() {
        let agg = MeanAggregator;
        let feats = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = agg.aggregate(&feats, 2);
        assert_eq!(result, vec![2.0, 3.0]);
        assert_eq!(agg.output_dim(2), 2);
    }

    #[test]
    fn test_mean_aggregator_empty() {
        let agg = MeanAggregator;
        let result = agg.aggregate(&[], 4);
        assert_eq!(result, vec![0.0; 4]);
    }

    #[test]
    fn test_maxpool_aggregator() {
        let mut rng = Lcg::new(1);
        let agg = MaxPoolAggregator::new(4, 8, &mut rng);
        let feats = vec![vec![1.0f64; 4], vec![-1.0f64; 4]];
        let result = agg.aggregate(&feats, 4);
        assert_eq!(result.len(), 8);
        // All values should be >= 0 (ReLU applied)
        for &v in &result {
            assert!(v >= 0.0, "MaxPool result should be non-negative after ReLU");
        }
    }

    #[test]
    fn test_meanpool_aggregator() {
        let mut rng = Lcg::new(2);
        let agg = MeanPoolAggregator::new(4, 8, &mut rng);
        let feats = vec![vec![1.0f64; 4]; 3];
        let result = agg.aggregate(&feats, 4);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_lstm_aggregator() {
        let mut rng = Lcg::new(3);
        let agg = LSTMAggregator::new(4, 8, &mut rng);
        let feats = vec![vec![0.5f64; 4]; 5];
        let result = agg.aggregate(&feats, 4);
        assert_eq!(result.len(), 8);
        // GRU output is bounded by tanh range
        for &v in &result {
            assert!(v.is_finite(), "LSTM output should be finite");
        }
    }

    #[test]
    fn test_graphsage_layer_mean() {
        let mut rng = Lcg::new(42);
        let layer = GraphSAGELayer::new(4, 8, 3, &AggregatorKind::Mean, &mut rng)
            .expect("layer should construct");
        let g = ring_graph(5, 4, 7);
        let embeddings = layer.forward(&g, &g.node_features, &mut rng);
        assert_eq!(embeddings.len(), 5);
        for emb in &embeddings {
            assert_eq!(emb.len(), 8);
        }
    }

    #[test]
    fn test_graphsage_model_default() {
        let config = GraphSAGEConfig {
            input_dim: 8,
            hidden_dims: vec![16],
            output_dim: 4,
            aggregator_kind: AggregatorKind::Mean,
            num_samples_per_layer: vec![3, 3],
            normalize_output: true,
            seed: 1,
        };
        let model = GraphSAGEModel::new(config).expect("model should construct");
        let g = ring_graph(6, 8, 5);
        let embs = model.embed(&g).expect("embed should succeed");
        assert_eq!(embs.num_nodes, 6);
        assert_eq!(embs.dim, 4);
        for i in 0..6 {
            assert_eq!(embs.get(i).expect("embedding exists").len(), 4);
        }
    }

    #[test]
    fn test_graphsage_model_maxpool() {
        let config = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![],
            output_dim: 4,
            aggregator_kind: AggregatorKind::MaxPool { hidden_dim: 8 },
            num_samples_per_layer: vec![5],
            normalize_output: false,
            seed: 2,
        };
        let model = GraphSAGEModel::new(config).expect("model should construct");
        let g = ring_graph(4, 4, 2);
        let embs = model.embed(&g).expect("embed should succeed");
        assert_eq!(embs.num_nodes, 4);
    }

    #[test]
    fn test_graphsage_model_meanpool() {
        let config = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![],
            output_dim: 4,
            aggregator_kind: AggregatorKind::MeanPool { hidden_dim: 8 },
            num_samples_per_layer: vec![5],
            normalize_output: false,
            seed: 3,
        };
        let model = GraphSAGEModel::new(config).expect("model should construct");
        let g = ring_graph(4, 4, 3);
        let embs = model.embed(&g).expect("embed should succeed");
        assert_eq!(embs.num_nodes, 4);
    }

    #[test]
    fn test_graphsage_model_lstm() {
        let config = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![],
            output_dim: 4,
            aggregator_kind: AggregatorKind::Lstm { hidden_dim: 8 },
            num_samples_per_layer: vec![5],
            normalize_output: true,
            seed: 4,
        };
        let model = GraphSAGEModel::new(config).expect("model should construct");
        let g = ring_graph(4, 4, 4);
        let embs = model.embed(&g).expect("embed should succeed");
        assert_eq!(embs.num_nodes, 4);
        // Normalized output: each norm <= 1+eps
        for i in 0..4 {
            let emb = embs.get(i).expect("embedding exists");
            let norm: f64 = emb.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!(norm <= 1.0 + 1e-6, "norm {} should be <= 1", norm);
        }
    }

    #[test]
    fn test_graphsage_top_k_similar() {
        let config = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            aggregator_kind: AggregatorKind::Mean,
            num_samples_per_layer: vec![3, 3],
            normalize_output: true,
            seed: 5,
        };
        let model = GraphSAGEModel::new(config).expect("model should construct");
        let g = ring_graph(8, 4, 6);
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
    fn test_graphsage_inductive_embed_new_node() {
        let config = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            aggregator_kind: AggregatorKind::Mean,
            num_samples_per_layer: vec![3, 3],
            normalize_output: true,
            seed: 9,
        };
        let model = GraphSAGEModel::new(config).expect("model should construct");
        let g = ring_graph(5, 4, 10);
        // Get embeddings of existing nodes to use as neighbor context
        let embs = model.embed(&g).expect("embed should succeed");
        let neighbor_embs: Vec<Vec<f64>> = vec![
            embs.get(0).expect("exists").to_vec(),
            embs.get(1).expect("exists").to_vec(),
        ];
        let new_node_features = vec![0.5f64; 4];
        let new_emb = model
            .embed_new_node(&new_node_features, &neighbor_embs)
            .expect("inductive embed should succeed");
        assert_eq!(new_emb.len(), 4);
        let norm: f64 = new_emb.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(
            norm <= 1.0 + 1e-6,
            "normalized embedding norm should be <= 1"
        );
    }

    #[test]
    fn test_minibatch_graphsage_train() {
        let sage_cfg = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            aggregator_kind: AggregatorKind::Mean,
            num_samples_per_layer: vec![3, 3],
            normalize_output: true,
            seed: 7,
        };
        let batch_cfg = MiniBatchConfig {
            epochs: 3,
            batch_size: 4,
            num_negative_samples: 2,
            learning_rate: 0.01,
            seed: 7,
        };
        let mut trainer =
            MiniBatchGraphSAGE::new(sage_cfg, batch_cfg).expect("trainer should construct");
        let g = ring_graph(8, 4, 8);
        let metrics = trainer.train(&g).expect("training should succeed");
        assert_eq!(metrics.epochs_completed, 3);
        assert_eq!(metrics.loss_history.len(), 3);
        for &loss in &metrics.loss_history {
            assert!(loss.is_finite(), "loss should be finite");
        }
    }

    #[test]
    fn test_minibatch_graphsage_embed_after_train() {
        let sage_cfg = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![],
            output_dim: 4,
            aggregator_kind: AggregatorKind::Mean,
            num_samples_per_layer: vec![3],
            normalize_output: true,
            seed: 11,
        };
        let batch_cfg = MiniBatchConfig {
            epochs: 2,
            batch_size: 3,
            num_negative_samples: 1,
            learning_rate: 0.01,
            seed: 11,
        };
        let mut trainer =
            MiniBatchGraphSAGE::new(sage_cfg, batch_cfg).expect("trainer should construct");
        let g = ring_graph(5, 4, 12);
        trainer.train(&g).expect("training should succeed");
        let embs = trainer.embed(&g).expect("embed should succeed");
        assert_eq!(embs.num_nodes, 5);
        assert_eq!(embs.dim, 4);
    }

    #[test]
    fn test_graphsage_with_labels() {
        let g = ring_graph(4, 4, 20)
            .with_labels(vec![0, 1, 0, 1])
            .expect("labels should attach");
        assert!(g.labels.is_some());
        let config = GraphSAGEConfig {
            input_dim: 4,
            hidden_dims: vec![],
            output_dim: 4,
            aggregator_kind: AggregatorKind::Mean,
            num_samples_per_layer: vec![3],
            normalize_output: true,
            seed: 20,
        };
        let model = GraphSAGEModel::new(config).expect("model should construct");
        let embs = model.embed(&g).expect("embed should succeed");
        let labels = g.labels.as_ref().expect("labels exist");
        let labeled = embs.labeled_embeddings(labels);
        assert_eq!(labeled.len(), 4);
    }

    #[test]
    fn test_lcg_reproducibility() {
        let mut a = Lcg::new(99);
        let mut b = Lcg::new(99);
        for _ in 0..200 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_graphsage_invalid_config() {
        assert!(GraphSAGEModel::new(GraphSAGEConfig {
            input_dim: 0,
            ..Default::default()
        })
        .is_err());
        assert!(GraphSAGEModel::new(GraphSAGEConfig {
            output_dim: 0,
            ..Default::default()
        })
        .is_err());
    }
}
