//! GraphSAGE encoder with hand-rolled forward + backward passes.
//!
//! # Architecture
//!
//! Two GraphSAGE layers, each computing:
//! ```text
//! h_agg = MEAN({ h_u : u ∈ sample(N(v), K) })
//! concat = CONCAT(h_v, h_agg)          // dim = 2 * in_dim
//! h_v_new = ReLU( W @ concat + b )     // dim = out_dim
//! ```
//!
//! Layer 1: `W1 ∈ ℝ^{hidden × 2·input}`, `b1 ∈ ℝ^{hidden}`
//! Layer 2: `W2 ∈ ℝ^{output × 2·hidden}`, `b2 ∈ ℝ^{output}`
//!
//! # Training objective
//!
//! Unsupervised link-prediction with margin-ranking loss:
//! ```text
//! L = max(0, 1 − cos_sim(h_s, h_o+) + cos_sim(h_s, h_o−))
//! ```
//!
//! Gradients are computed by hand-rolled chain rule and clipped to max-norm 1.0.

use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::rand_prelude::StdRng;
use scirs2_core::random::{seeded_rng, CoreRandom};

use super::aggregator::mean_aggregate;
use super::sampler::sample_neighbours;

/// Type alias for the three-tensor tuple returned by the cached forward pass:
/// `(activations, pre_activations, neighbour_aggregates)`.
type ForwardCache = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

// ─── Error ───────────────────────────────────────────────────────────────────

/// Errors produced by the GraphSAGE encoder.
#[derive(Debug, thiserror::Error)]
pub enum GnnError {
    #[error("dimension mismatch: {0}")]
    DimMismatch(String),
    #[error("empty graph: {0}")]
    EmptyGraph(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

pub type GnnResult<T> = Result<T, GnnError>;

// ─── Public data structures ───────────────────────────────────────────────────

/// Compact representation of a knowledge graph as adjacency lists.
#[derive(Debug, Clone)]
pub struct KgGraph {
    /// Total number of nodes.
    pub num_nodes: usize,
    /// Directed edges as `(src, dst)` pairs.
    pub edges: Vec<(usize, usize)>,
    /// Input node features: shape `[num_nodes, feat_dim]`.
    pub node_features: Array2<f64>,
}

/// Output entity embeddings produced by the encoder.
#[derive(Debug, Clone)]
pub struct EntityEmbeddings {
    /// Embedding matrix: shape `[num_nodes, emb_dim]`.
    pub embeddings: Array2<f64>,
    /// Optional string identifiers for nodes (positionally aligned).
    pub node_ids: Vec<String>,
}

/// Hyperparameter configuration for the GraphSAGE encoder.
#[derive(Debug, Clone)]
pub struct GraphSageConfig {
    /// Dimension of the input node features.
    pub input_dim: usize,
    /// Hidden dimension after the first GraphSAGE layer.
    pub hidden_dim: usize,
    /// Output dimension after the second layer (= embedding dim).
    pub output_dim: usize,
    /// Number of layers (currently 2 is fully supported).
    pub num_layers: usize,
    /// Dropout probability (0.0 disables).
    pub dropout: f64,
    /// Maximum neighbours to sample per node per layer.
    pub k_neighbors: usize,
    /// SGD learning rate.
    pub learning_rate: f64,
}

impl Default for GraphSageConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 64,
            output_dim: 64,
            num_layers: 2,
            dropout: 0.0,
            k_neighbors: 10,
            learning_rate: 0.01,
        }
    }
}

/// Training history produced by `GraphSageEncoder::train`.
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Per-epoch mean link-prediction loss.
    pub epoch_losses: Vec<f64>,
    /// Loss in the final epoch.
    pub final_loss: f64,
}

// ─── Internal weight storage ─────────────────────────────────────────────────

/// Parameters for one GraphSAGE layer.
///
/// Projects `[2·in_dim] → [out_dim]` via `ReLU(W @ concat + b)`.
#[derive(Debug, Clone)]
struct SageLayer {
    /// Weight matrix rows: shape `[out_dim][2 * in_dim]`.
    w: Vec<Vec<f64>>,
    /// Bias vector: shape `[out_dim]`.
    b: Vec<f64>,
    out_dim: usize,
    in2_dim: usize, // = 2 * in_dim
}

impl SageLayer {
    /// Xavier-uniform initialisation.
    fn new_xavier(in_dim: usize, out_dim: usize, rng: &mut CoreRandom<StdRng>) -> Self {
        let in2_dim = 2 * in_dim;
        let fan_in = in2_dim;
        let fan_out = out_dim;
        let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
        let w: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in2_dim)
                    .map(|_| {
                        let u = rng.random_range(0.0_f64..1.0_f64);
                        u * 2.0 * limit - limit
                    })
                    .collect()
            })
            .collect();
        let b = vec![0.0_f64; out_dim];
        Self {
            w,
            b,
            out_dim,
            in2_dim,
        }
    }

    /// Forward: `ReLU(W @ [self_h ‖ agg_h] + b)`.
    fn forward(&self, self_h: &[f64], agg_h: &[f64]) -> Vec<f64> {
        debug_assert_eq!(self_h.len() + agg_h.len(), self.in2_dim);
        let mut out = vec![0.0_f64; self.out_dim];
        for (i, row) in self.w.iter().enumerate() {
            let dot: f64 = row[..self_h.len()]
                .iter()
                .zip(self_h.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + row[self_h.len()..]
                    .iter()
                    .zip(agg_h.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
            out[i] = (dot + self.b[i]).max(0.0); // ReLU
        }
        out
    }

    /// Pre-activation (linear, no ReLU), used during backward pass.
    fn pre_activation(&self, self_h: &[f64], agg_h: &[f64]) -> Vec<f64> {
        debug_assert_eq!(self_h.len() + agg_h.len(), self.in2_dim);
        let mut out = vec![0.0_f64; self.out_dim];
        for (i, row) in self.w.iter().enumerate() {
            let dot: f64 = row[..self_h.len()]
                .iter()
                .zip(self_h.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + row[self_h.len()..]
                    .iter()
                    .zip(agg_h.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
            out[i] = dot + self.b[i];
        }
        out
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// GraphSAGE encoder.
pub struct GraphSageEncoder {
    layer1: SageLayer,
    layer2: SageLayer,
    config: GraphSageConfig,
    seed: u64,
}

impl GraphSageEncoder {
    /// Create a new encoder with seed 42.
    pub fn new(config: &GraphSageConfig) -> GnnResult<Self> {
        Self::new_with_seed(config, 42)
    }

    /// Create a new encoder with an explicit seed for reproducibility.
    pub fn new_with_seed(config: &GraphSageConfig, seed: u64) -> GnnResult<Self> {
        if config.input_dim == 0 {
            return Err(GnnError::InvalidConfig("input_dim must be > 0".into()));
        }
        if config.hidden_dim == 0 {
            return Err(GnnError::InvalidConfig("hidden_dim must be > 0".into()));
        }
        if config.output_dim == 0 {
            return Err(GnnError::InvalidConfig("output_dim must be > 0".into()));
        }
        let mut rng = seeded_rng(seed);
        let layer1 = SageLayer::new_xavier(config.input_dim, config.hidden_dim, &mut rng);
        let layer2 = SageLayer::new_xavier(config.hidden_dim, config.output_dim, &mut rng);
        Ok(Self {
            layer1,
            layer2,
            config: config.clone(),
            seed,
        })
    }

    // ─── Forward pass ─────────────────────────────────────────────────────

    /// Encode all nodes in `graph` and return entity embeddings.
    ///
    /// A fresh RNG derived from the stored seed is created on each call, so
    /// repeated calls on the same encoder produce identical results.
    pub fn encode(&self, graph: &KgGraph) -> GnnResult<EntityEmbeddings> {
        if graph.num_nodes == 0 {
            return Err(GnnError::EmptyGraph("graph has no nodes".into()));
        }
        let feat_rows = graph.node_features.nrows();
        if feat_rows != graph.num_nodes {
            return Err(GnnError::DimMismatch(format!(
                "node_features has {} rows but num_nodes = {}",
                feat_rows, graph.num_nodes
            )));
        }
        let feat_dim = graph.node_features.ncols();
        if feat_dim != self.config.input_dim {
            return Err(GnnError::DimMismatch(format!(
                "node_features has {} cols but config.input_dim = {}",
                feat_dim, self.config.input_dim
            )));
        }

        let input_h: Vec<Vec<f64>> = (0..graph.num_nodes)
            .map(|i| graph.node_features.row(i).to_vec())
            .collect();

        let mut rng1 = seeded_rng(self.seed.wrapping_add(1));
        let h1 = self.sage_layer_forward(&self.layer1, &input_h, graph, &mut rng1);

        let mut rng2 = seeded_rng(self.seed.wrapping_add(2));
        let h2 = self.sage_layer_forward(&self.layer2, &h1, graph, &mut rng2);

        let out_dim = self.config.output_dim;
        let mut embeddings = Array2::zeros((graph.num_nodes, out_dim));
        for (i, row) in h2.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                embeddings[[i, j]] = v;
            }
        }

        Ok(EntityEmbeddings {
            embeddings,
            node_ids: (0..graph.num_nodes).map(|i| i.to_string()).collect(),
        })
    }

    /// Run one GraphSAGE layer forward pass over all nodes.
    fn sage_layer_forward(
        &self,
        layer: &SageLayer,
        h_prev: &[Vec<f64>],
        graph: &KgGraph,
        rng: &mut CoreRandom<StdRng>,
    ) -> Vec<Vec<f64>> {
        let in_dim = if h_prev.is_empty() {
            0
        } else {
            h_prev[0].len()
        };
        let zero_agg = vec![0.0_f64; in_dim];

        (0..graph.num_nodes)
            .map(|v| {
                let neighbours = sample_neighbours(v, &graph.edges, self.config.k_neighbors, rng);
                let agg = if neighbours.is_empty() {
                    zero_agg.clone()
                } else {
                    let nb_embs: Vec<Vec<f64>> =
                        neighbours.iter().map(|&u| h_prev[u].clone()).collect();
                    mean_aggregate(&nb_embs)
                };
                layer.forward(&h_prev[v], &agg)
            })
            .collect()
    }

    // ─── Training ─────────────────────────────────────────────────────────

    /// Train the encoder for `num_epochs` using link-prediction loss.
    pub fn train(&mut self, graph: &KgGraph, num_epochs: usize) -> GnnResult<TrainingHistory> {
        if graph.num_nodes < 2 {
            return Err(GnnError::EmptyGraph(
                "need at least 2 nodes for training".into(),
            ));
        }
        if graph.edges.is_empty() {
            return Err(GnnError::EmptyGraph(
                "no edges to form positive pairs".into(),
            ));
        }

        let mut epoch_losses = Vec::with_capacity(num_epochs);
        let mut rng = seeded_rng(self.seed.wrapping_add(100));

        for _ in 0..num_epochs {
            // Forward pass, layer 1.
            let input_h: Vec<Vec<f64>> = (0..graph.num_nodes)
                .map(|i| graph.node_features.row(i).to_vec())
                .collect();

            let (h1, pre1, agg1) = self.forward_with_cache(
                self.config.k_neighbors,
                &self.layer1,
                &input_h,
                graph,
                &mut rng,
            );

            // Forward pass, layer 2.
            let (h2, pre2, agg2) = self.forward_with_cache(
                self.config.k_neighbors,
                &self.layer2,
                &h1,
                graph,
                &mut rng,
            );

            // Sample positive pair.
            let pos_idx = {
                let n = graph.edges.len();
                let u = rng.random_range(0.0_f64..1.0_f64);
                (u * n as f64) as usize % n
            };
            let (s, o_pos) = graph.edges[pos_idx];

            // Sample negative (not a neighbour of s).
            let o_neg = self.sample_negative(s, graph, &mut rng);

            let sim_pos = cosine_sim(&h2[s], &h2[o_pos]);
            let sim_neg = cosine_sim(&h2[s], &h2[o_neg]);
            let margin = 1.0_f64 - sim_pos + sim_neg;
            let loss = margin.max(0.0);
            epoch_losses.push(loss);

            if loss <= 0.0 {
                continue;
            }

            // Backward pass.
            let (grad_s, grad_opos, grad_oneg) = cosine_sim_grads(&h2[s], &h2[o_pos], &h2[o_neg]);

            let mut dl_dh2 = vec![vec![0.0_f64; self.config.output_dim]; graph.num_nodes];
            add_vec(&mut dl_dh2[s], &grad_s);
            add_vec(&mut dl_dh2[o_pos], &grad_opos);
            add_vec(&mut dl_dh2[o_neg], &grad_oneg);

            let (dw2, db2, dl_dh1) = backward_layer(
                &self.layer2,
                &dl_dh2,
                &h1,
                &pre2,
                &agg2,
                graph.num_nodes,
                self.config.hidden_dim,
            );

            let (dw1, db1, _) = backward_layer(
                &self.layer1,
                &dl_dh1,
                &input_h,
                &pre1,
                &agg1,
                graph.num_nodes,
                self.config.input_dim,
            );

            let lr = self.config.learning_rate;
            apply_grad_2d(&mut self.layer2.w, &dw2, lr);
            apply_grad_1d(&mut self.layer2.b, &db2, lr);
            apply_grad_2d(&mut self.layer1.w, &dw1, lr);
            apply_grad_1d(&mut self.layer1.b, &db1, lr);
        }

        let final_loss = epoch_losses.last().copied().unwrap_or(0.0);
        Ok(TrainingHistory {
            epoch_losses,
            final_loss,
        })
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    /// Forward pass saving pre-activations and aggregations for backprop.
    fn forward_with_cache(
        &self,
        k: usize,
        layer: &SageLayer,
        h_prev: &[Vec<f64>],
        graph: &KgGraph,
        rng: &mut CoreRandom<StdRng>,
    ) -> ForwardCache {
        let in_dim = if h_prev.is_empty() {
            0
        } else {
            h_prev[0].len()
        };
        let zero_agg = vec![0.0_f64; in_dim];

        let mut h_out = Vec::with_capacity(graph.num_nodes);
        let mut pre_acts = Vec::with_capacity(graph.num_nodes);
        let mut aggs = Vec::with_capacity(graph.num_nodes);

        for v in 0..graph.num_nodes {
            let neighbours = sample_neighbours(v, &graph.edges, k, rng);
            let agg = if neighbours.is_empty() {
                zero_agg.clone()
            } else {
                let nb_embs: Vec<Vec<f64>> =
                    neighbours.iter().map(|&u| h_prev[u].clone()).collect();
                mean_aggregate(&nb_embs)
            };
            let pre = layer.pre_activation(&h_prev[v], &agg);
            let out: Vec<f64> = pre.iter().map(|&z| z.max(0.0)).collect();
            pre_acts.push(pre);
            aggs.push(agg);
            h_out.push(out);
        }
        (h_out, pre_acts, aggs)
    }

    /// Sample a node that is not a neighbour of `src`.
    fn sample_negative(&self, src: usize, graph: &KgGraph, rng: &mut CoreRandom<StdRng>) -> usize {
        let neighbours: std::collections::HashSet<usize> = graph
            .edges
            .iter()
            .filter_map(|&(s, d)| if s == src { Some(d) } else { None })
            .collect();
        for _ in 0..200 {
            let u = rng.random_range(0.0_f64..1.0_f64);
            let candidate = (u * graph.num_nodes as f64) as usize % graph.num_nodes;
            if candidate != src && !neighbours.contains(&candidate) {
                return candidate;
            }
        }
        // Fallback: first node that is not src and not a neighbour.
        for c in 0..graph.num_nodes {
            if c != src && !neighbours.contains(&c) {
                return c;
            }
        }
        (src + 1) % graph.num_nodes
    }

    // ─── Test-helpers ─────────────────────────────────────────────────────

    /// Return `(analytic_grad_of_W1[0][0], W1[0][0])` for a fixed triple
    /// `(0 → 1)` as positive pair.  Intended for finite-difference checks.
    pub fn compute_grad_and_param_for_test(&mut self, graph: &KgGraph) -> (f64, f64) {
        let mut rng = seeded_rng(self.seed.wrapping_add(100));

        let input_h: Vec<Vec<f64>> = (0..graph.num_nodes)
            .map(|i| graph.node_features.row(i).to_vec())
            .collect();
        let (h1, pre1, agg1) = self.forward_with_cache(
            self.config.k_neighbors,
            &self.layer1,
            &input_h,
            graph,
            &mut rng,
        );
        let (h2, pre2, agg2) =
            self.forward_with_cache(self.config.k_neighbors, &self.layer2, &h1, graph, &mut rng);

        let s = 0_usize;
        let o_pos = 1_usize;
        let o_neg = self.sample_negative(s, graph, &mut rng);

        let margin = 1.0 - cosine_sim(&h2[s], &h2[o_pos]) + cosine_sim(&h2[s], &h2[o_neg]);
        if margin <= 0.0 {
            return (0.0, self.layer1.w[0][0]);
        }

        let (grad_s, grad_opos, grad_oneg) = cosine_sim_grads(&h2[s], &h2[o_pos], &h2[o_neg]);

        let mut dl_dh2 = vec![vec![0.0_f64; self.config.output_dim]; graph.num_nodes];
        add_vec(&mut dl_dh2[s], &grad_s);
        add_vec(&mut dl_dh2[o_pos], &grad_opos);
        add_vec(&mut dl_dh2[o_neg], &grad_oneg);

        let (_dw2, _db2, dl_dh1) = backward_layer(
            &self.layer2,
            &dl_dh2,
            &h1,
            &pre2,
            &agg2,
            graph.num_nodes,
            self.config.hidden_dim,
        );

        let (dw1, _db1, _) = backward_layer(
            &self.layer1,
            &dl_dh1,
            &input_h,
            &pre1,
            &agg1,
            graph.num_nodes,
            self.config.input_dim,
        );

        (dw1[0][0], self.layer1.w[0][0])
    }

    /// Compute loss with `W1[0][0]` perturbed by `eps`, then restore.
    /// Uses the same RNG seed as `compute_grad_and_param_for_test`.
    pub fn compute_loss_with_perturb(&mut self, graph: &KgGraph, eps: f64) -> f64 {
        self.layer1.w[0][0] += eps;

        let mut rng = seeded_rng(self.seed.wrapping_add(100));
        let input_h: Vec<Vec<f64>> = (0..graph.num_nodes)
            .map(|i| graph.node_features.row(i).to_vec())
            .collect();
        let (h1, _, _) = self.forward_with_cache(
            self.config.k_neighbors,
            &self.layer1,
            &input_h,
            graph,
            &mut rng,
        );
        let (h2, _, _) =
            self.forward_with_cache(self.config.k_neighbors, &self.layer2, &h1, graph, &mut rng);

        let o_neg = self.sample_negative(0, graph, &mut rng);
        let sim_pos = cosine_sim(&h2[0], &h2[1]);
        let sim_neg = cosine_sim(&h2[0], &h2[o_neg]);
        let loss = (1.0 - sim_pos + sim_neg).max(0.0);

        self.layer1.w[0][0] -= eps;
        loss
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Cosine similarity (returns 0 when either vector is near-zero).
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Gradient of cosine similarity w.r.t. the first argument `a`.
fn cos_grad_a(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return vec![0.0; dim];
    }
    let sim = dot / (na * nb);
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| bi / (na * nb) - ai * sim / (na * na))
        .collect()
}

/// Gradient of cosine similarity w.r.t. the second argument `b`.
fn cos_grad_b(a: &[f64], b: &[f64]) -> Vec<f64> {
    cos_grad_a(b, a)
}

/// Compute embedding-space gradients for the margin-ranking loss.
///
/// Loss = max(0, 1 − sim(s, o+) + sim(s, o−)).
/// Returns `(grad_s, grad_opos, grad_oneg)`.
fn cosine_sim_grads(h_s: &[f64], h_opos: &[f64], h_oneg: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dim = h_s.len();

    // dL/d_h_s = -d_sim_pos/d_h_s + d_sim_neg/d_h_s
    let d_simpos_ds = cos_grad_a(h_s, h_opos);
    let d_simneg_ds = cos_grad_a(h_s, h_oneg);
    let mut grad_s = vec![0.0_f64; dim];
    for i in 0..dim {
        grad_s[i] = -d_simpos_ds[i] + d_simneg_ds[i];
    }

    // dL/d_h_opos = -d_sim_pos/d_h_opos
    let grad_opos: Vec<f64> = cos_grad_b(h_s, h_opos).into_iter().map(|g| -g).collect();

    // dL/d_h_oneg = +d_sim_neg/d_h_oneg
    let grad_oneg = cos_grad_b(h_s, h_oneg);

    (grad_s, grad_opos, grad_oneg)
}

/// Element-wise in-place addition.
fn add_vec(dst: &mut [f64], src: &[f64]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// Compute gradients for one GraphSAGE layer via chain rule.
///
/// Returns `(dW, db, dl_dh_prev)`.
fn backward_layer(
    layer: &SageLayer,
    dl_dh: &[Vec<f64>],    // [num_nodes, out_dim]
    h_prev: &[Vec<f64>],   // [num_nodes, in_dim]
    pre_acts: &[Vec<f64>], // [num_nodes, out_dim]
    aggs: &[Vec<f64>],     // [num_nodes, in_dim]
    num_nodes: usize,
    in_dim: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let out_dim = layer.out_dim;
    let in2 = layer.in2_dim;

    let mut dw = vec![vec![0.0_f64; in2]; out_dim];
    let mut db = vec![0.0_f64; out_dim];
    let mut dl_dh_prev = vec![vec![0.0_f64; in_dim]; num_nodes];

    for v in 0..num_nodes {
        // ReLU mask.
        let d_pre: Vec<f64> = dl_dh[v]
            .iter()
            .zip(pre_acts[v].iter())
            .map(|(&g, &z)| if z > 0.0 { g } else { 0.0 })
            .collect();

        let self_h = &h_prev[v];
        let agg_h = &aggs[v];

        for (i, &dp) in d_pre.iter().enumerate() {
            for (j, &sh) in self_h.iter().enumerate() {
                dw[i][j] += dp * sh;
            }
            for (j, &ah) in agg_h.iter().enumerate() {
                dw[i][in_dim + j] += dp * ah;
            }
            db[i] += dp;
        }

        // Gradient into h_prev (through self-embedding part of concat).
        for (j, dh) in dl_dh_prev[v].iter_mut().enumerate() {
            for (i, &dp) in d_pre.iter().enumerate() {
                *dh += layer.w[i][j] * dp;
            }
        }
    }

    (dw, db, dl_dh_prev)
}

/// Clip 2-D gradient to max-norm 1.0 and apply SGD update.
fn apply_grad_2d(w: &mut [Vec<f64>], raw_grad: &[Vec<f64>], lr: f64) {
    let norm_sq: f64 = raw_grad
        .iter()
        .flat_map(|row| row.iter())
        .map(|&g| g * g)
        .sum();
    let norm = norm_sq.sqrt();
    let scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };
    for (row, grow) in w.iter_mut().zip(raw_grad.iter()) {
        for (wi, &gi) in row.iter_mut().zip(grow.iter()) {
            *wi -= lr * gi * scale;
        }
    }
}

/// Clip 1-D gradient to max-norm 1.0 and apply SGD update.
fn apply_grad_1d(b: &mut [f64], raw_grad: &[f64], lr: f64) {
    let norm_sq: f64 = raw_grad.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();
    let scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };
    for (bi, &gi) in b.iter_mut().zip(raw_grad.iter()) {
        *bi -= lr * gi * scale;
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array2;

    fn tiny_graph() -> KgGraph {
        KgGraph {
            num_nodes: 4,
            edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            node_features: Array2::zeros((4, 4)),
        }
    }

    fn tiny_config() -> GraphSageConfig {
        GraphSageConfig {
            input_dim: 4,
            hidden_dim: 4,
            output_dim: 4,
            num_layers: 2,
            dropout: 0.0,
            k_neighbors: 2,
            learning_rate: 0.01,
        }
    }

    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_sim(&v, &v);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_sim_zero_vec() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    #[test]
    fn test_forward_shape() {
        let graph = tiny_graph();
        let config = tiny_config();
        let enc = GraphSageEncoder::new_with_seed(&config, 99).expect("construct");
        let emb = enc.encode(&graph).expect("encode");
        assert_eq!(emb.embeddings.nrows(), 4);
        assert_eq!(emb.embeddings.ncols(), 4);
    }

    #[test]
    fn test_reject_zero_dim() {
        let mut cfg = tiny_config();
        cfg.input_dim = 0;
        assert!(GraphSageEncoder::new(&cfg).is_err());
    }

    #[test]
    fn test_reject_feat_dim_mismatch() {
        let config = tiny_config(); // input_dim = 4
        let enc = GraphSageEncoder::new_with_seed(&config, 1).expect("construct");
        let bad_graph = KgGraph {
            num_nodes: 2,
            edges: vec![(0, 1)],
            node_features: Array2::zeros((2, 8)),
        };
        assert!(enc.encode(&bad_graph).is_err());
    }
}
