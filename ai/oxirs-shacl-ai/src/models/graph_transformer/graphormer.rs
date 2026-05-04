//! Graphormer architecture (Ying et al. 2021).
//!
//! Key features:
//! - **Centrality encoding**: in/out degree → learnable embeddings added to node features.
//! - **Spatial encoding**: pairwise shortest-path distances (capped at `max_dist=20`),
//!   each distance gets a learnable scalar added to attention scores.
//! - **Standard transformer blocks**: multi-head self-attention + FFN + LayerNorm.
//!
//! Backward pass is hand-rolled (no autograd framework).

use std::collections::VecDeque;

use scirs2_core::ndarray_ext::{Array1, Array2};

use super::{
    attention::{AttentionCache, LayerNorm, MultiHeadAttention},
    positional_encoding::{CentralityEncoding, DetRng},
    GraphTransformerError,
};

// ---------------------------------------------------------------------------
// Forward-pass cache
// ---------------------------------------------------------------------------

/// Stores all intermediate activations needed for the backward pass of one
/// Graphormer layer.
#[derive(Debug, Clone)]
pub struct GraphormerLayerCache {
    /// Input to the layer (before attention sub-block): `[n, hidden_dim]`.
    pub x_in: Array2<f64>,
    /// Output of MHA (before add+norm): `[n, hidden_dim]`.
    pub attn_out: Array2<f64>,
    /// Residual + LayerNorm1 output: `[n, hidden_dim]`.
    pub norm1_out: Array2<f64>,
    /// FFN intermediate (after W1 + ReLU): `[n, ffn_dim]`.
    pub ffn_hidden: Array2<f64>,
    /// FFN output (before add+norm): `[n, hidden_dim]`.
    pub ffn_out: Array2<f64>,
    /// Residual + LayerNorm2 output (= layer output): `[n, hidden_dim]`.
    pub norm2_out: Array2<f64>,
    /// MHA internal cache (for backward).
    pub attn_cache: AttentionCache,
    /// Spatial bias used in this layer's forward: `[n, n]`.
    pub spatial_bias: Array2<f64>,
}

// ---------------------------------------------------------------------------
// Graphormer layer
// ---------------------------------------------------------------------------

/// A single Graphormer transformer block.
pub struct GraphormerLayer {
    /// Multi-head self-attention.
    pub attention: MultiHeadAttention,
    /// FFN W1: `[hidden_dim, ffn_dim]`.
    pub ffn_w1: Array2<f64>,
    /// FFN W2: `[ffn_dim, hidden_dim]`.
    pub ffn_w2: Array2<f64>,
    /// FFN b1: `[ffn_dim]`.
    pub ffn_b1: Array1<f64>,
    /// FFN b2: `[hidden_dim]`.
    pub ffn_b2: Array1<f64>,
    /// Layer normalisation 1 (after attention sub-block).
    pub layer_norm1: LayerNorm,
    /// Layer normalisation 2 (after FFN sub-block).
    pub layer_norm2: LayerNorm,
    /// Spatial bias scalars: `[max_dist+1]`. One scalar per distance bucket,
    /// broadcast over all attention heads.
    pub spatial_bias: Array1<f64>,
    /// Maximum shortest-path distance before clamping.
    pub max_dist: usize,
}

impl GraphormerLayer {
    /// Create a new layer.
    pub fn new(
        hidden_dim: usize,
        ffn_dim: usize,
        num_heads: usize,
        max_dist: usize,
        seed: u64,
    ) -> Result<Self, GraphTransformerError> {
        let head_dim = hidden_dim
            .checked_div(num_heads)
            .filter(|&d| d > 0 && d * num_heads == hidden_dim)
            .ok_or_else(|| {
                GraphTransformerError::Config(format!(
                    "hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
                ))
            })?;

        let mut rng = DetRng::new(seed);
        let attention = MultiHeadAttention::new(num_heads, head_dim, seed.wrapping_add(1))?;

        let mut ffn_w1 = Array2::<f64>::zeros((hidden_dim, ffn_dim));
        let mut ffn_w2 = Array2::<f64>::zeros((ffn_dim, hidden_dim));
        for i in 0..hidden_dim {
            for j in 0..ffn_dim {
                ffn_w1[[i, j]] = rng.xavier(hidden_dim, ffn_dim);
            }
        }
        for i in 0..ffn_dim {
            for j in 0..hidden_dim {
                ffn_w2[[i, j]] = rng.xavier(ffn_dim, hidden_dim);
            }
        }

        // Initialise spatial bias scalars small.
        let mut spatial_bias = Array1::<f64>::zeros(max_dist + 1);
        for v in spatial_bias.iter_mut() {
            *v = rng.xavier(max_dist + 1, 1);
        }

        Ok(Self {
            attention,
            ffn_w1,
            ffn_w2,
            ffn_b1: Array1::<f64>::zeros(ffn_dim),
            ffn_b2: Array1::<f64>::zeros(hidden_dim),
            layer_norm1: LayerNorm::new(hidden_dim),
            layer_norm2: LayerNorm::new(hidden_dim),
            spatial_bias,
            max_dist,
        })
    }

    /// Apply FFN: `x [n, hidden_dim]` → `[n, hidden_dim]`.
    fn ffn_forward(&self, x: &Array2<f64>, hidden: &mut Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let hidden_dim = x.ncols();
        let ffn_dim = self.ffn_w1.ncols();

        // W1 + b1 + ReLU.
        for i in 0..n {
            for j in 0..ffn_dim {
                let mut s = self.ffn_b1[j];
                for k in 0..hidden_dim {
                    s += x[[i, k]] * self.ffn_w1[[k, j]];
                }
                hidden[[i, j]] = s.max(0.0);
            }
        }

        // W2 + b2.
        let mut out = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                let mut s = self.ffn_b2[j];
                for k in 0..ffn_dim {
                    s += hidden[[i, k]] * self.ffn_w2[[k, j]];
                }
                out[[i, j]] = s;
            }
        }
        out
    }

    /// Build per-pair spatial bias matrix from a distance matrix.
    fn build_spatial_bias_matrix(&self, dist: &Array2<usize>, n: usize) -> Array2<f64> {
        let mut bias = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let d = dist[[i, j]].min(self.max_dist);
                bias[[i, j]] = self.spatial_bias[d];
            }
        }
        bias
    }

    /// Forward pass.
    ///
    /// `x`: `[n, hidden_dim]`.
    /// `dist`: `[n, n]` shortest-path distance matrix.
    ///
    /// Returns `(output [n, hidden_dim], cache)`.
    pub fn forward(
        &mut self,
        x: &Array2<f64>,
        dist: &Array2<usize>,
    ) -> Result<(Array2<f64>, GraphormerLayerCache), GraphTransformerError> {
        let n = x.nrows();
        let hidden_dim = x.ncols();
        let ffn_dim = self.ffn_w1.ncols();

        // Build spatial bias.
        let spatial_bias_mat = self.build_spatial_bias_matrix(dist, n);

        // MHA sub-block: MHA(x) + residual + LN1.
        let (attn_out, attn_cache) = self.attention.forward(x, Some(&spatial_bias_mat), None)?;

        let mut residual1 = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                residual1[[i, j]] = x[[i, j]] + attn_out[[i, j]];
            }
        }
        let norm1_out = self.layer_norm1.forward(&residual1);

        // FFN sub-block: FFN(norm1_out) + residual + LN2.
        let mut ffn_hidden = Array2::<f64>::zeros((n, ffn_dim));
        let ffn_out = self.ffn_forward(&norm1_out, &mut ffn_hidden);

        let mut residual2 = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                residual2[[i, j]] = norm1_out[[i, j]] + ffn_out[[i, j]];
            }
        }
        let norm2_out = self.layer_norm2.forward(&residual2);

        let cache = GraphormerLayerCache {
            x_in: x.clone(),
            attn_out,
            norm1_out: norm1_out.clone(),
            ffn_hidden,
            ffn_out,
            norm2_out: norm2_out.clone(),
            attn_cache,
            spatial_bias: spatial_bias_mat,
        };

        Ok((norm2_out, cache))
    }

    /// Backward pass through one Graphormer layer.
    ///
    /// `grad_out`: gradient w.r.t. this layer's output `[n, hidden_dim]`.
    /// Returns gradient w.r.t. this layer's input `[n, hidden_dim]`.
    pub fn backward(
        &mut self,
        grad_out: &Array2<f64>,
        cache: &GraphormerLayerCache,
        lr: f64,
    ) -> Array2<f64> {
        let n = grad_out.nrows();
        let hidden_dim = grad_out.ncols();
        let ffn_dim = self.ffn_w1.ncols();

        // --- Backprop through LN2 ---
        let residual2 = {
            let mut r = Array2::<f64>::zeros((n, hidden_dim));
            for i in 0..n {
                for j in 0..hidden_dim {
                    r[[i, j]] = cache.norm1_out[[i, j]] + cache.ffn_out[[i, j]];
                }
            }
            r
        };
        let d_residual2 = self.layer_norm2.backward(grad_out, &residual2, lr);

        // d_ffn_out = d_residual2 (since norm2 residual connection: grad flows both ways).
        // d_norm1_from_residual2 = d_residual2 as well.
        let d_ffn_out = d_residual2.clone();
        let d_norm1_from_res2 = d_residual2;

        // --- Backprop through FFN ---
        // FFN: hidden = ReLU(norm1_out @ W1 + b1); out = hidden @ W2 + b2
        // d_b2 = sum(d_ffn_out, axis=0); d_W2 = hidden^T @ d_ffn_out
        // d_hidden = d_ffn_out @ W2^T (before ReLU mask)
        let mut dw2 = Array2::<f64>::zeros((ffn_dim, hidden_dim));
        let mut db2 = Array1::<f64>::zeros(hidden_dim);
        let mut d_hidden_pre_relu = Array2::<f64>::zeros((n, ffn_dim));

        for i in 0..n {
            for j in 0..hidden_dim {
                db2[j] += d_ffn_out[[i, j]];
            }
        }
        for k in 0..ffn_dim {
            for j in 0..hidden_dim {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += cache.ffn_hidden[[i, k]] * d_ffn_out[[i, j]];
                }
                dw2[[k, j]] = s;
            }
        }
        // d_hidden (before relu) = d_ffn_out @ W2^T, then mask by relu gate.
        for i in 0..n {
            for k in 0..ffn_dim {
                let mut s = 0.0_f64;
                for j in 0..hidden_dim {
                    s += d_ffn_out[[i, j]] * self.ffn_w2[[k, j]];
                }
                // ReLU gate: hidden_pre_act = norm1_out @ W1 + b1
                // gate = 1 if ffn_hidden > 0, else 0
                d_hidden_pre_relu[[i, k]] = if cache.ffn_hidden[[i, k]] > 0.0 {
                    s
                } else {
                    0.0
                };
            }
        }

        // d_norm1_from_ffn = d_hidden_pre_relu @ W1^T
        let mut d_norm1_from_ffn = Array2::<f64>::zeros((n, hidden_dim));
        let mut dw1 = Array2::<f64>::zeros((hidden_dim, ffn_dim));
        let mut db1 = Array1::<f64>::zeros(ffn_dim);

        for i in 0..n {
            for k in 0..ffn_dim {
                db1[k] += d_hidden_pre_relu[[i, k]];
            }
        }
        for p in 0..hidden_dim {
            for k in 0..ffn_dim {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += cache.norm1_out[[i, p]] * d_hidden_pre_relu[[i, k]];
                }
                dw1[[p, k]] = s;
            }
        }
        for i in 0..n {
            for p in 0..hidden_dim {
                let mut s = 0.0_f64;
                for k in 0..ffn_dim {
                    s += d_hidden_pre_relu[[i, k]] * self.ffn_w1[[p, k]];
                }
                d_norm1_from_ffn[[i, p]] = s;
            }
        }

        // Total grad through norm1_out.
        let mut d_norm1_out = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                d_norm1_out[[i, j]] = d_norm1_from_ffn[[i, j]] + d_norm1_from_res2[[i, j]];
            }
        }

        // --- Backprop through LN1 ---
        let residual1 = {
            let mut r = Array2::<f64>::zeros((n, hidden_dim));
            for i in 0..n {
                for j in 0..hidden_dim {
                    r[[i, j]] = cache.x_in[[i, j]] + cache.attn_out[[i, j]];
                }
            }
            r
        };
        let d_residual1 = self.layer_norm1.backward(&d_norm1_out, &residual1, lr);

        // d_attn_out = d_residual1; d_x_from_residual1 = d_residual1.
        let d_attn_out = d_residual1.clone();
        let d_x_from_res1 = d_residual1;

        // --- Backprop through MHA ---
        let d_x_from_attn = self.attention.backward(&d_attn_out, &cache.attn_cache, lr);

        // Total gradient w.r.t. input x.
        let mut dx = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                dx[[i, j]] = d_x_from_attn[[i, j]] + d_x_from_res1[[i, j]];
            }
        }

        // Apply weight updates.
        for p in 0..hidden_dim {
            for k in 0..ffn_dim {
                self.ffn_w1[[p, k]] -= lr * dw1[[p, k]];
            }
            self.ffn_b2[p] -= lr * db2[p];
        }
        for k in 0..ffn_dim {
            for p in 0..hidden_dim {
                self.ffn_w2[[k, p]] -= lr * dw2[[k, p]];
            }
            self.ffn_b1[k] -= lr * db1[k];
        }

        dx
    }
}

// ---------------------------------------------------------------------------
// Full Graphormer model
// ---------------------------------------------------------------------------

/// Forward-pass cache for the full Graphormer model.
#[derive(Debug)]
pub struct GraphormerCache {
    /// Per-layer caches.
    pub layer_caches: Vec<GraphormerLayerCache>,
    /// Node features after centrality encoding + input projection: `[n, hidden_dim]`.
    pub embedded: Array2<f64>,
    /// Input projection weight (copy for backward): `[input_dim, hidden_dim]`.
    pub input_proj: Array2<f64>,
    /// Shortest-path distance matrix: `[n, n]`.
    pub dist: Array2<usize>,
    /// In-degrees used in forward.
    pub in_degrees: Vec<usize>,
    /// Out-degrees used in forward.
    pub out_degrees: Vec<usize>,
    /// Raw node features (before projection): `[n, input_dim]`.
    pub raw_features: Array2<f64>,
}

/// Full Graphormer model.
pub struct GraphormerModel {
    /// Centrality encoding.
    pub centrality_enc: CentralityEncoding,
    /// Input projection: `[input_dim, hidden_dim]`.
    pub input_proj: Array2<f64>,
    /// Transformer layers.
    pub layers: Vec<GraphormerLayer>,
    /// Output projection: `[hidden_dim, output_dim]`.
    pub output_proj: Array2<f64>,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Input dimension.
    pub input_dim: usize,
}

impl GraphormerModel {
    /// Create a new Graphormer model.
    ///
    /// `input_dim`: raw node feature dimension.
    /// `hidden_dim`: transformer hidden dimension (must be divisible by `num_heads`).
    /// `num_heads`: number of attention heads.
    /// `num_layers`: number of transformer layers.
    /// `output_dim`: dimension of per-node output.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        output_dim: usize,
    ) -> Result<Self, GraphTransformerError> {
        if hidden_dim % num_heads != 0 {
            return Err(GraphTransformerError::Config(format!(
                "hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
            )));
        }
        if num_layers == 0 {
            return Err(GraphTransformerError::Config(
                "num_layers must be >= 1".to_string(),
            ));
        }

        let max_degree = 64usize;
        let max_dist = 20usize;
        let ffn_dim = hidden_dim * 4;
        let mut rng = DetRng::new(12345);

        let centrality_enc = CentralityEncoding::new(max_degree, hidden_dim, 111);

        let mut input_proj = Array2::<f64>::zeros((input_dim, hidden_dim));
        for i in 0..input_dim {
            for j in 0..hidden_dim {
                input_proj[[i, j]] = rng.xavier(input_dim, hidden_dim);
            }
        }

        let layers: Result<Vec<_>, _> = (0..num_layers)
            .map(|l| {
                GraphormerLayer::new(
                    hidden_dim,
                    ffn_dim,
                    num_heads,
                    max_dist,
                    (l as u64).wrapping_mul(1337).wrapping_add(999),
                )
            })
            .collect();
        let layers = layers?;

        let mut output_proj = Array2::<f64>::zeros((hidden_dim, output_dim));
        for i in 0..hidden_dim {
            for j in 0..output_dim {
                output_proj[[i, j]] = rng.xavier(hidden_dim, output_dim);
            }
        }

        Ok(Self {
            centrality_enc,
            input_proj,
            layers,
            output_proj,
            num_heads,
            hidden_dim,
            output_dim,
            input_dim,
        })
    }

    /// Compute in- and out-degrees from adjacency matrix.
    fn compute_degrees(adj: &Array2<f64>, n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut in_deg = vec![0usize; n];
        let mut out_deg = vec![0usize; n];
        for i in 0..n {
            for j in 0..n {
                if adj[[i, j]] > 0.0 {
                    out_deg[i] += 1;
                    in_deg[j] += 1;
                }
            }
        }
        (in_deg, out_deg)
    }

    /// BFS-based all-pairs shortest-path distances (capped at `max_dist`).
    fn all_pairs_bfs(adj: &Array2<f64>, n: usize, max_dist: usize) -> Array2<usize> {
        let mut dist = Array2::<usize>::zeros((n, n));
        // Fill upper triangle with max_dist, self-distances are 0.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    dist[[i, j]] = max_dist;
                }
            }
        }

        // BFS from every node.
        for src in 0..n {
            let mut queue = VecDeque::new();
            queue.push_back(src);
            dist[[src, src]] = 0;
            let mut visited = vec![false; n];
            visited[src] = true;

            while let Some(cur) = queue.pop_front() {
                let cur_dist = dist[[src, cur]];
                if cur_dist >= max_dist {
                    continue;
                }
                for next in 0..n {
                    if adj[[cur, next]] > 0.0 && !visited[next] {
                        visited[next] = true;
                        dist[[src, next]] = cur_dist + 1;
                        queue.push_back(next);
                    }
                }
            }
        }
        dist
    }

    /// Project raw features through `input_proj`: `[n, input_dim]` → `[n, hidden_dim]`.
    fn project_features(&self, features: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();
        let hd = self.hidden_dim;
        let id = features.ncols();
        let effective_id = id.min(self.input_dim);
        let mut out = Array2::<f64>::zeros((n, hd));
        for i in 0..n {
            for j in 0..hd {
                let mut s = 0.0_f64;
                for k in 0..effective_id {
                    s += features[[i, k]] * self.input_proj[[k, j]];
                }
                out[[i, j]] = s;
            }
        }
        out
    }

    /// Forward pass.
    ///
    /// `node_features`: `[n, input_dim]`.
    /// `adj`: `[n, n]` adjacency matrix (unweighted or weighted; `> 0` means edge).
    ///
    /// Returns `([n, output_dim], cache)`.
    pub fn forward(
        &mut self,
        node_features: &Array2<f64>,
        adj: &Array2<f64>,
    ) -> Result<(Array2<f64>, GraphormerCache), GraphTransformerError> {
        let n = adj.nrows();
        if adj.ncols() != n {
            return Err(GraphTransformerError::NonSquareAdjacency {
                rows: n,
                cols: adj.ncols(),
            });
        }
        if n == 0 {
            return Err(GraphTransformerError::EmptyGraph);
        }

        // 1. Degrees and centrality encoding.
        let (in_degrees, out_degrees) = Self::compute_degrees(adj, n);
        let centrality = self.centrality_enc.encode(&in_degrees, &out_degrees);

        // 2. Project raw features and add centrality encoding.
        let projected = self.project_features(node_features);
        let mut embedded = Array2::<f64>::zeros((n, self.hidden_dim));
        for i in 0..n {
            for j in 0..self.hidden_dim {
                embedded[[i, j]] = projected[[i, j]] + centrality[[i, j]];
            }
        }

        // 3. Compute all-pairs shortest-path distances.
        let max_dist = self.layers.first().map(|l| l.max_dist).unwrap_or(20);
        let dist = Self::all_pairs_bfs(adj, n, max_dist);

        // 4. Forward through layers.
        let mut h = embedded.clone();
        let mut layer_caches = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            let (h_new, lc) = layer.forward(&h, &dist)?;
            layer_caches.push(lc);
            h = h_new;
        }

        // 5. Output projection: [n, hidden_dim] → [n, output_dim].
        let mut output = Array2::<f64>::zeros((n, self.output_dim));
        for i in 0..n {
            for j in 0..self.output_dim {
                let mut s = 0.0_f64;
                for k in 0..self.hidden_dim {
                    s += h[[i, k]] * self.output_proj[[k, j]];
                }
                output[[i, j]] = s;
            }
        }

        let cache = GraphormerCache {
            layer_caches,
            embedded,
            input_proj: self.input_proj.clone(),
            dist,
            in_degrees,
            out_degrees,
            raw_features: node_features.clone(),
        };

        Ok((output, cache))
    }

    /// Hand-rolled backward pass. Updates all parameters via SGD.
    ///
    /// `grad_output`: `[n, output_dim]`.
    pub fn backward(&mut self, grad_output: &Array2<f64>, cache: &GraphormerCache, lr: f64) {
        let n = grad_output.nrows();
        let hidden_dim = self.hidden_dim;

        // --- Grad through output projection ---
        // dW_out = h^T @ grad_output  [hidden_dim, output_dim]
        // d_h = grad_output @ W_out^T  [n, hidden_dim]
        let h_last = if let Some(lc) = cache.layer_caches.last() {
            lc.norm2_out.clone()
        } else {
            cache.embedded.clone()
        };

        let mut dw_out = Array2::<f64>::zeros((hidden_dim, self.output_dim));
        for k in 0..hidden_dim {
            for j in 0..self.output_dim {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += h_last[[i, k]] * grad_output[[i, j]];
                }
                dw_out[[k, j]] = s;
            }
        }

        let mut d_h = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for k in 0..hidden_dim {
                let mut s = 0.0_f64;
                for j in 0..self.output_dim {
                    s += grad_output[[i, j]] * self.output_proj[[k, j]];
                }
                d_h[[i, k]] = s;
            }
        }

        // Apply output projection update.
        for k in 0..hidden_dim {
            for j in 0..self.output_dim {
                self.output_proj[[k, j]] -= lr * dw_out[[k, j]];
            }
        }

        // --- Backprop through layers (reverse order) ---
        let mut grad = d_h;
        for (layer, lc) in self.layers.iter_mut().zip(cache.layer_caches.iter()).rev() {
            grad = layer.backward(&grad, lc, lr);
        }

        // --- Grad through input projection ---
        // d_input_proj: we skip this for the centrality backward simplification
        // since centrality_enc.backward needs the degree indices (available in cache).
        // grad currently is d_embedded = d_projected + d_centrality.
        // Both paths share the same grad (add residual).
        // Update input_proj.
        let id = self.input_dim.min(cache.raw_features.ncols());
        for p in 0..id {
            for k in 0..hidden_dim {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += cache.raw_features[[i, p]] * grad[[i, k]];
                }
                self.input_proj[[p, k]] -= lr * s;
            }
        }

        // --- Grad through centrality encoding ---
        self.centrality_enc
            .backward(&grad, &cache.in_degrees, &cache.out_degrees, lr);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_adj(n: usize) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, (i + 1) % n]] = 1.0;
            a[[(i + 1) % n, i]] = 1.0;
        }
        a
    }

    #[test]
    fn test_graphormer_forward_shape() {
        let adj = ring_adj(8);
        let mut feat = Array2::<f64>::zeros((8, 8));
        for i in 0..8 {
            feat[[i, i]] = 1.0;
        }
        let mut model = GraphormerModel::new(8, 16, 4, 2, 4).expect("model");
        let (out, _cache) = model.forward(&feat, &adj).expect("forward");
        assert_eq!(out.nrows(), 8);
        assert_eq!(out.ncols(), 4);
    }

    #[test]
    fn test_graphormer_backward_runs() {
        let adj = ring_adj(4);
        let mut feat = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            feat[[i, i]] = 1.0;
        }
        let mut model = GraphormerModel::new(4, 8, 2, 1, 2).expect("model");
        let (out, cache) = model.forward(&feat, &adj).expect("forward");
        let grad = Array2::<f64>::zeros((4, 2));
        // Zero gradient = no-op update, should not panic.
        model.backward(&out.mapv(|_| 0.0), &cache, 1e-3);
        let _ = grad;
    }

    #[test]
    fn test_all_pairs_bfs_ring() {
        let adj = ring_adj(6);
        let dist = GraphormerModel::all_pairs_bfs(&adj, 6, 20);
        // Shortest path from 0 to 3 in a 6-ring is 3.
        assert_eq!(dist[[0, 3]], 3);
        // Self-distance is 0.
        assert_eq!(dist[[0, 0]], 0);
    }

    #[test]
    fn test_spatial_bias_clamped() {
        let layer = GraphormerLayer::new(8, 16, 2, 5, 1).expect("layer");
        let mut dist = Array2::<usize>::zeros((3, 3));
        dist[[0, 1]] = 100; // Should be clamped to max_dist=5.
        let bias = layer.build_spatial_bias_matrix(&dist, 3);
        // Clamped distance maps to spatial_bias[5] — just check it's finite.
        assert!(bias[[0, 1]].is_finite());
    }
}
