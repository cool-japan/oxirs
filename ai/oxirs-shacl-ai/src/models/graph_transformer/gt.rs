//! Graph Transformer (GT) architecture (Dwivedi & Bresson 2020).
//!
//! Key features:
//! - **Laplacian eigenvector PE**: `L = I - D^{-1/2} A D^{-1/2}` → top-k eigenvectors.
//!   Random sign flip applied per forward pass for training robustness.
//! - **Sparse attention**: attention mask derived from adjacency (non-adjacent pairs
//!   receive a large negative bias, effectively masking them).
//! - **Standard transformer blocks**: multi-head self-attention + FFN + LayerNorm.
//!
//! Backward pass is hand-rolled.

use scirs2_core::ndarray_ext::{Array1, Array2};

use super::{
    attention::{AttentionCache, LayerNorm, MultiHeadAttention},
    positional_encoding::{DetRng, LaplacianPE},
    GraphTransformerError,
};

// ---------------------------------------------------------------------------
// Forward-pass cache
// ---------------------------------------------------------------------------

/// Intermediate activations for one GT layer's backward pass.
#[derive(Debug, Clone)]
pub struct GtLayerCache {
    /// Input to the layer: `[n, hidden_dim]`.
    pub x_in: Array2<f64>,
    /// MHA output (before residual): `[n, hidden_dim]`.
    pub attn_out: Array2<f64>,
    /// After residual + LN1: `[n, hidden_dim]`.
    pub norm1_out: Array2<f64>,
    /// FFN hidden (after W1 + ReLU): `[n, ffn_dim]`.
    pub ffn_hidden: Array2<f64>,
    /// FFN output (before residual): `[n, hidden_dim]`.
    pub ffn_out: Array2<f64>,
    /// After residual + LN2 (layer output): `[n, hidden_dim]`.
    pub norm2_out: Array2<f64>,
    /// MHA cache.
    pub attn_cache: AttentionCache,
}

// ---------------------------------------------------------------------------
// GT layer
// ---------------------------------------------------------------------------

/// A single Graph Transformer layer.
pub struct GraphTransformerLayer {
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
    /// Layer normalisation 1.
    pub layer_norm1: LayerNorm,
    /// Layer normalisation 2.
    pub layer_norm2: LayerNorm,
}

impl GraphTransformerLayer {
    /// Create a new GT layer.
    pub fn new(
        hidden_dim: usize,
        ffn_dim: usize,
        num_heads: usize,
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

        Ok(Self {
            attention,
            ffn_w1,
            ffn_w2,
            ffn_b1: Array1::<f64>::zeros(ffn_dim),
            ffn_b2: Array1::<f64>::zeros(hidden_dim),
            layer_norm1: LayerNorm::new(hidden_dim),
            layer_norm2: LayerNorm::new(hidden_dim),
        })
    }

    /// FFN forward: `x [n, hidden_dim]` → `[n, hidden_dim]`.
    fn ffn_forward(&self, x: &Array2<f64>, hidden: &mut Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let hidden_dim = x.ncols();
        let ffn_dim = self.ffn_w1.ncols();

        for i in 0..n {
            for j in 0..ffn_dim {
                let mut s = self.ffn_b1[j];
                for k in 0..hidden_dim {
                    s += x[[i, k]] * self.ffn_w1[[k, j]];
                }
                hidden[[i, j]] = s.max(0.0);
            }
        }

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

    /// Forward pass.
    ///
    /// `x`: `[n, hidden_dim]`.
    /// `mask`: optional adjacency-based attention mask `[n, n]` (large negative for non-edges).
    ///
    /// Returns `(output [n, hidden_dim], cache)`.
    pub fn forward(
        &mut self,
        x: &Array2<f64>,
        mask: Option<&Array2<f64>>,
    ) -> Result<(Array2<f64>, GtLayerCache), GraphTransformerError> {
        let n = x.nrows();
        let hidden_dim = x.ncols();
        let ffn_dim = self.ffn_w1.ncols();

        // MHA sub-block.
        let (attn_out, attn_cache) = self.attention.forward(x, None, mask)?;

        let mut residual1 = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                residual1[[i, j]] = x[[i, j]] + attn_out[[i, j]];
            }
        }
        let norm1_out = self.layer_norm1.forward(&residual1);

        // FFN sub-block.
        let mut ffn_hidden = Array2::<f64>::zeros((n, ffn_dim));
        let ffn_out = self.ffn_forward(&norm1_out, &mut ffn_hidden);

        let mut residual2 = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                residual2[[i, j]] = norm1_out[[i, j]] + ffn_out[[i, j]];
            }
        }
        let norm2_out = self.layer_norm2.forward(&residual2);

        let cache = GtLayerCache {
            x_in: x.clone(),
            attn_out,
            norm1_out: norm1_out.clone(),
            ffn_hidden,
            ffn_out,
            norm2_out: norm2_out.clone(),
            attn_cache,
        };

        Ok((norm2_out, cache))
    }

    /// Backward pass through one GT layer.
    pub fn backward(
        &mut self,
        grad_out: &Array2<f64>,
        cache: &GtLayerCache,
        lr: f64,
    ) -> Array2<f64> {
        let n = grad_out.nrows();
        let hidden_dim = grad_out.ncols();
        let ffn_dim = self.ffn_w1.ncols();

        // Backprop through LN2.
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
        let d_ffn_out = d_residual2.clone();
        let d_norm1_from_res2 = d_residual2;

        // Backprop through FFN.
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
        for i in 0..n {
            for k in 0..ffn_dim {
                let mut s = 0.0_f64;
                for j in 0..hidden_dim {
                    s += d_ffn_out[[i, j]] * self.ffn_w2[[k, j]];
                }
                d_hidden_pre_relu[[i, k]] = if cache.ffn_hidden[[i, k]] > 0.0 {
                    s
                } else {
                    0.0
                };
            }
        }

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

        let mut d_norm1_out = Array2::<f64>::zeros((n, hidden_dim));
        for i in 0..n {
            for j in 0..hidden_dim {
                d_norm1_out[[i, j]] = d_norm1_from_ffn[[i, j]] + d_norm1_from_res2[[i, j]];
            }
        }

        // Backprop through LN1.
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
        let d_attn_out = d_residual1.clone();
        let d_x_from_res1 = d_residual1;

        // Backprop through MHA.
        let d_x_from_attn = self.attention.backward(&d_attn_out, &cache.attn_cache, lr);

        // Total dx.
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
// Full GT model
// ---------------------------------------------------------------------------

/// Forward-pass cache for the full GT model.
pub struct GraphTransformerCache {
    /// Per-layer caches.
    pub layer_caches: Vec<GtLayerCache>,
    /// After PE projection + input projection: `[n, hidden_dim]`.
    pub embedded: Array2<f64>,
    /// Laplacian PE matrix used: `[n, pe_k]`.
    pub lap_pe: Array2<f64>,
    /// Adjacency mask: `[n, n]`.
    pub mask: Array2<f64>,
    /// Raw node features: `[n, input_dim]`.
    pub raw_features: Array2<f64>,
}

/// Full Graph Transformer model (Dwivedi & Bresson 2020).
pub struct GraphTransformerModel {
    /// Laplacian PE encoder.
    pub pe: LaplacianPE,
    /// PE projection: `[pe_k, hidden_dim]`.
    pub pe_proj: Array2<f64>,
    /// Input projection: `[input_dim, hidden_dim]`.
    pub input_proj: Array2<f64>,
    /// Transformer layers.
    pub layers: Vec<GraphTransformerLayer>,
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
    /// Number of PE eigenvectors.
    pub pe_k: usize,
    /// Seed for PE computation (random sign flip).
    pe_seed: u64,
}

impl GraphTransformerModel {
    /// Create a new GT model.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        output_dim: usize,
        pe_k: usize,
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

        let ffn_dim = hidden_dim * 4;
        let mut rng = DetRng::new(54321);

        let mut pe_proj = Array2::<f64>::zeros((pe_k, hidden_dim));
        for i in 0..pe_k {
            for j in 0..hidden_dim {
                pe_proj[[i, j]] = rng.xavier(pe_k, hidden_dim);
            }
        }

        let mut input_proj = Array2::<f64>::zeros((input_dim, hidden_dim));
        for i in 0..input_dim {
            for j in 0..hidden_dim {
                input_proj[[i, j]] = rng.xavier(input_dim, hidden_dim);
            }
        }

        let layers: Result<Vec<_>, _> = (0..num_layers)
            .map(|l| {
                GraphTransformerLayer::new(
                    hidden_dim,
                    ffn_dim,
                    num_heads,
                    (l as u64).wrapping_mul(2003).wrapping_add(7777),
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
            pe: LaplacianPE::new(pe_k),
            pe_proj,
            input_proj,
            layers,
            output_proj,
            num_heads,
            hidden_dim,
            output_dim,
            input_dim,
            pe_k,
            pe_seed: 42,
        })
    }

    /// Build adjacency-based attention mask.
    ///
    /// Non-edges get `-1e9` (effectively masking them from attention).
    fn adjacency_mask(adj: &Array2<f64>, n: usize) -> Array2<f64> {
        let mut mask = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j || adj[[i, j]] > 0.0 {
                    // Self-attention + neighbours: keep.
                    mask[[i, j]] = 0.0;
                } else {
                    mask[[i, j]] = -1.0e9;
                }
            }
        }
        mask
    }

    /// Project PE: `pe [n, pe_k] → [n, hidden_dim]`.
    fn project_pe(&self, pe: &Array2<f64>) -> Array2<f64> {
        let n = pe.nrows();
        let k = pe.ncols().min(self.pe_k);
        let mut out = Array2::<f64>::zeros((n, self.hidden_dim));
        for i in 0..n {
            for j in 0..self.hidden_dim {
                let mut s = 0.0_f64;
                for d in 0..k {
                    s += pe[[i, d]] * self.pe_proj[[d, j]];
                }
                out[[i, j]] = s;
            }
        }
        out
    }

    /// Project raw features: `[n, input_dim] → [n, hidden_dim]`.
    fn project_features(&self, features: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();
        let id = features.ncols().min(self.input_dim);
        let mut out = Array2::<f64>::zeros((n, self.hidden_dim));
        for i in 0..n {
            for j in 0..self.hidden_dim {
                let mut s = 0.0_f64;
                for k in 0..id {
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
    /// `adj`: `[n, n]` adjacency matrix.
    ///
    /// Returns `([n, output_dim], cache)`.
    pub fn forward(
        &mut self,
        node_features: &Array2<f64>,
        adj: &Array2<f64>,
    ) -> Result<(Array2<f64>, GraphTransformerCache), GraphTransformerError> {
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

        // 1. Compute Laplacian PE.
        let lap_pe = self.pe.compute(adj, self.pe_seed)?;
        // Increment seed each forward for sign-flip diversity.
        self.pe_seed = self.pe_seed.wrapping_add(1);

        // 2. Project PE and input features, then add.
        let pe_projected = self.project_pe(&lap_pe);
        let feat_projected = self.project_features(node_features);
        let mut embedded = Array2::<f64>::zeros((n, self.hidden_dim));
        for i in 0..n {
            for j in 0..self.hidden_dim {
                embedded[[i, j]] = feat_projected[[i, j]] + pe_projected[[i, j]];
            }
        }

        // 3. Build adjacency mask.
        let mask = Self::adjacency_mask(adj, n);

        // 4. Forward through layers.
        let mut h = embedded.clone();
        let mut layer_caches = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            let (h_new, lc) = layer.forward(&h, Some(&mask))?;
            layer_caches.push(lc);
            h = h_new;
        }

        // 5. Output projection.
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

        let cache = GraphTransformerCache {
            layer_caches,
            embedded,
            lap_pe,
            mask,
            raw_features: node_features.clone(),
        };

        Ok((output, cache))
    }

    /// Hand-rolled backward pass. Updates all parameters via SGD.
    pub fn backward(&mut self, grad_output: &Array2<f64>, cache: &GraphTransformerCache, lr: f64) {
        let n = grad_output.nrows();
        let hidden_dim = self.hidden_dim;

        // Grad through output projection.
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

        for k in 0..hidden_dim {
            for j in 0..self.output_dim {
                self.output_proj[[k, j]] -= lr * dw_out[[k, j]];
            }
        }

        // Backprop through layers (reverse).
        let mut grad = d_h;
        for (layer, lc) in self.layers.iter_mut().zip(cache.layer_caches.iter()).rev() {
            grad = layer.backward(&grad, lc, lr);
        }

        // Grad through input projection.
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

        // Grad through PE projection.
        let pe_k = self.pe_k.min(cache.lap_pe.ncols());
        for p in 0..pe_k {
            for k in 0..hidden_dim {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += cache.lap_pe[[i, p]] * grad[[i, k]];
                }
                self.pe_proj[[p, k]] -= lr * s;
            }
        }
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
    fn test_gt_forward_shape() {
        let adj = ring_adj(8);
        let mut feat = Array2::<f64>::zeros((8, 8));
        for i in 0..8 {
            feat[[i, i]] = 1.0;
        }
        let mut model = GraphTransformerModel::new(8, 16, 4, 2, 4, 3).expect("model");
        let (out, _cache) = model.forward(&feat, &adj).expect("forward");
        assert_eq!(out.nrows(), 8);
        assert_eq!(out.ncols(), 4);
    }

    #[test]
    fn test_gt_backward_runs() {
        let adj = ring_adj(4);
        let mut feat = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            feat[[i, i]] = 1.0;
        }
        let mut model = GraphTransformerModel::new(4, 8, 2, 1, 2, 2).expect("model");
        let (out, cache) = model.forward(&feat, &adj).expect("forward");
        // Zero grad should not panic.
        model.backward(&out.mapv(|_| 0.0), &cache, 1e-3);
    }

    #[test]
    fn test_adjacency_mask_shape() {
        let adj = ring_adj(5);
        let mask = GraphTransformerModel::adjacency_mask(&adj, 5);
        assert_eq!(mask.nrows(), 5);
        assert_eq!(mask.ncols(), 5);
        // Self-elements should be 0.
        for i in 0..5 {
            assert!((mask[[i, i]]).abs() < 1e-12);
        }
        // Non-edges in ring: e.g. (0, 2) should be masked.
        assert!(mask[[0, 2]] < -1e8);
    }

    #[test]
    fn test_gt_deterministic_init() {
        let model1 = GraphTransformerModel::new(4, 8, 2, 1, 2, 2).expect("m1");
        let model2 = GraphTransformerModel::new(4, 8, 2, 1, 2, 2).expect("m2");
        // Both models should have identical initial weights.
        let diff: f64 = model1
            .output_proj
            .iter()
            .zip(model2.output_proj.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert_eq!(
            diff, 0.0,
            "deterministic init should produce identical weights"
        );
    }
}
