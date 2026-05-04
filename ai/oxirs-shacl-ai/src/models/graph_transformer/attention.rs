//! Multi-head scaled dot-product attention with optional spatial bias.
//!
//! Used by both Graphormer (spatial bias from shortest-path distances) and
//! GT (adjacency mask for sparse attention).
//!
//! Backward pass is hand-rolled following the same pattern as:
//! - `ai/oxirs-shacl-ai/src/ml/gnn.rs:1090` (`fn backward_pass`)
//! - `ai/oxirs-shacl-ai/src/advanced_features/graph_neural_networks.rs:311`

use scirs2_core::ndarray_ext::{Array1, Array2};

use super::positional_encoding::DetRng;
use super::GraphTransformerError;

// ---------------------------------------------------------------------------
// Layer Normalisation
// ---------------------------------------------------------------------------

/// Layer normalisation over the last dimension.
///
/// For an `[n, d]` input, normalises each row independently, then applies
/// learnable per-feature affine transform `γ x̂ + β`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter (γ), shape `[hidden_dim]`.
    pub gamma: Array1<f64>,
    /// Shift parameter (β), shape `[hidden_dim]`.
    pub beta: Array1<f64>,
    /// Numerical-stability epsilon.
    pub eps: f64,
}

impl LayerNorm {
    /// Create a new `LayerNorm` with γ=1 and β=0.
    pub fn new(dim: usize) -> Self {
        let mut gamma = Array1::<f64>::zeros(dim);
        for x in gamma.iter_mut() {
            *x = 1.0;
        }
        Self {
            gamma,
            beta: Array1::<f64>::zeros(dim),
            eps: 1e-6,
        }
    }

    /// Forward: returns `[n, d]` normalised matrix.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let d = x.ncols();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            // Compute row mean and variance.
            let mean: f64 = (0..d).map(|j| x[[i, j]]).sum::<f64>() / d as f64;
            let var: f64 = (0..d).map(|j| (x[[i, j]] - mean).powi(2)).sum::<f64>() / d as f64;
            let std = (var + self.eps).sqrt();
            for j in 0..d {
                out[[i, j]] = self.gamma[j] * (x[[i, j]] - mean) / std + self.beta[j];
            }
        }
        out
    }

    /// Backward: returns gradient w.r.t. `x`, updates γ and β.
    ///
    /// Uses the standard LayerNorm backward formula:
    /// `dx = (1/std) * (dout·γ - mean(dout·γ) - x̂·mean(dout·γ·x̂))`
    ///
    /// γ/β gradients are accumulated across all rows first, then a single
    /// SGD step is applied.  This avoids the mutation-during-loop bug where
    /// earlier-row weight updates corrupt the `dout_scaled` computation for
    /// later rows.
    pub fn backward(&mut self, grad: &Array2<f64>, x: &Array2<f64>, lr: f64) -> Array2<f64> {
        let n = x.nrows();
        let d = x.ncols();
        let mut dx = Array2::<f64>::zeros((n, d));

        // Accumulators for γ and β gradients (summed over all rows).
        let mut d_gamma = Array1::<f64>::zeros(d);
        let mut d_beta = Array1::<f64>::zeros(d);

        for i in 0..n {
            let mean: f64 = (0..d).map(|j| x[[i, j]]).sum::<f64>() / d as f64;
            let var: f64 = (0..d).map(|j| (x[[i, j]] - mean).powi(2)).sum::<f64>() / d as f64;
            let std = (var + self.eps).sqrt();

            // x̂ (normalised x)
            let x_hat: Vec<f64> = (0..d).map(|j| (x[[i, j]] - mean) / std).collect();

            // dout_scaled = dout * gamma  (uses unmodified gamma throughout the loop)
            let dout_scaled: Vec<f64> = (0..d).map(|j| grad[[i, j]] * self.gamma[j]).collect();

            let mean_dout_scaled: f64 = dout_scaled.iter().sum::<f64>() / d as f64;
            let mean_dout_x_hat: f64 = dout_scaled
                .iter()
                .zip(x_hat.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / d as f64;

            for j in 0..d {
                dx[[i, j]] = (dout_scaled[j] - mean_dout_scaled - x_hat[j] * mean_dout_x_hat) / std;
            }

            // Accumulate — do NOT update self.gamma/beta here.
            for j in 0..d {
                d_gamma[j] += grad[[i, j]] * x_hat[j];
                d_beta[j] += grad[[i, j]];
            }
        }

        // Single SGD update after all rows have been processed.
        for j in 0..d {
            self.gamma[j] -= lr * d_gamma[j];
            self.beta[j] -= lr * d_beta[j];
        }

        dx
    }
}

// ---------------------------------------------------------------------------
// Weight matrix helper
// ---------------------------------------------------------------------------

fn xavier_matrix(rows: usize, cols: usize, rng: &mut DetRng) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            m[[i, j]] = rng.xavier(rows, cols);
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Multi-head self-attention
// ---------------------------------------------------------------------------

/// Cache of intermediate values needed by the backward pass.
#[derive(Debug, Clone)]
pub struct AttentionCache {
    /// Input `x`: `[n, hidden_dim]`.
    pub x: Array2<f64>,
    /// Query projections: `[n, hidden_dim]`.
    pub q: Array2<f64>,
    /// Key projections: `[n, hidden_dim]`.
    pub k: Array2<f64>,
    /// Value projections: `[n, hidden_dim]`.
    pub v: Array2<f64>,
    /// Attention weights (after softmax): `[n, n]`.
    pub attn: Array2<f64>,
    /// Pre-projection output (concat of heads): `[n, hidden_dim]`.
    pub pre_proj: Array2<f64>,
}

/// Multi-head scaled dot-product attention with optional bias and mask.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Total hidden dimension = `num_heads * head_dim`.
    pub hidden_dim: usize,
    /// Query weight: `[hidden_dim, hidden_dim]`.
    pub w_q: Array2<f64>,
    /// Key weight: `[hidden_dim, hidden_dim]`.
    pub w_k: Array2<f64>,
    /// Value weight: `[hidden_dim, hidden_dim]`.
    pub w_v: Array2<f64>,
    /// Output projection: `[hidden_dim, hidden_dim]`.
    pub w_o: Array2<f64>,
}

impl MultiHeadAttention {
    /// Create a new MHA layer with Xavier-uniform init.
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        seed: u64,
    ) -> Result<Self, GraphTransformerError> {
        if num_heads == 0 || head_dim == 0 {
            return Err(GraphTransformerError::Config(
                "num_heads and head_dim must be > 0".to_string(),
            ));
        }
        let hidden_dim = num_heads * head_dim;
        let mut rng = DetRng::new(seed);
        Ok(Self {
            num_heads,
            head_dim,
            hidden_dim,
            w_q: xavier_matrix(hidden_dim, hidden_dim, &mut rng),
            w_k: xavier_matrix(hidden_dim, hidden_dim, &mut rng),
            w_v: xavier_matrix(hidden_dim, hidden_dim, &mut rng),
            w_o: xavier_matrix(hidden_dim, hidden_dim, &mut rng),
        })
    }

    /// Linear projection: `x [n, d_in] @ W [d_in, d_out]` → `[n, d_out]`.
    fn proj(&self, x: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let d_out = w.ncols();
        let d_in = w.nrows();
        let mut out = Array2::<f64>::zeros((n, d_out));
        for i in 0..n {
            for k in 0..d_out {
                let mut s = 0.0_f64;
                for j in 0..d_in {
                    s += x[[i, j]] * w[[j, k]];
                }
                out[[i, k]] = s;
            }
        }
        out
    }

    /// Softmax over the last axis (rows) with numerical stabilisation.
    fn softmax_rows(scores: &mut Array2<f64>) {
        let n = scores.nrows();
        let m = scores.ncols();
        for i in 0..n {
            let max = (0..m)
                .map(|j| scores[[i, j]])
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0_f64;
            for j in 0..m {
                scores[[i, j]] = (scores[[i, j]] - max).exp();
                sum += scores[[i, j]];
            }
            let sum = sum + 1e-12;
            for j in 0..m {
                scores[[i, j]] /= sum;
            }
        }
    }

    /// Forward pass.
    ///
    /// `x`: `[n, hidden_dim]` node features.
    /// `spatial_bias`: optional `[n, n]` matrix added to all-head attention scores.
    /// `mask`: optional `[n, n]` matrix added to scores (use large negative for masking).
    ///
    /// Returns `(output [n, hidden_dim], cache)`.
    pub fn forward(
        &self,
        x: &Array2<f64>,
        spatial_bias: Option<&Array2<f64>>,
        mask: Option<&Array2<f64>>,
    ) -> Result<(Array2<f64>, AttentionCache), GraphTransformerError> {
        let n = x.nrows();
        if n == 0 {
            return Err(GraphTransformerError::EmptyGraph);
        }
        let d = self.hidden_dim;
        if x.ncols() != d {
            return Err(GraphTransformerError::DimensionMismatch {
                expected: d,
                actual: x.ncols(),
            });
        }

        let q = self.proj(x, &self.w_q);
        let k = self.proj(x, &self.w_k);
        let v = self.proj(x, &self.w_v);

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        // Accumulate output across heads: [n, hidden_dim].
        let mut pre_proj = Array2::<f64>::zeros((n, d));
        // We'll store averaged attention weights over heads for the cache.
        let mut attn_sum = Array2::<f64>::zeros((n, n));

        for h in 0..self.num_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            // Extract per-head slices of Q, K, V: [n, head_dim].
            let mut scores = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0_f64;
                    for d_idx in start..end {
                        dot += q[[i, d_idx]] * k[[j, d_idx]];
                    }
                    let mut s = dot * scale;
                    if let Some(bias) = spatial_bias {
                        s += bias[[i, j]];
                    }
                    if let Some(m) = mask {
                        s += m[[i, j]];
                    }
                    scores[[i, j]] = s;
                }
            }

            Self::softmax_rows(&mut scores);

            // Accumulate into attn_sum for cache.
            for i in 0..n {
                for j in 0..n {
                    attn_sum[[i, j]] += scores[[i, j]];
                }
            }

            // Weighted sum of values → head output slice: pre_proj[:, start..end].
            for i in 0..n {
                for d_idx in start..end {
                    let mut s = 0.0_f64;
                    for j in 0..n {
                        s += scores[[i, j]] * v[[j, d_idx]];
                    }
                    pre_proj[[i, d_idx]] = s;
                }
            }
        }

        // Average attention weights over heads.
        let num_heads_f = self.num_heads as f64;
        for i in 0..n {
            for j in 0..n {
                attn_sum[[i, j]] /= num_heads_f;
            }
        }

        // Output projection.
        let output = self.proj(&pre_proj, &self.w_o);

        let cache = AttentionCache {
            x: x.clone(),
            q,
            k,
            v,
            attn: attn_sum,
            pre_proj,
        };

        Ok((output, cache))
    }

    /// Hand-rolled backward pass. Updates `w_q`, `w_k`, `w_v`, `w_o` in place.
    ///
    /// Returns gradient w.r.t. input `x`: `[n, hidden_dim]`.
    ///
    /// Steps (following the pattern in `ml/gnn.rs:1090`):
    /// 1. Grad through `w_o`: `dW_o = pre_proj^T @ grad_out`.
    /// 2. Grad through `pre_proj`: `d_pre = grad_out @ w_o^T`.
    /// 3. For each head, recover per-head softmax scores and value-slice;
    ///    compute softmax gradient, then grad through Q/K/V projections.
    pub fn backward(
        &mut self,
        grad_output: &Array2<f64>,
        cache: &AttentionCache,
        lr: f64,
    ) -> Array2<f64> {
        let n = grad_output.nrows();
        let d = self.hidden_dim;
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        // --- Step 1: grad through w_o ---
        // dW_o = pre_proj^T @ grad_output   [d, d]
        let mut dw_o = Array2::<f64>::zeros((d, d));
        for i in 0..n {
            for j in 0..d {
                for k in 0..d {
                    dw_o[[j, k]] += cache.pre_proj[[i, j]] * grad_output[[i, k]];
                }
            }
        }

        // d_pre_proj = grad_output @ w_o^T   [n, d]
        let mut d_pre_proj = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let mut s = 0.0_f64;
                for k in 0..d {
                    s += grad_output[[i, k]] * self.w_o[[j, k]];
                }
                d_pre_proj[[i, j]] = s;
            }
        }

        // Initialise gradient accumulators for Q/K/V weights.
        let mut dw_q = Array2::<f64>::zeros((d, d));
        let mut dw_k = Array2::<f64>::zeros((d, d));
        let mut dw_v = Array2::<f64>::zeros((d, d));
        // Gradient w.r.t. the input x (accumulated over heads).
        let mut dx = Array2::<f64>::zeros((n, d));

        // --- Step 2: per-head backward ---
        for h in 0..self.num_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            // Recompute per-head attention scores (needed for softmax grad).
            let mut scores = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0_f64;
                    for d_idx in start..end {
                        dot += cache.q[[i, d_idx]] * cache.k[[j, d_idx]];
                    }
                    scores[[i, j]] = dot * scale;
                }
            }
            Self::softmax_rows(&mut scores); // attn_h [n, n]

            // Grad w.r.t. value pre-projection slice: d_v = attn_h^T @ d_pre_h
            // d_pre_h is the [n, head_dim] slice of d_pre_proj.
            let mut d_v_slice = Array2::<f64>::zeros((n, self.head_dim));
            for j in 0..n {
                for d_idx in 0..self.head_dim {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        s += scores[[i, j]] * d_pre_proj[[i, start + d_idx]];
                    }
                    d_v_slice[[j, d_idx]] = s;
                }
            }

            // Grad w.r.t. attention scores via softmax backward.
            // d_scores[i,j] = attn[i,j] * (d_attn_v[i,j] - sum_j(d_attn_v[i,j]*attn[i,j]))
            // where d_attn_v[i,j] = sum_{d'} d_pre_h[i,d'] * v[j, start+d']
            let mut d_attn_v = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0_f64;
                    for d_idx in 0..self.head_dim {
                        s += d_pre_proj[[i, start + d_idx]] * cache.v[[j, start + d_idx]];
                    }
                    d_attn_v[[i, j]] = s;
                }
            }

            let mut d_scores = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                let weighted_sum: f64 = (0..n).map(|j| d_attn_v[[i, j]] * scores[[i, j]]).sum();
                for j in 0..n {
                    d_scores[[i, j]] = scores[[i, j]] * (d_attn_v[[i, j]] - weighted_sum) * scale;
                }
            }

            // Grad w.r.t. Q: d_q[i, start..end] += sum_j d_scores[i,j] * k[j, start..end]
            let mut d_q_slice = Array2::<f64>::zeros((n, self.head_dim));
            for i in 0..n {
                for d_idx in 0..self.head_dim {
                    let mut s = 0.0_f64;
                    for j in 0..n {
                        s += d_scores[[i, j]] * cache.k[[j, start + d_idx]];
                    }
                    d_q_slice[[i, d_idx]] = s;
                }
            }

            // Grad w.r.t. K: d_k[j, start..end] += sum_i d_scores[i,j] * q[i, start..end]
            let mut d_k_slice = Array2::<f64>::zeros((n, self.head_dim));
            for j in 0..n {
                for d_idx in 0..self.head_dim {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        s += d_scores[[i, j]] * cache.q[[i, start + d_idx]];
                    }
                    d_k_slice[[j, d_idx]] = s;
                }
            }

            // Accumulate into full-width dW_q, dW_k, dW_v and dx.
            // dW_q[*, start..end] += x^T @ d_q_slice  (shape [d, head_dim])
            for p in 0..d {
                for d_idx in 0..self.head_dim {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        s += cache.x[[i, p]] * d_q_slice[[i, d_idx]];
                    }
                    dw_q[[p, start + d_idx]] += s;
                }
            }

            // dW_k similarly.
            for p in 0..d {
                for d_idx in 0..self.head_dim {
                    let mut s = 0.0_f64;
                    for i in 0..n {
                        s += cache.x[[i, p]] * d_k_slice[[i, d_idx]];
                    }
                    dw_k[[p, start + d_idx]] += s;
                }
            }

            // dW_v[*, start..end] += x^T @ d_v_slice.
            for p in 0..d {
                for d_idx in 0..self.head_dim {
                    let mut s = 0.0_f64;
                    for j in 0..n {
                        s += cache.x[[j, p]] * d_v_slice[[j, d_idx]];
                    }
                    dw_v[[p, start + d_idx]] += s;
                }
            }

            // Gradient w.r.t. x from Q path: dx += d_q @ w_q[*, start..end]^T
            for i in 0..n {
                for p in 0..d {
                    let mut s = 0.0_f64;
                    for d_idx in 0..self.head_dim {
                        s += d_q_slice[[i, d_idx]] * self.w_q[[p, start + d_idx]];
                    }
                    dx[[i, p]] += s;
                }
            }

            // Gradient w.r.t. x from K path: dx += d_k @ w_k[*, start..end]^T
            for i in 0..n {
                for p in 0..d {
                    let mut s = 0.0_f64;
                    for d_idx in 0..self.head_dim {
                        s += d_k_slice[[i, d_idx]] * self.w_k[[p, start + d_idx]];
                    }
                    dx[[i, p]] += s;
                }
            }

            // Gradient w.r.t. x from V path: dx += d_v @ w_v[*, start..end]^T
            for i in 0..n {
                for p in 0..d {
                    let mut s = 0.0_f64;
                    for d_idx in 0..self.head_dim {
                        s += d_v_slice[[i, d_idx]] * self.w_v[[p, start + d_idx]];
                    }
                    dx[[i, p]] += s;
                }
            }
        }

        // Apply weight updates (SGD).
        for i in 0..d {
            for j in 0..d {
                self.w_o[[i, j]] -= lr * dw_o[[i, j]];
                self.w_q[[i, j]] -= lr * dw_q[[i, j]];
                self.w_k[[i, j]] -= lr * dw_k[[i, j]];
                self.w_v[[i, j]] -= lr * dw_v[[i, j]];
            }
        }

        dx
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mha(n_heads: usize, head_dim: usize) -> MultiHeadAttention {
        MultiHeadAttention::new(n_heads, head_dim, 42).expect("ok")
    }

    fn eye(n: usize, d: usize, val: f64) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, d));
        for i in 0..n.min(d) {
            m[[i, i]] = val;
        }
        m
    }

    #[test]
    fn test_mha_forward_shape() {
        let mha = make_mha(2, 4); // hidden_dim = 8
        let x = Array2::<f64>::zeros((5, 8));
        let (out, _cache) = mha.forward(&x, None, None).expect("ok");
        assert_eq!(out.nrows(), 5);
        assert_eq!(out.ncols(), 8);
    }

    #[test]
    fn test_mha_backward_returns_correct_shape() {
        let mut mha = make_mha(2, 4);
        let x = eye(4, 8, 1.0);
        let (out, cache) = mha.forward(&x, None, None).expect("ok");
        let grad = Array2::<f64>::zeros((4, 8));
        let dx = mha.backward(&grad, &cache, 1e-3);
        assert_eq!(dx.nrows(), out.nrows());
        assert_eq!(dx.ncols(), out.ncols());
    }

    #[test]
    fn test_layer_norm_near_zero_mean_unit_std() {
        let ln = LayerNorm::new(8);
        let mut x = Array2::<f64>::zeros((4, 8));
        // Fill with distinct values.
        for i in 0..4 {
            for j in 0..8 {
                x[[i, j]] = (i * 8 + j) as f64;
            }
        }
        let out = ln.forward(&x);
        for i in 0..4 {
            let mean: f64 = (0..8).map(|j| out[[i, j]]).sum::<f64>() / 8.0;
            let var: f64 = (0..8).map(|j| (out[[i, j]] - mean).powi(2)).sum::<f64>() / 8.0;
            assert!(mean.abs() < 1e-9, "mean not ~0: {mean}");
            assert!((var - 1.0).abs() < 0.05, "var not ~1: {var}");
        }
    }

    #[test]
    fn test_layer_norm_backward_shape() {
        let mut ln = LayerNorm::new(8);
        let mut x = Array2::<f64>::zeros((4, 8));
        for i in 0..4 {
            for j in 0..8 {
                x[[i, j]] = (i * 8 + j) as f64 * 0.1;
            }
        }
        let grad = Array2::<f64>::zeros((4, 8));
        let dx = ln.backward(&grad, &x, 1e-3);
        assert_eq!(dx.nrows(), 4);
        assert_eq!(dx.ncols(), 8);
    }
}
