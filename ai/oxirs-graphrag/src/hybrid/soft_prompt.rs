//! Learnable linear projection from GNN embedding space to prompt dimension.

use scirs2_core::ndarray_ext::Array2;

/// Learnable linear projection from GNN embedding dim to prompt token dim.
pub struct SoftPromptProjector {
    /// Weight matrix W: [gnn_dim, prompt_dim]
    weights: Array2<f64>,
    /// Bias b: [prompt_dim]
    bias: Vec<f64>,
    /// Cached input (for backward pass)
    cached_input: Option<Array2<f64>>,
}

/// Training history for projector training.
#[derive(Debug, Clone, Default)]
pub struct ProjectorHistory {
    pub epoch_losses: Vec<f64>,
}

impl SoftPromptProjector {
    pub fn new(gnn_dim: usize, prompt_dim: usize, seed: u64) -> Self {
        // Xavier init
        let scale = (6.0_f64 / (gnn_dim + prompt_dim) as f64).sqrt();
        // Use a simple LCG to generate deterministic weights without rand dep.
        let mut state = seed;
        let mut next_f64 = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = 0x3FF0_0000_0000_0000_u64 | (state >> 12);
            f64::from_bits(bits) - 1.0 // [0, 1)
        };
        let total = gnn_dim * prompt_dim;
        let mut weights_data = vec![0.0f64; total];
        for v in &mut weights_data {
            *v = (next_f64() * 2.0 - 1.0) * scale;
        }
        // Safety: weights_data.len() == gnn_dim * prompt_dim, so shape is consistent.
        let weights = Array2::from_shape_vec((gnn_dim, prompt_dim), weights_data)
            .expect("shape mismatch — static sizes guaranteed by construction");
        let bias = vec![0.0; prompt_dim];
        Self {
            weights,
            bias,
            cached_input: None,
        }
    }

    /// Project GNN embeddings: input [n, gnn_dim] → output [n, prompt_dim].
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.cached_input = Some(input.clone());
        let (n, gnn_dim) = (input.nrows(), input.ncols());
        let prompt_dim = self.weights.ncols();
        debug_assert_eq!(
            gnn_dim,
            self.weights.nrows(),
            "input gnn_dim mismatch with projector"
        );
        let mut output: Array2<f64> = Array2::zeros((n, prompt_dim));
        for i in 0..n {
            for j in 0..prompt_dim {
                let mut sum = self.bias[j];
                for k in 0..gnn_dim {
                    sum += input[[i, k]] * self.weights[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }
        output
    }

    /// Backward pass: compute gradients and update weights with SGD.
    ///
    /// `d_output`: gradient of loss w.r.t. output, shape [n, prompt_dim].
    /// Returns gradient w.r.t. input for further backprop.
    ///
    /// # Panics
    ///
    /// Panics if `forward` has not been called before this method.
    pub fn backward(&mut self, d_output: &Array2<f64>, lr: f64) -> Array2<f64> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward — this is a programming error");
        let (n, gnn_dim) = (input.nrows(), input.ncols());
        let prompt_dim = self.weights.ncols();

        // d_weights[k, j] = sum_i(input[i, k] * d_output[i, j])
        let mut d_weights: Array2<f64> = Array2::zeros((gnn_dim, prompt_dim));
        for k in 0..gnn_dim {
            for j in 0..prompt_dim {
                for i in 0..n {
                    d_weights[[k, j]] += input[[i, k]] * d_output[[i, j]];
                }
            }
        }
        // d_bias[j] = sum_i d_output[i, j]
        let mut d_bias = vec![0.0f64; prompt_dim];
        for i in 0..n {
            for j in 0..prompt_dim {
                d_bias[j] += d_output[[i, j]];
            }
        }
        // d_input[i, k] = sum_j(weights[k, j] * d_output[i, j])
        let mut d_input: Array2<f64> = Array2::zeros((n, gnn_dim));
        for i in 0..n {
            for k in 0..gnn_dim {
                for j in 0..prompt_dim {
                    d_input[[i, k]] += self.weights[[k, j]] * d_output[[i, j]];
                }
            }
        }
        // SGD update
        for k in 0..gnn_dim {
            for j in 0..prompt_dim {
                self.weights[[k, j]] -= lr * d_weights[[k, j]];
            }
        }
        for (bias_val, &d) in self.bias.iter_mut().zip(d_bias.iter()) {
            *bias_val -= lr * d;
        }
        d_input
    }

    /// Returns a copy of the current weights for comparison in tests.
    pub fn weights_snapshot(&self) -> Array2<f64> {
        self.weights.clone()
    }

    pub fn prompt_dim(&self) -> usize {
        self.weights.ncols()
    }

    pub fn gnn_dim(&self) -> usize {
        self.weights.nrows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shape() {
        let mut proj = SoftPromptProjector::new(8, 16, 0);
        let input = Array2::zeros((4, 8));
        let out = proj.forward(&input);
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 16);
    }

    #[test]
    fn test_backward_updates_weights() {
        let mut proj = SoftPromptProjector::new(4, 4, 1);
        let input = Array2::from_elem((2, 4), 1.0);
        let before = proj.weights_snapshot();
        proj.forward(&input);
        let d_output = Array2::from_elem((2, 4), 0.1);
        proj.backward(&d_output, 0.01);
        let after = proj.weights_snapshot();
        // At least one weight should have changed.
        let changed =
            (0..4).any(|k| (0..4).any(|j| (before[[k, j]] - after[[k, j]]).abs() > 1e-15));
        assert!(changed, "weights should change after backward");
    }

    #[test]
    fn test_dim_accessors() {
        let proj = SoftPromptProjector::new(8, 16, 42);
        assert_eq!(proj.gnn_dim(), 8);
        assert_eq!(proj.prompt_dim(), 16);
    }
}
