//! Linear embedding projection layer (dimensionality reduction or expansion).
//!
//! `ProjectionLayer` wraps a weight matrix and bias vector together with an
//! optional activation function.  It can reduce high-dimensional embeddings
//! to a smaller space (e.g. 768 → 128) or expand them to a larger space.

/// Initialisation methods for the projection weights.
#[derive(Debug, Clone, PartialEq)]
pub enum InitMethod {
    /// Initialise all weights and biases to zero.
    Zeros,
    /// Initialise as identity-like (min-dim diagonal is 1, rest 0).
    Identity,
    /// Pseudo-random initialisation using the given seed (Xavier-style scaling).
    Random(u64),
}

/// Activation function applied element-wise to the projected output.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFn {
    /// Rectified linear unit: max(0, x).
    ReLU,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid: 1 / (1 + exp(−x)).
    Sigmoid,
    /// No activation (linear / identity).
    None,
}

/// The weight matrix and bias for a linear projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionMatrix {
    /// Number of input dimensions.
    pub input_dim: usize,
    /// Number of output dimensions.
    pub output_dim: usize,
    /// Row-major weight matrix: `weights[out][in]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of length `output_dim`.
    pub bias: Vec<f64>,
}

impl ProjectionMatrix {
    fn new_zeros(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            weights: vec![vec![0.0; input_dim]; output_dim],
            bias: vec![0.0; output_dim],
        }
    }

    fn new_identity(input_dim: usize, output_dim: usize) -> Self {
        let mut weights = vec![vec![0.0; input_dim]; output_dim];
        let min_dim = input_dim.min(output_dim);
        for (i, row) in weights.iter_mut().enumerate().take(min_dim) {
            row[i] = 1.0;
        }
        Self {
            input_dim,
            output_dim,
            weights,
            bias: vec![0.0; output_dim],
        }
    }

    /// Simple LCG-based pseudo-random weight initialisation (Xavier uniform).
    fn new_random(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        // Xavier uniform bound: sqrt(6 / (fan_in + fan_out))
        let limit = (6.0_f64 / (input_dim + output_dim) as f64).sqrt();
        let mut state = seed.wrapping_add(1);
        let mut weights = vec![vec![0.0; input_dim]; output_dim];
        for row in weights.iter_mut() {
            for w in row.iter_mut() {
                // LCG step
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Map to [0, 1)
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                // Map to [-limit, limit]
                *w = (u * 2.0 - 1.0) * limit;
            }
        }
        Self {
            input_dim,
            output_dim,
            weights,
            bias: vec![0.0; output_dim],
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// apply_activation
// ──────────────────────────────────────────────────────────────────────────────

/// Apply `act` element-wise to `val`.
fn apply_activation(val: f64, act: &ActivationFn) -> f64 {
    match act {
        ActivationFn::ReLU => val.max(0.0),
        ActivationFn::Tanh => val.tanh(),
        ActivationFn::Sigmoid => 1.0 / (1.0 + (-val).exp()),
        ActivationFn::None => val,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ProjectionLayer
// ──────────────────────────────────────────────────────────────────────────────

/// A linear projection layer: `output = activation(W·input + b)`.
#[derive(Debug, Clone)]
pub struct ProjectionLayer {
    matrix: ProjectionMatrix,
    activation: Option<ActivationFn>,
}

impl ProjectionLayer {
    /// Create a new projection layer with the given initialisation method.
    pub fn new(input_dim: usize, output_dim: usize, init: InitMethod) -> Self {
        let matrix = match init {
            InitMethod::Zeros => ProjectionMatrix::new_zeros(input_dim, output_dim),
            InitMethod::Identity => ProjectionMatrix::new_identity(input_dim, output_dim),
            InitMethod::Random(seed) => ProjectionMatrix::new_random(input_dim, output_dim, seed),
        };
        Self {
            matrix,
            activation: None,
        }
    }

    /// Attach an activation function (builder pattern).
    pub fn with_activation(mut self, activation: ActivationFn) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Project a single input vector.
    ///
    /// Returns an error string if `input.len() != self.input_dim()`.
    pub fn project(&self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            input.len(),
            self.matrix.input_dim,
            "input dimension mismatch"
        );
        let mut output = Vec::with_capacity(self.matrix.output_dim);
        for (i, row) in self.matrix.weights.iter().enumerate() {
            let mut sum = self.matrix.bias[i];
            for (w, x) in row.iter().zip(input.iter()) {
                sum += w * x;
            }
            let activated = if let Some(act) = &self.activation {
                apply_activation(sum, act)
            } else {
                sum
            };
            output.push(activated);
        }
        output
    }

    /// Project a batch of input vectors.
    pub fn project_batch(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        inputs.iter().map(|inp| self.project(inp)).collect()
    }

    /// Return the input dimensionality.
    pub fn input_dim(&self) -> usize {
        self.matrix.input_dim
    }

    /// Return the output dimensionality.
    pub fn output_dim(&self) -> usize {
        self.matrix.output_dim
    }

    /// Replace the weight matrix.
    ///
    /// Returns `Err` if dimensions do not match `(output_dim × input_dim)`.
    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>) -> Result<(), String> {
        if weights.len() != self.matrix.output_dim {
            return Err(format!(
                "expected {} output rows, got {}",
                self.matrix.output_dim,
                weights.len()
            ));
        }
        for (i, row) in weights.iter().enumerate() {
            if row.len() != self.matrix.input_dim {
                return Err(format!(
                    "row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    self.matrix.input_dim
                ));
            }
        }
        self.matrix.weights = weights;
        Ok(())
    }

    /// Replace the bias vector.
    ///
    /// Returns `Err` if `bias.len() != output_dim`.
    pub fn set_bias(&mut self, bias: Vec<f64>) -> Result<(), String> {
        if bias.len() != self.matrix.output_dim {
            return Err(format!(
                "expected bias length {}, got {}",
                self.matrix.output_dim,
                bias.len()
            ));
        }
        self.matrix.bias = bias;
        Ok(())
    }

    /// Return the total number of learnable parameters (weights + biases).
    pub fn parameter_count(&self) -> usize {
        self.matrix.input_dim * self.matrix.output_dim + self.matrix.output_dim
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── apply_activation ──────────────────────────────────────────────────────

    #[test]
    fn test_activation_relu_positive() {
        assert_eq!(apply_activation(2.5, &ActivationFn::ReLU), 2.5);
    }

    #[test]
    fn test_activation_relu_negative() {
        assert_eq!(apply_activation(-3.0, &ActivationFn::ReLU), 0.0);
    }

    #[test]
    fn test_activation_relu_zero() {
        assert_eq!(apply_activation(0.0, &ActivationFn::ReLU), 0.0);
    }

    #[test]
    fn test_activation_tanh() {
        let v = apply_activation(0.0, &ActivationFn::Tanh);
        assert!((v - 0.0).abs() < 1e-10);
        let v2 = apply_activation(1.0, &ActivationFn::Tanh);
        assert!((v2 - 1.0_f64.tanh()).abs() < 1e-10);
    }

    #[test]
    fn test_activation_sigmoid_at_zero() {
        let v = apply_activation(0.0, &ActivationFn::Sigmoid);
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_activation_sigmoid_large_positive() {
        let v = apply_activation(100.0, &ActivationFn::Sigmoid);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_sigmoid_large_negative() {
        let v = apply_activation(-100.0, &ActivationFn::Sigmoid);
        assert!(v < 1e-6);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_activation_none_is_identity() {
        assert_eq!(apply_activation(3.14, &ActivationFn::None), 3.14);
        assert_eq!(apply_activation(-7.0, &ActivationFn::None), -7.0);
    }

    // ── ProjectionLayer construction ──────────────────────────────────────────

    #[test]
    fn test_new_zeros() {
        let layer = ProjectionLayer::new(4, 2, InitMethod::Zeros);
        assert_eq!(layer.input_dim(), 4);
        assert_eq!(layer.output_dim(), 2);
        // All outputs should be 0
        let out = layer.project(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn test_new_identity_square() {
        let layer = ProjectionLayer::new(3, 3, InitMethod::Identity);
        let input = vec![1.0, 2.0, 3.0];
        let out = layer.project(&input);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_new_identity_reduce_dim() {
        let layer = ProjectionLayer::new(4, 2, InitMethod::Identity);
        let input = vec![5.0, 7.0, 9.0, 11.0];
        let out = layer.project(&input);
        // Only the first 2 inputs are copied (diagonal)
        assert!((out[0] - 5.0).abs() < 1e-10);
        assert!((out[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_new_random_produces_output() {
        let layer = ProjectionLayer::new(8, 4, InitMethod::Random(42));
        let input = vec![1.0; 8];
        let out = layer.project(&input);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_new_random_different_seeds_differ() {
        let l1 = ProjectionLayer::new(4, 2, InitMethod::Random(1));
        let l2 = ProjectionLayer::new(4, 2, InitMethod::Random(2));
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let o1 = l1.project(&input);
        let o2 = l2.project(&input);
        assert_ne!(o1, o2);
    }

    #[test]
    fn test_new_random_same_seed_same_output() {
        let l1 = ProjectionLayer::new(4, 2, InitMethod::Random(99));
        let l2 = ProjectionLayer::new(4, 2, InitMethod::Random(99));
        let input = vec![1.0, 0.5, -0.5, -1.0];
        assert_eq!(l1.project(&input), l2.project(&input));
    }

    // ── parameter_count ───────────────────────────────────────────────────────

    #[test]
    fn test_parameter_count() {
        let layer = ProjectionLayer::new(10, 5, InitMethod::Zeros);
        // 10 * 5 weights + 5 biases = 55
        assert_eq!(layer.parameter_count(), 55);
    }

    #[test]
    fn test_parameter_count_large() {
        let layer = ProjectionLayer::new(768, 128, InitMethod::Zeros);
        assert_eq!(layer.parameter_count(), 768 * 128 + 128);
    }

    // ── set_weights ───────────────────────────────────────────────────────────

    #[test]
    fn test_set_weights_valid() {
        let mut layer = ProjectionLayer::new(3, 2, InitMethod::Zeros);
        let weights = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert!(layer.set_weights(weights).is_ok());
        let out = layer.project(&[1.0, 1.0, 1.0]);
        assert!((out[0] - 6.0).abs() < 1e-10); // 1+2+3
        assert!((out[1] - 15.0).abs() < 1e-10); // 4+5+6
    }

    #[test]
    fn test_set_weights_wrong_row_count() {
        let mut layer = ProjectionLayer::new(3, 2, InitMethod::Zeros);
        let err = layer.set_weights(vec![vec![1.0, 2.0, 3.0]]);
        assert!(err.is_err());
    }

    #[test]
    fn test_set_weights_wrong_col_count() {
        let mut layer = ProjectionLayer::new(3, 2, InitMethod::Zeros);
        let err = layer.set_weights(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(err.is_err());
    }

    // ── set_bias ──────────────────────────────────────────────────────────────

    #[test]
    fn test_set_bias_valid() {
        let mut layer = ProjectionLayer::new(2, 2, InitMethod::Identity);
        assert!(layer.set_bias(vec![10.0, 20.0]).is_ok());
        let out = layer.project(&[1.0, 2.0]);
        assert!((out[0] - 11.0).abs() < 1e-10);
        assert!((out[1] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_bias_wrong_length() {
        let mut layer = ProjectionLayer::new(2, 2, InitMethod::Zeros);
        let err = layer.set_bias(vec![1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }

    // ── with_activation ───────────────────────────────────────────────────────

    #[test]
    fn test_relu_activation_clips_negative() {
        let mut layer =
            ProjectionLayer::new(2, 2, InitMethod::Identity).with_activation(ActivationFn::ReLU);
        assert!(layer.set_bias(vec![-5.0, -5.0]).is_ok());
        let out = layer.project(&[1.0, 1.0]);
        // 1 - 5 = -4 → relu → 0
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn test_tanh_activation_bounds() {
        let layer =
            ProjectionLayer::new(1, 1, InitMethod::Identity).with_activation(ActivationFn::Tanh);
        let out = layer.project(&[100.0]);
        // tanh of large positive → ~1.0
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_activation_bounds() {
        let layer =
            ProjectionLayer::new(1, 1, InitMethod::Identity).with_activation(ActivationFn::Sigmoid);
        let out0 = layer.project(&[0.0]);
        assert!((out0[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_no_activation_is_linear() {
        let layer =
            ProjectionLayer::new(1, 1, InitMethod::Identity).with_activation(ActivationFn::None);
        let out = layer.project(&[42.0]);
        assert!((out[0] - 42.0).abs() < 1e-10);
    }

    // ── project_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_project_batch_empty() {
        let layer = ProjectionLayer::new(3, 2, InitMethod::Zeros);
        let result = layer.project_batch(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_project_batch_multiple() {
        let layer = ProjectionLayer::new(2, 2, InitMethod::Identity);
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let results = layer.project_batch(&inputs);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![1.0, 2.0]);
        assert_eq!(results[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_project_batch_consistency() {
        let layer = ProjectionLayer::new(4, 2, InitMethod::Random(7));
        let input = vec![1.0, 0.5, -0.5, -1.0];
        let single = layer.project(&input);
        let batch = layer.project_batch(&[input]);
        assert_eq!(batch[0], single);
    }

    // ── dimensionality ────────────────────────────────────────────────────────

    #[test]
    fn test_reduce_dim_768_to_128() {
        let layer = ProjectionLayer::new(768, 128, InitMethod::Zeros);
        assert_eq!(layer.input_dim(), 768);
        assert_eq!(layer.output_dim(), 128);
        let input = vec![0.0; 768];
        let out = layer.project(&input);
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn test_expand_dim() {
        let layer = ProjectionLayer::new(32, 256, InitMethod::Identity);
        assert_eq!(layer.output_dim(), 256);
        let input = vec![1.0; 32];
        let out = layer.project(&input);
        assert_eq!(out.len(), 256);
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_single_dim_projection() {
        let mut layer = ProjectionLayer::new(1, 1, InitMethod::Zeros);
        assert!(layer.set_weights(vec![vec![3.0]]).is_ok());
        assert!(layer.set_bias(vec![1.0]).is_ok());
        let out = layer.project(&[2.0]);
        assert!((out[0] - 7.0).abs() < 1e-10); // 3*2 + 1
    }

    #[test]
    fn test_zero_input_with_bias() {
        let mut layer = ProjectionLayer::new(3, 2, InitMethod::Zeros);
        assert!(layer.set_bias(vec![1.0, 2.0]).is_ok());
        let out = layer.project(&[0.0, 0.0, 0.0]);
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn test_init_method_equality() {
        assert_eq!(InitMethod::Zeros, InitMethod::Zeros);
        assert_eq!(InitMethod::Random(42), InitMethod::Random(42));
        assert_ne!(InitMethod::Random(1), InitMethod::Random(2));
    }

    #[test]
    fn test_activation_fn_equality() {
        assert_eq!(ActivationFn::ReLU, ActivationFn::ReLU);
        assert_ne!(ActivationFn::ReLU, ActivationFn::Tanh);
    }
}
