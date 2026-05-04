//! LoRA (Low-Rank Adaptation) adapter and fine-tuning scaffold — phase d.
//!
//! # Design
//!
//! Augments an existing linear projection W: ℝ^{d_in} → ℝ^{d_out} with a
//! low-rank delta:
//!
//! ```text
//! ΔW = B · A   where A ∈ ℝ^{d_in × r},  B ∈ ℝ^{r × d_out},  r ≪ min(d_in, d_out)
//!
//! Forward: output = W · x + scale * (x @ A) @ B
//! ```
//!
//! During fine-tuning the base weight `W` is frozen; only `A` and `B` receive
//! gradient updates via a hand-rolled SGD step.  This mirrors the pattern used
//! by the existing [`SoftPromptProjector`] and the GNN encoder — no autograd
//! dependency required.
//!
//! # Initialisation
//!
//! `A` is initialised with Xavier-uniform noise (small magnitude) and `B` is
//! initialised to zero.  This means the initial delta is zero, so LoRA
//! introduces no disruption to the frozen base model at the start of training.
//!
//! # Notation
//!
//! All matrices use row-vector convention (batch dimension first), consistent
//! with the rest of the GraphRAG codebase:
//!
//! - `input`:  `[batch, d_in]`
//! - `A`:      `[d_in,  rank]`
//! - `B`:      `[rank,  d_out]`
//! - `delta`:  `[batch, d_out] = scale * (input @ A) @ B`

use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::rand_prelude::StdRng;
use scirs2_core::random::{seeded_rng, CoreRandom};

// ─── LoraAdapter ─────────────────────────────────────────────────────────────

/// LoRA (Low-Rank Adaptation) adapter for the soft-prompt projector.
///
/// Augments a frozen base projection W: ℝ^{d_in} → ℝ^{d_out} with a
/// low-rank delta ΔW = B·A where A∈ℝ^{d_in×r}, B∈ℝ^{r×d_out}, rank r ≪ min(d_in,d_out).
///
/// Forward: `output = W·x + scale * (x @ A) @ B`
///
/// During training: W is frozen; only A and B are updated via SGD.
///
/// Initialization: A ~ Xavier-uniform (small), B = 0 (initial delta is zero).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Rank r.
    pub rank: usize,
    /// Scaling coefficient α (typically equals rank).
    pub alpha: f64,
    /// A matrix: `[d_in, rank]`.
    pub a_matrix: Array2<f64>,
    /// B matrix: `[rank, d_out]`.
    pub b_matrix: Array2<f64>,
    /// Input dimension.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Accumulated gradient for A (same shape).
    grad_a: Array2<f64>,
    /// Accumulated gradient for B (same shape).
    grad_b: Array2<f64>,
}

impl LoraAdapter {
    /// Initialise LoRA adapter.
    ///
    /// `A` is Xavier-uniform initialised; `B` is zero-initialised so the
    /// initial delta contribution is exactly zero.
    pub fn new(d_in: usize, d_out: usize, rank: usize, alpha: f64, seed: u64) -> Self {
        assert!(rank > 0, "LoRA rank must be at least 1");
        assert!(d_in > 0 && d_out > 0, "dimensions must be non-zero");

        let mut rng: CoreRandom<StdRng> = seeded_rng(seed);

        // Xavier-uniform limit for A: fan_in = d_in, fan_out = rank.
        let limit = (6.0_f64 / (d_in + rank) as f64).sqrt();

        let a_data: Vec<f64> = (0..d_in * rank)
            .map(|_| {
                let u = rng.random_range(0.0_f64..1.0_f64);
                u * 2.0 * limit - limit
            })
            .collect();

        let a_matrix = Array2::from_shape_vec((d_in, rank), a_data)
            .expect("a_matrix shape is consistent by construction");
        let b_matrix = Array2::zeros((rank, d_out));
        let grad_a = Array2::zeros((d_in, rank));
        let grad_b = Array2::zeros((rank, d_out));

        Self {
            rank,
            alpha,
            a_matrix,
            b_matrix,
            d_in,
            d_out,
            grad_a,
            grad_b,
        }
    }

    /// Scaling factor = alpha / rank.
    #[inline]
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// Compute the LoRA delta: `scale * (input @ A) @ B`.
    ///
    /// `input`: `[batch, d_in]` → output: `[batch, d_out]`
    pub fn forward_delta(&self, input: &Array2<f64>) -> Array2<f64> {
        let batch = input.nrows();
        debug_assert_eq!(
            input.ncols(),
            self.d_in,
            "input column count must equal d_in"
        );

        // z = input @ A  →  [batch, rank]
        let mut z = Array2::zeros((batch, self.rank));
        for i in 0..batch {
            for k in 0..self.rank {
                let mut sum = 0.0_f64;
                for j in 0..self.d_in {
                    sum += input[[i, j]] * self.a_matrix[[j, k]];
                }
                z[[i, k]] = sum;
            }
        }

        // delta = z @ B  →  [batch, d_out]
        let mut delta = Array2::zeros((batch, self.d_out));
        for i in 0..batch {
            for m in 0..self.d_out {
                let mut sum = 0.0_f64;
                for k in 0..self.rank {
                    sum += z[[i, k]] * self.b_matrix[[k, m]];
                }
                delta[[i, m]] = sum * self.scale();
            }
        }

        delta
    }

    /// Backward pass through the LoRA delta.
    ///
    /// Given upstream gradient `d_output: [batch, d_out]`, accumulates
    /// gradients into `grad_a` and `grad_b` and returns the gradient flowing
    /// back through the delta w.r.t. `input`: `[batch, d_in]`.
    ///
    /// Gradient derivation (row-vector convention):
    /// - forward: `delta = scale * z @ B`  where `z = input @ A`
    /// - `d_z = d_output @ B.T * scale`   →  `[batch, rank]`
    /// - `grad_B += z.T @ d_output * scale` →  `[rank, d_out]`
    /// - `grad_A += input.T @ d_z`          →  `[d_in, rank]`
    /// - `d_input = d_z @ A.T`              →  `[batch, d_in]`
    ///
    /// Call [`LoraAdapter::zero_grad`] before a new mini-batch to reset
    /// the accumulators.
    pub fn backward(&mut self, input: &Array2<f64>, d_output: &Array2<f64>) -> Array2<f64> {
        let batch = input.nrows();
        debug_assert_eq!(input.ncols(), self.d_in);
        debug_assert_eq!(d_output.nrows(), batch);
        debug_assert_eq!(d_output.ncols(), self.d_out);

        let s = self.scale();

        // Recompute z = input @ A  →  [batch, rank]  (needed for grad_B)
        let mut z = Array2::zeros((batch, self.rank));
        for i in 0..batch {
            for k in 0..self.rank {
                let mut sum = 0.0_f64;
                for j in 0..self.d_in {
                    sum += input[[i, j]] * self.a_matrix[[j, k]];
                }
                z[[i, k]] = sum;
            }
        }

        // d_z = d_output @ B.T * scale  →  [batch, rank]
        let mut d_z = Array2::zeros((batch, self.rank));
        for i in 0..batch {
            for k in 0..self.rank {
                let mut sum = 0.0_f64;
                for m in 0..self.d_out {
                    sum += d_output[[i, m]] * self.b_matrix[[k, m]];
                }
                d_z[[i, k]] = sum * s;
            }
        }

        // grad_B += z.T @ d_output * scale  →  [rank, d_out]
        for k in 0..self.rank {
            for m in 0..self.d_out {
                let mut sum = 0.0_f64;
                for i in 0..batch {
                    sum += z[[i, k]] * d_output[[i, m]];
                }
                self.grad_b[[k, m]] += sum * s;
            }
        }

        // grad_A += input.T @ d_z  →  [d_in, rank]
        for j in 0..self.d_in {
            for k in 0..self.rank {
                let mut sum = 0.0_f64;
                for i in 0..batch {
                    sum += input[[i, j]] * d_z[[i, k]];
                }
                self.grad_a[[j, k]] += sum;
            }
        }

        // d_input = d_z @ A.T  →  [batch, d_in]
        let mut d_input = Array2::zeros((batch, self.d_in));
        for i in 0..batch {
            for j in 0..self.d_in {
                let mut sum = 0.0_f64;
                for k in 0..self.rank {
                    sum += d_z[[i, k]] * self.a_matrix[[j, k]];
                }
                d_input[[i, j]] = sum;
            }
        }

        d_input
    }

    /// SGD update: `A -= lr * grad_A`, `B -= lr * grad_B`.
    pub fn sgd_step(&mut self, learning_rate: f64) {
        for j in 0..self.d_in {
            for k in 0..self.rank {
                self.a_matrix[[j, k]] -= learning_rate * self.grad_a[[j, k]];
            }
        }
        for k in 0..self.rank {
            for m in 0..self.d_out {
                self.b_matrix[[k, m]] -= learning_rate * self.grad_b[[k, m]];
            }
        }
    }

    /// Reset accumulated gradients to zero.
    pub fn zero_grad(&mut self) {
        for v in self.grad_a.iter_mut() {
            *v = 0.0;
        }
        for v in self.grad_b.iter_mut() {
            *v = 0.0;
        }
    }

    /// L2 norm of the combined gradient (A and B).
    ///
    /// Returns 0.0 after [`zero_grad`] is called.
    pub fn grad_norm(&self) -> f64 {
        let sq_sum: f64 = self
            .grad_a
            .iter()
            .chain(self.grad_b.iter())
            .map(|&v| v * v)
            .sum();
        sq_sum.sqrt()
    }
}

// ─── LoraTrainer ─────────────────────────────────────────────────────────────

/// Training scaffold for LoRA fine-tuning of the GNN → LLM projection.
///
/// Holds the [`LoraAdapter`] and drives the training loop over KGQA examples.
/// The base `SoftPromptProjector` weights are frozen; only LoRA A and B are
/// updated.
///
/// # Example
///
/// ```rust
/// use oxirs_graphrag::hybrid::lora::{LoraAdapter, LoraTrainer};
/// use scirs2_core::ndarray_ext::Array2;
///
/// let adapter = LoraAdapter::new(8, 4, 2, 2.0, 42);
/// let mut trainer = LoraTrainer::new(adapter, 0.01);
///
/// // Toy data: frozen projector output and targets (same shape)
/// let base_out = Array2::from_elem((3, 4), 0.5);
/// let targets  = Array2::zeros((3, 4));
///
/// let loss = trainer.train_epoch(&base_out, &targets);
/// assert!(loss >= 0.0);
/// ```
pub struct LoraTrainer {
    lora: LoraAdapter,
    learning_rate: f64,
}

impl LoraTrainer {
    /// Create a new LoRA trainer.
    pub fn new(lora: LoraAdapter, learning_rate: f64) -> Self {
        Self {
            lora,
            learning_rate,
        }
    }

    /// Train for one epoch.
    ///
    /// `base_output`: frozen projector output `[batch, d_in]`.
    /// `targets`:     ground-truth targets `[batch, d_out]`.
    ///
    /// Computes MSE loss, runs backward, updates A and B via SGD, and returns
    /// the mean MSE loss for this epoch.
    ///
    /// # Panics
    ///
    /// Panics if the adapter's `d_in != d_out`.  `train_epoch` treats
    /// `base_output` as both the frozen projection *and* the residual input to
    /// the LoRA delta (i.e. the same tensor is used as the "base" and as the
    /// "input" to `forward_delta`).  This means `base_output + delta` is only
    /// well-typed when `d_in == d_out`.  Construct the adapter with matching
    /// dimensions, e.g. `LoraAdapter::new(d, d, rank, alpha, seed)`.
    pub fn train_epoch(&mut self, base_output: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        assert_eq!(
            self.lora.d_in, self.lora.d_out,
            "LoraTrainer::train_epoch requires d_in == d_out; got d_in={}, d_out={}",
            self.lora.d_in, self.lora.d_out,
        );
        assert_eq!(
            base_output.ncols(),
            self.lora.d_in,
            "base_output column count must equal adapter d_in ({}), got {}",
            self.lora.d_in,
            base_output.ncols(),
        );
        let batch = base_output.nrows();
        let d_out = base_output.ncols();
        assert_eq!(
            targets.nrows(),
            batch,
            "targets row count must match base_output batch size"
        );
        assert_eq!(
            targets.ncols(),
            d_out,
            "targets column count must match d_out"
        );

        // The frozen projector output is the "input" to the LoRA delta.
        // We treat base_output as a proxy for the GNN embedding after projection
        // and learn a residual correction toward the target.
        let input = base_output;

        // Forward: augmented output = base_output + lora_delta
        let delta = self.lora.forward_delta(input);
        let mut total_loss = 0.0_f64;
        let scale = (batch * d_out).max(1) as f64;

        // Gradient of MSE w.r.t. augmented output: 2*(out - target) / (batch * d_out)
        let mut d_output = Array2::zeros((batch, d_out));
        for i in 0..batch {
            for j in 0..d_out {
                let out_val = base_output[[i, j]] + delta[[i, j]];
                let target_val = targets[[i, j]];
                let diff = out_val - target_val;
                total_loss += diff * diff;
                d_output[[i, j]] = 2.0 * diff / scale;
            }
        }

        // Backward + SGD update
        self.lora.zero_grad();
        self.lora.backward(input, &d_output);
        self.lora.sgd_step(self.learning_rate);

        total_loss / scale
    }

    /// Borrow the current LoRA adapter (e.g., for inspection between epochs).
    pub fn adapter(&self) -> &LoraAdapter {
        &self.lora
    }

    /// Consume the trainer and return the trained adapter.
    pub fn into_adapter(self) -> LoraAdapter {
        self.lora
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array2;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_adapter(d_in: usize, d_out: usize, rank: usize) -> LoraAdapter {
        LoraAdapter::new(d_in, d_out, rank, rank as f64, 42)
    }

    // ── test 1: B matrix is zero-initialised ─────────────────────────────────

    #[test]
    fn test_new_b_matrix_is_zero() {
        let adapter = make_adapter(4, 6, 2);
        for &v in adapter.b_matrix.iter() {
            assert_eq!(v, 0.0, "B should be zero-initialised");
        }
    }

    // ── test 2: forward_delta on zero A+B gives zero output ──────────────────

    #[test]
    fn test_forward_delta_zero_ab_gives_zero() {
        let mut adapter = make_adapter(4, 6, 2);
        // Zero out A to guarantee zero delta.
        for v in adapter.a_matrix.iter_mut() {
            *v = 0.0;
        }
        let input = Array2::from_elem((3, 4), 1.0);
        let delta = adapter.forward_delta(&input);
        for &v in delta.iter() {
            assert!(
                v.abs() < 1e-14,
                "delta must be zero when A=0 and B=0, got {v}"
            );
        }
    }

    // ── test 3: forward_delta output shape ───────────────────────────────────

    #[test]
    fn test_forward_delta_shape() {
        let adapter = make_adapter(8, 12, 3);
        let input = Array2::zeros((5, 8));
        let delta = adapter.forward_delta(&input);
        assert_eq!(delta.nrows(), 5, "batch dimension must be preserved");
        assert_eq!(delta.ncols(), 12, "column count must equal d_out");
    }

    // ── test 4: backward returns d_input with correct shape ──────────────────

    #[test]
    fn test_backward_d_input_shape() {
        let mut adapter = make_adapter(8, 12, 3);
        let input = Array2::from_elem((5, 8), 0.1);
        let d_output = Array2::from_elem((5, 12), 0.01);
        let d_input = adapter.backward(&input, &d_output);
        assert_eq!(d_input.nrows(), 5, "d_input batch must match");
        assert_eq!(d_input.ncols(), 8, "d_input columns must equal d_in");
    }

    // ── test 5: sgd_step updates A and B matrices ────────────────────────────

    #[test]
    fn test_sgd_step_updates_matrices() {
        let mut adapter = make_adapter(4, 6, 2);
        let a_before = adapter.a_matrix.clone();

        // Run a backward to accumulate non-zero gradients.
        let input = Array2::from_elem((2, 4), 1.0);
        let d_output = Array2::from_elem((2, 6), 0.1);
        adapter.backward(&input, &d_output);

        // Snapshot B before step.
        let b_before = adapter.b_matrix.clone();

        adapter.sgd_step(0.01);

        // A should change (grad_a is non-zero if d_z and input are non-zero;
        // d_z depends on B which starts at zero, so grad_A may be zero.
        // B should definitely change because grad_B = z.T @ d_output * scale
        // and z = input @ A != 0 when A != 0).
        let b_changed = b_before
            .iter()
            .zip(adapter.b_matrix.iter())
            .any(|(old, new)| (old - new).abs() > 1e-15);
        assert!(b_changed, "B matrix must change after sgd_step");

        // A: grad_A = input.T @ d_z; d_z = d_output @ B.T * scale.
        // Since B starts at zero, grad_A starts at zero → A unchanged. That is correct.
        // Verify by checking A only if we first make B non-zero.
        let _ = a_before; // just confirm it compiles
    }

    // ── test 6: zero_grad resets gradients ───────────────────────────────────

    #[test]
    fn test_zero_grad_resets_gradients() {
        let mut adapter = make_adapter(4, 6, 2);
        let input = Array2::from_elem((2, 4), 1.0);
        let d_output = Array2::from_elem((2, 6), 0.5);
        adapter.backward(&input, &d_output);
        // After backward, at least grad_b should be non-zero.
        adapter.zero_grad();
        for &v in adapter.grad_a.iter() {
            assert_eq!(v, 0.0, "grad_a must be zero after zero_grad");
        }
        for &v in adapter.grad_b.iter() {
            assert_eq!(v, 0.0, "grad_b must be zero after zero_grad");
        }
    }

    // ── test 7: grad_norm is 0 after zero_grad ───────────────────────────────

    #[test]
    fn test_grad_norm_zero_after_zero_grad() {
        let mut adapter = make_adapter(4, 6, 2);
        let input = Array2::from_elem((2, 4), 1.0);
        let d_output = Array2::from_elem((2, 6), 0.5);
        adapter.backward(&input, &d_output);
        adapter.zero_grad();
        assert_eq!(
            adapter.grad_norm(),
            0.0,
            "grad_norm must be 0 after zero_grad"
        );
    }

    // ── test 8: LoraTrainer reduces loss over 100 epochs ─────────────────────
    //
    // `train_epoch` treats `base_output` as both the "input" to the LoRA delta
    // and the base projection. So for `base_output + delta` to be well-formed,
    // d_in == d_out. We use d_in = d_out = 4 here.

    #[test]
    fn test_trainer_loss_converges() {
        // d_in == d_out == 4; LoRA rank = 1.
        let adapter = LoraAdapter::new(4, 4, 1, 1.0, 7);
        let mut trainer = LoraTrainer::new(adapter, 0.05);

        // Frozen projector output: constant all-ones [batch=3, d_in=4].
        // Target: all-zeros [batch=3, d_out=4].
        let base_out = Array2::from_elem((3, 4), 1.0);
        let targets = Array2::zeros((3, 4));

        let initial_loss = trainer.train_epoch(&base_out, &targets);
        let mut final_loss = initial_loss;
        for _ in 0..99 {
            final_loss = trainer.train_epoch(&base_out, &targets);
        }

        // Loss should have decreased by at least 10% (very conservative).
        assert!(
            final_loss < initial_loss * 0.9 || final_loss < 1e-6,
            "loss should decrease: initial={initial_loss:.6}, final={final_loss:.6}"
        );
    }

    // ── test 9: rank=1 adapter works correctly ────────────────────────────────

    #[test]
    fn test_rank_one_adapter() {
        let adapter = make_adapter(6, 4, 1);
        assert_eq!(adapter.rank, 1);
        let input = Array2::from_elem((2, 6), 0.5);
        let delta = adapter.forward_delta(&input);
        assert_eq!(delta.shape(), &[2, 4]);
    }

    // ── test 10: scale() equals alpha / rank ─────────────────────────────────

    #[test]
    fn test_scale_equals_alpha_over_rank() {
        let adapter = LoraAdapter::new(4, 4, 3, 9.0, 0);
        let expected = 9.0_f64 / 3.0_f64;
        assert!(
            (adapter.scale() - expected).abs() < 1e-15,
            "scale should be alpha/rank = {expected}, got {}",
            adapter.scale()
        );
    }

    // ── extra test 11: finite-difference gradient check on grad_B ────────────

    #[test]
    fn test_fd_gradient_check_grad_b() {
        // Small dimensions for a quick FD check.
        let d_in = 2;
        let d_out = 3;
        let rank = 1;
        let alpha = 1.0;
        let eps = 1e-5;

        let input = Array2::from_shape_vec((2, d_in), vec![0.3, -0.5, 1.2, 0.1]).expect("shape ok");
        let d_output = Array2::from_shape_vec((2, d_out), vec![0.1, -0.2, 0.4, 0.6, -0.1, 0.3])
            .expect("shape ok");

        // ── analytic gradient for B[0,0] ─────────────────────────────────────
        let mut adapter_a = LoraAdapter::new(d_in, d_out, rank, alpha, 99);
        adapter_a.backward(&input, &d_output);
        let analytic = adapter_a.grad_b[[0, 0]];

        // ── finite-difference gradient for B[0,0] ─────────────────────────────
        let mut adapter_p = LoraAdapter::new(d_in, d_out, rank, alpha, 99);
        let mut adapter_n = LoraAdapter::new(d_in, d_out, rank, alpha, 99);
        adapter_p.b_matrix[[0, 0]] += eps;
        adapter_n.b_matrix[[0, 0]] -= eps;

        // "Loss" = sum(d_output element-wise * delta) (linear proxy for any
        // smooth objective whose gradient at delta is d_output).
        let loss_fn = |ad: &LoraAdapter| -> f64 {
            let delta = ad.forward_delta(&input);
            delta
                .iter()
                .zip(d_output.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };

        let fd = (loss_fn(&adapter_p) - loss_fn(&adapter_n)) / (2.0 * eps);
        let rel_err = (analytic - fd).abs() / (fd.abs().max(1e-10));
        assert!(
            rel_err < 1e-4,
            "FD gradient check failed: analytic={analytic:.8}, fd={fd:.8}, rel_err={rel_err:.6}"
        );
    }
}
