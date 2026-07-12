//! Integration tests for the LoRA adapter and LoRA trainer (phase d).

use oxirs_graphrag::hybrid::lora::{LoraAdapter, LoraTrainer};
use scirs2_core::ndarray_ext::Array2;

// ── Test 1: B matrix is zero-initialised ─────────────────────────────────────

#[test]
fn test_b_matrix_zero_init() {
    let adapter = LoraAdapter::new(8, 16, 4, 4.0, 0);
    for &v in adapter.b_matrix.iter() {
        assert_eq!(v, 0.0, "B must be zero-initialised, got {v}");
    }
}

// ── Test 2: forward_delta with A=0 and B=0 gives zero output ─────────────────

#[test]
fn test_forward_delta_zero_params_gives_zero() {
    let mut adapter = LoraAdapter::new(6, 4, 2, 2.0, 1);
    // A is Xavier-init; zero it out manually.
    for v in adapter.a_matrix.iter_mut() {
        *v = 0.0;
    }
    // B starts at zero already.
    let input = Array2::from_elem((4, 6), 1.0);
    let delta = adapter.forward_delta(&input);
    for &v in delta.iter() {
        assert!(v.abs() < 1e-14, "delta must be zero, got {v}");
    }
}

// ── Test 3: forward_delta output shape [batch, d_out] ────────────────────────

#[test]
fn test_forward_delta_output_shape() {
    let adapter = LoraAdapter::new(10, 7, 3, 3.0, 2);
    let input = Array2::zeros((6, 10));
    let delta = adapter.forward_delta(&input);
    assert_eq!(delta.nrows(), 6, "batch size preserved");
    assert_eq!(delta.ncols(), 7, "output size equals d_out");
}

// ── Test 4: backward returns d_input with shape [batch, d_in] ────────────────

#[test]
fn test_backward_d_input_shape() {
    let mut adapter = LoraAdapter::new(10, 7, 3, 3.0, 3);
    let input = Array2::from_elem((6, 10), 0.5);
    let d_output = Array2::from_elem((6, 7), 0.1);
    let d_input = adapter.backward(&input, &d_output);
    assert_eq!(d_input.nrows(), 6, "d_input batch size preserved");
    assert_eq!(d_input.ncols(), 10, "d_input columns equal d_in");
}

// ── Test 5: sgd_step updates A and B ─────────────────────────────────────────

#[test]
fn test_sgd_step_updates_b_matrix() {
    let mut adapter = LoraAdapter::new(4, 5, 2, 2.0, 4);
    let input = Array2::from_elem((3, 4), 1.0);
    let d_output = Array2::from_elem((3, 5), 0.2);

    // After one backward, grad_B is non-zero (B=0 but z=input@A is non-zero if A!=0).
    adapter.backward(&input, &d_output);

    let b_before: Vec<f64> = adapter.b_matrix.iter().copied().collect();
    adapter.sgd_step(0.01);
    let b_after: Vec<f64> = adapter.b_matrix.iter().copied().collect();

    // At least one element of B must have changed.
    let changed = b_before
        .iter()
        .zip(b_after.iter())
        .any(|(old, new)| (old - new).abs() > 1e-15);
    assert!(changed, "sgd_step must update B matrix");
}

// ── Test 6: zero_grad resets grad_a and grad_b to 0 (verified via grad_norm) ──

#[test]
fn test_zero_grad_clears_both_gradients() {
    let mut adapter = LoraAdapter::new(5, 3, 2, 2.0, 5);
    let input = Array2::from_elem((2, 5), 1.0);
    let d_output = Array2::from_elem((2, 3), 0.5);
    // Accumulate some gradients.
    adapter.backward(&input, &d_output);
    // grad_norm should be non-zero after backward (grad_B = z.T @ d_output * scale).
    let norm_before = adapter.grad_norm();
    assert!(
        norm_before > 0.0,
        "grad_norm should be positive after backward"
    );
    adapter.zero_grad();
    assert_eq!(
        adapter.grad_norm(),
        0.0,
        "grad_norm must be 0 after zero_grad"
    );
}

// ── Test 7: grad_norm is 0 after zero_grad ───────────────────────────────────

#[test]
fn test_grad_norm_zero_after_zero_grad() {
    let mut adapter = LoraAdapter::new(5, 3, 2, 2.0, 6);
    let input = Array2::from_elem((2, 5), 1.0);
    let d_output = Array2::from_elem((2, 3), 0.5);
    adapter.backward(&input, &d_output);
    adapter.zero_grad();
    assert_eq!(
        adapter.grad_norm(),
        0.0,
        "grad_norm must be 0 after zero_grad"
    );
}

// ── Test 8: LoraTrainer reduces loss over 100 epochs ─────────────────────────
//
// `train_epoch` takes the raw `input` fed into the frozen base projection
// and the frozen projector's `base_output` (= W · input) as two separate
// `[batch, d_in]` / `[batch, d_out]` tensors, so d_in and d_out are free to
// differ. This test uses a square adapter purely for a simple toy setup —
// see `test_trainer_rectangular_dims_loss_decreases` below for d_in != d_out.

#[test]
fn test_trainer_loss_decreases_over_epochs() {
    // d_in == d_out == 4; LoRA rank = 2.
    let adapter = LoraAdapter::new(4, 4, 2, 2.0, 42);
    let mut trainer = LoraTrainer::new(adapter, 0.05);

    // Constant raw input / frozen projector output (all 1.0) [batch=3, d=4].
    // Target: all-zeros [batch=3, d_out=4].
    let input = Array2::from_elem((3, 4), 1.0);
    let base_out = Array2::from_elem((3, 4), 1.0);
    let targets = Array2::zeros((3, 4));

    let initial_loss = trainer.train_epoch(&input, &base_out, &targets);
    let mut current_loss = initial_loss;
    for _ in 0..99 {
        current_loss = trainer.train_epoch(&input, &base_out, &targets);
    }

    assert!(
        current_loss < initial_loss * 0.9 || current_loss < 1e-6,
        "loss must decrease: initial={initial_loss:.6}, after 100 epochs={current_loss:.6}"
    );
}

// ── Test 8b: LoraTrainer with a rectangular adapter (d_in != d_out) ──────────
//
// Regression test for the fix that separated `input` from `base_output`:
// LoRA's ΔW = B·A decomposition is rectangular by construction, so training
// must work when d_in != d_out (the common case for real projections).

#[test]
fn test_trainer_rectangular_dims_loss_decreases() {
    // d_in = 6, d_out = 3; LoRA rank = 2.
    let adapter = LoraAdapter::new(6, 3, 2, 2.0, 43);
    let mut trainer = LoraTrainer::new(adapter, 0.05);

    let input = Array2::from_elem((4, 6), 1.0);
    let base_out = Array2::from_elem((4, 3), 1.0);
    let targets = Array2::zeros((4, 3));

    let initial_loss = trainer.train_epoch(&input, &base_out, &targets);
    assert!(initial_loss.is_finite() && initial_loss >= 0.0);

    let mut current_loss = initial_loss;
    for _ in 0..99 {
        current_loss = trainer.train_epoch(&input, &base_out, &targets);
    }

    assert!(
        current_loss < initial_loss * 0.9 || current_loss < 1e-6,
        "loss must decrease with d_in != d_out: initial={initial_loss:.6}, after 100 epochs={current_loss:.6}"
    );
}

// ── Test 9: rank=1 adapter produces correct shapes ───────────────────────────

#[test]
fn test_rank_one_adapter_shapes() {
    let adapter = LoraAdapter::new(8, 5, 1, 1.0, 9);
    assert_eq!(adapter.rank, 1);
    assert_eq!(adapter.a_matrix.shape(), &[8, 1]);
    assert_eq!(adapter.b_matrix.shape(), &[1, 5]);

    let input = Array2::from_elem((4, 8), 0.3);
    let delta = adapter.forward_delta(&input);
    assert_eq!(delta.shape(), &[4, 5]);
}

// ── Test 10: scale() equals alpha / rank ─────────────────────────────────────

#[test]
fn test_scale_formula() {
    let adapter = LoraAdapter::new(4, 4, 5, 15.0, 0);
    let expected = 15.0_f64 / 5.0_f64;
    assert!(
        (adapter.scale() - expected).abs() < f64::EPSILON,
        "scale() should be alpha/rank = {expected}, got {}",
        adapter.scale()
    );
}

// ── Test 11: into_adapter consumes trainer and returns adapter ────────────────

#[test]
fn test_into_adapter_consumes_trainer() {
    let adapter = LoraAdapter::new(4, 3, 2, 2.0, 11);
    let trainer = LoraTrainer::new(adapter, 0.01);
    let recovered = trainer.into_adapter();
    assert_eq!(recovered.d_in, 4);
    assert_eq!(recovered.d_out, 3);
    assert_eq!(recovered.rank, 2);
}

// ── Test 12: adapter() borrows without consuming ──────────────────────────────

#[test]
fn test_adapter_borrow() {
    let adapter = LoraAdapter::new(4, 3, 2, 2.0, 12);
    let trainer = LoraTrainer::new(adapter, 0.01);
    let borrowed = trainer.adapter();
    assert_eq!(borrowed.d_in, 4);
    // trainer is still alive here.
    drop(trainer);
}

// ── Test 13: finite-difference gradient check on grad_A via sgd_step ──────────
//
// With non-zero B we can verify that grad_A flows correctly by observing
// how much A changes after one sgd_step with a known lr, compared to a
// direct FD estimate of the loss gradient w.r.t. A[0,0].
//
// Because grad_a is private we use the sgd_step oracle: after one step,
// A[0,0]_new = A[0,0]_old - lr * grad_A[0,0].
// We can also compute grad_A[0,0] via finite differences on the loss.

#[test]
fn test_fd_gradient_check_grad_a() {
    let d_in = 2;
    let d_out = 2;
    let rank = 1;
    let eps = 1e-5;
    let lr = 1.0; // lr=1 so delta_A = -grad_A

    let input = Array2::from_shape_vec((2, d_in), vec![0.5, -0.3, 1.1, 0.2]).expect("shape ok");
    let d_output = Array2::from_shape_vec((2, d_out), vec![0.1, -0.2, 0.4, 0.3]).expect("shape ok");

    // Adapter with non-zero B so that grad_A != 0.
    let mut adapter_sgd = LoraAdapter::new(d_in, d_out, rank, 1.0, 77);
    adapter_sgd.b_matrix[[0, 0]] = 0.5;
    adapter_sgd.b_matrix[[0, 1]] = -0.3;
    let a00_before = adapter_sgd.a_matrix[[0, 0]];
    adapter_sgd.backward(&input, &d_output);
    adapter_sgd.sgd_step(lr);
    let a00_after = adapter_sgd.a_matrix[[0, 0]];
    // grad_A[0,0] recovered as: (a00_before - a00_after) / lr
    let analytic = (a00_before - a00_after) / lr;

    // FD estimate for A[0,0].
    let loss_fn = |ad: &LoraAdapter| -> f64 {
        let delta = ad.forward_delta(&input);
        delta
            .iter()
            .zip(d_output.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };

    let mut ap = LoraAdapter::new(d_in, d_out, rank, 1.0, 77);
    ap.b_matrix[[0, 0]] = 0.5;
    ap.b_matrix[[0, 1]] = -0.3;
    ap.a_matrix[[0, 0]] = a00_before + eps;

    let mut an = LoraAdapter::new(d_in, d_out, rank, 1.0, 77);
    an.b_matrix[[0, 0]] = 0.5;
    an.b_matrix[[0, 1]] = -0.3;
    an.a_matrix[[0, 0]] = a00_before - eps;

    let fd = (loss_fn(&ap) - loss_fn(&an)) / (2.0 * eps);
    let rel_err = (analytic - fd).abs() / (fd.abs().max(1e-10));
    assert!(
        rel_err < 1e-3,
        "FD grad_A check: analytic(from sgd)={analytic:.8}, fd={fd:.8}, rel_err={rel_err:.6}"
    );
}
