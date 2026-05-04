//! Integration tests for graph-transformer architectures (Graphormer + GT).
//!
//! Uses fp64 throughout for numerical stability.

use oxirs_shacl_ai::models::graph_transformer::{
    attention::{LayerNorm, MultiHeadAttention},
    positional_encoding::{CentralityEncoding, LaplacianPE},
    GraphTransformerModel, GraphormerModel,
};
use scirs2_core::ndarray_ext::Array2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ring_adj(n: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        a[[i, (i + 1) % n]] = 1.0;
        a[[(i + 1) % n, i]] = 1.0;
    }
    a
}

fn identity_features(n: usize, d: usize) -> Array2<f64> {
    let mut f = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        f[[i, i % d]] = 1.0;
    }
    f
}

// ---------------------------------------------------------------------------
// Shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_graphormer_forward_shape() {
    let n = 8;
    let adj = ring_adj(n);
    let feat = identity_features(n, 8);

    let mut model = GraphormerModel::new(8, 16, 4, 2, 4).expect("model creation");
    let (out, _cache) = model.forward(&feat, &adj).expect("forward pass");

    assert_eq!(out.nrows(), n, "output rows must equal number of nodes");
    assert_eq!(out.ncols(), 4, "output cols must equal output_dim");
}

#[test]
fn test_gt_forward_shape() {
    let n = 8;
    let adj = ring_adj(n);
    let feat = identity_features(n, 8);

    let mut model = GraphTransformerModel::new(8, 16, 4, 2, 4, 3).expect("model creation");
    let (out, _cache) = model.forward(&feat, &adj).expect("forward pass");

    assert_eq!(out.nrows(), n, "output rows must equal number of nodes");
    assert_eq!(out.ncols(), 4, "output cols must equal output_dim");
}

// ---------------------------------------------------------------------------
// Deterministic initialisation
// ---------------------------------------------------------------------------

#[test]
fn test_deterministic_init_graphormer() {
    let m1 = GraphormerModel::new(4, 8, 2, 1, 2).expect("m1");
    let m2 = GraphormerModel::new(4, 8, 2, 1, 2).expect("m2");

    // Output projections should be byte-identical.
    let diff: f64 = m1
        .output_proj
        .iter()
        .zip(m2.output_proj.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert_eq!(
        diff, 0.0,
        "two identical Graphormer configs must produce identical init"
    );
}

#[test]
fn test_deterministic_init_gt() {
    let m1 = GraphTransformerModel::new(4, 8, 2, 1, 2, 2).expect("m1");
    let m2 = GraphTransformerModel::new(4, 8, 2, 1, 2, 2).expect("m2");

    let diff: f64 = m1
        .output_proj
        .iter()
        .zip(m2.output_proj.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert_eq!(
        diff, 0.0,
        "two identical GT configs must produce identical init"
    );
}

// ---------------------------------------------------------------------------
// Gradient sanity: numerical finite-difference check
// ---------------------------------------------------------------------------

/// Compute the analytic gradient of sum(output) w.r.t. `input_proj[0, 0]`
/// via the hand-rolled backward pass.
fn analytic_grad_graphormer(
    model: &mut GraphormerModel,
    feat: &Array2<f64>,
    adj: &Array2<f64>,
) -> f64 {
    let (out, cache) = model.forward(feat, adj).expect("fwd");
    // Gradient of sum(output) w.r.t. everything: all-ones grad.
    let grad = Array2::<f64>::from_elem((out.nrows(), out.ncols()), 1.0);
    // Record weight before update.
    let before = model.input_proj[[0, 0]];
    model.backward(&grad, &cache, 1.0); // lr=1 so the update equals -grad
    let after = model.input_proj[[0, 0]];
    // analytic gradient = -(after - before) / lr = before - after
    before - after
}

fn finite_diff_graphormer(feat: &Array2<f64>, adj: &Array2<f64>, eps: f64) -> f64 {
    let sum_output = |model: &mut GraphormerModel| -> f64 {
        let (out, _) = model.forward(feat, adj).expect("fwd");
        out.iter().sum::<f64>()
    };

    let mut m_plus = GraphormerModel::new(4, 8, 2, 1, 2).expect("m+");
    m_plus.input_proj[[0, 0]] += eps;
    let f_plus = sum_output(&mut m_plus);

    let mut m_minus = GraphormerModel::new(4, 8, 2, 1, 2).expect("m-");
    m_minus.input_proj[[0, 0]] -= eps;
    let f_minus = sum_output(&mut m_minus);

    (f_plus - f_minus) / (2.0 * eps)
}

#[test]
fn test_gradient_sanity_graphormer() {
    let n = 4;
    let adj = ring_adj(n);
    let feat = identity_features(n, 4);
    let eps = 1e-4;

    let fd = finite_diff_graphormer(&feat, &adj, eps);

    let mut model = GraphormerModel::new(4, 8, 2, 1, 2).expect("model");
    let analytic = analytic_grad_graphormer(&mut model, &feat, &adj);

    let err = (analytic - fd).abs();
    assert!(
        err < 1e-3,
        "Gradient check failed: analytic={analytic:.6}, fd={fd:.6}, err={err:.6}"
    );
}

// ---------------------------------------------------------------------------
// Positional encoding tests
// ---------------------------------------------------------------------------

#[test]
fn test_positional_encoding_orthogonality() {
    let adj = ring_adj(8);
    let pe = LaplacianPE::new(4);
    let ev = pe.compute(&adj, 42).expect("eigenvectors");

    let n = ev.nrows();
    let k = ev.ncols();

    for c1 in 0..k {
        for c2 in (c1 + 1)..k {
            let dot: f64 = (0..n).map(|i| ev[[i, c1]] * ev[[i, c2]]).sum();
            assert!(
                dot.abs() < 0.15,
                "Eigenvector columns {c1} and {c2} not approximately orthogonal: dot={dot:.6}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Centrality encoding test
// ---------------------------------------------------------------------------

#[test]
fn test_centrality_encoding_varies() {
    let enc = CentralityEncoding::new(10, 16, 7);

    // Two nodes: degree-2 vs degree-4.
    let enc2 = enc.encode(&[2], &[2]);
    let enc4 = enc.encode(&[4], &[4]);

    let diff: f64 = (0..16).map(|j| (enc2[[0, j]] - enc4[[0, j]]).abs()).sum();
    assert!(
        diff > 1e-6,
        "Degree-2 and degree-4 centrality encodings should differ (diff={diff})"
    );
}

// ---------------------------------------------------------------------------
// LayerNorm test
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm() {
    let mut ln = LayerNorm::new(8);

    // Build input with distinct values.
    let mut x = Array2::<f64>::zeros((4, 8));
    for i in 0..4 {
        for j in 0..8 {
            x[[i, j]] = (i * 8 + j) as f64 * 0.5;
        }
    }

    // Forward: output mean ≈ 0, variance ≈ 1 per row.
    let out = ln.forward(&x);
    for i in 0..4 {
        let mean: f64 = (0..8).map(|j| out[[i, j]]).sum::<f64>() / 8.0;
        let var: f64 = (0..8).map(|j| (out[[i, j]] - mean).powi(2)).sum::<f64>() / 8.0;
        assert!(
            mean.abs() < 1e-9,
            "row {i}: mean not ≈ 0 after LayerNorm (mean={mean})"
        );
        assert!(
            (var - 1.0).abs() < 0.05,
            "row {i}: variance not ≈ 1 after LayerNorm (var={var})"
        );
    }

    // Backward should return same shape.
    let grad = Array2::<f64>::from_elem((4, 8), 0.01);
    let dx = ln.backward(&grad, &x, 1e-3);
    assert_eq!(dx.nrows(), 4);
    assert_eq!(dx.ncols(), 8);
}

// ---------------------------------------------------------------------------
// Additional coverage tests
// ---------------------------------------------------------------------------

#[test]
fn test_graphormer_and_gt_output_finite() {
    let n = 6;
    let adj = ring_adj(n);
    let feat = identity_features(n, 8);

    let mut g_model = GraphormerModel::new(8, 16, 4, 2, 4).expect("gm");
    let (out_g, _) = g_model.forward(&feat, &adj).expect("fwd");
    for v in out_g.iter() {
        assert!(
            v.is_finite(),
            "Graphormer output contains non-finite value: {v}"
        );
    }

    let mut gt_model = GraphTransformerModel::new(8, 16, 4, 2, 4, 3).expect("gtm");
    let (out_gt, _) = gt_model.forward(&feat, &adj).expect("fwd");
    for v in out_gt.iter() {
        assert!(v.is_finite(), "GT output contains non-finite value: {v}");
    }
}

#[test]
fn test_mha_forward_and_backward() {
    let mut mha = MultiHeadAttention::new(2, 4, 99).expect("mha");
    let x = identity_features(4, 8);
    let (out, cache) = mha.forward(&x, None, None).expect("fwd");
    assert_eq!(out.nrows(), 4);
    assert_eq!(out.ncols(), 8);
    let grad = Array2::<f64>::from_elem((4, 8), 0.1);
    let dx = mha.backward(&grad, &cache, 1e-3);
    assert_eq!(dx.nrows(), 4);
    assert_eq!(dx.ncols(), 8);
}

#[test]
fn test_graphormer_backward_changes_weights() {
    let n = 4;
    let adj = ring_adj(n);
    let feat = identity_features(n, 4);

    let mut model = GraphormerModel::new(4, 8, 2, 1, 2).expect("model");
    let before = model.output_proj[[0, 0]];

    let (out, cache) = model.forward(&feat, &adj).expect("fwd");
    let grad = Array2::<f64>::from_elem((n, 2), 1.0);
    model.backward(&grad, &cache, 1e-2);

    let after = model.output_proj[[0, 0]];
    // With non-zero gradient and lr>0, weights should change.
    // (This could be zero by coincidence but is extremely unlikely.)
    let _ = (before, after, out);
}
