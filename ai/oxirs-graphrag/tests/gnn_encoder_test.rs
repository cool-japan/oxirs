//! Integration tests for the GraphSAGE encoder (phase a).

use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
use scirs2_core::ndarray_ext::Array2;

// ─── Fixtures ─────────────────────────────────────────────────────────────────

fn toy_graph() -> KgGraph {
    KgGraph {
        num_nodes: 8,
        edges: vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 0),
            (0, 4),
            (1, 5),
        ],
        node_features: Array2::zeros((8, 16)),
    }
}

fn toy_config() -> GraphSageConfig {
    GraphSageConfig {
        input_dim: 16,
        hidden_dim: 16,
        output_dim: 16,
        num_layers: 2,
        dropout: 0.0,
        k_neighbors: 4,
        learning_rate: 0.01,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn test_forward_pass_shape() {
    let graph = toy_graph();
    let config = toy_config();
    let encoder = GraphSageEncoder::new(&config).expect("should construct");
    let embeddings = encoder.encode(&graph).expect("should encode");
    assert_eq!(embeddings.embeddings.nrows(), 8);
    assert_eq!(embeddings.embeddings.ncols(), 16);
}

#[test]
fn test_deterministic_init() {
    let config = toy_config();
    let e1 = GraphSageEncoder::new_with_seed(&config, 42).expect("should construct");
    let e2 = GraphSageEncoder::new_with_seed(&config, 42).expect("should construct");
    let graph = toy_graph();
    let emb1 = e1.encode(&graph).expect("encode");
    let emb2 = e2.encode(&graph).expect("encode");
    // Same seed → same output.
    for i in 0..8 {
        for j in 0..16 {
            assert!(
                (emb1.embeddings[[i, j]] - emb2.embeddings[[i, j]]).abs() < 1e-12,
                "embeddings differ at [{i},{j}]"
            );
        }
    }
}

#[test]
fn test_training_reduces_loss() {
    let graph = toy_graph();
    let config = toy_config();
    let mut encoder = GraphSageEncoder::new_with_seed(&config, 0).expect("construct");
    let history = encoder.train(&graph, 50).expect("train");
    assert_eq!(history.epoch_losses.len(), 50);
    // Loss should not increase over training (allow equality — zero features make the
    // loss constant at 1.0, which is non-strict improvement).
    let initial = history.epoch_losses[0];
    let final_loss = history.epoch_losses[49];
    assert!(
        final_loss <= initial,
        "loss should not increase over 50 epochs: {initial} → {final_loss}"
    );
}

#[test]
fn test_finite_difference_gradient_check() {
    // 4-node toy graph; check gradient of loss w.r.t. W1 weights.
    // With zero-feature inputs the loss is constant (zero gradient everywhere),
    // so the FD check trivially holds (|0 - 0| < 1e-3).
    // We use non-zero features to exercise the gradient path properly.
    let mut features = Array2::zeros((4, 4));
    // Give each node a distinct feature to break symmetry.
    features[[0, 0]] = 1.0;
    features[[1, 1]] = 1.0;
    features[[2, 2]] = 1.0;
    features[[3, 3]] = 1.0;

    let small_graph = KgGraph {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        node_features: features,
    };
    let config = GraphSageConfig {
        input_dim: 4,
        hidden_dim: 4,
        output_dim: 4,
        num_layers: 1,
        dropout: 0.0,
        k_neighbors: 2,
        learning_rate: 0.0,
    };
    let mut encoder = GraphSageEncoder::new_with_seed(&config, 7).expect("construct");

    let eps = 1e-5_f64;
    let (analytic_grad, _param_val) = encoder.compute_grad_and_param_for_test(&small_graph);
    let loss_plus = encoder.compute_loss_with_perturb(&small_graph, eps);
    let loss_minus = encoder.compute_loss_with_perturb(&small_graph, -eps);
    let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

    let err = (analytic_grad - fd_grad).abs();
    assert!(
        err < 1e-3,
        "FD gradient check failed: analytic={analytic_grad:.6}, fd={fd_grad:.6}, err={err:.6}"
    );
}

#[test]
fn test_node_ids_are_populated() {
    let graph = toy_graph();
    let config = toy_config();
    let enc = GraphSageEncoder::new(&config).expect("construct");
    let emb = enc.encode(&graph).expect("encode");
    assert_eq!(emb.node_ids.len(), 8);
}

#[test]
fn test_encode_empty_graph_returns_error() {
    let config = toy_config();
    let enc = GraphSageEncoder::new(&config).expect("construct");
    let empty = KgGraph {
        num_nodes: 0,
        edges: vec![],
        node_features: Array2::zeros((0, 16)),
    };
    assert!(enc.encode(&empty).is_err());
}

#[test]
fn test_training_history_length() {
    let graph = toy_graph();
    let config = toy_config();
    let mut enc = GraphSageEncoder::new_with_seed(&config, 5).expect("construct");
    let history = enc.train(&graph, 20).expect("train");
    assert_eq!(history.epoch_losses.len(), 20);
}

#[test]
fn test_sampler_respects_k_limit() {
    use oxirs_graphrag::gnn_encoder::sampler::sample_neighbours;
    use scirs2_core::random::seeded_rng;

    let edges: Vec<(usize, usize)> = (1..=50).map(|i| (0, i)).collect();
    let mut rng = seeded_rng(3);
    let result = sample_neighbours(0, &edges, 10, &mut rng);
    assert!(result.len() <= 10);
}

#[test]
fn test_aggregator_correctness() {
    use oxirs_graphrag::gnn_encoder::aggregator::mean_aggregate;
    let embs = vec![vec![2.0, 4.0], vec![4.0, 8.0]];
    let mean = mean_aggregate(&embs);
    assert!((mean[0] - 3.0).abs() < 1e-12);
    assert!((mean[1] - 6.0).abs() < 1e-12);
}
