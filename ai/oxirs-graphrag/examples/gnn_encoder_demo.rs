//! GraphSAGE encoder demo — phase a of the hybrid GNN+LLM architecture.
//!
//! Builds a 10-node synthetic knowledge graph, runs training for 100 epochs,
//! and reports the per-layer weight statistics and final embedding shape.

use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
use scirs2_core::ndarray_ext::Array2;

fn main() {
    // ── Build a small synthetic knowledge graph ───────────────────────────────
    let num_nodes = 10;
    let feat_dim = 8;

    // Identity features: node i has a 1 at position i (for nodes 0..feat_dim).
    let mut node_features = Array2::zeros((num_nodes, feat_dim));
    for i in 0..num_nodes.min(feat_dim) {
        node_features[[i, i]] = 1.0;
    }

    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 0),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ];

    let graph = KgGraph {
        num_nodes,
        edges,
        node_features,
    };

    // ── Configure the encoder ─────────────────────────────────────────────────
    let config = GraphSageConfig {
        input_dim: feat_dim,
        hidden_dim: 16,
        output_dim: 16,
        num_layers: 2,
        dropout: 0.0,
        k_neighbors: 5,
        learning_rate: 0.01,
    };

    let mut encoder =
        GraphSageEncoder::new_with_seed(&config, 42).expect("failed to construct encoder");

    // ── Encode before training ────────────────────────────────────────────────
    let emb_before = encoder.encode(&graph).expect("encode before training");
    println!(
        "Embeddings before training: shape [{} × {}]",
        emb_before.embeddings.nrows(),
        emb_before.embeddings.ncols()
    );

    // ── Train ─────────────────────────────────────────────────────────────────
    let history = encoder.train(&graph, 100).expect("training failed");
    println!(
        "Training complete — 100 epochs.\n  Loss[0]  = {:.6}\n  Loss[99] = {:.6}",
        history.epoch_losses[0], history.epoch_losses[99]
    );

    // ── Encode after training ─────────────────────────────────────────────────
    let emb_after = encoder.encode(&graph).expect("encode after training");
    println!(
        "Embeddings after training:  shape [{} × {}]",
        emb_after.embeddings.nrows(),
        emb_after.embeddings.ncols()
    );

    // Report a sample embedding.
    let row: Vec<f64> = emb_after.embeddings.row(0).to_vec();
    println!(
        "Node 0 embedding (first 4 dims): {:?}",
        &row[..4.min(row.len())]
    );

    println!("Demo complete.");
}
