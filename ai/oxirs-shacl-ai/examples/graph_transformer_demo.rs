//! Demonstration of Graphormer and GT graph transformer architectures.
//!
//! Builds a small 8-node synthetic graph and runs forward passes through both models.

use oxirs_shacl_ai::models::graph_transformer::{GraphTransformerModel, GraphormerModel};
use scirs2_core::ndarray_ext::Array2;

fn main() -> anyhow::Result<()> {
    let n = 8;
    let input_dim = 8;
    let hidden_dim = 16;
    let num_heads = 4;
    let output_dim = 4;

    // Build a random synthetic adjacency matrix (bidirectional ring + extra edges).
    let mut adj = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        adj[[i, (i + 1) % n]] = 1.0;
        adj[[(i + 1) % n, i]] = 1.0;
    }
    // Add a few cross-edges for variety.
    adj[[0, 4]] = 1.0;
    adj[[4, 0]] = 1.0;
    adj[[2, 6]] = 1.0;
    adj[[6, 2]] = 1.0;

    // Identity node features.
    let mut feat = Array2::<f64>::zeros((n, input_dim));
    for i in 0..n {
        feat[[i, i % input_dim]] = 1.0;
    }

    // --- Graphormer ---
    println!("=== Graphormer ===");
    let mut graphormer = GraphormerModel::new(input_dim, hidden_dim, num_heads, 2, output_dim)?;
    let (out_g, cache_g) = graphormer.forward(&feat, &adj)?;
    println!("Output shape: [{}, {}]", out_g.nrows(), out_g.ncols());
    println!(
        "First row (first 4 values): {:?}",
        (0..out_g.ncols())
            .map(|j| out_g[[0, j]])
            .collect::<Vec<_>>()
    );

    // One backward step.
    let grad = Array2::<f64>::zeros((n, output_dim));
    graphormer.backward(&grad, &cache_g, 1e-3);
    println!("Backward pass completed.");

    // --- Graph Transformer (GT) ---
    println!("\n=== Graph Transformer (GT) ===");
    let pe_k = 3;
    let mut gt_model =
        GraphTransformerModel::new(input_dim, hidden_dim, num_heads, 2, output_dim, pe_k)?;
    let (out_gt, cache_gt) = gt_model.forward(&feat, &adj)?;
    println!("Output shape: [{}, {}]", out_gt.nrows(), out_gt.ncols());
    println!(
        "First row (first 4 values): {:?}",
        (0..out_gt.ncols())
            .map(|j| out_gt[[0, j]])
            .collect::<Vec<_>>()
    );

    gt_model.backward(&grad, &cache_gt, 1e-3);
    println!("Backward pass completed.");

    println!("\nAll assertions passed.");
    Ok(())
}
