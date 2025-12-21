//! PageRank algorithm implementation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use colored::Colorize;
use scirs2_core::ndarray_ext::Array1;

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute PageRank algorithm on the graph
pub fn execute_pagerank(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "PageRank Analysis (Power Iteration with SciRS2 Arrays)"
            .green()
            .bold()
    );
    println!(
        "  Damping factor: {}",
        config.damping_factor.to_string().yellow()
    );
    println!(
        "  Max iterations: {}",
        config.max_iterations.to_string().yellow()
    );
    println!(
        "  Tolerance: {}",
        format!("{:.2e}", config.tolerance).yellow()
    );
    println!();

    let n = graph.node_count();
    let d = config.damping_factor;
    let mut scores = Array1::from_elem(n, 1.0 / n as f64);
    let mut new_scores = Array1::zeros(n);
    let out_degrees: Vec<usize> = (0..n).map(|i| graph.out_degree(i)).collect();

    if config.enable_parallel && n > 50_000 {
        println!(
            "{}",
            "ℹ Parallel PageRank: For >50K nodes, use scirs2_core::parallel_ops::par_chunks"
                .cyan()
                .dimmed()
        );
        println!(
            "{}",
            "  Speedup: 2-8x on multi-core systems (depends on edge distribution)"
                .cyan()
                .dimmed()
        );
    }

    let mut converged = false;
    for iter in 0..config.max_iterations {
        new_scores.fill(0.0);
        for node in 0..n {
            let degree = out_degrees[node];
            if degree > 0 {
                let contrib = scores[node] / degree as f64;
                for &neighbor in graph.neighbors(node) {
                    new_scores[neighbor] += d * contrib;
                }
            }
        }

        let sum: f64 = new_scores.iter().sum();
        let random_jump = (1.0 - d) / n as f64 + d * (1.0 - sum) / n as f64;
        for score in new_scores.iter_mut() {
            *score += random_jump;
        }

        let diff = (&new_scores - &scores).mapv(|x| x.abs()).sum();
        if diff < config.tolerance {
            converged = true;
            println!(
                "{}",
                format!(
                    "Converged after {} iterations (diff: {:.2e})",
                    iter + 1,
                    diff
                )
                .green()
            );
            break;
        }
        scores.assign(&new_scores);
    }

    if !converged {
        println!(
            "{}",
            format!(
                "Did not converge after {} iterations",
                config.max_iterations
            )
            .yellow()
        );
    }

    let mut ranked: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(id, &score)| (id, score))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!(
        "{}",
        format!("Top {} Resources by PageRank:", config.top_k)
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<65} | {:>10}", "Rank", "Resource", "Score");
    println!("{}", "─".repeat(100).cyan());

    for (rank, (node_id, score)) in ranked.iter().take(config.top_k).enumerate() {
        if let Some(node_name) = graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 65 {
                format!("{}...", &node_name[..62])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<65} | {}",
                (rank + 1).to_string().yellow(),
                truncated,
                format!("{:.6}", score).green()
            );
        }
    }

    Ok(())
}
