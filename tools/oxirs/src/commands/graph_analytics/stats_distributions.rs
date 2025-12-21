//! Graph statistics and degree distributions
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use colored::Colorize;
use scirs2_core::ndarray_ext::Array1;

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute degree distribution analysis
pub fn execute_degree_distribution(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!("{}", "Degree Distribution Analysis".green().bold());
    println!();

    let n = graph.node_count();
    let degrees = Array1::from_vec(
        (0..n)
            .map(|i| graph.out_degree(i) as f64)
            .collect::<Vec<_>>(),
    );

    let mean = degrees.mean().unwrap_or(0.0);
    let std = degrees.std(0.0);
    let max = degrees
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let min = degrees
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    println!("{}", "Degree Statistics:".yellow().bold());
    println!("  Mean degree:   {}", format!("{:.2}", mean).green());
    println!("  Std deviation: {}", format!("{:.2}", std).green());
    println!("  Max degree:    {}", format!("{:.0}", max).green());
    println!("  Min degree:    {}", format!("{:.0}", min).green());
    println!();

    let mut degree_pairs: Vec<(usize, f64)> = degrees.iter().copied().enumerate().collect();
    degree_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!(
        "{}",
        format!("Top {} Hub Nodes:", config.top_k).yellow().bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<70} | {:>8}", "Rank", "Resource", "Degree");
    println!("{}", "─".repeat(100).cyan());

    for (i, (node_id, degree)) in degree_pairs.iter().take(config.top_k).enumerate() {
        if let Some(node_name) = graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 70 {
                format!("{}...", &node_name[..67])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<70} | {}",
                (i + 1).to_string().yellow(),
                truncated,
                format!("{:.0}", degree).green()
            );
        }
    }

    println!();
    println!("{}", "Degree Distribution Histogram:".yellow().bold());
    let bins = 10;
    let bin_width = (max - min) / bins as f64;
    for i in 0..bins {
        let bin_start = min + i as f64 * bin_width;
        let bin_end = bin_start + bin_width;
        let count = degrees
            .iter()
            .filter(|&&d| d >= bin_start && d < bin_end)
            .count();
        let bar_length = (count as f64 / n as f64 * 50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!(
            "  [{:6.1}-{:6.1}): {} {}",
            bin_start,
            bin_end,
            bar.green(),
            format!("({})", count).dimmed()
        );
    }

    Ok(())
}

/// Execute comprehensive graph statistics
pub fn execute_graph_stats(graph: &RdfGraph) -> Result<()> {
    println!("{}", "Comprehensive Graph Statistics".green().bold());
    println!("{}", "─".repeat(70).cyan());

    let n = graph.node_count();
    let m = graph.edge_count;

    println!();
    println!("{}", "Basic Metrics:".yellow().bold());
    println!("  Total nodes: {}", n.to_string().green());
    println!("  Total edges: {}", m.to_string().green());
    println!(
        "  Density: {}",
        format!(
            "{:.6}",
            if n > 1 {
                (m as f64) / ((n * (n - 1)) as f64)
            } else {
                0.0
            }
        )
        .green()
    );

    let degrees = Array1::from_vec(
        (0..n)
            .map(|i| graph.out_degree(i) as f64)
            .collect::<Vec<_>>(),
    );

    let mean = degrees.mean().unwrap_or(0.0);
    let std = degrees.std(0.0);
    let max = degrees
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let min = degrees
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    println!();
    println!("{}", "Degree Distribution:".yellow().bold());
    println!("  Mean degree:   {}", format!("{:.2}", mean).green());
    println!("  Std deviation: {}", format!("{:.2}", std).green());
    println!("  Max degree:    {}", format!("{:.0}", max).green());
    println!("  Min degree:    {}", format!("{:.0}", min).green());

    let mut degree_pairs: Vec<(usize, f64)> = degrees.iter().copied().enumerate().collect();
    degree_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!("{}", "Top Hub Nodes (by degree):".yellow().bold());
    for (i, (node_id, degree)) in degree_pairs.iter().take(10).enumerate() {
        if let Some(node_name) = graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 75 {
                format!("{}...", &node_name[..72])
            } else {
                node_name.to_string()
            };
            println!(
                "  {} {} (degree: {})",
                (i + 1).to_string().cyan(),
                truncated,
                format!("{:.0}", degree).green()
            );
        }
    }

    Ok(())
}
