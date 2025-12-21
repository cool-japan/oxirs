//! Shortest paths algorithms
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{Context, Result};
use colored::Colorize;
use std::collections::VecDeque;

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute shortest paths analysis
pub fn execute_shortest_paths(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!("{}", "Shortest Paths Analysis (BFS)".green().bold());
    println!();

    let source_uri = config
        .source_node
        .as_ref()
        .context("Source node required for shortest paths")?;

    let source_id = graph
        .node_to_id
        .get(source_uri)
        .copied()
        .context(format!("Source node not found: {}", source_uri))?;

    println!("Source: {}", source_uri.yellow());
    println!();

    let n = graph.node_count();
    let mut distances = vec![f64::INFINITY; n];
    let mut predecessors = vec![None; n];
    let mut queue = VecDeque::new();

    distances[source_id] = 0.0;
    queue.push_back(source_id);

    while let Some(node) = queue.pop_front() {
        let dist = distances[node];
        for &neighbor in graph.neighbors(node) {
            if distances[neighbor].is_infinite() {
                distances[neighbor] = dist + 1.0;
                predecessors[neighbor] = Some(node);
                queue.push_back(neighbor);
            }
        }
    }

    if let Some(target_uri) = &config.target_node {
        let target_id = graph
            .node_to_id
            .get(target_uri)
            .copied()
            .context(format!("Target node not found: {}", target_uri))?;

        println!("Target: {}", target_uri.cyan());
        println!();

        if distances[target_id].is_finite() {
            println!(
                "Shortest path distance: {}",
                format!("{:.0}", distances[target_id]).green().bold()
            );
            println!();

            let mut path = vec![target_id];
            let mut current = target_id;
            while let Some(pred) = predecessors[current] {
                path.push(pred);
                current = pred;
            }
            path.reverse();

            println!("{}", "Path:".yellow().bold());
            for (i, &node_id) in path.iter().enumerate() {
                if let Some(node_name) = graph.get_node_name(node_id) {
                    println!("  {} {}", (i + 1).to_string().cyan(), node_name);
                }
            }
        } else {
            println!("{}", "No path exists to target".red().bold());
        }
    } else {
        let mut reachable: Vec<(usize, f64)> = distances
            .iter()
            .enumerate()
            .filter(|(_, &dist)| dist.is_finite() && dist > 0.0)
            .map(|(id, &dist)| (id, dist))
            .collect();

        reachable.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        println!(
            "{}",
            format!(
                "Top {} Closest Resources:",
                config.top_k.min(reachable.len())
            )
            .cyan()
            .bold()
        );
        println!("{}", "─".repeat(100).cyan());
        println!("{:>5} | {:<70} | {:>8}", "Rank", "Resource", "Distance");
        println!("{}", "─".repeat(100).cyan());

        for (rank, (node_id, distance)) in reachable.iter().take(config.top_k).enumerate() {
            if let Some(node_name) = graph.get_node_name(*node_id) {
                let truncated = if node_name.len() > 70 {
                    format!("{}...", &node_name[..67])
                } else {
                    node_name.to_string()
                };
                println!(
                    "{:>5} | {:<70} | {}",
                    (rank + 1).to_string().yellow(),
                    truncated,
                    format!("{:.0}", distance).green()
                );
            }
        }
    }

    Ok(())
}
