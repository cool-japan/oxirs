//! Centrality algorithms for graph analysis
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Context;
use anyhow::Result;
use colored::Colorize;
use scirs2_graph::measures::{hits_algorithm, katz_centrality, HitsScores};
use scirs2_graph::{betweenness_centrality, closeness_centrality, eigenvector_centrality};

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute betweenness centrality analysis
pub fn execute_betweenness_centrality(
    rdf_graph: &RdfGraph,
    config: &AnalyticsConfig,
) -> Result<()> {
    println!(
        "{}",
        "Betweenness Centrality (SciRS2-Graph Advanced Algorithm)"
            .green()
            .bold()
    );
    println!(
        "{}",
        "  Computing shortest paths between all node pairs...".dimmed()
    );
    println!();

    let graph = rdf_graph.to_scirs2_graph()?;
    let centrality = betweenness_centrality(&graph, false);

    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{}",
        format!("Top {} Nodes by Betweenness Centrality:", config.top_k)
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<65} | {:>12}", "Rank", "Resource", "Centrality");
    println!("{}", "─".repeat(100).cyan());

    for (rank, (node_id, centrality_value)) in
        sorted_centrality.iter().take(config.top_k).enumerate()
    {
        if let Some(node_name) = rdf_graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 65 {
                format!("{}...", &node_name[..62])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<65} | {}",
                (rank + 1).to_string().yellow(),
                truncated,
                format!("{:.6}", centrality_value).green()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Betweenness centrality measures the number of shortest paths passing through each node."
            .dimmed()
    );
    println!(
        "{}",
        "High betweenness indicates bridge nodes connecting different parts of the graph.".dimmed()
    );

    Ok(())
}

/// Execute closeness centrality analysis
pub fn execute_closeness_centrality(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Closeness Centrality (SciRS2-Graph Advanced Algorithm)"
            .green()
            .bold()
    );
    println!(
        "{}",
        "  Computing average distances to all reachable nodes...".dimmed()
    );
    println!();

    let graph = rdf_graph.to_scirs2_graph()?;
    let centrality = closeness_centrality(&graph, true);

    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{}",
        format!("Top {} Nodes by Closeness Centrality:", config.top_k)
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<65} | {:>12}", "Rank", "Resource", "Centrality");
    println!("{}", "─".repeat(100).cyan());

    for (rank, (node_id, centrality_value)) in
        sorted_centrality.iter().take(config.top_k).enumerate()
    {
        if let Some(node_name) = rdf_graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 65 {
                format!("{}...", &node_name[..62])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<65} | {}",
                (rank + 1).to_string().yellow(),
                truncated,
                format!("{:.6}", centrality_value).green()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Closeness centrality measures average distance to all other nodes.".dimmed()
    );
    println!(
        "{}",
        "High closeness indicates nodes that can quickly reach other nodes.".dimmed()
    );

    Ok(())
}

/// Execute eigenvector centrality analysis
pub fn execute_eigenvector_centrality(
    rdf_graph: &RdfGraph,
    config: &AnalyticsConfig,
) -> Result<()> {
    println!(
        "{}",
        "Eigenvector Centrality (SciRS2-Graph Advanced Algorithm)"
            .green()
            .bold()
    );
    println!(
        "{}",
        "  Computing principal eigenvector of adjacency matrix...".dimmed()
    );
    println!();

    let graph = rdf_graph.to_scirs2_graph()?;
    let centrality = eigenvector_centrality(&graph, 100, 1e-6)
        .context("Failed to compute eigenvector centrality")?;

    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{}",
        format!("Top {} Nodes by Eigenvector Centrality:", config.top_k)
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<65} | {:>12}", "Rank", "Resource", "Centrality");
    println!("{}", "─".repeat(100).cyan());

    for (rank, (node_id, centrality_value)) in
        sorted_centrality.iter().take(config.top_k).enumerate()
    {
        if let Some(node_name) = rdf_graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 65 {
                format!("{}...", &node_name[..62])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<65} | {}",
                (rank + 1).to_string().yellow(),
                truncated,
                format!("{:.6}", centrality_value).green()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Eigenvector centrality measures influence based on connections to other influential nodes."
            .dimmed()
    );
    println!(
        "{}",
        "High eigenvector centrality indicates nodes connected to other important nodes.".dimmed()
    );

    Ok(())
}

/// Execute Katz centrality analysis
pub fn execute_katz_centrality(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Katz Centrality (SciRS2-Graph Advanced Algorithm)"
            .green()
            .bold()
    );
    println!(
        "{}",
        format!(
            "  Using alpha={}, beta={}",
            config.katz_alpha, config.katz_beta
        )
        .dimmed()
    );
    println!();

    let graph = rdf_graph.to_scirs2_graph()?;
    let centrality = katz_centrality(&graph, config.katz_alpha, config.katz_beta)
        .context("Failed to compute Katz centrality")?;

    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{}",
        format!("Top {} Nodes by Katz Centrality:", config.top_k)
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<65} | {:>12}", "Rank", "Resource", "Centrality");
    println!("{}", "─".repeat(100).cyan());

    for (rank, (node_id, centrality_value)) in
        sorted_centrality.iter().take(config.top_k).enumerate()
    {
        if let Some(node_name) = rdf_graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 65 {
                format!("{}...", &node_name[..62])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<65} | {}",
                (rank + 1).to_string().yellow(),
                truncated,
                format!("{:.6}", centrality_value).green()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Katz centrality extends eigenvector centrality by accounting for distant neighbors."
            .dimmed()
    );
    println!(
        "{}",
        "Alpha controls attenuation, beta is a constant added to each node.".dimmed()
    );

    Ok(())
}

/// Execute HITS algorithm
pub fn execute_hits_algorithm(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "HITS Algorithm - Hubs and Authorities (SciRS2-Graph)"
            .green()
            .bold()
    );
    println!("{}", "  Computing hub and authority scores...".dimmed());
    println!();

    let graph = rdf_graph.to_scirs2_digraph()?;
    let hits: HitsScores<usize> = hits_algorithm(&graph, config.max_iterations, config.tolerance)
        .context("Failed to compute HITS scores")?;

    let mut sorted_hubs: Vec<(usize, f64)> = hits.hubs.into_iter().collect();
    sorted_hubs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_authorities: Vec<(usize, f64)> = hits.authorities.into_iter().collect();
    sorted_authorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{}",
        format!("Top {} Hub Nodes:", config.top_k).cyan().bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<70} | {:>10}", "Rank", "Resource", "Hub Score");
    println!("{}", "─".repeat(100).cyan());

    for (rank, (node_id, score)) in sorted_hubs.iter().take(config.top_k).enumerate() {
        if let Some(node_name) = rdf_graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 70 {
                format!("{}...", &node_name[..67])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<70} | {}",
                (rank + 1).to_string().yellow(),
                truncated,
                format!("{:.6}", score).green()
            );
        }
    }

    println!();
    println!(
        "{}",
        format!("Top {} Authority Nodes:", config.top_k)
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<70} | {:>10}", "Rank", "Resource", "Auth Score");
    println!("{}", "─".repeat(100).cyan());

    for (rank, (node_id, score)) in sorted_authorities.iter().take(config.top_k).enumerate() {
        if let Some(node_name) = rdf_graph.get_node_name(*node_id) {
            let truncated = if node_name.len() > 70 {
                format!("{}...", &node_name[..67])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<70} | {}",
                (rank + 1).to_string().yellow(),
                truncated,
                format!("{:.6}", score).green()
            );
        }
    }

    println!();
    println!(
        "{}",
        "HITS identifies 'hubs' (nodes pointing to many authorities) and 'authorities' (highly cited nodes)."
            .dimmed()
    );
    println!(
        "{}",
        "Useful for finding important resources in directed knowledge graphs.".dimmed()
    );

    Ok(())
}
