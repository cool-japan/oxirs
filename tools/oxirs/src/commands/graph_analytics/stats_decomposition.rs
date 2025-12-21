//! Graph decomposition and structural analysis
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Context;
use anyhow::Result;
use colored::Colorize;
use scirs2_graph::{center_nodes, diameter, k_core_decomposition, radius};
use std::collections::HashMap;

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute K-core decomposition
pub fn execute_kcore_decomposition(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "K-Core Decomposition (Dense Subgraph Discovery)"
            .green()
            .bold()
    );
    println!();

    let scirs2_graph = graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Computing k-core decomposition...");
    println!();

    let k_cores = k_core_decomposition(&scirs2_graph);
    let max_core = k_cores.values().copied().max().unwrap_or(0);

    println!("{}", "Core Statistics:".yellow().bold());
    println!("  Maximum core number: {}", max_core.to_string().green());
    println!();

    let mut core_counts: HashMap<usize, usize> = HashMap::new();
    for &core_num in k_cores.values() {
        *core_counts.entry(core_num).or_insert(0) += 1;
    }

    println!("{}", "Core Distribution:".yellow().bold());
    println!("{}", "─".repeat(60).cyan());
    println!("{:>10} | {:>15} | Density", "Core", "Node Count");
    println!("{}", "─".repeat(60).cyan());

    let mut sorted_cores: Vec<_> = core_counts.iter().collect();
    sorted_cores.sort_by_key(|(k, _)| std::cmp::Reverse(**k));

    for (&core_num, &count) in sorted_cores.iter().take(config.top_k) {
        let density = count as f64 / graph.node_count() as f64 * 100.0;
        let bar_length = (density / 2.0) as usize;
        let bar = "█".repeat(bar_length);
        println!(
            "{:>10} | {:>15} | {} {:.1}%",
            core_num.to_string().yellow(),
            count.to_string().green(),
            bar.cyan(),
            density
        );
    }

    if let Some(k) = config.k_core_value {
        println!();
        println!("{}", format!("Nodes in {}-core:", k).yellow().bold());
        println!("{}", "─".repeat(100).cyan());

        let k_core_nodes: Vec<_> = k_cores
            .iter()
            .filter(|(_, &core)| core == k)
            .map(|(&node, _)| node)
            .collect();

        println!("  Found {} nodes in {}-core", k_core_nodes.len(), k);

        for (i, &node_id) in k_core_nodes.iter().take(config.top_k).enumerate() {
            if let Some(node_name) = graph.get_node_name(node_id) {
                let truncated = if node_name.len() > 85 {
                    format!("{}...", &node_name[..82])
                } else {
                    node_name.to_string()
                };
                println!("  {} {}", (i + 1).to_string().cyan(), truncated);
            }
        }

        if k_core_nodes.len() > config.top_k {
            println!(
                "  {} more...",
                (k_core_nodes.len() - config.top_k).to_string().dimmed()
            );
        }
    } else {
        println!();
        println!(
            "{}",
            format!("Sample nodes from {}-core (maximum):", max_core)
                .yellow()
                .bold()
        );
        println!("{}", "─".repeat(100).cyan());

        let max_core_nodes: Vec<_> = k_cores
            .iter()
            .filter(|(_, &core)| core == max_core)
            .map(|(&node, _)| node)
            .collect();

        for (i, &node_id) in max_core_nodes.iter().take(config.top_k).enumerate() {
            if let Some(node_name) = graph.get_node_name(node_id) {
                let truncated = if node_name.len() > 85 {
                    format!("{}...", &node_name[..82])
                } else {
                    node_name.to_string()
                };
                println!("  {} {}", (i + 1).to_string().cyan(), truncated);
            }
        }

        if max_core_nodes.len() > config.top_k {
            println!(
                "  {} more...",
                (max_core_nodes.len() - config.top_k).to_string().dimmed()
            );
        }
    }

    println!();
    println!(
        "{}",
        "K-core decomposition identifies densely connected subgraphs.".dimmed()
    );
    println!(
        "{}",
        "Higher core numbers indicate nodes in denser parts of the graph.".dimmed()
    );

    Ok(())
}

/// Execute triangle counting
pub fn execute_triangle_counting(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Triangle Counting (Clustering Coefficient Analysis)"
            .green()
            .bold()
    );
    println!();

    let scirs2_graph = graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Finding triangles in the graph...");
    println!();

    use scirs2_graph::algorithms::motifs::{find_motifs, MotifType};
    let triangles = find_motifs(&scirs2_graph, MotifType::Triangle);
    let triangle_count = triangles.len();
    let node_count = graph.node_count();

    println!("{}", "Triangle Statistics:".yellow().bold());
    println!("  Total triangles: {}", triangle_count.to_string().green());
    println!("  Nodes: {}", node_count.to_string().yellow());
    println!();

    let total_triples = calculate_connected_triples(graph);
    let global_clustering = if total_triples > 0 {
        (3.0 * triangle_count as f64) / total_triples as f64
    } else {
        0.0
    };

    println!("{}", "Clustering Metrics:".yellow().bold());
    println!(
        "  Global clustering coefficient: {}",
        format!("{:.6}", global_clustering).green()
    );

    let mut node_triangle_count: HashMap<usize, usize> = HashMap::new();
    for triangle in &triangles {
        for node in triangle {
            if let Some(&node_id) = graph.node_to_id.get(&format!("{:?}", node)) {
                *node_triangle_count.entry(node_id).or_insert(0) += 1;
            }
        }
    }

    println!();
    println!(
        "{}",
        format!("Top {} Nodes by Triangle Participation:", config.top_k)
            .yellow()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());
    println!("{:>5} | {:<70} | {:>12}", "Rank", "Resource", "Triangles");
    println!("{}", "─".repeat(100).cyan());

    let mut sorted_nodes: Vec<_> = node_triangle_count.iter().collect();
    sorted_nodes.sort_by_key(|(_, &count)| std::cmp::Reverse(count));

    for (i, (&node_id, &count)) in sorted_nodes.iter().take(config.top_k).enumerate() {
        if let Some(node_name) = graph.get_node_name(node_id) {
            let truncated = if node_name.len() > 70 {
                format!("{}...", &node_name[..67])
            } else {
                node_name.to_string()
            };
            println!(
                "{:>5} | {:<70} | {}",
                (i + 1).to_string().yellow(),
                truncated,
                count.to_string().green()
            );
        }
    }

    if !triangles.is_empty() {
        println!();
        println!(
            "{}",
            "Sample Triangles (showing up to 5):"
                .to_string()
                .yellow()
                .bold()
        );
        println!("{}", "─".repeat(100).cyan());

        for (i, triangle) in triangles.iter().take(5).enumerate() {
            println!("  Triangle {}:", (i + 1).to_string().cyan());
            for (j, node) in triangle.iter().enumerate() {
                let node_str = format!("{:?}", node);
                if let Some(&node_id) = graph.node_to_id.get(&node_str) {
                    if let Some(node_name) = graph.get_node_name(node_id) {
                        let truncated = if node_name.len() > 85 {
                            format!("{}...", &node_name[..82])
                        } else {
                            node_name.to_string()
                        };
                        println!("    {} - {}", (j + 1), truncated);
                    }
                }
            }
        }

        if triangles.len() > 5 {
            println!(
                "  {} more triangles...",
                (triangles.len() - 5).to_string().dimmed()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Triangles indicate transitive relationships in the graph.".dimmed()
    );
    println!(
        "{}",
        "High clustering coefficient suggests community structure.".dimmed()
    );

    Ok(())
}

/// Calculate the number of connected triples in the graph
fn calculate_connected_triples(graph: &RdfGraph) -> usize {
    let mut triples = 0;
    for node in 0..graph.node_count() {
        let neighbors = graph.neighbors(node);
        let degree = neighbors.len();
        if degree >= 2 {
            triples += degree * (degree - 1) / 2;
        }
    }
    triples
}

/// Execute diameter and radius calculation
pub fn execute_diameter_radius(graph: &RdfGraph, _config: &AnalyticsConfig) -> Result<()> {
    println!("{}", "Graph Diameter and Radius Analysis".green().bold());
    println!();

    let scirs2_graph = graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Computing graph diameter and radius...");
    println!();

    let graph_diameter = diameter(&scirs2_graph);
    let graph_radius = radius(&scirs2_graph);

    println!("{}", "Graph Metrics:".yellow().bold());
    println!("{}", "─".repeat(60).cyan());

    match graph_diameter {
        Some(d) => {
            println!(
                "  Diameter (maximum eccentricity): {}",
                format!("{:.2}", d).green()
            );
        }
        None => {
            println!("  Diameter: {}", "undefined (disconnected graph)".dimmed());
        }
    }

    match graph_radius {
        Some(r) => {
            println!(
                "  Radius (minimum eccentricity):   {}",
                format!("{:.2}", r).green()
            );
        }
        None => {
            println!("  Radius: {}", "undefined (disconnected graph)".dimmed());
        }
    }

    if let (Some(d), Some(r)) = (graph_diameter, graph_radius) {
        println!();
        println!("{}", "Interpretation:".yellow().bold());
        println!(
            "  • Diameter ({:.2}) is the longest shortest path in the graph",
            d
        );
        println!(
            "  • Radius ({:.2}) is the smallest eccentricity of any node",
            r
        );
        println!("  • Center nodes have eccentricity equal to the radius");

        let ratio = d / r;
        println!();
        println!(
            "  Diameter/Radius ratio: {}",
            format!("{:.2}", ratio).cyan()
        );

        if ratio < 1.5 {
            println!("  → {}", "Compact graph structure".green());
        } else if ratio < 2.5 {
            println!("  → {}", "Moderate graph structure".yellow());
        } else {
            println!("  → {}", "Extended graph structure".red());
        }
    }

    println!();
    println!(
        "{}",
        "Diameter and radius measure graph compactness and connectivity.".dimmed()
    );
    println!(
        "{}",
        "Lower values indicate more compact, well-connected graphs.".dimmed()
    );

    Ok(())
}

/// Execute center nodes identification
pub fn execute_center_nodes(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Graph Center Nodes (Minimum Eccentricity)".green().bold()
    );
    println!();

    let scirs2_graph = graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Finding center nodes...");
    println!();

    let center_node_ids: Vec<usize> = center_nodes(&scirs2_graph);
    let graph_radius = radius(&scirs2_graph);

    println!("{}", "Center Nodes:".yellow().bold());
    println!("{}", "─".repeat(100).cyan());

    if center_node_ids.is_empty() {
        println!(
            "  {} No center nodes found (disconnected graph)",
            "⚠".yellow()
        );
    } else {
        if let Some(r) = graph_radius {
            println!(
                "  Found {} center nodes with eccentricity {:.2}",
                center_node_ids.len().to_string().green(),
                r.to_string().yellow()
            );
        } else {
            println!(
                "  Found {} center nodes",
                center_node_ids.len().to_string().green()
            );
        }

        println!();

        let display_count = config.top_k.min(center_node_ids.len());
        println!("{:>5} | {:<85}", "Rank", "Resource");
        println!("{}", "─".repeat(100).cyan());

        for (i, &node_id) in center_node_ids.iter().take(display_count).enumerate() {
            if let Some(node_name) = graph.get_node_name(node_id) {
                let truncated = if node_name.len() > 85 {
                    format!("{}...", &node_name[..82])
                } else {
                    node_name.to_string()
                };
                println!("{:>5} | {}", (i + 1).to_string().yellow(), truncated);
            }
        }

        if center_node_ids.len() > display_count {
            println!(
                "  {} more center nodes...",
                (center_node_ids.len() - display_count).to_string().dimmed()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Center nodes have minimum eccentricity (minimum of maximum distances).".dimmed()
    );
    println!(
        "{}",
        "They are the most 'central' nodes in terms of distance to other nodes.".dimmed()
    );

    Ok(())
}
