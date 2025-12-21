//! Extended motif pattern analysis
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Context;
use anyhow::Result;
use colored::Colorize;
use scirs2_graph::algorithms::motifs::{find_motifs, MotifType};

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute extended motif analysis
pub fn execute_extended_motifs(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Extended Motif Analysis (Squares, Stars, Cliques, Paths)"
            .green()
            .bold()
    );
    println!();

    let scirs2_graph = rdf_graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Analyzing graph motifs...");
    println!();

    let triangles = find_motifs(&scirs2_graph, MotifType::Triangle);
    let squares = find_motifs(&scirs2_graph, MotifType::Square);
    let stars = find_motifs(&scirs2_graph, MotifType::Star3);
    let cliques = find_motifs(&scirs2_graph, MotifType::Clique4);
    let paths = find_motifs(&scirs2_graph, MotifType::Path3);

    println!("{}", "Motif Counts:".yellow().bold());
    println!("{}", "─".repeat(70).cyan());
    println!("{:>20} | {:>15} | Description", "Motif Type", "Count");
    println!("{}", "─".repeat(70).cyan());

    println!(
        "{:>20} | {:>15} | 3-node cycles (transitivity)",
        "Triangles",
        triangles.len().to_string().green()
    );
    println!(
        "{:>20} | {:>15} | 4-node cycles (rectangles)",
        "Squares",
        squares.len().to_string().green()
    );
    println!(
        "{:>20} | {:>15} | Hub with 3 spokes",
        "3-Stars",
        stars.len().to_string().green()
    );
    println!(
        "{:>20} | {:>15} | Fully connected 4-node subgraphs",
        "4-Cliques",
        cliques.len().to_string().green()
    );
    println!(
        "{:>20} | {:>15} | Linear 4-node paths",
        "3-Paths",
        paths.len().to_string().green()
    );

    let total_motifs = triangles.len() + squares.len() + stars.len() + cliques.len() + paths.len();
    println!("{}", "─".repeat(70).cyan());
    println!(
        "{:>20} | {:>15}",
        "Total Motifs",
        total_motifs.to_string().cyan().bold()
    );

    let motif_types = vec![
        ("Triangles", &triangles, 3),
        ("Squares", &squares, 4),
        ("3-Stars", &stars, 4),
        ("4-Cliques", &cliques, 4),
        ("3-Paths", &paths, 4),
    ];

    for (motif_name, motif_list, expected_size) in motif_types {
        if !motif_list.is_empty() {
            println!();
            println!(
                "{}",
                format!(
                    "Sample {} (showing up to {}):",
                    motif_name,
                    config.top_k.min(3)
                )
                .yellow()
                .bold()
            );
            println!("{}", "─".repeat(100).cyan());

            for (i, motif) in motif_list.iter().take(config.top_k.min(3)).enumerate() {
                println!(
                    "  {} {}:",
                    motif_name.trim_end_matches('s'),
                    (i + 1).to_string().cyan()
                );

                for (j, node) in motif.iter().take(expected_size).enumerate() {
                    let node_str = format!("{:?}", node);
                    if let Some(&node_id) = rdf_graph.node_to_id.get(&node_str) {
                        if let Some(node_name) = rdf_graph.get_node_name(node_id) {
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

            if motif_list.len() > config.top_k.min(3) {
                println!(
                    "  {} more {}...",
                    (motif_list.len() - config.top_k.min(3))
                        .to_string()
                        .dimmed(),
                    motif_name.to_lowercase()
                );
            }
        }
    }

    println!();
    println!(
        "{}",
        "Motifs reveal recurring structural patterns in the RDF graph.".dimmed()
    );
    println!(
        "{}",
        "Different motifs indicate different types of relationships and organization.".dimmed()
    );

    Ok(())
}
