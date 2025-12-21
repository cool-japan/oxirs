//! Community detection algorithms
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use colored::Colorize;
use scirs2_core::random::Random;
use scirs2_graph::louvain_communities_result;
use std::collections::HashMap;

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute community detection using label propagation
pub fn execute_community_detection(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Community Detection (Label Propagation with SciRS2)"
            .green()
            .bold()
    );
    println!();

    let n = graph.node_count();
    let mut labels = scirs2_core::ndarray_ext::Array1::from_vec((0..n).collect());
    let mut new_labels = labels.clone();
    let mut rng = Random::default();
    let max_iterations = 10;

    for iter in 0..max_iterations {
        let mut changed = false;
        let mut nodes: Vec<usize> = (0..n).collect();

        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            nodes.swap(i, j);
        }

        for &node in &nodes {
            let neighbors = graph.neighbors(node);
            if neighbors.is_empty() {
                continue;
            }

            let mut label_counts: HashMap<usize, usize> = HashMap::new();
            for &neighbor in neighbors {
                *label_counts.entry(labels[neighbor]).or_insert(0) += 1;
            }

            if let Some((&most_common, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
                if most_common != labels[node] {
                    new_labels[node] = most_common;
                    changed = true;
                }
            }
        }

        labels.assign(&new_labels);
        if !changed {
            println!(
                "{}",
                format!("Converged after {} iterations", iter + 1).green()
            );
            break;
        }
    }

    let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &label) in labels.iter().enumerate() {
        communities.entry(label).or_default().push(node);
    }

    let community_count = communities.len();
    println!(
        "{}",
        format!("Detected {} communities", community_count)
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(100).cyan());

    let mut sorted_communities: Vec<_> = communities.into_iter().collect();
    sorted_communities.sort_by_key(|(_, nodes)| std::cmp::Reverse(nodes.len()));

    for (i, (_label, nodes)) in sorted_communities
        .iter()
        .take(config.top_k.min(10))
        .enumerate()
    {
        println!();
        println!(
            "{}",
            format!("Community {} ({} nodes):", i + 1, nodes.len())
                .yellow()
                .bold()
        );

        for (j, &node_id) in nodes.iter().take(10).enumerate() {
            if let Some(node_name) = graph.get_node_name(node_id) {
                let truncated = if node_name.len() > 85 {
                    format!("{}...", &node_name[..82])
                } else {
                    node_name.to_string()
                };
                println!("  {} {}", (j + 1).to_string().cyan(), truncated);
            }
        }

        if nodes.len() > 10 {
            println!("  {} more...", (nodes.len() - 10).to_string().dimmed());
        }
    }

    Ok(())
}

/// Execute Louvain community detection
pub fn execute_louvain_communities(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Louvain Community Detection (SciRS2-Graph - Modularity Optimization)"
            .green()
            .bold()
    );
    println!(
        "{}",
        "  Optimizing modularity to find communities...".dimmed()
    );
    println!();

    let graph = rdf_graph.to_scirs2_graph()?;
    let community_result = louvain_communities_result(&graph);
    let communities = &community_result.node_communities;

    let mut community_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node, &community_id) in communities {
        community_map.entry(community_id).or_default().push(*node);
    }

    let community_count = community_map.len();
    println!(
        "{}",
        format!(
            "Detected {} communities using Louvain method",
            community_count
        )
        .cyan()
        .bold()
    );
    println!("{}", "─".repeat(100).cyan());

    let mut sorted_communities: Vec<_> = community_map.into_iter().collect();
    sorted_communities.sort_by_key(|(_, nodes)| std::cmp::Reverse(nodes.len()));

    let display_count = config.top_k.min(10);
    for (i, (community_id, nodes)) in sorted_communities.iter().take(display_count).enumerate() {
        println!();
        println!(
            "{}",
            format!(
                "Community {} (ID: {}, {} nodes):",
                i + 1,
                community_id,
                nodes.len()
            )
            .yellow()
            .bold()
        );

        let sample_size = 10;
        for (j, &node_id) in nodes.iter().take(sample_size).enumerate() {
            if let Some(node_name) = rdf_graph.get_node_name(node_id) {
                let truncated = if node_name.len() > 85 {
                    format!("{}...", &node_name[..82])
                } else {
                    node_name.to_string()
                };
                println!("  {} {}", (j + 1).to_string().cyan(), truncated);
            }
        }

        if nodes.len() > sample_size {
            println!(
                "  {} more...",
                (nodes.len() - sample_size).to_string().dimmed()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Louvain method optimizes modularity to find densely connected communities.".dimmed()
    );
    println!(
        "{}",
        "Higher modularity indicates better community structure.".dimmed()
    );

    Ok(())
}
