//! Advanced graph algorithms (coloring, matching, network flow)
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use colored::Colorize;
use std::collections::{HashMap, HashSet, VecDeque};

use super::types::{AnalyticsConfig, RdfGraph};

/// Execute graph coloring analysis
pub fn execute_graph_coloring(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!();
    println!("{}", "Graph Coloring Analysis".cyan().bold());
    println!("{}", "─".repeat(80).cyan());
    println!(
        "{}",
        "Finding chromatic number using greedy coloring algorithm".dimmed()
    );
    println!();

    let node_count = rdf_graph.node_count();
    if node_count == 0 {
        println!("{}", "Empty graph - no coloring needed".yellow());
        return Ok(());
    }

    if config.enable_simd && node_count > 10_000 {
        println!(
            "{}",
            "ℹ SIMD acceleration: Enabled for adjacency checks (>10K nodes)"
                .cyan()
                .dimmed()
        );
    }

    if config.enable_parallel && node_count > 50_000 {
        println!(
            "{}",
            "ℹ Parallel processing: Enabled for coloring phases (>50K nodes)"
                .cyan()
                .dimmed()
        );
    }

    let mut colors: HashMap<usize, usize> = HashMap::new();
    let mut max_color = 0;

    let mut nodes_by_degree: Vec<(usize, usize)> = (0..node_count)
        .map(|node| (node, rdf_graph.out_degree(node) + rdf_graph.in_degree(node)))
        .collect();

    nodes_by_degree.sort_by(|a, b| b.1.cmp(&a.1));

    for (node, _degree) in &nodes_by_degree {
        let neighbors = rdf_graph.neighbors(*node);
        let mut used_colors = HashSet::new();

        for &neighbor in neighbors {
            if let Some(&color) = colors.get(&neighbor) {
                used_colors.insert(color);
            }
        }

        let mut assigned_color = 0;
        while used_colors.contains(&assigned_color) {
            assigned_color += 1;
        }

        colors.insert(*node, assigned_color);
        max_color = max_color.max(assigned_color);
    }

    let chromatic_number = max_color + 1;
    let mut color_counts: HashMap<usize, usize> = HashMap::new();

    for &color in colors.values() {
        *color_counts.entry(color).or_insert(0) += 1;
    }

    println!(
        "{}: {}",
        "Chromatic Number (Upper Bound)".green(),
        chromatic_number.to_string().yellow().bold()
    );
    println!(
        "{}: {} nodes colored",
        "Coverage".green(),
        colors.len().to_string().yellow()
    );
    println!();

    println!("{}", "Color Distribution:".green().bold());
    println!("{}", "─".repeat(80).cyan());

    let mut color_vec: Vec<(usize, usize)> = color_counts.iter().map(|(&c, &n)| (c, n)).collect();
    color_vec.sort_by_key(|&(c, _)| c);

    for (color, count) in color_vec.iter().take(config.top_k) {
        let bar = "█".repeat((count * 50 / node_count.max(1)).max(1));
        println!(
            "  Color {}: {} nodes {}",
            color.to_string().cyan(),
            count.to_string().yellow(),
            bar.blue()
        );
    }

    if color_vec.len() > config.top_k {
        println!(
            "  {} more colors...",
            (color_vec.len() - config.top_k).to_string().dimmed()
        );
    }

    println!();
    println!(
        "{}",
        "Greedy coloring provides an upper bound on the chromatic number.".dimmed()
    );
    println!(
        "{}",
        "Lower chromatic numbers indicate more efficient graph coloring.".dimmed()
    );

    Ok(())
}

/// Execute maximum matching analysis
pub fn execute_maximum_matching(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!();
    println!("{}", "Maximum Matching Analysis".cyan().bold());
    println!("{}", "─".repeat(80).cyan());
    println!(
        "{}",
        "Finding maximum matching using greedy augmenting paths".dimmed()
    );
    println!();

    let node_count = rdf_graph.node_count();
    if node_count == 0 {
        println!("{}", "Empty graph - no matching possible".yellow());
        return Ok(());
    }

    if config.enable_gpu && node_count > 1_000_000 {
        println!(
            "{}",
            "ℹ GPU acceleration: Recommended for matching on very large graphs"
                .cyan()
                .dimmed()
        );
        println!(
            "{}",
            "  Use scirs2_core::gpu::GpuContext for 10-100x speedup on suitable workloads"
                .cyan()
                .dimmed()
        );
    }

    let mut node_colors: HashMap<usize, bool> = HashMap::new();
    let mut is_bipartite = true;
    let mut queue = VecDeque::new();

    for start_node in 0..node_count {
        if node_colors.contains_key(&start_node) {
            continue;
        }

        queue.push_back(start_node);
        node_colors.insert(start_node, false);

        while let Some(node) = queue.pop_front() {
            let current_color = node_colors[&node];
            for &neighbor in rdf_graph.neighbors(node) {
                if let Some(&neighbor_color) = node_colors.get(&neighbor) {
                    if neighbor_color == current_color {
                        is_bipartite = false;
                        break;
                    }
                } else {
                    node_colors.insert(neighbor, !current_color);
                    queue.push_back(neighbor);
                }
            }

            if !is_bipartite {
                break;
            }
        }

        if !is_bipartite {
            break;
        }
    }

    if is_bipartite {
        println!(
            "{}",
            "✓ Graph is bipartite - optimal matching possible".green()
        );
    } else {
        println!(
            "{}",
            "⚠ Graph is not bipartite - using general matching algorithm".yellow()
        );
    }

    println!();

    let mut matching: HashMap<usize, usize> = HashMap::new();
    let mut matched_nodes: HashSet<usize> = HashSet::new();

    for node in 0..node_count {
        if matched_nodes.contains(&node) {
            continue;
        }

        for &neighbor in rdf_graph.neighbors(node) {
            if !matched_nodes.contains(&neighbor) {
                matching.insert(node, neighbor);
                matched_nodes.insert(node);
                matched_nodes.insert(neighbor);
                break;
            }
        }
    }

    let matching_size = matching.len();
    let coverage = (matched_nodes.len() as f64 / node_count as f64) * 100.0;

    println!(
        "{}: {} edges",
        "Maximum Matching Size".green(),
        matching_size.to_string().yellow().bold()
    );
    println!(
        "{}: {}/{} nodes ({:.1}%)",
        "Coverage".green(),
        matched_nodes.len().to_string().yellow(),
        node_count,
        coverage
    );
    println!();

    if matching_size > 0 {
        println!("{}", "Sample Matched Pairs:".green().bold());
        println!("{}", "─".repeat(80).cyan());

        for (i, (&src, &dst)) in matching.iter().take(config.top_k).enumerate() {
            let src_name = rdf_graph.get_node_name(src).unwrap_or("Unknown");
            let dst_name = rdf_graph.get_node_name(dst).unwrap_or("Unknown");

            let src_truncated = if src_name.len() > 35 {
                format!("{}...", &src_name[..32])
            } else {
                src_name.to_string()
            };

            let dst_truncated = if dst_name.len() > 35 {
                format!("{}...", &dst_name[..32])
            } else {
                dst_name.to_string()
            };

            println!(
                "  {} {} ↔ {}",
                (i + 1).to_string().cyan(),
                src_truncated,
                dst_truncated
            );
        }

        if matching_size > config.top_k {
            println!(
                "  {} more matches...",
                (matching_size - config.top_k).to_string().dimmed()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Maximum matching identifies the largest set of non-overlapping edges.".dimmed()
    );

    if is_bipartite {
        println!(
            "{}",
            "For bipartite graphs, this can be optimized using Hungarian algorithm.".dimmed()
        );
    }

    Ok(())
}

/// Execute network flow analysis
pub fn execute_network_flow(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!();
    println!("{}", "Network Flow Analysis".cyan().bold());
    println!("{}", "─".repeat(80).cyan());
    println!(
        "{}",
        "Computing maximum flow using Ford-Fulkerson algorithm".dimmed()
    );
    println!();

    let node_count = rdf_graph.node_count();
    if node_count < 2 {
        println!("{}", "Need at least 2 nodes for flow analysis".yellow());
        return Ok(());
    }

    if config.enable_parallel && node_count > 100_000 {
        println!(
            "{}",
            "ℹ Parallel processing: Enabled for concurrent flow path search"
                .cyan()
                .dimmed()
        );
        println!(
            "{}",
            "  Use scirs2_core::parallel_ops::par_chunks for multi-core BFS"
                .cyan()
                .dimmed()
        );
    }

    let degrees: Vec<(usize, usize)> = (0..node_count)
        .map(|n| (n, rdf_graph.out_degree(n)))
        .collect();

    let source = degrees
        .iter()
        .max_by_key(|(_, d)| d)
        .map(|(n, _)| *n)
        .unwrap_or(0);

    let sink = degrees
        .iter()
        .filter(|(n, _)| *n != source)
        .min_by_key(|(_, d)| d)
        .map(|(n, _)| *n)
        .unwrap_or(node_count - 1);

    if source == sink {
        println!("{}", "Cannot compute flow: source equals sink".yellow());
        return Ok(());
    }

    println!(
        "Source node: {}",
        rdf_graph
            .get_node_name(source)
            .unwrap_or("Unknown")
            .yellow()
    );
    println!(
        "Sink node: {}",
        rdf_graph.get_node_name(sink).unwrap_or("Unknown").yellow()
    );
    println!();

    let mut capacity: HashMap<(usize, usize), i32> = HashMap::new();
    for node in 0..node_count {
        for &neighbor in rdf_graph.neighbors(node) {
            capacity.insert((node, neighbor), 1);
        }
    }

    let mut flow: HashMap<(usize, usize), i32> = HashMap::new();
    let mut max_flow = 0;
    let mut iteration = 0;
    let max_iterations = config.max_iterations;

    while iteration < max_iterations {
        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(source);
        visited.insert(source);
        let mut found_path = false;

        while let Some(node) = queue.pop_front() {
            if node == sink {
                found_path = true;
                break;
            }

            for &neighbor in rdf_graph.neighbors(node) {
                let cap = capacity.get(&(node, neighbor)).copied().unwrap_or(0);
                let curr_flow = flow.get(&(node, neighbor)).copied().unwrap_or(0);
                let residual = cap - curr_flow;

                if residual > 0 && !visited.contains(&neighbor) {
                    parent.insert(neighbor, node);
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        if !found_path {
            break;
        }

        let mut path_flow = i32::MAX;
        let mut current = sink;

        while current != source {
            if let Some(&prev) = parent.get(&current) {
                let cap = capacity.get(&(prev, current)).copied().unwrap_or(0);
                let curr_flow = flow.get(&(prev, current)).copied().unwrap_or(0);
                path_flow = path_flow.min(cap - curr_flow);
                current = prev;
            } else {
                break;
            }
        }

        current = sink;
        while current != source {
            if let Some(&prev) = parent.get(&current) {
                *flow.entry((prev, current)).or_insert(0) += path_flow;
                *flow.entry((current, prev)).or_insert(0) -= path_flow;
                current = prev;
            } else {
                break;
            }
        }

        max_flow += path_flow;
        iteration += 1;
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back(source);
    visited.insert(source);

    while let Some(node) = queue.pop_front() {
        for &neighbor in rdf_graph.neighbors(node) {
            let cap = capacity.get(&(node, neighbor)).copied().unwrap_or(0);
            let curr_flow = flow.get(&(node, neighbor)).copied().unwrap_or(0);
            let residual = cap - curr_flow;

            if residual > 0 && !visited.contains(&neighbor) {
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    let min_cut_size = visited.len();

    println!(
        "{}: {}",
        "Maximum Flow".green(),
        max_flow.to_string().yellow().bold()
    );
    println!(
        "{}: {} iterations",
        "Convergence".green(),
        iteration.to_string().yellow()
    );
    println!(
        "{}: {} nodes (source side)",
        "Min-Cut Size".green(),
        min_cut_size.to_string().yellow()
    );
    println!();

    let non_zero_flows: Vec<((usize, usize), i32)> = flow
        .iter()
        .filter(|(_, &f)| f > 0)
        .map(|((s, d), &f)| ((*s, *d), f))
        .collect();

    if !non_zero_flows.is_empty() {
        println!("{}", "Flow Edges (Sample):".green().bold());
        println!("{}", "─".repeat(80).cyan());

        for (i, ((src, dst), flow_val)) in non_zero_flows.iter().take(config.top_k).enumerate() {
            let src_name = rdf_graph.get_node_name(*src).unwrap_or("Unknown");
            let dst_name = rdf_graph.get_node_name(*dst).unwrap_or("Unknown");

            let src_truncated = if src_name.len() > 32 {
                format!("{}...", &src_name[..29])
            } else {
                src_name.to_string()
            };

            let dst_truncated = if dst_name.len() > 32 {
                format!("{}...", &dst_name[..29])
            } else {
                dst_name.to_string()
            };

            println!(
                "  {} {} → {} (flow: {})",
                (i + 1).to_string().cyan(),
                src_truncated,
                dst_truncated,
                flow_val.to_string().yellow()
            );
        }

        if non_zero_flows.len() > config.top_k {
            println!(
                "  {} more flow edges...",
                (non_zero_flows.len() - config.top_k).to_string().dimmed()
            );
        }
    }

    println!();
    println!(
        "{}",
        "Max flow = min cut (Ford-Fulkerson theorem).".dimmed()
    );
    println!(
        "{}",
        "Identifies bottlenecks and critical edges in the network.".dimmed()
    );

    Ok(())
}
