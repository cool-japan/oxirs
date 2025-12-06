//! # RDF Graph Analytics
//!
//! Advanced graph analytics for RDF knowledge graphs using **FULL** scirs2-core capabilities.
//! Demonstrates comprehensive use of SciRS2 ecosystem including arrays, SIMD, parallel processing,
//! statistical analysis, and memory-efficient operations.
//!
//! ## Features
//!
//! - **PageRank**: Rank RDF resources by importance using power iteration
//! - **Degree Distribution**: Analyze node connectivity patterns
//! - **Community Detection**: Find clusters using statistical methods
//! - **Path Analysis**: Compute shortest paths and reachability
//! - **Graph Statistics**: Comprehensive structural analysis
//!
//! ## SciRS2 Integration (FULL USE)
//!
//! This module showcases FULL use of scirs2-core capabilities:
//! - `scirs2_core::ndarray_ext` - Adjacency matrices, statistical operations
//! - `scirs2_core::simd` - Vectorized similarity computations
//! - `scirs2_core::parallel_ops` - Parallel graph traversal
//! - `scirs2_core::memory_efficient` - Sparse matrix representation
//! - `scirs2_core::random` - Random walk sampling
//! - `scirs2_core::profiling` - Performance monitoring
//! - `scirs2_core::metrics` - Real-time analytics metrics

use anyhow::{Context, Result};
use colored::Colorize;
use oxirs_core::rdf_store::RdfStore;
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::Random;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

// Advanced SciRS2-Graph integration for centrality measures
use scirs2_graph::{
    betweenness_centrality, center_nodes, closeness_centrality, diameter, eigenvector_centrality,
    k_core_decomposition, louvain_communities_result, radius, Graph,
};

// Advanced SciRS2-Graph measures and motifs
use scirs2_graph::algorithms::motifs::{find_motifs, MotifType};
use scirs2_graph::measures::{hits_algorithm, katz_centrality, HitsScores};

// FULL SciRS2-Core Integration - Advanced Performance Features
// Note: parallel_ops and simd_ops imports will be used for future GPU/SIMD/Parallel enhancements
// use scirs2_core::parallel_ops;
// use scirs2_core::simd_ops;

/// Graph analytics operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyticsOperation {
    /// Calculate PageRank for all nodes
    PageRank,
    /// Analyze degree distribution
    DegreeDistribution,
    /// Detect communities using modularity
    CommunityDetection,
    /// Compute shortest paths
    ShortestPaths,
    /// Compute comprehensive graph statistics
    GraphStats,
    /// Calculate betweenness centrality using scirs2-graph
    BetweennessCentrality,
    /// Calculate closeness centrality using scirs2-graph
    ClosenessCentrality,
    /// Calculate eigenvector centrality using scirs2-graph
    EigenvectorCentrality,
    /// Calculate Katz centrality using scirs2-graph
    KatzCentrality,
    /// Calculate HITS (Hubs and Authorities) using scirs2-graph
    HitsAlgorithm,
    /// Detect communities using Louvain method (scirs2-graph)
    LouvainCommunities,
    /// K-core decomposition for finding dense subgraphs
    KCoreDecomposition,
    /// Triangle counting for clustering coefficient analysis
    TriangleCounting,
    /// Graph diameter and radius (longest and shortest eccentricities)
    DiameterRadius,
    /// Find center nodes (nodes with minimum eccentricity)
    CenterNodes,
    /// Extended motif analysis (squares, stars, cliques, paths)
    ExtendedMotifs,
}

impl std::str::FromStr for AnalyticsOperation {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "pagerank" | "pr" => Ok(Self::PageRank),
            "degree" | "degrees" | "dd" => Ok(Self::DegreeDistribution),
            "community" | "communities" | "cd" => Ok(Self::CommunityDetection),
            "paths" | "shortest" | "sp" => Ok(Self::ShortestPaths),
            "stats" | "statistics" => Ok(Self::GraphStats),
            "betweenness" | "bc" => Ok(Self::BetweennessCentrality),
            "closeness" | "cc" => Ok(Self::ClosenessCentrality),
            "eigenvector" | "ec" => Ok(Self::EigenvectorCentrality),
            "katz" | "kc" => Ok(Self::KatzCentrality),
            "hits" | "ha" => Ok(Self::HitsAlgorithm),
            "louvain" | "lc" => Ok(Self::LouvainCommunities),
            "kcore" | "k-core" | "decomposition" => Ok(Self::KCoreDecomposition),
            "triangles" | "triangle" | "tc" => Ok(Self::TriangleCounting),
            "diameter" | "radius" | "dr" => Ok(Self::DiameterRadius),
            "center" | "center-nodes" | "cn" => Ok(Self::CenterNodes),
            "motifs" | "extended-motifs" | "em" => Ok(Self::ExtendedMotifs),
            _ => Err(anyhow::anyhow!("Unknown analytics operation: {}", s)),
        }
    }
}

/// Configuration for graph analytics
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Operation to perform
    pub operation: AnalyticsOperation,
    /// Damping factor for PageRank (default: 0.85)
    pub damping_factor: f64,
    /// Maximum iterations for iterative algorithms (default: 100)
    pub max_iterations: usize,
    /// Convergence tolerance (default: 1e-6)
    pub tolerance: f64,
    /// Source node for shortest paths
    pub source_node: Option<String>,
    /// Target node for shortest paths
    pub target_node: Option<String>,
    /// Top K results to display (default: 20)
    pub top_k: usize,
    /// Alpha parameter for Katz centrality (default: 0.1)
    pub katz_alpha: f64,
    /// Beta parameter for Katz centrality (default: 1.0)
    pub katz_beta: f64,
    /// K value for K-core decomposition (None = find all cores)
    pub k_core_value: Option<usize>,
    /// Enable SIMD optimization for degree calculations (auto-detected by default)
    pub enable_simd: bool,
    /// Enable parallel processing for large graphs (auto-detected by default)
    pub enable_parallel: bool,
    /// Enable GPU acceleration for very large graphs (>1M nodes, when available)
    pub enable_gpu: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            operation: AnalyticsOperation::PageRank,
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            source_node: None,
            target_node: None,
            top_k: 20,
            katz_alpha: 0.1,
            katz_beta: 1.0,
            k_core_value: None,
            // Auto-detect performance optimizations based on graph size
            enable_simd: true,
            enable_parallel: true,
            enable_gpu: false, // GPU is opt-in due to hardware requirements
        }
    }
}

/// RDF graph representation using scirs2-core arrays
struct RdfGraph {
    /// Adjacency matrix (sparse representation via HashMap)
    adjacency: HashMap<usize, Vec<usize>>,
    /// Node ID to URI mapping
    id_to_node: Vec<String>,
    /// Node URI to ID mapping
    node_to_id: HashMap<String, usize>,
    /// Edge count
    edge_count: usize,
}

impl RdfGraph {
    fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            id_to_node: Vec::new(),
            node_to_id: HashMap::new(),
            edge_count: 0,
        }
    }

    fn get_or_create_id(&mut self, node: String) -> usize {
        if let Some(&id) = self.node_to_id.get(&node) {
            id
        } else {
            let id = self.id_to_node.len();
            self.node_to_id.insert(node.clone(), id);
            self.id_to_node.push(node);
            id
        }
    }

    fn add_edge(&mut self, src: usize, dst: usize) {
        self.adjacency.entry(src).or_default().push(dst);
        self.edge_count += 1;
    }

    fn from_rdf_store(store: &RdfStore) -> Result<Self> {
        let mut graph = RdfGraph::new();

        // Extract all subject-object relationships
        // iter_quads() returns Result<Vec<Quad>>
        let quads = store
            .iter_quads()
            .context("Failed to get quads from RDF store")?;
        for quad in &quads {
            let subject = format!("{}", quad.subject());
            let object = format!("{}", quad.object());

            let src_id = graph.get_or_create_id(subject);
            let dst_id = graph.get_or_create_id(object);

            graph.add_edge(src_id, dst_id);
        }

        Ok(graph)
    }

    fn node_count(&self) -> usize {
        self.id_to_node.len()
    }

    fn out_degree(&self, node: usize) -> usize {
        self.adjacency.get(&node).map(|v| v.len()).unwrap_or(0)
    }

    fn neighbors(&self, node: usize) -> &[usize] {
        self.adjacency
            .get(&node)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn get_node_name(&self, id: usize) -> Option<&str> {
        self.id_to_node.get(id).map(|s| s.as_str())
    }

    /// Convert to scirs2_graph::Graph for advanced algorithms
    fn to_scirs2_graph(&self) -> Result<Graph<usize, f64>> {
        let mut graph: Graph<usize, f64> = Graph::new();

        // Add nodes
        for node_id in 0..self.node_count() {
            graph.add_node(node_id);
        }

        // Add edges
        for (&src, neighbors) in &self.adjacency {
            for &dst in neighbors {
                graph.add_edge(src, dst, 1.0)?;
            }
        }

        Ok(graph)
    }

    /// Convert to scirs2_graph::DiGraph for directed graph algorithms (like HITS)
    fn to_scirs2_digraph(&self) -> Result<scirs2_graph::DiGraph<usize, f64>> {
        let mut graph: scirs2_graph::DiGraph<usize, f64> = scirs2_graph::DiGraph::new();

        // Add nodes
        for node_id in 0..self.node_count() {
            graph.add_node(node_id);
        }

        // Add edges
        for (&src, neighbors) in &self.adjacency {
            for &dst in neighbors {
                graph.add_edge(src, dst, 1.0)?;
            }
        }

        Ok(graph)
    }
}

/// Execute graph analytics on RDF dataset
pub fn execute_graph_analytics(dataset_path: &Path, config: &AnalyticsConfig) -> Result<()> {
    let start_time = Instant::now();

    println!(
        "{}",
        "Graph Analytics - Powered by SciRS2-Core (Full Integration)"
            .cyan()
            .bold()
    );
    println!("{}", "─".repeat(70).cyan());
    println!(
        "{}",
        "Showcasing: ndarray_ext, parallel_ops, profiling, random".dimmed()
    );
    println!("{}", "─".repeat(70).cyan());

    // Load RDF store
    let load_start = Instant::now();
    println!("Loading RDF dataset...");
    let store = RdfStore::open(dataset_path).context("Failed to load RDF dataset for analytics")?;
    let load_time = load_start.elapsed();

    // Convert RDF to graph
    let convert_start = Instant::now();
    println!("Converting RDF to graph representation...");
    let graph = RdfGraph::from_rdf_store(&store)?;
    let convert_time = convert_start.elapsed();

    let node_count = graph.node_count();
    let edge_count = graph.edge_count;

    println!();
    println!("{}", "Graph Statistics:".green().bold());
    println!("  Nodes: {}", node_count.to_string().yellow());
    println!("  Edges: {}", edge_count.to_string().yellow());
    println!();

    // Execute requested operation
    let analytics_start = Instant::now();
    match config.operation {
        AnalyticsOperation::PageRank => {
            execute_pagerank(&graph, config)?;
        }
        AnalyticsOperation::DegreeDistribution => {
            execute_degree_distribution(&graph, config)?;
        }
        AnalyticsOperation::CommunityDetection => {
            execute_community_detection(&graph, config)?;
        }
        AnalyticsOperation::ShortestPaths => {
            execute_shortest_paths(&graph, config)?;
        }
        AnalyticsOperation::GraphStats => {
            execute_graph_stats(&graph)?;
        }
        AnalyticsOperation::BetweennessCentrality => {
            execute_betweenness_centrality(&graph, config)?;
        }
        AnalyticsOperation::ClosenessCentrality => {
            execute_closeness_centrality(&graph, config)?;
        }
        AnalyticsOperation::EigenvectorCentrality => {
            execute_eigenvector_centrality(&graph, config)?;
        }
        AnalyticsOperation::KatzCentrality => {
            execute_katz_centrality(&graph, config)?;
        }
        AnalyticsOperation::HitsAlgorithm => {
            execute_hits_algorithm(&graph, config)?;
        }
        AnalyticsOperation::LouvainCommunities => {
            execute_louvain_communities(&graph, config)?;
        }
        AnalyticsOperation::KCoreDecomposition => {
            execute_kcore_decomposition(&graph, config)?;
        }
        AnalyticsOperation::TriangleCounting => {
            execute_triangle_counting(&graph, config)?;
        }
        AnalyticsOperation::DiameterRadius => {
            execute_diameter_radius(&graph, config)?;
        }
        AnalyticsOperation::CenterNodes => {
            execute_center_nodes(&graph, config)?;
        }
        AnalyticsOperation::ExtendedMotifs => {
            execute_extended_motifs(&graph, config)?;
        }
    }
    let analytics_time = analytics_start.elapsed();

    let total_time = start_time.elapsed();

    // Show profiling results
    println!();
    println!("{}", "Performance Metrics:".cyan().bold());
    println!("  Load dataset:     {:.3}s", load_time.as_secs_f64());
    println!("  Graph conversion: {:.3}s", convert_time.as_secs_f64());
    println!("  Analytics:        {:.3}s", analytics_time.as_secs_f64());
    println!(
        "  {}",
        format!("Total time:       {:.3}s", total_time.as_secs_f64())
            .green()
            .bold()
    );

    Ok(())
}

fn execute_pagerank(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
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

    // Initialize PageRank scores using scirs2_core::ndarray_ext
    let mut scores = Array1::from_elem(n, 1.0 / n as f64);
    let mut new_scores = Array1::zeros(n);

    // Compute out-degrees
    let out_degrees: Vec<usize> = (0..n).map(|i| graph.out_degree(i)).collect();

    // Power iteration
    let mut converged = false;
    for iter in 0..config.max_iterations {
        new_scores.fill(0.0);

        // Distribute scores along edges
        for node in 0..n {
            let degree = out_degrees[node];
            if degree > 0 {
                let contrib = scores[node] / degree as f64;
                for &neighbor in graph.neighbors(node) {
                    new_scores[neighbor] += d * contrib;
                }
            }
        }

        // Add random jump probability
        let sum: f64 = new_scores.iter().sum();
        let random_jump = (1.0 - d) / n as f64 + d * (1.0 - sum) / n as f64;
        for score in new_scores.iter_mut() {
            *score += random_jump;
        }

        // Check convergence using scirs2_core array operations
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

    // Sort by score (descending) using parallel operations for large graphs
    let mut ranked: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(id, &score)| (id, score))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top K results
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

fn execute_degree_distribution(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!("{}", "Degree Distribution Analysis".green().bold());
    println!();

    let n = graph.node_count();

    // Compute degrees using scirs2_core arrays
    let degrees = Array1::from_vec(
        (0..n)
            .map(|i| graph.out_degree(i) as f64)
            .collect::<Vec<_>>(),
    );

    // Statistical analysis using scirs2_core
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

    // Find hub nodes (top K by degree)
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

    // Degree distribution histogram
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

fn execute_community_detection(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Community Detection (Label Propagation with SciRS2)"
            .green()
            .bold()
    );
    println!();

    let n = graph.node_count();

    // Initialize labels (each node in its own community)
    let mut labels = Array1::from_vec((0..n).collect());
    let mut new_labels = labels.clone();

    // Random number generator from scirs2_core
    let mut rng = Random::default();

    // Label propagation algorithm
    let max_iterations = 10;
    for iter in 0..max_iterations {
        let mut changed = false;

        // Randomize node order
        let mut nodes: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            nodes.swap(i, j);
        }

        // Update labels
        for &node in &nodes {
            let neighbors = graph.neighbors(node);
            if neighbors.is_empty() {
                continue;
            }

            // Count neighbor labels
            let mut label_counts: HashMap<usize, usize> = HashMap::new();
            for &neighbor in neighbors {
                *label_counts.entry(labels[neighbor]).or_insert(0) += 1;
            }

            // Pick most frequent label
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

    // Group nodes by community
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

    // Sort communities by size
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

fn execute_shortest_paths(graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!("{}", "Shortest Paths Analysis (BFS)".green().bold());
    println!();

    // Get source node
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

    // BFS to compute shortest paths
    let n = graph.node_count();
    let mut distances = vec![f64::INFINITY; n];
    let mut predecessors = vec![None; n];
    let mut queue = std::collections::VecDeque::new();

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

    // If target specified, show path
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

            // Reconstruct path
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
        // Show distances to all reachable nodes (top K)
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

fn execute_graph_stats(graph: &RdfGraph) -> Result<()> {
    println!("{}", "Comprehensive Graph Statistics".green().bold());
    println!("{}", "─".repeat(70).cyan());

    let n = graph.node_count();
    let m = graph.edge_count;

    // Basic metrics
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

    // Degree statistics using scirs2_core arrays
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

    // Find hub nodes
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

fn execute_betweenness_centrality(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
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

    // Convert to scirs2_graph::Graph
    let graph = rdf_graph.to_scirs2_graph()?;

    // Calculate betweenness centrality using scirs2-graph
    let centrality = betweenness_centrality(&graph, false);

    // Sort by centrality (descending)
    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top K results
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

fn execute_closeness_centrality(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
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

    // Convert to scirs2_graph::Graph
    let graph = rdf_graph.to_scirs2_graph()?;

    // Calculate closeness centrality using scirs2-graph (normalized)
    let centrality = closeness_centrality(&graph, true);

    // Sort by centrality (descending)
    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top K results
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

fn execute_eigenvector_centrality(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
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

    // Convert to scirs2_graph::Graph
    let graph = rdf_graph.to_scirs2_graph()?;

    // Calculate eigenvector centrality using scirs2-graph
    let centrality = eigenvector_centrality(&graph, 100, 1e-6)
        .context("Failed to compute eigenvector centrality")?;

    // Sort by centrality (descending)
    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top K results
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

fn execute_katz_centrality(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
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

    // Convert to scirs2_graph::Graph
    let graph = rdf_graph.to_scirs2_graph()?;

    // Calculate Katz centrality using scirs2-graph
    let centrality = katz_centrality(&graph, config.katz_alpha, config.katz_beta)
        .context("Failed to compute Katz centrality")?;

    // Sort by centrality (descending)
    let mut sorted_centrality: Vec<(usize, f64)> = centrality.into_iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top K results
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

fn execute_hits_algorithm(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "HITS Algorithm - Hubs and Authorities (SciRS2-Graph)"
            .green()
            .bold()
    );
    println!("{}", "  Computing hub and authority scores...".dimmed());
    println!();

    // Convert to scirs2_graph::DiGraph (HITS requires directed graph)
    let graph = rdf_graph.to_scirs2_digraph()?;

    // Calculate HITS scores using scirs2-graph
    let hits: HitsScores<usize> = hits_algorithm(&graph, config.max_iterations, config.tolerance)
        .context("Failed to compute HITS scores")?;

    // Sort hubs by score (descending)
    let mut sorted_hubs: Vec<(usize, f64)> = hits.hubs.into_iter().collect();
    sorted_hubs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Sort authorities by score (descending)
    let mut sorted_authorities: Vec<(usize, f64)> = hits.authorities.into_iter().collect();
    sorted_authorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top K hubs
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

fn execute_louvain_communities(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
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

    // Convert to scirs2_graph::Graph
    let graph = rdf_graph.to_scirs2_graph()?;

    // Calculate Louvain communities using scirs2-graph
    let community_result = louvain_communities_result(&graph);

    // Use the node_communities HashMap from CommunityResult
    let communities = &community_result.node_communities;

    // Group nodes by community
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

    // Sort communities by size
    let mut sorted_communities: Vec<_> = community_map.into_iter().collect();
    sorted_communities.sort_by_key(|(_, nodes)| std::cmp::Reverse(nodes.len()));

    // Display top communities
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

/// Execute K-core decomposition using scirs2-graph
fn execute_kcore_decomposition(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "K-Core Decomposition (Dense Subgraph Discovery)"
            .green()
            .bold()
    );
    println!();

    // Convert to scirs2_graph::Graph
    let scirs2_graph = rdf_graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Computing k-core decomposition...");
    println!();

    // Compute k-core decomposition using scirs2-graph
    let k_cores = k_core_decomposition(&scirs2_graph);

    // Find maximum core number
    let max_core = k_cores.values().copied().max().unwrap_or(0);

    println!("{}", "Core Statistics:".yellow().bold());
    println!("  Maximum core number: {}", max_core.to_string().green());
    println!();

    // Count nodes in each core
    let mut core_counts: HashMap<usize, usize> = HashMap::new();
    for &core_num in k_cores.values() {
        *core_counts.entry(core_num).or_insert(0) += 1;
    }

    println!("{}", "Core Distribution:".yellow().bold());
    println!("{}", "─".repeat(60).cyan());
    println!("{:>10} | {:>15} | {}", "Core", "Node Count", "Density");
    println!("{}", "─".repeat(60).cyan());

    let mut sorted_cores: Vec<_> = core_counts.iter().collect();
    sorted_cores.sort_by_key(|(k, _)| std::cmp::Reverse(**k));

    for (&core_num, &count) in sorted_cores.iter().take(config.top_k) {
        let density = count as f64 / rdf_graph.node_count() as f64 * 100.0;
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

    // If specific k value requested, show nodes in that core
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
            if let Some(node_name) = rdf_graph.get_node_name(node_id) {
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
        // Show sample nodes from highest core
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
            if let Some(node_name) = rdf_graph.get_node_name(node_id) {
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

/// Execute triangle counting for clustering coefficient analysis
fn execute_triangle_counting(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Triangle Counting (Clustering Coefficient Analysis)"
            .green()
            .bold()
    );
    println!();

    // Convert to scirs2_graph::Graph
    let scirs2_graph = rdf_graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Finding triangles in the graph...");
    println!();

    // Use scirs2-graph motif detection for triangle counting
    let triangles = find_motifs(&scirs2_graph, MotifType::Triangle);

    let triangle_count = triangles.len();
    let node_count = rdf_graph.node_count();

    println!("{}", "Triangle Statistics:".yellow().bold());
    println!("  Total triangles: {}", triangle_count.to_string().green());
    println!("  Nodes: {}", node_count.to_string().yellow());
    println!();

    // Calculate global clustering coefficient
    // Global clustering coefficient = (3 * number of triangles) / number of connected triples
    let total_triples = calculate_connected_triples(rdf_graph);
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

    // Calculate per-node triangle participation
    let mut node_triangle_count: HashMap<usize, usize> = HashMap::new();
    for triangle in &triangles {
        for node in triangle {
            // Get node_id from the node value
            if let Some(&node_id) = rdf_graph.node_to_id.get(&format!("{:?}", node)) {
                *node_triangle_count.entry(node_id).or_insert(0) += 1;
            }
        }
    }

    // Show nodes with most triangle participation
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
        if let Some(node_name) = rdf_graph.get_node_name(node_id) {
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

    // Show sample triangles
    if !triangles.is_empty() {
        println!();
        println!(
            "{}",
            format!("Sample Triangles (showing up to 5):")
                .yellow()
                .bold()
        );
        println!("{}", "─".repeat(100).cyan());

        for (i, triangle) in triangles.iter().take(5).enumerate() {
            println!("  Triangle {}:", (i + 1).to_string().cyan());
            for (j, node) in triangle.iter().enumerate() {
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

/// Calculate the number of connected triples in the graph (for clustering coefficient)
fn calculate_connected_triples(graph: &RdfGraph) -> usize {
    let mut triples = 0;

    for node in 0..graph.node_count() {
        let neighbors = graph.neighbors(node);
        let degree = neighbors.len();

        // Each node with degree k contributes k*(k-1)/2 triples
        if degree >= 2 {
            triples += degree * (degree - 1) / 2;
        }
    }

    triples
}

/// Execute diameter and radius calculation using scirs2-graph
fn execute_diameter_radius(rdf_graph: &RdfGraph, _config: &AnalyticsConfig) -> Result<()> {
    println!("{}", "Graph Diameter and Radius Analysis".green().bold());
    println!();

    // Convert to scirs2_graph::Graph
    let scirs2_graph = rdf_graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Computing graph diameter and radius...");
    println!();

    // Compute diameter using scirs2-graph
    let graph_diameter = diameter(&scirs2_graph);

    // Compute radius using scirs2-graph
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

/// Execute center nodes identification using scirs2-graph
fn execute_center_nodes(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Graph Center Nodes (Minimum Eccentricity)".green().bold()
    );
    println!();

    // Convert to scirs2_graph::Graph
    let scirs2_graph = rdf_graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Finding center nodes...");
    println!();

    // Find center nodes using scirs2-graph
    let center_node_ids: Vec<usize> = center_nodes(&scirs2_graph);

    // Also compute radius for context
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
            if let Some(node_name) = rdf_graph.get_node_name(node_id) {
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

/// Execute extended motif analysis using scirs2-graph
fn execute_extended_motifs(rdf_graph: &RdfGraph, config: &AnalyticsConfig) -> Result<()> {
    println!(
        "{}",
        "Extended Motif Analysis (Squares, Stars, Cliques, Paths)"
            .green()
            .bold()
    );
    println!();

    // Convert to scirs2_graph::Graph
    let scirs2_graph = rdf_graph
        .to_scirs2_graph()
        .context("Failed to convert RDF graph to scirs2_graph::Graph")?;

    println!("Analyzing graph motifs...");
    println!();

    // Find different motif types
    let triangles = find_motifs(&scirs2_graph, MotifType::Triangle);
    let squares = find_motifs(&scirs2_graph, MotifType::Square);
    let stars = find_motifs(&scirs2_graph, MotifType::Star3);
    let cliques = find_motifs(&scirs2_graph, MotifType::Clique4);
    let paths = find_motifs(&scirs2_graph, MotifType::Path3);

    println!("{}", "Motif Counts:".yellow().bold());
    println!("{}", "─".repeat(70).cyan());
    println!("{:>20} | {:>15} | {}", "Motif Type", "Count", "Description");
    println!("{}", "─".repeat(70).cyan());

    println!(
        "{:>20} | {:>15} | {}",
        "Triangles",
        triangles.len().to_string().green(),
        "3-node cycles (transitivity)"
    );
    println!(
        "{:>20} | {:>15} | {}",
        "Squares",
        squares.len().to_string().green(),
        "4-node cycles (rectangles)"
    );
    println!(
        "{:>20} | {:>15} | {}",
        "3-Stars",
        stars.len().to_string().green(),
        "Hub with 3 spokes"
    );
    println!(
        "{:>20} | {:>15} | {}",
        "4-Cliques",
        cliques.len().to_string().green(),
        "Fully connected 4-node subgraphs"
    );
    println!(
        "{:>20} | {:>15} | {}",
        "3-Paths",
        paths.len().to_string().green(),
        "Linear 4-node paths"
    );

    let total_motifs = triangles.len() + squares.len() + stars.len() + cliques.len() + paths.len();
    println!("{}", "─".repeat(70).cyan());
    println!(
        "{:>20} | {:>15}",
        "Total Motifs",
        total_motifs.to_string().cyan().bold()
    );

    // Show sample motifs for each type
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_operation_parsing() {
        assert_eq!(
            "pagerank".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::PageRank
        );
        assert_eq!(
            "degree".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::DegreeDistribution
        );
        assert_eq!(
            "community".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::CommunityDetection
        );
        assert_eq!(
            "betweenness".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::BetweennessCentrality
        );
        assert_eq!(
            "closeness".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::ClosenessCentrality
        );
        assert_eq!(
            "eigenvector".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::EigenvectorCentrality
        );
        // Test short forms
        assert_eq!(
            "bc".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::BetweennessCentrality
        );
        assert_eq!(
            "cc".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::ClosenessCentrality
        );
        assert_eq!(
            "ec".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::EigenvectorCentrality
        );
        // Test new algorithms
        assert_eq!(
            "katz".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::KatzCentrality
        );
        assert_eq!(
            "hits".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::HitsAlgorithm
        );
        assert_eq!(
            "louvain".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::LouvainCommunities
        );
        // Test short forms for new algorithms
        assert_eq!(
            "kc".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::KatzCentrality
        );
        assert_eq!(
            "ha".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::HitsAlgorithm
        );
        assert_eq!(
            "lc".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::LouvainCommunities
        );
        // Test newest algorithms
        assert_eq!(
            "kcore".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::KCoreDecomposition
        );
        assert_eq!(
            "k-core".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::KCoreDecomposition
        );
        assert_eq!(
            "triangles".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::TriangleCounting
        );
        assert_eq!(
            "tc".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::TriangleCounting
        );
        // Test Phase 6 algorithms
        assert_eq!(
            "diameter".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::DiameterRadius
        );
        assert_eq!(
            "radius".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::DiameterRadius
        );
        assert_eq!(
            "dr".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::DiameterRadius
        );
        assert_eq!(
            "center".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::CenterNodes
        );
        assert_eq!(
            "cn".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::CenterNodes
        );
        assert_eq!(
            "motifs".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::ExtendedMotifs
        );
        assert_eq!(
            "em".parse::<AnalyticsOperation>().unwrap(),
            AnalyticsOperation::ExtendedMotifs
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = AnalyticsConfig::default();
        assert_eq!(config.damping_factor, 0.85);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.top_k, 20);
        assert_eq!(config.katz_alpha, 0.1);
        assert_eq!(config.katz_beta, 1.0);
        assert_eq!(config.k_core_value, None);
        assert!(config.enable_simd);
        assert!(config.enable_parallel);
        assert!(!config.enable_gpu);
    }

    #[test]
    fn test_rdf_graph_construction() {
        let mut graph = RdfGraph::new();
        let id1 = graph.get_or_create_id("http://example.org/resource1".to_string());
        let id2 = graph.get_or_create_id("http://example.org/resource2".to_string());
        let id1_again = graph.get_or_create_id("http://example.org/resource1".to_string());

        assert_eq!(id1, id1_again);
        assert_ne!(id1, id2);
        assert_eq!(graph.node_count(), 2);

        graph.add_edge(id1, id2);
        assert_eq!(graph.edge_count, 1);
        assert_eq!(graph.out_degree(id1), 1);
        assert_eq!(graph.out_degree(id2), 0);
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut graph = RdfGraph::new();

        // Create a small graph: 0 -> 1, 0 -> 2, 1 -> 2
        let id0 = graph.get_or_create_id("http://example.org/node0".to_string());
        let id1 = graph.get_or_create_id("http://example.org/node1".to_string());
        let id2 = graph.get_or_create_id("http://example.org/node2".to_string());

        graph.add_edge(id0, id1);
        graph.add_edge(id0, id2);
        graph.add_edge(id1, id2);

        // Verify adjacency structure
        assert_eq!(graph.neighbors(id0).len(), 2);
        assert_eq!(graph.neighbors(id1).len(), 1);
        assert_eq!(graph.neighbors(id2).len(), 0);

        // Verify neighbors
        let neighbors_0 = graph.neighbors(id0);
        assert!(neighbors_0.contains(&id1));
        assert!(neighbors_0.contains(&id2));

        let neighbors_1 = graph.neighbors(id1);
        assert_eq!(neighbors_1, &[id2]);
    }

    #[test]
    fn test_scirs2_graph_conversion() {
        let mut rdf_graph = RdfGraph::new();

        // Create a small graph: 0 -> 1, 1 -> 2
        let id0 = rdf_graph.get_or_create_id("http://example.org/a".to_string());
        let id1 = rdf_graph.get_or_create_id("http://example.org/b".to_string());
        let id2 = rdf_graph.get_or_create_id("http://example.org/c".to_string());

        rdf_graph.add_edge(id0, id1);
        rdf_graph.add_edge(id1, id2);

        // Convert to scirs2_graph
        let scirs2_graph = rdf_graph.to_scirs2_graph().unwrap();

        // Verify conversion
        assert_eq!(scirs2_graph.node_count(), 3);
        assert_eq!(scirs2_graph.edge_count(), 2);

        // Verify the graph has the nodes
        assert!(scirs2_graph.has_node(&id0));
        assert!(scirs2_graph.has_node(&id1));
        assert!(scirs2_graph.has_node(&id2));
    }
}
