//! Execute graph analytics on RDF dataset
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{Context, Result};
use colored::Colorize;
use oxirs_core::rdf_store::RdfStore;
use std::path::Path;
use std::time::Instant;

use super::advanced::{execute_graph_coloring, execute_maximum_matching, execute_network_flow};
use super::centrality::{
    execute_betweenness_centrality, execute_closeness_centrality, execute_eigenvector_centrality,
    execute_hits_algorithm, execute_katz_centrality,
};
use super::community::{execute_community_detection, execute_louvain_communities};
use super::paths::execute_shortest_paths;
use super::patterns::execute_extended_motifs;
use super::ranking::execute_pagerank;
use super::stats_decomposition::{
    execute_center_nodes, execute_diameter_radius, execute_kcore_decomposition,
    execute_triangle_counting,
};
use super::stats_distributions::{execute_degree_distribution, execute_graph_stats};
use super::types::{AnalyticsConfig, AnalyticsOperation, BenchmarkResult, GraphMetrics, RdfGraph};

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

    let load_start = Instant::now();
    println!("Loading RDF dataset...");
    let store = RdfStore::open(dataset_path).context("Failed to load RDF dataset for analytics")?;
    let load_time = load_start.elapsed();

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
        AnalyticsOperation::GraphColoring => {
            execute_graph_coloring(&graph, config)?;
        }
        AnalyticsOperation::MaximumMatching => {
            execute_maximum_matching(&graph, config)?;
        }
        AnalyticsOperation::NetworkFlow => {
            execute_network_flow(&graph, config)?;
        }
    }

    let analytics_time = analytics_start.elapsed();
    let total_time = start_time.elapsed();

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

    if config.enable_benchmarking {
        let benchmark = BenchmarkResult::new(
            &format!("{:?}", config.operation),
            node_count,
            edge_count,
            total_time.as_secs_f64() * 1000.0,
            load_time.as_secs_f64() * 1000.0,
            convert_time.as_secs_f64() * 1000.0,
            analytics_time.as_secs_f64() * 1000.0,
        );
        benchmark.display();
    }

    if let Some(export_path) = &config.export_path {
        let density = if node_count > 1 {
            edge_count as f64 / (node_count as f64 * (node_count as f64 - 1.0))
        } else {
            0.0
        };
        let metrics = GraphMetrics::new(
            &format!("{:?}", config.operation),
            node_count,
            edge_count,
            density,
            analytics_time.as_secs_f64() * 1000.0,
        );
        if export_path.ends_with(".json") {
            metrics.export_json(export_path)?;
        } else if export_path.ends_with(".csv") {
            metrics.export_csv(export_path)?;
        } else {
            println!(
                "{}",
                "Warning: Unknown export format. Use .json or .csv extension.".yellow()
            );
        }
    }

    Ok(())
}
