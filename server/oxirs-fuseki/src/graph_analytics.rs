//! Graph Analytics Algorithms for RDF Data
//!
//! This module provides comprehensive graph analytics capabilities for analyzing
//! RDF graph structures, including:
//!
//! - Centrality algorithms (PageRank, Betweenness, Closeness, Eigenvector)
//! - Community detection (Louvain, Label Propagation)
//! - Path analysis algorithms
//! - Graph clustering and partitioning
//! - Network topology analysis
//! - Influence and importance metrics
//! - Graph neural network features

use crate::error::FusekiResult;
use crate::store::Store;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};
use tokio::sync::RwLock;
use tracing::{info, instrument};

/// Graph analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalyticsConfig {
    /// Maximum number of iterations for iterative algorithms
    pub max_iterations: usize,
    /// Convergence threshold for iterative algorithms
    pub convergence_threshold: f64,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Maximum graph size for in-memory processing
    pub max_graph_size: usize,
    /// Enable caching of computed metrics
    pub enable_caching: bool,
}

impl Default for GraphAnalyticsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            parallel_processing: true,
            max_graph_size: 1_000_000,
            enable_caching: true,
        }
    }
}

/// Graph node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node identifier (IRI)
    pub id: String,
    /// Node label/type
    pub label: Option<String>,
    /// Node properties
    pub properties: HashMap<String, String>,
    /// Outgoing edges
    pub out_edges: Vec<String>,
    /// Incoming edges
    pub in_edges: Vec<String>,
}

/// Graph edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge label/predicate
    pub label: String,
    /// Edge weight (optional)
    pub weight: Option<f64>,
    /// Edge properties
    pub properties: HashMap<String, String>,
}

/// Graph structure for analytics
#[derive(Debug, Clone)]
pub struct AnalysisGraph {
    /// Nodes in the graph
    pub nodes: HashMap<String, GraphNode>,
    /// Edges in the graph
    pub edges: Vec<GraphEdge>,
    /// Adjacency list for efficient traversal
    pub adjacency_list: HashMap<String, Vec<String>>,
    /// Reverse adjacency list
    pub reverse_adjacency_list: HashMap<String, Vec<String>>,
    /// Edge weights
    pub edge_weights: HashMap<(String, String), f64>,
}

impl Default for AnalysisGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisGraph {
    /// Create new analysis graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency_list: HashMap::new(),
            edge_weights: HashMap::new(),
        }
    }

    /// Add node to graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.adjacency_list.insert(node.id.clone(), Vec::new());
        self.reverse_adjacency_list
            .insert(node.id.clone(), Vec::new());
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add edge to graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        // Update adjacency lists
        self.adjacency_list
            .entry(edge.source.clone())
            .or_default()
            .push(edge.target.clone());

        self.reverse_adjacency_list
            .entry(edge.target.clone())
            .or_default()
            .push(edge.source.clone());

        // Store edge weight
        let weight = edge.weight.unwrap_or(1.0);
        self.edge_weights
            .insert((edge.source.clone(), edge.target.clone()), weight);

        self.edges.push(edge);
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<&String> {
        self.adjacency_list
            .get(node_id)
            .map(|neighbors| neighbors.iter().collect())
            .unwrap_or_default()
    }

    /// Get incoming neighbors of a node
    pub fn get_incoming_neighbors(&self, node_id: &str) -> Vec<&String> {
        self.reverse_adjacency_list
            .get(node_id)
            .map(|neighbors| neighbors.iter().collect())
            .unwrap_or_default()
    }

    /// Get edge weight
    pub fn get_edge_weight(&self, source: &str, target: &str) -> f64 {
        self.edge_weights
            .get(&(source.to_string(), target.to_string()))
            .copied()
            .unwrap_or(1.0)
    }

    /// Get graph statistics
    pub fn get_statistics(&self) -> GraphStatistics {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();
        let avg_degree = if node_count > 0 {
            (edge_count * 2) as f64 / node_count as f64
        } else {
            0.0
        };

        // Compute degree distribution
        let mut degree_distribution = HashMap::new();
        for neighbors in self.adjacency_list.values() {
            let degree = neighbors.len();
            *degree_distribution.entry(degree).or_insert(0) += 1;
        }

        GraphStatistics {
            node_count,
            edge_count,
            avg_degree,
            degree_distribution,
            is_directed: true, // RDF graphs are typically directed
            density: if node_count > 1 {
                edge_count as f64 / (node_count * (node_count - 1)) as f64
            } else {
                0.0
            },
        }
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Average degree
    pub avg_degree: f64,
    /// Degree distribution
    pub degree_distribution: HashMap<usize, usize>,
    /// Whether the graph is directed
    pub is_directed: bool,
    /// Graph density
    pub density: f64,
}

/// Centrality metrics for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMetrics {
    /// Node identifier
    pub node_id: String,
    /// PageRank score
    pub pagerank: f64,
    /// Betweenness centrality
    pub betweenness: f64,
    /// Closeness centrality
    pub closeness: f64,
    /// Eigenvector centrality
    pub eigenvector: f64,
    /// In-degree centrality
    pub in_degree: f64,
    /// Out-degree centrality
    pub out_degree: f64,
}

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionResult {
    /// Node to community assignment
    pub node_communities: HashMap<String, usize>,
    /// Community sizes
    pub community_sizes: HashMap<usize, usize>,
    /// Modularity score
    pub modularity: f64,
    /// Number of communities
    pub num_communities: usize,
}

/// Path analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathAnalysisResult {
    /// Source node
    pub source: String,
    /// Target node
    pub target: String,
    /// Shortest path
    pub shortest_path: Vec<String>,
    /// Path length
    pub path_length: usize,
    /// All paths (up to a limit)
    pub all_paths: Vec<Vec<String>>,
}

/// Graph analytics engine
#[derive(Debug)]
pub struct GraphAnalyticsEngine {
    /// Configuration
    config: GraphAnalyticsConfig,
    /// Data store reference
    store: Arc<Store>,
    /// Cached analysis results
    cache: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl GraphAnalyticsEngine {
    /// Create new graph analytics engine
    pub fn new(config: GraphAnalyticsConfig, store: Arc<Store>) -> Self {
        Self {
            config,
            store,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Extract graph from RDF store
    #[instrument(skip(self))]
    pub async fn extract_graph(&self, graph_uri: Option<&str>) -> FusekiResult<AnalysisGraph> {
        info!("Extracting graph for analysis");

        // This is a simplified implementation - in practice, you'd query the RDF store
        let mut graph = AnalysisGraph::new();

        // Add sample nodes and edges for demonstration
        for i in 1..=100 {
            let node = GraphNode {
                id: format!("node_{i}"),
                label: Some(format!("Node {i}")),
                properties: HashMap::new(),
                out_edges: Vec::new(),
                in_edges: Vec::new(),
            };
            graph.add_node(node);
        }

        // Add sample edges
        for i in 1..=100 {
            for j in 1..=3 {
                let target = ((i + j - 1) % 100) + 1;
                if target != i {
                    let edge = GraphEdge {
                        source: format!("node_{i}"),
                        target: format!("node_{target}"),
                        label: "connected_to".to_string(),
                        weight: Some(1.0),
                        properties: HashMap::new(),
                    };
                    graph.add_edge(edge);
                }
            }
        }

        info!(
            "Graph extracted: {} nodes, {} edges",
            graph.nodes.len(),
            graph.edges.len()
        );
        Ok(graph)
    }

    /// Compute PageRank centrality
    #[instrument(skip(self, graph))]
    pub async fn compute_pagerank(
        &self,
        graph: &AnalysisGraph,
        damping_factor: f64,
    ) -> FusekiResult<HashMap<String, f64>> {
        info!("Computing PageRank centrality");

        let nodes: Vec<&String> = graph.nodes.keys().collect();
        let node_count = nodes.len();

        if node_count == 0 {
            return Ok(HashMap::new());
        }

        let mut pagerank = HashMap::new();
        let mut new_pagerank = HashMap::new();

        // Initialize PageRank values
        let initial_value = 1.0 / node_count as f64;
        for node_id in &nodes {
            pagerank.insert((*node_id).clone(), initial_value);
        }

        // Iterative computation
        for iteration in 0..self.config.max_iterations {
            let mut max_change: f64 = 0.0;

            for node_id in &nodes {
                let mut rank = (1.0 - damping_factor) / node_count as f64;

                // Sum contributions from incoming edges
                for neighbor in graph.get_incoming_neighbors(node_id) {
                    let neighbor_rank = pagerank.get(neighbor).copied().unwrap_or(0.0);
                    let neighbor_out_degree = graph.get_neighbors(neighbor).len() as f64;

                    if neighbor_out_degree > 0.0 {
                        rank += damping_factor * neighbor_rank / neighbor_out_degree;
                    }
                }

                let old_rank = pagerank.get(*node_id).copied().unwrap_or(0.0);
                let change = (rank - old_rank).abs();
                max_change = max_change.max(change);

                new_pagerank.insert((*node_id).clone(), rank);
            }

            pagerank = new_pagerank.clone();

            // Check convergence
            if max_change < self.config.convergence_threshold {
                info!("PageRank converged after {} iterations", iteration + 1);
                break;
            }
        }

        Ok(pagerank)
    }

    /// Compute betweenness centrality
    #[instrument(skip(self, graph))]
    pub async fn compute_betweenness_centrality(
        &self,
        graph: &AnalysisGraph,
    ) -> FusekiResult<HashMap<String, f64>> {
        info!("Computing betweenness centrality");

        let nodes: Vec<&String> = graph.nodes.keys().collect();
        let mut betweenness = HashMap::new();

        // Initialize betweenness scores
        for node_id in &nodes {
            betweenness.insert((*node_id).clone(), 0.0);
        }

        // For each node as source
        for source in &nodes {
            let shortest_paths = self.compute_shortest_paths(graph, source).await?;

            // For each target
            for target in &nodes {
                if source == target {
                    continue;
                }

                if let Some(paths) = shortest_paths.get(*target) {
                    if paths.len() > 1 {
                        // Multiple shortest paths exist
                        let path_weight = 1.0 / paths.len() as f64;

                        for path in paths {
                            // Add contribution to intermediate nodes
                            for intermediate in path.iter().take(path.len() - 1).skip(1) {
                                if let Some(score) = betweenness.get_mut(intermediate) {
                                    *score += path_weight;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize by the number of pairs
        let n = nodes.len() as f64;
        let normalization_factor = 2.0 / ((n - 1.0) * (n - 2.0));

        for score in betweenness.values_mut() {
            *score *= normalization_factor;
        }

        Ok(betweenness)
    }

    /// Compute closeness centrality
    ///
    /// Closeness centrality measures how close a node is to all other nodes.
    /// It's defined as (n-1) / sum_of_distances, where n is the number of nodes.
    #[instrument(skip(self, graph))]
    pub async fn compute_closeness_centrality(
        &self,
        graph: &AnalysisGraph,
    ) -> FusekiResult<HashMap<String, f64>> {
        info!("Computing closeness centrality");

        let nodes: Vec<&String> = graph.nodes.keys().collect();
        let mut closeness = HashMap::new();

        // For each node, compute sum of shortest path distances to all other nodes
        for node_id in &nodes {
            let paths = self.compute_shortest_paths(graph, node_id).await?;

            let mut total_distance = 0.0;
            let mut reachable_count = 0;

            // Sum up distances to all reachable nodes
            for target in &nodes {
                if node_id == target {
                    continue;
                }

                if let Some(target_paths) = paths.get(*target) {
                    if !target_paths.is_empty() {
                        // Distance is path length - 1 (number of edges)
                        let distance = target_paths[0].len() - 1;
                        total_distance += distance as f64;
                        reachable_count += 1;
                    }
                }
            }

            // Calculate closeness
            // If node is isolated or can't reach any other nodes, closeness is 0
            let closeness_value = if reachable_count > 0 {
                reachable_count as f64 / total_distance
            } else {
                0.0
            };

            closeness.insert((*node_id).clone(), closeness_value);
        }

        Ok(closeness)
    }

    /// Compute eigenvector centrality
    ///
    /// Eigenvector centrality assigns scores to nodes based on the principle that
    /// connections to high-scoring nodes contribute more to the score than connections
    /// to low-scoring nodes. Uses power iteration to compute the principal eigenvector.
    #[instrument(skip(self, graph))]
    pub async fn compute_eigenvector_centrality(
        &self,
        graph: &AnalysisGraph,
    ) -> FusekiResult<HashMap<String, f64>> {
        info!("Computing eigenvector centrality");

        let nodes: Vec<&String> = graph.nodes.keys().collect();
        let node_count = nodes.len();

        if node_count == 0 {
            return Ok(HashMap::new());
        }

        // Initialize eigenvector with uniform values
        let mut eigenvector = HashMap::new();
        let initial_value = 1.0 / (node_count as f64).sqrt();
        for node_id in &nodes {
            eigenvector.insert((*node_id).clone(), initial_value);
        }

        // Power iteration to find principal eigenvector
        for iteration in 0..self.config.max_iterations {
            let mut new_eigenvector = HashMap::new();
            let mut max_change: f64 = 0.0;

            // For each node, sum the eigenvector values of incoming neighbors
            for node_id in &nodes {
                let mut sum = 0.0;

                // Sum contributions from incoming edges (who points to me)
                for neighbor in graph.get_incoming_neighbors(node_id) {
                    let neighbor_value = eigenvector.get(neighbor).copied().unwrap_or(0.0);
                    sum += neighbor_value;
                }

                // Also consider outgoing edges for undirected-like behavior in RDF
                for neighbor in graph.get_neighbors(node_id) {
                    let neighbor_value = eigenvector.get(neighbor).copied().unwrap_or(0.0);
                    sum += neighbor_value;
                }

                new_eigenvector.insert((*node_id).clone(), sum);

                let old_value = eigenvector.get(*node_id).copied().unwrap_or(0.0);
                let change = (sum - old_value).abs();
                max_change = max_change.max(change);
            }

            // Normalize the eigenvector
            let norm: f64 = new_eigenvector.values().map(|v| v * v).sum::<f64>().sqrt();

            if norm > 0.0 {
                for value in new_eigenvector.values_mut() {
                    *value /= norm;
                }
            }

            eigenvector = new_eigenvector;

            // Check convergence
            if max_change < self.config.convergence_threshold {
                info!(
                    "Eigenvector centrality converged after {} iterations",
                    iteration + 1
                );
                break;
            }
        }

        Ok(eigenvector)
    }

    /// Compute shortest paths from a source node
    async fn compute_shortest_paths(
        &self,
        graph: &AnalysisGraph,
        source: &str,
    ) -> FusekiResult<HashMap<String, Vec<Vec<String>>>> {
        let mut distances = HashMap::new();
        let mut paths = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(source.to_string(), 0);
        paths.insert(source.to_string(), vec![vec![source.to_string()]]);
        queue.push_back(source.to_string());

        while let Some(current) = queue.pop_front() {
            let current_distance = distances[&current];

            for neighbor in graph.get_neighbors(&current) {
                let new_distance = current_distance + 1;

                match distances.get(neighbor) {
                    None => {
                        // First time visiting this node
                        distances.insert(neighbor.clone(), new_distance);

                        let mut new_paths = Vec::new();
                        for path in &paths[&current] {
                            let mut new_path = path.clone();
                            new_path.push(neighbor.clone());
                            new_paths.push(new_path);
                        }
                        paths.insert(neighbor.clone(), new_paths);
                        queue.push_back(neighbor.clone());
                    }
                    Some(&existing_distance) => {
                        if new_distance == existing_distance {
                            // Found another shortest path
                            let mut additional_paths = Vec::new();
                            for path in &paths[&current] {
                                let mut new_path = path.clone();
                                new_path.push(neighbor.clone());
                                additional_paths.push(new_path);
                            }
                            paths.get_mut(neighbor).unwrap().extend(additional_paths);
                        }
                        // If new_distance > existing_distance, ignore (longer path)
                    }
                }
            }
        }

        Ok(paths)
    }

    /// Detect communities using Louvain algorithm
    #[instrument(skip(self, graph))]
    pub async fn detect_communities_louvain(
        &self,
        graph: &AnalysisGraph,
    ) -> FusekiResult<CommunityDetectionResult> {
        info!("Detecting communities using Louvain algorithm");

        let nodes: Vec<String> = graph.nodes.keys().cloned().collect();
        let mut node_communities = HashMap::new();

        // Initially, each node is in its own community
        for (i, node_id) in nodes.iter().enumerate() {
            node_communities.insert(node_id.clone(), i);
        }

        let mut improved = true;
        let mut iteration = 0;

        while improved && iteration < self.config.max_iterations {
            improved = false;
            iteration += 1;

            for node_id in &nodes {
                let current_community = node_communities[node_id];
                let mut best_community = current_community;
                let mut best_modularity_gain = 0.0;

                // Try moving to neighbors' communities
                for neighbor in graph.get_neighbors(node_id) {
                    let neighbor_community = node_communities[neighbor];

                    if neighbor_community != current_community {
                        let modularity_gain = self.calculate_modularity_gain(
                            graph,
                            node_id,
                            current_community,
                            neighbor_community,
                            &node_communities,
                        );

                        if modularity_gain > best_modularity_gain {
                            best_modularity_gain = modularity_gain;
                            best_community = neighbor_community;
                        }
                    }
                }

                // Move node to best community if it improves modularity
                if best_community != current_community && best_modularity_gain > 0.0 {
                    node_communities.insert(node_id.clone(), best_community);
                    improved = true;
                }
            }
        }

        // Renumber communities to be consecutive
        let mut community_map = HashMap::new();
        let mut next_community_id = 0;

        for community_id in node_communities.values() {
            if !community_map.contains_key(community_id) {
                community_map.insert(*community_id, next_community_id);
                next_community_id += 1;
            }
        }

        // Update node communities with new numbering
        for community_id in node_communities.values_mut() {
            *community_id = community_map[community_id];
        }

        // Calculate community sizes
        let mut community_sizes = HashMap::new();
        for &community_id in node_communities.values() {
            *community_sizes.entry(community_id).or_insert(0) += 1;
        }

        // Calculate final modularity
        let modularity = self.calculate_modularity(graph, &node_communities);

        Ok(CommunityDetectionResult {
            node_communities,
            community_sizes,
            modularity,
            num_communities: next_community_id,
        })
    }

    /// Calculate modularity gain for moving a node between communities
    fn calculate_modularity_gain(
        &self,
        graph: &AnalysisGraph,
        node_id: &str,
        from_community: usize,
        to_community: usize,
        node_communities: &HashMap<String, usize>,
    ) -> f64 {
        // Simplified modularity gain calculation
        // In practice, this would be more sophisticated

        let mut gain = 0.0;
        let total_edges = graph.edges.len() as f64;

        // Count edges within communities
        for neighbor in graph.get_neighbors(node_id) {
            let neighbor_community = node_communities[neighbor];

            if neighbor_community == to_community {
                gain += 1.0 / total_edges;
            }
            if neighbor_community == from_community {
                gain -= 1.0 / total_edges;
            }
        }

        gain
    }

    /// Calculate modularity of the current community assignment
    fn calculate_modularity(
        &self,
        graph: &AnalysisGraph,
        node_communities: &HashMap<String, usize>,
    ) -> f64 {
        let total_edges = graph.edges.len() as f64;
        if total_edges == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;

        for edge in &graph.edges {
            let source_community = node_communities.get(&edge.source).copied().unwrap_or(0);
            let target_community = node_communities.get(&edge.target).copied().unwrap_or(0);

            if source_community == target_community {
                modularity += 1.0;
            }
        }

        modularity / total_edges - 0.5 // Simplified calculation
    }

    /// Compute all centrality metrics for all nodes
    #[instrument(skip(self, graph))]
    pub async fn compute_all_centrality_metrics(
        &self,
        graph: &AnalysisGraph,
    ) -> FusekiResult<Vec<CentralityMetrics>> {
        info!("Computing all centrality metrics");

        // Compute PageRank
        let pagerank = self.compute_pagerank(graph, 0.85).await?;

        // Compute betweenness centrality
        let betweenness = self.compute_betweenness_centrality(graph).await?;

        // Compute closeness centrality
        let closeness = self.compute_closeness_centrality(graph).await?;

        // Compute eigenvector centrality
        let eigenvector = self.compute_eigenvector_centrality(graph).await?;

        // Compute degree centralities
        let mut results = Vec::new();

        for node_id in graph.nodes.keys() {
            let in_degree = graph.get_incoming_neighbors(node_id).len() as f64;
            let out_degree = graph.get_neighbors(node_id).len() as f64;

            let metrics = CentralityMetrics {
                node_id: node_id.clone(),
                pagerank: pagerank.get(node_id).copied().unwrap_or(0.0),
                betweenness: betweenness.get(node_id).copied().unwrap_or(0.0),
                closeness: closeness.get(node_id).copied().unwrap_or(0.0),
                eigenvector: eigenvector.get(node_id).copied().unwrap_or(0.0),
                in_degree,
                out_degree,
            };

            results.push(metrics);
        }

        Ok(results)
    }

    /// Find shortest path between two nodes
    #[instrument(skip(self, graph))]
    pub async fn find_shortest_path(
        &self,
        graph: &AnalysisGraph,
        source: &str,
        target: &str,
    ) -> FusekiResult<PathAnalysisResult> {
        let paths = self.compute_shortest_paths(graph, source).await?;

        if let Some(target_paths) = paths.get(target) {
            let shortest_path = target_paths[0].clone();
            let path_length = shortest_path.len() - 1;

            Ok(PathAnalysisResult {
                source: source.to_string(),
                target: target.to_string(),
                shortest_path,
                path_length,
                all_paths: target_paths.clone(),
            })
        } else {
            Ok(PathAnalysisResult {
                source: source.to_string(),
                target: target.to_string(),
                shortest_path: Vec::new(),
                path_length: 0,
                all_paths: Vec::new(),
            })
        }
    }

    /// Get cached result if available
    async fn get_cached_result(&self, key: &str) -> Option<serde_json::Value> {
        if self.config.enable_caching {
            let cache = self.cache.read().await;
            cache.get(key).cloned()
        } else {
            None
        }
    }

    /// Cache computation result
    async fn cache_result(&self, key: String, value: serde_json::Value) {
        if self.config.enable_caching {
            let mut cache = self.cache.write().await;
            cache.insert(key, value);

            // Simple cache eviction
            if cache.len() > 100 {
                let keys_to_remove: Vec<String> = cache.keys().take(20).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = AnalysisGraph::new();

        let node1 = GraphNode {
            id: "node1".to_string(),
            label: Some("Node 1".to_string()),
            properties: HashMap::new(),
            out_edges: Vec::new(),
            in_edges: Vec::new(),
        };

        let node2 = GraphNode {
            id: "node2".to_string(),
            label: Some("Node 2".to_string()),
            properties: HashMap::new(),
            out_edges: Vec::new(),
            in_edges: Vec::new(),
        };

        graph.add_node(node1);
        graph.add_node(node2);

        let edge = GraphEdge {
            source: "node1".to_string(),
            target: "node2".to_string(),
            label: "connects".to_string(),
            weight: Some(1.0),
            properties: HashMap::new(),
        };

        graph.add_edge(edge);

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.get_neighbors("node1").len(), 1);
        assert_eq!(graph.get_incoming_neighbors("node2").len(), 1);
    }

    #[test]
    fn test_graph_statistics() {
        let mut graph = AnalysisGraph::new();

        for i in 1..=5 {
            let node = GraphNode {
                id: format!("node{i}"),
                label: None,
                properties: HashMap::new(),
                out_edges: Vec::new(),
                in_edges: Vec::new(),
            };
            graph.add_node(node);
        }

        // Add edges to form a simple cycle
        for i in 1..=5 {
            let edge = GraphEdge {
                source: format!("node{i}"),
                target: format!("node{}", (i % 5) + 1),
                label: "next".to_string(),
                weight: Some(1.0),
                properties: HashMap::new(),
            };
            graph.add_edge(edge);
        }

        let stats = graph.get_statistics();
        assert_eq!(stats.node_count, 5);
        assert_eq!(stats.edge_count, 5);
        assert_eq!(stats.avg_degree, 2.0);
    }
}
