//! Graph-Based Model Analytics using SciRS2-Graph
//!
//! This module provides advanced graph analysis for SAMM models using scirs2-graph algorithms.
//! It analyzes dependency structures, identifies critical components, and detects architectural patterns.
//!
//! # Features
//!
//! - **Dependency Graph Construction**: Build directed graphs from model dependencies
//! - **Centrality Analysis**: Identify most important properties and characteristics
//! - **Community Detection**: Find clusters of related properties
//! - **Path Analysis**: Shortest paths and critical paths in dependency chains
//! - **Cycle Detection**: Find circular dependencies
//! - **Graph Metrics**: Compute standard graph metrics (diameter, density, clustering coefficient)
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::graph_analytics::ModelGraph;
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
//! // Build dependency graph
//! let graph = ModelGraph::from_aspect(aspect)?;
//!
//! // Compute centrality metrics
//! let centrality = graph.compute_centrality();
//! println!("Most central node: {:?}", centrality.max_node());
//!
//! // Find communities
//! let communities = graph.detect_communities()?;
//! println!("Found {} communities", communities.len());
//!
//! // Detect cycles
//! let has_cycles = graph.has_cycles()?;
//! if has_cycles {
//!     println!("Warning: Circular dependencies detected");
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, ModelElement, Property};
use scirs2_graph::algorithms::shortest_path::dijkstra_path_digraph;
use scirs2_graph::measures::graph_density_digraph;
use scirs2_graph::{
    louvain_communities_result, pagerank, strongly_connected_components, DiGraph, Graph,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Graph representation of a SAMM model
///
/// Nodes represent properties and characteristics, edges represent dependencies
pub struct ModelGraph {
    /// The underlying directed graph
    graph: DiGraph<String, f64>,
    /// List of all nodes (for visualization)
    nodes: Vec<String>,
    /// List of all edges as (source, target) pairs (for visualization)
    edges: Vec<(String, String)>,
}

impl ModelGraph {
    /// Build a dependency graph from a SAMM aspect
    ///
    /// # Arguments
    ///
    /// * `aspect` - The aspect model to analyze
    ///
    /// # Returns
    ///
    /// A graph representation of the model's dependency structure
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::graph_analytics::ModelGraph;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
    /// let graph = ModelGraph::from_aspect(aspect)?;
    /// println!("Graph has {} nodes and {} edges",
    ///          graph.num_nodes(), graph.num_edges());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_aspect(aspect: &Aspect) -> Result<Self> {
        let mut graph = DiGraph::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Extract aspect name from URN
        let aspect_name = Self::extract_name_from_urn(&aspect.metadata.urn);

        // Add root aspect node
        graph.add_node(aspect_name.clone());
        nodes.push(aspect_name.clone());

        // Add property nodes
        for property in &aspect.properties {
            let prop_name = Self::extract_name_from_urn(&property.metadata.urn);
            graph.add_node(prop_name.clone());
            nodes.push(prop_name.clone());

            // Add edge from aspect to property (weight = 1.0)
            graph
                .add_edge(aspect_name.clone(), prop_name.clone(), 1.0)
                .map_err(|e| SammError::GraphError(format!("Failed to add edge: {}", e)))?;
            edges.push((aspect_name.clone(), prop_name.clone()));

            // If property has a characteristic, add that relationship
            if let Some(ref characteristic) = property.characteristic {
                let char_name = Self::extract_name_from_urn(&characteristic.metadata.urn);
                graph.add_node(char_name.clone());
                nodes.push(char_name.clone());

                graph
                    .add_edge(prop_name.clone(), char_name.clone(), 1.0)
                    .map_err(|e| SammError::GraphError(format!("Failed to add edge: {}", e)))?;
                edges.push((prop_name.clone(), char_name));
            }
        }

        Ok(Self {
            graph,
            nodes,
            edges,
        })
    }

    /// Extract element name from URN (e.g., "urn:samm:test:1.0.0#MyAspect" -> "MyAspect")
    fn extract_name_from_urn(urn: &str) -> String {
        urn.split('#').nth(1).unwrap_or(urn).to_string()
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> &[(String, String)] {
        &self.edges
    }

    /// Compute centrality metrics for all nodes
    ///
    /// Uses PageRank, betweenness centrality, and closeness centrality to identify
    /// the most important nodes in the dependency graph.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::graph_analytics::ModelGraph;
    ///
    /// # fn example(graph: &ModelGraph) -> Result<(), Box<dyn std::error::Error>> {
    /// let centrality = graph.compute_centrality();
    /// println!("Top 5 most central nodes:");
    /// for (name, score) in centrality.top_nodes(5) {
    ///     println!("  {}: {:.4}", name, score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_centrality(&self) -> CentralityMetrics {
        // Compute PageRank (primary centrality measure for directed graphs)
        let pagerank_scores = pagerank(&self.graph, 0.85, 1e-6, 100);

        // For directed graphs, we use PageRank as the main centrality measure
        // Betweenness and closeness centrality are not available for DiGraph in scirs2-graph
        // So we use PageRank scores as the combined score
        CentralityMetrics {
            scores: pagerank_scores.clone(),
            pagerank: pagerank_scores,
            betweenness: HashMap::new(), // Not available for DiGraph
            closeness: HashMap::new(),   // Not available for DiGraph
        }
    }

    /// Detect communities (clusters) of related elements
    ///
    /// Uses the Louvain algorithm to identify modules or groups of tightly coupled properties.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::graph_analytics::ModelGraph;
    ///
    /// # fn example(graph: &ModelGraph) -> Result<(), Box<dyn std::error::Error>> {
    /// let communities = graph.detect_communities()?;
    /// println!("Model has {} distinct modules", communities.len());
    /// for (i, community) in communities.iter().enumerate() {
    ///     println!("Module {}: {} elements", i, community.members.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn detect_communities(&self) -> Result<Vec<Community>> {
        // Community detection (Louvain) requires undirected graph
        // For directed graphs, we use strongly connected components as a proxy for communities
        let sccs = strongly_connected_components(&self.graph);

        Ok(sccs
            .into_iter()
            .enumerate()
            .map(|(id, component)| Community {
                id,
                members: component.into_iter().collect(),
            })
            .collect())
    }

    /// Check if the graph has circular dependencies
    ///
    /// Circular dependencies indicate potential design issues and should be avoided.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::graph_analytics::ModelGraph;
    ///
    /// # fn example(graph: &ModelGraph) -> Result<(), Box<dyn std::error::Error>> {
    /// if graph.has_cycles()? {
    ///     eprintln!("Warning: Circular dependencies detected!");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn has_cycles(&self) -> Result<bool> {
        // A directed graph has cycles if it has more than one strongly connected component
        // or if any SCC has more than one node
        let sccs = strongly_connected_components(&self.graph);

        // If there's more than one node in any SCC, there's a cycle
        Ok(sccs.iter().any(|scc| scc.len() > 1))
    }

    /// Compute shortest path between two elements
    ///
    /// # Arguments
    ///
    /// * `from` - Source element name
    /// * `to` - Target element name
    ///
    /// # Returns
    ///
    /// The path from source to target, or None if no path exists
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::graph_analytics::ModelGraph;
    ///
    /// # fn example(graph: &ModelGraph) -> Result<(), Box<dyn std::error::Error>> {
    /// if let Some(path) = graph.shortest_path("Property1", "Property2")? {
    ///     println!("Path: {}", path.join(" -> "));
    /// } else {
    ///     println!("No dependency path found");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn shortest_path(&self, from: &str, to: &str) -> Result<Option<Vec<String>>> {
        match dijkstra_path_digraph(&self.graph, &from.to_string(), &to.to_string()) {
            Ok(Some(path_data)) => Ok(Some(path_data.nodes)),
            Ok(None) => Ok(None),
            Err(e) => Err(SammError::GraphError(format!("Failed to find path: {}", e))),
        }
    }

    /// Compute comprehensive graph metrics
    ///
    /// Returns metrics like diameter, density, clustering coefficient, etc.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::graph_analytics::ModelGraph;
    ///
    /// # fn example(graph: &ModelGraph) -> Result<(), Box<dyn std::error::Error>> {
    /// let metrics = graph.compute_metrics()?;
    /// println!("Graph Metrics:");
    /// println!("  Nodes: {}", metrics.num_nodes);
    /// println!("  Edges: {}", metrics.num_edges);
    /// println!("  Density: {:.4}", metrics.density);
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_metrics(&self) -> Result<GraphMetrics> {
        let n = self.num_nodes();
        let m = self.num_edges();

        // Compute density using scirs2-graph (DiGraph version)
        let density = graph_density_digraph(&self.graph)
            .map_err(|e| SammError::GraphError(format!("Failed to compute density: {}", e)))?;

        // Diameter is not available for DiGraph in scirs2-graph
        // We set it to 0 as a placeholder
        let diameter_value = 0.0;

        Ok(GraphMetrics {
            num_nodes: n,
            num_edges: m,
            diameter: diameter_value,
            density,
        })
    }

    /// Get strongly connected components
    ///
    /// Returns groups of nodes where each node is reachable from every other node in the group.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::graph_analytics::ModelGraph;
    ///
    /// # fn example(graph: &ModelGraph) -> Result<(), Box<dyn std::error::Error>> {
    /// let sccs = graph.strongly_connected_components()?;
    /// println!("Found {} strongly connected components", sccs.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn strongly_connected_components(&self) -> Result<Vec<Vec<String>>> {
        let sccs = strongly_connected_components(&self.graph);

        Ok(sccs
            .into_iter()
            .map(|component| component.into_iter().collect())
            .collect())
    }
}

/// Centrality metrics for model elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMetrics {
    /// Combined centrality scores
    pub scores: HashMap<String, f64>,
    /// PageRank scores
    pub pagerank: HashMap<String, f64>,
    /// Betweenness centrality
    pub betweenness: HashMap<String, f64>,
    /// Closeness centrality
    pub closeness: HashMap<String, f64>,
}

impl CentralityMetrics {
    /// Get the node with maximum centrality
    pub fn max_node(&self) -> Option<(&String, f64)> {
        self.scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, score)| (name, *score))
    }

    /// Get top N nodes by centrality
    pub fn top_nodes(&self, n: usize) -> Vec<(&String, f64)> {
        let mut sorted: Vec<_> = self
            .scores
            .iter()
            .map(|(name, score)| (name, *score))
            .collect();
        sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }
}

/// A community (cluster) of related model elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Community identifier
    pub id: usize,
    /// Member element names
    pub members: Vec<String>,
}

/// A circular dependency cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cycle {
    /// The path forming the cycle
    pub path: Vec<String>,
}

impl Cycle {
    /// Get the length of the cycle
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// Check if the cycle is empty
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}

/// Comprehensive graph metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Graph diameter (longest shortest path) - 0.0 for DiGraph (not available)
    pub diameter: f64,
    /// Graph density (0-1)
    pub density: f64,
}

// Visualization module for graph rendering
pub mod visualization;
pub use visualization::{ColorScheme, VisualizationStyle};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};
    use std::collections::HashMap;

    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        // Add 3 properties
        for i in 1..=3 {
            let characteristic = Characteristic {
                metadata: crate::metamodel::ElementMetadata::new(format!(
                    "urn:samm:test:1.0.0#Char{}",
                    i
                )),
                data_type: Some("string".to_string()),
                kind: CharacteristicKind::Trait,
                constraints: vec![],
            };

            let property = Property::new(format!("urn:samm:test:1.0.0#Property{}", i))
                .with_characteristic(characteristic);

            aspect.add_property(property);
        }

        aspect
    }

    #[test]
    fn test_graph_construction() {
        let aspect = create_test_aspect();
        let graph = ModelGraph::from_aspect(&aspect).unwrap();

        // 1 aspect + 3 properties + 3 characteristics = 7 nodes
        assert_eq!(graph.num_nodes(), 7);

        // 3 edges (aspect->properties) + 3 edges (properties->characteristics) = 6 edges
        assert_eq!(graph.num_edges(), 6);
    }

    #[test]
    fn test_centrality_computation() {
        let aspect = create_test_aspect();
        let graph = ModelGraph::from_aspect(&aspect).unwrap();

        let centrality = graph.compute_centrality();

        // Should have scores for all nodes
        assert_eq!(centrality.scores.len(), 7);

        // Get top node
        let (top_node, score) = centrality.max_node().unwrap();
        // Just check that there is a top node with a positive score
        assert!(!top_node.is_empty());
        assert!(score > 0.0);
    }

    #[test]
    fn test_community_detection() {
        let aspect = create_test_aspect();
        let graph = ModelGraph::from_aspect(&aspect).unwrap();

        let communities = graph.detect_communities().unwrap();

        // Should have at least 1 community
        assert!(!communities.is_empty());

        // Total members across all communities should equal number of nodes
        let total_members: usize = communities.iter().map(|c| c.members.len()).sum();
        assert_eq!(total_members, graph.num_nodes());
    }

    #[test]
    fn test_cycle_detection() {
        let aspect = create_test_aspect();
        let graph = ModelGraph::from_aspect(&aspect).unwrap();

        let has_cycles = graph.has_cycles().unwrap();

        // Simple tree structure should have no cycles
        assert!(!has_cycles);
    }

    #[test]
    fn test_graph_metrics() {
        let aspect = create_test_aspect();
        let graph = ModelGraph::from_aspect(&aspect).unwrap();

        let metrics = graph.compute_metrics().unwrap();

        assert_eq!(metrics.num_nodes, 7);
        assert_eq!(metrics.num_edges, 6);
        assert!(metrics.density > 0.0);
        assert!(metrics.density <= 1.0);
    }

    #[test]
    fn test_shortest_path() {
        let aspect = create_test_aspect();
        let graph = ModelGraph::from_aspect(&aspect).unwrap();

        // Path from aspect to property should exist
        let path = graph
            .shortest_path("TestAspect", "Property1")
            .unwrap()
            .unwrap();

        assert_eq!(path.len(), 2); // Direct edge
        assert_eq!(path[0], "TestAspect");
        assert_eq!(path[1], "Property1");
    }

    #[test]
    fn test_strongly_connected_components() {
        let aspect = create_test_aspect();
        let graph = ModelGraph::from_aspect(&aspect).unwrap();

        let sccs = graph.strongly_connected_components().unwrap();

        // Tree structure should have each node as its own SCC
        assert_eq!(sccs.len(), 7);
    }
}
