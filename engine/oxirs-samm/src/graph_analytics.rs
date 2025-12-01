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
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, connected_components, diameter,
    dijkstra_path, eigenvector_centrality, graph_density, louvain_communities_result,
    pagerank, strongly_connected_components, DiGraph, Graph,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Graph representation of a SAMM model
///
/// Nodes represent properties and characteristics, edges represent dependencies
#[derive(Debug, Clone)]
pub struct ModelGraph {
    /// The underlying directed graph
    graph: DiGraph<String, f64>,
    /// Mapping from node indices to element names
    node_map: HashMap<usize, String>,
    /// Reverse mapping from element names to node indices
    name_to_id: HashMap<String, usize>,
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
        let mut node_map = HashMap::new();
        let mut name_to_id = HashMap::new();

        // Extract aspect name from URN
        let aspect_name = Self::extract_name_from_urn(&aspect.urn);

        // Add root aspect node
        let aspect_id = graph.add_node(aspect_name.clone());
        node_map.insert(aspect_id, aspect_name.clone());
        name_to_id.insert(aspect_name, aspect_id);

        // Add property nodes
        for property in &aspect.properties {
            let prop_name = Self::extract_name_from_urn(&property.urn);
            let prop_id = graph.add_node(prop_name.clone());
            node_map.insert(prop_id, prop_name.clone());
            name_to_id.insert(prop_name, prop_id);

            // Add edge from aspect to property (weight = 1.0)
            graph
                .add_edge(aspect_id, prop_id, 1.0)
                .map_err(|e| SammError::GraphError(format!("Failed to add edge: {}", e)))?;

            // If property has a characteristic, add that relationship
            if let Some(ref characteristic) = property.characteristic {
                let char_name = Self::extract_name_from_urn(&characteristic.urn);
                let char_id = graph.add_node(char_name.clone());
                node_map.insert(char_id, char_name.clone());
                name_to_id.insert(char_name, char_id);
                graph
                    .add_edge(prop_id, char_id, 1.0)
                    .map_err(|e| SammError::GraphError(format!("Failed to add edge: {}", e)))?;
            }
        }

        Ok(Self {
            graph,
            node_map,
            name_to_id,
        })
    }

    /// Extract element name from URN (e.g., "urn:samm:test:1.0.0#MyAspect" -> "MyAspect")
    fn extract_name_from_urn(urn: &str) -> String {
        urn.split('#')
            .nth(1)
            .unwrap_or(urn)
            .to_string()
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
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
        // Compute different centrality measures
        let pagerank_scores = pagerank(&self.graph, 0.85, 100, 1e-6);
        let betweenness = betweenness_centrality(&self.graph);
        let closeness = closeness_centrality(&self.graph);

        // Map to names
        let pagerank_named = self.map_scores_to_names(&pagerank_scores);
        let betweenness_named = self.map_scores_to_names(&betweenness);
        let closeness_named = self.map_scores_to_names(&closeness);

        // Combine metrics (weighted average)
        let mut combined_scores = HashMap::new();
        for (id, name) in &self.node_map {
            let pr_score = pagerank_scores.get(id).copied().unwrap_or(0.0);
            let bc_score = betweenness.get(id).copied().unwrap_or(0.0);
            let cc_score = closeness.get(id).copied().unwrap_or(0.0);

            // Weighted combination: 50% PageRank, 30% Betweenness, 20% Closeness
            let combined = 0.5 * pr_score + 0.3 * bc_score + 0.2 * cc_score;
            combined_scores.insert(name.clone(), combined);
        }

        CentralityMetrics {
            scores: combined_scores,
            pagerank: pagerank_named,
            betweenness: betweenness_named,
            closeness: closeness_named,
        }
    }

    /// Map node ID scores to element names
    fn map_scores_to_names(&self, scores: &HashMap<usize, f64>) -> HashMap<String, f64> {
        scores
            .iter()
            .filter_map(|(id, score)| {
                self.node_map.get(id).map(|name| (name.clone(), *score))
            })
            .collect()
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
        let community_result = louvain_communities_result(&self.graph, 1.0);

        // Convert CommunityStructure to our Community type
        let mut communities_map: HashMap<usize, Vec<String>> = HashMap::new();

        for (node_idx, community_id) in community_result.communities.iter().enumerate() {
            if let Some(name) = self.node_map.get(&node_idx) {
                communities_map
                    .entry(*community_id)
                    .or_insert_with(Vec::new)
                    .push(name.clone());
            }
        }

        Ok(communities_map
            .into_iter()
            .map(|(id, members)| Community { id, members })
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
        let from_id = self.name_to_id.get(from).ok_or_else(|| {
            SammError::ValidationError(format!("Element '{}' not found in graph", from))
        })?;

        let to_id = self.name_to_id.get(to).ok_or_else(|| {
            SammError::ValidationError(format!("Element '{}' not found in graph", to))
        })?;

        match dijkstra_path(&self.graph, *from_id, *to_id) {
            Ok(Some(path_data)) => {
                let path = path_data.nodes
                    .iter()
                    .filter_map(|id| self.node_map.get(id).cloned())
                    .collect();
                Ok(Some(path))
            }
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

        // Compute density using scirs2-graph
        let density = graph_density(&self.graph)
            .map_err(|e| SammError::GraphError(format!("Failed to compute density: {}", e)))?;

        // Try to compute diameter (may fail for disconnected graphs)
        let diameter_value = diameter(&self.graph).unwrap_or(0);

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
            .map(|component| {
                component
                    .iter()
                    .filter_map(|id| self.node_map.get(id).cloned())
                    .collect()
            })
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
        let mut sorted: Vec<_> = self.scores.iter().map(|(name, score)| (name, *score)).collect();
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
    /// Graph diameter (longest shortest path)
    pub diameter: usize,
    /// Graph density (0-1)
    pub density: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};
    use std::collections::HashMap;

    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect {
            urn: "urn:samm:test:1.0.0#TestAspect".to_string(),
            properties: vec![],
            operations: vec![],
            events: vec![],
            preferred_names: HashMap::new(),
            descriptions: HashMap::new(),
            see_references: vec![],
        };

        // Add 3 properties
        for i in 1..=3 {
            aspect.properties.push(Property {
                urn: format!("urn:samm:test:1.0.0#Property{}", i),
                characteristic: Some(Characteristic {
                    urn: format!("urn:samm:test:1.0.0#Char{}", i),
                    kind: CharacteristicKind::Trait,
                    data_type: Some("string".to_string()),
                    base_characteristic: None,
                    values: vec![],
                    unit: None,
                    min_value: None,
                    max_value: None,
                    default_value: None,
                    preferred_names: HashMap::new(),
                    descriptions: HashMap::new(),
                    see_references: vec![],
                }),
                example_value: None,
                optional: false,
                not_in_payload: false,
                payload_name: None,
                preferred_names: HashMap::new(),
                descriptions: HashMap::new(),
                see_references: vec![],
            });
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
        let (top_node, _score) = centrality.max_node().unwrap();
        assert!(top_node.contains("TestAspect") || top_node.contains("Property"));
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
