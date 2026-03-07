//! Network analysis and routing algorithms
//!
//! This module provides graph-based spatial network analysis including:
//! - Shortest path algorithms (Dijkstra, A*)
//! - Network connectivity analysis
//! - Route optimization
//!
//! # Overview
//!
//! Network analysis is essential for:
//! - Transportation routing
//! - Utility network analysis (water, electricity)
//! - Supply chain optimization
//! - Accessibility analysis
//!
//! # Examples
//!
//! ```rust
//! use oxirs_geosparql::analysis::network::{Network, dijkstra_shortest_path};
//! use oxirs_geosparql::geometry::Geometry;
//!
//! // Create network from LineString geometries
//! let roads = vec![
//!     Geometry::from_wkt("LINESTRING(0 0, 1 0)").unwrap(),
//!     Geometry::from_wkt("LINESTRING(1 0, 2 1)").unwrap(),
//! ];
//!
//! let network = Network::from_linestrings(&roads).unwrap();
//!
//! // Find shortest path
//! let path = dijkstra_shortest_path(&network, 0, 2).unwrap();
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::{Distance, Euclidean};
use geo_types::{Coord, Point};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Network node representing a vertex in the spatial network
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID
    pub id: usize,
    /// Geographic coordinate
    pub coord: Coord<f64>,
    /// Optional node attributes
    pub attributes: HashMap<String, String>,
}

/// Network edge representing a connection between nodes
#[derive(Debug, Clone)]
pub struct Edge {
    /// Edge ID
    pub id: usize,
    /// Source node ID
    pub from: usize,
    /// Target node ID
    pub to: usize,
    /// Edge weight (typically distance or cost)
    pub weight: f64,
    /// Optional geometry (polyline)
    pub geometry: Option<Vec<Coord<f64>>>,
    /// Optional edge attributes
    pub attributes: HashMap<String, String>,
}

/// Spatial network graph
#[derive(Debug, Clone)]
pub struct Network {
    /// Network nodes
    pub nodes: Vec<Node>,
    /// Network edges
    pub edges: Vec<Edge>,
    /// Adjacency list for efficient traversal
    pub adjacency: HashMap<usize, Vec<usize>>,
}

impl Network {
    /// Create a new empty network
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Create network from LineString geometries
    ///
    /// Automatically creates nodes at endpoints and edges from LineStrings
    pub fn from_linestrings(linestrings: &[Geometry]) -> Result<Self> {
        let mut network = Network::new();
        let mut coord_to_node: HashMap<(i64, i64), usize> = HashMap::new();

        let coord_hash =
            |c: &Coord<f64>| -> (i64, i64) { ((c.x * 1e6) as i64, (c.y * 1e6) as i64) };

        for geom in linestrings {
            if let geo_types::Geometry::LineString(ls) = &geom.geom {
                if ls.0.len() < 2 {
                    continue;
                }

                // Get or create start node
                let start_coord = ls.0[0];
                let start_hash = coord_hash(&start_coord);
                let start_id = *coord_to_node.entry(start_hash).or_insert_with(|| {
                    let id = network.nodes.len();
                    network.nodes.push(Node {
                        id,
                        coord: start_coord,
                        attributes: HashMap::new(),
                    });
                    id
                });

                // Get or create end node
                let end_coord = ls.0[ls.0.len() - 1];
                let end_hash = coord_hash(&end_coord);
                let end_id = *coord_to_node.entry(end_hash).or_insert_with(|| {
                    let id = network.nodes.len();
                    network.nodes.push(Node {
                        id,
                        coord: end_coord,
                        attributes: HashMap::new(),
                    });
                    id
                });

                // Calculate edge weight (Euclidean distance)
                let weight = Euclidean::distance(Point::from(start_coord), Point::from(end_coord));

                // Create edge
                let edge_id = network.edges.len();
                network.edges.push(Edge {
                    id: edge_id,
                    from: start_id,
                    to: end_id,
                    weight,
                    geometry: Some(ls.0.clone()),
                    attributes: HashMap::new(),
                });

                // Update adjacency
                network.adjacency.entry(start_id).or_default().push(edge_id);
            }
        }

        Ok(network)
    }

    /// Add a node to the network
    pub fn add_node(&mut self, coord: Coord<f64>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            coord,
            attributes: HashMap::new(),
        });
        id
    }

    /// Add an edge to the network
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> Result<usize> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err(GeoSparqlError::InvalidParameter(
                "Node IDs out of bounds".to_string(),
            ));
        }

        let id = self.edges.len();
        self.edges.push(Edge {
            id,
            from,
            to,
            weight,
            geometry: None,
            attributes: HashMap::new(),
        });

        self.adjacency.entry(from).or_default().push(id);

        Ok(id)
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_id: usize) -> Vec<usize> {
        if let Some(edges) = self.adjacency.get(&node_id) {
            edges
                .iter()
                .map(|edge_id| self.edges[*edge_id].to)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get edge weight between two nodes
    pub fn edge_weight(&self, from: usize, to: usize) -> Option<f64> {
        if let Some(edges) = self.adjacency.get(&from) {
            for edge_id in edges {
                let edge = &self.edges[*edge_id];
                if edge.to == to {
                    return Some(edge.weight);
                }
            }
        }
        None
    }

    /// Get total number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get total number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Find node closest to a coordinate
    pub fn find_nearest_node(&self, coord: &Coord<f64>) -> Option<usize> {
        let point = Point::from(*coord);
        self.nodes
            .iter()
            .min_by(|a, b| {
                let dist_a = Euclidean::distance(point, Point::from(a.coord));
                let dist_b = Euclidean::distance(point, Point::from(b.coord));
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|node| node.id)
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// State for priority queue in Dijkstra's algorithm
#[derive(Debug, Clone)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of a shortest path search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    /// Node IDs in the path from start to end
    pub nodes: Vec<usize>,
    /// Total cost of the path
    pub cost: f64,
    /// Edge IDs used in the path
    pub edges: Vec<usize>,
}

/// Find shortest path using Dijkstra's algorithm
///
/// # Arguments
///
/// * `network` - The spatial network
/// * `start` - Start node ID
/// * `end` - End node ID
///
/// # Returns
///
/// PathResult containing the shortest path or error if no path exists
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::analysis::network::{Network, dijkstra_shortest_path};
///
/// let mut network = Network::new();
/// let n0 = network.add_node((0.0, 0.0).into());
/// let n1 = network.add_node((1.0, 0.0).into());
/// let n2 = network.add_node((2.0, 1.0).into());
/// network.add_edge(n0, n1, 1.0).unwrap();
/// network.add_edge(n1, n2, 1.5).unwrap();
///
/// let path = dijkstra_shortest_path(&network, n0, n2).unwrap();
/// assert_eq!(path.cost, 2.5);
/// ```
pub fn dijkstra_shortest_path(network: &Network, start: usize, end: usize) -> Result<PathResult> {
    if start >= network.nodes.len() || end >= network.nodes.len() {
        return Err(GeoSparqlError::InvalidParameter(
            "Node IDs out of bounds".to_string(),
        ));
    }

    let mut distances: Vec<f64> = vec![f64::INFINITY; network.nodes.len()];
    let mut previous: Vec<Option<usize>> = vec![None; network.nodes.len()];
    let mut heap = BinaryHeap::new();

    distances[start] = 0.0;
    heap.push(State {
        cost: 0.0,
        node: start,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if node == end {
            break;
        }

        if cost > distances[node] {
            continue;
        }

        if let Some(edges) = network.adjacency.get(&node) {
            for edge_id in edges {
                let edge = &network.edges[*edge_id];
                let next = State {
                    cost: cost + edge.weight,
                    node: edge.to,
                };

                if next.cost < distances[next.node] {
                    distances[next.node] = next.cost;
                    previous[next.node] = Some(node);
                    heap.push(next);
                }
            }
        }
    }

    if distances[end].is_infinite() {
        return Err(GeoSparqlError::ComputationError(
            "No path found between nodes".to_string(),
        ));
    }

    // Reconstruct path
    let mut path_nodes = Vec::new();
    let mut current = end;

    while let Some(prev) = previous[current] {
        path_nodes.push(current);
        current = prev;
    }
    path_nodes.push(start);
    path_nodes.reverse();

    // Find edge IDs
    let mut path_edges = Vec::new();
    for i in 0..path_nodes.len() - 1 {
        let from = path_nodes[i];
        let to = path_nodes[i + 1];

        if let Some(edges) = network.adjacency.get(&from) {
            for edge_id in edges {
                if network.edges[*edge_id].to == to {
                    path_edges.push(*edge_id);
                    break;
                }
            }
        }
    }

    Ok(PathResult {
        nodes: path_nodes,
        cost: distances[end],
        edges: path_edges,
    })
}

/// Find shortest path using A* algorithm with Euclidean distance heuristic
///
/// A* is more efficient than Dijkstra when the goal is known, as it uses
/// a heuristic to guide the search.
///
/// # Arguments
///
/// * `network` - The spatial network
/// * `start` - Start node ID
/// * `end` - End node ID
///
/// # Returns
///
/// PathResult containing the shortest path
pub fn astar_shortest_path(network: &Network, start: usize, end: usize) -> Result<PathResult> {
    if start >= network.nodes.len() || end >= network.nodes.len() {
        return Err(GeoSparqlError::InvalidParameter(
            "Node IDs out of bounds".to_string(),
        ));
    }

    let goal_coord = network.nodes[end].coord;

    let heuristic = |node_id: usize| -> f64 {
        let node_coord = network.nodes[node_id].coord;
        Euclidean::distance(Point::from(node_coord), Point::from(goal_coord))
    };

    let mut g_scores: Vec<f64> = vec![f64::INFINITY; network.nodes.len()];
    let mut f_scores: Vec<f64> = vec![f64::INFINITY; network.nodes.len()];
    let mut previous: Vec<Option<usize>> = vec![None; network.nodes.len()];
    let mut heap = BinaryHeap::new();

    g_scores[start] = 0.0;
    f_scores[start] = heuristic(start);

    heap.push(State {
        cost: f_scores[start],
        node: start,
    });

    let mut visited = HashSet::new();

    while let Some(State { cost: _, node }) = heap.pop() {
        if node == end {
            break;
        }

        if visited.contains(&node) {
            continue;
        }
        visited.insert(node);

        if let Some(edges) = network.adjacency.get(&node) {
            for edge_id in edges {
                let edge = &network.edges[*edge_id];
                let tentative_g = g_scores[node] + edge.weight;

                if tentative_g < g_scores[edge.to] {
                    previous[edge.to] = Some(node);
                    g_scores[edge.to] = tentative_g;
                    f_scores[edge.to] = tentative_g + heuristic(edge.to);

                    heap.push(State {
                        cost: f_scores[edge.to],
                        node: edge.to,
                    });
                }
            }
        }
    }

    if g_scores[end].is_infinite() {
        return Err(GeoSparqlError::ComputationError(
            "No path found between nodes".to_string(),
        ));
    }

    // Reconstruct path
    let mut path_nodes = Vec::new();
    let mut current = end;

    while let Some(prev) = previous[current] {
        path_nodes.push(current);
        current = prev;
    }
    path_nodes.push(start);
    path_nodes.reverse();

    // Find edge IDs
    let mut path_edges = Vec::new();
    for i in 0..path_nodes.len() - 1 {
        let from = path_nodes[i];
        let to = path_nodes[i + 1];

        if let Some(edges) = network.adjacency.get(&from) {
            for edge_id in edges {
                if network.edges[*edge_id].to == to {
                    path_edges.push(*edge_id);
                    break;
                }
            }
        }
    }

    Ok(PathResult {
        nodes: path_nodes,
        cost: g_scores[end],
        edges: path_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let mut network = Network::new();

        let n0 = network.add_node((0.0, 0.0).into());
        let n1 = network.add_node((1.0, 0.0).into());
        let n2 = network.add_node((1.0, 1.0).into());

        assert_eq!(network.node_count(), 3);

        network.add_edge(n0, n1, 1.0).unwrap();
        network.add_edge(n1, n2, 1.0).unwrap();

        assert_eq!(network.edge_count(), 2);
    }

    #[test]
    fn test_from_linestrings() {
        let lines = vec![
            Geometry::from_wkt("LINESTRING(0 0, 1 0)").unwrap(),
            Geometry::from_wkt("LINESTRING(1 0, 2 1)").unwrap(),
        ];

        let network = Network::from_linestrings(&lines).unwrap();

        assert_eq!(network.node_count(), 3);
        assert_eq!(network.edge_count(), 2);
    }

    #[test]
    fn test_dijkstra_simple() {
        let mut network = Network::new();

        let n0 = network.add_node((0.0, 0.0).into());
        let n1 = network.add_node((1.0, 0.0).into());
        let n2 = network.add_node((2.0, 0.0).into());

        network.add_edge(n0, n1, 1.0).unwrap();
        network.add_edge(n1, n2, 1.0).unwrap();

        let path = dijkstra_shortest_path(&network, n0, n2).unwrap();

        assert_eq!(path.nodes, vec![n0, n1, n2]);
        assert_eq!(path.cost, 2.0);
        assert_eq!(path.edges.len(), 2);
    }

    #[test]
    fn test_dijkstra_alternative_route() {
        let mut network = Network::new();

        let n0 = network.add_node((0.0, 0.0).into());
        let n1 = network.add_node((1.0, 0.0).into());
        let n2 = network.add_node((1.0, 1.0).into());
        let n3 = network.add_node((2.0, 1.0).into());

        // Two routes: direct (cost 3) and via n1 (cost 2.5)
        network.add_edge(n0, n3, 3.0).unwrap();
        network.add_edge(n0, n1, 1.0).unwrap();
        network.add_edge(n1, n2, 0.5).unwrap();
        network.add_edge(n2, n3, 1.0).unwrap();

        let path = dijkstra_shortest_path(&network, n0, n3).unwrap();

        assert_eq!(path.cost, 2.5);
        assert_eq!(path.nodes.len(), 4);
    }

    #[test]
    fn test_astar_simple() {
        let mut network = Network::new();

        let n0 = network.add_node((0.0, 0.0).into());
        let n1 = network.add_node((1.0, 0.0).into());
        let n2 = network.add_node((2.0, 0.0).into());

        network.add_edge(n0, n1, 1.0).unwrap();
        network.add_edge(n1, n2, 1.0).unwrap();

        let path = astar_shortest_path(&network, n0, n2).unwrap();

        assert_eq!(path.nodes, vec![n0, n1, n2]);
        assert_eq!(path.cost, 2.0);
    }

    #[test]
    fn test_no_path() {
        let mut network = Network::new();

        let n0 = network.add_node((0.0, 0.0).into());
        let n1 = network.add_node((1.0, 0.0).into());
        let n2 = network.add_node((10.0, 10.0).into());

        network.add_edge(n0, n1, 1.0).unwrap();

        // No path from n0 to n2
        assert!(dijkstra_shortest_path(&network, n0, n2).is_err());
    }

    #[test]
    fn test_find_nearest_node() {
        let mut network = Network::new();

        network.add_node((0.0, 0.0).into());
        network.add_node((1.0, 0.0).into());
        network.add_node((2.0, 0.0).into());

        let nearest = network
            .find_nearest_node(&Coord { x: 0.9, y: 0.0 })
            .unwrap();
        assert_eq!(nearest, 1);
    }

    #[test]
    fn test_neighbors() {
        let mut network = Network::new();

        let n0 = network.add_node((0.0, 0.0).into());
        let n1 = network.add_node((1.0, 0.0).into());
        let n2 = network.add_node((2.0, 0.0).into());

        network.add_edge(n0, n1, 1.0).unwrap();
        network.add_edge(n0, n2, 2.0).unwrap();

        let neighbors = network.neighbors(n0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&n1));
        assert!(neighbors.contains(&n2));
    }
}
