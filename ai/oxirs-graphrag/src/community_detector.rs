//! Graph community detection using a greedy label-propagation approach.
//!
//! This module implements a Louvain-inspired greedy community detection
//! algorithm. Each node iteratively adopts the community label of its
//! most-connected neighbour until the assignment stabilises or the maximum
//! number of iterations is reached. Small communities below `min_community_size`
//! are merged into their largest neighbour community.
//!
//! # Example
//!
//! ```rust
//! use oxirs_graphrag::community_detector::{CommunityGraph, CommunityDetector};
//!
//! let mut graph = CommunityGraph::new();
//! graph.add_node(1, "Alice");
//! graph.add_node(2, "Bob");
//! graph.add_node(3, "Carol");
//! graph.add_edge(1, 2, 1.0);
//! graph.add_edge(2, 3, 1.0);
//!
//! let detector = CommunityDetector::new(1, 100);
//! let result = detector.detect(&mut graph);
//! assert!(!result.communities.is_empty());
//! assert!(result.modularity >= -1.0);
//! ```

use std::collections::HashMap;

/// A node in the community graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: u64,
    /// Human-readable label
    pub label: String,
    /// Assigned community ID (None until detection runs)
    pub community: Option<u32>,
}

impl GraphNode {
    /// Create a new unassigned graph node
    pub fn new(id: u64, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            community: None,
        }
    }
}

/// A weighted edge between two nodes
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: u64,
    pub to: u64,
    pub weight: f64,
}

impl GraphEdge {
    /// Create a new edge
    pub fn new(from: u64, to: u64, weight: f64) -> Self {
        Self { from, to, weight }
    }
}

/// A detected community grouping
#[derive(Debug, Clone)]
pub struct Community {
    /// Unique community ID
    pub id: u32,
    /// Node IDs that belong to this community
    pub members: Vec<u64>,
    /// Count of edges whose both endpoints are inside the community
    pub internal_edges: u64,
    /// Count of all edges incident to any node in the community
    pub total_edges: u64,
}

impl Community {
    /// Return the number of members
    pub fn size(&self) -> usize {
        self.members.len()
    }
}

/// Undirected weighted graph with adjacency lists
#[derive(Debug, Clone, Default)]
pub struct CommunityGraph {
    /// Node collection indexed by ID
    pub nodes: HashMap<u64, GraphNode>,
    /// All edges (stored once; adjacency is bidirectional)
    pub edges: Vec<GraphEdge>,
    /// Adjacency list: node_id → [(neighbour_id, weight)]
    pub adjacency: HashMap<u64, Vec<(u64, f64)>>,
}

impl CommunityGraph {
    /// Create an empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add a node; overwrites if the ID already exists
    pub fn add_node(&mut self, id: u64, label: &str) {
        self.nodes.insert(id, GraphNode::new(id, label));
        self.adjacency.entry(id).or_default();
    }

    /// Add an undirected edge between `from` and `to` with the given weight.
    /// If either node does not exist it is created with an empty label.
    pub fn add_edge(&mut self, from: u64, to: u64, weight: f64) {
        self.adjacency.entry(from).or_default();
        self.adjacency.entry(to).or_default();

        self.adjacency
            .get_mut(&from)
            .expect("from adjacency entry just inserted")
            .push((to, weight));
        self.adjacency
            .get_mut(&to)
            .expect("to adjacency entry just inserted")
            .push((from, weight));

        self.edges.push(GraphEdge::new(from, to, weight));
    }

    /// Sum of edge weights incident to a node (weighted degree)
    pub fn degree(&self, node_id: u64) -> f64 {
        self.adjacency
            .get(&node_id)
            .map(|neighbours| neighbours.iter().map(|(_, w)| w).sum())
            .unwrap_or(0.0)
    }

    /// Sum of all edge weights in the graph (counted once per edge)
    pub fn total_weight(&self) -> f64 {
        self.edges.iter().map(|e| e.weight).sum()
    }
}

/// Outcome of a community detection run
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Final communities (after merging small ones)
    pub communities: Vec<Community>,
    /// Newman–Girvan modularity Q ∈ [-1, 1]
    pub modularity: f64,
    /// Number of label-propagation iterations performed
    pub iterations: u32,
}

/// Community detection engine
#[derive(Debug, Clone)]
pub struct CommunityDetector {
    /// Communities smaller than this are merged into a neighbour community
    pub min_community_size: usize,
    /// Maximum number of label-propagation iterations
    pub max_iterations: u32,
    /// Modularity resolution parameter γ (1.0 = standard modularity)
    pub resolution: f64,
}

impl CommunityDetector {
    /// Create a detector with the given parameters.
    /// `resolution` defaults to 1.0 (standard Newman–Girvan modularity).
    pub fn new(min_community_size: usize, max_iterations: u32) -> Self {
        Self {
            min_community_size,
            max_iterations,
            resolution: 1.0,
        }
    }

    /// Create a detector with a custom resolution parameter
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Run community detection on `graph`.
    ///
    /// Each node's `community` field is updated in place.
    /// Returns a [`DetectionResult`] with the final partition and quality metrics.
    pub fn detect(&self, graph: &mut CommunityGraph) -> DetectionResult {
        // Initialise: each node belongs to its own community
        let node_ids: Vec<u64> = graph.nodes.keys().copied().collect();

        if node_ids.is_empty() {
            return DetectionResult {
                communities: vec![],
                modularity: 0.0,
                iterations: 0,
            };
        }

        // Assign initial community IDs (community_id == node index for simplicity)
        let mut community_map: HashMap<u64, u32> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();

        let mut iterations = 0u32;
        let mut changed = true;

        while changed && iterations < self.max_iterations {
            changed = false;
            iterations += 1;

            // Process nodes in a deterministic order
            let mut sorted_ids = node_ids.clone();
            sorted_ids.sort_unstable();

            for &node_id in &sorted_ids {
                let best_community = self.best_neighbour_community(graph, node_id, &community_map);
                let current = *community_map.get(&node_id).unwrap_or(&0);
                if best_community != current {
                    community_map.insert(node_id, best_community);
                    changed = true;
                }
            }
        }

        // Write community assignments back to nodes
        for (&node_id, &comm) in &community_map {
            if let Some(node) = graph.nodes.get_mut(&node_id) {
                node.community = Some(comm);
            }
        }

        // Build Community structs
        let mut community_members: HashMap<u32, Vec<u64>> = HashMap::new();
        for (&node_id, &comm) in &community_map {
            community_members.entry(comm).or_default().push(node_id);
        }

        let mut communities: Vec<Community> = community_members
            .into_iter()
            .map(|(comm_id, members)| {
                let member_set: std::collections::HashSet<u64> = members.iter().copied().collect();
                let internal = graph
                    .edges
                    .iter()
                    .filter(|e| member_set.contains(&e.from) && member_set.contains(&e.to))
                    .count() as u64;
                let total = graph
                    .edges
                    .iter()
                    .filter(|e| member_set.contains(&e.from) || member_set.contains(&e.to))
                    .count() as u64;
                Community {
                    id: comm_id,
                    members,
                    internal_edges: internal,
                    total_edges: total,
                }
            })
            .collect();

        // Sort for deterministic output
        communities.sort_by_key(|c| c.id);

        let mut result = DetectionResult {
            communities,
            modularity: 0.0,
            iterations,
        };

        // Merge communities that are too small
        self.merge_small_communities(&mut result, graph);

        // Compute modularity after merging
        result.modularity = self.compute_modularity(graph);

        result
    }

    /// Find the community with the highest connection weight from `node_id`.
    /// Falls back to the node's own current community if it has no neighbours.
    fn best_neighbour_community(
        &self,
        graph: &CommunityGraph,
        node_id: u64,
        community_map: &HashMap<u64, u32>,
    ) -> u32 {
        let current = *community_map.get(&node_id).unwrap_or(&0);
        let neighbours = match graph.adjacency.get(&node_id) {
            Some(n) => n,
            None => return current,
        };

        if neighbours.is_empty() {
            return current;
        }

        // Accumulate total weight per neighbouring community
        let mut comm_weight: HashMap<u32, f64> = HashMap::new();
        for &(nb_id, weight) in neighbours {
            let nb_comm = *community_map.get(&nb_id).unwrap_or(&0);
            *comm_weight.entry(nb_comm).or_insert(0.0) += weight;
        }

        // Choose community with maximum weight
        comm_weight
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c)
            .unwrap_or(current)
    }

    /// Compute the Newman–Girvan modularity Q for the current community assignment.
    ///
    /// Q = (1/2m) Σ_{ij} [ A_ij − γ k_i k_j / 2m ] δ(c_i, c_j)
    ///
    /// where m is the total weight, k_i is the degree of node i, A_ij is the
    /// edge weight between i and j (0 if no edge), and δ is 1 iff same community.
    pub fn compute_modularity(&self, graph: &CommunityGraph) -> f64 {
        let two_m = graph.total_weight() * 2.0;
        if two_m == 0.0 {
            return 0.0;
        }

        // Build community assignment lookup
        let comm_of: HashMap<u64, u32> = graph
            .nodes
            .values()
            .filter_map(|n| n.community.map(|c| (n.id, c)))
            .collect();

        // Build edge-weight lookup (sum for multi-edges)
        let mut edge_weight_map: HashMap<(u64, u64), f64> = HashMap::new();
        for e in &graph.edges {
            let key = if e.from <= e.to {
                (e.from, e.to)
            } else {
                (e.to, e.from)
            };
            *edge_weight_map.entry(key).or_insert(0.0) += e.weight;
        }

        let node_ids: Vec<u64> = graph.nodes.keys().copied().collect();
        let mut q = 0.0_f64;

        for &i in &node_ids {
            let ci = match comm_of.get(&i) {
                Some(&c) => c,
                None => continue,
            };
            let ki = graph.degree(i);
            for &j in &node_ids {
                let cj = match comm_of.get(&j) {
                    Some(&c) => c,
                    None => continue,
                };
                if ci != cj {
                    continue;
                }
                let key = if i <= j { (i, j) } else { (j, i) };
                let a_ij = edge_weight_map.get(&key).copied().unwrap_or(0.0);
                let kj = graph.degree(j);
                q += a_ij - self.resolution * ki * kj / two_m;
            }
        }

        q / two_m
    }

    /// Compute the modularity gain of moving `node_id` into `target_community`.
    ///
    /// Returns a positive value if the move improves modularity.
    pub fn modularity_gain(
        &self,
        graph: &CommunityGraph,
        node_id: u64,
        target_community: u32,
    ) -> f64 {
        let two_m = graph.total_weight() * 2.0;
        if two_m == 0.0 {
            return 0.0;
        }

        let ki = graph.degree(node_id);

        // Weight of edges from node_id to target community members
        let k_in: f64 = graph
            .adjacency
            .get(&node_id)
            .map(|neighbours| {
                neighbours
                    .iter()
                    .filter_map(|(nb_id, w)| {
                        graph.nodes.get(nb_id).and_then(|n| {
                            if n.community == Some(target_community) {
                                Some(*w)
                            } else {
                                None
                            }
                        })
                    })
                    .sum()
            })
            .unwrap_or(0.0);

        // Total weight of edges incident to target community
        let sigma_tot: f64 = graph
            .nodes
            .values()
            .filter(|n| n.community == Some(target_community))
            .map(|n| graph.degree(n.id))
            .sum();

        // ΔQ = [ k_in/m - γ * sigma_tot * ki / (2m²) ]
        2.0 * k_in / two_m - self.resolution * sigma_tot * ki / (two_m * two_m)
    }

    /// Merge communities whose member count is below `min_community_size`.
    ///
    /// Each small community's members are reassigned to the community that
    /// shares the most edge weight with them.  The reassignment is reflected
    /// in `graph.nodes[*].community` and in `result.communities`.
    pub fn merge_small_communities(
        &self,
        result: &mut DetectionResult,
        graph: &mut CommunityGraph,
    ) {
        if self.min_community_size <= 1 {
            return;
        }

        let small_ids: Vec<u32> = result
            .communities
            .iter()
            .filter(|c| c.members.len() < self.min_community_size)
            .map(|c| c.id)
            .collect();

        for small_id in small_ids {
            // Find the target community with most edge-weight connection
            let members: Vec<u64> = result
                .communities
                .iter()
                .find(|c| c.id == small_id)
                .map(|c| c.members.clone())
                .unwrap_or_default();

            if members.is_empty() {
                continue;
            }

            // Accumulate weight to every other community
            let mut target_weight: HashMap<u32, f64> = HashMap::new();
            for &m in &members {
                if let Some(neighbours) = graph.adjacency.get(&m) {
                    for &(nb_id, weight) in neighbours {
                        if let Some(node) = graph.nodes.get(&nb_id) {
                            if let Some(nb_comm) = node.community {
                                if nb_comm != small_id {
                                    *target_weight.entry(nb_comm).or_insert(0.0) += weight;
                                }
                            }
                        }
                    }
                }
            }

            let best_target = target_weight
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| c);

            let Some(target_id) = best_target else {
                // No neighbour community found — skip
                continue;
            };

            // Reassign community in nodes
            for &m in &members {
                if let Some(node) = graph.nodes.get_mut(&m) {
                    node.community = Some(target_id);
                }
            }

            // Move members in the Community structs
            let moved_members = members.clone();
            if let Some(target) = result.communities.iter_mut().find(|c| c.id == target_id) {
                target.members.extend_from_slice(&moved_members);
            }

            // Remove the small community
            result.communities.retain(|c| c.id != small_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> CommunityGraph {
        let mut g = CommunityGraph::new();
        g.add_node(1, "A");
        g.add_node(2, "B");
        g.add_node(3, "C");
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 1, 1.0);
        g
    }

    fn two_cliques() -> CommunityGraph {
        // Clique 1: 1-2-3 strongly connected
        // Clique 2: 4-5-6 strongly connected
        // Weak bridge: 3-4
        let mut g = CommunityGraph::new();
        for id in 1..=6 {
            g.add_node(id, &format!("n{}", id));
        }
        g.add_edge(1, 2, 10.0);
        g.add_edge(2, 3, 10.0);
        g.add_edge(1, 3, 10.0);
        g.add_edge(4, 5, 10.0);
        g.add_edge(5, 6, 10.0);
        g.add_edge(4, 6, 10.0);
        g.add_edge(3, 4, 0.1); // weak bridge
        g
    }

    // --- CommunityGraph construction ---

    #[test]
    fn test_add_node_stores_label() {
        let mut g = CommunityGraph::new();
        g.add_node(42, "TestNode");
        let node = g.nodes.get(&42).expect("node should exist");
        assert_eq!(node.label, "TestNode");
        assert_eq!(node.community, None);
    }

    #[test]
    fn test_add_edge_updates_adjacency_both_directions() {
        let mut g = CommunityGraph::new();
        g.add_node(1, "A");
        g.add_node(2, "B");
        g.add_edge(1, 2, 3.0);

        let adj1 = g.adjacency.get(&1).expect("adj[1] should exist");
        assert!(adj1
            .iter()
            .any(|(nb, w)| *nb == 2 && (*w - 3.0).abs() < 1e-10));

        let adj2 = g.adjacency.get(&2).expect("adj[2] should exist");
        assert!(adj2
            .iter()
            .any(|(nb, w)| *nb == 1 && (*w - 3.0).abs() < 1e-10));
    }

    #[test]
    fn test_degree_empty_node() {
        let mut g = CommunityGraph::new();
        g.add_node(1, "A");
        assert_eq!(g.degree(1), 0.0);
    }

    #[test]
    fn test_degree_with_edges() {
        let g = triangle_graph();
        // Each node connects to two others with weight 1.0 each
        assert!((g.degree(1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_weight_triangle() {
        let g = triangle_graph();
        assert!((g.total_weight() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_weight_empty() {
        let g = CommunityGraph::new();
        assert_eq!(g.total_weight(), 0.0);
    }

    #[test]
    fn test_degree_missing_node_returns_zero() {
        let g = CommunityGraph::new();
        assert_eq!(g.degree(9999), 0.0);
    }

    // --- Detection on simple graphs ---

    #[test]
    fn test_detect_empty_graph_returns_empty() {
        let mut g = CommunityGraph::new();
        let detector = CommunityDetector::new(1, 100);
        let result = detector.detect(&mut g);
        assert!(result.communities.is_empty());
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_detect_single_node() {
        let mut g = CommunityGraph::new();
        g.add_node(1, "solo");
        let detector = CommunityDetector::new(1, 100);
        let result = detector.detect(&mut g);
        assert_eq!(result.communities.len(), 1);
        assert_eq!(result.communities[0].members.len(), 1);
    }

    #[test]
    fn test_detect_triangle_assigns_communities() {
        let mut g = triangle_graph();
        let detector = CommunityDetector::new(1, 50);
        let result = detector.detect(&mut g);
        // All nodes should have a community assigned
        for node in g.nodes.values() {
            assert!(
                node.community.is_some(),
                "Node {} has no community",
                node.id
            );
        }
        assert!(!result.communities.is_empty());
    }

    #[test]
    fn test_detect_two_cliques_partition() {
        let mut g = two_cliques();
        let detector = CommunityDetector::new(1, 200);
        let result = detector.detect(&mut g);
        // We expect either 1 or 2 communities (algorithm may or may not split)
        assert!(!result.communities.is_empty());
        // Total members across all communities == 6
        let total_members: usize = result.communities.iter().map(|c| c.members.len()).sum();
        assert_eq!(total_members, 6);
    }

    #[test]
    fn test_detect_respects_max_iterations() {
        let mut g = two_cliques();
        let detector = CommunityDetector::new(1, 3); // very few iterations
        let result = detector.detect(&mut g);
        assert!(result.iterations <= 3);
    }

    #[test]
    fn test_detect_covers_all_nodes() {
        let mut g = two_cliques();
        let detector = CommunityDetector::new(1, 100);
        let result = detector.detect(&mut g);
        let total_members: usize = result.communities.iter().map(|c| c.members.len()).sum();
        assert_eq!(total_members, g.nodes.len());
    }

    // --- Modularity ---

    #[test]
    fn test_modularity_empty_graph() {
        let g = CommunityGraph::new();
        let detector = CommunityDetector::new(1, 10);
        let q = detector.compute_modularity(&g);
        assert_eq!(q, 0.0);
    }

    #[test]
    fn test_modularity_in_range_after_detection() {
        let mut g = two_cliques();
        let detector = CommunityDetector::new(1, 100);
        let result = detector.detect(&mut g);
        // Modularity must lie in [-1, 1]
        assert!(
            result.modularity >= -1.0 && result.modularity <= 1.0,
            "Q={} out of range",
            result.modularity
        );
    }

    #[test]
    fn test_modularity_single_community_is_non_positive() {
        let mut g = triangle_graph();
        // Force all into community 0
        for node in g.nodes.values_mut() {
            node.community = Some(0);
        }
        let detector = CommunityDetector::new(1, 10);
        let q = detector.compute_modularity(&g);
        // All-in-one partition has Q <= 0 for most graphs
        assert!(q <= 0.0 + 1e-10, "Expected Q <= 0, got {}", q);
    }

    // --- Modularity gain ---

    #[test]
    fn test_modularity_gain_returns_finite() {
        let mut g = triangle_graph();
        let detector = CommunityDetector::new(1, 50);
        detector.detect(&mut g);
        let gain = detector.modularity_gain(&g, 1, 0);
        assert!(gain.is_finite());
    }

    #[test]
    fn test_modularity_gain_empty_graph() {
        let g = CommunityGraph::new();
        let detector = CommunityDetector::new(1, 10);
        let gain = detector.modularity_gain(&g, 1, 0);
        assert_eq!(gain, 0.0);
    }

    // --- Small community merging ---

    #[test]
    fn test_merge_small_communities_removes_tiny() {
        let mut g = CommunityGraph::new();
        g.add_node(1, "A");
        g.add_node(2, "B");
        g.add_node(3, "C");
        g.add_edge(1, 2, 5.0);
        g.add_edge(2, 3, 1.0);

        let detector = CommunityDetector::new(2, 100); // min size 2
        let result = detector.detect(&mut g);

        // All communities should have >= 2 members OR be the only community
        for comm in &result.communities {
            assert!(
                comm.members.len() >= 2 || result.communities.len() == 1,
                "Community {} has only {} member(s)",
                comm.id,
                comm.members.len()
            );
        }
    }

    #[test]
    fn test_merge_preserves_total_node_count() {
        let mut g = two_cliques();
        let detector = CommunityDetector::new(3, 100);
        let result = detector.detect(&mut g);
        let total: usize = result.communities.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 6, "All 6 nodes must appear in some community");
    }

    // --- CommunityDetector builder ---

    #[test]
    fn test_with_resolution_sets_field() {
        let d = CommunityDetector::new(1, 50).with_resolution(0.5);
        assert!((d.resolution - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_default_resolution_is_one() {
        let d = CommunityDetector::new(1, 50);
        assert!((d.resolution - 1.0).abs() < 1e-10);
    }

    // --- Community struct ---

    #[test]
    fn test_community_size() {
        let c = Community {
            id: 0,
            members: vec![1, 2, 3],
            internal_edges: 3,
            total_edges: 4,
        };
        assert_eq!(c.size(), 3);
    }

    // --- GraphNode ---

    #[test]
    fn test_graph_node_initial_community_is_none() {
        let node = GraphNode::new(7, "test");
        assert_eq!(node.community, None);
    }

    // --- Disconnected graph ---

    #[test]
    fn test_detect_disconnected_graph() {
        let mut g = CommunityGraph::new();
        // Two isolated nodes
        g.add_node(1, "X");
        g.add_node(2, "Y");
        let detector = CommunityDetector::new(1, 50);
        let result = detector.detect(&mut g);
        let total: usize = result.communities.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_detect_star_graph() {
        let mut g = CommunityGraph::new();
        g.add_node(0, "center");
        for i in 1..=5 {
            g.add_node(i, &format!("leaf{}", i));
            g.add_edge(0, i, 1.0);
        }
        let detector = CommunityDetector::new(1, 100);
        let result = detector.detect(&mut g);
        let total: usize = result.communities.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 6);
    }
}

// ---------------------------------------------------------------------------
// v1.1.0 round 16: String-keyed community detection API
// ---------------------------------------------------------------------------

/// A directed or undirected edge in the knowledge graph using String node names.
#[derive(Debug, Clone, PartialEq)]
pub struct KgEdge {
    /// Source node identifier.
    pub from: String,
    /// Target node identifier.
    pub to: String,
    /// Edge weight (e.g. predicate frequency or confidence).
    pub weight: f64,
}

impl KgEdge {
    /// Create a new knowledge-graph edge.
    pub fn new(from: impl Into<String>, to: impl Into<String>, weight: f64) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            weight,
        }
    }
}

/// A community returned by the string-keyed detector.
#[derive(Debug, Clone)]
pub struct KgCommunity {
    /// Unique community identifier.
    pub community_id: usize,
    /// Names of nodes belonging to this community.
    pub members: Vec<String>,
}

/// Configuration for the string-keyed community detector.
#[derive(Debug, Clone)]
pub struct KgDetectionConfig {
    /// Minimum community size; communities below this are merged into the largest
    /// neighbouring community.
    pub min_community_size: usize,
    /// Maximum number of communities to return (0 = unlimited).
    pub max_communities: usize,
}

impl Default for KgDetectionConfig {
    fn default() -> Self {
        Self {
            min_community_size: 1,
            max_communities: 0,
        }
    }
}

/// Greedy community detector operating on String-keyed node sets.
///
/// Uses iterative label propagation: each node starts in its own community
/// and repeatedly adopts the most frequent label among its neighbours.
/// The process continues for a fixed number of rounds.
pub struct KgCommunityDetector;

impl Default for KgCommunityDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl KgCommunityDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        KgCommunityDetector
    }

    /// Detect communities using label propagation.
    ///
    /// * Nodes with no edges form singleton communities (or are merged if
    ///   `config.min_community_size > 1`).
    /// * When `config.max_communities > 0` only the largest communities are
    ///   kept and the remaining nodes are merged into the largest one.
    pub fn detect(
        &self,
        nodes: &[String],
        edges: &[KgEdge],
        config: &KgDetectionConfig,
    ) -> Vec<KgCommunity> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Build adjacency: node_name → Vec<(neighbour, weight)>
        let mut adjacency: std::collections::HashMap<&str, Vec<(&str, f64)>> =
            std::collections::HashMap::new();
        for n in nodes {
            adjacency.entry(n.as_str()).or_default();
        }
        for e in edges {
            adjacency
                .entry(e.from.as_str())
                .or_default()
                .push((e.to.as_str(), e.weight));
            adjacency
                .entry(e.to.as_str())
                .or_default()
                .push((e.from.as_str(), e.weight));
        }

        // Initialise: each node in its own label group (indexed by position)
        let mut labels: std::collections::HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        // Propagation rounds (deterministic order for reproducibility)
        let max_rounds = 20_usize;
        for _ in 0..max_rounds {
            let mut changed = false;
            let node_order: Vec<&str> = nodes.iter().map(String::as_str).collect();
            for &node in &node_order {
                let neighbours = adjacency.get(node).cloned().unwrap_or_default();
                if neighbours.is_empty() {
                    continue;
                }
                // Tally neighbour labels weighted by edge weight
                let mut tally: std::collections::HashMap<usize, f64> =
                    std::collections::HashMap::new();
                for (nb, w) in &neighbours {
                    if let Some(&lbl) = labels.get(nb) {
                        *tally.entry(lbl).or_insert(0.0) += w;
                    }
                }
                // Adopt the label with highest total weight (ties: prefer current)
                let current = *labels.get(node).unwrap_or(&0);
                let best = tally
                    .into_iter()
                    .max_by(|(la, wa), (lb, wb)| {
                        wa.partial_cmp(wb)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| {
                                // Tie-break: prefer current label
                                if *la == current {
                                    std::cmp::Ordering::Greater
                                } else if *lb == current {
                                    std::cmp::Ordering::Less
                                } else {
                                    std::cmp::Ordering::Equal
                                }
                            })
                    })
                    .map(|(lbl, _)| lbl);

                if let Some(best_lbl) = best {
                    if best_lbl != current {
                        if let Some(lbl) = labels.get_mut(node) {
                            *lbl = best_lbl;
                        }
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Collect label → members
        let mut groups: std::collections::HashMap<usize, Vec<String>> =
            std::collections::HashMap::new();
        for n in nodes {
            let lbl = *labels.get(n.as_str()).unwrap_or(&0);
            groups.entry(lbl).or_default().push(n.clone());
        }

        // Build initial community list (sorted for determinism)
        let mut communities: Vec<KgCommunity> = {
            let mut kvs: Vec<(usize, Vec<String>)> = groups.into_iter().collect();
            kvs.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
            kvs.into_iter()
                .enumerate()
                .map(|(id, (_, members))| KgCommunity {
                    community_id: id,
                    members,
                })
                .collect()
        };

        // Apply min_community_size: merge small communities into the largest
        if config.min_community_size > 1 {
            let mut large: Vec<KgCommunity> = Vec::new();
            let mut small_members: Vec<String> = Vec::new();
            for c in communities {
                if c.members.len() >= config.min_community_size {
                    large.push(c);
                } else {
                    small_members.extend(c.members);
                }
            }
            if !small_members.is_empty() {
                if large.is_empty() {
                    large.push(KgCommunity {
                        community_id: 0,
                        members: small_members,
                    });
                } else {
                    large[0].members.extend(small_members);
                }
            }
            communities = large;
        }

        // Apply max_communities cap
        if config.max_communities > 0 && communities.len() > config.max_communities {
            // Merge excess communities into the first (largest)
            let keep = config.max_communities;
            let excess: Vec<String> = communities.drain(keep..).flat_map(|c| c.members).collect();
            if !excess.is_empty() {
                communities[0].members.extend(excess);
            }
        }

        // Re-assign sequential community_ids
        for (i, c) in communities.iter_mut().enumerate() {
            c.community_id = i;
        }

        communities
    }

    /// Return the community a node belongs to, if any.
    pub fn node_community<'a>(
        &self,
        node: &str,
        communities: &'a [KgCommunity],
    ) -> Option<&'a KgCommunity> {
        communities
            .iter()
            .find(|c| c.members.iter().any(|m| m == node))
    }

    /// Compute the modularity Q of a community partition.
    ///
    /// Q = Σ_c [ (e_c / m) - (a_c / (2m))² ]
    /// where `e_c` is the fraction of edge weight inside community c,
    /// `a_c` is the sum of degrees in c, and `m` is the total edge weight.
    /// Returns 0.0 for empty graphs.
    pub fn modularity(
        &self,
        nodes: &[String],
        edges: &[KgEdge],
        communities: &[KgCommunity],
    ) -> f64 {
        if edges.is_empty() || nodes.is_empty() {
            return 0.0;
        }

        // Total weight
        let total_weight: f64 = edges.iter().map(|e| e.weight).sum::<f64>() * 2.0;
        if total_weight < 1e-12 {
            return 0.0;
        }
        let m = total_weight / 2.0;

        // Node → community map
        let mut node_comm: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for c in communities {
            for member in &c.members {
                node_comm.insert(member.as_str(), c.community_id);
            }
        }

        // Degree of each node
        let mut degree: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for e in edges {
            *degree.entry(e.from.as_str()).or_insert(0.0) += e.weight;
            *degree.entry(e.to.as_str()).or_insert(0.0) += e.weight;
        }

        let mut q = 0.0_f64;
        for e in edges {
            let ci = node_comm.get(e.from.as_str()).copied();
            let cj = node_comm.get(e.to.as_str()).copied();
            if ci == cj && ci.is_some() {
                let ki = degree.get(e.from.as_str()).copied().unwrap_or(0.0);
                let kj = degree.get(e.to.as_str()).copied().unwrap_or(0.0);
                // Count each undirected edge once (stored once → factor ×2 / 2m)
                q += e.weight / m - (ki * kj) / (2.0 * m * m);
            }
        }
        // Subtract penalty for cross-community pairs (already handled by absence)
        q
    }

    /// Count intra-community edges (both endpoints in the same community).
    pub fn intra_edges(edges: &[KgEdge], community: &KgCommunity) -> usize {
        let members: std::collections::HashSet<&str> =
            community.members.iter().map(String::as_str).collect();
        edges
            .iter()
            .filter(|e| members.contains(e.from.as_str()) && members.contains(e.to.as_str()))
            .count()
    }
}

#[cfg(test)]
mod kg_tests {
    use super::*;

    fn edge(from: &str, to: &str, w: f64) -> KgEdge {
        KgEdge::new(from, to, w)
    }

    fn nodes(names: &[&str]) -> Vec<String> {
        names.iter().map(|&s| s.to_string()).collect()
    }

    fn default_config() -> KgDetectionConfig {
        KgDetectionConfig::default()
    }

    fn det() -> KgCommunityDetector {
        KgCommunityDetector::new()
    }

    // --- Empty / single node ---

    #[test]
    fn test_empty_graph_returns_no_communities() {
        let d = det();
        let result = d.detect(&[], &[], &default_config());
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_node_forms_own_community() {
        let d = det();
        let ns = nodes(&["A"]);
        let result = d.detect(&ns, &[], &default_config());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].members, vec!["A"]);
    }

    // --- Isolated nodes ---

    #[test]
    fn test_isolated_nodes_each_own_community() {
        let d = det();
        let ns = nodes(&["A", "B", "C"]);
        let result = d.detect(&ns, &[], &default_config());
        // Each isolated node gets its own community
        let total: usize = result.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 3);
        assert_eq!(result.len(), 3);
    }

    // --- Connected cliques ---

    #[test]
    fn test_two_cliques_grouped() {
        let d = det();
        let ns = nodes(&["A", "B", "C", "X", "Y", "Z"]);
        let es = vec![
            edge("A", "B", 10.0),
            edge("B", "C", 10.0),
            edge("A", "C", 10.0),
            edge("X", "Y", 10.0),
            edge("Y", "Z", 10.0),
            edge("X", "Z", 10.0),
            // Weak link between cliques
            edge("C", "X", 0.1),
        ];
        let result = d.detect(&ns, &es, &default_config());
        // All 6 nodes should be covered
        let total: usize = result.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 6);
        // Should produce >= 1 community
        assert!(!result.is_empty());
    }

    #[test]
    fn test_fully_connected_single_community() {
        let d = det();
        let ns = nodes(&["A", "B", "C"]);
        let es = vec![
            edge("A", "B", 5.0),
            edge("B", "C", 5.0),
            edge("A", "C", 5.0),
        ];
        let result = d.detect(&ns, &es, &default_config());
        let total: usize = result.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 3);
    }

    // --- min_community_size ---

    #[test]
    fn test_min_community_size_merges_singletons() {
        let d = det();
        let ns = nodes(&["A", "B", "C", "D"]);
        // No edges → all singletons
        let config = KgDetectionConfig {
            min_community_size: 2,
            max_communities: 0,
        };
        let result = d.detect(&ns, &[], &config);
        // Singletons merged: all in one or two communities, each >= 2
        let total: usize = result.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 4);
        for c in &result {
            assert!(
                c.members.len() >= 2 || result.len() == 1,
                "community too small: {}",
                c.members.len()
            );
        }
    }

    #[test]
    fn test_min_community_size_one_no_effect() {
        let d = det();
        let ns = nodes(&["A", "B"]);
        let config = KgDetectionConfig {
            min_community_size: 1,
            max_communities: 0,
        };
        let result = d.detect(&ns, &[], &config);
        assert_eq!(result.len(), 2);
    }

    // --- max_communities ---

    #[test]
    fn test_max_communities_cap() {
        let d = det();
        let ns = nodes(&["A", "B", "C", "D", "E"]);
        // No edges → 5 communities; cap at 2
        let config = KgDetectionConfig {
            min_community_size: 1,
            max_communities: 2,
        };
        let result = d.detect(&ns, &[], &config);
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_max_communities_zero_means_unlimited() {
        let d = det();
        let ns = nodes(&["A", "B", "C"]);
        let config = KgDetectionConfig {
            min_community_size: 1,
            max_communities: 0,
        };
        let result = d.detect(&ns, &[], &config);
        // 3 isolated nodes → 3 communities
        assert_eq!(result.len(), 3);
    }

    // --- node_community ---

    #[test]
    fn test_node_community_found() {
        let d = det();
        let ns = nodes(&["A", "B"]);
        let result = d.detect(&ns, &[], &default_config());
        let comm = d.node_community("A", &result);
        assert!(comm.is_some());
    }

    #[test]
    fn test_node_community_not_found() {
        let d = det();
        let ns = nodes(&["A"]);
        let result = d.detect(&ns, &[], &default_config());
        let comm = d.node_community("Z", &result);
        assert!(comm.is_none());
    }

    // --- intra_edges ---

    #[test]
    fn test_intra_edges_count() {
        let community = KgCommunity {
            community_id: 0,
            members: vec!["A".to_string(), "B".to_string()],
        };
        let edges = vec![
            edge("A", "B", 1.0),
            edge("A", "C", 1.0), // C not in community
        ];
        assert_eq!(KgCommunityDetector::intra_edges(&edges, &community), 1);
    }

    #[test]
    fn test_intra_edges_all_internal() {
        let community = KgCommunity {
            community_id: 0,
            members: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        };
        let edges = vec![
            edge("A", "B", 1.0),
            edge("B", "C", 1.0),
            edge("A", "C", 1.0),
        ];
        assert_eq!(KgCommunityDetector::intra_edges(&edges, &community), 3);
    }

    #[test]
    fn test_intra_edges_none() {
        let community = KgCommunity {
            community_id: 0,
            members: vec!["A".to_string()],
        };
        let edges = vec![edge("B", "C", 1.0)];
        assert_eq!(KgCommunityDetector::intra_edges(&edges, &community), 0);
    }

    // --- modularity ---

    #[test]
    fn test_modularity_empty_graph_zero() {
        let d = det();
        let q = d.modularity(&[], &[], &[]);
        assert!((q).abs() < 1e-9);
    }

    #[test]
    fn test_modularity_in_valid_range() {
        let d = det();
        let ns = nodes(&["A", "B", "C", "D"]);
        let es = vec![edge("A", "B", 1.0), edge("C", "D", 1.0)];
        let config = default_config();
        let comms = d.detect(&ns, &es, &config);
        let q = d.modularity(&ns, &es, &comms);
        // Q ∈ [-0.5, 1.0]
        assert!((-0.5..=1.0).contains(&q), "Q={q} out of range");
    }

    #[test]
    fn test_modularity_no_edges_zero() {
        let d = det();
        let ns = nodes(&["A", "B"]);
        let comms = vec![KgCommunity {
            community_id: 0,
            members: vec!["A".to_string(), "B".to_string()],
        }];
        let q = d.modularity(&ns, &[], &comms);
        assert!((q).abs() < 1e-9);
    }

    // --- KgEdge ---

    #[test]
    fn test_kg_edge_new() {
        let e = KgEdge::new("from", "to", 2.5);
        assert_eq!(e.from, "from");
        assert_eq!(e.to, "to");
        assert!((e.weight - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_kg_edge_clone() {
        let e = KgEdge::new("a", "b", 1.0);
        let c = e.clone();
        assert_eq!(c.from, "a");
    }

    // --- KgDetectionConfig default ---

    #[test]
    fn test_detection_config_default() {
        let cfg = KgDetectionConfig::default();
        assert_eq!(cfg.min_community_size, 1);
        assert_eq!(cfg.max_communities, 0);
    }

    // --- Coverage: all nodes covered ---

    #[test]
    fn test_all_nodes_in_some_community() {
        let d = det();
        let ns = nodes(&["N1", "N2", "N3", "N4", "N5"]);
        let es = vec![edge("N1", "N2", 1.0), edge("N3", "N4", 1.0)];
        let result = d.detect(&ns, &es, &default_config());
        let total: usize = result.iter().map(|c| c.members.len()).sum();
        assert_eq!(total, 5);
    }
}
