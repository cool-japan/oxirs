//! # Graph Embedder
//!
//! Node2Vec-inspired random walks and structural graph embeddings for knowledge graphs.
//!
//! Provides:
//! - Biased random walks with return parameter p and in-out parameter q
//! - Simple structural embeddings based on neighborhood aggregation
//! - Cosine similarity between node embedding vectors
//! - Graph connectivity checks (BFS) and adjacency matrix construction

use scirs2_core::random::Random;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A directed/undirected weighted edge in the graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
}

/// Simple sparse graph with node count and edge list
#[derive(Debug, Clone)]
pub struct Graph {
    pub node_count: usize,
    pub edges: Vec<GraphEdge>,
}

impl Graph {
    /// Create a new empty graph with `node_count` nodes and no edges
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            edges: Vec::new(),
        }
    }

    /// Add a directed edge from → to with the given weight
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f32) {
        self.edges.push(GraphEdge { from, to, weight });
    }

    /// Total number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// BFS from node 0 to check if all nodes are reachable (i.e. graph is connected).
    ///
    /// Treats edges as undirected for the connectivity check.
    pub fn is_connected(&self) -> bool {
        if self.node_count == 0 {
            return true;
        }
        let mut visited = vec![false; self.node_count];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0usize);
        visited[0] = true;
        let mut count = 1usize;

        while let Some(node) = queue.pop_front() {
            for edge in &self.edges {
                let neighbor = if edge.from == node {
                    Some(edge.to)
                } else if edge.to == node {
                    Some(edge.from)
                } else {
                    None
                };
                if let Some(nb) = neighbor {
                    if nb < self.node_count && !visited[nb] {
                        visited[nb] = true;
                        count += 1;
                        queue.push_back(nb);
                    }
                }
            }
        }
        count == self.node_count
    }

    /// Build a dense adjacency matrix (node_count × node_count).
    ///
    /// For undirected usage the matrix is symmetric (both from→to and to→from are set).
    pub fn adjacency_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.node_count;
        let mut mat = vec![vec![0.0f32; n]; n];
        for edge in &self.edges {
            if edge.from < n && edge.to < n {
                mat[edge.from][edge.to] = edge.weight;
                mat[edge.to][edge.from] = edge.weight; // symmetric
            }
        }
        mat
    }
}

/// Configuration for random walk generation
#[derive(Debug, Clone)]
pub struct WalkConfig {
    /// Length of each random walk (number of nodes visited)
    pub walk_length: usize,
    /// Number of walks starting from each node
    pub walks_per_node: usize,
    /// Return parameter p (controls likelihood of revisiting a node)
    pub return_param_p: f32,
    /// In-out parameter q (controls exploration vs. exploitation)
    pub in_out_param_q: f32,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            walk_length: 10,
            walks_per_node: 5,
            return_param_p: 1.0,
            in_out_param_q: 1.0,
        }
    }
}

/// Node embedding vector
#[derive(Debug, Clone)]
pub struct NodeEmbedding {
    pub node_id: usize,
    pub vector: Vec<f32>,
}

/// Result from graph embedding
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    pub embeddings: Vec<NodeEmbedding>,
    pub walk_count: usize,
}

// ─── Graph Embedder ───────────────────────────────────────────────────────────

/// Graph embedding engine: Node2Vec-inspired random walks and structural embeddings
pub struct GraphEmbedder;

impl GraphEmbedder {
    /// Generate biased random walks for all nodes.
    ///
    /// Uses Node2Vec-style second-order biased walks governed by p and q.
    /// When p=1 and q=1 the walk degenerates to a uniform random walk (DeepWalk).
    ///
    /// Returns a list of walks, each being a sequence of node IDs.
    pub fn random_walks(graph: &Graph, config: &WalkConfig) -> Vec<Vec<usize>> {
        let mut rng = Random::default();
        let mut walks = Vec::with_capacity(graph.node_count * config.walks_per_node);

        // Build adjacency list for fast neighbor lookup
        let adj = Self::build_adjacency(graph);

        for _ in 0..config.walks_per_node {
            for start in 0..graph.node_count {
                let walk = Self::single_walk(
                    &adj,
                    graph.node_count,
                    start,
                    config.walk_length,
                    config.return_param_p,
                    config.in_out_param_q,
                    &mut rng,
                );
                walks.push(walk);
            }
        }
        walks
    }

    /// Generate random-walk-based embeddings.
    ///
    /// Each node's embedding is derived from its walk co-occurrence profile
    /// (simplified: averaged node-ID features from walk context).
    pub fn embed(graph: &Graph, config: &WalkConfig, dim: usize) -> EmbeddingResult {
        let walks = Self::random_walks(graph, config);
        let walk_count = walks.len();
        let n = graph.node_count;

        // Co-occurrence accumulation: for each node, accumulate context node IDs
        let mut accum = vec![vec![0.0f64; n]; n];
        let window = 2usize; // context window half-size

        for walk in &walks {
            for (idx, &center) in walk.iter().enumerate() {
                let lo = idx.saturating_sub(window);
                let hi = (idx + window + 1).min(walk.len());
                for &ctx in &walk[lo..hi] {
                    if ctx != center {
                        accum[center][ctx] += 1.0;
                    }
                }
            }
        }

        // Project co-occurrence row into `dim`-dimensional space via hash embedding
        let embeddings: Vec<NodeEmbedding> = (0..n)
            .map(|node_id| {
                let row = &accum[node_id];
                let vector = Self::project_row(row, dim, node_id);
                NodeEmbedding { node_id, vector }
            })
            .collect();

        EmbeddingResult {
            embeddings,
            walk_count,
        }
    }

    /// Compute structural embeddings based purely on local neighborhood topology.
    ///
    /// Each node's embedding aggregates neighbour degree statistics projected into `dim`.
    pub fn structural_embedding(graph: &Graph, dim: usize) -> Vec<NodeEmbedding> {
        let n = graph.node_count;
        (0..n)
            .map(|node_id| {
                let neighbors = Self::neighbors(graph, node_id);
                // Feature: [degree, sum(neighbor_degree), sum(neighbor_weight), ...]
                let deg = neighbors.len() as f64;
                let sum_nb_deg: f64 = neighbors
                    .iter()
                    .map(|&nb| Self::degree(graph, nb) as f64)
                    .sum();
                let sum_weight: f64 = graph
                    .edges
                    .iter()
                    .filter(|e| e.from == node_id || e.to == node_id)
                    .map(|e| e.weight as f64)
                    .sum();

                let raw = vec![deg, sum_nb_deg, sum_weight, node_id as f64];
                let vector = Self::project_row(&raw, dim, node_id);
                NodeEmbedding { node_id, vector }
            })
            .collect()
    }

    /// Cosine similarity between two node embeddings, in range [-1, 1].
    pub fn node_similarity(a: &NodeEmbedding, b: &NodeEmbedding) -> f32 {
        let len = a.vector.len().min(b.vector.len());
        if len == 0 {
            return 0.0;
        }
        let dot: f32 = a.vector[..len]
            .iter()
            .zip(b.vector[..len].iter())
            .map(|(x, y)| x * y)
            .sum();
        let norm_a: f32 = a.vector[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.vector[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Return all neighbors of `node` (both directions for undirected usage).
    pub fn neighbors(graph: &Graph, node: usize) -> Vec<usize> {
        let mut nbs: Vec<usize> = graph
            .edges
            .iter()
            .filter_map(|e| {
                if e.from == node {
                    Some(e.to)
                } else if e.to == node {
                    Some(e.from)
                } else {
                    None
                }
            })
            .collect();
        nbs.sort_unstable();
        nbs.dedup();
        nbs
    }

    /// Degree of a node (number of unique neighbors).
    pub fn degree(graph: &Graph, node: usize) -> usize {
        Self::neighbors(graph, node).len()
    }

    // ─── private helpers ──────────────────────────────────────────────────────

    /// Build adjacency list: Vec<Vec<(neighbor, weight)>>
    fn build_adjacency(graph: &Graph) -> Vec<Vec<(usize, f32)>> {
        let n = graph.node_count;
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for edge in &graph.edges {
            if edge.from < n && edge.to < n {
                adj[edge.from].push((edge.to, edge.weight));
                adj[edge.to].push((edge.from, edge.weight)); // undirected
            }
        }
        adj
    }

    /// Perform a single Node2Vec walk starting from `start`.
    fn single_walk(
        adj: &[Vec<(usize, f32)>],
        _node_count: usize,
        start: usize,
        walk_length: usize,
        p: f32,
        q: f32,
        rng: &mut Random,
    ) -> Vec<usize> {
        let mut walk = Vec::with_capacity(walk_length);
        walk.push(start);

        if adj[start].is_empty() || walk_length <= 1 {
            // Isolated node — repeat self
            while walk.len() < walk_length {
                walk.push(start);
            }
            return walk;
        }

        // First step: uniform random neighbor
        let first_idx = (rng.random_range(0.0..1.0) * adj[start].len() as f64) as usize;
        walk.push(adj[start][first_idx].0);

        while walk.len() < walk_length {
            let cur = *walk.last().expect("walk is non-empty");
            let prev = walk[walk.len() - 2];

            if adj[cur].is_empty() {
                walk.push(cur); // stuck — stay
                continue;
            }

            // Compute unnormalised transition probabilities (Node2Vec bias)
            let weights: Vec<f32> = adj[cur]
                .iter()
                .map(|&(nb, w)| {
                    let bias = if nb == prev {
                        1.0 / p // return
                    } else if adj[prev].iter().any(|&(x, _)| x == nb) {
                        1.0 // common neighbor
                    } else {
                        1.0 / q // explore away
                    };
                    w * bias
                })
                .collect();

            let total: f32 = weights.iter().sum();
            let sample = (rng.random_range(0.0..1.0) as f32) * total;
            let mut cumulative = 0.0f32;
            let mut chosen = adj[cur][0].0;
            for (i, &wt) in weights.iter().enumerate() {
                cumulative += wt;
                if sample <= cumulative {
                    chosen = adj[cur][i].0;
                    break;
                }
            }
            walk.push(chosen);
        }
        walk
    }

    /// Project a raw feature slice into a `dim`-dimensional f32 vector
    /// using a simple deterministic hashing / sinusoidal expansion.
    fn project_row(row: &[f64], dim: usize, node_id: usize) -> Vec<f32> {
        use std::f64::consts::PI;
        if dim == 0 {
            return vec![];
        }

        // Compute a scalar summary of the row
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        let sum: f64 = row.iter().sum();

        let mut vec = Vec::with_capacity(dim);
        for d in 0..dim {
            // Deterministic sinusoidal projection
            let angle =
                (node_id as f64 * 0.1 + d as f64 * 1.3 + sum * 0.01) * PI / (dim as f64 + 1.0);
            let val = (angle.sin() * (norm + 1.0).ln()) as f32;
            vec.push(val);
        }
        vec
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── helpers ─────────────────────────────────────────────────────────────

    /// Triangle graph: 0-1-2-0
    fn triangle() -> Graph {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);
        g
    }

    /// Path graph: 0-1-2-3
    fn path4() -> Graph {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g
    }

    /// Disconnected graph: {0-1} and {2-3}
    fn disconnected() -> Graph {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        g
    }

    fn default_config() -> WalkConfig {
        WalkConfig {
            walk_length: 5,
            walks_per_node: 3,
            return_param_p: 1.0,
            in_out_param_q: 1.0,
        }
    }

    // ─── Graph construction ───────────────────────────────────────────────────

    #[test]
    fn test_graph_new_no_edges() {
        let g = Graph::new(5);
        assert_eq!(g.node_count, 5);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_edge_increments_count() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 2.0);
        assert_eq!(g.edge_count(), 1);
        g.add_edge(1, 2, 1.5);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_edge_stored_correctly() {
        let mut g = Graph::new(3);
        g.add_edge(0, 2, 0.7);
        let e = &g.edges[0];
        assert_eq!(e.from, 0);
        assert_eq!(e.to, 2);
        assert!((e.weight - 0.7).abs() < 1e-6);
    }

    // ─── is_connected ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_connected_triangle() {
        assert!(triangle().is_connected());
    }

    #[test]
    fn test_is_connected_path() {
        assert!(path4().is_connected());
    }

    #[test]
    fn test_is_connected_disconnected() {
        assert!(!disconnected().is_connected());
    }

    #[test]
    fn test_is_connected_single_node() {
        let g = Graph::new(1);
        assert!(g.is_connected());
    }

    #[test]
    fn test_is_connected_empty_graph() {
        let g = Graph::new(0);
        assert!(g.is_connected()); // vacuously true
    }

    // ─── neighbors and degree ─────────────────────────────────────────────────

    #[test]
    fn test_neighbors_triangle() {
        let g = triangle();
        let nb0 = GraphEmbedder::neighbors(&g, 0);
        assert!(nb0.contains(&1), "0 should be neighbor of 1");
        assert!(nb0.contains(&2), "2 should be neighbor of 0");
        assert_eq!(nb0.len(), 2);
    }

    #[test]
    fn test_neighbors_path_endpoint() {
        let g = path4();
        let nb0 = GraphEmbedder::neighbors(&g, 0);
        assert_eq!(nb0, vec![1]);
    }

    #[test]
    fn test_neighbors_isolated_node() {
        let g = Graph::new(3); // no edges
        let nb = GraphEmbedder::neighbors(&g, 1);
        assert!(nb.is_empty());
    }

    #[test]
    fn test_degree_triangle() {
        let g = triangle();
        assert_eq!(GraphEmbedder::degree(&g, 0), 2);
        assert_eq!(GraphEmbedder::degree(&g, 1), 2);
        assert_eq!(GraphEmbedder::degree(&g, 2), 2);
    }

    #[test]
    fn test_degree_path_middle() {
        let g = path4();
        assert_eq!(GraphEmbedder::degree(&g, 1), 2);
    }

    #[test]
    fn test_degree_isolated() {
        let g = Graph::new(3);
        assert_eq!(GraphEmbedder::degree(&g, 0), 0);
    }

    // ─── adjacency_matrix ─────────────────────────────────────────────────────

    #[test]
    fn test_adjacency_matrix_size() {
        let g = triangle();
        let mat = g.adjacency_matrix();
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_adjacency_matrix_symmetric() {
        let g = path4();
        let mat = g.adjacency_matrix();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (mat[i][j] - mat[j][i]).abs() < 1e-6,
                    "adjacency matrix must be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_adjacency_matrix_zero_diagonal() {
        let g = triangle();
        let mat = g.adjacency_matrix();
        for (i, row) in mat.iter().enumerate() {
            assert_eq!(row[i], 0.0, "diagonal must be zero (no self-loops)");
        }
    }

    #[test]
    fn test_adjacency_matrix_edge_weight() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 3.5);
        let mat = g.adjacency_matrix();
        assert!((mat[0][1] - 3.5).abs() < 1e-6);
        assert!((mat[1][0] - 3.5).abs() < 1e-6);
    }

    // ─── random_walks ─────────────────────────────────────────────────────────

    #[test]
    fn test_random_walks_count() {
        let g = triangle();
        let config = default_config();
        let walks = GraphEmbedder::random_walks(&g, &config);
        // walks_per_node * node_count = 3 * 3 = 9
        assert_eq!(walks.len(), 9, "expected 9 walks");
    }

    #[test]
    fn test_random_walks_length() {
        let g = triangle();
        let config = default_config();
        let walks = GraphEmbedder::random_walks(&g, &config);
        for w in &walks {
            assert_eq!(
                w.len(),
                config.walk_length,
                "each walk must have walk_length nodes"
            );
        }
    }

    #[test]
    fn test_random_walks_node_ids_valid() {
        let g = path4();
        let config = default_config();
        let walks = GraphEmbedder::random_walks(&g, &config);
        for w in &walks {
            for &node in w {
                assert!(node < g.node_count, "node id must be < node_count");
            }
        }
    }

    #[test]
    fn test_random_walks_isolated_nodes() {
        let g = Graph::new(3); // all isolated
        let config = WalkConfig {
            walk_length: 4,
            walks_per_node: 2,
            ..Default::default()
        };
        let walks = GraphEmbedder::random_walks(&g, &config);
        assert_eq!(walks.len(), 6);
        for w in &walks {
            assert_eq!(w.len(), 4);
        }
    }

    // ─── embed ────────────────────────────────────────────────────────────────

    #[test]
    fn test_embed_returns_node_count_embeddings() {
        let g = triangle();
        let config = default_config();
        let result = GraphEmbedder::embed(&g, &config, 8);
        assert_eq!(result.embeddings.len(), g.node_count);
    }

    #[test]
    fn test_embed_correct_walk_count() {
        let g = triangle();
        let config = default_config();
        let result = GraphEmbedder::embed(&g, &config, 8);
        assert_eq!(result.walk_count, config.walks_per_node * g.node_count);
    }

    #[test]
    fn test_embed_dimension() {
        let g = triangle();
        let config = default_config();
        let result = GraphEmbedder::embed(&g, &config, 16);
        for emb in &result.embeddings {
            assert_eq!(emb.vector.len(), 16, "embedding dimension must match dim");
        }
    }

    #[test]
    fn test_embed_node_ids_assigned() {
        let g = path4();
        let config = default_config();
        let result = GraphEmbedder::embed(&g, &config, 4);
        for (i, emb) in result.embeddings.iter().enumerate() {
            assert_eq!(emb.node_id, i);
        }
    }

    // ─── structural_embedding ─────────────────────────────────────────────────

    #[test]
    fn test_structural_embedding_count() {
        let g = triangle();
        let embeddings = GraphEmbedder::structural_embedding(&g, 8);
        assert_eq!(embeddings.len(), g.node_count);
    }

    #[test]
    fn test_structural_embedding_dimension() {
        let g = path4();
        let dim = 12;
        let embeddings = GraphEmbedder::structural_embedding(&g, dim);
        for emb in &embeddings {
            assert_eq!(emb.vector.len(), dim);
        }
    }

    #[test]
    fn test_structural_embedding_node_ids() {
        let g = triangle();
        let embeddings = GraphEmbedder::structural_embedding(&g, 4);
        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(emb.node_id, i);
        }
    }

    // ─── node_similarity ──────────────────────────────────────────────────────

    #[test]
    fn test_node_similarity_self_is_one() {
        let emb = NodeEmbedding {
            node_id: 0,
            vector: vec![1.0, 0.0, 0.0],
        };
        let sim = GraphEmbedder::node_similarity(&emb, &emb);
        assert!((sim - 1.0).abs() < 1e-6, "self similarity should be 1.0");
    }

    #[test]
    fn test_node_similarity_orthogonal_is_zero() {
        let a = NodeEmbedding {
            node_id: 0,
            vector: vec![1.0, 0.0],
        };
        let b = NodeEmbedding {
            node_id: 1,
            vector: vec![0.0, 1.0],
        };
        let sim = GraphEmbedder::node_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have similarity 0"
        );
    }

    #[test]
    fn test_node_similarity_range() {
        let g = path4();
        let embeddings = GraphEmbedder::structural_embedding(&g, 8);
        for a in &embeddings {
            for b in &embeddings {
                let sim = GraphEmbedder::node_similarity(a, b);
                assert!(
                    (-1.0..=1.0).contains(&sim),
                    "similarity {sim} must be in [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn test_node_similarity_empty_vectors_is_zero() {
        let a = NodeEmbedding {
            node_id: 0,
            vector: vec![],
        };
        let b = NodeEmbedding {
            node_id: 1,
            vector: vec![],
        };
        assert_eq!(GraphEmbedder::node_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_node_similarity_opposite_vectors() {
        let a = NodeEmbedding {
            node_id: 0,
            vector: vec![1.0, 0.0],
        };
        let b = NodeEmbedding {
            node_id: 1,
            vector: vec![-1.0, 0.0],
        };
        let sim = GraphEmbedder::node_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "opposite vectors: similarity = -1"
        );
    }

    // ─── edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_embed_single_node() {
        let g = Graph::new(1);
        let config = WalkConfig {
            walk_length: 3,
            walks_per_node: 2,
            ..Default::default()
        };
        let result = GraphEmbedder::embed(&g, &config, 4);
        assert_eq!(result.embeddings.len(), 1);
        assert_eq!(result.walk_count, 2);
    }

    #[test]
    fn test_structural_embedding_zero_dim() {
        let g = triangle();
        let embeddings = GraphEmbedder::structural_embedding(&g, 0);
        for emb in &embeddings {
            assert!(emb.vector.is_empty());
        }
    }

    #[test]
    fn test_walk_config_default() {
        let c = WalkConfig::default();
        assert_eq!(c.walk_length, 10);
        assert_eq!(c.walks_per_node, 5);
    }

    #[test]
    fn test_walks_total_count_formula() {
        let g = path4(); // 4 nodes
        let config = WalkConfig {
            walk_length: 6,
            walks_per_node: 4,
            ..Default::default()
        };
        let walks = GraphEmbedder::random_walks(&g, &config);
        assert_eq!(walks.len(), 4 * 4, "4 nodes * 4 walks = 16");
    }

    // ─── Additional tests (round 11 extra coverage) ───────────────────────────

    #[test]
    fn test_adjacency_matrix_path4_size() {
        let g = path4(); // 4 nodes
        let mat = g.adjacency_matrix();
        assert_eq!(mat.len(), 4);
        for row in &mat {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_adjacency_matrix_path4_symmetric() {
        let g = path4();
        let mat = g.adjacency_matrix();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (mat[i][j] - mat[j][i]).abs() < 1e-6,
                    "adjacency matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_adjacency_matrix_no_self_loops_for_path() {
        let g = path4();
        let mat = g.adjacency_matrix();
        for (i, row) in mat.iter().enumerate() {
            assert_eq!(row[i], 0.0);
        }
    }

    #[test]
    fn test_degree_path_endpoint_is_one() {
        // In path4: 0-1-2-3, node 0 and node 3 have degree 1
        let g = path4();
        assert_eq!(GraphEmbedder::degree(&g, 0), 1);
        assert_eq!(GraphEmbedder::degree(&g, 3), 1);
    }

    #[test]
    fn test_degree_path_middle_is_two() {
        // In path4: 0-1-2-3, node 1 and node 2 have degree 2
        let g = path4();
        assert_eq!(GraphEmbedder::degree(&g, 1), 2);
        assert_eq!(GraphEmbedder::degree(&g, 2), 2);
    }

    #[test]
    fn test_embed_walk_count_equals_nodes_times_walks() {
        let g = path4();
        let config = WalkConfig {
            walk_length: 5,
            walks_per_node: 3,
            ..Default::default()
        };
        let result = GraphEmbedder::embed(&g, &config, 4);
        assert_eq!(
            result.walk_count,
            4 * 3,
            "walk_count = nodes * walks_per_node"
        );
    }

    #[test]
    fn test_structural_embedding_node_ids_sequential() {
        let g = path4();
        let embeddings = GraphEmbedder::structural_embedding(&g, 6);
        let ids: Vec<usize> = embeddings.iter().map(|e| e.node_id).collect();
        let expected: Vec<usize> = (0..4).collect();
        assert_eq!(ids, expected, "node_ids must be sequential from 0");
    }
}
