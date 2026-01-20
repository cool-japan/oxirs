//! Community Detection for Knowledge Graphs
//!
//! This module provides community detection algorithms for identifying densely
//! connected groups of entities in knowledge graphs. Communities represent
//! semantic groups that can improve understanding and navigation of large graphs.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::{debug, info};

use crate::Triple;

/// Community detection algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunityAlgorithm {
    /// Louvain modularity-based algorithm
    Louvain,
    /// Label propagation algorithm
    LabelPropagation,
    /// Girvan-Newman edge betweenness
    GirvanNewman,
    /// Embedding-based communities (using embeddings similarity)
    EmbeddingBased,
}

/// Community detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityConfig {
    /// Algorithm to use
    pub algorithm: CommunityAlgorithm,
    /// Maximum iterations (for iterative algorithms)
    pub max_iterations: usize,
    /// Resolution parameter for Louvain
    pub resolution: f32,
    /// Minimum community size
    pub min_community_size: usize,
    /// Similarity threshold for embedding-based detection
    pub similarity_threshold: f32,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Louvain,
            max_iterations: 100,
            resolution: 1.0,
            min_community_size: 2,
            similarity_threshold: 0.7,
            random_seed: None,
        }
    }
}

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    /// Community assignments (entity_id -> community_id)
    pub assignments: HashMap<String, usize>,
    /// Number of communities found
    pub num_communities: usize,
    /// Community sizes
    pub community_sizes: Vec<usize>,
    /// Modularity score (quality metric)
    pub modularity: f32,
    /// Coverage (fraction of edges within communities)
    pub coverage: f32,
    /// Community members (community_id -> set of entity_ids)
    pub communities: HashMap<usize, HashSet<String>>,
}

/// Graph structure for community detection
struct Graph {
    /// Adjacency list
    edges: HashMap<String, HashSet<String>>,
    /// Edge weights (for weighted graphs)
    weights: HashMap<(String, String), f32>,
    /// Total number of edges
    num_edges: usize,
}

impl Graph {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
            weights: HashMap::new(),
            num_edges: 0,
        }
    }

    fn add_edge(&mut self, from: &str, to: &str, weight: f32) {
        self.edges
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string());

        self.edges
            .entry(to.to_string())
            .or_default()
            .insert(from.to_string());

        self.weights
            .insert((from.to_string(), to.to_string()), weight);
        self.weights
            .insert((to.to_string(), from.to_string()), weight);

        self.num_edges += 1;
    }

    fn get_neighbors(&self, node: &str) -> Option<&HashSet<String>> {
        self.edges.get(node)
    }

    fn get_weight(&self, from: &str, to: &str) -> f32 {
        self.weights
            .get(&(from.to_string(), to.to_string()))
            .copied()
            .unwrap_or(1.0)
    }

    fn degree(&self, node: &str) -> usize {
        self.edges.get(node).map(|s| s.len()).unwrap_or(0)
    }

    fn nodes(&self) -> Vec<String> {
        self.edges.keys().cloned().collect()
    }
}

/// Community detector
pub struct CommunityDetector {
    config: CommunityConfig,
    rng: Random,
}

impl CommunityDetector {
    /// Create new community detector
    pub fn new(config: CommunityConfig) -> Self {
        let rng = Random::default();

        Self { config, rng }
    }

    /// Detect communities from knowledge graph triples
    pub fn detect_from_triples(&mut self, triples: &[Triple]) -> Result<CommunityResult> {
        // Build graph from triples
        let mut graph = Graph::new();

        for triple in triples {
            // Add edge from subject to object (undirected)
            graph.add_edge(&triple.subject.to_string(), &triple.object.to_string(), 1.0);
        }

        info!(
            "Detecting communities in graph with {} nodes and {} edges using {:?}",
            graph.nodes().len(),
            graph.num_edges,
            self.config.algorithm
        );

        self.detect_from_graph(&graph)
    }

    /// Detect communities from entity embeddings
    pub fn detect_from_embeddings(
        &mut self,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<CommunityResult> {
        info!("Detecting communities from {} embeddings", embeddings.len());

        match self.config.algorithm {
            CommunityAlgorithm::EmbeddingBased => self.embedding_based_detection(embeddings),
            _ => {
                // Build similarity graph
                let graph = self.build_similarity_graph(embeddings);
                self.detect_from_graph(&graph)
            }
        }
    }

    /// Detect communities from graph structure
    fn detect_from_graph(&mut self, graph: &Graph) -> Result<CommunityResult> {
        match self.config.algorithm {
            CommunityAlgorithm::Louvain => self.louvain_detection(graph),
            CommunityAlgorithm::LabelPropagation => self.label_propagation(graph),
            CommunityAlgorithm::GirvanNewman => self.girvan_newman(graph),
            CommunityAlgorithm::EmbeddingBased => {
                Err(anyhow!("Embedding-based detection requires embeddings"))
            }
        }
    }

    /// Louvain modularity optimization
    fn louvain_detection(&mut self, graph: &Graph) -> Result<CommunityResult> {
        let nodes = graph.nodes();
        let m = graph.num_edges as f32;

        // Initialize: each node in its own community
        let mut community: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.clone(), i))
            .collect();

        let mut improved = true;
        let mut iteration = 0;

        while improved && iteration < self.config.max_iterations {
            improved = false;
            iteration += 1;

            // For each node, try moving to neighbor's community
            for node in &nodes {
                let current_comm = community[node];
                let best_comm = self.find_best_community(node, current_comm, &community, graph, m);

                if best_comm != current_comm {
                    community.insert(node.clone(), best_comm);
                    improved = true;
                }
            }

            debug!("Louvain iteration {}: improved = {}", iteration, improved);
        }

        self.create_result(&community, graph)
    }

    /// Find best community for a node (max modularity gain)
    fn find_best_community(
        &self,
        node: &str,
        current_comm: usize,
        community: &HashMap<String, usize>,
        graph: &Graph,
        m: f32,
    ) -> usize {
        let neighbors = match graph.get_neighbors(node) {
            Some(n) => n,
            None => return current_comm,
        };

        // Get neighboring communities
        let mut neighbor_comms: HashSet<usize> = HashSet::new();
        for neighbor in neighbors {
            if let Some(&comm) = community.get(neighbor) {
                neighbor_comms.insert(comm);
            }
        }

        // Compute modularity gain for each community
        let current_modularity =
            self.compute_modularity_contribution(node, current_comm, community, graph, m);

        let mut best_comm = current_comm;
        let mut best_modularity = current_modularity;

        for &comm in &neighbor_comms {
            if comm == current_comm {
                continue;
            }

            let modularity = self.compute_modularity_contribution(node, comm, community, graph, m);

            if modularity > best_modularity {
                best_modularity = modularity;
                best_comm = comm;
            }
        }

        best_comm
    }

    /// Compute modularity contribution for a node in a community
    fn compute_modularity_contribution(
        &self,
        node: &str,
        comm: usize,
        community: &HashMap<String, usize>,
        graph: &Graph,
        m: f32,
    ) -> f32 {
        let neighbors = match graph.get_neighbors(node) {
            Some(n) => n,
            None => return 0.0,
        };

        let k_i = graph.degree(node) as f32;

        // Sum of weights to nodes in community
        let mut e_ic = 0.0;
        let mut k_c = 0.0;

        for neighbor in neighbors {
            if let Some(&neighbor_comm) = community.get(neighbor) {
                if neighbor_comm == comm {
                    e_ic += graph.get_weight(node, neighbor);
                    k_c += graph.degree(neighbor) as f32;
                }
            }
        }

        (e_ic - (self.config.resolution * k_i * k_c) / (2.0 * m)) / m
    }

    /// Label propagation algorithm
    fn label_propagation(&mut self, graph: &Graph) -> Result<CommunityResult> {
        let nodes = graph.nodes();

        // Initialize: each node with unique label
        let mut labels: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.clone(), i))
            .collect();

        for iteration in 0..self.config.max_iterations {
            let mut changed = false;

            // Randomize node order
            let mut node_order = nodes.clone();
            for i in (1..node_order.len()).rev() {
                let j = self.rng.random_range(0..i + 1);
                node_order.swap(i, j);
            }

            // Update labels
            for node in &node_order {
                let old_label = labels[node];
                let new_label = self.majority_label(node, &labels, graph);

                if new_label != old_label {
                    labels.insert(node.clone(), new_label);
                    changed = true;
                }
            }

            debug!(
                "Label propagation iteration {}: changed = {}",
                iteration + 1,
                changed
            );

            if !changed {
                info!("Label propagation converged at iteration {}", iteration + 1);
                break;
            }
        }

        self.create_result(&labels, graph)
    }

    /// Get majority label from neighbors
    fn majority_label(&self, node: &str, labels: &HashMap<String, usize>, graph: &Graph) -> usize {
        let neighbors = match graph.get_neighbors(node) {
            Some(n) => n,
            None => return labels[node],
        };

        let mut label_counts: HashMap<usize, usize> = HashMap::new();

        for neighbor in neighbors {
            if let Some(&label) = labels.get(neighbor) {
                *label_counts.entry(label).or_insert(0) += 1;
            }
        }

        // Return most frequent label
        label_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| label)
            .unwrap_or_else(|| labels[node])
    }

    /// Girvan-Newman edge betweenness clustering
    fn girvan_newman(&mut self, graph: &Graph) -> Result<CommunityResult> {
        // Simplified implementation: iteratively remove highest betweenness edges
        // Full implementation would require connected components analysis

        let nodes = graph.nodes();
        let mut assignments: HashMap<String, usize> = HashMap::new();

        // Use BFS to identify connected components
        let mut visited = HashSet::new();
        let mut community_id = 0;

        for node in &nodes {
            if visited.contains(node) {
                continue;
            }

            // BFS from this node
            let component = self.bfs_component(node, graph, &visited);

            for comp_node in &component {
                assignments.insert(comp_node.clone(), community_id);
                visited.insert(comp_node.clone());
            }

            community_id += 1;
        }

        self.create_result(&assignments, graph)
    }

    /// BFS to find connected component
    fn bfs_component(
        &self,
        start: &str,
        graph: &Graph,
        visited: &HashSet<String>,
    ) -> HashSet<String> {
        let mut component = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());
        component.insert(start.to_string());

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = graph.get_neighbors(&node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) && !component.contains(neighbor) {
                        component.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        component
    }

    /// Embedding-based community detection
    fn embedding_based_detection(
        &mut self,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<CommunityResult> {
        let entity_list: Vec<String> = embeddings.keys().cloned().collect();
        let mut assignments: HashMap<String, usize> = HashMap::new();
        let mut community_id = 0;

        let mut unassigned: HashSet<String> = entity_list.iter().cloned().collect();

        while !unassigned.is_empty() {
            // Pick random seed
            let seed = unassigned.iter().next().unwrap().clone();
            let mut community = HashSet::new();
            community.insert(seed.clone());
            unassigned.remove(&seed);

            // Grow community by similarity
            let mut changed = true;
            while changed {
                changed = false;

                for entity in &entity_list {
                    if community.contains(entity) || !unassigned.contains(entity) {
                        continue;
                    }

                    // Check similarity to community members
                    let avg_similarity =
                        self.average_similarity_to_community(entity, &community, embeddings);

                    if avg_similarity >= self.config.similarity_threshold {
                        community.insert(entity.clone());
                        unassigned.remove(entity);
                        changed = true;
                    }
                }
            }

            // Assign community
            if community.len() >= self.config.min_community_size {
                for member in community {
                    assignments.insert(member, community_id);
                }
                community_id += 1;
            } else {
                // Assign to noise/outlier community
                for member in community {
                    assignments.insert(member, usize::MAX);
                }
            }
        }

        // Build dummy graph for result creation
        let mut graph = Graph::new();
        for entity in &entity_list {
            graph.edges.insert(entity.clone(), HashSet::new());
        }

        self.create_result(&assignments, &graph)
    }

    /// Compute average similarity to community members
    fn average_similarity_to_community(
        &self,
        entity: &str,
        community: &HashSet<String>,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> f32 {
        if community.is_empty() {
            return 0.0;
        }

        let entity_emb = &embeddings[entity];

        let total_sim: f32 = community
            .iter()
            .map(|member| {
                let member_emb = &embeddings[member];
                self.cosine_similarity(entity_emb, member_emb)
            })
            .sum();

        total_sim / community.len() as f32
    }

    /// Cosine similarity
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Build similarity graph from embeddings
    fn build_similarity_graph(&self, embeddings: &HashMap<String, Array1<f32>>) -> Graph {
        let mut graph = Graph::new();
        let entity_list: Vec<String> = embeddings.keys().cloned().collect();

        for i in 0..entity_list.len() {
            for j in (i + 1)..entity_list.len() {
                let sim = self
                    .cosine_similarity(&embeddings[&entity_list[i]], &embeddings[&entity_list[j]]);

                if sim >= self.config.similarity_threshold {
                    graph.add_edge(&entity_list[i], &entity_list[j], sim);
                }
            }
        }

        graph
    }

    /// Create community result from assignments
    fn create_result(
        &self,
        assignments: &HashMap<String, usize>,
        graph: &Graph,
    ) -> Result<CommunityResult> {
        // Compute community sizes
        let mut community_sizes: HashMap<usize, usize> = HashMap::new();
        let mut communities: HashMap<usize, HashSet<String>> = HashMap::new();

        for (entity, &comm) in assignments {
            if comm != usize::MAX {
                *community_sizes.entry(comm).or_insert(0) += 1;
                communities.entry(comm).or_default().insert(entity.clone());
            }
        }

        let num_communities = community_sizes.len();
        let sizes: Vec<usize> = (0..num_communities)
            .map(|i| community_sizes.get(&i).copied().unwrap_or(0))
            .collect();

        // Compute modularity
        let modularity = self.compute_modularity(assignments, graph);

        // Compute coverage
        let coverage = self.compute_coverage(assignments, graph);

        Ok(CommunityResult {
            assignments: assignments.clone(),
            num_communities,
            community_sizes: sizes,
            modularity,
            coverage,
            communities,
        })
    }

    /// Compute overall modularity
    fn compute_modularity(&self, assignments: &HashMap<String, usize>, graph: &Graph) -> f32 {
        let m = graph.num_edges as f32;
        if m == 0.0 {
            return 0.0;
        }

        let nodes = graph.nodes();
        let mut modularity = 0.0;

        for node_i in &nodes {
            for node_j in &nodes {
                if let (Some(&comm_i), Some(&comm_j)) =
                    (assignments.get(node_i), assignments.get(node_j))
                {
                    if comm_i == comm_j && comm_i != usize::MAX {
                        let a_ij = if graph
                            .get_neighbors(node_i)
                            .map(|n| n.contains(node_j))
                            .unwrap_or(false)
                        {
                            1.0
                        } else {
                            0.0
                        };

                        let k_i = graph.degree(node_i) as f32;
                        let k_j = graph.degree(node_j) as f32;

                        modularity += a_ij - (k_i * k_j) / (2.0 * m);
                    }
                }
            }
        }

        modularity / (2.0 * m)
    }

    /// Compute coverage (fraction of edges within communities)
    fn compute_coverage(&self, assignments: &HashMap<String, usize>, graph: &Graph) -> f32 {
        if graph.num_edges == 0 {
            return 0.0;
        }

        let mut internal_edges = 0;

        for (node, neighbors) in &graph.edges {
            if let Some(&comm) = assignments.get(node) {
                if comm == usize::MAX {
                    continue;
                }

                for neighbor in neighbors {
                    if let Some(&neighbor_comm) = assignments.get(neighbor) {
                        if comm == neighbor_comm {
                            internal_edges += 1;
                        }
                    }
                }
            }
        }

        // Each edge is counted twice
        (internal_edges / 2) as f32 / graph.num_edges as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_community_detection_from_triples() {
        let triples = vec![
            Triple::new(
                NamedNode::new("a").unwrap(),
                NamedNode::new("r").unwrap(),
                NamedNode::new("b").unwrap(),
            ),
            Triple::new(
                NamedNode::new("b").unwrap(),
                NamedNode::new("r").unwrap(),
                NamedNode::new("c").unwrap(),
            ),
            Triple::new(
                NamedNode::new("d").unwrap(),
                NamedNode::new("r").unwrap(),
                NamedNode::new("e").unwrap(),
            ),
        ];

        let config = CommunityConfig::default();
        let mut detector = CommunityDetector::new(config);
        let result = detector.detect_from_triples(&triples).unwrap();

        assert!(result.num_communities > 0);
        assert_eq!(result.assignments.len(), 5); // a, b, c, d, e
    }

    #[test]
    fn test_embedding_based_detection() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 0.0]);
        embeddings.insert("e2".to_string(), array![0.9, 0.1]);
        embeddings.insert("e3".to_string(), array![0.0, 1.0]);
        embeddings.insert("e4".to_string(), array![0.1, 0.9]);

        let config = CommunityConfig {
            algorithm: CommunityAlgorithm::EmbeddingBased,
            similarity_threshold: 0.8,
            ..Default::default()
        };

        let mut detector = CommunityDetector::new(config);
        let result = detector.detect_from_embeddings(&embeddings).unwrap();

        assert!(result.num_communities >= 1);
        // Similar embeddings should be in same community
        assert_eq!(result.assignments.get("e1"), result.assignments.get("e2"));
    }
}
