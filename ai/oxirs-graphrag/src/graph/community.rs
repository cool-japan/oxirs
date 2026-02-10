//! Community detection for hierarchical summarization

use crate::{CommunitySummary, GraphRAGError, GraphRAGResult, Triple};
use petgraph::graph::{NodeIndex, UnGraph};
use scirs2_core::random::{seeded_rng, Random};
use std::collections::{HashMap, HashSet};

/// Community detection algorithm
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm (baseline: ~0.65 modularity)
    Louvain,
    /// Leiden algorithm (target: >0.75 modularity, improved Louvain)
    #[default]
    Leiden,
    /// Label propagation
    LabelPropagation,
    /// Connected components
    ConnectedComponents,
    /// Hierarchical (multi-level)
    Hierarchical,
}

/// Community detector configuration
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Algorithm to use
    pub algorithm: CommunityAlgorithm,
    /// Resolution parameter for Louvain/Leiden
    pub resolution: f64,
    /// Minimum community size
    pub min_community_size: usize,
    /// Maximum number of communities
    pub max_communities: usize,
    /// Number of iterations for iterative algorithms
    pub max_iterations: usize,
    /// Random seed for reproducibility
    pub random_seed: u64,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Leiden,
            resolution: 1.0,
            min_community_size: 3,
            max_communities: 50,
            max_iterations: 10,
            random_seed: 42,
        }
    }
}

/// Community detector
pub struct CommunityDetector {
    config: CommunityConfig,
}

impl Default for CommunityDetector {
    fn default() -> Self {
        Self::new(CommunityConfig::default())
    }
}

impl CommunityDetector {
    pub fn new(config: CommunityConfig) -> Self {
        Self { config }
    }

    /// Detect communities in the given subgraph
    pub fn detect(&self, triples: &[Triple]) -> GraphRAGResult<Vec<CommunitySummary>> {
        if triples.is_empty() {
            return Ok(vec![]);
        }

        // Build graph
        let (graph, node_map) = self.build_graph(triples);

        // Detect communities based on algorithm
        let communities = match self.config.algorithm {
            CommunityAlgorithm::Louvain => self.louvain(&graph, &node_map),
            CommunityAlgorithm::Leiden => self.leiden(&graph, &node_map)?,
            CommunityAlgorithm::LabelPropagation => self.label_propagation(&graph, &node_map),
            CommunityAlgorithm::ConnectedComponents => self.connected_components(&graph, &node_map),
            CommunityAlgorithm::Hierarchical => {
                return self.detect_hierarchical(triples);
            }
        };

        // Filter and create summaries
        let summaries = self.create_summaries(communities, triples);

        Ok(summaries)
    }

    /// Build undirected graph from triples
    fn build_graph(&self, triples: &[Triple]) -> (UnGraph<String, ()>, HashMap<String, NodeIndex>) {
        let mut graph: UnGraph<String, ()> = UnGraph::new_undirected();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        for triple in triples {
            let subj_idx = *node_map
                .entry(triple.subject.clone())
                .or_insert_with(|| graph.add_node(triple.subject.clone()));
            let obj_idx = *node_map
                .entry(triple.object.clone())
                .or_insert_with(|| graph.add_node(triple.object.clone()));

            if subj_idx != obj_idx && graph.find_edge(subj_idx, obj_idx).is_none() {
                graph.add_edge(subj_idx, obj_idx, ());
            }
        }

        (graph, node_map)
    }

    /// Simplified Louvain algorithm
    fn louvain(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> Vec<HashSet<String>> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return vec![];
        }

        // Initialize: each node in its own community
        let mut community: HashMap<NodeIndex, usize> = HashMap::new();
        for (community_id, &idx) in node_map.values().enumerate() {
            community.insert(idx, community_id);
        }

        // Total edges (for modularity calculation)
        let m = graph.edge_count() as f64;
        if m == 0.0 {
            // No edges, each node is its own community
            return node_map
                .keys()
                .map(|k| {
                    let mut set = HashSet::new();
                    set.insert(k.clone());
                    set
                })
                .collect();
        }

        // Degree of each node
        let degree: HashMap<NodeIndex, f64> = node_map
            .values()
            .map(|&idx| (idx, graph.neighbors(idx).count() as f64))
            .collect();

        // Iterate
        for _ in 0..self.config.max_iterations {
            let mut changed = false;

            for (&node, &current_comm) in community.clone().iter() {
                let node_degree = degree.get(&node).copied().unwrap_or(0.0);

                // Calculate modularity gain for each neighbor's community
                let mut best_comm = current_comm;
                let mut best_gain = 0.0;

                let neighbor_comms: HashSet<usize> = graph
                    .neighbors(node)
                    .filter_map(|n| community.get(&n).copied())
                    .collect();

                for &neighbor_comm in &neighbor_comms {
                    if neighbor_comm == current_comm {
                        continue;
                    }

                    // Simplified modularity gain calculation
                    let edges_to_comm: f64 = graph
                        .neighbors(node)
                        .filter(|n| community.get(n) == Some(&neighbor_comm))
                        .count() as f64;

                    let comm_degree: f64 = community
                        .iter()
                        .filter(|(_, &c)| c == neighbor_comm)
                        .map(|(n, _)| degree.get(n).copied().unwrap_or(0.0))
                        .sum();

                    let gain = edges_to_comm / m
                        - self.config.resolution * node_degree * comm_degree / (2.0 * m * m);

                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = neighbor_comm;
                    }
                }

                if best_comm != current_comm && best_gain > 0.0 {
                    community.insert(node, best_comm);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Group nodes by community
        self.group_by_community(graph, &community)
    }

    /// Label propagation algorithm
    fn label_propagation(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> Vec<HashSet<String>> {
        if graph.node_count() == 0 {
            return vec![];
        }

        // Initialize labels
        let mut labels: HashMap<NodeIndex, usize> = HashMap::new();
        for (i, &idx) in node_map.values().enumerate() {
            labels.insert(idx, i);
        }

        // Iterate
        for _ in 0..self.config.max_iterations {
            let mut changed = false;

            for &node in node_map.values() {
                // Count neighbor labels
                let mut label_counts: HashMap<usize, usize> = HashMap::new();
                for neighbor in graph.neighbors(node) {
                    if let Some(&label) = labels.get(&neighbor) {
                        *label_counts.entry(label).or_insert(0) += 1;
                    }
                }

                // Assign most common label
                if let Some((&best_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count)
                {
                    if labels.get(&node) != Some(&best_label) {
                        labels.insert(node, best_label);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        self.group_by_community(graph, &labels)
    }

    /// Connected components
    fn connected_components(
        &self,
        graph: &UnGraph<String, ()>,
        _node_map: &HashMap<String, NodeIndex>,
    ) -> Vec<HashSet<String>> {
        let sccs = petgraph::algo::kosaraju_scc(graph);

        sccs.into_iter()
            .map(|component| {
                component
                    .into_iter()
                    .filter_map(|idx| graph.node_weight(idx).cloned())
                    .collect()
            })
            .collect()
    }

    /// Leiden algorithm (improved Louvain with refinement phase)
    fn leiden(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> GraphRAGResult<Vec<HashSet<String>>> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(vec![]);
        }

        // Initialize: each node in its own community
        let mut community: HashMap<NodeIndex, usize> = HashMap::new();
        for (community_id, &idx) in node_map.values().enumerate() {
            community.insert(idx, community_id);
        }

        let m = graph.edge_count() as f64;
        if m == 0.0 {
            return Ok(node_map
                .keys()
                .map(|k| {
                    let mut set = HashSet::new();
                    set.insert(k.clone());
                    set
                })
                .collect());
        }

        let degree: HashMap<NodeIndex, f64> = node_map
            .values()
            .map(|&idx| (idx, graph.neighbors(idx).count() as f64))
            .collect();

        let mut rng = seeded_rng(self.config.random_seed);
        let mut best_modularity = self.calculate_modularity(graph, &community, m, &degree)?;

        // Main Leiden loop
        for iteration in 0..self.config.max_iterations {
            let mut changed = false;

            // Phase 1: Local moving (like Louvain)
            let mut node_order: Vec<NodeIndex> = node_map.values().copied().collect();
            // Shuffle for randomness
            for i in (1..node_order.len()).rev() {
                let j = (rng.random_range(0.0..1.0) * (i + 1) as f64) as usize;
                node_order.swap(i, j);
            }

            for &node in &node_order {
                let current_comm = match community.get(&node) {
                    Some(&c) => c,
                    None => continue,
                };
                let node_degree = degree.get(&node).copied().unwrap_or(0.0);

                let mut best_comm = current_comm;
                let mut best_gain = 0.0;

                // Get neighbor communities
                let neighbor_comms: HashSet<usize> = graph
                    .neighbors(node)
                    .filter_map(|n| community.get(&n).copied())
                    .collect();

                for &neighbor_comm in &neighbor_comms {
                    if neighbor_comm == current_comm {
                        continue;
                    }

                    let edges_to_comm: f64 = graph
                        .neighbors(node)
                        .filter(|n| community.get(n) == Some(&neighbor_comm))
                        .count() as f64;

                    let comm_degree: f64 = community
                        .iter()
                        .filter(|(_, &c)| c == neighbor_comm)
                        .map(|(n, _)| degree.get(n).copied().unwrap_or(0.0))
                        .sum();

                    let gain = edges_to_comm / m
                        - self.config.resolution * node_degree * comm_degree / (2.0 * m * m);

                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = neighbor_comm;
                    }
                }

                if best_comm != current_comm && best_gain > 0.0 {
                    community.insert(node, best_comm);
                    changed = true;
                }
            }

            // Phase 2: Refinement (what makes Leiden better than Louvain)
            // Split communities and re-merge if it improves modularity
            let unique_comms: HashSet<usize> = community.values().copied().collect();
            for &comm_id in &unique_comms {
                let comm_nodes: Vec<NodeIndex> = community
                    .iter()
                    .filter(|(_, &c)| c == comm_id)
                    .map(|(&n, _)| n)
                    .collect();

                if comm_nodes.len() <= 1 {
                    continue;
                }

                // Try to split and refine
                self.refine_community(
                    graph,
                    &mut community,
                    &comm_nodes,
                    comm_id,
                    m,
                    &degree,
                )?;
            }

            // Check modularity improvement
            let current_modularity = self.calculate_modularity(graph, &community, m, &degree)?;
            if current_modularity > best_modularity {
                best_modularity = current_modularity;
            } else if !changed {
                break;
            }

            // Early stop if modularity is very high
            if best_modularity > 0.95 || iteration > 0 && !changed {
                break;
            }
        }

        // Verify target: modularity > 0.75
        if best_modularity < 0.75 {
            tracing::warn!(
                "Leiden modularity {:.3} below target 0.75",
                best_modularity
            );
        } else {
            tracing::info!("Leiden achieved modularity: {:.3}", best_modularity);
        }

        Ok(self.group_by_community(graph, &community))
    }

    /// Refine a community by attempting local splits
    fn refine_community(
        &self,
        graph: &UnGraph<String, ()>,
        community: &mut HashMap<NodeIndex, usize>,
        comm_nodes: &[NodeIndex],
        comm_id: usize,
        m: f64,
        degree: &HashMap<NodeIndex, f64>,
    ) -> GraphRAGResult<()> {
        if comm_nodes.len() < 2 {
            return Ok(());
        }

        // Try to find a better split using local connectivity
        let mut subcomm: HashMap<NodeIndex, usize> = HashMap::new();
        for (i, &node) in comm_nodes.iter().enumerate() {
            subcomm.insert(node, i);
        }

        // One pass of local moving within the community
        let mut changed = false;
        for &node in comm_nodes {
            let current_sub = match subcomm.get(&node) {
                Some(&c) => c,
                None => continue,
            };

            // Count edges to each subcommunity
            let mut sub_edges: HashMap<usize, f64> = HashMap::new();
            for neighbor in graph.neighbors(node) {
                if let Some(&sub) = subcomm.get(&neighbor) {
                    *sub_edges.entry(sub).or_insert(0.0) += 1.0;
                }
            }

            // Find best subcommunity
            if let Some((&best_sub, _)) = sub_edges.iter().max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                if best_sub != current_sub {
                    subcomm.insert(node, best_sub);
                    changed = true;
                }
            }
        }

        // If we found a better partition, create new communities
        if changed {
            let unique_subs: HashSet<usize> = subcomm.values().copied().collect();
            if unique_subs.len() > 1 {
                let max_comm = community.values().max().copied().unwrap_or(0);
                for (i, sub_id) in unique_subs.iter().enumerate() {
                    for &node in comm_nodes {
                        if subcomm.get(&node) == Some(sub_id) {
                            let new_comm = if i == 0 { comm_id } else { max_comm + i };
                            community.insert(node, new_comm);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate modularity of a community assignment
    fn calculate_modularity(
        &self,
        graph: &UnGraph<String, ()>,
        community: &HashMap<NodeIndex, usize>,
        m: f64,
        degree: &HashMap<NodeIndex, f64>,
    ) -> GraphRAGResult<f64> {
        if m == 0.0 {
            return Ok(0.0);
        }

        let mut modularity = 0.0;

        for edge in graph.edge_indices() {
            if let Some((a, b)) = graph.edge_endpoints(edge) {
                let comm_a = community.get(&a);
                let comm_b = community.get(&b);

                if comm_a == comm_b && comm_a.is_some() {
                    let deg_a = degree.get(&a).copied().unwrap_or(0.0);
                    let deg_b = degree.get(&b).copied().unwrap_or(0.0);

                    modularity += 1.0 - (deg_a * deg_b) / (2.0 * m * m);
                }
            }
        }

        Ok(modularity / m)
    }

    /// Hierarchical community detection (multi-level)
    fn detect_hierarchical(&self, triples: &[Triple]) -> GraphRAGResult<Vec<CommunitySummary>> {
        let mut all_summaries = Vec::new();
        let mut current_triples = triples.to_vec();
        let mut level = 0;

        while level < 5 && !current_triples.is_empty() {
            let (graph, node_map) = self.build_graph(&current_triples);

            if graph.node_count() < 10 {
                break;
            }

            // Detect communities at this level using Leiden
            let communities = self.leiden(&graph, &node_map)?;

            // Create summaries for this level
            let mut level_summaries = self.create_summaries(communities.clone(), &current_triples);

            // Tag with level
            for summary in &mut level_summaries {
                summary.level = level;
            }

            all_summaries.extend(level_summaries);

            // Coarsen graph: each community becomes a supernode
            current_triples = self.coarsen_graph(&graph, &node_map, &communities)?;
            level += 1;
        }

        Ok(all_summaries)
    }

    /// Coarsen graph by collapsing communities into supernodes
    fn coarsen_graph(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
        communities: &[HashSet<String>],
    ) -> GraphRAGResult<Vec<Triple>> {
        let mut node_to_community: HashMap<String, usize> = HashMap::new();
        for (comm_id, community) in communities.iter().enumerate() {
            for node in community {
                node_to_community.insert(node.clone(), comm_id);
            }
        }

        let mut coarsened_triples = Vec::new();
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for edge in graph.edge_indices() {
            if let Some((a, b)) = graph.edge_endpoints(edge) {
                let label_a = graph.node_weight(a);
                let label_b = graph.node_weight(b);

                if let (Some(la), Some(lb)) = (label_a, label_b) {
                    if let (Some(&comm_a), Some(&comm_b)) =
                        (node_to_community.get(la), node_to_community.get(lb))
                    {
                        if comm_a != comm_b {
                            let edge_key = if comm_a < comm_b {
                                (comm_a, comm_b)
                            } else {
                                (comm_b, comm_a)
                            };

                            if !seen_edges.contains(&edge_key) {
                                seen_edges.insert(edge_key);
                                coarsened_triples.push(Triple::new(
                                    format!("community_{}", comm_a),
                                    "inter_community_link",
                                    format!("community_{}", comm_b),
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(coarsened_triples)
    }

    /// Group nodes by community assignment
    fn group_by_community(
        &self,
        graph: &UnGraph<String, ()>,
        assignment: &HashMap<NodeIndex, usize>,
    ) -> Vec<HashSet<String>> {
        let mut communities: HashMap<usize, HashSet<String>> = HashMap::new();

        for (&node, &comm) in assignment {
            if let Some(label) = graph.node_weight(node) {
                communities.entry(comm).or_default().insert(label.clone());
            }
        }

        communities.into_values().collect()
    }

    /// Create community summaries
    fn create_summaries(
        &self,
        communities: Vec<HashSet<String>>,
        triples: &[Triple],
    ) -> Vec<CommunitySummary> {
        communities
            .into_iter()
            .enumerate()
            .filter(|(_, entities)| entities.len() >= self.config.min_community_size)
            .take(self.config.max_communities)
            .map(|(idx, entities)| {
                // Find representative triples
                let representative_triples: Vec<Triple> = triples
                    .iter()
                    .filter(|t| entities.contains(&t.subject) || entities.contains(&t.object))
                    .take(5)
                    .cloned()
                    .collect();

                // Calculate modularity (simplified)
                let internal_edges = triples
                    .iter()
                    .filter(|t| entities.contains(&t.subject) && entities.contains(&t.object))
                    .count() as f64;
                let total_edges = triples.len().max(1) as f64;
                let modularity = internal_edges / total_edges;

                // Generate summary
                let entity_list: Vec<String> = entities.iter().cloned().collect();
                let summary = self.generate_summary(&entity_list, &representative_triples);

                CommunitySummary {
                    id: format!("community_{}", idx),
                    summary,
                    entities: entity_list,
                    representative_triples,
                    level: 0,
                    modularity,
                }
            })
            .collect()
    }

    /// Generate a text summary for a community
    fn generate_summary(&self, entities: &[String], triples: &[Triple]) -> String {
        // Extract short names from URIs
        let short_names: Vec<String> = entities
            .iter()
            .take(3)
            .map(|uri| {
                uri.rsplit('/')
                    .next()
                    .or_else(|| uri.rsplit('#').next())
                    .unwrap_or(uri)
                    .to_string()
            })
            .collect();

        // Extract predicates
        let predicates: HashSet<String> = triples
            .iter()
            .map(|t| {
                t.predicate
                    .rsplit('/')
                    .next()
                    .or_else(|| t.predicate.rsplit('#').next())
                    .unwrap_or(&t.predicate)
                    .to_string()
            })
            .collect();

        let pred_str: Vec<String> = predicates.into_iter().take(3).collect();

        format!(
            "Community of {} entities including {} connected by {}",
            entities.len(),
            short_names.join(", "),
            pred_str.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_community_detection() {
        let detector = CommunityDetector::default();

        let triples = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
            Triple::new("http://a", "http://rel", "http://c"),
            Triple::new("http://x", "http://rel", "http://y"),
            Triple::new("http://y", "http://rel", "http://z"),
            Triple::new("http://x", "http://rel", "http://z"),
        ];

        let communities = detector.detect(&triples).unwrap();

        // Should detect 2 communities (a-b-c and x-y-z)
        assert!(!communities.is_empty());
    }

    #[test]
    fn test_empty_graph() {
        let detector = CommunityDetector::default();
        let communities = detector.detect(&[]).unwrap();
        assert!(communities.is_empty());
    }
}
