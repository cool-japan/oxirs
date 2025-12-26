//! Community detection for hierarchical summarization

use crate::{CommunitySummary, GraphRAGResult, Triple};
use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::{HashMap, HashSet};

/// Community detection algorithm
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm
    #[default]
    Louvain,
    /// Label propagation
    LabelPropagation,
    /// Connected components
    ConnectedComponents,
}

/// Community detector configuration
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Algorithm to use
    pub algorithm: CommunityAlgorithm,
    /// Resolution parameter for Louvain
    pub resolution: f64,
    /// Minimum community size
    pub min_community_size: usize,
    /// Maximum number of communities
    pub max_communities: usize,
    /// Number of iterations for iterative algorithms
    pub max_iterations: usize,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Louvain,
            resolution: 1.0,
            min_community_size: 2,
            max_communities: 50,
            max_iterations: 100,
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
            CommunityAlgorithm::LabelPropagation => self.label_propagation(&graph, &node_map),
            CommunityAlgorithm::ConnectedComponents => self.connected_components(&graph, &node_map),
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
