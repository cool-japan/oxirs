//! Community-aware graph embeddings (GraphSAGE, Node2Vec)

use crate::{GraphRAGError, GraphRAGResult, Triple};
use petgraph::graph::{NodeIndex, UnGraph};
use scirs2_core::random::{rand_prelude::StdRng, seeded_rng, CoreRandom};
use std::collections::{HashMap, HashSet};

/// Community structure for embeddings
#[derive(Debug, Clone)]
pub struct CommunityStructure {
    /// Node to community mapping
    pub node_to_community: HashMap<String, usize>,
    /// Community to nodes mapping
    pub community_to_nodes: HashMap<usize, HashSet<String>>,
    /// Modularity score
    pub modularity: f64,
}

impl CommunityStructure {
    /// Create from community assignments
    pub fn from_assignments(assignments: &[(String, usize)], modularity: f64) -> Self {
        let mut node_to_community = HashMap::new();
        let mut community_to_nodes: HashMap<usize, HashSet<String>> = HashMap::new();

        for (node, comm) in assignments {
            node_to_community.insert(node.clone(), *comm);
            community_to_nodes
                .entry(*comm)
                .or_default()
                .insert(node.clone());
        }

        Self {
            node_to_community,
            community_to_nodes,
            modularity,
        }
    }
}

/// Configuration for community-aware embeddings
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Embedding dimension (default: 128)
    pub embedding_dim: usize,
    /// Walk length for Node2Vec (default: 80)
    pub walk_length: usize,
    /// Number of walks per node (default: 10)
    pub num_walks: usize,
    /// Return parameter p for Node2Vec (default: 1.0)
    pub p: f64,
    /// In-out parameter q for Node2Vec (default: 1.0)
    pub q: f64,
    /// Community bias for random walks (default: 2.0, higher = prefer same community)
    pub community_bias: f64,
    /// Window size for skip-gram (default: 5)
    pub window_size: usize,
    /// Random seed
    pub random_seed: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            walk_length: 80,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            community_bias: 2.0,
            window_size: 5,
            random_seed: 42,
        }
    }
}

/// Community-aware graph embeddings
pub struct CommunityAwareEmbeddings {
    config: EmbeddingConfig,
    rng: CoreRandom<StdRng>,
}

impl CommunityAwareEmbeddings {
    /// Create new embeddings generator
    pub fn new(config: EmbeddingConfig) -> Self {
        let rng = seeded_rng(config.random_seed);
        Self { config, rng }
    }

    /// Generate embeddings using GraphSAGE with community awareness
    pub fn embed_graphsage(
        &mut self,
        triples: &[Triple],
        communities: &CommunityStructure,
    ) -> GraphRAGResult<HashMap<String, Vec<f64>>> {
        let (graph, node_map) = self.build_graph(triples);

        if graph.node_count() == 0 {
            return Ok(HashMap::new());
        }

        let mut embeddings: HashMap<String, Vec<f64>> = HashMap::new();

        // Initialize with random features
        for (node_label, &node_idx) in &node_map {
            let mut features = vec![0.0; self.config.embedding_dim];
            for f in &mut features {
                *f = self.rng.random_range(0.0..1.0) * 2.0 - 1.0; // [-1, 1]
            }
            embeddings.insert(node_label.clone(), features);
        }

        // GraphSAGE aggregation (2 iterations)
        for _ in 0..2 {
            let mut new_embeddings = embeddings.clone();

            for (node_label, &node_idx) in &node_map {
                let node_community = communities.node_to_community.get(node_label);

                // Get neighbors, prioritizing same-community neighbors
                let mut same_comm_neighbors = Vec::new();
                let mut other_neighbors = Vec::new();

                for neighbor_idx in graph.neighbors(node_idx) {
                    if let Some(neighbor_label) = graph.node_weight(neighbor_idx) {
                        let neighbor_community = communities.node_to_community.get(neighbor_label);

                        if node_community == neighbor_community {
                            same_comm_neighbors.push(neighbor_label.clone());
                        } else {
                            other_neighbors.push(neighbor_label.clone());
                        }
                    }
                }

                // Aggregate: prioritize same-community neighbors
                let mut aggregated = vec![0.0; self.config.embedding_dim];
                let mut count = 0.0;

                for neighbor in &same_comm_neighbors {
                    if let Some(neighbor_emb) = embeddings.get(neighbor) {
                        for (i, &val) in neighbor_emb.iter().enumerate() {
                            aggregated[i] += val * self.config.community_bias;
                        }
                        count += self.config.community_bias;
                    }
                }

                for neighbor in &other_neighbors {
                    if let Some(neighbor_emb) = embeddings.get(neighbor) {
                        for (i, &val) in neighbor_emb.iter().enumerate() {
                            aggregated[i] += val;
                        }
                        count += 1.0;
                    }
                }

                if count > 0.0 {
                    for val in &mut aggregated {
                        *val /= count;
                    }

                    // Combine with own embedding
                    if let Some(own_emb) = embeddings.get(node_label) {
                        for (i, &val) in own_emb.iter().enumerate() {
                            aggregated[i] = (aggregated[i] + val) / 2.0;
                        }
                    }

                    new_embeddings.insert(node_label.clone(), aggregated);
                }
            }

            embeddings = new_embeddings;
        }

        // Normalize embeddings
        for emb in embeddings.values_mut() {
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for val in emb {
                    *val /= norm;
                }
            }
        }

        Ok(embeddings)
    }

    /// Generate embeddings using Node2Vec with community-biased random walks
    pub fn embed_node2vec(
        &mut self,
        triples: &[Triple],
        communities: &CommunityStructure,
    ) -> GraphRAGResult<HashMap<String, Vec<f64>>> {
        let (graph, node_map) = self.build_graph(triples);

        if graph.node_count() == 0 {
            return Ok(HashMap::new());
        }

        // Generate community-biased random walks
        let walks = self.generate_community_biased_walks(&graph, &node_map, communities)?;

        // Train skip-gram model (simplified)
        let embeddings = self.train_skip_gram(&walks, &node_map)?;

        Ok(embeddings)
    }

    /// Generate community-biased random walks
    fn generate_community_biased_walks(
        &mut self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
        communities: &CommunityStructure,
    ) -> GraphRAGResult<Vec<Vec<String>>> {
        let mut walks = Vec::new();

        for _ in 0..self.config.num_walks {
            for (node_label, &start_idx) in node_map {
                let walk = self.node2vec_walk(graph, start_idx, node_label, communities);
                walks.push(walk);
            }
        }

        Ok(walks)
    }

    /// Single Node2Vec random walk with community bias
    fn node2vec_walk(
        &mut self,
        graph: &UnGraph<String, ()>,
        start: NodeIndex,
        start_label: &str,
        communities: &CommunityStructure,
    ) -> Vec<String> {
        let mut walk = vec![start_label.to_string()];
        let mut current = start;
        let mut prev: Option<NodeIndex> = None;
        let start_community = communities.node_to_community.get(start_label);

        for _ in 1..self.config.walk_length {
            let neighbors: Vec<NodeIndex> = graph.neighbors(current).collect();

            if neighbors.is_empty() {
                break;
            }

            // Calculate transition probabilities with community bias
            let mut probs = vec![0.0; neighbors.len()];

            for (i, &neighbor) in neighbors.iter().enumerate() {
                let mut prob = 1.0;

                // Node2Vec bias
                if let Some(p) = prev {
                    if neighbor == p {
                        prob /= self.config.p; // Return parameter
                    } else if !graph.neighbors(p).any(|n| n == neighbor) {
                        prob /= self.config.q; // In-out parameter
                    }
                }

                // Community bias: prefer staying in same community
                if let Some(neighbor_label) = graph.node_weight(neighbor) {
                    let neighbor_community = communities.node_to_community.get(neighbor_label);
                    if start_community == neighbor_community {
                        prob *= self.config.community_bias;
                    }
                }

                probs[i] = prob;
            }

            // Normalize probabilities
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            }

            // Sample next node
            let r = self.rng.random_range(0.0..1.0);
            let mut cumsum = 0.0;
            let mut next_idx = 0;

            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    next_idx = i;
                    break;
                }
            }

            let next = neighbors[next_idx];
            if let Some(next_label) = graph.node_weight(next) {
                walk.push(next_label.clone());
            }

            prev = Some(current);
            current = next;
        }

        walk
    }

    /// Train skip-gram model on walks (simplified)
    fn train_skip_gram(
        &mut self,
        walks: &[Vec<String>],
        node_map: &HashMap<String, NodeIndex>,
    ) -> GraphRAGResult<HashMap<String, Vec<f64>>> {
        // Initialize embeddings randomly
        let mut embeddings: HashMap<String, Vec<f64>> = HashMap::new();
        for node_label in node_map.keys() {
            let mut emb = vec![0.0; self.config.embedding_dim];
            for val in &mut emb {
                *val = (self.rng.random_range(0.0..1.0) - 0.5) * 0.1; // Small random init
            }
            embeddings.insert(node_label.clone(), emb);
        }

        // Skip-gram training (simplified, no negative sampling)
        let learning_rate = 0.025;
        let num_epochs = 5;

        for _ in 0..num_epochs {
            for walk in walks {
                for (i, target) in walk.iter().enumerate() {
                    let start = i.saturating_sub(self.config.window_size);
                    let end = (i + self.config.window_size + 1).min(walk.len());

                    for (offset, context) in walk[start..end].iter().enumerate() {
                        let j = start + offset;
                        if i == j {
                            continue;
                        }

                        // Update embeddings to be similar
                        if let (Some(target_emb), Some(context_emb)) =
                            (embeddings.get(target), embeddings.get(context))
                        {
                            let mut target_update = vec![0.0; self.config.embedding_dim];
                            let mut context_update = vec![0.0; self.config.embedding_dim];

                            for k in 0..self.config.embedding_dim {
                                let diff = context_emb[k] - target_emb[k];
                                target_update[k] = learning_rate * diff;
                                context_update[k] = -learning_rate * diff;
                            }

                            if let Some(emb) = embeddings.get_mut(target) {
                                for (k, &update) in target_update.iter().enumerate() {
                                    emb[k] += update;
                                }
                            }

                            if let Some(emb) = embeddings.get_mut(context) {
                                for (k, &update) in context_update.iter().enumerate() {
                                    emb[k] += update;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize
        for emb in embeddings.values_mut() {
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for val in emb {
                    *val /= norm;
                }
            }
        }

        Ok(embeddings)
    }

    /// Build graph from triples
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_community_aware_embeddings() {
        let triples = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
            Triple::new("http://a", "http://rel", "http://c"),
        ];

        let assignments = vec![
            ("http://a".to_string(), 0),
            ("http://b".to_string(), 0),
            ("http://c".to_string(), 0),
        ];

        let communities = CommunityStructure::from_assignments(&assignments, 0.8);

        let config = EmbeddingConfig {
            embedding_dim: 16,
            ..Default::default()
        };

        let mut embedder = CommunityAwareEmbeddings::new(config);
        let embeddings = embedder
            .embed_graphsage(&triples, &communities)
            .expect("embeddings failed");

        assert_eq!(embeddings.len(), 3);
        for emb in embeddings.values() {
            assert_eq!(emb.len(), 16);
        }
    }

    #[test]
    fn test_node2vec_embeddings() {
        let triples = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
            Triple::new("http://c", "http://rel", "http://d"),
        ];

        let assignments = vec![
            ("http://a".to_string(), 0),
            ("http://b".to_string(), 0),
            ("http://c".to_string(), 1),
            ("http://d".to_string(), 1),
        ];

        let communities = CommunityStructure::from_assignments(&assignments, 0.7);

        let config = EmbeddingConfig {
            embedding_dim: 16,
            walk_length: 10,
            num_walks: 5,
            ..Default::default()
        };

        let mut embedder = CommunityAwareEmbeddings::new(config);
        let embeddings = embedder
            .embed_node2vec(&triples, &communities)
            .expect("embeddings failed");

        assert_eq!(embeddings.len(), 4);
    }
}
