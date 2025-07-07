//! Index construction algorithms for HNSW

use crate::hnsw::{HnswIndex, Node};
use crate::Vector;
use anyhow::Result;
use std::collections::HashSet;

impl HnswIndex {
    /// Add a new vector to the index
    pub fn add_vector(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Check if URI already exists
        if self.uri_to_id().contains_key(&uri) {
            return Err(anyhow::anyhow!("URI {} already exists", uri));
        }

        let node_id = self.nodes().len();

        // Generate random level for the new node
        let level = self.generate_random_level();

        // Create new node
        let node = Node::new(uri.clone(), vector, level);

        // Add node to the index
        self.nodes_mut().push(node);
        self.uri_to_id_mut().insert(uri, node_id);

        // If this is the first node, set it as entry point
        if self.entry_point().is_none() {
            self.set_entry_point(Some(node_id));
            return Ok(());
        }

        // TODO: Implement full HNSW construction algorithm
        // This would include:
        // 1. Finding entry points at each level
        // 2. Selecting M nearest neighbors
        // 3. Connecting the new node to existing nodes
        // 4. Pruning connections if necessary

        self.stats_mut()
            .total_insertions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Calculate distance between two nodes
    fn calculate_distance_between_nodes(&self, node1_id: usize, node2_id: usize) -> Option<f32> {
        let nodes = self.nodes();
        let node1 = nodes.get(node1_id)?;
        let node2 = nodes.get(node2_id)?;

        // Use cosine distance (1 - cosine_similarity) for similarity calculations
        match node1.vector.cosine_similarity(&node2.vector) {
            Ok(similarity) => Some(1.0 - similarity),
            _ => None,
        }
    }

    /// Generate a random level for a new node
    fn generate_random_level(&mut self) -> usize {
        // Simple implementation of level generation
        // Real implementation would use exponential decay

        // Update RNG state (simple linear congruential generator)
        *self.rng_state_mut() = self
            .rng_state()
            .wrapping_mul(1103515245)
            .wrapping_add(12345);
        let random_value = (self.rng_state() >> 16) as f64 / 65535.0;

        let mut level = 0;
        let mut prob = 1.0 / self.level_multiplier();

        while random_value < prob && level < 16 {
            level += 1;
            prob *= 1.0 / self.level_multiplier();
        }

        level
    }

    /// Select M neighbors for connection
    fn select_neighbors_simple(&self, candidates: &[usize], m: usize) -> Vec<usize> {
        // Simple neighbor selection - just take the closest M
        candidates.iter().take(m).cloned().collect()
    }

    /// Select neighbors using heuristic for better connectivity
    fn select_neighbors_heuristic(&self, candidates: &[usize], m: usize) -> Vec<usize> {
        // Implementation of heuristic neighbor selection algorithm
        // This provides better connectivity than simple closest selection

        if candidates.is_empty() {
            return Vec::new();
        }

        if candidates.len() <= m {
            return candidates.to_vec();
        }

        let mut selected = HashSet::new();
        let mut candidates_with_distance: Vec<(usize, f32)> = Vec::new();

        // Calculate distances for all candidates (assuming they're already sorted by distance)
        for &candidate_id in candidates {
            if let Some(distance) =
                self.calculate_distance_between_nodes(candidate_id, candidates[0])
            {
                candidates_with_distance.push((candidate_id, distance));
            }
        }

        // Sort by distance (closest first)
        candidates_with_distance
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select first candidate (closest)
        if let Some((first_id, _)) = candidates_with_distance.first() {
            selected.insert(*first_id);
        }

        // For remaining slots, use diversity-based selection
        while selected.len() < m && selected.len() < candidates_with_distance.len() {
            let mut best_candidate = None;
            let mut best_score = f32::NEG_INFINITY;

            for &(candidate_id, candidate_distance) in &candidates_with_distance {
                if selected.contains(&candidate_id) {
                    continue;
                }

                // Calculate diversity score (prefer candidates far from already selected)
                let mut min_distance_to_selected = f32::INFINITY;
                for &selected_id in &selected {
                    if let Some(dist) =
                        self.calculate_distance_between_nodes(candidate_id, selected_id)
                    {
                        min_distance_to_selected = min_distance_to_selected.min(dist);
                    }
                }

                // Score combines closeness to query and distance from selected
                let diversity_weight = 0.3;
                let score = -candidate_distance + diversity_weight * min_distance_to_selected;

                if score > best_score {
                    best_score = score;
                    best_candidate = Some(candidate_id);
                }
            }

            if let Some(best_id) = best_candidate {
                selected.insert(best_id);
            } else {
                break;
            }
        }

        selected.into_iter().collect()
    }

    /// Prune connections to maintain M connections per node
    fn prune_connections(&mut self, node_id: usize, level: usize) -> Result<()> {
        // Implementation of connection pruning to maintain graph connectivity
        // while respecting the maximum connection limit (M)

        if node_id >= self.nodes().len() {
            return Err(anyhow::anyhow!("Invalid node ID: {}", node_id));
        }

        let max_connections = if level == 0 {
            self.config().m_l0 // Level 0 allows more connections
        } else {
            self.config().m
        };

        // Get current connections for this node at this level
        let current_connections = if let Some(node) = self.nodes().get(node_id) {
            if let Some(connections) = node.get_connections(level) {
                connections.clone()
            } else {
                return Ok(()); // No connections at this level
            }
        } else {
            return Err(anyhow::anyhow!("Node not found: {}", node_id));
        };

        // If we're within the limit, no pruning needed
        if current_connections.len() <= max_connections {
            return Ok(());
        }

        // Calculate distances from current node to all connected nodes
        let mut connection_distances: Vec<(usize, f32)> = Vec::new();

        for &connected_id in &current_connections {
            if let Some(distance) = self.calculate_distance_between_nodes(node_id, connected_id) {
                connection_distances.push((connected_id, distance));
            }
        }

        // Sort by distance (closest first)
        connection_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Use heuristic selection to choose best connections
        let candidates: Vec<usize> = connection_distances.into_iter().map(|(id, _)| id).collect();
        let selected_connections = self.select_neighbors_heuristic(&candidates, max_connections);
        let selected_set: HashSet<usize> = selected_connections.into_iter().collect();

        // Update the node's connections at this level
        if let Some(node) = self.nodes_mut().get_mut(node_id) {
            if let Some(connections) = node.get_connections_mut(level) {
                *connections = selected_set.clone();
            }
        }

        // Remove bidirectional connections for pruned nodes
        let connections_to_remove: HashSet<usize> = current_connections
            .difference(&selected_set)
            .cloned()
            .collect();

        for removed_id in connections_to_remove {
            if let Some(removed_node) = self.nodes_mut().get_mut(removed_id) {
                removed_node.remove_connection(level, node_id);
            }
        }

        // Add bidirectional connections for newly selected nodes
        for &selected_id in &selected_set {
            if let Some(selected_node) = self.nodes_mut().get_mut(selected_id) {
                selected_node.add_connection(level, node_id);
            }
        }

        Ok(())
    }

    /// Build index from a batch of vectors
    pub fn build_batch(&mut self, vectors: Vec<(String, Vector)>) -> Result<()> {
        // Placeholder for batch construction
        // This would be more efficient than adding vectors one by one
        for (uri, vector) in vectors {
            self.add_vector(uri, vector)?;
        }
        Ok(())
    }
}
