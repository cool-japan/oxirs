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
        let node = Node::new(uri.clone(), vector.clone(), level);

        // Add node to the index
        self.nodes_mut().push(node);
        self.uri_to_id_mut().insert(uri, node_id);

        // If this is the first node, set it as entry point
        if self.entry_point().is_none() {
            self.set_entry_point(Some(node_id));
            self.stats_mut()
                .total_insertions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }

        // Full HNSW construction algorithm implementation
        let entry_point = self
            .entry_point()
            .expect("entry point should exist after initial insertion");

        // Phase 1: Find nearest neighbors at each level from top to insertion level
        let mut nearest_points = vec![entry_point];
        let mut visited = std::collections::HashSet::new();
        visited.insert(entry_point);

        // Get the maximum level of the entry point
        let entry_level = if let Some(node) = self.nodes().get(entry_point) {
            node.level()
        } else {
            0
        };

        // Navigate from top level down to the new node's level
        for lc in (level + 1..=entry_level).rev() {
            nearest_points =
                self.search_layer_for_construction(&vector, &nearest_points, 1, lc, &mut visited)?;
        }

        // Phase 2: Insert the node at its assigned level and all levels below
        for lc in (0..=level).rev() {
            // Find M nearest neighbors at this level
            let ef_construction = self.config().ef_construction.max(self.config().m);
            let candidates = self.search_layer_for_construction(
                &vector,
                &nearest_points,
                ef_construction,
                lc,
                &mut visited,
            )?;

            // Select M neighbors for connection
            let m = if lc == 0 {
                self.config().m_l0
            } else {
                self.config().m
            };
            let neighbors = self.select_neighbors_for_insertion(&candidates, m);

            // Collect neighbors that need pruning
            let max_connections = if lc == 0 {
                self.config().m_l0
            } else {
                self.config().m
            };
            let mut neighbors_to_prune = Vec::new();

            // Bidirectionally link the new node with selected neighbors
            for &neighbor_id in &neighbors {
                // Add connection from new node to neighbor
                if let Some(new_node) = self.nodes_mut().get_mut(node_id) {
                    new_node.add_connection(lc, neighbor_id);
                }

                // Add connection from neighbor to new node
                if let Some(neighbor_node) = self.nodes_mut().get_mut(neighbor_id) {
                    neighbor_node.add_connection(lc, node_id);

                    // Check if pruning is needed
                    if let Some(connections) = neighbor_node.get_connections(lc) {
                        if connections.len() > max_connections {
                            neighbors_to_prune.push(neighbor_id);
                        }
                    }
                }
            }

            // Prune neighbors after releasing borrows
            for neighbor_id in neighbors_to_prune {
                self.prune_connections(neighbor_id, lc)?;
            }

            // Update nearest points for next level
            nearest_points = neighbors;
        }

        // Phase 3: Update entry point if new node has higher level
        if level > entry_level {
            self.set_entry_point(Some(node_id));
        }

        self.stats_mut()
            .total_insertions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Search layer during construction (similar to search_layer but optimized for insertion)
    fn search_layer_for_construction(
        &self,
        query: &Vector,
        entry_points: &[usize],
        num_closest: usize,
        level: usize,
        visited: &mut std::collections::HashSet<usize>,
    ) -> Result<Vec<usize>> {
        use crate::hnsw::Candidate;
        use std::collections::BinaryHeap;

        let mut candidates = BinaryHeap::new();
        let mut dynamic_list: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();

        // Initialize with entry points
        for &entry_id in entry_points {
            if !visited.contains(&entry_id) {
                visited.insert(entry_id);
                let distance = self.calculate_distance(query, entry_id)?;
                let candidate = Candidate::new(entry_id, distance);
                candidates.push(candidate);
                dynamic_list.push(std::cmp::Reverse(candidate));
            }
        }

        // Main search loop
        while let Some(current) = candidates.pop() {
            // Early termination
            if let Some(&std::cmp::Reverse(best)) = dynamic_list.peek() {
                if current.distance > best.distance && dynamic_list.len() >= num_closest {
                    break;
                }
            }

            // Explore neighbors
            if let Some(node) = self.nodes().get(current.id) {
                if let Some(connections) = node.get_connections(level) {
                    for &neighbor_id in connections {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);

                            let distance = self.calculate_distance(query, neighbor_id)?;
                            let neighbor_candidate = Candidate::new(neighbor_id, distance);

                            let should_add = if dynamic_list.len() < num_closest {
                                true
                            } else if let Some(&std::cmp::Reverse(worst)) = dynamic_list.peek() {
                                distance < worst.distance
                            } else {
                                false
                            };

                            if should_add {
                                candidates.push(neighbor_candidate);
                                dynamic_list.push(std::cmp::Reverse(neighbor_candidate));

                                if dynamic_list.len() > num_closest {
                                    dynamic_list.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return candidate IDs
        let results: Vec<usize> = dynamic_list
            .into_iter()
            .map(|std::cmp::Reverse(candidate)| candidate.id)
            .collect();

        Ok(results)
    }

    /// Select neighbors for insertion with heuristic
    fn select_neighbors_for_insertion(&self, candidates: &[usize], m: usize) -> Vec<usize> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        // Use heuristic selection for better graph connectivity
        self.select_neighbors_heuristic(candidates, m)
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

    /// Update an existing vector in the index
    pub fn update_vector(&mut self, uri: &str, new_vector: Vector) -> Result<()> {
        // Find the node ID for this URI
        let node_id = self
            .uri_to_id()
            .get(uri)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("URI {} not found", uri))?;

        // Get the node's level before updating
        let level = if let Some(node) = self.nodes().get(node_id) {
            node.level()
        } else {
            return Err(anyhow::anyhow!("Node {} not found", node_id));
        };

        // Update the vector data
        if let Some(node) = self.nodes_mut().get_mut(node_id) {
            node.vector = new_vector.clone();
            node.vector_data_f32 = new_vector.as_f32();
        }

        // Recompute connections at each level to maintain graph quality
        // This is a simplified approach; for production, consider incremental updates
        for lc in 0..=level {
            // Get current connections at this level
            let current_connections = if let Some(node) = self.nodes().get(node_id) {
                if let Some(connections) = node.get_connections(lc) {
                    connections.clone()
                } else {
                    continue;
                }
            } else {
                continue;
            };

            // Prune and reconnect based on new distances
            if !current_connections.is_empty() {
                self.prune_connections(node_id, lc)?;
            }
        }

        self.stats_mut()
            .total_updates
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Remove a vector from the index
    pub fn remove_vector(&mut self, uri: &str) -> Result<()> {
        // Find the node ID for this URI
        let node_id = self
            .uri_to_id()
            .get(uri)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("URI {} not found", uri))?;

        // Get the node's level and connections before removal
        let (_level, all_connections) = if let Some(node) = self.nodes().get(node_id) {
            let mut conns = Vec::new();
            for lc in 0..=node.level() {
                if let Some(connections) = node.get_connections(lc) {
                    conns.push((lc, connections.clone()));
                }
            }
            (node.level(), conns)
        } else {
            return Err(anyhow::anyhow!("Node {} not found", node_id));
        };

        // Remove all bidirectional connections
        for (lc, connections) in all_connections {
            for &connected_id in &connections {
                if let Some(connected_node) = self.nodes_mut().get_mut(connected_id) {
                    connected_node.remove_connection(lc, node_id);
                }
            }
        }

        // If this was the entry point, find a new one
        if self.entry_point() == Some(node_id) {
            // Find another node with the maximum level
            let mut new_entry_point = None;
            let mut max_level = 0;

            for (id, node) in self.nodes().iter().enumerate() {
                if id != node_id && node.level() >= max_level {
                    max_level = node.level();
                    new_entry_point = Some(id);
                }
            }

            self.set_entry_point(new_entry_point);
        }

        // Mark the node as removed (we keep the slot to maintain node IDs)
        // In production, consider tombstoning or compaction strategies
        if let Some(node) = self.nodes_mut().get_mut(node_id) {
            // Clear the node's data while keeping the slot
            node.connections.clear();
            node.vector_data_f32.clear();
        }

        // Remove from URI mapping
        self.uri_to_id_mut().remove(uri);

        self.stats_mut()
            .total_deletions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Rebuild connections for a node at a specific level
    /// Used after updates to maintain graph quality
    fn rebuild_connections_at_level(&mut self, node_id: usize, level: usize) -> Result<()> {
        if node_id >= self.nodes().len() {
            return Err(anyhow::anyhow!("Invalid node ID: {}", node_id));
        }

        // Get the node's vector
        let query_vector = if let Some(node) = self.nodes().get(node_id) {
            node.vector.clone()
        } else {
            return Err(anyhow::anyhow!("Node not found: {}", node_id));
        };

        // Find nearest neighbors at this level (excluding self)
        let mut candidates = Vec::new();
        for (candidate_id, candidate_node) in self.nodes().iter().enumerate() {
            if candidate_id == node_id {
                continue;
            }

            // Check if candidate has connections at this level
            if let Some(connections) = candidate_node.get_connections(level) {
                if !connections.is_empty() || candidate_node.level() >= level {
                    if let Ok(distance) = query_vector.cosine_similarity(&candidate_node.vector) {
                        let distance_metric = 1.0 - distance; // Convert similarity to distance
                        candidates.push((candidate_id, distance_metric));
                    }
                }
            }
        }

        // Sort by distance
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select M neighbors
        let m = if level == 0 {
            self.config().m_l0
        } else {
            self.config().m
        };
        let selected_neighbors: Vec<usize> = candidates.iter().take(m).map(|(id, _)| *id).collect();

        // Update connections
        if let Some(node) = self.nodes_mut().get_mut(node_id) {
            if let Some(connections) = node.get_connections_mut(level) {
                connections.clear();
                for &neighbor_id in &selected_neighbors {
                    connections.insert(neighbor_id);
                }
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
