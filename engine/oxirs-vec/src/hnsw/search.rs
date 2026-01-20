//! Search algorithms for HNSW index

use crate::hnsw::{Candidate, HnswIndex};
use crate::Vector;
use anyhow::Result;
use std::collections::BinaryHeap;

impl HnswIndex {
    /// Search for k nearest neighbors
    pub fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        // Try query cache first if enabled
        if let Some(ref cache) = self.query_cache() {
            if let Some(cached_results) = cache.get(query, k) {
                return Ok(cached_results);
            }
        }

        // Cache miss - perform actual search
        if self.nodes().is_empty() || self.entry_point().is_none() {
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point().unwrap();
        let mut visited = std::collections::HashSet::new();
        let mut current_best = Vec::new();

        // Start from entry point
        let initial_distance = self.calculate_distance(query, entry_point)?;
        current_best.push(Candidate::new(entry_point, initial_distance));
        visited.insert(entry_point);

        // Get the highest level of the entry point
        let start_level = if let Some(node) = self.nodes().get(entry_point) {
            node.level()
        } else {
            0
        };

        // Phase 1: Search from top level down to level 1 (greedy search with beam size 1)
        for level in (1..=start_level).rev() {
            current_best = self.search_layer(query, &current_best, 1, level, &mut visited)?;
        }

        // Phase 2: Search at level 0 with dynamic beam size (ef parameter)
        let ef = (k as f32 * 1.5).max(16.0) as usize; // Dynamic ef based on k
        current_best = self.search_layer(query, &current_best, ef, 0, &mut visited)?;

        // Phase 3: Extract top k results
        let mut final_results = current_best;
        final_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_results.truncate(k);

        // Convert to output format
        let results: Vec<(String, f32)> = final_results
            .into_iter()
            .filter_map(|candidate| {
                self.nodes()
                    .get(candidate.id)
                    .map(|node| (node.uri.clone(), candidate.distance))
            })
            .collect();

        // Cache the results if caching is enabled
        if let Some(ref cache) = self.query_cache() {
            cache.put(query, k, results.clone());
        }

        Ok(results)
    }

    /// Core layer search algorithm for HNSW
    /// Implements the greedy search with dynamic candidate list
    fn search_layer(
        &self,
        query: &Vector,
        entry_points: &[Candidate],
        num_closest: usize,
        level: usize,
        visited: &mut std::collections::HashSet<usize>,
    ) -> Result<Vec<Candidate>> {
        // Priority queue for candidates (max-heap based on distance)
        let mut candidates = BinaryHeap::new();

        // Priority queue for best results so far (min-heap based on distance)
        let mut dynamic_list: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();

        // Initialize with entry points
        for entry in entry_points {
            if !visited.contains(&entry.id) {
                visited.insert(entry.id);
                candidates.push(*entry);
                dynamic_list.push(std::cmp::Reverse(*entry));
            }
        }

        // Main search loop
        while let Some(current) = candidates.pop() {
            // Early termination condition
            if let Some(&std::cmp::Reverse(best)) = dynamic_list.peek() {
                if current.distance > best.distance && dynamic_list.len() >= num_closest {
                    break; // Current candidate is farther than the farthest in our result set
                }
            }

            // Explore neighbors of current node at the specified level
            if let Some(node) = self.nodes().get(current.id) {
                if let Some(connections) = node.get_connections(level) {
                    for &neighbor_id in connections {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);

                            let distance = self.calculate_distance(query, neighbor_id)?;
                            let neighbor_candidate = Candidate::new(neighbor_id, distance);

                            // Check if this neighbor should be explored further
                            let should_add = if dynamic_list.len() < num_closest {
                                true // Still building initial set
                            } else if let Some(&std::cmp::Reverse(worst)) = dynamic_list.peek() {
                                distance < worst.distance // Better than current worst
                            } else {
                                false
                            };

                            if should_add {
                                candidates.push(neighbor_candidate);
                                dynamic_list.push(std::cmp::Reverse(neighbor_candidate));

                                // Keep only the best num_closest candidates
                                if dynamic_list.len() > num_closest {
                                    dynamic_list.pop(); // Remove worst
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert result to Vec<Candidate>
        let results: Vec<Candidate> = dynamic_list
            .into_iter()
            .map(|std::cmp::Reverse(candidate)| candidate)
            .collect();

        Ok(results)
    }

    /// Search with early stopping and beam search
    pub fn beam_search(
        &self,
        query: &Vector,
        beam_width: usize,
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Implementation of beam search for HNSW
        // Beam search maintains a beam of best candidates at each level

        if self.nodes().is_empty() || self.entry_point().is_none() {
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point().unwrap();
        let mut visited = std::collections::HashSet::new();
        let mut beam: BinaryHeap<Candidate> = BinaryHeap::new();

        // Start with entry point
        let initial_distance = self.calculate_distance(query, entry_point)?;
        beam.push(Candidate::new(entry_point, initial_distance));
        visited.insert(entry_point);

        // Get the highest level of the entry point
        let start_level = if let Some(node) = self.nodes().get(entry_point) {
            node.level()
        } else {
            0
        };

        // Search from top level down to level 1
        for level in (1..=start_level).rev() {
            beam = self.beam_search_layer(query, beam, 1, level, &mut visited)?;
        }

        // Search at level 0 with full beam width
        beam = self.beam_search_layer(query, beam, beam_width, 0, &mut visited)?;

        // Convert beam to results, taking top k
        let mut results: Vec<(String, f32)> = beam
            .into_sorted_vec()
            .into_iter()
            .take(k)
            .filter_map(|candidate| {
                self.nodes()
                    .get(candidate.id)
                    .map(|node| (node.uri.clone(), candidate.distance))
            })
            .collect();

        // Sort by distance (ascending - closest first)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Helper function for beam search at a specific layer
    fn beam_search_layer(
        &self,
        query: &Vector,
        initial_beam: BinaryHeap<Candidate>,
        beam_width: usize,
        level: usize,
        visited: &mut std::collections::HashSet<usize>,
    ) -> Result<BinaryHeap<Candidate>> {
        let mut beam = initial_beam;
        let mut candidates = BinaryHeap::new();

        // Explore neighbors of current beam
        while let Some(current) = beam.pop() {
            if let Some(node) = self.nodes().get(current.id) {
                if let Some(connections) = node.get_connections(level) {
                    for &neighbor_id in connections {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);
                            let distance = self.calculate_distance(query, neighbor_id)?;
                            candidates.push(Candidate::new(neighbor_id, distance));
                        }
                    }
                }
            }
        }

        // Keep only the best beam_width candidates
        let mut new_beam = BinaryHeap::new();
        for candidate in candidates.into_sorted_vec().into_iter().take(beam_width) {
            new_beam.push(candidate);
        }

        Ok(new_beam)
    }

    /// Parallel search across multiple threads
    pub fn parallel_search(
        &self,
        query: &Vector,
        k: usize,
        num_threads: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Implementation of parallel HNSW search
        // Uses oxirs-core parallel processing abstractions

        if self.nodes().is_empty() || self.entry_point().is_none() {
            return Ok(Vec::new());
        }

        // Use multiple entry points for parallel search
        let entry_points = self.get_multiple_entry_points(num_threads);

        // Divide search space among threads
        let results_per_thread = (k + num_threads - 1) / num_threads; // Ceiling division

        // Use oxirs-core's parallel processing
        let all_results: Vec<Vec<(String, f32)>> =
            oxirs_core::parallel::parallel_map(&entry_points, |&entry_point| {
                self.search_from_entry_point(query, results_per_thread * 2, entry_point)
                    .unwrap_or_else(|_| Vec::new())
            });

        // Merge results from all threads
        let mut merged_results: Vec<(String, f32)> = all_results.into_iter().flatten().collect();

        // Remove duplicates and sort by distance
        merged_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        merged_results.dedup_by(|a, b| a.0 == b.0);

        // Take top k results
        merged_results.truncate(k);

        Ok(merged_results)
    }

    /// Get multiple entry points for parallel search
    fn get_multiple_entry_points(&self, num_points: usize) -> Vec<usize> {
        let mut entry_points = Vec::new();

        if let Some(main_entry) = self.entry_point() {
            entry_points.push(main_entry);

            // Add nodes from high levels as additional entry points
            for (id, node) in self.nodes().iter().enumerate() {
                if entry_points.len() >= num_points {
                    break;
                }
                if id != main_entry && node.level() > 0 {
                    entry_points.push(id);
                }
            }

            // Fill remaining slots with random nodes if needed
            while entry_points.len() < num_points && entry_points.len() < self.nodes().len() {
                for (id, _) in self.nodes().iter().enumerate() {
                    if !entry_points.contains(&id) {
                        entry_points.push(id);
                        break;
                    }
                }
            }
        }

        entry_points
    }

    /// Search from a specific entry point
    fn search_from_entry_point(
        &self,
        query: &Vector,
        k: usize,
        entry_point: usize,
    ) -> Result<Vec<(String, f32)>> {
        // Simplified search from entry point
        // In a full implementation, this would be the regular HNSW search algorithm
        let mut candidates = BinaryHeap::new();
        let mut visited = std::collections::HashSet::new();

        // Start from the given entry point
        let initial_distance = self.calculate_distance(query, entry_point)?;
        candidates.push(Candidate::new(entry_point, initial_distance));
        visited.insert(entry_point);

        // Explore neighbors (simplified breadth-first exploration)
        let mut to_explore = vec![entry_point];
        let mut explored_count = 0;
        let max_explore = k * 10; // Limit exploration

        while let Some(current_id) = to_explore.pop() {
            if explored_count >= max_explore {
                break;
            }
            explored_count += 1;

            if let Some(node) = self.nodes().get(current_id) {
                // Explore connections at level 0
                if let Some(connections) = node.get_connections(0) {
                    for &neighbor_id in connections {
                        if !visited.contains(&neighbor_id) {
                            visited.insert(neighbor_id);
                            let distance = self.calculate_distance(query, neighbor_id)?;
                            candidates.push(Candidate::new(neighbor_id, distance));
                            to_explore.push(neighbor_id);
                        }
                    }
                }
            }
        }

        // Convert to results
        let results: Vec<(String, f32)> = candidates
            .into_sorted_vec()
            .into_iter()
            .take(k)
            .filter_map(|candidate| {
                self.nodes()
                    .get(candidate.id)
                    .map(|node| (node.uri.clone(), candidate.distance))
            })
            .collect();

        Ok(results)
    }

    /// Range search - find all neighbors within a distance threshold
    pub fn range_search(&self, query: &Vector, radius: f32) -> Result<Vec<(String, f32)>> {
        // Implementation of range search for HNSW
        // Finds all vectors within a specified distance radius

        if self.nodes().is_empty() || self.entry_point().is_none() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut to_explore = Vec::new();

        // Start from entry point
        let entry_point = self.entry_point().unwrap();
        to_explore.push(entry_point);
        visited.insert(entry_point);

        // Breadth-first exploration with distance filtering
        while let Some(current_id) = to_explore.pop() {
            let distance = self.calculate_distance(query, current_id)?;

            // If within radius, add to results
            if distance <= radius {
                if let Some(node) = self.nodes().get(current_id) {
                    results.push((node.uri.clone(), distance));
                }

                // Explore neighbors at level 0
                if let Some(node) = self.nodes().get(current_id) {
                    if let Some(connections) = node.get_connections(0) {
                        for &neighbor_id in connections {
                            if !visited.contains(&neighbor_id) {
                                visited.insert(neighbor_id);
                                to_explore.push(neighbor_id);
                            }
                        }
                    }
                }
            } else {
                // Even if current node is outside radius, explore neighbors
                // if the current distance is close to the radius (within some tolerance)
                let tolerance = radius * 0.1; // 10% tolerance
                if distance <= radius + tolerance {
                    if let Some(node) = self.nodes().get(current_id) {
                        if let Some(connections) = node.get_connections(0) {
                            for &neighbor_id in connections {
                                if !visited.contains(&neighbor_id) {
                                    visited.insert(neighbor_id);
                                    to_explore.push(neighbor_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort results by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Calculate distance between query and node
    pub(crate) fn calculate_distance(&self, query: &Vector, node_id: usize) -> Result<f32> {
        if let Some(node) = self.nodes().get(node_id) {
            // Use the configured similarity metric
            self.config().metric.distance(query, &node.vector)
        } else {
            Err(anyhow::anyhow!("Node {} not found", node_id))
        }
    }
}
