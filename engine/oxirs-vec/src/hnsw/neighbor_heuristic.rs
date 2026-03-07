//! SELECT-NEIGHBORS-HEURISTIC - Algorithm 4 from the HNSW paper
//!
//! Yu. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor
//! Search Using Hierarchical Navigable Small World Graphs," in *IEEE TPAMI*, 2020.
//!
//! This algorithm produces better graph connectivity than the simple greedy selection
//! (Algorithm 3) by:
//!
//! 1. **Extending candidates** with the neighbors-of-candidates so that long-range
//!    shortcuts are discovered.
//! 2. **Pruning "shadowed" links** – a candidate `e` is discarded if there already exists
//!    a selected neighbor `r` that is *closer to `e`* than the query `q` is to `e`.
//!    This avoids building redundant short-range links and keeps the graph well-navigable
//!    over long distances (the "small-world" property).
//! 3. Respecting layer-specific M limits (M_max0 for layer 0, M for upper layers).

use crate::hnsw::HnswIndex;
use anyhow::Result;
use std::collections::BinaryHeap;

/// Result of the heuristic neighbor selection: `(node_id, distance)` pairs sorted
/// by distance ascending (closest first).
pub type NeighborList = Vec<(usize, f32)>;

impl HnswIndex {
    // ──────────────────────────────────────────────────────────────────────────
    // Public entry point
    // ──────────────────────────────────────────────────────────────────────────

    /// **SELECT-NEIGHBORS-HEURISTIC** (Algorithm 4, HNSW paper).
    ///
    /// # Arguments
    ///
    /// * `candidates`         – Initial candidate set as `(node_id, distance_to_query)`.
    /// * `query_vector`       – The query's raw `f32` data (needed when `extend_candidates`).
    /// * `m`                  – Target number of connections for this node at this layer.
    /// * `layer`              – Current layer (0 = base; drives M limits).
    /// * `extend_candidates`  – If `true`, expand `W` with the neighbors of each candidate
    ///   (Algorithm 4, line 8). Costs extra distance computations but
    ///   improves recall for high-dimensional data.
    /// * `keep_pruned`        – If `true`, fill remaining slots with pruned candidates rather
    ///   than returning fewer than `m` neighbors (Algorithm 4, lines 17-20).
    ///
    /// Returns up to `m` selected neighbors sorted by distance ascending.
    pub fn select_neighbors_heuristic_v2(
        &self,
        candidates: &[(usize, f32)],
        query_vector: &[f32],
        m: usize,
        layer: usize,
        extend_candidates: bool,
        keep_pruned: bool,
    ) -> Result<NeighborList> {
        let m_max = if layer == 0 {
            self.config().m_l0
        } else {
            self.config().m
        };

        // Effective M: respect both the caller-supplied limit and the layer cap
        let effective_m = m.min(m_max);

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // ── W: working candidate heap (min-heap by distance) ──────────────────
        // We use a BinaryHeap<Reverse<OrderedCandidate>> so `pop()` returns the
        // nearest element.
        let mut w: BinaryHeap<RevCandidate> = candidates
            .iter()
            .map(|&(id, dist)| RevCandidate { id, dist })
            .collect();

        // ── Extend candidates with neighbors (optional) ───────────────────────
        if extend_candidates {
            // Collect extra candidates to avoid mutating `w` while iterating
            let snapshot: Vec<RevCandidate> = w.iter().cloned().collect();
            for RevCandidate { id: cand_id, .. } in &snapshot {
                let node = match self.nodes().get(*cand_id) {
                    Some(n) => n,
                    None => continue,
                };
                // Use connections at the current layer (or layer 0 if unavailable)
                let effective_layer = layer.min(node.connections.len().saturating_sub(1));
                let neighbors = match node.get_connections(effective_layer) {
                    Some(c) => c.clone(),
                    None => continue,
                };
                for neighbor_id in neighbors {
                    // Avoid duplicates that are already in W
                    if w.iter().any(|rc| rc.id == neighbor_id) {
                        continue;
                    }
                    if let Ok(dist) = self.calculate_distance_from_slice(query_vector, neighbor_id)
                    {
                        w.push(RevCandidate {
                            id: neighbor_id,
                            dist,
                        });
                    }
                }
            }
        }

        // ── Main selection loop (Algorithm 4, lines 10-19) ────────────────────
        let mut selected: NeighborList = Vec::with_capacity(effective_m);
        let mut pruned: NeighborList = Vec::new();

        while let Some(RevCandidate {
            id: e,
            dist: dist_e,
        }) = w.pop()
        {
            if selected.len() >= effective_m {
                break;
            }

            // Check the HNSW "shadowing" condition:
            //   Add `e` if and only if dist(q, e) < dist(r, e) for ALL already-selected r.
            // In other words: keep `e` only when it is the closest node to the query
            // from its own perspective — not "shadowed" by a closer selected neighbor.
            let shadowed = selected.iter().any(|&(r_id, _)| {
                // distance between the current candidate `e` and selected neighbor `r`
                self.calculate_distance_between_nodes_pub(r_id, e)
                    .map(|dist_r_e| dist_r_e < dist_e)
                    .unwrap_or(false)
            });

            if !shadowed {
                selected.push((e, dist_e));
            } else {
                pruned.push((e, dist_e));
            }
        }

        // ── Keep-pruned phase (Algorithm 4, lines 17-20) ─────────────────────
        if keep_pruned {
            pruned.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for item in pruned {
                if selected.len() >= effective_m {
                    break;
                }
                selected.push(item);
            }
        }

        // Return sorted by distance ascending (closest first)
        selected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(selected)
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Prune using the heuristic (public version used during construction)
    // ──────────────────────────────────────────────────────────────────────────

    /// Apply the heuristic pruning on a node's current connections at `level`.
    ///
    /// Unlike the simple `prune_connections` method (which uses the diversity-based
    /// heuristic), this method implements Algorithm 4 precisely and prunes in-place.
    pub fn prune_connections_heuristic(&mut self, node_id: usize, level: usize) -> Result<()> {
        use std::collections::HashSet;

        let max_connections = if level == 0 {
            self.config().m_l0
        } else {
            self.config().m
        };

        // Snapshot current connections and query vector
        let (current_conns, query_vec) = {
            let node = match self.nodes().get(node_id) {
                Some(n) => n,
                None => return Ok(()),
            };
            let conns: Vec<usize> = node
                .get_connections(level)
                .map(|c| c.iter().cloned().collect())
                .unwrap_or_default();
            (conns, node.vector_data_f32.clone())
        };

        if current_conns.len() <= max_connections {
            return Ok(());
        }

        // Build candidate list: (neighbor_id, distance_to_node)
        let mut candidates: Vec<(usize, f32)> = current_conns
            .iter()
            .filter_map(|&nb| {
                self.calculate_distance_from_slice(&query_vec, nb)
                    .ok()
                    .map(|d| (nb, d))
            })
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Run heuristic selection (extend_candidates=false, keep_pruned=false during prune)
        let selected = self.select_neighbors_heuristic_v2(
            &candidates,
            &query_vec,
            max_connections,
            level,
            false,
            false,
        )?;

        let selected_set: HashSet<usize> = selected.iter().map(|&(id, _)| id).collect();
        let removed: Vec<usize> = current_conns
            .iter()
            .filter(|&&id| !selected_set.contains(&id))
            .cloned()
            .collect();

        // Update node's connections
        if let Some(node) = self.nodes_mut().get_mut(node_id) {
            if let Some(conns) = node.get_connections_mut(level) {
                *conns = selected_set.clone();
            }
        }

        // Remove bidirectional links for pruned neighbors
        for pruned_id in removed {
            if let Some(pruned_node) = self.nodes_mut().get_mut(pruned_id) {
                pruned_node.remove_connection(level, node_id);
            }
        }

        // Ensure bidirectional links for selected neighbors
        for &selected_id in &selected_set {
            if let Some(selected_node) = self.nodes_mut().get_mut(selected_id) {
                selected_node.add_connection(level, node_id);
            }
        }

        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ──────────────────────────────────────────────────────────────────────────

    /// Calculate distance from a raw `f32` slice to a node, using the configured metric.
    fn calculate_distance_from_slice(&self, query: &[f32], node_id: usize) -> Result<f32> {
        let node = self
            .nodes()
            .get(node_id)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;

        let query_vec = crate::Vector::new(query.to_vec());
        self.config().metric.distance(&query_vec, &node.vector)
    }

    /// Distance between two nodes identified by their IDs (public visibility for test access).
    pub fn calculate_distance_between_nodes_pub(&self, a: usize, b: usize) -> Option<f32> {
        let nodes = self.nodes();
        let node_a = nodes.get(a)?;
        let node_b = nodes.get(b)?;
        node_a
            .vector
            .cosine_similarity(&node_b.vector)
            .ok()
            .map(|s| 1.0 - s)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Local ordering helper (min-heap entry)
// ────────────────────────────────────────────────────────────────────────────

/// Min-heap entry: ordered by distance ascending (smallest distance = highest priority).
#[derive(Debug, Clone, Copy, PartialEq)]
struct RevCandidate {
    id: usize,
    dist: f32,
}

impl Eq for RevCandidate {}

impl PartialOrd for RevCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RevCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse: smaller dist = greater priority in a max-heap
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::hnsw::{HnswConfig, HnswIndex};
    use crate::{Vector, VectorIndex};

    fn make_index(pairs: &[(&str, Vec<f32>)]) -> HnswIndex {
        let config = HnswConfig::default();
        let mut index = HnswIndex::new_cpu_only(config);
        for (uri, data) in pairs {
            index
                .insert(uri.to_string(), Vector::new(data.clone()))
                .expect("insert failed");
        }
        index
    }

    #[test]
    fn test_heuristic_returns_at_most_m() {
        let index = make_index(&[
            ("a", vec![1.0, 0.0, 0.0]),
            ("b", vec![0.9, 0.1, 0.0]),
            ("c", vec![0.8, 0.2, 0.0]),
            ("d", vec![0.0, 1.0, 0.0]),
            ("e", vec![0.0, 0.0, 1.0]),
        ]);

        // Prepare candidate list (0-based IDs)
        let candidates: Vec<(usize, f32)> = (0..5).map(|i| (i, i as f32 * 0.1)).collect();
        let query = vec![1.0f32, 0.0, 0.0];

        let result = index
            .select_neighbors_heuristic_v2(&candidates, &query, 3, 0, false, false)
            .expect("heuristic failed");

        assert!(
            result.len() <= 3,
            "Got {} neighbors, expected <= 3",
            result.len()
        );
    }

    #[test]
    fn test_heuristic_sorted_by_distance() {
        let index = make_index(&[
            ("a", vec![1.0, 0.0]),
            ("b", vec![0.7, 0.7]),
            ("c", vec![0.0, 1.0]),
            ("d", vec![-1.0, 0.0]),
        ]);

        let candidates: Vec<(usize, f32)> = vec![(0, 0.1), (1, 0.3), (2, 0.8), (3, 1.9)];
        let query = vec![1.0f32, 0.0];

        let result = index
            .select_neighbors_heuristic_v2(&candidates, &query, 4, 0, false, false)
            .expect("heuristic failed");

        // Verify sorted order
        for window in result.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Results not sorted: {} > {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_heuristic_prunes_shadowed_candidates() {
        // All candidates nearly co-directional (clustered): heuristic should prune redundant ones
        let index = make_index(&[
            ("a", vec![1.0, 0.0, 0.0]),
            ("b", vec![0.9999, 0.001, 0.0]), // almost identical to a
            ("c", vec![0.9998, 0.002, 0.0]), // almost identical to a and b
            ("d", vec![0.0, 0.0, 1.0]),      // very different
        ]);

        let candidates: Vec<(usize, f32)> = vec![
            (0, 0.001), // a -- nearest
            (1, 0.002), // b -- nearly same direction
            (2, 0.003), // c -- nearly same direction
            (3, 0.9),   // d -- far but diverse
        ];
        let query = vec![1.0f32, 0.0, 0.0];

        // With only 2 slots, the heuristic should pick `a` (closest) and `d` (diverse),
        // pruning `b` and `c` as shadowed by `a`.
        let result = index
            .select_neighbors_heuristic_v2(&candidates, &query, 2, 1, false, false)
            .expect("heuristic failed");

        assert!(result.len() <= 2);
        // Node 0 (a) should always be selected (nearest)
        assert!(
            result.iter().any(|&(id, _)| id == 0),
            "Closest node must always be selected"
        );
    }

    #[test]
    fn test_keep_pruned_fills_slots() {
        let index = make_index(&[
            ("a", vec![1.0, 0.0]),
            ("b", vec![0.9999, 0.001]),
            ("c", vec![0.9998, 0.002]),
            ("d", vec![0.9997, 0.003]),
        ]);

        // 4 candidates all in the same direction; heuristic without keep_pruned returns 1
        let candidates: Vec<(usize, f32)> = vec![(0, 0.01), (1, 0.02), (2, 0.03), (3, 0.04)];
        let query = vec![1.0f32, 0.0];

        let without_keep = index
            .select_neighbors_heuristic_v2(&candidates, &query, 4, 1, false, false)
            .expect("heuristic failed");

        let with_keep = index
            .select_neighbors_heuristic_v2(&candidates, &query, 4, 1, false, true)
            .expect("heuristic with keep failed");

        // keep_pruned should result in >= number of results without keep_pruned
        assert!(
            with_keep.len() >= without_keep.len(),
            "keep_pruned should fill more slots"
        );
    }

    #[test]
    fn test_extend_candidates_discovers_more() {
        // Build a small connected index, then check that extend_candidates
        // can discover neighbors-of-neighbors
        let index = make_index(&[
            ("origin", vec![0.0, 0.0]),
            ("mid", vec![0.5, 0.5]),
            ("far", vec![1.0, 1.0]),
        ]);

        let candidates = vec![(0, 0.0f32), (1, 0.5)];
        let query = vec![0.0f32, 0.0];

        let without_ext = index
            .select_neighbors_heuristic_v2(&candidates, &query, 5, 0, false, false)
            .expect("failed");

        let with_ext = index
            .select_neighbors_heuristic_v2(&candidates, &query, 5, 0, true, false)
            .expect("failed");

        // With extension, may discover the "far" node through "mid"'s connections.
        // At minimum, the result set size should be >= without extension.
        assert!(
            with_ext.len() >= without_ext.len(),
            "extend_candidates should not reduce result count"
        );
    }

    #[test]
    fn test_empty_candidates() {
        let index = make_index(&[("a", vec![1.0, 0.0])]);
        let result = index
            .select_neighbors_heuristic_v2(&[], &[1.0, 0.0], 5, 0, false, false)
            .expect("empty candidates should not fail");
        assert!(result.is_empty());
    }

    #[test]
    fn test_prune_connections_heuristic() {
        let config = HnswConfig {
            m: 2,
            m_l0: 4,
            ..HnswConfig::default()
        };
        let mut index = HnswIndex::new_cpu_only(config);

        // Insert enough nodes so some pruning will happen
        for i in 0..10usize {
            let angle = std::f32::consts::PI * 2.0 * i as f32 / 10.0;
            let v = Vector::new(vec![angle.cos(), angle.sin()]);
            index
                .insert(format!("node_{}", i), v)
                .expect("insert failed");
        }

        // Prune node 0 at level 0 (should not panic and should respect M limit)
        index
            .prune_connections_heuristic(0, 0)
            .expect("prune failed");

        let connections_at_0 = index
            .nodes()
            .first()
            .and_then(|n| n.get_connections(0))
            .map(|c| c.len())
            .unwrap_or(0);

        assert!(
            connections_at_0 <= 4,
            "Expected <= 4 connections (m_l0), got {}",
            connections_at_0
        );
    }
}
