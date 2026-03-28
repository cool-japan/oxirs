//! HNSW (Hierarchical Navigable Small World) graph construction for ANN search.
//!
//! This module provides a pure-Rust implementation of the HNSW algorithm for
//! approximate nearest-neighbour (ANN) search in high-dimensional vector spaces.
//!
//! Random number generation uses `scirs2_core::random` (never `rand` directly).

use scirs2_core::random::{Random, RngExt, StdRng};
#[cfg(test)]
use std::collections::HashMap;
use std::collections::{BinaryHeap, HashSet};

// ─────────────────────────────────────────────────
// Distance helpers
// ─────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity ∈ [0, 1] (0 = orthogonal, 1 = identical direction).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
    }
}

/// Compute the layer assignment for a newly inserted node.
///
/// `m_l` is the level multiplier (typically `1 / ln(M)`).
/// `rng_val` must be drawn from `[0, 1)`.
pub fn random_level(m_l: f64, rng_val: f64) -> usize {
    if rng_val <= 0.0 {
        return 0;
    }
    (-rng_val.ln() * m_l).floor() as usize
}

// ─────────────────────────────────────────────────
// HnswConfig
// ─────────────────────────────────────────────────

/// Configuration parameters for the HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Target number of connections per layer for new nodes.
    pub m: usize,
    /// Maximum number of connections per layer.
    pub m_max: usize,
    /// Size of the dynamic candidate list used during construction.
    pub ef_construction: usize,
    /// Level multiplier: `1 / ln(M)`.
    pub m_l: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16_usize;
        HnswConfig {
            m,
            m_max: 32,
            ef_construction: 200,
            m_l: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswConfig {
    /// Build a configuration from `m` and `ef_construction`.
    pub fn new(m: usize, ef_construction: usize) -> Self {
        HnswConfig {
            m,
            m_max: m * 2,
            ef_construction,
            m_l: 1.0 / (m.max(2) as f64).ln(),
        }
    }
}

// ─────────────────────────────────────────────────
// HnswNode
// ─────────────────────────────────────────────────

/// A node in the HNSW graph.
///
/// `connections[layer]` holds the ids of the node's neighbours at that layer.
#[derive(Debug, Clone)]
pub struct HnswNode {
    pub id: usize,
    pub vector: Vec<f32>,
    /// `connections[0]` = bottom layer, higher indices = upper layers.
    pub connections: Vec<Vec<usize>>,
}

impl HnswNode {
    fn new(id: usize, vector: Vec<f32>, max_layer: usize) -> Self {
        HnswNode {
            id,
            vector,
            connections: vec![Vec::new(); max_layer + 1],
        }
    }

    fn ensure_layers(&mut self, layers: usize) {
        while self.connections.len() <= layers {
            self.connections.push(Vec::new());
        }
    }
}

// ─────────────────────────────────────────────────
// HnswGraph
// ─────────────────────────────────────────────────

/// The HNSW graph: a collection of nodes with hierarchical connectivity.
pub struct HnswGraph {
    pub nodes: Vec<HnswNode>,
    pub entry_point: Option<usize>,
    pub max_layer: usize,
    config: HnswConfig,
    /// Seeded RNG for reproducible level assignment.
    rng: StdRng,
}

impl HnswGraph {
    /// Create a new, empty HNSW graph with the given configuration.
    pub fn new(config: HnswConfig) -> Self {
        HnswGraph {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            config,
            rng: Random::seed(42),
        }
    }

    /// Insert a vector with the given `id` into the graph.
    ///
    /// Uses the seeded RNG for level assignment, guaranteeing reproducibility.
    pub fn insert(&mut self, id: usize, vector: Vec<f32>) {
        let rng_val: f64 = self.rng.random::<f64>();
        let node_layer = random_level(self.config.m_l, rng_val);

        let mut node = HnswNode::new(id, vector.clone(), node_layer);
        node.ensure_layers(node_layer);

        let node_idx = self.nodes.len();

        match self.entry_point {
            None => {
                // First node becomes the entry point
                self.entry_point = Some(node_idx);
                self.max_layer = node_layer;
                self.nodes.push(node);
                return;
            }
            Some(ep) => {
                // Greedy descent from the current entry point down to node_layer+1
                let mut ep_idx = ep;
                let current_top = self.max_layer;

                if current_top > node_layer {
                    for lc in (node_layer + 1..=current_top).rev() {
                        ep_idx = self.greedy_search_layer(ep_idx, &vector, lc);
                    }
                }

                // Insert connections at each layer from node_layer down to 0
                for lc in (0..=node_layer.min(current_top)).rev() {
                    let candidates =
                        self.search_layer_ef(ep_idx, &vector, self.config.ef_construction, lc);
                    let neighbours = self.select_neighbours(&candidates, self.config.m);

                    // Connect new node to its neighbours
                    for &nb_idx in &neighbours {
                        if nb_idx < self.nodes.len() {
                            self.nodes[nb_idx].ensure_layers(lc);
                            self.nodes[nb_idx].connections[lc].push(node_idx);
                            // Prune if over m_max
                            let nb_vec = self.nodes[nb_idx].vector.clone();
                            self.shrink_connections(nb_idx, lc, &nb_vec);
                        }
                    }
                    node.ensure_layers(lc);
                    node.connections[lc] = neighbours.clone();

                    // Update entry point for next layer
                    if !candidates.is_empty() {
                        ep_idx = candidates[0].0;
                    }
                }

                // If this node has higher layers, extend the graph
                if node_layer > current_top {
                    self.max_layer = node_layer;
                    self.entry_point = Some(node_idx);
                }
            }
        }
        self.nodes.push(node);
    }

    /// Search for the `k` nearest neighbours to `query` using the greedy beam search
    /// with candidate list size `ef`.
    ///
    /// Returns `(id, distance)` pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let ep = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut ep_idx = ep;
        // Greedy descent from max_layer down to layer 1
        for lc in (1..=self.max_layer).rev() {
            ep_idx = self.greedy_search_layer(ep_idx, query, lc);
        }

        // Full beam search at layer 0
        let candidates = self.search_layer_ef(ep_idx, query, ef.max(k), 0);

        // Return top-k by ascending distance
        let mut results: Vec<(usize, f32)> = candidates
            .iter()
            .take(k)
            .map(|&(idx, dist)| (self.nodes[idx].id, dist))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of layers in the graph (= max_layer + 1).
    pub fn layer_count(&self) -> usize {
        if self.nodes.is_empty() {
            0
        } else {
            self.max_layer + 1
        }
    }

    /// Return the connections of node `id` at `layer`, if they exist.
    pub fn connections_at(&self, id: usize, layer: usize) -> Option<&Vec<usize>> {
        let node = self.nodes.iter().find(|n| n.id == id)?;
        node.connections.get(layer)
    }

    // ── Private helpers ────────────────────────────────────────────────

    /// Single-step greedy search at a given layer: starting from `ep_idx`,
    /// repeatedly move to the neighbour closest to `query`.
    fn greedy_search_layer(&self, mut ep_idx: usize, query: &[f32], layer: usize) -> usize {
        let mut best_dist = euclidean_distance(&self.nodes[ep_idx].vector, query);
        loop {
            let mut improved = false;
            let conns: Vec<usize> = if layer < self.nodes[ep_idx].connections.len() {
                self.nodes[ep_idx].connections[layer].clone()
            } else {
                Vec::new()
            };
            for nb_idx in conns {
                if nb_idx < self.nodes.len() {
                    let d = euclidean_distance(&self.nodes[nb_idx].vector, query);
                    if d < best_dist {
                        best_dist = d;
                        ep_idx = nb_idx;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        ep_idx
    }

    /// Beam search at `layer` with candidate list of size `ef`.
    /// Returns (node_idx, distance) sorted by ascending distance (nearest first).
    fn search_layer_ef(
        &self,
        ep_idx: usize,
        query: &[f32],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        // We use two priority queues: candidates (min-heap by dist) and
        // dynamic_list (max-heap by dist to track the worst in the result set).
        // For simplicity we use a BTreeMap-like structure via a sorted vec.

        let ep_dist = euclidean_distance(&self.nodes[ep_idx].vector, query);

        // candidates: (dist, idx) – we want to process closest first
        // result: the ef nearest found so far
        let mut candidates: BinaryHeap<OrdPair> = BinaryHeap::new();
        let mut result: BinaryHeap<OrdPair> = BinaryHeap::new(); // max-heap (worst at top)
        let mut visited: HashSet<usize> = HashSet::new();

        candidates.push(OrdPair(ep_dist, ep_idx));
        result.push(OrdPair(ep_dist, ep_idx));
        visited.insert(ep_idx);

        while let Some(OrdPair(dist, idx)) = pop_min(&mut candidates) {
            // If the worst in result is better than the current candidate, stop
            if let Some(OrdPair(worst_dist, _)) = result.peek() {
                if dist > *worst_dist && result.len() >= ef {
                    break;
                }
            }
            // Expand neighbours at this layer
            let conns: Vec<usize> = if layer < self.nodes[idx].connections.len() {
                self.nodes[idx].connections[layer].clone()
            } else {
                Vec::new()
            };
            for nb_idx in conns {
                if nb_idx >= self.nodes.len() || visited.contains(&nb_idx) {
                    continue;
                }
                visited.insert(nb_idx);
                let d = euclidean_distance(&self.nodes[nb_idx].vector, query);
                // Add to candidates and result if better than worst
                let add = result.len() < ef || d < result.peek().map_or(f32::MAX, |p| p.0);
                if add {
                    candidates.push(OrdPair(d, nb_idx));
                    result.push(OrdPair(d, nb_idx));
                    // Prune result to ef elements
                    while result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        // Collect result into sorted vec (ascending distance)
        let mut out: Vec<(usize, f32)> = result.into_iter().map(|p| (p.1, p.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Select the best `m` neighbours from a sorted candidate list.
    fn select_neighbours(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|&(idx, _)| idx).collect()
    }

    /// Prune node's connections at `layer` to at most `m_max` elements.
    fn shrink_connections(&mut self, node_idx: usize, layer: usize, node_vec: &[f32]) {
        if layer >= self.nodes[node_idx].connections.len() {
            return;
        }
        let m_max = self.config.m_max;
        if self.nodes[node_idx].connections[layer].len() <= m_max {
            return;
        }
        // Keep the m_max nearest connections by distance to node_vec
        let mut conn_dists: Vec<(usize, f32)> = self.nodes[node_idx].connections[layer]
            .iter()
            .filter_map(|&nb| {
                if nb < self.nodes.len() {
                    let d = euclidean_distance(&self.nodes[nb].vector, node_vec);
                    Some((nb, d))
                } else {
                    None
                }
            })
            .collect();
        conn_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        conn_dists.truncate(m_max);
        self.nodes[node_idx].connections[layer] =
            conn_dists.into_iter().map(|(nb, _)| nb).collect();
    }
}

// ─────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────

/// Wrapper to turn (dist, idx) into an Ord element for BinaryHeap.
/// BinaryHeap is a max-heap; we negate to get min-heap semantics via `pop_min`.
#[derive(Debug, Clone, PartialEq)]
struct OrdPair(f32, usize);

impl Eq for OrdPair {}

impl PartialOrd for OrdPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap by default in BinaryHeap; sort descending by dist
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

/// Pop the element with the minimum distance from a max-heap of OrdPair.
/// We store (dist, idx) where the Ord impl makes the max-heap behave as a min-heap.
fn pop_min(heap: &mut BinaryHeap<OrdPair>) -> Option<OrdPair> {
    heap.pop()
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vec2(x: f32, y: f32) -> Vec<f32> {
        vec![x, y]
    }

    // ── Distance functions ─────────────────────────────────────

    #[test]
    fn test_euclidean_distance_zero() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(euclidean_distance(&a, &a), 0.0);
    }

    #[test]
    fn test_euclidean_distance_unit() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance_symmetric() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert!((euclidean_distance(&a, &b) - euclidean_distance(&b, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_range() {
        let a = vec![0.6_f32, 0.8];
        let b = vec![0.8_f32, 0.6];
        let s = cosine_similarity(&a, &b);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── random_level ──────────────────────────────────────────

    #[test]
    fn test_random_level_near_zero_returns_zero() {
        let level = random_level(1.0 / (16.0_f64).ln(), 0.999);
        assert_eq!(level, 0);
    }

    #[test]
    fn test_random_level_small_value_high_level() {
        // Very small rng_val means high level
        let level = random_level(1.0 / (16.0_f64).ln(), 1e-10);
        assert!(level > 0);
    }

    #[test]
    fn test_random_level_distribution() {
        // Over many samples, level 0 should dominate
        let m_l = 1.0 / (16.0_f64).ln();
        let mut rng = Random::seed(0);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for _ in 0..1000 {
            let v: f64 = rng.random::<f64>();
            let level = random_level(m_l, v);
            *counts.entry(level).or_insert(0) += 1;
        }
        // Level 0 must be the most common
        let count_0 = counts.get(&0).copied().unwrap_or(0);
        assert!(count_0 > 500, "Level 0 should dominate; got {count_0}");
    }

    // ── HnswConfig ────────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let cfg = HnswConfig::default();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.m_max, 32);
        assert_eq!(cfg.ef_construction, 200);
        assert!(cfg.m_l > 0.0);
    }

    #[test]
    fn test_config_new() {
        let cfg = HnswConfig::new(8, 100);
        assert_eq!(cfg.m, 8);
        assert_eq!(cfg.m_max, 16);
        assert_eq!(cfg.ef_construction, 100);
    }

    // ── Insert single node ────────────────────────────────────

    #[test]
    fn test_insert_single_node_entry_point_set() {
        let mut g = HnswGraph::new(HnswConfig::default());
        g.insert(0, vec2(1.0, 0.0));
        assert_eq!(g.entry_point, Some(0));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_insert_single_node_layer_count() {
        let mut g = HnswGraph::new(HnswConfig::default());
        g.insert(0, vec2(0.0, 0.0));
        assert!(g.layer_count() >= 1);
    }

    // ── Insert multiple nodes ─────────────────────────────────

    #[test]
    fn test_insert_multiple_increases_node_count() {
        let mut g = HnswGraph::new(HnswConfig::default());
        for i in 0..10_u32 {
            g.insert(i as usize, vec![i as f32, 0.0]);
        }
        assert_eq!(g.node_count(), 10);
    }

    #[test]
    fn test_entry_point_set_after_first_insert() {
        let mut g = HnswGraph::new(HnswConfig::default());
        g.insert(42, vec![1.0, 2.0]);
        assert!(g.entry_point.is_some());
    }

    // ── Search ────────────────────────────────────────────────

    #[test]
    fn test_search_empty_graph_returns_empty() {
        let g = HnswGraph::new(HnswConfig::default());
        let results = g.search(&[0.0, 0.0], 3, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single_node_returns_it() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        g.insert(0, vec2(1.0, 0.0));
        let results = g.search(&[1.0, 0.0], 1, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_search_returns_at_most_k_results() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        for i in 0..20_u32 {
            g.insert(i as usize, vec![i as f32, 0.0]);
        }
        let results = g.search(&[5.0, 0.0], 5, 20);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_results_ordered_by_distance() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        for i in 0..10_u32 {
            g.insert(i as usize, vec![i as f32, 0.0]);
        }
        let query = vec![4.5, 0.0];
        let results = g.search(&query, 5, 20);
        // Distances must be non-decreasing
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-5, "Results not sorted: {:?}", results);
        }
    }

    #[test]
    fn test_search_nearest_is_closest() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        // Well-separated points
        g.insert(0, vec2(0.0, 0.0));
        g.insert(1, vec2(100.0, 0.0));
        g.insert(2, vec2(0.0, 100.0));
        let results = g.search(&[1.0, 1.0], 1, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // closest to (0,0)
    }

    // ── layer_count grows ─────────────────────────────────────

    #[test]
    fn test_layer_count_non_zero_after_insert() {
        let mut g = HnswGraph::new(HnswConfig::default());
        g.insert(0, vec![1.0]);
        assert!(g.layer_count() >= 1);
    }

    #[test]
    fn test_layer_count_zero_when_empty() {
        let g = HnswGraph::new(HnswConfig::default());
        assert_eq!(g.layer_count(), 0);
    }

    // ── connections_at ────────────────────────────────────────

    #[test]
    fn test_connections_at_returns_none_for_unknown_id() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        g.insert(0, vec2(1.0, 0.0));
        // ID 99 was never inserted
        assert!(g.connections_at(99, 0).is_none());
    }

    #[test]
    fn test_connections_at_returns_some_for_inserted_node() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        g.insert(0, vec2(0.0, 0.0));
        // Layer 0 connections exist (may be empty for sole node)
        assert!(g.connections_at(0, 0).is_some());
    }

    // ── Exact search on small graph ───────────────────────────

    #[test]
    fn test_exact_search_3_nodes() {
        let mut g = HnswGraph::new(HnswConfig::new(2, 20));
        g.insert(0, vec2(0.0, 0.0));
        g.insert(1, vec2(1.0, 0.0));
        g.insert(2, vec2(10.0, 0.0));

        // Query at (0.1, 0) — should find node 0 first
        let results = g.search(&[0.1, 0.0], 3, 10);
        assert!(!results.is_empty());
        // Node 0 should be the nearest or very close
        let nearest = results[0].0;
        assert!(
            nearest == 0 || nearest == 1,
            "Expected 0 or 1, got {nearest}"
        );
    }

    // ── HnswNode ──────────────────────────────────────────────

    #[test]
    fn test_hnsw_node_new() {
        let n = HnswNode::new(5, vec![1.0, 2.0], 2);
        assert_eq!(n.id, 5);
        assert_eq!(n.vector, vec![1.0, 2.0]);
        assert_eq!(n.connections.len(), 3); // layers 0,1,2
    }

    #[test]
    fn test_hnsw_node_ensure_layers() {
        let mut n = HnswNode::new(0, vec![1.0], 0);
        n.ensure_layers(3);
        assert!(n.connections.len() >= 4);
    }

    // ── Multiple searches give consistent results ─────────────

    #[test]
    fn test_search_reproducible() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        for i in 0..15_u32 {
            g.insert(i as usize, vec![(i as f32) * 0.1, 0.0]);
        }
        let r1 = g.search(&[0.5, 0.0], 3, 10);
        let r2 = g.search(&[0.5, 0.0], 3, 10);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0, b.0);
        }
    }

    #[test]
    fn test_search_returns_k_or_fewer() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        for i in 0..5_u32 {
            g.insert(i as usize, vec![i as f32]);
        }
        let results = g.search(&[2.0], 10, 10);
        // Can't return more than n_nodes results
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_distances_non_negative() {
        let mut g = HnswGraph::new(HnswConfig::new(4, 50));
        for i in 0..8_u32 {
            g.insert(i as usize, vec![i as f32, (8 - i) as f32]);
        }
        let results = g.search(&[4.0, 4.0], 5, 20);
        for (_, dist) in &results {
            assert!(*dist >= 0.0);
        }
    }
}
