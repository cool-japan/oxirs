//! GPU-simulated HNSW index with parallel insert and fast approximate search.
//!
//! This module provides `GpuHnswIndex`, which simulates GPU-accelerated HNSW
//! graph construction using CPU-side parallelism (via `std::thread`) as a
//! pure-Rust stand-in for actual GPU batching.  The API intentionally mirrors
//! a real GPU implementation so that the caller can be swapped for a CUDA
//! version later without interface changes.
//!
//! # Design
//!
//! * **Simulated GPU batching**: vectors are accumulated in a staging batch;
//!   when the batch is full the entire batch is "uploaded" (simulated) and
//!   graph edges are computed in parallel across batch items.
//! * **Layered graph**: a standard HNSW multi-layer graph where each node
//!   stores at most `max_connections` bi-directional edges per layer.
//! * **Approximate search**: greedy beam search starting from the entry point.
//! * **No `unwrap()`**: all fallible operations propagate `anyhow::Error`.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

// ── configuration ─────────────────────────────────────────────────────────────

/// Configuration for `GpuHnswIndex`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHnswConfig {
    /// Simulated GPU batch size (number of vectors per "GPU upload").
    pub batch_size: usize,
    /// Maximum connections per layer (M parameter).
    pub max_connections: usize,
    /// Maximum connections at layer 0 (M0 parameter, typically 2×M).
    pub max_connections_layer0: usize,
    /// ef_construction: candidate list size during graph construction.
    pub ef_construction: usize,
    /// ef_search: candidate list size during approximate search.
    pub ef_search: usize,
    /// Layer probability multiplier (1/ln(M)).
    pub level_multiplier: f64,
    /// Number of simulated GPU worker threads for batch construction.
    pub gpu_workers: usize,
}

impl Default for GpuHnswConfig {
    fn default() -> Self {
        Self {
            batch_size: 64,
            max_connections: 16,
            max_connections_layer0: 32,
            ef_construction: 200,
            ef_search: 50,
            level_multiplier: 1.0 / (16_f64).ln(),
            gpu_workers: 4,
        }
    }
}

// ── graph node ────────────────────────────────────────────────────────────────

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
struct HnswNode {
    /// The raw floating-point vector.
    vector: Vec<f32>,
    /// `neighbors[layer]` holds the neighbor IDs for that layer.
    neighbors: Vec<Vec<usize>>,
    /// Maximum layer this node appears in.
    max_layer: usize,
}

impl HnswNode {
    fn new(vector: Vec<f32>, max_layer: usize, layers: usize) -> Self {
        Self {
            vector,
            neighbors: vec![Vec::new(); layers],
            max_layer,
        }
    }
}

// ── candidate / search helpers ────────────────────────────────────────────────

/// A (distance, node_id) pair ordered for a max-heap (farthest first).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    dist: f32,
    id: usize,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap on distance (farthest first)
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── GPU batch stats ───────────────────────────────────────────────────────────

/// Statistics collected during GPU-simulated batch construction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuBatchStats {
    /// Total number of batches processed.
    pub batches_processed: u64,
    /// Total vectors inserted.
    pub vectors_inserted: u64,
    /// Total distance computations performed.
    pub distance_computations: u64,
    /// Average batch processing time in microseconds (simulated).
    pub avg_batch_us: f64,
}

// ── index stats ───────────────────────────────────────────────────────────────

/// Overall index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHnswStats {
    /// Number of vectors in the index.
    pub vector_count: usize,
    /// Number of layers in the graph.
    pub layer_count: usize,
    /// GPU batch statistics.
    pub batch_stats: GpuBatchStats,
    /// Configured batch size.
    pub batch_size: usize,
    /// ef_search parameter.
    pub ef_search: usize,
}

// ── main struct ───────────────────────────────────────────────────────────────

/// GPU-simulated HNSW index.
///
/// Inserts are accumulated in a staging buffer; once the buffer reaches
/// `config.batch_size` the batch is flushed via parallel construction
/// threads (simulating GPU parallelism).
pub struct GpuHnswIndex {
    config: GpuHnswConfig,
    /// All nodes keyed by numeric ID.
    nodes: Vec<HnswNode>,
    /// URI → node ID.
    uri_to_id: HashMap<String, usize>,
    /// Node ID → URI.
    id_to_uri: Vec<String>,
    /// Entry point into the top layer.
    entry_point: Option<usize>,
    /// Current top layer in the graph.
    top_layer: usize,
    /// Staging batch: (uri, vector) pairs waiting to be flushed.
    pending_batch: Vec<(String, Vec<f32>)>,
    /// Accumulated statistics.
    batch_stats: GpuBatchStats,
    /// Simple LCG RNG state for deterministic level generation.
    rng_state: u64,
}

impl GpuHnswIndex {
    /// Create a new GPU-simulated HNSW index.
    pub fn new(config: GpuHnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            uri_to_id: HashMap::new(),
            id_to_uri: Vec::new(),
            entry_point: None,
            top_layer: 0,
            pending_batch: Vec::new(),
            batch_stats: GpuBatchStats::default(),
            rng_state: 0x9e3779b97f4a7c15,
        }
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Insert a vector into the index.
    ///
    /// The vector is first placed into the staging batch.  When the batch
    /// reaches `config.batch_size` it is flushed automatically.
    pub fn insert(&mut self, uri: String, vector: Vec<f32>) -> Result<()> {
        if self.uri_to_id.contains_key(&uri) {
            return Err(anyhow!("URI '{}' already exists in index", uri));
        }
        self.pending_batch.push((uri, vector));
        if self.pending_batch.len() >= self.config.batch_size {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Force-flush the pending staging batch regardless of its size.
    pub fn flush(&mut self) -> Result<()> {
        if !self.pending_batch.is_empty() {
            self.flush_batch()?;
        }
        Ok(())
    }

    /// Search for the `k` approximate nearest neighbours of `query`.
    ///
    /// Any unflushed vectors in the staging batch are **not** searched.
    /// Call `flush` first if you need them included.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let entry = self
            .entry_point
            .ok_or_else(|| anyhow!("No entry point set"))?;

        // Greedy descent through upper layers
        let mut current_nearest = entry;
        for layer in (1..=self.top_layer).rev() {
            current_nearest = self.greedy_search_layer(query, current_nearest, layer)?;
        }

        // Beam search at layer 0
        let candidates =
            self.beam_search_layer(query, current_nearest, 0, self.config.ef_search)?;

        // Collect top-k
        let mut results: Vec<(String, f32)> = candidates
            .into_iter()
            .map(|c| {
                let uri = self.id_to_uri[c.id].clone();
                (uri, c.dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    /// Number of vectors currently committed to the graph (excludes pending batch).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Number of vectors in the pending (un-flushed) batch.
    pub fn pending_count(&self) -> usize {
        self.pending_batch.len()
    }

    /// Returns `true` if the graph contains no committed nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return a snapshot of current statistics.
    pub fn stats(&self) -> GpuHnswStats {
        GpuHnswStats {
            vector_count: self.nodes.len(),
            layer_count: self.top_layer + 1,
            batch_stats: self.batch_stats.clone(),
            batch_size: self.config.batch_size,
            ef_search: self.config.ef_search,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &GpuHnswConfig {
        &self.config
    }

    // ── private internals ─────────────────────────────────────────────────────

    /// Flush the pending staging batch by constructing graph edges in parallel.
    ///
    /// We simulate GPU batching by distributing the level-assignment step
    /// (embarrassingly parallel) across `gpu_workers` threads, then serially
    /// inserting each node into the graph (graph mutation requires the global
    /// state so cannot be done in parallel without complex locking).
    fn flush_batch(&mut self) -> Result<()> {
        let batch = std::mem::take(&mut self.pending_batch);
        let batch_len = batch.len();

        // ── Simulate GPU batch: compute random levels in parallel ─────────
        let workers = self.config.gpu_workers.max(1);
        let level_multiplier = self.config.level_multiplier;

        // Share seeds for parallel workers
        let seeds: Vec<u64> = (0..batch_len)
            .map(|i| {
                let mut s = self
                    .rng_state
                    .wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15));
                // xorshift64
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            })
            .collect();

        // Assign one seed to rng_state for next call
        self.rng_state = seeds.last().copied().unwrap_or(self.rng_state);

        // Parallel level computation (simulate GPU kernel)
        let levels_shared: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(vec![0usize; batch_len]));
        let chunk_size = (batch_len + workers - 1) / workers;

        std::thread::scope(|scope| {
            let seeds_ref = &seeds;
            let levels_ref = Arc::clone(&levels_shared);
            for worker_id in 0..workers {
                let start = worker_id * chunk_size;
                let end = (start + chunk_size).min(batch_len);
                if start >= end {
                    break;
                }
                let lm = level_multiplier;
                let levels_clone = Arc::clone(&levels_ref);
                scope.spawn(move || {
                    let mut local_results = Vec::with_capacity(end - start);
                    for (i, &seed) in seeds_ref.iter().enumerate().skip(start).take(end - start) {
                        // Use the seed to generate a uniform float in (0, 1)
                        let uniform = (seed >> 11) as f64 / (u64::MAX >> 11) as f64;
                        // HNSW level = floor(-ln(uniform) * level_multiplier)
                        let level = if uniform > 0.0 {
                            (-uniform.ln() * lm).floor() as usize
                        } else {
                            0
                        };
                        local_results.push((i, level));
                    }
                    // Write back
                    if let Ok(mut guard) = levels_clone.lock() {
                        for (idx, lvl) in local_results {
                            guard[idx] = lvl;
                        }
                    }
                });
            }
        });

        let levels = Arc::try_unwrap(levels_shared)
            .map_err(|_| anyhow!("Arc unwrap failed"))?
            .into_inner()
            .map_err(|e| anyhow!("Mutex poisoned: {e}"))?;

        // ── Serial graph construction ──────────────────────────────────────
        let mut dist_count = 0u64;
        for (item_idx, (uri, vector)) in batch.into_iter().enumerate() {
            let node_level = levels[item_idx];
            let node_id = self.nodes.len();
            let layer_count = node_level + 1;

            // Pre-allocate layers (extend top_layer if needed)
            let total_layers = self.top_layer.max(node_level) + 1;
            let new_node = HnswNode::new(vector.clone(), node_level, total_layers);
            self.nodes.push(new_node);
            self.uri_to_id.insert(uri.clone(), node_id);
            self.id_to_uri.push(uri);

            if let Some(ep) = self.entry_point {
                // Extend existing nodes' neighbor lists if necessary
                let current_max = self.top_layer;
                if node_level > current_max {
                    // Extend all existing node neighbor vecs
                    for n in &mut self.nodes {
                        let extra = (node_level + 1).saturating_sub(n.neighbors.len());
                        n.neighbors
                            .extend(std::iter::repeat_with(Vec::new).take(extra));
                    }
                    self.top_layer = node_level;
                }

                // Greedy descent through layers above node_level
                let mut current_ep = ep;
                for layer in (layer_count..=self.top_layer).rev() {
                    current_ep =
                        self.greedy_search_layer_mut(&vector, current_ep, layer, &mut dist_count)?;
                }

                // For each layer from min(node_level, top_layer) down to 0
                let max_conns = self.config.max_connections;
                let max_conns_l0 = self.config.max_connections_layer0;

                for layer in (0..layer_count).rev() {
                    let ef = self.config.ef_construction;
                    let candidates = self.beam_search_layer_with_count(
                        &vector,
                        current_ep,
                        layer,
                        ef,
                        &mut dist_count,
                    )?;

                    // Pick the best neighbors (simple select-n)
                    let m = if layer == 0 { max_conns_l0 } else { max_conns };
                    let selected: Vec<usize> = candidates.iter().take(m).map(|c| c.id).collect();

                    // Add bidirectional edges
                    self.nodes[node_id].neighbors[layer].extend_from_slice(&selected);

                    for &neighbor_id in &selected {
                        // Prune neighbor's list if over capacity
                        self.nodes[neighbor_id].neighbors[layer].push(node_id);
                        let cap = if layer == 0 { max_conns_l0 } else { max_conns };
                        self.prune_connections(neighbor_id, layer, cap);
                    }

                    // Update ep for next layer
                    if let Some(best) = candidates.first() {
                        current_ep = best.id;
                    }
                }

                if node_level > current_max {
                    self.entry_point = Some(node_id);
                }
            } else {
                // First node — just set as entry point.
                // Ensure neighbor vecs have the right length.
                let total = self.top_layer.max(node_level) + 1;
                let extra = total.saturating_sub(self.nodes[node_id].neighbors.len());
                self.nodes[node_id]
                    .neighbors
                    .extend(std::iter::repeat_with(Vec::new).take(extra));
                self.top_layer = node_level;
                self.entry_point = Some(node_id);
            }
        }

        // Update stats
        let time_us = (batch_len as f64 * 12.5) + 100.0; // Simulated GPU time
        let prev_batches = self.batch_stats.batches_processed as f64;
        let new_avg = if prev_batches > 0.0 {
            (self.batch_stats.avg_batch_us * prev_batches + time_us) / (prev_batches + 1.0)
        } else {
            time_us
        };

        self.batch_stats.batches_processed += 1;
        self.batch_stats.vectors_inserted += batch_len as u64;
        self.batch_stats.distance_computations += dist_count;
        self.batch_stats.avg_batch_us = new_avg;

        Ok(())
    }

    /// Greedy single-hop search at `layer` (used during graph construction descent).
    fn greedy_search_layer(&self, query: &[f32], entry: usize, layer: usize) -> Result<usize> {
        let mut current = entry;
        let mut current_dist = self.euclidean_sq(query, &self.nodes[current].vector);

        loop {
            let mut improved = false;
            for &neighbor in &self.nodes[current].neighbors[layer] {
                if neighbor >= self.nodes.len() {
                    continue;
                }
                let d = self.euclidean_sq(query, &self.nodes[neighbor].vector);
                if d < current_dist {
                    current_dist = d;
                    current = neighbor;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        Ok(current)
    }

    /// Greedy search during graph construction (mutable, tracks distance count).
    fn greedy_search_layer_mut(
        &self,
        query: &[f32],
        entry: usize,
        layer: usize,
        dist_count: &mut u64,
    ) -> Result<usize> {
        let mut current = entry;
        *dist_count += 1;
        let mut current_dist = self.euclidean_sq(query, &self.nodes[current].vector);

        loop {
            let mut improved = false;
            let neighbors = self.nodes[current].neighbors[layer].clone();
            for neighbor in neighbors {
                if neighbor >= self.nodes.len() {
                    continue;
                }
                *dist_count += 1;
                let d = self.euclidean_sq(query, &self.nodes[neighbor].vector);
                if d < current_dist {
                    current_dist = d;
                    current = neighbor;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        Ok(current)
    }

    /// Beam (ef) search at a specific layer — returns ordered candidate list (closest first).
    fn beam_search_layer(
        &self,
        query: &[f32],
        entry: usize,
        layer: usize,
        ef: usize,
    ) -> Result<Vec<Candidate>> {
        let mut dummy = 0u64;
        self.beam_search_layer_with_count(query, entry, layer, ef, &mut dummy)
    }

    /// Beam search with distance counter (used during construction).
    fn beam_search_layer_with_count(
        &self,
        query: &[f32],
        entry: usize,
        layer: usize,
        ef: usize,
        dist_count: &mut u64,
    ) -> Result<Vec<Candidate>> {
        if entry >= self.nodes.len() {
            return Ok(Vec::new());
        }

        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(entry);

        *dist_count += 1;
        let d_entry = self.euclidean_sq(query, &self.nodes[entry].vector);

        // candidates = max-heap (farthest first, for pruning)
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        // to_visit = min-heap (closest first, for expansion)
        let mut to_visit: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();

        candidates.push(Candidate {
            dist: d_entry,
            id: entry,
        });
        to_visit.push(std::cmp::Reverse(Candidate {
            dist: d_entry,
            id: entry,
        }));

        while let Some(std::cmp::Reverse(current)) = to_visit.pop() {
            // Terminate if current candidate is farther than worst in result set
            if let Some(worst) = candidates.peek() {
                if current.dist > worst.dist {
                    break;
                }
            }

            let neighbors = if layer < self.nodes[current.id].neighbors.len() {
                self.nodes[current.id].neighbors[layer].clone()
            } else {
                Vec::new()
            };

            for neighbor in neighbors {
                if neighbor >= self.nodes.len() || visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                *dist_count += 1;
                let d = self.euclidean_sq(query, &self.nodes[neighbor].vector);
                let worst_dist = candidates.peek().map(|c| c.dist).unwrap_or(f32::MAX);

                if d < worst_dist || candidates.len() < ef {
                    candidates.push(Candidate {
                        dist: d,
                        id: neighbor,
                    });
                    to_visit.push(std::cmp::Reverse(Candidate {
                        dist: d,
                        id: neighbor,
                    }));
                    if candidates.len() > ef {
                        candidates.pop();
                    }
                }
            }
        }

        // Convert to sorted vec (closest first)
        let mut result: Vec<Candidate> = candidates.into_vec();
        result.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(result)
    }

    /// Prune a node's neighbor list at `layer` to at most `cap` entries.
    fn prune_connections(&mut self, node_id: usize, layer: usize, cap: usize) {
        if self.nodes[node_id].neighbors[layer].len() > cap {
            // Collect with distances from node's own vector
            let node_vec = self.nodes[node_id].vector.clone();
            let mut with_dist: Vec<(usize, f32)> = self.nodes[node_id].neighbors[layer]
                .iter()
                .filter(|&&n| n < self.nodes.len())
                .map(|&n| (n, self.euclidean_sq(&node_vec, &self.nodes[n].vector)))
                .collect();
            with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            with_dist.truncate(cap);
            self.nodes[node_id].neighbors[layer] = with_dist.into_iter().map(|(n, _)| n).collect();
        }
    }

    /// Squared Euclidean distance (used as distance metric; no sqrt needed for ordering).
    #[inline]
    fn euclidean_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn make_index(batch_size: usize) -> GpuHnswIndex {
        let config = GpuHnswConfig {
            batch_size,
            max_connections: 8,
            max_connections_layer0: 16,
            ef_construction: 20,
            ef_search: 16,
            gpu_workers: 2,
            ..Default::default()
        };
        GpuHnswIndex::new(config)
    }

    fn vec2(x: f32, y: f32) -> Vec<f32> {
        vec![x, y]
    }

    // ── basic functionality ────────────────────────────────────────────────

    #[test]
    fn test_new_index_is_empty() {
        let index = make_index(4);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.pending_count(), 0);
    }

    #[test]
    fn test_insert_pending_accumulates() -> Result<()> {
        let mut index = make_index(8);
        index.insert("a".to_string(), vec2(1.0, 0.0))?;
        index.insert("b".to_string(), vec2(0.0, 1.0))?;
        assert_eq!(index.pending_count(), 2);
        assert_eq!(index.len(), 0); // Not yet flushed
        Ok(())
    }

    #[test]
    fn test_auto_flush_on_batch_full() -> Result<()> {
        let mut index = make_index(3);
        for i in 0..3 {
            index.insert(format!("v{}", i), vec![i as f32, 0.0])?;
        }
        // Batch of 3 triggers auto-flush
        assert_eq!(index.len(), 3);
        assert_eq!(index.pending_count(), 0);
        Ok(())
    }

    #[test]
    fn test_manual_flush() -> Result<()> {
        let mut index = make_index(16);
        index.insert("x".to_string(), vec2(1.0, 1.0))?;
        assert_eq!(index.pending_count(), 1);
        index.flush()?;
        assert_eq!(index.len(), 1);
        assert_eq!(index.pending_count(), 0);
        Ok(())
    }

    #[test]
    fn test_search_empty_returns_empty() -> Result<()> {
        let index = make_index(4);
        let result = index.search(&[1.0, 0.0], 5)?;
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_search_single_vector() -> Result<()> {
        let mut index = make_index(4);
        index.insert("only".to_string(), vec2(1.0, 0.0))?;
        index.flush()?;
        let result = index.search(&[1.0, 0.0], 1)?;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "only");
        Ok(())
    }

    #[test]
    fn test_search_nearest_neighbour() -> Result<()> {
        let mut index = make_index(8);
        index.insert("origin".to_string(), vec2(0.0, 0.0))?;
        index.insert("right".to_string(), vec2(10.0, 0.0))?;
        index.insert("up".to_string(), vec2(0.0, 10.0))?;
        index.flush()?;

        // Query near origin
        let result = index.search(&[0.1, 0.1], 1)?;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "origin");
        Ok(())
    }

    #[test]
    fn test_search_top_k_ordering() -> Result<()> {
        let mut index = make_index(4);
        for i in 0..4 {
            index.insert(format!("v{}", i), vec![i as f32 * 2.0, 0.0])?;
        }
        index.flush()?;

        let result = index.search(&[0.0, 0.0], 2)?;
        assert!(result.len() <= 2);
        // Closest should come first
        if result.len() == 2 {
            assert!(
                result[0].1 <= result[1].1,
                "Results should be ordered by distance"
            );
        }
        Ok(())
    }

    #[test]
    fn test_duplicate_uri_rejected() -> Result<()> {
        let mut index = make_index(8);
        index.insert("dup".to_string(), vec2(1.0, 0.0))?;
        index.flush()?;
        let err = index.insert("dup".to_string(), vec2(2.0, 0.0));
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn test_stats_accumulate() -> Result<()> {
        let mut index = make_index(4);
        for i in 0..8 {
            index.insert(format!("v{}", i), vec![i as f32, 0.0])?;
        }
        index.flush()?; // flush any remainder
        let stats = index.stats();
        assert_eq!(stats.vector_count, 8);
        assert!(stats.batch_stats.batches_processed >= 2);
        assert_eq!(stats.batch_stats.vectors_inserted, 8);
        Ok(())
    }

    #[test]
    fn test_stats_avg_batch_time_positive() -> Result<()> {
        let mut index = make_index(2);
        index.insert("a".to_string(), vec2(0.0, 0.0))?;
        index.insert("b".to_string(), vec2(1.0, 0.0))?;
        let stats = index.stats();
        assert!(stats.batch_stats.avg_batch_us > 0.0);
        Ok(())
    }

    #[test]
    fn test_larger_dataset_correctness() -> Result<()> {
        let mut index = make_index(10);
        // Add 50 vectors in a line along x-axis
        for i in 0..50 {
            index.insert(format!("v{}", i), vec![i as f32, 0.0])?;
        }
        index.flush()?;

        assert_eq!(index.len(), 50);

        // Nearest to x=25 should be v25
        let result = index.search(&[25.0, 0.0], 3)?;
        assert!(!result.is_empty());
        // The closest vector should be very close to 25.0
        assert!(result[0].1 < 2.0_f32);
        Ok(())
    }

    #[test]
    fn test_multi_batch_flush_consistency() -> Result<()> {
        let mut index = make_index(5);
        for i in 0..20 {
            index.insert(format!("v{}", i), vec![i as f32, (i % 3) as f32])?;
        }
        index.flush()?;
        let stats = index.stats();
        assert_eq!(stats.vector_count, 20);
        assert!(stats.batch_stats.batches_processed >= 4);
        Ok(())
    }

    #[test]
    fn test_config_accessors() {
        let config = GpuHnswConfig {
            batch_size: 32,
            max_connections: 12,
            ..Default::default()
        };
        let index = GpuHnswIndex::new(config);
        assert_eq!(index.config().batch_size, 32);
        assert_eq!(index.config().max_connections, 12);
    }

    #[test]
    fn test_gpu_workers_default() {
        let config = GpuHnswConfig::default();
        assert_eq!(config.gpu_workers, 4);
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_single_dimension_vectors() -> Result<()> {
        let mut index = make_index(4);
        index.insert("a".to_string(), vec![1.0])?;
        index.insert("b".to_string(), vec![5.0])?;
        index.insert("c".to_string(), vec![10.0])?;
        index.insert("d".to_string(), vec![3.0])?;
        index.flush()?;
        let result = index.search(&[4.5], 2)?;
        assert!(!result.is_empty());
        Ok(())
    }

    #[test]
    fn test_high_dimensional_vectors() -> Result<()> {
        let dim = 128;
        let mut index = make_index(8);
        for i in 0..16 {
            let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 * 0.01).collect();
            index.insert(format!("v{}", i), v)?;
        }
        index.flush()?;
        let query: Vec<f32> = (0..dim).map(|d| d as f32 * 0.01).collect();
        let result = index.search(&query, 3)?;
        assert!(!result.is_empty());
        assert_eq!(result[0].0, "v0"); // v0 is at 0..dim * 0.01
        Ok(())
    }

    #[test]
    fn test_search_returns_at_most_k() -> Result<()> {
        let mut index = make_index(4);
        for i in 0..10 {
            index.insert(format!("v{}", i), vec![i as f32])?;
        }
        index.flush()?;
        let result = index.search(&[5.0], 3)?;
        assert!(result.len() <= 3);
        Ok(())
    }

    #[test]
    fn test_distance_computations_counted() -> Result<()> {
        let mut index = make_index(4);
        for i in 0..8 {
            index.insert(format!("v{}", i), vec![i as f32, 0.0])?;
        }
        index.flush()?;
        let stats = index.stats();
        // Some distance computations should have occurred during construction
        assert!(stats.batch_stats.distance_computations > 0);
        Ok(())
    }

    #[test]
    fn test_pending_not_searched() -> Result<()> {
        let mut index = make_index(100); // Large batch so nothing auto-flushes
        index.insert("pending".to_string(), vec2(0.0, 0.0))?;
        // pending_count = 1, len = 0
        assert_eq!(index.pending_count(), 1);
        assert_eq!(index.len(), 0);
        // Search on empty committed graph
        let result = index.search(&[0.0, 0.0], 1)?;
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_flush_empty_pending_noop() -> Result<()> {
        let mut index = make_index(4);
        index.insert("a".to_string(), vec2(1.0, 0.0))?;
        index.flush()?;
        // Second flush on empty pending
        index.flush()?;
        assert_eq!(index.len(), 1);
        Ok(())
    }

    #[test]
    fn test_layer_count_in_stats() -> Result<()> {
        let mut index = make_index(4);
        index.insert("a".to_string(), vec2(0.0, 0.0))?;
        index.flush()?;
        let stats = index.stats();
        assert!(stats.layer_count >= 1);
        Ok(())
    }
}
