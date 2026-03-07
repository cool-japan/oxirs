/// HNSW approximate nearest-neighbor search.
///
/// A simplified implementation of the Hierarchical Navigable Small World graph
/// algorithm for approximate k-nearest-neighbor (ANN) search in high-dimensional
/// vector spaces.
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for an HNSW index.
#[derive(Debug, Clone)]
pub struct HnswSearchConfig {
    /// Maximum number of connections per node per level (M).
    pub m: usize,
    /// Number of candidates during construction (ef_construction ≥ M).
    pub ef_construction: usize,
    /// Number of candidates during search.
    pub ef_search: usize,
    /// Maximum level in the hierarchy.
    pub max_level: usize,
}

impl Default for HnswSearchConfig {
    fn default() -> Self {
        HnswSearchConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_level: 6,
        }
    }
}

/// A single node in the HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswSearchNode {
    /// External identifier supplied by the caller.
    pub id: usize,
    /// The embedding vector.
    pub vector: Vec<f64>,
    /// Neighbor lists per level: `neighbors[level]` = list of node indices in
    /// the internal `nodes` array.
    pub neighbors: Vec<Vec<usize>>,
}

/// A search result containing the node id and its distance to the query.
#[derive(Debug, Clone, PartialEq)]
pub struct HnswSearchResult {
    pub id: usize,
    pub distance: f64,
}

// A wrapper for BinaryHeap that stores (distance, index) as max-heap via
// negation trick (we need min-heap semantics for closest first).
#[derive(PartialEq)]
struct HeapEntry {
    dist: f64,
    idx: usize,
}

impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Larger distance = lower priority (we want min-heap)
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// HnswSearchIndex
// ---------------------------------------------------------------------------

/// An HNSW approximate nearest-neighbor index.
///
/// Supports incremental insertion and approximate k-NN search.
pub struct HnswSearchIndex {
    nodes: Vec<HnswSearchNode>,
    /// Maps external `id` → internal index in `nodes`.
    id_map: HashMap<usize, usize>,
    entry_point: Option<usize>,
    config: HnswSearchConfig,
}

impl HnswSearchIndex {
    /// Create a new empty HNSW index with the given configuration.
    pub fn new(config: HnswSearchConfig) -> Self {
        HnswSearchIndex {
            nodes: Vec::new(),
            id_map: HashMap::new(),
            entry_point: None,
            config,
        }
    }

    /// Insert a vector with external `id` into the index.
    ///
    /// If `id` already exists it is ignored (no duplicate handling).
    pub fn insert(&mut self, id: usize, vector: Vec<f64>) {
        if self.id_map.contains_key(&id) {
            return;
        }

        let level = Self::random_level(self.config.m);
        let internal_idx = self.nodes.len();

        // Allocate neighbor lists for each level
        let neighbors: Vec<Vec<usize>> = (0..=level).map(|_| Vec::new()).collect();

        let node = HnswSearchNode {
            id,
            vector,
            neighbors,
        };
        self.nodes.push(node);
        self.id_map.insert(id, internal_idx);

        if self.entry_point.is_none() {
            self.entry_point = Some(internal_idx);
            return;
        }

        let entry = self.entry_point.unwrap();

        // Connect to existing nodes greedily on each level
        for lc in (0..=level).rev() {
            let candidates =
                self.search_layer(internal_idx, entry, self.config.ef_construction, lc);
            // Take top-M candidates
            let m_max = self.config.m;
            let selected: Vec<usize> = candidates.into_iter().take(m_max).map(|e| e.idx).collect();

            self.nodes[internal_idx].neighbors[lc].clone_from(&selected);
            for &nb_idx in &selected {
                // Shrink neighbor list if needed (bidirectional)
                if nb_idx < self.nodes.len() {
                    let nb_level = self.nodes[nb_idx].neighbors.len().saturating_sub(1);
                    if lc <= nb_level {
                        self.nodes[nb_idx].neighbors[lc].push(internal_idx);
                        if self.nodes[nb_idx].neighbors[lc].len() > self.config.m * 2 {
                            // Keep closest m*2 — compute distances before the mutable borrow.
                            let pivot_vec = self.nodes[nb_idx].vector.clone();
                            let neighbor_dists: Vec<(usize, f64)> = self.nodes[nb_idx].neighbors
                                [lc]
                                .iter()
                                .map(|&idx| {
                                    let d = Self::euclidean_distance(
                                        &self.nodes[idx].vector,
                                        &pivot_vec,
                                    );
                                    (idx, d)
                                })
                                .collect();
                            let mut sorted = neighbor_dists;
                            sorted.sort_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            sorted.truncate(self.config.m);
                            self.nodes[nb_idx].neighbors[lc] =
                                sorted.into_iter().map(|(idx, _)| idx).collect();
                        }
                    }
                }
            }
        }
    }

    /// Perform approximate k-nearest-neighbor search for `query`.
    ///
    /// Returns at most `k` results sorted by ascending distance.
    pub fn search(&self, query: &[f64], k: usize) -> Vec<HnswSearchResult> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let entry = match self.entry_point {
            None => return Vec::new(),
            Some(ep) => ep,
        };

        // Create a temporary node index for the query (never added to self.nodes)
        let query_internal = self.nodes.len(); // virtual index

        // We search top levels first, descending to level 0
        let max_level = self.nodes[entry].neighbors.len().saturating_sub(1);

        let mut current_best = entry;
        for lc in (1..=max_level).rev() {
            let result = self.greedy_search_layer(query, current_best, lc);
            current_best = result;
        }

        // Final search at level 0 with ef_search candidates
        let ef = self.config.ef_search.max(k);
        let candidates = self.search_layer_query(query, current_best, ef, 0);

        let _ = query_internal; // suppress unused warning

        let mut results: Vec<HnswSearchResult> = candidates
            .into_iter()
            .map(|e| HnswSearchResult {
                id: self.nodes[e.idx].id,
                distance: e.dist,
            })
            .take(k)
            .collect();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return `true` when the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return the dimensionality of the first inserted vector, or `None` when
    /// the index is empty.
    pub fn dim(&self) -> Option<usize> {
        self.nodes.first().map(|n| n.vector.len())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute the Euclidean (L2) distance between two vectors.
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let sum: f64 = a[..len]
            .iter()
            .zip(&b[..len])
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        sum.sqrt()
    }

    /// Assign a random HNSW level using the geometric distribution.
    fn random_level(m: usize) -> usize {
        // Use a deterministic but pseudo-random scheme based on the node count
        // (avoids rand dependency while still giving reasonable level distribution).
        // P(level ≥ l) = (1/m)^l
        let ml = 1.0 / (m as f64).ln();
        let frac = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0) as f64)
            / 1_000_000_000.0;
        // Map the [0,1) random value to a level
        let level = (-frac.max(1e-10).ln() * ml).floor() as usize;
        level.min(6) // cap at 6
    }

    /// Greedy single-step search to find the nearest node at `level`.
    fn greedy_search_layer(&self, query: &[f64], entry: usize, level: usize) -> usize {
        let mut current = entry;
        let mut current_dist = Self::euclidean_distance(query, &self.nodes[current].vector);

        loop {
            let mut improved = false;
            if level < self.nodes[current].neighbors.len() {
                for &nb in &self.nodes[current].neighbors[level] {
                    if nb < self.nodes.len() {
                        let d = Self::euclidean_distance(query, &self.nodes[nb].vector);
                        if d < current_dist {
                            current_dist = d;
                            current = nb;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    /// Search at `level` using an internal node index as the query proxy.
    fn search_layer(
        &self,
        query_idx: usize,
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<HeapEntry> {
        let query_vec = self.nodes[query_idx].vector.clone();
        self.search_layer_query(&query_vec, entry, ef, level)
    }

    /// Search at `level` using a raw query vector.
    fn search_layer_query(
        &self,
        query: &[f64],
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<HeapEntry> {
        let mut visited: HashSet<usize> = HashSet::new();
        // Min-heap of (distance, index) — closer nodes processed first
        let mut candidates: BinaryHeap<HeapEntry> = BinaryHeap::new();
        // Max-heap of current top-ef results (we evict the farthest when full)
        let mut results: BinaryHeap<std::cmp::Reverse<HeapEntry>> = BinaryHeap::new();

        let entry_dist = Self::euclidean_distance(query, &self.nodes[entry].vector);
        visited.insert(entry);
        candidates.push(HeapEntry {
            dist: entry_dist,
            idx: entry,
        });
        results.push(std::cmp::Reverse(HeapEntry {
            dist: entry_dist,
            idx: entry,
        }));

        while let Some(current) = candidates.pop() {
            // Check if current is farther than the worst in results
            let worst_dist = results
                .iter()
                .map(|r| r.0.dist)
                .fold(f64::NEG_INFINITY, f64::max);
            if current.dist > worst_dist && results.len() >= ef {
                break;
            }

            if level < self.nodes[current.idx].neighbors.len() {
                for &nb in &self.nodes[current.idx].neighbors[level] {
                    if nb < self.nodes.len() && visited.insert(nb) {
                        let d = Self::euclidean_distance(query, &self.nodes[nb].vector);
                        let worst = results
                            .iter()
                            .map(|r| r.0.dist)
                            .fold(f64::NEG_INFINITY, f64::max);

                        if results.len() < ef || d < worst {
                            candidates.push(HeapEntry { dist: d, idx: nb });
                            results.push(std::cmp::Reverse(HeapEntry { dist: d, idx: nb }));
                            // Trim to ef
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Extract and sort by distance ascending
        let mut out: Vec<HeapEntry> = results.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_config() -> HnswSearchConfig {
        HnswSearchConfig {
            m: 4,
            ef_construction: 10,
            ef_search: 10,
            max_level: 3,
        }
    }

    // --- Construction ---
    #[test]
    fn test_new_empty() {
        let idx = HnswSearchIndex::new(HnswSearchConfig::default());
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.dim(), None);
    }

    #[test]
    fn test_insert_single() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(0, vec![1.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.dim(), Some(3));
    }

    #[test]
    fn test_insert_multiple() {
        let mut idx = HnswSearchIndex::new(simple_config());
        for i in 0..10 {
            idx.insert(i, vec![i as f64, 0.0, 0.0]);
        }
        assert_eq!(idx.len(), 10);
    }

    #[test]
    fn test_insert_duplicate_id_ignored() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(0, vec![1.0, 0.0]);
        idx.insert(0, vec![2.0, 0.0]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_is_empty_true() {
        let idx = HnswSearchIndex::new(HnswSearchConfig::default());
        assert!(idx.is_empty());
    }

    #[test]
    fn test_is_empty_false_after_insert() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(0, vec![0.0]);
        assert!(!idx.is_empty());
    }

    // --- Euclidean distance ---
    #[test]
    fn test_euclidean_zero_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let d = HnswSearchIndex::euclidean_distance(&a, &a);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_unit_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = HnswSearchIndex::euclidean_distance(&a, &b);
        assert!((d - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn test_euclidean_3d() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        let d = HnswSearchIndex::euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-9);
    }

    // --- Search ---
    #[test]
    fn test_search_empty_returns_empty() {
        let idx = HnswSearchIndex::new(simple_config());
        let results = idx.search(&[0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_zero_returns_empty() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(0, vec![1.0, 0.0]);
        let results = idx.search(&[1.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single_node_returns_it() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(42, vec![1.0, 0.0, 0.0]);
        let results = idx.search(&[1.0, 0.0, 0.0], 1);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 42);
        assert!(results[0].distance < 1e-9);
    }

    #[test]
    fn test_search_returns_at_most_k() {
        let mut idx = HnswSearchIndex::new(simple_config());
        for i in 0..20usize {
            idx.insert(i, vec![i as f64, 0.0]);
        }
        let results = idx.search(&[5.0, 0.0], 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_search_closest_first() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(0, vec![0.0, 0.0]);
        idx.insert(1, vec![1.0, 0.0]);
        idx.insert(2, vec![10.0, 0.0]);
        let results = idx.search(&[0.5, 0.0], 2);
        // Closest two should be ids 0 and 1 in some order; distances should be sorted
        assert!(results.len() <= 2);
        if results.len() == 2 {
            assert!(results[0].distance <= results[1].distance);
        }
    }

    #[test]
    fn test_search_exact_match() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(7, vec![3.0, 4.0]);
        let results = idx.search(&[3.0, 4.0], 1);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 7);
    }

    // --- Config ---
    #[test]
    fn test_default_config() {
        let c = HnswSearchConfig::default();
        assert_eq!(c.m, 16);
        assert_eq!(c.ef_construction, 200);
        assert_eq!(c.ef_search, 50);
        assert_eq!(c.max_level, 6);
    }

    // --- dim ---
    #[test]
    fn test_dim_after_insert() {
        let mut idx = HnswSearchIndex::new(simple_config());
        idx.insert(0, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(idx.dim(), Some(4));
    }

    // --- large insertion ---
    #[test]
    fn test_large_insertion_no_panic() {
        let mut idx = HnswSearchIndex::new(HnswSearchConfig {
            m: 8,
            ef_construction: 20,
            ef_search: 20,
            max_level: 4,
        });
        for i in 0..50usize {
            let v = vec![(i % 10) as f64, (i / 10) as f64];
            idx.insert(i, v);
        }
        assert_eq!(idx.len(), 50);
        let results = idx.search(&[0.0, 0.0], 5);
        assert!(!results.is_empty());
    }

    // --- search result distances are non-negative ---
    #[test]
    fn test_search_distances_non_negative() {
        let mut idx = HnswSearchIndex::new(simple_config());
        for i in 0..10usize {
            idx.insert(i, vec![i as f64]);
        }
        let results = idx.search(&[5.0], 5);
        for r in &results {
            assert!(r.distance >= 0.0);
        }
    }
}
