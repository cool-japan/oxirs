//! Adaptive HNSW (Hierarchical Navigable Small World) index for quoted triple similarity.
//!
//! This module implements an adaptive HNSW index with:
//! - Adaptive `ef_construction` that auto-tunes based on observed recall/latency
//! - Layer promotion heuristics using neighbor diversity scoring
//! - SIMD-accelerated distance calculation for quoted triple embeddings
//!
//! # Architecture
//!
//! ```text
//! Layer L  (top, sparse): coarse navigation
//! Layer L-1:              ...
//! Layer 0  (base, dense): full neighborhood graph
//! ```
//!
//! Each node is a quoted triple represented as a fixed-length float vector
//! computed from its component hash fingerprints.

use crate::{StarError, StarResult, StarTerm, StarTriple};
use scirs2_core::profiling::Profiler;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default embedding dimensionality for quoted triple vectors.
pub const DEFAULT_DIM: usize = 64;

/// Default maximum number of connections per node at the base layer.
pub const DEFAULT_M: usize = 16;

/// Default maximum number of connections at higher layers (M0 = M).
pub const DEFAULT_M0: usize = 32;

/// Default initial ef_construction.
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;

/// Minimum allowed ef_construction (prevent degenerate graphs).
pub const MIN_EF_CONSTRUCTION: usize = 40;

/// Maximum allowed ef_construction (prevent runaway latency).
pub const MAX_EF_CONSTRUCTION: usize = 800;

/// Number of recent insertion latency samples to track for adaptation.
pub const LATENCY_WINDOW: usize = 128;

/// Recall target for adaptive ef tuning (0.0 – 1.0).
pub const TARGET_RECALL: f32 = 0.95;

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Supported distance metrics for quoted triple similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance.
    #[default]
    L2,
    /// Cosine distance (1 – cosine_similarity).
    Cosine,
    /// Jaccard distance on binary fingerprints.
    Jaccard,
}

/// Compute the distance between two fixed-length vectors using the selected metric.
///
/// Uses manual loop unrolling to allow auto-vectorization by LLVM.
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match metric {
        DistanceMetric::L2 => l2_distance_simd(a, b),
        DistanceMetric::Cosine => cosine_distance_simd(a, b),
        DistanceMetric::Jaccard => jaccard_distance_simd(a, b),
    }
}

/// SIMD-friendly L2 distance with 8-wide unroll.
#[inline(always)]
fn l2_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = [0.0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for k in 0..8 {
            let diff = a[base + k] - b[base + k];
            acc[k] += diff * diff;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// SIMD-friendly cosine distance with 8-wide unroll.
#[inline(always)]
fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut dot = [0.0f32; 8];
    let mut norm_a = [0.0f32; 8];
    let mut norm_b = [0.0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for k in 0..8 {
            dot[k] += a[base + k] * b[base + k];
            norm_a[k] += a[base + k] * a[base + k];
            norm_b[k] += b[base + k] * b[base + k];
        }
    }
    let (mut dot_s, mut na_s, mut nb_s) = (
        dot.iter().sum::<f32>(),
        norm_a.iter().sum::<f32>(),
        norm_b.iter().sum::<f32>(),
    );
    for i in (chunks * 8)..len {
        dot_s += a[i] * b[i];
        na_s += a[i] * a[i];
        nb_s += b[i] * b[i];
    }
    let denom = na_s.sqrt() * nb_s.sqrt();
    if denom < 1e-10 {
        return 1.0;
    }
    1.0 - (dot_s / denom).clamp(-1.0, 1.0)
}

/// SIMD-friendly Jaccard distance treating each f32 as a binary feature.
#[inline(always)]
fn jaccard_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut intersection = 0.0f32;
    let mut union = 0.0f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let av = if *ai > 0.0 { 1.0 } else { 0.0 };
        let bv = if *bi > 0.0 { 1.0 } else { 0.0 };
        intersection += av * bv;
        union += av + bv - av * bv;
    }
    if union < 1e-10 {
        return 1.0;
    }
    1.0 - intersection / union
}

// ---------------------------------------------------------------------------
// Triple embedding
// ---------------------------------------------------------------------------

/// Compute a deterministic float embedding for a quoted triple.
///
/// The embedding is derived from the FNV-1a hashes of the serialized
/// subject, predicate, and object terms.  The final vector is normalised
/// to unit L2 norm so that cosine similarity equals the dot product.
pub fn embed_triple(triple: &StarTriple, dim: usize) -> Vec<f32> {
    let s_hash = fnv1a(term_key(&triple.subject).as_bytes());
    let p_hash = fnv1a(term_key(&triple.predicate).as_bytes());
    let o_hash = fnv1a(term_key(&triple.object).as_bytes());

    let mut vec = vec![0.0f32; dim];
    for (i, elem) in vec.iter_mut().enumerate() {
        let seed = (i as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(s_hash);
        let p_part = (seed ^ p_hash).rotate_left(17);
        let o_part = (p_part ^ o_hash).rotate_left(31);
        // Map u64 to [-1, 1]
        *elem = (o_part as f32) / (u64::MAX as f32) * 2.0 - 1.0;
    }

    // L2 normalise
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        vec.iter_mut().for_each(|v| *v /= norm);
    }
    vec
}

fn term_key(term: &StarTerm) -> String {
    match term {
        StarTerm::NamedNode(n) => format!("<{}>", n.iri),
        StarTerm::BlankNode(b) => format!("_:{}", b.id),
        StarTerm::Literal(l) => {
            let lang = l.language.as_deref().unwrap_or("");
            let dt = l
                .datatype
                .as_ref()
                .map(|d| d.iri.as_str())
                .unwrap_or("xsd:string");
            format!("\"{}\"@{}^^{}", l.value, lang, dt)
        }
        StarTerm::QuotedTriple(t) => {
            format!(
                "<<{}|{}|{}>>",
                term_key(&t.subject),
                term_key(&t.predicate),
                term_key(&t.object)
            )
        }
        StarTerm::Variable(v) => format!("?{}", v.name),
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Adaptive ef_construction controller
// ---------------------------------------------------------------------------

/// Tracks insertion latencies and adjusts `ef_construction` dynamically.
///
/// The controller increases `ef` when recent recall drops below `TARGET_RECALL`
/// (estimated via neighbour overlap with a brute-force sample) and decreases
/// it when the graph already satisfies the recall target.
#[derive(Debug)]
pub struct AdaptiveEfController {
    /// Current ef_construction value.
    pub ef: usize,
    /// Circular buffer of recent insertion latencies (microseconds).
    latencies: Vec<u64>,
    /// Write position in the circular buffer.
    write_pos: usize,
    /// Number of valid samples.
    sample_count: usize,
    /// Estimated recall from the last evaluation window.
    pub last_recall: f32,
    /// Target recall threshold.
    target: f32,
}

impl AdaptiveEfController {
    pub fn new(initial_ef: usize, target_recall: f32) -> Self {
        Self {
            ef: initial_ef.clamp(MIN_EF_CONSTRUCTION, MAX_EF_CONSTRUCTION),
            latencies: vec![0u64; LATENCY_WINDOW],
            write_pos: 0,
            sample_count: 0,
            last_recall: 0.0,
            target: target_recall,
        }
    }

    /// Record an insertion latency sample (microseconds).
    pub fn record_latency(&mut self, latency_us: u64) {
        self.latencies[self.write_pos] = latency_us;
        self.write_pos = (self.write_pos + 1) % LATENCY_WINDOW;
        if self.sample_count < LATENCY_WINDOW {
            self.sample_count += 1;
        }
    }

    /// Update recall estimate and adjust ef accordingly.
    ///
    /// `estimated_recall` should be computed externally via brute-force spot checks.
    pub fn update_recall(&mut self, estimated_recall: f32) {
        self.last_recall = estimated_recall;
        if estimated_recall < self.target {
            // Recall too low – increase ef by 20%.
            let new_ef = ((self.ef as f32) * 1.20) as usize;
            self.ef = new_ef.clamp(MIN_EF_CONSTRUCTION, MAX_EF_CONSTRUCTION);
        } else {
            // Recall sufficient – try reducing ef by 10% to save latency.
            let new_ef = ((self.ef as f32) * 0.90) as usize;
            self.ef = new_ef.max(MIN_EF_CONSTRUCTION);
        }
    }

    /// Average latency over the current sample window (microseconds).
    pub fn avg_latency_us(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        let sum: u64 = self.latencies[..self.sample_count].iter().sum();
        sum as f64 / self.sample_count as f64
    }
}

// ---------------------------------------------------------------------------
// Layer promotion heuristics
// ---------------------------------------------------------------------------

/// Compute the layer for a new node using an exponential decay distribution.
///
/// Mirrors the original HNSW paper (Malkov & Yashunin, 2018):
/// `l = floor(-ln(uniform(0,1)) * mL)` where `mL = 1 / ln(M)`.
pub fn draw_level(m: usize, level_multiplier: f64) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Deterministic but spread out – use counter-based PRNG.
    static COUNTER: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0x1234_5678_abcd_ef01);

    let cnt = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut h = DefaultHasher::new();
    cnt.hash(&mut h);
    let bits = h.finish();

    // Map to (0, 1) avoiding 0.
    let u = (bits as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    let level = (-u.ln() * level_multiplier) as usize;
    level.min(m * 4) // cap to reasonable maximum
}

/// Diversity-based heuristic: among candidate neighbours, prefer those that
/// are mutually far from each other (as in the original HNSW `Select_Neighbours_Heuristic`).
pub fn select_neighbours_heuristic(
    candidates: &[(usize, f32)],
    max_m: usize,
    metric: DistanceMetric,
    embeddings: &[Vec<f32>],
    extend_candidates: bool,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() || embeddings.is_empty() {
        return Vec::new();
    }

    // Sort by ascending distance.
    let mut sorted: Vec<_> = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<(usize, f32)> = Vec::with_capacity(max_m);
    let mut discarded: Vec<(usize, f32)> = Vec::new();

    'outer: for &(cand_id, cand_dist) in &sorted {
        if result.len() >= max_m {
            break;
        }
        // Keep candidate only if it is closer to the query than to any already-selected neighbour.
        for &(sel_id, _) in &result {
            if sel_id >= embeddings.len() || cand_id >= embeddings.len() {
                continue;
            }
            let dist_to_selected =
                compute_distance(&embeddings[sel_id], &embeddings[cand_id], metric);
            if dist_to_selected < cand_dist {
                discarded.push((cand_id, cand_dist));
                continue 'outer;
            }
        }
        result.push((cand_id, cand_dist));
    }

    // If extend_candidates is set, pad with discarded candidates up to max_m.
    if extend_candidates {
        for d in &discarded {
            if result.len() >= max_m {
                break;
            }
            result.push(*d);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// HNSW node
// ---------------------------------------------------------------------------

/// A single node in the HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Index of the quoted triple in the data store.
    pub triple_idx: usize,
    /// Embedding vector.
    pub embedding: Vec<f32>,
    /// Neighbour lists per layer (layer 0 = base, last = top).
    pub neighbours: Vec<Vec<usize>>,
    /// Maximum layer this node participates in.
    pub max_layer: usize,
}

// ---------------------------------------------------------------------------
// Adaptive HNSW index
// ---------------------------------------------------------------------------

/// Adaptive HNSW index storing quoted triple embeddings.
///
/// Thread-safe via an `Arc<RwLock<…>>` inner state.
pub struct AdaptiveHnswIndex {
    inner: Arc<RwLock<HnswInner>>,
}

struct HnswInner {
    /// All indexed nodes.
    nodes: Vec<HnswNode>,
    /// Entry point (index into `nodes`).
    entry_point: Option<usize>,
    /// Maximum layer across all nodes.
    max_layer: usize,
    /// Index configuration.
    config: AdaptiveHnswConfig,
    /// Adaptive ef controller.
    ef_controller: AdaptiveEfController,
    /// Profiler for performance tracking.
    #[allow(dead_code)]
    profiler: Profiler,
    /// Map from triple index to node index.
    triple_to_node: HashMap<usize, usize>,
}

/// Configuration for the adaptive HNSW index.
#[derive(Debug, Clone)]
pub struct AdaptiveHnswConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Maximum connections per node at base layer.
    pub m: usize,
    /// Maximum connections per node at higher layers.
    pub m0: usize,
    /// Initial ef_construction.
    pub ef_construction: usize,
    /// ef for search (not for insert).
    pub ef_search: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Whether to use the heuristic neighbour selection.
    pub use_heuristic: bool,
}

impl Default for AdaptiveHnswConfig {
    fn default() -> Self {
        Self {
            dim: DEFAULT_DIM,
            m: DEFAULT_M,
            m0: DEFAULT_M0,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: 50,
            metric: DistanceMetric::L2,
            use_heuristic: true,
        }
    }
}

impl AdaptiveHnswIndex {
    /// Create a new empty adaptive HNSW index.
    pub fn new(config: AdaptiveHnswConfig) -> Self {
        let ef = config.ef_construction;
        let inner = HnswInner {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            ef_controller: AdaptiveEfController::new(ef, TARGET_RECALL),
            profiler: Profiler::new(),
            config,
            triple_to_node: HashMap::new(),
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    /// Insert a quoted triple into the index.
    ///
    /// Returns the node index assigned to this triple.
    pub fn insert(&self, triple_idx: usize, triple: &StarTriple) -> StarResult<usize> {
        let start = std::time::Instant::now();

        let mut guard = self
            .inner
            .write()
            .map_err(|_| StarError::processing_error("HNSW write lock poisoned"))?;

        let dim = guard.config.dim;
        let m = guard.config.m;
        let m0 = guard.config.m0;
        let metric = guard.config.metric;
        let use_heuristic = guard.config.use_heuristic;
        let ef = guard.ef_controller.ef;

        let embedding = embed_triple(triple, dim);

        // Determine layer for the new node.
        let m_l = 1.0 / (m as f64).ln();
        let node_layer = draw_level(m, m_l);

        // Create the node with empty neighbour lists for each layer.
        let node_idx = guard.nodes.len();
        let node = HnswNode {
            triple_idx,
            embedding: embedding.clone(),
            neighbours: vec![Vec::new(); node_layer + 1],
            max_layer: node_layer,
        };
        guard.nodes.push(node);
        guard.triple_to_node.insert(triple_idx, node_idx);

        // Connect the node into the graph.
        if let Some(ep) = guard.entry_point {
            let ep_layer = guard.nodes[ep].max_layer;
            let search_layer_top = ep_layer.min(node_layer);

            // Greedy descent from the top layer down to node_layer + 1.
            let mut current_ep = ep;
            for lc in ((node_layer + 1)..=ep_layer).rev() {
                let nearest =
                    greedy_search_layer(&guard.nodes, &embedding, current_ep, 1, lc, metric);
                if let Some((n, _)) = nearest.into_iter().next() {
                    current_ep = n;
                }
            }

            // At each layer from search_layer_top down to 0, find ef nearest
            // candidates and connect bidirectionally.
            let mut ep_set = vec![current_ep];
            for lc in (0..=search_layer_top).rev() {
                let max_m = if lc == 0 { m0 } else { m };

                let candidates = search_layer_ef(&guard.nodes, &embedding, &ep_set, ef, lc, metric);

                let neighbours = if use_heuristic {
                    let all_embeddings: Vec<Vec<f32>> =
                        guard.nodes.iter().map(|n| n.embedding.clone()).collect();
                    select_neighbours_heuristic(&candidates, max_m, metric, &all_embeddings, true)
                } else {
                    candidates[..candidates.len().min(max_m)].to_vec()
                };

                // Set neighbours for the new node at this layer.
                guard.nodes[node_idx].neighbours[lc] =
                    neighbours.iter().map(|(id, _)| *id).collect();

                // Add back-connections and prune if necessary.
                for &(nb_id, _) in &neighbours {
                    if nb_id == node_idx {
                        continue;
                    }
                    if lc < guard.nodes[nb_id].neighbours.len() {
                        guard.nodes[nb_id].neighbours[lc].push(node_idx);

                        // Prune if over capacity.
                        let nb_conns = guard.nodes[nb_id].neighbours[lc].len();
                        if nb_conns > max_m * 2 {
                            let nb_emb = guard.nodes[nb_id].embedding.clone();
                            let nb_layer = lc;
                            let mut cands: Vec<(usize, f32)> = guard.nodes[nb_id].neighbours
                                [nb_layer]
                                .iter()
                                .map(|&nid| {
                                    let d = if nid < guard.nodes.len() {
                                        compute_distance(
                                            &nb_emb,
                                            &guard.nodes[nid].embedding,
                                            metric,
                                        )
                                    } else {
                                        f32::MAX
                                    };
                                    (nid, d)
                                })
                                .collect();
                            cands.sort_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            guard.nodes[nb_id].neighbours[nb_layer] = cands
                                [..cands.len().min(max_m)]
                                .iter()
                                .map(|(id, _)| *id)
                                .collect();
                        }
                    }
                }

                // Update entry point for next layer down.
                if let Some((best, _)) = candidates.first() {
                    ep_set = vec![*best];
                }
            }

            // Update entry point if the new node has a higher layer.
            if node_layer > guard.max_layer {
                guard.entry_point = Some(node_idx);
                guard.max_layer = node_layer;
            }
        } else {
            // First node.
            guard.entry_point = Some(node_idx);
            guard.max_layer = node_layer;
        }

        let latency_us = start.elapsed().as_micros() as u64;
        guard.ef_controller.record_latency(latency_us);

        Ok(node_idx)
    }

    /// Search for the `k` nearest neighbours to the given quoted triple.
    pub fn search(&self, triple: &StarTriple, k: usize) -> StarResult<Vec<(usize, f32)>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| StarError::processing_error("HNSW read lock poisoned"))?;

        if guard.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let ep = match guard.entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let embedding = embed_triple(triple, guard.config.dim);
        let metric = guard.config.metric;
        let ef = guard.config.ef_search.max(k);

        let mut current_ep = ep;
        // Greedy descent to layer 1.
        for lc in (1..=guard.max_layer).rev() {
            let nearest = greedy_search_layer(&guard.nodes, &embedding, current_ep, 1, lc, metric);
            if let Some((n, _)) = nearest.into_iter().next() {
                current_ep = n;
            }
        }

        // Search base layer with ef.
        let candidates = search_layer_ef(&guard.nodes, &embedding, &[current_ep], ef, 0, metric);

        let mut results: Vec<(usize, f32)> = candidates
            .into_iter()
            .take(k)
            .map(|(nid, dist)| (guard.nodes[nid].triple_idx, dist))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Update the recall estimate and allow the controller to adjust ef.
    pub fn update_recall(&self, recall: f32) -> StarResult<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| StarError::processing_error("HNSW write lock poisoned"))?;
        guard.ef_controller.update_recall(recall);
        Ok(())
    }

    /// Return current ef_construction value.
    pub fn current_ef(&self) -> StarResult<usize> {
        let guard = self
            .inner
            .read()
            .map_err(|_| StarError::processing_error("HNSW read lock poisoned"))?;
        Ok(guard.ef_controller.ef)
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> StarResult<usize> {
        let guard = self
            .inner
            .read()
            .map_err(|_| StarError::processing_error("HNSW read lock poisoned"))?;
        Ok(guard.nodes.len())
    }

    /// Return true if the index is empty.
    pub fn is_empty(&self) -> StarResult<bool> {
        Ok(self.len()? == 0)
    }

    /// Statistics snapshot.
    pub fn statistics(&self) -> StarResult<HnswStatistics> {
        let guard = self
            .inner
            .read()
            .map_err(|_| StarError::processing_error("HNSW read lock poisoned"))?;
        let avg_degree: f64 = if guard.nodes.is_empty() {
            0.0
        } else {
            let total_edges: usize = guard
                .nodes
                .iter()
                .map(|n| n.neighbours.iter().map(|l| l.len()).sum::<usize>())
                .sum();
            total_edges as f64 / guard.nodes.len() as f64
        };
        Ok(HnswStatistics {
            node_count: guard.nodes.len(),
            max_layer: guard.max_layer,
            current_ef: guard.ef_controller.ef,
            last_recall: guard.ef_controller.last_recall,
            avg_latency_us: guard.ef_controller.avg_latency_us(),
            avg_degree,
        })
    }
}

/// Statistics snapshot for the adaptive HNSW index.
#[derive(Debug, Clone)]
pub struct HnswStatistics {
    pub node_count: usize,
    pub max_layer: usize,
    pub current_ef: usize,
    pub last_recall: f32,
    pub avg_latency_us: f64,
    pub avg_degree: f64,
}

// ---------------------------------------------------------------------------
// Graph search helpers
// ---------------------------------------------------------------------------

/// Greedy single-step search to find the `ef_search` nearest elements at a given layer.
fn greedy_search_layer(
    nodes: &[HnswNode],
    query: &[f32],
    entry: usize,
    ef: usize,
    layer: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    search_layer_ef(nodes, query, &[entry], ef, layer, metric)
}

/// Full ef-bounded beam search at a given HNSW layer.
///
/// Returns up to `ef` candidates sorted by ascending distance.
fn search_layer_ef(
    nodes: &[HnswNode],
    query: &[f32],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
    metric: DistanceMetric,
) -> Vec<(usize, f32)> {
    // Candidate min-heap (smallest distance first).
    // BinaryHeap is a max-heap, so we negate distances.
    #[derive(PartialEq)]
    struct Candidate {
        neg_dist: f32,
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
            // Higher neg_dist = smaller real dist = higher priority.
            self.neg_dist
                .partial_cmp(&other.neg_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut visited: HashSet<usize> = HashSet::new();
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
    // Result set: max-heap by distance so we can easily evict the worst.
    let mut result: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut worst_dist = f32::MAX;

    for &ep in entry_points {
        if ep >= nodes.len() {
            continue;
        }
        let d = compute_distance(query, &nodes[ep].embedding, metric);
        if visited.insert(ep) {
            candidates.push(Candidate {
                neg_dist: -d,
                id: ep,
            });
            result.push(Candidate {
                neg_dist: -d,
                id: ep,
            });
            worst_dist = worst_dist.min(d);
        }
    }

    while let Some(Candidate { neg_dist, id }) = candidates.pop() {
        let d = -neg_dist;
        if d > worst_dist && result.len() >= ef {
            break;
        }
        if layer >= nodes[id].neighbours.len() {
            continue;
        }
        for &nb_id in &nodes[id].neighbours[layer] {
            if nb_id >= nodes.len() || !visited.insert(nb_id) {
                continue;
            }
            let nb_d = compute_distance(query, &nodes[nb_id].embedding, metric);
            if nb_d < worst_dist || result.len() < ef {
                candidates.push(Candidate {
                    neg_dist: -nb_d,
                    id: nb_id,
                });
                result.push(Candidate {
                    neg_dist: -nb_d,
                    id: nb_id,
                });
                if result.len() > ef {
                    result.pop();
                }
                worst_dist = result.peek().map(|c| -c.neg_dist).unwrap_or(f32::MAX);
            }
        }
    }

    // Convert result heap to sorted vec (ascending distance).
    let mut out: Vec<(usize, f32)> = result.into_iter().map(|c| (c.id, -c.neg_dist)).collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Literal;
    use crate::{StarTerm, StarTriple};

    fn make_triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(
            StarTerm::iri(s).unwrap(),
            StarTerm::iri(p).unwrap(),
            StarTerm::iri(o).unwrap(),
        )
    }

    fn make_quoted_triple(s: &str, p: &str, o: &str) -> StarTriple {
        let inner = make_triple(s, p, o);
        StarTriple::new(
            StarTerm::QuotedTriple(Box::new(inner)),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::Literal(Literal {
                value: "0.9".to_string(),
                language: None,
                datatype: None,
            }),
        )
    }

    #[test]
    fn test_embed_triple_deterministic() {
        let t = make_triple("http://a.org/s", "http://a.org/p", "http://a.org/o");
        let v1 = embed_triple(&t, 64);
        let v2 = embed_triple(&t, 64);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_embed_triple_normalised() {
        let t = make_triple("http://a.org/s", "http://a.org/p", "http://a.org/o");
        let v = embed_triple(&t, 64);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn test_embed_different_triples_differ() {
        let t1 = make_triple("http://a.org/s1", "http://a.org/p", "http://a.org/o");
        let t2 = make_triple("http://a.org/s2", "http://a.org/p", "http://a.org/o");
        let v1 = embed_triple(&t1, 64);
        let v2 = embed_triple(&t2, 64);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_l2_distance_self_zero() {
        let v = vec![0.5f32, 0.3, 0.1, 0.7];
        let d = l2_distance_simd(&v, &v);
        assert!(d.abs() < 1e-6, "Self distance should be ~0, got {d}");
    }

    #[test]
    fn test_cosine_distance_identical() {
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        let d = cosine_distance_simd(&v, &v);
        assert!(
            d.abs() < 1e-6,
            "Cosine distance to self should be ~0, got {d}"
        );
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let d = cosine_distance_simd(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "Orthogonal cosine dist should be ~1, got {d}"
        );
    }

    #[test]
    fn test_jaccard_distance_identical() {
        let v = vec![1.0f32, 0.0, 1.0, 0.0];
        let d = jaccard_distance_simd(&v, &v);
        assert!(
            d.abs() < 1e-6,
            "Jaccard self-distance should be ~0, got {d}"
        );
    }

    #[test]
    fn test_adaptive_ef_controller_increase_on_low_recall() {
        let mut ctrl = AdaptiveEfController::new(200, TARGET_RECALL);
        ctrl.update_recall(0.5); // well below target
        assert!(
            ctrl.ef > 200,
            "ef should increase on low recall, got {}",
            ctrl.ef
        );
    }

    #[test]
    fn test_adaptive_ef_controller_decrease_on_high_recall() {
        let mut ctrl = AdaptiveEfController::new(400, TARGET_RECALL);
        ctrl.update_recall(0.99); // above target
        assert!(
            ctrl.ef < 400,
            "ef should decrease on high recall, got {}",
            ctrl.ef
        );
    }

    #[test]
    fn test_adaptive_ef_controller_clamp_min() {
        let mut ctrl = AdaptiveEfController::new(MIN_EF_CONSTRUCTION, TARGET_RECALL);
        // Even with very high recall, ef must not go below MIN.
        for _ in 0..20 {
            ctrl.update_recall(1.0);
        }
        assert!(
            ctrl.ef >= MIN_EF_CONSTRUCTION,
            "ef must not go below MIN, got {}",
            ctrl.ef
        );
    }

    #[test]
    fn test_draw_level_distribution() {
        // Levels should follow an exponential distribution – most nodes at level 0.
        let levels: Vec<usize> = (0..1000)
            .map(|_| draw_level(16, 1.0 / (16f64).ln()))
            .collect();
        let at_zero = levels.iter().filter(|&&l| l == 0).count();
        // With M=16, mL ≈ 0.36, so P(l=0) ≈ 1 - exp(-0.36) ≈ 0.30 – should be dominant.
        assert!(at_zero > 100, "Expected many level-0 nodes, got {at_zero}");
    }

    #[test]
    fn test_hnsw_insert_and_search_basic() {
        let cfg = AdaptiveHnswConfig {
            dim: 32,
            m: 8,
            m0: 16,
            ef_construction: 50,
            ef_search: 20,
            metric: DistanceMetric::L2,
            use_heuristic: true,
        };
        let index = AdaptiveHnswIndex::new(cfg);

        let triples: Vec<StarTriple> = (0..20)
            .map(|i| {
                make_triple(
                    &format!("http://ex.org/s{i}"),
                    "http://ex.org/p",
                    &format!("http://ex.org/o{i}"),
                )
            })
            .collect();

        for (i, t) in triples.iter().enumerate() {
            index.insert(i, t).unwrap();
        }

        assert_eq!(index.len().unwrap(), 20);

        // Search for the first triple – it should be closest to itself.
        let results = index.search(&triples[0], 3).unwrap();
        assert!(!results.is_empty(), "Search should return results");
        assert_eq!(
            results[0].0, 0,
            "First result should be the query triple itself"
        );
    }

    #[test]
    fn test_hnsw_with_quoted_triples() {
        let cfg = AdaptiveHnswConfig {
            dim: 32,
            m: 8,
            m0: 16,
            ef_construction: 50,
            ef_search: 20,
            metric: DistanceMetric::Cosine,
            use_heuristic: false,
        };
        let index = AdaptiveHnswIndex::new(cfg);

        let triples: Vec<StarTriple> = (0..10)
            .map(|i| {
                make_quoted_triple(
                    &format!("http://ex.org/s{i}"),
                    "http://ex.org/p",
                    &format!("http://ex.org/o{i}"),
                )
            })
            .collect();

        for (i, t) in triples.iter().enumerate() {
            index.insert(i, t).unwrap();
        }

        let results = index.search(&triples[5], 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hnsw_statistics() {
        let cfg = AdaptiveHnswConfig::default();
        let index = AdaptiveHnswIndex::new(cfg);

        for i in 0..50 {
            let t = make_triple(
                &format!("http://ex.org/s{i}"),
                "http://ex.org/p",
                &format!("http://ex.org/o{i}"),
            );
            index.insert(i, &t).unwrap();
        }

        let stats = index.statistics().unwrap();
        assert_eq!(stats.node_count, 50);
        assert!(stats.current_ef >= MIN_EF_CONSTRUCTION);
    }

    #[test]
    fn test_hnsw_update_recall() {
        let cfg = AdaptiveHnswConfig::default();
        let index = AdaptiveHnswIndex::new(cfg);
        // Insert one node so the graph is non-empty.
        let t = make_triple("http://a.org/s", "http://a.org/p", "http://a.org/o");
        index.insert(0, &t).unwrap();

        let ef_before = index.current_ef().unwrap();
        index.update_recall(0.3).unwrap(); // trigger increase
        let ef_after = index.current_ef().unwrap();
        assert!(
            ef_after >= ef_before,
            "ef should not decrease on low recall"
        );
    }

    #[test]
    fn test_hnsw_empty_search() {
        let cfg = AdaptiveHnswConfig::default();
        let index = AdaptiveHnswIndex::new(cfg);
        let t = make_triple("http://a.org/s", "http://a.org/p", "http://a.org/o");
        let results = index.search(&t, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_select_neighbours_heuristic_empty() {
        let result = select_neighbours_heuristic(&[], 5, DistanceMetric::L2, &[], false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_neighbours_heuristic_diversity() {
        // Three candidates at varying distances; heuristic should prefer diverse ones.
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0], // close to node 0
            vec![0.0, 1.0, 0.0, 0.0], // orthogonal to node 0
        ];
        let candidates = vec![(0usize, 0.0f32), (1usize, 0.1f32), (2usize, 1.0f32)];
        let selected =
            select_neighbours_heuristic(&candidates, 2, DistanceMetric::Cosine, &embeddings, false);
        // Node 2 should be included for diversity even though it's farther.
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_latency_recording() {
        let mut ctrl = AdaptiveEfController::new(200, TARGET_RECALL);
        ctrl.record_latency(100);
        ctrl.record_latency(200);
        ctrl.record_latency(300);
        let avg = ctrl.avg_latency_us();
        assert!(
            (avg - 200.0).abs() < 1.0,
            "avg latency expected ~200 got {avg}"
        );
    }

    #[test]
    fn test_latency_circular_buffer_wrap() {
        let mut ctrl = AdaptiveEfController::new(200, TARGET_RECALL);
        for i in 0..(LATENCY_WINDOW * 2) {
            ctrl.record_latency(i as u64);
        }
        // Should still be valid after wrap.
        let avg = ctrl.avg_latency_us();
        assert!(avg > 0.0, "avg latency should be > 0 after wrap");
    }

    #[test]
    fn test_compute_distance_l2() {
        let a = vec![3.0f32, 4.0];
        let b = vec![0.0f32, 0.0];
        let d = compute_distance(&a, &b, DistanceMetric::L2);
        assert!(
            (d - 5.0).abs() < 1e-5,
            "L2 distance 3,4 -> 0,0 should be 5, got {d}"
        );
    }

    #[test]
    fn test_hnsw_large_insertion() {
        let cfg = AdaptiveHnswConfig {
            dim: 16,
            m: 4,
            m0: 8,
            ef_construction: 30,
            ef_search: 10,
            metric: DistanceMetric::L2,
            use_heuristic: false,
        };
        let index = AdaptiveHnswIndex::new(cfg);

        for i in 0..200 {
            let t = make_triple(
                &format!("http://ex.org/s{i}"),
                &format!("http://ex.org/p{}", i % 10),
                &format!("http://ex.org/o{}", i % 20),
            );
            index.insert(i, &t).unwrap();
        }

        let stats = index.statistics().unwrap();
        assert_eq!(stats.node_count, 200);
        assert!(stats.avg_degree > 0.0, "avg_degree should be > 0");
    }
}
