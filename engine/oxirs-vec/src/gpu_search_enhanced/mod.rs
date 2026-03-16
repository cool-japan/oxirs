//! GPU-enhanced vector search primitives for OxiRS Vector Search.
//!
//! This module provides three high-level building blocks for
//! performance-critical vector search workloads:
//!
//! - [`SimdVectorSearch`] – SIMD-accelerated flat (brute-force) search using
//!   parallel dot products via `scirs2_core::simd`.
//! - [`BatchSearchEngine`] – Concurrent batch search across multiple query
//!   vectors using `scirs2_core::parallel_ops`.
//! - [`SearchMetrics`] – Lightweight instrumentation for measuring throughput
//!   and latency percentiles.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::parallel_ops::{IntoParallelRefIterator, ParallelIterator};
use scirs2_core::simd::simd_dot_f32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Compute the cosine distance between two `f32` slices using SIMD dot
/// products from `scirs2_core`.  Returns `f32::INFINITY` when either vector
/// has zero magnitude.
fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    let a_arr = Array1::from_vec(a.to_vec());
    let b_arr = Array1::from_vec(b.to_vec());

    let dot = simd_dot_f32(&a_arr.view(), &b_arr.view());
    let norm_a = simd_dot_f32(&a_arr.view(), &a_arr.view()).sqrt();
    let norm_b = simd_dot_f32(&b_arr.view(), &b_arr.view()).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        f32::INFINITY
    } else {
        let sim = dot / (norm_a * norm_b);
        1.0 - sim.clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// SimdVectorSearch
// ---------------------------------------------------------------------------

/// A stored entry in the [`SimdVectorSearch`] flat index.
#[derive(Debug, Clone)]
struct IndexEntry {
    id: String,
    data: Vec<f32>,
}

/// SIMD-accelerated flat (brute-force) vector search.
///
/// Stores all vectors in memory and computes cosine distances to a query
/// using `scirs2_core`'s SIMD dot-product primitive.  Parallel execution
/// is used for batches that exceed the configured threshold.
#[derive(Debug)]
pub struct SimdVectorSearch {
    entries: Vec<IndexEntry>,
    /// Use parallel computation when the candidate set exceeds this size.
    parallel_threshold: usize,
}

impl SimdVectorSearch {
    /// Create a new empty flat index with the given parallel threshold.
    pub fn new(parallel_threshold: usize) -> Self {
        Self {
            entries: Vec::new(),
            parallel_threshold,
        }
    }

    /// Create a flat index with the default threshold (256 vectors).
    pub fn default_threshold() -> Self {
        Self::new(256)
    }

    /// Insert a vector under `id`.  If `id` already exists, the vector is
    /// replaced.
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if vector.is_empty() {
            return Err(anyhow!("vector must not be empty"));
        }
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.data = vector;
        } else {
            self.entries.push(IndexEntry { id, data: vector });
        }
        Ok(())
    }

    /// Return the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search for the `k` nearest neighbours of `query` by cosine distance.
    ///
    /// Returns a `Vec` of `(id, distance)` pairs sorted by ascending distance,
    /// length ≤ `k`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if query.is_empty() {
            return Err(anyhow!("query vector must not be empty"));
        }
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(usize, f32)> = if self.entries.len() >= self.parallel_threshold {
            // Parallel path
            let indexed: Vec<(usize, &IndexEntry)> = self.entries.iter().enumerate().collect();
            let mut v: Vec<(usize, f32)> = indexed
                .par_iter()
                .map(|&(idx, entry)| (idx, cosine_distance_simd(query, &entry.data)))
                .collect();
            v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            v
        } else {
            // Sequential path for small sets
            let mut v: Vec<(usize, f32)> = self
                .entries
                .iter()
                .enumerate()
                .map(|(idx, entry)| (idx, cosine_distance_simd(query, &entry.data)))
                .collect();
            v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            v
        };

        scored.truncate(k);
        let results = scored
            .into_iter()
            .map(|(idx, dist)| (self.entries[idx].id.clone(), dist))
            .collect();
        Ok(results)
    }

    /// Compute raw cosine distances from `query` to all indexed vectors and
    /// return them in insertion order (not sorted).
    pub fn all_distances(&self, query: &[f32]) -> Result<Vec<(String, f32)>> {
        if query.is_empty() {
            return Err(anyhow!("query vector must not be empty"));
        }
        let results = self
            .entries
            .iter()
            .map(|e| (e.id.clone(), cosine_distance_simd(query, &e.data)))
            .collect();
        Ok(results)
    }
}

impl Default for SimdVectorSearch {
    fn default() -> Self {
        Self::default_threshold()
    }
}

// ---------------------------------------------------------------------------
// BatchSearchEngine
// ---------------------------------------------------------------------------

/// Concurrent batch search over multiple query vectors.
///
/// Wraps a [`SimdVectorSearch`] index and executes multiple queries in
/// parallel using `scirs2_core::parallel_ops`.
#[derive(Debug)]
pub struct BatchSearchEngine {
    index: Arc<SimdVectorSearch>,
}

impl BatchSearchEngine {
    /// Wrap an existing [`SimdVectorSearch`] index.
    pub fn new(index: SimdVectorSearch) -> Self {
        Self {
            index: Arc::new(index),
        }
    }

    /// Execute `queries` in parallel, each returning the `k` nearest
    /// neighbours.  The outer `Vec` preserves query ordering.
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<(String, f32)>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let index = Arc::clone(&self.index);

        let results: Vec<Vec<(String, f32)>> = queries
            .par_iter()
            .map(|q| index.search(q, k).unwrap_or_default())
            .collect();

        Ok(results)
    }

    /// Search for a single query and record the latency into `metrics`.
    pub fn timed_search(
        &self,
        query: &[f32],
        k: usize,
        metrics: &SearchMetrics,
    ) -> Result<Vec<(String, f32)>> {
        let start = Instant::now();
        let result = self.index.search(query, k)?;
        let elapsed_us = start.elapsed().as_micros() as u64;
        metrics.record_query(elapsed_us);
        Ok(result)
    }

    /// Return the number of vectors in the underlying index.
    pub fn index_size(&self) -> usize {
        self.index.len()
    }
}

// ---------------------------------------------------------------------------
// SearchMetrics
// ---------------------------------------------------------------------------

/// Lightweight, lock-free performance metrics for vector search operations.
///
/// Tracks:
/// - total number of queries executed
/// - cumulative latency in microseconds (for mean computation)
/// - minimum and maximum observed latency
/// - approximate p50, p90, p99 percentiles (computed from a sorted snapshot)
#[derive(Debug)]
pub struct SearchMetrics {
    total_queries: AtomicU64,
    total_latency_us: AtomicU64,
    min_latency_us: AtomicU64,
    max_latency_us: AtomicU64,
    // Simple reservoir for percentile estimation (bounded to 4096 samples)
    reservoir: parking_lot::Mutex<Vec<u64>>,
    reservoir_cap: usize,
}

impl SearchMetrics {
    const DEFAULT_RESERVOIR_CAP: usize = 4096;

    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            total_queries: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            min_latency_us: AtomicU64::new(u64::MAX),
            max_latency_us: AtomicU64::new(0),
            reservoir: parking_lot::Mutex::new(Vec::with_capacity(Self::DEFAULT_RESERVOIR_CAP)),
            reservoir_cap: Self::DEFAULT_RESERVOIR_CAP,
        }
    }

    /// Record a single query with its observed latency in microseconds.
    pub fn record_query(&self, latency_us: u64) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);

        // Update min
        let mut current_min = self.min_latency_us.load(Ordering::Relaxed);
        while latency_us < current_min {
            match self.min_latency_us.compare_exchange_weak(
                current_min,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(updated) => current_min = updated,
            }
        }

        // Update max
        let mut current_max = self.max_latency_us.load(Ordering::Relaxed);
        while latency_us > current_max {
            match self.max_latency_us.compare_exchange_weak(
                current_max,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(updated) => current_max = updated,
            }
        }

        // Push to reservoir (capped)
        let mut res = self.reservoir.lock();
        if res.len() < self.reservoir_cap {
            res.push(latency_us);
        }
    }

    /// Total number of queries recorded.
    pub fn total_queries(&self) -> u64 {
        self.total_queries.load(Ordering::Relaxed)
    }

    /// Mean latency in microseconds, or `None` if no queries have been
    /// recorded yet.
    pub fn mean_latency_us(&self) -> Option<f64> {
        let n = self.total_queries();
        if n == 0 {
            return None;
        }
        Some(self.total_latency_us.load(Ordering::Relaxed) as f64 / n as f64)
    }

    /// Minimum observed latency in microseconds.
    pub fn min_latency_us(&self) -> Option<u64> {
        let v = self.min_latency_us.load(Ordering::Relaxed);
        if v == u64::MAX {
            None
        } else {
            Some(v)
        }
    }

    /// Maximum observed latency in microseconds.
    pub fn max_latency_us(&self) -> Option<u64> {
        let v = self.max_latency_us.load(Ordering::Relaxed);
        if v == 0 && self.total_queries() == 0 {
            None
        } else {
            Some(v)
        }
    }

    /// Approximate throughput in queries per second, computed from the mean
    /// latency. Returns `None` if no queries recorded or mean is zero.
    pub fn throughput_qps(&self) -> Option<f64> {
        let mean = self.mean_latency_us()?;
        if mean == 0.0 {
            return None;
        }
        Some(1_000_000.0 / mean)
    }

    /// Compute the p-th percentile (0–100) latency from the current
    /// reservoir sample.  Returns `None` if no samples are available.
    pub fn percentile_us(&self, p: f64) -> Option<u64> {
        let mut res = self.reservoir.lock();
        if res.is_empty() {
            return None;
        }
        res.sort_unstable();
        let idx = ((p / 100.0) * (res.len() - 1) as f64).round() as usize;
        Some(res[idx.min(res.len() - 1)])
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.total_queries.store(0, Ordering::Relaxed);
        self.total_latency_us.store(0, Ordering::Relaxed);
        self.min_latency_us.store(u64::MAX, Ordering::Relaxed);
        self.max_latency_us.store(0, Ordering::Relaxed);
        self.reservoir.lock().clear();
    }
}

impl Default for SearchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn unit_vec(dim: usize, hot_dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot_dim % dim] = 1.0;
        v
    }

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn build_index(n: usize, dim: usize) -> SimdVectorSearch {
        let mut idx = SimdVectorSearch::new(16);
        for i in 0..n {
            let v = random_vec(dim, i as u64 + 1);
            idx.insert(format!("v{}", i), v)
                .expect("insert should succeed");
        }
        idx
    }

    // ------------------------------------------------------------------
    // SimdVectorSearch tests
    // ------------------------------------------------------------------

    #[test]
    fn test_simd_search_basic_knn() -> Result<()> {
        let mut idx = SimdVectorSearch::new(4);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0])?;
        idx.insert("b".into(), vec![0.0, 1.0, 0.0])?;
        idx.insert("c".into(), vec![0.9, 0.1, 0.0])?;

        let query = vec![1.0, 0.0, 0.0];
        let results = idx.search(&query, 2)?;
        assert_eq!(results.len(), 2);
        // "a" (identical) must be first
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 < 1e-5);
        Ok(())
    }

    #[test]
    fn test_simd_search_empty_index() -> Result<()> {
        let idx = SimdVectorSearch::new(16);
        let results = idx.search(&[1.0, 0.0], 5)?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_simd_search_single_entry() -> Result<()> {
        let mut idx = SimdVectorSearch::new(4);
        idx.insert("only".into(), vec![0.6, 0.8])?;
        let results = idx.search(&[0.6, 0.8], 10)?;
        assert_eq!(results.len(), 1);
        assert!(results[0].1 < 1e-5);
        Ok(())
    }

    #[test]
    fn test_simd_search_k_larger_than_index() -> Result<()> {
        let idx = build_index(5, 4);
        let query = random_vec(4, 999);
        let results = idx.search(&query, 100)?;
        assert_eq!(results.len(), 5, "should return at most index size");
        Ok(())
    }

    #[test]
    fn test_simd_search_results_sorted_ascending() -> Result<()> {
        let idx = build_index(50, 8);
        let query = random_vec(8, 42);
        let results = idx.search(&query, 20)?;
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", w);
        }
        Ok(())
    }

    #[test]
    fn test_simd_search_parallel_threshold_switch() -> Result<()> {
        // Parallel threshold of 4; index has 8 entries → parallel path
        let mut idx = SimdVectorSearch::new(4);
        for i in 0..8_usize {
            idx.insert(format!("p{}", i), unit_vec(4, i))?;
        }
        let query = unit_vec(4, 0);
        let results = idx.search(&query, 3)?;
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "p0");
        Ok(())
    }

    #[test]
    fn test_simd_search_update_existing_id() -> Result<()> {
        let mut idx = SimdVectorSearch::new(4);
        idx.insert("x".into(), vec![1.0, 0.0])?;
        idx.insert("x".into(), vec![0.0, 1.0])?; // update

        assert_eq!(idx.len(), 1);
        let results = idx.search(&[0.0, 1.0], 1)?;
        assert_eq!(results[0].0, "x");
        assert!(results[0].1 < 1e-5);
        Ok(())
    }

    #[test]
    fn test_simd_all_distances_length() -> Result<()> {
        let idx = build_index(10, 4);
        let query = random_vec(4, 7);
        let all = idx.all_distances(&query)?;
        assert_eq!(all.len(), 10);
        Ok(())
    }

    #[test]
    fn test_simd_orthogonal_max_distance() -> Result<()> {
        let mut idx = SimdVectorSearch::new(4);
        idx.insert("y".into(), vec![0.0, 1.0])?;

        let query = vec![1.0, 0.0];
        let results = idx.search(&query, 1)?;
        assert!((results[0].1 - 1.0).abs() < 1e-4);
        Ok(())
    }

    // ------------------------------------------------------------------
    // BatchSearchEngine tests
    // ------------------------------------------------------------------

    #[test]
    fn test_batch_search_basic() -> Result<()> {
        let engine = BatchSearchEngine::new(build_index(20, 4));
        let queries: Vec<Vec<f32>> = (0..5).map(|i| random_vec(4, i as u64)).collect();
        let results = engine.batch_search(&queries, 3)?;
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.len() <= 3);
        }
        Ok(())
    }

    #[test]
    fn test_batch_search_empty_queries() -> Result<()> {
        let engine = BatchSearchEngine::new(build_index(10, 4));
        let results = engine.batch_search(&[], 5)?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_batch_search_order_preserved() -> Result<()> {
        let mut idx = SimdVectorSearch::new(4);
        idx.insert("origin".into(), vec![0.0, 0.0, 0.0, 0.0])?;
        idx.insert("x_axis".into(), vec![1.0, 0.0, 0.0, 0.0])?;
        idx.insert("y_axis".into(), vec![0.0, 1.0, 0.0, 0.0])?;

        let engine = BatchSearchEngine::new(idx);
        let queries = vec![
            vec![1.0_f32, 0.0, 0.0, 0.0], // closest to x_axis
            vec![0.0_f32, 1.0, 0.0, 0.0], // closest to y_axis
        ];
        let results = engine.batch_search(&queries, 1)?;
        assert_eq!(results[0][0].0, "x_axis");
        assert_eq!(results[1][0].0, "y_axis");
        Ok(())
    }

    #[test]
    fn test_batch_search_large_concurrent() -> Result<()> {
        let engine = BatchSearchEngine::new(build_index(200, 16));
        let queries: Vec<Vec<f32>> = (0..50).map(|i| random_vec(16, i as u64 + 100)).collect();
        let results = engine.batch_search(&queries, 5)?;
        assert_eq!(results.len(), 50);
        for r in &results {
            assert!(!r.is_empty());
        }
        Ok(())
    }

    #[test]
    fn test_batch_timed_search_records_metrics() -> Result<()> {
        let engine = BatchSearchEngine::new(build_index(30, 8));
        let metrics = SearchMetrics::new();
        let query = random_vec(8, 77);

        let results = engine.timed_search(&query, 3, &metrics)?;
        assert!(!results.is_empty());
        assert_eq!(metrics.total_queries(), 1);
        assert!(metrics.mean_latency_us().is_some());
        Ok(())
    }

    // ------------------------------------------------------------------
    // SearchMetrics tests
    // ------------------------------------------------------------------

    #[test]
    fn test_metrics_basic_recording() -> Result<()> {
        let m = SearchMetrics::new();
        m.record_query(100);
        m.record_query(200);
        m.record_query(300);

        assert_eq!(m.total_queries(), 3);
        let mean = m.mean_latency_us().expect("mean latency should be Some");
        assert!((mean - 200.0).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_metrics_min_max() -> Result<()> {
        let m = SearchMetrics::new();
        m.record_query(50);
        m.record_query(150);
        m.record_query(300);

        let __val = m.min_latency_us().expect("min latency should be Some");
        assert_eq!(__val, 50);
        let __val = m.max_latency_us().expect("max latency should be Some");
        assert_eq!(__val, 300);
        Ok(())
    }

    #[test]
    fn test_metrics_percentile_p50() -> Result<()> {
        let m = SearchMetrics::new();
        for lat in [10_u64, 20, 30, 40, 50] {
            m.record_query(lat);
        }
        // p50 of [10,20,30,40,50] is index 2 → 30
        let p50 = m.percentile_us(50.0).expect("p50 should be Some");
        assert_eq!(p50, 30);
        Ok(())
    }

    #[test]
    fn test_metrics_reset() {
        let m = SearchMetrics::new();
        m.record_query(100);
        m.reset();
        assert_eq!(m.total_queries(), 0);
        assert!(m.mean_latency_us().is_none());
    }

    #[test]
    fn test_metrics_throughput_qps() -> Result<()> {
        let m = SearchMetrics::new();
        m.record_query(1_000); // 1 ms = 1 000 µs → 1 000 QPS
        let qps = m.throughput_qps().expect("throughput_qps should be Some");
        assert!((qps - 1_000.0).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_metrics_no_queries_returns_none() {
        let m = SearchMetrics::new();
        assert!(m.mean_latency_us().is_none());
        assert!(m.min_latency_us().is_none());
        assert!(m.throughput_qps().is_none());
        assert!(m.percentile_us(50.0).is_none());
    }
}
