//! ANN (Approximate Nearest Neighbour) recall and latency benchmarking.
//!
//! Provides utilities for evaluating the quality and performance of ANN indices:
//! recall@k, QPS, build time, memory usage, ground-truth generation,
//! precision-recall tradeoff, latency percentiles, and report generation.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── Ground truth ────────────────────────────────────────────────────────────

/// A single query result: (vector_id, distance).
pub type Neighbour = (usize, f32);

/// Brute-force computation of exact k nearest neighbours for a set of queries.
///
/// `dataset` is the indexed corpus; `queries` are the query vectors;
/// `k` is the number of neighbours to return for each query.
/// Distance is squared Euclidean.
pub fn brute_force_knn(
    dataset: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<Neighbour>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(usize, f32)> = dataset
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);
            dists
        })
        .collect()
}

// ── Recall@k ────────────────────────────────────────────────────────────────

/// Compute recall@k: the fraction of true k-NN that appear in the
/// approximate result set.
///
/// `ground_truth` and `approximate` must be parallel slices (one entry per query).
pub fn recall_at_k(ground_truth: &[Vec<Neighbour>], approximate: &[Vec<Neighbour>]) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let mut total_recall = 0.0;
    for (gt, ap) in ground_truth.iter().zip(approximate.iter()) {
        let gt_ids: std::collections::HashSet<usize> = gt.iter().map(|n| n.0).collect();
        let ap_ids: std::collections::HashSet<usize> = ap.iter().map(|n| n.0).collect();
        let found = gt_ids.intersection(&ap_ids).count();
        if gt_ids.is_empty() {
            continue;
        }
        total_recall += found as f64 / gt_ids.len() as f64;
    }
    total_recall / ground_truth.len() as f64
}

/// Compute recall@k for individual queries (not averaged).
pub fn per_query_recall(
    ground_truth: &[Vec<Neighbour>],
    approximate: &[Vec<Neighbour>],
) -> Vec<f64> {
    ground_truth
        .iter()
        .zip(approximate.iter())
        .map(|(gt, ap)| {
            let gt_ids: std::collections::HashSet<usize> = gt.iter().map(|n| n.0).collect();
            let ap_ids: std::collections::HashSet<usize> = ap.iter().map(|n| n.0).collect();
            let found = gt_ids.intersection(&ap_ids).count();
            if gt_ids.is_empty() {
                0.0
            } else {
                found as f64 / gt_ids.len() as f64
            }
        })
        .collect()
}

// ── Precision ───────────────────────────────────────────────────────────────

/// Compute precision: the fraction of returned results that are true neighbours.
pub fn precision(ground_truth: &[Vec<Neighbour>], approximate: &[Vec<Neighbour>]) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for (gt, ap) in ground_truth.iter().zip(approximate.iter()) {
        let gt_ids: std::collections::HashSet<usize> = gt.iter().map(|n| n.0).collect();
        let ap_ids: std::collections::HashSet<usize> = ap.iter().map(|n| n.0).collect();
        let found = gt_ids.intersection(&ap_ids).count();
        if ap_ids.is_empty() {
            continue;
        }
        total += found as f64 / ap_ids.len() as f64;
    }
    total / ground_truth.len() as f64
}

// ── QPS measurement ─────────────────────────────────────────────────────────

/// Measure queries per second for a given search function.
///
/// `search_fn` takes a query vector and returns the approximate result.
pub fn measure_qps<F>(queries: &[Vec<f32>], mut search_fn: F) -> QpsResult
where
    F: FnMut(&[f32]) -> Vec<Neighbour>,
{
    let mut latencies = Vec::with_capacity(queries.len());
    let overall_start = Instant::now();

    for q in queries {
        let start = Instant::now();
        let _ = search_fn(q);
        latencies.push(start.elapsed());
    }

    let total_time = overall_start.elapsed();
    let qps = if total_time.as_secs_f64() > 0.0 {
        queries.len() as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    latencies.sort();

    QpsResult {
        qps,
        total_queries: queries.len(),
        total_time,
        latencies,
    }
}

/// Result of a QPS measurement.
#[derive(Debug, Clone)]
pub struct QpsResult {
    /// Queries per second.
    pub qps: f64,
    /// Total number of queries executed.
    pub total_queries: usize,
    /// Wall-clock time for all queries.
    pub total_time: Duration,
    /// Per-query latencies (sorted ascending).
    pub latencies: Vec<Duration>,
}

impl QpsResult {
    /// Median latency.
    pub fn p50(&self) -> Duration {
        percentile_duration(&self.latencies, 50.0)
    }

    /// 95th-percentile latency.
    pub fn p95(&self) -> Duration {
        percentile_duration(&self.latencies, 95.0)
    }

    /// 99th-percentile latency.
    pub fn p99(&self) -> Duration {
        percentile_duration(&self.latencies, 99.0)
    }

    /// Mean latency.
    pub fn mean_latency(&self) -> Duration {
        if self.latencies.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.latencies.iter().sum();
        total / self.latencies.len() as u32
    }

    /// Minimum latency.
    pub fn min_latency(&self) -> Duration {
        self.latencies.first().copied().unwrap_or(Duration::ZERO)
    }

    /// Maximum latency.
    pub fn max_latency(&self) -> Duration {
        self.latencies.last().copied().unwrap_or(Duration::ZERO)
    }
}

/// Helper: compute the p-th percentile from a sorted duration list.
fn percentile_duration(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((pct / 100.0) * (sorted.len() as f64 - 1.0))
        .round()
        .max(0.0) as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

// ── Build time tracking ─────────────────────────────────────────────────────

/// Track how long an index build takes.
pub struct BuildTimer {
    label: String,
    start: Instant,
}

impl BuildTimer {
    /// Start a new build timer with a label.
    pub fn start(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
        }
    }

    /// Stop the timer and return the result.
    pub fn stop(self) -> BuildTimeResult {
        BuildTimeResult {
            label: self.label,
            duration: self.start.elapsed(),
        }
    }
}

/// Result of a build-time measurement.
#[derive(Debug, Clone)]
pub struct BuildTimeResult {
    /// Label for the build operation.
    pub label: String,
    /// Elapsed time.
    pub duration: Duration,
}

// ── Memory estimation ───────────────────────────────────────────────────────

/// Estimate the memory footprint of a flat vector index (vectors only).
///
/// Returns bytes.
pub fn estimate_flat_memory(n_vectors: usize, dimension: usize) -> usize {
    n_vectors * dimension * std::mem::size_of::<f32>()
}

/// Estimate memory for an HNSW-like graph index.
///
/// Accounts for vectors + adjacency lists.
pub fn estimate_hnsw_memory(
    n_vectors: usize,
    dimension: usize,
    m: usize,        // max connections per layer
    n_levels: usize, // number of HNSW levels
) -> usize {
    let vector_bytes = n_vectors * dimension * std::mem::size_of::<f32>();
    // Each node has up to m neighbours per level; store as usize ids
    let graph_bytes = n_vectors * m * n_levels * std::mem::size_of::<usize>();
    vector_bytes + graph_bytes
}

/// Estimate memory for a product-quantised (PQ) index.
pub fn estimate_pq_memory(n_vectors: usize, n_subspaces: usize) -> usize {
    // Each vector → n_subspaces bytes (codes)
    n_vectors * n_subspaces
}

// ── Precision-recall tradeoff ───────────────────────────────────────────────

/// A data point on the precision-recall curve.
#[derive(Debug, Clone)]
pub struct PrecisionRecallPoint {
    /// Recall value [0, 1].
    pub recall: f64,
    /// Precision value [0, 1].
    pub precision: f64,
    /// The parameter setting that produced this point.
    pub parameter: String,
}

/// Run a sweep over a set of parameter values and collect recall/precision
/// at each setting.
///
/// `search_with_param` receives a parameter value and returns the
/// approximate results for all queries.
pub fn precision_recall_sweep<F>(
    ground_truth: &[Vec<Neighbour>],
    queries: &[Vec<f32>],
    param_values: &[String],
    mut search_with_param: F,
) -> Vec<PrecisionRecallPoint>
where
    F: FnMut(&str, &[Vec<f32>]) -> Vec<Vec<Neighbour>>,
{
    let mut curve = Vec::with_capacity(param_values.len());
    for param in param_values {
        let approx = search_with_param(param, queries);
        let r = recall_at_k(ground_truth, &approx);
        let p = precision(ground_truth, &approx);
        curve.push(PrecisionRecallPoint {
            recall: r,
            precision: p,
            parameter: param.clone(),
        });
    }
    curve
}

// ── Benchmark report ────────────────────────────────────────────────────────

/// A complete benchmark report.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Name of the index / algorithm.
    pub index_name: String,
    /// Number of vectors in the dataset.
    pub dataset_size: usize,
    /// Dimensionality.
    pub dimension: usize,
    /// Number of queries.
    pub n_queries: usize,
    /// k used for recall measurement.
    pub k: usize,
    /// Overall recall@k.
    pub recall: f64,
    /// Overall precision.
    pub precision: f64,
    /// QPS result.
    pub qps: f64,
    /// P50 latency.
    pub p50_us: u64,
    /// P95 latency.
    pub p95_us: u64,
    /// P99 latency.
    pub p99_us: u64,
    /// Estimated memory in bytes.
    pub memory_bytes: usize,
    /// Build time in milliseconds.
    pub build_time_ms: u64,
    /// Extra key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl BenchmarkReport {
    /// Format the report as a human-readable string.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== ANN Benchmark Report: {} ===\n",
            self.index_name
        ));
        out.push_str(&format!(
            "Dataset: {} vectors × {} dims\n",
            self.dataset_size, self.dimension
        ));
        out.push_str(&format!("Queries: {}, k={}\n", self.n_queries, self.k));
        out.push_str(&format!("Recall@{}: {:.4}\n", self.k, self.recall));
        out.push_str(&format!("Precision: {:.4}\n", self.precision));
        out.push_str(&format!("QPS: {:.1}\n", self.qps));
        out.push_str(&format!(
            "Latency p50: {} µs, p95: {} µs, p99: {} µs\n",
            self.p50_us, self.p95_us, self.p99_us
        ));
        out.push_str(&format!(
            "Memory: {:.2} MB\n",
            self.memory_bytes as f64 / (1024.0 * 1024.0)
        ));
        out.push_str(&format!("Build time: {} ms\n", self.build_time_ms));
        if !self.metadata.is_empty() {
            out.push_str("Metadata:\n");
            for (k, v) in &self.metadata {
                out.push_str(&format!("  {k}: {v}\n"));
            }
        }
        out
    }

    /// Format the report as a JSON string.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"index_name\": \"{}\",\n", self.index_name));
        out.push_str(&format!("  \"dataset_size\": {},\n", self.dataset_size));
        out.push_str(&format!("  \"dimension\": {},\n", self.dimension));
        out.push_str(&format!("  \"n_queries\": {},\n", self.n_queries));
        out.push_str(&format!("  \"k\": {},\n", self.k));
        out.push_str(&format!("  \"recall\": {:.6},\n", self.recall));
        out.push_str(&format!("  \"precision\": {:.6},\n", self.precision));
        out.push_str(&format!("  \"qps\": {:.1},\n", self.qps));
        out.push_str(&format!("  \"p50_us\": {},\n", self.p50_us));
        out.push_str(&format!("  \"p95_us\": {},\n", self.p95_us));
        out.push_str(&format!("  \"p99_us\": {},\n", self.p99_us));
        out.push_str(&format!("  \"memory_bytes\": {},\n", self.memory_bytes));
        out.push_str(&format!("  \"build_time_ms\": {}\n", self.build_time_ms));
        out.push('}');
        out
    }
}

// ── Distance ratio (approximation quality) ──────────────────────────────────

/// Compute the average distance ratio: sum of approx_dist / true_dist for
/// each query's k-th neighbour.  A perfect index has ratio = 1.0.
pub fn average_distance_ratio(
    ground_truth: &[Vec<Neighbour>],
    approximate: &[Vec<Neighbour>],
) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for (gt, ap) in ground_truth.iter().zip(approximate.iter()) {
        for (g, a) in gt.iter().zip(ap.iter()) {
            if g.1 > 1e-12 {
                total += a.1 as f64 / g.1 as f64;
                count += 1;
            }
        }
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_dataset() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![5.0, 5.0],
            vec![10.0, 10.0],
        ]
    }

    fn simple_queries() -> Vec<Vec<f32>> {
        vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![5.0, 5.0]]
    }

    // ── Brute-force ground truth ────────────────────────────────────────────

    #[test]
    fn test_brute_force_knn_basic() {
        let data = simple_dataset();
        let queries = vec![vec![0.0, 0.0]];
        let gt = brute_force_knn(&data, &queries, 3);
        assert_eq!(gt.len(), 1);
        assert_eq!(gt[0].len(), 3);
        // Nearest to (0,0) should be index 0 (itself) with distance 0
        assert_eq!(gt[0][0].0, 0);
        assert!((gt[0][0].1).abs() < 1e-6);
    }

    #[test]
    fn test_brute_force_knn_k_larger_than_dataset() {
        let data = vec![vec![1.0], vec![2.0]];
        let queries = vec![vec![0.0]];
        let gt = brute_force_knn(&data, &queries, 10);
        // Should return at most dataset.len() neighbours
        assert_eq!(gt[0].len(), 2);
    }

    #[test]
    fn test_brute_force_ordering() {
        let data = simple_dataset();
        let queries = vec![vec![0.0, 0.0]];
        let gt = brute_force_knn(&data, &queries, 4);
        // Distances should be non-decreasing
        for i in 1..gt[0].len() {
            assert!(gt[0][i].1 >= gt[0][i - 1].1);
        }
    }

    // ── Recall@k ────────────────────────────────────────────────────────────

    #[test]
    fn test_recall_perfect() {
        let gt = vec![vec![(0, 0.0), (1, 1.0), (2, 1.0)]];
        let ap = vec![vec![(0, 0.0), (1, 1.0), (2, 1.0)]];
        let r = recall_at_k(&gt, &ap);
        assert!(
            (r - 1.0).abs() < 1e-10,
            "Perfect recall should be 1.0, got {r}"
        );
    }

    #[test]
    fn test_recall_zero() {
        let gt = vec![vec![(0, 0.0), (1, 1.0)]];
        let ap = vec![vec![(5, 10.0), (6, 11.0)]];
        let r = recall_at_k(&gt, &ap);
        assert!(r.abs() < 1e-10, "No overlap → recall = 0, got {r}");
    }

    #[test]
    fn test_recall_partial() {
        let gt = vec![vec![(0, 0.0), (1, 1.0), (2, 1.0), (3, 2.0)]];
        let ap = vec![vec![(0, 0.0), (1, 1.0), (5, 5.0), (6, 6.0)]];
        let r = recall_at_k(&gt, &ap);
        // 2 out of 4 true neighbours found
        assert!((r - 0.5).abs() < 1e-10, "Recall = 0.5, got {r}");
    }

    #[test]
    fn test_recall_empty() {
        let r = recall_at_k(&[], &[]);
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn test_per_query_recall() {
        let gt = vec![vec![(0, 0.0), (1, 1.0)], vec![(2, 0.0), (3, 1.0)]];
        let ap = vec![
            vec![(0, 0.0), (1, 1.0)], // perfect
            vec![(2, 0.0), (5, 5.0)], // 1 of 2
        ];
        let pq = per_query_recall(&gt, &ap);
        assert!((pq[0] - 1.0).abs() < 1e-10);
        assert!((pq[1] - 0.5).abs() < 1e-10);
    }

    // ── Precision ───────────────────────────────────────────────────────────

    #[test]
    fn test_precision_perfect() {
        let gt = vec![vec![(0, 0.0), (1, 1.0)]];
        let ap = vec![vec![(0, 0.0), (1, 1.0)]];
        let p = precision(&gt, &ap);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_half() {
        let gt = vec![vec![(0, 0.0), (1, 1.0)]];
        let ap = vec![vec![(0, 0.0), (5, 10.0)]]; // 1 true of 2 returned
        let p = precision(&gt, &ap);
        assert!((p - 0.5).abs() < 1e-10);
    }

    // ── QPS measurement ─────────────────────────────────────────────────────

    #[test]
    fn test_measure_qps() {
        let queries = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let data = simple_dataset();
        let result = measure_qps(&queries, |q| {
            // Trivial linear scan
            let mut dists: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(3);
            dists
        });
        assert!(result.qps > 0.0, "QPS should be positive");
        assert_eq!(result.total_queries, 2);
        assert_eq!(result.latencies.len(), 2);
    }

    #[test]
    fn test_qps_latency_percentiles() {
        let queries: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, 0.0]).collect();
        let result = measure_qps(&queries, |_q| vec![(0, 0.0)]);
        // p50 <= p95 <= p99 (monotonicity)
        assert!(result.p50() <= result.p95());
        assert!(result.p95() <= result.p99());
    }

    #[test]
    fn test_qps_mean_latency() {
        let queries = vec![vec![0.0], vec![1.0]];
        let result = measure_qps(&queries, |_q| vec![(0, 0.0)]);
        assert!(result.mean_latency() >= result.min_latency());
        assert!(result.mean_latency() <= result.max_latency());
    }

    // ── Build timer ─────────────────────────────────────────────────────────

    #[test]
    fn test_build_timer() {
        let timer = BuildTimer::start("test_build");
        // Do a tiny amount of work
        let _sum: u64 = (0..1000).sum();
        let result = timer.stop();
        assert_eq!(result.label, "test_build");
        assert!(result.duration >= Duration::ZERO);
    }

    // ── Memory estimation ───────────────────────────────────────────────────

    #[test]
    fn test_estimate_flat_memory() {
        let mem = estimate_flat_memory(1000, 128);
        // 1000 * 128 * 4 = 512,000 bytes
        assert_eq!(mem, 512_000);
    }

    #[test]
    fn test_estimate_hnsw_memory() {
        let mem = estimate_hnsw_memory(1000, 128, 16, 4);
        let vector_bytes = 1000 * 128 * 4;
        let graph_bytes = 1000 * 16 * 4 * 8; // usize = 8 bytes on 64-bit
        assert_eq!(mem, vector_bytes + graph_bytes);
    }

    #[test]
    fn test_estimate_pq_memory() {
        let mem = estimate_pq_memory(10_000, 8);
        assert_eq!(mem, 80_000);
    }

    // ── Distance ratio ──────────────────────────────────────────────────────

    #[test]
    fn test_distance_ratio_perfect() {
        let gt = vec![vec![(0, 1.0), (1, 2.0)]];
        let ap = vec![vec![(0, 1.0), (1, 2.0)]];
        let ratio = average_distance_ratio(&gt, &ap);
        assert!((ratio - 1.0).abs() < 1e-6, "Perfect match → ratio = 1.0");
    }

    #[test]
    fn test_distance_ratio_worse() {
        let gt = vec![vec![(0, 1.0), (1, 2.0)]];
        let ap = vec![vec![(0, 2.0), (1, 4.0)]]; // double the true distances
        let ratio = average_distance_ratio(&gt, &ap);
        assert!(
            (ratio - 2.0).abs() < 1e-6,
            "Double distances → ratio = 2.0, got {ratio}"
        );
    }

    #[test]
    fn test_distance_ratio_empty() {
        let ratio = average_distance_ratio(&[], &[]);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    // ── Precision-recall sweep ──────────────────────────────────────────────

    #[test]
    fn test_precision_recall_sweep() {
        let data = simple_dataset();
        let queries = simple_queries();
        let gt = brute_force_knn(&data, &queries, 3);
        let params = vec!["exact".to_string()];
        let curve = precision_recall_sweep(&gt, &queries, &params, |_param, qs| {
            brute_force_knn(&data, qs, 3) // exact → perfect recall
        });
        assert_eq!(curve.len(), 1);
        assert!((curve[0].recall - 1.0).abs() < 1e-10);
        assert!((curve[0].precision - 1.0).abs() < 1e-10);
    }

    // ── Benchmark report ────────────────────────────────────────────────────

    #[test]
    fn test_report_text() {
        let report = BenchmarkReport {
            index_name: "HNSW".to_string(),
            dataset_size: 10_000,
            dimension: 128,
            n_queries: 1000,
            k: 10,
            recall: 0.95,
            precision: 0.93,
            qps: 5000.0,
            p50_us: 100,
            p95_us: 250,
            p99_us: 500,
            memory_bytes: 10_000_000,
            build_time_ms: 1500,
            metadata: HashMap::new(),
        };
        let text = report.to_text();
        assert!(text.contains("HNSW"));
        assert!(text.contains("10000"));
        assert!(text.contains("0.95"));
    }

    #[test]
    fn test_report_json() {
        let report = BenchmarkReport {
            index_name: "Flat".to_string(),
            dataset_size: 5000,
            dimension: 64,
            n_queries: 500,
            k: 5,
            recall: 1.0,
            precision: 1.0,
            qps: 2000.0,
            p50_us: 200,
            p95_us: 400,
            p99_us: 800,
            memory_bytes: 1_280_000,
            build_time_ms: 0,
            metadata: HashMap::new(),
        };
        let json = report.to_json();
        assert!(json.contains("\"index_name\": \"Flat\""));
        assert!(json.contains("\"recall\": 1.0"));
    }

    #[test]
    fn test_report_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("ef_search".to_string(), "64".to_string());
        let report = BenchmarkReport {
            index_name: "HNSW".to_string(),
            dataset_size: 100,
            dimension: 16,
            n_queries: 10,
            k: 5,
            recall: 0.8,
            precision: 0.8,
            qps: 100.0,
            p50_us: 500,
            p95_us: 1000,
            p99_us: 2000,
            memory_bytes: 10_000,
            build_time_ms: 100,
            metadata: meta,
        };
        let text = report.to_text();
        assert!(text.contains("ef_search"));
        assert!(text.contains("64"));
    }

    // ── Percentile helper ───────────────────────────────────────────────────

    #[test]
    fn test_percentile_empty() {
        let p = percentile_duration(&[], 50.0);
        assert_eq!(p, Duration::ZERO);
    }

    #[test]
    fn test_percentile_single() {
        let durs = vec![Duration::from_micros(100)];
        let p = percentile_duration(&durs, 50.0);
        assert_eq!(p, Duration::from_micros(100));
    }

    #[test]
    fn test_percentile_sorted() {
        let durs: Vec<Duration> = (1..=100).map(Duration::from_micros).collect();
        let p50 = percentile_duration(&durs, 50.0);
        let p99 = percentile_duration(&durs, 99.0);
        assert!(p50 < p99);
        // p50 should be around 50 µs
        assert!(p50.as_micros() >= 49 && p50.as_micros() <= 51);
    }

    // ── Integration: end-to-end benchmark ───────────────────────────────────

    #[test]
    fn test_end_to_end_benchmark() {
        let data = simple_dataset();
        let queries = simple_queries();
        let k = 3;

        // Ground truth
        let gt = brute_force_knn(&data, &queries, k);
        assert_eq!(gt.len(), queries.len());

        // "Approximate" search (same as exact for this test)
        let qps_result = measure_qps(&queries, |q| {
            let mut dists: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);
            dists
        });

        // Collect approximate results
        let approx: Vec<Vec<Neighbour>> = queries
            .iter()
            .map(|q| {
                let mut dists: Vec<(usize, f32)> = data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                        (i, d)
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.truncate(k);
                dists
            })
            .collect();

        let recall = recall_at_k(&gt, &approx);
        assert!(
            (recall - 1.0).abs() < 1e-10,
            "Exact search should give recall = 1.0"
        );

        let prec = precision(&gt, &approx);
        assert!((prec - 1.0).abs() < 1e-10);

        assert!(qps_result.qps > 0.0);
    }
}
