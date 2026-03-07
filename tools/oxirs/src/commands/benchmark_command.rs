//! # Benchmark Command
//!
//! SPARQL endpoint benchmark CLI command providing configurable load testing
//! with statistical analysis (min/max/avg/percentiles/QPS).
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::benchmark_command::{BenchmarkCommand, BenchmarkConfig};
//!
//! let cmd = BenchmarkCommand::new();
//! let config = BenchmarkConfig {
//!     endpoint_url: "http://localhost:3030/sparql".to_string(),
//!     queries: vec!["SELECT * WHERE { ?s ?p ?o } LIMIT 10".to_string()],
//!     iterations: 10,
//!     concurrency: 1,
//!     timeout_ms: 5_000,
//!     warmup_iterations: 2,
//! };
//!
//! let errors = cmd.validate_config(&config);
//! assert!(errors.is_empty());
//!
//! let report = cmd.run_simulated(&config);
//! assert_eq!(report.per_query_stats.len(), 1);
//! ```

// ─── BenchmarkConfig ─────────────────────────────────────────────────────────

/// Configuration for a SPARQL endpoint benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// URL of the SPARQL endpoint to benchmark.
    pub endpoint_url: String,
    /// List of SPARQL queries to benchmark.
    pub queries: Vec<String>,
    /// Number of iterations per query (excluding warm-up).
    pub iterations: usize,
    /// Number of concurrent workers (simulated in `run_simulated`).
    pub concurrency: usize,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Number of warm-up iterations (not included in statistics).
    pub warmup_iterations: usize,
}

// ─── QueryResult ─────────────────────────────────────────────────────────────

/// The outcome of a single query execution attempt.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Zero-based index of the query in `BenchmarkConfig::queries`.
    pub query_index: usize,
    /// Zero-based iteration number (0 = first measured iteration).
    pub iteration: usize,
    /// Measured latency in milliseconds.
    pub latency_ms: u64,
    /// `true` when the query executed without error.
    pub success: bool,
    /// Number of result rows returned.
    pub row_count: usize,
    /// Non-`None` when `success == false`.
    pub error: Option<String>,
}

// ─── BenchmarkStats ──────────────────────────────────────────────────────────

/// Aggregated statistics for all iterations of a single query.
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// Zero-based query index.
    pub query_index: usize,
    /// Total number of measured iterations.
    pub total_runs: usize,
    /// Number of successful iterations.
    pub successful: usize,
    /// Number of failed iterations.
    pub failed: usize,
    /// Minimum latency in milliseconds.
    pub min_ms: u64,
    /// Maximum latency in milliseconds.
    pub max_ms: u64,
    /// Mean latency in milliseconds.
    pub avg_ms: f64,
    /// 50th-percentile (median) latency.
    pub p50_ms: u64,
    /// 95th-percentile latency.
    pub p95_ms: u64,
    /// 99th-percentile latency.
    pub p99_ms: u64,
    /// Queries per second (successful runs / total elapsed seconds).
    pub qps: f64,
}

// ─── BenchmarkReport ─────────────────────────────────────────────────────────

/// Full benchmark report produced by a single `run_simulated` call.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// The configuration used for this benchmark run.
    pub config: BenchmarkConfig,
    /// Per-query aggregated statistics.
    pub per_query_stats: Vec<BenchmarkStats>,
    /// Overall queries-per-second across all queries and iterations.
    pub overall_qps: f64,
    /// Total number of failed query executions.
    pub total_errors: usize,
    /// Total benchmark wall-clock duration in milliseconds (simulated).
    pub duration_ms: u64,
    /// Overall mean latency across every measured iteration of every query (ms).
    pub overall_avg_latency_ms: f64,
}

// ─── BenchmarkCommand ────────────────────────────────────────────────────────

/// SPARQL endpoint benchmark CLI command.
pub struct BenchmarkCommand;

impl BenchmarkCommand {
    /// Creates a new `BenchmarkCommand`.
    pub fn new() -> Self {
        Self
    }

    // ── Configuration validation ──────────────────────────────────────────────

    /// Validates `config` and returns a list of human-readable error messages.
    ///
    /// An empty list means the configuration is valid.
    pub fn validate_config(&self, config: &BenchmarkConfig) -> Vec<String> {
        let mut errors = Vec::new();

        if config.endpoint_url.trim().is_empty() {
            errors.push("endpoint_url must not be empty".to_string());
        }

        if config.queries.is_empty() {
            errors.push("queries list must not be empty".to_string());
        } else {
            for (i, q) in config.queries.iter().enumerate() {
                if q.trim().is_empty() {
                    errors.push(format!("query[{i}] must not be empty"));
                }
            }
        }

        if config.iterations == 0 {
            errors.push("iterations must be greater than 0".to_string());
        }

        if config.concurrency == 0 {
            errors.push("concurrency must be greater than 0".to_string());
        }

        if config.timeout_ms == 0 {
            errors.push("timeout_ms must be greater than 0".to_string());
        }

        errors
    }

    // ── Simulated benchmark run ───────────────────────────────────────────────

    /// Runs a benchmark simulation without making real HTTP requests.
    ///
    /// Latencies are generated deterministically based on `query_index` and
    /// `iteration` so that results are reproducible in tests:
    ///
    /// ```text
    /// base_latency = 10 + (query_index * 5)
    /// latency      = base_latency + (iteration % 20)
    /// row_count    = (query_index + 1) * 10
    /// ```
    ///
    /// Every 7th iteration (iteration % 7 == 6) is simulated as a failure.
    pub fn run_simulated(&self, config: &BenchmarkConfig) -> BenchmarkReport {
        let mut all_results: Vec<Vec<QueryResult>> = Vec::new();
        let mut total_latency: u64 = 0;

        for (qi, _query) in config.queries.iter().enumerate() {
            let mut results: Vec<QueryResult> = Vec::new();

            // Warm-up iterations: run but discard (still advance the sequence)
            for _wi in 0..config.warmup_iterations {
                // Intentionally not stored in results
            }

            // Measured iterations
            for iter in 0..config.iterations {
                let base_latency = 10u64 + (qi as u64 * 5);
                let latency_ms = base_latency + (iter as u64 % 20);
                let is_failure = iter % 7 == 6;
                let row_count = if is_failure { 0 } else { (qi + 1) * 10 };

                total_latency += latency_ms;
                results.push(QueryResult {
                    query_index: qi,
                    iteration: iter,
                    latency_ms,
                    success: !is_failure,
                    row_count,
                    error: if is_failure {
                        Some("simulated timeout".to_string())
                    } else {
                        None
                    },
                });
            }

            all_results.push(results);
        }

        // Compute per-query stats
        let per_query_stats: Vec<BenchmarkStats> = all_results
            .iter()
            .enumerate()
            .map(|(qi, results)| {
                let mut stats = self.compute_stats(results);
                stats.query_index = qi;
                stats
            })
            .collect();

        // Simulated total duration: sum of average latency per query
        let duration_ms: u64 = per_query_stats.iter().map(|s| s.avg_ms as u64).sum::<u64>()
            + per_query_stats.len() as u64;

        let total_successful: usize = per_query_stats.iter().map(|s| s.successful).sum();
        let total_errors: usize = per_query_stats.iter().map(|s| s.failed).sum();

        let overall_qps = if duration_ms > 0 {
            (total_successful as f64) / (duration_ms as f64 / 1_000.0)
        } else {
            0.0
        };

        let total_iterations: usize = config.queries.len() * config.iterations;
        let overall_avg_latency_ms = if total_iterations > 0 {
            total_latency as f64 / total_iterations as f64
        } else {
            0.0
        };

        BenchmarkReport {
            config: config.clone(),
            per_query_stats,
            overall_qps,
            total_errors,
            duration_ms,
            overall_avg_latency_ms,
        }
    }

    // ── Statistics computation ────────────────────────────────────────────────

    /// Computes aggregated statistics from a slice of [`QueryResult`].
    ///
    /// If `results` is empty, all numeric fields are zero/0.0.
    /// The `query_index` in the returned struct is taken from the first result
    /// (or 0 when empty).
    ///
    /// Percentiles are computed by:
    /// 1. Collecting all latencies for successful runs into a sorted `Vec<u64>`.
    /// 2. Picking the element at `floor(p * n / 100)` (clamped to last index).
    pub fn compute_stats(&self, results: &[QueryResult]) -> BenchmarkStats {
        if results.is_empty() {
            return BenchmarkStats {
                query_index: 0,
                total_runs: 0,
                successful: 0,
                failed: 0,
                min_ms: 0,
                max_ms: 0,
                avg_ms: 0.0,
                p50_ms: 0,
                p95_ms: 0,
                p99_ms: 0,
                qps: 0.0,
            };
        }

        let query_index = results[0].query_index;
        let total_runs = results.len();
        let successful = results.iter().filter(|r| r.success).count();
        let failed = total_runs - successful;

        // Collect all latencies (including failed — they represent real timing)
        let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
        latencies.sort_unstable();

        let min_ms = *latencies.first().unwrap_or(&0);
        let max_ms = *latencies.last().unwrap_or(&0);
        let sum: u64 = latencies.iter().sum();
        let avg_ms = if total_runs > 0 {
            sum as f64 / total_runs as f64
        } else {
            0.0
        };

        let p50_ms = Self::percentile(&latencies, 50);
        let p95_ms = Self::percentile(&latencies, 95);
        let p99_ms = Self::percentile(&latencies, 99);

        // QPS: successful runs / total time in seconds
        // Total time is estimated as sum of all latencies (sequential model)
        let total_time_sec = sum as f64 / 1_000.0;
        let qps = if total_time_sec > 0.0 {
            successful as f64 / total_time_sec
        } else {
            0.0
        };

        BenchmarkStats {
            query_index,
            total_runs,
            successful,
            failed,
            min_ms,
            max_ms,
            avg_ms,
            p50_ms,
            p95_ms,
            p99_ms,
            qps,
        }
    }

    /// Returns the value at the given percentile `p` (0–100) from a
    /// **sorted** slice.  Returns 0 for an empty slice.
    fn percentile(sorted: &[u64], p: usize) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        let n = sorted.len();
        // index = floor(p * n / 100), clamped to [0, n-1]
        let index = (p * n / 100).min(n - 1);
        sorted[index]
    }

    // ── Report formatting ─────────────────────────────────────────────────────

    /// Formats a [`BenchmarkReport`] as a human-readable multi-line string
    /// with a table of per-query statistics.
    pub fn format_report(&self, report: &BenchmarkReport) -> String {
        let mut out = String::new();

        out.push_str("=== SPARQL Benchmark Report ===\n");
        out.push_str(&format!("Endpoint : {}\n", report.config.endpoint_url));
        out.push_str(&format!(
            "Queries  : {} | Iterations: {} | Concurrency: {}\n",
            report.config.queries.len(),
            report.config.iterations,
            report.config.concurrency,
        ));
        out.push_str(&format!(
            "Warmup   : {} iterations | Timeout: {} ms\n",
            report.config.warmup_iterations, report.config.timeout_ms,
        ));
        out.push_str(&format!("Duration : {} ms\n", report.duration_ms));
        out.push_str(&format!(
            "Overall  : {:.2} QPS | {} errors | avg latency: {:.2} ms\n",
            report.overall_qps, report.total_errors, report.overall_avg_latency_ms,
        ));
        out.push('\n');

        // Table header
        out.push_str(&format!(
            "{:>5}  {:>6}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}\n",
            "Query",
            "Runs",
            "OK",
            "Fail",
            "Min(ms)",
            "Max(ms)",
            "Avg(ms)",
            "P50(ms)",
            "P95(ms)",
            "P99(ms)",
            "QPS",
        ));
        out.push_str(&"-".repeat(99));
        out.push('\n');

        for stats in &report.per_query_stats {
            out.push_str(&format!(
                "{:>5}  {:>6}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8.2}  {:>8}  {:>8}  {:>8}  {:>8.2}\n",
                stats.query_index,
                stats.total_runs,
                stats.successful,
                stats.failed,
                stats.min_ms,
                stats.max_ms,
                stats.avg_ms,
                stats.p50_ms,
                stats.p95_ms,
                stats.p99_ms,
                stats.qps,
            ));
        }

        out
    }
}

impl Default for BenchmarkCommand {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> BenchmarkConfig {
        BenchmarkConfig {
            endpoint_url: "http://localhost:3030/sparql".to_string(),
            queries: vec![
                "SELECT * WHERE { ?s ?p ?o } LIMIT 10".to_string(),
                "SELECT ?s WHERE { ?s a <http://example.org/Thing> }".to_string(),
            ],
            iterations: 20,
            concurrency: 4,
            timeout_ms: 5_000,
            warmup_iterations: 3,
        }
    }

    fn single_query_config(iterations: usize) -> BenchmarkConfig {
        BenchmarkConfig {
            endpoint_url: "http://localhost:3030/sparql".to_string(),
            queries: vec!["SELECT * WHERE { ?s ?p ?o } LIMIT 1".to_string()],
            iterations,
            concurrency: 1,
            timeout_ms: 1_000,
            warmup_iterations: 0,
        }
    }

    // ── validate_config ───────────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_config_returns_empty() {
        let cmd = BenchmarkCommand::new();
        let errors = cmd.validate_config(&valid_config());
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn test_validate_empty_endpoint_url() {
        let cmd = BenchmarkCommand::new();
        let mut cfg = valid_config();
        cfg.endpoint_url = String::new();
        let errors = cmd.validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("endpoint_url")));
    }

    #[test]
    fn test_validate_whitespace_only_endpoint() {
        let cmd = BenchmarkCommand::new();
        let mut cfg = valid_config();
        cfg.endpoint_url = "   ".to_string();
        let errors = cmd.validate_config(&cfg);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validate_empty_queries_list() {
        let cmd = BenchmarkCommand::new();
        let mut cfg = valid_config();
        cfg.queries = Vec::new();
        let errors = cmd.validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("queries")));
    }

    #[test]
    fn test_validate_empty_query_string() {
        let cmd = BenchmarkCommand::new();
        let mut cfg = valid_config();
        cfg.queries = vec!["".to_string()];
        let errors = cmd.validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("query[0]")));
    }

    #[test]
    fn test_validate_zero_iterations() {
        let cmd = BenchmarkCommand::new();
        let mut cfg = valid_config();
        cfg.iterations = 0;
        let errors = cmd.validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("iterations")));
    }

    #[test]
    fn test_validate_zero_concurrency() {
        let cmd = BenchmarkCommand::new();
        let mut cfg = valid_config();
        cfg.concurrency = 0;
        let errors = cmd.validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("concurrency")));
    }

    #[test]
    fn test_validate_zero_timeout() {
        let cmd = BenchmarkCommand::new();
        let mut cfg = valid_config();
        cfg.timeout_ms = 0;
        let errors = cmd.validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("timeout")));
    }

    #[test]
    fn test_validate_multiple_errors_accumulated() {
        let cmd = BenchmarkCommand::new();
        let cfg = BenchmarkConfig {
            endpoint_url: String::new(),
            queries: Vec::new(),
            iterations: 0,
            concurrency: 0,
            timeout_ms: 0,
            warmup_iterations: 0,
        };
        let errors = cmd.validate_config(&cfg);
        assert!(errors.len() >= 4);
    }

    // ── compute_stats ─────────────────────────────────────────────────────────

    fn make_results(query_index: usize, latencies: &[(u64, bool)]) -> Vec<QueryResult> {
        latencies
            .iter()
            .enumerate()
            .map(|(i, (lat, ok))| QueryResult {
                query_index,
                iteration: i,
                latency_ms: *lat,
                success: *ok,
                row_count: if *ok { 10 } else { 0 },
                error: if *ok { None } else { Some("err".to_string()) },
            })
            .collect()
    }

    #[test]
    fn test_stats_empty_results() {
        let cmd = BenchmarkCommand::new();
        let stats = cmd.compute_stats(&[]);
        assert_eq!(stats.total_runs, 0);
        assert_eq!(stats.min_ms, 0);
        assert_eq!(stats.max_ms, 0);
        assert_eq!(stats.avg_ms, 0.0);
    }

    #[test]
    fn test_stats_single_result() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(0, &[(100, true)]);
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.total_runs, 1);
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.min_ms, 100);
        assert_eq!(stats.max_ms, 100);
        assert!((stats.avg_ms - 100.0).abs() < 1e-9);
        assert_eq!(stats.p50_ms, 100);
        assert_eq!(stats.p95_ms, 100);
        assert_eq!(stats.p99_ms, 100);
    }

    #[test]
    fn test_stats_min_max() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(0, &[(50, true), (200, true), (100, true), (10, true)]);
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.min_ms, 10);
        assert_eq!(stats.max_ms, 200);
    }

    #[test]
    fn test_stats_average() {
        let cmd = BenchmarkCommand::new();
        // 10, 20, 30 → avg = 20
        let results = make_results(0, &[(10, true), (20, true), (30, true)]);
        let stats = cmd.compute_stats(&results);
        assert!((stats.avg_ms - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_success_failure_count() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(
            1,
            &[(10, true), (15, false), (20, true), (25, false), (30, true)],
        );
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.successful, 3);
        assert_eq!(stats.failed, 2);
        assert_eq!(stats.total_runs, 5);
    }

    #[test]
    fn test_stats_p50_median() {
        let cmd = BenchmarkCommand::new();
        // 1,2,3,4,5 → sorted, p50 index = 50*5/100 = 2 → value=3
        let results = make_results(0, &[(5, true), (1, true), (3, true), (2, true), (4, true)]);
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.p50_ms, 3);
    }

    #[test]
    fn test_stats_p95() {
        let cmd = BenchmarkCommand::new();
        // 20 items: 10..29
        let latencies: Vec<(u64, bool)> = (10..30).map(|v| (v, true)).collect();
        let results = make_results(0, &latencies);
        let stats = cmd.compute_stats(&results);
        // p95 index = 95*20/100 = 19 → sorted[19] = 29
        assert_eq!(stats.p95_ms, 29);
    }

    #[test]
    fn test_stats_p99() {
        let cmd = BenchmarkCommand::new();
        // 100 items: 1..=100
        let latencies: Vec<(u64, bool)> = (1u64..=100).map(|v| (v, true)).collect();
        let results = make_results(0, &latencies);
        let stats = cmd.compute_stats(&results);
        // p99 index = 99*100/100 = 99 → sorted[99] = 100
        assert_eq!(stats.p99_ms, 100);
    }

    #[test]
    fn test_stats_qps_is_positive() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(0, &[(100, true), (200, true), (150, true)]);
        let stats = cmd.compute_stats(&results);
        assert!(stats.qps > 0.0);
    }

    #[test]
    fn test_stats_query_index_preserved() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(3, &[(50, true)]);
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.query_index, 3);
    }

    #[test]
    fn test_stats_all_failures() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(0, &[(100, false), (200, false)]);
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 2);
    }

    // ── run_simulated ─────────────────────────────────────────────────────────

    #[test]
    fn test_run_simulated_produces_per_query_stats() {
        let cmd = BenchmarkCommand::new();
        let report = cmd.run_simulated(&valid_config());
        assert_eq!(report.per_query_stats.len(), 2);
    }

    #[test]
    fn test_run_simulated_correct_total_runs() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(10);
        let report = cmd.run_simulated(&config);
        assert_eq!(report.per_query_stats[0].total_runs, 10);
    }

    #[test]
    fn test_run_simulated_deterministic() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(20);
        let r1 = cmd.run_simulated(&config);
        let r2 = cmd.run_simulated(&config);
        assert_eq!(r1.per_query_stats[0].min_ms, r2.per_query_stats[0].min_ms);
        assert_eq!(r1.per_query_stats[0].avg_ms, r2.per_query_stats[0].avg_ms);
    }

    #[test]
    fn test_run_simulated_every_7th_is_failure() {
        let cmd = BenchmarkCommand::new();
        // With 21 iterations: iterations 6, 13, 20 are failures → 3 failures
        let config = single_query_config(21);
        let report = cmd.run_simulated(&config);
        assert_eq!(report.per_query_stats[0].failed, 3);
        assert_eq!(report.per_query_stats[0].successful, 18);
    }

    #[test]
    fn test_run_simulated_no_failures_if_less_than_7() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(6);
        let report = cmd.run_simulated(&config);
        assert_eq!(report.per_query_stats[0].failed, 0);
        assert_eq!(report.per_query_stats[0].successful, 6);
    }

    #[test]
    fn test_run_simulated_total_errors_sum() {
        let cmd = BenchmarkCommand::new();
        // 2 queries × 14 iterations → each has 2 failures per query → total 4
        let config = BenchmarkConfig {
            endpoint_url: "http://localhost/sparql".to_string(),
            queries: vec!["Q1".to_string(), "Q2".to_string()],
            iterations: 14,
            concurrency: 1,
            timeout_ms: 1_000,
            warmup_iterations: 0,
        };
        let report = cmd.run_simulated(&config);
        // iterations 6 and 13 fail for each query → 2 per query × 2 queries = 4
        assert_eq!(report.total_errors, 4);
    }

    #[test]
    fn test_run_simulated_overall_qps_positive() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(10);
        let report = cmd.run_simulated(&config);
        assert!(report.overall_qps > 0.0);
    }

    #[test]
    fn test_run_simulated_duration_ms_positive() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(5);
        let report = cmd.run_simulated(&config);
        assert!(report.duration_ms > 0);
    }

    #[test]
    fn test_run_simulated_query_index_labels() {
        let cmd = BenchmarkCommand::new();
        let report = cmd.run_simulated(&valid_config());
        for (i, stats) in report.per_query_stats.iter().enumerate() {
            assert_eq!(stats.query_index, i);
        }
    }

    #[test]
    fn test_run_simulated_warmup_does_not_affect_run_count() {
        let cmd = BenchmarkCommand::new();
        let config = BenchmarkConfig {
            endpoint_url: "http://localhost/sparql".to_string(),
            queries: vec!["SELECT 1".to_string()],
            iterations: 10,
            concurrency: 1,
            timeout_ms: 1_000,
            warmup_iterations: 5, // warmup should not count
        };
        let report = cmd.run_simulated(&config);
        assert_eq!(report.per_query_stats[0].total_runs, 10);
    }

    #[test]
    fn test_run_simulated_multiple_queries_independent() {
        let cmd = BenchmarkCommand::new();
        let config = BenchmarkConfig {
            endpoint_url: "http://localhost/sparql".to_string(),
            queries: vec!["Q0".to_string(), "Q1".to_string(), "Q2".to_string()],
            iterations: 10,
            concurrency: 1,
            timeout_ms: 1_000,
            warmup_iterations: 0,
        };
        let report = cmd.run_simulated(&config);
        assert_eq!(report.per_query_stats.len(), 3);
        // Higher query index → higher base latency
        assert!(
            report.per_query_stats[1].min_ms > report.per_query_stats[0].min_ms,
            "Q1 base latency should be higher than Q0"
        );
    }

    // ── format_report ─────────────────────────────────────────────────────────

    #[test]
    fn test_format_report_contains_endpoint() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(5);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("localhost:3030"));
    }

    #[test]
    fn test_format_report_contains_header() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(5);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("Benchmark Report"));
    }

    #[test]
    fn test_format_report_contains_iterations() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(42);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("42"));
    }

    #[test]
    fn test_format_report_contains_qps() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(10);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("QPS") || output.contains("qps") || output.contains("qps"));
    }

    #[test]
    fn test_format_report_contains_errors() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(7);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("error") || output.contains("Error") || output.contains("Fail"));
    }

    #[test]
    fn test_format_report_contains_min_max() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(5);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("Min") || output.contains("min"));
        assert!(output.contains("Max") || output.contains("max"));
    }

    #[test]
    fn test_format_report_contains_percentiles() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(5);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("P50") || output.contains("p50"));
        assert!(output.contains("P95") || output.contains("p95"));
        assert!(output.contains("P99") || output.contains("p99"));
    }

    #[test]
    fn test_format_report_is_non_empty() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(5);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_format_report_contains_duration() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(5);
        let report = cmd.run_simulated(&config);
        let output = cmd.format_report(&report);
        assert!(output.contains("Duration") || output.contains("ms"));
    }

    // ── percentile helper (via compute_stats) ─────────────────────────────────

    #[test]
    fn test_percentile_two_elements_p50() {
        let cmd = BenchmarkCommand::new();
        // 10, 20 → p50 index = 50*2/100 = 1 → 20
        let results = make_results(0, &[(10, true), (20, true)]);
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.p50_ms, 20);
    }

    #[test]
    fn test_percentile_single_element_all_same() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(0, &[(42, true)]);
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.p50_ms, 42);
        assert_eq!(stats.p95_ms, 42);
        assert_eq!(stats.p99_ms, 42);
    }

    #[test]
    fn test_stats_p50_less_than_or_equal_p95() {
        let cmd = BenchmarkCommand::new();
        let latencies: Vec<(u64, bool)> = (1u64..=50).map(|v| (v * 2, true)).collect();
        let results = make_results(0, &latencies);
        let stats = cmd.compute_stats(&results);
        assert!(stats.p50_ms <= stats.p95_ms);
        assert!(stats.p95_ms <= stats.p99_ms);
    }

    #[test]
    fn test_run_simulated_overall_avg_latency_positive() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(10);
        let report = cmd.run_simulated(&config);
        assert!(
            report.overall_avg_latency_ms > 0.0,
            "overall_avg_latency_ms should be positive"
        );
    }

    #[test]
    fn test_run_simulated_avg_latency_within_range() {
        let cmd = BenchmarkCommand::new();
        // Single query, qi=0: base=10, iter in 0..10 → latencies 10..19; avg = 14.5
        let config = single_query_config(10);
        let report = cmd.run_simulated(&config);
        let expected_avg = (10u64..20).sum::<u64>() as f64 / 10.0; // 14.5
        assert!(
            (report.overall_avg_latency_ms - expected_avg).abs() < 1e-9,
            "expected {}, got {}",
            expected_avg,
            report.overall_avg_latency_ms
        );
    }

    // ── default impl ──────────────────────────────────────────────────────────

    #[test]
    fn test_benchmark_command_default() {
        let cmd = BenchmarkCommand;
        let cfg = single_query_config(1);
        let errors = cmd.validate_config(&cfg);
        assert!(errors.is_empty());
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_run_simulated_one_iteration() {
        let cmd = BenchmarkCommand::new();
        let config = single_query_config(1);
        let report = cmd.run_simulated(&config);
        assert_eq!(report.per_query_stats[0].total_runs, 1);
        assert_eq!(report.per_query_stats[0].successful, 1);
    }

    #[test]
    fn test_run_simulated_config_preserved_in_report() {
        let cmd = BenchmarkCommand::new();
        let config = valid_config();
        let report = cmd.run_simulated(&config);
        assert_eq!(report.config.endpoint_url, config.endpoint_url);
        assert_eq!(report.config.iterations, config.iterations);
    }

    #[test]
    fn test_stats_with_mixed_latencies() {
        let cmd = BenchmarkCommand::new();
        let results = make_results(
            0,
            &[
                (1, true),
                (1000, true),
                (500, true),
                (250, true),
                (750, true),
            ],
        );
        let stats = cmd.compute_stats(&results);
        assert_eq!(stats.min_ms, 1);
        assert_eq!(stats.max_ms, 1000);
        assert!(stats.avg_ms > 0.0);
    }
}
