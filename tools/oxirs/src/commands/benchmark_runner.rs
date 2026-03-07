//! SPARQL benchmark runner command.
//!
//! Executes benchmark suites over an in-memory triple store, computes
//! statistical metrics (min, max, mean, stddev, p50, p95, p99), generates
//! Markdown reports, and detects regressions against a baseline suite.

use std::collections::HashMap;
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a benchmark suite.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Suite name.
    pub name: String,
    /// SPARQL queries to benchmark (one per entry).
    pub queries: Vec<String>,
    /// Number of un-measured warm-up iterations.
    pub warmup_rounds: u32,
    /// Number of measured iterations.
    pub measurement_rounds: u32,
    /// Maximum milliseconds allowed per individual query execution.
    pub timeout_ms: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            queries: Vec::new(),
            warmup_rounds: 3,
            measurement_rounds: 10,
            timeout_ms: 5_000,
        }
    }
}

/// Per-query benchmark statistics.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Query label / SPARQL text.
    pub query: String,
    /// Minimum latency (ms).
    pub min_ms: f64,
    /// Maximum latency (ms).
    pub max_ms: f64,
    /// Mean latency (ms).
    pub mean_ms: f64,
    /// Standard deviation (ms).
    pub stddev_ms: f64,
    /// 50th percentile latency (ms).
    pub p50_ms: f64,
    /// 95th percentile latency (ms).
    pub p95_ms: f64,
    /// 99th percentile latency (ms).
    pub p99_ms: f64,
    /// Fraction of runs that timed out (0.0–1.0).
    pub error_rate: f64,
}

/// A complete benchmark suite: config + per-query results.
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub config: BenchmarkConfig,
    pub results: Vec<BenchmarkResult>,
}

/// Regression/improvement report comparing two suites.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Queries whose mean latency increased by > 10 %.
    pub regressions: Vec<String>,
    /// Queries whose mean latency decreased by > 10 %.
    pub improvements: Vec<String>,
    /// Queries within ±10 % of baseline.
    pub unchanged: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// BenchmarkRunner
// ──────────────────────────────────────────────────────────────────────────────

/// Executes benchmark suites over a simulated in-memory store.
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    /// Registered queries: label → SPARQL text.
    queries: Vec<(String, String)>,
}

impl BenchmarkRunner {
    /// Create a runner with a default configuration.
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            queries: Vec::new(),
        }
    }

    /// Create a runner with an explicit configuration.
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            config,
            queries: Vec::new(),
        }
    }

    /// Register a query.
    pub fn add_query(&mut self, name: impl Into<String>, sparql: impl Into<String>) {
        self.queries.push((name.into(), sparql.into()));
    }

    /// Execute the benchmark against a simulated triple store.
    ///
    /// `store_triples` is a slice of `(subject, predicate, object)` tuples that
    /// form the in-memory dataset.  The query is "executed" by counting matching
    /// triples (a deterministic simulation that produces realistic timing).
    pub fn run_benchmark(&self, store_triples: &[(String, String, String)]) -> BenchmarkSuite {
        let mut results = Vec::new();

        for (name, sparql) in &self.queries {
            let result = self.benchmark_single_query(name, sparql, store_triples);
            results.push(result);
        }

        BenchmarkSuite {
            config: BenchmarkConfig {
                queries: self.queries.iter().map(|(_, q)| q.clone()).collect(),
                ..self.config.clone()
            },
            results,
        }
    }

    // ── Statistics helpers ────────────────────────────────────────────────────

    /// Compute a percentile value from a sorted sample slice.
    ///
    /// `p` must be in [0.0, 100.0].
    pub fn compute_percentile(samples: &[f64], p: f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    // ── Reporting ─────────────────────────────────────────────────────────────

    /// Generate a Markdown table report for a `BenchmarkSuite`.
    pub fn generate_report(suite: &BenchmarkSuite) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Benchmark Report: {}\n\n", suite.config.name));
        out.push_str(&format!(
            "Configuration: warmup={}, rounds={}, timeout={}ms\n\n",
            suite.config.warmup_rounds, suite.config.measurement_rounds, suite.config.timeout_ms
        ));

        out.push_str(
            "| Query | Min (ms) | Max (ms) | Mean (ms) | StdDev | p50 | p95 | p99 | Error % |\n",
        );
        out.push_str(
            "|-------|----------|----------|-----------|--------|-----|-----|-----|---------|\n",
        );

        for r in &suite.results {
            out.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1} |\n",
                r.query,
                r.min_ms,
                r.max_ms,
                r.mean_ms,
                r.stddev_ms,
                r.p50_ms,
                r.p95_ms,
                r.p99_ms,
                r.error_rate * 100.0
            ));
        }
        out
    }

    /// Compare two suites and produce a regression report.
    ///
    /// A regression is when the mean latency of a query in `current` is more
    /// than 10 % higher than in `baseline`. An improvement is the reverse.
    pub fn compare_suites(baseline: &BenchmarkSuite, current: &BenchmarkSuite) -> ComparisonReport {
        let base_map: HashMap<&str, f64> = baseline
            .results
            .iter()
            .map(|r| (r.query.as_str(), r.mean_ms))
            .collect();

        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut unchanged = Vec::new();

        for result in &current.results {
            if let Some(&base_mean) = base_map.get(result.query.as_str()) {
                let delta = if base_mean > 0.0 {
                    (result.mean_ms - base_mean) / base_mean
                } else {
                    0.0
                };
                if delta > 0.10 {
                    regressions.push(format!(
                        "{}: {:.3}ms → {:.3}ms (+{:.1}%)",
                        result.query,
                        base_mean,
                        result.mean_ms,
                        delta * 100.0
                    ));
                } else if delta < -0.10 {
                    improvements.push(format!(
                        "{}: {:.3}ms → {:.3}ms ({:.1}%)",
                        result.query,
                        base_mean,
                        result.mean_ms,
                        delta * 100.0
                    ));
                } else {
                    unchanged.push(result.query.clone());
                }
            } else {
                // New query not in baseline
                improvements.push(format!("{}: new query", result.query));
            }
        }

        ComparisonReport {
            regressions,
            improvements,
            unchanged,
        }
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn benchmark_single_query(
        &self,
        name: &str,
        sparql: &str,
        store: &[(String, String, String)],
    ) -> BenchmarkResult {
        let total_rounds = self.config.warmup_rounds + self.config.measurement_rounds;
        let mut samples: Vec<f64> = Vec::with_capacity(self.config.measurement_rounds as usize);
        let mut errors = 0u32;

        for round in 0..total_rounds {
            let t0 = Instant::now();
            // Simulate query execution: count matching triples
            let _result = simulate_sparql(sparql, store);
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;

            if elapsed_ms > self.config.timeout_ms as f64 {
                errors += 1;
            }

            // Only record measurement rounds
            if round >= self.config.warmup_rounds {
                samples.push(elapsed_ms);
            }
        }

        compute_benchmark_result(
            name.to_string(),
            &samples,
            errors,
            self.config.measurement_rounds,
        )
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Simulate SPARQL execution by scanning the triple store.
fn simulate_sparql(sparql: &str, store: &[(String, String, String)]) -> usize {
    // Very simple scan: count triples whose predicate appears in the query text.
    // This gives deterministic-ish behaviour without a real SPARQL engine.
    let upper = sparql.to_uppercase();
    if upper.contains("COUNT") {
        store.len()
    } else {
        store
            .iter()
            .filter(|(s, p, _)| sparql.contains(s.as_str()) || sparql.contains(p.as_str()))
            .count()
    }
}

/// Compute `BenchmarkResult` from raw timing samples.
fn compute_benchmark_result(
    query: String,
    samples: &[f64],
    errors: u32,
    total: u32,
) -> BenchmarkResult {
    if samples.is_empty() {
        return BenchmarkResult {
            query,
            min_ms: 0.0,
            max_ms: 0.0,
            mean_ms: 0.0,
            stddev_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            error_rate: if total > 0 {
                errors as f64 / total as f64
            } else {
                0.0
            },
        };
    }

    let min_ms = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance =
        samples.iter().map(|&s| (s - mean_ms).powi(2)).sum::<f64>() / samples.len().max(1) as f64;
    let stddev_ms = variance.sqrt();

    let p50_ms = BenchmarkRunner::compute_percentile(samples, 50.0);
    let p95_ms = BenchmarkRunner::compute_percentile(samples, 95.0);
    let p99_ms = BenchmarkRunner::compute_percentile(samples, 99.0);

    let error_rate = if total > 0 {
        errors as f64 / total as f64
    } else {
        0.0
    };

    BenchmarkResult {
        query,
        min_ms,
        max_ms,
        mean_ms,
        stddev_ms,
        p50_ms,
        p95_ms,
        p99_ms,
        error_rate,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_store() -> Vec<(String, String, String)> {
        (0..10)
            .map(|i| {
                (
                    format!("http://example.org/s{i}"),
                    "http://example.org/p".to_string(),
                    format!("http://example.org/o{i}"),
                )
            })
            .collect()
    }

    // ── BenchmarkConfig ───────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = BenchmarkConfig::default();
        assert_eq!(cfg.warmup_rounds, 3);
        assert_eq!(cfg.measurement_rounds, 10);
        assert_eq!(cfg.timeout_ms, 5_000);
    }

    // ── BenchmarkRunner::new ──────────────────────────────────────────────────

    #[test]
    fn test_runner_new() {
        let runner = BenchmarkRunner::new();
        assert!(runner.queries.is_empty());
    }

    #[test]
    fn test_runner_default() {
        let runner = BenchmarkRunner::default();
        assert!(runner.queries.is_empty());
    }

    #[test]
    fn test_runner_with_config() {
        let cfg = BenchmarkConfig {
            name: "my-suite".to_string(),
            ..BenchmarkConfig::default()
        };
        let runner = BenchmarkRunner::with_config(cfg);
        assert_eq!(runner.config.name, "my-suite");
    }

    // ── add_query ─────────────────────────────────────────────────────────────

    #[test]
    fn test_add_query() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q1", "SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(runner.queries.len(), 1);
    }

    #[test]
    fn test_add_multiple_queries() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q1", "SELECT * WHERE { ?s ?p ?o }");
        runner.add_query("q2", "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }");
        assert_eq!(runner.queries.len(), 2);
    }

    // ── run_benchmark ─────────────────────────────────────────────────────────

    #[test]
    fn test_run_benchmark_returns_results() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("count", "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        assert_eq!(suite.results.len(), 1);
    }

    #[test]
    fn test_run_benchmark_result_fields_finite() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        let r = &suite.results[0];
        assert!(r.min_ms.is_finite());
        assert!(r.max_ms.is_finite());
        assert!(r.mean_ms.is_finite());
        assert!(r.stddev_ms.is_finite());
        assert!(r.p50_ms.is_finite());
        assert!(r.p95_ms.is_finite());
        assert!(r.p99_ms.is_finite());
    }

    #[test]
    fn test_run_benchmark_min_le_max() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        let r = &suite.results[0];
        assert!(r.min_ms <= r.max_ms);
    }

    #[test]
    fn test_run_benchmark_mean_between_min_max() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        let r = &suite.results[0];
        assert!(r.mean_ms >= r.min_ms);
        assert!(r.mean_ms <= r.max_ms + 1e-9); // allow floating-point rounding
    }

    #[test]
    fn test_run_benchmark_percentiles_ordered() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        let r = &suite.results[0];
        assert!(r.p50_ms <= r.p95_ms + 1e-9);
        assert!(r.p95_ms <= r.p99_ms + 1e-9);
    }

    #[test]
    fn test_run_benchmark_error_rate_zero_on_fast_queries() {
        let mut runner = BenchmarkRunner::with_config(BenchmarkConfig {
            timeout_ms: 60_000, // 60 s timeout — always OK
            measurement_rounds: 5,
            warmup_rounds: 1,
            ..BenchmarkConfig::default()
        });
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        assert_eq!(suite.results[0].error_rate, 0.0);
    }

    #[test]
    fn test_run_benchmark_query_name_preserved() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("my-query", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        assert_eq!(suite.results[0].query, "my-query");
    }

    #[test]
    fn test_run_benchmark_empty_store() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&[]);
        assert_eq!(suite.results.len(), 1);
    }

    #[test]
    fn test_run_benchmark_multiple_queries() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q1", "SELECT * WHERE { ?s ?p ?o }");
        runner.add_query("q2", "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }");
        runner.add_query("q3", "SELECT ?s WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        assert_eq!(suite.results.len(), 3);
    }

    // ── compute_percentile ────────────────────────────────────────────────────

    #[test]
    fn test_percentile_empty() {
        assert_eq!(BenchmarkRunner::compute_percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        assert!((BenchmarkRunner::compute_percentile(&[42.0], 50.0) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_p0_is_min() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((BenchmarkRunner::compute_percentile(&s, 0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_p100_is_max() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((BenchmarkRunner::compute_percentile(&s, 100.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_p50_median() {
        let s = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((BenchmarkRunner::compute_percentile(&s, 50.0) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_p95() {
        let s: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p95 = BenchmarkRunner::compute_percentile(&s, 95.0);
        assert!((95.0..=100.0).contains(&p95));
    }

    #[test]
    fn test_percentile_unsorted_input() {
        let s = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        // p0 should be 1.0, p100 should be 5.0
        assert!((BenchmarkRunner::compute_percentile(&s, 0.0) - 1.0).abs() < 1e-9);
        assert!((BenchmarkRunner::compute_percentile(&s, 100.0) - 5.0).abs() < 1e-9);
    }

    // ── generate_report ───────────────────────────────────────────────────────

    #[test]
    fn test_generate_report_contains_name() {
        let mut runner = BenchmarkRunner::with_config(BenchmarkConfig {
            name: "my-bench".to_string(),
            ..BenchmarkConfig::default()
        });
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        let report = BenchmarkRunner::generate_report(&suite);
        assert!(report.contains("my-bench"));
    }

    #[test]
    fn test_generate_report_contains_table_header() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("q", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        let report = BenchmarkRunner::generate_report(&suite);
        assert!(report.contains("| Query |"));
        assert!(report.contains("Min (ms)"));
    }

    #[test]
    fn test_generate_report_contains_query_row() {
        let mut runner = BenchmarkRunner::new();
        runner.add_query("my-query", "SELECT * WHERE { ?s ?p ?o }");
        let suite = runner.run_benchmark(&small_store());
        let report = BenchmarkRunner::generate_report(&suite);
        assert!(report.contains("my-query"));
    }

    #[test]
    fn test_generate_report_no_queries() {
        let runner = BenchmarkRunner::new();
        let suite = runner.run_benchmark(&small_store());
        let report = BenchmarkRunner::generate_report(&suite);
        assert!(report.contains("# Benchmark Report"));
    }

    // ── compare_suites ────────────────────────────────────────────────────────

    #[test]
    fn test_compare_suites_unchanged_same_perf() {
        // Create two identical synthetic suites
        let make_suite = |mean: f64| BenchmarkSuite {
            config: BenchmarkConfig::default(),
            results: vec![BenchmarkResult {
                query: "q1".to_string(),
                min_ms: mean * 0.9,
                max_ms: mean * 1.1,
                mean_ms: mean,
                stddev_ms: 0.01,
                p50_ms: mean,
                p95_ms: mean * 1.05,
                p99_ms: mean * 1.1,
                error_rate: 0.0,
            }],
        };

        let baseline = make_suite(10.0);
        let current = make_suite(10.0);
        let report = BenchmarkRunner::compare_suites(&baseline, &current);
        assert!(report.regressions.is_empty());
        assert_eq!(report.unchanged, vec!["q1"]);
    }

    #[test]
    fn test_compare_suites_regression_detected() {
        let make_suite = |mean: f64| BenchmarkSuite {
            config: BenchmarkConfig::default(),
            results: vec![BenchmarkResult {
                query: "q1".to_string(),
                min_ms: mean,
                max_ms: mean,
                mean_ms: mean,
                stddev_ms: 0.0,
                p50_ms: mean,
                p95_ms: mean,
                p99_ms: mean,
                error_rate: 0.0,
            }],
        };

        let baseline = make_suite(10.0);
        let current = make_suite(20.0); // 100% slower → regression
        let report = BenchmarkRunner::compare_suites(&baseline, &current);
        assert_eq!(report.regressions.len(), 1);
        assert!(report.regressions[0].contains("q1"));
    }

    #[test]
    fn test_compare_suites_improvement_detected() {
        let make_suite = |mean: f64| BenchmarkSuite {
            config: BenchmarkConfig::default(),
            results: vec![BenchmarkResult {
                query: "q1".to_string(),
                min_ms: mean,
                max_ms: mean,
                mean_ms: mean,
                stddev_ms: 0.0,
                p50_ms: mean,
                p95_ms: mean,
                p99_ms: mean,
                error_rate: 0.0,
            }],
        };

        let baseline = make_suite(20.0);
        let current = make_suite(5.0); // 75% faster → improvement
        let report = BenchmarkRunner::compare_suites(&baseline, &current);
        assert_eq!(report.improvements.len(), 1);
        assert!(report.improvements[0].contains("q1"));
    }

    #[test]
    fn test_compare_suites_new_query_is_improvement() {
        let baseline = BenchmarkSuite {
            config: BenchmarkConfig::default(),
            results: vec![],
        };
        let current = BenchmarkSuite {
            config: BenchmarkConfig::default(),
            results: vec![BenchmarkResult {
                query: "new-q".to_string(),
                min_ms: 1.0,
                max_ms: 2.0,
                mean_ms: 1.5,
                stddev_ms: 0.5,
                p50_ms: 1.5,
                p95_ms: 2.0,
                p99_ms: 2.0,
                error_rate: 0.0,
            }],
        };
        let report = BenchmarkRunner::compare_suites(&baseline, &current);
        assert!(!report.improvements.is_empty());
        assert!(report.improvements[0].contains("new-q"));
    }

    #[test]
    fn test_compare_suites_multiple_queries() {
        let make_result = |name: &str, mean: f64| BenchmarkResult {
            query: name.to_string(),
            min_ms: mean,
            max_ms: mean,
            mean_ms: mean,
            stddev_ms: 0.0,
            p50_ms: mean,
            p95_ms: mean,
            p99_ms: mean,
            error_rate: 0.0,
        };

        let baseline = BenchmarkSuite {
            config: BenchmarkConfig::default(),
            results: vec![
                make_result("q1", 10.0),
                make_result("q2", 10.0),
                make_result("q3", 10.0),
            ],
        };
        let current = BenchmarkSuite {
            config: BenchmarkConfig::default(),
            results: vec![
                make_result("q1", 20.0), // regression
                make_result("q2", 5.0),  // improvement
                make_result("q3", 10.5), // unchanged (≤10%)
            ],
        };

        let report = BenchmarkRunner::compare_suites(&baseline, &current);
        assert_eq!(report.regressions.len(), 1);
        assert_eq!(report.improvements.len(), 1);
        assert_eq!(report.unchanged.len(), 1);
    }

    // ── stddev calculation ────────────────────────────────────────────────────

    #[test]
    fn test_stddev_uniform_samples() {
        let samples = vec![5.0; 10];
        let result = compute_benchmark_result("q".to_string(), &samples, 0, 10);
        assert!(
            result.stddev_ms < 1e-9,
            "stddev should be ~0 for uniform samples"
        );
    }

    #[test]
    fn test_stddev_nonzero_for_varied_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_benchmark_result("q".to_string(), &samples, 0, 5);
        assert!(result.stddev_ms > 0.0);
    }

    // ── simulate_sparql ───────────────────────────────────────────────────────

    #[test]
    fn test_simulate_sparql_count() {
        let store = small_store();
        let count = simulate_sparql("SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }", &store);
        assert_eq!(count, store.len());
    }

    #[test]
    fn test_simulate_sparql_empty_store() {
        let count = simulate_sparql("SELECT * WHERE { ?s ?p ?o }", &[]);
        assert_eq!(count, 0);
    }
}
