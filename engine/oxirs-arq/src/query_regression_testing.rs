//! # Query Regression Testing Framework
//!
//! This module provides a comprehensive framework for detecting performance
//! regressions in SPARQL query execution. It supports:
//!
//! - **Golden Query Sets**: Reference queries with expected performance baselines
//! - **Statistical Regression Detection**: Using significance tests and confidence intervals
//! - **Execution Recording**: Historical execution metrics with rolling windows
//! - **Automated Regression Reports**: Detailed analysis of performance changes
//! - **CI/CD Integration**: Hooks for continuous integration pipelines
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use oxirs_arq::query_regression_testing::{
//!     RegressionTestSuite, GoldenQuery, RegressionConfig, ExecutionResult,
//! };
//!
//! // Create a regression test suite
//! let config = RegressionConfig::default();
//! let mut suite = RegressionTestSuite::new("sparql_benchmarks", config);
//!
//! // Add golden queries with baselines
//! suite.add_golden_query(GoldenQuery::new(
//!     "simple_select",
//!     "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100",
//!     10.0,  // baseline_ms
//! ));
//!
//! // Record execution results
//! suite.record_execution("simple_select", ExecutionResult::success(8.5));
//!
//! // Check for regressions
//! let report = suite.analyze_regressions();
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::SystemTime;

/// Configuration for regression testing
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Threshold for regression detection (default: 1.2 = 20% slower)
    pub regression_threshold: f64,
    /// Threshold for improvement detection (default: 0.8 = 20% faster)
    pub improvement_threshold: f64,
    /// Minimum number of samples for statistical significance
    pub min_samples: usize,
    /// Rolling window size for baseline calculation
    pub rolling_window_size: usize,
    /// Confidence level for statistical tests (0.0-1.0)
    pub confidence_level: f64,
    /// Maximum history entries per query
    pub max_history_entries: usize,
    /// Enable detailed logging
    pub verbose: bool,
    /// Number of standard deviations for outlier detection
    pub outlier_std_devs: f64,
    /// Minimum execution time to consider (filters noise)
    pub min_execution_time_ms: f64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            regression_threshold: 1.2,
            improvement_threshold: 0.8,
            min_samples: 5,
            rolling_window_size: 20,
            confidence_level: 0.95,
            max_history_entries: 1000,
            verbose: false,
            outlier_std_devs: 3.0,
            min_execution_time_ms: 0.1,
        }
    }
}

impl RegressionConfig {
    /// Create a strict configuration for CI/CD
    pub fn strict() -> Self {
        Self {
            regression_threshold: 1.1, // 10% regression triggers alert
            improvement_threshold: 0.9,
            min_samples: 10,
            rolling_window_size: 30,
            confidence_level: 0.99,
            max_history_entries: 2000,
            verbose: true,
            outlier_std_devs: 2.5,
            min_execution_time_ms: 0.1,
        }
    }

    /// Create a lenient configuration for development
    pub fn lenient() -> Self {
        Self {
            regression_threshold: 1.5, // 50% regression triggers alert
            improvement_threshold: 0.5,
            min_samples: 3,
            rolling_window_size: 10,
            confidence_level: 0.90,
            max_history_entries: 500,
            verbose: false,
            outlier_std_devs: 4.0,
            min_execution_time_ms: 0.05,
        }
    }
}

/// A golden query with expected performance baseline
#[derive(Debug, Clone)]
pub struct GoldenQuery {
    /// Unique identifier for this query
    pub id: String,
    /// The SPARQL query text
    pub query: String,
    /// Description of what this query tests
    pub description: String,
    /// Baseline execution time in milliseconds
    pub baseline_ms: f64,
    /// Expected result count (optional)
    pub expected_result_count: Option<usize>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Priority (1=highest, 5=lowest)
    pub priority: u8,
    /// Whether this query is active
    pub active: bool,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub updated_at: SystemTime,
}

impl GoldenQuery {
    /// Create a new golden query with basic parameters
    pub fn new(id: impl Into<String>, query: impl Into<String>, baseline_ms: f64) -> Self {
        let now = SystemTime::now();
        Self {
            id: id.into(),
            query: query.into(),
            description: String::new(),
            baseline_ms,
            expected_result_count: None,
            tags: Vec::new(),
            priority: 3,
            active: true,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set the description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set expected result count
    pub fn with_expected_count(mut self, count: usize) -> Self {
        self.expected_result_count = Some(count);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.clamp(1, 5);
        self
    }

    /// Update the baseline
    pub fn update_baseline(&mut self, new_baseline_ms: f64) {
        self.baseline_ms = new_baseline_ms;
        self.updated_at = SystemTime::now();
    }
}

/// Result of a single query execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Whether execution succeeded
    pub success: bool,
    /// Number of results returned
    pub result_count: Option<usize>,
    /// Memory used in bytes
    pub memory_bytes: Option<usize>,
    /// Error message if failed
    pub error: Option<String>,
    /// Timestamp of execution
    pub timestamp: SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ExecutionResult {
    /// Create a successful execution result
    pub fn success(execution_time_ms: f64) -> Self {
        Self {
            execution_time_ms,
            success: true,
            result_count: None,
            memory_bytes: None,
            error: None,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a failed execution result
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            execution_time_ms: 0.0,
            success: false,
            result_count: None,
            memory_bytes: None,
            error: Some(error.into()),
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set result count
    pub fn with_result_count(mut self, count: usize) -> Self {
        self.result_count = Some(count);
        self
    }

    /// Set memory usage
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Regression status for a query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionStatus {
    /// Performance is stable
    Stable,
    /// Performance has improved
    Improved,
    /// Performance has regressed
    Regressed,
    /// Not enough data
    InsufficientData,
    /// Query is failing
    Failing,
}

impl fmt::Display for RegressionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stable => write!(f, "STABLE"),
            Self::Improved => write!(f, "IMPROVED"),
            Self::Regressed => write!(f, "REGRESSED"),
            Self::InsufficientData => write!(f, "INSUFFICIENT_DATA"),
            Self::Failing => write!(f, "FAILING"),
        }
    }
}

/// Statistics for a query's execution history
#[derive(Debug, Clone, Default)]
pub struct ExecutionStatistics {
    /// Number of executions
    pub count: usize,
    /// Number of successes
    pub success_count: usize,
    /// Number of failures
    pub failure_count: usize,
    /// Minimum execution time
    pub min_ms: f64,
    /// Maximum execution time
    pub max_ms: f64,
    /// Mean execution time
    pub mean_ms: f64,
    /// Median execution time
    pub median_ms: f64,
    /// Standard deviation
    pub std_dev_ms: f64,
    /// 95th percentile
    pub p95_ms: f64,
    /// 99th percentile
    pub p99_ms: f64,
    /// Coefficient of variation
    pub cv: f64,
}

impl ExecutionStatistics {
    /// Calculate statistics from execution results
    pub fn from_results(results: &[ExecutionResult]) -> Self {
        if results.is_empty() {
            return Self::default();
        }

        let successes: Vec<f64> = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.execution_time_ms)
            .collect();

        let success_count = successes.len();
        let failure_count = results.len() - success_count;

        if successes.is_empty() {
            return Self {
                count: results.len(),
                success_count: 0,
                failure_count,
                ..Default::default()
            };
        }

        let min_ms = successes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = successes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_ms = successes.iter().sum::<f64>() / successes.len() as f64;

        let variance = if successes.len() > 1 {
            successes.iter().map(|x| (x - mean_ms).powi(2)).sum::<f64>()
                / (successes.len() - 1) as f64
        } else {
            0.0
        };
        let std_dev_ms = variance.sqrt();

        let mut sorted = successes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_ms = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let p95_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
        let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);

        let cv = if mean_ms > 0.0 {
            std_dev_ms / mean_ms
        } else {
            0.0
        };

        Self {
            count: results.len(),
            success_count,
            failure_count,
            min_ms,
            max_ms,
            mean_ms,
            median_ms,
            std_dev_ms,
            p95_ms: sorted[p95_idx],
            p99_ms: sorted[p99_idx],
            cv,
        }
    }
}

/// Detailed regression analysis for a single query
#[derive(Debug, Clone)]
pub struct QueryRegressionAnalysis {
    /// Query ID
    pub query_id: String,
    /// Current regression status
    pub status: RegressionStatus,
    /// Baseline execution time
    pub baseline_ms: f64,
    /// Current mean execution time
    pub current_mean_ms: f64,
    /// Ratio of current to baseline (>1 = slower)
    pub ratio: f64,
    /// Percentage change from baseline
    pub change_percent: f64,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Whether the change is statistically significant
    pub is_significant: bool,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Recent execution statistics
    pub recent_stats: ExecutionStatistics,
    /// Historical execution statistics
    pub historical_stats: ExecutionStatistics,
    /// Trend direction (-1 = improving, 0 = stable, 1 = degrading)
    pub trend: i8,
    /// Detailed message
    pub message: String,
}

impl QueryRegressionAnalysis {
    /// Check if this query needs attention
    pub fn needs_attention(&self) -> bool {
        matches!(
            self.status,
            RegressionStatus::Regressed | RegressionStatus::Failing
        )
    }
}

/// Overall regression report for the test suite
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Suite name
    pub suite_name: String,
    /// Report generation timestamp
    pub generated_at: SystemTime,
    /// Overall status
    pub overall_status: RegressionStatus,
    /// Individual query analyses
    pub analyses: Vec<QueryRegressionAnalysis>,
    /// Summary statistics
    pub summary: ReportSummary,
    /// Configuration used
    pub config: RegressionConfig,
}

impl RegressionReport {
    /// Get all regressed queries
    pub fn regressed_queries(&self) -> Vec<&QueryRegressionAnalysis> {
        self.analyses
            .iter()
            .filter(|a| a.status == RegressionStatus::Regressed)
            .collect()
    }

    /// Get all improved queries
    pub fn improved_queries(&self) -> Vec<&QueryRegressionAnalysis> {
        self.analyses
            .iter()
            .filter(|a| a.status == RegressionStatus::Improved)
            .collect()
    }

    /// Get all failing queries
    pub fn failing_queries(&self) -> Vec<&QueryRegressionAnalysis> {
        self.analyses
            .iter()
            .filter(|a| a.status == RegressionStatus::Failing)
            .collect()
    }

    /// Check if the report indicates any issues
    pub fn has_issues(&self) -> bool {
        self.summary.regressed_count > 0 || self.summary.failing_count > 0
    }

    /// Generate a human-readable summary
    pub fn summary_text(&self) -> String {
        let mut text = format!("Regression Report: {}\n", self.suite_name);
        text.push_str(&format!("Generated: {:?}\n\n", self.generated_at));
        text.push_str(&format!("Overall Status: {}\n\n", self.overall_status));
        text.push_str(&format!(
            "Summary:\n  Total: {}\n  Stable: {}\n  Improved: {}\n  Regressed: {}\n  Failing: {}\n  Insufficient Data: {}\n",
            self.summary.total_count,
            self.summary.stable_count,
            self.summary.improved_count,
            self.summary.regressed_count,
            self.summary.failing_count,
            self.summary.insufficient_data_count
        ));

        if !self.regressed_queries().is_empty() {
            text.push_str("\nRegressed Queries:\n");
            for analysis in self.regressed_queries() {
                text.push_str(&format!(
                    "  - {}: {:.1}% slower ({:.2}ms -> {:.2}ms)\n",
                    analysis.query_id,
                    analysis.change_percent,
                    analysis.baseline_ms,
                    analysis.current_mean_ms
                ));
            }
        }

        if !self.improved_queries().is_empty() {
            text.push_str("\nImproved Queries:\n");
            for analysis in self.improved_queries() {
                text.push_str(&format!(
                    "  - {}: {:.1}% faster ({:.2}ms -> {:.2}ms)\n",
                    analysis.query_id,
                    -analysis.change_percent,
                    analysis.baseline_ms,
                    analysis.current_mean_ms
                ));
            }
        }

        text
    }
}

/// Summary statistics for a regression report
#[derive(Debug, Clone, Default)]
pub struct ReportSummary {
    /// Total number of queries analyzed
    pub total_count: usize,
    /// Number of stable queries
    pub stable_count: usize,
    /// Number of improved queries
    pub improved_count: usize,
    /// Number of regressed queries
    pub regressed_count: usize,
    /// Number of failing queries
    pub failing_count: usize,
    /// Number of queries with insufficient data
    pub insufficient_data_count: usize,
    /// Average regression percentage (for regressed queries)
    pub avg_regression_percent: f64,
    /// Average improvement percentage (for improved queries)
    pub avg_improvement_percent: f64,
    /// Worst regression percentage
    pub worst_regression_percent: f64,
    /// Best improvement percentage
    pub best_improvement_percent: f64,
}

/// Execution history for a query
#[derive(Debug, Clone)]
struct QueryHistory {
    /// Execution results in chronological order
    results: VecDeque<ExecutionResult>,
    /// Maximum entries
    max_entries: usize,
}

impl QueryHistory {
    fn new(max_entries: usize) -> Self {
        Self {
            results: VecDeque::new(),
            max_entries,
        }
    }

    fn add(&mut self, result: ExecutionResult) {
        self.results.push_back(result);
        while self.results.len() > self.max_entries {
            self.results.pop_front();
        }
    }

    fn recent(&self, count: usize) -> Vec<&ExecutionResult> {
        self.results.iter().rev().take(count).collect()
    }

    fn all(&self) -> Vec<&ExecutionResult> {
        self.results.iter().collect()
    }
}

/// Main regression test suite
#[derive(Debug)]
pub struct RegressionTestSuite {
    /// Suite name
    name: String,
    /// Configuration
    config: RegressionConfig,
    /// Golden queries
    golden_queries: HashMap<String, GoldenQuery>,
    /// Execution history per query
    history: HashMap<String, QueryHistory>,
    /// Statistics
    stats: SuiteStatistics,
}

/// Statistics for the test suite
#[derive(Debug, Clone, Default)]
pub struct SuiteStatistics {
    /// Total executions recorded
    pub total_executions: usize,
    /// Total analyses performed
    pub total_analyses: usize,
    /// Regressions detected
    pub regressions_detected: usize,
    /// Improvements detected
    pub improvements_detected: usize,
    /// Last analysis timestamp
    pub last_analysis: Option<SystemTime>,
}

impl RegressionTestSuite {
    /// Create a new regression test suite
    pub fn new(name: impl Into<String>, config: RegressionConfig) -> Self {
        Self {
            name: name.into(),
            config,
            golden_queries: HashMap::new(),
            history: HashMap::new(),
            stats: SuiteStatistics::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(name: impl Into<String>) -> Self {
        Self::new(name, RegressionConfig::default())
    }

    /// Add a golden query
    pub fn add_golden_query(&mut self, query: GoldenQuery) {
        let id = query.id.clone();
        self.golden_queries.insert(id.clone(), query);
        self.history
            .entry(id)
            .or_insert_with(|| QueryHistory::new(self.config.max_history_entries));
    }

    /// Remove a golden query
    pub fn remove_golden_query(&mut self, id: &str) -> Option<GoldenQuery> {
        self.history.remove(id);
        self.golden_queries.remove(id)
    }

    /// Get a golden query by ID
    pub fn get_golden_query(&self, id: &str) -> Option<&GoldenQuery> {
        self.golden_queries.get(id)
    }

    /// Get all golden queries
    pub fn golden_queries(&self) -> impl Iterator<Item = &GoldenQuery> {
        self.golden_queries.values()
    }

    /// Record an execution result
    pub fn record_execution(&mut self, query_id: &str, result: ExecutionResult) -> bool {
        if let Some(history) = self.history.get_mut(query_id) {
            history.add(result);
            self.stats.total_executions += 1;
            true
        } else if self.golden_queries.contains_key(query_id) {
            let mut history = QueryHistory::new(self.config.max_history_entries);
            history.add(result);
            self.history.insert(query_id.to_string(), history);
            self.stats.total_executions += 1;
            true
        } else {
            false
        }
    }

    /// Record multiple execution results for a query
    pub fn record_executions(&mut self, query_id: &str, results: Vec<ExecutionResult>) -> usize {
        let mut recorded = 0;
        for result in results {
            if self.record_execution(query_id, result) {
                recorded += 1;
            }
        }
        recorded
    }

    /// Analyze a single query for regression
    pub fn analyze_query(&self, query_id: &str) -> Option<QueryRegressionAnalysis> {
        let query = self.golden_queries.get(query_id)?;
        let history = self.history.get(query_id)?;

        let all_results: Vec<ExecutionResult> = history.all().into_iter().cloned().collect();
        let recent_results: Vec<ExecutionResult> = history
            .recent(self.config.rolling_window_size)
            .into_iter()
            .cloned()
            .collect();

        // Check for insufficient data
        if recent_results.len() < self.config.min_samples {
            return Some(QueryRegressionAnalysis {
                query_id: query_id.to_string(),
                status: RegressionStatus::InsufficientData,
                baseline_ms: query.baseline_ms,
                current_mean_ms: 0.0,
                ratio: 1.0,
                change_percent: 0.0,
                p_value: 1.0,
                is_significant: false,
                ci_lower: 0.0,
                ci_upper: 0.0,
                recent_stats: ExecutionStatistics::default(),
                historical_stats: ExecutionStatistics::default(),
                trend: 0,
                message: format!(
                    "Insufficient data: {} samples (need {})",
                    recent_results.len(),
                    self.config.min_samples
                ),
            });
        }

        // Check for failing queries
        let recent_failures = recent_results.iter().filter(|r| !r.success).count();
        let failure_rate = recent_failures as f64 / recent_results.len() as f64;
        if failure_rate > 0.5 {
            return Some(QueryRegressionAnalysis {
                query_id: query_id.to_string(),
                status: RegressionStatus::Failing,
                baseline_ms: query.baseline_ms,
                current_mean_ms: 0.0,
                ratio: f64::INFINITY,
                change_percent: f64::INFINITY,
                p_value: 0.0,
                is_significant: true,
                ci_lower: 0.0,
                ci_upper: 0.0,
                recent_stats: ExecutionStatistics::from_results(&recent_results),
                historical_stats: ExecutionStatistics::from_results(&all_results),
                trend: 1,
                message: format!("Query failing: {:.1}% failure rate", failure_rate * 100.0),
            });
        }

        // Calculate statistics
        let recent_stats = ExecutionStatistics::from_results(&recent_results);
        let historical_stats = ExecutionStatistics::from_results(&all_results);

        // Filter out outliers for analysis
        let filtered_times: Vec<f64> = recent_results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.execution_time_ms)
            .filter(|&t| {
                if recent_stats.std_dev_ms > 0.0 {
                    (t - recent_stats.mean_ms).abs()
                        <= self.config.outlier_std_devs * recent_stats.std_dev_ms
                } else {
                    true
                }
            })
            .collect();

        if filtered_times.is_empty() {
            return Some(QueryRegressionAnalysis {
                query_id: query_id.to_string(),
                status: RegressionStatus::InsufficientData,
                baseline_ms: query.baseline_ms,
                current_mean_ms: 0.0,
                ratio: 1.0,
                change_percent: 0.0,
                p_value: 1.0,
                is_significant: false,
                ci_lower: 0.0,
                ci_upper: 0.0,
                recent_stats,
                historical_stats,
                trend: 0,
                message: "All samples filtered as outliers".to_string(),
            });
        }

        let current_mean: f64 = filtered_times.iter().sum::<f64>() / filtered_times.len() as f64;
        let ratio = current_mean / query.baseline_ms;
        let change_percent = (ratio - 1.0) * 100.0;

        // Calculate confidence interval using t-distribution approximation
        let (ci_lower, ci_upper, p_value) =
            self.calculate_statistics(&filtered_times, query.baseline_ms);
        let is_significant = p_value < (1.0 - self.config.confidence_level);

        // Determine trend from historical data
        let trend = self.calculate_trend(&all_results);

        // Determine status
        let status = if ratio > self.config.regression_threshold && is_significant {
            RegressionStatus::Regressed
        } else if ratio < self.config.improvement_threshold && is_significant {
            RegressionStatus::Improved
        } else {
            RegressionStatus::Stable
        };

        let message = match status {
            RegressionStatus::Regressed => format!(
                "Performance regressed by {:.1}% (baseline: {:.2}ms, current: {:.2}ms)",
                change_percent, query.baseline_ms, current_mean
            ),
            RegressionStatus::Improved => format!(
                "Performance improved by {:.1}% (baseline: {:.2}ms, current: {:.2}ms)",
                -change_percent, query.baseline_ms, current_mean
            ),
            RegressionStatus::Stable => format!(
                "Performance stable ({:.1}% change, baseline: {:.2}ms, current: {:.2}ms)",
                change_percent, query.baseline_ms, current_mean
            ),
            _ => String::new(),
        };

        Some(QueryRegressionAnalysis {
            query_id: query_id.to_string(),
            status,
            baseline_ms: query.baseline_ms,
            current_mean_ms: current_mean,
            ratio,
            change_percent,
            p_value,
            is_significant,
            ci_lower,
            ci_upper,
            recent_stats,
            historical_stats,
            trend,
            message,
        })
    }

    /// Analyze all queries for regressions
    pub fn analyze_regressions(&mut self) -> RegressionReport {
        self.stats.total_analyses += 1;
        self.stats.last_analysis = Some(SystemTime::now());

        let mut analyses = Vec::new();
        for query_id in self.golden_queries.keys() {
            if let Some(analysis) = self.analyze_query(query_id) {
                if analysis.status == RegressionStatus::Regressed {
                    self.stats.regressions_detected += 1;
                } else if analysis.status == RegressionStatus::Improved {
                    self.stats.improvements_detected += 1;
                }
                analyses.push(analysis);
            }
        }

        // Sort by status priority and change magnitude
        analyses.sort_by(|a, b| {
            let status_order = |s: &RegressionStatus| match s {
                RegressionStatus::Failing => 0,
                RegressionStatus::Regressed => 1,
                RegressionStatus::Improved => 2,
                RegressionStatus::Stable => 3,
                RegressionStatus::InsufficientData => 4,
            };
            let a_order = status_order(&a.status);
            let b_order = status_order(&b.status);
            if a_order != b_order {
                a_order.cmp(&b_order)
            } else {
                b.change_percent
                    .abs()
                    .partial_cmp(&a.change_percent.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let summary = self.calculate_summary(&analyses);
        let overall_status = if summary.regressed_count > 0 || summary.failing_count > 0 {
            RegressionStatus::Regressed
        } else if summary.improved_count > 0 {
            RegressionStatus::Improved
        } else if summary.insufficient_data_count == summary.total_count {
            RegressionStatus::InsufficientData
        } else {
            RegressionStatus::Stable
        };

        RegressionReport {
            suite_name: self.name.clone(),
            generated_at: SystemTime::now(),
            overall_status,
            analyses,
            summary,
            config: self.config.clone(),
        }
    }

    /// Update baseline for a query based on recent performance
    pub fn update_baseline(&mut self, query_id: &str) -> Option<f64> {
        let history = self.history.get(query_id)?;
        let recent: Vec<f64> = history
            .recent(self.config.rolling_window_size)
            .into_iter()
            .filter(|r| r.success)
            .map(|r| r.execution_time_ms)
            .collect();

        if recent.len() >= self.config.min_samples {
            let new_baseline = recent.iter().sum::<f64>() / recent.len() as f64;
            if let Some(query) = self.golden_queries.get_mut(query_id) {
                query.update_baseline(new_baseline);
                return Some(new_baseline);
            }
        }
        None
    }

    /// Get suite statistics
    pub fn statistics(&self) -> &SuiteStatistics {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &RegressionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RegressionConfig) {
        self.config = config;
    }

    /// Clear all execution history
    pub fn clear_history(&mut self) {
        for history in self.history.values_mut() {
            history.results.clear();
        }
    }

    /// Export suite data for persistence
    pub fn export(&self) -> SuiteExport {
        SuiteExport {
            name: self.name.clone(),
            config: self.config.clone(),
            golden_queries: self.golden_queries.values().cloned().collect(),
            stats: self.stats.clone(),
        }
    }

    /// Import suite data
    pub fn import(data: SuiteExport) -> Self {
        let mut suite = Self::new(data.name, data.config);
        for query in data.golden_queries {
            suite.add_golden_query(query);
        }
        suite.stats = data.stats;
        suite
    }

    // Private helper methods

    fn calculate_statistics(&self, samples: &[f64], baseline: f64) -> (f64, f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0, 1.0);
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;

        if samples.len() < 2 {
            return (mean, mean, 0.5);
        }

        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_error = (variance / n).sqrt();

        // T-critical value for 95% CI (approximation)
        let t_critical = if n > 30.0 { 1.96 } else { 2.0 + 1.0 / n.sqrt() };

        let ci_lower = mean - t_critical * std_error;
        let ci_upper = mean + t_critical * std_error;

        // Calculate p-value using t-test against baseline
        let t_stat = (mean - baseline) / std_error;
        let p_value = self.approximate_p_value(t_stat.abs(), (n - 1.0) as usize);

        (ci_lower, ci_upper, p_value)
    }

    fn approximate_p_value(&self, t_stat: f64, df: usize) -> f64 {
        // Simple approximation of two-tailed p-value
        // For more accuracy, use a proper statistical library
        let df = df as f64;
        // Note: x would be used for incomplete beta function approximation
        let _x = df / (df + t_stat * t_stat);

        // Beta function approximation for incomplete beta
        // This is a simplified version
        if t_stat < 0.5 {
            1.0
        } else if t_stat > 5.0 {
            0.0001
        } else {
            // Linear interpolation approximation
            let p = 2.0 * (1.0 - 0.5 * (1.0 + (t_stat / (1.0 + t_stat / df.sqrt())).tanh()));
            p.clamp(0.0001, 1.0)
        }
    }

    fn calculate_trend(&self, results: &[ExecutionResult]) -> i8 {
        let times: Vec<f64> = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.execution_time_ms)
            .collect();

        if times.len() < 3 {
            return 0;
        }

        // Simple linear trend using first and last thirds
        let third = times.len() / 3;
        let first_third_avg: f64 = times[..third].iter().sum::<f64>() / third as f64;
        let last_third_avg: f64 = times[times.len() - third..].iter().sum::<f64>() / third as f64;

        let ratio = last_third_avg / first_third_avg;
        if ratio > 1.1 {
            1 // Degrading
        } else if ratio < 0.9 {
            -1 // Improving
        } else {
            0 // Stable
        }
    }

    fn calculate_summary(&self, analyses: &[QueryRegressionAnalysis]) -> ReportSummary {
        let mut summary = ReportSummary {
            total_count: analyses.len(),
            ..Default::default()
        };

        let mut regression_percents = Vec::new();
        let mut improvement_percents = Vec::new();

        for analysis in analyses {
            match analysis.status {
                RegressionStatus::Stable => summary.stable_count += 1,
                RegressionStatus::Improved => {
                    summary.improved_count += 1;
                    improvement_percents.push(-analysis.change_percent);
                }
                RegressionStatus::Regressed => {
                    summary.regressed_count += 1;
                    regression_percents.push(analysis.change_percent);
                }
                RegressionStatus::InsufficientData => summary.insufficient_data_count += 1,
                RegressionStatus::Failing => summary.failing_count += 1,
            }
        }

        if !regression_percents.is_empty() {
            summary.avg_regression_percent =
                regression_percents.iter().sum::<f64>() / regression_percents.len() as f64;
            summary.worst_regression_percent = regression_percents
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
        }

        if !improvement_percents.is_empty() {
            summary.avg_improvement_percent =
                improvement_percents.iter().sum::<f64>() / improvement_percents.len() as f64;
            summary.best_improvement_percent = improvement_percents
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
        }

        summary
    }
}

/// Export format for suite persistence
#[derive(Debug, Clone)]
pub struct SuiteExport {
    /// Suite name
    pub name: String,
    /// Configuration
    pub config: RegressionConfig,
    /// Golden queries
    pub golden_queries: Vec<GoldenQuery>,
    /// Statistics
    pub stats: SuiteStatistics,
}

/// Builder for creating regression test suites
#[derive(Debug, Default)]
pub struct RegressionTestSuiteBuilder {
    name: String,
    config: Option<RegressionConfig>,
    golden_queries: Vec<GoldenQuery>,
}

impl RegressionTestSuiteBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: None,
            golden_queries: Vec::new(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: RegressionConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a golden query
    pub fn add_query(mut self, query: GoldenQuery) -> Self {
        self.golden_queries.push(query);
        self
    }

    /// Add multiple golden queries
    pub fn add_queries(mut self, queries: Vec<GoldenQuery>) -> Self {
        self.golden_queries.extend(queries);
        self
    }

    /// Build the suite
    pub fn build(self) -> RegressionTestSuite {
        let mut suite = RegressionTestSuite::new(self.name, self.config.unwrap_or_default());
        for query in self.golden_queries {
            suite.add_golden_query(query);
        }
        suite
    }
}

/// Comparison result between two regression reports
#[derive(Debug, Clone)]
pub struct ReportComparison {
    /// Queries that newly regressed
    pub new_regressions: Vec<String>,
    /// Queries that were fixed (no longer regressed)
    pub fixed_regressions: Vec<String>,
    /// Queries that newly improved
    pub new_improvements: Vec<String>,
    /// Queries that degraded from improved to stable/regressed
    pub lost_improvements: Vec<String>,
    /// Overall status change
    pub status_change: Option<(RegressionStatus, RegressionStatus)>,
}

impl ReportComparison {
    /// Compare two reports
    pub fn compare(old: &RegressionReport, new: &RegressionReport) -> Self {
        let old_status: HashMap<&str, RegressionStatus> = old
            .analyses
            .iter()
            .map(|a| (a.query_id.as_str(), a.status))
            .collect();

        let new_status: HashMap<&str, RegressionStatus> = new
            .analyses
            .iter()
            .map(|a| (a.query_id.as_str(), a.status))
            .collect();

        let mut comparison = Self {
            new_regressions: Vec::new(),
            fixed_regressions: Vec::new(),
            new_improvements: Vec::new(),
            lost_improvements: Vec::new(),
            status_change: if old.overall_status != new.overall_status {
                Some((old.overall_status, new.overall_status))
            } else {
                None
            },
        };

        for (query_id, &new_stat) in &new_status {
            let old_stat = old_status.get(query_id).copied();
            match (old_stat, new_stat) {
                (Some(RegressionStatus::Stable), RegressionStatus::Regressed)
                | (Some(RegressionStatus::Improved), RegressionStatus::Regressed)
                | (None, RegressionStatus::Regressed) => {
                    comparison.new_regressions.push(query_id.to_string());
                }
                (Some(RegressionStatus::Regressed), RegressionStatus::Stable)
                | (Some(RegressionStatus::Regressed), RegressionStatus::Improved) => {
                    comparison.fixed_regressions.push(query_id.to_string());
                }
                (Some(RegressionStatus::Stable), RegressionStatus::Improved)
                | (None, RegressionStatus::Improved) => {
                    comparison.new_improvements.push(query_id.to_string());
                }
                (Some(RegressionStatus::Improved), RegressionStatus::Stable)
                | (Some(RegressionStatus::Improved), RegressionStatus::Failing) => {
                    comparison.lost_improvements.push(query_id.to_string());
                }
                _ => {}
            }
        }

        comparison
    }

    /// Check if there are any changes
    pub fn has_changes(&self) -> bool {
        !self.new_regressions.is_empty()
            || !self.fixed_regressions.is_empty()
            || !self.new_improvements.is_empty()
            || !self.lost_improvements.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_query_creation() {
        let query = GoldenQuery::new("test_query", "SELECT * WHERE { ?s ?p ?o }", 10.0)
            .with_description("Test query")
            .with_tag("basic")
            .with_expected_count(100)
            .with_priority(1);

        assert_eq!(query.id, "test_query");
        assert_eq!(query.baseline_ms, 10.0);
        assert_eq!(query.description, "Test query");
        assert_eq!(query.tags, vec!["basic"]);
        assert_eq!(query.expected_result_count, Some(100));
        assert_eq!(query.priority, 1);
    }

    #[test]
    fn test_execution_result() {
        let success = ExecutionResult::success(5.5)
            .with_result_count(50)
            .with_memory(1024)
            .with_metadata("version", "1.0");

        assert!(success.success);
        assert_eq!(success.execution_time_ms, 5.5);
        assert_eq!(success.result_count, Some(50));
        assert_eq!(success.memory_bytes, Some(1024));
        assert_eq!(success.metadata.get("version"), Some(&"1.0".to_string()));

        let failure = ExecutionResult::failure("Query timeout");
        assert!(!failure.success);
        assert_eq!(failure.error, Some("Query timeout".to_string()));
    }

    #[test]
    fn test_execution_statistics() {
        let results: Vec<ExecutionResult> = vec![
            ExecutionResult::success(10.0),
            ExecutionResult::success(12.0),
            ExecutionResult::success(11.0),
            ExecutionResult::success(9.0),
            ExecutionResult::success(13.0),
        ];

        let stats = ExecutionStatistics::from_results(&results);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.success_count, 5);
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.min_ms, 9.0);
        assert_eq!(stats.max_ms, 13.0);
        assert_eq!(stats.mean_ms, 11.0);
        assert_eq!(stats.median_ms, 11.0);
    }

    #[test]
    fn test_suite_creation() {
        let config = RegressionConfig::default();
        let mut suite = RegressionTestSuite::new("test_suite", config);

        suite.add_golden_query(GoldenQuery::new("q1", "SELECT ?s WHERE { ?s ?p ?o }", 10.0));
        suite.add_golden_query(GoldenQuery::new("q2", "SELECT ?p WHERE { ?s ?p ?o }", 15.0));

        assert_eq!(suite.golden_queries().count(), 2);
        assert!(suite.get_golden_query("q1").is_some());
        assert!(suite.get_golden_query("q3").is_none());
    }

    #[test]
    fn test_record_execution() {
        let mut suite = RegressionTestSuite::with_defaults("test");
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        assert!(suite.record_execution("q1", ExecutionResult::success(9.5)));
        assert!(suite.record_execution("q1", ExecutionResult::success(10.5)));
        assert!(!suite.record_execution("nonexistent", ExecutionResult::success(5.0)));

        assert_eq!(suite.statistics().total_executions, 2);
    }

    #[test]
    fn test_regression_detection() {
        let config = RegressionConfig {
            min_samples: 3,
            regression_threshold: 1.2,
            ..Default::default()
        };
        let mut suite = RegressionTestSuite::new("test", config);
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        // Add results that show regression (50% slower)
        for _ in 0..5 {
            suite.record_execution("q1", ExecutionResult::success(15.0));
        }

        let analysis = suite.analyze_query("q1").unwrap();
        assert_eq!(analysis.status, RegressionStatus::Regressed);
        assert!(analysis.ratio > 1.2);
    }

    #[test]
    fn test_improvement_detection() {
        let config = RegressionConfig {
            min_samples: 3,
            improvement_threshold: 0.8,
            ..Default::default()
        };
        let mut suite = RegressionTestSuite::new("test", config);
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        // Add results that show improvement (50% faster)
        for _ in 0..5 {
            suite.record_execution("q1", ExecutionResult::success(5.0));
        }

        let analysis = suite.analyze_query("q1").unwrap();
        assert_eq!(analysis.status, RegressionStatus::Improved);
        assert!(analysis.ratio < 0.8);
    }

    #[test]
    fn test_insufficient_data() {
        let config = RegressionConfig {
            min_samples: 10,
            ..Default::default()
        };
        let mut suite = RegressionTestSuite::new("test", config);
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        // Only add 3 samples (less than min_samples)
        for _ in 0..3 {
            suite.record_execution("q1", ExecutionResult::success(10.0));
        }

        let analysis = suite.analyze_query("q1").unwrap();
        assert_eq!(analysis.status, RegressionStatus::InsufficientData);
    }

    #[test]
    fn test_failing_query_detection() {
        let config = RegressionConfig {
            min_samples: 3,
            ..Default::default()
        };
        let mut suite = RegressionTestSuite::new("test", config);
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        // Add mostly failures
        for _ in 0..4 {
            suite.record_execution("q1", ExecutionResult::failure("Timeout"));
        }
        suite.record_execution("q1", ExecutionResult::success(10.0));

        let analysis = suite.analyze_query("q1").unwrap();
        assert_eq!(analysis.status, RegressionStatus::Failing);
    }

    #[test]
    fn test_regression_report() {
        let config = RegressionConfig {
            min_samples: 3,
            ..Default::default()
        };
        let mut suite = RegressionTestSuite::new("test", config);

        suite.add_golden_query(GoldenQuery::new(
            "stable",
            "SELECT ?s WHERE { ?s ?p ?o }",
            10.0,
        ));
        suite.add_golden_query(GoldenQuery::new(
            "regressed",
            "SELECT ?p WHERE { ?s ?p ?o }",
            10.0,
        ));
        suite.add_golden_query(GoldenQuery::new(
            "improved",
            "SELECT ?o WHERE { ?s ?p ?o }",
            10.0,
        ));

        // Stable query
        for _ in 0..5 {
            suite.record_execution("stable", ExecutionResult::success(10.0));
        }

        // Regressed query
        for _ in 0..5 {
            suite.record_execution("regressed", ExecutionResult::success(15.0));
        }

        // Improved query
        for _ in 0..5 {
            suite.record_execution("improved", ExecutionResult::success(5.0));
        }

        let report = suite.analyze_regressions();
        assert_eq!(report.summary.total_count, 3);
        assert!(report.summary.regressed_count >= 1);
        assert!(report.summary.improved_count >= 1);
    }

    #[test]
    fn test_update_baseline() {
        let config = RegressionConfig {
            min_samples: 3,
            rolling_window_size: 5,
            ..Default::default()
        };
        let mut suite = RegressionTestSuite::new("test", config);
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        // Add consistent results around 8ms
        for _ in 0..5 {
            suite.record_execution("q1", ExecutionResult::success(8.0));
        }

        let new_baseline = suite.update_baseline("q1").unwrap();
        assert!((new_baseline - 8.0).abs() < 0.1);
        assert_eq!(
            suite.get_golden_query("q1").unwrap().baseline_ms,
            new_baseline
        );
    }

    #[test]
    fn test_report_comparison() {
        let config = RegressionConfig {
            min_samples: 3,
            ..Default::default()
        };

        // Create first suite and report
        let mut suite1 = RegressionTestSuite::new("test", config.clone());
        suite1.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));
        suite1.add_golden_query(GoldenQuery::new("q2", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        for _ in 0..5 {
            suite1.record_execution("q1", ExecutionResult::success(10.0)); // Stable
            suite1.record_execution("q2", ExecutionResult::success(10.0)); // Stable
        }
        let report1 = suite1.analyze_regressions();

        // Create second suite with changes
        let mut suite2 = RegressionTestSuite::new("test", config);
        suite2.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));
        suite2.add_golden_query(GoldenQuery::new("q2", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        for _ in 0..5 {
            suite2.record_execution("q1", ExecutionResult::success(15.0)); // Regressed
            suite2.record_execution("q2", ExecutionResult::success(5.0)); // Improved
        }
        let report2 = suite2.analyze_regressions();

        let comparison = ReportComparison::compare(&report1, &report2);
        assert!(comparison.has_changes());
        assert!(comparison.new_regressions.contains(&"q1".to_string()));
        assert!(comparison.new_improvements.contains(&"q2".to_string()));
    }

    #[test]
    fn test_suite_builder() {
        let suite = RegressionTestSuiteBuilder::new("builder_test")
            .with_config(RegressionConfig::strict())
            .add_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0))
            .add_query(GoldenQuery::new("q2", "SELECT * WHERE { ?s ?p ?o }", 20.0))
            .build();

        assert_eq!(suite.golden_queries().count(), 2);
        assert_eq!(suite.config().regression_threshold, 1.1); // Strict config
    }

    #[test]
    fn test_suite_export_import() {
        let mut suite = RegressionTestSuite::with_defaults("export_test");
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));
        suite.record_execution("q1", ExecutionResult::success(9.0));

        let export = suite.export();
        let imported = RegressionTestSuite::import(export);

        assert_eq!(imported.golden_queries().count(), 1);
        assert!(imported.get_golden_query("q1").is_some());
    }

    #[test]
    fn test_regression_status_display() {
        assert_eq!(format!("{}", RegressionStatus::Stable), "STABLE");
        assert_eq!(format!("{}", RegressionStatus::Regressed), "REGRESSED");
        assert_eq!(format!("{}", RegressionStatus::Improved), "IMPROVED");
        assert_eq!(format!("{}", RegressionStatus::Failing), "FAILING");
        assert_eq!(
            format!("{}", RegressionStatus::InsufficientData),
            "INSUFFICIENT_DATA"
        );
    }

    #[test]
    fn test_report_summary_text() {
        let config = RegressionConfig {
            min_samples: 3,
            ..Default::default()
        };
        let mut suite = RegressionTestSuite::new("summary_test", config);
        suite.add_golden_query(GoldenQuery::new("q1", "SELECT * WHERE { ?s ?p ?o }", 10.0));

        for _ in 0..5 {
            suite.record_execution("q1", ExecutionResult::success(15.0));
        }

        let report = suite.analyze_regressions();
        let summary = report.summary_text();

        assert!(summary.contains("summary_test"));
        assert!(summary.contains("Total:"));
    }

    #[test]
    fn test_config_presets() {
        let strict = RegressionConfig::strict();
        assert_eq!(strict.regression_threshold, 1.1);
        assert_eq!(strict.confidence_level, 0.99);

        let lenient = RegressionConfig::lenient();
        assert_eq!(lenient.regression_threshold, 1.5);
        assert_eq!(lenient.confidence_level, 0.90);
    }
}
