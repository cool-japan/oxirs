//! # Query Execution History
//!
//! This module provides comprehensive tracking and analysis of SPARQL query execution history.
//! It supports workload analysis, performance trending, and historical query inspection.
//!
//! ## Features
//!
//! - **Execution Recording**: Capture detailed execution metrics for each query
//! - **Historical Analysis**: Analyze patterns and trends over time
//! - **Query Classification**: Categorize queries by type, complexity, and performance
//! - **Workload Profiling**: Understand query workload characteristics
//! - **Slow Query Detection**: Identify and track slow queries
//! - **Resource Usage Tracking**: Monitor memory and CPU usage patterns
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use oxirs_arq::query_execution_history::{
//!     QueryExecutionHistory, ExecutionRecord, HistoryConfig,
//! };
//!
//! // Create history tracker
//! let config = HistoryConfig::default();
//! let mut history = QueryExecutionHistory::new(config);
//!
//! // Record executions
//! history.record(ExecutionRecord::new(
//!     "SELECT ?s WHERE { ?s ?p ?o }",
//!     10.5, // execution time ms
//!     100,  // result count
//! ));
//!
//! // Analyze history
//! let analysis = history.analyze();
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, SystemTime};

/// Configuration for query execution history
#[derive(Debug, Clone)]
pub struct HistoryConfig {
    /// Maximum number of records to retain
    pub max_records: usize,
    /// Slow query threshold in milliseconds
    pub slow_query_threshold_ms: f64,
    /// Enable query text storage (can be disabled for privacy)
    pub store_query_text: bool,
    /// Maximum query text length to store
    pub max_query_text_length: usize,
    /// Enable detailed metrics collection
    pub detailed_metrics: bool,
    /// Time window for analysis in seconds
    pub analysis_window_secs: u64,
    /// Number of top queries to track
    pub top_queries_count: usize,
    /// Enable automatic cleanup of old records
    pub auto_cleanup: bool,
    /// Retention period in seconds
    pub retention_period_secs: u64,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            max_records: 10000,
            slow_query_threshold_ms: 1000.0,
            store_query_text: true,
            max_query_text_length: 1000,
            detailed_metrics: true,
            analysis_window_secs: 3600, // 1 hour
            top_queries_count: 20,
            auto_cleanup: true,
            retention_period_secs: 86400, // 24 hours
        }
    }
}

impl HistoryConfig {
    /// Create a minimal configuration for low memory usage
    pub fn minimal() -> Self {
        Self {
            max_records: 1000,
            slow_query_threshold_ms: 500.0,
            store_query_text: false,
            max_query_text_length: 200,
            detailed_metrics: false,
            analysis_window_secs: 1800, // 30 minutes
            top_queries_count: 10,
            auto_cleanup: true,
            retention_period_secs: 3600, // 1 hour
        }
    }

    /// Create a comprehensive configuration for detailed analysis
    pub fn comprehensive() -> Self {
        Self {
            max_records: 100000,
            slow_query_threshold_ms: 2000.0,
            store_query_text: true,
            max_query_text_length: 5000,
            detailed_metrics: true,
            analysis_window_secs: 86400, // 24 hours
            top_queries_count: 50,
            auto_cleanup: true,
            retention_period_secs: 604800, // 7 days
        }
    }
}

/// Query execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionStatus {
    /// Query completed successfully
    Success,
    /// Query failed with an error
    Failed,
    /// Query was cancelled
    Cancelled,
    /// Query timed out
    Timeout,
}

impl fmt::Display for ExecutionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "SUCCESS"),
            Self::Failed => write!(f, "FAILED"),
            Self::Cancelled => write!(f, "CANCELLED"),
            Self::Timeout => write!(f, "TIMEOUT"),
        }
    }
}

/// Query form type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryFormType {
    Select,
    Ask,
    Construct,
    Describe,
    Update,
    Unknown,
}

impl fmt::Display for QueryFormType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Select => write!(f, "SELECT"),
            Self::Ask => write!(f, "ASK"),
            Self::Construct => write!(f, "CONSTRUCT"),
            Self::Describe => write!(f, "DESCRIBE"),
            Self::Update => write!(f, "UPDATE"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

impl QueryFormType {
    /// Detect query form from query text
    pub fn detect(query: &str) -> Self {
        let query_upper = query.to_uppercase();
        let query_trimmed = query_upper.trim();

        if query_trimmed.starts_with("SELECT") {
            Self::Select
        } else if query_trimmed.starts_with("ASK") {
            Self::Ask
        } else if query_trimmed.starts_with("CONSTRUCT") {
            Self::Construct
        } else if query_trimmed.starts_with("DESCRIBE") {
            Self::Describe
        } else if query_trimmed.starts_with("INSERT")
            || query_trimmed.starts_with("DELETE")
            || query_trimmed.starts_with("LOAD")
            || query_trimmed.starts_with("CLEAR")
            || query_trimmed.starts_with("CREATE")
            || query_trimmed.starts_with("DROP")
        {
            Self::Update
        } else {
            Self::Unknown
        }
    }
}

/// Detailed execution metrics
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Planning time in milliseconds
    pub planning_time_ms: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Result serialization time in milliseconds
    pub serialization_time_ms: f64,
    /// Number of triples scanned
    pub triples_scanned: usize,
    /// Number of joins performed
    pub joins_performed: usize,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of index lookups
    pub index_lookups: usize,
}

impl ExecutionMetrics {
    /// Get total time
    pub fn total_time_ms(&self) -> f64 {
        self.planning_time_ms + self.execution_time_ms + self.serialization_time_ms
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// A single execution record
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Unique identifier for this execution
    pub id: u64,
    /// Query text (may be truncated or empty based on config)
    pub query_text: Option<String>,
    /// Query fingerprint for grouping similar queries
    pub query_fingerprint: String,
    /// Query form type
    pub query_form: QueryFormType,
    /// Execution status
    pub status: ExecutionStatus,
    /// Total execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of results returned
    pub result_count: usize,
    /// Timestamp of execution start
    pub started_at: SystemTime,
    /// Timestamp of execution end
    pub ended_at: SystemTime,
    /// Detailed metrics (optional)
    pub metrics: Option<ExecutionMetrics>,
    /// User or client identifier
    pub user_id: Option<String>,
    /// Source endpoint or client IP
    pub source: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional tags
    pub tags: Vec<String>,
}

impl ExecutionRecord {
    /// Create a new successful execution record
    pub fn new(query: impl Into<String>, execution_time_ms: f64, result_count: usize) -> Self {
        let query_str = query.into();
        let query_form = QueryFormType::detect(&query_str);
        let fingerprint = Self::compute_fingerprint(&query_str);
        let now = SystemTime::now();

        Self {
            id: Self::generate_id(),
            query_text: Some(query_str),
            query_fingerprint: fingerprint,
            query_form,
            status: ExecutionStatus::Success,
            execution_time_ms,
            result_count,
            started_at: now,
            ended_at: now,
            metrics: None,
            user_id: None,
            source: None,
            error: None,
            tags: Vec::new(),
        }
    }

    /// Create a failed execution record
    pub fn failed(query: impl Into<String>, error: impl Into<String>) -> Self {
        let query_str = query.into();
        let query_form = QueryFormType::detect(&query_str);
        let fingerprint = Self::compute_fingerprint(&query_str);
        let now = SystemTime::now();

        Self {
            id: Self::generate_id(),
            query_text: Some(query_str),
            query_fingerprint: fingerprint,
            query_form,
            status: ExecutionStatus::Failed,
            execution_time_ms: 0.0,
            result_count: 0,
            started_at: now,
            ended_at: now,
            metrics: None,
            user_id: None,
            source: None,
            error: Some(error.into()),
            tags: Vec::new(),
        }
    }

    /// Set detailed metrics
    pub fn with_metrics(mut self, metrics: ExecutionMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Set user ID
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Check if this is a slow query
    pub fn is_slow(&self, threshold_ms: f64) -> bool {
        self.execution_time_ms >= threshold_ms
    }

    fn generate_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    fn compute_fingerprint(query: &str) -> String {
        // Simple fingerprint based on query structure
        let normalized = query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        // Remove literal values for fingerprinting
        let fingerprint = normalized
            .chars()
            .filter(|c| {
                c.is_alphabetic() || c.is_whitespace() || *c == '?' || *c == '{' || *c == '}'
            })
            .collect::<String>();

        format!("{:x}", md5::compute(fingerprint.as_bytes()))
    }
}

/// Time-based statistics for a period
#[derive(Debug, Clone, Default)]
pub struct PeriodStatistics {
    /// Period start time
    pub start_time: Option<SystemTime>,
    /// Period end time
    pub end_time: Option<SystemTime>,
    /// Total queries in period
    pub total_queries: usize,
    /// Successful queries
    pub successful_queries: usize,
    /// Failed queries
    pub failed_queries: usize,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: f64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Minimum execution time in milliseconds
    pub min_execution_time_ms: f64,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: f64,
    /// 95th percentile execution time
    pub p95_execution_time_ms: f64,
    /// Total results returned
    pub total_results: usize,
    /// Queries per second
    pub queries_per_second: f64,
}

/// Query form distribution
#[derive(Debug, Clone, Default)]
pub struct FormDistribution {
    pub select_count: usize,
    pub ask_count: usize,
    pub construct_count: usize,
    pub describe_count: usize,
    pub update_count: usize,
    pub unknown_count: usize,
}

impl FormDistribution {
    fn add(&mut self, form: QueryFormType) {
        match form {
            QueryFormType::Select => self.select_count += 1,
            QueryFormType::Ask => self.ask_count += 1,
            QueryFormType::Construct => self.construct_count += 1,
            QueryFormType::Describe => self.describe_count += 1,
            QueryFormType::Update => self.update_count += 1,
            QueryFormType::Unknown => self.unknown_count += 1,
        }
    }

    fn total(&self) -> usize {
        self.select_count
            + self.ask_count
            + self.construct_count
            + self.describe_count
            + self.update_count
            + self.unknown_count
    }
}

/// Grouped query statistics
#[derive(Debug, Clone)]
pub struct QueryGroupStats {
    /// Query fingerprint
    pub fingerprint: String,
    /// Sample query text
    pub sample_query: Option<String>,
    /// Query form type
    pub query_form: QueryFormType,
    /// Total executions
    pub execution_count: usize,
    /// Total execution time
    pub total_time_ms: f64,
    /// Average execution time
    pub avg_time_ms: f64,
    /// Minimum execution time
    pub min_time_ms: f64,
    /// Maximum execution time
    pub max_time_ms: f64,
    /// Success rate
    pub success_rate: f64,
    /// Average result count
    pub avg_result_count: f64,
    /// Last executed
    pub last_executed: SystemTime,
}

/// Slow query entry
#[derive(Debug, Clone)]
pub struct SlowQueryEntry {
    /// The execution record
    pub record: ExecutionRecord,
    /// Rank among slow queries
    pub rank: usize,
    /// How much slower than average
    pub slowness_factor: f64,
}

/// Historical analysis result
#[derive(Debug, Clone)]
pub struct HistoryAnalysis {
    /// Analysis timestamp
    pub generated_at: SystemTime,
    /// Overall statistics
    pub overall_stats: PeriodStatistics,
    /// Query form distribution
    pub form_distribution: FormDistribution,
    /// Top queries by frequency
    pub top_by_frequency: Vec<QueryGroupStats>,
    /// Top queries by total time
    pub top_by_total_time: Vec<QueryGroupStats>,
    /// Slow queries
    pub slow_queries: Vec<SlowQueryEntry>,
    /// Hourly breakdown (last 24 hours)
    pub hourly_stats: Vec<PeriodStatistics>,
    /// Error rate
    pub error_rate: f64,
    /// Unique query fingerprints
    pub unique_queries: usize,
    /// User distribution
    pub user_stats: HashMap<String, usize>,
}

impl HistoryAnalysis {
    /// Get a text summary
    pub fn summary_text(&self) -> String {
        let mut text = String::from("Query Execution History Analysis\n");
        text.push_str(&format!("Generated: {:?}\n\n", self.generated_at));

        text.push_str("Overall Statistics:\n");
        text.push_str(&format!(
            "  Total Queries: {}\n",
            self.overall_stats.total_queries
        ));
        text.push_str(&format!(
            "  Success Rate: {:.2}%\n",
            (1.0 - self.error_rate) * 100.0
        ));
        text.push_str(&format!(
            "  Avg Execution Time: {:.2}ms\n",
            self.overall_stats.avg_execution_time_ms
        ));
        text.push_str(&format!(
            "  P95 Execution Time: {:.2}ms\n",
            self.overall_stats.p95_execution_time_ms
        ));
        text.push_str(&format!("  Unique Query Types: {}\n", self.unique_queries));

        text.push_str("\nQuery Form Distribution:\n");
        let total = self.form_distribution.total();
        if total > 0 {
            text.push_str(&format!(
                "  SELECT: {} ({:.1}%)\n",
                self.form_distribution.select_count,
                self.form_distribution.select_count as f64 / total as f64 * 100.0
            ));
            text.push_str(&format!(
                "  ASK: {} ({:.1}%)\n",
                self.form_distribution.ask_count,
                self.form_distribution.ask_count as f64 / total as f64 * 100.0
            ));
            text.push_str(&format!(
                "  CONSTRUCT: {} ({:.1}%)\n",
                self.form_distribution.construct_count,
                self.form_distribution.construct_count as f64 / total as f64 * 100.0
            ));
        }

        if !self.slow_queries.is_empty() {
            text.push_str(&format!("\nSlow Queries: {}\n", self.slow_queries.len()));
        }

        text
    }
}

/// Main query execution history tracker
#[derive(Debug)]
pub struct QueryExecutionHistory {
    /// Configuration
    config: HistoryConfig,
    /// Execution records (newest first)
    records: VecDeque<ExecutionRecord>,
    /// Query group statistics
    groups: HashMap<String, QueryGroupStats>,
    /// Statistics
    stats: HistoryStatistics,
}

/// History tracker statistics
#[derive(Debug, Clone, Default)]
pub struct HistoryStatistics {
    /// Total records ever recorded
    pub total_recorded: usize,
    /// Total records evicted
    pub total_evicted: usize,
    /// Total analyses performed
    pub total_analyses: usize,
    /// Last cleanup time
    pub last_cleanup: Option<SystemTime>,
}

impl QueryExecutionHistory {
    /// Create a new history tracker
    pub fn new(config: HistoryConfig) -> Self {
        Self {
            config,
            records: VecDeque::new(),
            groups: HashMap::new(),
            stats: HistoryStatistics::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(HistoryConfig::default())
    }

    /// Record an execution
    pub fn record(&mut self, mut record: ExecutionRecord) {
        // Truncate query text if needed
        if let Some(ref mut text) = record.query_text {
            if !self.config.store_query_text {
                record.query_text = None;
            } else if text.len() > self.config.max_query_text_length {
                text.truncate(self.config.max_query_text_length);
                text.push_str("...");
            }
        }

        // Update group statistics
        self.update_group_stats(&record);

        // Add to records
        self.records.push_front(record);
        self.stats.total_recorded += 1;

        // Enforce max records
        while self.records.len() > self.config.max_records {
            self.records.pop_back();
            self.stats.total_evicted += 1;
        }

        // Auto cleanup if enabled
        if self.config.auto_cleanup {
            self.cleanup_old_records();
        }
    }

    /// Record multiple executions
    pub fn record_batch(&mut self, records: Vec<ExecutionRecord>) {
        for record in records {
            self.record(record);
        }
    }

    /// Get recent records
    pub fn recent(&self, count: usize) -> Vec<&ExecutionRecord> {
        self.records.iter().take(count).collect()
    }

    /// Get slow queries
    pub fn slow_queries(&self, limit: usize) -> Vec<SlowQueryEntry> {
        let avg_time = self.calculate_avg_time();
        let threshold = self.config.slow_query_threshold_ms;

        let mut slow: Vec<_> = self
            .records
            .iter()
            .filter(|r| r.is_slow(threshold))
            .collect();

        slow.sort_by(|a, b| {
            b.execution_time_ms
                .partial_cmp(&a.execution_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        slow.iter()
            .take(limit)
            .enumerate()
            .map(|(i, r)| SlowQueryEntry {
                record: (*r).clone(),
                rank: i + 1,
                slowness_factor: if avg_time > 0.0 {
                    r.execution_time_ms / avg_time
                } else {
                    1.0
                },
            })
            .collect()
    }

    /// Get records by fingerprint
    pub fn by_fingerprint(&self, fingerprint: &str) -> Vec<&ExecutionRecord> {
        self.records
            .iter()
            .filter(|r| r.query_fingerprint == fingerprint)
            .collect()
    }

    /// Get records by user
    pub fn by_user(&self, user_id: &str) -> Vec<&ExecutionRecord> {
        self.records
            .iter()
            .filter(|r| r.user_id.as_deref() == Some(user_id))
            .collect()
    }

    /// Get records by status
    pub fn by_status(&self, status: ExecutionStatus) -> Vec<&ExecutionRecord> {
        self.records.iter().filter(|r| r.status == status).collect()
    }

    /// Get records within time range
    pub fn in_time_range(&self, start: SystemTime, end: SystemTime) -> Vec<&ExecutionRecord> {
        self.records
            .iter()
            .filter(|r| r.started_at >= start && r.started_at <= end)
            .collect()
    }

    /// Perform full analysis
    pub fn analyze(&mut self) -> HistoryAnalysis {
        self.stats.total_analyses += 1;

        let overall_stats = self.calculate_period_stats(&self.records.iter().collect::<Vec<_>>());
        let form_distribution = self.calculate_form_distribution();
        let top_by_frequency = self.top_by_frequency();
        let top_by_total_time = self.top_by_total_time();
        let slow_queries = self.slow_queries(self.config.top_queries_count);
        let hourly_stats = self.calculate_hourly_stats();
        let user_stats = self.calculate_user_stats();

        let error_count = self
            .records
            .iter()
            .filter(|r| r.status != ExecutionStatus::Success)
            .count();
        let error_rate = if self.records.is_empty() {
            0.0
        } else {
            error_count as f64 / self.records.len() as f64
        };

        HistoryAnalysis {
            generated_at: SystemTime::now(),
            overall_stats,
            form_distribution,
            top_by_frequency,
            top_by_total_time,
            slow_queries,
            hourly_stats,
            error_rate,
            unique_queries: self.groups.len(),
            user_stats,
        }
    }

    /// Get group statistics
    pub fn group_stats(&self) -> &HashMap<String, QueryGroupStats> {
        &self.groups
    }

    /// Get history statistics
    pub fn statistics(&self) -> &HistoryStatistics {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &HistoryConfig {
        &self.config
    }

    /// Clear all records
    pub fn clear(&mut self) {
        self.records.clear();
        self.groups.clear();
    }

    /// Get record count
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    // Private methods

    fn update_group_stats(&mut self, record: &ExecutionRecord) {
        let fingerprint = &record.query_fingerprint;

        if let Some(group) = self.groups.get_mut(fingerprint) {
            let n = group.execution_count as f64;
            group.execution_count += 1;
            group.total_time_ms += record.execution_time_ms;
            group.avg_time_ms = group.total_time_ms / group.execution_count as f64;
            group.min_time_ms = group.min_time_ms.min(record.execution_time_ms);
            group.max_time_ms = group.max_time_ms.max(record.execution_time_ms);

            let success_count = if record.status == ExecutionStatus::Success {
                (group.success_rate * n) + 1.0
            } else {
                group.success_rate * n
            };
            group.success_rate = success_count / (n + 1.0);

            group.avg_result_count =
                (group.avg_result_count * n + record.result_count as f64) / (n + 1.0);
            group.last_executed = record.started_at;
        } else {
            self.groups.insert(
                fingerprint.clone(),
                QueryGroupStats {
                    fingerprint: fingerprint.clone(),
                    sample_query: record.query_text.clone(),
                    query_form: record.query_form,
                    execution_count: 1,
                    total_time_ms: record.execution_time_ms,
                    avg_time_ms: record.execution_time_ms,
                    min_time_ms: record.execution_time_ms,
                    max_time_ms: record.execution_time_ms,
                    success_rate: if record.status == ExecutionStatus::Success {
                        1.0
                    } else {
                        0.0
                    },
                    avg_result_count: record.result_count as f64,
                    last_executed: record.started_at,
                },
            );
        }
    }

    fn calculate_avg_time(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let total: f64 = self.records.iter().map(|r| r.execution_time_ms).sum();
        total / self.records.len() as f64
    }

    fn calculate_period_stats(&self, records: &[&ExecutionRecord]) -> PeriodStatistics {
        if records.is_empty() {
            return PeriodStatistics::default();
        }

        let mut stats = PeriodStatistics {
            start_time: records.last().map(|r| r.started_at),
            end_time: records.first().map(|r| r.started_at),
            total_queries: records.len(),
            ..Default::default()
        };

        let mut times: Vec<f64> = Vec::new();

        for record in records {
            if record.status == ExecutionStatus::Success {
                stats.successful_queries += 1;
            } else {
                stats.failed_queries += 1;
            }
            stats.total_execution_time_ms += record.execution_time_ms;
            stats.total_results += record.result_count;
            times.push(record.execution_time_ms);
        }

        if !times.is_empty() {
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            stats.avg_execution_time_ms = stats.total_execution_time_ms / times.len() as f64;
            stats.min_execution_time_ms = times[0];
            stats.max_execution_time_ms = times[times.len() - 1];
            let p95_idx = ((times.len() as f64 * 0.95) as usize).min(times.len() - 1);
            stats.p95_execution_time_ms = times[p95_idx];
        }

        // Calculate QPS
        if let (Some(start), Some(end)) = (stats.start_time, stats.end_time) {
            if let Ok(duration) = end.duration_since(start) {
                let secs = duration.as_secs_f64();
                if secs > 0.0 {
                    stats.queries_per_second = stats.total_queries as f64 / secs;
                }
            }
        }

        stats
    }

    fn calculate_form_distribution(&self) -> FormDistribution {
        let mut dist = FormDistribution::default();
        for record in &self.records {
            dist.add(record.query_form);
        }
        dist
    }

    fn top_by_frequency(&self) -> Vec<QueryGroupStats> {
        let mut groups: Vec<_> = self.groups.values().cloned().collect();
        groups.sort_by(|a, b| b.execution_count.cmp(&a.execution_count));
        groups.truncate(self.config.top_queries_count);
        groups
    }

    fn top_by_total_time(&self) -> Vec<QueryGroupStats> {
        let mut groups: Vec<_> = self.groups.values().cloned().collect();
        groups.sort_by(|a, b| {
            b.total_time_ms
                .partial_cmp(&a.total_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        groups.truncate(self.config.top_queries_count);
        groups
    }

    fn calculate_hourly_stats(&self) -> Vec<PeriodStatistics> {
        let now = SystemTime::now();
        let mut hourly_stats = Vec::new();

        for hour in 0..24 {
            let hour_start = now - Duration::from_secs((hour + 1) * 3600);
            let hour_end = now - Duration::from_secs(hour * 3600);

            let records: Vec<_> = self
                .records
                .iter()
                .filter(|r| r.started_at >= hour_start && r.started_at < hour_end)
                .collect();

            hourly_stats.push(self.calculate_period_stats(&records));
        }

        hourly_stats
    }

    fn calculate_user_stats(&self) -> HashMap<String, usize> {
        let mut user_stats = HashMap::new();
        for record in &self.records {
            if let Some(ref user_id) = record.user_id {
                *user_stats.entry(user_id.clone()).or_insert(0) += 1;
            }
        }
        user_stats
    }

    fn cleanup_old_records(&mut self) {
        let now = SystemTime::now();
        let cutoff = now - Duration::from_secs(self.config.retention_period_secs);

        let original_len = self.records.len();
        self.records.retain(|r| r.started_at >= cutoff);
        self.stats.total_evicted += original_len - self.records.len();
        self.stats.last_cleanup = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_record_creation() {
        let record = ExecutionRecord::new("SELECT ?s WHERE { ?s ?p ?o }", 10.5, 100);

        assert!(record.query_text.is_some());
        assert_eq!(record.execution_time_ms, 10.5);
        assert_eq!(record.result_count, 100);
        assert_eq!(record.status, ExecutionStatus::Success);
        assert_eq!(record.query_form, QueryFormType::Select);
    }

    #[test]
    fn test_failed_record() {
        let record = ExecutionRecord::failed("SELECT * WHERE { ?s ?p ?o }", "Syntax error");

        assert_eq!(record.status, ExecutionStatus::Failed);
        assert_eq!(record.error, Some("Syntax error".to_string()));
    }

    #[test]
    fn test_query_form_detection() {
        assert_eq!(
            QueryFormType::detect("SELECT ?s WHERE { ?s ?p ?o }"),
            QueryFormType::Select
        );
        assert_eq!(
            QueryFormType::detect("ASK WHERE { ?s ?p ?o }"),
            QueryFormType::Ask
        );
        assert_eq!(
            QueryFormType::detect("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            QueryFormType::Construct
        );
        assert_eq!(
            QueryFormType::detect("DESCRIBE <http://example.org>"),
            QueryFormType::Describe
        );
        assert_eq!(
            QueryFormType::detect("INSERT DATA { <s> <p> <o> }"),
            QueryFormType::Update
        );
    }

    #[test]
    fn test_history_record() {
        let mut history = QueryExecutionHistory::with_defaults();

        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            10.0,
            50,
        ));
        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            15.0,
            60,
        ));

        assert_eq!(history.len(), 2);
        assert_eq!(history.stats.total_recorded, 2);
    }

    #[test]
    fn test_slow_query_detection() {
        let config = HistoryConfig {
            slow_query_threshold_ms: 100.0,
            ..Default::default()
        };
        let mut history = QueryExecutionHistory::new(config);

        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            50.0,
            10,
        ));
        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            150.0,
            20,
        ));
        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            200.0,
            30,
        ));

        let slow = history.slow_queries(10);
        assert_eq!(slow.len(), 2);
        assert_eq!(slow[0].rank, 1);
        assert!(slow[0].record.execution_time_ms >= slow[1].record.execution_time_ms);
    }

    #[test]
    fn test_group_statistics() {
        let mut history = QueryExecutionHistory::with_defaults();

        // Same query fingerprint
        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            10.0,
            50,
        ));
        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            20.0,
            60,
        ));
        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            30.0,
            70,
        ));

        let groups = history.group_stats();
        assert_eq!(groups.len(), 1);

        let group = groups.values().next().unwrap();
        assert_eq!(group.execution_count, 3);
        assert_eq!(group.avg_time_ms, 20.0);
        assert_eq!(group.min_time_ms, 10.0);
        assert_eq!(group.max_time_ms, 30.0);
    }

    #[test]
    fn test_analysis() {
        let mut history = QueryExecutionHistory::with_defaults();

        for i in 0..10 {
            history.record(ExecutionRecord::new(
                "SELECT ?s WHERE { ?s ?p ?o }",
                (i as f64) * 10.0,
                i * 10,
            ));
        }

        let analysis = history.analyze();
        assert_eq!(analysis.overall_stats.total_queries, 10);
        assert!(analysis.overall_stats.avg_execution_time_ms > 0.0);
    }

    #[test]
    fn test_by_status() {
        let mut history = QueryExecutionHistory::with_defaults();

        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            10.0,
            50,
        ));
        history.record(ExecutionRecord::failed(
            "SELECT ?s WHERE { ?s ?p ?o }",
            "Error",
        ));

        let successes = history.by_status(ExecutionStatus::Success);
        let failures = history.by_status(ExecutionStatus::Failed);

        assert_eq!(successes.len(), 1);
        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn test_by_user() {
        let mut history = QueryExecutionHistory::with_defaults();

        history.record(
            ExecutionRecord::new("SELECT ?s WHERE { ?s ?p ?o }", 10.0, 50).with_user("alice"),
        );
        history.record(
            ExecutionRecord::new("SELECT ?s WHERE { ?s ?p ?o }", 15.0, 60).with_user("bob"),
        );
        history.record(
            ExecutionRecord::new("SELECT ?s WHERE { ?s ?p ?o }", 20.0, 70).with_user("alice"),
        );

        let alice_queries = history.by_user("alice");
        assert_eq!(alice_queries.len(), 2);
    }

    #[test]
    fn test_max_records_enforcement() {
        let config = HistoryConfig {
            max_records: 5,
            ..Default::default()
        };
        let mut history = QueryExecutionHistory::new(config);

        for i in 0..10 {
            history.record(ExecutionRecord::new(
                format!("SELECT ?s{} WHERE {{ ?s ?p ?o }}", i),
                10.0,
                50,
            ));
        }

        assert_eq!(history.len(), 5);
        assert_eq!(history.stats.total_evicted, 5);
    }

    #[test]
    fn test_form_distribution() {
        let mut history = QueryExecutionHistory::with_defaults();

        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            10.0,
            50,
        ));
        history.record(ExecutionRecord::new("ASK WHERE { ?s ?p ?o }", 5.0, 1));
        history.record(ExecutionRecord::new(
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
            15.0,
            30,
        ));

        let analysis = history.analyze();
        assert_eq!(analysis.form_distribution.select_count, 1);
        assert_eq!(analysis.form_distribution.ask_count, 1);
        assert_eq!(analysis.form_distribution.construct_count, 1);
    }

    #[test]
    fn test_execution_metrics() {
        let metrics = ExecutionMetrics {
            planning_time_ms: 5.0,
            execution_time_ms: 50.0,
            serialization_time_ms: 10.0,
            cache_hits: 80,
            cache_misses: 20,
            ..Default::default()
        };

        assert_eq!(metrics.total_time_ms(), 65.0);
        assert_eq!(metrics.cache_hit_ratio(), 0.8);
    }

    #[test]
    fn test_config_presets() {
        let minimal = HistoryConfig::minimal();
        assert_eq!(minimal.max_records, 1000);
        assert!(!minimal.store_query_text);

        let comprehensive = HistoryConfig::comprehensive();
        assert_eq!(comprehensive.max_records, 100000);
        assert!(comprehensive.store_query_text);
    }

    #[test]
    fn test_clear_history() {
        let mut history = QueryExecutionHistory::with_defaults();

        history.record(ExecutionRecord::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            10.0,
            50,
        ));
        assert!(!history.is_empty());

        history.clear();
        assert!(history.is_empty());
    }

    #[test]
    fn test_execution_status_display() {
        assert_eq!(format!("{}", ExecutionStatus::Success), "SUCCESS");
        assert_eq!(format!("{}", ExecutionStatus::Failed), "FAILED");
        assert_eq!(format!("{}", ExecutionStatus::Cancelled), "CANCELLED");
        assert_eq!(format!("{}", ExecutionStatus::Timeout), "TIMEOUT");
    }

    #[test]
    fn test_record_with_tags() {
        let record = ExecutionRecord::new("SELECT ?s WHERE { ?s ?p ?o }", 10.0, 50)
            .with_tag("batch")
            .with_tag("priority");

        assert_eq!(record.tags.len(), 2);
        assert!(record.tags.contains(&"batch".to_string()));
    }
}
