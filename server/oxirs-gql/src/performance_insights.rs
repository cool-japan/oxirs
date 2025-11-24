//! Query Performance Insights Dashboard
//!
//! This module provides comprehensive performance analytics and insights
//! for GraphQL queries, enabling optimization and monitoring.
//!
//! ## Features
//!
//! - **Query Profiling**: Detailed timing breakdowns for each query
//! - **Trend Analysis**: Historical performance trends and patterns
//! - **Anomaly Detection**: Identify performance regressions
//! - **Recommendations**: Automated optimization suggestions
//! - **Interactive Dashboard**: Real-time performance visualization
//! - **Alerting**: Performance threshold notifications

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Performance metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Query execution time
    ExecutionTime,
    /// Parse time
    ParseTime,
    /// Validation time
    ValidationTime,
    /// Resolution time
    ResolutionTime,
    /// Serialization time
    SerializationTime,
    /// Database query time
    DatabaseTime,
    /// Cache hit/miss
    CacheMetric,
    /// Memory usage
    MemoryUsage,
    /// CPU usage
    CpuUsage,
    /// Throughput (requests/second)
    Throughput,
    /// Error rate
    ErrorRate,
}

/// Aggregation period
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregationPeriod {
    /// Per minute
    Minute,
    /// Per hour
    Hour,
    /// Per day
    Day,
    /// Per week
    Week,
}

impl AggregationPeriod {
    /// Get the duration for this aggregation period
    #[allow(dead_code)]
    pub fn duration(&self) -> Duration {
        match self {
            AggregationPeriod::Minute => Duration::from_secs(60),
            AggregationPeriod::Hour => Duration::from_secs(3600),
            AggregationPeriod::Day => Duration::from_secs(86400),
            AggregationPeriod::Week => Duration::from_secs(604800),
        }
    }
}

/// Query execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTrace {
    /// Trace ID
    pub trace_id: String,
    /// Query fingerprint (normalized query)
    pub query_fingerprint: String,
    /// Operation name
    pub operation_name: Option<String>,
    /// Operation type (query/mutation/subscription)
    pub operation_type: String,
    /// Start time
    pub started_at: SystemTime,
    /// Total duration (ms)
    pub total_duration_ms: u64,
    /// Phase timings
    pub phases: Vec<PhaseTrace>,
    /// Field traces
    pub field_traces: Vec<FieldTrace>,
    /// Variables (sanitized)
    pub variables: HashMap<String, String>,
    /// Result size (bytes)
    pub result_size_bytes: usize,
    /// Error count
    pub error_count: u32,
    /// Cache hit
    pub cache_hit: bool,
    /// Client info
    pub client_info: Option<ClientInfo>,
}

impl QueryTrace {
    /// Create a new query trace
    pub fn new(query_fingerprint: &str, operation_type: &str) -> Self {
        Self {
            trace_id: uuid::Uuid::new_v4().to_string(),
            query_fingerprint: query_fingerprint.to_string(),
            operation_name: None,
            operation_type: operation_type.to_string(),
            started_at: SystemTime::now(),
            total_duration_ms: 0,
            phases: Vec::new(),
            field_traces: Vec::new(),
            variables: HashMap::new(),
            result_size_bytes: 0,
            error_count: 0,
            cache_hit: false,
            client_info: None,
        }
    }

    /// Set operation name
    pub fn with_operation_name(mut self, name: &str) -> Self {
        self.operation_name = Some(name.to_string());
        self
    }

    /// Add phase trace
    pub fn add_phase(&mut self, phase: PhaseTrace) {
        self.phases.push(phase);
    }

    /// Add field trace
    pub fn add_field_trace(&mut self, trace: FieldTrace) {
        self.field_traces.push(trace);
    }

    /// Finalize trace with duration
    pub fn finalize(&mut self, duration_ms: u64) {
        self.total_duration_ms = duration_ms;
    }
}

/// Phase trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTrace {
    /// Phase name
    pub name: String,
    /// Duration (ms)
    pub duration_ms: u64,
    /// Start offset from query start (ms)
    pub start_offset_ms: u64,
    /// Additional details
    pub details: HashMap<String, String>,
}

impl PhaseTrace {
    /// Create a new phase trace
    pub fn new(name: &str, duration_ms: u64, start_offset_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            duration_ms,
            start_offset_ms,
            details: HashMap::new(),
        }
    }
}

/// Field trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldTrace {
    /// Field path (e.g., "Query.users.name")
    pub path: String,
    /// Parent type
    pub parent_type: String,
    /// Field name
    pub field_name: String,
    /// Return type
    pub return_type: String,
    /// Duration (ms)
    pub duration_ms: u64,
    /// Start offset (ms)
    pub start_offset_ms: u64,
    /// Is resolver
    pub is_resolver: bool,
    /// Error
    pub error: Option<String>,
}

impl FieldTrace {
    /// Create a new field trace
    pub fn new(path: &str, parent_type: &str, field_name: &str) -> Self {
        Self {
            path: path.to_string(),
            parent_type: parent_type.to_string(),
            field_name: field_name.to_string(),
            return_type: String::new(),
            duration_ms: 0,
            start_offset_ms: 0,
            is_resolver: false,
            error: None,
        }
    }
}

/// Client info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client name
    pub name: Option<String>,
    /// Client version
    pub version: Option<String>,
    /// IP address (anonymized)
    pub ip_hash: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
}

/// Aggregated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Time bucket start
    pub bucket_start: SystemTime,
    /// Time bucket end
    pub bucket_end: SystemTime,
    /// Request count
    pub request_count: u64,
    /// Error count
    pub error_count: u64,
    /// Average duration (ms)
    pub avg_duration_ms: f64,
    /// P50 duration (ms)
    pub p50_duration_ms: u64,
    /// P95 duration (ms)
    pub p95_duration_ms: u64,
    /// P99 duration (ms)
    pub p99_duration_ms: u64,
    /// Max duration (ms)
    pub max_duration_ms: u64,
    /// Min duration (ms)
    pub min_duration_ms: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Throughput (requests/second)
    pub throughput: f64,
    /// Total bytes processed
    pub total_bytes: u64,
}

impl Default for AggregatedMetrics {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            bucket_start: now,
            bucket_end: now,
            request_count: 0,
            error_count: 0,
            avg_duration_ms: 0.0,
            p50_duration_ms: 0,
            p95_duration_ms: 0,
            p99_duration_ms: 0,
            max_duration_ms: 0,
            min_duration_ms: u64::MAX,
            cache_hit_rate: 0.0,
            throughput: 0.0,
            total_bytes: 0,
        }
    }
}

/// Query statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    /// Query fingerprint
    pub fingerprint: String,
    /// Operation name
    pub operation_name: Option<String>,
    /// Total executions
    pub total_executions: u64,
    /// Error count
    pub error_count: u64,
    /// Average duration (ms)
    pub avg_duration_ms: f64,
    /// P95 duration (ms)
    pub p95_duration_ms: u64,
    /// P99 duration (ms)
    pub p99_duration_ms: u64,
    /// Max duration (ms)
    pub max_duration_ms: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average result size
    pub avg_result_size: f64,
    /// First seen
    pub first_seen: SystemTime,
    /// Last seen
    pub last_seen: SystemTime,
}

impl QueryStats {
    fn new(fingerprint: &str) -> Self {
        let now = SystemTime::now();
        Self {
            fingerprint: fingerprint.to_string(),
            operation_name: None,
            total_executions: 0,
            error_count: 0,
            avg_duration_ms: 0.0,
            p95_duration_ms: 0,
            p99_duration_ms: 0,
            max_duration_ms: 0,
            cache_hit_rate: 0.0,
            avg_result_size: 0.0,
            first_seen: now,
            last_seen: now,
        }
    }
}

/// Performance insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsight {
    /// Insight ID
    pub id: String,
    /// Insight type
    pub insight_type: InsightType,
    /// Severity
    pub severity: Severity,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Affected queries
    pub affected_queries: Vec<String>,
    /// Recommendation
    pub recommendation: Option<String>,
    /// Detected at
    pub detected_at: SystemTime,
    /// Data supporting the insight
    pub supporting_data: HashMap<String, String>,
}

/// Insight type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    /// Slow query detected
    SlowQuery,
    /// High error rate
    HighErrorRate,
    /// Performance regression
    Regression,
    /// Optimization opportunity
    Optimization,
    /// N+1 query pattern
    NPlusOne,
    /// Low cache hit rate
    LowCacheHit,
    /// High memory usage
    HighMemory,
    /// Unusual traffic pattern
    TrafficAnomaly,
}

/// Severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Critical
    Critical,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Alert name
    pub name: String,
    /// Metric type to monitor
    pub metric_type: MetricType,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Duration to exceed threshold
    pub duration: Duration,
    /// Enabled
    pub enabled: bool,
    /// Notification channels
    pub channels: Vec<String>,
}

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Retention period for traces
    pub trace_retention: Duration,
    /// Maximum traces to keep
    pub max_traces: usize,
    /// Aggregation periods to compute
    pub aggregation_periods: Vec<AggregationPeriod>,
    /// Slow query threshold (ms)
    pub slow_query_threshold_ms: u64,
    /// Enable detailed field tracing
    pub enable_field_tracing: bool,
    /// Sample rate for detailed traces (0.0-1.0)
    pub trace_sample_rate: f64,
    /// Alert configurations
    pub alerts: Vec<AlertConfig>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            trace_retention: Duration::from_secs(86400), // 24 hours
            max_traces: 10000,
            aggregation_periods: vec![
                AggregationPeriod::Minute,
                AggregationPeriod::Hour,
                AggregationPeriod::Day,
            ],
            slow_query_threshold_ms: 1000,
            enable_field_tracing: true,
            trace_sample_rate: 1.0, // 100% sampling
            alerts: Vec::new(),
        }
    }
}

/// Internal state
struct DashboardState {
    /// Query traces (recent)
    traces: VecDeque<QueryTrace>,
    /// Query statistics by fingerprint
    query_stats: HashMap<String, QueryStats>,
    /// Duration samples by fingerprint
    duration_samples: HashMap<String, Vec<u64>>,
    /// Aggregated metrics by period
    aggregations: HashMap<AggregationPeriod, VecDeque<AggregatedMetrics>>,
    /// Current period metrics
    current_period_metrics: HashMap<AggregationPeriod, Vec<u64>>,
    /// Performance insights
    insights: Vec<PerformanceInsight>,
    /// Triggered alerts
    triggered_alerts: Vec<(SystemTime, String, String)>,
    /// Start time
    started_at: Instant,
}

impl DashboardState {
    fn new() -> Self {
        Self {
            traces: VecDeque::new(),
            query_stats: HashMap::new(),
            duration_samples: HashMap::new(),
            aggregations: HashMap::new(),
            current_period_metrics: HashMap::new(),
            insights: Vec::new(),
            triggered_alerts: Vec::new(),
            started_at: Instant::now(),
        }
    }
}

/// Performance Insights Dashboard
///
/// Provides comprehensive performance analytics and monitoring
/// for GraphQL queries.
pub struct PerformanceInsightsDashboard {
    /// Configuration
    config: DashboardConfig,
    /// Internal state
    state: Arc<RwLock<DashboardState>>,
}

impl PerformanceInsightsDashboard {
    /// Create a new dashboard
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(DashboardState::new())),
        }
    }

    /// Record a query trace
    pub async fn record_trace(&self, trace: QueryTrace) {
        let mut state = self.state.write().await;

        // Update duration statistics first
        let samples = state
            .duration_samples
            .entry(trace.query_fingerprint.clone())
            .or_default();
        samples.push(trace.total_duration_ms);

        // Limit samples
        if samples.len() > 1000 {
            samples.drain(0..100);
        }

        // Compute statistics from samples
        let sum: u64 = samples.iter().sum();
        let avg_duration = sum as f64 / samples.len() as f64;

        let mut sorted = samples.clone();
        sorted.sort();
        let p95_duration = sorted[(sorted.len() as f64 * 0.95) as usize];
        let p99_duration = sorted[(sorted.len() as f64 * 0.99) as usize];

        // Now update query stats with computed values
        let stats = state
            .query_stats
            .entry(trace.query_fingerprint.clone())
            .or_insert_with(|| QueryStats::new(&trace.query_fingerprint));

        stats.total_executions += 1;
        stats.error_count += trace.error_count as u64;
        stats.last_seen = SystemTime::now();

        if stats.operation_name.is_none() {
            stats.operation_name = trace.operation_name.clone();
        }

        stats.avg_duration_ms = avg_duration;
        stats.p95_duration_ms = p95_duration;
        stats.p99_duration_ms = p99_duration;
        stats.max_duration_ms = stats.max_duration_ms.max(trace.total_duration_ms);

        // Update aggregation metrics
        for period in &self.config.aggregation_periods {
            let metrics = state.current_period_metrics.entry(*period).or_default();
            metrics.push(trace.total_duration_ms);
        }

        // Store trace
        state.traces.push_back(trace.clone());

        // Limit traces
        while state.traces.len() > self.config.max_traces {
            state.traces.pop_front();
        }

        // Check for slow query
        if trace.total_duration_ms > self.config.slow_query_threshold_ms {
            self.detect_slow_query(&mut state, &trace);
        }
    }

    fn detect_slow_query(&self, state: &mut DashboardState, trace: &QueryTrace) {
        let insight = PerformanceInsight {
            id: uuid::Uuid::new_v4().to_string(),
            insight_type: InsightType::SlowQuery,
            severity: if trace.total_duration_ms > self.config.slow_query_threshold_ms * 5 {
                Severity::Critical
            } else {
                Severity::Warning
            },
            title: "Slow Query Detected".to_string(),
            description: format!(
                "Query took {}ms (threshold: {}ms)",
                trace.total_duration_ms, self.config.slow_query_threshold_ms
            ),
            affected_queries: vec![trace.query_fingerprint.clone()],
            recommendation: Some(
                "Consider adding indexes, caching, or optimizing resolvers".to_string(),
            ),
            detected_at: SystemTime::now(),
            supporting_data: {
                let mut data = HashMap::new();
                data.insert(
                    "duration_ms".to_string(),
                    trace.total_duration_ms.to_string(),
                );
                data.insert("operation".to_string(), trace.operation_type.clone());
                data
            },
        };

        state.insights.push(insight);

        // Limit insights
        if state.insights.len() > 1000 {
            state.insights.drain(0..100);
        }
    }

    /// Get aggregated metrics for a period
    pub async fn get_aggregated_metrics(
        &self,
        period: AggregationPeriod,
    ) -> Vec<AggregatedMetrics> {
        let state = self.state.read().await;
        state
            .aggregations
            .get(&period)
            .map(|v| v.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get query statistics
    pub async fn get_query_stats(&self) -> Vec<QueryStats> {
        let state = self.state.read().await;
        let mut stats: Vec<_> = state.query_stats.values().cloned().collect();
        stats.sort_by(|a, b| b.total_executions.cmp(&a.total_executions));
        stats
    }

    /// Get slowest queries
    pub async fn get_slowest_queries(&self, limit: usize) -> Vec<QueryStats> {
        let state = self.state.read().await;
        let mut stats: Vec<_> = state.query_stats.values().cloned().collect();
        stats.sort_by(|a, b| b.p99_duration_ms.cmp(&a.p99_duration_ms));
        stats.truncate(limit);
        stats
    }

    /// Get most frequent queries
    pub async fn get_most_frequent_queries(&self, limit: usize) -> Vec<QueryStats> {
        let state = self.state.read().await;
        let mut stats: Vec<_> = state.query_stats.values().cloned().collect();
        stats.sort_by(|a, b| b.total_executions.cmp(&a.total_executions));
        stats.truncate(limit);
        stats
    }

    /// Get recent traces
    pub async fn get_recent_traces(&self, limit: usize) -> Vec<QueryTrace> {
        let state = self.state.read().await;
        state.traces.iter().rev().take(limit).cloned().collect()
    }

    /// Get trace by ID
    pub async fn get_trace(&self, trace_id: &str) -> Option<QueryTrace> {
        let state = self.state.read().await;
        state
            .traces
            .iter()
            .find(|t| t.trace_id == trace_id)
            .cloned()
    }

    /// Get performance insights
    pub async fn get_insights(&self) -> Vec<PerformanceInsight> {
        let state = self.state.read().await;
        state.insights.clone()
    }

    /// Get insights by severity
    pub async fn get_insights_by_severity(&self, severity: Severity) -> Vec<PerformanceInsight> {
        let state = self.state.read().await;
        state
            .insights
            .iter()
            .filter(|i| i.severity == severity)
            .cloned()
            .collect()
    }

    /// Generate optimization recommendations
    pub async fn generate_recommendations(&self) -> Vec<Recommendation> {
        let state = self.state.read().await;
        let mut recommendations = Vec::new();

        // Check for slow queries
        for stats in state.query_stats.values() {
            if stats.p95_duration_ms > self.config.slow_query_threshold_ms {
                recommendations.push(Recommendation {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: RecommendationCategory::Performance,
                    priority: Priority::High,
                    title: format!("Optimize slow query: {:?}", stats.operation_name),
                    description: format!(
                        "P95 latency is {}ms. Consider adding caching or optimizing resolvers.",
                        stats.p95_duration_ms
                    ),
                    affected_queries: vec![stats.fingerprint.clone()],
                    estimated_impact: Some("Could improve response time by 50%+".to_string()),
                });
            }

            // Check for low cache hit rate
            if stats.cache_hit_rate < 0.5 && stats.total_executions > 100 {
                recommendations.push(Recommendation {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: RecommendationCategory::Caching,
                    priority: Priority::Medium,
                    title: format!("Improve caching for: {:?}", stats.operation_name),
                    description: format!(
                        "Cache hit rate is {:.1}%. Consider adjusting cache TTL or query patterns.",
                        stats.cache_hit_rate * 100.0
                    ),
                    affected_queries: vec![stats.fingerprint.clone()],
                    estimated_impact: Some("Could reduce database load significantly".to_string()),
                });
            }
        }

        // Check for high error rate
        let total_executions: u64 = state.query_stats.values().map(|s| s.total_executions).sum();
        let total_errors: u64 = state.query_stats.values().map(|s| s.error_count).sum();
        if total_executions > 0 {
            let error_rate = total_errors as f64 / total_executions as f64;
            if error_rate > 0.01 {
                recommendations.push(Recommendation {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: RecommendationCategory::Reliability,
                    priority: Priority::High,
                    title: "High overall error rate".to_string(),
                    description: format!(
                        "Error rate is {:.2}%. Investigate error sources.",
                        error_rate * 100.0
                    ),
                    affected_queries: Vec::new(),
                    estimated_impact: Some(
                        "Improving reliability will enhance user experience".to_string(),
                    ),
                });
            }
        }

        recommendations
    }

    /// Get dashboard summary
    pub async fn get_summary(&self) -> DashboardSummary {
        let state = self.state.read().await;

        let total_executions: u64 = state.query_stats.values().map(|s| s.total_executions).sum();
        let total_errors: u64 = state.query_stats.values().map(|s| s.error_count).sum();

        let all_durations: Vec<u64> = state
            .duration_samples
            .values()
            .flat_map(|v| v.iter())
            .copied()
            .collect();

        let avg_duration = if !all_durations.is_empty() {
            all_durations.iter().sum::<u64>() as f64 / all_durations.len() as f64
        } else {
            0.0
        };

        let uptime = state.started_at.elapsed();

        DashboardSummary {
            total_queries: total_executions,
            total_errors,
            error_rate: if total_executions > 0 {
                total_errors as f64 / total_executions as f64
            } else {
                0.0
            },
            avg_duration_ms: avg_duration,
            unique_queries: state.query_stats.len(),
            active_insights: state.insights.len(),
            critical_insights: state
                .insights
                .iter()
                .filter(|i| i.severity == Severity::Critical)
                .count(),
            uptime_seconds: uptime.as_secs(),
        }
    }

    /// Export metrics for external monitoring
    pub async fn export_prometheus_metrics(&self) -> String {
        let summary = self.get_summary().await;

        let mut output = String::new();
        output.push_str("# HELP graphql_queries_total Total number of GraphQL queries\n");
        output.push_str("# TYPE graphql_queries_total counter\n");
        output.push_str(&format!(
            "graphql_queries_total {}\n",
            summary.total_queries
        ));

        output.push_str("# HELP graphql_errors_total Total number of GraphQL errors\n");
        output.push_str("# TYPE graphql_errors_total counter\n");
        output.push_str(&format!("graphql_errors_total {}\n", summary.total_errors));

        output.push_str("# HELP graphql_duration_ms_avg Average query duration in milliseconds\n");
        output.push_str("# TYPE graphql_duration_ms_avg gauge\n");
        output.push_str(&format!(
            "graphql_duration_ms_avg {:.2}\n",
            summary.avg_duration_ms
        ));

        output.push_str("# HELP graphql_unique_queries Number of unique query patterns\n");
        output.push_str("# TYPE graphql_unique_queries gauge\n");
        output.push_str(&format!(
            "graphql_unique_queries {}\n",
            summary.unique_queries
        ));

        output
    }

    /// Clear old data based on retention policy
    pub async fn cleanup(&self) {
        let mut state = self.state.write().await;

        let retention = self.config.trace_retention;
        let cutoff = SystemTime::now()
            .checked_sub(retention)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        // Remove old traces
        state.traces.retain(|t| t.started_at > cutoff);

        // Remove old insights
        state.insights.retain(|i| i.detected_at > cutoff);

        // Remove old alerts
        state.triggered_alerts.retain(|(t, _, _)| *t > cutoff);
    }
}

impl Default for PerformanceInsightsDashboard {
    fn default() -> Self {
        Self::new(DashboardConfig::default())
    }
}

/// Recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation ID
    pub id: String,
    /// Category
    pub category: RecommendationCategory,
    /// Priority
    pub priority: Priority,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Affected queries
    pub affected_queries: Vec<String>,
    /// Estimated impact
    pub estimated_impact: Option<String>,
}

/// Recommendation category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Caching,
    Reliability,
    Security,
    Schema,
}

/// Priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Dashboard summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    /// Total queries processed
    pub total_queries: u64,
    /// Total errors
    pub total_errors: u64,
    /// Error rate
    pub error_rate: f64,
    /// Average duration (ms)
    pub avg_duration_ms: f64,
    /// Number of unique query patterns
    pub unique_queries: usize,
    /// Number of active insights
    pub active_insights: usize,
    /// Number of critical insights
    pub critical_insights: usize,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// Active tracing session for recording a query
pub struct TracingSession {
    trace: QueryTrace,
    start_time: Instant,
    phase_start: Option<(String, Instant)>,
}

impl TracingSession {
    /// Start a new tracing session
    pub fn start(query_fingerprint: &str, operation_type: &str) -> Self {
        Self {
            trace: QueryTrace::new(query_fingerprint, operation_type),
            start_time: Instant::now(),
            phase_start: None,
        }
    }

    /// Set operation name
    pub fn set_operation_name(&mut self, name: &str) {
        self.trace.operation_name = Some(name.to_string());
    }

    /// Start a phase
    pub fn start_phase(&mut self, name: &str) {
        self.phase_start = Some((name.to_string(), Instant::now()));
    }

    /// End the current phase
    pub fn end_phase(&mut self) {
        if let Some((name, start)) = self.phase_start.take() {
            let duration = start.elapsed().as_millis() as u64;
            let start_offset = (self.start_time.elapsed() - start.elapsed()).as_millis() as u64;
            self.trace
                .add_phase(PhaseTrace::new(&name, duration, start_offset));
        }
    }

    /// Record a field trace
    pub fn record_field(
        &mut self,
        path: &str,
        parent_type: &str,
        field_name: &str,
        duration_ms: u64,
    ) {
        let mut field_trace = FieldTrace::new(path, parent_type, field_name);
        field_trace.duration_ms = duration_ms;
        field_trace.start_offset_ms = self.start_time.elapsed().as_millis() as u64 - duration_ms;
        self.trace.add_field_trace(field_trace);
    }

    /// Set cache hit
    pub fn set_cache_hit(&mut self, hit: bool) {
        self.trace.cache_hit = hit;
    }

    /// Set result size
    pub fn set_result_size(&mut self, size: usize) {
        self.trace.result_size_bytes = size;
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.trace.error_count += 1;
    }

    /// Finish the session and return the trace
    pub fn finish(mut self) -> QueryTrace {
        let duration = self.start_time.elapsed().as_millis() as u64;
        self.trace.finalize(duration);
        self.trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let dashboard = PerformanceInsightsDashboard::default();
        let summary = dashboard.get_summary().await;
        assert_eq!(summary.total_queries, 0);
    }

    #[tokio::test]
    async fn test_record_trace() {
        let dashboard = PerformanceInsightsDashboard::default();

        let mut trace = QueryTrace::new("{ users { id } }", "query");
        trace.total_duration_ms = 50;
        trace.result_size_bytes = 1024;

        dashboard.record_trace(trace).await;

        let summary = dashboard.get_summary().await;
        assert_eq!(summary.total_queries, 1);
        assert_eq!(summary.unique_queries, 1);
    }

    #[tokio::test]
    async fn test_slow_query_detection() {
        let config = DashboardConfig {
            slow_query_threshold_ms: 100,
            ..Default::default()
        };
        let dashboard = PerformanceInsightsDashboard::new(config);

        let mut trace = QueryTrace::new("{ slowQuery }", "query");
        trace.total_duration_ms = 500; // Slow query

        dashboard.record_trace(trace).await;

        let insights = dashboard.get_insights().await;
        assert!(!insights.is_empty());
        assert_eq!(insights[0].insight_type, InsightType::SlowQuery);
    }

    #[tokio::test]
    async fn test_query_stats() {
        let dashboard = PerformanceInsightsDashboard::default();

        // Record multiple traces for same query
        for i in 0..10 {
            let mut trace = QueryTrace::new("{ users { id } }", "query");
            trace.total_duration_ms = 50 + i * 10;
            dashboard.record_trace(trace).await;
        }

        let stats = dashboard.get_query_stats().await;
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].total_executions, 10);
    }

    #[tokio::test]
    async fn test_slowest_queries() {
        let dashboard = PerformanceInsightsDashboard::default();

        // Record traces with different durations
        for (i, duration) in [100, 500, 200].iter().enumerate() {
            let mut trace = QueryTrace::new(&format!("query{}", i), "query");
            trace.total_duration_ms = *duration;
            dashboard.record_trace(trace).await;
        }

        let slowest = dashboard.get_slowest_queries(2).await;
        assert_eq!(slowest.len(), 2);
        assert!(slowest[0].p99_duration_ms >= slowest[1].p99_duration_ms);
    }

    #[tokio::test]
    async fn test_tracing_session() {
        let mut session = TracingSession::start("{ users }", "query");
        session.set_operation_name("GetUsers");

        session.start_phase("parse");
        tokio::time::sleep(Duration::from_millis(1)).await;
        session.end_phase();

        session.start_phase("execute");
        tokio::time::sleep(Duration::from_millis(1)).await;
        session.end_phase();

        session.set_cache_hit(true);
        session.set_result_size(1024);

        let trace = session.finish();
        assert_eq!(trace.operation_name, Some("GetUsers".to_string()));
        assert_eq!(trace.phases.len(), 2);
        assert!(trace.cache_hit);
    }

    #[tokio::test]
    async fn test_prometheus_export() {
        let dashboard = PerformanceInsightsDashboard::default();

        let mut trace = QueryTrace::new("{ users }", "query");
        trace.total_duration_ms = 50;
        dashboard.record_trace(trace).await;

        let metrics = dashboard.export_prometheus_metrics().await;
        assert!(metrics.contains("graphql_queries_total 1"));
    }

    #[tokio::test]
    async fn test_recommendations() {
        let config = DashboardConfig {
            slow_query_threshold_ms: 50,
            ..Default::default()
        };
        let dashboard = PerformanceInsightsDashboard::new(config);

        // Record slow queries
        for _ in 0..10 {
            let mut trace = QueryTrace::new("{ slowQuery }", "query");
            trace.total_duration_ms = 200;
            dashboard.record_trace(trace).await;
        }

        let recommendations = dashboard.generate_recommendations().await;
        assert!(!recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_get_trace_by_id() {
        let dashboard = PerformanceInsightsDashboard::default();

        let trace = QueryTrace::new("{ users }", "query");
        let trace_id = trace.trace_id.clone();
        dashboard.record_trace(trace).await;

        let retrieved = dashboard.get_trace(&trace_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().trace_id, trace_id);
    }

    #[tokio::test]
    async fn test_insights_by_severity() {
        let config = DashboardConfig {
            slow_query_threshold_ms: 100,
            ..Default::default()
        };
        let dashboard = PerformanceInsightsDashboard::new(config);

        // Critical slow query (5x threshold)
        let mut trace = QueryTrace::new("{ verySlowQuery }", "query");
        trace.total_duration_ms = 600;
        dashboard.record_trace(trace).await;

        // Warning slow query
        let mut trace = QueryTrace::new("{ slowQuery }", "query");
        trace.total_duration_ms = 200;
        dashboard.record_trace(trace).await;

        let critical = dashboard.get_insights_by_severity(Severity::Critical).await;
        assert!(!critical.is_empty());

        let warnings = dashboard.get_insights_by_severity(Severity::Warning).await;
        assert!(!warnings.is_empty());
    }
}
