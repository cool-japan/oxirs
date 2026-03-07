//! # Query Monitor
//!
//! Federation query execution monitoring and alerting.
//!
//! Tracks active and completed federated query executions, emits [`QueryAlert`]s when
//! configured thresholds are breached (slow queries, timeouts, high row counts, error rate).

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a single query execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    /// The query is still running.
    Running,
    /// The query completed successfully.
    Completed {
        /// Wall-clock time from start to finish in milliseconds.
        duration_ms: u64,
        /// Number of result rows returned.
        row_count: usize,
    },
    /// The query failed with an error message.
    Failed(String),
    /// The query was forcibly terminated because it exceeded the configured timeout.
    Timeout,
}

/// A record of one federated query execution.
#[derive(Debug, Clone)]
pub struct QueryExecution {
    /// Unique identifier for this execution.
    pub id: String,
    /// The SPARQL / GraphQL query text.
    pub query_text: String,
    /// Unix-epoch millisecond timestamp at which execution began.
    pub started_at: u64,
    /// SPARQL endpoints contacted by this query.
    pub endpoints: Vec<String>,
    /// Current lifecycle status.
    pub status: ExecutionStatus,
}

/// Classification of a monitoring alert.
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    /// Query exceeded the slow-query threshold; payload is the actual duration in ms.
    SlowQuery(u64),
    /// The error rate crossed the configured threshold; payload is the current rate.
    ErrorRate(f64),
    /// A query was terminated due to timeout.
    TimeoutDetected,
    /// A completed query returned more rows than the configured maximum.
    HighRowCount(usize),
}

/// A single monitoring alert emitted by [`QueryMonitor`].
#[derive(Debug, Clone)]
pub struct QueryAlert {
    /// The execution that triggered the alert.
    pub execution_id: String,
    /// What kind of condition was detected.
    pub alert_type: AlertType,
    /// Human-readable description.
    pub message: String,
    /// Unix-epoch millisecond timestamp of the alert.
    pub timestamp: u64,
}

/// Tuning parameters for the monitor.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Queries taking longer than this many milliseconds are flagged as slow.
    pub slow_query_threshold_ms: u64,
    /// Queries that do not complete within this many milliseconds will be timed out.
    pub timeout_ms: u64,
    /// Queries returning more rows than this are flagged.
    pub max_row_count: usize,
    /// When the error rate (failed / total finished) exceeds this fraction an alert is emitted.
    pub error_rate_threshold: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            slow_query_threshold_ms: 5_000,
            timeout_ms: 30_000,
            max_row_count: 100_000,
            error_rate_threshold: 0.1,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// QueryMonitor
// ────────────────────────────────────────────────────────────────────────────

/// Tracks federated query executions and emits alerts when thresholds are exceeded.
///
/// # Example
/// ```rust
/// use oxirs_federate::query_monitor::{MonitorConfig, QueryMonitor};
///
/// let config = MonitorConfig { slow_query_threshold_ms: 1000, ..Default::default() };
/// let mut monitor = QueryMonitor::new(config);
///
/// monitor.start_execution("q1", "SELECT * WHERE { ?s ?p ?o }", vec!["http://ep1".to_string()]);
/// let alerts = monitor.complete_execution("q1", 5, 2000);
/// // alerts will contain a SlowQuery alert because 2000 > 1000
/// assert!(!alerts.is_empty());
/// ```
#[derive(Debug)]
pub struct QueryMonitor {
    config: MonitorConfig,
    /// Active or finished executions, keyed by ID.
    executions: HashMap<String, QueryExecution>,
    /// Monotonically increasing logical clock used when a real clock is unavailable.
    logical_clock: u64,
    /// Accumulated alerts across all executions.
    alert_log: Vec<QueryAlert>,
    /// Count of executions that terminated with `Failed`.
    failed_count: usize,
    /// Count of executions that terminated with `Completed`.
    completed_count: usize,
}

impl QueryMonitor {
    /// Create a monitor with the given configuration.
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            executions: HashMap::new(),
            logical_clock: 0,
            alert_log: Vec::new(),
            failed_count: 0,
            completed_count: 0,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn tick(&mut self) -> u64 {
        self.logical_clock += 1;
        self.logical_clock
    }

    fn push_alert(&mut self, execution_id: &str, alert_type: AlertType, message: String) {
        let ts = self.tick();
        self.alert_log.push(QueryAlert {
            execution_id: execution_id.to_string(),
            alert_type,
            message,
            timestamp: ts,
        });
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Register a new query execution as `Running`.
    ///
    /// If an execution with the same `id` already exists it is silently overwritten.
    pub fn start_execution(&mut self, id: &str, query: &str, endpoints: Vec<String>) {
        let started_at = self.tick();
        self.executions.insert(
            id.to_string(),
            QueryExecution {
                id: id.to_string(),
                query_text: query.to_string(),
                started_at,
                endpoints,
                status: ExecutionStatus::Running,
            },
        );
    }

    /// Mark an execution as `Completed` and emit alerts as appropriate.
    ///
    /// Alerts emitted:
    /// - [`AlertType::SlowQuery`] when `duration_ms` exceeds the threshold.
    /// - [`AlertType::HighRowCount`] when `row_count` exceeds the threshold.
    /// - [`AlertType::ErrorRate`] when the global error rate now exceeds the threshold.
    ///
    /// Returns the newly emitted alerts (may be empty).
    pub fn complete_execution(
        &mut self,
        id: &str,
        row_count: usize,
        duration_ms: u64,
    ) -> Vec<QueryAlert> {
        let before = self.alert_log.len();

        if let Some(exec) = self.executions.get_mut(id) {
            exec.status = ExecutionStatus::Completed {
                duration_ms,
                row_count,
            };
        }
        self.completed_count += 1;

        // Slow-query alert
        if duration_ms > self.config.slow_query_threshold_ms {
            self.push_alert(
                id,
                AlertType::SlowQuery(duration_ms),
                format!(
                    "Query {} took {} ms (threshold: {} ms)",
                    id, duration_ms, self.config.slow_query_threshold_ms
                ),
            );
        }

        // High row-count alert
        if row_count > self.config.max_row_count {
            self.push_alert(
                id,
                AlertType::HighRowCount(row_count),
                format!(
                    "Query {} returned {} rows (max: {})",
                    id, row_count, self.config.max_row_count
                ),
            );
        }

        // Error-rate alert
        let rate = self.error_rate();
        if rate > self.config.error_rate_threshold {
            self.push_alert(
                id,
                AlertType::ErrorRate(rate),
                format!(
                    "Error rate {:.2}% exceeds threshold {:.2}%",
                    rate * 100.0,
                    self.config.error_rate_threshold * 100.0
                ),
            );
        }

        self.alert_log[before..].to_vec()
    }

    /// Mark an execution as `Failed` and emit alerts as appropriate.
    ///
    /// Alerts emitted:
    /// - [`AlertType::ErrorRate`] when the updated error rate exceeds the threshold.
    pub fn fail_execution(&mut self, id: &str, error: &str) -> Vec<QueryAlert> {
        let before = self.alert_log.len();

        if let Some(exec) = self.executions.get_mut(id) {
            exec.status = ExecutionStatus::Failed(error.to_string());
        }
        self.failed_count += 1;

        let rate = self.error_rate();
        if rate > self.config.error_rate_threshold {
            self.push_alert(
                id,
                AlertType::ErrorRate(rate),
                format!(
                    "Error rate {:.2}% exceeds threshold {:.2}% after failure of query {}",
                    rate * 100.0,
                    self.config.error_rate_threshold * 100.0,
                    id
                ),
            );
        }

        self.alert_log[before..].to_vec()
    }

    /// Mark an execution as `Timeout` and emit a [`AlertType::TimeoutDetected`] alert.
    pub fn timeout_execution(&mut self, id: &str) -> Vec<QueryAlert> {
        let before = self.alert_log.len();

        if let Some(exec) = self.executions.get_mut(id) {
            exec.status = ExecutionStatus::Timeout;
        }
        self.failed_count += 1;

        self.push_alert(
            id,
            AlertType::TimeoutDetected,
            format!("Query {} timed out after {} ms", id, self.config.timeout_ms),
        );

        // Also check error rate after timeout
        let rate = self.error_rate();
        if rate > self.config.error_rate_threshold {
            self.push_alert(
                id,
                AlertType::ErrorRate(rate),
                format!(
                    "Error rate {:.2}% exceeds threshold after timeout of query {}",
                    rate * 100.0,
                    id
                ),
            );
        }

        self.alert_log[before..].to_vec()
    }

    /// Number of executions currently in the `Running` state.
    pub fn active_count(&self) -> usize {
        self.executions
            .values()
            .filter(|e| e.status == ExecutionStatus::Running)
            .count()
    }

    /// Number of executions that have reached `Completed` status.
    pub fn completed_count(&self) -> usize {
        self.completed_count
    }

    /// All alerts emitted so far (oldest first).
    pub fn alerts(&self) -> &[QueryAlert] {
        &self.alert_log
    }

    /// Fraction of finished executions that failed.
    ///
    /// Returns `0.0` when no executions have finished yet.
    pub fn error_rate(&self) -> f64 {
        let total = self.failed_count + self.completed_count;
        if total == 0 {
            0.0
        } else {
            self.failed_count as f64 / total as f64
        }
    }

    /// Return a reference to a specific execution, or `None` if not found.
    pub fn get_execution(&self, id: &str) -> Option<&QueryExecution> {
        self.executions.get(id)
    }

    /// Total number of tracked executions (active + finished).
    pub fn total_count(&self) -> usize {
        self.executions.len()
    }

    /// Clear all stored alert history.
    pub fn clear_alerts(&mut self) {
        self.alert_log.clear();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_monitor() -> QueryMonitor {
        QueryMonitor::new(MonitorConfig {
            slow_query_threshold_ms: 1_000,
            timeout_ms: 10_000,
            max_row_count: 100,
            error_rate_threshold: 0.5,
        })
    }

    fn endpoints(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    // ── start_execution ───────────────────────────────────────────────────

    #[test]
    fn test_start_execution_active_count() {
        let mut m = default_monitor();
        m.start_execution("q1", "SELECT ?s WHERE {}", endpoints(&["ep1"]));
        assert_eq!(m.active_count(), 1);
    }

    #[test]
    fn test_start_multiple_active() {
        let mut m = default_monitor();
        for i in 0..5 {
            m.start_execution(&format!("q{}", i), "SELECT *", endpoints(&[]));
        }
        assert_eq!(m.active_count(), 5);
    }

    #[test]
    fn test_start_stores_query_text() {
        let mut m = default_monitor();
        m.start_execution("q1", "SELECT ?x {}", endpoints(&[]));
        let exec = m.get_execution("q1").expect("should exist");
        assert_eq!(exec.query_text, "SELECT ?x {}");
    }

    #[test]
    fn test_start_stores_endpoints() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&["http://ep1", "http://ep2"]));
        let exec = m.get_execution("q1").expect("should exist");
        assert_eq!(exec.endpoints.len(), 2);
    }

    #[test]
    fn test_start_status_is_running() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        assert_eq!(
            m.get_execution("q1").expect("exists").status,
            ExecutionStatus::Running
        );
    }

    // ── complete_execution ────────────────────────────────────────────────

    #[test]
    fn test_complete_no_alerts_when_within_threshold() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.complete_execution("q1", 10, 500); // duration < 1000, rows < 100
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_complete_slow_query_alert() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.complete_execution("q1", 5, 2_000); // 2000 > threshold 1000
        let has_slow = alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::SlowQuery(_)));
        assert!(has_slow);
    }

    #[test]
    fn test_complete_slow_query_contains_duration() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.complete_execution("q1", 5, 5_000);
        let slow = alerts
            .iter()
            .find(|a| matches!(a.alert_type, AlertType::SlowQuery(_)))
            .expect("should have slow-query alert");
        assert_eq!(slow.alert_type, AlertType::SlowQuery(5_000));
    }

    #[test]
    fn test_complete_high_row_count_alert() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.complete_execution("q1", 200, 100); // 200 > max 100
        let has_high = alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::HighRowCount(_)));
        assert!(has_high);
    }

    #[test]
    fn test_complete_row_count_at_threshold_no_alert() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.complete_execution("q1", 100, 100); // exactly at threshold
        let has_high = alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::HighRowCount(_)));
        assert!(!has_high); // must exceed, not equal
    }

    #[test]
    fn test_complete_increments_completed_count() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.complete_execution("q1", 0, 100);
        assert_eq!(m.completed_count(), 1);
    }

    #[test]
    fn test_complete_reduces_active_count() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        assert_eq!(m.active_count(), 1);
        m.complete_execution("q1", 0, 100);
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_complete_updates_status() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.complete_execution("q1", 42, 300);
        let exec = m.get_execution("q1").expect("exists");
        assert!(matches!(
            exec.status,
            ExecutionStatus::Completed {
                row_count: 42,
                duration_ms: 300
            }
        ));
    }

    // ── fail_execution ────────────────────────────────────────────────────

    #[test]
    fn test_fail_execution_increments_failed() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.fail_execution("q1", "connection refused");
        // error_rate denominator = 1, failed = 1 → rate = 1.0
        assert!((m.error_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fail_execution_updates_status() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.fail_execution("q1", "err");
        let exec = m.get_execution("q1").expect("exists");
        assert!(matches!(exec.status, ExecutionStatus::Failed(_)));
    }

    #[test]
    fn test_fail_no_alert_when_rate_below_threshold() {
        let mut m = QueryMonitor::new(MonitorConfig {
            error_rate_threshold: 0.8, // high threshold
            ..default_monitor().config
        });
        // complete one first so failure rate = 0.5
        m.start_execution("q1", "Q", endpoints(&[]));
        m.complete_execution("q1", 0, 100);
        m.start_execution("q2", "Q", endpoints(&[]));
        let alerts = m.fail_execution("q2", "err"); // rate = 0.5 < 0.8
        assert!(!alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::ErrorRate(_))));
    }

    #[test]
    fn test_fail_alert_when_rate_above_threshold() {
        let mut m = default_monitor(); // threshold 0.5
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.fail_execution("q1", "err"); // rate = 1.0 > 0.5
        assert!(alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::ErrorRate(_))));
    }

    // ── timeout_execution ─────────────────────────────────────────────────

    #[test]
    fn test_timeout_emits_timeout_alert() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.timeout_execution("q1");
        assert!(alerts
            .iter()
            .any(|a| a.alert_type == AlertType::TimeoutDetected));
    }

    #[test]
    fn test_timeout_updates_status() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.timeout_execution("q1");
        let exec = m.get_execution("q1").expect("exists");
        assert_eq!(exec.status, ExecutionStatus::Timeout);
    }

    #[test]
    fn test_timeout_counts_as_failure_for_error_rate() {
        let mut m = default_monitor(); // threshold 0.5
        m.start_execution("q1", "Q", endpoints(&[]));
        m.timeout_execution("q1");
        assert!((m.error_rate() - 1.0).abs() < 1e-9);
    }

    // ── active_count / completed_count ────────────────────────────────────

    #[test]
    fn test_active_count_zero_initially() {
        let m = default_monitor();
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_completed_count_zero_initially() {
        let m = default_monitor();
        assert_eq!(m.completed_count(), 0);
    }

    #[test]
    fn test_active_count_after_mix() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.start_execution("q2", "Q", endpoints(&[]));
        m.start_execution("q3", "Q", endpoints(&[]));
        m.complete_execution("q1", 0, 100);
        m.fail_execution("q2", "err");
        assert_eq!(m.active_count(), 1); // only q3 still running
    }

    #[test]
    fn test_total_count() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.start_execution("q2", "Q", endpoints(&[]));
        assert_eq!(m.total_count(), 2);
    }

    // ── alerts() ─────────────────────────────────────────────────────────

    #[test]
    fn test_alerts_empty_initially() {
        let m = default_monitor();
        assert!(m.alerts().is_empty());
    }

    #[test]
    fn test_alerts_accumulate_across_executions() {
        let mut m = QueryMonitor::new(MonitorConfig {
            slow_query_threshold_ms: 10,
            ..default_monitor().config
        });
        m.start_execution("q1", "Q", endpoints(&[]));
        m.complete_execution("q1", 0, 100); // slow: 1 alert
        m.start_execution("q2", "Q", endpoints(&[]));
        m.complete_execution("q2", 0, 200); // slow: another alert
        assert!(m.alerts().len() >= 2);
    }

    #[test]
    fn test_clear_alerts() {
        let mut m = QueryMonitor::new(MonitorConfig {
            slow_query_threshold_ms: 10,
            ..default_monitor().config
        });
        m.start_execution("q1", "Q", endpoints(&[]));
        m.complete_execution("q1", 0, 100);
        assert!(!m.alerts().is_empty());
        m.clear_alerts();
        assert!(m.alerts().is_empty());
    }

    // ── error_rate() ──────────────────────────────────────────────────────

    #[test]
    fn test_error_rate_zero_when_no_executions() {
        let m = default_monitor();
        assert!((m.error_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_error_rate_zero_with_only_completions() {
        let mut m = default_monitor();
        for i in 0..5 {
            m.start_execution(&format!("q{}", i), "Q", endpoints(&[]));
            m.complete_execution(&format!("q{}", i), 0, 100);
        }
        assert!((m.error_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_error_rate_half_half() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.complete_execution("q1", 0, 100);
        m.start_execution("q2", "Q", endpoints(&[]));
        m.fail_execution("q2", "err");
        assert!((m.error_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_error_rate_one_all_failures() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        m.fail_execution("q1", "err");
        assert!((m.error_rate() - 1.0).abs() < 1e-9);
    }

    // ── MonitorConfig ─────────────────────────────────────────────────────

    #[test]
    fn test_default_config_reasonable() {
        let cfg = MonitorConfig::default();
        assert!(cfg.slow_query_threshold_ms > 0);
        assert!(cfg.timeout_ms > 0);
        assert!(cfg.max_row_count > 0);
        assert!(cfg.error_rate_threshold > 0.0 && cfg.error_rate_threshold < 1.0);
    }

    // ── Alert IDs / messages ──────────────────────────────────────────────

    #[test]
    fn test_alert_execution_id_matches() {
        let mut m = QueryMonitor::new(MonitorConfig {
            slow_query_threshold_ms: 10,
            ..default_monitor().config
        });
        m.start_execution("my-query", "Q", endpoints(&[]));
        let alerts = m.complete_execution("my-query", 0, 100);
        assert!(alerts.iter().all(|a| a.execution_id == "my-query"));
    }

    #[test]
    fn test_timeout_alert_message_mentions_timeout() {
        let mut m = default_monitor();
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.timeout_execution("q1");
        let ta = alerts
            .iter()
            .find(|a| a.alert_type == AlertType::TimeoutDetected)
            .expect("should have timeout alert");
        assert!(ta.message.contains("timed out") || ta.message.contains("timeout"));
    }

    #[test]
    fn test_slow_query_alert_message_contains_duration() {
        let mut m = QueryMonitor::new(MonitorConfig {
            slow_query_threshold_ms: 100,
            ..default_monitor().config
        });
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.complete_execution("q1", 0, 9_999);
        let sa = alerts
            .iter()
            .find(|a| matches!(a.alert_type, AlertType::SlowQuery(_)))
            .expect("should have slow alert");
        assert!(sa.message.contains("9999"));
    }

    // ── No duplicate alerts on exact threshold ────────────────────────────

    #[test]
    fn test_no_slow_alert_at_exactly_threshold() {
        let mut m = default_monitor(); // threshold 1000
        m.start_execution("q1", "Q", endpoints(&[]));
        let alerts = m.complete_execution("q1", 0, 1_000); // equal, not exceeding
        assert!(!alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::SlowQuery(_))));
    }

    // ── Overwriting a running execution ───────────────────────────────────

    #[test]
    fn test_start_execution_overwrites_existing() {
        let mut m = default_monitor();
        m.start_execution("q1", "old query", endpoints(&[]));
        m.start_execution("q1", "new query", endpoints(&[]));
        let exec = m.get_execution("q1").expect("exists");
        assert_eq!(exec.query_text, "new query");
        assert_eq!(m.total_count(), 1); // still one entry
    }

    // ── Unknown execution ID ──────────────────────────────────────────────

    #[test]
    fn test_complete_unknown_id_no_panic() {
        let mut m = default_monitor();
        // Completing an execution that was never started should not panic.
        let alerts = m.complete_execution("ghost", 0, 100);
        // No crash; completed_count is updated, no row/slow alerts for ghost.
        assert_eq!(m.completed_count(), 1);
        let _ = alerts; // may or may not contain alerts depending on rate
    }

    #[test]
    fn test_get_execution_returns_none_for_unknown() {
        let m = default_monitor();
        assert!(m.get_execution("unknown").is_none());
    }
}
