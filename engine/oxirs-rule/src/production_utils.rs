//! # Production Utilities
//!
//! Utilities for deploying and monitoring oxirs-rule engines in production environments.
//!
//! ## Features
//!
//! - **Health Checks**: Rule engine health monitoring
//! - **Metrics Collection**: Performance metrics for production
//! - **Error Tracking**: Enhanced error reporting and debugging
//! - **Resource Monitoring**: Memory and CPU usage tracking
//! - **Audit Logging**: Rule execution audit trails
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::production_utils::{ProductionMonitor, HealthStatus};
//!
//! let mut monitor = ProductionMonitor::new("my-rule-engine");
//!
//! // Check health
//! let health = monitor.check_health();
//! assert_eq!(health.status, HealthStatus::Healthy);
//!
//! // Record metrics
//! monitor.record_rule_execution("rule1", std::time::Duration::from_millis(10));
//!
//! // Get statistics
//! let stats = monitor.get_statistics();
//! println!("Total executions: {}", stats.total_executions);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Production monitor for rule engines
#[derive(Debug)]
pub struct ProductionMonitor {
    /// Engine identifier
    engine_id: String,
    /// Start time
    start_time: Instant,
    /// Rule execution times
    execution_times: HashMap<String, Vec<Duration>>,
    /// Error counts
    error_counts: HashMap<String, usize>,
    /// Total executions
    total_executions: usize,
    /// Resource snapshots
    resource_snapshots: Vec<ResourceSnapshot>,
    /// Audit log
    audit_log: Vec<AuditEntry>,
    /// Configuration
    config: MonitorConfig,
}

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Enable audit logging
    pub enable_audit: bool,
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Maximum audit log size
    pub max_audit_entries: usize,
    /// Performance threshold (warn if execution exceeds)
    pub performance_threshold_ms: u64,
    /// Memory threshold (warn if usage exceeds in MB)
    pub memory_threshold_mb: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enable_audit: true,
            enable_resource_monitoring: true,
            max_audit_entries: 10000,
            performance_threshold_ms: 1000,
            memory_threshold_mb: 1024,
        }
    }
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System is degraded but operational
    Degraded,
    /// System is unhealthy
    Unhealthy,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Overall status
    pub status: HealthStatus,
    /// Timestamp
    pub timestamp: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Total executions
    pub total_executions: usize,
    /// Error rate (errors per 100 executions)
    pub error_rate: f64,
    /// Average execution time in milliseconds
    pub avg_execution_ms: f64,
    /// Memory usage in MB
    pub memory_mb: usize,
    /// Issues detected
    pub issues: Vec<String>,
}

/// Resource snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// Memory usage estimate (in bytes)
    pub memory_bytes: usize,
    /// Active rules count
    pub active_rules: usize,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp
    pub timestamp: String,
    /// Event type
    pub event_type: AuditEventType,
    /// Rule name
    pub rule_name: Option<String>,
    /// Details
    pub details: String,
    /// Duration (if applicable)
    pub duration_ms: Option<u64>,
}

/// Audit event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Rule execution started
    ExecutionStarted,
    /// Rule execution completed
    ExecutionCompleted,
    /// Rule execution failed
    ExecutionFailed,
    /// Engine started
    EngineStarted,
    /// Engine stopped
    EngineStopped,
    /// Configuration changed
    ConfigChanged,
}

/// Production statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionStatistics {
    /// Total executions
    pub total_executions: usize,
    /// Total errors
    pub total_errors: usize,
    /// Average execution time (ms)
    pub avg_execution_ms: f64,
    /// P50 execution time (ms)
    pub p50_execution_ms: f64,
    /// P95 execution time (ms)
    pub p95_execution_ms: f64,
    /// P99 execution time (ms)
    pub p99_execution_ms: f64,
    /// Max execution time (ms)
    pub max_execution_ms: f64,
    /// Rules by execution count
    pub rules_by_count: Vec<(String, usize)>,
    /// Rules by total time
    pub rules_by_time: Vec<(String, f64)>,
    /// Slowest rules
    pub slowest_rules: Vec<(String, f64)>,
}

impl ProductionMonitor {
    /// Create a new production monitor
    pub fn new(engine_id: impl Into<String>) -> Self {
        let engine_id = engine_id.into();
        let mut monitor = Self {
            engine_id: engine_id.clone(),
            start_time: Instant::now(),
            execution_times: HashMap::new(),
            error_counts: HashMap::new(),
            total_executions: 0,
            resource_snapshots: Vec::new(),
            audit_log: Vec::new(),
            config: MonitorConfig::default(),
        };

        // Log engine start
        monitor.log_audit(
            AuditEventType::EngineStarted,
            None,
            format!("Engine {} started", engine_id),
            None,
        );

        monitor
    }

    /// Get engine identifier
    pub fn engine_id(&self) -> &str {
        &self.engine_id
    }

    /// Create with custom configuration
    pub fn with_config(engine_id: impl Into<String>, config: MonitorConfig) -> Self {
        let mut monitor = Self::new(engine_id);
        monitor.config = config;
        monitor
    }

    /// Record rule execution
    pub fn record_rule_execution(&mut self, rule_name: impl Into<String>, duration: Duration) {
        let rule_name = rule_name.into();
        self.total_executions += 1;

        // Record timing
        self.execution_times
            .entry(rule_name.clone())
            .or_default()
            .push(duration);

        // Check performance threshold
        if duration.as_millis() as u64 > self.config.performance_threshold_ms {
            self.log_audit(
                AuditEventType::ExecutionCompleted,
                Some(rule_name.clone()),
                format!("Slow execution: {}ms", duration.as_millis()),
                Some(duration.as_millis() as u64),
            );
        }

        // Audit log
        if self.config.enable_audit {
            self.log_audit(
                AuditEventType::ExecutionCompleted,
                Some(rule_name),
                "Rule executed successfully".to_string(),
                Some(duration.as_millis() as u64),
            );
        }
    }

    /// Record error
    pub fn record_error(&mut self, rule_name: impl Into<String>, error: &str) {
        let rule_name = rule_name.into();
        *self.error_counts.entry(rule_name.clone()).or_insert(0) += 1;

        // Audit log
        if self.config.enable_audit {
            self.log_audit(
                AuditEventType::ExecutionFailed,
                Some(rule_name),
                error.to_string(),
                None,
            );
        }
    }

    /// Take resource snapshot
    pub fn snapshot_resources(&mut self, active_rules: usize) {
        if !self.config.enable_resource_monitoring {
            return;
        }

        // Estimate memory usage (simplified)
        let memory_bytes = self.estimate_memory_usage();

        self.resource_snapshots.push(ResourceSnapshot {
            timestamp: Instant::now(),
            memory_bytes,
            active_rules,
        });

        // Check memory threshold
        let memory_mb = memory_bytes / (1024 * 1024);
        if memory_mb > self.config.memory_threshold_mb {
            self.log_audit(
                AuditEventType::ConfigChanged,
                None,
                format!("High memory usage: {}MB", memory_mb),
                None,
            );
        }
    }

    /// Check health
    pub fn check_health(&self) -> HealthCheck {
        let uptime = self.start_time.elapsed();
        let total_errors: usize = self.error_counts.values().sum();
        let error_rate = if self.total_executions > 0 {
            (total_errors as f64 / self.total_executions as f64) * 100.0
        } else {
            0.0
        };

        // Calculate average execution time
        let all_times: Vec<Duration> = self.execution_times.values().flatten().copied().collect();

        let avg_execution_ms = if !all_times.is_empty() {
            let sum: Duration = all_times.iter().sum();
            (sum.as_micros() as f64 / all_times.len() as f64) / 1000.0
        } else {
            0.0
        };

        // Determine status and issues
        let mut status = HealthStatus::Healthy;
        let mut issues = Vec::new();

        if error_rate > 10.0 {
            status = HealthStatus::Unhealthy;
            issues.push(format!("High error rate: {:.2}%", error_rate));
        } else if error_rate > 5.0 {
            status = HealthStatus::Degraded;
            issues.push(format!("Elevated error rate: {:.2}%", error_rate));
        }

        if avg_execution_ms > self.config.performance_threshold_ms as f64 {
            if status == HealthStatus::Healthy {
                status = HealthStatus::Degraded;
            }
            issues.push(format!(
                "High average execution time: {:.2}ms",
                avg_execution_ms
            ));
        }

        // Memory check
        let memory_mb = if let Some(snapshot) = self.resource_snapshots.last() {
            snapshot.memory_bytes / (1024 * 1024)
        } else {
            0
        };

        if memory_mb > self.config.memory_threshold_mb {
            if status == HealthStatus::Healthy {
                status = HealthStatus::Degraded;
            }
            issues.push(format!("High memory usage: {}MB", memory_mb));
        }

        HealthCheck {
            status,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            uptime_secs: uptime.as_secs(),
            total_executions: self.total_executions,
            error_rate,
            avg_execution_ms,
            memory_mb,
            issues,
        }
    }

    /// Get production statistics
    pub fn get_statistics(&self) -> ProductionStatistics {
        let all_times: Vec<Duration> = self.execution_times.values().flatten().copied().collect();

        let (avg_ms, p50_ms, p95_ms, p99_ms, max_ms) = if !all_times.is_empty() {
            let mut sorted_times = all_times.clone();
            sorted_times.sort();

            let sum: Duration = sorted_times.iter().sum();
            let avg = (sum.as_micros() as f64 / sorted_times.len() as f64) / 1000.0;

            let p50_idx = (sorted_times.len() as f64 * 0.50) as usize;
            let p95_idx = (sorted_times.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted_times.len() as f64 * 0.99) as usize;

            let p50 = (sorted_times
                .get(p50_idx)
                .unwrap_or(&Duration::ZERO)
                .as_micros() as f64)
                / 1000.0;
            let p95 = (sorted_times
                .get(p95_idx)
                .unwrap_or(&Duration::ZERO)
                .as_micros() as f64)
                / 1000.0;
            let p99 = (sorted_times
                .get(p99_idx)
                .unwrap_or(&Duration::ZERO)
                .as_micros() as f64)
                / 1000.0;
            let max = (sorted_times.last().unwrap_or(&Duration::ZERO).as_micros() as f64) / 1000.0;

            (avg, p50, p95, p99, max)
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

        // Rules by execution count
        let mut rules_by_count: Vec<_> = self
            .execution_times
            .iter()
            .map(|(name, times)| (name.clone(), times.len()))
            .collect();
        rules_by_count.sort_by(|a, b| b.1.cmp(&a.1));

        // Rules by total time
        let mut rules_by_time: Vec<_> = self
            .execution_times
            .iter()
            .map(|(name, times)| {
                let total: Duration = times.iter().sum();
                (name.clone(), (total.as_micros() as f64) / 1000.0)
            })
            .collect();
        rules_by_time.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Slowest rules (by average time)
        let mut slowest_rules: Vec<_> = self
            .execution_times
            .iter()
            .map(|(name, times)| {
                let avg: Duration = times.iter().sum::<Duration>() / times.len() as u32;
                (name.clone(), (avg.as_micros() as f64) / 1000.0)
            })
            .collect();
        slowest_rules.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let total_errors = self.error_counts.values().sum();

        ProductionStatistics {
            total_executions: self.total_executions,
            total_errors,
            avg_execution_ms: avg_ms,
            p50_execution_ms: p50_ms,
            p95_execution_ms: p95_ms,
            p99_execution_ms: p99_ms,
            max_execution_ms: max_ms,
            rules_by_count: rules_by_count.into_iter().take(10).collect(),
            rules_by_time: rules_by_time.into_iter().take(10).collect(),
            slowest_rules: slowest_rules.into_iter().take(10).collect(),
        }
    }

    /// Get audit log
    pub fn get_audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Export audit log as JSON
    pub fn export_audit_log_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.audit_log)
    }

    /// Export statistics as JSON
    pub fn export_statistics_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.get_statistics())
    }

    /// Clear audit log
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.execution_times.clear();
        self.error_counts.clear();
        self.total_executions = 0;
        self.resource_snapshots.clear();
    }

    // Internal helper methods

    fn log_audit(
        &mut self,
        event_type: AuditEventType,
        rule_name: Option<String>,
        details: String,
        duration_ms: Option<u64>,
    ) {
        if !self.config.enable_audit {
            return;
        }

        // Enforce max audit entries
        if self.audit_log.len() >= self.config.max_audit_entries {
            self.audit_log.remove(0);
        }

        self.audit_log.push(AuditEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            event_type,
            rule_name,
            details,
            duration_ms,
        });
    }

    fn estimate_memory_usage(&self) -> usize {
        // Rough estimate based on internal structures
        let execution_times_size = self
            .execution_times
            .values()
            .map(|v| v.len() * std::mem::size_of::<Duration>())
            .sum::<usize>();

        let audit_log_size = self.audit_log.len() * 200; // Rough estimate

        let resource_snapshots_size =
            self.resource_snapshots.len() * std::mem::size_of::<ResourceSnapshot>();

        execution_times_size + audit_log_size + resource_snapshots_size + 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_monitor_creation() {
        let monitor = ProductionMonitor::new("test-engine");
        assert_eq!(monitor.engine_id, "test-engine");
        assert_eq!(monitor.total_executions, 0);
    }

    #[test]
    fn test_record_rule_execution() {
        let mut monitor = ProductionMonitor::new("test");
        monitor.record_rule_execution("rule1", Duration::from_millis(10));

        assert_eq!(monitor.total_executions, 1);
        assert!(monitor.execution_times.contains_key("rule1"));
    }

    #[test]
    fn test_record_error() {
        let mut monitor = ProductionMonitor::new("test");
        monitor.record_error("rule1", "Test error");

        assert_eq!(monitor.error_counts.get("rule1"), Some(&1));
    }

    #[test]
    fn test_health_check_healthy() {
        let monitor = ProductionMonitor::new("test");
        let health = monitor.check_health();

        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.total_executions, 0);
        assert!(health.issues.is_empty());
    }

    #[test]
    fn test_health_check_degraded() {
        let mut monitor = ProductionMonitor::new("test");

        // Simulate many executions with some errors
        for _ in 0..100 {
            monitor.record_rule_execution("rule1", Duration::from_millis(5));
        }
        for _ in 0..7 {
            monitor.record_error("rule1", "Error");
        }

        let health = monitor.check_health();
        assert_eq!(health.status, HealthStatus::Degraded);
        assert!(health.error_rate > 5.0);
    }

    #[test]
    fn test_statistics() {
        let mut monitor = ProductionMonitor::new("test");

        monitor.record_rule_execution("rule1", Duration::from_millis(10));
        monitor.record_rule_execution("rule1", Duration::from_millis(20));
        monitor.record_rule_execution("rule2", Duration::from_millis(30));

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_executions, 3);
        assert!(stats.avg_execution_ms > 0.0);
    }

    #[test]
    fn test_audit_log() {
        let mut monitor = ProductionMonitor::new("test");
        monitor.record_rule_execution("rule1", Duration::from_millis(10));

        let audit = monitor.get_audit_log();
        assert!(!audit.is_empty());

        // Should have engine start + execution
        assert!(audit
            .iter()
            .any(|e| e.event_type == AuditEventType::EngineStarted));
    }

    #[test]
    fn test_export_json() {
        let mut monitor = ProductionMonitor::new("test");
        monitor.record_rule_execution("rule1", Duration::from_millis(10));

        let stats_json = monitor.export_statistics_json();
        assert!(stats_json.is_ok());

        let audit_json = monitor.export_audit_log_json();
        assert!(audit_json.is_ok());
    }

    #[test]
    fn test_resource_snapshot() {
        let mut monitor = ProductionMonitor::new("test");
        monitor.snapshot_resources(5);

        assert_eq!(monitor.resource_snapshots.len(), 1);
        assert_eq!(monitor.resource_snapshots[0].active_rules, 5);
    }

    #[test]
    fn test_reset_statistics() {
        let mut monitor = ProductionMonitor::new("test");
        monitor.record_rule_execution("rule1", Duration::from_millis(10));
        monitor.record_error("rule1", "error");

        monitor.reset_statistics();

        assert_eq!(monitor.total_executions, 0);
        assert!(monitor.execution_times.is_empty());
        assert!(monitor.error_counts.is_empty());
    }
}
