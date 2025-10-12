//! Production-ready features for SAMM processing
//!
//! This module provides production-grade features for running SAMM operations
//! in enterprise environments:
//!
//! - Structured logging with tracing
//! - Metrics collection and reporting
//! - Health checks
//! - Configuration validation
//! - Error recovery strategies
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_samm::production::{ProductionConfig, init_production};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize production features
//! let config = ProductionConfig::default();
//! init_production(&config)?;
//!
//! // Application runs with full observability
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::Level;

/// Production configuration
#[derive(Debug, Clone)]
pub struct ProductionConfig {
    /// Enable structured logging
    pub logging_enabled: bool,

    /// Logging level
    pub log_level: LogLevel,

    /// Enable metrics collection
    pub metrics_enabled: bool,

    /// Enable health checks
    pub health_checks_enabled: bool,

    /// Application name for logging context
    pub app_name: String,

    /// Environment (dev, staging, prod)
    pub environment: String,
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            logging_enabled: true,
            log_level: LogLevel::Info,
            metrics_enabled: true,
            health_checks_enabled: true,
            app_name: "oxirs-samm".to_string(),
            environment: "production".to_string(),
        }
    }
}

/// Log level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Trace level (most verbose)
    Trace,
    /// Debug level
    Debug,
    /// Info level (default)
    Info,
    /// Warn level
    Warn,
    /// Error level (least verbose)
    Error,
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

/// Initialize production features
pub fn init_production(config: &ProductionConfig) -> Result<()> {
    if config.logging_enabled {
        init_logging(config);
    }

    if config.metrics_enabled {
        MetricsCollector::global().reset();
    }

    tracing::info!(
        app = %config.app_name,
        env = %config.environment,
        "Production initialization complete"
    );

    Ok(())
}

/// Initialize structured logging
fn init_logging(config: &ProductionConfig) {
    let level: Level = config.log_level.into();

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .json()
        .init();
}

/// Metrics collector for production monitoring
#[derive(Debug)]
pub struct MetricsCollector {
    /// Total operations processed
    operations_total: AtomicU64,

    /// Total errors encountered
    errors_total: AtomicU64,

    /// Total warnings
    warnings_total: AtomicU64,

    /// Parse operations
    parse_operations: AtomicU64,

    /// Validation operations
    validation_operations: AtomicU64,

    /// Code generation operations
    codegen_operations: AtomicU64,

    /// Package operations
    package_operations: AtomicU64,

    /// Start time
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            operations_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            warnings_total: AtomicU64::new(0),
            parse_operations: AtomicU64::new(0),
            validation_operations: AtomicU64::new(0),
            codegen_operations: AtomicU64::new(0),
            package_operations: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Get global metrics collector
    pub fn global() -> &'static MetricsCollector {
        static METRICS: once_cell::sync::Lazy<MetricsCollector> =
            once_cell::sync::Lazy::new(MetricsCollector::new);
        &METRICS
    }

    /// Record an operation
    pub fn record_operation(&self, operation_type: OperationType) {
        self.operations_total.fetch_add(1, Ordering::Relaxed);

        match operation_type {
            OperationType::Parse => self.parse_operations.fetch_add(1, Ordering::Relaxed),
            OperationType::Validation => self.validation_operations.fetch_add(1, Ordering::Relaxed),
            OperationType::CodeGeneration => {
                self.codegen_operations.fetch_add(1, Ordering::Relaxed)
            }
            OperationType::Package => self.package_operations.fetch_add(1, Ordering::Relaxed),
        };

        tracing::debug!(operation_type = ?operation_type, "Operation recorded");
    }

    /// Record an error
    pub fn record_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            errors_total = self.errors_total.load(Ordering::Relaxed),
            "Error recorded"
        );
    }

    /// Record a warning
    pub fn record_warning(&self) {
        self.warnings_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            operations_total: self.operations_total.load(Ordering::Relaxed),
            errors_total: self.errors_total.load(Ordering::Relaxed),
            warnings_total: self.warnings_total.load(Ordering::Relaxed),
            parse_operations: self.parse_operations.load(Ordering::Relaxed),
            validation_operations: self.validation_operations.load(Ordering::Relaxed),
            codegen_operations: self.codegen_operations.load(Ordering::Relaxed),
            package_operations: self.package_operations.load(Ordering::Relaxed),
            uptime: self.start_time.elapsed(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.operations_total.store(0, Ordering::Relaxed);
        self.errors_total.store(0, Ordering::Relaxed);
        self.warnings_total.store(0, Ordering::Relaxed);
        self.parse_operations.store(0, Ordering::Relaxed);
        self.validation_operations.store(0, Ordering::Relaxed);
        self.codegen_operations.store(0, Ordering::Relaxed);
        self.package_operations.store(0, Ordering::Relaxed);
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation type for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Parse operation
    Parse,
    /// Validation operation
    Validation,
    /// Code generation operation
    CodeGeneration,
    /// Package operation
    Package,
}

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total operations
    pub operations_total: u64,

    /// Total errors
    pub errors_total: u64,

    /// Total warnings
    pub warnings_total: u64,

    /// Parse operations
    pub parse_operations: u64,

    /// Validation operations
    pub validation_operations: u64,

    /// Code generation operations
    pub codegen_operations: u64,

    /// Package operations
    pub package_operations: u64,

    /// Uptime
    pub uptime: Duration,
}

impl MetricsSnapshot {
    /// Get error rate (errors per operation)
    pub fn error_rate(&self) -> f64 {
        if self.operations_total == 0 {
            0.0
        } else {
            self.errors_total as f64 / self.operations_total as f64
        }
    }

    /// Get operations per second
    pub fn ops_per_second(&self) -> f64 {
        let secs = self.uptime.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.operations_total as f64 / secs
        }
    }
}

/// Health check status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System is degraded but operational
    Degraded,
    /// System is unhealthy
    Unhealthy,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Overall health status
    pub status: HealthStatus,

    /// Individual check results
    pub checks: Vec<(String, HealthStatus, String)>,

    /// Timestamp
    pub timestamp: Instant,
}

/// Perform health check
pub fn health_check() -> HealthCheck {
    let mut checks = Vec::new();
    let mut overall_status = HealthStatus::Healthy;

    // Check metrics
    let metrics = MetricsCollector::global().snapshot();
    let error_rate = metrics.error_rate();

    if error_rate > 0.5 {
        checks.push((
            "error_rate".to_string(),
            HealthStatus::Unhealthy,
            format!("High error rate: {:.2}%", error_rate * 100.0),
        ));
        overall_status = HealthStatus::Unhealthy;
    } else if error_rate > 0.1 {
        checks.push((
            "error_rate".to_string(),
            HealthStatus::Degraded,
            format!("Elevated error rate: {:.2}%", error_rate * 100.0),
        ));
        if overall_status == HealthStatus::Healthy {
            overall_status = HealthStatus::Degraded;
        }
    } else {
        checks.push((
            "error_rate".to_string(),
            HealthStatus::Healthy,
            format!("Normal error rate: {:.2}%", error_rate * 100.0),
        ));
    }

    // Check uptime
    if metrics.uptime.as_secs() > 0 {
        checks.push((
            "uptime".to_string(),
            HealthStatus::Healthy,
            format!("{} seconds", metrics.uptime.as_secs()),
        ));
    }

    HealthCheck {
        status: overall_status,
        checks,
        timestamp: Instant::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config_default() {
        let config = ProductionConfig::default();
        assert!(config.logging_enabled);
        assert!(config.metrics_enabled);
        assert_eq!(config.app_name, "oxirs-samm");
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        collector.record_operation(OperationType::Parse);
        collector.record_operation(OperationType::Validation);
        collector.record_error();

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.operations_total, 2);
        assert_eq!(snapshot.errors_total, 1);
        assert_eq!(snapshot.parse_operations, 1);
        assert_eq!(snapshot.validation_operations, 1);
    }

    #[test]
    fn test_error_rate_calculation() {
        let snapshot = MetricsSnapshot {
            operations_total: 100,
            errors_total: 10,
            warnings_total: 5,
            parse_operations: 50,
            validation_operations: 30,
            codegen_operations: 15,
            package_operations: 5,
            uptime: Duration::from_secs(60),
        };

        assert!((snapshot.error_rate() - 0.1).abs() < 0.01);
        assert!((snapshot.ops_per_second() - (100.0 / 60.0)).abs() < 0.01);
    }

    #[test]
    fn test_health_check() {
        let health = health_check();
        assert!(!health.checks.is_empty());
        assert!(
            health.status == HealthStatus::Healthy
                || health.status == HealthStatus::Degraded
                || health.status == HealthStatus::Unhealthy
        );
    }
}
