//! Health check logic, alerting, and threshold monitoring for embedding service.
//!
//! This module implements liveness/readiness/comprehensive health checks
//! and alert handler infrastructure.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::monitoring_metrics::{MetricsCollector, PerformanceMetrics};

// ====================================================================================
// ALERT INFRASTRUCTURE
// ====================================================================================

/// Alert handling trait
pub trait AlertHandler {
    fn handle_alert(&self, alert: Alert) -> Result<()>;
}

/// Alert types
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    HighLatency,
    LowThroughput,
    HighErrorRate,
    LowCacheHitRate,
    QualityDrift,
    PerformanceDrift,
    ResourceExhaustion,
    SystemFailure,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum acceptable P95 latency (ms)
    pub max_p95_latency_ms: f64,
    /// Minimum acceptable throughput (req/s)
    pub min_throughput_rps: f64,
    /// Maximum acceptable error rate
    pub max_error_rate: f64,
    /// Minimum acceptable cache hit rate
    pub min_cache_hit_rate: f64,
    /// Maximum acceptable quality drift
    pub max_quality_drift: f64,
    /// Maximum acceptable memory usage (MB)
    pub max_memory_usage_mb: f64,
    /// Maximum acceptable GPU memory usage (MB)
    pub max_gpu_memory_mb: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_p95_latency_ms: 500.0,
            min_throughput_rps: 100.0,
            max_error_rate: 0.05,    // 5%
            min_cache_hit_rate: 0.8, // 80%
            max_quality_drift: 0.1,
            max_memory_usage_mb: 4096.0, // 4GB
            max_gpu_memory_mb: 8192.0,   // 8GB
        }
    }
}

/// Console alert handler implementation
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&self, alert: Alert) -> Result<()> {
        println!(
            "ALERT [{}]: {} - {}",
            format!("{:?}", alert.severity).to_uppercase(),
            alert.message,
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
        Ok(())
    }
}

/// Slack alert handler (placeholder)
pub struct SlackAlertHandler {
    pub webhook_url: String,
}

impl AlertHandler for SlackAlertHandler {
    fn handle_alert(&self, alert: Alert) -> Result<()> {
        // In production, this would send to Slack
        tracing::info!(
            "Would send Slack alert to {}: {}",
            self.webhook_url,
            alert.message
        );
        Ok(())
    }
}

// ====================================================================================
// HEALTH CHECK FUNCTIONALITY
// ====================================================================================

/// Health status for the embedding service
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    /// Service is healthy and operational
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Overall health status
    pub status: HealthStatus,
    /// Timestamp of health check
    pub timestamp: DateTime<Utc>,
    /// Individual component health
    pub components: HashMap<String, ComponentHealth>,
    /// Additional details
    pub details: HashMap<String, String>,
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: HealthStatus,
    /// Component message
    pub message: String,
    /// Last check time
    pub last_check: DateTime<Utc>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Health checker for embedding service
pub struct HealthChecker {
    /// Model load status
    models_loaded: Arc<RwLock<bool>>,
    /// Last successful request time
    last_request_time: Arc<RwLock<DateTime<Utc>>>,
    /// Error rate threshold
    error_rate_threshold: f64,
    /// Latency threshold (ms)
    latency_threshold_ms: f64,
    /// Memory threshold (MB)
    memory_threshold_mb: f64,
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self {
            models_loaded: Arc::new(RwLock::new(false)),
            last_request_time: Arc::new(RwLock::new(Utc::now())),
            error_rate_threshold: 0.1,    // 10%
            latency_threshold_ms: 1000.0, // 1 second
            memory_threshold_mb: 8192.0,  // 8GB
            metrics,
        }
    }

    /// Set models loaded status
    pub fn set_models_loaded(&self, loaded: bool) -> Result<()> {
        let mut status = self
            .models_loaded
            .write()
            .map_err(|e| anyhow!("Failed to write lock: {}", e))?;
        *status = loaded;
        Ok(())
    }

    /// Update last request time
    pub fn update_last_request_time(&self) -> Result<()> {
        let mut time = self
            .last_request_time
            .write()
            .map_err(|e| anyhow!("Failed to write lock: {}", e))?;
        *time = Utc::now();
        Ok(())
    }

    /// Perform liveness check (basic service availability)
    pub fn check_liveness(&self) -> HealthCheckResult {
        let mut components = HashMap::new();

        // Check if service is running (always healthy if we can respond)
        components.insert(
            "service".to_string(),
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: "Service is running".to_string(),
                last_check: Utc::now(),
                metrics: HashMap::new(),
            },
        );

        HealthCheckResult {
            status: HealthStatus::Healthy,
            timestamp: Utc::now(),
            components,
            details: HashMap::new(),
        }
    }

    /// Perform readiness check (service ready to handle requests)
    pub fn check_readiness(&self) -> HealthCheckResult {
        let mut components = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;

        // Check if models are loaded
        let models_loaded = self.models_loaded.read().map(|g| *g).unwrap_or(false);
        if !models_loaded {
            overall_status = HealthStatus::Unhealthy;
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Unhealthy,
                    message: "Models not loaded".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        } else {
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Healthy,
                    message: "Models loaded and ready".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        }

        // Check cache availability
        let cache_hit_rate = self.metrics.get_cache_hit_rate();
        components.insert(
            "cache".to_string(),
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: format!("Cache hit rate: {:.2}%", cache_hit_rate * 100.0),
                last_check: Utc::now(),
                metrics: [("hit_rate".to_string(), cache_hit_rate)]
                    .into_iter()
                    .collect(),
            },
        );

        HealthCheckResult {
            status: overall_status,
            timestamp: Utc::now(),
            components,
            details: HashMap::new(),
        }
    }

    /// Perform comprehensive health check
    pub fn check_health(&self, performance_metrics: &PerformanceMetrics) -> HealthCheckResult {
        let mut components = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;

        // Check models
        let models_loaded = self.models_loaded.read().map(|g| *g).unwrap_or(false);
        if !models_loaded {
            overall_status = HealthStatus::Unhealthy;
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Unhealthy,
                    message: "Models not loaded".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        } else {
            components.insert(
                "models".to_string(),
                ComponentHealth {
                    status: HealthStatus::Healthy,
                    message: "Models operational".to_string(),
                    last_check: Utc::now(),
                    metrics: HashMap::new(),
                },
            );
        }

        // Check latency
        let latency_status =
            if performance_metrics.latency.p95_latency_ms > self.latency_threshold_ms {
                if overall_status == HealthStatus::Healthy {
                    overall_status = HealthStatus::Degraded;
                }
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };

        components.insert(
            "latency".to_string(),
            ComponentHealth {
                status: latency_status,
                message: format!(
                    "P95 latency: {:.2}ms",
                    performance_metrics.latency.p95_latency_ms
                ),
                last_check: Utc::now(),
                metrics: [
                    (
                        "p50".to_string(),
                        performance_metrics.latency.p50_latency_ms,
                    ),
                    (
                        "p95".to_string(),
                        performance_metrics.latency.p95_latency_ms,
                    ),
                    (
                        "p99".to_string(),
                        performance_metrics.latency.p99_latency_ms,
                    ),
                ]
                .into_iter()
                .collect(),
            },
        );

        // Check error rate
        let error_rate = if performance_metrics.throughput.total_requests > 0 {
            performance_metrics.errors.total_errors as f64
                / performance_metrics.throughput.total_requests as f64
        } else {
            0.0
        };

        let error_status = if error_rate > self.error_rate_threshold {
            if overall_status == HealthStatus::Healthy {
                overall_status = HealthStatus::Degraded;
            }
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        components.insert(
            "errors".to_string(),
            ComponentHealth {
                status: error_status,
                message: format!("Error rate: {:.2}%", error_rate * 100.0),
                last_check: Utc::now(),
                metrics: [("error_rate".to_string(), error_rate)]
                    .into_iter()
                    .collect(),
            },
        );

        // Check memory
        let memory_status =
            if performance_metrics.resources.memory_usage_mb > self.memory_threshold_mb {
                if overall_status == HealthStatus::Healthy {
                    overall_status = HealthStatus::Degraded;
                }
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };

        components.insert(
            "memory".to_string(),
            ComponentHealth {
                status: memory_status,
                message: format!(
                    "Memory usage: {:.2}MB / {:.2}MB",
                    performance_metrics.resources.memory_usage_mb, self.memory_threshold_mb
                ),
                last_check: Utc::now(),
                metrics: [
                    (
                        "usage_mb".to_string(),
                        performance_metrics.resources.memory_usage_mb,
                    ),
                    ("threshold_mb".to_string(), self.memory_threshold_mb),
                ]
                .into_iter()
                .collect(),
            },
        );

        // Check cache
        let cache_hit_rate = self.metrics.get_cache_hit_rate();
        components.insert(
            "cache".to_string(),
            ComponentHealth {
                status: HealthStatus::Healthy,
                message: format!("Cache hit rate: {:.2}%", cache_hit_rate * 100.0),
                last_check: Utc::now(),
                metrics: [("hit_rate".to_string(), cache_hit_rate)]
                    .into_iter()
                    .collect(),
            },
        );

        HealthCheckResult {
            status: overall_status,
            timestamp: Utc::now(),
            components,
            details: HashMap::new(),
        }
    }

    /// Get metrics endpoint (Prometheus format)
    pub fn get_metrics_endpoint(&self) -> Result<String> {
        self.metrics.export_prometheus()
    }
}
