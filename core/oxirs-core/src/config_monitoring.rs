//! Monitoring, metrics, tracing, logging, and health-check configuration.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub metrics: MetricsConfig,
    pub tracing: TracingConfig,
    pub logging: LoggingConfig,
    pub health_checks: HealthCheckConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub exporter: MetricsExporter,
    pub collection_interval: Duration,
    pub retention_period: Duration,
    pub custom_metrics: Vec<CustomMetric>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricsExporter {
    None,
    Prometheus,
    StatsD,
    OpenMetrics,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub name: String,
    pub metric_type: MetricType,
    pub description: String,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub enabled: bool,
    pub exporter: TraceExporter,
    pub sampling_rate: f64,
    pub max_spans: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceExporter {
    None,
    Jaeger,
    Zipkin,
    OpenTelemetry,
    Console,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub format: LogFormat,
    pub targets: Vec<LogTarget>,
    pub structured: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    Plain,
    Json,
    Logfmt,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogTarget {
    Console,
    File {
        path: PathBuf,
        rotation: FileRotation,
    },
    Syslog {
        facility: String,
    },
    Network {
        endpoint: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRotation {
    pub max_size: usize,
    pub max_age: Duration,
    pub max_files: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub enabled: bool,
    pub interval: Duration,
    pub timeout: Duration,
    pub endpoints: Vec<HealthCheckEndpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckEndpoint {
    pub name: String,
    pub path: String,
    pub check_type: HealthCheckType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthCheckType {
    Liveness,
    Readiness,
    Startup,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: MetricsConfig::default(),
            tracing: TracingConfig::default(),
            logging: LoggingConfig::default(),
            health_checks: HealthCheckConfig::default(),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            exporter: MetricsExporter::Prometheus,
            collection_interval: Duration::from_secs(15),
            retention_period: Duration::from_secs(3600),
            custom_metrics: Vec::new(),
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exporter: TraceExporter::Jaeger,
            sampling_rate: 0.1,
            max_spans: 1000,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Json,
            targets: vec![LogTarget::Console],
            structured: true,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            endpoints: Vec::new(),
        }
    }
}
