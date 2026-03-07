//! Prometheus-compatible metrics for Modbus operations
//!
//! Provides lightweight, thread-safe counters and gauges that expose
//! Modbus operational statistics in the Prometheus text exposition format.
//! This enables integration with Prometheus / Grafana without requiring
//! external metric libraries.

pub mod prometheus;

pub use prometheus::{MetricsSnapshot, ModbusMetrics, PrometheusExporter};
