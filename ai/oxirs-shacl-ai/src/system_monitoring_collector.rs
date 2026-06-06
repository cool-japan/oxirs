//! Metrics collection for the system monitoring subsystem.
//!
//! Provides the [`MetricsCollector`] implementation: CPU/memory/disk/network
//! sampling, process-level metrics, cgroup-aware collection, and async polling
//! stubs.  All collector-side types live in
//! [`crate::system_monitoring_types`].

// Re-export the collector type and its metric types so callers can import from
// this module instead of reaching into the types module directly.
pub use crate::system_monitoring_types::{
    CustomMetric, ErrorMetric, MetricsCollector, PerformanceMetric, QualityMetric, SystemMetric,
};
