//! Configuration for federation monitoring

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for federation monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationMonitorConfig {
    pub enable_detailed_metrics: bool,
    pub metrics_retention_hours: u64,
    pub max_recent_queries: usize,
    pub max_recent_events: usize,
    pub enable_prometheus_export: bool,
    pub health_check_interval: Duration,
    pub max_trace_spans: usize,
}

impl Default for FederationMonitorConfig {
    fn default() -> Self {
        Self {
            enable_detailed_metrics: true,
            metrics_retention_hours: 24,
            max_recent_queries: 1000,
            max_recent_events: 500,
            enable_prometheus_export: true,
            health_check_interval: Duration::from_secs(30),
            max_trace_spans: 10000,
        }
    }
}
