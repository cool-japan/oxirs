//! Comprehensive System Monitoring for SHACL-AI
//!
//! This module provides real-time monitoring capabilities for:
//! - Performance metrics and trends
//! - Quality assessment and degradation detection
//! - Error tracking and analysis
//! - Resource utilization monitoring
//! - Health checks and alerting
//!
//! The implementation is split across focused sibling modules; this module is a
//! thin facade that re-exports their public API so existing call sites keep
//! working through `crate::system_monitoring::*`:
//!
//! - [`crate::system_monitoring_types`] — shared configuration, metrics, alert,
//!   dashboard, health-check and notification primitives.
//! - [`crate::system_monitoring_monitor`] — the [`SystemMonitor`] facade plus
//!   the implementation stubs for its collaborating components.
//! - [`crate::system_monitoring_anomaly`] — the [`AnomalyDetector`] engine.
//! - [`crate::system_monitoring_alerting`] — alerting-side re-export grouping.
//! - [`crate::system_monitoring_collector`] — collector-side re-export grouping.

pub use crate::system_monitoring_alerting::*;
pub use crate::system_monitoring_anomaly::*;
pub use crate::system_monitoring_collector::*;
pub use crate::system_monitoring_monitor::*;
pub use crate::system_monitoring_types::*;
