//! Real-time analytics and monitoring for vector search operations
//!
//! This module provides comprehensive monitoring, analytics, and performance insights
//! for vector search systems including dashboards, alerts, and benchmarking.
//!
//! # Module layout
//!
//! - [`crate::rta_engine`]: The `VectorAnalyticsEngine`, core event types, alert types,
//!   `AnalyticsReport`, and `SystemInfo`.
//! - [`crate::rta_aggregators`]: Metrics collectors, performance monitors, query analyzers,
//!   alert managers, notification channels, dashboard data, and profiler.
//! - [`crate::rta_tests`]: Unit tests.

pub use crate::rta_aggregators::*;
pub use crate::rta_engine::*;
