//! # Comprehensive Performance Analysis Engine (Facade)
//!
//! Re-exports the public API of the performance analyzer subsystem from its
//! sibling modules:
//!
//! - [`performance_analyzer_types`](crate::performance_analyzer_types) —
//!   configuration, metrics, bottleneck, recommendation, trend, alert, and
//!   ML query-plan optimizer data types
//! - [`performance_analyzer_collector`](crate::performance_analyzer_collector) —
//!   the [`PerformanceAnalyzer`] engine for collecting metrics, detecting
//!   bottlenecks, computing trends, and raising alerts
//! - [`performance_analyzer_reporter`](crate::performance_analyzer_reporter) —
//!   recommendation reporting plus the ML-driven [`QueryPlanOptimizer`]

pub use crate::performance_analyzer_collector::*;
pub use crate::performance_analyzer_reporter::*;
pub use crate::performance_analyzer_types::*;
