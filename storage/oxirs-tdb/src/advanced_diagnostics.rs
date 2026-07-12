//! Advanced diagnostic tools for TDB store monitoring and analysis
//!
//! This module provides sophisticated diagnostic capabilities beyond basic health checks:
//! - Query performance analysis and optimization recommendations
//! - Transaction pattern analysis and contention detection
//! - Storage fragmentation analysis and compaction recommendations
//! - Index usage statistics and optimization suggestions
//! - Predictive health monitoring with trend analysis
//! - Auto-tuning recommendations for configuration optimization
//! - Anomaly detection using statistical methods
//! - Capacity planning and forecasting
//!
//! The implementation is split across sibling modules:
//! - `advanced_diagnostics_types`: report/analysis data types plus the
//!   [`AdvancedDiagnosticEngine`] struct and its internal trackers
//! - `advanced_diagnostics_engine`: the [`AdvancedDiagnosticEngine`] implementation
//! - `advanced_diagnostics_tests`: unit tests for the public API
//!
//! [`AdvancedDiagnosticEngine`]: crate::advanced_diagnostics_types::AdvancedDiagnosticEngine

pub use crate::advanced_diagnostics_types::*;
