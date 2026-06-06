//! Advanced diagnostic tools for TDB store.
//!
//! This module provides comprehensive diagnostic capabilities:
//! - System health checks
//! - Performance analysis
//! - Storage diagnostics
//! - Index consistency checks
//! - Memory usage analysis
//! - WAL integrity verification
//! - Dictionary consistency checks
//! - B+Tree structure validation
//! - Corruption detection and repair
//!
//! Implementation is split across sibling modules:
//! - `diagnostics_types`: data types (level, severity, results, reports, summaries,
//!   repair recommendations)
//! - `diagnostics_collectors`: [`DiagnosticEngine`], [`DiagnosticContext`], the
//!   [`DiagnosticCheck`] trait, and all built-in checks
//! - `diagnostics_tests`: unit tests for the public API
//!
//! [`DiagnosticEngine`]: crate::diagnostics_collectors::DiagnosticEngine
//! [`DiagnosticContext`]: crate::diagnostics_collectors::DiagnosticContext
//! [`DiagnosticCheck`]: crate::diagnostics_collectors::DiagnosticCheck

pub use crate::diagnostics_collectors::{DiagnosticCheck, DiagnosticContext, DiagnosticEngine};
pub use crate::diagnostics_types::{
    DiagnosticLevel, DiagnosticReport, DiagnosticResult, DiagnosticSummary, HealthStatus,
    RepairAction, RepairRecommendation, Severity,
};
