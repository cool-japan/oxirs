//! # Cluster Metrics System
//!
//! Comprehensive performance metrics collection, benchmarking, and regression detection
//! for OxiRS Cluster operations using SciRS2-Core.
//!
//! ## Features
//! - Enhanced histogram metrics with percentile calculations
//! - Gauge metrics for real-time resource monitoring
//! - Timer metrics for critical operation paths
//! - Counter metrics with rate calculation
//! - Rolling window metrics for trend analysis
//! - Exponential decay metrics for recency-weighted statistics
//! - Comprehensive benchmarking suite
//! - Advanced performance regression detection with statistical tests
//!
//! ## Phase 2 v0.2.0 Implementation
//!
//! The implementation is split across sibling modules:
//! - `cluster_metrics_stats`: [`ClusterOperation`] taxonomy and [`EnhancedLatencyStats`]
//! - `cluster_metrics_types`: [`OperationTimer`], result/record types, and
//!   regression-detection types
//! - `cluster_metrics_manager`: the [`ClusterMetricsManager`] implementation
//! - `cluster_metrics_tests`: unit tests for the public API
//!
//! [`ClusterOperation`]: crate::cluster_metrics_stats::ClusterOperation
//! [`EnhancedLatencyStats`]: crate::cluster_metrics_stats::EnhancedLatencyStats
//! [`OperationTimer`]: crate::cluster_metrics_types::OperationTimer
//! [`ClusterMetricsManager`]: crate::cluster_metrics_manager::ClusterMetricsManager

pub use crate::cluster_metrics_manager::ClusterMetricsManager;
pub use crate::cluster_metrics_stats::{ClusterOperation, EnhancedLatencyStats};
pub use crate::cluster_metrics_types::{
    BenchmarkComparison, BenchmarkResultRecord, OperationBaseline, OperationMetrics,
    OperationTimer, PerformanceRegression, RegressionSeverity,
};
