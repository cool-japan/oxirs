//! Advanced Benchmarking Framework for Vector Search Systems
//!
//! This module provides comprehensive benchmarking capabilities including:
//! - ANN-Benchmarks integration and compatibility
//! - Multi-dimensional performance analysis
//! - Quality metrics (recall, precision, NDCG)
//! - Scalability and throughput testing
//! - Statistical significance testing
//! - Memory and latency profiling
//! - Automated hyperparameter tuning
//! - Comparative analysis across algorithms
//!
//! # Module layout
//!
//! - [`crate::bench_metrics`]: All data types, metric structs, and configuration.
//! - [`crate::bench_runner`]: The benchmark suite runner, statistical analyzer,
//!   performance profiler, and hyperparameter tuner.
//! - [`crate::bench_tests`]: Unit tests.

pub use crate::bench_metrics::*;
pub use crate::bench_runner::*;
