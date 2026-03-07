//! # Stream Metrics
//!
//! Real-time metrics collection for stream processing pipelines.

pub mod stream_metrics;

pub use stream_metrics::{StreamLatencyHistogram, StreamMetrics, StreamMetricsCollector};
