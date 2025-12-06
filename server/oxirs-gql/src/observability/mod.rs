//! Observability & Monitoring Module
//!
//! This module provides comprehensive observability features for GraphQL operations,
//! including custom metrics, distributed tracing, logging, and profiling.

pub mod cardinality_optimizer;
pub mod core;
pub mod custom_metrics;
pub mod error_aggregation;
pub mod query_debugger;
pub mod query_replay;
pub mod structured_logging;

pub use cardinality_optimizer::{
    CardinalityConfig, CardinalityOptimizer, LabelCardinality, LabelFilter, MetricCardinality,
    NormalizationStrategy,
};
// Re-export core observability types
pub use core::*;
pub use custom_metrics::{
    AggregationStrategy, ComputedMetric, CustomMetricsRegistry, MetricDataPoint, MetricFilter,
    MetricMetadata, MetricType, MetricValue,
};
pub use error_aggregation::{
    ErrorAggregator, ErrorCategory, ErrorGroup, ErrorOccurrence, ErrorSeverity, ErrorStatistics,
};
pub use query_debugger::{
    ExecutionPlan, ExecutionStep, ExecutionSummary, QueryDebugger, StepStatus, StepType,
};
pub use query_replay::{
    QueryFilter, QueryReplay, RecordedQuery, ReplayConfig, ReplayResult, ReplaySummary,
};
pub use structured_logging::{
    LogEntry, LogEntryBuilder, LogLevel, LoggerConfig, PerformanceMetrics, QueryContext,
    RequestContext, StructuredLogger,
};
