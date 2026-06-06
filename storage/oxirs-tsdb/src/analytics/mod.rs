//! Advanced analytics for time-series data stored in OxiRS TSDB.
//!
//! ## Modules
//!
//! - `anomaly` -- Anomaly detection (Z-score, IQR, EWMA, Isolation Forest)
//! - `forecasting` -- Time-series forecasting (Naive, SES, Holt, Holt-Winters)
//! - `arrow_export` -- Apache Arrow / Parquet export and DuckDB SQL generation
//! - `sql_export` -- DuckDB-compatible SQL DDL/DML generation and columnar export
//! - `kalman` -- Kalman filter smoothing, prediction, and anomaly detection
//! - `gpu_aggregations` -- GPU-accelerated time-series aggregation simulator
//! - `arrow_ipc` -- Pure-Rust Apache Arrow IPC stream format export
//! - `parquet_export` -- Pure-Rust simplified Parquet-compatible columnar export

pub mod anomaly;
pub mod arrow_export;
pub(crate) mod arrow_export_builder;
#[cfg(test)]
mod arrow_export_tests;
pub mod arrow_export_types;
pub mod arrow_ipc;
pub mod forecasting;
pub mod gpu_aggregations;
pub mod kalman;
pub mod kalman_forecasting;
pub mod parquet_export;
pub mod rollup_engine;
pub mod sql_export;

pub use arrow_export::{
    AggregationFunction, ArrowExporter, ColumnarExport, ColumnarStats, DuckDbQueryAdapter,
    ExportedPoint, ParquetCompression, ParquetExporter,
};

pub use sql_export::{DataValueType, MetricSchema, MetricSchemaBuilder, SqlDataPoint, SqlExporter};

// Kalman forecasting re-exports
pub use kalman_forecasting::{evaluate_kalman, KalmanForecaster, KalmanHoltWinters, KalmanState};

// Kalman filter re-exports
pub use kalman::{AdaptiveKalmanFilter, AnomalyEvent, KalmanAnomaly, KalmanFilter};

// GPU aggregation re-exports — OO API
pub use gpu_aggregations::{GpuAggMetrics, GpuAggOp, GpuDownsampler, GpuTimeSeriesAggregator};
// GPU aggregation re-exports — free-function columnar API
pub use gpu_aggregations::{
    avg_column, count_column, gpu_sum, max_column, min_column, rolling_avg, rolling_sum,
    sum_column, GpuAggError,
};

// Arrow IPC re-exports
pub use arrow_ipc::{
    ArrowColumn, ArrowDataType, ArrowField, ArrowIpcReader, ArrowIpcWriter, ArrowRecordBatch,
    ArrowSchema, TaggedDataPoint, TimeUnit,
};
// Parquet export re-exports (use qualified name to avoid collision with arrow_export::ParquetCompression)
pub use parquet_export::{
    ParquetColumn, ParquetCompression as ParquetIpcCompression, ParquetReader, ParquetValues,
    ParquetWriter,
};

// Rollup engine re-exports
pub use rollup_engine::{
    RawDataPoint, RollupConfig, RollupDataPoint, RollupEngine, RollupStats, RollupTier, TierConfig,
};
