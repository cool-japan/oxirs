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
pub mod arrow_ipc;
pub mod forecasting;
pub mod gpu_aggregations;
pub mod kalman;
pub mod parquet_export;
pub mod rollup_engine;
pub mod sql_export;

pub use arrow_export::{
    AggregationFunction, ArrowExporter, ColumnarExport, ColumnarStats, DuckDbQueryAdapter,
    ExportedPoint, ParquetCompression, ParquetExporter,
};

pub use sql_export::{DataValueType, MetricSchema, MetricSchemaBuilder, SqlDataPoint, SqlExporter};

// Kalman filter re-exports
pub use kalman::{AdaptiveKalmanFilter, AnomalyEvent, KalmanAnomaly, KalmanFilter};

// GPU aggregation re-exports
pub use gpu_aggregations::{GpuAggMetrics, GpuAggOp, GpuDownsampler, GpuTimeSeriesAggregator};

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
