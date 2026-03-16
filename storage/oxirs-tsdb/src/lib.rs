//! Time-series optimizations for OxiRS
//!
//! **Status**: Production Ready (v0.2.2)
//!
//! This crate provides high-performance time-series storage and query
//! capabilities for IoT-scale RDF data.
//!
//! # Features
//!
//! - Gorilla compression - 40:1 storage reduction (Facebook, VLDB 2015)
//! - Delta-of-delta timestamps - <2 bits per timestamp
//! - SPARQL temporal extensions - WINDOW, RESAMPLE, INTERPOLATE
//! - 500K+ writes/sec - High-throughput ingestion
//! - Hybrid storage - Seamless RDF + Time-Series integration
//! - Retention policies - Automatic downsampling and expiration
//! - Write-Ahead Log - Crash recovery and durability with CRC32 protection
//! - Background compaction - Automatic storage optimization
//! - Columnar storage - Disk-backed binary format
//! - Series indexing - Efficient chunk lookups
//! - Raft replication - Distributed consensus with quorum commits
//! - Arrow/Parquet export - Analytics interoperability
//!
//! # Architecture
//!
//! ```text
//! +---------------------------------------------+
//! |         Hybrid Storage Model                |
//! +---------------------------------------------+
//! |                                             |
//! |  +--------------+    +-----------------+    |
//! |  |  RDF Store   |<-->| Time-Series DB  |    |
//! |  |  (oxirs-tdb) |    |  (this crate)   |    |
//! |  +--------------+    +-----------------+    |
//! |        |                     |              |
//! |        | Semantic            | High-freq    |
//! |        | metadata            | sensor data  |
//! |        +----------+----------+              |
//! |                   |                         |
//! |        +----------v---------+               |
//! |        | Unified SPARQL     |               |
//! |        | Query Layer        |               |
//! |        +--------------------+               |
//! +---------------------------------------------+
//! ```
//!
//! # Quick Start
//!
//! ## Basic Usage
//!
//! ```
//! use oxirs_tsdb::HybridStore;
//! use chrono::Utc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create hybrid store (RDF + time-series)
//! let store = HybridStore::new()?;
//!
//! // Direct time-series insertion
//! let series_id = 1;
//! let timestamp = Utc::now();
//! let value = 22.5;
//! store.insert_ts(series_id, timestamp, value)?;
//!
//! // Query time range
//! let start = timestamp - chrono::Duration::hours(1);
//! let end = timestamp + chrono::Duration::hours(1);
//! let points = store.query_ts_range(series_id, start, end)?;
//! # Ok(())
//! # }
//! ```

/// Configuration types for TSDB storage and query options.
pub mod config;
/// Error types and result aliases for TSDB operations.
pub mod error;
/// Query engine for time-series data with aggregations and window functions.
pub mod query;
/// Series definitions including data points and metadata.
pub mod series;
/// Storage layer with compression and chunk management.
pub mod storage;
/// Write path with WAL and compaction for durable data ingestion.
pub mod write;

/// SPARQL temporal extensions for time-series queries.
pub mod sparql;

/// Integration with oxirs-core RDF Store (hybrid storage).
pub mod integration;

// Re-exports
pub use config::{AggregationFunction, TsdbConfig};
pub use error::{TsdbError, TsdbResult};
pub use series::{DataPoint, SeriesDescriptor, SeriesMetadata};
pub use storage::{ChunkEntry, ColumnarStore, SeriesIndex};
pub use storage::{DeltaOfDeltaCompressor, DeltaOfDeltaDecompressor};
pub use storage::{GorillaCompressor, GorillaDecompressor, TimeChunk};
pub use write::{
    BufferConfig, BufferStats, CompactionConfig, CompactionStats, Compactor, RetentionEnforcer,
    RetentionStats, WalEntry, WriteAheadLog, WriteBuffer,
};

// Query re-exports
pub use query::{
    Aggregation, AggregationResult, InterpolateMethod, Interpolator, QueryBuilder, QueryEngine,
    QueryResult, RangeQuery, ResampleBucket, Resampler, TimeRange, WindowFunction, WindowSpec,
};

// Integration re-exports
pub use integration::{Confidence, DetectionResult, HybridStore, RdfBridge};

// SPARQL re-exports
pub use sparql::{
    interpolate_function, register_temporal_functions, resample_function, window_function,
    QueryRouter, RoutingDecision, TemporalFunctionRegistry, TemporalValue,
};

/// Advanced compression algorithms: Gorilla XOR, RLE, Dictionary, Adaptive.
pub mod compression;

/// Raft consensus state machine for distributed TSDB replication.
pub mod replication;

// Compression re-exports
pub use compression::{
    dict_decode, dict_encode, gorilla_decode, gorilla_encode, rle_decode, rle_encode,
    AdaptiveCompressor, CompressedBlock, CompressionAlgorithm, DictionaryBlock, DictionaryEncoder,
    GorillaDecoder, GorillaEncoder, RleBlock, RleEncoder, RleRun,
};

// Replication re-exports
pub use replication::{
    AppendEntriesArgs, AppendEntriesReply, LogEntry, RaftError, RaftResult, RaftRole, RaftState,
    RequestVoteArgs, RequestVoteReply, TsdbCommand, WriteEntry,
};

/// Advanced analytics: anomaly detection and time-series forecasting.
pub mod analytics;

/// Statistical anomaly detection for time-series data.
pub mod anomaly_detector;

// Replication group re-exports
pub use replication::{ReplicationGroup, TsdbRaftNode};

// Write path re-exports (batch writer)
pub use write::{BatchWriter, BatchWriterConfig, CrcWal, MetricPoint};

// Analytics re-exports (Arrow / Parquet export and columnar/SQL export)
pub use analytics::{
    AggregationFunction as ExportAggregation, ArrowExporter, ColumnarExport, ColumnarStats,
    DuckDbQueryAdapter, ExportedPoint, ParquetCompression, ParquetExporter,
};

// SQL export re-exports
pub use analytics::{DataValueType, MetricSchema, MetricSchemaBuilder, SqlDataPoint, SqlExporter};

// Kalman filter re-exports
pub use analytics::{AdaptiveKalmanFilter, AnomalyEvent, KalmanAnomaly, KalmanFilter};

// GPU aggregation re-exports
pub use analytics::{GpuAggMetrics, GpuAggOp, GpuDownsampler, GpuTimeSeriesAggregator};

// Arrow IPC re-exports
pub use analytics::{
    ArrowColumn, ArrowDataType, ArrowField, ArrowIpcReader, ArrowIpcWriter, ArrowRecordBatch,
    ArrowSchema, TaggedDataPoint, TimeUnit,
};

// Parquet export re-exports
pub use analytics::{
    ParquetColumn, ParquetIpcCompression, ParquetReader, ParquetValues, ParquetWriter,
};

/// Gorilla/Delta-of-Delta compression for time-series data (v1.1.0).
pub mod gorilla_compression;

/// Statistical anomaly detection for time-series data (v1.2.0).
pub mod anomaly_detection;

/// Retention policies for time-series data expiration and downsampling (v1.1.0).
pub mod retention_policy;

/// Time-series downsampling algorithms: LTTB, Average, MinMax, First, Last, Sum, Count (v1.2.0).
pub mod downsampler;

/// Time-series value compression codecs: Delta, RLE, Zigzag, Gorilla, Plain (v1.5.0).
pub mod compression_codec;

/// Forward/backward time series iterator with windowing (v1.6.0).
pub mod series_iterator;

/// Time-series alerting rules engine (v1.7.0).
pub mod alert_rule;

/// Continuous aggregate queries with materialized views (v1.8.0).
pub mod continuous_query;

/// Tag-based time series indexing with inverted index (v1.9.0).
pub mod tag_index;

/// In-memory write buffer for time-series ingestion: size/time flush policies, WAL integration, backpressure (v2.0.0).
pub mod write_buffer;

/// Time-series event correlation engine: Pearson correlation, sliding-window
/// evaluation, threshold triggers, and event purging (v1.1.0 round 14)
pub mod event_correlator;

/// Time-series forecasting using exponential smoothing and moving averages (v1.1.0 round 15)
pub mod forecaster;

/// Time-series rollup/downsampling engine: windowed Mean/Sum/Min/Max/Count/First/Last
/// aggregation, multi-resolution rollup, and LTTB visual downsampling (v1.1.0 round 16)
pub mod rollup_engine;
