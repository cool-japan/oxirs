//! Apache Arrow and Parquet export for TSDB time-series data.
//!
//! Converts TSDB `DataPoint` collections to Arrow `RecordBatch` structures
//! and optionally serializes them to Parquet files for interoperability with
//! external analytics tools (DuckDB, Spark, Polars, etc.).
//!
//! ## Schema
//!
//! Every exported batch uses the following schema:
//!
//! | Column      | Arrow type              | Notes                            |
//! |-------------|-------------------------|----------------------------------|
//! | `timestamp` | `Int64`                 | Unix epoch milliseconds (UTC)    |
//! | `metric`    | `Utf8`                  | Metric / series name             |
//! | `value`     | `Float64`               | Observed measurement             |
//! | `tags_json` | `Utf8`                  | JSON-encoded tag key-value pairs |

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// -- Feature-gated Arrow / Parquet imports ------------------------------------
#[cfg(feature = "arrow-export")]
use {
    arrow::{
        array::{Float64Array, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    std::sync::Arc,
};

// -- Parquet-specific imports -------------------------------------------------
#[cfg(feature = "arrow-export")]
use {
    parquet::{
        arrow::ArrowWriter, basic::Compression as PqCompression, file::properties::WriterProperties,
    },
    std::{fs::File, path::Path},
};

// =============================================================================
// Public data types (always compiled -- no feature gate)
// =============================================================================

/// A single time-series measurement ready for Arrow export.
///
/// This is a flat, owned representation that can be accumulated across
/// multiple series before batch conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportedPoint {
    /// Unix epoch milliseconds (UTC).
    pub timestamp_ms: i64,
    /// Series / metric name.
    pub metric: String,
    /// Observed value.
    pub value: f64,
    /// Tag key-value pairs serialised as a JSON object string.
    pub tags_json: String,
}

impl ExportedPoint {
    /// Construct a point with no tags.
    pub fn new(timestamp_ms: i64, metric: impl Into<String>, value: f64) -> Self {
        Self {
            timestamp_ms,
            metric: metric.into(),
            value,
            tags_json: "{}".to_string(),
        }
    }

    /// Construct a point with a tag map.
    pub fn with_tags(
        timestamp_ms: i64,
        metric: impl Into<String>,
        value: f64,
        tags: &HashMap<String, String>,
    ) -> TsdbResult<Self> {
        let tags_json =
            serde_json::to_string(tags).map_err(|e| TsdbError::Serialization(e.to_string()))?;
        Ok(Self {
            timestamp_ms,
            metric: metric.into(),
            value,
            tags_json,
        })
    }

    /// Parse the embedded JSON tag string back to a map.
    pub fn parse_tags(&self) -> TsdbResult<HashMap<String, String>> {
        serde_json::from_str(&self.tags_json).map_err(|e| TsdbError::Serialization(e.to_string()))
    }
}

// =============================================================================
// Compression enum (always compiled)
// =============================================================================

/// Parquet compression codec selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParquetCompression {
    /// No compression (fastest writes, largest files).
    None,
    /// Snappy compression (balanced speed / size).
    #[default]
    Snappy,
    /// Zstandard compression (best size, slower writes).
    Zstd,
    /// Gzip compression (maximum compatibility).
    Gzip,
}

impl ParquetCompression {
    /// Return a human-readable label for this codec.
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Snappy => "snappy",
            Self::Zstd => "zstd",
            Self::Gzip => "gzip",
        }
    }
}

impl std::fmt::Display for ParquetCompression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// =============================================================================
// ArrowExporter
// =============================================================================

/// Converts TSDB data points to Apache Arrow `RecordBatch` objects.
#[derive(Debug, Default)]
pub struct ArrowExporter {
    /// Maximum rows per batch (0 = unlimited).
    max_rows_per_batch: usize,
}

impl ArrowExporter {
    /// Create an exporter with no row-count limit.
    pub fn new() -> Self {
        Self {
            max_rows_per_batch: 0,
        }
    }

    /// Create an exporter that caps each batch at `max_rows` rows.
    pub fn with_max_rows(max_rows: usize) -> Self {
        Self {
            max_rows_per_batch: max_rows,
        }
    }

    /// Return the configured maximum rows per batch (0 = unlimited).
    pub fn max_rows_per_batch(&self) -> usize {
        self.max_rows_per_batch
    }

    /// Export a slice of points to an Arrow `RecordBatch`.
    #[cfg(feature = "arrow-export")]
    pub fn export_batch(&self, points: &[ExportedPoint]) -> TsdbResult<RecordBatch> {
        let schema = Self::schema();

        let timestamps: Int64Array = points.iter().map(|p| p.timestamp_ms).collect();
        let metrics: StringArray = points.iter().map(|p| Some(p.metric.as_str())).collect();
        let values: Float64Array = points.iter().map(|p| p.value).collect();
        let tags: StringArray = points.iter().map(|p| Some(p.tags_json.as_str())).collect();

        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(timestamps),
                Arc::new(metrics),
                Arc::new(values),
                Arc::new(tags),
            ],
        )
        .map_err(|e| TsdbError::Arrow(e.to_string()))
    }

    /// Split points into multiple batches capped at `max_rows_per_batch`.
    #[cfg(feature = "arrow-export")]
    pub fn export_batches(&self, points: &[ExportedPoint]) -> TsdbResult<Vec<RecordBatch>> {
        if points.is_empty() {
            return Ok(vec![]);
        }
        let chunk_size = if self.max_rows_per_batch == 0 {
            points.len()
        } else {
            self.max_rows_per_batch
        };
        points
            .chunks(chunk_size)
            .map(|chunk| self.export_batch(chunk))
            .collect()
    }

    /// Return a stub row count without the `arrow-export` feature.
    #[cfg(not(feature = "arrow-export"))]
    pub fn export_batch_count(&self, points: &[ExportedPoint]) -> usize {
        points.len()
    }

    /// Arrow schema used for all TSDB exports.
    #[cfg(feature = "arrow-export")]
    pub fn schema() -> Schema {
        Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("metric", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
            Field::new("tags_json", DataType::Utf8, false),
        ])
    }

    /// Filter exported points to a specific metric name.
    pub fn filter_by_metric<'a>(
        points: &'a [ExportedPoint],
        metric: &str,
    ) -> Vec<&'a ExportedPoint> {
        points.iter().filter(|p| p.metric == metric).collect()
    }

    /// Filter exported points to a time range (inclusive on both ends).
    pub fn filter_by_time_range(
        points: &[ExportedPoint],
        start_ms: i64,
        end_ms: i64,
    ) -> Vec<ExportedPoint> {
        points
            .iter()
            .filter(|p| p.timestamp_ms >= start_ms && p.timestamp_ms <= end_ms)
            .cloned()
            .collect()
    }

    /// Compute basic statistics over a collection of exported points.
    pub fn compute_stats(points: &[ExportedPoint]) -> ExportStats {
        if points.is_empty() {
            return ExportStats::default();
        }
        let count = points.len();
        let sum: f64 = points.iter().map(|p| p.value).sum();
        let mean = sum / count as f64;
        let min = points.iter().map(|p| p.value).fold(f64::INFINITY, f64::min);
        let max = points
            .iter()
            .map(|p| p.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let variance = if count > 1 {
            points.iter().map(|p| (p.value - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let first_ts = points.iter().map(|p| p.timestamp_ms).min().unwrap_or(0);
        let last_ts = points.iter().map(|p| p.timestamp_ms).max().unwrap_or(0);
        let distinct_metrics: std::collections::HashSet<&str> =
            points.iter().map(|p| p.metric.as_str()).collect();

        ExportStats {
            count,
            sum,
            mean,
            min,
            max,
            variance,
            stddev: variance.sqrt(),
            first_timestamp_ms: first_ts,
            last_timestamp_ms: last_ts,
            distinct_metrics: distinct_metrics.len(),
        }
    }

    /// Group exported points by metric name.
    pub fn group_by_metric(points: &[ExportedPoint]) -> HashMap<String, Vec<ExportedPoint>> {
        let mut groups: HashMap<String, Vec<ExportedPoint>> = HashMap::new();
        for p in points {
            groups.entry(p.metric.clone()).or_default().push(p.clone());
        }
        groups
    }

    /// Sort exported points by timestamp in ascending order.
    pub fn sort_by_timestamp(points: &mut [ExportedPoint]) {
        points.sort_by_key(|p| p.timestamp_ms);
    }
}

/// Statistics computed over a set of exported points.
#[derive(Debug, Clone, Default)]
pub struct ExportStats {
    /// Number of points.
    pub count: usize,
    /// Sum of values.
    pub sum: f64,
    /// Arithmetic mean of values.
    pub mean: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Sample variance.
    pub variance: f64,
    /// Sample standard deviation.
    pub stddev: f64,
    /// Earliest timestamp in the set.
    pub first_timestamp_ms: i64,
    /// Latest timestamp in the set.
    pub last_timestamp_ms: i64,
    /// Number of distinct metric names.
    pub distinct_metrics: usize,
}

// =============================================================================
// ParquetExporter
// =============================================================================

/// Exports TSDB data to Parquet files via the Arrow columnar format.
#[derive(Debug)]
pub struct ParquetExporter {
    #[cfg_attr(not(feature = "arrow-export"), allow(dead_code))]
    arrow: ArrowExporter,
    compression: ParquetCompression,
    /// Target row-group size in bytes (Parquet default is 128 MiB).
    row_group_size: usize,
}

impl ParquetExporter {
    /// Create a Parquet exporter with Snappy compression and default row-group
    /// size (134_217_728 bytes = 128 MiB).
    pub fn new() -> Self {
        Self {
            arrow: ArrowExporter::new(),
            compression: ParquetCompression::Snappy,
            row_group_size: 134_217_728,
        }
    }

    /// Set compression codec.
    pub fn with_compression(mut self, codec: ParquetCompression) -> Self {
        self.compression = codec;
        self
    }

    /// Set row-group size in bytes.
    pub fn with_row_group_size(mut self, bytes: usize) -> Self {
        self.row_group_size = bytes;
        self
    }

    /// Return the current compression codec.
    pub fn compression(&self) -> ParquetCompression {
        self.compression
    }

    /// Return the configured row-group size in bytes.
    pub fn row_group_size(&self) -> usize {
        self.row_group_size
    }

    /// Write points to a Parquet file at `path`.
    #[cfg(feature = "arrow-export")]
    pub fn write_file(&self, points: &[ExportedPoint], path: &Path) -> TsdbResult<u64> {
        let batch = self.arrow.export_batch(points)?;

        let codec = match self.compression {
            ParquetCompression::None => PqCompression::UNCOMPRESSED,
            ParquetCompression::Snappy => PqCompression::SNAPPY,
            ParquetCompression::Zstd => PqCompression::ZSTD(Default::default()),
            ParquetCompression::Gzip => PqCompression::GZIP(Default::default()),
        };

        let props = WriterProperties::builder()
            .set_compression(codec)
            .set_max_row_group_row_count(Some(self.row_group_size / 8))
            .build();

        let file = File::create(path).map_err(|e| TsdbError::Io(e.to_string()))?;

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
            .map_err(|e| TsdbError::Arrow(e.to_string()))?;

        writer
            .write(&batch)
            .map_err(|e| TsdbError::Arrow(e.to_string()))?;

        let metadata = writer
            .close()
            .map_err(|e| TsdbError::Arrow(e.to_string()))?;

        Ok(metadata.file_metadata().num_rows() as u64)
    }

    /// Count rows that would be exported (works without `arrow-export`).
    pub fn count_rows(&self, points: &[ExportedPoint]) -> usize {
        points.len()
    }

    /// Return metadata about a planned export.
    pub fn export_metadata(&self, points: &[ExportedPoint]) -> ExportMetadata {
        let stats = ArrowExporter::compute_stats(points);
        ExportMetadata {
            row_count: points.len(),
            compression: self.compression,
            row_group_size: self.row_group_size,
            distinct_metrics: stats.distinct_metrics,
            time_span_ms: stats
                .last_timestamp_ms
                .saturating_sub(stats.first_timestamp_ms),
        }
    }
}

impl Default for ParquetExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a planned or completed Parquet export.
#[derive(Debug, Clone)]
pub struct ExportMetadata {
    /// Total row count.
    pub row_count: usize,
    /// Compression codec used.
    pub compression: ParquetCompression,
    /// Configured row-group size (bytes).
    pub row_group_size: usize,
    /// Number of distinct metrics in the dataset.
    pub distinct_metrics: usize,
    /// Time span covered (last_ts - first_ts) in milliseconds.
    pub time_span_ms: i64,
}

// =============================================================================
// DuckDbQueryAdapter
// =============================================================================

/// Generates DuckDB-compatible SQL queries over exported Parquet files.
///
/// This adapter does **not** embed DuckDB as a dependency; instead it produces
/// SQL strings that can be sent to an external DuckDB process or connection
/// pool.
#[derive(Debug, Clone)]
pub struct DuckDbQueryAdapter {
    /// Path to the Parquet file(s).  Glob patterns are supported by DuckDB.
    parquet_path: String,
}

impl DuckDbQueryAdapter {
    /// Create an adapter for a Parquet file or glob (e.g. `/data/*.parquet`).
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            parquet_path: path.into(),
        }
    }

    /// Return the configured Parquet path / glob.
    pub fn parquet_path(&self) -> &str {
        &self.parquet_path
    }

    /// Generate SQL to select all rows for a specific metric within a time range.
    pub fn select_metric(&self, metric: &str, start_ms: i64, end_ms: i64) -> String {
        format!(
            "SELECT timestamp, metric, value, tags_json \
             FROM read_parquet('{}') \
             WHERE metric = '{}' AND timestamp BETWEEN {} AND {} \
             ORDER BY timestamp ASC;",
            self.parquet_path, metric, start_ms, end_ms
        )
    }

    /// Generate SQL to compute per-metric aggregates over a time range.
    pub fn aggregate_metric(
        &self,
        metric: &str,
        start_ms: i64,
        end_ms: i64,
        aggregation: AggregationFunction,
    ) -> String {
        let agg_expr = match aggregation {
            AggregationFunction::Avg => "AVG(value)",
            AggregationFunction::Min => "MIN(value)",
            AggregationFunction::Max => "MAX(value)",
            AggregationFunction::Sum => "SUM(value)",
            AggregationFunction::Count => "COUNT(*)",
            AggregationFunction::StdDev => "STDDEV(value)",
            AggregationFunction::Percentile50 => {
                "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value)"
            }
            AggregationFunction::Percentile95 => {
                "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value)"
            }
            AggregationFunction::Percentile99 => {
                "PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY value)"
            }
        };
        format!(
            "SELECT metric, {agg} AS result \
             FROM read_parquet('{path}') \
             WHERE metric = '{metric}' AND timestamp BETWEEN {start} AND {end} \
             GROUP BY metric;",
            agg = agg_expr,
            path = self.parquet_path,
            metric = metric,
            start = start_ms,
            end = end_ms,
        )
    }

    /// Generate SQL to downsample data into fixed-width time buckets (ms).
    pub fn resample(&self, metric: &str, bucket_ms: i64) -> String {
        format!(
            "SELECT \
               (timestamp / {bucket}) * {bucket} AS bucket_start_ms, \
               metric, \
               AVG(value) AS avg_value, \
               MIN(value) AS min_value, \
               MAX(value) AS max_value, \
               COUNT(*) AS sample_count \
             FROM read_parquet('{path}') \
             WHERE metric = '{metric}' \
             GROUP BY bucket_start_ms, metric \
             ORDER BY bucket_start_ms ASC;",
            bucket = bucket_ms,
            path = self.parquet_path,
            metric = metric,
        )
    }

    /// Generate SQL to export query results back to another Parquet file.
    pub fn export_query_to_parquet(&self, query: &str, output_path: &str) -> String {
        format!(
            "COPY ({query}) TO '{output}' (FORMAT PARQUET, COMPRESSION SNAPPY);",
            query = query,
            output = output_path,
        )
    }

    /// Generate SQL to list distinct metrics in the dataset.
    pub fn list_metrics(&self) -> String {
        format!(
            "SELECT DISTINCT metric FROM read_parquet('{}') ORDER BY metric;",
            self.parquet_path
        )
    }

    /// Generate SQL to compute a time-range summary for all metrics.
    pub fn time_range_summary(&self) -> String {
        format!(
            "SELECT metric, \
               MIN(timestamp) AS first_ts_ms, \
               MAX(timestamp) AS last_ts_ms, \
               COUNT(*) AS total_points, \
               AVG(value) AS mean_value \
             FROM read_parquet('{}') \
             GROUP BY metric \
             ORDER BY metric;",
            self.parquet_path
        )
    }

    /// Generate SQL to join two metric time-series by timestamp bucket.
    pub fn join_metrics(&self, metric_a: &str, metric_b: &str, bucket_ms: i64) -> String {
        format!(
            "SELECT \
               a.bucket_start_ms, \
               a.avg_value AS {metric_a}_avg, \
               b.avg_value AS {metric_b}_avg \
             FROM (\
               SELECT (timestamp / {bucket}) * {bucket} AS bucket_start_ms, \
                      AVG(value) AS avg_value \
               FROM read_parquet('{path}') \
               WHERE metric = '{metric_a}' \
               GROUP BY bucket_start_ms\
             ) a \
             INNER JOIN (\
               SELECT (timestamp / {bucket}) * {bucket} AS bucket_start_ms, \
                      AVG(value) AS avg_value \
               FROM read_parquet('{path}') \
               WHERE metric = '{metric_b}' \
               GROUP BY bucket_start_ms\
             ) b ON a.bucket_start_ms = b.bucket_start_ms \
             ORDER BY a.bucket_start_ms ASC;",
            metric_a = metric_a,
            metric_b = metric_b,
            bucket = bucket_ms,
            path = self.parquet_path,
        )
    }

    /// Generate SQL to compute rate of change (derivative) per bucket.
    pub fn rate_of_change(&self, metric: &str, bucket_ms: i64) -> String {
        format!(
            "SELECT \
               bucket_start_ms, \
               avg_value, \
               avg_value - LAG(avg_value) OVER (ORDER BY bucket_start_ms) AS delta, \
               (avg_value - LAG(avg_value) OVER (ORDER BY bucket_start_ms)) / \
                 NULLIF(({bucket}::DOUBLE / 1000.0), 0) AS rate_per_sec \
             FROM (\
               SELECT (timestamp / {bucket}) * {bucket} AS bucket_start_ms, \
                      AVG(value) AS avg_value \
               FROM read_parquet('{path}') \
               WHERE metric = '{metric}' \
               GROUP BY bucket_start_ms\
             ) \
             ORDER BY bucket_start_ms ASC;",
            bucket = bucket_ms,
            path = self.parquet_path,
            metric = metric,
        )
    }

    /// Generate SQL to create a DuckDB view for repeated queries.
    pub fn create_view(&self, view_name: &str) -> String {
        format!(
            "CREATE OR REPLACE VIEW {view} AS \
             SELECT * FROM read_parquet('{}');",
            self.parquet_path,
            view = view_name,
        )
    }

    /// Generate SQL to count total data points per metric.
    pub fn count_per_metric(&self) -> String {
        format!(
            "SELECT metric, COUNT(*) AS point_count \
             FROM read_parquet('{}') \
             GROUP BY metric \
             ORDER BY point_count DESC;",
            self.parquet_path,
        )
    }
}

/// Aggregation function for DuckDB SQL generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationFunction {
    /// Arithmetic mean.
    Avg,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Sum of all values.
    Sum,
    /// Row count.
    Count,
    /// Population standard deviation.
    StdDev,
    /// 50th percentile (median).
    Percentile50,
    /// 95th percentile.
    Percentile95,
    /// 99th percentile.
    Percentile99,
}

// =============================================================================
// ColumnarExport — always-compiled flat export without Arrow/Parquet deps
// =============================================================================

/// A flat, owned columnar representation of exported time-series data.
///
/// `ColumnarExport` stores three parallel vectors:
/// - `timestamps`: Unix epoch milliseconds.
/// - `metrics`: series / metric names.
/// - `values`: observed measurement values.
///
/// It is independent of the `arrow-export` feature and is always compiled.
/// Use it as a lightweight bridge to CSV, TSV, or JSON exports for small to
/// medium datasets, or as a staging area before Arrow conversion.
#[derive(Debug, Clone, Default)]
pub struct ColumnarExport {
    /// Parallel array of Unix epoch milliseconds.
    pub timestamps: Vec<i64>,
    /// Parallel array of metric / series names.
    pub metrics: Vec<String>,
    /// Parallel array of observed values.
    pub values: Vec<f64>,
}

impl ColumnarExport {
    /// Create an empty `ColumnarExport`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a `ColumnarExport` pre-allocated for `capacity` rows.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            metrics: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// Populate from a slice of [`ExportedPoint`]s.
    pub fn from_points(points: &[ExportedPoint]) -> Self {
        let mut out = Self::with_capacity(points.len());
        for p in points {
            out.timestamps.push(p.timestamp_ms);
            out.metrics.push(p.metric.clone());
            out.values.push(p.value);
        }
        out
    }

    /// Return the number of rows in this export.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Return `true` if the export contains no rows.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Append a single row.
    pub fn push(&mut self, timestamp_ms: i64, metric: impl Into<String>, value: f64) {
        self.timestamps.push(timestamp_ms);
        self.metrics.push(metric.into());
        self.values.push(value);
    }

    /// Convert to a `Vec<ExportedPoint>`.
    pub fn to_points(&self) -> Vec<ExportedPoint> {
        self.timestamps
            .iter()
            .zip(self.metrics.iter())
            .zip(self.values.iter())
            .map(|((ts, m), v)| ExportedPoint::new(*ts, m.clone(), *v))
            .collect()
    }

    /// Write this export as CSV to the file at `path`.
    ///
    /// The header row is `timestamp_ms,metric,value`.
    pub fn to_csv(&self, path: &std::path::Path) -> TsdbResult<()> {
        use std::io::Write as IoWrite;
        let file = std::fs::File::create(path).map_err(|e| TsdbError::Io(e.to_string()))?;
        let mut writer = std::io::BufWriter::new(file);
        writeln!(writer, "timestamp_ms,metric,value").map_err(|e| TsdbError::Io(e.to_string()))?;
        for i in 0..self.len() {
            // Escape commas / quotes inside the metric name.
            let metric_escaped = if self.metrics[i].contains(',') || self.metrics[i].contains('"') {
                format!("\"{}\"", self.metrics[i].replace('"', "\"\""))
            } else {
                self.metrics[i].clone()
            };
            writeln!(
                writer,
                "{},{},{}",
                self.timestamps[i], metric_escaped, self.values[i]
            )
            .map_err(|e| TsdbError::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Write this export as TSV (tab-separated values) to the file at `path`.
    ///
    /// The header row is `timestamp_ms\tmetric\tvalue`.
    pub fn to_tsv(&self, path: &std::path::Path) -> TsdbResult<()> {
        use std::io::Write as IoWrite;
        let file = std::fs::File::create(path).map_err(|e| TsdbError::Io(e.to_string()))?;
        let mut writer = std::io::BufWriter::new(file);
        writeln!(writer, "timestamp_ms\tmetric\tvalue")
            .map_err(|e| TsdbError::Io(e.to_string()))?;
        for i in 0..self.len() {
            writeln!(
                writer,
                "{}\t{}\t{}",
                self.timestamps[i], self.metrics[i], self.values[i]
            )
            .map_err(|e| TsdbError::Io(e.to_string()))?;
        }
        Ok(())
    }

    /// Serialize this export to a JSON array.
    ///
    /// Each element is `{"timestamp_ms": <i64>, "metric": <str>, "value": <f64>}`.
    pub fn to_json(&self) -> TsdbResult<serde_json::Value> {
        let rows: Vec<serde_json::Value> = self
            .timestamps
            .iter()
            .zip(self.metrics.iter())
            .zip(self.values.iter())
            .map(|((ts, m), v)| {
                serde_json::json!({
                    "timestamp_ms": ts,
                    "metric": m,
                    "value": v,
                })
            })
            .collect();
        Ok(serde_json::Value::Array(rows))
    }

    /// Serialize this export to a JSON string.
    pub fn to_json_string(&self) -> TsdbResult<String> {
        let v = self.to_json()?;
        serde_json::to_string(&v).map_err(|e| TsdbError::Serialization(e.to_string()))
    }

    /// Filter rows to only those matching the given metric name, returning a
    /// new `ColumnarExport`.
    pub fn filter_metric(&self, metric: &str) -> Self {
        let mut out = Self::new();
        for i in 0..self.len() {
            if self.metrics[i] == metric {
                out.push(self.timestamps[i], self.metrics[i].clone(), self.values[i]);
            }
        }
        out
    }

    /// Filter rows to the given time range [start_ms, end_ms] inclusive.
    pub fn filter_time_range(&self, start_ms: i64, end_ms: i64) -> Self {
        let mut out = Self::new();
        for i in 0..self.len() {
            let ts = self.timestamps[i];
            if ts >= start_ms && ts <= end_ms {
                out.push(ts, self.metrics[i].clone(), self.values[i]);
            }
        }
        out
    }

    /// Sort rows by timestamp in ascending order (in-place).
    pub fn sort_by_timestamp(&mut self) {
        // Collect indices sorted by timestamp.
        let n = self.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by_key(|&i| self.timestamps[i]);

        let old_ts = self.timestamps.clone();
        let old_m = self.metrics.clone();
        let old_v = self.values.clone();
        for (new_pos, &old_pos) in indices.iter().enumerate() {
            self.timestamps[new_pos] = old_ts[old_pos];
            self.metrics[new_pos] = old_m[old_pos].clone();
            self.values[new_pos] = old_v[old_pos];
        }
    }

    /// Compute basic descriptive statistics over the `value` column.
    pub fn value_stats(&self) -> ColumnarStats {
        if self.values.is_empty() {
            return ColumnarStats::default();
        }
        let n = self.values.len();
        let sum: f64 = self.values.iter().sum();
        let mean = sum / n as f64;
        let min = self.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let variance = if n > 1 {
            self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        ColumnarStats {
            count: n,
            sum,
            mean,
            min,
            max,
            variance,
            stddev: variance.sqrt(),
        }
    }

    /// Read a CSV file previously written by `to_csv` back into a
    /// `ColumnarExport`.
    pub fn from_csv(path: &std::path::Path) -> TsdbResult<Self> {
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(path).map_err(|e| TsdbError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut out = Self::new();
        let mut first = true;
        for line in reader.lines() {
            let line = line.map_err(|e| TsdbError::Io(e.to_string()))?;
            if first {
                first = false;
                continue; // skip header
            }
            if line.is_empty() {
                continue;
            }
            // Parse: timestamp_ms,metric,value  (metric may be quoted)
            let parts = parse_csv_line(&line);
            if parts.len() < 3 {
                return Err(TsdbError::Serialization(format!(
                    "CSV line has fewer than 3 fields: {line}"
                )));
            }
            let ts: i64 = parts[0].parse().map_err(|_| {
                TsdbError::Serialization(format!("invalid timestamp: {}", parts[0]))
            })?;
            let metric = parts[1].clone();
            let value: f64 = parts[2]
                .parse()
                .map_err(|_| TsdbError::Serialization(format!("invalid value: {}", parts[2])))?;
            out.push(ts, metric, value);
        }
        Ok(out)
    }
}

/// Descriptive statistics over a [`ColumnarExport`]'s value column.
#[derive(Debug, Clone, Default)]
pub struct ColumnarStats {
    /// Number of data points.
    pub count: usize,
    /// Sum of values.
    pub sum: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Sample variance.
    pub variance: f64,
    /// Sample standard deviation.
    pub stddev: f64,
}

/// Minimal CSV line parser that handles optional double-quoted fields.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if !in_quotes => {
                in_quotes = true;
            }
            '"' if in_quotes => {
                // Escaped quote "".
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            other => {
                current.push(other);
            }
        }
    }
    fields.push(current);
    fields
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // -- ExportedPoint --------------------------------------------------------

    #[test]
    fn test_exported_point_new_no_tags() {
        let p = ExportedPoint::new(1_700_000_000_000, "cpu", 88.5);
        assert_eq!(p.metric, "cpu");
        assert_eq!(p.value, 88.5);
        assert_eq!(p.tags_json, "{}");
    }

    #[test]
    fn test_exported_point_with_tags_roundtrip() {
        let mut tags = HashMap::new();
        tags.insert("host".to_string(), "srv-01".to_string());
        tags.insert("region".to_string(), "eu-west".to_string());

        let p = ExportedPoint::with_tags(1_700_000_000_000, "mem", 4096.0, &tags)
            .expect("serialization should succeed");

        let parsed = p.parse_tags().expect("deserialization should succeed");
        assert_eq!(parsed["host"], "srv-01");
        assert_eq!(parsed["region"], "eu-west");
    }

    #[test]
    fn test_exported_point_serialization() {
        let p = ExportedPoint::new(42_000, "latency_ms", 1.5);
        let json = serde_json::to_string(&p).expect("json serialize");
        let back: ExportedPoint = serde_json::from_str(&json).expect("json deserialize");
        assert_eq!(p, back);
    }

    #[test]
    fn test_exported_point_clone() {
        let p = ExportedPoint::new(1, "x", 0.0);
        let q = p.clone();
        assert_eq!(p, q);
    }

    // -- ArrowExporter --------------------------------------------------------

    #[test]
    fn test_arrow_exporter_default() {
        let e = ArrowExporter::default();
        assert_eq!(e.max_rows_per_batch(), 0);
    }

    #[test]
    fn test_arrow_exporter_with_max_rows() {
        let e = ArrowExporter::with_max_rows(100);
        assert_eq!(e.max_rows_per_batch(), 100);
    }

    #[test]
    fn test_arrow_filter_by_metric() {
        let pts = vec![
            ExportedPoint::new(1, "cpu", 10.0),
            ExportedPoint::new(2, "mem", 20.0),
            ExportedPoint::new(3, "cpu", 30.0),
        ];
        let filtered = ArrowExporter::filter_by_metric(&pts, "cpu");
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|p| p.metric == "cpu"));
    }

    #[test]
    fn test_arrow_filter_by_time_range() {
        let pts = vec![
            ExportedPoint::new(100, "x", 1.0),
            ExportedPoint::new(200, "x", 2.0),
            ExportedPoint::new(300, "x", 3.0),
            ExportedPoint::new(400, "x", 4.0),
        ];
        let filtered = ArrowExporter::filter_by_time_range(&pts, 150, 350);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].timestamp_ms, 200);
        assert_eq!(filtered[1].timestamp_ms, 300);
    }

    #[test]
    fn test_arrow_filter_by_time_range_empty() {
        let pts: Vec<ExportedPoint> = vec![];
        let filtered = ArrowExporter::filter_by_time_range(&pts, 0, 1000);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_arrow_filter_by_time_range_no_match() {
        let pts = vec![ExportedPoint::new(1000, "x", 1.0)];
        let filtered = ArrowExporter::filter_by_time_range(&pts, 0, 500);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_arrow_filter_by_time_range_inclusive_boundary() {
        let pts = vec![
            ExportedPoint::new(100, "x", 1.0),
            ExportedPoint::new(200, "x", 2.0),
        ];
        let filtered = ArrowExporter::filter_by_time_range(&pts, 100, 200);
        assert_eq!(filtered.len(), 2);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_arrow_export_batch_schema() {
        use arrow::datatypes::DataType;
        let schema = ArrowExporter::schema();
        assert_eq!(schema.field(0).name(), "timestamp");
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).name(), "metric");
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(2).name(), "value");
        assert_eq!(schema.field(2).data_type(), &DataType::Float64);
        assert_eq!(schema.field(3).name(), "tags_json");
        assert_eq!(schema.field(3).data_type(), &DataType::Utf8);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_arrow_export_batch_row_count() {
        let pts: Vec<_> = (0..50)
            .map(|i| ExportedPoint::new(i * 1000, "temp", i as f64))
            .collect();
        let exporter = ArrowExporter::new();
        let batch = exporter.export_batch(&pts).expect("export should succeed");
        assert_eq!(batch.num_rows(), 50);
        assert_eq!(batch.num_columns(), 4);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_arrow_export_batch_values() {
        let pts = vec![
            ExportedPoint::new(1_000, "pressure", 101.325),
            ExportedPoint::new(2_000, "pressure", 102.0),
        ];
        let exporter = ArrowExporter::new();
        let batch = exporter.export_batch(&pts).expect("export");

        use arrow::array::Float64Array;
        let values = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64Array");
        assert!((values.value(0) - 101.325).abs() < 1e-9);
        assert!((values.value(1) - 102.0).abs() < 1e-9);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_arrow_export_batches_chunking() {
        let pts: Vec<_> = (0..25)
            .map(|i| ExportedPoint::new(i, "x", i as f64))
            .collect();
        let exporter = ArrowExporter::with_max_rows(10);
        let batches = exporter.export_batches(&pts).expect("export batches");
        assert_eq!(batches.len(), 3); // 10 + 10 + 5
        assert_eq!(batches[0].num_rows(), 10);
        assert_eq!(batches[2].num_rows(), 5);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_arrow_export_batches_empty() {
        let pts: Vec<ExportedPoint> = vec![];
        let exporter = ArrowExporter::new();
        let batches = exporter.export_batches(&pts).expect("empty export");
        assert!(batches.is_empty());
    }

    // -- ExportStats ----------------------------------------------------------

    #[test]
    fn test_compute_stats_basic() {
        let pts = vec![
            ExportedPoint::new(100, "cpu", 10.0),
            ExportedPoint::new(200, "cpu", 20.0),
            ExportedPoint::new(300, "cpu", 30.0),
        ];
        let stats = ArrowExporter::compute_stats(&pts);
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < f64::EPSILON);
        assert!((stats.min - 10.0).abs() < f64::EPSILON);
        assert!((stats.max - 30.0).abs() < f64::EPSILON);
        assert!((stats.sum - 60.0).abs() < f64::EPSILON);
        assert_eq!(stats.first_timestamp_ms, 100);
        assert_eq!(stats.last_timestamp_ms, 300);
        assert_eq!(stats.distinct_metrics, 1);
    }

    #[test]
    fn test_compute_stats_empty() {
        let pts: Vec<ExportedPoint> = vec![];
        let stats = ArrowExporter::compute_stats(&pts);
        assert_eq!(stats.count, 0);
        assert!((stats.mean - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_stats_single_point() {
        let pts = vec![ExportedPoint::new(1000, "mem", 42.0)];
        let stats = ArrowExporter::compute_stats(&pts);
        assert_eq!(stats.count, 1);
        assert!((stats.mean - 42.0).abs() < f64::EPSILON);
        assert!((stats.variance - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.distinct_metrics, 1);
    }

    #[test]
    fn test_compute_stats_multiple_metrics() {
        let pts = vec![
            ExportedPoint::new(100, "cpu", 10.0),
            ExportedPoint::new(200, "mem", 20.0),
            ExportedPoint::new(300, "disk", 30.0),
        ];
        let stats = ArrowExporter::compute_stats(&pts);
        assert_eq!(stats.distinct_metrics, 3);
    }

    #[test]
    fn test_compute_stats_stddev() {
        let pts = vec![
            ExportedPoint::new(1, "x", 2.0),
            ExportedPoint::new(2, "x", 4.0),
            ExportedPoint::new(3, "x", 6.0),
            ExportedPoint::new(4, "x", 8.0),
        ];
        let stats = ArrowExporter::compute_stats(&pts);
        // variance of [2,4,6,8] = sum((x-5)^2)/3 = (9+1+1+9)/3 = 20/3
        assert!((stats.variance - 20.0 / 3.0).abs() < 1e-9);
        assert!((stats.stddev - (20.0_f64 / 3.0).sqrt()).abs() < 1e-9);
    }

    // -- group_by_metric / sort -----------------------------------------------

    #[test]
    fn test_group_by_metric() {
        let pts = vec![
            ExportedPoint::new(1, "cpu", 10.0),
            ExportedPoint::new(2, "mem", 20.0),
            ExportedPoint::new(3, "cpu", 30.0),
            ExportedPoint::new(4, "disk", 40.0),
        ];
        let groups = ArrowExporter::group_by_metric(&pts);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups["cpu"].len(), 2);
        assert_eq!(groups["mem"].len(), 1);
        assert_eq!(groups["disk"].len(), 1);
    }

    #[test]
    fn test_sort_by_timestamp() {
        let mut pts = vec![
            ExportedPoint::new(300, "x", 3.0),
            ExportedPoint::new(100, "x", 1.0),
            ExportedPoint::new(200, "x", 2.0),
        ];
        ArrowExporter::sort_by_timestamp(&mut pts);
        assert_eq!(pts[0].timestamp_ms, 100);
        assert_eq!(pts[1].timestamp_ms, 200);
        assert_eq!(pts[2].timestamp_ms, 300);
    }

    // -- ParquetExporter ------------------------------------------------------

    #[test]
    fn test_parquet_exporter_default() {
        let e = ParquetExporter::default();
        assert_eq!(e.compression(), ParquetCompression::Snappy);
        assert_eq!(e.row_group_size(), 134_217_728);
    }

    #[test]
    fn test_parquet_exporter_builder() {
        let e = ParquetExporter::new()
            .with_compression(ParquetCompression::Zstd)
            .with_row_group_size(1024 * 1024);
        assert_eq!(e.compression(), ParquetCompression::Zstd);
        assert_eq!(e.row_group_size(), 1024 * 1024);
    }

    #[test]
    fn test_parquet_exporter_count_rows() {
        let pts: Vec<_> = (0..100)
            .map(|i| ExportedPoint::new(i, "x", i as f64))
            .collect();
        let e = ParquetExporter::new();
        assert_eq!(e.count_rows(&pts), 100);
    }

    #[test]
    fn test_parquet_compression_label() {
        assert_eq!(ParquetCompression::None.label(), "none");
        assert_eq!(ParquetCompression::Snappy.label(), "snappy");
        assert_eq!(ParquetCompression::Zstd.label(), "zstd");
        assert_eq!(ParquetCompression::Gzip.label(), "gzip");
    }

    #[test]
    fn test_parquet_compression_display() {
        assert_eq!(format!("{}", ParquetCompression::Snappy), "snappy");
        assert_eq!(format!("{}", ParquetCompression::Zstd), "zstd");
    }

    #[test]
    fn test_export_metadata() {
        let pts = vec![
            ExportedPoint::new(1000, "cpu", 10.0),
            ExportedPoint::new(2000, "cpu", 20.0),
            ExportedPoint::new(3000, "mem", 30.0),
        ];
        let e = ParquetExporter::new().with_compression(ParquetCompression::Zstd);
        let meta = e.export_metadata(&pts);
        assert_eq!(meta.row_count, 3);
        assert_eq!(meta.compression, ParquetCompression::Zstd);
        assert_eq!(meta.distinct_metrics, 2);
        assert_eq!(meta.time_span_ms, 2000);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_parquet_write_snappy() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_tsdb_test_snappy.parquet");

        let pts: Vec<_> = (0..20)
            .map(|i| ExportedPoint::new(i * 1_000, "temp", 20.0 + i as f64))
            .collect();

        let exporter = ParquetExporter::new().with_compression(ParquetCompression::Snappy);
        let rows = exporter.write_file(&pts, &path).expect("write parquet");
        assert_eq!(rows, 20);
        assert!(path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_parquet_write_zstd() {
        let path = std::env::temp_dir().join("oxirs_tsdb_test_zstd.parquet");
        let pts: Vec<_> = (0..10)
            .map(|i| ExportedPoint::new(i, "pressure", i as f64 * 1.1))
            .collect();

        let exporter = ParquetExporter::new().with_compression(ParquetCompression::Zstd);
        let rows = exporter
            .write_file(&pts, &path)
            .expect("write parquet zstd");
        assert_eq!(rows, 10);
        let _ = std::fs::remove_file(path);
    }

    #[cfg(feature = "arrow-export")]
    #[test]
    fn test_parquet_write_no_compression() {
        let path = std::env::temp_dir().join("oxirs_tsdb_test_none.parquet");
        let pts = vec![ExportedPoint::new(0, "v", 1.0)];

        let exporter = ParquetExporter::new().with_compression(ParquetCompression::None);
        let rows = exporter.write_file(&pts, &path).expect("write");
        assert_eq!(rows, 1);
        let _ = std::fs::remove_file(path);
    }

    // -- DuckDbQueryAdapter ---------------------------------------------------

    #[test]
    fn test_duckdb_adapter_select_metric_sql() {
        let adapter = DuckDbQueryAdapter::new("/data/tsdb.parquet");
        let sql = adapter.select_metric("cpu", 1_000_000, 2_000_000);
        assert!(sql.contains("read_parquet('/data/tsdb.parquet')"));
        assert!(sql.contains("metric = 'cpu'"));
        assert!(sql.contains("1000000"));
        assert!(sql.contains("2000000"));
        assert!(sql.contains("ORDER BY timestamp ASC"));
    }

    #[test]
    fn test_duckdb_adapter_aggregate_avg_sql() {
        let adapter = DuckDbQueryAdapter::new("/tmp/data.parquet");
        let sql = adapter.aggregate_metric("mem", 0, 9_999, AggregationFunction::Avg);
        assert!(sql.contains("AVG(value)"));
        assert!(sql.contains("GROUP BY metric"));
    }

    #[test]
    fn test_duckdb_adapter_aggregate_all_functions() {
        let adapter = DuckDbQueryAdapter::new("/tmp/t.parquet");
        for (func, expected) in [
            (AggregationFunction::Min, "MIN(value)"),
            (AggregationFunction::Max, "MAX(value)"),
            (AggregationFunction::Sum, "SUM(value)"),
            (AggregationFunction::Count, "COUNT(*)"),
            (AggregationFunction::StdDev, "STDDEV(value)"),
        ] {
            let sql = adapter.aggregate_metric("x", 0, 1, func);
            assert!(sql.contains(expected), "expected {expected} for {func:?}");
        }
    }

    #[test]
    fn test_duckdb_adapter_percentile_functions() {
        let adapter = DuckDbQueryAdapter::new("/tmp/t.parquet");
        let sql = adapter.aggregate_metric("cpu", 0, 1000, AggregationFunction::Percentile50);
        assert!(sql.contains("PERCENTILE_CONT(0.5)"));

        let sql = adapter.aggregate_metric("cpu", 0, 1000, AggregationFunction::Percentile95);
        assert!(sql.contains("PERCENTILE_CONT(0.95)"));

        let sql = adapter.aggregate_metric("cpu", 0, 1000, AggregationFunction::Percentile99);
        assert!(sql.contains("PERCENTILE_CONT(0.99)"));
    }

    #[test]
    fn test_duckdb_adapter_resample_sql() {
        let adapter = DuckDbQueryAdapter::new("/tmp/t.parquet");
        let sql = adapter.resample("cpu", 60_000);
        assert!(sql.contains("60000"));
        assert!(sql.contains("AVG(value)"));
        assert!(sql.contains("GROUP BY bucket_start_ms, metric"));
    }

    #[test]
    fn test_duckdb_adapter_list_metrics_sql() {
        let adapter = DuckDbQueryAdapter::new("/data/*.parquet");
        let sql = adapter.list_metrics();
        assert!(sql.contains("DISTINCT metric"));
        assert!(sql.contains("read_parquet('/data/*.parquet')"));
    }

    #[test]
    fn test_duckdb_adapter_time_range_summary_sql() {
        let adapter = DuckDbQueryAdapter::new("data.parquet");
        let sql = adapter.time_range_summary();
        assert!(sql.contains("MIN(timestamp)"));
        assert!(sql.contains("MAX(timestamp)"));
        assert!(sql.contains("AVG(value)"));
        assert!(sql.contains("GROUP BY metric"));
    }

    #[test]
    fn test_duckdb_adapter_export_query_to_parquet() {
        let adapter = DuckDbQueryAdapter::new("input.parquet");
        let inner = adapter.select_metric("cpu", 0, 1_000);
        let sql = adapter.export_query_to_parquet(&inner, "output.parquet");
        assert!(sql.starts_with("COPY ("));
        assert!(sql.contains("output.parquet"));
        assert!(sql.contains("FORMAT PARQUET"));
    }

    #[test]
    fn test_duckdb_adapter_parquet_path() {
        let adapter = DuckDbQueryAdapter::new("/mnt/data/*.parquet");
        assert_eq!(adapter.parquet_path(), "/mnt/data/*.parquet");
    }

    #[test]
    fn test_duckdb_adapter_join_metrics() {
        let adapter = DuckDbQueryAdapter::new("data.parquet");
        let sql = adapter.join_metrics("cpu", "mem", 60_000);
        assert!(sql.contains("cpu_avg"));
        assert!(sql.contains("mem_avg"));
        assert!(sql.contains("INNER JOIN"));
        assert!(sql.contains("60000"));
    }

    #[test]
    fn test_duckdb_adapter_rate_of_change() {
        let adapter = DuckDbQueryAdapter::new("data.parquet");
        let sql = adapter.rate_of_change("cpu", 1000);
        assert!(sql.contains("LAG(avg_value)"));
        assert!(sql.contains("rate_per_sec"));
        assert!(sql.contains("1000"));
    }

    #[test]
    fn test_duckdb_adapter_create_view() {
        let adapter = DuckDbQueryAdapter::new("data.parquet");
        let sql = adapter.create_view("tsdb_data");
        assert!(sql.contains("CREATE OR REPLACE VIEW tsdb_data"));
        assert!(sql.contains("read_parquet"));
    }

    #[test]
    fn test_duckdb_adapter_count_per_metric() {
        let adapter = DuckDbQueryAdapter::new("data.parquet");
        let sql = adapter.count_per_metric();
        assert!(sql.contains("COUNT(*)"));
        assert!(sql.contains("GROUP BY metric"));
        assert!(sql.contains("ORDER BY point_count DESC"));
    }

    // -- ColumnarExport -------------------------------------------------------

    #[test]
    fn test_columnar_export_new_is_empty() {
        let ce = ColumnarExport::new();
        assert!(ce.is_empty());
        assert_eq!(ce.len(), 0);
    }

    #[test]
    fn test_columnar_export_push_and_len() {
        let mut ce = ColumnarExport::new();
        ce.push(1_000, "cpu", 55.0);
        ce.push(2_000, "mem", 70.0);
        assert_eq!(ce.len(), 2);
        assert!(!ce.is_empty());
    }

    #[test]
    fn test_columnar_export_from_points() {
        let pts = vec![
            ExportedPoint::new(100, "temp", 22.0),
            ExportedPoint::new(200, "temp", 23.5),
            ExportedPoint::new(300, "pressure", 1013.0),
        ];
        let ce = ColumnarExport::from_points(&pts);
        assert_eq!(ce.len(), 3);
        assert_eq!(ce.metrics[0], "temp");
        assert_eq!(ce.metrics[2], "pressure");
        assert!((ce.values[1] - 23.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_columnar_export_to_points_roundtrip() {
        let pts = vec![
            ExportedPoint::new(10, "x", 1.0),
            ExportedPoint::new(20, "y", 2.0),
        ];
        let ce = ColumnarExport::from_points(&pts);
        let back = ce.to_points();
        assert_eq!(back.len(), pts.len());
        for (a, b) in pts.iter().zip(back.iter()) {
            assert_eq!(a.timestamp_ms, b.timestamp_ms);
            assert_eq!(a.metric, b.metric);
            assert!((a.value - b.value).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_columnar_export_csv_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_tsdb_test_columnar.csv");

        let mut ce = ColumnarExport::with_capacity(4);
        ce.push(1_000, "cpu", 10.0);
        ce.push(2_000, "mem", 20.5);
        ce.push(3_000, "disk", 30.1);

        ce.to_csv(&path).expect("csv write should succeed");
        let loaded = ColumnarExport::from_csv(&path).expect("csv read should succeed");

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.metrics[1], "mem");
        assert!((loaded.values[2] - 30.1).abs() < 1e-9);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_columnar_export_csv_with_comma_in_metric() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_tsdb_test_comma_metric.csv");

        let mut ce = ColumnarExport::new();
        ce.push(1_000, "node,A", 5.0);

        ce.to_csv(&path).expect("csv write");
        let loaded = ColumnarExport::from_csv(&path).expect("csv read");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.metrics[0], "node,A");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_columnar_export_tsv_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_tsdb_test_columnar.tsv");

        let mut ce = ColumnarExport::new();
        ce.push(100, "sensor_1", 99.9);
        ce.push(200, "sensor_2", 88.8);

        ce.to_tsv(&path).expect("tsv write");

        // Read back and verify manually (TSV is tab-separated).
        let content = std::fs::read_to_string(&path).expect("read tsv");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines[0], "timestamp_ms\tmetric\tvalue");
        assert!(lines[1].contains("sensor_1"));
        assert!(lines[2].contains("sensor_2"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_columnar_export_to_json_structure() {
        let mut ce = ColumnarExport::new();
        ce.push(1_000, "cpu", 42.0);
        ce.push(2_000, "mem", 84.0);

        let json = ce.to_json().expect("json conversion");
        let arr = json.as_array().expect("should be array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["metric"], "cpu");
        assert_eq!(arr[0]["timestamp_ms"], 1_000_i64);
        assert!((arr[0]["value"].as_f64().expect("should succeed") - 42.0).abs() < f64::EPSILON);
        assert_eq!(arr[1]["metric"], "mem");
    }

    #[test]
    fn test_columnar_export_to_json_empty() {
        let ce = ColumnarExport::new();
        let json = ce.to_json().expect("json for empty");
        let arr = json.as_array().expect("array");
        assert!(arr.is_empty());
    }

    #[test]
    fn test_columnar_export_to_json_string() {
        let mut ce = ColumnarExport::new();
        ce.push(5, "x", 1.0);
        let s = ce.to_json_string().expect("json string");
        assert!(s.contains("\"metric\""));
        assert!(s.contains("\"timestamp_ms\""));
        assert!(s.contains("\"value\""));
    }

    #[test]
    fn test_columnar_export_filter_metric() {
        let mut ce = ColumnarExport::new();
        ce.push(1, "cpu", 1.0);
        ce.push(2, "mem", 2.0);
        ce.push(3, "cpu", 3.0);

        let filtered = ce.filter_metric("cpu");
        assert_eq!(filtered.len(), 2);
        assert!(filtered.metrics.iter().all(|m| m == "cpu"));
    }

    #[test]
    fn test_columnar_export_filter_time_range() {
        let mut ce = ColumnarExport::new();
        for i in 0..10_i64 {
            ce.push(i * 100, "x", i as f64);
        }
        let filtered = ce.filter_time_range(200, 500);
        assert_eq!(filtered.len(), 4); // 200, 300, 400, 500
        assert_eq!(filtered.timestamps[0], 200);
        assert_eq!(filtered.timestamps[3], 500);
    }

    #[test]
    fn test_columnar_export_filter_time_range_empty() {
        let ce = ColumnarExport::new();
        let filtered = ce.filter_time_range(0, 1000);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_columnar_export_sort_by_timestamp() {
        let mut ce = ColumnarExport::new();
        ce.push(300, "c", 3.0);
        ce.push(100, "a", 1.0);
        ce.push(200, "b", 2.0);

        ce.sort_by_timestamp();
        assert_eq!(ce.timestamps[0], 100);
        assert_eq!(ce.metrics[0], "a");
        assert_eq!(ce.timestamps[1], 200);
        assert_eq!(ce.timestamps[2], 300);
    }

    #[test]
    fn test_columnar_export_value_stats_basic() {
        let mut ce = ColumnarExport::new();
        ce.push(1, "x", 10.0);
        ce.push(2, "x", 20.0);
        ce.push(3, "x", 30.0);

        let stats = ce.value_stats();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < f64::EPSILON);
        assert!((stats.min - 10.0).abs() < f64::EPSILON);
        assert!((stats.max - 30.0).abs() < f64::EPSILON);
        assert!((stats.sum - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_columnar_export_value_stats_empty() {
        let ce = ColumnarExport::new();
        let stats = ce.value_stats();
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_columnar_export_value_stats_single() {
        let mut ce = ColumnarExport::new();
        ce.push(1, "x", 7.0);
        let stats = ce.value_stats();
        assert_eq!(stats.count, 1);
        assert!((stats.mean - 7.0).abs() < f64::EPSILON);
        assert!((stats.variance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_columnar_export_with_capacity() {
        let ce = ColumnarExport::with_capacity(100);
        assert!(ce.is_empty());
        assert_eq!(ce.len(), 0);
    }

    #[test]
    fn test_columnar_export_csv_empty_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_tsdb_test_empty.csv");

        let ce = ColumnarExport::new();
        ce.to_csv(&path).expect("csv write empty");
        let loaded = ColumnarExport::from_csv(&path).expect("csv read empty");
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_columnar_export_tsv_multiple_metrics() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_tsdb_multi_metric.tsv");

        let mut ce = ColumnarExport::new();
        for i in 0..5_i64 {
            ce.push(i * 1000, "cpu", i as f64 * 10.0);
            ce.push(i * 1000 + 1, "mem", i as f64 * 20.0);
        }

        ce.to_tsv(&path).expect("tsv write");
        let content = std::fs::read_to_string(&path).expect("tsv read");
        let lines: Vec<&str> = content.lines().collect();
        // header + 10 data rows
        assert_eq!(lines.len(), 11);
        let _ = std::fs::remove_file(&path);
    }
}
