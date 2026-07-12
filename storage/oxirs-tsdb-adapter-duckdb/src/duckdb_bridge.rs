//! DuckDB ↔ TSDB chunk bridge.
//!
//! This module lives in the `oxirs-tsdb-adapter-duckdb` quarantine crate
//! (`publish = false`) so the C `libduckdb-sys` dependency it pulls never
//! reaches the published `oxirs-tsdb` Pure-Rust surface (COOLJAPAN Pure Rust
//! Policy v2). It provides zero-copy interoperability between TSDB time chunks
//! and an embedded DuckDB connection, using Apache Arrow `RecordBatch` as the
//! transport format.
//!
//! ## Direction overview
//!
//! ```text
//! ┌─────────────┐  decompress + Arrow batch   ┌────────────────────┐
//! │  TimeChunk  │ ──────────────────────────► │  DuckDB table      │
//! └─────────────┘   (register_tsdb_chunk)     └────────────────────┘
//!                                                       │
//!                                                       │ SQL
//!                                                       ▼
//! ┌─────────────┐    Arrow RecordBatch back   ┌────────────────────┐
//! │  TimeChunk  │ ◄────────────────────────── │  query_arrow / SQL │
//! └─────────────┘   (read_into_chunk)         └────────────────────┘
//! ```
//!
//! The bridge uses the workspace-aligned `arrow` 58 crate that DuckDB also
//! consumes via its `vtab-arrow` and `appender-arrow` features, so the
//! `RecordBatch` value can be moved between the two sides without an IPC
//! round-trip.
//!
//! ## Wire schema
//!
//! Every Arrow batch produced or consumed by this bridge has exactly three
//! columns (in this order):
//!
//! | column      | Arrow type | DuckDB type | meaning                     |
//! |-------------|------------|-------------|-----------------------------|
//! | `timestamp` | Int64      | BIGINT      | Unix epoch milliseconds     |
//! | `value`     | Float64    | DOUBLE      | observed measurement        |
//! | `series_id` | Utf8       | VARCHAR     | logical series id           |
//!
//! `register_tsdb_chunk` creates a table that follows this schema and bulk
//! loads the chunk via DuckDB's Arrow appender; `read_into_chunk` runs the
//! supplied SQL, expects results in the same schema, and rebuilds a
//! [`TimeChunk`] using the existing TSDB compression path.

use std::sync::Arc;

// Route ALL Arrow usage through duckdb's re-export (`duckdb::arrow`, arrow 58.3.0)
// so the crate shares duckdb's exact arrow version — see Cargo.toml note.
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use duckdb::arrow::array::{Array, Float64Array, Int64Array, StringArray};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;
use duckdb::Connection;

use oxirs_tsdb::{DataPoint, TimeChunk, TsdbError, TsdbResult};

// =============================================================================
// Constants
// =============================================================================

/// Standard column name for the timestamp column in TSDB↔DuckDB transfer.
pub const TS_COL: &str = "timestamp";
/// Standard column name for the value column.
pub const VAL_COL: &str = "value";
/// Standard column name for the series identifier column.
pub const SERIES_COL: &str = "series_id";

// =============================================================================
// Helpers
// =============================================================================

/// Build the canonical Arrow [`Schema`] used by the bridge.
fn bridge_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(TS_COL, DataType::Int64, false),
        Field::new(VAL_COL, DataType::Float64, false),
        Field::new(SERIES_COL, DataType::Utf8, false),
    ]))
}

/// Decompress a [`TimeChunk`] into an Arrow [`RecordBatch`] following the
/// canonical bridge schema.
///
/// `series_label` is what gets written into the `series_id` column for every
/// row. Pass `chunk.series_id` rendered as a string when you want to preserve
/// the original numeric id.
pub fn chunk_to_record_batch(chunk: &TimeChunk, series_label: &str) -> TsdbResult<RecordBatch> {
    let points = chunk.decompress()?;
    points_to_record_batch(&points, series_label)
}

/// Convert a slice of [`DataPoint`]s plus a series label to an Arrow
/// [`RecordBatch`] with the canonical bridge schema.
pub fn points_to_record_batch(points: &[DataPoint], series_label: &str) -> TsdbResult<RecordBatch> {
    let timestamps: Int64Array = points
        .iter()
        .map(|p| p.timestamp.timestamp_millis())
        .collect();
    let values: Float64Array = points.iter().map(|p| p.value).collect();
    let labels: StringArray = (0..points.len()).map(|_| Some(series_label)).collect();

    RecordBatch::try_new(
        bridge_schema(),
        vec![Arc::new(timestamps), Arc::new(values), Arc::new(labels)],
    )
    .map_err(|e| TsdbError::Arrow(e.to_string()))
}

/// Pull `(timestamp_ms, value, series_label)` triples from a slice of
/// [`RecordBatch`]es returned by DuckDB.
///
/// Validates that each batch matches the canonical schema. Returns the
/// concatenation of all rows together with the series label encountered in
/// the first row (used by [`read_into_chunk`] to construct a [`TimeChunk`]).
pub fn record_batches_to_points(
    batches: &[RecordBatch],
) -> TsdbResult<(Vec<DataPoint>, Option<String>)> {
    let mut points: Vec<DataPoint> = Vec::new();
    let mut series_label: Option<String> = None;

    for batch in batches {
        if batch.num_columns() != 3 {
            return Err(TsdbError::Arrow(format!(
                "expected 3 columns in DuckDB result, got {}",
                batch.num_columns()
            )));
        }

        let ts_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TsdbError::Arrow("first column must be Int64".into()))?;
        let val_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| TsdbError::Arrow("second column must be Float64".into()))?;
        let lbl_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| TsdbError::Arrow("third column must be Utf8".into()))?;

        for i in 0..batch.num_rows() {
            if ts_col.is_null(i) || val_col.is_null(i) {
                continue;
            }
            let ts_ms = ts_col.value(i);
            let value = val_col.value(i);
            let dt = DateTime::<Utc>::from_timestamp_millis(ts_ms).ok_or_else(|| {
                TsdbError::Query(format!("invalid timestamp_ms in DuckDB result: {ts_ms}"))
            })?;
            if !lbl_col.is_null(i) && series_label.is_none() {
                series_label = Some(lbl_col.value(i).to_string());
            }
            points.push(DataPoint {
                timestamp: dt,
                value,
            });
        }
    }

    Ok((points, series_label))
}

// =============================================================================
// Bridge configuration
// =============================================================================

/// Options used by [`register_tsdb_chunk`] when materialising a chunk into a
/// DuckDB table.
#[derive(Debug, Clone)]
pub struct RegisterOptions {
    /// Whether to drop an existing table with the same name before creating
    /// the new one. Defaults to `true`.
    pub drop_if_exists: bool,
    /// String label written into the `series_id` column. When `None` the
    /// numeric chunk id is used.
    pub series_label: Option<String>,
}

impl Default for RegisterOptions {
    fn default() -> Self {
        Self {
            drop_if_exists: true,
            series_label: None,
        }
    }
}

/// Options used by [`read_into_chunk`] when re-ingesting a SQL result back
/// into a [`TimeChunk`].
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// Series identifier for the rebuilt chunk.
    pub series_id: u64,
    /// Chunk duration that gets passed to [`TimeChunk::new`]. Defaults to
    /// 2 hours, matching the engine default.
    pub chunk_duration: ChronoDuration,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            series_id: 0,
            chunk_duration: ChronoDuration::hours(2),
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Register a TSDB [`TimeChunk`] inside a DuckDB connection as a regular
/// table named `table_name`.
///
/// The chunk is decompressed and bulk-loaded through DuckDB's Arrow appender
/// API, giving zero-copy ingestion at the column level.
///
/// Returns the row count actually loaded.
pub fn register_tsdb_chunk(
    conn: &Connection,
    table_name: &str,
    chunk: &TimeChunk,
    opts: &RegisterOptions,
) -> TsdbResult<usize> {
    validate_table_name(table_name)?;

    let label = opts
        .series_label
        .clone()
        .unwrap_or_else(|| chunk.series_id.to_string());
    let batch = chunk_to_record_batch(chunk, &label)?;

    if opts.drop_if_exists {
        conn.execute_batch(&format!("DROP TABLE IF EXISTS {table_name};"))
            .map_err(|e| TsdbError::Query(format!("DuckDB DROP failed: {e}")))?;
    }
    conn.execute_batch(&format!(
        "CREATE TABLE {table_name} ({TS_COL} BIGINT, {VAL_COL} DOUBLE, {SERIES_COL} VARCHAR);"
    ))
    .map_err(|e| TsdbError::Query(format!("DuckDB CREATE TABLE failed: {e}")))?;

    let row_count = batch.num_rows();
    if row_count > 0 {
        let mut appender = conn
            .appender(table_name)
            .map_err(|e| TsdbError::Query(format!("DuckDB appender open failed: {e}")))?;
        appender
            .append_record_batch(batch)
            .map_err(|e| TsdbError::Query(format!("DuckDB appender failed: {e}")))?;
    }
    Ok(row_count)
}

/// Run the supplied `sql` against `conn`, expect the result to follow the
/// canonical bridge schema, and rebuild a [`TimeChunk`] from the rows.
///
/// The `sql` query MUST project columns in `(timestamp, value, series_id)`
/// order (or use `SELECT * FROM <table>` against a table created by
/// [`register_tsdb_chunk`]).
///
/// When the SQL returns zero rows, an empty result is returned.
pub fn read_into_chunk(
    conn: &Connection,
    sql: &str,
    opts: &ReadOptions,
) -> TsdbResult<Option<TimeChunk>> {
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| TsdbError::Query(format!("DuckDB prepare failed: {e}")))?;
    let arrow_iter = stmt
        .query_arrow([])
        .map_err(|e| TsdbError::Query(format!("DuckDB query_arrow failed: {e}")))?;
    let batches: Vec<RecordBatch> = arrow_iter.collect();

    let (mut points, _label) = record_batches_to_points(&batches)?;
    if points.is_empty() {
        return Ok(None);
    }

    points.sort_by_key(|p| p.timestamp.timestamp_millis());
    let start_time = points
        .first()
        .ok_or_else(|| TsdbError::Query("non-empty point list lost first element".into()))?
        .timestamp;

    let chunk = TimeChunk::new(opts.series_id, start_time, opts.chunk_duration, points)?;
    Ok(Some(chunk))
}

/// Run an arbitrary SQL statement that does not project the canonical schema
/// and return the raw Arrow result batches.
///
/// This is useful for inspection / aggregation queries that do not need to be
/// re-ingested back into a chunk (e.g., `SELECT AVG(value) FROM …`).
pub fn run_sql(conn: &Connection, sql: &str) -> TsdbResult<Vec<RecordBatch>> {
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| TsdbError::Query(format!("DuckDB prepare failed: {e}")))?;
    let arrow_iter = stmt
        .query_arrow([])
        .map_err(|e| TsdbError::Query(format!("DuckDB query_arrow failed: {e}")))?;
    Ok(arrow_iter.collect())
}

/// Open a fresh in-memory DuckDB connection.
///
/// Convenience helper for callers that don't already manage a connection
/// pool. Returns a typed [`TsdbError`] on failure to match the rest of the
/// crate's error conventions.
pub fn open_in_memory() -> TsdbResult<Connection> {
    Connection::open_in_memory()
        .map_err(|e| TsdbError::Query(format!("DuckDB open_in_memory failed: {e}")))
}

// =============================================================================
// Internal validation
// =============================================================================

/// Reject names that would let a caller smuggle SQL into [`register_tsdb_chunk`].
fn validate_table_name(name: &str) -> TsdbResult<()> {
    if name.is_empty() {
        return Err(TsdbError::Config("table name must not be empty".into()));
    }
    let valid = name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err(TsdbError::Config(format!(
            "invalid table name '{name}' (allowed: ASCII alphanumeric and underscore)"
        )));
    }
    let first = name
        .chars()
        .next()
        .ok_or_else(|| TsdbError::Config("table name must not be empty".into()))?;
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(TsdbError::Config(format!(
            "table name '{name}' must start with a letter or underscore"
        )));
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn sample_chunk() -> TimeChunk {
        let start = Utc.with_ymd_and_hms(2026, 4, 30, 12, 0, 0).unwrap();
        let mut points = Vec::new();
        for i in 0..50_i64 {
            points.push(DataPoint {
                timestamp: start + Duration::seconds(i),
                value: 20.0 + (i as f64) * 0.5,
            });
        }
        TimeChunk::new(7, start, Duration::hours(2), points).expect("chunk")
    }

    #[test]
    fn chunk_to_arrow_batch_shape() {
        let chunk = sample_chunk();
        let batch = chunk_to_record_batch(&chunk, "temp.sensor").expect("batch");
        assert_eq!(batch.num_rows(), 50);
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(0).name(), TS_COL);
        assert_eq!(batch.schema().field(1).name(), VAL_COL);
        assert_eq!(batch.schema().field(2).name(), SERIES_COL);
    }

    #[test]
    fn validate_table_name_accepts_simple() {
        assert!(validate_table_name("foo").is_ok());
        assert!(validate_table_name("_x").is_ok());
        assert!(validate_table_name("a_b_2").is_ok());
    }

    #[test]
    fn validate_table_name_rejects_dangerous() {
        assert!(validate_table_name("").is_err());
        assert!(validate_table_name("foo;DROP").is_err());
        assert!(validate_table_name("1abc").is_err());
        assert!(validate_table_name("hi'there").is_err());
        assert!(validate_table_name("a b").is_err());
    }

    #[test]
    fn record_batches_to_points_handles_empty() {
        let batches: Vec<RecordBatch> = vec![];
        let (pts, lbl) = record_batches_to_points(&batches).expect("ok");
        assert!(pts.is_empty());
        assert!(lbl.is_none());
    }

    #[test]
    fn points_to_record_batch_schema_consistency() {
        let pts = vec![
            DataPoint {
                timestamp: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
                value: 1.0,
            },
            DataPoint {
                timestamp: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 1).unwrap(),
                value: 2.0,
            },
        ];
        let batch = points_to_record_batch(&pts, "label").expect("batch");
        let lbl_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string col");
        assert_eq!(lbl_col.value(0), "label");
        assert_eq!(lbl_col.value(1), "label");
    }

    #[test]
    fn open_in_memory_works() {
        let conn = open_in_memory().expect("open");
        // Sanity: simple SQL.
        conn.execute_batch("SELECT 1;").expect("select 1");
    }

    #[test]
    fn register_and_read_round_trip() {
        let chunk = sample_chunk();
        let conn = open_in_memory().expect("open");
        let n = register_tsdb_chunk(
            &conn,
            "tsdb_round_trip",
            &chunk,
            &RegisterOptions::default(),
        )
        .expect("register");
        assert_eq!(n, 50);

        let opts = ReadOptions {
            series_id: chunk.series_id,
            chunk_duration: Duration::hours(2),
        };
        let rebuilt = read_into_chunk(
            &conn,
            "SELECT timestamp, value, series_id FROM tsdb_round_trip ORDER BY timestamp",
            &opts,
        )
        .expect("read")
        .expect("rebuilt chunk");

        let original_points = chunk.decompress().expect("decompress original");
        let rebuilt_points = rebuilt.decompress().expect("decompress rebuilt");
        assert_eq!(original_points.len(), rebuilt_points.len());
        for (a, b) in original_points.iter().zip(rebuilt_points.iter()) {
            assert_eq!(
                a.timestamp.timestamp_millis(),
                b.timestamp.timestamp_millis()
            );
            assert!((a.value - b.value).abs() < 1e-9);
        }
    }

    #[test]
    fn read_into_chunk_returns_none_for_empty_result() {
        let chunk = sample_chunk();
        let conn = open_in_memory().expect("open");
        register_tsdb_chunk(&conn, "empty_test", &chunk, &RegisterOptions::default())
            .expect("register");
        let opts = ReadOptions::default();
        let result = read_into_chunk(
            &conn,
            "SELECT timestamp, value, series_id FROM empty_test WHERE value < 0",
            &opts,
        )
        .expect("read");
        assert!(result.is_none());
    }

    #[test]
    fn run_sql_returns_arbitrary_shape() {
        let chunk = sample_chunk();
        let conn = open_in_memory().expect("open");
        register_tsdb_chunk(&conn, "agg_test", &chunk, &RegisterOptions::default())
            .expect("register");
        let batches = run_sql(
            &conn,
            "SELECT count(*) AS n, avg(value) AS mean FROM agg_test",
        )
        .expect("agg");
        assert!(!batches.is_empty());
        let b0 = &batches[0];
        assert_eq!(b0.num_rows(), 1);
        assert_eq!(b0.num_columns(), 2);
    }

    #[test]
    fn read_into_chunk_filter_subset() {
        let chunk = sample_chunk();
        let conn = open_in_memory().expect("open");
        register_tsdb_chunk(&conn, "filter_test", &chunk, &RegisterOptions::default())
            .expect("register");

        let opts = ReadOptions {
            series_id: chunk.series_id,
            chunk_duration: Duration::hours(2),
        };
        let rebuilt = read_into_chunk(
            &conn,
            "SELECT timestamp, value, series_id FROM filter_test WHERE value >= 30 ORDER BY timestamp",
            &opts,
        )
        .expect("read")
        .expect("non-empty");
        let pts = rebuilt.decompress().expect("decompress");
        assert!(!pts.is_empty());
        for p in &pts {
            assert!(p.value >= 30.0);
        }
    }

    #[test]
    fn register_with_explicit_label() {
        let chunk = sample_chunk();
        let conn = open_in_memory().expect("open");
        let opts = RegisterOptions {
            drop_if_exists: true,
            series_label: Some("custom.label".into()),
        };
        register_tsdb_chunk(&conn, "lbl_test", &chunk, &opts).expect("register");
        let batches = run_sql(&conn, "SELECT DISTINCT series_id FROM lbl_test").expect("run");
        let b = &batches[0];
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string col");
        assert_eq!(col.value(0), "custom.label");
    }

    #[test]
    fn register_drop_if_exists_replaces_table() {
        let chunk = sample_chunk();
        let conn = open_in_memory().expect("open");
        register_tsdb_chunk(&conn, "drop_test", &chunk, &RegisterOptions::default())
            .expect("first");
        // Re-register replaces; should not error.
        register_tsdb_chunk(&conn, "drop_test", &chunk, &RegisterOptions::default())
            .expect("second");
        let batches = run_sql(&conn, "SELECT count(*) FROM drop_test").expect("count");
        let n = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64")
            .value(0);
        assert_eq!(n, 50);
    }
}
