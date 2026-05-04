//! Integration tests for the DuckDB ↔ TSDB chunk bridge.
//!
//! These tests are gated on the `duckdb` Cargo feature. Build with:
//!
//! ```bash
//! cargo nextest run -p oxirs-tsdb --features duckdb
//! ```
//!
//! When the C build of `libduckdb` is unavailable on a host (CI workers
//! without a C toolchain, etc.) these tests are simply not compiled because
//! the entire file is `#[cfg(feature = "duckdb")]`-gated.

#![cfg(feature = "duckdb")]

use chrono::{Duration, TimeZone, Utc};
use oxirs_tsdb::duckdb_bridge::{SERIES_COL, TS_COL, VAL_COL};
use oxirs_tsdb::{
    chunk_to_record_batch, duckdb_open_in_memory, duckdb_run_sql, points_to_record_batch,
    read_into_chunk, register_tsdb_chunk, DataPoint, DuckDbReadOptions, DuckDbRegisterOptions,
    TimeChunk,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_chunk(series_id: u64, count: i64) -> TimeChunk {
    let start = Utc.with_ymd_and_hms(2026, 4, 30, 12, 0, 0).unwrap();
    let mut points = Vec::new();
    for i in 0..count {
        points.push(DataPoint {
            timestamp: start + Duration::seconds(i),
            value: 10.0 + (i as f64) * 0.25,
        });
    }
    TimeChunk::new(series_id, start, Duration::hours(2), points).expect("chunk")
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[test]
fn chunk_to_arrow_batch_has_canonical_schema() {
    let chunk = make_chunk(1, 100);
    let batch = chunk_to_record_batch(&chunk, "metrics.test").expect("batch");
    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(batch.schema().field(0).name(), TS_COL);
    assert_eq!(batch.schema().field(1).name(), VAL_COL);
    assert_eq!(batch.schema().field(2).name(), SERIES_COL);
}

#[test]
fn duckdb_register_and_count_rows() {
    let chunk = make_chunk(2, 50);
    let conn = duckdb_open_in_memory().expect("open");
    let n = register_tsdb_chunk(
        &conn,
        "tsdb_count",
        &chunk,
        &DuckDbRegisterOptions::default(),
    )
    .expect("register");
    assert_eq!(n, 50);
    let batches = duckdb_run_sql(&conn, "SELECT count(*) FROM tsdb_count").expect("count");
    assert_eq!(batches.len(), 1);
    let arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("i64");
    assert_eq!(arr.value(0), 50);
}

#[test]
fn full_round_trip_through_duckdb() {
    let chunk = make_chunk(3, 100);
    let conn = duckdb_open_in_memory().expect("open");
    register_tsdb_chunk(
        &conn,
        "tsdb_round_trip",
        &chunk,
        &DuckDbRegisterOptions::default(),
    )
    .expect("register");
    let opts = DuckDbReadOptions {
        series_id: chunk.series_id,
        chunk_duration: Duration::hours(2),
    };
    let rebuilt = read_into_chunk(
        &conn,
        "SELECT timestamp, value, series_id FROM tsdb_round_trip ORDER BY timestamp",
        &opts,
    )
    .expect("read")
    .expect("non-empty");

    let original = chunk.decompress().expect("orig");
    let restored = rebuilt.decompress().expect("rest");
    assert_eq!(original.len(), restored.len());
    for (a, b) in original.iter().zip(restored.iter()) {
        assert_eq!(
            a.timestamp.timestamp_millis(),
            b.timestamp.timestamp_millis()
        );
        assert!((a.value - b.value).abs() < 1e-9);
    }
}

#[test]
fn duckdb_aggregate_via_sql_returns_arrow() {
    let chunk = make_chunk(4, 100);
    let conn = duckdb_open_in_memory().expect("open");
    register_tsdb_chunk(&conn, "agg_test", &chunk, &DuckDbRegisterOptions::default())
        .expect("register");
    let batches = duckdb_run_sql(
        &conn,
        "SELECT count(*) AS n, avg(value) AS mean, min(value) AS lo, max(value) AS hi FROM agg_test",
    )
    .expect("agg");
    assert_eq!(batches.len(), 1);
    let b0 = &batches[0];
    assert_eq!(b0.num_rows(), 1);
    assert_eq!(b0.num_columns(), 4);
    let n = b0
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("i64")
        .value(0);
    assert_eq!(n, 100);
    let mean = b0
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .expect("f64")
        .value(0);
    // mean of 10, 10.25, …, 10 + 99·0.25 = 10 + 24.75/2 ≈ 22.375
    assert!((mean - 22.375).abs() < 1e-9, "mean={mean}");
}

#[test]
fn duckdb_filter_subset_returns_correct_chunk() {
    let chunk = make_chunk(5, 100);
    let conn = duckdb_open_in_memory().expect("open");
    register_tsdb_chunk(
        &conn,
        "filter_test",
        &chunk,
        &DuckDbRegisterOptions::default(),
    )
    .expect("register");

    let opts = DuckDbReadOptions {
        series_id: chunk.series_id,
        chunk_duration: Duration::hours(2),
    };
    let rebuilt = read_into_chunk(
        &conn,
        "SELECT timestamp, value, series_id FROM filter_test WHERE value >= 20 ORDER BY timestamp",
        &opts,
    )
    .expect("read")
    .expect("non-empty");
    let pts = rebuilt.decompress().expect("decompress");
    assert!(!pts.is_empty());
    for p in &pts {
        assert!(p.value >= 20.0);
    }
}

#[test]
fn duckdb_empty_filter_returns_none() {
    let chunk = make_chunk(6, 50);
    let conn = duckdb_open_in_memory().expect("open");
    register_tsdb_chunk(
        &conn,
        "empty_test",
        &chunk,
        &DuckDbRegisterOptions::default(),
    )
    .expect("register");
    let opts = DuckDbReadOptions::default();
    let result = read_into_chunk(
        &conn,
        "SELECT timestamp, value, series_id FROM empty_test WHERE value < 0",
        &opts,
    )
    .expect("read");
    assert!(result.is_none());
}

#[test]
fn duckdb_resample_via_sql() {
    let chunk = make_chunk(7, 1_200);
    let conn = duckdb_open_in_memory().expect("open");
    register_tsdb_chunk(&conn, "resample", &chunk, &DuckDbRegisterOptions::default())
        .expect("register");
    let batches = duckdb_run_sql(
        &conn,
        "SELECT (timestamp / 60000) * 60000 AS bucket_ms, AVG(value) AS avg_v \
         FROM resample GROUP BY bucket_ms ORDER BY bucket_ms",
    )
    .expect("resample");
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0);
}

#[test]
fn points_to_record_batch_with_explicit_label() {
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
    let batch = points_to_record_batch(&pts, "explicit").expect("batch");
    assert_eq!(batch.num_rows(), 2);
    let label_col = batch
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("string");
    assert_eq!(label_col.value(0), "explicit");
    assert_eq!(label_col.value(1), "explicit");
}

#[test]
fn duckdb_multiple_chunks_in_one_connection() {
    let chunk_a = make_chunk(10, 30);
    let chunk_b = make_chunk(11, 20);
    let conn = duckdb_open_in_memory().expect("open");
    register_tsdb_chunk(
        &conn,
        "multi_a",
        &chunk_a,
        &DuckDbRegisterOptions::default(),
    )
    .expect("a");
    register_tsdb_chunk(
        &conn,
        "multi_b",
        &chunk_b,
        &DuckDbRegisterOptions::default(),
    )
    .expect("b");
    let batches = duckdb_run_sql(
        &conn,
        "SELECT (SELECT count(*) FROM multi_a) AS a_count, \
                (SELECT count(*) FROM multi_b) AS b_count",
    )
    .expect("count");
    let row = &batches[0];
    let a_count = row
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("i64")
        .value(0);
    let b_count = row
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("i64")
        .value(0);
    assert_eq!(a_count, 30);
    assert_eq!(b_count, 20);
}

#[test]
fn duckdb_register_with_explicit_label_visible_in_query() {
    let chunk = make_chunk(20, 5);
    let conn = duckdb_open_in_memory().expect("open");
    let opts = DuckDbRegisterOptions {
        drop_if_exists: true,
        series_label: Some("tagged.label".into()),
    };
    register_tsdb_chunk(&conn, "tagged", &chunk, &opts).expect("register");
    let batches = duckdb_run_sql(&conn, "SELECT DISTINCT series_id FROM tagged").expect("run");
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("string");
    assert_eq!(col.value(0), "tagged.label");
}

#[test]
fn duckdb_invalid_table_name_rejected() {
    let chunk = make_chunk(30, 10);
    let conn = duckdb_open_in_memory().expect("open");
    let err = register_tsdb_chunk(
        &conn,
        "bad name; DROP TABLE foo;",
        &chunk,
        &DuckDbRegisterOptions::default(),
    )
    .err();
    assert!(
        err.is_some(),
        "register should reject malicious table names"
    );
}

#[test]
fn duckdb_replace_existing_table() {
    let conn = duckdb_open_in_memory().expect("open");
    let chunk1 = make_chunk(40, 25);
    let chunk2 = make_chunk(40, 75);
    register_tsdb_chunk(&conn, "replace", &chunk1, &DuckDbRegisterOptions::default())
        .expect("first");
    register_tsdb_chunk(&conn, "replace", &chunk2, &DuckDbRegisterOptions::default())
        .expect("second");
    let batches = duckdb_run_sql(&conn, "SELECT count(*) FROM replace").expect("count");
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("i64")
        .value(0);
    assert_eq!(n, 75);
}
