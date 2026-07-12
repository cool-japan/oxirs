//! `oxirs tsdb duckdb` — inspect TSDB chunks via embedded DuckDB SQL.
//!
//! This subcommand decompresses a serialised TSDB chunk, registers it as a
//! `tsdb_chunk` table inside an in-memory DuckDB connection, and runs the
//! user-supplied SQL.
//!
//! Build with the `tsdb-duckdb` feature flag:
//!
//! ```bash
//! cargo build -p oxirs --features tsdb-duckdb
//! ```
//!
//! The whole module is gated behind `#[cfg(feature = "tsdb-duckdb")]`; it
//! never enters a default build because DuckDB pulls C dependencies that
//! violate the COOLJAPAN Pure Rust default policy.

#![cfg(feature = "tsdb-duckdb")]

use std::path::PathBuf;

use anyhow::{Context, Result};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use oxirs_tsdb::TimeChunk;
use oxirs_tsdb_adapter_duckdb::{
    duckdb_open_in_memory, duckdb_run_sql, register_tsdb_chunk, DuckDbRegisterOptions,
};

/// Run the `oxirs tsdb duckdb` subcommand.
pub async fn run(
    chunk: PathBuf,
    sql: String,
    series_label: Option<String>,
    format: String,
) -> Result<()> {
    if !chunk.exists() {
        anyhow::bail!("chunk path does not exist: {}", chunk.display());
    }

    // Read and deserialize the chunk. We accept either a JSON-encoded chunk
    // (the `serde_json` form used by `TimeChunk` / `serde::Serialize`) or a
    // raw binary chunk. JSON is preferred because the rest of `oxirs-tsdb`
    // ships chunks via JSON over the wire.
    let bytes = std::fs::read(&chunk)
        .with_context(|| format!("could not read chunk file: {}", chunk.display()))?;
    let chunk: TimeChunk = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to parse chunk as JSON; expected a `TimeChunk` produced by oxirs-tsdb. \
             Path: {}",
            chunk.display()
        )
    })?;

    let conn = duckdb_open_in_memory().context("could not open in-memory DuckDB connection")?;
    let opts = DuckDbRegisterOptions {
        drop_if_exists: true,
        series_label,
    };
    let row_count = register_tsdb_chunk(&conn, "tsdb_chunk", &chunk, &opts)
        .context("could not register chunk as DuckDB table")?;
    eprintln!(
        "registered {} rows in DuckDB table `tsdb_chunk` (series_id={}, range={}..{})",
        row_count,
        chunk.series_id,
        chunk.start_time.to_rfc3339(),
        chunk.end_time.to_rfc3339(),
    );

    let batches = duckdb_run_sql(&conn, &sql).context("DuckDB query failed")?;

    match format.to_lowercase().as_str() {
        "table" => print_table(&batches)?,
        "csv" => print_csv(&batches)?,
        "json" => print_json(&batches)?,
        other => {
            anyhow::bail!("unsupported output format '{other}' (allowed: table, csv, json)");
        }
    }
    Ok(())
}

// ─── output formats ──────────────────────────────────────────────────────────

fn print_table(batches: &[RecordBatch]) -> Result<()> {
    if batches.is_empty() {
        println!("(empty result)");
        return Ok(());
    }
    let formatted =
        pretty_format_batches(batches).context("could not pretty-print Arrow result batches")?;
    println!("{formatted}");
    Ok(())
}

fn print_csv(batches: &[RecordBatch]) -> Result<()> {
    if batches.is_empty() {
        return Ok(());
    }
    let schema = batches[0].schema();
    let header: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    println!("{}", header.join(","));
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut cells: Vec<String> = Vec::with_capacity(batch.num_columns());
            for col in 0..batch.num_columns() {
                cells.push(format_cell(batch.column(col), row));
            }
            println!("{}", cells.join(","));
        }
    }
    Ok(())
}

fn print_json(batches: &[RecordBatch]) -> Result<()> {
    let mut rows: Vec<serde_json::Value> = Vec::new();
    for batch in batches {
        let schema = batch.schema();
        for row in 0..batch.num_rows() {
            let mut obj = serde_json::Map::new();
            for col in 0..batch.num_columns() {
                let name = schema.field(col).name().clone();
                obj.insert(name, json_cell(batch.column(col), row));
            }
            rows.push(serde_json::Value::Object(obj));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Array(rows))
            .context("could not serialise rows to JSON")?
    );
    Ok(())
}

fn format_cell(col: &dyn Array, row: usize) -> String {
    if col.is_null(row) {
        return String::new();
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
        return arr.value(row).to_string();
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
        return arr.value(row).to_string();
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
        let s = arr.value(row);
        if s.contains(',') || s.contains('"') {
            return format!("\"{}\"", s.replace('"', "\"\""));
        }
        return s.to_string();
    }
    "?".to_string()
}

fn json_cell(col: &dyn Array, row: usize) -> serde_json::Value {
    if col.is_null(row) {
        return serde_json::Value::Null;
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
        return serde_json::Value::from(arr.value(row));
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
        return serde_json::json!(arr.value(row));
    }
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
        return serde_json::Value::String(arr.value(row).to_string());
    }
    serde_json::Value::String(format_cell(col, row))
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use oxirs_tsdb::{DataPoint, TimeChunk};
    use std::env;

    fn make_chunk_file(name: &str) -> PathBuf {
        let start = Utc.with_ymd_and_hms(2026, 4, 30, 0, 0, 0).unwrap();
        let mut points = Vec::new();
        for i in 0..20_i64 {
            points.push(DataPoint {
                timestamp: start + Duration::seconds(i),
                value: 1.0 + (i as f64) * 0.5,
            });
        }
        let chunk = TimeChunk::new(99, start, Duration::hours(2), points).expect("chunk");
        let path = env::temp_dir().join(name);
        let bytes = serde_json::to_vec(&chunk).expect("serialise");
        std::fs::write(&path, bytes).expect("write");
        path
    }

    #[tokio::test]
    async fn duckdb_subcommand_runs_simple_count() {
        let path = make_chunk_file("oxirs_tsdb_duckdb_test_1.json");
        let res = run(
            path.clone(),
            "SELECT count(*) AS n FROM tsdb_chunk".into(),
            Some("custom.label".into()),
            "csv".into(),
        )
        .await;
        let _ = std::fs::remove_file(&path);
        res.expect("subcommand should succeed");
    }

    #[tokio::test]
    async fn duckdb_subcommand_rejects_unknown_format() {
        let path = make_chunk_file("oxirs_tsdb_duckdb_test_fmt.json");
        let res = run(path.clone(), "SELECT 1".into(), None, "yaml".into()).await;
        let _ = std::fs::remove_file(&path);
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn duckdb_subcommand_rejects_missing_path() {
        let path = env::temp_dir().join("oxirs_tsdb_duckdb_does_not_exist_xyz.json");
        let res = run(path, "SELECT 1".into(), None, "table".into()).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn duckdb_subcommand_rejects_invalid_chunk() {
        let path = env::temp_dir().join("oxirs_tsdb_duckdb_garbage.json");
        std::fs::write(&path, b"not a chunk").expect("write");
        let res = run(path.clone(), "SELECT 1".into(), None, "table".into()).await;
        let _ = std::fs::remove_file(&path);
        assert!(res.is_err());
    }
}
