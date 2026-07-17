//! Time-series database CLI commands
//!
//! Provides CLI commands for:
//! - Querying time-series data (direct range queries + local aggregation)
//! - Inserting data points (single value or CSV batch)
//! - Viewing real compression statistics from the on-disk series index
//! - Exporting time-series to CSV
//! - Micro-benchmarking real write throughput
//!
//! All commands are backed by `oxirs_tsdb::ColumnarStore`, a real disk-backed,
//! Gorilla/delta-of-delta-compressed columnar time-series store rooted at
//! `<dataset>/tsdb/`. Commands that are not yet wired to a real backend
//! (chunk compaction, retention-policy enforcement, and SPARQL temporal
//! extensions) fail loudly with an explicit error instead of silently
//! reporting fake success.

use crate::cli::CliContext;
use crate::cli_actions::{RetentionAction, TsdbAction};
use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use colored::Colorize;
use oxirs_tsdb::query::aggregate::Aggregator;
use oxirs_tsdb::{Aggregation, ColumnarStore, DataPoint, TimeChunk};
use prettytable::{row, Table};
use std::path::{Path, PathBuf};

/// Default chunk duration used for CLI-driven inserts/benchmarks.
const CHUNK_DURATION: ChronoDuration = ChronoDuration::hours(2);
/// In-memory chunk cache size for the CLI-opened store.
const CACHE_SIZE: usize = 128;

/// Execute time-series database command
pub async fn execute(action: TsdbAction, _ctx: &CliContext) -> Result<()> {
    match action {
        TsdbAction::Query {
            dataset,
            series,
            start,
            end,
            sparql,
            aggregate,
            format,
        } => query_command(dataset, series, start, end, sparql, aggregate, format).await,
        TsdbAction::Insert {
            dataset,
            series,
            timestamp,
            value,
            from_csv,
        } => insert_command(dataset, series, timestamp, value, from_csv).await,
        TsdbAction::Stats {
            dataset,
            series,
            detailed,
        } => stats_command(dataset, series, detailed).await,
        TsdbAction::Compact {
            dataset,
            series,
            force,
        } => compact_command(dataset, series, force).await,
        TsdbAction::Retention { action } => retention_command(action).await,
        TsdbAction::Export {
            dataset,
            series,
            output,
            format,
            start,
            end,
        } => export_command(dataset, series, output, format, start, end).await,
        TsdbAction::Benchmark {
            dataset,
            points,
            series_count,
        } => benchmark_command(dataset, points, series_count).await,
        #[cfg(feature = "tsdb-duckdb")]
        TsdbAction::DuckDb {
            chunk,
            sql,
            series_label,
            format,
        } => crate::tools::tsdb_duckdb::run(chunk, sql, series_label, format).await,
        #[cfg(not(feature = "tsdb-duckdb"))]
        TsdbAction::DuckDb { .. } => {
            anyhow::bail!(
                "this oxirs binary was built without the `tsdb-duckdb` feature; \
                 rebuild with `cargo build -p oxirs --features tsdb-duckdb` to enable \
                 DuckDB inspection of TSDB chunks"
            )
        }
    }
}

/// Open (or create) the real, disk-backed columnar time-series store rooted
/// at `<dataset>/tsdb/`.
fn open_columnar_store(dataset: &str) -> Result<ColumnarStore> {
    let tsdb_dir = PathBuf::from(dataset).join("tsdb");
    ColumnarStore::new(&tsdb_dir, CHUNK_DURATION, CACHE_SIZE).map_err(|e| {
        anyhow::anyhow!(
            "Failed to open time-series store at {}: {e}",
            tsdb_dir.display()
        )
    })
}

/// Write a single data point as its own (one-point) chunk.
fn write_single_point(
    store: &ColumnarStore,
    series: u64,
    timestamp: DateTime<Utc>,
    value: f64,
) -> Result<()> {
    let point = DataPoint::new(timestamp, value);
    let chunk = TimeChunk::new(series, timestamp, CHUNK_DURATION, vec![point])
        .map_err(|e| anyhow::anyhow!("Failed to build time-series chunk: {e}"))?;
    store
        .write_chunk(&chunk)
        .map_err(|e| anyhow::anyhow!("Failed to write chunk: {e}"))?;
    Ok(())
}

/// Parse a `timestamp,value` CSV file and insert every row as a real data
/// point. An optional header row (`timestamp,...`) is skipped.
fn insert_from_csv(store: &ColumnarStore, series: u64, path: &Path) -> Result<usize> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read CSV file {}", path.display()))?;

    let mut inserted = 0usize;
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if i == 0 && line.to_lowercase().starts_with("timestamp") {
            continue; // Skip an optional header row.
        }

        let mut parts = line.splitn(2, ',');
        let ts_str = parts
            .next()
            .with_context(|| format!("Missing timestamp column on line {}", i + 1))?
            .trim();
        let value_str = parts
            .next()
            .with_context(|| format!("Missing value column on line {}", i + 1))?
            .trim();

        let ts: DateTime<Utc> = ts_str
            .parse()
            .with_context(|| format!("Invalid timestamp on line {}: '{ts_str}'", i + 1))?;
        let value: f64 = value_str
            .parse()
            .with_context(|| format!("Invalid value on line {}: '{value_str}'", i + 1))?;

        write_single_point(store, series, ts, value)?;
        inserted += 1;
    }

    Ok(inserted)
}

/// Map a CLI aggregation name to the real `oxirs_tsdb::query::Aggregation`.
fn parse_aggregation(input: &str) -> Result<Aggregation> {
    match input.to_lowercase().as_str() {
        "avg" | "average" | "mean" => Ok(Aggregation::Avg),
        "min" => Ok(Aggregation::Min),
        "max" => Ok(Aggregation::Max),
        "sum" => Ok(Aggregation::Sum),
        "count" => Ok(Aggregation::Count),
        "first" => Ok(Aggregation::First),
        "last" => Ok(Aggregation::Last),
        "stddev" | "std" => Ok(Aggregation::StdDev),
        "variance" | "var" => Ok(Aggregation::Variance),
        "median" => Ok(Aggregation::Median),
        other => Err(anyhow::anyhow!(
            "Unsupported aggregation '{other}'; supported: avg, min, max, sum, count, \
             first, last, stddev, variance, median"
        )),
    }
}

fn aggregation_label(aggregation: Aggregation) -> &'static str {
    match aggregation {
        Aggregation::Avg => "avg",
        Aggregation::Min => "min",
        Aggregation::Max => "max",
        Aggregation::Sum => "sum",
        Aggregation::Count => "count",
        Aggregation::First => "first",
        Aggregation::Last => "last",
        Aggregation::StdDev => "stddev",
        Aggregation::Variance => "variance",
        Aggregation::Median => "median",
        Aggregation::Percentile(_) => "percentile",
    }
}

/// Print real query results in the requested output format.
fn print_points(points: &[DataPoint], format: &str) {
    match format.to_lowercase().as_str() {
        "csv" => {
            println!("timestamp,value");
            for p in points {
                println!("{},{}", p.timestamp.to_rfc3339(), p.value);
            }
        }
        _ => {
            let mut table = Table::new();
            table.add_row(row!["Timestamp", "Value"]);
            for p in points {
                table.add_row(row![p.timestamp.to_rfc3339(), p.value]);
            }
            table.printstd();
        }
    }
    println!(
        "\n{} {} data point(s)",
        "Total:".green().bold(),
        points.len()
    );
}

fn format_storage_size(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

async fn query_command(
    dataset: String,
    series: Option<u64>,
    start: Option<String>,
    end: Option<String>,
    sparql: Option<String>,
    aggregate: Option<String>,
    format: String,
) -> Result<()> {
    println!("{}", "Time-series query".bright_cyan().bold());
    println!("Dataset: {}", dataset.yellow());

    if let Some(sparql_query) = sparql {
        println!("\n{}", "SPARQL temporal query:".bright_cyan());
        println!("{}", sparql_query);
        anyhow::bail!(
            "SPARQL temporal extensions (ts:window / ts:resample / ts:interpolate) are \
             not yet wired into `oxirs tsdb query`; use --series/--start/--end for a \
             direct range query instead."
        );
    }

    let series_id = series.context("Series ID required for direct queries")?;
    let start_time = parse_timestamp(start.as_deref())?;
    let end_time = parse_timestamp(end.as_deref())?;

    println!("Series: {}", series_id);
    println!("Range: {} to {}", start_time, end_time);

    let store = open_columnar_store(&dataset)?;
    let points = store
        .query_range(series_id, start_time, end_time)
        .map_err(|e| anyhow::anyhow!("Query failed: {e}"))?;

    if let Some(agg) = aggregate {
        let aggregation = parse_aggregation(&agg)?;
        let mut aggregator = Aggregator::new();
        aggregator.add_batch(&points);
        let value = aggregator
            .result(aggregation)
            .map_err(|e| anyhow::anyhow!("Aggregation failed: {e}"))?;
        println!(
            "\n{} {} = {}",
            "Result:".green().bold(),
            aggregation_label(aggregation),
            value
        );
        println!("Points aggregated: {}", points.len());
        return Ok(());
    }

    print_points(&points, &format);

    Ok(())
}

async fn insert_command(
    dataset: String,
    series: u64,
    timestamp: Option<String>,
    value: f64,
    from_csv: Option<PathBuf>,
) -> Result<()> {
    println!("{}", "Insert time-series data".bright_cyan().bold());
    println!("Dataset: {}", dataset.yellow());
    println!("Series: {}", series);

    let store = open_columnar_store(&dataset)?;

    if let Some(csv_file) = from_csv {
        println!("Batch insert from: {}", csv_file.display());
        let inserted = insert_from_csv(&store, series, &csv_file)?;
        println!(
            "\n{} Inserted {} data point(s) from CSV",
            "✓".green(),
            inserted
        );
        return Ok(());
    }

    let ts = if let Some(ts_str) = timestamp {
        ts_str
            .parse::<DateTime<Utc>>()
            .context("Invalid timestamp")?
    } else {
        Utc::now()
    };

    println!("Timestamp: {}", ts);
    println!("Value: {}", value);

    write_single_point(&store, series, ts, value)?;

    println!("\n{} Data point inserted", "✓".green());

    Ok(())
}

async fn stats_command(dataset: String, series: Option<u64>, detailed: bool) -> Result<()> {
    println!("{}", "Time-series statistics".bright_cyan().bold());
    println!("Dataset: {}", dataset.yellow());

    let store = open_columnar_store(&dataset)?;
    let index = store.index();

    let series_ids: Vec<u64> = match series {
        Some(id) => vec![id],
        None => index
            .series_ids()
            .map_err(|e| anyhow::anyhow!("Failed to list series: {e}"))?,
    };

    let mut table = Table::new();
    table.add_row(row!["Series", "Points", "Chunks", "Compression", "Storage"]);

    let mut total_points = 0usize;
    let mut total_chunks = 0usize;
    let mut total_compressed = 0usize;
    let mut total_uncompressed = 0usize;

    for sid in &series_ids {
        let chunks = index
            .get_chunks_for_series(*sid)
            .map_err(|e| anyhow::anyhow!("Failed to read series {sid}: {e}"))?;
        let points: usize = chunks.iter().map(|c| c.point_count).sum();
        let compressed: usize = chunks.iter().map(|c| c.compressed_size).sum();
        let uncompressed: usize = chunks.iter().map(|c| c.uncompressed_size).sum();

        total_points += points;
        total_chunks += chunks.len();
        total_compressed += compressed;
        total_uncompressed += uncompressed;

        if detailed {
            let ratio = if compressed > 0 {
                uncompressed as f64 / compressed as f64
            } else {
                0.0
            };
            table.add_row(row![
                sid.to_string(),
                points.to_string(),
                chunks.len().to_string(),
                format!("{ratio:.1}:1"),
                format_storage_size(compressed)
            ]);
        }
    }

    if !detailed {
        let ratio = if total_compressed > 0 {
            total_uncompressed as f64 / total_compressed as f64
        } else {
            0.0
        };
        table.add_row(row![
            "Total",
            total_points.to_string(),
            total_chunks.to_string(),
            format!("{ratio:.1}:1"),
            format_storage_size(total_compressed)
        ]);
    }

    table.printstd();

    println!(
        "\n{} {} series, {} chunks, {} data point(s)",
        "Summary:".green(),
        series_ids.len(),
        total_chunks,
        total_points
    );

    Ok(())
}

async fn compact_command(dataset: String, series: Option<u64>, force: bool) -> Result<()> {
    let _ = (dataset, series, force);
    anyhow::bail!(
        "`oxirs tsdb compact` is not yet implemented: chunk compaction (merging small \
         chunks, reclaiming tombstoned space via oxirs_tsdb::write::Compactor) is not yet \
         wired into the CLI."
    )
}

async fn retention_command(action: RetentionAction) -> Result<()> {
    let _ = action;
    anyhow::bail!(
        "`oxirs tsdb retention` is not yet implemented: retention-policy persistence and \
         enforcement (oxirs_tsdb::write::RetentionEnforcer) is not yet wired into the CLI."
    )
}

async fn export_command(
    dataset: String,
    series: u64,
    output: PathBuf,
    format: String,
    start: Option<String>,
    end: Option<String>,
) -> Result<()> {
    println!("{}", "Export time-series".bright_cyan().bold());
    println!("Dataset: {}", dataset.yellow());
    println!("Series: {}", series);
    println!("Output: {}", output.display());
    println!("Format: {}", format);

    if !format.eq_ignore_ascii_case("csv") {
        anyhow::bail!(
            "Time-series export to '{format}' is not yet supported; only 'csv' is \
             currently wired. Use --format csv."
        );
    }

    let start_time = parse_timestamp(start.as_deref())?;
    let end_time = parse_timestamp(end.as_deref())?;

    let store = open_columnar_store(&dataset)?;
    let points = store
        .query_range(series, start_time, end_time)
        .map_err(|e| anyhow::anyhow!("Query failed: {e}"))?;

    let mut csv = String::from("timestamp,value\n");
    for point in &points {
        csv.push_str(&format!(
            "{},{}\n",
            point.timestamp.to_rfc3339(),
            point.value
        ));
    }
    std::fs::write(&output, csv)
        .with_context(|| format!("Failed to write {}", output.display()))?;

    println!(
        "\n{} Exported {} data point(s) to {}",
        "✓".green(),
        points.len(),
        output.display()
    );

    Ok(())
}

async fn benchmark_command(dataset: String, points: usize, series_count: usize) -> Result<()> {
    println!(
        "{}",
        "Benchmark time-series performance".bright_cyan().bold()
    );
    println!("Dataset: {}", dataset.yellow());
    println!("Data points: {}", points.to_string().cyan());
    println!("Series: {}", series_count);

    if points == 0 || series_count == 0 {
        anyhow::bail!("--points and --series-count must both be greater than zero");
    }

    println!("\n{} Running write benchmark...", "Info:".cyan());

    let store = open_columnar_store(&dataset)?;
    // MSRV-safe ceiling division (crate MSRV is 1.70; `usize::div_ceil` is 1.73+).
    let per_series = (points + series_count - 1) / series_count;

    let start_wall = std::time::Instant::now();
    let mut written = 0usize;
    for series_idx in 0..series_count {
        if written >= points {
            break;
        }
        let series_id = series_idx as u64 + 1;
        let base_time = Utc::now();

        let mut batch = Vec::with_capacity(per_series);
        for i in 0..per_series {
            if written >= points {
                break;
            }
            batch.push(DataPoint::new(
                base_time + ChronoDuration::milliseconds(i as i64),
                i as f64,
            ));
            written += 1;
        }
        if batch.is_empty() {
            continue;
        }

        let chunk_start = batch[0].timestamp;
        let chunk_span = (batch
            .last()
            .expect("batch checked non-empty above")
            .timestamp
            - chunk_start)
            .max(ChronoDuration::seconds(1));
        let chunk = TimeChunk::new(series_id, chunk_start, chunk_span, batch)
            .map_err(|e| anyhow::anyhow!("Failed to build benchmark chunk: {e}"))?;
        store
            .write_chunk(&chunk)
            .map_err(|e| anyhow::anyhow!("Benchmark write failed: {e}"))?;
    }
    let elapsed = start_wall.elapsed();
    let writes_per_sec = if elapsed.as_secs_f64() > 0.0 {
        written as f64 / elapsed.as_secs_f64()
    } else {
        written as f64
    };

    println!("\n{} Benchmark complete", "✓".green());
    println!("  Points written: {}", written);
    println!("  Elapsed: {:.3}s", elapsed.as_secs_f64());
    println!(
        "  Throughput: {:.0} writes/sec (real, measured)",
        writes_per_sec
    );

    Ok(())
}

fn parse_timestamp(s: Option<&str>) -> Result<DateTime<Utc>> {
    match s {
        Some(ts_str) => ts_str
            .parse::<DateTime<Utc>>()
            .context("Invalid timestamp format (use ISO 8601)"),
        None => Ok(Utc::now()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dataset_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "oxirs-tsdb-cmd-test-{tag}-{}",
            std::process::id() as u64 * 104_729 + line!() as u64
        ))
    }

    /// Regression test: `oxirs tsdb insert` + `oxirs tsdb query` must round
    /// trip through a real, disk-backed store instead of printing "TODO"
    /// and silently discarding the data (the old `Ok(())` stub behavior).
    #[test]
    fn test_insert_and_query_round_trip_through_real_store() {
        let dir = temp_dataset_dir("roundtrip");
        std::fs::create_dir_all(&dir).expect("create temp dataset dir");
        let dataset = dir.to_string_lossy().to_string();

        let store = open_columnar_store(&dataset).expect("open columnar store");

        let ts = Utc::now();
        write_single_point(&store, 42, ts, 22.5).expect("write point");

        let points = store
            .query_range(
                42,
                ts - ChronoDuration::minutes(1),
                ts + ChronoDuration::minutes(1),
            )
            .expect("query range");

        assert_eq!(points.len(), 1, "the inserted point must be queryable back");
        assert!((points[0].value - 22.5).abs() < f64::EPSILON);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test: CSV batch insert must write real, queryable points.
    #[test]
    fn test_insert_from_csv_writes_real_points() {
        let dir = temp_dataset_dir("csv");
        std::fs::create_dir_all(&dir).expect("create temp dataset dir");
        let dataset = dir.to_string_lossy().to_string();
        let store = open_columnar_store(&dataset).expect("open columnar store");

        let csv_path = dir.join("points.csv");
        let now = Utc::now();
        let csv_content = format!(
            "timestamp,value\n{},1.0\n{},2.0\n{},3.0\n",
            now.to_rfc3339(),
            (now + ChronoDuration::seconds(1)).to_rfc3339(),
            (now + ChronoDuration::seconds(2)).to_rfc3339(),
        );
        std::fs::write(&csv_path, csv_content).expect("write csv");

        let inserted = insert_from_csv(&store, 7, &csv_path).expect("csv insert");
        assert_eq!(inserted, 3);

        let points = store
            .query_range(
                7,
                now - ChronoDuration::minutes(1),
                now + ChronoDuration::minutes(1),
            )
            .expect("query range");
        assert_eq!(points.len(), 3);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test: stats must reflect the real on-disk series index,
    /// not the old hardcoded "1M points / 40:1" placeholder table rows.
    #[test]
    fn test_stats_reflects_real_written_data() {
        let dir = temp_dataset_dir("stats");
        std::fs::create_dir_all(&dir).expect("create temp dataset dir");
        let dataset = dir.to_string_lossy().to_string();
        let store = open_columnar_store(&dataset).expect("open columnar store");

        let ts = Utc::now();
        write_single_point(&store, 1, ts, 10.0).expect("write point");
        write_single_point(&store, 1, ts + ChronoDuration::hours(3), 20.0).expect("write point");

        let index = store.index();
        assert_eq!(index.series_count().expect("series_count"), 1);
        let chunks = index
            .get_chunks_for_series(1)
            .expect("get_chunks_for_series");
        let total_points: usize = chunks.iter().map(|c| c.point_count).sum();
        assert_eq!(
            total_points, 2,
            "real stats must reflect exactly the 2 points written"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parse_aggregation_supports_common_names() {
        assert!(matches!(
            parse_aggregation("avg").unwrap(),
            Aggregation::Avg
        ));
        assert!(matches!(
            parse_aggregation("MIN").unwrap(),
            Aggregation::Min
        ));
        assert!(matches!(
            parse_aggregation("sum").unwrap(),
            Aggregation::Sum
        ));
        assert!(parse_aggregation("bogus").is_err());
    }
}
