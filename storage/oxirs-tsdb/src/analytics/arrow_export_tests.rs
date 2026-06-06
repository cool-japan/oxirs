use super::arrow_export_builder::{ArrowExporter, ParquetExporter};
use super::arrow_export_types::{
    AggregationFunction, ColumnarExport, DuckDbQueryAdapter, ExportedPoint, ParquetCompression,
};
use std::collections::HashMap;

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
    assert_eq!(batches.len(), 3);
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
    assert!((stats.variance - 20.0 / 3.0).abs() < 1e-9);
    assert!((stats.stddev - (20.0_f64 / 3.0).sqrt()).abs() < 1e-9);
}

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
    let parquet = std::env::temp_dir()
        .join(format!("oxirs_tsdb_avg_{}.parquet", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let adapter = DuckDbQueryAdapter::new(parquet);
    let sql = adapter.aggregate_metric("mem", 0, 9_999, AggregationFunction::Avg);
    assert!(sql.contains("AVG(value)"));
    assert!(sql.contains("GROUP BY metric"));
}

#[test]
fn test_duckdb_adapter_aggregate_all_functions() {
    let parquet = std::env::temp_dir()
        .join(format!("oxirs_tsdb_aggall_{}.parquet", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let adapter = DuckDbQueryAdapter::new(parquet);
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
    let parquet = std::env::temp_dir()
        .join(format!("oxirs_tsdb_pct_{}.parquet", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let adapter = DuckDbQueryAdapter::new(parquet);
    let sql = adapter.aggregate_metric("cpu", 0, 1000, AggregationFunction::Percentile50);
    assert!(sql.contains("PERCENTILE_CONT(0.5)"));

    let sql = adapter.aggregate_metric("cpu", 0, 1000, AggregationFunction::Percentile95);
    assert!(sql.contains("PERCENTILE_CONT(0.95)"));

    let sql = adapter.aggregate_metric("cpu", 0, 1000, AggregationFunction::Percentile99);
    assert!(sql.contains("PERCENTILE_CONT(0.99)"));
}

#[test]
fn test_duckdb_adapter_resample_sql() {
    let parquet = std::env::temp_dir()
        .join(format!(
            "oxirs_tsdb_resample_{}.parquet",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned();
    let adapter = DuckDbQueryAdapter::new(parquet);
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
    assert_eq!(filtered.len(), 4);
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
    assert_eq!(lines.len(), 11);
    let _ = std::fs::remove_file(&path);
}
