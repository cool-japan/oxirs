use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportedPoint {
    pub timestamp_ms: i64,
    pub metric: String,
    pub value: f64,
    pub tags_json: String,
}

impl ExportedPoint {
    pub fn new(timestamp_ms: i64, metric: impl Into<String>, value: f64) -> Self {
        Self {
            timestamp_ms,
            metric: metric.into(),
            value,
            tags_json: "{}".to_string(),
        }
    }

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

    pub fn parse_tags(&self) -> TsdbResult<HashMap<String, String>> {
        serde_json::from_str(&self.tags_json).map_err(|e| TsdbError::Serialization(e.to_string()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParquetCompression {
    None,
    #[default]
    Snappy,
    Zstd,
    Gzip,
}

impl ParquetCompression {
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

#[derive(Debug, Clone, Default)]
pub struct ExportStats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub variance: f64,
    pub stddev: f64,
    pub first_timestamp_ms: i64,
    pub last_timestamp_ms: i64,
    pub distinct_metrics: usize,
}

#[derive(Debug, Clone)]
pub struct ExportMetadata {
    pub row_count: usize,
    pub compression: ParquetCompression,
    pub row_group_size: usize,
    pub distinct_metrics: usize,
    pub time_span_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationFunction {
    Avg,
    Min,
    Max,
    Sum,
    Count,
    StdDev,
    Percentile50,
    Percentile95,
    Percentile99,
}

#[derive(Debug, Clone, Default)]
pub struct ColumnarStats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub variance: f64,
    pub stddev: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ColumnarExport {
    pub timestamps: Vec<i64>,
    pub metrics: Vec<String>,
    pub values: Vec<f64>,
}

impl ColumnarExport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(capacity),
            metrics: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    pub fn from_points(points: &[ExportedPoint]) -> Self {
        let mut out = Self::with_capacity(points.len());
        for p in points {
            out.timestamps.push(p.timestamp_ms);
            out.metrics.push(p.metric.clone());
            out.values.push(p.value);
        }
        out
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    pub fn push(&mut self, timestamp_ms: i64, metric: impl Into<String>, value: f64) {
        self.timestamps.push(timestamp_ms);
        self.metrics.push(metric.into());
        self.values.push(value);
    }

    pub fn to_points(&self) -> Vec<ExportedPoint> {
        self.timestamps
            .iter()
            .zip(self.metrics.iter())
            .zip(self.values.iter())
            .map(|((ts, m), v)| ExportedPoint::new(*ts, m.clone(), *v))
            .collect()
    }

    pub fn to_csv(&self, path: &std::path::Path) -> TsdbResult<()> {
        use std::io::Write as IoWrite;
        let file = std::fs::File::create(path).map_err(|e| TsdbError::Io(e.to_string()))?;
        let mut writer = std::io::BufWriter::new(file);
        writeln!(writer, "timestamp_ms,metric,value").map_err(|e| TsdbError::Io(e.to_string()))?;
        for i in 0..self.len() {
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

    pub fn to_json_string(&self) -> TsdbResult<String> {
        let v = self.to_json()?;
        serde_json::to_string(&v).map_err(|e| TsdbError::Serialization(e.to_string()))
    }

    pub fn filter_metric(&self, metric: &str) -> Self {
        let mut out = Self::new();
        for i in 0..self.len() {
            if self.metrics[i] == metric {
                out.push(self.timestamps[i], self.metrics[i].clone(), self.values[i]);
            }
        }
        out
    }

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

    pub fn sort_by_timestamp(&mut self) {
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
                continue;
            }
            if line.is_empty() {
                continue;
            }
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

pub fn parse_csv_line(line: &str) -> Vec<String> {
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

#[derive(Debug, Clone)]
pub struct DuckDbQueryAdapter {
    parquet_path: String,
}

impl DuckDbQueryAdapter {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            parquet_path: path.into(),
        }
    }

    pub fn parquet_path(&self) -> &str {
        &self.parquet_path
    }

    pub fn select_metric(&self, metric: &str, start_ms: i64, end_ms: i64) -> String {
        format!(
            "SELECT timestamp, metric, value, tags_json \
             FROM read_parquet('{}') \
             WHERE metric = '{}' AND timestamp BETWEEN {} AND {} \
             ORDER BY timestamp ASC;",
            self.parquet_path, metric, start_ms, end_ms
        )
    }

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

    pub fn export_query_to_parquet(&self, query: &str, output_path: &str) -> String {
        format!(
            "COPY ({query}) TO '{output}' (FORMAT PARQUET, COMPRESSION SNAPPY);",
            query = query,
            output = output_path,
        )
    }

    pub fn list_metrics(&self) -> String {
        format!(
            "SELECT DISTINCT metric FROM read_parquet('{}') ORDER BY metric;",
            self.parquet_path
        )
    }

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

    pub fn create_view(&self, view_name: &str) -> String {
        format!(
            "CREATE OR REPLACE VIEW {view} AS \
             SELECT * FROM read_parquet('{}');",
            self.parquet_path,
            view = view_name,
        )
    }

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
