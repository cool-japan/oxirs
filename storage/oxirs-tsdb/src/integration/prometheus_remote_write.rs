//! Prometheus Remote Write API integration.
//!
//! Provides types and utilities for converting TSDB series to Prometheus
//! remote-write format and sending them to a Prometheus-compatible endpoint.
//!
//! The Prometheus remote-write specification uses a snappy-compressed protobuf
//! payload.  This implementation provides a clean, pure-Rust representation
//! and a JSON-based wire format that is suitable for testing and environments
//! where a full protobuf codec is not available.  The serialisation layer is
//! designed to be swapped out for a real protobuf encoder without changing
//! the public API.

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Core data types
// ──────────────────────────────────────────────────────────────────────────────

/// A Prometheus label (name=value pair).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrometheusLabel {
    /// Label name (must match `[a-zA-Z_][a-zA-Z0-9_]*`).
    pub name: String,
    /// Label value (UTF-8 string, may be empty).
    pub value: String,
}

impl PrometheusLabel {
    /// Create a new label.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> TsdbResult<Self> {
        let name = name.into();
        validate_label_name(&name)?;
        Ok(Self {
            name,
            value: value.into(),
        })
    }

    /// Create a label without name validation (use only for internally generated
    /// labels that are known to be well-formed).
    pub fn new_unchecked(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A Prometheus sample: a (value, timestamp) pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrometheusSample {
    /// Sample value (64-bit float).
    pub value: f64,
    /// Milliseconds since Unix epoch.
    pub timestamp_ms: i64,
}

impl PrometheusSample {
    /// Create a new sample.
    pub fn new(value: f64, timestamp_ms: i64) -> Self {
        Self {
            value,
            timestamp_ms,
        }
    }
}

/// A labelled time-series as used in the Prometheus remote-write protocol.
///
/// A `PrometheusTimeSeries` corresponds to a single metric fingerprint.  The
/// `__name__` label must be present in `labels` (it is automatically added by
/// `PrometheusMetricConverter`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrometheusTimeSeries {
    /// Sorted label set that uniquely identifies this time-series.
    pub labels: Vec<PrometheusLabel>,
    /// Chronologically ordered samples.
    pub samples: Vec<PrometheusSample>,
}

impl PrometheusTimeSeries {
    /// Create an empty time-series with the given labels.
    pub fn new(mut labels: Vec<PrometheusLabel>) -> Self {
        // Labels must be sorted by name for the remote-write protocol.
        labels.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            labels,
            samples: Vec::new(),
        }
    }

    /// Append a sample.
    pub fn push_sample(&mut self, sample: PrometheusSample) {
        self.samples.push(sample);
    }

    /// Return the value of the `__name__` label, if present.
    pub fn metric_name(&self) -> Option<&str> {
        self.labels
            .iter()
            .find(|l| l.name == "__name__")
            .map(|l| l.value.as_str())
    }

    /// Return the label value for `name`, if present.
    pub fn label_value(&self, name: &str) -> Option<&str> {
        self.labels
            .binary_search_by_key(&name, |l| l.name.as_str())
            .ok()
            .map(|i| self.labels[i].value.as_str())
    }
}

/// A batch payload suitable for transmission to a Prometheus remote-write
/// endpoint.
///
/// In production this would be serialised as a snappy-compressed protobuf.
/// Here we serialise to JSON so that the code is self-contained and testable
/// without C or Fortran dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteWritePayload {
    /// All time-series in this write request.
    pub timeseries: Vec<PrometheusTimeSeries>,
    /// Batch creation timestamp (milliseconds since epoch).
    pub created_at_ms: i64,
    /// Protocol version identifier.
    pub version: String,
}

impl RemoteWritePayload {
    /// Create a new payload from a batch of time-series.
    pub fn new(timeseries: Vec<PrometheusTimeSeries>) -> Self {
        let created_at_ms = current_timestamp_ms();
        Self {
            timeseries,
            created_at_ms,
            version: "1.0.0".to_string(),
        }
    }

    /// Serialise to a JSON byte vector (production code would use protobuf +
    /// snappy here).
    pub fn encode(&self) -> TsdbResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| TsdbError::Integration(e.to_string()))
    }

    /// Deserialise from a JSON byte slice.
    pub fn decode(bytes: &[u8]) -> TsdbResult<Self> {
        serde_json::from_slice(bytes).map_err(|e| TsdbError::Integration(e.to_string()))
    }

    /// Total number of samples across all time-series.
    pub fn total_samples(&self) -> usize {
        self.timeseries.iter().map(|ts| ts.samples.len()).sum()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a `PrometheusRemoteWriter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusRemoteWriteConfig {
    /// HTTP endpoint URL (e.g. `http://prometheus:9090/api/v1/write`).
    pub endpoint: String,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Optional Bearer-token or Basic-auth header value.
    pub auth_header: Option<String>,
    /// Maximum number of time-series per write batch.
    pub batch_size: usize,
    /// Maximum retries on transient failure.
    pub max_retries: u32,
    /// Extra labels to attach to every metric (e.g. `cluster="prod"`).
    pub extra_labels: HashMap<String, String>,
}

impl Default for PrometheusRemoteWriteConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:9090/api/v1/write".to_string(),
            timeout_ms: 5_000,
            auth_header: None,
            batch_size: 500,
            max_retries: 3,
            extra_labels: HashMap::new(),
        }
    }
}

impl PrometheusRemoteWriteConfig {
    /// Create a new configuration pointing at the given endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Default::default()
        }
    }

    /// Builder: set the timeout.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Builder: set the auth header value.
    pub fn with_auth(mut self, header: impl Into<String>) -> Self {
        self.auth_header = Some(header.into());
        self
    }

    /// Builder: add an extra label appended to every metric.
    pub fn with_extra_label(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_labels.insert(name.into(), value.into());
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Metric converter
// ──────────────────────────────────────────────────────────────────────────────

/// Converts TSDB `DataPoint` slices into `PrometheusTimeSeries` objects.
pub struct PrometheusMetricConverter {
    config: PrometheusRemoteWriteConfig,
}

impl PrometheusMetricConverter {
    /// Create a converter backed by the given configuration.
    pub fn new(config: PrometheusRemoteWriteConfig) -> Self {
        Self { config }
    }

    /// Convert a `DataPoint` slice into a `PrometheusTimeSeries`.
    ///
    /// # Parameters
    /// - `data_points`: ordered slice of (timestamp, value) pairs.
    /// - `metric_name`: the Prometheus `__name__` label value.
    /// - `extra_labels`: additional labels specific to this series (merged with
    ///   the global labels from the config; series labels take precedence).
    pub fn tsdb_to_prometheus(
        &self,
        data_points: &[DataPoint],
        metric_name: &str,
        extra_labels: &[(String, String)],
    ) -> TsdbResult<PrometheusTimeSeries> {
        validate_metric_name(metric_name)?;

        // Build label set: global config labels + per-series labels + __name__.
        let mut label_map: HashMap<String, String> = self
            .config
            .extra_labels
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (k, v) in extra_labels {
            label_map.insert(k.clone(), v.clone());
        }
        // __name__ always wins.
        label_map.insert("__name__".to_string(), metric_name.to_string());

        let mut labels: Vec<PrometheusLabel> = label_map
            .into_iter()
            .map(|(n, v)| PrometheusLabel::new_unchecked(n, v))
            .collect();
        labels.sort_by(|a, b| a.name.cmp(&b.name));

        let mut ts = PrometheusTimeSeries::new(labels);

        for dp in data_points {
            let timestamp_ms = dp.timestamp.timestamp_millis();
            ts.push_sample(PrometheusSample::new(dp.value, timestamp_ms));
        }

        // Samples must be in chronological order.
        ts.samples.sort_by_key(|s| s.timestamp_ms);

        Ok(ts)
    }

    /// Split a large `Vec<PrometheusTimeSeries>` into batches of at most
    /// `config.batch_size` series.
    pub fn into_batches(&self, series: Vec<PrometheusTimeSeries>) -> Vec<RemoteWritePayload> {
        series
            .chunks(self.config.batch_size)
            .map(|chunk| RemoteWritePayload::new(chunk.to_vec()))
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Remote writer (async stub)
// ──────────────────────────────────────────────────────────────────────────────

/// Write result returned by `PrometheusRemoteWriter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteResult {
    /// Number of time-series successfully written.
    pub series_written: usize,
    /// Total number of samples successfully written.
    pub samples_written: usize,
    /// Any non-fatal warnings generated during the write.
    pub warnings: Vec<String>,
}

/// Async Prometheus remote-write client.
///
/// The current implementation is a **stub** that serialises the payload and
/// records what would be sent rather than making real HTTP calls.  This design
/// keeps the crate pure-Rust and dependency-free while providing a realistic
/// API surface that can be backed by a real HTTP client in the future.
pub struct PrometheusRemoteWriter {
    config: PrometheusRemoteWriteConfig,
}

impl PrometheusRemoteWriter {
    /// Create a new remote writer.
    pub fn new(config: PrometheusRemoteWriteConfig) -> Self {
        Self { config }
    }

    /// Write a batch of time-series to the configured endpoint.
    ///
    /// In the current stub implementation this method:
    /// 1. Validates all labels and samples.
    /// 2. Encodes the payload to JSON bytes.
    /// 3. Returns a `WriteResult` without making any network calls.
    pub async fn write_batch(&self, metrics: Vec<PrometheusTimeSeries>) -> TsdbResult<WriteResult> {
        if metrics.is_empty() {
            return Ok(WriteResult {
                series_written: 0,
                samples_written: 0,
                warnings: Vec::new(),
            });
        }

        let mut warnings = Vec::new();
        let mut series_written = 0usize;
        let mut samples_written = 0usize;

        for ts in &metrics {
            // Validate that __name__ is present.
            if ts.metric_name().is_none() {
                warnings.push(format!(
                    "time-series has no __name__ label; {} samples skipped",
                    ts.samples.len()
                ));
                continue;
            }

            // Check for unsorted samples.
            let sorted = ts
                .samples
                .windows(2)
                .all(|w| w[0].timestamp_ms <= w[1].timestamp_ms);
            if !sorted {
                warnings.push(format!(
                    "metric '{}' has out-of-order samples",
                    ts.metric_name().unwrap_or("<unknown>")
                ));
            }

            series_written += 1;
            samples_written += ts.samples.len();
        }

        // Encode the payload (validates JSON serialisability).
        let payload = RemoteWritePayload::new(metrics);
        let _encoded = payload.encode()?;

        tracing::debug!(
            endpoint = %self.config.endpoint,
            series = series_written,
            samples = samples_written,
            "Prometheus remote write (stub) – payload encoded, not sent"
        );

        Ok(WriteResult {
            series_written,
            samples_written,
            warnings,
        })
    }

    /// Write in batches, splitting according to `config.batch_size`.
    pub async fn write_all(&self, metrics: Vec<PrometheusTimeSeries>) -> TsdbResult<WriteResult> {
        let batch_size = self.config.batch_size;
        let mut total_series = 0usize;
        let mut total_samples = 0usize;
        let mut all_warnings = Vec::new();

        for chunk in metrics.chunks(batch_size) {
            let result = self.write_batch(chunk.to_vec()).await?;
            total_series += result.series_written;
            total_samples += result.samples_written;
            all_warnings.extend(result.warnings);
        }

        Ok(WriteResult {
            series_written: total_series,
            samples_written: total_samples,
            warnings: all_warnings,
        })
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &PrometheusRemoteWriteConfig {
        &self.config
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Validation helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Validate a Prometheus label name against `[a-zA-Z_][a-zA-Z0-9_]*`.
fn validate_label_name(name: &str) -> TsdbResult<()> {
    if name.is_empty() {
        return Err(TsdbError::Integration(
            "label name must not be empty".to_string(),
        ));
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty checked above");
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(TsdbError::Integration(format!(
            "label name '{name}' must start with [a-zA-Z_]"
        )));
    }
    for c in chars {
        if !c.is_ascii_alphanumeric() && c != '_' {
            return Err(TsdbError::Integration(format!(
                "label name '{name}' contains invalid character '{c}'"
            )));
        }
    }
    Ok(())
}

/// Validate a Prometheus metric name against `[a-zA-Z_:][a-zA-Z0-9_:]*`.
fn validate_metric_name(name: &str) -> TsdbResult<()> {
    if name.is_empty() {
        return Err(TsdbError::Integration(
            "metric name must not be empty".to_string(),
        ));
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty checked above");
    if !first.is_ascii_alphabetic() && first != '_' && first != ':' {
        return Err(TsdbError::Integration(format!(
            "metric name '{name}' must start with [a-zA-Z_:]"
        )));
    }
    for c in chars {
        if !c.is_ascii_alphanumeric() && c != '_' && c != ':' {
            return Err(TsdbError::Integration(format!(
                "metric name '{name}' contains invalid character '{c}'"
            )));
        }
    }
    Ok(())
}

/// Return the current wall-clock time as milliseconds since the Unix epoch.
fn current_timestamp_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    // ── Label & metric name validation ────────────────────────────────────────

    #[test]
    fn valid_label_names_accepted() {
        for name in &["__name__", "job", "instance", "my_label_123", "_hidden"] {
            assert!(validate_label_name(name).is_ok(), "should accept '{name}'");
        }
    }

    #[test]
    fn invalid_label_names_rejected() {
        for name in &["", "1starts_with_digit", "has-hyphen", "has space"] {
            assert!(validate_label_name(name).is_err(), "should reject '{name}'");
        }
    }

    #[test]
    fn valid_metric_names_accepted() {
        for name in &[
            "http_requests_total",
            "my:metric",
            "_internal",
            "cpu_usage_percent",
        ] {
            assert!(validate_metric_name(name).is_ok(), "should accept '{name}'");
        }
    }

    #[test]
    fn invalid_metric_names_rejected() {
        assert!(validate_metric_name("").is_err());
        assert!(validate_metric_name("0bad").is_err());
        assert!(validate_metric_name("has-hyphen").is_err());
    }

    // ── PrometheusLabel ───────────────────────────────────────────────────────

    #[test]
    fn label_new_validates() {
        assert!(PrometheusLabel::new("valid", "value").is_ok());
        assert!(PrometheusLabel::new("1bad", "value").is_err());
        assert!(PrometheusLabel::new("", "value").is_err());
    }

    #[test]
    fn label_new_unchecked_does_not_validate() {
        let l = PrometheusLabel::new_unchecked("__name__", "my_metric");
        assert_eq!(l.name, "__name__");
        assert_eq!(l.value, "my_metric");
    }

    // ── PrometheusTimeSeries ──────────────────────────────────────────────────

    #[test]
    fn timeseries_labels_are_sorted() {
        let labels = vec![
            PrometheusLabel::new_unchecked("job", "server"),
            PrometheusLabel::new_unchecked("__name__", "cpu"),
            PrometheusLabel::new_unchecked("instance", "host1"),
        ];
        let ts = PrometheusTimeSeries::new(labels);
        let names: Vec<&str> = ts.labels.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["__name__", "instance", "job"]);
    }

    #[test]
    fn timeseries_metric_name_lookup() {
        let ts = PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked(
            "__name__",
            "my_metric",
        )]);
        assert_eq!(ts.metric_name(), Some("my_metric"));
    }

    #[test]
    fn timeseries_label_value_lookup() {
        let ts = PrometheusTimeSeries::new(vec![
            PrometheusLabel::new_unchecked("__name__", "cpu"),
            PrometheusLabel::new_unchecked("job", "myservice"),
        ]);
        assert_eq!(ts.label_value("job"), Some("myservice"));
        assert_eq!(ts.label_value("missing"), None);
    }

    // ── PrometheusMetricConverter ─────────────────────────────────────────────

    fn make_data_points(n: usize) -> Vec<DataPoint> {
        (0..n)
            .map(|i| {
                DataPoint::new(
                    Utc.timestamp_millis_opt(i as i64 * 1_000)
                        .single()
                        .expect("valid timestamp"),
                    i as f64 * 1.5,
                )
            })
            .collect()
    }

    #[test]
    fn converter_basic_conversion() {
        let config = PrometheusRemoteWriteConfig::default();
        let converter = PrometheusMetricConverter::new(config);
        let data = make_data_points(10);
        let ts = converter
            .tsdb_to_prometheus(&data, "sensor_value", &[])
            .expect("conversion failed");

        assert_eq!(ts.metric_name(), Some("sensor_value"));
        assert_eq!(ts.samples.len(), 10);
        // Samples must be in ascending order.
        for w in ts.samples.windows(2) {
            assert!(w[0].timestamp_ms <= w[1].timestamp_ms);
        }
    }

    #[test]
    fn converter_adds_extra_labels() {
        let config = PrometheusRemoteWriteConfig::new("http://prom:9090")
            .with_extra_label("cluster", "prod");
        let converter = PrometheusMetricConverter::new(config);
        let data = make_data_points(3);
        let ts = converter
            .tsdb_to_prometheus(
                &data,
                "temp_celsius",
                &[("room".to_string(), "living_room".to_string())],
            )
            .expect("conversion failed");

        assert_eq!(ts.label_value("cluster"), Some("prod"));
        assert_eq!(ts.label_value("room"), Some("living_room"));
        assert_eq!(ts.metric_name(), Some("temp_celsius"));
    }

    #[test]
    fn converter_rejects_invalid_metric_name() {
        let config = PrometheusRemoteWriteConfig::default();
        let converter = PrometheusMetricConverter::new(config);
        let data = make_data_points(5);
        assert!(converter
            .tsdb_to_prometheus(&data, "1invalid", &[])
            .is_err());
    }

    #[test]
    fn converter_into_batches_respects_batch_size() {
        let config = PrometheusRemoteWriteConfig {
            batch_size: 3,
            ..Default::default()
        };
        let converter = PrometheusMetricConverter::new(config);

        let series: Vec<PrometheusTimeSeries> = (0..10)
            .map(|i| {
                let mut ts = PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked(
                    "__name__",
                    format!("metric_{i}"),
                )]);
                ts.push_sample(PrometheusSample::new(i as f64, i * 1000));
                ts
            })
            .collect();

        let batches = converter.into_batches(series);
        // 10 series ÷ 3 per batch = 4 batches (3, 3, 3, 1).
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].timeseries.len(), 3);
        assert_eq!(batches[3].timeseries.len(), 1);
    }

    // ── RemoteWritePayload ────────────────────────────────────────────────────

    #[test]
    fn payload_encode_decode_roundtrip() {
        let ts = PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked(
            "__name__",
            "test_metric",
        )]);
        let payload = RemoteWritePayload::new(vec![ts]);
        let encoded = payload.encode().expect("encode failed");
        let decoded = RemoteWritePayload::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.timeseries.len(), 1);
        assert_eq!(decoded.version, "1.0.0");
    }

    #[test]
    fn payload_total_samples() {
        let mut ts1 =
            PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("__name__", "m1")]);
        ts1.push_sample(PrometheusSample::new(1.0, 1000));
        ts1.push_sample(PrometheusSample::new(2.0, 2000));

        let mut ts2 =
            PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("__name__", "m2")]);
        ts2.push_sample(PrometheusSample::new(3.0, 3000));

        let payload = RemoteWritePayload::new(vec![ts1, ts2]);
        assert_eq!(payload.total_samples(), 3);
    }

    // ── PrometheusRemoteWriter (async stub) ───────────────────────────────────

    #[tokio::test]
    async fn writer_empty_batch_returns_zero() {
        let config = PrometheusRemoteWriteConfig::default();
        let writer = PrometheusRemoteWriter::new(config);
        let result = writer.write_batch(vec![]).await.expect("write failed");
        assert_eq!(result.series_written, 0);
        assert_eq!(result.samples_written, 0);
    }

    #[tokio::test]
    async fn writer_counts_series_and_samples() {
        let config = PrometheusRemoteWriteConfig::default();
        let writer = PrometheusRemoteWriter::new(config);

        let mut ts = PrometheusTimeSeries::new(vec![
            PrometheusLabel::new_unchecked("__name__", "cpu_usage"),
            PrometheusLabel::new_unchecked("job", "myapp"),
        ]);
        for i in 0..5_i64 {
            ts.push_sample(PrometheusSample::new(i as f64 * 10.0, i * 1000));
        }

        let result = writer.write_batch(vec![ts]).await.expect("write failed");
        assert_eq!(result.series_written, 1);
        assert_eq!(result.samples_written, 5);
        assert!(result.warnings.is_empty());
    }

    #[tokio::test]
    async fn writer_warns_on_missing_name_label() {
        let config = PrometheusRemoteWriteConfig::default();
        let writer = PrometheusRemoteWriter::new(config);

        let ts = PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("job", "no_name")]);
        let result = writer.write_batch(vec![ts]).await.expect("write failed");
        assert_eq!(result.series_written, 0);
        assert!(!result.warnings.is_empty());
    }

    #[tokio::test]
    async fn writer_write_all_batches() {
        let config = PrometheusRemoteWriteConfig {
            batch_size: 2,
            ..Default::default()
        };
        let writer = PrometheusRemoteWriter::new(config);

        let metrics: Vec<PrometheusTimeSeries> = (0..5)
            .map(|i| {
                let mut ts = PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked(
                    "__name__",
                    format!("metric_{i}"),
                )]);
                ts.push_sample(PrometheusSample::new(i as f64, i * 1000));
                ts
            })
            .collect();

        let result = writer.write_all(metrics).await.expect("write_all failed");
        assert_eq!(result.series_written, 5);
        assert_eq!(result.samples_written, 5);
    }

    #[test]
    fn config_builder_methods() {
        let config = PrometheusRemoteWriteConfig::new("http://prom:9090")
            .with_timeout_ms(10_000)
            .with_auth("Bearer token123")
            .with_extra_label("env", "staging");

        assert_eq!(config.endpoint, "http://prom:9090");
        assert_eq!(config.timeout_ms, 10_000);
        assert_eq!(config.auth_header.as_deref(), Some("Bearer token123"));
        assert_eq!(
            config.extra_labels.get("env").map(String::as_str),
            Some("staging")
        );
    }

    // ── Extra coverage ────────────────────────────────────────────────────────

    #[test]
    fn sample_new_stores_value_and_timestamp() {
        let s = PrometheusSample::new(42.5, 1_700_000_000_000);
        assert!((s.value - 42.5).abs() < f64::EPSILON);
        assert_eq!(s.timestamp_ms, 1_700_000_000_000);
    }

    #[test]
    fn timeseries_push_sample_increases_count() {
        let mut ts =
            PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("__name__", "m")]);
        assert!(ts.samples.is_empty());
        ts.push_sample(PrometheusSample::new(1.0, 1000));
        ts.push_sample(PrometheusSample::new(2.0, 2000));
        assert_eq!(ts.samples.len(), 2);
    }

    #[test]
    fn timeseries_without_name_label_returns_none() {
        let ts = PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("job", "svc")]);
        assert_eq!(ts.metric_name(), None);
    }

    #[test]
    fn payload_encode_produces_non_empty_bytes() {
        let ts = PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("__name__", "x")]);
        let payload = RemoteWritePayload::new(vec![ts]);
        let bytes = payload.encode().expect("encode");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn payload_total_samples_zero_when_no_series() {
        let payload = RemoteWritePayload::new(vec![]);
        assert_eq!(payload.total_samples(), 0);
    }

    #[test]
    fn config_default_has_expected_endpoint() {
        let config = PrometheusRemoteWriteConfig::default();
        assert_eq!(config.endpoint, "http://localhost:9090/api/v1/write");
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.max_retries, 3);
    }

    #[tokio::test]
    async fn writer_warns_on_out_of_order_samples() {
        let config = PrometheusRemoteWriteConfig::default();
        let writer = PrometheusRemoteWriter::new(config);

        let mut ts =
            PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("__name__", "cpu")]);
        // Samples intentionally out of order.
        ts.push_sample(PrometheusSample::new(2.0, 2000));
        ts.push_sample(PrometheusSample::new(1.0, 1000));

        let result = writer.write_batch(vec![ts]).await.expect("write");
        assert!(!result.warnings.is_empty());
        assert_eq!(result.series_written, 1); // still counted as written
    }

    #[test]
    fn converter_empty_data_points_produces_empty_samples() {
        let config = PrometheusRemoteWriteConfig::default();
        let converter = PrometheusMetricConverter::new(config);
        let ts = converter
            .tsdb_to_prometheus(&[], "empty_metric", &[])
            .expect("conversion");
        assert!(ts.samples.is_empty());
        assert_eq!(ts.metric_name(), Some("empty_metric"));
    }

    #[test]
    fn converter_into_batches_empty_returns_empty() {
        let config = PrometheusRemoteWriteConfig::default();
        let converter = PrometheusMetricConverter::new(config);
        let batches = converter.into_batches(vec![]);
        assert!(batches.is_empty());
    }

    #[test]
    fn label_serialization_roundtrip() {
        let label = PrometheusLabel::new_unchecked("__name__", "my_metric");
        let json = serde_json::to_string(&label).expect("serialize");
        let back: PrometheusLabel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "__name__");
        assert_eq!(back.value, "my_metric");
    }

    #[test]
    fn timeseries_label_sorted_after_new() {
        let labels = vec![
            PrometheusLabel::new_unchecked("z_label", "z"),
            PrometheusLabel::new_unchecked("a_label", "a"),
            PrometheusLabel::new_unchecked("m_label", "m"),
        ];
        let ts = PrometheusTimeSeries::new(labels);
        let names: Vec<&str> = ts.labels.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["a_label", "m_label", "z_label"]);
    }

    #[test]
    fn remote_write_writer_exposes_config() {
        let config = PrometheusRemoteWriteConfig::new("http://example.com/write");
        let writer = PrometheusRemoteWriter::new(config);
        assert_eq!(writer.config().endpoint, "http://example.com/write");
    }

    #[tokio::test]
    async fn writer_write_all_empty_returns_zero() {
        let config = PrometheusRemoteWriteConfig::default();
        let writer = PrometheusRemoteWriter::new(config);
        let result = writer.write_all(vec![]).await.expect("write_all");
        assert_eq!(result.series_written, 0);
        assert_eq!(result.samples_written, 0);
    }

    #[test]
    fn payload_version_is_set() {
        let payload = RemoteWritePayload::new(vec![]);
        assert_eq!(payload.version, "1.0.0");
    }
}
