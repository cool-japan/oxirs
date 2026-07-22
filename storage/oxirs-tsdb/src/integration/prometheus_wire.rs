//! Prometheus remote-write wire protocol: real protobuf encoding, real
//! Snappy compression, and real HTTP transport.
//!
//! This module hand-encodes the exact `prompb.WriteRequest` protobuf schema
//! used by the Prometheus remote-write protocol directly against the
//! protobuf wire format (varints + length-delimited submessages). The
//! schema is small, stable, and part of a published API contract
//! (<https://github.com/prometheus/prometheus/blob/main/prompb/types.proto>
//! and `remote.proto`), so hand-encoding it avoids pulling in a `protoc`
//! build-time dependency while still producing bytes that any real
//! Prometheus/Cortex/Mimir/VictoriaMetrics/Thanos receiver can decode:
//!
//! ```text
//! message WriteRequest {
//!   repeated TimeSeries timeseries = 1;
//! }
//! message TimeSeries {
//!   repeated Label labels = 1;
//!   repeated Sample samples = 2;
//! }
//! message Label {
//!   string name = 1;
//!   string value = 2;
//! }
//! message Sample {
//!   double value = 1;
//!   int64 timestamp = 2;
//! }
//! ```
//!
//! The encoded bytes are then compressed with the raw Snappy block format
//! (via `oxiarc-snappy`, pure Rust) — exactly the codec the remote-write
//! spec requires — and POSTed with the headers a genuine receiver expects
//! (`Content-Encoding: snappy`, `Content-Type: application/x-protobuf`,
//! `X-Prometheus-Remote-Write-Version: 0.1.0`).

use super::prometheus_remote_write::{
    PrometheusLabel, PrometheusRemoteWriteConfig, PrometheusSample, PrometheusTimeSeries,
};
use crate::error::{TsdbError, TsdbResult};
use std::time::Duration;

// ──────────────────────────────────────────────────────────────────────────────
// Protobuf wire-format primitives
// ──────────────────────────────────────────────────────────────────────────────

/// Protobuf wire type: varint.
const WIRE_VARINT: u8 = 0;
/// Protobuf wire type: 64-bit fixed (used for `double`).
const WIRE_FIXED64: u8 = 1;
/// Protobuf wire type: length-delimited (strings, bytes, embedded messages).
const WIRE_LEN_DELIMITED: u8 = 2;

/// Encode an unsigned integer as a protobuf base-128 varint.
fn write_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            buf.push(byte | 0x80);
        } else {
            buf.push(byte);
            break;
        }
    }
}

/// Encode a field tag: `(field_number << 3) | wire_type`.
fn write_tag(field_number: u32, wire_type: u8, buf: &mut Vec<u8>) {
    write_varint(((field_number as u64) << 3) | wire_type as u64, buf);
}

/// Encode a length-delimited field (string, bytes, or embedded message).
fn write_len_delimited(field_number: u32, data: &[u8], buf: &mut Vec<u8>) {
    write_tag(field_number, WIRE_LEN_DELIMITED, buf);
    write_varint(data.len() as u64, buf);
    buf.extend_from_slice(data);
}

/// Encode a protobuf `string` field.
fn write_string_field(field_number: u32, value: &str, buf: &mut Vec<u8>) {
    write_len_delimited(field_number, value.as_bytes(), buf);
}

/// Encode a protobuf `double` field (fixed64, little-endian IEEE-754 bits).
fn write_double_field(field_number: u32, value: f64, buf: &mut Vec<u8>) {
    write_tag(field_number, WIRE_FIXED64, buf);
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode a protobuf `int64` field. Negative values are encoded as their
/// full two's-complement bit pattern varint-coded (the standard, non-zigzag
/// protobuf `int64` behaviour — this deliberately matches `prompb.Sample`'s
/// field type, not `sint64`).
fn write_int64_field(field_number: u32, value: i64, buf: &mut Vec<u8>) {
    write_tag(field_number, WIRE_VARINT, buf);
    write_varint(value as u64, buf);
}

// ──────────────────────────────────────────────────────────────────────────────
// prompb message encoders
// ──────────────────────────────────────────────────────────────────────────────

fn encode_label(label: &PrometheusLabel) -> Vec<u8> {
    let mut buf = Vec::new();
    write_string_field(1, &label.name, &mut buf);
    write_string_field(2, &label.value, &mut buf);
    buf
}

fn encode_sample(sample: &PrometheusSample) -> Vec<u8> {
    let mut buf = Vec::new();
    write_double_field(1, sample.value, &mut buf);
    write_int64_field(2, sample.timestamp_ms, &mut buf);
    buf
}

fn encode_timeseries(series: &PrometheusTimeSeries) -> Vec<u8> {
    let mut buf = Vec::new();
    for label in &series.labels {
        write_len_delimited(1, &encode_label(label), &mut buf);
    }
    for sample in &series.samples {
        write_len_delimited(2, &encode_sample(sample), &mut buf);
    }
    buf
}

/// Encode a batch of time-series as a `prompb.WriteRequest` protobuf message.
pub(crate) fn encode_write_request(series: &[PrometheusTimeSeries]) -> Vec<u8> {
    let mut buf = Vec::new();
    for ts in series {
        write_len_delimited(1, &encode_timeseries(ts), &mut buf);
    }
    buf
}

// ──────────────────────────────────────────────────────────────────────────────
// HTTP transport
// ──────────────────────────────────────────────────────────────────────────────

/// Send a batch of time-series to a real Prometheus-compatible remote-write
/// endpoint over HTTP, honoring `config.timeout_ms`, `config.auth_header`,
/// and `config.max_retries`.
///
/// Retries transient failures (network errors, HTTP 429, and 5xx server
/// errors) up to `config.max_retries` times with linear backoff. Any other
/// non-2xx status (a malformed-request 4xx) is returned immediately as an
/// error without retrying, since retrying an inherently-rejected request
/// cannot succeed. On success (2xx) returns `Ok(())`; this function never
/// reports success without the receiver actually acknowledging the write.
pub(crate) async fn send_remote_write(
    config: &PrometheusRemoteWriteConfig,
    series: &[PrometheusTimeSeries],
) -> TsdbResult<()> {
    if series.is_empty() {
        return Ok(());
    }

    let protobuf_bytes = encode_write_request(series);
    let compressed = oxiarc_snappy::compress(&protobuf_bytes);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(config.timeout_ms))
        .build()
        .map_err(|e| TsdbError::Integration(format!("failed to build HTTP client: {e}")))?;

    let mut attempt: u32 = 0;
    loop {
        let mut request = client
            .post(&config.endpoint)
            .header("Content-Encoding", "snappy")
            .header("Content-Type", "application/x-protobuf")
            .header("X-Prometheus-Remote-Write-Version", "0.1.0")
            .body(compressed.clone());

        if let Some(auth) = &config.auth_header {
            request = request.header("Authorization", auth.clone());
        }

        let outcome = request.send().await;

        match outcome {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return Ok(());
                }

                let retryable = status.is_server_error() || status.as_u16() == 429;
                let body_snippet = response.text().await.unwrap_or_default();
                let err_msg = format!(
                    "Prometheus remote-write to '{}' failed with HTTP {}: {}",
                    config.endpoint,
                    status.as_u16(),
                    body_snippet.chars().take(500).collect::<String>()
                );

                if !retryable || attempt >= config.max_retries {
                    return Err(TsdbError::Integration(err_msg));
                }
            }
            Err(e) => {
                if attempt >= config.max_retries {
                    return Err(TsdbError::Integration(format!(
                        "Prometheus remote-write to '{}' failed after {} attempt(s): {e}",
                        config.endpoint,
                        attempt + 1
                    )));
                }
            }
        }

        attempt += 1;
        tokio::time::sleep(Duration::from_millis(100u64 * attempt as u64)).await;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal recursive-descent protobuf decoder used only to verify our
    /// hand-rolled encoder against the wire format in tests (not part of
    /// the public API).
    fn decode_varint(data: &[u8], pos: &mut usize) -> u64 {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = data[*pos];
            *pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result
    }

    /// Parses `[(field_number, wire_type, payload_bytes)]` top-level fields.
    fn decode_fields(data: &[u8]) -> Vec<(u32, u8, Vec<u8>)> {
        let mut fields = Vec::new();
        let mut pos = 0usize;
        while pos < data.len() {
            let tag = decode_varint(data, &mut pos);
            let field_number = (tag >> 3) as u32;
            let wire_type = (tag & 0x7) as u8;
            match wire_type {
                0 => {
                    let start = pos;
                    let _ = decode_varint(data, &mut pos);
                    fields.push((field_number, wire_type, data[start..pos].to_vec()));
                }
                1 => {
                    fields.push((field_number, wire_type, data[pos..pos + 8].to_vec()));
                    pos += 8;
                }
                2 => {
                    let len = decode_varint(data, &mut pos) as usize;
                    fields.push((field_number, wire_type, data[pos..pos + len].to_vec()));
                    pos += len;
                }
                other => panic!("unsupported wire type {other} in test decoder"),
            }
        }
        fields
    }

    #[test]
    fn regression_encode_write_request_is_valid_protobuf_wire_format() {
        let series = PrometheusTimeSeries::new(vec![
            PrometheusLabel::new_unchecked("__name__", "cpu_usage"),
            PrometheusLabel::new_unchecked("job", "node"),
        ]);
        let mut series = series;
        series.push_sample(PrometheusSample::new(42.5, 1_700_000_000_000));
        series.push_sample(PrometheusSample::new(-1.5, 1_700_000_001_000));

        let encoded = encode_write_request(std::slice::from_ref(&series));

        // Top level: one field 1 (timeseries), wire type 2 (length-delimited).
        let top_fields = decode_fields(&encoded);
        assert_eq!(top_fields.len(), 1);
        assert_eq!(top_fields[0].0, 1);
        assert_eq!(top_fields[0].1, WIRE_LEN_DELIMITED);

        // Inside TimeSeries: 2 labels (field 1) + 2 samples (field 2).
        let ts_fields = decode_fields(&top_fields[0].2);
        let label_fields: Vec<_> = ts_fields.iter().filter(|f| f.0 == 1).collect();
        let sample_fields: Vec<_> = ts_fields.iter().filter(|f| f.0 == 2).collect();
        assert_eq!(label_fields.len(), 2);
        assert_eq!(sample_fields.len(), 2);

        // Decode the first label and confirm it round-trips as UTF-8 name/value.
        let label0_fields = decode_fields(&label_fields[0].2);
        let name_bytes = &label0_fields
            .iter()
            .find(|f| f.0 == 1)
            .expect("name field")
            .2;
        assert_eq!(std::str::from_utf8(name_bytes).unwrap(), "__name__");

        // Decode the first sample and confirm the double value round-trips.
        let sample0_fields = decode_fields(&sample_fields[0].2);
        let value_bytes = &sample0_fields
            .iter()
            .find(|f| f.0 == 1)
            .expect("value field")
            .2;
        let value = f64::from_le_bytes(value_bytes.as_slice().try_into().unwrap());
        assert_eq!(value, 42.5);
        let ts_bytes = &sample0_fields
            .iter()
            .find(|f| f.0 == 2)
            .expect("timestamp field")
            .2;
        let mut pos = 0usize;
        let ts_raw = decode_varint(ts_bytes, &mut pos);
        assert_eq!(ts_raw as i64, 1_700_000_000_000);
    }

    #[test]
    fn regression_encode_write_request_negative_timestamp_roundtrips() {
        // int64 fields must encode negative values as their full
        // two's-complement varint (not silently truncated/wrapped).
        let mut series =
            PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("__name__", "m")]);
        series.push_sample(PrometheusSample::new(1.0, -12345));

        let encoded = encode_write_request(std::slice::from_ref(&series));
        let top_fields = decode_fields(&encoded);
        let ts_fields = decode_fields(&top_fields[0].2);
        let sample_field = ts_fields.iter().find(|f| f.0 == 2).expect("sample field");
        let sample_fields = decode_fields(&sample_field.2);
        let ts_bytes = &sample_fields
            .iter()
            .find(|f| f.0 == 2)
            .expect("timestamp field")
            .2;
        let mut pos = 0usize;
        let ts_raw = decode_varint(ts_bytes, &mut pos) as i64;
        assert_eq!(ts_raw, -12345);
    }

    #[test]
    fn regression_encode_write_request_empty_series_is_empty() {
        assert!(encode_write_request(&[]).is_empty());
    }

    #[tokio::test]
    async fn regression_send_remote_write_empty_series_is_noop_ok() {
        let config = PrometheusRemoteWriteConfig::new("http://127.0.0.1:1/unreachable");
        // No network call should even be attempted for an empty batch.
        let result = send_remote_write(&config, &[]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn regression_send_remote_write_fails_loudly_on_unreachable_endpoint() {
        // A real network failure (connection refused / no listener) must
        // surface as an Err, never as a silently-successful Ok — this is
        // the core fail-loud contract the original stub violated.
        let config =
            PrometheusRemoteWriteConfig::new("http://127.0.0.1:1/unreachable").with_timeout_ms(500);
        let config = PrometheusRemoteWriteConfig {
            max_retries: 0,
            ..config
        };
        let series =
            PrometheusTimeSeries::new(vec![PrometheusLabel::new_unchecked("__name__", "m")]);
        let mut series = series;
        series.push_sample(PrometheusSample::new(1.0, 1000));

        let result = send_remote_write(&config, std::slice::from_ref(&series)).await;
        assert!(
            result.is_err(),
            "sending to an unreachable endpoint must return Err, not Ok"
        );
    }
}
