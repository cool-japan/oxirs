//! Run-Length Encoding (RLE) compression for time-series data
//!
//! RLE is highly effective for step-function data such as IoT device states,
//! alarm flags, or any series that holds a constant value for extended periods.
//!
//! ## Encoding
//!
//! Each *run* describes a contiguous sequence of identical values:
//!
//! ```text
//! RleRun { start_timestamp, value, count }
//! ```
//!
//! The end timestamp of a run is implicit: it is `start_timestamp + (count - 1) * step`,
//! where `step` is the nominal sampling interval.  For sparse data where the
//! sampling interval is not fixed, each [`RleRun`] exposes `end_timestamp` which
//! is set by the encoder to the timestamp of the last sample in the run.

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single RLE run: one or more consecutive samples that share the same value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RleRun {
    /// Timestamp (ms since epoch) of the first sample in this run.
    pub start_timestamp: i64,
    /// Timestamp (ms since epoch) of the last sample in this run.
    pub end_timestamp: i64,
    /// The common value of all samples in this run.
    ///
    /// Two f64 values are considered equal when their bit patterns match
    /// (i.e. `a.to_bits() == b.to_bits()`), so that `NaN != NaN` does *not*
    /// collapse different NaN payloads, and `+0.0 != -0.0`.
    pub value: f64,
    /// Number of samples in this run (>= 1).
    pub count: u32,
}

impl RleRun {
    /// Return the compressed bit-pattern equality of two f64 values.
    #[inline]
    fn values_equal(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Streaming RLE encoder for `(timestamp_ms, value)` pairs.
///
/// Feed samples in time-ascending order via [`push`](Self::push),
/// then call [`finish`](Self::finish) to obtain the run list.
pub struct RleEncoder {
    current_value: Option<f64>,
    run_length: u32,
    current_start: i64,
    current_last: i64,
    encoded: Vec<RleRun>,
}

impl RleEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self {
            current_value: None,
            run_length: 0,
            current_start: 0,
            current_last: 0,
            encoded: Vec::new(),
        }
    }

    /// Push the next `(timestamp_ms, value)` sample.
    ///
    /// Samples should arrive in non-decreasing timestamp order.
    pub fn push(&mut self, timestamp: i64, value: f64) -> TsdbResult<()> {
        match self.current_value {
            None => {
                // First sample
                self.current_value = Some(value);
                self.run_length = 1;
                self.current_start = timestamp;
                self.current_last = timestamp;
            }
            Some(prev) if RleRun::values_equal(prev, value) => {
                // Continue existing run
                if timestamp < self.current_last {
                    return Err(TsdbError::Compression(format!(
                        "RLE: timestamps must be non-decreasing, got {} after {}",
                        timestamp, self.current_last
                    )));
                }
                self.run_length += 1;
                self.current_last = timestamp;
            }
            Some(prev) => {
                // Value changed: flush current run
                if timestamp < self.current_last {
                    return Err(TsdbError::Compression(format!(
                        "RLE: timestamps must be non-decreasing, got {} after {}",
                        timestamp, self.current_last
                    )));
                }
                self.encoded.push(RleRun {
                    start_timestamp: self.current_start,
                    end_timestamp: self.current_last,
                    value: prev,
                    count: self.run_length,
                });
                self.current_value = Some(value);
                self.run_length = 1;
                self.current_start = timestamp;
                self.current_last = timestamp;
            }
        }
        Ok(())
    }

    /// Flush any remaining run and return the complete list of [`RleRun`]s.
    pub fn finish(mut self) -> Vec<RleRun> {
        if let Some(v) = self.current_value {
            self.encoded.push(RleRun {
                start_timestamp: self.current_start,
                end_timestamp: self.current_last,
                value: v,
                count: self.run_length,
            });
        }
        self.encoded
    }
}

impl Default for RleEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Encode a slice of `(timestamp_ms, value)` pairs into a list of [`RleRun`]s.
///
/// Equivalent to constructing an [`RleEncoder`], pushing all samples, and
/// calling [`RleEncoder::finish`].
pub fn rle_encode(data: &[(i64, f64)]) -> TsdbResult<Vec<RleRun>> {
    let mut encoder = RleEncoder::new();
    for &(ts, val) in data {
        encoder.push(ts, val)?;
    }
    Ok(encoder.finish())
}

/// Decode a list of [`RleRun`]s back into `(timestamp_ms, value)` pairs.
///
/// For each run, the encoder stored `count` samples.  When the sampling
/// interval is uniform within a run, we distribute the timestamps linearly
/// between `start_timestamp` and `end_timestamp`.  When a run has `count == 1`,
/// only `start_timestamp` is emitted.
///
/// # Panics
/// Never panics (uses `?` internally).
pub fn rle_decode(runs: &[RleRun]) -> Vec<(i64, f64)> {
    let total: usize = runs.iter().map(|r| r.count as usize).sum();
    let mut out = Vec::with_capacity(total);

    for run in runs {
        if run.count == 0 {
            continue;
        }
        if run.count == 1 {
            out.push((run.start_timestamp, run.value));
        } else {
            // Distribute timestamps evenly across the run
            let span = run.end_timestamp - run.start_timestamp;
            let step = span / (run.count as i64 - 1);
            for i in 0..run.count as i64 {
                let ts = if i == run.count as i64 - 1 {
                    // Use exact end timestamp for last sample to avoid rounding drift
                    run.end_timestamp
                } else {
                    run.start_timestamp + i * step
                };
                out.push((ts, run.value));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// RleBlock: a self-contained serialisable compressed block
// ---------------------------------------------------------------------------

/// A self-contained, serialisable RLE-compressed block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RleBlock {
    /// The encoded runs.
    pub runs: Vec<RleRun>,
    /// Total number of original samples represented by this block.
    pub total_samples: u64,
}

impl RleBlock {
    /// Build a block from raw data.
    pub fn from_data(data: &[(i64, f64)]) -> TsdbResult<Self> {
        let runs = rle_encode(data)?;
        let total_samples = data.len() as u64;
        Ok(Self {
            runs,
            total_samples,
        })
    }

    /// Decode back to `(timestamp_ms, value)` pairs.
    pub fn decode(&self) -> Vec<(i64, f64)> {
        rle_decode(&self.runs)
    }

    /// Approximate compression ratio (original bytes / encoded bytes).
    ///
    /// Assumes each original sample costs 16 bytes (i64 + f64), and each
    /// [`RleRun`] costs approximately 28 bytes on the wire.
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.total_samples as f64 * 16.0;
        let encoded_bytes = self.runs.len() as f64 * 28.0 + 8.0; // +8 for header
        if encoded_bytes == 0.0 {
            1.0
        } else {
            original_bytes / encoded_bytes
        }
    }

    /// Number of distinct runs (lower is better compression).
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_encode_decode() {
        let runs = rle_encode(&[]).expect("encode failed");
        assert!(runs.is_empty());
        let decoded = rle_decode(&runs);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_single_sample() {
        let data = vec![(1000i64, 42.0f64)];
        let runs = rle_encode(&data).expect("encode");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].count, 1);
        assert_eq!(runs[0].value, 42.0);
        let decoded = rle_decode(&runs);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_constant_series_compresses_to_one_run() {
        let data: Vec<(i64, f64)> = (0..1000).map(|i| (i as i64 * 1000, 7.5)).collect();
        let runs = rle_encode(&data).expect("encode");
        assert_eq!(
            runs.len(),
            1,
            "constant series should produce exactly one run"
        );
        assert_eq!(runs[0].count, 1000);
        assert_eq!(runs[0].value, 7.5);
        assert_eq!(runs[0].start_timestamp, 0);
        assert_eq!(runs[0].end_timestamp, 999 * 1000);
    }

    #[test]
    fn test_alternating_values_round_trip() {
        let data: Vec<(i64, f64)> = (0..10)
            .map(|i| (i as i64 * 100, if i % 2 == 0 { 0.0 } else { 1.0 }))
            .collect();
        let runs = rle_encode(&data).expect("encode");
        // Every sample changes → 10 runs
        assert_eq!(runs.len(), 10);
        let decoded = rle_decode(&runs);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_step_function() {
        // [0,0,0,1,1,1,0,0,2,2]
        let data: Vec<(i64, f64)> = vec![
            (0, 0.0),
            (1, 0.0),
            (2, 0.0),
            (3, 1.0),
            (4, 1.0),
            (5, 1.0),
            (6, 0.0),
            (7, 0.0),
            (8, 2.0),
            (9, 2.0),
        ];
        let runs = rle_encode(&data).expect("encode");
        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0].count, 3);
        assert_eq!(runs[1].count, 3);
        assert_eq!(runs[2].count, 2);
        assert_eq!(runs[3].count, 2);
        let decoded = rle_decode(&runs);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_nan_not_coalesced() {
        // Two NaN with different bit patterns should NOT be merged
        let nan1 = f64::from_bits(0x7FF8_0000_0000_0001);
        let nan2 = f64::from_bits(0x7FF8_0000_0000_0002);
        let data = vec![(0i64, nan1), (1000, nan2)];
        let runs = rle_encode(&data).expect("encode");
        assert_eq!(runs.len(), 2, "different NaN payloads should not be merged");
    }

    #[test]
    fn test_positive_zero_vs_negative_zero() {
        // +0.0 and -0.0 have different bit patterns, so should NOT be merged
        let pos_zero = 0.0f64;
        let neg_zero = -0.0f64;
        assert_ne!(pos_zero.to_bits(), neg_zero.to_bits());
        let data = vec![(0i64, pos_zero), (1000, neg_zero)];
        let runs = rle_encode(&data).expect("encode");
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn test_out_of_order_timestamps_error() {
        let mut enc = RleEncoder::new();
        enc.push(2000, 1.0).expect("push ok");
        enc.push(1000, 1.0)
            .expect_err("should fail: ts goes backward");
    }

    #[test]
    fn test_rle_block_compression_ratio() {
        let data: Vec<(i64, f64)> = (0..10_000)
            .map(|i| (i as i64 * 100, (i / 100) as f64))
            .collect();
        let block = RleBlock::from_data(&data).expect("build block");
        assert_eq!(block.run_count(), 100); // 100 distinct constant regions
        let ratio = block.compression_ratio();
        assert!(
            ratio > 1.0,
            "should have positive compression: {:.2}",
            ratio
        );
    }

    #[test]
    fn test_rle_block_decode_round_trip() {
        let data: Vec<(i64, f64)> = vec![
            (0, 10.0),
            (1000, 10.0),
            (2000, 10.0),
            (3000, 20.0),
            (4000, 20.0),
            (5000, 30.0),
        ];
        let block = RleBlock::from_data(&data).expect("build block");
        let decoded = block.decode();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encoder_default() {
        let enc = RleEncoder::default();
        let runs = enc.finish();
        assert!(runs.is_empty());
    }

    #[test]
    fn test_large_constant_block() {
        let n = 100_000usize;
        let data: Vec<(i64, f64)> = (0..n)
            .map(|i| (i as i64 * 10, std::f64::consts::PI))
            .collect();
        let runs = rle_encode(&data).expect("encode");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].count, n as u32);
    }
}
