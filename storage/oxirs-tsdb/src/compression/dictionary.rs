//! Dictionary compression for categorical time-series data
//!
//! Maps repeated string values to compact integer codes, enabling efficient
//! storage of label / enumeration series (device states, alert levels, etc.).
//!
//! ## Design
//!
//! * A **dictionary** maps each unique string value to a `u32` code.
//! * The encoded data stream stores `(timestamp_ms, code)` pairs.
//! * The [`DictionaryBlock`] bundles dictionary + encoded stream and supports
//!   efficient random-access lookup by timestamp.
//!
//! ## Compression characteristics
//!
//! If there are `V` distinct values and `N` total samples, storage is roughly:
//!
//! ```text
//! V × avg_string_len  +  N × 4 bytes  (vs N × avg_string_len uncompressed)
//! ```
//!
//! This is especially effective when `V << N`.

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DictionaryEncoder
// ---------------------------------------------------------------------------

/// Streaming encoder for `(timestamp_ms, string_value)` pairs.
///
/// Assign compact `u32` codes to unique string values and build a
/// [`DictionaryBlock`] via [`finish`](Self::finish).
pub struct DictionaryEncoder {
    /// Forward map: string value → code
    dictionary: HashMap<String, u32>,
    /// Reverse map: code → string value (index == code)
    reverse: Vec<String>,
    /// Encoded `(timestamp_ms, code)` pairs
    encoded: Vec<(i64, u32)>,
}

impl DictionaryEncoder {
    /// Create a new, empty encoder.
    pub fn new() -> Self {
        Self {
            dictionary: HashMap::new(),
            reverse: Vec::new(),
            encoded: Vec::new(),
        }
    }

    /// Encode a single `(timestamp_ms, value)` sample.
    ///
    /// Returns the assigned code for `value`.
    pub fn encode(&mut self, timestamp: i64, value: &str) -> TsdbResult<u32> {
        let code = match self.dictionary.get(value) {
            Some(&c) => c,
            None => {
                let next_code = self.reverse.len() as u32;
                if next_code == u32::MAX {
                    return Err(TsdbError::Compression(
                        "Dictionary overflow: more than u32::MAX distinct values".to_string(),
                    ));
                }
                self.dictionary.insert(value.to_owned(), next_code);
                self.reverse.push(value.to_owned());
                next_code
            }
        };
        self.encoded.push((timestamp, code));
        Ok(code)
    }

    /// Consume the encoder and produce a [`DictionaryBlock`].
    pub fn finish(self) -> DictionaryBlock {
        DictionaryBlock {
            dictionary: self.reverse,
            encoded: self.encoded,
        }
    }

    /// Number of distinct values seen so far.
    pub fn cardinality(&self) -> usize {
        self.reverse.len()
    }
}

impl Default for DictionaryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DictionaryBlock
// ---------------------------------------------------------------------------

/// A self-contained, serialisable dictionary-compressed block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DictionaryBlock {
    /// Reverse dictionary: `dictionary[code]` → string value.
    pub dictionary: Vec<String>,
    /// Encoded `(timestamp_ms, code)` pairs.
    pub encoded: Vec<(i64, u32)>,
}

impl DictionaryBlock {
    /// Build a block from a slice of `(timestamp_ms, &str)` pairs.
    pub fn from_data(data: &[(i64, &str)]) -> TsdbResult<Self> {
        let mut enc = DictionaryEncoder::new();
        for &(ts, val) in data {
            enc.encode(ts, val)?;
        }
        Ok(enc.finish())
    }

    /// Decode all `(timestamp_ms, value_str)` pairs.
    pub fn decode(&self) -> TsdbResult<Vec<(i64, &str)>> {
        self.encoded
            .iter()
            .map(|&(ts, code)| {
                self.lookup_code(code).map(|s| (ts, s)).ok_or_else(|| {
                    TsdbError::Decompression(format!("Dictionary: unknown code {}", code))
                })
            })
            .collect()
    }

    /// Decode all samples into owned `String` values.
    pub fn decode_owned(&self) -> TsdbResult<Vec<(i64, String)>> {
        self.encoded
            .iter()
            .map(|&(ts, code)| {
                self.lookup_code(code)
                    .map(|s| (ts, s.to_owned()))
                    .ok_or_else(|| {
                        TsdbError::Decompression(format!("Dictionary: unknown code {}", code))
                    })
            })
            .collect()
    }

    /// Look up the string for a given code.
    ///
    /// Returns `None` if the code is out of range.
    pub fn lookup_code(&self, code: u32) -> Option<&str> {
        self.dictionary.get(code as usize).map(|s| s.as_str())
    }

    /// Find the code assigned to `value`, or `None` if it is not in the
    /// dictionary.
    pub fn find_code(&self, value: &str) -> Option<u32> {
        self.dictionary
            .iter()
            .position(|s| s == value)
            .map(|i| i as u32)
    }

    /// Return the string value at the given `timestamp_ms`, or `None` if no
    /// sample exists at that exact timestamp.
    pub fn get_value(&self, timestamp: i64) -> Option<&str> {
        // Linear search – for large blocks callers should use `decode()` and
        // build their own index.
        let code = self
            .encoded
            .iter()
            .find(|&&(ts, _)| ts == timestamp)
            .map(|&(_, code)| code)?;
        self.lookup_code(code)
    }

    /// Number of distinct values in the dictionary.
    pub fn cardinality(&self) -> usize {
        self.dictionary.len()
    }

    /// Total number of encoded samples.
    pub fn len(&self) -> usize {
        self.encoded.len()
    }

    /// Returns `true` if the block contains no samples.
    pub fn is_empty(&self) -> bool {
        self.encoded.is_empty()
    }

    /// Approximate compression ratio vs. storing raw strings.
    ///
    /// Assumes average string length of 10 bytes and 8-byte timestamp.
    pub fn compression_ratio(&self, avg_string_len: usize) -> f64 {
        let original = self.encoded.len() * (8 + avg_string_len);
        // Dictionary size: sum of string lengths + 4 bytes for pointer
        let dict_bytes: usize = self.dictionary.iter().map(|s| s.len() + 4).sum();
        let encoded_bytes = self.encoded.len() * 12 + dict_bytes; // (i64, u32) = 12 bytes
        if encoded_bytes == 0 {
            1.0
        } else {
            original as f64 / encoded_bytes as f64
        }
    }

    /// Filter encoded samples to a timestamp range `[start, end]` (inclusive).
    pub fn filter_range(&self, start: i64, end: i64) -> TsdbResult<Vec<(i64, &str)>> {
        self.encoded
            .iter()
            .filter(|&&(ts, _)| ts >= start && ts <= end)
            .map(|&(ts, code)| {
                self.lookup_code(code).map(|s| (ts, s)).ok_or_else(|| {
                    TsdbError::Decompression(format!(
                        "Dictionary: unknown code {} at ts {}",
                        code, ts
                    ))
                })
            })
            .collect()
    }

    /// Return the frequency of each distinct value as `(value, count)` pairs,
    /// sorted by count descending.
    pub fn value_frequencies(&self) -> Vec<(&str, usize)> {
        let mut counts: Vec<usize> = vec![0; self.dictionary.len()];
        for &(_, code) in &self.encoded {
            if (code as usize) < counts.len() {
                counts[code as usize] += 1;
            }
        }
        let mut freq: Vec<(&str, usize)> = self
            .dictionary
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), counts[i]))
            .collect();
        freq.sort_by_key(|f| Reverse(f.1));
        freq
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Encode a slice of `(timestamp_ms, &str)` pairs into a [`DictionaryBlock`].
pub fn dict_encode(data: &[(i64, &str)]) -> TsdbResult<DictionaryBlock> {
    DictionaryBlock::from_data(data)
}

/// Decode a [`DictionaryBlock`] into `(timestamp_ms, String)` pairs.
pub fn dict_decode(block: &DictionaryBlock) -> TsdbResult<Vec<(i64, String)>> {
    block.decode_owned()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_encode_decode() {
        let block = DictionaryBlock::from_data(&[]).expect("build block");
        assert!(block.is_empty());
        assert_eq!(block.cardinality(), 0);
        let decoded = block.decode().expect("decode");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_single_sample() {
        let data = vec![(1000i64, "on")];
        let block = DictionaryBlock::from_data(&data).expect("build");
        assert_eq!(block.len(), 1);
        assert_eq!(block.cardinality(), 1);
        let decoded = block.decode().expect("decode");
        assert_eq!(decoded, vec![(1000, "on")]);
    }

    #[test]
    fn test_round_trip_multiple_values() {
        let data = vec![
            (0i64, "idle"),
            (1000, "running"),
            (2000, "running"),
            (3000, "idle"),
            (4000, "error"),
            (5000, "idle"),
            (6000, "running"),
        ];
        let block = DictionaryBlock::from_data(&data).expect("build");
        // Only 3 distinct values
        assert_eq!(block.cardinality(), 3);
        let decoded = block.decode().expect("decode");
        assert_eq!(
            decoded,
            data.iter().map(|&(t, s)| (t, s)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_find_code_and_lookup() {
        let data = vec![(0i64, "alpha"), (1000, "beta"), (2000, "alpha")];
        let block = DictionaryBlock::from_data(&data).expect("build");
        let code_alpha = block.find_code("alpha").expect("alpha should exist");
        let code_beta = block.find_code("beta").expect("beta should exist");
        assert_ne!(code_alpha, code_beta);
        assert_eq!(block.lookup_code(code_alpha), Some("alpha"));
        assert_eq!(block.lookup_code(code_beta), Some("beta"));
        assert!(block.find_code("gamma").is_none());
    }

    #[test]
    fn test_get_value_by_timestamp() {
        let data = vec![(0i64, "off"), (1000, "on"), (2000, "standby")];
        let block = DictionaryBlock::from_data(&data).expect("build");
        assert_eq!(block.get_value(0), Some("off"));
        assert_eq!(block.get_value(1000), Some("on"));
        assert_eq!(block.get_value(2000), Some("standby"));
        assert_eq!(block.get_value(9999), None);
    }

    #[test]
    fn test_filter_range() {
        let data = vec![
            (0i64, "a"),
            (1000, "b"),
            (2000, "c"),
            (3000, "d"),
            (4000, "e"),
        ];
        let block = DictionaryBlock::from_data(&data).expect("build");
        let filtered = block.filter_range(1000, 3000).expect("filter");
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0], (1000, "b"));
        assert_eq!(filtered[1], (2000, "c"));
        assert_eq!(filtered[2], (3000, "d"));
    }

    #[test]
    fn test_value_frequencies() {
        let data = vec![
            (0i64, "a"),
            (1000, "b"),
            (2000, "a"),
            (3000, "a"),
            (4000, "c"),
            (5000, "b"),
        ];
        let block = DictionaryBlock::from_data(&data).expect("build");
        let freq = block.value_frequencies();
        // "a" appears 3 times → should be first
        assert_eq!(freq[0].0, "a");
        assert_eq!(freq[0].1, 3);
        // "b" appears 2 times → second
        assert_eq!(freq[1].0, "b");
        assert_eq!(freq[1].1, 2);
        // "c" appears 1 time → third
        assert_eq!(freq[2].0, "c");
        assert_eq!(freq[2].1, 1);
    }

    #[test]
    fn test_compression_ratio_high_repetition() {
        // 1000 samples with only 2 distinct values (16 char average)
        let data: Vec<(i64, &str)> = (0..1000i64)
            .map(|i| {
                (
                    i * 1000,
                    if i % 2 == 0 {
                        "device_status_on"
                    } else {
                        "device_status_off"
                    },
                )
            })
            .collect();
        let block = DictionaryBlock::from_data(&data).expect("build");
        assert_eq!(block.cardinality(), 2);
        let ratio = block.compression_ratio(16);
        assert!(ratio > 1.0, "should be compressed: {:.2}", ratio);
    }

    #[test]
    fn test_owned_decode() {
        let data = vec![(0i64, "hello"), (1000, "world")];
        let block = DictionaryBlock::from_data(&data).expect("build");
        let decoded = dict_decode(&block).expect("decode");
        assert_eq!(
            decoded,
            vec![(0, "hello".to_string()), (1000, "world".to_string())]
        );
    }

    #[test]
    fn test_encoder_encode_returns_code() {
        let mut enc = DictionaryEncoder::new();
        let code_a = enc.encode(0, "a").expect("ok");
        let code_b = enc.encode(1000, "b").expect("ok");
        let code_a2 = enc.encode(2000, "a").expect("ok");
        // Same value should return same code
        assert_eq!(code_a, code_a2);
        assert_ne!(code_a, code_b);
        assert_eq!(enc.cardinality(), 2);
    }

    #[test]
    fn test_all_unique_values() {
        let data: Vec<(i64, String)> = (0..100).map(|i| (i as i64, format!("val_{}", i))).collect();
        let refs: Vec<(i64, &str)> = data.iter().map(|(t, s)| (*t, s.as_str())).collect();
        let block = DictionaryBlock::from_data(&refs).expect("build");
        assert_eq!(block.cardinality(), 100);
        assert_eq!(block.len(), 100);
        let decoded = block.decode().expect("decode");
        for (i, (ts, val)) in decoded.iter().enumerate() {
            assert_eq!(*ts, i as i64);
            assert_eq!(*val, format!("val_{}", i));
        }
    }

    #[test]
    fn test_empty_string_value() {
        let data = vec![(0i64, ""), (1000, "nonempty"), (2000, "")];
        let block = DictionaryBlock::from_data(&data).expect("build");
        assert_eq!(block.cardinality(), 2);
        let decoded = block.decode().expect("decode");
        assert_eq!(decoded, vec![(0, ""), (1000, "nonempty"), (2000, "")]);
    }
}
