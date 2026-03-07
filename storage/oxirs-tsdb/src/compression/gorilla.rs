//! Gorilla XOR compression for time-series data
//!
//! Based on: "Gorilla: A Fast, Scalable, In-Memory Time Series Database"
//! Pelkonen et al., Facebook Engineering, VLDB 2015.
//!
//! ## Algorithm overview
//!
//! ### Timestamps (delta-of-delta encoding)
//! 1. Store the first timestamp as raw i64.
//! 2. For subsequent timestamps, compute `delta = ts - prev_ts` and
//!    `dod = delta - prev_delta`.
//! 3. Encode `dod` with a variable-length scheme:
//!    - `dod == 0`             → 1 bit: `0`
//!    - `-63 <= dod <= 64`     → 9 bits: `10` + 7-bit signed
//!    - `-255 <= dod <= 256`   → 12 bits: `110` + 9-bit signed
//!    - `-2047 <= dod <= 2048` → 16 bits: `1110` + 12-bit signed
//!    - otherwise             → 68 bits: `1111` + 64-bit signed
//!
//! ### Values (XOR encoding)
//! 1. Store the first value as raw f64 bits (64 bits).
//! 2. For subsequent values, XOR with the previous value.
//!    - `xor == 0`         → 1 bit: `0`
//!    - reuse prev window  → 2 bits: `10` + meaningful bits
//!    - new window         → 2 bits: `11` + 5-bit leading zeros + 6-bit meaningful
//!      (6-bit value 0 represents 64 meaningful bits)
//!
//! ## Important implementation notes
//!
//! * Leading zeros are capped at 31 (5 bits) per the Gorilla paper.
//! * Meaningful bits in range 1..=64; stored in 6 bits where `0` means 64.
//! * Shift amounts are always guarded to avoid UB/panic.

use crate::error::{TsdbError, TsdbResult};
use bit_vec::BitVec;
use std::ops::Range;

// ---------------------------------------------------------------------------
// Bit buffer (write side)
// ---------------------------------------------------------------------------

/// Write-only bit buffer backed by a `Vec<u8>` (msb-first within each byte).
pub struct BitWriter {
    data: Vec<u8>,
    /// Number of valid bits in the last byte (0..=7).
    pending_bits: u8,
}

impl BitWriter {
    /// Create an empty `BitWriter`.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            pending_bits: 0,
        }
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        if self.pending_bits == 0 {
            self.data.push(0);
        }
        if bit {
            let last = self.data.len() - 1;
            self.data[last] |= 1 << (7 - self.pending_bits);
        }
        self.pending_bits = (self.pending_bits + 1) % 8;
    }

    /// Write the `num_bits` least-significant bits of `value` (msb first).
    /// `num_bits` must be in range 1..=64.
    #[inline]
    pub fn write_bits(&mut self, value: u64, num_bits: u8) {
        debug_assert!((1..=64).contains(&num_bits));
        for shift in (0..num_bits).rev() {
            self.write_bit((value >> shift) & 1 == 1);
        }
    }

    /// Write a signed integer using `num_bits` bits (two's complement, msb first).
    pub fn write_signed(&mut self, value: i64, num_bits: u8) {
        let mask: u64 = if num_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << num_bits) - 1
        };
        self.write_bits((value as u64) & mask, num_bits);
    }

    /// Consume the writer and return the underlying bytes.
    pub fn finish(self) -> Vec<u8> {
        self.data
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Gorilla encoder
// ---------------------------------------------------------------------------

/// Encode a stream of `(timestamp_ms, value)` pairs using Gorilla encoding.
///
/// Wire format:
/// ```text
/// [GORILLA_MAGIC: u32 be][count: u32 be][bit-stream bytes…]
/// ```
pub struct GorillaEncoder {
    /// Leading zeros of previous XOR (255 = sentinel for "no previous XOR").
    leading_zeros: u8,
    /// Trailing zeros of previous XOR.
    trailing_zeros: u8,
    last_timestamp: i64,
    last_delta: i64,
    last_value_bits: u64,
    writer: BitWriter,
    count: u32,
    has_first: bool,
}

/// Magic number prefix that identifies a Gorilla-encoded block.
pub const GORILLA_MAGIC: u32 = 0x474F5201; // "GOR\x01"

impl GorillaEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self {
            leading_zeros: 255, // sentinel: no previous XOR
            trailing_zeros: 0,
            last_timestamp: 0,
            last_delta: 0,
            last_value_bits: 0,
            writer: BitWriter::new(),
            count: 0,
            has_first: false,
        }
    }

    /// Encode the very first `(timestamp_ms, value)` sample (uncompressed).
    pub fn encode_first(&mut self, timestamp: i64, value: f64) {
        debug_assert!(!self.has_first, "encode_first called twice");
        self.writer.write_signed(timestamp, 64);
        self.writer.write_bits(value.to_bits(), 64);
        self.last_timestamp = timestamp;
        self.last_delta = 0;
        self.last_value_bits = value.to_bits();
        self.count = 1;
        self.has_first = true;
    }

    /// Encode a subsequent `(timestamp_ms, value)` sample.
    pub fn encode(&mut self, timestamp: i64, value: f64) -> TsdbResult<()> {
        if !self.has_first {
            return Err(TsdbError::Compression(
                "encode_first must be called before encode".to_string(),
            ));
        }
        self.encode_timestamp(timestamp);
        self.encode_value(value);
        self.count += 1;
        Ok(())
    }

    fn encode_timestamp(&mut self, timestamp: i64) {
        let delta = timestamp.wrapping_sub(self.last_timestamp);
        let dod = delta.wrapping_sub(self.last_delta);
        match dod {
            0 => {
                self.writer.write_bit(false);
            }
            -63..=64 => {
                self.writer.write_bit(true);
                self.writer.write_bit(false);
                self.writer.write_signed(dod, 7);
            }
            -255..=256 => {
                self.writer.write_bit(true);
                self.writer.write_bit(true);
                self.writer.write_bit(false);
                self.writer.write_signed(dod, 9);
            }
            -2047..=2048 => {
                self.writer.write_bit(true);
                self.writer.write_bit(true);
                self.writer.write_bit(true);
                self.writer.write_bit(false);
                self.writer.write_signed(dod, 12);
            }
            _ => {
                self.writer.write_bit(true);
                self.writer.write_bit(true);
                self.writer.write_bit(true);
                self.writer.write_bit(true);
                self.writer.write_signed(dod, 64);
            }
        }
        self.last_timestamp = timestamp;
        self.last_delta = delta;
    }

    fn encode_value(&mut self, value: f64) {
        let bits = value.to_bits();
        let xor = bits ^ self.last_value_bits;

        if xor == 0 {
            self.writer.write_bit(false);
        } else {
            self.writer.write_bit(true);

            let leading = (xor.leading_zeros() as u8).min(31); // cap at 31 (5 bits)
            let trailing = xor.trailing_zeros() as u8;
            // meaningful is always in range 1..=64
            let meaningful = 64u8.saturating_sub(leading).saturating_sub(trailing).max(1);

            // Reuse previous window if XOR fits within prior window dimensions.
            let can_reuse = self.leading_zeros != 255
                && leading >= self.leading_zeros
                && trailing >= self.trailing_zeros;

            if can_reuse {
                self.writer.write_bit(false); // reuse
                let prev_meaningful = 64u8
                    .saturating_sub(self.leading_zeros)
                    .saturating_sub(self.trailing_zeros)
                    .max(1);
                // Shift right by trailing_zeros; guard against >= 64
                let shifted = safe_shr_u64(xor, self.trailing_zeros as u32);
                self.writer.write_bits(shifted, prev_meaningful);
            } else {
                self.writer.write_bit(true); // new window
                self.writer.write_bits(leading as u64, 5);
                // Encode meaningful: 6 bits; value 0 represents 64.
                let meaningful_enc: u8 = if meaningful == 64 { 0 } else { meaningful };
                self.writer.write_bits(meaningful_enc as u64, 6);
                let shifted = safe_shr_u64(xor, trailing as u32);
                self.writer.write_bits(shifted, meaningful);
                self.leading_zeros = leading;
                self.trailing_zeros = trailing;
            }
        }
        self.last_value_bits = bits;
    }

    /// Finish encoding and return the compressed byte vector.
    pub fn finish(self) -> Vec<u8> {
        let bit_bytes = self.writer.finish();
        let mut out = Vec::with_capacity(8 + bit_bytes.len());
        out.extend_from_slice(&GORILLA_MAGIC.to_be_bytes());
        out.extend_from_slice(&self.count.to_be_bytes());
        out.extend_from_slice(&bit_bytes);
        out
    }
}

impl Default for GorillaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Gorilla decoder
// ---------------------------------------------------------------------------

/// Decode a stream produced by [`GorillaEncoder`].
pub struct GorillaDecoder {
    bits: BitVec,
    pos: usize,
    last_timestamp: i64,
    last_delta: i64,
    last_value_bits: u64,
    leading_zeros: u8,
    trailing_zeros: u8,
    total: u32,
    done: u32,
}

impl GorillaDecoder {
    /// Construct a decoder from compressed bytes.
    pub fn new(data: &[u8]) -> TsdbResult<Self> {
        if data.len() < 8 {
            return Err(TsdbError::Decompression(
                "Gorilla data too short: missing header".to_string(),
            ));
        }
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if magic != GORILLA_MAGIC {
            return Err(TsdbError::Decompression(format!(
                "Gorilla magic mismatch: expected 0x{:08X}, got 0x{:08X}",
                GORILLA_MAGIC, magic
            )));
        }
        let total = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        if total == 0 {
            return Ok(Self {
                bits: BitVec::new(),
                pos: 0,
                last_timestamp: 0,
                last_delta: 0,
                last_value_bits: 0,
                leading_zeros: 255,
                trailing_zeros: 0,
                total: 0,
                done: 0,
            });
        }

        let bits = BitVec::from_bytes(&data[8..]);
        if bits.len() < 128 {
            return Err(TsdbError::Decompression(
                "Gorilla data too short: missing first sample".to_string(),
            ));
        }

        // Read first timestamp (64 bits signed)
        let ts = read_bits_signed(&bits, 0..64);
        // Read first value (64 bits IEEE-754)
        let val_bits = read_bits_u64(&bits, 64..128);

        Ok(Self {
            bits,
            pos: 128,
            last_timestamp: ts,
            last_delta: 0,
            last_value_bits: val_bits,
            leading_zeros: 255,
            trailing_zeros: 0,
            total,
            done: 1, // first sample is already loaded
        })
    }

    /// Decode all `(timestamp_ms, value)` pairs.
    pub fn decode_all(&mut self) -> TsdbResult<Vec<(i64, f64)>> {
        if self.total == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(self.total as usize);
        // Emit the first sample (already read in `new`)
        out.push((self.last_timestamp, f64::from_bits(self.last_value_bits)));

        while let Some(result) = self.decode_next() {
            out.push(result?);
        }
        Ok(out)
    }

    /// Decode the next sample after the first, or `None` when exhausted.
    pub fn decode_next(&mut self) -> Option<TsdbResult<(i64, f64)>> {
        if self.done >= self.total {
            return None;
        }
        let ts = match self.decode_timestamp() {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let val = match self.decode_value() {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        self.done += 1;
        Some(Ok((ts, val)))
    }

    fn read_bit(&mut self) -> Option<bool> {
        if self.pos >= self.bits.len() {
            None
        } else {
            let b = self.bits[self.pos];
            self.pos += 1;
            Some(b)
        }
    }

    fn require_bits_u64(&mut self, n: usize) -> TsdbResult<u64> {
        if self.pos + n > self.bits.len() {
            return Err(TsdbError::Decompression(
                "Gorilla stream truncated while reading bits".to_string(),
            ));
        }
        let v = read_bits_u64(&self.bits, self.pos..self.pos + n);
        self.pos += n;
        Ok(v)
    }

    fn require_bits_signed(&mut self, n: usize) -> TsdbResult<i64> {
        let raw = self.require_bits_u64(n)?;
        Ok(sign_extend(raw, n))
    }

    fn decode_timestamp(&mut self) -> TsdbResult<i64> {
        let b0 = self.read_bit().ok_or_else(|| {
            TsdbError::Decompression(
                "Gorilla: unexpected end reading timestamp prefix bit 0".to_string(),
            )
        })?;
        let dod: i64 = if !b0 {
            0
        } else {
            let b1 = self.read_bit().ok_or_else(|| {
                TsdbError::Decompression(
                    "Gorilla: unexpected end reading timestamp prefix bit 1".to_string(),
                )
            })?;
            if !b1 {
                self.require_bits_signed(7)?
            } else {
                let b2 = self.read_bit().ok_or_else(|| {
                    TsdbError::Decompression(
                        "Gorilla: unexpected end reading timestamp prefix bit 2".to_string(),
                    )
                })?;
                if !b2 {
                    self.require_bits_signed(9)?
                } else {
                    let b3 = self.read_bit().ok_or_else(|| {
                        TsdbError::Decompression(
                            "Gorilla: unexpected end reading timestamp prefix bit 3".to_string(),
                        )
                    })?;
                    if !b3 {
                        self.require_bits_signed(12)?
                    } else {
                        self.require_bits_signed(64)?
                    }
                }
            }
        };
        let delta = self.last_delta.wrapping_add(dod);
        let ts = self.last_timestamp.wrapping_add(delta);
        self.last_delta = delta;
        self.last_timestamp = ts;
        Ok(ts)
    }

    fn decode_value(&mut self) -> TsdbResult<f64> {
        let changed = self.read_bit().ok_or_else(|| {
            TsdbError::Decompression("Gorilla: unexpected end reading value change bit".to_string())
        })?;
        if !changed {
            return Ok(f64::from_bits(self.last_value_bits));
        }

        let new_window = self.read_bit().ok_or_else(|| {
            TsdbError::Decompression("Gorilla: unexpected end reading value window bit".to_string())
        })?;

        let xor = if !new_window && self.leading_zeros != 255 {
            // Reuse previous window
            let prev_meaningful = 64u8
                .saturating_sub(self.leading_zeros)
                .saturating_sub(self.trailing_zeros)
                .max(1);
            let shifted = self.require_bits_u64(prev_meaningful as usize)?;
            safe_shl_u64(shifted, self.trailing_zeros as u32)
        } else {
            // New window: read 5-bit leading, 6-bit meaningful
            let leading = self.require_bits_u64(5)? as u8;
            let meaningful_enc = self.require_bits_u64(6)? as u8;
            // 0 encodes as 64 meaningful bits
            let meaningful: u8 = if meaningful_enc == 0 {
                64
            } else {
                meaningful_enc
            };
            let trailing = 64u8.saturating_sub(leading).saturating_sub(meaningful);
            let shifted = self.require_bits_u64(meaningful as usize)?;
            self.leading_zeros = leading;
            self.trailing_zeros = trailing;
            safe_shl_u64(shifted, trailing as u32)
        };

        let val_bits = self.last_value_bits ^ xor;
        self.last_value_bits = val_bits;
        Ok(f64::from_bits(val_bits))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Shift right by `n` bits, returning 0 for shifts >= 64.
#[inline]
fn safe_shr_u64(v: u64, n: u32) -> u64 {
    if n >= 64 {
        0
    } else {
        v >> n
    }
}

/// Shift left by `n` bits, returning 0 for shifts >= 64.
#[inline]
fn safe_shl_u64(v: u64, n: u32) -> u64 {
    if n >= 64 {
        0
    } else {
        v << n
    }
}

fn read_bits_u64(bits: &BitVec, range: Range<usize>) -> u64 {
    let len = range.len();
    let mut v: u64 = 0;
    for (i, idx) in range.enumerate() {
        if idx < bits.len() && bits[idx] {
            v |= 1u64 << (len - 1 - i);
        }
    }
    v
}

fn read_bits_signed(bits: &BitVec, range: Range<usize>) -> i64 {
    let n = range.len();
    let raw = read_bits_u64(bits, range);
    sign_extend(raw, n)
}

fn sign_extend(value: u64, bits: usize) -> i64 {
    if bits == 0 {
        return 0;
    }
    if bits >= 64 {
        return value as i64;
    }
    let sign_bit = 1u64 << (bits - 1);
    if value & sign_bit != 0 {
        (value | (!0u64 << bits)) as i64
    } else {
        value as i64
    }
}

/// Convenience function: encode a slice of `(timestamp_ms, value)` pairs.
pub fn gorilla_encode(data: &[(i64, f64)]) -> TsdbResult<Vec<u8>> {
    if data.is_empty() {
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&GORILLA_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        return Ok(out);
    }
    let mut enc = GorillaEncoder::new();
    let (first_ts, first_val) = data[0];
    enc.encode_first(first_ts, first_val);
    for &(ts, val) in &data[1..] {
        enc.encode(ts, val)?;
    }
    Ok(enc.finish())
}

/// Convenience function: decode bytes produced by [`gorilla_encode`].
pub fn gorilla_decode(data: &[u8]) -> TsdbResult<Vec<(i64, f64)>> {
    let mut dec = GorillaDecoder::new(data)?;
    dec.decode_all()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[(i64, f64)]) {
        let compressed = gorilla_encode(data).expect("encode failed");
        let decoded = gorilla_decode(&compressed).expect("decode failed");
        assert_eq!(
            data.len(),
            decoded.len(),
            "length mismatch: expected {} got {}",
            data.len(),
            decoded.len()
        );
        for (i, (&orig, &dec)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig.0, dec.0,
                "sample {} timestamp mismatch: expected {} got {}",
                i, orig.0, dec.0
            );
            assert_eq!(
                orig.1.to_bits(),
                dec.1.to_bits(),
                "sample {} value mismatch: expected {:?} got {:?}",
                i,
                orig.1,
                dec.1
            );
        }
    }

    #[test]
    fn test_empty_encode_decode() {
        let compressed = gorilla_encode(&[]).expect("encode failed");
        let decoded = gorilla_decode(&compressed).expect("decode failed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_single_sample() {
        round_trip(&[(1_700_000_000_000, 42.0)]);
    }

    #[test]
    fn test_constant_values() {
        let data: Vec<(i64, f64)> = (0..1000)
            .map(|i| (1_700_000_000_000 + i * 1000, 25.0))
            .collect();
        let compressed = gorilla_encode(&data).expect("encode failed");
        let ratio = (data.len() * 16) as f64 / compressed.len() as f64;
        assert!(ratio > 20.0, "compression ratio too low: {:.1}", ratio);
        let decoded = gorilla_decode(&compressed).expect("decode failed");
        assert_eq!(data.len(), decoded.len());
        for (&orig, &dec) in data.iter().zip(decoded.iter()) {
            assert_eq!(orig.0, dec.0);
            assert_eq!(orig.1.to_bits(), dec.1.to_bits());
        }
    }

    #[test]
    fn test_regular_timestamps_small_value_variance() {
        let base_ts = 1_640_000_000_000i64;
        let data: Vec<(i64, f64)> = (0..500)
            .map(|i| {
                let ts = base_ts + i as i64 * 1000;
                let val = 20.0 + (i % 10) as f64 * 0.1;
                (ts, val)
            })
            .collect();
        round_trip(&data);
        let compressed = gorilla_encode(&data).expect("encode");
        let ratio = (data.len() * 16) as f64 / compressed.len() as f64;
        assert!(ratio > 2.0, "ratio too low: {:.1}", ratio);
    }

    #[test]
    fn test_irregular_timestamps() {
        let data: Vec<(i64, f64)> = vec![
            (1000, 1.0),
            (2500, 2.0),
            (3100, 2.0),
            (10000, 3.5),
            (10001, 3.5),
            (10002, 4.0),
        ];
        round_trip(&data);
    }

    #[test]
    fn test_special_float_values() {
        let data = vec![
            (0i64, 0.0_f64),
            (1000, -0.0_f64),
            (2000, f64::MAX),
            (3000, f64::MIN),
            (4000, f64::MIN_POSITIVE),
            (5000, 1.0 / 3.0),
        ];
        round_trip(&data);
    }

    #[test]
    fn test_large_timestamp_jumps() {
        // Timestamps spanning large ranges (wrapping arithmetic protects us)
        let data = vec![
            (0i64, 1.0),
            (1_000_000_000_000i64, 2.0),
            (1_000_000_000_001i64, 3.0),
        ];
        round_trip(&data);
    }

    #[test]
    fn test_magic_mismatch() {
        let bad: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x01];
        let result = gorilla_decode(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_two_samples() {
        round_trip(&[(0, 1.23456789), (1000, 9.87654321)]);
    }

    #[test]
    fn test_alternating_values() {
        let data: Vec<(i64, f64)> = (0..200)
            .map(|i| (i as i64 * 100, if i % 2 == 0 { 0.0 } else { 1.0 }))
            .collect();
        round_trip(&data);
    }

    #[test]
    fn test_monotonically_increasing() {
        let data: Vec<(i64, f64)> = (0..100)
            .map(|i| (i as i64 * 1000, i as f64 * 1.5))
            .collect();
        round_trip(&data);
    }

    #[test]
    fn test_negative_timestamps() {
        // Timestamps before Unix epoch
        let data: Vec<(i64, f64)> = vec![(-1_000_000, 1.0), (-999_000, 2.0), (-998_000, 3.0)];
        round_trip(&data);
    }

    #[test]
    fn test_all_same_value_then_different() {
        let mut data: Vec<(i64, f64)> = (0..50).map(|i| (i as i64 * 1000, 7.0)).collect();
        data.push((50_000, 99.0));
        round_trip(&data);
    }

    #[test]
    fn test_compression_better_than_raw_for_sensor_data() {
        let data: Vec<(i64, f64)> = (0..1000)
            .map(|i| (i as i64 * 1000, 22.0 + (i % 5) as f64 * 0.1))
            .collect();
        let compressed = gorilla_encode(&data).expect("encode");
        let raw_size = data.len() * 16;
        assert!(
            compressed.len() < raw_size,
            "compressed ({}) should be smaller than raw ({})",
            compressed.len(),
            raw_size
        );
    }

    #[test]
    fn test_xor_of_zero_and_negative_zero() {
        // +0.0 and -0.0 have different bit patterns; XOR != 0, so they encode separately.
        let data = vec![(0i64, 0.0f64), (1000, -0.0f64)];
        round_trip(&data);
    }

    #[test]
    fn test_large_sequence_correct() {
        // 10K samples to stress test the codec
        let data: Vec<(i64, f64)> = (0..10_000)
            .map(|i| {
                let ts = 1_700_000_000_000i64 + i as i64 * 100;
                // Values with different XOR patterns
                let val = match i % 4 {
                    0 => 1.0,
                    1 => 2.0,
                    2 => 1.5,
                    _ => std::f64::consts::PI,
                };
                (ts, val)
            })
            .collect();
        round_trip(&data);
    }
}
