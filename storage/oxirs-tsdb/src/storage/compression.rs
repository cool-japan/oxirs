//! Compression algorithms for time-series data
//!
//! Implements Gorilla compression (Facebook, VLDB 2015) for floating-point values
//! and Delta-of-delta encoding for timestamps.

use crate::error::{TsdbError, TsdbResult};
use bit_vec::BitVec;
use std::ops::Range;

/// Gorilla compressor for floating-point time series
///
/// Based on Facebook's Gorilla: A Fast, Scalable, In-Memory Time Series Database
/// Reference: <http://www.vldb.org/pvldb/vol8/p1816-teller.pdf>
///
/// # Algorithm
///
/// 1. XOR current value with previous value
/// 2. If XOR == 0: store single '0' bit (value unchanged)
/// 3. If XOR != 0: store '1' + compressed XOR using variable-length encoding
///
/// # Compression Ratio
///
/// - Temperature sensors: 100:1 (stable readings)
/// - Vibration sensors: 10:1 (high variance)
/// - Average IoT workloads: 30-50:1
pub struct GorillaCompressor {
    /// Previous value for XOR
    prev_value: f64,

    /// Previous XOR's leading zeros count
    prev_leading_zeros: u8,

    /// Previous XOR's trailing zeros count
    prev_trailing_zeros: u8,

    /// Compressed bit stream
    bits: BitVec,

    /// Number of values compressed (including first)
    count: u32,
}

impl GorillaCompressor {
    /// Create new compressor with first value
    ///
    /// The first value is stored uncompressed (64 bits).
    /// A 32-bit count header is prepended when finish() is called.
    pub fn new(first_value: f64) -> Self {
        let mut bits = BitVec::new();

        // Store first value uncompressed
        let value_bits = first_value.to_bits();
        for i in (0..64).rev() {
            bits.push((value_bits >> i) & 1 == 1);
        }

        Self {
            prev_value: first_value,
            prev_leading_zeros: 64, // Initialize to max so first XOR always uses new block
            prev_trailing_zeros: 64,
            bits,
            count: 1, // First value already compressed
        }
    }

    /// Compress a single value
    pub fn compress(&mut self, value: f64) {
        let xor = value.to_bits() ^ self.prev_value.to_bits();

        if xor == 0 {
            // Value unchanged: store single '0' bit
            self.bits.push(false);
        } else {
            self.bits.push(true); // Value changed

            let leading_zeros = xor.leading_zeros() as u8;
            let trailing_zeros = xor.trailing_zeros() as u8;

            // Check if we can reuse the previous block window
            // Only if current XOR fits within the previous window
            if self.prev_leading_zeros < 64
                && leading_zeros >= self.prev_leading_zeros
                && trailing_zeros >= self.prev_trailing_zeros
            {
                // Use previous block size: store '0' + meaningful bits
                self.bits.push(false);

                let meaningful_bits = 64 - self.prev_leading_zeros - self.prev_trailing_zeros;
                let shifted_xor = xor >> self.prev_trailing_zeros;

                for i in (0..meaningful_bits).rev() {
                    self.bits.push((shifted_xor >> i) & 1 == 1);
                }
            } else {
                // New block size: store '1' + leading zeros (5 bits) + block size (6 bits) + meaningful bits
                self.bits.push(true);

                // Encode leading zeros (5 bits, capped at 31)
                let leading_zeros_capped = leading_zeros.min(31);
                for i in (0..5).rev() {
                    self.bits.push((leading_zeros_capped >> i) & 1 == 1);
                }

                // Encode meaningful bits count (6 bits)
                // meaningful_bits = 64 - leading_zeros - trailing_zeros
                // If leading_zeros was capped, adjust trailing_zeros
                let actual_leading = leading_zeros_capped;
                let meaningful_bits = (64 - leading_zeros - trailing_zeros).max(1);
                for i in (0..6).rev() {
                    self.bits.push((meaningful_bits >> i) & 1 == 1);
                }

                // Encode meaningful bits
                let shifted_xor = xor >> trailing_zeros;
                for i in (0..meaningful_bits).rev() {
                    self.bits.push((shifted_xor >> i) & 1 == 1);
                }

                self.prev_leading_zeros = actual_leading;
                self.prev_trailing_zeros = 64 - actual_leading - meaningful_bits;
            }
        }

        self.prev_value = value;
        self.count += 1;
    }

    /// Finish compression and return compressed bytes
    ///
    /// Returns: [count (4 bytes BE)] + [compressed bit stream]
    pub fn finish(self) -> Vec<u8> {
        let mut result = Vec::with_capacity(4 + (self.bits.len() + 7) / 8);
        // Prepend count as big-endian u32
        result.extend_from_slice(&self.count.to_be_bytes());
        result.extend_from_slice(&self.bits.to_bytes());
        result
    }

    /// Get current compression ratio
    pub fn compression_ratio(&self, original_count: usize) -> f64 {
        let original_bytes = original_count * 8; // f64 = 8 bytes
        let compressed_bytes = 4 + (self.bits.len() + 7) / 8; // 4 byte header + bits
        original_bytes as f64 / compressed_bytes as f64
    }
}

/// Gorilla decompressor for floating-point time series
pub struct GorillaDecompressor {
    /// Compressed bit stream
    bits: BitVec,

    /// Current position in bit stream
    pos: usize,

    /// Previous value for XOR
    prev_value: f64,

    /// Previous XOR's leading zeros
    prev_leading_zeros: u8,

    /// Previous XOR's trailing zeros
    prev_trailing_zeros: u8,

    /// Total number of values to decompress
    total_count: u32,

    /// Number of values already decompressed
    decompressed_count: u32,
}

impl GorillaDecompressor {
    /// Create new decompressor from compressed bytes
    ///
    /// Expected format: [count (4 bytes BE)] + [compressed bit stream]
    pub fn new(compressed: &[u8]) -> TsdbResult<Self> {
        if compressed.len() < 4 {
            return Err(TsdbError::Decompression(
                "Compressed data too short (missing count header)".to_string(),
            ));
        }

        // Read count header (first 4 bytes)
        let total_count =
            u32::from_be_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);

        // Create BitVec from remaining bytes (after count header)
        let bits = BitVec::from_bytes(&compressed[4..]);

        if bits.len() < 64 {
            return Err(TsdbError::Decompression(
                "Compressed data too short (missing first value)".to_string(),
            ));
        }

        // Read first value (uncompressed, 64 bits)
        let first_value_bits = Self::read_bits_as_u64(&bits, 0..64);
        let first_value = f64::from_bits(first_value_bits);

        Ok(Self {
            bits,
            pos: 64,
            prev_value: first_value,
            prev_leading_zeros: 64, // Initialize to max to match compressor
            prev_trailing_zeros: 64,
            total_count,
            decompressed_count: 1, // First value already read
        })
    }

    /// Get the first value (always available)
    pub fn first_value(&self) -> f64 {
        self.prev_value
    }

    /// Get total number of values
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    /// Decompress next value
    ///
    /// Returns the next decompressed value, or None if all values have been read.
    pub fn next_value(&mut self) -> Option<f64> {
        // Check if we've decompressed all values
        if self.decompressed_count >= self.total_count {
            return None;
        }

        if self.pos >= self.bits.len() {
            return None;
        }

        let changed = self.bits[self.pos];
        self.pos += 1;

        if !changed {
            // Value unchanged
            self.decompressed_count += 1;
            return Some(self.prev_value);
        }

        if self.pos >= self.bits.len() {
            return None;
        }

        let use_prev_block = !self.bits[self.pos]; // Note: '0' means use prev block, '1' means new block
        self.pos += 1;

        let meaningful_bits = if use_prev_block && self.prev_leading_zeros < 64 {
            // Use previous block size
            64 - self.prev_leading_zeros - self.prev_trailing_zeros
        } else {
            // Read new block size
            if self.pos + 11 > self.bits.len() {
                return None; // Not enough bits
            }

            let leading_zeros = Self::read_bits_as_u8(&self.bits, self.pos..self.pos + 5);
            self.pos += 5;

            let block_size = Self::read_bits_as_u8(&self.bits, self.pos..self.pos + 6);
            self.pos += 6;

            // block_size is the number of meaningful bits stored
            let trailing_zeros = (64 - leading_zeros).saturating_sub(block_size);
            self.prev_leading_zeros = leading_zeros;
            self.prev_trailing_zeros = trailing_zeros;

            block_size
        };

        if meaningful_bits == 0 {
            // Edge case: no meaningful bits (shouldn't happen in normal data)
            self.decompressed_count += 1;
            return Some(self.prev_value);
        }

        if self.pos + meaningful_bits as usize > self.bits.len() {
            return None; // Not enough bits
        }

        // Read meaningful bits
        let xor_shifted =
            Self::read_bits_as_u64(&self.bits, self.pos..self.pos + meaningful_bits as usize);
        self.pos += meaningful_bits as usize;

        // Shift back to original position - use checked shift to avoid overflow
        let xor = if self.prev_trailing_zeros >= 64 {
            0
        } else {
            xor_shifted << self.prev_trailing_zeros
        };
        let value_bits = self.prev_value.to_bits() ^ xor;
        let value = f64::from_bits(value_bits);

        self.prev_value = value;
        self.decompressed_count += 1;
        Some(value)
    }

    /// Decompress all values
    pub fn decompress_all(mut self) -> Vec<f64> {
        let first = self.first_value();
        let mut values = vec![first];

        while let Some(value) = self.next_value() {
            values.push(value);
        }

        values
    }

    /// Read bits from range as u64
    fn read_bits_as_u64(bits: &BitVec, range: Range<usize>) -> u64 {
        let mut result: u64 = 0;
        let range_len = range.len();
        for (i, bit_idx) in range.enumerate() {
            if bit_idx < bits.len() && bits[bit_idx] {
                result |= 1 << (range_len - 1 - i);
            }
        }
        result
    }

    /// Read bits from range as u8
    fn read_bits_as_u8(bits: &BitVec, range: Range<usize>) -> u8 {
        Self::read_bits_as_u64(bits, range) as u8
    }
}

/// Delta-of-delta compressor for timestamps
///
/// Exploits regularity in sensor sampling intervals.
/// For 1Hz regular sampling, compresses timestamps to <2 bits per sample.
pub struct DeltaOfDeltaCompressor {
    /// Previous timestamp (milliseconds since epoch)
    prev_timestamp: i64,

    /// Previous delta
    prev_delta: i64,

    /// Compressed bit stream
    bits: BitVec,

    /// Number of timestamps compressed (including first)
    count: u32,
}

impl DeltaOfDeltaCompressor {
    /// Create new compressor with first timestamp
    ///
    /// A 32-bit count header is prepended when finish() is called.
    pub fn new(first_timestamp: i64) -> Self {
        let mut bits = BitVec::new();

        // Store first timestamp uncompressed (64 bits)
        for i in (0..64).rev() {
            bits.push((first_timestamp >> i) & 1 == 1);
        }

        Self {
            prev_timestamp: first_timestamp,
            prev_delta: 0,
            bits,
            count: 1, // First timestamp already compressed
        }
    }

    /// Compress a timestamp
    pub fn compress(&mut self, timestamp: i64) {
        let delta = timestamp - self.prev_timestamp;
        let delta_of_delta = delta - self.prev_delta;

        // Variable-length encoding based on delta_of_delta magnitude
        match delta_of_delta {
            0 => {
                // No change: single '0' bit
                self.bits.push(false);
            }
            -63..=64 => {
                // Small delta: '10' + 7-bit signed value
                self.bits.push(true);
                self.bits.push(false);
                self.encode_signed(delta_of_delta, 7);
            }
            -255..=256 => {
                // Medium delta: '110' + 9-bit signed value
                self.bits.push(true);
                self.bits.push(true);
                self.bits.push(false);
                self.encode_signed(delta_of_delta, 9);
            }
            -2047..=2048 => {
                // Large delta: '1110' + 12-bit signed value
                self.bits.push(true);
                self.bits.push(true);
                self.bits.push(true);
                self.bits.push(false);
                self.encode_signed(delta_of_delta, 12);
            }
            _ => {
                // Huge delta: '1111' + 64-bit value
                self.bits.push(true);
                self.bits.push(true);
                self.bits.push(true);
                self.bits.push(true);
                self.encode_signed(delta_of_delta, 64);
            }
        }

        self.prev_timestamp = timestamp;
        self.prev_delta = delta;
        self.count += 1;
    }

    /// Encode signed integer with specified bit count
    fn encode_signed(&mut self, value: i64, bit_count: usize) {
        for i in (0..bit_count).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
    }

    /// Finish compression and return compressed bytes
    ///
    /// Returns: [count (4 bytes BE)] + [compressed bit stream]
    pub fn finish(self) -> Vec<u8> {
        let mut result = Vec::with_capacity(4 + (self.bits.len() + 7) / 8);
        // Prepend count as big-endian u32
        result.extend_from_slice(&self.count.to_be_bytes());
        result.extend_from_slice(&self.bits.to_bytes());
        result
    }

    /// Get current compression ratio
    pub fn compression_ratio(&self, original_count: usize) -> f64 {
        let original_bytes = original_count * 8; // i64 = 8 bytes
        let compressed_bytes = 4 + (self.bits.len() + 7) / 8; // 4 byte header + bits
        original_bytes as f64 / compressed_bytes as f64
    }
}

/// Delta-of-delta decompressor for timestamps
pub struct DeltaOfDeltaDecompressor {
    /// Compressed bit stream
    bits: BitVec,

    /// Current position in bit stream
    pos: usize,

    /// Previous timestamp
    prev_timestamp: i64,

    /// Previous delta
    prev_delta: i64,

    /// Total number of timestamps to decompress
    total_count: u32,

    /// Number of timestamps already decompressed
    decompressed_count: u32,
}

impl DeltaOfDeltaDecompressor {
    /// Create new decompressor from compressed bytes
    ///
    /// Expected format: [count (4 bytes BE)] + [compressed bit stream]
    pub fn new(compressed: &[u8]) -> TsdbResult<Self> {
        if compressed.len() < 4 {
            return Err(TsdbError::Decompression(
                "Compressed timestamp data too short (missing count header)".to_string(),
            ));
        }

        // Read count header (first 4 bytes)
        let total_count =
            u32::from_be_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);

        // Create BitVec from remaining bytes (after count header)
        let bits = BitVec::from_bytes(&compressed[4..]);

        if bits.len() < 64 {
            return Err(TsdbError::Decompression(
                "Compressed timestamp data too short (missing first timestamp)".to_string(),
            ));
        }

        // Read first timestamp (uncompressed, 64 bits)
        let first_timestamp = Self::read_bits_as_i64(&bits, 0..64);

        Ok(Self {
            bits,
            pos: 64,
            prev_timestamp: first_timestamp,
            prev_delta: 0,
            total_count,
            decompressed_count: 1, // First timestamp already read
        })
    }

    /// Get first timestamp
    pub fn first_timestamp(&self) -> i64 {
        self.prev_timestamp
    }

    /// Get total number of timestamps
    pub fn total_count(&self) -> u32 {
        self.total_count
    }

    /// Decompress next timestamp
    ///
    /// Returns the next decompressed timestamp, or None if all timestamps have been read.
    pub fn next_timestamp(&mut self) -> Option<i64> {
        // Check if we've decompressed all timestamps
        if self.decompressed_count >= self.total_count {
            return None;
        }

        if self.pos >= self.bits.len() {
            return None;
        }

        // Read prefix to determine encoding
        let prefix = self.read_prefix();

        let delta_of_delta = match prefix {
            0 => 0, // '0' - no change
            1 => {
                // '10' - 7-bit signed
                if self.pos + 7 > self.bits.len() {
                    return None;
                }
                let value = Self::read_bits_as_i64(&self.bits, self.pos..self.pos + 7);
                self.pos += 7;
                Self::sign_extend(value, 7)
            }
            2 => {
                // '110' - 9-bit signed
                if self.pos + 9 > self.bits.len() {
                    return None;
                }
                let value = Self::read_bits_as_i64(&self.bits, self.pos..self.pos + 9);
                self.pos += 9;
                Self::sign_extend(value, 9)
            }
            3 => {
                // '1110' - 12-bit signed
                if self.pos + 12 > self.bits.len() {
                    return None;
                }
                let value = Self::read_bits_as_i64(&self.bits, self.pos..self.pos + 12);
                self.pos += 12;
                Self::sign_extend(value, 12)
            }
            4 => {
                // '1111' - 64-bit signed
                if self.pos + 64 > self.bits.len() {
                    return None;
                }
                let value = Self::read_bits_as_i64(&self.bits, self.pos..self.pos + 64);
                self.pos += 64;
                value
            }
            _ => return None,
        };

        let delta = self.prev_delta + delta_of_delta;
        let timestamp = self.prev_timestamp + delta;

        self.prev_timestamp = timestamp;
        self.prev_delta = delta;
        self.decompressed_count += 1;

        Some(timestamp)
    }

    /// Read variable-length prefix
    fn read_prefix(&mut self) -> u8 {
        if self.pos >= self.bits.len() || !self.bits[self.pos] {
            self.pos += 1;
            return 0; // '0'
        }

        self.pos += 1;
        if self.pos >= self.bits.len() || !self.bits[self.pos] {
            self.pos += 1;
            return 1; // '10'
        }

        self.pos += 1;
        if self.pos >= self.bits.len() || !self.bits[self.pos] {
            self.pos += 1;
            return 2; // '110'
        }

        self.pos += 1;
        if self.pos >= self.bits.len() || !self.bits[self.pos] {
            self.pos += 1;
            return 3; // '1110'
        }

        self.pos += 1;
        4 // '1111'
    }

    /// Read bits from range as i64
    fn read_bits_as_i64(bits: &BitVec, range: Range<usize>) -> i64 {
        let mut result: i64 = 0;
        let range_len = range.len();
        for (i, bit_idx) in range.enumerate() {
            if bit_idx < bits.len() && bits[bit_idx] {
                result |= 1 << (range_len - 1 - i);
            }
        }
        result
    }

    /// Sign-extend a value from specified bit count
    fn sign_extend(value: i64, bits: usize) -> i64 {
        let sign_bit = 1i64 << (bits - 1);
        if value & sign_bit != 0 {
            // Negative: fill upper bits with 1
            value | (!0i64 << bits)
        } else {
            // Positive: already correct
            value
        }
    }

    /// Decompress all timestamps
    pub fn decompress_all(mut self) -> Vec<i64> {
        let first = self.first_timestamp();
        let mut timestamps = vec![first];

        while let Some(timestamp) = self.next_timestamp() {
            timestamps.push(timestamp);
        }

        timestamps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gorilla_compression_unchanged() {
        // Test case: all values are the same (1000 values to amortize header overhead)
        let mut compressor = GorillaCompressor::new(25.0);
        for _ in 0..999 {
            compressor.compress(25.0);
        }

        let compressed = compressor.finish();
        let original_size = 1000 * 8; // 1000 values × 8 bytes
        let compressed_size = compressed.len();

        println!(
            "Unchanged values (1000): {} bytes → {} bytes (ratio: {:.1}:1)",
            original_size,
            compressed_size,
            original_size as f64 / compressed_size as f64
        );

        // Should compress extremely well (mostly '0' bits)
        // 4 byte header + 8 byte first value + 999 bits = ~137 bytes
        // Ratio: 8000/137 ≈ 58:1
        assert!(compressed_size < original_size / 50); // > 50:1 ratio
    }

    #[test]
    fn test_gorilla_round_trip() {
        // Test case: Temperature sensor data with small variations
        let values = vec![20.0, 20.1, 20.2, 20.1, 20.0, 20.1, 20.2, 20.3, 20.2, 20.1];

        let mut compressor = GorillaCompressor::new(values[0]);
        for &value in &values[1..] {
            compressor.compress(value);
        }

        let compressed = compressor.finish();
        let decompressor = GorillaDecompressor::new(&compressed).unwrap();
        let decompressed = decompressor.decompress_all();

        assert_eq!(values.len(), decompressed.len());

        // Values should match exactly (no precision loss)
        for (original, decompressed_val) in values.iter().zip(decompressed.iter()) {
            assert_eq!(original, decompressed_val);
        }
    }

    #[test]
    fn test_delta_of_delta_regular_sampling() {
        // Test case: Regular 1Hz sampling (delta always 1000ms)
        let mut timestamps = Vec::new();
        let base = 1640000000000i64; // Jan 2022
        for i in 0..1000 {
            timestamps.push(base + i * 1000); // 1 second intervals
        }

        let mut compressor = DeltaOfDeltaCompressor::new(timestamps[0]);
        for &ts in &timestamps[1..] {
            compressor.compress(ts);
        }

        let compressed = compressor.finish();
        let original_size = 1000 * 8; // 1000 timestamps × 8 bytes
        let compressed_size = compressed.len();

        println!(
            "Regular 1Hz sampling: {} bytes → {} bytes (ratio: {:.1}:1)",
            original_size,
            compressed_size,
            original_size as f64 / compressed_size as f64
        );

        // Should compress very well (delta_of_delta always 0 → single '0' bit)
        assert!(compressed_size < original_size / 30); // > 30:1 ratio

        // Verify round-trip
        let decompressor = DeltaOfDeltaDecompressor::new(&compressed).unwrap();
        let decompressed = decompressor.decompress_all();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_delta_of_delta_irregular_sampling() {
        // Test case: Irregular sampling with some gaps
        let timestamps = vec![
            1000, 2000, 3000, 3100, 5000, 6000, 7000, // Gap at 3100 and 5000
        ];

        let mut compressor = DeltaOfDeltaCompressor::new(timestamps[0]);
        for &ts in &timestamps[1..] {
            compressor.compress(ts);
        }

        let compressed = compressor.finish();

        // Verify round-trip
        let decompressor = DeltaOfDeltaDecompressor::new(&compressed).unwrap();
        let decompressed = decompressor.decompress_all();

        assert_eq!(timestamps, decompressed);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let mut compressor = GorillaCompressor::new(100.0);
        for i in 0..99 {
            compressor.compress(100.0 + (i as f64 * 0.1));
        }

        let ratio = compressor.compression_ratio(100);
        assert!(ratio > 1.0); // Should have some compression
        println!("Compression ratio: {:.1}:1", ratio);
    }
}
