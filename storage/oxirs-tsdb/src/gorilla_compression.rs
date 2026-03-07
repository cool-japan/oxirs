//! Gorilla / Delta-of-Delta Compression for Time-Series Data
//!
//! This module implements Facebook's Gorilla compression algorithm (VLDB 2015)
//! for floating-point values, and delta-of-delta encoding for timestamps.
//! Together, these achieve 12:1 to 40:1 compression ratios for typical
//! time-series workloads.
//!
//! # Gorilla XOR Compression (for float values)
//!
//! Consecutive sensor readings tend to have similar floating-point representations.
//! Gorilla exploits this by XORing consecutive values and encoding only the
//! meaningful bits of the XOR result.
//!
//! Encoding scheme:
//! - If XOR == 0: write a single `0` bit (values are identical)
//! - If leading/trailing zeros fit within the previous window: write `10` + meaningful bits
//! - Otherwise: write `11` + 6-bit leading zeros + 6-bit meaningful length + meaningful bits
//!
//! # Delta-of-Delta Compression (for timestamps)
//!
//! Regular time-series have near-constant intervals between readings.
//! Delta-of-delta encoding stores the difference of differences:
//!
//! ```text
//! Timestamps:   1000  1060  1120  1180  1240
//! Deltas:             60    60    60    60
//! DoD:                      0     0     0     (very compressible!)
//! ```
//!
//! Encoding:
//! - DoD == 0: 1 bit (`0`)
//! - DoD in [-63, 64]: 2 bits (`10`) + 7-bit value
//! - DoD in [-255, 256]: 3 bits (`110`) + 9-bit value
//! - DoD in [-2047, 2048]: 4 bits (`1110`) + 12-bit value
//! - Otherwise: 4 bits (`1111`) + 32-bit value

/// A compressed time-series block using Gorilla encoding.
#[derive(Debug, Clone)]
pub struct GorillaBlock {
    /// Compressed bit stream for timestamps (delta-of-delta encoded).
    pub timestamp_bits: BitWriter,
    /// Compressed bit stream for float values (XOR encoded).
    pub value_bits: BitWriter,
    /// Number of data points in this block.
    pub count: usize,
    /// First timestamp in the block (stored uncompressed).
    pub first_timestamp: i64,
    /// First value in the block (stored uncompressed).
    pub first_value: f64,
    /// Statistics.
    pub stats: BlockStats,
}

/// Statistics about a compressed block.
#[derive(Debug, Clone, Default)]
pub struct BlockStats {
    /// Total uncompressed timestamp bytes.
    pub raw_timestamp_bytes: usize,
    /// Total compressed timestamp bits.
    pub compressed_timestamp_bits: usize,
    /// Total uncompressed value bytes.
    pub raw_value_bytes: usize,
    /// Total compressed value bits.
    pub compressed_value_bits: usize,
    /// Number of identical consecutive values.
    pub identical_value_count: usize,
    /// Number of zero delta-of-deltas.
    pub zero_dod_count: usize,
}

impl BlockStats {
    /// Returns the compression ratio for timestamps.
    pub fn timestamp_compression_ratio(&self) -> f64 {
        if self.compressed_timestamp_bits == 0 {
            return 0.0;
        }
        (self.raw_timestamp_bytes * 8) as f64 / self.compressed_timestamp_bits as f64
    }

    /// Returns the compression ratio for values.
    pub fn value_compression_ratio(&self) -> f64 {
        if self.compressed_value_bits == 0 {
            return 0.0;
        }
        (self.raw_value_bytes * 8) as f64 / self.compressed_value_bits as f64
    }

    /// Returns the overall compression ratio.
    pub fn overall_compression_ratio(&self) -> f64 {
        let raw = (self.raw_timestamp_bytes + self.raw_value_bytes) * 8;
        let compressed = self.compressed_timestamp_bits + self.compressed_value_bits;
        if compressed == 0 {
            return 0.0;
        }
        raw as f64 / compressed as f64
    }
}

/// A bit-level writer for building compressed streams.
#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_position: u8, // 0..8, number of bits written in current_byte
}

impl BitWriter {
    /// Creates a new empty bit writer.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bit_position: 0,
        }
    }

    /// Creates a bit writer with pre-allocated capacity (in bytes).
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(bytes),
            current_byte: 0,
            bit_position: 0,
        }
    }

    /// Writes a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_position);
        }
        self.bit_position += 1;
        if self.bit_position == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_position = 0;
        }
    }

    /// Writes `count` bits from the given value (MSB first).
    pub fn write_bits(&mut self, value: u64, count: u8) {
        for i in (0..count).rev() {
            self.write_bit((value >> i) & 1 == 1);
        }
    }

    /// Returns the total number of bits written.
    pub fn bit_count(&self) -> usize {
        self.buffer.len() * 8 + self.bit_position as usize
    }

    /// Returns the number of bytes used (including partial last byte).
    pub fn byte_count(&self) -> usize {
        self.buffer.len() + if self.bit_position > 0 { 1 } else { 0 }
    }

    /// Finalizes the writer and returns the byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_position > 0 {
            self.buffer.push(self.current_byte);
        }
        self.buffer
    }

    /// Returns a reference to the internal buffer (may not include partial byte).
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }
}

/// A bit-level reader for decoding compressed streams.
#[derive(Debug, Clone)]
pub struct BitReader {
    buffer: Vec<u8>,
    byte_position: usize,
    bit_position: u8, // 0..8
}

impl BitReader {
    /// Creates a new bit reader from the given bytes.
    pub fn new(buffer: Vec<u8>) -> Self {
        Self {
            buffer,
            byte_position: 0,
            bit_position: 0,
        }
    }

    /// Reads a single bit. Returns None at end of stream.
    pub fn read_bit(&mut self) -> Option<bool> {
        if self.byte_position >= self.buffer.len() {
            return None;
        }
        let bit = (self.buffer[self.byte_position] >> (7 - self.bit_position)) & 1 == 1;
        self.bit_position += 1;
        if self.bit_position == 8 {
            self.byte_position += 1;
            self.bit_position = 0;
        }
        Some(bit)
    }

    /// Reads `count` bits as a u64 value (MSB first).
    pub fn read_bits(&mut self, count: u8) -> Option<u64> {
        let mut value: u64 = 0;
        for _ in 0..count {
            let bit = self.read_bit()?;
            value = (value << 1) | (bit as u64);
        }
        Some(value)
    }

    /// Returns the current bit position in the stream.
    pub fn position_bits(&self) -> usize {
        self.byte_position * 8 + self.bit_position as usize
    }

    /// Returns true if there are more bits to read.
    pub fn has_more(&self) -> bool {
        self.byte_position < self.buffer.len()
    }
}

/// Gorilla XOR encoder for floating-point values.
pub struct GorillaEncoder {
    writer: BitWriter,
    prev_value: u64,   // previous value as raw bits
    prev_leading: u8,  // previous leading zeros
    prev_trailing: u8, // previous trailing zeros
    count: usize,
    identical_count: usize,
}

impl GorillaEncoder {
    /// Creates a new Gorilla encoder.
    pub fn new() -> Self {
        Self {
            writer: BitWriter::new(),
            prev_value: 0,
            prev_leading: u8::MAX,
            prev_trailing: 0,
            count: 0,
            identical_count: 0,
        }
    }

    /// Encodes the first value (stored uncompressed).
    pub fn encode_first(&mut self, value: f64) {
        let bits = value.to_bits();
        self.writer.write_bits(bits, 64);
        self.prev_value = bits;
        self.count = 1;
    }

    /// Encodes a subsequent value using XOR compression.
    pub fn encode(&mut self, value: f64) {
        let bits = value.to_bits();
        let xor = self.prev_value ^ bits;

        if xor == 0 {
            // Values are identical: write single 0 bit
            self.writer.write_bit(false);
            self.identical_count += 1;
        } else {
            self.writer.write_bit(true);

            let leading = xor.leading_zeros() as u8;
            let trailing = xor.trailing_zeros() as u8;

            if leading >= self.prev_leading && trailing >= self.prev_trailing {
                // Fits within the previous meaningful bit window
                self.writer.write_bit(false); // '10' prefix
                let meaningful_bits = 64 - self.prev_leading - self.prev_trailing;
                let meaningful = xor >> self.prev_trailing;
                self.writer.write_bits(meaningful, meaningful_bits);
            } else {
                // New window needed
                self.writer.write_bit(true); // '11' prefix
                self.writer.write_bits(leading as u64, 6);
                let meaningful_bits = 64 - leading - trailing;
                self.writer.write_bits(meaningful_bits as u64, 6);
                let meaningful = xor >> trailing;
                self.writer.write_bits(meaningful, meaningful_bits);
                self.prev_leading = leading;
                self.prev_trailing = trailing;
            }
        }

        self.prev_value = bits;
        self.count += 1;
    }

    /// Finishes encoding and returns the compressed bits.
    pub fn finish(self) -> (BitWriter, usize, usize) {
        let identical = self.identical_count;
        let count = self.count;
        (self.writer, count, identical)
    }
}

impl Default for GorillaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Gorilla XOR decoder for floating-point values.
pub struct GorillaDecoder {
    reader: BitReader,
    prev_value: u64,
    prev_leading: u8,
    prev_trailing: u8,
    first: bool,
}

impl GorillaDecoder {
    /// Creates a new Gorilla decoder from compressed data.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            reader: BitReader::new(data),
            prev_value: 0,
            prev_leading: 0,
            prev_trailing: 0,
            first: true,
        }
    }

    /// Decodes the next value. Returns None when data is exhausted.
    pub fn decode_next(&mut self) -> Option<f64> {
        if self.first {
            self.first = false;
            let bits = self.reader.read_bits(64)?;
            self.prev_value = bits;
            return Some(f64::from_bits(bits));
        }

        let control = self.reader.read_bit()?;

        if !control {
            // XOR == 0: same value
            return Some(f64::from_bits(self.prev_value));
        }

        let control2 = self.reader.read_bit()?;

        if !control2 {
            // '10': reuse previous window
            let meaningful_bits = 64 - self.prev_leading - self.prev_trailing;
            let meaningful = self.reader.read_bits(meaningful_bits)?;
            let xor = meaningful << self.prev_trailing;
            let value = self.prev_value ^ xor;
            self.prev_value = value;
            Some(f64::from_bits(value))
        } else {
            // '11': new window
            let leading = self.reader.read_bits(6)? as u8;
            let meaningful_bits = self.reader.read_bits(6)? as u8;

            if meaningful_bits == 0 {
                return None; // end marker or corrupt data
            }

            let meaningful = self.reader.read_bits(meaningful_bits)?;
            let trailing = 64 - leading - meaningful_bits;
            let xor = meaningful << trailing;
            let value = self.prev_value ^ xor;
            self.prev_leading = leading;
            self.prev_trailing = trailing;
            self.prev_value = value;
            Some(f64::from_bits(value))
        }
    }

    /// Decodes all remaining values.
    pub fn decode_all(&mut self, expected_count: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(expected_count);
        for _ in 0..expected_count {
            match self.decode_next() {
                Some(v) => values.push(v),
                None => break,
            }
        }
        values
    }
}

/// Delta-of-Delta encoder for timestamps.
pub struct DodEncoder {
    writer: BitWriter,
    prev_timestamp: i64,
    prev_delta: i64,
    count: usize,
    zero_dod_count: usize,
}

impl DodEncoder {
    /// Creates a new delta-of-delta encoder.
    pub fn new() -> Self {
        Self {
            writer: BitWriter::new(),
            prev_timestamp: 0,
            prev_delta: 0,
            count: 0,
            zero_dod_count: 0,
        }
    }

    /// Encodes the first timestamp (stored as raw 64-bit).
    pub fn encode_first(&mut self, timestamp: i64) {
        self.writer.write_bits(timestamp as u64, 64);
        self.prev_timestamp = timestamp;
        self.count = 1;
    }

    /// Encodes the second timestamp (delta stored as full 64-bit).
    ///
    /// The first delta is stored uncompressed because it establishes the baseline
    /// interval for subsequent delta-of-delta encoding.
    pub fn encode_second(&mut self, timestamp: i64) {
        let delta = timestamp - self.prev_timestamp;
        self.writer.write_bits(delta as u64, 64);
        self.prev_delta = delta;
        self.prev_timestamp = timestamp;
        self.count = 2;
    }

    /// Encodes subsequent timestamps using delta-of-delta.
    pub fn encode(&mut self, timestamp: i64) {
        let delta = timestamp - self.prev_timestamp;
        let dod = delta - self.prev_delta;

        if dod == 0 {
            self.writer.write_bit(false); // single 0 bit
            self.zero_dod_count += 1;
        } else if (-63..=64).contains(&dod) {
            self.writer.write_bits(0b10, 2);
            // 7-bit signed value
            self.writer.write_bits((dod as u64) & 0x7F, 7);
        } else if (-255..=256).contains(&dod) {
            self.writer.write_bits(0b110, 3);
            // 9-bit signed value
            self.writer.write_bits((dod as u64) & 0x1FF, 9);
        } else if (-2047..=2048).contains(&dod) {
            self.writer.write_bits(0b1110, 4);
            // 12-bit signed value
            self.writer.write_bits((dod as u64) & 0xFFF, 12);
        } else {
            self.writer.write_bits(0b1111, 4);
            // Full 32-bit value
            self.writer.write_bits((dod as u64) & 0xFFFFFFFF, 32);
        }

        self.prev_delta = delta;
        self.prev_timestamp = timestamp;
        self.count += 1;
    }

    /// Returns the number of encoded timestamps.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Finishes encoding and returns the compressed bits.
    pub fn finish(self) -> (BitWriter, usize, usize) {
        let zeros = self.zero_dod_count;
        let count = self.count;
        (self.writer, count, zeros)
    }
}

impl Default for DodEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Delta-of-Delta decoder for timestamps.
pub struct DodDecoder {
    reader: BitReader,
    prev_timestamp: i64,
    prev_delta: i64,
    is_first: bool,
    is_second: bool,
}

impl DodDecoder {
    /// Creates a new delta-of-delta decoder from compressed data.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            reader: BitReader::new(data),
            prev_timestamp: 0,
            prev_delta: 0,
            is_first: true,
            is_second: false,
        }
    }

    /// Decodes the next timestamp.
    pub fn decode_next(&mut self) -> Option<i64> {
        if self.is_first {
            self.is_first = false;
            self.is_second = true;
            let bits = self.reader.read_bits(64)?;
            self.prev_timestamp = bits as i64;
            return Some(self.prev_timestamp);
        }

        if self.is_second {
            self.is_second = false;
            let bits = self.reader.read_bits(64)?;
            let delta = bits as i64;
            self.prev_delta = delta;
            self.prev_timestamp += delta;
            return Some(self.prev_timestamp);
        }

        let bit0 = self.reader.read_bit()?;
        if !bit0 {
            // DoD == 0
            self.prev_timestamp += self.prev_delta;
            return Some(self.prev_timestamp);
        }

        let bit1 = self.reader.read_bit()?;
        if !bit1 {
            // '10' + 7-bit value
            let raw = self.reader.read_bits(7)?;
            let dod = sign_extend(raw, 7);
            self.prev_delta += dod;
            self.prev_timestamp += self.prev_delta;
            return Some(self.prev_timestamp);
        }

        let bit2 = self.reader.read_bit()?;
        if !bit2 {
            // '110' + 9-bit value
            let raw = self.reader.read_bits(9)?;
            let dod = sign_extend(raw, 9);
            self.prev_delta += dod;
            self.prev_timestamp += self.prev_delta;
            return Some(self.prev_timestamp);
        }

        let bit3 = self.reader.read_bit()?;
        if !bit3 {
            // '1110' + 12-bit value
            let raw = self.reader.read_bits(12)?;
            let dod = sign_extend(raw, 12);
            self.prev_delta += dod;
            self.prev_timestamp += self.prev_delta;
            return Some(self.prev_timestamp);
        }

        // '1111' + 32-bit value
        let raw = self.reader.read_bits(32)?;
        let dod = sign_extend(raw, 32);
        self.prev_delta += dod;
        self.prev_timestamp += self.prev_delta;
        Some(self.prev_timestamp)
    }

    /// Decodes all remaining timestamps.
    pub fn decode_all(&mut self, expected_count: usize) -> Vec<i64> {
        let mut timestamps = Vec::with_capacity(expected_count);
        for _ in 0..expected_count {
            match self.decode_next() {
                Some(t) => timestamps.push(t),
                None => break,
            }
        }
        timestamps
    }
}

/// Sign-extends a value from `bits` width to i64.
fn sign_extend(value: u64, bits: u8) -> i64 {
    if bits == 0 || bits >= 64 {
        return value as i64;
    }
    let sign_bit = 1u64 << (bits - 1);
    if value & sign_bit != 0 {
        // Negative: fill upper bits with 1s
        let mask = !((1u64 << bits) - 1);
        (value | mask) as i64
    } else {
        value as i64
    }
}

/// Compresses a time-series block (timestamps + values).
///
/// Returns a `GorillaBlock` with compressed data and statistics.
pub fn compress_block(timestamps: &[i64], values: &[f64]) -> Option<GorillaBlock> {
    if timestamps.is_empty() || timestamps.len() != values.len() {
        return None;
    }

    let count = timestamps.len();
    let first_timestamp = timestamps[0];
    let first_value = values[0];

    // Encode timestamps
    let mut dod_enc = DodEncoder::new();
    dod_enc.encode_first(timestamps[0]);
    if count > 1 {
        dod_enc.encode_second(timestamps[1]);
    }
    for ts in timestamps.iter().take(count).skip(2) {
        dod_enc.encode(*ts);
    }
    let (ts_writer, _, zero_dods) = dod_enc.finish();

    // Encode values
    let mut gorilla_enc = GorillaEncoder::new();
    gorilla_enc.encode_first(values[0]);
    for val in values.iter().take(count).skip(1) {
        gorilla_enc.encode(*val);
    }
    let (val_writer, _, identical_count) = gorilla_enc.finish();

    let stats = BlockStats {
        raw_timestamp_bytes: count * 8,
        compressed_timestamp_bits: ts_writer.bit_count(),
        raw_value_bytes: count * 8,
        compressed_value_bits: val_writer.bit_count(),
        identical_value_count: identical_count,
        zero_dod_count: zero_dods,
    };

    Some(GorillaBlock {
        timestamp_bits: ts_writer,
        value_bits: val_writer,
        count,
        first_timestamp,
        first_value,
        stats,
    })
}

/// Decompresses a GorillaBlock back to timestamps and values.
pub fn decompress_block(block: &GorillaBlock) -> (Vec<i64>, Vec<f64>) {
    let ts_data = block.timestamp_bits.clone().finish();
    let val_data = block.value_bits.clone().finish();

    let mut dod_dec = DodDecoder::new(ts_data);
    let timestamps = dod_dec.decode_all(block.count);

    let mut gorilla_dec = GorillaDecoder::new(val_data);
    let values = gorilla_dec.decode_all(block.count);

    (timestamps, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BitWriter / BitReader ───────────────────────────────────────────────

    #[test]
    fn test_bit_writer_single_bits() {
        let mut w = BitWriter::new();
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(true);
        assert_eq!(w.bit_count(), 3);
    }

    #[test]
    fn test_bit_writer_full_byte() {
        let mut w = BitWriter::new();
        for _ in 0..8 {
            w.write_bit(true);
        }
        assert_eq!(w.byte_count(), 1);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0xFF]);
    }

    #[test]
    fn test_bit_writer_write_bits() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010, 4);
        assert_eq!(w.bit_count(), 4);
    }

    #[test]
    fn test_bit_reader_single_bits() {
        let mut r = BitReader::new(vec![0b10110000]);
        assert_eq!(r.read_bit(), Some(true));
        assert_eq!(r.read_bit(), Some(false));
        assert_eq!(r.read_bit(), Some(true));
        assert_eq!(r.read_bit(), Some(true));
    }

    #[test]
    fn test_bit_reader_read_bits() {
        let mut r = BitReader::new(vec![0b11001010]);
        let val = r.read_bits(4);
        assert_eq!(val, Some(0b1100));
    }

    #[test]
    fn test_bit_roundtrip() {
        let mut w = BitWriter::new();
        w.write_bits(42, 8);
        w.write_bits(7, 4);
        let bytes = w.finish();

        let mut r = BitReader::new(bytes);
        assert_eq!(r.read_bits(8), Some(42));
        assert_eq!(r.read_bits(4), Some(7));
    }

    #[test]
    fn test_bit_reader_empty() {
        let mut r = BitReader::new(vec![]);
        assert_eq!(r.read_bit(), None);
    }

    #[test]
    fn test_bit_reader_has_more() {
        let mut r = BitReader::new(vec![0xFF]);
        assert!(r.has_more());
        for _ in 0..8 {
            r.read_bit();
        }
        assert!(!r.has_more());
    }

    // ── Gorilla XOR encoding ────────────────────────────────────────────────

    #[test]
    fn test_gorilla_identical_values() {
        let mut enc = GorillaEncoder::new();
        enc.encode_first(25.0);
        enc.encode(25.0);
        enc.encode(25.0);
        let (writer, count, identical) = enc.finish();
        assert_eq!(count, 3);
        assert_eq!(identical, 2);
        // Should be very compact: 64 bits + 2 bits (two zeros)
        assert!(writer.bit_count() < 70);
    }

    #[test]
    fn test_gorilla_similar_values() {
        let mut enc = GorillaEncoder::new();
        enc.encode_first(22.5);
        enc.encode(22.6);
        enc.encode(22.7);
        enc.encode(22.8);
        let (writer, count, _) = enc.finish();
        assert_eq!(count, 4);
        // Should be more compact than 4 * 64 bits
        assert!(writer.bit_count() < 256);
    }

    #[test]
    fn test_gorilla_roundtrip_identical() {
        let values = [42.0; 10];
        let mut enc = GorillaEncoder::new();
        enc.encode_first(values[0]);
        for v in &values[1..] {
            enc.encode(*v);
        }
        let (writer, _, _) = enc.finish();
        let data = writer.finish();

        let mut dec = GorillaDecoder::new(data);
        let decoded = dec.decode_all(10);
        assert_eq!(decoded.len(), 10);
        for (orig, dec_val) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec_val).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_gorilla_roundtrip_varying() {
        let values = [22.5, 22.6, 23.1, 22.9, 22.5, 23.0, 22.8];
        let mut enc = GorillaEncoder::new();
        enc.encode_first(values[0]);
        for v in &values[1..] {
            enc.encode(*v);
        }
        let (writer, _, _) = enc.finish();
        let data = writer.finish();

        let mut dec = GorillaDecoder::new(data);
        let decoded = dec.decode_all(values.len());
        assert_eq!(decoded.len(), values.len());
        for (orig, dec_val) in values.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec_val).abs() < f64::EPSILON,
                "Mismatch: {orig} vs {dec_val}"
            );
        }
    }

    #[test]
    fn test_gorilla_single_value() {
        let mut enc = GorillaEncoder::new();
        enc.encode_first(99.9);
        let (writer, count, _) = enc.finish();
        assert_eq!(count, 1);

        let data = writer.finish();
        let mut dec = GorillaDecoder::new(data);
        let decoded = dec.decode_all(1);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 99.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gorilla_large_jumps() {
        let values = [0.0, 1000.0, -500.0, 1e10, -1e10];
        let mut enc = GorillaEncoder::new();
        enc.encode_first(values[0]);
        for v in &values[1..] {
            enc.encode(*v);
        }
        let (writer, _, _) = enc.finish();
        let data = writer.finish();

        let mut dec = GorillaDecoder::new(data);
        let decoded = dec.decode_all(values.len());
        for (orig, dec_val) in values.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec_val).abs() < f64::EPSILON,
                "Mismatch: {orig} vs {dec_val}"
            );
        }
    }

    // ── Delta-of-Delta encoding ─────────────────────────────────────────────

    #[test]
    fn test_dod_constant_interval() {
        let timestamps: Vec<i64> = (0..10).map(|i| 1000 + i * 60).collect();
        let mut enc = DodEncoder::new();
        enc.encode_first(timestamps[0]);
        enc.encode_second(timestamps[1]);
        for t in &timestamps[2..] {
            enc.encode(*t);
        }
        let (writer, count, zeros) = enc.finish();
        assert_eq!(count, 10);
        assert_eq!(zeros, 8); // All DoDs are 0 after first two
                              // Very compact: mostly single bits
        assert!(writer.bit_count() < 200);
    }

    #[test]
    fn test_dod_roundtrip_constant() {
        let timestamps: Vec<i64> = (0..20).map(|i| 1000000 + i * 60000).collect();
        let mut enc = DodEncoder::new();
        enc.encode_first(timestamps[0]);
        if timestamps.len() > 1 {
            enc.encode_second(timestamps[1]);
        }
        for t in timestamps.iter().skip(2) {
            enc.encode(*t);
        }
        let (writer, _, _) = enc.finish();
        let data = writer.finish();

        let mut dec = DodDecoder::new(data);
        let decoded = dec.decode_all(timestamps.len());
        assert_eq!(decoded, timestamps);
    }

    #[test]
    fn test_dod_roundtrip_varying() {
        let timestamps = vec![1000, 1060, 1130, 1180, 1300, 1360, 1400];
        let mut enc = DodEncoder::new();
        enc.encode_first(timestamps[0]);
        enc.encode_second(timestamps[1]);
        for t in timestamps.iter().skip(2) {
            enc.encode(*t);
        }
        let (writer, _, _) = enc.finish();
        let data = writer.finish();

        let mut dec = DodDecoder::new(data);
        let decoded = dec.decode_all(timestamps.len());
        assert_eq!(decoded, timestamps);
    }

    #[test]
    fn test_dod_single_timestamp() {
        let mut enc = DodEncoder::new();
        enc.encode_first(42);
        let (writer, count, _) = enc.finish();
        assert_eq!(count, 1);

        let data = writer.finish();
        let mut dec = DodDecoder::new(data);
        let decoded = dec.decode_all(1);
        assert_eq!(decoded, vec![42]);
    }

    #[test]
    fn test_dod_two_timestamps() {
        let mut enc = DodEncoder::new();
        enc.encode_first(1000);
        enc.encode_second(1060);
        let (writer, count, _) = enc.finish();
        assert_eq!(count, 2);

        let data = writer.finish();
        let mut dec = DodDecoder::new(data);
        let decoded = dec.decode_all(2);
        assert_eq!(decoded, vec![1000, 1060]);
    }

    // ── Full block compression ──────────────────────────────────────────────

    #[test]
    fn test_compress_decompress_block() {
        let timestamps: Vec<i64> = (0..100).map(|i| 1000000 + i * 60).collect();
        let values: Vec<f64> = (0..100).map(|i| 22.5 + (i as f64) * 0.01).collect();

        let block = compress_block(&timestamps, &values).expect("compress");
        assert_eq!(block.count, 100);
        assert_eq!(block.first_timestamp, 1000000);
        assert!((block.first_value - 22.5).abs() < f64::EPSILON);

        let (dec_ts, dec_vals) = decompress_block(&block);
        assert_eq!(dec_ts, timestamps);
        assert_eq!(dec_vals.len(), values.len());
        for (orig, dec_val) in values.iter().zip(dec_vals.iter()) {
            assert!(
                (orig - dec_val).abs() < f64::EPSILON,
                "Value mismatch: {orig} vs {dec_val}"
            );
        }
    }

    #[test]
    fn test_compress_empty() {
        assert!(compress_block(&[], &[]).is_none());
    }

    #[test]
    fn test_compress_mismatched_lengths() {
        assert!(compress_block(&[1, 2], &[1.0]).is_none());
    }

    #[test]
    fn test_compress_single_point() {
        let block = compress_block(&[42], &[3.125]).expect("compress");
        assert_eq!(block.count, 1);
        let (ts, vals) = decompress_block(&block);
        assert_eq!(ts, vec![42]);
        assert!((vals[0] - 3.125).abs() < f64::EPSILON);
    }

    // ── Block statistics ────────────────────────────────────────────────────

    #[test]
    fn test_block_stats_compression_ratio() {
        let timestamps: Vec<i64> = (0..50).map(|i| 1000 + i * 60).collect();
        let values: Vec<f64> = vec![22.5; 50]; // all identical

        let block = compress_block(&timestamps, &values).expect("compress");
        let ratio = block.stats.overall_compression_ratio();
        assert!(ratio > 1.0, "Expected compression, got ratio {ratio}");
    }

    #[test]
    fn test_block_stats_identical_values() {
        let timestamps: Vec<i64> = (0..20).map(|i| 1000 + i * 60).collect();
        let values = vec![42.0; 20];

        let block = compress_block(&timestamps, &values).expect("compress");
        assert_eq!(block.stats.identical_value_count, 19);
        assert!(block.stats.value_compression_ratio() > 5.0);
    }

    #[test]
    fn test_block_stats_constant_interval() {
        let timestamps: Vec<i64> = (0..20).map(|i| 1000 + i * 60).collect();
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();

        let block = compress_block(&timestamps, &values).expect("compress");
        assert_eq!(block.stats.zero_dod_count, 18);
        assert!(block.stats.timestamp_compression_ratio() > 3.0);
    }

    #[test]
    fn test_block_stats_empty_ratio() {
        let stats = BlockStats::default();
        assert_eq!(stats.timestamp_compression_ratio(), 0.0);
        assert_eq!(stats.value_compression_ratio(), 0.0);
        assert_eq!(stats.overall_compression_ratio(), 0.0);
    }

    // ── Sign extension ──────────────────────────────────────────────────────

    #[test]
    fn test_sign_extend_positive() {
        assert_eq!(sign_extend(5, 7), 5);
    }

    #[test]
    fn test_sign_extend_negative() {
        // -1 in 7 bits = 0b1111111 = 127
        assert_eq!(sign_extend(127, 7), -1);
    }

    #[test]
    fn test_sign_extend_zero() {
        assert_eq!(sign_extend(0, 7), 0);
    }

    #[test]
    fn test_sign_extend_max_positive_7bit() {
        // Max positive in 7 bits = 63
        assert_eq!(sign_extend(63, 7), 63);
    }

    #[test]
    fn test_sign_extend_min_negative_7bit() {
        // Min negative in 7 bits = -64 = 0b1000000 = 64
        assert_eq!(sign_extend(64, 7), -64);
    }

    // ── Bit writer capacity ─────────────────────────────────────────────────

    #[test]
    fn test_bit_writer_with_capacity() {
        let w = BitWriter::with_capacity(100);
        assert_eq!(w.bit_count(), 0);
        assert_eq!(w.byte_count(), 0);
    }

    #[test]
    fn test_bit_writer_as_bytes() {
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        assert_eq!(w.as_bytes(), &[0xFF]);
    }

    // ── Bit reader position ─────────────────────────────────────────────────

    #[test]
    fn test_bit_reader_position() {
        let mut r = BitReader::new(vec![0xFF, 0x00]);
        assert_eq!(r.position_bits(), 0);
        r.read_bits(4);
        assert_eq!(r.position_bits(), 4);
        r.read_bits(8);
        assert_eq!(r.position_bits(), 12);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_gorilla_zero_values() {
        let values = vec![0.0; 10];
        let mut enc = GorillaEncoder::new();
        enc.encode_first(values[0]);
        for v in &values[1..] {
            enc.encode(*v);
        }
        let (writer, _, identical) = enc.finish();
        assert_eq!(identical, 9);
        let data = writer.finish();

        let mut dec = GorillaDecoder::new(data);
        let decoded = dec.decode_all(10);
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_gorilla_negative_values() {
        let values = [-10.5, -10.4, -10.6, -10.5];
        let mut enc = GorillaEncoder::new();
        enc.encode_first(values[0]);
        for v in &values[1..] {
            enc.encode(*v);
        }
        let (writer, _, _) = enc.finish();
        let data = writer.finish();

        let mut dec = GorillaDecoder::new(data);
        let decoded = dec.decode_all(values.len());
        for (orig, dec_val) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec_val).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_dod_large_gap() {
        let timestamps = vec![1000, 1060, 5000, 5060, 5120];
        let mut enc = DodEncoder::new();
        enc.encode_first(timestamps[0]);
        enc.encode_second(timestamps[1]);
        for t in timestamps.iter().skip(2) {
            enc.encode(*t);
        }
        let (writer, _, _) = enc.finish();
        let data = writer.finish();

        let mut dec = DodDecoder::new(data);
        let decoded = dec.decode_all(timestamps.len());
        assert_eq!(decoded, timestamps);
    }

    #[test]
    fn test_compress_realistic_sensor_data() {
        // Simulate a temperature sensor: ~60s intervals, small fluctuations
        let timestamps: Vec<i64> = (0..50).map(|i| 1700000000 + i * 60).collect();
        let values: Vec<f64> = (0..50)
            .map(|i| 22.5 + (i as f64 * 0.1).sin() * 0.5)
            .collect();

        let block = compress_block(&timestamps, &values).expect("compress");
        let ratio = block.stats.overall_compression_ratio();
        assert!(
            ratio > 2.0,
            "Expected good compression for sensor data, got ratio {ratio}"
        );

        let (dec_ts, dec_vals) = decompress_block(&block);
        assert_eq!(dec_ts, timestamps);
        for (orig, dec_val) in values.iter().zip(dec_vals.iter()) {
            assert!((orig - dec_val).abs() < f64::EPSILON);
        }
    }
}
