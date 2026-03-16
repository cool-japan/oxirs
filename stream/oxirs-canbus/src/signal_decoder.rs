//! CAN signal decoding with bit extraction and physical-value scaling.
//!
//! Implements Intel (little-endian) and Motorola (big-endian) bit-layout
//! extraction from CAN frame payloads, sign extension, and the standard
//! `physical = raw * scale + offset` conversion used by DBC/AUTOSAR signal
//! definitions.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Bit ordering convention for a signal's storage in the CAN frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ByteOrder {
    /// Intel/Motorola-style "Intel byte order": LSB is at `start_bit`,
    /// bits run from low to high byte index.
    LittleEndian,
    /// Motorola "big-endian" style: the MSB is at `start_bit`,
    /// counting runs towards lower-significance bytes.
    BigEndian,
}

/// Numeric type of the raw value after bit extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    /// Unsigned integer raw value.
    UnsignedInt,
    /// Signed integer raw value (sign extension applied before scaling).
    SignedInt,
    /// 32-bit IEEE 754 float occupying exactly 32 bits.
    Float32,
    /// 64-bit IEEE 754 float occupying exactly 64 bits.
    Float64,
}

/// Complete description of a signal within a CAN frame.
#[derive(Debug, Clone)]
pub struct SignalDefinition {
    /// Human-readable signal name.
    pub name: String,
    /// First bit position of the signal in the payload.
    /// For `LittleEndian` this is the LSB position; for `BigEndian` this is the MSB position.
    pub start_bit: u8,
    /// Total number of bits occupied by this signal.
    pub bit_length: u8,
    /// Byte ordering.
    pub byte_order: ByteOrder,
    /// Numeric representation of the raw extracted bits.
    pub value_type: ValueType,
    /// Multiplicative factor: `physical = raw * scale + offset`.
    pub scale: f64,
    /// Additive offset: `physical = raw * scale + offset`.
    pub offset: f64,
    /// Minimum valid physical value.
    pub min_value: f64,
    /// Maximum valid physical value.
    pub max_value: f64,
    /// Engineering unit string (e.g. `"rpm"`, `"km/h"`).
    pub unit: String,
}

/// Decoded result for a single signal.
#[derive(Debug, Clone)]
pub struct DecodedSignal {
    /// Signal name.
    pub name: String,
    /// Raw unsigned bit-extracted value (before scale/offset).
    pub raw_value: u64,
    /// Physical value: `raw * scale + offset` (or IEEE 754 reinterpretation
    /// for `Float32`/`Float64`).
    pub physical_value: f64,
    /// Engineering unit string.
    pub unit: String,
    /// `true` when `physical_value` is within `[min_value, max_value]`.
    pub is_valid: bool,
}

/// Decoded result for all signals in one CAN frame.
#[derive(Debug, Clone)]
pub struct DecoderResult {
    /// CAN frame identifier.
    pub frame_id: u32,
    /// Decoded signals.
    pub signals: Vec<DecodedSignal>,
}

/// Registry mapping CAN frame IDs to lists of signal definitions.
pub struct SignalDecoder {
    /// frame_id → ordered list of signal definitions.
    signals: HashMap<u32, Vec<SignalDefinition>>,
}

impl Default for SignalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit extraction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract `bit_length` bits from `data` using Intel (little-endian) layout.
///
/// `start_bit` is the position of the LSB, counting from byte 0 bit 0.
pub fn extract_bits_le(data: &[u8], start_bit: u8, bit_length: u8) -> u64 {
    let mut value: u64 = 0;
    for i in 0..bit_length {
        let bit_pos = start_bit as usize + i as usize;
        let byte_idx = bit_pos / 8;
        let bit_idx = bit_pos % 8;
        if byte_idx < data.len() {
            let bit = ((data[byte_idx] >> bit_idx) & 1) as u64;
            value |= bit << i;
        }
    }
    value
}

/// Extract `bit_length` bits from `data` using Motorola (big-endian) layout.
///
/// `start_bit` is the Motorola MSB position, using the conventional DBC
/// bit numbering: bit `n` = byte `n/8`, bit-in-byte `7 - (n%8)`.
pub fn extract_bits_be(data: &[u8], start_bit: u8, bit_length: u8) -> u64 {
    let mut value: u64 = 0;
    // Convert DBC-style Motorola start_bit (MSB) to linear bit position.
    // DBC Motorola bit numbering: bit_number = byte_index*8 + (7 - bit_in_byte)
    let start_byte = (start_bit / 8) as usize;
    let start_bit_in_byte = start_bit % 8;

    // Collect bits MSB-first.
    let mut bits_remaining = bit_length as i32;
    let mut current_byte = start_byte;
    let mut current_bit = start_bit_in_byte as i32;

    while bits_remaining > 0 {
        if current_byte < data.len() {
            let bit = ((data[current_byte] >> current_bit) & 1) as u64;
            value |= bit << (bits_remaining - 1);
        }
        bits_remaining -= 1;
        current_bit -= 1;
        if current_bit < 0 {
            // Move to the next byte, starting from the most-significant bit.
            current_byte += 1;
            current_bit = 7;
        }
    }
    value
}

/// Sign-extend a `bit_length`-bit unsigned value to a signed 64-bit integer.
pub fn to_signed(raw: u64, bit_length: u8) -> i64 {
    if bit_length == 0 || bit_length >= 64 {
        return raw as i64;
    }
    let sign_bit = 1u64 << (bit_length - 1);
    if raw & sign_bit != 0 {
        // Negative: fill upper bits with ones.
        let mask = u64::MAX << bit_length;
        (raw | mask) as i64
    } else {
        raw as i64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SignalDecoder
// ─────────────────────────────────────────────────────────────────────────────

impl SignalDecoder {
    /// Create an empty decoder.
    pub fn new() -> Self {
        Self {
            signals: HashMap::new(),
        }
    }

    /// Register a signal definition for the given `frame_id`.
    pub fn add_signal(&mut self, frame_id: u32, signal: SignalDefinition) {
        self.signals.entry(frame_id).or_default().push(signal);
    }

    /// Decode all registered signals for `frame_id` from `data`.
    ///
    /// Returns an empty [`DecoderResult`] when no signals are registered for
    /// the given frame ID.
    pub fn decode(&self, frame_id: u32, data: &[u8]) -> DecoderResult {
        let defs = match self.signals.get(&frame_id) {
            Some(d) => d,
            None => {
                return DecoderResult {
                    frame_id,
                    signals: Vec::new(),
                }
            }
        };
        let signals = defs
            .iter()
            .map(|def| Self::decode_signal(data, def))
            .collect();
        DecoderResult { frame_id, signals }
    }

    /// Decode a single signal definition from a byte slice.
    pub fn decode_signal(data: &[u8], def: &SignalDefinition) -> DecodedSignal {
        let raw = match def.byte_order {
            ByteOrder::LittleEndian => extract_bits_le(data, def.start_bit, def.bit_length),
            ByteOrder::BigEndian => extract_bits_be(data, def.start_bit, def.bit_length),
        };

        let physical = match def.value_type {
            ValueType::UnsignedInt => raw as f64 * def.scale + def.offset,
            ValueType::SignedInt => {
                let signed = to_signed(raw, def.bit_length);
                signed as f64 * def.scale + def.offset
            }
            ValueType::Float32 => {
                // Reinterpret 32 bits as IEEE 754 float32.
                let raw32 = raw as u32;
                f32::from_bits(raw32) as f64
            }
            ValueType::Float64 => {
                // Reinterpret 64 bits as IEEE 754 float64.
                f64::from_bits(raw)
            }
        };

        let is_valid = physical >= def.min_value && physical <= def.max_value;

        DecodedSignal {
            name: def.name.clone(),
            raw_value: raw,
            physical_value: physical,
            unit: def.unit.clone(),
            is_valid,
        }
    }

    /// Number of signals registered for `frame_id`.
    pub fn signal_count(&self, frame_id: u32) -> usize {
        self.signals.get(&frame_id).map_or(0, |v| v.len())
    }

    /// All CAN frame IDs for which signals are registered.
    pub fn known_frames(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.signals.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_le_signal(
        name: &str,
        start: u8,
        bits: u8,
        scale: f64,
        offset: f64,
    ) -> SignalDefinition {
        SignalDefinition {
            name: name.to_string(),
            start_bit: start,
            bit_length: bits,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::UnsignedInt,
            scale,
            offset,
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            unit: String::new(),
        }
    }

    // ── add_signal / signal_count ─────────────────────────────────────────────

    #[test]
    fn test_add_signal_and_count() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(0x100, simple_le_signal("speed", 0, 8, 1.0, 0.0));
        assert_eq!(dec.signal_count(0x100), 1);
    }

    #[test]
    fn test_signal_count_unknown_frame() {
        let dec = SignalDecoder::new();
        assert_eq!(dec.signal_count(0xFF), 0);
    }

    #[test]
    fn test_multiple_signals_same_frame() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(0x200, simple_le_signal("a", 0, 8, 1.0, 0.0));
        dec.add_signal(0x200, simple_le_signal("b", 8, 8, 1.0, 0.0));
        assert_eq!(dec.signal_count(0x200), 2);
    }

    // ── known_frames ──────────────────────────────────────────────────────────

    #[test]
    fn test_known_frames_empty() {
        let dec = SignalDecoder::new();
        assert!(dec.known_frames().is_empty());
    }

    #[test]
    fn test_known_frames_sorted() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(0x300, simple_le_signal("x", 0, 8, 1.0, 0.0));
        dec.add_signal(0x100, simple_le_signal("y", 0, 8, 1.0, 0.0));
        dec.add_signal(0x200, simple_le_signal("z", 0, 8, 1.0, 0.0));
        assert_eq!(dec.known_frames(), vec![0x100, 0x200, 0x300]);
    }

    // ── decode – single-byte signal ───────────────────────────────────────────

    #[test]
    fn test_decode_single_byte_signal() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(0x100, simple_le_signal("val", 0, 8, 1.0, 0.0));
        let data = [0xABu8, 0, 0, 0, 0, 0, 0, 0];
        let result = dec.decode(0x100, &data);
        assert_eq!(result.signals.len(), 1);
        assert_eq!(result.signals[0].raw_value, 0xAB);
        assert!((result.signals[0].physical_value - 171.0).abs() < 1e-9);
    }

    #[test]
    fn test_decode_unknown_frame_empty() {
        let dec = SignalDecoder::new();
        let result = dec.decode(0xDEAD, &[0u8; 8]);
        assert!(result.signals.is_empty());
    }

    // ── decode – multi-byte LE signal ──────────────────────────────────────────

    #[test]
    fn test_decode_multi_byte_le() {
        let mut dec = SignalDecoder::new();
        // 16-bit value at bits 0..15
        dec.add_signal(0x10, simple_le_signal("rpm", 0, 16, 0.25, 0.0));
        // raw = 0x0140 = 320 → physical = 320 * 0.25 = 80.0
        let data = [0x40u8, 0x01, 0, 0, 0, 0, 0, 0];
        let result = dec.decode(0x10, &data);
        assert_eq!(result.signals[0].raw_value, 0x0140);
        assert!((result.signals[0].physical_value - 80.0).abs() < 1e-9);
    }

    // ── scale + offset applied ────────────────────────────────────────────────

    #[test]
    fn test_scale_and_offset() {
        let mut dec = SignalDecoder::new();
        let sig = SignalDefinition {
            name: "temp".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::UnsignedInt,
            scale: 0.5,
            offset: -40.0,
            min_value: -40.0,
            max_value: 87.5,
            unit: "degC".into(),
        };
        dec.add_signal(0x1, sig);
        // raw = 200 → physical = 200 * 0.5 + (-40) = 60.0
        let data = [200u8, 0, 0, 0, 0, 0, 0, 0];
        let result = dec.decode(0x1, &data);
        assert!((result.signals[0].physical_value - 60.0).abs() < 1e-9);
    }

    // ── min/max validation ────────────────────────────────────────────────────

    #[test]
    fn test_is_valid_within_range() {
        let sig = SignalDefinition {
            name: "x".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::UnsignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 100.0,
            unit: String::new(),
        };
        let decoded = SignalDecoder::decode_signal(&[50u8], &sig);
        assert!(decoded.is_valid);
    }

    #[test]
    fn test_is_valid_false_when_outside_range() {
        let sig = SignalDefinition {
            name: "x".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::UnsignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 10.0, // 200 > 10
            unit: String::new(),
        };
        let decoded = SignalDecoder::decode_signal(&[200u8], &sig);
        assert!(!decoded.is_valid);
    }

    // ── BigEndian decoding ────────────────────────────────────────────────────

    #[test]
    fn test_big_endian_single_byte() {
        // start_bit = 7 (MSB of byte 0), bit_length = 8
        let sig = SignalDefinition {
            name: "be_val".into(),
            start_bit: 7,
            bit_length: 8,
            byte_order: ByteOrder::BigEndian,
            value_type: ValueType::UnsignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 255.0,
            unit: String::new(),
        };
        // Byte 0 = 0xAB; should extract to 0xAB = 171
        let decoded = SignalDecoder::decode_signal(&[0xABu8, 0, 0, 0, 0, 0, 0, 0], &sig);
        assert_eq!(decoded.raw_value, 0xAB);
    }

    // ── SignedInt sign extension ──────────────────────────────────────────────

    #[test]
    fn test_signed_int_positive() {
        let sig = SignalDefinition {
            name: "s".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::SignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: -128.0,
            max_value: 127.0,
            unit: String::new(),
        };
        // raw = 0x7F = 127 (positive)
        let decoded = SignalDecoder::decode_signal(&[0x7Fu8], &sig);
        assert!((decoded.physical_value - 127.0).abs() < 1e-9);
    }

    #[test]
    fn test_signed_int_negative() {
        let sig = SignalDefinition {
            name: "s".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::SignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: -128.0,
            max_value: 127.0,
            unit: String::new(),
        };
        // raw = 0xFF = 255 unsigned → sign-extended 8-bit = -1
        let decoded = SignalDecoder::decode_signal(&[0xFFu8], &sig);
        assert!((decoded.physical_value - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_to_signed_negative_4bit() {
        // 4-bit value 0b1000 = 8 → signed = -8
        assert_eq!(to_signed(0b1000, 4), -8);
    }

    #[test]
    fn test_to_signed_positive_4bit() {
        // 4-bit value 0b0111 = 7 → signed = 7
        assert_eq!(to_signed(0b0111, 4), 7);
    }

    // ── physical_value formula ────────────────────────────────────────────────

    #[test]
    fn test_physical_value_formula_raw_times_scale_plus_offset() {
        let sig = simple_le_signal("f", 0, 8, 2.5, 10.0);
        // raw = 4 → 4 * 2.5 + 10 = 20.0
        let decoded = SignalDecoder::decode_signal(&[4u8], &sig);
        assert!((decoded.physical_value - 20.0).abs() < 1e-9);
    }

    // ── unit stored ───────────────────────────────────────────────────────────

    #[test]
    fn test_unit_stored_in_decoded_signal() {
        let mut sig = simple_le_signal("spd", 0, 8, 1.0, 0.0);
        sig.unit = "km/h".into();
        let decoded = SignalDecoder::decode_signal(&[100u8], &sig);
        assert_eq!(decoded.unit, "km/h");
    }

    // ── extract_bits_le ───────────────────────────────────────────────────────

    #[test]
    fn test_extract_bits_le_first_byte() {
        let data = [0b1010_1010u8];
        // bits 0..3 of 0xAA = 0b1010 = 10
        assert_eq!(extract_bits_le(&data, 0, 4), 0b1010);
    }

    #[test]
    fn test_extract_bits_le_cross_byte() {
        // two bytes: [0xFF, 0x00]
        // bits 4..11 = upper nibble of byte 0 + lower nibble of byte 1
        // 0xFF bits 4..7 = 0b1111, 0x00 bits 0..3 = 0b0000 → raw = 0b0000_1111 = 15
        let data = [0xFFu8, 0x00];
        assert_eq!(extract_bits_le(&data, 4, 8), 0x0F);
    }

    // ── extract_bits_be ───────────────────────────────────────────────────────

    #[test]
    fn test_extract_bits_be_full_byte() {
        let data = [0xCDu8, 0, 0, 0, 0, 0, 0, 0];
        // start_bit=7 (MSB byte 0), bit_length=8
        assert_eq!(extract_bits_be(&data, 7, 8), 0xCD);
    }

    // ── multiple signals from same frame ──────────────────────────────────────

    #[test]
    fn test_decode_two_signals_same_frame() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(0x55, simple_le_signal("a", 0, 8, 1.0, 0.0));
        dec.add_signal(0x55, simple_le_signal("b", 8, 8, 1.0, 0.0));
        let data = [10u8, 20, 0, 0, 0, 0, 0, 0];
        let result = dec.decode(0x55, &data);
        assert_eq!(result.signals.len(), 2);
        let a = result
            .signals
            .iter()
            .find(|s| s.name == "a")
            .expect("should succeed");
        let b = result
            .signals
            .iter()
            .find(|s| s.name == "b")
            .expect("should succeed");
        assert!((a.physical_value - 10.0).abs() < 1e-9);
        assert!((b.physical_value - 20.0).abs() < 1e-9);
    }

    // ── Float32 decoding ──────────────────────────────────────────────────────

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_float32_decode() {
        let value: f32 = 3.14;
        let bits = value.to_bits();
        let mut data = [0u8; 8];
        data[..4].copy_from_slice(&bits.to_le_bytes());
        let sig = SignalDefinition {
            name: "f32".into(),
            start_bit: 0,
            bit_length: 32,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::Float32,
            scale: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 10.0,
            unit: String::new(),
        };
        let decoded = SignalDecoder::decode_signal(&data, &sig);
        assert!((decoded.physical_value - 3.14_f64).abs() < 1e-5);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_extract_bits_le_all_zeros() {
        let data = [0u8; 8];
        assert_eq!(extract_bits_le(&data, 0, 8), 0);
    }

    #[test]
    fn test_extract_bits_le_all_ones() {
        let data = [0xFFu8; 8];
        assert_eq!(extract_bits_le(&data, 0, 8), 0xFF);
    }

    #[test]
    fn test_extract_bits_le_single_bit_low() {
        let data = [0b0000_0001u8];
        assert_eq!(extract_bits_le(&data, 0, 1), 1);
    }

    #[test]
    fn test_extract_bits_le_single_bit_high() {
        let data = [0b1000_0000u8];
        assert_eq!(extract_bits_le(&data, 7, 1), 1);
    }

    #[test]
    fn test_to_signed_zero() {
        assert_eq!(to_signed(0, 8), 0);
    }

    #[test]
    fn test_to_signed_max_positive_8bit() {
        assert_eq!(to_signed(127, 8), 127);
    }

    #[test]
    fn test_to_signed_min_negative_8bit() {
        assert_eq!(to_signed(128, 8), -128);
    }

    #[test]
    fn test_to_signed_minus_one_16bit() {
        // 0xFFFF as 16-bit signed = -1
        assert_eq!(to_signed(0xFFFF, 16), -1);
    }

    #[test]
    fn test_raw_value_stored_correctly() {
        let sig = simple_le_signal("r", 0, 8, 2.0, 5.0);
        // raw = 10 → physical = 10*2+5 = 25
        let decoded = SignalDecoder::decode_signal(&[10u8], &sig);
        assert_eq!(decoded.raw_value, 10);
    }

    #[test]
    fn test_name_stored_in_decoded_signal() {
        let sig = simple_le_signal("engine_speed", 0, 8, 1.0, 0.0);
        let decoded = SignalDecoder::decode_signal(&[0u8], &sig);
        assert_eq!(decoded.name, "engine_speed");
    }

    #[test]
    fn test_signed_int_zero() {
        let sig = SignalDefinition {
            name: "s".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::SignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: -128.0,
            max_value: 127.0,
            unit: String::new(),
        };
        let decoded = SignalDecoder::decode_signal(&[0u8], &sig);
        assert!((decoded.physical_value - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_decode_empty_frame_id_returns_empty() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(1, simple_le_signal("a", 0, 8, 1.0, 0.0));
        let result = dec.decode(999, &[0u8; 8]);
        assert!(result.signals.is_empty());
    }

    #[test]
    fn test_known_frames_after_adding_multiple() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(5, simple_le_signal("a", 0, 8, 1.0, 0.0));
        dec.add_signal(5, simple_le_signal("b", 8, 8, 1.0, 0.0));
        dec.add_signal(10, simple_le_signal("c", 0, 8, 1.0, 0.0));
        let frames = dec.known_frames();
        assert_eq!(frames, vec![5, 10]);
    }

    #[test]
    fn test_signal_count_for_different_frames_independent() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(1, simple_le_signal("a", 0, 8, 1.0, 0.0));
        dec.add_signal(1, simple_le_signal("b", 8, 8, 1.0, 0.0));
        dec.add_signal(2, simple_le_signal("c", 0, 8, 1.0, 0.0));
        assert_eq!(dec.signal_count(1), 2);
        assert_eq!(dec.signal_count(2), 1);
    }

    #[test]
    fn test_is_valid_at_exact_min() {
        let sig = SignalDefinition {
            name: "v".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::UnsignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: 50.0,
            max_value: 200.0,
            unit: String::new(),
        };
        let decoded = SignalDecoder::decode_signal(&[50u8], &sig);
        assert!(decoded.is_valid);
    }

    #[test]
    fn test_is_valid_at_exact_max() {
        let sig = SignalDefinition {
            name: "v".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::UnsignedInt,
            scale: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 100.0,
            unit: String::new(),
        };
        let decoded = SignalDecoder::decode_signal(&[100u8], &sig);
        assert!(decoded.is_valid);
    }

    #[test]
    fn test_default_signal_decoder() {
        let dec = SignalDecoder::default();
        assert!(dec.known_frames().is_empty());
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_float64_decode() {
        let value: f64 = 2.718_281_828;
        let bits = value.to_bits();
        let mut data = [0u8; 8];
        data.copy_from_slice(&bits.to_le_bytes());
        let sig = SignalDefinition {
            name: "f64".into(),
            start_bit: 0,
            bit_length: 64,
            byte_order: ByteOrder::LittleEndian,
            value_type: ValueType::Float64,
            scale: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 10.0,
            unit: String::new(),
        };
        let decoded = SignalDecoder::decode_signal(&data, &sig);
        assert!((decoded.physical_value - 2.718_281_828).abs() < 1e-9);
    }

    #[test]
    fn test_extract_bits_le_nibble_high() {
        // Upper nibble of byte 0
        let data = [0b1111_0000u8];
        assert_eq!(extract_bits_le(&data, 4, 4), 0b1111);
    }

    #[test]
    fn test_extract_bits_be_4bit_nibble() {
        // start_bit=3 (MSB), bit_length=4 → bits 3..0 of byte 0
        // data[0] = 0b0000_1010: bits 3..0 = 1010
        let data = [0b0000_1010u8];
        assert_eq!(extract_bits_be(&data, 3, 4), 0b1010);
    }

    #[test]
    fn test_to_signed_16bit_half_max() {
        // 0x7FFF as 16-bit = 32767
        assert_eq!(to_signed(0x7FFF, 16), 32767);
    }

    #[test]
    fn test_decode_zero_data() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(1, simple_le_signal("v", 0, 8, 1.0, 0.0));
        let result = dec.decode(1, &[0u8; 8]);
        assert_eq!(result.signals[0].raw_value, 0);
        assert!((result.signals[0].physical_value - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_id_stored_in_result() {
        let mut dec = SignalDecoder::new();
        dec.add_signal(0xABC, simple_le_signal("s", 0, 8, 1.0, 0.0));
        let result = dec.decode(0xABC, &[1u8; 8]);
        assert_eq!(result.frame_id, 0xABC);
    }
}
