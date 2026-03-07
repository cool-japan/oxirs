//! # Register Encoder
//!
//! Modbus register data encoding and decoding.
//!
//! Provides stateless encode/decode operations for the data types commonly used in Modbus
//! deployments: IEEE 754 single-precision floats (big/little endian), 32-bit signed integers,
//! 16-bit unsigned integers, BCD-encoded decimals, and linearly-scaled integers.

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Supported Modbus register data encodings.
#[derive(Debug, Clone, PartialEq)]
pub enum DataEncoding {
    /// IEEE 754 single-precision float, big-endian word order (ABCD).
    Float32BE,
    /// IEEE 754 single-precision float, little-endian word order (CDAB).
    Float32LE,
    /// 32-bit signed integer, big-endian word order (ABCD).
    Int32BE,
    /// 32-bit signed integer, little-endian word order (CDAB).
    Int32LE,
    /// 16-bit unsigned integer (single register).
    Uint16,
    /// 16-bit signed integer (single register).
    Int16,
    /// Binary-coded decimal (up to 4 decimal digits per register).
    BCD,
    /// Linearly scaled integer: `physical = raw * scale + offset`.
    ScaledInt {
        /// Multiplier applied to the raw register value.
        scale: f64,
        /// Additive offset applied after scaling.
        offset: f64,
    },
}

/// A pair of Modbus registers that together represent a 32-bit value.
///
/// `high` is the more-significant 16 bits; `low` is the less-significant 16 bits
/// in a big-endian layout.  For little-endian layout the caller should interpret
/// `high` as the low-address word and `low` as the high-address word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterPair {
    /// High (first transmitted / low-address) 16-bit register.
    pub high: u16,
    /// Low (second transmitted / high-address) 16-bit register.
    pub low: u16,
}

impl RegisterPair {
    /// Create a new pair.
    pub fn new(high: u16, low: u16) -> Self {
        Self { high, low }
    }

    /// Combine the two registers into a big-endian 32-bit word.
    fn to_u32_be(self) -> u32 {
        ((self.high as u32) << 16) | (self.low as u32)
    }

    /// Combine the two registers into a little-endian 32-bit word
    /// (high register holds bits 15–0, low register holds bits 31–16).
    fn to_u32_le(self) -> u32 {
        ((self.low as u32) << 16) | (self.high as u32)
    }
}

/// Errors produced by the encoder/decoder.
#[derive(Debug, Clone, PartialEq)]
pub enum EncoderError {
    /// A numeric value was outside the representable range.
    ValueOutOfRange(f64),
    /// A register contained an invalid BCD digit (> 9).
    InvalidBCD,
    /// An empty slice was provided where at least one register was required.
    EmptyInput,
}

impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderError::ValueOutOfRange(v) => write!(f, "value out of range: {}", v),
            EncoderError::InvalidBCD => write!(f, "invalid BCD digit (> 9)"),
            EncoderError::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for EncoderError {}

// ────────────────────────────────────────────────────────────────────────────
// RegisterEncoder
// ────────────────────────────────────────────────────────────────────────────

/// Stateless encoder and decoder for Modbus register values.
///
/// All methods are pure functions with no internal state; the struct exists purely
/// as a namespace.
///
/// # BCD encoding
///
/// Each `u16` register stores four decimal digits in packed BCD, occupying 4 bits each:
/// - nibbles 3–0 of `register[0]` → thousands and hundreds of the overall number
/// - nibbles 3–0 of `register[1]` → tens and units
///
/// So a value like `1234` is encoded in one register as `0x1234u16`.
/// Values that do not fit in the available registers return [`EncoderError::ValueOutOfRange`].
#[derive(Debug, Default, Clone, Copy)]
pub struct RegisterEncoder;

impl RegisterEncoder {
    // ── Float32 Big-Endian (ABCD) ─────────────────────────────────────────

    /// Encode an `f32` as two big-endian Modbus registers.
    ///
    /// The bit pattern of the float is split: high 16 bits → `pair.high`, low 16 bits → `pair.low`.
    pub fn encode_f32_be(value: f32) -> RegisterPair {
        let bits = value.to_bits();
        RegisterPair {
            high: (bits >> 16) as u16,
            low: bits as u16,
        }
    }

    /// Decode two big-endian Modbus registers into an `f32`.
    pub fn decode_f32_be(pair: RegisterPair) -> f32 {
        f32::from_bits(pair.to_u32_be())
    }

    // ── Float32 Little-Endian (CDAB) ──────────────────────────────────────

    /// Encode an `f32` as two little-endian (CDAB / word-swapped) Modbus registers.
    ///
    /// The bit pattern of the float is split: low 16 bits → `pair.high` (first register),
    /// high 16 bits → `pair.low` (second register).
    pub fn encode_f32_le(value: f32) -> RegisterPair {
        let bits = value.to_bits();
        RegisterPair {
            high: bits as u16,        // low word first
            low: (bits >> 16) as u16, // high word second
        }
    }

    /// Decode two little-endian (CDAB) Modbus registers into an `f32`.
    pub fn decode_f32_le(pair: RegisterPair) -> f32 {
        f32::from_bits(pair.to_u32_le())
    }

    // ── Int32 Big-Endian ──────────────────────────────────────────────────

    /// Encode a signed 32-bit integer into two big-endian Modbus registers.
    pub fn encode_i32_be(value: i32) -> RegisterPair {
        let bits = value as u32;
        RegisterPair {
            high: (bits >> 16) as u16,
            low: bits as u16,
        }
    }

    /// Decode two big-endian Modbus registers into a signed 32-bit integer.
    pub fn decode_i32_be(pair: RegisterPair) -> i32 {
        pair.to_u32_be() as i32
    }

    // ── Uint16 ────────────────────────────────────────────────────────────

    /// Encode a `u16` into a single Modbus register (identity).
    pub fn encode_u16(value: u16) -> u16 {
        value
    }

    /// Decode a single Modbus register as a `u16` (identity).
    pub fn decode_u16(reg: u16) -> u16 {
        reg
    }

    // ── BCD ───────────────────────────────────────────────────────────────

    /// BCD-encode `value` into one or more 16-bit Modbus registers.
    ///
    /// Each register holds four decimal digits (packed nibbles).
    /// - Values 0–9999 fit in one register.
    /// - Values 10 000–99 999 999 fit in two registers.
    /// - Values larger than 99 999 999 return [`EncoderError::ValueOutOfRange`].
    pub fn encode_bcd(value: u32) -> Result<Vec<u16>, EncoderError> {
        // Maximum value for 8 BCD digits (two registers): 99_999_999
        if value > 99_999_999 {
            return Err(EncoderError::ValueOutOfRange(value as f64));
        }

        if value <= 9_999 {
            // One register suffices
            let reg = Self::u32_to_bcd_register(value);
            Ok(vec![reg])
        } else {
            // Two registers
            let high_val = value / 10_000;
            let low_val = value % 10_000;
            let high_reg = Self::u32_to_bcd_register(high_val);
            let low_reg = Self::u32_to_bcd_register(low_val);
            Ok(vec![high_reg, low_reg])
        }
    }

    /// Pack a value in 0–9999 into a single BCD register.
    fn u32_to_bcd_register(val: u32) -> u16 {
        let d3 = (val / 1000) as u16;
        let d2 = ((val % 1000) / 100) as u16;
        let d1 = ((val % 100) / 10) as u16;
        let d0 = (val % 10) as u16;
        (d3 << 12) | (d2 << 8) | (d1 << 4) | d0
    }

    /// BCD-decode one or more Modbus registers into a `u32`.
    ///
    /// Each nibble must be in 0–9; otherwise [`EncoderError::InvalidBCD`] is returned.
    /// An empty slice returns [`EncoderError::EmptyInput`].
    pub fn decode_bcd(regs: &[u16]) -> Result<u32, EncoderError> {
        if regs.is_empty() {
            return Err(EncoderError::EmptyInput);
        }

        let mut result: u32 = 0;
        for &reg in regs {
            result *= 10_000;
            result += Self::bcd_register_to_u32(reg)?;
        }
        Ok(result)
    }

    /// Unpack four BCD nibbles from a register, returning an error on an invalid nibble.
    fn bcd_register_to_u32(reg: u16) -> Result<u32, EncoderError> {
        let d3 = ((reg >> 12) & 0xF) as u32;
        let d2 = ((reg >> 8) & 0xF) as u32;
        let d1 = ((reg >> 4) & 0xF) as u32;
        let d0 = (reg & 0xF) as u32;

        if d3 > 9 || d2 > 9 || d1 > 9 || d0 > 9 {
            return Err(EncoderError::InvalidBCD);
        }
        Ok(d3 * 1000 + d2 * 100 + d1 * 10 + d0)
    }

    // ── Scaled Integer ────────────────────────────────────────────────────

    /// Encode a physical value into a raw 16-bit register using linear scaling.
    ///
    /// Formula: `raw = (value - offset) / scale`, clamped to `[0, 65535]`.
    ///
    /// A `scale` of zero is treated as `1.0` to prevent division-by-zero.
    pub fn encode_scaled(value: f64, scale: f64, offset: f64) -> u16 {
        let effective_scale = if scale == 0.0 { 1.0 } else { scale };
        let raw = (value - offset) / effective_scale;
        raw.clamp(0.0, u16::MAX as f64) as u16
    }

    /// Decode a raw 16-bit register into a physical value using linear scaling.
    ///
    /// Formula: `physical = raw * scale + offset`.
    pub fn decode_scaled(raw: u16, scale: f64, offset: f64) -> f64 {
        raw as f64 * scale + offset
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check two f32 values are within a relative tolerance.
    fn f32_close(a: f32, b: f32) -> bool {
        if a == b {
            return true;
        }
        let rel = ((a - b).abs() / b.abs().max(f32::EPSILON)).abs();
        rel < 1e-6_f32
    }

    fn f64_close(a: f64, b: f64) -> bool {
        if a == b {
            return true;
        }
        let rel = ((a - b).abs() / b.abs().max(f64::EPSILON)).abs();
        rel < 1e-9_f64
    }

    // ── encode_f32_be / decode_f32_be ─────────────────────────────────────

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_f32_be_roundtrip_positive() {
        let v = 3.14159_f32;
        let pair = RegisterEncoder::encode_f32_be(v);
        let decoded = RegisterEncoder::decode_f32_be(pair);
        assert!(f32_close(v, decoded));
    }

    #[test]
    fn test_f32_be_roundtrip_negative() {
        let v = -273.15_f32;
        let pair = RegisterEncoder::encode_f32_be(v);
        let decoded = RegisterEncoder::decode_f32_be(pair);
        assert!(f32_close(v, decoded));
    }

    #[test]
    fn test_f32_be_roundtrip_zero() {
        let pair = RegisterEncoder::encode_f32_be(0.0);
        let decoded = RegisterEncoder::decode_f32_be(pair);
        assert_eq!(decoded, 0.0_f32);
    }

    #[test]
    fn test_f32_be_nan_roundtrip() {
        let pair = RegisterEncoder::encode_f32_be(f32::NAN);
        let decoded = RegisterEncoder::decode_f32_be(pair);
        assert!(decoded.is_nan());
    }

    #[test]
    fn test_f32_be_infinity_roundtrip() {
        let pair = RegisterEncoder::encode_f32_be(f32::INFINITY);
        let decoded = RegisterEncoder::decode_f32_be(pair);
        assert!(decoded.is_infinite() && decoded.is_sign_positive());
    }

    #[test]
    fn test_f32_be_neg_infinity_roundtrip() {
        let pair = RegisterEncoder::encode_f32_be(f32::NEG_INFINITY);
        let decoded = RegisterEncoder::decode_f32_be(pair);
        assert!(decoded.is_infinite() && decoded.is_sign_negative());
    }

    #[test]
    fn test_f32_be_register_layout() {
        // 1.0f32 bit pattern = 0x3F800000
        let pair = RegisterEncoder::encode_f32_be(1.0_f32);
        assert_eq!(pair.high, 0x3F80);
        assert_eq!(pair.low, 0x0000);
    }

    // ── encode_f32_le / decode_f32_le ─────────────────────────────────────

    #[test]
    fn test_f32_le_roundtrip_positive() {
        let v = 12345.678_f32;
        let pair = RegisterEncoder::encode_f32_le(v);
        let decoded = RegisterEncoder::decode_f32_le(pair);
        assert!(f32_close(v, decoded));
    }

    #[test]
    fn test_f32_le_roundtrip_negative() {
        let v = -0.001_f32;
        let pair = RegisterEncoder::encode_f32_le(v);
        let decoded = RegisterEncoder::decode_f32_le(pair);
        assert!(f32_close(v, decoded));
    }

    #[test]
    fn test_f32_le_roundtrip_zero() {
        let pair = RegisterEncoder::encode_f32_le(0.0);
        let decoded = RegisterEncoder::decode_f32_le(pair);
        assert_eq!(decoded, 0.0_f32);
    }

    #[test]
    fn test_f32_le_nan_roundtrip() {
        let pair = RegisterEncoder::encode_f32_le(f32::NAN);
        let decoded = RegisterEncoder::decode_f32_le(pair);
        assert!(decoded.is_nan());
    }

    #[test]
    fn test_f32_le_infinity_roundtrip() {
        let pair = RegisterEncoder::encode_f32_le(f32::INFINITY);
        let decoded = RegisterEncoder::decode_f32_le(pair);
        assert!(decoded.is_infinite() && decoded.is_sign_positive());
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_f32_le_register_layout_differs_from_be() {
        // The LE register layout should differ from BE for non-trivial values.
        let v = 3.14_f32;
        let be = RegisterEncoder::encode_f32_be(v);
        let le = RegisterEncoder::encode_f32_le(v);
        // In LE, words are swapped
        assert_eq!(be.high, le.low);
        assert_eq!(be.low, le.high);
    }

    // ── encode_i32_be / decode_i32_be ─────────────────────────────────────

    #[test]
    fn test_i32_be_roundtrip_positive() {
        let v = 123_456_789_i32;
        let pair = RegisterEncoder::encode_i32_be(v);
        let decoded = RegisterEncoder::decode_i32_be(pair);
        assert_eq!(decoded, v);
    }

    #[test]
    fn test_i32_be_roundtrip_negative() {
        let v = -42_i32;
        let pair = RegisterEncoder::encode_i32_be(v);
        let decoded = RegisterEncoder::decode_i32_be(pair);
        assert_eq!(decoded, v);
    }

    #[test]
    fn test_i32_be_roundtrip_zero() {
        let pair = RegisterEncoder::encode_i32_be(0);
        let decoded = RegisterEncoder::decode_i32_be(pair);
        assert_eq!(decoded, 0);
    }

    #[test]
    fn test_i32_be_roundtrip_max() {
        let pair = RegisterEncoder::encode_i32_be(i32::MAX);
        let decoded = RegisterEncoder::decode_i32_be(pair);
        assert_eq!(decoded, i32::MAX);
    }

    #[test]
    fn test_i32_be_roundtrip_min() {
        let pair = RegisterEncoder::encode_i32_be(i32::MIN);
        let decoded = RegisterEncoder::decode_i32_be(pair);
        assert_eq!(decoded, i32::MIN);
    }

    // ── encode_u16 / decode_u16 ───────────────────────────────────────────

    #[test]
    fn test_u16_identity() {
        for v in [0u16, 1, 255, 1000, 32767, 65535] {
            assert_eq!(
                RegisterEncoder::decode_u16(RegisterEncoder::encode_u16(v)),
                v
            );
        }
    }

    #[test]
    fn test_u16_zero() {
        assert_eq!(RegisterEncoder::encode_u16(0), 0);
        assert_eq!(RegisterEncoder::decode_u16(0), 0);
    }

    #[test]
    fn test_u16_max() {
        assert_eq!(RegisterEncoder::encode_u16(u16::MAX), u16::MAX);
    }

    // ── encode_bcd / decode_bcd ───────────────────────────────────────────

    #[test]
    fn test_bcd_roundtrip_zero() {
        let regs = RegisterEncoder::encode_bcd(0).expect("ok");
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 0);
    }

    #[test]
    fn test_bcd_roundtrip_single_digit() {
        let regs = RegisterEncoder::encode_bcd(7).expect("ok");
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 7);
    }

    #[test]
    fn test_bcd_roundtrip_4_digits() {
        let regs = RegisterEncoder::encode_bcd(1234).expect("ok");
        assert_eq!(regs.len(), 1);
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 1234);
    }

    #[test]
    fn test_bcd_roundtrip_max_single_register() {
        let regs = RegisterEncoder::encode_bcd(9999).expect("ok");
        assert_eq!(regs.len(), 1);
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 9999);
    }

    #[test]
    fn test_bcd_roundtrip_two_registers() {
        let regs = RegisterEncoder::encode_bcd(10_000).expect("ok");
        assert_eq!(regs.len(), 2);
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 10_000);
    }

    #[test]
    fn test_bcd_roundtrip_large_value() {
        let regs = RegisterEncoder::encode_bcd(98_765_432).expect("ok");
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 98_765_432);
    }

    #[test]
    fn test_bcd_roundtrip_max_allowed() {
        let regs = RegisterEncoder::encode_bcd(99_999_999).expect("ok");
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 99_999_999);
    }

    #[test]
    fn test_bcd_encode_too_large_returns_error() {
        let err = RegisterEncoder::encode_bcd(100_000_000).unwrap_err();
        assert_eq!(err, EncoderError::ValueOutOfRange(100_000_000.0));
    }

    #[test]
    fn test_bcd_decode_empty_returns_error() {
        let err = RegisterEncoder::decode_bcd(&[]).unwrap_err();
        assert_eq!(err, EncoderError::EmptyInput);
    }

    #[test]
    fn test_bcd_decode_invalid_nibble_returns_error() {
        // 0xABCD has nibble A = 10, which is invalid BCD
        let err = RegisterEncoder::decode_bcd(&[0xABCD_u16]).unwrap_err();
        assert_eq!(err, EncoderError::InvalidBCD);
    }

    #[test]
    fn test_bcd_single_register_layout() {
        // 1234 → 0x1234
        let regs = RegisterEncoder::encode_bcd(1234).expect("ok");
        assert_eq!(regs[0], 0x1234);
    }

    #[test]
    fn test_bcd_roundtrip_5678() {
        let regs = RegisterEncoder::encode_bcd(5678).expect("ok");
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 5678);
    }

    #[test]
    fn test_bcd_roundtrip_mid_two_register() {
        let regs = RegisterEncoder::encode_bcd(50_001).expect("ok");
        assert_eq!(RegisterEncoder::decode_bcd(&regs).expect("ok"), 50_001);
    }

    // ── encode_scaled / decode_scaled ────────────────────────────────────

    #[test]
    fn test_scaled_encode_basic() {
        // scale=0.1, offset=0 → physical 25.5 → raw = 255
        let raw = RegisterEncoder::encode_scaled(25.5, 0.1, 0.0);
        assert_eq!(raw, 255);
    }

    #[test]
    fn test_scaled_decode_basic() {
        // raw=255, scale=0.1, offset=0 → 25.5
        let phys = RegisterEncoder::decode_scaled(255, 0.1, 0.0);
        assert!(f64_close(phys, 25.5));
    }

    #[test]
    fn test_scaled_roundtrip() {
        let value = 100.0_f64;
        let scale = 0.5;
        let offset = 10.0;
        let raw = RegisterEncoder::encode_scaled(value, scale, offset);
        let decoded = RegisterEncoder::decode_scaled(raw, scale, offset);
        // Small rounding error expected from integer truncation
        assert!((decoded - value).abs() < scale);
    }

    #[test]
    fn test_scaled_clamp_negative_to_zero() {
        // Physical value below offset produces negative raw → clamped to 0
        let raw = RegisterEncoder::encode_scaled(-100.0, 1.0, 0.0);
        assert_eq!(raw, 0);
    }

    #[test]
    fn test_scaled_clamp_above_max() {
        // Physical value way above range → clamped to u16::MAX
        let raw = RegisterEncoder::encode_scaled(1_000_000.0, 1.0, 0.0);
        assert_eq!(raw, u16::MAX);
    }

    #[test]
    fn test_scaled_with_offset() {
        // scale=1.0, offset=100.0 → physical 150 → raw = (150 - 100) / 1 = 50
        let raw = RegisterEncoder::encode_scaled(150.0, 1.0, 100.0);
        assert_eq!(raw, 50);
        let phys = RegisterEncoder::decode_scaled(50, 1.0, 100.0);
        assert!(f64_close(phys, 150.0));
    }

    #[test]
    fn test_scaled_zero_scale_treated_as_one() {
        // scale=0 → effective scale=1 to prevent division by zero
        let raw = RegisterEncoder::encode_scaled(42.0, 0.0, 0.0);
        assert_eq!(raw, 42);
    }

    #[test]
    fn test_scaled_decode_zero_raw() {
        let phys = RegisterEncoder::decode_scaled(0, 0.1, 5.0);
        assert!(f64_close(phys, 5.0)); // 0 * 0.1 + 5.0
    }

    // ── DataEncoding enum ─────────────────────────────────────────────────

    #[test]
    fn test_data_encoding_variants_exist() {
        let _ = DataEncoding::Float32BE;
        let _ = DataEncoding::Float32LE;
        let _ = DataEncoding::Int32BE;
        let _ = DataEncoding::Int32LE;
        let _ = DataEncoding::Uint16;
        let _ = DataEncoding::Int16;
        let _ = DataEncoding::BCD;
        let _ = DataEncoding::ScaledInt {
            scale: 0.1,
            offset: 0.0,
        };
    }

    // ── EncoderError Display ──────────────────────────────────────────────

    #[test]
    fn test_error_display_out_of_range() {
        let e = EncoderError::ValueOutOfRange(999.9);
        assert!(e.to_string().contains("out of range"));
    }

    #[test]
    fn test_error_display_invalid_bcd() {
        let e = EncoderError::InvalidBCD;
        assert!(
            e.to_string().contains("BCD")
                || e.to_string().contains("bcd")
                || e.to_string().contains("invalid")
        );
    }

    #[test]
    fn test_error_display_empty_input() {
        let e = EncoderError::EmptyInput;
        assert!(e.to_string().contains("empty"));
    }

    // ── RegisterPair helpers ──────────────────────────────────────────────

    #[test]
    fn test_register_pair_new() {
        let p = RegisterPair::new(0xABCD, 0x1234);
        assert_eq!(p.high, 0xABCD);
        assert_eq!(p.low, 0x1234);
    }

    #[test]
    fn test_f32_be_large_value_roundtrip() {
        let v = 1_000_000.0_f32;
        let pair = RegisterEncoder::encode_f32_be(v);
        let decoded = RegisterEncoder::decode_f32_be(pair);
        assert!(f32_close(v, decoded));
    }

    #[test]
    fn test_f32_le_large_negative_roundtrip() {
        let v = -9876.543_f32;
        let pair = RegisterEncoder::encode_f32_le(v);
        let decoded = RegisterEncoder::decode_f32_le(pair);
        assert!(f32_close(v, decoded));
    }

    #[test]
    fn test_i32_be_various_values() {
        for v in [-1_000_000_i32, -1, 0, 1, 999_999] {
            let pair = RegisterEncoder::encode_i32_be(v);
            assert_eq!(
                RegisterEncoder::decode_i32_be(pair),
                v,
                "failed for v={}",
                v
            );
        }
    }

    #[test]
    fn test_bcd_boundary_9999_and_10000() {
        let r1 = RegisterEncoder::encode_bcd(9999).expect("ok");
        assert_eq!(r1.len(), 1);
        let r2 = RegisterEncoder::encode_bcd(10_000).expect("ok");
        assert_eq!(r2.len(), 2);
    }
}
