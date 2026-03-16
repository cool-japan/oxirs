//! Advanced Modbus data decoder with byte-order awareness
//!
//! Provides type-safe decoding of raw Modbus register arrays into strongly-typed
//! [`ModbusTypedValue`] variants. Supports all four byte-order modes commonly
//! found in industrial devices:
//!
//! | Mode                | Word order | Byte order within word |
//! |---------------------|-----------|------------------------|
//! | BigEndian           | Hi-Lo     | Hi-Lo (standard)       |
//! | LittleEndian        | Lo-Hi     | Hi-Lo                  |
//! | BigEndianSwapped    | Hi-Lo     | Lo-Hi (CDAB)           |
//! | LittleEndianSwapped | Lo-Hi     | Lo-Hi (BADC)           |
//!
//! The decoder intentionally avoids `unwrap()` and propagates all errors via
//! [`ModbusError`].

use crate::error::{ModbusError, ModbusResult};
use crate::mapping::ByteOrder;
use std::fmt;

/// Strongly-typed value decoded from a Modbus register array.
///
/// This is a richer type than [`crate::mapping::ModbusValue`]; it preserves
/// the exact Rust primitive and supports lossless f64 conversion for
/// arithmetic / scaling.
#[derive(Debug, Clone, PartialEq)]
pub enum ModbusTypedValue {
    /// Boolean decoded from a coil or discrete input
    Bool(bool),
    /// Signed 16-bit integer (single register)
    I16(i16),
    /// Unsigned 16-bit integer (single register)
    U16(u16),
    /// Signed 32-bit integer (two registers)
    I32(i32),
    /// Unsigned 32-bit integer (two registers)
    U32(u32),
    /// IEEE-754 single-precision float (two registers)
    F32(f32),
    /// IEEE-754 double-precision float (four registers)
    F64(f64),
    /// ASCII string decoded from N register pairs
    Str(String),
}

impl ModbusTypedValue {
    /// Convert to `f64` for scaling and range checks.
    ///
    /// Returns `None` for string values which cannot be meaningfully converted.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ModbusTypedValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            ModbusTypedValue::I16(v) => Some(*v as f64),
            ModbusTypedValue::U16(v) => Some(*v as f64),
            ModbusTypedValue::I32(v) => Some(*v as f64),
            ModbusTypedValue::U32(v) => Some(*v as f64),
            ModbusTypedValue::F32(v) => Some(*v as f64),
            ModbusTypedValue::F64(v) => Some(*v),
            ModbusTypedValue::Str(_) => None,
        }
    }

    /// Apply a linear scale factor and additive offset, returning an `f64`.
    ///
    /// Formula: `physical = raw * scale_factor + offset`
    ///
    /// Returns `None` when the value cannot be converted to `f64` (strings).
    pub fn scale(&self, scale_factor: f64, offset: f64) -> Option<f64> {
        self.as_f64().map(|v| v * scale_factor + offset)
    }

    /// Name of the type as a static string.
    pub fn type_name(&self) -> &'static str {
        match self {
            ModbusTypedValue::Bool(_) => "Bool",
            ModbusTypedValue::I16(_) => "I16",
            ModbusTypedValue::U16(_) => "U16",
            ModbusTypedValue::I32(_) => "I32",
            ModbusTypedValue::U32(_) => "U32",
            ModbusTypedValue::F32(_) => "F32",
            ModbusTypedValue::F64(_) => "F64",
            ModbusTypedValue::Str(_) => "Str",
        }
    }
}

impl fmt::Display for ModbusTypedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModbusTypedValue::Bool(v) => write!(f, "{}", v),
            ModbusTypedValue::I16(v) => write!(f, "{}", v),
            ModbusTypedValue::U16(v) => write!(f, "{}", v),
            ModbusTypedValue::I32(v) => write!(f, "{}", v),
            ModbusTypedValue::U32(v) => write!(f, "{}", v),
            ModbusTypedValue::F32(v) => write!(f, "{}", v),
            ModbusTypedValue::F64(v) => write!(f, "{}", v),
            ModbusTypedValue::Str(v) => write!(f, "{}", v),
        }
    }
}

/// Data type specifier for the decoder — a superset of
/// [`crate::mapping::ModbusDataType`] that includes explicit `Bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderDataType {
    /// Single-bit boolean (from coil / discrete-input value)
    Bool,
    /// Signed 16-bit integer (1 register)
    I16,
    /// Unsigned 16-bit integer (1 register)
    U16,
    /// Signed 32-bit integer (2 registers)
    I32,
    /// Unsigned 32-bit integer (2 registers)
    U32,
    /// IEEE-754 single-precision float (2 registers)
    F32,
    /// IEEE-754 double-precision float (4 registers)
    F64,
    /// ASCII string spanning `n` registers (2 chars/register)
    Str(usize),
}

impl DecoderDataType {
    /// Number of 16-bit registers required for this type.
    pub fn register_count(self) -> usize {
        match self {
            DecoderDataType::Bool | DecoderDataType::I16 | DecoderDataType::U16 => 1,
            DecoderDataType::I32 | DecoderDataType::U32 | DecoderDataType::F32 => 2,
            DecoderDataType::F64 => 4,
            DecoderDataType::Str(n) => (n + 1) / 2,
        }
    }
}

impl From<crate::mapping::ModbusDataType> for DecoderDataType {
    fn from(dt: crate::mapping::ModbusDataType) -> Self {
        use crate::mapping::ModbusDataType as Mdt;
        match dt {
            Mdt::Int16 => DecoderDataType::I16,
            Mdt::Uint16 | Mdt::Bit(_) => DecoderDataType::U16,
            Mdt::Int32 => DecoderDataType::I32,
            Mdt::Uint32 => DecoderDataType::U32,
            Mdt::Float32 => DecoderDataType::F32,
            Mdt::Float64 => DecoderDataType::F64,
            Mdt::String(n) => DecoderDataType::Str(n),
        }
    }
}

/// Stateless decoder for Modbus register arrays.
///
/// All methods are pure functions – no mutable state is held. Create one
/// instance and reuse it freely.
pub struct ModbusDecoder;

impl ModbusDecoder {
    /// Decode a slice of raw `u16` Modbus registers into a typed value.
    ///
    /// # Arguments
    ///
    /// * `regs` - Slice starting at the register of interest. Must contain at
    ///   least `data_type.register_count()` elements.
    /// * `data_type` - How to interpret the raw register data.
    /// * `byte_order` - Word and byte ordering used by the target device.
    ///
    /// # Errors
    ///
    /// Returns [`ModbusError`] when the slice is too short or the decoded
    /// floating-point value is non-finite (NaN / infinity).
    pub fn decode(
        regs: &[u16],
        data_type: DecoderDataType,
        byte_order: ByteOrder,
    ) -> ModbusResult<ModbusTypedValue> {
        let required = data_type.register_count();
        if regs.len() < required {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "decoder needs {} register(s) for {:?}, got {}",
                    required,
                    data_type,
                    regs.len()
                ),
            )));
        }

        match data_type {
            DecoderDataType::Bool => {
                // Treat register as boolean: non-zero = true
                Ok(ModbusTypedValue::Bool(regs[0] != 0))
            }
            DecoderDataType::I16 => Ok(ModbusTypedValue::I16(regs[0] as i16)),
            DecoderDataType::U16 => Ok(ModbusTypedValue::U16(regs[0])),
            DecoderDataType::I32 => {
                let raw = Self::regs_to_u32(regs, byte_order);
                Ok(ModbusTypedValue::I32(raw as i32))
            }
            DecoderDataType::U32 => {
                let raw = Self::regs_to_u32(regs, byte_order);
                Ok(ModbusTypedValue::U32(raw))
            }
            DecoderDataType::F32 => {
                let raw = Self::regs_to_u32(regs, byte_order);
                let v = f32::from_bits(raw);
                if !v.is_finite() {
                    return Err(ModbusError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("decoded F32 is non-finite: {}", v),
                    )));
                }
                Ok(ModbusTypedValue::F32(v))
            }
            DecoderDataType::F64 => {
                let raw = Self::regs_to_u64(regs, byte_order);
                let v = f64::from_bits(raw);
                if !v.is_finite() {
                    return Err(ModbusError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("decoded F64 is non-finite: {}", v),
                    )));
                }
                Ok(ModbusTypedValue::F64(v))
            }
            DecoderDataType::Str(char_count) => Ok(ModbusTypedValue::Str(Self::regs_to_string(
                regs, char_count,
            ))),
        }
    }

    /// Apply scale factor and offset to a decoded value, producing an `f64`.
    ///
    /// Returns `None` for string values.
    pub fn scale(value: &ModbusTypedValue, scale_factor: f64, offset: f64) -> Option<f64> {
        value.scale(scale_factor, offset)
    }

    // ── private helpers ────────────────────────────────────────────────────

    /// Reassemble two registers into a `u32`, respecting byte order.
    ///
    /// Byte order semantics (Modbus convention uses 16-bit "words"):
    ///
    /// ```text
    /// BigEndian           (AB CD): regs[0]=AB_CD, regs[1]=EF_GH → 0xABCDEFGH
    /// LittleEndian        (EF GH + AB CD): regs[0]=EF_GH, regs[1]=AB_CD → 0xABCDEFGH
    /// BigEndianSwapped    (BA DC): regs[0]=BA_DC, regs[1]=FE_HG → 0xABCDEFGH
    /// LittleEndianSwapped (FE HG + BA DC): ...
    /// ```
    fn regs_to_u32(regs: &[u16], order: ByteOrder) -> u32 {
        match order {
            ByteOrder::BigEndian => {
                // Standard Modbus: high word first, natural byte order
                ((regs[0] as u32) << 16) | (regs[1] as u32)
            }
            ByteOrder::LittleEndian => {
                // Low word first
                ((regs[1] as u32) << 16) | (regs[0] as u32)
            }
            ByteOrder::BigEndianSwapped => {
                // High word first, bytes within each word swapped (CDAB)
                let hi = ((regs[0] & 0xFF) << 8) | (regs[0] >> 8);
                let lo = ((regs[1] & 0xFF) << 8) | (regs[1] >> 8);
                ((hi as u32) << 16) | (lo as u32)
            }
            ByteOrder::LittleEndianSwapped => {
                // Low word first, bytes within each word swapped (BADC)
                let hi = ((regs[1] & 0xFF) << 8) | (regs[1] >> 8);
                let lo = ((regs[0] & 0xFF) << 8) | (regs[0] >> 8);
                ((hi as u32) << 16) | (lo as u32)
            }
        }
    }

    /// Reassemble four registers into a `u64`, respecting byte order.
    fn regs_to_u64(regs: &[u16], order: ByteOrder) -> u64 {
        match order {
            ByteOrder::BigEndian => {
                ((regs[0] as u64) << 48)
                    | ((regs[1] as u64) << 32)
                    | ((regs[2] as u64) << 16)
                    | (regs[3] as u64)
            }
            ByteOrder::LittleEndian => {
                ((regs[3] as u64) << 48)
                    | ((regs[2] as u64) << 32)
                    | ((regs[1] as u64) << 16)
                    | (regs[0] as u64)
            }
            ByteOrder::BigEndianSwapped => {
                // Swap bytes in each 16-bit word, keep word order hi→lo
                let mut result: u64 = 0;
                for (i, &reg) in regs[..4].iter().enumerate() {
                    let swapped = ((reg & 0xFF) << 8) | (reg >> 8);
                    result |= (swapped as u64) << ((3 - i) * 16);
                }
                result
            }
            ByteOrder::LittleEndianSwapped => {
                // Swap bytes in each 16-bit word, reverse word order
                let mut result: u64 = 0;
                for (i, &reg) in regs[..4].iter().enumerate() {
                    let swapped = ((reg & 0xFF) << 8) | (reg >> 8);
                    result |= (swapped as u64) << (i * 16);
                }
                result
            }
        }
    }

    /// Decode a Modbus register slice into an ASCII string of `char_count` chars.
    ///
    /// Null bytes and trailing whitespace are stripped.
    fn regs_to_string(regs: &[u16], char_count: usize) -> String {
        let n_regs = (char_count + 1) / 2;
        let mut bytes: Vec<u8> = Vec::with_capacity(char_count);
        for &reg in regs.iter().take(n_regs) {
            bytes.push((reg >> 8) as u8);
            bytes.push((reg & 0xFF) as u8);
        }
        bytes.truncate(char_count);
        // Strip null terminator
        while bytes.last() == Some(&0) {
            bytes.pop();
        }
        String::from_utf8_lossy(&bytes).trim_end().to_string()
    }
}

/// Encoder: convert typed values back into Modbus register arrays.
///
/// Useful for write operations (FC 0x06 / 0x10).
pub struct ModbusEncoder;

impl ModbusEncoder {
    /// Encode a typed value into a `Vec<u16>` suitable for Modbus writes.
    ///
    /// # Errors
    ///
    /// Returns an error when the value and data type are incompatible.
    pub fn encode(
        value: &ModbusTypedValue,
        data_type: DecoderDataType,
        byte_order: ByteOrder,
    ) -> ModbusResult<Vec<u16>> {
        match (value, data_type) {
            (ModbusTypedValue::Bool(v), DecoderDataType::Bool | DecoderDataType::U16) => {
                Ok(vec![if *v { 0xFF00 } else { 0x0000 }])
            }
            (ModbusTypedValue::I16(v), DecoderDataType::I16) => Ok(vec![*v as u16]),
            (ModbusTypedValue::U16(v), DecoderDataType::U16) => Ok(vec![*v]),
            (ModbusTypedValue::I32(v), DecoderDataType::I32) => {
                Ok(Self::u32_to_regs(*v as u32, byte_order))
            }
            (ModbusTypedValue::U32(v), DecoderDataType::U32) => {
                Ok(Self::u32_to_regs(*v, byte_order))
            }
            (ModbusTypedValue::F32(v), DecoderDataType::F32) => {
                Ok(Self::u32_to_regs(v.to_bits(), byte_order))
            }
            (ModbusTypedValue::F64(v), DecoderDataType::F64) => {
                Ok(Self::u64_to_regs(v.to_bits(), byte_order))
            }
            (ModbusTypedValue::Str(s), DecoderDataType::Str(char_count)) => {
                Ok(Self::string_to_regs(s, char_count))
            }
            _ => Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("cannot encode {} as {:?}", value.type_name(), data_type),
            ))),
        }
    }

    fn u32_to_regs(v: u32, order: ByteOrder) -> Vec<u16> {
        let hi = (v >> 16) as u16;
        let lo = (v & 0xFFFF) as u16;
        match order {
            ByteOrder::BigEndian => vec![hi, lo],
            ByteOrder::LittleEndian => vec![lo, hi],
            ByteOrder::BigEndianSwapped => {
                vec![hi.swap_bytes(), lo.swap_bytes()]
            }
            ByteOrder::LittleEndianSwapped => {
                vec![lo.swap_bytes(), hi.swap_bytes()]
            }
        }
    }

    fn u64_to_regs(v: u64, order: ByteOrder) -> Vec<u16> {
        let w3 = (v >> 48) as u16;
        let w2 = ((v >> 32) & 0xFFFF) as u16;
        let w1 = ((v >> 16) & 0xFFFF) as u16;
        let w0 = (v & 0xFFFF) as u16;
        match order {
            ByteOrder::BigEndian => vec![w3, w2, w1, w0],
            ByteOrder::LittleEndian => vec![w0, w1, w2, w3],
            ByteOrder::BigEndianSwapped => {
                vec![
                    w3.swap_bytes(),
                    w2.swap_bytes(),
                    w1.swap_bytes(),
                    w0.swap_bytes(),
                ]
            }
            ByteOrder::LittleEndianSwapped => {
                vec![
                    w0.swap_bytes(),
                    w1.swap_bytes(),
                    w2.swap_bytes(),
                    w3.swap_bytes(),
                ]
            }
        }
    }

    fn string_to_regs(s: &str, char_count: usize) -> Vec<u16> {
        let n_regs = (char_count + 1) / 2;
        let bytes = s.as_bytes();
        let mut regs = Vec::with_capacity(n_regs);
        for i in 0..n_regs {
            let hi = bytes.get(i * 2).copied().unwrap_or(0);
            let lo = bytes.get(i * 2 + 1).copied().unwrap_or(0);
            regs.push(((hi as u16) << 8) | (lo as u16));
        }
        regs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── decode roundtrip tests ───────────────────────────────────────────

    #[test]
    fn test_decode_bool() {
        let v = ModbusDecoder::decode(&[0xFF00], DecoderDataType::Bool, ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::Bool(true));

        let v = ModbusDecoder::decode(&[0x0000], DecoderDataType::Bool, ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::Bool(false));
    }

    #[test]
    fn test_decode_i16() {
        // Positive
        let v = ModbusDecoder::decode(&[255], DecoderDataType::I16, ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::I16(255));

        // Negative (two's complement)
        let v = ModbusDecoder::decode(&[0xFFFF], DecoderDataType::I16, ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::I16(-1));
    }

    #[test]
    fn test_decode_u16() {
        let v = ModbusDecoder::decode(&[0xFFFF], DecoderDataType::U16, ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::U16(65535));
    }

    #[test]
    fn test_decode_i32_big_endian() {
        // 0x0001_0000 = 65536
        let regs = [0x0001, 0x0000];
        let v = ModbusDecoder::decode(&regs, DecoderDataType::I32, ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::I32(65536));
    }

    #[test]
    fn test_decode_i32_little_endian() {
        // Little-endian: regs[0]=lo, regs[1]=hi
        // [0x0000, 0x0001] → hi=0x0001, lo=0x0000 → 0x0001_0000 = 65536
        let regs = [0x0000, 0x0001];
        let v = ModbusDecoder::decode(&regs, DecoderDataType::I32, ByteOrder::LittleEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::I32(65536));
    }

    #[test]
    fn test_decode_u32_all_byte_orders() {
        // Value: 0xDEAD_BEEF
        let be_regs = [0xDEAD, 0xBEEF]; // big-endian
        let le_regs = [0xBEEF, 0xDEAD]; // little-endian (word-swapped)

        let v_be = ModbusDecoder::decode(&be_regs, DecoderDataType::U32, ByteOrder::BigEndian)
            .expect("should succeed");
        let v_le = ModbusDecoder::decode(&le_regs, DecoderDataType::U32, ByteOrder::LittleEndian)
            .expect("should succeed");

        assert_eq!(v_be, ModbusTypedValue::U32(0xDEADBEEF));
        assert_eq!(v_le, ModbusTypedValue::U32(0xDEADBEEF));
    }

    #[test]
    fn test_decode_f32_big_endian() {
        // IEEE 754: 1.0 = 0x3F80_0000
        let regs = [0x3F80, 0x0000];
        let v = ModbusDecoder::decode(&regs, DecoderDataType::F32, ByteOrder::BigEndian)
            .expect("should succeed");
        match v {
            ModbusTypedValue::F32(f) => assert!((f - 1.0f32).abs() < 1e-6),
            _ => panic!("Expected F32"),
        }
    }

    #[test]
    fn test_decode_f32_big_endian_swapped() {
        // 1.0 BE = [0x3F80, 0x0000]
        // After byte-swap in each word: [0x803F, 0x0000]
        let swapped_regs = [0x803F, 0x0000];
        let v = ModbusDecoder::decode(
            &swapped_regs,
            DecoderDataType::F32,
            ByteOrder::BigEndianSwapped,
        )
        .expect("should succeed");
        match v {
            ModbusTypedValue::F32(f) => assert!((f - 1.0f32).abs() < 1e-6),
            _ => panic!("Expected F32"),
        }
    }

    #[test]
    fn test_decode_f64_big_endian() {
        // 1.0_f64 = 0x3FF0_0000_0000_0000
        let regs = [0x3FF0, 0x0000, 0x0000, 0x0000];
        let v = ModbusDecoder::decode(&regs, DecoderDataType::F64, ByteOrder::BigEndian)
            .expect("should succeed");
        match v {
            ModbusTypedValue::F64(f) => assert!((f - 1.0_f64).abs() < 1e-12),
            _ => panic!("Expected F64"),
        }
    }

    #[test]
    fn test_decode_string() {
        // "ABCD" = 0x4142, 0x4344
        let regs = [0x4142, 0x4344];
        let v = ModbusDecoder::decode(&regs, DecoderDataType::Str(4), ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::Str("ABCD".to_string()));
    }

    #[test]
    fn test_decode_string_with_nulls() {
        // "AB\0\0" — trailing nulls should be stripped
        let regs = [0x4142, 0x0000];
        let v = ModbusDecoder::decode(&regs, DecoderDataType::Str(4), ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(v, ModbusTypedValue::Str("AB".to_string()));
    }

    #[test]
    fn test_insufficient_registers_error() {
        let result = ModbusDecoder::decode(&[0x3F80], DecoderDataType::F32, ByteOrder::BigEndian);
        assert!(result.is_err());
    }

    // ── scale tests ──────────────────────────────────────────────────────

    #[test]
    fn test_scale_linear() {
        let v = ModbusTypedValue::I16(625);
        // 625 * 0.1 + (-40.0) = 22.5
        let scaled = ModbusDecoder::scale(&v, 0.1, -40.0).expect("should succeed");
        assert!((scaled - 22.5).abs() < 1e-9);
    }

    #[test]
    fn test_scale_string_returns_none() {
        let v = ModbusTypedValue::Str("hello".to_string());
        assert!(ModbusDecoder::scale(&v, 1.0, 0.0).is_none());
    }

    // ── encoder roundtrip tests ──────────────────────────────────────────

    #[test]
    fn test_encode_i16_roundtrip() {
        let original = ModbusTypedValue::I16(-1024);
        let encoded = ModbusEncoder::encode(&original, DecoderDataType::I16, ByteOrder::BigEndian)
            .expect("should succeed");
        let decoded = ModbusDecoder::decode(&encoded, DecoderDataType::I16, ByteOrder::BigEndian)
            .expect("should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_u32_roundtrip_all_orders() {
        let orders = [
            ByteOrder::BigEndian,
            ByteOrder::LittleEndian,
            ByteOrder::BigEndianSwapped,
            ByteOrder::LittleEndianSwapped,
        ];
        let original = ModbusTypedValue::U32(0xCAFEBABE);
        for order in orders {
            let encoded = ModbusEncoder::encode(&original, DecoderDataType::U32, order)
                .expect("should succeed");
            let decoded = ModbusDecoder::decode(&encoded, DecoderDataType::U32, order)
                .expect("should succeed");
            assert_eq!(
                decoded, original,
                "roundtrip failed for byte order {:?}",
                order
            );
        }
    }

    #[test]
    fn test_encode_f32_roundtrip() {
        let original = ModbusTypedValue::F32(22.5);
        let encoded = ModbusEncoder::encode(&original, DecoderDataType::F32, ByteOrder::BigEndian)
            .expect("should succeed");
        let decoded = ModbusDecoder::decode(&encoded, DecoderDataType::F32, ByteOrder::BigEndian)
            .expect("should succeed");
        match decoded {
            ModbusTypedValue::F32(f) => assert!((f - 22.5).abs() < 1e-5),
            _ => panic!("Expected F32"),
        }
    }

    #[test]
    fn test_encode_f64_roundtrip() {
        let original = ModbusTypedValue::F64(std::f64::consts::PI);
        let orders = [ByteOrder::BigEndian, ByteOrder::LittleEndian];
        for order in orders {
            let encoded = ModbusEncoder::encode(&original, DecoderDataType::F64, order)
                .expect("should succeed");
            let decoded = ModbusDecoder::decode(&encoded, DecoderDataType::F64, order)
                .expect("should succeed");
            match decoded {
                ModbusTypedValue::F64(f) => {
                    assert!(
                        (f - std::f64::consts::PI).abs() < 1e-12,
                        "f64 roundtrip failed for {:?}",
                        order
                    );
                }
                _ => panic!("Expected F64"),
            }
        }
    }

    #[test]
    fn test_encode_string_roundtrip() {
        let original = ModbusTypedValue::Str("Test".to_string());
        let encoded =
            ModbusEncoder::encode(&original, DecoderDataType::Str(4), ByteOrder::BigEndian)
                .expect("should succeed");
        let decoded =
            ModbusDecoder::decode(&encoded, DecoderDataType::Str(4), ByteOrder::BigEndian)
                .expect("should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_type_mismatch_error() {
        let v = ModbusTypedValue::Str("hello".to_string());
        let result = ModbusEncoder::encode(&v, DecoderDataType::F32, ByteOrder::BigEndian);
        assert!(result.is_err());
    }

    // ── as_f64 / Display tests ───────────────────────────────────────────

    #[test]
    fn test_as_f64_all_types() {
        assert_eq!(ModbusTypedValue::Bool(true).as_f64(), Some(1.0));
        assert_eq!(ModbusTypedValue::Bool(false).as_f64(), Some(0.0));
        assert_eq!(ModbusTypedValue::I16(-5).as_f64(), Some(-5.0));
        assert_eq!(ModbusTypedValue::U16(100).as_f64(), Some(100.0));
        assert_eq!(ModbusTypedValue::I32(-1).as_f64(), Some(-1.0));
        assert_eq!(ModbusTypedValue::U32(42).as_f64(), Some(42.0));
        assert!(
            (ModbusTypedValue::F32(std::f32::consts::PI)
                .as_f64()
                .expect("should succeed")
                - std::f64::consts::PI)
                .abs()
                < 1e-5
        );
        assert!(
            (ModbusTypedValue::F64(std::f64::consts::E)
                .as_f64()
                .expect("should succeed")
                - std::f64::consts::E)
                .abs()
                < 1e-12
        );
        assert!(ModbusTypedValue::Str("x".to_string()).as_f64().is_none());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ModbusTypedValue::I16(-42)), "-42");
        assert_eq!(format!("{}", ModbusTypedValue::Bool(true)), "true");
    }

    #[test]
    fn test_type_name() {
        assert_eq!(ModbusTypedValue::Bool(false).type_name(), "Bool");
        assert_eq!(ModbusTypedValue::F32(0.0).type_name(), "F32");
        assert_eq!(ModbusTypedValue::Str(String::new()).type_name(), "Str");
    }
}
