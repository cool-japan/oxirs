//! Extended Modbus data type library with full IEEE 754 and BCD support
//!
//! This module provides `ModbusDataType` — a rich enum covering:
//! - IEEE 754 float32 and float64
//! - 16-bit and 32-bit signed/unsigned integers
//! - BCD (Binary-Coded Decimal) encoding
//! - ASCII string types
//!
//! Register-to-value conversion is endianness-aware:
//! `Endianness` controls both word order (big vs. little) and optional
//! byte-swap within each 16-bit word (the CDAB / BADC modes common in
//! industrial devices).

use crate::error::{ModbusError, ModbusResult};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Endianness ────────────────────────────────────────────────────────────────

/// Byte / word ordering for multi-register values.
///
/// The naming follows common Modbus / SCADA convention:
///
/// | Variant            | 32-bit byte layout | Example (0xDEADBEEF)     |
/// |--------------------|-------------------|--------------------------|
/// | `BigEndian`        | AB CD EF GH       | `DE AD BE EF`            |
/// | `LittleEndian`     | EF GH AB CD       | `BE EF DE AD`            |
/// | `BigEndianSwapped` | BA DC FE HG       | `AD DE EF BE`            |
/// | `LittleEndianSwapped` | FE HG BA DC   | `EF BE AD DE`            |
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Endianness {
    /// Standard Modbus: high word first, natural byte order (ABCD)
    #[default]
    BigEndian,
    /// Low word first, natural byte order (CDAB)
    LittleEndian,
    /// High word first, bytes within each word swapped (BADC)
    BigEndianSwapped,
    /// Low word first, bytes within each word swapped (DCBA)
    LittleEndianSwapped,
}

impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Endianness::BigEndian => write!(f, "big_endian"),
            Endianness::LittleEndian => write!(f, "little_endian"),
            Endianness::BigEndianSwapped => write!(f, "big_endian_swapped"),
            Endianness::LittleEndianSwapped => write!(f, "little_endian_swapped"),
        }
    }
}

// ── ModbusDataTypeKind ────────────────────────────────────────────────────────

/// Extended Modbus data type identifier.
///
/// This enum is richer than the basic `crate::mapping::ModbusDataType` —
/// it adds BCD variants and explicit integer widths, making it suitable
/// for device profiles that need precise serialisation descriptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModbusDataTypeKind {
    /// Single-bit boolean (coil / discrete input)
    Bool,
    /// Signed 16-bit integer (1 register)
    Int16,
    /// Unsigned 16-bit integer (1 register)
    Uint16,
    /// Signed 32-bit integer (2 registers)
    Int32,
    /// Unsigned 32-bit integer (2 registers)
    Uint32,
    /// Signed 64-bit integer (4 registers)
    Int64,
    /// Unsigned 64-bit integer (4 registers)
    Uint64,
    /// IEEE 754 single-precision float (2 registers)
    Float32,
    /// IEEE 754 double-precision float (4 registers)
    Float64,
    /// BCD-encoded 16-bit value stored in 1 register (max 9999)
    Bcd16,
    /// BCD-encoded 32-bit value stored in 2 registers (max 99999999)
    Bcd32,
    /// ASCII string occupying `n` characters; uses `(n+1)/2` registers
    #[serde(rename = "string")]
    AsciiString(usize),
}

impl ModbusDataTypeKind {
    /// Number of 16-bit Modbus registers required for this type.
    pub fn register_count(self) -> usize {
        match self {
            ModbusDataTypeKind::Bool
            | ModbusDataTypeKind::Int16
            | ModbusDataTypeKind::Uint16
            | ModbusDataTypeKind::Bcd16 => 1,
            ModbusDataTypeKind::Int32
            | ModbusDataTypeKind::Uint32
            | ModbusDataTypeKind::Float32
            | ModbusDataTypeKind::Bcd32 => 2,
            ModbusDataTypeKind::Int64
            | ModbusDataTypeKind::Uint64
            | ModbusDataTypeKind::Float64 => 4,
            ModbusDataTypeKind::AsciiString(n) => (n + 1) / 2,
        }
    }

    /// XSD datatype IRI for RDF literal annotation.
    pub fn xsd_datatype(self) -> &'static str {
        match self {
            ModbusDataTypeKind::Bool => "http://www.w3.org/2001/XMLSchema#boolean",
            ModbusDataTypeKind::Int16 => "http://www.w3.org/2001/XMLSchema#short",
            ModbusDataTypeKind::Uint16 => "http://www.w3.org/2001/XMLSchema#unsignedShort",
            ModbusDataTypeKind::Int32 => "http://www.w3.org/2001/XMLSchema#int",
            ModbusDataTypeKind::Uint32 => "http://www.w3.org/2001/XMLSchema#unsignedInt",
            ModbusDataTypeKind::Int64 => "http://www.w3.org/2001/XMLSchema#long",
            ModbusDataTypeKind::Uint64 => "http://www.w3.org/2001/XMLSchema#unsignedLong",
            ModbusDataTypeKind::Float32 => "http://www.w3.org/2001/XMLSchema#float",
            ModbusDataTypeKind::Float64 => "http://www.w3.org/2001/XMLSchema#double",
            ModbusDataTypeKind::Bcd16 | ModbusDataTypeKind::Bcd32 => {
                "http://www.w3.org/2001/XMLSchema#unsignedInt"
            }
            ModbusDataTypeKind::AsciiString(_) => "http://www.w3.org/2001/XMLSchema#string",
        }
    }
}

impl fmt::Display for ModbusDataTypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModbusDataTypeKind::Bool => write!(f, "bool"),
            ModbusDataTypeKind::Int16 => write!(f, "int16"),
            ModbusDataTypeKind::Uint16 => write!(f, "uint16"),
            ModbusDataTypeKind::Int32 => write!(f, "int32"),
            ModbusDataTypeKind::Uint32 => write!(f, "uint32"),
            ModbusDataTypeKind::Int64 => write!(f, "int64"),
            ModbusDataTypeKind::Uint64 => write!(f, "uint64"),
            ModbusDataTypeKind::Float32 => write!(f, "float32"),
            ModbusDataTypeKind::Float64 => write!(f, "float64"),
            ModbusDataTypeKind::Bcd16 => write!(f, "bcd16"),
            ModbusDataTypeKind::Bcd32 => write!(f, "bcd32"),
            ModbusDataTypeKind::AsciiString(n) => write!(f, "string{}", n),
        }
    }
}

// ── TypedValue ────────────────────────────────────────────────────────────────

/// A strongly-typed value decoded from Modbus registers.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedValue {
    /// Boolean bit value
    Bool(bool),
    /// Signed 16-bit integer
    Int16(i16),
    /// Unsigned 16-bit integer
    Uint16(u16),
    /// Signed 32-bit integer
    Int32(i32),
    /// Unsigned 32-bit integer
    Uint32(u32),
    /// Signed 64-bit integer
    Int64(i64),
    /// Unsigned 64-bit integer
    Uint64(u64),
    /// IEEE 754 single precision
    Float32(f32),
    /// IEEE 754 double precision
    Float64(f64),
    /// BCD-decoded unsigned 16-bit value (max 9999)
    Bcd16(u16),
    /// BCD-decoded unsigned 32-bit value (max 99999999)
    Bcd32(u32),
    /// ASCII string
    AsciiString(String),
}

impl TypedValue {
    /// Convert to `f64` for scaling arithmetic.
    ///
    /// Returns `None` for string values.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            TypedValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            TypedValue::Int16(v) => Some(*v as f64),
            TypedValue::Uint16(v) => Some(*v as f64),
            TypedValue::Int32(v) => Some(*v as f64),
            TypedValue::Uint32(v) => Some(*v as f64),
            TypedValue::Int64(v) => Some(*v as f64),
            TypedValue::Uint64(v) => Some(*v as f64),
            TypedValue::Float32(v) => Some(*v as f64),
            TypedValue::Float64(v) => Some(*v),
            TypedValue::Bcd16(v) => Some(*v as f64),
            TypedValue::Bcd32(v) => Some(*v as f64),
            TypedValue::AsciiString(_) => None,
        }
    }

    /// Return the type name as a static string.
    pub fn type_name(&self) -> &'static str {
        match self {
            TypedValue::Bool(_) => "bool",
            TypedValue::Int16(_) => "int16",
            TypedValue::Uint16(_) => "uint16",
            TypedValue::Int32(_) => "int32",
            TypedValue::Uint32(_) => "uint32",
            TypedValue::Int64(_) => "int64",
            TypedValue::Uint64(_) => "uint64",
            TypedValue::Float32(_) => "float32",
            TypedValue::Float64(_) => "float64",
            TypedValue::Bcd16(_) => "bcd16",
            TypedValue::Bcd32(_) => "bcd32",
            TypedValue::AsciiString(_) => "ascii_string",
        }
    }
}

impl fmt::Display for TypedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypedValue::Bool(v) => write!(f, "{}", v),
            TypedValue::Int16(v) => write!(f, "{}", v),
            TypedValue::Uint16(v) => write!(f, "{}", v),
            TypedValue::Int32(v) => write!(f, "{}", v),
            TypedValue::Uint32(v) => write!(f, "{}", v),
            TypedValue::Int64(v) => write!(f, "{}", v),
            TypedValue::Uint64(v) => write!(f, "{}", v),
            TypedValue::Float32(v) => write!(f, "{}", v),
            TypedValue::Float64(v) => write!(f, "{}", v),
            TypedValue::Bcd16(v) => write!(f, "{}", v),
            TypedValue::Bcd32(v) => write!(f, "{}", v),
            TypedValue::AsciiString(s) => write!(f, "{}", s),
        }
    }
}

// ── BCD helpers ───────────────────────────────────────────────────────────────

/// Decode a single BCD-encoded nibble.
///
/// Returns `None` when the nibble value exceeds 9 (invalid BCD digit).
#[inline]
fn decode_bcd_nibble(nibble: u8) -> Option<u8> {
    if nibble <= 9 {
        Some(nibble)
    } else {
        None
    }
}

/// Decode a 16-bit BCD register (4 digits, range 0–9999).
///
/// Each nibble (4 bits) represents one decimal digit in the range 0–9.
/// Returns an error when any nibble contains an invalid digit (> 9).
pub fn decode_bcd16(reg: u16) -> ModbusResult<u16> {
    let d3 = decode_bcd_nibble(((reg >> 12) & 0xF) as u8).ok_or_else(|| {
        ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid BCD digit in high nibble: {:#06x}", reg),
        ))
    })?;
    let d2 = decode_bcd_nibble(((reg >> 8) & 0xF) as u8).ok_or_else(|| {
        ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid BCD digit in nibble 2: {:#06x}", reg),
        ))
    })?;
    let d1 = decode_bcd_nibble(((reg >> 4) & 0xF) as u8).ok_or_else(|| {
        ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid BCD digit in nibble 1: {:#06x}", reg),
        ))
    })?;
    let d0 = decode_bcd_nibble((reg & 0xF) as u8).ok_or_else(|| {
        ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid BCD digit in low nibble: {:#06x}", reg),
        ))
    })?;
    Ok(d3 as u16 * 1000 + d2 as u16 * 100 + d1 as u16 * 10 + d0 as u16)
}

/// Encode a decimal value (0–9999) into a 16-bit BCD register.
///
/// Returns an error when the value exceeds 9999.
pub fn encode_bcd16(value: u16) -> ModbusResult<u16> {
    if value > 9999 {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("BCD16 value {} exceeds maximum 9999", value),
        )));
    }
    let d3 = value / 1000;
    let d2 = (value % 1000) / 100;
    let d1 = (value % 100) / 10;
    let d0 = value % 10;
    Ok((d3 << 12) | (d2 << 8) | (d1 << 4) | d0)
}

/// Decode two BCD-encoded 16-bit registers into an 8-digit decimal (range 0–99999999).
///
/// The high register holds the upper 4 BCD digits; the low register holds the
/// lower 4 BCD digits.
pub fn decode_bcd32(hi: u16, lo: u16) -> ModbusResult<u32> {
    let upper = decode_bcd16(hi)? as u32;
    let lower = decode_bcd16(lo)? as u32;
    Ok(upper * 10_000 + lower)
}

/// Encode a decimal value (0–99999999) into two BCD 16-bit registers.
///
/// Returns `(hi_register, lo_register)` where `hi` holds the upper 4 digits.
pub fn encode_bcd32(value: u32) -> ModbusResult<(u16, u16)> {
    if value > 99_999_999 {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("BCD32 value {} exceeds maximum 99999999", value),
        )));
    }
    let upper = (value / 10_000) as u16;
    let lower = (value % 10_000) as u16;
    Ok((encode_bcd16(upper)?, encode_bcd16(lower)?))
}

// ── Register-to-value converter ───────────────────────────────────────────────

/// Assemble two 16-bit registers into a `u32` according to `Endianness`.
pub fn regs_to_u32(regs: &[u16], endianness: Endianness) -> u32 {
    match endianness {
        Endianness::BigEndian => ((regs[0] as u32) << 16) | (regs[1] as u32),
        Endianness::LittleEndian => ((regs[1] as u32) << 16) | (regs[0] as u32),
        Endianness::BigEndianSwapped => {
            let hi = regs[0].swap_bytes();
            let lo = regs[1].swap_bytes();
            ((hi as u32) << 16) | (lo as u32)
        }
        Endianness::LittleEndianSwapped => {
            let hi = regs[1].swap_bytes();
            let lo = regs[0].swap_bytes();
            ((hi as u32) << 16) | (lo as u32)
        }
    }
}

/// Assemble four 16-bit registers into a `u64` according to `Endianness`.
pub fn regs_to_u64(regs: &[u16], endianness: Endianness) -> u64 {
    match endianness {
        Endianness::BigEndian => {
            ((regs[0] as u64) << 48)
                | ((regs[1] as u64) << 32)
                | ((regs[2] as u64) << 16)
                | (regs[3] as u64)
        }
        Endianness::LittleEndian => {
            ((regs[3] as u64) << 48)
                | ((regs[2] as u64) << 32)
                | ((regs[1] as u64) << 16)
                | (regs[0] as u64)
        }
        Endianness::BigEndianSwapped => {
            let mut result: u64 = 0;
            for (i, &reg) in regs[..4].iter().enumerate() {
                let swapped = reg.swap_bytes();
                result |= (swapped as u64) << ((3 - i) * 16);
            }
            result
        }
        Endianness::LittleEndianSwapped => {
            let mut result: u64 = 0;
            for (i, &reg) in regs[..4].iter().enumerate() {
                let swapped = reg.swap_bytes();
                result |= (swapped as u64) << (i * 16);
            }
            result
        }
    }
}

/// Decode a `u32` back into two 16-bit registers using `Endianness`.
pub fn u32_to_regs(value: u32, endianness: Endianness) -> [u16; 2] {
    let hi = (value >> 16) as u16;
    let lo = (value & 0xFFFF) as u16;
    match endianness {
        Endianness::BigEndian => [hi, lo],
        Endianness::LittleEndian => [lo, hi],
        Endianness::BigEndianSwapped => [hi.swap_bytes(), lo.swap_bytes()],
        Endianness::LittleEndianSwapped => [lo.swap_bytes(), hi.swap_bytes()],
    }
}

/// Decode a `u64` back into four 16-bit registers using `Endianness`.
pub fn u64_to_regs(value: u64, endianness: Endianness) -> [u16; 4] {
    let w3 = (value >> 48) as u16;
    let w2 = ((value >> 32) & 0xFFFF) as u16;
    let w1 = ((value >> 16) & 0xFFFF) as u16;
    let w0 = (value & 0xFFFF) as u16;
    match endianness {
        Endianness::BigEndian => [w3, w2, w1, w0],
        Endianness::LittleEndian => [w0, w1, w2, w3],
        Endianness::BigEndianSwapped => [
            w3.swap_bytes(),
            w2.swap_bytes(),
            w1.swap_bytes(),
            w0.swap_bytes(),
        ],
        Endianness::LittleEndianSwapped => [
            w0.swap_bytes(),
            w1.swap_bytes(),
            w2.swap_bytes(),
            w3.swap_bytes(),
        ],
    }
}

/// Decode `char_count` ASCII characters from a register slice.
///
/// Trailing null bytes and whitespace are stripped.
pub fn regs_to_string(regs: &[u16], char_count: usize) -> String {
    let n_regs = (char_count + 1) / 2;
    let mut bytes: Vec<u8> = Vec::with_capacity(char_count);
    for &reg in regs.iter().take(n_regs) {
        bytes.push((reg >> 8) as u8);
        bytes.push((reg & 0xFF) as u8);
    }
    bytes.truncate(char_count);
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8_lossy(&bytes).trim_end().to_string()
}

/// Encode an ASCII string into register words.
pub fn string_to_regs(s: &str, char_count: usize) -> Vec<u16> {
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

// ── Main decode / encode API ─────────────────────────────────────────────────

/// Decode a slice of Modbus registers into a [`TypedValue`].
///
/// The slice must start at the register of interest and contain at least
/// `data_type.register_count()` elements.
///
/// # Errors
///
/// - Slice too short for the requested type
/// - Non-finite float (NaN / infinity) after decoding
/// - Invalid BCD digit
pub fn decode_registers(
    regs: &[u16],
    kind: ModbusDataTypeKind,
    endianness: Endianness,
) -> ModbusResult<TypedValue> {
    let required = kind.register_count();
    if regs.len() < required {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "need {} register(s) for {}, got {}",
                required,
                kind,
                regs.len()
            ),
        )));
    }

    match kind {
        ModbusDataTypeKind::Bool => Ok(TypedValue::Bool(regs[0] != 0)),
        ModbusDataTypeKind::Int16 => Ok(TypedValue::Int16(regs[0] as i16)),
        ModbusDataTypeKind::Uint16 => Ok(TypedValue::Uint16(regs[0])),
        ModbusDataTypeKind::Int32 => {
            let raw = regs_to_u32(regs, endianness);
            Ok(TypedValue::Int32(raw as i32))
        }
        ModbusDataTypeKind::Uint32 => {
            let raw = regs_to_u32(regs, endianness);
            Ok(TypedValue::Uint32(raw))
        }
        ModbusDataTypeKind::Int64 => {
            let raw = regs_to_u64(regs, endianness);
            Ok(TypedValue::Int64(raw as i64))
        }
        ModbusDataTypeKind::Uint64 => {
            let raw = regs_to_u64(regs, endianness);
            Ok(TypedValue::Uint64(raw))
        }
        ModbusDataTypeKind::Float32 => {
            let raw = regs_to_u32(regs, endianness);
            let v = f32::from_bits(raw);
            if !v.is_finite() {
                return Err(ModbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("decoded float32 is non-finite: {}", v),
                )));
            }
            Ok(TypedValue::Float32(v))
        }
        ModbusDataTypeKind::Float64 => {
            let raw = regs_to_u64(regs, endianness);
            let v = f64::from_bits(raw);
            if !v.is_finite() {
                return Err(ModbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("decoded float64 is non-finite: {}", v),
                )));
            }
            Ok(TypedValue::Float64(v))
        }
        ModbusDataTypeKind::Bcd16 => {
            let v = decode_bcd16(regs[0])?;
            Ok(TypedValue::Bcd16(v))
        }
        ModbusDataTypeKind::Bcd32 => {
            let v = decode_bcd32(regs[0], regs[1])?;
            Ok(TypedValue::Bcd32(v))
        }
        ModbusDataTypeKind::AsciiString(char_count) => {
            Ok(TypedValue::AsciiString(regs_to_string(regs, char_count)))
        }
    }
}

/// Encode a [`TypedValue`] into a `Vec<u16>` for Modbus write operations.
///
/// # Errors
///
/// Returns an error when value type and `kind` are incompatible.
pub fn encode_value(
    value: &TypedValue,
    kind: ModbusDataTypeKind,
    endianness: Endianness,
) -> ModbusResult<Vec<u16>> {
    match (value, kind) {
        (TypedValue::Bool(v), ModbusDataTypeKind::Bool | ModbusDataTypeKind::Uint16) => {
            Ok(vec![if *v { 0xFF00 } else { 0x0000 }])
        }
        (TypedValue::Int16(v), ModbusDataTypeKind::Int16) => Ok(vec![*v as u16]),
        (TypedValue::Uint16(v), ModbusDataTypeKind::Uint16) => Ok(vec![*v]),
        (TypedValue::Int32(v), ModbusDataTypeKind::Int32) => {
            Ok(u32_to_regs(*v as u32, endianness).to_vec())
        }
        (TypedValue::Uint32(v), ModbusDataTypeKind::Uint32) => {
            Ok(u32_to_regs(*v, endianness).to_vec())
        }
        (TypedValue::Int64(v), ModbusDataTypeKind::Int64) => {
            Ok(u64_to_regs(*v as u64, endianness).to_vec())
        }
        (TypedValue::Uint64(v), ModbusDataTypeKind::Uint64) => {
            Ok(u64_to_regs(*v, endianness).to_vec())
        }
        (TypedValue::Float32(v), ModbusDataTypeKind::Float32) => {
            Ok(u32_to_regs(v.to_bits(), endianness).to_vec())
        }
        (TypedValue::Float64(v), ModbusDataTypeKind::Float64) => {
            Ok(u64_to_regs(v.to_bits(), endianness).to_vec())
        }
        (TypedValue::Bcd16(v), ModbusDataTypeKind::Bcd16) => Ok(vec![encode_bcd16(*v)?]),
        (TypedValue::Bcd32(v), ModbusDataTypeKind::Bcd32) => {
            let (hi, lo) = encode_bcd32(*v)?;
            Ok(vec![hi, lo])
        }
        (TypedValue::AsciiString(s), ModbusDataTypeKind::AsciiString(n)) => {
            Ok(string_to_regs(s, n))
        }
        _ => Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("cannot encode {} as {}", value.type_name(), kind,),
        ))),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BCD16 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bcd16_decode_zero() {
        assert_eq!(decode_bcd16(0x0000).unwrap(), 0);
    }

    #[test]
    fn test_bcd16_decode_max() {
        // 9999 = 0x9999 in BCD
        assert_eq!(decode_bcd16(0x9999).unwrap(), 9999);
    }

    #[test]
    fn test_bcd16_decode_arbitrary() {
        // 1234 → 0x1234
        assert_eq!(decode_bcd16(0x1234).unwrap(), 1234);
    }

    #[test]
    fn test_bcd16_decode_invalid_digit() {
        // 0xA000 has an invalid BCD nibble (A = 10)
        assert!(decode_bcd16(0xA000).is_err());
    }

    #[test]
    fn test_bcd16_encode_decode_roundtrip() {
        for v in [0u16, 1, 42, 999, 1234, 5678, 9999] {
            let encoded = encode_bcd16(v).unwrap();
            let decoded = decode_bcd16(encoded).unwrap();
            assert_eq!(decoded, v, "roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_bcd16_encode_overflow() {
        assert!(encode_bcd16(10000).is_err());
    }

    // ── BCD32 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bcd32_decode_zero() {
        assert_eq!(decode_bcd32(0x0000, 0x0000).unwrap(), 0);
    }

    #[test]
    fn test_bcd32_decode_max() {
        assert_eq!(decode_bcd32(0x9999, 0x9999).unwrap(), 99_999_999);
    }

    #[test]
    fn test_bcd32_decode_arbitrary() {
        // 12345678
        let (hi, lo) = encode_bcd32(12_345_678).unwrap();
        assert_eq!(decode_bcd32(hi, lo).unwrap(), 12_345_678);
    }

    #[test]
    fn test_bcd32_encode_overflow() {
        assert!(encode_bcd32(100_000_000).is_err());
    }

    #[test]
    fn test_bcd32_roundtrip() {
        for v in [0u32, 1, 12345678, 99999999] {
            let (hi, lo) = encode_bcd32(v).unwrap();
            let decoded = decode_bcd32(hi, lo).unwrap();
            assert_eq!(decoded, v, "BCD32 roundtrip failed for {}", v);
        }
    }

    // ── Float32 endianness ────────────────────────────────────────────────

    #[test]
    fn test_float32_big_endian_decode() {
        // 1.0f32 = 0x3F80_0000
        let regs = [0x3F80u16, 0x0000u16];
        let v =
            decode_registers(&regs, ModbusDataTypeKind::Float32, Endianness::BigEndian).unwrap();
        match v {
            TypedValue::Float32(f) => assert!((f - 1.0f32).abs() < 1e-6),
            _ => panic!("Expected Float32"),
        }
    }

    #[test]
    fn test_float32_little_endian_decode() {
        // 1.0f32 = 0x3F80_0000 → LE regs = [0x0000, 0x3F80]
        let regs = [0x0000u16, 0x3F80u16];
        let v =
            decode_registers(&regs, ModbusDataTypeKind::Float32, Endianness::LittleEndian).unwrap();
        match v {
            TypedValue::Float32(f) => assert!((f - 1.0f32).abs() < 1e-6),
            _ => panic!("Expected Float32"),
        }
    }

    #[test]
    fn test_float32_big_endian_swapped_roundtrip() {
        let original = TypedValue::Float32(std::f32::consts::PI);
        let encoded = encode_value(
            &original,
            ModbusDataTypeKind::Float32,
            Endianness::BigEndianSwapped,
        )
        .unwrap();
        let decoded = decode_registers(
            &encoded,
            ModbusDataTypeKind::Float32,
            Endianness::BigEndianSwapped,
        )
        .unwrap();
        match decoded {
            TypedValue::Float32(f) => assert!((f - std::f32::consts::PI).abs() < 1e-5),
            _ => panic!("Expected Float32"),
        }
    }

    #[test]
    fn test_float32_little_endian_swapped_roundtrip() {
        let original = TypedValue::Float32(-273.15f32);
        let encoded = encode_value(
            &original,
            ModbusDataTypeKind::Float32,
            Endianness::LittleEndianSwapped,
        )
        .unwrap();
        let decoded = decode_registers(
            &encoded,
            ModbusDataTypeKind::Float32,
            Endianness::LittleEndianSwapped,
        )
        .unwrap();
        match decoded {
            TypedValue::Float32(f) => assert!((f - (-273.15f32)).abs() < 1e-3),
            _ => panic!("Expected Float32"),
        }
    }

    // ── Float64 endianness ────────────────────────────────────────────────

    #[test]
    fn test_float64_big_endian_roundtrip() {
        let original = TypedValue::Float64(std::f64::consts::PI);
        let encoded = encode_value(
            &original,
            ModbusDataTypeKind::Float64,
            Endianness::BigEndian,
        )
        .unwrap();
        let decoded =
            decode_registers(&encoded, ModbusDataTypeKind::Float64, Endianness::BigEndian).unwrap();
        match decoded {
            TypedValue::Float64(f) => assert!((f - std::f64::consts::PI).abs() < 1e-12),
            _ => panic!("Expected Float64"),
        }
    }

    #[test]
    fn test_float64_little_endian_roundtrip() {
        let original = TypedValue::Float64(1.23456789e10);
        let encoded = encode_value(
            &original,
            ModbusDataTypeKind::Float64,
            Endianness::LittleEndian,
        )
        .unwrap();
        let decoded = decode_registers(
            &encoded,
            ModbusDataTypeKind::Float64,
            Endianness::LittleEndian,
        )
        .unwrap();
        match decoded {
            TypedValue::Float64(f) => assert!((f - 1.23456789e10).abs() < 1.0),
            _ => panic!("Expected Float64"),
        }
    }

    #[test]
    fn test_float64_nan_rejected() {
        // Craft NaN bits
        let nan_bits = u64::MAX; // all ones → NaN in IEEE 754
        let regs = u64_to_regs(nan_bits, Endianness::BigEndian);
        assert!(
            decode_registers(&regs, ModbusDataTypeKind::Float64, Endianness::BigEndian).is_err()
        );
    }

    // ── Int32 / Uint32 ────────────────────────────────────────────────────

    #[test]
    fn test_int32_all_endianness_roundtrip() {
        let orders = [
            Endianness::BigEndian,
            Endianness::LittleEndian,
            Endianness::BigEndianSwapped,
            Endianness::LittleEndianSwapped,
        ];
        for order in orders {
            let original = TypedValue::Int32(-123456);
            let enc = encode_value(&original, ModbusDataTypeKind::Int32, order).unwrap();
            let dec = decode_registers(&enc, ModbusDataTypeKind::Int32, order).unwrap();
            assert_eq!(dec, original, "Int32 roundtrip failed for {:?}", order);
        }
    }

    #[test]
    fn test_uint32_deadbeef_big_endian() {
        let regs = [0xDEADu16, 0xBEEFu16];
        let v = decode_registers(&regs, ModbusDataTypeKind::Uint32, Endianness::BigEndian).unwrap();
        assert_eq!(v, TypedValue::Uint32(0xDEAD_BEEF));
    }

    // ── Int64 / Uint64 ────────────────────────────────────────────────────

    #[test]
    fn test_int64_roundtrip_big_endian() {
        let original = TypedValue::Int64(i64::MIN);
        let enc =
            encode_value(&original, ModbusDataTypeKind::Int64, Endianness::BigEndian).unwrap();
        let dec = decode_registers(&enc, ModbusDataTypeKind::Int64, Endianness::BigEndian).unwrap();
        assert_eq!(dec, original);
    }

    #[test]
    fn test_uint64_roundtrip_little_endian() {
        let original = TypedValue::Uint64(u64::MAX);
        let enc = encode_value(
            &original,
            ModbusDataTypeKind::Uint64,
            Endianness::LittleEndian,
        )
        .unwrap();
        let dec =
            decode_registers(&enc, ModbusDataTypeKind::Uint64, Endianness::LittleEndian).unwrap();
        assert_eq!(dec, original);
    }

    // ── String ────────────────────────────────────────────────────────────

    #[test]
    fn test_string_encode_decode() {
        let s = "Hello";
        let enc = string_to_regs(s, 6);
        let dec = regs_to_string(&enc, 5);
        assert_eq!(dec, "Hello");
    }

    #[test]
    fn test_string_null_stripped() {
        let regs = [0x4142u16, 0x0000u16];
        let dec = regs_to_string(&regs, 4);
        assert_eq!(dec, "AB");
    }

    // ── TypedValue helpers ────────────────────────────────────────────────

    #[test]
    fn test_typed_value_as_f64() {
        assert_eq!(TypedValue::Bool(true).as_f64(), Some(1.0));
        assert_eq!(TypedValue::Int16(-5).as_f64(), Some(-5.0));
        assert_eq!(TypedValue::Uint32(42).as_f64(), Some(42.0));
        assert!(TypedValue::AsciiString("x".into()).as_f64().is_none());
    }

    #[test]
    fn test_typed_value_display() {
        assert_eq!(format!("{}", TypedValue::Bool(false)), "false");
        assert_eq!(format!("{}", TypedValue::Bcd16(1234)), "1234");
    }

    // ── Insufficient registers ────────────────────────────────────────────

    #[test]
    fn test_insufficient_registers_error() {
        let regs = [0x3F80u16]; // only 1 register, Float32 needs 2
        assert!(
            decode_registers(&regs, ModbusDataTypeKind::Float32, Endianness::BigEndian).is_err()
        );
    }

    // ── register_count and xsd_datatype ──────────────────────────────────

    #[test]
    fn test_register_count() {
        assert_eq!(ModbusDataTypeKind::Bool.register_count(), 1);
        assert_eq!(ModbusDataTypeKind::Int16.register_count(), 1);
        assert_eq!(ModbusDataTypeKind::Uint32.register_count(), 2);
        assert_eq!(ModbusDataTypeKind::Float32.register_count(), 2);
        assert_eq!(ModbusDataTypeKind::Float64.register_count(), 4);
        assert_eq!(ModbusDataTypeKind::Int64.register_count(), 4);
        assert_eq!(ModbusDataTypeKind::Bcd16.register_count(), 1);
        assert_eq!(ModbusDataTypeKind::Bcd32.register_count(), 2);
        assert_eq!(ModbusDataTypeKind::AsciiString(10).register_count(), 5);
    }

    #[test]
    fn test_xsd_datatypes() {
        assert!(ModbusDataTypeKind::Float64
            .xsd_datatype()
            .contains("double"));
        assert!(ModbusDataTypeKind::Bool.xsd_datatype().contains("boolean"));
        assert!(ModbusDataTypeKind::AsciiString(4)
            .xsd_datatype()
            .contains("string"));
    }

    // ── Endianness Display ────────────────────────────────────────────────

    #[test]
    fn test_endianness_display() {
        assert_eq!(Endianness::BigEndian.to_string(), "big_endian");
        assert_eq!(
            Endianness::LittleEndianSwapped.to_string(),
            "little_endian_swapped"
        );
    }

    // ── type_mismatch encode error ────────────────────────────────────────

    #[test]
    fn test_encode_type_mismatch() {
        let v = TypedValue::AsciiString("hello".into());
        assert!(encode_value(&v, ModbusDataTypeKind::Float32, Endianness::BigEndian).is_err());
    }
}
