//! Type coercion between Modbus register words and strongly-typed Rust values.
//!
//! All functions are pure (no I/O, no state) and handle every edge case
//! explicitly rather than silently saturating or panicking.
//!
//! # Big-endian word order
//! Multi-register types (U32, I32, F32) use big-endian word order:
//! - `registers[0]` is the **high** (most-significant) word.
//! - `registers[1]` is the **low** (least-significant) word.
//!
//! This matches the Modbus Application Protocol V1.1b3 convention used by most
//! industrial devices.

use crate::opcua::config::DataTypeSpec;
use thiserror::Error;

/// A typed value produced by or consumed by the coercion layer.
#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// IEEE 754 single-precision float.
    F32(f32),
    /// Boolean.
    Bool(bool),
}

/// Errors that can occur during register-value coercion.
#[derive(Debug, Error)]
pub enum CoercionError {
    /// A signed integer value cannot be represented as the target unsigned type.
    #[error("register value {value} out of range for type {target_type}")]
    OutOfRange { value: i64, target_type: String },

    /// Not enough registers were supplied for the requested type.
    #[error("insufficient registers: need {needed}, got {got}")]
    InsufficientRegisters { needed: usize, got: usize },

    /// A floating-point NaN cannot be represented in the target integer/bool type.
    #[error("NaN is not representable in target type {0}")]
    NanNotRepresentable(String),

    /// A floating-point infinity cannot be represented in the target integer/bool type.
    #[error("Infinity is not representable in target type {0}")]
    InfinityNotRepresentable(String),

    /// A DataValue variant does not match the expected DataTypeSpec.
    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },
}

/// Convert Modbus register word(s) to a typed [`DataValue`].
///
/// - Single-register types (`U16`, `I16`, `Bool`) read exactly one word.
/// - Dual-register types (`U32`, `I32`, `F32`) read two words in big-endian
///   word order (word 0 = high, word 1 = low).
///
/// # Errors
/// Returns [`CoercionError::InsufficientRegisters`] when `registers` is shorter
/// than required by `spec`.
pub fn registers_to_value(
    registers: &[u16],
    spec: &DataTypeSpec,
) -> Result<DataValue, CoercionError> {
    let needed = spec.register_count();
    if registers.len() < needed {
        return Err(CoercionError::InsufficientRegisters {
            needed,
            got: registers.len(),
        });
    }

    match spec {
        DataTypeSpec::U16 => Ok(DataValue::U16(registers[0])),

        DataTypeSpec::I16 => Ok(DataValue::I16(registers[0] as i16)),

        DataTypeSpec::Bool => Ok(DataValue::Bool(registers[0] != 0)),

        DataTypeSpec::U32 => {
            let high = registers[0] as u32;
            let low = registers[1] as u32;
            Ok(DataValue::U32((high << 16) | low))
        }

        DataTypeSpec::I32 => {
            let high = registers[0] as u32;
            let low = registers[1] as u32;
            let raw = (high << 16) | low;
            Ok(DataValue::I32(raw as i32))
        }

        DataTypeSpec::F32 => {
            let high = registers[0] as u32;
            let low = registers[1] as u32;
            let bits = (high << 16) | low;
            Ok(DataValue::F32(f32::from_bits(bits)))
        }
    }
}

/// Convert a typed [`DataValue`] back to Modbus register word(s).
///
/// - Single-register types produce a 1-element `Vec<u16>`.
/// - Dual-register types produce a 2-element `Vec<u16>` in big-endian word
///   order (index 0 = high word).
///
/// # Errors
/// - [`CoercionError::OutOfRange`] when a signed value cannot be represented in
///   an unsigned target (e.g. encoding `I16(-1)` as `U16`).
/// - [`CoercionError::NanNotRepresentable`] when the value is a `F32(NaN)` and
///   the target type cannot hold NaN.
/// - [`CoercionError::InfinityNotRepresentable`] when the value is `F32(±inf)`
///   and the target type cannot hold infinity.
/// - [`CoercionError::TypeMismatch`] when the `DataValue` variant does not
///   correspond to `spec` (e.g. passing `U32` value for a `Bool` spec).
pub fn value_to_registers(
    value: &DataValue,
    spec: &DataTypeSpec,
) -> Result<Vec<u16>, CoercionError> {
    match (value, spec) {
        // ── U16 ──────────────────────────────────────────────────────────────
        (DataValue::U16(v), DataTypeSpec::U16) => Ok(vec![*v]),

        (DataValue::I16(v), DataTypeSpec::U16) => {
            if *v < 0 {
                return Err(CoercionError::OutOfRange {
                    value: *v as i64,
                    target_type: "u16".to_owned(),
                });
            }
            Ok(vec![*v as u16])
        }

        (DataValue::Bool(b), DataTypeSpec::U16) => Ok(vec![if *b { 1u16 } else { 0u16 }]),

        // ── I16 ──────────────────────────────────────────────────────────────
        (DataValue::I16(v), DataTypeSpec::I16) => Ok(vec![*v as u16]),

        (DataValue::U16(v), DataTypeSpec::I16) => {
            // Bit-cast is always valid; value ≥ 32768 will read back as negative.
            Ok(vec![*v])
        }

        (DataValue::Bool(b), DataTypeSpec::I16) => Ok(vec![if *b { 1u16 } else { 0u16 }]),

        (DataValue::F32(f), DataTypeSpec::I16) => {
            if f.is_nan() {
                return Err(CoercionError::NanNotRepresentable("i16".to_owned()));
            }
            if f.is_infinite() {
                return Err(CoercionError::InfinityNotRepresentable("i16".to_owned()));
            }
            Err(CoercionError::TypeMismatch {
                expected: "i16-compatible scalar".to_owned(),
                got: "f32".to_owned(),
            })
        }

        // ── Bool ─────────────────────────────────────────────────────────────
        (DataValue::Bool(b), DataTypeSpec::Bool) => Ok(vec![if *b { 1u16 } else { 0u16 }]),

        (DataValue::U16(v), DataTypeSpec::Bool) => Ok(vec![if *v != 0 { 1u16 } else { 0u16 }]),

        (DataValue::I16(v), DataTypeSpec::Bool) => Ok(vec![if *v != 0 { 1u16 } else { 0u16 }]),

        // ── U32 ──────────────────────────────────────────────────────────────
        (DataValue::U32(v), DataTypeSpec::U32) => {
            let high = (v >> 16) as u16;
            let low = (v & 0xFFFF) as u16;
            Ok(vec![high, low])
        }

        (DataValue::I32(v), DataTypeSpec::U32) => {
            if *v < 0 {
                return Err(CoercionError::OutOfRange {
                    value: *v as i64,
                    target_type: "u32".to_owned(),
                });
            }
            let raw = *v as u32;
            Ok(vec![(raw >> 16) as u16, (raw & 0xFFFF) as u16])
        }

        // ── I32 ──────────────────────────────────────────────────────────────
        (DataValue::I32(v), DataTypeSpec::I32) => {
            let raw = *v as u32;
            Ok(vec![(raw >> 16) as u16, (raw & 0xFFFF) as u16])
        }

        (DataValue::U32(v), DataTypeSpec::I32) => {
            // Bit-cast always valid.
            let raw = *v;
            Ok(vec![(raw >> 16) as u16, (raw & 0xFFFF) as u16])
        }

        // ── F32 ──────────────────────────────────────────────────────────────
        (DataValue::F32(f), DataTypeSpec::F32) => {
            let bits = f.to_bits();
            Ok(vec![(bits >> 16) as u16, (bits & 0xFFFF) as u16])
        }

        // ── NaN / Infinity guard for integer targets ──────────────────────────
        (DataValue::F32(f), DataTypeSpec::U16) => {
            if f.is_nan() {
                return Err(CoercionError::NanNotRepresentable("u16".to_owned()));
            }
            if f.is_infinite() {
                return Err(CoercionError::InfinityNotRepresentable("u16".to_owned()));
            }
            Err(CoercionError::TypeMismatch {
                expected: "u16-compatible scalar".to_owned(),
                got: "f32".to_owned(),
            })
        }

        (DataValue::F32(f), DataTypeSpec::U32) => {
            if f.is_nan() {
                return Err(CoercionError::NanNotRepresentable("u32".to_owned()));
            }
            if f.is_infinite() {
                return Err(CoercionError::InfinityNotRepresentable("u32".to_owned()));
            }
            Err(CoercionError::TypeMismatch {
                expected: "u32-compatible scalar".to_owned(),
                got: "f32".to_owned(),
            })
        }

        (DataValue::F32(f), DataTypeSpec::I32) => {
            if f.is_nan() {
                return Err(CoercionError::NanNotRepresentable("i32".to_owned()));
            }
            if f.is_infinite() {
                return Err(CoercionError::InfinityNotRepresentable("i32".to_owned()));
            }
            Err(CoercionError::TypeMismatch {
                expected: "i32-compatible scalar".to_owned(),
                got: "f32".to_owned(),
            })
        }

        (DataValue::F32(f), DataTypeSpec::Bool) => {
            if f.is_nan() {
                return Err(CoercionError::NanNotRepresentable("bool".to_owned()));
            }
            if f.is_infinite() {
                return Err(CoercionError::InfinityNotRepresentable("bool".to_owned()));
            }
            Err(CoercionError::TypeMismatch {
                expected: "bool-compatible scalar".to_owned(),
                got: "f32".to_owned(),
            })
        }

        // ── Catch-all type mismatch ───────────────────────────────────────────
        (val, target) => Err(CoercionError::TypeMismatch {
            expected: format!("{:?}", target),
            got: format!("{:?}", std::mem::discriminant(val)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcua::config::DataTypeSpec;

    #[test]
    fn test_registers_to_value_u16() {
        let v = registers_to_value(&[42u16], &DataTypeSpec::U16).expect("ok");
        assert_eq!(v, DataValue::U16(42));
    }

    #[test]
    fn test_registers_to_value_i16_positive() {
        let v = registers_to_value(&[100u16], &DataTypeSpec::I16).expect("ok");
        assert_eq!(v, DataValue::I16(100));
    }

    #[test]
    fn test_registers_to_value_bool_true() {
        let v = registers_to_value(&[5u16], &DataTypeSpec::Bool).expect("ok");
        assert_eq!(v, DataValue::Bool(true));
    }

    #[test]
    fn test_registers_to_value_bool_false() {
        let v = registers_to_value(&[0u16], &DataTypeSpec::Bool).expect("ok");
        assert_eq!(v, DataValue::Bool(false));
    }

    #[test]
    fn test_insufficient_registers() {
        let err = registers_to_value(&[1u16], &DataTypeSpec::U32).unwrap_err();
        assert!(matches!(
            err,
            CoercionError::InsufficientRegisters { needed: 2, got: 1 }
        ));
    }
}
