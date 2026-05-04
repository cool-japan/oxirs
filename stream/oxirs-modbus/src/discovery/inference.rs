//! Type inference for discovered Modbus registers.
//!
//! [`TypeInferrer`] operates on [`DiscoveredRegister`] values produced by
//! [`DiscoveryDriver`] and assigns a best-guess [`InferredType`] together with
//! a [`ConfidenceLevel`] and a human-readable diagnostic note.
//!
//! Heuristics (applied in priority order):
//!
//! 1. **Float32** — two consecutive same-FC registers whose combined 32-bit
//!    value is a finite IEEE 754 float in the plausible industrial range
//!    `(1e-6, 1e6)`.
//! 2. **Boolean** — raw value is exactly `0` or `1`.
//! 3. **Int16** — treated as `i16::from_ne_bytes`; if that gives a negative
//!    number (value ≥ 0x8000) the type is signed.
//! 4. **ScaledInt** — value in `(1, 10_000)` that produces a plausible
//!    `(0, 1000)` result when multiplied by `0.1` (common industrial scale).
//! 5. **UInt16** — positive value that doesn't match the above.
//!
//! Cross-reference: [`crate::register_validator::RegisterDataType`] covers the
//! same domain but is intended for *validation* after mapping is known; this
//! module is for *discovery* of an unknown device.

use super::driver::DiscoveredRegister;

// ─── Inferred type ────────────────────────────────────────────────────────────

/// Best-guess data type for a discovered register.
#[derive(Debug, Clone, PartialEq)]
pub enum InferredType {
    /// Unsigned 16-bit integer (single register).
    UInt16,
    /// Signed 16-bit integer, two's complement (single register).
    Int16,
    /// IEEE 754 32-bit float spanning two consecutive registers (big-endian).
    Float32,
    /// Industrial scaled integer: `physical = raw * scale`.
    ScaledInt {
        /// Multiplicative scaling factor (e.g. `0.1` for one decimal place).
        scale: f64,
    },
    /// Boolean coil — raw value is 0 or 1.
    Boolean,
    /// Type could not be determined with acceptable confidence.
    Unknown,
}

// ─── Confidence ───────────────────────────────────────────────────────────────

/// Confidence in the inferred type assignment.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ConfidenceLevel {
    /// Fewer than 50% confident — type is a rough guess.
    Low,
    /// 50–80% confident — heuristic pattern matched but ambiguous.
    Medium,
    /// Greater than 80% confident — strong structural evidence (e.g. valid
    /// IEEE 754 float or negative i16).
    High,
}

// ─── Inference result ─────────────────────────────────────────────────────────

/// Complete inference result for a single register (or Float32 pair).
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Register address (first register for Float32 pairs).
    pub address: u16,
    /// Inferred data type.
    pub inferred_type: InferredType,
    /// Confidence in the inferred type.
    pub confidence: ConfidenceLevel,
    /// Raw 16-bit value (first register for Float32 pairs).
    pub raw_value: u16,
    /// Inferred floating-point representation of the value.
    pub inferred_f64: f64,
    /// Human-readable diagnostic note explaining the inference decision.
    pub notes: String,
}

// ─── Inferrer ────────────────────────────────────────────────────────────────

/// Stateless type inferrer for Modbus register values.
pub struct TypeInferrer;

impl TypeInferrer {
    /// Infer the type of a single 16-bit raw register value.
    ///
    /// Returns `(InferredType, ConfidenceLevel, f64_value, notes)`.
    pub fn infer_single(raw: u16) -> (InferredType, ConfidenceLevel, f64, String) {
        // 1. Boolean: exactly 0 or 1
        if raw == 0 || raw == 1 {
            return (
                InferredType::Boolean,
                ConfidenceLevel::Low,
                raw as f64,
                "Only 0 or 1 observed".into(),
            );
        }

        // 2. Signed 16-bit: value ≥ 0x8000 means negative in two's complement
        let as_i16 = raw as i16;
        if as_i16 < 0 {
            return (
                InferredType::Int16,
                ConfidenceLevel::Medium,
                as_i16 as f64,
                "Negative value in 16-bit two's complement range".into(),
            );
        }

        // 3. Scaled integer heuristic (common in industrial devices)
        // If 1 < raw < 10_000 and raw / 10.0 falls in (0, 1000), suggest
        // scale=0.1 as an industrial "one decimal place" convention.
        if raw > 1 && raw < 10_000 {
            let scaled = raw as f64 / 10.0;
            if scaled > 0.0 && scaled < 1_000.0 {
                return (
                    InferredType::ScaledInt { scale: 0.1 },
                    ConfidenceLevel::Low,
                    scaled,
                    "Plausible 0.1 scale (industrial one-decimal-place convention)".into(),
                );
            }
        }

        // 4. Default: unsigned 16-bit
        (
            InferredType::UInt16,
            ConfidenceLevel::Medium,
            raw as f64,
            "Positive value above scaled-int threshold — treating as UInt16".into(),
        )
    }

    /// Infer types for a slice of consecutive registers.
    ///
    /// Adjacent registers with matching function codes are tested for Float32
    /// before falling back to per-register inference. When a Float32 is
    /// detected, **both** source registers are consumed and only one
    /// [`InferenceResult`] is emitted (at the lower address).
    pub fn infer_batch(registers: &[DiscoveredRegister]) -> Vec<InferenceResult> {
        let mut results = Vec::with_capacity(registers.len());
        let mut i = 0;

        while i < registers.len() {
            let reg = &registers[i];

            // Try Float32 if a consecutive same-FC neighbour exists.
            if let Some(next) = registers.get(i + 1) {
                if next.address == reg.address + 1 && next.function_code == reg.function_code {
                    let combined = ((reg.raw_u16 as u32) << 16) | next.raw_u16 as u32;
                    let f = f32::from_bits(combined);

                    // Accept as Float32 only when the decoded value is finite
                    // and within a plausible industrial range.
                    if f.is_finite() && f.abs() > 1e-6 && f.abs() < 1e6 {
                        results.push(InferenceResult {
                            address: reg.address,
                            inferred_type: InferredType::Float32,
                            confidence: ConfidenceLevel::High,
                            raw_value: reg.raw_u16,
                            inferred_f64: f as f64,
                            notes: format!(
                                "Float32 BE spanning registers {}-{}: {f}",
                                reg.address, next.address
                            ),
                        });
                        i += 2; // consume both
                        continue;
                    }
                }
            }

            let (inferred_type, confidence, inferred_f64, notes) = Self::infer_single(reg.raw_u16);
            results.push(InferenceResult {
                address: reg.address,
                inferred_type,
                confidence,
                raw_value: reg.raw_u16,
                inferred_f64,
                notes,
            });
            i += 1;
        }

        results
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::driver::{DiscoveredRegister, DiscoveryFunctionCode};

    #[test]
    fn infer_boolean_zero() {
        let (t, _, val, _) = TypeInferrer::infer_single(0);
        assert_eq!(t, InferredType::Boolean);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn infer_boolean_one() {
        let (t, _, _, _) = TypeInferrer::infer_single(1);
        assert_eq!(t, InferredType::Boolean);
    }

    #[test]
    fn infer_int16_negative() {
        let (t, _, val, _) = TypeInferrer::infer_single((-50i16) as u16);
        assert_eq!(t, InferredType::Int16);
        assert!((val - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn infer_int16_confidence_medium() {
        let (_, c, _, _) = TypeInferrer::infer_single((-1i16) as u16);
        assert_eq!(c, ConfidenceLevel::Medium);
    }

    #[test]
    fn infer_scaled_int_225() {
        let (t, _, val, _) = TypeInferrer::infer_single(225);
        assert!(matches!(t, InferredType::ScaledInt { scale } if (scale - 0.1).abs() < 1e-12));
        assert!((val - 22.5).abs() < 1e-10);
    }

    #[test]
    fn infer_uint16_large_value() {
        // 25_000 is above the ScaledInt cutoff (10_000) and positive — UInt16
        let (t, _, val, _) = TypeInferrer::infer_single(25_000);
        assert_eq!(t, InferredType::UInt16);
        assert!((val - 25_000.0).abs() < 1e-10);
    }

    #[test]
    fn infer_float32_from_batch() {
        let f: f32 = 22.5;
        let bits = f.to_bits();
        let hi = (bits >> 16) as u16;
        let lo = (bits & 0xFFFF) as u16;

        let registers = vec![
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 10, hi),
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 11, lo),
        ];
        let results = TypeInferrer::infer_batch(&registers);

        assert_eq!(
            results.len(),
            1,
            "Pair should collapse to one Float32 result"
        );
        assert!(
            matches!(results[0].inferred_type, InferredType::Float32),
            "Expected Float32, got {:?}",
            results[0].inferred_type
        );
        assert!(
            (results[0].inferred_f64 - 22.5).abs() < 0.001,
            "Float value mismatch: {}",
            results[0].inferred_f64
        );
        assert_eq!(results[0].confidence, ConfidenceLevel::High);
    }

    #[test]
    fn infer_batch_non_float_pair_stays_separate() {
        // Values that when combined don't form a plausible IEEE 754 float
        let registers = vec![
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 0, 0xFFFF),
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 1, 0xFFFF),
        ];
        // 0xFFFF_FFFF is NaN in IEEE 754 — should NOT collapse
        let results = TypeInferrer::infer_batch(&registers);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn infer_batch_different_fc_no_float32_merge() {
        let f: f32 = 10.0;
        let bits = f.to_bits();
        let hi = (bits >> 16) as u16;
        let lo = (bits & 0xFFFF) as u16;

        let registers = vec![
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 0, hi),
            // Different function code — must NOT be merged into Float32
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadInputRegisters, 1, lo),
        ];
        let results = TypeInferrer::infer_batch(&registers);
        assert_eq!(
            results.len(),
            2,
            "Different FC must not trigger Float32 merge"
        );
    }

    #[test]
    fn infer_batch_non_consecutive_no_float32_merge() {
        let f: f32 = 10.0;
        let bits = f.to_bits();
        let hi = (bits >> 16) as u16;
        let lo = (bits & 0xFFFF) as u16;

        let registers = vec![
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 0, hi),
            // Address gap — must NOT be merged
            DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 2, lo),
        ];
        let results = TypeInferrer::infer_batch(&registers);
        assert_eq!(
            results.len(),
            2,
            "Non-consecutive addresses must not trigger Float32 merge"
        );
    }

    #[test]
    fn infer_batch_empty_input() {
        let results = TypeInferrer::infer_batch(&[]);
        assert!(results.is_empty());
    }
}
