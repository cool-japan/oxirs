//! Candidate register map emitter.
//!
//! [`RegisterMapEmitter`] converts [`InferenceResult`] values produced by
//! [`TypeInferrer`] into a [`CandidateRegisterMap`] that can be serialised to
//! JSON or a human-readable YAML-like format, and consumed by
//! `register_validator` or stored in a device profile.
//!
//! [`CandidateRegisterMap`]: crate::discovery::emitter::CandidateRegisterMap

use serde::{Deserialize, Serialize};

use super::inference::{ConfidenceLevel, InferenceResult, InferredType};

// ─── Output types ─────────────────────────────────────────────────────────────

/// One entry in the candidate register map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRegisterEntry {
    /// 0-based Modbus register address.
    pub address: u16,
    /// Auto-generated register name (`register_0xAAAA`).
    pub name: String,
    /// Inferred data type string (`uint16`, `int16`, `float32`,
    /// `scaled_int16`, `boolean`, `unknown`).
    pub data_type: String,
    /// Scaling factor (only present for `scaled_int16`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// Confidence string (`low`, `medium`, `high`).
    pub confidence: String,
    /// Raw 16-bit value observed during the scan.
    pub raw_value: u16,
    /// Inferred physical value after type decoding.
    pub inferred_value: f64,
    /// Human-readable diagnostic note from the inferrer.
    pub notes: String,
}

/// A candidate register map — the primary output of the discovery pipeline.
///
/// Produced by [`RegisterMapEmitter::emit`] and can be serialised to JSON
/// with [`RegisterMapEmitter::to_json`] or inspected as a YAML-like string
/// with [`RegisterMapEmitter::to_yaml_like`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRegisterMap {
    /// Modbus unit / slave ID that was probed.
    pub unit_id: u8,
    /// Inferred register entries.
    pub entries: Vec<CandidateRegisterEntry>,
    /// Total number of registers that were probed during the scan.
    pub total_registers_probed: usize,
    /// Number of entries with [`ConfidenceLevel::High`].
    pub high_confidence_count: usize,
}

// ─── Emitter ─────────────────────────────────────────────────────────────────

/// Converts inference results into a serialisable [`CandidateRegisterMap`].
pub struct RegisterMapEmitter;

impl RegisterMapEmitter {
    /// Build a [`CandidateRegisterMap`] from inference results.
    ///
    /// * `unit_id` — the Modbus unit ID that was probed.
    /// * `results` — ordered slice of [`InferenceResult`] values.
    /// * `total_probed` — total registers probed (may exceed `results.len()`
    ///   for Float32 pairs where two registers collapse to one result).
    pub fn emit(
        unit_id: u8,
        results: &[InferenceResult],
        total_probed: usize,
    ) -> CandidateRegisterMap {
        let entries: Vec<CandidateRegisterEntry> = results
            .iter()
            .map(|r| {
                let (data_type, scale) = match &r.inferred_type {
                    InferredType::UInt16 => ("uint16".to_owned(), None),
                    InferredType::Int16 => ("int16".to_owned(), None),
                    InferredType::Float32 => ("float32".to_owned(), None),
                    InferredType::ScaledInt { scale } => ("scaled_int16".to_owned(), Some(*scale)),
                    InferredType::Boolean => ("boolean".to_owned(), None),
                    InferredType::Unknown => ("unknown".to_owned(), None),
                };
                let confidence = match r.confidence {
                    ConfidenceLevel::Low => "low",
                    ConfidenceLevel::Medium => "medium",
                    ConfidenceLevel::High => "high",
                };
                CandidateRegisterEntry {
                    address: r.address,
                    name: format!("register_{:#06x}", r.address),
                    data_type,
                    scale,
                    confidence: confidence.to_owned(),
                    raw_value: r.raw_value,
                    inferred_value: r.inferred_f64,
                    notes: r.notes.clone(),
                }
            })
            .collect();

        let high_confidence_count = results
            .iter()
            .filter(|r| r.confidence == ConfidenceLevel::High)
            .count();

        CandidateRegisterMap {
            unit_id,
            entries,
            total_registers_probed: total_probed,
            high_confidence_count,
        }
    }

    /// Serialize a [`CandidateRegisterMap`] to pretty-printed JSON.
    pub fn to_json(map: &CandidateRegisterMap) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(map)
    }

    /// Produce a YAML-like multi-line summary (no external YAML dependency).
    ///
    /// Useful for quick CLI output or log messages.
    pub fn to_yaml_like(map: &CandidateRegisterMap) -> String {
        let mut out = format!(
            "unit_id: {}\ntotal_probed: {}\nhigh_confidence: {}\nregisters:\n",
            map.unit_id, map.total_registers_probed, map.high_confidence_count
        );
        for entry in &map.entries {
            out.push_str(&format!(
                "  - address: {:#06x}\n    name: {}\n    type: {}\n    confidence: {}\n    raw: {}\n    inferred: {:.4}\n",
                entry.address,
                entry.name,
                entry.data_type,
                entry.confidence,
                entry.raw_value,
                entry.inferred_value,
            ));
            if let Some(scale) = entry.scale {
                out.push_str(&format!("    scale: {scale}\n"));
            }
        }
        out
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::inference::{ConfidenceLevel, InferenceResult, InferredType};

    fn make_result(
        address: u16,
        t: InferredType,
        confidence: ConfidenceLevel,
        raw: u16,
        val: f64,
    ) -> InferenceResult {
        InferenceResult {
            address,
            inferred_type: t,
            confidence,
            raw_value: raw,
            inferred_f64: val,
            notes: "test".into(),
        }
    }

    #[test]
    fn emit_basic() {
        let results = vec![
            make_result(
                0,
                InferredType::UInt16,
                ConfidenceLevel::Medium,
                1000,
                1000.0,
            ),
            make_result(
                1,
                InferredType::Int16,
                ConfidenceLevel::Medium,
                65436,
                -100.0,
            ),
            make_result(
                2,
                InferredType::Float32,
                ConfidenceLevel::High,
                0x4134,
                22.5,
            ),
            make_result(
                4,
                InferredType::ScaledInt { scale: 0.1 },
                ConfidenceLevel::Low,
                225,
                22.5,
            ),
            make_result(5, InferredType::Boolean, ConfidenceLevel::Low, 1, 1.0),
        ];
        let map = RegisterMapEmitter::emit(1, &results, 6);

        assert_eq!(map.unit_id, 1);
        assert_eq!(map.entries.len(), 5);
        assert_eq!(map.total_registers_probed, 6);
        assert_eq!(map.high_confidence_count, 1); // only Float32

        assert_eq!(map.entries[0].data_type, "uint16");
        assert_eq!(map.entries[1].data_type, "int16");
        assert_eq!(map.entries[2].data_type, "float32");
        assert_eq!(map.entries[3].data_type, "scaled_int16");
        assert_eq!(map.entries[3].scale, Some(0.1));
        assert_eq!(map.entries[4].data_type, "boolean");
    }

    #[test]
    fn emit_unknown_type() {
        let results = vec![make_result(
            0,
            InferredType::Unknown,
            ConfidenceLevel::Low,
            0,
            0.0,
        )];
        let map = RegisterMapEmitter::emit(1, &results, 1);
        assert_eq!(map.entries[0].data_type, "unknown");
    }

    #[test]
    fn to_json_contains_required_keys() {
        let results = vec![make_result(
            1,
            InferredType::UInt16,
            ConfidenceLevel::Medium,
            42,
            42.0,
        )];
        let map = RegisterMapEmitter::emit(1, &results, 1);
        let json = RegisterMapEmitter::to_json(&map).expect("JSON serialization");
        assert!(json.contains("\"address\""));
        assert!(json.contains("\"data_type\""));
        assert!(json.contains("\"confidence\""));
        assert!(json.contains("\"unit_id\""));
    }

    #[test]
    fn to_yaml_like_contains_structure() {
        let results = vec![make_result(
            1,
            InferredType::UInt16,
            ConfidenceLevel::Medium,
            42,
            42.0,
        )];
        let map = RegisterMapEmitter::emit(1, &results, 1);
        let yaml = RegisterMapEmitter::to_yaml_like(&map);
        assert!(yaml.contains("unit_id:"));
        assert!(yaml.contains("registers:"));
        assert!(yaml.contains("0x0001"));
        assert!(yaml.contains("uint16"));
    }

    #[test]
    fn to_yaml_like_includes_scale_for_scaled_int() {
        let results = vec![make_result(
            0,
            InferredType::ScaledInt { scale: 0.1 },
            ConfidenceLevel::Low,
            225,
            22.5,
        )];
        let map = RegisterMapEmitter::emit(1, &results, 1);
        let yaml = RegisterMapEmitter::to_yaml_like(&map);
        assert!(
            yaml.contains("scale:"),
            "YAML should include scale for ScaledInt"
        );
    }

    #[test]
    fn to_json_no_scale_field_for_uint16() {
        let results = vec![make_result(
            0,
            InferredType::UInt16,
            ConfidenceLevel::Medium,
            100,
            100.0,
        )];
        let map = RegisterMapEmitter::emit(1, &results, 1);
        let json = RegisterMapEmitter::to_json(&map).expect("JSON serialization");
        // scale field should be absent (skip_serializing_if = "Option::is_none")
        assert!(
            !json.contains("\"scale\""),
            "UInt16 must not emit scale field"
        );
    }

    #[test]
    fn register_name_format() {
        let results = vec![make_result(
            0x00FF,
            InferredType::UInt16,
            ConfidenceLevel::Medium,
            0,
            0.0,
        )];
        let map = RegisterMapEmitter::emit(1, &results, 1);
        assert_eq!(map.entries[0].name, "register_0x00ff");
    }
}
