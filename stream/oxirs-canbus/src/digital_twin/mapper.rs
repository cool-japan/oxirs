//! J1939 SPN value extraction and conversion to DTDL [`TwinValue`].
//!
//! SAE J1939 defines two global "not-available" indicators that sensors transmit
//! when a measurement cannot be taken:
//!
//! - `0xFE` — *parameter-specific indicator* (a device-defined condition)
//! - `0xFF` — *not available* (the sensor is not installed or not responding)
//!
//! When either indicator appears in byte 0 of the data field, [`extract_spn`]
//! returns [`SpnValue::NotAvailable`] and [`spn_to_twin_value`] returns `None`,
//! so the bridge skips the write rather than propagating a meaningless value.
//!
//! # Byte-offset simplification
//!
//! Real J1939 SPNs can span arbitrary bit ranges across the 8-byte payload.
//! This module uses a simplified extraction rule: **byte 0 of the frame data
//! field** is used as the SPN raw byte.  A production implementation would
//! consult the full J1939-71 bit-offset table, which is out of scope here.

use oxirs_physics::digital_twin::twin_value::TwinValue;

use crate::digital_twin::config::{MappingDirection, RegisterMapping};

// ─────────────────────────────────────────────────────────────────────────────
// SpnValue
// ─────────────────────────────────────────────────────────────────────────────

/// Decoded SPN value, or the J1939 "not available" sentinel.
#[derive(Debug, Clone, PartialEq)]
pub enum SpnValue {
    /// Floating-point physical value after scaling (currently unused by default
    /// extraction, but provided for callers that apply a linear scale/offset).
    Float(f64),
    /// Integer raw or scaled value.
    Integer(i64),
    /// Boolean flag (e.g. a 1-bit SPN like "lamp on/off").
    Boolean(bool),
    /// J1939 indicator 0xFE (parameter-specific) or 0xFF (not available).
    /// No DTDL property should be written for this value.
    NotAvailable,
}

// ─────────────────────────────────────────────────────────────────────────────
// Extraction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a typed SPN value from a J1939 data field.
///
/// # Simplification
///
/// Byte 0 of `data` is used as the raw SPN byte regardless of `_spn`.  The
/// `_spn` parameter is present so callers can thread the SPN identity through
/// for future bit-precise extraction without changing the API.
///
/// # J1939 not-available indicators
///
/// Returns [`SpnValue::NotAvailable`] when byte 0 is `0xFE` or `0xFF`.
pub fn extract_spn(data: &[u8; 8], _spn: u32) -> SpnValue {
    match data[0] {
        0xFE | 0xFF => SpnValue::NotAvailable,
        v => SpnValue::Integer(i64::from(v)),
    }
}

/// Convert an [`SpnValue`] to a [`TwinValue`], returning `None` for
/// [`SpnValue::NotAvailable`].
///
/// `None` signals the bridge that no DTDL property write should occur.
pub fn spn_to_twin_value(spn_val: SpnValue) -> Option<TwinValue> {
    match spn_val {
        SpnValue::Float(f) => Some(TwinValue::Float(f)),
        SpnValue::Integer(i) => Some(TwinValue::Integer(i)),
        SpnValue::Boolean(b) => Some(TwinValue::Boolean(b)),
        SpnValue::NotAvailable => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyMapper
// ─────────────────────────────────────────────────────────────────────────────

/// Lookup table from (PGN, SPN) pairs to DTDL twin property names.
///
/// Built from a [`BridgeConfig`](crate::digital_twin::config::BridgeConfig) mapping list at
/// bridge start-up.  The mapper is intentionally immutable after construction —
/// configuration changes require a bridge restart.
pub struct PropertyMapper {
    mappings: Vec<RegisterMapping>,
}

impl PropertyMapper {
    /// Construct from a list of [`RegisterMapping`] entries.
    pub fn new(mappings: Vec<RegisterMapping>) -> Self {
        Self { mappings }
    }

    /// Find the DTDL twin property name for a (PGN, SPN) pair in the read
    /// (J1939 → DTDL) direction.
    ///
    /// Returns `None` when no mapping exists for the given pair, or when the
    /// matching mapping has direction [`Write`](MappingDirection::Write) only.
    pub fn find_property_for_read(&self, pgn: u32, spn: u32) -> Option<&str> {
        self.mappings
            .iter()
            .find(|m| {
                m.pgn == pgn
                    && m.spn == spn
                    && matches!(
                        m.direction,
                        MappingDirection::Read | MappingDirection::Bidirectional
                    )
            })
            .map(|m| m.twin_property.as_str())
    }

    /// Find the DTDL twin property name for a (PGN, SPN) pair in the write
    /// (DTDL → J1939) direction.
    ///
    /// Returns `None` when no mapping exists, or when the mapping is read-only.
    pub fn find_property_for_write(&self, pgn: u32, spn: u32) -> Option<&str> {
        self.mappings
            .iter()
            .find(|m| {
                m.pgn == pgn
                    && m.spn == spn
                    && matches!(
                        m.direction,
                        MappingDirection::Write | MappingDirection::Bidirectional
                    )
            })
            .map(|m| m.twin_property.as_str())
    }

    /// Convenience: find a read-direction mapping by PGN only (SPN is ignored).
    ///
    /// When multiple mappings share the same PGN, the first in configuration
    /// order is returned.  This is the fast-path used by the bridge's frame
    /// processor when the SPN identity is not decoded from the payload.
    pub fn find_property(&self, pgn: u32, spn: u32) -> Option<&str> {
        self.find_property_for_read(pgn, spn)
    }

    /// Number of loaded mappings.
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digital_twin::config::MappingDirection;

    fn make_mapper() -> PropertyMapper {
        PropertyMapper::new(vec![
            RegisterMapping {
                pgn: 65262,
                spn: 110,
                twin_property: "engine.coolant_temp_c".to_string(),
                direction: MappingDirection::Read,
            },
            RegisterMapping {
                pgn: 65265,
                spn: 84,
                twin_property: "vehicle.speed_kmh".to_string(),
                direction: MappingDirection::Bidirectional,
            },
            RegisterMapping {
                pgn: 61444,
                spn: 190,
                twin_property: "engine.rpm".to_string(),
                direction: MappingDirection::Write,
            },
        ])
    }

    // ── extract_spn ──────────────────────────────────────────────────────────

    #[test]
    fn extract_spn_normal_byte() {
        let data = [75u8, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(extract_spn(&data, 110), SpnValue::Integer(75));
    }

    #[test]
    fn extract_spn_zero() {
        let data = [0u8; 8];
        assert_eq!(extract_spn(&data, 84), SpnValue::Integer(0));
    }

    #[test]
    fn extract_spn_na_0xfe() {
        let data = [0xFEu8, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(extract_spn(&data, 110), SpnValue::NotAvailable);
    }

    #[test]
    fn extract_spn_na_0xff() {
        let data = [0xFFu8, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(extract_spn(&data, 110), SpnValue::NotAvailable);
    }

    #[test]
    fn extract_spn_max_valid() {
        // 0xFD = 253 — just below the NA indicators
        let data = [0xFDu8, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(extract_spn(&data, 0), SpnValue::Integer(253));
    }

    // ── spn_to_twin_value ────────────────────────────────────────────────────

    #[test]
    fn spn_to_twin_value_integer() {
        assert_eq!(
            spn_to_twin_value(SpnValue::Integer(75)),
            Some(TwinValue::Integer(75))
        );
    }

    #[test]
    fn spn_to_twin_value_float() {
        assert_eq!(
            spn_to_twin_value(SpnValue::Float(98.6)),
            Some(TwinValue::Float(98.6))
        );
    }

    #[test]
    fn spn_to_twin_value_boolean() {
        assert_eq!(
            spn_to_twin_value(SpnValue::Boolean(true)),
            Some(TwinValue::Boolean(true))
        );
    }

    #[test]
    fn spn_to_twin_value_not_available_is_none() {
        assert_eq!(spn_to_twin_value(SpnValue::NotAvailable), None);
    }

    // ── PropertyMapper ───────────────────────────────────────────────────────

    #[test]
    fn mapper_find_property_read_direction() {
        let mapper = make_mapper();
        assert_eq!(
            mapper.find_property(65262, 110),
            Some("engine.coolant_temp_c")
        );
    }

    #[test]
    fn mapper_find_property_bidirectional() {
        let mapper = make_mapper();
        assert_eq!(mapper.find_property(65265, 84), Some("vehicle.speed_kmh"));
    }

    #[test]
    fn mapper_find_property_write_only_returns_none_for_read() {
        let mapper = make_mapper();
        // pgn=61444, spn=190 is Write-only — should NOT appear in the read path
        assert_eq!(mapper.find_property(61444, 190), None);
    }

    #[test]
    fn mapper_find_property_for_write() {
        let mapper = make_mapper();
        // Bidirectional also appears in write path
        assert_eq!(
            mapper.find_property_for_write(65265, 84),
            Some("vehicle.speed_kmh")
        );
        // Write-only appears in write path
        assert_eq!(
            mapper.find_property_for_write(61444, 190),
            Some("engine.rpm")
        );
    }

    #[test]
    fn mapper_find_property_missing_pgn() {
        let mapper = make_mapper();
        assert_eq!(mapper.find_property(99999, 0), None);
    }

    #[test]
    fn mapper_find_property_wrong_spn() {
        let mapper = make_mapper();
        // pgn=65262 exists but spn=999 doesn't
        assert_eq!(mapper.find_property(65262, 999), None);
    }

    #[test]
    fn mapper_count() {
        let mapper = make_mapper();
        assert_eq!(mapper.mapping_count(), 3);
    }
}
