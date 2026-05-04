//! Bridge configuration types for the J1939 ↔ DTDL bridge.
//!
//! [`BridgeConfig`] is the top-level configuration struct, typically loaded from
//! a TOML file. It contains a list of [`RegisterMapping`] entries that define how
//! individual SAE J1939 PGN/SPN pairs map to named DTDL twin properties.
//!
//! # TOML example
//!
//! ```toml
//! poll_interval_ms = 100
//!
//! [[mapping]]
//! pgn = 65262
//! spn = 110
//! twin_property = "engine.coolant_temp_c"
//! direction = "read"
//!
//! [[mapping]]
//! pgn = 65265
//! spn = 84
//! twin_property = "vehicle.speed_kmh"
//! direction = "bidirectional"
//! ```

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// MappingDirection
// ─────────────────────────────────────────────────────────────────────────────

/// Direction of the J1939 ↔ DTDL data flow for a single mapping entry.
///
/// - [`Read`](Self::Read) — data flows from J1939 network into the DTDL twin
///   (sensor observation).
/// - [`Write`](Self::Write) — data flows from the DTDL twin back to the J1939
///   network (actuator command).
/// - [`Bidirectional`](Self::Bidirectional) — data can flow in both directions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappingDirection {
    /// J1939 → DTDL (sensor / observation path).
    #[serde(rename = "read")]
    Read,
    /// DTDL → J1939 (actuator / command path).
    #[serde(rename = "write")]
    Write,
    /// Both directions are active.
    #[serde(rename = "bidirectional")]
    Bidirectional,
}

// ─────────────────────────────────────────────────────────────────────────────
// RegisterMapping
// ─────────────────────────────────────────────────────────────────────────────

/// A single PGN/SPN ↔ DTDL-property binding.
///
/// `pgn` and `spn` identify the J1939 signal; `twin_property` names the DTDL
/// property; `direction` controls the data-flow direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterMapping {
    /// SAE J1939 Parameter Group Number (e.g. 65262 for ET1 — Engine Temperatures).
    pub pgn: u32,
    /// SAE J1939 Suspect Parameter Number within the PGN (e.g. 110 for coolant
    /// temperature).
    pub spn: u32,
    /// Dot-separated DTDL property path (e.g. `"engine.coolant_temp_c"`).
    pub twin_property: String,
    /// Data-flow direction.
    pub direction: MappingDirection,
}

// ─────────────────────────────────────────────────────────────────────────────
// BridgeConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level configuration for the [`super::bridge::J1939DtdlBridge`].
///
/// Deserialises from TOML (see module-level docstring for an example).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// How often (in milliseconds) the bridge polls the J1939 source when no
    /// frame has been received. Has no effect when the source is event-driven
    /// rather than polled.
    pub poll_interval_ms: u64,
    /// Zero or more PGN/SPN ↔ DTDL-property mappings.
    #[serde(rename = "mapping", default)]
    pub mappings: Vec<RegisterMapping>,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 100,
            mappings: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_toml_bridge_config() {
        let toml_str = r#"
poll_interval_ms = 200
[[mapping]]
pgn = 65262
spn = 110
twin_property = "engine.coolant_temp_c"
direction = "read"
[[mapping]]
pgn = 65265
spn = 84
twin_property = "vehicle.speed_kmh"
direction = "bidirectional"
"#;
        let cfg: BridgeConfig = toml::from_str(toml_str).expect("should parse");
        assert_eq!(cfg.poll_interval_ms, 200);
        assert_eq!(cfg.mappings.len(), 2);
        assert_eq!(cfg.mappings[0].pgn, 65262);
        assert_eq!(cfg.mappings[0].spn, 110);
        assert_eq!(cfg.mappings[0].twin_property, "engine.coolant_temp_c");
        assert_eq!(cfg.mappings[0].direction, MappingDirection::Read);
        assert_eq!(cfg.mappings[1].direction, MappingDirection::Bidirectional);
    }

    #[test]
    fn bridge_config_default_is_empty() {
        let cfg = BridgeConfig::default();
        assert_eq!(cfg.poll_interval_ms, 100);
        assert!(cfg.mappings.is_empty());
    }

    #[test]
    fn mapping_direction_variants() {
        assert_ne!(MappingDirection::Read, MappingDirection::Write);
        assert_ne!(MappingDirection::Read, MappingDirection::Bidirectional);
        assert_ne!(MappingDirection::Write, MappingDirection::Bidirectional);
    }

    #[test]
    fn round_trip_json_bridge_config() {
        let cfg = BridgeConfig {
            poll_interval_ms: 50,
            mappings: vec![RegisterMapping {
                pgn: 61444,
                spn: 190,
                twin_property: "engine.rpm".to_string(),
                direction: MappingDirection::Write,
            }],
        };
        let json = serde_json::to_string(&cfg).expect("should serialize");
        let cfg2: BridgeConfig = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(cfg2.mappings[0].pgn, 61444);
        assert_eq!(cfg2.mappings[0].direction, MappingDirection::Write);
    }
}
