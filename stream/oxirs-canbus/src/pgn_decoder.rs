//! SAE J1939 PGN (Parameter Group Number) decoder.
//!
//! Implements the J1939 extended 29-bit CAN identifier layout:
//!
//! ```text
//! Bits 28–26 : priority    (3 bits)
//! Bit  25    : reserved    (1 bit)
//! Bit  24    : data_page   (1 bit)
//! Bits 23–16 : PF byte     (8 bits)
//! Bits 15–8  : PS byte     (8 bits)  — destination (PDU1) or group ext (PDU2)
//! Bits  7–0  : source addr (8 bits)
//! ```
//!
//! When PF < 240 (0xF0) the frame uses **PDU Format 1** and the PS byte is a
//! destination address.  When PF ≥ 240 the frame uses **PDU Format 2** and the
//! PS byte is a group extension that forms part of the PGN.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// PDU Format 1 / PDU Format 2 boundary value.
///
/// PF bytes strictly less than this value indicate PDU Format 1 (peer-to-peer);
/// values equal to or greater indicate PDU Format 2 (broadcast).
pub const PDU2_PF_BOUNDARY: u8 = 0xF0;

// ────────────────────────────────────────────────────────────────────────────
// PgnInfo
// ────────────────────────────────────────────────────────────────────────────

/// A decoded J1939 PGN descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct PgnInfo {
    /// The 18-bit (PDU2) or partial (PDU1) PGN value.
    pub pgn: u32,
    /// Human-readable name of the PGN (e.g. "Engine Controller 1").
    pub name: String,
    /// Data page bit (0 or 1).
    pub data_page: u8,
    /// PF byte: PDU Format identifier.
    pub pdu_format: u8,
    /// PS byte: destination address (PDU1) or group extension (PDU2).
    pub pdu_specific: u8,
    /// `true` when this PGN uses PDU Format 1 (PF < 0xF0), i.e. peer-to-peer.
    pub is_pdu1: bool,
    /// Optional nominal transmission rate, e.g. `"10ms"`, `"100ms"`, `"on-request"`.
    pub transmission_rate: Option<String>,
}

impl PgnInfo {
    /// Create a new `PgnInfo`.
    pub fn new(
        pgn: u32,
        name: impl Into<String>,
        data_page: u8,
        pdu_format: u8,
        pdu_specific: u8,
        transmission_rate: Option<impl Into<String>>,
    ) -> Self {
        let is_pdu1 = pdu_format < PDU2_PF_BOUNDARY;
        Self {
            pgn,
            name: name.into(),
            data_page,
            pdu_format,
            pdu_specific,
            is_pdu1,
            transmission_rate: transmission_rate.map(|r| r.into()),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PgnRegistry
// ────────────────────────────────────────────────────────────────────────────

/// A registry that maps PGN numbers to their descriptors.
pub struct PgnRegistry {
    entries: HashMap<u32, PgnInfo>,
}

impl PgnRegistry {
    /// Create an empty registry.
    pub fn empty() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with the most common J1939 PGNs.
    ///
    /// Included PGNs:
    /// - `0xF004` / 61444 — EEC1 Electronic Engine Controller 1
    /// - `0xFEEA` / 65258 — CCVS1 Cruise Control/Vehicle Speed
    /// - `0xFEEB` / 65259 — LFE1 Fuel Economy
    /// - `0xFEF1` / 65265 — EBC1 Electronic Brake Controller 1
    /// - `0xFEF2` / 65266 — EBC2 Electronic Brake Controller 2
    /// - `0xFECA` / 65226 — DM1 Active Diagnostics
    /// - `0xFECB` / 65227 — DM2 Previously Active Diagnostics
    /// - `0xFF00` / 65280 — Proprietary B (broadcast)
    pub fn standard() -> Self {
        let mut r = Self::empty();

        // EEC1 — Engine Controller 1 (PF=0xF0, PDU2, broadcast)
        r.register(PgnInfo::new(
            0xF004,
            "EEC1 Electronic Engine Controller 1",
            0,
            0xF0,
            0x04,
            Some("10ms"),
        ));

        // CCVS1 — Cruise Control / Vehicle Speed (PF=0xFE, PDU2)
        r.register(PgnInfo::new(
            0xFEEA,
            "CCVS1 Cruise Control Vehicle Speed",
            0,
            0xFE,
            0xEA,
            Some("100ms"),
        ));

        // LFE1 — Fuel Economy (PF=0xFE, PDU2)
        r.register(PgnInfo::new(
            0xFEEB,
            "LFE1 Fuel Economy",
            0,
            0xFE,
            0xEB,
            Some("100ms"),
        ));

        // EBC1 — Electronic Brake Controller 1 (PF=0xFE, PDU2)
        r.register(PgnInfo::new(
            0xFEF1,
            "EBC1 Electronic Brake Controller 1",
            0,
            0xFE,
            0xF1,
            Some("100ms"),
        ));

        // EBC2 — Electronic Brake Controller 2 (PF=0xFE, PDU2)
        r.register(PgnInfo::new(
            0xFEF2,
            "EBC2 Electronic Brake Controller 2",
            0,
            0xFE,
            0xF2,
            Some("100ms"),
        ));

        // DM1 — Active Diagnostics Trouble Codes (PF=0xFE, PDU2)
        r.register(PgnInfo::new(
            0xFECA,
            "DM1 Active Diagnostics Trouble Codes",
            0,
            0xFE,
            0xCA,
            Some("on-request"),
        ));

        // DM2 — Previously Active Diagnostics (PF=0xFE, PDU2)
        r.register(PgnInfo::new(
            0xFECB,
            "DM2 Previously Active Diagnostics",
            0,
            0xFE,
            0xCB,
            Some("on-request"),
        ));

        // Proprietary B — user-defined broadcast PGN (PF=0xFF, PDU2)
        r.register(PgnInfo::new(
            0xFF00,
            "Proprietary B",
            0,
            0xFF,
            0x00,
            None::<&str>,
        ));

        r
    }

    /// Register a PGN descriptor (overwrites any existing entry with the same PGN).
    pub fn register(&mut self, info: PgnInfo) {
        self.entries.insert(info.pgn, info);
    }

    /// Look up a PGN descriptor by its number.
    pub fn lookup(&self, pgn: u32) -> Option<&PgnInfo> {
        self.entries.get(&pgn)
    }

    /// Return `true` if the registry contains a descriptor for `pgn`.
    pub fn is_known(&self, pgn: u32) -> bool {
        self.entries.contains_key(&pgn)
    }

    /// Return the total number of registered PGNs.
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// DecodedId
// ────────────────────────────────────────────────────────────────────────────

/// The fields extracted from a 29-bit J1939 extended CAN identifier.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedId {
    /// Priority field (3 bits, 0–7; lower value = higher priority).
    pub priority: u8,
    /// Reserved bit.
    pub reserved: bool,
    /// Data page bit (0 or 1).
    pub data_page: u8,
    /// PDU Format byte (PF).
    pub pf: u8,
    /// PDU Specific byte (PS): destination address (PDU1) or group extension (PDU2).
    pub ps: u8,
    /// Source address byte (SA).
    pub sa: u8,
    /// `true` when PF < 0xF0 (PDU Format 1, peer-to-peer).
    pub is_pdu1: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// PgnDecoder
// ────────────────────────────────────────────────────────────────────────────

/// Decodes and encodes 29-bit J1939 extended CAN identifiers.
#[derive(Debug, Clone, Default)]
pub struct PgnDecoder;

impl PgnDecoder {
    /// Create a new `PgnDecoder`.
    pub fn new() -> Self {
        Self
    }

    /// Decode a 29-bit J1939 extended CAN ID into its component fields.
    ///
    /// Only the lower 29 bits of `can_id` are significant; the upper 3 bits are masked away.
    ///
    /// Bit layout (bit 28 is the most-significant used bit):
    /// ```text
    /// [28:26] priority  (3 bits)
    /// [25]    reserved  (1 bit)
    /// [24]    data_page (1 bit)
    /// [23:16] PF        (8 bits)
    /// [15:8]  PS        (8 bits)
    /// [7:0]   SA        (8 bits)
    /// ```
    pub fn decode_can_id(&self, can_id: u32) -> DecodedId {
        let id = can_id & 0x1FFF_FFFF; // mask to 29 bits

        let priority = ((id >> 26) & 0x07) as u8;
        let reserved = ((id >> 25) & 0x01) != 0;
        let data_page = ((id >> 24) & 0x01) as u8;
        let pf = ((id >> 16) & 0xFF) as u8;
        let ps = ((id >> 8) & 0xFF) as u8;
        let sa = (id & 0xFF) as u8;
        let is_pdu1 = pf < PDU2_PF_BOUNDARY;

        DecodedId {
            priority,
            reserved,
            data_page,
            pf,
            ps,
            sa,
            is_pdu1,
        }
    }

    /// Build a 29-bit J1939 CAN ID from its component fields.
    ///
    /// Only the lower bits of each argument are used:
    /// - `priority`: 3 bits (0–7)
    /// - `data_page`: 1 bit (0–1)
    /// - `pf`, `ps`, `sa`: 8 bits each
    ///
    /// The reserved bit is set to `0`.
    pub fn encode_can_id(&self, priority: u8, data_page: u8, pf: u8, ps: u8, sa: u8) -> u32 {
        let p = (priority as u32 & 0x07) << 26;
        let dp = (data_page as u32 & 0x01) << 24;
        let pf32 = (pf as u32) << 16;
        let ps32 = (ps as u32) << 8;
        let sa32 = sa as u32;
        p | dp | pf32 | ps32 | sa32
    }

    /// Extract the 18-bit PGN from a decoded J1939 ID.
    ///
    /// For PDU Format 2 (PF ≥ 0xF0) the PGN includes the PS byte as a group extension:
    /// ```text
    /// PGN = (data_page << 17) | (PF << 8) | PS
    /// ```
    /// For PDU Format 1 (PF < 0xF0) the PS byte is a destination address and is **not**
    /// part of the PGN:
    /// ```text
    /// PGN = (data_page << 17) | (PF << 8)
    /// ```
    pub fn pgn_from_decoded(&self, decoded: &DecodedId) -> u32 {
        let dp = (decoded.data_page as u32) << 17;
        let pf = (decoded.pf as u32) << 8;
        if decoded.is_pdu1 {
            dp | pf
        } else {
            dp | pf | (decoded.ps as u32)
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn decoder() -> PgnDecoder {
        PgnDecoder::new()
    }

    /// Build a canonical EEC1 CAN ID: priority=6, reserved=0, dp=0, PF=0xF0, PS=0x04, SA=0x00
    fn eec1_can_id() -> u32 {
        // 6 << 26 | 0xF0 << 16 | 0x04 << 8 | 0x00
        (6 << 26) | (0xF0_u32 << 16) | (0x04_u32 << 8)
    }

    // ── decode_can_id — priority ──────────────────────────────────────────

    #[test]
    fn test_decode_priority_zero() {
        let d = decoder();
        let id = d.encode_can_id(0, 0, 0xFE, 0xF1, 0x11);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.priority, 0);
    }

    #[test]
    fn test_decode_priority_max() {
        let d = decoder();
        let id = d.encode_can_id(7, 0, 0xFE, 0xF1, 0x11);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.priority, 7);
    }

    #[test]
    fn test_decode_priority_six() {
        let d = decoder();
        let dec = d.decode_can_id(eec1_can_id());
        assert_eq!(dec.priority, 6);
    }

    // ── decode_can_id — data_page ─────────────────────────────────────────

    #[test]
    fn test_decode_data_page_zero() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 0xFE, 0xF1, 0x00);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.data_page, 0);
    }

    #[test]
    fn test_decode_data_page_one() {
        let d = decoder();
        let id = d.encode_can_id(6, 1, 0xFE, 0xF1, 0x00);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.data_page, 1);
    }

    // ── decode_can_id — pf / ps / sa ──────────────────────────────────────

    #[test]
    fn test_decode_pf_ps_sa_basic() {
        let d = decoder();
        let id = d.encode_can_id(3, 0, 0xFE, 0xF1, 0x55);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.pf, 0xFE);
        assert_eq!(dec.ps, 0xF1);
        assert_eq!(dec.sa, 0x55);
    }

    #[test]
    fn test_decode_sa_max() {
        let d = decoder();
        let id = d.encode_can_id(0, 0, 0xFE, 0x00, 0xFF);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.sa, 0xFF);
    }

    #[test]
    fn test_decode_pf_zero() {
        let d = decoder();
        let id = d.encode_can_id(0, 0, 0x00, 0x10, 0x01);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.pf, 0x00);
    }

    // ── decode_can_id — is_pdu1 ───────────────────────────────────────────

    #[test]
    fn test_decode_is_pdu1_when_pf_lt_240() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 0xEF, 0x01, 0x00); // PF = 0xEF < 0xF0
        let dec = d.decode_can_id(id);
        assert!(dec.is_pdu1);
    }

    #[test]
    fn test_decode_is_pdu1_boundary_pf_239() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 239, 0x01, 0x00); // 239 = 0xEF
        let dec = d.decode_can_id(id);
        assert!(dec.is_pdu1);
    }

    #[test]
    fn test_decode_is_pdu2_when_pf_eq_240() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 240, 0xF0, 0x00); // PF = 0xF0 = 240
        let dec = d.decode_can_id(id);
        assert!(!dec.is_pdu1);
    }

    #[test]
    fn test_decode_is_pdu2_when_pf_gt_240() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 0xFF, 0x00, 0x00); // PF = 0xFF
        let dec = d.decode_can_id(id);
        assert!(!dec.is_pdu1);
    }

    #[test]
    fn test_decode_is_pdu2_eec1() {
        let d = decoder();
        let dec = d.decode_can_id(eec1_can_id());
        assert!(!dec.is_pdu1); // EEC1 PF = 0xF0 → PDU2
    }

    // ── decode_can_id — reserved bit ─────────────────────────────────────

    #[test]
    fn test_decode_reserved_not_set_normally() {
        let d = decoder();
        // encode_can_id always sets reserved to 0
        let id = d.encode_can_id(3, 0, 0xFE, 0x00, 0x00);
        let dec = d.decode_can_id(id);
        assert!(!dec.reserved);
    }

    #[test]
    fn test_decode_reserved_set_in_raw_id() {
        let d = decoder();
        // Manually set bit 25
        let id = 1u32 << 25;
        let dec = d.decode_can_id(id);
        assert!(dec.reserved);
    }

    // ── encode / decode round-trip ────────────────────────────────────────

    #[test]
    fn test_encode_decode_round_trip_ebc1() {
        let d = decoder();
        // EBC1: priority=6, dp=0, PF=0xFE, PS=0xF1 (PS = 0xF1 → PGN part), SA=0x11
        let raw = d.encode_can_id(6, 0, 0xFE, 0xF1, 0x11);
        let dec = d.decode_can_id(raw);
        assert_eq!(dec.priority, 6);
        assert_eq!(dec.data_page, 0);
        assert_eq!(dec.pf, 0xFE);
        assert_eq!(dec.ps, 0xF1);
        assert_eq!(dec.sa, 0x11);
        assert!(!dec.is_pdu1);
    }

    #[test]
    fn test_encode_decode_round_trip_pdu1() {
        let d = decoder();
        let raw = d.encode_can_id(3, 0, 0xC0, 0x7F, 0xAA); // PF=0xC0 < 0xF0 → PDU1
        let dec = d.decode_can_id(raw);
        assert_eq!(dec.pf, 0xC0);
        assert_eq!(dec.ps, 0x7F);
        assert_eq!(dec.sa, 0xAA);
        assert!(dec.is_pdu1);
    }

    #[test]
    fn test_encode_decode_round_trip_zero_id() {
        let d = decoder();
        let raw = d.encode_can_id(0, 0, 0, 0, 0);
        let dec = d.decode_can_id(raw);
        assert_eq!(dec.priority, 0);
        assert_eq!(dec.data_page, 0);
        assert_eq!(dec.pf, 0);
        assert_eq!(dec.ps, 0);
        assert_eq!(dec.sa, 0);
    }

    // ── pgn_from_decoded ──────────────────────────────────────────────────

    #[test]
    fn test_pgn_from_decoded_pdu2_eec1() {
        let d = decoder();
        let dec = d.decode_can_id(eec1_can_id());
        let pgn = d.pgn_from_decoded(&dec);
        // PF=0xF0, PS=0x04, dp=0 → PGN = 0xF0<<8 | 0x04 = 0xF004
        assert_eq!(pgn, 0xF004);
    }

    #[test]
    fn test_pgn_from_decoded_pdu2_ebc1() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 0xFE, 0xF1, 0x00);
        let dec = d.decode_can_id(id);
        let pgn = d.pgn_from_decoded(&dec);
        // PF=0xFE, PS=0xF1, dp=0 → PGN = 0xFE<<8 | 0xF1 = 0xFEF1
        assert_eq!(pgn, 0xFEF1);
    }

    #[test]
    fn test_pgn_from_decoded_pdu2_feea() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 0xFE, 0xEA, 0x00);
        let dec = d.decode_can_id(id);
        let pgn = d.pgn_from_decoded(&dec);
        assert_eq!(pgn, 0xFEEA);
    }

    #[test]
    fn test_pgn_from_decoded_pdu2_feeb() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 0xFE, 0xEB, 0x00);
        let dec = d.decode_can_id(id);
        let pgn = d.pgn_from_decoded(&dec);
        assert_eq!(pgn, 0xFEEB);
    }

    #[test]
    fn test_pgn_from_decoded_pdu2_fef2() {
        let d = decoder();
        let id = d.encode_can_id(6, 0, 0xFE, 0xF2, 0x00);
        let dec = d.decode_can_id(id);
        let pgn = d.pgn_from_decoded(&dec);
        assert_eq!(pgn, 0xFEF2);
    }

    #[test]
    fn test_pgn_from_decoded_pdu1_excludes_ps() {
        let d = decoder();
        // PDU1: PF=0xC0 < 0xF0, PS is destination (not part of PGN)
        let id = d.encode_can_id(6, 0, 0xC0, 0x05, 0x00); // PS=0x05 is destination
        let dec = d.decode_can_id(id);
        let pgn = d.pgn_from_decoded(&dec);
        // PGN = 0xC0<<8 = 0xC000 (PS excluded)
        assert_eq!(pgn, 0xC000);
    }

    #[test]
    fn test_pgn_from_decoded_pdu1_different_ps_same_pgn() {
        let d = decoder();
        let id1 = d.encode_can_id(6, 0, 0xC0, 0x05, 0x00); // PS=0x05
        let id2 = d.encode_can_id(6, 0, 0xC0, 0x10, 0x00); // PS=0x10
        let dec1 = d.decode_can_id(id1);
        let dec2 = d.decode_can_id(id2);
        // Both should give same PGN because PS is not part of PGN in PDU1
        assert_eq!(d.pgn_from_decoded(&dec1), d.pgn_from_decoded(&dec2));
    }

    #[test]
    fn test_pgn_from_decoded_data_page_1() {
        let d = decoder();
        let id = d.encode_can_id(6, 1, 0xFE, 0xF1, 0x00); // dp=1
        let dec = d.decode_can_id(id);
        let pgn = d.pgn_from_decoded(&dec);
        // PGN = (1<<17) | (0xFE<<8) | 0xF1 = 0x2_FEF1
        assert_eq!(pgn, (1u32 << 17) | 0xFEF1);
    }

    // ── PgnRegistry::standard ─────────────────────────────────────────────

    #[test]
    fn test_standard_registry_count_nonzero() {
        let reg = PgnRegistry::standard();
        assert!(reg.count() > 0);
    }

    #[test]
    fn test_standard_registry_has_eec1() {
        let reg = PgnRegistry::standard();
        assert!(reg.is_known(0xF004));
        let info = reg.lookup(0xF004).expect("EEC1 should be present");
        assert!(info.name.contains("EEC1") || info.name.contains("Engine"));
    }

    #[test]
    fn test_standard_registry_has_ebc1() {
        let reg = PgnRegistry::standard();
        assert!(reg.is_known(0xFEF1));
        let info = reg.lookup(0xFEF1).expect("EBC1 should be present");
        assert!(info.name.contains("EBC1") || info.name.contains("Brake"));
    }

    #[test]
    fn test_standard_registry_has_ebc2() {
        let reg = PgnRegistry::standard();
        assert!(reg.is_known(0xFEF2));
    }

    #[test]
    fn test_standard_registry_has_vehicle_speed() {
        let reg = PgnRegistry::standard();
        assert!(reg.is_known(0xFEEA));
        let info = reg.lookup(0xFEEA).expect("CCVS1 should be present");
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_standard_registry_has_fuel_economy() {
        let reg = PgnRegistry::standard();
        assert!(reg.is_known(0xFEEB));
    }

    // ── PgnRegistry::lookup ───────────────────────────────────────────────

    #[test]
    fn test_lookup_known_pgn_returns_info() {
        let reg = PgnRegistry::standard();
        let info = reg.lookup(0xFEF1).expect("should find EBC1");
        assert_eq!(info.pgn, 0xFEF1);
    }

    #[test]
    fn test_lookup_unknown_pgn_returns_none() {
        let reg = PgnRegistry::standard();
        assert!(reg.lookup(0xDEAD).is_none());
    }

    // ── PgnRegistry::is_known ─────────────────────────────────────────────

    #[test]
    fn test_is_known_returns_true_for_registered() {
        let reg = PgnRegistry::standard();
        assert!(reg.is_known(0xF004));
    }

    #[test]
    fn test_is_known_returns_false_for_unregistered() {
        let reg = PgnRegistry::standard();
        assert!(!reg.is_known(0xAAAA));
    }

    // ── PgnRegistry::register (custom PGN) ───────────────────────────────

    #[test]
    fn test_register_custom_pgn() {
        let mut reg = PgnRegistry::empty();
        reg.register(PgnInfo::new(
            0x1234,
            "Custom PGN",
            0,
            0x12,
            0x34,
            Some("50ms"),
        ));
        assert!(reg.is_known(0x1234));
        let info = reg.lookup(0x1234).expect("custom PGN should be found");
        assert_eq!(info.name, "Custom PGN");
        assert_eq!(info.transmission_rate.as_deref(), Some("50ms"));
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut reg = PgnRegistry::empty();
        reg.register(PgnInfo::new(0x100, "Original", 0, 0x01, 0x00, None::<&str>));
        reg.register(PgnInfo::new(0x100, "Updated", 0, 0x01, 0x00, None::<&str>));
        let info = reg.lookup(0x100).expect("should exist");
        assert_eq!(info.name, "Updated");
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_empty_registry_has_count_zero() {
        let reg = PgnRegistry::empty();
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_standard_registry_count_at_least_five() {
        let reg = PgnRegistry::standard();
        // Must include at least the five required PGNs
        assert!(reg.count() >= 5);
    }

    // ── PgnInfo::new is_pdu1 inference ────────────────────────────────────

    #[test]
    fn test_pgn_info_is_pdu1_inferred_for_pf_lt_240() {
        let info = PgnInfo::new(0xC000, "Test PDU1", 0, 0xC0, 0x00, None::<&str>);
        assert!(info.is_pdu1);
    }

    #[test]
    fn test_pgn_info_is_pdu2_inferred_for_pf_ge_240() {
        let info = PgnInfo::new(0xF004, "Test PDU2", 0, 0xF0, 0x04, Some("10ms"));
        assert!(!info.is_pdu1);
    }

    // ── Default impl ──────────────────────────────────────────────────────

    #[test]
    fn test_pgn_decoder_default() {
        let d = PgnDecoder;
        let id = d.encode_can_id(6, 0, 0xFE, 0xF1, 0x00);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.pf, 0xFE);
    }

    // ── 29-bit mask ───────────────────────────────────────────────────────

    #[test]
    fn test_decode_masks_to_29_bits() {
        let d = decoder();
        // Set bits 29 and 30 (should be masked away)
        let id_with_high_bits = 0x6000_0000_u32 | d.encode_can_id(3, 0, 0xFE, 0xF1, 0x00);
        let dec = d.decode_can_id(id_with_high_bits);
        assert_eq!(dec.priority, 3);
        assert_eq!(dec.pf, 0xFE);
    }

    // ── encode_can_id priority masking ────────────────────────────────────

    #[test]
    fn test_encode_masks_priority_to_3_bits() {
        let d = decoder();
        // Priority 0xFF should be masked to 7 (0x07)
        let id = d.encode_can_id(0xFF, 0, 0xFE, 0x00, 0x00);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.priority, 7);
    }

    #[test]
    fn test_encode_masks_data_page_to_1_bit() {
        let d = decoder();
        let id = d.encode_can_id(0, 0xFF, 0xFE, 0x00, 0x00);
        let dec = d.decode_can_id(id);
        assert_eq!(dec.data_page, 1);
    }

    // ── Priority levels 0-7 decode correctly ──────────────────────────────

    #[test]
    fn test_all_priority_levels() {
        let d = decoder();
        for p in 0_u8..=7 {
            let id = d.encode_can_id(p, 0, 0xFE, 0x00, 0x00);
            let dec = d.decode_can_id(id);
            assert_eq!(dec.priority, p, "priority {} mismatch", p);
        }
    }

    // ── DecodedId clone/equality ──────────────────────────────────────────

    #[test]
    fn test_decoded_id_equality() {
        let d = decoder();
        let id = d.encode_can_id(3, 0, 0xFE, 0xF1, 0x11);
        let dec1 = d.decode_can_id(id);
        let dec2 = d.decode_can_id(id);
        assert_eq!(dec1, dec2);
    }

    #[test]
    fn test_decoded_id_clone() {
        let d = decoder();
        let id = d.encode_can_id(3, 0, 0xFE, 0xF1, 0x11);
        let dec = d.decode_can_id(id);
        let cloned = dec.clone();
        assert_eq!(dec.pf, cloned.pf);
        assert_eq!(dec.sa, cloned.sa);
    }

    // ── PgnInfo transmission_rate ─────────────────────────────────────────

    #[test]
    fn test_pgn_info_transmission_rate_none() {
        let info = PgnInfo::new(0x100, "Test", 0, 0x01, 0x00, None::<&str>);
        assert!(info.transmission_rate.is_none());
    }

    #[test]
    fn test_pgn_info_transmission_rate_on_request() {
        let info = PgnInfo::new(0xFECA, "DM1", 0, 0xFE, 0xCA, Some("on-request"));
        assert_eq!(info.transmission_rate.as_deref(), Some("on-request"));
    }

    #[test]
    fn test_standard_registry_eec1_transmission_rate() {
        let reg = PgnRegistry::standard();
        let info = reg.lookup(0xF004).expect("EEC1 must be in registry");
        assert_eq!(info.transmission_rate.as_deref(), Some("10ms"));
    }

    // ── PgnRegistry count ─────────────────────────────────────────────────

    #[test]
    fn test_registry_count_increases_on_register() {
        let mut reg = PgnRegistry::empty();
        assert_eq!(reg.count(), 0);
        reg.register(PgnInfo::new(0x01, "A", 0, 0x00, 0x01, None::<&str>));
        assert_eq!(reg.count(), 1);
        reg.register(PgnInfo::new(0x02, "B", 0, 0x00, 0x02, None::<&str>));
        assert_eq!(reg.count(), 2);
    }

    // ── SA field across round-trips ───────────────────────────────────────

    #[test]
    fn test_all_sa_values() {
        let d = decoder();
        for sa in [0x00_u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
            let id = d.encode_can_id(0, 0, 0xFE, 0x00, sa);
            let dec = d.decode_can_id(id);
            assert_eq!(dec.sa, sa, "SA={:#04X} mismatch", sa);
        }
    }

    // ── pgn_from_decoded vs registry lookup pgn field ────────────────────

    #[test]
    fn test_pgn_from_decoded_matches_registry_pgn_for_ebc1() {
        let d = decoder();
        let reg = PgnRegistry::standard();
        // Simulate an EBC1 frame: SA=0x11, rest EBC1 parameters
        let id = d.encode_can_id(6, 0, 0xFE, 0xF1, 0x11);
        let dec = d.decode_can_id(id);
        let decoded_pgn = d.pgn_from_decoded(&dec);
        let registry_info = reg.lookup(decoded_pgn).expect("EBC1 must be in registry");
        assert_eq!(registry_info.pgn, decoded_pgn);
    }
}
