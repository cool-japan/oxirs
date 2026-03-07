//! # SAE J1939 Protocol Parser
//!
//! Parses and encodes SAE J1939 CAN frames using the standard 29-bit extended
//! CAN identifier format. Provides PGN lookup, peer-to-peer / broadcast
//! classification, and signal decoding helpers for common PGNs.
//!
//! ## Frame Format (29-bit CAN ID)
//!
//! ```text
//! Bits 28-26 : Priority         (3 bits)
//! Bit  25    : Reserved         (1 bit, always 0)
//! Bit  24    : Data Page         (1 bit)
//! Bits 23-16 : PF (PDU Format)  (8 bits) — determines PDU1 vs PDU2
//! Bits 15-8  : PS (PDU Specific)(8 bits) — destination (PDU1) or group ext (PDU2)
//! Bits  7-0  : Source Address   (8 bits)
//! ```
//!
//! PDU1 (PF < 0xF0): peer-to-peer; PS is destination address.
//! PDU2 (PF >= 0xF0): broadcast; PS is part of PGN.

/// SAE J1939 29-bit CAN identifier wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J1939Id {
    /// Raw 29-bit CAN ID.
    pub raw: u32,
}

/// Classification of a PGN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgnType {
    /// Peer-to-peer message (PF < 0xF0). Destination address is explicit.
    PeerToPeer,
    /// Broadcast message (PF >= 0xF0). No specific destination.
    Broadcast,
}

/// Static information about a known PGN.
#[derive(Debug, Clone, Copy)]
pub struct PgnInfo {
    pub pgn: u32,
    pub name: &'static str,
    pub pgn_type: PgnType,
    /// Typical data length in bytes (0 = variable).
    pub data_length: u8,
}

/// Decoded J1939 CAN frame.
#[derive(Debug, Clone)]
pub struct J1939Frame {
    /// Message priority (0 = highest, 7 = lowest).
    pub priority: u8,
    /// Parameter Group Number.
    pub pgn: u32,
    /// Source address (ECU address).
    pub source_address: u8,
    /// Destination address (Some for PDU1, None for PDU2/broadcast).
    pub destination_address: Option<u8>,
    /// Frame payload bytes.
    pub data: Vec<u8>,
}

// ─────────────────────────────────────────────
// Known PGN table
// ─────────────────────────────────────────────

static KNOWN_PGNS: &[PgnInfo] = &[
    PgnInfo {
        pgn: 0xFECA,
        name: "DM1 - Diagnostic Message 1",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
    PgnInfo {
        pgn: 0xFEEB,
        name: "IC1 - Inlet/Exhaust Conditions 1",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
    PgnInfo {
        pgn: 0xFEF1,
        name: "CCVS1 - Cruise Control/Vehicle Speed 1",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
    PgnInfo {
        pgn: 0xFEF2,
        name: "LFE1 - Fuel Economy - Liquid",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
    PgnInfo {
        pgn: 0xFEE0,
        name: "HOURS - Engine Hours, Revolutions",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
    PgnInfo {
        pgn: 0xFEE5,
        name: "ET1 - Engine Temperature 1",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
    PgnInfo {
        pgn: 0xFEEF,
        name: "EEC1 - Electronic Engine Controller 1",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
    PgnInfo {
        pgn: 0xFEF0,
        name: "EEC2 - Electronic Engine Controller 2",
        pgn_type: PgnType::Broadcast,
        data_length: 8,
    },
];

// ─────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────

/// SAE J1939 protocol parser.
#[derive(Debug, Default)]
pub struct J1939Parser;

impl J1939Parser {
    /// Create a new parser instance.
    pub fn new() -> Self {
        Self
    }

    /// Parse a raw 29-bit CAN ID into a [`J1939Id`].
    pub fn parse_id(&self, can_id: u32) -> J1939Id {
        J1939Id {
            raw: can_id & 0x1FFF_FFFF,
        }
    }

    /// Decode a complete J1939 frame from a raw CAN ID and payload.
    pub fn decode_frame(&self, can_id: u32, data: &[u8]) -> J1939Frame {
        let priority = Self::extract_priority(can_id);
        let pgn = Self::extract_pgn(can_id);
        let source_address = Self::extract_source_address(can_id);
        let destination_address = Self::extract_destination(can_id);
        J1939Frame {
            priority,
            pgn,
            source_address,
            destination_address,
            data: data.to_vec(),
        }
    }

    /// Extract the 3-bit priority from a CAN ID (bits 28–26).
    pub fn extract_priority(can_id: u32) -> u8 {
        ((can_id >> 26) & 0x07) as u8
    }

    /// Extract the PGN from a 29-bit CAN ID.
    ///
    /// For PDU1 (PF < 0xF0): PGN = (R<<17) | (DP<<16) | (PF<<8)
    ///   — PS is destination address, not part of PGN.
    /// For PDU2 (PF >= 0xF0): PGN = (R<<17) | (DP<<16) | (PF<<8) | PS
    pub fn extract_pgn(can_id: u32) -> u32 {
        // R  = bit 25
        let r = (can_id >> 25) & 0x01;
        // DP = bit 24
        let dp = (can_id >> 24) & 0x01;
        // PF = bits 23–16
        let pf = (can_id >> 16) & 0xFF;
        // PS = bits 15–8
        let ps = (can_id >> 8) & 0xFF;

        if pf < 0xF0 {
            // PDU1 — PS is destination, not in PGN
            (r << 17) | (dp << 16) | (pf << 8)
        } else {
            // PDU2 — PS is group extension, included in PGN
            (r << 17) | (dp << 16) | (pf << 8) | ps
        }
    }

    /// Extract the 8-bit source address (bits 7–0).
    pub fn extract_source_address(can_id: u32) -> u8 {
        (can_id & 0xFF) as u8
    }

    /// Extract the destination address for PDU1 messages (PF < 0xF0).
    ///
    /// Returns `None` for broadcast (PDU2) messages.
    pub fn extract_destination(can_id: u32) -> Option<u8> {
        let pf = (can_id >> 16) & 0xFF;
        if pf < 0xF0 {
            let ps = (can_id >> 8) & 0xFF;
            Some(ps as u8)
        } else {
            None
        }
    }

    /// Reconstruct the 29-bit CAN ID from a decoded frame.
    pub fn encode_frame(frame: &J1939Frame) -> u32 {
        let priority = (frame.priority as u32 & 0x07) << 26;
        let pf = (frame.pgn >> 8) & 0xFF;
        let dp = (frame.pgn >> 16) & 0x01;
        let r = (frame.pgn >> 17) & 0x01;

        if pf < 0xF0 {
            // PDU1: PS is destination address
            let dest = frame.destination_address.unwrap_or(0xFF) as u32;
            priority
                | (r << 25)
                | (dp << 24)
                | (pf << 16)
                | (dest << 8)
                | frame.source_address as u32
        } else {
            // PDU2: PS is group extension, included in PGN
            let ps = frame.pgn & 0xFF;
            priority | (r << 25) | (dp << 24) | (pf << 16) | (ps << 8) | frame.source_address as u32
        }
    }

    /// Look up static information for a known PGN.
    pub fn pgn_info(pgn: u32) -> Option<PgnInfo> {
        KNOWN_PGNS.iter().find(|p| p.pgn == pgn).copied()
    }

    /// Return `true` if the PGN represents a peer-to-peer message.
    ///
    /// A PGN with PF (bits 15–8) < 0xF0 is peer-to-peer.
    pub fn is_peer_to_peer(pgn: u32) -> bool {
        let pf = (pgn >> 8) & 0xFF;
        pf < 0xF0
    }

    /// Decode vehicle speed from a CCVS1 frame.
    ///
    /// Bytes 1–2 (little-endian), resolution 1/256 km/h.
    /// Returns `None` if data is too short or value is the error indicator (0xFFFF).
    pub fn decode_vehicle_speed(data: &[u8]) -> Option<f64> {
        if data.len() < 3 {
            return None;
        }
        let raw = u16::from_le_bytes([data[1], data[2]]);
        if raw == 0xFFFF {
            return None;
        }
        Some(raw as f64 / 256.0)
    }

    /// Decode engine speed from an EEC1 frame.
    ///
    /// Bytes 3–4 (little-endian), resolution 0.125 rpm.
    /// Returns `None` if data is too short or value is the error indicator (0xFFFF).
    pub fn decode_engine_speed(data: &[u8]) -> Option<f64> {
        if data.len() < 5 {
            return None;
        }
        let raw = u16::from_le_bytes([data[3], data[4]]);
        if raw == 0xFFFF {
            return None;
        }
        Some(raw as f64 * 0.125)
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a 29-bit CAN ID
    // priority(3) | R(1) | DP(1) | PF(8) | PS(8) | SA(8)
    fn build_id(priority: u8, r: u8, dp: u8, pf: u8, ps: u8, sa: u8) -> u32 {
        ((priority as u32 & 0x07) << 26)
            | ((r as u32 & 0x01) << 25)
            | ((dp as u32 & 0x01) << 24)
            | ((pf as u32) << 16)
            | ((ps as u32) << 8)
            | (sa as u32)
    }

    // ── J1939Parser::new ───────────────────────────────────────────────

    #[test]
    fn test_new() {
        let _parser = J1939Parser::new();
    }

    // ── extract_priority ──────────────────────────────────────────────

    #[test]
    fn test_extract_priority_3() {
        let id = build_id(3, 0, 0, 0xFE, 0xCA, 0x00);
        assert_eq!(J1939Parser::extract_priority(id), 3);
    }

    #[test]
    fn test_extract_priority_6() {
        let id = build_id(6, 0, 0, 0xFE, 0xCA, 0x11);
        assert_eq!(J1939Parser::extract_priority(id), 6);
    }

    #[test]
    fn test_extract_priority_zero() {
        let id = build_id(0, 0, 0, 0xFE, 0xCA, 0x00);
        assert_eq!(J1939Parser::extract_priority(id), 0);
    }

    // ── extract_source_address ────────────────────────────────────────

    #[test]
    fn test_extract_source_address() {
        let id = build_id(3, 0, 0, 0xFE, 0xEF, 0x11);
        assert_eq!(J1939Parser::extract_source_address(id), 0x11);
    }

    #[test]
    fn test_extract_source_address_zero() {
        let id = build_id(3, 0, 0, 0xFE, 0xEF, 0x00);
        assert_eq!(J1939Parser::extract_source_address(id), 0x00);
    }

    // ── extract_pgn PDU2 (broadcast) ─────────────────────────────────

    #[test]
    fn test_extract_pgn_eec1() {
        // EEC1 PGN = 0xFEEF, PF=0xFE ≥ 0xF0, PS=0xEF → PGN includes PS
        let id = build_id(3, 0, 0, 0xFE, 0xEF, 0x11);
        let pgn = J1939Parser::extract_pgn(id);
        assert_eq!(pgn, 0xFEEF, "PGN mismatch: {pgn:#X}");
    }

    #[test]
    fn test_extract_pgn_ccvs1() {
        let id = build_id(6, 0, 0, 0xFE, 0xF1, 0x00);
        let pgn = J1939Parser::extract_pgn(id);
        assert_eq!(pgn, 0xFEF1);
    }

    // ── extract_pgn PDU1 (peer-to-peer) ──────────────────────────────

    #[test]
    fn test_extract_pgn_pdu1_excludes_ps() {
        // PF = 0xEC (< 0xF0) → PDU1; PS is destination, not in PGN
        let id = build_id(6, 0, 0, 0xEC, 0x42, 0x00);
        let pgn = J1939Parser::extract_pgn(id);
        // PGN = PF<<8 = 0xEC00
        assert_eq!(pgn, 0xEC00);
    }

    // ── extract_destination ───────────────────────────────────────────

    #[test]
    fn test_destination_pdu1() {
        let id = build_id(6, 0, 0, 0xEC, 0x42, 0x00);
        assert_eq!(J1939Parser::extract_destination(id), Some(0x42));
    }

    #[test]
    fn test_destination_pdu2_none() {
        let id = build_id(3, 0, 0, 0xFE, 0xEF, 0x11);
        assert_eq!(J1939Parser::extract_destination(id), None);
    }

    // ── decode_frame ──────────────────────────────────────────────────

    #[test]
    fn test_decode_frame_eec1() {
        let parser = J1939Parser::new();
        let id = build_id(3, 0, 0, 0xFE, 0xEF, 0x11);
        let data = vec![0u8; 8];
        let frame = parser.decode_frame(id, &data);
        assert_eq!(frame.priority, 3);
        assert_eq!(frame.pgn, 0xFEEF);
        assert_eq!(frame.source_address, 0x11);
        assert_eq!(frame.destination_address, None);
        assert_eq!(frame.data.len(), 8);
    }

    #[test]
    fn test_decode_frame_pdu1_has_destination() {
        let parser = J1939Parser::new();
        let id = build_id(6, 0, 0, 0xEC, 0x42, 0x05);
        let frame = parser.decode_frame(id, &[]);
        assert_eq!(frame.destination_address, Some(0x42));
    }

    // ── encode_frame ──────────────────────────────────────────────────

    #[test]
    fn test_encode_frame_roundtrip_pdu2() {
        let parser = J1939Parser::new();
        let id_orig = build_id(3, 0, 0, 0xFE, 0xEF, 0x11);
        let data = vec![0u8; 8];
        let frame = parser.decode_frame(id_orig, &data);
        let id_re = J1939Parser::encode_frame(&frame);
        // Mask to 29 bits
        assert_eq!(id_re & 0x1FFF_FFFF, id_orig & 0x1FFF_FFFF);
    }

    #[test]
    fn test_encode_frame_roundtrip_pdu1() {
        let parser = J1939Parser::new();
        let id_orig = build_id(6, 0, 0, 0xEC, 0x42, 0x05);
        let frame = parser.decode_frame(id_orig, &[]);
        let id_re = J1939Parser::encode_frame(&frame);
        assert_eq!(id_re & 0x1FFF_FFFF, id_orig & 0x1FFF_FFFF);
    }

    // ── pgn_info ──────────────────────────────────────────────────────

    #[test]
    fn test_pgn_info_eec1() {
        let info = J1939Parser::pgn_info(0xFEEF).expect("EEC1 should be known");
        assert_eq!(info.pgn, 0xFEEF);
        assert!(info.name.contains("EEC1"));
    }

    #[test]
    fn test_pgn_info_ccvs1() {
        let info = J1939Parser::pgn_info(0xFEF1).expect("CCVS1 should be known");
        assert_eq!(info.pgn, 0xFEF1);
    }

    #[test]
    fn test_pgn_info_dm1() {
        let info = J1939Parser::pgn_info(0xFECA).expect("DM1 should be known");
        assert!(info.name.contains("DM1"));
    }

    #[test]
    fn test_pgn_info_lfe1() {
        let info = J1939Parser::pgn_info(0xFEF2).expect("LFE1 should be known");
        assert!(info.name.contains("LFE1"));
    }

    #[test]
    fn test_pgn_info_hours() {
        let info = J1939Parser::pgn_info(0xFEE0).expect("HOURS should be known");
        assert!(info.name.to_lowercase().contains("hour"));
    }

    #[test]
    fn test_pgn_info_et1() {
        let info = J1939Parser::pgn_info(0xFEE5).expect("ET1 should be known");
        assert!(info.name.contains("ET1"));
    }

    #[test]
    fn test_pgn_info_eec2() {
        let info = J1939Parser::pgn_info(0xFEF0).expect("EEC2 should be known");
        assert!(info.name.contains("EEC2"));
    }

    #[test]
    fn test_pgn_info_ic1() {
        let info = J1939Parser::pgn_info(0xFEEB).expect("IC1 should be known");
        assert!(info.name.contains("IC1"));
    }

    #[test]
    fn test_pgn_info_unknown() {
        assert!(J1939Parser::pgn_info(0x0001).is_none());
    }

    // ── is_peer_to_peer ───────────────────────────────────────────────

    #[test]
    fn test_is_peer_to_peer_pdu1() {
        // PF = 0xEC < 0xF0 → peer-to-peer
        assert!(J1939Parser::is_peer_to_peer(0xEC00));
    }

    #[test]
    fn test_is_peer_to_peer_pdu2_false() {
        // PGN 0xFEEF → PF = 0xFE >= 0xF0 → broadcast
        assert!(!J1939Parser::is_peer_to_peer(0xFEEF));
    }

    // ── decode_vehicle_speed ──────────────────────────────────────────

    #[test]
    fn test_decode_vehicle_speed_typical() {
        // 80 km/h = 80.0 * 256 = 20480 = 0x5000
        let raw: u16 = 20480;
        let data = vec![0u8, raw as u8, (raw >> 8) as u8, 0, 0, 0, 0, 0];
        let speed = J1939Parser::decode_vehicle_speed(&data).expect("should decode");
        assert!((speed - 80.0).abs() < 0.01, "speed={speed}");
    }

    #[test]
    fn test_decode_vehicle_speed_zero() {
        let data = vec![0u8; 8];
        let speed = J1939Parser::decode_vehicle_speed(&data).expect("should decode");
        assert!((speed).abs() < 0.01);
    }

    #[test]
    fn test_decode_vehicle_speed_error_indicator() {
        let data = vec![0, 0xFF, 0xFF, 0, 0, 0, 0, 0];
        assert!(J1939Parser::decode_vehicle_speed(&data).is_none());
    }

    #[test]
    fn test_decode_vehicle_speed_too_short() {
        assert!(J1939Parser::decode_vehicle_speed(&[0u8, 0xFF]).is_none());
    }

    // ── decode_engine_speed ───────────────────────────────────────────

    #[test]
    fn test_decode_engine_speed_typical() {
        // 2000 rpm = 2000 / 0.125 = 16000 = 0x3E80
        let raw: u16 = 16000;
        let data = vec![0u8, 0, 0, raw as u8, (raw >> 8) as u8, 0, 0, 0];
        let rpm = J1939Parser::decode_engine_speed(&data).expect("should decode");
        assert!((rpm - 2000.0).abs() < 0.01, "rpm={rpm}");
    }

    #[test]
    fn test_decode_engine_speed_zero() {
        let data = vec![0u8; 8];
        let rpm = J1939Parser::decode_engine_speed(&data).expect("should decode");
        assert!((rpm).abs() < 0.01);
    }

    #[test]
    fn test_decode_engine_speed_error_indicator() {
        let data = vec![0, 0, 0, 0xFF, 0xFF, 0, 0, 0];
        assert!(J1939Parser::decode_engine_speed(&data).is_none());
    }

    #[test]
    fn test_decode_engine_speed_too_short() {
        assert!(J1939Parser::decode_engine_speed(&[0u8; 4]).is_none());
    }

    // ── parse_id ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_id_masks_to_29bits() {
        let parser = J1939Parser::new();
        let raw_id = 0xFFFF_FFFF;
        let j_id = parser.parse_id(raw_id);
        assert_eq!(j_id.raw, 0x1FFF_FFFF);
    }

    // ── all known PGNs are broadcast ─────────────────────────────────

    #[test]
    fn test_known_pgns_all_broadcast() {
        for info in KNOWN_PGNS {
            assert_eq!(
                info.pgn_type,
                PgnType::Broadcast,
                "PGN {:#X} should be broadcast",
                info.pgn
            );
        }
    }
}
