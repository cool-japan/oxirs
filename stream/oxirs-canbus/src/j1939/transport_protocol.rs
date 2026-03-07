//! J1939-21 Transport Protocol (TP) – Enhanced implementation
//!
//! The J1939 Transport Protocol enables transmission of messages larger than
//! 8 bytes by fragmenting them across multiple CAN frames. This module
//! provides full TP.CM (Connection Management) and TP.DT (Data Transfer)
//! support including:
//!
//! - **BAM** (Broadcast Announce Message) – one-to-all broadcast
//! - **RTS/CTS** (Request To Send / Clear To Send) – peer-to-peer connection
//! - **EOM** (End Of Message Acknowledgment) – transfer completion
//! - **Abort** – session termination with reason codes
//! - **Timeout** handling with configurable duration
//! - **Reassembler** tracking sessions by `(src, dst, pgn)` triplet
//!
//! # Protocol Overview
//!
//! ```text
//! Sender                           Receiver
//!   │                                │
//!   │──── TP.CM_RTS ────────────────>│  "I want to send N packets"
//!   │<─── TP.CM_CTS ─────────────────│  "Send M packets starting at seq K"
//!   │──── TP.DT[1..M] ──────────────>│
//!   │<─── TP.CM_CTS ─────────────────│  (repeat until all sent)
//!   │──── TP.DT[M+1..N] ────────────>│
//!   │<─── TP.CM_EOM_ACK ─────────────│  "All done"
//! ```
//!
//! # BAM (broadcast)
//!
//! ```text
//! Sender ──── TP.CM_BAM ────────────> *  "I'll broadcast N packets"
//!        ──── TP.DT[1] ──────────────> *
//!        ──── TP.DT[2] ──────────────> *
//!        ...
//!        ──── TP.DT[N] ──────────────> *
//! ```
//!
//! # Example
//!
//! ```rust
//! use std::time::Duration;
//! use oxirs_canbus::j1939::transport_protocol::{TpReassembler, TpControlMessage, TpDataTransfer};
//!
//! let mut reassembler = TpReassembler::new(Duration::from_secs(5));
//!
//! // Simulate a BAM session: 9 bytes in 2 packets, PGN 61444
//! let bam = TpControlMessage::Bam {
//!     total_message_size: 9,
//!     total_packets: 2,
//!     pgn: 61444,
//! };
//! reassembler.handle_cm(0x00, 0xFF, bam).expect("BAM should be accepted");
//!
//! let dt1 = TpDataTransfer { sequence_number: 1, data: [0xAA; 7] };
//! assert!(reassembler.handle_dt(0x00, 0xFF, dt1).expect("DT should be ok").is_none());
//!
//! let dt2 = TpDataTransfer { sequence_number: 2, data: [0xBB; 7] };
//! let result = reassembler.handle_dt(0x00, 0xFF, dt2).expect("DT should be ok");
//! assert!(result.is_some());
//! let (pgn, payload) = result.unwrap();
//! assert_eq!(pgn, 61444);
//! assert_eq!(payload.len(), 9);
//! ```

use crate::error::CanbusError;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// PGN Constants
// ============================================================================

/// TP.CM PGN (0xEC00 = 60416)
pub const TP_CM_PGN: u32 = 0xEC00;
/// TP.DT PGN (0xEB00 = 60160)
pub const TP_DT_PGN: u32 = 0xEB00;

// ============================================================================
// Abort Reason
// ============================================================================

/// Abort reason codes per SAE J1939-21 Table 5
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortReason {
    /// Already in one or more connection managed sessions and cannot support another
    AlreadyInProgress,
    /// System resources needed for another task so a connection managed session cannot be supported
    SystemResourcesNeeded,
    /// A timeout occurred
    Timeout,
    /// CTS message was received when data transfer was in progress
    CtsWhileTransferring,
    /// Maximum retransmit request limit reached
    MaxRetransmit,
    /// Unexpected data transfer packet
    UnexpectedDataPacket,
    /// Bad sequence number (and therefore a bad message)
    BadSequenceNumber,
    /// Duplicate sequence number (and therefore a bad message)
    DuplicateSequenceNumber,
    /// Message is too large (message size > 1785 bytes)
    MessageTooLarge,
    /// Unknown or manufacturer-specific abort
    Unknown(u8),
}

impl AbortReason {
    /// Decode from raw byte.
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => AbortReason::AlreadyInProgress,
            2 => AbortReason::SystemResourcesNeeded,
            3 => AbortReason::Timeout,
            4 => AbortReason::CtsWhileTransferring,
            5 => AbortReason::MaxRetransmit,
            6 => AbortReason::UnexpectedDataPacket,
            7 => AbortReason::BadSequenceNumber,
            8 => AbortReason::DuplicateSequenceNumber,
            9 => AbortReason::MessageTooLarge,
            other => AbortReason::Unknown(other),
        }
    }

    /// Return the raw byte value.
    pub fn as_raw(self) -> u8 {
        match self {
            AbortReason::AlreadyInProgress => 1,
            AbortReason::SystemResourcesNeeded => 2,
            AbortReason::Timeout => 3,
            AbortReason::CtsWhileTransferring => 4,
            AbortReason::MaxRetransmit => 5,
            AbortReason::UnexpectedDataPacket => 6,
            AbortReason::BadSequenceNumber => 7,
            AbortReason::DuplicateSequenceNumber => 8,
            AbortReason::MessageTooLarge => 9,
            AbortReason::Unknown(v) => v,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            AbortReason::AlreadyInProgress => "Already in connection managed session",
            AbortReason::SystemResourcesNeeded => "System resources needed for another task",
            AbortReason::Timeout => "Timeout occurred",
            AbortReason::CtsWhileTransferring => "CTS received during data transfer",
            AbortReason::MaxRetransmit => "Maximum retransmit request limit reached",
            AbortReason::UnexpectedDataPacket => "Unexpected data transfer packet",
            AbortReason::BadSequenceNumber => "Bad sequence number",
            AbortReason::DuplicateSequenceNumber => "Duplicate sequence number",
            AbortReason::MessageTooLarge => "Message too large (>1785 bytes)",
            AbortReason::Unknown(_) => "Unknown abort reason",
        }
    }
}

impl std::fmt::Display for AbortReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code {})", self.description(), self.as_raw())
    }
}

// ============================================================================
// TP Control Message
// ============================================================================

/// J1939 Transport Protocol Connection Management message
#[derive(Debug, Clone)]
pub enum TpControlMessage {
    /// Request To Send (byte 0 = 16) – peer-to-peer session initiation
    Rts {
        /// Total message size in bytes (2 bytes LE)
        total_message_size: u16,
        /// Total number of packets
        total_packets: u8,
        /// Maximum number of packets per CTS response
        max_packets_per_cts: u8,
        /// PGN being transferred (3 bytes)
        pgn: u32,
    },
    /// Clear To Send (byte 0 = 17) – receiver indicates readiness
    Cts {
        /// Number of TP.DT packets the receiver can accept
        packets_to_send: u8,
        /// Next sequence number to send
        next_packet: u8,
        /// PGN being transferred
        pgn: u32,
    },
    /// End of Message Acknowledgment (byte 0 = 19)
    Eom {
        /// Total message size (should match RTS)
        total_message_size: u16,
        /// Total packets received
        total_packets: u8,
        /// PGN acknowledged
        pgn: u32,
    },
    /// Connection Abort (byte 0 = 255)
    Abort {
        /// Reason for aborting
        abort_reason: AbortReason,
        /// PGN of aborted session
        pgn: u32,
    },
    /// Broadcast Announce Message (byte 0 = 32) – broadcast without handshake
    Bam {
        /// Total message size in bytes
        total_message_size: u16,
        /// Total number of packets
        total_packets: u8,
        /// PGN being broadcast
        pgn: u32,
    },
}

impl TpControlMessage {
    /// Control byte values per J1939-21
    pub const CTRL_RTS: u8 = 16;
    pub const CTRL_CTS: u8 = 17;
    pub const CTRL_EOM: u8 = 19;
    pub const CTRL_BAM: u8 = 32;
    pub const CTRL_ABORT: u8 = 255;

    /// Encode to 8 bytes.
    pub fn encode(&self) -> [u8; 8] {
        match self {
            TpControlMessage::Rts {
                total_message_size,
                total_packets,
                max_packets_per_cts,
                pgn,
            } => {
                let sz = total_message_size.to_le_bytes();
                let pgn_b = pgn.to_le_bytes();
                [
                    Self::CTRL_RTS,
                    sz[0],
                    sz[1],
                    *total_packets,
                    *max_packets_per_cts,
                    pgn_b[0],
                    pgn_b[1],
                    pgn_b[2],
                ]
            }
            TpControlMessage::Cts {
                packets_to_send,
                next_packet,
                pgn,
            } => {
                let pgn_b = pgn.to_le_bytes();
                [
                    Self::CTRL_CTS,
                    *packets_to_send,
                    *next_packet,
                    0xFF,
                    0xFF,
                    pgn_b[0],
                    pgn_b[1],
                    pgn_b[2],
                ]
            }
            TpControlMessage::Eom {
                total_message_size,
                total_packets,
                pgn,
            } => {
                let sz = total_message_size.to_le_bytes();
                let pgn_b = pgn.to_le_bytes();
                [
                    Self::CTRL_EOM,
                    sz[0],
                    sz[1],
                    *total_packets,
                    0xFF,
                    pgn_b[0],
                    pgn_b[1],
                    pgn_b[2],
                ]
            }
            TpControlMessage::Abort { abort_reason, pgn } => {
                let pgn_b = pgn.to_le_bytes();
                [
                    Self::CTRL_ABORT,
                    abort_reason.as_raw(),
                    0xFF,
                    0xFF,
                    0xFF,
                    pgn_b[0],
                    pgn_b[1],
                    pgn_b[2],
                ]
            }
            TpControlMessage::Bam {
                total_message_size,
                total_packets,
                pgn,
            } => {
                let sz = total_message_size.to_le_bytes();
                let pgn_b = pgn.to_le_bytes();
                [
                    Self::CTRL_BAM,
                    sz[0],
                    sz[1],
                    *total_packets,
                    0xFF,
                    pgn_b[0],
                    pgn_b[1],
                    pgn_b[2],
                ]
            }
        }
    }

    /// Decode from 8 bytes.
    pub fn decode(data: &[u8]) -> Result<Self, CanbusError> {
        if data.len() < 8 {
            return Err(CanbusError::Config(format!(
                "TP.CM message needs 8 bytes, got {}",
                data.len()
            )));
        }
        let ctrl = data[0];
        let pgn = u32::from_le_bytes([data[5], data[6], data[7], 0]);

        match ctrl {
            Self::CTRL_RTS => Ok(TpControlMessage::Rts {
                total_message_size: u16::from_le_bytes([data[1], data[2]]),
                total_packets: data[3],
                max_packets_per_cts: data[4],
                pgn,
            }),
            Self::CTRL_CTS => Ok(TpControlMessage::Cts {
                packets_to_send: data[1],
                next_packet: data[2],
                pgn,
            }),
            Self::CTRL_EOM => Ok(TpControlMessage::Eom {
                total_message_size: u16::from_le_bytes([data[1], data[2]]),
                total_packets: data[3],
                pgn,
            }),
            Self::CTRL_ABORT => Ok(TpControlMessage::Abort {
                abort_reason: AbortReason::from_raw(data[1]),
                pgn,
            }),
            Self::CTRL_BAM => Ok(TpControlMessage::Bam {
                total_message_size: u16::from_le_bytes([data[1], data[2]]),
                total_packets: data[3],
                pgn,
            }),
            other => Err(CanbusError::Config(format!(
                "Unknown TP.CM control byte: {}",
                other
            ))),
        }
    }

    /// Return the PGN embedded in this message.
    pub fn pgn(&self) -> u32 {
        match self {
            TpControlMessage::Rts { pgn, .. } => *pgn,
            TpControlMessage::Cts { pgn, .. } => *pgn,
            TpControlMessage::Eom { pgn, .. } => *pgn,
            TpControlMessage::Abort { pgn, .. } => *pgn,
            TpControlMessage::Bam { pgn, .. } => *pgn,
        }
    }
}

// ============================================================================
// TP Data Transfer
// ============================================================================

/// J1939 Transport Protocol Data Transfer packet (TP.DT)
#[derive(Debug, Clone)]
pub struct TpDataTransfer {
    /// Sequence number (1–255, wraps at 255 back to 1)
    pub sequence_number: u8,
    /// 7 bytes of payload (unused bytes filled with 0xFF)
    pub data: [u8; 7],
}

impl TpDataTransfer {
    /// Encode to 8 bytes (seq + 7 data bytes).
    pub fn encode(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[0] = self.sequence_number;
        out[1..].copy_from_slice(&self.data);
        out
    }

    /// Decode from 8 bytes.
    pub fn decode(data: &[u8]) -> Result<Self, CanbusError> {
        if data.len() < 8 {
            return Err(CanbusError::Config(format!(
                "TP.DT message needs 8 bytes, got {}",
                data.len()
            )));
        }
        let mut payload = [0xFFu8; 7];
        payload.copy_from_slice(&data[1..8]);
        Ok(Self {
            sequence_number: data[0],
            data: payload,
        })
    }
}

// ============================================================================
// Internal Session State
// ============================================================================

/// Internal state of an in-progress TP session
#[derive(Debug)]
struct TpSession {
    /// PGN being transferred
    pgn: u32,
    /// Expected total message size in bytes
    total_size: usize,
    /// Total expected number of packets
    total_packets: u8,
    /// Number of TP.DT packets received so far
    packets_received: u8,
    /// Packet store: seq_number (1-based) -> 7 payload bytes
    data: HashMap<u8, [u8; 7]>,
    /// Session type (BAM or RTS/CTS; retained for protocol validation)
    #[allow(dead_code)]
    session_type: SessionType,
    /// Time the session was initiated
    started_at: Instant,
    /// Last activity timestamp
    last_activity: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionType {
    Bam,
    RtsCts,
}

impl TpSession {
    fn new_bam(pgn: u32, total_size: u16, total_packets: u8) -> Self {
        let now = Instant::now();
        Self {
            pgn,
            total_size: total_size as usize,
            total_packets,
            packets_received: 0,
            data: HashMap::new(),
            session_type: SessionType::Bam,
            started_at: now,
            last_activity: now,
        }
    }

    fn new_rts(pgn: u32, total_size: u16, total_packets: u8) -> Self {
        let now = Instant::now();
        Self {
            pgn,
            total_size: total_size as usize,
            total_packets,
            packets_received: 0,
            data: HashMap::new(),
            session_type: SessionType::RtsCts,
            started_at: now,
            last_activity: now,
        }
    }

    fn is_complete(&self) -> bool {
        self.packets_received >= self.total_packets
    }

    /// Reassemble the ordered packet data into a single `Vec<u8>`.
    fn reassemble(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.total_size);
        for seq in 1..=self.total_packets {
            if let Some(pkt) = self.data.get(&seq) {
                out.extend_from_slice(pkt);
            }
        }
        out.truncate(self.total_size);
        out
    }
}

// ============================================================================
// TpReassembler
// ============================================================================

/// Session key: (source_address, destination_address, pgn)
type SessionKey = (u8, u8, u32);

/// J1939-21 Transport Protocol reassembler
///
/// Maintains per-session state and reassembles multi-packet messages.
/// Sessions are identified by `(src, dst, pgn)`.
///
/// Call [`TpReassembler::handle_cm`] with decoded `TpControlMessage` values
/// and [`TpReassembler::handle_dt`] with decoded `TpDataTransfer` values.
/// When the last packet of a session arrives, `handle_dt` returns
/// `Some((pgn, reassembled_data))`.
///
/// Periodically call [`TpReassembler::evict_expired`] to clean up stale sessions.
pub struct TpReassembler {
    sessions: HashMap<SessionKey, TpSession>,
    timeout: Duration,
}

impl TpReassembler {
    /// Create a new reassembler with the given session timeout.
    pub fn new(timeout: Duration) -> Self {
        Self {
            sessions: HashMap::new(),
            timeout,
        }
    }

    /// Handle a TP.CM (Connection Management) message.
    ///
    /// - `src`  – J1939 source address
    /// - `dst`  – J1939 destination address (0xFF for BAM)
    /// - `msg`  – decoded control message
    ///
    /// Returns `Err` for protocol violations.
    pub fn handle_cm(
        &mut self,
        src: u8,
        dst: u8,
        msg: TpControlMessage,
    ) -> Result<(), CanbusError> {
        match &msg {
            TpControlMessage::Bam {
                total_message_size,
                total_packets,
                pgn,
            } => {
                if *total_packets == 0 {
                    return Err(CanbusError::Config(
                        "BAM with 0 packets is invalid".to_string(),
                    ));
                }
                if *total_message_size == 0 {
                    return Err(CanbusError::Config(
                        "BAM with 0 message size is invalid".to_string(),
                    ));
                }
                let key = (src, 0xFF, *pgn);
                self.sessions.insert(
                    key,
                    TpSession::new_bam(*pgn, *total_message_size, *total_packets),
                );
                Ok(())
            }

            TpControlMessage::Rts {
                total_message_size,
                total_packets,
                pgn,
                ..
            } => {
                if *total_packets == 0 {
                    return Err(CanbusError::Config(
                        "RTS with 0 packets is invalid".to_string(),
                    ));
                }
                let key = (src, dst, *pgn);
                self.sessions.insert(
                    key,
                    TpSession::new_rts(*pgn, *total_message_size, *total_packets),
                );
                Ok(())
            }

            TpControlMessage::Abort { pgn, .. } => {
                // Remove session on abort
                let key = (src, dst, *pgn);
                self.sessions.remove(&key);
                Ok(())
            }

            TpControlMessage::Eom { pgn, .. } => {
                // Sender-side EOM – receiver has already assembled; no action needed
                let key = (src, dst, *pgn);
                self.sessions.remove(&key);
                Ok(())
            }

            TpControlMessage::Cts { .. } => {
                // CTS is a receiver->sender message; ignore in reassembler
                Ok(())
            }
        }
    }

    /// Handle a TP.DT (Data Transfer) packet.
    ///
    /// Returns `Ok(Some((pgn, data)))` when all packets have been received and
    /// the message is fully reassembled.
    /// Returns `Ok(None)` when more packets are still expected.
    /// Returns `Err` for protocol violations (unexpected packet, bad seq, etc.).
    pub fn handle_dt(
        &mut self,
        src: u8,
        dst: u8,
        dt: TpDataTransfer,
    ) -> Result<Option<(u32, Vec<u8>)>, CanbusError> {
        // Try (src, dst, *) and (src, 0xFF, *) to handle BAM/RTS
        let key = self
            .find_session_key(src, dst)
            .ok_or_else(|| CanbusError::Config(format!("No TP session for src={src} dst={dst}")))?;

        let session = self
            .sessions
            .get_mut(&key)
            .ok_or_else(|| CanbusError::Config("Session disappeared unexpectedly".to_string()))?;

        // Validate sequence number
        if dt.sequence_number == 0 || dt.sequence_number > session.total_packets {
            return Err(CanbusError::Config(format!(
                "Bad TP.DT sequence number: {} (expected 1..={})",
                dt.sequence_number, session.total_packets
            )));
        }

        if session.data.contains_key(&dt.sequence_number) {
            return Err(CanbusError::Config(format!(
                "Duplicate TP.DT sequence number: {}",
                dt.sequence_number
            )));
        }

        session.data.insert(dt.sequence_number, dt.data);
        session.packets_received += 1;
        session.last_activity = Instant::now();

        if session.is_complete() {
            let payload = session.reassemble();
            let pgn = session.pgn;
            self.sessions.remove(&key);
            Ok(Some((pgn, payload)))
        } else {
            Ok(None)
        }
    }

    /// Find a session key matching (src, dst) or (src, 0xFF).
    fn find_session_key(&self, src: u8, dst: u8) -> Option<SessionKey> {
        // Exact match first
        let exact_key = self
            .sessions
            .keys()
            .find(|k| k.0 == src && k.1 == dst)
            .copied();
        if exact_key.is_some() {
            return exact_key;
        }
        // BAM match (dst == 0xFF)
        self.sessions
            .keys()
            .find(|k| k.0 == src && k.1 == 0xFF)
            .copied()
    }

    /// Remove sessions that have exceeded the configured timeout.
    ///
    /// Returns the number of sessions evicted.
    pub fn evict_expired(&mut self) -> usize {
        let timeout = self.timeout;
        let before = self.sessions.len();
        self.sessions
            .retain(|_, session| session.last_activity.elapsed() < timeout);
        before - self.sessions.len()
    }

    /// Return the number of currently active sessions.
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Return `true` if a session exists for `(src, dst)`.
    pub fn has_session(&self, src: u8, dst: u8) -> bool {
        self.find_session_key(src, dst).is_some()
    }

    /// Return elapsed time for the oldest active session, if any.
    pub fn oldest_session_age(&self) -> Option<Duration> {
        self.sessions.values().map(|s| s.started_at.elapsed()).max()
    }

    /// Abort a session by `(src, dst, pgn)` and return the reason.
    pub fn abort_session(&mut self, src: u8, dst: u8, pgn: u32) -> bool {
        let key = (src, dst, pgn);
        self.sessions.remove(&key).is_some()
    }
}

impl Default for TpReassembler {
    fn default() -> Self {
        Self::new(Duration::from_secs(5))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AbortReason
    // -------------------------------------------------------------------------

    #[test]
    fn test_abort_reason_roundtrip() {
        let reasons = [
            AbortReason::AlreadyInProgress,
            AbortReason::SystemResourcesNeeded,
            AbortReason::Timeout,
            AbortReason::CtsWhileTransferring,
            AbortReason::MaxRetransmit,
            AbortReason::UnexpectedDataPacket,
            AbortReason::BadSequenceNumber,
            AbortReason::DuplicateSequenceNumber,
            AbortReason::MessageTooLarge,
        ];
        for reason in &reasons {
            let raw = reason.as_raw();
            let decoded = AbortReason::from_raw(raw);
            assert_eq!(decoded.as_raw(), raw, "Roundtrip failed for {:?}", reason);
        }
    }

    #[test]
    fn test_abort_reason_unknown() {
        let r = AbortReason::from_raw(99);
        assert!(matches!(r, AbortReason::Unknown(99)));
        assert_eq!(r.as_raw(), 99);
    }

    // -------------------------------------------------------------------------
    // TpControlMessage encode/decode
    // -------------------------------------------------------------------------

    #[test]
    fn test_tp_cm_bam_roundtrip() {
        let msg = TpControlMessage::Bam {
            total_message_size: 9,
            total_packets: 2,
            pgn: 61444,
        };
        let encoded = msg.encode();
        assert_eq!(encoded[0], TpControlMessage::CTRL_BAM);
        let decoded = TpControlMessage::decode(&encoded).expect("decode should succeed");
        match decoded {
            TpControlMessage::Bam {
                total_message_size,
                total_packets,
                pgn,
            } => {
                assert_eq!(total_message_size, 9);
                assert_eq!(total_packets, 2);
                assert_eq!(pgn, 61444);
            }
            _ => panic!("Expected Bam"),
        }
    }

    #[test]
    fn test_tp_cm_rts_roundtrip() {
        let msg = TpControlMessage::Rts {
            total_message_size: 100,
            total_packets: 15,
            max_packets_per_cts: 5,
            pgn: 65265,
        };
        let encoded = msg.encode();
        assert_eq!(encoded[0], TpControlMessage::CTRL_RTS);
        let decoded = TpControlMessage::decode(&encoded).expect("decode should succeed");
        match decoded {
            TpControlMessage::Rts {
                total_message_size,
                total_packets,
                max_packets_per_cts,
                pgn,
            } => {
                assert_eq!(total_message_size, 100);
                assert_eq!(total_packets, 15);
                assert_eq!(max_packets_per_cts, 5);
                assert_eq!(pgn, 65265);
            }
            _ => panic!("Expected Rts"),
        }
    }

    #[test]
    fn test_tp_cm_cts_roundtrip() {
        let msg = TpControlMessage::Cts {
            packets_to_send: 5,
            next_packet: 6,
            pgn: 65265,
        };
        let encoded = msg.encode();
        assert_eq!(encoded[0], TpControlMessage::CTRL_CTS);
        let decoded = TpControlMessage::decode(&encoded).expect("decode should succeed");
        match decoded {
            TpControlMessage::Cts {
                packets_to_send,
                next_packet,
                pgn,
            } => {
                assert_eq!(packets_to_send, 5);
                assert_eq!(next_packet, 6);
                assert_eq!(pgn, 65265);
            }
            _ => panic!("Expected Cts"),
        }
    }

    #[test]
    fn test_tp_cm_eom_roundtrip() {
        let msg = TpControlMessage::Eom {
            total_message_size: 100,
            total_packets: 15,
            pgn: 65265,
        };
        let encoded = msg.encode();
        assert_eq!(encoded[0], TpControlMessage::CTRL_EOM);
        let decoded = TpControlMessage::decode(&encoded).expect("decode should succeed");
        match decoded {
            TpControlMessage::Eom {
                total_message_size,
                total_packets,
                pgn,
            } => {
                assert_eq!(total_message_size, 100);
                assert_eq!(total_packets, 15);
                assert_eq!(pgn, 65265);
            }
            _ => panic!("Expected Eom"),
        }
    }

    #[test]
    fn test_tp_cm_abort_roundtrip() {
        let msg = TpControlMessage::Abort {
            abort_reason: AbortReason::Timeout,
            pgn: 61444,
        };
        let encoded = msg.encode();
        assert_eq!(encoded[0], TpControlMessage::CTRL_ABORT);
        let decoded = TpControlMessage::decode(&encoded).expect("decode should succeed");
        match decoded {
            TpControlMessage::Abort { abort_reason, pgn } => {
                assert_eq!(abort_reason.as_raw(), AbortReason::Timeout.as_raw());
                assert_eq!(pgn, 61444);
            }
            _ => panic!("Expected Abort"),
        }
    }

    #[test]
    fn test_tp_cm_decode_unknown_ctrl_byte() {
        let data = [0x63u8, 0, 0, 0, 0, 0, 0, 0];
        let result = TpControlMessage::decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tp_cm_decode_too_short() {
        let data = [0x20u8, 0, 0];
        let result = TpControlMessage::decode(&data);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // TpDataTransfer
    // -------------------------------------------------------------------------

    #[test]
    fn test_tp_dt_roundtrip() {
        let dt = TpDataTransfer {
            sequence_number: 3,
            data: [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        };
        let encoded = dt.encode();
        assert_eq!(encoded[0], 3);
        assert_eq!(&encoded[1..], &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]);
        let decoded = TpDataTransfer::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.sequence_number, 3);
        assert_eq!(decoded.data, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]);
    }

    // -------------------------------------------------------------------------
    // TpReassembler – BAM flow
    // -------------------------------------------------------------------------

    #[test]
    fn test_reassembler_bam_complete() {
        let mut r = TpReassembler::new(Duration::from_secs(5));

        let bam = TpControlMessage::Bam {
            total_message_size: 9,
            total_packets: 2,
            pgn: 61444,
        };
        r.handle_cm(0x00, 0xFF, bam).expect("BAM should succeed");
        assert_eq!(r.active_session_count(), 1);

        let dt1 = TpDataTransfer {
            sequence_number: 1,
            data: [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11],
        };
        let res1 = r.handle_dt(0x00, 0xFF, dt1).expect("DT1 should succeed");
        assert!(res1.is_none());

        let dt2 = TpDataTransfer {
            sequence_number: 2,
            data: [0x22, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        };
        let res2 = r.handle_dt(0x00, 0xFF, dt2).expect("DT2 should succeed");
        assert!(res2.is_some());
        let (pgn, data) = res2.expect("result should exist");
        assert_eq!(pgn, 61444);
        assert_eq!(data.len(), 9); // truncated to total_message_size
        assert_eq!(&data[..7], &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11]);
        assert_eq!(&data[7..9], &[0x22, 0x33]);

        // Session should be removed after completion
        assert_eq!(r.active_session_count(), 0);
    }

    #[test]
    fn test_reassembler_bam_no_session_error() {
        let mut r = TpReassembler::new(Duration::from_secs(5));
        let dt = TpDataTransfer {
            sequence_number: 1,
            data: [0u8; 7],
        };
        let result = r.handle_dt(0x00, 0xFF, dt);
        assert!(result.is_err());
    }

    #[test]
    fn test_reassembler_bad_sequence_number() {
        let mut r = TpReassembler::new(Duration::from_secs(5));
        let bam = TpControlMessage::Bam {
            total_message_size: 7,
            total_packets: 1,
            pgn: 61444,
        };
        r.handle_cm(0x00, 0xFF, bam).expect("BAM should succeed");

        let dt = TpDataTransfer {
            sequence_number: 0,
            data: [0u8; 7],
        };
        let result = r.handle_dt(0x00, 0xFF, dt);
        assert!(result.is_err());
    }

    #[test]
    fn test_reassembler_duplicate_sequence_number() {
        let mut r = TpReassembler::new(Duration::from_secs(5));
        let bam = TpControlMessage::Bam {
            total_message_size: 14,
            total_packets: 2,
            pgn: 61444,
        };
        r.handle_cm(0x00, 0xFF, bam).expect("BAM should succeed");

        let dt1 = TpDataTransfer {
            sequence_number: 1,
            data: [0xAA; 7],
        };
        r.handle_dt(0x00, 0xFF, dt1.clone())
            .expect("DT1 should succeed");

        // Duplicate seq 1
        let result = r.handle_dt(0x00, 0xFF, dt1);
        assert!(result.is_err());
    }

    #[test]
    fn test_reassembler_abort_removes_session() {
        let mut r = TpReassembler::new(Duration::from_secs(5));
        let bam = TpControlMessage::Bam {
            total_message_size: 14,
            total_packets: 2,
            pgn: 61444,
        };
        r.handle_cm(0x00, 0xFF, bam).expect("BAM should succeed");
        assert_eq!(r.active_session_count(), 1);

        let abort = TpControlMessage::Abort {
            abort_reason: AbortReason::Timeout,
            pgn: 61444,
        };
        r.handle_cm(0x00, 0xFF, abort)
            .expect("Abort should succeed");
        assert_eq!(r.active_session_count(), 0);
    }

    #[test]
    fn test_reassembler_evict_expired() {
        let mut r = TpReassembler::new(Duration::from_nanos(1)); // 1ns timeout
        let bam = TpControlMessage::Bam {
            total_message_size: 14,
            total_packets: 2,
            pgn: 61444,
        };
        r.handle_cm(0x00, 0xFF, bam).expect("BAM should succeed");
        assert_eq!(r.active_session_count(), 1);

        // Sleep briefly to ensure timeout
        std::thread::sleep(Duration::from_millis(10));
        let evicted = r.evict_expired();
        assert_eq!(evicted, 1);
        assert_eq!(r.active_session_count(), 0);
    }

    #[test]
    fn test_reassembler_bam_invalid_params() {
        let mut r = TpReassembler::new(Duration::from_secs(5));

        let bad = TpControlMessage::Bam {
            total_message_size: 9,
            total_packets: 0, // invalid
            pgn: 61444,
        };
        assert!(r.handle_cm(0x00, 0xFF, bad).is_err());
    }

    #[test]
    fn test_reassembler_rts_session() {
        let mut r = TpReassembler::new(Duration::from_secs(5));

        let rts = TpControlMessage::Rts {
            total_message_size: 7,
            total_packets: 1,
            max_packets_per_cts: 1,
            pgn: 65265,
        };
        r.handle_cm(0x01, 0x80, rts).expect("RTS should succeed");
        assert!(r.has_session(0x01, 0x80));

        let dt = TpDataTransfer {
            sequence_number: 1,
            data: [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77],
        };
        let result = r.handle_dt(0x01, 0x80, dt).expect("DT should succeed");
        assert!(result.is_some());
        let (pgn, data) = result.expect("result should exist");
        assert_eq!(pgn, 65265);
        assert_eq!(data.len(), 7);
    }

    #[test]
    fn test_reassembler_oldest_session_age() {
        let mut r = TpReassembler::new(Duration::from_secs(5));
        assert!(r.oldest_session_age().is_none());

        let bam = TpControlMessage::Bam {
            total_message_size: 14,
            total_packets: 2,
            pgn: 61444,
        };
        r.handle_cm(0x00, 0xFF, bam).expect("BAM should succeed");
        assert!(r.oldest_session_age().is_some());
    }
}
