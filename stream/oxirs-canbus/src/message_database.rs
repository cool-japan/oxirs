//! CAN message database (DBC-like).
//!
//! Provides a structured in-memory database of CAN message and signal
//! definitions, analogous to the DBC (Database CAN) format used in automotive
//! development. Features:
//!
//! - Message definitions: CAN ID, name, DLC, signal list.
//! - Signal definitions: start bit, length, factor, offset, min/max, unit.
//! - Intel (little-endian) and Motorola (big-endian) byte order support.
//! - Signal value decoding from raw 8-byte CAN payloads.
//! - Signal value encoding back to raw bytes.
//! - Message lookup by CAN ID.
//! - Message filtering by ID range.
//! - Signal range validation (decoded value within [min, max]).
//! - Database statistics: total message count and signal count.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Byte order
// ─────────────────────────────────────────────────────────────────────────────

/// Bit ordering / byte order for CAN signal extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Intel / little-endian: LSB at `start_bit`.
    Intel,
    /// Motorola / big-endian: MSB at `start_bit`.
    Motorola,
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal definition
// ─────────────────────────────────────────────────────────────────────────────

/// Definition of a CAN signal within a message frame.
#[derive(Debug, Clone)]
pub struct SignalDef {
    /// Signal name.
    pub name: String,
    /// Bit position of LSB (Intel) or MSB (Motorola) in the frame.
    pub start_bit: u8,
    /// Signal length in bits (1–64).
    pub bit_length: u8,
    /// Byte order / bit ordering.
    pub byte_order: ByteOrder,
    /// Scale factor applied to the raw integer: physical = raw * factor + offset.
    pub factor: f64,
    /// Offset applied after scaling.
    pub offset: f64,
    /// Minimum valid physical value (inclusive).
    pub min_value: f64,
    /// Maximum valid physical value (inclusive).
    pub max_value: f64,
    /// Engineering unit string (e.g. `"km/h"`, `"°C"`).
    pub unit: String,
}

impl SignalDef {
    /// Compute the physical value from a raw integer.
    #[inline]
    pub fn to_physical(&self, raw: u64) -> f64 {
        raw as f64 * self.factor + self.offset
    }

    /// Compute the raw integer from a physical value (inverse of `to_physical`).
    #[inline]
    pub fn to_raw(&self, physical: f64) -> u64 {
        ((physical - self.offset) / self.factor).round() as u64
    }

    /// Whether a decoded physical value is within the signal's [min, max] range.
    pub fn in_range(&self, physical: f64) -> bool {
        physical >= self.min_value && physical <= self.max_value
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message definition
// ─────────────────────────────────────────────────────────────────────────────

/// Definition of a CAN message frame.
#[derive(Debug, Clone)]
pub struct MessageDef {
    /// CAN message identifier (11-bit or 29-bit extended).
    pub can_id: u32,
    /// Descriptive name (e.g. `"EngineStatus"`).
    pub name: String,
    /// Data Length Code: number of data bytes (0–8 for CAN 2.0).
    pub dlc: u8,
    /// Signals contained in this message.
    pub signals: Vec<SignalDef>,
}

impl MessageDef {
    /// Create a new message definition with no signals.
    pub fn new(can_id: u32, name: impl Into<String>, dlc: u8) -> Self {
        Self {
            can_id,
            name: name.into(),
            dlc,
            signals: Vec::new(),
        }
    }

    /// Add a signal definition to this message.
    pub fn with_signal(mut self, signal: SignalDef) -> Self {
        self.signals.push(signal);
        self
    }

    /// Number of signals in this message.
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }

    /// Look up a signal by name.
    pub fn signal(&self, name: &str) -> Option<&SignalDef> {
        self.signals.iter().find(|s| s.name == name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoded signal value
// ─────────────────────────────────────────────────────────────────────────────

/// Result of decoding a single signal from a CAN frame.
#[derive(Debug, Clone)]
pub struct DecodedSignal {
    /// Signal name.
    pub name: String,
    /// Raw integer bits extracted from the frame.
    pub raw_value: u64,
    /// Physical value after applying factor and offset.
    pub physical_value: f64,
    /// Engineering unit.
    pub unit: String,
    /// Whether `physical_value` is within the signal's [min, max].
    pub in_range: bool,
}

/// Result of decoding all signals from a CAN message frame.
#[derive(Debug, Clone)]
pub struct DecodedMessage {
    /// CAN ID of the decoded message.
    pub can_id: u32,
    /// Message name.
    pub name: String,
    /// Decoded signals.
    pub signals: Vec<DecodedSignal>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoded signal value
// ─────────────────────────────────────────────────────────────────────────────

/// A physical signal value to be encoded into a CAN frame.
pub struct SignalValue {
    /// Signal name.
    pub name: String,
    /// Physical value.
    pub physical: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Database statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a [`MessageDatabase`].
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    /// Total number of message definitions.
    pub message_count: usize,
    /// Total number of signal definitions across all messages.
    pub signal_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit extraction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract bits using Intel (little-endian) byte order.
///
/// `start_bit` is the position of the LSB in the flattened byte array
/// (byte 0 bit 0 = position 0, byte 0 bit 7 = position 7, byte 1 bit 0 = 8, …).
fn extract_bits_intel(data: &[u8], start_bit: u8, bit_length: u8) -> u64 {
    let mut result = 0u64;
    for i in 0..bit_length as usize {
        let bit_pos = start_bit as usize + i;
        let byte_idx = bit_pos / 8;
        let bit_idx = bit_pos % 8;
        if byte_idx < data.len() {
            let bit = ((data[byte_idx] >> bit_idx) & 1) as u64;
            result |= bit << i;
        }
    }
    result
}

/// Extract bits using Motorola (big-endian) byte order.
///
/// `start_bit` is the position of the MSB in the Motorola convention:
/// byte n, bit b → position = n * 8 + (7 - b).
fn extract_bits_motorola(data: &[u8], start_bit: u8, bit_length: u8) -> u64 {
    // Convert Motorola start_bit to byte/bit indices.
    let start_byte = (start_bit / 8) as usize;
    let start_bit_in_byte = start_bit % 8;

    let mut result = 0u64;
    let mut bits_remaining = bit_length as usize;
    let mut current_byte = start_byte;
    let mut current_bit = start_bit_in_byte as i32;

    while bits_remaining > 0 {
        if current_byte >= data.len() {
            break;
        }
        let bit = ((data[current_byte] >> current_bit) & 1) as u64;
        result |= bit << (bits_remaining - 1);
        bits_remaining -= 1;
        current_bit -= 1;
        if current_bit < 0 {
            current_bit = 7;
            current_byte += 1;
        }
    }
    result
}

/// Write `bit_length` bits of `value` using Intel (LE) byte order into `data`.
fn write_bits_intel(data: &mut [u8], start_bit: u8, bit_length: u8, value: u64) {
    for i in 0..bit_length as usize {
        let bit_pos = start_bit as usize + i;
        let byte_idx = bit_pos / 8;
        let bit_idx = bit_pos % 8;
        if byte_idx < data.len() {
            let bit = ((value >> i) & 1) as u8;
            data[byte_idx] = (data[byte_idx] & !(1 << bit_idx)) | (bit << bit_idx);
        }
    }
}

/// Write `bit_length` bits of `value` using Motorola (BE) byte order into `data`.
fn write_bits_motorola(data: &mut [u8], start_bit: u8, bit_length: u8, value: u64) {
    let start_byte = (start_bit / 8) as usize;
    let start_bit_in_byte = start_bit % 8;

    let mut bits_remaining = bit_length as usize;
    let mut current_byte = start_byte;
    let mut current_bit = start_bit_in_byte as i32;

    while bits_remaining > 0 {
        if current_byte >= data.len() {
            break;
        }
        let bit = ((value >> (bits_remaining - 1)) & 1) as u8;
        data[current_byte] = (data[current_byte] & !(1 << current_bit)) | (bit << current_bit);
        bits_remaining -= 1;
        current_bit -= 1;
        if current_bit < 0 {
            current_bit = 7;
            current_byte += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message database
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory CAN message database.
pub struct MessageDatabase {
    /// Messages keyed by CAN ID.
    messages: HashMap<u32, MessageDef>,
}

impl MessageDatabase {
    /// Create a new empty database.
    pub fn new() -> Self {
        Self {
            messages: HashMap::new(),
        }
    }

    // ── registration ──────────────────────────────────────────────────────────

    /// Add or replace a message definition.
    pub fn add_message(&mut self, message: MessageDef) {
        self.messages.insert(message.can_id, message);
    }

    /// Look up a message definition by CAN ID.
    pub fn message(&self, can_id: u32) -> Option<&MessageDef> {
        self.messages.get(&can_id)
    }

    /// Return all message definitions whose CAN ID falls in `[lo, hi]`.
    pub fn messages_in_range(&self, lo: u32, hi: u32) -> Vec<&MessageDef> {
        let mut msgs: Vec<&MessageDef> = self
            .messages
            .values()
            .filter(|m| m.can_id >= lo && m.can_id <= hi)
            .collect();
        msgs.sort_by_key(|m| m.can_id);
        msgs
    }

    // ── decoding ──────────────────────────────────────────────────────────────

    /// Decode all signals of the message with `can_id` from `payload`.
    ///
    /// Returns `None` if the CAN ID is not registered in the database.
    pub fn decode(&self, can_id: u32, payload: &[u8]) -> Option<DecodedMessage> {
        let msg_def = self.messages.get(&can_id)?;
        let signals = msg_def
            .signals
            .iter()
            .map(|sig| {
                let raw = match sig.byte_order {
                    ByteOrder::Intel => extract_bits_intel(payload, sig.start_bit, sig.bit_length),
                    ByteOrder::Motorola => {
                        extract_bits_motorola(payload, sig.start_bit, sig.bit_length)
                    }
                };
                let physical = sig.to_physical(raw);
                DecodedSignal {
                    name: sig.name.clone(),
                    raw_value: raw,
                    physical_value: physical,
                    unit: sig.unit.clone(),
                    in_range: sig.in_range(physical),
                }
            })
            .collect();

        Some(DecodedMessage {
            can_id,
            name: msg_def.name.clone(),
            signals,
        })
    }

    // ── encoding ──────────────────────────────────────────────────────────────

    /// Encode `signal_values` into an 8-byte CAN payload for message `can_id`.
    ///
    /// Returns `None` if the CAN ID is not registered. Unknown signal names
    /// in `signal_values` are silently ignored.
    pub fn encode(&self, can_id: u32, signal_values: &[SignalValue]) -> Option<[u8; 8]> {
        let msg_def = self.messages.get(&can_id)?;
        let mut payload = [0u8; 8];

        for sv in signal_values {
            if let Some(sig) = msg_def.signal(&sv.name) {
                let raw = sig.to_raw(sv.physical);
                match sig.byte_order {
                    ByteOrder::Intel => {
                        write_bits_intel(&mut payload, sig.start_bit, sig.bit_length, raw);
                    }
                    ByteOrder::Motorola => {
                        write_bits_motorola(&mut payload, sig.start_bit, sig.bit_length, raw);
                    }
                }
            }
        }
        Some(payload)
    }

    // ── statistics ────────────────────────────────────────────────────────────

    /// Return summary statistics for the database.
    pub fn stats(&self) -> DatabaseStats {
        let message_count = self.messages.len();
        let signal_count = self.messages.values().map(|m| m.signal_count()).sum();
        DatabaseStats {
            message_count,
            signal_count,
        }
    }

    /// Number of message definitions in the database.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

impl Default for MessageDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn speed_signal() -> SignalDef {
        SignalDef {
            name: "VehicleSpeed".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::Intel,
            factor: 0.5,
            offset: 0.0,
            min_value: 0.0,
            max_value: 120.0,
            unit: "km/h".into(),
        }
    }

    fn rpm_signal() -> SignalDef {
        SignalDef {
            name: "EngineRPM".into(),
            start_bit: 8,
            bit_length: 16,
            byte_order: ByteOrder::Intel,
            factor: 0.25,
            offset: 0.0,
            min_value: 0.0,
            max_value: 8000.0,
            unit: "rpm".into(),
        }
    }

    fn engine_msg() -> MessageDef {
        MessageDef::new(0x100, "EngineStatus", 8)
            .with_signal(speed_signal())
            .with_signal(rpm_signal())
    }

    fn db_with_engine() -> MessageDatabase {
        let mut db = MessageDatabase::new();
        db.add_message(engine_msg());
        db
    }

    // ── message registration ──────────────────────────────────────────────────

    #[test]
    fn test_add_and_lookup_message() {
        let db = db_with_engine();
        let msg = db.message(0x100);
        assert!(msg.is_some());
        assert_eq!(msg.expect("message exists").name, "EngineStatus");
    }

    #[test]
    fn test_lookup_nonexistent_returns_none() {
        let db = db_with_engine();
        assert!(db.message(0xDEAD).is_none());
    }

    #[test]
    fn test_overwrite_existing_message() {
        let mut db = MessageDatabase::new();
        db.add_message(MessageDef::new(0x100, "Old", 4));
        db.add_message(MessageDef::new(0x100, "New", 8));
        assert_eq!(db.message(0x100).expect("message exists").name, "New");
    }

    // ── signal decoding (Intel) ───────────────────────────────────────────────

    #[test]
    fn test_decode_speed_intel() {
        let db = db_with_engine();
        // byte 0 = 200 → raw=200, physical = 200 * 0.5 = 100 km/h
        let payload = [200u8, 0, 0, 0, 0, 0, 0, 0];
        let decoded = db.decode(0x100, &payload).expect("message found");
        let speed = decoded
            .signals
            .iter()
            .find(|s| s.name == "VehicleSpeed")
            .expect("signal found");
        assert!((speed.physical_value - 100.0).abs() < 1e-9);
        assert_eq!(speed.raw_value, 200);
        assert_eq!(speed.unit, "km/h");
        assert!(speed.in_range);
    }

    #[test]
    fn test_decode_rpm_intel_16bit() {
        let db = db_with_engine();
        // RPM at bits 8..23 (bytes 1-2): raw = 4000, physical = 4000 * 0.25 = 1000
        let mut payload = [0u8; 8];
        payload[1] = (4000u16 & 0xFF) as u8;
        payload[2] = ((4000u16 >> 8) & 0xFF) as u8;
        let decoded = db.decode(0x100, &payload).expect("message found");
        let rpm = decoded
            .signals
            .iter()
            .find(|s| s.name == "EngineRPM")
            .expect("signal found");
        assert!((rpm.physical_value - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_decode_unknown_can_id_returns_none() {
        let db = db_with_engine();
        assert!(db.decode(0x999, &[0u8; 8]).is_none());
    }

    // ── signal decoding (Motorola) ────────────────────────────────────────────

    #[test]
    fn test_decode_motorola_single_byte() {
        let mut db = MessageDatabase::new();
        let sig = SignalDef {
            name: "Temp".into(),
            start_bit: 7,
            bit_length: 8,
            byte_order: ByteOrder::Motorola,
            factor: 1.0,
            offset: -40.0,
            min_value: -40.0,
            max_value: 215.0,
            unit: "°C".into(),
        };
        db.add_message(MessageDef::new(0x200, "Climate", 8).with_signal(sig));
        // byte 0 = 100 → raw=100, physical = 100 - 40 = 60 °C
        let payload = [100u8, 0, 0, 0, 0, 0, 0, 0];
        let decoded = db.decode(0x200, &payload).expect("ok");
        let temp = &decoded.signals[0];
        assert!((temp.physical_value - 60.0).abs() < 1e-9);
    }

    // ── signal encoding ───────────────────────────────────────────────────────

    #[test]
    fn test_encode_speed_intel() {
        let db = db_with_engine();
        let values = [SignalValue {
            name: "VehicleSpeed".into(),
            physical: 100.0,
        }];
        let payload = db.encode(0x100, &values).expect("ok");
        // raw = (100.0 - 0.0) / 0.5 = 200 = 0xC8
        assert_eq!(payload[0], 200);
    }

    #[test]
    fn test_encode_unknown_can_id_returns_none() {
        let db = db_with_engine();
        let values: Vec<SignalValue> = vec![];
        assert!(db.encode(0xDEAD, &values).is_none());
    }

    #[test]
    fn test_encode_unknown_signal_ignored() {
        let db = db_with_engine();
        let values = [SignalValue {
            name: "NonExistent".into(),
            physical: 42.0,
        }];
        let payload = db.encode(0x100, &values).expect("ok");
        // All bytes should remain zero.
        assert_eq!(payload, [0u8; 8]);
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let db = db_with_engine();
        let values = [SignalValue {
            name: "VehicleSpeed".into(),
            physical: 80.0,
        }];
        let payload = db.encode(0x100, &values).expect("ok");
        let decoded = db.decode(0x100, &payload).expect("ok");
        let speed = decoded
            .signals
            .iter()
            .find(|s| s.name == "VehicleSpeed")
            .expect("ok");
        assert!((speed.physical_value - 80.0).abs() < 1e-9);
    }

    // ── range validation ──────────────────────────────────────────────────────

    #[test]
    fn test_in_range_true_when_within_bounds() {
        let sig = speed_signal();
        assert!(sig.in_range(60.0));
    }

    #[test]
    fn test_in_range_false_when_above_max() {
        let sig = speed_signal();
        // max = 120, raw=255 → physical=127.5
        assert!(!sig.in_range(127.5));
    }

    #[test]
    fn test_in_range_at_exact_boundary() {
        let sig = speed_signal();
        assert!(sig.in_range(0.0));
        assert!(sig.in_range(120.0));
    }

    #[test]
    fn test_decoded_signal_in_range_false() {
        let db = db_with_engine();
        // raw=255, physical=127.5 > 120 → out of range
        let payload = [255u8, 0, 0, 0, 0, 0, 0, 0];
        let decoded = db.decode(0x100, &payload).expect("ok");
        let speed = decoded
            .signals
            .iter()
            .find(|s| s.name == "VehicleSpeed")
            .expect("ok");
        assert!(!speed.in_range);
    }

    // ── filtering ─────────────────────────────────────────────────────────────

    #[test]
    fn test_messages_in_range_returns_correct_messages() {
        let mut db = MessageDatabase::new();
        db.add_message(MessageDef::new(0x100, "A", 8));
        db.add_message(MessageDef::new(0x200, "B", 8));
        db.add_message(MessageDef::new(0x300, "C", 8));
        let results = db.messages_in_range(0x100, 0x200);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].can_id, 0x100);
        assert_eq!(results[1].can_id, 0x200);
    }

    #[test]
    fn test_messages_in_range_empty_when_no_match() {
        let db = db_with_engine();
        let results = db.messages_in_range(0x500, 0x600);
        assert!(results.is_empty());
    }

    #[test]
    fn test_messages_in_range_sorted_by_id() {
        let mut db = MessageDatabase::new();
        db.add_message(MessageDef::new(0x300, "C", 8));
        db.add_message(MessageDef::new(0x100, "A", 8));
        db.add_message(MessageDef::new(0x200, "B", 8));
        let results = db.messages_in_range(0x100, 0x300);
        assert_eq!(results[0].can_id, 0x100);
        assert_eq!(results[1].can_id, 0x200);
        assert_eq!(results[2].can_id, 0x300);
    }

    // ── statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_message_count() {
        let db = db_with_engine();
        assert_eq!(db.stats().message_count, 1);
    }

    #[test]
    fn test_stats_signal_count() {
        let db = db_with_engine();
        assert_eq!(db.stats().signal_count, 2);
    }

    #[test]
    fn test_stats_empty_database() {
        let db = MessageDatabase::new();
        let stats = db.stats();
        assert_eq!(stats.message_count, 0);
        assert_eq!(stats.signal_count, 0);
    }

    // ── message def helpers ───────────────────────────────────────────────────

    #[test]
    fn test_message_signal_count() {
        let msg = engine_msg();
        assert_eq!(msg.signal_count(), 2);
    }

    #[test]
    fn test_message_signal_lookup_by_name() {
        let msg = engine_msg();
        assert!(msg.signal("VehicleSpeed").is_some());
        assert!(msg.signal("NonExistent").is_none());
    }

    #[test]
    fn test_message_with_no_signals() {
        let msg = MessageDef::new(0x001, "Empty", 0);
        assert_eq!(msg.signal_count(), 0);
    }

    // ── signal def helpers ────────────────────────────────────────────────────

    #[test]
    fn test_signal_to_physical_and_back() {
        let sig = speed_signal();
        let raw = 120u64;
        let physical = sig.to_physical(raw);
        let back = sig.to_raw(physical);
        assert_eq!(back, raw);
    }

    #[test]
    fn test_signal_to_raw_rounds_correctly() {
        let sig = SignalDef {
            name: "t".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::Intel,
            factor: 0.1,
            offset: 0.0,
            min_value: 0.0,
            max_value: 25.0,
            unit: String::new(),
        };
        // physical=2.5 → raw = round(2.5 / 0.1) = 25
        assert_eq!(sig.to_raw(2.5), 25);
    }

    // ── decode empty payload ───────────────────────────────────────────────────

    #[test]
    fn test_decode_zero_payload_all_zeros() {
        let db = db_with_engine();
        let payload = [0u8; 8];
        let decoded = db.decode(0x100, &payload).expect("ok");
        for sig in &decoded.signals {
            assert_eq!(sig.raw_value, 0);
        }
    }

    // ── default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_default_database_is_empty() {
        let db = MessageDatabase::default();
        assert_eq!(db.message_count(), 0);
    }

    // ── decode message name preserved ─────────────────────────────────────────

    #[test]
    fn test_decoded_message_name_matches_definition() {
        let db = db_with_engine();
        let decoded = db.decode(0x100, &[0u8; 8]).expect("ok");
        assert_eq!(decoded.name, "EngineStatus");
        assert_eq!(decoded.can_id, 0x100);
    }

    // ── Intel bit extraction edge cases ──────────────────────────────────────

    #[test]
    fn test_extract_bits_intel_single_bit() {
        let data = [0b0000_0100u8, 0, 0, 0, 0, 0, 0, 0];
        // start_bit=2, bit_length=1 → bit 2 of byte 0 = 1
        assert_eq!(extract_bits_intel(&data, 2, 1), 1);
    }

    #[test]
    fn test_extract_bits_intel_all_bytes() {
        let data = [0xFFu8; 8];
        // Extract 8 bits starting at bit 0
        assert_eq!(extract_bits_intel(&data, 0, 8), 0xFF);
    }

    // ── encode multiple signals ───────────────────────────────────────────────

    #[test]
    fn test_encode_two_signals() {
        let db = db_with_engine();
        let values = [
            SignalValue {
                name: "VehicleSpeed".into(),
                physical: 60.0,
            },
            SignalValue {
                name: "EngineRPM".into(),
                physical: 2000.0,
            },
        ];
        let payload = db.encode(0x100, &values).expect("ok");
        // Speed: raw=120 at byte 0.
        assert_eq!(payload[0], 120);
        // RPM: raw=8000, bytes 1-2 LE.
        let rpm_raw = payload[1] as u16 | ((payload[2] as u16) << 8);
        assert_eq!(rpm_raw, 8000);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_signal_factor_applied_correctly() {
        let sig = SignalDef {
            name: "t".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::Intel,
            factor: 0.1,
            offset: -40.0,
            min_value: -40.0,
            max_value: 85.5,
            unit: "°C".into(),
        };
        // raw=650, physical = 650 * 0.1 - 40 = 25.0
        assert!((sig.to_physical(650) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_message_dlc_preserved() {
        let msg = MessageDef::new(0x123, "Test", 6);
        assert_eq!(msg.dlc, 6);
    }

    #[test]
    fn test_decode_all_zero_payload_produces_offset_physical_value() {
        // When factor=1, offset=100, raw=0 → physical=100
        let mut db = MessageDatabase::new();
        let sig = SignalDef {
            name: "v".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::Intel,
            factor: 1.0,
            offset: 100.0,
            min_value: 100.0,
            max_value: 355.0,
            unit: String::new(),
        };
        db.add_message(MessageDef::new(0x10, "M", 8).with_signal(sig));
        let decoded = db.decode(0x10, &[0u8; 8]).expect("ok");
        assert!((decoded.signals[0].physical_value - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_message_database_message_count_after_multiple_adds() {
        let mut db = MessageDatabase::new();
        db.add_message(MessageDef::new(0x1, "A", 8));
        db.add_message(MessageDef::new(0x2, "B", 8));
        db.add_message(MessageDef::new(0x3, "C", 8));
        assert_eq!(db.message_count(), 3);
    }

    #[test]
    fn test_signal_name_preserved_in_decoded_result() {
        let db = db_with_engine();
        let decoded = db.decode(0x100, &[0u8; 8]).expect("ok");
        let names: Vec<&str> = decoded.signals.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"VehicleSpeed"));
        assert!(names.contains(&"EngineRPM"));
    }

    #[test]
    fn test_unit_preserved_in_decoded_result() {
        let db = db_with_engine();
        let decoded = db.decode(0x100, &[0u8; 8]).expect("ok");
        let speed = decoded
            .signals
            .iter()
            .find(|s| s.name == "VehicleSpeed")
            .expect("ok");
        assert_eq!(speed.unit, "km/h");
    }

    #[test]
    fn test_decode_single_bit_signal() {
        let mut db = MessageDatabase::new();
        let sig = SignalDef {
            name: "flag".into(),
            start_bit: 3,
            bit_length: 1,
            byte_order: ByteOrder::Intel,
            factor: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 1.0,
            unit: String::new(),
        };
        db.add_message(MessageDef::new(0x50, "Flags", 8).with_signal(sig));
        let payload = [0b0000_1000u8, 0, 0, 0, 0, 0, 0, 0]; // bit 3 = 1
        let decoded = db.decode(0x50, &payload).expect("ok");
        assert_eq!(decoded.signals[0].raw_value, 1);
        assert!((decoded.signals[0].physical_value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_signal_in_range_at_min() {
        let sig = rpm_signal();
        assert!(sig.in_range(0.0));
    }

    #[test]
    fn test_signal_in_range_at_max() {
        let sig = rpm_signal();
        assert!(sig.in_range(8000.0));
    }

    #[test]
    fn test_signal_out_of_range_below_min() {
        let sig = SignalDef {
            name: "v".into(),
            start_bit: 0,
            bit_length: 8,
            byte_order: ByteOrder::Intel,
            factor: 1.0,
            offset: 0.0,
            min_value: 10.0,
            max_value: 200.0,
            unit: String::new(),
        };
        assert!(!sig.in_range(5.0));
    }

    #[test]
    fn test_filter_by_id_range_single_message() {
        let mut db = MessageDatabase::new();
        db.add_message(MessageDef::new(0x50, "M", 8));
        let results = db.messages_in_range(0x50, 0x50);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].can_id, 0x50);
    }

    #[test]
    fn test_stats_multiple_messages_with_signals() {
        let mut db = MessageDatabase::new();
        db.add_message(
            MessageDef::new(0x1, "A", 8)
                .with_signal(speed_signal())
                .with_signal(rpm_signal()),
        );
        db.add_message(MessageDef::new(0x2, "B", 8).with_signal(speed_signal()));
        let stats = db.stats();
        assert_eq!(stats.message_count, 2);
        assert_eq!(stats.signal_count, 3);
    }

    #[test]
    fn test_encode_zero_physical_value() {
        let db = db_with_engine();
        let values = [SignalValue {
            name: "VehicleSpeed".into(),
            physical: 0.0,
        }];
        let payload = db.encode(0x100, &values).expect("ok");
        assert_eq!(payload[0], 0);
    }
}
