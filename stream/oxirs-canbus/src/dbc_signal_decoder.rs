//! # DBC Signal Decoder
//!
//! Provides a standalone DBC signal decoder for extracting and interpreting CAN
//! signals from raw frame data using DBC-style signal definitions (bit position,
//! bit length, endianness, factor, offset, unit).
//!
//! ## Features
//!
//! - **Signal extraction**: Extract raw bits from CAN payloads using bit position and length
//! - **Endianness**: Support for little-endian (Intel) and big-endian (Motorola) byte orders
//! - **Scaling**: Apply factor and offset to convert raw integer values to physical values
//! - **Range checking**: Validate physical values against min/max bounds
//! - **Batch decoding**: Decode all signals in a message with a single call
//! - **Signal lookup**: Query signal definitions by message ID and signal name

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Signal definition types
// ─────────────────────────────────────────────

/// Byte order of a CAN signal within the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ByteOrder {
    /// Little-endian (Intel) byte order.
    LittleEndian,
    /// Big-endian (Motorola) byte order.
    BigEndian,
}

/// Whether the raw value is signed or unsigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueSign {
    Unsigned,
    Signed,
}

/// Definition of a single CAN signal within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDef {
    /// Signal name.
    pub name: String,
    /// Start bit position (DBC-style: bit number in the payload).
    pub start_bit: u32,
    /// Length in bits.
    pub bit_length: u32,
    /// Byte order.
    pub byte_order: ByteOrder,
    /// Signed/unsigned.
    pub value_sign: ValueSign,
    /// Scale factor applied to raw value: physical = raw * factor + offset.
    pub factor: f64,
    /// Offset applied after scaling.
    pub offset: f64,
    /// Minimum physical value (for range checking).
    pub min_value: f64,
    /// Maximum physical value (for range checking).
    pub max_value: f64,
    /// Engineering unit (e.g., "rpm", "km/h").
    pub unit: String,
    /// Optional description.
    pub description: Option<String>,
    /// Receiver nodes.
    pub receivers: Vec<String>,
}

impl SignalDef {
    /// Create a new signal definition.
    pub fn new(
        name: impl Into<String>,
        start_bit: u32,
        bit_length: u32,
        byte_order: ByteOrder,
    ) -> Self {
        Self {
            name: name.into(),
            start_bit,
            bit_length,
            byte_order,
            value_sign: ValueSign::Unsigned,
            factor: 1.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 0.0,
            unit: String::new(),
            description: None,
            receivers: Vec::new(),
        }
    }

    /// Set factor and offset.
    pub fn with_factor_offset(mut self, factor: f64, offset: f64) -> Self {
        self.factor = factor;
        self.offset = offset;
        self
    }

    /// Set min/max range.
    pub fn with_range(mut self, min_value: f64, max_value: f64) -> Self {
        self.min_value = min_value;
        self.max_value = max_value;
        self
    }

    /// Set the unit.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }

    /// Set signed.
    pub fn signed(mut self) -> Self {
        self.value_sign = ValueSign::Signed;
        self
    }

    /// Convert a raw integer value to physical value.
    pub fn raw_to_physical(&self, raw: i64) -> f64 {
        (raw as f64) * self.factor + self.offset
    }

    /// Convert a physical value back to raw integer.
    pub fn physical_to_raw(&self, physical: f64) -> i64 {
        if self.factor.abs() < f64::EPSILON {
            return 0;
        }
        ((physical - self.offset) / self.factor).round() as i64
    }

    /// Check if a physical value is within the defined range.
    pub fn in_range(&self, physical: f64) -> bool {
        // If min == max == 0, no range check
        if (self.min_value - self.max_value).abs() < f64::EPSILON
            && self.min_value.abs() < f64::EPSILON
        {
            return true;
        }
        physical >= self.min_value && physical <= self.max_value
    }
}

/// Definition of a CAN message containing signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDef {
    /// CAN message ID.
    pub message_id: u32,
    /// Message name.
    pub name: String,
    /// Data length in bytes.
    pub dlc: u8,
    /// Transmitter node.
    pub transmitter: String,
    /// Signals in this message.
    pub signals: Vec<SignalDef>,
}

impl MessageDef {
    /// Create a new message definition.
    pub fn new(message_id: u32, name: impl Into<String>, dlc: u8) -> Self {
        Self {
            message_id,
            name: name.into(),
            dlc,
            transmitter: String::new(),
            signals: Vec::new(),
        }
    }

    /// Add a signal to this message.
    pub fn add_signal(&mut self, signal: SignalDef) {
        self.signals.push(signal);
    }

    /// Get a signal by name.
    pub fn get_signal(&self, name: &str) -> Option<&SignalDef> {
        self.signals.iter().find(|s| s.name == name)
    }
}

/// Result of decoding a single signal from a CAN frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedSignal {
    /// Signal name.
    pub name: String,
    /// Raw integer value extracted from the frame.
    pub raw_value: i64,
    /// Physical value after applying factor and offset.
    pub physical_value: f64,
    /// Engineering unit.
    pub unit: String,
    /// Whether the physical value is within the defined range.
    pub in_range: bool,
}

/// Error type for signal decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Message ID not found in the database.
    UnknownMessage(u32),
    /// Signal name not found in the message.
    UnknownSignal(String),
    /// Payload too short for the signal definition.
    PayloadTooShort { expected: usize, actual: usize },
    /// Bit position out of range.
    BitOutOfRange {
        start_bit: u32,
        bit_length: u32,
        payload_bits: u32,
    },
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::UnknownMessage(id) => write!(f, "unknown message ID: 0x{id:X}"),
            DecodeError::UnknownSignal(name) => write!(f, "unknown signal: {name}"),
            DecodeError::PayloadTooShort { expected, actual } => {
                write!(
                    f,
                    "payload too short: expected {expected} bytes, got {actual}"
                )
            }
            DecodeError::BitOutOfRange {
                start_bit,
                bit_length,
                payload_bits,
            } => {
                write!(
                    f,
                    "bit range [{start_bit}..{}] out of payload range [0..{payload_bits}]",
                    start_bit + bit_length
                )
            }
        }
    }
}

impl std::error::Error for DecodeError {}

// ─────────────────────────────────────────────
// DbcSignalDecoder
// ─────────────────────────────────────────────

/// Decoder that extracts and interprets CAN signals from raw frame data.
pub struct DbcSignalDecoder {
    messages: HashMap<u32, MessageDef>,
}

impl DbcSignalDecoder {
    /// Create an empty decoder.
    pub fn new() -> Self {
        Self {
            messages: HashMap::new(),
        }
    }

    /// Create a decoder from a list of message definitions.
    pub fn from_messages(messages: Vec<MessageDef>) -> Self {
        let map = messages.into_iter().map(|m| (m.message_id, m)).collect();
        Self { messages: map }
    }

    /// Add a message definition.
    pub fn add_message(&mut self, msg: MessageDef) {
        self.messages.insert(msg.message_id, msg);
    }

    /// Get the number of registered messages.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Get a message definition by ID.
    pub fn get_message(&self, message_id: u32) -> Option<&MessageDef> {
        self.messages.get(&message_id)
    }

    /// Get total signal count across all messages.
    pub fn total_signal_count(&self) -> usize {
        self.messages.values().map(|m| m.signals.len()).sum()
    }

    /// Decode all signals in a CAN frame.
    pub fn decode_frame(
        &self,
        message_id: u32,
        payload: &[u8],
    ) -> Result<Vec<DecodedSignal>, DecodeError> {
        let msg = self
            .messages
            .get(&message_id)
            .ok_or(DecodeError::UnknownMessage(message_id))?;

        let mut results = Vec::with_capacity(msg.signals.len());
        for signal in &msg.signals {
            let raw = extract_signal(payload, signal)?;
            let physical = signal.raw_to_physical(raw);
            results.push(DecodedSignal {
                name: signal.name.clone(),
                raw_value: raw,
                physical_value: physical,
                unit: signal.unit.clone(),
                in_range: signal.in_range(physical),
            });
        }
        Ok(results)
    }

    /// Decode a single signal by name from a CAN frame.
    pub fn decode_signal(
        &self,
        message_id: u32,
        signal_name: &str,
        payload: &[u8],
    ) -> Result<DecodedSignal, DecodeError> {
        let msg = self
            .messages
            .get(&message_id)
            .ok_or(DecodeError::UnknownMessage(message_id))?;

        let signal = msg
            .signals
            .iter()
            .find(|s| s.name == signal_name)
            .ok_or_else(|| DecodeError::UnknownSignal(signal_name.to_string()))?;

        let raw = extract_signal(payload, signal)?;
        let physical = signal.raw_to_physical(raw);
        Ok(DecodedSignal {
            name: signal.name.clone(),
            raw_value: raw,
            physical_value: physical,
            unit: signal.unit.clone(),
            in_range: signal.in_range(physical),
        })
    }

    /// Encode a physical value into a CAN payload.
    pub fn encode_signal(
        &self,
        message_id: u32,
        signal_name: &str,
        physical_value: f64,
        payload: &mut [u8],
    ) -> Result<(), DecodeError> {
        let msg = self
            .messages
            .get(&message_id)
            .ok_or(DecodeError::UnknownMessage(message_id))?;

        let signal = msg
            .signals
            .iter()
            .find(|s| s.name == signal_name)
            .ok_or_else(|| DecodeError::UnknownSignal(signal_name.to_string()))?;

        let raw = signal.physical_to_raw(physical_value);
        pack_signal(payload, signal, raw)?;
        Ok(())
    }
}

impl Default for DbcSignalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Bit extraction / packing
// ─────────────────────────────────────────────

/// Extract a signal's raw value from a CAN payload.
fn extract_signal(payload: &[u8], signal: &SignalDef) -> Result<i64, DecodeError> {
    let payload_bits = (payload.len() as u32) * 8;

    match signal.byte_order {
        ByteOrder::LittleEndian => {
            if signal.start_bit + signal.bit_length > payload_bits {
                return Err(DecodeError::BitOutOfRange {
                    start_bit: signal.start_bit,
                    bit_length: signal.bit_length,
                    payload_bits,
                });
            }
            let mut raw: u64 = 0;
            for i in 0..signal.bit_length {
                let bit_pos = signal.start_bit + i;
                let byte_idx = (bit_pos / 8) as usize;
                let bit_in_byte = bit_pos % 8;
                if byte_idx < payload.len() && (payload[byte_idx] >> bit_in_byte) & 1 == 1 {
                    raw |= 1u64 << i;
                }
            }
            // Sign extend if signed
            if signal.value_sign == ValueSign::Signed && signal.bit_length > 0 {
                let sign_bit = 1u64 << (signal.bit_length - 1);
                if raw & sign_bit != 0 {
                    // Sign extend
                    let mask = !((1u64 << signal.bit_length) - 1);
                    raw |= mask;
                }
            }
            Ok(raw as i64)
        }
        ByteOrder::BigEndian => {
            // Motorola byte order: start bit is the MSB position
            let mut raw: u64 = 0;
            let start_byte = (signal.start_bit / 8) as usize;
            let start_bit_in_byte = signal.start_bit % 8;

            let mut bits_remaining = signal.bit_length;
            let mut current_byte = start_byte;
            let mut current_bit = start_bit_in_byte as i32;
            let mut result_bit = signal.bit_length as i32 - 1;

            while bits_remaining > 0 {
                if current_byte >= payload.len() {
                    return Err(DecodeError::BitOutOfRange {
                        start_bit: signal.start_bit,
                        bit_length: signal.bit_length,
                        payload_bits,
                    });
                }
                if (payload[current_byte] >> current_bit as u32) & 1 == 1 {
                    raw |= 1u64 << result_bit as u32;
                }
                bits_remaining -= 1;
                result_bit -= 1;
                current_bit -= 1;
                if current_bit < 0 {
                    current_byte += 1;
                    current_bit = 7;
                }
            }

            if signal.value_sign == ValueSign::Signed && signal.bit_length > 0 {
                let sign_bit = 1u64 << (signal.bit_length - 1);
                if raw & sign_bit != 0 {
                    let mask = !((1u64 << signal.bit_length) - 1);
                    raw |= mask;
                }
            }
            Ok(raw as i64)
        }
    }
}

/// Pack a raw value into a CAN payload.
fn pack_signal(payload: &mut [u8], signal: &SignalDef, raw: i64) -> Result<(), DecodeError> {
    let payload_bits = (payload.len() as u32) * 8;
    let raw_unsigned = raw as u64;

    match signal.byte_order {
        ByteOrder::LittleEndian => {
            if signal.start_bit + signal.bit_length > payload_bits {
                return Err(DecodeError::BitOutOfRange {
                    start_bit: signal.start_bit,
                    bit_length: signal.bit_length,
                    payload_bits,
                });
            }
            for i in 0..signal.bit_length {
                let bit_pos = signal.start_bit + i;
                let byte_idx = (bit_pos / 8) as usize;
                let bit_in_byte = bit_pos % 8;
                if byte_idx < payload.len() {
                    if (raw_unsigned >> i) & 1 == 1 {
                        payload[byte_idx] |= 1u8 << bit_in_byte;
                    } else {
                        payload[byte_idx] &= !(1u8 << bit_in_byte);
                    }
                }
            }
            Ok(())
        }
        ByteOrder::BigEndian => {
            let start_byte = (signal.start_bit / 8) as usize;
            let start_bit_in_byte = signal.start_bit % 8;

            let mut bits_remaining = signal.bit_length;
            let mut current_byte = start_byte;
            let mut current_bit = start_bit_in_byte as i32;
            let mut source_bit = signal.bit_length as i32 - 1;

            while bits_remaining > 0 {
                if current_byte >= payload.len() {
                    return Err(DecodeError::BitOutOfRange {
                        start_bit: signal.start_bit,
                        bit_length: signal.bit_length,
                        payload_bits,
                    });
                }
                if (raw_unsigned >> source_bit as u32) & 1 == 1 {
                    payload[current_byte] |= 1u8 << current_bit as u32;
                } else {
                    payload[current_byte] &= !(1u8 << current_bit as u32);
                }
                bits_remaining -= 1;
                source_bit -= 1;
                current_bit -= 1;
                if current_bit < 0 {
                    current_byte += 1;
                    current_bit = 7;
                }
            }
            Ok(())
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_data_message() -> MessageDef {
        let mut msg = MessageDef::new(0x7E8, "EngineData", 8);
        msg.add_signal(
            SignalDef::new("EngineSpeed", 0, 16, ByteOrder::LittleEndian)
                .with_factor_offset(0.25, 0.0)
                .with_range(0.0, 16383.75)
                .with_unit("rpm"),
        );
        msg.add_signal(
            SignalDef::new("Throttle", 16, 8, ByteOrder::LittleEndian)
                .with_factor_offset(0.392157, 0.0)
                .with_range(0.0, 100.0)
                .with_unit("%"),
        );
        msg.add_signal(
            SignalDef::new("CoolantTemp", 24, 8, ByteOrder::LittleEndian)
                .with_factor_offset(1.0, -40.0)
                .with_range(-40.0, 215.0)
                .with_unit("degC")
                .signed(),
        );
        msg
    }

    fn make_decoder() -> DbcSignalDecoder {
        let mut dec = DbcSignalDecoder::new();
        dec.add_message(engine_data_message());
        dec
    }

    // ═══ SignalDef tests ═════════════════════════════════

    #[test]
    fn test_signal_def_creation() {
        let sig = SignalDef::new("Speed", 0, 16, ByteOrder::LittleEndian);
        assert_eq!(sig.name, "Speed");
        assert_eq!(sig.start_bit, 0);
        assert_eq!(sig.bit_length, 16);
        assert_eq!(sig.byte_order, ByteOrder::LittleEndian);
        assert_eq!(sig.value_sign, ValueSign::Unsigned);
    }

    #[test]
    fn test_raw_to_physical() {
        let sig = SignalDef::new("S", 0, 16, ByteOrder::LittleEndian).with_factor_offset(0.25, 0.0);
        assert!((sig.raw_to_physical(4000) - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_raw_to_physical_with_offset() {
        let sig = SignalDef::new("S", 0, 8, ByteOrder::LittleEndian).with_factor_offset(1.0, -40.0);
        assert!((sig.raw_to_physical(80) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_physical_to_raw() {
        let sig = SignalDef::new("S", 0, 16, ByteOrder::LittleEndian).with_factor_offset(0.25, 0.0);
        assert_eq!(sig.physical_to_raw(1000.0), 4000);
    }

    #[test]
    fn test_physical_to_raw_zero_factor() {
        let sig = SignalDef::new("S", 0, 8, ByteOrder::LittleEndian).with_factor_offset(0.0, 0.0);
        assert_eq!(sig.physical_to_raw(100.0), 0);
    }

    #[test]
    fn test_in_range() {
        let sig = SignalDef::new("S", 0, 16, ByteOrder::LittleEndian).with_range(0.0, 100.0);
        assert!(sig.in_range(50.0));
        assert!(sig.in_range(0.0));
        assert!(sig.in_range(100.0));
        assert!(!sig.in_range(-1.0));
        assert!(!sig.in_range(101.0));
    }

    #[test]
    fn test_in_range_no_bounds() {
        let sig = SignalDef::new("S", 0, 16, ByteOrder::LittleEndian);
        // min == max == 0 => no range check
        assert!(sig.in_range(9999.0));
    }

    // ═══ MessageDef tests ════════════════════════════════

    #[test]
    fn test_message_def_creation() {
        let msg = MessageDef::new(0x100, "TestMsg", 8);
        assert_eq!(msg.message_id, 0x100);
        assert_eq!(msg.dlc, 8);
        assert!(msg.signals.is_empty());
    }

    #[test]
    fn test_message_get_signal() {
        let msg = engine_data_message();
        assert!(msg.get_signal("EngineSpeed").is_some());
        assert!(msg.get_signal("Nonexistent").is_none());
    }

    // ═══ Decoder construction tests ══════════════════════

    #[test]
    fn test_decoder_new() {
        let dec = DbcSignalDecoder::new();
        assert_eq!(dec.message_count(), 0);
    }

    #[test]
    fn test_decoder_default() {
        let dec = DbcSignalDecoder::default();
        assert_eq!(dec.message_count(), 0);
    }

    #[test]
    fn test_decoder_from_messages() {
        let dec = DbcSignalDecoder::from_messages(vec![engine_data_message()]);
        assert_eq!(dec.message_count(), 1);
    }

    #[test]
    fn test_decoder_total_signals() {
        let dec = make_decoder();
        assert_eq!(dec.total_signal_count(), 3);
    }

    #[test]
    fn test_decoder_get_message() {
        let dec = make_decoder();
        assert!(dec.get_message(0x7E8).is_some());
        assert!(dec.get_message(0x000).is_none());
    }

    // ═══ Little-endian decoding tests ════════════════════

    #[test]
    fn test_decode_le_16bit() {
        let dec = make_decoder();
        // EngineSpeed: start=0, len=16, LE, factor=0.25
        // Raw value 4000 = 0x0FA0 => LE bytes: [0xA0, 0x0F]
        let payload = [0xA0, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = dec.decode_signal(0x7E8, "EngineSpeed", &payload);
        assert!(result.is_ok());
        let decoded = result.expect("decode should succeed");
        assert_eq!(decoded.raw_value, 4000);
        assert!((decoded.physical_value - 1000.0).abs() < 1e-6);
        assert_eq!(decoded.unit, "rpm");
    }

    #[test]
    fn test_decode_le_8bit() {
        let dec = make_decoder();
        // Throttle: start=16, len=8, LE, factor=0.392157
        // Raw=255 => 255 * 0.392157 = ~100.0
        let payload = [0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = dec.decode_signal(0x7E8, "Throttle", &payload);
        assert!(result.is_ok());
        let decoded = result.expect("decode should succeed");
        assert_eq!(decoded.raw_value, 255);
        assert!((decoded.physical_value - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_decode_le_with_offset() {
        let dec = make_decoder();
        // CoolantTemp: start=24, len=8, LE, factor=1.0, offset=-40, signed
        // Raw=120 => 120 - 40 = 80 degC
        let payload = [0x00, 0x00, 0x00, 120, 0x00, 0x00, 0x00, 0x00];
        let result = dec.decode_signal(0x7E8, "CoolantTemp", &payload);
        assert!(result.is_ok());
        let decoded = result.expect("decode should succeed");
        assert!((decoded.physical_value - 80.0).abs() < 1e-6);
    }

    // ═══ Big-endian decoding tests ═══════════════════════

    #[test]
    fn test_decode_be_16bit() {
        let mut dec = DbcSignalDecoder::new();
        let mut msg = MessageDef::new(0x200, "TestBE", 8);
        msg.add_signal(
            SignalDef::new("BESignal", 7, 16, ByteOrder::BigEndian).with_factor_offset(1.0, 0.0),
        );
        dec.add_message(msg);

        // Big-endian 16-bit at start_bit=7: MSB at bit7 of byte0
        // Value 0x1234: byte0 = 0x12, byte1 = 0x34
        let payload = [0x12, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = dec.decode_signal(0x200, "BESignal", &payload);
        assert!(result.is_ok());
        let decoded = result.expect("decode should succeed");
        assert_eq!(decoded.raw_value, 0x1234);
    }

    // ═══ Batch decode tests ══════════════════════════════

    #[test]
    fn test_decode_frame_all_signals() {
        let dec = make_decoder();
        let payload = [0xA0, 0x0F, 0x80, 0x78, 0x00, 0x00, 0x00, 0x00];
        let result = dec.decode_frame(0x7E8, &payload);
        assert!(result.is_ok());
        let signals = result.expect("decode should succeed");
        assert_eq!(signals.len(), 3);
    }

    // ═══ Error handling tests ════════════════════════════

    #[test]
    fn test_unknown_message() {
        let dec = make_decoder();
        let result = dec.decode_frame(0x999, &[0; 8]);
        assert!(result.is_err());
        match result {
            Err(DecodeError::UnknownMessage(id)) => assert_eq!(id, 0x999),
            _ => panic!("expected UnknownMessage error"),
        }
    }

    #[test]
    fn test_unknown_signal() {
        let dec = make_decoder();
        let result = dec.decode_signal(0x7E8, "NonExistent", &[0; 8]);
        assert!(result.is_err());
        match result {
            Err(DecodeError::UnknownSignal(name)) => assert_eq!(name, "NonExistent"),
            _ => panic!("expected UnknownSignal error"),
        }
    }

    #[test]
    fn test_payload_too_short() {
        let dec = make_decoder();
        let result = dec.decode_signal(0x7E8, "CoolantTemp", &[0; 2]);
        assert!(result.is_err());
    }

    // ═══ Encode/decode roundtrip tests ═══════════════════

    #[test]
    fn test_encode_decode_roundtrip_le() {
        let dec = make_decoder();
        let mut payload = [0u8; 8];
        let encode_result = dec.encode_signal(0x7E8, "EngineSpeed", 1000.0, &mut payload);
        assert!(encode_result.is_ok());

        let decoded = dec
            .decode_signal(0x7E8, "EngineSpeed", &payload)
            .expect("decode should succeed");
        assert!((decoded.physical_value - 1000.0).abs() < 0.5);
    }

    #[test]
    fn test_encode_unknown_message() {
        let dec = make_decoder();
        let mut payload = [0u8; 8];
        let result = dec.encode_signal(0x999, "Foo", 0.0, &mut payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_unknown_signal() {
        let dec = make_decoder();
        let mut payload = [0u8; 8];
        let result = dec.encode_signal(0x7E8, "Nonexistent", 0.0, &mut payload);
        assert!(result.is_err());
    }

    // ═══ Signed value tests ══════════════════════════════

    #[test]
    fn test_signed_negative_value() {
        let mut dec = DbcSignalDecoder::new();
        let mut msg = MessageDef::new(0x300, "SignedTest", 8);
        msg.add_signal(
            SignalDef::new("SignedVal", 0, 8, ByteOrder::LittleEndian)
                .with_factor_offset(1.0, 0.0)
                .signed(),
        );
        dec.add_message(msg);

        // -1 as unsigned 8-bit = 0xFF
        let payload = [0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = dec
            .decode_signal(0x300, "SignedVal", &payload)
            .expect("decode should succeed");
        assert_eq!(decoded.raw_value, -1);
        assert!((decoded.physical_value - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_signed_positive_value() {
        let mut dec = DbcSignalDecoder::new();
        let mut msg = MessageDef::new(0x300, "SignedTest", 8);
        msg.add_signal(
            SignalDef::new("SignedVal", 0, 8, ByteOrder::LittleEndian)
                .with_factor_offset(1.0, 0.0)
                .signed(),
        );
        dec.add_message(msg);

        let payload = [0x7F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = dec
            .decode_signal(0x300, "SignedVal", &payload)
            .expect("decode should succeed");
        assert_eq!(decoded.raw_value, 127);
    }

    // ═══ DecodeError display tests ═══════════════════════

    #[test]
    fn test_decode_error_display() {
        let err = DecodeError::UnknownMessage(0x123);
        let msg = format!("{err}");
        assert!(msg.contains("0x123"));
    }

    #[test]
    fn test_decode_error_display_signal() {
        let err = DecodeError::UnknownSignal("Foo".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Foo"));
    }

    #[test]
    fn test_decode_error_display_payload() {
        let err = DecodeError::PayloadTooShort {
            expected: 8,
            actual: 2,
        };
        let msg = format!("{err}");
        assert!(msg.contains("8"));
        assert!(msg.contains("2"));
    }

    // ═══ Range checking in decoded output ════════════════

    #[test]
    fn test_decoded_signal_in_range() {
        let dec = make_decoder();
        let payload = [0xA0, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = dec
            .decode_signal(0x7E8, "EngineSpeed", &payload)
            .expect("decode should succeed");
        assert!(decoded.in_range); // 1000 rpm within [0, 16383.75]
    }

    #[test]
    fn test_zero_payload_decode() {
        let dec = make_decoder();
        let payload = [0u8; 8];
        let result = dec.decode_frame(0x7E8, &payload);
        assert!(result.is_ok());
        let signals = result.expect("decode should succeed");
        // All zero raw => physical values based on offset
        for sig in &signals {
            if sig.name == "CoolantTemp" {
                // offset = -40
                assert!((sig.physical_value - (-40.0)).abs() < 1e-6);
            } else {
                assert!((sig.physical_value - 0.0).abs() < 1e-6);
            }
        }
    }
}
