//! Signal extraction and decoding from CAN frames
//!
//! This module provides functionality to:
//! - Extract raw bit values from CAN frame data
//! - Handle Intel (little-endian) and Motorola (big-endian) byte ordering
//! - Convert between raw integer values and physical values
//! - Support signed and unsigned signal types
//! - Decode multiplexed signals
//!
//! # Signal Bit Layout
//!
//! ## Intel Byte Order (Little-Endian)
//! Start bit is the LSB position. Bits are numbered continuously across bytes:
//! ```text
//! Byte:    0         1         2
//! Bits: 7..0     15..8     23..16
//!       ├─────────┼─────────┼─────────┤
//!       │   LSB   │         │   MSB   │
//! ```
//!
//! ## Motorola Byte Order (Big-Endian)
//! Start bit is the MSB position in the first byte. Signal spans down across bytes:
//! ```text
//! Byte:    0         1         2
//! Bits: 7..0     15..8     23..16
//!       ├─────────┼─────────┼─────────┤
//!       │   MSB   │         │   LSB   │
//! ```

use crate::dbc::parser::{
    ByteOrder, DbcDatabase, DbcMessage, DbcSignal, MultiplexerType, ValueType,
};
use crate::error::{CanbusError, CanbusResult};
use std::collections::HashMap;
use thiserror::Error;

/// Error during signal extraction
#[derive(Error, Debug)]
pub enum SignalExtractionError {
    /// Signal data extends beyond frame data
    #[error("Signal '{signal}' requires {required} bytes but frame has only {available}")]
    InsufficientData {
        /// Name of the signal being extracted
        signal: String,
        /// Number of bytes required to extract the signal
        required: usize,
        /// Number of bytes available in the frame
        available: usize,
    },

    /// Multiplexer value doesn't match expected
    #[error("Signal '{signal}' expects multiplexer value {expected} but got {actual}")]
    MultiplexerMismatch {
        /// Name of the multiplexed signal
        signal: String,
        /// Expected multiplexer value for this signal
        expected: u32,
        /// Actual multiplexer value in the frame
        actual: u32,
    },

    /// No multiplexer signal found in message
    #[error("Message {msg_id} has multiplexed signals but no multiplexer signal")]
    MissingMultiplexer {
        /// CAN message ID that is missing a multiplexer
        msg_id: u32,
    },
}

/// Decoded signal value with metadata
#[derive(Debug, Clone)]
pub struct DecodedSignalValue {
    /// Signal name
    pub name: String,
    /// Raw integer value extracted from frame
    pub raw_value: i64,
    /// Physical value after applying factor and offset
    pub physical_value: f64,
    /// Unit string (if available)
    pub unit: String,
    /// Value description/enum name (if available)
    pub description: Option<String>,
}

impl DecodedSignalValue {
    /// Create a new decoded signal value
    pub fn new(signal: &DbcSignal, raw_value: i64) -> Self {
        let physical_value = signal.to_physical(raw_value);
        let description = signal
            .get_value_description(raw_value)
            .map(|s| s.to_string());

        Self {
            name: signal.name.clone(),
            raw_value,
            physical_value,
            unit: signal.unit.clone(),
            description,
        }
    }
}

/// Generic signal value enum for type-safe value handling
#[derive(Debug, Clone, PartialEq)]
pub enum SignalValue {
    /// Unsigned integer (for unsigned signals)
    Unsigned(u64),
    /// Signed integer (for signed signals)
    Signed(i64),
    /// Physical value after scaling
    Physical(f64),
    /// Enumeration value (raw + description)
    Enum {
        /// Raw integer value
        raw: i64,
        /// Human-readable description
        description: String,
    },
}

impl SignalValue {
    /// Get as signed integer
    pub fn as_signed(&self) -> Option<i64> {
        match self {
            SignalValue::Signed(v) => Some(*v),
            SignalValue::Unsigned(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Get as unsigned integer
    pub fn as_unsigned(&self) -> Option<u64> {
        match self {
            SignalValue::Unsigned(v) => Some(*v),
            SignalValue::Signed(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Get as float (physical value)
    pub fn as_float(&self) -> Option<f64> {
        match self {
            SignalValue::Physical(v) => Some(*v),
            SignalValue::Unsigned(v) => Some(*v as f64),
            SignalValue::Signed(v) => Some(*v as f64),
            SignalValue::Enum { raw, .. } => Some(*raw as f64),
        }
    }
}

/// Signal decoder for extracting signals from CAN frames
pub struct SignalDecoder<'a> {
    /// Reference to DBC database (for future extensions)
    #[allow(dead_code)]
    database: &'a DbcDatabase,
    /// Cache of message ID to signals for fast lookup
    message_cache: HashMap<u32, &'a DbcMessage>,
}

impl<'a> SignalDecoder<'a> {
    /// Create a new signal decoder from a DBC database
    pub fn new(database: &'a DbcDatabase) -> Self {
        let mut message_cache = HashMap::new();
        for msg in &database.messages {
            message_cache.insert(msg.id, msg);
        }

        Self {
            database,
            message_cache,
        }
    }

    /// Decode all signals from a CAN frame
    ///
    /// Returns a map of signal names to decoded values
    pub fn decode_message(
        &self,
        message_id: u32,
        data: &[u8],
    ) -> CanbusResult<HashMap<String, DecodedSignalValue>> {
        let message = self
            .message_cache
            .get(&message_id)
            .ok_or_else(|| CanbusError::SignalNotFound(format!("Message ID {:X}", message_id)))?;

        let mut results = HashMap::new();

        // Find multiplexer value if message has multiplexed signals
        let mux_value = self.find_multiplexer_value(message, data)?;

        for signal in &message.signals {
            // Skip multiplexed signals that don't match current mux value
            if let MultiplexerType::Multiplexed(expected) = signal.multiplexer {
                if let Some(actual) = mux_value {
                    if expected != actual {
                        continue;
                    }
                } else {
                    // No multiplexer found but signal is multiplexed
                    continue;
                }
            }

            match self.extract_signal(signal, data) {
                Ok(raw_value) => {
                    let decoded = DecodedSignalValue::new(signal, raw_value);
                    results.insert(signal.name.clone(), decoded);
                }
                Err(e) => {
                    tracing::warn!("Failed to extract signal '{}': {}", signal.name, e);
                }
            }
        }

        Ok(results)
    }

    /// Decode a single signal by name
    pub fn decode_signal(
        &self,
        message_id: u32,
        signal_name: &str,
        data: &[u8],
    ) -> CanbusResult<DecodedSignalValue> {
        let message = self
            .message_cache
            .get(&message_id)
            .ok_or_else(|| CanbusError::SignalNotFound(format!("Message ID {:X}", message_id)))?;

        let signal = message
            .get_signal(signal_name)
            .ok_or_else(|| CanbusError::SignalNotFound(signal_name.to_string()))?;

        let raw_value = self.extract_signal(signal, data)?;
        Ok(DecodedSignalValue::new(signal, raw_value))
    }

    /// Extract raw value from CAN frame data for a signal
    pub fn extract_signal(&self, signal: &DbcSignal, data: &[u8]) -> CanbusResult<i64> {
        match signal.byte_order {
            ByteOrder::LittleEndian => self.extract_intel(signal, data),
            ByteOrder::BigEndian => self.extract_motorola(signal, data),
        }
    }

    /// Extract Intel (little-endian) signal
    ///
    /// Start bit is the LSB position. Bits are numbered 0-7 in each byte,
    /// with byte 0 containing bits 0-7, byte 1 containing bits 8-15, etc.
    fn extract_intel(&self, signal: &DbcSignal, data: &[u8]) -> CanbusResult<i64> {
        let start_bit = signal.start_bit as usize;
        let bit_length = signal.bit_length as usize;

        // Calculate byte range needed
        let end_bit = start_bit + bit_length - 1;
        let start_byte = start_bit / 8;
        let end_byte = end_bit / 8;

        if end_byte >= data.len() {
            return Err(CanbusError::Config(format!(
                "Signal '{}' requires {} bytes but frame has only {}",
                signal.name,
                end_byte + 1,
                data.len()
            )));
        }

        // Extract bits into u64
        let mut raw: u64 = 0;
        let mut bit_pos = 0;

        #[allow(clippy::needless_range_loop)]
        for byte_idx in start_byte..=end_byte {
            let byte = data[byte_idx] as u64;

            // Calculate bit range within this byte
            let byte_start_bit = if byte_idx == start_byte {
                start_bit % 8
            } else {
                0
            };

            let byte_end_bit = if byte_idx == end_byte { end_bit % 8 } else { 7 };

            // Extract bits from this byte
            for bit in byte_start_bit..=byte_end_bit {
                if (byte >> bit) & 1 == 1 {
                    raw |= 1 << bit_pos;
                }
                bit_pos += 1;
            }
        }

        // Convert to signed if needed
        Ok(self.apply_sign(raw, bit_length, signal.value_type))
    }

    /// Extract Motorola (big-endian) signal
    ///
    /// Start bit is the MSB position in the start byte.
    /// Signal extends down and across bytes to lower bit positions.
    fn extract_motorola(&self, signal: &DbcSignal, data: &[u8]) -> CanbusResult<i64> {
        let start_bit = signal.start_bit as usize;
        let bit_length = signal.bit_length as usize;

        // For Motorola, start_bit is the MSB position
        // Convert to byte and bit position
        let start_byte = start_bit / 8;
        let start_bit_in_byte = start_bit % 8;

        // Calculate total bytes needed
        let bits_in_first_byte = start_bit_in_byte + 1;
        let remaining_bits = bit_length.saturating_sub(bits_in_first_byte);
        let additional_bytes = (remaining_bits + 7) / 8;
        let end_byte = start_byte + additional_bytes;

        if end_byte >= data.len() {
            return Err(CanbusError::Config(format!(
                "Signal '{}' requires {} bytes but frame has only {}",
                signal.name,
                end_byte + 1,
                data.len()
            )));
        }

        // Extract bits from MSB to LSB order
        let mut raw: u64 = 0;
        let mut bits_remaining = bit_length;
        let mut current_byte = start_byte;
        let mut current_bit = start_bit_in_byte as i32;

        while bits_remaining > 0 && current_byte < data.len() {
            let byte = data[current_byte] as u64;

            // Extract bits from current byte (MSB first for Motorola)
            while current_bit >= 0 && bits_remaining > 0 {
                raw <<= 1;
                if (byte >> current_bit) & 1 == 1 {
                    raw |= 1;
                }
                current_bit -= 1;
                bits_remaining -= 1;
            }

            // Move to next byte
            current_byte += 1;
            current_bit = 7;
        }

        // Convert to signed if needed
        Ok(self.apply_sign(raw, bit_length, signal.value_type))
    }

    /// Apply sign extension for signed signals
    fn apply_sign(&self, raw: u64, bit_length: usize, value_type: ValueType) -> i64 {
        match value_type {
            ValueType::Unsigned => raw as i64,
            ValueType::Signed => {
                // Check if MSB is set (negative number)
                let sign_bit = 1u64 << (bit_length - 1);
                if raw & sign_bit != 0 {
                    // Sign extend: set all upper bits to 1
                    let mask = !((1u64 << bit_length) - 1);
                    (raw | mask) as i64
                } else {
                    raw as i64
                }
            }
        }
    }

    /// Find multiplexer signal value in message
    fn find_multiplexer_value(
        &self,
        message: &DbcMessage,
        data: &[u8],
    ) -> CanbusResult<Option<u32>> {
        // Check if any signals are multiplexed
        let has_multiplexed = message
            .signals
            .iter()
            .any(|s| matches!(s.multiplexer, MultiplexerType::Multiplexed(_)));

        if !has_multiplexed {
            return Ok(None);
        }

        // Find the multiplexer signal
        let mux_signal = message
            .signals
            .iter()
            .find(|s| s.multiplexer == MultiplexerType::Multiplexer);

        match mux_signal {
            Some(signal) => {
                let value = self.extract_signal(signal, data)?;
                Ok(Some(value as u32))
            }
            None => Ok(None), // No multiplexer found - decode all non-multiplexed signals
        }
    }

    /// Get decoded physical value directly
    pub fn get_physical_value(
        &self,
        message_id: u32,
        signal_name: &str,
        data: &[u8],
    ) -> CanbusResult<f64> {
        let decoded = self.decode_signal(message_id, signal_name, data)?;
        Ok(decoded.physical_value)
    }

    /// Decode all signals and return as generic SignalValue map
    pub fn decode_to_values(
        &self,
        message_id: u32,
        data: &[u8],
    ) -> CanbusResult<HashMap<String, SignalValue>> {
        let decoded = self.decode_message(message_id, data)?;

        Ok(decoded
            .into_iter()
            .map(|(name, value)| {
                let signal_value = if let Some(desc) = value.description {
                    SignalValue::Enum {
                        raw: value.raw_value,
                        description: desc,
                    }
                } else {
                    SignalValue::Physical(value.physical_value)
                };
                (name, signal_value)
            })
            .collect())
    }
}

/// Signal encoder for creating CAN frame data from signal values
pub struct SignalEncoder<'a> {
    /// Reference to DBC database (for future extensions)
    #[allow(dead_code)]
    database: &'a DbcDatabase,
    /// Cache of message ID to signals for fast lookup
    message_cache: HashMap<u32, &'a DbcMessage>,
}

impl<'a> SignalEncoder<'a> {
    /// Create a new signal encoder from a DBC database
    pub fn new(database: &'a DbcDatabase) -> Self {
        let mut message_cache = HashMap::new();
        for msg in &database.messages {
            message_cache.insert(msg.id, msg);
        }

        Self {
            database,
            message_cache,
        }
    }

    /// Encode signal values into CAN frame data
    ///
    /// Takes a map of signal names to physical values and creates frame data
    pub fn encode_message(
        &self,
        message_id: u32,
        values: &HashMap<String, f64>,
    ) -> CanbusResult<Vec<u8>> {
        let message = self
            .message_cache
            .get(&message_id)
            .ok_or_else(|| CanbusError::SignalNotFound(format!("Message ID {:X}", message_id)))?;

        let mut data = vec![0u8; message.dlc as usize];

        for signal in &message.signals {
            if let Some(&physical_value) = values.get(&signal.name) {
                let raw_value = signal.to_raw(physical_value);
                self.encode_signal(signal, raw_value, &mut data)?;
            }
        }

        Ok(data)
    }

    /// Encode a single signal into frame data
    pub fn encode_signal(
        &self,
        signal: &DbcSignal,
        raw_value: i64,
        data: &mut [u8],
    ) -> CanbusResult<()> {
        match signal.byte_order {
            ByteOrder::LittleEndian => self.encode_intel(signal, raw_value, data),
            ByteOrder::BigEndian => self.encode_motorola(signal, raw_value, data),
        }
    }

    /// Encode Intel (little-endian) signal
    fn encode_intel(
        &self,
        signal: &DbcSignal,
        raw_value: i64,
        data: &mut [u8],
    ) -> CanbusResult<()> {
        let start_bit = signal.start_bit as usize;
        let bit_length = signal.bit_length as usize;

        // Mask the value to the signal's bit length
        let mask = if bit_length >= 64 {
            u64::MAX
        } else {
            (1u64 << bit_length) - 1
        };
        let value = (raw_value as u64) & mask;

        // Calculate byte range
        let end_bit = start_bit + bit_length - 1;
        let start_byte = start_bit / 8;
        let end_byte = end_bit / 8;

        if end_byte >= data.len() {
            return Err(CanbusError::Config(format!(
                "Signal '{}' requires {} bytes but buffer has only {}",
                signal.name,
                end_byte + 1,
                data.len()
            )));
        }

        // Write bits
        let mut bit_pos = 0;

        #[allow(clippy::needless_range_loop)]
        for byte_idx in start_byte..=end_byte {
            let byte_start_bit = if byte_idx == start_byte {
                start_bit % 8
            } else {
                0
            };

            let byte_end_bit = if byte_idx == end_byte { end_bit % 8 } else { 7 };

            for bit in byte_start_bit..=byte_end_bit {
                if (value >> bit_pos) & 1 == 1 {
                    data[byte_idx] |= 1 << bit;
                } else {
                    data[byte_idx] &= !(1 << bit);
                }
                bit_pos += 1;
            }
        }

        Ok(())
    }

    /// Encode Motorola (big-endian) signal
    fn encode_motorola(
        &self,
        signal: &DbcSignal,
        raw_value: i64,
        data: &mut [u8],
    ) -> CanbusResult<()> {
        let start_bit = signal.start_bit as usize;
        let bit_length = signal.bit_length as usize;

        // Mask the value to the signal's bit length
        let mask = if bit_length >= 64 {
            u64::MAX
        } else {
            (1u64 << bit_length) - 1
        };
        let value = (raw_value as u64) & mask;

        let start_byte = start_bit / 8;
        let start_bit_in_byte = start_bit % 8;

        // Calculate bytes needed
        let bits_in_first_byte = start_bit_in_byte + 1;
        let remaining_bits = bit_length.saturating_sub(bits_in_first_byte);
        let additional_bytes = (remaining_bits + 7) / 8;
        let end_byte = start_byte + additional_bytes;

        if end_byte >= data.len() {
            return Err(CanbusError::Config(format!(
                "Signal '{}' requires {} bytes but buffer has only {}",
                signal.name,
                end_byte + 1,
                data.len()
            )));
        }

        // Write bits from MSB to LSB
        let mut bits_remaining = bit_length;
        let mut current_byte = start_byte;
        let mut current_bit = start_bit_in_byte as i32;
        let mut value_bit = bit_length as i32 - 1; // Start from MSB of value

        while bits_remaining > 0 && current_byte < data.len() {
            while current_bit >= 0 && bits_remaining > 0 && value_bit >= 0 {
                if (value >> value_bit) & 1 == 1 {
                    data[current_byte] |= 1 << current_bit;
                } else {
                    data[current_byte] &= !(1 << current_bit);
                }
                current_bit -= 1;
                value_bit -= 1;
                bits_remaining -= 1;
            }

            current_byte += 1;
            current_bit = 7;
        }

        Ok(())
    }

    /// Encode a single signal by name with physical value
    pub fn encode_signal_by_name(
        &self,
        message_id: u32,
        signal_name: &str,
        physical_value: f64,
        data: &mut [u8],
    ) -> CanbusResult<()> {
        let message = self
            .message_cache
            .get(&message_id)
            .ok_or_else(|| CanbusError::SignalNotFound(format!("Message ID {:X}", message_id)))?;

        let signal = message
            .get_signal(signal_name)
            .ok_or_else(|| CanbusError::SignalNotFound(signal_name.to_string()))?;

        let raw_value = signal.to_raw(physical_value);
        self.encode_signal(signal, raw_value, data)
    }
}

/// Utility functions for bit manipulation
pub mod bits {
    /// Extract bits from a byte array (Intel/little-endian)
    pub fn extract_bits_intel(data: &[u8], start_bit: usize, bit_length: usize) -> u64 {
        let mut result: u64 = 0;
        let end_bit = start_bit + bit_length;

        for bit in start_bit..end_bit {
            let byte_idx = bit / 8;
            let bit_in_byte = bit % 8;

            if byte_idx < data.len() && (data[byte_idx] >> bit_in_byte) & 1 == 1 {
                result |= 1 << (bit - start_bit);
            }
        }

        result
    }

    /// Extract bits from a byte array (Motorola/big-endian)
    pub fn extract_bits_motorola(data: &[u8], start_bit: usize, bit_length: usize) -> u64 {
        let start_byte = start_bit / 8;
        let start_bit_in_byte = start_bit % 8;

        let mut result: u64 = 0;
        let mut bits_remaining = bit_length;
        let mut current_byte = start_byte;
        let mut current_bit = start_bit_in_byte as i32;

        while bits_remaining > 0 && current_byte < data.len() {
            while current_bit >= 0 && bits_remaining > 0 {
                result <<= 1;
                if (data[current_byte] >> current_bit) & 1 == 1 {
                    result |= 1;
                }
                current_bit -= 1;
                bits_remaining -= 1;
            }

            current_byte += 1;
            current_bit = 7;
        }

        result
    }

    /// Set bits in a byte array (Intel/little-endian)
    pub fn set_bits_intel(data: &mut [u8], start_bit: usize, bit_length: usize, value: u64) {
        let end_bit = start_bit + bit_length;

        for bit in start_bit..end_bit {
            let byte_idx = bit / 8;
            let bit_in_byte = bit % 8;
            let value_bit = bit - start_bit;

            if byte_idx < data.len() {
                if (value >> value_bit) & 1 == 1 {
                    data[byte_idx] |= 1 << bit_in_byte;
                } else {
                    data[byte_idx] &= !(1 << bit_in_byte);
                }
            }
        }
    }

    /// Set bits in a byte array (Motorola/big-endian)
    pub fn set_bits_motorola(data: &mut [u8], start_bit: usize, bit_length: usize, value: u64) {
        let start_byte = start_bit / 8;
        let start_bit_in_byte = start_bit % 8;

        let mut bits_remaining = bit_length;
        let mut current_byte = start_byte;
        let mut current_bit = start_bit_in_byte as i32;
        let mut value_bit = bit_length as i32 - 1;

        while bits_remaining > 0 && current_byte < data.len() {
            while current_bit >= 0 && bits_remaining > 0 && value_bit >= 0 {
                if (value >> value_bit) & 1 == 1 {
                    data[current_byte] |= 1 << current_bit;
                } else {
                    data[current_byte] &= !(1 << current_bit);
                }
                current_bit -= 1;
                value_bit -= 1;
                bits_remaining -= 1;
            }

            current_byte += 1;
            current_bit = 7;
        }
    }

    /// Sign extend a value
    pub fn sign_extend(value: u64, bit_length: usize) -> i64 {
        let sign_bit = 1u64 << (bit_length - 1);
        if value & sign_bit != 0 {
            let mask = !((1u64 << bit_length) - 1);
            (value | mask) as i64
        } else {
            value as i64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dbc::parser::parse_dbc;

    const TEST_DBC: &str = r#"
VERSION ""

BU_: Engine Dashboard

BO_ 2024 EngineData: 8 Engine
 SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
 SG_ EngineTemp : 16|8@1+ (1,-40) [-40|215] "degC" Dashboard
 SG_ ThrottlePos : 24|8@1+ (0.392157,0) [0|100] "%" Dashboard

BO_ 100 TestSigned: 8 Engine
 SG_ SignedValue : 0|16@1- (1,0) [-32768|32767] "" Dashboard

BO_ 200 BigEndianMsg: 8 Engine
 SG_ BigEndianSig : 7|16@0+ (1,0) [0|65535] "" Dashboard

BO_ 300 MuxMsg: 8 Engine
 SG_ MuxSwitch M : 0|8@1+ (1,0) [0|255] "" Dashboard
 SG_ Signal_0 m0 : 8|16@1+ (1,0) [0|65535] "" Dashboard
 SG_ Signal_1 m1 : 8|16@1+ (1,0) [0|65535] "" Dashboard
 SG_ Signal_2 m2 : 8|16@1+ (0.1,0) [0|6553.5] "" Dashboard
"#;

    #[test]
    fn test_decode_intel_signal() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let decoder = SignalDecoder::new(&db);

        // EngineSpeed at 16000 raw = 2000 RPM
        let data = [0x00, 0x3E, 0x85, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = decoder.decode_signal(2024, "EngineSpeed", &data).unwrap();

        // Raw value should be 0x3E00 = 15872
        assert_eq!(decoded.raw_value, 15872);
        // Physical = 15872 * 0.125 = 1984 RPM
        assert!((decoded.physical_value - 1984.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_with_offset() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let decoder = SignalDecoder::new(&db);

        // EngineTemp at offset=16, raw=125 -> physical = 125 - 40 = 85°C
        let data = [0x00, 0x00, 0x7D, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = decoder.decode_signal(2024, "EngineTemp", &data).unwrap();

        assert_eq!(decoded.raw_value, 125);
        assert!((decoded.physical_value - 85.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_signed_signal() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let decoder = SignalDecoder::new(&db);

        // Test positive value
        let data_pos = [0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = decoder
            .decode_signal(100, "SignedValue", &data_pos)
            .unwrap();
        assert_eq!(decoded.raw_value, 256);

        // Test negative value: 0xFFFF = -1 as signed 16-bit
        let data_neg = [0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = decoder
            .decode_signal(100, "SignedValue", &data_neg)
            .unwrap();
        assert_eq!(decoded.raw_value, -1);

        // Test -100: 0xFF9C
        let data_neg100 = [0x9C, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = decoder
            .decode_signal(100, "SignedValue", &data_neg100)
            .unwrap();
        assert_eq!(decoded.raw_value, -100);
    }

    #[test]
    fn test_decode_big_endian() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let decoder = SignalDecoder::new(&db);

        // Big-endian signal starting at bit 7, spanning bytes 0-1
        // Value 0x1234 should be in data[0]=0x12, data[1]=0x34
        let data = [0x12, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let decoded = decoder.decode_signal(200, "BigEndianSig", &data).unwrap();

        assert_eq!(decoded.raw_value, 0x1234);
    }

    #[test]
    fn test_decode_full_message() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let decoder = SignalDecoder::new(&db);

        // Set up frame with known values
        let data = [
            0x00, 0x40, // EngineSpeed = 0x4000 = 16384 -> 2048 RPM
            0x96, // EngineTemp = 150 -> 110°C
            0x80, // ThrottlePos = 128 -> ~50.2%
            0x00, 0x00, 0x00, 0x00,
        ];

        let values = decoder.decode_message(2024, &data).unwrap();

        assert!(values.contains_key("EngineSpeed"));
        assert!(values.contains_key("EngineTemp"));
        assert!(values.contains_key("ThrottlePos"));

        let speed = &values["EngineSpeed"];
        assert_eq!(speed.raw_value, 16384);
        assert!((speed.physical_value - 2048.0).abs() < 0.01);

        let temp = &values["EngineTemp"];
        assert_eq!(temp.raw_value, 150);
        assert!((temp.physical_value - 110.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_multiplexed() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let decoder = SignalDecoder::new(&db);

        // Mux value = 0, Signal_0 = 0x1234
        let data_mux0 = [0x00, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00];
        let values = decoder.decode_message(300, &data_mux0).unwrap();

        assert!(values.contains_key("MuxSwitch"));
        assert!(values.contains_key("Signal_0"));
        assert!(!values.contains_key("Signal_1")); // Not active
        assert!(!values.contains_key("Signal_2")); // Not active

        // Mux value = 2, Signal_2 = 0x2710 = 10000 -> 1000.0
        let data_mux2 = [0x02, 0x10, 0x27, 0x00, 0x00, 0x00, 0x00, 0x00];
        let values = decoder.decode_message(300, &data_mux2).unwrap();

        assert!(values.contains_key("Signal_2"));
        assert!(!values.contains_key("Signal_0"));
        assert!(!values.contains_key("Signal_1"));

        let sig2 = &values["Signal_2"];
        assert!((sig2.physical_value - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_encode_intel_signal() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let encoder = SignalEncoder::new(&db);

        let mut values = HashMap::new();
        values.insert("EngineSpeed".to_string(), 2048.0); // 2048 RPM -> raw 16384

        let data = encoder.encode_message(2024, &values).unwrap();

        // Should have 0x4000 in bytes 0-1
        assert_eq!(data[0], 0x00);
        assert_eq!(data[1], 0x40);
    }

    #[test]
    fn test_encode_with_offset() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let encoder = SignalEncoder::new(&db);

        let mut values = HashMap::new();
        values.insert("EngineTemp".to_string(), 85.0); // 85°C -> raw 125

        let data = encoder.encode_message(2024, &values).unwrap();

        // Temp at byte 2
        assert_eq!(data[2], 125);
    }

    #[test]
    fn test_roundtrip_encoding() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let encoder = SignalEncoder::new(&db);
        let decoder = SignalDecoder::new(&db);

        // Encode values
        let mut input = HashMap::new();
        input.insert("EngineSpeed".to_string(), 3500.0);
        input.insert("EngineTemp".to_string(), 90.0);
        input.insert("ThrottlePos".to_string(), 75.0);

        let data = encoder.encode_message(2024, &input).unwrap();

        // Decode back
        let output = decoder.decode_message(2024, &data).unwrap();

        // Check values match within tolerance
        assert!((output["EngineSpeed"].physical_value - 3500.0).abs() < 0.125);
        assert!((output["EngineTemp"].physical_value - 90.0).abs() < 1.0);
        assert!((output["ThrottlePos"].physical_value - 75.0).abs() < 0.5);
    }

    #[test]
    fn test_bits_utility_intel() {
        let data = [0xFF, 0x00, 0xAA, 0x55];

        // Extract bits 0-7 (first byte) = 0xFF
        assert_eq!(bits::extract_bits_intel(&data, 0, 8), 0xFF);

        // Extract bits 8-15 (second byte) = 0x00
        assert_eq!(bits::extract_bits_intel(&data, 8, 8), 0x00);

        // Extract bits 4-11 (spanning bytes 0-1) = 0x0F
        assert_eq!(bits::extract_bits_intel(&data, 4, 8), 0x0F);
    }

    #[test]
    fn test_bits_utility_motorola() {
        let data = [0x12, 0x34, 0x56, 0x78];

        // Extract 16-bit starting at bit 7 (MSB of byte 0)
        // Should be 0x1234
        assert_eq!(bits::extract_bits_motorola(&data, 7, 16), 0x1234);
    }

    #[test]
    fn test_sign_extend() {
        // 8-bit: 0xFF = -1
        assert_eq!(bits::sign_extend(0xFF, 8), -1);

        // 8-bit: 0x80 = -128
        assert_eq!(bits::sign_extend(0x80, 8), -128);

        // 8-bit: 0x7F = 127
        assert_eq!(bits::sign_extend(0x7F, 8), 127);

        // 16-bit: 0xFFFF = -1
        assert_eq!(bits::sign_extend(0xFFFF, 16), -1);

        // 16-bit: 0xFF9C = -100
        assert_eq!(bits::sign_extend(0xFF9C, 16), -100);
    }

    #[test]
    fn test_set_bits_intel() {
        let mut data = [0u8; 4];

        // Set bits 0-7 to 0xFF
        bits::set_bits_intel(&mut data, 0, 8, 0xFF);
        assert_eq!(data[0], 0xFF);

        // Set bits 4-11 to 0xAB
        let mut data2 = [0u8; 4];
        bits::set_bits_intel(&mut data2, 4, 8, 0xAB);
        assert_eq!(data2[0], 0xB0);
        assert_eq!(data2[1], 0x0A);
    }

    #[test]
    fn test_signal_value_conversions() {
        let unsigned = SignalValue::Unsigned(100);
        assert_eq!(unsigned.as_unsigned(), Some(100));
        assert_eq!(unsigned.as_signed(), Some(100));
        assert_eq!(unsigned.as_float(), Some(100.0));

        let signed = SignalValue::Signed(-50);
        assert_eq!(signed.as_signed(), Some(-50));
        assert_eq!(signed.as_unsigned(), None);
        assert_eq!(signed.as_float(), Some(-50.0));

        let physical = SignalValue::Physical(3.5);
        assert_eq!(physical.as_float(), Some(3.5));
    }
}
