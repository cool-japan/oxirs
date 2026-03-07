//! CAN frame types and parsing
//!
//! Supports both CAN 2.0 (11-bit/29-bit IDs, 8-byte payload)
//! and CAN FD (29-bit IDs, up to 64-byte payload).

use crate::error::{CanbusError, CanbusResult};
use serde::{Deserialize, Serialize};

/// CAN identifier (11-bit standard or 29-bit extended)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanId {
    /// Standard 11-bit CAN ID (0x000-0x7FF)
    Standard(u16),

    /// Extended 29-bit CAN ID (0x00000000-0x1FFFFFFF)
    Extended(u32),
}

impl CanId {
    /// Create standard CAN ID
    pub fn standard(id: u16) -> CanbusResult<Self> {
        if id > 0x7FF {
            return Err(CanbusError::InvalidCanId(id as u32));
        }
        Ok(Self::Standard(id))
    }

    /// Create extended CAN ID
    pub fn extended(id: u32) -> CanbusResult<Self> {
        if id > 0x1FFFFFFF {
            return Err(CanbusError::InvalidCanId(id));
        }
        Ok(Self::Extended(id))
    }

    /// Get raw ID value
    pub fn as_raw(&self) -> u32 {
        match self {
            Self::Standard(id) => *id as u32,
            Self::Extended(id) => *id,
        }
    }

    /// Check if extended ID
    pub fn is_extended(&self) -> bool {
        matches!(self, Self::Extended(_))
    }

    /// Extract J1939 Parameter Group Number (PGN) from extended CAN ID
    ///
    /// J1939 29-bit CAN ID format:
    /// - Priority (3 bits): bits 26-28
    /// - Reserved/DP (2 bits): bits 24-25
    /// - PF (8 bits): bits 16-23
    /// - PS (8 bits): bits 8-15
    /// - Source Address (8 bits): bits 0-7
    ///
    /// PGN = (DP << 16) | (PF << 8) | PS (if PF >= 240)
    /// PGN = (DP << 16) | (PF << 8) (if PF < 240, PS is destination address)
    pub fn extract_j1939_pgn(&self) -> Option<u32> {
        match self {
            Self::Extended(id) => {
                let pf = (*id >> 16) & 0xFF;
                let ps = (*id >> 8) & 0xFF;
                let dp = (*id >> 24) & 0x01;

                let pgn = if pf >= 240 {
                    // PDU2 format (broadcast)
                    (dp << 16) | (pf << 8) | ps
                } else {
                    // PDU1 format (peer-to-peer)
                    (dp << 16) | (pf << 8)
                };

                Some(pgn)
            }
            Self::Standard(_) => None, // J1939 uses extended IDs only
        }
    }

    /// Extract J1939 source address
    pub fn extract_j1939_source_address(&self) -> Option<u8> {
        match self {
            Self::Extended(id) => Some((id & 0xFF) as u8),
            Self::Standard(_) => None,
        }
    }

    /// Extract J1939 priority
    pub fn extract_j1939_priority(&self) -> Option<u8> {
        match self {
            Self::Extended(id) => Some(((*id >> 26) & 0x07) as u8),
            Self::Standard(_) => None,
        }
    }
}

/// CAN frame (CAN 2.0 or CAN FD)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanFrame {
    /// CAN identifier
    pub id: CanId,

    /// Data payload (max 8 bytes for CAN 2.0, max 64 bytes for CAN FD)
    pub data: Vec<u8>,

    /// Remote Transmission Request flag
    pub rtr: bool,

    /// CAN FD flag
    pub fd: bool,
}

impl CanFrame {
    /// Create a new CAN frame
    pub fn new(id: CanId, data: Vec<u8>) -> CanbusResult<Self> {
        // Validate data length
        if !Self::is_valid_data_length(data.len(), false) {
            return Err(CanbusError::FrameTooLarge(data.len()));
        }

        Ok(Self {
            id,
            data,
            rtr: false,
            fd: false,
        })
    }

    /// Create a CAN FD frame
    pub fn new_fd(id: CanId, data: Vec<u8>) -> CanbusResult<Self> {
        if !Self::is_valid_data_length(data.len(), true) {
            return Err(CanbusError::FrameTooLarge(data.len()));
        }

        Ok(Self {
            id,
            data,
            rtr: false,
            fd: true,
        })
    }

    /// Check if data length is valid
    fn is_valid_data_length(len: usize, fd: bool) -> bool {
        if fd {
            len <= 64
        } else {
            len <= 8
        }
    }

    /// Get Data Length Code (DLC)
    pub fn dlc(&self) -> u8 {
        self.data.len() as u8
    }

    /// Extract byte from data payload
    pub fn get_byte(&self, index: usize) -> Option<u8> {
        self.data.get(index).copied()
    }

    /// Extract bit from data payload
    pub fn get_bit(&self, byte_index: usize, bit_index: u8) -> Option<bool> {
        if bit_index > 7 {
            return None;
        }
        self.data
            .get(byte_index)
            .map(|byte| (byte >> bit_index) & 1 == 1)
    }

    /// Extract multi-byte value (little-endian, Intel byte order)
    pub fn extract_value_le(&self, start_byte: usize, byte_count: usize) -> Option<u64> {
        if start_byte + byte_count > self.data.len() {
            return None;
        }

        let mut value: u64 = 0;
        for i in 0..byte_count {
            value |= (self.data[start_byte + i] as u64) << (i * 8);
        }

        Some(value)
    }

    /// Extract multi-byte value (big-endian, Motorola byte order)
    pub fn extract_value_be(&self, start_byte: usize, byte_count: usize) -> Option<u64> {
        if start_byte + byte_count > self.data.len() {
            return None;
        }

        let mut value: u64 = 0;
        for i in 0..byte_count {
            value = (value << 8) | (self.data[start_byte + i] as u64);
        }

        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_id_standard() {
        let id = CanId::standard(0x123).expect("valid standard CAN ID");
        assert_eq!(id.as_raw(), 0x123);
        assert!(!id.is_extended());
    }

    #[test]
    fn test_can_id_extended() {
        let id = CanId::extended(0x18FEF100).expect("valid extended CAN ID");
        assert_eq!(id.as_raw(), 0x18FEF100);
        assert!(id.is_extended());
    }

    #[test]
    fn test_can_id_invalid() {
        assert!(CanId::standard(0x800).is_err()); // > 11 bits
        assert!(CanId::extended(0x20000000).is_err()); // > 29 bits
    }

    #[test]
    fn test_j1939_pgn_extraction() {
        // PGN 61444 (0xF004): Electronic Engine Controller 1
        // Example CAN ID: 0x0CF00400
        // Priority: 3, PF: 240 (0xF0), PS: 4, SA: 0
        let id = CanId::extended(0x0CF00400).expect("valid extended CAN ID");

        let pgn = id.extract_j1939_pgn().expect("operation should succeed");
        assert_eq!(pgn, 61444); // 0xF004

        let priority = id
            .extract_j1939_priority()
            .expect("operation should succeed");
        assert_eq!(priority, 3);

        let sa = id
            .extract_j1939_source_address()
            .expect("operation should succeed");
        assert_eq!(sa, 0);
    }

    #[test]
    fn test_can_frame_creation() {
        let id = CanId::standard(0x123).expect("valid standard CAN ID");
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let frame = CanFrame::new(id, data.clone()).expect("construction should succeed");

        assert_eq!(frame.id, id);
        assert_eq!(frame.data, data);
        assert_eq!(frame.dlc(), 4);
        assert!(!frame.fd);
    }

    #[test]
    fn test_can_frame_too_large() {
        let id = CanId::standard(0x123).expect("valid standard CAN ID");
        let data = vec![0; 9]; // 9 bytes > 8 byte limit for CAN 2.0

        let result = CanFrame::new(id, data);
        assert!(result.is_err());
    }

    #[test]
    fn test_can_fd_frame() {
        let id = CanId::extended(0x12345678).expect("valid extended CAN ID");
        let data = vec![0; 64]; // CAN FD supports up to 64 bytes

        let frame = CanFrame::new_fd(id, data).expect("valid CAN frame");
        assert_eq!(frame.dlc(), 64);
        assert!(frame.fd);
    }

    #[test]
    fn test_extract_value_le() {
        let id = CanId::standard(0x100).expect("valid standard CAN ID");
        let data = vec![0x12, 0x34, 0x56, 0x78];
        let frame = CanFrame::new(id, data).expect("valid CAN frame");

        // Extract 2 bytes little-endian
        let value = frame
            .extract_value_le(0, 2)
            .expect("value extraction should succeed");
        assert_eq!(value, 0x3412); // 0x34 << 8 | 0x12

        // Extract 4 bytes little-endian
        let value = frame
            .extract_value_le(0, 4)
            .expect("value extraction should succeed");
        assert_eq!(value, 0x78563412);
    }

    #[test]
    fn test_extract_value_be() {
        let id = CanId::standard(0x100).expect("valid standard CAN ID");
        let data = vec![0x12, 0x34, 0x56, 0x78];
        let frame = CanFrame::new(id, data).expect("valid CAN frame");

        // Extract 2 bytes big-endian
        let value = frame
            .extract_value_be(0, 2)
            .expect("value extraction should succeed");
        assert_eq!(value, 0x1234);

        // Extract 4 bytes big-endian
        let value = frame
            .extract_value_be(0, 4)
            .expect("value extraction should succeed");
        assert_eq!(value, 0x12345678);
    }

    #[test]
    fn test_get_bit() {
        let id = CanId::standard(0x100).expect("valid standard CAN ID");
        let data = vec![0b10101010]; // Alternating bits
        let frame = CanFrame::new(id, data).expect("valid CAN frame");

        assert_eq!(frame.get_bit(0, 0), Some(false)); // Bit 0
        assert_eq!(frame.get_bit(0, 1), Some(true)); // Bit 1
        assert_eq!(frame.get_bit(0, 2), Some(false)); // Bit 2
        assert_eq!(frame.get_bit(0, 7), Some(true)); // Bit 7
        assert_eq!(frame.get_bit(1, 0), None); // Out of bounds
    }
}
