//! J1939 Protocol Implementation
//!
//! SAE J1939 is the standard protocol for heavy-duty vehicles (trucks, buses,
//! agricultural equipment, marine vessels). This module provides:
//!
//! - Parameter Group Number (PGN) handling
//! - Multi-packet message reassembly (Transport Protocol)
//! - Address claiming
//! - Signal extraction and decoding

use crate::protocol::frame::{CanFrame, CanId};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::Duration;

/// J1939 Transport Protocol Connection Mode (TP.CM) PGNs
pub const PGN_TP_CM: u32 = 60416; // 0xEC00
/// J1939 Transport Protocol Data Transfer (TP.DT) PGN
pub const PGN_TP_DT: u32 = 60160; // 0xEB00
/// J1939 Address Claimed PGN
pub const PGN_ADDRESS_CLAIMED: u32 = 60928; // 0xEE00
/// J1939 Request PGN
pub const PGN_REQUEST: u32 = 59904; // 0xEA00

/// TP.CM Control Byte values
pub mod tp_control {
    /// Request To Send
    pub const RTS: u8 = 16;
    /// Clear To Send
    pub const CTS: u8 = 17;
    /// End Of Message Acknowledgment
    pub const EOM_ACK: u8 = 19;
    /// Broadcast Announce Message
    pub const BAM: u8 = 32;
    /// Connection Abort
    pub const ABORT: u8 = 255;
}

/// Abort reason codes
pub mod abort_reason {
    /// Already in one or more connection managed sessions
    pub const BUSY: u8 = 1;
    /// System resources needed for another task
    pub const RESOURCES: u8 = 2;
    /// A timeout occurred
    pub const TIMEOUT: u8 = 3;
}

/// J1939 priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Highest priority (0) - Critical control messages
    Highest = 0,
    /// High priority (1-2) - Safety/control messages
    High = 2,
    /// Normal priority (3-4) - General purpose
    Normal = 3,
    /// Low priority (5-6) - Non-critical status
    Low = 5,
    /// Lowest priority (7) - Background/diagnostic
    Lowest = 7,
}

impl From<u8> for Priority {
    fn from(value: u8) -> Self {
        match value {
            0 => Priority::Highest,
            1 | 2 => Priority::High,
            3 | 4 => Priority::Normal,
            5 | 6 => Priority::Low,
            _ => Priority::Lowest,
        }
    }
}

/// J1939 Parameter Group Number with decoded fields
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pgn {
    /// Raw PGN value
    value: u32,
    /// Data Page bit
    data_page: bool,
    /// PDU Format (PF)
    pf: u8,
    /// PDU Specific (PS) - can be group extension or destination address
    ps: u8,
}

impl Pgn {
    /// Create PGN from raw value
    pub fn new(value: u32) -> Self {
        let data_page = (value >> 16) & 0x01 != 0;
        let pf = ((value >> 8) & 0xFF) as u8;
        let ps = (value & 0xFF) as u8;

        Self {
            value,
            data_page,
            pf,
            ps,
        }
    }

    /// Create PGN from components
    pub fn from_components(data_page: bool, pf: u8, ps: u8) -> Self {
        let value = ((data_page as u32) << 16) | ((pf as u32) << 8) | (ps as u32);
        Self {
            value,
            data_page,
            pf,
            ps,
        }
    }

    /// Get raw PGN value
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Check if this is a PDU1 format (peer-to-peer)
    pub fn is_pdu1(&self) -> bool {
        self.pf < 240
    }

    /// Check if this is a PDU2 format (broadcast)
    pub fn is_pdu2(&self) -> bool {
        self.pf >= 240
    }

    /// Get destination address (only valid for PDU1 format)
    pub fn destination_address(&self) -> Option<u8> {
        if self.is_pdu1() {
            Some(self.ps)
        } else {
            None
        }
    }

    /// Get group extension (only valid for PDU2 format)
    pub fn group_extension(&self) -> Option<u8> {
        if self.is_pdu2() {
            Some(self.ps)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Pgn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PGN {} (0x{:05X})", self.value, self.value)
    }
}

/// Decoded J1939 message header
#[derive(Debug, Clone)]
pub struct J1939Header {
    /// Priority (0-7)
    pub priority: u8,
    /// Parameter Group Number
    pub pgn: Pgn,
    /// Source Address (0-253, 254=null, 255=global)
    pub source_address: u8,
    /// Destination address (for PDU1 format)
    pub destination_address: Option<u8>,
}

impl J1939Header {
    /// Decode header from CAN ID
    pub fn from_can_id(can_id: &CanId) -> Option<Self> {
        match can_id {
            CanId::Extended(id) => {
                let priority = ((*id >> 26) & 0x07) as u8;
                let pf = ((*id >> 16) & 0xFF) as u8;
                let ps = ((*id >> 8) & 0xFF) as u8;
                let source_address = (*id & 0xFF) as u8;
                let dp = ((*id >> 24) & 0x01) != 0;

                let (pgn, destination_address) = if pf < 240 {
                    // PDU1: PS is destination address
                    (Pgn::from_components(dp, pf, 0), Some(ps))
                } else {
                    // PDU2: PS is group extension
                    (Pgn::from_components(dp, pf, ps), None)
                };

                Some(Self {
                    priority,
                    pgn,
                    source_address,
                    destination_address,
                })
            }
            CanId::Standard(_) => None, // J1939 requires extended IDs
        }
    }

    /// Build CAN ID from header
    pub fn to_can_id(&self) -> CanId {
        let mut id: u32 = (self.priority as u32) << 26;
        id |= (self.pgn.data_page as u32) << 24;
        id |= (self.pgn.pf as u32) << 16;

        if let Some(dest) = self.destination_address {
            id |= (dest as u32) << 8;
        } else {
            id |= (self.pgn.ps as u32) << 8;
        }

        id |= self.source_address as u32;

        // Safe because we control the construction
        CanId::Extended(id)
    }
}

/// J1939 message (single-frame or reassembled multi-frame)
#[derive(Debug, Clone)]
pub struct J1939Message {
    /// Message header
    pub header: J1939Header,
    /// Message data
    pub data: Vec<u8>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Whether this was a multi-packet message
    pub is_multipacket: bool,
}

impl J1939Message {
    /// Create from single CAN frame
    pub fn from_frame(frame: &CanFrame) -> Option<Self> {
        let header = J1939Header::from_can_id(&frame.id)?;

        Some(Self {
            header,
            data: frame.data.clone(),
            timestamp: Utc::now(),
            is_multipacket: false,
        })
    }

    /// Get raw data as slice
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Extract unsigned integer from data (little-endian)
    pub fn extract_u8(&self, byte_offset: usize) -> Option<u8> {
        self.data.get(byte_offset).copied()
    }

    /// Extract unsigned 16-bit integer from data (little-endian)
    pub fn extract_u16(&self, byte_offset: usize) -> Option<u16> {
        if byte_offset + 2 > self.data.len() {
            return None;
        }
        Some(u16::from_le_bytes([
            self.data[byte_offset],
            self.data[byte_offset + 1],
        ]))
    }

    /// Extract unsigned 32-bit integer from data (little-endian)
    pub fn extract_u32(&self, byte_offset: usize) -> Option<u32> {
        if byte_offset + 4 > self.data.len() {
            return None;
        }
        Some(u32::from_le_bytes([
            self.data[byte_offset],
            self.data[byte_offset + 1],
            self.data[byte_offset + 2],
            self.data[byte_offset + 3],
        ]))
    }

    /// Extract bit field from data
    pub fn extract_bits(&self, byte_offset: usize, bit_offset: u8, bit_count: u8) -> Option<u64> {
        if bit_count == 0 || bit_count > 64 {
            return None;
        }

        let total_bits = byte_offset * 8 + bit_offset as usize + bit_count as usize;
        let bytes_needed = (total_bits + 7) / 8;

        if bytes_needed > self.data.len() {
            return None;
        }

        // Build value from bytes
        let mut value: u64 = 0;
        let start_bit = byte_offset * 8 + bit_offset as usize;

        for i in 0..bit_count as usize {
            let bit_pos = start_bit + i;
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;

            if let Some(&byte) = self.data.get(byte_idx) {
                if (byte >> bit_idx) & 1 == 1 {
                    value |= 1 << i;
                }
            }
        }

        Some(value)
    }
}

/// State of a multi-packet message transfer
#[derive(Debug)]
struct MultiPacketTransfer {
    /// Expected total message size
    total_size: u16,
    /// Number of packets expected
    packet_count: u8,
    /// PGN being transferred
    pgn: u32,
    /// Collected data packets (indexed by sequence number - 1)
    packets: HashMap<u8, Vec<u8>>,
    /// Source address
    source_address: u8,
    /// Destination address (None for BAM)
    destination_address: Option<u8>,
    /// Start time for timeout detection
    start_time: std::time::Instant,
}

/// J1939 Transport Protocol handler for multi-packet messages
pub struct TransportProtocol {
    /// Active transfers by (source, destination, pgn) key
    active_transfers: HashMap<(u8, u8, u32), MultiPacketTransfer>,
    /// Timeout for incomplete transfers
    timeout: Duration,
}

impl TransportProtocol {
    /// Create new transport protocol handler
    pub fn new() -> Self {
        Self {
            active_transfers: HashMap::new(),
            timeout: Duration::from_secs(5),
        }
    }

    /// Set transfer timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Process a CAN frame and return completed J1939 message if available
    pub fn process_frame(&mut self, frame: &CanFrame) -> Option<J1939Message> {
        let header = J1939Header::from_can_id(&frame.id)?;
        let pgn_value = header.pgn.value();

        // Clean up expired transfers
        self.cleanup_expired();

        match pgn_value {
            PGN_TP_CM => self.handle_tp_cm(frame, &header),
            PGN_TP_DT => self.handle_tp_dt(frame, &header),
            _ => {
                // Single-frame message
                Some(J1939Message::from_frame(frame)?)
            }
        }
    }

    /// Handle Transport Protocol Connection Management message
    fn handle_tp_cm(&mut self, frame: &CanFrame, header: &J1939Header) -> Option<J1939Message> {
        if frame.data.len() < 8 {
            return None;
        }

        let control_byte = frame.data[0];
        let dest_addr = header.destination_address.unwrap_or(255);

        match control_byte {
            tp_control::BAM | tp_control::RTS => {
                // Broadcast Announce Message or Request To Send
                let total_size = u16::from_le_bytes([frame.data[1], frame.data[2]]);
                let packet_count = frame.data[3];
                let pgn = u32::from_le_bytes([frame.data[5], frame.data[6], frame.data[7], 0]);

                let key = (header.source_address, dest_addr, pgn);

                self.active_transfers.insert(
                    key,
                    MultiPacketTransfer {
                        total_size,
                        packet_count,
                        pgn,
                        packets: HashMap::new(),
                        source_address: header.source_address,
                        destination_address: if control_byte == tp_control::BAM {
                            None
                        } else {
                            header.destination_address
                        },
                        start_time: std::time::Instant::now(),
                    },
                );

                None
            }
            tp_control::EOM_ACK => {
                // End of Message Acknowledgment - transfer complete on sender side
                None
            }
            tp_control::ABORT => {
                // Connection aborted
                let pgn = u32::from_le_bytes([frame.data[5], frame.data[6], frame.data[7], 0]);
                let key = (header.source_address, dest_addr, pgn);
                self.active_transfers.remove(&key);
                None
            }
            _ => None,
        }
    }

    /// Handle Transport Protocol Data Transfer message
    fn handle_tp_dt(&mut self, frame: &CanFrame, header: &J1939Header) -> Option<J1939Message> {
        if frame.data.len() < 2 {
            return None;
        }

        let sequence_number = frame.data[0];
        let dest_addr = header.destination_address.unwrap_or(255);

        // Find matching active transfer
        let transfer_key = self
            .active_transfers
            .keys()
            .find(|k| k.0 == header.source_address && (k.1 == dest_addr || k.1 == 255))
            .cloned()?;

        let transfer = self.active_transfers.get_mut(&transfer_key)?;

        // Store packet data (bytes 1-7)
        let packet_data: Vec<u8> = frame.data[1..].to_vec();
        transfer.packets.insert(sequence_number, packet_data);

        // Check if all packets received
        if transfer.packets.len() == transfer.packet_count as usize {
            // Reassemble message
            let mut data = Vec::with_capacity(transfer.total_size as usize);

            for seq in 1..=transfer.packet_count {
                if let Some(packet) = transfer.packets.get(&seq) {
                    data.extend(packet);
                }
            }

            // Truncate to actual size
            data.truncate(transfer.total_size as usize);

            let pgn = Pgn::new(transfer.pgn);
            let message = J1939Message {
                header: J1939Header {
                    priority: header.priority,
                    pgn,
                    source_address: transfer.source_address,
                    destination_address: transfer.destination_address,
                },
                data,
                timestamp: Utc::now(),
                is_multipacket: true,
            };

            // Remove completed transfer
            self.active_transfers.remove(&transfer_key);

            return Some(message);
        }

        None
    }

    /// Clean up expired transfers
    fn cleanup_expired(&mut self) {
        let now = std::time::Instant::now();
        self.active_transfers
            .retain(|_, transfer| now.duration_since(transfer.start_time) < self.timeout);
    }

    /// Get number of active transfers
    pub fn active_transfer_count(&self) -> usize {
        self.active_transfers.len()
    }
}

impl Default for TransportProtocol {
    fn default() -> Self {
        Self::new()
    }
}

/// J1939 Address manager for address claiming
pub struct AddressManager {
    /// Our claimed address (None if not claimed)
    claimed_address: Option<u8>,
    /// Our NAME (64-bit device identifier)
    name: u64,
    /// Known devices by address
    devices: HashMap<u8, DeviceInfo>,
}

/// Information about a J1939 device on the network
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device NAME (64-bit identifier)
    pub name: u64,
    /// Last seen timestamp
    pub last_seen: std::time::Instant,
}

impl AddressManager {
    /// Create new address manager with device NAME
    pub fn new(name: u64) -> Self {
        Self {
            claimed_address: None,
            name,
            devices: HashMap::new(),
        }
    }

    /// Get our claimed address
    pub fn address(&self) -> Option<u8> {
        self.claimed_address
    }

    /// Get our NAME
    pub fn name(&self) -> u64 {
        self.name
    }

    /// Process address claim message
    pub fn process_address_claim(&mut self, source_address: u8, name_bytes: &[u8]) -> bool {
        if name_bytes.len() < 8 {
            return false;
        }

        let name = u64::from_le_bytes([
            name_bytes[0],
            name_bytes[1],
            name_bytes[2],
            name_bytes[3],
            name_bytes[4],
            name_bytes[5],
            name_bytes[6],
            name_bytes[7],
        ]);

        self.devices.insert(
            source_address,
            DeviceInfo {
                name,
                last_seen: std::time::Instant::now(),
            },
        );

        // Check if this conflicts with our address
        if Some(source_address) == self.claimed_address {
            // Lower NAME wins (we use our NAME for comparison)
            if name < self.name {
                // We lose - need to find new address or go to null address
                self.claimed_address = None;
                return false;
            }
            // We win - reclaim address
        }

        true
    }

    /// Claim an address on the network
    pub fn claim_address(&mut self, address: u8) -> CanFrame {
        self.claimed_address = Some(address);

        // Build address claim frame
        let name_bytes = self.name.to_le_bytes();
        let id = CanId::Extended(
            (6 << 26) | // Priority 6
            ((PGN_ADDRESS_CLAIMED >> 8) << 16) |
            (255 << 8) | // Global address
            (address as u32),
        );

        CanFrame {
            id,
            data: name_bytes.to_vec(),
            rtr: false,
            fd: false,
        }
    }

    /// Get device at address
    pub fn get_device(&self, address: u8) -> Option<&DeviceInfo> {
        self.devices.get(&address)
    }

    /// Get all known devices
    pub fn devices(&self) -> &HashMap<u8, DeviceInfo> {
        &self.devices
    }
}

/// Complete J1939 processor combining all features
pub struct J1939Processor {
    /// Transport protocol handler
    transport: TransportProtocol,
    /// Address manager (optional)
    address_manager: Option<AddressManager>,
}

impl J1939Processor {
    /// Create new J1939 processor
    pub fn new() -> Self {
        Self {
            transport: TransportProtocol::new(),
            address_manager: None,
        }
    }

    /// Enable address management with device NAME
    pub fn with_address_manager(mut self, name: u64) -> Self {
        self.address_manager = Some(AddressManager::new(name));
        self
    }

    /// Process a CAN frame
    pub fn process(&mut self, frame: &CanFrame) -> Option<J1939Message> {
        let header = J1939Header::from_can_id(&frame.id)?;

        // Handle address claim if we have an address manager
        if header.pgn.value() == PGN_ADDRESS_CLAIMED {
            if let Some(ref mut am) = self.address_manager {
                am.process_address_claim(header.source_address, &frame.data);
            }
        }

        // Process through transport protocol
        self.transport.process_frame(frame)
    }

    /// Get transport protocol reference
    pub fn transport(&self) -> &TransportProtocol {
        &self.transport
    }

    /// Get mutable transport protocol reference
    pub fn transport_mut(&mut self) -> &mut TransportProtocol {
        &mut self.transport
    }

    /// Get address manager reference
    pub fn address_manager(&self) -> Option<&AddressManager> {
        self.address_manager.as_ref()
    }

    /// Get mutable address manager reference
    pub fn address_manager_mut(&mut self) -> Option<&mut AddressManager> {
        self.address_manager.as_mut()
    }
}

impl Default for J1939Processor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgn_creation() {
        let pgn = Pgn::new(61444); // EEC1
        assert_eq!(pgn.value(), 61444);
        assert!(pgn.is_pdu2()); // PF = 240
    }

    #[test]
    fn test_pgn_components() {
        let pgn = Pgn::from_components(false, 240, 4); // PGN 61444
        assert_eq!(pgn.value(), 61444);
        assert!(!pgn.data_page);
        assert_eq!(pgn.pf, 240);
        assert_eq!(pgn.ps, 4);
    }

    #[test]
    fn test_pgn_pdu1_format() {
        // PGN with PF < 240 (peer-to-peer)
        let pgn = Pgn::from_components(false, 200, 0);
        assert!(pgn.is_pdu1());
        assert!(!pgn.is_pdu2());
    }

    #[test]
    fn test_pgn_pdu2_format() {
        // PGN with PF >= 240 (broadcast)
        let pgn = Pgn::new(65265); // CCVS
        assert!(pgn.is_pdu2());
        assert!(!pgn.is_pdu1());
    }

    #[test]
    fn test_j1939_header_from_can_id() {
        // CAN ID: 0x0CF00400 (EEC1 from ECU at address 0)
        let can_id = CanId::extended(0x0CF00400).unwrap();
        let header = J1939Header::from_can_id(&can_id).unwrap();

        assert_eq!(header.priority, 3);
        assert_eq!(header.pgn.value(), 61444);
        assert_eq!(header.source_address, 0);
    }

    #[test]
    fn test_j1939_header_to_can_id() {
        let header = J1939Header {
            priority: 3,
            pgn: Pgn::new(61444),
            source_address: 0,
            destination_address: None,
        };

        let can_id = header.to_can_id();
        assert!(can_id.is_extended());
        assert_eq!(can_id.as_raw(), 0x0CF00400);
    }

    #[test]
    fn test_j1939_message_from_frame() {
        let can_id = CanId::extended(0x0CF00400).unwrap();
        let frame =
            CanFrame::new(can_id, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]).unwrap();

        let message = J1939Message::from_frame(&frame).unwrap();
        assert_eq!(message.header.pgn.value(), 61444);
        assert_eq!(message.data.len(), 8);
    }

    #[test]
    fn test_j1939_message_extract_u16() {
        let can_id = CanId::extended(0x0CF00400).unwrap();
        let frame =
            CanFrame::new(can_id, vec![0x34, 0x12, 0x78, 0x56, 0x00, 0x00, 0x00, 0x00]).unwrap();

        let message = J1939Message::from_frame(&frame).unwrap();
        assert_eq!(message.extract_u16(0), Some(0x1234));
        assert_eq!(message.extract_u16(2), Some(0x5678));
    }

    #[test]
    fn test_transport_protocol_bam() {
        let mut tp = TransportProtocol::new();

        // BAM announcement (PGN 60416)
        let bam_id = CanId::extended(0x1CECFF00).unwrap(); // Priority 7, PGN 60416, SA 0
        let bam_data = vec![
            tp_control::BAM,
            0x09,
            0x00, // Total size: 9 bytes
            0x02, // Packet count: 2
            0xFF, // Reserved
            0x04,
            0xF0,
            0x00, // PGN 61444 (0xF004) in little-endian
        ];
        let bam_frame = CanFrame::new(bam_id, bam_data).unwrap();

        assert!(tp.process_frame(&bam_frame).is_none());
        assert_eq!(tp.active_transfer_count(), 1);
    }

    #[test]
    fn test_transport_protocol_complete_transfer() {
        let mut tp = TransportProtocol::new();

        // BAM announcement
        let bam_id = CanId::extended(0x1CECFF00).unwrap();
        let bam_data = vec![
            tp_control::BAM,
            0x09,
            0x00, // Total size: 9 bytes
            0x02, // Packet count: 2
            0xFF,
            0x04,
            0xF0,
            0x00, // PGN 61444 (0xF004) in little-endian
        ];
        let bam_frame = CanFrame::new(bam_id, bam_data).unwrap();
        tp.process_frame(&bam_frame);

        // Data packet 1 (PGN 60160)
        let dt_id = CanId::extended(0x1CEBFF00).unwrap();
        let dt1_data = vec![0x01, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11];
        let dt1_frame = CanFrame::new(dt_id, dt1_data).unwrap();
        assert!(tp.process_frame(&dt1_frame).is_none());

        // Data packet 2
        let dt2_data = vec![0x02, 0x22, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let dt2_frame = CanFrame::new(dt_id, dt2_data).unwrap();
        let message = tp.process_frame(&dt2_frame);

        assert!(message.is_some());
        let msg = message.unwrap();
        assert_eq!(msg.header.pgn.value(), 61444);
        assert_eq!(msg.data.len(), 9);
        assert!(msg.is_multipacket);
    }

    #[test]
    fn test_priority_from_u8() {
        assert_eq!(Priority::from(0), Priority::Highest);
        assert_eq!(Priority::from(3), Priority::Normal);
        assert_eq!(Priority::from(7), Priority::Lowest);
    }

    #[test]
    fn test_address_manager() {
        let mut am = AddressManager::new(0x123456789ABCDEF0);
        assert_eq!(am.address(), None);

        let claim_frame = am.claim_address(0x80);
        assert!(claim_frame.id.is_extended());
        assert_eq!(am.address(), Some(0x80));
    }

    #[test]
    fn test_address_claim_conflict() {
        let mut am = AddressManager::new(0x1000000000000000);
        am.claim_address(0x80);

        // Another device claims same address with lower NAME (wins)
        let lower_name = 0x0800000000000000u64.to_le_bytes();
        let result = am.process_address_claim(0x80, &lower_name);

        // We should lose our address
        assert!(!result || am.address() != Some(0x80));
    }

    #[test]
    fn test_j1939_processor() {
        let mut processor = J1939Processor::new();

        let can_id = CanId::extended(0x0CF00400).unwrap();
        let frame =
            CanFrame::new(can_id, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]).unwrap();

        let message = processor.process(&frame);
        assert!(message.is_some());
    }

    #[test]
    fn test_extract_bits() {
        let can_id = CanId::extended(0x0CF00400).unwrap();
        let frame = CanFrame::new(can_id, vec![0b10101010, 0b11001100]).unwrap();
        let message = J1939Message::from_frame(&frame).unwrap();

        // Extract bits 0-3 from byte 0
        let bits = message.extract_bits(0, 0, 4);
        assert_eq!(bits, Some(0b1010));

        // Extract bits 4-7 from byte 0
        let bits = message.extract_bits(0, 4, 4);
        assert_eq!(bits, Some(0b1010));
    }
}
