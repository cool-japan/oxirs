//! CANopen protocol implementation – CiA DS-301
//!
//! Implements the core CANopen communication profile including:
//!
//! - **NMT** (Network Management) state machine
//! - **SDO** (Service Data Object) expedited and segmented transfer
//! - **PDO** (Process Data Object) transmit/receive
//! - **Heartbeat** / Node Guarding
//! - **EMCY** (Emergency) objects
//! - In-memory **Object Dictionary**
//!
//! # CAN ID Allocation (11-bit)
//!
//! ```text
//! Function Code (4 bits)  Node-ID (7 bits)
//!  NMT    : 0x000
//!  SYNC   : 0x080
//!  EMCY   : 0x080 + NodeID
//!  TPDO1  : 0x180 + NodeID  (PDO1 transmit from device)
//!  RPDO1  : 0x200 + NodeID  (PDO1 receive  by  device)
//!  TPDO2  : 0x280 + NodeID
//!  RPDO2  : 0x300 + NodeID
//!  TPDO3  : 0x380 + NodeID
//!  RPDO3  : 0x400 + NodeID
//!  TPDO4  : 0x480 + NodeID
//!  RPDO4  : 0x500 + NodeID
//!  SDO TX : 0x580 + NodeID  (response from device)
//!  SDO RX : 0x600 + NodeID  (request  to  device)
//!  NMT HB : 0x700 + NodeID  (heartbeat)
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxirs_canbus::canopen::{CanOpenNode, ObjectDictionary, NmtState};
//!
//! let od = ObjectDictionary::new();
//! let mut node = CanOpenNode::new(1, od);
//! assert_eq!(node.nmt_state(), NmtState::Initializing);
//! ```

use crate::error::CanbusError;
use std::collections::HashMap;

// ============================================================================
// CAN-ID helpers
// ============================================================================

/// Compute the CAN ID for a given function code + node ID.
#[inline]
pub const fn canopen_can_id(function_code: u16, node_id: u8) -> u32 {
    (function_code as u32) | (node_id as u32)
}

/// NMT: 0x000 (broadcast – no node ID added)
pub const CANOPEN_NMT_ID: u32 = 0x000;
/// SYNC: 0x080
pub const CANOPEN_SYNC_ID: u32 = 0x080;

/// EMCY base: 0x080 + NodeID
pub const CANOPEN_EMCY_BASE: u16 = 0x080;
/// TPDO1 base: 0x180 + NodeID
pub const CANOPEN_TPDO1_BASE: u16 = 0x180;
/// RPDO1 base: 0x200 + NodeID
pub const CANOPEN_RPDO1_BASE: u16 = 0x200;
/// TPDO2 base: 0x280 + NodeID
pub const CANOPEN_TPDO2_BASE: u16 = 0x280;
/// RPDO2 base: 0x300 + NodeID
pub const CANOPEN_RPDO2_BASE: u16 = 0x300;
/// TPDO3 base: 0x380 + NodeID
pub const CANOPEN_TPDO3_BASE: u16 = 0x380;
/// RPDO3 base: 0x400 + NodeID
pub const CANOPEN_RPDO3_BASE: u16 = 0x400;
/// TPDO4 base: 0x480 + NodeID
pub const CANOPEN_TPDO4_BASE: u16 = 0x480;
/// RPDO4 base: 0x500 + NodeID
pub const CANOPEN_RPDO4_BASE: u16 = 0x500;
/// SDO TX (device → master) base: 0x580 + NodeID
pub const CANOPEN_SDO_TX_BASE: u16 = 0x580;
/// SDO RX (master → device) base: 0x600 + NodeID
pub const CANOPEN_SDO_RX_BASE: u16 = 0x600;
/// NMT heartbeat base: 0x700 + NodeID
pub const CANOPEN_HB_BASE: u16 = 0x700;

/// Return the PDO transmit base for PDO number 1-4.
pub fn tpdo_base(pdo_num: u8) -> Option<u16> {
    match pdo_num {
        1 => Some(CANOPEN_TPDO1_BASE),
        2 => Some(CANOPEN_TPDO2_BASE),
        3 => Some(CANOPEN_TPDO3_BASE),
        4 => Some(CANOPEN_TPDO4_BASE),
        _ => None,
    }
}

/// Return the PDO receive base for PDO number 1-4.
pub fn rpdo_base(pdo_num: u8) -> Option<u16> {
    match pdo_num {
        1 => Some(CANOPEN_RPDO1_BASE),
        2 => Some(CANOPEN_RPDO2_BASE),
        3 => Some(CANOPEN_RPDO3_BASE),
        4 => Some(CANOPEN_RPDO4_BASE),
        _ => None,
    }
}

// ============================================================================
// Minimal CAN message representation (avoid coupling to protocol module)
// ============================================================================

/// A simplified CAN message used by the CANopen node to avoid a hard
/// dependency on the protocol-layer `CanFrame` internals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanMessage {
    /// 11-bit CAN identifier.
    pub can_id: u32,
    /// Data bytes (0–8).
    pub data: Vec<u8>,
}

impl CanMessage {
    /// Create a new CAN message.
    pub fn new(can_id: u32, data: Vec<u8>) -> Self {
        Self { can_id, data }
    }

    /// Convenience constructor for fixed-size payloads.
    pub fn with_bytes(can_id: u32, bytes: &[u8]) -> Self {
        Self {
            can_id,
            data: bytes.to_vec(),
        }
    }
}

// ============================================================================
// NMT State Machine
// ============================================================================

/// CANopen NMT node state (CiA DS-301 §7.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NmtState {
    /// Node is powering up, running boot procedure.
    Initializing = 0x00,
    /// Node is ready; object dictionary accessible but no PDO communication.
    PreOperational = 0x7F,
    /// Node is fully operational; PDO communication enabled.
    Operational = 0x05,
    /// Node has stopped PDO and SDO communication; only NMT commands accepted.
    Stopped = 0x04,
}

impl NmtState {
    /// Encode to the heartbeat state byte.
    pub fn heartbeat_byte(self) -> u8 {
        self as u8
    }

    /// Decode from a heartbeat state byte.
    pub fn from_heartbeat_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::Initializing),
            0x7F => Some(Self::PreOperational),
            0x05 => Some(Self::Operational),
            0x04 => Some(Self::Stopped),
            _ => None,
        }
    }
}

impl std::fmt::Display for NmtState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initializing => write!(f, "Initializing"),
            Self::PreOperational => write!(f, "Pre-Operational"),
            Self::Operational => write!(f, "Operational"),
            Self::Stopped => write!(f, "Stopped"),
        }
    }
}

// ============================================================================
// NMT Commands
// ============================================================================

/// NMT command specifiers (CiA DS-301 §7.3.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NmtCommand {
    /// Start the remote node (→ Operational).
    StartRemoteNode = 0x01,
    /// Stop the remote node (→ Stopped).
    StopRemoteNode = 0x02,
    /// Enter Pre-Operational state.
    EnterPreOperational = 0x80,
    /// Reset the node application (→ Initializing then Pre-Operational).
    ResetNode = 0x81,
    /// Reset only the communication parameters (→ Initializing).
    ResetCommunication = 0x82,
}

impl NmtCommand {
    /// Parse from raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::StartRemoteNode),
            0x02 => Some(Self::StopRemoteNode),
            0x80 => Some(Self::EnterPreOperational),
            0x81 => Some(Self::ResetNode),
            0x82 => Some(Self::ResetCommunication),
            _ => None,
        }
    }
}

impl std::fmt::Display for NmtCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StartRemoteNode => write!(f, "StartRemoteNode"),
            Self::StopRemoteNode => write!(f, "StopRemoteNode"),
            Self::EnterPreOperational => write!(f, "EnterPreOperational"),
            Self::ResetNode => write!(f, "ResetNode"),
            Self::ResetCommunication => write!(f, "ResetCommunication"),
        }
    }
}

// ============================================================================
// SDO Commands
// ============================================================================

/// SDO command specifier (upper 3 bits of first byte, CiA DS-301 §7.2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdoCommand {
    /// Initiate upload (read from device) – sent by master.
    InitiateUploadRequest,
    /// Upload response (device → master): expedited or segmented.
    InitiateUploadResponse {
        expedited: bool,
        size_indicated: bool,
        data_size_n: u8,
    },
    /// Initiate download (write to device) – sent by master: expedited or segmented.
    InitiateDownloadRequest {
        expedited: bool,
        size_indicated: bool,
        data_size_n: u8,
    },
    /// Download initiate response (device → master).
    InitiateDownloadResponse,
    /// Segment download request (master → device).
    DownloadSegmentRequest {
        toggle: bool,
        last: bool,
        seg_size: u8,
    },
    /// Segment download response (device → master).
    DownloadSegmentResponse { toggle: bool },
    /// Segment upload request (master → device).
    UploadSegmentRequest { toggle: bool },
    /// Segment upload response (device → master).
    UploadSegmentResponse {
        toggle: bool,
        last: bool,
        seg_size: u8,
    },
    /// Abort transfer (either direction).
    Abort,
}

/// A decoded SDO frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdoFrame {
    /// The SDO command.
    pub command: SdoCommand,
    /// Object dictionary index (16-bit).
    pub index: u16,
    /// Object dictionary sub-index (8-bit).
    pub sub_index: u8,
    /// Data payload (4 bytes for expedited, segment data for segmented).
    pub data: Vec<u8>,
}

impl SdoFrame {
    /// Parse an SDO RX frame (master → device, CAN ID 0x600 + NodeID).
    pub fn decode_rx(raw: &[u8]) -> Result<Self, CanbusError> {
        if raw.len() < 4 {
            return Err(CanbusError::Config(
                "SDO frame too short (need ≥4 bytes)".to_string(),
            ));
        }
        let cs = raw[0]; // command specifier byte
        let index = u16::from_le_bytes([raw[1], raw[2]]);
        let sub_index = raw[3];
        let data = if raw.len() > 4 {
            raw[4..8.min(raw.len())].to_vec()
        } else {
            vec![0u8; 4]
        };

        // CiA DS-301 Command Specifier (upper 3 bits of byte 0):
        //   000 (0x00) – Download Segment Request  (master → device)
        //   001 (0x01) – Initiate Download Request (master → device, write)
        //   010 (0x02) – Initiate Upload Request   (master → device, read)
        //   011 (0x03) – Upload Segment Request    (master → device)
        //   100 (0x04) – Block Download (not implemented, treated as abort)
        //   111 (0x07) – Abort Transfer
        let upper = (cs >> 5) & 0x07;
        let command = match upper {
            0x01 => {
                // Initiate download request (master writes to device)
                // CS layout: ccs[7:5]=001 | n[4:3] | e[2] | s[1] | 0[0]
                // n = number of bytes in data that do NOT contain data (0–3)
                let expedited = (cs >> 1) & 0x01 != 0;
                let size_indicated = cs & 0x01 != 0;
                let data_size_n = (cs >> 2) & 0x03;
                SdoCommand::InitiateDownloadRequest {
                    expedited,
                    size_indicated,
                    data_size_n,
                }
            }
            0x02 => SdoCommand::InitiateUploadRequest,
            0x00 => {
                // Download segment request (master sends next segment)
                // CS layout: ccs[7:5]=000 | t[4] | n[3:1] | c[0]
                let toggle = (cs >> 4) & 0x01 != 0;
                let seg_size = (cs >> 1) & 0x07;
                let last = cs & 0x01 != 0;
                SdoCommand::DownloadSegmentRequest {
                    toggle,
                    last,
                    seg_size,
                }
            }
            0x03 => {
                // Upload segment request (master requests next upload segment)
                let toggle = (cs >> 4) & 0x01 != 0;
                SdoCommand::UploadSegmentRequest { toggle }
            }
            0x07 => SdoCommand::Abort,
            _ => {
                return Err(CanbusError::Config(format!(
                    "Unknown SDO RX command specifier: 0x{:02X} (upper={})",
                    cs, upper
                )))
            }
        };

        Ok(Self {
            command,
            index,
            sub_index,
            data,
        })
    }

    /// Encode an SDO TX frame (device → master, CAN ID 0x580 + NodeID).
    pub fn encode_tx(&self) -> Vec<u8> {
        let mut out = vec![0u8; 8];
        out[1] = (self.index & 0xFF) as u8;
        out[2] = (self.index >> 8) as u8;
        out[3] = self.sub_index;
        let payload_end = self.data.len().min(4);
        out[4..4 + payload_end].copy_from_slice(&self.data[..payload_end]);

        match &self.command {
            SdoCommand::InitiateUploadResponse {
                expedited,
                size_indicated,
                data_size_n,
            } => {
                let mut cs: u8 = 0x40; // upper=010 for upload response
                if *expedited {
                    cs |= 0x02;
                }
                if *size_indicated {
                    cs |= 0x01;
                }
                cs |= (*data_size_n & 0x03) << 2;
                out[0] = cs;
            }
            SdoCommand::InitiateDownloadResponse => {
                out[0] = 0x60;
            }
            SdoCommand::DownloadSegmentResponse { toggle } => {
                let mut cs: u8 = 0x20;
                if *toggle {
                    cs |= 0x10;
                }
                out[0] = cs;
            }
            SdoCommand::UploadSegmentResponse {
                toggle,
                last,
                seg_size,
            } => {
                let mut cs: u8 = 0x00; // upper=000 for upload segment response
                if *toggle {
                    cs |= 0x10;
                }
                if *last {
                    cs |= 0x01;
                }
                cs |= (*seg_size & 0x07) << 1;
                out[0] = cs;
            }
            SdoCommand::Abort => {
                out[0] = 0x80;
            }
            _ => {}
        }
        out
    }
}

// ============================================================================
// Object Dictionary
// ============================================================================

/// A single Object Dictionary entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OdEntry {
    /// Object index.
    pub index: u16,
    /// Sub-index.
    pub sub_index: u8,
    /// Human-readable name.
    pub name: String,
    /// Value storage.
    pub data: Vec<u8>,
    /// Whether the entry is read-only.
    pub read_only: bool,
}

impl OdEntry {
    /// Create a writable entry.
    pub fn new(index: u16, sub_index: u8, name: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            index,
            sub_index,
            name: name.into(),
            data,
            read_only: false,
        }
    }

    /// Create a read-only entry.
    pub fn read_only(index: u16, sub_index: u8, name: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            index,
            sub_index,
            name: name.into(),
            data,
            read_only: true,
        }
    }
}

/// Key for looking up an OD entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OdKey {
    index: u16,
    sub_index: u8,
}

/// In-memory CANopen Object Dictionary.
#[derive(Debug, Clone)]
pub struct ObjectDictionary {
    entries: HashMap<OdKey, OdEntry>,
}

impl ObjectDictionary {
    /// Create an empty object dictionary.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add or replace an entry.
    pub fn insert(&mut self, entry: OdEntry) {
        let key = OdKey {
            index: entry.index,
            sub_index: entry.sub_index,
        };
        self.entries.insert(key, entry);
    }

    /// Look up an entry by index + sub-index (immutable).
    pub fn get(&self, index: u16, sub_index: u8) -> Option<&OdEntry> {
        self.entries.get(&OdKey { index, sub_index })
    }

    /// Look up an entry mutably.
    pub fn get_mut(&mut self, index: u16, sub_index: u8) -> Option<&mut OdEntry> {
        self.entries.get_mut(&OdKey { index, sub_index })
    }

    /// Read the raw bytes of an entry.
    pub fn read(&self, index: u16, sub_index: u8) -> Result<Vec<u8>, CanbusError> {
        self.get(index, sub_index)
            .map(|e| e.data.clone())
            .ok_or_else(|| {
                CanbusError::Config(format!(
                    "OD entry 0x{:04X}:{:02X} not found",
                    index, sub_index
                ))
            })
    }

    /// Write raw bytes to an entry (rejects writes to read-only entries).
    pub fn write(&mut self, index: u16, sub_index: u8, data: Vec<u8>) -> Result<(), CanbusError> {
        let entry = self.get_mut(index, sub_index).ok_or_else(|| {
            CanbusError::Config(format!(
                "OD entry 0x{:04X}:{:02X} not found",
                index, sub_index
            ))
        })?;
        if entry.read_only {
            return Err(CanbusError::Config(format!(
                "OD entry 0x{:04X}:{:02X} is read-only",
                index, sub_index
            )));
        }
        entry.data = data;
        Ok(())
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the dictionary has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for ObjectDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SDO abort codes (CiA DS-301 Table 21)
// ============================================================================

/// SDO abort codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SdoAbortCode {
    /// Toggle bit not alternated.
    ToggleBitNotAlternated = 0x0503_0000,
    /// SDO protocol timed out.
    SdoProtocolTimedOut = 0x0504_0000,
    /// Command specifier not valid or unknown.
    CommandSpecifierUnknown = 0x0504_0001,
    /// Invalid block size.
    InvalidBlockSize = 0x0504_0002,
    /// Invalid sequence number.
    InvalidSequenceNumber = 0x0504_0003,
    /// CRC error.
    CrcError = 0x0504_0004,
    /// Out of memory.
    OutOfMemory = 0x0504_0005,
    /// Unsupported access to object.
    UnsupportedAccess = 0x0601_0000,
    /// Attempt to read write-only object.
    WriteOnly = 0x0601_0001,
    /// Attempt to write read-only object.
    ReadOnly = 0x0601_0002,
    /// Object does not exist.
    ObjectDoesNotExist = 0x0602_0000,
    /// Object cannot be mapped to PDO.
    CannotMapToPdo = 0x0604_0041,
    /// Number/length of object to be mapped would exceed PDO length.
    PdoLengthExceeded = 0x0604_0042,
    /// General parameter incompatibility.
    ParameterIncompatibility = 0x0604_0043,
    /// General internal incompatibility.
    InternalIncompatibility = 0x0604_0047,
    /// Access failed due to hardware error.
    HardwareError = 0x0606_0000,
    /// Data type/length mismatch.
    DataTypeMismatch = 0x0607_0010,
    /// Data type length too high.
    DataTypeLengthTooHigh = 0x0607_0012,
    /// Data type length too low.
    DataTypeLengthTooLow = 0x0607_0013,
    /// Sub-index does not exist.
    SubIndexDoesNotExist = 0x0609_0011,
    /// Value range exceeded.
    ValueRangeExceeded = 0x0609_0030,
    /// Value too high.
    ValueTooHigh = 0x0609_0031,
    /// Value too low.
    ValueTooLow = 0x0609_0032,
    /// Max < min.
    MaxLessThanMin = 0x0609_0036,
    /// General error.
    GeneralError = 0x0800_0000,
    /// Data cannot be transferred or stored.
    DataTransferError = 0x0800_0020,
    /// Data cannot be transferred due to local control.
    LocalControlError = 0x0800_0021,
    /// Data cannot be transferred in present device state.
    DeviceStateError = 0x0800_0022,
    /// Object dictionary dynamic generation fails.
    OdDynamicGenerationError = 0x0800_0023,
}

impl SdoAbortCode {
    /// Encode to 4-byte little-endian.
    pub fn to_bytes(self) -> [u8; 4] {
        (self as u32).to_le_bytes()
    }
}

// ============================================================================
// EMCY object
// ============================================================================

/// A CANopen Emergency (EMCY) object (CiA DS-301 §7.2.7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmcyObject {
    /// Emergency error code (2 bytes, little-endian).
    pub error_code: u16,
    /// Error register (OD index 0x1001).
    pub error_register: u8,
    /// Manufacturer-specific error field (5 bytes).
    pub manufacturer_data: [u8; 5],
}

impl EmcyObject {
    /// Encode to 8-byte CAN payload.
    pub fn encode(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        let ec = self.error_code.to_le_bytes();
        out[0] = ec[0];
        out[1] = ec[1];
        out[2] = self.error_register;
        out[3..8].copy_from_slice(&self.manufacturer_data);
        out
    }

    /// Decode from an 8-byte CAN payload.
    pub fn decode(raw: &[u8]) -> Result<Self, CanbusError> {
        if raw.len() < 8 {
            return Err(CanbusError::Config(
                "EMCY frame must be 8 bytes".to_string(),
            ));
        }
        let error_code = u16::from_le_bytes([raw[0], raw[1]]);
        let error_register = raw[2];
        let mut manufacturer_data = [0u8; 5];
        manufacturer_data.copy_from_slice(&raw[3..8]);
        Ok(Self {
            error_code,
            error_register,
            manufacturer_data,
        })
    }
}

// ============================================================================
// CANopen Node
// ============================================================================

/// A CANopen device node (CiA DS-301).
///
/// Implements the NMT state machine and handles incoming SDO and NMT
/// messages, generating responses.
#[derive(Debug)]
pub struct CanOpenNode {
    /// Node ID (1–127).
    node_id: u8,
    /// Current NMT state.
    nmt_state: NmtState,
    /// Object dictionary.
    od: ObjectDictionary,
    /// SDO segmented upload state (for segmented reads).
    sdo_upload_state: Option<SdoSegmentState>,
    /// SDO segmented download state (for segmented writes).
    sdo_download_state: Option<SdoDownloadState>,
}

/// Ongoing SDO segmented upload state.
#[derive(Debug)]
struct SdoSegmentState {
    index: u16,
    sub_index: u8,
    data: Vec<u8>,
    offset: usize,
    toggle: bool,
}

/// Ongoing SDO segmented download (write) state.
#[derive(Debug)]
struct SdoDownloadState {
    index: u16,
    sub_index: u8,
    total_size: Option<u32>,
    buffer: Vec<u8>,
    toggle: bool,
}

impl CanOpenNode {
    /// Create a new CANopen node.
    pub fn new(node_id: u8, od: ObjectDictionary) -> Self {
        assert!(
            (1..=127).contains(&node_id),
            "CANopen node ID must be 1-127"
        );
        Self {
            node_id,
            nmt_state: NmtState::Initializing,
            od,
            sdo_upload_state: None,
            sdo_download_state: None,
        }
    }

    /// Return the current NMT state.
    pub fn nmt_state(&self) -> NmtState {
        self.nmt_state
    }

    /// Return the node ID.
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    /// Provide access to the object dictionary.
    pub fn od(&self) -> &ObjectDictionary {
        &self.od
    }

    /// Mutably access the object dictionary.
    pub fn od_mut(&mut self) -> &mut ObjectDictionary {
        &mut self.od
    }

    /// Complete the boot-up sequence (Initializing → Pre-Operational).
    ///
    /// Returns the mandatory boot-up message (NMT HB with state 0x00).
    pub fn boot_up(&mut self) -> CanMessage {
        self.nmt_state = NmtState::PreOperational;
        // Boot-up message: CAN ID 0x700+NodeID, data=[0x00]
        CanMessage::new(canopen_can_id(CANOPEN_HB_BASE, self.node_id), vec![0x00])
    }

    // -----------------------------------------------------------------------
    // NMT processing
    // -----------------------------------------------------------------------

    /// Process an incoming NMT command frame.
    ///
    /// Returns `Some(CanMessage)` if a heartbeat response should be sent
    /// (e.g. after a reset), or `None` if no immediate response is required.
    pub fn process_nmt(&mut self, msg: &CanMessage) -> Option<CanMessage> {
        if msg.can_id != CANOPEN_NMT_ID {
            return None;
        }
        if msg.data.len() < 2 {
            return None;
        }
        let cs_byte = msg.data[0];
        let target_id = msg.data[1];

        // Only respond if addressed to us or broadcast (0)
        if target_id != 0 && target_id != self.node_id {
            return None;
        }

        let cmd = NmtCommand::from_byte(cs_byte)?;
        match cmd {
            NmtCommand::StartRemoteNode => {
                self.nmt_state = NmtState::Operational;
                None
            }
            NmtCommand::StopRemoteNode => {
                self.nmt_state = NmtState::Stopped;
                None
            }
            NmtCommand::EnterPreOperational => {
                self.nmt_state = NmtState::PreOperational;
                None
            }
            NmtCommand::ResetNode => {
                self.nmt_state = NmtState::Initializing;
                self.sdo_upload_state = None;
                self.sdo_download_state = None;
                // Send boot-up message
                let msg = self.boot_up();
                Some(msg)
            }
            NmtCommand::ResetCommunication => {
                self.nmt_state = NmtState::Initializing;
                self.sdo_upload_state = None;
                self.sdo_download_state = None;
                let msg = self.boot_up();
                Some(msg)
            }
        }
    }

    // -----------------------------------------------------------------------
    // SDO processing
    // -----------------------------------------------------------------------

    /// Process an incoming SDO RX frame (CAN ID 0x600 + NodeID).
    ///
    /// Returns `Some(CanMessage)` with the SDO TX response frame, or `None`
    /// if the message is not addressed to this node or is otherwise ignored.
    pub fn process_sdo_rx(&mut self, msg: &CanMessage) -> Option<CanMessage> {
        let expected_id = canopen_can_id(CANOPEN_SDO_RX_BASE, self.node_id);
        if msg.can_id != expected_id {
            return None;
        }
        // SDO not allowed in Stopped state.
        if self.nmt_state == NmtState::Stopped {
            return None;
        }

        let sdo = match SdoFrame::decode_rx(&msg.data) {
            Ok(f) => f,
            Err(_) => return self.sdo_abort(0, 0, SdoAbortCode::CommandSpecifierUnknown),
        };

        match &sdo.command {
            SdoCommand::InitiateUploadRequest => {
                self.handle_sdo_upload_initiate(sdo.index, sdo.sub_index)
            }
            SdoCommand::UploadSegmentRequest { toggle } => {
                let t = *toggle;
                self.handle_sdo_upload_segment(t)
            }
            SdoCommand::InitiateDownloadRequest {
                expedited,
                size_indicated,
                data_size_n,
            } => self.handle_sdo_download_initiate(
                sdo.index,
                sdo.sub_index,
                *expedited,
                *size_indicated,
                *data_size_n,
                &sdo.data,
            ),
            SdoCommand::DownloadSegmentRequest {
                toggle,
                last,
                seg_size,
            } => {
                let (t, l, ss) = (*toggle, *last, *seg_size);
                self.handle_sdo_download_segment(t, l, ss, &sdo.data)
            }
            SdoCommand::Abort => {
                self.sdo_upload_state = None;
                self.sdo_download_state = None;
                None
            }
            _ => self.sdo_abort(
                sdo.index,
                sdo.sub_index,
                SdoAbortCode::CommandSpecifierUnknown,
            ),
        }
    }

    fn handle_sdo_upload_initiate(&mut self, index: u16, sub_index: u8) -> Option<CanMessage> {
        match self.od.read(index, sub_index) {
            Err(_) => self.sdo_abort(index, sub_index, SdoAbortCode::ObjectDoesNotExist),
            Ok(data) => {
                if data.len() <= 4 {
                    // Expedited response
                    let n = (4 - data.len()) as u8;
                    let resp = SdoFrame {
                        command: SdoCommand::InitiateUploadResponse {
                            expedited: true,
                            size_indicated: true,
                            data_size_n: n,
                        },
                        index,
                        sub_index,
                        data: {
                            let mut d = vec![0u8; 4];
                            d[..data.len()].copy_from_slice(&data);
                            d
                        },
                    };
                    Some(self.sdo_tx_message(resp))
                } else {
                    // Segmented upload: respond with size, start segmentation
                    let size = data.len() as u32;
                    self.sdo_upload_state = Some(SdoSegmentState {
                        index,
                        sub_index,
                        data,
                        offset: 0,
                        toggle: false,
                    });
                    let resp = SdoFrame {
                        command: SdoCommand::InitiateUploadResponse {
                            expedited: false,
                            size_indicated: true,
                            data_size_n: 0,
                        },
                        index,
                        sub_index,
                        data: size.to_le_bytes().to_vec(),
                    };
                    Some(self.sdo_tx_message(resp))
                }
            }
        }
    }

    fn handle_sdo_upload_segment(&mut self, toggle: bool) -> Option<CanMessage> {
        // Check state exists and extract needed values, avoiding long mutable borrows.
        let (state_index, state_sub_index, state_toggle) = match &self.sdo_upload_state {
            Some(s) => (s.index, s.sub_index, s.toggle),
            None => {
                return self.sdo_abort(0, 0, SdoAbortCode::CommandSpecifierUnknown);
            }
        };

        if toggle != state_toggle {
            return self.sdo_abort(
                state_index,
                state_sub_index,
                SdoAbortCode::ToggleBitNotAlternated,
            );
        }

        // Now take a mutable reference to do the actual segment work.
        let (seg_data, cur_toggle, last, seg_size_n) = {
            let state = self.sdo_upload_state.as_mut().expect("checked above");
            let remaining = state.data.len().saturating_sub(state.offset);
            let seg_len = remaining.min(7);
            let last = remaining <= 7;
            let seg_size_n = (7 - seg_len) as u8;
            let seg_data: Vec<u8> = state.data[state.offset..state.offset + seg_len].to_vec();
            let cur_toggle = state.toggle;
            state.offset += seg_len;
            state.toggle = !state.toggle;
            (seg_data, cur_toggle, last, seg_size_n)
        };

        let resp = SdoFrame {
            command: SdoCommand::UploadSegmentResponse {
                toggle: cur_toggle,
                last,
                seg_size: seg_size_n,
            },
            index: 0,
            sub_index: 0,
            data: {
                let mut d = vec![0u8; 7];
                d[..seg_data.len()].copy_from_slice(&seg_data);
                d
            },
        };

        if last {
            self.sdo_upload_state = None;
        }

        Some(self.sdo_tx_message(resp))
    }

    fn handle_sdo_download_initiate(
        &mut self,
        index: u16,
        sub_index: u8,
        expedited: bool,
        size_indicated: bool,
        data_size_n: u8,
        data: &[u8],
    ) -> Option<CanMessage> {
        if expedited {
            // Expedited: actual data is in bytes 4-7 of the SDO frame
            let actual_len = if size_indicated {
                (4 - data_size_n as usize).min(data.len())
            } else {
                data.len()
            };
            let write_data = data[..actual_len].to_vec();
            if let Err(e) = self.od.write(index, sub_index, write_data) {
                let code = if e.to_string().contains("read-only") {
                    SdoAbortCode::ReadOnly
                } else {
                    SdoAbortCode::ObjectDoesNotExist
                };
                return self.sdo_abort(index, sub_index, code);
            }
            let resp = SdoFrame {
                command: SdoCommand::InitiateDownloadResponse,
                index,
                sub_index,
                data: vec![0u8; 4],
            };
            Some(self.sdo_tx_message(resp))
        } else {
            // Segmented download: read total size if size_indicated
            let total_size = if size_indicated {
                let s = if data.len() >= 4 {
                    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
                } else {
                    0
                };
                Some(s)
            } else {
                None
            };
            self.sdo_download_state = Some(SdoDownloadState {
                index,
                sub_index,
                total_size,
                buffer: Vec::new(),
                toggle: false,
            });
            let resp = SdoFrame {
                command: SdoCommand::InitiateDownloadResponse,
                index,
                sub_index,
                data: vec![0u8; 4],
            };
            Some(self.sdo_tx_message(resp))
        }
    }

    fn handle_sdo_download_segment(
        &mut self,
        toggle: bool,
        last: bool,
        seg_size: u8,
        data: &[u8],
    ) -> Option<CanMessage> {
        let state = match &mut self.sdo_download_state {
            Some(s) => s,
            None => return self.sdo_abort(0, 0, SdoAbortCode::CommandSpecifierUnknown),
        };

        if toggle != state.toggle {
            let (i, s) = (state.index, state.sub_index);
            return self.sdo_abort(i, s, SdoAbortCode::ToggleBitNotAlternated);
        }

        let actual_len = (7 - seg_size as usize).min(data.len());
        state.buffer.extend_from_slice(&data[..actual_len]);
        state.toggle = !state.toggle;

        // Validate against total size if it was indicated in the initiate frame.
        if let Some(expected) = state.total_size {
            if state.buffer.len() > expected as usize {
                let (i, s) = (state.index, state.sub_index);
                return self.sdo_abort(i, s, SdoAbortCode::DataTransferError);
            }
        }

        if last {
            let index = state.index;
            let sub_index = state.sub_index;
            let buffer = state.buffer.clone();
            self.sdo_download_state = None;

            if let Err(e) = self.od.write(index, sub_index, buffer) {
                let code = if e.to_string().contains("read-only") {
                    SdoAbortCode::ReadOnly
                } else {
                    SdoAbortCode::ObjectDoesNotExist
                };
                return self.sdo_abort(index, sub_index, code);
            }
        }

        let resp = SdoFrame {
            command: SdoCommand::DownloadSegmentResponse { toggle },
            index: 0,
            sub_index: 0,
            data: vec![0u8; 4],
        };
        Some(self.sdo_tx_message(resp))
    }

    fn sdo_abort(&mut self, index: u16, sub_index: u8, code: SdoAbortCode) -> Option<CanMessage> {
        self.sdo_upload_state = None;
        self.sdo_download_state = None;
        let resp = SdoFrame {
            command: SdoCommand::Abort,
            index,
            sub_index,
            data: code.to_bytes().to_vec(),
        };
        Some(self.sdo_tx_message(resp))
    }

    fn sdo_tx_message(&self, frame: SdoFrame) -> CanMessage {
        CanMessage::new(
            canopen_can_id(CANOPEN_SDO_TX_BASE, self.node_id),
            frame.encode_tx(),
        )
    }

    // -----------------------------------------------------------------------
    // Heartbeat
    // -----------------------------------------------------------------------

    /// Create a heartbeat CAN message for the current NMT state.
    pub fn create_heartbeat(&self) -> CanMessage {
        CanMessage::new(
            canopen_can_id(CANOPEN_HB_BASE, self.node_id),
            vec![self.nmt_state.heartbeat_byte()],
        )
    }

    // -----------------------------------------------------------------------
    // PDO helpers
    // -----------------------------------------------------------------------

    /// Assemble a TPDO CAN message for the given PDO number (1–4).
    ///
    /// * `pdo_num` – PDO number 1–4.
    /// * `data`    – Up to 8 bytes of mapped process data.
    pub fn send_tpdo(&self, pdo_num: u8, data: &[u8]) -> Result<CanMessage, CanbusError> {
        let base = tpdo_base(pdo_num)
            .ok_or_else(|| CanbusError::Config(format!("Invalid PDO number: {}", pdo_num)))?;
        if data.len() > 8 {
            return Err(CanbusError::FrameTooLarge(data.len()));
        }
        Ok(CanMessage::with_bytes(
            canopen_can_id(base, self.node_id) as u32,
            data,
        ))
    }

    /// Assemble an EMCY (Emergency) CAN message.
    pub fn send_emcy(&self, emcy: &EmcyObject) -> CanMessage {
        CanMessage::new(
            canopen_can_id(CANOPEN_EMCY_BASE, self.node_id),
            emcy.encode().to_vec(),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- NMT state machine -------------------------------------------------

    fn make_node() -> CanOpenNode {
        let mut od = ObjectDictionary::new();
        od.insert(OdEntry::read_only(
            0x1000,
            0,
            "Device Type",
            vec![0x01, 0x00, 0x01, 0x91],
        ));
        od.insert(OdEntry::read_only(0x1001, 0, "Error Register", vec![0x00]));
        od.insert(OdEntry::new(0x6000, 0, "Digital Input", vec![0x00]));
        od.insert(OdEntry::new(0x6001, 0, "Digital Output", vec![0x00]));
        CanOpenNode::new(1, od)
    }

    #[test]
    fn test_nmt_initial_state_is_initializing() {
        let node = make_node();
        assert_eq!(node.nmt_state(), NmtState::Initializing);
    }

    #[test]
    fn test_boot_up_transitions_to_pre_operational() {
        let mut node = make_node();
        let boot_msg = node.boot_up();
        assert_eq!(node.nmt_state(), NmtState::PreOperational);
        // Boot-up message: CAN ID 0x701 (0x700 + NodeID=1), data=[0x00]
        assert_eq!(boot_msg.can_id, 0x701);
        assert_eq!(boot_msg.data, vec![0x00]);
    }

    #[test]
    fn test_nmt_start_transitions_to_operational() {
        let mut node = make_node();
        node.boot_up();
        let nmt = CanMessage::with_bytes(0x000, &[0x01, 0x01]); // Start Node 1
        let resp = node.process_nmt(&nmt);
        assert!(resp.is_none());
        assert_eq!(node.nmt_state(), NmtState::Operational);
    }

    #[test]
    fn test_nmt_stop_transitions_to_stopped() {
        let mut node = make_node();
        node.boot_up();
        let nmt = CanMessage::with_bytes(0x000, &[0x02, 0x01]);
        node.process_nmt(&nmt);
        assert_eq!(node.nmt_state(), NmtState::Stopped);
    }

    #[test]
    fn test_nmt_enter_pre_operational() {
        let mut node = make_node();
        node.boot_up();
        let nmt_start = CanMessage::with_bytes(0x000, &[0x01, 0x01]);
        node.process_nmt(&nmt_start);
        assert_eq!(node.nmt_state(), NmtState::Operational);

        let nmt_pre = CanMessage::with_bytes(0x000, &[0x80, 0x00]); // broadcast
        node.process_nmt(&nmt_pre);
        assert_eq!(node.nmt_state(), NmtState::PreOperational);
    }

    #[test]
    fn test_nmt_reset_node_returns_bootup() {
        let mut node = make_node();
        node.boot_up();
        let nmt_reset = CanMessage::with_bytes(0x000, &[0x81, 0x01]);
        let resp = node.process_nmt(&nmt_reset);
        assert!(resp.is_some());
        let bootup = resp.expect("ResetNode should produce bootup message");
        assert_eq!(bootup.can_id, 0x701);
        assert_eq!(bootup.data[0], 0x00);
        assert_eq!(node.nmt_state(), NmtState::PreOperational);
    }

    #[test]
    fn test_nmt_broadcast_reset_affects_node() {
        let mut node = make_node();
        node.boot_up();
        // Transition to Operational first
        node.process_nmt(&CanMessage::with_bytes(0x000, &[0x01, 0x00]));
        assert_eq!(node.nmt_state(), NmtState::Operational);

        // Reset via broadcast
        node.process_nmt(&CanMessage::with_bytes(0x000, &[0x81, 0x00]));
        assert_eq!(node.nmt_state(), NmtState::PreOperational);
    }

    #[test]
    fn test_nmt_addressed_to_other_node_ignored() {
        let mut node = make_node();
        node.boot_up();
        // Addressed to node 2, our node is 1 – should be ignored
        let nmt = CanMessage::with_bytes(0x000, &[0x01, 0x02]);
        node.process_nmt(&nmt);
        assert_eq!(node.nmt_state(), NmtState::PreOperational);
    }

    // ---- SDO upload (read) -------------------------------------------------

    #[test]
    fn test_sdo_expedited_upload_device_type() {
        let mut node = make_node();
        node.boot_up();

        // Upload request: [0x40, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00]
        // index=0x1000, sub=0x00
        let req = CanMessage::with_bytes(0x601, &[0x40, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00]);
        let resp = node.process_sdo_rx(&req).expect("should respond");

        // Response CAN ID: 0x581
        assert_eq!(resp.can_id, 0x581);
        // CS byte: expedited=true, size_indicated=true, n=(4-4)=0 → 0x43
        assert_eq!(resp.data[0], 0x43, "expedited upload response CS");
        assert_eq!(resp.data[1], 0x00); // index lo
        assert_eq!(resp.data[2], 0x10); // index hi
        assert_eq!(resp.data[3], 0x00); // sub-index
                                        // data[4..8] = device type bytes
        assert_eq!(&resp.data[4..8], &[0x01, 0x00, 0x01, 0x91]);
    }

    #[test]
    fn test_sdo_expedited_upload_smaller_data() {
        let mut node = make_node();
        node.boot_up();
        // Error register is 1 byte → n=3 → CS=0x4F (0x40 | 0x01 | 0x02 | 0x0C)
        let req = CanMessage::with_bytes(0x601, &[0x40, 0x01, 0x10, 0x00, 0, 0, 0, 0]);
        let resp = node.process_sdo_rx(&req).expect("respond");
        assert_eq!(resp.can_id, 0x581);
        // n=3 means 1 byte used → CS = 0x40 | expedited(0x02) | size_indicated(0x01) | n<<2 = 0x40|0x02|0x01|0x0C = 0x4F
        assert_eq!(resp.data[0], 0x4F);
        assert_eq!(resp.data[4], 0x00); // error register = 0
    }

    #[test]
    fn test_sdo_upload_nonexistent_object_aborts() {
        let mut node = make_node();
        node.boot_up();
        // Object 0x9999 does not exist
        let req = CanMessage::with_bytes(0x601, &[0x40, 0x99, 0x99, 0x00, 0, 0, 0, 0]);
        let resp = node.process_sdo_rx(&req).expect("respond with abort");
        assert_eq!(resp.data[0], 0x80, "abort command specifier");
    }

    // ---- SDO download (write) ----------------------------------------------

    #[test]
    fn test_sdo_expedited_download() {
        let mut node = make_node();
        node.boot_up();

        // Write 1 byte to 0x6001:00 (Digital Output)
        // CS for expedited, size_indicated=true, n=3 (1 byte used): 0x2F
        let req = CanMessage::with_bytes(0x601, &[0x2F, 0x01, 0x60, 0x00, 0xFF, 0, 0, 0]);
        let resp = node.process_sdo_rx(&req).expect("respond");
        assert_eq!(resp.data[0], 0x60, "download response CS");
        // Verify OD was updated
        let val = node.od().read(0x6001, 0).expect("read OD");
        assert_eq!(val, vec![0xFF]);
    }

    #[test]
    fn test_sdo_download_to_read_only_object_aborts() {
        let mut node = make_node();
        node.boot_up();
        // Attempt to write to device type (0x1000) which is read-only
        let req = CanMessage::with_bytes(0x601, &[0x23, 0x00, 0x10, 0x00, 0x99, 0x99, 0x99, 0x99]);
        let resp = node.process_sdo_rx(&req).expect("respond with abort");
        assert_eq!(resp.data[0], 0x80, "should be abort");
    }

    // ---- PDO creation ------------------------------------------------------

    #[test]
    fn test_tpdo1_can_id() {
        let node = make_node();
        let pdo = node.send_tpdo(1, &[0x01, 0x02, 0x03, 0x04]).expect("TPDO1");
        // CAN ID = 0x180 + 1 = 0x181
        assert_eq!(pdo.can_id, 0x181);
        assert_eq!(&pdo.data[..4], &[0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_tpdo2_can_id() {
        let node = make_node();
        let pdo = node.send_tpdo(2, &[0xAA, 0xBB]).expect("TPDO2");
        assert_eq!(pdo.can_id, 0x281); // 0x280 + 1
    }

    #[test]
    fn test_tpdo_invalid_number() {
        let node = make_node();
        assert!(node.send_tpdo(0, &[0]).is_err());
        assert!(node.send_tpdo(5, &[0]).is_err());
    }

    #[test]
    fn test_tpdo_data_too_long() {
        let node = make_node();
        assert!(node.send_tpdo(1, &[0u8; 9]).is_err());
    }

    // ---- Heartbeat ---------------------------------------------------------

    #[test]
    fn test_heartbeat_pre_operational() {
        let mut node = make_node();
        node.boot_up(); // → PreOperational
        let hb = node.create_heartbeat();
        assert_eq!(hb.can_id, 0x701);
        assert_eq!(hb.data[0], NmtState::PreOperational.heartbeat_byte());
        assert_eq!(hb.data[0], 0x7F);
    }

    #[test]
    fn test_heartbeat_operational() {
        let mut node = make_node();
        node.boot_up();
        node.process_nmt(&CanMessage::with_bytes(0x000, &[0x01, 0x01]));
        let hb = node.create_heartbeat();
        assert_eq!(hb.data[0], NmtState::Operational.heartbeat_byte());
        assert_eq!(hb.data[0], 0x05);
    }

    // ---- NMT state encode/decode -------------------------------------------

    #[test]
    fn test_nmt_state_heartbeat_roundtrip() {
        let states = [
            NmtState::Initializing,
            NmtState::PreOperational,
            NmtState::Operational,
            NmtState::Stopped,
        ];
        for s in states {
            let encoded = s.heartbeat_byte();
            let decoded = NmtState::from_heartbeat_byte(encoded).expect("decode");
            assert_eq!(decoded, s);
        }
    }

    // ---- Object Dictionary -------------------------------------------------

    #[test]
    fn test_od_read_write() {
        let mut od = ObjectDictionary::new();
        od.insert(OdEntry::new(0x2000, 0, "Test Value", vec![0x00, 0x00]));
        od.write(0x2000, 0, vec![0x12, 0x34]).expect("write");
        let val = od.read(0x2000, 0).expect("read");
        assert_eq!(val, vec![0x12, 0x34]);
    }

    #[test]
    fn test_od_read_only_rejects_write() {
        let mut od = ObjectDictionary::new();
        od.insert(OdEntry::read_only(0x1000, 0, "Const", vec![0xDE, 0xAD]));
        assert!(od.write(0x1000, 0, vec![0x00]).is_err());
        // Read still works
        let val = od.read(0x1000, 0).expect("read");
        assert_eq!(val, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_od_missing_entry_returns_error() {
        let od = ObjectDictionary::new();
        assert!(od.read(0xFFFF, 0).is_err());
    }

    // ---- EMCY object -------------------------------------------------------

    #[test]
    fn test_emcy_encode_decode() {
        let emcy = EmcyObject {
            error_code: 0x4210,
            error_register: 0x08,
            manufacturer_data: [0x01, 0x02, 0x03, 0x04, 0x05],
        };
        let encoded = emcy.encode();
        assert_eq!(encoded.len(), 8);
        assert_eq!(encoded[0], 0x10); // error_code lo
        assert_eq!(encoded[1], 0x42); // error_code hi
        assert_eq!(encoded[2], 0x08); // error_register
        assert_eq!(&encoded[3..8], &[0x01, 0x02, 0x03, 0x04, 0x05]);

        let decoded = EmcyObject::decode(&encoded).expect("decode EMCY");
        assert_eq!(decoded, emcy);
    }

    #[test]
    fn test_emcy_can_id() {
        let node = make_node();
        let emcy = EmcyObject {
            error_code: 0x1000,
            error_register: 0x00,
            manufacturer_data: [0; 5],
        };
        let msg = node.send_emcy(&emcy);
        assert_eq!(msg.can_id, 0x081); // 0x080 + NodeID=1
    }

    // ---- SDO not allowed in Stopped state ----------------------------------

    #[test]
    fn test_sdo_ignored_when_stopped() {
        let mut node = make_node();
        node.boot_up();
        node.process_nmt(&CanMessage::with_bytes(0x000, &[0x02, 0x01])); // Stop
        assert_eq!(node.nmt_state(), NmtState::Stopped);

        let req = CanMessage::with_bytes(0x601, &[0x40, 0x00, 0x10, 0x00, 0, 0, 0, 0]);
        let resp = node.process_sdo_rx(&req);
        assert!(resp.is_none(), "SDO should be ignored in Stopped state");
    }

    // ---- CAN-ID allocation helpers -----------------------------------------

    #[test]
    fn test_canopen_can_id_allocation() {
        assert_eq!(canopen_can_id(CANOPEN_TPDO1_BASE, 5), 0x185);
        assert_eq!(canopen_can_id(CANOPEN_SDO_TX_BASE, 3), 0x583);
        assert_eq!(canopen_can_id(CANOPEN_HB_BASE, 10), 0x70A);
        assert_eq!(canopen_can_id(CANOPEN_RPDO1_BASE, 127), 0x27F);
    }
}
