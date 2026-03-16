//! Modbus coil read/write controller.
//!
//! Implements FC01 (Read Coils), FC05 (Write Single Coil), and FC15
//! (Write Multiple Coils) encoding/decoding, plus an in-memory coil
//! address space for unit-testing without a real device.

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A validated Modbus coil address (0–9999).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoilAddress(u16);

impl CoilAddress {
    /// Maximum valid coil address.
    pub const MAX: u16 = 9999;

    /// Create a `CoilAddress`, returning `None` when `addr > 9999`.
    pub fn new(addr: u16) -> Option<Self> {
        if addr <= Self::MAX {
            Some(Self(addr))
        } else {
            None
        }
    }

    /// Create a `CoilAddress` without bounds-checking.
    ///
    /// # Safety
    /// Caller must guarantee `addr <= 9999`.
    pub fn new_unchecked(addr: u16) -> Self {
        Self(addr)
    }

    /// Return the raw address value.
    pub fn value(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for CoilAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Logical state of a Modbus coil.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoilState {
    /// Coil is energised / ON.
    On,
    /// Coil is de-energised / OFF.
    Off,
}

impl CoilState {
    /// `true` when the state is `On`.
    pub fn is_on(self) -> bool {
        matches!(self, CoilState::On)
    }

    /// `true` when the state is `Off`.
    pub fn is_off(self) -> bool {
        matches!(self, CoilState::Off)
    }

    /// Toggle the state.
    pub fn toggled(self) -> Self {
        match self {
            CoilState::On => CoilState::Off,
            CoilState::Off => CoilState::On,
        }
    }
}

impl From<bool> for CoilState {
    fn from(b: bool) -> Self {
        if b {
            CoilState::On
        } else {
            CoilState::Off
        }
    }
}

impl From<CoilState> for bool {
    fn from(s: CoilState) -> Self {
        s.is_on()
    }
}

/// A contiguous range of coil addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoilRange {
    /// First address in the range.
    pub start: CoilAddress,
    /// Number of coils.
    pub count: u16,
}

impl CoilRange {
    /// Create a coil range, returning `None` if the range exceeds address 9999.
    pub fn new(start: CoilAddress, count: u16) -> Option<Self> {
        if count == 0 {
            return None;
        }
        let end = start.0.checked_add(count - 1)?;
        if end > CoilAddress::MAX {
            return None;
        }
        Some(Self { start, count })
    }

    /// Iterate over addresses in this range.
    pub fn addresses(self) -> impl Iterator<Item = CoilAddress> {
        (self.start.0..self.start.0 + self.count).map(CoilAddress)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CoilController
// ──────────────────────────────────────────────────────────────────────────────

/// In-memory Modbus coil address space (0–9999).
pub struct CoilController {
    /// Bit-packed coil state. Index 0 = address 0.
    coils: Vec<bool>,
}

impl CoilController {
    /// Create a controller with all 10 000 coils initialised to `Off`.
    pub fn new() -> Self {
        Self {
            coils: vec![false; 10_000],
        }
    }

    // ── FC01 Read Coils ───────────────────────────────────────────────────────

    /// Read a single coil (FC01 single).
    pub fn read_coil(&self, addr: CoilAddress) -> CoilState {
        CoilState::from(self.coils[addr.0 as usize])
    }

    /// Read multiple coils (FC01 multiple).
    pub fn read_coils(&self, range: CoilRange) -> Vec<CoilState> {
        range.addresses().map(|addr| self.read_coil(addr)).collect()
    }

    // ── FC05 Write Single Coil ────────────────────────────────────────────────

    /// Write a single coil (FC05).
    pub fn write_coil(&mut self, addr: CoilAddress, state: CoilState) {
        self.coils[addr.0 as usize] = state.is_on();
    }

    // ── FC15 Write Multiple Coils ─────────────────────────────────────────────

    /// Write multiple coils starting at `start` (FC15).
    ///
    /// Writes are clamped to the valid address space; extra states are ignored.
    pub fn write_coils(&mut self, start: CoilAddress, states: &[CoilState]) {
        for (offset, &state) in states.iter().enumerate() {
            let addr = start.0 as usize + offset;
            if addr >= self.coils.len() {
                break;
            }
            self.coils[addr] = state.is_on();
        }
    }

    /// Toggle the coil at `addr`.
    pub fn toggle(&mut self, addr: CoilAddress) {
        let idx = addr.0 as usize;
        self.coils[idx] = !self.coils[idx];
    }

    /// Count of coils currently `On`.
    pub fn count_on(&self) -> usize {
        self.coils.iter().filter(|&&b| b).count()
    }

    /// Count of coils currently `Off`.
    pub fn count_off(&self) -> usize {
        self.coils.iter().filter(|&&b| !b).count()
    }
}

impl Default for CoilController {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PDU Encoding / Decoding
// ──────────────────────────────────────────────────────────────────────────────

/// Encode a FC01 Read Coils request PDU.
///
/// ```text
/// Byte 0 : Function code 0x01
/// Byte 1–2 : Starting address (big-endian)
/// Byte 3–4 : Quantity of coils (big-endian)
/// ```
pub fn encode_fc01_request(range: CoilRange) -> Vec<u8> {
    let start = range.start.0;
    let count = range.count;
    vec![
        0x01,
        (start >> 8) as u8,
        (start & 0xFF) as u8,
        (count >> 8) as u8,
        (count & 0xFF) as u8,
    ]
}

/// Decode a FC01 Read Coils response PDU (coil status bytes only, no header).
///
/// `bytes` is the raw coil-status bytes (each byte holds 8 coil states,
/// LSB first). `count` is the number of coils actually requested.
pub fn decode_fc01_response(bytes: &[u8], count: u16) -> Vec<CoilState> {
    let mut states = Vec::with_capacity(count as usize);
    'outer: for byte in bytes {
        for bit in 0..8u8 {
            if states.len() as u16 >= count {
                break 'outer;
            }
            states.push(CoilState::from((byte >> bit) & 1 != 0));
        }
    }
    states
}

/// Encode a FC05 Write Single Coil request PDU.
///
/// ```text
/// Byte 0 : Function code 0x05
/// Byte 1–2 : Output address (big-endian)
/// Byte 3–4 : Output value (0xFF00 = ON, 0x0000 = OFF)
/// ```
pub fn encode_fc05_request(addr: CoilAddress, state: CoilState) -> Vec<u8> {
    let a = addr.0;
    let val: u16 = if state.is_on() { 0xFF00 } else { 0x0000 };
    vec![
        0x05,
        (a >> 8) as u8,
        (a & 0xFF) as u8,
        (val >> 8) as u8,
        (val & 0xFF) as u8,
    ]
}

/// Encode a FC15 Write Multiple Coils request PDU.
///
/// ```text
/// Byte 0   : Function code 0x0F
/// Byte 1–2 : Starting address (big-endian)
/// Byte 3–4 : Quantity of outputs (big-endian)
/// Byte 5   : Byte count
/// Byte 6…  : Outputs value (LSB-first packed bits)
/// ```
pub fn encode_fc15_request(start: CoilAddress, states: &[CoilState]) -> Vec<u8> {
    let count = states.len() as u16;
    let byte_count = ((count + 7) / 8) as u8;
    let s = start.0;

    let mut pdu = vec![
        0x0F,
        (s >> 8) as u8,
        (s & 0xFF) as u8,
        (count >> 8) as u8,
        (count & 0xFF) as u8,
        byte_count,
    ];

    // Pack coil states into bytes, LSB first
    for chunk_start in (0..states.len()).step_by(8) {
        let mut byte = 0u8;
        for bit in 0..8usize {
            let idx = chunk_start + bit;
            if idx < states.len() && states[idx].is_on() {
                byte |= 1 << bit;
            }
        }
        pdu.push(byte);
    }
    pdu
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CoilAddress ───────────────────────────────────────────────────────────

    #[test]
    fn test_coil_address_valid() {
        assert!(CoilAddress::new(0).is_some());
        assert!(CoilAddress::new(9999).is_some());
        assert!(CoilAddress::new(5000).is_some());
    }

    #[test]
    fn test_coil_address_invalid() {
        assert!(CoilAddress::new(10000).is_none());
        assert!(CoilAddress::new(u16::MAX).is_none());
    }

    #[test]
    fn test_coil_address_value() {
        let addr = CoilAddress::new(42).expect("should succeed");
        assert_eq!(addr.value(), 42);
    }

    #[test]
    fn test_coil_address_display() {
        let addr = CoilAddress::new(100).expect("should succeed");
        assert_eq!(format!("{addr}"), "100");
    }

    #[test]
    fn test_coil_address_unchecked() {
        let addr = CoilAddress::new_unchecked(500);
        assert_eq!(addr.value(), 500);
    }

    #[test]
    fn test_coil_address_ordering() {
        let a = CoilAddress::new(10).expect("should succeed");
        let b = CoilAddress::new(20).expect("should succeed");
        assert!(a < b);
    }

    // ── CoilState ─────────────────────────────────────────────────────────────

    #[test]
    fn test_coil_state_is_on() {
        assert!(CoilState::On.is_on());
        assert!(!CoilState::Off.is_on());
    }

    #[test]
    fn test_coil_state_is_off() {
        assert!(CoilState::Off.is_off());
        assert!(!CoilState::On.is_off());
    }

    #[test]
    fn test_coil_state_toggled() {
        assert_eq!(CoilState::On.toggled(), CoilState::Off);
        assert_eq!(CoilState::Off.toggled(), CoilState::On);
    }

    #[test]
    fn test_coil_state_from_bool() {
        assert_eq!(CoilState::from(true), CoilState::On);
        assert_eq!(CoilState::from(false), CoilState::Off);
    }

    #[test]
    fn test_coil_state_into_bool() {
        assert!(bool::from(CoilState::On));
        assert!(!bool::from(CoilState::Off));
    }

    // ── CoilRange ─────────────────────────────────────────────────────────────

    #[test]
    fn test_coil_range_valid() {
        let r = CoilRange::new(CoilAddress::new(0).expect("should succeed"), 100)
            .expect("should succeed");
        assert_eq!(r.count, 100);
    }

    #[test]
    fn test_coil_range_overflow() {
        let r = CoilRange::new(CoilAddress::new(9990).expect("should succeed"), 20);
        assert!(r.is_none(), "range should exceed address space");
    }

    #[test]
    fn test_coil_range_zero_count() {
        let r = CoilRange::new(CoilAddress::new(0).expect("should succeed"), 0);
        assert!(r.is_none());
    }

    #[test]
    fn test_coil_range_addresses() {
        let r = CoilRange::new(CoilAddress::new(10).expect("should succeed"), 5)
            .expect("should succeed");
        let addrs: Vec<u16> = r.addresses().map(|a| a.value()).collect();
        assert_eq!(addrs, vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_coil_range_boundary() {
        let r = CoilRange::new(CoilAddress::new(9999).expect("should succeed"), 1)
            .expect("should succeed");
        let addrs: Vec<u16> = r.addresses().map(|a| a.value()).collect();
        assert_eq!(addrs, vec![9999]);
    }

    // ── CoilController ────────────────────────────────────────────────────────

    #[test]
    fn test_controller_all_off_initially() {
        let ctrl = CoilController::new();
        let addr = CoilAddress::new(0).expect("should succeed");
        assert_eq!(ctrl.read_coil(addr), CoilState::Off);
    }

    #[test]
    fn test_controller_write_and_read_single() {
        let mut ctrl = CoilController::new();
        let addr = CoilAddress::new(50).expect("should succeed");
        ctrl.write_coil(addr, CoilState::On);
        assert_eq!(ctrl.read_coil(addr), CoilState::On);
    }

    #[test]
    fn test_controller_write_and_read_multiple() {
        let mut ctrl = CoilController::new();
        let start = CoilAddress::new(0).expect("should succeed");
        let states = vec![CoilState::On, CoilState::Off, CoilState::On, CoilState::On];
        ctrl.write_coils(start, &states);
        let range = CoilRange::new(start, 4).expect("should succeed");
        let read = ctrl.read_coils(range);
        assert_eq!(read, states);
    }

    #[test]
    fn test_controller_toggle() {
        let mut ctrl = CoilController::new();
        let addr = CoilAddress::new(1).expect("should succeed");
        assert_eq!(ctrl.read_coil(addr), CoilState::Off);
        ctrl.toggle(addr);
        assert_eq!(ctrl.read_coil(addr), CoilState::On);
        ctrl.toggle(addr);
        assert_eq!(ctrl.read_coil(addr), CoilState::Off);
    }

    #[test]
    fn test_controller_count_on() {
        let mut ctrl = CoilController::new();
        ctrl.write_coil(CoilAddress::new(0).expect("should succeed"), CoilState::On);
        ctrl.write_coil(CoilAddress::new(1).expect("should succeed"), CoilState::On);
        ctrl.write_coil(CoilAddress::new(2).expect("should succeed"), CoilState::On);
        assert_eq!(ctrl.count_on(), 3);
    }

    #[test]
    fn test_controller_count_off() {
        let ctrl = CoilController::new();
        assert_eq!(ctrl.count_off(), 10_000);
    }

    #[test]
    fn test_controller_write_overwrite() {
        let mut ctrl = CoilController::new();
        let addr = CoilAddress::new(5).expect("should succeed");
        ctrl.write_coil(addr, CoilState::On);
        ctrl.write_coil(addr, CoilState::Off);
        assert_eq!(ctrl.read_coil(addr), CoilState::Off);
    }

    #[test]
    fn test_controller_write_coils_partial_range() {
        let mut ctrl = CoilController::new();
        let start = CoilAddress::new(9998).expect("should succeed");
        // Only two states fit (9998, 9999); third is clamped
        ctrl.write_coils(start, &[CoilState::On, CoilState::On, CoilState::On]);
        assert_eq!(
            ctrl.read_coil(CoilAddress::new(9998).expect("should succeed")),
            CoilState::On
        );
        assert_eq!(
            ctrl.read_coil(CoilAddress::new(9999).expect("should succeed")),
            CoilState::On
        );
    }

    #[test]
    fn test_controller_default_same_as_new() {
        let ctrl = CoilController::default();
        assert_eq!(ctrl.count_off(), 10_000);
    }

    // ── encode_fc01_request ───────────────────────────────────────────────────

    #[test]
    fn test_encode_fc01_function_code() {
        let range = CoilRange::new(CoilAddress::new(0).expect("should succeed"), 1)
            .expect("should succeed");
        let pdu = encode_fc01_request(range);
        assert_eq!(pdu[0], 0x01);
    }

    #[test]
    fn test_encode_fc01_address_bytes() {
        let range = CoilRange::new(CoilAddress::new(0x0102).expect("should succeed"), 1)
            .expect("should succeed");
        let pdu = encode_fc01_request(range);
        assert_eq!(pdu[1], 0x01);
        assert_eq!(pdu[2], 0x02);
    }

    #[test]
    fn test_encode_fc01_count_bytes() {
        let range = CoilRange::new(CoilAddress::new(0).expect("should succeed"), 0x0105)
            .expect("should succeed");
        let pdu = encode_fc01_request(range);
        assert_eq!(pdu[3], 0x01);
        assert_eq!(pdu[4], 0x05);
    }

    #[test]
    fn test_encode_fc01_length() {
        let range = CoilRange::new(CoilAddress::new(0).expect("should succeed"), 10)
            .expect("should succeed");
        assert_eq!(encode_fc01_request(range).len(), 5);
    }

    // ── decode_fc01_response ──────────────────────────────────────────────────

    #[test]
    fn test_decode_fc01_all_on() {
        let states = decode_fc01_response(&[0xFF], 8);
        assert_eq!(states, vec![CoilState::On; 8]);
    }

    #[test]
    fn test_decode_fc01_all_off() {
        let states = decode_fc01_response(&[0x00], 8);
        assert_eq!(states, vec![CoilState::Off; 8]);
    }

    #[test]
    fn test_decode_fc01_lsb_first() {
        // 0b00000001 → coil 0 = On, rest Off
        let states = decode_fc01_response(&[0x01], 8);
        assert_eq!(states[0], CoilState::On);
        assert_eq!(states[1], CoilState::Off);
    }

    #[test]
    fn test_decode_fc01_count_limits() {
        let states = decode_fc01_response(&[0xFF], 3);
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_decode_fc01_multi_byte() {
        // 0xFF (8 On) + 0x00 (8 Off) → 16 coils
        let states = decode_fc01_response(&[0xFF, 0x00], 16);
        assert_eq!(&states[..8], &[CoilState::On; 8]);
        assert_eq!(&states[8..], &[CoilState::Off; 8]);
    }

    #[test]
    fn test_decode_fc01_pattern() {
        // 0b10101010 → Off On Off On Off On Off On (LSB first: Off,On,Off,On,Off,On,Off,On)
        let states = decode_fc01_response(&[0b1010_1010], 8);
        assert_eq!(states[0], CoilState::Off);
        assert_eq!(states[1], CoilState::On);
        assert_eq!(states[2], CoilState::Off);
        assert_eq!(states[3], CoilState::On);
    }

    // ── encode_fc05_request ───────────────────────────────────────────────────

    #[test]
    fn test_encode_fc05_function_code() {
        let pdu = encode_fc05_request(CoilAddress::new(0).expect("should succeed"), CoilState::On);
        assert_eq!(pdu[0], 0x05);
    }

    #[test]
    fn test_encode_fc05_on_value() {
        let pdu = encode_fc05_request(CoilAddress::new(0).expect("should succeed"), CoilState::On);
        assert_eq!(pdu[3], 0xFF);
        assert_eq!(pdu[4], 0x00);
    }

    #[test]
    fn test_encode_fc05_off_value() {
        let pdu = encode_fc05_request(CoilAddress::new(0).expect("should succeed"), CoilState::Off);
        assert_eq!(pdu[3], 0x00);
        assert_eq!(pdu[4], 0x00);
    }

    #[test]
    fn test_encode_fc05_address() {
        let pdu = encode_fc05_request(
            CoilAddress::new(0x1234).expect("should succeed"),
            CoilState::On,
        );
        assert_eq!(pdu[1], 0x12);
        assert_eq!(pdu[2], 0x34);
    }

    #[test]
    fn test_encode_fc05_length() {
        let pdu = encode_fc05_request(CoilAddress::new(0).expect("should succeed"), CoilState::Off);
        assert_eq!(pdu.len(), 5);
    }

    // ── encode_fc15_request ───────────────────────────────────────────────────

    #[test]
    fn test_encode_fc15_function_code() {
        let pdu = encode_fc15_request(
            CoilAddress::new(0).expect("should succeed"),
            &[CoilState::On],
        );
        assert_eq!(pdu[0], 0x0F);
    }

    #[test]
    fn test_encode_fc15_count() {
        let states = vec![CoilState::On; 10];
        let pdu = encode_fc15_request(CoilAddress::new(0).expect("should succeed"), &states);
        let count = u16::from_be_bytes([pdu[3], pdu[4]]);
        assert_eq!(count, 10);
    }

    #[test]
    fn test_encode_fc15_byte_count() {
        // 10 coils → ceil(10/8) = 2 bytes
        let states = vec![CoilState::On; 10];
        let pdu = encode_fc15_request(CoilAddress::new(0).expect("should succeed"), &states);
        assert_eq!(pdu[5], 2);
    }

    #[test]
    fn test_encode_fc15_all_on_single_byte() {
        let states = vec![CoilState::On; 8];
        let pdu = encode_fc15_request(CoilAddress::new(0).expect("should succeed"), &states);
        assert_eq!(pdu[6], 0xFF);
    }

    #[test]
    fn test_encode_fc15_all_off_single_byte() {
        let states = vec![CoilState::Off; 8];
        let pdu = encode_fc15_request(CoilAddress::new(0).expect("should succeed"), &states);
        assert_eq!(pdu[6], 0x00);
    }

    #[test]
    fn test_encode_fc15_address() {
        let states = vec![CoilState::On];
        let pdu = encode_fc15_request(CoilAddress::new(0x0055).expect("should succeed"), &states);
        assert_eq!(pdu[1], 0x00);
        assert_eq!(pdu[2], 0x55);
    }

    // ── Roundtrip encode/decode ────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_8_coils() {
        let original = vec![
            CoilState::On,
            CoilState::Off,
            CoilState::On,
            CoilState::Off,
            CoilState::On,
            CoilState::Off,
            CoilState::On,
            CoilState::Off,
        ];
        let pdu = encode_fc15_request(CoilAddress::new(0).expect("should succeed"), &original);
        // Coil bytes start at index 6 in FC15
        let coil_bytes = &pdu[6..];
        let decoded = decode_fc01_response(coil_bytes, 8);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_10_coils() {
        let original = vec![
            CoilState::On,
            CoilState::Off,
            CoilState::On,
            CoilState::On,
            CoilState::Off,
            CoilState::On,
            CoilState::Off,
            CoilState::Off,
            CoilState::On,
            CoilState::On,
        ];
        let pdu = encode_fc15_request(CoilAddress::new(0).expect("should succeed"), &original);
        let coil_bytes = &pdu[6..];
        let decoded = decode_fc01_response(coil_bytes, 10);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_1_coil_on() {
        let pdu = encode_fc15_request(
            CoilAddress::new(0).expect("should succeed"),
            &[CoilState::On],
        );
        let decoded = decode_fc01_response(&pdu[6..], 1);
        assert_eq!(decoded, vec![CoilState::On]);
    }

    #[test]
    fn test_roundtrip_1_coil_off() {
        let pdu = encode_fc15_request(
            CoilAddress::new(0).expect("should succeed"),
            &[CoilState::Off],
        );
        let decoded = decode_fc01_response(&pdu[6..], 1);
        assert_eq!(decoded, vec![CoilState::Off]);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Higher-level CoilBank with labelling, write-protection, and PDU helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Error type for `CoilBank` operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CoilError {
    /// No coil is registered at the given address.
    AddressNotFound(u16),
    /// The coil at the given address is write-protected.
    WriteProtected(u16),
    /// The address range is invalid (e.g. zero count or overflow).
    InvalidRange { start: u16, count: u16 },
}

impl std::fmt::Display for CoilError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoilError::AddressNotFound(a) => write!(f, "coil address {a} not found"),
            CoilError::WriteProtected(a) => write!(f, "coil address {a} is write-protected"),
            CoilError::InvalidRange { start, count } => {
                write!(f, "invalid coil range start={start} count={count}")
            }
        }
    }
}

impl std::error::Error for CoilError {}

/// A single registered coil with metadata.
#[derive(Debug, Clone)]
pub struct Coil {
    /// Modbus address of the coil.
    pub address: u16,
    /// Current on/off state.
    pub state: CoilState,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// When `true`, write operations are rejected.
    pub write_protected: bool,
    /// Timestamp (ms since epoch or monotonic counter) of the last state change.
    pub last_changed_ms: u64,
}

impl Coil {
    fn new(address: u16, label: Option<String>, initial: CoilState) -> Self {
        Self {
            address,
            state: initial,
            label,
            write_protected: false,
            last_changed_ms: 0,
        }
    }
}

/// A managed bank of named, labelled coils with read/write and PDU encoding support.
#[derive(Debug, Default)]
pub struct CoilBank {
    coils: std::collections::HashMap<u16, Coil>,
    /// Monotonic counter used for `last_changed_ms`.
    tick: u64,
}

impl CoilBank {
    /// Create an empty coil bank.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a coil at `address` with an optional label and initial state.
    pub fn add(&mut self, address: u16, label: Option<String>, initial: CoilState) {
        self.coils
            .insert(address, Coil::new(address, label, initial));
    }

    /// Read the state of a single coil.
    pub fn read(&self, address: u16) -> Result<CoilState, CoilError> {
        self.coils
            .get(&address)
            .map(|c| c.state)
            .ok_or(CoilError::AddressNotFound(address))
    }

    /// Write the state of a single coil (fails if write-protected).
    pub fn write(&mut self, address: u16, state: CoilState) -> Result<(), CoilError> {
        let tick = self.tick;
        let coil = self
            .coils
            .get_mut(&address)
            .ok_or(CoilError::AddressNotFound(address))?;
        if coil.write_protected {
            return Err(CoilError::WriteProtected(address));
        }
        coil.state = state;
        coil.last_changed_ms = tick;
        self.tick += 1;
        Ok(())
    }

    /// Write consecutive coils starting at `start`; returns the count actually written.
    pub fn write_multiple(&mut self, start: u16, states: &[CoilState]) -> Result<usize, CoilError> {
        if states.is_empty() {
            return Err(CoilError::InvalidRange { start, count: 0 });
        }
        let mut written = 0usize;
        for (offset, &state) in states.iter().enumerate() {
            let addr = start.saturating_add(offset as u16);
            if self.coils.contains_key(&addr) {
                self.write(addr, state)?;
                written += 1;
            }
        }
        Ok(written)
    }

    /// Read consecutive coils starting at `start` for `count` addresses.
    pub fn read_multiple(
        &self,
        start: u16,
        count: u16,
    ) -> Result<Vec<(u16, CoilState)>, CoilError> {
        if count == 0 {
            return Err(CoilError::InvalidRange { start, count });
        }
        let mut out = Vec::with_capacity(count as usize);
        for offset in 0..count {
            let addr = start.saturating_add(offset);
            let state = self.read(addr)?;
            out.push((addr, state));
        }
        Ok(out)
    }

    /// Toggle the coil at `address` and return the new state.
    pub fn toggle(&mut self, address: u16) -> Result<CoilState, CoilError> {
        let new_state = {
            let coil = self
                .coils
                .get(&address)
                .ok_or(CoilError::AddressNotFound(address))?;
            if coil.write_protected {
                return Err(CoilError::WriteProtected(address));
            }
            coil.state.toggled()
        };
        self.write(address, new_state)?;
        Ok(new_state)
    }

    /// Set or clear the write-protection flag for a coil.
    pub fn set_write_protected(&mut self, address: u16, protected: bool) -> Result<(), CoilError> {
        let coil = self
            .coils
            .get_mut(&address)
            .ok_or(CoilError::AddressNotFound(address))?;
        coil.write_protected = protected;
        Ok(())
    }

    /// Return sorted addresses of coils currently in the `On` state.
    pub fn active_coils(&self) -> Vec<u16> {
        let mut addrs: Vec<u16> = self
            .coils
            .values()
            .filter(|c| matches!(c.state, CoilState::On))
            .map(|c| c.address)
            .collect();
        addrs.sort_unstable();
        addrs
    }

    /// Return all coils sorted by address.
    pub fn all_coils(&self) -> Vec<&Coil> {
        let mut coils: Vec<&Coil> = self.coils.values().collect();
        coils.sort_by_key(|c| c.address);
        coils
    }

    /// Number of registered coils.
    pub fn coil_count(&self) -> usize {
        self.coils.len()
    }

    /// Encode an FC01 Read Coils response PDU body for `count` coils starting at `start`.
    ///
    /// Bits are packed LSB-first per Modbus specification, padded to a full byte boundary.
    pub fn encode_pdu_read_response(&self, start: u16, count: u16) -> Result<Vec<u8>, CoilError> {
        if count == 0 {
            return Err(CoilError::InvalidRange { start, count });
        }
        let byte_count = ((count + 7) / 8) as usize;
        let mut bytes = vec![0u8; byte_count];
        for offset in 0..count {
            let addr = start.saturating_add(offset);
            let state = self.read(addr)?;
            if matches!(state, CoilState::On) {
                let byte_idx = (offset / 8) as usize;
                let bit_idx = offset % 8;
                bytes[byte_idx] |= 1 << bit_idx;
            }
        }
        Ok(bytes)
    }

    /// Decode an FC15 Write Multiple Coils request data field.
    ///
    /// `data` is the raw coil-value bytes (LSB-first), and `count` is the number of
    /// coils requested. Returns a list of `(address, state)` pairs.
    pub fn decode_pdu_write_request(
        &self,
        start: u16,
        data: &[u8],
        count: u16,
    ) -> Result<Vec<(u16, CoilState)>, CoilError> {
        if count == 0 {
            return Err(CoilError::InvalidRange { start, count });
        }
        let mut out = Vec::with_capacity(count as usize);
        let mut decoded = 0u16;
        'outer: for byte in data {
            for bit in 0..8u16 {
                if decoded >= count {
                    break 'outer;
                }
                let on = (byte >> bit) & 1 != 0;
                let addr = start.saturating_add(decoded);
                out.push((addr, CoilState::from(on)));
                decoded += 1;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod bank_tests {
    use super::*;

    fn make_bank() -> CoilBank {
        let mut b = CoilBank::new();
        b.add(0, Some("coil_0".into()), CoilState::Off);
        b.add(1, Some("coil_1".into()), CoilState::On);
        b.add(2, None, CoilState::Off);
        b.add(3, None, CoilState::Off);
        b
    }

    // ── CoilError ──────────────────────────────────────────────────────────────

    #[test]
    fn test_coil_error_display_not_found() {
        let e = CoilError::AddressNotFound(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_coil_error_display_write_protected() {
        let e = CoilError::WriteProtected(10);
        assert!(e.to_string().contains("10"));
    }

    #[test]
    fn test_coil_error_display_invalid_range() {
        let e = CoilError::InvalidRange { start: 5, count: 0 };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_coil_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(CoilError::AddressNotFound(1));
        assert!(!e.to_string().is_empty());
    }

    // ── add / read ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_and_read() {
        let b = make_bank();
        assert_eq!(b.read(0), Ok(CoilState::Off));
        assert_eq!(b.read(1), Ok(CoilState::On));
    }

    #[test]
    fn test_read_not_found() {
        let b = make_bank();
        assert_eq!(b.read(99), Err(CoilError::AddressNotFound(99)));
    }

    #[test]
    fn test_coil_count() {
        let b = make_bank();
        assert_eq!(b.coil_count(), 4);
    }

    // ── write ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_write_changes_state() {
        let mut b = make_bank();
        b.write(0, CoilState::On).expect("write should succeed");
        assert_eq!(b.read(0), Ok(CoilState::On));
    }

    #[test]
    fn test_write_not_found() {
        let mut b = make_bank();
        assert_eq!(
            b.write(50, CoilState::On),
            Err(CoilError::AddressNotFound(50))
        );
    }

    #[test]
    fn test_write_protected_rejected() {
        let mut b = make_bank();
        b.set_write_protected(0, true).expect("should succeed");
        assert_eq!(b.write(0, CoilState::On), Err(CoilError::WriteProtected(0)));
    }

    // ── write_multiple ─────────────────────────────────────────────────────────

    #[test]
    fn test_write_multiple_all_present() {
        let mut b = make_bank();
        let states = vec![CoilState::On, CoilState::Off, CoilState::On, CoilState::On];
        let written = b.write_multiple(0, &states).expect("should succeed");
        assert_eq!(written, 4);
        assert_eq!(b.read(0), Ok(CoilState::On));
        assert_eq!(b.read(1), Ok(CoilState::Off));
    }

    #[test]
    fn test_write_multiple_empty_error() {
        let mut b = make_bank();
        let result = b.write_multiple(0, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_multiple_skips_missing() {
        let mut b = make_bank();
        // address 10 does not exist; only 0-3 do
        let states = vec![CoilState::On, CoilState::On]; // 0 and 1 exist
        let written = b.write_multiple(0, &states).expect("should succeed");
        assert_eq!(written, 2);
    }

    // ── read_multiple ──────────────────────────────────────────────────────────

    #[test]
    fn test_read_multiple_returns_correct_states() {
        let b = make_bank();
        let result = b.read_multiple(0, 2).expect("should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (0, CoilState::Off));
        assert_eq!(result[1], (1, CoilState::On));
    }

    #[test]
    fn test_read_multiple_zero_count_error() {
        let b = make_bank();
        assert!(b.read_multiple(0, 0).is_err());
    }

    #[test]
    fn test_read_multiple_missing_address_error() {
        let b = make_bank();
        assert!(b.read_multiple(50, 1).is_err());
    }

    // ── toggle ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_toggle_off_to_on() {
        let mut b = make_bank();
        let new_state = b.toggle(0).expect("toggle should succeed");
        assert_eq!(new_state, CoilState::On);
        assert_eq!(b.read(0), Ok(CoilState::On));
    }

    #[test]
    fn test_toggle_on_to_off() {
        let mut b = make_bank();
        let new_state = b.toggle(1).expect("toggle should succeed");
        assert_eq!(new_state, CoilState::Off);
    }

    #[test]
    fn test_toggle_not_found() {
        let mut b = make_bank();
        assert_eq!(b.toggle(99), Err(CoilError::AddressNotFound(99)));
    }

    #[test]
    fn test_toggle_write_protected() {
        let mut b = make_bank();
        b.set_write_protected(0, true).expect("should succeed");
        assert_eq!(b.toggle(0), Err(CoilError::WriteProtected(0)));
    }

    // ── set_write_protected ────────────────────────────────────────────────────

    #[test]
    fn test_set_write_protected_not_found() {
        let mut b = make_bank();
        assert!(b.set_write_protected(99, true).is_err());
    }

    #[test]
    fn test_set_write_protected_then_unprotect() {
        let mut b = make_bank();
        b.set_write_protected(0, true).expect("should succeed");
        assert!(b.write(0, CoilState::On).is_err());
        b.set_write_protected(0, false).expect("should succeed");
        assert!(b.write(0, CoilState::On).is_ok());
    }

    // ── active_coils ────────────────────────────────────────────────────────────

    #[test]
    fn test_active_coils_sorted() {
        let b = make_bank(); // coil 1 is On
        let active = b.active_coils();
        assert_eq!(active, vec![1u16]);
    }

    #[test]
    fn test_active_coils_after_write() {
        let mut b = make_bank();
        b.write(0, CoilState::On).expect("should succeed");
        b.write(2, CoilState::On).expect("should succeed");
        let mut active = b.active_coils();
        active.sort_unstable();
        assert_eq!(active, vec![0u16, 1, 2]);
    }

    // ── all_coils ───────────────────────────────────────────────────────────────

    #[test]
    fn test_all_coils_sorted_by_address() {
        let b = make_bank();
        let all = b.all_coils();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].address, 0);
        assert_eq!(all[1].address, 1);
        assert_eq!(all[2].address, 2);
        assert_eq!(all[3].address, 3);
    }

    // ── encode_pdu_read_response ────────────────────────────────────────────────

    #[test]
    fn test_encode_pdu_read_zero_count_error() {
        let b = make_bank();
        assert!(b.encode_pdu_read_response(0, 0).is_err());
    }

    #[test]
    fn test_encode_pdu_read_single_off() {
        let b = make_bank();
        let bytes = b.encode_pdu_read_response(0, 1).expect("should succeed");
        assert_eq!(bytes, vec![0x00]);
    }

    #[test]
    fn test_encode_pdu_read_single_on() {
        let b = make_bank();
        let bytes = b.encode_pdu_read_response(1, 1).expect("should succeed");
        assert_eq!(bytes, vec![0x01]);
    }

    #[test]
    fn test_encode_pdu_read_four_coils() {
        // Coil 0=Off, 1=On, 2=Off, 3=Off → 0b00000010 = 0x02
        let b = make_bank();
        let bytes = b.encode_pdu_read_response(0, 4).expect("should succeed");
        assert_eq!(bytes, vec![0x02]);
    }

    #[test]
    fn test_encode_pdu_read_pads_to_byte_boundary() {
        // 4 coils → 1 byte (no padding needed), 9 coils → 2 bytes
        let mut b = CoilBank::new();
        for i in 0..9u16 {
            b.add(i, None, CoilState::Off);
        }
        let bytes = b.encode_pdu_read_response(0, 9).expect("should succeed");
        assert_eq!(bytes.len(), 2);
    }

    // ── decode_pdu_write_request ────────────────────────────────────────────────

    #[test]
    fn test_decode_pdu_write_zero_count_error() {
        let b = make_bank();
        assert!(b.decode_pdu_write_request(0, &[0xFF], 0).is_err());
    }

    #[test]
    fn test_decode_pdu_write_single_on() {
        let b = make_bank();
        let result = b
            .decode_pdu_write_request(0, &[0x01], 1)
            .expect("should succeed");
        assert_eq!(result, vec![(0, CoilState::On)]);
    }

    #[test]
    fn test_decode_pdu_write_single_off() {
        let b = make_bank();
        let result = b
            .decode_pdu_write_request(0, &[0x00], 1)
            .expect("should succeed");
        assert_eq!(result, vec![(0, CoilState::Off)]);
    }

    #[test]
    fn test_decode_pdu_write_multiple() {
        let b = make_bank();
        // 0b00000011 → coil 0=On, 1=On, then 0b00000000 → 2=Off, 3=Off
        let result = b
            .decode_pdu_write_request(0, &[0x03, 0x00], 4)
            .expect("should succeed");
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], (0, CoilState::On));
        assert_eq!(result[1], (1, CoilState::On));
        assert_eq!(result[2], (2, CoilState::Off));
        assert_eq!(result[3], (3, CoilState::Off));
    }

    #[test]
    fn test_roundtrip_encode_decode() {
        let mut b = CoilBank::new();
        for i in 0..8u16 {
            b.add(
                i,
                None,
                if i % 2 == 0 {
                    CoilState::On
                } else {
                    CoilState::Off
                },
            );
        }
        let encoded = b.encode_pdu_read_response(0, 8).expect("encode ok");
        let decoded = b
            .decode_pdu_write_request(0, &encoded, 8)
            .expect("decode ok");
        for (addr, state) in &decoded {
            let expected = b.read(*addr).expect("read ok");
            assert_eq!(state, &expected, "addr {addr} mismatch");
        }
    }
}
