//! Modbus coil and discrete-input register map.
//!
//! Provides an in-memory model of Modbus coil space (function codes FC01/FC05/FC15)
//! and discrete-input space (FC02), including read-only block enforcement and
//! packed byte serialisation used in protocol frames.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A contiguous block of coil registers with optional write protection.
#[derive(Debug, Clone)]
pub struct CoilBlock {
    /// Address of the first coil in this block.
    pub start_address: u16,
    /// Coil values; index 0 corresponds to `start_address`.
    pub coils: Vec<bool>,
    /// When `true`, write operations to any address in this block are rejected.
    pub read_only: bool,
}

impl CoilBlock {
    /// Create a new block of `count` coils, all initialised to `false`.
    pub fn new(start_address: u16, count: usize, read_only: bool) -> Self {
        Self {
            start_address,
            coils: vec![false; count],
            read_only,
        }
    }

    /// Return the end address (exclusive) of this block.
    pub fn end_address(&self) -> u32 {
        self.start_address as u32 + self.coils.len() as u32
    }
}

/// Summary of a multi-coil write operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteResult {
    /// Number of coils that were successfully written.
    pub written: u16,
    /// Number of coils that were silently skipped because the address is inside
    /// a read-only block.
    pub skipped_read_only: u16,
}

/// Errors that can occur during coil register map operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoilError {
    /// The requested address is not mapped in the register map.
    AddressOutOfRange(u16),
    /// The target coil belongs to a read-only block.
    WriteProtected(u16),
    /// The requested count exceeds the Modbus-defined maximum (2000 for FC01).
    CountExceedsMax,
}

impl std::fmt::Display for CoilError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoilError::AddressOutOfRange(a) => write!(f, "address out of range: {a}"),
            CoilError::WriteProtected(a) => write!(f, "write protected at address: {a}"),
            CoilError::CountExceedsMax => write!(f, "count exceeds maximum allowed coils"),
        }
    }
}

impl std::error::Error for CoilError {}

/// Maximum number of coils that may be read in a single Modbus request (FC01).
pub const MAX_READ_COILS: u16 = 2000;

// ─────────────────────────────────────────────────────────────────────────────
// RegisterMap
// ─────────────────────────────────────────────────────────────────────────────

/// Combined Modbus coil and discrete-input register map.
///
/// Coils (FC01/FC05/FC15) support both reading and writing (subject to
/// read-only protection). Discrete inputs (FC02) are read-only from a Modbus
/// client perspective; they are set via [`RegisterMap::set_discrete_input`].
pub struct RegisterMap {
    /// Individual coil overrides — take precedence over `coil_blocks`.
    coils: HashMap<u16, bool>,
    /// Discrete input space (read-only from client perspective).
    discrete_inputs: HashMap<u16, bool>,
    /// Named contiguous coil blocks with optional write protection.
    coil_blocks: Vec<CoilBlock>,
}

impl Default for RegisterMap {
    fn default() -> Self {
        Self::new()
    }
}

impl RegisterMap {
    /// Create an empty register map.
    pub fn new() -> Self {
        Self {
            coils: HashMap::new(),
            discrete_inputs: HashMap::new(),
            coil_blocks: Vec::new(),
        }
    }

    /// Register a `CoilBlock`.  Coils within the block are accessible via the
    /// standard read/write methods.  Individual coil overrides in `self.coils`
    /// take precedence over block values.
    pub fn add_coil_block(&mut self, block: CoilBlock) {
        self.coil_blocks.push(block);
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Check whether `address` falls inside a read-only block.
    fn is_read_only(&self, address: u16) -> bool {
        self.coil_blocks.iter().any(|b| {
            b.read_only && address >= b.start_address && (address as u32) < b.end_address()
        })
    }

    /// Return the coil value from a registered block, if `address` is covered.
    fn block_read(&self, address: u16) -> Option<bool> {
        for block in &self.coil_blocks {
            if address >= block.start_address {
                let offset = (address - block.start_address) as usize;
                if offset < block.coils.len() {
                    return Some(block.coils[offset]);
                }
            }
        }
        None
    }

    /// Write a coil value into the matching block (if any).
    fn block_write(&mut self, address: u16, value: bool) {
        for block in &mut self.coil_blocks {
            if address >= block.start_address {
                let offset = (address - block.start_address) as usize;
                if offset < block.coils.len() {
                    block.coils[offset] = value;
                    return;
                }
            }
        }
    }

    /// Determine if `address` is mapped (either as an individual coil or
    /// within a registered block).
    fn coil_exists(&self, address: u16) -> bool {
        self.coils.contains_key(&address) || self.block_read(address).is_some()
    }

    // ── coil operations ───────────────────────────────────────────────────────

    /// Read the value of a single coil.
    ///
    /// Returns `None` if the address is not mapped.
    pub fn read_coil(&self, address: u16) -> Option<bool> {
        if let Some(&v) = self.coils.get(&address) {
            return Some(v);
        }
        self.block_read(address)
    }

    /// Write a single coil value.
    ///
    /// Fails with [`CoilError::WriteProtected`] if the address is inside a
    /// read-only block, and with [`CoilError::AddressOutOfRange`] if the
    /// address is not mapped at all.
    pub fn write_coil(&mut self, address: u16, value: bool) -> Result<(), CoilError> {
        if self.is_read_only(address) {
            return Err(CoilError::WriteProtected(address));
        }
        // Allow writing to the flat map; if the address is in a block, update
        // the block entry as well for consistency.
        if self.coil_exists(address) || self.coils.contains_key(&address) {
            self.coils.insert(address, value);
            self.block_write(address, value);
            Ok(())
        } else {
            // Not in the flat map and not in a block — address is unmapped.
            Err(CoilError::AddressOutOfRange(address))
        }
    }

    /// Read `count` contiguous coils starting at `address`.
    ///
    /// Returns [`CoilError::CountExceedsMax`] when `count > MAX_READ_COILS`.
    /// Returns [`CoilError::AddressOutOfRange`] if any address in the range is
    /// not mapped.
    pub fn read_coils(&self, address: u16, count: u16) -> Result<Vec<bool>, CoilError> {
        if count > MAX_READ_COILS {
            return Err(CoilError::CountExceedsMax);
        }
        let mut result = Vec::with_capacity(count as usize);
        for i in 0..count {
            let addr = address.wrapping_add(i);
            match self.read_coil(addr) {
                Some(v) => result.push(v),
                None => return Err(CoilError::AddressOutOfRange(addr)),
            }
        }
        Ok(result)
    }

    /// Write `values` to coils starting at `address`.
    ///
    /// Silently skips coils inside read-only blocks and counts them in
    /// [`WriteResult::skipped_read_only`].
    ///
    /// Returns [`CoilError::CountExceedsMax`] if `values.len() > MAX_READ_COILS`.
    /// Returns [`CoilError::AddressOutOfRange`] if any target address is not mapped
    /// (unless it is in a read-only block, in which case it is just skipped).
    pub fn write_coils(&mut self, address: u16, values: &[bool]) -> Result<WriteResult, CoilError> {
        if values.len() > MAX_READ_COILS as usize {
            return Err(CoilError::CountExceedsMax);
        }
        let mut written: u16 = 0;
        let mut skipped_read_only: u16 = 0;
        for (i, &value) in values.iter().enumerate() {
            let addr = address.wrapping_add(i as u16);
            if self.is_read_only(addr) {
                skipped_read_only += 1;
                continue;
            }
            if self.coil_exists(addr) || self.coils.contains_key(&addr) {
                self.coils.insert(addr, value);
                self.block_write(addr, value);
                written += 1;
            } else {
                return Err(CoilError::AddressOutOfRange(addr));
            }
        }
        Ok(WriteResult {
            written,
            skipped_read_only,
        })
    }

    // ── discrete inputs ───────────────────────────────────────────────────────

    /// Read a discrete input value.  Returns `None` if not mapped.
    pub fn read_discrete_input(&self, address: u16) -> Option<bool> {
        self.discrete_inputs.get(&address).copied()
    }

    /// Set a discrete input value (server/device side — not subject to
    /// read-only coil-block restrictions).
    pub fn set_discrete_input(&mut self, address: u16, value: bool) {
        self.discrete_inputs.insert(address, value);
    }

    // ── statistics ────────────────────────────────────────────────────────────

    /// Number of individually-mapped coils (does not include block coils).
    pub fn coil_count(&self) -> usize {
        self.coils.len()
    }

    /// Number of mapped discrete inputs.
    pub fn discrete_input_count(&self) -> usize {
        self.discrete_inputs.len()
    }

    // ── direct write ─────────────────────────────────────────────────────────

    /// Write a coil value bypassing all read-only checks.
    ///
    /// This is intended for server-side state injection (e.g. a simulated PLC
    /// updating its own coil state).
    pub fn set_coil_direct(&mut self, address: u16, value: bool) {
        self.coils.insert(address, value);
        self.block_write(address, value);
    }

    // ── packed coil serialisation ─────────────────────────────────────────────

    /// Pack `count` coils starting at `address` into bytes, LSB-first per
    /// byte, as required by the Modbus FC01/FC15 PDU encoding.
    ///
    /// Returns [`CoilError::CountExceedsMax`] if `count > MAX_READ_COILS`.
    /// Returns [`CoilError::AddressOutOfRange`] if any address in the range is
    /// not mapped.
    pub fn packed_coils(&self, address: u16, count: u16) -> Result<Vec<u8>, CoilError> {
        let coils = self.read_coils(address, count)?;
        let byte_count = (count as usize + 7) / 8;
        let mut bytes = vec![0u8; byte_count];
        for (i, &coil) in coils.iter().enumerate() {
            if coil {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        Ok(bytes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn writable_map_with_coils(addrs: &[u16]) -> RegisterMap {
        let mut rm = RegisterMap::new();
        for &a in addrs {
            rm.set_coil_direct(a, false);
        }
        rm
    }

    // ── read / write coil ─────────────────────────────────────────────────────

    #[test]
    fn test_write_and_read_coil() {
        let mut rm = RegisterMap::new();
        rm.set_coil_direct(100, false);
        rm.write_coil(100, true).unwrap();
        assert_eq!(rm.read_coil(100), Some(true));
    }

    #[test]
    fn test_read_unmapped_coil_returns_none() {
        let rm = RegisterMap::new();
        assert!(rm.read_coil(0).is_none());
    }

    #[test]
    fn test_write_unmapped_coil_errors() {
        let mut rm = RegisterMap::new();
        let res = rm.write_coil(42, true);
        assert_eq!(res, Err(CoilError::AddressOutOfRange(42)));
    }

    #[test]
    fn test_write_coil_false() {
        let mut rm = writable_map_with_coils(&[5]);
        rm.write_coil(5, true).unwrap();
        rm.write_coil(5, false).unwrap();
        assert_eq!(rm.read_coil(5), Some(false));
    }

    // ── read_coils (range) ────────────────────────────────────────────────────

    #[test]
    fn test_read_coils_range() {
        let mut rm = writable_map_with_coils(&[0, 1, 2, 3]);
        rm.write_coil(1, true).unwrap();
        rm.write_coil(3, true).unwrap();
        let vals = rm.read_coils(0, 4).unwrap();
        assert_eq!(vals, vec![false, true, false, true]);
    }

    #[test]
    fn test_read_coils_exceeds_max_errors() {
        let rm = RegisterMap::new();
        assert_eq!(rm.read_coils(0, 2001), Err(CoilError::CountExceedsMax));
    }

    #[test]
    fn test_read_coils_address_out_of_range() {
        let rm = writable_map_with_coils(&[0]);
        let res = rm.read_coils(0, 2); // address 1 is not mapped
        assert!(matches!(res, Err(CoilError::AddressOutOfRange(1))));
    }

    #[test]
    fn test_read_coils_empty_count() {
        let rm = RegisterMap::new();
        let vals = rm.read_coils(100, 0).unwrap();
        assert!(vals.is_empty());
    }

    // ── write_coils ───────────────────────────────────────────────────────────

    #[test]
    fn test_write_coils_multiple() {
        let mut rm = writable_map_with_coils(&[10, 11, 12]);
        let res = rm.write_coils(10, &[true, false, true]).unwrap();
        assert_eq!(res.written, 3);
        assert_eq!(res.skipped_read_only, 0);
        assert_eq!(rm.read_coil(10), Some(true));
        assert_eq!(rm.read_coil(11), Some(false));
        assert_eq!(rm.read_coil(12), Some(true));
    }

    #[test]
    fn test_write_coils_read_only_skipped() {
        let mut rm = RegisterMap::new();
        let block = CoilBlock {
            start_address: 0,
            coils: vec![false, false, false],
            read_only: true,
        };
        rm.add_coil_block(block);
        // Also add a writable coil at address 3
        rm.set_coil_direct(3, false);
        let res = rm.write_coils(0, &[true, true, true, true]).unwrap();
        assert_eq!(res.skipped_read_only, 3);
        assert_eq!(res.written, 1);
    }

    #[test]
    fn test_write_coils_exceeds_max_errors() {
        let mut rm = RegisterMap::new();
        let values = vec![false; 2001];
        assert_eq!(rm.write_coils(0, &values), Err(CoilError::CountExceedsMax));
    }

    #[test]
    fn test_write_coils_address_out_of_range() {
        let mut rm = writable_map_with_coils(&[0]);
        let res = rm.write_coils(0, &[true, false]); // address 1 not mapped
        assert!(matches!(res, Err(CoilError::AddressOutOfRange(1))));
    }

    // ── read-only block ───────────────────────────────────────────────────────

    #[test]
    fn test_read_only_block_blocks_write_coil() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(100, 5, true));
        let res = rm.write_coil(102, true);
        assert_eq!(res, Err(CoilError::WriteProtected(102)));
    }

    #[test]
    fn test_read_only_block_read_allowed() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock {
            start_address: 0,
            coils: vec![true, false],
            read_only: true,
        });
        assert_eq!(rm.read_coil(0), Some(true));
        assert_eq!(rm.read_coil(1), Some(false));
    }

    #[test]
    fn test_set_coil_direct_bypasses_read_only() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(0, 4, true));
        // Direct write should bypass read-only guard
        rm.set_coil_direct(0, true);
        assert_eq!(rm.read_coil(0), Some(true));
    }

    #[test]
    fn test_writable_block_allows_write() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(200, 3, false));
        rm.write_coil(200, true).unwrap();
        assert_eq!(rm.read_coil(200), Some(true));
    }

    // ── add_coil_block ────────────────────────────────────────────────────────

    #[test]
    fn test_add_coil_block_accessible() {
        let mut rm = RegisterMap::new();
        let block = CoilBlock {
            start_address: 500,
            coils: vec![false, true, false],
            read_only: false,
        };
        rm.add_coil_block(block);
        assert_eq!(rm.read_coil(501), Some(true));
    }

    #[test]
    fn test_multiple_blocks() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(0, 4, false));
        rm.add_coil_block(CoilBlock::new(100, 4, false));
        rm.write_coil(2, true).unwrap();
        rm.write_coil(101, true).unwrap();
        assert_eq!(rm.read_coil(2), Some(true));
        assert_eq!(rm.read_coil(101), Some(true));
    }

    // ── discrete inputs ───────────────────────────────────────────────────────

    #[test]
    fn test_read_discrete_input() {
        let mut rm = RegisterMap::new();
        rm.set_discrete_input(10, true);
        assert_eq!(rm.read_discrete_input(10), Some(true));
    }

    #[test]
    fn test_read_unmapped_discrete_input_none() {
        let rm = RegisterMap::new();
        assert!(rm.read_discrete_input(0).is_none());
    }

    #[test]
    fn test_set_discrete_input_overwrite() {
        let mut rm = RegisterMap::new();
        rm.set_discrete_input(5, true);
        rm.set_discrete_input(5, false);
        assert_eq!(rm.read_discrete_input(5), Some(false));
    }

    #[test]
    fn test_discrete_input_count() {
        let mut rm = RegisterMap::new();
        rm.set_discrete_input(0, true);
        rm.set_discrete_input(1, false);
        assert_eq!(rm.discrete_input_count(), 2);
    }

    // ── coil_count ────────────────────────────────────────────────────────────

    #[test]
    fn test_coil_count_flat_map() {
        let mut rm = RegisterMap::new();
        rm.set_coil_direct(0, false);
        rm.set_coil_direct(1, true);
        assert_eq!(rm.coil_count(), 2);
    }

    #[test]
    fn test_coil_count_initial_zero() {
        let rm = RegisterMap::new();
        assert_eq!(rm.coil_count(), 0);
    }

    // ── packed_coils ──────────────────────────────────────────────────────────

    #[test]
    fn test_packed_coils_single_byte() {
        let mut rm = writable_map_with_coils(&[0, 1, 2, 3, 4, 5, 6, 7]);
        // Set coils 0, 2, 4 → bits 0, 2, 4 → byte = 0b00010101 = 0x15
        rm.write_coil(0, true).unwrap();
        rm.write_coil(2, true).unwrap();
        rm.write_coil(4, true).unwrap();
        let bytes = rm.packed_coils(0, 8).unwrap();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b0001_0101);
    }

    #[test]
    fn test_packed_coils_two_bytes() {
        let addrs: Vec<u16> = (0..9).collect();
        let mut rm = writable_map_with_coils(&addrs);
        // Bit 8 = address 8 in second byte (bit 0 of byte 1)
        rm.write_coil(8, true).unwrap();
        let bytes = rm.packed_coils(0, 9).unwrap();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[1], 0b0000_0001);
    }

    #[test]
    fn test_packed_coils_all_false() {
        let addrs: Vec<u16> = (0..4).collect();
        let rm = writable_map_with_coils(&addrs);
        let bytes = rm.packed_coils(0, 4).unwrap();
        assert_eq!(bytes, vec![0u8]);
    }

    #[test]
    fn test_packed_coils_count_exceeds_max_errors() {
        let rm = RegisterMap::new();
        assert_eq!(rm.packed_coils(0, 2001), Err(CoilError::CountExceedsMax));
    }

    #[test]
    fn test_packed_coils_address_not_mapped_errors() {
        let rm = RegisterMap::new();
        assert!(matches!(
            rm.packed_coils(100, 1),
            Err(CoilError::AddressOutOfRange(100))
        ));
    }

    // ── WriteResult ───────────────────────────────────────────────────────────

    #[test]
    fn test_write_result_all_written() {
        let mut rm = writable_map_with_coils(&[0, 1, 2]);
        let res = rm.write_coils(0, &[true, true, true]).unwrap();
        assert_eq!(
            res,
            WriteResult {
                written: 3,
                skipped_read_only: 0
            }
        );
    }

    #[test]
    fn test_write_result_all_skipped() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(0, 4, true));
        let res = rm.write_coils(0, &[true, true, true, true]).unwrap();
        assert_eq!(
            res,
            WriteResult {
                written: 0,
                skipped_read_only: 4
            }
        );
    }

    // ── additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn test_coil_block_end_address() {
        let b = CoilBlock::new(10, 5, false);
        assert_eq!(b.end_address(), 15);
    }

    #[test]
    fn test_coil_block_end_address_zero_coils() {
        let b = CoilBlock::new(100, 0, false);
        assert_eq!(b.end_address(), 100);
    }

    #[test]
    fn test_coil_error_display_address_out_of_range() {
        let e = CoilError::AddressOutOfRange(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_coil_error_display_write_protected() {
        let e = CoilError::WriteProtected(100);
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn test_coil_error_display_count_exceeds_max() {
        let e = CoilError::CountExceedsMax;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_read_coil_from_block_default_false() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(0, 8, false));
        // All initialised to false
        for i in 0..8u16 {
            assert_eq!(rm.read_coil(i), Some(false));
        }
    }

    #[test]
    fn test_write_coil_into_block_persists() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(50, 4, false));
        rm.write_coil(51, true).unwrap();
        // Verify via block-level read
        let vals = rm.read_coils(50, 4).unwrap();
        assert_eq!(vals, vec![false, true, false, false]);
    }

    #[test]
    fn test_discrete_input_count_after_multiple_set() {
        let mut rm = RegisterMap::new();
        for i in 0..10u16 {
            rm.set_discrete_input(i, i % 2 == 0);
        }
        assert_eq!(rm.discrete_input_count(), 10);
    }

    #[test]
    fn test_read_all_discrete_inputs_correct_values() {
        let mut rm = RegisterMap::new();
        rm.set_discrete_input(0, true);
        rm.set_discrete_input(1, false);
        rm.set_discrete_input(2, true);
        assert_eq!(rm.read_discrete_input(0), Some(true));
        assert_eq!(rm.read_discrete_input(1), Some(false));
        assert_eq!(rm.read_discrete_input(2), Some(true));
    }

    #[test]
    fn test_packed_coils_all_set() {
        let addrs: Vec<u16> = (0..8).collect();
        let mut rm = writable_map_with_coils(&addrs);
        for a in 0..8u16 {
            rm.write_coil(a, true).unwrap();
        }
        let bytes = rm.packed_coils(0, 8).unwrap();
        assert_eq!(bytes[0], 0xFF);
    }

    #[test]
    fn test_packed_coils_partial_byte() {
        // Only 3 coils, should fit in 1 byte with upper bits zero
        let addrs: Vec<u16> = (0..3).collect();
        let mut rm = writable_map_with_coils(&addrs);
        rm.write_coil(0, true).unwrap(); // bit 0
        rm.write_coil(2, true).unwrap(); // bit 2
        let bytes = rm.packed_coils(0, 3).unwrap();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0] & 0b111, 0b101);
    }

    #[test]
    fn test_write_coil_and_read_back_via_block() {
        let mut rm = RegisterMap::new();
        rm.add_coil_block(CoilBlock::new(1000, 10, false));
        rm.write_coil(1005, true).unwrap();
        assert_eq!(rm.read_coil(1005), Some(true));
    }

    #[test]
    fn test_default_register_map_is_empty() {
        let rm = RegisterMap::default();
        assert_eq!(rm.coil_count(), 0);
        assert_eq!(rm.discrete_input_count(), 0);
    }

    #[test]
    fn test_write_coils_empty_slice_ok() {
        let mut rm = RegisterMap::new();
        let res = rm.write_coils(0, &[]).unwrap();
        assert_eq!(res.written, 0);
        assert_eq!(res.skipped_read_only, 0);
    }

    #[test]
    fn test_coil_block_new_sets_read_only() {
        let b = CoilBlock::new(0, 4, true);
        assert!(b.read_only);
    }

    #[test]
    fn test_coil_block_new_not_read_only() {
        let b = CoilBlock::new(0, 4, false);
        assert!(!b.read_only);
    }
}
