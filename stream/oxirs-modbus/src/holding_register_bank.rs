//! Modbus holding register bank (FC03 / FC06 / FC16).
//!
//! Provides an in-memory bank of 16-bit holding registers with optional
//! write-protection, range checking, and per-register timestamps.

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single holding register value with its last-update timestamp.
#[derive(Debug, Clone)]
pub struct RegisterValue {
    pub raw: u16,
    pub timestamp_ms: u64,
}

/// Error variants for register bank operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterError {
    /// The requested address is outside the bank's range.
    AddressOutOfRange(u16),
    /// The register at this address is write-protected.
    WriteProtected(u16),
    /// The requested count is zero or exceeds the available range.
    InvalidCount,
}

impl std::fmt::Display for RegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterError::AddressOutOfRange(addr) => {
                write!(f, "address 0x{:04X} is out of range", addr)
            }
            RegisterError::WriteProtected(addr) => {
                write!(f, "register 0x{:04X} is write-protected", addr)
            }
            RegisterError::InvalidCount => write!(f, "invalid register count"),
        }
    }
}

impl std::error::Error for RegisterError {}

/// Return value for multi-register reads.
#[derive(Debug, Clone)]
pub struct ReadResult {
    pub values: Vec<u16>,
    pub timestamp_ms: u64,
}

/// Return value for multi-register writes.
#[derive(Debug, Clone)]
pub struct WriteResult {
    pub written_count: usize,
    pub skipped_protected: usize,
}

/// An in-memory Modbus holding register bank.
pub struct HoldingRegisterBank {
    registers: Vec<RegisterValue>,
    start_address: u16,
    write_protected: Vec<bool>,
}

impl HoldingRegisterBank {
    /// Create a bank of `count` registers starting at `start_address`, all
    /// initialised to zero.
    pub fn new(start_address: u16, count: usize) -> Self {
        Self {
            registers: vec![
                RegisterValue {
                    raw: 0,
                    timestamp_ms: 0
                };
                count
            ],
            start_address,
            write_protected: vec![false; count],
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Convert an absolute Modbus address to a bank-relative index,
    /// returning `RegisterError::AddressOutOfRange` if it is out of range.
    fn index(&self, address: u16) -> Result<usize, RegisterError> {
        address
            .checked_sub(self.start_address)
            .and_then(|offset| {
                let idx = offset as usize;
                if idx < self.registers.len() {
                    Some(idx)
                } else {
                    None
                }
            })
            .ok_or(RegisterError::AddressOutOfRange(address))
    }

    // ── public API ───────────────────────────────────────────────────────────

    /// FC03: Read `count` consecutive registers starting at `address`.
    pub fn read(
        &self,
        address: u16,
        count: u16,
        current_time_ms: u64,
    ) -> Result<ReadResult, RegisterError> {
        if count == 0 {
            return Err(RegisterError::InvalidCount);
        }
        let start_idx = self.index(address)?;
        let end_idx = start_idx
            .checked_add(count as usize)
            .filter(|&end| end <= self.registers.len())
            .ok_or(RegisterError::InvalidCount)?;

        let values: Vec<u16> = self.registers[start_idx..end_idx]
            .iter()
            .map(|r| r.raw)
            .collect();

        Ok(ReadResult {
            values,
            timestamp_ms: current_time_ms,
        })
    }

    /// FC06: Write a single register at `address`.
    ///
    /// Returns `WriteProtected` if the register is protected.
    pub fn write_single(
        &mut self,
        address: u16,
        value: u16,
        current_time_ms: u64,
    ) -> Result<(), RegisterError> {
        let idx = self.index(address)?;
        if self.write_protected[idx] {
            return Err(RegisterError::WriteProtected(address));
        }
        self.registers[idx] = RegisterValue {
            raw: value,
            timestamp_ms: current_time_ms,
        };
        Ok(())
    }

    /// FC16: Write multiple consecutive registers starting at `address`.
    ///
    /// Protected registers are skipped (not written) and counted in
    /// `skipped_protected`.
    pub fn write_multiple(
        &mut self,
        address: u16,
        values: &[u16],
        current_time_ms: u64,
    ) -> Result<WriteResult, RegisterError> {
        if values.is_empty() {
            return Err(RegisterError::InvalidCount);
        }
        let start_idx = self.index(address)?;
        let end_idx = start_idx
            .checked_add(values.len())
            .filter(|&end| end <= self.registers.len())
            .ok_or(RegisterError::InvalidCount)?;

        let mut written_count = 0;
        let mut skipped_protected = 0;

        for (i, &val) in values.iter().enumerate() {
            let reg_idx = start_idx + i;
            if self.write_protected[reg_idx] {
                skipped_protected += 1;
            } else {
                self.registers[reg_idx] = RegisterValue {
                    raw: val,
                    timestamp_ms: current_time_ms,
                };
                written_count += 1;
            }
        }

        // Validate end index was computed (used to satisfy borrowck above)
        let _ = end_idx;

        Ok(WriteResult {
            written_count,
            skipped_protected,
        })
    }

    /// Set or clear the write-protection flag for a single register.
    pub fn set_write_protected(
        &mut self,
        address: u16,
        protected: bool,
    ) -> Result<(), RegisterError> {
        let idx = self.index(address)?;
        self.write_protected[idx] = protected;
        Ok(())
    }

    /// Return the raw value of a register without a timestamp.
    pub fn get_raw(&self, address: u16) -> Result<u16, RegisterError> {
        let idx = self.index(address)?;
        Ok(self.registers[idx].raw)
    }

    /// Total number of registers in the bank.
    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    /// The base (first) address of the bank.
    pub fn start_address(&self) -> u16 {
        self.start_address
    }

    /// Reset all registers to 0 and clear their timestamps and protection flags.
    pub fn clear(&mut self) {
        for reg in &mut self.registers {
            reg.raw = 0;
            reg.timestamp_ms = 0;
        }
        for flag in &mut self.write_protected {
            *flag = false;
        }
    }

    /// Return the `timestamp_ms` of the last write to the given register.
    pub fn last_written_at(&self, address: u16) -> Result<u64, RegisterError> {
        let idx = self.index(address)?;
        Ok(self.registers[idx].timestamp_ms)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bank() -> HoldingRegisterBank {
        HoldingRegisterBank::new(100, 20)
    }

    // ── read ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_read_single_default_zero() {
        let b = bank();
        let r = b.read(100, 1, 50).expect("should succeed");
        assert_eq!(r.values, &[0]);
        assert_eq!(r.timestamp_ms, 50);
    }

    #[test]
    fn test_read_multiple_defaults() {
        let b = bank();
        let r = b.read(100, 5, 0).expect("should succeed");
        assert_eq!(r.values.len(), 5);
        assert!(r.values.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_read_last_register() {
        let b = bank(); // 100..119
        let r = b.read(119, 1, 0).expect("should succeed");
        assert_eq!(r.values, &[0]);
    }

    #[test]
    fn test_read_address_out_of_range() {
        let b = bank();
        let e = b.read(120, 1, 0).unwrap_err();
        assert!(matches!(e, RegisterError::AddressOutOfRange(120)));
    }

    #[test]
    fn test_read_count_zero_error() {
        let b = bank();
        let e = b.read(100, 0, 0).unwrap_err();
        assert!(matches!(e, RegisterError::InvalidCount));
    }

    #[test]
    fn test_read_count_exceeds_range_error() {
        let b = bank(); // 20 registers
        let e = b.read(100, 21, 0).unwrap_err();
        assert!(matches!(e, RegisterError::InvalidCount));
    }

    #[test]
    fn test_read_correct_values_after_write() {
        let mut b = bank();
        b.write_single(102, 0xABCD, 10).expect("should succeed");
        b.write_single(103, 0x1234, 10).expect("should succeed");
        let r = b.read(102, 2, 20).expect("should succeed");
        assert_eq!(r.values, &[0xABCD, 0x1234]);
    }

    // ── write_single ─────────────────────────────────────────────────────────

    #[test]
    fn test_write_single_basic() {
        let mut b = bank();
        b.write_single(105, 42, 99).expect("should succeed");
        assert_eq!(b.get_raw(105).expect("should succeed"), 42);
    }

    #[test]
    fn test_write_single_updates_timestamp() {
        let mut b = bank();
        b.write_single(100, 1, 1234).expect("should succeed");
        assert_eq!(b.last_written_at(100).expect("should succeed"), 1234);
    }

    #[test]
    fn test_write_single_address_out_of_range() {
        let mut b = bank();
        let e = b.write_single(200, 1, 0).unwrap_err();
        assert!(matches!(e, RegisterError::AddressOutOfRange(200)));
    }

    #[test]
    fn test_write_single_write_protected() {
        let mut b = bank();
        b.set_write_protected(110, true).expect("should succeed");
        let e = b.write_single(110, 99, 0).unwrap_err();
        assert!(matches!(e, RegisterError::WriteProtected(110)));
    }

    // ── write_multiple ───────────────────────────────────────────────────────

    #[test]
    fn test_write_multiple_basic() {
        let mut b = bank();
        let r = b
            .write_multiple(100, &[1, 2, 3], 10)
            .expect("should succeed");
        assert_eq!(r.written_count, 3);
        assert_eq!(r.skipped_protected, 0);
        assert_eq!(b.get_raw(100).expect("should succeed"), 1);
        assert_eq!(b.get_raw(101).expect("should succeed"), 2);
        assert_eq!(b.get_raw(102).expect("should succeed"), 3);
    }

    #[test]
    fn test_write_multiple_skips_protected() {
        let mut b = bank();
        b.set_write_protected(101, true).expect("should succeed");
        let r = b
            .write_multiple(100, &[10, 20, 30], 0)
            .expect("should succeed");
        assert_eq!(r.written_count, 2);
        assert_eq!(r.skipped_protected, 1);
        assert_eq!(b.get_raw(100).expect("should succeed"), 10);
        assert_eq!(b.get_raw(101).expect("should succeed"), 0); // protected: unchanged
        assert_eq!(b.get_raw(102).expect("should succeed"), 30);
    }

    #[test]
    fn test_write_multiple_empty_values_error() {
        let mut b = bank();
        let e = b.write_multiple(100, &[], 0).unwrap_err();
        assert!(matches!(e, RegisterError::InvalidCount));
    }

    #[test]
    fn test_write_multiple_exceeds_range() {
        let mut b = bank(); // 20 regs: 100-119
        let values: Vec<u16> = vec![0; 21];
        let e = b.write_multiple(100, &values, 0).unwrap_err();
        assert!(matches!(e, RegisterError::InvalidCount));
    }

    #[test]
    fn test_write_multiple_address_out_of_range() {
        let mut b = bank();
        let e = b.write_multiple(200, &[1, 2], 0).unwrap_err();
        assert!(matches!(e, RegisterError::AddressOutOfRange(200)));
    }

    // ── set_write_protected ──────────────────────────────────────────────────

    #[test]
    fn test_set_write_protected_prevents_write() {
        let mut b = bank();
        b.set_write_protected(115, true).expect("should succeed");
        assert!(matches!(
            b.write_single(115, 1, 0),
            Err(RegisterError::WriteProtected(115))
        ));
    }

    #[test]
    fn test_set_write_protected_false_allows_write() {
        let mut b = bank();
        b.set_write_protected(115, true).expect("should succeed");
        b.set_write_protected(115, false).expect("should succeed");
        b.write_single(115, 77, 0).expect("should succeed");
        assert_eq!(b.get_raw(115).expect("should succeed"), 77);
    }

    #[test]
    fn test_set_write_protected_out_of_range() {
        let mut b = bank();
        let e = b.set_write_protected(200, true).unwrap_err();
        assert!(matches!(e, RegisterError::AddressOutOfRange(200)));
    }

    // ── get_raw ──────────────────────────────────────────────────────────────

    #[test]
    fn test_get_raw_initial_zero() {
        let b = bank();
        assert_eq!(b.get_raw(100).expect("should succeed"), 0);
    }

    #[test]
    fn test_get_raw_out_of_range() {
        let b = bank();
        assert!(matches!(
            b.get_raw(99),
            Err(RegisterError::AddressOutOfRange(99))
        ));
    }

    // ── clear ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_all_registers() {
        let mut b = bank();
        for i in 0..10 {
            b.write_single(100 + i as u16, (i + 1) as u16, 100)
                .expect("should succeed");
        }
        b.clear();
        for i in 0..20 {
            assert_eq!(b.get_raw(100 + i as u16).expect("should succeed"), 0);
            assert_eq!(
                b.last_written_at(100 + i as u16).expect("should succeed"),
                0
            );
        }
    }

    #[test]
    fn test_clear_removes_write_protection() {
        let mut b = bank();
        b.set_write_protected(105, true).expect("should succeed");
        b.clear();
        // After clear, the register should be writable again
        b.write_single(105, 1, 0).expect("should succeed");
        assert_eq!(b.get_raw(105).expect("should succeed"), 1);
    }

    // ── last_written_at ──────────────────────────────────────────────────────

    #[test]
    fn test_last_written_at_initial_zero() {
        let b = bank();
        assert_eq!(b.last_written_at(100).expect("should succeed"), 0);
    }

    #[test]
    fn test_last_written_at_after_write() {
        let mut b = bank();
        b.write_single(108, 5, 9876).expect("should succeed");
        assert_eq!(b.last_written_at(108).expect("should succeed"), 9876);
    }

    #[test]
    fn test_last_written_at_out_of_range() {
        let b = bank();
        assert!(matches!(
            b.last_written_at(50),
            Err(RegisterError::AddressOutOfRange(50))
        ));
    }

    // ── meta ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_register_count() {
        let b = HoldingRegisterBank::new(0, 50);
        assert_eq!(b.register_count(), 50);
    }

    #[test]
    fn test_start_address() {
        let b = HoldingRegisterBank::new(400, 10);
        assert_eq!(b.start_address(), 400);
    }

    #[test]
    fn test_boundary_address_at_start() {
        let mut b = HoldingRegisterBank::new(0, 5);
        b.write_single(0, 0xFFFF, 1).expect("should succeed");
        assert_eq!(b.get_raw(0).expect("should succeed"), 0xFFFF);
    }

    #[test]
    fn test_boundary_address_before_start_errors() {
        let b = HoldingRegisterBank::new(10, 5);
        assert!(matches!(
            b.get_raw(9),
            Err(RegisterError::AddressOutOfRange(9))
        ));
    }

    #[test]
    fn test_error_display_address_out_of_range() {
        let e = RegisterError::AddressOutOfRange(0x1234);
        assert!(e.to_string().contains("1234"));
    }

    #[test]
    fn test_error_display_write_protected() {
        let e = RegisterError::WriteProtected(0x5678);
        assert!(e.to_string().contains("5678"));
    }

    #[test]
    fn test_error_display_invalid_count() {
        let e = RegisterError::InvalidCount;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_multi_register_read_boundary() {
        let b = HoldingRegisterBank::new(100, 20); // 100..119
                                                   // Read exactly to the last register: start=110, count=10 → indices 110..119 inclusive
        let r = b.read(110, 10, 0).expect("should succeed");
        assert_eq!(r.values.len(), 10);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_write_single_multiple_addresses() {
        let mut b = bank();
        b.write_single(100, 0xAAAA, 1).expect("should succeed");
        b.write_single(110, 0xBBBB, 2).expect("should succeed");
        b.write_single(119, 0xCCCC, 3).expect("should succeed");
        assert_eq!(b.get_raw(100).expect("should succeed"), 0xAAAA);
        assert_eq!(b.get_raw(110).expect("should succeed"), 0xBBBB);
        assert_eq!(b.get_raw(119).expect("should succeed"), 0xCCCC);
    }

    #[test]
    fn test_read_timestamp_is_current_time() {
        let b = bank();
        let ts = 9876543;
        let r = b.read(100, 1, ts).expect("should succeed");
        assert_eq!(r.timestamp_ms, ts);
    }

    #[test]
    fn test_write_multiple_all_protected() {
        let mut b = bank();
        b.set_write_protected(100, true).expect("should succeed");
        b.set_write_protected(101, true).expect("should succeed");
        b.set_write_protected(102, true).expect("should succeed");
        let r = b
            .write_multiple(100, &[1, 2, 3], 0)
            .expect("should succeed");
        assert_eq!(r.written_count, 0);
        assert_eq!(r.skipped_protected, 3);
    }

    #[test]
    fn test_clear_does_not_affect_bank_dimensions() {
        let mut b = HoldingRegisterBank::new(200, 30);
        b.clear();
        assert_eq!(b.register_count(), 30);
        assert_eq!(b.start_address(), 200);
    }

    #[test]
    fn test_read_max_value() {
        let mut b = bank();
        b.write_single(100, u16::MAX, 0).expect("should succeed");
        let r = b.read(100, 1, 0).expect("should succeed");
        assert_eq!(r.values[0], u16::MAX);
    }

    #[test]
    fn test_write_multiple_updates_timestamps() {
        let mut b = bank();
        let ts = 5555;
        b.write_multiple(100, &[1, 2, 3], ts)
            .expect("should succeed");
        assert_eq!(b.last_written_at(100).expect("should succeed"), ts);
        assert_eq!(b.last_written_at(101).expect("should succeed"), ts);
        assert_eq!(b.last_written_at(102).expect("should succeed"), ts);
    }

    #[test]
    fn test_single_register_bank() {
        let mut b = HoldingRegisterBank::new(0, 1);
        b.write_single(0, 42, 1).expect("should succeed");
        assert_eq!(b.get_raw(0).expect("should succeed"), 42);
        // Reading address 1 should be out of range
        assert!(matches!(
            b.read(1, 1, 0),
            Err(RegisterError::AddressOutOfRange(1))
        ));
    }

    #[test]
    fn test_set_and_clear_protection_multiple_times() {
        let mut b = bank();
        b.set_write_protected(105, true).expect("should succeed");
        b.set_write_protected(105, false).expect("should succeed");
        b.set_write_protected(105, true).expect("should succeed");
        assert!(matches!(
            b.write_single(105, 1, 0),
            Err(RegisterError::WriteProtected(105))
        ));
    }

    #[test]
    fn test_write_result_fields() {
        let mut b = bank();
        b.set_write_protected(101, true).expect("should succeed");
        let r = b
            .write_multiple(100, &[10, 20, 30], 0)
            .expect("should succeed");
        assert_eq!(r.written_count, 2);
        assert_eq!(r.skipped_protected, 1);
    }

    #[test]
    fn test_read_result_fields() {
        let mut b = bank();
        b.write_single(100, 7, 10).expect("should succeed");
        let r = b.read(100, 1, 999).expect("should succeed");
        assert_eq!(r.values, vec![7]);
        assert_eq!(r.timestamp_ms, 999);
    }

    #[test]
    fn test_large_bank_sequential_write_read() {
        let mut b = HoldingRegisterBank::new(0, 100);
        for i in 0..100u16 {
            b.write_single(i, i * 2, i as u64).expect("should succeed");
        }
        for i in 0..100u16 {
            assert_eq!(b.get_raw(i).expect("should succeed"), i * 2);
        }
    }
}
