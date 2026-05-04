//! Discovery driver: probes a Modbus device across a configurable address
//! window and collects raw register data for subsequent type inference.
//!
//! The driver is deliberately synchronous and generic over a [`ModbusAccess`]
//! trait so it can be driven by a real Modbus TCP/RTU connection **or** an
//! in-process mock in unit tests.

use std::time::Duration;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that may occur during register discovery.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// The device returned a Modbus exception for the given function code and address.
    #[error("Device exception (FC {function_code:#04x} at {address}): {message}")]
    DeviceException {
        /// Modbus function code that triggered the exception.
        function_code: u8,
        /// Starting address of the request.
        address: u16,
        /// Human-readable description of the exception.
        message: String,
    },

    /// An I/O error occurred at the transport layer.
    #[error("IO error: {0}")]
    Io(String),

    /// The request timed out before a response arrived.
    #[error("Timeout")]
    Timeout,
}

// ─── Function codes ───────────────────────────────────────────────────────────

/// Modbus function codes supported by the discovery driver.
///
/// Only the four *read* function codes are used during passive discovery; write
/// codes are intentionally excluded to avoid mutating device state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiscoveryFunctionCode {
    /// FC 0x01 — Read Coils.
    ReadCoils = 0x01,
    /// FC 0x02 — Read Discrete Inputs.
    ReadDiscreteInputs = 0x02,
    /// FC 0x03 — Read Holding Registers.
    ReadHoldingRegisters = 0x03,
    /// FC 0x04 — Read Input Registers.
    ReadInputRegisters = 0x04,
}

impl DiscoveryFunctionCode {
    /// All four discovery function codes in scan order.
    pub fn all() -> &'static [DiscoveryFunctionCode] {
        &[
            DiscoveryFunctionCode::ReadCoils,
            DiscoveryFunctionCode::ReadDiscreteInputs,
            DiscoveryFunctionCode::ReadHoldingRegisters,
            DiscoveryFunctionCode::ReadInputRegisters,
        ]
    }

    /// Raw byte value of this function code.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ─── Discovered register ─────────────────────────────────────────────────────

/// A single register value captured during a discovery scan.
#[derive(Debug, Clone)]
pub struct DiscoveredRegister {
    /// The Modbus function code used to read this register.
    pub function_code: DiscoveryFunctionCode,
    /// The register address (0-based).
    pub address: u16,
    /// Raw 16-bit register value (big-endian from device).
    pub raw_u16: u16,
    /// The same value as two big-endian bytes `[hi, lo]`.
    pub raw_bytes: [u8; 2],
}

impl DiscoveredRegister {
    /// Construct a [`DiscoveredRegister`] from a raw 16-bit value.
    pub fn new(fc: DiscoveryFunctionCode, address: u16, raw: u16) -> Self {
        let raw_bytes = raw.to_be_bytes();
        DiscoveredRegister {
            function_code: fc,
            address,
            raw_u16: raw,
            raw_bytes,
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a single discovery scan pass.
///
/// Adjust [`address_start`] / [`address_end`] to narrow the search space and
/// [`read_batch_size`] to match the device's max-register-per-request limit.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Inclusive start address for the scan.
    pub address_start: u16,
    /// Exclusive end address for the scan.
    pub address_end: u16,
    /// Maximum number of registers to request in a single read.
    ///
    /// Must be ≥ 1. Values above 125 are clamped to comply with Modbus spec.
    pub read_batch_size: u16,
    /// Delay between successive read requests to avoid overwhelming slow devices.
    ///
    /// Set to zero in unit tests; use 10–50 ms against real PLCs.
    pub inter_request_delay: Duration,
    /// Modbus unit / slave ID to interrogate.
    pub unit_id: u8,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        DiscoveryConfig {
            address_start: 0,
            address_end: 100,
            read_batch_size: 10,
            inter_request_delay: Duration::from_millis(10),
            unit_id: 1,
        }
    }
}

// ─── Access trait ─────────────────────────────────────────────────────────────

/// Synchronous Modbus register access interface.
///
/// Implementors can be a real TCP/RTU client or an in-process mock.
/// The trait is deliberately synchronous: the discovery driver runs in a
/// single-threaded blocking context, making it easy to test without Tokio.
pub trait ModbusAccess: Send + Sync {
    /// FC 0x03 — Read Holding Registers.
    fn read_holding_registers(
        &mut self,
        address: u16,
        count: u16,
    ) -> Result<Vec<u16>, DiscoveryError>;

    /// FC 0x04 — Read Input Registers.
    fn read_input_registers(
        &mut self,
        address: u16,
        count: u16,
    ) -> Result<Vec<u16>, DiscoveryError>;

    /// FC 0x01 — Read Coils.
    fn read_coils(&mut self, address: u16, count: u16) -> Result<Vec<bool>, DiscoveryError>;

    /// FC 0x02 — Read Discrete Inputs.
    fn read_discrete_inputs(
        &mut self,
        address: u16,
        count: u16,
    ) -> Result<Vec<bool>, DiscoveryError>;
}

// ─── Driver ──────────────────────────────────────────────────────────────────

/// The main register discovery driver.
///
/// Walks FC 0x03 (holding) and FC 0x04 (input) across the configured address
/// window. On [`DiscoveryError::DeviceException`] the problematic batch is
/// silently skipped so that sparse register maps are handled gracefully.
///
/// # Example (unit test with mock)
///
/// ```no_run
/// use oxirs_modbus::discovery::{DiscoveryDriver, DiscoveryConfig, driver::ModbusAccess, driver::DiscoveryError};
///
/// struct AlwaysZero;
/// impl ModbusAccess for AlwaysZero {
///     fn read_holding_registers(&mut self, _a: u16, count: u16) -> Result<Vec<u16>, DiscoveryError> {
///         Ok(vec![0u16; count as usize])
///     }
///     fn read_input_registers(&mut self, _a: u16, count: u16) -> Result<Vec<u16>, DiscoveryError> {
///         Ok(vec![0u16; count as usize])
///     }
///     fn read_coils(&mut self, _a: u16, count: u16) -> Result<Vec<bool>, DiscoveryError> {
///         Ok(vec![false; count as usize])
///     }
///     fn read_discrete_inputs(&mut self, _a: u16, count: u16) -> Result<Vec<bool>, DiscoveryError> {
///         Ok(vec![false; count as usize])
///     }
/// }
///
/// let cfg = DiscoveryConfig { address_start: 0, address_end: 5, read_batch_size: 5, ..Default::default() };
/// let mut driver = DiscoveryDriver::new(AlwaysZero, cfg);
/// let results = driver.scan().expect("scan failed");
/// assert!(!results.is_empty());
/// ```
pub struct DiscoveryDriver<T: ModbusAccess> {
    access: T,
    config: DiscoveryConfig,
}

impl<T: ModbusAccess> DiscoveryDriver<T> {
    /// Create a new driver with the given access implementation and configuration.
    pub fn new(access: T, config: DiscoveryConfig) -> Self {
        DiscoveryDriver { access, config }
    }

    /// Run a full scan over the configured address window.
    ///
    /// Holding registers (FC 0x03) are scanned first, then input registers
    /// (FC 0x04). Batches that trigger a [`DiscoveryError::DeviceException`]
    /// are silently skipped. Any other error is returned immediately.
    pub fn scan(&mut self) -> Result<Vec<DiscoveredRegister>, DiscoveryError> {
        let mut registers = Vec::new();

        self.scan_holding_registers(&mut registers)?;
        self.scan_input_registers(&mut registers)?;

        Ok(registers)
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn scan_holding_registers(
        &mut self,
        out: &mut Vec<DiscoveredRegister>,
    ) -> Result<(), DiscoveryError> {
        let start = self.config.address_start;
        let end = self.config.address_end;
        let batch = self.config.read_batch_size.clamp(1, 125);

        let mut addr = start;
        while addr < end {
            let count = batch.min(end - addr);
            match self.access.read_holding_registers(addr, count) {
                Ok(values) => {
                    for (i, v) in values.into_iter().enumerate() {
                        out.push(DiscoveredRegister::new(
                            DiscoveryFunctionCode::ReadHoldingRegisters,
                            addr + i as u16,
                            v,
                        ));
                    }
                }
                Err(DiscoveryError::DeviceException { .. }) => {
                    // Register block not supported on this device — skip.
                }
                Err(other) => return Err(other),
            }
            addr = addr.saturating_add(count);
        }

        Ok(())
    }

    fn scan_input_registers(
        &mut self,
        out: &mut Vec<DiscoveredRegister>,
    ) -> Result<(), DiscoveryError> {
        let start = self.config.address_start;
        let end = self.config.address_end;
        let batch = self.config.read_batch_size.clamp(1, 125);

        let mut addr = start;
        while addr < end {
            let count = batch.min(end - addr);
            match self.access.read_input_registers(addr, count) {
                Ok(values) => {
                    for (i, v) in values.into_iter().enumerate() {
                        out.push(DiscoveredRegister::new(
                            DiscoveryFunctionCode::ReadInputRegisters,
                            addr + i as u16,
                            v,
                        ));
                    }
                }
                Err(DiscoveryError::DeviceException { .. }) => {}
                Err(other) => return Err(other),
            }
            addr = addr.saturating_add(count);
        }

        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantDevice(u16);

    impl ModbusAccess for ConstantDevice {
        fn read_holding_registers(
            &mut self,
            _address: u16,
            count: u16,
        ) -> Result<Vec<u16>, DiscoveryError> {
            Ok(vec![self.0; count as usize])
        }

        fn read_input_registers(
            &mut self,
            _address: u16,
            count: u16,
        ) -> Result<Vec<u16>, DiscoveryError> {
            Ok(vec![self.0; count as usize])
        }

        fn read_coils(&mut self, _address: u16, count: u16) -> Result<Vec<bool>, DiscoveryError> {
            Ok(vec![false; count as usize])
        }

        fn read_discrete_inputs(
            &mut self,
            _address: u16,
            count: u16,
        ) -> Result<Vec<bool>, DiscoveryError> {
            Ok(vec![false; count as usize])
        }
    }

    struct ExceptionDevice;

    impl ModbusAccess for ExceptionDevice {
        fn read_holding_registers(
            &mut self,
            address: u16,
            _count: u16,
        ) -> Result<Vec<u16>, DiscoveryError> {
            Err(DiscoveryError::DeviceException {
                function_code: 0x03,
                address,
                message: "Illegal data address".into(),
            })
        }

        fn read_input_registers(
            &mut self,
            address: u16,
            _count: u16,
        ) -> Result<Vec<u16>, DiscoveryError> {
            Err(DiscoveryError::DeviceException {
                function_code: 0x04,
                address,
                message: "Illegal data address".into(),
            })
        }

        fn read_coils(&mut self, _address: u16, _count: u16) -> Result<Vec<bool>, DiscoveryError> {
            Err(DiscoveryError::DeviceException {
                function_code: 0x01,
                address: 0,
                message: "Not supported".into(),
            })
        }

        fn read_discrete_inputs(
            &mut self,
            _address: u16,
            _count: u16,
        ) -> Result<Vec<bool>, DiscoveryError> {
            Err(DiscoveryError::DeviceException {
                function_code: 0x02,
                address: 0,
                message: "Not supported".into(),
            })
        }
    }

    #[test]
    fn scan_returns_holding_and_input_registers() {
        let cfg = DiscoveryConfig {
            address_start: 0,
            address_end: 5,
            read_batch_size: 5,
            inter_request_delay: Duration::ZERO,
            unit_id: 1,
        };
        let mut driver = DiscoveryDriver::new(ConstantDevice(42), cfg);
        let result = driver.scan().expect("scan should succeed");

        let holding: Vec<_> = result
            .iter()
            .filter(|r| r.function_code == DiscoveryFunctionCode::ReadHoldingRegisters)
            .collect();
        let input: Vec<_> = result
            .iter()
            .filter(|r| r.function_code == DiscoveryFunctionCode::ReadInputRegisters)
            .collect();

        assert_eq!(holding.len(), 5);
        assert_eq!(input.len(), 5);
        assert!(holding.iter().all(|r| r.raw_u16 == 42));
    }

    #[test]
    fn scan_skips_exception_batches() {
        let cfg = DiscoveryConfig {
            address_start: 0,
            address_end: 10,
            read_batch_size: 10,
            inter_request_delay: Duration::ZERO,
            unit_id: 1,
        };
        let mut driver = DiscoveryDriver::new(ExceptionDevice, cfg);
        let result = driver
            .scan()
            .expect("scan should succeed (exceptions are skipped)");
        assert!(result.is_empty());
    }

    #[test]
    fn scan_respects_address_window() {
        let cfg = DiscoveryConfig {
            address_start: 10,
            address_end: 15,
            read_batch_size: 5,
            inter_request_delay: Duration::ZERO,
            unit_id: 1,
        };
        let mut driver = DiscoveryDriver::new(ConstantDevice(0), cfg);
        let result = driver.scan().expect("scan should succeed");

        // Addresses should all be in [10, 15)
        for reg in &result {
            assert!(
                reg.address >= 10 && reg.address < 15,
                "address {} out of window",
                reg.address
            );
        }
    }

    #[test]
    fn discovered_register_raw_bytes_match_value() {
        let reg = DiscoveredRegister::new(DiscoveryFunctionCode::ReadHoldingRegisters, 0, 0xABCD);
        assert_eq!(reg.raw_bytes, [0xAB, 0xCD]);
    }

    #[test]
    fn discovery_function_code_as_u8() {
        assert_eq!(DiscoveryFunctionCode::ReadHoldingRegisters.as_u8(), 0x03);
        assert_eq!(DiscoveryFunctionCode::ReadInputRegisters.as_u8(), 0x04);
        assert_eq!(DiscoveryFunctionCode::ReadCoils.as_u8(), 0x01);
        assert_eq!(DiscoveryFunctionCode::ReadDiscreteInputs.as_u8(), 0x02);
    }

    #[test]
    fn discovery_config_defaults_are_sensible() {
        let cfg = DiscoveryConfig::default();
        assert_eq!(cfg.address_start, 0);
        assert!(cfg.address_end > 0);
        assert!(cfg.read_batch_size > 0);
        assert_eq!(cfg.unit_id, 1);
    }
}
