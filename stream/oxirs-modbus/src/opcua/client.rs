//! Modbus client facade and mock implementation.
//!
//! The [`ModbusClientFacade`] trait isolates the bridge from any concrete
//! Modbus library so that tests can use the in-process [`MockModbusClient`]
//! without opening real TCP sockets.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Errors that can occur in the Modbus client facade.
#[derive(Debug, Error)]
pub enum ModbusClientError {
    /// The requested register address is not known to the mock.
    #[error("no register at address {0}")]
    RegisterNotFound(u16),

    /// The requested coil address is not known to the mock.
    #[error("no coil at address {0}")]
    CoilNotFound(u16),

    /// Not enough register data available at the given address.
    #[error(
        "insufficient data: requested {requested} registers at {addr}, only {available} available"
    )]
    InsufficientData {
        addr: u16,
        requested: u16,
        available: usize,
    },

    /// The underlying transport reported an error.
    #[error("transport error: {0}")]
    Transport(String),

    /// Generic catch-all.
    #[error("Modbus client error: {0}")]
    Other(String),
}

/// Thin abstraction over a Modbus client.
///
/// Implementations must be `Send + Sync` so that the bridge can share them
/// across tokio tasks.
#[async_trait]
pub trait ModbusClientFacade: Send + Sync {
    /// Read `count` holding registers (FC03) starting at `addr`.
    async fn read_holding_registers(
        &self,
        addr: u16,
        count: u16,
    ) -> Result<Vec<u16>, ModbusClientError>;

    /// Write `values` to consecutive holding registers starting at `addr` (FC16).
    async fn write_registers(&self, addr: u16, values: &[u16]) -> Result<(), ModbusClientError>;

    /// Read `count` coils (FC01) starting at `addr`.
    async fn read_coils(&self, addr: u16, count: u16) -> Result<Vec<bool>, ModbusClientError>;

    /// Write a single coil (FC05) at `addr`.
    async fn write_coil(&self, addr: u16, value: bool) -> Result<(), ModbusClientError>;
}

/// In-process mock Modbus client for use in tests.
///
/// Registers and coils are stored in `HashMap`s and can be pre-populated
/// before the test starts.
pub struct MockModbusClient {
    /// Holding registers keyed by start address; each entry is a contiguous
    /// block of register values accessible starting at that address.
    pub registers: Arc<Mutex<HashMap<u16, Vec<u16>>>>,
    /// Discrete coils keyed by address.
    pub coils: Arc<Mutex<HashMap<u16, bool>>>,
}

impl Default for MockModbusClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockModbusClient {
    /// Create a new empty mock client.
    pub fn new() -> Self {
        Self {
            registers: Arc::new(Mutex::new(HashMap::new())),
            coils: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Pre-populate `count` holding registers starting at `start_addr`.
    ///
    /// All registers in the range will be set to the given `values`.  The
    /// bridge reads registers one-at-a-time (or in short bursts), so it is
    /// easiest to store them under their individual addresses.
    pub fn set_register(&self, addr: u16, values: Vec<u16>) {
        self.registers
            .lock()
            .expect("registers lock poisoned")
            .insert(addr, values);
    }

    /// Pre-populate a single coil.
    pub fn set_coil(&self, addr: u16, value: bool) {
        self.coils
            .lock()
            .expect("coils lock poisoned")
            .insert(addr, value);
    }

    /// Read back the current registers at `addr`.
    pub fn get_register(&self, addr: u16) -> Option<Vec<u16>> {
        self.registers
            .lock()
            .expect("registers lock poisoned")
            .get(&addr)
            .cloned()
    }

    /// Read back the current coil at `addr`.
    pub fn get_coil(&self, addr: u16) -> Option<bool> {
        self.coils
            .lock()
            .expect("coils lock poisoned")
            .get(&addr)
            .copied()
    }
}

#[async_trait]
impl ModbusClientFacade for MockModbusClient {
    async fn read_holding_registers(
        &self,
        addr: u16,
        count: u16,
    ) -> Result<Vec<u16>, ModbusClientError> {
        let regs = self
            .registers
            .lock()
            .map_err(|_| ModbusClientError::Other("lock poisoned".to_owned()))?;

        if let Some(block) = regs.get(&addr) {
            let available = block.len();
            let needed = count as usize;
            if available < needed {
                return Err(ModbusClientError::InsufficientData {
                    addr,
                    requested: count,
                    available,
                });
            }
            return Ok(block[..needed].to_vec());
        }

        // Try sequential lookups for individual register addresses.
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let reg_addr = addr.wrapping_add(i);
            if let Some(block) = regs.get(&reg_addr) {
                out.push(*block.first().unwrap_or(&0));
            } else {
                return Err(ModbusClientError::RegisterNotFound(reg_addr));
            }
        }
        Ok(out)
    }

    async fn write_registers(&self, addr: u16, values: &[u16]) -> Result<(), ModbusClientError> {
        let mut regs = self
            .registers
            .lock()
            .map_err(|_| ModbusClientError::Other("lock poisoned".to_owned()))?;
        regs.insert(addr, values.to_vec());
        Ok(())
    }

    async fn read_coils(&self, addr: u16, count: u16) -> Result<Vec<bool>, ModbusClientError> {
        let coils = self
            .coils
            .lock()
            .map_err(|_| ModbusClientError::Other("lock poisoned".to_owned()))?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let c_addr = addr.wrapping_add(i);
            match coils.get(&c_addr) {
                Some(&v) => out.push(v),
                None => return Err(ModbusClientError::CoilNotFound(c_addr)),
            }
        }
        Ok(out)
    }

    async fn write_coil(&self, addr: u16, value: bool) -> Result<(), ModbusClientError> {
        let mut coils = self
            .coils
            .lock()
            .map_err(|_| ModbusClientError::Other("lock poisoned".to_owned()))?;
        coils.insert(addr, value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_read_registers() {
        let client = MockModbusClient::new();
        client.set_register(100, vec![0x1234, 0x5678]);
        let result = client.read_holding_registers(100, 2).await.expect("ok");
        assert_eq!(result, vec![0x1234, 0x5678]);
    }

    #[tokio::test]
    async fn mock_write_registers() {
        let client = MockModbusClient::new();
        client
            .write_registers(200, &[10u16, 20u16])
            .await
            .expect("ok");
        let result = client.read_holding_registers(200, 2).await.expect("ok");
        assert_eq!(result, vec![10, 20]);
    }

    #[tokio::test]
    async fn mock_read_missing_register() {
        let client = MockModbusClient::new();
        let err = client.read_holding_registers(999, 1).await.unwrap_err();
        assert!(matches!(err, ModbusClientError::RegisterNotFound(999)));
    }

    #[tokio::test]
    async fn mock_read_coil() {
        let client = MockModbusClient::new();
        client.set_coil(5, true);
        let coils = client.read_coils(5, 1).await.expect("ok");
        assert_eq!(coils, vec![true]);
    }

    #[tokio::test]
    async fn mock_write_coil() {
        let client = MockModbusClient::new();
        client.write_coil(10, false).await.expect("ok");
        assert_eq!(client.get_coil(10), Some(false));
    }
}
