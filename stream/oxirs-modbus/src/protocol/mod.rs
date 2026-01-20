//! Modbus protocol implementations
//!
//! This module provides low-level Modbus protocol handling for both
//! TCP and RTU variants.

pub mod crc;
pub mod frame;
pub mod functions;
pub mod tcp;

#[cfg(feature = "rtu")]
pub mod rtu;

pub use crc::{append_crc, calculate_crc, verify_crc};
pub use frame::FunctionCode;
pub use tcp::ModbusTcpClient;

#[cfg(feature = "rtu")]
pub use rtu::ModbusRtuClient;
