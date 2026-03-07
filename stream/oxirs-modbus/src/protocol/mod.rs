//! Modbus protocol implementations
//!
//! This module provides low-level Modbus protocol handling for TCP,
//! RTU, and ASCII variants.

pub mod ascii;
pub mod crc;
pub mod frame;
pub mod functions;
pub mod tcp;

#[cfg(feature = "rtu")]
pub mod rtu;

pub use ascii::{compute_lrc, decode_ascii, encode_ascii, AsciiCodec, AsciiFrame, AsciiTransport};
pub use crc::{append_crc, calculate_crc, verify_crc};
pub use frame::FunctionCode;
pub use functions::{
    pack_bits, unpack_bits, ReadCoilsRequest, ReadCoilsResponse, ReadDiscreteInputsRequest,
    ReadDiscreteInputsResponse, WriteMultipleCoilsRequest, WriteMultipleCoilsResponse,
    WriteMultipleRegistersRequest, WriteMultipleRegistersResponse, MAX_READ_COILS,
    MAX_READ_DISCRETE_INPUTS, MAX_WRITE_COILS, MAX_WRITE_REGISTERS,
};
pub use tcp::ModbusTcpClient;

#[cfg(feature = "rtu")]
pub use rtu::ModbusRtuClient;
