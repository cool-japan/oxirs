//! Advanced Modbus data codec
//!
//! Provides byte-order-aware encoding and decoding of Modbus register data
//! into strongly-typed Rust values. Supports all four byte-order modes found
//! in real industrial devices.

pub mod data_decoder;

pub use data_decoder::{DecoderDataType, ModbusDecoder, ModbusEncoder, ModbusTypedValue};
