//! Modbus frame parsing and serialization
//!
//! This module handles Modbus Application Data Unit (ADU) and
//! Protocol Data Unit (PDU) structures.

use crate::error::{ModbusError, ModbusResult};
use bytes::{BufMut, BytesMut};

/// Modbus function codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FunctionCode {
    /// Read Coils (0x01)
    ReadCoils = 0x01,
    /// Read Discrete Inputs (0x02)
    ReadDiscreteInputs = 0x02,
    /// Read Holding Registers (0x03)
    ReadHoldingRegisters = 0x03,
    /// Read Input Registers (0x04)
    ReadInputRegisters = 0x04,
    /// Write Single Coil (0x05)
    WriteSingleCoil = 0x05,
    /// Write Single Register (0x06)
    WriteSingleRegister = 0x06,
    /// Write Multiple Coils (0x0F)
    WriteMultipleCoils = 0x0F,
    /// Write Multiple Registers (0x10)
    WriteMultipleRegisters = 0x10,
}

impl FunctionCode {
    /// Create from u8
    pub fn from_u8(code: u8) -> ModbusResult<Self> {
        match code {
            0x01 => Ok(Self::ReadCoils),
            0x02 => Ok(Self::ReadDiscreteInputs),
            0x03 => Ok(Self::ReadHoldingRegisters),
            0x04 => Ok(Self::ReadInputRegisters),
            0x05 => Ok(Self::WriteSingleCoil),
            0x06 => Ok(Self::WriteSingleRegister),
            0x0F => Ok(Self::WriteMultipleCoils),
            0x10 => Ok(Self::WriteMultipleRegisters),
            _ => Err(ModbusError::InvalidFunctionCode(code)),
        }
    }

    /// Convert to u8
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Modbus TCP Application Data Unit (ADU)
#[derive(Debug, Clone)]
pub struct ModbusTcpFrame {
    /// Transaction identifier (for matching requests/responses)
    pub transaction_id: u16,

    /// Protocol identifier (always 0 for Modbus)
    pub protocol_id: u16,

    /// Unit identifier (slave address)
    pub unit_id: u8,

    /// Function code
    pub function_code: FunctionCode,

    /// Data payload
    pub data: Vec<u8>,
}

impl ModbusTcpFrame {
    /// Build a request frame for reading holding registers
    pub fn read_holding_registers(
        transaction_id: u16,
        unit_id: u8,
        start_addr: u16,
        count: u16,
    ) -> Self {
        let mut data = BytesMut::with_capacity(4);
        data.put_u16(start_addr);
        data.put_u16(count);

        Self {
            transaction_id,
            protocol_id: 0,
            unit_id,
            function_code: FunctionCode::ReadHoldingRegisters,
            data: data.to_vec(),
        }
    }

    /// Build a request frame for writing a single register
    pub fn write_single_register(transaction_id: u16, unit_id: u8, addr: u16, value: u16) -> Self {
        let mut data = BytesMut::with_capacity(4);
        data.put_u16(addr);
        data.put_u16(value);

        Self {
            transaction_id,
            protocol_id: 0,
            unit_id,
            function_code: FunctionCode::WriteSingleRegister,
            data: data.to_vec(),
        }
    }

    /// Serialize frame to bytes for transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        let length = 1 + 1 + self.data.len(); // unit_id + function_code + data
        let mut bytes = BytesMut::with_capacity(7 + self.data.len());

        // MBAP Header (7 bytes)
        bytes.put_u16(self.transaction_id);
        bytes.put_u16(self.protocol_id);
        bytes.put_u16(length as u16);
        bytes.put_u8(self.unit_id);

        // PDU
        bytes.put_u8(self.function_code.as_u8());
        bytes.put(self.data.as_slice());

        bytes.to_vec()
    }

    /// Parse response frame from bytes
    pub fn from_bytes(bytes: &[u8]) -> ModbusResult<Self> {
        if bytes.len() < 8 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Frame too short",
            )));
        }

        let transaction_id = u16::from_be_bytes([bytes[0], bytes[1]]);
        let protocol_id = u16::from_be_bytes([bytes[2], bytes[3]]);
        let _length = u16::from_be_bytes([bytes[4], bytes[5]]);
        let unit_id = bytes[6];
        let function_code_byte = bytes[7];

        // Check for exception response (high bit set)
        if function_code_byte & 0x80 != 0 {
            let original_function = function_code_byte & 0x7F;
            let exception_code = if bytes.len() > 8 { bytes[8] } else { 0 };
            return Err(ModbusError::ModbusException {
                code: exception_code,
                function: original_function,
            });
        }

        let function_code = FunctionCode::from_u8(function_code_byte)?;
        let data = bytes[8..].to_vec();

        Ok(Self {
            transaction_id,
            protocol_id,
            unit_id,
            function_code,
            data,
        })
    }

    /// Extract register values from Read Holding Registers response
    pub fn extract_registers(&self) -> ModbusResult<Vec<u16>> {
        if self.data.is_empty() {
            return Ok(Vec::new());
        }

        let byte_count = self.data[0] as usize;
        if self.data.len() < 1 + byte_count {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Incomplete register data",
            )));
        }

        let mut registers = Vec::with_capacity(byte_count / 2);
        let mut pos = 1;

        while pos + 1 < self.data.len() && pos < 1 + byte_count {
            let value = u16::from_be_bytes([self.data[pos], self.data[pos + 1]]);
            registers.push(value);
            pos += 2;
        }

        Ok(registers)
    }
}

/// Modbus frame abstraction (TCP or RTU)
#[derive(Debug, Clone)]
pub enum ModbusFrame {
    /// Modbus TCP frame
    Tcp(ModbusTcpFrame),
    /// Modbus RTU frame (to be implemented)
    #[cfg(feature = "rtu")]
    Rtu(ModbusRtuFrame),
}

/// Modbus RTU frame structure
///
/// Contains the PDU (Protocol Data Unit) plus unit identifier and CRC-16.
#[cfg(feature = "rtu")]
#[derive(Debug, Clone)]
pub struct ModbusRtuFrame {
    /// Unit identifier (slave address, 1-247)
    pub unit_id: u8,
    /// Modbus function code
    pub function_code: FunctionCode,
    /// Data payload (address, count, values)
    pub data: Vec<u8>,
    /// CRC-16 checksum (Modbus polynomial)
    pub crc: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_code_conversion() {
        assert_eq!(
            FunctionCode::from_u8(0x03).unwrap(),
            FunctionCode::ReadHoldingRegisters
        );
        assert_eq!(FunctionCode::ReadHoldingRegisters.as_u8(), 0x03);
    }

    #[test]
    fn test_invalid_function_code() {
        assert!(FunctionCode::from_u8(0xFF).is_err());
    }

    #[test]
    fn test_build_read_holding_registers() {
        let frame = ModbusTcpFrame::read_holding_registers(1, 1, 0, 10);
        assert_eq!(frame.transaction_id, 1);
        assert_eq!(frame.unit_id, 1);
        assert_eq!(frame.function_code, FunctionCode::ReadHoldingRegisters);
        assert_eq!(frame.data.len(), 4); // start_addr (2) + count (2)
    }

    #[test]
    fn test_serialize_frame() {
        let frame = ModbusTcpFrame::read_holding_registers(1, 1, 0, 10);
        let bytes = frame.to_bytes();

        // MBAP header: tid(2) + pid(2) + length(2) + unit(1) = 7 bytes
        // PDU: function(1) + data(4) = 5 bytes
        assert_eq!(bytes.len(), 12);

        // Verify transaction ID
        assert_eq!(u16::from_be_bytes([bytes[0], bytes[1]]), 1);

        // Verify protocol ID (always 0)
        assert_eq!(u16::from_be_bytes([bytes[2], bytes[3]]), 0);

        // Verify function code
        assert_eq!(bytes[7], 0x03);
    }

    #[test]
    fn test_parse_response() {
        // Simulate response: 5 registers with values [100, 200, 300, 400, 500]
        let response = vec![
            0x00, 0x01, // Transaction ID
            0x00, 0x00, // Protocol ID
            0x00, 0x0D, // Length (13 bytes)
            0x01, // Unit ID
            0x03, // Function code
            0x0A, // Byte count (10 bytes = 5 registers)
            0x00, 0x64, // Register 1: 100
            0x00, 0xC8, // Register 2: 200
            0x01, 0x2C, // Register 3: 300
            0x01, 0x90, // Register 4: 400
            0x01, 0xF4, // Register 5: 500
        ];

        let frame = ModbusTcpFrame::from_bytes(&response).unwrap();
        assert_eq!(frame.transaction_id, 1);
        assert_eq!(frame.function_code, FunctionCode::ReadHoldingRegisters);

        let registers = frame.extract_registers().unwrap();
        assert_eq!(registers, vec![100, 200, 300, 400, 500]);
    }

    #[test]
    fn test_exception_response() {
        // Exception response: function code | 0x80, exception code
        let response = vec![
            0x00, 0x01, // Transaction ID
            0x00, 0x00, // Protocol ID
            0x00, 0x03, // Length
            0x01, // Unit ID
            0x83, // Function code (0x03 | 0x80 = exception)
            0x02, // Exception code (Illegal Data Address)
        ];

        let result = ModbusTcpFrame::from_bytes(&response);
        assert!(result.is_err());

        if let Err(ModbusError::ModbusException { code, function }) = result {
            assert_eq!(code, 0x02);
            assert_eq!(function, 0x03);
        } else {
            panic!("Expected ModbusException error");
        }
    }
}
