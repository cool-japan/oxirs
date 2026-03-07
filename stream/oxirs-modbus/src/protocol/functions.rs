//! Modbus function code PDU implementations
//!
//! Provides request/response structs with encode/decode for:
//! - FC 0x01 — Read Coils
//! - FC 0x02 — Read Discrete Inputs
//! - FC 0x0F — Write Multiple Coils
//! - FC 0x10 — Write Multiple Registers
//!
//! Quantity limits follow Modbus Application Protocol V1.1b3:
//! - Read Coils / Discrete Inputs: 1–2000 bits
//! - Write Multiple Coils: 1–1968 bits
//! - Write Multiple Registers: 1–123 registers

use crate::error::{ModbusError, ModbusResult};

// ── constants ──────────────────────────────────────────────────────────────
/// Maximum number of coils that may be read in one request.
pub const MAX_READ_COILS: u16 = 2000;
/// Maximum number of discrete inputs that may be read in one request.
pub const MAX_READ_DISCRETE_INPUTS: u16 = 2000;
/// Maximum number of coils that may be written in one request.
pub const MAX_WRITE_COILS: u16 = 1968;
/// Maximum number of registers that may be written in one request.
pub const MAX_WRITE_REGISTERS: u16 = 123;

// ── helpers ────────────────────────────────────────────────────────────────

/// Pack a slice of booleans into a byte vector (LSB first, padded to byte boundary).
pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let byte_count = (bits.len() + 7) / 8;
    let mut bytes = vec![0u8; byte_count];
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    bytes
}

/// Unpack `count` booleans from a packed byte slice (LSB first).
pub fn unpack_bits(bytes: &[u8], count: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let value = if byte_idx < bytes.len() {
            (bytes[byte_idx] >> bit_idx) & 1 == 1
        } else {
            false
        };
        bits.push(value);
    }
    bits
}

// ── FC 0x01 — Read Coils ──────────────────────────────────────────────────

/// PDU request for FC 0x01 Read Coils.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadCoilsRequest {
    /// Starting address of the first coil.
    pub start_address: u16,
    /// Number of coils to read (1–2000).
    pub quantity: u16,
}

impl ReadCoilsRequest {
    /// Create a new request, validating quantity.
    pub fn new(start_address: u16, quantity: u16) -> ModbusResult<Self> {
        if quantity == 0 || quantity > MAX_READ_COILS {
            return Err(ModbusError::InvalidCount(quantity));
        }
        Ok(Self {
            start_address,
            quantity,
        })
    }

    /// Encode to PDU bytes (function code + 4 data bytes).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(5);
        buf.push(0x01);
        buf.extend_from_slice(&self.start_address.to_be_bytes());
        buf.extend_from_slice(&self.quantity.to_be_bytes());
        buf
    }

    /// Decode from the PDU data bytes (after the function code has been stripped).
    pub fn decode(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 4 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "ReadCoilsRequest: expected 4 data bytes",
            )));
        }
        let start_address = u16::from_be_bytes([data[0], data[1]]);
        let quantity = u16::from_be_bytes([data[2], data[3]]);
        Self::new(start_address, quantity)
    }
}

/// PDU response for FC 0x01 Read Coils.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadCoilsResponse {
    /// Status of each coil (in request order).
    pub coil_status: Vec<bool>,
}

impl ReadCoilsResponse {
    /// Create a response from a vector of boolean coil values.
    pub fn new(coil_status: Vec<bool>) -> Self {
        Self { coil_status }
    }

    /// Encode to PDU bytes (function code + byte count + packed bits).
    pub fn encode(&self) -> Vec<u8> {
        let packed = pack_bits(&self.coil_status);
        let mut buf = Vec::with_capacity(2 + packed.len());
        buf.push(0x01);
        buf.push(packed.len() as u8);
        buf.extend_from_slice(&packed);
        buf
    }

    /// Decode from PDU data bytes (after function code stripped), given the expected coil count.
    pub fn decode(data: &[u8], quantity: usize) -> ModbusResult<Self> {
        if data.is_empty() {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "ReadCoilsResponse: missing byte count",
            )));
        }
        let byte_count = data[0] as usize;
        if data.len() < 1 + byte_count {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "ReadCoilsResponse: incomplete packed bytes",
            )));
        }
        let packed = &data[1..1 + byte_count];
        let coil_status = unpack_bits(packed, quantity);
        Ok(Self { coil_status })
    }
}

// ── FC 0x02 — Read Discrete Inputs ────────────────────────────────────────

/// PDU request for FC 0x02 Read Discrete Inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadDiscreteInputsRequest {
    /// Starting address of the first discrete input.
    pub start_address: u16,
    /// Number of discrete inputs to read (1–2000).
    pub quantity: u16,
}

impl ReadDiscreteInputsRequest {
    /// Create a new request, validating quantity.
    pub fn new(start_address: u16, quantity: u16) -> ModbusResult<Self> {
        if quantity == 0 || quantity > MAX_READ_DISCRETE_INPUTS {
            return Err(ModbusError::InvalidCount(quantity));
        }
        Ok(Self {
            start_address,
            quantity,
        })
    }

    /// Encode to PDU bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(5);
        buf.push(0x02);
        buf.extend_from_slice(&self.start_address.to_be_bytes());
        buf.extend_from_slice(&self.quantity.to_be_bytes());
        buf
    }

    /// Decode from PDU data bytes (after function code stripped).
    pub fn decode(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 4 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "ReadDiscreteInputsRequest: expected 4 data bytes",
            )));
        }
        let start_address = u16::from_be_bytes([data[0], data[1]]);
        let quantity = u16::from_be_bytes([data[2], data[3]]);
        Self::new(start_address, quantity)
    }
}

/// PDU response for FC 0x02 Read Discrete Inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadDiscreteInputsResponse {
    /// Status of each discrete input (in request order).
    pub input_status: Vec<bool>,
}

impl ReadDiscreteInputsResponse {
    /// Create a response from a vector of boolean input values.
    pub fn new(input_status: Vec<bool>) -> Self {
        Self { input_status }
    }

    /// Encode to PDU bytes.
    pub fn encode(&self) -> Vec<u8> {
        let packed = pack_bits(&self.input_status);
        let mut buf = Vec::with_capacity(2 + packed.len());
        buf.push(0x02);
        buf.push(packed.len() as u8);
        buf.extend_from_slice(&packed);
        buf
    }

    /// Decode from PDU data bytes (after function code stripped), given expected input count.
    pub fn decode(data: &[u8], quantity: usize) -> ModbusResult<Self> {
        if data.is_empty() {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "ReadDiscreteInputsResponse: missing byte count",
            )));
        }
        let byte_count = data[0] as usize;
        if data.len() < 1 + byte_count {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "ReadDiscreteInputsResponse: incomplete packed bytes",
            )));
        }
        let packed = &data[1..1 + byte_count];
        let input_status = unpack_bits(packed, quantity);
        Ok(Self { input_status })
    }
}

// ── FC 0x0F — Write Multiple Coils ────────────────────────────────────────

/// PDU request for FC 0x0F Write Multiple Coils.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteMultipleCoilsRequest {
    /// Starting address of the first coil to write.
    pub start_address: u16,
    /// Output values (1–1968).
    pub outputs: Vec<bool>,
}

impl WriteMultipleCoilsRequest {
    /// Create a new request, validating quantity.
    pub fn new(start_address: u16, outputs: Vec<bool>) -> ModbusResult<Self> {
        let qty = outputs.len() as u16;
        if qty == 0 || qty > MAX_WRITE_COILS {
            return Err(ModbusError::InvalidCount(qty));
        }
        Ok(Self {
            start_address,
            outputs,
        })
    }

    /// Number of coils.
    pub fn quantity(&self) -> u16 {
        self.outputs.len() as u16
    }

    /// Encode to PDU bytes.
    pub fn encode(&self) -> Vec<u8> {
        let packed = pack_bits(&self.outputs);
        let qty = self.outputs.len() as u16;
        let mut buf = Vec::with_capacity(6 + packed.len());
        buf.push(0x0F);
        buf.extend_from_slice(&self.start_address.to_be_bytes());
        buf.extend_from_slice(&qty.to_be_bytes());
        buf.push(packed.len() as u8);
        buf.extend_from_slice(&packed);
        buf
    }

    /// Decode from PDU data bytes (after function code stripped).
    pub fn decode(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 5 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "WriteMultipleCoilsRequest: too short",
            )));
        }
        let start_address = u16::from_be_bytes([data[0], data[1]]);
        let quantity = u16::from_be_bytes([data[2], data[3]]);
        let byte_count = data[4] as usize;
        if data.len() < 5 + byte_count {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "WriteMultipleCoilsRequest: incomplete packed bytes",
            )));
        }
        let packed = &data[5..5 + byte_count];
        let outputs = unpack_bits(packed, quantity as usize);
        Self::new(start_address, outputs)
    }
}

/// PDU response for FC 0x0F Write Multiple Coils.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteMultipleCoilsResponse {
    /// Starting address echoed from request.
    pub start_address: u16,
    /// Number of coils written.
    pub quantity: u16,
}

impl WriteMultipleCoilsResponse {
    /// Create a new response.
    pub fn new(start_address: u16, quantity: u16) -> Self {
        Self {
            start_address,
            quantity,
        }
    }

    /// Encode to PDU bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(5);
        buf.push(0x0F);
        buf.extend_from_slice(&self.start_address.to_be_bytes());
        buf.extend_from_slice(&self.quantity.to_be_bytes());
        buf
    }

    /// Decode from PDU data bytes (after function code stripped).
    pub fn decode(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 4 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "WriteMultipleCoilsResponse: expected 4 data bytes",
            )));
        }
        let start_address = u16::from_be_bytes([data[0], data[1]]);
        let quantity = u16::from_be_bytes([data[2], data[3]]);
        Ok(Self {
            start_address,
            quantity,
        })
    }
}

// ── FC 0x10 — Write Multiple Registers ───────────────────────────────────

/// PDU request for FC 0x10 Write Multiple Registers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteMultipleRegistersRequest {
    /// Starting address of the first register to write.
    pub start_address: u16,
    /// Register values to write (1–123).
    pub values: Vec<u16>,
}

impl WriteMultipleRegistersRequest {
    /// Create a new request, validating quantity.
    pub fn new(start_address: u16, values: Vec<u16>) -> ModbusResult<Self> {
        let qty = values.len() as u16;
        if qty == 0 || qty > MAX_WRITE_REGISTERS {
            return Err(ModbusError::InvalidCount(qty));
        }
        Ok(Self {
            start_address,
            values,
        })
    }

    /// Number of registers.
    pub fn quantity(&self) -> u16 {
        self.values.len() as u16
    }

    /// Encode to PDU bytes.
    pub fn encode(&self) -> Vec<u8> {
        let qty = self.values.len() as u16;
        let byte_count = (qty * 2) as u8;
        let mut buf = Vec::with_capacity(6 + self.values.len() * 2);
        buf.push(0x10);
        buf.extend_from_slice(&self.start_address.to_be_bytes());
        buf.extend_from_slice(&qty.to_be_bytes());
        buf.push(byte_count);
        for &v in &self.values {
            buf.extend_from_slice(&v.to_be_bytes());
        }
        buf
    }

    /// Decode from PDU data bytes (after function code stripped).
    pub fn decode(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 5 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "WriteMultipleRegistersRequest: too short",
            )));
        }
        let start_address = u16::from_be_bytes([data[0], data[1]]);
        let quantity = u16::from_be_bytes([data[2], data[3]]) as usize;
        let byte_count = data[4] as usize;
        if data.len() < 5 + byte_count {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "WriteMultipleRegistersRequest: incomplete register data",
            )));
        }
        let mut values = Vec::with_capacity(quantity);
        for i in 0..quantity {
            let offset = 5 + i * 2;
            if offset + 1 >= data.len() {
                break;
            }
            values.push(u16::from_be_bytes([data[offset], data[offset + 1]]));
        }
        Self::new(start_address, values)
    }
}

/// PDU response for FC 0x10 Write Multiple Registers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteMultipleRegistersResponse {
    /// Starting address echoed from request.
    pub start_address: u16,
    /// Number of registers written.
    pub quantity: u16,
}

impl WriteMultipleRegistersResponse {
    /// Create a new response.
    pub fn new(start_address: u16, quantity: u16) -> Self {
        Self {
            start_address,
            quantity,
        }
    }

    /// Encode to PDU bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(5);
        buf.push(0x10);
        buf.extend_from_slice(&self.start_address.to_be_bytes());
        buf.extend_from_slice(&self.quantity.to_be_bytes());
        buf
    }

    /// Decode from PDU data bytes (after function code stripped).
    pub fn decode(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 4 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "WriteMultipleRegistersResponse: expected 4 data bytes",
            )));
        }
        let start_address = u16::from_be_bytes([data[0], data[1]]);
        let quantity = u16::from_be_bytes([data[2], data[3]]);
        Ok(Self {
            start_address,
            quantity,
        })
    }
}

// ── tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── bit packing helpers ──────────────────────────────────────────────

    #[test]
    fn test_pack_bits_empty() {
        assert!(pack_bits(&[]).is_empty());
    }

    #[test]
    fn test_pack_bits_single_byte() {
        // [T, F, T, F, T, F, T, F] → 0b01010101 = 0x55
        let bits = [true, false, true, false, true, false, true, false];
        assert_eq!(pack_bits(&bits), vec![0x55]);
    }

    #[test]
    fn test_pack_bits_partial_byte() {
        // [T, T, F] → 0b00000011 = 0x03
        let bits = [true, true, false];
        assert_eq!(pack_bits(&bits), vec![0x03]);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let original: Vec<bool> = (0..13).map(|i| i % 3 == 0).collect();
        let packed = pack_bits(&original);
        let unpacked = unpack_bits(&packed, original.len());
        assert_eq!(unpacked, original);
    }

    // ── FC 0x01 Read Coils ───────────────────────────────────────────────

    #[test]
    fn test_read_coils_request_new_valid() {
        let req = ReadCoilsRequest::new(100, 10).expect("valid");
        assert_eq!(req.start_address, 100);
        assert_eq!(req.quantity, 10);
    }

    #[test]
    fn test_read_coils_request_zero_quantity_rejected() {
        assert!(ReadCoilsRequest::new(0, 0).is_err());
    }

    #[test]
    fn test_read_coils_request_max_quantity() {
        assert!(ReadCoilsRequest::new(0, 2000).is_ok());
    }

    #[test]
    fn test_read_coils_request_over_max_rejected() {
        assert!(ReadCoilsRequest::new(0, 2001).is_err());
    }

    #[test]
    fn test_read_coils_request_encode_decode() {
        let req = ReadCoilsRequest::new(200, 16).expect("valid");
        let encoded = req.encode();
        // function code (1) + start_address (2) + quantity (2) = 5 bytes
        assert_eq!(encoded.len(), 5);
        assert_eq!(encoded[0], 0x01);
        let decoded = ReadCoilsRequest::decode(&encoded[1..]).expect("decoded");
        assert_eq!(decoded, req);
    }

    #[test]
    fn test_read_coils_response_encode_decode() {
        let status: Vec<bool> = vec![true, false, true, true, false, false, true, false, true];
        let resp = ReadCoilsResponse::new(status.clone());
        let encoded = resp.encode();
        // function code (1) + byte_count (1) + 2 packed bytes = 4
        assert_eq!(encoded[0], 0x01);
        let decoded = ReadCoilsResponse::decode(&encoded[1..], status.len()).expect("decoded");
        assert_eq!(decoded.coil_status, status);
    }

    #[test]
    fn test_read_coils_response_aligned_bits() {
        // Exactly 8 coils → 1 byte
        let status = vec![true; 8];
        let resp = ReadCoilsResponse::new(status.clone());
        let encoded = resp.encode();
        // function_code(1) + byte_count(1) + data(1) = 3
        assert_eq!(encoded.len(), 3);
        let decoded = ReadCoilsResponse::decode(&encoded[1..], 8).expect("ok");
        assert_eq!(decoded.coil_status, status);
    }

    // ── FC 0x02 Read Discrete Inputs ────────────────────────────────────

    #[test]
    fn test_read_discrete_inputs_request_valid() {
        let req = ReadDiscreteInputsRequest::new(0, 1).expect("valid");
        assert_eq!(req.quantity, 1);
    }

    #[test]
    fn test_read_discrete_inputs_request_zero_rejected() {
        assert!(ReadDiscreteInputsRequest::new(0, 0).is_err());
    }

    #[test]
    fn test_read_discrete_inputs_request_max() {
        assert!(ReadDiscreteInputsRequest::new(0, 2000).is_ok());
    }

    #[test]
    fn test_read_discrete_inputs_request_over_max() {
        assert!(ReadDiscreteInputsRequest::new(0, 2001).is_err());
    }

    #[test]
    fn test_read_discrete_inputs_encode_decode() {
        let req = ReadDiscreteInputsRequest::new(50, 5).expect("valid");
        let encoded = req.encode();
        assert_eq!(encoded[0], 0x02);
        let decoded = ReadDiscreteInputsRequest::decode(&encoded[1..]).expect("decoded");
        assert_eq!(decoded, req);
    }

    #[test]
    fn test_read_discrete_inputs_response_encode_decode() {
        let status = vec![false, true, false, true, true];
        let resp = ReadDiscreteInputsResponse::new(status.clone());
        let encoded = resp.encode();
        assert_eq!(encoded[0], 0x02);
        let decoded = ReadDiscreteInputsResponse::decode(&encoded[1..], status.len()).expect("ok");
        assert_eq!(decoded.input_status, status);
    }

    // ── FC 0x0F Write Multiple Coils ────────────────────────────────────

    #[test]
    fn test_write_multiple_coils_request_valid() {
        let outputs = vec![true, false, true];
        let req = WriteMultipleCoilsRequest::new(10, outputs.clone()).expect("valid");
        assert_eq!(req.start_address, 10);
        assert_eq!(req.quantity(), 3);
    }

    #[test]
    fn test_write_multiple_coils_request_zero_rejected() {
        assert!(WriteMultipleCoilsRequest::new(0, vec![]).is_err());
    }

    #[test]
    fn test_write_multiple_coils_request_over_max_rejected() {
        let too_many = vec![false; 1969];
        assert!(WriteMultipleCoilsRequest::new(0, too_many).is_err());
    }

    #[test]
    fn test_write_multiple_coils_encode_decode() {
        let outputs: Vec<bool> = (0..10).map(|i| i % 2 == 0).collect();
        let req = WriteMultipleCoilsRequest::new(0, outputs.clone()).expect("valid");
        let encoded = req.encode();
        assert_eq!(encoded[0], 0x0F);
        let decoded = WriteMultipleCoilsRequest::decode(&encoded[1..]).expect("decoded");
        assert_eq!(decoded.outputs, outputs);
    }

    #[test]
    fn test_write_multiple_coils_response_encode_decode() {
        let resp = WriteMultipleCoilsResponse::new(20, 15);
        let encoded = resp.encode();
        assert_eq!(encoded[0], 0x0F);
        let decoded = WriteMultipleCoilsResponse::decode(&encoded[1..]).expect("decoded");
        assert_eq!(decoded, resp);
    }

    #[test]
    fn test_write_multiple_coils_bit_packing() {
        // All-true outputs produce 0xFF bytes
        let outputs = vec![true; 8];
        let req = WriteMultipleCoilsRequest::new(0, outputs).expect("valid");
        let encoded = req.encode();
        // function(1) + start(2) + qty(2) + byte_count(1) + data(1) = 7
        assert_eq!(encoded.len(), 7);
        // Packed byte: all bits set
        assert_eq!(encoded[6], 0xFF);
    }

    // ── FC 0x10 Write Multiple Registers ─────────────────────────────────

    #[test]
    fn test_write_multiple_registers_request_valid() {
        let values = vec![100u16, 200, 300];
        let req = WriteMultipleRegistersRequest::new(40001, values.clone()).expect("valid");
        assert_eq!(req.start_address, 40001);
        assert_eq!(req.quantity(), 3);
    }

    #[test]
    fn test_write_multiple_registers_request_zero_rejected() {
        assert!(WriteMultipleRegistersRequest::new(0, vec![]).is_err());
    }

    #[test]
    fn test_write_multiple_registers_request_over_max_rejected() {
        let too_many = vec![0u16; 124];
        assert!(WriteMultipleRegistersRequest::new(0, too_many).is_err());
    }

    #[test]
    fn test_write_multiple_registers_encode_decode() {
        let values: Vec<u16> = (1..=5).collect();
        let req = WriteMultipleRegistersRequest::new(100, values.clone()).expect("valid");
        let encoded = req.encode();
        assert_eq!(encoded[0], 0x10);
        let decoded = WriteMultipleRegistersRequest::decode(&encoded[1..]).expect("decoded");
        assert_eq!(decoded.values, values);
    }

    #[test]
    fn test_write_multiple_registers_response_encode_decode() {
        let resp = WriteMultipleRegistersResponse::new(100, 5);
        let encoded = resp.encode();
        assert_eq!(encoded[0], 0x10);
        let decoded = WriteMultipleRegistersResponse::decode(&encoded[1..]).expect("decoded");
        assert_eq!(decoded, resp);
    }

    #[test]
    fn test_write_multiple_registers_encoding_length() {
        // 3 registers: func(1) + start(2) + qty(2) + byte_count(1) + data(6) = 12
        let req = WriteMultipleRegistersRequest::new(0, vec![1, 2, 3]).expect("valid");
        assert_eq!(req.encode().len(), 12);
    }

    #[test]
    fn test_write_multiple_registers_max_count() {
        let values = vec![0xFFFFu16; 123];
        assert!(WriteMultipleRegistersRequest::new(0, values).is_ok());
    }
}
