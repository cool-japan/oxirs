//! Modbus data type conversions
//!
//! Handles conversion from raw Modbus register values to typed values
//! and RDF literals with proper XSD datatypes.

use crate::error::{ModbusError, ModbusResult};
use std::fmt;
use std::str::FromStr;

/// Modbus data types for register interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModbusDataType {
    /// Signed 16-bit integer (single register)
    Int16,
    /// Unsigned 16-bit integer (single register)
    Uint16,
    /// Signed 32-bit integer (two registers, big-endian)
    Int32,
    /// Unsigned 32-bit integer (two registers, big-endian)
    Uint32,
    /// IEEE 754 single precision float (two registers, big-endian)
    Float32,
    /// IEEE 754 double precision float (four registers, big-endian)
    Float64,
    /// Single bit extraction from register (0-15)
    Bit(u8),
    /// String (multiple registers, ASCII)
    String(usize),
}

impl ModbusDataType {
    /// Number of 16-bit registers required for this data type
    pub fn register_count(&self) -> usize {
        match self {
            ModbusDataType::Int16 | ModbusDataType::Uint16 => 1,
            ModbusDataType::Int32 | ModbusDataType::Uint32 | ModbusDataType::Float32 => 2,
            ModbusDataType::Float64 => 4,
            ModbusDataType::Bit(_) => 1,
            ModbusDataType::String(len) => (*len + 1) / 2, // 2 chars per register
        }
    }

    /// XSD datatype IRI for RDF literals
    pub fn xsd_datatype(&self) -> &'static str {
        match self {
            ModbusDataType::Int16 => "http://www.w3.org/2001/XMLSchema#short",
            ModbusDataType::Uint16 => "http://www.w3.org/2001/XMLSchema#unsignedShort",
            ModbusDataType::Int32 => "http://www.w3.org/2001/XMLSchema#int",
            ModbusDataType::Uint32 => "http://www.w3.org/2001/XMLSchema#unsignedInt",
            ModbusDataType::Float32 | ModbusDataType::Float64 => {
                "http://www.w3.org/2001/XMLSchema#float"
            }
            ModbusDataType::Bit(_) => "http://www.w3.org/2001/XMLSchema#boolean",
            ModbusDataType::String(_) => "http://www.w3.org/2001/XMLSchema#string",
        }
    }
}

impl FromStr for ModbusDataType {
    type Err = ModbusError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s_upper = s.to_uppercase();
        match s_upper.as_str() {
            "INT16" | "SHORT" => Ok(ModbusDataType::Int16),
            "UINT16" | "WORD" | "USHORT" => Ok(ModbusDataType::Uint16),
            "INT32" | "DINT" | "INT" => Ok(ModbusDataType::Int32),
            "UINT32" | "DWORD" | "UDINT" => Ok(ModbusDataType::Uint32),
            "FLOAT32" | "REAL" | "FLOAT" => Ok(ModbusDataType::Float32),
            "FLOAT64" | "LREAL" | "DOUBLE" => Ok(ModbusDataType::Float64),
            _ if s_upper.starts_with("BIT") => {
                // Parse "BIT0", "BIT15", etc.
                let bit_num: u8 = s_upper[3..].parse().map_err(|_| {
                    ModbusError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Invalid bit number in {}", s),
                    ))
                })?;
                if bit_num > 15 {
                    return Err(ModbusError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Bit number must be 0-15",
                    )));
                }
                Ok(ModbusDataType::Bit(bit_num))
            }
            _ if s_upper.starts_with("STRING") => {
                // Parse "STRING10", "STRING32", etc.
                let len: usize = s_upper[6..].parse().map_err(|_| {
                    ModbusError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Invalid string length in {}", s),
                    ))
                })?;
                Ok(ModbusDataType::String(len))
            }
            _ => Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unknown data type: {}", s),
            ))),
        }
    }
}

impl fmt::Display for ModbusDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModbusDataType::Int16 => write!(f, "INT16"),
            ModbusDataType::Uint16 => write!(f, "UINT16"),
            ModbusDataType::Int32 => write!(f, "INT32"),
            ModbusDataType::Uint32 => write!(f, "UINT32"),
            ModbusDataType::Float32 => write!(f, "FLOAT32"),
            ModbusDataType::Float64 => write!(f, "FLOAT64"),
            ModbusDataType::Bit(n) => write!(f, "BIT{}", n),
            ModbusDataType::String(len) => write!(f, "STRING{}", len),
        }
    }
}

/// Decoded value from Modbus registers
#[derive(Debug, Clone, PartialEq)]
pub enum ModbusValue {
    /// Signed integer value
    Int(i64),
    /// Unsigned integer value
    Uint(u64),
    /// Floating point value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// String value
    String(String),
}

impl ModbusValue {
    /// Convert to f64 (for scaling)
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ModbusValue::Int(v) => Some(*v as f64),
            ModbusValue::Uint(v) => Some(*v as f64),
            ModbusValue::Float(v) => Some(*v),
            ModbusValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            ModbusValue::String(_) => None,
        }
    }

    /// Convert to RDF literal string representation
    pub fn to_rdf_literal(&self) -> String {
        match self {
            ModbusValue::Int(v) => v.to_string(),
            ModbusValue::Uint(v) => v.to_string(),
            ModbusValue::Float(v) => {
                // Format float with appropriate precision
                if v.fract() == 0.0 {
                    format!("{:.1}", v)
                } else {
                    format!("{}", v)
                }
            }
            ModbusValue::Bool(v) => if *v { "true" } else { "false" }.to_string(),
            ModbusValue::String(s) => s.clone(),
        }
    }
}

impl fmt::Display for ModbusValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModbusValue::Int(v) => write!(f, "{}", v),
            ModbusValue::Uint(v) => write!(f, "{}", v),
            ModbusValue::Float(v) => write!(f, "{}", v),
            ModbusValue::Bool(v) => write!(f, "{}", v),
            ModbusValue::String(s) => write!(f, "{}", s),
        }
    }
}

/// Linear scaling parameters (physical = raw * multiplier + offset)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearScaling {
    /// Multiplier (default: 1.0)
    pub multiplier: f64,
    /// Offset (default: 0.0)
    pub offset: f64,
}

impl Default for LinearScaling {
    fn default() -> Self {
        Self {
            multiplier: 1.0,
            offset: 0.0,
        }
    }
}

impl LinearScaling {
    /// Create new scaling with multiplier and offset
    pub fn new(multiplier: f64, offset: f64) -> Self {
        Self { multiplier, offset }
    }

    /// Apply scaling: physical = raw * multiplier + offset
    pub fn apply(&self, raw: f64) -> f64 {
        raw * self.multiplier + self.offset
    }

    /// Reverse scaling: raw = (physical - offset) / multiplier
    pub fn reverse(&self, physical: f64) -> f64 {
        (physical - self.offset) / self.multiplier
    }

    /// Check if scaling is identity (no change)
    pub fn is_identity(&self) -> bool {
        (self.multiplier - 1.0).abs() < f64::EPSILON && self.offset.abs() < f64::EPSILON
    }
}

/// Decode raw register values to typed values
pub fn decode_registers(registers: &[u16], data_type: ModbusDataType) -> ModbusResult<ModbusValue> {
    let required = data_type.register_count();
    if registers.len() < required {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "Need {} registers for {:?}, got {}",
                required,
                data_type,
                registers.len()
            ),
        )));
    }

    match data_type {
        ModbusDataType::Int16 => {
            let value = registers[0] as i16;
            Ok(ModbusValue::Int(value as i64))
        }
        ModbusDataType::Uint16 => {
            let value = registers[0];
            Ok(ModbusValue::Uint(value as u64))
        }
        ModbusDataType::Int32 => {
            // Big-endian: high word first
            let value = ((registers[0] as i32) << 16) | (registers[1] as i32);
            Ok(ModbusValue::Int(value as i64))
        }
        ModbusDataType::Uint32 => {
            // Big-endian: high word first
            let value = ((registers[0] as u32) << 16) | (registers[1] as u32);
            Ok(ModbusValue::Uint(value as u64))
        }
        ModbusDataType::Float32 => {
            // Big-endian: high word first
            let bits = ((registers[0] as u32) << 16) | (registers[1] as u32);
            let value = f32::from_bits(bits);
            Ok(ModbusValue::Float(value as f64))
        }
        ModbusDataType::Float64 => {
            // Big-endian: high word first
            let bits = ((registers[0] as u64) << 48)
                | ((registers[1] as u64) << 32)
                | ((registers[2] as u64) << 16)
                | (registers[3] as u64);
            let value = f64::from_bits(bits);
            Ok(ModbusValue::Float(value))
        }
        ModbusDataType::Bit(bit_num) => {
            let value = (registers[0] >> bit_num) & 1 == 1;
            Ok(ModbusValue::Bool(value))
        }
        ModbusDataType::String(len) => {
            let mut chars = Vec::with_capacity(len);
            for reg in registers.iter().take((len + 1) / 2) {
                chars.push((reg >> 8) as u8);
                chars.push((reg & 0xFF) as u8);
            }
            chars.truncate(len);
            // Remove null terminator if present
            while chars.last() == Some(&0) {
                chars.pop();
            }
            let s = String::from_utf8_lossy(&chars).to_string();
            Ok(ModbusValue::String(s))
        }
    }
}

/// Encode typed values to raw register values
pub fn encode_value(value: &ModbusValue, data_type: ModbusDataType) -> ModbusResult<Vec<u16>> {
    match (value, data_type) {
        (ModbusValue::Int(v), ModbusDataType::Int16) => {
            let val = *v as i16;
            Ok(vec![val as u16])
        }
        (ModbusValue::Uint(v), ModbusDataType::Uint16) => {
            let val = *v as u16;
            Ok(vec![val])
        }
        (ModbusValue::Int(v), ModbusDataType::Uint16) => {
            let val = *v as u16;
            Ok(vec![val])
        }
        (ModbusValue::Int(v), ModbusDataType::Int32) => {
            let val = *v as i32;
            Ok(vec![(val >> 16) as u16, (val & 0xFFFF) as u16])
        }
        (ModbusValue::Uint(v), ModbusDataType::Uint32) => {
            let val = *v as u32;
            Ok(vec![(val >> 16) as u16, (val & 0xFFFF) as u16])
        }
        (ModbusValue::Int(v), ModbusDataType::Uint32) => {
            let val = *v as u32;
            Ok(vec![(val >> 16) as u16, (val & 0xFFFF) as u16])
        }
        (ModbusValue::Float(v), ModbusDataType::Float32) => {
            let bits = (*v as f32).to_bits();
            Ok(vec![(bits >> 16) as u16, (bits & 0xFFFF) as u16])
        }
        (ModbusValue::Float(v), ModbusDataType::Float64) => {
            let bits = v.to_bits();
            Ok(vec![
                (bits >> 48) as u16,
                ((bits >> 32) & 0xFFFF) as u16,
                ((bits >> 16) & 0xFFFF) as u16,
                (bits & 0xFFFF) as u16,
            ])
        }
        (ModbusValue::Bool(v), ModbusDataType::Bit(bit_num)) => {
            // Note: This only sets/clears the specified bit
            // In practice, you'd need to read-modify-write
            let val = if *v { 1u16 << bit_num } else { 0 };
            Ok(vec![val])
        }
        (ModbusValue::String(s), ModbusDataType::String(len)) => {
            let mut registers = Vec::with_capacity((len + 1) / 2);
            let bytes = s.as_bytes();
            for i in (0..len).step_by(2) {
                let high = bytes.get(i).copied().unwrap_or(0);
                let low = bytes.get(i + 1).copied().unwrap_or(0);
                registers.push(((high as u16) << 8) | (low as u16));
            }
            Ok(registers)
        }
        _ => Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Cannot encode {:?} as {:?}", value, data_type),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_from_str() {
        assert_eq!(
            "INT16".parse::<ModbusDataType>().unwrap(),
            ModbusDataType::Int16
        );
        assert_eq!(
            "UINT16".parse::<ModbusDataType>().unwrap(),
            ModbusDataType::Uint16
        );
        assert_eq!(
            "FLOAT32".parse::<ModbusDataType>().unwrap(),
            ModbusDataType::Float32
        );
        assert_eq!(
            "BIT5".parse::<ModbusDataType>().unwrap(),
            ModbusDataType::Bit(5)
        );
        assert_eq!(
            "STRING10".parse::<ModbusDataType>().unwrap(),
            ModbusDataType::String(10)
        );
    }

    #[test]
    fn test_decode_int16() {
        // Positive value
        let regs = [0x00FF];
        let val = decode_registers(&regs, ModbusDataType::Int16).unwrap();
        assert_eq!(val, ModbusValue::Int(255));

        // Negative value (two's complement)
        let regs = [0xFFFF];
        let val = decode_registers(&regs, ModbusDataType::Int16).unwrap();
        assert_eq!(val, ModbusValue::Int(-1));
    }

    #[test]
    fn test_decode_uint16() {
        let regs = [0xFFFF];
        let val = decode_registers(&regs, ModbusDataType::Uint16).unwrap();
        assert_eq!(val, ModbusValue::Uint(65535));
    }

    #[test]
    fn test_decode_int32() {
        // Big-endian: 0x0001_0000 = 65536
        let regs = [0x0001, 0x0000];
        let val = decode_registers(&regs, ModbusDataType::Int32).unwrap();
        assert_eq!(val, ModbusValue::Int(65536));

        // Negative: -1
        let regs = [0xFFFF, 0xFFFF];
        let val = decode_registers(&regs, ModbusDataType::Int32).unwrap();
        assert_eq!(val, ModbusValue::Int(-1));
    }

    #[test]
    fn test_decode_float32() {
        // IEEE 754: 1.0 = 0x3F800000
        let regs = [0x3F80, 0x0000];
        let val = decode_registers(&regs, ModbusDataType::Float32).unwrap();
        match val {
            ModbusValue::Float(v) => assert!((v - 1.0).abs() < 0.0001),
            _ => panic!("Expected Float"),
        }

        // IEEE 754: 22.5 = 0x41B40000
        let regs = [0x41B4, 0x0000];
        let val = decode_registers(&regs, ModbusDataType::Float32).unwrap();
        match val {
            ModbusValue::Float(v) => assert!((v - 22.5).abs() < 0.0001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_decode_bit() {
        let regs = [0b0000_0000_0010_0100]; // Bits 2 and 5 are set

        let val = decode_registers(&regs, ModbusDataType::Bit(2)).unwrap();
        assert_eq!(val, ModbusValue::Bool(true));

        let val = decode_registers(&regs, ModbusDataType::Bit(5)).unwrap();
        assert_eq!(val, ModbusValue::Bool(true));

        let val = decode_registers(&regs, ModbusDataType::Bit(0)).unwrap();
        assert_eq!(val, ModbusValue::Bool(false));
    }

    #[test]
    fn test_decode_string() {
        // "AB" = 0x4142
        let regs = [0x4142, 0x4344];
        let val = decode_registers(&regs, ModbusDataType::String(4)).unwrap();
        assert_eq!(val, ModbusValue::String("ABCD".to_string()));
    }

    #[test]
    fn test_linear_scaling() {
        let scaling = LinearScaling::new(0.1, -40.0);

        // Temperature sensor: raw 625 = 22.5°C
        let raw = 625.0;
        let physical = scaling.apply(raw);
        assert!((physical - 22.5).abs() < 0.01);

        // Reverse scaling
        let back = scaling.reverse(physical);
        assert!((back - raw).abs() < 0.01);
    }

    #[test]
    fn test_scaling_identity() {
        let identity = LinearScaling::default();
        assert!(identity.is_identity());

        let non_identity = LinearScaling::new(0.1, 0.0);
        assert!(!non_identity.is_identity());
    }

    #[test]
    fn test_encode_int16() {
        let val = ModbusValue::Int(-1);
        let regs = encode_value(&val, ModbusDataType::Int16).unwrap();
        assert_eq!(regs, vec![0xFFFF]);
    }

    #[test]
    fn test_encode_float32() {
        let val = ModbusValue::Float(1.0);
        let regs = encode_value(&val, ModbusDataType::Float32).unwrap();
        assert_eq!(regs, vec![0x3F80, 0x0000]);
    }

    #[test]
    fn test_rdf_literal() {
        assert_eq!(ModbusValue::Int(42).to_rdf_literal(), "42");
        assert_eq!(ModbusValue::Float(22.5).to_rdf_literal(), "22.5");
        assert_eq!(ModbusValue::Bool(true).to_rdf_literal(), "true");
        assert_eq!(
            ModbusValue::String("test".to_string()).to_rdf_literal(),
            "test"
        );
    }

    #[test]
    fn test_xsd_datatypes() {
        assert_eq!(
            ModbusDataType::Int16.xsd_datatype(),
            "http://www.w3.org/2001/XMLSchema#short"
        );
        assert_eq!(
            ModbusDataType::Float32.xsd_datatype(),
            "http://www.w3.org/2001/XMLSchema#float"
        );
        assert_eq!(
            ModbusDataType::Bit(0).xsd_datatype(),
            "http://www.w3.org/2001/XMLSchema#boolean"
        );
    }

    #[test]
    fn test_register_count() {
        assert_eq!(ModbusDataType::Int16.register_count(), 1);
        assert_eq!(ModbusDataType::Float32.register_count(), 2);
        assert_eq!(ModbusDataType::Float64.register_count(), 4);
        assert_eq!(ModbusDataType::String(10).register_count(), 5);
    }
}
