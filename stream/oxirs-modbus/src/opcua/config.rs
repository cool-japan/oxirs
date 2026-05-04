//! Bridge configuration: TOML-deserializable types for the Modbus ↔ OPC UA bridge.
//!
//! # Example TOML
//! ```toml
//! poll_interval_ms = 1000
//! modbus_host = "192.168.1.100"
//! modbus_port = 502
//! opcua_endpoint = "opc.tcp://localhost:4840"
//!
//! [[mappings]]
//! modbus_register = 40001
//! opcua_node_id = "ns=2;s=Temperature1"
//! data_type = "f32"
//! direction = "read"
//!
//! [[mappings]]
//! modbus_register = 1
//! opcua_node_id = "ns=2;s=Pump1"
//! data_type = "bool"
//! direction = "bidirectional"
//! ```

use serde::{Deserialize, Serialize};

/// Top-level bridge configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BridgeConfig {
    /// How often (in milliseconds) to poll Modbus registers and push to OPC UA.
    pub poll_interval_ms: u64,
    /// Modbus device hostname or IP.
    pub modbus_host: String,
    /// Modbus device TCP port (typically 502).
    pub modbus_port: u16,
    /// Optional OPC UA endpoint URL for the embedded server.
    /// When `None`, values are published in-process only (e.g. for testing).
    pub opcua_endpoint: Option<String>,
    /// List of register-to-node mappings.
    pub mappings: Vec<RegisterMapping>,
}

/// Maps one Modbus register to one OPC UA Variable node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterMapping {
    /// Modbus register address (0-based; e.g. 40001 in 1-based notation = 40000 here).
    pub modbus_register: u16,
    /// OPC UA node identifier string (e.g. `"ns=2;s=Temperature1"`).
    pub opcua_node_id: String,
    /// Data type used for coercion between the 16-bit Modbus word(s) and the typed value.
    pub data_type: DataTypeSpec,
    /// Data flow direction.
    pub direction: Direction,
}

/// Supported data types for register↔value coercion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum DataTypeSpec {
    /// Unsigned 16-bit integer (1 register).
    U16,
    /// Signed 16-bit integer (1 register).
    I16,
    /// Unsigned 32-bit integer (2 registers, big-endian word order).
    U32,
    /// Signed 32-bit integer (2 registers, big-endian word order).
    I32,
    /// IEEE 754 single-precision float (2 registers, big-endian word order).
    F32,
    /// Boolean: register 0 → false, any nonzero → true; true → 1u16 on write.
    Bool,
}

impl DataTypeSpec {
    /// Number of 16-bit Modbus registers this type occupies.
    pub fn register_count(&self) -> usize {
        match self {
            Self::U16 | Self::I16 | Self::Bool => 1,
            Self::U32 | Self::I32 | Self::F32 => 2,
        }
    }
}

/// Data-flow direction for a register mapping.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    /// Modbus → OPC UA only (read from device, publish to OPC UA).
    Read,
    /// OPC UA → Modbus only (subscribe to OPC UA writes, forward to device).
    Write,
    /// Both directions.
    Bidirectional,
}

impl Direction {
    /// Whether Modbus registers should be polled and published to OPC UA.
    pub fn is_readable(&self) -> bool {
        matches!(self, Self::Read | Self::Bidirectional)
    }

    /// Whether OPC UA writes should be forwarded to Modbus.
    pub fn is_writable(&self) -> bool {
        matches!(self, Self::Write | Self::Bidirectional)
    }
}
