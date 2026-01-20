//! Register mapping configuration
//!
//! Defines mappings from Modbus registers to RDF predicates
//! with data type conversion and scaling.

use super::data_types::{LinearScaling, ModbusDataType};
use crate::error::{ModbusError, ModbusResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Register type (function code determines which register space)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegisterType {
    /// Holding registers (FC 0x03, 0x06, 0x10)
    #[default]
    Holding,
    /// Input registers (FC 0x04)
    Input,
    /// Coils (FC 0x01, 0x05, 0x0F)
    Coil,
    /// Discrete inputs (FC 0x02)
    DiscreteInput,
}

/// Byte order for multi-register values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ByteOrder {
    /// Big-endian (most significant word first) - Modbus standard
    #[default]
    BigEndian,
    /// Little-endian (least significant word first)
    LittleEndian,
    /// Big-endian with swapped words
    BigEndianSwapped,
    /// Little-endian with swapped words
    LittleEndianSwapped,
}

/// Enum value mapping (raw value → label)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnumMapping {
    /// Value-to-label mapping
    pub values: HashMap<u16, String>,
    /// Default label for unknown values
    #[serde(default)]
    pub default_label: Option<String>,
}

impl EnumMapping {
    /// Create new enum mapping
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a value-label pair
    pub fn with_value(mut self, value: u16, label: impl Into<String>) -> Self {
        self.values.insert(value, label.into());
        self
    }

    /// Look up label for a raw value
    pub fn get_label(&self, value: u16) -> Option<&str> {
        self.values
            .get(&value)
            .map(|s| s.as_str())
            .or(self.default_label.as_deref())
    }
}

/// Single register mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterMapping {
    /// Register address (0-65535)
    pub address: u16,

    /// Register type (holding, input, coil, discrete_input)
    #[serde(default)]
    pub register_type: RegisterType,

    /// Data type for decoding
    #[serde(with = "data_type_serde")]
    pub data_type: ModbusDataType,

    /// RDF predicate IRI
    pub predicate: String,

    /// Human-readable name
    #[serde(default)]
    pub name: Option<String>,

    /// Unit of measurement (QUDT URI or symbol)
    #[serde(default)]
    pub unit: Option<String>,

    /// Linear scaling parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scaling: Option<ScalingConfig>,

    /// Enum value mapping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<EnumMapping>,

    /// Byte order for multi-register types
    #[serde(default)]
    pub byte_order: ByteOrder,

    /// Deadband for change detection (absolute threshold)
    #[serde(default)]
    pub deadband: Option<f64>,

    /// Description
    #[serde(default)]
    pub description: Option<String>,
}

/// Scaling configuration for TOML serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Scale multiplier (physical_value = raw_value * multiplier + offset)
    pub multiplier: f64,
    /// Offset to add after scaling (default: 0.0)
    #[serde(default)]
    pub offset: f64,
}

impl From<ScalingConfig> for LinearScaling {
    fn from(cfg: ScalingConfig) -> Self {
        LinearScaling::new(cfg.multiplier, cfg.offset)
    }
}

impl From<LinearScaling> for ScalingConfig {
    fn from(s: LinearScaling) -> Self {
        ScalingConfig {
            multiplier: s.multiplier,
            offset: s.offset,
        }
    }
}

/// Serde helper for ModbusDataType
mod data_type_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(dt: &ModbusDataType, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&dt.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ModbusDataType, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl RegisterMapping {
    /// Create a new register mapping
    pub fn new(address: u16, data_type: ModbusDataType, predicate: impl Into<String>) -> Self {
        Self {
            address,
            register_type: RegisterType::default(),
            data_type,
            predicate: predicate.into(),
            name: None,
            unit: None,
            scaling: None,
            enum_values: None,
            byte_order: ByteOrder::default(),
            deadband: None,
            description: None,
        }
    }

    /// Set register type
    pub fn with_register_type(mut self, rt: RegisterType) -> Self {
        self.register_type = rt;
        self
    }

    /// Set name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set unit
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Set linear scaling
    pub fn with_scaling(mut self, multiplier: f64, offset: f64) -> Self {
        self.scaling = Some(ScalingConfig { multiplier, offset });
        self
    }

    /// Set deadband
    pub fn with_deadband(mut self, deadband: f64) -> Self {
        self.deadband = Some(deadband);
        self
    }

    /// Get scaling as LinearScaling
    pub fn get_scaling(&self) -> LinearScaling {
        self.scaling
            .as_ref()
            .map(|s| LinearScaling::new(s.multiplier, s.offset))
            .unwrap_or_default()
    }
}

/// Complete register map for a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterMap {
    /// Device identifier
    pub device_id: String,

    /// Base IRI for RDF subjects
    pub base_iri: String,

    /// Device description
    #[serde(default)]
    pub description: Option<String>,

    /// Register mappings
    pub registers: Vec<RegisterMapping>,

    /// Default polling interval (milliseconds)
    #[serde(default = "default_polling_interval")]
    pub polling_interval_ms: u64,
}

fn default_polling_interval() -> u64 {
    1000
}

impl RegisterMap {
    /// Create a new register map
    pub fn new(device_id: impl Into<String>, base_iri: impl Into<String>) -> Self {
        Self {
            device_id: device_id.into(),
            base_iri: base_iri.into(),
            description: None,
            registers: Vec::new(),
            polling_interval_ms: default_polling_interval(),
        }
    }

    /// Add a register mapping
    pub fn add_register(&mut self, mapping: RegisterMapping) {
        self.registers.push(mapping);
    }

    /// Load from TOML file
    pub fn from_toml<P: AsRef<Path>>(path: P) -> ModbusResult<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to read TOML file: {}", e),
            ))
        })?;
        Self::from_toml_str(&content)
    }

    /// Parse from TOML string
    pub fn from_toml_str(content: &str) -> ModbusResult<Self> {
        toml::from_str(content).map_err(|e| {
            ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse TOML: {}", e),
            ))
        })
    }

    /// Save to TOML file
    pub fn to_toml<P: AsRef<Path>>(&self, path: P) -> ModbusResult<()> {
        let content = self.to_toml_str()?;
        std::fs::write(path, content).map_err(|e| {
            ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write TOML file: {}", e),
            ))
        })
    }

    /// Serialize to TOML string
    pub fn to_toml_str(&self) -> ModbusResult<String> {
        toml::to_string_pretty(self).map_err(|e| {
            ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to serialize TOML: {}", e),
            ))
        })
    }

    /// Get subject IRI for this device
    pub fn subject_iri(&self) -> String {
        format!("{}/{}", self.base_iri.trim_end_matches('/'), self.device_id)
    }

    /// Get mapping for a specific address
    pub fn get_mapping(
        &self,
        address: u16,
        register_type: RegisterType,
    ) -> Option<&RegisterMapping> {
        self.registers
            .iter()
            .find(|m| m.address == address && m.register_type == register_type)
    }

    /// Get all holding register mappings
    pub fn holding_registers(&self) -> impl Iterator<Item = &RegisterMapping> {
        self.registers
            .iter()
            .filter(|m| m.register_type == RegisterType::Holding)
    }

    /// Get all input register mappings
    pub fn input_registers(&self) -> impl Iterator<Item = &RegisterMapping> {
        self.registers
            .iter()
            .filter(|m| m.register_type == RegisterType::Input)
    }

    /// Calculate optimal batch reads
    /// Returns list of (start_address, count) tuples
    pub fn batch_reads(&self, register_type: RegisterType, max_count: u16) -> Vec<(u16, u16)> {
        let mut addresses: Vec<u16> = self
            .registers
            .iter()
            .filter(|m| m.register_type == register_type)
            .map(|m| m.address)
            .collect();

        if addresses.is_empty() {
            return Vec::new();
        }

        addresses.sort_unstable();
        addresses.dedup();

        let mut batches = Vec::new();
        let mut batch_start = addresses[0];
        let mut batch_end = addresses[0];

        for &addr in &addresses[1..] {
            // Check if we can extend the current batch
            let gap = addr - batch_end;
            let new_count = addr - batch_start + 1;

            if gap <= 10 && new_count <= max_count {
                // Extend batch (small gap, within limit)
                batch_end = addr;
            } else {
                // Start new batch
                batches.push((batch_start, batch_end - batch_start + 1));
                batch_start = addr;
                batch_end = addr;
            }
        }

        // Add final batch
        batches.push((batch_start, batch_end - batch_start + 1));

        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_mapping_builder() {
        let mapping = RegisterMapping::new(
            100,
            ModbusDataType::Float32,
            "http://example.org/temperature",
        )
        .with_name("Temperature")
        .with_unit("CEL")
        .with_scaling(0.1, -40.0)
        .with_deadband(0.5);

        assert_eq!(mapping.address, 100);
        assert_eq!(mapping.name, Some("Temperature".to_string()));
        assert_eq!(mapping.unit, Some("CEL".to_string()));
        assert!(mapping.scaling.is_some());
        assert_eq!(mapping.deadband, Some(0.5));
    }

    #[test]
    fn test_register_map_toml() {
        let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");
        map.add_register(
            RegisterMapping::new(0, ModbusDataType::Float32, "http://example.org/temperature")
                .with_name("Temperature")
                .with_unit("CEL"),
        );
        map.add_register(
            RegisterMapping::new(2, ModbusDataType::Uint16, "http://example.org/status")
                .with_name("Status"),
        );

        let toml = map.to_toml_str().unwrap();
        assert!(toml.contains("plc001"));
        assert!(toml.contains("temperature"));

        // Round-trip
        let parsed = RegisterMap::from_toml_str(&toml).unwrap();
        assert_eq!(parsed.device_id, "plc001");
        assert_eq!(parsed.registers.len(), 2);
    }

    #[test]
    fn test_subject_iri() {
        let map = RegisterMap::new("plc001", "http://factory.example.com/device/");
        assert_eq!(
            map.subject_iri(),
            "http://factory.example.com/device/plc001"
        );

        let map2 = RegisterMap::new("plc002", "http://factory.example.com/device");
        assert_eq!(
            map2.subject_iri(),
            "http://factory.example.com/device/plc002"
        );
    }

    #[test]
    fn test_batch_reads() {
        let mut map = RegisterMap::new("test", "http://test");

        // Add registers at addresses 0, 1, 2, 10, 11, 100
        for addr in [0, 1, 2, 10, 11, 100] {
            map.add_register(RegisterMapping::new(
                addr,
                ModbusDataType::Uint16,
                format!("http://test/{}", addr),
            ));
        }

        let batches = map.batch_reads(RegisterType::Holding, 125);

        // Should create batches: [0-11], [100]
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], (0, 12)); // 0 to 11 inclusive
        assert_eq!(batches[1], (100, 1)); // Just 100
    }

    #[test]
    fn test_enum_mapping() {
        let mut em = EnumMapping::new()
            .with_value(0, "Off")
            .with_value(1, "On")
            .with_value(2, "Error");
        em.default_label = Some("Unknown".to_string());

        assert_eq!(em.get_label(0), Some("Off"));
        assert_eq!(em.get_label(1), Some("On"));
        assert_eq!(em.get_label(99), Some("Unknown"));
    }

    #[test]
    fn test_register_type_default() {
        assert_eq!(RegisterType::default(), RegisterType::Holding);
    }

    #[test]
    fn test_byte_order_default() {
        assert_eq!(ByteOrder::default(), ByteOrder::BigEndian);
    }
}
