//! Maps between OPC UA node IDs and Modbus registers using a [`BridgeConfig`].

use crate::opcua::config::{BridgeConfig, Direction, RegisterMapping};

/// Provides lookup between Modbus register addresses and OPC UA node IDs.
pub struct RegisterMapper {
    /// The bridge configuration this mapper was constructed from.
    pub config: BridgeConfig,
}

impl RegisterMapper {
    /// Create a new mapper from a bridge configuration.
    pub fn new(config: BridgeConfig) -> Self {
        Self { config }
    }

    /// Find the mapping for a given Modbus register address, if one exists.
    pub fn find_mapping(&self, modbus_register: u16) -> Option<&RegisterMapping> {
        self.config
            .mappings
            .iter()
            .find(|m| m.modbus_register == modbus_register)
    }

    /// Find the mapping for a given OPC UA node ID string, if one exists.
    pub fn find_mapping_by_node(&self, node_id: &str) -> Option<&RegisterMapping> {
        self.config
            .mappings
            .iter()
            .find(|m| m.opcua_node_id == node_id)
    }

    /// All mappings whose direction includes reading from Modbus (`Read` or
    /// `Bidirectional`).
    pub fn all_readable(&self) -> Vec<&RegisterMapping> {
        self.config
            .mappings
            .iter()
            .filter(|m| m.direction.is_readable())
            .collect()
    }

    /// All mappings whose direction includes writing to Modbus (`Write` or
    /// `Bidirectional`).
    pub fn all_writable(&self) -> Vec<&RegisterMapping> {
        self.config
            .mappings
            .iter()
            .filter(|m| m.direction.is_writable())
            .collect()
    }

    /// All mappings in the configuration.
    pub fn all_mappings(&self) -> &[RegisterMapping] {
        &self.config.mappings
    }

    /// Number of registered mappings.
    pub fn len(&self) -> usize {
        self.config.mappings.len()
    }

    /// Whether there are no mappings.
    pub fn is_empty(&self) -> bool {
        self.config.mappings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcua::config::{BridgeConfig, DataTypeSpec, Direction, RegisterMapping};

    fn sample_config() -> BridgeConfig {
        BridgeConfig {
            poll_interval_ms: 1000,
            modbus_host: "127.0.0.1".to_owned(),
            modbus_port: 502,
            opcua_endpoint: None,
            mappings: vec![
                RegisterMapping {
                    modbus_register: 100,
                    opcua_node_id: "ns=2;s=Temp".to_owned(),
                    data_type: DataTypeSpec::F32,
                    direction: Direction::Read,
                },
                RegisterMapping {
                    modbus_register: 200,
                    opcua_node_id: "ns=2;s=Pump".to_owned(),
                    data_type: DataTypeSpec::Bool,
                    direction: Direction::Bidirectional,
                },
                RegisterMapping {
                    modbus_register: 300,
                    opcua_node_id: "ns=2;s=Setpoint".to_owned(),
                    data_type: DataTypeSpec::U16,
                    direction: Direction::Write,
                },
            ],
        }
    }

    #[test]
    fn find_by_register() {
        let mapper = RegisterMapper::new(sample_config());
        let m = mapper.find_mapping(100).expect("should find");
        assert_eq!(m.opcua_node_id, "ns=2;s=Temp");
    }

    #[test]
    fn find_by_node() {
        let mapper = RegisterMapper::new(sample_config());
        let m = mapper
            .find_mapping_by_node("ns=2;s=Pump")
            .expect("should find");
        assert_eq!(m.modbus_register, 200);
    }

    #[test]
    fn missing_register_returns_none() {
        let mapper = RegisterMapper::new(sample_config());
        assert!(mapper.find_mapping(999).is_none());
    }

    #[test]
    fn readable_count() {
        let mapper = RegisterMapper::new(sample_config());
        // Read(100) and Bidirectional(200) are readable, Write(300) is not.
        assert_eq!(mapper.all_readable().len(), 2);
    }

    #[test]
    fn writable_count() {
        let mapper = RegisterMapper::new(sample_config());
        // Write(300) and Bidirectional(200) are writable, Read(100) is not.
        assert_eq!(mapper.all_writable().len(), 2);
    }
}
