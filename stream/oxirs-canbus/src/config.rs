use serde::{Deserialize, Serialize};

/// CANbus client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanbusConfig {
    /// CAN interface name (e.g., "can0", "vcan0")
    pub interface: String,

    /// Optional DBC file path for signal definitions
    pub dbc_file: Option<String>,

    /// CAN ID filters (empty = accept all)
    pub filters: Vec<CanFilter>,

    /// Enable J1939 protocol handling
    pub j1939_enabled: bool,

    /// RDF mapping configuration
    pub rdf_mapping: Option<RdfMappingConfig>,
}

/// CAN filter for selective message reception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanFilter {
    /// CAN identifier to match
    pub can_id: u32,

    /// Mask for identifier matching
    /// (0 = don't care, 1 = must match)
    pub mask: u32,
}

/// RDF mapping configuration for CAN messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfMappingConfig {
    /// Device identifier
    pub device_id: String,

    /// Base IRI for RDF subjects
    pub base_iri: String,

    /// Named graph IRI for storing CAN data
    pub graph_iri: String,
}

impl Default for CanbusConfig {
    fn default() -> Self {
        Self {
            interface: "can0".to_string(),
            dbc_file: None,
            filters: Vec::new(),
            j1939_enabled: false,
            rdf_mapping: None,
        }
    }
}

impl CanFilter {
    /// Create filter that accepts all CAN IDs
    pub fn accept_all() -> Self {
        Self { can_id: 0, mask: 0 }
    }

    /// Create filter for specific CAN ID (exact match)
    pub fn exact(can_id: u32) -> Self {
        Self {
            can_id,
            mask: 0x1FFFFFFF, // Match all 29 bits (extended)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CanbusConfig::default();
        assert_eq!(config.interface, "can0");
        assert!(!config.j1939_enabled);
    }

    #[test]
    fn test_can_filter() {
        let filter = CanFilter::exact(0x123);
        assert_eq!(filter.can_id, 0x123);
        assert_eq!(filter.mask, 0x1FFFFFFF);
    }
}
