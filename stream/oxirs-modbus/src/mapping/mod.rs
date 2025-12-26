//! RDF mapping from Modbus registers
//!
//! This module provides register-to-RDF mapping functionality,
//! including data type conversions, scaling, and unit handling.
//!
//! # Overview
//!
//! The mapping system converts raw Modbus register values to typed
//! values suitable for RDF triple generation.
//!
//! # Example
//!
//! ```
//! use oxirs_modbus::mapping::{
//!     RegisterMap, RegisterMapping, ModbusDataType, RegisterType
//! };
//!
//! // Create a register map
//! let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");
//!
//! // Add temperature sensor mapping
//! map.add_register(
//!     RegisterMapping::new(
//!         0,
//!         ModbusDataType::Float32,
//!         "http://factory.example.com/property/temperature"
//!     )
//!     .with_name("Temperature")
//!     .with_unit("CEL")
//!     .with_scaling(0.1, 0.0)  // Scale raw value
//!     .with_deadband(0.5)      // Only update on >0.5 change
//! );
//!
//! // Get optimal batch reads
//! let batches = map.batch_reads(RegisterType::Holding, 125);
//! ```
//!
//! # Configuration
//!
//! Register maps can be loaded from TOML:
//!
//! ```toml
//! device_id = "plc001"
//! base_iri = "http://factory.example.com/device"
//! polling_interval_ms = 1000
//!
//! [[registers]]
//! address = 0
//! data_type = "FLOAT32"
//! predicate = "http://factory.example.com/property/temperature"
//! name = "Temperature"
//! unit = "CEL"
//!
//! [registers.scaling]
//! multiplier = 0.1
//! offset = 0.0
//! ```

pub mod data_types;
pub mod register_map;

#[cfg(feature = "samm-integration")]
pub mod samm_integration;

// Re-exports
pub use data_types::{decode_registers, encode_value, LinearScaling, ModbusDataType, ModbusValue};
pub use register_map::{
    ByteOrder, EnumMapping, RegisterMap, RegisterMapping, RegisterType, ScalingConfig,
};

#[cfg(feature = "samm-integration")]
pub use samm_integration::{validate_for_samm, SammGenerator, SammValidationResult};
