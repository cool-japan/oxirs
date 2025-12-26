//! RDF integration and triple generation
//!
//! This module handles conversion of Modbus register data
//! to RDF triples with W3C PROV-O provenance tracking.
//!
//! # Overview
//!
//! The RDF module provides:
//! - Triple generation from Modbus register values
//! - W3C PROV-O provenance metadata (timestamps)
//! - QUDT unit handling
//! - Change detection with deadband filtering
//! - SPARQL UPDATE execution for graph persistence
//!
//! # Example
//!
//! ```ignore
//! use oxirs_modbus::rdf::{ModbusTripleGenerator, QudtUnit, GraphUpdater, SparqlEndpointConfig};
//! use oxirs_modbus::mapping::{RegisterMap, RegisterMapping, ModbusDataType, RegisterType};
//! use std::collections::HashMap;
//! use chrono::Utc;
//!
//! // Create register map
//! let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");
//! map.add_register(
//!     RegisterMapping::new(
//!         0, ModbusDataType::Float32,
//!         "http://factory.example.com/property/temperature"
//!     )
//!     .with_unit(&QudtUnit::celsius())
//! );
//!
//! // Create triple generator
//! let mut generator = ModbusTripleGenerator::new(map);
//!
//! // Generate triples from register values
//! let mut values = HashMap::new();
//! values.insert(0u16, vec![0x41B4, 0x0000]); // 22.5°C
//!
//! let triples = generator.generate_triples(
//!     &values,
//!     RegisterType::Holding,
//!     Utc::now()
//! ).unwrap();
//!
//! // Create graph updater for SPARQL UPDATE
//! let config = SparqlEndpointConfig::new("http://localhost:3030/dataset/update");
//! let mut updater = GraphUpdater::new(config)
//!     .with_graph("http://factory.example.com/graph/modbus");
//!
//! // Build INSERT DATA query (or execute via HTTP)
//! let query = updater.insert_generated_local(&triples).unwrap();
//! println!("SPARQL: {}", query);
//! ```

pub mod graph_updater;
pub mod triple_generator;

// Re-exports
pub use graph_updater::{BatchUpdater, GraphUpdater, SparqlEndpointConfig, UpdateStats};
pub use triple_generator::{GeneratedTriple, ModbusTripleGenerator, QudtUnit};
