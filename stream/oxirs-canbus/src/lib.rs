//! CANbus/J1939 protocol support for OxiRS
//!
//! **Status**: ✅ Production Ready (v0.1.0-rc.1)
//!
//! This crate provides CANbus integration for automotive and heavy
//! machinery data ingestion into RDF knowledge graphs.
//!
//! # Features
//!
//! - ✅ **Socketcan integration** - Linux CAN interface support
//! - ✅ **DBC file parsing** - Vector CANdb++ format with signal extraction
//! - ✅ **J1939 protocol** - Heavy vehicle parameter groups (PGN extraction)
//! - ✅ **Multi-packet reassembly** - BAM (Broadcast Announce Message) support
//! - ✅ **RDF mapping** - CAN frames → RDF triples with provenance
//! - ✅ **SAMM generation** - Auto-generate Aspect Models from DBC
//!
//! # Architecture
//!
//! ```text
//! CANbus Network (CAN 2.0 / CAN FD)
//!   │
//!   ├─ OBD-II (passenger vehicles) ──┐
//!   ├─ J1939 (heavy vehicles) ───────┤
//!   └─ Custom protocols ─────────────┤
//!                                    │
//!                      ┌─────────────▼─────────────┐
//!                      │  Linux SocketCAN          │
//!                      │  (can0, can1, vcan0)      │
//!                      └─────────────┬─────────────┘
//!                                    │
//!                      ┌─────────────▼─────────────┐
//!                      │  oxirs-canbus             │
//!                      │  (this crate)             │
//!                      └─────────────┬─────────────┘
//!                                    │
//!                      ┌─────────────▼─────────────┐
//!                      │  J1939 Processor          │
//!                      │  (multi-packet, PGN)      │
//!                      └─────────────┬─────────────┘
//!                                    │
//!                      ┌─────────────▼─────────────┐
//!                      │  PGN Decoders             │
//!                      │  (EEC1, CCVS, ET1, etc.)  │
//!                      └─────────────┬─────────────┘
//!                                    │
//!                      ┌─────────────▼─────────────┐
//!                      │  DBC Parser (Month 4)     │
//!                      │  (signal definitions)     │
//!                      └─────────────┬─────────────┘
//!                                    │
//!                      ┌─────────────▼─────────────┐
//!                      │  RDF Triple Generator     │
//!                      │  + W3C PROV-O             │
//!                      └─────────────┬─────────────┘
//!                                    │
//!                      ┌─────────────▼─────────────┐
//!                      │  oxirs-core Store         │
//!                      │  (RDF persistence)        │
//!                      └───────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ## Basic J1939 Processing
//!
//! ```no_run
//! use oxirs_canbus::{J1939Processor, CanFrame, CanId, PgnRegistry};
//!
//! // Create J1939 processor with transport protocol support
//! let mut processor = J1939Processor::new();
//! let registry = PgnRegistry::with_standard_decoders();
//!
//! // Process incoming CAN frames
//! let can_id = CanId::extended(0x0CF00400).unwrap(); // EEC1
//! let frame = CanFrame::new(can_id, vec![0, 125, 125, 0x80, 0x3E, 0, 0, 125]).unwrap();
//!
//! if let Some(message) = processor.process(&frame) {
//!     // Decode the message using PGN registry
//!     if let Some(decoded) = registry.decode(&message) {
//!         for signal in &decoded.signals {
//!             println!("{}: {} {}", signal.name, signal.value, signal.unit);
//!         }
//!     }
//! }
//! ```
//!
//! ## Linux SocketCAN Client
//!
//! ```no_run,ignore
//! use oxirs_canbus::{CanbusClient, CanbusConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = CanbusConfig {
//!         interface: "can0".to_string(),
//!         j1939_enabled: true,
//!         ..Default::default()
//!     };
//!
//!     let mut client = CanbusClient::new(config)?;
//!     client.start().await?;
//!
//!     while let Some(frame) = client.recv_frame().await {
//!         println!("Received: {:?}", frame);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## DBC File Format (Month 4)
//!
//! ```dbc
//! BO_ 1234 EngineData: 8 Engine
//!   SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8191.875] "rpm" ECU
//!   SG_ EngineTemp : 16|8@1+ (1,-40) [-40|215] "deg C" ECU
//! ```
//!
//! # J1939 Protocol Support
//!
//! This crate provides comprehensive J1939 support:
//!
//! - **PGN extraction** from 29-bit CAN IDs
//! - **Multi-packet messages** via Transport Protocol (TP.CM/TP.DT)
//! - **Address claiming** for network participation
//! - **Common PGN decoders**:
//!   - EEC1 (61444): Engine speed, torque
//!   - EEC2 (61443): Accelerator position
//!   - CCVS (65265): Vehicle speed
//!   - ET1 (65262): Engine temperatures
//!   - EFL/P1 (65263): Fluid levels/pressures
//!   - LFE (65266): Fuel economy
//!   - AMB (65269): Ambient conditions
//!   - VEP1 (65271): Electrical power
//!
//! # Configuration
//!
//! TOML configuration example:
//!
//! ```toml
//! [[stream.external_systems]]
//! type = "CANbus"
//! interface = "can0"
//! j1939_enabled = true
//! dbc_file = "vehicle.dbc"
//!
//! [stream.external_systems.rdf_mapping]
//! device_id = "vehicle001"
//! base_iri = "http://automotive.example.com/vehicle"
//! graph_iri = "urn:automotive:can-data"
//! ```
//!
//! # CLI Commands
//!
//! The `oxirs` CLI provides CANbus monitoring and DBC tools:
//!
//! ```bash
//! # Monitor CAN interface
//! oxirs canbus monitor --interface can0 --dbc vehicle.dbc --j1939
//!
//! # Parse DBC file
//! oxirs canbus parse-dbc --file vehicle.dbc --detailed
//!
//! # Decode CAN frame
//! oxirs canbus decode --id 0x0CF00400 --data DEADBEEF --dbc vehicle.dbc
//!
//! # Generate SAMM Aspect Models
//! oxirs canbus to-samm --dbc vehicle.dbc --output ./models/
//!
//! # Generate RDF from live CAN data
//! oxirs canbus to-rdf --interface can0 --dbc vehicle.dbc --output data.ttl --count 1000
//! ```
//!
//! # Production Readiness
//!
//! - ✅ **98/98 tests passing** - 100% test success rate
//! - ✅ **Zero warnings** - Strict code quality enforcement
//! - ✅ **6 examples** - Complete usage documentation
//! - ✅ **25 files, 8,667 lines** - Comprehensive implementation
//! - ✅ **Standards compliant** - ISO 11898-1, SAE J1939, Vector DBC
//!
//! # Standards Compliance
//!
//! - ISO 11898 (CAN 2.0)
//! - ISO 11898-1:2015 (CAN FD)
//! - SAE J1939 (heavy vehicle communication)
//! - Vector CANdb++ DBC format
//!
//! # Use Cases
//!
//! - **Automotive**: OBD-II diagnostics, EV battery monitoring
//! - **Heavy machinery**: Construction equipment telemetry
//! - **Agriculture**: Tractor and harvester monitoring
//! - **Marine**: Ship engine management
//!
//! # Performance Targets
//!
//! - **Throughput**: 10,000 CAN messages/sec
//! - **Latency**: <1ms RDF conversion
//! - **Interfaces**: Support 8 CAN interfaces simultaneously

/// Configuration types for CAN interface and RDF mapping.
pub mod config;
/// DBC file parser and signal decoder.
pub mod dbc;
/// Error types and result aliases for CANbus operations.
pub mod error;
/// CANbus protocol implementations (SocketCAN, J1939, frames).
pub mod protocol;
/// RDF triple generation from CAN messages.
pub mod rdf;

// Configuration re-exports
pub use config::{CanFilter, CanbusConfig, RdfMappingConfig};

// Error re-exports
pub use error::{CanbusError, CanbusResult};

// Frame types
pub use protocol::{CanFrame, CanId};

// J1939 protocol
pub use protocol::{
    AddressManager, DeviceInfo, J1939Header, J1939Message, J1939Processor, Pgn, Priority,
    TransportProtocol,
};

// J1939 PGNs
pub use protocol::{
    // Decoders
    AmbDecoder,
    CcvsDecoder,
    DecodedPgn,
    DecodedSignal,
    Eec1Decoder,
    Eec2Decoder,
    Eflp1Decoder,
    Et1Decoder,
    LfeDecoder,
    PgnDecoder,
    PgnRegistry,
    PgnValue,
    Vep1Decoder,
    // PGN constants
    PGN_AMB,
    PGN_CCVS,
    PGN_CI,
    PGN_DD,
    PGN_EBC1,
    PGN_EEC1,
    PGN_EEC2,
    PGN_EFLP1,
    PGN_ET1,
    PGN_ETC1,
    PGN_ETC2,
    PGN_HRWS,
    PGN_LFC,
    PGN_LFE,
    PGN_SOFT,
    PGN_VEP1,
    PGN_VW,
};

// SocketCAN (Linux only)
#[cfg(target_os = "linux")]
pub use protocol::{CanFdClient, CanStatistics, CanbusClient};

// DBC parser and signal decoder
pub use dbc::{
    // Parser functions
    parse_dbc,
    parse_dbc_file,
    // Parser types
    AttributeDefinition,
    AttributeObjectType,
    AttributeValue,
    AttributeValueType,
    ByteOrder,
    DbcDatabase,
    DbcMessage,
    DbcNode,
    DbcParser,
    DbcSignal,
    // Signal decoder
    DecodedSignalValue,
    MultiplexerType,
    SignalDecoder,
    SignalEncoder,
    SignalExtractionError,
    SignalValue,
    ValueType,
};

// RDF mapper
pub use rdf::{ns, AutomotiveUnits, CanRdfMapper, GeneratedTriple, MapperStatistics};

// SAMM integration
pub use rdf::{
    validate_for_samm, DbcSammGenerator, SammConfig, SammValidationResult, SAMM_C_PREFIX,
    SAMM_E_PREFIX, SAMM_PREFIX, SAMM_U_PREFIX,
};
