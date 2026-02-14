//! RDF integration and triple generation from CAN messages
//!
//! Converts CAN frames to RDF triples with W3C PROV-O provenance.
//!
//! # Features
//!
//! - **CAN to RDF mapping** - Decode CAN signals and generate RDF triples
//! - **DBC integration** - Use DBC files for signal definitions
//! - **PROV-O provenance** - W3C standard provenance tracking
//! - **QUDT units** - Standard unit ontology for automotive measurements
//!
//! # Example
//!
//! ```no_run
//! use oxirs_canbus::rdf::{CanRdfMapper, ns, AutomotiveUnits};
//! use oxirs_canbus::{parse_dbc, CanFrame, CanId};
//! use oxirs_canbus::config::RdfMappingConfig;
//!
//! let dbc_content = r#"
//! BO_ 2024 EngineData: 8 Engine
//!  SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
//! "#;
//!
//! let db = parse_dbc(dbc_content).expect("DBC parsing should succeed");
//! let config = RdfMappingConfig {
//!     device_id: "vehicle001".to_string(),
//!     base_iri: "http://automotive.example.com/vehicle".to_string(),
//!     graph_iri: "urn:automotive:can-data".to_string(),
//! };
//!
//! let mut mapper = CanRdfMapper::new(db, config);
//!
//! // Map a CAN frame to RDF triples
//! let id = CanId::standard(2024).expect("valid standard CAN ID");
//! let frame = CanFrame::new(id, vec![0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]).expect("valid CAN frame");
//! let triples = mapper.map_frame(&frame).expect("frame mapping should succeed");
//!
//! // Each triple includes provenance
//! for generated in &triples {
//!     println!("Signal: {}", generated.signal_name);
//!     println!("Triple: {:?}", generated.triple);
//!
//!     // Get PROV-O provenance triples
//!     for prov in generated.provenance_triples() {
//!         println!("Provenance: {:?}", prov);
//!     }
//! }
//! ```

pub mod mapper;
pub mod samm_integration;

// Re-export main types
pub use mapper::{ns, AutomotiveUnits, CanRdfMapper, GeneratedTriple, MapperStatistics};

// Re-export SAMM integration
pub use samm_integration::{
    validate_for_samm, DbcSammGenerator, SammConfig, SammValidationResult, SAMM_C_PREFIX,
    SAMM_E_PREFIX, SAMM_PREFIX, SAMM_U_PREFIX,
};
