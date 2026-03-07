//! RDF/SPARQL Integration for Physics Simulations
//!
//! This module provides the bridge between OxiRS RDF triplestores and the
//! physics simulation engine.  It is organised into three focused sub-modules:
//!
//! - [`sparql_builder`]: Constructs SPARQL 1.1 SELECT and UPDATE queries for
//!   reading entity properties from and writing simulation results to an RDF
//!   triplestore.
//!
//! - [`literal_parser`]: Parses RDF typed literals (e.g. `"9.81 m/s^2"^^xsd:string`)
//!   into strongly-typed Rust values with full SI unit conversion.
//!
//! - [`physics_rdf`]: Full roundtrip conversion between physics simulation
//!   results and RDF triples using SOSA/SSN and QUDT ontologies, plus a
//!   lightweight SPARQL-like query layer over the generated triples.
//!
//! # Architecture
//!
//! ```text
//! RDF Triplestore
//!      │
//!      │  SPARQL SELECT (via sparql_builder)
//!      ▼
//! [VariableBinding rows]
//!      │
//!      │  parse_rdf_literal (via literal_parser)
//!      ▼
//! PhysicalValue (value + PhysicalUnit)
//!      │
//!      │  convert_unit (via literal_parser)
//!      ▼
//! SI-normalised value consumed by simulation
//!      │
//!      │  build_update_query (via sparql_builder)
//!      │  PhysicsToRdf::convert (via physics_rdf)
//!      ▼
//! SPARQL UPDATE / Turtle → RDF Triplestore
//! ```
//!
//! # Quick Start
//!
//! ```rust
//! use oxirs_physics::rdf::sparql_builder::{PhysicsPropertyQuery, PhysicsProperty};
//! use oxirs_physics::rdf::literal_parser::{parse_rdf_literal, PhysicalUnit, convert_unit};
//!
//! // 1. Build a SELECT query
//! let query = PhysicsPropertyQuery::new("urn:example:motor:42")
//!     .with_property(PhysicsProperty::Mass)
//!     .with_property(PhysicsProperty::Temperature)
//!     .build_select_query();
//!
//! assert!(query.contains("SELECT"));
//!
//! // 2. Parse a literal returned from the triplestore
//! let raw_val = parse_rdf_literal("75.0 kg", None).expect("parse failed");
//! assert_eq!(raw_val.unit, PhysicalUnit::KiloGram);
//!
//! // 3. Convert to SI (identity for kg)
//! let si = convert_unit(&raw_val, &PhysicalUnit::KiloGram).expect("conversion failed");
//! assert!((si.value - 75.0).abs() < 1e-10);
//! ```

pub mod literal_parser;
pub mod physics_rdf;
pub mod sparql_builder;

// Re-export frequently used types so consumers can import from `oxirs_physics::rdf`.
pub use literal_parser::{
    convert_unit, parse_rdf_literal, parse_unit_str, PhysicalUnit, PhysicalValue,
};
pub use physics_rdf::{
    PhysicsToRdf, RdfBoundaryCondition, RdfMaterialProperty, RdfToPhysics, SparqlPhysicsQuery,
    Triple, NS_EX, NS_PHYS, NS_PROV, NS_QUDT, NS_RDF, NS_RDFS, NS_SOSA, NS_SSN, NS_UNIT, NS_XSD,
};
pub use sparql_builder::{
    build_batch_select_query, build_property_replace_query, build_provenance_query,
    PhysicsProperty, PhysicsPropertyQuery, PrefixMap,
};
