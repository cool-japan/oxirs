//! DTDL v3 — Digital Twin Definition Language Support
//!
//! This module implements Microsoft DTDL v3 parsing, RDF mapping, and
//! semantic validation for the OxiRS physics crate.
//!
//! # Overview
//!
//! DTDL (Digital Twin Definition Language) is an open specification by
//! Microsoft for defining Digital Twin interfaces in IoT and industrial
//! contexts.  A DTDL v3 document is a JSON-LD object that declares one or
//! more **Interface** elements, each of which may contain:
//!
//! - **Telemetry** — time-series observable values (e.g. temperature)
//! - **Property** — settable configuration or state
//! - **Command** — request/response operation
//! - **Component** — composition via another Interface DTMI
//! - **Relationship** — directed link to other digital twins
//!
//! # Quick Start
//!
//! ```rust
//! use oxirs_physics::dtdl::{parse_dtdl_interface, interface_to_rdf, validate};
//!
//! let json = r#"{
//!     "@context": "dtmi:dtdl:context;3",
//!     "@type": "Interface",
//!     "@id": "dtmi:example:Thermostat;1",
//!     "displayName": "Thermostat",
//!     "contents": [
//!         { "@type": ["Telemetry","Temperature"], "name": "temperature", "schema": "double", "unit": "Celsius" },
//!         { "@type": "Property", "name": "targetTemperature", "schema": "double", "writable": true }
//!     ]
//! }"#;
//!
//! let iface = parse_dtdl_interface(json).expect("parse failed");
//! assert_eq!(iface.id.0, "dtmi:example:Thermostat;1");
//!
//! let errors = validate(&iface);
//! assert!(errors.is_empty());
//!
//! let triples = interface_to_rdf(&iface);
//! assert!(!triples.is_empty());
//! ```
//!
//! # Module layout
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | [`types`] | Core data structures: [`Dtmi`], [`DtdlInterface`], [`DtdlContent`], etc. |
//! | [`parser`] | JSON → typed structs (`parse_dtdl_interface`) |
//! | [`mapper`] | [`DtdlInterface`] → [`RdfTriple`]s, QUDT unit mapping |
//! | [`validator`] | Semantic validation rules (`validate`, `is_valid`) |
//!
//! # Coexistence with DTDL v2
//!
//! A lightweight DTDL v2 parser (`parse_dtdl_json`, `DtdlModel`) already
//! lives in [`crate::digital_twin`].  This module provides a richer v3
//! implementation with full RDF mapping and DTMI validation.  Both modules
//! can be used concurrently; the v3 types live in `crate::dtdl::types` and
//! are distinct from the simpler `digital_twin` v2 types.

pub mod mapper;
pub mod parser;
pub mod types;
pub mod validator;

// ─── Flat re-exports ─────────────────────────────────────────────────────────

pub use mapper::{
    dtdl_unit_to_qudt, interface_to_rdf, rdf_type_to_content_kind, RdfTriple, OXPHY_NS, QUDT_UNIT,
    RDFS_LABEL, RDF_TYPE,
};
pub use parser::{parse_dtdl_interface, DtdlParseError};
pub use types::{
    primary_type, DtdlCommandElement, DtdlComponentElement, DtdlContent, DtdlInterface,
    DtdlPropertyElement, DtdlRelationshipElement, DtdlSchema, DtdlTelemetryElement,
    DtdlValidationError, Dtmi,
};
pub use validator::{is_valid, validate};
