//! Physics-RDF Roundtrip Conversion — facade re-exporting all sub-modules.
//!
//! Implementation is split across:
//! - [`crate::rdf::physics_rdf_types`]      — namespace constants, Triple, boundary/material structs
//! - [`crate::rdf::physics_rdf_mapper`]     — PhysicsToRdf, RdfToPhysics, helper fns
//! - [`crate::rdf::physics_rdf_serializer`] — SparqlPhysicsQuery, IRI helpers

pub use crate::rdf::physics_rdf_mapper::{PhysicsToRdf, RdfToPhysics};
pub use crate::rdf::physics_rdf_serializer::SparqlPhysicsQuery;
pub use crate::rdf::physics_rdf_types::{
    RdfBoundaryCondition, RdfMaterialProperty, Triple, NS_EX, NS_PHYS, NS_PROV, NS_QUDT, NS_RDF,
    NS_RDFS, NS_SOSA, NS_SSN, NS_UNIT, NS_XSD,
};
