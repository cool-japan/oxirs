//! W3C PROV-O ontology support for RDF data provenance tracking
//!
//! This module implements the W3C PROV-O (Provenance Ontology) for tracking
//! data provenance in RDF graphs. It supports the core PROV-O classes
//! (Entity, Activity, Agent), relations, and bundles.
//!
//! # References
//! - <https://www.w3.org/TR/prov-o/>
//! - <https://www.w3.org/TR/prov-dm/>
//!
//! # Module layout
//!
//! The implementation is split across sibling modules:
//! - `provenance_types`: namespace constants, [`AgentType`], and the core
//!   PROV-O classes [`ProvEntity`], [`ProvActivity`], [`ProvAgent`],
//!   [`ProvRelationKind`], and [`ProvRelation`].
//! - `provenance_bundle`: [`ProvBundle`] and [`QueryProvenanceTracker`].

mod provenance_bundle;
mod provenance_types;

#[cfg(test)]
mod provenance_tests;

pub use provenance_bundle::{ProvBundle, QueryProvenanceTracker};
pub use provenance_types::{
    AgentType, ProvActivity, ProvAgent, ProvEntity, ProvRelation, ProvRelationKind, PROV_NS,
    RDF_NS, XSD_NS,
};
