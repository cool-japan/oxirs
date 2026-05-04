//! Bidirectional synchronization between physics state and RDF.
//!
//! The existing parameter-extraction pipeline (`crate::simulation::ParameterExtractor`)
//! handles the **RDF → physics** direction: pull entity properties from an
//! RDF graph, parse typed literals, run a simulation. This module adds the
//! reverse direction:
//!
//! 1. [`state_to_rdf`] — render a physics state vector as RDF triples in a
//!    dedicated "state graph".
//! 2. [`rdf_to_state`] — re-extract parameters from the RDF graph and rebuild
//!    a physics state vector (used after an external update).
//! 3. [`bidirectional`] — orchestrates periodic, snapshot-isolated round
//!    trips between the two halves.
//!
//! State diffs use an in-house comparison (HashMap<String, f64>); we do not
//! pull in the SAMM aspect-differ for scalar diffs.

pub mod bidirectional;
pub mod rdf_to_state;
pub mod state_to_rdf;

pub use bidirectional::{
    BidirectionalSync, BidirectionalSyncConfig, BidirectionalSyncReport, SyncDirection,
};
pub use rdf_to_state::{RdfToStateExtractor, RdfToStateOutput};
pub use state_to_rdf::{
    state_diff, PhysicsState, PhysicsStateValue, StateDiff, StateGraphConfig, StateToRdfWriter,
};
