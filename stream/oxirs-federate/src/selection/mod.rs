//! Advanced source selection for federated SPARQL queries.
//!
//! This module provides the core [`SourceSelector`] engine that determines
//! which SPARQL endpoints should answer which triple patterns in a federated
//! query, alongside supporting types and cost estimation utilities.

pub mod source_selector;

pub use source_selector::{
    explain_selection, AssignmentStrategy, EndpointCapabilities, FederatedPattern, SelectionReason,
    SourceAssignment, SourceSelector,
};
