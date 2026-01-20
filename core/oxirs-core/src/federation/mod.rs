//! SPARQL Federation support for distributed query execution
//!
//! This module provides HTTP client functionality for executing SPARQL queries
//! against remote endpoints via the SERVICE clause.

pub mod client;
pub mod executor;
pub mod results;

pub use client::{FederationClient, FederationConfig};
pub use executor::FederationExecutor;
pub use results::{Binding, SparqlResultsParser};
