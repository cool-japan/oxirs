//! # ServiceRegistry - accessors Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use std::collections::{HashMap, HashSet};

impl ServiceRegistry {
    /// Get all registered SPARQL endpoints
    pub fn get_sparql_endpoints(&self) -> Vec<SparqlEndpoint> {
        self.sparql_endpoints
            .iter()
            .map(|entry| entry.clone())
            .collect()
    }
}
