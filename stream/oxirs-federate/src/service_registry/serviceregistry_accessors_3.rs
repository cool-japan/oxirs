//! # ServiceRegistry - accessors Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{HealthState, SparqlEndpoint};

impl ServiceRegistry {
    /// Get all healthy SPARQL endpoints
    pub fn get_healthy_sparql_endpoints(&self) -> Vec<SparqlEndpoint> {
        self.sparql_endpoints
            .iter()
            .filter(|entry| {
                self.health_status
                    .get(entry.key())
                    .map(|status| status.status == HealthState::Healthy)
                    .unwrap_or(false)
            })
            .map(|entry| entry.clone())
            .collect()
    }
}
