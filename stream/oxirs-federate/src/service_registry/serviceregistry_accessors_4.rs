//! # ServiceRegistry - accessors Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{GraphQLService, HealthState};

impl ServiceRegistry {
    /// Get all healthy GraphQL services
    pub fn get_healthy_graphql_services(&self) -> Vec<GraphQLService> {
        self.graphql_services
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
