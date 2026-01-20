//! # ServiceRegistry - accessors Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::GraphQLService;

impl ServiceRegistry {
    /// Get all registered GraphQL services
    pub fn get_graphql_services(&self) -> Vec<GraphQLService> {
        self.graphql_services
            .iter()
            .map(|entry| entry.clone())
            .collect()
    }
}
