//! # ServiceRegistry - Trait Implementations
//!
//! This module contains trait implementations for `ServiceRegistry`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ServiceRegistry;

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ServiceRegistry {
    fn clone(&self) -> Self {
        Self {
            sparql_endpoints: self.sparql_endpoints.clone(),
            graphql_services: self.graphql_services.clone(),
            health_status: self.health_status.clone(),
            capabilities_cache: self.capabilities_cache.clone(),
            extended_metadata: self.extended_metadata.clone(),
            service_patterns: self.service_patterns.clone(),
            http_client: self.http_client.clone(),
            config: self.config.clone(),
            health_monitor_handle: None,
        }
    }
}
