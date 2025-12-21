//! # ServiceRegistry - remove_service_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::{anyhow, Result};
use tracing::info;

impl ServiceRegistry {
    /// Remove a service
    pub async fn remove_service(&self, service_id: &str) -> Result<()> {
        info!("Removing service: {}", service_id);
        self.sparql_endpoints.remove(service_id);
        self.graphql_services.remove(service_id);
        self.health_status.remove(service_id);
        self.extended_metadata.remove(service_id);
        self.service_patterns.remove(service_id);
        let mut cache = self.capabilities_cache.write();
        cache.remove(service_id);
        Ok(())
    }
    /// Unregister a service (generic method)
    pub async fn unregister(&self, service_id: &str) -> Result<()> {
        let had_sparql = self.sparql_endpoints.contains_key(service_id);
        let had_graphql = self.graphql_services.contains_key(service_id);
        if !had_sparql && !had_graphql {
            return Err(anyhow!("Service '{}' not found", service_id));
        }
        self.remove_service(service_id).await
    }
}
