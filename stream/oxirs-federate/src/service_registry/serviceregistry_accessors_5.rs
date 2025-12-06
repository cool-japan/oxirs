//! # ServiceRegistry - accessors Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{HealthState, RegistryStats};
use anyhow::Result;

impl ServiceRegistry {
    /// Get registry statistics
    pub async fn get_stats(&self) -> Result<RegistryStats> {
        Ok(RegistryStats {
            total_sparql_endpoints: self.sparql_endpoints.len(),
            total_graphql_services: self.graphql_services.len(),
            healthy_services: self
                .health_status
                .iter()
                .filter(|entry| entry.status == HealthState::Healthy)
                .count(),
            degraded_services: self
                .health_status
                .iter()
                .filter(|entry| entry.status == HealthState::Degraded)
                .count(),
            unhealthy_services: self
                .health_status
                .iter()
                .filter(|entry| entry.status == HealthState::Unhealthy)
                .count(),
            last_health_check: self
                .health_status
                .iter()
                .map(|entry| entry.last_check)
                .max(),
        })
    }
}
