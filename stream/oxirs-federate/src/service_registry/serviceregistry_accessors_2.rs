//! # ServiceRegistry - accessors Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::HealthStatus;

impl ServiceRegistry {
    /// Get service health status
    pub fn get_health_status(&self, service_id: &str) -> Option<HealthStatus> {
        self.health_status
            .get(service_id)
            .map(|entry| entry.clone())
    }
}
