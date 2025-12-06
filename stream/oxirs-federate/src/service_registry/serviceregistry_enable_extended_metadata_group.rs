//! # ServiceRegistry - enable_extended_metadata_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use tracing::{debug, info};

impl ServiceRegistry {
    /// Enable extended metadata tracking for a service
    pub fn enable_extended_metadata(&self, service_id: &str) {
        let service_exists = self.sparql_endpoints.contains_key(service_id)
            || self.graphql_services.contains_key(service_id);
        if service_exists {
            if !self.extended_metadata.contains_key(service_id) {
                let basic_metadata = crate::service::ServiceMetadata::default();
                let extended = crate::metadata::ExtendedServiceMetadata::from_basic(basic_metadata);
                self.extended_metadata
                    .insert(service_id.to_string(), extended);
                info!("Extended metadata enabled for service: {}", service_id);
            } else {
                debug!(
                    "Extended metadata already enabled for service: {}",
                    service_id
                );
            }
        } else {
            debug!(
                "Service {} not found, ignoring extended metadata enable request",
                service_id
            );
        }
    }
}
