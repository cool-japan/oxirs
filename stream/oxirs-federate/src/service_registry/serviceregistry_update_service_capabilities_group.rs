//! # ServiceRegistry - update_service_capabilities_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::{anyhow, Result};
use std::collections::HashSet;
use tracing::{debug, info};

impl ServiceRegistry {
    /// Update service capabilities after assessment
    ///
    /// This method updates the capabilities of a registered service based on
    /// capability assessment results. It merges new capabilities with existing ones.
    ///
    /// # Arguments
    /// * `service_id` - The ID of the service to update
    /// * `new_capabilities` - Set of capabilities to add to the service
    ///
    /// # Returns
    /// * `Ok(())` if the service was found and updated
    /// * `Err` if the service was not found
    pub fn update_service_capabilities(
        &self,
        service_id: &str,
        new_capabilities: &HashSet<crate::ServiceCapability>,
    ) -> Result<()> {
        if let Some(mut endpoint_ref) = self.sparql_endpoints.get_mut(service_id) {
            let endpoint = endpoint_ref.value_mut();
            for cap in new_capabilities {
                match cap {
                    crate::ServiceCapability::FullTextSearch => {
                        endpoint.capabilities.supports_full_text_search = true;
                    }
                    crate::ServiceCapability::Geospatial => {
                        endpoint.capabilities.supports_geospatial = true;
                    }
                    crate::ServiceCapability::SparqlUpdate => {
                        endpoint.capabilities.supports_update = true;
                    }
                    crate::ServiceCapability::RdfStar => {
                        endpoint.capabilities.supports_rdf_star = true;
                    }
                    crate::ServiceCapability::Federation => {
                        endpoint.capabilities.supports_federation = true;
                    }
                    _ => {
                        debug!(
                            "Capability {:?} not directly applicable to SPARQL endpoint",
                            cap
                        );
                    }
                }
            }
            info!("Updated capabilities for SPARQL endpoint: {}", service_id);
            return Ok(());
        }
        if let Some(mut gql_service_ref) = self.graphql_services.get_mut(service_id) {
            let gql_service = gql_service_ref.value_mut();
            for cap in new_capabilities {
                match cap {
                    crate::ServiceCapability::GraphQLQuery => {}
                    crate::ServiceCapability::GraphQLMutation => {
                        debug!("GraphQL mutation capability detected");
                    }
                    crate::ServiceCapability::GraphQLSubscription => {
                        gql_service.capabilities.supports_subscriptions = true;
                    }
                    crate::ServiceCapability::GraphQLFederation => {
                        if gql_service.capabilities.federation_version.is_none() {
                            gql_service.capabilities.federation_version = Some("2.0".to_string());
                        }
                    }
                    crate::ServiceCapability::Federation => {
                        if gql_service.capabilities.federation_version.is_none() {
                            gql_service.capabilities.federation_version = Some("2.0".to_string());
                        }
                    }
                    _ => {
                        debug!(
                            "Capability {:?} not directly applicable to GraphQL service",
                            cap
                        );
                    }
                }
            }
            info!("Updated capabilities for GraphQL service: {}", service_id);
            return Ok(());
        }
        Err(anyhow!("Service with ID '{}' not found", service_id))
    }
}
