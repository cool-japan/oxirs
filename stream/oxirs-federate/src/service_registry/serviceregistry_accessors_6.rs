//! # ServiceRegistry - accessors Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;

impl ServiceRegistry {
    /// Get a specific service by ID
    pub fn get_service(&self, service_id: &str) -> Option<crate::FederatedService> {
        if let Some(endpoint) = self.sparql_endpoints.get(service_id) {
            let mut service = crate::FederatedService::new_sparql(
                endpoint.id.clone(),
                endpoint.name.clone(),
                endpoint.url.to_string(),
            );
            if endpoint.capabilities.supports_full_text_search {
                service
                    .capabilities
                    .insert(crate::ServiceCapability::FullTextSearch);
            }
            if endpoint.capabilities.supports_geospatial {
                service
                    .capabilities
                    .insert(crate::ServiceCapability::Geospatial);
            }
            if endpoint.capabilities.supports_update {
                service
                    .capabilities
                    .insert(crate::ServiceCapability::SparqlUpdate);
            }
            if endpoint.capabilities.supports_rdf_star {
                service
                    .capabilities
                    .insert(crate::ServiceCapability::RdfStar);
            }
            if let Some(extended_meta) = self.extended_metadata.get(service_id) {
                service.extended_metadata = Some(extended_meta.clone());
            }
            if let Some(patterns) = self.service_patterns.get(service_id) {
                service.data_patterns = patterns.clone();
            }
            return Some(service);
        }
        if let Some(gql_service) = self.graphql_services.get(service_id) {
            let mut service = crate::FederatedService::new_graphql(
                gql_service.id.clone(),
                gql_service.name.clone(),
                gql_service.url.to_string(),
            );
            if let Some(extended_meta) = self.extended_metadata.get(service_id) {
                service.extended_metadata = Some(extended_meta.clone());
            }
            if let Some(patterns) = self.service_patterns.get(service_id) {
                service.data_patterns = patterns.clone();
            }
            return Some(service);
        }
        None
    }
}
