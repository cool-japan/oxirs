//! # ServiceRegistry - populate_service_capabilities_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::SparqlCapabilities;

impl ServiceRegistry {
    /// Helper method to populate service capabilities from SPARQL capabilities
    pub(super) fn populate_service_capabilities(
        &self,
        service: &mut crate::FederatedService,
        capabilities: &SparqlCapabilities,
    ) {
        if capabilities.supports_full_text_search {
            service
                .capabilities
                .insert(crate::ServiceCapability::FullTextSearch);
        }
        if capabilities.supports_geospatial {
            service
                .capabilities
                .insert(crate::ServiceCapability::Geospatial);
        }
        if capabilities.supports_update {
            service
                .capabilities
                .insert(crate::ServiceCapability::SparqlUpdate);
        }
        if capabilities.supports_rdf_star {
            service
                .capabilities
                .insert(crate::ServiceCapability::RdfStar);
        }
    }
}
