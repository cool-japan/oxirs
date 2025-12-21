//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    GraphQLService, HealthStatus, RegistryConfig, ServiceCapabilities, SparqlEndpoint,
};
use dashmap::DashMap;
use parking_lot::RwLock;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;

/// Service registry for managing federated endpoints
#[derive(Debug)]
pub struct ServiceRegistry {
    /// Registered SPARQL endpoints
    pub(super) sparql_endpoints: Arc<DashMap<String, SparqlEndpoint>>,
    /// Registered GraphQL services
    pub(super) graphql_services: Arc<DashMap<String, GraphQLService>>,
    /// Service health status
    pub(super) health_status: Arc<DashMap<String, HealthStatus>>,
    /// Service capabilities cache
    pub(super) capabilities_cache: Arc<RwLock<HashMap<String, ServiceCapabilities>>>,
    /// Extended metadata tracking
    pub(super) extended_metadata: Arc<DashMap<String, crate::metadata::ExtendedServiceMetadata>>,
    /// Data patterns for each service
    pub(super) service_patterns: Arc<DashMap<String, Vec<String>>>,
    /// HTTP client for health checks and introspection
    pub(super) http_client: Client,
    /// Configuration
    pub(super) config: RegistryConfig,
    /// Health monitoring task handle
    pub(super) health_monitor_handle: Option<tokio::task::JoinHandle<()>>,
}

// Manual Clone implementation that skips the non-cloneable join handle
impl Clone for ServiceRegistry {
    fn clone(&self) -> Self {
        Self {
            sparql_endpoints: Arc::clone(&self.sparql_endpoints),
            graphql_services: Arc::clone(&self.graphql_services),
            health_status: Arc::clone(&self.health_status),
            capabilities_cache: Arc::clone(&self.capabilities_cache),
            extended_metadata: Arc::clone(&self.extended_metadata),
            service_patterns: Arc::clone(&self.service_patterns),
            http_client: self.http_client.clone(),
            config: self.config.clone(),
            health_monitor_handle: None, // Don't clone the join handle
        }
    }
}
