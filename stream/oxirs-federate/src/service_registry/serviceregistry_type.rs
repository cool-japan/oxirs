//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

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
