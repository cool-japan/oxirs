//! Service Registry Types
//!
//! Type and configuration definitions for federated services: service types,
//! capabilities, authentication, status, metadata, performance characteristics,
//! and connection pool descriptors.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::metadata::ExtendedServiceMetadata;

/// Type of federated service
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceType {
    /// SPARQL endpoint
    Sparql,
    /// GraphQL service
    GraphQL,
    /// Hybrid service supporting both SPARQL and GraphQL
    Hybrid,
    /// REST API with RDF support
    RestRdf,
    /// Custom service type
    Custom(String),
}

/// Service capabilities for federation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceCapability {
    /// Basic SPARQL 1.0 query support
    SparqlQuery,
    /// SPARQL 1.1 query support
    Sparql11Query,
    /// SPARQL 1.2 query support
    Sparql12Query,
    /// SPARQL UPDATE support
    SparqlUpdate,
    /// Graph Store Protocol support
    GraphStore,
    /// RDF-star support
    RdfStar,
    /// Full-text search capabilities
    FullTextSearch,
    /// Geospatial query support
    Geospatial,
    /// GraphQL query support
    GraphQLQuery,
    /// GraphQL mutation support
    GraphQLMutation,
    /// GraphQL subscription support
    GraphQLSubscription,
    /// Federation directives support
    GraphQLFederation,
    /// Custom extensions
    CustomExtension(String),
    /// Service can participate in transactions
    Transactional,
    /// Service supports versioning
    Versioning,
    /// SPARQL SERVICE clause support
    SparqlService,
    /// Federation support
    Federation,
    /// SPARQL aggregation support
    SparqlAggregation,
    /// SPARQL subqueries support
    SparqlSubqueries,
    /// SPARQL negation support
    SparqlNegation,
    /// SPARQL property paths support
    SparqlPropertyPaths,
    /// SPARQL GROUP BY support
    SparqlGroupBy,
    /// SPARQL VALUES support
    SparqlValues,
    /// RDF-star support (alternative name)
    RDFStar,
    /// RDFS reasoning support
    RDFSReasoning,
    /// Service supports real-time updates
    RealTimeUpdates,
    /// Service supports caching hints
    CacheHints,
    /// Service supports authentication
    Authentication,
    /// Service supports authorization
    Authorization,
    /// Service provides statistics
    Statistics,
    /// Service supports batch operations
    BatchOperations,
    /// Service supports temporal queries (NOW(), YEAR(), etc.)
    TemporalQueries,
    /// Service supports advanced filtering capabilities
    AdvancedFiltering,
    /// Service supports temporal queries
    TemporalQuery,
    /// Service supports numeric queries
    NumericQuery,
    /// Service supports schema repository queries
    SchemaRepositoryQuery,
    /// Service supports filter pushdown optimization
    FilterPushdown,
    /// Service supports projection pushdown optimization
    ProjectionPushdown,
    /// Service supports aggregation operations
    Aggregation,
    /// Service supports vector similarity search
    VectorSearch,
}

/// Authentication type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthType {
    /// No authentication required
    None,
    /// Basic HTTP authentication
    Basic,
    /// Bearer token authentication
    Bearer,
    /// API key authentication
    ApiKey,
    /// OAuth 2.0 authentication
    OAuth2,
    /// Custom authentication headers
    Custom,
}

/// Authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthCredentials {
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
    pub api_key: Option<String>,
    pub api_key_header: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub token_url: Option<String>,
    pub scope: Option<String>,
    pub custom_headers: Option<HashMap<String, String>>,
    pub refresh_token: Option<String>,
    pub token_endpoint: Option<String>,
}

/// Authentication configuration for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAuthConfig {
    pub auth_type: AuthType,
    pub credentials: AuthCredentials,
}

impl Default for ServiceAuthConfig {
    fn default() -> Self {
        Self {
            auth_type: AuthType::None,
            credentials: AuthCredentials::default(),
        }
    }
}

/// Service health status
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceStatus {
    /// Service is healthy and available
    Healthy,
    /// Service is partially available (degraded performance)
    Degraded,
    /// Service is temporarily unavailable
    Unavailable,
    /// Service status is unknown
    Unknown,
}

/// Overall health status for collections of services
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OverallHealthStatus {
    /// All services are healthy
    Healthy,
    /// Some services have issues but system is functional
    Degraded,
    /// Critical issues affecting functionality
    Critical,
    /// Unable to determine health status
    Unknown,
}

/// Service status information with load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatusInfo {
    /// Current service status
    pub status: ServiceStatus,
    /// Current load (0.0 to 1.0)
    pub current_load: f64,
    /// Last status check timestamp
    pub last_check: Option<chrono::DateTime<chrono::Utc>>,
    /// Response time in milliseconds
    pub response_time_ms: Option<f64>,
}

impl Default for ServiceStatusInfo {
    fn default() -> Self {
        Self {
            status: ServiceStatus::Unknown,
            current_load: 0.0,
            last_check: None,
            response_time_ms: None,
        }
    }
}

/// Configuration for the service registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceRegistryConfig {
    pub require_healthy_on_register: bool,
    pub health_check_interval: Duration,
    pub service_timeout: Duration,
    pub max_retry_attempts: usize,
    pub enable_capability_detection: bool,
    pub connection_pool_size: usize,
    pub enable_rate_limiting: bool,
}

impl Default for ServiceRegistryConfig {
    fn default() -> Self {
        Self {
            require_healthy_on_register: false,
            health_check_interval: Duration::from_secs(30),
            service_timeout: Duration::from_secs(10),
            max_retry_attempts: 3,
            enable_capability_detection: true,
            connection_pool_size: 10,
            enable_rate_limiting: true,
        }
    }
}

/// Represents a federated service (SPARQL endpoint or GraphQL service)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedService {
    /// Unique identifier for the service
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Service endpoint URL
    pub endpoint: String,
    /// Type of service (SPARQL, GraphQL, etc.)
    pub service_type: ServiceType,
    /// Service capabilities
    pub capabilities: HashSet<ServiceCapability>,
    /// Data patterns this service can handle
    pub data_patterns: Vec<String>,
    /// Authentication configuration
    pub auth: Option<ServiceAuthConfig>,
    /// Service metadata
    pub metadata: ServiceMetadata,
    /// Extended metadata (optional, for enhanced tracking)
    pub extended_metadata: Option<ExtendedServiceMetadata>,
    /// Performance characteristics
    pub performance: ServicePerformance,
    /// Current service status and load
    pub status: Option<ServiceStatusInfo>,
}

impl Default for FederatedService {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            endpoint: String::new(),
            service_type: ServiceType::Sparql,
            capabilities: HashSet::new(),
            data_patterns: vec!["*".to_string()],
            auth: None,
            metadata: ServiceMetadata::default(),
            extended_metadata: None,
            performance: ServicePerformance::default(),
            status: Some(ServiceStatusInfo::default()),
        }
    }
}

impl FederatedService {
    /// Create a new SPARQL service
    pub fn new_sparql(id: String, name: String, endpoint: String) -> Self {
        Self {
            id,
            name,
            endpoint,
            service_type: ServiceType::Sparql,
            capabilities: [
                ServiceCapability::SparqlQuery,
                ServiceCapability::SparqlUpdate,
            ]
            .into_iter()
            .collect(),
            data_patterns: vec!["*".to_string()], // Accept all patterns by default
            auth: None,
            metadata: ServiceMetadata::default(),
            extended_metadata: None,
            performance: ServicePerformance::default(),
            status: Some(ServiceStatusInfo::default()),
        }
    }

    /// Create a new GraphQL service
    pub fn new_graphql(id: String, name: String, endpoint: String) -> Self {
        Self {
            id,
            name,
            endpoint,
            service_type: ServiceType::GraphQL,
            capabilities: [
                ServiceCapability::GraphQLQuery,
                ServiceCapability::GraphQLMutation,
                ServiceCapability::GraphQLSubscription,
            ]
            .into_iter()
            .collect(),
            data_patterns: vec!["*".to_string()],
            auth: None,
            metadata: ServiceMetadata::default(),
            extended_metadata: None,
            performance: ServicePerformance::default(),
            status: Some(ServiceStatusInfo::default()),
        }
    }

    /// Validate service configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.id.is_empty() {
            return Err(anyhow::anyhow!("Service ID cannot be empty"));
        }

        if self.endpoint.is_empty() {
            return Err(anyhow::anyhow!("Service endpoint cannot be empty"));
        }

        // Validate URL format
        url::Url::parse(&self.endpoint)
            .map_err(|e| anyhow::anyhow!("Invalid endpoint URL: {}", e))?;

        if self.capabilities.is_empty() {
            return Err(anyhow::anyhow!("Service must have at least one capability"));
        }

        Ok(())
    }

    /// Check if service supports a specific capability
    pub fn supports_capability(&self, capability: &ServiceCapability) -> bool {
        self.capabilities.contains(capability)
    }

    /// Check if service can handle a specific data pattern
    pub fn handles_pattern(&self, pattern: &str) -> bool {
        self.data_patterns
            .iter()
            .any(|service_pattern| pattern_matches(pattern, service_pattern))
    }
}

/// Service metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServiceMetadata {
    pub description: Option<String>,
    pub version: Option<String>,
    pub maintainer: Option<String>,
    pub tags: Vec<String>,
    pub documentation_url: Option<String>,
    pub schema_url: Option<String>,
}

/// Service performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePerformance {
    pub average_response_time: Option<Duration>,
    pub avg_response_time_ms: f64,
    pub reliability_score: f64,
    pub max_concurrent_requests: Option<usize>,
    pub rate_limit: Option<RateLimit>,
    pub estimated_dataset_size: Option<u64>,
    pub supported_result_formats: Vec<String>,
    pub success_rate: Option<f64>,
    pub error_rate: Option<f64>,
    pub last_updated: Option<DateTime<Utc>>,
}

impl Default for ServicePerformance {
    fn default() -> Self {
        Self {
            average_response_time: None,
            avg_response_time_ms: 100.0,
            reliability_score: 0.9,
            max_concurrent_requests: None,
            rate_limit: None,
            estimated_dataset_size: None,
            supported_result_formats: vec![
                "application/sparql-results+json".to_string(),
                "application/sparql-results+xml".to_string(),
            ],
            success_rate: None,
            error_rate: None,
            last_updated: None,
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: usize,
    pub burst_limit: usize,
}

/// Statistics about the service registry
#[derive(Debug, Clone, Serialize)]
pub struct ServiceRegistryStats {
    pub total_services: usize,
    pub healthy_services: usize,
    pub capabilities_distribution: HashMap<ServiceCapability, usize>,
    #[serde(skip)]
    pub last_health_check: Option<Instant>,
}

/// OAuth2 token response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OAuth2TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub scope: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ConnectionPool {
    #[allow(dead_code)]
    pub(crate) service_id: String,
    #[allow(dead_code)]
    pub(crate) endpoint: String,
    pub(crate) max_connections: usize,
    pub(crate) active_connections: usize,
    pub(crate) created_at: u64,
    pub(crate) last_used: u64,
}

/// Connection pool statistics
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionPoolStats {
    pub max_connections: usize,
    pub active_connections: usize,
    pub created_at: u64,
    pub last_used: u64,
}

/// Check if a query pattern matches a service pattern
pub(crate) fn pattern_matches(query_pattern: &str, service_pattern: &str) -> bool {
    // Simple pattern matching - could be enhanced with regex or glob patterns
    if service_pattern == "*" {
        return true;
    }

    if service_pattern.contains('*') {
        // Simple wildcard matching
        let parts: Vec<&str> = service_pattern.split('*').collect();
        if parts.len() == 2 {
            let prefix = parts[0];
            let suffix = parts[1];
            return query_pattern.starts_with(prefix) && query_pattern.ends_with(suffix);
        }
    }

    query_pattern == service_pattern
}
