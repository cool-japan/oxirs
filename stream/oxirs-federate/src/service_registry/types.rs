//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use url::Url;

use super::functions::{deserialize_url, serialize_url};

/// Service description metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDescription {
    /// Default graph URIs
    pub default_graphs: Vec<String>,
    /// Named graph URIs
    pub named_graphs: Vec<String>,
    /// Supported languages
    pub languages: Vec<String>,
    /// Property functions
    pub property_functions: Vec<String>,
}
/// Connection configuration for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Connection timeout
    pub timeout: Duration,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Keep-alive settings
    pub keep_alive: bool,
    /// Compression enabled
    pub compression: bool,
}
/// Configuration for the service registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Health check interval
    pub health_check_interval: Duration,
    /// Service timeout for requests
    pub service_timeout: Duration,
    /// Maximum retry attempts for failed services
    pub max_retries: u32,
    /// Connection pool size per service
    pub connection_pool_size: usize,
    /// Enable automatic service discovery
    pub auto_discovery: bool,
    /// Service capability refresh interval
    pub capability_refresh_interval: Duration,
}
/// SPARQL endpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlEndpoint {
    /// Unique endpoint ID
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Endpoint URL
    #[serde(serialize_with = "serialize_url", deserialize_with = "deserialize_url")]
    pub url: Url,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// Endpoint capabilities
    pub capabilities: SparqlCapabilities,
    /// Performance statistics
    pub statistics: PerformanceStats,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last successful access
    pub last_access: Option<DateTime<Utc>>,
    /// Service metadata
    pub metadata: HashMap<String, String>,
    /// Connection pool configuration
    pub connection_config: ConnectionConfig,
}
/// GraphQL service metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLService {
    /// Unique service ID
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Service URL
    #[serde(serialize_with = "serialize_url", deserialize_with = "deserialize_url")]
    pub url: Url,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// GraphQL schema
    pub schema: Option<String>,
    /// Federation directives
    pub federation_directives: FederationDirectives,
    /// Service capabilities
    pub capabilities: GraphQLCapabilities,
    /// Performance statistics
    pub statistics: PerformanceStats,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last schema update
    pub schema_updated_at: Option<DateTime<Utc>>,
    /// Service metadata
    pub metadata: HashMap<String, String>,
}
/// GraphQL federation directives
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FederationDirectives {
    /// Entity key fields
    pub key_fields: HashMap<String, Vec<String>>,
    /// External fields
    pub external_fields: HashSet<String>,
    /// Required fields
    pub requires_fields: HashMap<String, Vec<String>>,
    /// Provided fields
    pub provides_fields: HashMap<String, Vec<String>>,
}
/// GraphQL service capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLCapabilities {
    /// GraphQL specification version
    pub graphql_version: String,
    /// Supports subscriptions
    pub supports_subscriptions: bool,
    /// Maximum query depth
    pub max_query_depth: Option<u32>,
    /// Maximum query complexity
    pub max_query_complexity: Option<u32>,
    /// Introspection enabled
    pub introspection_enabled: bool,
    /// Federation specification version
    pub federation_version: Option<String>,
}
/// Performance statistics for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// 95th percentile response time
    pub p95_response_time_ms: f64,
    /// Last recorded latency
    pub last_latency_ms: Option<u64>,
    /// Data freshness score (0.0 - 1.0)
    pub freshness_score: f64,
    /// Reliability score (0.0 - 1.0)
    pub reliability_score: f64,
}
/// Service type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceType {
    Sparql,
    GraphQL,
    Hybrid,
}
/// SPARQL endpoint capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlCapabilities {
    /// Supported SPARQL version
    pub sparql_version: SparqlVersion,
    /// Supported result formats
    pub result_formats: HashSet<String>,
    /// Supported graph formats for data import
    pub graph_formats: HashSet<String>,
    /// Custom functions available
    pub custom_functions: HashSet<String>,
    /// Maximum query complexity allowed
    pub max_query_complexity: Option<u32>,
    /// Supports federated queries (SERVICE)
    pub supports_federation: bool,
    /// Supports SPARQL UPDATE
    pub supports_update: bool,
    /// Supports named graphs
    pub supports_named_graphs: bool,
    /// Supports full text search
    pub supports_full_text_search: bool,
    /// Supports geospatial queries
    pub supports_geospatial: bool,
    /// Supports RDF-star (RDF*)
    pub supports_rdf_star: bool,
    /// Service description (SD) vocabulary support
    pub service_description: Option<ServiceDescription>,
}
/// SPARQL version enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparqlVersion {
    #[serde(rename = "1.0")]
    V10,
    #[serde(rename = "1.1")]
    V11,
    #[serde(rename = "1.2")]
    V12,
}
/// Capability data union
#[derive(Debug, Clone)]
pub enum CapabilityData {
    Sparql(SparqlCapabilities),
    GraphQL(GraphQLCapabilities),
}
/// Service health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Service ID
    pub service_id: String,
    /// Current health state
    pub status: HealthState,
    /// Last health check timestamp
    pub last_check: DateTime<Utc>,
    /// Consecutive failure count
    pub consecutive_failures: u32,
    /// Last error message
    pub last_error: Option<String>,
    /// Health check response time
    pub response_time_ms: Option<u64>,
}
/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthConfig {
    /// No authentication
    None,
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// API key authentication
    ApiKey { key: String, header: String },
    /// OAuth 2.0
    OAuth2 {
        token_url: String,
        client_id: String,
        client_secret: String,
    },
    /// Custom headers
    Custom { headers: HashMap<String, String> },
}
/// Health state enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthState {
    /// Service is healthy and operational
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
    /// Service is unknown (not yet checked)
    Unknown,
}
/// Service capabilities cache entry
#[derive(Debug, Clone)]
pub struct ServiceCapabilities {
    /// Service type
    pub service_type: ServiceType,
    /// Cached capabilities
    pub capabilities: CapabilityData,
    /// Cache timestamp
    pub cached_at: DateTime<Utc>,
    /// Cache expiry
    pub expires_at: DateTime<Utc>,
}
/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    pub total_sparql_endpoints: usize,
    pub total_graphql_services: usize,
    pub healthy_services: usize,
    pub degraded_services: usize,
    pub unhealthy_services: usize,
    pub last_health_check: Option<DateTime<Utc>>,
}
