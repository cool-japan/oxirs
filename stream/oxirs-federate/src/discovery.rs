//! Service Discovery and Capability Detection
//!
//! This module provides automatic discovery of federated services and their capabilities,
//! including endpoint discovery, schema introspection, and capability analysis.

use anyhow::{anyhow, Result};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::{
    graphql::FederatedSchema, AuthConfig, AuthCredentials, AuthType, FederatedService,
    ServiceCapability, ServiceMetadata, ServicePerformance, ServiceRegistry, ServiceType,
};

/// Service discovery manager
#[derive(Debug)]
pub struct ServiceDiscovery {
    client: Client,
    config: DiscoveryConfig,
}

impl ServiceDiscovery {
    /// Create a new service discovery manager
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("oxirs-federate-discovery/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config: DiscoveryConfig::default(),
        }
    }

    /// Create a new service discovery manager with custom configuration
    pub fn with_config(config: DiscoveryConfig) -> Self {
        let client = Client::builder()
            .timeout(config.discovery_timeout)
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Discover services from a list of potential endpoints
    pub async fn discover_services(&self, endpoints: &[String]) -> Result<Vec<FederatedService>> {
        info!("Discovering services from {} endpoints", endpoints.len());

        let mut discovered_services = Vec::new();

        for endpoint in endpoints {
            match self.discover_service_at_endpoint(endpoint).await {
                Ok(Some(service)) => {
                    info!("Discovered service: {} at {}", service.name, endpoint);
                    discovered_services.push(service);
                }
                Ok(None) => {
                    debug!("No service found at endpoint: {}", endpoint);
                }
                Err(e) => {
                    warn!("Failed to discover service at {}: {}", endpoint, e);
                }
            }
        }

        Ok(discovered_services)
    }

    /// Discover a service at a specific endpoint
    pub async fn discover_service_at_endpoint(
        &self,
        endpoint: &str,
    ) -> Result<Option<FederatedService>> {
        debug!("Discovering service at: {}", endpoint);

        // Try to detect SPARQL endpoint
        if let Ok(Some(sparql_service)) = self.discover_sparql_service(endpoint).await {
            return Ok(Some(sparql_service));
        }

        // Try to detect GraphQL endpoint
        if let Ok(Some(graphql_service)) = self.discover_graphql_service(endpoint).await {
            return Ok(Some(graphql_service));
        }

        // Try to detect hybrid service
        if let Ok(Some(hybrid_service)) = self.discover_hybrid_service(endpoint).await {
            return Ok(Some(hybrid_service));
        }

        Ok(None)
    }

    /// Discover SPARQL service capabilities
    async fn discover_sparql_service(&self, endpoint: &str) -> Result<Option<FederatedService>> {
        // Try common SPARQL endpoint paths
        let sparql_paths = vec!["/sparql", "/query", "/sparql/query", ""];

        for path in sparql_paths {
            let full_endpoint = if path.is_empty() {
                endpoint.to_string()
            } else {
                format!("{}{}", endpoint.trim_end_matches('/'), path)
            };

            if let Ok(capabilities) = self.detect_sparql_capabilities(&full_endpoint).await {
                let service_id = self.generate_service_id(&full_endpoint);
                let metadata = self.extract_sparql_metadata(&full_endpoint).await?;
                let performance = self.analyze_sparql_performance(&full_endpoint).await?;

                return Ok(Some(FederatedService {
                    id: service_id,
                    name: format!("SPARQL Service at {}", full_endpoint),
                    endpoint: full_endpoint,
                    service_type: ServiceType::Sparql,
                    capabilities,
                    data_patterns: vec!["*".to_string()], // TODO: Analyze actual patterns
                    auth: None, // TODO: Detect authentication requirements
                    metadata,
                    extended_metadata: None,
                    performance,
                }));
            }
        }

        Ok(None)
    }

    /// Discover GraphQL service capabilities
    async fn discover_graphql_service(&self, endpoint: &str) -> Result<Option<FederatedService>> {
        // Try common GraphQL endpoint paths
        let graphql_paths = vec!["/graphql", "/api/graphql", "/gql", ""];

        for path in graphql_paths {
            let full_endpoint = if path.is_empty() {
                endpoint.to_string()
            } else {
                format!("{}{}", endpoint.trim_end_matches('/'), path)
            };

            if let Ok((capabilities, schema)) =
                self.detect_graphql_capabilities(&full_endpoint).await
            {
                let service_id = self.generate_service_id(&full_endpoint);
                let metadata = self
                    .extract_graphql_metadata(&full_endpoint, &schema)
                    .await?;
                let performance = self.analyze_graphql_performance(&full_endpoint).await?;

                return Ok(Some(FederatedService {
                    id: service_id,
                    name: format!("GraphQL Service at {}", full_endpoint),
                    endpoint: full_endpoint,
                    service_type: ServiceType::GraphQL,
                    capabilities,
                    data_patterns: vec!["*".to_string()], // TODO: Extract from schema
                    auth: None, // TODO: Detect authentication requirements
                    metadata,
                    extended_metadata: None,
                    performance,
                }));
            }
        }

        Ok(None)
    }

    /// Discover hybrid service (supports both SPARQL and GraphQL)
    async fn discover_hybrid_service(&self, endpoint: &str) -> Result<Option<FederatedService>> {
        // Check if the service supports both protocols
        let sparql_result = self.discover_sparql_service(endpoint).await;
        let graphql_result = self.discover_graphql_service(endpoint).await;

        match (sparql_result, graphql_result) {
            (Ok(Some(_)), Ok(Some(_))) => {
                // Service supports both protocols
                let service_id = self.generate_service_id(endpoint);
                let mut capabilities = HashSet::new();
                capabilities.insert(ServiceCapability::SparqlQuery);
                capabilities.insert(ServiceCapability::SparqlUpdate);
                capabilities.insert(ServiceCapability::GraphQLQuery);
                capabilities.insert(ServiceCapability::GraphQLMutation);
                capabilities.insert(ServiceCapability::Federation);

                Ok(Some(FederatedService {
                    id: service_id,
                    name: format!("Hybrid Service at {}", endpoint),
                    endpoint: endpoint.to_string(),
                    service_type: ServiceType::Hybrid,
                    capabilities,
                    data_patterns: vec!["*".to_string()],
                    auth: None,
                    metadata: ServiceMetadata::default(),
                    extended_metadata: None,
                    performance: ServicePerformance::default(),
                }))
            }
            _ => Ok(None),
        }
    }

    /// Detect SPARQL capabilities at an endpoint
    async fn detect_sparql_capabilities(
        &self,
        endpoint: &str,
    ) -> Result<HashSet<ServiceCapability>> {
        let mut capabilities = HashSet::new();

        // Test basic SPARQL query support
        let test_query = "ASK { ?s ?p ?o }";
        if self.test_sparql_query(endpoint, test_query).await.is_ok() {
            capabilities.insert(ServiceCapability::SparqlQuery);
        }

        // Test SPARQL update support
        let test_update =
            "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }";
        if self.test_sparql_update(endpoint, test_update).await.is_ok() {
            capabilities.insert(ServiceCapability::SparqlUpdate);
        }

        // Test SERVICE clause support (basic check)
        let service_query = "SELECT * WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }";
        if self
            .test_sparql_query(endpoint, service_query)
            .await
            .is_ok()
        {
            capabilities.insert(ServiceCapability::SparqlService);
        }

        if !capabilities.is_empty() {
            capabilities.insert(ServiceCapability::Federation);
        }

        Ok(capabilities)
    }

    /// Detect GraphQL capabilities at an endpoint
    async fn detect_graphql_capabilities(
        &self,
        endpoint: &str,
    ) -> Result<(HashSet<ServiceCapability>, Option<GraphQLIntrospection>)> {
        let mut capabilities = HashSet::new();

        // Test basic GraphQL query support with introspection
        let introspection_query = r#"
            query IntrospectionQuery {
                __schema {
                    queryType { name }
                    mutationType { name }
                    subscriptionType { name }
                    types {
                        name
                        kind
                        description
                    }
                }
            }
        "#;

        if let Ok(introspection) = self.test_graphql_query(endpoint, introspection_query).await {
            capabilities.insert(ServiceCapability::GraphQLQuery);

            // Check for mutation support
            if introspection.schema.mutation_type.is_some() {
                capabilities.insert(ServiceCapability::GraphQLMutation);
            }

            // Check for subscription support
            if introspection.schema.subscription_type.is_some() {
                capabilities.insert(ServiceCapability::GraphQLSubscription);
            }

            capabilities.insert(ServiceCapability::Federation);

            return Ok((capabilities, Some(introspection)));
        }

        Ok((capabilities, None))
    }

    /// Test SPARQL query execution
    async fn test_sparql_query(&self, endpoint: &str, query: &str) -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );

        let response = self
            .client
            .post(endpoint)
            .headers(headers)
            .body(query.to_string())
            .send()
            .await
            .map_err(|e| anyhow!("SPARQL query test failed: {}", e))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!(
                "SPARQL endpoint returned error: {}",
                response.status()
            ))
        }
    }

    /// Test SPARQL update execution
    async fn test_sparql_update(&self, endpoint: &str, update: &str) -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-update"),
        );

        let response = self
            .client
            .post(endpoint)
            .headers(headers)
            .body(update.to_string())
            .send()
            .await
            .map_err(|e| anyhow!("SPARQL update test failed: {}", e))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!(
                "SPARQL update endpoint returned error: {}",
                response.status()
            ))
        }
    }

    /// Test GraphQL query execution
    async fn test_graphql_query(
        &self,
        endpoint: &str,
        query: &str,
    ) -> Result<GraphQLIntrospection> {
        let graphql_request = serde_json::json!({
            "query": query
        });

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let response = self
            .client
            .post(endpoint)
            .headers(headers)
            .json(&graphql_request)
            .send()
            .await
            .map_err(|e| anyhow!("GraphQL query test failed: {}", e))?;

        if response.status().is_success() {
            let introspection: GraphQLIntrospectionResponse = response
                .json()
                .await
                .map_err(|e| anyhow!("Failed to parse GraphQL introspection: {}", e))?;

            if let Some(data) = introspection.data {
                Ok(data)
            } else {
                Err(anyhow!("GraphQL introspection returned no data"))
            }
        } else {
            Err(anyhow!(
                "GraphQL endpoint returned error: {}",
                response.status()
            ))
        }
    }

    /// Extract SPARQL service metadata
    async fn extract_sparql_metadata(&self, endpoint: &str) -> Result<ServiceMetadata> {
        // Try to get service description from common locations
        let mut metadata = ServiceMetadata::default();

        // Try .well-known/void descriptor
        if let Ok(void_desc) = self.fetch_void_description(endpoint).await {
            metadata.description = void_desc.description;
            metadata.version = void_desc.version;
        }

        Ok(metadata)
    }

    /// Extract GraphQL service metadata from introspection
    async fn extract_graphql_metadata(
        &self,
        endpoint: &str,
        introspection: &Option<GraphQLIntrospection>,
    ) -> Result<ServiceMetadata> {
        let mut metadata = ServiceMetadata::default();

        if let Some(intro) = introspection {
            metadata.description = intro.schema.query_type.description.clone();
            // TODO: Extract more metadata from schema
        }

        Ok(metadata)
    }

    /// Analyze SPARQL service performance
    async fn analyze_sparql_performance(&self, endpoint: &str) -> Result<ServicePerformance> {
        let mut performance = ServicePerformance::default();

        // Measure response time with a simple query
        let test_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }";
        let start_time = Instant::now();

        if self.test_sparql_query(endpoint, test_query).await.is_ok() {
            performance.average_response_time = Some(start_time.elapsed());
        }

        performance.supported_result_formats = vec![
            "application/sparql-results+json".to_string(),
            "application/sparql-results+xml".to_string(),
            "text/csv".to_string(),
        ];

        Ok(performance)
    }

    /// Analyze GraphQL service performance
    async fn analyze_graphql_performance(&self, endpoint: &str) -> Result<ServicePerformance> {
        let mut performance = ServicePerformance::default();

        // Measure response time with introspection query
        let introspection_query = "{ __schema { queryType { name } } }";
        let start_time = Instant::now();

        if self
            .test_graphql_query(endpoint, introspection_query)
            .await
            .is_ok()
        {
            performance.average_response_time = Some(start_time.elapsed());
        }

        performance.supported_result_formats = vec!["application/json".to_string()];

        Ok(performance)
    }

    /// Fetch VoID (Vocabulary of Interlinked Datasets) description
    async fn fetch_void_description(&self, endpoint: &str) -> Result<VoidDescription> {
        let void_url = format!("{}/.well-known/void", endpoint.trim_end_matches('/'));

        let response = self
            .client
            .get(&void_url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch VoID description: {}", e))?;

        if response.status().is_success() {
            let void_desc: VoidDescription = response
                .json()
                .await
                .map_err(|e| anyhow!("Failed to parse VoID description: {}", e))?;
            Ok(void_desc)
        } else {
            Err(anyhow!("VoID description not available"))
        }
    }

    /// Generate a unique service ID from endpoint
    fn generate_service_id(&self, endpoint: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        endpoint.hash(&mut hasher);
        format!("service_{:x}", hasher.finish())
    }

    /// Update service capabilities in the registry
    pub async fn update_service_capabilities(&self, registry: &mut ServiceRegistry) -> Result<()> {
        info!("Updating service capabilities in registry");

        let endpoints: Vec<String> = registry
            .get_all_services()
            .map(|service| service.endpoint.clone())
            .collect();

        for endpoint in endpoints {
            if let Ok(Some(updated_service)) = self.discover_service_at_endpoint(&endpoint).await {
                // Find existing service and update its capabilities
                if let Some(existing_service) = registry.get_service(&updated_service.id) {
                    let mut updated = existing_service.clone();
                    updated.capabilities = updated_service.capabilities;
                    updated.metadata = updated_service.metadata;
                    updated.performance = updated_service.performance;

                    // Re-register the updated service
                    registry.unregister(&updated.id).await?;
                    registry.register(updated).await?;
                }
            }
        }

        Ok(())
    }
}

impl Default for ServiceDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for service discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub discovery_timeout: Duration,
    pub user_agent: String,
    pub enable_introspection: bool,
    pub enable_performance_analysis: bool,
    pub max_concurrent_discoveries: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            discovery_timeout: Duration::from_secs(10),
            user_agent: "oxirs-federate-discovery/1.0".to_string(),
            enable_introspection: true,
            enable_performance_analysis: true,
            max_concurrent_discoveries: 5,
        }
    }
}

/// VoID (Vocabulary of Interlinked Datasets) description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoidDescription {
    pub description: Option<String>,
    pub version: Option<String>,
    pub title: Option<String>,
    pub creator: Option<String>,
    pub homepage: Option<String>,
    pub sparql_endpoint: Option<String>,
}

/// GraphQL introspection response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLIntrospectionResponse {
    pub data: Option<GraphQLIntrospection>,
    pub errors: Option<Vec<serde_json::Value>>,
}

/// GraphQL introspection data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLIntrospection {
    #[serde(rename = "__schema")]
    pub schema: GraphQLSchemaIntrospection,
}

/// GraphQL schema introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLSchemaIntrospection {
    #[serde(rename = "queryType")]
    pub query_type: GraphQLTypeRef,
    #[serde(rename = "mutationType")]
    pub mutation_type: Option<GraphQLTypeRef>,
    #[serde(rename = "subscriptionType")]
    pub subscription_type: Option<GraphQLTypeRef>,
    pub types: Vec<GraphQLTypeIntrospection>,
}

/// GraphQL type reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLTypeRef {
    pub name: String,
    pub description: Option<String>,
}

/// GraphQL type introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLTypeIntrospection {
    pub name: String,
    pub kind: String,
    pub description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_discovery_creation() {
        let discovery = ServiceDiscovery::new();
        assert!(discovery.config.enable_introspection);
    }

    #[test]
    fn test_service_id_generation() {
        let discovery = ServiceDiscovery::new();
        let endpoint = "http://example.com/sparql";

        let id1 = discovery.generate_service_id(endpoint);
        let id2 = discovery.generate_service_id(endpoint);

        assert_eq!(id1, id2); // Same endpoint should generate same ID
        assert!(id1.starts_with("service_"));
    }

    #[tokio::test]
    async fn test_void_description_parsing() {
        let discovery = ServiceDiscovery::new();

        // This test would require a mock server in a real implementation
        // For now, just test the structure
        let void_desc = VoidDescription {
            description: Some("Test dataset".to_string()),
            version: Some("1.0".to_string()),
            title: Some("Test Title".to_string()),
            creator: Some("Test Creator".to_string()),
            homepage: Some("http://example.com".to_string()),
            sparql_endpoint: Some("http://example.com/sparql".to_string()),
        };

        assert_eq!(void_desc.description.unwrap(), "Test dataset");
    }

    #[test]
    fn test_discovery_config() {
        let config = DiscoveryConfig::default();
        assert_eq!(config.max_concurrent_discoveries, 5);
        assert!(config.enable_introspection);
        assert!(config.enable_performance_analysis);
    }
}
