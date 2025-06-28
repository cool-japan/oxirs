//! Service Capability Assessment
//!
//! This module provides comprehensive capability detection and assessment
//! for federated SPARQL and GraphQL services.

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
    metadata::{CapabilityDetail, QueryPattern},
    AuthConfig, FederatedService, ServiceCapability, ServiceType,
};

/// Capability assessor for detailed service analysis
#[derive(Debug)]
pub struct CapabilityAssessor {
    client: Client,
    config: AssessmentConfig,
}

impl CapabilityAssessor {
    /// Create a new capability assessor
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("oxirs-federate-capability-assessor/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config: AssessmentConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AssessmentConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Perform comprehensive capability assessment
    pub async fn assess_service(&self, service: &FederatedService) -> Result<AssessmentResult> {
        info!("Assessing capabilities for service: {}", service.id);
        let start_time = Instant::now();

        let mut result = AssessmentResult {
            service_id: service.id.clone(),
            detected_capabilities: HashSet::new(),
            capability_details: HashMap::new(),
            query_patterns: Vec::new(),
            performance_profile: PerformanceProfile::default(),
            limitations: Vec::new(),
            assessment_timestamp: chrono::Utc::now(),
        };

        match service.service_type {
            ServiceType::Sparql => {
                self.assess_sparql_capabilities(service, &mut result)
                    .await?;
            }
            ServiceType::GraphQL => {
                self.assess_graphql_capabilities(service, &mut result)
                    .await?;
            }
            ServiceType::Hybrid => {
                // Assess both SPARQL and GraphQL capabilities
                if let Err(e) = self.assess_sparql_capabilities(service, &mut result).await {
                    warn!("SPARQL assessment failed for hybrid service: {}", e);
                }
                if let Err(e) = self.assess_graphql_capabilities(service, &mut result).await {
                    warn!("GraphQL assessment failed for hybrid service: {}", e);
                }
            }
            ServiceType::RestRdf => {
                // REST-RDF typically supports SPARQL-like capabilities
                self.assess_sparql_capabilities(service, &mut result)
                    .await?;
            }
            ServiceType::Custom(_) => {
                // For custom services, try to assess both types
                if let Err(e) = self.assess_sparql_capabilities(service, &mut result).await {
                    warn!("SPARQL assessment failed for custom service: {}", e);
                }
                if let Err(e) = self.assess_graphql_capabilities(service, &mut result).await {
                    warn!("GraphQL assessment failed for custom service: {}", e);
                }
            }
        }

        let assessment_time = start_time.elapsed();
        debug!("Capability assessment completed in {:?}", assessment_time);

        Ok(result)
    }

    /// Assess SPARQL-specific capabilities
    async fn assess_sparql_capabilities(
        &self,
        service: &FederatedService,
        result: &mut AssessmentResult,
    ) -> Result<()> {
        // Test basic SPARQL 1.1 Query features
        let query_features = vec![
            ("SELECT", "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"),
            ("ASK", "ASK { ?s ?p ?o }"),
            ("DESCRIBE", "DESCRIBE <http://example.org/resource>"),
            (
                "CONSTRUCT",
                "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 1",
            ),
        ];

        for (feature, query) in query_features {
            if self.test_sparql_query(service, query).await.is_ok() {
                result
                    .detected_capabilities
                    .insert(ServiceCapability::SparqlQuery);
                debug!("Service supports SPARQL {} queries", feature);
            }
        }

        // Test SPARQL 1.1 Update features
        if self.config.test_updates {
            let update_features = vec![
                ("INSERT DATA", "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"),
                ("DELETE DATA", "DELETE DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"),
                ("DELETE/INSERT", "DELETE { ?s ?p ?o } INSERT { ?s ?p 'new' } WHERE { ?s ?p ?o }"),
            ];

            for (feature, update) in update_features {
                if self.test_sparql_update(service, update).await.is_ok() {
                    result
                        .detected_capabilities
                        .insert(ServiceCapability::SparqlUpdate);
                    debug!("Service supports SPARQL {} updates", feature);
                }
            }
        }

        // Test SPARQL 1.1 Federated Query (SERVICE clause)
        let service_query = "SELECT * WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }";
        if self.test_sparql_query(service, service_query).await.is_ok() {
            result
                .detected_capabilities
                .insert(ServiceCapability::SparqlService);
            result
                .detected_capabilities
                .insert(ServiceCapability::Federation);
        }

        // Test advanced SPARQL features
        self.test_advanced_sparql_features(service, result).await?;

        // Generate query patterns
        self.generate_sparql_query_patterns(result);

        Ok(())
    }

    /// Test advanced SPARQL features
    async fn test_advanced_sparql_features(
        &self,
        service: &FederatedService,
        result: &mut AssessmentResult,
    ) -> Result<()> {
        // Test aggregation functions
        let aggregation_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }";
        if self
            .test_sparql_query(service, aggregation_query)
            .await
            .is_ok()
        {
            result.capability_details.insert(
                "aggregation".to_string(),
                CapabilityDetail {
                    name: "SPARQL Aggregation".to_string(),
                    description: "Supports COUNT, SUM, AVG, MIN, MAX aggregation functions"
                        .to_string(),
                    version: Some("SPARQL 1.1".to_string()),
                    limitations: vec![],
                    performance_notes: None,
                },
            );
        }

        // Test subqueries
        let subquery = "SELECT ?s WHERE { ?s ?p ?o { SELECT ?s WHERE { ?s a ?type } LIMIT 10 } }";
        if self.test_sparql_query(service, subquery).await.is_ok() {
            result.capability_details.insert(
                "subqueries".to_string(),
                CapabilityDetail {
                    name: "SPARQL Subqueries".to_string(),
                    description: "Supports nested SELECT queries".to_string(),
                    version: Some("SPARQL 1.1".to_string()),
                    limitations: vec![],
                    performance_notes: Some(
                        "Performance may vary with subquery complexity".to_string(),
                    ),
                },
            );
        }

        // Test FILTER expressions
        let filter_query = "SELECT ?s WHERE { ?s ?p ?o FILTER(REGEX(?o, 'pattern', 'i')) }";
        if self.test_sparql_query(service, filter_query).await.is_ok() {
            result.capability_details.insert(
                "filters".to_string(),
                CapabilityDetail {
                    name: "SPARQL Filters".to_string(),
                    description: "Supports FILTER expressions including REGEX".to_string(),
                    version: Some("SPARQL 1.1".to_string()),
                    limitations: vec![],
                    performance_notes: Some(
                        "Complex filters may impact query performance".to_string(),
                    ),
                },
            );
        }

        // Test property paths
        let path_query =
            "SELECT ?s ?o WHERE { ?s (<http://example.org/p1>|<http://example.org/p2>)+ ?o }";
        if self.test_sparql_query(service, path_query).await.is_ok() {
            result.capability_details.insert(
                "property_paths".to_string(),
                CapabilityDetail {
                    name: "Property Paths".to_string(),
                    description: "Supports SPARQL 1.1 property path expressions".to_string(),
                    version: Some("SPARQL 1.1".to_string()),
                    limitations: vec![],
                    performance_notes: Some(
                        "Complex paths may be expensive to compute".to_string(),
                    ),
                },
            );
        }

        // Test named graphs
        let graph_query = "SELECT ?g ?s ?p ?o WHERE { GRAPH ?g { ?s ?p ?o } }";
        if self.test_sparql_query(service, graph_query).await.is_ok() {
            result.capability_details.insert(
                "named_graphs".to_string(),
                CapabilityDetail {
                    name: "Named Graphs".to_string(),
                    description: "Supports querying across named graphs".to_string(),
                    version: Some("SPARQL 1.1".to_string()),
                    limitations: vec![],
                    performance_notes: None,
                },
            );
        }

        Ok(())
    }

    /// Generate SPARQL query patterns
    fn generate_sparql_query_patterns(&self, result: &mut AssessmentResult) {
        result.query_patterns.push(QueryPattern {
            name: "simple_triple_pattern".to_string(),
            description: "Basic triple pattern query".to_string(),
            example_query: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100".to_string(),
            expected_response_time: Some(Duration::from_millis(100)),
            complexity_score: Some(1),
            tags: vec!["basic".to_string(), "discovery".to_string()],
        });

        if result.capability_details.contains_key("aggregation") {
            result.query_patterns.push(QueryPattern {
                name: "count_query".to_string(),
                description: "Count total triples".to_string(),
                example_query: "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }".to_string(),
                expected_response_time: Some(Duration::from_millis(500)),
                complexity_score: Some(3),
                tags: vec!["aggregation".to_string(), "statistics".to_string()],
            });
        }

        if result.capability_details.contains_key("filters") {
            result.query_patterns.push(QueryPattern {
                name: "filtered_search".to_string(),
                description: "Search with text filter".to_string(),
                example_query: "SELECT ?s ?label WHERE { ?s rdfs:label ?label FILTER(CONTAINS(LCASE(?label), 'search')) }".to_string(),
                expected_response_time: Some(Duration::from_millis(300)),
                complexity_score: Some(2),
                tags: vec!["search".to_string(), "filter".to_string()],
            });
        }
    }

    /// Assess GraphQL-specific capabilities
    async fn assess_graphql_capabilities(
        &self,
        service: &FederatedService,
        result: &mut AssessmentResult,
    ) -> Result<()> {
        // Test introspection
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
                        fields {
                            name
                            type {
                                name
                                kind
                            }
                        }
                    }
                }
            }
        "#;

        if let Ok(introspection) = self.test_graphql_query(service, introspection_query).await {
            result
                .detected_capabilities
                .insert(ServiceCapability::GraphQLQuery);

            // Parse introspection results
            if let Some(schema) = introspection.get("__schema") {
                if schema.get("mutationType").is_some() {
                    result
                        .detected_capabilities
                        .insert(ServiceCapability::GraphQLMutation);
                }
                if schema.get("subscriptionType").is_some() {
                    result
                        .detected_capabilities
                        .insert(ServiceCapability::GraphQLSubscription);
                }
            }

            result.capability_details.insert(
                "introspection".to_string(),
                CapabilityDetail {
                    name: "GraphQL Introspection".to_string(),
                    description: "Full schema introspection support".to_string(),
                    version: None,
                    limitations: vec![],
                    performance_notes: None,
                },
            );
        }

        // Test federation directives
        let federation_query = r#"
            query {
                _service {
                    sdl
                }
            }
        "#;

        if self
            .test_graphql_query(service, federation_query)
            .await
            .is_ok()
        {
            result
                .detected_capabilities
                .insert(ServiceCapability::Federation);
            result.capability_details.insert(
                "apollo_federation".to_string(),
                CapabilityDetail {
                    name: "Apollo Federation".to_string(),
                    description: "Supports Apollo Federation specification".to_string(),
                    version: Some("Federation 2.0".to_string()),
                    limitations: vec![],
                    performance_notes: None,
                },
            );
        }

        Ok(())
    }

    /// Test a SPARQL query
    async fn test_sparql_query(&self, service: &FederatedService, query: &str) -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );

        if let Some(auth) = &service.auth {
            self.add_auth_headers(&mut headers, auth)?;
        }

        let response = self
            .client
            .post(&service.endpoint)
            .headers(headers)
            .body(query.to_string())
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("Query failed with status: {}", response.status()))
        }
    }

    /// Test a SPARQL update
    async fn test_sparql_update(&self, service: &FederatedService, update: &str) -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-update"),
        );

        if let Some(auth) = &service.auth {
            self.add_auth_headers(&mut headers, auth)?;
        }

        let response = self
            .client
            .post(&service.endpoint)
            .headers(headers)
            .body(update.to_string())
            .send()
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("Update failed with status: {}", response.status()))
        }
    }

    /// Test a GraphQL query
    async fn test_graphql_query(
        &self,
        service: &FederatedService,
        query: &str,
    ) -> Result<serde_json::Value> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(auth) = &service.auth {
            self.add_auth_headers(&mut headers, auth)?;
        }

        let graphql_request = serde_json::json!({
            "query": query
        });

        let response = self
            .client
            .post(&service.endpoint)
            .headers(headers)
            .json(&graphql_request)
            .send()
            .await?;

        if response.status().is_success() {
            let result: serde_json::Value = response.json().await?;
            if let Some(data) = result.get("data") {
                Ok(data.clone())
            } else {
                Err(anyhow!("GraphQL response contains no data"))
            }
        } else {
            Err(anyhow!(
                "GraphQL query failed with status: {}",
                response.status()
            ))
        }
    }

    /// Add authentication headers
    fn add_auth_headers(&self, headers: &mut HeaderMap, auth: &AuthConfig) -> Result<()> {
        use crate::AuthType;
        use base64::encode;

        match &auth.auth_type {
            AuthType::Basic => {
                if let (Some(username), Some(password)) =
                    (&auth.credentials.username, &auth.credentials.password)
                {
                    let credentials = format!("{}:{}", username, password);
                    let encoded = encode(credentials.as_bytes());
                    let auth_value = format!("Basic {}", encoded);
                    headers.insert("Authorization", HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::Bearer => {
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {}", token);
                    headers.insert("Authorization", HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::ApiKey => {
                if let Some(api_key) = &auth.credentials.api_key {
                    headers.insert("X-API-Key", HeaderValue::from_str(api_key)?);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Measure query performance
    pub async fn measure_performance(
        &self,
        service: &FederatedService,
        test_queries: &[(&str, &str)],
    ) -> Result<PerformanceProfile> {
        let mut profile = PerformanceProfile::default();

        for (query_type, query) in test_queries {
            let start = Instant::now();
            let result = match service.service_type {
                ServiceType::Sparql => self.test_sparql_query(service, query).await,
                ServiceType::GraphQL => self.test_graphql_query(service, query).await.map(|_| ()),
                ServiceType::Hybrid => self.test_sparql_query(service, query).await,
                ServiceType::RestRdf => self.test_sparql_query(service, query).await,
                ServiceType::Custom(_) => self.test_sparql_query(service, query).await,
            };

            let duration = start.elapsed();

            if result.is_ok() {
                profile
                    .response_times
                    .insert(query_type.to_string(), duration);

                // Update average response time
                let total: u128 = profile.response_times.values().map(|d| d.as_millis()).sum();
                let count = profile.response_times.len() as u128;
                profile.avg_response_time = Duration::from_millis((total / count) as u64);
            }
        }

        Ok(profile)
    }
}

impl Default for CapabilityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for capability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentConfig {
    /// Request timeout
    pub timeout: Duration,
    /// User agent string
    pub user_agent: String,
    /// Whether to test update operations
    pub test_updates: bool,
    /// Whether to perform performance testing
    pub test_performance: bool,
    /// Maximum number of test queries
    pub max_test_queries: usize,
}

impl Default for AssessmentConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            user_agent: "oxirs-federate-capability-assessor/1.0".to_string(),
            test_updates: false,
            test_performance: true,
            max_test_queries: 20,
        }
    }
}

/// Result of capability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentResult {
    /// Service ID
    pub service_id: String,
    /// Detected capabilities
    pub detected_capabilities: HashSet<ServiceCapability>,
    /// Detailed capability information
    pub capability_details: HashMap<String, CapabilityDetail>,
    /// Supported query patterns
    pub query_patterns: Vec<QueryPattern>,
    /// Performance profile
    pub performance_profile: PerformanceProfile,
    /// Detected limitations
    pub limitations: Vec<String>,
    /// Assessment timestamp
    pub assessment_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Performance profile from assessment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Average response time
    pub avg_response_time: Duration,
    /// Response times for different query types
    pub response_times: HashMap<String, Duration>,
    /// Estimated throughput (queries per second)
    pub estimated_throughput: Option<f64>,
    /// Concurrent request handling capability
    pub concurrent_capability: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assessment_config() {
        let config = AssessmentConfig::default();
        assert!(!config.test_updates);
        assert!(config.test_performance);
        assert_eq!(config.max_test_queries, 20);
    }

    #[tokio::test]
    async fn test_capability_assessor_creation() {
        let assessor = CapabilityAssessor::new();
        assert!(assessor.config.test_performance);
    }

    #[test]
    fn test_performance_profile() {
        let mut profile = PerformanceProfile::default();
        profile
            .response_times
            .insert("simple".to_string(), Duration::from_millis(100));
        profile
            .response_times
            .insert("complex".to_string(), Duration::from_millis(500));

        let total: u128 = profile.response_times.values().map(|d| d.as_millis()).sum();
        let count = profile.response_times.len() as u128;
        let avg = Duration::from_millis((total / count) as u64);

        assert_eq!(avg.as_millis(), 300);
    }
}
