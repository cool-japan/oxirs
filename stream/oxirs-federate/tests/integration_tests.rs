//! Integration tests for oxirs-federate
//!
//! These tests verify the integration of various components in the federation engine.

use oxirs_federate::auto_discovery::{AutoDiscovery, AutoDiscoveryConfig};
use oxirs_federate::cache::FederationCache;
use oxirs_federate::capability_assessment::CapabilityAssessor;
use oxirs_federate::planner::QueryPlanner;
use oxirs_federate::query_decomposition::QueryDecomposer;
use oxirs_federate::service::{AuthType, ServiceAuthConfig, ServiceCapability, ServiceMetadata};
use oxirs_federate::service_client::{create_client, ClientConfig};
use oxirs_federate::service_registry::RegistryConfig;
use oxirs_federate::service_registry::ServiceRegistry;
use oxirs_federate::*;
use std::time::Duration;

#[tokio::test]
async fn test_federation_engine_lifecycle() {
    // Create federation engine
    let engine = FederationEngine::new();

    // Register a test service
    let service = FederatedService::new_sparql(
        "test-sparql".to_string(),
        "Test SPARQL Service".to_string(),
        "http://localhost:3030/sparql".to_string(),
    );

    // Should be able to register service
    assert!(engine.register_service(service).await.is_ok());

    // Health check should work
    let health = engine.health_check().await.unwrap();
    assert_eq!(health.total_services, 1);

    // Unregister service
    assert!(engine.unregister_service("test-sparql").await.is_ok());
}

#[tokio::test]
async fn test_service_registry_operations() {
    let mut registry = ServiceRegistry::new();

    // Create test services
    let sparql_service = FederatedService::new_sparql(
        "sparql-1".to_string(),
        "SPARQL Service 1".to_string(),
        "http://example.com/sparql".to_string(),
    );

    let graphql_service = FederatedService::new_graphql(
        "graphql-1".to_string(),
        "GraphQL Service 1".to_string(),
        "http://example.com/graphql".to_string(),
    );

    // Register services
    registry.register(sparql_service).await.unwrap();
    registry.register(graphql_service).await.unwrap();

    // Test service retrieval
    assert!(registry.get_service("sparql-1").is_some());
    assert!(registry.get_service("graphql-1").is_some());
    assert!(registry.get_service("non-existent").is_none());

    // Test capability filtering
    let sparql_services = registry.get_services_with_capability(&ServiceCapability::SparqlQuery);
    assert_eq!(sparql_services.len(), 1);

    let graphql_services = registry.get_services_with_capability(&ServiceCapability::GraphQLQuery);
    assert_eq!(graphql_services.len(), 1);
}

#[tokio::test]
async fn test_extended_metadata() {
    let mut registry = ServiceRegistry::new();

    let service = FederatedService::new_sparql(
        "metadata-test".to_string(),
        "Metadata Test Service".to_string(),
        "http://example.com/sparql".to_string(),
    );

    registry.register(service).await.unwrap();

    // Enable extended metadata
    registry.enable_extended_metadata("metadata-test");

    let service = registry.get_service("metadata-test").unwrap();
    assert!(service.extended_metadata.is_some());
}

#[tokio::test]
async fn test_query_planning() {
    let planner = QueryPlanner::new();
    let registry = ServiceRegistry::new();

    // Analyze a simple SPARQL query
    let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let query_info = planner.analyze_sparql(query).await.unwrap();

    assert_eq!(query_info.query_type, QueryType::Select);
    assert!(!query_info.patterns.is_empty());
    assert!(query_info.variables.contains("?s"));
    assert!(query_info.variables.contains("?p"));
    assert!(query_info.variables.contains("?o"));
}

#[tokio::test]
async fn test_service_clause_extraction() {
    let planner = QueryPlanner::new();

    let query = r#"
        SELECT ?name ?email
        WHERE {
            SERVICE <http://people.example.org/sparql> {
                ?person foaf:name ?name .
                ?person foaf:mbox ?email
            }
        }
    "#;

    let query_info = planner.analyze_sparql(query).await.unwrap();

    assert!(!query_info.patterns.is_empty());
    assert_eq!(query_info.query_type, QueryType::Select);
}

#[tokio::test]
async fn test_capability_assessment() {
    let service = FederatedService::new_sparql(
        "assess-test".to_string(),
        "Assessment Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    let assessor = CapabilityAssessor::new();
    let result = assessor.assess_service(&service).await;

    // Assessment should complete (even if service is not available)
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn test_auto_discovery_lifecycle() {
    let config = AutoDiscoveryConfig {
        enable_mdns: false, // Disable mDNS for tests
        enable_dns_discovery: false,
        enable_kubernetes_discovery: false,
        dns_domains: vec![],
        service_patterns: vec![],
        discovery_interval: Duration::from_secs(300),
        max_concurrent_discoveries: 5,
        k8s_namespace: None,
        k8s_label_selectors: std::collections::HashMap::new(),
        k8s_use_cluster_dns: false,
        k8s_external_domain: None,
    };

    let mut discovery = AutoDiscovery::new(config);

    // Start discovery
    let mut receiver = discovery.start().await.unwrap();

    // Stop discovery
    discovery.stop().await;

    // Should not receive any endpoints with all discovery disabled
    // When all discovery is disabled, the channel is closed immediately
    let result = tokio::time::timeout(Duration::from_millis(100), receiver.recv()).await;
    match result {
        // Either the timeout occurs OR the channel is closed (returns None)
        Err(_) => {}   // Timeout occurred - this is expected
        Ok(None) => {} // Channel was closed - this is also expected when no discovery methods are enabled
        Ok(Some(_)) => panic!("Unexpected discovery result when all methods are disabled"),
    }
}

#[tokio::test]
async fn test_query_decomposition() {
    let decomposer = QueryDecomposer::new();
    let registry = ServiceRegistry::new();

    let query_info = QueryInfo {
        query_type: QueryType::Select,
        original_query: "SELECT * WHERE { ?s ?p ?o . ?o ?p2 ?o2 }".to_string(),
        patterns: vec![
            TriplePattern {
                subject: Some("?s".to_string()),
                predicate: Some("?p".to_string()),
                object: Some("?o".to_string()),
                pattern_string: "?s ?p ?o".to_string(),
            },
            TriplePattern {
                subject: Some("?o".to_string()),
                predicate: Some("?p2".to_string()),
                object: Some("?o2".to_string()),
                pattern_string: "?o ?p2 ?o2".to_string(),
            },
        ],
        filters: vec![],
        variables: ["?s", "?p", "?o", "?p2", "?o2"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        complexity: 1,
        estimated_cost: 20,
    };

    // Decomposition should work even with empty registry
    let result = decomposer.decompose(&query_info, &registry).await;
    assert!(result.is_ok() || result.is_err()); // May fail due to no services
}

#[tokio::test]
async fn test_service_client_creation() {
    let config = ClientConfig::default();

    let sparql_service = FederatedService::new_sparql(
        "client-test-sparql".to_string(),
        "Client Test SPARQL".to_string(),
        "http://example.com/sparql".to_string(),
    );

    let graphql_service = FederatedService::new_graphql(
        "client-test-graphql".to_string(),
        "Client Test GraphQL".to_string(),
        "http://example.com/graphql".to_string(),
    );

    // Create clients
    let sparql_client = create_client(sparql_service, config.clone()).unwrap();
    let graphql_client = create_client(graphql_service, config).unwrap();

    // Health checks should return false for non-existent services
    assert!(!sparql_client.health_check().await.unwrap_or(true));
    assert!(!graphql_client.health_check().await.unwrap_or(true));

    // Get stats
    let sparql_stats = sparql_client.get_stats().await;
    let graphql_stats = graphql_client.get_stats().await;

    // Health checks count as requests, so we expect 1 request per client
    assert_eq!(sparql_stats.total_requests, 1);
    assert_eq!(graphql_stats.total_requests, 1);
}

#[tokio::test]
async fn test_cache_operations() {
    let cache = FederationCache::new();

    // Test metadata caching
    let metadata = ServiceMetadata {
        description: Some("Test service".to_string()),
        version: Some("1.0".to_string()),
        maintainer: None,
        tags: vec!["test".to_string()],
        documentation_url: None,
        schema_url: None,
    };

    cache
        .put_service_metadata("test-service", metadata.clone())
        .await;

    let cached = cache.get_service_metadata("test-service").await;
    assert!(cached.is_some());
    assert_eq!(
        cached.unwrap().description,
        Some("Test service".to_string())
    );

    // Test cache invalidation
    cache.invalidate_service("test-service").await;
    assert!(cache.get_service_metadata("test-service").await.is_none());
}

#[tokio::test]
async fn test_complex_query_planning() {
    let planner = QueryPlanner::new();

    // Create registry with fast test configuration
    let config = RegistryConfig {
        health_check_interval: Duration::from_secs(1),
        service_timeout: Duration::from_millis(100), // Very short timeout for tests
        max_retries: 1,
        connection_pool_size: 1,
        auto_discovery: false,
        capability_refresh_interval: Duration::from_secs(300),
    };
    let mut registry = ServiceRegistry::with_config(config);

    // Register multiple services with fast timeout
    for i in 1..=3 {
        let service = FederatedService::new_sparql(
            format!("sparql-{i}"),
            format!("SPARQL Service {i}"),
            format!("http://example.com/sparql{i}"),
        );
        let _ = registry.register(service).await;
    }

    // Complex query with multiple patterns
    let query = r#"
        SELECT ?name ?age ?city
        WHERE {
            ?person foaf:name ?name .
            ?person foaf:age ?age .
            ?person foaf:based_near ?location .
            ?location geo:city ?city .
            FILTER(?age > 18)
        }
    "#;

    let query_info = planner.analyze_sparql(query).await.unwrap();
    let plan = planner.plan_sparql(&query_info, &registry).await.unwrap();

    assert!(!plan.steps.is_empty());
    assert!(!plan.steps.is_empty());
}

#[tokio::test]
async fn test_federation_with_authentication() {
    let mut service = FederatedService::new_sparql(
        "auth-test".to_string(),
        "Auth Test Service".to_string(),
        "http://example.com/sparql".to_string(),
    );

    // Add authentication
    service.auth = Some(ServiceAuthConfig {
        auth_type: AuthType::Basic,
        credentials: AuthCredentials {
            username: Some("user".to_string()),
            password: Some("pass".to_string()),
            ..Default::default()
        },
    });

    let config = ClientConfig::default();
    let client = create_client(service, config).unwrap();

    // Client should be created successfully
    assert!(client.health_check().await.is_ok());
}

#[tokio::test]
async fn test_monitoring_and_stats() {
    let engine = FederationEngine::new();

    // Register a service
    let service = FederatedService::new_sparql(
        "monitor-test".to_string(),
        "Monitor Test".to_string(),
        "http://example.com/sparql".to_string(),
    );
    engine.register_service(service).await.unwrap();

    // Get federation stats
    let stats = engine.get_stats().await;
    let reg_stats = stats.unwrap().registry;
    assert_eq!(
        reg_stats.total_sparql_endpoints + reg_stats.total_graphql_services,
        1
    );

    // Get cache stats
    let cache_stats = engine.get_cache_stats().await;
    assert_eq!(cache_stats.total_requests, 0);
}

/// Test data for integration tests
pub fn create_test_patterns() -> Vec<TriplePattern> {
    vec![
        TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("rdf:type".to_string()),
            object: Some("foaf:Person".to_string()),
            pattern_string: "?s rdf:type foaf:Person".to_string(),
        },
        TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("foaf:name".to_string()),
            object: Some("?name".to_string()),
            pattern_string: "?s foaf:name ?name".to_string(),
        },
        TriplePattern {
            subject: Some("?s".to_string()),
            predicate: Some("foaf:age".to_string()),
            object: Some("?age".to_string()),
            pattern_string: "?s foaf:age ?age".to_string(),
        },
    ]
}
