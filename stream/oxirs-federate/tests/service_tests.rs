//! Unit tests for service registry and management

use oxirs_federate::service::{AuthType, RateLimit, ServiceAuthConfig, ServiceMetadata};
use oxirs_federate::*;
use std::collections::HashSet;
use std::time::Duration;

#[tokio::test]
async fn test_service_creation() {
    // Test SPARQL service creation
    let sparql = FederatedService::new_sparql(
        "sparql-test".to_string(),
        "SPARQL Test Service".to_string(),
        "http://example.com/sparql".to_string(),
    );

    assert_eq!(sparql.id, "sparql-test");
    assert_eq!(sparql.service_type, ServiceType::Sparql);
    assert!(sparql
        .capabilities
        .contains(&ServiceCapability::SparqlQuery));
    assert!(sparql
        .capabilities
        .contains(&ServiceCapability::SparqlUpdate));

    // Test GraphQL service creation
    let graphql = FederatedService::new_graphql(
        "graphql-test".to_string(),
        "GraphQL Test Service".to_string(),
        "http://example.com/graphql".to_string(),
    );

    assert_eq!(graphql.id, "graphql-test");
    assert_eq!(graphql.service_type, ServiceType::GraphQL);
    assert!(graphql
        .capabilities
        .contains(&ServiceCapability::GraphQLQuery));
    assert!(graphql
        .capabilities
        .contains(&ServiceCapability::GraphQLMutation));
    assert!(graphql
        .capabilities
        .contains(&ServiceCapability::GraphQLSubscription));
}

#[tokio::test]
async fn test_service_validation() {
    // Valid service
    let valid = FederatedService::new_sparql(
        "valid".to_string(),
        "Valid Service".to_string(),
        "http://example.com/sparql".to_string(),
    );
    assert!(valid.validate().is_ok());

    // Invalid service - empty ID
    let mut invalid = valid.clone();
    invalid.id = String::new();
    assert!(invalid.validate().is_err());

    // Invalid service - empty endpoint
    let mut invalid = valid.clone();
    invalid.endpoint = String::new();
    assert!(invalid.validate().is_err());

    // Invalid service - malformed URL
    let mut invalid = valid.clone();
    invalid.endpoint = "not-a-url".to_string();
    assert!(invalid.validate().is_err());

    // Invalid service - no capabilities
    let mut invalid = valid.clone();
    invalid.capabilities.clear();
    assert!(invalid.validate().is_err());
}

#[tokio::test]
async fn test_service_registry_registration() {
    let mut registry = ServiceRegistry::new();

    let service = FederatedService::new_sparql(
        "reg-test".to_string(),
        "Registration Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    // Should register successfully
    assert!(registry.register(service.clone()).await.is_ok());

    // Should fail to register duplicate
    assert!(registry.register(service).await.is_err());

    // Should be retrievable
    assert!(registry.get_service("reg-test").is_some());
}

#[tokio::test]
async fn test_service_unregistration() {
    let mut registry = ServiceRegistry::new();

    let service = FederatedService::new_sparql(
        "unreg-test".to_string(),
        "Unregistration Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    registry.register(service).await.unwrap();
    assert!(registry.get_service("unreg-test").is_some());

    // Should unregister successfully
    assert!(registry.unregister("unreg-test").await.is_ok());
    assert!(registry.get_service("unreg-test").is_none());

    // Should fail to unregister non-existent
    assert!(registry.unregister("unreg-test").await.is_err());
}

#[tokio::test]
async fn test_capability_filtering() {
    let mut registry = ServiceRegistry::new();

    // Register services with different capabilities
    let mut sparql = FederatedService::new_sparql(
        "cap-sparql".to_string(),
        "Capability SPARQL".to_string(),
        "http://example.com/sparql".to_string(),
    );
    sparql
        .capabilities
        .insert(ServiceCapability::FullTextSearch);

    let graphql = FederatedService::new_graphql(
        "cap-graphql".to_string(),
        "Capability GraphQL".to_string(),
        "http://example.com/graphql".to_string(),
    );

    registry.register(sparql).await.unwrap();
    registry.register(graphql).await.unwrap();

    // Test filtering
    let sparql_services = registry.get_services_with_capability(&ServiceCapability::SparqlQuery);
    assert_eq!(sparql_services.len(), 1);
    assert_eq!(sparql_services[0].id, "cap-sparql");

    let graphql_services = registry.get_services_with_capability(&ServiceCapability::GraphQLQuery);
    assert_eq!(graphql_services.len(), 1);
    assert_eq!(graphql_services[0].id, "cap-graphql");

    let fulltext_services =
        registry.get_services_with_capability(&ServiceCapability::FullTextSearch);
    assert_eq!(fulltext_services.len(), 1);

    let geo_services = registry.get_services_with_capability(&ServiceCapability::Geospatial);
    assert_eq!(geo_services.len(), 0);
}

#[tokio::test]
async fn test_pattern_matching() {
    let mut registry = ServiceRegistry::new();

    let mut service = FederatedService::new_sparql(
        "pattern-test".to_string(),
        "Pattern Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    // Add specific data patterns
    service.data_patterns = vec![
        "http://dbpedia.org/*".to_string(),
        "http://xmlns.com/foaf/*".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#*".to_string(),
    ];

    registry.register(service).await.unwrap();

    // Test pattern matching
    let patterns = vec![
        "http://dbpedia.org/resource/Berlin".to_string(),
        "http://xmlns.com/foaf/0.1/Person".to_string(),
    ];

    let matching_services = registry.get_services_for_patterns(&patterns);
    assert_eq!(matching_services.len(), 1);
    assert_eq!(matching_services[0].id, "pattern-test");

    // Test non-matching pattern
    let non_matching = vec!["http://example.org/unknown".to_string()];
    let result = registry.get_services_for_patterns(&non_matching);
    assert_eq!(result.len(), 0);
}

#[tokio::test]
async fn test_service_authentication() {
    let mut service = FederatedService::new_sparql(
        "auth-test".to_string(),
        "Auth Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    // Test Basic auth
    service.auth = Some(ServiceAuthConfig {
        auth_type: AuthType::Basic,
        credentials: AuthCredentials {
            username: Some("testuser".to_string()),
            password: Some("testpass".to_string()),
            ..Default::default()
        },
    });

    assert!(service.auth.is_some());
    assert_eq!(service.auth.as_ref().unwrap().auth_type, AuthType::Basic);

    // Test Bearer token auth
    service.auth = Some(ServiceAuthConfig {
        auth_type: AuthType::Bearer,
        credentials: AuthCredentials {
            token: Some("test-token-123".to_string()),
            ..Default::default()
        },
    });

    assert_eq!(service.auth.as_ref().unwrap().auth_type, AuthType::Bearer);

    // Test API key auth
    service.auth = Some(ServiceAuthConfig {
        auth_type: AuthType::ApiKey,
        credentials: AuthCredentials {
            api_key: Some("api-key-456".to_string()),
            ..Default::default()
        },
    });

    assert_eq!(service.auth.as_ref().unwrap().auth_type, AuthType::ApiKey);
}

#[tokio::test]
async fn test_service_performance_settings() {
    let mut service = FederatedService::new_sparql(
        "perf-test".to_string(),
        "Performance Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    // Set performance characteristics
    service.performance.average_response_time = Some(Duration::from_millis(150));
    service.performance.max_concurrent_requests = Some(10);
    service.performance.rate_limit = Some(RateLimit {
        requests_per_minute: 1000,
        burst_limit: 50,
    });

    assert_eq!(
        service.performance.average_response_time,
        Some(Duration::from_millis(150))
    );
    assert_eq!(service.performance.max_concurrent_requests, Some(10));
    assert_eq!(
        service
            .performance
            .rate_limit
            .as_ref()
            .unwrap()
            .requests_per_minute,
        1000
    );
}

#[tokio::test]
async fn test_registry_statistics() {
    let mut registry = ServiceRegistry::new();

    // Register multiple services
    for i in 1..=3 {
        let service = FederatedService::new_sparql(
            format!("stats-test-{}", i),
            format!("Stats Test {}", i),
            format!("http://example.com/sparql{}", i),
        );
        registry.register(service).await.unwrap();
    }

    let stats = registry.get_stats().await;
    assert_eq!(stats.total_services, 3);

    // Check capability distribution
    assert!(stats
        .capabilities_distribution
        .contains_key(&ServiceCapability::SparqlQuery));
    assert_eq!(
        *stats
            .capabilities_distribution
            .get(&ServiceCapability::SparqlQuery)
            .unwrap(),
        3
    );
}

#[tokio::test]
async fn test_health_check() {
    let registry = ServiceRegistry::new();

    // Empty registry should be healthy
    let health = registry.health_check().await.unwrap();
    assert_eq!(health.overall_status, ServiceStatus::Healthy);
    assert_eq!(health.total_services, 0);
    assert_eq!(health.healthy_services, 0);
}

#[tokio::test]
async fn test_rate_limiting() {
    let mut registry = ServiceRegistry::new();

    let mut service = FederatedService::new_sparql(
        "rate-test".to_string(),
        "Rate Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    service.performance.rate_limit = Some(RateLimit {
        requests_per_minute: 60,
        burst_limit: 10,
    });

    registry.register(service).await.unwrap();

    // Check rate limit
    assert!(registry.check_rate_limit("rate-test"));

    // Non-existent service should not be rate limited
    assert!(registry.check_rate_limit("non-existent"));
}

#[tokio::test]
async fn test_connection_pool_stats() {
    let registry = ServiceRegistry::new();

    let pool_stats = registry.get_connection_pool_stats().await;
    assert!(pool_stats.is_empty());
}

#[tokio::test]
async fn test_extended_metadata_operations() {
    let mut registry = ServiceRegistry::new();

    let service = FederatedService::new_sparql(
        "metadata-test".to_string(),
        "Metadata Test".to_string(),
        "http://example.com/sparql".to_string(),
    );

    registry.register(service).await.unwrap();

    // Enable extended metadata
    assert!(registry
        .enable_extended_metadata("metadata-test")
        .await
        .is_ok());

    // Should not error if already enabled
    assert!(registry
        .enable_extended_metadata("metadata-test")
        .await
        .is_ok());

    // Should error for non-existent service
    assert!(registry
        .enable_extended_metadata("non-existent")
        .await
        .is_err());

    // Verify extended metadata is present
    let service = registry.get_service("metadata-test").unwrap();
    assert!(service.extended_metadata.is_some());
}
