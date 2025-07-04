//! Integration tests for federation functionality

use oxirs_fuseki::federation::{
    discovery::{DiscoveryMethod, ServiceRegistration},
    FederationConfig, FederationManager, ServiceCapabilities, ServiceEndpoint, ServiceHealth,
    ServiceMetadata,
};
use std::time::Duration;
use url::Url;

#[tokio::test]
async fn test_federation_manager_creation() {
    let config = FederationConfig::default();
    let manager = FederationManager::new(config);

    // Start the manager
    manager.start().await.unwrap();

    // Register a test endpoint
    let endpoint = ServiceEndpoint {
        url: Url::parse("http://example.com/sparql").unwrap(),
        metadata: ServiceMetadata {
            name: "Test Endpoint".to_string(),
            description: Some("Test SPARQL endpoint".to_string()),
            tags: vec!["test".to_string()],
            location: Some("US-East".to_string()),
            version: Some("1.0.0".to_string()),
            contact: Some("test@example.com".to_string()),
        },
        health: ServiceHealth::Healthy,
        capabilities: ServiceCapabilities {
            sparql_features: vec!["SPARQL 1.1 Query".to_string()],
            dataset_size: Some(1_000_000),
            avg_response_time: Some(Duration::from_millis(100)),
            max_result_size: Some(10_000),
            result_formats: vec!["application/sparql-results+json".to_string()],
        },
    };

    manager
        .register_endpoint("test-endpoint".to_string(), endpoint)
        .await
        .unwrap();

    // Get healthy endpoints
    let healthy = manager.get_healthy_endpoints().await;
    assert_eq!(healthy.len(), 1);
    assert_eq!(healthy[0].0, "test-endpoint");

    // Stop the manager
    manager.stop().await.unwrap();
}

#[tokio::test]
async fn test_static_service_discovery() {
    let mut config = FederationConfig::default();
    config.enable_discovery = true;

    let manager = FederationManager::new(config);

    // Add static service registration
    let registrations = vec![
        ServiceRegistration {
            id: "service1".to_string(),
            url: Url::parse("http://service1.example.com/sparql").unwrap(),
            metadata: ServiceMetadata {
                name: "Service 1".to_string(),
                ..Default::default()
            },
        },
        ServiceRegistration {
            id: "service2".to_string(),
            url: Url::parse("http://service2.example.com/sparql").unwrap(),
            metadata: ServiceMetadata {
                name: "Service 2".to_string(),
                ..Default::default()
            },
        },
    ];

    // This would normally be done through the discovery module
    // For testing, we'll register them directly
    for reg in registrations {
        let endpoint = ServiceEndpoint {
            url: reg.url,
            metadata: reg.metadata,
            health: ServiceHealth::Unknown,
            capabilities: ServiceCapabilities::default(),
        };
        manager.register_endpoint(reg.id, endpoint).await.unwrap();
    }

    manager.start().await.unwrap();

    // Give discovery time to run
    tokio::time::sleep(Duration::from_millis(100)).await;

    let endpoints = manager.get_healthy_endpoints().await;
    // Health is Unknown, so no healthy endpoints yet
    assert_eq!(endpoints.len(), 0);

    manager.stop().await.unwrap();
}

#[tokio::test]
async fn test_circuit_breaker_config() {
    let mut config = FederationConfig::default();
    config.circuit_breaker.failure_threshold = 3;
    config.circuit_breaker.success_threshold = 2;
    config.circuit_breaker.timeout = Duration::from_secs(30);

    assert_eq!(config.circuit_breaker.failure_threshold, 3);
    assert_eq!(config.circuit_breaker.success_threshold, 2);
    assert_eq!(config.circuit_breaker.timeout, Duration::from_secs(30));
}
