//! Unit tests for service clients

use oxirs_federate::*;
use std::time::Duration;

#[tokio::test]
async fn test_client_config() {
    let config = ClientConfig::default();
    
    assert_eq!(config.user_agent, "oxirs-federate-client/1.0");
    assert_eq!(config.request_timeout, Duration::from_secs(30));
    assert_eq!(config.max_connections, 50);
    assert_eq!(config.max_idle_connections, 10);
    assert_eq!(config.circuit_breaker_threshold, 5);
}

#[tokio::test]
async fn test_sparql_client_creation() {
    let service = FederatedService::new_sparql(
        "sparql-client-test".to_string(),
        "SPARQL Client Test".to_string(),
        "http://example.com/sparql".to_string(),
    );
    
    let config = ClientConfig::default();
    let client = create_client(service, config);
    
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_graphql_client_creation() {
    let service = FederatedService::new_graphql(
        "graphql-client-test".to_string(),
        "GraphQL Client Test".to_string(),
        "http://example.com/graphql".to_string(),
    );
    
    let config = ClientConfig::default();
    let client = create_client(service, config);
    
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_client_with_basic_auth() {
    let mut service = FederatedService::new_sparql(
        "auth-client-test".to_string(),
        "Auth Client Test".to_string(),
        "http://example.com/sparql".to_string(),
    );
    
    service.auth = Some(AuthConfig {
        auth_type: AuthType::Basic,
        credentials: AuthCredentials {
            username: Some("user".to_string()),
            password: Some("pass".to_string()),
            token: None,
            api_key: None,
            oauth_config: None,
        },
    });
    
    let config = ClientConfig::default();
    let client = create_client(service, config);
    
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_client_with_bearer_auth() {
    let mut service = FederatedService::new_graphql(
        "bearer-client-test".to_string(),
        "Bearer Client Test".to_string(),
        "http://example.com/graphql".to_string(),
    );
    
    service.auth = Some(AuthConfig {
        auth_type: AuthType::Bearer,
        credentials: AuthCredentials {
            username: None,
            password: None,
            token: Some("test-token-123".to_string()),
            api_key: None,
            oauth_config: None,
        },
    });
    
    let config = ClientConfig::default();
    let client = create_client(service, config);
    
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_client_with_api_key_auth() {
    let mut service = FederatedService::new_sparql(
        "apikey-client-test".to_string(),
        "API Key Client Test".to_string(),
        "http://example.com/sparql".to_string(),
    );
    
    service.auth = Some(AuthConfig {
        auth_type: AuthType::ApiKey,
        credentials: AuthCredentials {
            username: None,
            password: None,
            token: None,
            api_key: Some("api-key-456".to_string()),
            oauth_config: None,
        },
    });
    
    let config = ClientConfig::default();
    let client = create_client(service, config);
    
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_client_stats() {
    let mut stats = ClientStats::default();
    
    // Update stats
    stats.total_requests = 100;
    stats.successful_requests = 95;
    stats.failed_requests = 5;
    stats.total_response_time = Duration::from_secs(10);
    
    // Test calculations
    assert_eq!(stats.success_rate(), 0.95);
    assert_eq!(stats.avg_response_time(), Duration::from_millis(105)); // 10000ms / 95
    
    // Test with no requests
    let empty_stats = ClientStats::default();
    assert_eq!(empty_stats.success_rate(), 0.0);
    assert_eq!(empty_stats.avg_response_time(), Duration::from_secs(0));
}

#[tokio::test]
async fn test_client_with_rate_limiting() {
    let mut service = FederatedService::new_sparql(
        "rate-limited-client".to_string(),
        "Rate Limited Client".to_string(),
        "http://example.com/sparql".to_string(),
    );
    
    service.performance.rate_limit = Some(RateLimit {
        requests_per_minute: 60,
        burst_limit: 10,
    });
    
    let config = ClientConfig::default();
    let client = create_client(service, config);
    
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_hybrid_service_client() {
    let mut service = FederatedService::new_sparql(
        "hybrid-client-test".to_string(),
        "Hybrid Client Test".to_string(),
        "http://example.com/endpoint".to_string(),
    );
    
    service.service_type = ServiceType::Hybrid;
    
    let config = ClientConfig::default();
    let client = create_client(service, config);
    
    // Should create a SPARQL client for hybrid services
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_client_error_tracking() {
    let mut stats = ClientStats::default();
    
    // Track different error types
    stats.error_counts.insert("timeout".to_string(), 5);
    stats.error_counts.insert("network".to_string(), 3);
    stats.error_counts.insert("auth".to_string(), 1);
    
    assert_eq!(*stats.error_counts.get("timeout").unwrap(), 5);
    assert_eq!(*stats.error_counts.get("network").unwrap(), 3);
    assert_eq!(*stats.error_counts.get("auth").unwrap(), 1);
    assert!(stats.error_counts.get("unknown").is_none());
}

#[tokio::test]
async fn test_custom_client_config() {
    let config = ClientConfig {
        user_agent: "custom-agent/2.0".to_string(),
        request_timeout: Duration::from_secs(60),
        max_connections: 100,
        max_idle_connections: 20,
        idle_timeout: Duration::from_secs(120),
        max_retry_duration: Duration::from_secs(90),
        circuit_breaker_threshold: 10,
        circuit_breaker_timeout: Duration::from_secs(120),
    };
    
    assert_eq!(config.user_agent, "custom-agent/2.0");
    assert_eq!(config.request_timeout, Duration::from_secs(60));
    assert_eq!(config.max_connections, 100);
    assert_eq!(config.circuit_breaker_threshold, 10);
}

// Mock tests for client behavior (would require a mock server in real implementation)
#[tokio::test]
async fn test_sparql_client_health_check() {
    let service = FederatedService::new_sparql(
        "health-check-test".to_string(),
        "Health Check Test".to_string(),
        "http://localhost:9999/sparql".to_string(), // Non-existent endpoint
    );
    
    let config = ClientConfig {
        request_timeout: Duration::from_secs(1), // Short timeout for test
        ..ClientConfig::default()
    };
    
    let client = create_client(service, config).unwrap();
    
    // Health check should fail for non-existent service
    let health = client.health_check().await.unwrap_or(true);
    assert!(!health);
}

#[tokio::test]
async fn test_graphql_client_health_check() {
    let service = FederatedService::new_graphql(
        "graphql-health-test".to_string(),
        "GraphQL Health Test".to_string(),
        "http://localhost:9998/graphql".to_string(), // Non-existent endpoint
    );
    
    let config = ClientConfig {
        request_timeout: Duration::from_secs(1), // Short timeout for test
        ..ClientConfig::default()
    };
    
    let client = create_client(service, config).unwrap();
    
    // Health check should fail for non-existent service
    let health = client.health_check().await.unwrap_or(true);
    assert!(!health);
}

#[tokio::test]
async fn test_client_stats_tracking() {
    let service = FederatedService::new_sparql(
        "stats-tracking-test".to_string(),
        "Stats Tracking Test".to_string(),
        "http://localhost:9997/sparql".to_string(),
    );
    
    let config = ClientConfig::default();
    let client = create_client(service, config).unwrap();
    
    // Get initial stats
    let stats = client.get_stats().await;
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.successful_requests, 0);
    assert_eq!(stats.failed_requests, 0);
}