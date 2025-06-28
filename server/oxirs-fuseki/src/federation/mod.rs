//! Advanced federation support for distributed SPARQL queries
//!
//! This module provides enhanced SERVICE delegation capabilities including:
//! - Remote endpoint discovery and registration
//! - Query cost estimation and optimization
//! - Parallel service execution
//! - Health monitoring and circuit breaking
//! - Cross-service query planning

pub mod discovery;
pub mod executor;
pub mod health;
pub mod planner;

use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use url::Url;

use crate::error::FusekiResult;

/// Federation configuration
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Enable automatic service discovery
    pub enable_discovery: bool,
    /// Discovery refresh interval
    pub discovery_interval: Duration,
    /// Maximum concurrent service requests
    pub max_concurrent_requests: usize,
    /// Request timeout for remote services
    pub request_timeout: Duration,
    /// Enable query cost estimation
    pub enable_cost_estimation: bool,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            enable_discovery: true,
            discovery_interval: Duration::from_secs(300), // 5 minutes
            max_concurrent_requests: 10,
            request_timeout: Duration::from_secs(30),
            enable_cost_estimation: true,
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

/// Circuit breaker configuration for service resilience
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit
    pub failure_threshold: u32,
    /// Success threshold to close circuit
    pub success_threshold: u32,
    /// Timeout before attempting to close circuit
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
        }
    }
}

/// Remote SPARQL service endpoint
#[derive(Debug, Clone)]
pub struct ServiceEndpoint {
    /// Service URL
    pub url: Url,
    /// Service metadata
    pub metadata: ServiceMetadata,
    /// Health status
    pub health: ServiceHealth,
    /// Query capabilities
    pub capabilities: ServiceCapabilities,
}

/// Service metadata for discovery and routing
#[derive(Debug, Clone, Default)]
pub struct ServiceMetadata {
    /// Service name
    pub name: String,
    /// Service description
    pub description: Option<String>,
    /// Service tags for categorization
    pub tags: Vec<String>,
    /// Geographic location (for proximity-based routing)
    pub location: Option<String>,
    /// Service version
    pub version: Option<String>,
}

/// Service health status
#[derive(Debug, Clone)]
pub enum ServiceHealth {
    /// Service is healthy and accepting requests
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy and circuit is open
    Unhealthy,
    /// Health status unknown
    Unknown,
}

/// Service capabilities for query planning
#[derive(Debug, Clone, Default)]
pub struct ServiceCapabilities {
    /// Supported SPARQL features
    pub sparql_features: Vec<String>,
    /// Estimated dataset size
    pub dataset_size: Option<u64>,
    /// Average query response time
    pub avg_response_time: Option<Duration>,
    /// Maximum result size
    pub max_result_size: Option<usize>,
    /// Supported result formats
    pub result_formats: Vec<String>,
}

/// Alias for compatibility with query planner
pub type EndpointCapabilities = ServiceCapabilities;

/// Federation manager for coordinating distributed queries
pub struct FederationManager {
    /// Configuration
    config: FederationConfig,
    /// Registered service endpoints
    endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    /// Service discovery component
    discovery: Arc<discovery::ServiceDiscovery>,
    /// Query planner
    planner: Arc<planner::QueryPlanner>,
    /// Health monitor
    health_monitor: Arc<health::HealthMonitor>,
}

impl FederationManager {
    /// Create a new federation manager
    pub fn new(config: FederationConfig) -> Self {
        let endpoints = Arc::new(RwLock::new(HashMap::new()));
        
        Self {
            discovery: Arc::new(discovery::ServiceDiscovery::new(
                config.clone(),
                endpoints.clone(),
            )),
            planner: Arc::new(planner::QueryPlanner::new(
                config.clone(),
                endpoints.clone(),
            )),
            health_monitor: Arc::new(health::HealthMonitor::new(
                config.clone(),
                endpoints.clone(),
            )),
            config,
            endpoints,
        }
    }

    /// Start federation services
    pub async fn start(&self) -> Result<()> {
        // Start service discovery
        if self.config.enable_discovery {
            self.discovery.start().await?;
        }
        
        // Start health monitoring
        self.health_monitor.start().await?;
        
        Ok(())
    }

    /// Stop federation services
    pub async fn stop(&self) -> Result<()> {
        self.discovery.stop().await?;
        self.health_monitor.stop().await?;
        Ok(())
    }

    /// Register a service endpoint manually
    pub async fn register_endpoint(&self, id: String, endpoint: ServiceEndpoint) -> Result<()> {
        let mut endpoints = self.endpoints.write().await;
        endpoints.insert(id, endpoint);
        Ok(())
    }

    /// Get all healthy endpoints
    pub async fn get_healthy_endpoints(&self) -> Vec<(String, ServiceEndpoint)> {
        let endpoints = self.endpoints.read().await;
        endpoints
            .iter()
            .filter(|(_, ep)| matches!(ep.health, ServiceHealth::Healthy))
            .map(|(id, ep)| (id.clone(), ep.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FederationConfig::default();
        assert!(config.enable_discovery);
        assert_eq!(config.max_concurrent_requests, 10);
    }
}