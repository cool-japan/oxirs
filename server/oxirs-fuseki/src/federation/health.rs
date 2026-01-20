//! Health monitoring for federated SPARQL endpoints

use reqwest::Client;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{Mutex, Notify, RwLock},
    time::interval,
};
use url::Url;

use crate::{
    error::FusekiResult,
    federation::{CircuitBreakerConfig, FederationConfig, ServiceEndpoint, ServiceHealth},
};

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Endpoint URL
    pub url: Url,
    /// Check timestamp
    pub timestamp: Instant,
    /// Response time
    pub response_time: Option<Duration>,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Circuit breaker state for a service
#[derive(Debug, Clone)]
struct CircuitBreaker {
    /// Current state
    state: CircuitState,
    /// Consecutive failures
    failure_count: u32,
    /// Consecutive successes
    success_count: u32,
    /// Last state change
    last_state_change: Instant,
    /// Configuration
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    /// Circuit is closed, requests allowed
    Closed,
    /// Circuit is open, requests blocked
    Open,
    /// Circuit is half-open, testing recovery
    HalfOpen,
}

impl CircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_state_change: Instant::now(),
            config,
        }
    }

    fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                    self.last_state_change = Instant::now();
                    tracing::info!("Circuit breaker closed after recovery");
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but reset if it does
                self.state = CircuitState::HalfOpen;
                self.success_count = 1;
                self.last_state_change = Instant::now();
            }
        }
    }

    fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.config.failure_threshold {
                    self.state = CircuitState::Open;
                    self.last_state_change = Instant::now();
                    tracing::warn!(
                        "Circuit breaker opened after {} failures",
                        self.failure_count
                    );
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.failure_count = self.config.failure_threshold;
                self.success_count = 0;
                self.last_state_change = Instant::now();
                tracing::warn!("Circuit breaker reopened after failure in half-open state");
            }
            CircuitState::Open => {
                // Already open, just update failure count
                self.failure_count += 1;
            }
        }
    }

    fn should_allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                if self.last_state_change.elapsed() >= self.config.timeout {
                    self.state = CircuitState::HalfOpen;
                    self.success_count = 0;
                    self.last_state_change = Instant::now();
                    tracing::info!("Circuit breaker entering half-open state");
                    true
                } else {
                    false
                }
            }
        }
    }

    fn get_health_status(&self) -> ServiceHealth {
        match self.state {
            CircuitState::Closed => ServiceHealth::Healthy,
            CircuitState::HalfOpen => ServiceHealth::Degraded,
            CircuitState::Open => ServiceHealth::Unhealthy,
        }
    }
}

/// Health monitor for service endpoints
pub struct HealthMonitor {
    config: FederationConfig,
    endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    circuit_breakers: Arc<Mutex<HashMap<String, CircuitBreaker>>>,
    http_client: Client,
    shutdown: Arc<Notify>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(
        config: FederationConfig,
        endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    ) -> Self {
        Self {
            config,
            endpoints,
            circuit_breakers: Arc::new(Mutex::new(HashMap::new())),
            http_client: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Start health monitoring
    pub async fn start(&self) -> FusekiResult<()> {
        let shutdown = self.shutdown.clone();
        let endpoints = self.endpoints.clone();
        let circuit_breakers = self.circuit_breakers.clone();
        let client = self.http_client.clone();
        let circuit_config = self.config.circuit_breaker.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::check_all_endpoints(&endpoints, &circuit_breakers, &client, &circuit_config).await;
                    }
                    _ = shutdown.notified() => {
                        tracing::info!("Health monitor shutting down");
                        break;
                    }
                }
            }
        });

        // Run initial health checks
        Self::check_all_endpoints(
            &self.endpoints,
            &self.circuit_breakers,
            &self.http_client,
            &self.config.circuit_breaker,
        )
        .await;

        Ok(())
    }

    /// Stop health monitoring
    pub async fn stop(&self) -> FusekiResult<()> {
        self.shutdown.notify_one();
        Ok(())
    }

    /// Check health of all endpoints
    async fn check_all_endpoints(
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        circuit_breakers: &Arc<Mutex<HashMap<String, CircuitBreaker>>>,
        client: &Client,
        circuit_config: &CircuitBreakerConfig,
    ) {
        let eps = endpoints.read().await.clone();

        for (id, endpoint) in eps {
            let result = Self::check_endpoint(&endpoint.url, client).await;

            // Update circuit breaker
            let mut breakers = circuit_breakers.lock().await;
            let breaker = breakers
                .entry(id.clone())
                .or_insert_with(|| CircuitBreaker::new(circuit_config.clone()));

            if result.success {
                breaker.record_success();
            } else {
                breaker.record_failure();
            }

            let health_status = breaker.get_health_status();
            drop(breakers);

            // Update endpoint health
            let mut eps = endpoints.write().await;
            if let Some(ep) = eps.get_mut(&id) {
                ep.health = health_status;

                // Update average response time if successful
                if let Some(response_time) = result.response_time {
                    if let Some(avg) = &mut ep.capabilities.avg_response_time {
                        // Simple moving average
                        *avg = (*avg + response_time) / 2;
                    } else {
                        ep.capabilities.avg_response_time = Some(response_time);
                    }
                }
            }
        }
    }

    /// Check a single endpoint's health
    async fn check_endpoint(url: &Url, client: &Client) -> HealthCheckResult {
        let start = Instant::now();

        // Simple ASK query for health check
        let query = "ASK { ?s ?p ?o } LIMIT 1";

        let result = client
            .get(url.as_str())
            .query(&[("query", query)])
            .header("Accept", "application/sparql-results+json")
            .send()
            .await;

        let response_time = start.elapsed();

        match result {
            Ok(response) => {
                let success = response.status().is_success();
                let error = if !success {
                    Some(format!("HTTP {}", response.status()))
                } else {
                    None
                };

                HealthCheckResult {
                    url: url.clone(),
                    timestamp: Instant::now(),
                    response_time: Some(response_time),
                    success,
                    error,
                }
            }
            Err(e) => HealthCheckResult {
                url: url.clone(),
                timestamp: Instant::now(),
                response_time: None,
                success: false,
                error: Some(e.to_string()),
            },
        }
    }

    /// Check if a service should be used based on circuit breaker
    pub async fn should_use_service(&self, service_id: &str) -> bool {
        let mut breakers = self.circuit_breakers.lock().await;

        if let Some(breaker) = breakers.get_mut(service_id) {
            breaker.should_allow_request()
        } else {
            // No circuit breaker yet, allow request
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(1),
        };

        let mut breaker = CircuitBreaker::new(config);

        // Initially closed
        assert_eq!(breaker.state, CircuitState::Closed);
        assert!(breaker.should_allow_request());

        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state, CircuitState::Closed);

        // Third failure opens circuit
        breaker.record_failure();
        assert_eq!(breaker.state, CircuitState::Open);
        assert!(!breaker.should_allow_request());

        // Wait for timeout
        std::thread::sleep(Duration::from_secs(1));

        // Should enter half-open state
        assert!(breaker.should_allow_request());
        assert_eq!(breaker.state, CircuitState::HalfOpen);

        // Success in half-open
        breaker.record_success();
        assert_eq!(breaker.state, CircuitState::HalfOpen);

        // Second success closes circuit
        breaker.record_success();
        assert_eq!(breaker.state, CircuitState::Closed);
    }
}
