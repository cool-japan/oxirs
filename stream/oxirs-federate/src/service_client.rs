//! Service Clients for SPARQL and GraphQL endpoints
//!
//! This module provides robust HTTP clients for communicating with federated
//! SPARQL and GraphQL services, including authentication, retry logic, and error handling.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use backoff::{backoff::Backoff, future::retry, ExponentialBackoff};
use bytes::Bytes;
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT},
    Client, Response, StatusCode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tower::limit::ConcurrencyLimitLayer;
use tower::ServiceBuilder;
use tracing::{debug, error, info, warn};

use crate::{
    executor::{GraphQLResponse, SparqlResults},
    auth::{AuthConfig, AuthCredentials}, FederatedService,
    service::AuthType,
};

/// Trait for service clients
#[async_trait]
pub trait ServiceClient: Send + Sync {
    /// Execute a query against the service
    async fn execute_query(&self, query: &str) -> Result<QueryResponse>;

    /// Execute an update against the service (SPARQL only)
    async fn execute_update(&self, update: &str) -> Result<()>;

    /// Check if the service is healthy
    async fn health_check(&self) -> Result<bool>;

    /// Get client statistics
    async fn get_stats(&self) -> ClientStats;
}

/// OAuth2 token information with expiration tracking
#[derive(Debug, Clone)]
struct OAuth2TokenInfo {
    access_token: String,
    token_type: String,
    expires_at: Instant,
    scope: Option<String>,
}

/// OAuth2 token response from authorization server
#[derive(Debug, Deserialize)]
struct OAuth2TokenResponse {
    access_token: String,
    token_type: String,
    expires_in: Option<u64>,
    scope: Option<String>,
}

/// SPARQL service client
pub struct SparqlClient {
    service: FederatedService,
    http_client: Client,
    config: ClientConfig,
    connection_pool: Arc<ConnectionPool>,
    rate_limiter: Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    stats: Arc<RwLock<ClientStats>>,
    oauth2_token: Arc<RwLock<Option<OAuth2TokenInfo>>>,
}

impl SparqlClient {
    /// Create a new SPARQL client
    pub fn new(service: FederatedService, config: ClientConfig) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .pool_max_idle_per_host(config.max_idle_connections)
            .pool_idle_timeout(config.idle_timeout)
            .build()?;

        let connection_pool = Arc::new(ConnectionPool::new(config.max_connections));

        let rate_limiter = if let Some(rate_limit) = &service.performance.rate_limit {
            let quota = Quota::per_minute(
                std::num::NonZeroU32::new(rate_limit.requests_per_minute as u32)
                    .unwrap_or(std::num::NonZeroU32::new(60).unwrap()),
            );
            Some(Arc::new(RateLimiter::direct(quota)))
        } else {
            None
        };

        let circuit_breaker = Arc::new(RwLock::new(CircuitBreaker::new(
            config.circuit_breaker_threshold,
            config.circuit_breaker_timeout,
        )));

        Ok(Self {
            service,
            http_client,
            config,
            connection_pool,
            rate_limiter,
            circuit_breaker,
            stats: Arc::new(RwLock::new(ClientStats::default())),
            oauth2_token: Arc::new(RwLock::new(None)),
        })
    }

    /// Execute a SPARQL query
    async fn execute_sparql_query(&self, query: &str) -> Result<SparqlResults> {
        let start_time = Instant::now();

        // Check circuit breaker
        {
            let breaker = self.circuit_breaker.read().await;
            if breaker.is_open() {
                self.record_error("circuit_breaker_open").await;
                return Err(anyhow!(
                    "Circuit breaker is open for service {}",
                    self.service.id
                ));
            }
        }

        // Check rate limit
        if let Some(limiter) = &self.rate_limiter {
            if limiter.check().is_err() {
                self.record_error("rate_limit_exceeded").await;
                return Err(anyhow!(
                    "Rate limit exceeded for service {}",
                    self.service.id
                ));
            }
        }

        // Acquire connection permit
        let _permit = self.connection_pool.acquire().await?;

        // Build request with OAuth2 token refresh if needed
        let mut headers = if let Some(auth) = &self.service.auth {
            if auth.auth_type == AuthType::OAuth2 {
                // Refresh OAuth2 token if needed before building headers
                if let Ok(Some(fresh_token)) = self.ensure_fresh_oauth2_token(auth).await {
                    let mut headers = self.build_headers(false)?;
                    let auth_value = format!("Bearer {}", fresh_token);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                    headers
                } else {
                    self.build_headers(true)?
                }
            } else {
                self.build_headers(true)?
            }
        } else {
            self.build_headers(true)?
        };
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );

        // Execute with retry
        let backoff = ExponentialBackoff {
            max_elapsed_time: Some(self.config.max_retry_duration),
            ..ExponentialBackoff::default()
        };

        let result = retry(backoff, || async {
            let response = self
                .http_client
                .post(&self.service.endpoint)
                .headers(headers.clone())
                .body(query.to_string())
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        backoff::Error::transient(anyhow::Error::from(e))
                    } else {
                        backoff::Error::permanent(anyhow::Error::from(e))
                    }
                })?;

            if response.status().is_server_error() {
                return Err(backoff::Error::transient(anyhow!(
                    "Server error: {}",
                    response.status()
                )));
            }

            if !response.status().is_success() {
                return Err(backoff::Error::permanent(anyhow!(
                    "Query failed: {}",
                    response.status()
                )));
            }

            Ok(response
                .json::<SparqlResults>()
                .await
                .map_err(|e| backoff::Error::permanent(anyhow::Error::from(e)))?)
        })
        .await;

        let duration = start_time.elapsed();

        match result {
            Ok(results) => {
                self.record_success(duration).await;
                self.circuit_breaker.write().await.record_success();
                Ok(results)
            }
            Err(e) => {
                self.record_error("query_failed").await;
                self.circuit_breaker.write().await.record_failure();
                Err(anyhow!("Query execution failed: {}", e))
            }
        }
    }

    /// Execute a SPARQL update
    async fn execute_sparql_update(&self, update: &str) -> Result<()> {
        let start_time = Instant::now();

        // Similar checks as query
        {
            let breaker = self.circuit_breaker.read().await;
            if breaker.is_open() {
                return Err(anyhow!("Circuit breaker is open"));
            }
        }

        if let Some(limiter) = &self.rate_limiter {
            if limiter.check().is_err() {
                return Err(anyhow!("Rate limit exceeded"));
            }
        }

        let _permit = self.connection_pool.acquire().await?;

        let mut headers = self.build_headers(true)?;
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-update"),
        );

        let response = self
            .http_client
            .post(&self.service.endpoint)
            .headers(headers)
            .body(update.to_string())
            .send()
            .await?;

        let duration = start_time.elapsed();

        if response.status().is_success() {
            self.record_success(duration).await;
            self.circuit_breaker.write().await.record_success();
            Ok(())
        } else {
            self.record_error("update_failed").await;
            self.circuit_breaker.write().await.record_failure();
            Err(anyhow!("Update failed: {}", response.status()))
        }
    }

    /// Build request headers with authentication
    fn build_headers(&self, include_auth: bool) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_str(&self.config.user_agent)?);

        if include_auth {
            if let Some(auth) = &self.service.auth {
                self.add_auth_header(&mut headers, auth)?;
            }
        }

        Ok(headers)
    }

    /// Add authentication header
    fn add_auth_header(&self, headers: &mut HeaderMap, auth: &AuthConfig) -> Result<()> {
        use base64::encode;

        match &auth.auth_type {
            AuthType::Basic => {
                if let (Some(username), Some(password)) =
                    (&auth.credentials.username, &auth.credentials.password)
                {
                    let credentials = format!("{}:{}", username, password);
                    let encoded = encode(credentials.as_bytes());
                    let auth_value = format!("Basic {}", encoded);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::Bearer => {
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {}", token);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::ApiKey => {
                if let Some(api_key) = &auth.credentials.api_key {
                    headers.insert("X-API-Key", HeaderValue::from_str(api_key)?);
                }
            }
            AuthType::OAuth2 => {
                // For OAuth2, we need to check token freshness in the query methods
                // For now, use the existing token if available
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {}", token);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::Custom => {
                if let Some(custom_headers) = &auth.credentials.custom_headers {
                    for (key, value) in custom_headers {
                        let header_name: HeaderName = key.parse()?;
                        headers.insert(header_name, HeaderValue::from_str(value)?);
                    }
                }
            }
            AuthType::None => {}
        }
        Ok(())
    }

    /// Record successful request
    async fn record_success(&self, duration: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.successful_requests += 1;
        stats.total_response_time += duration;
        stats.last_request_time = Some(Instant::now());
    }

    /// Record failed request
    async fn record_error(&self, error_type: &str) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.failed_requests += 1;
        *stats
            .error_counts
            .entry(error_type.to_string())
            .or_insert(0) += 1;
        stats.last_error_time = Some(Instant::now());
    }

    /// Ensure OAuth2 token is fresh and return the token if available
    async fn ensure_fresh_oauth2_token(&self, auth: &AuthConfig) -> Result<Option<String>> {
        let current_token = {
            let token_guard = self.oauth2_token.read().await;
            token_guard.clone()
        };

        // Check if we have a valid token that hasn't expired (with 60s buffer)
        if let Some(ref token_info) = current_token {
            if token_info.expires_at > Instant::now() + Duration::from_secs(60) {
                return Ok(Some(token_info.access_token.clone()));
            }
        }

        // Token is expired or missing, need to refresh
        if let Some(refresh_token) = &auth.credentials.refresh_token {
            match self.refresh_oauth2_token(auth, refresh_token).await {
                Ok(new_token) => {
                    let mut token_guard = self.oauth2_token.write().await;
                    *token_guard = Some(new_token.clone());
                    Ok(Some(new_token.access_token))
                }
                Err(e) => {
                    warn!("Failed to refresh OAuth2 token: {}", e);
                    // Fall back to existing token if available
                    if let Some(ref token_info) = current_token {
                        Ok(Some(token_info.access_token.clone()))
                    } else {
                        Err(e)
                    }
                }
            }
        } else {
            // No refresh token available, use existing token if any
            if let Some(token_info) = current_token {
                Ok(Some(token_info.access_token))
            } else {
                Err(anyhow!("No OAuth2 token or refresh token available"))
            }
        }
    }

    /// Refresh OAuth2 token using refresh token
    async fn refresh_oauth2_token(
        &self,
        auth: &AuthConfig,
        refresh_token: &str,
    ) -> Result<OAuth2TokenInfo> {
        let token_endpoint = auth
            .credentials
            .token_endpoint
            .as_ref()
            .ok_or_else(|| anyhow!("OAuth2 token endpoint not configured"))?;

        let client_id = auth
            .credentials
            .client_id
            .as_ref()
            .ok_or_else(|| anyhow!("OAuth2 client ID not configured"))?;

        let client_secret = auth
            .credentials
            .client_secret
            .as_ref()
            .ok_or_else(|| anyhow!("OAuth2 client secret not configured"))?;

        // Prepare request body for token refresh
        let mut form_data = HashMap::new();
        form_data.insert("grant_type", "refresh_token");
        form_data.insert("refresh_token", refresh_token);
        form_data.insert("client_id", client_id);
        form_data.insert("client_secret", client_secret);

        // Add scope if configured
        if let Some(scope) = &auth.credentials.scope {
            form_data.insert("scope", scope);
        }

        debug!("Refreshing OAuth2 token at endpoint: {}", token_endpoint);

        let response = self
            .http_client
            .post(token_endpoint)
            .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
            .form(&form_data)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "OAuth2 token refresh failed with status {}: {}",
                status,
                error_text
            ));
        }

        let token_response: OAuth2TokenResponse = response.json().await?;

        // Calculate expiration time (default to 1 hour if not provided)
        let expires_in = token_response.expires_in.unwrap_or(3600);
        let expires_at = Instant::now() + Duration::from_secs(expires_in);

        info!(
            "Successfully refreshed OAuth2 token, expires in {} seconds",
            expires_in
        );

        Ok(OAuth2TokenInfo {
            access_token: token_response.access_token,
            token_type: token_response.token_type,
            expires_at,
            scope: token_response.scope,
        })
    }
}

#[async_trait]
impl ServiceClient for SparqlClient {
    async fn execute_query(&self, query: &str) -> Result<QueryResponse> {
        let results = self.execute_sparql_query(query).await?;
        Ok(QueryResponse::Sparql(results))
    }

    async fn execute_update(&self, update: &str) -> Result<()> {
        self.execute_sparql_update(update).await
    }

    async fn health_check(&self) -> Result<bool> {
        let query = "ASK { ?s ?p ?o }";
        match self.execute_sparql_query(query).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn get_stats(&self) -> ClientStats {
        self.stats.read().await.clone()
    }
}

/// GraphQL service client
pub struct GraphQLClient {
    service: FederatedService,
    http_client: Client,
    config: ClientConfig,
    connection_pool: Arc<ConnectionPool>,
    rate_limiter: Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    stats: Arc<RwLock<ClientStats>>,
}

impl GraphQLClient {
    /// Create a new GraphQL client
    pub fn new(service: FederatedService, config: ClientConfig) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .pool_max_idle_per_host(config.max_idle_connections)
            .pool_idle_timeout(config.idle_timeout)
            .build()?;

        let connection_pool = Arc::new(ConnectionPool::new(config.max_connections));

        let rate_limiter = if let Some(rate_limit) = &service.performance.rate_limit {
            let quota = Quota::per_minute(
                std::num::NonZeroU32::new(rate_limit.requests_per_minute as u32)
                    .unwrap_or(std::num::NonZeroU32::new(60).unwrap()),
            );
            Some(Arc::new(RateLimiter::direct(quota)))
        } else {
            None
        };

        let circuit_breaker = Arc::new(RwLock::new(CircuitBreaker::new(
            config.circuit_breaker_threshold,
            config.circuit_breaker_timeout,
        )));

        Ok(Self {
            service,
            http_client,
            config,
            connection_pool,
            rate_limiter,
            circuit_breaker,
            stats: Arc::new(RwLock::new(ClientStats::default())),
        })
    }

    /// Execute a GraphQL query
    async fn execute_graphql_query(
        &self,
        query: &str,
        variables: Option<serde_json::Value>,
    ) -> Result<GraphQLResponse> {
        let start_time = Instant::now();

        // Check circuit breaker and rate limit (similar to SPARQL)
        {
            let breaker = self.circuit_breaker.read().await;
            if breaker.is_open() {
                return Err(anyhow!("Circuit breaker is open"));
            }
        }

        if let Some(limiter) = &self.rate_limiter {
            if limiter.check().is_err() {
                return Err(anyhow!("Rate limit exceeded"));
            }
        }

        let _permit = self.connection_pool.acquire().await?;

        let mut headers = self.build_headers()?;
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let request_body = GraphQLRequest {
            query: query.to_string(),
            variables,
            operation_name: None,
        };

        let backoff = ExponentialBackoff {
            max_elapsed_time: Some(self.config.max_retry_duration),
            ..ExponentialBackoff::default()
        };

        let result = retry(backoff, || async {
            let response = self
                .http_client
                .post(&self.service.endpoint)
                .headers(headers.clone())
                .json(&request_body)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        backoff::Error::transient(anyhow::Error::from(e))
                    } else {
                        backoff::Error::permanent(anyhow::Error::from(e))
                    }
                })?;

            if response.status().is_server_error() {
                return Err(backoff::Error::transient(anyhow!(
                    "Server error: {}",
                    response.status()
                )));
            }

            if !response.status().is_success() {
                return Err(backoff::Error::permanent(anyhow!(
                    "Query failed: {}",
                    response.status()
                )));
            }

            response
                .json::<GraphQLResponse>()
                .await
                .map_err(|e| backoff::Error::permanent(anyhow::Error::from(e)))
        })
        .await;

        let duration = start_time.elapsed();

        match result {
            Ok(response) => {
                self.record_success(duration).await;
                self.circuit_breaker.write().await.record_success();
                Ok(response)
            }
            Err(e) => {
                self.record_error("query_failed").await;
                self.circuit_breaker.write().await.record_failure();
                Err(anyhow!("GraphQL query failed: {}", e))
            }
        }
    }

    /// Build request headers
    fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_str(&self.config.user_agent)?);

        if let Some(auth) = &self.service.auth {
            self.add_auth_header(&mut headers, auth)?;
        }

        Ok(headers)
    }

    /// Add authentication header (same as SPARQL client)
    fn add_auth_header(&self, headers: &mut HeaderMap, auth: &AuthConfig) -> Result<()> {
        use base64::encode;

        match &auth.auth_type {
            AuthType::Basic => {
                if let (Some(username), Some(password)) =
                    (&auth.credentials.username, &auth.credentials.password)
                {
                    let credentials = format!("{}:{}", username, password);
                    let encoded = encode(credentials.as_bytes());
                    let auth_value = format!("Basic {}", encoded);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::Bearer => {
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {}", token);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::ApiKey => {
                if let Some(api_key) = &auth.credentials.api_key {
                    headers.insert("X-API-Key", HeaderValue::from_str(api_key)?);
                }
            }
            AuthType::OAuth2 => {
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {}", token);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::Custom => {
                if let Some(custom_headers) = &auth.credentials.custom_headers {
                    for (key, value) in custom_headers {
                        let header_name: HeaderName = key.parse()?;
                        headers.insert(header_name, HeaderValue::from_str(value)?);
                    }
                }
            }
            AuthType::None => {}
        }
        Ok(())
    }

    /// Record successful request
    async fn record_success(&self, duration: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.successful_requests += 1;
        stats.total_response_time += duration;
        stats.last_request_time = Some(Instant::now());
    }

    /// Record failed request
    async fn record_error(&self, error_type: &str) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.failed_requests += 1;
        *stats
            .error_counts
            .entry(error_type.to_string())
            .or_insert(0) += 1;
        stats.last_error_time = Some(Instant::now());
    }
}

#[async_trait]
impl ServiceClient for GraphQLClient {
    async fn execute_query(&self, query: &str) -> Result<QueryResponse> {
        let response = self.execute_graphql_query(query, None).await?;
        Ok(QueryResponse::GraphQL(response))
    }

    async fn execute_update(&self, _update: &str) -> Result<()> {
        Err(anyhow!("Updates not supported for GraphQL"))
    }

    async fn health_check(&self) -> Result<bool> {
        let query = "{ __schema { queryType { name } } }";
        match self.execute_graphql_query(query, None).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn get_stats(&self) -> ClientStats {
        self.stats.read().await.clone()
    }
}

/// Configuration for service clients
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// User agent string
    pub user_agent: String,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum number of connections
    pub max_connections: usize,
    /// Maximum idle connections per host
    pub max_idle_connections: usize,
    /// Idle connection timeout
    pub idle_timeout: Duration,
    /// Maximum retry duration
    pub max_retry_duration: Duration,
    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: usize,
    /// Circuit breaker timeout
    pub circuit_breaker_timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            user_agent: "oxirs-federate-client/1.0".to_string(),
            request_timeout: Duration::from_secs(30),
            max_connections: 50,
            max_idle_connections: 10,
            idle_timeout: Duration::from_secs(90),
            max_retry_duration: Duration::from_secs(60),
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
        }
    }
}

/// Query response types
#[derive(Debug, Clone)]
pub enum QueryResponse {
    Sparql(SparqlResults),
    GraphQL(GraphQLResponse),
}

/// GraphQL request structure
#[derive(Debug, Serialize)]
pub struct GraphQLRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
}

/// Connection pool for managing concurrent connections
struct ConnectionPool {
    semaphore: Semaphore,
}

impl ConnectionPool {
    fn new(max_connections: usize) -> Self {
        Self {
            semaphore: Semaphore::new(max_connections),
        }
    }

    async fn acquire(&self) -> Result<tokio::sync::SemaphorePermit> {
        self.semaphore
            .acquire()
            .await
            .map_err(|_| anyhow!("Failed to acquire connection permit"))
    }
}

/// Circuit breaker for handling service failures
#[derive(Debug)]
struct CircuitBreaker {
    failure_count: usize,
    failure_threshold: usize,
    last_failure_time: Option<Instant>,
    timeout: Duration,
    state: CircuitBreakerState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitBreaker {
    fn new(failure_threshold: usize, timeout: Duration) -> Self {
        Self {
            failure_count: 0,
            failure_threshold,
            last_failure_time: None,
            timeout,
            state: CircuitBreakerState::Closed,
        }
    }

    fn is_open(&self) -> bool {
        match self.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > self.timeout {
                        // Transition to half-open
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::HalfOpen => {
                // Transition back to closed
                self.state = CircuitBreakerState::Closed;
                self.failure_count = 0;
                self.last_failure_time = None;
            }
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            }
            _ => {}
        }
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitBreakerState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    // Transition to open
                    self.state = CircuitBreakerState::Open;
                    warn!(
                        "Circuit breaker opened after {} failures",
                        self.failure_count
                    );
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Transition back to open
                self.state = CircuitBreakerState::Open;
                warn!("Circuit breaker reopened after failure in half-open state");
            }
            _ => {}
        }
    }
}

/// Client statistics
#[derive(Debug, Clone, Default)]
pub struct ClientStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_response_time: Duration,
    pub error_counts: HashMap<String, u64>,
    pub last_request_time: Option<Instant>,
    pub last_error_time: Option<Instant>,
}

impl ClientStats {
    /// Calculate average response time
    pub fn avg_response_time(&self) -> Duration {
        if self.successful_requests > 0 {
            self.total_response_time / self.successful_requests as u32
        } else {
            Duration::from_secs(0)
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.successful_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

/// Create a service client based on service type
pub fn create_client(
    service: FederatedService,
    config: ClientConfig,
) -> Result<Box<dyn ServiceClient>> {
    match service.service_type {
        crate::ServiceType::Sparql => Ok(Box::new(SparqlClient::new(service, config)?)),
        crate::ServiceType::GraphQL => Ok(Box::new(GraphQLClient::new(service, config)?)),
        crate::ServiceType::Hybrid => {
            // For hybrid services, default to SPARQL client
            // Could be enhanced to select based on query type
            Ok(Box::new(SparqlClient::new(service, config)?))
        }
        crate::ServiceType::RestRdf => {
            // REST-RDF services use SPARQL-like interface
            Ok(Box::new(SparqlClient::new(service, config)?))
        }
        crate::ServiceType::Custom(_) => {
            // For custom services, default to SPARQL client
            Ok(Box::new(SparqlClient::new(service, config)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.max_connections, 50);
        assert_eq!(config.request_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::new(3, Duration::from_secs(60));

        assert!(!breaker.is_open());

        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        breaker.record_failure();

        // Should be open now
        assert!(breaker.is_open());

        // Success in half-open state
        breaker.state = CircuitBreakerState::HalfOpen;
        breaker.record_success();

        // Should be closed now
        assert!(!breaker.is_open());
        assert_eq!(breaker.state, CircuitBreakerState::Closed);
    }

    #[test]
    fn test_client_stats() {
        let mut stats = ClientStats::default();
        stats.total_requests = 100;
        stats.successful_requests = 95;
        stats.failed_requests = 5;

        assert_eq!(stats.success_rate(), 0.95);
    }
}
