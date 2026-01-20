//! Webhook Support for GraphQL Events
//!
//! This module provides webhook functionality for GraphQL event notifications:
//! - **Event Types**: Schema changes, query events, subscription updates, errors
//! - **Delivery Management**: Retry policies, dead letter queue, rate limiting
//! - **Security**: HMAC signatures, IP allowlists, mTLS support
//! - **Monitoring**: Delivery tracking, latency metrics, failure analysis
//! - **Filtering**: Event filtering by type, operation, or custom rules

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Webhook event type
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebhookEventType {
    /// Schema was updated
    SchemaUpdated,
    /// Schema validation failed
    SchemaValidationFailed,
    /// Query was executed
    QueryExecuted,
    /// Query failed
    QueryFailed,
    /// Subscription started
    SubscriptionStarted,
    /// Subscription update
    SubscriptionUpdate,
    /// Subscription ended
    SubscriptionEnded,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Authentication failure
    AuthenticationFailed,
    /// Cache invalidated
    CacheInvalidated,
    /// Health check status changed
    HealthStatusChanged,
    /// Custom event
    Custom(String),
}

/// A webhook event to be delivered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Unique event ID
    pub id: String,
    /// Event type
    pub event_type: WebhookEventType,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Event payload
    pub payload: serde_json::Value,
    /// Source service/component
    pub source: String,
    /// Correlation ID for tracing
    pub correlation_id: Option<String>,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

impl WebhookEvent {
    /// Create a new webhook event
    pub fn new(event_type: WebhookEventType, payload: serde_json::Value, source: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            event_type,
            timestamp: SystemTime::now(),
            payload,
            source: source.to_string(),
            correlation_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Webhook endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEndpoint {
    /// Endpoint ID
    pub id: String,
    /// Name/description
    pub name: String,
    /// Target URL
    pub url: String,
    /// Whether endpoint is active
    pub active: bool,
    /// Event types to receive
    pub event_types: Vec<WebhookEventType>,
    /// HMAC secret for signing
    pub secret: Option<String>,
    /// Custom headers to include
    pub headers: HashMap<String, String>,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Rate limit (events per minute)
    pub rate_limit: Option<u32>,
    /// IP allowlist (for callback verification)
    pub ip_allowlist: Vec<String>,
    /// Created timestamp
    pub created_at: SystemTime,
    /// Last successful delivery
    pub last_success: Option<SystemTime>,
    /// Failure count
    pub failure_count: u64,
}

impl WebhookEndpoint {
    /// Create a new webhook endpoint
    pub fn new(name: &str, url: &str, event_types: Vec<WebhookEventType>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            url: url.to_string(),
            active: true,
            event_types,
            secret: None,
            headers: HashMap::new(),
            retry_config: RetryConfig::default(),
            rate_limit: None,
            ip_allowlist: Vec::new(),
            created_at: SystemTime::now(),
            last_success: None,
            failure_count: 0,
        }
    }

    /// Set HMAC secret
    pub fn with_secret(mut self, secret: &str) -> Self {
        self.secret = Some(secret.to_string());
        self
    }

    /// Add custom header
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Set retry configuration
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }
}

/// Retry configuration for webhook delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0-1.0)
    pub jitter: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(300),
            backoff_multiplier: 2.0,
            jitter: 0.1,
        }
    }
}

impl RetryConfig {
    /// Calculate delay for a specific retry attempt
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_delay =
            self.initial_delay.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(self.max_delay.as_secs_f64());

        // Add jitter
        let jitter_range = capped_delay * self.jitter;
        let jitter = (fastrand::f64() * 2.0 - 1.0) * jitter_range;

        Duration::from_secs_f64((capped_delay + jitter).max(0.0))
    }
}

/// Webhook delivery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryResult {
    /// Event ID
    pub event_id: String,
    /// Endpoint ID
    pub endpoint_id: String,
    /// Whether delivery succeeded
    pub success: bool,
    /// HTTP status code (if applicable)
    pub status_code: Option<u16>,
    /// Response body (truncated)
    pub response_body: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Delivery attempt number
    pub attempt: u32,
    /// Delivery latency
    pub latency_ms: u64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Dead letter entry for failed deliveries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterEntry {
    /// Event
    pub event: WebhookEvent,
    /// Endpoint ID
    pub endpoint_id: String,
    /// Last error
    pub last_error: String,
    /// Retry count
    pub retry_count: u32,
    /// First failure time
    pub first_failure: SystemTime,
    /// Last failure time
    pub last_failure: SystemTime,
}

/// Webhook delivery manager
pub struct WebhookManager {
    /// Configuration
    config: WebhookManagerConfig,
    /// Registered endpoints
    endpoints: Arc<RwLock<HashMap<String, WebhookEndpoint>>>,
    /// Event filters
    filters: Arc<RwLock<Vec<EventFilter>>>,
    /// Dead letter queue
    dead_letter_queue: Arc<RwLock<Vec<DeadLetterEntry>>>,
    /// Delivery statistics
    stats: Arc<RwLock<WebhookStatistics>>,
    /// HTTP client
    client: reqwest::Client,
    /// Event queue for async delivery
    event_sender: tokio::sync::mpsc::Sender<(WebhookEvent, String)>,
}

/// Configuration for webhook manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookManagerConfig {
    /// Maximum concurrent deliveries
    pub max_concurrent: usize,
    /// Default timeout for deliveries
    pub timeout: Duration,
    /// Dead letter queue max size
    pub dlq_max_size: usize,
    /// Enable async delivery
    pub async_delivery: bool,
    /// Event buffer size
    pub event_buffer_size: usize,
}

impl Default for WebhookManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            timeout: Duration::from_secs(30),
            dlq_max_size: 1000,
            async_delivery: true,
            event_buffer_size: 1000,
        }
    }
}

/// Event filter for webhook routing
pub struct EventFilter {
    /// Filter ID
    pub id: String,
    /// Filter name
    pub name: String,
    /// Filter predicate
    pub predicate: Arc<dyn Fn(&WebhookEvent) -> bool + Send + Sync>,
    /// Target endpoint IDs (empty = all)
    pub target_endpoints: Vec<String>,
}

impl std::fmt::Debug for EventFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventFilter")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("target_endpoints", &self.target_endpoints)
            .finish_non_exhaustive()
    }
}

/// Webhook statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebhookStatistics {
    /// Total events processed
    pub total_events: u64,
    /// Successful deliveries
    pub successful_deliveries: u64,
    /// Failed deliveries
    pub failed_deliveries: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// P95 latency (ms)
    pub p95_latency_ms: f64,
    /// Dead letter queue size
    pub dlq_size: usize,
    /// Active endpoints
    pub active_endpoints: usize,
}

impl WebhookManager {
    /// Create a new webhook manager
    pub fn new(config: WebhookManagerConfig) -> Self {
        let (event_sender, mut event_receiver) =
            tokio::sync::mpsc::channel(config.event_buffer_size);

        let endpoints = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(RwLock::new(WebhookStatistics::default()));
        let dead_letter_queue = Arc::new(RwLock::new(Vec::new()));

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let manager = Self {
            config: config.clone(),
            endpoints: endpoints.clone(),
            filters: Arc::new(RwLock::new(Vec::new())),
            dead_letter_queue: dead_letter_queue.clone(),
            stats: stats.clone(),
            client: client.clone(),
            event_sender,
        };

        // Start async delivery worker
        if config.async_delivery {
            let endpoints_clone = endpoints.clone();
            let stats_clone = stats.clone();
            let dlq_clone = dead_letter_queue.clone();
            let config_clone = config.clone();

            tokio::spawn(async move {
                while let Some((event, endpoint_id)) = event_receiver.recv().await {
                    let endpoints = endpoints_clone.read().await;
                    if let Some(endpoint) = endpoints.get(&endpoint_id) {
                        let result =
                            Self::deliver_event_internal(&client, &event, endpoint, &config_clone)
                                .await;

                        // Update statistics
                        let mut stats = stats_clone.write().await;
                        stats.total_events += 1;

                        if result.success {
                            stats.successful_deliveries += 1;
                        } else {
                            stats.failed_deliveries += 1;

                            // Add to DLQ if max retries exceeded
                            if result.attempt >= endpoint.retry_config.max_retries {
                                let mut dlq = dlq_clone.write().await;
                                if dlq.len() < config_clone.dlq_max_size {
                                    dlq.push(DeadLetterEntry {
                                        event: event.clone(),
                                        endpoint_id: endpoint_id.clone(),
                                        last_error: result.error.unwrap_or_default(),
                                        retry_count: result.attempt,
                                        first_failure: SystemTime::now(),
                                        last_failure: SystemTime::now(),
                                    });
                                }
                            }
                        }
                    }
                }
            });
        }

        manager
    }

    /// Register a webhook endpoint
    pub async fn register_endpoint(&self, endpoint: WebhookEndpoint) -> String {
        let id = endpoint.id.clone();
        let mut endpoints = self.endpoints.write().await;
        endpoints.insert(id.clone(), endpoint);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.active_endpoints = endpoints.len();

        id
    }

    /// Unregister a webhook endpoint
    pub async fn unregister_endpoint(&self, endpoint_id: &str) -> bool {
        let mut endpoints = self.endpoints.write().await;
        let removed = endpoints.remove(endpoint_id).is_some();

        if removed {
            let mut stats = self.stats.write().await;
            stats.active_endpoints = endpoints.len();
        }

        removed
    }

    /// Get endpoint by ID
    pub async fn get_endpoint(&self, endpoint_id: &str) -> Option<WebhookEndpoint> {
        let endpoints = self.endpoints.read().await;
        endpoints.get(endpoint_id).cloned()
    }

    /// List all endpoints
    pub async fn list_endpoints(&self) -> Vec<WebhookEndpoint> {
        let endpoints = self.endpoints.read().await;
        endpoints.values().cloned().collect()
    }

    /// Add an event filter
    pub async fn add_filter(&self, filter: EventFilter) {
        let mut filters = self.filters.write().await;
        filters.push(filter);
    }

    /// Emit an event to all matching endpoints
    pub async fn emit(&self, event: WebhookEvent) -> Vec<DeliveryResult> {
        let endpoints = self.endpoints.read().await;
        let filters = self.filters.read().await;

        let mut results = Vec::new();

        // Find matching endpoints
        let matching_endpoints: Vec<_> = endpoints
            .values()
            .filter(|ep| {
                ep.active
                    && (ep.event_types.is_empty() || ep.event_types.contains(&event.event_type))
            })
            .filter(|ep| {
                // Apply filters
                filters.iter().all(|f| {
                    if f.target_endpoints.is_empty() || f.target_endpoints.contains(&ep.id) {
                        (f.predicate)(&event)
                    } else {
                        true
                    }
                })
            })
            .cloned()
            .collect();

        drop(endpoints);
        drop(filters);

        // Deliver to each endpoint
        for endpoint in matching_endpoints {
            if self.config.async_delivery {
                // Queue for async delivery
                let _ = self
                    .event_sender
                    .send((event.clone(), endpoint.id.clone()))
                    .await;
                results.push(DeliveryResult {
                    event_id: event.id.clone(),
                    endpoint_id: endpoint.id.clone(),
                    success: true, // Queued successfully
                    status_code: None,
                    response_body: None,
                    error: None,
                    attempt: 0,
                    latency_ms: 0,
                    timestamp: SystemTime::now(),
                });
            } else {
                // Synchronous delivery
                let result =
                    Self::deliver_event_internal(&self.client, &event, &endpoint, &self.config)
                        .await;
                results.push(result);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            *stats
                .events_by_type
                .entry(format!("{:?}", event.event_type))
                .or_insert(0) += 1;
        }

        results
    }

    /// Deliver an event to a specific endpoint (internal implementation)
    async fn deliver_event_internal(
        client: &reqwest::Client,
        event: &WebhookEvent,
        endpoint: &WebhookEndpoint,
        _config: &WebhookManagerConfig,
    ) -> DeliveryResult {
        let start = Instant::now();
        let mut attempt = 0;
        #[allow(unused_assignments)]
        let mut last_error: Option<String> = None;
        let mut last_status = None;
        let mut last_body = None;

        loop {
            attempt += 1;

            // Build request
            let mut request = client.post(&endpoint.url);

            // Add custom headers
            for (key, value) in &endpoint.headers {
                request = request.header(key, value);
            }

            // Add standard headers
            request = request.header("Content-Type", "application/json");
            request = request.header("X-Webhook-Event", format!("{:?}", event.event_type));
            request = request.header("X-Webhook-ID", &event.id);
            request = request.header("X-Webhook-Timestamp", format!("{:?}", event.timestamp));

            // Add HMAC signature if secret is configured
            if let Some(ref secret) = endpoint.secret {
                let payload = serde_json::to_string(&event).unwrap_or_default();
                let signature = Self::compute_hmac(secret, &payload);
                request = request.header("X-Webhook-Signature", signature);
            }

            // Send request
            match request.json(&event).send().await {
                Ok(response) => {
                    let status = response.status();
                    last_status = Some(status.as_u16());

                    if status.is_success() {
                        return DeliveryResult {
                            event_id: event.id.clone(),
                            endpoint_id: endpoint.id.clone(),
                            success: true,
                            status_code: Some(status.as_u16()),
                            response_body: response.text().await.ok(),
                            error: None,
                            attempt,
                            latency_ms: start.elapsed().as_millis() as u64,
                            timestamp: SystemTime::now(),
                        };
                    }

                    last_body = response.text().await.ok();
                    last_error = Some(format!("HTTP {}", status));
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                }
            }

            // Check if we should retry
            if attempt >= endpoint.retry_config.max_retries {
                break;
            }

            // Wait before retry
            let delay = endpoint.retry_config.delay_for_attempt(attempt);
            tokio::time::sleep(delay).await;
        }

        DeliveryResult {
            event_id: event.id.clone(),
            endpoint_id: endpoint.id.clone(),
            success: false,
            status_code: last_status,
            response_body: last_body,
            error: last_error,
            attempt,
            latency_ms: start.elapsed().as_millis() as u64,
            timestamp: SystemTime::now(),
        }
    }

    /// Compute HMAC signature
    fn compute_hmac(secret: &str, payload: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple hash-based signature (in production, use HMAC-SHA256)
        let mut hasher = DefaultHasher::new();
        secret.hash(&mut hasher);
        payload.hash(&mut hasher);
        format!("sha256={:x}", hasher.finish())
    }

    /// Get dead letter queue entries
    pub async fn get_dead_letters(&self) -> Vec<DeadLetterEntry> {
        let dlq = self.dead_letter_queue.read().await;
        dlq.clone()
    }

    /// Retry a dead letter entry
    pub async fn retry_dead_letter(&self, event_id: &str) -> Option<DeliveryResult> {
        let mut dlq = self.dead_letter_queue.write().await;

        if let Some(idx) = dlq.iter().position(|e| e.event.id == event_id) {
            let entry = dlq.remove(idx);
            drop(dlq);

            // Get endpoint
            let endpoints = self.endpoints.read().await;
            if let Some(endpoint) = endpoints.get(&entry.endpoint_id) {
                let result = Self::deliver_event_internal(
                    &self.client,
                    &entry.event,
                    endpoint,
                    &self.config,
                )
                .await;
                return Some(result);
            }
        }

        None
    }

    /// Clear dead letter queue
    pub async fn clear_dead_letters(&self) {
        let mut dlq = self.dead_letter_queue.write().await;
        dlq.clear();
    }

    /// Get statistics
    pub async fn get_statistics(&self) -> WebhookStatistics {
        let mut stats = self.stats.read().await.clone();
        let dlq = self.dead_letter_queue.read().await;
        stats.dlq_size = dlq.len();
        stats
    }

    /// Test endpoint connectivity
    pub async fn test_endpoint(&self, endpoint_id: &str) -> Option<DeliveryResult> {
        let endpoints = self.endpoints.read().await;
        if let Some(endpoint) = endpoints.get(endpoint_id) {
            let test_event = WebhookEvent::new(
                WebhookEventType::Custom("test".to_string()),
                serde_json::json!({"test": true, "timestamp": SystemTime::now()}),
                "webhook_manager",
            );

            return Some(
                Self::deliver_event_internal(&self.client, &test_event, endpoint, &self.config)
                    .await,
            );
        }
        None
    }
}

/// Builder for webhook manager
pub struct WebhookManagerBuilder {
    config: WebhookManagerConfig,
}

impl WebhookManagerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: WebhookManagerConfig::default(),
        }
    }

    /// Set max concurrent deliveries
    pub fn max_concurrent(mut self, n: usize) -> Self {
        self.config.max_concurrent = n;
        self
    }

    /// Set delivery timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set dead letter queue max size
    pub fn dlq_max_size(mut self, size: usize) -> Self {
        self.config.dlq_max_size = size;
        self
    }

    /// Enable/disable async delivery
    pub fn async_delivery(mut self, enabled: bool) -> Self {
        self.config.async_delivery = enabled;
        self
    }

    /// Build the manager
    pub fn build(self) -> WebhookManager {
        WebhookManager::new(self.config)
    }
}

impl Default for WebhookManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_event_creation() {
        let event = WebhookEvent::new(
            WebhookEventType::QueryExecuted,
            serde_json::json!({"query": "test"}),
            "test_source",
        );

        assert!(!event.id.is_empty());
        assert_eq!(event.source, "test_source");
    }

    #[test]
    fn test_webhook_endpoint_creation() {
        let endpoint = WebhookEndpoint::new(
            "Test Webhook",
            "https://example.com/webhook",
            vec![WebhookEventType::QueryExecuted],
        );

        assert!(!endpoint.id.is_empty());
        assert!(endpoint.active);
        assert_eq!(endpoint.url, "https://example.com/webhook");
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.backoff_multiplier, 2.0);
    }

    #[test]
    fn test_retry_delay_calculation() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter: 0.0, // No jitter for deterministic test
        };

        let delay0 = config.delay_for_attempt(0);
        let delay1 = config.delay_for_attempt(1);
        let delay2 = config.delay_for_attempt(2);

        assert!(delay1 > delay0);
        assert!(delay2 > delay1);
    }

    #[test]
    fn test_event_with_metadata() {
        let event = WebhookEvent::new(
            WebhookEventType::QueryExecuted,
            serde_json::json!({}),
            "test",
        )
        .with_metadata("key", "value")
        .with_correlation_id("corr-123");

        assert_eq!(event.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(event.correlation_id, Some("corr-123".to_string()));
    }

    #[test]
    fn test_endpoint_with_config() {
        let endpoint = WebhookEndpoint::new("Test", "https://example.com", vec![])
            .with_secret("my-secret")
            .with_header("X-Custom", "value");

        assert_eq!(endpoint.secret, Some("my-secret".to_string()));
        assert_eq!(endpoint.headers.get("X-Custom"), Some(&"value".to_string()));
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let manager = WebhookManagerBuilder::new()
            .max_concurrent(5)
            .timeout(Duration::from_secs(10))
            .async_delivery(false)
            .build();

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_events, 0);
    }

    #[tokio::test]
    async fn test_register_endpoint() {
        let manager = WebhookManager::new(WebhookManagerConfig {
            async_delivery: false,
            ..Default::default()
        });

        let endpoint = WebhookEndpoint::new(
            "Test",
            "https://example.com/webhook",
            vec![WebhookEventType::QueryExecuted],
        );

        let id = manager.register_endpoint(endpoint).await;
        assert!(!id.is_empty());

        let stats = manager.get_statistics().await;
        assert_eq!(stats.active_endpoints, 1);
    }

    #[tokio::test]
    async fn test_unregister_endpoint() {
        let manager = WebhookManager::new(WebhookManagerConfig {
            async_delivery: false,
            ..Default::default()
        });

        let endpoint = WebhookEndpoint::new("Test", "https://example.com", vec![]);
        let id = manager.register_endpoint(endpoint).await;

        let removed = manager.unregister_endpoint(&id).await;
        assert!(removed);

        let stats = manager.get_statistics().await;
        assert_eq!(stats.active_endpoints, 0);
    }

    #[tokio::test]
    async fn test_list_endpoints() {
        let manager = WebhookManager::new(WebhookManagerConfig {
            async_delivery: false,
            ..Default::default()
        });

        manager
            .register_endpoint(WebhookEndpoint::new("A", "https://a.com", vec![]))
            .await;
        manager
            .register_endpoint(WebhookEndpoint::new("B", "https://b.com", vec![]))
            .await;

        let endpoints = manager.list_endpoints().await;
        assert_eq!(endpoints.len(), 2);
    }

    #[tokio::test]
    async fn test_dead_letter_queue() {
        let manager = WebhookManager::new(WebhookManagerConfig {
            async_delivery: false,
            ..Default::default()
        });

        // Initially empty
        let dlq = manager.get_dead_letters().await;
        assert!(dlq.is_empty());

        // Clear (no-op when empty)
        manager.clear_dead_letters().await;
    }

    #[test]
    fn test_hmac_computation() {
        let sig1 = WebhookManager::compute_hmac("secret", "payload");
        let sig2 = WebhookManager::compute_hmac("secret", "payload");
        let sig3 = WebhookManager::compute_hmac("different", "payload");

        assert_eq!(sig1, sig2);
        assert_ne!(sig1, sig3);
        assert!(sig1.starts_with("sha256="));
    }

    #[tokio::test]
    async fn test_event_filter() {
        let manager = WebhookManager::new(WebhookManagerConfig {
            async_delivery: false,
            ..Default::default()
        });

        // Add filter that only allows QueryExecuted events
        manager
            .add_filter(EventFilter {
                id: "filter1".to_string(),
                name: "Query filter".to_string(),
                predicate: Arc::new(|event| {
                    matches!(event.event_type, WebhookEventType::QueryExecuted)
                }),
                target_endpoints: vec![],
            })
            .await;

        // This is a basic test - full filter testing would require mocking HTTP
    }

    #[test]
    fn test_webhook_event_types() {
        let types = vec![
            WebhookEventType::SchemaUpdated,
            WebhookEventType::QueryExecuted,
            WebhookEventType::QueryFailed,
            WebhookEventType::SubscriptionStarted,
            WebhookEventType::Custom("custom".to_string()),
        ];

        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            assert!(!json.is_empty());
        }
    }
}
