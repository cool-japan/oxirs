//! # Webhook Integration Module
//!
//! This module provides comprehensive webhook support for external system integration:
//! - Webhook registration and management
//! - Event filtering and routing
//! - Retry mechanisms with exponential backoff
//! - Security features (HMAC signing, rate limiting)
//! - Monitoring and diagnostics

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    store_integration::{ChangeNotification, UpdateNotification},
    EventMetadata, StreamEvent,
};

/// Webhook manager for handling HTTP notifications
pub struct WebhookManager {
    /// Registered webhooks
    webhooks: Arc<RwLock<HashMap<String, RegisteredWebhook>>>,
    /// HTTP client
    client: reqwest::Client,
    /// Event queue
    event_queue: Arc<RwLock<VecDeque<WebhookEvent>>>,
    /// Configuration
    config: WebhookConfig,
    /// Statistics
    stats: Arc<RwLock<WebhookStats>>,
    /// Event notifier
    event_notifier: broadcast::Sender<WebhookNotification>,
    /// Rate limiter
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

/// Registered webhook
#[derive(Debug, Clone)]
struct RegisteredWebhook {
    /// Webhook ID
    id: String,
    /// Webhook URL
    url: String,
    /// HTTP method
    method: HttpMethod,
    /// Custom headers
    headers: HashMap<String, String>,
    /// Event filters
    filters: Vec<EventFilter>,
    /// Security configuration
    security: WebhookSecurity,
    /// Retry configuration
    retry_config: RetryConfig,
    /// Rate limit configuration
    rate_limit: RateLimit,
    /// Webhook metadata
    metadata: WebhookMetadata,
    /// Statistics
    stats: WebhookStatistics,
    /// Created timestamp
    created_at: Instant,
    /// Last delivery attempt
    last_delivery: Option<Instant>,
    /// Status
    status: WebhookStatus,
}

/// HTTP methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

/// Event filter for webhook delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    /// Filter by event type
    pub event_types: Option<Vec<String>>,
    /// Filter by graph
    pub graph_filter: Option<String>,
    /// Filter by subject pattern (regex string)
    pub subject_pattern: Option<String>,
    /// Filter by predicate
    pub predicate_filter: Option<String>,
    /// Custom filter expression
    pub custom_filter: Option<String>,
}

/// Webhook security configuration
#[derive(Debug, Clone)]
struct WebhookSecurity {
    /// HMAC secret for signing
    hmac_secret: Option<String>,
    /// Authentication headers
    auth_headers: HashMap<String, String>,
    /// SSL/TLS verification
    verify_ssl: bool,
    /// Allowed response codes
    allowed_response_codes: Vec<u16>,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable jitter
    pub enable_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
    }
}

/// Rate limit configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    /// Requests per second
    pub requests_per_second: f64,
    /// Burst size
    pub burst_size: u32,
    /// Time window
    pub window: Duration,
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            requests_per_second: 10.0,
            burst_size: 20,
            window: Duration::from_secs(1),
        }
    }
}

/// Webhook metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookMetadata {
    /// Webhook name
    pub name: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Owner
    pub owner: Option<String>,
    /// Tags
    pub tags: Vec<String>,
    /// Custom properties
    pub properties: HashMap<String, String>,
}

/// Webhook statistics
#[derive(Debug, Clone, Default)]
struct WebhookStatistics {
    /// Total delivery attempts
    pub total_attempts: u64,
    /// Successful deliveries
    pub successful_deliveries: u64,
    /// Failed deliveries
    pub failed_deliveries: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Last success timestamp
    pub last_success: Option<Instant>,
    /// Last failure timestamp
    pub last_failure: Option<Instant>,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

/// Webhook status
#[derive(Debug, Clone, PartialEq)]
enum WebhookStatus {
    /// Webhook is active
    Active,
    /// Webhook is paused
    Paused,
    /// Webhook is disabled due to failures
    Disabled { reason: String },
    /// Webhook is being deleted
    Deleting,
}

/// Webhook event to be delivered
#[derive(Debug, Clone)]
struct WebhookEvent {
    /// Event ID
    id: String,
    /// Target webhook ID
    webhook_id: String,
    /// Event payload
    payload: WebhookPayload,
    /// Delivery attempts
    attempts: u32,
    /// Created timestamp
    created_at: Instant,
    /// Next retry time
    next_retry: Option<Instant>,
}

/// Webhook payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    /// Event ID
    pub event_id: String,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Event type
    pub event_type: String,
    /// Event data
    pub data: serde_json::Value,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Webhook manager configuration
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Maximum registered webhooks
    pub max_webhooks: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Worker thread count
    pub worker_threads: usize,
    /// Delivery timeout
    pub delivery_timeout: Duration,
    /// Queue processing interval
    pub queue_interval: Duration,
    /// Enable automatic retry
    pub enable_retry: bool,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Enable HMAC signing
    pub enable_hmac: bool,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            max_webhooks: 1000,
            max_queue_size: 10000,
            worker_threads: 4,
            delivery_timeout: Duration::from_secs(30),
            queue_interval: Duration::from_millis(100),
            enable_retry: true,
            enable_rate_limiting: true,
            enable_hmac: true,
        }
    }
}

/// Webhook manager statistics
#[derive(Debug, Clone, Default)]
pub struct WebhookStats {
    /// Total registered webhooks
    pub total_webhooks: usize,
    /// Active webhooks
    pub active_webhooks: usize,
    /// Total events queued
    pub events_queued: u64,
    /// Events delivered
    pub events_delivered: u64,
    /// Events failed
    pub events_failed: u64,
    /// Queue size
    pub queue_size: usize,
    /// Average delivery time
    pub avg_delivery_time: Duration,
    /// Rate limit hits
    pub rate_limit_hits: u64,
}

/// Webhook notification events
#[derive(Debug, Clone)]
pub enum WebhookNotification {
    /// Webhook registered
    WebhookRegistered { id: String, url: String },
    /// Webhook delivery succeeded
    DeliverySucceeded {
        webhook_id: String,
        event_id: String,
        duration: Duration,
    },
    /// Webhook delivery failed
    DeliveryFailed {
        webhook_id: String,
        event_id: String,
        error: String,
        attempts: u32,
    },
    /// Webhook disabled
    WebhookDisabled { id: String, reason: String },
    /// Rate limit exceeded
    RateLimitExceeded { webhook_id: String },
}

/// Rate limiter for webhook deliveries
struct RateLimiter {
    /// Rate limits per webhook
    limits: HashMap<String, TokenBucket>,
    /// Global rate limit
    global_limit: TokenBucket,
}

/// Token bucket for rate limiting
#[derive(Debug, Clone)]
struct TokenBucket {
    /// Available tokens
    tokens: f64,
    /// Token capacity
    capacity: f64,
    /// Refill rate (tokens per second)
    refill_rate: f64,
    /// Last refill time
    last_refill: Instant,
}

impl WebhookManager {
    /// Create a new webhook manager
    pub async fn new(config: WebhookConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.delivery_timeout)
            .build()?;

        let (tx, _) = broadcast::channel(1000);

        Ok(Self {
            webhooks: Arc::new(RwLock::new(HashMap::new())),
            client,
            event_queue: Arc::new(RwLock::new(VecDeque::new())),
            config,
            stats: Arc::new(RwLock::new(WebhookStats::default())),
            event_notifier: tx,
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new())),
        })
    }

    /// Start the webhook manager
    pub async fn start(&self) -> Result<()> {
        // Start queue processors
        for i in 0..self.config.worker_threads {
            self.start_queue_processor(i).await;
        }

        // Start rate limiter refill
        self.start_rate_limiter_refill().await;

        info!(
            "Webhook manager started with {} workers",
            self.config.worker_threads
        );
        Ok(())
    }

    /// Register a webhook
    pub async fn register_webhook(
        &self,
        url: String,
        method: HttpMethod,
        headers: HashMap<String, String>,
        filters: Vec<EventFilter>,
        security: WebhookSecurity,
        retry_config: RetryConfig,
        rate_limit: RateLimit,
        metadata: WebhookMetadata,
    ) -> Result<String> {
        // Check limits
        let webhooks = self.webhooks.read().await;
        if webhooks.len() >= self.config.max_webhooks {
            return Err(anyhow!("Maximum webhook limit reached"));
        }
        drop(webhooks);

        // Validate URL
        let parsed_url = reqwest::Url::parse(&url).map_err(|_| anyhow!("Invalid webhook URL"))?;

        if !parsed_url.scheme().starts_with("http") {
            return Err(anyhow!("Webhook URL must use HTTP or HTTPS"));
        }

        // Generate webhook ID
        let webhook_id = Uuid::new_v4().to_string();

        // Create registered webhook
        let webhook = RegisteredWebhook {
            id: webhook_id.clone(),
            url: url.clone(),
            method,
            headers,
            filters,
            security,
            retry_config,
            rate_limit: rate_limit.clone(),
            metadata,
            stats: WebhookStatistics::default(),
            created_at: Instant::now(),
            last_delivery: None,
            status: WebhookStatus::Active,
        };

        // Register webhook
        self.webhooks
            .write()
            .await
            .insert(webhook_id.clone(), webhook);

        // Initialize rate limiter for this webhook
        self.rate_limiter
            .write()
            .await
            .add_webhook(&webhook_id, rate_limit);

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_webhooks += 1;
        stats.active_webhooks = self.webhooks.read().await.len();
        drop(stats);

        // Notify
        let _ = self
            .event_notifier
            .send(WebhookNotification::WebhookRegistered {
                id: webhook_id.clone(),
                url,
            });

        info!("Registered webhook: {}", webhook_id);
        Ok(webhook_id)
    }

    /// Unregister a webhook
    pub async fn unregister_webhook(&self, webhook_id: &str) -> Result<()> {
        let mut webhooks = self.webhooks.write().await;
        webhooks
            .remove(webhook_id)
            .ok_or_else(|| anyhow!("Webhook not found"))?;

        // Remove from rate limiter
        self.rate_limiter.write().await.remove_webhook(webhook_id);

        // Update statistics
        self.stats.write().await.active_webhooks = webhooks.len();

        info!("Unregistered webhook: {}", webhook_id);
        Ok(())
    }

    /// Send event to webhooks
    pub async fn send_event(&self, event: StreamEvent) -> Result<()> {
        let webhooks = self.webhooks.read().await;
        let mut matching_webhooks = Vec::new();

        // Find matching webhooks
        for webhook in webhooks.values() {
            if webhook.status == WebhookStatus::Active
                && self.matches_filters(&event, &webhook.filters)
            {
                matching_webhooks.push(webhook.id.clone());
            }
        }
        drop(webhooks);

        if matching_webhooks.is_empty() {
            return Ok(());
        }

        // Create webhook payload
        let payload = self.create_payload(&event)?;

        // Queue events for delivery
        let mut queue = self.event_queue.write().await;
        for webhook_id in matching_webhooks {
            if queue.len() >= self.config.max_queue_size {
                warn!("Webhook queue full, dropping event");
                break;
            }

            let webhook_event = WebhookEvent {
                id: Uuid::new_v4().to_string(),
                webhook_id,
                payload: payload.clone(),
                attempts: 0,
                created_at: Instant::now(),
                next_retry: None,
            };

            queue.push_back(webhook_event);
            self.stats.write().await.events_queued += 1;
        }

        Ok(())
    }

    /// Check if event matches webhook filters
    fn matches_filters(&self, event: &StreamEvent, filters: &[EventFilter]) -> bool {
        if filters.is_empty() {
            return true;
        }

        filters.iter().any(|filter| {
            // Check event type filter
            if let Some(event_types) = &filter.event_types {
                let event_type = match event {
                    StreamEvent::TripleAdded { .. } => "triple_added",
                    StreamEvent::TripleRemoved { .. } => "triple_removed",
                    StreamEvent::GraphCreated { .. } => "graph_created",
                    StreamEvent::GraphDeleted { .. } => "graph_deleted",
                    StreamEvent::GraphCleared { .. } => "graph_cleared",
                    _ => "unknown",
                };

                if !event_types.contains(&event_type.to_string()) {
                    return false;
                }
            }

            // Check graph filter
            if let Some(graph_filter) = &filter.graph_filter {
                let event_graph = match event {
                    StreamEvent::TripleAdded { graph, .. }
                    | StreamEvent::TripleRemoved { graph, .. } => graph.as_ref(),
                    StreamEvent::GraphCreated { graph, .. }
                    | StreamEvent::GraphDeleted { graph, .. } => Some(graph),
                    StreamEvent::GraphCleared { graph, .. } => graph.as_ref(),
                    _ => None,
                };

                if event_graph != Some(graph_filter) {
                    return false;
                }
            }

            // Check subject pattern (using simple contains for now)
            if let Some(pattern) = &filter.subject_pattern {
                let subject = match event {
                    StreamEvent::TripleAdded { subject, .. }
                    | StreamEvent::TripleRemoved { subject, .. } => Some(subject),
                    _ => None,
                };

                if let Some(subj) = subject {
                    if !subj.contains(pattern) {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Check predicate filter
            if let Some(pred_filter) = &filter.predicate_filter {
                let predicate = match event {
                    StreamEvent::TripleAdded { predicate, .. }
                    | StreamEvent::TripleRemoved { predicate, .. } => Some(predicate),
                    _ => None,
                };

                if predicate != Some(pred_filter) {
                    return false;
                }
            }

            true
        })
    }

    /// Create webhook payload from stream event
    fn create_payload(&self, event: &StreamEvent) -> Result<WebhookPayload> {
        let (event_type, data) = match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "triple_added",
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph,
                    "metadata": metadata
                }),
            ),
            StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                "triple_removed",
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph,
                    "metadata": metadata
                }),
            ),
            StreamEvent::GraphCreated { graph, metadata } => (
                "graph_created",
                serde_json::json!({
                    "graph": graph,
                    "metadata": metadata
                }),
            ),
            StreamEvent::GraphDeleted { graph, metadata } => (
                "graph_deleted",
                serde_json::json!({
                    "graph": graph,
                    "metadata": metadata
                }),
            ),
            StreamEvent::GraphCleared { graph, metadata } => (
                "graph_cleared",
                serde_json::json!({
                    "graph": graph,
                    "metadata": metadata
                }),
            ),
            _ => return Err(anyhow!("Unsupported event type for webhook")),
        };

        Ok(WebhookPayload {
            event_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            data,
            metadata: HashMap::new(),
        })
    }

    /// Start queue processor
    async fn start_queue_processor(&self, worker_id: usize) {
        let queue = self.event_queue.clone();
        let webhooks = self.webhooks.clone();
        let client = self.client.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let event_notifier = self.event_notifier.clone();
        let rate_limiter = self.rate_limiter.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.queue_interval);

            loop {
                interval.tick().await;

                // Get next event from queue
                let event = {
                    let mut queue_guard = queue.write().await;
                    queue_guard.pop_front()
                };

                if let Some(mut event) = event {
                    // Check if it's time to retry
                    if let Some(next_retry) = event.next_retry {
                        if Instant::now() < next_retry {
                            // Put back in queue
                            queue.write().await.push_back(event);
                            continue;
                        }
                    }

                    // Get webhook details
                    let webhook = {
                        let webhooks_guard = webhooks.read().await;
                        webhooks_guard.get(&event.webhook_id).cloned()
                    };

                    if let Some(webhook) = webhook {
                        // Check rate limit
                        if config.enable_rate_limiting {
                            let allowed = rate_limiter.write().await.check_rate_limit(&webhook.id);
                            if !allowed {
                                // Put back in queue and skip
                                queue.write().await.push_back(event);
                                stats.write().await.rate_limit_hits += 1;
                                let _ =
                                    event_notifier.send(WebhookNotification::RateLimitExceeded {
                                        webhook_id: webhook.id.clone(),
                                    });
                                continue;
                            }
                        }

                        // Attempt delivery
                        event.attempts += 1;
                        let start_time = Instant::now();

                        match Self::deliver_webhook(&client, &webhook, &event.payload, &config)
                            .await
                        {
                            Ok(duration) => {
                                // Success
                                Self::update_webhook_stats(&webhooks, &webhook.id, true, duration)
                                    .await;
                                stats.write().await.events_delivered += 1;

                                let _ =
                                    event_notifier.send(WebhookNotification::DeliverySucceeded {
                                        webhook_id: webhook.id.clone(),
                                        event_id: event.id.clone(),
                                        duration,
                                    });

                                debug!(
                                    "Webhook delivery succeeded: {} -> {}",
                                    event.id, webhook.id
                                );
                            }
                            Err(e) => {
                                // Failure
                                let duration = start_time.elapsed();
                                Self::update_webhook_stats(&webhooks, &webhook.id, false, duration)
                                    .await;

                                error!(
                                    "Webhook delivery failed: {} -> {}: {}",
                                    event.id, webhook.id, e
                                );

                                // Check if we should retry
                                if config.enable_retry
                                    && event.attempts < webhook.retry_config.max_attempts
                                {
                                    // Calculate next retry time
                                    let delay = Self::calculate_retry_delay(
                                        &webhook.retry_config,
                                        event.attempts,
                                    );
                                    event.next_retry = Some(Instant::now() + delay);

                                    // Put back in queue
                                    queue.write().await.push_back(event.clone());

                                    debug!("Scheduling retry for {} in {:?}", event.id, delay);
                                } else {
                                    // Max retries reached
                                    stats.write().await.events_failed += 1;

                                    let _ =
                                        event_notifier.send(WebhookNotification::DeliveryFailed {
                                            webhook_id: webhook.id.clone(),
                                            event_id: event.id.clone(),
                                            error: e.to_string(),
                                            attempts: event.attempts,
                                        });

                                    // Check if webhook should be disabled
                                    Self::check_webhook_health(
                                        &webhooks,
                                        &webhook.id,
                                        &event_notifier,
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                }

                // Update queue size stat
                stats.write().await.queue_size = queue.read().await.len();
            }
        });

        debug!("Started webhook queue processor {}", worker_id);
    }

    /// Deliver webhook
    async fn deliver_webhook(
        client: &reqwest::Client,
        webhook: &RegisteredWebhook,
        payload: &WebhookPayload,
        config: &WebhookConfig,
    ) -> Result<Duration> {
        let start_time = Instant::now();

        // Prepare request
        let mut request = match webhook.method {
            HttpMethod::Get => client.get(&webhook.url),
            HttpMethod::Post => client.post(&webhook.url),
            HttpMethod::Put => client.put(&webhook.url),
            HttpMethod::Patch => client.patch(&webhook.url),
            HttpMethod::Delete => client.delete(&webhook.url),
        };

        // Add headers
        for (key, value) in &webhook.headers {
            request = request.header(key, value);
        }

        // Add security headers
        for (key, value) in &webhook.security.auth_headers {
            request = request.header(key, value);
        }

        // Add HMAC signature if enabled
        if config.enable_hmac {
            if let Some(secret) = &webhook.security.hmac_secret {
                let signature = Self::calculate_hmac_signature(payload, secret)?;
                request = request.header("X-Webhook-Signature", signature);
            }
        }

        // Add timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        request = request.header("X-Webhook-Timestamp", timestamp.to_string());

        // Set JSON body for non-GET requests
        if webhook.method != HttpMethod::Get {
            request = request.json(payload);
        }

        // Send request
        let response = request
            .send()
            .await
            .map_err(|e| anyhow!("Request failed: {}", e))?;

        let status = response.status();

        // Check if response is acceptable
        if webhook.security.allowed_response_codes.is_empty() {
            if !status.is_success() {
                return Err(anyhow!(
                    "HTTP {}: {}",
                    status.as_u16(),
                    response.text().await.unwrap_or_default()
                ));
            }
        } else {
            if !webhook
                .security
                .allowed_response_codes
                .contains(&status.as_u16())
            {
                return Err(anyhow!("Unexpected response code: {}", status.as_u16()));
            }
        }

        Ok(start_time.elapsed())
    }

    /// Calculate HMAC signature
    fn calculate_hmac_signature(payload: &WebhookPayload, secret: &str) -> Result<String> {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        type HmacSha256 = Hmac<Sha256>;

        let payload_json = serde_json::to_string(payload)?;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|_| anyhow!("Invalid HMAC key"))?;
        mac.update(payload_json.as_bytes());

        let result = mac.finalize();
        Ok(format!("sha256={}", hex::encode(result.into_bytes())))
    }

    /// Calculate retry delay
    fn calculate_retry_delay(config: &RetryConfig, attempt: u32) -> Duration {
        let base_delay = config.initial_delay.as_millis() as f64;
        let multiplier = config.backoff_multiplier.powi(attempt as i32 - 1);
        let delay_ms = (base_delay * multiplier) as u64;

        let delay = Duration::from_millis(delay_ms.min(config.max_delay.as_millis() as u64));

        if config.enable_jitter {
            // Add random jitter (±10%)
            let jitter = (delay.as_millis() as f64 * 0.1 * (rand::random::<f64>() - 0.5)) as u64;
            delay + Duration::from_millis(jitter)
        } else {
            delay
        }
    }

    /// Update webhook statistics
    async fn update_webhook_stats(
        webhooks: &Arc<RwLock<HashMap<String, RegisteredWebhook>>>,
        webhook_id: &str,
        success: bool,
        duration: Duration,
    ) {
        let mut webhooks_guard = webhooks.write().await;
        if let Some(webhook) = webhooks_guard.get_mut(webhook_id) {
            webhook.stats.total_attempts += 1;
            webhook.last_delivery = Some(Instant::now());

            if success {
                webhook.stats.successful_deliveries += 1;
                webhook.stats.last_success = Some(Instant::now());
                webhook.stats.consecutive_failures = 0;

                // Update average response time
                let count = webhook.stats.successful_deliveries;
                webhook.stats.avg_response_time = Duration::from_millis(
                    (webhook.stats.avg_response_time.as_millis() as u64 * (count - 1)
                        + duration.as_millis() as u64)
                        / count,
                );
            } else {
                webhook.stats.failed_deliveries += 1;
                webhook.stats.last_failure = Some(Instant::now());
                webhook.stats.consecutive_failures += 1;
            }
        }
    }

    /// Check webhook health and disable if necessary
    async fn check_webhook_health(
        webhooks: &Arc<RwLock<HashMap<String, RegisteredWebhook>>>,
        webhook_id: &str,
        event_notifier: &broadcast::Sender<WebhookNotification>,
    ) {
        let mut webhooks_guard = webhooks.write().await;
        if let Some(webhook) = webhooks_guard.get_mut(webhook_id) {
            // Disable webhook if too many consecutive failures
            if webhook.stats.consecutive_failures >= 10 {
                let reason = format!(
                    "Too many consecutive failures: {}",
                    webhook.stats.consecutive_failures
                );
                webhook.status = WebhookStatus::Disabled {
                    reason: reason.clone(),
                };

                let _ = event_notifier.send(WebhookNotification::WebhookDisabled {
                    id: webhook_id.to_string(),
                    reason,
                });

                warn!(
                    "Disabled webhook {} due to consecutive failures",
                    webhook_id
                );
            }
        }
    }

    /// Start rate limiter refill
    async fn start_rate_limiter_refill(&self) {
        let rate_limiter = self.rate_limiter.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));

            loop {
                interval.tick().await;
                rate_limiter.write().await.refill_tokens();
            }
        });
    }

    /// Get webhook statistics
    pub async fn get_webhook_stats(&self, webhook_id: &str) -> Result<WebhookStatistics> {
        let webhooks = self.webhooks.read().await;
        let webhook = webhooks
            .get(webhook_id)
            .ok_or_else(|| anyhow!("Webhook not found"))?;

        Ok(webhook.stats.clone())
    }

    /// Get manager statistics
    pub async fn get_stats(&self) -> WebhookStats {
        self.stats.read().await.clone()
    }

    /// List all webhooks
    pub async fn list_webhooks(&self) -> Vec<WebhookInfo> {
        let webhooks = self.webhooks.read().await;
        webhooks
            .values()
            .map(|w| WebhookInfo {
                id: w.id.clone(),
                url: w.url.clone(),
                method: w.method.clone(),
                status: format!("{:?}", w.status),
                created_at: w.created_at.elapsed(),
                last_delivery: w.last_delivery.map(|t| t.elapsed()),
                success_rate: if w.stats.total_attempts > 0 {
                    w.stats.successful_deliveries as f64 / w.stats.total_attempts as f64
                } else {
                    0.0
                },
            })
            .collect()
    }

    /// Subscribe to webhook notifications
    pub fn subscribe(&self) -> broadcast::Receiver<WebhookNotification> {
        self.event_notifier.subscribe()
    }
}

/// Webhook information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookInfo {
    pub id: String,
    pub url: String,
    pub method: HttpMethod,
    pub status: String,
    pub created_at: Duration,
    pub last_delivery: Option<Duration>,
    pub success_rate: f64,
}

impl RateLimiter {
    /// Create a new rate limiter
    fn new() -> Self {
        Self {
            limits: HashMap::new(),
            global_limit: TokenBucket::new(100.0, 200), // Global limit
        }
    }

    /// Add webhook to rate limiter
    fn add_webhook(&mut self, webhook_id: &str, config: RateLimit) {
        let bucket = TokenBucket::new(config.requests_per_second, config.burst_size);
        self.limits.insert(webhook_id.to_string(), bucket);
    }

    /// Remove webhook from rate limiter
    fn remove_webhook(&mut self, webhook_id: &str) {
        self.limits.remove(webhook_id);
    }

    /// Check rate limit
    fn check_rate_limit(&mut self, webhook_id: &str) -> bool {
        // Check global limit first
        if !self.global_limit.consume(1.0) {
            return false;
        }

        // Check webhook-specific limit
        if let Some(bucket) = self.limits.get_mut(webhook_id) {
            bucket.consume(1.0)
        } else {
            true // No limit configured
        }
    }

    /// Refill tokens
    fn refill_tokens(&mut self) {
        self.global_limit.refill();
        for bucket in self.limits.values_mut() {
            bucket.refill();
        }
    }
}

impl TokenBucket {
    /// Create a new token bucket
    fn new(refill_rate: f64, capacity: u32) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Consume tokens
    fn consume(&mut self, amount: f64) -> bool {
        self.refill();

        if self.tokens >= amount {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }

    /// Refill tokens
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();

        let tokens_to_add = elapsed * self.refill_rate;
        self.tokens = (self.tokens + tokens_to_add).min(self.capacity);
        self.last_refill = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_webhook_registration() {
        let manager = WebhookManager::new(WebhookConfig::default()).await.unwrap();

        let webhook_id = manager
            .register_webhook(
                "https://example.com/webhook".to_string(),
                HttpMethod::Post,
                HashMap::new(),
                vec![],
                WebhookSecurity {
                    hmac_secret: None,
                    auth_headers: HashMap::new(),
                    verify_ssl: true,
                    allowed_response_codes: vec![],
                },
                RetryConfig::default(),
                RateLimit::default(),
                WebhookMetadata {
                    name: Some("Test Webhook".to_string()),
                    description: None,
                    owner: None,
                    tags: vec![],
                    properties: HashMap::new(),
                },
            )
            .await
            .unwrap();

        assert!(!webhook_id.is_empty());

        let webhooks = manager.list_webhooks().await;
        assert_eq!(webhooks.len(), 1);
        assert_eq!(webhooks[0].id, webhook_id);
    }

    #[tokio::test]
    async fn test_event_filtering() {
        let manager = WebhookManager::new(WebhookConfig::default()).await.unwrap();

        let filter = EventFilter {
            event_types: Some(vec!["triple_added".to_string()]),
            graph_filter: None,
            subject_pattern: None,
            predicate_filter: None,
            custom_filter: None,
        };

        let event_match = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "test:object".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: "test".to_string(),
                timestamp: chrono::Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let event_no_match = StreamEvent::GraphCreated {
            graph: "test:graph".to_string(),
            metadata: EventMetadata {
                event_id: "test".to_string(),
                timestamp: chrono::Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        assert!(manager.matches_filters(&event_match, &[filter.clone()]));
        assert!(!manager.matches_filters(&event_no_match, &[filter]));
    }

    #[test]
    fn test_token_bucket() {
        let mut bucket = TokenBucket::new(10.0, 20);

        // Should be able to consume up to capacity
        assert!(bucket.consume(20.0));
        assert!(!bucket.consume(1.0));

        // Wait and refill
        std::thread::sleep(Duration::from_millis(100));
        bucket.refill();
        assert!(bucket.consume(1.0));
    }
}
