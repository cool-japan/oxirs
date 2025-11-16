//! Webhook System
//!
//! Provides webhook support for event notifications and integrations.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Webhook event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebhookEvent {
    /// New message received
    MessageReceived,
    /// Message sent
    MessageSent,
    /// Session created
    SessionCreated,
    /// Session ended
    SessionEnded,
    /// Query executed
    QueryExecuted,
    /// Error occurred
    ErrorOccurred,
    /// Custom event
    Custom,
}

impl WebhookEvent {
    /// Get event name as string
    pub fn as_str(&self) -> &str {
        match self {
            WebhookEvent::MessageReceived => "message.received",
            WebhookEvent::MessageSent => "message.sent",
            WebhookEvent::SessionCreated => "session.created",
            WebhookEvent::SessionEnded => "session.ended",
            WebhookEvent::QueryExecuted => "query.executed",
            WebhookEvent::ErrorOccurred => "error.occurred",
            WebhookEvent::Custom => "custom",
        }
    }
}

/// Webhook payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    /// Event type
    pub event: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Session ID
    pub session_id: String,
    /// Event data
    pub data: serde_json::Value,
    /// Webhook ID
    pub webhook_id: String,
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Webhook ID
    pub id: String,
    /// Webhook URL
    pub url: String,
    /// Events to subscribe to
    pub events: Vec<WebhookEvent>,
    /// Custom headers
    pub headers: HashMap<String, String>,
    /// Secret for signature verification
    pub secret: Option<String>,
    /// Timeout duration
    pub timeout: Duration,
    /// Retry policy
    pub retry_policy: RetryPolicy,
    /// Is enabled
    pub enabled: bool,
}

/// Retry policy for failed webhooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retries
    pub max_retries: usize,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            url: String::new(),
            events: vec![],
            headers: HashMap::new(),
            secret: None,
            timeout: Duration::from_secs(30),
            retry_policy: RetryPolicy::default(),
            enabled: true,
        }
    }
}

/// Webhook delivery result
#[derive(Debug, Clone)]
pub struct WebhookDeliveryResult {
    /// Webhook ID
    pub webhook_id: String,
    /// Success status
    pub success: bool,
    /// Response status code
    pub status_code: Option<u16>,
    /// Response body
    pub response_body: Option<String>,
    /// Error message
    pub error: Option<String>,
    /// Delivery attempts
    pub attempts: usize,
    /// Delivery duration
    pub duration: Duration,
}

/// Webhook manager
pub struct WebhookManager {
    webhooks: Arc<RwLock<HashMap<String, WebhookConfig>>>,
    client: Client,
    delivery_history: Arc<RwLock<Vec<WebhookDeliveryResult>>>,
}

impl WebhookManager {
    /// Create a new webhook manager
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        info!("Initialized webhook manager");

        Ok(Self {
            webhooks: Arc::new(RwLock::new(HashMap::new())),
            client,
            delivery_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Register a webhook
    pub async fn register(&self, config: WebhookConfig) -> Result<()> {
        info!("Registering webhook: {} for URL: {}", config.id, config.url);

        let mut webhooks = self.webhooks.write().await;
        webhooks.insert(config.id.clone(), config);

        Ok(())
    }

    /// Unregister a webhook
    pub async fn unregister(&self, webhook_id: &str) -> Result<()> {
        info!("Unregistering webhook: {}", webhook_id);

        let mut webhooks = self.webhooks.write().await;
        webhooks.remove(webhook_id);

        Ok(())
    }

    /// Trigger a webhook event
    pub async fn trigger(
        &self,
        event: WebhookEvent,
        session_id: String,
        data: serde_json::Value,
    ) -> Result<()> {
        debug!(
            "Triggering webhook event: {:?} for session: {}",
            event, session_id
        );

        let webhooks = self.webhooks.read().await;

        for webhook in webhooks.values() {
            if !webhook.enabled {
                continue;
            }

            if !webhook.events.contains(&event) {
                continue;
            }

            let payload = WebhookPayload {
                event: event.as_str().to_string(),
                timestamp: chrono::Utc::now(),
                session_id: session_id.clone(),
                data: data.clone(),
                webhook_id: webhook.id.clone(),
            };

            // Deliver webhook asynchronously
            let webhook_clone = webhook.clone();
            let client = self.client.clone();
            let delivery_history = self.delivery_history.clone();

            tokio::spawn(async move {
                let result = Self::deliver_webhook(&client, &webhook_clone, &payload).await;

                // Store delivery result
                let mut history = delivery_history.write().await;
                history.push(result);

                // Keep only last 1000 deliveries
                if history.len() > 1000 {
                    history.remove(0);
                }
            });
        }

        Ok(())
    }

    /// Deliver a webhook with retry logic
    async fn deliver_webhook(
        client: &Client,
        webhook: &WebhookConfig,
        payload: &WebhookPayload,
    ) -> WebhookDeliveryResult {
        let start = std::time::Instant::now();
        let mut attempts = 0;
        let mut backoff = webhook.retry_policy.initial_backoff;

        loop {
            attempts += 1;

            debug!(
                "Delivering webhook to {} (attempt {})",
                webhook.url, attempts
            );

            let mut request = client
                .post(&webhook.url)
                .timeout(webhook.timeout)
                .json(&payload);

            // Add custom headers
            for (key, value) in &webhook.headers {
                request = request.header(key, value);
            }

            // Add signature if secret is configured
            if let Some(ref secret) = webhook.secret {
                let signature = Self::generate_signature(payload, secret);
                request = request.header("X-Webhook-Signature", signature);
            }

            match request.send().await {
                Ok(response) => {
                    let status = response.status();
                    let body = response.text().await.ok();

                    if status.is_success() {
                        info!("Webhook delivered successfully to {}", webhook.url);

                        return WebhookDeliveryResult {
                            webhook_id: webhook.id.clone(),
                            success: true,
                            status_code: Some(status.as_u16()),
                            response_body: body,
                            error: None,
                            attempts,
                            duration: start.elapsed(),
                        };
                    } else {
                        warn!(
                            "Webhook delivery failed with status {}: {}",
                            status, webhook.url
                        );

                        if attempts >= webhook.retry_policy.max_retries {
                            return WebhookDeliveryResult {
                                webhook_id: webhook.id.clone(),
                                success: false,
                                status_code: Some(status.as_u16()),
                                response_body: body,
                                error: Some(format!("Failed with status {}", status)),
                                attempts,
                                duration: start.elapsed(),
                            };
                        }
                    }
                }
                Err(e) => {
                    error!("Webhook delivery error to {}: {}", webhook.url, e);

                    if attempts >= webhook.retry_policy.max_retries {
                        return WebhookDeliveryResult {
                            webhook_id: webhook.id.clone(),
                            success: false,
                            status_code: None,
                            response_body: None,
                            error: Some(e.to_string()),
                            attempts,
                            duration: start.elapsed(),
                        };
                    }
                }
            }

            // Wait before retry
            tokio::time::sleep(backoff).await;

            // Increase backoff
            backoff = Duration::from_secs_f64(
                (backoff.as_secs_f64() * webhook.retry_policy.backoff_multiplier)
                    .min(webhook.retry_policy.max_backoff.as_secs_f64()),
            );
        }
    }

    /// Generate HMAC signature for webhook payload
    fn generate_signature(payload: &WebhookPayload, secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        type HmacSha256 = Hmac<Sha256>;

        let payload_json = serde_json::to_string(payload).unwrap_or_default();

        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
        mac.update(payload_json.as_bytes());

        let result = mac.finalize();
        let hex_string = result
            .into_bytes()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        format!("sha256={}", hex_string)
    }

    /// Get delivery history
    pub async fn get_delivery_history(&self, limit: usize) -> Vec<WebhookDeliveryResult> {
        let history = self.delivery_history.read().await;
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get webhook statistics
    pub async fn get_statistics(&self) -> WebhookStatistics {
        let history = self.delivery_history.read().await;

        let total_deliveries = history.len();
        let successful_deliveries = history.iter().filter(|d| d.success).count();
        let failed_deliveries = total_deliveries - successful_deliveries;

        let average_duration = if !history.is_empty() {
            history.iter().map(|d| d.duration.as_millis()).sum::<u128>() / history.len() as u128
        } else {
            0
        };

        WebhookStatistics {
            total_deliveries,
            successful_deliveries,
            failed_deliveries,
            average_duration_ms: average_duration,
        }
    }

    /// List all registered webhooks
    pub async fn list_webhooks(&self) -> Vec<WebhookConfig> {
        let webhooks = self.webhooks.read().await;
        webhooks.values().cloned().collect()
    }
}

impl Default for WebhookManager {
    fn default() -> Self {
        Self::new().expect("Failed to create webhook manager")
    }
}

/// Webhook statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookStatistics {
    /// Total deliveries
    pub total_deliveries: usize,
    /// Successful deliveries
    pub successful_deliveries: usize,
    /// Failed deliveries
    pub failed_deliveries: usize,
    /// Average delivery duration (ms)
    pub average_duration_ms: u128,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_webhook_registration() {
        let manager = WebhookManager::new().unwrap();

        let config = WebhookConfig {
            id: "test-webhook".to_string(),
            url: "https://example.com/webhook".to_string(),
            events: vec![WebhookEvent::MessageReceived],
            ..Default::default()
        };

        manager.register(config).await.unwrap();

        let webhooks = manager.list_webhooks().await;
        assert_eq!(webhooks.len(), 1);
        assert_eq!(webhooks[0].id, "test-webhook");
    }

    #[tokio::test]
    async fn test_webhook_event_names() {
        assert_eq!(WebhookEvent::MessageReceived.as_str(), "message.received");
        assert_eq!(WebhookEvent::SessionCreated.as_str(), "session.created");
    }
}
