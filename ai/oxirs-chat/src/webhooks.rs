//! Webhook System
//!
//! Provides webhook support for event notifications and integrations.

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Maximum number of response body bytes retained in delivery history, to bound
/// per-entry memory regardless of how large a webhook endpoint's response is.
const MAX_STORED_RESPONSE_BYTES: usize = 64 * 1024;

/// True if `ip` is an internal / non-routable address a webhook must never be
/// allowed to target (SSRF egress guard: loopback, private, link-local incl. the
/// cloud metadata endpoint, CGNAT, unspecified, ULA).
fn is_disallowed_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local() // covers 169.254.0.0/16 (metadata 169.254.169.254)
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                // Carrier-grade NAT 100.64.0.0/10 (not covered by std helpers).
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 64)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // Unique local addresses fc00::/7.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // Link-local unicast fe80::/10.
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // IPv4-mapped addresses whose embedded v4 is disallowed.
                || v6
                    .to_ipv4_mapped()
                    .map(|v4| is_disallowed_ip(&IpAddr::V4(v4)))
                    .unwrap_or(false)
        }
    }
}

/// Validate a webhook target URL against the SSRF egress policy.
///
/// Rejects non-HTTP(S) schemes, obvious internal names, and any host that
/// resolves to a loopback/private/link-local/metadata address. Resolution is
/// performed here (at registration and again at delivery) so both a literal
/// internal IP and a hostname pointing at one are caught.
async fn validate_webhook_target(url_str: &str) -> Result<()> {
    let url = reqwest::Url::parse(url_str).context("invalid webhook URL")?;
    match url.scheme() {
        "http" | "https" => {}
        other => return Err(anyhow!("webhook URL scheme '{other}' is not allowed")),
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("webhook URL has no host"))?;
    let host_lower = host.to_ascii_lowercase();
    if host_lower == "localhost"
        || host_lower.ends_with(".local")
        || host_lower.ends_with(".internal")
        || host_lower.ends_with(".localhost")
    {
        return Err(anyhow!("webhook host '{host}' is not allowed"));
    }

    // If the host is an IP literal, check it directly; otherwise resolve it.
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_disallowed_ip(&ip) {
            return Err(anyhow!("webhook host '{host}' is a disallowed address"));
        }
        return Ok(());
    }

    let port = url.port_or_known_default().unwrap_or(0);
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("failed to resolve webhook host '{host}'"))?;
    let mut any = false;
    for addr in addrs {
        any = true;
        if is_disallowed_ip(&addr.ip()) {
            return Err(anyhow!(
                "webhook host '{host}' resolves to a disallowed address {}",
                addr.ip()
            ));
        }
    }
    if !any {
        return Err(anyhow!("webhook host '{host}' did not resolve"));
    }
    Ok(())
}

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
            // Do not follow redirects: a 3xx to an internal host would bypass
            // the SSRF egress checks performed against the original URL.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("Failed to create HTTP client")?;

        info!("Initialized webhook manager");

        Ok(Self {
            webhooks: Arc::new(RwLock::new(HashMap::new())),
            client,
            delivery_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Register a webhook.
    ///
    /// The target URL is validated against the SSRF egress policy (no
    /// loopback/private/link-local/metadata hosts) and rejected fail-loud here
    /// so an untrusted tenant cannot register an internal target.
    pub async fn register(&self, config: WebhookConfig) -> Result<()> {
        info!("Registering webhook: {} for URL: {}", config.id, config.url);

        validate_webhook_target(&config.url)
            .await
            .with_context(|| format!("rejected webhook URL '{}'", config.url))?;

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

        // SSRF egress guard: re-validate the target at delivery time (defends
        // against records mutated after registration and DNS-rebinding).
        if let Err(e) = validate_webhook_target(&webhook.url).await {
            error!("Blocked webhook delivery to {}: {}", webhook.url, e);
            return WebhookDeliveryResult {
                webhook_id: webhook.id.clone(),
                success: false,
                status_code: None,
                response_body: None,
                error: Some(format!("blocked by SSRF egress policy: {e}")),
                attempts: 0,
                duration: start.elapsed(),
            };
        }

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
                    let body = response.bytes().await.ok().map(|bytes| {
                        let capped = &bytes[..bytes.len().min(MAX_STORED_RESPONSE_BYTES)];
                        String::from_utf8_lossy(capped).into_owned()
                    });

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
        use hmac::{Hmac, KeyInit, Mac};
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
        let manager = WebhookManager::new().expect("should succeed");

        let config = WebhookConfig {
            id: "test-webhook".to_string(),
            // Public routable IP literal — avoids requiring DNS in the test env
            // while still passing the SSRF egress policy.
            url: "https://93.184.216.34/webhook".to_string(),
            events: vec![WebhookEvent::MessageReceived],
            ..Default::default()
        };

        manager.register(config).await.expect("should succeed");

        let webhooks = manager.list_webhooks().await;
        assert_eq!(webhooks.len(), 1);
        assert_eq!(webhooks[0].id, "test-webhook");
    }

    /// Regression: SSRF egress guard rejects internal webhook targets at
    /// registration (loopback, private, and cloud-metadata addresses).
    #[tokio::test]
    async fn regression_webhook_ssrf_internal_targets_rejected() {
        let manager = WebhookManager::new().expect("should succeed");
        for url in [
            "http://127.0.0.1/hook",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/hook",
            "http://192.168.1.10/hook",
            "http://localhost:8080/hook",
            "https://[::1]/hook",
            "ftp://93.184.216.34/hook",
        ] {
            let config = WebhookConfig {
                id: format!("bad-{url}"),
                url: url.to_string(),
                events: vec![WebhookEvent::MessageReceived],
                ..Default::default()
            };
            assert!(
                manager.register(config).await.is_err(),
                "internal/invalid target must be rejected: {url}"
            );
        }
        assert!(manager.list_webhooks().await.is_empty());
    }

    /// Regression: the IP egress classifier flags internal ranges and permits
    /// public addresses.
    #[test]
    fn regression_is_disallowed_ip_classification() {
        use std::net::IpAddr;
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "192.168.0.1",
            "172.16.5.5",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fc00::1",
        ] {
            assert!(
                is_disallowed_ip(&ip.parse::<IpAddr>().expect("ip")),
                "{ip} should be disallowed"
            );
        }
        for ip in [
            "93.184.216.34",
            "8.8.8.8",
            "1.1.1.1",
            "2606:4700:4700::1111",
        ] {
            assert!(
                !is_disallowed_ip(&ip.parse::<IpAddr>().expect("ip")),
                "{ip} should be allowed"
            );
        }
    }

    #[tokio::test]
    async fn test_webhook_event_names() {
        assert_eq!(WebhookEvent::MessageReceived.as_str(), "message.received");
        assert_eq!(WebhookEvent::SessionCreated.as_str(), "session.created");
    }
}
