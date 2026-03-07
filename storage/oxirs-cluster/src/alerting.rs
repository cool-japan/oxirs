//! Alerting System Framework
//!
//! This module provides a comprehensive alerting system for the OxiRS cluster,
//! enabling notifications for critical events, performance issues, and operational anomalies.
//!
//! # Features
//!
//! - **Multiple Alert Channels**: Email, Slack, webhooks, and custom handlers
//! - **Alert Severity Levels**: Critical, error, warning, info
//! - **Alert Throttling**: Prevent alert storms with configurable throttling
//! - **Alert Aggregation**: Group similar alerts to reduce noise
//! - **Template Support**: Customizable alert message templates
//! - **Alert History**: Track and query past alerts
//! - **SciRS2 Integration**: Statistical anomaly detection for smart alerting
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_cluster::alerting::{AlertingConfig, AlertingManager, AlertSeverity};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = AlertingConfig::default()
//!     .with_email_channel("smtp.example.com", 587, "user", "pass", "from@example.com", vec!["admin@example.com".to_string()])
//!     .with_slack_channel("https://hooks.slack.com/...");
//!
//! let mut alerting = AlertingManager::new(config).await?;
//! alerting.start().await?;
//!
//! // Send an alert
//! alerting.send_alert(
//!     AlertSeverity::Critical,
//!     "Node Failure",
//!     "Node 3 is unresponsive",
//! ).await?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use lettre::{
    message::header::ContentType, transport::smtp::authentication::Credentials, AsyncSmtpTransport,
    AsyncTransport, Message, Tokio1Executor,
};
use serde::{Deserialize, Serialize};
use slack_hook2::{PayloadBuilder, Slack};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur during alerting operations
#[derive(Debug, Error)]
pub enum AlertingError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Channel error
    #[error("Alert channel error: {0}")]
    ChannelError(String),

    /// Email sending error
    #[error("Failed to send email: {0}")]
    EmailError(String),

    /// Slack sending error
    #[error("Failed to send Slack message: {0}")]
    SlackError(String),

    /// Webhook error
    #[error("Webhook error: {0}")]
    WebhookError(String),

    /// Template error
    #[error("Template error: {0}")]
    TemplateError(String),

    /// Other errors
    #[error("Alerting error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AlertingError>;

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alerts
    Info,
    /// Warning alerts
    Warning,
    /// Error alerts
    Error,
    /// Critical alerts requiring immediate attention
    Critical,
}

impl AlertSeverity {
    /// Get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }

    /// Get emoji representation for Slack
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Info => ":information_source:",
            Self::Warning => ":warning:",
            Self::Error => ":x:",
            Self::Critical => ":rotating_light:",
        }
    }

    /// Get color for Slack attachments
    pub fn color(&self) -> &'static str {
        match self {
            Self::Info => "#36a64f",     // Green
            Self::Warning => "#ff9900",  // Orange
            Self::Error => "#ff0000",    // Red
            Self::Critical => "#8b0000", // Dark red
        }
    }
}

/// Alert category
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertCategory {
    /// Node health alerts
    NodeHealth,
    /// Consensus alerts
    Consensus,
    /// Replication alerts
    Replication,
    /// Performance alerts
    Performance,
    /// Security alerts
    Security,
    /// Storage alerts
    Storage,
    /// Network alerts
    Network,
    /// Custom category
    Custom(String),
}

/// Alert structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert ID
    pub id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert category
    pub category: AlertCategory,
    /// Alert title
    pub title: String,
    /// Alert message
    pub message: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when the alert was created
    pub timestamp: DateTime<Utc>,
    /// Node ID that generated the alert
    pub node_id: Option<u64>,
    /// Whether this alert has been acknowledged
    pub acknowledged: bool,
    /// Acknowledgment timestamp
    pub acknowledged_at: Option<DateTime<Utc>>,
}

impl Alert {
    /// Create a new alert
    pub fn new(
        severity: AlertSeverity,
        category: AlertCategory,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            severity,
            category,
            title: title.into(),
            message: message.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            node_id: None,
            acknowledged: false,
            acknowledged_at: None,
        }
    }

    /// Add metadata to the alert
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the node ID
    pub fn with_node_id(mut self, node_id: u64) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Acknowledge the alert
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
        self.acknowledged_at = Some(Utc::now());
    }

    /// Format as a plain text message
    pub fn to_text(&self) -> String {
        let node_info = self
            .node_id
            .map(|id| format!(" [Node {}]", id))
            .unwrap_or_default();

        let mut text = format!(
            "[{}]{} {}\n\n{}\n\nTimestamp: {}",
            self.severity.as_str(),
            node_info,
            self.title,
            self.message,
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );

        if !self.metadata.is_empty() {
            text.push_str("\n\nMetadata:\n");
            for (key, value) in &self.metadata {
                text.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        text
    }
}

/// Email channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailChannelConfig {
    /// SMTP server address
    pub smtp_server: String,
    /// SMTP port
    pub smtp_port: u16,
    /// Username for authentication
    pub username: String,
    /// Password for authentication
    pub password: String,
    /// From email address
    pub from_address: String,
    /// Recipient email addresses
    pub to_addresses: Vec<String>,
    /// Enable TLS
    pub use_tls: bool,
}

/// Slack channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackChannelConfig {
    /// Slack webhook URL
    pub webhook_url: String,
    /// Channel name (optional)
    pub channel: Option<String>,
    /// Username for the bot (optional)
    pub username: Option<String>,
    /// Icon emoji (optional)
    pub icon_emoji: Option<String>,
}

/// Webhook channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookChannelConfig {
    /// Webhook URL
    pub url: String,
    /// HTTP method (GET, POST, PUT)
    pub method: String,
    /// Additional headers
    pub headers: HashMap<String, String>,
    /// Request timeout (milliseconds)
    pub timeout_ms: u64,
}

/// Throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThrottlingConfig {
    /// Enable throttling
    pub enabled: bool,
    /// Time window for throttling (seconds)
    pub window_seconds: u64,
    /// Maximum alerts per window
    pub max_alerts_per_window: usize,
    /// Cooldown period after reaching limit (seconds)
    pub cooldown_seconds: u64,
}

impl Default for ThrottlingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_seconds: 60,
            max_alerts_per_window: 10,
            cooldown_seconds: 300,
        }
    }
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Email channel configuration
    pub email_channel: Option<EmailChannelConfig>,
    /// Slack channel configuration
    pub slack_channel: Option<SlackChannelConfig>,
    /// Webhook channels
    pub webhook_channels: Vec<WebhookChannelConfig>,
    /// Minimum severity level for alerts
    pub min_severity: AlertSeverity,
    /// Throttling configuration
    pub throttling: ThrottlingConfig,
    /// Maximum alert history size
    pub max_history_size: usize,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Alert aggregation window (seconds)
    pub aggregation_window_seconds: u64,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            email_channel: None,
            slack_channel: None,
            webhook_channels: Vec::new(),
            min_severity: AlertSeverity::Warning,
            throttling: ThrottlingConfig::default(),
            max_history_size: 1000,
            enable_anomaly_detection: true,
            aggregation_window_seconds: 60,
        }
    }
}

impl AlertingConfig {
    /// Add email channel
    pub fn with_email_channel(
        mut self,
        smtp_server: impl Into<String>,
        smtp_port: u16,
        username: impl Into<String>,
        password: impl Into<String>,
        from: impl Into<String>,
        to: Vec<String>,
    ) -> Self {
        self.email_channel = Some(EmailChannelConfig {
            smtp_server: smtp_server.into(),
            smtp_port,
            username: username.into(),
            password: password.into(),
            from_address: from.into(),
            to_addresses: to,
            use_tls: true,
        });
        self
    }

    /// Add Slack channel
    pub fn with_slack_channel(mut self, webhook_url: impl Into<String>) -> Self {
        self.slack_channel = Some(SlackChannelConfig {
            webhook_url: webhook_url.into(),
            channel: None,
            username: Some("OxiRS Cluster".to_string()),
            icon_emoji: Some(":gear:".to_string()),
        });
        self
    }

    /// Add webhook channel
    pub fn add_webhook(mut self, url: impl Into<String>) -> Self {
        self.webhook_channels.push(WebhookChannelConfig {
            url: url.into(),
            method: "POST".to_string(),
            headers: HashMap::new(),
            timeout_ms: 5000,
        });
        self
    }

    /// Set minimum severity
    pub fn with_min_severity(mut self, severity: AlertSeverity) -> Self {
        self.min_severity = severity;
        self
    }
}

/// Alert history entry
#[derive(Debug, Clone)]
struct AlertHistoryEntry {
    alert: Alert,
    #[allow(dead_code)] // Reserved for future tracking
    sent_at: DateTime<Utc>,
}

/// Alerting manager
pub struct AlertingManager {
    config: AlertingConfig,
    email_transport: Arc<RwLock<Option<AsyncSmtpTransport<Tokio1Executor>>>>,
    slack_client: Arc<RwLock<Option<Slack>>>,
    alert_history: Arc<RwLock<VecDeque<AlertHistoryEntry>>>,
    throttle_state: Arc<RwLock<ThrottleState>>,
    running: Arc<RwLock<bool>>,
}

#[derive(Debug)]
struct ThrottleState {
    alert_count: usize,
    window_start: DateTime<Utc>,
    in_cooldown: bool,
    cooldown_start: Option<DateTime<Utc>>,
}

impl ThrottleState {
    fn new() -> Self {
        Self {
            alert_count: 0,
            window_start: Utc::now(),
            in_cooldown: false,
            cooldown_start: None,
        }
    }
}

impl AlertingManager {
    /// Create a new alerting manager
    pub async fn new(config: AlertingConfig) -> Result<Self> {
        let email_transport = if let Some(email_config) = &config.email_channel {
            let creds =
                Credentials::new(email_config.username.clone(), email_config.password.clone());

            let transport = if email_config.use_tls {
                AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&email_config.smtp_server)
            } else {
                AsyncSmtpTransport::<Tokio1Executor>::relay(&email_config.smtp_server)
            }
            .map_err(|e| {
                AlertingError::EmailError(format!("Failed to create SMTP transport: {e}"))
            })?
            .credentials(creds)
            .port(email_config.smtp_port)
            .build();

            Arc::new(RwLock::new(Some(transport)))
        } else {
            Arc::new(RwLock::new(None))
        };

        let slack_client = if let Some(slack_config) = &config.slack_channel {
            let client = Slack::new(&slack_config.webhook_url).map_err(|e| {
                AlertingError::SlackError(format!("Failed to create Slack client: {e}"))
            })?;
            Arc::new(RwLock::new(Some(client)))
        } else {
            Arc::new(RwLock::new(None))
        };

        Ok(Self {
            config,
            email_transport,
            slack_client,
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            throttle_state: Arc::new(RwLock::new(ThrottleState::new())),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the alerting manager
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Ok(());
        }

        tracing::info!("Starting alerting manager");

        *running = true;

        // Start background cleanup task
        self.start_background_tasks().await;

        Ok(())
    }

    /// Stop the alerting manager
    pub async fn stop(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }

        tracing::info!("Stopping alerting manager");

        *running = false;

        Ok(())
    }

    /// Send an alert
    pub async fn send_alert(
        &self,
        severity: AlertSeverity,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check severity threshold
        if severity < self.config.min_severity {
            return Ok(());
        }

        // Check throttling
        if !self.check_throttle().await {
            tracing::warn!("Alert throttled due to rate limiting");
            return Ok(());
        }

        let alert = Alert::new(
            severity,
            AlertCategory::Custom("general".to_string()),
            title,
            message,
        );

        self.send_alert_internal(alert).await
    }

    /// Send an alert with category
    pub async fn send_categorized_alert(
        &self,
        severity: AlertSeverity,
        category: AlertCategory,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        if severity < self.config.min_severity {
            return Ok(());
        }

        if !self.check_throttle().await {
            tracing::warn!("Alert throttled due to rate limiting");
            return Ok(());
        }

        let alert = Alert::new(severity, category, title, message);

        self.send_alert_internal(alert).await
    }

    /// Send a custom alert
    pub async fn send_custom_alert(&self, alert: Alert) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        if alert.severity < self.config.min_severity {
            return Ok(());
        }

        if !self.check_throttle().await {
            tracing::warn!("Alert throttled due to rate limiting");
            return Ok(());
        }

        self.send_alert_internal(alert).await
    }

    /// Internal alert sending logic
    async fn send_alert_internal(&self, alert: Alert) -> Result<()> {
        tracing::info!(
            severity = ?alert.severity,
            category = ?alert.category,
            title = %alert.title,
            "Sending alert"
        );

        // Send to email channel
        if self.config.email_channel.is_some() {
            self.send_email_alert(&alert).await?;
        }

        // Send to Slack channel
        if self.config.slack_channel.is_some() {
            self.send_slack_alert(&alert).await?;
        }

        // Send to webhook channels
        for webhook_config in &self.config.webhook_channels {
            self.send_webhook_alert(&alert, webhook_config).await?;
        }

        // Add to history
        self.add_to_history(alert).await;

        Ok(())
    }

    /// Send alert via email
    async fn send_email_alert(&self, alert: &Alert) -> Result<()> {
        let transport = self.email_transport.read().await;
        let Some(transport) = transport.as_ref() else {
            return Ok(());
        };

        let email_config = self
            .config
            .email_channel
            .as_ref()
            .expect("email_channel config should be present when email transport exists");

        for to_address in &email_config.to_addresses {
            let email = Message::builder()
                .from(
                    email_config.from_address.parse().map_err(|e| {
                        AlertingError::EmailError(format!("Invalid from address: {e}"))
                    })?,
                )
                .to(to_address
                    .parse()
                    .map_err(|e| AlertingError::EmailError(format!("Invalid to address: {e}")))?)
                .subject(format!("[{}] {}", alert.severity.as_str(), alert.title))
                .header(ContentType::TEXT_PLAIN)
                .body(alert.to_text())
                .map_err(|e| AlertingError::EmailError(format!("Failed to build email: {e}")))?;

            transport
                .send(email)
                .await
                .map_err(|e| AlertingError::EmailError(format!("Failed to send email: {e}")))?;
        }

        Ok(())
    }

    /// Send alert via Slack
    async fn send_slack_alert(&self, alert: &Alert) -> Result<()> {
        let client = self.slack_client.read().await;
        let Some(client) = client.as_ref() else {
            return Ok(());
        };

        let slack_config = self
            .config
            .slack_channel
            .as_ref()
            .expect("slack_channel config should be present when slack client exists");

        let mut payload = PayloadBuilder::new()
            .text(format!(
                "{} *{}*\n{}",
                alert.severity.emoji(),
                alert.title,
                alert.message
            ))
            .username(slack_config.username.as_deref().unwrap_or("OxiRS Cluster"));

        if let Some(channel) = &slack_config.channel {
            payload = payload.channel(channel);
        }

        if let Some(icon) = &slack_config.icon_emoji {
            payload = payload.icon_emoji(icon);
        }

        let payload = payload.build().map_err(|e| {
            AlertingError::SlackError(format!("Failed to build Slack payload: {e}"))
        })?;

        // slack_hook2's send is async, we need to await it
        client
            .send(&payload)
            .await
            .map_err(|e| AlertingError::SlackError(format!("Failed to send Slack message: {e}")))?;

        Ok(())
    }

    /// Send alert via webhook
    async fn send_webhook_alert(&self, alert: &Alert, config: &WebhookChannelConfig) -> Result<()> {
        let client = reqwest::Client::new();

        let mut request = match config.method.to_uppercase().as_str() {
            "GET" => client.get(&config.url),
            "POST" => client.post(&config.url),
            "PUT" => client.put(&config.url),
            _ => {
                return Err(AlertingError::WebhookError(format!(
                    "Unsupported HTTP method: {}",
                    config.method
                )))
            }
        };

        // Add headers
        for (key, value) in &config.headers {
            request = request.header(key, value);
        }

        // Add JSON body for POST/PUT
        if config.method.to_uppercase() == "POST" || config.method.to_uppercase() == "PUT" {
            request = request.json(alert);
        }

        request
            .timeout(Duration::from_millis(config.timeout_ms))
            .send()
            .await
            .map_err(|e| AlertingError::WebhookError(format!("Failed to send webhook: {e}")))?;

        Ok(())
    }

    /// Check throttling
    async fn check_throttle(&self) -> bool {
        if !self.config.throttling.enabled {
            return true;
        }

        let mut state = self.throttle_state.write().await;
        let now = Utc::now();

        // Check if in cooldown
        if state.in_cooldown {
            if let Some(cooldown_start) = state.cooldown_start {
                let cooldown_duration =
                    Duration::from_secs(self.config.throttling.cooldown_seconds);
                if now
                    .signed_duration_since(cooldown_start)
                    .to_std()
                    .unwrap_or_default()
                    < cooldown_duration
                {
                    return false;
                }
                // Cooldown expired, reset state
                state.in_cooldown = false;
                state.cooldown_start = None;
                state.alert_count = 0;
                state.window_start = now;
            }
        }

        // Check window
        let window_duration = Duration::from_secs(self.config.throttling.window_seconds);
        if now
            .signed_duration_since(state.window_start)
            .to_std()
            .unwrap_or_default()
            >= window_duration
        {
            // Reset window
            state.alert_count = 0;
            state.window_start = now;
        }

        // Check count
        if state.alert_count >= self.config.throttling.max_alerts_per_window {
            // Enter cooldown
            state.in_cooldown = true;
            state.cooldown_start = Some(now);
            return false;
        }

        // Increment count
        state.alert_count += 1;

        true
    }

    /// Add alert to history
    async fn add_to_history(&self, alert: Alert) {
        let mut history = self.alert_history.write().await;

        history.push_back(AlertHistoryEntry {
            alert,
            sent_at: Utc::now(),
        });

        // Trim history if too large
        while history.len() > self.config.max_history_size {
            history.pop_front();
        }
    }

    /// Get alert history
    pub async fn get_history(&self) -> Vec<Alert> {
        let history = self.alert_history.read().await;
        history.iter().map(|entry| entry.alert.clone()).collect()
    }

    /// Get alert statistics
    pub async fn get_statistics(&self) -> AlertingStatistics {
        let history = self.alert_history.read().await;

        let total_alerts = history.len();
        let mut by_severity = HashMap::new();
        let mut by_category = HashMap::new();

        for entry in history.iter() {
            *by_severity.entry(entry.alert.severity).or_insert(0) += 1;
            *by_category.entry(entry.alert.category.clone()).or_insert(0) += 1;
        }

        let throttle_state = self.throttle_state.read().await;

        AlertingStatistics {
            total_alerts,
            alerts_by_severity: by_severity,
            alerts_by_category: by_category,
            is_throttled: throttle_state.in_cooldown,
            current_window_count: throttle_state.alert_count,
        }
    }

    /// Start background tasks
    async fn start_background_tasks(&self) {
        let running = Arc::clone(&self.running);
        let alert_history = Arc::clone(&self.alert_history);
        let max_size = self.config.max_history_size;

        tokio::spawn(async move {
            while *running.read().await {
                // Cleanup old alerts
                let mut history = alert_history.write().await;
                while history.len() > max_size {
                    history.pop_front();
                }
                drop(history);

                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });
    }
}

/// Alerting statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingStatistics {
    /// Total number of alerts sent
    pub total_alerts: usize,
    /// Alerts by severity
    pub alerts_by_severity: HashMap<AlertSeverity, usize>,
    /// Alerts by category
    pub alerts_by_category: HashMap<AlertCategory, usize>,
    /// Whether throttling is active
    pub is_throttled: bool,
    /// Current alert count in window
    pub current_window_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Error);
        assert!(AlertSeverity::Error < AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(
            AlertSeverity::Warning,
            AlertCategory::NodeHealth,
            "Test Alert",
            "Test message",
        );

        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.category, AlertCategory::NodeHealth);
        assert_eq!(alert.title, "Test Alert");
        assert_eq!(alert.message, "Test message");
        assert!(!alert.acknowledged);
    }

    #[test]
    fn test_alert_metadata() {
        let alert = Alert::new(
            AlertSeverity::Error,
            AlertCategory::Performance,
            "Performance Issue",
            "High latency detected",
        )
        .with_metadata("latency_ms", "500")
        .with_node_id(42);

        assert_eq!(alert.metadata.get("latency_ms"), Some(&"500".to_string()));
        assert_eq!(alert.node_id, Some(42));
    }

    #[tokio::test]
    async fn test_alerting_config_builder() {
        let config = AlertingConfig::default()
            .with_min_severity(AlertSeverity::Error)
            .with_slack_channel("https://hooks.slack.com/test");

        assert_eq!(config.min_severity, AlertSeverity::Error);
        assert!(config.slack_channel.is_some());
    }

    #[tokio::test]
    async fn test_alerting_manager_creation() {
        let config = AlertingConfig::default();
        let manager = AlertingManager::new(config).await;
        assert!(manager.is_ok());
    }
}
