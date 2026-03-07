//! # SPARQL Subscription Protocol over WebSocket
//!
//! Implements a structured SPARQL subscription protocol where clients send
//! JSON-encoded subscription commands and the server streams results back as
//! they become available. This is distinct from the basic WebSocket module:
//! it provides a formal protocol with message framing, subscription lifecycle
//! management, and heartbeat-based keep-alive.
//!
//! ## Protocol Messages (Client -> Server)
//!
//! ```json
//! {"type": "subscribe", "id": "sub-1", "query": "SELECT ...", "interval_ms": 5000}
//! {"type": "unsubscribe", "id": "sub-1"}
//! {"type": "ping"}
//! ```
//!
//! ## Protocol Messages (Server -> Client)
//!
//! ```json
//! {"type": "subscribed", "id": "sub-1"}
//! {"type": "data", "id": "sub-1", "results": [...], "seq": 1}
//! {"type": "unsubscribed", "id": "sub-1"}
//! {"type": "error", "id": "sub-1", "message": "..."}
//! {"type": "pong"}
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Protocol messages
// ---------------------------------------------------------------------------

/// Message from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Subscribe to a SPARQL query.
    Subscribe {
        /// Subscription ID chosen by the client.
        id: String,
        /// SPARQL query string.
        query: String,
        /// Re-evaluation interval in milliseconds (default 5000).
        #[serde(default = "default_interval_ms")]
        interval_ms: u64,
        /// Optional limit on result rows per push.
        #[serde(default)]
        max_results: Option<usize>,
    },
    /// Unsubscribe from a running subscription.
    Unsubscribe {
        /// Subscription ID to cancel.
        id: String,
    },
    /// Ping (keep-alive).
    Ping,
}

fn default_interval_ms() -> u64 {
    5000
}

/// Message from server to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Subscription confirmed.
    Subscribed { id: String },
    /// Query results pushed for a subscription.
    Data {
        id: String,
        results: Vec<HashMap<String, String>>,
        seq: u64,
    },
    /// Subscription has been cancelled.
    Unsubscribed { id: String },
    /// Error related to a subscription or protocol.
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        message: String,
    },
    /// Pong reply to a ping.
    Pong,
    /// Server heartbeat.
    Heartbeat { timestamp_ms: u64 },
}

// ---------------------------------------------------------------------------
// Subscription state
// ---------------------------------------------------------------------------

/// State of a single subscription.
#[derive(Debug, Clone)]
pub struct SubscriptionState {
    /// Subscription ID.
    pub id: String,
    /// SPARQL query string.
    pub query: String,
    /// Re-evaluation interval.
    pub interval: Duration,
    /// Maximum result rows (None = unlimited).
    pub max_results: Option<usize>,
    /// Sequence counter for messages pushed.
    pub seq: u64,
    /// When the subscription was created.
    pub created_at: Instant,
    /// When the subscription was last evaluated.
    pub last_evaluated_at: Option<Instant>,
    /// Number of result rows pushed in total.
    pub total_rows_pushed: u64,
    /// Whether the subscription is active.
    pub active: bool,
}

impl SubscriptionState {
    /// Create a new subscription state.
    pub fn new(id: String, query: String, interval: Duration, max_results: Option<usize>) -> Self {
        Self {
            id,
            query,
            interval,
            max_results,
            seq: 0,
            created_at: Instant::now(),
            last_evaluated_at: None,
            total_rows_pushed: 0,
            active: true,
        }
    }

    /// Check if the subscription is due for re-evaluation.
    pub fn is_due(&self) -> bool {
        match self.last_evaluated_at {
            Some(last) => last.elapsed() >= self.interval,
            None => true, // Never evaluated yet
        }
    }

    /// Record an evaluation, advancing the sequence number.
    pub fn record_evaluation(&mut self, row_count: u64) {
        self.seq += 1;
        self.last_evaluated_at = Some(Instant::now());
        self.total_rows_pushed += row_count;
    }

    /// Deactivate the subscription.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Elapsed time since creation.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Subscription manager
// ---------------------------------------------------------------------------

/// Configuration for the subscription protocol.
#[derive(Debug, Clone)]
pub struct SubscriptionProtocolConfig {
    /// Maximum subscriptions per connection.
    pub max_subscriptions_per_connection: usize,
    /// Minimum allowed evaluation interval.
    pub min_interval: Duration,
    /// Maximum allowed evaluation interval.
    pub max_interval: Duration,
    /// Heartbeat interval for keep-alive.
    pub heartbeat_interval: Duration,
    /// Maximum query length in bytes.
    pub max_query_length: usize,
    /// Default maximum result rows per push.
    pub default_max_results: usize,
    /// Connection idle timeout.
    pub idle_timeout: Duration,
}

impl Default for SubscriptionProtocolConfig {
    fn default() -> Self {
        Self {
            max_subscriptions_per_connection: 50,
            min_interval: Duration::from_millis(500),
            max_interval: Duration::from_secs(3600),
            heartbeat_interval: Duration::from_secs(30),
            max_query_length: 100_000,
            default_max_results: 10_000,
            idle_timeout: Duration::from_secs(600),
        }
    }
}

/// Per-connection subscription manager.
pub struct SubscriptionProtocolManager {
    /// Configuration.
    config: SubscriptionProtocolConfig,
    /// Active subscriptions keyed by subscription ID.
    subscriptions: Arc<RwLock<HashMap<String, SubscriptionState>>>,
    /// Global statistics.
    stats: Arc<SubscriptionProtocolStats>,
    /// Whether the manager is running.
    running: Arc<AtomicBool>,
}

/// Aggregate statistics for the subscription protocol.
#[derive(Debug)]
pub struct SubscriptionProtocolStats {
    /// Total subscriptions created.
    pub total_subscriptions_created: AtomicU64,
    /// Total subscriptions cancelled.
    pub total_subscriptions_cancelled: AtomicU64,
    /// Total messages sent.
    pub total_messages_sent: AtomicU64,
    /// Total messages received.
    pub total_messages_received: AtomicU64,
    /// Total errors.
    pub total_errors: AtomicU64,
    /// Total pings received.
    pub total_pings: AtomicU64,
    /// Total data pushes sent.
    pub total_data_pushes: AtomicU64,
}

impl Default for SubscriptionProtocolStats {
    fn default() -> Self {
        Self {
            total_subscriptions_created: AtomicU64::new(0),
            total_subscriptions_cancelled: AtomicU64::new(0),
            total_messages_sent: AtomicU64::new(0),
            total_messages_received: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_pings: AtomicU64::new(0),
            total_data_pushes: AtomicU64::new(0),
        }
    }
}

/// Snapshot of statistics (for serialization / reporting).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSnapshot {
    pub total_subscriptions_created: u64,
    pub total_subscriptions_cancelled: u64,
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub total_errors: u64,
    pub total_pings: u64,
    pub total_data_pushes: u64,
}

impl SubscriptionProtocolStats {
    /// Take a consistent snapshot.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            total_subscriptions_created: self.total_subscriptions_created.load(Ordering::Relaxed),
            total_subscriptions_cancelled: self
                .total_subscriptions_cancelled
                .load(Ordering::Relaxed),
            total_messages_sent: self.total_messages_sent.load(Ordering::Relaxed),
            total_messages_received: self.total_messages_received.load(Ordering::Relaxed),
            total_errors: self.total_errors.load(Ordering::Relaxed),
            total_pings: self.total_pings.load(Ordering::Relaxed),
            total_data_pushes: self.total_data_pushes.load(Ordering::Relaxed),
        }
    }
}

impl SubscriptionProtocolManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: SubscriptionProtocolConfig) -> Self {
        Self {
            config,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(SubscriptionProtocolStats::default()),
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Handle an incoming client message and produce server responses.
    pub async fn handle_message(&self, msg: &ClientMessage) -> Vec<ServerMessage> {
        self.stats
            .total_messages_received
            .fetch_add(1, Ordering::Relaxed);

        match msg {
            ClientMessage::Subscribe {
                id,
                query,
                interval_ms,
                max_results,
            } => {
                self.handle_subscribe(id, query, *interval_ms, *max_results)
                    .await
            }
            ClientMessage::Unsubscribe { id } => self.handle_unsubscribe(id).await,
            ClientMessage::Ping => self.handle_ping(),
        }
    }

    /// Handle a subscribe command.
    async fn handle_subscribe(
        &self,
        id: &str,
        query: &str,
        interval_ms: u64,
        max_results: Option<usize>,
    ) -> Vec<ServerMessage> {
        // Validate
        if let Some(err) = self.validate_subscribe(id, query, interval_ms).await {
            self.stats.total_errors.fetch_add(1, Ordering::Relaxed);
            return vec![err];
        }

        let interval = Duration::from_millis(interval_ms);
        let state = SubscriptionState::new(
            id.to_string(),
            query.to_string(),
            interval,
            max_results.or(Some(self.config.default_max_results)),
        );

        {
            let mut subs = self.subscriptions.write().await;
            subs.insert(id.to_string(), state);
        }

        self.stats
            .total_subscriptions_created
            .fetch_add(1, Ordering::Relaxed);

        info!(
            "Subscription created: id={}, interval={}ms",
            id, interval_ms
        );

        vec![ServerMessage::Subscribed { id: id.to_string() }]
    }

    /// Validate a subscribe request.
    async fn validate_subscribe(
        &self,
        id: &str,
        query: &str,
        interval_ms: u64,
    ) -> Option<ServerMessage> {
        // Check subscription ID uniqueness
        {
            let subs = self.subscriptions.read().await;
            if subs.contains_key(id) {
                return Some(ServerMessage::Error {
                    id: Some(id.to_string()),
                    message: format!("Subscription ID '{id}' already exists"),
                });
            }

            // Check max subscriptions
            if subs.len() >= self.config.max_subscriptions_per_connection {
                return Some(ServerMessage::Error {
                    id: Some(id.to_string()),
                    message: format!(
                        "Maximum subscriptions ({}) reached",
                        self.config.max_subscriptions_per_connection
                    ),
                });
            }
        }

        // Check query length
        if query.len() > self.config.max_query_length {
            return Some(ServerMessage::Error {
                id: Some(id.to_string()),
                message: format!(
                    "Query too long ({} bytes, max {})",
                    query.len(),
                    self.config.max_query_length
                ),
            });
        }

        // Check interval bounds
        let interval = Duration::from_millis(interval_ms);
        if interval < self.config.min_interval {
            return Some(ServerMessage::Error {
                id: Some(id.to_string()),
                message: format!(
                    "Interval {}ms is below minimum {}ms",
                    interval_ms,
                    self.config.min_interval.as_millis()
                ),
            });
        }

        if interval > self.config.max_interval {
            return Some(ServerMessage::Error {
                id: Some(id.to_string()),
                message: format!(
                    "Interval {}ms is above maximum {}ms",
                    interval_ms,
                    self.config.max_interval.as_millis()
                ),
            });
        }

        // Check empty query
        if query.trim().is_empty() {
            return Some(ServerMessage::Error {
                id: Some(id.to_string()),
                message: "Query must not be empty".to_string(),
            });
        }

        None
    }

    /// Handle an unsubscribe command.
    async fn handle_unsubscribe(&self, id: &str) -> Vec<ServerMessage> {
        let mut subs = self.subscriptions.write().await;

        if let Some(sub) = subs.get_mut(id) {
            sub.deactivate();
            subs.remove(id);
            self.stats
                .total_subscriptions_cancelled
                .fetch_add(1, Ordering::Relaxed);
            info!("Subscription cancelled: id={}", id);
            vec![ServerMessage::Unsubscribed { id: id.to_string() }]
        } else {
            self.stats.total_errors.fetch_add(1, Ordering::Relaxed);
            vec![ServerMessage::Error {
                id: Some(id.to_string()),
                message: format!("Subscription '{id}' not found"),
            }]
        }
    }

    /// Handle a ping message.
    fn handle_ping(&self) -> Vec<ServerMessage> {
        self.stats.total_pings.fetch_add(1, Ordering::Relaxed);
        vec![ServerMessage::Pong]
    }

    /// Get subscriptions that are due for evaluation.
    pub async fn get_due_subscriptions(&self) -> Vec<String> {
        let subs = self.subscriptions.read().await;
        subs.values()
            .filter(|s| s.active && s.is_due())
            .map(|s| s.id.clone())
            .collect()
    }

    /// Create a data message for a subscription (after evaluation).
    pub async fn create_data_message(
        &self,
        sub_id: &str,
        results: Vec<HashMap<String, String>>,
    ) -> Option<ServerMessage> {
        let mut subs = self.subscriptions.write().await;

        if let Some(sub) = subs.get_mut(sub_id) {
            let row_count = results.len() as u64;

            // Apply max_results limit
            let limited_results = if let Some(max) = sub.max_results {
                results.into_iter().take(max).collect()
            } else {
                results
            };

            sub.record_evaluation(row_count);
            let seq = sub.seq;

            self.stats.total_data_pushes.fetch_add(1, Ordering::Relaxed);
            self.stats
                .total_messages_sent
                .fetch_add(1, Ordering::Relaxed);

            Some(ServerMessage::Data {
                id: sub_id.to_string(),
                results: limited_results,
                seq,
            })
        } else {
            None
        }
    }

    /// Create a heartbeat message.
    pub fn create_heartbeat(&self) -> ServerMessage {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        ServerMessage::Heartbeat { timestamp_ms }
    }

    /// Get current subscription count.
    pub async fn subscription_count(&self) -> usize {
        self.subscriptions.read().await.len()
    }

    /// Get subscription state by ID.
    pub async fn get_subscription(&self, id: &str) -> Option<SubscriptionState> {
        self.subscriptions.read().await.get(id).cloned()
    }

    /// Get all active subscription IDs.
    pub async fn active_subscription_ids(&self) -> Vec<String> {
        self.subscriptions
            .read()
            .await
            .values()
            .filter(|s| s.active)
            .map(|s| s.id.clone())
            .collect()
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    /// Check if the manager is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Stop the manager.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        debug!("Subscription protocol manager stopped");
    }

    /// Remove all inactive subscriptions.
    pub async fn cleanup_inactive(&self) -> usize {
        let mut subs = self.subscriptions.write().await;
        let before = subs.len();
        subs.retain(|_, s| s.active);
        let removed = before - subs.len();
        if removed > 0 {
            warn!("Cleaned up {} inactive subscriptions", removed);
        }
        removed
    }

    /// Get configuration reference.
    pub fn config(&self) -> &SubscriptionProtocolConfig {
        &self.config
    }

    /// Parse a raw JSON string into a ClientMessage.
    pub fn parse_client_message(raw: &str) -> Result<ClientMessage, String> {
        serde_json::from_str(raw).map_err(|e| format!("Invalid message: {e}"))
    }

    /// Serialize a ServerMessage to JSON string.
    pub fn serialize_server_message(msg: &ServerMessage) -> Result<String, String> {
        serde_json::to_string(msg).map_err(|e| format!("Serialization error: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SubscriptionProtocolConfig {
        SubscriptionProtocolConfig::default()
    }

    // --- Protocol message serialization ---

    #[test]
    fn test_client_subscribe_message_parse() {
        let raw = r#"{"type":"subscribe","id":"s1","query":"SELECT * WHERE {?s ?p ?o}","interval_ms":1000}"#;
        let msg = SubscriptionProtocolManager::parse_client_message(raw);
        assert!(msg.is_ok());
        match msg.expect("should parse") {
            ClientMessage::Subscribe {
                id,
                query,
                interval_ms,
                ..
            } => {
                assert_eq!(id, "s1");
                assert!(query.contains("SELECT"));
                assert_eq!(interval_ms, 1000);
            }
            _ => panic!("Expected Subscribe"),
        }
    }

    #[test]
    fn test_client_unsubscribe_message_parse() {
        let raw = r#"{"type":"unsubscribe","id":"s1"}"#;
        let msg = SubscriptionProtocolManager::parse_client_message(raw);
        assert!(msg.is_ok());
        match msg.expect("should parse") {
            ClientMessage::Unsubscribe { id } => assert_eq!(id, "s1"),
            _ => panic!("Expected Unsubscribe"),
        }
    }

    #[test]
    fn test_client_ping_message_parse() {
        let raw = r#"{"type":"ping"}"#;
        let msg = SubscriptionProtocolManager::parse_client_message(raw);
        assert!(msg.is_ok());
        assert!(matches!(msg.expect("should parse"), ClientMessage::Ping));
    }

    #[test]
    fn test_invalid_message_parse() {
        let raw = r#"{"invalid": true}"#;
        let msg = SubscriptionProtocolManager::parse_client_message(raw);
        assert!(msg.is_err());
    }

    #[test]
    fn test_server_message_serialize_subscribed() {
        let msg = ServerMessage::Subscribed {
            id: "s1".to_string(),
        };
        let json = SubscriptionProtocolManager::serialize_server_message(&msg);
        assert!(json.is_ok());
        let s = json.expect("should serialize");
        assert!(s.contains("subscribed"));
        assert!(s.contains("s1"));
    }

    #[test]
    fn test_server_message_serialize_data() {
        let mut row = HashMap::new();
        row.insert("name".to_string(), "Alice".to_string());
        let msg = ServerMessage::Data {
            id: "s1".to_string(),
            results: vec![row],
            seq: 1,
        };
        let json = SubscriptionProtocolManager::serialize_server_message(&msg);
        assert!(json.is_ok());
        let s = json.expect("should serialize");
        assert!(s.contains("data"));
        assert!(s.contains("Alice"));
    }

    #[test]
    fn test_server_message_serialize_error() {
        let msg = ServerMessage::Error {
            id: Some("s1".to_string()),
            message: "test error".to_string(),
        };
        let json = SubscriptionProtocolManager::serialize_server_message(&msg);
        assert!(json.is_ok());
        let s = json.expect("should serialize");
        assert!(s.contains("error"));
        assert!(s.contains("test error"));
    }

    #[test]
    fn test_server_message_serialize_pong() {
        let msg = ServerMessage::Pong;
        let json = SubscriptionProtocolManager::serialize_server_message(&msg);
        assert!(json.is_ok());
        assert!(json.expect("should serialize").contains("pong"));
    }

    #[test]
    fn test_server_message_serialize_heartbeat() {
        let msg = ServerMessage::Heartbeat {
            timestamp_ms: 1234567890,
        };
        let json = SubscriptionProtocolManager::serialize_server_message(&msg);
        assert!(json.is_ok());
        assert!(json.expect("should serialize").contains("heartbeat"));
    }

    // --- SubscriptionState ---

    #[test]
    fn test_subscription_state_new() {
        let state = SubscriptionState::new(
            "sub-1".to_string(),
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            Duration::from_secs(5),
            Some(100),
        );
        assert_eq!(state.id, "sub-1");
        assert_eq!(state.seq, 0);
        assert!(state.active);
        assert!(state.last_evaluated_at.is_none());
        assert_eq!(state.max_results, Some(100));
    }

    #[test]
    fn test_subscription_state_is_due() {
        let state = SubscriptionState::new(
            "s1".to_string(),
            "Q".to_string(),
            Duration::from_millis(10),
            None,
        );
        // Never evaluated => is_due
        assert!(state.is_due());
    }

    #[test]
    fn test_subscription_state_record_evaluation() {
        let mut state = SubscriptionState::new(
            "s1".to_string(),
            "Q".to_string(),
            Duration::from_secs(60),
            None,
        );
        state.record_evaluation(42);
        assert_eq!(state.seq, 1);
        assert_eq!(state.total_rows_pushed, 42);
        assert!(state.last_evaluated_at.is_some());

        state.record_evaluation(10);
        assert_eq!(state.seq, 2);
        assert_eq!(state.total_rows_pushed, 52);
    }

    #[test]
    fn test_subscription_state_deactivate() {
        let mut state = SubscriptionState::new(
            "s1".to_string(),
            "Q".to_string(),
            Duration::from_secs(5),
            None,
        );
        assert!(state.active);
        state.deactivate();
        assert!(!state.active);
    }

    #[test]
    fn test_subscription_state_age() {
        let state = SubscriptionState::new(
            "s1".to_string(),
            "Q".to_string(),
            Duration::from_secs(5),
            None,
        );
        // Age should be very small (just created)
        assert!(state.age() < Duration::from_secs(1));
    }

    // --- SubscriptionProtocolConfig ---

    #[test]
    fn test_config_default() {
        let config = SubscriptionProtocolConfig::default();
        assert_eq!(config.max_subscriptions_per_connection, 50);
        assert_eq!(config.min_interval, Duration::from_millis(500));
        assert_eq!(config.max_interval, Duration::from_secs(3600));
        assert_eq!(config.heartbeat_interval, Duration::from_secs(30));
        assert_eq!(config.max_query_length, 100_000);
        assert_eq!(config.default_max_results, 10_000);
    }

    // --- SubscriptionProtocolManager ---

    #[tokio::test]
    async fn test_manager_creation() {
        let manager = SubscriptionProtocolManager::new(default_config());
        assert_eq!(manager.subscription_count().await, 0);
        assert!(manager.is_running());
    }

    #[tokio::test]
    async fn test_handle_subscribe_success() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            interval_ms: 5000,
            max_results: None,
        };

        let responses = manager.handle_message(&msg).await;
        assert_eq!(responses.len(), 1);
        assert!(matches!(&responses[0], ServerMessage::Subscribed { id } if id == "s1"));
        assert_eq!(manager.subscription_count().await, 1);
    }

    #[tokio::test]
    async fn test_handle_subscribe_duplicate_id() {
        let manager = SubscriptionProtocolManager::new(default_config());

        // First subscribe
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: None,
        };
        manager.handle_message(&msg).await;

        // Duplicate
        let responses = manager.handle_message(&msg).await;
        assert_eq!(responses.len(), 1);
        assert!(matches!(&responses[0], ServerMessage::Error { id: Some(id), .. } if id == "s1"));
    }

    #[tokio::test]
    async fn test_handle_subscribe_empty_query() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "   ".to_string(),
            interval_ms: 5000,
            max_results: None,
        };

        let responses = manager.handle_message(&msg).await;
        assert!(matches!(&responses[0], ServerMessage::Error { .. }));
    }

    #[tokio::test]
    async fn test_handle_subscribe_interval_too_low() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 100, // Below 500ms minimum
            max_results: None,
        };

        let responses = manager.handle_message(&msg).await;
        assert!(matches!(&responses[0], ServerMessage::Error { .. }));
    }

    #[tokio::test]
    async fn test_handle_subscribe_interval_too_high() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 4_000_000, // Above 3600s maximum
            max_results: None,
        };

        let responses = manager.handle_message(&msg).await;
        assert!(matches!(&responses[0], ServerMessage::Error { .. }));
    }

    #[tokio::test]
    async fn test_handle_subscribe_max_subscriptions() {
        let mut config = default_config();
        config.max_subscriptions_per_connection = 2;
        let manager = SubscriptionProtocolManager::new(config);

        for i in 0..2 {
            let msg = ClientMessage::Subscribe {
                id: format!("s{i}"),
                query: "SELECT 1".to_string(),
                interval_ms: 5000,
                max_results: None,
            };
            manager.handle_message(&msg).await;
        }

        // Third should fail
        let msg = ClientMessage::Subscribe {
            id: "s2".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: None,
        };
        let responses = manager.handle_message(&msg).await;
        assert!(matches!(&responses[0], ServerMessage::Error { .. }));
    }

    #[tokio::test]
    async fn test_handle_subscribe_query_too_long() {
        let mut config = default_config();
        config.max_query_length = 10;
        let manager = SubscriptionProtocolManager::new(config);

        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT * WHERE { ?s ?p ?o . ?o ?q ?z }".to_string(),
            interval_ms: 5000,
            max_results: None,
        };

        let responses = manager.handle_message(&msg).await;
        assert!(matches!(&responses[0], ServerMessage::Error { .. }));
    }

    #[tokio::test]
    async fn test_handle_unsubscribe_success() {
        let manager = SubscriptionProtocolManager::new(default_config());

        // Subscribe first
        let sub = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: None,
        };
        manager.handle_message(&sub).await;
        assert_eq!(manager.subscription_count().await, 1);

        // Unsubscribe
        let unsub = ClientMessage::Unsubscribe {
            id: "s1".to_string(),
        };
        let responses = manager.handle_message(&unsub).await;
        assert_eq!(responses.len(), 1);
        assert!(matches!(&responses[0], ServerMessage::Unsubscribed { id } if id == "s1"));
        assert_eq!(manager.subscription_count().await, 0);
    }

    #[tokio::test]
    async fn test_handle_unsubscribe_not_found() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let unsub = ClientMessage::Unsubscribe {
            id: "nonexistent".to_string(),
        };
        let responses = manager.handle_message(&unsub).await;
        assert!(matches!(&responses[0], ServerMessage::Error { .. }));
    }

    #[tokio::test]
    async fn test_handle_ping() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let responses = manager.handle_message(&ClientMessage::Ping).await;
        assert_eq!(responses.len(), 1);
        assert!(matches!(&responses[0], ServerMessage::Pong));
    }

    #[tokio::test]
    async fn test_create_data_message() {
        let manager = SubscriptionProtocolManager::new(default_config());

        // Subscribe
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: None,
        };
        manager.handle_message(&msg).await;

        // Create data message
        let mut row = HashMap::new();
        row.insert("x".to_string(), "42".to_string());
        let data_msg = manager.create_data_message("s1", vec![row]).await;
        assert!(data_msg.is_some());

        if let Some(ServerMessage::Data { id, results, seq }) = data_msg {
            assert_eq!(id, "s1");
            assert_eq!(results.len(), 1);
            assert_eq!(seq, 1);
        } else {
            panic!("Expected Data message");
        }
    }

    #[tokio::test]
    async fn test_create_data_message_nonexistent() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let data_msg = manager.create_data_message("s1", vec![]).await;
        assert!(data_msg.is_none());
    }

    #[tokio::test]
    async fn test_create_data_message_increments_seq() {
        let manager = SubscriptionProtocolManager::new(default_config());

        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: None,
        };
        manager.handle_message(&msg).await;

        // Push data multiple times
        for expected_seq in 1..=5u64 {
            let data_msg = manager.create_data_message("s1", vec![]).await;
            if let Some(ServerMessage::Data { seq, .. }) = data_msg {
                assert_eq!(seq, expected_seq);
            }
        }
    }

    #[tokio::test]
    async fn test_create_heartbeat() {
        let manager = SubscriptionProtocolManager::new(default_config());
        let hb = manager.create_heartbeat();
        if let ServerMessage::Heartbeat { timestamp_ms } = hb {
            assert!(timestamp_ms > 0);
        } else {
            panic!("Expected Heartbeat");
        }
    }

    #[tokio::test]
    async fn test_get_subscription() {
        let manager = SubscriptionProtocolManager::new(default_config());

        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            interval_ms: 3000,
            max_results: Some(50),
        };
        manager.handle_message(&msg).await;

        let sub = manager.get_subscription("s1").await;
        assert!(sub.is_some());
        let sub = sub.expect("should exist");
        assert_eq!(sub.query, "SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(sub.interval, Duration::from_millis(3000));
        assert_eq!(sub.max_results, Some(50));
    }

    #[tokio::test]
    async fn test_active_subscription_ids() {
        let manager = SubscriptionProtocolManager::new(default_config());

        for i in 0..3 {
            let msg = ClientMessage::Subscribe {
                id: format!("s{i}"),
                query: "SELECT 1".to_string(),
                interval_ms: 5000,
                max_results: None,
            };
            manager.handle_message(&msg).await;
        }

        let ids = manager.active_subscription_ids().await;
        assert_eq!(ids.len(), 3);
    }

    #[tokio::test]
    async fn test_get_due_subscriptions() {
        let manager = SubscriptionProtocolManager::new(default_config());

        // Subscribe with very short interval
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 500, // 500ms
            max_results: None,
        };
        manager.handle_message(&msg).await;

        // Should be due immediately (never evaluated)
        let due = manager.get_due_subscriptions().await;
        assert!(due.contains(&"s1".to_string()));
    }

    #[tokio::test]
    async fn test_stop_manager() {
        let manager = SubscriptionProtocolManager::new(default_config());
        assert!(manager.is_running());
        manager.stop();
        assert!(!manager.is_running());
    }

    #[tokio::test]
    async fn test_cleanup_inactive() {
        let manager = SubscriptionProtocolManager::new(default_config());

        // Subscribe
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: None,
        };
        manager.handle_message(&msg).await;

        // Manually deactivate
        {
            let mut subs = manager.subscriptions.write().await;
            if let Some(sub) = subs.get_mut("s1") {
                sub.deactivate();
            }
        }

        let removed = manager.cleanup_inactive().await;
        assert_eq!(removed, 1);
        assert_eq!(manager.subscription_count().await, 0);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let manager = SubscriptionProtocolManager::new(default_config());

        // Subscribe
        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: None,
        };
        manager.handle_message(&msg).await;

        // Ping
        manager.handle_message(&ClientMessage::Ping).await;
        manager.handle_message(&ClientMessage::Ping).await;

        // Unsubscribe
        let msg = ClientMessage::Unsubscribe {
            id: "s1".to_string(),
        };
        manager.handle_message(&msg).await;

        let stats = manager.stats();
        assert_eq!(stats.total_subscriptions_created, 1);
        assert_eq!(stats.total_subscriptions_cancelled, 1);
        assert_eq!(stats.total_pings, 2);
        assert_eq!(stats.total_messages_received, 4);
    }

    #[tokio::test]
    async fn test_data_push_with_max_results() {
        let manager = SubscriptionProtocolManager::new(default_config());

        let msg = ClientMessage::Subscribe {
            id: "s1".to_string(),
            query: "SELECT 1".to_string(),
            interval_ms: 5000,
            max_results: Some(2),
        };
        manager.handle_message(&msg).await;

        // Push 5 rows, should be limited to 2
        let rows: Vec<HashMap<String, String>> = (0..5)
            .map(|i| {
                let mut row = HashMap::new();
                row.insert("val".to_string(), i.to_string());
                row
            })
            .collect();

        let data_msg = manager.create_data_message("s1", rows).await;
        if let Some(ServerMessage::Data { results, .. }) = data_msg {
            assert_eq!(results.len(), 2);
        } else {
            panic!("Expected Data message");
        }
    }

    #[tokio::test]
    async fn test_config_access() {
        let config = SubscriptionProtocolConfig {
            max_subscriptions_per_connection: 100,
            ..default_config()
        };
        let manager = SubscriptionProtocolManager::new(config);
        assert_eq!(manager.config().max_subscriptions_per_connection, 100);
    }

    #[test]
    fn test_stats_snapshot() {
        let stats = SubscriptionProtocolStats::default();
        stats
            .total_subscriptions_created
            .store(5, Ordering::Relaxed);
        stats.total_errors.store(2, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_subscriptions_created, 5);
        assert_eq!(snapshot.total_errors, 2);
    }

    #[test]
    fn test_default_interval_ms() {
        assert_eq!(default_interval_ms(), 5000);
    }
}
