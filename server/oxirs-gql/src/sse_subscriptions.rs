//! Server-Sent Events (SSE) Support for GraphQL Subscriptions
//!
//! This module provides real-time GraphQL subscriptions using SSE (Server-Sent Events),
//! a simpler alternative to WebSockets for server-to-client streaming.
//!
//! ## Features
//!
//! - **HTTP-Based**: Works over standard HTTP, easier to deploy
//! - **Auto-Reconnection**: Browsers automatically reconnect on disconnect
//! - **Event Streaming**: Multiple events can be sent over single connection
//! - **Cross-Origin Support**: Works with CORS-enabled environments
//! - **Heartbeat/Keep-Alive**: Automatic connection health monitoring
//! - **Event Filtering**: Client-side event filtering by type
//! - **Backpressure Handling**: Automatic flow control
//! - **Connection Management**: Graceful shutdown and cleanup

use crate::ast::Document;
use crate::execution::ExecutionContext;
use anyhow::{anyhow, Result};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::interval;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info};
use uuid::Uuid;

/// SSE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseConfig {
    /// Keep-alive interval (in seconds)
    pub keep_alive_interval: u64,

    /// Maximum connection lifetime (in seconds, None = unlimited)
    pub max_connection_lifetime: Option<u64>,

    /// Event buffer size per connection
    pub event_buffer_size: usize,

    /// Enable automatic reconnection hints
    pub enable_reconnection: bool,

    /// Reconnection retry interval (in milliseconds)
    pub retry_interval: u64,

    /// Enable compression
    pub enable_compression: bool,

    /// Maximum concurrent connections per client
    pub max_connections_per_client: usize,

    /// Event ID generation
    pub enable_event_ids: bool,
}

impl Default for SseConfig {
    fn default() -> Self {
        Self {
            keep_alive_interval: 30,
            max_connection_lifetime: Some(3600), // 1 hour
            event_buffer_size: 100,
            enable_reconnection: true,
            retry_interval: 3000, // 3 seconds
            enable_compression: false,
            max_connections_per_client: 10,
            enable_event_ids: true,
        }
    }
}

/// SSE event type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SseEvent {
    /// Subscription data
    #[serde(rename = "data")]
    Data { id: String, data: serde_json::Value },

    /// Subscription error
    #[serde(rename = "error")]
    Error {
        id: String,
        message: String,
        code: Option<String>,
    },

    /// Subscription complete
    #[serde(rename = "complete")]
    Complete { id: String },

    /// Keep-alive heartbeat
    #[serde(rename = "heartbeat")]
    Heartbeat { timestamp: String },

    /// Connection info
    #[serde(rename = "connection")]
    Connection {
        connection_id: String,
        retry: Option<u64>,
    },
}

impl SseEvent {
    /// Convert to SSE format
    pub fn to_sse_string(&self) -> String {
        let mut output = String::new();

        // Event type
        let event_type = match self {
            SseEvent::Data { .. } => "data",
            SseEvent::Error { .. } => "error",
            SseEvent::Complete { .. } => "complete",
            SseEvent::Heartbeat { .. } => "heartbeat",
            SseEvent::Connection { .. } => "connection",
        };
        output.push_str(&format!("event: {}\n", event_type));

        // Event ID (for reconnection)
        if let SseEvent::Data { id, .. } | SseEvent::Error { id, .. } | SseEvent::Complete { id } =
            self
        {
            output.push_str(&format!("id: {}\n", id));
        }

        // Event data (JSON serialized)
        let data_json = serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string());
        output.push_str(&format!("data: {}\n", data_json));

        // Retry hint (for reconnection)
        if let SseEvent::Connection {
            retry: Some(retry), ..
        } = self
        {
            output.push_str(&format!("retry: {}\n", retry));
        }

        // End of event
        output.push('\n');

        output
    }
}

/// SSE connection state
#[derive(Debug, Clone)]
pub struct SseConnection {
    /// Connection ID
    pub connection_id: String,

    /// Client identifier (IP, user ID, etc.)
    pub client_id: String,

    /// Active subscription IDs
    pub subscriptions: Vec<String>,

    /// Connection start time
    pub connected_at: Instant,

    /// Last activity time
    pub last_activity: Instant,

    /// Event counter
    pub events_sent: u64,
}

impl SseConnection {
    pub fn new(client_id: String) -> Self {
        Self {
            connection_id: Uuid::new_v4().to_string(),
            client_id,
            subscriptions: Vec::new(),
            connected_at: Instant::now(),
            last_activity: Instant::now(),
            events_sent: 0,
        }
    }

    /// Check if connection is stale
    pub fn is_stale(&self, max_idle: Duration) -> bool {
        self.last_activity.elapsed() > max_idle
    }
}

/// SSE subscription manager
pub struct SseSubscriptionManager {
    config: SseConfig,
    connections: Arc<RwLock<HashMap<String, SseConnection>>>,
    event_senders: Arc<RwLock<HashMap<String, mpsc::Sender<SseEvent>>>>,
    broadcast_tx: broadcast::Sender<SseEvent>,
}

impl SseSubscriptionManager {
    /// Create new SSE subscription manager
    pub fn new(config: SseConfig) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);

        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            event_senders: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
        }
    }

    /// Create new SSE connection
    pub async fn create_connection(
        &self,
        client_id: String,
    ) -> Result<(String, mpsc::Receiver<SseEvent>)> {
        // Check connection limit
        let connections = self.connections.read().await;
        let client_connections = connections
            .values()
            .filter(|c| c.client_id == client_id)
            .count();

        if client_connections >= self.config.max_connections_per_client {
            return Err(anyhow!(
                "Maximum connections per client exceeded: {}",
                self.config.max_connections_per_client
            ));
        }
        drop(connections);

        // Create connection
        let connection = SseConnection::new(client_id);
        let connection_id = connection.connection_id.clone();

        // Create event channel
        let (tx, rx) = mpsc::channel(self.config.event_buffer_size);

        // Store connection and sender
        self.connections
            .write()
            .await
            .insert(connection_id.clone(), connection);
        self.event_senders
            .write()
            .await
            .insert(connection_id.clone(), tx.clone());

        // Send initial connection event
        let connection_event = SseEvent::Connection {
            connection_id: connection_id.clone(),
            retry: if self.config.enable_reconnection {
                Some(self.config.retry_interval)
            } else {
                None
            },
        };
        let _ = tx.send(connection_event).await;

        info!("SSE connection created: {}", connection_id);

        Ok((connection_id, rx))
    }

    /// Subscribe to GraphQL subscription
    pub async fn subscribe(
        &self,
        connection_id: &str,
        subscription_id: String,
        _document: Document,
        _context: ExecutionContext,
    ) -> Result<()> {
        let mut connections = self.connections.write().await;
        if let Some(connection) = connections.get_mut(connection_id) {
            connection.subscriptions.push(subscription_id.clone());
            connection.last_activity = Instant::now();
            info!(
                "Subscription added: {} to connection {}",
                subscription_id, connection_id
            );
            Ok(())
        } else {
            Err(anyhow!("Connection not found: {}", connection_id))
        }
    }

    /// Unsubscribe from subscription
    pub async fn unsubscribe(&self, connection_id: &str, subscription_id: &str) -> Result<()> {
        let mut connections = self.connections.write().await;
        if let Some(connection) = connections.get_mut(connection_id) {
            connection.subscriptions.retain(|s| s != subscription_id);
            connection.last_activity = Instant::now();

            // Send complete event
            if let Some(sender) = self.event_senders.read().await.get(connection_id) {
                let event = SseEvent::Complete {
                    id: subscription_id.to_string(),
                };
                let _ = sender.send(event).await;
            }

            info!(
                "Subscription removed: {} from connection {}",
                subscription_id, connection_id
            );
            Ok(())
        } else {
            Err(anyhow!("Connection not found: {}", connection_id))
        }
    }

    /// Publish event to subscription
    pub async fn publish_event(
        &self,
        subscription_id: &str,
        data: serde_json::Value,
    ) -> Result<()> {
        let connections = self.connections.read().await;
        let event_senders = self.event_senders.read().await;

        let mut sent_count = 0;

        for (conn_id, connection) in connections.iter() {
            if connection
                .subscriptions
                .contains(&subscription_id.to_string())
            {
                if let Some(sender) = event_senders.get(conn_id) {
                    let event = SseEvent::Data {
                        id: subscription_id.to_string(),
                        data: data.clone(),
                    };

                    if sender.send(event).await.is_ok() {
                        sent_count += 1;
                    }
                }
            }
        }

        if sent_count > 0 {
            debug!(
                "Published event to {} connections for subscription {}",
                sent_count, subscription_id
            );
        }

        Ok(())
    }

    /// Broadcast event to all connections
    pub async fn broadcast_event(&self, event: SseEvent) -> Result<()> {
        let _ = self.broadcast_tx.send(event);
        Ok(())
    }

    /// Send error to subscription
    pub async fn publish_error(
        &self,
        subscription_id: &str,
        message: String,
        code: Option<String>,
    ) -> Result<()> {
        let connections = self.connections.read().await;
        let event_senders = self.event_senders.read().await;

        for (conn_id, connection) in connections.iter() {
            if connection
                .subscriptions
                .contains(&subscription_id.to_string())
            {
                if let Some(sender) = event_senders.get(conn_id) {
                    let event = SseEvent::Error {
                        id: subscription_id.to_string(),
                        message: message.clone(),
                        code: code.clone(),
                    };
                    let _ = sender.send(event).await;
                }
            }
        }

        Ok(())
    }

    /// Close connection
    pub async fn close_connection(&self, connection_id: &str) -> Result<()> {
        self.connections.write().await.remove(connection_id);
        self.event_senders.write().await.remove(connection_id);

        info!("SSE connection closed: {}", connection_id);
        Ok(())
    }

    /// Start background tasks (heartbeat, cleanup)
    pub async fn start_background_tasks(self: Arc<Self>) {
        let manager = Arc::clone(&self);

        // Heartbeat task
        tokio::spawn(async move {
            let mut heartbeat = interval(Duration::from_secs(manager.config.keep_alive_interval));

            loop {
                heartbeat.tick().await;

                let event = SseEvent::Heartbeat {
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };

                let senders = manager.event_senders.read().await;
                for sender in senders.values() {
                    let _ = sender.send(event.clone()).await;
                }
            }
        });

        // Cleanup task
        let manager = Arc::clone(&self);
        tokio::spawn(async move {
            let mut cleanup = interval(Duration::from_secs(60));

            loop {
                cleanup.tick().await;

                let mut connections = manager.connections.write().await;
                let mut to_remove = Vec::new();

                for (conn_id, connection) in connections.iter() {
                    // Check connection lifetime
                    if let Some(max_lifetime) = manager.config.max_connection_lifetime {
                        if connection.connected_at.elapsed() > Duration::from_secs(max_lifetime) {
                            to_remove.push(conn_id.clone());
                            continue;
                        }
                    }

                    // Check idle connections
                    if connection.is_stale(Duration::from_secs(300)) {
                        // 5 minutes
                        to_remove.push(conn_id.clone());
                    }
                }

                for conn_id in &to_remove {
                    connections.remove(conn_id);
                    manager.event_senders.write().await.remove(conn_id);
                    info!("Cleaned up stale connection: {}", conn_id);
                }
            }
        });
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> SseStats {
        let connections = self.connections.read().await;

        let total_connections = connections.len();
        let total_subscriptions: usize = connections.values().map(|c| c.subscriptions.len()).sum();
        let total_events: u64 = connections.values().map(|c| c.events_sent).sum();

        SseStats {
            total_connections,
            total_subscriptions,
            total_events,
        }
    }
}

/// SSE statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseStats {
    pub total_connections: usize,
    pub total_subscriptions: usize,
    pub total_events: u64,
}

/// SSE event stream
pub struct SseEventStream {
    receiver: ReceiverStream<SseEvent>,
}

impl SseEventStream {
    pub fn new(receiver: mpsc::Receiver<SseEvent>) -> Self {
        Self {
            receiver: ReceiverStream::new(receiver),
        }
    }
}

impl Stream for SseEventStream {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.receiver).poll_next(cx) {
            Poll::Ready(Some(event)) => Poll::Ready(Some(Ok(event.to_sse_string()))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_config_default() {
        let config = SseConfig::default();
        assert_eq!(config.keep_alive_interval, 30);
        assert_eq!(config.max_connections_per_client, 10);
        assert!(config.enable_reconnection);
    }

    #[test]
    fn test_sse_event_to_string() {
        let event = SseEvent::Data {
            id: "sub-123".to_string(),
            data: serde_json::json!({"result": "test"}),
        };

        let sse_string = event.to_sse_string();
        assert!(sse_string.contains("event: data"));
        assert!(sse_string.contains("id: sub-123"));
        assert!(sse_string.contains("data:"));
    }

    #[test]
    fn test_sse_connection_creation() {
        let connection = SseConnection::new("client-123".to_string());
        assert_eq!(connection.client_id, "client-123");
        assert!(connection.subscriptions.is_empty());
        assert_eq!(connection.events_sent, 0);
    }

    #[test]
    fn test_sse_connection_stale_check() {
        let connection = SseConnection::new("client-123".to_string());
        assert!(!connection.is_stale(Duration::from_secs(60)));
    }

    #[tokio::test]
    async fn test_sse_manager_creation() {
        let config = SseConfig::default();
        let manager = SseSubscriptionManager::new(config);

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.total_subscriptions, 0);
    }

    #[tokio::test]
    async fn test_sse_manager_create_connection() {
        let config = SseConfig::default();
        let manager = SseSubscriptionManager::new(config);

        let result = manager.create_connection("client-123".to_string()).await;
        assert!(result.is_ok());

        let (connection_id, _rx) = result.unwrap();
        assert!(!connection_id.is_empty());

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_connections, 1);
    }

    #[tokio::test]
    async fn test_sse_manager_connection_limit() {
        let config = SseConfig {
            max_connections_per_client: 2,
            ..Default::default()
        };
        let manager = SseSubscriptionManager::new(config);

        // First connection - OK
        let result1 = manager.create_connection("client-123".to_string()).await;
        assert!(result1.is_ok());

        // Second connection - OK
        let result2 = manager.create_connection("client-123".to_string()).await;
        assert!(result2.is_ok());

        // Third connection - should fail
        let result3 = manager.create_connection("client-123".to_string()).await;
        assert!(result3.is_err());
    }

    #[tokio::test]
    async fn test_sse_manager_subscribe() {
        let config = SseConfig::default();
        let manager = SseSubscriptionManager::new(config);

        let (connection_id, _rx) = manager
            .create_connection("client-123".to_string())
            .await
            .unwrap();

        let result = manager
            .subscribe(
                &connection_id,
                "sub-123".to_string(),
                Document {
                    definitions: Vec::new(),
                },
                ExecutionContext {
                    variables: HashMap::new(),
                    operation_name: None,
                    request_id: uuid::Uuid::new_v4().to_string(),
                    fragments: HashMap::new(),
                },
            )
            .await;

        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_subscriptions, 1);
    }

    #[tokio::test]
    async fn test_sse_manager_publish_event() {
        let config = SseConfig::default();
        let manager = SseSubscriptionManager::new(config);

        let (connection_id, mut rx) = manager
            .create_connection("client-123".to_string())
            .await
            .unwrap();

        manager
            .subscribe(
                &connection_id,
                "sub-123".to_string(),
                Document {
                    definitions: Vec::new(),
                },
                ExecutionContext {
                    variables: HashMap::new(),
                    operation_name: None,
                    request_id: uuid::Uuid::new_v4().to_string(),
                    fragments: HashMap::new(),
                },
            )
            .await
            .unwrap();

        let data = serde_json::json!({"test": "data"});
        let result = manager.publish_event("sub-123", data).await;
        assert!(result.is_ok());

        // Check that event was received (skip initial connection event)
        let _ = rx.recv().await; // connection event
        let event = rx.recv().await;
        assert!(event.is_some());
    }

    #[tokio::test]
    async fn test_sse_manager_close_connection() {
        let config = SseConfig::default();
        let manager = SseSubscriptionManager::new(config);

        let (connection_id, _rx) = manager
            .create_connection("client-123".to_string())
            .await
            .unwrap();

        let result = manager.close_connection(&connection_id).await;
        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_connections, 0);
    }

    #[test]
    fn test_sse_event_types() {
        let data_event = SseEvent::Data {
            id: "1".to_string(),
            data: serde_json::json!({}),
        };
        assert!(data_event.to_sse_string().contains("event: data"));

        let error_event = SseEvent::Error {
            id: "1".to_string(),
            message: "error".to_string(),
            code: None,
        };
        assert!(error_event.to_sse_string().contains("event: error"));

        let complete_event = SseEvent::Complete {
            id: "1".to_string(),
        };
        assert!(complete_event.to_sse_string().contains("event: complete"));

        let heartbeat_event = SseEvent::Heartbeat {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };
        assert!(heartbeat_event.to_sse_string().contains("event: heartbeat"));
    }

    #[test]
    fn test_sse_stats() {
        let stats = SseStats {
            total_connections: 10,
            total_subscriptions: 25,
            total_events: 1000,
        };

        assert_eq!(stats.total_connections, 10);
        assert_eq!(stats.total_subscriptions, 25);
        assert_eq!(stats.total_events, 1000);
    }
}
