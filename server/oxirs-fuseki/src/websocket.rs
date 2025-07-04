//! WebSocket Support for Live SPARQL Query Subscriptions
//!
//! This module implements real-time query subscriptions using WebSockets,
//! enabling clients to receive live updates when query results change.
//!
//! Features:
//! - SPARQL subscription syntax extension
//! - Change notification system with filters
//! - Connection lifecycle management
//! - Subscription multiplexing
//! - Real-time query result streaming
//! - Event-driven data updates

use crate::{
    auth::AuthUser,
    error::{FusekiError, FusekiResult},
    metrics::MetricsService,
    store::Store,
};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use futures::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, RwLock},
    time::{interval, timeout},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// WebSocket subscription manager
pub struct SubscriptionManager {
    /// Active subscriptions by ID
    subscriptions: Arc<DashMap<String, Subscription>>,
    /// Subscriptions by query hash for efficient lookup
    query_subscriptions: Arc<DashMap<u64, HashSet<String>>>,
    /// Active WebSocket connections
    connections: Arc<DashMap<String, ConnectionInfo>>,
    /// Change notification broadcaster
    change_broadcaster: broadcast::Sender<ChangeNotification>,
    /// Query executor for running subscribed queries
    query_executor: Arc<QueryExecutor>,
    /// Metrics service
    metrics: Arc<MetricsService>,
    /// Configuration
    config: Arc<WebSocketConfig>,
}

/// WebSocket configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Maximum subscriptions per connection
    pub max_subscriptions_per_connection: usize,
    /// Maximum total subscriptions
    pub max_total_subscriptions: usize,
    /// Query re-evaluation interval
    pub evaluation_interval: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Maximum message size
    pub max_message_size: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            max_subscriptions_per_connection: 100,
            max_total_subscriptions: 10000,
            evaluation_interval: Duration::from_secs(1),
            connection_timeout: Duration::from_secs(300),
            max_message_size: 10 * 1024 * 1024, // 10MB
            enable_compression: true,
            heartbeat_interval: Duration::from_secs(30),
        }
    }
}

/// Active subscription information
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique subscription ID
    pub id: String,
    /// Connection ID that owns this subscription
    pub connection_id: String,
    /// SPARQL query
    pub query: String,
    /// Query parameters
    pub parameters: QueryParameters,
    /// Filter for change notifications
    pub filter: Option<NotificationFilter>,
    /// Last query result hash for change detection
    pub last_result_hash: Option<u64>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last evaluation timestamp
    pub last_evaluated: Instant,
    /// Number of notifications sent
    pub notification_count: u64,
}

/// Query parameters for subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParameters {
    /// Default graph URIs
    pub default_graph_uri: Vec<String>,
    /// Named graph URIs
    pub named_graph_uri: Vec<String>,
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Result format
    pub format: String,
}

/// Filter for change notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationFilter {
    /// Minimum change threshold (percentage)
    pub min_change_threshold: Option<f64>,
    /// Specific variables to monitor
    pub monitored_variables: Option<Vec<String>>,
    /// Debounce time in milliseconds
    pub debounce_ms: Option<u64>,
    /// Maximum notifications per minute
    pub rate_limit: Option<u32>,
}

/// WebSocket connection information
#[derive(Debug)]
pub struct ConnectionInfo {
    /// Unique connection ID
    pub id: String,
    /// Authenticated user (if any)
    pub user: Option<AuthUser>,
    /// Active subscription IDs
    pub subscription_ids: HashSet<String>,
    /// Connection established time
    pub connected_at: Instant,
    /// Last activity time
    pub last_activity: Instant,
    /// Message sender channel
    pub sender: mpsc::Sender<Message>,
    /// Connection state
    pub state: ConnectionState,
}

/// Connection state
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Connected,
    Authenticated,
    Closing,
    Closed,
}

/// Change notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeNotification {
    /// Affected graphs
    pub graphs: Vec<String>,
    /// Type of change
    pub change_type: ChangeType,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Optional details
    pub details: Option<serde_json::Value>,
}

/// Type of change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Insert,
    Delete,
    Update,
    Clear,
    Load,
    Transaction,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsMessage {
    /// Subscribe to a query
    Subscribe {
        query: String,
        parameters: QueryParameters,
        filter: Option<NotificationFilter>,
    },
    /// Unsubscribe from a query
    Unsubscribe { subscription_id: String },
    /// Query result update
    QueryUpdate {
        subscription_id: String,
        result: QueryResult,
        changes: Option<ResultChanges>,
    },
    /// Acknowledge message
    Ack {
        message_id: String,
        success: bool,
        error: Option<String>,
    },
    /// Error message
    Error {
        code: String,
        message: String,
        details: Option<serde_json::Value>,
    },
    /// Heartbeat/ping
    Ping { timestamp: u64 },
    /// Heartbeat/pong
    Pong { timestamp: u64 },
    /// Authentication
    Auth { token: String },
    /// Subscription confirmation
    Subscribed {
        subscription_id: String,
        query: String,
    },
    /// Unsubscription confirmation
    Unsubscribed { subscription_id: String },
}

/// Query result for WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Result bindings
    pub bindings: Vec<HashMap<String, serde_json::Value>>,
    /// Result metadata
    pub metadata: ResultMetadata,
}

/// Result metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    /// Query execution time
    pub execution_time_ms: u64,
    /// Result count
    pub result_count: usize,
    /// Result hash for change detection
    pub result_hash: u64,
}

/// Changes between query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultChanges {
    /// Added bindings
    pub added: Vec<HashMap<String, serde_json::Value>>,
    /// Removed bindings
    pub removed: Vec<HashMap<String, serde_json::Value>>,
    /// Modified bindings (old, new)
    pub modified: Vec<(
        HashMap<String, serde_json::Value>,
        HashMap<String, serde_json::Value>,
    )>,
}

/// Query executor for subscriptions
pub struct QueryExecutor {
    /// Store reference
    store: Arc<Store>,
    /// Execution thread pool
    executor: tokio::runtime::Handle,
}

impl SubscriptionManager {
    pub fn new(store: Arc<Store>, metrics: Arc<MetricsService>, config: WebSocketConfig) -> Self {
        let (tx, _rx) = broadcast::channel(1000);

        Self {
            subscriptions: Arc::new(DashMap::new()),
            query_subscriptions: Arc::new(DashMap::new()),
            connections: Arc::new(DashMap::new()),
            change_broadcaster: tx,
            query_executor: Arc::new(QueryExecutor::new(store)),
            metrics,
            config: Arc::new(config),
        }
    }

    /// Start the subscription manager
    pub async fn start(&self) {
        info!("Starting WebSocket subscription manager");

        // Start evaluation loop
        let manager = self.clone();
        tokio::spawn(async move {
            manager.evaluation_loop().await;
        });

        // Start cleanup loop
        let manager = self.clone();
        tokio::spawn(async move {
            manager.cleanup_loop().await;
        });
    }

    /// Handle WebSocket upgrade request
    pub async fn handle_websocket(&self, ws: WebSocketUpgrade, user: Option<AuthUser>) -> Response {
        let connection_id = Uuid::new_v4().to_string();
        let manager = self.clone();

        ws.on_upgrade(move |socket| async move {
            if let Err(e) = manager.handle_connection(socket, connection_id, user).await {
                error!("WebSocket connection error: {}", e);
            }
        })
    }

    /// Handle a WebSocket connection
    async fn handle_connection(
        &self,
        ws: WebSocket,
        connection_id: String,
        user: Option<AuthUser>,
    ) -> FusekiResult<()> {
        info!("New WebSocket connection: {}", connection_id);
        self.metrics
            .increment_counter("websocket.connections.total", 1);

        let (sender, receiver) = ws.split();
        let (tx, rx) = mpsc::channel(100);

        // Create connection info
        let conn_info = ConnectionInfo {
            id: connection_id.clone(),
            user,
            subscription_ids: HashSet::new(),
            connected_at: Instant::now(),
            last_activity: Instant::now(),
            sender: tx,
            state: ConnectionState::Connected,
        };

        self.connections.insert(connection_id.clone(), conn_info);

        // Spawn sender task
        let sender_task = tokio::spawn(Self::message_sender(sender, rx));

        // Spawn receiver task
        let receiver_task = tokio::spawn(
            self.clone()
                .message_receiver(receiver, connection_id.clone()),
        );

        // Wait for tasks to complete
        let _ = tokio::try_join!(sender_task, receiver_task);

        // Cleanup connection
        self.cleanup_connection(&connection_id).await;

        info!("WebSocket connection closed: {}", connection_id);
        self.metrics
            .increment_counter("websocket.connections.closed", 1);

        Ok(())
    }

    /// Message sender task
    async fn message_sender(
        mut sender: SplitSink<WebSocket, Message>,
        mut receiver: mpsc::Receiver<Message>,
    ) {
        while let Some(msg) = receiver.recv().await {
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    }

    /// Message receiver task
    async fn message_receiver(self, mut receiver: SplitStream<WebSocket>, connection_id: String) {
        while let Some(result) = receiver.next().await {
            match result {
                Ok(msg) => {
                    if let Err(e) = self.handle_message(msg, &connection_id).await {
                        error!("Error handling message: {}", e);
                        self.send_error(&connection_id, "message_error", &e.to_string())
                            .await;
                    }
                }
                Err(e) => {
                    error!("WebSocket receive error: {}", e);
                    break;
                }
            }
        }
    }

    /// Handle incoming WebSocket message
    async fn handle_message(&self, msg: Message, connection_id: &str) -> FusekiResult<()> {
        // Update last activity
        if let Some(mut conn) = self.connections.get_mut(connection_id) {
            conn.last_activity = Instant::now();
        }

        match msg {
            Message::Text(text) => {
                let ws_msg: WsMessage = serde_json::from_str(&text)
                    .map_err(|e| FusekiError::bad_request(format!("Invalid message: {}", e)))?;
                self.handle_ws_message(ws_msg, connection_id).await?;
            }
            Message::Binary(data) => {
                // Handle compressed messages if enabled
                if self.config.enable_compression {
                    let decompressed = Self::decompress_message(&data)?;
                    let ws_msg: WsMessage = serde_json::from_slice(&decompressed)
                        .map_err(|e| FusekiError::bad_request(format!("Invalid message: {}", e)))?;
                    self.handle_ws_message(ws_msg, connection_id).await?;
                } else {
                    return Err(FusekiError::bad_request("Binary messages not supported"));
                }
            }
            Message::Ping(data) => {
                self.send_message(connection_id, Message::Pong(data))
                    .await?;
            }
            Message::Pong(_) => {
                // Handle pong for connection health
            }
            Message::Close(_) => {
                // Connection closing
                if let Some(mut conn) = self.connections.get_mut(connection_id) {
                    conn.state = ConnectionState::Closing;
                }
            }
        }

        Ok(())
    }

    /// Handle WebSocket protocol message
    async fn handle_ws_message(&self, msg: WsMessage, connection_id: &str) -> FusekiResult<()> {
        match msg {
            WsMessage::Subscribe {
                query,
                parameters,
                filter,
            } => {
                self.handle_subscribe(connection_id, query, parameters, filter)
                    .await?;
            }
            WsMessage::Unsubscribe { subscription_id } => {
                self.handle_unsubscribe(connection_id, &subscription_id)
                    .await?;
            }
            WsMessage::Ping { timestamp } => {
                self.send_ws_message(connection_id, WsMessage::Pong { timestamp })
                    .await?;
            }
            WsMessage::Auth { token } => {
                self.handle_auth(connection_id, &token).await?;
            }
            _ => {
                return Err(FusekiError::bad_request("Unexpected message type"));
            }
        }

        Ok(())
    }

    /// Handle subscription request
    async fn handle_subscribe(
        &self,
        connection_id: &str,
        query: String,
        parameters: QueryParameters,
        filter: Option<NotificationFilter>,
    ) -> FusekiResult<()> {
        // Check subscription limits
        self.check_subscription_limits(connection_id)?;

        // Validate query
        Self::validate_subscription_query(&query)?;

        // Create subscription
        let subscription_id = Uuid::new_v4().to_string();
        let subscription = Subscription {
            id: subscription_id.clone(),
            connection_id: connection_id.to_string(),
            query: query.clone(),
            parameters,
            filter,
            last_result_hash: None,
            created_at: Instant::now(),
            last_evaluated: Instant::now(),
            notification_count: 0,
        };

        // Store subscription
        self.subscriptions
            .insert(subscription_id.clone(), subscription.clone());

        // Update query index
        let query_hash = Self::hash_query(&query);
        self.query_subscriptions
            .entry(query_hash)
            .or_insert_with(HashSet::new)
            .insert(subscription_id.clone());

        // Update connection
        if let Some(mut conn) = self.connections.get_mut(connection_id) {
            conn.subscription_ids.insert(subscription_id.clone());
        }

        // Send confirmation
        self.send_ws_message(
            connection_id,
            WsMessage::Subscribed {
                subscription_id: subscription_id.clone(),
                query: query.clone(),
            },
        )
        .await?;

        // Evaluate immediately
        self.evaluate_subscription(&subscription_id).await?;

        self.metrics
            .increment_counter("websocket.subscriptions.created", 1);
        info!(
            "Created subscription {} for connection {}",
            subscription_id, connection_id
        );

        Ok(())
    }

    /// Handle unsubscribe request
    async fn handle_unsubscribe(
        &self,
        connection_id: &str,
        subscription_id: &str,
    ) -> FusekiResult<()> {
        // Verify ownership
        if let Some(sub) = self.subscriptions.get(subscription_id) {
            if sub.connection_id != connection_id {
                return Err(FusekiError::forbidden("Not subscription owner"));
            }
        } else {
            return Err(FusekiError::not_found("Subscription not found"));
        }

        // Remove subscription
        self.remove_subscription(subscription_id).await;

        // Send confirmation
        self.send_ws_message(
            connection_id,
            WsMessage::Unsubscribed {
                subscription_id: subscription_id.to_string(),
            },
        )
        .await?;

        Ok(())
    }

    /// Handle authentication
    async fn handle_auth(&self, connection_id: &str, token: &str) -> FusekiResult<()> {
        // TODO: Implement actual authentication
        // For now, just update connection state
        if let Some(mut conn) = self.connections.get_mut(connection_id) {
            conn.state = ConnectionState::Authenticated;
        }

        self.send_ws_message(
            connection_id,
            WsMessage::Ack {
                message_id: Uuid::new_v4().to_string(),
                success: true,
                error: None,
            },
        )
        .await?;

        Ok(())
    }

    /// Check subscription limits
    fn check_subscription_limits(&self, connection_id: &str) -> FusekiResult<()> {
        // Check per-connection limit
        if let Some(conn) = self.connections.get(connection_id) {
            if conn.subscription_ids.len() >= self.config.max_subscriptions_per_connection {
                return Err(FusekiError::bad_request(
                    "Maximum subscriptions per connection exceeded",
                ));
            }
        }

        // Check total limit
        if self.subscriptions.len() >= self.config.max_total_subscriptions {
            return Err(FusekiError::service_unavailable(
                "Maximum total subscriptions exceeded",
            ));
        }

        Ok(())
    }

    /// Validate subscription query
    pub fn validate_subscription_query(query: &str) -> FusekiResult<()> {
        // Basic validation
        if query.trim().is_empty() {
            return Err(FusekiError::bad_request("Empty query"));
        }

        // Check for supported query types
        let query_lower = query.to_lowercase();
        if !query_lower.contains("select") && !query_lower.contains("construct") {
            return Err(FusekiError::bad_request(
                "Only SELECT and CONSTRUCT queries supported for subscriptions",
            ));
        }

        // Prevent expensive queries
        if !query_lower.contains("limit") {
            return Err(FusekiError::bad_request(
                "Subscription queries must include LIMIT clause",
            ));
        }

        Ok(())
    }

    /// Evaluation loop for subscriptions
    async fn evaluation_loop(&self) {
        let mut interval = interval(self.config.evaluation_interval);

        loop {
            interval.tick().await;

            // Get all subscriptions
            let subscription_ids: Vec<String> = self
                .subscriptions
                .iter()
                .map(|entry| entry.key().clone())
                .collect();

            // Evaluate subscriptions sequentially due to borrowing constraints
            for id in subscription_ids {
                if let Err(e) = self.evaluate_subscription(&id).await {
                    error!("Subscription evaluation error for {}: {}", id, e);
                }
            }
        }
    }

    /// Evaluate a single subscription
    async fn evaluate_subscription(&self, subscription_id: &str) -> FusekiResult<()> {
        let subscription = match self.subscriptions.get(subscription_id) {
            Some(sub) => sub.clone(),
            None => return Ok(()), // Subscription removed
        };

        // Check rate limit
        if let Some(filter) = &subscription.filter {
            if let Some(rate_limit) = filter.rate_limit {
                let notifications_per_minute = subscription.notification_count as f64
                    / subscription.created_at.elapsed().as_secs_f64()
                    * 60.0;
                if notifications_per_minute > rate_limit as f64 {
                    return Ok(()); // Skip due to rate limit
                }
            }
        }

        // Execute query
        let result = self
            .query_executor
            .execute_query(&subscription.query, &subscription.parameters)
            .await?;

        // Calculate result hash
        let result_hash = Self::hash_result(&result);

        // Check for changes
        let has_changed = subscription
            .last_result_hash
            .map(|h| h != result_hash)
            .unwrap_or(true);

        if !has_changed {
            return Ok(()); // No changes
        }

        // Apply change filter
        if let Some(filter) = &subscription.filter {
            if !self
                .apply_notification_filter(&subscription, &result, filter)
                .await?
            {
                return Ok(()); // Filtered out
            }
        }

        // Calculate changes
        let changes = if subscription.last_result_hash.is_some() {
            // TODO: Calculate actual changes between results
            None
        } else {
            None
        };

        // Send update
        self.send_ws_message(
            &subscription.connection_id,
            WsMessage::QueryUpdate {
                subscription_id: subscription_id.to_string(),
                result: result.clone(),
                changes,
            },
        )
        .await?;

        // Update subscription
        if let Some(mut sub) = self.subscriptions.get_mut(subscription_id) {
            sub.last_result_hash = Some(result_hash);
            sub.last_evaluated = Instant::now();
            sub.notification_count += 1;
        }

        self.metrics
            .increment_counter("websocket.notifications.sent", 1);

        Ok(())
    }

    /// Apply notification filter
    async fn apply_notification_filter(
        &self,
        subscription: &Subscription,
        result: &QueryResult,
        filter: &NotificationFilter,
    ) -> FusekiResult<bool> {
        // Check minimum change threshold
        if let Some(threshold) = filter.min_change_threshold {
            // TODO: Calculate actual change percentage
            let change_percentage = 10.0; // Placeholder
            if change_percentage < threshold {
                return Ok(false);
            }
        }

        // Check debounce
        if let Some(debounce_ms) = filter.debounce_ms {
            let time_since_last = subscription.last_evaluated.elapsed().as_millis() as u64;
            if time_since_last < debounce_ms {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Cleanup loop for expired connections
    async fn cleanup_loop(&self) {
        let mut interval = interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            let now = Instant::now();
            let timeout = self.config.connection_timeout;

            // Find expired connections
            let expired: Vec<String> = self
                .connections
                .iter()
                .filter(|entry| now.duration_since(entry.last_activity) > timeout)
                .map(|entry| entry.key().clone())
                .collect();

            // Clean up expired connections
            for connection_id in expired {
                warn!("Cleaning up expired connection: {}", connection_id);
                self.cleanup_connection(&connection_id).await;
            }
        }
    }

    /// Clean up a connection and its subscriptions
    async fn cleanup_connection(&self, connection_id: &str) {
        if let Some((_, conn)) = self.connections.remove(connection_id) {
            // Remove all subscriptions
            for sub_id in &conn.subscription_ids {
                self.remove_subscription(sub_id).await;
            }
        }

        // Note: Metrics service doesn't have decrement_gauge, using direct gauge update
        // In a real implementation, you'd track the current count and decrement it
    }

    /// Remove a subscription
    async fn remove_subscription(&self, subscription_id: &str) {
        if let Some((_, sub)) = self.subscriptions.remove(subscription_id) {
            // Remove from query index
            let query_hash = Self::hash_query(&sub.query);
            if let Some(mut subs) = self.query_subscriptions.get_mut(&query_hash) {
                subs.remove(subscription_id);
                if subs.is_empty() {
                    drop(subs);
                    self.query_subscriptions.remove(&query_hash);
                }
            }

            // Remove from connection
            if let Some(mut conn) = self.connections.get_mut(&sub.connection_id) {
                conn.subscription_ids.remove(subscription_id);
            }
        }

        // Note: Metrics service doesn't have decrement_gauge, using direct gauge update
        // In a real implementation, you'd track the current count and decrement it
    }

    /// Send message to connection
    async fn send_message(&self, connection_id: &str, msg: Message) -> FusekiResult<()> {
        if let Some(conn) = self.connections.get(connection_id) {
            conn.sender
                .send(msg)
                .await
                .map_err(|_| FusekiError::internal("Failed to send message"))?;
        }
        Ok(())
    }

    /// Send WebSocket protocol message
    async fn send_ws_message(&self, connection_id: &str, msg: WsMessage) -> FusekiResult<()> {
        let json = serde_json::to_string(&msg)
            .map_err(|e| FusekiError::internal(format!("Serialization error: {}", e)))?;

        self.send_message(connection_id, Message::Text(json)).await
    }

    /// Send error message
    async fn send_error(&self, connection_id: &str, code: &str, message: &str) {
        let _ = self
            .send_ws_message(
                connection_id,
                WsMessage::Error {
                    code: code.to_string(),
                    message: message.to_string(),
                    details: None,
                },
            )
            .await;
    }

    /// Hash a query for indexing
    fn hash_query(query: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash query result for change detection
    fn hash_result(result: &QueryResult) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash bindings
        for binding in &result.bindings {
            for (k, v) in binding {
                k.hash(&mut hasher);
                v.to_string().hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Decompress message
    fn decompress_message(data: &[u8]) -> FusekiResult<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| FusekiError::bad_request(format!("Decompression error: {}", e)))?;

        Ok(decompressed)
    }

    /// Handle store change notification
    pub async fn handle_store_change(&self, notification: ChangeNotification) {
        // Broadcast to change listeners
        let _ = self.change_broadcaster.send(notification.clone());

        // Find affected subscriptions
        // In a real implementation, this would analyze which queries are affected
        // For now, re-evaluate all subscriptions
        let subscription_ids: Vec<String> = self
            .subscriptions
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        for sub_id in subscription_ids {
            if let Err(e) = self.evaluate_subscription(&sub_id).await {
                error!("Error evaluating subscription after change: {}", e);
            }
        }
    }
}

impl Clone for SubscriptionManager {
    fn clone(&self) -> Self {
        Self {
            subscriptions: Arc::clone(&self.subscriptions),
            query_subscriptions: Arc::clone(&self.query_subscriptions),
            connections: Arc::clone(&self.connections),
            change_broadcaster: self.change_broadcaster.clone(),
            query_executor: Arc::clone(&self.query_executor),
            metrics: Arc::clone(&self.metrics),
            config: Arc::clone(&self.config),
        }
    }
}

impl QueryExecutor {
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            store,
            executor: tokio::runtime::Handle::current(),
        }
    }

    /// Execute a subscription query
    pub async fn execute_query(
        &self,
        query: &str,
        parameters: &QueryParameters,
    ) -> FusekiResult<QueryResult> {
        let start = Instant::now();

        // Execute query against store
        // This is a placeholder - in real implementation would use oxirs-arq
        let bindings = self.execute_sparql_query(query, parameters).await?;

        let execution_time = start.elapsed().as_millis() as u64;
        let result_count = bindings.len();
        let result_hash = SubscriptionManager::hash_result(&QueryResult {
            bindings: bindings.clone(),
            metadata: ResultMetadata {
                execution_time_ms: execution_time,
                result_count,
                result_hash: 0,
            },
        });

        Ok(QueryResult {
            bindings,
            metadata: ResultMetadata {
                execution_time_ms: execution_time,
                result_count,
                result_hash,
            },
        })
    }

    /// Execute SPARQL query
    async fn execute_sparql_query(
        &self,
        query: &str,
        parameters: &QueryParameters,
    ) -> FusekiResult<Vec<HashMap<String, serde_json::Value>>> {
        // Placeholder implementation
        // In real implementation, this would:
        // 1. Parse the SPARQL query
        // 2. Execute against the store with specified graphs
        // 3. Format results according to requested format

        // For now, return mock data
        let mut bindings = Vec::new();

        if query.to_lowercase().contains("select") {
            // Mock SELECT results
            for i in 0..3 {
                let mut binding = HashMap::new();
                binding.insert(
                    "subject".to_string(),
                    serde_json::json!(format!("http://example.org/resource{}", i)),
                );
                binding.insert(
                    "predicate".to_string(),
                    serde_json::json!("http://example.org/property"),
                );
                binding.insert(
                    "object".to_string(),
                    serde_json::json!(format!("Value {}", i)),
                );
                bindings.push(binding);
            }
        }

        Ok(bindings)
    }
}

/// WebSocket endpoint handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<crate::server::AppState>,
    user: Option<AuthUser>,
) -> Response {
    if let Some(ref subscription_manager) = state.subscription_manager {
        subscription_manager.handle_websocket(ws, user).await
    } else {
        // Return error response if WebSocket is not configured
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "WebSocket support not configured",
        )
            .into_response()
    }
}

/// WebSocket query parameters
#[derive(Debug, Deserialize)]
pub struct WebSocketQuery {
    /// Authentication token (optional)
    pub token: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_validation() {
        // Valid queries
        assert!(SubscriptionManager::validate_subscription_query(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100"
        )
        .is_ok());

        assert!(SubscriptionManager::validate_subscription_query(
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 10"
        )
        .is_ok());

        // Invalid queries
        assert!(SubscriptionManager::validate_subscription_query("").is_err());
        assert!(SubscriptionManager::validate_subscription_query("ASK { ?s ?p ?o }").is_err());
        assert!(SubscriptionManager::validate_subscription_query(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
        )
        .is_err()); // No LIMIT
    }

    #[test]
    fn test_message_serialization() {
        let msg = WsMessage::Subscribe {
            query: "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10".to_string(),
            parameters: QueryParameters {
                default_graph_uri: vec![],
                named_graph_uri: vec![],
                timeout_ms: Some(5000),
                format: "json".to_string(),
            },
            filter: Some(NotificationFilter {
                min_change_threshold: Some(5.0),
                monitored_variables: Some(vec!["s".to_string()]),
                debounce_ms: Some(1000),
                rate_limit: Some(60),
            }),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WsMessage = serde_json::from_str(&json).unwrap();

        match deserialized {
            WsMessage::Subscribe { query, .. } => {
                assert!(query.contains("SELECT"));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_query_hashing() {
        let query1 = "SELECT ?s WHERE { ?s ?p ?o }";
        let query2 = "SELECT ?s WHERE { ?s ?p ?o }";
        let query3 = "SELECT ?x WHERE { ?x ?y ?z }";

        assert_eq!(
            SubscriptionManager::hash_query(query1),
            SubscriptionManager::hash_query(query2)
        );

        assert_ne!(
            SubscriptionManager::hash_query(query1),
            SubscriptionManager::hash_query(query3)
        );
    }
}
