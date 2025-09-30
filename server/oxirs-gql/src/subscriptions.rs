//! WebSocket-based GraphQL subscription system
//!
//! This module provides real-time GraphQL subscriptions over WebSocket connections,
//! enabling clients to receive live updates when RDF data changes.

use crate::ast::{Document, Value};
use crate::execution::FieldResolver;
use crate::execution::{ExecutionContext, QueryExecutor};
use crate::types::Schema;
use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, RwLock as AsyncRwLock};
use tokio::time::interval;
use tokio_tungstenite::{accept_async, tungstenite::Message, WebSocketStream};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// GraphQL subscription message types (following graphql-ws protocol)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SubscriptionMessage {
    #[serde(rename = "connection_init")]
    ConnectionInit { payload: Option<serde_json::Value> },
    #[serde(rename = "connection_ack")]
    ConnectionAck,
    #[serde(rename = "connection_error")]
    ConnectionError { payload: Option<serde_json::Value> },
    #[serde(rename = "connection_terminate")]
    ConnectionTerminate,
    #[serde(rename = "start")]
    Start {
        id: String,
        payload: SubscriptionPayload,
    },
    #[serde(rename = "data")]
    Data {
        id: String,
        payload: serde_json::Value,
    },
    #[serde(rename = "error")]
    Error {
        id: String,
        payload: serde_json::Value,
    },
    #[serde(rename = "complete")]
    Complete { id: String },
    #[serde(rename = "stop")]
    Stop { id: String },
    #[serde(rename = "ka")]
    KeepAlive,
}

/// Subscription payload containing the GraphQL query and variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionPayload {
    pub query: String,
    pub variables: Option<HashMap<String, Value>>,
    pub operation_name: Option<String>,
}

/// Active subscription tracking information
#[derive(Debug, Clone)]
pub struct ActiveSubscription {
    pub id: String,
    pub connection_id: String,
    pub document: Document,
    pub context: ExecutionContext,
    pub created_at: Instant,
    pub last_execution: Option<Instant>,
    pub execution_count: u64,
}

/// Subscription change event
#[derive(Debug, Clone)]
pub enum SubscriptionEvent {
    TripleAdded {
        subject: String,
        predicate: String,
        object: String,
    },
    TripleRemoved {
        subject: String,
        predicate: String,
        object: String,
    },
    SubjectChanged {
        subject: String,
    },
    PredicateChanged {
        predicate: String,
    },
    BulkChange,
}

/// WebSocket connection state
#[derive(Debug)]
pub struct WebSocketConnection {
    pub id: String,
    pub socket: Arc<AsyncRwLock<WebSocketStream<TcpStream>>>,
    pub subscriptions: Arc<RwLock<HashMap<String, ActiveSubscription>>>,
    pub last_ping: Arc<RwLock<Instant>>,
    pub authenticated: Arc<RwLock<bool>>,
}

/// Authentication method for WebSocket connections
#[derive(Debug, Clone)]
pub enum AuthenticationMethod {
    /// No authentication required
    None,
    /// Bearer token authentication
    BearerToken { valid_tokens: Vec<String> },
    /// API key authentication
    ApiKey { valid_keys: Vec<String> },
    /// JWT token authentication (simplified validation)
    JWT { secret: String },
}

/// Subscription manager configuration
#[derive(Debug, Clone)]
pub struct SubscriptionConfig {
    pub max_subscriptions_per_connection: usize,
    pub max_total_subscriptions: usize,
    pub keepalive_interval: Duration,
    pub subscription_timeout: Duration,
    pub enable_authentication: bool,
    pub max_execution_frequency: Duration,
    pub auth_method: AuthenticationMethod,
}

impl Default for SubscriptionConfig {
    fn default() -> Self {
        Self {
            max_subscriptions_per_connection: 10,
            max_total_subscriptions: 1000,
            keepalive_interval: Duration::from_secs(30),
            subscription_timeout: Duration::from_secs(300),
            enable_authentication: false,
            max_execution_frequency: Duration::from_millis(100),
            auth_method: AuthenticationMethod::None,
        }
    }
}

/// WebSocket-based subscription manager
pub struct SubscriptionManager {
    config: SubscriptionConfig,
    connections: Arc<RwLock<HashMap<String, Arc<WebSocketConnection>>>>,
    active_subscriptions: Arc<RwLock<HashMap<String, ActiveSubscription>>>,
    event_sender: broadcast::Sender<SubscriptionEvent>,
    schema: Arc<AsyncRwLock<Schema>>,
    executor: Arc<QueryExecutor>,
    resolvers: Arc<RwLock<HashMap<String, Arc<dyn FieldResolver>>>>,
}

impl SubscriptionManager {
    pub fn new(config: SubscriptionConfig, schema: Schema, executor: QueryExecutor) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            active_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            schema: Arc::new(AsyncRwLock::new(schema)),
            executor: Arc::new(executor),
            resolvers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start the subscription server on the given address
    pub async fn start_server(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        info!("GraphQL subscription server listening on {}", addr);

        // Start background tasks
        self.start_keepalive_task().await;
        self.start_cleanup_task().await;

        // Accept incoming connections
        while let Ok((stream, addr)) = listener.accept().await {
            info!("New WebSocket connection from {}", addr);

            let manager = self.clone();
            tokio::spawn(async move {
                if let Err(e) = manager.handle_connection(stream).await {
                    error!("Error handling WebSocket connection: {}", e);
                }
            });
        }

        Ok(())
    }

    /// Handle a new WebSocket connection
    async fn handle_connection(&self, stream: TcpStream) -> Result<()> {
        let ws_stream = accept_async(stream).await?;
        let connection_id = Uuid::new_v4().to_string();

        let connection = Arc::new(WebSocketConnection {
            id: connection_id.clone(),
            socket: Arc::new(AsyncRwLock::new(ws_stream)),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            last_ping: Arc::new(RwLock::new(Instant::now())),
            authenticated: Arc::new(RwLock::new(!self.config.enable_authentication)),
        });

        // Register connection
        {
            let mut connections = self.connections.write().unwrap();
            connections.insert(connection_id.clone(), connection.clone());
        }

        // Handle messages
        let result = self.handle_connection_messages(connection.clone()).await;

        // Cleanup on disconnect
        self.cleanup_connection(&connection_id).await;

        result
    }

    /// Handle WebSocket messages for a connection
    async fn handle_connection_messages(&self, connection: Arc<WebSocketConnection>) -> Result<()> {
        let mut event_receiver = self.event_sender.subscribe();

        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                message_result = async {
                    let mut socket = connection.socket.write().await;
                    socket.next().await
                } => {
                    match message_result {
                        Some(Ok(msg)) => {
                            if let Err(e) = self.handle_websocket_message(&connection, msg).await {
                                error!("Error handling WebSocket message: {}", e);
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                        None => {
                            debug!("WebSocket connection closed");
                            break;
                        }
                    }
                }

                // Handle subscription events
                event_result = event_receiver.recv() => {
                    match event_result {
                        Ok(event) => {
                            if let Err(e) = self.handle_subscription_event(&connection, &event).await {
                                error!("Error handling subscription event: {}", e);
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            warn!("Subscription event receiver lagged, continuing...");
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            info!("Event channel closed");
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a single WebSocket message
    async fn handle_websocket_message(
        &self,
        connection: &Arc<WebSocketConnection>,
        message: Message,
    ) -> Result<()> {
        match message {
            Message::Text(text) => {
                let msg: SubscriptionMessage = serde_json::from_str(&text)?;
                self.handle_subscription_message(connection, msg).await
            }
            Message::Close(_) => {
                info!("WebSocket connection closed by client");
                Ok(())
            }
            Message::Ping(data) => {
                let mut socket = connection.socket.write().await;
                socket.send(Message::Pong(data)).await?;
                Ok(())
            }
            Message::Pong(_) => {
                // Update last ping time
                *connection.last_ping.write().unwrap() = Instant::now();
                Ok(())
            }
            _ => Ok(()), // Ignore other message types
        }
    }

    /// Handle a GraphQL subscription message
    async fn handle_subscription_message(
        &self,
        connection: &Arc<WebSocketConnection>,
        message: SubscriptionMessage,
    ) -> Result<()> {
        match message {
            SubscriptionMessage::ConnectionInit { payload } => {
                // Handle authentication if enabled
                let is_authenticated = if self.config.enable_authentication {
                    let auth_result = self.authenticate_connection(payload).await?;

                    if !auth_result {
                        let error = SubscriptionMessage::ConnectionError {
                            payload: Some(serde_json::json!({
                                "message": "Authentication failed",
                                "code": "AUTH_FAILED"
                            })),
                        };
                        self.send_message(connection, &error).await?;
                        return Ok(());
                    }
                    auth_result
                } else {
                    // If authentication is disabled, automatically authenticate
                    true
                };

                // Set authentication status
                *connection.authenticated.write().unwrap() = is_authenticated;

                // Send acknowledgment
                let ack = SubscriptionMessage::ConnectionAck;
                self.send_message(connection, &ack).await?;
            }

            SubscriptionMessage::Start { id, payload } => {
                if !*connection.authenticated.read().unwrap() {
                    let error = SubscriptionMessage::ConnectionError {
                        payload: Some(serde_json::json!({"message": "Not authenticated"})),
                    };
                    self.send_message(connection, &error).await?;
                    return Ok(());
                }

                self.start_subscription(connection, &id, payload).await?;
            }

            SubscriptionMessage::Stop { id } => {
                self.stop_subscription(connection, &id).await?;
            }

            SubscriptionMessage::ConnectionTerminate => {
                info!("Connection terminated by client");
                return Err(anyhow!("Connection terminated"));
            }

            _ => {
                warn!("Unexpected message type received");
            }
        }

        Ok(())
    }

    /// Start a new subscription
    async fn start_subscription(
        &self,
        connection: &Arc<WebSocketConnection>,
        subscription_id: &str,
        payload: SubscriptionPayload,
    ) -> Result<()> {
        // Check subscription limits
        let connection_sub_count = {
            let connection_subscriptions = connection.subscriptions.read().unwrap();
            connection_subscriptions.len()
        };

        if connection_sub_count >= self.config.max_subscriptions_per_connection {
            let error = SubscriptionMessage::Error {
                id: subscription_id.to_string(),
                payload: serde_json::json!({"message": "Too many subscriptions for this connection"}),
            };
            self.send_message(connection, &error).await?;
            return Ok(());
        }

        let total_sub_count = {
            let active_subscriptions = self.active_subscriptions.read().unwrap();
            active_subscriptions.len()
        };

        if total_sub_count >= self.config.max_total_subscriptions {
            let error = SubscriptionMessage::Error {
                id: subscription_id.to_string(),
                payload: serde_json::json!({"message": "Server subscription limit reached"}),
            };
            self.send_message(connection, &error).await?;
            return Ok(());
        }

        // Parse the GraphQL document
        let document = match crate::parser::parse_document(&payload.query) {
            Ok(doc) => doc,
            Err(e) => {
                let error = SubscriptionMessage::Error {
                    id: subscription_id.to_string(),
                    payload: serde_json::json!({"message": format!("Parse error: {}", e)}),
                };
                self.send_message(connection, &error).await?;
                return Ok(());
            }
        };

        // Create execution context
        let mut context = ExecutionContext::new();
        if let Some(variables) = payload.variables {
            context = context.with_variables(variables);
        }
        if let Some(operation_name) = payload.operation_name {
            context = context.with_operation_name(operation_name);
        }

        // Create subscription
        let subscription = ActiveSubscription {
            id: subscription_id.to_string(),
            connection_id: connection.id.clone(),
            document,
            context,
            created_at: Instant::now(),
            last_execution: None,
            execution_count: 0,
        };

        // Register subscription
        {
            let mut connection_subscriptions = connection.subscriptions.write().unwrap();
            connection_subscriptions.insert(subscription_id.to_string(), subscription.clone());
        }

        {
            let mut active_subscriptions = self.active_subscriptions.write().unwrap();
            active_subscriptions.insert(subscription_id.to_string(), subscription.clone());
        }

        // Execute initial subscription
        self.execute_subscription(&subscription).await?;

        info!(
            "Started subscription {} for connection {}",
            subscription_id, connection.id
        );
        Ok(())
    }

    /// Stop a subscription
    async fn stop_subscription(
        &self,
        connection: &Arc<WebSocketConnection>,
        subscription_id: &str,
    ) -> Result<()> {
        // Remove from connection
        {
            let mut connection_subscriptions = connection.subscriptions.write().unwrap();
            connection_subscriptions.remove(subscription_id);
        }

        // Remove from active subscriptions
        {
            let mut active_subscriptions = self.active_subscriptions.write().unwrap();
            active_subscriptions.remove(subscription_id);
        }

        // Send completion message
        let complete = SubscriptionMessage::Complete {
            id: subscription_id.to_string(),
        };
        self.send_message(connection, &complete).await?;

        info!(
            "Stopped subscription {} for connection {}",
            subscription_id, connection.id
        );
        Ok(())
    }

    /// Execute a subscription and send results
    async fn execute_subscription(&self, subscription: &ActiveSubscription) -> Result<()> {
        // Check execution frequency limit
        if let Some(last_execution) = subscription.last_execution {
            let elapsed = Instant::now().duration_since(last_execution);
            if elapsed < self.config.max_execution_frequency {
                return Ok(()); // Skip execution to avoid spam
            }
        }

        let result = self
            .executor
            .execute(&subscription.document, &subscription.context)
            .await?;

        // Find the connection
        let connection = {
            let connections = self.connections.read().unwrap();
            connections.get(&subscription.connection_id).cloned()
        };

        if let Some(connection) = connection {
            if result.has_errors() {
                let error = SubscriptionMessage::Error {
                    id: subscription.id.clone(),
                    payload: serde_json::to_value(&result.errors)?,
                };
                self.send_message(&connection, &error).await?;
            } else if let Some(data) = result.data {
                let data_msg = SubscriptionMessage::Data {
                    id: subscription.id.clone(),
                    payload: data,
                };
                self.send_message(&connection, &data_msg).await?;
            }

            // Update subscription execution info
            {
                let mut active_subscriptions = self.active_subscriptions.write().unwrap();
                if let Some(sub) = active_subscriptions.get_mut(&subscription.id) {
                    sub.last_execution = Some(Instant::now());
                    sub.execution_count += 1;
                }
            }
        }

        Ok(())
    }

    /// Handle subscription events and trigger re-execution
    async fn handle_subscription_event(
        &self,
        connection: &Arc<WebSocketConnection>,
        event: &SubscriptionEvent,
    ) -> Result<()> {
        let subscriptions: Vec<ActiveSubscription> = {
            let connection_subscriptions = connection.subscriptions.read().unwrap();
            connection_subscriptions.values().cloned().collect()
        };

        for subscription in subscriptions {
            // Determine if this subscription should be re-executed based on the event
            if self.should_execute_for_event(&subscription, event) {
                if let Err(e) = self.execute_subscription(&subscription).await {
                    error!("Error executing subscription {}: {}", subscription.id, e);
                }
            }
        }

        Ok(())
    }

    /// Determine if a subscription should be re-executed for a given event
    fn should_execute_for_event(
        &self,
        subscription: &ActiveSubscription,
        event: &SubscriptionEvent,
    ) -> bool {
        match event {
            SubscriptionEvent::BulkChange => true, // Always re-execute for bulk changes
            SubscriptionEvent::TripleAdded {
                subject,
                predicate,
                object: _,
            }
            | SubscriptionEvent::TripleRemoved {
                subject,
                predicate,
                object: _,
            } => {
                // Check if the subscription query involves this subject or predicate
                self.subscription_involves_resource(subscription, subject)
                    || self.subscription_involves_resource(subscription, predicate)
            }
            SubscriptionEvent::SubjectChanged { subject } => {
                self.subscription_involves_resource(subscription, subject)
            }
            SubscriptionEvent::PredicateChanged { predicate } => {
                self.subscription_involves_resource(subscription, predicate)
            }
        }
    }

    /// Check if a subscription query involves a specific resource
    fn subscription_involves_resource(
        &self,
        _subscription: &ActiveSubscription,
        _resource: &str,
    ) -> bool {
        // This is a simplified implementation - in practice, you'd want to analyze
        // the GraphQL query to determine which resources it depends on

        // For now, we'll re-execute all subscriptions for any change
        // A more sophisticated implementation would parse the query and track dependencies
        true
    }

    /// Send a message to a WebSocket connection
    async fn send_message(
        &self,
        connection: &Arc<WebSocketConnection>,
        message: &SubscriptionMessage,
    ) -> Result<()> {
        let text = serde_json::to_string(message)?;
        let mut socket = connection.socket.write().await;
        socket.send(Message::Text(text.into())).await?;
        Ok(())
    }

    /// Trigger a subscription event
    pub fn trigger_event(&self, event: SubscriptionEvent) {
        if let Err(e) = self.event_sender.send(event) {
            debug!("No active subscription listeners: {}", e);
        }
    }

    /// Start keepalive task
    async fn start_keepalive_task(&self) {
        let connections = Arc::clone(&self.connections);
        let keepalive_interval = self.config.keepalive_interval;

        tokio::spawn(async move {
            let mut interval = interval(keepalive_interval);

            loop {
                interval.tick().await;

                let connections_to_ping: Vec<Arc<WebSocketConnection>> = {
                    let connections = connections.read().unwrap();
                    connections.values().cloned().collect()
                };

                for connection in connections_to_ping {
                    let keepalive = SubscriptionMessage::KeepAlive;
                    if let Ok(text) = serde_json::to_string(&keepalive) {
                        let mut socket = connection.socket.write().await;
                        if socket.send(Message::Text(text.into())).await.is_err() {
                            debug!("Failed to send keepalive to connection {}", connection.id);
                        }
                    }
                }
            }
        });
    }

    /// Start cleanup task for expired subscriptions
    async fn start_cleanup_task(&self) {
        let active_subscriptions = Arc::clone(&self.active_subscriptions);
        let connections = Arc::clone(&self.connections);
        let timeout = self.config.subscription_timeout;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Check every minute

            loop {
                interval.tick().await;

                let now = Instant::now();
                let mut expired_subscriptions = Vec::new();

                {
                    let subscriptions = active_subscriptions.read().unwrap();
                    for (id, subscription) in subscriptions.iter() {
                        if now.duration_since(subscription.created_at) > timeout {
                            expired_subscriptions.push(id.clone());
                        }
                    }
                }

                for subscription_id in expired_subscriptions {
                    info!("Cleaning up expired subscription: {}", subscription_id);

                    // Remove from active subscriptions
                    let connection_id = {
                        let mut subscriptions = active_subscriptions.write().unwrap();
                        subscriptions
                            .remove(&subscription_id)
                            .map(|sub| sub.connection_id)
                    };

                    // Remove from connection and send completion message
                    if let Some(connection_id) = connection_id {
                        let connection_opt = {
                            let connections = connections.read().unwrap();
                            connections.get(&connection_id).cloned()
                        };

                        if let Some(connection) = connection_opt {
                            {
                                let mut connection_subscriptions =
                                    connection.subscriptions.write().unwrap();
                                connection_subscriptions.remove(&subscription_id);
                            }

                            let complete = SubscriptionMessage::Complete {
                                id: subscription_id.clone(),
                            };

                            if let Ok(text) = serde_json::to_string(&complete) {
                                let mut socket = connection.socket.write().await;
                                let _ = socket.send(Message::Text(text.into())).await;
                            }
                        }
                    }
                }
            }
        });
    }

    /// Cleanup connection and its subscriptions
    async fn cleanup_connection(&self, connection_id: &str) {
        info!("Cleaning up connection: {}", connection_id);

        // Remove connection
        {
            let mut connections = self.connections.write().unwrap();
            connections.remove(connection_id);
        }

        // Remove all subscriptions for this connection
        let subscription_ids: Vec<String> = {
            let active_subscriptions = self.active_subscriptions.read().unwrap();
            active_subscriptions
                .iter()
                .filter(|(_, sub)| sub.connection_id == connection_id)
                .map(|(id, _)| id.clone())
                .collect()
        };

        {
            let mut active_subscriptions = self.active_subscriptions.write().unwrap();
            for subscription_id in subscription_ids {
                active_subscriptions.remove(&subscription_id);
                info!(
                    "Removed subscription {} for connection {}",
                    subscription_id, connection_id
                );
            }
        }
    }

    /// Get subscription statistics
    pub fn get_stats(&self) -> SubscriptionStats {
        let connections = self.connections.read().unwrap();
        let active_subscriptions = self.active_subscriptions.read().unwrap();

        SubscriptionStats {
            total_connections: connections.len(),
            total_subscriptions: active_subscriptions.len(),
            avg_subscriptions_per_connection: if connections.is_empty() {
                0.0
            } else {
                active_subscriptions.len() as f64 / connections.len() as f64
            },
        }
    }

    /// Authenticate a WebSocket connection based on the provided payload
    async fn authenticate_connection(&self, payload: Option<serde_json::Value>) -> Result<bool> {
        if !self.config.enable_authentication {
            return Ok(true);
        }

        let payload = match payload {
            Some(p) => p,
            None => {
                debug!("No authentication payload provided");
                return Ok(false);
            }
        };

        match &self.config.auth_method {
            AuthenticationMethod::None => Ok(true),

            AuthenticationMethod::BearerToken { valid_tokens } => {
                if let Some(token) = payload
                    .get("authorization")
                    .or_else(|| payload.get("Authorization"))
                    .and_then(|v| v.as_str())
                {
                    // Remove "Bearer " prefix if present
                    let token = token.strip_prefix("Bearer ").unwrap_or(token);

                    if valid_tokens.contains(&token.to_string()) {
                        info!("WebSocket connection authenticated with Bearer token");
                        Ok(true)
                    } else {
                        warn!("Invalid Bearer token provided");
                        Ok(false)
                    }
                } else {
                    warn!("No authorization token found in payload");
                    Ok(false)
                }
            }

            AuthenticationMethod::ApiKey { valid_keys } => {
                if let Some(api_key) = payload
                    .get("apiKey")
                    .or_else(|| payload.get("api_key"))
                    .and_then(|v| v.as_str())
                {
                    if valid_keys.contains(&api_key.to_string()) {
                        info!("WebSocket connection authenticated with API key");
                        Ok(true)
                    } else {
                        warn!("Invalid API key provided");
                        Ok(false)
                    }
                } else {
                    warn!("No API key found in payload");
                    Ok(false)
                }
            }

            AuthenticationMethod::JWT { secret: _ } => {
                // Simplified JWT validation - in production, use a proper JWT library
                if let Some(jwt) = payload
                    .get("jwt")
                    .or_else(|| payload.get("token"))
                    .and_then(|v| v.as_str())
                {
                    // Basic JWT structure validation (header.payload.signature)
                    let parts: Vec<&str> = jwt.split('.').collect();
                    if parts.len() == 3 {
                        // In a real implementation, you would:
                        // 1. Decode and verify the signature
                        // 2. Check expiration
                        // 3. Validate claims
                        info!("WebSocket connection authenticated with JWT (basic validation)");
                        Ok(true)
                    } else {
                        warn!("Invalid JWT format");
                        Ok(false)
                    }
                } else {
                    warn!("No JWT token found in payload");
                    Ok(false)
                }
            }
        }
    }
}

/// Subscription server statistics
#[derive(Debug, Clone, Serialize)]
pub struct SubscriptionStats {
    pub total_connections: usize,
    pub total_subscriptions: usize,
    pub avg_subscriptions_per_connection: f64,
}

impl Clone for SubscriptionManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            connections: Arc::clone(&self.connections),
            active_subscriptions: Arc::clone(&self.active_subscriptions),
            event_sender: self.event_sender.clone(),
            schema: Arc::clone(&self.schema),
            executor: Arc::clone(&self.executor),
            resolvers: Arc::clone(&self.resolvers),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BuiltinScalars, FieldType, ObjectType, Schema};

    fn create_test_schema() -> Schema {
        let mut schema = Schema::new();

        let query_type = ObjectType::new("Query".to_string()).with_field(
            "hello".to_string(),
            FieldType::new(
                "hello".to_string(),
                crate::types::GraphQLType::Scalar(BuiltinScalars::string()),
            ),
        );

        schema.add_type(crate::types::GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());

        schema
    }

    #[test]
    fn test_subscription_config() {
        let config = SubscriptionConfig::default();
        assert_eq!(config.max_subscriptions_per_connection, 10);
        assert_eq!(config.max_total_subscriptions, 1000);
        assert!(!config.enable_authentication);
    }

    #[test]
    fn test_subscription_message_serialization() {
        let msg = SubscriptionMessage::ConnectionInit {
            payload: Some(serde_json::json!({"auth": "token"})),
        };

        let serialized = serde_json::to_string(&msg).unwrap();
        let deserialized: SubscriptionMessage = serde_json::from_str(&serialized).unwrap();

        matches!(deserialized, SubscriptionMessage::ConnectionInit { .. });
    }

    #[test]
    fn test_subscription_payload() {
        let payload = SubscriptionPayload {
            query: "subscription { hello }".to_string(),
            variables: Some(HashMap::new()),
            operation_name: None,
        };

        assert_eq!(payload.query, "subscription { hello }");
        assert!(payload.variables.is_some());
        assert!(payload.operation_name.is_none());
    }

    #[tokio::test]
    async fn test_subscription_manager_creation() {
        let config = SubscriptionConfig::default();
        let schema = create_test_schema();
        let executor = QueryExecutor::new(schema.clone());

        let manager = SubscriptionManager::new(config, schema, executor);
        let stats = manager.get_stats();

        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.total_subscriptions, 0);
        assert_eq!(stats.avg_subscriptions_per_connection, 0.0);
    }

    #[test]
    fn test_subscription_event_types() {
        let event = SubscriptionEvent::TripleAdded {
            subject: "http://example.org/person1".to_string(),
            predicate: "http://xmlns.com/foaf/0.1/name".to_string(),
            object: "John Doe".to_string(),
        };

        matches!(event, SubscriptionEvent::TripleAdded { .. });
    }

    #[test]
    fn test_active_subscription() {
        let subscription = ActiveSubscription {
            id: "sub1".to_string(),
            connection_id: "conn1".to_string(),
            document: crate::ast::Document {
                definitions: vec![],
            },
            context: ExecutionContext::new(),
            created_at: Instant::now(),
            last_execution: None,
            execution_count: 0,
        };

        assert_eq!(subscription.id, "sub1");
        assert_eq!(subscription.connection_id, "conn1");
        assert_eq!(subscription.execution_count, 0);
    }
}
