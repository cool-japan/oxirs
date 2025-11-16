//! Real-time Update Notifications
//!
//! This module provides real-time notifications for dataset updates, query completions,
//! system status changes, and metrics via WebSocket connections.

use anyhow::{Context, Result};
use axum::extract::ws::{Message, WebSocket};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Maximum number of notifications to buffer per client
const MAX_NOTIFICATION_BUFFER: usize = 1000;

/// Notification event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NotificationEvent {
    /// Dataset was created
    DatasetCreated { dataset: String },
    /// Dataset was updated
    DatasetUpdated {
        dataset: String,
        triple_count: usize,
    },
    /// Dataset was deleted
    DatasetDeleted { dataset: String },
    /// Query completed
    QueryCompleted {
        query_id: String,
        duration_ms: u64,
        result_count: usize,
    },
    /// Query failed
    QueryFailed { query_id: String, error: String },
    /// System status changed
    SystemStatus { status: SystemStatus },
    /// Metrics update
    MetricsUpdate { metrics: SystemMetrics },
    /// Backup completed
    BackupCompleted {
        dataset: String,
        backup_id: String,
        size_bytes: u64,
    },
    /// Backup failed
    BackupFailed { dataset: String, error: String },
    /// Federation endpoint changed
    FederationUpdate {
        endpoint: String,
        status: EndpointStatus,
    },
}

/// System status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SystemStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
}

/// Federation endpoint status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EndpointStatus {
    Available,
    Unavailable,
    Slow { latency_ms: u64 },
}

/// System metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub active_queries: usize,
    pub queries_per_second: f64,
    pub avg_query_time_ms: f64,
}

impl PartialEq for SystemMetrics {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
            && (self.cpu_usage_percent - other.cpu_usage_percent).abs() < f64::EPSILON
            && self.memory_usage_mb == other.memory_usage_mb
            && self.active_queries == other.active_queries
            && (self.queries_per_second - other.queries_per_second).abs() < f64::EPSILON
            && (self.avg_query_time_ms - other.avg_query_time_ms).abs() < f64::EPSILON
    }
}

impl Eq for SystemMetrics {}

/// Notification message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event: NotificationEvent,
}

impl Notification {
    pub fn new(event: NotificationEvent) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            event,
        }
    }
}

/// Subscription filter
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubscriptionFilter {
    /// Filter by event types
    pub event_types: Option<Vec<String>>,
    /// Filter by dataset names
    pub datasets: Option<Vec<String>>,
    /// Filter by minimum severity
    pub min_severity: Option<NotificationSeverity>,
}

/// Notification severity level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NotificationSeverity {
    Info = 0,
    Warning = 1,
    Error = 2,
    Critical = 3,
}

impl NotificationEvent {
    /// Get the severity of this event
    pub fn severity(&self) -> NotificationSeverity {
        match self {
            Self::DatasetCreated { .. } | Self::DatasetUpdated { .. } => NotificationSeverity::Info,
            Self::DatasetDeleted { .. } => NotificationSeverity::Warning,
            Self::QueryCompleted { .. } => NotificationSeverity::Info,
            Self::QueryFailed { .. } => NotificationSeverity::Warning,
            Self::SystemStatus { status } => match status {
                SystemStatus::Healthy => NotificationSeverity::Info,
                SystemStatus::Degraded { .. } => NotificationSeverity::Warning,
                SystemStatus::Unhealthy { .. } => NotificationSeverity::Critical,
            },
            Self::MetricsUpdate { .. } => NotificationSeverity::Info,
            Self::BackupCompleted { .. } => NotificationSeverity::Info,
            Self::BackupFailed { .. } => NotificationSeverity::Error,
            Self::FederationUpdate { status, .. } => match status {
                EndpointStatus::Available => NotificationSeverity::Info,
                EndpointStatus::Unavailable => NotificationSeverity::Error,
                EndpointStatus::Slow { .. } => NotificationSeverity::Warning,
            },
        }
    }

    /// Check if this event matches the filter
    pub fn matches_filter(&self, filter: &SubscriptionFilter) -> bool {
        // Check severity
        if let Some(min_severity) = filter.min_severity {
            if self.severity() < min_severity {
                return false;
            }
        }

        // Check event type
        if let Some(ref event_types) = filter.event_types {
            let event_type = match self {
                Self::DatasetCreated { .. } => "dataset_created",
                Self::DatasetUpdated { .. } => "dataset_updated",
                Self::DatasetDeleted { .. } => "dataset_deleted",
                Self::QueryCompleted { .. } => "query_completed",
                Self::QueryFailed { .. } => "query_failed",
                Self::SystemStatus { .. } => "system_status",
                Self::MetricsUpdate { .. } => "metrics_update",
                Self::BackupCompleted { .. } => "backup_completed",
                Self::BackupFailed { .. } => "backup_failed",
                Self::FederationUpdate { .. } => "federation_update",
            };

            if !event_types.contains(&event_type.to_string()) {
                return false;
            }
        }

        // Check dataset
        if let Some(ref datasets) = filter.datasets {
            let dataset = match self {
                Self::DatasetCreated { dataset }
                | Self::DatasetUpdated { dataset, .. }
                | Self::DatasetDeleted { dataset }
                | Self::BackupCompleted { dataset, .. }
                | Self::BackupFailed { dataset, .. } => Some(dataset),
                _ => None,
            };

            if let Some(dataset) = dataset {
                if !datasets.contains(dataset) {
                    return false;
                }
            }
        }

        true
    }
}

/// Client connection information
#[derive(Debug)]
struct ClientConnection {
    id: String,
    filter: SubscriptionFilter,
    tx: broadcast::Sender<Notification>,
}

/// Real-time notification manager
pub struct NotificationManager {
    /// Active client connections
    clients: Arc<DashMap<String, ClientConnection>>,
    /// Global broadcast channel
    global_tx: broadcast::Sender<Notification>,
    /// Notification history (for replay)
    history: Arc<RwLock<Vec<Notification>>>,
    /// Maximum history size
    max_history: usize,
    /// Statistics
    stats: Arc<RwLock<NotificationStats>>,
}

/// Notification statistics
#[derive(Debug, Default)]
pub struct NotificationStats {
    pub total_notifications: u64,
    pub total_clients: u64,
    pub active_clients: usize,
    pub notifications_by_type: std::collections::HashMap<String, u64>,
    pub dropped_notifications: u64,
}

impl NotificationManager {
    /// Create a new notification manager
    pub fn new() -> Self {
        let (global_tx, _) = broadcast::channel(MAX_NOTIFICATION_BUFFER);

        Self {
            clients: Arc::new(DashMap::new()),
            global_tx,
            history: Arc::new(RwLock::new(Vec::new())),
            max_history: 100,
            stats: Arc::new(RwLock::new(NotificationStats::default())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(max_history: usize, buffer_size: usize) -> Self {
        let (global_tx, _) = broadcast::channel(buffer_size);

        Self {
            clients: Arc::new(DashMap::new()),
            global_tx,
            history: Arc::new(RwLock::new(Vec::new())),
            max_history,
            stats: Arc::new(RwLock::new(NotificationStats::default())),
        }
    }

    /// Broadcast a notification to all subscribed clients
    pub async fn broadcast(&self, event: NotificationEvent) -> Result<()> {
        let notification = Notification::new(event.clone());

        // Add to history
        {
            let mut history = self.history.write().await;
            history.push(notification.clone());
            if history.len() > self.max_history {
                history.remove(0);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_notifications += 1;
            *stats
                .notifications_by_type
                .entry(format!("{:?}", event))
                .or_insert(0) += 1;
        }

        // Broadcast to global channel
        match self.global_tx.send(notification.clone()) {
            Ok(receiver_count) => {
                debug!("Broadcasted notification to {} clients", receiver_count);
            }
            Err(e) => {
                warn!("Failed to broadcast notification: {}", e);
                let mut stats = self.stats.write().await;
                stats.dropped_notifications += 1;
            }
        }

        Ok(())
    }

    /// Register a new client connection
    pub async fn register_client(
        &self,
        filter: SubscriptionFilter,
    ) -> Result<(String, broadcast::Receiver<Notification>)> {
        let client_id = Uuid::new_v4().to_string();
        let rx = self.global_tx.subscribe();

        let (tx, _) = broadcast::channel(MAX_NOTIFICATION_BUFFER);
        let client_rx = tx.subscribe();

        let connection = ClientConnection {
            id: client_id.clone(),
            filter,
            tx,
        };

        self.clients.insert(client_id.clone(), connection);

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_clients += 1;
            stats.active_clients = self.clients.len();
        }

        info!("Registered new notification client: {}", client_id);

        Ok((client_id, rx))
    }

    /// Unregister a client connection
    pub async fn unregister_client(&self, client_id: &str) -> Result<()> {
        self.clients.remove(client_id);

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.active_clients = self.clients.len();
        }

        info!("Unregistered notification client: {}", client_id);
        Ok(())
    }

    /// Get notification history
    pub async fn get_history(&self, limit: Option<usize>) -> Vec<Notification> {
        let history = self.history.read().await;
        let limit = limit.unwrap_or(history.len());
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get statistics
    pub async fn get_statistics(&self) -> NotificationStats {
        let stats = self.stats.read().await;
        NotificationStats {
            total_notifications: stats.total_notifications,
            total_clients: stats.total_clients,
            active_clients: self.clients.len(),
            notifications_by_type: stats.notifications_by_type.clone(),
            dropped_notifications: stats.dropped_notifications,
        }
    }

    /// Handle WebSocket connection for notifications
    pub async fn handle_websocket(
        self: Arc<Self>,
        mut socket: WebSocket,
        filter: SubscriptionFilter,
    ) -> Result<()> {
        // Register client
        let (client_id, mut rx) = self
            .register_client(filter.clone())
            .await
            .context("Failed to register client")?;

        info!("WebSocket client connected: {}", client_id);

        // Send initial connection message
        let welcome = serde_json::json!({
            "type": "welcome",
            "client_id": client_id,
            "message": "Connected to OxiRS Fuseki real-time notifications",
        });

        if let Err(e) = socket
            .send(Message::Text(serde_json::to_string(&welcome)?))
            .await
        {
            error!("Failed to send welcome message: {}", e);
            self.unregister_client(&client_id).await?;
            return Ok(());
        }

        // Handle incoming messages and send notifications
        loop {
            tokio::select! {
                // Receive notifications
                result = rx.recv() => {
                    match result {
                        Ok(notification) => {
                            // Check if notification matches filter
                            if !notification.event.matches_filter(&filter) {
                                continue;
                            }

                            // Send notification to client
                            let json = serde_json::to_string(&notification)?;
                            if let Err(e) = socket.send(Message::Text(json)).await {
                                error!("Failed to send notification to client {}: {}", client_id, e);
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Client {} lagged behind by {} notifications", client_id, n);
                            // Send lag notification
                            let lag_msg = serde_json::json!({
                                "type": "lag_warning",
                                "lagged_by": n,
                            });
                            let _ = socket.send(Message::Text(serde_json::to_string(&lag_msg)?)).await;
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            info!("Notification channel closed for client {}", client_id);
                            break;
                        }
                    }
                }

                // Receive messages from client
                message = socket.recv() => {
                    match message {
                        Some(Ok(Message::Text(text))) => {
                            // Handle client commands
                            if let Err(e) = self.handle_client_message(&client_id, &text).await {
                                error!("Failed to handle client message: {}", e);
                            }
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            info!("Client {} disconnected", client_id);
                            break;
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if let Err(e) = socket.send(Message::Pong(data)).await {
                                error!("Failed to send pong: {}", e);
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error for client {}: {}", client_id, e);
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Unregister client
        self.unregister_client(&client_id).await?;
        info!("WebSocket client disconnected: {}", client_id);

        Ok(())
    }

    /// Handle client message commands
    async fn handle_client_message(&self, client_id: &str, message: &str) -> Result<()> {
        #[derive(Deserialize)]
        struct ClientCommand {
            command: String,
            #[serde(default)]
            params: serde_json::Value,
        }

        let cmd: ClientCommand =
            serde_json::from_str(message).context("Failed to parse client command")?;

        match cmd.command.as_str() {
            "ping" => {
                debug!("Received ping from client {}", client_id);
            }
            "get_history" => {
                let limit: Option<usize> = cmd
                    .params
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let history = self.get_history(limit).await;
                debug!(
                    "Sent {} historical notifications to client {}",
                    history.len(),
                    client_id
                );
            }
            "update_filter" => {
                // Update client filter
                if let Ok(new_filter) = serde_json::from_value::<SubscriptionFilter>(cmd.params) {
                    if let Some(mut client) = self.clients.get_mut(client_id) {
                        client.filter = new_filter;
                        info!("Updated filter for client {}", client_id);
                    }
                }
            }
            _ => {
                warn!("Unknown command from client {}: {}", client_id, cmd.command);
            }
        }

        Ok(())
    }
}

impl Default for NotificationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_notification_creation() {
        let event = NotificationEvent::DatasetCreated {
            dataset: "test".to_string(),
        };
        let notification = Notification::new(event);

        assert!(!notification.id.is_empty());
        assert!(notification.timestamp <= chrono::Utc::now());
    }

    #[tokio::test]
    async fn test_event_severity() {
        let info_event = NotificationEvent::QueryCompleted {
            query_id: "q1".to_string(),
            duration_ms: 100,
            result_count: 10,
        };
        assert_eq!(info_event.severity(), NotificationSeverity::Info);

        let error_event = NotificationEvent::BackupFailed {
            dataset: "test".to_string(),
            error: "disk full".to_string(),
        };
        assert_eq!(error_event.severity(), NotificationSeverity::Error);

        let critical_event = NotificationEvent::SystemStatus {
            status: SystemStatus::Unhealthy {
                reason: "out of memory".to_string(),
            },
        };
        assert_eq!(critical_event.severity(), NotificationSeverity::Critical);
    }

    #[tokio::test]
    async fn test_filter_matching() {
        let event = NotificationEvent::DatasetCreated {
            dataset: "test".to_string(),
        };

        // No filter - should match
        let filter = SubscriptionFilter::default();
        assert!(event.matches_filter(&filter));

        // Dataset filter - should match
        let filter = SubscriptionFilter {
            datasets: Some(vec!["test".to_string()]),
            ..Default::default()
        };
        assert!(event.matches_filter(&filter));

        // Dataset filter - should not match
        let filter = SubscriptionFilter {
            datasets: Some(vec!["other".to_string()]),
            ..Default::default()
        };
        assert!(!event.matches_filter(&filter));

        // Severity filter - should match
        let filter = SubscriptionFilter {
            min_severity: Some(NotificationSeverity::Info),
            ..Default::default()
        };
        assert!(event.matches_filter(&filter));

        // Severity filter - should not match
        let filter = SubscriptionFilter {
            min_severity: Some(NotificationSeverity::Critical),
            ..Default::default()
        };
        assert!(!event.matches_filter(&filter));
    }

    #[tokio::test]
    async fn test_notification_manager() {
        let manager = NotificationManager::new();

        // Broadcast notification
        let event = NotificationEvent::DatasetCreated {
            dataset: "test".to_string(),
        };
        manager.broadcast(event).await.unwrap();

        // Check statistics
        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_notifications, 1);

        // Check history
        let history = manager.get_history(None).await;
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_client_registration() {
        let manager = NotificationManager::new();

        // Register client
        let filter = SubscriptionFilter::default();
        let (client_id, _rx) = manager.register_client(filter).await.unwrap();

        // Check statistics
        let stats = manager.get_statistics().await;
        assert_eq!(stats.active_clients, 1);

        // Unregister client
        manager.unregister_client(&client_id).await.unwrap();

        // Check statistics
        let stats = manager.get_statistics().await;
        assert_eq!(stats.active_clients, 0);
    }

    #[tokio::test]
    async fn test_notification_history() {
        let manager = NotificationManager::with_config(10, 100);

        // Broadcast multiple notifications
        for i in 0..15 {
            let event = NotificationEvent::DatasetCreated {
                dataset: format!("test{}", i),
            };
            manager.broadcast(event).await.unwrap();
        }

        // Check history is limited
        let history = manager.get_history(None).await;
        assert_eq!(history.len(), 10);

        // Check limited history
        let limited = manager.get_history(Some(5)).await;
        assert_eq!(limited.len(), 5);
    }
}
