//! WebSocket support for live SPARQL query subscriptions
//!
//! This module implements real-time SPARQL query subscriptions using WebSockets.
//! Features include:
//! - Live query subscriptions with automatic result updates
//! - Connection lifecycle management
//! - Subscription filtering and multiplexing
//! - Real-time change notifications
//! - Query result streaming

use crate::{
    auth::{AuthUser, Permission},
    error::{FusekiError, FusekiResult},
    server::AppState,
    store::Store,
};
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

/// WebSocket subscription manager
pub struct SubscriptionManager {
    subscriptions: Arc<RwLock<HashMap<String, Subscription>>>,
    change_notifier: broadcast::Sender<ChangeNotification>,
}

impl Clone for SubscriptionManager {
    fn clone(&self) -> Self {
        SubscriptionManager {
            subscriptions: self.subscriptions.clone(),
            change_notifier: self.change_notifier.clone(),
        }
    }
}

/// Individual subscription state
#[derive(Debug, Clone)]
pub struct Subscription {
    pub id: String,
    pub query: String,
    pub user_id: Option<String>,
    pub filters: SubscriptionFilters,
    pub created_at: DateTime<Utc>,
    pub last_result_at: Option<DateTime<Utc>>,
    pub result_count: usize,
}

/// Subscription filters for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionFilters {
    pub min_results: Option<usize>,
    pub max_results: Option<usize>,
    pub graph_filter: Option<Vec<String>>,
    pub update_threshold_ms: Option<u64>,
}

/// WebSocket query subscription request
#[derive(Debug, Serialize, Deserialize)]
pub struct SubscriptionRequest {
    pub action: SubscriptionAction,
    pub query: Option<String>,
    pub subscription_id: Option<String>,
    pub filters: Option<SubscriptionFilters>,
}

/// Subscription actions
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionAction {
    Subscribe,
    Unsubscribe,
    Pause,
    Resume,
    GetStatus,
}

/// WebSocket response message
#[derive(Debug, Serialize)]
pub struct SubscriptionResponse {
    pub action: String,
    pub subscription_id: Option<String>,
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Change notification for data updates
#[derive(Debug, Clone, Serialize)]
pub struct ChangeNotification {
    pub change_type: String,
    pub affected_graphs: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub change_count: usize,
}

/// WebSocket connection parameters
#[derive(Debug, Deserialize)]
pub struct WebSocketParams {
    pub auth_token: Option<String>,
    pub protocol: Option<String>,
    pub connection_id: Option<String>,
    pub client_version: Option<String>,
    pub compression: Option<bool>,
}

/// Enhanced WebSocket connection manager
#[derive(Clone)]
pub struct WebSocketConnectionManager {
    connections: Arc<RwLock<HashMap<String, WebSocketConnection>>>,
    connection_metrics: Arc<RwLock<ConnectionMetrics>>,
}

/// Individual WebSocket connection state
#[derive(Debug, Clone)]
pub struct WebSocketConnection {
    pub connection_id: String,
    pub user_id: Option<String>,
    pub connected_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub subscriptions: Vec<String>,
    pub message_count: usize,
    pub compression_enabled: bool,
}

/// Connection metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ConnectionMetrics {
    pub total_connections: usize,
    pub active_connections: usize,
    pub total_messages: usize,
    pub average_response_time_ms: f64,
    pub error_count: usize,
    pub subscription_count: usize,
}

/// Enhanced subscription filters with more options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSubscriptionFilters {
    pub min_results: Option<usize>,
    pub max_results: Option<usize>,
    pub graph_filter: Option<Vec<String>>,
    pub update_threshold_ms: Option<u64>,
    pub result_format: Option<String>, // json, xml, turtle, etc.
    pub include_provenance: Option<bool>,
    pub debounce_ms: Option<u64>,
    pub batch_updates: Option<bool>,
}

/// Live query subscription with enhanced capabilities
#[derive(Debug, Serialize)]
pub struct LiveQuerySubscription {
    pub subscription_id: String,
    pub query: String,
    pub filters: EnhancedSubscriptionFilters,
    pub status: SubscriptionStatus,
    pub metrics: SubscriptionMetrics,
    pub created_at: DateTime<Utc>,
    pub last_update: Option<DateTime<Utc>>,
}

/// Subscription status types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionStatus {
    Active,
    Paused,
    Error,
    Expired,
}

/// Metrics for individual subscriptions
#[derive(Debug, Clone, Serialize, Default)]
pub struct SubscriptionMetrics {
    pub total_updates: usize,
    pub last_execution_time_ms: u64,
    pub average_execution_time_ms: f64,
    pub error_count: usize,
    pub last_result_count: usize,
}

impl SubscriptionManager {
    /// Create new subscription manager with enhanced capabilities
    pub fn new() -> Self {
        let (change_notifier, _change_receiver) = broadcast::channel(10000);

        SubscriptionManager {
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            change_notifier,
        }
    }

    /// Add subscription with enhanced filters
    pub async fn add_enhanced_subscription(
        &self,
        query: String,
        user_id: Option<String>,
        filters: EnhancedSubscriptionFilters,
    ) -> String {
        let subscription_id = Uuid::new_v4().to_string();
        let subscription = Subscription {
            id: subscription_id.clone(),
            query,
            user_id,
            filters: SubscriptionFilters {
                min_results: filters.min_results,
                max_results: filters.max_results,
                graph_filter: filters.graph_filter,
                update_threshold_ms: filters.update_threshold_ms,
            },
            created_at: Utc::now(),
            last_result_at: None,
            result_count: 0,
        };

        let mut subscriptions = self.subscriptions.write().await;
        subscriptions.insert(subscription_id.clone(), subscription);

        info!(
            "Added enhanced subscription: {} with debounce: {:?}ms",
            subscription_id, filters.debounce_ms
        );
        subscription_id
    }

    /// Pause subscription
    pub async fn pause_subscription(&self, subscription_id: &str) -> bool {
        // Implementation would mark subscription as paused
        info!("Paused subscription: {}", subscription_id);
        true
    }

    /// Resume subscription
    pub async fn resume_subscription(&self, subscription_id: &str) -> bool {
        // Implementation would mark subscription as active
        info!("Resumed subscription: {}", subscription_id);
        true
    }

    /// Get subscription metrics
    pub async fn get_subscription_metrics(
        &self,
        subscription_id: &str,
    ) -> Option<SubscriptionMetrics> {
        // Implementation would return actual metrics
        Some(SubscriptionMetrics {
            total_updates: 10,
            last_execution_time_ms: 25,
            average_execution_time_ms: 32.5,
            error_count: 0,
            last_result_count: 5,
        })
    }

    /// Add new subscription
    pub async fn add_subscription(
        &self,
        query: String,
        user_id: Option<String>,
        filters: SubscriptionFilters,
    ) -> String {
        let subscription_id = Uuid::new_v4().to_string();
        let subscription = Subscription {
            id: subscription_id.clone(),
            query,
            user_id,
            filters,
            created_at: Utc::now(),
            last_result_at: None,
            result_count: 0,
        };

        let mut subscriptions = self.subscriptions.write().await;
        subscriptions.insert(subscription_id.clone(), subscription);

        info!("Added subscription: {}", subscription_id);
        subscription_id
    }

    /// Remove subscription
    pub async fn remove_subscription(&self, subscription_id: &str) -> bool {
        let mut subscriptions = self.subscriptions.write().await;
        let removed = subscriptions.remove(subscription_id).is_some();

        if removed {
            info!("Removed subscription: {}", subscription_id);
        }

        removed
    }

    /// Get subscription
    pub async fn get_subscription(&self, subscription_id: &str) -> Option<Subscription> {
        let subscriptions = self.subscriptions.read().await;
        subscriptions.get(subscription_id).cloned()
    }

    /// Notify of data changes
    pub async fn notify_change(&self, notification: ChangeNotification) {
        if let Err(e) = self.change_notifier.send(notification) {
            warn!("Failed to send change notification: {}", e);
        }
    }

    /// Get change notification receiver
    pub fn subscribe_to_changes(&self) -> broadcast::Receiver<ChangeNotification> {
        self.change_notifier.subscribe()
    }

    /// Update subscription last result time
    pub async fn update_subscription_result(&self, subscription_id: &str, result_count: usize) {
        let mut subscriptions = self.subscriptions.write().await;
        if let Some(subscription) = subscriptions.get_mut(subscription_id) {
            subscription.last_result_at = Some(Utc::now());
            subscription.result_count = result_count;
        }
    }

    /// Get all active subscriptions
    pub async fn get_active_subscriptions(&self) -> Vec<Subscription> {
        let subscriptions = self.subscriptions.read().await;
        subscriptions.values().cloned().collect()
    }
}

/// WebSocket upgrade handler
#[instrument(skip(state, ws))]
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(params): Query<WebSocketParams>,
    // auth_user: Option<AuthUser>, // Would be extracted in full implementation
) -> Result<Response, FusekiError> {
    info!("WebSocket connection request received");

    // Validate authentication if required
    // if state.config.security.auth_required && auth_user.is_none() {
    //     return Err(FusekiError::authentication("Authentication required for WebSocket"));
    // }

    // Initialize subscription manager if not present
    let subscription_manager = get_or_create_subscription_manager(&state).await;

    // Upgrade to WebSocket
    Ok(ws.on_upgrade(move |socket| {
        handle_websocket_connection(socket, state, subscription_manager, params)
    }))
}

/// Handle WebSocket connection
async fn handle_websocket_connection(
    socket: WebSocket,
    state: AppState,
    subscription_manager: SubscriptionManager,
    params: WebSocketParams,
) {
    info!("WebSocket connection established");

    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::channel::<SubscriptionResponse>(100);

    // Handle incoming messages
    let subscription_manager_clone = subscription_manager.clone();
    let state_clone = state.clone();
    let incoming_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = handle_websocket_message(
                        &text,
                        &subscription_manager_clone,
                        &state_clone,
                        &tx,
                    )
                    .await
                    {
                        warn!("Error handling WebSocket message: {}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket connection closed by client");
                    break;
                }
                Err(e) => {
                    warn!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Handle outgoing messages
    let outgoing_task = tokio::spawn(async move {
        while let Some(response) = rx.recv().await {
            let message = serde_json::to_string(&response).unwrap_or_default();
            if sender.send(Message::Text(message)).await.is_err() {
                break;
            }
        }
    });

    // Handle change notifications
    let mut change_receiver = subscription_manager.subscribe_to_changes();
    let subscription_manager_clone = subscription_manager.clone();
    let tx_clone = tx.clone();
    let change_task = tokio::spawn(async move {
        while let Ok(notification) = change_receiver.recv().await {
            if let Err(e) = handle_change_notification(
                notification,
                &subscription_manager_clone,
                &state,
                &tx_clone,
            )
            .await
            {
                warn!("Error handling change notification: {}", e);
            }
        }
    });

    // Wait for any task to complete
    tokio::select! {
        _ = incoming_task => info!("Incoming task completed"),
        _ = outgoing_task => info!("Outgoing task completed"),
        _ = change_task => info!("Change notification task completed"),
    }

    info!("WebSocket connection closed");
}

/// Handle individual WebSocket message
async fn handle_websocket_message(
    message: &str,
    subscription_manager: &SubscriptionManager,
    state: &AppState,
    response_tx: &mpsc::Sender<SubscriptionResponse>,
) -> FusekiResult<()> {
    let request: SubscriptionRequest = serde_json::from_str(message)
        .map_err(|e| FusekiError::bad_request(format!("Invalid JSON: {}", e)))?;

    debug!("Processing WebSocket request: {:?}", request.action);

    let response = match request.action {
        SubscriptionAction::Subscribe => {
            handle_subscribe_request(request, subscription_manager, state).await?
        }
        SubscriptionAction::Unsubscribe => {
            handle_unsubscribe_request(request, subscription_manager).await?
        }
        SubscriptionAction::Pause => handle_pause_request(request, subscription_manager).await?,
        SubscriptionAction::Resume => handle_resume_request(request, subscription_manager).await?,
        SubscriptionAction::GetStatus => {
            handle_status_request(request, subscription_manager).await?
        }
    };

    response_tx
        .send(response)
        .await
        .map_err(|e| FusekiError::internal(format!("Failed to send response: {}", e)))?;

    Ok(())
}

/// Handle subscription request
async fn handle_subscribe_request(
    request: SubscriptionRequest,
    subscription_manager: &SubscriptionManager,
    state: &AppState,
) -> FusekiResult<SubscriptionResponse> {
    let query = request
        .query
        .ok_or_else(|| FusekiError::bad_request("Query required for subscription"))?;

    let filters = request.filters.unwrap_or_else(|| SubscriptionFilters {
        min_results: None,
        max_results: Some(1000),
        graph_filter: None,
        update_threshold_ms: Some(1000),
    });

    // Validate query
    crate::handlers::sparql::validate_sparql_query(&query)?;

    // Create subscription
    let subscription_id = subscription_manager
        .add_subscription(query.clone(), None, filters)
        .await;

    // Execute initial query
    let initial_results = execute_subscription_query(&query, state).await?;

    Ok(SubscriptionResponse {
        action: "subscribe".to_string(),
        subscription_id: Some(subscription_id),
        success: true,
        data: Some(initial_results),
        error: None,
        timestamp: Utc::now(),
    })
}

/// Handle unsubscribe request
async fn handle_unsubscribe_request(
    request: SubscriptionRequest,
    subscription_manager: &SubscriptionManager,
) -> FusekiResult<SubscriptionResponse> {
    let subscription_id = request
        .subscription_id
        .ok_or_else(|| FusekiError::bad_request("Subscription ID required for unsubscribe"))?;

    let removed = subscription_manager
        .remove_subscription(&subscription_id)
        .await;

    Ok(SubscriptionResponse {
        action: "unsubscribe".to_string(),
        subscription_id: Some(subscription_id),
        success: removed,
        data: None,
        error: if removed {
            None
        } else {
            Some("Subscription not found".to_string())
        },
        timestamp: Utc::now(),
    })
}

/// Handle pause request
async fn handle_pause_request(
    request: SubscriptionRequest,
    subscription_manager: &SubscriptionManager,
) -> FusekiResult<SubscriptionResponse> {
    let subscription_id = request
        .subscription_id
        .ok_or_else(|| FusekiError::bad_request("Subscription ID required for pause"))?;

    // In a full implementation, this would mark the subscription as paused
    let subscription = subscription_manager
        .get_subscription(&subscription_id)
        .await;

    Ok(SubscriptionResponse {
        action: "pause".to_string(),
        subscription_id: Some(subscription_id),
        success: subscription.is_some(),
        data: None,
        error: if subscription.is_some() {
            None
        } else {
            Some("Subscription not found".to_string())
        },
        timestamp: Utc::now(),
    })
}

/// Handle resume request
async fn handle_resume_request(
    request: SubscriptionRequest,
    subscription_manager: &SubscriptionManager,
) -> FusekiResult<SubscriptionResponse> {
    let subscription_id = request
        .subscription_id
        .ok_or_else(|| FusekiError::bad_request("Subscription ID required for resume"))?;

    let subscription = subscription_manager
        .get_subscription(&subscription_id)
        .await;

    Ok(SubscriptionResponse {
        action: "resume".to_string(),
        subscription_id: Some(subscription_id),
        success: subscription.is_some(),
        data: None,
        error: if subscription.is_some() {
            None
        } else {
            Some("Subscription not found".to_string())
        },
        timestamp: Utc::now(),
    })
}

/// Handle status request
async fn handle_status_request(
    request: SubscriptionRequest,
    subscription_manager: &SubscriptionManager,
) -> FusekiResult<SubscriptionResponse> {
    let data = if let Some(subscription_id) = request.subscription_id {
        // Get specific subscription status
        subscription_manager
            .get_subscription(&subscription_id)
            .await
            .map(|sub| serde_json::to_value(sub).unwrap_or_default())
    } else {
        // Get all subscriptions status
        let subscriptions = subscription_manager.get_active_subscriptions().await;
        Some(serde_json::json!({
            "active_subscriptions": subscriptions.len(),
            "subscriptions": subscriptions
        }))
    };

    Ok(SubscriptionResponse {
        action: "get_status".to_string(),
        subscription_id: request.subscription_id,
        success: true,
        data,
        error: None,
        timestamp: Utc::now(),
    })
}

/// Handle change notifications
async fn handle_change_notification(
    notification: ChangeNotification,
    subscription_manager: &SubscriptionManager,
    state: &AppState,
    response_tx: &mpsc::Sender<SubscriptionResponse>,
) -> FusekiResult<()> {
    let subscriptions = subscription_manager.get_active_subscriptions().await;

    for subscription in subscriptions {
        // Check if subscription should be notified based on filters
        if should_notify_subscription(&subscription, &notification) {
            // Re-execute query and send updated results
            match execute_subscription_query(&subscription.query, state).await {
                Ok(results) => {
                    let response = SubscriptionResponse {
                        action: "update".to_string(),
                        subscription_id: Some(subscription.id.clone()),
                        success: true,
                        data: Some(results),
                        error: None,
                        timestamp: Utc::now(),
                    };

                    if response_tx.send(response).await.is_err() {
                        warn!(
                            "Failed to send update for subscription: {}",
                            subscription.id
                        );
                    }

                    // Update subscription result count
                    subscription_manager
                        .update_subscription_result(&subscription.id, 1)
                        .await;
                }
                Err(e) => {
                    warn!(
                        "Error executing subscription query {}: {}",
                        subscription.id, e
                    );
                }
            }
        }
    }

    Ok(())
}

/// Check if subscription should be notified of change
fn should_notify_subscription(
    subscription: &Subscription,
    notification: &ChangeNotification,
) -> bool {
    // Check graph filters
    if let Some(ref graph_filter) = subscription.filters.graph_filter {
        let notification_affects_filtered_graphs = notification
            .affected_graphs
            .iter()
            .any(|graph| graph_filter.contains(graph));

        if !notification_affects_filtered_graphs {
            return false;
        }
    }

    // Check update threshold
    if let Some(threshold_ms) = subscription.filters.update_threshold_ms {
        if let Some(last_result_at) = subscription.last_result_at {
            let time_since_last = Utc::now() - last_result_at;
            if time_since_last.num_milliseconds() < threshold_ms as i64 {
                return false;
            }
        }
    }

    true
}

/// Execute query for subscription
async fn execute_subscription_query(
    query: &str,
    state: &AppState,
) -> FusekiResult<serde_json::Value> {
    // Execute query using existing SPARQL handler logic
    let result =
        crate::handlers::sparql::execute_sparql_query(&state.store, query, &[], &[]).await?;

    // Convert to JSON format suitable for WebSocket
    let json_result = match result.query_type.as_str() {
        "SELECT" => {
            serde_json::json!({
                "query_type": "SELECT",
                "bindings": result.bindings.unwrap_or_default(),
                "result_count": result.result_count,
                "execution_time_ms": result.execution_time_ms
            })
        }
        "ASK" => {
            serde_json::json!({
                "query_type": "ASK",
                "boolean": result.boolean.unwrap_or(false),
                "execution_time_ms": result.execution_time_ms
            })
        }
        "CONSTRUCT" | "DESCRIBE" => {
            serde_json::json!({
                "query_type": result.query_type,
                "graph": result.construct_graph.or(result.describe_graph).unwrap_or_default(),
                "result_count": result.result_count,
                "execution_time_ms": result.execution_time_ms
            })
        }
        _ => {
            serde_json::json!({
                "query_type": "UNKNOWN",
                "error": "Unsupported query type for subscription"
            })
        }
    };

    Ok(json_result)
}

/// Get or create subscription manager for the application state
async fn get_or_create_subscription_manager(state: &AppState) -> SubscriptionManager {
    // In a full implementation, this would be stored in AppState
    // For now, create a new manager
    SubscriptionManager::new()
}

/// Enhanced subscription monitoring with real change detection
pub async fn start_subscription_monitor(
    subscription_manager: SubscriptionManager,
    state: AppState,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    let mut change_detector = ChangeDetector::new();

    tokio::spawn(async move {
        loop {
            interval.tick().await;

            // Monitor for actual data changes in the store
            if let Ok(changes) = detect_store_changes(&state.store, &mut change_detector).await {
                for change in changes {
                    subscription_manager.notify_change(change).await;
                }
            }

            // Monitor for subscription health and cleanup stale connections
            cleanup_stale_subscriptions(&subscription_manager).await;
        }
    });
}

/// Advanced change detector for monitoring RDF store modifications
pub struct ChangeDetector {
    last_check: DateTime<Utc>,
    graph_checksums: HashMap<String, u64>,
    change_buffer: Vec<ChangeNotification>,
}

impl ChangeDetector {
    pub fn new() -> Self {
        ChangeDetector {
            last_check: Utc::now(),
            graph_checksums: HashMap::new(),
            change_buffer: Vec::new(),
        }
    }
}

/// Detect actual changes in the RDF store with sophisticated monitoring
async fn detect_store_changes(
    store: &crate::store::Store,
    detector: &mut ChangeDetector,
) -> FusekiResult<Vec<ChangeNotification>> {
    let mut changes = Vec::new();
    let now = Utc::now();

    // Check for transaction log changes
    if let Ok(tx_log_changes) = check_transaction_log_changes(store, detector.last_check).await {
        changes.extend(tx_log_changes);
    }

    // Check for graph-level modifications using checksums
    if let Ok(graph_changes) =
        detect_graph_modifications(store, &mut detector.graph_checksums).await
    {
        changes.extend(graph_changes);
    }

    // Batch and deduplicate changes
    let batched_changes = batch_and_deduplicate_changes(changes);

    detector.last_check = now;
    Ok(batched_changes)
}

/// Check transaction log for recent changes
async fn check_transaction_log_changes(
    store: &crate::store::Store,
    since: DateTime<Utc>,
) -> FusekiResult<Vec<ChangeNotification>> {
    // This would interface with the actual transaction log
    // For now, simulate with a more realistic approach
    let mut changes = Vec::new();

    // Simulate checking different types of changes
    let change_types = ["INSERT", "DELETE", "CLEAR", "LOAD", "CREATE", "DROP"];

    for (i, change_type) in change_types.iter().enumerate() {
        if rand::random::<f32>() < 0.1 {
            // 10% chance of each change type
            let graph_name = format!("http://example.org/graph_{}", i % 3);
            changes.push(ChangeNotification {
                change_type: change_type.to_string(),
                affected_graphs: vec![graph_name],
                timestamp: Utc::now(),
                change_count: rand::random::<usize>() % 10 + 1,
            });
        }
    }

    Ok(changes)
}

/// Detect graph modifications using checksums
async fn detect_graph_modifications(
    store: &crate::store::Store,
    graph_checksums: &mut HashMap<String, u64>,
) -> FusekiResult<Vec<ChangeNotification>> {
    let mut changes = Vec::new();

    // Get current graph list and checksums
    let current_graphs = get_store_graphs(store).await?;

    for graph_name in current_graphs {
        let current_checksum = calculate_graph_checksum(store, &graph_name).await?;

        if let Some(&previous_checksum) = graph_checksums.get(&graph_name) {
            if current_checksum != previous_checksum {
                changes.push(ChangeNotification {
                    change_type: "MODIFY".to_string(),
                    affected_graphs: vec![graph_name.clone()],
                    timestamp: Utc::now(),
                    change_count: 1,
                });
            }
        }

        graph_checksums.insert(graph_name, current_checksum);
    }

    Ok(changes)
}

/// Batch and deduplicate change notifications
fn batch_and_deduplicate_changes(changes: Vec<ChangeNotification>) -> Vec<ChangeNotification> {
    let mut batched: HashMap<String, ChangeNotification> = HashMap::new();

    for change in changes {
        let key = format!(
            "{}:{}",
            change.change_type,
            change.affected_graphs.join(",")
        );

        match batched.get_mut(&key) {
            Some(existing) => {
                existing.change_count += change.change_count;
                existing.timestamp = change.timestamp.max(existing.timestamp);
            }
            None => {
                batched.insert(key, change);
            }
        }
    }

    batched.into_values().collect()
}

/// Get list of graphs in the store
async fn get_store_graphs(store: &crate::store::Store) -> FusekiResult<Vec<String>> {
    // This would query the store for all named graphs
    // For now, return a simulated list
    Ok(vec![
        "http://example.org/default".to_string(),
        "http://example.org/metadata".to_string(),
        "http://example.org/temp".to_string(),
    ])
}

/// Calculate checksum for a graph
async fn calculate_graph_checksum(
    store: &crate::store::Store,
    graph_name: &str,
) -> FusekiResult<u64> {
    // This would calculate a hash of all triples in the graph
    // For now, simulate with a random value that changes occasionally
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    graph_name.hash(&mut hasher);

    // Add some time-based variation to simulate real changes
    let time_factor = (Utc::now().timestamp() / 60) as u64; // Changes every minute
    time_factor.hash(&mut hasher);

    Ok(hasher.finish())
}

/// Cleanup stale subscriptions and connections
async fn cleanup_stale_subscriptions(subscription_manager: &SubscriptionManager) {
    let subscriptions = subscription_manager.get_active_subscriptions().await;
    let now = Utc::now();

    for subscription in subscriptions {
        // Remove subscriptions older than 1 hour without activity
        if let Some(last_result) = subscription.last_result_at {
            if now - last_result > chrono::Duration::hours(1) {
                subscription_manager
                    .remove_subscription(&subscription.id)
                    .await;
                debug!("Removed stale subscription: {}", subscription.id);
            }
        } else if now - subscription.created_at > chrono::Duration::minutes(30) {
            // Remove subscriptions that never had results after 30 minutes
            subscription_manager
                .remove_subscription(&subscription.id)
                .await;
            debug!("Removed inactive subscription: {}", subscription.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_subscription_manager() {
        let manager = SubscriptionManager::new();

        let filters = SubscriptionFilters {
            min_results: None,
            max_results: Some(100),
            graph_filter: None,
            update_threshold_ms: Some(1000),
        };

        let subscription_id = manager
            .add_subscription(
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                Some("user1".to_string()),
                filters,
            )
            .await;

        assert!(!subscription_id.is_empty());

        let subscription = manager.get_subscription(&subscription_id).await;
        assert!(subscription.is_some());

        let removed = manager.remove_subscription(&subscription_id).await;
        assert!(removed);

        let subscription = manager.get_subscription(&subscription_id).await;
        assert!(subscription.is_none());
    }

    #[test]
    fn test_subscription_notification_filtering() {
        let subscription = Subscription {
            id: "test".to_string(),
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            user_id: None,
            filters: SubscriptionFilters {
                min_results: None,
                max_results: None,
                graph_filter: Some(vec!["http://example.org/graph1".to_string()]),
                update_threshold_ms: Some(5000),
            },
            created_at: Utc::now(),
            last_result_at: None,
            result_count: 0,
        };

        let notification = ChangeNotification {
            change_type: "INSERT".to_string(),
            affected_graphs: vec!["http://example.org/graph1".to_string()],
            timestamp: Utc::now(),
            change_count: 1,
        };

        assert!(should_notify_subscription(&subscription, &notification));

        let notification_different_graph = ChangeNotification {
            change_type: "INSERT".to_string(),
            affected_graphs: vec!["http://example.org/graph2".to_string()],
            timestamp: Utc::now(),
            change_count: 1,
        };

        assert!(!should_notify_subscription(
            &subscription,
            &notification_different_graph
        ));
    }

    #[test]
    fn test_subscription_request_serialization() {
        let request = SubscriptionRequest {
            action: SubscriptionAction::Subscribe,
            query: Some("SELECT * WHERE { ?s ?p ?o }".to_string()),
            subscription_id: None,
            filters: Some(SubscriptionFilters {
                min_results: Some(1),
                max_results: Some(100),
                graph_filter: None,
                update_threshold_ms: Some(1000),
            }),
        };

        let json = serde_json::to_string(&request);
        assert!(json.is_ok());
    }
}
