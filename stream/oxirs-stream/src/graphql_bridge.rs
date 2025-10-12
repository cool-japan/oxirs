//! # GraphQL Subscription Bridge
//!
//! This module bridges oxirs-stream with GraphQL subscriptions, enabling real-time
//! GraphQL updates when stream events occur.
//!
//! ## Features
//!
//! - **Stream-to-GraphQL Event Mapping**: Convert stream events to GraphQL subscription updates
//! - **Query-based Filtering**: Only trigger subscriptions for relevant data changes
//! - **WebSocket Integration**: Seamless integration with GraphQL subscription servers
//! - **Multi-subscriber Support**: Broadcast events to multiple GraphQL clients
//! - **Performance Optimization**: Debouncing and batching for high-frequency updates
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use oxirs_stream::{StreamConfig, StreamProducer};
//! use oxirs_stream::graphql_bridge::{BridgeConfig, GraphQLBridge};
//!
//! async fn setup_graphql_streaming() -> anyhow::Result<()> {
//!     let config = StreamConfig::default();
//!     let _producer = StreamProducer::new(config).await?;
//!
//!     // Create GraphQL bridge and inspect initial statistics
//!     let bridge = GraphQLBridge::new(BridgeConfig::default());
//!     let stats = bridge.get_stats().await;
//!     println!("Active subscriptions: {}", stats.active_subscriptions);
//!
//!     Ok(())
//! }
//! ```

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::StreamEvent;

/// GraphQL bridge configuration
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Maximum queue size for pending updates
    pub max_queue_size: usize,
    /// Debounce duration for frequent updates
    pub debounce_duration: Duration,
    /// Enable batching of updates
    pub enable_batching: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Update interval for batched updates
    pub batch_interval: Duration,
    /// Enable query filtering (only send relevant updates)
    pub enable_query_filtering: bool,
    /// Maximum concurrent subscriptions
    pub max_subscriptions: usize,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            debounce_duration: Duration::from_millis(100),
            enable_batching: true,
            max_batch_size: 100,
            batch_interval: Duration::from_millis(500),
            enable_query_filtering: true,
            max_subscriptions: 1000,
        }
    }
}

/// GraphQL subscription bridge
pub struct GraphQLBridge {
    /// Configuration
    config: BridgeConfig,
    /// Registered GraphQL subscriptions
    subscriptions: Arc<RwLock<HashMap<String, GraphQLSubscription>>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<GraphQLUpdate>,
    /// Statistics
    stats: Arc<RwLock<BridgeStats>>,
    /// Debounce tracker
    debounce_tracker: Arc<RwLock<HashMap<String, Instant>>>,
    /// Batch buffer
    batch_buffer: Arc<RwLock<Vec<GraphQLUpdate>>>,
}

/// GraphQL subscription registration
#[derive(Debug, Clone)]
pub struct GraphQLSubscription {
    /// Subscription ID
    pub id: String,
    /// GraphQL query
    pub query: String,
    /// Variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Filter patterns (for optimization)
    pub filters: Vec<SubscriptionFilter>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub last_update: Option<DateTime<Utc>>,
    /// Update count
    pub update_count: u64,
}

/// Subscription filter for query optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubscriptionFilter {
    /// Filter by subject pattern
    SubjectPattern(String),
    /// Filter by predicate pattern
    PredicatePattern(String),
    /// Filter by object pattern
    ObjectPattern(String),
    /// Filter by graph
    GraphFilter(String),
    /// Custom filter expression
    CustomFilter(String),
}

/// GraphQL update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLUpdate {
    /// Update ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Update type
    pub update_type: GraphQLUpdateType,
    /// Data payload
    pub data: serde_json::Value,
    /// Affected subscriptions (IDs)
    pub subscriptions: Vec<String>,
}

/// GraphQL update types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphQLUpdateType {
    /// Data added
    DataAdded,
    /// Data removed
    DataRemoved,
    /// Data modified
    DataModified,
    /// Bulk update
    BulkUpdate,
    /// Query result changed
    QueryResultChanged,
}

/// Bridge statistics
#[derive(Debug, Clone, Default)]
pub struct BridgeStats {
    /// Total events processed
    pub events_processed: u64,
    /// Total updates sent
    pub updates_sent: u64,
    /// Updates batched
    pub updates_batched: u64,
    /// Updates debounced
    pub updates_debounced: u64,
    /// Active subscriptions
    pub active_subscriptions: usize,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
}

impl GraphQLBridge {
    /// Create a new GraphQL bridge
    pub fn new(config: BridgeConfig) -> Self {
        let (event_sender, _) = broadcast::channel(config.max_queue_size);

        let bridge = Self {
            config,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            stats: Arc::new(RwLock::new(BridgeStats::default())),
            debounce_tracker: Arc::new(RwLock::new(HashMap::new())),
            batch_buffer: Arc::new(RwLock::new(Vec::new())),
        };

        // Start background tasks
        if bridge.config.enable_batching {
            bridge.start_batch_processor();
        }

        bridge
    }

    /// Register a GraphQL subscription
    pub async fn register_subscription(&self, subscription: GraphQLSubscription) -> Result<String> {
        let mut subscriptions = self.subscriptions.write().await;

        if subscriptions.len() >= self.config.max_subscriptions {
            return Err(anyhow!("Maximum subscriptions limit reached"));
        }

        let id = subscription.id.clone();
        subscriptions.insert(id.clone(), subscription);

        // Update stats
        self.stats.write().await.active_subscriptions = subscriptions.len();

        info!("Registered GraphQL subscription: {}", id);
        Ok(id)
    }

    /// Unregister a GraphQL subscription
    pub async fn unregister_subscription(&self, subscription_id: &str) -> Result<()> {
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions
            .remove(subscription_id)
            .ok_or_else(|| anyhow!("Subscription not found"))?;

        // Update stats
        self.stats.write().await.active_subscriptions = subscriptions.len();

        info!("Unregistered GraphQL subscription: {}", subscription_id);
        Ok(())
    }

    /// Process a stream event and trigger GraphQL updates
    pub async fn process_stream_event(&self, event: &StreamEvent) -> Result<()> {
        let start_time = Instant::now();

        // Convert stream event to GraphQL update
        let update = self.convert_stream_event_to_update(event).await?;

        // Check debouncing
        if self.should_debounce(&update).await {
            self.stats.write().await.updates_debounced += 1;
            return Ok(());
        }

        // Update debounce tracker
        self.update_debounce_tracker(&update).await;

        // Handle batching if enabled
        if self.config.enable_batching {
            self.add_to_batch(update).await?;
        } else {
            self.send_update(update).await?;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.events_processed += 1;

        let processing_time = start_time.elapsed().as_millis() as f64;
        stats.avg_processing_time_ms = (stats.avg_processing_time_ms + processing_time) / 2.0;

        Ok(())
    }

    /// Convert stream event to GraphQL update
    async fn convert_stream_event_to_update(&self, event: &StreamEvent) -> Result<GraphQLUpdate> {
        let (update_type, data) = match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                GraphQLUpdateType::DataAdded,
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph,
                    "timestamp": metadata.timestamp,
                }),
            ),
            StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                GraphQLUpdateType::DataRemoved,
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph,
                    "timestamp": metadata.timestamp,
                }),
            ),
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                GraphQLUpdateType::DataAdded,
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph,
                    "timestamp": metadata.timestamp,
                }),
            ),
            StreamEvent::QuadRemoved {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => (
                GraphQLUpdateType::DataRemoved,
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph,
                    "timestamp": metadata.timestamp,
                }),
            ),
            StreamEvent::QueryResultAdded {
                query_id,
                result,
                metadata,
            } => (
                GraphQLUpdateType::QueryResultChanged,
                serde_json::json!({
                    "query_id": query_id,
                    "result": result.bindings,
                    "execution_time": result.execution_time.as_millis(),
                    "timestamp": metadata.timestamp,
                }),
            ),
            StreamEvent::QueryResultRemoved {
                query_id,
                result,
                metadata,
            } => (
                GraphQLUpdateType::QueryResultChanged,
                serde_json::json!({
                    "query_id": query_id,
                    "result": result.bindings,
                    "execution_time": result.execution_time.as_millis(),
                    "timestamp": metadata.timestamp,
                }),
            ),
            _ => (
                GraphQLUpdateType::BulkUpdate,
                serde_json::json!({
                    "message": "Bulk update occurred",
                    "timestamp": Utc::now(),
                }),
            ),
        };

        // Find relevant subscriptions
        let relevant_subscriptions = self.find_relevant_subscriptions(&data).await;

        Ok(GraphQLUpdate {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            update_type,
            data,
            subscriptions: relevant_subscriptions,
        })
    }

    /// Find subscriptions relevant to the update data
    async fn find_relevant_subscriptions(&self, data: &serde_json::Value) -> Vec<String> {
        let subscriptions = self.subscriptions.read().await;

        if !self.config.enable_query_filtering {
            // Return all subscription IDs
            return subscriptions.keys().cloned().collect();
        }

        let mut relevant = Vec::new();

        for (id, subscription) in subscriptions.iter() {
            if self.subscription_matches_data(subscription, data) {
                relevant.push(id.clone());
            }
        }

        relevant
    }

    /// Check if a subscription matches the update data
    fn subscription_matches_data(
        &self,
        subscription: &GraphQLSubscription,
        data: &serde_json::Value,
    ) -> bool {
        if subscription.filters.is_empty() {
            // No filters means subscribe to everything
            return true;
        }

        // Check if any filter matches
        for filter in &subscription.filters {
            match filter {
                SubscriptionFilter::SubjectPattern(pattern) => {
                    if let Some(subject) = data.get("subject").and_then(|v| v.as_str()) {
                        if self.pattern_matches(pattern, subject) {
                            return true;
                        }
                    }
                }
                SubscriptionFilter::PredicatePattern(pattern) => {
                    if let Some(predicate) = data.get("predicate").and_then(|v| v.as_str()) {
                        if self.pattern_matches(pattern, predicate) {
                            return true;
                        }
                    }
                }
                SubscriptionFilter::ObjectPattern(pattern) => {
                    if let Some(object) = data.get("object").and_then(|v| v.as_str()) {
                        if self.pattern_matches(pattern, object) {
                            return true;
                        }
                    }
                }
                SubscriptionFilter::GraphFilter(graph_uri) => {
                    if let Some(graph) = data.get("graph").and_then(|v| v.as_str()) {
                        if graph == graph_uri {
                            return true;
                        }
                    }
                }
                SubscriptionFilter::CustomFilter(_expr) => {
                    // Custom filter evaluation would go here
                    // For now, be conservative and include it
                    return true;
                }
            }
        }

        false
    }

    /// Simple pattern matching (supports wildcards)
    fn pattern_matches(&self, pattern: &str, value: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.contains('*') {
            // Simple wildcard matching
            let regex_pattern = pattern.replace('*', ".*");
            if let Ok(regex) = regex::Regex::new(&regex_pattern) {
                return regex.is_match(value);
            }
        }

        pattern == value
    }

    /// Check if update should be debounced
    async fn should_debounce(&self, update: &GraphQLUpdate) -> bool {
        let tracker = self.debounce_tracker.read().await;

        if let Some(last_update) = tracker.get(&update.id) {
            let elapsed = Instant::now().duration_since(*last_update);
            elapsed < self.config.debounce_duration
        } else {
            false
        }
    }

    /// Update debounce tracker
    async fn update_debounce_tracker(&self, update: &GraphQLUpdate) {
        let mut tracker = self.debounce_tracker.write().await;
        tracker.insert(update.id.clone(), Instant::now());
    }

    /// Add update to batch buffer
    async fn add_to_batch(&self, update: GraphQLUpdate) -> Result<()> {
        let mut buffer = self.batch_buffer.write().await;
        buffer.push(update);

        if buffer.len() >= self.config.max_batch_size {
            // Flush immediately if batch is full
            let updates = std::mem::take(&mut *buffer);
            drop(buffer);
            self.send_batch(updates).await?;
        }

        Ok(())
    }

    /// Send individual update
    async fn send_update(&self, update: GraphQLUpdate) -> Result<()> {
        match self.event_sender.send(update.clone()) {
            Ok(receiver_count) => {
                debug!("Sent GraphQL update to {} receivers", receiver_count);
                self.stats.write().await.updates_sent += 1;
                Ok(())
            }
            Err(e) => {
                warn!("No active GraphQL subscription receivers: {}", e);
                Ok(())
            }
        }
    }

    /// Send batch of updates
    async fn send_batch(&self, updates: Vec<GraphQLUpdate>) -> Result<()> {
        for update in updates {
            self.send_update(update).await?;
        }

        self.stats.write().await.updates_batched += 1;
        Ok(())
    }

    /// Start batch processor task
    fn start_batch_processor(&self) {
        let batch_buffer = Arc::clone(&self.batch_buffer);
        let event_sender = self.event_sender.clone();
        let batch_interval = self.config.batch_interval;
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = interval(batch_interval);

            loop {
                interval.tick().await;

                let updates = {
                    let mut buffer = batch_buffer.write().await;
                    if buffer.is_empty() {
                        continue;
                    }
                    std::mem::take(&mut *buffer)
                };

                if !updates.is_empty() {
                    debug!("Processing batch of {} updates", updates.len());

                    for update in updates {
                        if let Err(e) = event_sender.send(update) {
                            warn!("Failed to send batched update: {}", e);
                        } else {
                            stats.write().await.updates_sent += 1;
                        }
                    }

                    stats.write().await.updates_batched += 1;
                }
            }
        });
    }

    /// Subscribe to GraphQL updates
    pub fn subscribe(&self) -> broadcast::Receiver<GraphQLUpdate> {
        self.event_sender.subscribe()
    }

    /// Get bridge statistics
    pub async fn get_stats(&self) -> BridgeStats {
        self.stats.read().await.clone()
    }

    /// List all registered subscriptions
    pub async fn list_subscriptions(&self) -> Vec<String> {
        self.subscriptions.read().await.keys().cloned().collect()
    }

    /// Get subscription details
    pub async fn get_subscription(&self, id: &str) -> Option<GraphQLSubscription> {
        self.subscriptions.read().await.get(id).cloned()
    }
}

/// Helper function to create a simple GraphQL subscription
pub fn create_simple_subscription(
    query: String,
    filters: Vec<SubscriptionFilter>,
) -> GraphQLSubscription {
    GraphQLSubscription {
        id: uuid::Uuid::new_v4().to_string(),
        query,
        variables: HashMap::new(),
        filters,
        created_at: Utc::now(),
        last_update: None,
        update_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_config_default() {
        let config = BridgeConfig::default();
        assert_eq!(config.max_queue_size, 10000);
        assert!(config.enable_batching);
        assert!(config.enable_query_filtering);
    }

    #[tokio::test]
    async fn test_graphql_bridge_creation() {
        let bridge = GraphQLBridge::new(BridgeConfig::default());
        let stats = bridge.get_stats().await;

        assert_eq!(stats.active_subscriptions, 0);
        assert_eq!(stats.events_processed, 0);
    }

    #[tokio::test]
    async fn test_subscription_registration() {
        let bridge = GraphQLBridge::new(BridgeConfig::default());

        let subscription = create_simple_subscription(
            "subscription { triples { subject predicate object } }".to_string(),
            vec![],
        );

        let id = bridge.register_subscription(subscription).await.unwrap();

        assert!(!id.is_empty());

        let stats = bridge.get_stats().await;
        assert_eq!(stats.active_subscriptions, 1);
    }

    #[tokio::test]
    async fn test_pattern_matching() {
        let bridge = GraphQLBridge::new(BridgeConfig::default());

        assert!(bridge.pattern_matches("*", "anything"));
        assert!(bridge.pattern_matches("http://example.org/*", "http://example.org/resource"));
        assert!(!bridge.pattern_matches("http://example.org/*", "http://other.org/resource"));
        assert!(bridge.pattern_matches("exact_match", "exact_match"));
        assert!(!bridge.pattern_matches("exact_match", "different"));
    }

    #[test]
    fn test_subscription_filter_types() {
        let filter = SubscriptionFilter::SubjectPattern("http://example.org/*".to_string());
        matches!(filter, SubscriptionFilter::SubjectPattern(_));

        let filter2 = SubscriptionFilter::GraphFilter("http://example.org/graph1".to_string());
        matches!(filter2, SubscriptionFilter::GraphFilter(_));
    }
}
