// Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Federated Subscription Support
//!
//! This module provides subscription support across federated GraphQL services,
//! enabling real-time updates to be aggregated and routed from multiple subgraphs.

use anyhow::{anyhow, Result};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

/// Subscription event from a subgraph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionEvent {
    /// Unique event ID
    pub id: String,
    /// Source subgraph/service name
    pub source: String,
    /// Event payload (GraphQL response)
    pub payload: serde_json::Value,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl SubscriptionEvent {
    /// Create a new subscription event
    pub fn new(source: String, payload: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source,
            payload,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Subscription routing strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionRoutingStrategy {
    /// Send to all matching subgraphs
    Broadcast,
    /// Send to first matching subgraph only
    Single,
    /// Round-robin across subgraphs
    RoundRobin,
    /// Route based on field ownership
    FieldBased,
}

/// Subscription aggregation strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Merge events from all sources
    Merge,
    /// Take first event
    First,
    /// Take latest event
    Latest,
    /// Custom aggregation logic
    Custom,
}

/// Federated subscription configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSubscriptionConfig {
    /// Subscription name/operation
    pub operation: String,
    /// Routing strategy
    pub routing: SubscriptionRoutingStrategy,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
    /// Target subgraphs
    pub targets: Vec<String>,
    /// Buffer size for events
    pub buffer_size: usize,
    /// Timeout for subscription setup (seconds)
    pub setup_timeout_secs: u64,
}

impl Default for FederatedSubscriptionConfig {
    fn default() -> Self {
        Self {
            operation: String::new(),
            routing: SubscriptionRoutingStrategy::Broadcast,
            aggregation: AggregationStrategy::Merge,
            targets: Vec::new(),
            buffer_size: 100,
            setup_timeout_secs: 30,
        }
    }
}

/// Active subscription tracking
#[derive(Debug)]
struct ActiveSubscription {
    /// Subscription ID
    id: String,
    /// Configuration
    config: FederatedSubscriptionConfig,
    /// Event sender
    sender: mpsc::UnboundedSender<SubscriptionEvent>,
    /// Subgraph connection IDs
    subgraph_connections: HashMap<String, String>,
}

/// Federated subscription manager
pub struct FederatedSubscriptionManager {
    /// Active subscriptions
    subscriptions: Arc<RwLock<HashMap<String, ActiveSubscription>>>,
    /// Subscription configurations
    configs: Arc<RwLock<HashMap<String, FederatedSubscriptionConfig>>>,
    /// Event handlers
    event_handlers: Arc<RwLock<Vec<Box<dyn SubscriptionEventHandler + Send + Sync>>>>,
}

/// Trait for handling subscription events
pub trait SubscriptionEventHandler {
    /// Handle a subscription event
    fn handle(&self, event: &SubscriptionEvent) -> Result<()>;
}

impl FederatedSubscriptionManager {
    /// Create a new federated subscription manager
    pub fn new() -> Self {
        Self {
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            configs: Arc::new(RwLock::new(HashMap::new())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a subscription configuration
    pub async fn register_subscription(&self, config: FederatedSubscriptionConfig) -> Result<()> {
        let mut configs = self.configs.write().await;
        configs.insert(config.operation.clone(), config);
        Ok(())
    }

    /// Create a new federated subscription
    pub async fn subscribe(
        &self,
        operation: String,
        variables: HashMap<String, serde_json::Value>,
    ) -> Result<Pin<Box<dyn Stream<Item = SubscriptionEvent> + Send>>> {
        let configs = self.configs.read().await;
        let config = configs
            .get(&operation)
            .ok_or_else(|| anyhow!("Subscription '{}' not configured", operation))?
            .clone();
        drop(configs);

        let subscription_id = Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::unbounded_channel();

        // Create active subscription entry
        let active_sub = ActiveSubscription {
            id: subscription_id.clone(),
            config: config.clone(),
            sender: tx.clone(),
            subgraph_connections: HashMap::new(),
        };

        // Store active subscription
        {
            let mut subs = self.subscriptions.write().await;
            subs.insert(subscription_id.clone(), active_sub);
        }

        // Start subscription on target subgraphs
        self.start_subgraph_subscriptions(&subscription_id, &config, &variables)
            .await?;

        // Create event stream from receiver
        let stream = UnboundedReceiverStream::new(rx);

        Ok(Box::pin(stream))
    }

    /// Start subscriptions on target subgraphs
    async fn start_subgraph_subscriptions(
        &self,
        subscription_id: &str,
        config: &FederatedSubscriptionConfig,
        _variables: &HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // In a real implementation, this would connect to each subgraph
        // and establish WebSocket/SSE subscriptions
        // For now, we just create placeholder connection IDs
        let mut subs = self.subscriptions.write().await;
        if let Some(active_sub) = subs.get_mut(subscription_id) {
            for target in &config.targets {
                let connection_id = Uuid::new_v4().to_string();
                active_sub
                    .subgraph_connections
                    .insert(target.clone(), connection_id);
            }
        }
        Ok(())
    }

    /// Publish an event to a subscription
    pub async fn publish_event(
        &self,
        subscription_id: &str,
        event: SubscriptionEvent,
    ) -> Result<()> {
        // Notify event handlers
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            if let Err(e) = handler.handle(&event) {
                tracing::warn!("Event handler error: {}", e);
            }
        }

        // Send to subscription
        let subs = self.subscriptions.read().await;
        if let Some(active_sub) = subs.get(subscription_id) {
            active_sub
                .sender
                .send(event)
                .map_err(|_| anyhow!("Failed to send event to subscription"))?;
        }
        Ok(())
    }

    /// Unsubscribe from a federated subscription
    pub async fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        let mut subs = self.subscriptions.write().await;
        subs.remove(subscription_id);
        Ok(())
    }

    /// List active subscriptions
    pub async fn list_active_subscriptions(&self) -> Vec<String> {
        let subs = self.subscriptions.read().await;
        subs.keys().cloned().collect()
    }

    /// Get subscription info
    pub async fn get_subscription_info(&self, subscription_id: &str) -> Option<SubscriptionInfo> {
        let subs = self.subscriptions.read().await;
        subs.get(subscription_id).map(|sub| SubscriptionInfo {
            id: sub.id.clone(),
            operation: sub.config.operation.clone(),
            targets: sub.config.targets.clone(),
            active_connections: sub.subgraph_connections.len(),
        })
    }

    /// Register an event handler
    pub async fn register_event_handler(
        &self,
        handler: Box<dyn SubscriptionEventHandler + Send + Sync>,
    ) -> Result<()> {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
        Ok(())
    }

    /// Aggregate events from multiple sources
    pub async fn aggregate_events(
        &self,
        events: Vec<SubscriptionEvent>,
        strategy: AggregationStrategy,
    ) -> Result<SubscriptionEvent> {
        match strategy {
            AggregationStrategy::Merge => self.merge_events(events).await,
            AggregationStrategy::First => events
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("No events to aggregate")),
            AggregationStrategy::Latest => events
                .into_iter()
                .max_by_key(|e| e.timestamp)
                .ok_or_else(|| anyhow!("No events to aggregate")),
            AggregationStrategy::Custom => {
                // Custom aggregation would be implemented via event handlers
                self.merge_events(events).await
            }
        }
    }

    /// Merge events from multiple sources
    async fn merge_events(&self, events: Vec<SubscriptionEvent>) -> Result<SubscriptionEvent> {
        if events.is_empty() {
            return Err(anyhow!("No events to merge"));
        }

        // Merge payloads
        let mut merged_payload = serde_json::Map::new();
        for event in &events {
            if let serde_json::Value::Object(obj) = &event.payload {
                for (key, value) in obj {
                    merged_payload.insert(key.clone(), value.clone());
                }
            }
        }

        // Use latest timestamp
        let latest_timestamp = events.iter().map(|e| e.timestamp).max().unwrap();

        Ok(SubscriptionEvent {
            id: Uuid::new_v4().to_string(),
            source: "merged".to_string(),
            payload: serde_json::Value::Object(merged_payload),
            timestamp: latest_timestamp,
            metadata: HashMap::new(),
        })
    }

    /// Route subscription to appropriate subgraphs
    pub fn route_subscription(
        &self,
        config: &FederatedSubscriptionConfig,
        _context: &RoutingContext,
    ) -> Vec<String> {
        match config.routing {
            SubscriptionRoutingStrategy::Broadcast => config.targets.clone(),
            SubscriptionRoutingStrategy::Single => {
                config.targets.first().cloned().into_iter().collect()
            }
            SubscriptionRoutingStrategy::RoundRobin => {
                // In a real implementation, this would maintain round-robin state
                config.targets.first().cloned().into_iter().collect()
            }
            SubscriptionRoutingStrategy::FieldBased => {
                // In a real implementation, this would analyze field ownership
                config.targets.clone()
            }
        }
    }
}

impl Default for FederatedSubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Routing context for subscription routing decisions
#[derive(Debug, Default)]
pub struct RoutingContext {
    /// Field selections
    pub fields: Vec<String>,
    /// Query variables
    pub variables: HashMap<String, serde_json::Value>,
    /// Additional context
    pub metadata: HashMap<String, String>,
}

/// Subscription information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionInfo {
    /// Subscription ID
    pub id: String,
    /// Operation name
    pub operation: String,
    /// Target subgraphs
    pub targets: Vec<String>,
    /// Number of active connections
    pub active_connections: usize,
}

/// Simple logging event handler
pub struct LoggingEventHandler;

impl SubscriptionEventHandler for LoggingEventHandler {
    fn handle(&self, event: &SubscriptionEvent) -> Result<()> {
        tracing::info!(
            "Subscription event from {}: {} at {}",
            event.source,
            event.id,
            event.timestamp
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[test]
    fn test_subscription_event_creation() {
        let payload = serde_json::json!({"data": {"field": "value"}});
        let event = SubscriptionEvent::new("test_service".to_string(), payload.clone());
        assert_eq!(event.source, "test_service");
        assert_eq!(event.payload, payload);
        assert!(!event.id.is_empty());
    }

    #[test]
    fn test_subscription_event_with_metadata() {
        let payload = serde_json::json!({"data": {}});
        let event = SubscriptionEvent::new("test".to_string(), payload)
            .with_metadata("key".to_string(), "value".to_string());
        assert_eq!(event.metadata.get("key"), Some(&"value".to_string()));
    }

    #[tokio::test]
    async fn test_register_subscription() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test_subscription".to_string(),
            targets: vec!["service1".to_string()],
            ..Default::default()
        };

        manager.register_subscription(config).await.unwrap();
        // Verify registration doesn't panic
    }

    #[tokio::test]
    async fn test_subscribe() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test_sub".to_string(),
            targets: vec!["service1".to_string()],
            ..Default::default()
        };
        manager.register_subscription(config).await.unwrap();

        let variables = HashMap::new();
        let stream = manager.subscribe("test_sub".to_string(), variables).await;
        assert!(stream.is_ok());
    }

    #[tokio::test]
    async fn test_list_active_subscriptions() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test_sub".to_string(),
            targets: vec!["service1".to_string()],
            ..Default::default()
        };
        manager.register_subscription(config).await.unwrap();

        let variables = HashMap::new();
        let _stream = manager
            .subscribe("test_sub".to_string(), variables)
            .await
            .unwrap();

        let active = manager.list_active_subscriptions().await;
        assert_eq!(active.len(), 1);
    }

    #[tokio::test]
    async fn test_unsubscribe() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test_sub".to_string(),
            targets: vec!["service1".to_string()],
            ..Default::default()
        };
        manager.register_subscription(config).await.unwrap();

        let variables = HashMap::new();
        let _stream = manager
            .subscribe("test_sub".to_string(), variables)
            .await
            .unwrap();

        let active = manager.list_active_subscriptions().await;
        let sub_id = active.first().unwrap().clone();

        manager.unsubscribe(&sub_id).await.unwrap();
        let active_after = manager.list_active_subscriptions().await;
        assert_eq!(active_after.len(), 0);
    }

    #[tokio::test]
    async fn test_get_subscription_info() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test_sub".to_string(),
            targets: vec!["service1".to_string(), "service2".to_string()],
            ..Default::default()
        };
        manager.register_subscription(config).await.unwrap();

        let variables = HashMap::new();
        let _stream = manager
            .subscribe("test_sub".to_string(), variables)
            .await
            .unwrap();

        let active = manager.list_active_subscriptions().await;
        let sub_id = active.first().unwrap().clone();

        let info = manager.get_subscription_info(&sub_id).await;
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.operation, "test_sub");
        assert_eq!(info.targets.len(), 2);
    }

    #[tokio::test]
    async fn test_merge_events() {
        let manager = FederatedSubscriptionManager::new();
        let events = vec![
            SubscriptionEvent::new(
                "service1".to_string(),
                serde_json::json!({"field1": "value1"}),
            ),
            SubscriptionEvent::new(
                "service2".to_string(),
                serde_json::json!({"field2": "value2"}),
            ),
        ];

        let merged = manager
            .aggregate_events(events, AggregationStrategy::Merge)
            .await
            .unwrap();

        assert_eq!(merged.source, "merged");
        if let serde_json::Value::Object(obj) = merged.payload {
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("field1"));
            assert!(obj.contains_key("field2"));
        } else {
            panic!("Expected object payload");
        }
    }

    #[tokio::test]
    async fn test_aggregate_first_strategy() {
        let manager = FederatedSubscriptionManager::new();
        let events = vec![
            SubscriptionEvent::new("service1".to_string(), serde_json::json!({"first": true})),
            SubscriptionEvent::new("service2".to_string(), serde_json::json!({"second": true})),
        ];

        let result = manager
            .aggregate_events(events, AggregationStrategy::First)
            .await
            .unwrap();

        assert_eq!(result.source, "service1");
    }

    #[tokio::test]
    async fn test_aggregate_latest_strategy() {
        let manager = FederatedSubscriptionManager::new();
        let mut event1 = SubscriptionEvent::new("service1".to_string(), serde_json::json!({}));
        event1.timestamp = chrono::Utc::now() - chrono::Duration::seconds(10);

        let event2 = SubscriptionEvent::new("service2".to_string(), serde_json::json!({}));

        let events = vec![event1, event2.clone()];

        let result = manager
            .aggregate_events(events, AggregationStrategy::Latest)
            .await
            .unwrap();

        assert_eq!(result.timestamp, event2.timestamp);
    }

    #[test]
    fn test_routing_broadcast() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test".to_string(),
            routing: SubscriptionRoutingStrategy::Broadcast,
            targets: vec!["s1".to_string(), "s2".to_string(), "s3".to_string()],
            ..Default::default()
        };
        let context = RoutingContext::default();

        let targets = manager.route_subscription(&config, &context);
        assert_eq!(targets.len(), 3);
    }

    #[test]
    fn test_routing_single() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test".to_string(),
            routing: SubscriptionRoutingStrategy::Single,
            targets: vec!["s1".to_string(), "s2".to_string()],
            ..Default::default()
        };
        let context = RoutingContext::default();

        let targets = manager.route_subscription(&config, &context);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0], "s1");
    }

    #[tokio::test]
    async fn test_logging_event_handler() {
        let handler = LoggingEventHandler;
        let event = SubscriptionEvent::new("test".to_string(), serde_json::json!({}));
        let result = handler.handle(&event);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_register_event_handler() {
        let manager = FederatedSubscriptionManager::new();
        let handler = Box::new(LoggingEventHandler);
        manager.register_event_handler(handler).await.unwrap();
    }

    #[tokio::test]
    async fn test_publish_event() {
        let manager = FederatedSubscriptionManager::new();
        let config = FederatedSubscriptionConfig {
            operation: "test_sub".to_string(),
            targets: vec!["service1".to_string()],
            ..Default::default()
        };
        manager.register_subscription(config).await.unwrap();

        let variables = HashMap::new();
        let mut stream = manager
            .subscribe("test_sub".to_string(), variables)
            .await
            .unwrap();

        let active = manager.list_active_subscriptions().await;
        let sub_id = active.first().unwrap().clone();

        let event = SubscriptionEvent::new("test".to_string(), serde_json::json!({"test": true}));
        manager.publish_event(&sub_id, event.clone()).await.unwrap();

        // Try to receive the event (with timeout)
        let received =
            tokio::time::timeout(std::time::Duration::from_millis(100), stream.next()).await;

        assert!(received.is_ok());
        assert!(received.unwrap().is_some());
    }
}
