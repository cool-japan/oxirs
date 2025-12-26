//! OPC UA Subscription Management

use super::types::NodeSubscription;
use std::collections::HashMap;

/// Subscription Manager for OPC UA monitored items
pub struct SubscriptionManager {
    subscriptions: HashMap<String, NodeSubscription>,
    monitored_items: HashMap<u32, String>, // monitored_item_id -> node_id
}

impl SubscriptionManager {
    /// Create a new subscription manager
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            monitored_items: HashMap::new(),
        }
    }

    /// Add a node subscription
    pub fn add_subscription(&mut self, node_id: String, subscription: NodeSubscription) {
        self.subscriptions.insert(node_id, subscription);
    }

    /// Get subscription by node ID
    pub fn get_subscription(&self, node_id: &str) -> Option<&NodeSubscription> {
        self.subscriptions.get(node_id)
    }

    /// Remove subscription
    pub fn remove_subscription(&mut self, node_id: &str) -> Option<NodeSubscription> {
        self.subscriptions.remove(node_id)
    }

    /// Get all subscriptions
    pub fn all_subscriptions(&self) -> Vec<&NodeSubscription> {
        self.subscriptions.values().collect()
    }

    /// Register monitored item
    pub fn register_monitored_item(&mut self, item_id: u32, node_id: String) {
        self.monitored_items.insert(item_id, node_id);
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}
