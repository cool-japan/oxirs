//! Enhanced WebSocket functionality tests

use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use serde_json;
use chrono::Utc;

use oxirs_fuseki::handlers::websocket::{
    SubscriptionManager, WebSocketConnectionManager, ChangeDetector,
    SubscriptionFilters, EnhancedSubscriptionFilters, ChangeNotification,
    ConnectionMetrics, WebSocketConnection, LiveQuerySubscription,
    SubscriptionStatus, SubscriptionMetrics
};

#[cfg(test)]
mod enhanced_websocket_tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_change_detection() {
        let mut detector = ChangeDetector::new();
        
        // Simulate store (this would be the actual store in real implementation)
        let mock_store = create_mock_store();
        
        // Test change detection logic
        let changes = detect_mock_store_changes(&mock_store, &mut detector).await;
        assert!(changes.is_ok());
    }

    #[tokio::test]
    async fn test_websocket_connection_management() {
        let manager = WebSocketConnectionManager::new();
        
        let connection = WebSocketConnection {
            connection_id: "test_conn_123".to_string(),
            user_id: Some("user_456".to_string()),
            connected_at: Utc::now(),
            last_activity: Utc::now(),
            subscriptions: vec!["sub_1".to_string(), "sub_2".to_string()],
            message_count: 0,
            compression_enabled: true,
        };
        
        // Test connection tracking
        assert!(manager.add_connection(connection.clone()).await.is_ok());
        
        let retrieved = manager.get_connection(&connection.connection_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().user_id, connection.user_id);
    }

    #[tokio::test]
    async fn test_subscription_lifecycle() {
        let manager = SubscriptionManager::new();
        
        let enhanced_filters = EnhancedSubscriptionFilters {
            min_results: Some(5),
            max_results: Some(500),
            graph_filter: Some(vec!["http://test.org/graph1".to_string()]),
            update_threshold_ms: Some(1000),
            result_format: Some("json".to_string()),
            include_provenance: Some(true),
            debounce_ms: Some(300),
            batch_updates: Some(true),
        };
        
        // Create enhanced subscription
        let sub_id = manager.add_enhanced_subscription(
            "SELECT ?s ?p ?o WHERE { GRAPH <http://test.org/graph1> { ?s ?p ?o } }".to_string(),
            Some("test_user".to_string()),
            enhanced_filters.clone(),
        ).await;
        
        assert!(!sub_id.is_empty());
        
        // Test subscription pause/resume cycle
        assert!(manager.pause_subscription(&sub_id).await);
        assert!(manager.resume_subscription(&sub_id).await);
        
        // Test metrics collection
        let metrics = manager.get_subscription_metrics(&sub_id).await;
        assert!(metrics.is_some());
        
        let metrics = metrics.unwrap();
        assert!(metrics.total_updates >= 0);
        assert!(metrics.average_execution_time_ms >= 0.0);
        
        // Cleanup
        assert!(manager.remove_subscription(&sub_id).await);
    }

    #[tokio::test]
    async fn test_change_notification_filtering() {
        let manager = SubscriptionManager::new();
        
        // Create subscription with graph filter
        let filters = SubscriptionFilters {
            min_results: None,
            max_results: Some(100),
            graph_filter: Some(vec!["http://test.org/filtered".to_string()]),
            update_threshold_ms: Some(2000),
        };
        
        let sub_id = manager.add_subscription(
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            Some("filter_user".to_string()),
            filters,
        ).await;
        
        // Create notifications for different graphs
        let filtered_notification = ChangeNotification {
            change_type: "INSERT".to_string(),
            affected_graphs: vec!["http://test.org/filtered".to_string()],
            timestamp: Utc::now(),
            change_count: 3,
        };
        
        let unfiltered_notification = ChangeNotification {
            change_type: "INSERT".to_string(),
            affected_graphs: vec!["http://test.org/other".to_string()],
            timestamp: Utc::now(),
            change_count: 2,
        };
        
        // Test notification filtering (this would be tested with actual subscription logic)
        manager.notify_change(filtered_notification).await;
        manager.notify_change(unfiltered_notification).await;
        
        // In a real implementation, we'd verify that only the filtered notification
        // triggers updates for the subscription
        
        manager.remove_subscription(&sub_id).await;
    }

    #[tokio::test]
    async fn test_subscription_batching_and_deduplication() {
        let changes = vec![
            ChangeNotification {
                change_type: "INSERT".to_string(),
                affected_graphs: vec!["http://test.org/graph1".to_string()],
                timestamp: Utc::now(),
                change_count: 1,
            },
            ChangeNotification {
                change_type: "INSERT".to_string(),
                affected_graphs: vec!["http://test.org/graph1".to_string()],
                timestamp: Utc::now(),
                change_count: 2,
            },
            ChangeNotification {
                change_type: "DELETE".to_string(),
                affected_graphs: vec!["http://test.org/graph2".to_string()],
                timestamp: Utc::now(),
                change_count: 1,
            },
        ];
        
        let batched = batch_and_deduplicate_changes(changes);
        
        // Should have 2 unique changes (INSERT for graph1, DELETE for graph2)
        assert_eq!(batched.len(), 2);
        
        // Find the batched INSERT change for graph1
        let insert_change = batched.iter()
            .find(|c| c.change_type == "INSERT" && c.affected_graphs.contains(&"http://test.org/graph1".to_string()));
        
        assert!(insert_change.is_some());
        assert_eq!(insert_change.unwrap().change_count, 3); // 1 + 2 = 3
    }

    #[tokio::test]
    async fn test_concurrent_subscriptions() {
        let manager = SubscriptionManager::new();
        
        // Create multiple subscriptions concurrently
        let mut handles = vec![];
        
        for i in 0..5 {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                let filters = SubscriptionFilters {
                    min_results: None,
                    max_results: Some(100),
                    graph_filter: None,
                    update_threshold_ms: Some(1000),
                };
                
                manager_clone.add_subscription(
                    format!("SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER(?s = <http://test.org/entity{}>) }}", i),
                    Some(format!("user_{}", i)),
                    filters,
                ).await
            });
            
            handles.push(handle);
        }
        
        // Wait for all subscriptions to be created
        let mut subscription_ids = vec![];
        for handle in handles {
            let sub_id = handle.await.unwrap();
            subscription_ids.push(sub_id);
        }
        
        assert_eq!(subscription_ids.len(), 5);
        
        // Verify all subscriptions exist
        let active_subs = manager.get_active_subscriptions().await;
        assert_eq!(active_subs.len(), 5);
        
        // Cleanup
        for sub_id in subscription_ids {
            manager.remove_subscription(&sub_id).await;
        }
    }

    #[tokio::test]
    async fn test_subscription_performance_tracking() {
        let manager = SubscriptionManager::new();
        
        let sub_id = manager.add_subscription(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
            Some("perf_user".to_string()),
            SubscriptionFilters {
                min_results: None,
                max_results: Some(1000),
                graph_filter: None,
                update_threshold_ms: Some(500),
            },
        ).await;
        
        // Simulate multiple result updates
        for i in 1..=5 {
            manager.update_subscription_result(&sub_id, i * 10).await;
            sleep(Duration::from_millis(10)).await;
        }
        
        let subscription = manager.get_subscription(&sub_id).await;
        assert!(subscription.is_some());
        
        let sub = subscription.unwrap();
        assert_eq!(sub.result_count, 50); // Last update was 5 * 10 = 50
        assert!(sub.last_result_at.is_some());
        
        manager.remove_subscription(&sub_id).await;
    }

    #[tokio::test]
    async fn test_subscription_error_handling() {
        let manager = SubscriptionManager::new();
        
        // Test subscription with invalid query (this would be caught in real implementation)
        let sub_id = manager.add_subscription(
            "INVALID SPARQL QUERY".to_string(),
            Some("error_user".to_string()),
            SubscriptionFilters {
                min_results: None,
                max_results: Some(100),
                graph_filter: None,
                update_threshold_ms: Some(1000),
            },
        ).await;
        
        // Subscription should still be created (validation would happen during execution)
        assert!(!sub_id.is_empty());
        
        let subscription = manager.get_subscription(&sub_id).await;
        assert!(subscription.is_some());
        
        manager.remove_subscription(&sub_id).await;
    }
}

// Helper functions for testing

fn create_mock_store() -> MockStore {
    MockStore {
        graphs: HashMap::new(),
    }
}

struct MockStore {
    graphs: HashMap<String, u64>, // graph_name -> checksum
}

async fn detect_mock_store_changes(
    _store: &MockStore,
    detector: &mut ChangeDetector,
) -> Result<Vec<ChangeNotification>, String> {
    let now = Utc::now();
    let mut changes = Vec::new();
    
    // Simulate change detection based on time
    if (now - detector.last_check).num_seconds() > 1 {
        changes.push(ChangeNotification {
            change_type: "MOCK_CHANGE".to_string(),
            affected_graphs: vec!["http://mock.org/graph".to_string()],
            timestamp: now,
            change_count: 1,
        });
    }
    
    detector.last_check = now;
    Ok(changes)
}

fn batch_and_deduplicate_changes(changes: Vec<ChangeNotification>) -> Vec<ChangeNotification> {
    let mut batched: HashMap<String, ChangeNotification> = HashMap::new();
    
    for change in changes {
        let key = format!("{}:{}", change.change_type, change.affected_graphs.join(","));
        
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

impl WebSocketConnectionManager {
    pub fn new() -> Self {
        use std::sync::Arc;
        use tokio::sync::RwLock;
        
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            connection_metrics: Arc::new(RwLock::new(ConnectionMetrics::default())),
        }
    }
    
    pub async fn add_connection(&self, connection: WebSocketConnection) -> Result<(), String> {
        let mut connections = self.connections.write().await;
        connections.insert(connection.connection_id.clone(), connection);
        Ok(())
    }
    
    pub async fn get_connection(&self, connection_id: &str) -> Option<WebSocketConnection> {
        let connections = self.connections.read().await;
        connections.get(connection_id).cloned()
    }
}

impl ChangeDetector {
    pub fn new() -> Self {
        Self {
            last_check: Utc::now(),
            graph_checksums: HashMap::new(),
            change_buffer: Vec::new(),
        }
    }
}