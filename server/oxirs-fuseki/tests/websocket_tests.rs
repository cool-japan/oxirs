//! Tests for WebSocket Live Query Subscriptions

use axum::{
    extract::ws::{Message, WebSocket},
    Router,
};
use std::time::SystemTime;
use futures::{SinkExt, StreamExt};
use oxirs_fuseki::{
    config::ServerConfig,
    metrics::MetricsService,
    server::Runtime,
    store::Store,
    websocket::{
        NotificationFilter, QueryParameters, SubscriptionManager, WebSocketConfig, WsMessage,
    },
};
use serde_json::json;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tower::ServiceExt;

#[tokio::test]
async fn test_websocket_connection() {
    // Create test store and config
    let store = Store::new().unwrap();
    let config = ServerConfig::default();
    let metrics = Arc::new(MetricsService::new(config.monitoring.clone()).unwrap());

    // Create subscription manager
    let ws_config = WebSocketConfig::default();
    let subscription_manager = SubscriptionManager::new(Arc::new(store), metrics, ws_config);

    // Start the manager
    let manager = subscription_manager.clone();
    tokio::spawn(async move {
        manager.start().await;
    });

    // Give it time to start
    sleep(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn test_subscription_lifecycle() {
    // Test message serialization/deserialization
    let subscribe_msg = WsMessage::Subscribe {
        query: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10".to_string(),
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

    // Serialize to JSON
    let json = serde_json::to_string(&subscribe_msg).unwrap();
    assert!(json.contains("subscribe"));
    assert!(json.contains("SELECT"));

    // Deserialize back
    let deserialized: WsMessage = serde_json::from_str(&json).unwrap();
    match deserialized {
        WsMessage::Subscribe { query, .. } => {
            assert!(query.contains("SELECT"));
        }
        _ => panic!("Wrong message type"),
    }
}

#[tokio::test]
async fn test_ping_pong() {
    let ping_msg = WsMessage::Ping {
        timestamp: 12345678,
    };

    let json = serde_json::to_string(&ping_msg).unwrap();
    let deserialized: WsMessage = serde_json::from_str(&json).unwrap();

    match deserialized {
        WsMessage::Ping { timestamp } => {
            assert_eq!(timestamp, 12345678);
        }
        _ => panic!("Wrong message type"),
    }
}

#[tokio::test]
async fn test_query_validation() {
    use oxirs_fuseki::websocket::SubscriptionManager;

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

    // No LIMIT clause
    assert!(
        SubscriptionManager::validate_subscription_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
            .is_err()
    );

    // ASK queries not supported
    assert!(SubscriptionManager::validate_subscription_query("ASK { ?s ?p ?o }").is_err());
}

#[tokio::test]
async fn test_notification_filter() {
    let filter = NotificationFilter {
        min_change_threshold: Some(10.0),
        monitored_variables: Some(vec!["person".to_string(), "name".to_string()]),
        debounce_ms: Some(500),
        rate_limit: Some(120),
    };

    let json = serde_json::to_string(&filter).unwrap();
    assert!(json.contains("min_change_threshold"));
    assert!(json.contains("monitored_variables"));
    assert!(json.contains("debounce_ms"));
    assert!(json.contains("rate_limit"));
}

#[tokio::test]
async fn test_query_result_message() {
    use oxirs_fuseki::websocket::{QueryResult, ResultMetadata};
    use std::collections::HashMap;

    let result = QueryResult {
        bindings: vec![
            HashMap::from([
                ("s".to_string(), json!("http://example.org/subject1")),
                ("p".to_string(), json!("http://example.org/predicate")),
                ("o".to_string(), json!("Object 1")),
            ]),
            HashMap::from([
                ("s".to_string(), json!("http://example.org/subject2")),
                ("p".to_string(), json!("http://example.org/predicate")),
                ("o".to_string(), json!("Object 2")),
            ]),
        ],
        metadata: ResultMetadata {
            execution_time_ms: 42,
            result_count: 2,
            result_hash: 12345,
        },
    };

    let update_msg = WsMessage::QueryUpdate {
        subscription_id: "sub-123".to_string(),
        result: result.clone(),
        changes: None,
    };

    let json = serde_json::to_string(&update_msg).unwrap();
    assert!(json.contains("query_update"));
    assert!(json.contains("subscription_id"));
    assert!(json.contains("result"));
    assert!(json.contains("metadata"));
}

#[tokio::test]
async fn test_error_message() {
    let error_msg = WsMessage::Error {
        code: "subscription_limit_exceeded".to_string(),
        message: "Maximum subscriptions per connection exceeded".to_string(),
        details: Some(json!({
            "current": 100,
            "maximum": 100
        })),
    };

    let json = serde_json::to_string(&error_msg).unwrap();
    assert!(json.contains("error"));
    assert!(json.contains("subscription_limit_exceeded"));
    assert!(json.contains("details"));
}

#[tokio::test]
async fn test_authentication_flow() {
    let auth_msg = WsMessage::Auth {
        token: "bearer-token-12345".to_string(),
    };

    let json = serde_json::to_string(&auth_msg).unwrap();
    assert!(json.contains("auth"));
    assert!(json.contains("bearer-token-12345"));

    // Test ACK message
    let ack_msg = WsMessage::Ack {
        message_id: "msg-123".to_string(),
        success: true,
        error: None,
    };

    let ack_json = serde_json::to_string(&ack_msg).unwrap();
    assert!(ack_json.contains("ack"));
    assert!(ack_json.contains("success"));
}

#[tokio::test]
async fn test_subscription_confirmation() {
    let subscribed_msg = WsMessage::Subscribed {
        subscription_id: "sub-456".to_string(),
        query: "SELECT ?s WHERE { ?s a ?o } LIMIT 10".to_string(),
    };

    let json = serde_json::to_string(&subscribed_msg).unwrap();
    assert!(json.contains("subscribed"));
    assert!(json.contains("sub-456"));

    let unsubscribed_msg = WsMessage::Unsubscribed {
        subscription_id: "sub-456".to_string(),
    };

    let unsub_json = serde_json::to_string(&unsubscribed_msg).unwrap();
    assert!(unsub_json.contains("unsubscribed"));
}

#[tokio::test]
async fn test_change_detection() {
    use oxirs_fuseki::websocket::{ChangeNotification, ChangeType};
    use std::time::Instant;

    let notification = ChangeNotification {
        graphs: vec!["http://example.org/graph1".to_string()],
        change_type: ChangeType::Insert,
        timestamp: SystemTime::now(),
        details: Some(json!({
            "triples_added": 42,
            "subjects": ["http://example.org/s1", "http://example.org/s2"]
        })),
    };

    // Test all change types
    let change_types = vec![
        ChangeType::Insert,
        ChangeType::Delete,
        ChangeType::Update,
        ChangeType::Clear,
        ChangeType::Load,
        ChangeType::Transaction,
    ];

    for change_type in change_types {
        let json = serde_json::to_string(&change_type).unwrap();
        assert!(!json.is_empty());
    }
}

#[tokio::test]
async fn test_result_changes() {
    use oxirs_fuseki::websocket::ResultChanges;
    use std::collections::HashMap;

    let changes = ResultChanges {
        added: vec![HashMap::from([("x".to_string(), json!("new_value1"))])],
        removed: vec![HashMap::from([("x".to_string(), json!("old_value1"))])],
        modified: vec![(
            HashMap::from([("x".to_string(), json!("old_value2"))]),
            HashMap::from([("x".to_string(), json!("new_value2"))]),
        )],
    };

    let json = serde_json::to_string(&changes).unwrap();
    assert!(json.contains("added"));
    assert!(json.contains("removed"));
    assert!(json.contains("modified"));
}

#[tokio::test]
async fn test_subscription_limits() {
    let ws_config = WebSocketConfig {
        max_subscriptions_per_connection: 5,
        max_total_subscriptions: 100,
        evaluation_interval: Duration::from_secs(1),
        connection_timeout: Duration::from_secs(300),
        max_message_size: 1024 * 1024,
        enable_compression: false,
        heartbeat_interval: Duration::from_secs(30),
    };

    assert_eq!(ws_config.max_subscriptions_per_connection, 5);
    assert_eq!(ws_config.max_total_subscriptions, 100);
}

#[tokio::test]
async fn test_query_parameters() {
    let params = QueryParameters {
        default_graph_uri: vec!["http://example.org/default".to_string()],
        named_graph_uri: vec![
            "http://example.org/graph1".to_string(),
            "http://example.org/graph2".to_string(),
        ],
        timeout_ms: Some(10000),
        format: "application/sparql-results+json".to_string(),
    };

    let json = serde_json::to_string(&params).unwrap();
    assert!(json.contains("default_graph_uri"));
    assert!(json.contains("named_graph_uri"));
    assert!(json.contains("timeout_ms"));
    assert!(json.contains("format"));
}

// Integration test with mock WebSocket
#[tokio::test]
async fn test_websocket_integration() {
    // This would require a full server setup
    // For now, just test the message flow

    let messages = vec![
        WsMessage::Ping { timestamp: 1234 },
        WsMessage::Subscribe {
            query: "SELECT * WHERE { ?s ?p ?o } LIMIT 5".to_string(),
            parameters: QueryParameters {
                default_graph_uri: vec![],
                named_graph_uri: vec![],
                timeout_ms: None,
                format: "json".to_string(),
            },
            filter: None,
        },
        WsMessage::Unsubscribe {
            subscription_id: "test-sub".to_string(),
        },
    ];

    for msg in messages {
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WsMessage = serde_json::from_str(&json).unwrap();

        // Verify round-trip serialization
        let json2 = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(json, json2);
    }
}
