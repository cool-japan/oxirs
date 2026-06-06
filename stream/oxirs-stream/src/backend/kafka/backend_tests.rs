//! Unit tests for the Kafka backend.

use super::backend::*;
use super::KafkaEvent;
use crate::{StreamConfig, StreamEvent};
use std::collections::HashMap;

#[tokio::test]
async fn test_kafka_backend_creation() {
    let config = StreamConfig {
        backend: crate::StreamBackendType::Kafka {
            brokers: vec!["localhost:9092".to_string()],
            security_protocol: None,
            sasl_config: None,
        },
        topic: "test_topic".to_string(),
        batch_size: 100,
        ..Default::default()
    };

    let backend = KafkaBackend::new(config);
    assert!(backend.is_ok());
}

#[test]
fn test_kafka_event_conversion() {
    use crate::EventMetadata;
    use chrono::Utc;

    let metadata = EventMetadata {
        event_id: "test".to_string(),
        timestamp: Utc::now(),
        source: "test".to_string(),
        user: None,
        context: None,
        caused_by: None,
        version: "1.0".to_string(),
        properties: HashMap::new(),
        checksum: None,
    };

    let stream_event = StreamEvent::TripleAdded {
        subject: "test:subject".to_string(),
        predicate: "test:predicate".to_string(),
        object: "test:object".to_string(),
        graph: Some("test:graph".to_string()),
        metadata,
    };

    let kafka_event = KafkaEvent::from(stream_event);
    assert_eq!(kafka_event.event_type, "triple_added");
    assert!(kafka_event.partition_key.is_some());
}

#[tokio::test]
async fn test_stats_update() {
    let config = StreamConfig {
        backend: crate::StreamBackendType::Kafka {
            brokers: vec!["localhost:9092".to_string()],
            security_protocol: None,
            sasl_config: None,
        },
        topic: "test_topic".to_string(),
        batch_size: 100,
        ..Default::default()
    };

    let backend = KafkaBackend::new(config).unwrap();

    // Test stats update
    backend.update_stats(1024, 50, false).await;
    let stats = backend.get_stats().await;

    assert_eq!(stats.events_published, 1);
    assert_eq!(stats.bytes_sent, 1024);
    assert_eq!(stats.max_latency_ms, 50);
    assert_eq!(stats.avg_latency_ms, 50.0);
}
