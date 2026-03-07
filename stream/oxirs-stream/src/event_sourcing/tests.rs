//! Tests for the async EventStore, PersistenceManager, and VectorClock.

use super::*;
use crate::{EventMetadata, StreamEvent, VectorClock};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

fn create_test_event() -> StreamEvent {
    StreamEvent::TripleAdded {
        subject: "http://test.org/subject".to_string(),
        predicate: "http://test.org/predicate".to_string(),
        object: "\"test_value\"".to_string(),
        graph: None,
        metadata: EventMetadata {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        },
    }
}

#[tokio::test]
async fn test_event_store_creation() {
    let config = EventStoreConfig::default();
    let store = EventStore::new(config);

    let stats = store.get_stats();
    assert_eq!(stats.total_events_stored.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn test_store_and_retrieve_event() {
    let config = EventStoreConfig::default();
    let store = EventStore::new(config);

    let event = create_test_event();
    let stored_event = store
        .store_event("test_stream".to_string(), event)
        .await
        .unwrap();

    assert_eq!(stored_event.stream_id, "test_stream");
    assert_eq!(stored_event.stream_version, 1);
    assert_eq!(stored_event.sequence_number, 1);

    let stream_events = store.get_stream_events("test_stream", None).await.unwrap();
    assert_eq!(stream_events.len(), 1);
    assert_eq!(stream_events[0].event_id, stored_event.event_id);
}

#[tokio::test]
async fn test_event_query() {
    let config = EventStoreConfig::default();
    let store = EventStore::new(config);

    // Store multiple events
    for i in 0..5 {
        let event = create_test_event();
        store
            .store_event(format!("stream_{}", i % 2), event)
            .await
            .unwrap();
    }

    // Query specific stream
    let query = EventQuery {
        stream_id: Some("stream_0".to_string()),
        event_types: None,
        time_range: None,
        sequence_range: None,
        source: None,
        custom_filters: HashMap::new(),
        limit: None,
        order: QueryOrder::SequenceAsc,
    };

    let results = store.query_events(query).await.unwrap();
    assert_eq!(results.len(), 3); // Events 0, 2, 4

    // Verify sequence order
    for i in 1..results.len() {
        assert!(results[i].sequence_number > results[i - 1].sequence_number);
    }
}

#[tokio::test]
async fn test_snapshot_creation() {
    let mut config = EventStoreConfig::default();
    config.snapshot_config.snapshot_interval = 3; // Snapshot every 3 events

    let store = EventStore::new(config);

    // Store events to trigger snapshot
    for _ in 0..3 {
        let event = create_test_event();
        store
            .store_event("test_stream".to_string(), event)
            .await
            .unwrap();
    }

    let snapshot = store.get_latest_snapshot("test_stream").await.unwrap();
    assert!(snapshot.is_some());

    let snapshot = snapshot.unwrap();
    assert_eq!(snapshot.stream_id, "test_stream");
    assert_eq!(snapshot.stream_version, 3);
}

#[tokio::test]
async fn test_replay_from_timestamp() {
    let config = EventStoreConfig::default();
    let store = EventStore::new(config);

    let start_time = chrono::Utc::now();

    // Store some events
    for i in 0..3 {
        let event = create_test_event();
        store
            .store_event(format!("stream_{i}"), event)
            .await
            .unwrap();
    }

    // Replay from start time
    let replayed_events = store.replay_from_timestamp(start_time).await.unwrap();
    assert!(replayed_events.len() >= 3);

    // Verify chronological order
    for i in 1..replayed_events.len() {
        assert!(replayed_events[i].stored_at >= replayed_events[i - 1].stored_at);
    }
}

#[tokio::test]
async fn test_persistence_manager() {
    let backend = PersistenceBackend::Memory;
    let manager = store::PersistenceManager::new(backend);

    let event = create_test_event();
    let stored_event = StoredEvent {
        event_id: uuid::Uuid::new_v4(),
        sequence_number: 1,
        stream_id: "test".to_string(),
        stream_version: 1,
        event_data: event,
        stored_at: chrono::Utc::now(),
        storage_metadata: StorageMetadata {
            checksum: "test".to_string(),
            compressed_size: None,
            original_size: 100,
            storage_location: "memory".to_string(),
            persistence_status: PersistenceStatus::InMemory,
        },
    };

    manager
        .queue_operation(PersistenceOperation::StoreEvent(Box::new(stored_event)))
        .await
        .unwrap();
    manager.process_pending_operations().await.unwrap();

    assert_eq!(manager.stats.operations_queued.load(Ordering::Relaxed), 1);
    assert_eq!(
        manager.stats.operations_completed.load(Ordering::Relaxed),
        1
    );
}

#[test]
fn test_vector_clock_operations() {
    let mut clock1 = VectorClock::new();
    let mut clock2 = VectorClock::new();

    // Test concurrent clocks
    clock1.increment("region1");
    clock2.increment("region2");
    assert!(clock1.is_concurrent(&clock2));

    // Test happens-before
    clock1.update(&clock2);
    clock1.increment("region1");
    assert!(clock2.happens_before(&clock1));
    assert!(!clock1.happens_before(&clock2));
}
