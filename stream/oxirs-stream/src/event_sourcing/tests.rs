//! Tests for the async EventStore, PersistenceManager, and VectorClock.

use super::*;
use crate::{EventMetadata, StreamEvent, VectorClock};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Build a config pointed at a fresh, uniquely named temp directory so tests
/// running concurrently never share (or race on) the same persistence files.
fn test_config() -> EventStoreConfig {
    let mut config = EventStoreConfig::default();
    let unique_dir =
        std::env::temp_dir().join(format!("oxirs-event-store-test-{}", uuid::Uuid::new_v4()));
    config.persistence_backend = PersistenceBackend::FileSystem {
        base_path: unique_dir.to_string_lossy().into_owned(),
    };
    config
}

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
    let config = test_config();
    let store = EventStore::new(config).await.unwrap();

    let stats = store.get_stats();
    assert_eq!(stats.total_events_stored.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn test_store_and_retrieve_event() {
    let config = test_config();
    let store = EventStore::new(config).await.unwrap();

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
    let config = test_config();
    let store = EventStore::new(config).await.unwrap();

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
    let mut config = test_config();
    config.snapshot_config.snapshot_interval = 3; // Snapshot every 3 events

    let store = EventStore::new(config).await.unwrap();

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
    let config = test_config();
    let store = EventStore::new(config).await.unwrap();

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
    let manager = store::PersistenceManager::new(backend).unwrap();

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

/// Regression test for stream/oxirs-stream/src/event_sourcing/store.rs:703 —
/// the FileSystem backend used to be a pure no-op (a `sleep()` + a fake byte
/// counter bump) so nothing was ever actually written to disk. This verifies
/// events genuinely land on disk as JSON-lines and fsync-flush before
/// `store_event` returns.
#[tokio::test]
async fn test_filesystem_backend_actually_writes_to_disk() {
    let config = test_config();
    let base_path = match &config.persistence_backend {
        PersistenceBackend::FileSystem { base_path } => base_path.clone(),
        _ => unreachable!(),
    };
    let store = EventStore::new(config).await.unwrap();

    store
        .store_event("durable_stream".to_string(), create_test_event())
        .await
        .unwrap();

    let events_file = std::path::Path::new(&base_path).join("events.jsonl");
    let contents = tokio::fs::read_to_string(&events_file).await.unwrap();
    assert!(
        !contents.trim().is_empty(),
        "expected at least one persisted event line on disk"
    );
    assert!(contents.contains("durable_stream"));

    let _ = tokio::fs::remove_dir_all(&base_path).await;
}

/// Regression test: load-on-open must recover previously persisted events
/// into memory (and mark them `Persisted`) when a new `EventStore` opens an
/// existing directory, rather than starting from empty every time.
#[tokio::test]
async fn test_load_on_open_recovers_persisted_events() {
    let config = test_config();
    let base_path = match &config.persistence_backend {
        PersistenceBackend::FileSystem { base_path } => base_path.clone(),
        _ => unreachable!(),
    };

    {
        let store = EventStore::new(config.clone()).await.unwrap();
        for _ in 0..3 {
            store
                .store_event("recovered_stream".to_string(), create_test_event())
                .await
                .unwrap();
        }
    }

    // Re-open a fresh EventStore against the same directory.
    let reopened = EventStore::new(config).await.unwrap();
    let events = reopened
        .get_stream_events("recovered_stream", None)
        .await
        .unwrap();
    assert_eq!(events.len(), 3, "load-on-open should recover all 3 events");
    for event in &events {
        assert!(matches!(
            event.storage_metadata.persistence_status,
            PersistenceStatus::Persisted
        ));
    }

    let _ = tokio::fs::remove_dir_all(&base_path).await;
}

/// Regression test: eviction must never drop an event before it has actually
/// been persisted. With a durable FileSystem backend, forcing memory over
/// `max_memory_events` should only remove events whose persistence
/// succeeded, and the store must never end up under-populated by dropping
/// something that was never written to disk.
#[tokio::test]
async fn test_eviction_only_drops_persisted_events() {
    let mut config = test_config();
    config.max_memory_events = 2;
    let store = EventStore::new(config).await.unwrap();

    for _ in 0..5 {
        store
            .store_event("evictable_stream".to_string(), create_test_event())
            .await
            .unwrap();
    }

    let stats = store.get_stats();
    // All 5 events should have been durably persisted even though only 2 are
    // allowed to stay resident in memory.
    assert_eq!(
        stats.failed_operations.load(Ordering::Relaxed),
        0,
        "no persistence attempt should have failed"
    );
    assert_eq!(
        stats.persistence_operations.load(Ordering::Relaxed),
        5,
        "every stored event should have been persisted before it was eligible for eviction"
    );
}

/// Regression test: construction must fail fast (not silently degrade to an
/// in-memory-only store) when the configured FileSystem persistence
/// directory cannot be created.
#[tokio::test]
async fn test_construction_fails_fast_on_unusable_directory() {
    // Create a plain file, then ask the store to use a path *through* that
    // file as its base directory — `create_dir_all` cannot succeed there.
    let blocking_file =
        std::env::temp_dir().join(format!("oxirs-blocker-{}", uuid::Uuid::new_v4()));
    tokio::fs::write(&blocking_file, b"not a directory")
        .await
        .unwrap();
    let unusable_path = blocking_file.join("nested").join("events");

    let mut config = test_config();
    config.persistence_backend = PersistenceBackend::FileSystem {
        base_path: unusable_path.to_string_lossy().into_owned(),
    };

    let result = EventStore::new(config).await;
    assert!(
        result.is_err(),
        "construction should fail fast when the persistence directory can't be created"
    );

    let _ = tokio::fs::remove_file(&blocking_file).await;
}
