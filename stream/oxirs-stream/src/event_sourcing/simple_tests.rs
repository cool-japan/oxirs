//! Tests for the simplified in-memory event sourcing utilities (v1.1.0).

use super::simple::*;
use std::sync::atomic::{AtomicU64, Ordering as AOrdering};

// ── SimpleEventStore ──────────────────────────────────────────────────────

#[test]
fn test_simple_store_empty_on_creation() {
    let store = SimpleEventStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

#[test]
fn test_simple_store_append() {
    let mut store = SimpleEventStore::new();
    let ev = store.append("agg-1", "Created", r#"{"name":"Alice"}"#);
    assert_eq!(ev.id, 1);
    assert_eq!(ev.aggregate_id, "agg-1");
    assert_eq!(ev.event_type, "Created");
    assert_eq!(ev.version, 1);
    assert_eq!(store.len(), 1);
}

#[test]
fn test_simple_store_version_increments_per_aggregate() {
    let mut store = SimpleEventStore::new();
    let e1 = store.append("agg-1", "A", "");
    let e2 = store.append("agg-1", "B", "");
    let e3 = store.append("agg-2", "A", "");
    assert_eq!(e1.version, 1);
    assert_eq!(e2.version, 2);
    assert_eq!(e3.version, 1);
}

#[test]
fn test_simple_store_load_aggregate() {
    let mut store = SimpleEventStore::new();
    store.append("agg-1", "E1", "");
    store.append("agg-2", "E2", "");
    store.append("agg-1", "E3", "");
    let events = store.load_aggregate("agg-1");
    assert_eq!(events.len(), 2);
    assert!(events.iter().all(|e| e.aggregate_id == "agg-1"));
}

#[test]
fn test_simple_store_load_all_events() {
    let mut store = SimpleEventStore::new();
    store.append("a", "X", "");
    store.append("b", "Y", "");
    assert_eq!(store.load_all_events().len(), 2);
}

#[test]
fn test_simple_store_load_from_version() {
    let mut store = SimpleEventStore::new();
    for i in 1..=5u64 {
        store.append("agg", format!("E{i}"), "");
    }
    let events = store.load_from_version("agg", 3);
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].version, 3);
}

#[test]
fn test_simple_store_current_version() {
    let mut store = SimpleEventStore::new();
    store.append("a", "E", "");
    store.append("a", "E", "");
    assert_eq!(store.current_version("a"), 2);
    assert_eq!(store.current_version("nonexistent"), 0);
}

#[test]
fn test_simple_store_default() {
    let store = SimpleEventStore::default();
    assert!(store.is_empty());
}

// ── EventStreamIter ───────────────────────────────────────────────────────

#[test]
fn test_event_stream_iter_all_events() {
    let mut store = SimpleEventStore::new();
    for i in 0..5 {
        store.append("agg", format!("E{i}"), "");
    }
    let iter = EventStreamIter::new(store.load_all_events());
    assert_eq!(iter.count(), 5);
}

#[test]
fn test_event_stream_iter_filter_aggregate() {
    let mut store = SimpleEventStore::new();
    store.append("a", "E", "");
    store.append("b", "E", "");
    store.append("a", "E", "");
    let iter = EventStreamIter::new(store.load_all_events()).for_aggregate("a");
    assert_eq!(iter.count(), 2);
}

#[test]
fn test_event_stream_iter_filter_type() {
    let mut store = SimpleEventStore::new();
    store.append("a", "TypeA", "");
    store.append("a", "TypeB", "");
    let iter = EventStreamIter::new(store.load_all_events()).for_type("TypeA");
    assert_eq!(iter.count(), 1);
}

#[test]
fn test_event_stream_iter_empty() {
    let iter = EventStreamIter::new(vec![]);
    assert_eq!(iter.count(), 0);
}

// ── SimpleSnapshotStore ───────────────────────────────────────────────────

#[test]
fn test_snapshot_store_save_and_load() {
    let mut store = SimpleSnapshotStore::new();
    let snap = SimpleSnapshot {
        aggregate_id: "agg-1".to_string(),
        state: r#"{"count":5}"#.to_string(),
        version: 5,
    };
    store.save(snap.clone());
    let loaded = store.load_snapshot("agg-1").expect("snapshot should exist");
    assert_eq!(loaded.version, 5);
}

#[test]
fn test_snapshot_store_delete() {
    let mut store = SimpleSnapshotStore::new();
    store.save(SimpleSnapshot {
        aggregate_id: "a".into(),
        state: "s".into(),
        version: 1,
    });
    assert!(store.delete("a"));
    assert!(store.load_snapshot("a").is_none());
}

#[test]
fn test_snapshot_store_len() {
    let mut store = SimpleSnapshotStore::new();
    store.save(SimpleSnapshot {
        aggregate_id: "a".into(),
        state: "s".into(),
        version: 1,
    });
    store.save(SimpleSnapshot {
        aggregate_id: "b".into(),
        state: "s".into(),
        version: 1,
    });
    assert_eq!(store.len(), 2);
}

#[test]
fn test_snapshot_store_default() {
    let store = SimpleSnapshotStore::default();
    assert!(store.is_empty());
}

// ── SimpleEventBus ────────────────────────────────────────────────────────

fn dummy_simple_event(aggregate_id: &str, event_type: &str) -> SimpleEvent {
    SimpleEvent {
        id: 1,
        aggregate_id: aggregate_id.to_string(),
        event_type: event_type.to_string(),
        payload: String::new(),
        version: 1,
        timestamp: 0,
    }
}

#[test]
fn test_simple_event_bus_publish() {
    let mut bus = SimpleEventBus::new();
    let counter = std::sync::Arc::new(AtomicU64::new(0));
    let c = std::sync::Arc::clone(&counter);
    bus.subscribe(
        "OrderPlaced",
        std::sync::Arc::new(move |_ev| {
            c.fetch_add(1, AOrdering::SeqCst);
        }),
    );
    let ev = dummy_simple_event("agg", "OrderPlaced");
    bus.publish(&ev);
    assert_eq!(counter.load(AOrdering::SeqCst), 1);
}

#[test]
fn test_simple_event_bus_wildcard() {
    let mut bus = SimpleEventBus::new();
    let counter = std::sync::Arc::new(AtomicU64::new(0));
    let c = std::sync::Arc::clone(&counter);
    bus.subscribe(
        "*",
        std::sync::Arc::new(move |_ev| {
            c.fetch_add(1, AOrdering::SeqCst);
        }),
    );
    bus.publish(&dummy_simple_event("agg", "TypeA"));
    bus.publish(&dummy_simple_event("agg", "TypeB"));
    assert_eq!(counter.load(AOrdering::SeqCst), 2);
}

#[test]
fn test_simple_event_bus_no_fire_different_type() {
    let mut bus = SimpleEventBus::new();
    let counter = std::sync::Arc::new(AtomicU64::new(0));
    let c = std::sync::Arc::clone(&counter);
    bus.subscribe(
        "TypeA",
        std::sync::Arc::new(move |_ev| {
            c.fetch_add(1, AOrdering::SeqCst);
        }),
    );
    bus.publish(&dummy_simple_event("agg", "TypeB"));
    assert_eq!(counter.load(AOrdering::SeqCst), 0);
}

#[test]
fn test_simple_event_bus_counts() {
    let mut bus = SimpleEventBus::new();
    bus.subscribe("A", std::sync::Arc::new(|_| {}));
    bus.subscribe("A", std::sync::Arc::new(|_| {}));
    bus.subscribe("*", std::sync::Arc::new(|_| {}));
    assert_eq!(bus.subscription_count(), 2);
    assert_eq!(bus.wildcard_count(), 1);
}

#[test]
fn test_simple_event_bus_default() {
    let bus = SimpleEventBus::default();
    assert_eq!(bus.subscription_count(), 0);
    assert_eq!(bus.wildcard_count(), 0);
}

// ── ProjectionRunner ──────────────────────────────────────────────────────

#[test]
fn test_projection_runner_count() {
    let mut store = SimpleEventStore::new();
    for _ in 0..4 {
        store.append("agg", "E", "");
    }
    let mut runner = ProjectionRunner::new("counter");
    let count = runner.run(&store, 0u64, |acc, _| acc + 1);
    assert_eq!(count, 4);
    assert_eq!(runner.processed_count(), 4);
}

#[test]
fn test_projection_runner_aggregate() {
    let mut store = SimpleEventStore::new();
    store.append("a", "E", "");
    store.append("b", "E", "");
    store.append("a", "E", "");
    let mut runner = ProjectionRunner::new("agg-count");
    let count = runner.run_for_aggregate(&store, "a", 0u64, |acc, _| acc + 1);
    assert_eq!(count, 2);
}

#[test]
fn test_projection_runner_accumulates_payloads() {
    let mut store = SimpleEventStore::new();
    store.append("agg", "E", "hello");
    store.append("agg", "E", " world");
    let mut runner = ProjectionRunner::new("concat");
    let result = runner.run(&store, String::new(), |mut acc, ev| {
        acc.push_str(&ev.payload);
        acc
    });
    assert_eq!(result, "hello world");
}

#[test]
fn test_projection_runner_zero_processed_initially() {
    let runner = ProjectionRunner::new("test");
    assert_eq!(runner.processed_count(), 0);
}

#[test]
fn test_large_store() {
    let mut store = SimpleEventStore::new();
    for i in 0..1000u64 {
        store.append("agg", format!("E{i}"), format!("payload-{i}"));
    }
    assert_eq!(store.len(), 1000);
    assert_eq!(store.current_version("agg"), 1000);
}

#[test]
fn test_simple_event_equality() {
    let e1 = SimpleEvent {
        id: 1,
        aggregate_id: "a".into(),
        event_type: "T".into(),
        payload: "p".into(),
        version: 1,
        timestamp: 0,
    };
    let e2 = e1.clone();
    assert_eq!(e1, e2);
}
