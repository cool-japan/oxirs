//! W3-S11 — fault tolerance: checkpoint propagation + exactly-once.
//!
//! Drives a Chandy-Lamport-style checkpoint across a 4-operator pipeline,
//! then walks an exactly-once ingress through a dedup-then-commit cycle
//! and confirms no double-counts after a simulated kill/recover sequence.

#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;
use std::sync::Arc;

use oxirs_stream::fault_tolerance::{
    CheckpointController, CheckpointControllerConfig, EndToEndExactlyOnceCoordinator,
    ExactlyOnceCoordinatorConfig, IdempotentProducer, IdempotentProducerConfig,
    InMemoryCheckpointStore, MarkerPropagator, MarkerPropagatorEvent, OperatorSnapshot,
    ProducerStamp,
};
use oxirs_stream::state::distributed_state::{InMemoryStateBackend, StateBackend};

#[tokio::test]
async fn marker_propagation_completes_for_diamond_topology() {
    // 4-operator diamond:
    //                      ┌── op-b ──┐
    //   source ─→ op-a ─┤              ├─→ op-d
    //                      └── op-c ──┘
    let propagator = Arc::new(MarkerPropagator::new());
    let store = Arc::new(InMemoryCheckpointStore::new());
    let controller = CheckpointController::new(
        CheckpointControllerConfig::default(),
        propagator.clone(),
        store.clone(),
    );
    for op in ["op-a", "op-b", "op-c", "op-d"] {
        controller.register_operator(op.to_string());
    }
    propagator.register_operator("op-a".to_string(), ["src"]);
    propagator.register_operator("op-b".to_string(), ["a-out"]);
    propagator.register_operator("op-c".to_string(), ["a-out"]);
    propagator.register_operator("op-d".to_string(), ["b-out", "c-out"]);

    let marker = controller.open();

    // Marker flows down from op-a → op-b/op-c.
    propagator
        .on_marker(&"op-a".to_string(), &"src".to_string(), &marker)
        .expect("ok");
    let snap_a = OperatorSnapshot {
        operator_id: "op-a".to_string(),
        checkpoint_id: marker.checkpoint_id,
        state_blob: vec![1],
        channel_logs: HashMap::new(),
        completed_at_ms: 0,
    };
    let done = controller.commit_snapshot(snap_a).await.expect("ok");
    assert!(!done);

    propagator
        .on_marker(&"op-b".to_string(), &"a-out".to_string(), &marker)
        .expect("ok");
    let snap_b = OperatorSnapshot {
        operator_id: "op-b".to_string(),
        checkpoint_id: marker.checkpoint_id,
        state_blob: vec![2],
        channel_logs: HashMap::new(),
        completed_at_ms: 0,
    };
    let done = controller.commit_snapshot(snap_b).await.expect("ok");
    assert!(!done);

    propagator
        .on_marker(&"op-c".to_string(), &"a-out".to_string(), &marker)
        .expect("ok");
    let snap_c = OperatorSnapshot {
        operator_id: "op-c".to_string(),
        checkpoint_id: marker.checkpoint_id,
        state_blob: vec![3],
        channel_logs: HashMap::new(),
        completed_at_ms: 0,
    };
    let done = controller.commit_snapshot(snap_c).await.expect("ok");
    assert!(!done);

    // op-d has two input edges — it sees marker on both before completing.
    let ev1 = propagator
        .on_marker(&"op-d".to_string(), &"b-out".to_string(), &marker)
        .expect("ok");
    assert_eq!(ev1, MarkerPropagatorEvent::StartSnapshot);
    let ev2 = propagator
        .on_marker(&"op-d".to_string(), &"c-out".to_string(), &marker)
        .expect("ok");
    assert_eq!(ev2, MarkerPropagatorEvent::Completed);

    let snap_d = OperatorSnapshot {
        operator_id: "op-d".to_string(),
        checkpoint_id: marker.checkpoint_id,
        state_blob: vec![4],
        channel_logs: HashMap::new(),
        completed_at_ms: 0,
    };
    let done = controller.commit_snapshot(snap_d).await.expect("ok");
    assert!(done, "diamond checkpoint must be complete on op-d");

    let progress = controller.progress(marker.checkpoint_id).expect("progress");
    assert!(progress.is_complete());
    assert_eq!(progress.committed.len(), 4);
}

#[tokio::test]
async fn exactly_once_no_double_count_after_kill_recover() {
    let backend: Arc<dyn StateBackend> = Arc::new(InMemoryStateBackend::new());
    let coord = EndToEndExactlyOnceCoordinator::new(
        ExactlyOnceCoordinatorConfig::default(),
        backend.clone(),
    );
    // A single producer producing a count.
    let producer = IdempotentProducer::new(IdempotentProducerConfig {
        producer_id: "counter-producer".into(),
        partition: 0,
        initial_sequence: 0,
    });

    // Phase 1: observe 5 events.
    let key = b"events_total".to_vec();
    let mut total: u64 = 0;
    for _ in 0..5 {
        let stamp = producer.issue();
        if let Some(txn_id) = coord.begin_transaction(stamp.clone()).expect("ok") {
            total += 1;
            coord
                .add_state_change(&txn_id, key.clone(), total.to_le_bytes().to_vec())
                .expect("ok");
            coord.commit_transaction(&txn_id).expect("commit");
        }
    }
    assert_eq!(total, 5);
    let stored = backend.get(&key).expect("ok").expect("hit");
    assert_eq!(u64::from_le_bytes(stored.try_into().expect("8 bytes")), 5);

    // Phase 2: simulate "kill / restart" by replaying the *same* stamps from
    // sequences 0..5. They should all be filtered as duplicates.
    let mut filtered = 0u64;
    for seq in 0..5 {
        let stamp = ProducerStamp {
            producer_id: "counter-producer".into(),
            partition: 0,
            sequence: seq,
        };
        match coord.begin_transaction(stamp).expect("ok") {
            Some(_) => panic!("replayed stamp at seq {} should be deduped", seq),
            None => filtered += 1,
        }
    }
    assert_eq!(filtered, 5);

    // Counter must still read 5 — no double-counting.
    let stored = backend.get(&key).expect("ok").expect("hit");
    assert_eq!(u64::from_le_bytes(stored.try_into().expect("8 bytes")), 5);

    // Phase 3: real new events resume from sequence 5.
    let recovery_producer = IdempotentProducer::new(IdempotentProducerConfig {
        producer_id: "counter-producer".into(),
        partition: 0,
        initial_sequence: 5,
    });
    for _ in 0..3 {
        let stamp = recovery_producer.issue();
        if let Some(txn_id) = coord.begin_transaction(stamp).expect("ok") {
            total += 1;
            coord
                .add_state_change(&txn_id, key.clone(), total.to_le_bytes().to_vec())
                .expect("ok");
            coord.commit_transaction(&txn_id).expect("commit");
        }
    }
    let stored = backend.get(&key).expect("ok").expect("hit");
    assert_eq!(u64::from_le_bytes(stored.try_into().expect("8 bytes")), 8);
}

#[tokio::test]
async fn checkpoint_store_round_trip_via_controller() {
    let propagator = Arc::new(MarkerPropagator::new());
    let store = Arc::new(InMemoryCheckpointStore::new());
    let controller = CheckpointController::new(
        CheckpointControllerConfig::default(),
        propagator.clone(),
        store.clone(),
    );
    controller.register_operator("op".to_string());
    propagator.register_operator("op".to_string(), ["e"]);

    let marker = controller.open();
    propagator
        .on_marker(&"op".to_string(), &"e".to_string(), &marker)
        .expect("ok");
    let snap = OperatorSnapshot {
        operator_id: "op".to_string(),
        checkpoint_id: marker.checkpoint_id,
        state_blob: b"snapshot-body".to_vec(),
        channel_logs: HashMap::new(),
        completed_at_ms: 0,
    };
    controller.commit_snapshot(snap).await.expect("commit");

    let latest = controller.latest_committed().await.expect("ok");
    assert_eq!(latest, Some(marker.checkpoint_id));

    let stored = controller
        .store()
        .get(&"op".to_string(), marker.checkpoint_id)
        .await
        .expect("ok")
        .expect("hit");
    assert_eq!(stored.state_blob, b"snapshot-body".to_vec());
}

#[tokio::test]
async fn channel_logging_records_only_after_first_marker() {
    let propagator = Arc::new(MarkerPropagator::new());
    propagator.register_operator("op".to_string(), ["a", "b"]);
    let cp_id = 7u64;
    let marker = oxirs_stream::fault_tolerance::Marker::new(cp_id);

    // Pre-marker traffic on edge "b" should not be recorded.
    let recorded = propagator
        .record_inflight(
            &"op".to_string(),
            cp_id,
            &"b".to_string(),
            b"early".to_vec(),
        )
        .expect("ok");
    assert!(!recorded);

    // First marker arrives on edge "a".
    propagator
        .on_marker(&"op".to_string(), &"a".to_string(), &marker)
        .expect("ok");

    // Now traffic on edge "b" is recorded.
    let recorded = propagator
        .record_inflight(
            &"op".to_string(),
            cp_id,
            &"b".to_string(),
            b"in-flight".to_vec(),
        )
        .expect("ok");
    assert!(recorded);

    // Marker arrives on "b": completes the round.
    let ev = propagator
        .on_marker(&"op".to_string(), &"b".to_string(), &marker)
        .expect("ok");
    assert_eq!(ev, MarkerPropagatorEvent::Completed);

    let logs = propagator.drain_channel_logs(&"op".to_string(), cp_id);
    let on_b = logs.get("b").expect("logged");
    assert_eq!(on_b, &vec![b"in-flight".to_vec()]);
    assert!(!logs.contains_key("a"));
}
