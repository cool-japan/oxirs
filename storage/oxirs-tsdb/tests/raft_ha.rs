//! Integration tests for Raft-based HA replication in oxirs-tsdb.
//!
//! Covers:
//! - `TsdbRaftOp` serialisation round-trips
//! - `WalReplicator` channel mechanics
//! - `TsdbStateMachine` linearisability checks
//! - `TsdbRaftSnapshot` creation, restoration, and network wire format
//! - `SnapshotStore` install/reject semantics
//! - `ReplicationGroup` 3-node leader election and log replication

use oxirs_tsdb::replication::{
    snapshot::{SnapshotDataPoint, SnapshotStore, TsdbRaftSnapshot},
    wal_replicator::{ApplyError, ReplicationError, TsdbRaftOp, TsdbStateMachine, WalReplicator},
    ReplicationGroup, WriteEntry,
};
use std::collections::HashMap;
use std::sync::mpsc;

// ── helper constructors ───────────────────────────────────────────────────────

fn insert_op(series: &str, ts: i64, val: f64) -> TsdbRaftOp {
    TsdbRaftOp::Insert {
        series_id: series.into(),
        timestamp: ts,
        value: val,
        tags: HashMap::new(),
    }
}

fn insert_op_with_tag(series: &str, ts: i64, val: f64, k: &str, v: &str) -> TsdbRaftOp {
    TsdbRaftOp::Insert {
        series_id: series.into(),
        timestamp: ts,
        value: val,
        tags: [(k.to_string(), v.to_string())].into(),
    }
}

// ── TsdbRaftOp round-trip ─────────────────────────────────────────────────────

#[test]
fn insert_op_serialises_and_deserialises() {
    let op = insert_op_with_tag("temp.sensor1", 1_714_300_000, 22.5, "location", "room1");
    let bytes = serde_json::to_vec(&op).expect("serialise");
    let decoded: TsdbRaftOp = serde_json::from_slice(&bytes).expect("deserialise");
    if let TsdbRaftOp::Insert {
        series_id,
        value,
        tags,
        ..
    } = decoded
    {
        assert_eq!(series_id, "temp.sensor1");
        assert!((value - 22.5).abs() < 1e-10);
        assert_eq!(tags["location"], "room1");
    } else {
        panic!("expected Insert variant after round-trip");
    }
}

#[test]
fn delete_op_serialises_and_deserialises() {
    let op = TsdbRaftOp::Delete {
        series_id: "s1".into(),
        from_ts: 1000,
        to_ts: 2000,
    };
    let bytes = serde_json::to_vec(&op).expect("serialise");
    let decoded: TsdbRaftOp = serde_json::from_slice(&bytes).expect("deserialise");
    assert_eq!(op, decoded);
}

#[test]
fn truncate_op_serialises_and_deserialises() {
    let op = TsdbRaftOp::Truncate {
        series_id: "metrics.cpu".into(),
    };
    let bytes = serde_json::to_vec(&op).expect("serialise");
    let decoded: TsdbRaftOp = serde_json::from_slice(&bytes).expect("deserialise");
    assert_eq!(op, decoded);
}

#[test]
fn op_built_in_round_trip_helpers() {
    let op = insert_op("a.b", 42, 0.5);
    let bytes = op.to_bytes().expect("to_bytes");
    let restored = TsdbRaftOp::from_bytes(&bytes).expect("from_bytes");
    assert_eq!(op, restored);
}

// ── WalReplicator ─────────────────────────────────────────────────────────────

#[test]
fn wal_replicator_sends_bytes_to_channel() {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let replicator = WalReplicator::new(tx);
    let op = insert_op("cpu.load", 1000, 0.85);
    replicator.replicate(&op).expect("replicate");
    let bytes = rx.recv().expect("receive bytes");
    let decoded: TsdbRaftOp = serde_json::from_slice(&bytes).expect("decode");
    assert!(matches!(decoded, TsdbRaftOp::Insert { .. }));
}

#[test]
fn wal_replicator_channel_closed_gives_channel_closed_error() {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    drop(rx);
    let replicator = WalReplicator::new(tx);
    let result = replicator.replicate(&TsdbRaftOp::Truncate {
        series_id: "s".into(),
    });
    assert!(
        matches!(result, Err(ReplicationError::ChannelClosed)),
        "expected ChannelClosed, got {result:?}"
    );
}

#[test]
fn wal_replicator_batch_ships_all_ops() {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let replicator = WalReplicator::new(tx);
    let ops = vec![
        insert_op("a", 1, 1.0),
        TsdbRaftOp::Delete {
            series_id: "b".into(),
            from_ts: 0,
            to_ts: 100,
        },
        TsdbRaftOp::Truncate {
            series_id: "c".into(),
        },
    ];
    replicator.replicate_batch(&ops).expect("batch replicate");
    let mut received = vec![];
    for _ in 0..3 {
        received.push(rx.recv().expect("recv"));
    }
    assert_eq!(received.len(), 3);
}

#[test]
fn wal_replicator_clone_shares_sender() {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let r1 = WalReplicator::new(tx);
    let r2 = r1.clone();
    r1.replicate(&insert_op("x", 1, 1.0)).expect("r1");
    r2.replicate(&insert_op("y", 2, 2.0)).expect("r2");
    let _b1 = rx.recv().expect("receive 1");
    let _b2 = rx.recv().expect("receive 2");
}

// ── TsdbStateMachine ──────────────────────────────────────────────────────────

#[test]
fn state_machine_starts_empty() {
    let sm = TsdbStateMachine::new();
    assert_eq!(sm.last_applied_index, 0);
    assert_eq!(sm.applied_count(), 0);
}

#[test]
fn state_machine_applies_sequential_ops() {
    let mut sm = TsdbStateMachine::new();
    sm.apply(1, insert_op("s", 100, 1.0)).expect("apply 1");
    sm.apply(
        2,
        TsdbRaftOp::Delete {
            series_id: "s".into(),
            from_ts: 0,
            to_ts: 50,
        },
    )
    .expect("apply 2");
    sm.apply(
        3,
        TsdbRaftOp::Truncate {
            series_id: "s".into(),
        },
    )
    .expect("apply 3");
    assert_eq!(sm.applied_count(), 3);
    assert_eq!(sm.last_applied_index, 3);
}

#[test]
fn state_machine_rejects_index_gap() {
    let mut sm = TsdbStateMachine::new();
    sm.apply(1, insert_op("s", 0, 0.0)).expect("apply 1");
    let result = sm.apply(3, insert_op("s", 1, 1.0)); // skipped 2
    assert!(
        matches!(
            result,
            Err(ApplyError::IndexGap {
                expected: 2,
                got: 3
            })
        ),
        "expected IndexGap {{expected:2, got:3}}, got {result:?}"
    );
}

#[test]
fn state_machine_rejects_duplicate_index() {
    let mut sm = TsdbStateMachine::new();
    sm.apply(1, insert_op("s", 0, 0.0)).expect("apply 1");
    let result = sm.apply(1, insert_op("s", 0, 0.0));
    assert!(result.is_err());
}

// ── TsdbRaftSnapshot ──────────────────────────────────────────────────────────

fn sample_points() -> Vec<SnapshotDataPoint> {
    vec![
        SnapshotDataPoint {
            series_id: "cpu.usage".into(),
            timestamp: 1_000,
            value: 42.0,
        },
        SnapshotDataPoint {
            series_id: "mem.free".into(),
            timestamp: 2_000,
            value: 1024.0,
        },
    ]
}

#[test]
fn snapshot_roundtrips_data_points() {
    let snap = TsdbRaftSnapshot::from_data_points(5, 2, &sample_points()).expect("create");
    let restored = snap.to_data_points().expect("restore");
    assert_eq!(restored, sample_points());
}

#[test]
fn snapshot_roundtrips_wire_bytes() {
    let snap = TsdbRaftSnapshot::from_data_points(3, 1, &sample_points()).expect("create");
    let wire = snap.to_wire_bytes().expect("wire encode");
    let decoded = TsdbRaftSnapshot::from_wire_bytes(&wire).expect("wire decode");
    assert_eq!(decoded.last_included_index, 3);
    assert_eq!(decoded.last_included_term, 1);
    let pts = decoded.to_data_points().expect("points");
    assert_eq!(pts, sample_points());
}

#[test]
fn snapshot_zero_index_rejected() {
    let result = TsdbRaftSnapshot::from_data_points(0, 1, &[]);
    assert!(result.is_err());
}

#[test]
fn snapshot_supersedes_older_index() {
    let old = TsdbRaftSnapshot::from_data_points(1, 1, &[]).unwrap();
    let new = TsdbRaftSnapshot::from_data_points(2, 1, &[]).unwrap();
    assert!(new.supersedes(&old));
    assert!(!old.supersedes(&new));
}

#[test]
fn snapshot_supersedes_same_index_higher_term() {
    let old = TsdbRaftSnapshot::from_data_points(3, 1, &[]).unwrap();
    let new = TsdbRaftSnapshot::from_data_points(3, 2, &[]).unwrap();
    assert!(new.supersedes(&old));
}

#[test]
fn empty_snapshot_valid() {
    let snap = TsdbRaftSnapshot::from_data_points(1, 1, &[]).expect("create");
    assert!(snap.to_data_points().expect("restore").is_empty());
}

// ── SnapshotStore ─────────────────────────────────────────────────────────────

#[test]
fn snapshot_store_initially_empty() {
    let store = SnapshotStore::new();
    assert!(store.latest().is_none());
    assert_eq!(store.last_included_index(), 0);
}

#[test]
fn snapshot_store_installs_first_snapshot() {
    let mut store = SnapshotStore::new();
    assert!(store.install(TsdbRaftSnapshot::from_data_points(1, 1, &[]).unwrap()));
    assert_eq!(store.last_included_index(), 1);
}

#[test]
fn snapshot_store_accepts_superseding_snapshot() {
    let mut store = SnapshotStore::new();
    store.install(TsdbRaftSnapshot::from_data_points(1, 1, &[]).unwrap());
    assert!(store.install(TsdbRaftSnapshot::from_data_points(5, 2, &[]).unwrap()));
    assert_eq!(store.last_included_index(), 5);
}

#[test]
fn snapshot_store_rejects_older_snapshot() {
    let mut store = SnapshotStore::new();
    store.install(TsdbRaftSnapshot::from_data_points(10, 3, &[]).unwrap());
    assert!(!store.install(TsdbRaftSnapshot::from_data_points(5, 2, &[]).unwrap()));
    assert_eq!(store.last_included_index(), 10);
}

// ── ReplicationGroup (3-node Raft) ────────────────────────────────────────────
//
// Use deterministic election timeouts [3, 5, 7] (same pattern as the unit
// tests in replication_group.rs) so that leader election converges quickly
// and reliably in a test context.

/// Build a deterministic 3-node group: node-0 times out at 3 ticks, node-1
/// at 5 ticks, node-2 at 7 ticks.
fn det_group() -> ReplicationGroup {
    ReplicationGroup::new(&["node-0", "node-1", "node-2"], Some(&[3, 5, 7]))
}

#[test]
fn three_node_group_elects_single_leader() {
    let mut group = det_group();
    group.tick_n(20);
    assert_eq!(
        group.leader_count(),
        1,
        "exactly one leader after tick_n(20)"
    );
}

#[test]
fn three_node_group_replicates_write_entry() {
    let mut group = det_group();
    group.tick_n(20);
    let entry = WriteEntry::new(1_714_300_000, "cpu.load", 0.75);
    let result = group.propose_and_commit(&entry, 20);
    assert!(
        result.is_ok(),
        "propose_and_commit should succeed: {result:?}"
    );
}

#[test]
fn three_node_group_node_ids_are_correct() {
    let group = det_group();
    let ids = group.node_ids();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&"node-0".to_string()));
    assert!(ids.contains(&"node-1".to_string()));
    assert!(ids.contains(&"node-2".to_string()));
}

#[test]
fn three_node_group_partition_reduces_leaders_to_zero() {
    let mut group = det_group();
    group.tick_n(20);
    group.partition("node-0");
    group.partition("node-1");
    group.partition("node-2");
    // No quorum → no leader.
    assert_eq!(group.leader_count(), 0);
}

#[test]
fn three_node_group_heals_after_partition() {
    let mut group = det_group();
    group.tick_n(20);
    group.partition("node-0");
    group.heal("node-0");
    group.tick_n(20);
    // At most one leader after healing (split-brain not possible with quorum).
    assert!(group.leader_count() <= 1);
}

#[test]
fn three_node_group_multiple_writes_succeed() {
    let mut group = det_group();
    group.tick_n(20);
    for i in 0_i64..5 {
        let entry = WriteEntry::new(i * 1000, format!("metric_{i}"), i as f64);
        let result = group.propose_and_commit(&entry, 20);
        assert!(result.is_ok(), "write {i} should succeed");
    }
}
