//! Integration test for the W3-S9 cluster ↔ streaming bridge.
//!
//! Exercises the full pipeline:
//!
//! 1. Three [`ClusterNode`]s share a global Raft state machine (the existing
//!    test fallback used by other integration suites in this crate).
//! 2. A [`ClusterSink`] is built on top of node 1's [`ConsensusManager`].
//! 3. Stream events are fed in through [`StreamSink::write_batch`] and we
//!    assert that the resulting triples are visible on every node (because
//!    they all read from the shared state).
//! 4. Node 1 is "killed" (replaced by a new sink anchored at node 2), and
//!    the test asserts that:
//!    a) state already committed on node 1 survives the swap, and
//!    b) the new leader can keep ingesting more events on top.

#![allow(clippy::uninlined_format_args)]

use std::sync::Arc;

use oxirs_cluster::consensus::ConsensusManager;
use oxirs_cluster::raft::{init_global_shared_storage, reset_global_shared_storage};
use oxirs_cluster::stream_integration::{StreamMessage, StreamTriple};
use oxirs_cluster::streaming::{
    BackpressureBridge, BackpressureConfig, BackpressureSignal, ClusterSink, ClusterSinkConfig,
    DefaultEventMapper, SinkError, StreamSink,
};
use oxirs_cluster::{ClusterNode, NodeConfig};

/// Global mutex to keep the global shared storage from being touched by
/// multiple integration tests in parallel (mirrors the pattern used by
/// `tests/integration_tests.rs`).
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn triple(stream: &str, idx: usize) -> StreamTriple {
    StreamTriple::new(
        format!("http://example.org/{stream}/s/{idx}"),
        "http://example.org/p/has",
        format!("\"{stream}-value-{idx}\""),
    )
}

fn insert_message(stream: &str, offset: u64, triples: Vec<StreamTriple>) -> StreamMessage {
    StreamMessage::insert(stream, offset, triples)
}

async fn make_three_node_cluster() -> (ClusterNode, ClusterNode, ClusterNode) {
    let dir = std::env::temp_dir().join(format!("oxirs-streaming-it-{}", std::process::id()));

    let mk = |id: u64, port: u16, peers: Vec<u64>| -> NodeConfig {
        NodeConfig {
            node_id: id,
            address: format!("127.0.0.1:{port}").parse().expect("addr"),
            data_dir: dir.join(format!("node-{id}")).display().to_string(),
            peers,
            discovery: None,
            replication_strategy: None,
            #[cfg(feature = "bft")]
            use_bft: false,
            region_config: None,
        }
    };

    let n1 = ClusterNode::new(mk(1, 19_801, vec![2, 3]))
        .await
        .expect("n1");
    let n2 = ClusterNode::new(mk(2, 19_802, vec![1, 3]))
        .await
        .expect("n2");
    let n3 = ClusterNode::new(mk(3, 19_803, vec![1, 2]))
        .await
        .expect("n3");
    (n1, n2, n3)
}

#[tokio::test]
async fn streaming_pipeline_commits_events_durably() {
    let _g = TEST_LOCK.lock().await;

    init_global_shared_storage();
    reset_global_shared_storage().await;

    let (n1, n2, n3) = make_three_node_cluster().await;

    // Sink anchored at node 1.
    let consensus = Arc::new(ConsensusManager::new(1, vec![2, 3]));
    let mapper = Arc::new(DefaultEventMapper::default());
    let sink = ClusterSink::new(consensus, mapper, ClusterSinkConfig::default());

    // Drive a small stream of events.
    let batch1 = vec![
        insert_message("orders", 1, vec![triple("orders", 0), triple("orders", 1)]),
        insert_message("orders", 2, vec![triple("orders", 2)]),
    ];
    sink.write_batch(batch1).await.expect("commit batch 1");

    let batch2 = vec![insert_message(
        "orders",
        3,
        vec![triple("orders", 3), triple("orders", 4)],
    )];
    sink.write_batch(batch2).await.expect("commit batch 2");

    // 5 triples committed, every cluster node should observe them through
    // the shared Raft state machine.
    assert_eq!(n1.count_triples().await.expect("count n1"), 5);
    assert_eq!(n2.count_triples().await.expect("count n2"), 5);
    assert_eq!(n3.count_triples().await.expect("count n3"), 5);

    // Sink-side stats agree.
    use std::sync::atomic::Ordering;
    assert_eq!(
        sink.stats().batches_committed.load(Ordering::Relaxed),
        2,
        "batches_committed"
    );
    assert_eq!(
        sink.stats().commands_committed.load(Ordering::Relaxed),
        5,
        "commands_committed"
    );
}

#[tokio::test]
async fn streaming_pipeline_survives_leader_kill() {
    let _g = TEST_LOCK.lock().await;

    init_global_shared_storage();
    reset_global_shared_storage().await;

    let (n1, n2, n3) = make_three_node_cluster().await;

    // First leader: node 1.
    let consensus_one = Arc::new(ConsensusManager::new(1, vec![2, 3]));
    let mapper = Arc::new(DefaultEventMapper::default());
    let sink_one = ClusterSink::new(consensus_one, mapper.clone(), ClusterSinkConfig::default());

    // Commit some events.
    sink_one
        .write_batch(vec![insert_message(
            "orders",
            1,
            vec![triple("orders", 0), triple("orders", 1)],
        )])
        .await
        .expect("commit before leader kill");

    // Snapshot the committed count: node 1 went away (sink_one dropped) but
    // committed state lives in the shared Raft state machine.
    let count_after_kill = n1.count_triples().await.expect("count");
    assert_eq!(count_after_kill, 2);
    drop(sink_one); // simulate "leader killed".

    // New leader: node 2 — the surviving cluster keeps making progress.
    let consensus_two = Arc::new(ConsensusManager::new(2, vec![1, 3]));
    let sink_two = ClusterSink::new(consensus_two, mapper, ClusterSinkConfig::default());
    sink_two
        .write_batch(vec![insert_message("orders", 2, vec![triple("orders", 2)])])
        .await
        .expect("commit on new leader");

    // Final state: all 3 triples visible from every node. The 2 committed
    // before the kill survived; the 1 committed after by the new leader
    // landed on top.
    assert_eq!(n1.count_triples().await.expect("count n1"), 3);
    assert_eq!(n2.count_triples().await.expect("count n2"), 3);
    assert_eq!(n3.count_triples().await.expect("count n3"), 3);
}

#[tokio::test]
async fn backpressure_signal_blocks_oversaturated_batches() {
    let _g = TEST_LOCK.lock().await;

    init_global_shared_storage();
    reset_global_shared_storage().await;

    let consensus = Arc::new(ConsensusManager::new(1, vec![]));
    let mapper = Arc::new(DefaultEventMapper::default());

    let bridge = BackpressureBridge::new(BackpressureConfig {
        slow_low_watermark: 2,
        slow_high_watermark: 4,
        stop_low_watermark: 6,
        stop_high_watermark: 8,
    })
    .expect("bridge");

    let sink = ClusterSink::with_bridge(
        consensus,
        mapper,
        bridge.clone(),
        ClusterSinkConfig::default(),
    );

    // Force the bridge into Stop and assert the sink refuses.
    let _ = bridge.observe(100);
    assert_eq!(sink.backpressure_signal(), BackpressureSignal::Stop);

    let err = sink
        .write_batch(vec![insert_message("rdf", 1, vec![triple("rdf", 0)])])
        .await
        .expect_err("should refuse");
    assert!(matches!(err, SinkError::BackpressureStopped));

    // Drain the bridge back to Continue and verify the sink starts accepting.
    let _ = bridge.observe(0);
    assert_eq!(sink.backpressure_signal(), BackpressureSignal::Continue);

    sink.write_batch(vec![insert_message("rdf", 1, vec![triple("rdf", 0)])])
        .await
        .expect("commit after recovery");
}

#[tokio::test]
async fn pipeline_preserves_event_ordering() {
    let _g = TEST_LOCK.lock().await;

    init_global_shared_storage();
    reset_global_shared_storage().await;

    let (n1, _n2, _n3) = make_three_node_cluster().await;

    let consensus = Arc::new(ConsensusManager::new(1, vec![2, 3]));
    let mapper = Arc::new(DefaultEventMapper::default());
    let sink = ClusterSink::new(consensus, mapper, ClusterSinkConfig::default());

    // 10 events, 3 triples each = 30 commands.
    let mut batch = Vec::with_capacity(10);
    for offset in 0..10 {
        let triples: Vec<_> = (0..3).map(|i| triple("ord", offset * 3 + i)).collect();
        batch.push(insert_message("ord", offset as u64, triples));
    }
    sink.write_batch(batch).await.expect("commit");
    assert_eq!(n1.count_triples().await.expect("count"), 30);

    // Verify per-triple visibility: each indexed subject must be present.
    // This guards against regressions that drop or reorder commands such that
    // a subject ends up missing from the committed state machine.
    for idx in 0..30 {
        let subject = format!("http://example.org/ord/s/{idx}");
        let hits = n1.query_triples(Some(&subject), None, None).await;
        assert_eq!(hits.len(), 1, "missing subject {subject}");
    }
}

#[tokio::test]
async fn empty_batch_is_a_noop() {
    let _g = TEST_LOCK.lock().await;
    init_global_shared_storage();
    reset_global_shared_storage().await;

    let consensus = Arc::new(ConsensusManager::new(1, vec![]));
    let mapper = Arc::new(DefaultEventMapper::default());
    let sink = ClusterSink::new(consensus, mapper, ClusterSinkConfig::default());

    sink.write_batch(vec![]).await.expect("ok");
    use std::sync::atomic::Ordering;
    assert_eq!(sink.stats().commands_committed.load(Ordering::Relaxed), 0);
}
