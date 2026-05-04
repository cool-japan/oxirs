//! W3-S11 — distributed stream processing across clusters.
//!
//! Drives a 3-node distributed shard topology, ingests a stream of events,
//! kills the leader mid-stream, and asserts that no event is lost, every
//! shard ends up assigned to a surviving node, and the final assignment is
//! consistent across surviving nodes.

#![allow(clippy::uninlined_format_args)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use oxirs_cluster::stream_integration::StreamMessage;
use oxirs_cluster::streaming::cluster_sink::{SinkError, StreamSink};
use oxirs_stream::distributed::{
    CoordinatorConfig, DistributedStreamCoordinator, EventShipper, InProcessShipperTransport,
    ShippedEvent, ShipperConfig, ShipperTransport,
};

/// Mock cluster sink that simulates a Raft proposal pipeline. Tracks which
/// proposals have committed and never fails (so we exercise the coordinator
/// rather than the cluster).
#[derive(Default)]
struct MockClusterSink {
    proposals: Mutex<Vec<StreamMessage>>,
    commits: AtomicU64,
}

#[async_trait]
impl StreamSink for MockClusterSink {
    async fn write_batch(&self, events: Vec<StreamMessage>) -> Result<(), SinkError> {
        self.commits.fetch_add(1, Ordering::Relaxed);
        self.proposals.lock().extend(events);
        Ok(())
    }
}

/// A node in the simulated 3-node distributed pipeline.
struct Node {
    id: String,
    coord: Arc<DistributedStreamCoordinator>,
    rx: Mutex<Option<mpsc::Receiver<ShippedEvent>>>,
    /// Receiver for cross-node shipping. Held to keep the channel alive.
    transport_rx: Mutex<Option<mpsc::Receiver<ShippedEvent>>>,
    sink: Arc<MockClusterSink>,
}

/// Build a 3-node distributed topology that all three coordinators agree on.
async fn build_three_node_cluster(
    n_shards: u32,
) -> (Vec<Arc<Node>>, Arc<InProcessShipperTransport>) {
    // Shared transport so cross-node shipping works.
    let transport = Arc::new(InProcessShipperTransport::new(1024));
    let mut nodes = Vec::new();
    for id in ["n1", "n2", "n3"] {
        let sink = Arc::new(MockClusterSink::default());
        let shipper = Arc::new(EventShipper::new(
            ShipperConfig {
                local_node: id.into(),
            },
            transport.clone() as Arc<dyn ShipperTransport>,
        ));
        // Local sink for in-process delivery to this node.
        let (tx, rx) = mpsc::channel(1024);
        shipper.install_local_sink(tx);
        // Spawn a transport receiver so other nodes can ship to us. We
        // store it on the Node so it stays alive for the lifetime of the test.
        let t_rx = transport.spawn_receiver(id.to_string());
        let cfg = CoordinatorConfig {
            coord_id: "cluster-1".into(),
            local_node: id.into(),
            n_shards,
            stream_id: None,
            idempotent_membership: true,
        };
        let coord =
            Arc::new(DistributedStreamCoordinator::new(cfg, sink.clone(), shipper).expect("ok"));
        nodes.push(Arc::new(Node {
            id: id.into(),
            coord,
            rx: Mutex::new(Some(rx)),
            transport_rx: Mutex::new(Some(t_rx)),
            sink,
        }));
    }
    // Bootstrap: every coordinator joins every node so they share the same
    // assignment. In a real cluster this happens via Raft; for the test we
    // drive it directly.
    for node in &nodes {
        for peer in ["n1", "n2", "n3"] {
            node.coord.join(peer.into()).await.expect("join");
        }
    }
    (nodes, transport)
}

#[tokio::test]
async fn three_node_cluster_balances_shards() {
    let (nodes, _t) = build_three_node_cluster(9).await;
    // Assignment is deterministic — every node should agree.
    let a = nodes[0].coord.current_assignment();
    let b = nodes[1].coord.current_assignment();
    let c = nodes[2].coord.current_assignment();
    assert_eq!(a, b);
    assert_eq!(b, c);
    let counts = a.counts();
    for count in counts.values() {
        assert_eq!(*count, 3, "9 shards across 3 nodes should give 3 each");
    }
}

#[tokio::test]
async fn three_node_cluster_routes_no_event_lost_on_kill() {
    let (nodes, _t) = build_three_node_cluster(6).await;
    let n1 = nodes[0].clone();
    let n2 = nodes[1].clone();
    let n3 = nodes[2].clone();

    // Phase 1: route 100 events from n1's perspective.
    for i in 0..100 {
        let key = format!("k-{i}");
        let payload = serde_json::json!({"i": i});
        n1.coord.route(&key, &payload).await.expect("route");
    }

    // Snapshot: count events delivered to each node so far. Drains both the
    // local-delivery channel and the cross-node transport channel.
    let drain = |node: &Arc<Node>| -> Vec<ShippedEvent> {
        let mut out = Vec::new();
        let mut rx_opt = node.rx.lock().take();
        if let Some(rx) = rx_opt.as_mut() {
            while let Ok(ev) = rx.try_recv() {
                out.push(ev);
            }
        }
        *node.rx.lock() = rx_opt;
        let mut t_opt = node.transport_rx.lock().take();
        if let Some(rx) = t_opt.as_mut() {
            while let Ok(ev) = rx.try_recv() {
                out.push(ev);
            }
        }
        *node.transport_rx.lock() = t_opt;
        out
    };
    let pre1 = drain(&n1);
    let pre2 = drain(&n2);
    let pre3 = drain(&n3);
    let pre_total = pre1.len() + pre2.len() + pre3.len();
    assert_eq!(pre_total, 100, "every routed event should be delivered");

    // Phase 2: simulate killing n1 — the surviving cluster removes it from the
    // topology and rebalances. Both n2 and n3 do this so they stay in sync.
    n2.coord.leave("n1").await.expect("leave-on-n2");
    n3.coord.leave("n1").await.expect("leave-on-n3");
    let assn2 = n2.coord.current_assignment();
    let assn3 = n3.coord.current_assignment();
    assert_eq!(assn2, assn3, "surviving nodes converge on same assignment");
    for owner in assn2.map.values() {
        assert!(owner == "n2" || owner == "n3");
    }

    // Phase 3: route 100 more events from n2 — every event should land on a
    // surviving node.
    for i in 100..200 {
        let key = format!("k-{i}");
        let payload = serde_json::json!({"i": i});
        n2.coord.route(&key, &payload).await.expect("route");
    }
    let post2 = drain(&n2);
    let post3 = drain(&n3);
    let post_total = post2.len() + post3.len();
    assert_eq!(post_total, 100, "no events lost after node kill");
}

#[tokio::test]
async fn key_routing_is_deterministic_across_nodes() {
    let (nodes, _t) = build_three_node_cluster(8).await;
    // Decide a single shard for a deterministic key on each node and assert
    // they all agree.
    let key = "deterministic-key";
    let s1 = nodes[0].coord.shard_for_key_value(key).expect("shard");
    let s2 = nodes[1].coord.shard_for_key_value(key).expect("shard");
    let s3 = nodes[2].coord.shard_for_key_value(key).expect("shard");
    assert_eq!(s1, s2);
    assert_eq!(s2, s3);
    let owner1 = nodes[0]
        .coord
        .current_assignment()
        .owner_of(s1)
        .cloned()
        .expect("owner");
    let owner2 = nodes[1]
        .coord
        .current_assignment()
        .owner_of(s2)
        .cloned()
        .expect("owner");
    assert_eq!(owner1, owner2);
}

#[tokio::test]
async fn coordinator_persists_assignment_through_sink() {
    let (nodes, _t) = build_three_node_cluster(4).await;
    // Each node committed at least 3 proposals (one per join).
    for node in &nodes {
        let commits = node.sink.commits.load(Ordering::Relaxed);
        assert!(commits >= 3, "node {} commits = {}", node.id, commits);
    }
}
