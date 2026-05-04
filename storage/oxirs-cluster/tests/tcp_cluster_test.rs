//! Phase C integration tests: real-TCP-network gossip and replication harness.
//!
//! Every test uses real TCP sockets on `127.0.0.1:0` (OS-assigned ports).
//! No `#[ignore]` — these are designed to complete well within 5 s on any
//! modern machine.
//!
//! Port strategy: all tests bind port 0.  The OS assigns a free ephemeral
//! port; there are no inter-test conflicts regardless of execution order or
//! parallelism.

use std::net::SocketAddr;
use std::time::Duration;

use oxirs_cluster::gossip::fanout::GossipFanout;
use oxirs_cluster::{
    ClusterMessage, GossipState, MessageCodec, NetworkStats, TcpClusterNetwork, TcpClusterNode,
    TcpNodeConfig, TcpNodeError,
};
// `set_with_version_on` is called via `TcpClusterNetwork` in the LWW test.
use tokio::net::TcpStream;
use tokio::time::sleep;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: short deadline for assertion messages
// ─────────────────────────────────────────────────────────────────────────────

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Two nodes — basic gossip convergence
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_two_nodes_gossip_converge() {
    let cfg_a = TcpNodeConfig {
        node_id: "a".into(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        fanout: GossipFanout::Unbounded,
        gossip_interval_ms: 20,
    };
    let cfg_b = TcpNodeConfig {
        node_id: "b".into(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        fanout: GossipFanout::Unbounded,
        gossip_interval_ms: 20,
    };

    let node_a = TcpClusterNode::start(cfg_a).await.expect("start a");
    let node_b = TcpClusterNode::start(cfg_b).await.expect("start b");

    node_a.add_peer(node_b.addr());
    node_b.add_peer(node_a.addr());

    node_a.set("foo", 42);

    // Wait up to 500 ms for node_b to see "foo" = 42.
    let deadline = tokio::time::Instant::now() + ms(500);
    loop {
        if node_b.get("foo") == Some(42) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "node_b did not receive 'foo'=42 within 500 ms"
        );
        sleep(ms(10)).await;
    }

    node_a.shutdown();
    node_b.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Three nodes — chain-like propagation with fanout = 1
// ─────────────────────────────────────────────────────────────────────────────

/// With `Bounded(1)` each node gossips to exactly one peer per round.
/// Convergence is still guaranteed because both A and B gossip outward.
#[tokio::test]
async fn test_three_nodes_chain_propagation() {
    let make_cfg = |id: &str| TcpNodeConfig {
        node_id: id.into(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        fanout: GossipFanout::Bounded(1),
        gossip_interval_ms: 20,
    };

    let node_a = TcpClusterNode::start(make_cfg("a")).await.expect("a");
    let node_b = TcpClusterNode::start(make_cfg("b")).await.expect("b");
    let node_c = TcpClusterNode::start(make_cfg("c")).await.expect("c");

    // Full-mesh wiring so each node can eventually reach the others.
    node_a.add_peer(node_b.addr());
    node_a.add_peer(node_c.addr());
    node_b.add_peer(node_a.addr());
    node_b.add_peer(node_c.addr());
    node_c.add_peer(node_a.addr());
    node_c.add_peer(node_b.addr());

    node_a.set("chain", 7);

    let deadline = tokio::time::Instant::now() + ms(1000);
    loop {
        let c_has_it = node_c.get("chain") == Some(7);
        let b_has_it = node_b.get("chain") == Some(7);
        if c_has_it && b_has_it {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "chain propagation did not complete within 1 s"
        );
        sleep(ms(10)).await;
    }

    node_a.shutdown();
    node_b.shutdown();
    node_c.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Five nodes — sqrt fanout converges
// ─────────────────────────────────────────────────────────────────────────────

/// `floor(sqrt(5)) = 2`.  Each node contacts 2 peers per round.
#[tokio::test]
async fn test_five_nodes_sqrt_fanout_converge() {
    let net = TcpClusterNetwork::spawn(5, 0, GossipFanout::Sqrt, 25)
        .await
        .expect("spawn 5");

    net.set_on(0, "sqrt_key", 55);

    let stats = net.wait_converged("sqrt_key", 55, ms(1500)).await;
    assert!(
        stats.converged,
        "5-node Sqrt fanout did not converge in 1.5 s (rounds={})",
        stats.rounds_to_converge
    );

    net.shutdown_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. State length counts distinct keys
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_state_len_counts_distinct_keys() {
    let cfg = TcpNodeConfig::localhost("solo", 0);
    let node = TcpClusterNode::start(cfg).await.expect("start");

    assert_eq!(node.state_len(), 0);
    node.set("a", 1);
    node.set("b", 2);
    node.set("a", 9); // update existing key
    assert_eq!(node.state_len(), 2, "should have 2 distinct keys");

    node.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Multiple keys all propagate
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_multiple_keys_all_propagate() {
    let net = TcpClusterNetwork::spawn(3, 0, GossipFanout::Unbounded, 20)
        .await
        .expect("spawn 3");

    let keys: &[(&str, u64)] = &[
        ("alpha", 1),
        ("beta", 2),
        ("gamma", 3),
        ("delta", 4),
        ("epsilon", 5),
    ];

    for &(k, v) in keys {
        net.set_on(0, k, v);
    }

    // Wait for all 5 keys to appear on node 2 with correct values.
    let deadline = tokio::time::Instant::now() + ms(1500);
    loop {
        let all_ok = keys.iter().all(|&(k, v)| net.get_on(2, k) == Some(v));
        if all_ok {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "not all keys propagated to node 2 within 1.5 s"
        );
        sleep(ms(10)).await;
    }

    net.shutdown_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. LWW — last write (highest version) wins
// ─────────────────────────────────────────────────────────────────────────────

/// Set the same key on two different nodes with explicit versions.
/// The write carrying the higher version must win on all nodes.
#[tokio::test]
async fn test_lww_last_write_wins() {
    let net = TcpClusterNetwork::spawn(2, 0, GossipFanout::Unbounded, 20)
        .await
        .expect("spawn 2");

    // Node 0 writes with a low explicit version (the "loser").
    net.set_with_version_on(0, "lww", 100, 1);
    // Node 1 writes with a much higher explicit version (the "winner").
    // Both gossip outward; the higher version must prevail everywhere.
    net.set_with_version_on(1, "lww", 200, 1000);

    // All nodes should converge to 200 (the higher-versioned write).
    let stats = net.wait_converged("lww", 200, ms(1000)).await;
    assert!(stats.converged, "LWW did not converge to 200 within 1 s");

    net.shutdown_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Graceful shutdown
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_graceful_shutdown() {
    let net = TcpClusterNetwork::spawn(3, 0, GossipFanout::Unbounded, 30)
        .await
        .expect("spawn 3");

    net.set_on(0, "tmp", 1);
    // Let gossip run briefly.
    sleep(ms(60)).await;
    // Shutdown should not panic.
    net.shutdown_all();
    // Give tasks a moment to exit.
    sleep(ms(50)).await;
    // If we reach here without a panic the test passes.
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Ping / Pong round-trip via raw TCP
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ping_pong() {
    let cfg = TcpNodeConfig::localhost("pong-target", 0);
    let node = TcpClusterNode::start(cfg).await.expect("start");
    let addr: SocketAddr = node.addr();

    // Connect manually.
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    let ping = ClusterMessage::Ping {
        sender_id: "tester".into(),
        seq: 99,
    };
    MessageCodec::write(&mut stream, &ping)
        .await
        .expect("write ping");

    let reply = MessageCodec::read(&mut stream).await.expect("read pong");
    match reply {
        ClusterMessage::Pong { seq, .. } => assert_eq!(seq, 99),
        other => panic!("expected Pong, got {other:?}"),
    }

    node.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Codec write/read round-trip (in-process via duplex)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_codec_write_read_roundtrip() {
    use tokio::io::duplex;

    let msg = ClusterMessage::Gossip {
        sender_id: "codec-test".into(),
        key: "hello".into(),
        value: 42,
        version: 7,
    };

    let (mut write_half, mut read_half) = duplex(4096);
    MessageCodec::write(&mut write_half, &msg)
        .await
        .expect("write");
    let received = MessageCodec::read(&mut read_half).await.expect("read");

    assert_eq!(msg, received);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. 10-node network — sqrt fanout, 3 keys, full convergence
// ─────────────────────────────────────────────────────────────────────────────

/// `floor(sqrt(10)) = 3`.  This test verifies that a 10-node network with
/// Sqrt fanout converges 3 keys within 3 s.
#[tokio::test]
async fn test_network_spawn_10_nodes_converge() {
    let net = TcpClusterNetwork::spawn(10, 0, GossipFanout::Sqrt, 30)
        .await
        .expect("spawn 10");

    let keys: &[(&str, u64)] = &[("x", 10), ("y", 20), ("z", 30)];
    for &(k, v) in keys {
        net.set_on(0, k, v);
    }

    // We check each key independently to get a precise failure message.
    for &(k, v) in keys {
        let stats: NetworkStats = net.wait_converged(k, v, ms(3000)).await;
        assert!(
            stats.converged,
            "key '{k}'={v} did not converge on all 10 nodes within 3 s \
             (rounds={}, time={}ms)",
            stats.rounds_to_converge, stats.total_time_ms
        );
    }

    net.shutdown_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Shutdown all — no panics after gossip rounds
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_network_shutdown_all() {
    let net = TcpClusterNetwork::spawn(5, 0, GossipFanout::Unbounded, 25)
        .await
        .expect("spawn 5");

    net.set_on(0, "shutdown_key", 1);
    // Let gossip run for a couple of rounds.
    sleep(ms(100)).await;
    // shutdown_all must not panic.
    net.shutdown_all();
    sleep(ms(50)).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. TcpNodeConfig::localhost produces a valid config
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_node_config_localhost() {
    let cfg = TcpNodeConfig::localhost("my-node", 0);
    assert_eq!(cfg.node_id, "my-node");
    assert_eq!(cfg.bind_addr, "127.0.0.1:0".parse::<SocketAddr>().unwrap());
    // Verify that a node actually starts with this config (port 0 → OS assigns).
    let node = TcpClusterNode::start(cfg)
        .await
        .expect("node should start from localhost config");
    assert_ne!(node.addr().port(), 0, "OS should assign a non-zero port");
    assert_eq!(node.addr().ip(), std::net::IpAddr::from([127, 0, 0, 1]));
    node.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. GossipState directly: empty and len semantics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_gossip_state_empty() {
    let state = GossipState::default();
    assert!(state.is_empty());
    assert_eq!(state.len(), 0);
    assert_eq!(state.get("missing"), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. TcpNodeError Display
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_tcp_node_error_display() {
    let err = TcpNodeError::Shutdown;
    let msg = err.to_string();
    assert!(msg.contains("shut down"), "unexpected: {msg}");

    let err2 = TcpNodeError::SendError("oops".into());
    let msg2 = err2.to_string();
    assert!(msg2.contains("oops"), "unexpected: {msg2}");
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Replicate RPC acknowledged over a real socket
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_replicate_ack_over_tcp() {
    let cfg = TcpNodeConfig::localhost("follower", 0);
    let node = TcpClusterNode::start(cfg).await.expect("start");
    let addr = node.addr();

    let mut stream = TcpStream::connect(addr).await.expect("connect");
    let replicate = ClusterMessage::Replicate {
        leader_id: "leader".into(),
        index: 5,
        term: 2,
        checksum: 0xABCD,
    };
    MessageCodec::write(&mut stream, &replicate)
        .await
        .expect("write replicate");

    let reply = MessageCodec::read(&mut stream).await.expect("read ack");
    match reply {
        ClusterMessage::ReplicateAck { index, success, .. } => {
            assert_eq!(index, 5);
            assert!(success);
        }
        other => panic!("expected ReplicateAck, got {other:?}"),
    }

    node.shutdown();
}
