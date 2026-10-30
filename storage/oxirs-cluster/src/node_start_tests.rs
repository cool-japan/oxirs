//! Tests for starting [`crate::ClusterNode`]: retries after a failure, and
//! the TLS configured through [`crate::NodeConfig::tls_config`].

use super::*;
use crate::tls::test_support::{install_provider, node_config};
use std::net::{IpAddr, Ipv4Addr};

fn free_addr() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local_addr")
}

fn two_node_config(dir: &std::path::Path, tls_config: TlsConfig) -> NodeConfig {
    let mut config = NodeConfig::new(1, free_addr()).with_tls(tls_config);
    config.data_dir = dir.join("node-1").display().to_string();
    config.discovery = None;
    config.add_peer(2);
    config.add_peer_address(2, free_addr());
    config
}

/// A start that failed left the node marked as running, so the next
/// `start()` reported success without starting anything.
#[tokio::test]
async fn test_failed_start_can_be_retried() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let mut config = NodeConfig::new(1, SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0));
    config.data_dir = dir.path().join("node-1").display().to_string();
    config.discovery = None;
    // Peer 2 has no address, so consensus cannot start.
    config.add_peer(2);
    let mut node = ClusterNode::new(config).await.expect("node");

    assert!(node.start().await.is_err());
    assert!(
        node.start().await.is_err(),
        "a failed start left the node marked as running"
    );
}

/// `NodeConfig::tls_config` reaches the Raft transport: TLS that cannot
/// be initialized keeps the node from starting, also on a retry.
#[tokio::test]
async fn test_node_with_broken_tls_does_not_start() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let broken = TlsConfig {
        enabled: true,
        cert_dir: dir.path().join("certs"),
        ca_file: None,
        auto_rotate: false,
        ..Default::default()
    };
    let mut node = ClusterNode::new(two_node_config(dir.path(), broken))
        .await
        .expect("node");
    for attempt in 1..=2 {
        let error = node
            .start()
            .await
            .expect_err("the node started without the TLS it was configured for");
        assert!(
            error.to_string().contains("TLS"),
            "attempt {attempt}: {error}"
        );
    }
}

/// Three nodes configured through `NodeConfig::tls_config` form a cluster
/// and replicate a write.
#[tokio::test]
async fn test_cluster_nodes_replicate_over_tls() {
    install_provider();
    let dir = tempfile::TempDir::new().expect("temp dir");
    let bundle = dir.path().join("cluster.pem");
    let addrs: HashMap<OxirsNodeId, SocketAddr> =
        [1, 2, 3].into_iter().map(|id| (id, free_addr())).collect();

    let mut pems = String::new();
    let mut configs = Vec::new();
    for id in [1, 2, 3] {
        let cert_dir = dir.path().join(format!("certs-{id}"));
        std::fs::create_dir_all(&cert_dir).expect("cert dir");
        let (tls_config, pem) = node_config(&cert_dir, id, &bundle, true).await;
        pems.push_str(&pem);
        let mut config = NodeConfig::new(id, addrs[&id]).with_tls(tls_config);
        config.data_dir = dir.path().join(format!("node-{id}")).display().to_string();
        config.discovery = None;
        for (&peer, &peer_addr) in addrs.iter().filter(|(&peer, _)| peer != id) {
            config.add_peer(peer);
            config.add_peer_address(peer, peer_addr);
        }
        configs.push(config);
    }
    std::fs::write(&bundle, pems).expect("bundle");

    let mut nodes = Vec::new();
    for config in configs {
        let mut node = ClusterNode::new(config).await.expect("node");
        node.start().await.expect("start over TLS");
        nodes.push(node);
    }

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    let leader = loop {
        let mut found = None;
        for (idx, node) in nodes.iter().enumerate() {
            if node.is_leader().await {
                found = Some(idx);
            }
        }
        if let Some(idx) = found {
            break idx;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "no leader was elected"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    nodes[leader]
        .insert_triple("<http://example.org/s>", "<http://example.org/p>", "\"o\"")
        .await
        .expect("insert through consensus over TLS");
    for (idx, node) in nodes.iter().enumerate() {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        while node.len().await != 1 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(node.len().await, 1, "node index {idx} did not replicate");
    }
    for (id, addr) in &addrs {
        assert!(
            !crate::raft_network::plaintext_vote_is_answered(*addr).await,
            "node {id} answered a plaintext Raft request"
        );
    }
    for node in &mut nodes {
        node.stop().await.expect("stop");
    }
}
