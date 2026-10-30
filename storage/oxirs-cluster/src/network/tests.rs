//! Unit tests for the network layer (declared from `network.rs`).

use super::*;
use crate::tls::test_support::{install_provider, node_config};
use std::net::{IpAddr, Ipv4Addr};

#[test]
fn test_network_config_default() {
    let config = NetworkConfig::default();
    assert_eq!(config.connection_timeout, Duration::from_secs(5));
    assert_eq!(config.request_timeout, Duration::from_secs(10));
    assert_eq!(config.max_connections, 100);
    assert_eq!(config.keep_alive_interval, Duration::from_secs(30));
}

#[test]
fn test_log_entry_creation() {
    let command = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };
    let entry = LogEntry::new(1, 1, command.clone());

    assert_eq!(entry.index, 1);
    assert_eq!(entry.term, 1);
    assert_eq!(entry.command, command);
}

#[test]
fn test_connection_creation() {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    let connection = Connection::new(1, addr);

    assert_eq!(connection.peer_id, 1);
    assert_eq!(connection.address, addr);
    assert!(!connection.is_connected);
}

#[test]
fn test_connection_staleness() {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    let connection = Connection::new(1, addr);

    // Fresh connection should not be stale
    assert!(!connection.is_stale(Duration::from_secs(10)));

    // Connection should be stale with very short timeout after some time passes
    std::thread::sleep(Duration::from_millis(1));
    assert!(connection.is_stale(Duration::from_nanos(1)));
}

#[test]
fn test_connection_activity_update() {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    let mut connection = Connection::new(1, addr);

    let old_activity = connection.last_activity;
    std::thread::sleep(Duration::from_millis(1));
    connection.update_activity();

    assert!(connection.last_activity > old_activity);
}

#[tokio::test]
async fn test_network_manager_creation() {
    let config = NetworkConfig::default();
    let manager = NetworkManager::new(1, config);

    assert_eq!(manager.node_id, 1);
    assert!(!*manager.running.read().await);

    let stats = manager.get_stats().await;
    assert_eq!(stats.node_id, 1);
    assert_eq!(stats.total_connections, 0);
    assert_eq!(stats.active_connections, 0);
}

#[tokio::test]
async fn test_rpc_message_serialization() {
    let message = RpcMessage::RequestVote {
        term: 1,
        candidate_id: 1,
        last_log_index: 0,
        last_log_term: 0,
    };

    let serialized = serde_json::to_string(&message).unwrap();
    let deserialized: RpcMessage = serde_json::from_str(&serialized).unwrap();

    match deserialized {
        RpcMessage::RequestVote {
            term,
            candidate_id,
            last_log_index,
            last_log_term,
        } => {
            assert_eq!(term, 1);
            assert_eq!(candidate_id, 1);
            assert_eq!(last_log_index, 0);
            assert_eq!(last_log_term, 0);
        }
        _ => panic!("Unexpected message type"),
    }
}

#[test]
fn test_network_error_display() {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
    let err = NetworkError::ConnectionFailed {
        peer_id: 1,
        address: addr,
        message: "refused".to_string(),
    };
    assert!(err
        .to_string()
        .contains("Connection failed to 1 at 127.0.0.1:8080: refused"));

    let err = NetworkError::Timeout {
        operation: "connect".to_string(),
        duration: Duration::from_secs(5),
    };
    assert!(err
        .to_string()
        .contains("Timeout: connect timed out after 5s"));

    let err = NetworkError::Serialization {
        message: "invalid json".to_string(),
    };
    assert!(err
        .to_string()
        .contains("Serialization error: invalid json"));

    let err = NetworkError::Protocol {
        message: "unknown message".to_string(),
    };
    assert!(err.to_string().contains("Protocol error: unknown message"));
}

// --- handle_message dispatch tests ---

#[tokio::test]
async fn test_handle_request_vote_returns_vote_response() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::RequestVote {
        term: 5,
        candidate_id: 2,
        last_log_index: 10,
        last_log_term: 4,
    };

    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::VoteResponse { term, vote_granted } => {
            assert_eq!(term, 5);
            assert!(!vote_granted, "conservative default should deny vote");
        }
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_append_entries_returns_success() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::AppendEntries {
        term: 3,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: Vec::new(),
        leader_commit: 0,
    };

    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::AppendEntriesResponse { term, success, .. } => {
            assert_eq!(term, 3);
            assert!(success);
        }
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_heartbeat_returns_heartbeat_response() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::Heartbeat {
        term: 7,
        leader_id: 2,
    };
    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::HeartbeatResponse { term } => assert_eq!(term, 7),
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_client_request_returns_success() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let cmd = crate::raft::RdfCommand::Clear;
    let msg = RpcMessage::ClientRequest { command: cmd };
    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::ClientResponse {
            response: crate::raft::RdfResponse::Success,
        } => {}
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_query_shard_returns_empty_results() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::QueryShard {
        shard_id: 42,
        subject: None,
        predicate: None,
        object: None,
    };
    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::QueryShardResponse { shard_id, results } => {
            assert_eq!(shard_id, 42);
            assert!(results.is_empty());
        }
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_transaction_prepare_votes_no() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::TransactionPrepare {
        tx_id: "tx-001".to_string(),
        shard_id: 1,
        operations: Vec::new(),
    };
    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::TransactionVote {
            tx_id,
            shard_id,
            vote,
        } => {
            assert_eq!(tx_id, "tx-001");
            assert_eq!(shard_id, 1);
            assert!(!vote, "conservative default should vote no");
        }
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_transaction_commit_returns_ack() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::TransactionCommit {
        tx_id: "tx-002".to_string(),
        shard_id: 5,
    };
    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::TransactionAck { tx_id, shard_id } => {
            assert_eq!(tx_id, "tx-002");
            assert_eq!(shard_id, 5);
        }
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_transaction_abort_returns_ack() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::TransactionAbort {
        tx_id: "tx-003".to_string(),
        shard_id: 9,
    };
    let resp = svc
        .handle_message(msg)
        .await
        .expect("handle_message failed");
    match resp {
        RpcMessage::TransactionAck { tx_id, shard_id } => {
            assert_eq!(tx_id, "tx-003");
            assert_eq!(shard_id, 9);
        }
        other => panic!("unexpected response: {:?}", other),
    }
}

#[tokio::test]
async fn test_handle_store_triple_returns_descriptive_error() {
    use oxirs_core::model::{Literal, NamedNode, Triple};

    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let triple = Triple::new(
        NamedNode::new("http://example.org/s").expect("valid IRI"),
        NamedNode::new("http://example.org/p").expect("valid IRI"),
        Literal::new("object-value"),
    );
    let msg = RpcMessage::StoreTriple {
        shard_id: 3,
        triple,
    };
    let err = svc
        .handle_message(msg)
        .await
        .expect_err("should return Err");
    let msg_str = err.to_string();
    assert!(
        msg_str.contains("StoreTriple") || msg_str.contains("shard"),
        "error should mention StoreTriple or shard, got: {}",
        msg_str
    );
    assert!(
        !msg_str.contains("not yet implemented"),
        "stub message should be gone, got: {}",
        msg_str
    );
}

#[tokio::test]
async fn test_handle_shard_transfer_returns_descriptive_error() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    let msg = RpcMessage::ShardTransfer {
        shard_id: 7,
        triples: Vec::new(),
        source_node: 2,
    };
    let err = svc
        .handle_message(msg)
        .await
        .expect_err("should return Err");
    let msg_str = err.to_string();
    assert!(
        !msg_str.contains("not yet implemented"),
        "stub message should be gone, got: {}",
        msg_str
    );
}

#[tokio::test]
async fn test_handle_terminal_responses_echo_back() {
    let config = NetworkConfig::default();
    let svc = NetworkService::new(1, config);

    // VoteResponse echoed back
    let msg = RpcMessage::VoteResponse {
        term: 2,
        vote_granted: true,
    };
    let resp = svc.handle_message(msg).await.expect("should succeed");
    match resp {
        RpcMessage::VoteResponse {
            term: 2,
            vote_granted: true,
        } => {}
        other => panic!("unexpected: {:?}", other),
    }

    // HeartbeatResponse echoed back
    let msg = RpcMessage::HeartbeatResponse { term: 8 };
    let resp = svc.handle_message(msg).await.expect("should succeed");
    match resp {
        RpcMessage::HeartbeatResponse { term: 8 } => {}
        other => panic!("unexpected: {:?}", other),
    }
}

// --- length-prefixed framing tests ---

#[tokio::test]
async fn test_frame_round_trip_identity() {
    // encode -> decode must yield an identical RpcMessage, proving real
    // oxicode (not bincode) serialization over the wire framing.
    let original = RpcMessage::AppendEntries {
        term: 42,
        leader_id: 7,
        prev_log_index: 3,
        prev_log_term: 2,
        entries: vec![LogEntry::new(
            4,
            42,
            RdfCommand::Insert {
                subject: "s".to_string(),
                predicate: "p".to_string(),
                object: "o".to_string(),
            },
        )],
        leader_commit: 3,
    };

    let (mut a, mut b) = tokio::io::duplex(64 * 1024);
    let to_send = original.clone();
    let writer = tokio::spawn(async move {
        write_frame(&mut a, &to_send, 16 * 1024 * 1024)
            .await
            .expect("write_frame failed");
    });

    let decoded = read_frame(&mut b, 16 * 1024 * 1024)
        .await
        .expect("read_frame failed");
    writer.await.expect("writer task panicked");

    match decoded {
        RpcMessage::AppendEntries {
            term,
            leader_id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
        } => {
            assert_eq!(term, 42);
            assert_eq!(leader_id, 7);
            assert_eq!(prev_log_index, 3);
            assert_eq!(prev_log_term, 2);
            assert_eq!(leader_commit, 3);
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].index, 4);
            assert_eq!(entries[0].term, 42);
        }
        other => panic!("frame round trip changed the message: {:?}", other),
    }
}

#[tokio::test]
async fn test_frame_rejects_oversize_message() {
    let msg = RpcMessage::Heartbeat {
        term: 1,
        leader_id: 1,
    };
    let (mut a, _b) = tokio::io::duplex(1024);
    // max_size of 1 byte forces the oversize guard to trip.
    let err = write_frame(&mut a, &msg, 1)
        .await
        .expect_err("oversize frame must be rejected");
    assert!(
        err.to_string().contains("exceeds max_message_size"),
        "unexpected error: {}",
        err
    );
}

#[tokio::test]
async fn test_send_rpc_round_trip_returns_peer_response() {
    // Stand up a loopback server that reads the request frame and replies
    // with a distinctive VoteResponse. The old fabricating transport always
    // returned vote_granted:false regardless of the peer; a real transport
    // must surface the peer's actual (vote_granted:true) response.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind loopback listener");
    let addr = listener.local_addr().expect("no local addr");

    let server = tokio::spawn(async move {
        let (mut socket, _peer) = listener.accept().await.expect("accept failed");
        let request = read_frame(&mut socket, 16 * 1024 * 1024)
            .await
            .expect("server read_frame failed");
        let term = match request {
            RpcMessage::RequestVote { term, .. } => term,
            other => panic!("server got unexpected request: {:?}", other),
        };
        let response = RpcMessage::VoteResponse {
            term,
            vote_granted: true,
        };
        write_frame(&mut socket, &response, 16 * 1024 * 1024)
            .await
            .expect("server write_frame failed");
    });

    let manager = NetworkManager::new(1, NetworkConfig::default());
    let (term, granted) = manager
        .send_request_vote(2, addr, 99, 5, 4)
        .await
        .expect("send_request_vote failed");

    assert_eq!(term, 99, "response term must echo the request term");
    assert!(
        granted,
        "must return the peer's actual vote (true), not a fabricated default"
    );

    server.await.expect("server task panicked");

    // Connection metadata + byte counters must reflect a real exchange.
    let stats = manager.get_message_stats().await;
    assert_eq!(stats.messages_sent, 1);
    assert_eq!(stats.messages_received, 1);
    assert!(stats.bytes_sent > 0);
}

#[tokio::test]
async fn test_send_message_errors_without_known_address() {
    // With no registered address and no live connection, delivery must fail
    // loudly rather than silently reporting success.
    let svc = NetworkService::new(1, NetworkConfig::default());
    let err = svc
        .send_message(
            99,
            RpcMessage::Heartbeat {
                term: 1,
                leader_id: 1,
            },
        )
        .await
        .expect_err("send_message must error when the peer address is unknown");
    assert!(
        err.to_string().contains("no known network address"),
        "unexpected error: {}",
        err
    );
}

#[tokio::test]
async fn test_send_message_delivers_to_registered_peer() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind loopback listener");
    let addr = listener.local_addr().expect("no local addr");

    let server = tokio::spawn(async move {
        let (mut socket, _peer) = listener.accept().await.expect("accept failed");
        let request = read_frame(&mut socket, 16 * 1024 * 1024)
            .await
            .expect("server read_frame failed");
        let term = match request {
            RpcMessage::Heartbeat { term, .. } => term,
            other => panic!("server got unexpected request: {:?}", other),
        };
        write_frame(
            &mut socket,
            &RpcMessage::HeartbeatResponse { term },
            16 * 1024 * 1024,
        )
        .await
        .expect("server write_frame failed");
    });

    let svc = NetworkService::new(1, NetworkConfig::default());
    svc.register_peer(2, addr).await;
    svc.send_message(
        2,
        RpcMessage::Heartbeat {
            term: 11,
            leader_id: 1,
        },
    )
    .await
    .expect("send_message to a registered, reachable peer must succeed");

    server.await.expect("server task panicked");
}

/// Regression: a *started* NetworkManager must actually accept inbound
/// connections and answer a `Heartbeat` with a `HeartbeatResponse`.
/// Previously `start_background_tasks` skipped spawning any accept loop
/// ("for now, we'll skip the listener task"), so the bound listener never
/// answered and health probes reported every peer unreachable even for a
/// healthy cluster.
#[tokio::test]
async fn regression_network_accept_loop_answers_heartbeat() {
    // Reserve a free port, then release it so the manager can bind it.
    let probe = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to reserve a port");
    let addr = probe.local_addr().expect("no local addr");
    drop(probe);

    let mut cfg = NetworkConfig::default();
    cfg.local_address = addr;

    let mut server = NetworkManager::new(1, cfg.clone());
    server.start().await.expect("server start failed");

    // A separate manager acts as the client and issues a real heartbeat RPC.
    let client = NetworkManager::new(2, cfg);
    let response = client
        .send_rpc(
            1,
            addr,
            RpcMessage::Heartbeat {
                term: 5,
                leader_id: 2,
            },
        )
        .await
        .expect("heartbeat RPC to a started server must get a real answer");

    match response {
        RpcMessage::HeartbeatResponse { term } => assert_eq!(term, 5),
        other => panic!("expected HeartbeatResponse, got {other:?}"),
    }

    server.stop().await.expect("server stop failed");
}

/// Reserve a free loopback port and release it so a manager can bind it.
async fn free_loopback_addr() -> SocketAddr {
    let probe = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to reserve a port");
    probe.local_addr().expect("no local addr")
}

/// With TLS configured, nodes talk only over mutually authenticated TLS:
/// a trusted peer gets a real answer, a plaintext client gets nothing.
/// (The TLS listener used to be plain TCP, so this never worked.)
#[tokio::test]
async fn test_tls_nodes_exchange_rpc_over_mutual_tls() {
    install_provider();
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let bundle = dir.path().join("cluster.pem");
    let (tls_1, pem_1) = node_config(dir.path(), 1, &bundle, true).await;
    let (tls_2, pem_2) = node_config(dir.path(), 2, &bundle, true).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).expect("failed to write bundle");

    let addr = free_loopback_addr().await;
    let mut server = NetworkManager::with_tls(
        1,
        NetworkConfig {
            local_address: addr,
            tls_config: tls_1,
            ..Default::default()
        },
    )
    .await
    .expect("server TLS initialization failed");
    server.start().await.expect("server start failed");
    let client = NetworkManager::with_tls(
        2,
        NetworkConfig {
            tls_config: tls_2,
            ..Default::default()
        },
    )
    .await
    .expect("client TLS initialization failed");

    let response = client
        .send_secure_rpc(
            1,
            addr,
            RpcMessage::Heartbeat {
                term: 9,
                leader_id: 2,
            },
        )
        .await
        .expect("heartbeat over mutual TLS failed");
    match response {
        RpcMessage::HeartbeatResponse { term } => assert_eq!(term, 9),
        other => panic!("expected HeartbeatResponse, got {other:?}"),
    }

    let plaintext = NetworkManager::new(3, NetworkConfig::default());
    let result = plaintext
        .send_rpc(
            1,
            addr,
            RpcMessage::Heartbeat {
                term: 9,
                leader_id: 3,
            },
        )
        .await;
    assert!(
        result.is_err(),
        "a TLS-only node answered a plaintext request"
    );

    server.stop().await.expect("server stop failed");
}

/// A node refuses to exchange RPCs with a peer it does not trust, even one
/// that claims the right node id.
#[tokio::test]
async fn test_tls_client_refuses_untrusted_node() {
    install_provider();
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let trusted = dir.path().join("trusted.pem");
    let rogue_bundle = dir.path().join("rogue.pem");
    let (tls_client, client_pem) = node_config(dir.path(), 2, &trusted, false).await;
    let (tls_rogue, rogue_pem) = node_config(dir.path(), 1, &rogue_bundle, false).await;
    std::fs::write(&trusted, client_pem).expect("failed to write bundle");
    std::fs::write(&rogue_bundle, rogue_pem).expect("failed to write bundle");

    let addr = free_loopback_addr().await;
    let mut rogue = NetworkManager::with_tls(
        1,
        NetworkConfig {
            local_address: addr,
            tls_config: tls_rogue,
            ..Default::default()
        },
    )
    .await
    .expect("rogue TLS initialization failed");
    rogue.start().await.expect("rogue start failed");
    let client = NetworkManager::with_tls(
        2,
        NetworkConfig {
            tls_config: tls_client,
            ..Default::default()
        },
    )
    .await
    .expect("client TLS initialization failed");

    let result = client
        .send_secure_rpc(
            1,
            addr,
            RpcMessage::Heartbeat {
                term: 1,
                leader_id: 2,
            },
        )
        .await;
    assert!(result.is_err(), "client talked to a node it does not trust");
    assert_eq!(client.get_message_stats().await.tls_handshakes_failed, 1);

    rogue.stop().await.expect("rogue stop failed");
}

/// `send_secure_rpc` used to fall back to plaintext without TLS.
#[tokio::test]
async fn test_send_secure_rpc_requires_tls() {
    let addr = free_loopback_addr().await;
    let mut server = NetworkManager::new(
        1,
        NetworkConfig {
            local_address: addr,
            ..Default::default()
        },
    );
    server.start().await.expect("server start failed");

    let client = NetworkManager::new(2, NetworkConfig::default());
    let result = client
        .send_secure_rpc(
            1,
            addr,
            RpcMessage::Heartbeat {
                term: 1,
                leader_id: 2,
            },
        )
        .await;
    assert!(result.is_err(), "secure RPC fell back to plaintext");

    server.stop().await.expect("server stop failed");
}

/// `encrypt_data` used to prepend "ENCRYPTED:" to the plaintext, or return it
/// unchanged without TLS.
#[tokio::test]
async fn test_encrypt_data_seals_with_the_data_key() {
    let manager = NetworkManager::new(1, NetworkConfig::default());
    assert!(
        manager.encrypt_data(b"secret").await.is_err(),
        "encrypted without a key"
    );
    assert!(manager.decrypt_data(b"secret").await.is_err());

    manager
        .set_data_encryption_key(AtRestKey::new("data", [4u8; 32]).expect("valid key id"))
        .await;
    let sealed = manager
        .encrypt_data(b"secret payload")
        .await
        .expect("encrypt_data");
    assert!(!sealed.windows(6).any(|w| w == b"secret"));
    assert_eq!(
        manager.decrypt_data(&sealed).await.expect("decrypt_data"),
        b"secret payload"
    );

    // Clones share the key.
    let clone = manager.clone();
    assert_eq!(
        clone
            .decrypt_data(&sealed)
            .await
            .expect("decrypt via clone"),
        b"secret payload"
    );

    // Tampered data, a foreign key and the old fake format are all rejected.
    let mut tampered = sealed.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    assert!(manager.decrypt_data(&tampered).await.is_err());
    let other = NetworkManager::new(2, NetworkConfig::default());
    other
        .set_data_encryption_key(AtRestKey::new("data", [5u8; 32]).expect("valid key id"))
        .await;
    assert!(other.decrypt_data(&sealed).await.is_err());
    assert!(manager.decrypt_data(b"ENCRYPTED:secret").await.is_err());
}

fn heartbeat(leader_id: OxirsNodeId) -> RpcMessage {
    RpcMessage::Heartbeat { term: 4, leader_id }
}

/// `NetworkManager::new` used to ignore `tls_config.enabled`: a manager
/// configured for TLS listened and dialled in plaintext.
#[tokio::test]
async fn test_new_manager_honors_the_tls_flag() {
    install_provider();
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let bundle = dir.path().join("cluster.pem");
    let (tls_1, pem_1) = node_config(dir.path(), 1, &bundle, true).await;
    let (tls_2, pem_2) = node_config(dir.path(), 2, &bundle, true).await;
    std::fs::write(&bundle, format!("{pem_1}{pem_2}")).expect("failed to write bundle");

    let addr = free_loopback_addr().await;
    let mut server = NetworkManager::new(
        1,
        NetworkConfig {
            local_address: addr,
            tls_config: tls_1,
            ..Default::default()
        },
    );
    server.start().await.expect("server start failed");

    let mut plaintext_config = NetworkConfig::default();
    plaintext_config.tls_config.enabled = false;
    let plaintext = NetworkManager::new(3, plaintext_config);
    assert!(
        plaintext.send_rpc(1, addr, heartbeat(3)).await.is_err(),
        "a node configured for TLS answered a plaintext request"
    );

    // A TLS client built with `new` dials over TLS without being started.
    let client = NetworkManager::new(
        2,
        NetworkConfig {
            tls_config: tls_2,
            ..Default::default()
        },
    );
    match client.send_rpc(1, addr, heartbeat(2)).await {
        Ok(RpcMessage::HeartbeatResponse { term }) => assert_eq!(term, 4),
        other => panic!("expected a HeartbeatResponse over TLS, got {other:?}"),
    }
    assert_eq!(client.get_message_stats().await.tls_handshakes_completed, 1);

    server.stop().await.expect("server stop failed");
}

/// With TLS enabled but impossible to initialize, nothing may fall back to
/// plaintext: `start()` fails and so does every outgoing RPC.
#[tokio::test]
async fn test_enabled_tls_that_cannot_initialize_fails_closed() {
    let dir = tempfile::TempDir::new().expect("failed to create temp dir");
    let broken_tls = TlsConfig {
        enabled: true,
        cert_dir: dir.path().to_path_buf(),
        ca_file: None,
        auto_rotate: false,
        ..Default::default()
    };

    let mut listener = NetworkManager::new(
        1,
        NetworkConfig {
            local_address: free_loopback_addr().await,
            tls_config: broken_tls.clone(),
            ..Default::default()
        },
    );
    assert!(
        listener.start().await.is_err(),
        "started without the TLS it was configured for"
    );

    let plaintext_addr = free_loopback_addr().await;
    let mut plaintext_config = NetworkConfig {
        local_address: plaintext_addr,
        ..Default::default()
    };
    plaintext_config.tls_config.enabled = false;
    let mut plaintext_server = NetworkManager::new(2, plaintext_config);
    plaintext_server.start().await.expect("server start failed");

    let client = NetworkManager::new(
        3,
        NetworkConfig {
            tls_config: broken_tls,
            ..Default::default()
        },
    );
    assert!(
        client
            .send_rpc(2, plaintext_addr, heartbeat(3))
            .await
            .is_err(),
        "an RPC fell back to plaintext"
    );

    plaintext_server.stop().await.expect("server stop failed");
}
