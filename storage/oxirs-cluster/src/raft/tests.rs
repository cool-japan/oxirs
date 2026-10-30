//! Tests for [`super`]: the RDF state machine and Raft consensus.

use super::*;

/// Triples of every kind of term, as a shard holds them.
fn shard_triples() -> Vec<oxirs_core::model::Triple> {
    use oxirs_core::model::{BlankNode, Literal, NamedNode, Triple};
    let iri = |value: &str| NamedNode::new(value).expect("IRI");
    vec![
        Triple::new(
            iri("http://example.org/s"),
            iri("http://example.org/p"),
            iri("http://example.org/o"),
        ),
        Triple::new(
            BlankNode::new("b1").expect("blank node"),
            iri("http://example.org/p"),
            Literal::new_simple_literal("a \"quoted\"\nvalue"),
        ),
        Triple::new(
            iri("http://example.org/s"),
            iri("http://example.org/label"),
            Literal::new_language_tagged_literal("hallo", "de").expect("language tag"),
        ),
        Triple::new(
            iri("http://example.org/s"),
            iri("http://example.org/count"),
            Literal::new_typed_literal("42", iri("http://www.w3.org/2001/XMLSchema#integer")),
        ),
    ]
}

/// Shards stored triples in N-Triples term syntax (`<iri>`) but read them
/// back as bare IRIs, so no triple written to a shard could be read,
/// queried or exported again.
#[test]
fn test_shard_triples_read_back_as_written() {
    let mut app = RdfApp::default();
    for triple in shard_triples() {
        app.insert_triple_to_shard(3, triple);
    }

    let mut read_back = app.get_shard_triples(3);
    let mut expected = shard_triples();
    read_back.sort_by_key(|t| t.to_string());
    expected.sort_by_key(|t| t.to_string());
    assert_eq!(read_back, expected);

    // Filters take IRIs without brackets, like the other backends.
    let by_subject = app.query_shard(3, Some("http://example.org/s"), None, None);
    assert_eq!(by_subject.len(), 3);
    let by_object = app.query_shard(3, None, None, Some("http://example.org/o"));
    assert_eq!(by_object.len(), 1);
    let by_predicate = app.query_shard(3, None, Some("http://example.org/label"), None);
    assert_eq!(by_predicate.len(), 1);
    assert_eq!(app.export_shard(3).len(), 4);
}

#[test]
fn test_rdf_command_serialization() {
    let cmd = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };

    let serialized = serde_json::to_string(&cmd).unwrap();
    let deserialized: RdfCommand = serde_json::from_str(&serialized).unwrap();

    assert_eq!(cmd, deserialized);
}

#[test]
fn test_rdf_response_serialization() {
    let response = RdfResponse::Success;

    let serialized = serde_json::to_string(&response).unwrap();
    let deserialized: RdfResponse = serde_json::from_str(&serialized).unwrap();

    assert_eq!(response, deserialized);
}

#[test]
fn test_rdf_app_apply_insert() {
    let mut app = RdfApp::default();

    let cmd = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };

    let response = app.apply_command(&cmd);
    assert_eq!(response, RdfResponse::Success);
    assert_eq!(app.len(), 1);
    assert!(!app.is_empty());
}

#[test]
fn test_rdf_app_apply_delete() {
    let mut app = RdfApp::default();

    // Insert first
    let insert_cmd = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };
    app.apply_command(&insert_cmd);
    assert_eq!(app.len(), 1);

    // Then delete
    let delete_cmd = RdfCommand::Delete {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };
    let response = app.apply_command(&delete_cmd);
    assert_eq!(response, RdfResponse::Success);
    assert_eq!(app.len(), 0);
    assert!(app.is_empty());
}

#[test]
fn test_rdf_app_apply_clear() {
    let mut app = RdfApp::default();

    // Insert some data
    app.apply_command(&RdfCommand::Insert {
        subject: "s1".to_string(),
        predicate: "p1".to_string(),
        object: "o1".to_string(),
    });
    app.apply_command(&RdfCommand::Insert {
        subject: "s2".to_string(),
        predicate: "p2".to_string(),
        object: "o2".to_string(),
    });
    assert_eq!(app.len(), 2);

    // Clear all
    let response = app.apply_command(&RdfCommand::Clear);
    assert_eq!(response, RdfResponse::Success);
    assert_eq!(app.len(), 0);
    assert!(app.is_empty());
}

#[test]
fn test_rdf_app_transactions() {
    let mut app = RdfApp::default();
    let tx_id = "tx1".to_string();

    // Begin transaction
    let response = app.apply_command(&RdfCommand::BeginTransaction {
        tx_id: tx_id.clone(),
    });
    assert_eq!(
        response,
        RdfResponse::TransactionStarted {
            tx_id: tx_id.clone()
        }
    );
    assert!(app.transactions.contains_key(&tx_id));

    // Commit transaction
    let response = app.apply_command(&RdfCommand::CommitTransaction {
        tx_id: tx_id.clone(),
    });
    assert_eq!(
        response,
        RdfResponse::TransactionCommitted {
            tx_id: tx_id.clone()
        }
    );
    assert!(!app.transactions.contains_key(&tx_id));
}

#[test]
fn test_rdf_app_transaction_rollback() {
    let mut app = RdfApp::default();
    let tx_id = "tx1".to_string();

    // Begin transaction
    app.apply_command(&RdfCommand::BeginTransaction {
        tx_id: tx_id.clone(),
    });
    assert!(app.transactions.contains_key(&tx_id));

    // Rollback transaction
    let response = app.apply_command(&RdfCommand::RollbackTransaction {
        tx_id: tx_id.clone(),
    });
    assert_eq!(
        response,
        RdfResponse::TransactionRolledBack {
            tx_id: tx_id.clone()
        }
    );
    assert!(!app.transactions.contains_key(&tx_id));
}

#[test]
fn test_rdf_app_query() {
    let mut app = RdfApp::default();

    // Insert test data
    app.apply_command(&RdfCommand::Insert {
        subject: "s1".to_string(),
        predicate: "p1".to_string(),
        object: "o1".to_string(),
    });
    app.apply_command(&RdfCommand::Insert {
        subject: "s1".to_string(),
        predicate: "p2".to_string(),
        object: "o2".to_string(),
    });
    app.apply_command(&RdfCommand::Insert {
        subject: "s2".to_string(),
        predicate: "p1".to_string(),
        object: "o3".to_string(),
    });

    // Query all triples
    let results = app.query(None, None, None);
    assert_eq!(results.len(), 3);

    // Query by subject
    let results = app.query(Some("s1"), None, None);
    assert_eq!(results.len(), 2);

    // Query by predicate
    let results = app.query(None, Some("p1"), None);
    assert_eq!(results.len(), 2);

    // Query by object
    let results = app.query(None, None, Some("o1"));
    assert_eq!(results.len(), 1);

    // Query specific triple
    let results = app.query(Some("s1"), Some("p1"), Some("o1"));
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0],
        ("s1".to_string(), "p1".to_string(), "o1".to_string())
    );
}

#[tokio::test]
async fn test_raft_node_creation() {
    let node = RaftNode::new(1);
    assert_eq!(node.node_id, 1);
    // Uninitialized nodes act as leaders in single-node mode
    assert!(node.is_leader().await);
    assert_eq!(node.current_term().await, 0);
    assert_eq!(node.len().await, 0);
    assert!(node.is_empty().await);
}

#[tokio::test]
async fn test_raft_node_local_operations() {
    let node = RaftNode::new(1);

    // Test insert command without Raft
    let cmd = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };

    let response = node.submit_command(cmd).await.unwrap();
    assert_eq!(response, RdfResponse::Success);
    assert_eq!(node.len().await, 1);
    assert!(!node.is_empty().await);

    // Test query
    let results = node.query(Some("s"), Some("p"), Some("o")).await;
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0],
        ("s".to_string(), "p".to_string(), "o".to_string())
    );
}

/// `init_raft` must fail loudly (not warn+Ok, and not silently fall back
/// to fake single-node "leadership") when a real multi-node cluster is
/// requested without first configuring the network via `set_network` —
/// there is no bind address to build a `RaftNetworkFactory`/listener
/// with. Real multi-node consensus *is* implemented (see
/// `test_multi_node_raft_elects_leader_and_replicates` below); this
/// covers the still-real precondition failure when setup is incomplete.
#[cfg(feature = "raft")]
#[tokio::test]
async fn test_init_raft_multi_node_without_network_config_fails_loudly() {
    let mut node = RaftNode::new(1);
    let peers: BTreeSet<OxirsNodeId> = [1u64, 2, 3].into_iter().collect();

    let result = node.init_raft(peers).await;
    assert!(
        result.is_err(),
        "requesting a multi-node cluster with no network config must fail, not silently succeed"
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("multi-node") || err.to_string().contains("network configuration"),
        "error message should clearly explain the missing network configuration: {err}"
    );
}

/// After a multi-node `init_raft()` fails (e.g. missing network config),
/// the node must never claim leadership or accept writes that pretend to
/// be replicated.
#[cfg(feature = "raft")]
#[tokio::test]
async fn test_node_after_failed_multi_node_init_is_not_a_fake_leader() {
    let mut node = RaftNode::new(42);
    let peers: BTreeSet<OxirsNodeId> = [42u64, 7, 9].into_iter().collect();

    let init_result = node.init_raft(peers).await;
    assert!(init_result.is_err());

    // The node must not silently masquerade as a single-node leader.
    assert!(
        !node.is_leader().await,
        "node must not claim leadership after multi-node Raft init failed"
    );

    // Writes must fail loudly instead of being applied to unreplicated
    // local-only fallback storage.
    let cmd = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };
    let submit_result = node.submit_command(cmd).await;
    assert!(
        submit_result.is_err(),
        "submit_command must fail rather than fabricate a successful unreplicated write"
    );
}

/// Bind three real `RaftNode`s to loopback TCP ports, wire them into one
/// cluster via `set_network` + `init_raft`, and verify actual OpenRaft
/// consensus: a leader is elected, a command submitted through it is
/// really replicated (not just applied to a global test fallback) to
/// every node's own independently-tracked applied state, and each
/// node's `is_leader()`/`current_term()` reflect genuine per-node raft
/// metrics.
#[cfg(feature = "raft")]
#[tokio::test]
async fn test_multi_node_raft_elects_leader_and_replicates() {
    use std::net::TcpListener as StdTcpListener;

    // Reserve 3 free loopback ports synchronously (bind then immediately
    // drop) so every node's address is known before any node starts —
    // real multi-node `init_raft` requires the full peer address map
    // upfront, mirroring how `TestCluster` in the integration tests
    // derives its addresses.
    let free_addr = || {
        let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        listener.local_addr().expect("local_addr")
    };
    let addrs: BTreeMap<OxirsNodeId, std::net::SocketAddr> = [
        (1u64, free_addr()),
        (2u64, free_addr()),
        (3u64, free_addr()),
    ]
    .into_iter()
    .collect();

    let mut nodes: Vec<RaftNode> = Vec::new();
    for (&id, &addr) in &addrs {
        let mut node = RaftNode::new(id);
        let peers: HashMap<OxirsNodeId, SocketAddr> = addrs
            .iter()
            .filter(|(&peer_id, _)| peer_id != id)
            .map(|(&peer_id, &peer_addr)| (peer_id, peer_addr))
            .collect();
        node.set_network(addr, peers);
        nodes.push(node);
    }

    let all_ids: BTreeSet<OxirsNodeId> = addrs.keys().copied().collect();
    for node in &mut nodes {
        node.init_raft(all_ids.clone())
            .await
            .expect("multi-node init_raft with full network config must succeed");
    }

    // Poll for a leader to emerge (real election takes a nonzero amount
    // of time after initialize()).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut leader_idx = None;
    while tokio::time::Instant::now() < deadline {
        for (idx, node) in nodes.iter().enumerate() {
            if node.is_leader().await {
                leader_idx = Some(idx);
                break;
            }
        }
        if leader_idx.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let leader_idx = leader_idx.expect("a leader must be elected within 5s");
    assert!(
        nodes[leader_idx].current_term().await >= 1,
        "an elected leader must have a real (non-zero) term"
    );

    // Submit a command through the real leader and verify it lands on
    // every node's own applied state (not a shared global fallback).
    let cmd = RdfCommand::Insert {
        subject: "http://example.org/s".to_string(),
        predicate: "http://example.org/p".to_string(),
        object: "\"o\"".to_string(),
    };
    let response = nodes[leader_idx]
        .submit_command(cmd)
        .await
        .expect("leader submit_command must succeed");
    assert_eq!(response, RdfResponse::Success);

    for (idx, node) in nodes.iter().enumerate() {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut count = node.len().await;
        while count != 1 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
            count = node.len().await;
        }
        assert_eq!(
            count, 1,
            "node index {idx} did not replicate the committed entry"
        );
    }

    for node in &mut nodes {
        node.shutdown().await.expect("shutdown must succeed");
    }
}

/// A genuine single-node deployment (empty peer set, or a peer set
/// containing only self) must continue to work as an honest one-node
/// "cluster": init succeeds and the node is its own leader.
#[cfg(feature = "raft")]
#[tokio::test]
async fn test_init_raft_single_node_mode_stays_leader() {
    let mut node = RaftNode::new(5);

    // Explicitly listing only self is equivalent to no peers.
    let peers: BTreeSet<OxirsNodeId> = [5u64].into_iter().collect();
    let result = node.init_raft(peers).await;
    assert!(result.is_ok(), "single-node init must succeed: {result:?}");
    assert!(node.is_leader().await);

    let cmd = RdfCommand::Insert {
        subject: "s".to_string(),
        predicate: "p".to_string(),
        object: "o".to_string(),
    };
    let response = node.submit_command(cmd).await;
    assert!(
        response.is_ok(),
        "single-node writes must still succeed: {response:?}"
    );
}

/// A successful `initialize()` (or one that failed only because a peer
/// / an earlier attempt already initialized the cluster) is the only
/// case where re-`initialize()`-ing this node in the future would be
/// unsafe: `bootstrap_attempted` must be latched, and the retry loop
/// must stop.
#[cfg(feature = "raft")]
#[test]
fn test_classify_bootstrap_attempt_success_and_already_initialized_are_done_and_latch() {
    let succeeded = classify_bootstrap_attempt(true, false, false);
    assert_eq!(succeeded, BootstrapDecision::Done);
    assert!(succeeded.should_latch_bootstrap_attempted());

    // Whether attempts remain must not matter once we succeeded.
    let succeeded_last_attempt = classify_bootstrap_attempt(true, false, true);
    assert_eq!(succeeded_last_attempt, BootstrapDecision::Done);
    assert!(succeeded_last_attempt.should_latch_bootstrap_attempted());

    let already_initialized = classify_bootstrap_attempt(false, true, false);
    assert_eq!(already_initialized, BootstrapDecision::Done);
    assert!(already_initialized.should_latch_bootstrap_attempted());
}

/// This is the regression case for the bootstrap-latch bug: a genuine
/// failure (not "already initialized") must be retried while attempts
/// remain, and once every attempt is exhausted, must be reported as
/// `Fatal` *without* latching `bootstrap_attempted` — latching here is
/// exactly what permanently stranded a `RaftNode` on a transient
/// failure before this fix.
#[cfg(feature = "raft")]
#[test]
fn test_classify_bootstrap_attempt_genuine_failure_after_retries_does_not_latch() {
    let mid_retry = classify_bootstrap_attempt(false, false, false);
    assert_eq!(mid_retry, BootstrapDecision::Retryable);
    assert!(!mid_retry.should_latch_bootstrap_attempted());

    let exhausted = classify_bootstrap_attempt(false, false, true);
    assert_eq!(exhausted, BootstrapDecision::Fatal);
    assert!(
        !exhausted.should_latch_bootstrap_attempted(),
        "a genuine failure must leave bootstrap_attempted unlatched so a future \
         init_raft() call can retry"
    );
}

/// The durable state-machine checkpoint is now rewritten only periodically
/// (every `STATE_MACHINE_CHECKPOINT_INTERVAL` applies) rather than on every
/// apply, to avoid an O(n^2) full-state re-serialization on the hot commit
/// path. This optimization must not lose any committed data: the fsync'd
/// durable log stays complete, and the persisted checkpoint plus a replay
/// of the log tail past its `last_applied` must reconstruct the exact state
/// — which is precisely what a restarted node (openraft) does. Applying one
/// entry at a time and choosing a non-multiple of the interval forces the
/// final applies to land *after* the last checkpoint, so this genuinely
/// exercises the checkpoint-lag-then-replay recovery path.
#[cfg(feature = "raft")]
#[tokio::test]
async fn regression_periodic_checkpoint_preserves_all_committed_state() {
    use openraft::{CommittedLeaderId, Entry, EntryPayload, LogId, RaftStorage};

    let dir = std::env::temp_dir().join(format!(
        "oxirs_sm_ckpt_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let n = STATE_MACHINE_CHECKPOINT_INTERVAL * 2 + 3;
    let entries: Vec<Entry<OxirsTypeConfig>> = (1..=n)
        .map(|i| Entry {
            log_id: LogId::new(CommittedLeaderId::new(1, 0), i),
            payload: EntryPayload::Normal(RdfCommand::Insert {
                subject: format!("http://s/{i}"),
                predicate: "http://p".to_string(),
                object: format!("\"o{i}\""),
            }),
        })
        .collect();

    {
        let mut store = OxirsStorage::new_with_dir(&dir).expect("open durable store");
        store
            .append_to_log(entries.clone())
            .await
            .expect("append to durable log");
        // Apply one entry at a time, exactly as openraft applies committed
        // single-write entries.
        for entry in &entries {
            store
                .apply_to_state_machine(std::slice::from_ref(entry))
                .await
                .expect("apply to state machine");
        }
        assert_eq!(
            store.state.read().await.triples.len(),
            n as usize,
            "in-memory state must reflect every applied entry"
        );
    }

    // Re-open on the same directory (simulating a process restart).
    let reopened = OxirsStorage::new_with_dir(&dir).expect("re-open durable store");

    // The durable log must retain every committed entry regardless of
    // checkpoint cadence — that is what makes the periodic checkpoint safe.
    assert_eq!(
        reopened.log.read().await.len(),
        n as usize,
        "durable log must retain every committed entry"
    );

    // The persisted checkpoint is expected to lag (the last few applies
    // landed after the final periodic checkpoint), and the durable log
    // tail past it must reconstruct the full state — exactly the recovery
    // openraft performs on startup.
    let checkpoint_index = reopened
        .last_applied
        .read()
        .await
        .map(|id| id.index)
        .unwrap_or(0);
    assert!(
        checkpoint_index < n,
        "checkpoint should lag the latest apply (got {checkpoint_index}, applied {n})"
    );
    let mut rebuilt = reopened.state.read().await.clone();
    for entry in reopened.log.read().await.iter() {
        if entry.log_id.index > checkpoint_index {
            if let EntryPayload::Normal(cmd) = &entry.payload {
                rebuilt.apply_command(cmd);
            }
        }
    }
    assert_eq!(
        rebuilt.triples.len(),
        n as usize,
        "checkpoint + durable-log replay must reconstruct the full state machine"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Raft RPC over mutually authenticated TLS
// ---------------------------------------------------------------------------

#[cfg(feature = "raft")]
mod tls {
    use super::*;
    use crate::tls::test_support::{install_provider, node_config};
    use crate::tls::{TlsConfig, TlsManager};
    use std::path::Path;

    fn free_addr() -> SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        listener.local_addr().expect("local_addr")
    }

    /// One TLS identity per node, each in its own directory under `dir`, all
    /// trusting one bundle of pinned certificates. The certificates of
    /// `untrusted` nodes are left out of the bundle.
    async fn identities(
        dir: &Path,
        ids: &[OxirsNodeId],
        untrusted: &[OxirsNodeId],
    ) -> HashMap<OxirsNodeId, Arc<TlsManager>> {
        install_provider();
        let bundle = dir.join("cluster.pem");
        let mut pems = String::new();
        let mut configs = Vec::new();
        for &id in ids {
            let node_dir = dir.join(format!("node-{id}"));
            std::fs::create_dir_all(&node_dir).expect("create cert dir");
            let (config, pem) = node_config(&node_dir, id, &bundle, true).await;
            if !untrusted.contains(&id) {
                pems.push_str(&pem);
            }
            configs.push((id, config));
        }
        std::fs::write(&bundle, pems).expect("write bundle");
        configs
            .into_iter()
            .map(|(id, config)| (id, Arc::new(TlsManager::new(config, id))))
            .collect()
    }

    /// Raft nodes for `addrs`, each wired to every other one, with TLS from
    /// `tls` where present.
    async fn start_nodes(
        addrs: &BTreeMap<OxirsNodeId, SocketAddr>,
        tls: &HashMap<OxirsNodeId, Arc<TlsManager>>,
    ) -> Vec<RaftNode> {
        let ids: BTreeSet<OxirsNodeId> = addrs.keys().copied().collect();
        let mut nodes = Vec::new();
        for (&id, &addr) in addrs {
            let mut node = RaftNode::new(id);
            let peers = addrs
                .iter()
                .filter(|(&peer, _)| peer != id)
                .map(|(&peer, &peer_addr)| (peer, peer_addr))
                .collect();
            node.set_network(addr, peers);
            if let Some(manager) = tls.get(&id) {
                node.set_tls(Arc::clone(manager));
            }
            nodes.push(node);
        }
        for node in &mut nodes {
            node.init_raft(ids.clone()).await.expect("init_raft");
        }
        nodes
    }

    async fn wait_for_leader(nodes: &[RaftNode], among: &[usize]) -> usize {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while tokio::time::Instant::now() < deadline {
            for &idx in among {
                if nodes[idx].is_leader().await {
                    return idx;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("no leader was elected among {among:?}");
    }

    async fn wait_for_len(node: &RaftNode, expected: usize) -> usize {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let mut len = node.len().await;
        while len != expected && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
            len = node.len().await;
        }
        len
    }

    fn insert_command() -> RdfCommand {
        RdfCommand::Insert {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "\"o\"".to_string(),
        }
    }

    /// A three-node cluster elects a leader, replicates and hands leadership
    /// over, with every Raft RPC over mutually authenticated TLS.
    #[tokio::test]
    async fn test_raft_cluster_runs_over_tls() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let addrs: BTreeMap<OxirsNodeId, SocketAddr> =
            [1, 2, 3].into_iter().map(|id| (id, free_addr())).collect();
        let tls = identities(dir.path(), &[1, 2, 3], &[]).await;
        let mut nodes = start_nodes(&addrs, &tls).await;

        let leader = wait_for_leader(&nodes, &[0, 1, 2]).await;
        let response = nodes[leader]
            .submit_command(insert_command())
            .await
            .expect("submit over TLS");
        assert_eq!(response, RdfResponse::Success);
        for (idx, node) in nodes.iter().enumerate() {
            assert_eq!(
                wait_for_len(node, 1).await,
                1,
                "node index {idx} did not replicate"
            );
        }

        let target = (leader + 1) % nodes.len();
        let target_id = nodes[target].node_id;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            match nodes[leader].transfer_leadership(target_id).await {
                Ok(()) => break,
                Err(e) if tokio::time::Instant::now() < deadline => {
                    tracing::debug!("leadership transfer not possible yet: {e}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => panic!("leadership transfer failed: {e}"),
            }
        }
        assert_eq!(wait_for_leader(&nodes, &[target]).await, target);

        for node in &mut nodes {
            node.shutdown().await.expect("shutdown");
        }
    }

    /// With TLS configured, the Raft listener does not answer plaintext.
    #[tokio::test]
    async fn test_tls_raft_listener_refuses_plaintext() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let tls_addrs: BTreeMap<OxirsNodeId, SocketAddr> =
            [1, 2].into_iter().map(|id| (id, free_addr())).collect();
        let tls = identities(dir.path(), &[1], &[]).await;
        // Only node 1 runs; its listener is up as soon as init_raft returns.
        let mut tls_node = RaftNode::new(1);
        tls_node.set_network(tls_addrs[&1], [(2, tls_addrs[&2])].into_iter().collect());
        tls_node.set_tls(Arc::clone(&tls[&1]));
        tls_node
            .init_raft([1, 2].into_iter().collect())
            .await
            .expect("init_raft");

        // A node without TLS answers the same probe, so the probe works.
        let plain_addrs: BTreeMap<OxirsNodeId, SocketAddr> =
            [1, 2].into_iter().map(|id| (id, free_addr())).collect();
        let mut plain_node = RaftNode::new(1);
        plain_node.set_network(
            plain_addrs[&1],
            [(2, plain_addrs[&2])].into_iter().collect(),
        );
        plain_node
            .init_raft([1, 2].into_iter().collect())
            .await
            .expect("init_raft");

        assert!(crate::raft_network::plaintext_vote_is_answered(plain_addrs[&1]).await);
        assert!(
            !crate::raft_network::plaintext_vote_is_answered(tls_addrs[&1]).await,
            "a TLS-configured Raft node answered a plaintext request"
        );

        tls_node.shutdown().await.expect("shutdown");
        plain_node.shutdown().await.expect("shutdown");
    }

    /// A node whose certificate the others do not trust takes no part: the
    /// two trusted nodes form a majority and replicate without it.
    #[tokio::test]
    async fn test_untrusted_raft_peer_is_shut_out() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let addrs: BTreeMap<OxirsNodeId, SocketAddr> =
            [1, 2, 3].into_iter().map(|id| (id, free_addr())).collect();
        let tls = identities(dir.path(), &[1, 2, 3], &[3]).await;
        let mut nodes = start_nodes(&addrs, &tls).await;

        let leader = wait_for_leader(&nodes, &[0, 1]).await;
        nodes[leader]
            .submit_command(insert_command())
            .await
            .expect("the trusted majority commits");
        assert_eq!(wait_for_len(&nodes[0], 1).await, 1);
        assert_eq!(wait_for_len(&nodes[1], 1).await, 1);

        tokio::time::sleep(Duration::from_secs(2)).await;
        assert_eq!(
            nodes[2].len().await,
            0,
            "the untrusted node received the entry"
        );
        assert!(!nodes[2].is_leader().await);

        for node in &mut nodes {
            node.shutdown().await.expect("shutdown");
        }
    }

    /// TLS that is configured but cannot be initialized stops Raft from
    /// starting at all; nothing listens in plaintext.
    #[tokio::test]
    async fn test_raft_tls_that_cannot_initialize_fails_closed() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let broken = TlsConfig {
            enabled: true,
            cert_dir: dir.path().to_path_buf(),
            ca_file: None,
            auto_rotate: false,
            ..Default::default()
        };
        let addr = free_addr();
        let mut node = RaftNode::new(1);
        node.set_network(addr, [(2, free_addr())].into_iter().collect());
        node.set_tls(Arc::new(TlsManager::new(broken, 1)));

        assert!(
            node.init_raft([1, 2].into_iter().collect()).await.is_err(),
            "Raft started without the TLS it was configured for"
        );
        std::net::TcpListener::bind(addr).expect("the Raft port is still free");
    }
}
