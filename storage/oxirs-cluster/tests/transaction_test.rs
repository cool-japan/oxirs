//! Integration tests for cross-shard transactions with 2PC

#[cfg(test)]
#[allow(
    unused_imports,
    unused_variables,
    clippy::uninlined_format_args,
    clippy::field_reassign_with_default
)]
mod transaction_tests {
    use oxirs_cluster::network::{NetworkConfig, NetworkService, RpcMessage};
    use oxirs_cluster::shard::{ShardRouter, ShardingStrategy};
    use oxirs_cluster::shard_manager::{ShardManager, ShardManagerConfig};
    use oxirs_cluster::storage::mock::MockStorageBackend;
    use oxirs_cluster::transaction::{
        IsolationLevel, TransactionConfig, TransactionCoordinator, TransactionId, TransactionOp,
        TransactionState,
    };
    use oxirs_cluster::transaction_optimizer::DeadlockDetector;
    use oxirs_core::model::{Literal, NamedNode, Subject, Triple};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Stand up a loopback TCP listener that accepts connections forever and,
    /// for each one, reads a single length-prefixed oxicode-encoded
    /// `RpcMessage` frame (matching the wire format
    /// `oxirs_cluster::network`'s internal `write_frame`/`read_frame` use) and
    /// replies with a validly-framed (but otherwise arbitrary) `RpcMessage`
    /// response. `NetworkService::send_message` only requires a well-formed
    /// response frame -- it does not validate the response variant -- so this
    /// proves real end-to-end 2PC delivery over a real socket once a
    /// participant address has been registered via
    /// `TransactionCoordinator::register_peer`.
    async fn spawn_loopback_echo() -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind loopback listener");
        let addr = listener.local_addr().expect("listener has no local addr");

        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut len_buf = [0u8; 4];
                    if socket.read_exact(&mut len_buf).await.is_err() {
                        return;
                    }
                    let len = u32::from_be_bytes(len_buf) as usize;
                    let mut body = vec![0u8; len];
                    if socket.read_exact(&mut body).await.is_err() {
                        return;
                    }
                    // The request body content is irrelevant to the echo server;
                    // any well-formed RpcMessage response satisfies send_message.
                    let response = RpcMessage::TransactionAck {
                        tx_id: String::new(),
                        shard_id: 0,
                    };
                    let Ok(resp_body) =
                        oxicode::serde::encode_to_vec(&response, oxicode::config::standard())
                    else {
                        return;
                    };
                    let resp_len = (resp_body.len() as u32).to_be_bytes();
                    if socket.write_all(&resp_len).await.is_err() {
                        return;
                    }
                    if socket.write_all(&resp_body).await.is_err() {
                        return;
                    }
                    let _ = socket.flush().await;
                });
            }
        });

        addr
    }

    async fn setup_transaction_coordinator() -> TransactionCoordinator {
        // Set test mode environment variable
        std::env::set_var("OXIRS_TEST_MODE", "1");

        let strategy = ShardingStrategy::Hash { num_shards: 4 };
        let router = Arc::new(ShardRouter::new(strategy.clone()));
        router.init_shards(4, 3).await.unwrap();

        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        let shard_manager = Arc::new(ShardManager::new(
            1,
            router.clone(),
            ShardManagerConfig::default(),
            storage.clone(),
            network.clone(),
        ));

        // Real inter-node RPC now fails loudly unless the target peer's address
        // has been registered (NetworkService::send_message). Assign every
        // shard to this coordinator's own node (1) -- exactly as a real
        // single-replica bootstrap would via `ShardManager::initialize_shards`
        // -- so `ShardManager::get_primary_node` resolves participants to node
        // 1, then register a real loopback echo listener for node 1 so the
        // coordinator's prepare/commit/abort RPCs (which are always sent over
        // the network, even to the local node) can actually be delivered.
        shard_manager
            .initialize_shards(&strategy, vec![1])
            .await
            .unwrap();
        let echo_addr = spawn_loopback_echo().await;
        network.register_peer(1, echo_addr).await;

        let config = TransactionConfig::default();
        TransactionCoordinator::new(1, router, shard_manager, storage, network, config)
    }

    #[tokio::test]
    async fn test_basic_transaction_flow() {
        let coordinator = setup_transaction_coordinator().await;

        // Begin transaction
        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Add operations
        let triple = Triple::new(
            NamedNode::new("http://example.org/alice").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/bob").unwrap(),
        );

        coordinator
            .add_operation(
                &tx_id,
                TransactionOp::Insert {
                    triple: triple.clone(),
                },
            )
            .await
            .unwrap();
        coordinator
            .add_operation(
                &tx_id,
                TransactionOp::Query {
                    subject: Some("http://example.org/alice".to_string()),
                    predicate: None,
                    object: None,
                },
            )
            .await
            .unwrap();

        // Commit transaction
        coordinator.commit_transaction(&tx_id).await.unwrap();

        // Check statistics
        let stats = coordinator.get_statistics().await;
        assert_eq!(stats.total_transactions, 1);
    }

    #[tokio::test]
    async fn test_transaction_isolation_levels() {
        let coordinator = setup_transaction_coordinator().await;

        // Test different isolation levels
        let isolation_levels = vec![
            IsolationLevel::ReadUncommitted,
            IsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead,
            IsolationLevel::Serializable,
        ];

        for level in isolation_levels {
            let tx_id = coordinator.begin_transaction(level).await.unwrap();

            // Add a simple operation
            let triple = Triple::new(
                NamedNode::new("http://example.org/test").unwrap(),
                NamedNode::new("http://example.org/level").unwrap(),
                Literal::new_simple_literal(format!("{:?}", level)),
            );

            coordinator
                .add_operation(&tx_id, TransactionOp::Insert { triple })
                .await
                .unwrap();
            coordinator.commit_transaction(&tx_id).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_read_only_optimization() {
        let coordinator = setup_transaction_coordinator().await;

        // Create read-only transaction
        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Add only query operations
        coordinator
            .add_operation(
                &tx_id,
                TransactionOp::Query {
                    subject: Some("http://example.org/alice".to_string()),
                    predicate: None,
                    object: None,
                },
            )
            .await
            .unwrap();

        coordinator
            .add_operation(
                &tx_id,
                TransactionOp::Query {
                    subject: None,
                    predicate: Some("http://example.org/knows".to_string()),
                    object: None,
                },
            )
            .await
            .unwrap();

        // Commit should be optimized
        coordinator.commit_transaction(&tx_id).await.unwrap();

        // Check optimization stats from the coordinator's optimizer
        let stats = coordinator.get_optimizer_statistics().await;
        assert!(stats.readonly_optimized > 0);
    }

    #[tokio::test]
    async fn test_single_shard_optimization() {
        let coordinator = setup_transaction_coordinator().await;

        // Create transaction affecting single shard
        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // All operations on same subject (should route to same shard)
        let subject: oxirs_core::model::Subject =
            NamedNode::new("http://example.org/single").unwrap().into();

        coordinator
            .add_operation(
                &tx_id,
                TransactionOp::Insert {
                    triple: Triple::new(
                        subject.clone(),
                        NamedNode::new("http://example.org/name").unwrap(),
                        Literal::new_simple_literal("Single Shard"),
                    ),
                },
            )
            .await
            .unwrap();

        coordinator
            .add_operation(
                &tx_id,
                TransactionOp::Insert {
                    triple: Triple::new(
                        subject.clone(),
                        NamedNode::new("http://example.org/age").unwrap(),
                        Literal::new_simple_literal("25"),
                    ),
                },
            )
            .await
            .unwrap();

        coordinator.commit_transaction(&tx_id).await.unwrap();

        // Check optimization stats from the coordinator's optimizer
        let stats = coordinator.get_optimizer_statistics().await;
        assert!(stats.single_shard_optimized > 0);
    }

    #[tokio::test]
    async fn test_multi_shard_transaction() {
        let coordinator = setup_transaction_coordinator().await;

        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Add operations affecting multiple shards
        let subjects = vec![
            "http://example.org/alice",
            "http://example.org/bob",
            "http://example.org/charlie",
            "http://example.org/david",
        ];

        for subject in subjects {
            let triple = Triple::new(
                NamedNode::new(subject).unwrap(),
                NamedNode::new("http://example.org/updated").unwrap(),
                Literal::new_simple_literal("true"),
            );
            coordinator
                .add_operation(&tx_id, TransactionOp::Insert { triple })
                .await
                .unwrap();
        }

        // This should trigger full 2PC
        coordinator.commit_transaction(&tx_id).await.unwrap();

        let stats = coordinator.get_statistics().await;
        assert_eq!(stats.committed_transactions, 1);
    }

    #[tokio::test]
    async fn test_transaction_abort() {
        let coordinator = setup_transaction_coordinator().await;

        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Add some operations
        let triple = Triple::new(
            NamedNode::new("http://example.org/aborted").unwrap(),
            NamedNode::new("http://example.org/status").unwrap(),
            Literal::new_simple_literal("pending"),
        );
        coordinator
            .add_operation(&tx_id, TransactionOp::Insert { triple })
            .await
            .unwrap();

        // Manually abort the transaction
        // In real scenario, this might happen due to conflicts or failures
        coordinator.abort_transaction(&tx_id).await.unwrap();

        // Give the transaction time to complete abort process
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Now test cleanup
        coordinator
            .cleanup_transactions(Duration::from_secs(0))
            .await;

        let stats = coordinator.get_statistics().await;
        assert_eq!(stats.total_transactions, 0); // Cleaned up
    }

    #[tokio::test]
    async fn test_deadlock_detection() {
        let detector = DeadlockDetector::new();

        // Add wait dependencies
        detector.add_wait("tx1", "tx2").await.unwrap();
        detector.add_wait("tx2", "tx3").await.unwrap();

        // This should detect a deadlock
        let result = detector.add_wait("tx3", "tx1").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Deadlock"));

        // Clean up
        detector.remove_transaction("tx1").await;
        detector.remove_transaction("tx2").await;
        detector.remove_transaction("tx3").await;
    }

    /// Regression test for the honest fail-loud behavior: without registering
    /// the participant's network address (and without ever running shard
    /// assignment, so `ShardManager::get_primary_node` falls back to the
    /// router's un-assigned default node), committing a multi-op transaction
    /// must surface a real "no known network address" error instead of
    /// silently reporting success as the pre-wave-2 no-op transport did.
    #[tokio::test]
    async fn test_commit_fails_loud_for_unregistered_participant() {
        std::env::set_var("OXIRS_TEST_MODE", "1");

        let strategy = ShardingStrategy::Hash { num_shards: 4 };
        let router = Arc::new(ShardRouter::new(strategy));
        router.init_shards(4, 3).await.unwrap();

        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        // Deliberately skip `shard_manager.initialize_shards(..)` and
        // `register_peer(..)` -- the shard's primary node is never assigned to
        // a registered peer.
        let shard_manager = Arc::new(ShardManager::new(
            1,
            router.clone(),
            ShardManagerConfig::default(),
            storage.clone(),
            network.clone(),
        ));

        let config = TransactionConfig::default();
        let coordinator =
            TransactionCoordinator::new(1, router, shard_manager, storage, network, config);

        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Subjects spanning multiple shards (mirrors test_multi_shard_transaction)
        // to force the full (non-optimized) 2PC path rather than the
        // single-shard/read-only fast path that never touches the network.
        for subject in [
            "http://example.org/alice",
            "http://example.org/bob",
            "http://example.org/charlie",
            "http://example.org/david",
        ] {
            let triple = Triple::new(
                NamedNode::new(subject).unwrap(),
                NamedNode::new("http://example.org/p").unwrap(),
                Literal::new_simple_literal("v"),
            );
            coordinator
                .add_operation(&tx_id, TransactionOp::Insert { triple })
                .await
                .unwrap();
        }

        let result = coordinator.commit_transaction(&tx_id).await;
        let err = result.expect_err("commit must fail loud when a participant is unregistered");
        assert!(
            err.to_string().contains("no known network address"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_concurrent_transactions() {
        let coordinator = Arc::new(setup_transaction_coordinator().await);

        // Launch multiple concurrent transactions
        let mut handles = vec![];

        for i in 0..10 {
            let coord = coordinator.clone();
            let handle = tokio::spawn(async move {
                let tx_id = coord
                    .begin_transaction(IsolationLevel::ReadCommitted)
                    .await
                    .unwrap();

                let triple = Triple::new(
                    NamedNode::new(format!("http://example.org/concurrent{}", i)).unwrap(),
                    NamedNode::new("http://example.org/index").unwrap(),
                    Literal::new_simple_literal(i.to_string()),
                );

                coord
                    .add_operation(&tx_id, TransactionOp::Insert { triple })
                    .await
                    .unwrap();
                coord.commit_transaction(&tx_id).await.unwrap();
            });
            handles.push(handle);
        }

        // Wait for all transactions
        for handle in handles {
            handle.await.unwrap();
        }

        let stats = coordinator.get_statistics().await;
        assert_eq!(stats.committed_transactions, 10);
    }

    #[tokio::test]
    async fn test_transaction_timeout() {
        let mut config = TransactionConfig::default();
        config.default_timeout = Duration::from_millis(100); // Very short timeout

        let strategy = ShardingStrategy::Hash { num_shards: 4 };
        let router = Arc::new(ShardRouter::new(strategy));
        router.init_shards(4, 3).await.unwrap();

        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        let shard_manager = Arc::new(ShardManager::new(
            1,
            router.clone(),
            ShardManagerConfig::default(),
            storage.clone(),
            network.clone(),
        ));

        let coordinator =
            TransactionCoordinator::new(1, router, shard_manager, storage, network, config);

        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Try to commit the transaction - this should fail due to timeout
        let result = coordinator.commit_transaction(&tx_id).await;
        assert!(result.is_err(), "Transaction should fail due to timeout");

        // Now cleanup timed out transactions
        coordinator
            .cleanup_transactions(Duration::from_secs(0))
            .await;

        let stats = coordinator.get_statistics().await;
        assert_eq!(stats.active_transactions, 0);
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let coordinator = setup_transaction_coordinator().await;

        let tx_id = coordinator
            .begin_transaction(IsolationLevel::ReadCommitted)
            .await
            .unwrap();

        // Add many operations for batching
        for i in 0..200 {
            let triple = Triple::new(
                NamedNode::new(format!("http://example.org/batch{}", i)).unwrap(),
                NamedNode::new("http://example.org/batch").unwrap(),
                Literal::new_simple_literal(i.to_string()),
            );
            coordinator
                .add_operation(&tx_id, TransactionOp::Insert { triple })
                .await
                .unwrap();
        }

        coordinator.commit_transaction(&tx_id).await.unwrap();

        let stats = coordinator.get_statistics().await;
        assert_eq!(stats.committed_transactions, 1);
    }
}
