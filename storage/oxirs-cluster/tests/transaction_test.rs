//! Integration tests for cross-shard transactions with 2PC

#[cfg(test)]
mod transaction_tests {
    use oxirs_cluster::network::{NetworkConfig, NetworkService};
    use oxirs_cluster::shard::{ShardRouter, ShardingStrategy};
    use oxirs_cluster::shard_manager::{ShardManager, ShardManagerConfig};
    use oxirs_cluster::storage::mock::MockStorageBackend;
    use oxirs_cluster::transaction::{
        IsolationLevel, TransactionConfig, TransactionCoordinator, TransactionId, TransactionOp,
        TransactionState,
    };
    use oxirs_cluster::transaction_optimizer::{DeadlockDetector, TwoPhaseOptimizer};
    use oxirs_core::model::{NamedNode, Subject, Triple};
    use std::sync::Arc;
    use std::time::Duration;

    async fn setup_transaction_coordinator() -> TransactionCoordinator {
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
                NamedNode::new(format!("{:?}", level)).unwrap(),
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
        let optimizer = TwoPhaseOptimizer::new();

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

        // Check optimization stats
        let stats = optimizer.get_statistics().await;
        assert!(stats.readonly_optimized > 0);
    }

    #[tokio::test]
    async fn test_single_shard_optimization() {
        let coordinator = setup_transaction_coordinator().await;
        let optimizer = TwoPhaseOptimizer::new();

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
                        NamedNode::new("Single Shard").unwrap(),
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
                        NamedNode::new("25").unwrap(),
                    ),
                },
            )
            .await
            .unwrap();

        coordinator.commit_transaction(&tx_id).await.unwrap();

        // Check optimization stats
        let stats = optimizer.get_statistics().await;
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
                NamedNode::new("true").unwrap(),
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
            NamedNode::new("pending").unwrap(),
        );
        coordinator
            .add_operation(&tx_id, TransactionOp::Insert { triple })
            .await
            .unwrap();

        // Manually abort the transaction
        // In real scenario, this might happen due to conflicts or failures
        // For now, we'll test cleanup
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
                    NamedNode::new(i.to_string()).unwrap(),
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

        // Transaction should fail due to timeout
        coordinator
            .cleanup_transactions(Duration::from_millis(150))
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
                NamedNode::new(i.to_string()).unwrap(),
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
