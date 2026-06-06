use crate::embeddings::EmbeddingStrategy;
use crate::store_integration_types::*;
use crate::Vector;
use std::time::Duration;

#[test]
fn test_store_integration_config() {
    let config = StoreIntegrationConfig::default();
    assert!(config.real_time_sync);
    assert_eq!(config.batch_size, 1000);
    assert!(config.incremental_updates);
}

#[test]
fn test_transaction_lifecycle() -> anyhow::Result<()> {
    let config = StoreIntegrationConfig::default();
    let tm = TransactionManager::new(config);

    let tx_id = tm.begin_transaction(IsolationLevel::ReadCommitted)?;
    assert!(tx_id > 0);

    let result = tm.commit_transaction(tx_id);
    assert!(result.is_ok());
    Ok(())
}

#[test]
fn test_cache_manager() -> anyhow::Result<()> {
    let config = StoreCacheConfig {
        enable_vector_cache: true,
        enable_query_cache: true,
        cache_size_mb: 128,
        cache_ttl: Duration::from_secs(300),
        enable_compression: false,
    };

    let cache_manager = CacheManager::new(config);
    let vector = Vector::new(vec![1.0, 2.0, 3.0]);

    cache_manager.cache_vector("test_uri".to_string(), vector.clone());
    let cached = cache_manager.get_vector("test_uri");

    assert!(cached.is_some());
    assert_eq!(cached.expect("test value").vector, vector);
    Ok(())
}

#[test]
fn test_streaming_engine() {
    let config = StreamingConfig {
        enable_streaming: true,
        buffer_size: 1000,
        flush_interval: Duration::from_millis(100),
        enable_backpressure: true,
        max_lag: Duration::from_secs(1),
    };

    let streaming_engine = StreamingEngine::new(config);
    let operation = StreamingOperation::VectorInsert {
        uri: "test_uri".to_string(),
        vector: Vector::new(vec![1.0, 2.0, 3.0]),
        priority: Priority::Normal,
    };

    let result = streaming_engine.submit_operation(operation);
    assert!(result.is_ok());
}

#[test]
fn test_integrated_vector_store() -> anyhow::Result<()> {
    let config = StoreIntegrationConfig::default();
    let store = IntegratedVectorStore::new(config, EmbeddingStrategy::TfIdf)?;

    let tx_id = store.begin_transaction(IsolationLevel::ReadCommitted)?;
    assert!(tx_id > 0);

    let vector = Vector::new(vec![1.0, 2.0, 3.0]);
    let result = store.transactional_insert(tx_id, "test_uri".to_string(), vector, None);
    assert!(result.is_ok());

    let commit_result = store.commit_transaction(tx_id);
    assert!(commit_result.is_ok());
    Ok(())
}
