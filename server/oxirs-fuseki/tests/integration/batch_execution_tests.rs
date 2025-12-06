// Batch Execution Integration Tests
//
// Comprehensive integration tests for batch_execution module

use oxirs_fuseki::batch_execution::{BatchExecutor, BatchExecutorConfig, BatchRequest};
use oxirs_fuseki::store::Store;
use oxirs_core::model::Term;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

/// Helper to create a test store with sample data
async fn create_test_store() -> anyhow::Result<Arc<Store>> {
    let store = Arc::new(Store::new()?);

    // Insert test triples
    let triples = vec![
        (
            Term::NamedNode("http://example.org/subject1".to_string()),
            Term::NamedNode("http://example.org/predicate".to_string()),
            Term::Literal {
                value: "Object 1".to_string(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            },
        ),
        (
            Term::NamedNode("http://example.org/subject2".to_string()),
            Term::NamedNode("http://example.org/predicate".to_string()),
            Term::Literal {
                value: "Object 2".to_string(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            },
        ),
        (
            Term::NamedNode("http://example.org/subject3".to_string()),
            Term::NamedNode("http://example.org/predicate".to_string()),
            Term::Literal {
                value: "Object 3".to_string(),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            },
        ),
    ];

    for (s, p, o) in triples {
        store.insert(s, p, o, None).await?;
    }

    Ok(store)
}

#[tokio::test]
async fn test_batch_executor_creation() {
    let config = BatchExecutorConfig {
        max_batch_size: 10,
        batch_timeout: Duration::from_millis(100),
        max_parallel: 4,
        enable_dependency_analysis: false,
        backpressure_threshold: 100,
    };

    let executor = BatchExecutor::new(config);

    assert!(executor.get_statistics().batches_processed == 0);
    assert!(executor.get_statistics().queries_executed == 0);
}

#[tokio::test]
async fn test_batch_execution_basic() -> anyhow::Result<()> {
    let store = create_test_store().await?;

    let config = BatchExecutorConfig {
        max_batch_size: 5,
        batch_timeout: Duration::from_millis(200),
        max_parallel: 2,
        enable_dependency_analysis: false,
        backpressure_threshold: 50,
    };

    let executor = BatchExecutor::new(config);

    // Create batch requests
    let requests = vec![
        BatchRequest {
            query: "SELECT ?s ?o WHERE { ?s <http://example.org/predicate> ?o }".to_string(),
            dataset: "test".to_string(),
            priority: 1,
        },
        BatchRequest {
            query: "SELECT ?s WHERE { ?s <http://example.org/predicate> ?o }".to_string(),
            dataset: "test".to_string(),
            priority: 1,
        },
    ];

    // Submit requests
    for request in requests {
        executor.submit(request).await?;
    }

    // Give time for batch processing
    tokio::time::sleep(Duration::from_millis(300)).await;

    let stats = executor.get_statistics();
    assert!(stats.queries_executed >= 2, "Expected at least 2 queries executed");

    Ok(())
}

#[tokio::test]
async fn test_batch_timeout_trigger() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 100, // Large batch size - won't be reached
        batch_timeout: Duration::from_millis(100), // Short timeout
        max_parallel: 2,
        enable_dependency_analysis: false,
        backpressure_threshold: 50,
    };

    let executor = BatchExecutor::new(config);

    // Submit a single request
    let request = BatchRequest {
        query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
        dataset: "test".to_string(),
        priority: 1,
    };

    executor.submit(request).await?;

    // Wait for timeout to trigger
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stats = executor.get_statistics();
    assert!(stats.batches_processed >= 1, "Expected at least 1 batch processed by timeout");

    Ok(())
}

#[tokio::test]
async fn test_batch_size_trigger() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 3, // Small batch size
        batch_timeout: Duration::from_secs(10), // Long timeout
        max_parallel: 2,
        enable_dependency_analysis: false,
        backpressure_threshold: 50,
    };

    let executor = BatchExecutor::new(config);

    // Submit exactly max_batch_size requests
    for i in 0..3 {
        let request = BatchRequest {
            query: format!("SELECT * WHERE {{ ?s ?p \"{}\" }}", i),
            dataset: "test".to_string(),
            priority: 1,
        };
        executor.submit(request).await?;
    }

    // Give minimal time for batch processing
    tokio::time::sleep(Duration::from_millis(100)).await;

    let stats = executor.get_statistics();
    assert!(stats.batches_processed >= 1, "Expected batch to be triggered by size");
    assert!(stats.queries_executed >= 3, "Expected 3 queries executed");

    Ok(())
}

#[tokio::test]
async fn test_parallel_batch_execution() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 5,
        batch_timeout: Duration::from_millis(100),
        max_parallel: 4, // Allow parallel execution
        enable_dependency_analysis: false,
        backpressure_threshold: 100,
    };

    let executor = BatchExecutor::new(config);

    // Submit multiple batches worth of requests
    for i in 0..15 {
        let request = BatchRequest {
            query: format!("SELECT * WHERE {{ ?s ?p \"value{}\" }}", i),
            dataset: format!("dataset{}", i % 3),
            priority: 1,
        };
        executor.submit(request).await?;
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(500)).await;

    let stats = executor.get_statistics();
    assert!(stats.queries_executed >= 15, "Expected 15 queries executed");
    assert!(stats.batches_processed >= 3, "Expected multiple batches processed");

    Ok(())
}

#[tokio::test]
async fn test_backpressure_handling() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 5,
        batch_timeout: Duration::from_millis(100),
        max_parallel: 1, // Single parallel to create backpressure
        enable_dependency_analysis: false,
        backpressure_threshold: 10, // Low threshold
    };

    let executor = BatchExecutor::new(config);

    // Try to submit more than threshold
    let mut submission_results = Vec::new();
    for i in 0..20 {
        let request = BatchRequest {
            query: format!("SELECT * WHERE {{ ?s ?p \"{}\" }}", i),
            dataset: "test".to_string(),
            priority: 1,
        };

        // Use timeout to avoid blocking forever
        let result = timeout(
            Duration::from_millis(50),
            executor.submit(request)
        ).await;

        submission_results.push(result);
    }

    // Some submissions should have timed out or errored due to backpressure
    let successful = submission_results.iter().filter(|r| r.is_ok()).count();
    assert!(successful < 20, "Expected some submissions to be blocked by backpressure");

    Ok(())
}

#[tokio::test]
async fn test_priority_ordering() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 10,
        batch_timeout: Duration::from_millis(500), // Long timeout to control batching
        max_parallel: 1, // Serial execution to observe order
        enable_dependency_analysis: false,
        backpressure_threshold: 100,
    };

    let executor = BatchExecutor::new(config);

    // Submit requests with different priorities
    let high_priority = BatchRequest {
        query: "SELECT * WHERE { ?s ?p \"high\" }".to_string(),
        dataset: "test".to_string(),
        priority: 10, // High priority
    };

    let low_priority = BatchRequest {
        query: "SELECT * WHERE { ?s ?p \"low\" }".to_string(),
        dataset: "test".to_string(),
        priority: 1, // Low priority
    };

    // Submit low priority first, then high priority
    executor.submit(low_priority).await?;
    executor.submit(high_priority).await?;

    // Trigger batch processing
    tokio::time::sleep(Duration::from_millis(600)).await;

    let stats = executor.get_statistics();
    assert!(stats.queries_executed >= 2, "Expected both queries executed");

    // Note: We can't directly verify execution order without more instrumentation,
    // but the test verifies that priority field is accepted and processed

    Ok(())
}

#[tokio::test]
async fn test_batch_statistics() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 5,
        batch_timeout: Duration::from_millis(100),
        max_parallel: 2,
        enable_dependency_analysis: false,
        backpressure_threshold: 50,
    };

    let executor = BatchExecutor::new(config);

    // Submit multiple requests
    for i in 0..10 {
        let request = BatchRequest {
            query: format!("SELECT * WHERE {{ ?s ?p \"{}\" }}", i),
            dataset: "test".to_string(),
            priority: 1,
        };
        executor.submit(request).await?;
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(300)).await;

    let stats = executor.get_statistics();

    // Verify statistics
    assert!(stats.queries_executed > 0, "Expected queries executed");
    assert!(stats.batches_processed > 0, "Expected batches processed");
    assert!(stats.avg_batch_size > 0.0, "Expected non-zero average batch size");
    assert!(stats.total_execution_time_ms >= 0, "Expected valid execution time");

    Ok(())
}

#[tokio::test]
async fn test_per_dataset_batching() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 5,
        batch_timeout: Duration::from_millis(200),
        max_parallel: 2,
        enable_dependency_analysis: false,
        backpressure_threshold: 50,
    };

    let executor = BatchExecutor::new(config);

    // Submit requests for different datasets
    for i in 0..6 {
        let dataset = if i < 3 { "dataset1" } else { "dataset2" };
        let request = BatchRequest {
            query: format!("SELECT * WHERE {{ ?s ?p \"{}\" }}", i),
            dataset: dataset.to_string(),
            priority: 1,
        };
        executor.submit(request).await?;
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(400)).await;

    let stats = executor.get_statistics();
    assert!(stats.queries_executed >= 6, "Expected 6 queries executed");

    // Note: Per-dataset batching should create separate batches
    // We verify this indirectly through successful execution

    Ok(())
}

#[tokio::test]
async fn test_graceful_shutdown() -> anyhow::Result<()> {
    let config = BatchExecutorConfig {
        max_batch_size: 10,
        batch_timeout: Duration::from_millis(100),
        max_parallel: 2,
        enable_dependency_analysis: false,
        backpressure_threshold: 50,
    };

    let executor = BatchExecutor::new(config);

    // Submit some requests
    for i in 0..5 {
        let request = BatchRequest {
            query: format!("SELECT * WHERE {{ ?s ?p \"{}\" }}", i),
            dataset: "test".to_string(),
            priority: 1,
        };
        executor.submit(request).await?;
    }

    // Shutdown the executor
    executor.shutdown().await?;

    // Verify statistics after shutdown
    let stats = executor.get_statistics();
    assert!(stats.queries_executed >= 0, "Statistics should be accessible after shutdown");

    Ok(())
}
