// Streaming Results Integration Tests
//
// Comprehensive integration tests for streaming_results module

use oxirs_fuseki::streaming_results::{
    StreamingManager, StreamingConfig, ResultStream, OutputFormat, CompressionType
};
use oxirs_fuseki::store::Store;
use oxirs_core::model::Term;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

/// Helper to create a test store with sample data
async fn create_test_store_with_data() -> anyhow::Result<Arc<Store>> {
    let store = Arc::new(Store::new()?);

    // Insert test triples for streaming
    for i in 0..100 {
        let triple = (
            Term::NamedNode(format!("http://example.org/subject{}", i)),
            Term::NamedNode("http://example.org/predicate".to_string()),
            Term::Literal {
                value: format!("Object {}", i),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            },
        );

        store.insert(triple.0, triple.1, triple.2, None).await?;
    }

    Ok(store)
}

#[tokio::test]
async fn test_streaming_manager_creation() {
    let config = StreamingConfig {
        default_chunk_size: 10,
        max_concurrent_streams: 5,
        enable_compression: true,
        default_compression: CompressionType::Gzip,
        compression_level: 6,
        backpressure_threshold: 100,
    };

    let manager = StreamingManager::new(config);

    assert!(manager.get_statistics().active_streams == 0);
    assert!(manager.get_statistics().total_streams_created == 0);
}

#[tokio::test]
async fn test_stream_creation_json() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "test_stream".to_string(),
            OutputFormat::Json,
            None,
        )
        .await?;

    assert!(!stream_id.is_empty());

    let stats = manager.get_statistics();
    assert!(stats.active_streams >= 1);
    assert!(stats.total_streams_created >= 1);

    Ok(())
}

#[tokio::test]
async fn test_stream_creation_xml() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "xml_stream".to_string(),
            OutputFormat::Xml,
            None,
        )
        .await?;

    assert!(!stream_id.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_stream_creation_csv() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "csv_stream".to_string(),
            OutputFormat::Csv,
            None,
        )
        .await?;

    assert!(!stream_id.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_stream_with_gzip_compression() -> anyhow::Result<()> {
    let config = StreamingConfig {
        enable_compression: true,
        default_compression: CompressionType::Gzip,
        compression_level: 6,
        ..Default::default()
    };

    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "compressed_stream".to_string(),
            OutputFormat::Json,
            Some(CompressionType::Gzip),
        )
        .await?;

    assert!(!stream_id.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_stream_with_brotli_compression() -> anyhow::Result<()> {
    let config = StreamingConfig {
        enable_compression: true,
        default_compression: CompressionType::Brotli,
        compression_level: 6,
        ..Default::default()
    };

    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "brotli_stream".to_string(),
            OutputFormat::Json,
            Some(CompressionType::Brotli),
        )
        .await?;

    assert!(!stream_id.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_stream_data_writing() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "write_test".to_string(),
            OutputFormat::Json,
            None,
        )
        .await?;

    // Write test data
    let test_data = b"{ \"test\": \"data\" }";
    manager.write_to_stream(&stream_id, test_data).await?;

    let stats = manager.get_statistics();
    assert!(stats.total_bytes_streamed > 0);

    Ok(())
}

#[tokio::test]
async fn test_stream_close() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "close_test".to_string(),
            OutputFormat::Json,
            None,
        )
        .await?;

    // Close the stream
    manager.close_stream(&stream_id).await?;

    let stats = manager.get_statistics();
    assert!(stats.active_streams == 0, "Stream should be closed");
    assert!(stats.total_streams_closed >= 1);

    Ok(())
}

#[tokio::test]
async fn test_multiple_concurrent_streams() -> anyhow::Result<()> {
    let config = StreamingConfig {
        max_concurrent_streams: 5,
        ..Default::default()
    };

    let manager = StreamingManager::new(config);

    // Create multiple streams
    let mut stream_ids = Vec::new();
    for i in 0..3 {
        let stream_id = manager
            .create_stream(
                format!("stream_{}", i),
                OutputFormat::Json,
                None,
            )
            .await?;
        stream_ids.push(stream_id);
    }

    let stats = manager.get_statistics();
    assert!(stats.active_streams >= 3);

    // Close all streams
    for stream_id in stream_ids {
        manager.close_stream(&stream_id).await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_stream_chunking() -> anyhow::Result<()> {
    let config = StreamingConfig {
        default_chunk_size: 10, // Small chunk size
        ..Default::default()
    };

    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "chunk_test".to_string(),
            OutputFormat::Json,
            None,
        )
        .await?;

    // Write data larger than chunk size
    let large_data = vec![b'x'; 100];
    manager.write_to_stream(&stream_id, &large_data).await?;

    let stats = manager.get_statistics();
    assert!(stats.total_bytes_streamed >= 100);

    manager.close_stream(&stream_id).await?;

    Ok(())
}

#[tokio::test]
async fn test_stream_backpressure() -> anyhow::Result<()> {
    let config = StreamingConfig {
        backpressure_threshold: 10, // Very low threshold
        max_concurrent_streams: 2,
        ..Default::default()
    };

    let manager = StreamingManager::new(config);

    // Try to create more streams than allowed
    let mut stream_ids = Vec::new();
    let mut creation_results = Vec::new();

    for i in 0..5 {
        let result = timeout(
            Duration::from_millis(50),
            manager.create_stream(
                format!("backpressure_{}", i),
                OutputFormat::Json,
                None,
            )
        ).await;

        creation_results.push(result);
    }

    // Some creations should have timed out or failed due to backpressure
    let successful = creation_results.iter().filter(|r| r.is_ok()).count();
    assert!(successful <= 2, "Expected backpressure to limit concurrent streams");

    Ok(())
}

#[tokio::test]
async fn test_compression_statistics() -> anyhow::Result<()> {
    let config = StreamingConfig {
        enable_compression: true,
        default_compression: CompressionType::Gzip,
        compression_level: 6,
        ..Default::default()
    };

    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "compression_stats".to_string(),
            OutputFormat::Json,
            Some(CompressionType::Gzip),
        )
        .await?;

    // Write compressible data
    let data = b"{ \"test\": \"data\" }".repeat(100);
    manager.write_to_stream(&stream_id, &data).await?;

    manager.close_stream(&stream_id).await?;

    let stats = manager.get_statistics();
    assert!(stats.total_bytes_streamed > 0);
    assert!(stats.compression_ratio > 0.0, "Expected compression ratio to be calculated");

    Ok(())
}

#[tokio::test]
async fn test_stream_lifecycle() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    // Create stream
    let stream_id = manager
        .create_stream(
            "lifecycle_test".to_string(),
            OutputFormat::Json,
            None,
        )
        .await?;

    let stats_after_create = manager.get_statistics();
    assert!(stats_after_create.active_streams >= 1);

    // Write data
    manager.write_to_stream(&stream_id, b"test data").await?;

    let stats_after_write = manager.get_statistics();
    assert!(stats_after_write.total_bytes_streamed > 0);

    // Close stream
    manager.close_stream(&stream_id).await?;

    let stats_after_close = manager.get_statistics();
    assert!(stats_after_close.active_streams == 0);
    assert!(stats_after_close.total_streams_closed >= 1);

    Ok(())
}

#[tokio::test]
async fn test_different_output_formats() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let formats = vec![
        OutputFormat::Json,
        OutputFormat::Xml,
        OutputFormat::Csv,
        OutputFormat::Tsv,
        OutputFormat::NTriples,
        OutputFormat::Turtle,
    ];

    for (i, format) in formats.iter().enumerate() {
        let stream_id = manager
            .create_stream(
                format!("format_test_{}", i),
                format.clone(),
                None,
            )
            .await?;

        assert!(!stream_id.is_empty());
        manager.close_stream(&stream_id).await?;
    }

    let stats = manager.get_statistics();
    assert!(stats.total_streams_created >= 6);

    Ok(())
}

#[tokio::test]
async fn test_zero_copy_streaming() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let stream_id = manager
        .create_stream(
            "zero_copy_test".to_string(),
            OutputFormat::Json,
            None,
        )
        .await?;

    // Simulate zero-copy by writing large data
    let large_data = vec![b'a'; 10000];
    manager.write_to_stream(&stream_id, &large_data).await?;

    let stats = manager.get_statistics();
    assert!(stats.total_bytes_streamed >= 10000);

    manager.close_stream(&stream_id).await?;

    Ok(())
}

#[tokio::test]
async fn test_stream_error_handling() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    // Try to write to non-existent stream
    let result = manager.write_to_stream("nonexistent", b"data").await;
    assert!(result.is_err(), "Expected error when writing to nonexistent stream");

    // Try to close non-existent stream
    let result = manager.close_stream("nonexistent").await;
    assert!(result.is_err(), "Expected error when closing nonexistent stream");

    Ok(())
}

#[tokio::test]
async fn test_stream_statistics_accuracy() -> anyhow::Result<()> {
    let config = StreamingConfig::default();
    let manager = StreamingManager::new(config);

    let initial_stats = manager.get_statistics();

    // Create and use a stream
    let stream_id = manager
        .create_stream(
            "stats_accuracy".to_string(),
            OutputFormat::Json,
            None,
        )
        .await?;

    let data = b"test data 123";
    manager.write_to_stream(&stream_id, data).await?;

    manager.close_stream(&stream_id).await?;

    let final_stats = manager.get_statistics();

    assert!(
        final_stats.total_streams_created > initial_stats.total_streams_created,
        "Stream creation count should increase"
    );
    assert!(
        final_stats.total_bytes_streamed >= data.len() as u64,
        "Bytes streamed should be at least the data size"
    );
    assert!(
        final_stats.total_streams_closed > initial_stats.total_streams_closed,
        "Stream closed count should increase"
    );

    Ok(())
}
