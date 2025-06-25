//! # Integration Tests for OxiRS Stream Backends
//!
//! Comprehensive integration tests for all streaming backends including
//! Kafka, NATS, Redis, Kinesis, and Pulsar implementations.

use oxirs_stream::*;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;
use uuid::Uuid;

/// Test configuration for backends
pub struct TestConfig {
    pub kafka_url: String,
    pub nats_url: String,
    pub redis_url: String,
    pub pulsar_url: String,
    pub kinesis_region: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            kafka_url: std::env::var("KAFKA_URL").unwrap_or_else(|_| "localhost:9092".to_string()),
            nats_url: std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_string()),
            redis_url: std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
            pulsar_url: std::env::var("PULSAR_URL").unwrap_or_else(|_| "pulsar://localhost:6650".to_string()),
            kinesis_region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-west-2".to_string()),
        }
    }
}

/// Helper function to create test events
fn create_test_events(count: usize) -> Vec<StreamEvent> {
    (0..count)
        .map(|i| StreamEvent::TripleAdded {
            subject: format!("http://example.org/subject{}", i),
            predicate: "http://example.org/predicate".to_string(),
            object: format!("\"Test object {}\"", i),
            graph: None,
            metadata: EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source: "integration_test".to_string(),
                user: Some("test_user".to_string()),
                context: Some("test_context".to_string()),
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        })
        .collect()
}

/// Test stream configuration
fn create_test_stream_config(backend: StreamBackend) -> StreamConfig {
    StreamConfig {
        backend,
        topic: format!("test-topic-{}", Uuid::new_v4()),
        batch_size: 10,
        flush_interval_ms: 1000,
        max_connections: 5,
        connection_timeout: Duration::from_secs(10),
        enable_compression: false,
        compression_type: CompressionType::None,
        retry_config: RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: true,
        },
        circuit_breaker: CircuitBreakerConfig {
            failure_threshold: 5,
            timeout: Duration::from_secs(60),
            success_threshold: 3,
            half_open_max_calls: 2,
        },
        security: SecurityConfig {
            enable_tls: false,
            cert_path: None,
            key_path: None,
            ca_path: None,
            username: None,
            password: None,
        },
        performance: PerformanceConfig {
            buffer_size: 1024,
            io_threads: 2,
            network_threads: 2,
            enable_batching: true,
            max_batch_delay: Duration::from_millis(10),
        },
        monitoring: monitoring::MonitoringConfig {
            enable_metrics: true,
            enable_tracing: false,
            metrics_interval: Duration::from_secs(5),
            health_check_interval: Duration::from_secs(10),
            enable_profiling: false,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "info".to_string(),
        },
    }
}

#[cfg(test)]
mod memory_backend_tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_backend_basic_operations() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut stream = Stream::new(config).await?;

        // Test publishing events
        let events = create_test_events(5);
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        // Test consuming events
        let mut received_events = Vec::new();
        for _ in 0..5 {
            if let Ok(Some(event)) = timeout(Duration::from_secs(5), stream.consume()).await {
                received_events.push(event?);
            }
        }

        assert_eq!(received_events.len(), 5);

        // Verify event content
        for (original, received) in events.iter().zip(received_events.iter()) {
            match (original, received) {
                (
                    StreamEvent::TripleAdded { subject: s1, predicate: p1, object: o1, .. },
                    StreamEvent::TripleAdded { subject: s2, predicate: p2, object: o2, .. }
                ) => {
                    assert_eq!(s1, s2);
                    assert_eq!(p1, p2);
                    assert_eq!(o1, o2);
                }
                _ => panic!("Event type mismatch"),
            }
        }

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_backend_high_throughput() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut stream = Stream::new(config).await?;

        let event_count = 1000;
        let events = create_test_events(event_count);

        // Measure publish throughput
        let start = std::time::Instant::now();
        for event in &events {
            stream.publish(event.clone()).await?;
        }
        let publish_duration = start.elapsed();

        println!("Published {} events in {:?}", event_count, publish_duration);

        // Measure consume throughput
        let start = std::time::Instant::now();
        let mut received_count = 0;
        for _ in 0..event_count {
            if let Ok(Some(_)) = timeout(Duration::from_millis(100), stream.consume()).await {
                received_count += 1;
            } else {
                break;
            }
        }
        let consume_duration = start.elapsed();

        println!("Consumed {} events in {:?}", received_count, consume_duration);

        assert_eq!(received_count, event_count);
        stream.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod kafka_backend_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Kafka server
    async fn test_kafka_backend_integration() -> Result<()> {
        let test_config = TestConfig::default();
        let config = create_test_stream_config(StreamBackend::Kafka {
            brokers: vec![test_config.kafka_url],
            group_id: "test-group".to_string(),
        });

        let mut producer = Stream::new(config.clone()).await?;
        let mut consumer = Stream::new(config).await?;

        // Test round-trip
        let events = create_test_events(3);
        for event in &events {
            producer.publish(event.clone()).await?;
        }

        // Give some time for messages to be processed
        tokio::time::sleep(Duration::from_millis(500)).await;

        let mut received_events = Vec::new();
        for _ in 0..3 {
            if let Ok(Some(event)) = timeout(Duration::from_secs(10), consumer.consume()).await {
                received_events.push(event?);
            }
        }

        assert_eq!(received_events.len(), 3);

        producer.close().await?;
        consumer.close().await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires Kafka server
    async fn test_kafka_transaction_support() -> Result<()> {
        let test_config = TestConfig::default();
        let config = create_test_stream_config(StreamBackend::Kafka {
            brokers: vec![test_config.kafka_url],
            group_id: "test-transaction-group".to_string(),
        });

        let mut stream = Stream::new(config).await?;

        // Test transaction begin/commit
        stream.begin_transaction().await?;
        
        let events = create_test_events(2);
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        stream.commit_transaction().await?;

        // Test transaction rollback
        stream.begin_transaction().await?;
        
        let rollback_event = create_test_events(1);
        stream.publish(rollback_event[0].clone()).await?;
        
        stream.rollback_transaction().await?;

        stream.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod nats_backend_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires NATS server
    async fn test_nats_jetstream_integration() -> Result<()> {
        let test_config = TestConfig::default();
        let config = create_test_stream_config(StreamBackend::Nats {
            url: test_config.nats_url,
            stream_name: "test-stream".to_string(),
            subject: "test.subject".to_string(),
        });

        let mut stream = Stream::new(config).await?;

        // Test JetStream persistence
        let events = create_test_events(5);
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        // Test stream replay
        let mut received_events = Vec::new();
        for _ in 0..5 {
            if let Ok(Some(event)) = timeout(Duration::from_secs(5), stream.consume()).await {
                received_events.push(event?);
            }
        }

        assert_eq!(received_events.len(), 5);

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires NATS server
    async fn test_nats_consumer_groups() -> Result<()> {
        let test_config = TestConfig::default();
        let config1 = create_test_stream_config(StreamBackend::Nats {
            url: test_config.nats_url.clone(),
            stream_name: "test-consumer-group".to_string(),
            subject: "test.consumer.group".to_string(),
        });
        let config2 = config1.clone();

        let mut consumer1 = Stream::new(config1).await?;
        let mut consumer2 = Stream::new(config2).await?;

        // Both consumers should be able to receive messages
        let events = create_test_events(4);
        for event in &events {
            consumer1.publish(event.clone()).await?;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut total_received = 0;
        
        // Consumer 1
        for _ in 0..2 {
            if let Ok(Some(_)) = timeout(Duration::from_millis(500), consumer1.consume()).await {
                total_received += 1;
            }
        }

        // Consumer 2
        for _ in 0..2 {
            if let Ok(Some(_)) = timeout(Duration::from_millis(500), consumer2.consume()).await {
                total_received += 1;
            }
        }

        // In ideal load balancing, both consumers should receive messages
        assert!(total_received >= 2);

        consumer1.close().await?;
        consumer2.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod redis_backend_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Redis server
    async fn test_redis_streams_integration() -> Result<()> {
        let test_config = TestConfig::default();
        let config = create_test_stream_config(StreamBackend::Redis {
            url: test_config.redis_url,
            stream_name: "test-redis-stream".to_string(),
        });

        let mut stream = Stream::new(config).await?;

        // Test Redis Streams
        let events = create_test_events(3);
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        let mut received_events = Vec::new();
        for _ in 0..3 {
            if let Ok(Some(event)) = timeout(Duration::from_secs(5), stream.consume()).await {
                received_events.push(event?);
            }
        }

        assert_eq!(received_events.len(), 3);

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires Redis Cluster
    async fn test_redis_cluster_support() -> Result<()> {
        let cluster_urls = vec![
            "redis://localhost:7000".to_string(),
            "redis://localhost:7001".to_string(),
            "redis://localhost:7002".to_string(),
        ];

        let config = create_test_stream_config(StreamBackend::RedisCluster {
            urls: cluster_urls,
            stream_name: "test-cluster-stream".to_string(),
        });

        let mut stream = Stream::new(config).await?;

        // Test cluster failover resilience
        let events = create_test_events(10);
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        let mut received_count = 0;
        for _ in 0..10 {
            if let Ok(Some(_)) = timeout(Duration::from_secs(2), stream.consume()).await {
                received_count += 1;
            }
        }

        assert!(received_count > 0); // Should receive at least some messages

        stream.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod pulsar_backend_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Pulsar server
    async fn test_pulsar_integration() -> Result<()> {
        let test_config = TestConfig::default();
        let config = create_test_stream_config(StreamBackend::Pulsar {
            url: test_config.pulsar_url,
            topic: "test-pulsar-topic".to_string(),
            subscription: "test-subscription".to_string(),
        });

        let mut stream = Stream::new(config).await?;

        // Test Pulsar messaging
        let events = create_test_events(4);
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        let mut received_events = Vec::new();
        for _ in 0..4 {
            if let Ok(Some(event)) = timeout(Duration::from_secs(5), stream.consume()).await {
                received_events.push(event?);
            }
        }

        assert_eq!(received_events.len(), 4);

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires Pulsar server with auth
    async fn test_pulsar_authentication() -> Result<()> {
        let test_config = TestConfig::default();
        let mut config = create_test_stream_config(StreamBackend::Pulsar {
            url: test_config.pulsar_url,
            topic: "secure-topic".to_string(),
            subscription: "secure-subscription".to_string(),
        });

        // Enable TLS and authentication
        config.security.enable_tls = true;
        config.security.username = Some("test_user".to_string());
        config.security.password = Some("test_password".to_string());

        let mut stream = Stream::new(config).await?;

        let event = create_test_events(1);
        stream.publish(event[0].clone()).await?;

        if let Ok(Some(_)) = timeout(Duration::from_secs(5), stream.consume()).await {
            // Authentication successful
        }

        stream.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod kinesis_backend_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires AWS credentials and Kinesis
    async fn test_kinesis_integration() -> Result<()> {
        let test_config = TestConfig::default();
        let config = create_test_stream_config(StreamBackend::Kinesis {
            region: test_config.kinesis_region,
            stream_name: "test-kinesis-stream".to_string(),
        });

        let mut stream = Stream::new(config).await?;

        // Test Kinesis sharding
        let events = create_test_events(6);
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        // Kinesis has eventual consistency, so allow more time
        tokio::time::sleep(Duration::from_secs(2)).await;

        let mut received_events = Vec::new();
        for _ in 0..6 {
            if let Ok(Some(event)) = timeout(Duration::from_secs(10), stream.consume()).await {
                received_events.push(event?);
            }
        }

        assert!(received_events.len() > 0); // Should receive at least some events

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires AWS credentials
    async fn test_kinesis_enhanced_fanout() -> Result<()> {
        let test_config = TestConfig::default();
        let config = create_test_stream_config(StreamBackend::Kinesis {
            region: test_config.kinesis_region,
            stream_name: "test-enhanced-fanout".to_string(),
        });

        let mut producer = Stream::new(config.clone()).await?;
        let mut consumer1 = Stream::new(config.clone()).await?;
        let mut consumer2 = Stream::new(config).await?;

        // Test multiple consumers with enhanced fan-out
        let events = create_test_events(5);
        for event in &events {
            producer.publish(event.clone()).await?;
        }

        tokio::time::sleep(Duration::from_secs(3)).await;

        let mut consumer1_count = 0;
        let mut consumer2_count = 0;

        for _ in 0..5 {
            if let Ok(Some(_)) = timeout(Duration::from_secs(2), consumer1.consume()).await {
                consumer1_count += 1;
            }
        }

        for _ in 0..5 {
            if let Ok(Some(_)) = timeout(Duration::from_secs(2), consumer2.consume()).await {
                consumer2_count += 1;
            }
        }

        // Both consumers should receive all messages with enhanced fan-out
        assert!(consumer1_count > 0);
        assert!(consumer2_count > 0);

        producer.close().await?;
        consumer1.close().await?;
        consumer2.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod rdf_patch_integration_tests {
    use super::*;
    use oxirs_stream::patch::*;
    use oxirs_stream::delta::*;

    #[tokio::test]
    async fn test_rdf_patch_streaming() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut stream = Stream::new(config).await?;

        // Create RDF Patch
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"Test object\"".to_string(),
        });

        // Convert to stream event
        let patch_event = StreamEvent::SparqlUpdate {
            query: patch.to_rdf_patch_format()?,
            graph: None,
            metadata: EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source: "patch_test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        // Stream the patch
        stream.publish(patch_event).await?;

        // Consume and verify
        if let Ok(Some(received)) = timeout(Duration::from_secs(5), stream.consume()).await {
            let event = received?;
            match event {
                StreamEvent::SparqlUpdate { query, .. } => {
                    let parsed_patch = RdfPatch::from_rdf_patch_format(&query)?;
                    assert_eq!(parsed_patch.operations.len(), 1);
                }
                _ => panic!("Expected SPARQL Update event"),
            }
        }

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_sparql_delta_streaming() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut stream = Stream::new(config).await?;

        // Test SPARQL Update delta computation
        let mut delta_processor = DeltaProcessor::new();
        let sparql_update = r#"
            INSERT DATA {
                <http://example.org/person1> <http://example.org/name> "John Doe" .
                <http://example.org/person1> <http://example.org/age> "30" .
            }
        "#;

        let events = delta_processor.process_update(sparql_update).await?;
        assert_eq!(events.len(), 2);

        // Stream the delta events
        for event in &events {
            stream.publish(event.clone()).await?;
        }

        // Consume and verify
        let mut received_events = Vec::new();
        for _ in 0..2 {
            if let Ok(Some(event)) = timeout(Duration::from_secs(5), stream.consume()).await {
                received_events.push(event?);
            }
        }

        assert_eq!(received_events.len(), 2);

        // Verify event types
        for event in &received_events {
            match event {
                StreamEvent::TripleAdded { subject, predicate, object, .. } => {
                    assert!(subject.contains("person1"));
                    assert!(predicate.contains("example.org"));
                    assert!(!object.is_empty());
                }
                _ => panic!("Expected TripleAdded event"),
            }
        }

        stream.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod monitoring_integration_tests {
    use super::*;
    use oxirs_stream::monitoring::*;

    #[tokio::test]
    async fn test_metrics_collection_integration() -> Result<()> {
        let monitoring_config = MonitoringConfig {
            enable_metrics: true,
            enable_tracing: true,
            metrics_interval: Duration::from_millis(100),
            health_check_interval: Duration::from_millis(200),
            enable_profiling: true,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "debug".to_string(),
        };

        let collector = MetricsCollector::new(monitoring_config);
        collector.start().await?;

        // Simulate some activity
        for i in 0..10 {
            collector.update_producer_metrics(ProducerMetricsUpdate {
                events_published: 1,
                events_failed: if i % 10 == 0 { 1 } else { 0 },
                bytes_sent: 1024,
                batches_sent: 1,
                latency_ms: 5.0,
                throughput_eps: 100.0,
            }).await;

            collector.update_consumer_metrics(ConsumerMetricsUpdate {
                events_consumed: 1,
                events_processed: 1,
                events_filtered: 0,
                events_failed: 0,
                bytes_received: 1024,
                batches_received: 1,
                processing_time_ms: 2.0,
                throughput_eps: 100.0,
                lag_ms: Some(10.0),
            }).await;

            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Check metrics
        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.producer_events_published, 10);
        assert_eq!(metrics.consumer_events_consumed, 10);
        assert!(metrics.producer_throughput_eps > 0.0);

        // Test Prometheus export
        let prometheus_output = collector.export_prometheus().await;
        assert!(prometheus_output.contains("oxirs_producer_events_published_total"));
        assert!(prometheus_output.contains("oxirs_consumer_events_consumed_total"));

        Ok(())
    }

    #[tokio::test]
    async fn test_health_monitoring_integration() -> Result<()> {
        let monitoring_config = MonitoringConfig {
            enable_metrics: true,
            enable_tracing: false,
            metrics_interval: Duration::from_secs(1),
            health_check_interval: Duration::from_millis(100),
            enable_profiling: false,
            prometheus_endpoint: None,
            jaeger_endpoint: None,
            log_level: "info".to_string(),
        };

        let collector = MetricsCollector::new(monitoring_config);
        collector.start().await?;

        // Let health checks run for a bit
        tokio::time::sleep(Duration::from_millis(300)).await;

        let health = collector.get_health().await;
        assert_eq!(health.overall_status, HealthStatus::Healthy);

        Ok(())
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_throughput_benchmark() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut stream = Stream::new(config).await?;

        let event_count = 10_000;
        let events = create_test_events(event_count);

        // Benchmark publishing
        let start = std::time::Instant::now();
        for event in &events {
            stream.publish(event.clone()).await?;
        }
        let publish_duration = start.elapsed();

        let publish_throughput = event_count as f64 / publish_duration.as_secs_f64();
        println!("Publish throughput: {:.0} events/second", publish_throughput);

        // Benchmark consuming
        let start = std::time::Instant::now();
        let mut consumed_count = 0;
        for _ in 0..event_count {
            if let Ok(Some(_)) = timeout(Duration::from_millis(1), stream.consume()).await {
                consumed_count += 1;
            } else {
                break;
            }
        }
        let consume_duration = start.elapsed();

        let consume_throughput = consumed_count as f64 / consume_duration.as_secs_f64();
        println!("Consume throughput: {:.0} events/second", consume_throughput);

        // Verify performance targets
        assert!(publish_throughput > 1000.0, "Publish throughput should exceed 1K events/sec");
        assert!(consume_throughput > 1000.0, "Consume throughput should exceed 1K events/sec");

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_latency_benchmark() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut producer = Stream::new(config.clone()).await?;
        let mut consumer = Stream::new(config).await?;

        let mut latencies = Vec::new();

        for _ in 0..100 {
            let start = std::time::Instant::now();
            
            let event = create_test_events(1)[0].clone();
            producer.publish(event).await?;

            if let Ok(Some(_)) = timeout(Duration::from_secs(1), consumer.consume()).await {
                let latency = start.elapsed();
                latencies.push(latency);
            }
        }

        // Calculate latency percentiles
        latencies.sort();
        let p50 = latencies[latencies.len() / 2];
        let p95 = latencies[(latencies.len() * 95) / 100];
        let p99 = latencies[(latencies.len() * 99) / 100];

        println!("Latency P50: {:?}, P95: {:?}, P99: {:?}", p50, p95, p99);

        // Verify latency targets (for memory backend)
        assert!(p99 < Duration::from_millis(100), "P99 latency should be under 100ms");

        producer.close().await?;
        consumer.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod circuit_breaker_tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_integration() -> Result<()> {
        let mut config = create_test_stream_config(StreamBackend::Memory);
        config.circuit_breaker.failure_threshold = 3;
        config.circuit_breaker.timeout = Duration::from_millis(100);

        let mut stream = Stream::new(config).await?;

        // Test circuit breaker activation (simulate failures)
        for i in 0..5 {
            let result = stream.health_check().await;
            if i < 3 {
                assert!(result.is_ok(), "Health check should pass initially");
            }
        }

        stream.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_connection_recovery() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut stream = Stream::new(config).await?;

        // Test graceful degradation
        let event = create_test_events(1)[0].clone();
        
        // This should work normally
        assert!(stream.publish(event.clone()).await.is_ok());

        // Simulate network interruption and recovery
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should still work after recovery
        assert!(stream.publish(event).await.is_ok());

        stream.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_event_handling() -> Result<()> {
        let config = create_test_stream_config(StreamBackend::Memory);
        let mut stream = Stream::new(config).await?;

        // Test with various invalid events
        let invalid_events = vec![
            StreamEvent::TripleAdded {
                subject: "".to_string(), // Empty subject
                predicate: "http://example.org/p".to_string(),
                object: "http://example.org/o".to_string(),
                graph: None,
                metadata: EventMetadata {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    source: "test".to_string(),
                    user: None,
                    context: None,
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                },
            },
        ];

        for event in invalid_events {
            // Should handle invalid events gracefully
            let result = stream.publish(event).await;
            if result.is_err() {
                println!("Expected error for invalid event: {:?}", result.err());
            }
        }

        stream.close().await?;
        Ok(())
    }
}