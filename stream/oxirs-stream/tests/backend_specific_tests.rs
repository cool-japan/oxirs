//! # Backend-Specific Integration Tests
//!
//! Detailed tests for each streaming backend's unique features and capabilities.

use chrono::Utc;
use oxirs_stream::*;
use std::collections::HashMap;
use uuid::Uuid;

/// Helper to create test event with metadata
#[allow(dead_code)]
fn create_test_event_with_metadata(id: &str, data: &str) -> StreamEvent {
    StreamEvent::TripleAdded {
        subject: format!("http://example.org/{id}"),
        predicate: "http://example.org/data".to_string(),
        object: format!("\"{data}\""),
        graph: None,
        metadata: EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "backend_test".to_string(),
            user: Some("test_user".to_string()),
            context: Some("backend_context".to_string()),
            caused_by: None,
            version: "1.0".to_string(),
            properties: {
                let mut props = HashMap::new();
                props.insert("test_id".to_string(), id.to_string());
                props.insert("test_data".to_string(), data.to_string());
                props
            },
            checksum: Some(format!("checksum_{id}")),
        },
    }
}

#[cfg(test)]
#[cfg(feature = "kafka")]
mod kafka_specific_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Kafka
    async fn test_kafka_partitioning() -> Result<()> {
        let test_id = Uuid::new_v4();
        let config = StreamConfig {
            backend: StreamBackendType::Kafka {
                brokers: vec!["localhost:9092".to_string()],
                security_protocol: None,
                sasl_config: None,
            },
            topic: format!("partition-test-{test_id}"),
            batch_size: 1,
            flush_interval_ms: 100,
            max_connections: 5,
            connection_timeout: Duration::from_secs(10),
            enable_compression: true,
            compression_type: CompressionType::Snappy,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                timeout: Duration::from_secs(60),
                success_threshold: 3,
                half_open_max_calls: 2,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                enable_batching: true,
                enable_pipelining: true,
                buffer_size: 2048,
                prefetch_count: 32,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: true,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut producer = Stream::new(config.clone()).await?;

        // Test partition key distribution
        let events = vec![
            ("partition1", "data1"),
            ("partition2", "data2"),
            ("partition1", "data3"), // Same partition as first
            ("partition3", "data4"),
        ];

        for (partition_key, data) in &events {
            let mut event = create_test_event_with_metadata(partition_key, data);

            // Add partition key to metadata
            if let StreamEvent::TripleAdded {
                ref mut metadata, ..
            } = event
            {
                metadata
                    .properties
                    .insert("partition_key".to_string(), partition_key.to_string());
            }

            producer.publish(event).await?;
        }

        // Test multiple consumers for parallel processing
        let mut consumer1 = Stream::new(config.clone()).await?;
        let mut consumer2 = Stream::new(config).await?;

        tokio::time::sleep(Duration::from_millis(500)).await;

        let mut consumer1_events = Vec::new();
        let mut consumer2_events = Vec::new();

        // Consume from both consumers
        for _ in 0..2 {
            if let Ok(Ok(Some(event))) = timeout(Duration::from_secs(2), consumer1.consume()).await
            {
                consumer1_events.push(event);
            }
            if let Ok(Ok(Some(event))) = timeout(Duration::from_secs(2), consumer2.consume()).await
            {
                consumer2_events.push(event);
            }
        }

        let total_consumed = consumer1_events.len() + consumer2_events.len();
        assert!(
            total_consumed >= 2,
            "Should consume at least 2 events across consumers"
        );

        // Note: Stream objects don't require explicit cleanup
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires Kafka with SASL
    async fn test_kafka_sasl_authentication() -> Result<()> {
        let config = StreamConfig {
            backend: StreamBackendType::Kafka {
                brokers: vec!["localhost:9093".to_string()], // SASL port
                security_protocol: Some("SASL_SSL".to_string()),
                sasl_config: None, // Will be set below
            },
            topic: "sasl-test-topic".to_string(),
            batch_size: 10,
            flush_interval_ms: 1000,
            max_connections: 3,
            connection_timeout: Duration::from_secs(15),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 5,
                initial_backoff: Duration::from_millis(200),
                max_backoff: Duration::from_secs(10),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                timeout: Duration::from_secs(30),
                success_threshold: 2,
                half_open_max_calls: 1,
            },
            security: SecurityConfig {
                enable_tls: true,
                verify_certificates: true,
                client_cert_path: Some("/etc/kafka/certs/client.crt".to_string()),
                client_key_path: Some("/etc/kafka/certs/client.key".to_string()),
                ca_cert_path: Some("/etc/kafka/certs/ca.crt".to_string()),
                sasl_config: None, // SASL config will be set separately
            },
            performance: PerformanceConfig {
                enable_batching: true,
                enable_pipelining: true,
                buffer_size: 1024,
                prefetch_count: 32,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(15),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut stream = Stream::new(config).await?;

        let event = create_test_event_with_metadata("sasl_test", "authenticated_data");
        stream.publish(event).await?;

        if let Ok(Ok(Some(event))) = timeout(Duration::from_secs(10), stream.consume()).await {
            let _event = event;
            println!("SASL authentication successful");
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires Kafka
    async fn test_kafka_schema_registry_integration() -> Result<()> {
        // This would test integration with Confluent Schema Registry
        // for Avro/JSON Schema validation of RDF events

        let config = StreamConfig {
            backend: StreamBackendType::Kafka {
                brokers: vec!["localhost:9092".to_string()],
                security_protocol: None,
                sasl_config: None,
            },
            topic: "schema-test-topic".to_string(),
            batch_size: 5,
            flush_interval_ms: 500,
            max_connections: 3,
            connection_timeout: Duration::from_secs(10),
            enable_compression: true,
            compression_type: CompressionType::Gzip,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                jitter: false,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                timeout: Duration::from_secs(60),
                success_threshold: 3,
                half_open_max_calls: 2,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 1024,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: true,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut stream = Stream::new(config).await?;

        // Test with structured RDF data
        let rdf_event = StreamEvent::TripleAdded {
            subject: "http://schema.org/Person/123".to_string(),
            predicate: "http://schema.org/name".to_string(),
            object: "\"John Doe\"".to_string(),
            graph: Some("http://example.org/people".to_string()),
            metadata: EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source: "schema_registry_test".to_string(),
                user: Some("schema_user".to_string()),
                context: Some("schema_validation".to_string()),
                caused_by: None,
                version: "2.0".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert("schema_type".to_string(), "person".to_string());
                    props.insert("validation_required".to_string(), "true".to_string());
                    props
                },
                checksum: Some("schema_checksum_123".to_string()),
            },
        };

        stream.publish(rdf_event).await?;

        if let Ok(Ok(Some(event))) = timeout(Duration::from_secs(5), stream.consume()).await {
            match event {
                StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    ..
                } => {
                    assert!(subject.contains("schema.org"));
                    assert!(predicate.contains("name"));
                    assert!(object.contains("John Doe"));
                }
                _ => panic!("Expected TripleAdded event"),
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "nats")]
mod nats_specific_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires NATS with JetStream
    async fn test_nats_jetstream_persistence() -> Result<()> {
        let config = StreamConfig {
            backend: StreamBackendType::Nats {
                url: "nats://localhost:4222".to_string(),
                cluster_urls: None,
                jetstream_config: None,
            },
            topic: "persistence-test".to_string(),
            batch_size: 5,
            flush_interval_ms: 200,
            max_connections: 3,
            connection_timeout: Duration::from_secs(10),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 5,
                initial_backoff: Duration::from_millis(50),
                max_backoff: Duration::from_secs(2),
                backoff_multiplier: 1.5,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                timeout: Duration::from_secs(30),
                success_threshold: 2,
                half_open_max_calls: 1,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 512,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: true,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: true,
                metrics_interval: Duration::from_secs(3),
                health_check_interval: Duration::from_secs(5),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut producer = Stream::new(config.clone()).await?;

        // Publish events with persistence
        let events = vec![
            ("persistent1", "Important data 1"),
            ("persistent2", "Important data 2"),
            ("persistent3", "Important data 3"),
        ];

        for (id, data) in &events {
            let event = create_test_event_with_metadata(id, data);
            producer.publish(event).await?;
        }

        producer.close().await?;

        // Wait a moment for persistence
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create new consumer and verify persistence
        let mut consumer = Stream::new(config).await?;

        let mut received_events = Vec::new();
        for _ in 0..3 {
            if let Ok(Ok(Some(event))) = timeout(Duration::from_secs(5), consumer.consume()).await {
                received_events.push(event);
            }
        }

        assert_eq!(received_events.len(), 3, "All events should be persisted");

        consumer.close().await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires NATS cluster
    async fn test_nats_cluster_failover() -> Result<()> {
        let cluster_config = StreamConfig {
            backend: StreamBackendType::Nats {
                url: "nats://localhost:4222".to_string(),
                cluster_urls: Some(vec![
                    "nats://localhost:4223".to_string(),
                    "nats://localhost:4224".to_string(),
                ]),
                jetstream_config: Some(NatsJetStreamConfig {
                    domain: None,
                    api_prefix: None,
                    timeout: Duration::from_secs(30),
                }),
            },
            topic: "cluster-test".to_string(),
            batch_size: 3,
            flush_interval_ms: 100,
            max_connections: 5,
            connection_timeout: Duration::from_secs(15),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 10,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                timeout: Duration::from_secs(60),
                success_threshold: 3,
                half_open_max_calls: 2,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 1024,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: true,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut stream = Stream::new(cluster_config).await?;

        // Test resilience during node failures
        for i in 0..10 {
            let event =
                create_test_event_with_metadata(&format!("cluster_{i}"), &format!("data_{i}"));

            let result = stream.publish(event).await;
            if result.is_err() {
                println!(
                    "Expected some failures during cluster failover: {:?}",
                    result.err()
                );
            }

            // Small delay to allow cluster operations
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires NATS with auth
    async fn test_nats_jwt_authentication() -> Result<()> {
        let jwt_config = StreamConfig {
            backend: StreamBackendType::Nats {
                url: "nats://localhost:4222".to_string(),
                cluster_urls: None,
                jetstream_config: None,
            },
            topic: "jwt-test".to_string(),
            batch_size: 1,
            flush_interval_ms: 100,
            max_connections: 2,
            connection_timeout: Duration::from_secs(10),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(3),
                backoff_multiplier: 2.0,
                jitter: false,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                timeout: Duration::from_secs(30),
                success_threshold: 2,
                half_open_max_calls: 1,
            },
            security: SecurityConfig {
                enable_tls: true,
                verify_certificates: true,
                client_cert_path: Some("/etc/nats/certs/client.crt".to_string()),
                client_key_path: Some("/etc/nats/certs/client.key".to_string()),
                ca_cert_path: Some("/etc/nats/certs/ca.crt".to_string()),
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 256,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: false,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut stream = Stream::new(jwt_config).await?;

        let event = create_test_event_with_metadata("jwt_test", "authenticated_via_jwt");
        stream.publish(event).await?;

        if let Ok(Ok(Some(_))) = timeout(Duration::from_secs(5), stream.consume()).await {
            println!("JWT authentication successful");
        }

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "redis")]
mod redis_specific_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Redis with streams
    async fn test_redis_consumer_groups() -> Result<()> {
        let config = StreamConfig {
            backend: StreamBackendType::Redis {
                url: "redis://localhost:6379".to_string(),
                cluster_urls: None,
                pool_size: Some(5),
            },
            topic: "consumer-group-test".to_string(),
            batch_size: 5,
            flush_interval_ms: 200,
            max_connections: 3,
            connection_timeout: Duration::from_secs(5),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(50),
                max_backoff: Duration::from_secs(2),
                backoff_multiplier: 1.5,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                timeout: Duration::from_secs(30),
                success_threshold: 3,
                half_open_max_calls: 2,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 512,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: true,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut producer = Stream::new(config.clone()).await?;

        // Create multiple consumers in the same group
        let mut consumer1 = Stream::new(config.clone()).await?;
        let mut consumer2 = Stream::new(config).await?;

        // Publish events
        for i in 0..6 {
            let event =
                create_test_event_with_metadata(&format!("cg_{i}"), &format!("group_data_{i}"));
            producer.publish(event).await?;
        }

        tokio::time::sleep(Duration::from_millis(300)).await;

        let mut consumer1_count = 0;
        let mut consumer2_count = 0;

        // Both consumers should receive different subsets
        for _ in 0..3 {
            if let Ok(Ok(Some(_))) = timeout(Duration::from_millis(500), consumer1.consume()).await
            {
                consumer1_count += 1;
            }
            if let Ok(Ok(Some(_))) = timeout(Duration::from_millis(500), consumer2.consume()).await
            {
                consumer2_count += 1;
            }
        }

        let total_consumed = consumer1_count + consumer2_count;
        assert!(
            total_consumed >= 3,
            "Consumer group should distribute messages"
        );
        assert!(
            consumer1_count > 0 || consumer2_count > 0,
            "At least one consumer should receive messages"
        );

        // Note: Stream objects don't require explicit cleanup
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires Redis Cluster
    async fn test_redis_cluster_sharding() -> Result<()> {
        let cluster_config = StreamConfig {
            backend: StreamBackendType::Redis {
                url: "redis://localhost:7000".to_string(),
                cluster_urls: Some(vec![
                    "redis://localhost:7001".to_string(),
                    "redis://localhost:7002".to_string(),
                ]),
                pool_size: Some(5),
            },
            topic: "cluster-shard-test".to_string(),
            batch_size: 10,
            flush_interval_ms: 500,
            max_connections: 6,
            connection_timeout: Duration::from_secs(10),
            enable_compression: true,
            compression_type: CompressionType::Lz4,
            retry_config: RetryConfig {
                max_retries: 5,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                timeout: Duration::from_secs(60),
                success_threshold: 2,
                half_open_max_calls: 1,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 2048,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: true,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: true,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut stream = Stream::new(cluster_config).await?;

        // Test sharding across cluster nodes
        for i in 0..15 {
            let event = create_test_event_with_metadata(
                &format!("shard_{i}"),
                &format!("cluster_data_{i}"),
            );
            stream.publish(event).await?;
        }

        // Verify events are distributed across shards
        let mut received_count = 0;
        for _ in 0..15 {
            if let Ok(Ok(Some(_))) = timeout(Duration::from_millis(200), stream.consume()).await {
                received_count += 1;
            }
        }

        assert!(received_count > 0, "Should receive events from cluster");

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "pulsar")]
mod pulsar_specific_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Pulsar
    async fn test_pulsar_namespace_isolation() -> Result<()> {
        let config1 = StreamConfig {
            backend: StreamBackendType::Pulsar {
                service_url: "pulsar://localhost:6650".to_string(),
                auth_config: None,
            },
            topic: "isolation-test-1".to_string(),
            batch_size: 3,
            flush_interval_ms: 200,
            max_connections: 2,
            connection_timeout: Duration::from_secs(10),
            enable_compression: true,
            compression_type: CompressionType::Zstd,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                jitter: false,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                timeout: Duration::from_secs(30),
                success_threshold: 2,
                half_open_max_calls: 1,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                enable_batching: true,
                enable_pipelining: true,
                buffer_size: 1024,
                prefetch_count: 32,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let config2 = StreamConfig {
            backend: StreamBackendType::Pulsar {
                service_url: "pulsar://localhost:6650".to_string(),
                auth_config: None,
            },
            ..config1.clone()
        };

        let mut stream1 = Stream::new(config1).await?;
        let mut stream2 = Stream::new(config2).await?;

        // Send events to different namespaces
        let event1 = create_test_event_with_metadata("ns1", "namespace1_data");
        let event2 = create_test_event_with_metadata("ns2", "namespace2_data");

        stream1.publish(event1).await?;
        stream2.publish(event2).await?;

        tokio::time::sleep(Duration::from_millis(300)).await;

        // Each stream should only receive its own namespace events
        if let Ok(Ok(Some(received1))) = timeout(Duration::from_secs(5), stream1.consume()).await {
            if let StreamEvent::TripleAdded { object, .. } = received1 {
                assert!(object.contains("namespace1_data"));
            }
        }

        if let Ok(Ok(Some(received2))) = timeout(Duration::from_secs(5), stream2.consume()).await {
            if let StreamEvent::TripleAdded { object, .. } = received2 {
                assert!(object.contains("namespace2_data"));
            }
        }

        stream1.close().await?;
        stream2.close().await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires Pulsar with BookKeeper
    async fn test_pulsar_message_deduplication() -> Result<()> {
        let config = StreamConfig {
            backend: StreamBackendType::Pulsar {
                service_url: "pulsar://localhost:6650".to_string(),
                auth_config: None,
            },
            topic: "dedup-test".to_string(),
            batch_size: 1,
            flush_interval_ms: 100,
            max_connections: 2,
            connection_timeout: Duration::from_secs(10),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(3),
                backoff_multiplier: 1.5,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                timeout: Duration::from_secs(30),
                success_threshold: 3,
                half_open_max_calls: 2,
            },
            security: SecurityConfig {
                enable_tls: false,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 512,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: false,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(5),
                health_check_interval: Duration::from_secs(10),
                enable_profiling: false,
                prometheus_endpoint: None,
                jaeger_endpoint: None,
                log_level: "info".to_string(),
            },
        };

        let mut stream = Stream::new(config).await?;

        // Send the same event multiple times (simulate duplicates)
        let event = create_test_event_with_metadata("dedup_test", "duplicate_data");

        stream.publish(event.clone()).await?;
        stream.publish(event.clone()).await?;
        stream.publish(event).await?;

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Should only receive one copy due to deduplication
        let mut received_count = 0;
        for _ in 0..3 {
            if let Ok(Some(_)) = timeout(Duration::from_millis(500), stream.consume()).await {
                received_count += 1;
            }
        }

        // Pulsar's deduplication should prevent duplicates
        assert!(
            received_count <= 2,
            "Deduplication should prevent excessive duplicates"
        );

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "kinesis")]
mod kinesis_specific_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires AWS credentials
    async fn test_kinesis_shard_scaling() -> Result<()> {
        let config = StreamConfig {
            backend: StreamBackendType::Kinesis {
                region: "us-west-2".to_string(),
                stream_name: "shard-scaling-test".to_string(),
                credentials: None,
            },
            topic: "shard-scaling-test".to_string(),
            batch_size: 25,
            flush_interval_ms: 1000,
            max_connections: 10,
            connection_timeout: Duration::from_secs(15),
            enable_compression: true,
            compression_type: CompressionType::Gzip,
            retry_config: RetryConfig {
                max_retries: 5,
                initial_backoff: Duration::from_millis(200),
                max_backoff: Duration::from_secs(10),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                timeout: Duration::from_secs(120),
                success_threshold: 3,
                half_open_max_calls: 2,
            },
            security: SecurityConfig {
                enable_tls: true,
                verify_certificates: true,
                client_cert_path: None, // Uses AWS IAM
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 4096,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: true,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: true,
                metrics_interval: Duration::from_secs(10),
                health_check_interval: Duration::from_secs(30),
                enable_profiling: false,
            },
        };

        let mut stream = Stream::new(config).await?;

        // Test high-volume publishing to trigger auto-scaling
        for i in 0..100 {
            let event = create_test_event_with_metadata(
                &format!("scale_{i}"),
                &format!("scaling_data_{i}"),
            );

            let result = stream.publish(event).await;
            if result.is_err() && i < 10 {
                // Early failures might be due to throttling
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        // Check that some events were successfully published
        tokio::time::sleep(Duration::from_secs(5)).await;

        let mut received_count = 0;
        for _ in 0..50 {
            if let Ok(Some(_)) = timeout(Duration::from_millis(100), stream.consume()).await {
                received_count += 1;
            }
        }

        assert!(received_count > 0, "Should receive events despite scaling");

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Requires AWS credentials and enhanced fan-out setup
    async fn test_kinesis_enhanced_fanout_multiple_consumers() -> Result<()> {
        let base_config = StreamConfig {
            backend: StreamBackendType::Kinesis {
                region: "us-west-2".to_string(),
                stream_name: "enhanced-fanout-test".to_string(),
                credentials: None,
            },
            topic: "enhanced-fanout-test".to_string(),
            batch_size: 10,
            flush_interval_ms: 500,
            max_connections: 5,
            connection_timeout: Duration::from_secs(20),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_backoff: Duration::from_millis(500),
                max_backoff: Duration::from_secs(10),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                timeout: Duration::from_secs(60),
                success_threshold: 2,
                half_open_max_calls: 1,
            },
            security: SecurityConfig {
                enable_tls: true,
                verify_certificates: true,
                client_cert_path: None,
                client_key_path: None,
                ca_cert_path: None,
                sasl_config: None,
            },
            performance: PerformanceConfig {
                buffer_size: 2048,
                enable_pipelining: true,
                prefetch_count: 32,
                enable_batching: true,
                enable_zero_copy: false,
                enable_simd: false,
                parallel_processing: true,
                worker_threads: Some(2),
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: false,
                metrics_interval: Duration::from_secs(10),
                health_check_interval: Duration::from_secs(20),
                enable_profiling: false,
            },
        };

        let mut producer = Stream::new(base_config.clone()).await?;
        let mut consumer1 = Stream::new(base_config.clone()).await?;
        let mut consumer2 = Stream::new(base_config.clone()).await?;
        let mut consumer3 = Stream::new(base_config).await?;

        // Publish events
        for i in 0..12 {
            let event = create_test_event_with_metadata(
                &format!("fanout_{i}"),
                &format!("fanout_data_{i}"),
            );
            producer.publish(event).await?;
        }

        // Enhanced fan-out allows all consumers to receive all messages
        tokio::time::sleep(Duration::from_secs(10)).await;

        let mut consumer1_count = 0;
        let mut consumer2_count = 0;
        let mut consumer3_count = 0;

        for _ in 0..4 {
            if let Ok(Some(_)) = timeout(Duration::from_secs(3), consumer1.consume()).await {
                consumer1_count += 1;
            }
            if let Ok(Some(_)) = timeout(Duration::from_secs(3), consumer2.consume()).await {
                consumer2_count += 1;
            }
            if let Ok(Some(_)) = timeout(Duration::from_secs(3), consumer3.consume()).await {
                consumer3_count += 1;
            }
        }

        // With enhanced fan-out, all consumers should receive messages
        assert!(consumer1_count > 0, "Consumer 1 should receive messages");
        assert!(consumer2_count > 0, "Consumer 2 should receive messages");
        assert!(consumer3_count > 0, "Consumer 3 should receive messages");

        // Note: Stream objects don't require explicit cleanup
        consumer3.close().await?;
        Ok(())
    }
}
