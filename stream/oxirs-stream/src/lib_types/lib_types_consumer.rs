//! Stream consumer types and implementation

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use crate::backend::{self, StreamBackend};
use crate::circuit_breaker::{self, SharedCircuitBreakerExt};
use crate::event::StreamEvent;

use super::lib_types_config::{StreamBackendType, StreamConfig};

/// Consumer statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct ConsumerStats {
    pub events_consumed: u64,
    pub events_failed: u64,
    pub _bytes_received: u64,
    pub avg_processing_time_ms: f64,
    pub max_processing_time_ms: u64,
    pub _consumer_lag: u64,
    pub circuit_breaker_trips: u64,
    pub last_message: Option<DateTime<Utc>>,
    pub backend_type: String,
    pub _batch_size: usize,
}

/// Backend-agnostic consumer wrapper (crate-private)
pub(crate) enum BackendConsumer {
    #[cfg(feature = "kafka")]
    Kafka(Box<backend::kafka::KafkaConsumer>),
    #[cfg(feature = "nats")]
    Nats(Box<backend::nats::NatsConsumer>),
    #[cfg(feature = "redis")]
    Redis(Box<backend::redis::RedisConsumer>),
    #[cfg(feature = "kinesis")]
    Kinesis(Box<backend::kinesis::KinesisConsumer>),
    #[cfg(feature = "pulsar")]
    Pulsar(Box<backend::pulsar::PulsarConsumer>),
    #[cfg(feature = "rabbitmq")]
    RabbitMQ(Box<backend::rabbitmq::RabbitMQConsumer>),
    Memory(Box<MemoryConsumer>),
}

/// Memory-based consumer for testing and development
pub(crate) struct MemoryConsumer {
    backend: Box<dyn StreamBackend + Send + Sync>,
    topic: String,
    current_offset: u64,
    stats: ConsumerStats,
}

impl MemoryConsumer {
    pub(crate) fn with_topic(topic: String) -> Self {
        Self {
            backend: Box::new(backend::memory::MemoryBackend::new()),
            topic,
            current_offset: 0,
            stats: ConsumerStats {
                backend_type: "memory".to_string(),
                ..Default::default()
            },
        }
    }

    pub(crate) async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        let start_time = Instant::now();

        self.backend
            .as_mut()
            .connect()
            .await
            .map_err(|e| anyhow!("Backend connect failed: {}", e))?;

        let topic_name = crate::types::TopicName::new(self.topic.clone());
        let events = self
            .backend
            .receive_events(
                &topic_name,
                None,
                crate::types::StreamPosition::Offset(self.current_offset),
                1,
            )
            .await
            .map_err(|e| anyhow!("Receive events failed: {}", e))?;

        if let Some((event, offset)) = events.first() {
            self.current_offset = offset.value() + 1;

            self.stats.events_consumed += 1;
            let processing_time = start_time.elapsed().as_millis() as u64;
            self.stats.max_processing_time_ms =
                self.stats.max_processing_time_ms.max(processing_time);
            self.stats.avg_processing_time_ms =
                (self.stats.avg_processing_time_ms + processing_time as f64) / 2.0;
            self.stats.last_message = Some(Utc::now());

            debug!("Memory consumer: consumed event via backend");
            Ok(Some(event.clone()))
        } else {
            debug!("Memory consumer: no events available");
            Ok(None)
        }
    }

    pub(crate) fn reset(&mut self) {
        self.current_offset = 0;
    }
}

/// Enhanced stream consumer for receiving RDF changes with backend support
pub struct StreamConsumer {
    pub(crate) _config: StreamConfig,
    pub(crate) backend_consumer: BackendConsumer,
    pub(crate) stats: Arc<RwLock<ConsumerStats>>,
    pub(crate) circuit_breaker: Option<circuit_breaker::SharedCircuitBreaker>,
    pub(crate) last_poll: Instant,
    pub(crate) _message_buffer: Arc<RwLock<Vec<StreamEvent>>>,
    pub(crate) consumer_group: Option<String>,
}

impl StreamConsumer {
    /// Create a new enhanced stream consumer with backend support
    pub async fn new(config: StreamConfig) -> Result<Self> {
        Self::new_with_group(config, None).await
    }

    /// Create a new stream consumer with a specific consumer group
    pub async fn new_with_group(
        config: StreamConfig,
        consumer_group: Option<String>,
    ) -> Result<Self> {
        let circuit_breaker = if config.circuit_breaker.enabled {
            Some(circuit_breaker::new_shared_circuit_breaker(
                circuit_breaker::CircuitBreakerConfig {
                    enabled: config.circuit_breaker.enabled,
                    failure_threshold: config.circuit_breaker.failure_threshold,
                    success_threshold: config.circuit_breaker.success_threshold,
                    timeout: config.circuit_breaker.timeout,
                    half_open_max_calls: config.circuit_breaker.half_open_max_calls,
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        let backend_consumer = Self::build_backend_consumer(&config).await?;

        let stats = Arc::new(RwLock::new(ConsumerStats {
            backend_type: match backend_consumer {
                #[cfg(feature = "kafka")]
                BackendConsumer::Kafka(_) => "kafka".to_string(),
                #[cfg(feature = "nats")]
                BackendConsumer::Nats(_) => "nats".to_string(),
                #[cfg(feature = "redis")]
                BackendConsumer::Redis(_) => "redis".to_string(),
                #[cfg(feature = "kinesis")]
                BackendConsumer::Kinesis(_) => "kinesis".to_string(),
                #[cfg(feature = "pulsar")]
                BackendConsumer::Pulsar(_) => "pulsar".to_string(),
                #[cfg(feature = "rabbitmq")]
                BackendConsumer::RabbitMQ(_) => "rabbitmq".to_string(),
                BackendConsumer::Memory(_) => "memory".to_string(),
            },
            _batch_size: config.batch_size,
            ..Default::default()
        }));

        info!(
            "Created stream consumer with backend: {} and group: {:?}",
            stats.read().await.backend_type,
            consumer_group
        );

        Ok(Self {
            _config: config,
            backend_consumer,
            stats,
            circuit_breaker,
            last_poll: Instant::now(),
            _message_buffer: Arc::new(RwLock::new(Vec::new())),
            consumer_group,
        })
    }

    async fn build_backend_consumer(config: &StreamConfig) -> Result<BackendConsumer> {
        match &config.backend {
            #[cfg(feature = "kafka")]
            StreamBackendType::Kafka {
                brokers,
                security_protocol,
                sasl_config,
            } => {
                let stream_config = Self::make_stream_config(
                    config,
                    crate::StreamBackendType::Kafka {
                        brokers: brokers.clone(),
                        security_protocol: security_protocol.clone(),
                        sasl_config: sasl_config.clone(),
                    },
                );
                let mut consumer = backend::kafka::KafkaConsumer::new(stream_config)?;
                consumer.connect().await?;
                Ok(BackendConsumer::Kafka(Box::new(consumer)))
            }
            #[cfg(feature = "nats")]
            StreamBackendType::Nats {
                url,
                cluster_urls,
                jetstream_config,
            } => {
                let stream_config = Self::make_stream_config(
                    config,
                    crate::StreamBackendType::Nats {
                        url: url.clone(),
                        cluster_urls: cluster_urls.clone(),
                        jetstream_config: jetstream_config.clone(),
                    },
                );
                let mut consumer = backend::nats::NatsConsumer::new(stream_config)?;
                consumer.connect().await?;
                Ok(BackendConsumer::Nats(Box::new(consumer)))
            }
            #[cfg(feature = "redis")]
            StreamBackendType::Redis {
                url,
                cluster_urls,
                pool_size,
            } => {
                let stream_config = Self::make_stream_config(
                    config,
                    crate::StreamBackendType::Redis {
                        url: url.clone(),
                        cluster_urls: cluster_urls.clone(),
                        pool_size: *pool_size,
                    },
                );
                let mut consumer = backend::redis::RedisConsumer::new(stream_config)?;
                consumer.connect().await?;
                Ok(BackendConsumer::Redis(Box::new(consumer)))
            }
            #[cfg(feature = "kinesis")]
            StreamBackendType::Kinesis {
                region,
                stream_name,
                credentials,
            } => {
                let stream_config = Self::make_stream_config(
                    config,
                    crate::StreamBackendType::Kinesis {
                        region: region.clone(),
                        stream_name: stream_name.clone(),
                        credentials: credentials.clone(),
                    },
                );
                let mut consumer = backend::kinesis::KinesisConsumer::new(stream_config)?;
                consumer.connect().await?;
                Ok(BackendConsumer::Kinesis(Box::new(consumer)))
            }
            #[cfg(feature = "pulsar")]
            StreamBackendType::Pulsar {
                service_url,
                auth_config,
            } => {
                let stream_config = Self::make_stream_config(
                    config,
                    crate::StreamBackendType::Pulsar {
                        service_url: service_url.clone(),
                        auth_config: auth_config.clone(),
                    },
                );
                let mut consumer = backend::pulsar::PulsarConsumer::new(stream_config)?;
                consumer.connect().await?;
                Ok(BackendConsumer::Pulsar(Box::new(consumer)))
            }
            #[cfg(feature = "rabbitmq")]
            StreamBackendType::RabbitMQ {
                url,
                exchange,
                queue,
            } => {
                let stream_config = Self::make_stream_config(
                    config,
                    crate::StreamBackendType::RabbitMQ {
                        url: url.clone(),
                        exchange: exchange.clone(),
                        queue: queue.clone(),
                    },
                );
                let mut consumer = backend::rabbitmq::RabbitMQConsumer::new(stream_config)?;
                consumer.connect().await?;
                Ok(BackendConsumer::RabbitMQ(Box::new(consumer)))
            }
            StreamBackendType::Memory {
                max_size: _,
                persistence: _,
            } => Ok(BackendConsumer::Memory(Box::new(
                MemoryConsumer::with_topic(config.topic.clone()),
            ))),
        }
    }

    fn make_stream_config(
        config: &StreamConfig,
        backend: crate::StreamBackendType,
    ) -> crate::StreamConfig {
        crate::StreamConfig {
            backend,
            topic: config.topic.clone(),
            batch_size: config.batch_size,
            flush_interval_ms: config.flush_interval_ms,
            max_connections: config.max_connections,
            connection_timeout: config.connection_timeout,
            enable_compression: config.enable_compression,
            compression_type: config.compression_type.clone(),
            retry_config: config.retry_config.clone(),
            circuit_breaker: config.circuit_breaker.clone(),
            security: config.security.clone(),
            performance: config.performance.clone(),
            monitoring: config.monitoring.clone(),
        }
    }

    /// Consume stream events with circuit breaker protection
    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        let start_time = Instant::now();

        if let Some(cb) = &self.circuit_breaker {
            if !cb.can_execute().await {
                self.stats.write().await.circuit_breaker_trips += 1;
                return Err(anyhow!("Circuit breaker is open - cannot consume events"));
            }
        }

        let result = self.consume_single_event().await;

        match &result {
            Ok(Some(_)) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_success_with_duration(start_time.elapsed()).await;
                }

                let mut stats = self.stats.write().await;
                stats.events_consumed += 1;
                let processing_time = start_time.elapsed().as_millis() as u64;
                stats.max_processing_time_ms = stats.max_processing_time_ms.max(processing_time);
                stats.avg_processing_time_ms =
                    (stats.avg_processing_time_ms + processing_time as f64) / 2.0;
                stats.last_message = Some(Utc::now());
            }
            Ok(None) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_success_with_duration(start_time.elapsed()).await;
                }
            }
            Err(_) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_failure_with_type(circuit_breaker::FailureType::NetworkError)
                        .await;
                }

                self.stats.write().await.events_failed += 1;
            }
        }

        self.last_poll = Instant::now();
        result
    }

    async fn consume_single_event(&mut self) -> Result<Option<StreamEvent>> {
        match &mut self.backend_consumer {
            #[cfg(feature = "kafka")]
            BackendConsumer::Kafka(consumer) => consumer.consume().await,
            #[cfg(feature = "nats")]
            BackendConsumer::Nats(consumer) => consumer.consume().await,
            #[cfg(feature = "redis")]
            BackendConsumer::Redis(consumer) => consumer.consume().await,
            #[cfg(feature = "kinesis")]
            BackendConsumer::Kinesis(consumer) => consumer.consume().await,
            #[cfg(feature = "pulsar")]
            BackendConsumer::Pulsar(consumer) => consumer.consume().await,
            #[cfg(feature = "rabbitmq")]
            BackendConsumer::RabbitMQ(consumer) => consumer.consume().await,
            BackendConsumer::Memory(consumer) => consumer.consume().await,
        }
    }

    /// Consume multiple events as a batch
    pub async fn consume_batch(
        &mut self,
        max_events: usize,
        timeout: Duration,
    ) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let start_time = Instant::now();

        while events.len() < max_events && start_time.elapsed() < timeout {
            match tokio::time::timeout(Duration::from_millis(50), self.consume()).await {
                Ok(Ok(Some(event))) => events.push(event),
                Ok(Ok(None)) => continue,
                Ok(Err(e)) => return Err(e),
                Err(_) => break,
            }
        }

        if !events.is_empty() {
            debug!(
                "Consumed batch of {} events in {:?}",
                events.len(),
                start_time.elapsed()
            );
        }

        Ok(events)
    }

    /// Start consuming events with a callback function
    pub async fn start_consuming<F>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(StreamEvent) -> Result<()> + Send,
    {
        info!("Starting stream consumer loop");

        loop {
            match self.consume().await {
                Ok(Some(event)) => {
                    if let Err(e) = callback(event) {
                        error!("Callback error: {}", e);
                        self.stats.write().await.events_failed += 1;
                    }
                }
                Ok(None) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(e) => {
                    error!("Consumer error: {}", e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Start consuming events with an async callback function
    pub async fn start_consuming_async<F, Fut>(&mut self, mut callback: F) -> Result<()>
    where
        F: FnMut(StreamEvent) -> Fut + Send,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        info!("Starting async stream consumer loop");

        loop {
            match self.consume().await {
                Ok(Some(event)) => {
                    if let Err(e) = callback(event).await {
                        error!("Async callback error: {}", e);
                        self.stats.write().await.events_failed += 1;
                    }
                }
                Ok(None) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(e) => {
                    error!("Consumer error: {}", e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Get consumer statistics
    pub async fn get_stats(&self) -> ConsumerStats {
        self.stats.read().await.clone()
    }

    /// Get consumer health status
    pub async fn health_check(&self) -> bool {
        if let Some(cb) = &self.circuit_breaker {
            cb.is_healthy().await
        } else {
            true
        }
    }

    /// Get the consumer group name if any
    pub fn consumer_group(&self) -> Option<&String> {
        self.consumer_group.as_ref()
    }

    /// Reset consumer position (for testing with memory backend)
    pub async fn reset_position(&mut self) -> Result<()> {
        match &mut self.backend_consumer {
            BackendConsumer::Memory(consumer) => {
                consumer.reset();
                Ok(())
            }
            #[cfg(feature = "kafka")]
            BackendConsumer::Kafka(_) => {
                Err(anyhow!("Reset position not supported for Kafka backend"))
            }
            #[cfg(feature = "nats")]
            BackendConsumer::Nats(_) => {
                Err(anyhow!("Reset position not supported for NATS backend"))
            }
            #[cfg(feature = "redis")]
            BackendConsumer::Redis(_) => {
                Err(anyhow!("Reset position not supported for Redis backend"))
            }
            #[cfg(feature = "kinesis")]
            BackendConsumer::Kinesis(_) => {
                Err(anyhow!("Reset position not supported for Kinesis backend"))
            }
            #[cfg(feature = "pulsar")]
            BackendConsumer::Pulsar(_) => {
                Err(anyhow!("Reset position not supported for Pulsar backend"))
            }
            #[cfg(feature = "rabbitmq")]
            BackendConsumer::RabbitMQ(_) => {
                Err(anyhow!("Reset position not supported for RabbitMQ backend"))
            }
        }
    }

    /// Set test events for memory backend (deprecated with backend implementation)
    pub async fn set_test_events(&mut self, _events: Vec<StreamEvent>) -> Result<()> {
        Err(anyhow!("set_test_events is deprecated with backend implementation - use producer to publish events"))
    }
}
