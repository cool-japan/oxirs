//! Stream producer types and implementation

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info};

use crate::backend::{self, StreamBackend};
use crate::circuit_breaker::{self, SharedCircuitBreakerExt};
use crate::event::StreamEvent;

use super::lib_types_config::{StreamBackendType, StreamConfig};

/// Producer statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct ProducerStats {
    pub events_published: u64,
    pub events_failed: u64,
    pub _bytes_sent: u64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: u64,
    pub batch_count: u64,
    pub flush_count: u64,
    pub circuit_breaker_trips: u64,
    pub last_publish: Option<DateTime<Utc>>,
    pub backend_type: String,
}

// Type aliases for complex types
type MemoryEventVec = Vec<(DateTime<Utc>, StreamEvent)>;
type MemoryEventStore = Arc<RwLock<MemoryEventVec>>;

// Global shared storage for memory backend events
static MEMORY_EVENTS: std::sync::OnceLock<MemoryEventStore> = std::sync::OnceLock::new();

pub fn get_memory_events() -> MemoryEventStore {
    MEMORY_EVENTS
        .get_or_init(|| Arc::new(RwLock::new(Vec::new())))
        .clone()
}

/// Clear the global memory storage (for testing)
pub async fn clear_memory_events() {
    let events = get_memory_events();
    events.write().await.clear();
    backend::memory::clear_memory_storage().await;
}

/// Backend-agnostic producer wrapper (crate-private)
pub(crate) enum BackendProducer {
    // Kafka/Pulsar variants removed: those backends were quarantined into the
    // publish=false oxirs-stream-adapter-{rdkafka,pulsar} crates (Pure Rust Policy v2).
    #[cfg(feature = "nats")]
    Nats(Box<backend::nats::NatsProducer>),
    #[cfg(feature = "redis")]
    Redis(backend::redis::RedisProducer),
    #[cfg(feature = "kinesis")]
    Kinesis(backend::kinesis::KinesisProducer),
    #[cfg(feature = "rabbitmq")]
    RabbitMQ(Box<backend::rabbitmq::RabbitMQProducer>),
    Memory(MemoryProducer),
}

/// Memory-based producer for testing and development
pub(crate) struct MemoryProducer {
    backend: Box<dyn StreamBackend + Send + Sync>,
    topic: String,
    stats: ProducerStats,
}

impl MemoryProducer {
    pub(crate) fn with_topic(topic: String) -> Self {
        Self {
            backend: Box::new(backend::memory::MemoryBackend::new()),
            topic,
            stats: ProducerStats {
                backend_type: "memory".to_string(),
                ..Default::default()
            },
        }
    }

    pub(crate) async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();

        self.backend
            .as_mut()
            .connect()
            .await
            .map_err(|e| anyhow!("Backend connect failed: {}", e))?;

        let topic_name = crate::types::TopicName::new(self.topic.clone());
        self.backend
            .create_topic(&topic_name, 1)
            .await
            .map_err(|e| anyhow!("Topic creation failed: {}", e))?;

        self.backend
            .send_event(&topic_name, event)
            .await
            .map_err(|e| anyhow!("Send event failed: {}", e))?;

        self.stats.events_published += 1;
        let latency = start_time.elapsed().as_millis() as u64;
        self.stats.max_latency_ms = self.stats.max_latency_ms.max(latency);
        self.stats.avg_latency_ms = (self.stats.avg_latency_ms + latency as f64) / 2.0;
        self.stats.last_publish = Some(Utc::now());

        debug!("Memory producer: published event via backend");
        Ok(())
    }

    pub(crate) async fn flush(&mut self) -> Result<()> {
        self.stats.flush_count += 1;
        debug!("Memory producer: flush completed");
        Ok(())
    }
}

/// Enhanced stream producer for publishing RDF changes with backend support
pub struct StreamProducer {
    pub(crate) config: StreamConfig,
    pub(crate) backend_producer: BackendProducer,
    pub(crate) stats: Arc<RwLock<ProducerStats>>,
    pub(crate) circuit_breaker: Option<circuit_breaker::SharedCircuitBreaker>,
    pub(crate) last_flush: Instant,
    pub(crate) _pending_events: Arc<RwLock<Vec<StreamEvent>>>,
    pub(crate) batch_buffer: Arc<RwLock<Vec<StreamEvent>>>,
    pub(crate) flush_semaphore: Arc<Semaphore>,
}

impl StreamProducer {
    /// Create a new enhanced stream producer with backend support
    pub async fn new(config: StreamConfig) -> Result<Self> {
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

        let backend_producer = Self::build_backend_producer(&config).await?;

        let stats = Arc::new(RwLock::new(ProducerStats {
            backend_type: match backend_producer {
                #[cfg(feature = "nats")]
                BackendProducer::Nats(_) => "nats".to_string(),
                #[cfg(feature = "redis")]
                BackendProducer::Redis(_) => "redis".to_string(),
                #[cfg(feature = "kinesis")]
                BackendProducer::Kinesis(_) => "kinesis".to_string(),
                #[cfg(feature = "rabbitmq")]
                BackendProducer::RabbitMQ(_) => "rabbitmq".to_string(),
                BackendProducer::Memory(_) => "memory".to_string(),
            },
            ..Default::default()
        }));

        info!(
            "Created stream producer with backend: {}",
            stats.read().await.backend_type
        );

        Ok(Self {
            config,
            backend_producer,
            stats,
            circuit_breaker,
            last_flush: Instant::now(),
            _pending_events: Arc::new(RwLock::new(Vec::new())),
            batch_buffer: Arc::new(RwLock::new(Vec::new())),
            flush_semaphore: Arc::new(Semaphore::new(1)),
        })
    }

    async fn build_backend_producer(config: &StreamConfig) -> Result<BackendProducer> {
        match &config.backend {
            // Kafka backend quarantined into the publish=false `oxirs-stream-adapter-rdkafka`
            // crate (Pure Rust Policy v2): build `KafkaBackend` there via the `StreamBackend` trait.
            StreamBackendType::Kafka { .. } => Err(anyhow!(
                "Kafka backend moved to the publish=false `oxirs-stream-adapter-rdkafka` crate \
                 (COOLJAPAN Pure Rust Policy v2). Construct `oxirs_stream_adapter_rdkafka::KafkaBackend` \
                 and drive it via the `oxirs_stream::backend::StreamBackend` trait."
            )),
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
                let mut producer = backend::nats::NatsProducer::new(stream_config)?;
                producer.connect().await?;
                Ok(BackendProducer::Nats(Box::new(producer)))
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
                let mut producer = backend::redis::RedisProducer::new(stream_config)?;
                producer.connect().await?;
                Ok(BackendProducer::Redis(producer))
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
                let mut producer = backend::kinesis::KinesisProducer::new(stream_config)?;
                producer.connect().await?;
                Ok(BackendProducer::Kinesis(producer))
            }
            // Pulsar backend quarantined into the publish=false `oxirs-stream-adapter-pulsar`
            // crate (Pure Rust Policy v2): construct PulsarProducer/PulsarConsumer there directly.
            StreamBackendType::Pulsar { .. } => Err(anyhow!(
                "Pulsar backend moved to the publish=false `oxirs-stream-adapter-pulsar` crate \
                 (COOLJAPAN Pure Rust Policy v2). Construct \
                 `oxirs_stream_adapter_pulsar::PulsarProducer` / `PulsarConsumer` directly."
            )),
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
                let mut producer = backend::rabbitmq::RabbitMQProducer::new(stream_config)?;
                producer.connect().await?;
                Ok(BackendProducer::RabbitMQ(Box::new(producer)))
            }
            StreamBackendType::Memory {
                max_size: _,
                persistence: _,
            } => Ok(BackendProducer::Memory(MemoryProducer::with_topic(
                config.topic.clone(),
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

    /// Publish a stream event with circuit breaker protection and batching
    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();

        if let Some(cb) = &self.circuit_breaker {
            if !cb.can_execute().await {
                self.stats.write().await.circuit_breaker_trips += 1;
                return Err(anyhow!("Circuit breaker is open - cannot publish events"));
            }
        }

        if self.config.performance.enable_batching {
            let mut batch_buffer = self.batch_buffer.write().await;
            batch_buffer.push(event);

            if batch_buffer.len() >= self.config.batch_size {
                let events = std::mem::take(&mut *batch_buffer);
                drop(batch_buffer);
                return self.publish_batch_internal(events).await;
            }

            return Ok(());
        }

        let result = self.publish_single_event(event).await;

        match &result {
            Ok(_) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_success_with_duration(start_time.elapsed()).await;
                }

                let mut stats = self.stats.write().await;
                stats.events_published += 1;
                let latency = start_time.elapsed().as_millis() as u64;
                stats.max_latency_ms = stats.max_latency_ms.max(latency);
                stats.avg_latency_ms = (stats.avg_latency_ms + latency as f64) / 2.0;
                stats.last_publish = Some(Utc::now());
            }
            Err(_) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_failure_with_type(circuit_breaker::FailureType::NetworkError)
                        .await;
                }

                self.stats.write().await.events_failed += 1;
            }
        }

        result
    }

    async fn publish_single_event(&mut self, event: StreamEvent) -> Result<()> {
        match &mut self.backend_producer {
            #[cfg(feature = "nats")]
            BackendProducer::Nats(producer) => producer.publish(event).await,
            #[cfg(feature = "redis")]
            BackendProducer::Redis(producer) => producer.publish(event).await,
            #[cfg(feature = "kinesis")]
            BackendProducer::Kinesis(producer) => producer.publish(event).await,
            #[cfg(feature = "rabbitmq")]
            BackendProducer::RabbitMQ(producer) => producer.publish(event).await,
            BackendProducer::Memory(producer) => producer.publish(event).await,
        }
    }

    /// Publish multiple events as a batch
    pub async fn publish_batch(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        self.publish_batch_internal(events).await
    }

    pub(crate) async fn publish_batch_internal(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        let start_time = Instant::now();
        let event_count = events.len();

        let result = match &mut self.backend_producer {
            #[cfg(feature = "nats")]
            BackendProducer::Nats(producer) => producer.publish_batch(events).await,
            #[cfg(feature = "redis")]
            BackendProducer::Redis(producer) => producer.publish_batch(events).await,
            #[cfg(feature = "kinesis")]
            BackendProducer::Kinesis(producer) => producer.publish_batch(events).await,
            #[cfg(feature = "rabbitmq")]
            BackendProducer::RabbitMQ(producer) => producer.publish_batch(events).await,
            BackendProducer::Memory(producer) => {
                for event in events {
                    producer.publish(event).await?;
                }
                Ok(())
            }
        };

        let mut stats = self.stats.write().await;
        match &result {
            Ok(_) => {
                stats.events_published += event_count as u64;
                stats.batch_count += 1;
                let latency = start_time.elapsed().as_millis() as u64;
                stats.max_latency_ms = stats.max_latency_ms.max(latency);
                stats.avg_latency_ms = (stats.avg_latency_ms + latency as f64) / 2.0;
                stats.last_publish = Some(Utc::now());
            }
            Err(_) => {
                stats.events_failed += event_count as u64;
            }
        }

        debug!(
            "Published batch of {} events in {:?}",
            event_count,
            start_time.elapsed()
        );
        result
    }

    /// Flush any pending events and buffers
    pub async fn flush(&mut self) -> Result<()> {
        let _permit = self
            .flush_semaphore
            .acquire()
            .await
            .map_err(|_| anyhow!("Failed to acquire flush semaphore"))?;

        let start_time = Instant::now();

        if self.config.performance.enable_batching {
            let events = {
                let mut batch_buffer = self.batch_buffer.write().await;
                if !batch_buffer.is_empty() {
                    std::mem::take(&mut *batch_buffer)
                } else {
                    Vec::new()
                }
            };
            if !events.is_empty() {
                drop(_permit);
                self.publish_batch_internal(events).await?;
            }
        }

        let result = match &mut self.backend_producer {
            #[cfg(feature = "nats")]
            BackendProducer::Nats(producer) => producer.flush().await,
            #[cfg(feature = "redis")]
            BackendProducer::Redis(producer) => producer.flush().await,
            #[cfg(feature = "kinesis")]
            BackendProducer::Kinesis(producer) => producer.flush().await,
            #[cfg(feature = "rabbitmq")]
            BackendProducer::RabbitMQ(producer) => producer.flush().await,
            BackendProducer::Memory(producer) => producer.flush().await,
        };

        if result.is_ok() {
            self.stats.write().await.flush_count += 1;
            self.last_flush = Instant::now();
            debug!("Flushed producer buffers in {:?}", start_time.elapsed());
        }

        result
    }

    /// Get producer statistics
    pub async fn get_stats(&self) -> ProducerStats {
        self.stats.read().await.clone()
    }

    /// Publish an RDF patch as a series of events
    pub async fn publish_patch(&mut self, patch: &super::lib_types_patch::RdfPatch) -> Result<()> {
        super::lib_types_patch::publish_patch(self, patch).await
    }

    /// Get producer health status
    pub async fn health_check(&self) -> bool {
        if let Some(cb) = &self.circuit_breaker {
            cb.is_healthy().await
        } else {
            true
        }
    }
}
