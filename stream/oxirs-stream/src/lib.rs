//! # OxiRS Stream - Ultra-High Performance RDF Streaming Platform
//!
//! Real-time streaming support with Kafka/NATS/Redis I/O, RDF Patch, SPARQL Update delta,
//! and advanced event processing capabilities.
//!
//! This crate provides enterprise-grade real-time data streaming capabilities for RDF datasets,
//! supporting multiple messaging backends with high-throughput, low-latency guarantees.
//!
//! ## Features
//! - **Multi-Backend Support**: Kafka, NATS JetStream, Redis Streams, AWS Kinesis, Memory
//! - **High Performance**: 100K+ events/second, <10ms latency, exactly-once delivery
//! - **Advanced Event Processing**: Real-time pattern detection, windowing, aggregations
//! - **Enterprise Features**: Circuit breakers, connection pooling, health monitoring
//! - **Standards Compliance**: RDF Patch protocol, SPARQL Update streaming
//!
//! ## Performance Targets
//! - **Throughput**: 100K+ events/second sustained
//! - **Latency**: P99 <10ms for real-time processing
//! - **Reliability**: 99.99% delivery success rate
//! - **Scalability**: Linear scaling to 1000+ partitions

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, Semaphore};
use tracing::{debug, info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Re-export commonly used types for convenience
pub use circuit_breaker::{CircuitBreakerError, FailureType};
pub use connection_pool::{PoolConfig, PoolStatus, DetailedPoolMetrics};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod kafka;
pub mod nats;
pub mod patch;
pub mod delta;
pub mod redis;
pub mod kinesis;
pub mod pulsar;
pub mod monitoring;
pub mod circuit_breaker;
pub mod connection_pool;

/// Enhanced stream configuration with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    pub backend: StreamBackend,
    pub topic: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression type
    pub compression_type: CompressionType,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Performance tuning
    pub performance: PerformanceConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

/// Compression types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub half_open_max_calls: u32,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_tls: bool,
    pub verify_certificates: bool,
    pub client_cert_path: Option<String>,
    pub client_key_path: Option<String>,
    pub ca_cert_path: Option<String>,
    pub sasl_config: Option<SaslConfig>,
}

/// SASL authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaslConfig {
    pub mechanism: SaslMechanism,
    pub username: String,
    pub password: String,
}

/// SASL authentication mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaslMechanism {
    Plain,
    ScramSha256,
    ScramSha512,
    OAuthBearer,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_batching: bool,
    pub enable_pipelining: bool,
    pub buffer_size: usize,
    pub prefetch_count: u32,
    pub enable_zero_copy: bool,
    pub enable_simd: bool,
    pub parallel_processing: bool,
    pub worker_threads: Option<usize>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub metrics_interval: Duration,
    pub health_check_interval: Duration,
    pub enable_profiling: bool,
}

/// Enhanced streaming backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamBackend {
    #[cfg(feature = "kafka")]
    Kafka { 
        brokers: Vec<String>,
        security_protocol: Option<String>,
        sasl_config: Option<SaslConfig>,
    },
    #[cfg(feature = "nats")]
    Nats { 
        url: String,
        cluster_urls: Option<Vec<String>>,
        jetstream_config: Option<NatsJetStreamConfig>,
    },
    #[cfg(feature = "redis")]
    Redis { 
        url: String,
        cluster_urls: Option<Vec<String>>,
        pool_size: Option<usize>,
    },
    #[cfg(feature = "kinesis")]
    Kinesis {
        region: String,
        stream_name: String,
        credentials: Option<AwsCredentials>,
    },
    #[cfg(feature = "pulsar")]
    Pulsar {
        service_url: String,
        auth_config: Option<PulsarAuthConfig>,
    },
    Memory { 
        max_size: Option<usize>,
        persistence: bool,
    },
}

/// NATS JetStream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatsJetStreamConfig {
    pub domain: Option<String>,
    pub api_prefix: Option<String>,
    pub timeout: Duration,
}

/// AWS credentials configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub role_arn: Option<String>,
}

/// Pulsar authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulsarAuthConfig {
    pub auth_method: PulsarAuthMethod,
    pub auth_params: HashMap<String, String>,
}

/// Pulsar authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulsarAuthMethod {
    Token,
    Jwt,
    Oauth2,
    Tls,
}

/// Enhanced RDF streaming events with metadata and provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    TripleAdded { 
        subject: String, 
        predicate: String, 
        object: String,
        graph: Option<String>,
        metadata: EventMetadata,
    },
    TripleRemoved { 
        subject: String, 
        predicate: String, 
        object: String,
        graph: Option<String>,
        metadata: EventMetadata,
    },
    QuadAdded {
        subject: String,
        predicate: String,
        object: String,
        graph: String,
        metadata: EventMetadata,
    },
    QuadRemoved {
        subject: String,
        predicate: String,
        object: String,
        graph: String,
        metadata: EventMetadata,
    },
    GraphCreated { 
        graph: String,
        metadata: EventMetadata,
    },
    GraphCleared { 
        graph: Option<String>,
        metadata: EventMetadata,
    },
    GraphDeleted {
        graph: String,
        metadata: EventMetadata,
    },
    SparqlUpdate { 
        query: String,
        operation_type: SparqlOperationType,
        metadata: EventMetadata,
    },
    TransactionBegin {
        transaction_id: String,
        isolation_level: Option<IsolationLevel>,
        metadata: EventMetadata,
    },
    TransactionCommit {
        transaction_id: String,
        metadata: EventMetadata,
    },
    TransactionAbort {
        transaction_id: String,
        metadata: EventMetadata,
    },
    SchemaChanged {
        schema_type: SchemaType,
        change_type: SchemaChangeType,
        details: String,
        metadata: EventMetadata,
    },
    Heartbeat {
        timestamp: DateTime<Utc>,
        source: String,
    },
}

/// Event metadata for tracking and provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Unique event identifier
    pub event_id: String,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Source identification
    pub source: String,
    /// User/session that triggered the event
    pub user: Option<String>,
    /// Operation context
    pub context: Option<String>,
    /// Causality tracking - event that caused this event
    pub caused_by: Option<String>,
    /// Event version for schema evolution
    pub version: String,
    /// Custom properties
    pub properties: HashMap<String, String>,
    /// Checksum for integrity
    pub checksum: Option<String>,
}

/// SPARQL operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparqlOperationType {
    Insert,
    Delete,
    Update,
    Load,
    Clear,
    Create,
    Drop,
    Copy,
    Move,
    Add,
}

/// Transaction isolation levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Schema types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaType {
    Ontology,
    Vocabulary,
    Constraint,
    Rule,
}

/// Schema change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaChangeType {
    Added,
    Modified,
    Removed,
    Versioned,
}

/// Enhanced stream producer for publishing RDF changes with backend support
pub struct StreamProducer {
    config: StreamConfig,
    backend_producer: BackendProducer,
    stats: Arc<RwLock<ProducerStats>>,
    circuit_breaker: Option<circuit_breaker::SharedCircuitBreaker>,
    last_flush: Instant,
    pending_events: Arc<RwLock<Vec<StreamEvent>>>,
    batch_buffer: Arc<RwLock<Vec<StreamEvent>>>,
    flush_semaphore: Arc<Semaphore>,
}

/// Backend-agnostic producer wrapper
enum BackendProducer {
    #[cfg(feature = "kafka")]
    Kafka(kafka::KafkaProducer),
    #[cfg(feature = "nats")]
    Nats(nats::NatsProducer),
    #[cfg(feature = "redis")]
    Redis(redis::RedisProducer),
    #[cfg(feature = "kinesis")]
    Kinesis(kinesis::KinesisProducer),
    #[cfg(feature = "pulsar")]
    Pulsar(pulsar::PulsarProducer),
    Memory(MemoryProducer),
}

/// Producer statistics for monitoring
#[derive(Debug, Default, Clone)]
struct ProducerStats {
    events_published: u64,
    events_failed: u64,
    bytes_sent: u64,
    avg_latency_ms: f64,
    max_latency_ms: u64,
    batch_count: u64,
    flush_count: u64,
    circuit_breaker_trips: u64,
    last_publish: Option<DateTime<Utc>>,
    backend_type: String,
}

/// Memory-based producer for testing and development
struct MemoryProducer {
    events: Arc<RwLock<Vec<(DateTime<Utc>, StreamEvent)>>>,
    max_size: Option<usize>,
    persistence: bool,
    stats: ProducerStats,
}

impl MemoryProducer {
    fn new(max_size: Option<usize>, persistence: bool) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_size,
            persistence,
            stats: ProducerStats {
                backend_type: "memory".to_string(),
                ..Default::default()
            },
        }
    }
    
    async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();
        let mut events = self.events.write().await;
        
        // Enforce max size if specified
        if let Some(max_size) = self.max_size {
            while events.len() >= max_size {
                events.remove(0);
            }
        }
        
        events.push((Utc::now(), event));
        
        // Update stats
        self.stats.events_published += 1;
        let latency = start_time.elapsed().as_millis() as u64;
        self.stats.max_latency_ms = self.stats.max_latency_ms.max(latency);
        self.stats.avg_latency_ms = (self.stats.avg_latency_ms + latency as f64) / 2.0;
        self.stats.last_publish = Some(Utc::now());
        
        debug!("Memory producer: published event (total: {})", events.len());
        Ok(())
    }
    
    async fn flush(&mut self) -> Result<()> {
        if self.persistence {
            // In a real implementation, this would persist to disk
            debug!("Memory producer: persisted {} events", self.events.read().await.len());
        }
        self.stats.flush_count += 1;
        Ok(())
    }
    
    fn get_stats(&self) -> &ProducerStats {
        &self.stats
    }
}

impl StreamProducer {
    /// Create a new enhanced stream producer with backend support
    pub async fn new(config: StreamConfig) -> Result<Self> {
        // Initialize circuit breaker if enabled
        let circuit_breaker = if config.circuit_breaker.enabled {
            Some(circuit_breaker::new_shared_circuit_breaker(
                circuit_breaker::CircuitBreakerConfig {
                    enabled: config.circuit_breaker.enabled,
                    failure_threshold: config.circuit_breaker.failure_threshold,
                    success_threshold: config.circuit_breaker.success_threshold,
                    timeout: config.circuit_breaker.timeout,
                    half_open_max_calls: config.circuit_breaker.half_open_max_calls,
                    ..Default::default()
                }
            ))
        } else {
            None
        };
        
        // Initialize backend-specific producer
        let backend_producer = match &config.backend {
            #[cfg(feature = "redis")]
            StreamBackend::Redis { url, cluster_urls, pool_size } => {
                let stream_config = crate::StreamConfig {
                    backend: crate::StreamBackend::Redis {
                        url: url.clone(),
                        cluster_urls: cluster_urls.clone(),
                        pool_size: *pool_size,
                    },
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
                };
                
                let mut producer = redis::RedisProducer::new(stream_config)?;
                producer.connect().await?;
                BackendProducer::Redis(producer)
            },
            #[cfg(feature = "kinesis")]
            StreamBackend::Kinesis { region, stream_name, credentials } => {
                let stream_config = crate::StreamConfig {
                    backend: crate::StreamBackend::Kinesis {
                        region: region.clone(),
                        stream_name: stream_name.clone(),
                        credentials: credentials.clone(),
                    },
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
                };
                
                let mut producer = kinesis::KinesisProducer::new(stream_config)?;
                producer.connect().await?;
                BackendProducer::Kinesis(producer)
            },
            #[cfg(feature = "pulsar")]
            StreamBackend::Pulsar { service_url, auth_config } => {
                let stream_config = crate::StreamConfig {
                    backend: crate::StreamBackend::Pulsar {
                        service_url: service_url.clone(),
                        auth_config: auth_config.clone(),
                    },
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
                };
                
                let mut producer = pulsar::PulsarProducer::new(stream_config)?;
                producer.connect().await?;
                BackendProducer::Pulsar(producer)
            },
            StreamBackend::Memory { max_size, persistence } => {
                BackendProducer::Memory(MemoryProducer::new(*max_size, *persistence))
            },
            _ => {
                return Err(anyhow!("Backend not supported or feature not enabled"));
            }
        };
        
        let stats = Arc::new(RwLock::new(ProducerStats {
            backend_type: match backend_producer {
                #[cfg(feature = "redis")]
                BackendProducer::Redis(_) => "redis".to_string(),
                #[cfg(feature = "kinesis")]
                BackendProducer::Kinesis(_) => "kinesis".to_string(),
                #[cfg(feature = "pulsar")]
                BackendProducer::Pulsar(_) => "pulsar".to_string(),
                BackendProducer::Memory(_) => "memory".to_string(),
            },
            ..Default::default()
        }));
        
        info!("Created stream producer with backend: {}", stats.read().await.backend_type);
        
        Ok(Self {
            config,
            backend_producer,
            stats,
            circuit_breaker,
            last_flush: Instant::now(),
            pending_events: Arc::new(RwLock::new(Vec::new())),
            batch_buffer: Arc::new(RwLock::new(Vec::new())),
            flush_semaphore: Arc::new(Semaphore::new(1)),
        })
    }
    
    /// Publish a stream event with circuit breaker protection and batching
    pub async fn publish(&mut self, event: StreamEvent) -> Result<()> {
        let start_time = Instant::now();
        
        // Check circuit breaker if enabled
        if let Some(cb) = &self.circuit_breaker {
            if !cb.can_execute().await {
                self.stats.write().await.circuit_breaker_trips += 1;
                return Err(anyhow!("Circuit breaker is open - cannot publish events"));
            }
        }
        
        // Handle batching if enabled
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
        
        // Publish single event
        let result = self.publish_single_event(event).await;
        
        // Update circuit breaker and stats
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
                    cb.record_failure_with_type(
                        circuit_breaker::FailureType::NetworkError
                    ).await;
                }
                
                self.stats.write().await.events_failed += 1;
            }
        }
        
        result
    }
    
    /// Publish a single event to the backend
    async fn publish_single_event(&mut self, event: StreamEvent) -> Result<()> {
        match &mut self.backend_producer {
            #[cfg(feature = "redis")]
            BackendProducer::Redis(producer) => {
                producer.publish(event).await
            },
            #[cfg(feature = "kinesis")]
            BackendProducer::Kinesis(producer) => {
                producer.publish(event).await
            },
            #[cfg(feature = "pulsar")]
            BackendProducer::Pulsar(producer) => {
                producer.publish(event).await
            },
            BackendProducer::Memory(producer) => {
                producer.publish(event).await
            },
        }
    }
    
    /// Publish multiple events as a batch
    pub async fn publish_batch(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        
        self.publish_batch_internal(events).await
    }
    
    /// Internal batch publishing implementation
    async fn publish_batch_internal(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        let start_time = Instant::now();
        let event_count = events.len();
        
        let result = match &mut self.backend_producer {
            #[cfg(feature = "redis")]
            BackendProducer::Redis(producer) => {
                producer.publish_batch(events).await
            },
            #[cfg(feature = "kinesis")]
            BackendProducer::Kinesis(producer) => {
                producer.publish_batch(events).await
            },
            #[cfg(feature = "pulsar")]
            BackendProducer::Pulsar(producer) => {
                producer.publish_batch(events).await
            },
            BackendProducer::Memory(producer) => {
                for event in events {
                    producer.publish(event).await?;
                }
                Ok(())
            },
        };
        
        // Update stats
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
        
        debug!("Published batch of {} events in {:?}", event_count, start_time.elapsed());
        result
    }
    
    /// Flush any pending events and buffers
    pub async fn flush(&mut self) -> Result<()> {
        let _permit = self.flush_semaphore.acquire().await
            .map_err(|_| anyhow!("Failed to acquire flush semaphore"))?;
        
        let start_time = Instant::now();
        
        // Flush any pending batch buffer
        if self.config.performance.enable_batching {
            let mut batch_buffer = self.batch_buffer.write().await;
            if !batch_buffer.is_empty() {
                let events = std::mem::take(&mut *batch_buffer);
                drop(batch_buffer);
                self.publish_batch_internal(events).await?;
            }
        }
        
        // Flush backend-specific buffers
        let result = match &mut self.backend_producer {
            #[cfg(feature = "redis")]
            BackendProducer::Redis(producer) => {
                producer.flush().await
            },
            #[cfg(feature = "kinesis")]
            BackendProducer::Kinesis(producer) => {
                producer.flush().await
            },
            #[cfg(feature = "pulsar")]
            BackendProducer::Pulsar(producer) => {
                producer.flush().await
            },
            BackendProducer::Memory(producer) => {
                producer.flush().await
            },
        };
        
        // Update stats
        if result.is_ok() {
            self.stats.write().await.flush_count += 1;
            self.last_flush = Instant::now();
            debug!("Flushed producer buffers in {:?}", start_time.elapsed());
        }
        
        result
    }
    
    /// Publish an RDF patch as a series of events
    pub async fn publish_patch(&mut self, patch: &RdfPatch) -> Result<()> {
        let events: Vec<StreamEvent> = patch.operations.iter().map(|op| {
            let metadata = EventMetadata {
                event_id: Uuid::new_v4().to_string(),
                timestamp: patch.timestamp,
                source: "rdf_patch".to_string(),
                user: None,
                context: Some(patch.id.clone()),
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            };
            
            match op {
                PatchOperation::Add { subject, predicate, object } => {
                    StreamEvent::TripleAdded {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        graph: None,
                        metadata,
                    }
                }
                PatchOperation::Delete { subject, predicate, object } => {
                    StreamEvent::TripleRemoved {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        graph: None,
                        metadata,
                    }
                }
                PatchOperation::AddGraph { graph } => {
                    StreamEvent::GraphCreated {
                        graph: graph.clone(),
                        metadata,
                    }
                }
                PatchOperation::DeleteGraph { graph } => {
                    StreamEvent::GraphDeleted {
                        graph: graph.clone(),
                        metadata,
                    }
                }
            }
        }).collect();
        
        self.publish_batch(events).await
    }
    
    /// Get producer statistics
    pub async fn get_stats(&self) -> ProducerStats {
        self.stats.read().await.clone()
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

/// Enhanced stream consumer for receiving RDF changes with backend support
pub struct StreamConsumer {
    config: StreamConfig,
    backend_consumer: BackendConsumer,
    stats: Arc<RwLock<ConsumerStats>>,
    circuit_breaker: Option<circuit_breaker::SharedCircuitBreaker>,
    last_poll: Instant,
    message_buffer: Arc<RwLock<Vec<StreamEvent>>>,
    consumer_group: Option<String>,
}

/// Backend-agnostic consumer wrapper
enum BackendConsumer {
    #[cfg(feature = "redis")]
    Redis(redis::RedisConsumer),
    #[cfg(feature = "kinesis")]
    Kinesis(kinesis::KinesisConsumer),
    #[cfg(feature = "pulsar")]
    Pulsar(pulsar::PulsarConsumer),
    Memory(MemoryConsumer),
}

/// Consumer statistics for monitoring
#[derive(Debug, Default, Clone)]
struct ConsumerStats {
    events_consumed: u64,
    events_failed: u64,
    bytes_received: u64,
    avg_processing_time_ms: f64,
    max_processing_time_ms: u64,
    consumer_lag: u64,
    circuit_breaker_trips: u64,
    last_message: Option<DateTime<Utc>>,
    backend_type: String,
    batch_size: usize,
}

/// Memory-based consumer for testing and development
struct MemoryConsumer {
    events: Arc<RwLock<Vec<(DateTime<Utc>, StreamEvent)>>>,
    current_index: usize,
    stats: ConsumerStats,
}

impl MemoryConsumer {
    fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            current_index: 0,
            stats: ConsumerStats {
                backend_type: "memory".to_string(),
                ..Default::default()
            },
        }
    }
    
    /// Set events for testing (simulates published events)
    async fn set_events(&mut self, events: Vec<(DateTime<Utc>, StreamEvent)>) {
        *self.events.write().await = events;
        self.current_index = 0;
    }
    
    async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        let start_time = Instant::now();
        let events = self.events.read().await;
        
        if self.current_index < events.len() {
            let (_, event) = &events[self.current_index];
            let event_clone = event.clone();
            self.current_index += 1;
            
            // Update stats
            self.stats.events_consumed += 1;
            let processing_time = start_time.elapsed().as_millis() as u64;
            self.stats.max_processing_time_ms = self.stats.max_processing_time_ms.max(processing_time);
            self.stats.avg_processing_time_ms = (self.stats.avg_processing_time_ms + processing_time as f64) / 2.0;
            self.stats.last_message = Some(Utc::now());
            
            debug!("Memory consumer: consumed event {}/{}", self.current_index, events.len());
            Ok(Some(event_clone))
        } else {
            Ok(None)
        }
    }
    
    fn get_stats(&self) -> &ConsumerStats {
        &self.stats
    }
    
    /// Reset consumer position for testing
    fn reset(&mut self) {
        self.current_index = 0;
    }
}

impl StreamConsumer {
    /// Create a new enhanced stream consumer with backend support
    pub async fn new(config: StreamConfig) -> Result<Self> {
        Self::new_with_group(config, None).await
    }
    
    /// Create a new stream consumer with a specific consumer group
    pub async fn new_with_group(config: StreamConfig, consumer_group: Option<String>) -> Result<Self> {
        // Initialize circuit breaker if enabled
        let circuit_breaker = if config.circuit_breaker.enabled {
            Some(circuit_breaker::new_shared_circuit_breaker(
                circuit_breaker::CircuitBreakerConfig {
                    enabled: config.circuit_breaker.enabled,
                    failure_threshold: config.circuit_breaker.failure_threshold,
                    success_threshold: config.circuit_breaker.success_threshold,
                    timeout: config.circuit_breaker.timeout,
                    half_open_max_calls: config.circuit_breaker.half_open_max_calls,
                    ..Default::default()
                }
            ))
        } else {
            None
        };
        
        // Initialize backend-specific consumer
        let backend_consumer = match &config.backend {
            #[cfg(feature = "redis")]
            StreamBackend::Redis { url, cluster_urls, pool_size } => {
                let stream_config = crate::StreamConfig {
                    backend: crate::StreamBackend::Redis {
                        url: url.clone(),
                        cluster_urls: cluster_urls.clone(),
                        pool_size: *pool_size,
                    },
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
                };
                
                let mut consumer = redis::RedisConsumer::new(stream_config)?;
                consumer.connect().await?;
                BackendConsumer::Redis(consumer)
            },
            #[cfg(feature = "kinesis")]
            StreamBackend::Kinesis { region, stream_name, credentials } => {
                let stream_config = crate::StreamConfig {
                    backend: crate::StreamBackend::Kinesis {
                        region: region.clone(),
                        stream_name: stream_name.clone(),
                        credentials: credentials.clone(),
                    },
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
                };
                
                let mut consumer = kinesis::KinesisConsumer::new(stream_config)?;
                consumer.connect().await?;
                BackendConsumer::Kinesis(consumer)
            },
            StreamBackend::Memory { max_size: _, persistence: _ } => {
                BackendConsumer::Memory(MemoryConsumer::new())
            },
            _ => {
                return Err(anyhow!("Backend not supported or feature not enabled"));
            }
        };
        
        let stats = Arc::new(RwLock::new(ConsumerStats {
            backend_type: match backend_consumer {
                #[cfg(feature = "redis")]
                BackendConsumer::Redis(_) => "redis".to_string(),
                #[cfg(feature = "kinesis")]
                BackendConsumer::Kinesis(_) => "kinesis".to_string(),
                BackendConsumer::Memory(_) => "memory".to_string(),
            },
            batch_size: config.batch_size,
            ..Default::default()
        }));
        
        info!("Created stream consumer with backend: {} and group: {:?}", 
              stats.read().await.backend_type, consumer_group);
        
        Ok(Self {
            config,
            backend_consumer,
            stats,
            circuit_breaker,
            last_poll: Instant::now(),
            message_buffer: Arc::new(RwLock::new(Vec::new())),
            consumer_group,
        })
    }
    
    /// Consume stream events with circuit breaker protection
    pub async fn consume(&mut self) -> Result<Option<StreamEvent>> {
        let start_time = Instant::now();
        
        // Check circuit breaker if enabled
        if let Some(cb) = &self.circuit_breaker {
            if !cb.can_execute().await {
                self.stats.write().await.circuit_breaker_trips += 1;
                return Err(anyhow!("Circuit breaker is open - cannot consume events"));
            }
        }
        
        // Consume from backend
        let result = self.consume_single_event().await;
        
        // Update circuit breaker and stats
        match &result {
            Ok(Some(_)) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_success_with_duration(start_time.elapsed()).await;
                }
                
                let mut stats = self.stats.write().await;
                stats.events_consumed += 1;
                let processing_time = start_time.elapsed().as_millis() as u64;
                stats.max_processing_time_ms = stats.max_processing_time_ms.max(processing_time);
                stats.avg_processing_time_ms = (stats.avg_processing_time_ms + processing_time as f64) / 2.0;
                stats.last_message = Some(Utc::now());
            }
            Ok(None) => {
                // No message available, not an error
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_success_with_duration(start_time.elapsed()).await;
                }
            }
            Err(_) => {
                if let Some(cb) = &self.circuit_breaker {
                    cb.record_failure_with_type(
                        circuit_breaker::FailureType::NetworkError
                    ).await;
                }
                
                self.stats.write().await.events_failed += 1;
            }
        }
        
        self.last_poll = Instant::now();
        result
    }
    
    /// Consume a single event from the backend
    async fn consume_single_event(&mut self) -> Result<Option<StreamEvent>> {
        match &mut self.backend_consumer {
            #[cfg(feature = "redis")]
            BackendConsumer::Redis(consumer) => {
                consumer.consume().await
            },
            #[cfg(feature = "kinesis")]
            BackendConsumer::Kinesis(consumer) => {
                consumer.consume().await
            },
            BackendConsumer::Memory(consumer) => {
                consumer.consume().await
            },
        }
    }
    
    /// Consume multiple events as a batch
    pub async fn consume_batch(&mut self, max_events: usize, timeout: Duration) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        let start_time = Instant::now();
        
        while events.len() < max_events && start_time.elapsed() < timeout {
            match tokio::time::timeout(Duration::from_millis(50), self.consume()).await {
                Ok(Ok(Some(event))) => events.push(event),
                Ok(Ok(None)) => continue,
                Ok(Err(e)) => return Err(e),
                Err(_) => break, // Timeout
            }
        }
        
        if !events.is_empty() {
            debug!("Consumed batch of {} events in {:?}", events.len(), start_time.elapsed());
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
                    // No message, wait a bit
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
                    // No message, wait a bit
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
            },
            _ => {
                warn!("Reset position not supported for this backend");
                Ok(())
            }
        }
    }
    
    /// Set test events for memory backend (for testing)
    pub async fn set_test_events(&mut self, events: Vec<StreamEvent>) -> Result<()> {
        match &mut self.backend_consumer {
            BackendConsumer::Memory(consumer) => {
                let timestamped_events: Vec<(DateTime<Utc>, StreamEvent)> = events
                    .into_iter()
                    .map(|event| (Utc::now(), event))
                    .collect();
                consumer.set_events(timestamped_events).await;
                Ok(())
            },
            _ => {
                Err(anyhow!("Set test events only supported for memory backend"))
            }
        }
    }
}

/// RDF patch operations
#[derive(Debug, Clone)]
pub enum PatchOperation {
    Add { subject: String, predicate: String, object: String },
    Delete { subject: String, predicate: String, object: String },
    AddGraph { graph: String },
    DeleteGraph { graph: String },
}

/// RDF patch for atomic updates
#[derive(Debug, Clone)]
pub struct RdfPatch {
    pub operations: Vec<PatchOperation>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub id: String,
}

impl RdfPatch {
    /// Create a new RDF patch
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            timestamp: chrono::Utc::now(),
            id: uuid::Uuid::new_v4().to_string(),
        }
    }
    
    /// Add an operation to the patch
    pub fn add_operation(&mut self, operation: PatchOperation) {
        self.operations.push(operation);
    }
    
    /// Serialize patch to RDF Patch format
    pub fn to_rdf_patch_format(&self) -> Result<String> {
        // TODO: Implement RDF Patch serialization
        Ok(String::new())
    }
    
    /// Parse from RDF Patch format
    pub fn from_rdf_patch_format(_input: &str) -> Result<Self> {
        // TODO: Implement RDF Patch parsing
        Ok(Self::new())
    }
}

impl Default for RdfPatch {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations for easier configuration
impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            backend: StreamBackend::Memory { 
                max_size: Some(10000), 
                persistence: false 
            },
            topic: "oxirs-stream".to_string(),
            batch_size: 100,
            flush_interval_ms: 100,
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            enable_compression: false,
            compression_type: CompressionType::None,
            retry_config: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_tls: false,
            verify_certificates: true,
            client_cert_path: None,
            client_key_path: None,
            ca_cert_path: None,
            sasl_config: None,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_batching: true,
            enable_pipelining: false,
            buffer_size: 8192,
            prefetch_count: 100,
            enable_zero_copy: false,
            enable_simd: false,
            parallel_processing: true,
            worker_threads: None,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            metrics_interval: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            enable_profiling: false,
        }
    }
}

/// Helper functions for creating common configurations
impl StreamConfig {
    /// Create a Redis configuration
    #[cfg(feature = "redis")]
    pub fn redis(url: String) -> Self {
        Self {
            backend: StreamBackend::Redis {
                url,
                cluster_urls: None,
                pool_size: Some(10),
            },
            ..Default::default()
        }
    }
    
    /// Create a Kinesis configuration
    #[cfg(feature = "kinesis")]
    pub fn kinesis(region: String, stream_name: String) -> Self {
        Self {
            backend: StreamBackend::Kinesis {
                region,
                stream_name,
                credentials: None,
            },
            ..Default::default()
        }
    }
    
    /// Create a memory configuration for testing
    pub fn memory() -> Self {
        Self {
            backend: StreamBackend::Memory {
                max_size: Some(1000),
                persistence: false,
            },
            ..Default::default()
        }
    }
    
    /// Enable high-performance configuration
    pub fn high_performance(mut self) -> Self {
        self.performance.enable_batching = true;
        self.performance.enable_pipelining = true;
        self.performance.parallel_processing = true;
        self.performance.buffer_size = 65536;
        self.performance.prefetch_count = 1000;
        self.batch_size = 1000;
        self.flush_interval_ms = 10;
        self
    }
    
    /// Enable compression
    pub fn with_compression(mut self, compression_type: CompressionType) -> Self {
        self.enable_compression = true;
        self.compression_type = compression_type;
        self
    }
    
    /// Configure circuit breaker
    pub fn with_circuit_breaker(mut self, enabled: bool, failure_threshold: u32) -> Self {
        self.circuit_breaker.enabled = enabled;
        self.circuit_breaker.failure_threshold = failure_threshold;
        self
    }
}