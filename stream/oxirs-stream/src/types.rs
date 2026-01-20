//! # Stream Types
//!
//! Common types used throughout the streaming module.

use crate::event;
use oxicode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Topic name wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub struct TopicName(String);

impl TopicName {
    pub fn new(name: String) -> Self {
        Self(name)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TopicName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for TopicName {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for TopicName {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Partition identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub struct PartitionId(u32);

impl PartitionId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn value(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for PartitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Message offset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub struct Offset(u64);

impl Offset {
    pub fn new(offset: u64) -> Self {
        Self(offset)
    }

    pub fn value(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stream position for seeking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum StreamPosition {
    /// Start from the beginning
    Beginning,
    /// Start from the end
    End,
    /// Start from a specific offset
    Offset(u64),
}

/// Enhanced event metadata for tracking and provenance with advanced features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Source system or component
    pub source: String,
    /// User who triggered the event
    pub user: Option<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Trace identifier for distributed tracing
    pub trace_id: Option<String>,
    /// Causality token for event ordering
    pub causality_token: Option<String>,
    /// Event version for schema evolution
    pub version: Option<String>,

    // Enhanced metadata fields (TODO items)
    /// Event timestamp with high precision
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Operation context with request details
    pub operation_context: Option<OperationContext>,
    /// Event priority for processing order
    pub priority: EventPriority,
    /// Partition information for routing
    pub partition: Option<PartitionId>,
    /// Event correlation ID for related events
    pub correlation_id: Option<String>,
    /// Checksum for data integrity
    pub checksum: Option<String>,
    /// Schema version for data format
    pub schema_version: String,
    /// Event tags for filtering and routing
    pub tags: HashMap<String, String>,
    /// Event TTL (time to live) in seconds
    pub ttl_seconds: Option<u64>,
    /// Compression type used for payload
    pub compression: Option<CompressionType>,
    /// Serialization format used
    pub serialization_format: SerializationFormat,
    /// Message size in bytes
    pub message_size: Option<usize>,
    /// Processing hints for consumers
    pub processing_hints: ProcessingHints,
}

/// Conversion from types::EventMetadata to event::EventMetadata
impl From<EventMetadata> for event::EventMetadata {
    fn from(metadata: EventMetadata) -> Self {
        Self {
            event_id: format!(
                "evt_{}",
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ), // Generate simple ID
            timestamp: metadata.timestamp,
            source: metadata.source,
            user: metadata.user,
            context: metadata.operation_context.map(|ctx| ctx.operation_type),
            caused_by: metadata.causality_token,
            version: metadata.version.unwrap_or(metadata.schema_version),
            properties: HashMap::new(), // Could be populated from custom fields
            checksum: metadata.checksum,
        }
    }
}

/// Conversion from event::EventMetadata to types::EventMetadata
impl From<event::EventMetadata> for EventMetadata {
    fn from(metadata: event::EventMetadata) -> Self {
        Self {
            source: metadata.source,
            user: metadata.user,
            session_id: None,
            trace_id: None,
            causality_token: metadata.caused_by,
            version: Some(metadata.version),
            timestamp: metadata.timestamp,
            operation_context: metadata.context.map(|ctx| OperationContext {
                operation_type: ctx,
                request_id: None,
                client_info: None,
                metrics: None,
                auth_context: None,
                custom_fields: HashMap::new(),
            }),
            priority: EventPriority::Normal,
            partition: None,
            correlation_id: None,
            checksum: metadata.checksum,
            schema_version: "1.0".to_string(),
            tags: metadata.properties,
            ttl_seconds: None,
            compression: None,
            serialization_format: SerializationFormat::Json,
            message_size: None,
            processing_hints: ProcessingHints::default(),
        }
    }
}

/// Operation context for enhanced tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationContext {
    /// Operation type (INSERT, DELETE, UPDATE, etc.)
    pub operation_type: String,
    /// Request ID from the original request
    pub request_id: Option<String>,
    /// Client information
    pub client_info: Option<ClientInfo>,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
    /// Authentication context
    pub auth_context: Option<AuthContext>,
    /// Additional custom context
    pub custom_fields: HashMap<String, String>,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ClientInfo {
    /// Client application name
    pub application: String,
    /// Client version
    pub version: String,
    /// Client IP address
    pub ip_address: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Geographic location
    pub location: Option<GeoLocation>,
}

/// Geographic location information
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct GeoLocation {
    /// Country code (ISO 3166-1 alpha-2)
    pub country: String,
    /// Region or state
    pub region: Option<String>,
    /// City
    pub city: Option<String>,
    /// Latitude
    pub lat: Option<f64>,
    /// Longitude
    pub lon: Option<f64>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PerformanceMetrics {
    /// Processing latency in microseconds
    pub processing_latency_us: Option<u64>,
    /// Queue wait time in microseconds
    pub queue_wait_time_us: Option<u64>,
    /// Serialization time in microseconds
    pub serialization_time_us: Option<u64>,
    /// Network latency in microseconds
    pub network_latency_us: Option<u64>,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<u64>,
    /// CPU time used in microseconds
    pub cpu_time_us: Option<u64>,
}

/// Authentication context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthContext {
    /// Authenticated user ID
    pub user_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// Permissions granted
    pub permissions: Vec<String>,
    /// Authentication method used
    pub auth_method: String,
    /// Token expiration time
    pub token_expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Event priority levels
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    Default,
    Encode,
    Decode,
)]
pub enum EventPriority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Compression types for payload optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Encode, Decode)]
pub enum CompressionType {
    #[default]
    None,
    Gzip,
    Lz4,
    Zstd,
    Snappy,
    Brotli,
}

/// Serialization formats supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Encode, Decode)]
pub enum SerializationFormat {
    #[default]
    Json,
    MessagePack,
    Protobuf,
    Avro,
    Cbor,
    Bincode,
}

/// Processing hints for optimized handling
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ProcessingHints {
    /// Whether event can be processed out of order
    pub allow_out_of_order: bool,
    /// Whether event can be deduplicated
    pub allow_deduplication: bool,
    /// Batch processing preference
    pub batch_preference: BatchPreference,
    /// Required consistency level
    pub consistency_level: ConsistencyLevel,
    /// Retry policy for failures
    pub retry_policy: RetryPolicy,
    /// Processing timeout in milliseconds
    pub processing_timeout_ms: Option<u64>,
}

/// Batch processing preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum BatchPreference {
    /// Process immediately, don't batch
    Immediate,
    /// Can be batched for efficiency
    Batchable,
    /// Must be batched with related events
    RequiredBatch,
}

/// Consistency level requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum ConsistencyLevel {
    /// Eventual consistency is acceptable
    Eventual,
    /// Strong consistency required within partition
    PerPartition,
    /// Strong consistency required globally
    Strong,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct RetryPolicy {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Whether to use jitter
    pub use_jitter: bool,
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self {
            source: "oxirs-stream".to_string(),
            user: None,
            session_id: None,
            trace_id: None,
            causality_token: None,
            version: Some("1.0".to_string()),
            timestamp: chrono::Utc::now(),
            operation_context: None,
            priority: EventPriority::Normal,
            partition: None,
            correlation_id: None,
            checksum: None,
            schema_version: "1.0".to_string(),
            tags: HashMap::new(),
            ttl_seconds: None,
            compression: None,
            serialization_format: SerializationFormat::Json,
            message_size: None,
            processing_hints: ProcessingHints::default(),
        }
    }
}

impl Default for ProcessingHints {
    fn default() -> Self {
        Self {
            allow_out_of_order: false,
            allow_deduplication: true,
            batch_preference: BatchPreference::Batchable,
            consistency_level: ConsistencyLevel::PerPartition,
            retry_policy: RetryPolicy::default(),
            processing_timeout_ms: Some(30000), // 30 seconds
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 10000,
            backoff_multiplier: 2.0,
            use_jitter: true,
        }
    }
}

/// Enhanced serialization utilities for different formats
pub mod serialization {
    use super::*;
    use anyhow::{anyhow, Result};

    /// Serialize event metadata using specified format
    pub fn serialize_metadata(
        metadata: &EventMetadata,
        format: SerializationFormat,
    ) -> Result<Vec<u8>> {
        match format {
            SerializationFormat::Json => {
                serde_json::to_vec(metadata).map_err(|e| anyhow!("JSON serialization failed: {e}"))
            }
            SerializationFormat::MessagePack => rmp_serde::to_vec(metadata)
                .map_err(|e| anyhow!("MessagePack serialization failed: {e}")),
            SerializationFormat::Cbor => {
                serde_cbor::to_vec(metadata).map_err(|e| anyhow!("CBOR serialization failed: {e}"))
            }
            SerializationFormat::Bincode => {
                oxicode::serde::encode_to_vec(metadata, oxicode::config::standard())
                    .map_err(|e| anyhow!("Bincode serialization failed: {e}"))
            }
            SerializationFormat::Protobuf | SerializationFormat::Avro => {
                // These would require schema generation and external dependencies
                // For now, fallback to JSON
                serde_json::to_vec(metadata)
                    .map_err(|e| anyhow!("Protobuf/Avro serialization fallback failed: {e}"))
            }
        }
    }

    /// Deserialize event metadata from specified format
    pub fn deserialize_metadata(data: &[u8], format: SerializationFormat) -> Result<EventMetadata> {
        match format {
            SerializationFormat::Json => serde_json::from_slice(data)
                .map_err(|e| anyhow!("JSON deserialization failed: {e}")),
            SerializationFormat::MessagePack => rmp_serde::from_slice(data)
                .map_err(|e| anyhow!("MessagePack deserialization failed: {e}")),
            SerializationFormat::Cbor => serde_cbor::from_slice(data)
                .map_err(|e| anyhow!("CBOR deserialization failed: {e}")),
            SerializationFormat::Bincode => {
                oxicode::serde::decode_from_slice(data, oxicode::config::standard())
                    .map(|(v, _)| v)
                    .map_err(|e| anyhow!("Bincode deserialization failed: {e}"))
            }
            SerializationFormat::Protobuf | SerializationFormat::Avro => {
                // These would require schema generation and external dependencies
                // For now, fallback to JSON
                serde_json::from_slice(data)
                    .map_err(|e| anyhow!("Protobuf/Avro deserialization fallback failed: {e}"))
            }
        }
    }

    /// Compress data using specified compression type
    pub fn compress_data(data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
        match compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => {
                use flate2::write::GzEncoder;
                use flate2::Compression;
                use std::io::Write;

                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            CompressionType::Lz4 => Ok(lz4_flex::compress_prepend_size(data)),
            CompressionType::Zstd => {
                zstd::bulk::compress(data, 0).map_err(|e| anyhow!("Zstd compression failed: {e}"))
            }
            CompressionType::Snappy => Ok(snap::raw::Encoder::new().compress_vec(data)?),
            CompressionType::Brotli => {
                use brotli::CompressorWriter;
                use std::io::Write;
                let mut compressed = Vec::new();
                {
                    let mut compressor = CompressorWriter::new(&mut compressed, 4096, 6, 22);
                    compressor.write_all(data)?;
                } // Compressor is dropped here, flushing data to compressed
                Ok(compressed)
            }
        }
    }

    /// Decompress data using specified compression type
    pub fn decompress_data(data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
        match compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => {
                use flate2::read::GzDecoder;
                use std::io::Read;

                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            CompressionType::Lz4 => lz4_flex::decompress_size_prepended(data)
                .map_err(|e| anyhow!("LZ4 decompression failed: {e}")),
            CompressionType::Zstd => {
                zstd::bulk::decompress(data, 1024 * 1024) // 1MB max decompressed size
                    .map_err(|e| anyhow!("Zstd decompression failed: {e}"))
            }
            CompressionType::Snappy => snap::raw::Decoder::new()
                .decompress_vec(data)
                .map_err(|e| anyhow!("Snappy decompression failed: {e}")),
            CompressionType::Brotli => {
                use std::io::Read;
                let mut decompressed = Vec::new();
                let mut decompressor = brotli::Decompressor::new(data, 4096);
                decompressor.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
        }
    }
}

/// Event processing utilities
pub mod processing {
    use super::*;
    use std::time::{Duration, Instant};

    /// Event processor for handling metadata and optimizations
    pub struct EventProcessor {
        pub deduplication_cache: std::collections::HashSet<String>,
        pub batch_buffer: Vec<(crate::event::StreamEvent, EventMetadata)>,
        pub last_flush: Instant,
        pub flush_interval: Duration,
    }

    impl Default for EventProcessor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl EventProcessor {
        pub fn new() -> Self {
            Self {
                deduplication_cache: std::collections::HashSet::new(),
                batch_buffer: Vec::new(),
                last_flush: Instant::now(),
                flush_interval: Duration::from_millis(100),
            }
        }

        /// Process event with metadata enhancements
        pub fn process_event(
            &mut self,
            mut event: crate::event::StreamEvent,
        ) -> anyhow::Result<Option<crate::event::StreamEvent>> {
            // Extract and enhance metadata
            let metadata = self.extract_metadata(&event)?;
            let enhanced_metadata = self.enhance_metadata(metadata)?;

            // Check for deduplication
            if enhanced_metadata.processing_hints.allow_deduplication {
                if let Some(correlation_id) = &enhanced_metadata.correlation_id {
                    if self.deduplication_cache.contains(correlation_id) {
                        return Ok(None); // Duplicate event, skip
                    }
                    self.deduplication_cache.insert(correlation_id.clone());
                }
            }

            // Update event metadata
            self.update_event_metadata(&mut event, enhanced_metadata)?;

            // Handle batching
            match self.get_batch_preference(&event) {
                BatchPreference::Immediate => Ok(Some(event)),
                BatchPreference::Batchable | BatchPreference::RequiredBatch => {
                    self.add_to_batch(event);

                    // Check if we should flush the batch
                    if self.should_flush_batch() {
                        // For simplicity, return the last event
                        // In a real implementation, this would return a batch
                        Ok(self.batch_buffer.last().map(|(e, _)| e.clone()))
                    } else {
                        Ok(None)
                    }
                }
            }
        }

        fn extract_metadata(
            &self,
            event: &crate::event::StreamEvent,
        ) -> anyhow::Result<EventMetadata> {
            // Extract metadata from event based on event type
            match event {
                crate::event::StreamEvent::TripleAdded { metadata, .. } => {
                    Ok(metadata.clone().into())
                }
                crate::event::StreamEvent::TripleRemoved { metadata, .. } => {
                    Ok(metadata.clone().into())
                }
                crate::event::StreamEvent::GraphCreated { metadata, .. } => {
                    Ok(metadata.clone().into())
                }
                crate::event::StreamEvent::SparqlUpdate { metadata, .. } => {
                    Ok(metadata.clone().into())
                }
                crate::event::StreamEvent::TransactionBegin { metadata, .. } => {
                    Ok(metadata.clone().into())
                }
                crate::event::StreamEvent::Heartbeat { metadata, .. } => {
                    Ok(metadata.clone().into())
                }
                _ => Ok(EventMetadata::default()),
            }
        }

        fn enhance_metadata(&self, mut metadata: EventMetadata) -> anyhow::Result<EventMetadata> {
            // Add timestamp if not present
            if metadata.timestamp == chrono::DateTime::<chrono::Utc>::MIN_UTC {
                metadata.timestamp = chrono::Utc::now();
            }

            // Generate correlation ID if not present
            if metadata.correlation_id.is_none() {
                metadata.correlation_id = Some(uuid::Uuid::new_v4().to_string());
            }

            // Set default schema version
            if metadata.schema_version.is_empty() {
                metadata.schema_version = "1.0".to_string();
            }

            // Add performance metrics
            if metadata.operation_context.is_none() {
                metadata.operation_context = Some(OperationContext {
                    operation_type: "stream_event".to_string(),
                    request_id: Some(uuid::Uuid::new_v4().to_string()),
                    client_info: None,
                    metrics: Some(PerformanceMetrics {
                        processing_latency_us: Some(0),
                        queue_wait_time_us: Some(0),
                        serialization_time_us: Some(0),
                        network_latency_us: Some(0),
                        memory_usage_bytes: Some(0),
                        cpu_time_us: Some(0),
                    }),
                    auth_context: None,
                    custom_fields: HashMap::new(),
                });
            }

            Ok(metadata)
        }

        fn update_event_metadata(
            &self,
            event: &mut crate::event::StreamEvent,
            metadata: EventMetadata,
        ) -> anyhow::Result<()> {
            let event_metadata = event::EventMetadata::from(metadata);
            match event {
                crate::event::StreamEvent::TripleAdded { metadata: m, .. } => *m = event_metadata,
                crate::event::StreamEvent::TripleRemoved { metadata: m, .. } => *m = event_metadata,
                crate::event::StreamEvent::GraphCreated { metadata: m, .. } => *m = event_metadata,
                crate::event::StreamEvent::SparqlUpdate { metadata: m, .. } => *m = event_metadata,
                crate::event::StreamEvent::TransactionBegin { metadata: m, .. } => {
                    *m = event_metadata
                }
                crate::event::StreamEvent::Heartbeat { metadata: m, .. } => *m = event_metadata,
                _ => {}
            }
            Ok(())
        }

        fn get_batch_preference(&self, event: &crate::event::StreamEvent) -> BatchPreference {
            match event {
                crate::event::StreamEvent::Heartbeat { .. } => BatchPreference::Immediate,
                crate::event::StreamEvent::TransactionBegin { .. } => BatchPreference::Immediate,
                crate::event::StreamEvent::TransactionCommit { .. } => BatchPreference::Immediate,
                crate::event::StreamEvent::TransactionAbort { .. } => BatchPreference::Immediate,
                _ => BatchPreference::Batchable,
            }
        }

        fn add_to_batch(&mut self, event: crate::event::StreamEvent) {
            let metadata = self.extract_metadata(&event).unwrap_or_default();
            self.batch_buffer.push((event, metadata));
        }

        fn should_flush_batch(&self) -> bool {
            self.batch_buffer.len() >= 100 || self.last_flush.elapsed() >= self.flush_interval
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::types::serialization::{compress_data, decompress_data};

        #[test]
        fn test_compression_round_trip() {
            let test_data = b"Hello, World! This is a test message for compression.";
            let compression_types = vec![
                CompressionType::None,
                CompressionType::Gzip,
                CompressionType::Lz4,
                CompressionType::Zstd,
                CompressionType::Snappy,
                CompressionType::Brotli,
            ];

            for compression in compression_types {
                let compressed = compress_data(test_data, compression).unwrap();
                let decompressed = decompress_data(&compressed, compression).unwrap();
                assert_eq!(
                    test_data,
                    decompressed.as_slice(),
                    "Failed round-trip for {compression:?}"
                );
            }
        }

        #[test]
        fn test_compression_effectiveness() {
            let test_data = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // Repetitive data
            let compression_types = vec![
                CompressionType::Gzip,
                CompressionType::Lz4,
                CompressionType::Zstd,
                CompressionType::Snappy,
                CompressionType::Brotli,
            ];

            for compression in compression_types {
                let compressed = compress_data(test_data, compression).unwrap();
                // Compressed data should be smaller than original for repetitive data
                assert!(
                    compressed.len() < test_data.len(),
                    "Compression {compression:?} did not reduce size"
                );
            }
        }

        #[test]
        fn test_empty_data_compression() {
            let test_data = b"";
            let compression_types = vec![
                CompressionType::None,
                CompressionType::Gzip,
                CompressionType::Lz4,
                CompressionType::Zstd,
                CompressionType::Snappy,
                CompressionType::Brotli,
            ];

            for compression in compression_types {
                let compressed = compress_data(test_data, compression).unwrap();
                let decompressed = decompress_data(&compressed, compression).unwrap();
                assert_eq!(
                    test_data,
                    decompressed.as_slice(),
                    "Failed empty data round-trip for {compression:?}"
                );
            }
        }

        #[test]
        fn test_large_data_compression() {
            let test_data = vec![42u8; 10000]; // 10KB of data
            let compression_types = vec![
                CompressionType::None,
                CompressionType::Gzip,
                CompressionType::Lz4,
                CompressionType::Zstd,
                CompressionType::Snappy,
                CompressionType::Brotli,
            ];

            for compression in compression_types {
                let compressed = compress_data(&test_data, compression).unwrap();
                let decompressed = decompress_data(&compressed, compression).unwrap();
                assert_eq!(
                    test_data, decompressed,
                    "Failed large data round-trip for {compression:?}"
                );
            }
        }

        #[test]
        fn test_random_data_compression() {
            use scirs2_core::random::{Random, Rng};
            let mut random_gen = Random::default();
            let test_data: Vec<u8> = (0..1000).map(|_| random_gen.random()).collect();
            let compression_types = vec![
                CompressionType::None,
                CompressionType::Gzip,
                CompressionType::Lz4,
                CompressionType::Zstd,
                CompressionType::Snappy,
                CompressionType::Brotli,
            ];

            for compression in compression_types {
                let compressed = compress_data(&test_data, compression).unwrap();
                let decompressed = decompress_data(&compressed, compression).unwrap();
                assert_eq!(
                    test_data, decompressed,
                    "Failed random data round-trip for {compression:?}"
                );
            }
        }
    }
}
