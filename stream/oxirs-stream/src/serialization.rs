//! # Event Serialization Module
//!
//! This module provides comprehensive serialization support for stream events with:
//! - Multiple format support (JSON, Protobuf, Avro, Binary)
//! - Schema evolution and versioning
//! - Compression integration
//! - Format auto-detection
//! - Schema registry integration

use anyhow::{anyhow, Result};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use chrono::{DateTime, Utc};
use crc32fast;
use futures::stream::{BoxStream, StreamExt as _};
use lz4_flex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{Cursor, Read as _};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio_stream::Stream;
use tracing::{debug, info, warn};

use crate::{CompressionType, EventMetadata, StreamEvent};

/// Serialization format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// JSON format (human-readable)
    Json,
    /// Protocol Buffers (efficient binary)
    Protobuf,
    /// Apache Avro (schema-based)
    Avro,
    /// Custom binary format
    Binary,
    /// MessagePack format
    MessagePack,
    /// CBOR (Concise Binary Object Representation)
    Cbor,
}

impl SerializationFormat {
    /// Get format identifier bytes
    pub fn magic_bytes(&self) -> &[u8] {
        match self {
            SerializationFormat::Json => b"JSON",
            SerializationFormat::Protobuf => b"PB03",
            SerializationFormat::Avro => b"Obj\x01",
            SerializationFormat::Binary => b"BIN1",
            SerializationFormat::MessagePack => b"MSGP",
            SerializationFormat::Cbor => b"CBOR",
        }
    }

    /// Detect format from magic bytes
    pub fn detect(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        let magic = &data[0..4];
        match magic {
            b"JSON" => Some(SerializationFormat::Json),
            b"PB03" => Some(SerializationFormat::Protobuf),
            b"Obj\x01" => Some(SerializationFormat::Avro),
            b"BIN1" => Some(SerializationFormat::Binary),
            b"MSGP" => Some(SerializationFormat::MessagePack),
            b"CBOR" => Some(SerializationFormat::Cbor),
            _ => {
                // Try to detect JSON by checking for common patterns
                if data.starts_with(b"{") || data.starts_with(b"[") {
                    Some(SerializationFormat::Json)
                } else {
                    None
                }
            }
        }
    }
}

/// Event serializer with format support
#[derive(Clone)]
pub struct EventSerializer {
    format: SerializationFormat,
    compression: Option<CompressionType>,
    schema_registry: Option<Arc<SchemaRegistry>>,
    options: SerializerOptions,
}

/// Serializer options
#[derive(Debug, Clone)]
pub struct SerializerOptions {
    /// Include schema ID in serialized data
    pub include_schema_id: bool,
    /// Include format magic bytes
    pub include_magic_bytes: bool,
    /// Pretty print JSON
    pub pretty_json: bool,
    /// Validate against schema
    pub validate_schema: bool,
    /// Maximum serialized size
    pub max_size: Option<usize>,
}

impl Default for SerializerOptions {
    fn default() -> Self {
        Self {
            include_schema_id: true,
            include_magic_bytes: true,
            pretty_json: false,
            validate_schema: true,
            max_size: Some(1024 * 1024), // 1MB default
        }
    }
}

/// Schema registry for managing schemas
pub struct SchemaRegistry {
    schemas: Arc<RwLock<HashMap<String, Schema>>>,
    /// Schema evolution rules
    evolution_rules: EvolutionRules,
}

/// Schema definition
#[derive(Debug, Clone)]
pub struct Schema {
    pub id: String,
    pub version: u32,
    pub format: SerializationFormat,
    pub definition: SchemaDefinition,
    pub compatibility: CompatibilityMode,
}

/// Schema definition types
#[derive(Debug, Clone)]
pub enum SchemaDefinition {
    /// JSON Schema
    JsonSchema(serde_json::Value),
    /// Protobuf descriptor
    ProtobufDescriptor(Vec<u8>),
    /// Avro schema
    AvroSchema(String),
    /// Custom schema
    Custom(HashMap<String, serde_json::Value>),
}

/// Schema compatibility modes
#[derive(Debug, Clone, Copy)]
pub enum CompatibilityMode {
    /// No compatibility checking
    None,
    /// Can read previous version
    Backward,
    /// Can read next version
    Forward,
    /// Can read both previous and next
    Full,
}

/// Schema evolution rules
#[derive(Debug, Clone)]
pub struct EvolutionRules {
    /// Allow field addition
    pub allow_field_addition: bool,
    /// Allow field removal
    pub allow_field_removal: bool,
    /// Allow type promotion
    pub allow_type_promotion: bool,
    /// Required fields
    pub required_fields: Vec<String>,
}

impl Default for EvolutionRules {
    fn default() -> Self {
        Self {
            allow_field_addition: true,
            allow_field_removal: false,
            allow_type_promotion: true,
            required_fields: vec!["event_id".to_string(), "timestamp".to_string()],
        }
    }
}

impl EventSerializer {
    /// Create a new event serializer
    pub fn new(format: SerializationFormat) -> Self {
        Self {
            format,
            compression: None,
            schema_registry: None,
            options: SerializerOptions::default(),
        }
    }

    /// Set compression type
    pub fn with_compression(mut self, compression: CompressionType) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Set schema registry
    pub fn with_schema_registry(mut self, registry: Arc<SchemaRegistry>) -> Self {
        self.schema_registry = Some(registry);
        self
    }

    /// Set serializer options
    pub fn with_options(mut self, options: SerializerOptions) -> Self {
        self.options = options;
        self
    }

    /// Serialize a stream event
    pub async fn serialize(&self, event: &StreamEvent) -> Result<Bytes> {
        let mut buffer = BytesMut::new();

        // Add magic bytes if enabled
        if self.options.include_magic_bytes {
            buffer.put(self.format.magic_bytes());
        }

        // Add schema ID if enabled and registry is available
        if self.options.include_schema_id {
            if let Some(registry) = &self.schema_registry {
                let schema_id = registry.get_schema_id_for_event(event).await?;
                buffer.put_u32(schema_id.parse::<u32>().unwrap_or(0));
            }
        }

        // Serialize based on format
        let serialized = match self.format {
            SerializationFormat::Json => self.serialize_json(event)?,
            SerializationFormat::Binary => self.serialize_binary(event)?,
            SerializationFormat::MessagePack => self.serialize_messagepack(event)?,
            SerializationFormat::Cbor => self.serialize_cbor(event)?,
            SerializationFormat::Protobuf => self.serialize_protobuf(event)?,
            SerializationFormat::Avro => self.serialize_avro(event).await?,
        };

        // Apply compression if enabled
        let data = if let Some(compression) = &self.compression {
            self.compress(&serialized, compression)?
        } else {
            serialized
        };

        // Check size limit
        if let Some(max_size) = self.options.max_size {
            if data.len() > max_size {
                return Err(anyhow!(
                    "Serialized data exceeds maximum size: {} > {}",
                    data.len(),
                    max_size
                ));
            }
        }

        buffer.put(&data[..]);
        Ok(buffer.freeze())
    }

    /// Deserialize a stream event
    pub async fn deserialize(&self, data: &[u8]) -> Result<StreamEvent> {
        let mut cursor = std::io::Cursor::new(data);
        let mut offset = 0;

        // Skip magic bytes if present
        if self.options.include_magic_bytes && data.len() >= 4 {
            let magic = &data[0..4];
            if magic == self.format.magic_bytes() {
                offset += 4;
                cursor.set_position(4);
            }
        }

        // Skip schema ID if present
        if self.options.include_schema_id && self.schema_registry.is_some() {
            if data.len() >= offset + 4 {
                offset += 4;
                cursor.set_position(offset as u64);
            }
        }

        // Get remaining data
        let event_data = &data[offset..];

        // Decompress if needed
        let decompressed = if let Some(compression) = &self.compression {
            self.decompress(event_data, compression)?
        } else {
            event_data.to_vec()
        };

        // Deserialize based on format
        match self.format {
            SerializationFormat::Json => self.deserialize_json(&decompressed),
            SerializationFormat::Binary => self.deserialize_binary(&decompressed),
            SerializationFormat::MessagePack => self.deserialize_messagepack(&decompressed),
            SerializationFormat::Cbor => self.deserialize_cbor(&decompressed),
            SerializationFormat::Protobuf => self.deserialize_protobuf(&decompressed),
            SerializationFormat::Avro => self.deserialize_avro(&decompressed).await,
        }
    }

    /// Serialize to JSON
    fn serialize_json(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        if self.options.pretty_json {
            serde_json::to_vec_pretty(event)
                .map_err(|e| anyhow!("JSON serialization failed: {}", e))
        } else {
            serde_json::to_vec(event).map_err(|e| anyhow!("JSON serialization failed: {}", e))
        }
    }

    /// Deserialize from JSON
    fn deserialize_json(&self, data: &[u8]) -> Result<StreamEvent> {
        serde_json::from_slice(data).map_err(|e| anyhow!("JSON deserialization failed: {}", e))
    }

    /// Serialize to binary format
    fn serialize_binary(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        // Custom binary format implementation
        let mut buffer = Vec::new();

        // Write version
        buffer.push(1); // Version 1

        // Write event type
        let event_type = match event {
            StreamEvent::TripleAdded { .. } => 1,
            StreamEvent::TripleRemoved { .. } => 2,
            StreamEvent::QuadAdded { .. } => 3,
            StreamEvent::QuadRemoved { .. } => 4,
            StreamEvent::GraphCreated { .. } => 5,
            StreamEvent::GraphCleared { .. } => 6,
            StreamEvent::GraphDeleted { .. } => 7,
            StreamEvent::GraphMetadataUpdated { .. } => 17,
            StreamEvent::GraphPermissionsChanged { .. } => 18,
            StreamEvent::GraphStatisticsUpdated { .. } => 19,
            StreamEvent::GraphRenamed { .. } => 20,
            StreamEvent::GraphMerged { .. } => 21,
            StreamEvent::GraphSplit { .. } => 22,
            StreamEvent::SparqlUpdate { .. } => 8,
            StreamEvent::TransactionBegin { .. } => 9,
            StreamEvent::TransactionCommit { .. } => 10,
            StreamEvent::TransactionAbort { .. } => 11,
            StreamEvent::SchemaChanged { .. } => 12,
            StreamEvent::SchemaDefinitionAdded { .. } => 23,
            StreamEvent::SchemaDefinitionRemoved { .. } => 24,
            StreamEvent::SchemaDefinitionModified { .. } => 25,
            StreamEvent::OntologyImported { .. } => 26,
            StreamEvent::OntologyRemoved { .. } => 27,
            StreamEvent::ConstraintAdded { .. } => 28,
            StreamEvent::ConstraintRemoved { .. } => 29,
            StreamEvent::ConstraintViolated { .. } => 30,
            StreamEvent::IndexCreated { .. } => 31,
            StreamEvent::IndexDropped { .. } => 32,
            StreamEvent::IndexRebuilt { .. } => 33,
            StreamEvent::ShapeAdded { .. } => 34,
            StreamEvent::ShapeRemoved { .. } => 35,
            StreamEvent::ShapeModified { .. } => 36,
            StreamEvent::ShapeValidationStarted { .. } => 37,
            StreamEvent::ShapeValidationCompleted { .. } => 38,
            StreamEvent::ShapeViolationDetected { .. } => 39,
            StreamEvent::QueryResultAdded { .. } => 14,
            StreamEvent::QueryResultRemoved { .. } => 15,
            StreamEvent::QueryCompleted { .. } => 16,
            StreamEvent::SchemaUpdated { .. } => 40,
            StreamEvent::ShapeUpdated { .. } => 41,
            StreamEvent::Heartbeat { .. } => 13,
            StreamEvent::ErrorOccurred { .. } => 42,
        };
        buffer.push(event_type);

        // Serialize fields based on event type
        match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                graph,
                metadata,
            } => {
                self.write_string(&mut buffer, subject);
                self.write_string(&mut buffer, predicate);
                self.write_string(&mut buffer, object);
                self.write_optional_string(&mut buffer, graph.as_deref());
                self.write_metadata(&mut buffer, metadata)?;
            }
            // ... implement other event types similarly
            _ => {
                return Err(anyhow!(
                    "Binary serialization not implemented for this event type"
                ))
            }
        }

        Ok(buffer)
    }

    /// Helper to write string to binary buffer
    fn write_string(&self, buffer: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(bytes);
    }

    /// Helper to write optional string
    fn write_optional_string(&self, buffer: &mut Vec<u8>, s: Option<&str>) {
        match s {
            Some(s) => {
                buffer.push(1); // Present
                self.write_string(buffer, s);
            }
            None => {
                buffer.push(0); // Not present
            }
        }
    }

    /// Helper to write metadata
    fn write_metadata(&self, buffer: &mut Vec<u8>, metadata: &EventMetadata) -> Result<()> {
        // Serialize metadata as JSON for simplicity
        let metadata_json = serde_json::to_vec(metadata)?;
        buffer.extend_from_slice(&(metadata_json.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&metadata_json);
        Ok(())
    }

    /// Deserialize from binary format
    fn deserialize_binary(&self, data: &[u8]) -> Result<StreamEvent> {
        if data.len() < 2 {
            return Err(anyhow!("Binary data too short"));
        }

        let version = data[0];
        if version != 1 {
            return Err(anyhow!("Unsupported binary format version: {}", version));
        }

        let event_type = data[1];
        let mut cursor = std::io::Cursor::new(&data[2..]);

        match event_type {
            1 => {
                // TripleAdded
                let subject = self.read_string(&mut cursor)?;
                let predicate = self.read_string(&mut cursor)?;
                let object = self.read_string(&mut cursor)?;
                let graph = self.read_optional_string(&mut cursor)?;
                let metadata = self.read_metadata(&mut cursor)?;

                Ok(StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                    graph,
                    metadata,
                })
            }
            // ... implement other event types
            _ => Err(anyhow!("Unknown event type: {}", event_type)),
        }
    }

    /// Helper to read string from cursor
    fn read_string(&self, cursor: &mut std::io::Cursor<&[u8]>) -> Result<String> {
        use std::io::Read;

        let mut len_bytes = [0u8; 4];
        cursor.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut bytes = vec![0u8; len];
        cursor.read_exact(&mut bytes)?;

        String::from_utf8(bytes).map_err(|e| anyhow!("Invalid UTF-8: {}", e))
    }

    /// Helper to read optional string
    fn read_optional_string(&self, cursor: &mut std::io::Cursor<&[u8]>) -> Result<Option<String>> {
        use std::io::Read;

        let mut present = [0u8; 1];
        cursor.read_exact(&mut present)?;

        if present[0] == 1 {
            Ok(Some(self.read_string(cursor)?))
        } else {
            Ok(None)
        }
    }

    /// Helper to read metadata
    fn read_metadata(&self, cursor: &mut std::io::Cursor<&[u8]>) -> Result<EventMetadata> {
        use std::io::Read;

        let mut len_bytes = [0u8; 4];
        cursor.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut json_bytes = vec![0u8; len];
        cursor.read_exact(&mut json_bytes)?;

        serde_json::from_slice(&json_bytes).map_err(|e| anyhow!("Failed to parse metadata: {}", e))
    }

    /// Serialize to MessagePack
    fn serialize_messagepack(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        rmp_serde::to_vec(event).map_err(|e| anyhow!("MessagePack serialization failed: {}", e))
    }

    /// Deserialize from MessagePack
    fn deserialize_messagepack(&self, data: &[u8]) -> Result<StreamEvent> {
        rmp_serde::from_slice(data)
            .map_err(|e| anyhow!("MessagePack deserialization failed: {}", e))
    }

    /// Serialize to CBOR
    fn serialize_cbor(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        serde_cbor::to_vec(event).map_err(|e| anyhow!("CBOR serialization failed: {}", e))
    }

    /// Deserialize from CBOR
    fn deserialize_cbor(&self, data: &[u8]) -> Result<StreamEvent> {
        serde_cbor::from_slice(data).map_err(|e| anyhow!("CBOR deserialization failed: {}", e))
    }

    /// Serialize to Protocol Buffers
    fn serialize_protobuf(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        // Use prost for Protocol Buffers serialization
        // For now, we'll use a JSON-based approach until proper proto definitions are created
        let json_data = serde_json::to_value(event)?;
        let proto_event = ProtobufStreamEvent::from_json(&json_data)?;

        let mut buf = Vec::new();
        prost::Message::encode(&proto_event, &mut buf)?;
        Ok(buf)
    }

    /// Deserialize from Protocol Buffers
    fn deserialize_protobuf(&self, data: &[u8]) -> Result<StreamEvent> {
        let proto_event = ProtobufStreamEvent::decode(data)?;
        let json_value = proto_event.to_json()?;
        let event: StreamEvent = serde_json::from_value(json_value)?;
        Ok(event)
    }

    /// Serialize to Apache Avro
    async fn serialize_avro(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        // Get schema from registry if available
        let schema = if let Some(registry) = &self.schema_registry {
            registry.get_avro_schema_for_event(event).await?
        } else {
            // Use default schema
            get_default_avro_schema()
        };

        // Convert event to Avro value
        let avro_value = to_avro_value(event, &schema)?;

        // Serialize with schema
        let mut writer = Vec::new();
        let mut encoder = apache_avro::Writer::new(&schema, &mut writer);
        encoder.append(avro_value)?;
        encoder.flush()?;

        Ok(writer)
    }

    /// Deserialize from Apache Avro
    async fn deserialize_avro(&self, data: &[u8]) -> Result<StreamEvent> {
        // Extract schema from data header
        let reader = apache_avro::Reader::new(data)?;
        let schema = reader.writer_schema().clone();

        // Read the first (and only) record
        if let Some(record) = reader.into_iter().next() {
            let avro_value = record?;
            let event = from_avro_value(&avro_value, &schema)?;
            Ok(event)
        } else {
            Err(anyhow!("No Avro record found in data"))
        }
    }

    /// Compress data
    fn compress(&self, data: &[u8], compression: &CompressionType) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use std::io::Write;

        match compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => {
                let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(data)?;
                encoder
                    .finish()
                    .map_err(|e| anyhow!("Gzip compression failed: {}", e))
            }
            CompressionType::Zstd => {
                zstd::encode_all(data, 3).map_err(|e| anyhow!("Zstd compression failed: {}", e))
            }
            _ => Err(anyhow!(
                "Compression type {:?} not implemented",
                compression
            )),
        }
    }

    /// Decompress data
    fn decompress(&self, data: &[u8], compression: &CompressionType) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        match compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => {
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            CompressionType::Zstd => {
                zstd::decode_all(data).map_err(|e| anyhow!("Zstd decompression failed: {}", e))
            }
            _ => Err(anyhow!(
                "Decompression type {:?} not implemented",
                compression
            )),
        }
    }
}

impl SchemaRegistry {
    /// Create a new schema registry
    pub fn new(evolution_rules: EvolutionRules) -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
            evolution_rules,
        }
    }

    /// Register a schema
    pub async fn register_schema(&self, schema: Schema) -> Result<String> {
        let schema_id = schema.id.clone();
        self.schemas.write().await.insert(schema_id.clone(), schema);
        Ok(schema_id)
    }

    /// Get schema by ID
    pub async fn get_schema(&self, id: &str) -> Result<Schema> {
        self.schemas
            .read()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Schema {} not found", id))
    }

    /// Get schema ID for an event
    pub async fn get_schema_id_for_event(&self, _event: &StreamEvent) -> Result<String> {
        // In a real implementation, this would determine the appropriate schema
        Ok("default-v1".to_string())
    }

    /// Validate schema evolution
    pub async fn validate_evolution(&self, old_schema: &Schema, new_schema: &Schema) -> Result<()> {
        match old_schema.compatibility {
            CompatibilityMode::None => Ok(()),
            CompatibilityMode::Backward => {
                // Check if new schema can read old data
                self.check_backward_compatibility(old_schema, new_schema)
            }
            CompatibilityMode::Forward => {
                // Check if old schema can read new data
                self.check_forward_compatibility(old_schema, new_schema)
            }
            CompatibilityMode::Full => {
                // Check both directions
                self.check_backward_compatibility(old_schema, new_schema)?;
                self.check_forward_compatibility(old_schema, new_schema)
            }
        }
    }

    /// Check backward compatibility
    fn check_backward_compatibility(
        &self,
        _old_schema: &Schema,
        _new_schema: &Schema,
    ) -> Result<()> {
        // Implementation would check if new schema can read old data
        Ok(())
    }

    /// Check forward compatibility
    fn check_forward_compatibility(
        &self,
        _old_schema: &Schema,
        _new_schema: &Schema,
    ) -> Result<()> {
        // Implementation would check if old schema can read new data
        Ok(())
    }
}

/// Format converter for converting between serialization formats
pub struct FormatConverter {
    source_format: SerializationFormat,
    target_format: SerializationFormat,
    schema_registry: Option<Arc<SchemaRegistry>>,
}

impl FormatConverter {
    /// Create a new format converter
    pub fn new(source: SerializationFormat, target: SerializationFormat) -> Self {
        Self {
            source_format: source,
            target_format: target,
            schema_registry: None,
        }
    }

    /// Convert data between formats
    pub async fn convert(&self, data: &[u8]) -> Result<Bytes> {
        // Deserialize from source format
        let source_serializer = EventSerializer::new(self.source_format);
        let event = source_serializer.deserialize(data).await?;

        // Serialize to target format
        let target_serializer = EventSerializer::new(self.target_format);
        target_serializer.serialize(&event).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamEvent;

    #[tokio::test]
    async fn test_json_serialization() {
        let event = StreamEvent::Heartbeat {
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        let serializer = EventSerializer::new(SerializationFormat::Json);
        let serialized = serializer.serialize(&event).await.unwrap();
        let deserialized = serializer.deserialize(&serialized).await.unwrap();

        match deserialized {
            StreamEvent::Heartbeat { source, .. } => {
                assert_eq!(source, "test");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_format_detection() {
        let json_data = b"{\"test\": \"data\"}";
        assert_eq!(
            SerializationFormat::detect(json_data),
            Some(SerializationFormat::Json)
        );

        let magic_data = b"PB03some_data";
        assert_eq!(
            SerializationFormat::detect(magic_data),
            Some(SerializationFormat::Protobuf)
        );
    }

    #[tokio::test]
    async fn test_compression() {
        let event = StreamEvent::Heartbeat {
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        let serializer =
            EventSerializer::new(SerializationFormat::Json).with_compression(CompressionType::Gzip);

        let serialized = serializer.serialize(&event).await.unwrap();
        let deserialized = serializer.deserialize(&serialized).await.unwrap();

        match deserialized {
            StreamEvent::Heartbeat { source, .. } => {
                assert_eq!(source, "test");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_messagepack_serialization() {
        let metadata = EventMetadata::default();
        let event = StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: None,
            metadata,
        };

        let serializer = EventSerializer::new(SerializationFormat::MessagePack);
        let serialized = serializer.serialize(&event).await.unwrap();
        let deserialized = serializer.deserialize(&serialized).await.unwrap();

        match deserialized {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                ..
            } => {
                assert_eq!(subject, "http://example.org/subject");
                assert_eq!(predicate, "http://example.org/predicate");
                assert_eq!(object, "http://example.org/object");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_format_conversion() {
        let event = StreamEvent::Heartbeat {
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        // Serialize to JSON
        let json_serializer = EventSerializer::new(SerializationFormat::Json);
        let json_data = json_serializer.serialize(&event).await.unwrap();

        // Convert to MessagePack
        let converter =
            FormatConverter::new(SerializationFormat::Json, SerializationFormat::MessagePack);
        let msgpack_data = converter.convert(&json_data).await.unwrap();

        // Verify by deserializing
        let msgpack_serializer = EventSerializer::new(SerializationFormat::MessagePack);
        let deserialized = msgpack_serializer.deserialize(&msgpack_data).await.unwrap();

        match deserialized {
            StreamEvent::Heartbeat { source, .. } => {
                assert_eq!(source, "test");
            }
            _ => panic!("Wrong event type"),
        }
    }
}

// Supporting types and functions for Protobuf and Avro serialization

/// Protobuf representation of StreamEvent
/// This is a simplified version - in practice you'd use proper .proto definitions
#[derive(Debug, Clone)]
pub struct ProtobufStreamEvent {
    pub event_type: String,
    pub data: Vec<u8>,
    pub metadata: Vec<u8>,
}

impl ProtobufStreamEvent {
    /// Convert from JSON value
    pub fn from_json(json: &serde_json::Value) -> Result<Self> {
        // Extract event type
        let event_type = "StreamEvent".to_string(); // Simplified

        // Serialize the entire JSON as data
        let data = serde_json::to_vec(json)?;

        // Empty metadata for now
        let metadata = Vec::new();

        Ok(Self {
            event_type,
            data,
            metadata,
        })
    }

    /// Convert to JSON value
    pub fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.data).map_err(|e| anyhow!("Failed to parse JSON: {}", e))
    }

    /// Encode using prost
    pub fn encode(&self, buf: &mut Vec<u8>) -> Result<()> {
        // Simplified encoding - in practice use proper prost::Message
        buf.extend_from_slice(&self.data);
        Ok(())
    }

    /// Decode using prost
    pub fn decode(data: &[u8]) -> Result<Self> {
        // Simplified decoding - in practice use proper prost::Message
        Ok(Self {
            event_type: "StreamEvent".to_string(),
            data: data.to_vec(),
            metadata: Vec::new(),
        })
    }
}

impl prost::Message for ProtobufStreamEvent {
    fn encode_raw<B>(&self, buf: &mut B)
    where
        B: prost::bytes::BufMut,
    {
        // Simplified implementation
        buf.put_slice(&self.data);
    }

    fn merge_field<B>(
        &mut self,
        _tag: u32,
        _wire_type: prost::encoding::WireType,
        _buf: &mut B,
        _ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError>
    where
        B: prost::bytes::Buf,
    {
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.data.len()
    }

    fn clear(&mut self) {
        self.data.clear();
        self.metadata.clear();
    }
}

/// Get default Avro schema for StreamEvent
pub fn get_default_avro_schema() -> apache_avro::Schema {
    let schema_str = r#"
    {
        "type": "record",
        "name": "StreamEvent",
        "fields": [
            {"name": "event_type", "type": "string"},
            {"name": "data", "type": "bytes"},
            {"name": "metadata", "type": ["null", "bytes"], "default": null}
        ]
    }
    "#;

    apache_avro::Schema::parse_str(schema_str).expect("Failed to parse default Avro schema")
}

/// Convert StreamEvent to Avro value
pub fn to_avro_value(
    event: &StreamEvent,
    _schema: &apache_avro::Schema,
) -> Result<apache_avro::types::Value> {
    // Simplified conversion - serialize to JSON then to bytes
    let json_data = serde_json::to_vec(event)?;

    let mut fields = Vec::new();
    fields.push((
        "event_type".to_string(),
        apache_avro::types::Value::String("StreamEvent".to_string()),
    ));
    fields.push((
        "data".to_string(),
        apache_avro::types::Value::Bytes(json_data),
    ));
    fields.push((
        "metadata".to_string(),
        apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
    ));

    Ok(apache_avro::types::Value::Record(fields))
}

/// Convert Avro value to StreamEvent
pub fn from_avro_value(
    value: &apache_avro::types::Value,
    _schema: &apache_avro::Schema,
) -> Result<StreamEvent> {
    match value {
        apache_avro::types::Value::Record(fields) => {
            // Extract data field
            for (name, field_value) in fields {
                if name == "data" {
                    if let apache_avro::types::Value::Bytes(bytes) = field_value {
                        let event: StreamEvent = serde_json::from_slice(bytes)?;
                        return Ok(event);
                    }
                }
            }
            Err(anyhow!("No data field found in Avro record"))
        }
        _ => Err(anyhow!("Expected Avro record, got {:?}", value)),
    }
}

impl SchemaRegistry {
    /// Get Avro schema for event
    pub async fn get_avro_schema_for_event(
        &self,
        _event: &StreamEvent,
    ) -> Result<apache_avro::Schema> {
        // In practice, this would look up the appropriate schema
        Ok(get_default_avro_schema())
    }
}

/// Delta compression support for event streams
pub struct DeltaCompressor {
    /// Previous event states for delta calculation
    previous_states: Arc<RwLock<HashMap<String, StreamEvent>>>,
    /// Compression algorithm to use
    compression_type: DeltaCompressionType,
    /// Maximum states to keep in memory
    max_states: usize,
}

/// Delta compression algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DeltaCompressionType {
    /// XOR-based delta compression
    Xor,
    /// Prefix compression for strings
    Prefix,
    /// Dictionary-based compression
    Dictionary,
    /// LZ4-based delta compression
    Lz4Delta,
}

impl DeltaCompressor {
    /// Create a new delta compressor
    pub fn new(compression_type: DeltaCompressionType, max_states: usize) -> Self {
        Self {
            previous_states: Arc::new(RwLock::new(HashMap::new())),
            compression_type,
            max_states,
        }
    }

    /// Compress event using delta compression
    pub async fn compress_delta(
        &self,
        event: &StreamEvent,
        event_id: &str,
    ) -> Result<DeltaCompressedEvent> {
        let mut states = self.previous_states.write().await;

        // Clean up old states if we exceed the limit
        if states.len() >= self.max_states {
            let keys_to_remove: Vec<String> = states
                .keys()
                .take(states.len() - self.max_states + 1)
                .cloned()
                .collect();
            for key in keys_to_remove {
                states.remove(&key);
            }
        }

        let delta = if let Some(previous) = states.get(event_id) {
            self.calculate_delta(previous, event)?
        } else {
            // First event, store as full event
            EventDelta::Full(event.clone())
        };

        // Update state
        states.insert(event_id.to_string(), event.clone());

        Ok(DeltaCompressedEvent {
            event_id: event_id.to_string(),
            delta,
            compression_type: self.compression_type,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Calculate delta between two events
    fn calculate_delta(&self, previous: &StreamEvent, current: &StreamEvent) -> Result<EventDelta> {
        match self.compression_type {
            DeltaCompressionType::Xor => self.calculate_xor_delta(previous, current),
            DeltaCompressionType::Prefix => self.calculate_prefix_delta(previous, current),
            DeltaCompressionType::Dictionary => self.calculate_dictionary_delta(previous, current),
            DeltaCompressionType::Lz4Delta => self.calculate_lz4_delta(previous, current),
        }
    }

    /// XOR-based delta compression
    fn calculate_xor_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_bytes = serde_json::to_vec(previous)?;
        let curr_bytes = serde_json::to_vec(current)?;

        if prev_bytes.len() != curr_bytes.len() {
            // If sizes differ, store as full event
            return Ok(EventDelta::Full(current.clone()));
        }

        let xor_bytes: Vec<u8> = prev_bytes
            .iter()
            .zip(curr_bytes.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        Ok(EventDelta::Xor(xor_bytes))
    }

    /// Prefix compression for string fields
    fn calculate_prefix_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_json = serde_json::to_value(previous)?;
        let curr_json = serde_json::to_value(current)?;

        let diff = self.calculate_json_prefix_diff(&prev_json, &curr_json)?;
        Ok(EventDelta::Prefix(diff))
    }

    /// Dictionary-based compression
    fn calculate_dictionary_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_strings = self.extract_strings_from_event(previous);
        let curr_strings = self.extract_strings_from_event(current);

        let mut dictionary = HashMap::new();
        let mut dict_id = 0u16;

        // Build dictionary from common strings
        for string in &prev_strings {
            if curr_strings.contains(string) && !dictionary.contains_key(string) {
                dictionary.insert(string.clone(), dict_id);
                dict_id += 1;
            }
        }

        // Replace strings with dictionary IDs
        let compressed_event = self.replace_strings_with_ids(current, &dictionary)?;

        Ok(EventDelta::Dictionary {
            dictionary,
            compressed_event,
        })
    }

    /// LZ4-based delta compression
    fn calculate_lz4_delta(
        &self,
        previous: &StreamEvent,
        current: &StreamEvent,
    ) -> Result<EventDelta> {
        let prev_bytes = serde_json::to_vec(previous)?;
        let curr_bytes = serde_json::to_vec(current)?;

        // Simple delta: store additions and removals
        let diff_bytes = self.calculate_byte_diff(&prev_bytes, &curr_bytes);
        let compressed = lz4_flex::compress_prepend_size(&diff_bytes);

        Ok(EventDelta::Lz4(compressed))
    }

    /// Calculate JSON prefix differences
    fn calculate_json_prefix_diff(
        &self,
        prev: &serde_json::Value,
        curr: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        match (prev, curr) {
            (serde_json::Value::Object(prev_obj), serde_json::Value::Object(curr_obj)) => {
                let mut diff = serde_json::Map::new();
                for (key, curr_val) in curr_obj {
                    if let Some(prev_val) = prev_obj.get(key) {
                        if prev_val != curr_val {
                            diff.insert(key.clone(), curr_val.clone());
                        }
                    } else {
                        diff.insert(key.clone(), curr_val.clone());
                    }
                }
                Ok(serde_json::Value::Object(diff))
            }
            _ => Ok(curr.clone()),
        }
    }

    /// Extract all strings from an event
    fn extract_strings_from_event(&self, event: &StreamEvent) -> Vec<String> {
        let mut strings = Vec::new();
        if let Ok(json) = serde_json::to_value(event) {
            self.extract_strings_from_json(&json, &mut strings);
        }
        strings
    }

    /// Recursively extract strings from JSON value
    fn extract_strings_from_json(&self, value: &serde_json::Value, strings: &mut Vec<String>) {
        match value {
            serde_json::Value::String(s) => strings.push(s.clone()),
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.extract_strings_from_json(item, strings);
                }
            }
            serde_json::Value::Object(obj) => {
                for (_, val) in obj {
                    self.extract_strings_from_json(val, strings);
                }
            }
            _ => {}
        }
    }

    /// Replace strings with dictionary IDs
    fn replace_strings_with_ids(
        &self,
        event: &StreamEvent,
        dictionary: &HashMap<String, u16>,
    ) -> Result<serde_json::Value> {
        let mut json = serde_json::to_value(event)?;
        self.replace_strings_in_json(&mut json, dictionary);
        Ok(json)
    }

    /// Recursively replace strings in JSON
    fn replace_strings_in_json(
        &self,
        value: &mut serde_json::Value,
        dictionary: &HashMap<String, u16>,
    ) {
        match value {
            serde_json::Value::String(s) => {
                if let Some(&id) = dictionary.get(s) {
                    *value = serde_json::Value::Number(serde_json::Number::from(id));
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.replace_strings_in_json(item, dictionary);
                }
            }
            serde_json::Value::Object(obj) => {
                for val in obj.values_mut() {
                    self.replace_strings_in_json(val, dictionary);
                }
            }
            _ => {}
        }
    }

    /// Calculate byte-level differences
    fn calculate_byte_diff(&self, prev: &[u8], curr: &[u8]) -> Vec<u8> {
        // Simple implementation - could be enhanced with more sophisticated diff algorithms
        let mut diff = Vec::new();

        // Store length difference
        diff.extend_from_slice(&(curr.len() as u32).to_le_bytes());
        diff.extend_from_slice(&(prev.len() as u32).to_le_bytes());

        // Store the current bytes (simplified)
        diff.extend_from_slice(curr);

        diff
    }

    /// Decompress delta-compressed event
    pub async fn decompress_delta(
        &self,
        compressed: &DeltaCompressedEvent,
        previous_event: Option<&StreamEvent>,
    ) -> Result<StreamEvent> {
        match &compressed.delta {
            EventDelta::Full(event) => Ok(event.clone()),
            EventDelta::Xor(xor_bytes) => {
                if let Some(prev) = previous_event {
                    let prev_bytes = serde_json::to_vec(prev)?;
                    if prev_bytes.len() == xor_bytes.len() {
                        let restored_bytes: Vec<u8> = prev_bytes
                            .iter()
                            .zip(xor_bytes.iter())
                            .map(|(a, b)| a ^ b)
                            .collect();
                        let event = serde_json::from_slice(&restored_bytes)?;
                        Ok(event)
                    } else {
                        Err(anyhow!("XOR delta length mismatch"))
                    }
                } else {
                    Err(anyhow!("Previous event required for XOR decompression"))
                }
            }
            EventDelta::Prefix(diff) => {
                if let Some(prev) = previous_event {
                    let mut prev_json = serde_json::to_value(prev)?;
                    self.apply_json_diff(&mut prev_json, diff)?;
                    let event = serde_json::from_value(prev_json)?;
                    Ok(event)
                } else {
                    Err(anyhow!("Previous event required for prefix decompression"))
                }
            }
            EventDelta::Dictionary {
                dictionary,
                compressed_event,
            } => {
                let mut restored_json = compressed_event.clone();
                let reverse_dict: HashMap<u16, String> =
                    dictionary.iter().map(|(k, &v)| (v, k.clone())).collect();
                self.restore_strings_from_ids(&mut restored_json, &reverse_dict);
                let event = serde_json::from_value(restored_json)?;
                Ok(event)
            }
            EventDelta::Lz4(compressed_bytes) => {
                let decompressed = lz4_flex::decompress_size_prepended(compressed_bytes)?;
                // Restore from diff (simplified - would need more sophisticated restoration)
                let event = serde_json::from_slice(&decompressed)?;
                Ok(event)
            }
        }
    }

    /// Apply JSON diff to base JSON
    fn apply_json_diff(
        &self,
        base: &mut serde_json::Value,
        diff: &serde_json::Value,
    ) -> Result<()> {
        if let (Some(base_obj), Some(diff_obj)) = (base.as_object_mut(), diff.as_object()) {
            for (key, diff_val) in diff_obj {
                base_obj.insert(key.clone(), diff_val.clone());
            }
        } else {
            *base = diff.clone();
        }
        Ok(())
    }

    /// Restore strings from dictionary IDs
    fn restore_strings_from_ids(
        &self,
        value: &mut serde_json::Value,
        reverse_dict: &HashMap<u16, String>,
    ) {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(id) = n.as_u64() {
                    if let Some(string) = reverse_dict.get(&(id as u16)) {
                        *value = serde_json::Value::String(string.clone());
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.restore_strings_from_ids(item, reverse_dict);
                }
            }
            serde_json::Value::Object(obj) => {
                for val in obj.values_mut() {
                    self.restore_strings_from_ids(val, reverse_dict);
                }
            }
            _ => {}
        }
    }
}

/// Delta-compressed event representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCompressedEvent {
    pub event_id: String,
    pub delta: EventDelta,
    pub compression_type: DeltaCompressionType,
    pub timestamp: DateTime<Utc>,
}

/// Event delta representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventDelta {
    /// Full event (no compression possible)
    Full(StreamEvent),
    /// XOR-based delta
    Xor(Vec<u8>),
    /// Prefix-based delta
    Prefix(serde_json::Value),
    /// Dictionary-based compression
    Dictionary {
        dictionary: HashMap<String, u16>,
        compressed_event: serde_json::Value,
    },
    /// LZ4 compressed delta
    Lz4(Vec<u8>),
}

/// Streaming serializer for batch processing
pub struct StreamingSerializer {
    serializer: EventSerializer,
    delta_compressor: Option<DeltaCompressor>,
    batch_size: usize,
    current_batch: Vec<StreamEvent>,
}

impl StreamingSerializer {
    /// Create a new streaming serializer
    pub fn new(serializer: EventSerializer, batch_size: usize) -> Self {
        Self {
            serializer,
            delta_compressor: None,
            batch_size,
            current_batch: Vec::new(),
        }
    }

    /// Enable delta compression
    pub fn with_delta_compression(
        mut self,
        compression_type: DeltaCompressionType,
        max_states: usize,
    ) -> Self {
        self.delta_compressor = Some(DeltaCompressor::new(compression_type, max_states));
        self
    }

    /// Add event to batch
    pub async fn add_event(&mut self, event: StreamEvent) -> Result<Option<Bytes>> {
        self.current_batch.push(event);

        if self.current_batch.len() >= self.batch_size {
            self.flush_batch().await
        } else {
            Ok(None)
        }
    }

    /// Flush current batch
    pub async fn flush_batch(&mut self) -> Result<Option<Bytes>> {
        if self.current_batch.is_empty() {
            return Ok(None);
        }

        let batch = std::mem::take(&mut self.current_batch);
        let serialized = self.serialize_batch(&batch).await?;
        Ok(Some(serialized))
    }

    /// Serialize a batch of events
    async fn serialize_batch(&self, batch: &[StreamEvent]) -> Result<Bytes> {
        let mut buffer = BytesMut::new();

        // Write batch header
        buffer.put_u32(batch.len() as u32);
        buffer.put_u64(chrono::Utc::now().timestamp_millis() as u64);

        // Serialize each event
        for event in batch {
            let event_data = self.serializer.serialize(event).await?;
            buffer.put_u32(event_data.len() as u32);
            buffer.put(event_data);
        }

        Ok(buffer.freeze())
    }

    /// Deserialize a batch of events
    pub async fn deserialize_batch(&self, data: &[u8]) -> Result<Vec<StreamEvent>> {
        let mut cursor = std::io::Cursor::new(data);
        let mut events = Vec::new();

        // Read batch header
        let batch_size = cursor.get_u32();
        let _timestamp = cursor.get_u64();

        // Read each event
        for _ in 0..batch_size {
            let event_size = cursor.get_u32() as usize;
            let event_data =
                &data[cursor.position() as usize..(cursor.position() as usize + event_size)];
            cursor.advance(event_size);

            let event = self.serializer.deserialize(event_data).await?;
            events.push(event);
        }

        Ok(events)
    }

    /// Create a stream of serialized batches
    pub fn create_batch_stream(
        &self,
        events: impl Stream<Item = StreamEvent> + Send + 'static,
    ) -> BoxStream<'static, Result<Bytes>> {
        let serializer = self.serializer.clone();
        let batch_size = self.batch_size;

        Box::pin(events.chunks(batch_size).then(move |chunk| {
            let serializer = serializer.clone();
            async move {
                let streaming_serializer = StreamingSerializer::new(serializer, batch_size);
                streaming_serializer.serialize_batch(&chunk).await
            }
        }))
    }
}

/// Enhanced binary format with streaming support
pub struct EnhancedBinaryFormat {
    version: u8,
    enable_compression: bool,
    enable_checksums: bool,
    chunk_size: usize,
}

impl EnhancedBinaryFormat {
    /// Create a new enhanced binary format
    pub fn new() -> Self {
        Self {
            version: 2, // Enhanced version
            enable_compression: true,
            enable_checksums: true,
            chunk_size: 8192, // 8KB chunks
        }
    }

    /// Configure compression
    pub fn with_compression(mut self, enable: bool) -> Self {
        self.enable_compression = enable;
        self
    }

    /// Configure checksums
    pub fn with_checksums(mut self, enable: bool) -> Self {
        self.enable_checksums = enable;
        self
    }

    /// Set chunk size for streaming
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Serialize event in enhanced binary format
    pub async fn serialize(&self, event: &StreamEvent) -> Result<Bytes> {
        let mut buffer = BytesMut::new();

        // Header
        buffer.put(&b"BIN2"[..]); // Magic bytes for v2
        buffer.put_u8(self.version);
        buffer.put_u8(self.get_flags());

        // Serialize event data
        let event_json = serde_json::to_vec(event)?;

        // Apply compression if enabled
        let data = if self.enable_compression {
            lz4_flex::compress_prepend_size(&event_json)
        } else {
            event_json
        };

        // Add checksum if enabled
        if self.enable_checksums {
            let checksum = crc32fast::hash(&data);
            buffer.put_u32(checksum);
        }

        // Add data length and data
        buffer.put_u32(data.len() as u32);
        buffer.put(&data[..]);

        Ok(buffer.freeze())
    }

    /// Deserialize event from enhanced binary format
    pub async fn deserialize(&self, data: &[u8]) -> Result<StreamEvent> {
        let mut cursor = std::io::Cursor::new(data);

        // Check magic bytes
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"BIN2" {
            return Err(anyhow!("Invalid magic bytes for enhanced binary format"));
        }

        // Read version and flags
        let version = cursor.get_u8();
        if version != self.version {
            return Err(anyhow!(
                "Unsupported enhanced binary format version: {}",
                version
            ));
        }

        let flags = cursor.get_u8();
        let has_compression = (flags & 0x01) != 0;
        let has_checksum = (flags & 0x02) != 0;

        // Read checksum if present
        let expected_checksum = if has_checksum {
            Some(cursor.get_u32())
        } else {
            None
        };

        // Read data
        let data_len = cursor.get_u32() as usize;
        let mut event_data = vec![0u8; data_len];
        cursor.read_exact(&mut event_data)?;

        // Verify checksum
        if let Some(expected) = expected_checksum {
            let actual = crc32fast::hash(&event_data);
            if actual != expected {
                return Err(anyhow!(
                    "Checksum mismatch: expected {}, got {}",
                    expected,
                    actual
                ));
            }
        }

        // Decompress if needed
        let decompressed = if has_compression {
            lz4_flex::decompress_size_prepended(&event_data)?
        } else {
            event_data
        };

        // Deserialize event
        let event = serde_json::from_slice(&decompressed)?;
        Ok(event)
    }

    /// Create streaming chunks for large events
    pub async fn serialize_streaming(&self, event: &StreamEvent) -> Result<Vec<Bytes>> {
        let serialized = self.serialize(event).await?;
        let mut chunks = Vec::new();

        if serialized.len() <= self.chunk_size {
            chunks.push(serialized);
        } else {
            // Split into chunks
            let chunk_count = (serialized.len() + self.chunk_size - 1) / self.chunk_size;

            for i in 0..chunk_count {
                let start = i * self.chunk_size;
                let end = std::cmp::min(start + self.chunk_size, serialized.len());

                let mut chunk_buffer = BytesMut::new();
                chunk_buffer.put(&b"CHNK"[..]); // Chunk magic
                chunk_buffer.put_u32(i as u32); // Chunk index
                chunk_buffer.put_u32(chunk_count as u32); // Total chunks
                chunk_buffer.put_u32((end - start) as u32); // Chunk size
                chunk_buffer.put(&serialized[start..end]);

                chunks.push(chunk_buffer.freeze());
            }
        }

        Ok(chunks)
    }

    /// Reassemble streaming chunks
    pub async fn deserialize_streaming(&self, chunks: Vec<Bytes>) -> Result<StreamEvent> {
        if chunks.len() == 1 && !chunks[0].starts_with(b"CHNK") {
            // Single chunk, deserialize directly
            return self.deserialize(&chunks[0]).await;
        }

        // Reassemble chunks
        let mut chunk_data: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
        let mut total_chunks = 0;

        for chunk in chunks {
            if !chunk.starts_with(b"CHNK") {
                return Err(anyhow!("Invalid chunk format"));
            }

            let mut cursor = std::io::Cursor::new(&chunk[4..]);
            let chunk_index = cursor.get_u32();
            let chunk_count = cursor.get_u32();
            let chunk_size = cursor.get_u32() as usize;

            total_chunks = chunk_count;

            let data = chunk[16..16 + chunk_size].to_vec();
            chunk_data.insert(chunk_index, data);
        }

        if chunk_data.len() != total_chunks as usize {
            return Err(anyhow!(
                "Missing chunks: got {}, expected {}",
                chunk_data.len(),
                total_chunks
            ));
        }

        // Reassemble data
        let mut reassembled = Vec::new();
        for (_index, data) in chunk_data {
            reassembled.extend(data);
        }

        // Deserialize reassembled data
        self.deserialize(&reassembled).await
    }

    /// Get format flags
    fn get_flags(&self) -> u8 {
        let mut flags = 0u8;
        if self.enable_compression {
            flags |= 0x01;
        }
        if self.enable_checksums {
            flags |= 0x02;
        }
        flags
    }
}

impl Default for EnhancedBinaryFormat {
    fn default() -> Self {
        Self::new()
    }
}

// Required imports are now at the top of the file
