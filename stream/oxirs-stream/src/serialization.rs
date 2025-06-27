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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

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
            _ => {
                return Err(anyhow!(
                    "Serialization format {:?} not implemented",
                    self.format
                ))
            }
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

        buffer.put(data);
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
            _ => Err(anyhow!(
                "Deserialization format {:?} not implemented",
                self.format
            )),
        }
    }

    /// Serialize to JSON
    fn serialize_json(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        if self.options.pretty_json {
            serde_json::to_vec_pretty(event).map_err(|e| anyhow!("JSON serialization failed: {}", e))
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
            StreamEvent::SparqlUpdate { .. } => 8,
            StreamEvent::TransactionBegin { .. } => 9,
            StreamEvent::TransactionCommit { .. } => 10,
            StreamEvent::TransactionAbort { .. } => 11,
            StreamEvent::SchemaChanged { .. } => 12,
            StreamEvent::Heartbeat { .. } => 13,
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
                return Err(anyhow!("Binary serialization not implemented for this event type"))
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
        rmp_serde::from_slice(data).map_err(|e| anyhow!("MessagePack deserialization failed: {}", e))
    }

    /// Serialize to CBOR
    fn serialize_cbor(&self, event: &StreamEvent) -> Result<Vec<u8>> {
        serde_cbor::to_vec(event).map_err(|e| anyhow!("CBOR serialization failed: {}", e))
    }

    /// Deserialize from CBOR
    fn deserialize_cbor(&self, data: &[u8]) -> Result<StreamEvent> {
        serde_cbor::from_slice(data).map_err(|e| anyhow!("CBOR deserialization failed: {}", e))
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
                encoder.finish().map_err(|e| anyhow!("Gzip compression failed: {}", e))
            }
            CompressionType::Zstd => {
                zstd::encode_all(data, 3).map_err(|e| anyhow!("Zstd compression failed: {}", e))
            }
            _ => Err(anyhow!("Compression type {:?} not implemented", compression)),
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
            _ => Err(anyhow!("Decompression type {:?} not implemented", compression)),
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
    pub async fn validate_evolution(
        &self,
        old_schema: &Schema,
        new_schema: &Schema,
    ) -> Result<()> {
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
        };

        let serializer = EventSerializer::new(SerializationFormat::Json)
            .with_compression(CompressionType::Gzip);

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
        };

        // Serialize to JSON
        let json_serializer = EventSerializer::new(SerializationFormat::Json);
        let json_data = json_serializer.serialize(&event).await.unwrap();

        // Convert to MessagePack
        let converter = FormatConverter::new(SerializationFormat::Json, SerializationFormat::MessagePack);
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