//! Serialization type definitions and schema management.
//!
//! This module contains the core types used across the serialization subsystem:
//! - [`SerializationFormat`] enum and format detection
//! - [`SerializerOptions`] and configuration
//! - [`SchemaRegistry`], [`Schema`], [`SchemaDefinition`], [`CompatibilityMode`]
//! - [`EvolutionRules`]
//! - [`DeltaCompressionType`], [`DeltaCompressedEvent`], [`EventDelta`]
//! - [`ProtobufStreamEvent`] and Avro helper functions

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::StreamEvent;

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
    pub(crate) schemas: Arc<RwLock<HashMap<String, Schema>>>,
    /// Schema evolution rules
    pub(crate) evolution_rules: EvolutionRules,
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
            .ok_or_else(|| anyhow!("Schema {id} not found"))
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

    /// Get Avro schema for event
    pub async fn get_avro_schema_for_event(
        &self,
        _event: &StreamEvent,
    ) -> Result<apache_avro::Schema> {
        // In practice, this would look up the appropriate schema
        Ok(get_default_avro_schema())
    }
}

/// Protobuf representation of StreamEvent.
///
/// This is a simplified version - in practice you'd use proper .proto definitions.
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
    fn encode_raw(&self, buf: &mut impl prost::bytes::BufMut) {
        // Simplified implementation
        buf.put_slice(&self.data);
    }

    fn merge_field(
        &mut self,
        _tag: u32,
        _wire_type: prost::encoding::WireType,
        _buf: &mut impl prost::bytes::Buf,
        _ctx: prost::encoding::DecodeContext,
    ) -> Result<(), prost::DecodeError> {
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

    let fields = vec![
        (
            "event_type".to_string(),
            apache_avro::types::Value::String("StreamEvent".to_string()),
        ),
        (
            "data".to_string(),
            apache_avro::types::Value::Bytes(json_data),
        ),
        (
            "metadata".to_string(),
            apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
        ),
    ];

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
    Full(Box<StreamEvent>),
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
