//! MQTT Backend Types
//!
//! MQTT 3.1.1 and 5.0 support for IoT and Industry 4.0 integration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MQTT Backend Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttConfig {
    /// Broker URL (tcp://host:1883 or ssl://host:8883)
    pub broker_url: String,

    /// MQTT client ID
    pub client_id: String,

    /// Clean session flag (true = no persistent session)
    pub clean_session: bool,

    /// Keep alive interval in seconds
    pub keep_alive_secs: u16,

    /// Default QoS level
    pub default_qos: QoS,

    /// Username (optional)
    pub username: Option<String>,

    /// Password (optional)
    pub password: Option<String>,

    /// TLS configuration (for mqtts://)
    pub tls: Option<MqttTlsConfig>,

    /// Reconnect configuration
    pub reconnect: MqttReconnectConfig,

    /// Last will & testament (optional)
    pub last_will: Option<LastWillConfig>,

    /// MQTT protocol version
    pub protocol_version: MqttProtocolVersion,

    /// Session expiry interval (MQTT 5.0 only, seconds)
    pub session_expiry_interval: Option<u32>,

    /// Maximum packet size (MQTT 5.0 only, bytes)
    pub max_packet_size: Option<u32>,

    /// Request/response information (MQTT 5.0)
    pub request_response_info: bool,

    /// Request problem information (MQTT 5.0)
    pub request_problem_info: bool,
}

impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            broker_url: "tcp://localhost:1883".to_string(),
            client_id: format!("oxirs-mqtt-{}", uuid::Uuid::new_v4()),
            clean_session: true,
            keep_alive_secs: 60,
            default_qos: QoS::AtLeastOnce,
            username: None,
            password: None,
            tls: None,
            reconnect: MqttReconnectConfig::default(),
            last_will: None,
            protocol_version: MqttProtocolVersion::V311,
            session_expiry_interval: None,
            max_packet_size: None,
            request_response_info: false,
            request_problem_info: true,
        }
    }
}

/// MQTT Quality of Service levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QoS {
    /// At most once (fire and forget) - QoS 0
    AtMostOnce = 0,
    /// At least once (acknowledged delivery) - QoS 1
    AtLeastOnce = 1,
    /// Exactly once (assured delivery) - QoS 2
    ExactlyOnce = 2,
}

impl From<QoS> for u8 {
    fn from(qos: QoS) -> Self {
        qos as u8
    }
}

impl TryFrom<u8> for QoS {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(QoS::AtMostOnce),
            1 => Ok(QoS::AtLeastOnce),
            2 => Ok(QoS::ExactlyOnce),
            _ => Err(format!("Invalid QoS level: {}", value)),
        }
    }
}

/// MQTT Protocol Version
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MqttProtocolVersion {
    /// MQTT 3.1.1
    V311,
    /// MQTT 5.0
    V5,
}

/// TLS Configuration for MQTT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttTlsConfig {
    /// CA certificate path (PEM format)
    pub ca_cert_path: Option<String>,

    /// Client certificate path (PEM format)
    pub client_cert_path: Option<String>,

    /// Client private key path (PEM format)
    pub client_key_path: Option<String>,

    /// Skip certificate verification (insecure, for testing only)
    pub insecure_skip_verify: bool,

    /// ALPN protocols
    pub alpn_protocols: Vec<String>,
}

impl Default for MqttTlsConfig {
    fn default() -> Self {
        Self {
            ca_cert_path: None,
            client_cert_path: None,
            client_key_path: None,
            insecure_skip_verify: false,
            alpn_protocols: vec!["mqtt".to_string()],
        }
    }
}

/// Reconnect Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttReconnectConfig {
    /// Initial delay before first reconnect attempt (milliseconds)
    pub initial_delay_ms: u64,

    /// Maximum delay between reconnect attempts (milliseconds)
    pub max_delay_ms: u64,

    /// Exponential backoff factor
    pub backoff_factor: f64,

    /// Maximum number of reconnect attempts (0 = infinite)
    pub max_attempts: u32,

    /// Jitter for reconnect delays (0.0 - 1.0)
    pub jitter: f64,
}

impl Default for MqttReconnectConfig {
    fn default() -> Self {
        Self {
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_factor: 2.0,
            max_attempts: 0,
            jitter: 0.1,
        }
    }
}

/// Last Will and Testament Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LastWillConfig {
    /// LWT topic
    pub topic: String,

    /// LWT message payload
    pub payload: Vec<u8>,

    /// LWT QoS
    pub qos: QoS,

    /// LWT retain flag
    pub retain: bool,

    /// Will delay interval (MQTT 5.0 only, seconds)
    pub will_delay_interval: Option<u32>,

    /// Message expiry interval (MQTT 5.0 only, seconds)
    pub message_expiry_interval: Option<u32>,

    /// Content type (MQTT 5.0 only)
    pub content_type: Option<String>,

    /// Response topic (MQTT 5.0 only)
    pub response_topic: Option<String>,

    /// User properties (MQTT 5.0 only)
    pub user_properties: HashMap<String, String>,
}

/// Topic Subscription with RDF Mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicSubscription {
    /// MQTT topic pattern (supports wildcards: +, #)
    /// Examples: "factory/+/sensor/#", "plant/building1/temperature"
    pub topic_pattern: String,

    /// QoS for this subscription
    pub qos: QoS,

    /// Payload format for parsing
    pub payload_format: PayloadFormat,

    /// RDF mapping configuration
    pub rdf_mapping: TopicRdfMapping,

    /// Subscription options (MQTT 5.0)
    pub options: Option<SubscriptionOptions>,
}

/// Payload Format Types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PayloadFormat {
    /// JSON payload with optional schema
    Json {
        schema: Option<String>,
        root_path: Option<String>,
    },

    /// Eclipse Sparkplug B (Industry 4.0 standard)
    SparkplugB {
        /// Sparkplug namespace (spBv1.0)
        namespace: String,
    },

    /// Protocol Buffers
    Protobuf {
        /// Protobuf schema definition
        schema: String,
        /// Message type name
        message_type: String,
    },

    /// Apache Avro
    Avro {
        /// Avro schema (JSON)
        schema: String,
    },

    /// CSV format
    Csv {
        /// CSV delimiter
        delimiter: char,
        /// Column headers
        headers: Vec<String>,
        /// Skip header row
        skip_header: bool,
    },

    /// Plain text (single value)
    PlainText {
        /// Value datatype (xsd:string, xsd:double, etc.)
        datatype: String,
    },

    /// Raw bytes (base64 encoded)
    Raw,
}

/// Topic to RDF Mapping Rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicRdfMapping {
    /// Subject URI pattern
    ///
    /// Supports placeholders:
    ///   - `{topic.0}`, `{topic.1}`, ... for topic segments
    ///   - `{payload.field}` for payload fields
    ///
    /// Example: `"urn:factory:{topic.0}:sensor:{topic.2}"`
    pub subject_pattern: String,

    /// Predicate mappings (payload field -> RDF predicate URI)
    pub predicate_map: HashMap<String, String>,

    /// Named graph pattern (optional)
    pub graph_pattern: Option<String>,

    /// rdf:type for the entity
    pub type_uri: Option<String>,

    /// Timestamp field in payload (if any)
    pub timestamp_field: Option<String>,

    /// Timestamp predicate URI (default: sosa:resultTime)
    pub timestamp_predicate: Option<String>,

    /// Value transformation rules
    pub transformations: Vec<ValueTransformation>,
}

/// Value Transformation Rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueTransformation {
    /// Field to transform
    pub field: String,

    /// Transformation type
    pub operation: TransformOperation,
}

/// Transformation Operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TransformOperation {
    /// Scale by factor
    Scale { factor: f64 },

    /// Add offset
    Offset { value: f64 },

    /// Unit conversion
    UnitConversion { from: String, to: String },

    /// Apply formula
    Formula { expression: String },

    /// Lookup table
    LookupTable { table: HashMap<String, String> },

    /// Regex replace
    RegexReplace {
        pattern: String,
        replacement: String,
    },
}

/// Subscription Options (MQTT 5.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionOptions {
    /// No local flag
    pub no_local: bool,

    /// Retain as published flag
    pub retain_as_published: bool,

    /// Retain handling
    pub retain_handling: RetainHandling,
}

/// Retain Handling (MQTT 5.0)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetainHandling {
    /// Send retained messages at subscribe
    SendAtSubscribe = 0,
    /// Send retained messages at subscribe if new subscription
    SendAtSubscribeNew = 1,
    /// Don't send retained messages
    DontSend = 2,
}

/// MQTT Statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MqttStats {
    /// Messages published
    pub messages_published: u64,

    /// Messages received
    pub messages_received: u64,

    /// Bytes sent
    pub bytes_sent: u64,

    /// Bytes received
    pub bytes_received: u64,

    /// Connection count
    pub connection_count: u64,

    /// Reconnection count
    pub reconnection_count: u64,

    /// Last connection time
    pub last_connected_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Last disconnection time
    pub last_disconnected_at: Option<chrono::DateTime<chrono::Utc>>,

    /// QoS 0 messages
    pub qos0_count: u64,

    /// QoS 1 messages
    pub qos1_count: u64,

    /// QoS 2 messages
    pub qos2_count: u64,

    /// Publish failures
    pub publish_failures: u64,

    /// Subscribe failures
    pub subscribe_failures: u64,
}

/// MQTT Message with metadata
#[derive(Debug, Clone)]
pub struct MqttMessage {
    /// Topic name
    pub topic: String,

    /// QoS level
    pub qos: QoS,

    /// Retain flag
    pub retain: bool,

    /// Payload
    pub payload: Vec<u8>,

    /// Message properties (MQTT 5.0)
    pub properties: Option<MqttMessageProperties>,

    /// Reception timestamp
    pub received_at: chrono::DateTime<chrono::Utc>,
}

/// MQTT 5.0 Message Properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttMessageProperties {
    /// Payload format indicator (0 = bytes, 1 = UTF-8)
    pub payload_format_indicator: Option<u8>,

    /// Message expiry interval (seconds)
    pub message_expiry_interval: Option<u32>,

    /// Topic alias
    pub topic_alias: Option<u16>,

    /// Response topic
    pub response_topic: Option<String>,

    /// Correlation data
    pub correlation_data: Option<Vec<u8>>,

    /// User properties
    pub user_properties: HashMap<String, String>,

    /// Subscription identifier
    pub subscription_identifier: Option<u32>,

    /// Content type
    pub content_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qos_conversion() {
        assert_eq!(u8::from(QoS::AtMostOnce), 0);
        assert_eq!(u8::from(QoS::AtLeastOnce), 1);
        assert_eq!(u8::from(QoS::ExactlyOnce), 2);

        assert_eq!(QoS::try_from(0).unwrap(), QoS::AtMostOnce);
        assert_eq!(QoS::try_from(1).unwrap(), QoS::AtLeastOnce);
        assert_eq!(QoS::try_from(2).unwrap(), QoS::ExactlyOnce);
        assert!(QoS::try_from(3).is_err());
    }

    #[test]
    fn test_default_config() {
        let config = MqttConfig::default();
        assert!(config.broker_url.starts_with("tcp://"));
        assert!(!config.client_id.is_empty());
        assert_eq!(config.default_qos, QoS::AtLeastOnce);
        assert_eq!(config.keep_alive_secs, 60);
    }

    #[test]
    fn test_default_stats() {
        let stats = MqttStats::default();
        assert_eq!(stats.messages_published, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.connection_count, 0);
    }
}
