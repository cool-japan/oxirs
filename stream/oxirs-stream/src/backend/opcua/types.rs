//! OPC UA Backend Types
//!
//! OPC UA Unified Architecture support for industrial automation

use serde::{Deserialize, Serialize};

/// OPC UA Client Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcUaConfig {
    /// OPC UA endpoint URL (opc.tcp://host:4840)
    pub endpoint_url: String,

    /// Application name
    pub application_name: String,

    /// Application URI
    pub application_uri: String,

    /// Security policy
    pub security_policy: SecurityPolicy,

    /// Security mode
    pub security_mode: MessageSecurityMode,

    /// User identity
    pub user_identity: UserIdentity,

    /// Session timeout (milliseconds)
    pub session_timeout_ms: u64,

    /// Publishing interval for subscriptions (milliseconds)
    pub publishing_interval_ms: u32,

    /// Sampling interval for monitored items (milliseconds)
    pub sampling_interval_ms: u32,

    /// Queue size for monitored items
    pub queue_size: u32,

    /// Client certificate configuration
    pub client_certificate: Option<CertificateConfig>,

    /// Server certificate path or PEM
    pub server_certificate: Option<String>,

    /// Auto-accept untrusted certificates (insecure, for testing)
    pub accept_untrusted_certs: bool,

    /// Session renewal settings
    pub session_renewal: SessionRenewalConfig,
}

impl Default for OpcUaConfig {
    fn default() -> Self {
        Self {
            endpoint_url: "opc.tcp://localhost:4840".to_string(),
            application_name: format!("OxiRS-OPC-UA-{}", uuid::Uuid::new_v4()),
            application_uri: "urn:OxiRS:OpcUaClient".to_string(),
            security_policy: SecurityPolicy::None,
            security_mode: MessageSecurityMode::None,
            user_identity: UserIdentity::Anonymous,
            session_timeout_ms: 60000,
            publishing_interval_ms: 1000,
            sampling_interval_ms: 100,
            queue_size: 10,
            client_certificate: None,
            server_certificate: None,
            accept_untrusted_certs: false,
            session_renewal: SessionRenewalConfig::default(),
        }
    }
}

/// Security Policies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityPolicy {
    /// No security
    None,
    /// Basic128Rsa15 (deprecated, legacy only)
    Basic128Rsa15,
    /// Basic256 (deprecated, legacy only)
    Basic256,
    /// Basic256Sha256 (recommended minimum)
    Basic256Sha256,
    /// Aes128_Sha256_RsaOaep
    Aes128Sha256RsaOaep,
    /// Aes256_Sha256_RsaPss (highest security)
    Aes256Sha256RsaPss,
}

impl SecurityPolicy {
    /// Get the OPC UA security policy URI
    pub fn to_uri(&self) -> String {
        let base = "http://opcfoundation.org/UA/SecurityPolicy#";
        match self {
            SecurityPolicy::None => format!("{}None", base),
            SecurityPolicy::Basic128Rsa15 => format!("{}Basic128Rsa15", base),
            SecurityPolicy::Basic256 => format!("{}Basic256", base),
            SecurityPolicy::Basic256Sha256 => format!("{}Basic256Sha256", base),
            SecurityPolicy::Aes128Sha256RsaOaep => format!("{}Aes128_Sha256_RsaOaep", base),
            SecurityPolicy::Aes256Sha256RsaPss => format!("{}Aes256_Sha256_RsaPss", base),
        }
    }
}

/// Message Security Mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageSecurityMode {
    /// No security
    None,
    /// Sign only
    Sign,
    /// Sign and encrypt
    SignAndEncrypt,
}

/// User Identity Options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserIdentity {
    /// Anonymous (no authentication)
    Anonymous,

    /// Username and password
    UserPassword { username: String, password: String },

    /// X.509 certificate
    X509Certificate { cert_path: String, key_path: String },

    /// Issued token
    IssuedToken {
        token: Vec<u8>,
        encryption_algorithm: String,
    },
}

/// Certificate Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateConfig {
    /// Certificate path (PEM or DER)
    pub cert_path: String,

    /// Private key path (PEM or DER)
    pub key_path: String,

    /// Certificate format
    pub format: CertificateFormat,
}

/// Certificate Format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CertificateFormat {
    /// PEM format (base64 encoded)
    Pem,
    /// DER format (binary)
    Der,
}

/// Session Renewal Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRenewalConfig {
    /// Auto-renew session before expiry
    pub auto_renew: bool,

    /// Renew at percentage of timeout (0.0 - 1.0)
    pub renew_at_ratio: f64,

    /// Retry count for session renewal
    pub retry_count: u32,

    /// Retry delay (milliseconds)
    pub retry_delay_ms: u64,
}

impl Default for SessionRenewalConfig {
    fn default() -> Self {
        Self {
            auto_renew: true,
            renew_at_ratio: 0.75,
            retry_count: 3,
            retry_delay_ms: 5000,
        }
    }
}

/// Node Subscription Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSubscription {
    /// OPC UA Node ID (e.g., "ns=2;s=Temperature" or "ns=2;i=1001")
    pub node_id: String,

    /// Browse name (optional, for display)
    pub browse_name: Option<String>,

    /// Display name (optional, for display)
    pub display_name: Option<String>,

    /// RDF subject URI (entity this node belongs to)
    pub rdf_subject: String,

    /// RDF predicate URI (property this node represents)
    pub rdf_predicate: String,

    /// Named graph URI (optional)
    pub rdf_graph: Option<String>,

    /// Unit URI (QUDT, UCUM, or custom)
    pub unit_uri: Option<String>,

    /// SAMM property reference (optional)
    pub samm_property: Option<String>,

    /// Deadband configuration for value changes
    pub deadband: Option<Deadband>,

    /// Data change filter
    pub data_change_filter: Option<DataChangeFilter>,
}

/// Deadband Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deadband {
    /// Deadband type
    pub deadband_type: DeadbandType,

    /// Deadband value
    pub value: f64,
}

/// Deadband Type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeadbandType {
    /// No deadband
    None,
    /// Absolute deadband
    Absolute,
    /// Percent deadband
    Percent,
}

/// Data Change Filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataChangeFilter {
    /// Report on status change
    Status,
    /// Report on status or value change
    StatusValue,
    /// Report on status, value, or timestamp change
    StatusValueTimestamp,
}

/// OPC UA Data Change Event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcUaDataChange {
    /// Node ID
    pub node_id: String,

    /// Value
    pub value: OpcUaValue,

    /// Status code (good = 0x00000000)
    pub status_code: u32,

    /// Source timestamp (from device)
    pub source_timestamp: Option<chrono::DateTime<chrono::Utc>>,

    /// Server timestamp (from OPC UA server)
    pub server_timestamp: chrono::DateTime<chrono::Utc>,
}

/// OPC UA Value Types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpcUaValue {
    Boolean {
        value: bool,
    },
    SByte {
        value: i8,
    },
    Byte {
        value: u8,
    },
    Int16 {
        value: i16,
    },
    UInt16 {
        value: u16,
    },
    Int32 {
        value: i32,
    },
    UInt32 {
        value: u32,
    },
    Int64 {
        value: i64,
    },
    UInt64 {
        value: u64,
    },
    Float {
        value: f32,
    },
    Double {
        value: f64,
    },
    String {
        value: String,
    },
    DateTime {
        value: chrono::DateTime<chrono::Utc>,
    },
    Guid {
        value: String,
    },
    ByteString {
        value: Vec<u8>,
    },
    StatusCode {
        value: u32,
    },
    QualifiedName {
        namespace_index: u16,
        name: String,
    },
    LocalizedText {
        locale: Option<String>,
        text: String,
    },
}

impl OpcUaValue {
    /// Convert to RDF literal with datatype
    pub fn to_rdf_literal(&self) -> String {
        match self {
            OpcUaValue::Boolean { value } => {
                format!("\"{}\"^^<http://www.w3.org/2001/XMLSchema#boolean>", value)
            }
            OpcUaValue::Int32 { value: _ } | OpcUaValue::Int16 { value: _ } => {
                format!(
                    "\"{}\"^^<http://www.w3.org/2001/XMLSchema#integer>",
                    match self {
                        OpcUaValue::Int32 { value } => *value as i64,
                        OpcUaValue::Int16 { value } => *value as i64,
                        _ => 0,
                    }
                )
            }
            OpcUaValue::Float { value } => {
                format!("\"{}\"^^<http://www.w3.org/2001/XMLSchema#float>", value)
            }
            OpcUaValue::Double { value } => {
                format!("\"{}\"^^<http://www.w3.org/2001/XMLSchema#double>", value)
            }
            OpcUaValue::String { value } => format!("\"{}\"", value),
            OpcUaValue::DateTime { value } => {
                format!(
                    "\"{}\"^^<http://www.w3.org/2001/XMLSchema#dateTime>",
                    value.to_rfc3339()
                )
            }
            _ => format!("\"{}\"", serde_json::to_string(self).unwrap_or_default()),
        }
    }

    /// Get the XSD datatype URI
    pub fn xsd_datatype(&self) -> &'static str {
        match self {
            OpcUaValue::Boolean { .. } => "http://www.w3.org/2001/XMLSchema#boolean",
            OpcUaValue::Int32 { .. } | OpcUaValue::Int16 { .. } | OpcUaValue::Int64 { .. } => {
                "http://www.w3.org/2001/XMLSchema#integer"
            }
            OpcUaValue::UInt32 { .. } | OpcUaValue::UInt16 { .. } | OpcUaValue::UInt64 { .. } => {
                "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
            }
            OpcUaValue::Float { .. } => "http://www.w3.org/2001/XMLSchema#float",
            OpcUaValue::Double { .. } => "http://www.w3.org/2001/XMLSchema#double",
            OpcUaValue::String { .. } => "http://www.w3.org/2001/XMLSchema#string",
            OpcUaValue::DateTime { .. } => "http://www.w3.org/2001/XMLSchema#dateTime",
            _ => "http://www.w3.org/2001/XMLSchema#string",
        }
    }
}

/// OPC UA Statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpcUaStats {
    /// Data changes received
    pub data_changes_received: u64,

    /// Events received
    pub events_received: u64,

    /// Session count
    pub session_count: u64,

    /// Session renewal count
    pub session_renewals: u64,

    /// Subscription count
    pub subscription_count: u64,

    /// Monitored items count
    pub monitored_items_count: u64,

    /// Last connection time
    pub last_connected_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Last disconnection time
    pub last_disconnected_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Publish requests sent
    pub publish_requests: u64,

    /// Read operations
    pub read_operations: u64,

    /// Write operations
    pub write_operations: u64,

    /// Browse operations
    pub browse_operations: u64,

    /// Error count
    pub error_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_policy_uri() {
        assert_eq!(
            SecurityPolicy::None.to_uri(),
            "http://opcfoundation.org/UA/SecurityPolicy#None"
        );
        assert_eq!(
            SecurityPolicy::Basic256Sha256.to_uri(),
            "http://opcfoundation.org/UA/SecurityPolicy#Basic256Sha256"
        );
    }

    #[test]
    fn test_opcua_value_to_rdf() {
        let value = OpcUaValue::Double { value: 25.5 };
        let rdf = value.to_rdf_literal();
        assert!(rdf.contains("25.5"));
        assert!(rdf.contains("double"));

        let value = OpcUaValue::Boolean { value: true };
        let rdf = value.to_rdf_literal();
        assert!(rdf.contains("true"));
        assert!(rdf.contains("boolean"));
    }

    #[test]
    fn test_default_config() {
        let config = OpcUaConfig::default();
        assert!(config.endpoint_url.starts_with("opc.tcp://"));
        assert_eq!(config.security_policy, SecurityPolicy::None);
        assert!(matches!(config.user_identity, UserIdentity::Anonymous));
    }
}
