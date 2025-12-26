//! IDS Common Types
//!
//! Shared type definitions for IDS connector implementation

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// IDS URI type (ensures proper IDS identifier format)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdsUri(String);

impl IdsUri {
    /// Create a new IDS URI
    pub fn new(uri: impl Into<String>) -> Result<Self, IdsError> {
        let uri = uri.into();

        // Validate URI format
        if !uri.starts_with("http://") && !uri.starts_with("https://") && !uri.starts_with("urn:") {
            return Err(IdsError::InvalidUri(format!(
                "Invalid IDS URI format: {}",
                uri
            )));
        }

        Ok(Self(uri))
    }

    /// Get the URI string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to String
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for IdsUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<String> for IdsUri {
    type Error = IdsError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<IdsUri> for String {
    fn from(uri: IdsUri) -> Self {
        uri.0
    }
}

/// IDS Error types
#[derive(Debug, Error)]
pub enum IdsError {
    #[error("Invalid URI: {0}")]
    InvalidUri(String),

    #[error("Policy violation: {0}")]
    PolicyViolation(String),

    #[error("Contract negotiation failed: {0}")]
    NegotiationFailed(String),

    #[error("Contract not found: {0}")]
    ContractNotFound(String),

    #[error("Invalid contract state: expected {expected}, got {actual}")]
    InvalidContractState { expected: String, actual: String },

    #[error("DAPS authentication failed: {0}")]
    DapsAuthFailed(String),

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Trust verification failed: {0}")]
    TrustVerificationFailed(String),

    #[error("Data residency violation: {0}")]
    ResidencyViolation(String),

    #[error("GDPR compliance violation: {0}")]
    GdprViolation(String),

    #[error("Lineage tracking failed: {0}")]
    LineageTrackingFailed(String),

    #[error("Catalog error: {0}")]
    CatalogError(String),

    #[error("Message protocol error: {0}")]
    MessageProtocolError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl IdsError {
    /// Get HTTP status code for this error
    pub fn status_code(&self) -> u16 {
        match self {
            IdsError::InvalidUri(_) => 400,
            IdsError::PolicyViolation(_) => 403,
            IdsError::NegotiationFailed(_) => 422,
            IdsError::ContractNotFound(_) => 404,
            IdsError::InvalidContractState { .. } => 409,
            IdsError::DapsAuthFailed(_) => 401,
            IdsError::InvalidToken(_) => 401,
            IdsError::TrustVerificationFailed(_) => 403,
            IdsError::ResidencyViolation(_) => 451, // RFC 7725: Unavailable For Legal Reasons
            IdsError::GdprViolation(_) => 451,
            IdsError::LineageTrackingFailed(_) => 500,
            IdsError::CatalogError(_) => 500,
            IdsError::MessageProtocolError(_) => 400,
            IdsError::SerializationError(_) => 500,
            IdsError::InternalError(_) => 500,
        }
    }
}

/// Result type for IDS operations
pub type IdsResult<T> = Result<T, IdsError>;

/// Party in IDS context (data provider, consumer, broker)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Party {
    /// Party identifier (IDS URI)
    pub id: IdsUri,

    /// Party name
    pub name: String,

    /// Organization name
    pub organization: Option<String>,

    /// Contact information
    pub contact: Option<ContactInfo>,

    /// Gaia-X participant ID (optional)
    pub gaiax_participant_id: Option<String>,
}

/// Contact information for a party
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContactInfo {
    pub email: Option<String>,
    pub phone: Option<String>,
    pub address: Option<String>,
}

/// Security profile levels for IDS
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityProfile {
    /// Base security profile
    BaseSecurityProfile,

    /// Trust security profile
    TrustSecurityProfile,

    /// Trust+ security profile
    TrustPlusSecurityProfile,
}

impl SecurityProfile {
    /// Get the IDS URI for this security profile
    pub fn to_uri(&self) -> &'static str {
        match self {
            SecurityProfile::BaseSecurityProfile => {
                "https://w3id.org/idsa/code/BASE_SECURITY_PROFILE"
            }
            SecurityProfile::TrustSecurityProfile => {
                "https://w3id.org/idsa/code/TRUST_SECURITY_PROFILE"
            }
            SecurityProfile::TrustPlusSecurityProfile => {
                "https://w3id.org/idsa/code/TRUST_PLUS_SECURITY_PROFILE"
            }
        }
    }
}

/// Transfer protocol supported by IDS connector
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransferProtocol {
    /// HTTPS
    Https,

    /// IDSCP2 (IDS Communication Protocol v2)
    Idscp2,

    /// Multipart form data
    MultipartFormData,
}

impl TransferProtocol {
    /// Get the IDS URI for this protocol
    pub fn to_uri(&self) -> &'static str {
        match self {
            TransferProtocol::Https => "https://w3id.org/idsa/code/HTTPS",
            TransferProtocol::Idscp2 => "https://w3id.org/idsa/code/IDSCP2",
            TransferProtocol::MultipartFormData => "https://w3id.org/idsa/code/MULTIPART_FORM_DATA",
        }
    }
}

/// Adequacy decision status (GDPR Article 45)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AdequacyStatus {
    /// Adequate protection (European Commission decision)
    Adequate,

    /// No adequacy decision
    NotAdequate,

    /// Adequacy under review
    UnderReview,
}

/// Legal jurisdiction for data processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Jurisdiction {
    /// ISO 3166-1 alpha-2 country code
    pub country_code: String,

    /// Legal framework applicable
    pub legal_framework: Vec<String>,

    /// Data protection authority
    pub data_protection_authority: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ids_uri_creation() {
        let uri = IdsUri::new("https://example.org/resource/123").unwrap();
        assert_eq!(uri.as_str(), "https://example.org/resource/123");

        let uri = IdsUri::new("urn:ids:resource:456").unwrap();
        assert_eq!(uri.as_str(), "urn:ids:resource:456");

        let result = IdsUri::new("invalid-uri");
        assert!(result.is_err());
    }

    #[test]
    fn test_security_profile_uri() {
        assert_eq!(
            SecurityProfile::BaseSecurityProfile.to_uri(),
            "https://w3id.org/idsa/code/BASE_SECURITY_PROFILE"
        );
        assert_eq!(
            SecurityProfile::TrustPlusSecurityProfile.to_uri(),
            "https://w3id.org/idsa/code/TRUST_PLUS_SECURITY_PROFILE"
        );
    }

    #[test]
    fn test_security_profile_ordering() {
        assert!(SecurityProfile::BaseSecurityProfile < SecurityProfile::TrustSecurityProfile);
        assert!(SecurityProfile::TrustSecurityProfile < SecurityProfile::TrustPlusSecurityProfile);
    }

    #[test]
    fn test_transfer_protocol_uri() {
        assert_eq!(
            TransferProtocol::Https.to_uri(),
            "https://w3id.org/idsa/code/HTTPS"
        );
        assert_eq!(
            TransferProtocol::Idscp2.to_uri(),
            "https://w3id.org/idsa/code/IDSCP2"
        );
    }
}
