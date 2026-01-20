//! Dynamic Attribute Provisioning Service (DAPS) Client
//!
//! Authenticates IDS connectors and issues security tokens.
//! Implements IDSA DAPS specification for Dynamic Attribute Token (DAT) handling.

use crate::ids::types::{IdsError, IdsResult, IdsUri, SecurityProfile};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{
    decode, encode, Algorithm, DecodingKey, EncodingKey, Header, TokenData, Validation,
};
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// DAPS Client for authenticating with Dynamic Attribute Provisioning Service
pub struct DapsClient {
    /// DAPS server URL
    daps_url: String,
    /// Client credentials (optional, for testing without actual DAPS)
    credentials: Option<DapsCredentials>,
    /// Cached DAPS public key
    daps_public_key: Arc<RwLock<Option<Vec<u8>>>>,
}

/// DAPS Client Credentials
pub struct DapsCredentials {
    /// Connector ID
    connector_id: IdsUri,
    /// Ed25519 key pair for signing assertions
    key_pair: Ed25519KeyPair,
}

impl DapsCredentials {
    /// Create new credentials with generated key pair
    pub fn new(connector_id: IdsUri) -> IdsResult<Self> {
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|e| IdsError::InternalError(format!("Failed to generate key pair: {}", e)))?;

        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|e| IdsError::InternalError(format!("Failed to parse key pair: {}", e)))?;

        Ok(Self {
            connector_id,
            key_pair,
        })
    }

    /// Create from existing PKCS#8 key
    pub fn from_pkcs8(connector_id: IdsUri, pkcs8_bytes: &[u8]) -> IdsResult<Self> {
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes)
            .map_err(|e| IdsError::InternalError(format!("Failed to parse key pair: {}", e)))?;

        Ok(Self {
            connector_id,
            key_pair,
        })
    }

    /// Get public key bytes
    pub fn public_key(&self) -> &[u8] {
        self.key_pair.public_key().as_ref()
    }
}

impl DapsClient {
    /// Create a new DAPS client
    pub fn new(daps_url: impl Into<String>) -> Self {
        Self {
            daps_url: daps_url.into(),
            credentials: None,
            daps_public_key: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a DAPS client with credentials
    pub fn with_credentials(daps_url: impl Into<String>, credentials: DapsCredentials) -> Self {
        Self {
            daps_url: daps_url.into(),
            credentials: Some(credentials),
            daps_public_key: Arc::new(RwLock::new(None)),
        }
    }

    /// Get DAPS URL
    pub fn daps_url(&self) -> &str {
        &self.daps_url
    }

    /// Request a Dynamic Attribute Token (DAT) from DAPS
    pub async fn get_token(&self, connector_id: &IdsUri) -> IdsResult<DapsToken> {
        // Build DAPS token request
        let client_assertion = self.create_client_assertion(connector_id)?;

        let request_body = serde_json::json!({
            "grant_type": "client_credentials",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": client_assertion,
            "scope": "idsc:IDS_CONNECTOR_ATTRIBUTES_ALL"
        });

        // Send HTTP POST to DAPS
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/token", self.daps_url))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| IdsError::DapsAuthFailed(format!("DAPS request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(IdsError::DapsAuthFailed(format!(
                "DAPS returned {}: {}",
                status, error_text
            )));
        }

        let token_response: DapsTokenResponse = response.json().await.map_err(|e| {
            IdsError::DapsAuthFailed(format!("Failed to parse DAPS response: {}", e))
        })?;

        Ok(DapsToken {
            access_token: token_response.access_token,
            token_type: token_response.token_type,
            expires_at: Utc::now() + Duration::seconds(token_response.expires_in as i64),
            scope: token_response
                .scope
                .split_whitespace()
                .map(String::from)
                .collect(),
        })
    }

    /// Create JWT client assertion for DAPS authentication
    fn create_client_assertion(&self, connector_id: &IdsUri) -> IdsResult<String> {
        let now = Utc::now();
        let exp = now + Duration::minutes(5); // Short-lived assertion

        let claims = ClientAssertionClaims {
            iss: connector_id.as_str().to_string(),
            sub: connector_id.as_str().to_string(),
            aud: self.daps_url.clone(),
            jti: Uuid::new_v4().to_string(),
            iat: now.timestamp(),
            exp: exp.timestamp(),
            nbf: now.timestamp(),
        };

        // If we have credentials, use them to sign
        if let Some(ref credentials) = self.credentials {
            // Sign with Ed25519 (convert to EdDSA JWT)
            let header = Header {
                alg: Algorithm::EdDSA,
                ..Default::default()
            };

            // Use the key pair's private key for signing
            // Note: jsonwebtoken doesn't directly support ring's Ed25519, so we use a workaround
            // In production, you'd use PEM-encoded keys
            let encoding_key = EncodingKey::from_ed_der(credentials.key_pair.public_key().as_ref());

            encode(&header, &claims, &encoding_key).map_err(|e| {
                IdsError::DapsAuthFailed(format!("Failed to create client assertion: {}", e))
            })
        } else {
            // Fallback: Create unsigned assertion for testing/development
            // This would not be accepted by a real DAPS server
            let header = Header {
                alg: Algorithm::HS256,
                ..Default::default()
            };

            // Use a placeholder key for development
            let encoding_key = EncodingKey::from_secret(b"development-only-key");

            encode(&header, &claims, &encoding_key).map_err(|e| {
                IdsError::DapsAuthFailed(format!("Failed to create client assertion: {}", e))
            })
        }
    }

    /// Validate a DAPS Dynamic Attribute Token (DAT)
    pub fn validate_token(&self, token: &str) -> IdsResult<DapsTokenClaims> {
        self.validate_token_with_options(token, &TokenValidationOptions::default())
    }

    /// Validate a DAPS token with custom options
    pub fn validate_token_with_options(
        &self,
        token: &str,
        options: &TokenValidationOptions,
    ) -> IdsResult<DapsTokenClaims> {
        // Decode without verification first to get claims (for development/testing)
        // In production, you'd verify against DAPS public key
        let mut validation = Validation::new(Algorithm::RS256);
        validation.validate_exp = options.check_expiration;
        validation.validate_nbf = true;

        if let Some(ref expected_issuer) = options.expected_issuer {
            validation.set_issuer(&[expected_issuer.as_str()]);
        } else {
            validation.set_issuer(&[&self.daps_url]);
        }

        if let Some(ref expected_audience) = options.expected_audience {
            validation.set_audience(&[expected_audience.as_str()]);
        }

        // For development, decode without signature verification
        // In production, use DAPS public key
        let token_data: TokenData<DapsTokenClaims> = if options.skip_signature_verification {
            // Unsafe: Skip signature verification (development only!)
            let mut unsafe_validation = Validation::new(Algorithm::RS256);
            unsafe_validation.insecure_disable_signature_validation();
            unsafe_validation.validate_exp = options.check_expiration;

            decode(
                token,
                &DecodingKey::from_secret(b"unused"),
                &unsafe_validation,
            )
            .map_err(|e| IdsError::InvalidToken(format!("Failed to decode token: {}", e)))?
        } else {
            // Proper validation with public key
            let public_key = options.daps_public_key.as_ref().ok_or_else(|| {
                IdsError::InvalidToken("DAPS public key required for validation".to_string())
            })?;

            let decoding_key = DecodingKey::from_rsa_pem(public_key)
                .map_err(|e| IdsError::InvalidToken(format!("Invalid DAPS public key: {}", e)))?;

            decode(token, &decoding_key, &validation)
                .map_err(|e| IdsError::InvalidToken(format!("Token validation failed: {}", e)))?
        };

        let claims = token_data.claims;

        // Additional validation
        if options.check_expiration {
            let now = Utc::now().timestamp();
            if claims.exp < now {
                return Err(IdsError::InvalidToken("Token has expired".to_string()));
            }
        }

        // Validate issuer
        if options.validate_issuer && claims.iss != self.daps_url {
            return Err(IdsError::InvalidToken(format!(
                "Invalid issuer: expected {}, got {}",
                self.daps_url, claims.iss
            )));
        }

        Ok(claims)
    }

    /// Extract claims from token without validation (for debugging)
    pub fn decode_token_unverified(&self, token: &str) -> IdsResult<DapsTokenClaims> {
        let mut validation = Validation::new(Algorithm::RS256);
        validation.insecure_disable_signature_validation();
        validation.validate_exp = false;

        let token_data: TokenData<DapsTokenClaims> =
            decode(token, &DecodingKey::from_secret(b"unused"), &validation)
                .map_err(|e| IdsError::InvalidToken(format!("Failed to decode token: {}", e)))?;

        Ok(token_data.claims)
    }

    /// Fetch and cache DAPS public key
    pub async fn fetch_daps_public_key(&self) -> IdsResult<Vec<u8>> {
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{}/.well-known/jwks.json", self.daps_url))
            .send()
            .await
            .map_err(|e| {
                IdsError::DapsAuthFailed(format!("Failed to fetch DAPS public key: {}", e))
            })?;

        if !response.status().is_success() {
            return Err(IdsError::DapsAuthFailed(
                "Failed to fetch DAPS public key".to_string(),
            ));
        }

        let jwks: serde_json::Value = response.json().await.map_err(|e| {
            IdsError::DapsAuthFailed(format!("Failed to parse JWKS response: {}", e))
        })?;

        // Extract first key from JWKS
        let key = jwks
            .get("keys")
            .and_then(|k| k.get(0))
            .ok_or_else(|| IdsError::DapsAuthFailed("No keys in JWKS".to_string()))?;

        // Convert to PEM format (simplified - in production use proper JWKS parsing)
        let key_bytes = serde_json::to_vec(key)
            .map_err(|e| IdsError::DapsAuthFailed(format!("Failed to serialize key: {}", e)))?;

        // Cache the key
        let mut cached = self.daps_public_key.write().await;
        *cached = Some(key_bytes.clone());

        Ok(key_bytes)
    }
}

/// Token validation options
#[derive(Debug, Clone)]
pub struct TokenValidationOptions {
    /// Check token expiration
    pub check_expiration: bool,
    /// Validate issuer claim
    pub validate_issuer: bool,
    /// Expected issuer (overrides DAPS URL)
    pub expected_issuer: Option<String>,
    /// Expected audience
    pub expected_audience: Option<String>,
    /// Skip signature verification (development only!)
    pub skip_signature_verification: bool,
    /// DAPS public key for verification
    pub daps_public_key: Option<Vec<u8>>,
}

impl Default for TokenValidationOptions {
    fn default() -> Self {
        Self {
            check_expiration: true,
            validate_issuer: true,
            expected_issuer: None,
            expected_audience: None,
            skip_signature_verification: false,
            daps_public_key: None,
        }
    }
}

impl TokenValidationOptions {
    /// Create options for development (skip signature verification)
    pub fn development() -> Self {
        Self {
            check_expiration: false,
            validate_issuer: false,
            skip_signature_verification: true,
            ..Default::default()
        }
    }
}

/// Client assertion claims for DAPS authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClientAssertionClaims {
    /// Issuer (connector ID)
    iss: String,
    /// Subject (connector ID)
    sub: String,
    /// Audience (DAPS URL)
    aud: String,
    /// JWT ID (unique identifier)
    jti: String,
    /// Issued at
    iat: i64,
    /// Expiration
    exp: i64,
    /// Not before
    nbf: i64,
}

/// DAPS Token Response (from DAPS server)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DapsTokenResponse {
    /// Access token (the DAT)
    pub access_token: String,
    /// Token type (usually "Bearer")
    pub token_type: String,
    /// Expiration in seconds
    pub expires_in: u64,
    /// Scope (space-separated)
    pub scope: String,
}

/// DAPS Token (Dynamic Attribute Token)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DapsToken {
    /// The actual token string
    pub access_token: String,
    /// Token type
    pub token_type: String,
    /// Expiration time
    pub expires_at: DateTime<Utc>,
    /// Granted scopes
    pub scope: Vec<String>,
}

impl DapsToken {
    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Get time until expiration
    pub fn time_until_expiry(&self) -> Duration {
        self.expires_at - Utc::now()
    }

    /// Check if token will expire within given duration
    pub fn expires_within(&self, duration: Duration) -> bool {
        Utc::now() + duration > self.expires_at
    }
}

/// DAPS Token Claims (JWT payload for DAT)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DapsTokenClaims {
    /// Subject (connector ID)
    pub sub: String,

    /// Issuer (DAPS URL)
    pub iss: String,

    /// Audience
    #[serde(default)]
    pub aud: Vec<String>,

    /// Expiration time (Unix timestamp)
    pub exp: i64,

    /// Issued at (Unix timestamp)
    pub iat: i64,

    /// Not before (Unix timestamp, optional)
    #[serde(default)]
    pub nbf: Option<i64>,

    /// JWT ID
    #[serde(default)]
    pub jti: Option<String>,

    /// Scopes granted
    #[serde(default)]
    pub scope: Vec<String>,

    /// Security profile
    #[serde(rename = "securityProfile", default)]
    pub security_profile: String,

    /// Connector attributes (IDS-specific)
    #[serde(rename = "@type", default)]
    pub connector_type: Option<String>,

    /// Extended attributes
    #[serde(rename = "extendedGuarantee", default)]
    pub extended_guarantee: Option<String>,

    /// Transport certificate SHA256 fingerprint
    #[serde(rename = "transportCertsSha256", default)]
    pub transport_certs_sha256: Option<Vec<String>>,

    /// Referring connector
    #[serde(rename = "referringConnector", default)]
    pub referring_connector: Option<String>,
}

impl DapsTokenClaims {
    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        Utc::now().timestamp() > self.exp
    }

    /// Get security profile as enum
    pub fn get_security_profile(&self) -> SecurityProfile {
        match self.security_profile.as_str() {
            "idsc:BASE_SECURITY_PROFILE" | "BASE_SECURITY_PROFILE" => {
                SecurityProfile::BaseSecurityProfile
            }
            "idsc:TRUST_SECURITY_PROFILE" | "TRUST_SECURITY_PROFILE" => {
                SecurityProfile::TrustSecurityProfile
            }
            "idsc:TRUST_PLUS_SECURITY_PROFILE" | "TRUST_PLUS_SECURITY_PROFILE" => {
                SecurityProfile::TrustPlusSecurityProfile
            }
            _ => SecurityProfile::BaseSecurityProfile,
        }
    }

    /// Check if claims include required scope
    pub fn has_scope(&self, required_scope: &str) -> bool {
        self.scope.iter().any(|s| s == required_scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daps_client_creation() {
        let client = DapsClient::new("https://daps.example.org");
        assert_eq!(client.daps_url(), "https://daps.example.org");
    }

    #[test]
    fn test_daps_token_expiration() {
        let token = DapsToken {
            access_token: "test_token".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() + Duration::hours(1),
            scope: vec!["idsc:IDS_CONNECTOR_ATTRIBUTES_ALL".to_string()],
        };

        assert!(!token.is_expired());
        assert!(!token.expires_within(Duration::minutes(30)));
        assert!(token.expires_within(Duration::hours(2)));

        let expired_token = DapsToken {
            access_token: "test_token".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() - Duration::hours(1),
            scope: vec![],
        };

        assert!(expired_token.is_expired());
    }

    #[test]
    fn test_daps_claims_security_profile() {
        let claims = DapsTokenClaims {
            sub: "urn:ids:connector:example".to_string(),
            iss: "https://daps.example.org".to_string(),
            aud: vec!["idsc:IDS_CONNECTORS_ALL".to_string()],
            exp: (Utc::now() + Duration::hours(1)).timestamp(),
            iat: Utc::now().timestamp(),
            nbf: None,
            jti: None,
            scope: vec!["idsc:IDS_CONNECTOR_ATTRIBUTES_ALL".to_string()],
            security_profile: "idsc:TRUST_SECURITY_PROFILE".to_string(),
            connector_type: None,
            extended_guarantee: None,
            transport_certs_sha256: None,
            referring_connector: None,
        };

        assert_eq!(
            claims.get_security_profile(),
            SecurityProfile::TrustSecurityProfile
        );
        assert!(claims.has_scope("idsc:IDS_CONNECTOR_ATTRIBUTES_ALL"));
        assert!(!claims.has_scope("idsc:SOME_OTHER_SCOPE"));
    }

    #[test]
    fn test_client_assertion_creation() {
        let client = DapsClient::new("https://daps.example.org");
        let connector_id = IdsUri::new("urn:ids:connector:test").expect("valid URI");

        // Should create assertion without credentials (development mode)
        let assertion = client.create_client_assertion(&connector_id);
        assert!(assertion.is_ok());

        let token = assertion.expect("assertion");
        // Should be a valid JWT format (header.payload.signature)
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_token_validation_options() {
        let default_opts = TokenValidationOptions::default();
        assert!(default_opts.check_expiration);
        assert!(default_opts.validate_issuer);
        assert!(!default_opts.skip_signature_verification);

        let dev_opts = TokenValidationOptions::development();
        assert!(!dev_opts.check_expiration);
        assert!(!dev_opts.validate_issuer);
        assert!(dev_opts.skip_signature_verification);
    }

    #[test]
    fn test_credentials_creation() {
        let connector_id = IdsUri::new("urn:ids:connector:test").expect("valid URI");
        let credentials = DapsCredentials::new(connector_id);
        assert!(credentials.is_ok());

        let creds = credentials.expect("credentials");
        assert!(!creds.public_key().is_empty());
    }
}
