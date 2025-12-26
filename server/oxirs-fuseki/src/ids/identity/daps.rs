//! Dynamic Attribute Provisioning Service (DAPS) Client
//!
//! Authenticates IDS connectors and issues security tokens

use crate::ids::types::{IdsError, IdsResult, IdsUri};
use base64::Engine;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// DAPS Client
pub struct DapsClient {
    daps_url: String,
}

impl DapsClient {
    pub fn new(daps_url: impl Into<String>) -> Self {
        Self {
            daps_url: daps_url.into(),
        }
    }

    pub async fn get_token(&self, connector_id: &IdsUri) -> IdsResult<DapsToken> {
        // Build DAPS token request
        let request_body = serde_json::json!({
            "grant_type": "client_credentials",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": self.create_client_assertion(connector_id)?,
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
            let error_text = response.text().await.unwrap_or_default();
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
            expires_at: Utc::now() + chrono::Duration::seconds(token_response.expires_in as i64),
            scope: token_response
                .scope
                .split_whitespace()
                .map(String::from)
                .collect(),
        })
    }

    /// Create JWT client assertion for DAPS
    fn create_client_assertion(&self, connector_id: &IdsUri) -> IdsResult<String> {
        // TODO: Sign JWT with connector's private key
        // For now, return a placeholder
        let encoded = base64::engine::general_purpose::STANDARD.encode(connector_id.as_str());
        Ok(format!(
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.{}.signature",
            encoded
        ))
    }

    /// Validate DAPS token
    pub fn validate_token(&self, token: &str) -> IdsResult<DapsTokenClaims> {
        // TODO: Verify JWT signature with DAPS public key
        // TODO: Check expiration
        // TODO: Validate issuer

        // For now, return mock claims
        Ok(DapsTokenClaims {
            sub: "urn:ids:connector:example".to_string(),
            iss: self.daps_url.clone(),
            aud: vec!["idsc:IDS_CONNECTORS_ALL".to_string()],
            exp: (Utc::now() + chrono::Duration::hours(1)).timestamp(),
            iat: Utc::now().timestamp(),
            scope: vec!["idsc:IDS_CONNECTOR_ATTRIBUTES_ALL".to_string()],
            security_profile: "idsc:BASE_SECURITY_PROFILE".to_string(),
        })
    }
}

/// DAPS Token Response (from DAPS server)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DapsTokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64, // seconds
    pub scope: String,   // space-separated
}

/// DAPS Token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DapsToken {
    pub access_token: String,
    pub token_type: String,
    pub expires_at: DateTime<Utc>,
    pub scope: Vec<String>,
}

impl DapsToken {
    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Get time until expiration
    pub fn time_until_expiry(&self) -> chrono::Duration {
        self.expires_at - Utc::now()
    }
}

/// DAPS Token Claims (JWT payload)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DapsTokenClaims {
    /// Subject (connector ID)
    pub sub: String,

    /// Issuer (DAPS URL)
    pub iss: String,

    /// Audience
    pub aud: Vec<String>,

    /// Expiration time (Unix timestamp)
    pub exp: i64,

    /// Issued at (Unix timestamp)
    pub iat: i64,

    /// Scopes granted
    pub scope: Vec<String>,

    /// Security profile
    #[serde(rename = "securityProfile")]
    pub security_profile: String,
}
