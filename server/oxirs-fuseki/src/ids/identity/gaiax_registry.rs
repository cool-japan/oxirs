//! Gaia-X Trust Framework Integration
//!
//! Implements Gaia-X participant verification and Self-Description handling.
//! <https://gaia-x.eu/>

use crate::ids::types::{IdsError, IdsResult, IdsUri};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Gaia-X Registry Client
pub struct GaiaxRegistry {
    /// Registry URL (e.g., Gaia-X Compliance Service)
    registry_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// Cache for verified participants
    participant_cache: Arc<RwLock<ParticipantCache>>,
    /// Cache TTL in seconds
    cache_ttl: i64,
}

/// Cache for participant verification results
struct ParticipantCache {
    entries: HashMap<String, CachedParticipant>,
}

struct CachedParticipant {
    verified: bool,
    verified_at: DateTime<Utc>,
    self_description: Option<GaiaxSelfDescription>,
}

impl GaiaxRegistry {
    /// Create a new Gaia-X Registry client
    pub fn new(registry_url: impl Into<String>) -> Self {
        Self {
            registry_url: registry_url.into(),
            client: reqwest::Client::new(),
            participant_cache: Arc::new(RwLock::new(ParticipantCache {
                entries: HashMap::new(),
            })),
            cache_ttl: 3600, // 1 hour default
        }
    }

    /// Create with custom cache TTL
    pub fn with_cache_ttl(registry_url: impl Into<String>, ttl_seconds: i64) -> Self {
        Self {
            registry_url: registry_url.into(),
            client: reqwest::Client::new(),
            participant_cache: Arc::new(RwLock::new(ParticipantCache {
                entries: HashMap::new(),
            })),
            cache_ttl: ttl_seconds,
        }
    }

    /// Get registry URL
    pub fn registry_url(&self) -> &str {
        &self.registry_url
    }

    /// Verify a Gaia-X participant
    pub async fn verify_participant(&self, participant_id: &str) -> IdsResult<bool> {
        // Check cache first
        {
            let cache = self.participant_cache.read().await;
            if let Some(cached) = cache.entries.get(participant_id) {
                let age = Utc::now() - cached.verified_at;
                if age.num_seconds() < self.cache_ttl {
                    return Ok(cached.verified);
                }
            }
        }

        // Fetch and verify from registry
        let result = self.fetch_and_verify(participant_id).await;

        // Update cache
        let verified = result.as_ref().copied().unwrap_or(false);
        {
            let mut cache = self.participant_cache.write().await;
            cache.entries.insert(
                participant_id.to_string(),
                CachedParticipant {
                    verified,
                    verified_at: Utc::now(),
                    self_description: None,
                },
            );
        }

        result
    }

    /// Fetch and verify participant from registry
    async fn fetch_and_verify(&self, participant_id: &str) -> IdsResult<bool> {
        // Try to fetch Self-Description from participant's endpoint or registry
        let sd_result = self.get_self_description(participant_id).await;

        match sd_result {
            Ok(sd) => {
                // Verify the Self-Description
                let verification = self.verify_self_description(&sd).await?;
                Ok(verification.compliant)
            }
            Err(_) => {
                // If we can't fetch SD, try compliance service directly
                self.check_compliance_service(participant_id).await
            }
        }
    }

    /// Get Self-Description for a participant
    pub async fn get_self_description(
        &self,
        participant_id: &str,
    ) -> IdsResult<GaiaxSelfDescription> {
        // First try the participant's own endpoint
        let participant_url = if participant_id.starts_with("http") {
            format!("{}/.well-known/participant.json", participant_id)
        } else {
            // Try registry lookup
            format!("{}/api/participants/{}", self.registry_url, participant_id)
        };

        let response = self
            .client
            .get(&participant_url)
            .header("Accept", "application/ld+json")
            .send()
            .await
            .map_err(|e| {
                IdsError::InternalError(format!("Failed to fetch Self-Description: {}", e))
            })?;

        if !response.status().is_success() {
            return Err(IdsError::TrustVerificationFailed(format!(
                "Failed to fetch Self-Description for {}: {}",
                participant_id,
                response.status()
            )));
        }

        let sd: GaiaxSelfDescription = response.json().await.map_err(|e| {
            IdsError::SerializationError(format!("Failed to parse Self-Description: {}", e))
        })?;

        // Update cache with Self-Description
        {
            let mut cache = self.participant_cache.write().await;
            if let Some(entry) = cache.entries.get_mut(participant_id) {
                entry.self_description = Some(sd.clone());
            }
        }

        Ok(sd)
    }

    /// Verify a Self-Description against Gaia-X Compliance Service
    pub async fn verify_self_description(
        &self,
        sd: &GaiaxSelfDescription,
    ) -> IdsResult<ComplianceResult> {
        let compliance_url = format!("{}/api/compliance", self.registry_url);

        let response = self
            .client
            .post(&compliance_url)
            .header("Content-Type", "application/ld+json")
            .json(sd)
            .send()
            .await
            .map_err(|e| {
                IdsError::InternalError(format!("Failed to verify Self-Description: {}", e))
            })?;

        if !response.status().is_success() {
            // Non-compliant or error
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            return Ok(ComplianceResult {
                compliant: false,
                errors: vec![format!(
                    "Compliance check failed: {} - {}",
                    status, error_text
                )],
                warnings: vec![],
                verified_at: Utc::now(),
            });
        }

        // Parse compliance response
        let result: ComplianceResponse = response.json().await.map_err(|e| {
            IdsError::SerializationError(format!("Failed to parse compliance response: {}", e))
        })?;

        Ok(ComplianceResult {
            compliant: result.compliant,
            errors: result.errors.unwrap_or_default(),
            warnings: result.warnings.unwrap_or_default(),
            verified_at: Utc::now(),
        })
    }

    /// Check participant against Gaia-X Compliance Service
    async fn check_compliance_service(&self, participant_id: &str) -> IdsResult<bool> {
        let check_url = format!(
            "{}/api/participants/{}/compliance",
            self.registry_url, participant_id
        );

        let response = self.client.get(&check_url).send().await.map_err(|e| {
            IdsError::InternalError(format!("Failed to check compliance service: {}", e))
        })?;

        if response.status().is_success() {
            let result: ComplianceResponse = response.json().await.map_err(|e| {
                IdsError::SerializationError(format!("Failed to parse compliance response: {}", e))
            })?;
            Ok(result.compliant)
        } else {
            Ok(false)
        }
    }

    /// Clear participant cache
    pub async fn clear_cache(&self) {
        let mut cache = self.participant_cache.write().await;
        cache.entries.clear();
    }

    /// Get cached Self-Description if available
    pub async fn get_cached_self_description(
        &self,
        participant_id: &str,
    ) -> Option<GaiaxSelfDescription> {
        let cache = self.participant_cache.read().await;
        cache
            .entries
            .get(participant_id)
            .and_then(|e| e.self_description.clone())
    }
}

/// Gaia-X Self-Description
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GaiaxSelfDescription {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    /// Self-Description ID
    #[serde(rename = "@id")]
    pub id: String,

    /// Type
    #[serde(rename = "@type")]
    pub sd_type: Vec<String>,

    /// Credential subject (the participant description)
    #[serde(default)]
    pub credential_subject: Option<ParticipantCredentialSubject>,

    /// Verifiable credentials
    #[serde(default)]
    pub verifiable_credential: Option<Vec<serde_json::Value>>,

    /// Proof
    #[serde(default)]
    pub proof: Option<GaiaxProof>,
}

/// Participant credential subject
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParticipantCredentialSubject {
    /// Participant ID
    #[serde(rename = "@id")]
    pub id: String,

    /// Legal name
    #[serde(default)]
    pub legal_name: Option<String>,

    /// Legal registration number
    #[serde(default)]
    pub legal_registration_number: Option<Vec<LegalRegistration>>,

    /// Headquarters address
    #[serde(default)]
    pub headquarters_address: Option<Address>,

    /// Legal address
    #[serde(default)]
    pub legal_address: Option<Address>,

    /// Terms and conditions
    #[serde(default)]
    pub terms_and_conditions: Option<Vec<TermsAndConditions>>,

    /// Data protection officer contact
    #[serde(default)]
    pub data_protection_contact: Option<String>,
}

/// Legal registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegalRegistration {
    /// Registration type (e.g., "vatID", "EUID", "EORI")
    #[serde(rename = "@type")]
    pub reg_type: Option<String>,

    /// Registration number
    pub number: String,

    /// Issuing authority
    #[serde(default)]
    pub issuing_authority: Option<String>,
}

/// Address
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Address {
    /// Country code
    pub country_code: String,

    /// Locality
    #[serde(default)]
    pub locality: Option<String>,

    /// Postal code
    #[serde(default)]
    pub postal_code: Option<String>,

    /// Street address
    #[serde(default)]
    pub street_address: Option<String>,
}

/// Terms and Conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TermsAndConditions {
    /// URL to terms
    pub url: String,

    /// Hash of terms document
    #[serde(default)]
    pub hash: Option<String>,
}

/// Gaia-X Proof
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GaiaxProof {
    /// Proof type
    #[serde(rename = "type")]
    pub proof_type: String,

    /// Created timestamp
    pub created: DateTime<Utc>,

    /// Verification method
    pub verification_method: String,

    /// Proof purpose
    pub proof_purpose: String,

    /// JWS signature
    #[serde(default)]
    pub jws: Option<String>,
}

/// Compliance check response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComplianceResponse {
    compliant: bool,
    #[serde(default)]
    errors: Option<Vec<String>>,
    #[serde(default)]
    warnings: Option<Vec<String>>,
}

/// Compliance verification result
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    /// Is the participant compliant
    pub compliant: bool,
    /// Errors that caused non-compliance
    pub errors: Vec<String>,
    /// Warnings (participant is compliant but with issues)
    pub warnings: Vec<String>,
    /// When verification was performed
    pub verified_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaiax_registry_creation() {
        let registry = GaiaxRegistry::new("https://compliance.gaia-x.eu");
        assert_eq!(registry.registry_url(), "https://compliance.gaia-x.eu");
    }

    #[test]
    fn test_gaiax_registry_with_custom_ttl() {
        let registry = GaiaxRegistry::with_cache_ttl("https://compliance.gaia-x.eu", 7200);
        assert_eq!(registry.cache_ttl, 7200);
    }

    #[tokio::test]
    async fn test_participant_cache() {
        let registry = GaiaxRegistry::new("https://compliance.gaia-x.eu");

        // Manually insert into cache for testing
        {
            let mut cache = registry.participant_cache.write().await;
            cache.entries.insert(
                "test-participant".to_string(),
                CachedParticipant {
                    verified: true,
                    verified_at: Utc::now(),
                    self_description: None,
                },
            );
        }

        // Should return cached result
        // Note: This would fail in real use because we're not mocking HTTP
        // In production tests, use a mock server

        // Clear cache
        registry.clear_cache().await;

        let cache = registry.participant_cache.read().await;
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn test_self_description_parsing() {
        let json = r#"{
            "@context": ["https://www.w3.org/2018/credentials/v1", "https://registry.gaia-x.eu/v1/api"],
            "@id": "https://example.org/participant/123",
            "@type": ["VerifiablePresentation", "gx:LegalParticipant"],
            "credentialSubject": {
                "@id": "https://example.org/participant/123",
                "legalName": "Example Corp",
                "headquartersAddress": {
                    "countryCode": "DE"
                }
            }
        }"#;

        let sd: Result<GaiaxSelfDescription, _> = serde_json::from_str(json);
        assert!(sd.is_ok());

        let sd = sd.expect("parse SD");
        assert_eq!(sd.id, "https://example.org/participant/123");
        assert!(sd.sd_type.contains(&"gx:LegalParticipant".to_string()));

        if let Some(subject) = sd.credential_subject {
            assert_eq!(subject.legal_name, Some("Example Corp".to_string()));
        }
    }

    #[test]
    fn test_compliance_result() {
        let result = ComplianceResult {
            compliant: true,
            errors: vec![],
            warnings: vec!["Minor issue".to_string()],
            verified_at: Utc::now(),
        };

        assert!(result.compliant);
        assert!(result.errors.is_empty());
        assert_eq!(result.warnings.len(), 1);
    }
}
