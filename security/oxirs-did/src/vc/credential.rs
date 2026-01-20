//! Verifiable Credential (W3C VC Data Model 2.0)

use crate::{Did, DidResult, Proof};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Verifiable Credential structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerifiableCredential {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    /// Credential identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Credential types
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,

    /// Credential issuer
    pub issuer: CredentialIssuerInfo,

    /// Issuance date (VC 1.1 compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issuance_date: Option<DateTime<Utc>>,

    /// Valid from (VC 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,

    /// Expiration date (VC 1.1 compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiration_date: Option<DateTime<Utc>>,

    /// Valid until (VC 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<DateTime<Utc>>,

    /// Credential subject(s)
    pub credential_subject: CredentialSubjectContainer,

    /// Cryptographic proof(s)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<ProofContainer>,

    /// Credential status (for revocation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_status: Option<CredentialStatus>,

    /// Terms of use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terms_of_use: Option<Vec<TermsOfUse>>,

    /// Evidence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<Evidence>>,

    /// Credential schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential_schema: Option<Vec<CredentialSchema>>,

    /// Refresh service
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_service: Option<RefreshService>,
}

/// Issuer can be a simple DID or an object with more info
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CredentialIssuerInfo {
    Did(Did),
    Object {
        id: Did,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image: Option<String>,
    },
}

impl CredentialIssuerInfo {
    pub fn did(&self) -> &Did {
        match self {
            CredentialIssuerInfo::Did(did) => did,
            CredentialIssuerInfo::Object { id, .. } => id,
        }
    }
}

/// Credential subject container (can be single or array)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CredentialSubjectContainer {
    Single(CredentialSubject),
    Multiple(Vec<CredentialSubject>),
}

impl CredentialSubjectContainer {
    pub fn subjects(&self) -> Vec<&CredentialSubject> {
        match self {
            CredentialSubjectContainer::Single(s) => vec![s],
            CredentialSubjectContainer::Multiple(v) => v.iter().collect(),
        }
    }
}

/// Credential subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialSubject {
    /// Subject identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Subject claims (dynamic properties)
    #[serde(flatten)]
    pub claims: HashMap<String, serde_json::Value>,
}

impl CredentialSubject {
    pub fn new(id: Option<&str>) -> Self {
        Self {
            id: id.map(String::from),
            claims: HashMap::new(),
        }
    }

    pub fn with_claim(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.claims.insert(key.to_string(), value.into());
        self
    }
}

/// Proof container (can be single or array)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProofContainer {
    Single(Box<Proof>),
    Multiple(Vec<Proof>),
}

impl ProofContainer {
    pub fn proofs(&self) -> Vec<&Proof> {
        match self {
            ProofContainer::Single(p) => vec![p],
            ProofContainer::Multiple(v) => v.iter().collect(),
        }
    }
}

/// Credential status for revocation checking
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialStatus {
    pub id: String,
    #[serde(rename = "type")]
    pub status_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_purpose: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_list_index: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_list_credential: Option<String>,
}

/// Terms of use
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TermsOfUse {
    #[serde(rename = "type")]
    pub terms_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Evidence {
    #[serde(rename = "type")]
    pub evidence_type: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Credential schema
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialSchema {
    pub id: String,
    #[serde(rename = "type")]
    pub schema_type: String,
}

/// Refresh service
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RefreshService {
    pub id: String,
    #[serde(rename = "type")]
    pub service_type: String,
}

impl VerifiableCredential {
    /// Create a new unsigned credential
    pub fn new(issuer: Did, subject: CredentialSubject, credential_types: Vec<String>) -> Self {
        let mut types = vec!["VerifiableCredential".to_string()];
        types.extend(credential_types);

        let now = Utc::now();

        Self {
            context: vec!["https://www.w3.org/ns/credentials/v2".to_string()],
            id: Some(format!("urn:uuid:{}", uuid::Uuid::new_v4())),
            credential_type: types,
            issuer: CredentialIssuerInfo::Did(issuer),
            issuance_date: Some(now),
            valid_from: Some(now),
            expiration_date: None,
            valid_until: None,
            credential_subject: CredentialSubjectContainer::Single(subject),
            proof: None,
            credential_status: None,
            terms_of_use: None,
            evidence: None,
            credential_schema: None,
            refresh_service: None,
        }
    }

    /// Set expiration date
    pub fn with_expiration(mut self, expires: DateTime<Utc>) -> Self {
        self.expiration_date = Some(expires);
        self.valid_until = Some(expires);
        self
    }

    /// Add additional context
    pub fn with_context(mut self, context: &str) -> Self {
        self.context.push(context.to_string());
        self
    }

    /// Check if credential is expired
    pub fn is_expired(&self) -> bool {
        let now = Utc::now();
        if let Some(exp) = self.expiration_date {
            return now > exp;
        }
        if let Some(until) = self.valid_until {
            return now > until;
        }
        false
    }

    /// Check if credential is not yet valid
    pub fn is_not_yet_valid(&self) -> bool {
        let now = Utc::now();
        if let Some(from) = self.valid_from {
            return now < from;
        }
        if let Some(issued) = self.issuance_date {
            return now < issued;
        }
        false
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> DidResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::DidError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> DidResult<Self> {
        serde_json::from_str(json).map_err(|e| crate::DidError::SerializationError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_credential() {
        let issuer = Did::new_key_ed25519(&[0u8; 32]).unwrap();
        let subject = CredentialSubject::new(Some("did:example:alice"))
            .with_claim("name", "Alice")
            .with_claim("age", 30);

        let vc = VerifiableCredential::new(
            issuer.clone(),
            subject,
            vec!["ExampleCredential".to_string()],
        );

        assert!(vc
            .credential_type
            .contains(&"VerifiableCredential".to_string()));
        assert!(vc
            .credential_type
            .contains(&"ExampleCredential".to_string()));
        assert!(!vc.is_expired());
    }

    #[test]
    fn test_credential_serialization() {
        let issuer = Did::new_key_ed25519(&[0u8; 32]).unwrap();
        let subject = CredentialSubject::new(Some("did:example:bob"))
            .with_claim("degree", "Bachelor of Science");

        let vc = VerifiableCredential::new(
            issuer,
            subject,
            vec!["UniversityDegreeCredential".to_string()],
        );

        let json = vc.to_json().unwrap();
        let parsed = VerifiableCredential::from_json(&json).unwrap();

        assert_eq!(vc.id, parsed.id);
        assert_eq!(vc.credential_type, parsed.credential_type);
    }

    #[test]
    fn test_credential_expiration() {
        let issuer = Did::new_key_ed25519(&[0u8; 32]).unwrap();
        let subject = CredentialSubject::new(None);

        let expired = VerifiableCredential::new(issuer.clone(), subject.clone(), vec![])
            .with_expiration(Utc::now() - chrono::Duration::hours(1));

        let valid = VerifiableCredential::new(issuer, subject, vec![])
            .with_expiration(Utc::now() + chrono::Duration::hours(1));

        assert!(expired.is_expired());
        assert!(!valid.is_expired());
    }
}
