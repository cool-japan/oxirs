//! DID Document structure

use super::Did;
use crate::{DidResult, Service, VerificationMethod};
use serde::{Deserialize, Serialize};

/// DID Document (W3C DID Core 1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DidDocument {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    /// DID subject
    pub id: Did,

    /// Also known as (alternative identifiers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub also_known_as: Option<Vec<String>>,

    /// Controller DIDs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controller: Option<DidController>,

    /// Verification methods
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verification_method: Vec<VerificationMethod>,

    /// Authentication verification relationships
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authentication: Vec<VerificationRelationship>,

    /// Assertion method verification relationships
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assertion_method: Vec<VerificationRelationship>,

    /// Key agreement verification relationships
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_agreement: Vec<VerificationRelationship>,

    /// Capability invocation verification relationships
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capability_invocation: Vec<VerificationRelationship>,

    /// Capability delegation verification relationships
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capability_delegation: Vec<VerificationRelationship>,

    /// Service endpoints
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub service: Vec<Service>,
}

/// Controller can be a single DID or multiple DIDs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DidController {
    Single(Did),
    Multiple(Vec<Did>),
}

/// Verification relationship (can be reference or embedded)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VerificationRelationship {
    Reference(String),
    Embedded(VerificationMethod),
}

impl DidDocument {
    /// Create a minimal DID Document
    pub fn new(id: Did) -> Self {
        Self {
            context: vec![
                "https://www.w3.org/ns/did/v1".to_string(),
                "https://w3id.org/security/suites/ed25519-2020/v1".to_string(),
            ],
            id,
            also_known_as: None,
            controller: None,
            verification_method: Vec::new(),
            authentication: Vec::new(),
            assertion_method: Vec::new(),
            key_agreement: Vec::new(),
            capability_invocation: Vec::new(),
            capability_delegation: Vec::new(),
            service: Vec::new(),
        }
    }

    /// Create DID Document from did:key
    pub fn from_key_ed25519(public_key: &[u8]) -> DidResult<Self> {
        let did = Did::new_key_ed25519(public_key)?;
        let key_id = did.key_id("key-1");

        let verification_method = VerificationMethod::ed25519(&key_id, did.as_str(), public_key);

        let mut doc = Self::new(did);
        doc.verification_method.push(verification_method);
        doc.authentication
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.assertion_method
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.capability_invocation
            .push(VerificationRelationship::Reference(key_id.clone()));
        doc.capability_delegation
            .push(VerificationRelationship::Reference(key_id));

        Ok(doc)
    }

    /// Get verification method by ID
    pub fn get_verification_method(&self, id: &str) -> Option<&VerificationMethod> {
        self.verification_method.iter().find(|vm| vm.id == id)
    }

    /// Get the first verification method for assertion
    pub fn get_assertion_method(&self) -> Option<&VerificationMethod> {
        for rel in &self.assertion_method {
            match rel {
                VerificationRelationship::Reference(ref_id) => {
                    if let Some(vm) = self.get_verification_method(ref_id) {
                        return Some(vm);
                    }
                }
                VerificationRelationship::Embedded(vm) => {
                    return Some(vm);
                }
            }
        }
        None
    }

    /// Get the first verification method for authentication
    pub fn get_authentication_method(&self) -> Option<&VerificationMethod> {
        for rel in &self.authentication {
            match rel {
                VerificationRelationship::Reference(ref_id) => {
                    if let Some(vm) = self.get_verification_method(ref_id) {
                        return Some(vm);
                    }
                }
                VerificationRelationship::Embedded(vm) => {
                    return Some(vm);
                }
            }
        }
        None
    }

    /// Add a service endpoint
    pub fn add_service(&mut self, id: &str, service_type: &str, endpoint: &str) {
        self.service.push(Service {
            id: id.to_string(),
            service_type: service_type.to_string(),
            service_endpoint: endpoint.to_string(),
        });
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
    fn test_did_document_from_key() {
        let public_key = [0u8; 32];
        let doc = DidDocument::from_key_ed25519(&public_key).unwrap();

        assert_eq!(doc.verification_method.len(), 1);
        assert!(!doc.authentication.is_empty());
        assert!(!doc.assertion_method.is_empty());
    }

    #[test]
    fn test_did_document_serialization() {
        let public_key = [0u8; 32];
        let doc = DidDocument::from_key_ed25519(&public_key).unwrap();

        let json = doc.to_json().unwrap();
        let parsed = DidDocument::from_json(&json).unwrap();

        assert_eq!(doc.id, parsed.id);
        assert_eq!(
            doc.verification_method.len(),
            parsed.verification_method.len()
        );
    }

    #[test]
    fn test_get_assertion_method() {
        let public_key = [0u8; 32];
        let doc = DidDocument::from_key_ed25519(&public_key).unwrap();

        let am = doc.get_assertion_method();
        assert!(am.is_some());
        assert_eq!(am.unwrap().method_type, "Ed25519VerificationKey2020");
    }
}
