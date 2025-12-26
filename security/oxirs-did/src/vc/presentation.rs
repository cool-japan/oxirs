//! Verifiable Presentation

use super::VerifiableCredential;
use crate::vc::credential::ProofContainer;
use crate::{Did, DidResult, Proof, ProofPurpose};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Verifiable Presentation - a collection of credentials presented by a holder
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerifiablePresentation {
    /// JSON-LD context
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    /// Presentation identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Presentation types
    #[serde(rename = "type")]
    pub presentation_type: Vec<String>,

    /// Holder DID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holder: Option<Did>,

    /// Included credentials
    pub verifiable_credential: Vec<VerifiableCredential>,

    /// Proof of presentation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<ProofContainer>,
}

impl VerifiablePresentation {
    /// Create a new unsigned presentation
    pub fn new(holder: Did, credentials: Vec<VerifiableCredential>) -> Self {
        Self {
            context: vec!["https://www.w3.org/ns/credentials/v2".to_string()],
            id: Some(format!("urn:uuid:{}", uuid::Uuid::new_v4())),
            presentation_type: vec!["VerifiablePresentation".to_string()],
            holder: Some(holder),
            verifiable_credential: credentials,
            proof: None,
        }
    }

    /// Sign the presentation
    pub fn sign(
        mut self,
        signer: &crate::proof::ed25519::Ed25519Signer,
        holder_did: &Did,
        challenge: Option<&str>,
        domain: Option<&str>,
    ) -> DidResult<Self> {
        // Create canonical form
        let mut vp_copy = self.clone();
        vp_copy.proof = None;
        let canonical = serde_json::to_string(&vp_copy)
            .map_err(|e| crate::DidError::SerializationError(e.to_string()))?;

        // Hash and sign
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(canonical.as_bytes());
        let signature = signer.sign(&hash);

        // Create proof
        let mut proof = Proof::ed25519(
            &holder_did.key_id("key-1"),
            ProofPurpose::Authentication,
            &signature,
        );

        if let Some(c) = challenge {
            proof = proof.with_challenge(c);
        }
        if let Some(d) = domain {
            proof = proof.with_domain(d);
        }

        self.proof = Some(ProofContainer::Single(Box::new(proof)));

        Ok(self)
    }

    /// Add additional context
    pub fn with_context(mut self, context: &str) -> Self {
        self.context.push(context.to_string());
        self
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
    use crate::vc::CredentialIssuer;
    use crate::DidResolver;
    use crate::{CredentialSubject, Keystore};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_create_presentation() {
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let issuer = crate::vc::CredentialIssuer::new(keystore.clone(), resolver.clone());

        // Create issuer and holder
        let issuer_did = keystore.generate_ed25519().await.unwrap();
        let holder_did = keystore.generate_ed25519().await.unwrap();

        // Issue credential
        let subject = CredentialSubject::new(Some(holder_did.as_str())).with_claim("name", "Alice");

        let vc = issuer
            .issue(&issuer_did, subject, vec!["TestCredential".to_string()])
            .await
            .unwrap();

        // Create presentation
        let vp = VerifiablePresentation::new(holder_did.clone(), vec![vc]);

        assert!(vp.holder.is_some());
        assert_eq!(vp.verifiable_credential.len(), 1);
    }

    #[tokio::test]
    async fn test_sign_presentation() {
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let issuer = crate::vc::CredentialIssuer::new(keystore.clone(), resolver.clone());

        let issuer_did = keystore.generate_ed25519().await.unwrap();
        let holder_did = keystore.generate_ed25519().await.unwrap();

        let subject = CredentialSubject::new(Some(holder_did.as_str()));
        let vc = issuer.issue(&issuer_did, subject, vec![]).await.unwrap();

        let holder_signer = keystore.get_signer(&holder_did).await.unwrap();

        let vp = VerifiablePresentation::new(holder_did.clone(), vec![vc])
            .sign(
                &holder_signer,
                &holder_did,
                Some("challenge123"),
                Some("example.com"),
            )
            .unwrap();

        assert!(vp.proof.is_some());
    }
}
