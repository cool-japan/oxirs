//! Credential issuer

use super::{CredentialSubject, VerifiableCredential};
use crate::vc::credential::{CredentialIssuerInfo, CredentialSubjectContainer, ProofContainer};
use crate::{Did, DidResolver, DidResult, Keystore, Proof, ProofPurpose};
use std::sync::Arc;

/// Credential issuer for creating signed VCs
pub struct CredentialIssuer {
    /// Key store for signing
    keystore: Arc<Keystore>,
    /// DID resolver
    resolver: Arc<DidResolver>,
}

impl CredentialIssuer {
    /// Create a new credential issuer
    pub fn new(keystore: Arc<Keystore>, resolver: Arc<DidResolver>) -> Self {
        Self { keystore, resolver }
    }

    /// Issue a new verifiable credential
    pub async fn issue(
        &self,
        issuer_did: &Did,
        subject: CredentialSubject,
        credential_types: Vec<String>,
    ) -> DidResult<VerifiableCredential> {
        // Create unsigned credential
        let mut vc = VerifiableCredential::new(issuer_did.clone(), subject, credential_types);

        // Get signer
        let signer = self.keystore.get_signer(issuer_did).await?;

        // Create canonical form for signing
        let canonical = self.create_canonical_form(&vc)?;

        // Hash and sign
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(canonical.as_bytes());
        let signature = signer.sign(&hash);

        // Create proof
        let proof = Proof::ed25519(
            &issuer_did.key_id("key-1"),
            ProofPurpose::AssertionMethod,
            &signature,
        );

        vc.proof = Some(ProofContainer::Single(Box::new(proof)));

        Ok(vc)
    }

    /// Issue credential with multiple subjects
    pub async fn issue_multi_subject(
        &self,
        issuer_did: &Did,
        subjects: Vec<CredentialSubject>,
        credential_types: Vec<String>,
    ) -> DidResult<VerifiableCredential> {
        let first_subject = subjects
            .first()
            .cloned()
            .unwrap_or_else(|| CredentialSubject::new(None));

        let mut vc = VerifiableCredential::new(issuer_did.clone(), first_subject, credential_types);
        vc.credential_subject = CredentialSubjectContainer::Multiple(subjects);

        // Sign
        let signer = self.keystore.get_signer(issuer_did).await?;
        let canonical = self.create_canonical_form(&vc)?;

        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(canonical.as_bytes());
        let signature = signer.sign(&hash);

        let proof = Proof::ed25519(
            &issuer_did.key_id("key-1"),
            ProofPurpose::AssertionMethod,
            &signature,
        );

        vc.proof = Some(ProofContainer::Single(Box::new(proof)));

        Ok(vc)
    }

    /// Issue with rich issuer information
    pub async fn issue_with_issuer_info(
        &self,
        issuer_did: &Did,
        issuer_name: &str,
        subject: CredentialSubject,
        credential_types: Vec<String>,
    ) -> DidResult<VerifiableCredential> {
        let mut vc = self.issue(issuer_did, subject, credential_types).await?;

        vc.issuer = CredentialIssuerInfo::Object {
            id: issuer_did.clone(),
            name: Some(issuer_name.to_string()),
            description: None,
            image: None,
        };

        Ok(vc)
    }

    /// Create canonical form for signing
    fn create_canonical_form(&self, vc: &VerifiableCredential) -> DidResult<String> {
        // Create a copy without proof for signing
        let mut vc_copy = vc.clone();
        vc_copy.proof = None;

        serde_json::to_string(&vc_copy)
            .map_err(|e| crate::DidError::SerializationError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_issue_credential() {
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let issuer = CredentialIssuer::new(keystore.clone(), resolver);

        // Generate issuer key
        let issuer_did = keystore.generate_ed25519().await.unwrap();

        let subject =
            CredentialSubject::new(Some("did:example:holder")).with_claim("name", "Alice");

        let vc = issuer
            .issue(&issuer_did, subject, vec!["TestCredential".to_string()])
            .await
            .unwrap();

        assert!(vc.proof.is_some());
        assert!(vc
            .credential_type
            .contains(&"VerifiableCredential".to_string()));
        assert!(vc.credential_type.contains(&"TestCredential".to_string()));
    }
}
