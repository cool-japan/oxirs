//! Credential verifier

use super::VerifiableCredential;
use crate::{DidResolver, DidResult, VerificationResult};
use std::sync::Arc;

/// Credential verifier
pub struct CredentialVerifier {
    /// DID resolver
    resolver: Arc<DidResolver>,
}

impl CredentialVerifier {
    /// Create a new credential verifier
    pub fn new(resolver: Arc<DidResolver>) -> Self {
        Self { resolver }
    }

    /// Verify a verifiable credential
    pub async fn verify(&self, vc: &VerifiableCredential) -> DidResult<VerificationResult> {
        let mut result = VerificationResult::success(vc.issuer.did().as_str());

        // Check 1: Proof exists
        let proof_container = match &vc.proof {
            Some(p) => p,
            None => {
                return Ok(VerificationResult::failure("No proof found").with_check(
                    "proof_exists",
                    false,
                    Some("Credential has no proof"),
                ));
            }
        };

        let proofs = proof_container.proofs();
        if proofs.is_empty() {
            return Ok(VerificationResult::failure("Empty proof array").with_check(
                "proof_exists",
                false,
                Some("Proof array is empty"),
            ));
        }

        result = result.with_check("proof_exists", true, None);

        // Check 2: Expiration
        if vc.is_expired() {
            return Ok(VerificationResult::failure("Credential expired")
                .with_check("proof_exists", true, None)
                .with_check("not_expired", false, Some("Credential has expired")));
        }
        result = result.with_check("not_expired", true, None);

        // Check 3: Not before valid
        if vc.is_not_yet_valid() {
            return Ok(VerificationResult::failure("Credential not yet valid")
                .with_check("proof_exists", true, None)
                .with_check("not_expired", true, None)
                .with_check("valid_from", false, Some("Credential is not yet valid")));
        }
        result = result.with_check("valid_from", true, None);

        // Check 4: Signature verification
        let proof = proofs[0];

        // Resolve issuer DID
        let issuer_did = vc.issuer.did();
        let did_doc = match self.resolver.resolve(issuer_did).await {
            Ok(doc) => doc,
            Err(e) => {
                return Ok(
                    VerificationResult::failure(&format!("DID resolution failed: {}", e))
                        .with_check("proof_exists", true, None)
                        .with_check("not_expired", true, None)
                        .with_check("valid_from", true, None)
                        .with_check("did_resolved", false, Some(&e.to_string())),
                );
            }
        };
        result = result.with_check("did_resolved", true, None);

        // Get verification method
        let vm = match did_doc.get_assertion_method() {
            Some(vm) => vm,
            None => {
                return Ok(
                    VerificationResult::failure("No assertion method in DID Document")
                        .with_check("proof_exists", true, None)
                        .with_check("not_expired", true, None)
                        .with_check("valid_from", true, None)
                        .with_check("did_resolved", true, None)
                        .with_check("verification_method", false, None),
                );
            }
        };
        result = result.with_check("verification_method", true, None);

        // Get public key
        let public_key = match vm.get_public_key_bytes() {
            Ok(pk) => pk,
            Err(e) => {
                return Ok(
                    VerificationResult::failure(&format!("Invalid public key: {}", e))
                        .with_check("proof_exists", true, None)
                        .with_check("not_expired", true, None)
                        .with_check("valid_from", true, None)
                        .with_check("did_resolved", true, None)
                        .with_check("verification_method", true, None)
                        .with_check("public_key", false, None),
                );
            }
        };
        result = result.with_check("public_key", true, None);

        // Verify signature
        let signature = match proof.get_signature_bytes() {
            Ok(sig) => sig,
            Err(e) => {
                return Ok(VerificationResult::failure(&format!(
                    "Invalid signature format: {}",
                    e
                ))
                .with_check("proof_exists", true, None)
                .with_check("not_expired", true, None)
                .with_check("valid_from", true, None)
                .with_check("did_resolved", true, None)
                .with_check("verification_method", true, None)
                .with_check("public_key", true, None)
                .with_check("signature_format", false, None));
            }
        };
        result = result.with_check("signature_format", true, None);

        // Create canonical form
        let mut vc_copy = vc.clone();
        vc_copy.proof = None;
        let canonical = serde_json::to_string(&vc_copy)
            .map_err(|e| crate::DidError::SerializationError(e.to_string()))?;

        // Hash and verify
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(canonical.as_bytes());

        let verifier = crate::proof::ed25519::Ed25519Verifier::from_bytes(&public_key)?;
        let valid = verifier.verify(&hash, &signature)?;

        if valid {
            result = result.with_check("signature_valid", true, None);
            Ok(result)
        } else {
            Ok(VerificationResult::failure("Invalid signature")
                .with_check("proof_exists", true, None)
                .with_check("not_expired", true, None)
                .with_check("valid_from", true, None)
                .with_check("did_resolved", true, None)
                .with_check("verification_method", true, None)
                .with_check("public_key", true, None)
                .with_check("signature_format", true, None)
                .with_check(
                    "signature_valid",
                    false,
                    Some("Signature verification failed"),
                ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vc::CredentialIssuer;
    use crate::{CredentialSubject, Keystore};

    #[tokio::test]
    async fn test_verify_credential() {
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());

        // Issue a credential
        let issuer = CredentialIssuer::new(keystore.clone(), resolver.clone());
        let issuer_did = keystore.generate_ed25519().await.unwrap();

        let subject = CredentialSubject::new(Some("did:example:holder")).with_claim("name", "Test");

        let vc = issuer
            .issue(&issuer_did, subject, vec!["TestCredential".to_string()])
            .await
            .unwrap();

        // Verify
        let verifier = CredentialVerifier::new(resolver);
        let result = verifier.verify(&vc).await.unwrap();

        assert!(result.valid);
        assert!(result.checks.iter().all(|c| c.passed));
    }
}
