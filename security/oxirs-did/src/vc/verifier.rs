//! Credential verifier

use super::VerifiableCredential;
use crate::vc::credential::CredentialStatus;
use crate::{DidError, DidResolver, DidResult, VerificationResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Resolves the revocation/suspension status of a credential from its
/// `credentialStatus` entry.
///
/// A verifier that is asked to verify a credential carrying a `credentialStatus`
/// MUST consult one of these. Implementations fetch/consult the referenced status
/// list (e.g. a StatusList2021Credential) and report whether the credential's bit
/// is set.
///
/// # Fail-loud contract
/// `is_revoked` returns `Err` when the referenced status list cannot be resolved
/// (unreachable, unknown, malformed). Callers treat that as a hard verification
/// failure — a credential whose status cannot be checked is never accepted.
#[async_trait::async_trait]
pub trait RevocationChecker: Send + Sync {
    /// Returns `Ok(true)` if the credential referenced by `status` is
    /// revoked/suspended, `Ok(false)` if it is valid, and `Err` if the status
    /// list cannot be resolved.
    async fn is_revoked(&self, status: &CredentialStatus) -> DidResult<bool>;
}

/// In-memory [`RevocationChecker`] backed by published StatusList2021 lists.
///
/// Register the decoded [`StatusList2021`](crate::revocation::StatusList2021)
/// documents keyed by their `statusListCredential` URL. Lookups consult the bit
/// at the credential's `statusListIndex`.
#[derive(Default)]
pub struct StatusList2021RevocationChecker {
    lists: HashMap<String, crate::revocation::StatusList2021>,
}

impl StatusList2021RevocationChecker {
    /// Create an empty checker with no registered status lists.
    pub fn new() -> Self {
        Self {
            lists: HashMap::new(),
        }
    }

    /// Register (or replace) the status list published at `status_list_url`.
    pub fn register_list(
        &mut self,
        status_list_url: &str,
        list: crate::revocation::StatusList2021,
    ) {
        self.lists.insert(status_list_url.to_string(), list);
    }
}

#[async_trait::async_trait]
impl RevocationChecker for StatusList2021RevocationChecker {
    async fn is_revoked(&self, status: &CredentialStatus) -> DidResult<bool> {
        let url = status.status_list_credential.as_deref().ok_or_else(|| {
            DidError::VerificationFailed(
                "credentialStatus is missing statusListCredential".to_string(),
            )
        })?;
        let list = self.lists.get(url).ok_or_else(|| {
            DidError::VerificationFailed(format!(
                "status list not available for verification: {url}"
            ))
        })?;
        let index: usize = status
            .status_list_index
            .as_deref()
            .ok_or_else(|| {
                DidError::VerificationFailed(
                    "credentialStatus is missing statusListIndex".to_string(),
                )
            })?
            .parse()
            .map_err(|e| DidError::InvalidFormat(format!("invalid statusListIndex: {e}")))?;
        list.is_revoked(index)
    }
}

/// Credential verifier
pub struct CredentialVerifier {
    /// DID resolver
    resolver: Arc<DidResolver>,
    /// Optional revocation-status resolver. Required to verify any credential
    /// that carries a `credentialStatus` entry (see [`RevocationChecker`]).
    revocation_checker: Option<Arc<dyn RevocationChecker>>,
}

impl CredentialVerifier {
    /// Create a new credential verifier
    pub fn new(resolver: Arc<DidResolver>) -> Self {
        Self {
            resolver,
            revocation_checker: None,
        }
    }

    /// Attach a [`RevocationChecker`] used to resolve `credentialStatus` entries.
    ///
    /// Without one, any credential carrying a `credentialStatus` fails
    /// verification (its revocation status cannot be established — fail-loud).
    pub fn with_revocation_checker(mut self, checker: Arc<dyn RevocationChecker>) -> Self {
        self.revocation_checker = Some(checker);
        self
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

            // Check 5: Revocation status (only when the credential declares one).
            //
            // A credential carrying a `credentialStatus` MUST have that status
            // resolved. If it is revoked/suspended, verification fails. If the
            // status list cannot be resolved — including the case where no
            // RevocationChecker is configured — verification also fails
            // (fail-loud): a credential whose revocation status cannot be
            // established is never silently accepted.
            if let Some(status) = &vc.credential_status {
                let outcome = match &self.revocation_checker {
                    Some(checker) => checker.is_revoked(status).await,
                    None => Err(DidError::VerificationFailed(
                        "credential declares a credentialStatus but no RevocationChecker is \
                         configured; cannot establish revocation status"
                            .to_string(),
                    )),
                };

                match outcome {
                    Ok(false) => {
                        result = result.with_check("not_revoked", true, None);
                        Ok(result)
                    }
                    Ok(true) => {
                        result.valid = false;
                        result.error = Some("Credential has been revoked".to_string());
                        Ok(result.with_check(
                            "not_revoked",
                            false,
                            Some("Credential is revoked in its status list"),
                        ))
                    }
                    Err(e) => {
                        result.valid = false;
                        result.error =
                            Some(format!("Revocation status could not be resolved: {e}"));
                        Ok(result.with_check("revocation_checked", false, Some(&e.to_string())))
                    }
                }
            } else {
                Ok(result)
            }
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

    // ── Revocation regression tests (Finding: credentialStatus never checked) ──

    async fn issue_vc_with_status(
        keystore: &Arc<Keystore>,
        status: Option<CredentialStatus>,
    ) -> VerifiableCredential {
        use crate::vc::credential::ProofContainer;
        use crate::{Proof, ProofPurpose};
        use sha2::{Digest, Sha256};

        let issuer_did = keystore.generate_ed25519().await.unwrap();
        let subject = CredentialSubject::new(Some("did:example:holder")).with_claim("name", "Test");
        let mut vc = VerifiableCredential::new(
            issuer_did.clone(),
            subject,
            vec!["TestCredential".to_string()],
        );
        // Status is bound into the signature (set BEFORE signing).
        vc.credential_status = status;

        let signer = keystore.get_signer(&issuer_did).await.unwrap();
        let mut unsigned = vc.clone();
        unsigned.proof = None;
        let canonical = serde_json::to_string(&unsigned).unwrap();
        let hash = Sha256::digest(canonical.as_bytes());
        let signature = signer.sign(&hash);
        let proof = Proof::ed25519(
            &issuer_did.key_id("key-1"),
            ProofPurpose::AssertionMethod,
            &signature,
        );
        vc.proof = Some(ProofContainer::Single(Box::new(proof)));
        vc
    }

    fn status_entry(index: usize) -> CredentialStatus {
        CredentialStatus {
            id: format!("https://example.com/status/1#{index}"),
            status_type: "StatusList2021Entry".to_string(),
            status_purpose: Some("revocation".to_string()),
            status_list_index: Some(index.to_string()),
            status_list_credential: Some("https://example.com/status/1".to_string()),
        }
    }

    fn checker_with(revoked_index: Option<usize>) -> StatusList2021RevocationChecker {
        let mut list = crate::revocation::StatusList2021::new(
            "https://example.com/status/1",
            "did:key:issuer",
            crate::revocation::MIN_LIST_SIZE,
        )
        .unwrap();
        if let Some(i) = revoked_index {
            list.set_status(i, true).unwrap();
        }
        let mut checker = StatusList2021RevocationChecker::new();
        checker.register_list("https://example.com/status/1", list);
        checker
    }

    #[tokio::test]
    async fn regression_revoked_credential_fails_verification() {
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let vc = issue_vc_with_status(&keystore, Some(status_entry(7))).await;

        let checker = checker_with(Some(7));
        let verifier = CredentialVerifier::new(resolver).with_revocation_checker(Arc::new(checker));
        let result = verifier.verify(&vc).await.unwrap();

        assert!(!result.valid, "revoked credential must not verify");
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "not_revoked" && !c.passed));
    }

    #[tokio::test]
    async fn regression_non_revoked_credential_passes_verification() {
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let vc = issue_vc_with_status(&keystore, Some(status_entry(7))).await;

        let checker = checker_with(None); // index 7 not revoked
        let verifier = CredentialVerifier::new(resolver).with_revocation_checker(Arc::new(checker));
        let result = verifier.verify(&vc).await.unwrap();

        assert!(result.valid, "non-revoked credential must verify");
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "not_revoked" && c.passed));
    }

    #[tokio::test]
    async fn regression_status_without_checker_fails_loud() {
        // A credential carrying a credentialStatus but no configured checker
        // must NOT silently pass — revocation cannot be established.
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let vc = issue_vc_with_status(&keystore, Some(status_entry(7))).await;

        let verifier = CredentialVerifier::new(resolver); // no checker
        let result = verifier.verify(&vc).await.unwrap();

        assert!(
            !result.valid,
            "must fail loud when status cannot be checked"
        );
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "revocation_checked" && !c.passed));
    }

    #[tokio::test]
    async fn regression_unresolvable_status_list_fails_loud() {
        // Status references a list the checker does not know about → hard failure.
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let vc = issue_vc_with_status(&keystore, Some(status_entry(7))).await;

        let empty_checker = StatusList2021RevocationChecker::new(); // no lists
        let verifier =
            CredentialVerifier::new(resolver).with_revocation_checker(Arc::new(empty_checker));
        let result = verifier.verify(&vc).await.unwrap();

        assert!(!result.valid, "unresolvable status list must fail loud");
        assert!(result
            .checks
            .iter()
            .any(|c| c.name == "revocation_checked" && !c.passed));
    }

    #[tokio::test]
    async fn regression_no_status_still_verifies_without_checker() {
        // Credentials WITHOUT a credentialStatus are unaffected by the new check.
        let keystore = Arc::new(Keystore::new());
        let resolver = Arc::new(DidResolver::new());
        let vc = issue_vc_with_status(&keystore, None).await;

        let verifier = CredentialVerifier::new(resolver);
        let result = verifier.verify(&vc).await.unwrap();
        assert!(result.valid);
    }
}
