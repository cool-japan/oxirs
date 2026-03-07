//! Cloud KMS Integration for DID key management
//!
//! Provides mock Cloud KMS backends (AWS KMS, GCP KMS, Azure Key Vault) for
//! managing cryptographic keys used with DIDs and Verifiable Credentials.
//!
//! All signing operations use HMAC-SHA256 over (key_id || data) with the stored
//! private key bytes as the HMAC key — sufficient for testing and mock usage.
//! Production deployments should wire in actual cloud SDK calls.

use crate::{DidDocument, DidError, DidResult, VerificationMethod};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::RwLock;

type HmacSha256 = Hmac<Sha256>;

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm / metadata types
// ─────────────────────────────────────────────────────────────────────────────

/// Cryptographic algorithm supported by the KMS
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KmsAlgorithm {
    Ed25519,
    EcP256,
    EcP384,
    Rsa2048,
    Rsa4096,
}

impl KmsAlgorithm {
    /// Human-readable name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ed25519 => "Ed25519",
            Self::EcP256 => "EC_P256",
            Self::EcP384 => "EC_P384",
            Self::Rsa2048 => "RSA_2048",
            Self::Rsa4096 => "RSA_4096",
        }
    }

    /// Nominal key size in bytes
    pub fn key_size_bytes(&self) -> usize {
        match self {
            Self::Ed25519 => 32,
            Self::EcP256 => 32,
            Self::EcP384 => 48,
            Self::Rsa2048 => 256,
            Self::Rsa4096 => 512,
        }
    }
}

/// Intended usage of a KMS key
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyUsage {
    SignVerify,
    EncryptDecrypt,
}

/// Metadata about a managed key
#[derive(Debug, Clone)]
pub struct KmsKeyMetadata {
    pub key_id: String,
    pub algorithm: KmsAlgorithm,
    /// Unix epoch seconds at creation
    pub created_at: i64,
    pub enabled: bool,
    pub key_usage: KeyUsage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal key entry (stored per backend)
// ─────────────────────────────────────────────────────────────────────────────

struct KmsKeyEntry {
    metadata: KmsKeyMetadata,
    /// Deterministically derived pseudo-private-key bytes
    private_key_bytes: Vec<u8>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over a Cloud KMS backend
pub trait KmsBackend: Send + Sync {
    fn create_key(&self, key_id: &str, algorithm: KmsAlgorithm) -> DidResult<KmsKeyMetadata>;

    fn sign(&self, key_id: &str, data: &[u8]) -> DidResult<Vec<u8>>;

    fn verify(&self, key_id: &str, data: &[u8], signature: &[u8]) -> DidResult<bool>;

    fn get_public_key(&self, key_id: &str) -> DidResult<Vec<u8>>;

    fn delete_key(&self, key_id: &str) -> DidResult<()>;

    fn list_keys(&self) -> DidResult<Vec<KmsKeyMetadata>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helper — HMAC-SHA256 signing
// ─────────────────────────────────────────────────────────────────────────────

/// Sign `data` with HMAC-SHA256 keyed by `private_key_bytes`, prepending `key_id`
/// to the message for domain separation.
fn hmac_sign(private_key_bytes: &[u8], key_id: &str, data: &[u8]) -> Vec<u8> {
    let mut mac =
        HmacSha256::new_from_slice(private_key_bytes).expect("HMAC accepts any key length");
    mac.update(key_id.as_bytes());
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Derive deterministic pseudo-private-key bytes from key_id and algorithm
fn derive_key_bytes(key_id: &str, algorithm: &KmsAlgorithm) -> Vec<u8> {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(key_id.as_bytes());
    hasher.update(algorithm.as_str().as_bytes());
    let base = hasher.finalize().to_vec();
    // Extend to cover largest key size (512 bytes for RSA-4096)
    let mut key = Vec::with_capacity(algorithm.key_size_bytes());
    let mut counter: u8 = 0;
    while key.len() < algorithm.key_size_bytes() {
        let mut h2 = sha2::Sha256::new();
        h2.update(&base);
        h2.update([counter]);
        key.extend_from_slice(&h2.finalize());
        counter = counter.wrapping_add(1);
    }
    key.truncate(algorithm.key_size_bytes());
    key
}

/// Derive a mock public key from private key bytes (first half, reversed)
fn derive_public_key(private_key_bytes: &[u8]) -> Vec<u8> {
    let half = private_key_bytes.len() / 2;
    let mut pub_key = private_key_bytes[..half.max(1)].to_vec();
    pub_key.reverse();
    pub_key
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Macro to generate mock backend structs to avoid repetition
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! impl_mock_kms {
    ($name:ident, $display:literal) => {
        /// Mock $display KMS backend
        pub struct $name {
            keys: RwLock<HashMap<String, KmsKeyEntry>>,
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    keys: RwLock::new(HashMap::new()),
                }
            }
        }

        impl KmsBackend for $name {
            fn create_key(
                &self,
                key_id: &str,
                algorithm: KmsAlgorithm,
            ) -> DidResult<KmsKeyMetadata> {
                let mut store = self
                    .keys
                    .write()
                    .map_err(|e| DidError::InternalError(format!("KMS lock poisoned: {}", e)))?;

                if store.contains_key(key_id) {
                    return Err(DidError::InvalidKey(format!(
                        "Key '{}' already exists in {} KMS",
                        key_id, $display
                    )));
                }

                let private_key_bytes = derive_key_bytes(key_id, &algorithm);
                let metadata = KmsKeyMetadata {
                    key_id: key_id.to_string(),
                    algorithm,
                    created_at: now_unix(),
                    enabled: true,
                    key_usage: KeyUsage::SignVerify,
                };

                let entry = KmsKeyEntry {
                    metadata: metadata.clone(),
                    private_key_bytes,
                };
                store.insert(key_id.to_string(), entry);
                Ok(metadata)
            }

            fn sign(&self, key_id: &str, data: &[u8]) -> DidResult<Vec<u8>> {
                let store = self
                    .keys
                    .read()
                    .map_err(|e| DidError::InternalError(format!("KMS lock poisoned: {}", e)))?;
                let entry = store.get(key_id).ok_or_else(|| {
                    DidError::KeyNotFound(format!("Key '{}' not found in {} KMS", key_id, $display))
                })?;
                if !entry.metadata.enabled {
                    return Err(DidError::SigningFailed(format!(
                        "Key '{}' is disabled",
                        key_id
                    )));
                }
                Ok(hmac_sign(&entry.private_key_bytes, key_id, data))
            }

            fn verify(&self, key_id: &str, data: &[u8], signature: &[u8]) -> DidResult<bool> {
                let expected = self.sign(key_id, data)?;
                Ok(expected == signature)
            }

            fn get_public_key(&self, key_id: &str) -> DidResult<Vec<u8>> {
                let store = self
                    .keys
                    .read()
                    .map_err(|e| DidError::InternalError(format!("KMS lock poisoned: {}", e)))?;
                let entry = store.get(key_id).ok_or_else(|| {
                    DidError::KeyNotFound(format!("Key '{}' not found in {} KMS", key_id, $display))
                })?;
                Ok(derive_public_key(&entry.private_key_bytes))
            }

            fn delete_key(&self, key_id: &str) -> DidResult<()> {
                let mut store = self
                    .keys
                    .write()
                    .map_err(|e| DidError::InternalError(format!("KMS lock poisoned: {}", e)))?;
                store
                    .remove(key_id)
                    .ok_or_else(|| {
                        DidError::KeyNotFound(format!(
                            "Key '{}' not found in {} KMS",
                            key_id, $display
                        ))
                    })
                    .map(|_| ())
            }

            fn list_keys(&self) -> DidResult<Vec<KmsKeyMetadata>> {
                let store = self
                    .keys
                    .read()
                    .map_err(|e| DidError::InternalError(format!("KMS lock poisoned: {}", e)))?;
                let mut list: Vec<KmsKeyMetadata> =
                    store.values().map(|e| e.metadata.clone()).collect();
                // Sort by key_id for deterministic output
                list.sort_by(|a, b| a.key_id.cmp(&b.key_id));
                Ok(list)
            }
        }
    };
}

impl_mock_kms!(MockAwsKms, "AWS");
impl_mock_kms!(MockGcpKms, "GCP");
impl_mock_kms!(MockAzureKms, "Azure");

// ─────────────────────────────────────────────────────────────────────────────
// Provider enum + factory
// ─────────────────────────────────────────────────────────────────────────────

/// Select which cloud provider mock to instantiate
pub enum KmsProvider {
    MockAws,
    MockGcp,
    MockAzure,
}

/// Create a mock KMS backend for the given provider
pub fn create_mock_kms(provider: KmsProvider) -> Box<dyn KmsBackend> {
    match provider {
        KmsProvider::MockAws => Box::new(MockAwsKms::new()),
        KmsProvider::MockGcp => Box::new(MockGcpKms::new()),
        KmsProvider::MockAzure => Box::new(MockAzureKms::new()),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KmsDidSigner — high-level DID operations backed by a KMS
// ─────────────────────────────────────────────────────────────────────────────

/// Uses a KMS backend to sign DIDs and Verifiable Credentials
pub struct KmsDidSigner {
    backend: Box<dyn KmsBackend>,
    key_id: String,
}

impl KmsDidSigner {
    pub fn new(backend: Box<dyn KmsBackend>, key_id: &str) -> Self {
        Self {
            backend,
            key_id: key_id.to_string(),
        }
    }

    /// Build a minimal DID Document with a verification method whose public key
    /// is derived from the KMS-managed key.
    pub fn create_did_document(&self, did: &str) -> DidResult<DidDocument> {
        let public_key = self.backend.get_public_key(&self.key_id)?;
        let key_fragment = format!("{}#kms-key-0", did);

        let vm = VerificationMethod::ed25519(&key_fragment, did, &public_key);

        use crate::did::document::{DidDocument as DocType, VerificationRelationship};
        use crate::Did;

        let did_obj = Did::new(did)?;
        let mut doc = DocType::new(did_obj);
        doc.verification_method.push(vm);
        doc.authentication
            .push(VerificationRelationship::Reference(key_fragment.clone()));
        doc.assertion_method
            .push(VerificationRelationship::Reference(key_fragment));

        Ok(doc)
    }

    /// Sign a JSON credential by appending a `proof` object with the HMAC
    /// signature (base64url encoded) and the KMS key reference.
    pub fn sign_credential(&self, credential: &serde_json::Value) -> DidResult<serde_json::Value> {
        let serialized = serde_json::to_vec(credential)
            .map_err(|e| DidError::SerializationError(e.to_string()))?;

        let sig = self.backend.sign(&self.key_id, &serialized)?;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;
        let sig_b64 = URL_SAFE_NO_PAD.encode(&sig);

        let mut signed = credential.clone();
        if let Some(obj) = signed.as_object_mut() {
            obj.insert(
                "proof".to_string(),
                serde_json::json!({
                    "type": "KmsHmacSignature2024",
                    "verificationMethod": self.key_id,
                    "signatureValue": sig_b64
                }),
            );
        }
        Ok(signed)
    }

    /// Verify a credential that was signed by `sign_credential`.
    pub fn verify_credential(&self, signed_credential: &serde_json::Value) -> DidResult<bool> {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        // Extract and remove the proof to reconstruct original payload
        let proof = signed_credential
            .get("proof")
            .ok_or_else(|| DidError::InvalidProof("Missing proof field".to_string()))?;

        let sig_b64 = proof
            .get("signatureValue")
            .and_then(|v| v.as_str())
            .ok_or_else(|| DidError::InvalidProof("Missing signatureValue".to_string()))?;

        let signature = URL_SAFE_NO_PAD
            .decode(sig_b64)
            .map_err(|e| DidError::InvalidProof(format!("Invalid base64: {}", e)))?;

        // Reconstruct credential without proof
        let mut without_proof = signed_credential.clone();
        if let Some(obj) = without_proof.as_object_mut() {
            obj.remove("proof");
        }
        let serialized = serde_json::to_vec(&without_proof)
            .map_err(|e| DidError::SerializationError(e.to_string()))?;

        self.backend.verify(&self.key_id, &serialized, &signature)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AWS KMS ──────────────────────────────────────────────────────────────

    #[test]
    fn test_aws_create_ed25519_key() {
        let kms = MockAwsKms::new();
        let meta = kms.create_key("my-key", KmsAlgorithm::Ed25519).unwrap();
        assert_eq!(meta.key_id, "my-key");
        assert_eq!(meta.algorithm.as_str(), "Ed25519");
        assert!(meta.enabled);
    }

    #[test]
    fn test_aws_create_duplicate_key_fails() {
        let kms = MockAwsKms::new();
        kms.create_key("dup", KmsAlgorithm::Ed25519).unwrap();
        assert!(kms.create_key("dup", KmsAlgorithm::Ed25519).is_err());
    }

    #[test]
    fn test_aws_sign_and_verify() {
        let kms = MockAwsKms::new();
        kms.create_key("signing-key", KmsAlgorithm::EcP256).unwrap();

        let data = b"hello world";
        let sig = kms.sign("signing-key", data).unwrap();
        assert!(!sig.is_empty());

        let valid = kms.verify("signing-key", data, &sig).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_aws_verify_wrong_data_fails() {
        let kms = MockAwsKms::new();
        kms.create_key("k1", KmsAlgorithm::Ed25519).unwrap();

        let sig = kms.sign("k1", b"original").unwrap();
        let valid = kms.verify("k1", b"tampered", &sig).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_aws_sign_missing_key_error() {
        let kms = MockAwsKms::new();
        assert!(kms.sign("nonexistent", b"data").is_err());
    }

    #[test]
    fn test_aws_get_public_key() {
        let kms = MockAwsKms::new();
        kms.create_key("pk-key", KmsAlgorithm::Ed25519).unwrap();
        let pub_key = kms.get_public_key("pk-key").unwrap();
        assert!(!pub_key.is_empty());
    }

    #[test]
    fn test_aws_delete_key() {
        let kms = MockAwsKms::new();
        kms.create_key("del-key", KmsAlgorithm::Ed25519).unwrap();
        kms.delete_key("del-key").unwrap();
        assert!(kms.sign("del-key", b"data").is_err());
    }

    #[test]
    fn test_aws_delete_missing_key_error() {
        let kms = MockAwsKms::new();
        assert!(kms.delete_key("ghost").is_err());
    }

    #[test]
    fn test_aws_list_keys() {
        let kms = MockAwsKms::new();
        kms.create_key("a", KmsAlgorithm::Ed25519).unwrap();
        kms.create_key("b", KmsAlgorithm::EcP256).unwrap();

        let keys = kms.list_keys().unwrap();
        assert_eq!(keys.len(), 2);
        // Sorted by key_id
        assert_eq!(keys[0].key_id, "a");
        assert_eq!(keys[1].key_id, "b");
    }

    #[test]
    fn test_aws_all_algorithms_create() {
        let kms = MockAwsKms::new();
        kms.create_key("ed", KmsAlgorithm::Ed25519).unwrap();
        kms.create_key("p256", KmsAlgorithm::EcP256).unwrap();
        kms.create_key("p384", KmsAlgorithm::EcP384).unwrap();
        kms.create_key("rsa2048", KmsAlgorithm::Rsa2048).unwrap();
        kms.create_key("rsa4096", KmsAlgorithm::Rsa4096).unwrap();

        let keys = kms.list_keys().unwrap();
        assert_eq!(keys.len(), 5);
    }

    // ── GCP KMS ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gcp_create_and_sign() {
        let kms = MockGcpKms::new();
        kms.create_key("gcp-key", KmsAlgorithm::EcP256).unwrap();

        let sig = kms.sign("gcp-key", b"gcp-data").unwrap();
        let valid = kms.verify("gcp-key", b"gcp-data", &sig).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_gcp_list_empty() {
        let kms = MockGcpKms::new();
        let keys = kms.list_keys().unwrap();
        assert!(keys.is_empty());
    }

    #[test]
    fn test_gcp_public_key_differs_from_private() {
        let kms = MockGcpKms::new();
        kms.create_key("gcp-pk", KmsAlgorithm::Ed25519).unwrap();

        let pub_key = kms.get_public_key("gcp-pk").unwrap();
        // Public key is derived from private key — should not be empty
        assert!(!pub_key.is_empty());
    }

    #[test]
    fn test_gcp_delete_and_recreate() {
        let kms = MockGcpKms::new();
        kms.create_key("reuse", KmsAlgorithm::Ed25519).unwrap();
        kms.delete_key("reuse").unwrap();
        // Should succeed after deletion
        kms.create_key("reuse", KmsAlgorithm::Ed25519).unwrap();
    }

    // ── Azure KMS ────────────────────────────────────────────────────────────

    #[test]
    fn test_azure_create_and_sign() {
        let kms = MockAzureKms::new();
        kms.create_key("az-key", KmsAlgorithm::Rsa2048).unwrap();

        let sig = kms.sign("az-key", b"azure-data").unwrap();
        let valid = kms.verify("az-key", b"azure-data", &sig).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_azure_wrong_signature() {
        let kms = MockAzureKms::new();
        kms.create_key("az2", KmsAlgorithm::EcP256).unwrap();

        let bad_sig = vec![0u8; 32];
        let valid = kms.verify("az2", b"some-data", &bad_sig).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_azure_list_after_delete() {
        let kms = MockAzureKms::new();
        kms.create_key("x", KmsAlgorithm::Ed25519).unwrap();
        kms.create_key("y", KmsAlgorithm::Ed25519).unwrap();
        kms.delete_key("x").unwrap();

        let keys = kms.list_keys().unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].key_id, "y");
    }

    // ── Factory ──────────────────────────────────────────────────────────────

    #[test]
    fn test_create_mock_kms_aws() {
        let kms = create_mock_kms(KmsProvider::MockAws);
        kms.create_key("factory-aws", KmsAlgorithm::Ed25519)
            .unwrap();
        let keys = kms.list_keys().unwrap();
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn test_create_mock_kms_gcp() {
        let kms = create_mock_kms(KmsProvider::MockGcp);
        kms.create_key("factory-gcp", KmsAlgorithm::EcP256).unwrap();
        let keys = kms.list_keys().unwrap();
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn test_create_mock_kms_azure() {
        let kms = create_mock_kms(KmsProvider::MockAzure);
        kms.create_key("factory-azure", KmsAlgorithm::Rsa2048)
            .unwrap();
        let keys = kms.list_keys().unwrap();
        assert_eq!(keys.len(), 1);
    }

    // ── KmsDidSigner ─────────────────────────────────────────────────────────

    #[test]
    fn test_kms_did_signer_create_document() {
        let backend = create_mock_kms(KmsProvider::MockAws);
        backend
            .create_key("did-signer-key", KmsAlgorithm::Ed25519)
            .unwrap();

        let signer = KmsDidSigner::new(backend, "did-signer-key");
        let did_str = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK";
        let doc = signer.create_did_document(did_str).unwrap();

        assert_eq!(doc.id.as_str(), did_str);
        assert_eq!(doc.verification_method.len(), 1);
        assert!(!doc.authentication.is_empty());
    }

    #[test]
    fn test_kms_did_signer_sign_credential() {
        let backend = create_mock_kms(KmsProvider::MockGcp);
        backend.create_key("vc-key", KmsAlgorithm::EcP256).unwrap();

        let signer = KmsDidSigner::new(backend, "vc-key");
        let credential = serde_json::json!({
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "type": ["VerifiableCredential"],
            "issuer": "did:example:issuer",
            "credentialSubject": { "id": "did:example:subject", "name": "Alice" }
        });

        let signed = signer.sign_credential(&credential).unwrap();
        assert!(signed.get("proof").is_some());
        let proof = signed.get("proof").unwrap();
        assert_eq!(proof["type"].as_str().unwrap(), "KmsHmacSignature2024");
        assert!(proof.get("signatureValue").is_some());
    }

    #[test]
    fn test_kms_did_signer_verify_credential() {
        let backend = create_mock_kms(KmsProvider::MockAzure);
        backend
            .create_key("verify-key", KmsAlgorithm::Ed25519)
            .unwrap();

        let signer = KmsDidSigner::new(backend, "verify-key");
        let credential = serde_json::json!({
            "id": "http://example.com/vc/1",
            "type": ["VerifiableCredential"],
            "issuer": "did:example:issuer"
        });

        let signed = signer.sign_credential(&credential).unwrap();
        let valid = signer.verify_credential(&signed).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_kms_did_signer_tampered_credential_fails() {
        let backend = create_mock_kms(KmsProvider::MockAws);
        backend
            .create_key("tamper-key", KmsAlgorithm::EcP256)
            .unwrap();

        let signer = KmsDidSigner::new(backend, "tamper-key");
        let credential = serde_json::json!({
            "type": ["VerifiableCredential"],
            "issuer": "did:example:issuer"
        });

        let mut signed = signer.sign_credential(&credential).unwrap();
        // Tamper with the credential payload
        if let Some(obj) = signed.as_object_mut() {
            obj.insert("issuer".to_string(), serde_json::json!("did:evil:attacker"));
        }

        let valid = signer.verify_credential(&signed).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_kms_did_signer_missing_proof_error() {
        let backend = create_mock_kms(KmsProvider::MockAws);
        backend
            .create_key("no-proof-key", KmsAlgorithm::Ed25519)
            .unwrap();

        let signer = KmsDidSigner::new(backend, "no-proof-key");
        let credential = serde_json::json!({ "type": "VerifiableCredential" });
        // Not signed — no proof field
        assert!(signer.verify_credential(&credential).is_err());
    }

    #[test]
    fn test_key_metadata_fields() {
        let kms = MockAwsKms::new();
        let meta = kms.create_key("meta-test", KmsAlgorithm::EcP384).unwrap();

        assert_eq!(meta.key_id, "meta-test");
        assert_eq!(meta.algorithm.as_str(), "EC_P384");
        assert!(matches!(meta.key_usage, KeyUsage::SignVerify));
        assert!(meta.created_at > 0);
        assert!(meta.enabled);
    }

    #[test]
    fn test_algorithm_key_sizes() {
        assert_eq!(KmsAlgorithm::Ed25519.key_size_bytes(), 32);
        assert_eq!(KmsAlgorithm::EcP256.key_size_bytes(), 32);
        assert_eq!(KmsAlgorithm::EcP384.key_size_bytes(), 48);
        assert_eq!(KmsAlgorithm::Rsa2048.key_size_bytes(), 256);
        assert_eq!(KmsAlgorithm::Rsa4096.key_size_bytes(), 512);
    }

    #[test]
    fn test_signatures_are_deterministic() {
        let kms = MockAwsKms::new();
        kms.create_key("det-key", KmsAlgorithm::Ed25519).unwrap();

        let sig1 = kms.sign("det-key", b"same data").unwrap();
        let sig2 = kms.sign("det-key", b"same data").unwrap();
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_different_keys_produce_different_signatures() {
        let kms = MockAwsKms::new();
        kms.create_key("key-a", KmsAlgorithm::Ed25519).unwrap();
        kms.create_key("key-b", KmsAlgorithm::Ed25519).unwrap();

        let sig_a = kms.sign("key-a", b"data").unwrap();
        let sig_b = kms.sign("key-b", b"data").unwrap();
        assert_ne!(sig_a, sig_b);
    }
}
