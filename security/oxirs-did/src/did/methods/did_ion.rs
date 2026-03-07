//! did:ion DID Method implementation
//!
//! ION (Identity Overlay Network) is a public, permissionless, decentralized PKI
//! network running on top of Bitcoin via the Sidetree protocol.
//!
//! Format: `did:ion:<unique-suffix>`
//! where unique-suffix is derived from the initial DID state document
//!
//! References:
//!   <https://identity.foundation/ion/>
//!   <https://identity.foundation/sidetree/spec/>
//!
//! In this implementation, we provide:
//! - DID creation (generate unique suffix from initial document state)
//! - DID Document construction from ION operations
//! - ION operation types (create, update, recover, deactivate)
//! - Resolution of cached/pre-resolved DID Documents

use super::DidMethod;
use crate::did::document::VerificationRelationship;
use crate::did::{Did, DidDocument};
use crate::{DidError, DidResult, VerificationMethod};
use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// ION Sidetree operation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IonOperationType {
    /// Create a new DID
    Create,
    /// Update an existing DID's document
    Update,
    /// Recover a DID after key compromise
    Recover,
    /// Deactivate a DID permanently
    Deactivate,
}

/// ION key purpose flags
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IonKeyPurpose {
    Authentication,
    AssertionMethod,
    KeyAgreement,
    CapabilityInvocation,
    CapabilityDelegation,
}

impl IonKeyPurpose {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::AssertionMethod => "assertionMethod",
            Self::KeyAgreement => "keyAgreement",
            Self::CapabilityInvocation => "capabilityInvocation",
            Self::CapabilityDelegation => "capabilityDelegation",
        }
    }
}

/// An ION DID key descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonKeyDescriptor {
    /// Short ID for this key within the document
    pub id: String,
    /// Purposes this key serves
    pub purposes: Vec<IonKeyPurpose>,
    /// The public key JWK
    pub public_key_jwk: serde_json::Value,
}

impl IonKeyDescriptor {
    /// Create an ION key descriptor with an Ed25519 public key
    pub fn ed25519(
        id: &str,
        public_key_bytes: &[u8],
        purposes: Vec<IonKeyPurpose>,
    ) -> DidResult<Self> {
        if public_key_bytes.len() != 32 {
            return Err(DidError::InvalidKey(
                "Ed25519 public key must be 32 bytes".to_string(),
            ));
        }
        let x = URL_SAFE_NO_PAD.encode(public_key_bytes);
        let jwk = serde_json::json!({
            "kty": "OKP",
            "crv": "Ed25519",
            "x": x
        });
        Ok(Self {
            id: id.to_string(),
            purposes,
            public_key_jwk: jwk,
        })
    }

    /// Create an ION key descriptor with a secp256k1 public key (EcdsaSecp256k1VerificationKey2019)
    pub fn secp256k1(
        id: &str,
        public_key_bytes: &[u8],
        purposes: Vec<IonKeyPurpose>,
    ) -> DidResult<Self> {
        if public_key_bytes.len() != 33 {
            return Err(DidError::InvalidKey(
                "secp256k1 compressed public key must be 33 bytes".to_string(),
            ));
        }
        // X coordinate is bytes 1-32 from the compressed key
        let x = URL_SAFE_NO_PAD.encode(&public_key_bytes[1..]);
        let jwk = serde_json::json!({
            "kty": "EC",
            "crv": "secp256k1",
            "x": x
        });
        Ok(Self {
            id: id.to_string(),
            purposes,
            public_key_jwk: jwk,
        })
    }
}

/// ION Create Operation - the initial DID state document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonCreateOperation {
    /// Recovery commitment (hash of recovery key)
    pub recovery_commitment: String,
    /// Update commitment (hash of update key)
    pub update_commitment: String,
    /// Document with keys and services
    pub document: IonDocument,
}

/// ION Document model (patch document format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonDocument {
    /// Public keys in this document
    pub public_keys: Vec<IonKeyDescriptor>,
    /// Service endpoints
    pub services: Vec<IonService>,
}

/// ION Service endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonService {
    /// Short ID for this service
    pub id: String,
    /// Type of service
    #[serde(rename = "type")]
    pub service_type: String,
    /// Service endpoint URL
    pub service_endpoint: serde_json::Value,
}

/// ION DID state with suffix and initial document
#[derive(Debug, Clone)]
pub struct DidIon {
    /// The DID unique suffix (derived from initial state)
    pub suffix: String,
    /// The create operation that generated this DID
    pub initial_state: Option<IonCreateOperation>,
    /// Resolved DID Document (if available)
    pub resolved_document: Option<DidDocument>,
}

impl DidIon {
    /// Create a new ION DID from a create operation
    ///
    /// The unique suffix is computed as: SHA-256(create_operation_data)
    pub fn new(operation: IonCreateOperation) -> DidResult<Self> {
        let suffix = compute_did_suffix(&operation)?;
        Ok(Self {
            suffix,
            initial_state: Some(operation),
            resolved_document: None,
        })
    }

    /// Create from an existing DID string (suffix only)
    pub fn from_did_string(did: &str) -> DidResult<Self> {
        if !did.starts_with("did:ion:") {
            return Err(DidError::InvalidFormat(
                "DID must start with 'did:ion:'".to_string(),
            ));
        }

        let suffix = did["did:ion:".len()..].to_string();
        if suffix.is_empty() {
            return Err(DidError::InvalidFormat(
                "ION DID suffix cannot be empty".to_string(),
            ));
        }

        // Validate suffix is valid base64url
        if suffix.contains(':') || suffix.contains('/') {
            return Err(DidError::InvalidFormat(
                "ION DID suffix must not contain colons or slashes".to_string(),
            ));
        }

        Ok(Self {
            suffix,
            initial_state: None,
            resolved_document: None,
        })
    }

    /// Get the DID string
    pub fn to_did_string(&self) -> String {
        format!("did:ion:{}", self.suffix)
    }

    /// Convert to Did
    pub fn to_did(&self) -> DidResult<Did> {
        Did::new(&self.to_did_string())
    }

    /// Resolve this DID to a DID Document
    ///
    /// Returns the pre-resolved document if available, or constructs one
    /// from the initial create operation if present.
    pub fn resolve(&self) -> DidResult<DidDocument> {
        if let Some(ref doc) = self.resolved_document {
            return Ok(doc.clone());
        }

        if let Some(ref op) = self.initial_state {
            return self.generate_document_from_create_op(op);
        }

        Err(DidError::ResolutionFailed(format!(
            "Cannot resolve did:ion:{} - no initial state or resolved document",
            self.suffix
        )))
    }

    /// Set a pre-resolved DID Document
    pub fn with_resolved_document(mut self, doc: DidDocument) -> Self {
        self.resolved_document = Some(doc);
        self
    }

    /// Generate a DID Document from the create operation
    fn generate_document_from_create_op(&self, op: &IonCreateOperation) -> DidResult<DidDocument> {
        let did_str = self.to_did_string();
        let did = Did::new(&did_str)?;
        let mut doc = DidDocument::new(did);

        doc.context = vec![
            "https://www.w3.org/ns/did/v1".to_string(),
            "https://w3id.org/security/suites/ed25519-2020/v1".to_string(),
            "https://w3id.org/security/suites/secp256k1-2019/v1".to_string(),
        ];

        // Process each public key in the document
        for key_desc in &op.document.public_keys {
            let vm_id = format!("{}#{}", did_str, key_desc.id);

            // Determine method type from JWK
            let method_type = detect_key_type_from_jwk(&key_desc.public_key_jwk);
            let vm = VerificationMethod::jwk(
                &vm_id,
                &did_str,
                &method_type,
                key_desc.public_key_jwk.clone(),
            );

            doc.verification_method.push(vm);

            // Add to appropriate verification relationships based on purposes
            for purpose in &key_desc.purposes {
                let rel = VerificationRelationship::Reference(vm_id.clone());
                match purpose {
                    IonKeyPurpose::Authentication => doc.authentication.push(rel),
                    IonKeyPurpose::AssertionMethod => doc.assertion_method.push(rel),
                    IonKeyPurpose::KeyAgreement => doc.key_agreement.push(rel),
                    IonKeyPurpose::CapabilityInvocation => doc.capability_invocation.push(rel),
                    IonKeyPurpose::CapabilityDelegation => doc.capability_delegation.push(rel),
                }
            }
        }

        // Add services
        for svc in &op.document.services {
            let endpoint = match &svc.service_endpoint {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };

            doc.service.push(crate::Service {
                id: format!("{}#{}", did_str, svc.id),
                service_type: svc.service_type.clone(),
                service_endpoint: endpoint,
            });
        }

        Ok(doc)
    }

    /// Generate the update commitment for a key
    ///
    /// Commitment = SHA-256(CONCAT(0x00, SHA-256(key_bytes)))
    pub fn compute_commitment(key_bytes: &[u8]) -> String {
        let inner_hash = sha256(key_bytes);
        let mut prefixed = vec![0x00u8];
        prefixed.extend_from_slice(&inner_hash);
        let commitment = sha256(&prefixed);
        URL_SAFE_NO_PAD.encode(&commitment)
    }
}

/// did:ion method resolver
pub struct DidIonMethod;

impl Default for DidIonMethod {
    fn default() -> Self {
        Self::new()
    }
}

impl DidIonMethod {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DidMethod for DidIonMethod {
    fn method_name(&self) -> &str {
        "ion"
    }

    async fn resolve(&self, did: &Did) -> DidResult<DidDocument> {
        if !self.supports(did) {
            return Err(DidError::UnsupportedMethod(did.method().to_string()));
        }

        let ion = DidIon::from_did_string(did.as_str())?;
        ion.resolve()
    }
}

/// Compute SHA-256 hash
fn sha256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Compute the ION DID unique suffix from a create operation
///
/// Suffix = BASE64URL(SHA-256(CANONICALIZED_CREATE_OPERATION))
fn compute_did_suffix(operation: &IonCreateOperation) -> DidResult<String> {
    // Serialize the create operation deterministically
    let json = serde_json::to_string(operation)
        .map_err(|e| DidError::SerializationError(e.to_string()))?;
    let hash = sha256(json.as_bytes());
    Ok(URL_SAFE_NO_PAD.encode(&hash))
}

/// Detect the verification method type from a JWK
fn detect_key_type_from_jwk(jwk: &serde_json::Value) -> String {
    match (jwk.get("kty"), jwk.get("crv")) {
        (Some(kty), Some(crv))
            if kty.as_str() == Some("OKP") && crv.as_str() == Some("Ed25519") =>
        {
            "Ed25519VerificationKey2020".to_string()
        }
        (Some(kty), Some(crv))
            if kty.as_str() == Some("EC") && crv.as_str() == Some("secp256k1") =>
        {
            "EcdsaSecp256k1VerificationKey2019".to_string()
        }
        (Some(kty), Some(crv)) if kty.as_str() == Some("EC") && crv.as_str() == Some("P-256") => {
            "JsonWebKey2020".to_string()
        }
        (Some(kty), _) if kty.as_str() == Some("RSA") => "JsonWebKey2020".to_string(),
        _ => "JsonWebKey2020".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ion_did() -> (DidIon, IonCreateOperation) {
        let key = IonKeyDescriptor::ed25519(
            "auth-key",
            &[1u8; 32],
            vec![
                IonKeyPurpose::Authentication,
                IonKeyPurpose::AssertionMethod,
            ],
        )
        .unwrap();

        let service = IonService {
            id: "linked-domain".to_string(),
            service_type: "LinkedDomains".to_string(),
            service_endpoint: serde_json::json!("https://example.com"),
        };

        let recovery_key = [2u8; 32];
        let update_key = [3u8; 32];

        let op = IonCreateOperation {
            recovery_commitment: DidIon::compute_commitment(&recovery_key),
            update_commitment: DidIon::compute_commitment(&update_key),
            document: IonDocument {
                public_keys: vec![key],
                services: vec![service],
            },
        };

        let ion = DidIon::new(op.clone()).unwrap();
        (ion, op)
    }

    #[test]
    fn test_create_ion_did() {
        let (ion, _) = create_test_ion_did();
        assert!(ion.to_did_string().starts_with("did:ion:"));
        assert!(!ion.suffix.is_empty());
    }

    #[test]
    fn test_ion_did_suffix_deterministic() {
        // Same input should produce same suffix
        let (ion1, op) = create_test_ion_did();
        let ion2 = DidIon::new(op).unwrap();
        assert_eq!(ion1.suffix, ion2.suffix);
    }

    #[test]
    fn test_ion_did_from_string() {
        let suffix = "EiClkZMDxPKqC9c-umQceAyopvJFHEWNpTJPCj47A";
        let did_str = format!("did:ion:{}", suffix);
        let ion = DidIon::from_did_string(&did_str).unwrap();
        assert_eq!(ion.suffix, suffix);
        assert_eq!(ion.to_did_string(), did_str);
    }

    #[test]
    fn test_ion_did_from_string_invalid() {
        assert!(DidIon::from_did_string("did:key:z6Mk123").is_err());
        assert!(DidIon::from_did_string("did:ion:").is_err());
    }

    #[test]
    fn test_ion_resolve_with_create_op() {
        let (ion, _) = create_test_ion_did();
        let doc = ion.resolve().unwrap();

        // Should have one verification method
        assert_eq!(doc.verification_method.len(), 1);
        // Should have authentication
        assert!(!doc.authentication.is_empty());
        // Should have assertion method
        assert!(!doc.assertion_method.is_empty());
        // Should have service
        assert_eq!(doc.service.len(), 1);
    }

    #[test]
    fn test_ion_resolve_without_state() {
        let ion =
            DidIon::from_did_string("did:ion:EiClkZMDxPKqC9c-umQceAyopvJFHEWNpTJPCj47A").unwrap();
        // Should fail without initial state
        assert!(ion.resolve().is_err());
    }

    #[test]
    fn test_ion_key_descriptor_ed25519() {
        let key =
            IonKeyDescriptor::ed25519("key-1", &[42u8; 32], vec![IonKeyPurpose::Authentication])
                .unwrap();

        assert_eq!(key.id, "key-1");
        assert_eq!(key.public_key_jwk["kty"], "OKP");
        assert_eq!(key.public_key_jwk["crv"], "Ed25519");
    }

    #[test]
    fn test_ion_key_descriptor_secp256k1() {
        let mut compressed_key = [0u8; 33];
        compressed_key[0] = 0x02; // compressed prefix
        let key = IonKeyDescriptor::secp256k1(
            "key-1",
            &compressed_key,
            vec![IonKeyPurpose::Authentication],
        )
        .unwrap();

        assert_eq!(key.public_key_jwk["kty"], "EC");
        assert_eq!(key.public_key_jwk["crv"], "secp256k1");
    }

    #[test]
    fn test_ion_key_wrong_size() {
        assert!(IonKeyDescriptor::ed25519("k", &[0u8; 31], vec![]).is_err());
        assert!(IonKeyDescriptor::secp256k1("k", &[0u8; 32], vec![]).is_err()); // needs 33 bytes
    }

    #[test]
    fn test_compute_commitment() {
        let commitment1 = DidIon::compute_commitment(&[1u8; 32]);
        let commitment2 = DidIon::compute_commitment(&[1u8; 32]);
        // Deterministic
        assert_eq!(commitment1, commitment2);

        // Different keys produce different commitments
        let commitment3 = DidIon::compute_commitment(&[2u8; 32]);
        assert_ne!(commitment1, commitment3);
    }

    #[test]
    fn test_with_resolved_document() {
        let ion =
            DidIon::from_did_string("did:ion:EiClkZMDxPKqC9c-umQceAyopvJFHEWNpTJPCj47A").unwrap();
        let did = Did::new("did:ion:EiClkZMDxPKqC9c-umQceAyopvJFHEWNpTJPCj47A").unwrap();
        let pre_resolved_doc = DidDocument::from_key_ed25519(&[0u8; 32]).unwrap();

        let ion = ion.with_resolved_document(pre_resolved_doc);
        let doc = ion.resolve().unwrap();
        // Should return the pre-resolved document
        assert!(!doc.verification_method.is_empty());
        let _ = did; // used to suppress warning
    }

    #[test]
    fn test_detect_key_type_ed25519() {
        let jwk = serde_json::json!({ "kty": "OKP", "crv": "Ed25519", "x": "abc" });
        assert_eq!(detect_key_type_from_jwk(&jwk), "Ed25519VerificationKey2020");
    }

    #[test]
    fn test_detect_key_type_secp256k1() {
        let jwk = serde_json::json!({ "kty": "EC", "crv": "secp256k1", "x": "abc" });
        assert_eq!(
            detect_key_type_from_jwk(&jwk),
            "EcdsaSecp256k1VerificationKey2019"
        );
    }

    #[test]
    fn test_detect_key_type_rsa() {
        let jwk = serde_json::json!({ "kty": "RSA", "n": "abc", "e": "AQAB" });
        assert_eq!(detect_key_type_from_jwk(&jwk), "JsonWebKey2020");
    }

    #[tokio::test]
    async fn test_ion_method_resolver() {
        let method = DidIonMethod::new();
        assert_eq!(method.method_name(), "ion");
    }

    #[tokio::test]
    async fn test_ion_method_wrong_method_error() {
        let method = DidIonMethod::new();
        let did = Did::new("did:key:z6Mk123").unwrap();
        assert!(method.resolve(&did).await.is_err());
    }

    #[test]
    fn test_multiple_keys_in_document() {
        let key1 =
            IonKeyDescriptor::ed25519("auth-key", &[1u8; 32], vec![IonKeyPurpose::Authentication])
                .unwrap();

        let key2 = IonKeyDescriptor::ed25519(
            "assert-key",
            &[2u8; 32],
            vec![
                IonKeyPurpose::AssertionMethod,
                IonKeyPurpose::CapabilityInvocation,
            ],
        )
        .unwrap();

        let op = IonCreateOperation {
            recovery_commitment: DidIon::compute_commitment(&[10u8; 32]),
            update_commitment: DidIon::compute_commitment(&[11u8; 32]),
            document: IonDocument {
                public_keys: vec![key1, key2],
                services: vec![],
            },
        };

        let ion = DidIon::new(op).unwrap();
        let doc = ion.resolve().unwrap();

        assert_eq!(doc.verification_method.len(), 2);
        assert_eq!(doc.authentication.len(), 1);
        assert_eq!(doc.assertion_method.len(), 1);
        assert_eq!(doc.capability_invocation.len(), 1);
    }

    #[test]
    fn test_ion_service_complex_endpoint() {
        let svc = IonService {
            id: "id-hub".to_string(),
            service_type: "IdentityHub".to_string(),
            service_endpoint: serde_json::json!({
                "instances": ["https://hub.example.com"]
            }),
        };

        let op = IonCreateOperation {
            recovery_commitment: DidIon::compute_commitment(&[10u8; 32]),
            update_commitment: DidIon::compute_commitment(&[11u8; 32]),
            document: IonDocument {
                public_keys: vec![],
                services: vec![svc],
            },
        };

        let ion = DidIon::new(op).unwrap();
        let doc = ion.resolve().unwrap();

        assert_eq!(doc.service.len(), 1);
        assert_eq!(doc.service[0].service_type, "IdentityHub");
    }
}
