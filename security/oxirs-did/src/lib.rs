//! # OxiRS DID
//!
//! [![Version](https://img.shields.io/badge/version-0.2.2-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Production Release (v0.2.2)
//!
//! W3C Decentralized Identifiers (DID) and Verifiable Credentials (VC) implementation
//! for OxiRS, enabling signed RDF graphs and trust layer for data sovereignty.
//!
//! ## Features
//!
//! - **DID Methods**: did:key (Ed25519), did:web (HTTP-based)
//! - **Verifiable Credentials**: W3C VC Data Model 2.0
//! - **Signed Graphs**: RDF Dataset Canonicalization + Ed25519 signatures
//! - **Key Management**: Secure key storage and derivation
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_did::{Did, DidResolver, VerifiableCredential, CredentialIssuer};
//!
//! // Create DID from key
//! let did = Did::new_key(&public_key)?;
//!
//! // Issue credential
//! let issuer = CredentialIssuer::new(keystore);
//! let vc = issuer.issue(subject, types, &issuer_did).await?;
//!
//! // Verify credential
//! let verifier = CredentialVerifier::new(resolver);
//! let result = verifier.verify(&vc).await?;
//! ```

pub mod did;
pub mod did_web;
pub mod key_management;
pub mod kms;
pub mod proof;
pub mod rdf_integration;
pub mod revocation;
#[cfg(feature = "bbs-plus")]
pub mod signatures;
pub mod signed_graph;
pub mod url;
pub mod vc;
#[cfg(feature = "zkp")]
pub mod zkp;

// v1.1.0 DID document versioning
pub mod document_versioning;

// v1.1.0: Verifiable Credential exchange protocols (VP creation, verification, JWT-like encoding)
pub mod credential_exchange;

// v1.1.0 round 5: DH/ECDH key agreement for DID-based communication
pub mod key_agreement;

// v1.1.0 round 6: W3C Verifiable Presentation builder
pub mod presentation_builder;

// v1.1.0 round 7: Verifiable Credential structural verification (W3C VC Data Model)
pub mod vc_verifier;

// v1.1.0 round 13: VP construction, credential selection, proof stubs, and selective disclosure
pub mod vc_presenter;

// v1.1.0 round 14: DID trust chain validation (leaf→root certification chain)
pub mod trust_chain;

// v1.1.0 round 15: DID authentication method management and challenge-response
pub mod authentication;

// v1.1.0 round 16: Verifiable Presentation request/response handling and validation
pub mod presentation_request;

// v1.1.0 round 11: In-memory DID document resolver with registration, deactivation and service management
pub mod did_resolver;

// v1.1.0 round 12: DID identity registry with resolution, update, deactivation, and method lookup
pub mod identity_registry;

// v1.1.0 round 13: W3C Verifiable Credential schema validation
pub mod credential_schema;

// v1.1.0 round 12: DID key lifecycle management (generation, rotation, status, purposes)
pub mod key_manager;

// v1.1.0 round 11: Linked Data Proof purpose validation (authentication, assertion, key agreement, capability)
pub mod proof_purpose;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Re-exports
#[cfg(feature = "did-ethr")]
pub use did::methods::{DidEthr, DidEthrMethod, EthNetwork};
#[cfg(feature = "did-ion")]
pub use did::methods::{
    DidIon, DidIonMethod, IonCreateOperation, IonDocument, IonKeyDescriptor, IonKeyPurpose,
    IonOperationType, IonService,
};
pub use did::{ChainNamespace, Did, DidDocument, DidPkh, DidPkhMethod, DidResolver};
pub use key_management::{
    generate_rotation_key, KeyExpiry, KeyRotation, KeyRotationManager, KeyRotationReason,
    KeyRotationRecord, KeyRotationRegistry, Keystore, LifecycleKeyRotationRecord,
    VerificationKey as ManagedVerificationKey,
};
pub use kms::{
    create_mock_kms, KeyUsage, KmsAlgorithm, KmsBackend, KmsDidSigner, KmsKeyMetadata, KmsProvider,
    MockAwsKms, MockAzureKms, MockGcpKms,
};
pub use proof::{
    jws::{
        attach_jws_proof, extract_jws_proof, sign_document, verify_document, CompactJws,
        JsonWebSignature2020, JwsAlgorithm, JwsHeader, JwsSigner, JwsVerifier,
    },
    Proof, ProofPurpose, ProofType,
};
pub use revocation::{
    BloomFilter, CredentialStatus, RevocationEntry, RevocationList2020, RevocationRegistry,
    RevocationRegistry2020, RevocationStatus, StatusList2021, StatusList2021Inner,
    StatusListCredential, StatusPurpose, MIN_LIST_SIZE,
};
#[cfg(feature = "bbs-plus")]
pub use signatures::{
    BbsKeyPair, BbsPlusSignature, BbsProof, BbsProofRequest, EcdsaJwsSigner, EcdsaJwsVerifier,
    Ed25519JwsSigner, Ed25519JwsVerifier, Es256Signer, Es256Verifier,
    JwsAlgorithm as SignaturesJwsAlgorithm, JwsHeader as SignaturesJwsHeader, JwsPayload,
    JwsSignature, JwsSignatureHeader, JwsSigner as SignaturesJwsSigner, JwsSignerTrait,
    JwsVerifier as SignaturesJwsVerifier, JwsVerifierTrait, MockJwsSigner, MockJwsVerifier,
    P256KeyPair, Rs256Signer, Rs256Verifier, RsaKeyPair,
};
pub use signed_graph::SignedGraph;
pub use url::{DereferencedResource, DidDereferencer, DidUrl};
pub use vc::{
    CredentialIssuer, CredentialSubject, CredentialVerifier, VerifiableCredential,
    VerifiablePresentation,
};
#[cfg(feature = "zkp")]
pub use zkp::{
    prove_selective, verify_selective, AttributeCommitment, CredentialAttribute,
    DisclosurePresentation, PedersenParams, PedersenSelectiveDisclosureProof, SchnorrProof,
    SelectiveDisclosureCredential, SelectiveDisclosureProof, SelectiveDisclosureRequest,
    ZkpProofRequest,
};

/// DID error types
#[derive(Error, Debug)]
pub enum DidError {
    #[error("Invalid DID format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported DID method: {0}")]
    UnsupportedMethod(String),

    #[error("Resolution failed: {0}")]
    ResolutionFailed(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Signing failed: {0}")]
    SigningFailed(String),

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Invalid key: {0}")]
    InvalidKey(String),

    #[error("Credential expired")]
    CredentialExpired,

    #[error("Invalid proof: {0}")]
    InvalidProof(String),

    #[error("Canonicalization failed: {0}")]
    CanonicalizationFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type DidResult<T> = Result<T, DidError>;

/// Verification method in DID Document
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationMethod {
    /// Verification method ID
    pub id: String,
    /// Type of verification method
    #[serde(rename = "type")]
    pub method_type: String,
    /// Controller DID
    pub controller: String,
    /// Public key in multibase format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_key_multibase: Option<String>,
    /// Public key in JWK format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_key_jwk: Option<serde_json::Value>,
    /// Blockchain account ID (CAIP-10 format, for did:pkh)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blockchain_account_id: Option<String>,
}

impl VerificationMethod {
    /// Create Ed25519 verification method
    pub fn ed25519(id: &str, controller: &str, public_key: &[u8]) -> Self {
        // Multibase encode with base58btc prefix 'z'
        let multibase = format!("z{}", bs58::encode(public_key).into_string());

        Self {
            id: id.to_string(),
            method_type: "Ed25519VerificationKey2020".to_string(),
            controller: controller.to_string(),
            public_key_multibase: Some(multibase),
            public_key_jwk: None,
            blockchain_account_id: None,
        }
    }

    /// Create a blockchain account verification method (for did:pkh)
    ///
    /// Uses CAIP-10 blockchain account ID format instead of a public key.
    pub fn blockchain(
        id: &str,
        controller: &str,
        method_type: &str,
        blockchain_account_id: &str,
    ) -> Self {
        Self {
            id: id.to_string(),
            method_type: method_type.to_string(),
            controller: controller.to_string(),
            public_key_multibase: None,
            public_key_jwk: None,
            blockchain_account_id: Some(blockchain_account_id.to_string()),
        }
    }

    /// Create a JWK verification method
    pub fn jwk(id: &str, controller: &str, method_type: &str, jwk: serde_json::Value) -> Self {
        Self {
            id: id.to_string(),
            method_type: method_type.to_string(),
            controller: controller.to_string(),
            public_key_multibase: None,
            public_key_jwk: Some(jwk),
            blockchain_account_id: None,
        }
    }

    /// Get public key bytes
    pub fn get_public_key_bytes(&self) -> DidResult<Vec<u8>> {
        if let Some(ref multibase) = self.public_key_multibase {
            // Remove multibase prefix and decode
            if let Some(stripped) = multibase.strip_prefix('z') {
                bs58::decode(stripped)
                    .into_vec()
                    .map_err(|e| DidError::InvalidKey(e.to_string()))
            } else {
                Err(DidError::InvalidKey("Unknown multibase prefix".to_string()))
            }
        } else if self.blockchain_account_id.is_some() {
            Err(DidError::InvalidKey(
                "Blockchain account verification methods do not expose raw public keys".to_string(),
            ))
        } else {
            Err(DidError::InvalidKey("No public key available".to_string()))
        }
    }
}

/// Service endpoint in DID Document
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Service {
    /// Service ID
    pub id: String,
    /// Service type
    #[serde(rename = "type")]
    pub service_type: String,
    /// Service endpoint URL
    pub service_endpoint: String,
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether verification succeeded
    pub valid: bool,
    /// Verified issuer DID
    pub issuer: Option<String>,
    /// Verification timestamp
    pub verified_at: DateTime<Utc>,
    /// Error message if verification failed
    pub error: Option<String>,
    /// Checks performed
    pub checks: Vec<VerificationCheck>,
}

/// Individual verification check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheck {
    /// Check name
    pub name: String,
    /// Whether check passed
    pub passed: bool,
    /// Details
    pub details: Option<String>,
}

impl VerificationResult {
    pub fn success(issuer: &str) -> Self {
        Self {
            valid: true,
            issuer: Some(issuer.to_string()),
            verified_at: Utc::now(),
            error: None,
            checks: vec![],
        }
    }

    pub fn failure(error: &str) -> Self {
        Self {
            valid: false,
            issuer: None,
            verified_at: Utc::now(),
            error: Some(error.to_string()),
            checks: vec![],
        }
    }

    pub fn with_check(mut self, name: &str, passed: bool, details: Option<&str>) -> Self {
        self.checks.push(VerificationCheck {
            name: name.to_string(),
            passed,
            details: details.map(String::from),
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_method_ed25519() {
        let public_key = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let vm = VerificationMethod::ed25519("did:key:z123#key-1", "did:key:z123", &public_key);

        assert_eq!(vm.method_type, "Ed25519VerificationKey2020");
        assert!(vm.public_key_multibase.is_some());

        let recovered = vm.get_public_key_bytes().unwrap();
        assert_eq!(recovered, public_key);
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult::success("did:key:z123")
            .with_check("signature", true, None)
            .with_check("expiration", true, Some("Not expired"));

        assert!(result.valid);
        assert_eq!(result.checks.len(), 2);
    }
}
