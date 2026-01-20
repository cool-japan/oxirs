//! Cryptographic proof module

pub mod ed25519;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Proof type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofType {
    /// Ed25519 Signature 2020
    #[serde(rename = "Ed25519Signature2020")]
    Ed25519Signature2020,
    /// Data Integrity Proof
    #[serde(rename = "DataIntegrityProof")]
    DataIntegrityProof,
    /// JSON Web Signature 2020
    #[serde(rename = "JsonWebSignature2020")]
    JsonWebSignature2020,
}

impl ProofType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProofType::Ed25519Signature2020 => "Ed25519Signature2020",
            ProofType::DataIntegrityProof => "DataIntegrityProof",
            ProofType::JsonWebSignature2020 => "JsonWebSignature2020",
        }
    }
}

/// Proof purpose
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProofPurpose {
    /// Assertion method (for credentials)
    AssertionMethod,
    /// Authentication
    Authentication,
    /// Key agreement
    KeyAgreement,
    /// Capability invocation
    CapabilityInvocation,
    /// Capability delegation
    CapabilityDelegation,
}

impl ProofPurpose {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProofPurpose::AssertionMethod => "assertionMethod",
            ProofPurpose::Authentication => "authentication",
            ProofPurpose::KeyAgreement => "keyAgreement",
            ProofPurpose::CapabilityInvocation => "capabilityInvocation",
            ProofPurpose::CapabilityDelegation => "capabilityDelegation",
        }
    }
}

/// Cryptographic proof
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Proof {
    /// Proof type
    #[serde(rename = "type")]
    pub proof_type: String,

    /// Creation timestamp
    pub created: DateTime<Utc>,

    /// Verification method (key ID)
    pub verification_method: String,

    /// Proof purpose
    pub proof_purpose: String,

    /// Proof value (signature in multibase)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_value: Option<String>,

    /// JWS (for JsonWebSignature2020)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jws: Option<String>,

    /// Challenge (for authentication)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,

    /// Domain
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,

    /// Nonce
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,

    /// Cryptosuite (for DataIntegrityProof)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cryptosuite: Option<String>,
}

impl Proof {
    /// Create a new Ed25519Signature2020 proof
    pub fn ed25519(verification_method: &str, purpose: ProofPurpose, signature: &[u8]) -> Self {
        // Encode signature in multibase (base58btc with 'z' prefix)
        let proof_value = format!("z{}", bs58::encode(signature).into_string());

        Self {
            proof_type: ProofType::Ed25519Signature2020.as_str().to_string(),
            created: Utc::now(),
            verification_method: verification_method.to_string(),
            proof_purpose: purpose.as_str().to_string(),
            proof_value: Some(proof_value),
            jws: None,
            challenge: None,
            domain: None,
            nonce: None,
            cryptosuite: None,
        }
    }

    /// Create a DataIntegrityProof
    pub fn data_integrity(
        cryptosuite: &str,
        verification_method: &str,
        purpose: ProofPurpose,
        signature: &[u8],
    ) -> Self {
        let proof_value = format!("z{}", bs58::encode(signature).into_string());

        Self {
            proof_type: ProofType::DataIntegrityProof.as_str().to_string(),
            created: Utc::now(),
            verification_method: verification_method.to_string(),
            proof_purpose: purpose.as_str().to_string(),
            proof_value: Some(proof_value),
            jws: None,
            challenge: None,
            domain: None,
            nonce: None,
            cryptosuite: Some(cryptosuite.to_string()),
        }
    }

    /// Get signature bytes from proof value
    pub fn get_signature_bytes(&self) -> crate::DidResult<Vec<u8>> {
        if let Some(ref proof_value) = self.proof_value {
            if let Some(stripped) = proof_value.strip_prefix('z') {
                bs58::decode(stripped)
                    .into_vec()
                    .map_err(|e| crate::DidError::InvalidProof(e.to_string()))
            } else {
                Err(crate::DidError::InvalidProof(
                    "Unknown proof value encoding".to_string(),
                ))
            }
        } else if let Some(ref _jws) = self.jws {
            // TODO: Parse JWS
            Err(crate::DidError::InvalidProof(
                "JWS parsing not yet implemented".to_string(),
            ))
        } else {
            Err(crate::DidError::InvalidProof("No proof value".to_string()))
        }
    }

    /// Set challenge (for authentication proofs)
    pub fn with_challenge(mut self, challenge: &str) -> Self {
        self.challenge = Some(challenge.to_string());
        self
    }

    /// Set domain
    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = Some(domain.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed25519_proof() {
        let signature = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let proof = Proof::ed25519(
            "did:key:z123#key-1",
            ProofPurpose::AssertionMethod,
            &signature,
        );

        assert_eq!(proof.proof_type, "Ed25519Signature2020");
        assert!(proof.proof_value.is_some());

        let recovered = proof.get_signature_bytes().unwrap();
        assert_eq!(recovered, signature);
    }

    #[test]
    fn test_data_integrity_proof() {
        let signature = vec![1, 2, 3, 4];
        let proof = Proof::data_integrity(
            "eddsa-rdfc-2022",
            "did:key:z456#key-1",
            ProofPurpose::Authentication,
            &signature,
        );

        assert_eq!(proof.proof_type, "DataIntegrityProof");
        assert_eq!(proof.cryptosuite, Some("eddsa-rdfc-2022".to_string()));
    }
}
