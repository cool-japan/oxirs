//! Cryptographic provenance chains with digital signatures
//!
//! This module provides secure, verifiable provenance tracking using
//! Ed25519 digital signatures for RDF-star annotations. Each provenance
//! record can be cryptographically signed to ensure authenticity and
//! integrity.

use crate::annotations::{ProvenanceRecord, TripleAnnotation};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use scirs2_core::random::{rng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors related to cryptographic provenance operations
#[derive(Error, Debug)]
pub enum CryptoProvenanceError {
    #[error("Signature verification failed: {0}")]
    VerificationFailed(String),

    #[error("Invalid signature format: {0}")]
    InvalidSignature(String),

    #[error("Invalid public key format: {0}")]
    InvalidPublicKey(String),

    #[error("Missing signature in provenance record")]
    MissingSignature,

    #[error("Provenance chain broken at index {0}")]
    BrokenChain(usize),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Cryptographically signed provenance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedProvenanceRecord {
    /// The underlying provenance record
    pub record: ProvenanceRecord,

    /// Digital signature of the record content
    pub signature: String,

    /// Public key of the signer (hex-encoded)
    pub public_key: String,

    /// Hash of the previous record in the chain (for integrity)
    pub previous_hash: Option<String>,

    /// Chain index (position in the provenance chain)
    pub chain_index: usize,
}

/// Key pair for signing provenance records
#[derive(Clone)]
pub struct ProvenanceKeyPair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl ProvenanceKeyPair {
    /// Generate a new random key pair
    pub fn generate() -> Self {
        let mut rng_instance = rng();
        let mut seed = [0u8; 32];
        rng_instance.fill_bytes(&mut seed);

        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();

        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Create key pair from existing seed
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(seed);
        let verifying_key = signing_key.verifying_key();

        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Get the public key as hex string
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.verifying_key.as_bytes())
    }

    /// Get the public key bytes
    pub fn public_key_bytes(&self) -> &[u8; 32] {
        self.verifying_key.as_bytes()
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Verify a signature
    pub fn verify(
        &self,
        message: &[u8],
        signature: &Signature,
    ) -> Result<(), CryptoProvenanceError> {
        self.verifying_key
            .verify(message, signature)
            .map_err(|e| CryptoProvenanceError::VerificationFailed(e.to_string()))
    }
}

/// Manager for cryptographic provenance chains
pub struct CryptoProvenanceManager {
    /// Key pairs indexed by agent identifier
    key_pairs: HashMap<String, ProvenanceKeyPair>,
}

impl CryptoProvenanceManager {
    /// Create a new provenance manager
    pub fn new() -> Self {
        Self {
            key_pairs: HashMap::new(),
        }
    }

    /// Register a key pair for an agent
    pub fn register_agent(&mut self, agent: String, key_pair: ProvenanceKeyPair) {
        debug!(
            "Registering agent '{}' with public key {}",
            agent,
            key_pair.public_key_hex()
        );
        self.key_pairs.insert(agent, key_pair);
    }

    /// Generate and register a new key pair for an agent
    pub fn generate_agent_key(&mut self, agent: String) -> ProvenanceKeyPair {
        let key_pair = ProvenanceKeyPair::generate();
        info!("Generated new key pair for agent '{}'", agent);
        self.register_agent(agent, key_pair.clone());
        key_pair
    }

    /// Sign a provenance record
    pub fn sign_record(
        &self,
        record: ProvenanceRecord,
        agent: &str,
        previous_hash: Option<String>,
        chain_index: usize,
    ) -> Result<SignedProvenanceRecord, CryptoProvenanceError> {
        let key_pair = self.key_pairs.get(agent).ok_or_else(|| {
            CryptoProvenanceError::VerificationFailed(format!("Agent '{}' not registered", agent))
        })?;

        // Serialize the record for signing
        let record_json = serde_json::to_string(&record)
            .map_err(|e| CryptoProvenanceError::SerializationError(e.to_string()))?;

        // Create message to sign (includes previous hash for chain integrity)
        let mut message = record_json.clone();
        if let Some(ref prev_hash) = previous_hash {
            message.push_str(prev_hash);
        }
        message.push_str(&chain_index.to_string());

        // Sign the message
        let signature = key_pair.sign(message.as_bytes());

        Ok(SignedProvenanceRecord {
            record,
            signature: hex::encode(signature.to_bytes()),
            public_key: key_pair.public_key_hex(),
            previous_hash,
            chain_index,
        })
    }

    /// Verify a signed provenance record
    pub fn verify_record(
        &self,
        signed_record: &SignedProvenanceRecord,
    ) -> Result<(), CryptoProvenanceError> {
        // Decode public key
        let public_key_bytes = hex::decode(&signed_record.public_key)
            .map_err(|e| CryptoProvenanceError::InvalidPublicKey(e.to_string()))?;

        let public_key_array: [u8; 32] = public_key_bytes.try_into().map_err(|_| {
            CryptoProvenanceError::InvalidPublicKey("Invalid key length".to_string())
        })?;

        let verifying_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|e| CryptoProvenanceError::InvalidPublicKey(e.to_string()))?;

        // Decode signature
        let signature_bytes = hex::decode(&signed_record.signature)
            .map_err(|e| CryptoProvenanceError::InvalidSignature(e.to_string()))?;

        let signature_array: [u8; 64] = signature_bytes.try_into().map_err(|_| {
            CryptoProvenanceError::InvalidSignature("Invalid signature length".to_string())
        })?;

        let signature = Signature::from_bytes(&signature_array);

        // Reconstruct the signed message
        let record_json = serde_json::to_string(&signed_record.record)
            .map_err(|e| CryptoProvenanceError::SerializationError(e.to_string()))?;

        let mut message = record_json;
        if let Some(ref prev_hash) = signed_record.previous_hash {
            message.push_str(prev_hash);
        }
        message.push_str(&signed_record.chain_index.to_string());

        // Verify signature
        verifying_key
            .verify(message.as_bytes(), &signature)
            .map_err(|e| CryptoProvenanceError::VerificationFailed(e.to_string()))
    }

    /// Verify an entire provenance chain
    pub fn verify_chain(
        &self,
        chain: &[SignedProvenanceRecord],
    ) -> Result<(), CryptoProvenanceError> {
        if chain.is_empty() {
            return Ok(());
        }

        // Verify first record has no previous hash
        if chain[0].previous_hash.is_some() {
            return Err(CryptoProvenanceError::BrokenChain(0));
        }

        // Verify each record
        for (i, record) in chain.iter().enumerate() {
            // Verify signature
            self.verify_record(record)?;

            // Verify chain index
            if record.chain_index != i {
                warn!(
                    "Chain index mismatch at position {}: expected {}, got {}",
                    i, i, record.chain_index
                );
                return Err(CryptoProvenanceError::BrokenChain(i));
            }

            // Verify hash linkage (except for first record)
            if i > 0 {
                let prev_hash = compute_record_hash(&chain[i - 1]);
                if record.previous_hash.as_ref() != Some(&prev_hash) {
                    warn!("Hash chain broken at position {}", i);
                    return Err(CryptoProvenanceError::BrokenChain(i));
                }
            }
        }

        info!(
            "Provenance chain verified successfully ({} records)",
            chain.len()
        );
        Ok(())
    }

    /// Create a signed provenance chain from regular provenance records
    pub fn create_signed_chain(
        &self,
        records: Vec<ProvenanceRecord>,
    ) -> Result<Vec<SignedProvenanceRecord>, CryptoProvenanceError> {
        let mut signed_chain = Vec::new();
        let mut previous_hash = None;

        for (index, record) in records.into_iter().enumerate() {
            let agent = record.agent.clone();
            let signed_record = self.sign_record(record, &agent, previous_hash.clone(), index)?;

            previous_hash = Some(compute_record_hash(&signed_record));
            signed_chain.push(signed_record);
        }

        Ok(signed_chain)
    }
}

impl Default for CryptoProvenanceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute SHA-256 hash of a signed provenance record
pub fn compute_record_hash(record: &SignedProvenanceRecord) -> String {
    let mut hasher = Sha256::new();

    // Hash the record content
    let record_json = serde_json::to_string(&record.record).unwrap_or_default();
    hasher.update(record_json.as_bytes());
    hasher.update(&record.signature);
    hasher.update(&record.public_key);
    hasher.update(record.chain_index.to_le_bytes());

    hex::encode(hasher.finalize())
}

/// Extension trait for TripleAnnotation to add cryptographic provenance
pub trait CryptoProvenanceExt {
    /// Convert regular provenance to signed provenance
    fn to_signed_provenance(
        &self,
        manager: &CryptoProvenanceManager,
    ) -> Result<Vec<SignedProvenanceRecord>, CryptoProvenanceError>;

    /// Verify all signed provenance in the annotation
    fn verify_provenance(
        &self,
        manager: &CryptoProvenanceManager,
    ) -> Result<(), CryptoProvenanceError>;

    /// Add a new signed provenance record
    fn add_signed_provenance(
        &mut self,
        record: ProvenanceRecord,
        manager: &CryptoProvenanceManager,
    ) -> Result<SignedProvenanceRecord, CryptoProvenanceError>;
}

/// Signed annotation with cryptographic provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedAnnotation {
    /// The base annotation
    pub annotation: TripleAnnotation,

    /// Cryptographically signed provenance chain
    pub signed_provenance: Vec<SignedProvenanceRecord>,

    /// Annotation signature (signature of the entire annotation)
    pub annotation_signature: Option<String>,

    /// Public key of annotation creator
    pub creator_public_key: Option<String>,
}

impl SignedAnnotation {
    /// Create a new signed annotation
    pub fn new(annotation: TripleAnnotation) -> Self {
        Self {
            annotation,
            signed_provenance: Vec::new(),
            annotation_signature: None,
            creator_public_key: None,
        }
    }

    /// Sign the entire annotation
    pub fn sign_annotation(
        &mut self,
        key_pair: &ProvenanceKeyPair,
    ) -> Result<(), CryptoProvenanceError> {
        let annotation_json = serde_json::to_string(&self.annotation)
            .map_err(|e| CryptoProvenanceError::SerializationError(e.to_string()))?;

        let signature = key_pair.sign(annotation_json.as_bytes());

        self.annotation_signature = Some(hex::encode(signature.to_bytes()));
        self.creator_public_key = Some(key_pair.public_key_hex());

        Ok(())
    }

    /// Verify the annotation signature
    pub fn verify_annotation_signature(&self) -> Result<(), CryptoProvenanceError> {
        let signature_hex = self
            .annotation_signature
            .as_ref()
            .ok_or(CryptoProvenanceError::MissingSignature)?;

        let public_key_hex =
            self.creator_public_key
                .as_ref()
                .ok_or(CryptoProvenanceError::InvalidPublicKey(
                    "Missing public key".to_string(),
                ))?;

        // Decode public key
        let public_key_bytes = hex::decode(public_key_hex)
            .map_err(|e| CryptoProvenanceError::InvalidPublicKey(e.to_string()))?;

        let public_key_array: [u8; 32] = public_key_bytes.try_into().map_err(|_| {
            CryptoProvenanceError::InvalidPublicKey("Invalid key length".to_string())
        })?;

        let verifying_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|e| CryptoProvenanceError::InvalidPublicKey(e.to_string()))?;

        // Decode signature
        let signature_bytes = hex::decode(signature_hex)
            .map_err(|e| CryptoProvenanceError::InvalidSignature(e.to_string()))?;

        let signature_array: [u8; 64] = signature_bytes.try_into().map_err(|_| {
            CryptoProvenanceError::InvalidSignature("Invalid signature length".to_string())
        })?;

        let signature = Signature::from_bytes(&signature_array);

        // Reconstruct message
        let annotation_json = serde_json::to_string(&self.annotation)
            .map_err(|e| CryptoProvenanceError::SerializationError(e.to_string()))?;

        verifying_key
            .verify(annotation_json.as_bytes(), &signature)
            .map_err(|e| CryptoProvenanceError::VerificationFailed(e.to_string()))
    }

    /// Verify both annotation signature and provenance chain
    pub fn verify_all(
        &self,
        manager: &CryptoProvenanceManager,
    ) -> Result<(), CryptoProvenanceError> {
        // Verify annotation signature if present
        if self.annotation_signature.is_some() {
            self.verify_annotation_signature()?;
        }

        // Verify provenance chain
        manager.verify_chain(&self.signed_provenance)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_key_pair_generation() {
        let key_pair = ProvenanceKeyPair::generate();
        let public_key = key_pair.public_key_hex();

        assert_eq!(public_key.len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_sign_and_verify_record() {
        let mut manager = CryptoProvenanceManager::new();
        let agent = "agent1".to_string();

        manager.generate_agent_key(agent.clone());

        let record = ProvenanceRecord {
            action: "created".to_string(),
            agent: agent.clone(),
            timestamp: Utc::now(),
            activity: Some("test activity".to_string()),
            method: Some("manual".to_string()),
        };

        let signed = manager.sign_record(record, &agent, None, 0).unwrap();

        assert!(manager.verify_record(&signed).is_ok());
    }

    #[test]
    fn test_provenance_chain() {
        let mut manager = CryptoProvenanceManager::new();
        let agent = "agent1".to_string();

        manager.generate_agent_key(agent.clone());

        let records = vec![
            ProvenanceRecord {
                action: "created".to_string(),
                agent: agent.clone(),
                timestamp: Utc::now(),
                activity: Some("creation".to_string()),
                method: Some("manual".to_string()),
            },
            ProvenanceRecord {
                action: "modified".to_string(),
                agent: agent.clone(),
                timestamp: Utc::now(),
                activity: Some("modification".to_string()),
                method: Some("automatic".to_string()),
            },
        ];

        let chain = manager.create_signed_chain(records).unwrap();

        assert_eq!(chain.len(), 2);
        assert!(manager.verify_chain(&chain).is_ok());
    }

    #[test]
    fn test_tampered_chain_detection() {
        let mut manager = CryptoProvenanceManager::new();
        let agent = "agent1".to_string();

        manager.generate_agent_key(agent.clone());

        let records = vec![
            ProvenanceRecord {
                action: "created".to_string(),
                agent: agent.clone(),
                timestamp: Utc::now(),
                activity: Some("creation".to_string()),
                method: Some("manual".to_string()),
            },
            ProvenanceRecord {
                action: "modified".to_string(),
                agent: agent.clone(),
                timestamp: Utc::now(),
                activity: Some("modification".to_string()),
                method: Some("automatic".to_string()),
            },
        ];

        let mut chain = manager.create_signed_chain(records).unwrap();

        // Tamper with the second record
        chain[1].record.action = "tampered".to_string();

        // Verification should fail
        assert!(manager.verify_record(&chain[1]).is_err());
    }

    #[test]
    fn test_signed_annotation() {
        let key_pair = ProvenanceKeyPair::generate();
        let mut annotation = TripleAnnotation::new();
        annotation.confidence = Some(0.9);
        annotation.source = Some("test source".to_string());

        let mut signed_annotation = SignedAnnotation::new(annotation);
        signed_annotation.sign_annotation(&key_pair).unwrap();

        assert!(signed_annotation.verify_annotation_signature().is_ok());
    }

    #[test]
    fn test_tampered_annotation_detection() {
        let key_pair = ProvenanceKeyPair::generate();
        let mut annotation = TripleAnnotation::new();
        annotation.confidence = Some(0.9);

        let mut signed_annotation = SignedAnnotation::new(annotation);
        signed_annotation.sign_annotation(&key_pair).unwrap();

        // Tamper with the annotation
        signed_annotation.annotation.confidence = Some(0.1);

        // Verification should fail
        assert!(signed_annotation.verify_annotation_signature().is_err());
    }
}
