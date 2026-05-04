//! Simulated PKCS#11 HSM signer for DID key management.
//!
//! This module provides a software-simulated PKCS#11 interface that can be
//! used for testing and development.  In production, this module would be
//! replaced by a real PKCS#11 library binding (e.g., `cryptoki` crate).
//!
//! # Feature gate
//! This module is available when the crate is built with `feature = "hsm"`.
//!
//! # Security notice
//! The current implementation stores key material in memory.  A production
//! HSM signer must **never** export key material from the hardware boundary.
//! Replace [`Pkcs11Slot::sign`] with actual PKCS#11 `C_Sign` calls.

use crate::{DidError, DidResult};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ─────────────────────────────────────────────────────────────────────────────
// Key handle
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque handle to a key object in an HSM slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyHandle(pub u64);

// ─────────────────────────────────────────────────────────────────────────────
// PKCS#11 Mechanism
// ─────────────────────────────────────────────────────────────────────────────

/// Subset of PKCS#11 signing mechanisms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkcs11Mechanism {
    /// Ed25519 signature (CKM_EDDSA)
    CkmEddsa,
    /// ECDSA with SHA-256 (CKM_ECDSA_SHA256)
    CkmEcdsaSha256,
    /// RSASSA-PSS with SHA-256 (CKM_SHA256_RSA_PKCS_PSS)
    CkmSha256RsaPkcsPss,
}

impl Pkcs11Mechanism {
    pub fn name(self) -> &'static str {
        match self {
            Self::CkmEddsa => "CKM_EDDSA",
            Self::CkmEcdsaSha256 => "CKM_ECDSA_SHA256",
            Self::CkmSha256RsaPkcsPss => "CKM_SHA256_RSA_PKCS_PSS",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Key object stored in a slot
// ─────────────────────────────────────────────────────────────────────────────

/// Simulated key stored inside an HSM slot.
#[derive(Debug, Clone)]
pub struct Pkcs11KeyObject {
    /// Key label (CKA_LABEL).
    pub label: String,
    /// Mechanism this key supports.
    pub mechanism: Pkcs11Mechanism,
    /// Raw key bytes (simulation only — never exported in real HSMs).
    key_bytes: Vec<u8>,
    /// Public key bytes (may be exposed).
    pub public_key: Vec<u8>,
}

impl Pkcs11KeyObject {
    fn new(
        label: &str,
        mechanism: Pkcs11Mechanism,
        key_bytes: Vec<u8>,
        public_key: Vec<u8>,
    ) -> Self {
        Self {
            label: label.to_string(),
            mechanism,
            key_bytes,
            public_key,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PKCS#11 Slot (simulated token)
// ─────────────────────────────────────────────────────────────────────────────

/// A simulated PKCS#11 token slot.
///
/// In a real implementation, this wraps a `CK_SESSION_HANDLE` and delegates
/// to the underlying PKCS#11 module (`.so` / `.dll`).
pub struct Pkcs11Slot {
    slot_id: u64,
    label: String,
    keys: RwLock<HashMap<KeyHandle, Pkcs11KeyObject>>,
    next_handle: RwLock<u64>,
}

impl Pkcs11Slot {
    /// Create a new (empty) simulated slot.
    pub fn new(slot_id: u64, label: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            slot_id,
            label: label.into(),
            keys: RwLock::new(HashMap::new()),
            next_handle: RwLock::new(1),
        })
    }

    /// Slot identifier.
    pub fn slot_id(&self) -> u64 {
        self.slot_id
    }

    /// Slot label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Generate a simulated key pair and return its handle.
    ///
    /// In a real implementation this calls `C_GenerateKeyPair`.
    pub fn generate_key(&self, label: &str, mechanism: Pkcs11Mechanism) -> DidResult<KeyHandle> {
        // Simulate key generation — derive key bytes from slot_id + label + mechanism
        // NEVER do this in production; use the HSM's key generation hardware.
        let mut seed = format!("{}|{}|{}", self.slot_id, label, mechanism.name());
        let key_bytes: Vec<u8> = Sha256::digest(seed.as_bytes()).to_vec();
        seed.push_str("|pub");
        let public_key: Vec<u8> = Sha256::digest(seed.as_bytes()).to_vec();

        let obj = Pkcs11KeyObject::new(label, mechanism, key_bytes, public_key);

        let handle = {
            let mut next = self
                .next_handle
                .write()
                .map_err(|_| DidError::InternalError("Lock poisoned".into()))?;
            let h = KeyHandle(*next);
            *next += 1;
            h
        };
        self.keys
            .write()
            .map_err(|_| DidError::InternalError("Lock poisoned".into()))?
            .insert(handle, obj);
        Ok(handle)
    }

    /// Sign `data` using the key referenced by `handle`.
    ///
    /// In a real implementation this calls `C_Sign`.
    /// The simulated signature is `HMAC-SHA256(key_bytes, data)`.
    pub fn sign(&self, handle: KeyHandle, data: &[u8]) -> DidResult<Vec<u8>> {
        let keys = self
            .keys
            .read()
            .map_err(|_| DidError::InternalError("Lock poisoned".into()))?;
        let obj = keys
            .get(&handle)
            .ok_or_else(|| DidError::KeyNotFound(format!("Key handle {:?} not found", handle)))?;

        // Simulated signature: SHA-256(key_bytes || data)
        let mut hasher = Sha256::new();
        hasher.update(&obj.key_bytes);
        hasher.update(data);
        Ok(hasher.finalize().to_vec())
    }

    /// Retrieve the public key bytes for the given handle.
    pub fn get_public_key(&self, handle: KeyHandle) -> DidResult<Vec<u8>> {
        let keys = self
            .keys
            .read()
            .map_err(|_| DidError::InternalError("Lock poisoned".into()))?;
        let obj = keys
            .get(&handle)
            .ok_or_else(|| DidError::KeyNotFound(format!("Key handle {:?} not found", handle)))?;
        Ok(obj.public_key.clone())
    }

    /// List all key labels in this slot.
    pub fn list_key_labels(&self) -> DidResult<Vec<String>> {
        let keys = self
            .keys
            .read()
            .map_err(|_| DidError::InternalError("Lock poisoned".into()))?;
        Ok(keys.values().map(|o| o.label.clone()).collect())
    }

    /// Delete a key by handle (simulates `C_DestroyObject`).
    pub fn destroy_key(&self, handle: KeyHandle) -> DidResult<()> {
        let mut keys = self
            .keys
            .write()
            .map_err(|_| DidError::InternalError("Lock poisoned".into()))?;
        if keys.remove(&handle).is_none() {
            return Err(DidError::KeyNotFound(format!(
                "Key handle {:?} not found",
                handle
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_sign() {
        let slot = Pkcs11Slot::new(0, "TestToken");
        let handle = slot
            .generate_key("my-key", Pkcs11Mechanism::CkmEddsa)
            .unwrap();
        let sig = slot.sign(handle, b"hello world").unwrap();
        assert_eq!(sig.len(), 32, "SHA-256 signature is 32 bytes");
    }

    #[test]
    fn test_sign_deterministic() {
        let slot = Pkcs11Slot::new(1, "DeterministicSlot");
        let h = slot
            .generate_key("key1", Pkcs11Mechanism::CkmEcdsaSha256)
            .unwrap();
        let sig1 = slot.sign(h, b"data").unwrap();
        let sig2 = slot.sign(h, b"data").unwrap();
        assert_eq!(sig1, sig2, "simulated sign must be deterministic");
    }

    #[test]
    fn test_get_public_key() {
        let slot = Pkcs11Slot::new(2, "PubKeySlot");
        let h = slot.generate_key("k", Pkcs11Mechanism::CkmEddsa).unwrap();
        let pub_key = slot.get_public_key(h).unwrap();
        assert_eq!(pub_key.len(), 32, "simulated public key is 32 bytes");
    }

    #[test]
    fn test_list_key_labels() {
        let slot = Pkcs11Slot::new(3, "ListSlot");
        slot.generate_key("alpha", Pkcs11Mechanism::CkmEddsa)
            .unwrap();
        slot.generate_key("beta", Pkcs11Mechanism::CkmEcdsaSha256)
            .unwrap();
        let labels = slot.list_key_labels().unwrap();
        assert!(labels.contains(&"alpha".to_string()));
        assert!(labels.contains(&"beta".to_string()));
    }

    #[test]
    fn test_destroy_key() {
        let slot = Pkcs11Slot::new(4, "DestroySlot");
        let h = slot
            .generate_key("temp", Pkcs11Mechanism::CkmEddsa)
            .unwrap();
        slot.destroy_key(h).unwrap();
        assert!(
            slot.sign(h, b"after destroy").is_err(),
            "destroyed key must not sign"
        );
    }

    #[test]
    fn test_unknown_handle_errors() {
        let slot = Pkcs11Slot::new(5, "EmptySlot");
        let bogus = KeyHandle(999);
        assert!(slot.sign(bogus, b"data").is_err());
        assert!(slot.get_public_key(bogus).is_err());
        assert!(slot.destroy_key(bogus).is_err());
    }

    #[test]
    fn test_mechanism_names() {
        assert_eq!(Pkcs11Mechanism::CkmEddsa.name(), "CKM_EDDSA");
        assert_eq!(Pkcs11Mechanism::CkmEcdsaSha256.name(), "CKM_ECDSA_SHA256");
    }
}
