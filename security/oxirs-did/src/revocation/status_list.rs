//! StatusList2021 implementation for VC/DID revocation
//!
//! Implements the W3C Verifiable Credentials Status specification:
//! <https://www.w3.org/TR/vc-status-list-2021/>
//!
//! StatusList2021 uses a compressed bitstring to efficiently represent
//! the revocation status of many credentials in a single list.
//! A credential is revoked if its bit at its assigned index is set to 1.
//!
//! The bitstring is GZIP-compressed and then base64url-encoded.
//! Each credential holder is assigned a unique index into this list.
//!
//! # Example
//! ```
//! use oxirs_did::revocation::StatusList2021;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut list = StatusList2021::new("https://example.com/status/1", "https://example.com/issuer", 131072)?;
//! list.set_status(42, true)?;  // Revoke credential at index 42
//! assert!(list.is_revoked(42)?);
//! assert!(!list.is_revoked(43)?);
//! # Ok(())
//! # }
//! ```

use crate::{DidError, DidResult};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};

/// Minimum list size (per spec, at least 131072 entries to prevent correlation attacks)
pub const MIN_LIST_SIZE: usize = 131_072;

/// The status purpose of this list
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StatusPurpose {
    /// Credential revocation
    Revocation,
    /// Credential suspension (temporary revocation)
    Suspension,
}

impl StatusPurpose {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Suspension => "suspension",
        }
    }
}

/// StatusList2021 bitstring credential status list
///
/// Stores a compressed bitstring representing the revocation status
/// of credentials indexed into this list.
pub struct StatusList2021 {
    /// Unique identifier for this status list
    pub id: String,
    /// The issuer DID or URL
    pub issuer: String,
    /// Status purpose (revocation or suspension)
    pub purpose: StatusPurpose,
    /// The raw bitstring (uncompressed, in-memory)
    bits: Vec<bool>,
    /// Total capacity of the list
    size: usize,
}

impl StatusList2021 {
    /// Create a new StatusList2021 with the given size
    ///
    /// # Arguments
    /// * `id` - The URL identifier for this status list
    /// * `issuer` - The issuer DID or URL
    /// * `size` - Number of credential slots (minimum 131072 per spec)
    pub fn new(id: &str, issuer: &str, size: usize) -> DidResult<Self> {
        if size == 0 {
            return Err(DidError::InvalidFormat(
                "StatusList2021 size must be greater than zero".to_string(),
            ));
        }

        Ok(Self {
            id: id.to_string(),
            issuer: issuer.to_string(),
            purpose: StatusPurpose::Revocation,
            bits: vec![false; size],
            size,
        })
    }

    /// Create a new StatusList2021 with a custom purpose
    pub fn new_with_purpose(
        id: &str,
        issuer: &str,
        size: usize,
        purpose: StatusPurpose,
    ) -> DidResult<Self> {
        let mut list = Self::new(id, issuer, size)?;
        list.purpose = purpose;
        Ok(list)
    }

    /// Get the size of the status list
    pub fn size(&self) -> usize {
        self.size
    }

    /// Set the revocation/suspension status of a credential
    ///
    /// # Arguments
    /// * `index` - The credential's assigned index in the list
    /// * `revoked` - True to revoke/suspend, false to reinstate
    pub fn set_status(&mut self, index: usize, revoked: bool) -> DidResult<()> {
        if index >= self.size {
            return Err(DidError::InvalidFormat(format!(
                "Index {} out of bounds for list of size {}",
                index, self.size
            )));
        }
        self.bits[index] = revoked;
        Ok(())
    }

    /// Check if a credential is revoked/suspended
    ///
    /// # Arguments
    /// * `index` - The credential's assigned index in the list
    pub fn is_revoked(&self, index: usize) -> DidResult<bool> {
        if index >= self.size {
            return Err(DidError::InvalidFormat(format!(
                "Index {} out of bounds for list of size {}",
                index, self.size
            )));
        }
        Ok(self.bits[index])
    }

    /// Get the number of revoked credentials
    pub fn revoked_count(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    /// Encode the bitstring for transmission
    ///
    /// The spec requires GZIP compression + base64url encoding.
    /// We implement a pure-Rust bit-packing approach (8 bits per byte)
    /// followed by base64url encoding.
    ///
    /// In a production implementation, GZIP (via flate2) would further compress this.
    pub fn encode_bitstring(&self) -> DidResult<String> {
        let byte_count = (self.size + 7) / 8;
        let mut bytes = vec![0u8; byte_count];

        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                let byte_idx = i / 8;
                let bit_idx = 7 - (i % 8); // MSB first within each byte
                bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        // Apply simple RLE-like compression: just base64url-encode the packed bytes
        // Note: A full implementation would use GZIP here
        Ok(URL_SAFE_NO_PAD.encode(&bytes))
    }

    /// Decode a bitstring from the encoded format
    pub fn decode_bitstring(encoded: &str, expected_size: usize) -> DidResult<Vec<bool>> {
        let bytes = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|e| DidError::SerializationError(format!("Base64 decode error: {}", e)))?;

        let mut bits = Vec::with_capacity(expected_size);
        for byte in &bytes {
            for bit_idx in (0..8).rev() {
                bits.push((byte >> bit_idx) & 1 == 1);
                if bits.len() >= expected_size {
                    break;
                }
            }
            if bits.len() >= expected_size {
                break;
            }
        }

        // Pad to expected size if needed
        while bits.len() < expected_size {
            bits.push(false);
        }

        Ok(bits)
    }

    /// Serialize this StatusList2021 as a Verifiable Credential JSON-LD document
    ///
    /// The resulting document can be published at the list's URL for
    /// verifiers to fetch and check credential status.
    pub fn to_credential(&self) -> DidResult<serde_json::Value> {
        let encoded_list = self.encode_bitstring()?;

        Ok(serde_json::json!({
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://w3id.org/vc/status-list/2021/v1"
            ],
            "id": self.id,
            "type": ["VerifiableCredential", "StatusList2021Credential"],
            "issuer": self.issuer,
            "issuedAt": chrono::Utc::now().to_rfc3339(),
            "credentialSubject": {
                "id": format!("{}#list", self.id),
                "type": "StatusList2021",
                "statusPurpose": self.purpose.as_str(),
                "encodedList": encoded_list
            }
        }))
    }

    /// Deserialize from a Verifiable Credential JSON-LD document
    pub fn from_credential(
        credential: &serde_json::Value,
        expected_size: usize,
    ) -> DidResult<Self> {
        let id = credential["id"]
            .as_str()
            .ok_or_else(|| DidError::SerializationError("Missing credential id".to_string()))?
            .to_string();

        let issuer = match &credential["issuer"] {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(obj) => obj["id"].as_str().unwrap_or_default().to_string(),
            _ => {
                return Err(DidError::SerializationError(
                    "Invalid issuer format".to_string(),
                ))
            }
        };

        let subject = &credential["credentialSubject"];

        let purpose_str = subject["statusPurpose"].as_str().unwrap_or("revocation");

        let purpose = match purpose_str {
            "revocation" => StatusPurpose::Revocation,
            "suspension" => StatusPurpose::Suspension,
            other => {
                return Err(DidError::InvalidFormat(format!(
                    "Unknown status purpose: {}",
                    other
                )))
            }
        };

        let encoded_list = subject["encodedList"]
            .as_str()
            .ok_or_else(|| DidError::SerializationError("Missing encodedList".to_string()))?;

        let bits = Self::decode_bitstring(encoded_list, expected_size)?;
        let size = bits.len();

        Ok(Self {
            id,
            issuer,
            purpose,
            bits,
            size,
        })
    }

    /// Batch-set multiple indices at once
    pub fn set_batch(&mut self, indices: &[(usize, bool)]) -> DidResult<()> {
        for &(index, revoked) in indices {
            self.set_status(index, revoked)?;
        }
        Ok(())
    }

    /// Get all revoked indices
    pub fn revoked_indices(&self) -> Vec<usize> {
        self.bits
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Credential status entry that gets embedded in a Verifiable Credential
///
/// This is added to a credential to specify which status list
/// and index to check for revocation status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialStatus {
    /// Unique ID for this credential status entry
    pub id: String,
    /// Status type (StatusList2021Entry)
    #[serde(rename = "type")]
    pub status_type: String,
    /// Index in the status list
    pub status_list_index: String,
    /// URL of the StatusList2021 credential
    pub status_list_credential: String,
    /// Status purpose (revocation or suspension)
    pub status_purpose: String,
}

impl CredentialStatus {
    /// Create a new StatusList2021 credential status entry
    ///
    /// # Arguments
    /// * `status_list_url` - URL where the StatusList2021 credential is hosted
    /// * `index` - This credential's assigned index in the list
    /// * `purpose` - The purpose of this status check
    pub fn new_status_list_2021(
        status_list_url: &str,
        index: usize,
        purpose: StatusPurpose,
    ) -> Self {
        Self {
            id: format!("{}#{}", status_list_url, index),
            status_type: "StatusList2021Entry".to_string(),
            status_list_index: index.to_string(),
            status_list_credential: status_list_url.to_string(),
            status_purpose: purpose.as_str().to_string(),
        }
    }

    /// Get the index as a usize
    pub fn index(&self) -> DidResult<usize> {
        self.status_list_index
            .parse::<usize>()
            .map_err(|e| DidError::InvalidFormat(format!("Invalid status list index: {}", e)))
    }
}

/// Simple in-memory revocation registry for managing credential status
///
/// Provides a higher-level API for managing revocation without
/// manual index tracking.
pub struct RevocationRegistry {
    /// The underlying status list
    status_list: StatusList2021,
    /// Next available index for new credentials
    next_index: usize,
    /// Map from credential ID to assigned index
    credential_indices: std::collections::HashMap<String, usize>,
}

impl RevocationRegistry {
    /// Create a new registry with the given capacity
    pub fn new(list_id: &str, issuer: &str, capacity: usize) -> DidResult<Self> {
        let size = capacity.max(MIN_LIST_SIZE);
        let status_list = StatusList2021::new(list_id, issuer, size)?;

        Ok(Self {
            status_list,
            next_index: 0,
            credential_indices: std::collections::HashMap::new(),
        })
    }

    /// Register a new credential and get its status entry
    ///
    /// Returns the CredentialStatus to embed in the issued credential.
    pub fn register_credential(&mut self, credential_id: &str) -> DidResult<CredentialStatus> {
        if self.next_index >= self.status_list.size() {
            return Err(DidError::InternalError(
                "Status list capacity exhausted".to_string(),
            ));
        }

        let index = self.next_index;
        self.next_index += 1;
        self.credential_indices
            .insert(credential_id.to_string(), index);

        let status = CredentialStatus::new_status_list_2021(
            &self.status_list.id,
            index,
            self.status_list.purpose,
        );

        Ok(status)
    }

    /// Revoke a credential by its ID
    pub fn revoke(&mut self, credential_id: &str) -> DidResult<()> {
        let index = self.get_credential_index(credential_id)?;
        self.status_list.set_status(index, true)
    }

    /// Reinstate a previously revoked credential
    pub fn reinstate(&mut self, credential_id: &str) -> DidResult<()> {
        let index = self.get_credential_index(credential_id)?;
        self.status_list.set_status(index, false)
    }

    /// Check if a credential is revoked
    pub fn is_revoked(&self, credential_id: &str) -> DidResult<bool> {
        let index = self.get_credential_index(credential_id)?;
        self.status_list.is_revoked(index)
    }

    /// Check status by index (for verifiers using the credential status entry)
    pub fn is_revoked_at_index(&self, index: usize) -> DidResult<bool> {
        self.status_list.is_revoked(index)
    }

    /// Get the StatusList2021 credential for publishing
    pub fn get_status_list_credential(&self) -> DidResult<serde_json::Value> {
        self.status_list.to_credential()
    }

    /// Get the number of registered credentials
    pub fn registered_count(&self) -> usize {
        self.next_index
    }

    /// Get the number of revoked credentials
    pub fn revoked_count(&self) -> usize {
        self.status_list.revoked_count()
    }

    fn get_credential_index(&self, credential_id: &str) -> DidResult<usize> {
        self.credential_indices
            .get(credential_id)
            .copied()
            .ok_or_else(|| {
                DidError::KeyNotFound(format!(
                    "Credential not registered in registry: {}",
                    credential_id
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_list_basic() {
        let mut list = StatusList2021::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();

        assert!(!list.is_revoked(0).unwrap());
        assert!(!list.is_revoked(100).unwrap());

        list.set_status(42, true).unwrap();
        assert!(list.is_revoked(42).unwrap());
        assert!(!list.is_revoked(43).unwrap());
    }

    #[test]
    fn test_status_list_revoke_reinstate() {
        let mut list = StatusList2021::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();

        list.set_status(10, true).unwrap();
        assert!(list.is_revoked(10).unwrap());

        // Reinstate
        list.set_status(10, false).unwrap();
        assert!(!list.is_revoked(10).unwrap());
    }

    #[test]
    fn test_status_list_out_of_bounds() {
        let mut list =
            StatusList2021::new("https://example.com/status/1", "did:key:z6Mk", 100).unwrap();

        assert!(list.set_status(100, true).is_err());
        assert!(list.is_revoked(100).is_err());
    }

    #[test]
    fn test_encode_decode_bitstring_roundtrip() {
        let mut list =
            StatusList2021::new("https://example.com/status/1", "did:key:z6Mk", 1024).unwrap();

        list.set_status(0, true).unwrap();
        list.set_status(7, true).unwrap();
        list.set_status(100, true).unwrap();
        list.set_status(511, true).unwrap();

        let encoded = list.encode_bitstring().unwrap();
        let decoded = StatusList2021::decode_bitstring(&encoded, 1024).unwrap();

        assert!(decoded[0]);
        assert!(decoded[7]);
        assert!(decoded[100]);
        assert!(decoded[511]);
        assert!(!decoded[1]);
        assert!(!decoded[50]);
    }

    #[test]
    fn test_to_credential() {
        let mut list = StatusList2021::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();

        list.set_status(42, true).unwrap();

        let credential = list.to_credential().unwrap();

        assert_eq!(credential["id"], "https://example.com/status/1");
        let types = credential["type"].as_array().unwrap();
        assert!(types.contains(&serde_json::json!("StatusList2021Credential")));
        assert!(credential["credentialSubject"]["encodedList"].is_string());
    }

    #[test]
    fn test_from_credential_roundtrip() {
        let mut list =
            StatusList2021::new("https://example.com/status/1", "did:key:z6Mk", 1000).unwrap();

        list.set_status(5, true).unwrap();
        list.set_status(999, true).unwrap();

        let credential = list.to_credential().unwrap();
        let recovered = StatusList2021::from_credential(&credential, 1000).unwrap();

        assert!(recovered.is_revoked(5).unwrap());
        assert!(recovered.is_revoked(999).unwrap());
        assert!(!recovered.is_revoked(6).unwrap());
        assert_eq!(recovered.id, list.id);
    }

    #[test]
    fn test_credential_status_entry() {
        let status = CredentialStatus::new_status_list_2021(
            "https://example.com/status/1",
            42,
            StatusPurpose::Revocation,
        );

        assert_eq!(status.status_type, "StatusList2021Entry");
        assert_eq!(status.status_list_index, "42");
        assert_eq!(status.index().unwrap(), 42);
        assert_eq!(status.status_purpose, "revocation");
    }

    #[test]
    fn test_revocation_registry_basic() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();

        let status = registry
            .register_credential("urn:uuid:credential-1")
            .unwrap();
        assert_eq!(status.index().unwrap(), 0);

        let status2 = registry
            .register_credential("urn:uuid:credential-2")
            .unwrap();
        assert_eq!(status2.index().unwrap(), 1);

        assert_eq!(registry.registered_count(), 2);
        assert_eq!(registry.revoked_count(), 0);
    }

    #[test]
    fn test_revocation_registry_revoke() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();

        registry.register_credential("urn:uuid:cred-1").unwrap();
        registry.register_credential("urn:uuid:cred-2").unwrap();

        assert!(!registry.is_revoked("urn:uuid:cred-1").unwrap());

        registry.revoke("urn:uuid:cred-1").unwrap();

        assert!(registry.is_revoked("urn:uuid:cred-1").unwrap());
        assert!(!registry.is_revoked("urn:uuid:cred-2").unwrap());
        assert_eq!(registry.revoked_count(), 1);
    }

    #[test]
    fn test_revocation_registry_reinstate() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();

        registry.register_credential("urn:uuid:cred-1").unwrap();
        registry.revoke("urn:uuid:cred-1").unwrap();
        assert!(registry.is_revoked("urn:uuid:cred-1").unwrap());

        registry.reinstate("urn:uuid:cred-1").unwrap();
        assert!(!registry.is_revoked("urn:uuid:cred-1").unwrap());
    }

    #[test]
    fn test_revocation_registry_unregistered_credential() {
        let mut registry = RevocationRegistry::new(
            "https://example.com/status/1",
            "did:key:z6Mk",
            MIN_LIST_SIZE,
        )
        .unwrap();

        let result = registry.revoke("urn:uuid:unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_set() {
        let mut list =
            StatusList2021::new("https://example.com/status/1", "did:key:z6Mk", 1000).unwrap();

        list.set_batch(&[(1, true), (5, true), (10, true), (20, false)])
            .unwrap();

        assert!(list.is_revoked(1).unwrap());
        assert!(list.is_revoked(5).unwrap());
        assert!(list.is_revoked(10).unwrap());
        assert!(!list.is_revoked(20).unwrap());
    }

    #[test]
    fn test_revoked_indices() {
        let mut list =
            StatusList2021::new("https://example.com/status/1", "did:key:z6Mk", 100).unwrap();

        list.set_status(3, true).unwrap();
        list.set_status(7, true).unwrap();
        list.set_status(42, true).unwrap();

        let indices = list.revoked_indices();
        assert_eq!(indices, vec![3, 7, 42]);
    }

    #[test]
    fn test_status_list_zero_size_error() {
        let result = StatusList2021::new("https://example.com/status/1", "did:key:z6Mk", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_suspension_purpose() {
        let list = StatusList2021::new_with_purpose(
            "https://example.com/status/1",
            "did:key:z6Mk",
            1000,
            StatusPurpose::Suspension,
        )
        .unwrap();

        let credential = list.to_credential().unwrap();
        assert_eq!(
            credential["credentialSubject"]["statusPurpose"],
            "suspension"
        );
    }
}
