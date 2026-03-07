//! RevocationList2020 — W3C Credential Status revocation bitmap
//!
//! Implements the W3C Revocation List 2020 specification:
//! <https://w3c-ccg.github.io/vc-status-rl-2020/>
//!
//! Key features:
//! - `RevocationList2020`: fixed-size bitset (default 16 384 entries)
//! - `RevocationEntry`: index + reason code
//! - `RevocationRegistry2020`: in-memory O(1) check with bloom filter for
//!   fast non-membership proof
//! - `RevocationStatus` enum: Valid, Revoked(reason), Unknown

use crate::{DidError, DidResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ── RevocationStatus ─────────────────────────────────────────────────────────

/// Result of querying whether a credential has been revoked
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevocationStatus {
    /// Credential is currently valid (not revoked)
    Valid,
    /// Credential has been revoked — reason string gives human-readable context
    Revoked { reason: String },
    /// Credential ID is not tracked by this registry — cannot determine status
    Unknown,
}

impl RevocationStatus {
    /// Returns `true` if the credential is revoked
    pub fn is_revoked(&self) -> bool {
        matches!(self, RevocationStatus::Revoked { .. })
    }

    /// Returns `true` if the credential is valid
    pub fn is_valid(&self) -> bool {
        matches!(self, RevocationStatus::Valid)
    }

    /// Returns `true` if the credential is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, RevocationStatus::Unknown)
    }
}

// ── RevocationEntry ──────────────────────────────────────────────────────────

/// A single revocation record stored in the list
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RevocationEntry {
    /// Bit-list index assigned to this credential
    pub index: usize,
    /// Credential URI / identifier
    pub credential_id: String,
    /// Optional reason code (e.g. "keyCompromise", "superseded")
    pub reason: String,
    /// ISO-8601 timestamp at revocation time
    pub revoked_at: String,
}

impl RevocationEntry {
    /// Create with an explicit reason
    pub fn new(index: usize, credential_id: &str, reason: &str, revoked_at: &str) -> Self {
        Self {
            index,
            credential_id: credential_id.to_string(),
            reason: reason.to_string(),
            revoked_at: revoked_at.to_string(),
        }
    }
}

// ── Bloom filter (simple, deterministic) ────────────────────────────────────

/// Minimal bloom filter for fast non-membership proofs.
///
/// Uses k=3 independent SHA-256-derived hash functions.
/// The filter is sized to give < 1 % false-positive rate for up to 10 000 items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bitset stored as bytes
    bits: Vec<u8>,
    /// Number of hash functions
    k: usize,
    /// Number of bits in the filter (= bits.len() * 8)
    m: usize,
    /// Count of items inserted
    count: usize,
}

impl BloomFilter {
    /// Create a new bloom filter with `m` bits and `k` hash functions.
    /// Sensible defaults: `m = 131_072` (16 KiB), `k = 3`.
    pub fn new(m: usize, k: usize) -> Self {
        let byte_count = (m + 7) / 8;
        Self {
            bits: vec![0u8; byte_count],
            k,
            m,
            count: 0,
        }
    }

    /// Create with defaults optimised for up to 10 000 items with ≈1 % FP rate
    pub fn with_defaults() -> Self {
        Self::new(131_072, 3)
    }

    /// Insert an item
    pub fn insert(&mut self, item: &str) {
        for i in 0..self.k {
            let bit = self.hash(item, i);
            let byte_idx = bit / 8;
            let bit_idx = bit % 8;
            self.bits[byte_idx] |= 1 << bit_idx;
        }
        self.count += 1;
    }

    /// Query: returns `false` if the item is **definitely not** in the set.
    /// Returns `true` if the item is **possibly** in the set (may be a FP).
    pub fn might_contain(&self, item: &str) -> bool {
        for i in 0..self.k {
            let bit = self.hash(item, i);
            let byte_idx = bit / 8;
            let bit_idx = bit % 8;
            if self.bits[byte_idx] & (1 << bit_idx) == 0 {
                return false;
            }
        }
        true
    }

    /// Number of items inserted
    pub fn count(&self) -> usize {
        self.count
    }

    /// Estimated false-positive rate given current fill
    pub fn false_positive_rate(&self) -> f64 {
        // FP ≈ (1 - e^{-k·n/m})^k
        let exponent = -(self.k as f64) * (self.count as f64) / (self.m as f64);
        (1.0 - exponent.exp()).powi(self.k as i32)
    }

    // Deterministic hash function: SHA-256(seed || item) mod m
    fn hash(&self, item: &str, seed: usize) -> usize {
        let mut hasher = Sha256::new();
        hasher.update(seed.to_be_bytes());
        hasher.update(item.as_bytes());
        let digest = hasher.finalize();
        // Use first 8 bytes as u64
        let value = u64::from_be_bytes(digest[..8].try_into().unwrap_or([0u8; 8]));
        (value as usize) % self.m
    }
}

// ── RevocationList2020 ───────────────────────────────────────────────────────

/// Bitset-based revocation list (W3C Revocation List 2020).
///
/// Unlike `StatusList2021` this structure is not GZIP-compressed in memory;
/// it exposes raw bit operations.  It can be serialised to a VC-style JSON
/// credential via `to_credential()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationList2020 {
    /// Unique identifier for this list (URL)
    pub id: String,
    /// Issuer DID
    pub issuer: String,
    /// Raw bitset (1 bit per credential)
    bits: Vec<u8>,
    /// Total capacity in bit positions
    capacity: usize,
}

impl RevocationList2020 {
    /// Create a new list with `capacity` bit positions.
    ///
    /// Minimum capacity is `16_384` to prevent correlation attacks.
    pub fn new(id: &str, issuer: &str, capacity: usize) -> DidResult<Self> {
        const MIN_CAPACITY: usize = 16_384;
        if capacity < MIN_CAPACITY {
            return Err(DidError::InvalidKey(format!(
                "RevocationList2020 capacity must be at least {MIN_CAPACITY}, got {capacity}"
            )));
        }
        let byte_count = (capacity + 7) / 8;
        Ok(Self {
            id: id.to_string(),
            issuer: issuer.to_string(),
            bits: vec![0u8; byte_count],
            capacity,
        })
    }

    /// Check whether the bit at `index` is set (credential revoked)
    pub fn is_revoked(&self, index: usize) -> DidResult<bool> {
        self.check_bounds(index)?;
        let byte = self.bits[index / 8];
        Ok(byte & (1 << (index % 8)) != 0)
    }

    /// Set or clear the bit at `index`
    pub fn set_status(&mut self, index: usize, revoked: bool) -> DidResult<()> {
        self.check_bounds(index)?;
        if revoked {
            self.bits[index / 8] |= 1 << (index % 8);
        } else {
            self.bits[index / 8] &= !(1 << (index % 8));
        }
        Ok(())
    }

    /// Total bit capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Count of revoked credentials
    pub fn revoked_count(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Indices of all revoked credentials
    pub fn revoked_indices(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for (byte_idx, byte) in self.bits.iter().enumerate() {
            for bit_idx in 0..8 {
                let global = byte_idx * 8 + bit_idx;
                if global >= self.capacity {
                    break;
                }
                if byte & (1 << bit_idx) != 0 {
                    result.push(global);
                }
            }
        }
        result
    }

    /// Serialise to a VC-style JSON credential (not signed)
    pub fn to_credential(&self) -> DidResult<serde_json::Value> {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
        let encoded = URL_SAFE_NO_PAD.encode(&self.bits);
        Ok(serde_json::json!({
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://w3id.org/vc-revocation-list-2020/v1"
            ],
            "id": self.id,
            "type": ["VerifiableCredential", "RevocationList2020Credential"],
            "issuer": self.issuer,
            "credentialSubject": {
                "id": format!("{}#list", self.id),
                "type": "RevocationList2020",
                "encodedList": encoded
            }
        }))
    }

    fn check_bounds(&self, index: usize) -> DidResult<()> {
        if index >= self.capacity {
            Err(DidError::InvalidKey(format!(
                "index {index} out of range (capacity {})",
                self.capacity
            )))
        } else {
            Ok(())
        }
    }
}

// ── RevocationRegistry2020 ───────────────────────────────────────────────────

/// High-level registry combining `RevocationList2020` with a bloom filter
/// for O(1) non-membership proofs and a credential → index map.
#[derive(Debug, Clone)]
pub struct RevocationRegistry2020 {
    list: RevocationList2020,
    bloom: BloomFilter,
    /// credential_id → (bit index, RevocationEntry if revoked)
    credential_map: HashMap<String, usize>,
    /// revoked entries, keyed by bit index
    entries: HashMap<usize, RevocationEntry>,
    /// Next free index
    next_index: usize,
}

impl RevocationRegistry2020 {
    /// Create a new registry
    pub fn new(id: &str, issuer: &str, capacity: usize) -> DidResult<Self> {
        Ok(Self {
            list: RevocationList2020::new(id, issuer, capacity)?,
            bloom: BloomFilter::with_defaults(),
            credential_map: HashMap::new(),
            entries: HashMap::new(),
            next_index: 0,
        })
    }

    /// Register a credential ID and assign it a bit index.
    /// Returns the assigned index.
    pub fn register(&mut self, credential_id: &str) -> DidResult<usize> {
        if self.credential_map.contains_key(credential_id) {
            return Err(DidError::InvalidKey(format!(
                "Credential already registered: {credential_id}"
            )));
        }
        if self.next_index >= self.list.capacity() {
            return Err(DidError::InvalidKey("Revocation list is full".to_string()));
        }
        let index = self.next_index;
        self.next_index += 1;
        self.credential_map.insert(credential_id.to_string(), index);
        Ok(index)
    }

    /// Check status using bloom filter for fast non-membership, then exact check.
    pub fn check_status(&self, credential_id: &str) -> RevocationStatus {
        // Unknown if not registered
        let Some(&index) = self.credential_map.get(credential_id) else {
            return RevocationStatus::Unknown;
        };

        // Bloom filter quick negative check
        if !self.bloom.might_contain(credential_id) {
            return RevocationStatus::Valid;
        }

        // Exact check via bitset
        match self.list.is_revoked(index) {
            Ok(true) => {
                let reason = self
                    .entries
                    .get(&index)
                    .map_or("unspecified", |e| e.reason.as_str())
                    .to_string();
                RevocationStatus::Revoked { reason }
            }
            _ => RevocationStatus::Valid,
        }
    }

    /// Revoke a credential
    pub fn revoke(&mut self, credential_id: &str, reason: &str) -> DidResult<()> {
        let index = self.resolve_index(credential_id)?;
        self.list.set_status(index, true)?;
        self.bloom.insert(credential_id);
        let ts = chrono::Utc::now().to_rfc3339();
        self.entries.insert(
            index,
            RevocationEntry::new(index, credential_id, reason, &ts),
        );
        Ok(())
    }

    /// Reinstate a revoked credential
    pub fn reinstate(&mut self, credential_id: &str) -> DidResult<()> {
        let index = self.resolve_index(credential_id)?;
        self.list.set_status(index, false)?;
        self.entries.remove(&index);
        // Note: the bloom filter cannot un-set a bit; subsequent checks fall
        // through to the exact bitset and return Valid correctly.
        Ok(())
    }

    /// Count of revoked credentials
    pub fn revoked_count(&self) -> usize {
        self.list.revoked_count()
    }

    /// Number of registered credentials
    pub fn registered_count(&self) -> usize {
        self.credential_map.len()
    }

    /// Access the underlying revocation list (for serialisation/export)
    pub fn list(&self) -> &RevocationList2020 {
        &self.list
    }

    /// Reference to the bloom filter
    pub fn bloom(&self) -> &BloomFilter {
        &self.bloom
    }

    /// Get all revocation entries
    pub fn entries(&self) -> impl Iterator<Item = &RevocationEntry> {
        self.entries.values()
    }

    fn resolve_index(&self, credential_id: &str) -> DidResult<usize> {
        self.credential_map
            .get(credential_id)
            .copied()
            .ok_or_else(|| {
                DidError::InvalidKey(format!("Credential not registered: {credential_id}"))
            })
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "https://example.com/status/rl2020";
    const ISSUER: &str = "did:key:z6Mk";

    fn make_registry(cap: usize) -> RevocationRegistry2020 {
        RevocationRegistry2020::new(ID, ISSUER, cap).unwrap()
    }

    // ── RevocationStatus ──────────────────────────────────────────────────────

    #[test]
    fn test_status_valid_is_not_revoked() {
        assert!(!RevocationStatus::Valid.is_revoked());
        assert!(RevocationStatus::Valid.is_valid());
        assert!(!RevocationStatus::Valid.is_unknown());
    }

    #[test]
    fn test_status_revoked_is_revoked() {
        let s = RevocationStatus::Revoked {
            reason: "test".to_string(),
        };
        assert!(s.is_revoked());
        assert!(!s.is_valid());
    }

    #[test]
    fn test_status_unknown() {
        assert!(RevocationStatus::Unknown.is_unknown());
    }

    // ── BloomFilter ───────────────────────────────────────────────────────────

    #[test]
    fn test_bloom_insert_and_query() {
        let mut bf = BloomFilter::with_defaults();
        bf.insert("urn:uuid:cred-1");
        assert!(bf.might_contain("urn:uuid:cred-1"));
    }

    #[test]
    fn test_bloom_non_member_definite_negative() {
        let bf = BloomFilter::with_defaults();
        // Empty filter must return false for any item
        assert!(!bf.might_contain("urn:uuid:never-inserted"));
    }

    #[test]
    fn test_bloom_count() {
        let mut bf = BloomFilter::with_defaults();
        bf.insert("a");
        bf.insert("b");
        bf.insert("c");
        assert_eq!(bf.count(), 3);
    }

    #[test]
    fn test_bloom_false_positive_rate_low_for_empty() {
        let bf = BloomFilter::with_defaults();
        assert!(bf.false_positive_rate() < 0.01);
    }

    #[test]
    fn test_bloom_custom_params() {
        let mut bf = BloomFilter::new(1024, 2);
        bf.insert("item");
        assert!(bf.might_contain("item"));
        assert!(!bf.might_contain("other-item-definitely-not-here-xyz"));
    }

    // ── RevocationList2020 ────────────────────────────────────────────────────

    #[test]
    fn test_list_new_min_capacity_error() {
        assert!(RevocationList2020::new(ID, ISSUER, 1024).is_err());
    }

    #[test]
    fn test_list_set_and_check() {
        let mut list = RevocationList2020::new(ID, ISSUER, 16_384).unwrap();
        assert!(!list.is_revoked(42).unwrap());
        list.set_status(42, true).unwrap();
        assert!(list.is_revoked(42).unwrap());
    }

    #[test]
    fn test_list_set_false_clears_bit() {
        let mut list = RevocationList2020::new(ID, ISSUER, 16_384).unwrap();
        list.set_status(10, true).unwrap();
        list.set_status(10, false).unwrap();
        assert!(!list.is_revoked(10).unwrap());
    }

    #[test]
    fn test_list_out_of_bounds_error() {
        let list = RevocationList2020::new(ID, ISSUER, 16_384).unwrap();
        assert!(list.is_revoked(16_384).is_err());
        assert!(list.is_revoked(99_999).is_err());
    }

    #[test]
    fn test_list_revoked_count() {
        let mut list = RevocationList2020::new(ID, ISSUER, 16_384).unwrap();
        list.set_status(1, true).unwrap();
        list.set_status(5, true).unwrap();
        list.set_status(1000, true).unwrap();
        assert_eq!(list.revoked_count(), 3);
    }

    #[test]
    fn test_list_revoked_indices() {
        let mut list = RevocationList2020::new(ID, ISSUER, 16_384).unwrap();
        list.set_status(3, true).unwrap();
        list.set_status(7, true).unwrap();
        list.set_status(200, true).unwrap();
        assert_eq!(list.revoked_indices(), vec![3, 7, 200]);
    }

    #[test]
    fn test_list_to_credential_json() {
        let mut list = RevocationList2020::new(ID, ISSUER, 16_384).unwrap();
        list.set_status(0, true).unwrap();
        let cred = list.to_credential().unwrap();
        assert_eq!(cred["type"][1], "RevocationList2020Credential");
        assert_eq!(cred["issuer"], ISSUER);
        assert_eq!(cred["credentialSubject"]["type"], "RevocationList2020");
        assert!(cred["credentialSubject"]["encodedList"].is_string());
    }

    #[test]
    fn test_list_capacity() {
        let list = RevocationList2020::new(ID, ISSUER, 32_768).unwrap();
        assert_eq!(list.capacity(), 32_768);
    }

    // ── RevocationRegistry2020 ────────────────────────────────────────────────

    #[test]
    fn test_registry_register_and_valid() {
        let mut reg = make_registry(16_384);
        reg.register("urn:uuid:a").unwrap();
        assert_eq!(reg.check_status("urn:uuid:a"), RevocationStatus::Valid);
    }

    #[test]
    fn test_registry_unknown_credential() {
        let reg = make_registry(16_384);
        assert_eq!(
            reg.check_status("urn:uuid:never-registered"),
            RevocationStatus::Unknown
        );
    }

    #[test]
    fn test_registry_revoke_and_check() {
        let mut reg = make_registry(16_384);
        reg.register("urn:uuid:cred-1").unwrap();
        reg.revoke("urn:uuid:cred-1", "keyCompromise").unwrap();
        match reg.check_status("urn:uuid:cred-1") {
            RevocationStatus::Revoked { reason } => assert_eq!(reason, "keyCompromise"),
            other => panic!("Expected Revoked, got {other:?}"),
        }
    }

    #[test]
    fn test_registry_reinstate() {
        let mut reg = make_registry(16_384);
        reg.register("urn:uuid:cred-2").unwrap();
        reg.revoke("urn:uuid:cred-2", "superseded").unwrap();
        reg.reinstate("urn:uuid:cred-2").unwrap();
        assert_eq!(reg.check_status("urn:uuid:cred-2"), RevocationStatus::Valid);
    }

    #[test]
    fn test_registry_revoked_count() {
        let mut reg = make_registry(16_384);
        for i in 0..5 {
            reg.register(&format!("urn:uuid:{i}")).unwrap();
        }
        reg.revoke("urn:uuid:0", "a").unwrap();
        reg.revoke("urn:uuid:2", "b").unwrap();
        assert_eq!(reg.revoked_count(), 2);
    }

    #[test]
    fn test_registry_registered_count() {
        let mut reg = make_registry(16_384);
        reg.register("urn:uuid:x").unwrap();
        reg.register("urn:uuid:y").unwrap();
        assert_eq!(reg.registered_count(), 2);
    }

    #[test]
    fn test_registry_double_register_error() {
        let mut reg = make_registry(16_384);
        reg.register("urn:uuid:dup").unwrap();
        assert!(reg.register("urn:uuid:dup").is_err());
    }

    #[test]
    fn test_registry_revoke_unregistered_error() {
        let mut reg = make_registry(16_384);
        assert!(reg.revoke("urn:uuid:ghost", "reason").is_err());
    }

    #[test]
    fn test_registry_reinstate_unregistered_error() {
        let mut reg = make_registry(16_384);
        assert!(reg.reinstate("urn:uuid:ghost").is_err());
    }

    #[test]
    fn test_registry_entries_after_revoke() {
        let mut reg = make_registry(16_384);
        reg.register("urn:uuid:e1").unwrap();
        reg.revoke("urn:uuid:e1", "expired").unwrap();
        let entries: Vec<_> = reg.entries().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].credential_id, "urn:uuid:e1");
        assert_eq!(entries[0].reason, "expired");
    }

    #[test]
    fn test_registry_entries_cleared_after_reinstate() {
        let mut reg = make_registry(16_384);
        reg.register("urn:uuid:e2").unwrap();
        reg.revoke("urn:uuid:e2", "admin").unwrap();
        reg.reinstate("urn:uuid:e2").unwrap();
        let entries: Vec<_> = reg.entries().collect();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_registry_multiple_credentials() {
        let mut reg = make_registry(16_384);
        for i in 0..10 {
            reg.register(&format!("urn:uuid:multi-{i}")).unwrap();
        }
        reg.revoke("urn:uuid:multi-3", "r3").unwrap();
        reg.revoke("urn:uuid:multi-7", "r7").unwrap();

        assert!(reg.check_status("urn:uuid:multi-3").is_revoked());
        assert!(reg.check_status("urn:uuid:multi-7").is_revoked());
        assert!(reg.check_status("urn:uuid:multi-5").is_valid());
        assert_eq!(reg.revoked_count(), 2);
    }

    // ── RevocationEntry ───────────────────────────────────────────────────────

    #[test]
    fn test_revocation_entry_fields() {
        let entry =
            RevocationEntry::new(42, "urn:uuid:test", "keyCompromise", "2026-01-01T00:00:00Z");
        assert_eq!(entry.index, 42);
        assert_eq!(entry.credential_id, "urn:uuid:test");
        assert_eq!(entry.reason, "keyCompromise");
        assert_eq!(entry.revoked_at, "2026-01-01T00:00:00Z");
    }
}
