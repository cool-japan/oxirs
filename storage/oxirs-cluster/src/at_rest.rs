//! Authenticated encryption for cluster data at rest.
//!
//! Backups, write-ahead-log records, snapshots and
//! [`NetworkManager::encrypt_data`](crate::network::NetworkManager::encrypt_data)
//! payloads are sealed into a self-describing, versioned envelope:
//!
//! ```text
//! magic "OXCLAR01" (8) | key_id_len (1) | key_id (1..=255) | nonce (24) | ciphertext ‖ tag (16)
//! ```
//!
//! The cipher is XChaCha20-Poly1305 with a fresh random 192-bit nonce per
//! envelope, which stays safe for any realistic number of messages under one
//! long-lived key. The header and a caller-chosen context string are bound as
//! associated data, so an envelope cannot be relabelled with another key id or
//! replayed into a different kind of file (a WAL record as a snapshot, say).
//!
//! # Example
//!
//! ```rust
//! use oxirs_cluster::at_rest::{self, AtRestKey};
//!
//! let key = AtRestKey::new("backup-2026", [7u8; 32]).unwrap();
//! let sealed = at_rest::seal(&key, b"example", b"secret triples").unwrap();
//! assert!(at_rest::is_sealed(&sealed));
//! assert_eq!(at_rest::open(&key, b"example", &sealed).unwrap(), b"secret triples");
//! ```

use crate::error::{ClusterError, Result};
use oxicrypto_core::Aead;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Leading bytes of every sealed envelope ("OXirs CLuster At Rest", v01).
pub const ENVELOPE_MAGIC: [u8; 8] = *b"OXCLAR01";

const NONCE_LEN: usize = 24;
const TAG_LEN: usize = 16;

/// A 256-bit key for sealing data at rest, together with the identifier that
/// is recorded in every envelope it seals. `Debug` never prints the key.
#[derive(Clone)]
pub struct AtRestKey {
    id: String,
    key: [u8; 32],
}

impl AtRestKey {
    /// Wrap `key` under an explicit identifier.
    ///
    /// # Errors
    ///
    /// Fails when `id` is empty or longer than 255 bytes.
    pub fn new(id: impl Into<String>, key: [u8; 32]) -> Result<Self> {
        let id = id.into();
        if id.is_empty() || id.len() > usize::from(u8::MAX) {
            return Err(ClusterError::Encryption(format!(
                "key id must be 1 to 255 bytes long, got {} bytes",
                id.len()
            )));
        }
        Ok(Self { id, key })
    }

    /// Wrap `key` under an identifier derived from the key itself
    /// (`fp-` followed by 16 hex digits of a domain-separated SHA-256), so
    /// different keys get different ids without revealing anything usable.
    pub fn from_key(key: [u8; 32]) -> Self {
        let digest = Sha256::new()
            .chain_update(b"oxirs-cluster/at-rest/key-id/v1")
            .chain_update(key)
            .finalize();
        Self {
            id: format!("fp-{}", hex::encode(&digest[..8])),
            key,
        }
    }

    /// Load a key file holding either 32 raw bytes or 64 hexadecimal digits
    /// (surrounding whitespace allowed). The id is derived as in
    /// [`AtRestKey::from_key`].
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be read or holds anything else.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            ClusterError::Encryption(format!("failed to read key file {}: {e}", path.display()))
        })?;
        let key = parse_key_bytes(&bytes).ok_or_else(|| {
            ClusterError::Encryption(format!(
                "key file {} must hold 32 raw bytes or 64 hexadecimal digits",
                path.display()
            ))
        })?;
        Ok(Self::from_key(key))
    }

    /// The identifier recorded in envelopes sealed with this key.
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl fmt::Debug for AtRestKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtRestKey")
            .field("id", &self.id)
            .field("key", &"<redacted>")
            .finish()
    }
}

/// Parse key material: exactly 32 raw bytes, or 64 hex digits.
fn parse_key_bytes(bytes: &[u8]) -> Option<[u8; 32]> {
    if let Ok(raw) = <[u8; 32]>::try_from(bytes) {
        return Some(raw);
    }
    let text = std::str::from_utf8(bytes).ok()?.trim();
    let decoded = hex::decode(text).ok()?;
    <[u8; 32]>::try_from(decoded.as_slice()).ok()
}

/// Keys addressable by identifier, for opening data sealed under any of them.
#[derive(Debug, Clone, Default)]
pub struct AtRestKeyring {
    keys: HashMap<String, AtRestKey>,
}

impl AtRestKeyring {
    /// Create an empty keyring.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `key`, replacing (and returning) any key with the same id.
    pub fn insert(&mut self, key: AtRestKey) -> Option<AtRestKey> {
        self.keys.insert(key.id.clone(), key)
    }

    /// Look a key up by id.
    pub fn get(&self, id: &str) -> Option<&AtRestKey> {
        self.keys.get(id)
    }

    /// Number of keys held.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the keyring holds no keys.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Whether `data` starts with the envelope magic, i.e. claims to be sealed.
pub fn is_sealed(data: &[u8]) -> bool {
    data.starts_with(&ENVELOPE_MAGIC)
}

/// Split an envelope into its header (magic, id length, id) and the rest,
/// returning the key id as well.
fn parse_header(envelope: &[u8]) -> Result<(&[u8], &str, &[u8])> {
    let malformed =
        |reason: &str| ClusterError::Encryption(format!("malformed envelope: {reason}"));
    if !is_sealed(envelope) {
        return Err(malformed("missing magic, data is not sealed"));
    }
    let id_len = usize::from(
        *envelope
            .get(ENVELOPE_MAGIC.len())
            .ok_or_else(|| malformed("truncated header"))?,
    );
    let header_len = ENVELOPE_MAGIC.len() + 1 + id_len;
    if id_len == 0 || envelope.len() < header_len + NONCE_LEN + TAG_LEN {
        return Err(malformed("truncated header or body"));
    }
    let (header, body) = envelope.split_at(header_len);
    let key_id = std::str::from_utf8(&header[ENVELOPE_MAGIC.len() + 1..])
        .map_err(|_| malformed("key id is not UTF-8"))?;
    Ok((header, key_id, body))
}

/// The key id recorded in an envelope header.
///
/// # Errors
///
/// Fails when `envelope` is not a well-formed envelope.
pub fn envelope_key_id(envelope: &[u8]) -> Result<&str> {
    parse_header(envelope).map(|(_, key_id, _)| key_id)
}

/// Seal `plaintext` under `key`, binding `context` as associated data.
///
/// # Errors
///
/// Fails when the OS entropy source is unavailable or encryption fails.
pub fn seal(key: &AtRestKey, context: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    let id_len = u8::try_from(key.id.len())
        .map_err(|_| ClusterError::Encryption("key id longer than 255 bytes".to_string()))?;
    let nonce = oxicrypto_rand::random_nonce::<NONCE_LEN>()
        .map_err(|e| ClusterError::Encryption(format!("OS entropy source failed: {e}")))?;

    let mut envelope = Vec::with_capacity(
        ENVELOPE_MAGIC.len() + 1 + key.id.len() + NONCE_LEN + plaintext.len() + TAG_LEN,
    );
    envelope.extend_from_slice(&ENVELOPE_MAGIC);
    envelope.push(id_len);
    envelope.extend_from_slice(key.id.as_bytes());
    let aad = [envelope.as_slice(), context].concat();

    let ciphertext = oxicrypto_aead::XChaCha20Poly1305
        .seal_to_vec(&key.key, &nonce, &aad, plaintext)
        .map_err(|e| ClusterError::Encryption(format!("encryption failed: {e}")))?;
    envelope.extend_from_slice(&nonce);
    envelope.extend_from_slice(&ciphertext);
    Ok(envelope)
}

/// Open an envelope sealed under `key` with the same `context`.
///
/// # Errors
///
/// Fails when `envelope` is not a well-formed envelope, was sealed under a
/// different key id, or does not authenticate (tampered data, a different
/// context, or different key bytes behind the same id).
pub fn open(key: &AtRestKey, context: &[u8], envelope: &[u8]) -> Result<Vec<u8>> {
    let (header, key_id, body) = parse_header(envelope)?;
    if key_id != key.id {
        return Err(ClusterError::Encryption(format!(
            "data was sealed with key '{key_id}', but key '{}' was supplied",
            key.id
        )));
    }
    let (nonce, ciphertext) = body.split_at(NONCE_LEN);
    let aad = [header, context].concat();
    oxicrypto_aead::XChaCha20Poly1305
        .open_to_vec(&key.key, nonce, &aad, ciphertext)
        .map_err(|_| {
            ClusterError::Encryption(format!(
                "authentication failed for data sealed with key '{key_id}': \
                 it was modified, belongs elsewhere, or the key is wrong"
            ))
        })
}

/// Open an envelope with whichever key in `keyring` matches its key id.
///
/// # Errors
///
/// Fails when no key in `keyring` has the envelope's id, or as for [`open`].
pub fn open_with_keyring(
    keyring: &AtRestKeyring,
    context: &[u8],
    envelope: &[u8],
) -> Result<Vec<u8>> {
    let key_id = envelope_key_id(envelope)?;
    let key = keyring.get(key_id).ok_or_else(|| {
        ClusterError::Encryption(format!("no key with id '{key_id}' in the keyring"))
    })?;
    open(key, context, envelope)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTEXT: &[u8] = b"oxirs-cluster/test";

    fn key(id: &str, byte: u8) -> AtRestKey {
        AtRestKey::new(id, [byte; 32]).expect("valid key id")
    }

    #[test]
    fn test_seal_open_round_trip_hides_plaintext() {
        let key = key("k1", 1);
        let plaintext = b"PLAINTEXT-MARKER subject predicate object";
        let sealed = seal(&key, CONTEXT, plaintext).unwrap();

        assert!(is_sealed(&sealed));
        assert_eq!(envelope_key_id(&sealed).unwrap(), "k1");
        assert!(!sealed
            .windows(b"PLAINTEXT-MARKER".len())
            .any(|w| w == b"PLAINTEXT-MARKER"));
        assert_eq!(open(&key, CONTEXT, &sealed).unwrap(), plaintext);
    }

    #[test]
    fn test_seal_uses_fresh_nonces() {
        let key = key("k1", 1);
        let a = seal(&key, CONTEXT, b"same").unwrap();
        let b = seal(&key, CONTEXT, b"same").unwrap();
        assert_ne!(a, b, "two envelopes of the same plaintext are identical");
    }

    #[test]
    fn test_empty_plaintext_round_trips() {
        let key = key("k1", 1);
        let sealed = seal(&key, CONTEXT, b"").unwrap();
        assert_eq!(open(&key, CONTEXT, &sealed).unwrap(), b"");
    }

    #[test]
    fn test_tampered_body_is_rejected() {
        let key = key("k1", 1);
        let mut sealed = seal(&key, CONTEXT, b"payload").unwrap();
        let last = sealed.len() - 1;
        sealed[last] ^= 0x01;
        assert!(open(&key, CONTEXT, &sealed).is_err());

        let mut sealed = seal(&key, CONTEXT, b"payload").unwrap();
        let first_body_byte = ENVELOPE_MAGIC.len() + 1 + 2;
        sealed[first_body_byte] ^= 0x01; // nonce
        assert!(open(&key, CONTEXT, &sealed).is_err());
    }

    #[test]
    fn test_relabelled_key_id_is_rejected() {
        // Same key bytes under a different id: the header is authenticated,
        // so rewriting the id breaks the tag.
        let original = key("k1", 1);
        let relabelled = key("k2", 1);
        let mut sealed = seal(&original, CONTEXT, b"payload").unwrap();
        sealed[ENVELOPE_MAGIC.len() + 2] = b'2';
        assert_eq!(envelope_key_id(&sealed).unwrap(), "k2");
        assert!(open(&relabelled, CONTEXT, &sealed).is_err());
    }

    #[test]
    fn test_wrong_key_and_wrong_context_are_rejected() {
        let key_1 = key("k1", 1);
        let sealed = seal(&key_1, CONTEXT, b"payload").unwrap();

        let other_id = key("k2", 2);
        let err = open(&other_id, CONTEXT, &sealed).unwrap_err().to_string();
        assert!(err.contains("'k1'"), "unexpected error: {err}");

        let same_id_other_bytes = key("k1", 2);
        assert!(open(&same_id_other_bytes, CONTEXT, &sealed).is_err());

        assert!(open(&key_1, b"oxirs-cluster/other", &sealed).is_err());
    }

    #[test]
    fn test_unsealed_and_truncated_input_is_rejected() {
        let key = key("k1", 1);
        assert!(open(&key, CONTEXT, b"plain bytes").is_err());
        assert!(open(&key, CONTEXT, &ENVELOPE_MAGIC).is_err());

        let sealed = seal(&key, CONTEXT, b"payload").unwrap();
        assert!(open(&key, CONTEXT, &sealed[..sealed.len() - TAG_LEN]).is_err());
    }

    #[test]
    fn test_keyring_selects_key_by_id() {
        let mut keyring = AtRestKeyring::new();
        keyring.insert(key("old", 1));
        keyring.insert(key("new", 2));
        assert_eq!(keyring.len(), 2);

        let sealed_old = seal(&key("old", 1), CONTEXT, b"old data").unwrap();
        let sealed_new = seal(&key("new", 2), CONTEXT, b"new data").unwrap();
        assert_eq!(
            open_with_keyring(&keyring, CONTEXT, &sealed_old).unwrap(),
            b"old data"
        );
        assert_eq!(
            open_with_keyring(&keyring, CONTEXT, &sealed_new).unwrap(),
            b"new data"
        );

        let sealed_unknown = seal(&key("gone", 3), CONTEXT, b"data").unwrap();
        assert!(open_with_keyring(&keyring, CONTEXT, &sealed_unknown).is_err());
    }

    #[test]
    fn test_key_id_validation() {
        assert!(AtRestKey::new("", [0; 32]).is_err());
        assert!(AtRestKey::new("x".repeat(256), [0; 32]).is_err());
        assert!(AtRestKey::new("x".repeat(255), [0; 32]).is_ok());
    }

    #[test]
    fn test_derived_ids_differ_per_key() {
        let a = AtRestKey::from_key([1; 32]);
        let b = AtRestKey::from_key([2; 32]);
        assert!(a.id().starts_with("fp-"));
        assert_ne!(a.id(), b.id());
        assert_eq!(a.id(), AtRestKey::from_key([1; 32]).id());
    }

    #[test]
    fn test_key_file_formats() {
        let dir = tempfile::TempDir::new().unwrap();
        let raw = dir.path().join("raw.key");
        let hex_file = dir.path().join("hex.key");
        let bad = dir.path().join("bad.key");
        std::fs::write(&raw, [5u8; 32]).unwrap();
        std::fs::write(&hex_file, format!("{}\n", hex::encode([5u8; 32]))).unwrap();
        std::fs::write(&bad, "not a key").unwrap();

        let from_raw = AtRestKey::from_file(&raw).unwrap();
        let from_hex = AtRestKey::from_file(&hex_file).unwrap();
        assert_eq!(from_raw.id(), from_hex.id());
        let sealed = seal(&from_raw, CONTEXT, b"payload").unwrap();
        assert_eq!(open(&from_hex, CONTEXT, &sealed).unwrap(), b"payload");

        assert!(AtRestKey::from_file(&bad).is_err());
        assert!(AtRestKey::from_file(&dir.path().join("missing.key")).is_err());
    }

    #[test]
    fn test_debug_redacts_key_material() {
        let key = AtRestKey::new("k1", [0xAB; 32]).unwrap();
        let printed = format!("{key:?}");
        assert!(printed.contains("k1"));
        assert!(
            !printed.contains("171"),
            "decimal key bytes leaked: {printed}"
        );
        assert!(
            !printed.to_lowercase().contains("abab"),
            "hex key bytes leaked: {printed}"
        );
    }
}
