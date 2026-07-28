//! Key derivation for DID methods.
//!
//! Implements the standard IETF key-derivation functions over HMAC-SHA-2:
//!
//! * **HKDF** (RFC 5869) — Extract-then-Expand, using HMAC-SHA-256
//!   ([`KdfAlgorithm::Hkdf256`]) or HMAC-SHA-512 ([`KdfAlgorithm::Hkdf512`]).
//! * **PBKDF2** (RFC 8018) — HMAC-SHA-256 PRF ([`KdfAlgorithm::Pbkdf2Sha256`]).
//!
//! HMAC is computed over the pure-Rust `sha2` hashes already used throughout
//! this crate (no C/Fortran dependency, no external crypto FFI). Outputs match
//! the published RFC test vectors (see the module tests).

use scirs2_core::random::Random;
use sha2::{Digest, Sha256, Sha512};
use std::fmt;

// bring the Rng trait into scope for gen_range
use scirs2_core::random::Rng;

// ── HMAC / hash sizes ──────────────────────────────────────────────────────────

/// SHA-256 output length in bytes.
const SHA256_OUTPUT: usize = 32;
/// SHA-256 internal block size in bytes.
const SHA256_BLOCK: usize = 64;
/// SHA-512 output length in bytes.
const SHA512_OUTPUT: usize = 64;
/// SHA-512 internal block size in bytes.
const SHA512_BLOCK: usize = 128;

/// Generic HMAC (RFC 2104) over a `sha2` digest `D`.
///
/// `block_size` is the hash's internal block size (64 for SHA-256, 128 for
/// SHA-512). This is a total function — it never panics and never allocates a
/// fallible resource, so it keeps the KDF entry points infallible.
fn hmac<D: Digest>(key: &[u8], msg: &[u8], block_size: usize) -> Vec<u8> {
    // Derive the block-sized key K0.
    let mut k0 = vec![0u8; block_size];
    if key.len() > block_size {
        let digest = D::digest(key);
        let n = digest.len().min(block_size);
        k0[..n].copy_from_slice(&digest[..n]);
    } else {
        k0[..key.len()].copy_from_slice(key);
    }

    let ipad: Vec<u8> = k0.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k0.iter().map(|b| b ^ 0x5c).collect();

    let mut inner = D::new();
    inner.update(&ipad);
    inner.update(msg);
    let inner_hash = inner.finalize();

    let mut outer = D::new();
    outer.update(&opad);
    outer.update(inner_hash);
    outer.finalize().to_vec()
}

/// HMAC-SHA-256.
fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    hmac::<Sha256>(key, msg, SHA256_BLOCK)
}

/// HMAC-SHA-512.
fn hmac_sha512(key: &[u8], msg: &[u8]) -> Vec<u8> {
    hmac::<Sha512>(key, msg, SHA512_BLOCK)
}

/// Generic HKDF-Expand (RFC 5869 §2.3).
fn hkdf_expand_generic(
    prk: &[u8],
    info: &[u8],
    length: usize,
    hash_len: usize,
    mac: impl Fn(&[u8], &[u8]) -> Vec<u8>,
) -> Vec<u8> {
    if length == 0 {
        return Vec::new();
    }
    // RFC 5869 limit: L <= 255 * HashLen. Beyond that expansion is undefined;
    // we stop at the last valid block rather than fabricate more output.
    let max_blocks = 255usize;
    let mut okm = Vec::with_capacity(length);
    let mut t: Vec<u8> = Vec::new();
    let mut counter: usize = 1;
    while okm.len() < length && counter <= max_blocks {
        let mut data = Vec::with_capacity(t.len() + info.len() + 1);
        data.extend_from_slice(&t);
        data.extend_from_slice(info);
        data.push(counter as u8);
        t = mac(prk, &data);
        let take = (length - okm.len()).min(hash_len);
        okm.extend_from_slice(&t[..take.min(t.len())]);
        counter += 1;
    }
    okm.truncate(length);
    okm
}

/// Generic PBKDF2 (RFC 8018 §5.2) with an HMAC PRF.
fn pbkdf2_generic(
    password: &[u8],
    salt: &[u8],
    iterations: u32,
    length: usize,
    hash_len: usize,
    mac: impl Fn(&[u8], &[u8]) -> Vec<u8>,
) -> Vec<u8> {
    if length == 0 {
        return Vec::new();
    }
    let blocks = length.div_ceil(hash_len);
    let mut out = Vec::with_capacity(blocks * hash_len);

    for block_index in 1u32..=(blocks as u32) {
        // U1 = PRF(password, salt || INT_32_BE(block_index))
        let mut seed = salt.to_vec();
        seed.extend_from_slice(&block_index.to_be_bytes());
        let mut u = mac(password, &seed);
        let mut t = u.clone();

        // Ui = PRF(password, U(i-1)); T = U1 xor U2 xor … xor Uc
        for _ in 1..iterations {
            u = mac(password, &u);
            for (a, b) in t.iter_mut().zip(u.iter()) {
                *a ^= b;
            }
        }
        out.extend_from_slice(&t);
    }
    out.truncate(length);
    out
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can occur during key derivation.
#[derive(Debug, PartialEq, Eq)]
pub enum KdfError {
    /// Requested output length was zero.
    InvalidOutputLength,
    /// Iteration count was zero (PBKDF2).
    InvalidIterations,
    /// Input key material was empty.
    EmptyInput,
}

impl fmt::Display for KdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KdfError::InvalidOutputLength => write!(f, "output length must be > 0"),
            KdfError::InvalidIterations => write!(f, "iteration count must be > 0"),
            KdfError::EmptyInput => write!(f, "input key material must not be empty"),
        }
    }
}

impl std::error::Error for KdfError {}

// ── Public types ──────────────────────────────────────────────────────────────

/// Parameters shared across KDF calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HkdfParams {
    /// Optional salt (used in extract phase)
    pub salt: Vec<u8>,
    /// Context / application-specific info (used in expand phase)
    pub info: Vec<u8>,
    /// Desired output length in bytes
    pub output_length: usize,
}

impl HkdfParams {
    /// Create parameters with the given salt, info, and output length.
    pub fn new(salt: Vec<u8>, info: Vec<u8>, output_length: usize) -> Self {
        Self {
            salt,
            info,
            output_length,
        }
    }
}

/// A derived key with metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedKey {
    /// Raw key bytes
    pub key_bytes: Vec<u8>,
    /// Hex-encoded first 8 bytes, used as identifier
    pub key_id: String,
    /// Human-readable algorithm name
    pub algorithm: String,
}

/// Supported key derivation algorithms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KdfAlgorithm {
    /// HKDF (RFC 5869) with the HMAC-SHA-256 PRF.
    Hkdf256,
    /// HKDF (RFC 5869) with the HMAC-SHA-512 PRF.
    Hkdf512,
    /// PBKDF2 (RFC 8018) with the HMAC-SHA-256 PRF.
    Pbkdf2Sha256 {
        /// Number of iterations (must be > 0)
        iterations: u32,
    },
}

// ── KeyDerivation ─────────────────────────────────────────────────────────────

/// Stateless key derivation engine.
pub struct KeyDerivation;

impl KeyDerivation {
    /// Derive a key from input key material using the specified algorithm.
    ///
    /// # Errors
    /// - `KdfError::EmptyInput` if `ikm` is empty.
    /// - `KdfError::InvalidOutputLength` if `params.output_length` is 0.
    /// - `KdfError::InvalidIterations` for PBKDF2 with `iterations == 0`.
    pub fn derive(
        ikm: &[u8],
        params: &HkdfParams,
        algorithm: KdfAlgorithm,
    ) -> Result<DerivedKey, KdfError> {
        if ikm.is_empty() {
            return Err(KdfError::EmptyInput);
        }
        if params.output_length == 0 {
            return Err(KdfError::InvalidOutputLength);
        }

        let (key_bytes, alg_name) = match algorithm {
            KdfAlgorithm::Hkdf256 => {
                let prk = Self::hkdf_extract(&params.salt, ikm);
                let key = Self::hkdf_expand(&prk, &params.info, params.output_length);
                (key, "HKDF-256".to_string())
            }
            KdfAlgorithm::Hkdf512 => {
                // RFC 5869 HKDF instantiated with HMAC-SHA-512.
                let prk = Self::hkdf_extract_sha512(&params.salt, ikm);
                let key = Self::hkdf_expand_sha512(&prk, &params.info, params.output_length);
                (key, "HKDF-512".to_string())
            }
            KdfAlgorithm::Pbkdf2Sha256 { iterations } => {
                if iterations == 0 {
                    return Err(KdfError::InvalidIterations);
                }
                let key = Self::pbkdf2_derive(ikm, &params.salt, iterations, params.output_length);
                (key, format!("PBKDF2-SHA256-{}", iterations))
            }
        };

        let key_id = Self::key_id_from_bytes(&key_bytes);
        Ok(DerivedKey {
            key_bytes,
            key_id,
            algorithm: alg_name,
        })
    }

    /// HKDF-Extract (RFC 5869 §2.2) with HMAC-SHA-256: `PRK = HMAC(salt, IKM)`.
    ///
    /// An empty `salt` is replaced by a string of `HashLen` (32) zero bytes, per
    /// the RFC. Returns the 32-byte pseudorandom key.
    pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> Vec<u8> {
        let effective_salt: Vec<u8> = if salt.is_empty() {
            vec![0u8; SHA256_OUTPUT]
        } else {
            salt.to_vec()
        };
        hmac_sha256(&effective_salt, ikm)
    }

    /// HKDF-Expand (RFC 5869 §2.3) with HMAC-SHA-256.
    pub fn hkdf_expand(prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
        hkdf_expand_generic(prk, info, length, SHA256_OUTPUT, hmac_sha256)
    }

    /// HKDF-Extract with HMAC-SHA-512 (64-byte PRK).
    fn hkdf_extract_sha512(salt: &[u8], ikm: &[u8]) -> Vec<u8> {
        let effective_salt: Vec<u8> = if salt.is_empty() {
            vec![0u8; SHA512_OUTPUT]
        } else {
            salt.to_vec()
        };
        hmac_sha512(&effective_salt, ikm)
    }

    /// HKDF-Expand with HMAC-SHA-512.
    fn hkdf_expand_sha512(prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
        hkdf_expand_generic(prk, info, length, SHA512_OUTPUT, hmac_sha512)
    }

    /// PBKDF2 (RFC 8018 §5.2) with the HMAC-SHA-256 PRF.
    ///
    /// `U1 = HMAC(password, salt || INT_32_BE(i))`,
    /// `Uj = HMAC(password, U(j-1))`,
    /// `T  = U1 XOR U2 XOR … XOR U(iterations)`.
    pub fn pbkdf2_derive(password: &[u8], salt: &[u8], iterations: u32, length: usize) -> Vec<u8> {
        pbkdf2_generic(
            password,
            salt,
            iterations,
            length,
            SHA256_OUTPUT,
            hmac_sha256,
        )
    }

    /// Generate `length` random bytes using scirs2_core::random.
    pub fn generate_salt(length: usize) -> Vec<u8> {
        if length == 0 {
            return vec![];
        }
        let mut rng = Random::default();
        (0..length).map(|_| rng.gen_range(0u8..=255)).collect()
    }

    /// Produce a hex-encoded key ID from the first 8 bytes of `key_bytes`.
    ///
    /// If fewer than 8 bytes are available all bytes are encoded.
    pub fn key_id_from_bytes(key_bytes: &[u8]) -> String {
        let slice = if key_bytes.len() > 8 {
            &key_bytes[..8]
        } else {
            key_bytes
        };
        slice.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn simple_params(len: usize) -> HkdfParams {
        HkdfParams::new(b"test-salt".to_vec(), b"test-info".to_vec(), len)
    }

    fn ikm() -> &'static [u8] {
        b"secret-input-key-material"
    }

    // ── hkdf_extract ──────────────────────────────────────────────────────────

    #[test]
    fn test_extract_is_deterministic() {
        let prk1 = KeyDerivation::hkdf_extract(b"salt", ikm());
        let prk2 = KeyDerivation::hkdf_extract(b"salt", ikm());
        assert_eq!(prk1, prk2);
    }

    #[test]
    fn test_extract_returns_32_bytes() {
        let prk = KeyDerivation::hkdf_extract(b"salt", ikm());
        assert_eq!(prk.len(), 32);
    }

    #[test]
    fn test_extract_empty_salt_uses_zero_vector() {
        // RFC 5869: an absent salt is set to HashLen (32) zero bytes.
        let prk_empty = KeyDerivation::hkdf_extract(&[], ikm());
        let prk_zeros = KeyDerivation::hkdf_extract(&[0u8; 32], ikm());
        assert_eq!(prk_empty, prk_zeros);
    }

    #[test]
    fn test_extract_different_salts_different_prk() {
        let prk1 = KeyDerivation::hkdf_extract(b"salt-a", ikm());
        let prk2 = KeyDerivation::hkdf_extract(b"salt-b", ikm());
        assert_ne!(prk1, prk2);
    }

    #[test]
    fn test_extract_different_ikm_different_prk() {
        let prk1 = KeyDerivation::hkdf_extract(b"salt", b"ikm-a");
        let prk2 = KeyDerivation::hkdf_extract(b"salt", b"ikm-b");
        assert_ne!(prk1, prk2);
    }

    // ── hkdf_expand ───────────────────────────────────────────────────────────

    #[test]
    fn test_expand_exact_length() {
        let prk = KeyDerivation::hkdf_extract(b"salt", ikm());
        let out = KeyDerivation::hkdf_expand(&prk, b"info", 32);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_expand_arbitrary_length() {
        let prk = KeyDerivation::hkdf_extract(b"salt", ikm());
        let out = KeyDerivation::hkdf_expand(&prk, b"info", 77);
        assert_eq!(out.len(), 77);
    }

    #[test]
    fn test_expand_zero_length_empty() {
        let prk = KeyDerivation::hkdf_extract(b"salt", ikm());
        let out = KeyDerivation::hkdf_expand(&prk, b"info", 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_expand_is_deterministic() {
        let prk = KeyDerivation::hkdf_extract(b"salt", ikm());
        let o1 = KeyDerivation::hkdf_expand(&prk, b"info", 20);
        let o2 = KeyDerivation::hkdf_expand(&prk, b"info", 20);
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_expand_different_info_different_output() {
        let prk = KeyDerivation::hkdf_extract(b"salt", ikm());
        let o1 = KeyDerivation::hkdf_expand(&prk, b"info-A", 16);
        let o2 = KeyDerivation::hkdf_expand(&prk, b"info-B", 16);
        assert_ne!(o1, o2);
    }

    // ── pbkdf2_derive ─────────────────────────────────────────────────────────

    #[test]
    fn test_pbkdf2_deterministic() {
        let k1 = KeyDerivation::pbkdf2_derive(b"password", b"salt", 1000, 32);
        let k2 = KeyDerivation::pbkdf2_derive(b"password", b"salt", 1000, 32);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_pbkdf2_exact_length() {
        let k = KeyDerivation::pbkdf2_derive(b"password", b"salt", 100, 24);
        assert_eq!(k.len(), 24);
    }

    #[test]
    fn test_pbkdf2_different_passwords_different_keys() {
        let k1 = KeyDerivation::pbkdf2_derive(b"password1", b"salt", 100, 16);
        let k2 = KeyDerivation::pbkdf2_derive(b"password2", b"salt", 100, 16);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_pbkdf2_different_salts_different_keys() {
        let k1 = KeyDerivation::pbkdf2_derive(b"password", b"salt1", 100, 16);
        let k2 = KeyDerivation::pbkdf2_derive(b"password", b"salt2", 100, 16);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_pbkdf2_zero_length_empty() {
        let k = KeyDerivation::pbkdf2_derive(b"pw", b"salt", 10, 0);
        assert!(k.is_empty());
    }

    // ── derive() ─────────────────────────────────────────────────────────────

    #[test]
    fn test_derive_hkdf256_success() {
        let p = simple_params(32);
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("derive failed");
        assert_eq!(dk.key_bytes.len(), 32);
        assert_eq!(dk.algorithm, "HKDF-256");
    }

    #[test]
    fn test_derive_hkdf512_longer_output() {
        let p = simple_params(64);
        let dk256 =
            KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("derive 256 failed");
        let dk512 =
            KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf512).expect("derive 512 failed");
        // Both should be 64 bytes but differ (different internal PRK)
        assert_eq!(dk256.key_bytes.len(), 64);
        assert_eq!(dk512.key_bytes.len(), 64);
        assert_ne!(dk256.key_bytes, dk512.key_bytes);
    }

    #[test]
    fn test_derive_pbkdf2_success() {
        let p = simple_params(16);
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Pbkdf2Sha256 { iterations: 1000 })
            .expect("derive failed");
        assert_eq!(dk.key_bytes.len(), 16);
        assert!(dk.algorithm.contains("PBKDF2"));
    }

    #[test]
    fn test_derive_empty_ikm_error() {
        let p = simple_params(16);
        let err = KeyDerivation::derive(&[], &p, KdfAlgorithm::Hkdf256)
            .expect_err("should fail for empty ikm");
        assert_eq!(err, KdfError::EmptyInput);
    }

    #[test]
    fn test_derive_zero_output_length_error() {
        let p = HkdfParams::new(vec![], vec![], 0);
        let err = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256)
            .expect_err("should fail for zero length");
        assert_eq!(err, KdfError::InvalidOutputLength);
    }

    #[test]
    fn test_derive_pbkdf2_zero_iterations_error() {
        let p = simple_params(16);
        let err = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Pbkdf2Sha256 { iterations: 0 })
            .expect_err("should fail for zero iterations");
        assert_eq!(err, KdfError::InvalidIterations);
    }

    #[test]
    fn test_derive_same_inputs_same_output() {
        let p = simple_params(32);
        let dk1 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("ok");
        let dk2 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("ok");
        assert_eq!(dk1.key_bytes, dk2.key_bytes);
        assert_eq!(dk1.key_id, dk2.key_id);
    }

    // ── generate_salt ─────────────────────────────────────────────────────────

    #[test]
    fn test_generate_salt_correct_length() {
        let salt = KeyDerivation::generate_salt(32);
        assert_eq!(salt.len(), 32);
    }

    #[test]
    fn test_generate_salt_zero_length_empty() {
        let salt = KeyDerivation::generate_salt(0);
        assert!(salt.is_empty());
    }

    #[test]
    fn test_generate_salt_non_deterministic() {
        // Two calls should (with very high probability) differ
        let s1 = KeyDerivation::generate_salt(16);
        let s2 = KeyDerivation::generate_salt(16);
        // This assertion may theoretically fail (1/256^16 chance), but is reliable in practice.
        // We still test it to catch systematic bugs.
        assert_ne!(s1, s2, "salts should differ (statistical test)");
    }

    // ── key_id_from_bytes ─────────────────────────────────────────────────────

    #[test]
    fn test_key_id_is_hex() {
        let id =
            KeyDerivation::key_id_from_bytes(&[0xde, 0xad, 0xbe, 0xef, 0x01, 0x23, 0x45, 0x67]);
        assert_eq!(id, "deadbeef01234567");
    }

    #[test]
    fn test_key_id_exactly_8_bytes() {
        let bytes = vec![0u8; 8];
        let id = KeyDerivation::key_id_from_bytes(&bytes);
        assert_eq!(id.len(), 16); // 8 bytes * 2 hex chars
    }

    #[test]
    fn test_key_id_fewer_than_8_bytes() {
        let bytes = vec![0xABu8; 4];
        let id = KeyDerivation::key_id_from_bytes(&bytes);
        assert_eq!(id.len(), 8); // 4 bytes * 2 hex chars
    }

    #[test]
    fn test_key_id_from_derive_is_hex() {
        let p = simple_params(32);
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("ok");
        // key_id must be 16 hex chars (8 bytes)
        assert_eq!(dk.key_id.len(), 16);
        assert!(
            dk.key_id.chars().all(|c| c.is_ascii_hexdigit()),
            "id={}",
            dk.key_id
        );
    }

    // ── cross-checks ─────────────────────────────────────────────────────────

    #[test]
    fn test_hkdf256_vs_hkdf512_differ_same_params() {
        let p = simple_params(32);
        let k256 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("ok");
        let k512 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf512).expect("ok");
        assert_ne!(k256.key_bytes, k512.key_bytes);
    }

    #[test]
    fn test_different_info_leads_to_different_derived_keys() {
        let p1 = HkdfParams::new(b"salt".to_vec(), b"context-A".to_vec(), 32);
        let p2 = HkdfParams::new(b"salt".to_vec(), b"context-B".to_vec(), 32);
        let k1 = KeyDerivation::derive(ikm(), &p1, KdfAlgorithm::Hkdf256).expect("ok");
        let k2 = KeyDerivation::derive(ikm(), &p2, KdfAlgorithm::Hkdf256).expect("ok");
        assert_ne!(k1.key_bytes, k2.key_bytes);
    }

    #[test]
    fn test_kdf_error_display() {
        assert!(!KdfError::InvalidOutputLength.to_string().is_empty());
        assert!(!KdfError::InvalidIterations.to_string().is_empty());
        assert!(!KdfError::EmptyInput.to_string().is_empty());
    }

    // ── Additional tests to reach ≥45 ─────────────────────────────────────────

    #[test]
    fn test_hkdf_params_new() {
        let p = HkdfParams::new(b"s".to_vec(), b"i".to_vec(), 16);
        assert_eq!(p.salt, b"s");
        assert_eq!(p.info, b"i");
        assert_eq!(p.output_length, 16);
    }

    #[test]
    fn test_derived_key_fields() {
        let p = simple_params(16);
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("ok");
        assert_eq!(dk.key_bytes.len(), 16);
        assert_eq!(dk.algorithm, "HKDF-256");
        assert!(!dk.key_id.is_empty());
    }

    #[test]
    fn test_pbkdf2_more_iterations_still_deterministic() {
        let k = KeyDerivation::pbkdf2_derive(b"pw", b"salt", 5000, 16);
        let k2 = KeyDerivation::pbkdf2_derive(b"pw", b"salt", 5000, 16);
        assert_eq!(k, k2);
    }

    #[test]
    fn test_extract_large_ikm() {
        let large_ikm = vec![0x42u8; 256];
        let prk = KeyDerivation::hkdf_extract(b"salt", &large_ikm);
        assert_eq!(prk.len(), 32);
    }

    #[test]
    fn test_expand_large_output() {
        let prk = KeyDerivation::hkdf_extract(b"salt", ikm());
        let out = KeyDerivation::hkdf_expand(&prk, b"info", 200);
        assert_eq!(out.len(), 200);
    }

    #[test]
    fn test_derive_hkdf512_algorithm_name() {
        let p = simple_params(16);
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf512).expect("ok");
        assert_eq!(dk.algorithm, "HKDF-512");
    }

    #[test]
    fn test_derive_pbkdf2_algorithm_name_contains_iterations() {
        let p = simple_params(16);
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Pbkdf2Sha256 { iterations: 2048 })
            .expect("ok");
        assert!(dk.algorithm.contains("2048"), "alg={}", dk.algorithm);
    }

    #[test]
    fn test_key_id_empty_input_empty_string() {
        let id = KeyDerivation::key_id_from_bytes(&[]);
        assert!(id.is_empty());
    }

    #[test]
    fn test_key_id_single_byte() {
        let id = KeyDerivation::key_id_from_bytes(&[0xFF]);
        assert_eq!(id, "ff");
    }

    #[test]
    fn test_generate_salt_small() {
        let s = KeyDerivation::generate_salt(1);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_extract_reproducible_with_binary_salt() {
        let salt = [0u8, 1, 2, 3, 255, 254, 253, 252];
        let prk1 = KeyDerivation::hkdf_extract(&salt, ikm());
        let prk2 = KeyDerivation::hkdf_extract(&salt, ikm());
        assert_eq!(prk1, prk2);
    }

    #[test]
    fn test_derive_single_byte_output() {
        let p = HkdfParams::new(b"s".to_vec(), b"i".to_vec(), 1);
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("ok");
        assert_eq!(dk.key_bytes.len(), 1);
    }

    #[test]
    fn test_kdf_error_equality() {
        assert_eq!(KdfError::EmptyInput, KdfError::EmptyInput);
        assert_ne!(KdfError::EmptyInput, KdfError::InvalidOutputLength);
    }

    // ── RFC test vectors (prove the crypto is real, not a placeholder PRF) ─────

    #[test]
    fn regression_hmac_sha256_rfc4231_tc1() {
        // RFC 4231, Test Case 1.
        let key = [0x0bu8; 20];
        let mac = hmac_sha256(&key, b"Hi There");
        assert_eq!(
            hex::encode(mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn regression_hkdf_sha256_rfc5869_tc1() {
        // RFC 5869, Test Case 1 (HKDF-SHA256).
        let ikm = [0x0bu8; 22];
        let salt = hex::decode("000102030405060708090a0b0c").expect("valid hex");
        let info = hex::decode("f0f1f2f3f4f5f6f7f8f9").expect("valid hex");

        let prk = KeyDerivation::hkdf_extract(&salt, &ikm);
        assert_eq!(
            hex::encode(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        );

        let okm = KeyDerivation::hkdf_expand(&prk, &info, 42);
        assert_eq!(
            hex::encode(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        );
    }

    #[test]
    fn regression_pbkdf2_sha256_rfc7914() {
        // RFC 7914 §11: PBKDF2-HMAC-SHA256("passwd", "salt", 1, 64).
        let dk = KeyDerivation::pbkdf2_derive(b"passwd", b"salt", 1, 64);
        assert_eq!(
            hex::encode(dk),
            "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc\
             49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783"
        );
    }

    #[test]
    fn regression_hkdf512_differs_and_is_64_byte_prk() {
        // HKDF-512 must use a genuinely different (SHA-512) PRF than HKDF-256.
        let prk = KeyDerivation::hkdf_extract_sha512(b"salt", ikm());
        assert_eq!(prk.len(), 64);
        let p = simple_params(32);
        let k256 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256).expect("ok");
        let k512 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf512).expect("ok");
        assert_ne!(k256.key_bytes, k512.key_bytes);
    }
}
