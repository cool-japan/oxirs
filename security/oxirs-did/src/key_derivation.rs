//! HKDF-like key derivation for DID methods.
//!
//! Implements structurally-correct KDF patterns (HKDF-Extract/Expand, PBKDF2)
//! using a simplified FNV-1a-based XOR-hash PRF — pure Rust, no external
//! crypto crates required.  The focus is on API correctness and pattern
//! demonstration; for production use replace the PRF with a real HMAC-SHA256.

use scirs2_core::random::Random;
use std::fmt;

// bring the Rng trait into scope for gen_range
use scirs2_core::random::Rng;

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
        Self { salt, info, output_length }
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
    /// HKDF with 256-bit internal state
    Hkdf256,
    /// HKDF with 512-bit internal state
    Hkdf512,
    /// Simplified PBKDF2 with SHA-256-equivalent iterations
    Pbkdf2Sha256 {
        /// Number of iterations (must be > 0)
        iterations: u32,
    },
}

// ── Internal PRF ──────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash used as a building block for the PRF.
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Produce 8 pseudo-random bytes from a key and message using double FNV-1a.
///
/// `prf(key, data)` — not a cryptographic HMAC, but structurally equivalent
/// for test/demonstration purposes.
fn prf_block(key: &[u8], data: &[u8]) -> [u8; 8] {
    // Inner: H(key || data)
    let mut combined = key.to_vec();
    combined.extend_from_slice(data);
    let inner = fnv1a_64(&combined);

    // Outer: H(key XOR 0x5c^8 || inner_bytes)
    let mut outer_key: Vec<u8> = key.iter().map(|b| b ^ 0x5c).collect();
    outer_key.extend_from_slice(&inner.to_le_bytes());
    let outer = fnv1a_64(&outer_key);

    outer.to_le_bytes()
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
                // For HKDF-512 we use a doubled PRK to simulate a larger internal state.
                let prk = Self::hkdf_extract(&params.salt, ikm);
                let prk2 = Self::hkdf_extract(ikm, &params.salt);
                let combined_prk: Vec<u8> = prk.iter().chain(prk2.iter()).copied().collect();
                let key = Self::hkdf_expand(&combined_prk, &params.info, params.output_length);
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
        Ok(DerivedKey { key_bytes, key_id, algorithm: alg_name })
    }

    /// HKDF extract phase: PRK = PRF(salt, IKM).
    ///
    /// Returns a fixed-length pseudorandom key suitable for expansion.
    pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> Vec<u8> {
        // Use PRF blocks iteratively to produce 32 bytes of PRK.
        let effective_salt: Vec<u8> = if salt.is_empty() {
            vec![0u8; 8]
        } else {
            salt.to_vec()
        };

        let mut prk = Vec::with_capacity(32);
        for counter in 0u8..4 {
            let mut data = ikm.to_vec();
            data.push(counter);
            let block = prf_block(&effective_salt, &data);
            prk.extend_from_slice(&block);
        }
        prk
    }

    /// HKDF expand phase: produce `length` bytes from PRK and context info.
    ///
    /// Uses counter-mode expansion: T(i) = PRF(PRK, T(i-1) || info || i).
    pub fn hkdf_expand(prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
        if length == 0 {
            return vec![];
        }

        let mut output = Vec::with_capacity(length);
        let mut t_prev: Vec<u8> = vec![];

        let mut counter: u8 = 1;
        while output.len() < length {
            let mut data = t_prev.clone();
            data.extend_from_slice(info);
            data.push(counter);

            let block = prf_block(prk, &data);
            t_prev = block.to_vec();
            output.extend_from_slice(&block);
            counter = counter.wrapping_add(1);
        }

        output.truncate(length);
        output
    }

    /// Simplified PBKDF2: iteratively applies PRF to password + salt.
    ///
    /// `U1 = PRF(password, salt || 0x00000001)`
    /// `Ui = PRF(password, U(i-1))`
    /// `T  = U1 XOR U2 XOR … XOR Uiterations`
    pub fn pbkdf2_derive(
        password: &[u8],
        salt: &[u8],
        iterations: u32,
        length: usize,
    ) -> Vec<u8> {
        if length == 0 {
            return vec![];
        }

        let blocks_needed = (length + 7) / 8;
        let mut output = Vec::with_capacity(blocks_needed * 8);

        for block_idx in 1u32..=(blocks_needed as u32) {
            // U1 = PRF(password, salt || block_index)
            let mut seed = salt.to_vec();
            seed.extend_from_slice(&block_idx.to_be_bytes());
            let u1 = prf_block(password, &seed);

            let mut acc = u1;
            let mut u_prev = u1.to_vec();

            for _ in 1..iterations {
                let u_next = prf_block(password, &u_prev);
                for (a, b) in acc.iter_mut().zip(u_next.iter()) {
                    *a ^= b;
                }
                u_prev = u_next.to_vec();
            }

            output.extend_from_slice(&acc);
        }

        output.truncate(length);
        output
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
        let slice = if key_bytes.len() > 8 { &key_bytes[..8] } else { key_bytes };
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
        let prk_empty = KeyDerivation::hkdf_extract(&[], ikm());
        let prk_zeros = KeyDerivation::hkdf_extract(&[0u8; 8], ikm());
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
        let dk = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256)
            .expect("derive failed");
        assert_eq!(dk.key_bytes.len(), 32);
        assert_eq!(dk.algorithm, "HKDF-256");
    }

    #[test]
    fn test_derive_hkdf512_longer_output() {
        let p = simple_params(64);
        let dk256 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf256)
            .expect("derive 256 failed");
        let dk512 = KeyDerivation::derive(ikm(), &p, KdfAlgorithm::Hkdf512)
            .expect("derive 512 failed");
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
        let id = KeyDerivation::key_id_from_bytes(&[0xde, 0xad, 0xbe, 0xef, 0x01, 0x23, 0x45, 0x67]);
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
        assert!(dk.key_id.chars().all(|c| c.is_ascii_hexdigit()), "id={}", dk.key_id);
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
}
