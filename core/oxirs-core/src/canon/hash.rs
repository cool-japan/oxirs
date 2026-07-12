//! SHA-256 hashing utilities for URDNA2015 canonicalization.
//!
//! Provides the hash primitive required by the W3C RDF Dataset Normalization
//! Algorithm (URDNA2015 / RDNA 2015). SHA-256 is used as specified in the W3C
//! RDF Canonicalization specification (<https://www.w3.org/TR/rdf-canon/>).

use sha2::{Digest, Sha256};

/// Compute the SHA-256 hash of a UTF-8 string and return the lowercase hex digest.
///
/// This is the core hash primitive used throughout URDNA2015: every intermediate
/// hash (first-degree, N-degree) is produced by this function.
#[inline]
pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Compute the SHA-256 hash of raw bytes and return the lowercase hex digest.
#[inline]
pub fn sha256_hex_bytes(input: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let result = hasher.finalize();
    hex::encode(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_known_value() {
        // FIPS 180-4 test vector: SHA-256("abc")
        let h = sha256_hex("abc");
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn test_sha256_empty_string() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let h = sha256_hex("");
        assert_eq!(
            h,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_is_deterministic() {
        let a = sha256_hex("hello world");
        let b = sha256_hex("hello world");
        assert_eq!(a, b);
    }
}
