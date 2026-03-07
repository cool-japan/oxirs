//! Advanced signature suite implementations
//!
//! This module provides:
//! - ES256 (ECDSA with P-256 and SHA-256) signatures for JWS
//! - RS256 (RSASSA-PKCS1-v1_5 with SHA-256) signatures for JWS
//! - BBS+ group signatures for selective disclosure
//! - Ed25519 JWS signer/verifier (`Ed25519JwsSigner`, `Ed25519JwsVerifier`)
//! - ECDSA P-256 JWS signer/verifier (`EcdsaJwsSigner`, `EcdsaJwsVerifier`)
//! - Structured `JwsSignature` type with compact serialization
//! - Typed `JwsAlgorithm` enum and `JwsHeader` with `b64` support
//! - `MockJwsSigner` for deterministic test signing

pub mod bbs_plus;
pub mod ed25519_jws;
pub mod es256;
pub mod jws_algorithm;
pub mod rs256;

pub use bbs_plus::{BbsKeyPair, BbsPlusSignature, BbsProof, BbsProofRequest};
pub use ed25519_jws::{
    EcdsaJwsSigner, EcdsaJwsVerifier, Ed25519JwsSigner, Ed25519JwsVerifier, JwsPayload,
    JwsSignature, JwsSignatureHeader, JwsSignerTrait, JwsVerifierTrait,
};
pub use es256::{Es256Signer, Es256Verifier, P256KeyPair};
pub use jws_algorithm::{
    JwsAlgorithm, JwsHeader, JwsSigner, JwsVerifier, MockJwsSigner, MockJwsVerifier,
};
pub use rs256::{Rs256Signer, Rs256Verifier, RsaKeyPair};
