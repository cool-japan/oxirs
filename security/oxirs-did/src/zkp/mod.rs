//! Zero-Knowledge Proof module for credential selective disclosure
//!
//! This module provides ZKP-based selective disclosure for verifiable credentials.
//! It allows credential holders to present only selected attributes from a credential
//! without revealing the entire credential or the undisclosed attributes.
//!
//! ## Modules
//!
//! - `selective_disclosure`: Core credential and proof types
//! - `pedersen`: Pedersen commitment scheme with Schnorr proofs-of-knowledge,
//!   `SelectiveDisclosureRequest`, `prove_selective` / `verify_selective`

pub mod pedersen;
pub mod selective_disclosure;

pub use pedersen::{
    prove_selective, verify_selective, AttributeCommitment, PedersenParams,
    PedersenSelectiveDisclosureProof, SchnorrProof, SelectiveDisclosureRequest,
};
pub use selective_disclosure::{
    CredentialAttribute, DisclosurePresentation, SelectiveDisclosureCredential,
    SelectiveDisclosureProof, ZkpProofRequest,
};
