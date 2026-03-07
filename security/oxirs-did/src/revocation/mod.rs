//! Revocation support for Verifiable Credentials
//!
//! This module implements two revocation specifications:
//!
//! - **StatusList2021**: W3C StatusList 2021 (GZIP-compressed bitstring)
//! - **RevocationList2020**: W3C Revocation List 2020 (raw bitset + bloom filter)
//!
//! # Overview
//!
//! - `StatusList2021`: Compressed bitstring-based revocation list
//! - `CredentialStatus`: Status entry embedded in credentials
//! - `RevocationRegistry`: High-level API for StatusList2021
//! - `StatusListCredential`: VC wrapper for StatusList2021
//! - `RevocationList2020`: Raw bitset with W3C RL2020 serialization
//! - `RevocationRegistry2020`: Registry with bloom filter for fast O(1) checks
//! - `RevocationStatus`: Valid / Revoked(reason) / Unknown
//! - `BloomFilter`: Probabilistic non-membership proof structure

pub mod revocation_list;
pub mod status_list;
pub mod status_list_extended;

pub use revocation_list::{
    BloomFilter, RevocationEntry, RevocationList2020, RevocationRegistry2020, RevocationStatus,
};
pub use status_list::{
    CredentialStatus, RevocationRegistry, StatusList2021, StatusPurpose, MIN_LIST_SIZE,
};
pub use status_list_extended::{StatusList2021Inner, StatusListCredential};
