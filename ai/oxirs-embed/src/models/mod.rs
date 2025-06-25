//! Embedding model implementations
//!
//! This module provides various knowledge graph embedding models including:
//! - TransE: Translation-based embeddings
//! - ComplEx: Complex number embeddings for asymmetric relations
//! - DistMult: Bilinear diagonal model
//! - RotatE: Rotation-based embeddings
//! - TuckER: Tucker decomposition based embeddings (optional)

pub mod transe;
pub mod complex;
pub mod distmult;
pub mod rotate;

#[cfg(feature = "tucker")]
pub mod tucker;

#[cfg(feature = "quatd")]
pub mod quatd;

pub mod base;
pub mod common;

// Re-export all models
pub use transe::TransE;
pub use complex::ComplEx;
pub use distmult::DistMult;
pub use rotate::RotatE;

#[cfg(feature = "tucker")]
pub use tucker::TuckER;

#[cfg(feature = "quatd")]
pub use quatd::QuatD;

pub use base::*;
pub use common::*;