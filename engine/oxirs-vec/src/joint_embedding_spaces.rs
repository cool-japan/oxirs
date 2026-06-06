//! Joint Embedding Spaces for Cross-Modal Vector Search — thin facade module.
//!
//! This module implements advanced joint embedding spaces that enable:
//! - CLIP-style text-image alignment
//! - Cross-modal attention mechanisms
//! - Contrastive learning for alignment
//! - Multi-modal fusion strategies
//! - Domain adaptation and transfer learning
//!
//! The implementation lives in sibling modules:
//! - [`crate::joint_embedding_spaces_types`]: configs, structs, enums, type aliases.
//! - [`crate::joint_embedding_spaces_aligner`]: projectors, attention, scheduler,
//!   domain adapter, and the [`JointEmbeddingSpace`] core.
//! - [`crate::joint_embedding_spaces_transfer`]: CLIP aligner, augmentation,
//!   curriculum learning, and zero-shot transfer helpers.

pub use crate::joint_embedding_spaces_aligner::*;
pub use crate::joint_embedding_spaces_transfer::*;
pub use crate::joint_embedding_spaces_types::*;
