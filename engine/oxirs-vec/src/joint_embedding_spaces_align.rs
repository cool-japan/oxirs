//! Joint Embedding Spaces — alignment algorithms.
//!
//! Re-exports alignment implementation from joint_embedding_spaces_aligner.
//! Contains: Procrustes alignment, CCA, linear map learning, manifold alignment,
//! cross-modal attention, temperature scheduling, domain adaptation.

pub use crate::joint_embedding_spaces_aligner::JointEmbeddingSpace;
pub use crate::joint_embedding_spaces_types::{
    CrossModalAttention, DomainAdapter, LinearProjector, TemperatureScheduler,
};
